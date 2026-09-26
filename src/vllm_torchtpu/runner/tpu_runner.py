# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ensure environment overrides are applied before any other imports,
# especially torch_tpu which might read them at import time.
import vllm_torchtpu.env_override  # noqa: F401  # isort: skip

import bisect
import contextlib
import copy
import dataclasses
import math
import os
import time
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

# TODO: Remove this after jax dependency is removed
import jax
import numpy as np
import torch
import torch_tpu  # noqa: F401
import vllm.envs as vllm_envs

# TODO: Remove this after jax dependency is removed
from jax.sharding import Mesh
from packaging import version
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    RotaryEmbedding,
)
from vllm.model_executor.layers.rotary_embedding.mrope_interleaved import (
    MRotaryEmbeddingInterleaved,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    DraftTokenIds,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput

from vllm_torchtpu import envs, utils
from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.distributed.pp_wave import PPWave, pp_rank_flags
from vllm_torchtpu.kernels.experimental.batched_rpa import configs as rpa_configs
from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper as rpa_batched
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.attention import (
    KV_LAYOUT_BY_VLLM_LAYOUT,
    TPU_STR_DTYPE_TO_TORCH_DTYPE,
    PallasAttentionBackend,
    PallasAttentionBackendImpl,
)
from vllm_torchtpu.layers.adapter.custom_ops.mamba_state_copy_op import (
    copy_mamba_state_blocks,
)
from vllm_torchtpu.layers.adapter.quantization import get_tpu_quantization_config
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import validate_online_fp8
from vllm_torchtpu.layers.adapter.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilderContext,
    stage_block_table_uploads,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    SequenceLayoutKind,
    create_sequence_layout_planner,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
from vllm_torchtpu.platforms.tpu_block_size_utils import unified_kv_layout_enabled
from vllm_torchtpu.runner import utils as runner_utils
from vllm_torchtpu.runner.kv_cache_manager import KVCacheManager
from vllm_torchtpu.runner.mm_encoder_manager import maybe_create_mm_encoder_manager
from vllm_torchtpu.runner.mm_video_chunking import chunked_video_encoding
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpecDecodeMetadata,
    SpeculativeDecodingManager,
)
from vllm_torchtpu.runner.structured_decoding_manager import StructuredDecodingManager
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncPreResults,
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.dspark import DSparkProposer
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer
from vllm_torchtpu.spec_decode.utils import DraftChunkInputs, normalize_draft_config
from vllm_torchtpu.tracing.annotation import TraceAnnotation
from vllm_torchtpu.tracing.options import resolve_profile_dir_and_opts
from vllm_torchtpu.tracing.utils import (
    extract_kv_lens_for_tracing,
    extract_request_ids_for_tracing,
)
from vllm_torchtpu.utils import synchronize_device, synchronize_tensors

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput


@dataclass
class ExecuteModelState:
    scheduler_output: "SchedulerOutput"
    logits_list: list[torch.Tensor]
    pooler_output_list: list[Any]
    num_reqs_list: list[int]
    spec_decode_metadata_list: list["SpecDecodeMetadata | None"]
    # Per-chunk draft inputs (token ids, positions, attn ctx, aux
    # hidden states) captured from the target verify pass. None if
    # spec decode proposer is disabled.
    draft_chunks: list[DraftChunkInputs] = field(default_factory=list)
    # Per chunk, the device mamba state slot ids per mamba group (base slot
    # per batch position); only populated for hybrid models with spec
    # decoding, so sample_tokens can scatter this step's read offsets per
    # physical slot. The compact pool shares one slot tensor across groups
    # and so contributes a single-element list.
    mamba_state_indices_list: list[list[torch.Tensor] | None] = field(
        default_factory=list
    )


# Per-step mamba read-offset operations, consolidated into single cached
# TPU programs (avoiding per-group eager launches). Each body is defined
# once; the compiled variant is derived from it, and the dispatchers pick
# eager on CPU (unit tests, pre-init paths) since the tpu backend cannot
# trace CPU tensors.
def _reset_read_offsets_impl(
    read_offsets: torch.Tensor,
    keep_dev: torch.Tensor,
    stacked_groups: torch.Tensor,
    null: torch.Tensor,
) -> None:
    """Scatter reset across all mamba groups in a single program.
    `stacked_groups` is [G, width]; `keep_dev` [width] broadcasts across G.
    Non-new rows collapse onto the null slot, where writing 0 is a no-op."""
    targets = torch.where(keep_dev, stacked_groups, null)
    zero = torch.zeros((), dtype=read_offsets.dtype, device=read_offsets.device)
    read_offsets.index_put_((targets.reshape(-1).long(),), zero)


def _rollback_offsets_seed_impl(
    read_offsets: torch.Tensor, dst_t: torch.Tensor
) -> None:
    read_offsets.index_put_(
        (dst_t,),
        torch.zeros(
            dst_t.shape[0], dtype=read_offsets.dtype, device=read_offsets.device
        ),
    )


def _rollback_offsets_migrate_impl(
    read_offsets: torch.Tensor, src_t: torch.Tensor, dst_t: torch.Tensor
) -> None:
    read_offsets.index_put_((dst_t,), read_offsets[src_t])


_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)
_reset_read_offsets_compiled = _tpu_compile(_reset_read_offsets_impl)
_rollback_offsets_seed_compiled = _tpu_compile(_rollback_offsets_seed_impl)
_rollback_offsets_migrate_compiled = _tpu_compile(_rollback_offsets_migrate_impl)


def _reset_read_offsets(
    read_offsets: torch.Tensor,
    keep_dev: torch.Tensor,
    stacked_groups: torch.Tensor,
    null: torch.Tensor,
) -> None:
    fn = (
        _reset_read_offsets_compiled
        if read_offsets.device.type == "tpu"
        else _reset_read_offsets_impl
    )
    fn(read_offsets, keep_dev, stacked_groups, null)


def _rollback_offsets_seed(read_offsets: torch.Tensor, dst_t: torch.Tensor) -> None:
    fn = (
        _rollback_offsets_seed_compiled
        if read_offsets.device.type == "tpu"
        else _rollback_offsets_seed_impl
    )
    fn(read_offsets, dst_t)


def _rollback_offsets_migrate(
    read_offsets: torch.Tensor, src_t: torch.Tensor, dst_t: torch.Tensor
) -> None:
    fn = (
        _rollback_offsets_migrate_compiled
        if read_offsets.device.type == "tpu"
        else _rollback_offsets_migrate_impl
    )
    fn(read_offsets, src_t, dst_t)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _substitute_placeholder_token(
    input_ids: torch.Tensor,
    token_in_tpu_cur_input_indices: torch.Tensor,
    token_in_tpu_pre_next_tokens_indices: torch.Tensor,
    next_tokens: torch.Tensor,
):
    """Substitute placeholder tokens from TPU for async scheduler.

    Padding scheme (set up by `_apply_async_token_substitution`):
      - `token_in_tpu_cur_input_indices`: real-slot indices followed by all
        other indices in ascending order; the tail entries re-target slots
        that should keep their current values.
      - `token_in_tpu_pre_next_tokens_indices`: real `next_tokens` indices
        followed by `-1` sentinels at the padded positions.
    The `-1` sentinel is the mask source; the program is shape-only so it
    does not recompile when the active count changes.

    Args:
        input_ids: device tensor [N] where N is the bucketed input size.
        token_in_tpu_cur_input_indices: int tensor [N], destination slots.
        token_in_tpu_pre_next_tokens_indices: int tensor [N], source slots
            in `next_tokens`; -1 marks a padding slot to leave unchanged.
        next_tokens: int tensor of tokens from the previous async step.
    Return:
        input_ids with real placeholders replaced; padding slots untouched.
    """
    assert (
        input_ids.shape[0]
        == token_in_tpu_cur_input_indices.shape[0]
        == token_in_tpu_pre_next_tokens_indices.shape[0]
    )
    mask = token_in_tpu_pre_next_tokens_indices > -1
    # clamp_min(0) gives a safe in-range gather index for the -1 sentinel
    # slots; their gathered values are discarded by `mask` in `where`.
    safe_idx = torch.clamp_min(token_in_tpu_pre_next_tokens_indices, 0)
    new_token_values = next_tokens[safe_idx].to(input_ids.dtype)
    original_values = input_ids[token_in_tpu_cur_input_indices]
    update_values = torch.where(mask, new_token_values, original_values)
    input_ids.scatter_(0, token_in_tpu_cur_input_indices, update_values)
    return input_ids


logger = init_logger(__name__)


def _spec_warmup_all_token_ids(
    req_ids: list[str], num_computed_tokens: list[int]
) -> dict[str, list[int]]:
    """Build complete synthetic token histories for async cached requests."""
    assert len(req_ids) == len(num_computed_tokens), (
        "spec-decode warmup request IDs and computed-token counts must align: "
        f"requests={len(req_ids)}, counts={len(num_computed_tokens)}"
    )
    # The current sampled token is not computed yet, so the complete sequence
    # contains num_computed + 1 tokens at scheduler handoff.
    return {
        req_id: [0] * (num_computed + 1)
        for req_id, num_computed in zip(req_ids, num_computed_tokens)
    }


@contextmanager
def _suspend_kv_transfer_group() -> Iterator[None]:
    """Hide the process-global connector during worker-local warmup."""
    connector = kv_transfer_state._KV_CONNECTOR_AGENT
    kv_transfer_state._KV_CONNECTOR_AGENT = None
    try:
        yield
    finally:
        kv_transfer_state._KV_CONNECTOR_AGENT = connector


_KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP = (
    "invalid_block_group_index"
    in getattr(KVConnectorOutput, "__dataclass_fields__", {})
)


def _build_kv_connector_output(
    *,
    finished_sending: set[str] | None,
    finished_recving: set[str] | None,
    kv_connector_worker_meta: Any | None,
    invalid_block_ids: set[int],
    invalid_block_group_index: int | None,
    kv_connector_stats: Any | None = None,
) -> KVConnectorOutput:
    """Build a connector output across the vLLM 0.23 API boundary.

    vLLM 0.23 supports invalid block IDs only for a single KV cache group and
    does not expose ``invalid_block_group_index``. Newer vLLM versions support
    group-scoped recovery for hybrid caches. Do not pass the newer keyword to
    0.23, but fail closed if a hybrid load actually fails: silently dropping
    the group would make overlapping block IDs ambiguous to its scheduler.
    """
    kwargs: dict[str, Any] = {
        "finished_sending": finished_sending,
        "finished_recving": finished_recving,
        "kv_connector_worker_meta": kv_connector_worker_meta,
        "invalid_block_ids": invalid_block_ids,
        "kv_connector_stats": kv_connector_stats,
    }
    if _KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP:
        kwargs["invalid_block_group_index"] = invalid_block_group_index
    return KVConnectorOutput(**kwargs)


# Smallest output size
MIN_NUM_SEQS = 8
SAMPLING_EPS = 1e-5


def _get_sequence_layout_planner_for_runner(runner: Any):
    return runner.sequence_layout_planner


def _validate_libtpu_version() -> None:
    """Validate libtpu opportunistically if it is present."""
    libtpu_version = importlib_metadata.version("libtpu")
    parsed_version = version.parse(libtpu_version)
    if parsed_version < version.parse("0.0.35"):
        raise RuntimeError("Argmax is having accuracy issue with libtpu < 0.0.35")
    if parsed_version < version.parse("0.0.36"):
        logger.warning_once(
            "libtpu < 0.0.36 may enable "
            "--xla_tpu_impure_enable_large_2nd_minor_layout by default, "
            "which can hurt performance. "
            "Upgrade libtpu to >= 0.0.36, or set "
            "LIBTPU_INIT_ARGS=--xla_tpu_impure_enable_large_2nd_minor_layout=false."
        )


@contextlib.contextmanager
def _torch_tpu_wrapper():
    """Alias torch.cuda.* to torch.tpu.* during inherited GPUModelRunner init.

    Mirrors vllm/v1/worker/xpu_model_runner.py:_torch_cuda_wrapper. The parent
    GPUModelRunner constructs torch.cuda.Stream/Event for comm and async copy;
    this wrapper makes those resolve to the TPU equivalents (verified present
    on torch.tpu: Stream, Event, current_stream, default_stream, stream,
    set_stream, synchronize, set_device, current_device, device_count).

    The 5 APIs not on torch.tpu are stubbed:
      - mem_get_info -> torch.accelerator.get_memory_info (works on TPU)
      - empty_cache  -> no-op (XLA manages HBM)
      - graph / CUDAGraph / graph_pool_handle -> never reached because we
        disable cudagraphs in __init__ via compilation_config overrides.
    """
    saved = {}
    aliased = (
        "Stream",
        "Event",
        "current_stream",
        "default_stream",
        "stream",
        "set_stream",
        "synchronize",
        "set_device",
        "current_device",
        "device_count",
        "is_available",
    )
    for name in aliased:
        saved[name] = getattr(torch.cuda, name, None)
        setattr(torch.cuda, name, getattr(torch.tpu, name))
    saved["mem_get_info"] = getattr(torch.cuda, "mem_get_info", None)
    torch.cuda.mem_get_info = lambda *a, **kw: torch.accelerator.get_memory_info()
    saved["empty_cache"] = getattr(torch.cuda, "empty_cache", None)
    torch.cuda.empty_cache = lambda: None
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                if hasattr(torch.cuda, k):
                    delattr(torch.cuda, k)
            else:
                setattr(torch.cuda, k, v)


_TOPK_INDICES = "topk_indices"
_TOPK_BUFFER_ATTRIBUTE = "topk_indices_buffer"

# Hand-off tensors that carry indices.
# -1 is "no token", so such a row attends to nothing.
_PP_CARRIED_FILL: dict[str, int] = {_TOPK_INDICES: -1}


def _find_topk_indices_buffer(model: torch.nn.Module) -> torch.Tensor | None:
    """The one top-k table this process's layers share, or None."""
    found: dict[int, torch.Tensor] = {}
    for module in model.modules():
        buffer = getattr(module, _TOPK_BUFFER_ATTRIBUTE, None)
        if isinstance(buffer, torch.Tensor):
            found.setdefault(id(buffer), buffer)
    if not found:
        return None
    if len(found) > 1:
        shapes = sorted(tuple(b.shape) for b in found.values())
        raise RuntimeError(
            "the model's layers hold several distinct top-k index tables "
            f"{shapes}; the pipeline hand-off carries one per stage"
        )
    return next(iter(found.values()))


# Recompilation-avoidance contract:
#   1. Input prep happens on CPU; H2D via `cpu_tensor.to(xla_device)`.
#   2. Forward is split into 4 `@torch.compile(backend="tpu")` subgraphs
#      (backbone, compute_selected_logits, sample_from_logits/structured_decode,
#      gather_logprobs) so _dummy_run and execute_model trace identically.
#   3. `_dummy_run` exercises every padding bucket so all shapes are AOT-
#      compiled before the first real request.
class TPUModelRunner(GPUModelRunner):
    @property
    def _is_async_drafter(self) -> bool:
        if not self.speculative_config:
            return False
        return (
            self.speculative_config.use_eagle()
            or self.speculative_config.method == "dflash"
        )

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        *,
        profiler_rank: int,
        profiler_world_size: int,
    ):
        # Slice-global rank/world used to label phased-profiler traces; the
        # worker resolves them from the TPU rank binding. Required rather than
        # defaulted, because the obvious default -- parallel_config -- is the
        # wrong scope and fails silently. See `_init_phased_profiling`.
        self._profiler_rank = profiler_rank
        self._profiler_world_size = profiler_world_size
        pcp_config = PcpStaticSupportValidator.from_vllm_config(vllm_config)
        self._pcp_enabled = pcp_config.enabled
        self._mtp_enabled = pcp_config.speculative_method == "mtp"
        sequence_layout_planner = create_sequence_layout_planner(vllm_config)
        if sequence_layout_planner.requires_backend_preinit:
            # GPUModelRunner probes torch.cuda.mem_get_info during init. The
            # TPU shim maps that to torch.accelerator.get_memory_info, which
            # initializes TorchTPU/PJRT from TORCH_TPU_TOPOLOGY=1,1,1 and would
            # otherwise leave JAX seeing only one chip. Initialize JAX first
            # so partial sequence layout backends own the full local device
            # set.
            layout_devices = list(jax.local_devices())
            logger.info(
                "Pre-initialized JAX backend for partial sequence layout "
                "| world_size=%d | visible_devices=%d",
                sequence_layout_planner.backend_preinit_world_size,
                len(layout_devices),
            )
        # Disable cudagraphs before parent init so its dispatch self-disables.
        # TPU uses AOT bucket precompile (_precompile_* methods) instead.
        vllm_config.compilation_config.cudagraph_capture_sizes = []
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        # DSpark/DFlash checkpoints may spell the mask token dspark_noise_token_id;
        # the parent __init__ transiently builds an upstream EagleProposer
        # that raises unless dflash_config.mask_token_id resolves.
        if (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.method in ("dflash", "dspark")
        ):
            normalize_draft_config(
                vllm_config.speculative_config.draft_model_config.hf_config
            )
        with _torch_tpu_wrapper():
            super().__init__(vllm_config, device)
        self.sequence_layout_planner = sequence_layout_planner
        _validate_libtpu_version()
        self.is_pooling_model = (
            self.model_config is not None and self.model_config.runner_type == "pooling"
        )

        if self.is_pooling_model:
            from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
            from vllm.v1.worker.tpu_input_batch import InputBatch

            def get_pooling_metadata(batch):
                reqs = batch.requests[: batch.num_reqs]
                pooling_params = [r.pooling_params for r in reqs]
                pooling_states = [
                    PoolingStates(r.req_id, r.num_computed_tokens) for r in reqs
                ]
                prompt_lens = torch.tensor(
                    [r.prompt_token_ids_len for r in reqs], dtype=torch.int32
                )
                return PoolingMetadata(
                    prompt_lens=prompt_lens,
                    prompt_token_ids=None,
                    prompt_token_ids_cpu=None,
                    pooling_params=pooling_params,
                    pooling_states=pooling_states,
                )

            InputBatch.get_pooling_metadata = get_pooling_metadata

        # Parent already set: vllm_config, *_config, device, dtype,
        # max_model_len, max_num_reqs, max_num_tokens, num_query_heads,
        # inputs_embeds_size, mm_registry, uses_mrope, supports_mm_inputs,
        # kv_caches, encoder_cache, shared_kv_cache_layers, requests,
        # num_prompt_logprobs, input_batch, etc. The block below is only the
        # TPU-specific delta + overrides.

        self.original_parallel_config = vllm_config.parallel_config
        self.device_config = vllm_config.device_config

        # Set by `_update_*_page_size_padded` for hybrid attention
        # models so vLLM sees a uniform page size across groups.
        self._hybrid_uniform_page_size_bytes: int | None = None

        # Compact-mamba sizing.
        # When set, mamba/GDN layers allocate only `_mamba_num_blocks`
        # recurrent slots (= max_num_reqs + 1, slot 0 = null block) instead of
        # the full attention `num_blocks`, and the per-request slot id is
        # carried in `AttentionMetadata.mamba_state_indices` instead of being
        # derived from the attention `block_tables[:, 0]`. None until the
        # override runs (and stays None if it is skipped, e.g. CPU-only tests
        # or a user-pinned num_gpu_blocks_override) — in which case mamba and
        # attention share the uniform `num_blocks` as before.
        self._mamba_num_blocks: int | None = None
        # req_id -> physical mamba slot id, plus the free-slot pool. Slots are
        # allocated when a request first appears and returned when it leaves
        # the persistent batch. Keying on req_id (rather than persistent-batch
        # position) makes the mapping immune to condense/swap reordering in
        # upstream vLLM's InputBatch, which we do not subclass.
        self._mamba_slot_by_req_id: dict[str, int] = {}
        self._free_mamba_slots: list[int] = []
        # State checkpoints per request under speculative decoding: one per
        # verify window position, so rejected drafts roll back by *selecting*
        # a checkpoint rather than by copying state. 1 without spec decoding.
        # The two layouts hold the group differently, hence two names:
        #   * compact pool -- `num_spec + 1` consecutive slots in the mamba
        #     tensor, addressed as base + offset (`_init_mamba_slot_pool`);
        #   * unified pool -- `num_spec + 1` ordinary pool blocks named
        #     individually by `mamba_ckpt_indices`, following vLLM's
        #     `MambaSpec.num_speculative_blocks`. Naming them keeps a block
        #     sized for a single state; holding the whole group in one block
        #     instead would scale `block_size` with `num_spec` (Qwen3.5-397B
        #     at TP=1, K=4: 21760 tokens rather than 4352).
        num_ckpts = 1
        if self.vllm_config.speculative_config is not None:
            num_ckpts = self.vllm_config.speculative_config.num_speculative_tokens + 1
        self._mamba_slot_stride: int = num_ckpts
        self._mamba_ckpt_window: int = (
            num_ckpts if unified_kv_layout_enabled(self.vllm_config) else 1
        )
        # True once the slot pool is initialized (hybrid model with mamba
        # layers); gates per-step mamba_state_indices construction.
        self._has_mamba_state: bool = False
        # Slot-indexed mamba read offsets for speculative decoding with
        # hybrid (attention + mamba) models; allocated in
        # `initialize_kv_cache` once the mamba pool size is known. See
        # `AttentionMetadata.mamba_slot_read_offsets`.
        self.mamba_slot_read_offsets: torch.Tensor | None = None
        # req_ids whose read offset has been initialized since they entered
        # the batch; see `_reset_read_offsets_for_new_requests`.
        self._mamba_offset_seeded: set[str] = set()
        # Per-width reusable buffers for that per-step reset; see
        # `_read_offset_reset_scratch`.
        self._read_offset_scratch: dict[
            tuple[int, str],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        # DecodeBench fills local state instead of transferring manager blocks,
        # so it can retain the compact request-indexed Mamba layout.
        transfer_config = self.vllm_config.kv_transfer_config
        self._uniform_mamba_layout: bool = (
            transfer_config is not None
            and transfer_config.kv_connector != "DecodeBenchConnector"
        )
        # Unified layout: attention KV and mamba state are fungible
        # block-table blocks in one attention-shaped pool per cache tensor.
        self._unified_kv_layout: bool = unified_kv_layout_enabled(self.vllm_config)
        self.kv_cache_raw_tensors: list[torch.Tensor] = []
        # Set by initialize_kv_cache; None until then (dummy runs check this).
        self.kv_cache_config: KVCacheConfig | None = None
        # Layout plan of the most recent prepared batch; None before the
        # first execute_model.
        self._last_sequence_layout_plan = None

        # Block-major KV cache (opt-in via VLLM_TPU_BLOCK_MAJOR_KV). When
        # enabled, `_initialize_kv_cache_states` allocates a single bundled tensor
        # `[num_blocks, num_layers, page_size, K2/p, p, head_dim]` and
        # `_kv_cache_bundle_layer_index` maps each attention layer name to its index
        # along dim 1 of the bundle. The attention forward path routes through
        # the bundled RPA kernel when these are set.
        self._kv_cache_bundle: torch.Tensor | None = None
        self._kv_cache_bundle_layer_index: dict[str, int] = {}

        # EP-DP lockstep state, refreshed each step by execute_model /
        # execute_dummy_batch: the coordinated (max-across-ranks) token
        # bucket and chunk count.
        self._dp_target_bucket: int | None = None
        self._dp_step_num_chunks: int = 0
        self._dp_step_max_reqs: int = 0
        self.check_recompilation = vllm_envs.VLLM_XLA_CHECK_RECOMPILATION
        self.use_spmd = vllm_envs.VLLM_XLA_USE_SPMD

        # XLA graph tracker (TPU-specific debug aid).
        self.enforce_eager = self.model_config.enforce_eager
        self.num_xla_graphs = 0
        self._update_num_xla_graphs("init")
        # torchTPU doesn't support SymInt yet, so widen Dynamo cache limits.
        torch._dynamo.config.cache_size_limit = 1024
        torch._dynamo.config.accumulated_recompile_limit = 1024

        # Override parent's kv_cache_dtype with the TPU mapping (handles
        # auto -> bfloat16 for our supported dtype set).
        cache_config = self.cache_config
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = (
                TPU_STR_DTYPE_TO_TORCH_DTYPE[self.dtype]
                if isinstance(self.dtype, str)
                else self.dtype
            )
        else:
            self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        self._hidden_states_dtype = self.dtype

        # Compile bucket + padding (TPU AOT compile rounds inputs up).
        self.sliding_window = self.model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.most_model_len = envs.VLLM_TPU_MOST_MODEL_LEN
        self._fast_token_substitution: bool = bool(
            envs.VLLM_TPU_FAST_TOKEN_SUBSTITUTION
        )
        # Sync max_num_blocks_per_req with the underlying GPUInputBatch's
        # block_table capacity if already created during super().__init__().
        # This ensures that calculations derived from max_num_blocks_per_req
        # (such as spec-decode warmup block count `nblk`) strictly respect the
        # actual BlockTable allocation limits, preventing bounds mismatch errors
        # (e.g., broadcasting shape (8,) into capacity (5,)).
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        if (
            hasattr(self, "input_batch")
            and self.input_batch is not None
            and hasattr(self.input_batch, "block_table")
        ):
            try:  # noqa: SIM105 (keep the default if the block table is not ready)
                self.max_num_blocks_per_req = int(
                    self.input_batch.block_table[0].get_cpu_tensor().shape[1]
                )
            except (IndexError, TypeError, AttributeError, KeyError):
                pass
        self.num_blocks_per_most_len_req = (
            cdiv(self.most_model_len, self.block_size)
            if self.most_model_len is not None
            else None
        )
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        self.max_num_reqs = max(self.max_num_reqs, MIN_NUM_SEQS)
        self.num_tokens_paddings = vllm_config.compilation_config.compile_sizes
        # Override parent's max_num_tokens with the last (largest) padding
        # bucket so all CPU staging buffers cover it.
        self.max_num_tokens = self.num_tokens_paddings[-1]
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            self.parallel_config, "attention"
        )
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.vocab_size = self.model_config.get_vocab_size()

        # Pallas attention bookkeeping.
        self._attention_kernels_initialized = False

        # CPU staging tensors (TPU prepares inputs on CPU then transfers).
        self.input_ids_cpu = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device="cpu"
        )
        self.positions_cpu = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device="cpu"
        )
        self.positions_np = self.positions_cpu.numpy()
        self.block_table_cpu = torch.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu",
        )
        # CPU staging for compact-mamba per-request recurrent-slot ids; H2D
        # copied each step into `AttentionMetadata.mamba_state_indices`.
        self.mamba_state_indices_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device="cpu"
        )
        # Block-table-derived mamba state: the state-block position each
        # request's state currently lives at, the mamba-group -> raw-pool
        # mapping (built in initialize_kv_cache), and the block-seed copies
        # staged by _prepare_inputs and flushed before the forward.
        self._mamba_state_pos: dict[str, int] = {}
        self._mamba_copy_plan: list[tuple[int, list[torch.Tensor]]] = []
        # Manager-block -> pool-block split: >1 when the pool is born at a
        # smaller attention-kernel granularity (batched RPA).
        self._pool_block_split: int = 1
        # Each entry seeds one set of raw pool buffers with the same
        # (src, dst) pairs in one program: (raws, src_ids, dst_ids).
        self._pending_mamba_state_copies: list[
            tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]
        ] = []
        if self.uses_mrope:
            # Override parent's int64 mrope buffer with int32 so the H2D copy is
            # dtype-identical to what the TPU model expects. Parent's
            # _calc_mrope_positions writes via .cpu and .np — both route here.
            self.mrope_positions = self._make_buffer(
                3, self.max_num_tokens + 1, dtype=torch.int32
            )
        self.query_start_loc_cpu = torch.zeros(
            self.max_num_tokens + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
        )
        self.seq_lens_np = self.seq_lens_cpu.numpy()
        self._block_table_stage_cpu: torch.Tensor | None = None
        if self.supports_mm_inputs:
            self.is_mm_embed_cpu = torch.zeros(
                self.max_num_tokens,
                dtype=torch.bool,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.num_reqs_paddings = _get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs
        )

        # Pallas SMEM-aware num_reqs caps.
        self.num_reqs_most_model_len = (
            min(
                PallasAttentionBackend.get_max_num_seqs(
                    self.most_model_len, self.block_size
                ),
                self.max_num_reqs,
                self.max_num_tokens,
            )
            if self.most_model_len is not None
            else None
        )
        self.num_reqs_max_model_len = min(
            PallasAttentionBackend.get_max_num_seqs(
                self.max_model_len, self.block_size
            ),
            self.max_num_reqs,
            self.max_num_tokens,
        )

        self.sample_from_logits_func = self.sample_from_logits

        # TPU async-scheduling state (passed between execute_model and
        # sample_tokens, mirroring vLLM's async path).
        self.mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
        self.execute_model_state: ExecuteModelState | None = None
        self._pre_async_results: AsyncPreResults | None = None
        # 1 on TP rank 0, else 0; built lazily on first use by
        # _sync_replicated_drafts_across_tp (needs the TP group initialized).
        self._tp_rank0_mask: torch.Tensor | None = None

        # Caches for decode-fast-path: avoid re-creating identical device
        # tensors across consecutive decode steps.
        self._attn_layer_names: list[str] | None = None
        self._request_distribution_cpu = torch.zeros(3, dtype=torch.int32)
        # GDN windowed distribution staging for spec decode with mamba
        # layers; see AttentionMetadata.mamba_request_distribution. Staged
        # together with the RPA distribution in one [6] tensor whose device
        # copy is sliced into the two metadata fields ([0:3] RPA, [3:6]
        # GDN). The fields must never be separate equal-valued device
        # tensors: torch-tpu's lazy tracing CSEs those into one node on
        # steps where the values coincide (pure decode), dynamo then folds
        # the RPA and GDN op inputs into a single graph input, and later
        # verify steps silently feed the GDN windowed distribution to
        # ragged paged attention (in-window query rows never attend).
        self._combined_request_distribution_cpu = torch.zeros(6, dtype=torch.int32)
        self._decode_device_cache_key: tuple | None = None
        # Snapshot of the in-flight uploads from the host staging tensors
        # (input_ids_cpu, positions_cpu, seq_lens_cpu, ...); the next chunk
        # waits on it before rewriting them.
        self._input_staging_fence: Any = None
        self._cached_query_start_loc: torch.Tensor | None = None
        self._cached_logits_indices: torch.Tensor | None = None
        self._cached_request_distribution: torch.Tensor | None = None

        self.speculative_config = self.vllm_config.speculative_config
        # Maximum query length to be considered a decode.
        self.reorder_batch_threshold = 1
        self.spec_decode_manager = SpeculativeDecodingManager(self)
        self._init_speculative_decoding()
        self.structured_decoding_manager = StructuredDecodingManager(self)
        self.kv_cache_manager = KVCacheManager(self)

        # Dedicated RNG for non-greedy sampling, to keep the draws consistent
        # across a replica's TP ranks (else they diverge -> collective hang).
        self._sampling_generator: torch.Generator | None = None
        # JAX Mesh for shard_map ops in TPU kernels. Real DP is vLLM
        # multi-engine DP, so this per-worker mesh only has a model axis.
        self.mesh = self._create_mesh_for_parallelism()
        self.batch_counter = 0

        # Pipeline parallelism: this worker runs one stage's layer slice.
        # Stages before the last hand their hidden states to the next stage
        # instead of sampling; stages after the first take them as input.
        self._pp_is_first, self._pp_is_last = pp_rank_flags(self.parallel_config)
        # Hand-off wave over ICI; built in capture_model, where every stage
        # reaches the same point.
        self._pp_wave: PPWave | None = None
        # A sparse model's shared top-k index table, when this stage's layers
        # cannot all recompute it; set at load time under a pipeline.
        self._pp_topk_buffer: torch.Tensor | None = None
        # Token granule the pipeline chunk scheduler cuts at; set with the
        # extra buckets once the KV block size is final.
        self._pp_chunk_granularity: int | None = None
        # (pairs, bq, bkv) per kernel mode for the batched attention
        # kernel, per (sequences, pages per sequence) it is called with;
        # computed once the KV block size is final.
        self._attention_capacity: dict[
            tuple[int, int], dict[str, tuple[int, int, int]]
        ] = {}
        # Whether an attention layer runs the batched kernel; None until
        # asked.
        self._attention_runs_batched_kernel: bool | None = None
        # The page size the attention kernel sees, set with the KV cache.
        self._attention_kernel_block_size: int | None = None
        # The KV cache group of the full-attention layers, set with the KV
        # cache: its block table width is the page count per sequence the
        # kernel is called with, which sizes the kernel's schedule table.
        self._attention_kv_cache_group_id: int | None = None
        # Random host inputs for profile runs, one buffer per shape.
        self._profile_inputs: dict[str, torch.Tensor] = {}
        # Per-key (trailing shape, dtype) of the tensors a stage receives.
        self._pp_intermediate_template: (
            dict[str, tuple[tuple[int, ...], torch.dtype]] | None
        ) = None
        # The step whose KV-connector progress an intermediate stage still
        # owes in sample_tokens.
        self._pp_pending_scheduler_output: SchedulerOutput | None = None

        # Tracks token padding in each step.
        self._token_padding_state: token_padding.TokenPaddingState | None = None

        self._init_phased_profiling()

    def _init_phased_profiling(self) -> None:
        """Resolves whether USE_PHASED_PROFILER is set and where its traces go.

        The trace directory comes from `profiler_config`, the same field the
        standard torch profiler uses; the env var only selects which profiler
        consumes it. The profiler itself is armed later by
        `start_phased_profiling`/`stop_phased_profiling`, called from
        TPUWorker.profile() so both profilers share one trigger.
        """
        profiler_config = self.vllm_config.profiler_config
        self.phased_profiling_dir = (
            profiler_config.torch_profiler_dir if envs.USE_PHASED_PROFILER else ""
        )
        self.phase_based_profiler = None

    def start_phased_profiling(
        self,
        profile_prefix: str | None = None,
        profiler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Arms the phase-based profiler. Called from TPUWorker.profile()."""
        if not self.phased_profiling_dir:
            logger.warning("Phased profiling directory is not set. Skipping profiling.")
            return
        if self.phase_based_profiler is not None:
            logger.warning(
                "Phased profiler is already running. Ignoring start request."
            )
            return
        profiler_config = self.vllm_config.profiler_config
        additional_config = self.vllm_config.additional_config
        decode_kv_len_threshold = additional_config.get(
            runner_utils.PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY,
            runner_utils.PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD,
        )
        prefill_kv_len_threshold = additional_config.get(
            runner_utils.PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD_KEY,
            runner_utils.PHASED_PROFILER_PREFILL_ONLY_KV_LEN_THRESHOLD,
        )
        # Same scoping as the standard profiler: the prefix names the run, and
        # the per-phase subdirectories sit beneath it.

        profile_dir, standard_opts, advanced_opts = resolve_profile_dir_and_opts(
            self.phased_profiling_dir, profile_prefix, profiler_kwargs=profiler_kwargs
        )

        # Deliberately not read from parallel_config: its rank is TPxPP-scoped,
        # so every DP replica would call itself rank 0 and their traces would
        # overwrite each other on merge. The worker resolves the slice-global
        # rank from the TPU rank binding and passes it in.
        self.phase_based_profiler = runner_utils.PhaseBasedProfiler(
            profile_dir,
            worker_rank=self._profiler_rank,
            world_size=self._profiler_world_size,
            # max_iterations defaults to 0 ("no limit") for standard torch
            # profiling; that's meaningless for the phased profiler, so fall
            # back to its own default when unset.
            num_steps_to_profile_for=(
                profiler_config.max_iterations
                or runner_utils.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR
            ),
            num_decode_steps_to_skip=profiler_config.delay_iterations,
            decode_kv_len_threshold=decode_kv_len_threshold,
            prefill_kv_len_threshold=prefill_kv_len_threshold,
            standard_opts=standard_opts,
            advanced_opts=advanced_opts,
        )

    def stop_phased_profiling(self) -> None:
        """Disarms the phase-based profiler. Called from TPUWorker.profile()."""
        if self.phase_based_profiler is None:
            logger.warning("Phased profiler is not running. Ignoring stop request.")
            return
        self.phase_based_profiler.finish()
        self.phase_based_profiler = None

    # ----- Backend hooks overridden from GPUModelRunner -----

    def _init_device_properties(self) -> None:
        # GPU sets self.num_sms from torch.cuda.get_device_properties; TPU
        # has no SM concept. The parent's call here is during __init__ but
        # we still want a no-op override so subclass invariants hold.
        pass

    def _sync_device(self) -> None:
        synchronize_device()

    def maybe_setup_kv_connector(
        self,
        scheduler_output,
        wait_for_completion: bool = False,
        report_completion: bool = True,
    ) -> None:
        if not has_kv_transfer_group():
            return
        kv_connector = get_kv_transfer_group()
        assert scheduler_output.kv_connector_metadata is not None
        # Upstream parity (gpu_model_runner calls this before every forward;
        # base-class default is a no-op): lets connectors with async saves
        # fence in-flight stores whose source blocks the scheduler is about
        # to reuse (OffloadingConnector jobs_to_flush) before the forward
        # overwrites them.
        kv_connector.handle_preemptions(scheduler_output.kv_connector_metadata)
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)
        # forward_context is unused by TPUConnector; pass None.
        if wait_for_completion:
            kv_connector.start_load_kv(
                None, wait_for_completion=True, report_completion=report_completion
            )
        else:
            kv_connector.start_load_kv(None)

    def maybe_wait_for_kv_save(self) -> None:
        if has_kv_transfer_group():
            get_kv_transfer_group().wait_for_save()

    def get_finished_kv_transfers(self, scheduler_output):
        if not has_kv_transfer_group():
            return None, None, None, set(), None, None
        kv_connector = get_kv_transfer_group()
        finished_sending, finished_recving = kv_connector.get_finished(
            scheduler_output.finished_req_ids
        )
        invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
        invalid_block_group_index = (
            kv_connector.get_block_ids_with_load_errors_group_index()
            if invalid_block_ids
            else None
        )
        # vLLM >=0.21 job model: store completions (and load completions) are
        # reported to the scheduler via the worker meta's `completed_jobs`,
        # NOT via finished_sending. Without plumbing this, the
        # OffloadingConnector's complete_store is never called, host-pool
        # blocks stay not-ready (ref_cnt=-1), and prefix-reuse lookups defer
        # forever -> the engine busy-spins / hangs. Mirrors
        # KVConnectorModelRunnerMixin._get_kv_connector_output.
        worker_meta = kv_connector.build_connector_worker_meta()
        kv_connector_stats = kv_connector.get_kv_connector_stats()
        # Mirror KVConnectorModelRunnerMixin._get_kv_connector_output:
        # metadata is bound per-step and must be cleared after use.
        kv_connector.clear_connector_metadata()
        return (
            finished_sending,
            finished_recving,
            worker_meta,
            invalid_block_ids,
            invalid_block_group_index,
            kv_connector_stats,
        )

    def kv_connector_no_forward(
        self, scheduler_output, vllm_config
    ) -> ModelRunnerOutput:
        # Only Raiden inline mode blocks; otherwise the no-forward step returns
        # immediately and the scheduler re-polls get_finished each step until
        # the loads land.
        self.maybe_setup_kv_connector(
            scheduler_output, wait_for_completion=dist_utils.get_raiden_inline_load()
        )
        (
            finished_sending,
            finished_recving,
            worker_meta,
            invalid_block_ids,
            invalid_block_group_index,
            kv_connector_stats,
        ) = self.get_finished_kv_transfers(scheduler_output)
        kv_connector_output = _build_kv_connector_output(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            kv_connector_worker_meta=worker_meta,
            invalid_block_ids=invalid_block_ids,
            invalid_block_group_index=invalid_block_group_index,
            kv_connector_stats=kv_connector_stats,
        )
        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    def _init_speculative_decoding(self) -> None:
        self.drafter = None
        self.rejection_sampler = None
        if self.speculative_config:
            self.rejection_sampler = RejectionSampler(
                self.speculative_config, self.device
            )
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "dflash":
                self.drafter = DFlashProposer(self, self.vllm_config)
            elif self.speculative_config.method == "dspark":
                self.drafter = DSparkProposer(self, self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = Eagle3Proposer(self, self.vllm_config)
            else:
                raise NotImplementedError(
                    "Unsupported speculative decoding method: "
                    f"{self.speculative_config.method}"
                )

    def _truncate_rope_caches(self) -> None:
        """Slice rotary cos_sin caches to max_model_len which can reduce
        the overhead of xla layout data copy. Applicable to text-only.
        """
        multimodal_config = self.model_config.multimodal_config
        if self.model_config.is_multimodal_model and not (
            multimodal_config is not None and multimodal_config.language_model_only
        ):
            logger.warning(
                "TPU_ROPE_CACHE_TRUNCATE skipped, multimodal positions are "
                "not bounded by max_model_len"
            )
            return
        max_len = self.model_config.max_model_len
        num_eligible = 0
        num_truncated = 0
        for name, module in self.model.named_modules():
            if type(module) not in (
                RotaryEmbedding,
                MRotaryEmbedding,
                MRotaryEmbeddingInterleaved,
            ):
                continue
            for buf_name in ("cos_sin_cache", "cos_sin_cache_bf16"):
                buf = getattr(module, buf_name, None)
                if buf is None or not isinstance(buf, torch.Tensor):
                    continue
                num_eligible += 1
                if buf.ndim >= 1 and buf.shape[0] > max_len:
                    truncated = buf[:max_len].clone(
                        memory_format=torch.contiguous_format
                    )
                    setattr(module, buf_name, truncated)
                    num_truncated += 1
                    logger.info(
                        "Truncated rope cache %s.%s rows %d -> %d",
                        name,
                        buf_name,
                        buf.shape[0],
                        max_len,
                    )
        if num_eligible:
            logger.info("Truncated %d of %d rope caches", num_truncated, num_eligible)
        else:
            logger.warning(
                "TPU_ROPE_CACHE_TRUNCATE is set but the model has no "
                "position-indexed rope cache, nothing was truncated"
            )

    def _create_mesh_for_parallelism(self) -> Mesh:
        local_devices = list(jax.local_devices())
        if not local_devices:
            raise ValueError("No TPU devices are visible to create JAX mesh.")

        # Per-worker JAX mesh is normally single-chip; vLLM native multiprocess
        # handles TP>1 outside this JAX SPMD path.
        if (
            self.parallel_config.world_size == 1
            and self.parallel_config.tensor_parallel_size > 1
        ):
            raise ValueError(
                "Single-process TPU mesh TP>1 is not supported in this path. "
                "Use vLLM multiprocess mode for --tensor-parallel-size > 1."
            )
        mesh_devices = np.asarray(local_devices[:1]).reshape((1,))
        mesh = Mesh(mesh_devices, axis_names=("model",))
        # Also report the global device view. This mesh is deliberately
        # single-device, but op-local multi-device meshes (see
        # distributed.pcp.get_or_create_pcp_mesh) are built from `jax.devices()`
        # and only work when every peer's device is visible here, so the counts
        # are worth having in every server log.
        logger.info(
            "Init mesh | tp_size=1 | device_id=%s | jax local_devices=%d "
            "global devices=%d ids=%s",
            getattr(local_devices[0], "id", str(local_devices[0])),
            len(local_devices),
            jax.device_count(),
            [getattr(d, "id", None) for d in jax.devices()],
        )
        return mesh

    # Entries described per report before the rest are summarized. A boot
    # compiles hundreds, and the interesting case -- a handful appearing once
    # traffic is running -- is well under this.
    _MAX_DESCRIBED_GRAPHS = 20

    # Newest entry seen by the previous report, which is what makes an entry
    # "first read since then". A class attribute rather than an __init__ one
    # because the first report runs from inside __init__, so an instance
    # attribute has to be assigned above that call to exist in time.
    _xla_graphs_checked_at = None

    @staticmethod
    def _describe_xla_graph(entry) -> str:
        """One line describing a cache entry, from whatever fields it carries.

        ``per_entry_stats`` holds an opaque pybind type whose fields differ
        between torch_tpu versions, so read them reflectively rather than
        pinning names that will silently vanish on an upgrade.
        """
        fields = []
        for name in sorted(dir(entry)):
            if name.startswith("_"):
                continue
            try:
                value = getattr(entry, name)
            except Exception:  # a property that needs live device state
                continue
            if callable(value):
                continue
            text = str(value).replace("\n", " ")
            if len(text) > 120:
                text = text[:117] + "..."
            fields.append(f"{name}={text}")
        return " ".join(fields) if fields else repr(entry)[:200]

    @staticmethod
    def _xla_graph_compile_secs(entry) -> float:
        duration = getattr(entry, "compilation_duration", None)
        total_seconds = getattr(duration, "total_seconds", None)
        return total_seconds() if total_seconds is not None else 0.0

    def _fresh_xla_graphs(self, entries) -> list:
        """The entries this process compiled since the previous report.

        There is no identifier on an entry to diff against, and the list is not
        append-only -- its tail routinely holds boot-time programs with
        ``read_count`` in the dozens -- so slicing off ``num_xla_graphs`` names
        the wrong graphs. What each entry does carry is when it was last read
        and how often, so an entry first read after the previous report is one
        this report is responsible for.
        """
        fresh, newest = [], self._xla_graphs_checked_at
        for entry in entries:
            last_read = getattr(entry, "last_read", None)
            if last_read is None:
                continue
            if newest is None or last_read > newest:
                newest = last_read
            if getattr(entry, "read_count", 0) != 1:
                continue  # read before, so it predates this report
            if (
                self._xla_graphs_checked_at is not None
                and last_read <= self._xla_graphs_checked_at
            ):
                continue
            fresh.append(entry)
        self._xla_graphs_checked_at = newest
        fresh.sort(key=self._xla_graph_compile_secs, reverse=True)
        return fresh

    def _update_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        stats = torch.tpu._get_cache_stats()
        entries = stats.per_entry_stats
        total_graphs = len(entries)
        if total_graphs < self.num_xla_graphs:
            # Entries were evicted, so the delta below would understate what
            # was compiled. Resynchronize rather than report a negative count.
            logger.info(
                "XLA cache shrank from %d to %d entries (case: %s); "
                "resynchronizing the recompilation counter",
                self.num_xla_graphs,
                total_graphs,
                case_str,
            )
            self.num_xla_graphs = total_graphs
            return

        fresh = self._fresh_xla_graphs(entries)
        new_compiled_graphs = total_graphs - self.num_xla_graphs
        if new_compiled_graphs == 0 and not fresh:
            logger.info("No new compiled graphs for case: %s", case_str)
            return

        logger.info("Total Requests: %s", stats.num_cache_reqs)
        logger.info("Total Hits: %s", stats.num_cache_hits)
        logger.info(
            "Total number of cached graphs: %s, new: %s, case: %s",
            total_graphs,
            new_compiled_graphs,
            case_str,
        )

        # A count says four graphs appeared; it does not say which op keeps
        # producing fresh shapes, which is the only thing that makes a runtime
        # recompilation actionable. Note the compile seconds: an entry with a
        # zero duration was served from a cache rather than compiled here, so
        # it costs nothing even though it counts.
        if fresh:
            secs = sum(map(self._xla_graph_compile_secs, fresh))
            logger.info(
                "  %d first read since the last check, %.3fs compiling",
                len(fresh),
                secs,
            )
            for i, entry in enumerate(fresh[: self._MAX_DESCRIBED_GRAPHS], 1):
                logger.info(
                    "  graph %d/%d: %s", i, len(fresh), self._describe_xla_graph(entry)
                )
            if len(fresh) > self._MAX_DESCRIBED_GRAPHS:
                logger.info(
                    "  ... %d more not shown", len(fresh) - self._MAX_DESCRIBED_GRAPHS
                )

        self.num_xla_graphs += new_compiled_graphs

    def _reorder_batch_for_rpa(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[int, int]:
        """Reorder active requests into an RPA-friendly decode-first layout.

        decode-only requests come first and all remaining requests stay in the
        mixed bucket. We do not create a dedicated prefill-only bucket here.

        With speculative decoding the order is three segments:
        [1-token decodes][spec verify windows][prefill/mixed]. Ragged paged
        attention keeps its 1-token decode front segment, while the GDN
        kernel's windowed mode covers the first two segments contiguously
        (see `AttentionMetadata.mamba_request_distribution`).

        Returns:
            (num_decode, num_windowed): the number of 1-token decode requests
            and the number of windowed requests (decodes + speculative verify
            windows) after reordering. Without spec decoding both are equal.
        """
        num_reqs = self.input_batch.num_reqs
        if num_reqs <= 0:
            return 0, 0

        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        def segment(req_id: str) -> int:
            # 0: 1-token decode, 1: speculative verify window,
            # 2: prefill/mixed.
            if scheduler_output.num_scheduled_tokens[req_id] == 1:
                return 0
            if req_id in spec_decode_tokens:
                return 1
            return 2

        def partition(start: int, end: int, bound: int) -> int:
            """Two-pointer partition of [start, end]: requests with
            segment <= bound before the rest. Returns the first index of the
            second part."""
            i, j = start, end
            while i < j:
                i_req_id = self.input_batch.req_ids[i]
                j_req_id = self.input_batch.req_ids[j]
                assert i_req_id is not None
                assert j_req_id is not None
                if segment(i_req_id) <= bound:
                    i += 1
                elif segment(j_req_id) > bound:
                    j -= 1
                else:
                    self.input_batch.swap_states(i, j)
                    i += 1
                    j -= 1
            if i == j and segment(self.input_batch.req_ids[i]) <= bound:
                i += 1
            return i

        # Pass 1: 1-token decode requests to the front.
        num_decode = partition(0, num_reqs - 1, 0)
        # Pass 2: speculative verify windows before prefill/mixed requests.
        num_windowed = num_decode
        if spec_decode_tokens and num_decode < num_reqs:
            num_windowed = partition(num_decode, num_reqs - 1, 1)
        return num_decode, num_windowed

    def _build_attention_metadata(
        self,
        num_tokens: int = 0,
        num_reqs: int = 0,
        max_query_len: int = 1,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        slot_mappings: torch.Tensor | None = None,
    ) -> tuple[dict[str, AttentionMetadata], None]:
        """Lean TPU group walk: stage one batched block-table upload, then
        build every group's metadata straight from the builder context.
        """
        ctx = self._attn_metadata_builder_ctx
        effective_num_reqs_padded = (
            num_reqs_padded if num_reqs_padded is not None else num_reqs
        )

        if ctx.staged_block_tables is None:
            # One batched H2D for every group's block table (zero-filled for
            # dummy runs); see `stage_block_table_uploads`.
            stage_block_table_uploads(self, ctx, effective_num_reqs_padded)

        common = SimpleNamespace(num_reqs=effective_num_reqs_padded)
        per_layer_attn_metadata: dict[str, AttentionMetadata] = {}
        for group_list in self.attn_groups:
            for group in group_list:
                built = group.metadata_builders[0].build(
                    common_prefix_len=0,
                    common_attn_metadata=common,
                )
                for layer_name in group.layer_names:
                    per_layer_attn_metadata[layer_name] = built
        return per_layer_attn_metadata, None

    def build_attention_metadata_for_layers(
        self,
        layer_names: Collection[str],
        num_reqs_padded: int,
    ) -> dict[str, AttentionMetadata]:
        """Build attention metadata for only the groups owning `layer_names`.

        `_build_attention_metadata` walks every KV cache group. A drafter
        needs only its own layers, and the group count tracks the layer count
        whenever vLLM's spec bucketing collapses — which a drafter mixing
        window and full attention causes on its own. Building every group and
        discarding all but the draft's costs a block-table H2D plus a handful
        of gathers per discarded group, per chunk, per step.

        Reads the same `_attn_metadata_builder_ctx` the full build does, so
        callers stage it identically.
        """
        # `AttentionMetadataBuilder.build` takes only `num_reqs` off the
        # common metadata; everything else it needs comes from
        # `_attn_metadata_builder_ctx`.
        selected_groups = []
        for group_list in self.attn_groups:
            for group in group_list:
                selected = [n for n in group.layer_names if n in layer_names]
                if selected:
                    selected_groups.append((group, selected))
        ctx = self._attn_metadata_builder_ctx
        if ctx.staged_block_tables is None and selected_groups:
            # One H2D for exactly the groups this targeted walk will build.
            stage_block_table_uploads(
                self,
                ctx,
                num_reqs_padded,
                gids={
                    group.metadata_builders[0].kv_cache_group_id
                    for group, _ in selected_groups
                },
            )
        common = SimpleNamespace(num_reqs=num_reqs_padded)
        metadata: dict[str, AttentionMetadata] = {}
        for group, selected in selected_groups:
            built = group.metadata_builders[0].build(
                common_prefix_len=0,
                common_attn_metadata=common,
            )
            for name in selected:
                metadata[name] = built
        return metadata

    # KV-cache spec derivation, sizing and allocation live on
    # `self.kv_cache_manager` now; these forward the runner's existing callers.
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        with set_current_vllm_config(self.vllm_config):
            return self.kv_cache_manager.get_kv_cache_spec()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        with set_current_vllm_config(self.vllm_config):
            self.kv_cache_manager.initialize_kv_cache(kv_cache_config)

    def get_kv_prewarm_shapes(self) -> list[int]:
        return self.kv_cache_manager.get_kv_prewarm_shapes()

    def prewarm_kv_offload_shape(self, p: int) -> None:
        self.kv_cache_manager.prewarm_kv_offload_shape(p)

    def _init_mamba_slot_pool(self, mamba_num_blocks: int) -> None:
        """(Re)initialize the per-request mamba recurrent-slot allocator.

        Slot 0 is the null/sentinel block (used for padded persistent-batch
        positions and never handed to a request). Usable slots are
        `[1, mamba_num_blocks)`, popped from the high end so the first request
        gets slot 1 (stable pool ordering for easier debugging).
        Called once after KV cache allocation, when the true mamba block count
        is known. The mapping itself is keyed on req_id and rebuilt lazily in
        `_build_mamba_state_indices`, so re-init only resets the free pool.

        With speculative decoding each request owns a *group* of
        `num_speculative_tokens + 1` consecutive slots; the tracked slot id is
        the group's base. During a verify step the GDN kernel writes one state
        checkpoint per window position into `base + t`, and the next step
        reads its initial state from `base + (num_accepted - 1)`. Rejected
        drafts are thus rolled back by *selection*, never by copying. Without
        spec decode the stride is 1 (one slot per request), matching the
        original layout.
        """
        stride = self._mamba_slot_stride
        num_groups = (mamba_num_blocks - 1) // stride
        self._mamba_slot_by_req_id = {}
        self._free_mamba_slots = [1 + g * stride for g in reversed(range(num_groups))]
        self._has_mamba_state = True

    def _build_mamba_state_indices(
        self, start_index: int, num_reqs: int, target_num_reqs: int
    ) -> torch.Tensor:
        """Build the device `mamba_state_indices` for the current chunk.

        Reconciles the req_id->slot map against the live persistent batch:
          * frees slots whose req_id is no longer present (request finished),
          * allocates a fresh unique slot for any new req_id.
        Returns an int32 device tensor of length `target_num_reqs`: positions
        `[0, num_reqs)` hold the slot for `req_ids[start_index + i]`; the
        padded tail `[num_reqs, target_num_reqs)` holds slot 0 (null block) so
        the GDN op (which scans the full length every step) cannot alias an
        active slot — the null-tail invariant.

        `num_reqs` may exceed the requests actually in the persistent batch:
        an idle DP rank replaying a peer's collective trace passes the
        coordinated shape (`num_reqs_max_model_len`) while its own batch is
        empty or shorter. Those positions carry no request, so they keep the
        null slot the tail already uses; only the live prefix is assigned.
        Without this, `run_dp_dummy_draft` walks off the end of `req_ids`
        with `IndexError` on the compact layout, killing the engine.

        Slot assignment follows the request through upstream vLLM's
        condense/swap reordering automatically because the map is keyed on the
        stable req_id, not the moving persistent-batch position.
        """
        live_req_ids = set(self.input_batch.req_id_to_index.keys())
        # Free slots for requests that left the batch.
        for req_id in list(self._mamba_slot_by_req_id.keys()):
            if req_id not in live_req_ids:
                self._free_mamba_slots.append(self._mamba_slot_by_req_id.pop(req_id))

        indices = self.mamba_state_indices_cpu[:target_num_reqs]
        indices.fill_(0)
        req_ids = self.input_batch.req_ids
        num_live = max(0, min(num_reqs, len(req_ids) - start_index))
        for i in range(num_live):
            req_id = req_ids[start_index + i]
            assert req_id is not None
            slot = self._mamba_slot_by_req_id.get(req_id)
            if slot is None:
                # New request: take a fresh unique slot. The pool is sized to
                # max_num_reqs + 1, so it can never be exhausted while the
                # batch holds at most max_num_reqs requests.
                slot = self._free_mamba_slots.pop()
                self._mamba_slot_by_req_id[req_id] = slot
            indices[i] = slot
        return indices.to(self.device, non_blocking=True)

    def _build_mamba_copy_plan(
        self,
        kv_cache_config: KVCacheConfig,
        raw_tensors: list[torch.Tensor],
    ) -> None:
        """Map each mamba kv-cache group to the raw pool buffers hosting its
        layers, for the per-step state-block seed copies.

        Seed copies only exist in mamba align mode, where the state block
        follows the request's last token through the block table. In any
        other mode each request keeps one fixed state block for its whole
        lifetime, so the plan stays empty and the per-step collector is a
        no-op.
        """
        if self.cache_config.mamba_cache_mode != "align":
            self._mamba_copy_plan = []
            self._mamba_state_block_size = None
            return
        from vllm_torchtpu.kv_cache_materializer import layer_to_pool_index

        layer_to_raw = {
            name: raw_tensors[index]
            for name, index in layer_to_pool_index(kv_cache_config).items()
        }
        plan: list[tuple[int, list[torch.Tensor]]] = []
        state_block_size: int | None = None
        manager_page_bytes: int | None = None
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if not isinstance(group.kv_cache_spec, MambaSpec):
                continue
            raws: list[torch.Tensor] = []
            seen: set[int] = set()
            for layer_name in group.layer_names:
                raw = layer_to_raw.get(layer_name)
                if raw is not None and id(raw) not in seen:
                    seen.add(id(raw))
                    raws.append(raw)
            if raws:
                plan.append((gid, raws))
                state_block_size = group.kv_cache_spec.block_size
                manager_page_bytes = group.kv_cache_spec.page_size_bytes
        self._mamba_state_block_size = state_block_size
        self._mamba_copy_plan = plan
        for raw in raw_tensors:
            if raw.dim() > 1:
                # Manager blocks span whole pool pages. Which axis of the
                # pool tensor carries tokens is backend-dependent — shape[1]
                # for RPA pools, the last axis for the seq-on-lane (HND)
                # layout, and for token-axis-packed caches (MLA) no single
                # axis IS the kernel block's token count — so derive the
                # manager-to-pool-block split from bytes, which is
                # layout-agnostic: per-kernel-block bytes are
                # prod(shape[1:]) * element_size in every layout, and the
                # manager page comes from the mamba spec the platform
                # derivation already sized against the backend's own
                # page-size function. The divisibility assert is the guard.
                pool_block_bytes = math.prod(raw.shape[1:]) * raw.element_size()
                assert manager_page_bytes is not None
                assert manager_page_bytes % pool_block_bytes == 0, (
                    manager_page_bytes,
                    pool_block_bytes,
                )
                self._pool_block_split = manager_page_bytes // pool_block_bytes
                break

    def _collect_mamba_state_seed_copies(
        self, scheduler_output, start_index: int, num_reqs: int
    ) -> None:
        """Stage this chunk's mamba state-block seed copies (the counterpart
        of upstream vLLM's ``preprocess_mamba``).

        The GDN op reads and writes state at the block holding the request's
        last token of the step (see AttentionMetadataBuilder). When that
        block advances — chunked-prefill boundary, decode crossing, or a
        prefix-cache resume, where the previous position is the cached
        boundary-state block — the new block must be seeded from the
        previous one, for every mamba group, on every pool buffer hosting
        that group's layers.
        """
        if not self._mamba_copy_plan:
            return
        live_req_ids = set(self.input_batch.req_id_to_index.keys())
        for req_id in list(self._mamba_state_pos.keys()):
            if req_id not in live_req_ids:
                del self._mamba_state_pos[req_id]

        # Column stride of the mamba block tables, in logical (whole-sequence)
        # tokens. Three block sizes meet here and only one is correct:
        #   * self.block_size is the attention/scheduler size -- unrelated to
        #     mamba tables, whose width is cdiv(max_model_len, mamba size) --
        #     and a stale __init__-time snapshot besides (the executor
        #     finalizes cache_config.block_size only after the runner is
        #     constructed);
        #   * the mamba groups' kv_cache_spec.block_size (the copy of
        #     cache_config.mamba_block_size that get_kv_cache_spec bakes into
        #     every MambaSpec, which then sizes these tables) is physical and
        #     per-rank, and cannot be stale: specs are created only after the
        #     platform finalizes block sizes;
        #   * the scheduler appends a block id to a mamba row only once per
        #     mamba_block_size * total_cp_world logical tokens, because the
        #     PCP coordinator patch presents specs to the hybrid coordinator
        #     at that granularity (_patch_vllm_hybrid_pcp_block_sizes). Mamba
        #     state is never token-sharded across CP ranks; the cp factor
        #     exists purely to mirror that allocation cadence.
        # Every reader of these tables must divide logical token counts by
        # the allocator's stride. This is the same derivation
        # AttentionMetadataBuilder.target_block_size uses for its state-slot
        # lookup -- spec block size times cp world -- so the two readers
        # cannot diverge. A smaller divisor visits columns the scheduler
        # never filled: beyond the row width it raises IndexError, within it
        # the null block silently swallows the recurrent state.
        try:
            cp_world_size = get_pcp_group().world_size
        except Exception:
            cp_world_size = 1
        col_stride = self._mamba_state_block_size * cp_world_size
        num_computed = self.input_batch.num_computed_tokens_cpu
        req_ids = self.input_batch.req_ids
        crossings: list[tuple[int, int, int]] = []
        for i in range(num_reqs):
            row = start_index + i
            req_id = req_ids[row]
            assert req_id is not None
            computed = int(num_computed[row])
            scheduled = scheduler_output.num_scheduled_tokens[req_id]
            curr = (computed + scheduled - 1) // col_stride
            prev = self._mamba_state_pos.get(req_id, (computed - 1) // col_stride)
            self._mamba_state_pos[req_id] = curr
            if 0 <= prev != curr:
                crossings.append((row, prev, curr))
        if not crossings:
            return

        # Speculative decoding: the checkpoint group slides with the state
        # column. vLLM's MambaManager appends one block and never shifts, so
        # after the crossing what was checkpoint 1 IS the state block,
        # checkpoint 2 is checkpoint 1, and a fresh block appears at the
        # end. Carrying the old state block over would therefore leave the
        # request resuming one checkpoint late (and, at the top of the
        # window, off an uninitialized block). What must move instead is the
        # checkpoint the request is about to resume from — which becomes
        # checkpoint 0 of the new group, so the read offset resets to 0.
        # That index lives in the device-resident read-offset buffer, hence
        # the device-side gather in `_spec_seed_sources`.
        window = self._mamba_ckpt_window
        spec_seed = self.mamba_slot_read_offsets is not None and window > 1

        # Keyed by the set of raw buffers a group lives on; groups sharing
        # the same buffers are seeded by one program.
        per_raw: dict[
            tuple[int, ...], tuple[list[torch.Tensor], list[int], list[int]]
        ] = {}
        per_raw_dev: dict[
            tuple[int, ...],
            tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        # Manager-level (src, dst) block pairs across mamba groups: without
        # spec decoding the per-block read offsets just follow the state to
        # the request's new state block.
        offset_pairs: list[tuple[int, int]] = []
        spec_dsts: list[int] = []
        for gid, raws in self._mamba_copy_plan:
            bt = self.input_batch.block_table[gid].get_cpu_tensor()
            row_width = bt.shape[1]
            pairs: list[tuple[int, int]] = []
            state_srcs: list[int] = []
            ckpt_rows: list[list[int]] = []
            for row, prev, curr in crossings:
                src = int(bt[row, prev])
                dst = int(bt[row, curr])
                if src == dst or src == 0 or dst == 0:
                    continue
                pairs.append((src, dst))
                if spec_seed:
                    state_srcs.append(src)
                    # The pre-crossing group's checkpoint blocks. A column
                    # the manager has not filled reads past the row or lands
                    # on the null block; either way fall back to the state
                    # block, so a short row seeds from live state instead of
                    # from nothing. The offset never names one in practice —
                    # it is bounded by the accepted count.
                    row_ckpts = []
                    for t in range(window):
                        col = prev + t
                        ckpt = int(bt[row, col]) if col < row_width else 0
                        row_ckpts.append(ckpt if ckpt != 0 else src)
                    ckpt_rows.append(row_ckpts)
            if not pairs:
                continue
            if spec_seed:
                dsts_g = [d for _, d in pairs]
                spec_dsts.extend(dsts_g)
                src_t = self._spec_seed_sources(state_srcs, ckpt_rows)
                dst_t = torch.tensor(dsts_g, dtype=torch.int32).to(
                    self.device, non_blocking=True
                )
                src_t, dst_t = self._expand_pool_split(src_t, dst_t)
                key = tuple(id(raw) for raw in raws)
                entry = per_raw_dev.setdefault(key, (raws, [], []))
                entry[1].append(src_t)
                entry[2].append(dst_t)
                continue
            offset_pairs.extend(pairs)
            if self._pool_block_split > 1:
                # The pool is born at kernel granularity: a manager state
                # block is `split` consecutive pool blocks.
                pairs = [
                    (s * self._pool_block_split + j, d * self._pool_block_split + j)
                    for s, d in pairs
                    for j in range(self._pool_block_split)
                ]
            key = tuple(id(raw) for raw in raws)
            entry = per_raw.setdefault(key, (raws, [], []))
            entry[1].extend(p[0] for p in pairs)
            entry[2].extend(p[1] for p in pairs)

        if spec_seed and spec_dsts:
            # The seeded checkpoint is checkpoint 0 of the new group.
            dst_t = torch.tensor(self._pad_to_bucket(spec_dsts), dtype=torch.long).to(
                self.device, non_blocking=True
            )
            assert self.mamba_slot_read_offsets is not None
            _rollback_offsets_seed(self.mamba_slot_read_offsets, dst_t)
        elif self.mamba_slot_read_offsets is not None and offset_pairs:
            # Migrate the read offsets with the state: gather at the old
            # blocks, scatter at the new ones (both device-side, before the
            # forward that reads the buffer). Padding with (0, 0) null-block
            # self-copies keeps the shapes on the seed-copy bucket ladder.
            offset_pairs = self._pad_to_bucket(offset_pairs, pad=(0, 0))
            src_t = torch.tensor([s for s, _ in offset_pairs], dtype=torch.long).to(
                self.device, non_blocking=True
            )
            dst_t = torch.tensor([d for _, d in offset_pairs], dtype=torch.long).to(
                self.device, non_blocking=True
            )
            _rollback_offsets_migrate(self.mamba_slot_read_offsets, src_t, dst_t)

        for raws, srcs_dev, dsts_dev in per_raw_dev.values():
            src_t = self._pad_dev_to_bucket(torch.cat(srcs_dev))
            dst_t = self._pad_dev_to_bucket(torch.cat(dsts_dev))
            self._pending_mamba_state_copies.append((raws, src_t, dst_t))

        for raws, srcs, dsts in per_raw.values():
            src_t = torch.tensor(self._pad_to_bucket(srcs), dtype=torch.int32).to(
                self.device, non_blocking=True
            )
            dst_t = torch.tensor(self._pad_to_bucket(dsts), dtype=torch.int32).to(
                self.device, non_blocking=True
            )
            self._pending_mamba_state_copies.append((raws, src_t, dst_t))

    @staticmethod
    def _bucket_len(n: int) -> int:
        """Smallest ladder length >= `n`, so the copy op sees few shapes."""
        padded = 8
        while padded < n:
            padded *= 4
        return padded

    @classmethod
    def _pad_to_bucket(cls, values: list, pad=0) -> list:
        """Pad a CPU pair/id list up to the seed-copy bucket ladder.

        Pads are the null block, i.e. `(0, 0)` self-copies that move no
        live state.
        """
        return values + [pad] * (cls._bucket_len(len(values)) - len(values))

    def _pad_dev_to_bucket(self, ids: torch.Tensor) -> torch.Tensor:
        """`_pad_to_bucket` for an already-on-device id tensor."""
        n = int(ids.shape[0])
        padded = self._bucket_len(n)
        if padded == n:
            return ids
        return torch.cat(
            [
                ids,
                torch.zeros(padded - n, dtype=ids.dtype, device=ids.device),
            ]
        )

    def _expand_pool_split(
        self, src: torch.Tensor, dst: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Device twin of the manager-block -> pool-block expansion.

        The pool is born at kernel granularity: a manager state block is
        `split` consecutive pool blocks.
        """
        split = self._pool_block_split
        if split <= 1:
            return src, dst
        j = torch.arange(split, dtype=src.dtype, device=src.device)
        return (
            (src.unsqueeze(1) * split + j).reshape(-1),
            (dst.unsqueeze(1) * split + j).reshape(-1),
        )

    def _spec_seed_sources(
        self, state_srcs: list[int], ckpt_rows: list[list[int]]
    ) -> torch.Tensor:
        """Seed source per crossing: the checkpoint being resumed from.

        `mamba_slot_read_offsets` is device-resident and indexed by block
        id, so which of the window's blocks holds the last accepted token's
        state is only known on device — the choice is a gather there rather
        than a CPU lookup.
        """
        offsets = self.mamba_slot_read_offsets
        assert offsets is not None
        state_t = torch.tensor(state_srcs, dtype=torch.long).to(
            self.device, non_blocking=True
        )
        ckpt_t = torch.tensor(ckpt_rows, dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        off = offsets[state_t].long().clamp_(0, ckpt_t.shape[1] - 1)
        return ckpt_t.gather(1, off.unsqueeze(1)).squeeze(1)

    def _flush_mamba_state_seed_copies(self) -> None:
        if not self._pending_mamba_state_copies:
            return
        for raws, src_t, dst_t in self._pending_mamba_state_copies:
            copy_mamba_state_blocks(raws, src_t, dst_t)
        self._pending_mamba_state_copies.clear()

    def is_unified_pool_used(self) -> bool:
        """Whether the attention-shaped unified KV block pool is active
        and allocated.
        """
        return bool(getattr(self, "kv_cache_raw_tensors", None))

    def _apply_kv_cache_block_copies(self, copies) -> None:
        """Copy CoW blocks within each unified pool buffer page-locally.

        Scheduler pairs name manager blocks. A manager block can span several
        kernel-granular rows, so each pair fans out by ``_pool_block_split``.
        """
        assert copies, "page-local KV block copy requires at least one pair"
        assert self.is_unified_pool_used(), (
            "page-local KV block copy requires the unified KV pool"
        )

        if getattr(self, "mamba_slot_read_offsets", None) is not None:
            mgr_pairs = [
                (block_copy.src_block_id, block_copy.dst_block_id)
                for block_copy in copies
            ]
            mgr_pairs = self._pad_to_bucket(mgr_pairs, pad=(0, 0))
            mgr_src_t = torch.tensor([s for s, _ in mgr_pairs], dtype=torch.long).to(
                self.device, non_blocking=True
            )
            mgr_dst_t = torch.tensor([d for _, d in mgr_pairs], dtype=torch.long).to(
                self.device, non_blocking=True
            )
            _rollback_offsets_migrate(
                self.mamba_slot_read_offsets, mgr_src_t, mgr_dst_t
            )

        split = self._pool_block_split
        pairs = [
            (block_copy.src_block_id * split + j, block_copy.dst_block_id * split + j)
            for block_copy in copies
            for j in range(split)
        ]
        pairs = self._pad_to_bucket(pairs, pad=(0, 0))
        src_t = torch.tensor([src for src, _ in pairs], dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        dst_t = torch.tensor([dst for _, dst in pairs], dtype=torch.int32).to(
            self.device, non_blocking=True
        )
        copy_mamba_state_blocks(self.kv_cache_raw_tensors, src_t, dst_t)

    def _prepare_async_token_substitution_indices(
        self, start_index: int, num_reqs: int, num_scheduled_tokens_per_req: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._pre_async_results is None:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        token_in_tpu_cur_input_indices_list = []
        token_in_tpu_pre_next_tokens_indices_list = []
        acc_cur_len = 0
        layout_plan = self._last_sequence_layout_plan

        # 1+K for spec — `[bonus, draft_1..K]`), so a request's source span
        # starts at `position * stride`.
        stride = 1
        if (
            self.speculative_config is not None
            and self._pre_async_results.spec_decode_num_rejected_tokens is not None
        ):
            stride = 1 + self.speculative_config.num_speculative_tokens

        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[start_index + i]
            n_sched = int(num_scheduled_tokens_per_req[i])
            acc_cur_len += n_sched
            assert req_id is not None
            if req_id not in self._pre_async_results.req_id_to_index_copy:
                continue

            # Map this request's last `n_sched` input slots to its source span
            # `[position*stride .. +n_sched-1]`. Non-spec (stride 1, n_sched 1)
            # reduces to the original single-slot mapping.
            src_start = self._pre_async_results.req_id_to_index_copy[req_id] * stride
            for j in range(n_sched):
                request_major_index = acc_cur_len - n_sched + j
                if layout_plan is None:
                    cur_input_index = request_major_index
                else:
                    cur_input_index = layout_plan.local_index_for_request_major_token(
                        request_major_index
                    )
                    if cur_input_index is None:
                        continue
                token_in_tpu_cur_input_indices_list.append(cur_input_index)
                token_in_tpu_pre_next_tokens_indices_list.append(src_start + j)

        if len(token_in_tpu_cur_input_indices_list) > 0:
            return (
                np.array(token_in_tpu_cur_input_indices_list, dtype=np.int32),
                np.array(token_in_tpu_pre_next_tokens_indices_list, dtype=np.int32),
            )
        else:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    def _apply_async_token_substitution(
        self,
        input_ids: torch.Tensor,
        token_in_tpu_cur_input_indices: np.ndarray,
        token_in_tpu_pre_next_tokens_indices: np.ndarray,
    ) -> torch.Tensor:
        """Apply async token substitution if needed."""
        if len(token_in_tpu_cur_input_indices) == 0:
            return input_ids

        if (
            self._pre_async_results is None
            or self._pre_async_results.next_tokens_tpu is None
        ):
            return input_ids

        if getattr(self, "_fast_token_substitution", False):
            # Fast path 1: Direct device token slicing for 1:1 steady-state decode
            # Bypasses the _substitute_placeholder_token kernel entirely when unpadded
            n_subst = len(token_in_tpu_cur_input_indices)
            if (
                n_subst == len(input_ids)
                and np.array_equal(
                    token_in_tpu_cur_input_indices,
                    np.arange(len(input_ids), dtype=np.int32),
                )
                and np.array_equal(
                    token_in_tpu_pre_next_tokens_indices,
                    np.arange(len(input_ids), dtype=np.int32),
                )
            ):
                next_tpu = self._pre_async_results.next_tokens_tpu.flatten().to(
                    input_ids.dtype
                )
                if len(next_tpu) >= len(input_ids):
                    return next_tpu[: len(input_ids)]

            # Fast path 2: Cache persistent device indices to eliminate per-step
            # H2D uploads
            # Avoids Eager XLA overhead by relying on the compiled kernel + cached
            # index tensors
            subst_key = (
                len(input_ids),
                tuple(token_in_tpu_cur_input_indices),
                tuple(token_in_tpu_pre_next_tokens_indices),
            )
            if (
                getattr(self, "_cached_subst_key", None) == subst_key
                and hasattr(self, "_cached_cur_input_indices")
                and hasattr(self, "_cached_pre_next_tokens_indices")
            ):
                cur_input_indices = self._cached_cur_input_indices
                pre_next_tokens_indices = self._cached_pre_next_tokens_indices
            else:
                idx_pad_len = len(input_ids) - len(token_in_tpu_cur_input_indices)
                _missing_mask = np.ones(len(input_ids), dtype=bool)
                _missing_mask[token_in_tpu_cur_input_indices] = False
                missing_values = np.flatnonzero(_missing_mask).astype(np.int32)
                padded_token_in_tpu_cur_input_indices = np.concatenate(
                    (token_in_tpu_cur_input_indices, missing_values)
                )

                padded_token_in_tpu_pre_next_tokens_indices = np.pad(
                    token_in_tpu_pre_next_tokens_indices,
                    (0, idx_pad_len),
                    mode="constant",
                    constant_values=-1,
                )

                cur_input_indices = torch.from_numpy(
                    padded_token_in_tpu_cur_input_indices
                ).to(self.device, non_blocking=True)
                pre_next_tokens_indices = torch.from_numpy(
                    padded_token_in_tpu_pre_next_tokens_indices
                ).to(self.device, non_blocking=True)
                self._cached_subst_key = subst_key
                self._cached_cur_input_indices = cur_input_indices
                self._cached_pre_next_tokens_indices = pre_next_tokens_indices
        else:
            idx_pad_len = len(input_ids) - len(token_in_tpu_cur_input_indices)
            _missing_mask = np.ones(len(input_ids), dtype=bool)
            _missing_mask[token_in_tpu_cur_input_indices] = False
            missing_values = np.flatnonzero(_missing_mask).astype(np.int32)
            padded_token_in_tpu_cur_input_indices = np.concatenate(
                (token_in_tpu_cur_input_indices, missing_values)
            )

            padded_token_in_tpu_pre_next_tokens_indices = np.pad(
                token_in_tpu_pre_next_tokens_indices,
                (0, idx_pad_len),
                mode="constant",
                constant_values=-1,
            )

            cur_input_indices = torch.from_numpy(
                padded_token_in_tpu_cur_input_indices
            ).to(self.device, non_blocking=True)
            pre_next_tokens_indices = torch.from_numpy(
                padded_token_in_tpu_pre_next_tokens_indices
            ).to(self.device, non_blocking=True)

        return _substitute_placeholder_token(
            input_ids,
            cur_input_indices,
            pre_next_tokens_indices,
            self._pre_async_results.next_tokens_tpu,
        )

    def _flush_disjoint_async_results(self) -> None:
        if self._pre_async_results is None:
            return
        current_req_ids = set(self.input_batch.req_ids[: self.input_batch.num_reqs])
        previous_req_ids = self._pre_async_results.req_id_to_index_copy
        if any(req_id in current_req_ids for req_id in previous_req_ids):
            return
        self._modify_prev_results()
        self._pre_async_results = None

    def _install_spec_token_room_guard(self) -> None:
        """Wrap InputBatch.update_req_spec_token_ids to clamp its *write*.

        Upstream stages the scheduled drafts at num_tokens_no_spec[req_index].
        Under async scheduling that position is optimistic -- every in-flight
        step charges 1+K assuming full acceptance, and the over-count is only
        rolled back by _modify_prev_results, which runs later (sample_tokens)
        than this call (execute_model -> _update_states). Near max_model_len
        the inflated position leaves fewer slots than K, and the unclamped
        write runs off the end of token_ids_cpu, killing the engine.

        Only the write may be clamped; the scheduled draft *count* must stay
        exactly what the scheduler charged. num_scheduled_tokens is already
        fixed at 1+K, and get_spec_decode_metadata anchors each request's
        logits window at the END of its query
        (cu_num_scheduled_tokens - num_sampled_tokens). Trimming the count
        slides that window forward by the trimmed amount, so the rejection
        sampler verifies the LAST `room` drafts while the tokens actually
        staged are the first ones -- silent greedy divergence instead of a
        crash. Dropping only the tail of the write is safe because these
        slots are placeholders under async: the substitution rewrites every
        scheduled slot from the device [bonus, draft_1..K] source, and
        draft_token_ids is re-extracted from the post-substitution input_ids.
        """
        input_batch = self.input_batch
        if getattr(input_batch, "_tpu_spec_room_guard", False):
            return
        orig = input_batch.update_req_spec_token_ids

        def guarded(request, scheduled_spec_tokens):
            req_id = request.req_id
            ids = scheduled_spec_tokens.get(req_id)
            idx = input_batch.req_id_to_index.get(req_id)
            if not ids or idx is None:
                return orig(request, scheduled_spec_tokens)

            room = max(
                0,
                input_batch.token_ids_cpu.shape[1]
                - int(input_batch.num_tokens_no_spec[idx]),
            )
            if len(ids) <= room:
                return orig(request, scheduled_spec_tokens)

            # Fires routinely for a request's final steps near max_model_len:
            # async placeholder accounting overshoots the scheduler-side bound
            # there. Stage only the drafts that fit, then restore the full
            # scheduled list so every downstream count is untouched.
            logger.debug(
                "Clamping the draft-token write for %s to %d of %d slots at "
                "the context limit.",
                req_id,
                room,
                len(ids),
            )
            scheduled_spec_tokens[req_id] = ids[:room]
            try:
                orig(request, scheduled_spec_tokens)
            finally:
                scheduled_spec_tokens[req_id] = ids

            # orig() recorded the clamped length; restore the scheduled one.
            request.prev_num_draft_len = len(ids)
            cur_spec_token_ids = getattr(input_batch, "spec_token_ids", None)
            if cur_spec_token_ids is not None:
                cur_spec_token_ids[idx].clear()
                cur_spec_token_ids[idx].extend(ids)
            return None

        input_batch.update_req_spec_token_ids = guarded
        input_batch._tpu_spec_room_guard = True

    def _update_states(self, scheduler_output: "SchedulerOutput") -> Any:
        # Disable vLLM's native GPU async spec-decode num_computed_tokens
        # correction. The TPU runner does its own async rejection correction,
        # so letting the base also correct would double-count.
        for req_state in self.requests.values():
            req_state.prev_num_draft_len = 0
        self._install_spec_token_room_guard()
        copies = scheduler_output.kv_cache_block_copies
        use_page_local_copy = bool(copies and self.is_unified_pool_used())
        # The base implementation uses an advanced-index assignment which
        # TorchTPU functionalizes into a full raw-storage result. Hide only
        # the copy field from it; block zeroing and state updates stay intact.
        base_output = (
            dataclasses.replace(scheduler_output, kv_cache_block_copies=None)
            if use_page_local_copy
            else scheduler_output
        )
        result = super()._update_states(base_output)
        if use_page_local_copy:
            self._apply_kv_cache_block_copies(copies)
        return result

    def _modify_prev_results(self):
        if self._pre_async_results is None:
            return

        pre_req_ids = self._pre_async_results.req_ids
        pre_request_seq_lens = self._pre_async_results.request_seq_lens
        pre_discard_sampled_tokens_req_indices = (
            self._pre_async_results.discard_sampled_tokens_req_indices
        )
        # Per-request draft count from the previous step (keyed by the
        # request_seq_lens index); sizes the optimistic-placeholder rollback
        # below. None/absent -> 0 drafts (non-spec or a non-spec mixed chunk).
        pre_num_draft_per_req = self._pre_async_results.num_draft_per_req

        pre_next_tokens_cpu = self._pre_async_results.wait_for_copy()
        assert pre_next_tokens_cpu is not None

        pre_next_tokens_cpu = pre_next_tokens_cpu[: len(pre_req_ids)]
        max_gen_len = pre_next_tokens_cpu.shape[-1]

        if max_gen_len == 1:
            valid_sampled_token_ids = pre_next_tokens_cpu.tolist()
        else:
            valid_mask = pre_next_tokens_cpu != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist() for seq in pre_next_tokens_cpu[valid_mask].split(gen_lens)
            ]

        for i in pre_discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()

        for pre_req_idx, req_state, seq_len, req_id in pre_request_seq_lens:
            sampled_ids = valid_sampled_token_ids[pre_req_idx]
            if not sampled_ids:
                continue

            # The previous step optimistically appended (1 + num_draft)
            # placeholder tokens to output_token_ids, assuming every draft was
            # accepted. Now that the true acceptance is known, drop the entire
            # optimistic guess and append the actual sampled tokens.
            n_placeholder = 1 + (
                pre_num_draft_per_req.get(pre_req_idx, 0)
                if pre_num_draft_per_req
                else 0
            )
            # At most (1 bonus + num_draft) tokens can commit this step, so the
            # optimistic placeholder count must cover the real sampled count.
            # num_sampled_tokens <= pre_num_placeholder_tokens guard.
            assert len(sampled_ids) <= n_placeholder, (
                f"req {req_id}: sampled {len(sampled_ids)} tokens > "
                f"{n_placeholder} optimistic placeholders -- output_token_ids "
                "rollback would drop the wrong elements"
            )
            del req_state.output_token_ids[-n_placeholder:]
            req_state.output_token_ids.extend(sampled_ids)

            if req_id not in self.input_batch.req_id_to_index:
                continue

            req_idx = self.input_batch.req_id_to_index[req_id]

            # num_tokens_no_spec was advanced to (seq_len + n_placeholder)
            # optimistically; roll back the over-count (= num_rejected) so it
            # reflects the real committed length (start_idx + len(sampled_ids)).
            end_idx = self.input_batch.num_tokens_no_spec[req_idx]
            start_idx = end_idx - n_placeholder
            self.input_batch.num_tokens_no_spec[req_idx] = start_idx + len(sampled_ids)

            target_slice = slice(seq_len - len(sampled_ids) + 1, seq_len + 1)
            # The committed tokens must fit within token_ids_cpu
            # ([num_reqs, max_model_len]); writing past the end is a sizing bug.
            assert seq_len + 1 <= self.input_batch.token_ids_cpu.shape[1], (
                f"req {req_id}: write end {seq_len + 1} exceeds max_model_len "
                f"{self.input_batch.token_ids_cpu.shape[1]}"
            )
            self.input_batch.token_ids_cpu[req_idx, target_slice] = sampled_ids

    def _update_placeholder(
        self,
        discard_sampled_tokens_req_indices,
        request_seq_lens,
        next_token_indices: dict[int, int],
        num_draft_per_req: dict[int, int] | None = None,
    ):
        placeholder_req_id_to_index: dict[str, int] = {}
        discard_set = set(discard_sampled_tokens_req_indices)
        for req_idx, req_state, seq_len, req_id in request_seq_lens:
            if req_idx in discard_set:
                continue

            # Async spec: optimistically advance by 1 (bonus) + num_draft;
            # the over-count is corrected on-device next step by
            # subtract_num_rejected_tokens.
            n_new = 1 + (num_draft_per_req.get(req_idx, 0) if num_draft_per_req else 0)
            end_idx = seq_len + n_new
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_state.output_token_ids.extend([0] * n_new)

            next_token_index = next_token_indices[req_idx]
            placeholder_req_id_to_index[req_state.req_id] = next_token_index

        return placeholder_req_id_to_index

    def _sync_replicated_drafts_across_tp(
        self, drafts: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Pin a replicated drafter's device proposal to rank 0's on every rank.

        Broadcast emulated as mask + all_reduce (tpu_dist has no broadcast);
        tiny (num_reqs x K int32), fully on device. No-ops for a sharded
        drafter, TP=1, the sync host-list form, and empty steps.
        """
        if drafts is None or not isinstance(drafts, torch.Tensor):
            return drafts
        if (
            self.speculative_config is None
            or self.speculative_config.draft_tensor_parallel_size != 1
        ):
            return drafts
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        if get_tensor_model_parallel_world_size() <= 1:
            return drafts
        if self._tp_rank0_mask is None:
            self._tp_rank0_mask = torch.tensor(
                1 if get_tensor_model_parallel_rank() == 0 else 0,
                dtype=drafts.dtype,
                device=drafts.device,
            )
        return self._tpu_pin_drafts_to_rank0(drafts, self._tp_rank0_mask)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _tpu_pin_drafts_to_rank0(
        self, drafts: torch.Tensor, rank0_mask: torch.Tensor
    ) -> torch.Tensor:
        # Must be compiled (eager collectives wedge the TPU cores), and the
        # rank must enter as runtime data (the 0/1 mask), not a trace-time
        # constant: XLA pairs collectives by channel id, so every rank must
        # compile the identical program.
        from vllm.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(drafts * rank0_mask)

    def _assemble_async_spec_substitution(self, drafts, next_tokens_per_chunk, state):
        """Async-spec substitution assembler.

        From the already-proposed on-device ``drafts`` (the unified eagle3
        propose call in ``sample_tokens``), builds the per-request
        ``[bonus, draft_1..K]`` substitution source and the stride-``1+K``
        ``next_token_indices``, and computes the per-request rejected count.
        Returns ``(next_tokens_tpu_chunks, next_token_indices,
        spec_decode_num_rejected_tokens, num_draft_per_req)``.
        """
        from vllm_torchtpu.runner.tpu_runner_async_output import (
            assemble_spec_next_tokens,
            compute_num_rejected,
        )

        assert drafts is not None, (
            "async spec substitution requires the eagle3 drafts proposed "
            "earlier in sample_tokens"
        )
        K = self.speculative_config.num_speculative_tokens
        next_tokens_tpu_chunks: list[torch.Tensor] = []
        next_token_indices: dict[int, int] = {}
        num_rejected_chunks: list[torch.Tensor] = []
        num_draft_per_req: dict[int, int] = {}
        req_offset = 0
        # Uniform `1+K` source span per request: req position `p` maps to source
        # `[p*(1+K) .. p*(1+K)+K]`, and `req_id_to_index_copy` stores `p` (the
        # index builder multiplies by the stride). Mixed-batch non-spec chunks
        # pad their single token to the same `1+K` width.
        for nt_chunk, num_reqs, md in zip(
            next_tokens_per_chunk, state.num_reqs_list, state.spec_decode_metadata_list
        ):
            if md is not None:
                drafts_chunk = drafts[req_offset : req_offset + num_reqs]
                src = assemble_spec_next_tokens(
                    nt_chunk, drafts_chunk, num_reqs
                )  # [num_reqs*(1+K)]
                num_rejected_chunks.append(
                    compute_num_rejected(nt_chunk, md.draft_lengths[:num_reqs])
                )
                for r in range(num_reqs):
                    num_draft_per_req[req_offset + r] = int(md.draft_lengths_cpu[r])
            else:
                # Mixed-batch non-spec chunk: pad the single sampled token to the
                # uniform 1+K span (token at slot 0, rest INVALID).
                token = nt_chunk[:num_reqs, 0]
                src = torch.full(
                    (num_reqs, 1 + K),
                    INVALID_TOKEN_ID,
                    dtype=token.dtype,
                    device=token.device,
                )
                src[:, 0] = token
                src = src.reshape(-1)  # [num_reqs*(1+K)]
                num_rejected_chunks.append(
                    torch.zeros(num_reqs, dtype=torch.int32, device=token.device)
                )
            next_tokens_tpu_chunks.append(src)
            for r in range(num_reqs):
                next_token_indices[req_offset + r] = req_offset + r
            req_offset += num_reqs
        spec_num_rejected = (
            torch.cat(num_rejected_chunks) if num_rejected_chunks else None
        )
        return (
            next_tokens_tpu_chunks,
            next_token_indices,
            spec_num_rejected,
            num_draft_per_req,
        )

    def _assemble_async_prefill_bootstrap(
        self, drafts, combined_selected_tokens, combined_real_lens
    ):
        """Async eagle3 bootstrap assembler for the prefill / pure-non-spec step.

        From the already-proposed prompt-context ``drafts`` (the unified eagle3
        propose call in ``sample_tokens``, device_seed path), parks a
        stride-(1+K) `[bonus, draft_1..K]` source so the NEXT step substitutes
        REAL prompt-context drafts. Returns ``(next_tokens_tpu_chunks,
        next_token_indices, spec_num_rejected=zeros, num_draft_per_req)``.
        """
        assert drafts is not None, (
            "async prefill bootstrap requires the eagle3 drafts proposed "
            "earlier in sample_tokens"
        )
        K = self.speculative_config.num_speculative_tokens
        next_tokens_tpu_chunks: list[torch.Tensor] = []
        next_token_indices: dict[int, int] = {}
        num_rejected_chunks: list[torch.Tensor] = []
        num_draft_per_req: dict[int, int] = {}
        req_offset = 0
        for sel, n in zip(combined_selected_tokens, combined_real_lens):
            bonus = sel.view(-1)[:n].unsqueeze(1)  # [n, 1]
            drafts_chunk = drafts[req_offset : req_offset + n]  # [n, K]
            src = torch.cat(
                [bonus.to(drafts_chunk.dtype), drafts_chunk], dim=1
            )  # [n, 1+K] = [bonus, draft_1..K]
            next_tokens_tpu_chunks.append(src.reshape(-1))
            num_rejected_chunks.append(
                torch.zeros(n, dtype=torch.int32, device=src.device)
            )
            for r in range(n):
                next_token_indices[req_offset + r] = req_offset + r
                num_draft_per_req[req_offset + r] = K
            req_offset += n
        spec_num_rejected = (
            torch.cat(num_rejected_chunks) if num_rejected_chunks else None
        )
        return (
            next_tokens_tpu_chunks,
            next_token_indices,
            spec_num_rejected,
            num_draft_per_req,
        )

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        start_index: int,
        num_decode_reqs: int,
        num_windowed_reqs: int | None = None,
    ):
        if num_windowed_reqs is None:
            num_windowed_reqs = num_decode_reqs
        rpa_num_decode_reqs = (
            num_windowed_reqs if self.reorder_batch_threshold > 1 else num_decode_reqs
        )
        assert scheduler_output.total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        assert start_index < num_reqs
        self._wait_input_staging_fence()

        # Get the number of scheduled tokens for each request.
        use_max_model_len = self.most_model_len is None
        num_scheduled_tokens_per_req = []
        max_num_scheduled_tokens_all_reqs = 0
        end_index = start_index

        # Use either most_model_len or max_model_len depending on request size.
        for i in range(start_index, num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if (
                not use_max_model_len
                and self.most_model_len is not None
                and num_tokens > self.most_model_len
            ):
                use_max_model_len = True
            num_scheduled_tokens_per_req.append(num_tokens)
        if use_max_model_len:
            if len(num_scheduled_tokens_per_req) > self.num_reqs_max_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[
                    : self.num_reqs_max_model_len
                ]
                end_index = start_index + self.num_reqs_max_model_len
            else:
                end_index = num_reqs
        else:
            assert self.num_reqs_most_model_len is not None
            if len(num_scheduled_tokens_per_req) > self.num_reqs_most_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[
                    : self.num_reqs_most_model_len
                ]
                end_index = start_index + self.num_reqs_most_model_len
            else:
                end_index = num_reqs
        max_num_scheduled_tokens_all_reqs = max(num_scheduled_tokens_per_req)
        num_scheduled_tokens_per_req = np.array(
            num_scheduled_tokens_per_req, dtype=np.int32
        )
        total_num_scheduled_tokens = sum(num_scheduled_tokens_per_req)
        assert max_num_scheduled_tokens_all_reqs > 0

        num_reqs = len(num_scheduled_tokens_per_req)

        sequence_layout_planner = _get_sequence_layout_planner_for_runner(self)
        sequence_layout_planner.reserve_host_token_capacity(
            self, int(total_num_scheduled_tokens)
        )

        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Fast path for decode-only: all requests have exactly 1 token.
        if max_num_scheduled_tokens_all_reqs == 1:
            # Pure decode: each request schedules exactly 1 token.
            # req_indices = [0, 1, ..., num_reqs-1]
            req_indices = self.arange_np[start_index : start_index + num_reqs]
            # positions = num_computed_tokens for each request
            positions_np = self.positions_np[:num_reqs]
            np.copyto(
                positions_np,
                self.input_batch.num_computed_tokens_cpu[
                    start_index : start_index + num_reqs
                ],
            )
            # token_indices = positions + req_index * max_model_len
            token_indices = (
                positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
            )
            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(token_indices),
                out=self.input_ids_cpu[:num_reqs],
            )
        else:
            # General path: mixed prefill + decode.
            # Get request indices.
            req_indices = np.repeat(
                self.arange_np[start_index : start_index + num_reqs],
                num_scheduled_tokens_per_req,
            )

            # Get batched arange.
            # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            arange = np.concatenate(
                [self.arange_np[:n] for n in num_scheduled_tokens_per_req]
            )

            # Get positions.
            positions_np = self.positions_np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                arange,
                out=positions_np,
            )

            # Get token indices.
            token_indices = (
                positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
            )

            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(token_indices),
                out=self.input_ids_cpu[:total_num_scheduled_tokens],
            )

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(
            num_scheduled_tokens_per_req, out=self.query_start_loc_np[1 : num_reqs + 1]
        )
        # Keep padded entries equal to the last valid location so padded
        # requests have zero length instead of a negative q_len.
        self.query_start_loc_np[num_reqs + 1 :] = self.query_start_loc_np[num_reqs]

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[
                start_index : start_index + num_reqs
            ]
            + num_scheduled_tokens_per_req
        )
        self._check_attention_schedule(
            num_reqs,
            num_scheduled_tokens_per_req,
            max(0, min(rpa_num_decode_reqs - start_index, num_reqs)),
            use_max_model_len,
        )

        request_major_input_ids_cpu: torch.Tensor
        if self._pcp_enabled and self._mtp_enabled:
            # prepare_real() may repack input_ids_cpu in place. Preserve only
            # the real request-major prefix before crossing that boundary.
            request_major_input_ids_cpu = self.input_ids_cpu.narrow(
                0, 0, int(total_num_scheduled_tokens)
            ).clone()

        if use_max_model_len:
            target_num_reqs = self.num_reqs_max_model_len
        else:
            assert self.num_reqs_most_model_len is not None
            target_num_reqs = self.num_reqs_most_model_len
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs
        )

        layout_plan = sequence_layout_planner.prepare_real(
            runner=self,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
            total_num_scheduled_tokens=int(total_num_scheduled_tokens),
            use_max_model_len=use_max_model_len,
            target_num_reqs=target_num_reqs,
            padded_num_reqs=padded_num_reqs,
        )
        self._last_sequence_layout_plan = layout_plan
        padded_total_num_scheduled_tokens = layout_plan.global_padded_num_tokens
        request_major_input_ids = None
        if self._pcp_enabled and self._mtp_enabled:
            request_major_input_ids_cpu_padded = torch.zeros(
                layout_plan.global_padded_num_tokens,
                dtype=request_major_input_ids_cpu.dtype,
            )
            request_major_input_ids_cpu_padded.narrow(
                0, 0, layout_plan.global_num_tokens
            ).copy_(request_major_input_ids_cpu)
            request_major_input_ids = request_major_input_ids_cpu_padded.to(
                self.device, non_blocking=True
            )
        local_total_num_scheduled_tokens = layout_plan.local_num_tokens
        local_padded_total_num_scheduled_tokens = layout_plan.local_padded_num_tokens
        local_token_slice = layout_plan.token_slice
        if layout_plan.kind is SequenceLayoutKind.ALL:
            # Zero out to avoid spurious values from prev iteration.
            self.input_ids_cpu[
                total_num_scheduled_tokens:padded_total_num_scheduled_tokens
            ] = 0
        self.input_ids = self.input_ids_cpu[local_token_slice].to(
            self.device, non_blocking=True
        )
        if self.uses_mrope:
            if layout_plan.kind is SequenceLayoutKind.ALL:
                self.mrope_positions.cpu[
                    :, total_num_scheduled_tokens:padded_total_num_scheduled_tokens
                ] = 0
            self.position_ids = self.mrope_positions.cpu[:, local_token_slice].to(
                self.device, non_blocking=True
            )
        else:
            self.position_ids = self.positions_cpu[local_token_slice].to(
                self.device, non_blocking=True
            )
        if use_max_model_len:
            seq_lens = self.seq_lens_cpu[: self.num_reqs_max_model_len].to(
                self.device, non_blocking=True
            )
            target_num_reqs = self.num_reqs_max_model_len
        else:
            assert self.num_reqs_most_model_len is not None
            seq_lens = self.seq_lens_cpu[: self.num_reqs_most_model_len].to(
                self.device, non_blocking=True
            )
            target_num_reqs = self.num_reqs_most_model_len

        # Async spec: seq_lens/positions were advanced optimistically (every
        # draft from the previous step assumed accepted). Subtract the real
        # per-request rejected count on-device, keyed by req position.
        if (
            self.scheduler_config.async_scheduling
            and self.speculative_config
            and self._pre_async_results is not None
            and self._pre_async_results.spec_decode_num_rejected_tokens is not None
        ):
            from vllm_torchtpu.runner.tpu_runner_async_output import (
                subtract_num_rejected_tokens,
            )

            seq_idx_np = np.full(target_num_reqs, -1, dtype=np.int32)
            pos_idx_np = np.full(padded_total_num_scheduled_tokens, -1, dtype=np.int32)
            acc = 0
            for i in range(num_reqs):
                req_id = self.input_batch.req_ids[start_index + i]
                n_sched = int(num_scheduled_tokens_per_req[i])
                pos = self._pre_async_results.req_id_to_index_copy.get(req_id)
                if pos is not None:
                    seq_idx_np[i] = pos
                    pos_idx_np[acc : acc + n_sched] = pos
                acc += n_sched
            seq_lens, self.position_ids = subtract_num_rejected_tokens(
                seq_lens,
                self.position_ids,
                self._pre_async_results.spec_decode_num_rejected_tokens,
                torch.from_numpy(seq_idx_np).to(self.device, non_blocking=True),
                torch.from_numpy(pos_idx_np).to(self.device, non_blocking=True),
            )

        # For decode-only case, cache constant device tensors to skip H2D.
        # query_start_loc, logits_indices, and request_distribution don't
        # change between decode steps for a given (num_reqs, padded_num_reqs).
        is_decode_only = max_num_scheduled_tokens_all_reqs == 1
        decode_cache_key = (
            num_reqs,
            padded_num_reqs,
            use_max_model_len,
            layout_plan.descriptor.cache_key,
        )

        # Speculative decoding metadata.
        spec_decode_metadata = None
        if self.speculative_config:
            num_draft_tokens = np.array(
                [
                    len(scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
                    for req_id in self.input_batch.req_ids[
                        start_index : start_index + num_reqs
                    ]
                ],
                dtype=np.int32,
            )
            if num_draft_tokens.any():
                spec_decode_metadata = (
                    self.spec_decode_manager.get_spec_decode_metadata(
                        num_draft_tokens,
                        self.query_start_loc_np[1 : num_reqs + 1],
                        padded_num_reqs,
                    )
                )

        # Default decode layouts keep logits_indices stable for a fixed
        # (num_reqs, padded_num_reqs) bucket. PCP streaming layouts provide
        # explicit rank-major logits indices that can change as q_start crosses
        # an interleave/rank boundary, so they must not reuse a stale cached
        # logits_indices tensor.
        can_cache_decode_metadata = (
            is_decode_only
            and spec_decode_metadata is None
            and layout_plan.logits_indices_cpu is None
        )

        if (
            can_cache_decode_metadata
            and decode_cache_key == self._decode_device_cache_key
        ):
            # Reuse cached device tensors — skip 3 H2D copies.
            query_start_loc = self._cached_query_start_loc
            logits_indices = self._cached_logits_indices
            request_distribution = self._cached_request_distribution
        else:
            # Compute and copy to device (first call or shape changed).
            if use_max_model_len:
                query_start_loc = self.query_start_loc_cpu[
                    : self.num_reqs_max_model_len + 1
                ].to(self.device, non_blocking=True)
            else:
                query_start_loc = self.query_start_loc_cpu[
                    : self.num_reqs_most_model_len + 1
                ].to(self.device, non_blocking=True)

            if spec_decode_metadata is not None:
                logits_indices = spec_decode_metadata.final_logits_indices
            elif layout_plan.logits_indices_cpu is not None:
                logits_indices = layout_plan.logits_indices_cpu.to(
                    self.device, non_blocking=True
                )
            else:
                # Indices at which we sample (positions of last token in the
                # sequence). Padded to avoid recompiling when `num_reqs` varies.
                logits_indices = (
                    self.query_start_loc_cpu[1 : padded_num_reqs + 1] - 1
                ).to(self.device, non_blocking=True)

            # For the V3 kernel, request_distribution is
            # [decode_end, prefill_end, mixed_end]. We put decode requests first,
            # no dedicated prefill-only bucket, and all remaining requests in
            # mixed mode.
            chunk_num_decode = max(0, min(rpa_num_decode_reqs - start_index, num_reqs))
            self._request_distribution_cpu[0] = chunk_num_decode
            self._request_distribution_cpu[1] = chunk_num_decode
            self._request_distribution_cpu[2] = num_reqs
            request_distribution = self._request_distribution_cpu.to(
                self.device, non_blocking=True
            )

            if can_cache_decode_metadata:
                # Cache for future decode steps.
                self._decode_device_cache_key = decode_cache_key
                self._cached_query_start_loc = query_start_loc
                self._cached_logits_indices = logits_indices
                self._cached_request_distribution = request_distribution

        # Compact-mamba: per-request recurrent-slot ids for this chunk. Only
        # built for hybrid models (slot pool initialized in
        # initialize_kv_cache); None otherwise so non-mamba models are
        # unaffected.
        mamba_state_indices = (
            self._build_mamba_state_indices(start_index, num_reqs, target_num_reqs)
            if self._has_mamba_state
            else None
        )

        # Spec decode with mamba layers: the GDN kernel's windowed segment
        # covers 1-token decodes AND speculative verify windows (the batch is
        # ordered [decode][verify][prefill/mixed] by _reorder_batch_for_rpa),
        # while RPA includes verify only when its decode tile supports it.
        mamba_request_distribution = None
        if self.mamba_slot_read_offsets is not None:
            chunk_num_windowed = max(0, min(num_windowed_reqs - start_index, num_reqs))
            chunk_num_decode = max(0, min(rpa_num_decode_reqs - start_index, num_reqs))
            # One H2D copy, two views (see the staging tensor's comment for
            # why these must not be independent device tensors). Overrides
            # the RPA tensor built above so both fields always come from the
            # same base tensor, matching the traced graph structure.
            combined = self._combined_request_distribution_cpu
            combined[0] = chunk_num_decode
            combined[1] = chunk_num_decode
            combined[2] = num_reqs
            combined[3] = chunk_num_windowed
            combined[4] = chunk_num_windowed
            combined[5] = num_reqs
            if getattr(self, "_fast_token_substitution", False):
                chunk_comb_key = (chunk_num_decode, chunk_num_windowed, num_reqs)
                if (
                    not hasattr(self, "_cached_comb_dev")
                    or getattr(self, "_cached_comb_key", None) != chunk_comb_key
                ):
                    self._cached_comb_dev = combined.to(self.device, non_blocking=True)
                    self._cached_comb_key = chunk_comb_key
                combined_device = self._cached_comb_dev
            else:
                combined_device = combined.to(self.device, non_blocking=True)
            request_distribution = combined_device[0:3]
            mamba_request_distribution = combined_device[3:6]

        self._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=start_index,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            mamba_state_indices=mamba_state_indices,
            mamba_slot_read_offsets=self.mamba_slot_read_offsets,
            mamba_ckpt_window=self._mamba_ckpt_window,
            mamba_request_distribution=mamba_request_distribution,
            # Unified pool + spec decode: the mamba group builders append
            # their per-request state block ids here for the post-sampling
            # read-offset scatter.
            unified_mamba_state_indices=(
                []
                if self._unified_kv_layout and self.mamba_slot_read_offsets is not None
                else None
            ),
            sequence_layout_descriptor=layout_plan.descriptor,
        )
        slot_mappings = self.empty_slot_mappings
        per_layer_attn_metadata, _unused_spec_decode_common_attn_metadata = (
            self._build_attention_metadata(
                num_tokens=local_total_num_scheduled_tokens,
                num_reqs=target_num_reqs,
                max_query_len=max_num_scheduled_tokens_all_reqs,
                num_tokens_padded=local_padded_total_num_scheduled_tokens,
                num_reqs_padded=target_num_reqs,
                slot_mappings=slot_mappings,
            )
        )

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA adapters
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copying to avoid accidental state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += (
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens
            )

            self.set_active_loras(self.input_batch, padded_num_scheduled_tokens_per_req)

        # Prepare token substitution indices
        cur_input_indices, pre_next_tokens_indices = (
            self._prepare_async_token_substitution_indices(
                start_index, num_reqs, num_scheduled_tokens_per_req
            )
        )

        self._collect_mamba_state_seed_copies(scheduler_output, start_index, num_reqs)

        return (
            per_layer_attn_metadata,
            logits_indices,
            padded_num_reqs,
            num_reqs,
            end_index,
            cur_input_indices,
            pre_next_tokens_indices,
            spec_decode_metadata,
            request_major_input_ids,
            layout_plan,
        )

    def _record_input_staging_fence(self) -> None:
        """Fence the uploads issued from the reusable host staging tensors.

        Every `.to(self.device, non_blocking=True)` of a staging tensor keeps
        reading its host memory until the transfer completes, and the same
        tensors are rewritten by the next chunk's input preparation. The
        event is recorded before the forward is dispatched, so waiting on it
        never waits for the forward itself.
        """
        fence = torch.tpu.Event()
        fence.record()
        self._input_staging_fence = fence

    def _wait_input_staging_fence(self) -> None:
        fence = self._input_staging_fence
        if fence is not None:
            fence.synchronize()
            self._input_staging_fence = None

    def _execute_mm_encoder(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> list[torch.Tensor]:
        """Run the vision encoder with optional video frame chunking.

        A long video encoded in one shot makes the vision attention kernel
        process every `t*h*w` patch in a single forward, which can exhaust
        VMEM. `MM_ENCODER_FRAME_CHUNK_PATCH_SIZE` bounds each forward to that
        many patches by splitting the clip on the temporal axis (bit-exact, see
        `vllm_torchtpu.runner.mm_video_chunking`). Everything else — batching,
        multimodal LoRA, `prompt_embeds`, the encoder cudagraph path and the
        encoder cache — stays on the base `GPUModelRunner` implementation.

        Args:
            scheduler_output: Scheduler output containing the scheduled batch
                and multimodal inputs.

        Returns:
            List of multimodal embedding tensors for inputs processed by the
            encoder.
        """
        with chunked_video_encoding(self.model, self.model_config):
            return super()._execute_mm_encoder(scheduler_output)

    def _get_model_inputs(
        self,
        input_ids: torch.Tensor,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ):
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            # _gather_mm_embeddings returns is_mm_embed sized to the number
            # of scheduled tokens, but TPU pads input_ids to a fixed bucket.
            # Pad the multimodal mask (False for pad positions) and move it
            # to the input device so the masked_scatter in embed_input_ids
            # aligns with the padded inputs_embeds.
            if is_mm_embed is not None:
                n = input_ids.shape[0]
                if is_mm_embed.shape[0] < n:
                    pad = torch.zeros(
                        n - is_mm_embed.shape[0],
                        dtype=is_mm_embed.dtype,
                        device=is_mm_embed.device,
                    )
                    is_mm_embed = torch.cat([is_mm_embed, pad])
                is_mm_embed = is_mm_embed.to(input_ids.device)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            return None, inputs_embeds
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            return input_ids, None

    def _dp_lockstep_enabled(self) -> bool:
        # EP combine reduces partial expert outputs across DP engines; those
        # ranks must enter the same collectives with matching token buckets.
        # Non-EP DP engines remain independently schedulable.
        return (
            utils.get_dp_size(self.parallel_config) > 1
            and self.parallel_config.enable_expert_parallel
        )

    def _count_input_chunks(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[int, int]:
        """Match _prepare_inputs chunking without staging any tensors.

        Returns (num_chunks, max_chunk_reqs). The second value is the exact
        maximum request count of any chunk — a most-model-len chunk holds up
        to num_reqs_most_model_len requests, which can exceed
        num_reqs_max_model_len, so no batch-level clamp is a safe substitute.
        """
        if scheduler_output.total_num_scheduled_tokens == 0:
            return 0, 0

        num_reqs = self.input_batch.num_reqs
        start_index = 0
        num_chunks = 0
        max_chunk_reqs = 0
        while start_index < num_reqs:
            use_max_model_len = self.most_model_len is None
            for i in range(start_index, num_reqs):
                req_id = self.input_batch.req_ids[i]
                assert req_id is not None
                num_tokens = scheduler_output.num_scheduled_tokens[req_id]
                if (
                    not use_max_model_len
                    and self.most_model_len is not None
                    and num_tokens > self.most_model_len
                ):
                    use_max_model_len = True

            if use_max_model_len:
                chunk_reqs = self.num_reqs_max_model_len
            else:
                assert self.num_reqs_most_model_len is not None
                chunk_reqs = self.num_reqs_most_model_len
            this_chunk_reqs = min(chunk_reqs, num_reqs - start_index)
            max_chunk_reqs = max(max_chunk_reqs, this_chunk_reqs)
            start_index += this_chunk_reqs
            num_chunks += 1
        return num_chunks, max_chunk_reqs

    def _dp_coordinated_step(
        self, local_num_tokens: int, local_num_chunks: int, local_num_reqs: int = 0
    ) -> "tuple[int | None, int]":
        """Return the DP-wide padded token bucket and forward chunk count."""
        if not self._dp_lockstep_enabled():
            return None, local_num_chunks
        from vllm.distributed.parallel_state import get_dp_group

        # Token/chunk metadata is synchronized on vLLM's CPU DP group. The
        # model forward below receives this result explicitly as
        # num_tokens_across_dp, so set_forward_context does not run a second
        # DP synchronization.
        t = torch.tensor(
            [int(local_num_tokens), int(local_num_chunks), int(local_num_reqs)],
            dtype=torch.int64,
            device="cpu",
        )
        torch.distributed.all_reduce(
            t, op=torch.distributed.ReduceOp.MAX, group=get_dp_group().cpu_group
        )
        bucket = _get_padded_token_len(self.num_tokens_paddings, int(t[0].item()))
        # Spec-decode draft forwards size themselves off the step's DP-wide
        # max per-chunk request count (see DFlashProposer), riding this same
        # all-reduce instead of paying the theoretical max bucket or a
        # second collective.
        self._dp_step_max_reqs = int(t[2].item())
        return bucket, int(t[1].item())

    def _run_dp_dummy_chunk(self, bucket: int) -> None:
        self._dummy_run(
            bucket,
            self.num_reqs_max_model_len,
            self.max_num_blocks_per_req,
            use_max_model_len=True,
            dp_lockstep=True,
        )

    def _run_dp_idle_pairing(self, bucket: int, num_chunks: int) -> None:
        """Idle-rank EP-DP pairing: emit the same cross-DP collective stream a
        busy rank does — ALL target dummy forwards, then ALL draft dummy
        forwards.
        """
        # Set before any dummy fires: both _run_dp_dummy_chunk and the draft
        # run_dp_dummy_draft read _dp_target_bucket for the coordinated,
        # equal-token shape.
        self._dp_target_bucket = bucket
        # Phase 1: all TARGET dummy forwards.
        for _ in range(num_chunks):
            self._run_dp_dummy_chunk(bucket)
        # Phase 2: all DRAFT dummy forwards (num_chunks * K).
        # use_eagle() covers every hidden-state drafter, dflash included.
        spec = self.speculative_config
        if num_chunks > 0 and spec is not None and spec.use_eagle():
            self.drafter.run_dp_dummy_draft(num_chunks)

    def _dp_num_tokens_across_dp(self, num_tokens: int) -> torch.Tensor | None:
        if not self._dp_lockstep_enabled():
            return None
        return torch.full(
            (utils.get_dp_size(self.parallel_config),),
            num_tokens,
            dtype=torch.int32,
            device="cpu",
        )

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | None:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )
        # Update cached state
        self._update_states(scheduler_output)
        if self.scheduler_config.async_scheduling:
            self._flush_disjoint_async_results()
        if scheduler_output.total_num_scheduled_tokens == 0:
            if not has_kv_transfer_group():
                return EMPTY_MODEL_RUNNER_OUTPUT
            # Even for a zero-token step and if DP lockstep is enabled, we must
            # still execute the KV coordination logic so the background ZMQ/SHM
            # threads can start KV transfer in disaggregated serving. This is
            # safe to do for a zero-token step since no collectives are
            # triggered.
            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)
        if self._pcp_enabled and self._mtp_enabled:
            # Validate the actual scheduled span before any forward/collective.
            # Ending exactly at the prompt boundary produces the first output;
            # consuming any generated token would enter unsupported decode or
            # target verification. Let the engine's fatal-error path handle it.
            for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
                req_index = self.input_batch.req_id_to_index[req_id]
                num_computed = int(self.input_batch.num_computed_tokens_cpu[req_index])
                num_prompt = int(self.input_batch.num_prompt_tokens[req_index])
                if num_computed + num_tokens > num_prompt:
                    raise RuntimeError(
                        "PCP + MTP only supports prefill and the first output "
                        "token; refusing decode/verify for "
                        f"request {req_id!r}: num_computed_tokens={num_computed}, "
                        f"num_scheduled_tokens={num_tokens}, "
                        f"num_prompt_tokens={num_prompt}. "
                        "Set SamplingParams(max_tokens=1) at the service entry."
                    )
        # Run the multimodal (vision) encoder. vLLM's base
        # GPUModelRunner.execute_model does this; our override must do it
        # explicitly, otherwise the encoder never runs, image placeholder
        # tokens get plain text embeddings, and the model produces garbage.
        if self.supports_mm_inputs and self._pp_is_first:
            has_encoder_inputs = bool(scheduler_output.scheduled_encoder_inputs)
            if has_encoder_inputs:
                synchronize_tensors()
            self._execute_mm_encoder(scheduler_output)
            if has_encoder_inputs:
                synchronize_tensors()

        num_decode_reqs, num_windowed_reqs = self._reorder_batch_for_rpa(
            scheduler_output
        )

        # Profile the current batch composition (prefill vs decode) if phased profiling
        # is enabled.
        if self.phase_based_profiler:
            self.batch_counter += 1
            padded_total_tokens = _get_padded_token_len(
                self.num_tokens_paddings, scheduler_output.total_num_scheduled_tokens
            )
            batch_composition_stats = runner_utils.get_batch_composition_stats(
                batch_id=self.batch_counter,
                input_batch=self.input_batch,
                total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
                num_reqs=self.input_batch.num_reqs,
                padded_total_num_scheduled_tokens=padded_total_tokens,
                scheduler_output=scheduler_output,
            )
            self.phase_based_profiler.step(batch_composition_stats)

        # Gather mm embeddings AFTER reordering so the mask order matches the
        # request order used by the chunk loop below. is_mm_embed_full spans
        # all scheduled tokens (req order); mm_embeds_flat is the concatenated
        # image embeddings. We slice both per request-chunk inside the loop.
        mm_embeds_flat = None
        is_mm_embed_full = None
        mm_tok_cumsum = None
        mm_cumsum_np = None
        if self.supports_mm_inputs and self._pp_is_first:
            mm_embeds_list, is_mm_embed_full = self._gather_mm_embeddings(
                scheduler_output
            )
            if mm_embeds_list:
                mm_embeds_flat = torch.cat(mm_embeds_list)
            req_tok = np.array(
                [
                    scheduler_output.num_scheduled_tokens[r]
                    for r in self.input_batch.req_ids
                ],
                dtype=np.int64,
            )
            mm_tok_cumsum = np.concatenate([[0], np.cumsum(req_tok)])
            # Per-token MM offsets on the host so the chunk loop can slice
            # mm_embeds without any device syncs -- mirrors the GPU runner,
            # which passes the full mask to embed_input_ids and lets
            # masked_scatter place the embeds (no manual counting).
            if mm_embeds_flat is not None:
                mm_cumsum_np = np.concatenate(
                    [[0], np.cumsum(is_mm_embed_full.cpu().numpy())]
                )

        local_num_chunks, local_max_chunk_reqs = self._count_input_chunks(
            scheduler_output
        )
        self._dp_target_bucket, target_num_chunks = self._dp_coordinated_step(
            scheduler_output.total_num_scheduled_tokens,
            local_num_chunks,
            # Exact per-chunk request bound on this rank; the DP-wide MAX of
            # it sizes every rank's draft forwards.
            local_num_reqs=local_max_chunk_reqs,
        )
        # Retained for the spec-decode propose phase (sample_tokens): under
        # EP-DP locksteps every rank must run the same number of draft forwards per
        # step, and the coordinated chunk count from this one all-reduce is the shared
        # bound — no additional draft-side collective is needed.
        self._dp_step_num_chunks = target_num_chunks

        start_index = 0
        chunk_index = 0
        logits_list = []
        pooler_output_list = []
        num_reqs_list = []
        spec_decode_metadata_list = []
        mamba_state_indices_list: list[list[torch.Tensor] | None] = []
        draft_chunks: list[DraftChunkInputs] = []
        is_draft_model = self._is_async_drafter

        # NOTE: setup current batch's metadata for kv connector.
        # Verified with TPURaidenConnector, TPUConnector, OffloadingConnector
        with set_forward_context(None, self.vllm_config):
            # Raiden overlap: block the worker main thread on the KV load only
            # in inline mode. In the default async mode the load (network pull
            # + DMA H2D into the cache) runs entirely on the Raiden C++
            # threads while this step's forward computes; the request stays in
            # WAITING_FOR_REMOTE_KVS until every rank reports done_recving.
            raiden_inline = dist_utils.get_raiden_inline_load()
            self.maybe_setup_kv_connector(
                scheduler_output,
                wait_for_completion=raiden_inline,
                report_completion=not raiden_inline,
            )

        while chunk_index < target_num_chunks:
            if start_index >= self.input_batch.num_reqs:
                assert self._dp_target_bucket is not None
                self._run_dp_dummy_chunk(self._dp_target_bucket)
                chunk_index += 1
                continue

            (
                attn_metadata,
                logits_indices,
                padded_num_reqs,
                num_reqs,
                end_index,
                cur_input_indices,
                pre_next_tokens_indices,
                spec_decode_metadata,
                request_major_input_ids,
                draft_sequence_layout_plan,
            ) = self._prepare_inputs(
                scheduler_output, start_index, num_decode_reqs, num_windowed_reqs
            )

            # Seed newly-advanced mamba state blocks before the forward reads
            # them (chunk boundaries, decode crossings, prefix-cache resumes).
            self._flush_mamba_state_seed_copies()

            # A recycled slot/block still carries the previous request's read
            # offset; clear it before this chunk's forward gathers it.
            self._reset_read_offsets_for_new_requests(start_index, num_reqs)

            input_ids = self._apply_async_token_substitution(
                self.input_ids, cur_input_indices, pre_next_tokens_indices
            )
            # Async spec: the drafts the target verifies live in the substituted
            # input_ids (host input_ids_cpu holds placeholders), so re-source the
            # rejection sampler's draft_token_ids from the device.
            if (
                self.scheduler_config.async_scheduling
                and spec_decode_metadata is not None
                and len(cur_input_indices) > 0
            ):
                from vllm_torchtpu.runner.tpu_runner_async_output import (
                    extract_draft_token_ids,
                )

                spec_decode_metadata.draft_token_ids = extract_draft_token_ids(
                    input_ids,
                    spec_decode_metadata.final_logits_indices,
                    spec_decode_metadata.target_logits_indices,
                )
            draft_input_ids_src = input_ids

            # Slice the per-chunk multimodal mask/embeddings (the chunk
            # covers requests [start_index, end_index) in req order). Offsets
            # come from the host-side cumsum, so no per-chunk device sync.
            chunk_mm_inputs = None
            if is_mm_embed_full is not None:
                tok_start = int(mm_tok_cumsum[start_index])
                tok_end = int(mm_tok_cumsum[end_index])
                is_mm_chunk = is_mm_embed_full[tok_start:tok_end]
                if mm_cumsum_np is not None:
                    mm_start = int(mm_cumsum_np[tok_start])
                    mm_cnt = int(mm_cumsum_np[tok_end]) - mm_start
                    chunk_embeds = (
                        [mm_embeds_flat[mm_start : mm_start + mm_cnt]]
                        if mm_cnt > 0
                        else []
                    )
                else:
                    chunk_embeds = []
                chunk_mm_inputs = (chunk_embeds, is_mm_chunk)

            self._record_input_staging_fence()
            if self._pp_is_first:
                input_ids, inputs_embeds = self._get_model_inputs(
                    input_ids, chunk_mm_inputs
                )
            else:
                # Later stages start from the previous stage's hidden
                # states; the token ids only size the forward.
                inputs_embeds = None
            # Run the decoder
            # set_forward_context: vLLM's native context for attention metadata
            # set_vllm_model_wrapper_context: TPU-specific context for mesh info
            # For multimodal models _get_model_inputs returns input_ids=None
            # and packs the tokens into inputs_embeds; use whichever exists.
            num_tokens_padded = (
                input_ids if input_ids is not None else inputs_embeds
            ).shape[0]
            self._token_padding_update(num_tokens_padded)
            intermediate_tensors = None
            if self._pp_wave is not None:
                intermediate_tensors = self._pp_wave.start_forward(num_tokens_padded)
                self._pp_take_topk_indices(intermediate_tensors, num_tokens_padded)

            trace_kwargs = {}
            if TraceAnnotation.is_enabled():
                trace_kwargs = extract_request_ids_for_tracing(
                    self.input_batch.req_ids, start_index, num_reqs
                )
                trace_kwargs.update(
                    extract_kv_lens_for_tracing(
                        self.input_batch.num_computed_tokens_cpu, start_index, num_reqs
                    )
                )

            if self.phase_based_profiler and "batch_composition_stats" in locals():
                stats_map = {
                    "num_prefill_tokens": "num_prefill_tokens",
                    "num_decode_tokens": "num_decode_tokens",
                    "phase": "phase",
                    "batch_id": "batch_id",
                    "total_num_scheduled_tokens": "total_num_scheduled_tokens",
                    "padded_total_num_scheduled_tokens": "padded_total_num_scheduled_tokens",  # noqa: E501
                    "min_kv_len": "min_kv_length",
                }
                for src_key, target_key in stats_map.items():
                    trace_kwargs[target_key] = batch_composition_stats.get(
                        src_key, "UNKNOWN" if src_key == "phase" else 0
                    )

            trace_name = f"ModelForward: {num_reqs} reqs, {num_tokens_padded} toks"
            if "min_kv_len" in trace_kwargs:
                trace_name += (
                    f", kv_len min={trace_kwargs['min_kv_len']} "
                    f"avg={trace_kwargs['avg_kv_len']} "
                    f"max={trace_kwargs['max_kv_len']}"
                )

            with (
                TraceAnnotation(
                    name=trace_name,
                    num_reqs=num_reqs,
                    **trace_kwargs,
                ),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=self._dp_num_tokens_across_dp(
                        num_tokens_padded
                    ),
                ),
                set_vllm_model_wrapper_context(
                    mesh=self.mesh,
                    vllm_config=self.vllm_config,
                    kv_cache_bundle=self._kv_cache_bundle,
                ),
            ):
                hidden_states, aux_hidden_states = self.forward_model(
                    input_ids=input_ids,
                    positions=self.position_ids,
                    inputs_embeds=inputs_embeds,
                    intermediate_tensors=intermediate_tensors,
                )

            if not self._pp_is_last:
                # The next stage continues this chunk; logits and sampling
                # happen on the last stage.
                self._pp_wave.end_forward(
                    self._pp_outgoing_tensors(hidden_states.tensors, num_tokens_padded),
                    num_tokens_padded,
                )
                start_index = end_index
                chunk_index += 1
                continue
            if self._pp_wave is not None:
                self._pp_wave.end_forward(None, num_tokens_padded)

            sequence_layout_planner = _get_sequence_layout_planner_for_runner(self)
            layout_plan = self._last_sequence_layout_plan
            sample_hidden_states = (
                sequence_layout_planner.maybe_select_logits_hidden_states(
                    hidden_states, layout_plan, logits_indices
                )
            )
            if sample_hidden_states is None:
                hidden_states = sequence_layout_planner.finalize_hidden_states(
                    hidden_states,
                    layout_plan,
                )
                logits = self.compute_selected_logits(hidden_states, logits_indices)
            else:
                logits = self.compute_logits_from_hidden_states(sample_hidden_states)

            if self.is_pooling_model:
                pooling_metadata = self.input_batch.get_pooling_metadata()
                # num_scheduled_tokens_np, seq_lens_cpu for this chunk
                req_ids = self.input_batch.req_ids[start_index : start_index + num_reqs]
                num_scheduled_tokens_np = np.array(
                    [
                        scheduler_output.num_scheduled_tokens[req_id]
                        for req_id in req_ids
                    ],
                    dtype=np.int32,
                )
                seq_lens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
                    start_index : start_index + num_reqs
                ]
                pooling_metadata.build_pooling_cursor(
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    seq_lens_cpu=seq_lens_cpu,
                    device=self.device,
                )
                pooler_output = self.model.pooler(
                    hidden_states=hidden_states,
                    pooling_metadata=pooling_metadata,
                )
                pooler_output_list.append(pooler_output)

            logits_list.append(logits)
            num_reqs_list.append(num_reqs)
            spec_decode_metadata_list.append(spec_decode_metadata)
            ctx = self._attn_metadata_builder_ctx
            if self.mamba_slot_read_offsets is None:
                # No mamba layers, or no spec decoding: nothing to roll back.
                mamba_state_indices_list.append(None)
            elif ctx.mamba_state_indices is not None:
                # Compact pool: one shared slot id per request across groups.
                mamba_state_indices_list.append([ctx.mamba_state_indices])
            else:
                # Unified pool: one state block per request per mamba group,
                # captured by the group builders during metadata build.
                mamba_state_indices_list.append(ctx.unified_mamba_state_indices)
            if is_draft_model:
                # Capture this chunk's draft inputs while they are valid.
                draft_chunk = DraftChunkInputs(
                    input_ids=(
                        request_major_input_ids
                        if request_major_input_ids is not None
                        else draft_input_ids_src
                    ),
                    position_ids=self.position_ids,
                    query_start_loc_np=self.query_start_loc_np[: num_reqs + 1].copy(),
                    attn_ctx=self._attn_metadata_builder_ctx,
                    start_index=start_index,
                    num_reqs=num_reqs,
                    aux_hidden_states=(
                        aux_hidden_states if aux_hidden_states is not None else []
                    ),
                    hidden_states=hidden_states,
                    draft_lengths=(
                        spec_decode_metadata.draft_lengths
                        if spec_decode_metadata is not None
                        else None
                    ),
                    attn_metadata=attn_metadata,
                    sequence_layout_plan=draft_sequence_layout_plan,
                )
                draft_chunks.append(draft_chunk)

            start_index = end_index
            chunk_index += 1

        if not self._pp_is_last:
            self._pp_pending_scheduler_output = scheduler_output
            return None
        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            logits_list=logits_list,
            pooler_output_list=pooler_output_list,
            num_reqs_list=num_reqs_list,
            spec_decode_metadata_list=spec_decode_metadata_list,
            draft_chunks=draft_chunks,
            mamba_state_indices_list=mamba_state_indices_list,
        )

        if self.is_pooling_model:
            return self.sample_tokens(None)
        return None

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if self.spec_decode_manager is None:
            return None
        return self.spec_decode_manager.take_draft_token_ids()

    def _get_sampling_generator(self) -> torch.Generator:
        """Return the dedicated non-greedy sampling generator, seeded once.

        Seeded lazily on first use from ``model_config.seed`` -- the same value
        on every worker -- then advanced naturally by the draws (never reseeded
        per step). A replica's TP ranks run in lockstep (same ``SchedulerOutput``
        -> same bucketed draw shapes each step), so with the same seed they stay
        at the same offset in the stream and sample identical tokens -- the
        property spec decode needs to keep the TP ranks from diverging. (DP
        replicas share the same base sequence but draw at their own pace on
        independent request streams, so they decorrelate, which is fine.)
        """
        if self._sampling_generator is None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(self.model_config.seed or 0))
            self._sampling_generator = gen
        return self._sampling_generator

    @torch.no_grad()
    def _update_mamba_slot_read_offsets(
        self,
        indices_per_group: list[torch.Tensor] | None,
        next_tokens: torch.Tensor | None,
        num_reqs: int,
    ) -> None:
        """Scatter this chunk's mamba read offsets into the slot-indexed buffer.

        Each entry holds one mamba group's base slot per batch position for
        the chunk (padded tail = slot 0, the null block, so its writes are
        harmless); all groups get the same offsets, since the buffer is
        indexed by slot/block id and groups allocate distinct ones. The
        compact pool shares one slot tensor across groups and so passes a
        single-element list.

        The offset is `num_accepted - 1` derived from the rejection-sampler
        output for verify chunks, and 0 for non-spec chunks (prefill / plain
        decode) — which also resets a group's offset after prefill. The next
        step's GDN kernel resumes each request from checkpoint `offset` of
        its state block (the checkpoint of its last accepted token).
        """
        if self.mamba_slot_read_offsets is None or not indices_per_group:
            return
        offsets = torch.zeros(
            indices_per_group[0].shape[0],
            dtype=torch.int32,
            device=indices_per_group[0].device,
        )
        if next_tokens is not None:
            # num_valid = accepted drafts + 1 (bonus); the checkpoint of the
            # last accepted token is at offset num_valid - 1.
            num_valid = (
                (next_tokens[:num_reqs] != INVALID_TOKEN_ID).sum(dim=1).to(torch.int32)
            )
            offsets[:num_reqs] = (num_valid - 1).clamp(min=0)
        if len(indices_per_group) == 1:
            self.mamba_slot_read_offsets.index_put_(
                (indices_per_group[0].long(),), offsets
            )
        else:
            all_indices = torch.cat(indices_per_group, dim=0).long()
            all_offsets = offsets.repeat(len(indices_per_group))
            self.mamba_slot_read_offsets.index_put_((all_indices,), all_offsets)

    def _mamba_state_index_groups(self) -> list[torch.Tensor] | None:
        """This chunk's mamba state slot/block ids, one tensor per group.

        The compact pool shares one slot tensor across every mamba group, so
        it yields a single-element list; the unified pool allocates a
        distinct state block per group and the group builders collect them
        during the metadata build.
        """
        ctx = self._attn_metadata_builder_ctx
        if self.mamba_slot_read_offsets is None or ctx is None:
            return None
        if ctx.mamba_state_indices is not None:
            return [ctx.mamba_state_indices]
        return ctx.unified_mamba_state_indices

    @torch.no_grad()
    def _reset_read_offsets_for_new_requests(
        self, start_index: int, num_reqs: int
    ) -> None:
        """Zero the read offset of every request entering the batch.

        `mamba_slot_read_offsets` is indexed by slot/block id, but a slot
        outlives the request that wrote it. A request that leaves the batch
        mid-verify-window leaves `num_accepted - 1` behind, and both layouts
        hand that storage straight to the next request:

          * compact -- `_build_mamba_state_indices` returns the slot to
            `_free_mamba_slots` and pops it for a new req_id;
          * unified -- vLLM's block manager frees the state block and
            reallocates it.

        Nothing else clears the entry. `_update_mamba_slot_read_offsets`
        only covers requests already in the batch and runs *after* the
        forward, and the seed-copy migration is doubly gated (align mode
        and the unified layout), so it never runs for a compact/`none`
        deployment at all. The new owner's first forward would then resume
        from a checkpoint belonging to the previous conversation — silent
        GDN state corruption, not a crash.

        The scatter is issued UNCONDITIONALLY, every step, whether or not
        any request is new. Under EP-DP lockstep every rank must run the
        same device program, and which requests are new is rank-local: an
        early return on "nothing to reset" makes one rank skip device work
        its peers perform, they fall out of step, and the next collective
        deadlocks. The mask makes an all-old batch a no-op instead —
        already-seen requests and padded positions are redirected to slot 0,
        the null block, whose offset is 0 regardless.
        """
        if self.mamba_slot_read_offsets is None:
            return
        groups = self._mamba_state_index_groups()
        if not groups:
            return
        req_ids = self.input_batch.req_ids
        # Forget requests that have left, so a later req_id reusing the same
        # string is still treated as new.
        self._mamba_offset_seeded &= set(self.input_batch.req_id_to_index.keys())

        width = groups[0].shape[0]
        keep, keep_dev, null = self._read_offset_reset_scratch(width, groups[0].device)
        keep.zero_()
        for i in range(min(num_reqs, max(0, len(req_ids) - start_index))):
            req_id = req_ids[start_index + i]
            if req_id is None or i >= width:
                continue
            if req_id not in self._mamba_offset_seeded:
                self._mamba_offset_seeded.add(req_id)
                keep[i] = True
        # No `if nothing_new: return` here — see the docstring. The device
        # work below must be issued on every rank on every step.
        keep_dev.copy_(keep, non_blocking=True)
        # Stack groups and reset offsets across all groups in one compiled program.
        stacked = torch.stack(groups) if len(groups) > 1 else groups[0].unsqueeze(0)
        _reset_read_offsets(self.mamba_slot_read_offsets, keep_dev, stacked, null)

    def _read_offset_reset_scratch(
        self, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reusable `(keep_cpu, keep_dev, null)` of length `width`.

        `_reset_read_offsets_for_new_requests` runs on every step, so its
        working set is allocated once per width rather than per step. Width
        is the padded batch bucket, so this holds one entry per bucket —
        a handful for the life of the process. `null` is a 0-d zero that
        broadcasts in the `torch.where`, replacing a full-width
        `zeros_like` per mamba group per step.
        """
        assert self.mamba_slot_read_offsets is not None
        key = (width, str(device))
        entry = self._read_offset_scratch.get(key)
        if entry is None:
            entry = (
                torch.zeros(width, dtype=torch.bool),
                torch.zeros(width, dtype=torch.bool, device=device),
                torch.zeros((), dtype=torch.int32, device=device),
            )
            self._read_offset_scratch[key] = entry
        return entry

    def _greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """argmax logits, only used when `all_greedy` is True."""
        dummy = torch.empty((1, 1), dtype=logits.dtype, device=logits.device)
        return self.sample_from_logits_func(
            logits,
            dummy,
            dummy,
            torch.empty((1, 1), dtype=torch.int32, device=logits.device),
            torch.empty((1, 1), dtype=torch.float32, device=logits.device),
            all_greedy=True,
        )

    def _sample_spec_verify_chunk(
        self,
        logits: torch.Tensor,
        md: SpecDecodeMetadata,
        grammar_output: "GrammarOutput | None",
        scheduler_output: "SchedulerOutput",
        cur_start_idx: int,
        cur_end_idx: int,
        all_greedy: bool,
        sampling_generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Bonus-token sampling + draft rejection for one spec-verify chunk.

        Returns the rejection sampler's [num_reqs, 1 + num_spec_tokens]
        selected tokens for the chunk.
        """
        if all_greedy and grammar_output is None:
            bonus_token_ids, target_logits = self.spec_bonus_and_target_logits(
                logits, md.bonus_logits_indices, md.target_logits_indices
            )
        else:
            bonus_logits, target_logits = self.spec_gather_bonus_and_target_logits(
                logits, md.bonus_logits_indices, md.target_logits_indices
            )
            if grammar_output is not None:
                target_logits, bonus_logits = (
                    self.structured_decoding_manager.mask_spec_logits(
                        target_logits,
                        bonus_logits,
                        grammar_output,
                        scheduler_output.scheduled_spec_decode_tokens,
                        md.draft_lengths_cpu,
                        cur_start_idx,
                        cur_end_idx,
                    )
                )
            if all_greedy:
                bonus_token_ids = self._greedy_sample(bonus_logits).view(-1)
            else:
                req_temperatures, req_top_k, req_top_p = (
                    self._build_padded_sampling_params(
                        cur_start_idx, cur_end_idx, bonus_logits
                    )
                )
                bonus_u = torch.rand_like(bonus_logits, generator=sampling_generator)
                bonus_token_ids = self.sample_from_logits_func(
                    bonus_logits,
                    req_temperatures,
                    bonus_u,
                    req_top_k,
                    req_top_p,
                    all_greedy=False,
                ).view(-1)
        accept_u = None
        if self.rejection_sampler.synthetic_mode or not all_greedy:
            accept_u = torch.rand(
                md.draft_token_ids.shape,
                dtype=torch.float32,
                device=target_logits.device,
                generator=sampling_generator,
            )
        if all_greedy:
            return self.rejection_sampler(
                draft_token_ids=md.draft_token_ids,
                num_draft_tokens=md.draft_lengths,
                target_logits=target_logits,
                bonus_token_ids=bonus_token_ids,
                segment_ids=md.segment_ids,
                group_indices=md.group_indices,
                max_draft_tokens=self.speculative_config.num_speculative_tokens,
                accept_u=accept_u,
            )
        assert accept_u is not None
        recover_u = torch.rand_like(
            target_logits, dtype=torch.float32, generator=sampling_generator
        )
        return self.rejection_sampler(
            draft_token_ids=md.draft_token_ids,
            num_draft_tokens=md.draft_lengths,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            segment_ids=md.segment_ids,
            group_indices=md.group_indices,
            max_draft_tokens=self.speculative_config.num_speculative_tokens,
            temperatures=req_temperatures[md.segment_ids],
            top_k=req_top_k[md.segment_ids],
            top_p=req_top_p[md.segment_ids],
            accept_u=accept_u,
            recover_u=recover_u,
            do_sampling=True,
        )

    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if self.execute_model_state is None:
            if not self._pp_is_last:
                return self._sample_tokens_pp_intermediate_stage()
            # Nothing to do, output isn't used.
            return None  # type: ignore[return-value]

        state = self.execute_model_state
        scheduler_output = state.scheduler_output
        self.execute_model_state = None

        # Used to keep rand draws consistent across TP ranks.
        sampling_generator = self._get_sampling_generator()

        # Hand the per-chunk draft inputs to the drafter.
        if self._is_async_drafter:
            self.drafter.draft_chunks = state.draft_chunks

        # Prepare inputs, the requests might be split into multiple
        # executions, combine the result of each execution.

        max_num_logprobs = self.input_batch.max_num_logprobs
        if max_num_logprobs == -1:
            raise NotImplementedError(
                "TPU runner does not support full logprobs (`logprobs=-1`) "
                "with the merged vLLM v1 sampler path yet."
            )
        needs_logprobs = max_num_logprobs is not None

        # Per-chunk bucketed (padded) tensors plus their real (unpadded)
        # request counts; kept at bucketed shapes so the async D2H copy hits
        # a precompiled program. Host-side trim happens in AsyncTPUCopyState.
        combined_selected_tokens: list[torch.Tensor] = []
        combined_selected_tokens_real_lens: list[int] = []
        next_tokens_tpu_chunks: list[torch.Tensor] = []
        next_token_indices: dict[int, int] = {}
        next_tokens_tpu_offset = 0
        combined_logprobs: list[Any] = []

        # True if the step has drafts to verify (a request carried drafts in
        # from the previous step). Not the same as self.speculative_config, the
        # static "spec configured" flag: on prefill/first-decode steps spec is
        # configured but no drafts exist yet, so is_spec_step is False there.
        is_spec_step = any(md is not None for md in state.spec_decode_metadata_list)
        # Per-chunk mamba slot ids (hybrid + spec decode only; None entries
        # otherwise) for the read-offset scatter after sampling.
        mamba_idx_list = state.mamba_state_indices_list or [None] * len(
            state.logits_list
        )
        if self.is_pooling_model:
            req_ids = cast(
                list[str], self.input_batch.req_ids[: self.input_batch.num_reqs]
            )
        elif is_spec_step:
            if needs_logprobs:
                raise NotImplementedError(
                    "Logprobs are not supported with speculative decoding on TPU yet."
                )
            # Per-chunk device rejection outputs, kept for the async-spec
            # producer below.
            next_tokens_per_chunk: list[torch.Tensor] = []
            cur_start_idx = 0
            all_greedy = self.input_batch.all_greedy
            for logits, num_reqs, md, mamba_indices in zip(
                state.logits_list,
                state.num_reqs_list,
                state.spec_decode_metadata_list,
                mamba_idx_list,
            ):
                cur_end_idx = cur_start_idx + num_reqs
                if md is not None:
                    next_tokens = self._sample_spec_verify_chunk(
                        logits,
                        md,
                        grammar_output,
                        scheduler_output,
                        cur_start_idx,
                        cur_end_idx,
                        all_greedy,
                        sampling_generator,
                    )
                    combined_selected_tokens.append(next_tokens)
                    combined_selected_tokens_real_lens.append(num_reqs)
                    next_tokens_per_chunk.append(next_tokens)
                    self._update_mamba_slot_read_offsets(
                        mamba_indices, next_tokens, num_reqs
                    )
                else:
                    if grammar_output is not None:
                        _, logits = self.structured_decoding_manager.mask_spec_logits(
                            None,
                            logits,
                            grammar_output,
                            scheduler_output.scheduled_spec_decode_tokens,
                            None,
                            cur_start_idx,
                            cur_end_idx,
                        )
                    if all_greedy:
                        selected = self._greedy_sample(logits)
                    else:
                        temperatures_tpu, top_k_tpu, top_p_tpu = (
                            self._build_padded_sampling_params(
                                cur_start_idx, cur_end_idx, logits
                            )
                        )
                        u = torch.rand_like(logits, generator=sampling_generator)
                        selected = self.sample_from_logits_func(
                            logits,
                            temperatures_tpu,
                            u,
                            top_k_tpu,
                            top_p_tpu,
                            all_greedy=False,
                        )
                    padded = torch.full(
                        (
                            selected.shape[0],
                            self.speculative_config.num_speculative_tokens + 1,
                        ),
                        INVALID_TOKEN_ID,
                        dtype=selected.dtype,
                        device=selected.device,
                    )
                    padded[:, 0] = selected.view(-1)
                    combined_selected_tokens.append(padded)
                    combined_selected_tokens_real_lens.append(num_reqs)
                    next_tokens_per_chunk.append(padded)
                    # Non-verify chunk in a spec step: reset the read offsets
                    # (the latest state checkpoint is at the group base).
                    self._update_mamba_slot_read_offsets(mamba_indices, None, num_reqs)
                self._update_num_xla_graphs("spec_step")
                cur_start_idx = cur_end_idx
        else:
            cur_start_idx = 0
            req_ids = cast(
                list[str], self.input_batch.req_ids[: self.input_batch.num_reqs]
            )
            all_greedy = self.input_batch.all_greedy
            for logits, num_reqs, mamba_indices in zip(
                state.logits_list, state.num_reqs_list, mamba_idx_list
            ):
                cur_end_idx = cur_start_idx + num_reqs
                # Non-spec step (prefill / plain decode): reset the mamba
                # read offsets so the next step resumes from the group base.
                self._update_mamba_slot_read_offsets(mamba_indices, None, num_reqs)
                if grammar_output is not None:
                    logits = self.structured_decoding_manager.mask_logits(
                        logits, grammar_output, cur_start_idx, cur_end_idx
                    )
                if all_greedy:
                    selected_token_ids = self._greedy_sample(logits)
                else:
                    temperatures_tpu, top_k_tpu, top_p_tpu = (
                        self._build_padded_sampling_params(
                            cur_start_idx, cur_end_idx, logits
                        )
                    )
                    u = torch.rand_like(logits, generator=sampling_generator)
                    selected_token_ids = self.sample_from_logits_func(
                        logits,
                        temperatures_tpu,
                        u,
                        top_k_tpu,
                        top_p_tpu,
                        all_greedy=all_greedy,
                    )
                # NOTE (NickLucche) Use the original logits (before any penalties or
                # temperature scaling) for the top-k logprobs. We can't enforce it
                # due to recompilations outside torch.compiled code, so just make
                # sure `sample_from_logits` does not modify the logits in-place.
                logprobs = (
                    self.gather_logprobs(logits, selected_token_ids)
                    if needs_logprobs
                    else None
                )

                # Keep the bucketed (padded) tensor; trim happens on the host
                # inside AsyncTPUCopyState to avoid per-`num_reqs` recompiles.
                combined_selected_tokens.append(selected_token_ids)
                combined_selected_tokens_real_lens.append(num_reqs)
                if self.scheduler_config.async_scheduling:
                    next_tokens_tpu_chunks.append(selected_token_ids.view(-1))
                    for req_idx in range(cur_start_idx, cur_end_idx):
                        next_token_indices[req_idx] = (
                            next_tokens_tpu_offset + req_idx - cur_start_idx
                        )
                    next_tokens_tpu_offset += selected_token_ids.shape[0]
                if needs_logprobs:
                    sliced_logprobs = LogprobsTensors(
                        logprobs.logprob_token_ids[:num_reqs],
                        logprobs.logprobs[:num_reqs],
                        logprobs.selected_token_ranks[:num_reqs],
                        logprobs.cu_num_generated_tokens,
                    )
                    combined_logprobs.append(sliced_logprobs)

                self._update_num_xla_graphs("decoding_step")
                cur_start_idx = cur_end_idx

        # NOTE: current kv load and save get h2d/d2h copies involved.
        # Those copies are blocking. Once they become async., kv_save
        # should be called right after each single forward pass,
        # instead of the forwards of the entire input batch.
        logprobs = []
        if needs_logprobs and len(combined_logprobs):
            # TODO: concatenate the LogprobsTensors (torch) first, then
            # call .tolists() once to match the GPU path and avoid the
            # per-chunk numpy round-trip.
            combined_logprobs_lists = []
            for i in range(len(combined_logprobs)):
                logprobs = combined_logprobs[i].tolists()
                combined_logprobs_lists.append(logprobs)

            logprobs_lists = LogprobsLists(
                logprob_token_ids=np.concatenate(
                    [lp.logprob_token_ids for lp in combined_logprobs_lists]
                ),
                logprobs=np.concatenate(
                    [lp.logprobs for lp in combined_logprobs_lists]
                ),
                sampled_token_ranks=np.concatenate(
                    [lp.sampled_token_ranks for lp in combined_logprobs_lists]
                ),
            )
            logprobs = logprobs_lists

        discard_sampled_tokens_req_indices = []
        num_reqs = self.input_batch.num_reqs
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        request_seq_lens = []
        for i, req_id in enumerate(req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (
                req_state.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[req_id]
            )
            # Ignore the sampled token from the partial request.
            # Rewind the generator state as if the token was not sampled.
            if seq_len < req_state.num_tokens:
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)
            else:
                request_seq_lens.append((i, req_state, seq_len, req_id))

        next_tokens_tpu = None
        copy_state = None
        if combined_selected_tokens:
            copy_state = AsyncTPUCopyState.from_device_chunks(
                combined_selected_tokens, combined_selected_tokens_real_lens
            )

        if self.scheduler_config.async_scheduling:
            self._modify_prev_results()

        # Poll the KV connector only now: the wait above proves the previous
        # forward finished, so a store deferred behind it can be handed to
        # raiden in this step rather than the next.
        self.maybe_wait_for_kv_save()
        (
            finished_sending,
            finished_recving,
            kv_worker_meta,
            invalid_block_ids,
            invalid_block_group_index,
            kv_connector_stats,
        ) = self.get_finished_kv_transfers(scheduler_output)

        kv_connector_output = (
            None
            if (
                finished_sending is None
                and finished_recving is None
                and kv_worker_meta is None
                and not invalid_block_ids
                and kv_connector_stats is None
            )
            else _build_kv_connector_output(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                kv_connector_worker_meta=kv_worker_meta,
                invalid_block_ids=invalid_block_ids,
                invalid_block_group_index=invalid_block_group_index,
                kv_connector_stats=kv_connector_stats,
            )
        )

        # Unified eagle3 draft propose -- ONE call serving both sync and async
        # (mirrors the tpu-inference reference, where a single
        # propose_draft_token_ids call runs every step and an async bool only
        # picks the return form). Seeds on-device from this step's rejection
        # output (spec step) or the just-sampled tokens (non-spec / prefill
        # step), so neither mode waits on a host copy of the sampled tokens.
        # Async gets the raw [num_reqs, K] device tensor back (consumed by the
        # substitution assemblers below); sync caches the host list in the
        # manager for take_draft_token_ids(). ngram has no device path and
        # keeps proposing from the host-materialized output in the sync block
        # below.
        is_async = self.scheduler_config.async_scheduling
        eagle3_drafts = None
        if self._is_async_drafter and combined_selected_tokens:
            if is_spec_step:
                eagle3_drafts = self.spec_decode_manager.propose_draft_token_ids(
                    sampled_token_ids=None,
                    discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
                    num_rejected_tokens_np=None,
                    scheduler_output=scheduler_output,
                    return_device=is_async,
                    next_tokens_per_chunk=next_tokens_per_chunk,
                )
            else:
                # Seed = the just-sampled tokens, kept on-device (no D2H):
                # propose takes the sync seed path but reads the seed from
                # `device_seed`.
                device_seed = torch.cat(
                    [
                        sel.view(-1)[:n]
                        for sel, n in zip(
                            combined_selected_tokens, combined_selected_tokens_real_lens
                        )
                    ]
                )
                eagle3_drafts = self.spec_decode_manager.propose_draft_token_ids(
                    sampled_token_ids=[],
                    discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
                    num_rejected_tokens_np=None,
                    scheduler_output=scheduler_output,
                    return_device=is_async,
                    device_seed=device_seed,
                )

        if self.scheduler_config.async_scheduling:
            # A replicated (draft_tp=1) drafter proposes independently per TP
            # rank and the proposals are not bit-identical; async verifies
            # each rank's OWN drafts (no driver round-trip like sync), so the
            # ranks drift apart and the output corrupts. Pin every rank to
            # rank 0's proposal; sharded drafters skip this.
            eagle3_drafts = self._sync_replicated_drafts_across_tp(eagle3_drafts)

            if (
                eagle3_drafts is not None
                and scheduler_output.has_structured_output_requests
            ):
                # The deferred grammar-bitmask will fetch these drafts via
                # take_draft_token_ids() to replace the async scheduler's -1
                # splaceholders. Stage after the TP pin above so the host
                # copy matches what gets substituted on device.
                self.spec_decode_manager.stage_draft_token_ids_for_host(eagle3_drafts)

            # Build the async substitution source from the drafts proposed
            # above: the [bonus, draft_1..K] source + 1+K next_token_indices,
            # and the per-request rejected count to park.
            spec_num_rejected = None
            num_draft_per_req = None
            if is_spec_step:
                (
                    next_tokens_tpu_chunks,
                    next_token_indices,
                    spec_num_rejected,
                    num_draft_per_req,
                ) = self._assemble_async_spec_substitution(
                    eagle3_drafts, next_tokens_per_chunk, state
                )
            elif self._is_async_drafter and next_tokens_tpu_chunks:
                # Prefill / pure-non-spec eagle3 bootstrap: the non-spec sampling
                # branch parked a stride-1 source (sampled token only). Rebuild
                # it as a stride-(1+K) [bonus, draft_1..K] source carrying real
                # prompt-context drafts so the NEXT step's verify gets real
                # drafts instead of placeholders.
                (
                    next_tokens_tpu_chunks,
                    next_token_indices,
                    spec_num_rejected,
                    num_draft_per_req,
                ) = self._assemble_async_prefill_bootstrap(
                    eagle3_drafts,
                    combined_selected_tokens,
                    combined_selected_tokens_real_lens,
                )
            req_id_to_index_copy = {}
            if not self.is_pooling_model:
                req_id_to_index_copy = self._update_placeholder(
                    discard_sampled_tokens_req_indices,
                    request_seq_lens,
                    next_token_indices,
                    num_draft_per_req,
                )
            if next_tokens_tpu_chunks:
                if len(next_tokens_tpu_chunks) == 1:
                    next_tokens_tpu = next_tokens_tpu_chunks[0]
                else:
                    next_tokens_tpu = torch.cat(next_tokens_tpu_chunks, dim=0)
                # Bound the async-spec source length so the torch.compiled
                # _substitute_placeholder_token (next step) sees a fixed shape
                # instead of recompiling per real num_reqs. Append-pad to the
                # max stride-(1+K) span: the substitution only reads
                # [position*stride .. +n_sched-1] for real reqs (positions <
                # num_reqs), so the appended tail is never indexed -- real
                # positions are unchanged. Spec only (stride 1+K); pure-non-spec
                # async keeps its stride-1 source + existing precompile.
                if spec_num_rejected is not None:
                    bound = self.max_num_reqs * (
                        1 + self.speculative_config.num_speculative_tokens
                    )
                    if next_tokens_tpu.shape[0] < bound:
                        next_tokens_tpu = torch.nn.functional.pad(
                            next_tokens_tpu, (0, bound - next_tokens_tpu.shape[0])
                        )
                self._pre_async_results = AsyncPreResults(
                    req_ids=req_ids,
                    next_tokens_tpu=next_tokens_tpu,
                    request_seq_lens=request_seq_lens,
                    discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
                    req_id_to_index_copy=req_id_to_index_copy,
                    copy_state=copy_state,
                    spec_decode_num_rejected_tokens=spec_num_rejected,
                    num_draft_per_req=num_draft_per_req,
                )
            else:
                self._pre_async_results = None

        pooler_output = []
        if self.is_pooling_model:
            for out in state.pooler_output_list:
                # Move to CPU as expected by vLLM
                if isinstance(out, torch.Tensor):
                    pooler_output.extend(list(out.cpu().unbind(dim=0)))
                else:
                    pooler_output.extend(out)

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            # Snapshot of the req_id_to_index for the VLLM scheduler.
            req_id_to_index=dict(self.input_batch.req_id_to_index),
            sampled_token_ids=[],  # Filled in AsyncTPUModelRunnerOutput get_output
            logprobs=logprobs,
            prompt_logprobs_dict={req_id: None for req_id in req_ids},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
        )

        async_output = AsyncTPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            copy_state=copy_state,
            discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
        )

        if not self.scheduler_config.async_scheduling:
            final_output = async_output.get_output()
            if not self.is_pooling_model:
                for i, req_state, seq_len, req_id in request_seq_lens:
                    if i in discard_sampled_tokens_req_indices:
                        continue
                    valid_tokens = final_output.sampled_token_ids[i]
                    if not valid_tokens:
                        continue
                    req_idx = self.input_batch.req_id_to_index[req_id]

                    # Update the persistent batch.
                    start_tok_idx = self.input_batch.num_tokens_no_spec[req_idx]
                    end_tok_idx = start_tok_idx + len(valid_tokens)
                    self.input_batch.token_ids_cpu[
                        req_idx, start_tok_idx:end_tok_idx
                    ] = valid_tokens
                    self.input_batch.num_tokens_no_spec[req_idx] = end_tok_idx

                    req_state.output_token_ids.extend(valid_tokens)

            if self.speculative_config and self.speculative_config.method == "ngram":
                # ngram drafts on the host from the committed token ids, so it
                # must run after materialization; eagle3/dflash were already proposed
                # from device state by the unified call above.
                self.spec_decode_manager.propose_draft_token_ids(
                    sampled_token_ids=final_output.sampled_token_ids,
                    discard_sampled_tokens_req_indices=discard_sampled_tokens_req_indices,
                    num_rejected_tokens_np=None,
                    scheduler_output=scheduler_output,
                )

            return final_output

        return async_output

    def load_model(self) -> None:
        self.device_config.device = torch.device("tpu")
        self.device = self.device_config.device
        self.vllm_config.device_config = self.device_config

        # Set TPU-specific quantization config before model loading.
        # This ensures that TPU-compatible quantization methods are used
        # instead of the default GPU-based ones (e.g., MXFP4 for MoE models).
        # For unquantized models (quantization=None), we still need to apply
        # our TPU config to override vLLM's default UnquantizedFusedMoEMethod
        # which uses torch_xla.
        logger.info(
            "Setting TPU quantization config for: %s", self.model_config.quantization
        )
        self.vllm_config.quant_config = get_tpu_quantization_config(self.vllm_config)

        model_loader = get_model_loader(self.load_config)
        logger.info("Loading model from scratch...")
        with (
            set_vllm_model_wrapper_context(
                mesh=self.mesh, vllm_config=self.vllm_config
            ),
            set_current_vllm_config(self.vllm_config),
        ):
            model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config
            )
        validate_online_fp8(model, self.vllm_config.quant_config)
        self.model = model
        if self.parallel_config.pipeline_parallel_size > 1:
            # A sparse model's layers share one top-k index table, and under a
            # pipeline the layer that writes it and the layers that reuse it
            # can sit on different chips. Hand the table between stages
            # instead.
            self._pp_topk_buffer = _find_topk_indices_buffer(self.model)
            if self._pp_topk_buffer is not None:
                # -1 is "no token": a row of it attends to nothing.
                self._pp_topk_buffer.fill_(-1)
        if envs.TPU_ROPE_CACHE_TRUNCATE:
            self._truncate_rope_caches()
        if envs.TPU_ROPE_CACHE_ROW_MAJOR:
            runner_utils.relayout_rope_caches(self.model)
        if envs.TPU_MOE_HASH_TABLE_ROW_MAJOR:
            runner_utils.relayout_hash_tables(self.model)

        # If using eagle3/mtp or dflash speculative decoding, load the draft model and
        # share the target's embeddings / LM head (if the draft requires it).
        if self._is_async_drafter:
            self.drafter.load_model(self.model)

        # Ensure attention custom ops exist before any compile/inference path,
        self._initialize_pallas_kernels()

        from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import (  # noqa: E501
            VllmCompressedTensorsW4ANMxfp4MoEMethod,
        )

        uses_mxfp4_moe = any(
            isinstance(
                getattr(module, "quant_method", None),
                VllmCompressedTensorsW4ANMxfp4MoEMethod,
            )
            for module in self.model.modules()
        )
        if envs.TPU_MOE_SKIP_PADDED_TOKENS or uses_mxfp4_moe:
            self._token_padding_state = token_padding.TokenPaddingState.create(
                self.max_num_tokens, self.device
            )
            token_padding.set_padding_state(self._token_padding_state)

    def _initialize_pallas_kernels(self):
        self._initialize_attention_kernels()
        self._initialize_quantization_kernels()

    def _initialize_quantization_kernels(self):
        from vllm_torchtpu.layers.adapter.linear_common import (
            _get_quantized_matmul_fp4_op,
            _get_quantized_matmul_op,
        )

        with set_current_vllm_config(self.vllm_config):
            # Pre-warm FP8 quantized-matmul lock; Dynamo can't trace Lock.
            _get_quantized_matmul_op()
            # Same for the NVFP4 W4A16 matmul op.
            _get_quantized_matmul_fp4_op()

    def _initialize_attention_kernels(self, force: bool = False) -> None:
        """Pre-build Pallas RPA and custom attention kernels before torch.compile."""
        if self._attention_kernels_initialized and not force:
            return
        from vllm_torchtpu.layers.adapter.attention import (
            _DRAFT_KV_BLOCK_CAP,
            PallasAttentionBackendImpl,
            _pallas_rpa_kernel_local,
        )

        # Only draft proposers expose _draft_attn_layer_names; NgramProposer
        # operates purely on token IDs. no draft attention layers to relocate.
        spec_drafter = self.drafter
        draft_attn_names = (
            getattr(spec_drafter, "_draft_attn_layer_names", None) or set()
        )

        layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        initialized_count = 0
        decode_query_sizes = []
        pinned_draft_layers = []
        with set_vllm_model_wrapper_context(
            mesh=self.mesh, vllm_config=self.vllm_config
        ):
            for name, attn_layer in layers.items():
                if isinstance(
                    getattr(attn_layer, "impl", None), PallasAttentionBackendImpl
                ):
                    is_draft = name in draft_attn_names
                    pre_w = attn_layer.impl.decode_query_size
                    if is_draft:
                        # A K+1-wide decode tile describes the TARGET's verify
                        # step; a proposer runs one query token per draft step,
                        # so pin it to 1. NOTE: on Gemma-4 the draft layers are
                        # observed to already be 1 here, so this is a guard
                        # rather than the fix for b/536992014.
                        attn_layer.impl.decode_query_size = 1
                        # Relocate a REPLICATED (tp=1) draft's attention to the
                        # LOCAL (non-shard_map) kernel. Independent of the tile
                        # width above: this is about the draft's TP topology.
                        if self.speculative_config.draft_tensor_parallel_size == 1:
                            # Instance attrs shadow the ClassVars; unique prefix
                            # keeps the local kernel op out of the sharded
                            # registry.
                            attn_layer.impl._kernel_entry = _pallas_rpa_kernel_local
                            attn_layer.impl._kernel_op_prefix = (
                                "pallas::rpa_kernel_local"
                            )
                            logger.info(
                                "Draft attn %s -> LOCAL (non-shard_map) RPA "
                                "kernel | DRAFT_KV_BLOCK_CAP=%d",
                                name,
                                _DRAFT_KV_BLOCK_CAP,
                            )
                    attn_layer.impl.initialize_kernel(attn_layer)
                    pinned_draft_layers.append(
                        (
                            "draft" if is_draft else "target",
                            type(attn_layer.impl).__name__,
                            pre_w,
                            attn_layer.impl.decode_query_size,
                            getattr(attn_layer.impl, "head_size", None),
                        )
                    )
                    # Target layers only. `reorder_batch_threshold` is read in
                    # exactly one place (`rpa_num_decode_reqs` in
                    # `_prepare_inputs`), which shapes the TARGET's batch; the
                    # draft never reads it and builds its own pure-decode
                    # distribution instead (eagle3 `_propose_step`). So draft
                    # widths have no business in this min() -- folding the
                    # draft's necessary 1 in would drag the threshold to 1 and
                    # silently disable #1008 for every spec config that has a
                    # draft, which is every config it was written for.
                    if not is_draft:
                        decode_query_sizes.append(attn_layer.impl.decode_query_size)
                    initialized_count += 1

            # Pre-build custom attention, compressor, and indexer kernels
            # (e.g. DeepSeek-V4 SWA/CSA/HCA)
            if hasattr(self, "model") and self.model is not None:
                for module in self.model.modules():
                    if hasattr(module, "_build_attn_op") and hasattr(module, "attn_op"):
                        module.__dict__.pop("_attn_op_instance", None)
                        _ = module.attn_op
                        initialized_count += 1
                    if hasattr(module, "_build_compressor_op") and hasattr(
                        module, "compressor_op"
                    ):
                        module.__dict__.pop("_compressor_op_instance", None)
                        _ = module.compressor_op
                        initialized_count += 1
                    if hasattr(module, "_build_indexer_op") and hasattr(
                        module, "indexer_op"
                    ):
                        _ = module.indexer_op
                        initialized_count += 1
                    for op_name in (
                        "mhc_ops",
                        "mhc_post_op",
                        "qnorm_rope_op",
                        "kv_rope_op",
                        "o_proj_op",
                    ):
                        if hasattr(type(module), op_name):
                            _ = getattr(module, op_name)
                            initialized_count += 1
        self.reorder_batch_threshold = min(decode_query_sizes, default=1)
        # State it, don't infer it: a target-only [4] and an undetected-draft
        # [4] print identically, so the draft count is the only way to tell
        # whether the pinning below actually matched any layer.
        from collections import Counter

        logger.info(
            "Attn layer census (role, impl, width_before_init, "
            "width_after_init, head_size) -> count: %s | draft names known=%d",
            dict(Counter(pinned_draft_layers)),
            len(draft_attn_names),
        )
        logger.info(
            "Pre-built attention/indexing kernels for %d layers/modules. "
            "decode_query_size=%s -> reorder_batch_threshold=%d "
            "(>1 means speculative verify runs in the RPAd decode stage).",
            initialized_count,
            sorted(set(decode_query_sizes)),
            self.reorder_batch_threshold,
        )
        self._attention_kernels_initialized = True

    @torch.no_grad()
    def _dummy_run(
        self,
        num_tokens: int,
        num_reqs: int,
        num_blocks: int,
        use_max_model_len: bool = True,
        dp_lockstep: bool = False,
        profile_prefix: int | None = None,
    ) -> float | None:
        """Run the forward at one bucket shape.

        With ``profile_prefix`` the batch is one prefill request of
        ``num_tokens`` tokens behind that many computed tokens (the same
        program: only the metadata values differ), and the forward's wall
        time in ms is returned once its outputs have materialized.
        """
        kv_cache_initialized = self.kv_cache_config is not None
        profiling = profile_prefix is not None

        # A profile batch carries random values so every token row routes
        # on its own, as real tokens do; a warmup batch traces shapes only.
        if profiling:
            input_ids = self._profile_input("ids", (self.max_num_tokens,), torch.int32)[
                :num_tokens
            ].to(self.device)
            # One prefill request, or several of at most the model length
            # when a bucket exceeds it.
            profile_query_lens = []
            left = num_tokens
            while left > 0:
                profile_query_lens.append(min(left, self.max_model_len))
                left -= profile_query_lens[-1]
            actual_num_reqs = len(profile_query_lens)
            assert actual_num_reqs <= num_reqs, (
                f"profiling {num_tokens} tokens needs {actual_num_reqs} "
                f"requests, the batch holds {num_reqs}"
            )
        else:
            input_ids = torch.zeros((num_tokens), dtype=torch.int32).to(self.device)
            actual_num_reqs = min(num_tokens, num_reqs)
        inputs_embeds = None
        if self.uses_mrope:
            position_ids = torch.zeros((3, num_tokens), dtype=torch.int32).to(
                self.device
            )
        else:
            position_ids = torch.zeros(num_tokens, dtype=torch.int32).to(self.device)
        if profiling:
            query_lens = profile_query_lens + [0] * (num_reqs - actual_num_reqs)
        else:
            query_lens = [1] * actual_num_reqs + [0] * (num_reqs - actual_num_reqs)
        query_start_loc = torch.cumsum(
            torch.tensor([0] + query_lens, dtype=torch.int32), dim=0, dtype=torch.int32
        ).to(self.device)
        seq_lens_cpu = torch.ones((num_reqs,), dtype=torch.int32)
        if profiling:
            for i, q in enumerate(profile_query_lens):
                seq_lens_cpu[i] = q
            seq_lens_cpu[0] += profile_prefix
        seq_lens = seq_lens_cpu.to(self.device)
        # V3: request_distribution = [decode_end, prefill_end, mixed_end].
        # Dummy runs use one scheduled token per active request, so model
        # them as pure decode to match the real single-chip path; a profile
        # run holds mixed requests, as a real prefill chunk does.
        distribution = [0, 0, actual_num_reqs] if profiling else [actual_num_reqs] * 3
        request_distribution = torch.tensor(distribution, dtype=torch.int32).to(
            self.device
        )
        dummy_layout_plan = _get_sequence_layout_planner_for_runner(self).prepare_dummy(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            kv_cache_initialized=kv_cache_initialized,
        )

        if self.kv_cache_config is not None:
            # Dummy compact-mamba slot ids (all null slot 0): the dummy run
            # only traces shapes/HBM, so the recurrent state read/written is
            # never consumed. Shape must match the GDN op's max_reqs
            # (= seq_lens length = num_reqs).
            dummy_mamba_state_indices = (
                torch.zeros((num_reqs,), dtype=torch.int32).to(self.device)
                if self._has_mamba_state
                else None
            )
            # Match _prepare_inputs: the spec-decode mamba fields are set iff
            # the model has mamba layers and spec decoding is enabled, so the
            # dummy runs compile the same GDN program signature — including
            # the two distributions being views of one [6] base tensor (see
            # _combined_request_distribution_cpu for why).
            dummy_mamba_request_distribution = None
            if self.mamba_slot_read_offsets is not None:
                combined_device = torch.tensor(distribution * 2, dtype=torch.int32).to(
                    self.device
                )
                request_distribution = combined_device[0:3]
                dummy_mamba_request_distribution = combined_device[3:6]
            self._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
                num_reqs=num_reqs,
                start_index=0,
                use_max_model_len=use_max_model_len,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                position_ids_override=position_ids,
                mamba_state_indices=dummy_mamba_state_indices,
                mamba_slot_read_offsets=self.mamba_slot_read_offsets,
                mamba_ckpt_window=self._mamba_ckpt_window,
                mamba_request_distribution=dummy_mamba_request_distribution,
                sequence_layout_descriptor=dummy_layout_plan.descriptor,
            )
            slot_mappings = self.empty_slot_mappings
            per_layer_attn_metadata, _unused_spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    max_query_len=1,
                    num_tokens_padded=num_tokens,
                    num_reqs_padded=num_reqs,
                    slot_mappings=slot_mappings,
                )
            )
        else:
            # Pre-init path (called before initialize_kv_cache): use the
            # caller-supplied num_blocks and a single shared metadata.
            if self._attn_layer_names is None:
                self._attn_layer_names = list(
                    get_layers_from_vllm_config(
                        self.vllm_config, (AttentionLayerBase, MambaBase)
                    ).keys()
                )
            block_tables = torch.zeros((num_reqs * num_blocks,), dtype=torch.int32).to(
                self.device
            )
            attn_metadata = AttentionMetadata(
                input_positions=position_ids,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                sequence_layout_kind=(dummy_layout_plan.descriptor.kind.value),
                sequence_layout_protocol=(dummy_layout_plan.descriptor.protocol),
                sequence_layout_version=(dummy_layout_plan.descriptor.version),
            )

            per_layer_attn_metadata = {
                layer_name: attn_metadata for layer_name in self._attn_layer_names
            }

        padding_state = getattr(self, "_token_padding_state", None)
        if padding_state is not None:
            # Every row of a warmup step is padding; every row of a profile
            # step is a real token.
            padding_state.update(num_tokens if profiling else 0, num_tokens)
        intermediate_tensors = None
        if not self._pp_is_first:
            intermediate_tensors = self._pp_intermediate_tensors(
                num_tokens, zeros=not profiling, random=profiling
            )
            self._pp_take_topk_indices(intermediate_tensors, num_tokens)
        with (
            self.maybe_select_dummy_loras(
                self.lora_config, np.array([num_tokens], dtype=np.int32)
            ),
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens if dp_lockstep else 0,
                num_tokens_across_dp=self._dp_num_tokens_across_dp(num_tokens)
                if dp_lockstep
                else None,
            ),
            set_vllm_model_wrapper_context(
                mesh=self.mesh,
                vllm_config=self.vllm_config,
                kv_cache_bundle=self._kv_cache_bundle,
            ),
        ):
            if self.supports_mm_inputs and self._pp_is_first:
                input_ids, inputs_embeds = self._get_model_inputs(input_ids, None)
            if profiling:
                # The inputs reach the device before the clock starts.
                inputs = [t for t in (input_ids, inputs_embeds) if t is not None]
                if intermediate_tensors is not None:
                    inputs += list(intermediate_tensors.tensors.values())
                if self._pp_topk_buffer is not None:
                    inputs.append(self._pp_topk_buffer)
                synchronize_tensors(inputs)
            started = time.perf_counter()
            out, _ = self.forward_model(
                input_ids=input_ids,
                positions=position_ids,
                inputs_embeds=inputs_embeds,
                intermediate_tensors=intermediate_tensors,
            )
            if isinstance(out, IntermediateTensors):
                # A stage before the last produces no logits; nothing
                # consumes its output in a warmup.
                synchronize_tensors(list(out.tensors.values()))
                return (time.perf_counter() - started) * 1000.0 if profiling else None
            if dp_lockstep:
                # Idle DP engines must issue the same TP logits collective as
                # busy engines before the next DP synchronization.
                _idx = torch.zeros(num_reqs, dtype=torch.int32, device=out.device)
                _ = self.compute_selected_logits(out, _idx)
            if not dp_lockstep:
                synchronize_tensors(out)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._hidden_states_dtype = out.dtype
        return elapsed_ms if profiling else None

    def _profile_input(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """Random host tensor for profile runs: token ids over the vocabulary
        or normal activations; built once per name."""
        cached = self._profile_inputs.get(name)
        if cached is None or tuple(cached.shape) != tuple(shape):
            if dtype == torch.int32:
                cached = torch.randint(0, self.vocab_size, shape, dtype=torch.int32)
            else:
                cached = torch.randn(shape, dtype=dtype)
            self._profile_inputs[name] = cached
        return cached

    def _attention_schedule_capacity(
        self, use_max_model_len: bool = True
    ) -> dict[str, tuple[int, int, int]] | None:
        """(pairs, bq, bkv) the batched attention kernel can schedule in
        one step, per kernel mode ("decode", "mixed"), for the sequence and
        page counts the kernel is called with in the context bucket; None
        when no attention layer runs that kernel."""
        if self._attention_runs_batched_kernel is None:
            layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
            impls = [getattr(layer, "impl", None) for layer in layers.values()]
            self._attention_runs_batched_kernel = any(
                impl.runs_batched_rpa_schedule()
                for impl in impls
                if isinstance(impl, PallasAttentionBackendImpl)
            )
        if not self._attention_runs_batched_kernel:
            return None
        num_seqs, pages_per_seq, page_size = self._attention_kernel_shapes(
            use_max_model_len
        )
        capacity = self._attention_capacity.get((num_seqs, pages_per_seq))
        if capacity is None:
            layout = KV_LAYOUT_BY_VLLM_LAYOUT[
                self.cache_config.get_resolved_kv_cache_layout()
            ]
            shared = dict(
                num_q_heads=self.model_config.get_num_attention_heads(
                    self.parallel_config
                ),
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                num_seqs=num_seqs,
                pages_per_seq=pages_per_seq,
                page_size=page_size,
                dtype_q=jax.numpy.dtype(
                    str(self.model_config.dtype).removeprefix("torch.")
                ),
                dtype_kv=jax.numpy.dtype(
                    str(self.kv_cache_dtype).removeprefix("torch.")
                ),
                kv_layout=layout,
            )
            capacity = {
                "decode": rpa_batched.schedule_capacity(
                    mode=rpa_configs.RpaCase.DECODE, **shared
                ),
                "mixed": rpa_batched.schedule_capacity(
                    mode=rpa_configs.RpaCase.MIXED, **shared
                ),
            }
            logger.info(
                "Batched attention schedule capacity per step: decode %s, "
                "mixed %s (pairs, query tile, KV tile) for %d seqs x %d "
                "pages of %d tokens, %d q heads, %d kv heads, head dim %d",
                capacity["decode"],
                capacity["mixed"],
                num_seqs,
                pages_per_seq,
                page_size,
                shared["num_q_heads"],
                shared["num_kv_heads"],
                shared["head_dim"],
            )
            self._attention_capacity[(num_seqs, pages_per_seq)] = capacity
        return capacity

    def _attention_kernel_shapes(self, use_max_model_len: bool) -> tuple[int, int, int]:
        """(sequences, pages per sequence, page size) the batched attention
        kernel is called with in the max- or most-model-len bucket: the
        sequence count the bucket pads to and the block table width it
        slices."""
        page_size = self._attention_kernel_block_size or self.cache_config.block_size
        if use_max_model_len or self.most_model_len is None:
            num_seqs = self.num_reqs_max_model_len
            group_id = self._attention_kv_cache_group_id
            if group_id is not None:
                pages_per_seq = self.input_batch.block_table[
                    group_id
                ].max_num_blocks_per_req
            else:
                pages_per_seq = cdiv(self.max_model_len, page_size)
        else:
            assert self.num_reqs_most_model_len is not None
            num_seqs = self.num_reqs_most_model_len
            pages_per_seq = cdiv(self.most_model_len, page_size)
        return num_seqs, pages_per_seq, page_size

    def _check_attention_schedule(
        self,
        num_reqs: int,
        q_lens: np.ndarray,
        num_decode: int,
        use_max_model_len: bool = True,
    ) -> None:
        """Refuse a step whose attention schedule would overrun the
        batched kernel's SMEM, which halts the TPU core. The first
        ``num_decode`` requests run in the decode kernel, the rest in the
        mixed kernel; each has its own capacity."""
        capacity = self._attention_schedule_capacity(use_max_model_len)
        if capacity is None or num_reqs <= 0:
            return
        from vllm_torchtpu.core.pp_chunks import schedule_pairs

        kv_lens = self.seq_lens_np[:num_reqs]
        for mode, lo, hi in (
            ("decode", 0, num_decode),
            ("mixed", num_decode, num_reqs),
        ):
            if hi <= lo:
                continue
            pairs, bq, bkv = capacity[mode]
            need = schedule_pairs(q_lens[lo:hi], kv_lens[lo:hi], bq, bkv)
            if need > pairs:
                raise RuntimeError(
                    f"This step's {mode} attention needs {need} (query "
                    f"block, KV block) pairs but the batched attention "
                    f"kernel's SMEM schedule holds {pairs} (tiles {bq}x"
                    f"{bkv}); running it would halt the TPU core. Use a "
                    "smaller --max-num-batched-tokens or fewer long-context "
                    "requests per step."
                )

    def _add_pipeline_chunk_buckets(self) -> None:
        """Compile a token bucket at every chunk granule multiple, so a step
        the pipeline chunk scheduler shortens pads to its own size. Runs once
        the KV block size is final."""
        from vllm_torchtpu.core.pp_chunk_scheduler import uses_dynamic_chunks
        from vllm_torchtpu.core.pp_chunks import chunk_buckets, chunk_granularity

        if (
            not uses_dynamic_chunks(self.vllm_config)
            or self._pp_chunk_granularity is not None
        ):
            return
        block_aligned = (
            self._has_mamba_state and self.cache_config.mamba_cache_mode == "align"
        )
        granularity = chunk_granularity(
            self.cache_config.block_size, self.max_num_tokens, block_aligned
        )
        self._pp_chunk_granularity = granularity
        extra = sorted(
            set(chunk_buckets(self.max_num_tokens, granularity))
            - set(self.num_tokens_paddings)
        )
        if extra:
            self.num_tokens_paddings = sorted(
                set(self.num_tokens_paddings) | set(extra)
            )
            self.vllm_config.compilation_config.compile_sizes = list(
                self.num_tokens_paddings
            )
        logger.info(
            "Pipeline chunks: granularity %d tokens (KV block %d, block "
            "aligned %s), buckets added %s",
            granularity,
            self.cache_config.block_size,
            block_aligned,
            extra,
        )

    def profile_pipeline_chunks(self) -> dict[str, Any]:
        """Time this stage's forward at the points the pipeline chunk
        scheduler fits its step cost model from. Returns the chunk
        granularity and (tokens, prefix, ms) samples."""
        from vllm_torchtpu.core.pp_chunks import chunk_pairs, profile_points

        capacity = self._attention_schedule_capacity()
        schedule = None if capacity is None else capacity["mixed"]
        if schedule is not None:
            need = chunk_pairs(schedule, self.max_num_tokens, 0)
            if need > schedule[0]:
                raise RuntimeError(
                    f"A full step of {self.max_num_tokens} tokens from one "
                    f"request needs {need} (query block, KV block) pairs "
                    f"but the batched attention kernel's SMEM schedule "
                    f"holds {schedule[0]}; running it would halt the TPU "
                    "core. Use a smaller --max-num-batched-tokens, or "
                    "fewer --max-num-seqs or a shorter --max-model-len."
                )
        points = profile_points(
            self.num_tokens_paddings, self.max_num_tokens, self.max_model_len, schedule
        )
        samples = []
        started = time.perf_counter()
        for tokens, prefix in points:
            # Each point is timed twice; the faster run is kept.
            ms = min(
                self._dummy_run(
                    tokens,
                    self.num_reqs_max_model_len,
                    self.max_num_blocks_per_req,
                    use_max_model_len=True,
                    profile_prefix=prefix,
                )
                for _ in range(2)
            )
            samples.append((int(tokens), int(prefix), float(ms)))
        logger.info(
            "Pipeline step profile (%d points, %.1f s): %s",
            len(samples),
            time.perf_counter() - started,
            ", ".join(f"{t}+{p}: {ms:.1f} ms" for t, p, ms in samples),
        )
        return {
            "granularity": int(self._pp_chunk_granularity),
            "samples": samples,
            "schedule": schedule,
        }

    @contextlib.contextmanager
    def _precompile_timed(self, name: str):
        """Common boilerplate for each AOT bucket precompile pass."""
        logger.info("Compiling %s with different input shapes.", name)
        start = time.perf_counter()
        yield
        logger.info("Compilation finished in %.2f [secs].", time.perf_counter() - start)
        self._update_num_xla_graphs(name)

    def _dummy_logits(self, num_reqs: int) -> torch.Tensor:
        return torch.zeros(
            (num_reqs, self.vocab_size),
            device=self.device,
            dtype=self._hidden_states_dtype,
        )

    def _precompile_compute_selected_logits(self) -> None:
        hsize = self.model_config.get_hidden_size()
        with self._precompile_timed("compute_selected_logits"):
            for num_tokens in self.num_tokens_paddings:
                dummy_hidden = torch.zeros(
                    (num_tokens, hsize),
                    device=self.device,
                    dtype=self._hidden_states_dtype,
                )
                for num_reqs in self.num_reqs_paddings:
                    indices = torch.zeros(
                        num_reqs, dtype=torch.int32, device=self.device
                    )
                    out = self.compute_selected_logits(dummy_hidden, indices)
                    synchronize_tensors(out)
                    logger.info(
                        "  -- num_tokens: %d, num_seqs: %d", num_tokens, num_reqs
                    )
                    if num_reqs >= min(num_tokens, self.max_num_reqs):
                        break

    def _precompile_compute_logits_from_hidden_states(self) -> None:
        hsize = self.model_config.get_hidden_size()
        with self._precompile_timed("compute_logits_from_hidden_states"):
            for num_reqs in self.num_reqs_paddings:
                dummy_hidden = torch.zeros(
                    (num_reqs, hsize),
                    device=self.device,
                    dtype=self._hidden_states_dtype,
                )
                out = self.compute_logits_from_hidden_states(dummy_hidden)
                synchronize_tensors(out)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_structured_decoding(self) -> None:
        with self._precompile_timed("structured_decoding"):
            arange = self.structured_decoding_manager.structured_decode_arange
            for num_reqs in self.num_reqs_paddings:
                out = self.structured_decoding_manager.structured_decode(
                    self.structured_decoding_manager.require_structured_out_cpu[
                        :num_reqs
                    ].to(self.device),
                    self.structured_decoding_manager.grammar_bitmask_cpu[:num_reqs].to(
                        self.device
                    ),
                    self._dummy_logits(num_reqs),
                    arange,
                )
                synchronize_tensors(out)
                logger.info("  -- num_seqs: %d", num_reqs)
            # Prefill-only PCP MTP K1 skips _precompile_rejection_sampler(), so
            # we skip it here as well.
            if self.speculative_config is None or (
                self._pcp_enabled and self._mtp_enabled
            ):
                return
            # For spec decoding, target logits have padded_logits_length rows.
            max_target_rows = (
                self.structured_decoding_manager.target_grammar_bitmask_cpu.shape[0]
            )
            for num_tokens in self.num_tokens_paddings:
                if num_tokens > max_target_rows:
                    break
                out = self.structured_decoding_manager.structured_decode(
                    self.structured_decoding_manager.require_structured_out_target_cpu[
                        :num_tokens
                    ].to(self.device),
                    self.structured_decoding_manager.target_grammar_bitmask_cpu[
                        :num_tokens
                    ].to(self.device),
                    self._dummy_logits(num_tokens),
                    arange,
                )
                synchronize_tensors(out)
                logger.info("  -- target_logits_length: %d", num_tokens)

    def _precompile_sample_from_logits(self) -> None:
        with self._precompile_timed("sample_from_logits"):
            for num_reqs in self.num_reqs_paddings:
                dummy_logits = self._dummy_logits(num_reqs)
                dummy_temperatures = torch.ones(
                    (num_reqs, 1), dtype=self._hidden_states_dtype, device=self.device
                )
                dummy_u = torch.rand_like(dummy_logits)
                dummy_top_k = torch.zeros(
                    (num_reqs, 1), dtype=torch.int32, device=self.device
                )
                dummy_top_p = torch.ones(
                    (num_reqs, 1), dtype=torch.float32, device=self.device
                )
                for all_greedy in [False, True]:
                    out = self.sample_from_logits_func(
                        dummy_logits,
                        dummy_temperatures,
                        dummy_u,
                        dummy_top_k,
                        dummy_top_p,
                        all_greedy=all_greedy,
                    )
                    synchronize_tensors(out)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_gather_logprobs(self) -> None:
        with self._precompile_timed("gather_logprobs"):
            for num_reqs in self.num_reqs_paddings:
                out = self.gather_logprobs(
                    self._dummy_logits(num_reqs),
                    torch.zeros((num_reqs, 1), dtype=torch.int64).to(self.device),
                )
                synchronize_tensors(out.logprobs)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_substitute_placeholder_token(self) -> None:
        if not self.scheduler_config.async_scheduling:
            return
        # next_tokens source lengths the runtime produces: stride-1 num_reqs
        # buckets (non-spec / pure-async), plus the bounded stride-(1+K) span
        # async eagle3 parks (max_num_reqs*(1+K); see sample_tokens) so the spec
        # substitute doesn't recompile per real num_reqs.
        next_lens = list(self.num_reqs_paddings)
        if self._is_async_drafter:
            next_lens.append(
                self.max_num_reqs * (1 + self.speculative_config.num_speculative_tokens)
            )
        with self._precompile_timed("substitute_placeholder_token"):
            for num_tokens in self.num_tokens_paddings:
                input_ids = torch.zeros(
                    num_tokens, dtype=torch.int32, device=self.device
                )
                cur_input_indices = torch.zeros(
                    num_tokens, dtype=torch.int32, device=self.device
                )
                # -1 sentinel marks every slot as padding so the precompiled
                # program is byte-identical to the runtime JIT call.
                pre_next_tokens_indices = torch.full(
                    (num_tokens,), -1, dtype=torch.int32, device=self.device
                )
                for nlen in next_lens:
                    next_tokens = torch.zeros(
                        nlen, dtype=torch.int64, device=self.device
                    )
                    out = _substitute_placeholder_token(
                        input_ids,
                        cur_input_indices,
                        pre_next_tokens_indices,
                        next_tokens,
                    )
                    synchronize_tensors(out)
                    logger.info("  -- num_tokens: %d, next_len: %d", num_tokens, nlen)

    def _precompile_rejection_sampler(self) -> None:
        """Warm the spec-decode verify path so it doesn't recompile at runtime.

        The rejection sampler is ``@torch.compile(backend="tpu",
        dynamic=False)`` -> a fresh graph per input shape, and the
        ``logits[...]`` gathers around it are eager. None are covered by
        ``_precompile_sampling_subgraphs`` (non-spec sampling only) or
        ``drafter.precompile()`` (draft forward only), so they recompile every
        time the verify length (``padded_logits_length``) lands in a new
        num-tokens bucket as requests finish and the batch shrinks. Replay the
        verify inner body (``_sample_spec_verify_chunk``) at every
        ``(padded_logits_length, padded_num_reqs)`` bucket the runtime can
        produce so the real call is a cache hit. Shared sync + async.

        Both verify sub-branches are distinct ``dynamic=False`` graphs, so warm
        each: greedy (``do_sampling=False`` ->
        ``_greedy_rejection_sample_with_segment``, prelude
        ``spec_bonus_and_target_logits``) and non-greedy (``do_sampling=True``
        -> ``_random_rejection_sample_with_segment``, prelude
        ``spec_gather_bonus_and_target_logits``). Which one runs is decided per
        step by ``input_batch.all_greedy`` and ``grammar_output``, so both are
        reachable and must be pre-compiled here; the greedy + structured-output
        combination reuses graphs warmed by these two passes.
        """
        if not (self._is_async_drafter):
            return
        k = self.speculative_config.num_speculative_tokens
        # The verify chunk samples at most num_reqs*(K+1) positions; bound the
        # padded_logits_length sweep to the bucket that covers the full batch.
        # Use min here because num_reqs*(K+1) can exceed max_num_batched_tokens
        # (which is the largest bucket), and logits rows never exceed the largest
        # bucket.
        max_logits_len = _get_padded_token_len(
            self.num_tokens_paddings,
            min(self.max_num_reqs * (k + 1), self.num_tokens_paddings[-1]),
        )
        with self._precompile_timed("rejection_sampler"):
            for num_tokens in self.num_tokens_paddings:
                if num_tokens > max_logits_len:
                    break
                # padded_logits_length-shaped inputs; dtypes mirror
                # get_spec_decode_metadata so the graph is byte-identical.
                dummy_logits = torch.zeros(
                    (num_tokens, self.vocab_size),
                    device=self.device,
                    dtype=self._hidden_states_dtype,
                )
                draft_token_ids = torch.zeros(
                    num_tokens, dtype=torch.int32, device=self.device
                )
                target_logits_indices = torch.zeros(
                    num_tokens, dtype=torch.int32, device=self.device
                )
                segment_ids = torch.zeros(
                    num_tokens, dtype=torch.int64, device=self.device
                )
                group_indices = torch.zeros(
                    num_tokens, dtype=torch.int32, device=self.device
                )
                for num_reqs in self.num_reqs_paddings:
                    # padded_num_reqs-shaped inputs.
                    draft_lengths = torch.zeros(
                        num_reqs, dtype=torch.int32, device=self.device
                    )
                    bonus_logits_indices = torch.zeros(
                        num_reqs, dtype=torch.int32, device=self.device
                    )
                    warm_accept_u = torch.zeros(
                        num_tokens, dtype=torch.float32, device=self.device
                    )
                    # --- greedy verify path (do_sampling=False) ---
                    bonus_token_ids, target_logits_warm = (
                        self.spec_bonus_and_target_logits(
                            dummy_logits, bonus_logits_indices, target_logits_indices
                        )
                    )
                    out = self.rejection_sampler(
                        draft_token_ids=draft_token_ids,
                        num_draft_tokens=draft_lengths,
                        target_logits=target_logits_warm,
                        bonus_token_ids=bonus_token_ids,
                        segment_ids=segment_ids,
                        group_indices=group_indices,
                        max_draft_tokens=k,
                        accept_u=(
                            warm_accept_u
                            if self.rejection_sampler.synthetic_mode
                            else None
                        ),
                    )
                    synchronize_tensors(out)
                    # --- non-greedy verify path (do_sampling=True) ---
                    _, target_logits_ng = self.spec_gather_bonus_and_target_logits(
                        dummy_logits, bonus_logits_indices, target_logits_indices
                    )
                    warm_temps = torch.zeros(
                        (num_tokens, 1),
                        dtype=self._hidden_states_dtype,
                        device=self.device,
                    )
                    warm_top_k = torch.zeros(
                        (num_tokens, 1), dtype=torch.int32, device=self.device
                    )
                    warm_top_p = torch.ones(
                        (num_tokens, 1), dtype=torch.float32, device=self.device
                    )
                    warm_recover_u = torch.zeros(
                        (num_tokens, self.vocab_size),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    out = self.rejection_sampler(
                        draft_token_ids=draft_token_ids,
                        num_draft_tokens=draft_lengths,
                        target_logits=target_logits_ng,
                        bonus_token_ids=bonus_token_ids,
                        segment_ids=segment_ids,
                        group_indices=group_indices,
                        max_draft_tokens=k,
                        temperatures=warm_temps,
                        top_k=warm_top_k,
                        top_p=warm_top_p,
                        accept_u=warm_accept_u,
                        recover_u=warm_recover_u,
                        do_sampling=True,
                    )
                    synchronize_tensors(out)
                    logger.info(
                        "  -- padded_logits_length: %d, num_seqs: %d",
                        num_tokens,
                        num_reqs,
                    )
                    if num_reqs >= min(num_tokens, self.max_num_reqs):
                        break

    def _precompile_sampling_subgraphs(self) -> None:
        """Compile sampling-path subgraphs so their bottom-HBM reservations
        are visible to vLLM's available-memory probe in profile_run."""
        if not self._pp_is_last:
            # Only the last stage holds the lm_head and samples.
            return
        self._precompile_compute_selected_logits()
        if _get_sequence_layout_planner_for_runner(
            self
        ).uses_selected_logits_hidden_states:
            self._precompile_compute_logits_from_hidden_states()
        self._precompile_structured_decoding()
        self._precompile_sample_from_logits()
        self._precompile_gather_logprobs()

    def capture_model(self) -> None:
        """Precompile every torch.compile subgraph across all input buckets."""
        self._add_pipeline_chunk_buckets()
        if self.parallel_config.pipeline_parallel_size > 1 and self._pp_wave is None:
            self._pp_intermediate_tensors(1, zeros=True)
            assert self._pp_intermediate_template is not None
            if (
                self._pp_topk_buffer is not None
                and self._pp_topk_buffer.shape[0] < self.max_num_tokens
            ):
                raise RuntimeError(
                    "the shared top-k table holds "
                    f"{self._pp_topk_buffer.shape[0]} rows, fewer than the "
                    f"{self.max_num_tokens} tokens a hand-off carries"
                )
            self._pp_wave = PPWave(
                self.device, self.max_num_tokens, self._pp_intermediate_template
            )
            self._pp_wave.warmup()
        if self.enforce_eager:
            return
        with self.maybe_setup_dummy_loras(self.lora_config):
            with self._precompile_timed("model backbone"):
                self._precompile_backbone()

            self._precompile_sampling_subgraphs()
            self._precompile_mamba_rollback_helpers()

            # Precompile multimodal vision encoder graphs
            self.encoder_cudagraph_manager = maybe_create_mm_encoder_manager(
                self.vllm_config, self.device, self.model
            )
            if self.encoder_cudagraph_manager is not None:
                with self._precompile_timed("multimodal vision encoder"):
                    self.encoder_cudagraph_manager.precompile_vision_encoder()

            # Warm the drafter's forward + sampling subgraphs at every
            # bucket shape it may see at runtime.
            if self._is_async_drafter:
                self.drafter.precompile()
                if self._pcp_enabled and self._mtp_enabled:
                    logger.info(
                        "Skipping rejection replay and decode/verify warmup "
                        "for prefill-only PCP MTP K1"
                    )
                    self._warmup_pcp_mtp_prefill()
                else:
                    self._precompile_rejection_sampler()
                    # The isolated precompiles above emit standalone fused
                    # programs; the real per-step propose+verify+sampling
                    # dispatch fuses them differently, so the first real
                    # request would otherwise compile new graphs. Warm those
                    # through the real two-phase spec-decode path.
                    self._warmup_spec_decode()

    def _warmup_pcp_mtp_prefill(self) -> None:
        """Warm the real PCP MTP K1 prefill proposal and nothing else.

        The real path may fail once while its draft forward is being cold
        compiled. Run one bounded retry during startup so no live request
        becomes the compiler probe.

        This lifecycle is intentionally separate from ``_warmup_spec_decode``:
        a prefill-only producer must never synthesize decode/verify work.
        """
        if not envs.SPEC_WARMUP:
            return
        if self.enforce_eager or not self._is_async_drafter:
            return
        P = min(int(self.num_tokens_paddings[0]), self.max_model_len)
        n0 = self.num_xla_graphs
        with (
            _suspend_kv_transfer_group(),
            self._precompile_timed("PCP MTP real prefill warmup"),
        ):
            if not self._warmup_one_pcp_mtp_prefill(P, quiet=True):  # noqa: SIM102 (retry)
                if not self._warmup_one_pcp_mtp_prefill(P):
                    raise RuntimeError(
                        "PCP MTP prefill warmup failed after retry; refusing "
                        "to start a producer whose first live proposal would "
                        "hit the same failure"
                    )
        logger.info(
            "PCP MTP prefill warmup compiled %d graphs", self.num_xla_graphs - n0
        )

    def _warmup_one_pcp_mtp_prefill(self, P: int, quiet: bool = False) -> bool:
        """Run one synthetic PCP prefill through target sampling + K1 propose."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

        rid = "__pcp_mtp_warmup__"
        max_pos = min(self.max_model_len, P + 1)
        top = int(self.kv_cache_config.num_blocks)
        block_ids_per_group = []
        max_nblk = 0
        for group in self.kv_cache_config.kv_cache_groups:
            group_block_size = int(group.kv_cache_spec.block_size)
            group_num_blocks = min(
                cdiv(max_pos, group_block_size),
                cdiv(self.max_model_len, group_block_size),
            )
            block_ids_per_group.append(list(range(top - group_num_blocks, top)))
            max_nblk = max(max_nblk, group_num_blocks)
        if top <= max_nblk + 1:
            raise RuntimeError(
                "PCP MTP prefill warmup has not enough KV blocks for its "
                f"synthetic request: num_blocks={top}, required>{max_nblk + 1}"
            )

        attempt_succeeded = True
        try:
            request = NewRequestData(
                req_id=rid,
                prompt_token_ids=[0] * P,
                mm_features=[],
                sampling_params=SamplingParams(temperature=0.0),
                pooling_params=None,
                block_ids=tuple(block_ids_per_group),
                num_computed_tokens=0,
                lora_request=None,
            )
            scheduler_output = SchedulerOutput.make_empty()
            scheduler_output.scheduled_new_reqs = [request]
            scheduler_output.num_scheduled_tokens = {rid: P}
            scheduler_output.total_num_scheduled_tokens = P
            self.execute_model(scheduler_output)
            self.sample_tokens(None)
            self.take_draft_token_ids()
        except Exception:
            attempt_succeeded = False
            if quiet:
                logger.warning(
                    "PCP MTP prefill warmup P=%d first attempt failed; retrying once.",
                    P,
                    exc_info=True,
                )
            else:
                logger.exception("PCP MTP prefill warmup P=%d retry failed", P)
        finally:
            cleanup_succeeded = self._warmup_spec_decode_cleanup(rid)

        if not cleanup_succeeded:
            raise RuntimeError(
                "PCP MTP prefill warmup cleanup failed; refusing to retry or "
                "start with residual synthetic request state"
            )
        return attempt_succeeded

    def _precompile_backbone(self) -> None:
        """Compile the backbone for every token bucket.

        One trace serves the whole ladder unless the model's structure depends
        on the token count, in which case a bucket the current graph cannot
        serve gets a trace of its own and hands its executable back, leaving
        one executable per bucket either way (see
        vllm_torchtpu.compilation.shape_variants).
        """
        with shape_variants.warmup() as buckets:
            for num_tokens in self.num_tokens_paddings:
                logger.info("  -- num_tokens: %d", num_tokens)
                if buckets.needs_retrace(num_tokens):
                    shape_variants.retrace(self.model, self.vllm_config)
                self._dummy_run(
                    num_tokens,
                    self.num_reqs_max_model_len,
                    self.max_num_blocks_per_req,
                    use_max_model_len=True,
                )
                if self.most_model_len is not None:
                    self._dummy_run(
                        num_tokens,
                        self.num_reqs_most_model_len,
                        self.num_blocks_per_most_len_req,
                        use_max_model_len=False,
                    )
            if buckets.refused:
                # A refusal that no trace picked up leaves a raising closure in
                # the live graph, which would surface as a failed request. Fail
                # the start instead, where it can be read.
                raise shape_variants.ShapeSpecializationError(
                    f"warmup left token buckets uncompiled: {sorted(buckets.refused)}"
                )

    def _precompile_mamba_state_seed_copies(self) -> None:
        """Warm the seed-copy program for every raw-pool set at every bucket
        length, so the first real block-boundary crossing does not compile
        mid-serving. Runs right after the pool zero-fill sync: the pools must
        be quiescent when their donation is first compiled (see the note in
        initialize_kv_cache). All-zero indices copy the null block onto
        itself; src and dst are distinct tensors (dynamo specializes on
        src-is-dst otherwise).
        """
        if not self._mamba_copy_plan and not self.kv_cache_raw_tensors:
            return
        raw_sets: dict[tuple[int, ...], list[torch.Tensor]] = {}
        for _, raws in self._mamba_copy_plan:
            raw_sets.setdefault(tuple(id(r) for r in raws), raws)
        if self.kv_cache_raw_tensors:
            raw_sets.setdefault(
                tuple(id(r) for r in self.kv_cache_raw_tensors),
                self.kv_cache_raw_tensors,
            )
        # Serving calls the program under execute_model's no_grad, and dynamo
        # guards on grad mode: compile it the same way.
        with self._precompile_timed("mamba state seed copies"), torch.no_grad():
            num_groups = max(
                1,
                len(self._mamba_copy_plan),
                len(self.kv_cache_config.kv_cache_groups)
                if self.kv_cache_config is not None
                else 0,
            )
            limit = self._bucket_len(
                num_groups * self.max_num_reqs * self._pool_block_split
            )
            n = 8
            while n <= limit:
                src = torch.zeros(n, dtype=torch.int32).to(self.device)
                dst = torch.zeros(n, dtype=torch.int32).to(self.device)
                for raws in raw_sets.values():
                    copy_mamba_state_blocks(raws, src, dst)
                n *= 4
            for raws in raw_sets.values():
                synchronize_tensors(raws)

    def _precompile_mamba_rollback_helpers(self) -> None:
        """Warm the compiled rollback scatters at every bucket length.

        Mamba block-boundary crossings do not occur during the synthetic
        warmup request (unlike the reset scatter, which the real warmup
        path exercises), so without this the first real crossing would
        compile XLA mid-serving while lockstep peers wait at their next
        collective. All-zero indices write the null slot, moving no state.
        Crossings accumulate across every mamba group in a step, so the
        ladder tops out at num_groups * max_num_reqs; src and dst must be
        DISTINCT tensors or dynamo specializes the migrate graph on
        src-is-dst and the real two-tensor call retraces mid-serving.
        """
        offsets = self.mamba_slot_read_offsets
        if offsets is None:
            return
        with self._precompile_timed("mamba rollback helpers"):
            num_groups = max(
                1,
                len(self._mamba_copy_plan),
                len(self.kv_cache_config.kv_cache_groups)
                if self.kv_cache_config is not None
                else 0,
            )
            limit = self._bucket_len(num_groups * self.max_num_reqs)
            n = 8
            while n <= limit:
                src = torch.zeros(n, dtype=torch.long, device=offsets.device)
                dst = torch.zeros(n, dtype=torch.long, device=offsets.device)
                _rollback_offsets_seed_compiled(offsets, dst)
                _rollback_offsets_migrate_compiled(offsets, src, dst)
                n *= 4

    def _warmup_spec_decode(self) -> None:
        """Warm the real spec-decode dispatch so the first request doesn't recompile.

        The isolated AOT precompile (``drafter.precompile`` /
        ``_precompile_rejection_sampler``) emits *standalone* fused programs, but
        torch-tpu's DEFER_AND_FUSE fuses the per-step propose + verify + sampling
        ops into *combined* programs that depend on the live dispatch sequence —
        which an isolated precompile cannot reproduce. The only way to emit those
        exact fused programs is to run the real two-phase execute path once. We
        do that here on synthetic requests -- both greedy and non-greedy
        (do_sampling=True), since the two verify+sample paths fuse into distinct
        programs -- then fully tear each request down so serving starts from a
        clean batch. execute_model / sample_tokens
        branch internally on async_scheduling, so this same body warms whichever
        mode the process is configured for: in async it auto-routes through the
        real async dispatch (the unified device-seeded propose,
        _assemble_async_spec_substitution, cross-step subtract/extract), which
        an isolated precompile likewise can't
        reproduce. The contiguous prefill + >=2 decode steps form the async
        cross-step chain (step N reads the AsyncPreResults parked by step N-1).

        Bounded + value-insensitive: only shapes matter, so token/position VALUES
        are irrelevant. Wrapped so any failure degrades to "no warmup" (the first
        request just pays the bounded recompiles, as before) rather than breaking
        serving.
        """
        if not envs.SPEC_WARMUP:
            return  # escape hatch / A-B toggle
        if self.enforce_eager:
            return
        if not (self._is_async_drafter):
            return
        if self.input_batch.num_reqs != 0:
            logger.warning("skip spec-decode warmup: input_batch not empty")
            return

        n0 = self.num_xla_graphs
        # These SchedulerOutputs are built locally and therefore have no
        # scheduler-generated kv_connector_metadata. They must not start real
        # PD transfers or bind synthetic requests into the connector.
        with (
            _suspend_kv_transfer_group(),
            self._precompile_timed("spec-decode real warmup"),
        ):
            # (1) First-pass / first-decode shapes depend on the prompt length:
            # sweep one synthetic request per prompt-token bucket at nr=1. Each
            # runs the REAL two-phase propose+verify+sampling dispatch, so
            # torch-tpu emits the same fused programs serving will hit.
            for i, P in enumerate(self.num_tokens_paddings):
                # A single request's prompt can't exceed max_model_len; skip
                # token buckets above it (they only arise as batched totals,
                # covered by the num_reqs sweep below). Without this, a P >
                # max_model_len synthetic prompt overflows token_ids_cpu
                # (shape [.., max_model_len]) when max_num_batched_tokens >
                # max_model_len, e.g. small max_model_len configs.
                if int(P) > self.max_model_len:
                    continue
                # Greedy and non-greedy (do_sampling=True) verify+sample are
                # DISTINCT fused programs, so sweep both at every prompt bucket.
                for greedy in (True, False):
                    if not self._warmup_one_spec_request(
                        int(P), idx=i, quiet=True, greedy=greedy
                    ):
                        # First attempt cold-compiles the draft forward; a
                        # SymInt from that cold compile can leak into the
                        # dynamic=False gather wrapper on the first bucket. Retry
                        # once now that the forward is compiled and returns
                        # concrete-shaped tensors (this attempt logs loudly if it
                        # also fails).
                        self._warmup_one_spec_request(int(P), idx=i, greedy=greedy)
            # (2) The propose+verify+sampling fusions are also keyed on the
            # decode batch size (num_reqs). Only relevant when serving actually
            # batches (>1 concurrent request); sweep real multi-request batches
            # so nr>1 serving steps are cached too -- both greedy and non-greedy.
            if self.scheduler_config.max_num_seqs > 1:
                P_small = int(self.num_tokens_paddings[0])
                for R in self.num_reqs_paddings:
                    if int(R) > 1:
                        self._warmup_spec_batch(int(R), P_small, greedy=True)
                        self._warmup_spec_batch(int(R), P_small, greedy=False)
        logger.info("spec-decode warmup compiled %d graphs", self.num_xla_graphs - n0)

    def _warmup_spec_batch(self, R: int, P: int, greedy: bool = True) -> None:
        """Warm an nr=R decode batch: prefill R synthetic requests then decode
        them together so the multi-request fused programs are compiled.

        greedy=False routes the non-greedy (do_sampling=True) verify+sample
        fusion, which is a distinct set of programs from the greedy path."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            NewRequestData,
            SchedulerOutput,
        )

        K = self.speculative_config.num_speculative_tokens
        room = self.max_model_len - P
        n_decode = min(max(5, K + 2), max(0, room // (1 + K) - 1))
        max_pos = min(self.max_model_len, P + n_decode * (1 + K))
        # Per-group block counts at each group's own block size (see
        # _warmup_one_spec_request); the per-request stride uses the widest
        # group so every group's ranges stay disjoint across requests.
        group_nblks = []
        for group in self.kv_cache_config.kv_cache_groups:
            gbs = int(group.kv_cache_spec.block_size)
            group_nblks.append(min(cdiv(max_pos, gbs), cdiv(self.max_model_len, gbs)))
        nblk = max(group_nblks)
        top = int(self.kv_cache_config.num_blocks)
        if top <= R * nblk + 1:
            logger.warning("skip spec-decode warmup R=%d: not enough kv blocks", R)
            return
        rids = [f"__spec_warmup_b{R}_{j}__" for j in range(R)]
        sp = SamplingParams(temperature=0.0 if greedy else 1.0)
        logger.info("spec-decode warmup batch: R=%d P=%d greedy=%s", R, P, greedy)
        try:
            # Match the low-block working set used by the LIFO BlockPool
            # workaround. High physical block ids can trigger the v7x SC2
            # indexed-KV halt even though they are inside the allocated pool.
            # Warmup runs before serving, with no live scheduler requests;
            # reserve block 0 and keep each synthetic request's range disjoint.
            new_reqs = []
            for j, rid in enumerate(rids):
                lo = 1 + j * nblk
                new_reqs.append(
                    NewRequestData(
                        req_id=rid,
                        prompt_token_ids=[0] * P,
                        mm_features=[],
                        sampling_params=sp,
                        pooling_params=None,
                        block_ids=tuple(
                            list(range(lo, lo + g_nblk)) for g_nblk in group_nblks
                        ),
                        num_computed_tokens=0,
                        lora_request=None,
                    )
                )
            # Prefill in chunks that respect the pre-allocated host token buffers.
            # A single R*P prefill overflows positions_np ([max_num_tokens]) when
            # Decode below schedules R*(1+K) <= max_num_tokens, so only
            # the prefill needs chunking; the nr=R decode shape is unchanged.
            reqs_per_chunk = max(1, self.max_num_tokens // P)
            for c0 in range(0, R, reqs_per_chunk):
                chunk = new_reqs[c0 : c0 + reqs_per_chunk]
                so = SchedulerOutput.make_empty()
                so.scheduled_new_reqs = chunk
                so.num_scheduled_tokens = {r.req_id: P for r in chunk}
                so.total_num_scheduled_tokens = len(chunk) * P
                assert self.execute_model(so) is None
                self.sample_tokens(None)
                self.take_draft_token_ids()
            # ---- decode all R together: real nr=R propose + verify + sample ---
            nct = [P] * R
            for t in range(n_decode):
                all_token_ids = _spec_warmup_all_token_ids(rids, nct)
                creq = CachedRequestData(
                    req_ids=list(rids),
                    resumed_req_ids=set(),
                    new_token_ids=[],
                    all_token_ids=all_token_ids,
                    new_block_ids=[None] * R,
                    num_computed_tokens=list(nct),
                    num_output_tokens=[t + 1] * R,
                )
                so = SchedulerOutput.make_empty()
                so.scheduled_cached_reqs = creq
                so.num_scheduled_tokens = {rid: 1 + K for rid in rids}
                so.total_num_scheduled_tokens = R * (1 + K)
                so.scheduled_spec_decode_tokens = {rid: [0] * K for rid in rids}
                assert self.execute_model(so) is None
                self.sample_tokens(None)
                self.take_draft_token_ids()
                nct = [x + 1 for x in nct]
        except Exception:
            logger.exception("spec-decode warmup R=%d failed; continuing", R)
        finally:
            self._warmup_spec_decode_cleanup(rids)

    def _warmup_one_spec_request(
        self, P: int, idx: int, quiet: bool = False, greedy: bool = True
    ) -> bool:
        """Run one synthetic spec-decode request (prefill P + decodes).

        Returns True on success, False if the warmup raised (and was caught).
        The caller retries a failed bucket once: the first attempt cold-compiles
        the draft forward, and a SymInt from that cold compile can leak into the
        dynamic=False gather wrapper on the very first bucket; the retry runs with
        the forward already compiled (returning concrete-shaped tensors).
        """
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import (
            CachedRequestData,
            NewRequestData,
            SchedulerOutput,
        )

        rid = f"__spec_warmup_{idx}__"
        K = self.speculative_config.num_speculative_tokens
        # The contexts (num_computed_tokens) to run decode steps at: a short walk
        # near the prefill.
        room = self.max_model_len - P
        n_dec = min(max(5, K + 2), max(0, room // (1 + K) - 1))
        decode_ncts = [P + t for t in range(n_dec)]
        # Reuse low physical block ids, as in _warmup_spec_batch and the LIFO
        # BlockPool workaround, rather than exercising the SC2-sensitive end
        # of the pool. Warmup runs before serving and tears down each request;
        # real requests overwrite their valid KV positions on first prefill.
        # Cover the highest context this request reaches, reserving block 0.
        # Count blocks per kv-cache group at that group's own block size:
        # groups differ on hybrid models (mamba vs attention), and the
        # runner-level self.block_size predates the executor's post-load
        # block-size finalization.
        max_pos = min(self.max_model_len, max([P] + decode_ncts) + (1 + K))
        top = int(self.kv_cache_config.num_blocks)
        block_ids_per_group = []
        max_nblk = 0
        for group in self.kv_cache_config.kv_cache_groups:
            gbs = int(group.kv_cache_spec.block_size)
            g_nblk = min(cdiv(max_pos, gbs), cdiv(self.max_model_len, gbs))
            block_ids_per_group.append(list(range(1, 1 + g_nblk)))
            max_nblk = max(max_nblk, g_nblk)
        if top <= max_nblk + 1:
            logger.warning("skip spec-decode warmup P=%d: not enough kv blocks", P)
            return
        block_ids = tuple(block_ids_per_group)
        sp = SamplingParams(temperature=0.0 if greedy else 1.0)
        try:
            # ---- prefill ----
            new_req = NewRequestData(
                req_id=rid,
                prompt_token_ids=[0] * P,
                mm_features=[],
                sampling_params=sp,
                pooling_params=None,
                block_ids=block_ids,
                num_computed_tokens=0,
                lora_request=None,
            )
            so = SchedulerOutput.make_empty()
            so.scheduled_new_reqs = [new_req]
            so.num_scheduled_tokens = {rid: P}
            so.total_num_scheduled_tokens = P
            assert self.execute_model(so) is None
            self.sample_tokens(None)
            d = self.take_draft_token_ids()
            cur = d.draft_token_ids[0] if d and d.draft_token_ids else [0] * K
            # ---- decode steps (real propose + verify + sampling) ----
            for t, nct in enumerate(decode_ncts):
                all_token_ids = _spec_warmup_all_token_ids([rid], [nct])
                creq = CachedRequestData(
                    req_ids=[rid],
                    resumed_req_ids=set(),
                    new_token_ids=[],
                    all_token_ids=all_token_ids,
                    new_block_ids=[None],
                    num_computed_tokens=[nct],
                    num_output_tokens=[t + 1],
                )
                so = SchedulerOutput.make_empty()
                so.scheduled_cached_reqs = creq
                so.num_scheduled_tokens = {rid: 1 + K}
                so.total_num_scheduled_tokens = 1 + K
                so.scheduled_spec_decode_tokens = {rid: [int(x) for x in list(cur)[:K]]}
                assert self.execute_model(so) is None
                self.sample_tokens(None)
                d = self.take_draft_token_ids()
                if d and d.draft_token_ids:
                    cur = d.draft_token_ids[0]
        except Exception:
            if quiet:
                # First attempt: a SymInt from the draft forward's cold compile
                # can leak into the dynamic=False gather wrapper. The caller
                # retries once (forward now compiled -> concrete shapes), so log
                # quietly and let the retry surface a genuine failure.
                # WARNING, not DEBUG: this except catches a *rank-local*
                # exception raised inside a *collective* region. The raising
                # rank unwinds and retries while its peers stay blocked in
                # the collective, so a swallowed failure here surfaces as a
                # cluster-wide hang with no log line to explain it. Keep it
                # visible even on the quiet first attempt.
                logger.warning(
                    "spec-decode warmup P=%d first attempt failed; retrying. "
                    "Under DP/EP this can desynchronize ranks — if startup "
                    "hangs after this line, that is the cause.",
                    P,
                    exc_info=True,
                )
            else:
                logger.exception("spec-decode warmup P=%d failed; continuing", P)
            return False
        finally:
            self._warmup_spec_decode_cleanup(rid)
        return True

    def _warmup_spec_decode_cleanup(self, rids) -> bool:
        """Tear down the synthetic warmup request(s) so serving starts clean."""
        from vllm.v1.core.sched.output import SchedulerOutput

        cleanup_succeeded = True
        if isinstance(rids, str):
            rids = [rids]
        # Reset the two-phase guard: a half-finished step would otherwise make
        # the first real execute_model raise the "State error".
        self.execute_model_state = None
        # The last async warmup step parks an AsyncPreResults (in-flight D2H copy
        # + optimistic 1+K placeholders) with no follower step to drain it.
        # Finish the copy and null it BEFORE the drain below so the drain's
        # _flush_disjoint_async_results early-returns instead of issuing a stray
        # D2H / placeholder rollback on the torn-down synthetic request, and no
        # in-flight copy host buffer is freed mid-transfer (defensive: covers the
        # `if present:`-skip path). No-op in sync (always None).
        if self._pre_async_results is not None:
            try:
                self._pre_async_results.wait_for_copy()
            except Exception:
                cleanup_succeeded = False
                logger.exception("spec-decode warmup async copy wait failed")
            self._pre_async_results = None
        try:
            idx_map = self.input_batch.req_id_to_index
            present = {r for r in rids if r in self.requests or r in idx_map}
            if present:
                so = SchedulerOutput.make_empty()
                so.finished_req_ids = set(present)
                # total==0 -> _update_states removes + condenses, then early
                # return (no forward / sample_tokens needed).
                self.execute_model(so)
                self.execute_model_state = None
        except Exception:
            cleanup_succeeded = False
            logger.exception("spec-decode warmup drain failed")
        # Belt-and-suspenders: drop any synthetic state defensively.
        for r in rids:
            try:
                self.requests.pop(r, None)
            except Exception:
                cleanup_succeeded = False
                logger.exception("spec-decode warmup request cleanup failed")
        for attr in ("_draft_token_ids",):
            try:
                if hasattr(self.spec_decode_manager, attr):
                    setattr(self.spec_decode_manager, attr, None)
            except Exception:
                cleanup_succeeded = False
                logger.exception("spec-decode warmup draft cleanup failed")
        try:
            if self.drafter is not None:
                self.drafter.draft_chunks = None
        except Exception:
            cleanup_succeeded = False
            logger.exception("spec-decode warmup chunk cleanup failed")
        self._pre_async_results = None
        self.mm_embed_inputs = None
        if self.input_batch.num_reqs != 0:
            cleanup_succeeded = False
            logger.warning(
                "spec-decode warmup left batch non-empty (%d)",
                self.input_batch.num_reqs,
            )
        remaining = {
            r
            for r in rids
            if r in self.requests or r in self.input_batch.req_id_to_index
        }
        if remaining:
            cleanup_succeeded = False
            logger.warning(
                "spec-decode warmup state still contains %s", sorted(remaining)
            )
        return cleanup_succeeded

    @contextmanager
    def _profile_isolated_cache(self) -> Iterator[None]:
        # Isolate the kv=0-specialized profile compile into a dedicated
        # subdirectory ("profile_cache") so it can be cached across warm
        # restarts without colliding with the real serving cache (kv > 0)
        # or across different models / TP sizes / dtypes.
        import hashlib

        from vllm.compilation.caching import aot_compile_hash_factors

        from vllm_torchtpu.compilation.tpu_compiler import compute_tpu_compilation_hash

        cc = self.vllm_config.compilation_config
        base_cache_dir = cc.cache_dir or vllm_envs.VLLM_CACHE_ROOT
        factors = aot_compile_hash_factors(self.vllm_config)
        factors.append(compute_tpu_compilation_hash(self.vllm_config))
        hash_key = hashlib.sha256(str(factors).encode()).hexdigest()[:10]
        saved_cache_dir = cc.cache_dir
        cc.cache_dir = os.path.join(base_cache_dir, "profile_cache", hash_key)
        os.makedirs(cc.cache_dir, exist_ok=True)
        try:
            yield
        finally:
            cc.cache_dir = saved_cache_dir

    def profile_run(
        self,
        num_tokens: int,
    ) -> None:
        from vllm.compilation.wrapper import reset_compile_wrapper

        from vllm_torchtpu.layers.adapter.linear_common import (
            _get_quantized_matmul_fp4_op,
            _get_quantized_matmul_op,
        )

        # set_current_vllm_config is required for the post-reset compile
        # path: reset_compile_wrapper restores the wrapper's original
        # forward code, so the next trace re-instantiates CustomOps that
        # call get_current_vllm_config().
        cc = self.vllm_config.compilation_config
        saved_cache = (cc.cache_dir, cc.local_cache_dir)
        saved_sizes = cc.compile_sizes
        with set_current_vllm_config(self.vllm_config):
            self._initialize_attention_kernels()
            # Pre-warm FP8 quantized-matmul lock; Dynamo can't trace Lock.
            _get_quantized_matmul_op()
            # Same for the NVFP4 W4A16 matmul op.
            _get_quantized_matmul_fp4_op()

            # Compile backbone here so XLA materializes the FP8 activation
            # slab; vLLM's post-probe sees the real HBM. Cache writes are
            # disabled so the kv=0-specialized graph never lands in the
            # persistent serving cache. Compile only the max token bucket:
            # every program loaded on the TPU holds a permanent bottom-of-HBM
            # program-region reservation, and the kv=0-specialized graphs are
            # dead after the dynamo reset below — one throwaway program keeps
            # the region within budget for capture_model's full bucket ladder.
            with self._profile_isolated_cache():
                cc.compile_sizes = [num_tokens]
                self._dummy_run(
                    num_tokens, self.num_reqs_max_model_len, self.max_num_blocks_per_req
                )
            reset_compile_wrapper(self.model)
            # reset_compile_wrapper wipes cache_dir; restore so
            # capture_model can read/write the persistent cache.
            cc.cache_dir, cc.local_cache_dir = saved_cache
            cc.compile_sizes = saved_sizes
            torch._dynamo.reset()
            synchronize_device()

            # Sampling subgraphs (Phase A) — uses the restored cache_dir.
            if not self.enforce_eager:
                with self.maybe_setup_dummy_loras(self.lora_config):
                    self._precompile_sampling_subgraphs()

    def _resolve_tpu_group_backend(
        self,
        layer_names: list[str],
        kv_cache_spec: KVCacheSpec,
    ) -> type[Any]:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)
        if layer_names and layer_names[0] in layers:
            return cast(type[Any], layers[layer_names[0]].get_attn_backend())
        if isinstance(kv_cache_spec, (AttentionSpec, MambaSpec)):
            return cast(type[Any], PallasAttentionBackend)
        raise NotImplementedError(f"Unsupported KV cache spec: {type(kv_cache_spec)!r}")

    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """Override to keep TPUModelRunner block accounting in sync with upstream.

        Upstream vLLM's `GPUModelRunner.may_reinitialize_input_batch` may
        re-create `self.input_batch` (and its internal `BlockTable` instances)
        if block sizes change or under specific sharding/prefix-caching layouts.
        When that happens, `max_num_blocks_per_req` inside the underlying
        `BlockTable` may change (for example, shrinking from 8 to 5 blocks).

        We must immediately sync `TPUModelRunner.max_num_blocks_per_req` and
        resize `block_table_cpu` so TPU staging and spec-decode warmup block
        allocations perfectly match the new underlying `BlockTable` capacity.
        """
        super().may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        if (
            hasattr(self, "input_batch")
            and self.input_batch is not None
            and hasattr(self.input_batch, "block_table")
        ):
            try:
                self.max_num_blocks_per_req = int(
                    self.input_batch.block_table[0].get_cpu_tensor().shape[1]
                )
                if (
                    self.block_table_cpu.shape[0] != self.max_num_reqs
                    or self.block_table_cpu.shape[1] != self.max_num_blocks_per_req
                ):
                    self.block_table_cpu = torch.zeros(
                        (self.max_num_reqs, self.max_num_blocks_per_req),
                        dtype=torch.int32,
                        device="cpu",
                    )
            except (IndexError, TypeError, AttributeError, KeyError):
                pass

    def _token_padding_update(self, num_tokens_padded: int) -> None:
        """Record which of this step's token rows are padding."""
        padding_state = getattr(self, "_token_padding_state", None)
        if padding_state is None:
            return
        plan = self._last_sequence_layout_plan
        if plan is not None and plan.kind is SequenceLayoutKind.ALL:
            num_valid_tokens = min(plan.local_num_tokens, num_tokens_padded)
        else:
            # Padding-mask support is currently limited to the ALL layout.
            num_valid_tokens = num_tokens_padded
        padding_state.update(num_valid_tokens, num_tokens_padded)

    def forward_model(
        self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None
    ):
        # @support_torch_compile annotations will be put on the vLLM model if it
        # supports torch.compile
        kwargs = {}
        if intermediate_tensors is not None:
            kwargs["intermediate_tensors"] = intermediate_tensors
        out = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        # A stage before the last returns the tensors the next stage
        # consumes rather than hidden states to sample from.
        if isinstance(out, IntermediateTensors):
            return out, None
        # @support_torch_compile may wrap a single tensor in a 1-element
        # list/tuple; unwrap that first.
        if isinstance(out, (list, tuple)) and len(out) == 1:
            out = out[0]
        # Draft Models: the target model's forward returns
        # (hidden_states, aux_hidden_states) when aux layers are registered.
        aux_hidden_states = None
        if isinstance(out, (list, tuple)) and len(out) == 2:
            out, aux_hidden_states = out
        return out, aux_hidden_states

    def _pp_intermediate_tensors(
        self, num_tokens: int, zeros: bool, random: bool = False
    ) -> IntermediateTensors:
        """Buffers shaped like this stage's input from the previous stage:
        zeros, uninitialized, or random values."""
        if self._pp_intermediate_template is None:
            probe = self.model.make_empty_intermediate_tensors(
                batch_size=1, dtype=self.model_config.dtype, device=self.device
            )
            template = {
                key: (tuple(tensor.shape[1:]), tensor.dtype)
                for key, tensor in probe.items()
            }
            if self._pp_topk_buffer is not None:
                template[_TOPK_INDICES] = (
                    tuple(self._pp_topk_buffer.shape[1:]),
                    self._pp_topk_buffer.dtype,
                )
            self._pp_intermediate_template = template
        tensors: dict[str, torch.Tensor] = {}
        for key, (shape, dtype) in self._pp_intermediate_template.items():
            fill = _PP_CARRIED_FILL.get(key)
            if fill is not None:
                tensors[key] = torch.full(
                    (num_tokens, *shape), fill, dtype=dtype, device=self.device
                )
            elif random:
                tensors[key] = self._profile_input(
                    key, (self.max_num_tokens, *shape), dtype
                )[:num_tokens].to(self.device)
            else:
                alloc = torch.zeros if zeros else torch.empty
                tensors[key] = alloc(
                    (num_tokens, *shape), dtype=dtype, device=self.device
                )
        return IntermediateTensors(tensors)

    def _pp_take_topk_indices(
        self, intermediate_tensors: IntermediateTensors | None, rows: int
    ) -> None:
        """Take the incoming top-k table into the buffer this stage's sparse
        layers read from.
        """
        if self._pp_topk_buffer is None or intermediate_tensors is None:
            return
        received = intermediate_tensors.tensors.pop(_TOPK_INDICES, None)
        if received is None:
            raise RuntimeError(
                "a pipeline stage of a sparse model was handed no top-k "
                "indices; every stage passes them on"
            )
        self._pp_topk_buffer[:rows].copy_(received)

    def _pp_outgoing_tensors(
        self, tensors: dict[str, torch.Tensor], rows: int
    ) -> dict[str, torch.Tensor]:
        """What this stage hands the next one: the model's activations, plus
        the top-k table whether this stage chose it or passed it through.
        """
        if self._pp_topk_buffer is None:
            return tensors
        return {**tensors, _TOPK_INDICES: self._pp_topk_buffer[:rows].clone()}

    def pp_push(self) -> None:
        """A pipeline hand-off that carries nothing, so the forwards in
        flight move one stage further; a no-op without a pipeline."""
        if self._pp_wave is not None:
            self._pp_wave.push()

    def pp_settle(self) -> None:
        """Close the pipeline hand-off burst once nothing is in flight; a
        no-op without a pipeline."""
        if self._pp_wave is not None:
            self._pp_wave.settle()

    def _sample_tokens_pp_intermediate_stage(self) -> ModelRunnerOutput:
        """Sampling runs on the last stage only; earlier stages report just
        their KV-connector progress so the aggregated step output is
        complete."""
        scheduler_output = self._pp_pending_scheduler_output
        self._pp_pending_scheduler_output = None
        if scheduler_output is None:
            return EMPTY_MODEL_RUNNER_OUTPUT
        self.maybe_wait_for_kv_save()
        (
            finished_sending,
            finished_recving,
            kv_worker_meta,
            invalid_block_ids,
            invalid_block_group_index,
            kv_connector_stats,
        ) = self.get_finished_kv_transfers(scheduler_output)
        if (
            finished_sending is None
            and finished_recving is None
            and kv_worker_meta is None
            and not invalid_block_ids
            and kv_connector_stats is None
        ):
            return EMPTY_MODEL_RUNNER_OUTPUT
        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = _build_kv_connector_output(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            kv_connector_worker_meta=kv_worker_meta,
            invalid_block_ids=invalid_block_ids,
            invalid_block_group_index=invalid_block_group_index,
            kv_connector_stats=kv_connector_stats,
        )
        return output

    def _build_padded_sampling_params(
        self,
        cur_start_idx: int,
        cur_end_idx: int,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Stage [padded_num_reqs] on CPU with neutral padding, then use one
        # fixed-shape H2D copy per tensor to keep decode shape-stable.
        padded_num_reqs = logits.shape[0]
        num_active_reqs = cur_end_idx - cur_start_idx
        temps_cpu = torch.ones(padded_num_reqs, dtype=logits.dtype)
        top_k_cpu = torch.zeros(padded_num_reqs, dtype=torch.int32)
        top_p_cpu = torch.ones(padded_num_reqs, dtype=torch.float32)

        temps_cpu[:num_active_reqs].copy_(
            self.input_batch.temperature_cpu_tensor[cur_start_idx:cur_end_idx]
        )
        active_top_k = self.input_batch.top_k_cpu_tensor[cur_start_idx:cur_end_idx]
        top_k_cpu[:num_active_reqs].copy_(
            torch.where(
                active_top_k >= self.vocab_size,
                torch.zeros_like(active_top_k),
                active_top_k,
            )
        )
        top_p_cpu[:num_active_reqs].copy_(
            self.input_batch.top_p_cpu_tensor[cur_start_idx:cur_end_idx]
        )

        return (
            temps_cpu.unsqueeze(1).to(logits.device, non_blocking=True),
            top_k_cpu.unsqueeze(1).to(logits.device, non_blocking=True),
            top_p_cpu.unsqueeze(1).to(logits.device, non_blocking=True),
        )

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def compute_selected_logits(
        self, hidden_states: torch.Tensor, indices_do_sample: torch.Tensor
    ) -> torch.Tensor:
        if self.is_pooling_model:
            return torch.empty((0,), device=hidden_states.device)
        selected = torch.index_select(hidden_states, 0, indices_do_sample)
        return self.model.compute_logits(selected)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def compute_logits_from_hidden_states(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def spec_bonus_and_target_logits(
        self,
        logits: torch.Tensor,
        bonus_logits_indices: torch.Tensor,
        target_logits_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Spec-decode verify prelude: gather the bonus-token logits + argmax, and
        # gather the target_logits rows for the rejection sampler, in ONE compiled
        # region keyed on [num_tokens, num_reqs] shapes. Run raw (the
        # logits[indices] gathers + argmax at the verify site), these are eager
        # ops torch-tpu's DEFER_AND_FUSE fuses with the live per-step dispatch
        # into per-context programs (the cold target-side recompiles). index_select
        # is value-identical to logits[indices]. Same pattern as the draft wraps.
        bonus_token_ids = torch.argmax(
            torch.index_select(logits, 0, bonus_logits_indices), dim=-1
        )
        target_logits = torch.index_select(logits, 0, target_logits_indices)
        return bonus_token_ids, target_logits

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def spec_gather_bonus_and_target_logits(
        self,
        logits: torch.Tensor,
        bonus_logits_indices: torch.Tensor,
        target_logits_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Non-greedy verify prelude: gather the bonus-token logits and the
        # target_logits rows in ONE compiled region keyed on [num_reqs,
        # num_tokens] shapes. Unlike the greedy spec_bonus_and_target_logits,
        # the bonus row is kept as FULL logits (not argmax'd) because the
        # non-greedy path samples the bonus token from them. Run raw, the
        # logits[indices] gathers are eager ops torch-tpu's DEFER_AND_FUSE fuses
        # with the live per-step dispatch into per-context programs (cold
        # recompiles).index_select is value-identical to logits[indices].
        bonus_logits = torch.index_select(logits, 0, bonus_logits_indices)
        target_logits = torch.index_select(logits, 0, target_logits_indices)
        return bonus_logits, target_logits

    def _apply_temperature(
        self, logits: torch.Tensor, temperatures: torch.Tensor
    ) -> torch.Tensor:
        safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
        return logits / safe_temperatures

    # TODO: Under SPMD mode, sample_from_logits has correctness issue.
    #       Re-enable the torch.compile once the issue is fixed in torchxla.
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def sample_from_logits(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        u: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        all_greedy: bool = False,
    ) -> torch.Tensor:
        """
        Sample with xla-friendly function. This function is to be traced
        separately from `forward` for lighter compilation overhead.
        """
        if all_greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        is_greedy = temperatures <= SAMPLING_EPS
        scaled_logits = self._apply_temperature(logits, temperatures)
        masked_logits = apply_top_k_top_p(scaled_logits, top_k, top_p)
        u_clamped = torch.clamp(
            u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
        )
        gumbel_noise = -torch.log(-torch.log(u_clamped))
        noisy_logits = masked_logits + gumbel_noise
        final_logits = torch.where(is_greedy, logits, noisy_logits)
        return torch.argmax(final_logits, dim=-1, keepdim=True)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def gather_logprobs(
        self, logits: torch.Tensor, sampled_tokens: torch.Tensor
    ) -> LogprobsTensors:
        """
        Gather the top_logprobs with corresponding tokens. Use a fixed number
        of logprobs as an alternative to having multiple pre-compiled graphs.
        Select the number of logprobs actually demanded by each request on CPU.
        """
        token_ids = sampled_tokens.to(torch.int64)
        token_logits = logits.gather(-1, token_ids)
        token_ranks = (logits >= token_logits).sum(dim=-1, dtype=torch.int32)
        log_normalizers = torch.logsumexp(logits, dim=-1, keepdim=True)
        token_logprobs = (token_logits - log_normalizers).to(torch.float32)

        max_logprobs = self.model_config.max_logprobs
        if max_logprobs > 0:
            topk_logits, topk_indices = torch.topk(logits, max_logprobs, dim=-1)
            topk_logprobs = (topk_logits - log_normalizers).to(torch.float32)
            logprob_token_ids = torch.cat((token_ids, topk_indices), dim=1)
            logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
        else:
            logprob_token_ids = token_ids
            logprobs = token_logprobs

        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids.to(torch.int32),
            logprobs=logprobs,
            selected_token_ranks=token_ranks,
        )


def _get_req_paddings(min_req_size: int, max_req_size: int) -> list[int]:
    logger.info("Preparing request paddings:")
    # assert min_req_size is power of 2
    assert (min_req_size & (min_req_size - 1) == 0) and min_req_size > 0
    paddings: list = []
    num = max(MIN_NUM_SEQS, min_req_size)
    while num <= max_req_size and (len(paddings) == 0 or paddings[-1] != num):
        paddings.append(num)
        logger.info("    %d", num)
        num = _get_padded_num_reqs_with_upper_limit(num + 1, max_req_size)
    return paddings


def _get_padded_num_reqs_with_upper_limit(x: int, upper_limit: int) -> int:
    res = MIN_NUM_SEQS if x <= MIN_NUM_SEQS else 1 << (x - 1).bit_length()
    return min(res, upper_limit)


def _get_padded_token_len(paddings: list[int], x: int) -> int:
    """Return the first element in paddings list greater or equal to x."""
    index = bisect.bisect_left(paddings, x)
    assert index < len(paddings)
    return paddings[index]


def _get_padded_num_kv_cache_update_slices(
    num_tokens: int, max_num_reqs: int, page_size: int
) -> int:
    """Calculates the padded number of KV cache update slices to avoid
    recompilation."""
    # NOTE(chengjiyao): let's say R_i is the token num for i-th request,
    # so it occupies most 2 + R_i // page_size pages. The total maximum
    # possible number of pages needed is sum(2 + R_i // page_size), which
    # is <= 2 * max_num_reqs + sum(R_i) // page_size
    # = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = 2 * max_num_reqs + num_tokens // page_size
    padded_num_slices = min(padded_num_slices, num_tokens)
    return padded_num_slices


def _prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


def _get_num_slices_per_kv_cache_update_block(page_size_bytes: int) -> int:
    """Find the optimum number of slices to copy per Pallas program instance.

    Increasing the number of slices copied in one instance of the kernel program
    will increase HBM bandwidth utilization via more in-flight DMAs.

    However, it will also use more VMEM, and experimentally, we observed
    performance regression at 128 slices on v6e, likely due to running
    out of scalar registers. Thus this function will limit the number of
    slices to 64.
    """
    # The default vmem_limit_bytes of a pallas kernel is 32MB. Here we
    # calculate num_slices_per_block based on 16MB in case any register spills.
    vmem_limit = 16 * 1024 * 1024
    num_slices_per_block = vmem_limit // page_size_bytes
    assert num_slices_per_block > 0, "Number of slices should be positive"
    num_slices_per_block = _prev_power_of_2(num_slices_per_block)
    if num_slices_per_block > 64:
        num_slices_per_block = 64
    return num_slices_per_block
