# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ensure environment overrides are applied before any other imports,
# especially torch_tpu which might read them at import time.
import vllm_torchtpu.env_override  # noqa: F401  # isort: skip

import bisect
import contextlib
import copy
import dataclasses
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, Iterator, cast

# TODO: Remove this after jax dependency is removed
import jax
import numpy as np
import torch
import torch_tpu  # noqa: F401
import vllm.envs as vllm_envs
# TODO: Remove this after jax dependency is removed
from jax.sharding import Mesh
from packaging import version
from torch_tpu._internal import sync
from vllm.config import (CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config, set_current_vllm_config)
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group,
                                          kv_transfer_state)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention import (Attention,
                                                  ChunkedLocalAttention,
                                                  MLAAttention)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.rotary_embedding import (MRotaryEmbedding,
                                                         RotaryEmbedding)
from vllm.model_executor.layers.rotary_embedding.mrope_interleaved import \
    MRotaryEmbeddingInterleaved
from vllm.model_executor.model_loader import get_model_loader
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec, MambaSpec,
                                        MLAAttentionSpec, SlidingWindowSpec,
                                        UniformTypeKVCacheSpecs)
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds,
                             LogprobsLists, LogprobsTensors, ModelRunnerOutput)
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorOutput
from vllm.v1.worker.utils import (AttentionGroup,
                                  add_kv_sharing_layers_to_kv_cache_groups,
                                  prepare_kernel_block_sizes)

from vllm_torchtpu import envs, utils
from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.kv_cache_materializer import (
    build_kernel_block_size_by_group_id, format_kv_cache_layout_summary,
    materialize_kv_cache_tensors)
from vllm_torchtpu.kv_cache_spec_normalizer import \
    normalize_kv_cache_specs_for_tpu
from vllm_torchtpu.layers.common.attention_metadata import (
    AttentionMetadata, AttentionMetadataBuilder,
    AttentionMetadataBuilderContext)
from vllm_torchtpu.layers.common.sequence_layout import (
    SequenceLayoutKind, create_sequence_layout_planner)
from vllm_torchtpu.layers.vllm import token_padding
from vllm_torchtpu.layers.vllm.attention import (TPU_STR_DTYPE_TO_TORCH_DTYPE,
                                                 PallasAttentionBackend,
                                                 PallasMLAttentionBackend)
from vllm_torchtpu.layers.vllm.custom_ops.mamba_state_copy_op import \
    copy_mamba_state_blocks
from vllm_torchtpu.layers.vllm.quantization import get_tpu_quantization_config
from vllm_torchtpu.layers.vllm.sample.rejection_sampler import RejectionSampler
from vllm_torchtpu.layers.vllm.sample.top_k_top_p import apply_top_k_top_p
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from vllm_torchtpu.platforms.tpu_block_size_utils import \
    unified_kv_layout_enabled
from vllm_torchtpu.runner import utils as runner_utils
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpecDecodeMetadata, SpeculativeDecodingManager)
from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID, AsyncPreResults, AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput)
from vllm_torchtpu.spec_decode.dflash import DFlashProposer
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer
from vllm_torchtpu.spec_decode.utils import DraftChunkInputs

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
    mamba_state_indices_list: list[list[torch.Tensor]
                                   | None] = field(default_factory=list)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _substitute_placeholder_token(
        input_ids: torch.Tensor, token_in_tpu_cur_input_indices: torch.Tensor,
        token_in_tpu_pre_next_tokens_indices: torch.Tensor,
        next_tokens: torch.Tensor):
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
    assert input_ids.shape[0] == token_in_tpu_cur_input_indices.shape[
        0] == token_in_tpu_pre_next_tokens_indices.shape[0]
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
        req_ids: list[str],
        num_computed_tokens: list[int]) -> dict[str, list[int]]:
    """Build complete synthetic token histories for async cached requests."""
    assert len(req_ids) == len(num_computed_tokens), (
        "spec-decode warmup request IDs and computed-token counts must align: "
        f"requests={len(req_ids)}, counts={len(num_computed_tokens)}")
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
    "invalid_block_group_index" in getattr(KVConnectorOutput,
                                           "__dataclass_fields__", {}))


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
    elif invalid_block_group_index is not None:
        raise RuntimeError(
            "Installed vLLM does not support cache-group-scoped KV load "
            "failure recovery; upgrade vLLM before recovering a failed "
            f"hybrid KV load (group={invalid_block_group_index})")
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
        raise RuntimeError(
            "Argmax is having accuracy issue with libtpu < 0.0.35")
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
    aliased = ("Stream", "Event", "current_stream", "default_stream", "stream",
               "set_stream", "synchronize", "set_device", "current_device",
               "device_count", "is_available")
    for name in aliased:
        saved[name] = getattr(torch.cuda, name, None)
        setattr(torch.cuda, name, getattr(torch.tpu, name))
    saved["mem_get_info"] = getattr(torch.cuda, "mem_get_info", None)
    torch.cuda.mem_get_info = lambda *a, **kw: torch.accelerator.get_memory_info(
    )
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
        return (self.speculative_config.use_eagle()
                or self.speculative_config.method == "dflash")

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
                len(layout_devices))
        # Disable cudagraphs before parent init so its dispatch self-disables.
        # TPU uses AOT bucket precompile (_precompile_* methods) instead.
        vllm_config.compilation_config.cudagraph_capture_sizes = []
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        with _torch_tpu_wrapper():
            super().__init__(vllm_config, device)
        self.sequence_layout_planner = sequence_layout_planner
        _validate_libtpu_version()
        self.is_pooling_model = (self.model_config is not None and
                                 self.model_config.runner_type == "pooling")

        if self.is_pooling_model:
            from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
            from vllm.v1.worker.tpu_input_batch import InputBatch

            def get_pooling_metadata(batch):
                reqs = batch.requests[:batch.num_reqs]
                pooling_params = [r.pooling_params for r in reqs]
                pooling_states = [
                    PoolingStates(r.req_id, r.num_computed_tokens)
                    for r in reqs
                ]
                prompt_lens = torch.tensor(
                    [r.prompt_token_ids_len for r in reqs], dtype=torch.int32)
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
            num_ckpts = (
                self.vllm_config.speculative_config.num_speculative_tokens + 1)
        self._mamba_slot_stride: int = num_ckpts
        self._mamba_ckpt_window: int = (num_ckpts if unified_kv_layout_enabled(
            self.vllm_config) else 1)
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
        self._read_offset_scratch: dict[tuple[int, str],
                                        tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor,
                                              torch.Tensor]] = {}
        # Use uniform Mamba layout for disagg until compact cache is supported
        # for disagg serving.
        self._uniform_mamba_layout: bool = (self.vllm_config.kv_transfer_config
                                            is not None)
        # Unified layout: attention KV and mamba state are fungible
        # block-table blocks in one attention-shaped pool per cache tensor.
        self._unified_kv_layout: bool = unified_kv_layout_enabled(
            self.vllm_config)
        self.kv_cache_raw_tensors: list[torch.Tensor] = []
        # Set by initialize_kv_cache; None until then (dummy runs check this).
        self.kv_cache_config: KVCacheConfig | None = None
        # Layout plan of the most recent prepared batch; None before the
        # first execute_model.
        self._last_sequence_layout_plan = None

        # EP-DP lockstep state, refreshed each step by execute_model /
        # execute_dummy_batch: the coordinated (max-across-ranks) token
        # bucket and chunk count.
        self._dp_target_bucket: int | None = None
        self._dp_step_num_chunks: int = 0
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
            self.kv_cache_dtype = (TPU_STR_DTYPE_TO_TORCH_DTYPE[self.dtype] if
                                   isinstance(self.dtype, str) else self.dtype)
        else:
            self.kv_cache_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]
        self._hidden_states_dtype = self.dtype

        # Compile bucket + padding (TPU AOT compile rounds inputs up).
        self.sliding_window = self.model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.most_model_len = envs.VLLM_TPU_MOST_MODEL_LEN
        # Sync max_num_blocks_per_req with the underlying GPUInputBatch's
        # block_table capacity if already created during super().__init__().
        # This ensures that calculations derived from max_num_blocks_per_req
        # (such as spec-decode warmup block count `nblk`) strictly respect the
        # actual BlockTable allocation limits, preventing bounds mismatch errors
        # (e.g., broadcasting shape (8,) into capacity (5,)).
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        if (hasattr(self, "input_batch") and self.input_batch is not None
                and hasattr(self.input_batch, "block_table")):
            try:
                self.max_num_blocks_per_req = int(
                    self.input_batch.block_table[0].get_cpu_tensor().shape[1])
            except (IndexError, TypeError, AttributeError, KeyError):
                pass
        self.num_blocks_per_most_len_req = (cdiv(
            self.most_model_len, self.block_size) if self.most_model_len
                                            is not None else None)
        # InputBatch needs to work with sampling tensors greater than padding
        # to avoid dynamic shapes. Also, avoid suboptimal alignment.
        self.max_num_reqs = max(self.max_num_reqs, MIN_NUM_SEQS)
        self.num_tokens_paddings = vllm_config.compilation_config.compile_sizes
        # Override parent's max_num_tokens with the last (largest) padding
        # bucket so all CPU staging buffers cover it.
        self.max_num_tokens = self.num_tokens_paddings[-1]
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            self.parallel_config, "attention")
        self.num_kv_heads = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.vocab_size = self.model_config.get_vocab_size()

        # Pallas attention bookkeeping.
        self._attention_kernels_initialized = False

        # CPU staging tensors (TPU prepares inputs on CPU then transfers).
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu")
        self.positions_np = self.positions_cpu.numpy()
        self.block_table_cpu = torch.zeros(
            (self.max_num_reqs, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu")
        # CPU staging for compact-mamba per-request recurrent-slot ids; H2D
        # copied each step into `AttentionMetadata.mamba_state_indices`.
        self.mamba_state_indices_cpu = torch.zeros(self.max_num_reqs,
                                                   dtype=torch.int32,
                                                   device="cpu")
        # Block-table-derived mamba state: the state-block position each
        # request's state currently lives at, the mamba-group -> raw-pool
        # mapping (built in initialize_kv_cache), and the block-seed copies
        # staged by _prepare_inputs and flushed before the forward.
        self._mamba_state_pos: dict[str, int] = {}
        self._mamba_copy_plan: list[tuple[int, list[torch.Tensor]]] = []
        # Manager-block -> pool-block split: >1 when the pool is born at a
        # smaller attention-kernel granularity (batched RPA).
        self._pool_block_split: int = 1
        self._pending_mamba_state_copies: list[tuple[torch.Tensor,
                                                     torch.Tensor,
                                                     torch.Tensor]] = []
        if self.uses_mrope:
            # Override parent's int64 mrope buffer with int32 so the H2D copy is
            # dtype-identical to what the TPU model expects. Parent's
            # _calc_mrope_positions writes via .cpu and .np — both route here.
            self.mrope_positions = self._make_buffer(3,
                                                     self.max_num_tokens + 1,
                                                     dtype=torch.int32)
        self.query_start_loc_cpu = torch.zeros(self.max_num_tokens + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=PIN_MEMORY)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=PIN_MEMORY)
        self.seq_lens_np = self.seq_lens_cpu.numpy()
        if self.supports_mm_inputs:
            self.is_mm_embed_cpu = torch.zeros(self.max_num_tokens,
                                               dtype=torch.bool,
                                               device="cpu",
                                               pin_memory=PIN_MEMORY)
        self.arange_np = np.arange(self.max_num_tokens, dtype=np.int64)
        self.num_reqs_paddings = _get_req_paddings(
            min_req_size=MIN_NUM_SEQS, max_req_size=self.max_num_reqs)

        # Pallas SMEM-aware num_reqs caps.
        self.num_reqs_most_model_len = (min(
            PallasAttentionBackend.get_max_num_seqs(
                self.most_model_len, self.block_size), self.max_num_reqs,
            self.max_num_tokens) if self.most_model_len is not None else None)
        self.num_reqs_max_model_len = min(
            PallasAttentionBackend.get_max_num_seqs(self.max_model_len,
                                                    self.block_size),
            self.max_num_reqs, self.max_num_tokens)

        # Structured decoding staging tensors.
        self.grammar_bitmask_cpu = torch.zeros(
            (self.max_num_reqs, cdiv(self.vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=PIN_MEMORY)
        self.require_structured_out_cpu = torch.zeros((self.max_num_reqs, 1),
                                                      dtype=torch.bool,
                                                      device="cpu",
                                                      pin_memory=PIN_MEMORY)
        self.structured_decode_arange = torch.arange(0,
                                                     32,
                                                     device="cpu",
                                                     pin_memory=PIN_MEMORY)
        self.sample_from_logits_func = self.sample_from_logits

        # TPU async-scheduling state (passed between execute_model and
        # sample_tokens, mirroring vLLM's async path).
        self.mm_embed_inputs: tuple[list[torch.Tensor],
                                    torch.Tensor] | None = None
        self.execute_model_state: ExecuteModelState | None = None
        self._pre_async_results: AsyncPreResults | None = None

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
        self._combined_request_distribution_cpu = torch.zeros(
            6, dtype=torch.int32)
        self._decode_device_cache_key: tuple | None = None
        self._cached_query_start_loc: torch.Tensor | None = None
        self._cached_logits_indices: torch.Tensor | None = None
        self._cached_request_distribution: torch.Tensor | None = None

        self.speculative_config = self.vllm_config.speculative_config
        self.spec_decode_manager = SpeculativeDecodingManager(self)
        self._init_speculative_decoding()

        # Dedicated RNG for non-greedy sampling, to keep the draws consistent
        # across a replica's TP ranks (else they diverge -> collective hang).
        self._sampling_generator: torch.Generator | None = None
        # JAX Mesh for shard_map ops in TPU kernels. Real DP is vLLM
        # multi-engine DP, so this per-worker mesh only has a model axis.
        self.mesh = self._create_mesh_for_parallelism()
        self.batch_counter = 0

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
        self.phased_profiling_dir = (profiler_config.torch_profiler_dir
                                     if envs.USE_PHASED_PROFILER else "")
        self.phase_based_profiler = None

    def start_phased_profiling(self,
                               profile_prefix: str | None = None) -> None:
        """Arms the phase-based profiler. Called from TPUWorker.profile()."""
        if not self.phased_profiling_dir:
            logger.warning(
                "Phased profiling directory is not set. Skipping profiling.")
            return
        if self.phase_based_profiler is not None:
            logger.warning(
                "Phased profiler is already running. Ignoring start request.")
            return
        profiler_config = self.vllm_config.profiler_config
        additional_config = self.vllm_config.additional_config
        decode_kv_len_threshold = additional_config.get(
            runner_utils.PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD_KEY,
            runner_utils.PHASED_PROFILER_DECODE_ONLY_KV_LEN_THRESHOLD)
        # Same scoping as the standard profiler: the prefix names the run, and
        # the per-phase subdirectories sit beneath it.
        profile_dir = self.phased_profiling_dir
        if profile_prefix:
            profile_dir = os.path.join(profile_dir, profile_prefix)
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
                or runner_utils.PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR),
            num_decode_steps_to_skip=profiler_config.delay_iterations,
            decode_kv_len_threshold=decode_kv_len_threshold,
        )

    def stop_phased_profiling(self) -> None:
        """Disarms the phase-based profiler. Called from TPUWorker.profile()."""
        if self.phase_based_profiler is None:
            logger.warning(
                "Phased profiler is not running. Ignoring stop request.")
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
        torch.tpu.synchronize()

    def maybe_setup_kv_connector(self,
                                 scheduler_output,
                                 wait_for_completion: bool = False,
                                 report_completion: bool = True) -> None:
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
        kv_connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata)
        # forward_context is unused by TPUConnector; pass None.
        if wait_for_completion:
            kv_connector.start_load_kv(None,
                                       wait_for_completion=True,
                                       report_completion=report_completion)
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
            scheduler_output.finished_req_ids)
        invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
        invalid_block_group_index = (
            kv_connector.get_block_ids_with_load_errors_group_index()
            if invalid_block_ids else None)
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
        return (finished_sending, finished_recving, worker_meta,
                invalid_block_ids, invalid_block_group_index,
                kv_connector_stats)

    def kv_connector_no_forward(self, scheduler_output,
                                vllm_config) -> ModelRunnerOutput:
        # Only Raiden inline mode blocks; otherwise the no-forward step returns
        # immediately and the scheduler re-polls get_finished each step until
        # the loads land.
        self.maybe_setup_kv_connector(
            scheduler_output,
            wait_for_completion=dist_utils.get_raiden_inline_load())
        (finished_sending, finished_recving, worker_meta, invalid_block_ids,
         invalid_block_group_index,
         kv_connector_stats) = self.get_finished_kv_transfers(scheduler_output)
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
            self.rejection_sampler = RejectionSampler()
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "dflash":
                self.drafter = DFlashProposer(self, self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = Eagle3Proposer(self, self.vllm_config)
            else:
                raise NotImplementedError(
                    "Unsupported speculative decoding method: "
                    f"{self.speculative_config.method}")

    def _truncate_rope_caches(self) -> None:
        """Slice rotary cos_sin caches to max_model_len which can reduce
        the overhead of xla layout data copy. Applicable to text-only.
        """
        multimodal_config = self.model_config.multimodal_config
        if self.model_config.is_multimodal_model and not (
                multimodal_config is not None
                and multimodal_config.language_model_only):
            logger.warning(
                "TPU_ROPE_CACHE_TRUNCATE skipped, multimodal positions are "
                "not bounded by max_model_len")
            return
        max_len = self.model_config.max_model_len
        num_eligible = 0
        num_truncated = 0
        for name, module in self.model.named_modules():
            if type(module) not in (RotaryEmbedding, MRotaryEmbedding,
                                    MRotaryEmbeddingInterleaved):
                continue
            for buf_name in ("cos_sin_cache", "cos_sin_cache_bf16"):
                buf = getattr(module, buf_name, None)
                if buf is None or not isinstance(buf, torch.Tensor):
                    continue
                num_eligible += 1
                if buf.ndim >= 1 and buf.shape[0] > max_len:
                    truncated = buf[:max_len].clone(
                        memory_format=torch.contiguous_format)
                    setattr(module, buf_name, truncated)
                    num_truncated += 1
                    logger.info("Truncated rope cache %s.%s rows %d -> %d",
                                name, buf_name, buf.shape[0], max_len)
        if num_eligible:
            logger.info("Truncated %d of %d rope caches", num_truncated,
                        num_eligible)
        else:
            logger.warning(
                "TPU_ROPE_CACHE_TRUNCATE is set but the model has no "
                "position-indexed rope cache, nothing was truncated")

    def _create_mesh_for_parallelism(self) -> Mesh:
        local_devices = list(jax.local_devices())
        if not local_devices:
            raise ValueError("No TPU devices are visible to create JAX mesh.")

        # Per-worker JAX mesh is normally single-chip; vLLM native multiprocess
        # handles TP>1 outside this JAX SPMD path.
        if self.parallel_config.world_size == 1 and \
                self.parallel_config.tensor_parallel_size > 1:
            raise ValueError(
                "Single-process TPU mesh TP>1 is not supported in this path. "
                "Use vLLM multiprocess mode for --tensor-parallel-size > 1.")
        mesh_devices = np.asarray(local_devices[:1]).reshape((1, ))
        mesh = Mesh(mesh_devices, axis_names=("model", ))
        logger.info("Init mesh | tp_size=1 | device_id=%s",
                    getattr(local_devices[0], "id", str(local_devices[0])))
        return mesh

    def _update_num_xla_graphs(self, case_str):
        check_comp = self.check_recompilation and not self.enforce_eager
        if not check_comp:
            return

        stats = torch.tpu._get_cache_stats()
        total_graphs = len(stats.per_entry_stats)
        new_compiled_graphs = total_graphs - self.num_xla_graphs
        if new_compiled_graphs == 0:
            logger.info(f"No new compiled graphs for case: {case_str}")
            return

        logger.info(f"Total Requests: {stats.num_cache_reqs}")
        logger.info(f"Total Hits: {stats.num_cache_hits}")
        logger.info(
            f"Total number of cached graphs: {total_graphs}, new: {new_compiled_graphs}, case: {case_str}"
        )
        self.num_xla_graphs += new_compiled_graphs

    def _reorder_batch_for_rpa(
            self, scheduler_output: "SchedulerOutput") -> tuple[int, int]:
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

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        layers = get_layers_from_vllm_config(
            self.vllm_config,
            (AttentionLayerBase, MambaBase),  # type: ignore[type-abstract]
        )
        block_size = self.vllm_config.cache_config.block_size
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype

        has_attention = any(
            isinstance(m, (Attention, MLAAttention)) for m in layers.values())
        has_mamba = any(isinstance(m, MambaBase) for m in layers.values())
        if has_attention and not self._unified_kv_layout:
            self._update_attention_page_size_padded(layers)
            if has_mamba:
                self._update_mamba_page_size_padded(layers)

        hma_enabled = (
            not self.scheduler_config.disable_hybrid_kv_cache_manager)

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            # Linear Attention path
            if isinstance(attn_module, MambaBase):
                spec = attn_module.get_kv_cache_spec(self.vllm_config)
                if spec is not None:
                    kv_cache_spec[layer_name] = spec
            # Classic Attention path
            elif isinstance(attn_module, Attention):
                if (kv_tgt_layer :=
                        attn_module.kv_sharing_target_layer_name) is not None:
                    if kv_tgt_layer not in layers:
                        raise ValueError(
                            f"Layer {layer_name} reuses KV cache from missing "
                            f"target layer {kv_tgt_layer}.")
                    target_module = layers[kv_tgt_layer]
                    if not isinstance(target_module, Attention):
                        raise ValueError(
                            f"Layer {layer_name} reuses KV cache from "
                            f"non-attention target layer {kv_tgt_layer}.")
                    if target_module.kv_sharing_target_layer_name is not None:
                        raise ValueError(
                            f"Layer {layer_name} reuses KV cache from "
                            f"{kv_tgt_layer}, which is itself a shared KV "
                            "layer.")
                    self._validate_shared_kv_cache_layout(
                        layer_name, attn_module, kv_tgt_layer, target_module,
                        hma_enabled)
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue

                if attn_module.attn_type == AttentionType.DECODER:
                    if isinstance(attn_module, ChunkedLocalAttention):
                        logger.warning_once(
                            "Using irope in Pallas is not supported yet, it "
                            "will fall back to global attention for long context."
                        )
                    page_size_padded = (
                        self._hybrid_uniform_page_size_bytes if
                        self._hybrid_uniform_page_size_bytes is not None else
                        PallasAttentionBackend.get_kv_cache_page_size_bytes(
                            block_size,
                            attn_module.num_kv_heads,
                            attn_module.head_size,
                            self.kv_cache_dtype,
                        ))
                    if attn_module.sliding_window is not None:
                        kv_cache_spec[layer_name] = SlidingWindowSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_dtype,
                            page_size_padded=page_size_padded,
                            sliding_window=attn_module.sliding_window,
                            indexes_kv_by_block_stride=True,
                        )
                    else:
                        kv_cache_spec[layer_name] = FullAttentionSpec(
                            block_size=block_size,
                            num_kv_heads=attn_module.num_kv_heads,
                            head_size=attn_module.head_size,
                            dtype=self.kv_cache_dtype,
                            page_size_padded=page_size_padded,
                            indexes_kv_by_block_stride=True,
                        )
                elif attn_module.attn_type in (
                        AttentionType.ENCODER,
                        AttentionType.ENCODER_ONLY,
                ):
                    # encoder-only attention does not need KV cache.
                    continue
                elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                    raise NotImplementedError
                else:
                    raise ValueError(
                        f"Unknown attention type: {attn_module.attn_type}")
            # MLAAttention path
            elif isinstance(attn_module, MLAAttention):
                if layer_name in kv_cache_spec:
                    continue
                page_size_padded = (
                    self._hybrid_uniform_page_size_bytes
                    if self._hybrid_uniform_page_size_bytes is not None else
                    PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
                        block_size,
                        1,
                        attn_module.head_size,
                        self.kv_cache_dtype,
                    ))
                kv_cache_spec[layer_name] = MLAAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    cache_dtype_str=cache_dtype_str,
                    page_size_padded=page_size_padded,
                    indexes_kv_by_block_stride=True,
                )
            else:
                continue

        # Note: that each shared layer's target actually owns a KV cache (and
        # that the shared layer was not itself allocated one) is enforced later
        # against the concrete KVCacheConfig in
        # `_maybe_add_kv_sharing_layers_to_kv_cache_groups`.
        return normalize_kv_cache_specs_for_tpu(
            kv_cache_spec,
            self.kv_cache_dtype,
            enable_unified_kv_layout=self._unified_kv_layout,
        )

    @staticmethod
    def _validate_shared_kv_cache_layout(
        layer_name: str,
        attn_module: Attention,
        target_layer_name: str,
        target_module: Attention,
        hma_enabled: bool,
    ) -> None:
        # When the hybrid KV cache manager is disabled, all specs are unified to
        # full attention: we store the full KV cache and apply each layer's own
        # window mask on read, so sliding_window doesn't affect storage and need
        # not match between the shared layer and its target. When the hybrid
        # manager is enabled (e.g. a mamba + sliding window + full attn model),
        # sliding-window layers get a smaller, window-sized cache, so a window
        # mismatch would mean the layers genuinely disagree on storage and the
        # share is unsafe -- enforce sliding_window in that case.
        fields = ["attn_type", "num_kv_heads", "head_size"]
        if hma_enabled:
            fields.append("sliding_window")
        mismatches = []
        for field_name in fields:
            if getattr(attn_module, field_name,
                       None) != getattr(target_module, field_name, None):
                mismatches.append(field_name)
        if mismatches:
            raise ValueError(f"Layer {layer_name} cannot reuse KV cache from "
                             f"{target_layer_name}: incompatible "
                             f"{', '.join(mismatches)}.")

        # With fp8 KV cache the shared layer reads the target's cached K/V and
        # dequantizes with its OWN scales, so mismatched scales/dtype would
        # silently corrupt attention.
        attn_quant = attn_module.impl.kv_cache_quantized_dtype
        target_quant = target_module.impl.kv_cache_quantized_dtype
        # Trigger the check if EITHER side is fp8: the asymmetric case (only one
        # side quantized) is just as unsafe -- e.g. the shared layer reads the
        # target's packed fp8 bytes as bf16 -- and must error rather than
        # silently corrupt, so None != "fp8_e4m3" correctly fails below.
        if attn_quant or target_quant:
            attn_kv_layout = (
                attn_module._k_scale_float,
                attn_module._v_scale_float,
                attn_quant,
            )
            target_kv_layout = (
                target_module._k_scale_float,
                target_module._v_scale_float,
                target_quant,
            )
            if attn_kv_layout != target_kv_layout:
                raise ValueError(
                    f"Layer {layer_name} reuses the KV cache of "
                    f"{target_layer_name} but their kv cache dtype or k/v "
                    f"scales differ (shared={attn_kv_layout}, "
                    f"target={target_kv_layout}); the shared layer would "
                    "read/dequantize the target's cached K/V incorrectly. "
                    "Cross-layer KV sharing requires a matching kv cache dtype "
                    "and matching k/v scales.")

    def _maybe_add_kv_sharing_layers_to_kv_cache_groups(
            self, kv_cache_config: KVCacheConfig) -> None:
        if not self.shared_kv_cache_layers:
            return
        group_layer_names = {
            layer_name
            for group in kv_cache_config.kv_cache_groups
            for layer_name in group.layer_names
        }
        allocated_layer_names = {
            layer_name
            for kv_cache_tensor in kv_cache_config.kv_cache_tensors
            for layer_name in kv_cache_tensor.shared_by
        }
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
        ):
            if target_layer_name not in group_layer_names:
                raise ValueError(
                    f"Layer {layer_name} reuses KV cache from "
                    f"{target_layer_name}, but the target layer is missing "
                    "from KV cache groups.")
            if (layer_name in group_layer_names
                    or layer_name in allocated_layer_names):
                raise ValueError(
                    f"Shared KV layer {layer_name} must not have an "
                    "independent KV cache allocation.")
        runner_only_attn_layers = self.runner_only_attn_layers
        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            runner_only_attn_layers,
        )

    def _add_shared_kv_cache_aliases(
            self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not self.shared_kv_cache_layers:
            return
        for layer_name in self.shared_kv_cache_layers:
            if layer_name in kv_caches:
                raise ValueError(
                    f"Shared KV layer {layer_name} was allocated its own KV "
                    "cache.")
        # `get_kv_cache_spec` already rejects a target that is itself a shared
        # layer, so the mapping is a flat shared -> owner relation (no chains or
        # cycles) and can be aliased directly.
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
        ):
            if target_layer_name not in kv_caches:
                raise ValueError(
                    f"Layer {layer_name} reuses KV cache from "
                    f"{target_layer_name}, but the target cache was not "
                    "allocated.")
            kv_caches[layer_name] = kv_caches[target_layer_name]

    def _update_attention_page_size_padded(
            self, layers: dict[str, AttentionLayerBase]) -> None:
        """Pad attention page sizes so vLLM's num_blocks matches what
        the TPU allocates per layer.

        If every attention layer already has the same natural TPU page
        size, there's nothing to compensate for: leave
        `mamba_page_size_padded` as a no-op default.

        If they differ, vLLM's own `unify_kv_cache_spec_page_size` will
        rescale the smaller layer's `block_size` to match but leaves
        its `page_size_padded` at the old, now-too-small
        value, which trips `AttentionSpec.page_size_bytes`'s own
        `page_size_padded >= real_page_size_bytes` assertion. Instead,
        pin every layer's `page_size_padded` to the max natural size
        up front (mirroring the Mamba-hybrid case below), so all
        layers already report equal `page_size_bytes` and
        `unify_kv_cache_spec_page_size` is a no-op.

        NOTE: Cannot leave `page_size_padded=None` on every layer, because
        vLLM core uses `page_size_padded` to size `num_blocks` against the HBM
        budget. With `page_size_padded=None`, that sizing falls back to
        generic page-size formula (https://github.com/vllm-project/vllm/blob/48aa8d8d7529d2314858d8487cc0a21789fc7ec1/vllm/v1/kv_cache_interface.py#L204-L218)
        which has no notion of TPU's packing/alignment rules (e.g.
        fp8 packs 4 elements per 32-bit lane, so `num_kv_heads * 2` gets
        rounded up to a multiple of 4 in the actual TPU tensor -- see
        `get_kv_cache_shape`). Where that rounding changes the byte count --
        e.g. a GQA layer TP-sharded down to a single KV head -- the real TPU
        allocation ends up larger than vLLM's generic formula accounted for,
        so `num_blocks` is silently oversized and HBM usage silently exceeds
        `gpu_memory_utilization`'s target, instead of failing loudly the way
        this assertion does. `page_size_padded` has to stay pinned to the
        true, packing-aware size on every layer; this function's job is only
        to make sure every layer is pinned to the *same* one.
        """
        attn_page_sizes = set()
        for m in layers.values():
            if isinstance(m, Attention):
                attn_page_sizes.add(
                    PallasAttentionBackend.get_kv_cache_page_size_bytes(
                        self.block_size,
                        m.num_kv_heads,
                        m.head_size,
                        self.kv_cache_dtype,
                    ))
            elif isinstance(m, MLAAttention):
                attn_page_sizes.add(
                    PallasMLAttentionBackend.get_kv_cache_page_size_bytes(
                        self.block_size,
                        1,
                        m.head_size,
                        self.kv_cache_dtype,
                    ))

        if not attn_page_sizes:
            return
        elif len(attn_page_sizes) > 1:
            uniform_page_size_bytes = max(attn_page_sizes)
            self._hybrid_uniform_page_size_bytes = uniform_page_size_bytes
            self.cache_config.mamba_page_size_padded = uniform_page_size_bytes
            logger.info(
                "Pure-attention hybrid KV cache: padding every layer "
                "spec to %d bytes (max of native sizes %s). Avoids "
                "vLLM's unify_kv_cache_spec_page_size leaving a stale "
                "page_size_padded behind when layer head_dims differ.",
                uniform_page_size_bytes, sorted(attn_page_sizes))
        else:
            self.cache_config.mamba_page_size_padded = attn_page_sizes.pop()

    def _update_mamba_page_size_padded(
            self, layers: dict[str, AttentionLayerBase]) -> None:
        """Pad attention and mamba page sizes so vLLM's num_blocks matches
        what the TPU allocates per layer.

        For hybrid attention+mamba models, vLLM groups a tensor's memory so
        that one `KVCacheTensor` is `shared_by` one layer from each kv-cache
        group (e.g., Qwen3.5: 1 full-attn + 3 linear-attn per shared_by).
        vLLM's scheduler assumes these layers share a single physical
        tensor at the byte level — each layer's block_table indexes into
        disjoint slots of the same backing allocation, and device kernels
        reinterpret the bytes as attention KV or mamba state depending on
        which layer is accessing the slot.

        TPU `jax.Array`s are strongly typed, so we cannot overlay an
        attention tensor and a mamba tensor on the same bytes.
        `initialize_kv_cache` therefore allocates one physical array per
        layer in the `shared_by` group, carving the group's byte budget
        into separate per-layer tensors. Without the compensation done
        here, vLLM's block pool would hold `num_shared_layers`× more
        block IDs than each per-layer array has slots — the scheduler
        would hand out block IDs beyond a layer's leading dimension,
        JAX's indexed writes would silently clip them, and multiple
        requests' mamba recurrent states would collapse onto the same
        slot (corrupted state → gibberish generation).

        The fix: set every layer's reported `page_size_padded` equal to the
        full per-`shared_by` footprint — `num_attn_groups × attn_page +
        num_mamba_groups × mamba_unpadded`, where `attn_page` is the
        TPU-actual per-block bytes (from `get_attention_page_size_bytes`,
        which accounts for dtype packing like fp8) and `mamba_unpadded` is
        the natural `prod(shape) × dtype_size`. vLLM then computes a
        smaller `num_blocks` that exactly matches what we allocate per layer
        on the TPU side. HBM usage is unchanged; only the block-ID
        accounting lines up.

        Args:
            layers: A dictionary mapping layer names to their corresponding
                attention module instances (e.g., `MambaBase`, `Attention`).
        """
        attn_modules = [
            m for m in layers.values()
            if isinstance(m, (Attention, MLAAttention))
        ]
        if not attn_modules:
            return

        first_attn_module = attn_modules[0]
        num_kv_heads = first_attn_module.num_kv_heads if isinstance(
            first_attn_module, Attention) else 1
        attention_backend = (PallasMLAttentionBackend if isinstance(
            first_attn_module, MLAAttention) else PallasAttentionBackend)
        attn_page_size_bytes = attention_backend.get_kv_cache_page_size_bytes(
            self.block_size, num_kv_heads, first_attn_module.head_size,
            self.kv_cache_dtype)

        mamba_modules = [
            m for m in layers.values() if isinstance(m, MambaBase)
        ]
        if not mamba_modules:
            # Not hybrid; set `mamba_page_size_padded` to the attention
            # page size as a no-op default (vLLM's platform interface sets
            # this too when it detects hybrid). No layer duplication will
            # happen without mamba layers, so no block-ID mismatch to fix.
            self.cache_config.mamba_page_size_padded = attn_page_size_bytes
            return

        # Compute the unpadded mamba page size from an actual mamba module's
        # spec (shapes × dtype-size), ignoring any existing padding.
        first_mamba_spec = mamba_modules[0].get_kv_cache_spec(self.vllm_config)
        assert isinstance(first_mamba_spec, MambaSpec)
        unpadded_mamba_page_size = dataclasses.replace(
            first_mamba_spec, page_size_padded=None).page_size_bytes

        # Derive vLLM's kv-cache group layout. vLLM splits each type into
        # equal-sized groups of `group_size` layers, then allocates
        # `group_size` `KVCacheTensor`s, each `shared_by` one layer from
        # every group — so each tensor covers `num_attn_groups +
        # num_mamba_groups` layers.
        #
        # Choosing `group_size` trades off padding vs. number of groups:
        #   * group_size = max_count → fewer groups (often 1 per type),
        #     but the smaller side pads its group up to max_count layers
        #     (wastes space if max ≫ min).
        #   * group_size = min_count → no padding, but the larger side
        #     splits into `ceil(max/min)` groups.
        # vLLM's rule: pick max_count only when counts are close enough
        # that the padding is minor (max < 1.5 × min), else min_count.
        #   e.g. 12 sliding-window + 13 full-attn → max (1 group each)
        #   e.g. 10 full-attn      + 30 mamba     → min (1 attn + 3 mamba)
        #
        # This duplicates the heuristic from
        # `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_groups_uniform_page_size`.
        # We can't call it directly because vLLM's grouping needs a fully
        # populated spec dict, while we need the group layout *before* we
        # can finish creating the specs (padding depends on grouping,
        # spec creation depends on padding). Keep in sync if that
        # heuristic ever changes — it has been stable since the hybrid
        # allocator landed.
        num_attn = len(attn_modules)
        num_mamba = len(mamba_modules)
        min_count = min(num_attn, num_mamba)
        max_count = max(num_attn, num_mamba)

        # Match vLLM exactly: float comparison, no int() truncation (matters
        # at e.g. min=3, max=4, where 4 < 4.5 but 4 < int(4.5)==4 differs).
        if max_count < min_count * 1.5:
            group_size = max_count
        else:
            group_size = min_count

        num_attn_groups = (num_attn + group_size - 1) // group_size
        num_mamba_groups = (num_mamba + group_size - 1) // group_size

        uniform_page_size_bytes = (num_attn_groups * attn_page_size_bytes +
                                   num_mamba_groups * unpadded_mamba_page_size)

        logger.info(
            "Hybrid KV cache: padding every layer spec to %d bytes "
            "(num_attn_groups=%d × attn_page=%d + "
            "num_mamba_groups=%d × mamba_unpadded=%d). This makes vLLM's "
            "num_blocks match per-layer TPU allocation when mamba layers "
            "cannot be truly shared.", uniform_page_size_bytes,
            num_attn_groups, attn_page_size_bytes, num_mamba_groups,
            unpadded_mamba_page_size)

        self._hybrid_uniform_page_size_bytes = int(uniform_page_size_bytes)
        self.cache_config.mamba_page_size_padded = int(uniform_page_size_bytes)

        # Prefer compact-mamba sizing: cap each mamba layer at
        # `max_num_reqs + 1` recurrent slots and give the freed HBM to the
        # attention pool. Mamba state is recurrent — one slot per active
        # request — so the uniform layout wastes `num_blocks - max_num_reqs`
        # mamba slots forever. On success this pins
        # `num_gpu_blocks_override` to the (larger) attention block count and
        # sets `_mamba_num_blocks`.
        self._maybe_set_compact_mamba_num_blocks_override(
            attn_page_size_bytes, int(unpadded_mamba_page_size),
            num_attn_groups, num_mamba_groups, group_size)

        # Fallback (compact sizing skipped, e.g. CPU-only tests or a
        # user-pinned num_gpu_blocks_override): pin vLLM's num_blocks via the two-step
        # flooring that keeps peak HBM within `gpu_memory_utilization ×
        # total_hbm` at high utilization. See `_maybe_set_num_blocks_override`
        # for the formula and rationale; the short version is that vLLM's
        # single-step `floor(avail / (uniform × group_size))` can land one
        # block higher than the two-step value, and that extra block ×
        # group_size × uniform bytes is enough to push past the budget against
        # imprecision in vLLM's `avail` estimate.
        if self._mamba_num_blocks is None:
            self._maybe_set_num_blocks_override(attn_page_size_bytes,
                                                int(uniform_page_size_bytes),
                                                group_size)

    def _available_kv_cache_hbm(self) -> int:
        """KV-cache HBM budget for the block-count override paths.

        Matches `TPUWorker.determine_available_memory()`: the
        `utils.compute_hbm_budget` result (which reserves the
        `gpu_memory_utilization` cap minus
        `utils.estimate_kv_connector_hbm_reserve` (HBM a connector allocates
        after profile_run, such as the offload H2D
        staging pool). Sizing overrides against this budget keeps them from
        filling the connector reserve back up with KV blocks and defeating
        the worker-side subtraction.
        """
        budget = utils.compute_hbm_budget(
            [self.device], self.cache_config.gpu_memory_utilization)
        return budget.available - utils.estimate_kv_connector_hbm_reserve(
            self.vllm_config)

    def _maybe_set_compact_mamba_num_blocks_override(
            self, attn_page_size_bytes: int,
            unpadded_mamba_page_size_bytes: int, num_attn_groups: int,
            num_mamba_groups: int, group_size: int) -> None:
        """Cap mamba layers at `max_num_reqs + 1` recurrent slots and pin
        `cache_config.num_gpu_blocks_override` so the freed HBM grows the
        attention pool.

        Tradeoff vs. the uniform num_blocks layout
        ------------------------------------------
        Mamba state is recurrent: one slot per *active* request, regardless of
        context length. The uniform layout (set by
        `_maybe_set_num_blocks_override`) gives every layer the same
        `num_blocks`, leaving `num_blocks - max_num_reqs` mamba slots idle
        forever. The compact layout caps mamba at `max_num_reqs + 1` (the `+1`
        is the null/sentinel slot), which is strictly better for any model
        where `num_blocks > max_num_reqs` — i.e. all production hybrid configs
        we run. Cost: the GDN op must index mamba state by per-request slot id
        (`AttentionMetadata.mamba_state_indices`) rather than by
        `block_tables[:, 0]`, since the mamba leading dim is now smaller than
        the attention pool.

        Sizing math
        -----------
        vLLM allocates `group_size` KVCacheTensors, each shared across
        `num_attn_groups` attention layers + `num_mamba_groups` mamba layers.
        With per-tensor budget `B = avail / group_size` and
        `N_mamba = max_num_reqs + 1`,
            N_attn = floor((B - num_mamba_groups × N_mamba × mamba_unpadded)
                            / (num_attn_groups × attn_page)).

        We do NOT round to a sharding divisor: our single-process
        TP layout does not shard the mamba/attention block (leading) dim — TP
        sharding is on the head dims via the model — so any positive block
        count is valid.

        Args:
            attn_page_size_bytes: TPU-actual bytes per block per attention
                layer (accounts for dtype packing like fp8).
            unpadded_mamba_page_size_bytes: bytes per slot per mamba layer
                (`prod(shape) × dtype_size`, no padding).
            num_attn_groups: # attention layers backed by each KVCacheTensor.
            num_mamba_groups: # mamba layers backed by each KVCacheTensor.
            group_size: # KVCacheTensors vLLM allocates (= layers per
                kv-cache group); the same value passed to
                `_maybe_set_num_blocks_override`.

        On success: sets `cache_config.num_gpu_blocks_override` (attention
        block count) and `_mamba_num_blocks`. Raises when the KV budget cannot
        hold the mamba slots; the remaining precondition-fail paths leave both
        unset so the caller falls back to uniform sizing.
        """
        if self._uniform_mamba_layout:
            logger.info("Compact mamba sizing skipped.")
            return
        cache_config = self.cache_config
        if cache_config.num_gpu_blocks_override is not None:
            return
        if group_size <= 0:
            return

        avail = self._available_kv_cache_hbm()
        if avail <= 0:
            return

        # +1 reserves slot 0 as the null/sentinel block (never handed to a
        # request); padded tail positions in `mamba_state_indices` also point
        # here, so their writes can never corrupt an active request's state.
        # With speculative decoding each request owns a *group* of
        # `num_spec + 1` consecutive slots so the GDN kernel can checkpoint
        # the state after every speculative window position (see
        # `_init_mamba_slot_pool` for the rollback scheme).
        mamba_num_blocks = self.max_num_reqs * self._mamba_slot_stride + 1

        avail_per_tensor = avail // group_size
        mamba_per_tensor = (num_mamba_groups * mamba_num_blocks *
                            unpadded_mamba_page_size_bytes)
        # Falling back to the uniform layout here would pad every block to the
        # mamba page and silently shrink the pool by ~50x, which surfaces as a
        # throughput collapse rather than a misconfiguration. Fail loudly.
        attn_per_tensor_avail = avail_per_tensor - mamba_per_tensor
        if attn_per_tensor_avail <= 0:
            raise ValueError(
                f"Compact-mamba KV sizing does not fit: mamba slots alone need "
                f"{mamba_per_tensor} B per KVCacheTensor (mamba_num_blocks="
                f"{mamba_num_blocks} x num_mamba_groups={num_mamba_groups} x "
                f"mamba_unpadded={unpadded_mamba_page_size_bytes}), but the "
                f"per-tensor KV budget is {avail_per_tensor} B. Raise "
                f"`gpu_memory_utilization` or lower `max_num_seqs`.")

        attn_num_blocks = attn_per_tensor_avail // (num_attn_groups *
                                                    attn_page_size_bytes)
        if attn_num_blocks <= 0:
            raise ValueError(
                f"Compact-mamba KV sizing does not fit: no attention blocks "
                f"remain (avail_per_tensor={avail_per_tensor} B, "
                f"mamba_per_tensor={mamba_per_tensor} B). Raise "
                f"`gpu_memory_utilization` or lower `max_num_seqs`.")

        cache_config.num_gpu_blocks_override = int(attn_num_blocks)
        self._mamba_num_blocks = int(mamba_num_blocks)

        # Total HBM = group_size KVCacheTensors, each holding num_attn_groups
        # attention blocks + num_mamba_groups mamba slots.
        attn_bytes = (group_size * num_attn_groups * attn_num_blocks *
                      attn_page_size_bytes)
        mamba_bytes = (group_size * num_mamba_groups * mamba_num_blocks *
                       unpadded_mamba_page_size_bytes)
        logger.info(
            "Compact-mamba KV cache: num_gpu_blocks_override=%d (attn), "
            "_mamba_num_blocks=%d. HBM split: attn=%.2f GiB; "
            "mamba=%.2f GiB; total=%.2f GiB / avail=%.2f GiB.",
            attn_num_blocks, mamba_num_blocks, attn_bytes / (2**30),
            mamba_bytes / (2**30), (attn_bytes + mamba_bytes) / (2**30),
            avail / (2**30))

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
        self._free_mamba_slots = [
            1 + g * stride for g in reversed(range(num_groups))
        ]
        self._has_mamba_state = True

    def _build_mamba_state_indices(self, start_index: int, num_reqs: int,
                                   target_num_reqs: int) -> torch.Tensor:
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
                self._free_mamba_slots.append(
                    self._mamba_slot_by_req_id.pop(req_id))

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
        layer_to_raw: dict[str, torch.Tensor] = {}
        for raw, kv_cache_tensor in zip(raw_tensors,
                                        kv_cache_config.kv_cache_tensors):
            for layer_name in kv_cache_tensor.shared_by:
                layer_to_raw[layer_name] = raw
        plan: list[tuple[int, list[torch.Tensor]]] = []
        state_block_size: int | None = None
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
        self._mamba_state_block_size = state_block_size
        self._mamba_copy_plan = plan
        for raw in raw_tensors:
            if raw.dim() > 1:
                self._pool_block_split = (self.cache_config.block_size //
                                          raw.shape[1])
                break

    def _collect_mamba_state_seed_copies(self, scheduler_output,
                                         start_index: int,
                                         num_reqs: int) -> None:
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
            cp_world_size = (get_dcp_group().world_size *
                             get_pcp_group().world_size)
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
            prev = self._mamba_state_pos.get(req_id,
                                             (computed - 1) // col_stride)
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

        per_raw: dict[int, tuple[torch.Tensor, list[int], list[int]]] = {}
        per_raw_dev: dict[int, tuple[torch.Tensor, list[torch.Tensor],
                                     list[torch.Tensor]]] = {}
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
                dst_t = torch.tensor(dsts_g,
                                     dtype=torch.int32).to(self.device,
                                                           non_blocking=True)
                src_t, dst_t = self._expand_pool_split(src_t, dst_t)
                for raw in raws:
                    entry = per_raw_dev.setdefault(id(raw), (raw, [], []))
                    entry[1].append(src_t)
                    entry[2].append(dst_t)
                continue
            offset_pairs.extend(pairs)
            if self._pool_block_split > 1:
                # The pool is born at kernel granularity: a manager state
                # block is `split` consecutive pool blocks.
                pairs = [(s * self._pool_block_split + j,
                          d * self._pool_block_split + j) for s, d in pairs
                         for j in range(self._pool_block_split)]
            for raw in raws:
                entry = per_raw.setdefault(id(raw), (raw, [], []))
                entry[1].extend(p[0] for p in pairs)
                entry[2].extend(p[1] for p in pairs)

        if spec_seed and spec_dsts:
            # The seeded checkpoint is checkpoint 0 of the new group.
            dst_t = torch.tensor(self._pad_to_bucket(spec_dsts),
                                 dtype=torch.long).to(self.device,
                                                      non_blocking=True)
            assert self.mamba_slot_read_offsets is not None
            self.mamba_slot_read_offsets.index_put_(
                (dst_t, ),
                torch.zeros(dst_t.shape[0],
                            dtype=self.mamba_slot_read_offsets.dtype,
                            device=self.mamba_slot_read_offsets.device))
        elif self.mamba_slot_read_offsets is not None and offset_pairs:
            # Migrate the read offsets with the state: gather at the old
            # blocks, scatter at the new ones (both device-side, before the
            # forward that reads the buffer). Padding with (0, 0) null-block
            # self-copies keeps the shapes on the seed-copy bucket ladder.
            offset_pairs = self._pad_to_bucket(offset_pairs, pad=(0, 0))
            src_t = torch.tensor([s for s, _ in offset_pairs],
                                 dtype=torch.long).to(self.device,
                                                      non_blocking=True)
            dst_t = torch.tensor([d for _, d in offset_pairs],
                                 dtype=torch.long).to(self.device,
                                                      non_blocking=True)
            self.mamba_slot_read_offsets.index_put_(
                (dst_t, ), self.mamba_slot_read_offsets[src_t])

        for raw, srcs_dev, dsts_dev in per_raw_dev.values():
            src_t = self._pad_dev_to_bucket(torch.cat(srcs_dev))
            dst_t = self._pad_dev_to_bucket(torch.cat(dsts_dev))
            self._pending_mamba_state_copies.append((raw, src_t, dst_t))

        for raw, srcs, dsts in per_raw.values():
            src_t = torch.tensor(self._pad_to_bucket(srcs),
                                 dtype=torch.int32).to(self.device,
                                                       non_blocking=True)
            dst_t = torch.tensor(self._pad_to_bucket(dsts),
                                 dtype=torch.int32).to(self.device,
                                                       non_blocking=True)
            self._pending_mamba_state_copies.append((raw, src_t, dst_t))

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
        return torch.cat([
            ids,
            torch.zeros(padded - n, dtype=ids.dtype, device=ids.device),
        ])

    def _expand_pool_split(
            self, src: torch.Tensor,
            dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Device twin of the manager-block -> pool-block expansion.

        The pool is born at kernel granularity: a manager state block is
        `split` consecutive pool blocks.
        """
        split = self._pool_block_split
        if split <= 1:
            return src, dst
        j = torch.arange(split, dtype=src.dtype, device=src.device)
        return ((src.unsqueeze(1) * split + j).reshape(-1),
                (dst.unsqueeze(1) * split + j).reshape(-1))

    def _spec_seed_sources(self, state_srcs: list[int],
                           ckpt_rows: list[list[int]]) -> torch.Tensor:
        """Seed source per crossing: the checkpoint being resumed from.

        `mamba_slot_read_offsets` is device-resident and indexed by block
        id, so which of the window's blocks holds the last accepted token's
        state is only known on device — the choice is a gather there rather
        than a CPU lookup.
        """
        offsets = self.mamba_slot_read_offsets
        assert offsets is not None
        state_t = torch.tensor(state_srcs,
                               dtype=torch.long).to(self.device,
                                                    non_blocking=True)
        ckpt_t = torch.tensor(ckpt_rows,
                              dtype=torch.int32).to(self.device,
                                                    non_blocking=True)
        off = offsets[state_t].long().clamp_(0, ckpt_t.shape[1] - 1)
        return ckpt_t.gather(1, off.unsqueeze(1)).squeeze(1)

    def _flush_mamba_state_seed_copies(self) -> None:
        if not self._pending_mamba_state_copies:
            return
        for raw, src_t, dst_t in self._pending_mamba_state_copies:
            copy_mamba_state_blocks(raw, src_t, dst_t)
        self._pending_mamba_state_copies.clear()

    def _maybe_set_num_blocks_override(self, attn_page_size_bytes: int,
                                       uniform_page_size_bytes: int,
                                       group_size: int) -> None:
        """Pin `cache_config.num_gpu_blocks_override` to the two-step
        flooring value that keeps peak HBM within the user-set
        `gpu_memory_utilization` budget at high utilization.

        Formula:
          `num_blocks_attn = floor(avail / (attn_page × group_size))`
          `num_blocks_tpu  = floor(attn_page × num_blocks_attn / uniform)`

        The two-step flooring can land 1 block lower than vLLM's
        single-step `floor(avail / (uniform × group_size))`. Since
        `uniform > attn_page`, that 1-block gap costs `group_size × uniform`
        bytes of HBM, which — against the imprecision in vLLM's `avail`
        estimate — is enough to tip high-utilization configurations into
        OOM. Pinning to `num_blocks_tpu` preserves the headroom the
        single-step formula silently removes.

        `avail` comes from `_available_kv_cache_hbm`, so the block count
        pinned here matches the KV-cache budget
        `TPUWorker.determine_available_memory()` hands vLLM.

        Skipped only if the user has explicitly set `num_gpu_blocks_override`.

        Args:
            attn_page_size_bytes: TPU-actual bytes per block for one
                attention layer, from `get_attention_page_size_bytes`
                (accounts for dtype packing like fp8).
            uniform_page_size_bytes: bytes per block for one `KVCacheTensor`
                shared across `num_attn_groups + num_mamba_groups` layers
                (the `_hybrid_uniform_page_size_bytes` value set above).
            group_size: number of layers per vLLM kv-cache group, used by
                vLLM to compute `num_blocks` from the attention tensor size.

        Returns:
            None. Side effect: sets `cache_config.num_gpu_blocks_override`
            if all preconditions hold; otherwise leaves it unset.
        """
        cache_config = self.cache_config
        if cache_config.num_gpu_blocks_override is not None:
            return

        avail = self._available_kv_cache_hbm()
        if avail <= 0:
            return

        naive_vllm_num_blocks = avail // (attn_page_size_bytes * group_size)
        if naive_vllm_num_blocks <= 0:
            return
        naive_tensor_size = attn_page_size_bytes * naive_vllm_num_blocks
        num_blocks_tpu = naive_tensor_size // uniform_page_size_bytes
        if num_blocks_tpu <= 0:
            return

        cache_config.num_gpu_blocks_override = int(num_blocks_tpu)
        logger.info(
            "Hybrid KV cache: setting num_gpu_blocks_override=%d to align "
            "the scheduler's block pool with per-layer TPU allocation "
            "(avail=%d, naive_vllm_num_blocks=%d).", num_blocks_tpu, avail,
            naive_vllm_num_blocks)

    def _prepare_async_token_substitution_indices(
        self, start_index: int, num_reqs: int,
        num_scheduled_tokens_per_req: np.ndarray
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
        if (self.speculative_config is not None
                and self._pre_async_results.spec_decode_num_rejected_tokens
                is not None):
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
            src_start = (self._pre_async_results.req_id_to_index_copy[req_id] *
                         stride)
            for j in range(n_sched):
                request_major_index = acc_cur_len - n_sched + j
                if layout_plan is None:
                    cur_input_index = request_major_index
                else:
                    cur_input_index = (
                        layout_plan.local_index_for_request_major_token(
                            request_major_index))
                    if cur_input_index is None:
                        continue
                token_in_tpu_cur_input_indices_list.append(cur_input_index)
                token_in_tpu_pre_next_tokens_indices_list.append(src_start + j)

        if len(token_in_tpu_cur_input_indices_list) > 0:
            return (np.array(token_in_tpu_cur_input_indices_list,
                             dtype=np.int32),
                    np.array(token_in_tpu_pre_next_tokens_indices_list,
                             dtype=np.int32))
        else:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    def _apply_async_token_substitution(
            self, input_ids: torch.Tensor,
            token_in_tpu_cur_input_indices: np.ndarray,
            token_in_tpu_pre_next_tokens_indices: np.ndarray) -> torch.Tensor:
        """Apply async token substitution if needed."""
        if len(token_in_tpu_cur_input_indices) == 0:
            return input_ids

        idx_pad_len = len(input_ids) - len(token_in_tpu_cur_input_indices)

        # Pad according to the instructions written inside
        # _substitute_placeholder_token: missing = the complement of
        # cur_input_indices over [0, len(input_ids)). O(n) boolean mask instead
        # of np.setdiff1d's O(n log n) sort; flatnonzero returns the same
        # ascending order setdiff1d did over the (sorted, unique) full range.
        _missing_mask = np.ones(len(input_ids), dtype=bool)
        _missing_mask[token_in_tpu_cur_input_indices] = False
        missing_values = np.flatnonzero(_missing_mask).astype(np.int32)
        padded_token_in_tpu_cur_input_indices = np.concatenate(
            (token_in_tpu_cur_input_indices, missing_values))

        padded_token_in_tpu_pre_next_tokens_indices = np.pad(
            token_in_tpu_pre_next_tokens_indices, (0, idx_pad_len),
            mode='constant',
            constant_values=-1)

        cur_input_indices = torch.from_numpy(
            padded_token_in_tpu_cur_input_indices).to(self.device,
                                                      non_blocking=True)
        pre_next_tokens_indices = torch.from_numpy(
            padded_token_in_tpu_pre_next_tokens_indices).to(self.device,
                                                            non_blocking=True)

        return _substitute_placeholder_token(
            input_ids, cur_input_indices, pre_next_tokens_indices,
            self._pre_async_results.next_tokens_tpu)

    def _flush_disjoint_async_results(self) -> None:
        if self._pre_async_results is None:
            return
        current_req_ids = set(
            self.input_batch.req_ids[:self.input_batch.num_reqs])
        previous_req_ids = self._pre_async_results.req_id_to_index_copy
        if any(req_id in current_req_ids for req_id in previous_req_ids):
            return
        self._modify_prev_results()
        self._pre_async_results = None

    def _update_states(self, scheduler_output: "SchedulerOutput") -> Any:
        # Disable vLLM's native GPU async spec-decode num_computed_tokens
        # correction. The TPU runner does its own async rejection correction,
        # so letting the base also correct would double-count.
        for req_state in self.requests.values():
            req_state.prev_num_draft_len = 0
        return super()._update_states(scheduler_output)

    def _modify_prev_results(self):
        if self._pre_async_results is None:
            return

        pre_req_ids = self._pre_async_results.req_ids
        pre_request_seq_lens = self._pre_async_results.request_seq_lens
        pre_discard_sampled_tokens_req_indices = self._pre_async_results.discard_sampled_tokens_req_indices
        # Per-request draft count from the previous step (keyed by the
        # request_seq_lens index); sizes the optimistic-placeholder rollback
        # below. None/absent -> 0 drafts (non-spec or a non-spec mixed chunk).
        pre_num_draft_per_req = self._pre_async_results.num_draft_per_req

        pre_next_tokens_cpu = self._pre_async_results.wait_for_copy()
        assert pre_next_tokens_cpu is not None

        pre_next_tokens_cpu = pre_next_tokens_cpu[:len(pre_req_ids)]
        max_gen_len = pre_next_tokens_cpu.shape[-1]

        if max_gen_len == 1:
            valid_sampled_token_ids = pre_next_tokens_cpu.tolist()
        else:
            valid_mask = pre_next_tokens_cpu != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in pre_next_tokens_cpu[valid_mask].split(gen_lens)
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
            n_placeholder = 1 + (pre_num_draft_per_req.get(pre_req_idx, 0)
                                 if pre_num_draft_per_req else 0)
            # At most (1 bonus + num_draft) tokens can commit this step, so the
            # optimistic placeholder count must cover the real sampled count.
            # num_sampled_tokens <= pre_num_placeholder_tokens guard.
            assert len(sampled_ids) <= n_placeholder, (
                f"req {req_id}: sampled {len(sampled_ids)} tokens > "
                f"{n_placeholder} optimistic placeholders -- output_token_ids "
                "rollback would drop the wrong elements")
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
            self.input_batch.num_tokens_no_spec[req_idx] = (start_idx +
                                                            len(sampled_ids))

            target_slice = slice(seq_len - len(sampled_ids) + 1, seq_len + 1)
            # The committed tokens must fit within token_ids_cpu
            # ([num_reqs, max_model_len]); writing past the end is a sizing bug.
            assert seq_len + 1 <= self.input_batch.token_ids_cpu.shape[1], (
                f"req {req_id}: write end {seq_len + 1} exceeds max_model_len "
                f"{self.input_batch.token_ids_cpu.shape[1]}")
            self.input_batch.token_ids_cpu[req_idx, target_slice] = sampled_ids

    def _update_placeholder(self,
                            discard_sampled_tokens_req_indices,
                            request_seq_lens,
                            next_token_indices: dict[int, int],
                            num_draft_per_req: dict[int, int] | None = None):
        placeholder_req_id_to_index: dict[str, int] = {}
        discard_set = set(discard_sampled_tokens_req_indices)
        for req_idx, req_state, seq_len, req_id in request_seq_lens:
            if req_idx in discard_set:
                continue

            # Async spec: optimistically advance by 1 (bonus) + num_draft;
            # the over-count is corrected on-device next step by subtract_num_rejected_tokens.
            n_new = 1 + (num_draft_per_req.get(req_idx, 0)
                         if num_draft_per_req else 0)
            end_idx = seq_len + n_new
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_state.output_token_ids.extend([0] * n_new)

            next_token_index = next_token_indices[req_idx]
            placeholder_req_id_to_index[req_state.req_id] = next_token_index

        return placeholder_req_id_to_index

    def _assemble_async_spec_substitution(self, drafts, next_tokens_per_chunk,
                                          state):
        """Async-spec substitution assembler.

        From the already-proposed on-device ``drafts`` (the unified eagle3
        propose call in ``sample_tokens``), builds the per-request
        ``[bonus, draft_1..K]`` substitution source and the stride-``1+K``
        ``next_token_indices``, and computes the per-request rejected count.
        Returns ``(next_tokens_tpu_chunks, next_token_indices,
        spec_decode_num_rejected_tokens, num_draft_per_req)``.
        """
        from vllm_torchtpu.runner.tpu_runner_async_output import (
            assemble_spec_next_tokens, compute_num_rejected)
        assert drafts is not None, (
            "async spec substitution requires the eagle3 drafts proposed "
            "earlier in sample_tokens")
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
        for nt_chunk, num_reqs, md in zip(next_tokens_per_chunk,
                                          state.num_reqs_list,
                                          state.spec_decode_metadata_list):
            if md is not None:
                drafts_chunk = drafts[req_offset:req_offset + num_reqs]
                src = assemble_spec_next_tokens(nt_chunk, drafts_chunk,
                                                num_reqs)  # [num_reqs*(1+K)]
                num_rejected_chunks.append(
                    compute_num_rejected(nt_chunk,
                                         md.draft_lengths[:num_reqs]))
                for r in range(num_reqs):
                    num_draft_per_req[req_offset + r] = int(
                        md.draft_lengths_cpu[r])
            else:
                # Mixed-batch non-spec chunk: pad the single sampled token to the
                # uniform 1+K span (token at slot 0, rest INVALID).
                token = nt_chunk[:num_reqs, 0]
                src = torch.full((num_reqs, 1 + K),
                                 INVALID_TOKEN_ID,
                                 dtype=token.dtype,
                                 device=token.device)
                src[:, 0] = token
                src = src.reshape(-1)  # [num_reqs*(1+K)]
                num_rejected_chunks.append(
                    torch.zeros(num_reqs,
                                dtype=torch.int32,
                                device=token.device))
            next_tokens_tpu_chunks.append(src)
            for r in range(num_reqs):
                next_token_indices[req_offset + r] = req_offset + r
            req_offset += num_reqs
        spec_num_rejected = (torch.cat(num_rejected_chunks)
                             if num_rejected_chunks else None)
        return (next_tokens_tpu_chunks, next_token_indices, spec_num_rejected,
                num_draft_per_req)

    def _assemble_async_prefill_bootstrap(self, drafts,
                                          combined_selected_tokens,
                                          combined_real_lens):
        """Async eagle3 bootstrap assembler for the prefill / pure-non-spec step.

        From the already-proposed prompt-context ``drafts`` (the unified eagle3
        propose call in ``sample_tokens``, device_seed path), parks a
        stride-(1+K) `[bonus, draft_1..K]` source so the NEXT step substitutes
        REAL prompt-context drafts. Returns ``(next_tokens_tpu_chunks,
        next_token_indices, spec_num_rejected=zeros, num_draft_per_req)``.
        """
        assert drafts is not None, (
            "async prefill bootstrap requires the eagle3 drafts proposed "
            "earlier in sample_tokens")
        K = self.speculative_config.num_speculative_tokens
        next_tokens_tpu_chunks: list[torch.Tensor] = []
        next_token_indices: dict[int, int] = {}
        num_rejected_chunks: list[torch.Tensor] = []
        num_draft_per_req: dict[int, int] = {}
        req_offset = 0
        for sel, n in zip(combined_selected_tokens, combined_real_lens):
            bonus = sel.view(-1)[:n].unsqueeze(1)  # [n, 1]
            drafts_chunk = drafts[req_offset:req_offset + n]  # [n, K]
            src = torch.cat([bonus.to(drafts_chunk.dtype), drafts_chunk],
                            dim=1)  # [n, 1+K] = [bonus, draft_1..K]
            next_tokens_tpu_chunks.append(src.reshape(-1))
            num_rejected_chunks.append(
                torch.zeros(n, dtype=torch.int32, device=src.device))
            for r in range(n):
                next_token_indices[req_offset + r] = req_offset + r
                num_draft_per_req[req_offset + r] = K
            req_offset += n
        spec_num_rejected = (torch.cat(num_rejected_chunks)
                             if num_rejected_chunks else None)
        return (next_tokens_tpu_chunks, next_token_indices, spec_num_rejected,
                num_draft_per_req)

    def _prepare_inputs(self,
                        scheduler_output: "SchedulerOutput",
                        start_index: int,
                        num_decode_reqs: int,
                        num_windowed_reqs: int | None = None):
        if num_windowed_reqs is None:
            num_windowed_reqs = num_decode_reqs
        assert scheduler_output.total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        assert start_index < num_reqs

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
            if (not use_max_model_len and self.most_model_len is not None
                    and num_tokens > self.most_model_len):
                use_max_model_len = True
            num_scheduled_tokens_per_req.append(num_tokens)
        if use_max_model_len:
            if len(num_scheduled_tokens_per_req) > self.num_reqs_max_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[:
                                                                            self
                                                                            .
                                                                            num_reqs_max_model_len]
                end_index = start_index + self.num_reqs_max_model_len
            else:
                end_index = num_reqs
        else:
            assert self.num_reqs_most_model_len is not None
            if len(num_scheduled_tokens_per_req
                   ) > self.num_reqs_most_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[:
                                                                            self
                                                                            .
                                                                            num_reqs_most_model_len]
                end_index = start_index + self.num_reqs_most_model_len
            else:
                end_index = num_reqs
        max_num_scheduled_tokens_all_reqs = max(num_scheduled_tokens_per_req)
        num_scheduled_tokens_per_req = np.array(num_scheduled_tokens_per_req,
                                                dtype=np.int32)
        total_num_scheduled_tokens = sum(num_scheduled_tokens_per_req)
        assert max_num_scheduled_tokens_all_reqs > 0

        num_reqs = len(num_scheduled_tokens_per_req)

        sequence_layout_planner = _get_sequence_layout_planner_for_runner(self)
        sequence_layout_planner.reserve_host_token_capacity(
            self, int(total_num_scheduled_tokens))

        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Fast path for decode-only: all requests have exactly 1 token.
        if max_num_scheduled_tokens_all_reqs == 1:
            # Pure decode: each request schedules exactly 1 token.
            # req_indices = [0, 1, ..., num_reqs-1]
            req_indices = self.arange_np[start_index:start_index + num_reqs]
            # positions = num_computed_tokens for each request
            positions_np = self.positions_np[:num_reqs]
            np.copyto(
                positions_np, self.input_batch.
                num_computed_tokens_cpu[start_index:start_index + num_reqs])
            # token_indices = positions + req_index * max_model_len
            token_indices = (
                positions_np +
                req_indices * self.input_batch.token_ids_cpu.shape[1])
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
                self.arange_np[start_index:start_index + num_reqs],
                num_scheduled_tokens_per_req)

            # Get batched arange.
            # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            arange = np.concatenate(
                [self.arange_np[:n] for n in num_scheduled_tokens_per_req])

            # Get positions.
            positions_np = self.positions_np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                arange,
                out=positions_np,
            )

            # Get token indices.
            token_indices = (
                positions_np +
                req_indices * self.input_batch.token_ids_cpu.shape[1])

            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(token_indices),
                out=self.input_ids_cpu[:total_num_scheduled_tokens],
            )

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_per_req,
                  out=self.query_start_loc_np[1:num_reqs + 1])
        # Keep padded entries equal to the last valid location so padded
        # requests have zero length instead of a negative q_len.
        self.query_start_loc_np[num_reqs +
                                1:] = self.query_start_loc_np[num_reqs]

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[start_index:start_index +
                                                     num_reqs] +
            num_scheduled_tokens_per_req)

        if use_max_model_len:
            target_num_reqs = self.num_reqs_max_model_len
        else:
            assert self.num_reqs_most_model_len is not None
            target_num_reqs = self.num_reqs_most_model_len
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs)

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
        padded_total_num_scheduled_tokens = (
            layout_plan.global_padded_num_tokens)
        local_total_num_scheduled_tokens = layout_plan.local_num_tokens
        local_padded_total_num_scheduled_tokens = (
            layout_plan.local_padded_num_tokens)
        local_token_slice = layout_plan.token_slice
        if layout_plan.kind is SequenceLayoutKind.ALL:
            # Zero out to avoid spurious values from prev iteration.
            self.input_ids_cpu[total_num_scheduled_tokens:
                               padded_total_num_scheduled_tokens] = 0
        self.input_ids = self.input_ids_cpu[local_token_slice].to(
            self.device, non_blocking=True)
        if self.uses_mrope:
            if layout_plan.kind is SequenceLayoutKind.ALL:
                self.mrope_positions.cpu[:, total_num_scheduled_tokens:
                                         padded_total_num_scheduled_tokens] = 0
            self.position_ids = self.mrope_positions.cpu[:,
                                                         local_token_slice].to(
                                                             self.device,
                                                             non_blocking=True)
        else:
            self.position_ids = self.positions_cpu[local_token_slice].to(
                self.device, non_blocking=True)
        if use_max_model_len:
            seq_lens = self.seq_lens_cpu[:self.num_reqs_max_model_len].to(
                self.device, non_blocking=True)
            target_num_reqs = self.num_reqs_max_model_len
        else:
            assert self.num_reqs_most_model_len is not None
            seq_lens = self.seq_lens_cpu[:self.num_reqs_most_model_len].to(
                self.device, non_blocking=True)
            target_num_reqs = self.num_reqs_most_model_len

        # Async spec: seq_lens/positions were advanced optimistically (every
        # draft from the previous step assumed accepted). Subtract the real
        # per-request rejected count on-device, keyed by req position.
        if (self.scheduler_config.async_scheduling and self.speculative_config
                and self._pre_async_results is not None
                and self._pre_async_results.spec_decode_num_rejected_tokens
                is not None):
            from vllm_torchtpu.runner.tpu_runner_async_output import \
                subtract_num_rejected_tokens
            seq_idx_np = np.full(target_num_reqs, -1, dtype=np.int32)
            pos_idx_np = np.full(padded_total_num_scheduled_tokens,
                                 -1,
                                 dtype=np.int32)
            acc = 0
            for i in range(num_reqs):
                req_id = self.input_batch.req_ids[start_index + i]
                n_sched = int(num_scheduled_tokens_per_req[i])
                pos = self._pre_async_results.req_id_to_index_copy.get(req_id)
                if pos is not None:
                    seq_idx_np[i] = pos
                    pos_idx_np[acc:acc + n_sched] = pos
                acc += n_sched
            seq_lens, self.position_ids = subtract_num_rejected_tokens(
                seq_lens, self.position_ids,
                self._pre_async_results.spec_decode_num_rejected_tokens,
                torch.from_numpy(seq_idx_np).to(self.device,
                                                non_blocking=True),
                torch.from_numpy(pos_idx_np).to(self.device,
                                                non_blocking=True))

        # For decode-only case, cache constant device tensors to skip H2D.
        # query_start_loc, logits_indices, and request_distribution don't
        # change between decode steps for a given (num_reqs, padded_num_reqs).
        is_decode_only = (max_num_scheduled_tokens_all_reqs == 1)
        decode_cache_key = (
            num_reqs,
            padded_num_reqs,
            use_max_model_len,
            layout_plan.descriptor.cache_key,
        )

        # Speculative decoding metadata.
        spec_decode_metadata = None
        if self.speculative_config:
            num_draft_tokens = np.array([
                len(
                    scheduler_output.scheduled_spec_decode_tokens.get(
                        req_id, ())) for req_id in
                self.input_batch.req_ids[start_index:start_index + num_reqs]
            ],
                                        dtype=np.int32)
            if num_draft_tokens.any():
                spec_decode_metadata = self.spec_decode_manager.get_spec_decode_metadata(
                    num_draft_tokens, self.query_start_loc_np[1:num_reqs + 1],
                    padded_num_reqs)

        # Default decode layouts keep logits_indices stable for a fixed
        # (num_reqs, padded_num_reqs) bucket. PCP streaming layouts provide
        # explicit rank-major logits indices that can change as q_start crosses
        # an interleave/rank boundary, so they must not reuse a stale cached
        # logits_indices tensor.
        can_cache_decode_metadata = (is_decode_only
                                     and spec_decode_metadata is None and
                                     layout_plan.logits_indices_cpu is None)

        if (can_cache_decode_metadata
                and decode_cache_key == self._decode_device_cache_key):
            # Reuse cached device tensors — skip 3 H2D copies.
            query_start_loc = self._cached_query_start_loc
            logits_indices = self._cached_logits_indices
            request_distribution = self._cached_request_distribution
        else:
            # Compute and copy to device (first call or shape changed).
            if use_max_model_len:
                query_start_loc = self.query_start_loc_cpu[:self.
                                                           num_reqs_max_model_len
                                                           + 1].to(
                                                               self.device,
                                                               non_blocking=True
                                                           )
            else:
                query_start_loc = self.query_start_loc_cpu[:self.
                                                           num_reqs_most_model_len
                                                           + 1].to(
                                                               self.device,
                                                               non_blocking=True
                                                           )

            if spec_decode_metadata is not None:
                logits_indices = spec_decode_metadata.final_logits_indices
            elif layout_plan.logits_indices_cpu is not None:
                logits_indices = layout_plan.logits_indices_cpu.to(
                    self.device, non_blocking=True)
            else:
                # Indices at which we sample (positions of last token in the
                # sequence). Padded to avoid recompiling when `num_reqs` varies.
                logits_indices = (
                    self.query_start_loc_cpu[1:padded_num_reqs + 1] - 1).to(
                        self.device, non_blocking=True)

            # For the V3 kernel, request_distribution is
            # [decode_end, prefill_end, mixed_end]. We put decode requests first,
            # no dedicated prefill-only bucket, and all remaining requests in mixed mode.
            chunk_num_decode = max(
                0, min(num_decode_reqs - start_index, num_reqs))
            self._request_distribution_cpu[0] = chunk_num_decode
            self._request_distribution_cpu[1] = chunk_num_decode
            self._request_distribution_cpu[2] = num_reqs
            request_distribution = self._request_distribution_cpu.to(
                self.device, non_blocking=True)

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
        mamba_state_indices = (self._build_mamba_state_indices(
            start_index, num_reqs, target_num_reqs)
                               if self._has_mamba_state else None)

        # Spec decode with mamba layers: the GDN kernel's windowed segment
        # covers 1-token decodes AND speculative verify windows (the batch is
        # ordered [decode][verify][prefill/mixed] by _reorder_batch_for_rpa),
        # while RPA keeps its 1-token-decode-only front segment.
        mamba_request_distribution = None
        if self.mamba_slot_read_offsets is not None:
            chunk_num_windowed = max(
                0, min(num_windowed_reqs - start_index, num_reqs))
            chunk_num_decode = max(
                0, min(num_decode_reqs - start_index, num_reqs))
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
                [] if self._unified_kv_layout
                and self.mamba_slot_read_offsets is not None else None),
            sequence_layout_descriptor=layout_plan.descriptor,
        )
        slot_mappings = self.empty_slot_mappings
        per_layer_attn_metadata, _unused_spec_decode_common_attn_metadata = self._build_attention_metadata(
            num_tokens=local_total_num_scheduled_tokens,
            num_reqs=target_num_reqs,
            max_query_len=max_num_scheduled_tokens_all_reqs,
            num_tokens_padded=local_padded_total_num_scheduled_tokens,
            num_reqs_padded=target_num_reqs,
            slot_mappings=slot_mappings,
        )

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA adapters
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copying to avoid accidental state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += (
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens)

            self.set_active_loras(self.input_batch,
                                  padded_num_scheduled_tokens_per_req)

        # Prepare token substitution indices
        cur_input_indices, pre_next_tokens_indices = self._prepare_async_token_substitution_indices(
            start_index, num_reqs, num_scheduled_tokens_per_req)

        self._collect_mamba_state_seed_copies(scheduler_output, start_index,
                                              num_reqs)

        return (
            per_layer_attn_metadata,
            logits_indices,
            padded_num_reqs,
            num_reqs,
            end_index,
            cur_input_indices,
            pre_next_tokens_indices,
            spec_decode_metadata,
        )

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
        return (utils.get_dp_size(self.parallel_config) > 1
                and self.parallel_config.enable_expert_parallel)

    def _count_input_chunks(self, scheduler_output: "SchedulerOutput") -> int:
        """Match _prepare_inputs chunking without staging any tensors."""
        if scheduler_output.total_num_scheduled_tokens == 0:
            return 0

        num_reqs = self.input_batch.num_reqs
        start_index = 0
        num_chunks = 0
        while start_index < num_reqs:
            use_max_model_len = self.most_model_len is None
            for i in range(start_index, num_reqs):
                req_id = self.input_batch.req_ids[i]
                assert req_id is not None
                num_tokens = scheduler_output.num_scheduled_tokens[req_id]
                if (not use_max_model_len and self.most_model_len is not None
                        and num_tokens > self.most_model_len):
                    use_max_model_len = True

            if use_max_model_len:
                chunk_reqs = self.num_reqs_max_model_len
            else:
                assert self.num_reqs_most_model_len is not None
                chunk_reqs = self.num_reqs_most_model_len
            start_index += min(chunk_reqs, num_reqs - start_index)
            num_chunks += 1
        return num_chunks

    def _dp_coordinated_step(
            self, local_num_tokens: int,
            local_num_chunks: int) -> "tuple[int | None, int]":
        """Return the DP-wide padded token bucket and forward chunk count."""
        if not self._dp_lockstep_enabled():
            return None, local_num_chunks
        from vllm.distributed.parallel_state import get_dp_group

        # Token/chunk metadata is synchronized on vLLM's CPU DP group. The
        # model forward below receives this result explicitly as
        # num_tokens_across_dp, so set_forward_context does not run a second
        # DP synchronization.
        t = torch.tensor([int(local_num_tokens),
                          int(local_num_chunks)],
                         dtype=torch.int64,
                         device="cpu")
        torch.distributed.all_reduce(t,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_dp_group().cpu_group)
        bucket = _get_padded_token_len(self.num_tokens_paddings,
                                       int(t[0].item()))
        return bucket, int(t[1].item())

    def _run_dp_dummy_chunk(self, bucket: int) -> None:
        self._dummy_run(bucket,
                        self.num_reqs_max_model_len,
                        self.max_num_blocks_per_req,
                        use_max_model_len=True,
                        dp_lockstep=True)

    def _run_dp_idle_pairing(self, bucket: int, num_chunks: int) -> None:
        """Idle-rank EP-DP pairing: emit the same cross-DP collective stream a
        busy rank does — ALL target dummy forwards, then ALL draft dummy
        forwards.
        """
        # Set before any dummy fires: both _run_dp_dummy_chunk and the draft
        # run_dp_dummy_draft read _dp_target_bucket for the coordinated, equal-token shape.
        self._dp_target_bucket = bucket
        # Phase 1: all TARGET dummy forwards.
        for _ in range(num_chunks):
            self._run_dp_dummy_chunk(bucket)
        # Phase 2: all DRAFT dummy forwards (num_chunks * K).
        spec = self.speculative_config
        if num_chunks > 0 and spec is not None and spec.use_eagle():
            self.drafter.run_dp_dummy_draft(num_chunks)

    def _dp_num_tokens_across_dp(self, num_tokens: int) -> torch.Tensor | None:
        if not self._dp_lockstep_enabled():
            return None
        return torch.full((utils.get_dp_size(self.parallel_config), ),
                          num_tokens,
                          dtype=torch.int32,
                          device="cpu")

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | None:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")
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
            return self.kv_connector_no_forward(scheduler_output,
                                                self.vllm_config)
        # Run the multimodal (vision) encoder. vLLM's base
        # GPUModelRunner.execute_model does this; our override must do it
        # explicitly, otherwise the encoder never runs, image placeholder
        # tokens get plain text embeddings, and the model produces garbage.
        if self.supports_mm_inputs:
            self._execute_mm_encoder(scheduler_output)

        num_decode_reqs, num_windowed_reqs = self._reorder_batch_for_rpa(
            scheduler_output)

        # Profile the current batch composition (prefill vs decode) if phased profiling is enabled.
        if self.phase_based_profiler:
            self.batch_counter += 1
            padded_total_tokens = _get_padded_token_len(
                self.num_tokens_paddings,
                scheduler_output.total_num_scheduled_tokens)
            batch_composition_stats = runner_utils.get_batch_composition_stats(
                batch_id=self.batch_counter,
                input_batch=self.input_batch,
                total_num_scheduled_tokens=scheduler_output.
                total_num_scheduled_tokens,
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
        if self.supports_mm_inputs:
            mm_embeds_list, is_mm_embed_full = self._gather_mm_embeddings(
                scheduler_output)
            if mm_embeds_list:
                mm_embeds_flat = torch.cat(mm_embeds_list)
            req_tok = np.array([
                scheduler_output.num_scheduled_tokens[r]
                for r in self.input_batch.req_ids
            ],
                               dtype=np.int64)
            mm_tok_cumsum = np.concatenate([[0], np.cumsum(req_tok)])
            # Per-token MM offsets on the host so the chunk loop can slice
            # mm_embeds without any device syncs -- mirrors the GPU runner,
            # which passes the full mask to embed_input_ids and lets
            # masked_scatter place the embeds (no manual counting).
            if mm_embeds_flat is not None:
                mm_cumsum_np = np.concatenate(
                    [[0], np.cumsum(is_mm_embed_full.cpu().numpy())])

        local_num_chunks = self._count_input_chunks(scheduler_output)
        self._dp_target_bucket, target_num_chunks = self._dp_coordinated_step(
            scheduler_output.total_num_scheduled_tokens, local_num_chunks)
        # Retained for the spec-decode propose phase (sample_tokens): under
        # EP-DP locksteps every rank must run the same number of draft forwards per step,
        # and the coordinated chunk count from this one all-reduce is the shared
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
        is_draft_model = (self._is_async_drafter)

        # NOTE: setup current batch's metadata for kv connector.
        # Verified with TPURaidenConnector, TPUConnector, OffloadingConnector
        with set_forward_context(None, self.vllm_config):
            # Raiden overlap: block the worker main thread on the KV load only
            # in inline mode. In the default async mode the load (network pull
            # + DMA H2D into the cache) runs entirely on the Raiden C++
            # threads while this step's forward computes; the request stays in
            # WAITING_FOR_REMOTE_KVS until every rank reports done_recving.
            raiden_inline = dist_utils.get_raiden_inline_load()
            self.maybe_setup_kv_connector(scheduler_output,
                                          wait_for_completion=raiden_inline,
                                          report_completion=not raiden_inline)

        while chunk_index < target_num_chunks:
            if start_index >= self.input_batch.num_reqs:
                assert self._dp_target_bucket is not None
                self._run_dp_dummy_chunk(self._dp_target_bucket)
                chunk_index += 1
                continue

            (attn_metadata, logits_indices, padded_num_reqs, num_reqs,
             end_index, cur_input_indices, pre_next_tokens_indices,
             spec_decode_metadata) = (self._prepare_inputs(
                 scheduler_output, start_index, num_decode_reqs,
                 num_windowed_reqs))

            # Seed newly-advanced mamba state blocks before the forward reads
            # them (chunk boundaries, decode crossings, prefix-cache resumes).
            self._flush_mamba_state_seed_copies()

            # A recycled slot/block still carries the previous request's read
            # offset; clear it before this chunk's forward gathers it.
            self._reset_read_offsets_for_new_requests(start_index, num_reqs)

            input_ids = self._apply_async_token_substitution(
                self.input_ids, cur_input_indices, pre_next_tokens_indices)
            # Async spec: the drafts the target verifies live in the substituted
            # input_ids (host input_ids_cpu holds placeholders), so re-source the
            # rejection sampler's draft_token_ids from the device.
            if (self.scheduler_config.async_scheduling
                    and spec_decode_metadata is not None
                    and len(cur_input_indices) > 0):
                from vllm_torchtpu.runner.tpu_runner_async_output import \
                    extract_draft_token_ids
                spec_decode_metadata.draft_token_ids = extract_draft_token_ids(
                    input_ids, spec_decode_metadata.final_logits_indices,
                    spec_decode_metadata.target_logits_indices)
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
                        [mm_embeds_flat[mm_start:mm_start +
                                        mm_cnt]] if mm_cnt > 0 else [])
                else:
                    chunk_embeds = []
                chunk_mm_inputs = (chunk_embeds, is_mm_chunk)

            input_ids, inputs_embeds = self._get_model_inputs(
                input_ids, chunk_mm_inputs)
            # Run the decoder
            # set_forward_context: vLLM's native context for attention metadata
            # set_vllm_model_wrapper_context: TPU-specific context for mesh info
            # For multimodal models _get_model_inputs returns input_ids=None
            # and packs the tokens into inputs_embeds; use whichever exists.
            num_tokens_padded = (input_ids if input_ids is not None else
                                 inputs_embeds).shape[0]
            if envs.TPU_MOE_SKIP_PADDED_TOKENS:
                self._token_padding_update(num_tokens_padded)
            with set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=self._dp_num_tokens_across_dp(
                        num_tokens_padded),
            ), set_vllm_model_wrapper_context(mesh=self.mesh,
                                              vllm_config=self.vllm_config):
                hidden_states, aux_hidden_states = self.forward_model(
                    input_ids=input_ids,
                    positions=self.position_ids,
                    inputs_embeds=inputs_embeds,
                )

            sequence_layout_planner = _get_sequence_layout_planner_for_runner(
                self)
            layout_plan = self._last_sequence_layout_plan
            sample_hidden_states = (
                sequence_layout_planner.maybe_select_logits_hidden_states(
                    hidden_states, layout_plan, logits_indices))
            if sample_hidden_states is None:
                hidden_states = sequence_layout_planner.finalize_hidden_states(
                    hidden_states,
                    layout_plan,
                )
                logits = self.compute_selected_logits(hidden_states,
                                                      logits_indices)
            else:
                logits = self.compute_logits_from_hidden_states(
                    sample_hidden_states)

            if self.is_pooling_model:
                pooling_metadata = self.input_batch.get_pooling_metadata()
                # num_scheduled_tokens_np, seq_lens_cpu for this chunk
                req_ids = self.input_batch.req_ids[start_index:start_index +
                                                   num_reqs]
                num_scheduled_tokens_np = np.array([
                    scheduler_output.num_scheduled_tokens[req_id]
                    for req_id in req_ids
                ],
                                                   dtype=np.int32)
                seq_lens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
                    start_index:start_index + num_reqs]
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
                mamba_state_indices_list.append(
                    ctx.unified_mamba_state_indices)
            if is_draft_model:
                # Capture this chunk's draft inputs while they are valid.
                draft_chunks.append(
                    DraftChunkInputs(
                        input_ids=draft_input_ids_src,
                        position_ids=self.position_ids,
                        query_start_loc_np=self.query_start_loc_np[:num_reqs +
                                                                   1].copy(),
                        attn_ctx=self._attn_metadata_builder_ctx,
                        start_index=start_index,
                        num_reqs=num_reqs,
                        aux_hidden_states=aux_hidden_states
                        if aux_hidden_states is not None else [],
                        hidden_states=hidden_states,
                        draft_lengths=(spec_decode_metadata.draft_lengths
                                       if spec_decode_metadata is not None else
                                       None),
                        attn_metadata=attn_metadata,
                    ))

            start_index = end_index
            chunk_index += 1

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
    def _update_mamba_slot_read_offsets(self,
                                        indices_per_group: list[torch.Tensor]
                                        | None,
                                        next_tokens: torch.Tensor | None,
                                        num_reqs: int) -> None:
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
        offsets = torch.zeros(indices_per_group[0].shape[0],
                              dtype=torch.int32,
                              device=indices_per_group[0].device)
        if next_tokens is not None:
            # num_valid = accepted drafts + 1 (bonus); the checkpoint of the
            # last accepted token is at offset num_valid - 1.
            num_valid = (next_tokens[:num_reqs]
                         != INVALID_TOKEN_ID).sum(dim=1).to(torch.int32)
            offsets[:num_reqs] = (num_valid - 1).clamp(min=0)
        for group_indices in indices_per_group:
            self.mamba_slot_read_offsets.index_put_((group_indices.long(), ),
                                                    offsets)

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
    def _reset_read_offsets_for_new_requests(self, start_index: int,
                                             num_reqs: int) -> None:
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
        self._mamba_offset_seeded &= set(
            self.input_batch.req_id_to_index.keys())

        width = groups[0].shape[0]
        keep, keep_dev, zeros, null = self._read_offset_reset_scratch(
            width, groups[0].device)
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
        for group_indices in groups:
            # Non-new rows collapse onto the null slot; writing 0 there is a
            # no-op, which keeps this scatter one fixed shape.
            targets = torch.where(keep_dev, group_indices, null)
            self.mamba_slot_read_offsets.index_put_((targets.long(), ), zeros)

    def _read_offset_reset_scratch(
        self, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reusable `(keep_cpu, keep_dev, zeros, null)` of length `width`.

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
            offsets = self.mamba_slot_read_offsets
            entry = (
                torch.zeros(width, dtype=torch.bool),
                torch.zeros(width, dtype=torch.bool, device=device),
                torch.zeros(width, dtype=offsets.dtype, device=offsets.device),
                torch.zeros((), dtype=torch.int32, device=device),
            )
            self._read_offset_scratch[key] = entry
        return entry

    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncTPUModelRunnerOutput:
        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
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
                "with the merged vLLM v1 sampler path yet.")
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
        is_spec_step = any(md is not None
                           for md in state.spec_decode_metadata_list)
        # Per-chunk mamba slot ids (hybrid + spec decode only; None entries
        # otherwise) for the read-offset scatter after sampling.
        mamba_idx_list = (state.mamba_state_indices_list
                          or [None] * len(state.logits_list))
        if self.is_pooling_model:
            req_ids = cast(
                list[str],
                self.input_batch.req_ids[:self.input_batch.num_reqs])
        elif is_spec_step:
            if needs_logprobs:
                raise NotImplementedError(
                    "Logprobs are not supported with speculative decoding on "
                    "TPU yet.")
            # Per-chunk device rejection outputs, kept for the async-spec
            # producer below.
            next_tokens_per_chunk: list[torch.Tensor] = []
            cur_start_idx = 0
            all_greedy = self.input_batch.all_greedy
            for logits, num_reqs, md, mamba_indices in zip(
                    state.logits_list, state.num_reqs_list,
                    state.spec_decode_metadata_list, mamba_idx_list):
                cur_end_idx = cur_start_idx + num_reqs
                if md is not None:
                    if all_greedy:
                        # Extract bonus tokens + gather target_logits from the
                        # target model logits in one compiled region (bonus argmax
                        # + the two gathers) so they don't eager-fuse per context.
                        bonus_token_ids, target_logits = (
                            self.spec_bonus_and_target_logits(
                                logits, md.bonus_logits_indices,
                                md.target_logits_indices))
                        next_tokens = self.rejection_sampler(
                            draft_token_ids=md.draft_token_ids,
                            num_draft_tokens=md.draft_lengths,
                            target_logits=target_logits,
                            bonus_token_ids=bonus_token_ids,
                            segment_ids=md.segment_ids,
                            group_indices=md.group_indices,
                            max_draft_tokens=self.speculative_config.
                            num_speculative_tokens,
                        )
                    else:
                        # Gather bonus + target logits in one compiled region
                        # (mirrors the greedy spec_bonus_and_target_logits) so
                        # the raw logits[...] gathers don't eager-fuse into
                        # per-context programs.
                        bonus_logits, target_logits = (
                            self.spec_gather_bonus_and_target_logits(
                                logits, md.bonus_logits_indices,
                                md.target_logits_indices))
                        req_temperatures, req_top_k, req_top_p = (
                            self._build_padded_sampling_params(
                                cur_start_idx, cur_end_idx, bonus_logits))
                        bonus_u = torch.rand_like(bonus_logits,
                                                  generator=sampling_generator)
                        bonus_token_ids = self.sample_from_logits_func(
                            bonus_logits,
                            req_temperatures,
                            bonus_u,
                            req_top_k,
                            req_top_p,
                            all_greedy=False,
                        ).view(-1)
                        accept_u = torch.rand(md.draft_token_ids.shape,
                                              dtype=torch.float32,
                                              device=target_logits.device,
                                              generator=sampling_generator)
                        recover_u = torch.rand_like(
                            target_logits,
                            dtype=torch.float32,
                            generator=sampling_generator)
                        next_tokens = self.rejection_sampler(
                            draft_token_ids=md.draft_token_ids,
                            num_draft_tokens=md.draft_lengths,
                            target_logits=target_logits,
                            bonus_token_ids=bonus_token_ids,
                            segment_ids=md.segment_ids,
                            group_indices=md.group_indices,
                            max_draft_tokens=self.speculative_config.
                            num_speculative_tokens,
                            temperatures=req_temperatures[md.segment_ids],
                            top_k=req_top_k[md.segment_ids],
                            top_p=req_top_p[md.segment_ids],
                            accept_u=accept_u,
                            recover_u=recover_u,
                            do_sampling=True,
                        )
                    combined_selected_tokens.append(next_tokens)
                    combined_selected_tokens_real_lens.append(num_reqs)
                    next_tokens_per_chunk.append(next_tokens)
                    self._update_mamba_slot_read_offsets(
                        mamba_indices, next_tokens, num_reqs)
                else:
                    dummy = torch.empty((1, 1),
                                        dtype=logits.dtype,
                                        device=logits.device)
                    dummy_int = torch.empty((1, 1),
                                            dtype=torch.int32,
                                            device=logits.device)
                    dummy_float = torch.empty((1, 1),
                                              dtype=torch.float32,
                                              device=logits.device)
                    if all_greedy:
                        selected = self.sample_from_logits_func(
                            logits,
                            dummy,
                            dummy,
                            dummy_int,
                            dummy_float,
                            all_greedy=True)
                    else:
                        temperatures_tpu, top_k_tpu, top_p_tpu = (
                            self._build_padded_sampling_params(
                                cur_start_idx, cur_end_idx, logits))
                        u = torch.rand_like(logits,
                                            generator=sampling_generator)
                        selected = self.sample_from_logits_func(
                            logits,
                            temperatures_tpu,
                            u,
                            top_k_tpu,
                            top_p_tpu,
                            all_greedy=False)
                    padded = torch.full(
                        (selected.shape[0],
                         self.speculative_config.num_speculative_tokens + 1),
                        INVALID_TOKEN_ID,
                        dtype=selected.dtype,
                        device=selected.device)
                    padded[:, 0] = selected.view(-1)
                    combined_selected_tokens.append(padded)
                    combined_selected_tokens_real_lens.append(num_reqs)
                    next_tokens_per_chunk.append(padded)
                    # Non-verify chunk in a spec step: reset the read offsets
                    # (the latest state checkpoint is at the group base).
                    self._update_mamba_slot_read_offsets(
                        mamba_indices, None, num_reqs)
                self._update_num_xla_graphs("spec_step")
                cur_start_idx = cur_end_idx
        else:
            cur_start_idx = 0
            req_ids = cast(
                list[str],
                self.input_batch.req_ids[:self.input_batch.num_reqs])
            all_greedy = self.input_batch.all_greedy
            for logits, num_reqs, mamba_indices in zip(state.logits_list,
                                                       state.num_reqs_list,
                                                       mamba_idx_list):
                cur_end_idx = cur_start_idx + num_reqs
                # Non-spec step (prefill / plain decode): reset the mamba
                # read offsets so the next step resumes from the group base.
                self._update_mamba_slot_read_offsets(mamba_indices, None,
                                                     num_reqs)
                if grammar_output is not None:
                    require_struct_decoding, grammar_bitmask_padded, arange = (
                        self.prepare_structured_decoding_input(
                            logits, grammar_output))
                    logits = self.structured_decode(require_struct_decoding,
                                                    grammar_bitmask_padded,
                                                    logits, arange)
                if all_greedy:
                    dummy_placeholder = torch.empty((1, 1),
                                                    dtype=logits.dtype,
                                                    device=logits.device)
                    selected_token_ids = self.sample_from_logits_func(
                        logits,
                        dummy_placeholder,
                        dummy_placeholder,
                        torch.empty((1, 1),
                                    dtype=torch.int32,
                                    device=logits.device),
                        torch.empty((1, 1),
                                    dtype=torch.float32,
                                    device=logits.device),
                        all_greedy=True)
                else:
                    temperatures_tpu, top_k_tpu, top_p_tpu = (
                        self._build_padded_sampling_params(
                            cur_start_idx, cur_end_idx, logits))
                    u = torch.rand_like(logits, generator=sampling_generator)
                    selected_token_ids = self.sample_from_logits_func(
                        logits,
                        temperatures_tpu,
                        u,
                        top_k_tpu,
                        top_p_tpu,
                        all_greedy=all_greedy)
                # NOTE (NickLucche) Use the original logits (before any penalties or
                # temperature scaling) for the top-k logprobs. We can't enforce it
                # due to recompilations outside torch.compiled code, so just make
                # sure `sample_from_logits` does not modify the logits in-place.
                logprobs = (self.gather_logprobs(logits, selected_token_ids)
                            if needs_logprobs else None)

                # Keep the bucketed (padded) tensor; trim happens on the host
                # inside AsyncTPUCopyState to avoid per-`num_reqs` recompiles.
                combined_selected_tokens.append(selected_token_ids)
                combined_selected_tokens_real_lens.append(num_reqs)
                if self.scheduler_config.async_scheduling:
                    next_tokens_tpu_chunks.append(selected_token_ids.view(-1))
                    for req_idx in range(cur_start_idx, cur_end_idx):
                        next_token_indices[req_idx] = (next_tokens_tpu_offset +
                                                       req_idx - cur_start_idx)
                    next_tokens_tpu_offset += selected_token_ids.shape[0]
                if needs_logprobs:
                    sliced_logprobs = LogprobsTensors(
                        logprobs.logprob_token_ids[:num_reqs],
                        logprobs.logprobs[:num_reqs],
                        logprobs.selected_token_ranks[:num_reqs],
                        logprobs.cu_num_generated_tokens)
                    combined_logprobs.append(sliced_logprobs)

                self._update_num_xla_graphs("decoding_step")
                cur_start_idx = cur_end_idx

        # NOTE: current kv load and save get h2d/d2h copies involved.
        # Those copies are blocking. Once they become async., kv_save
        # should be called right after each single forward pass,
        # instead of the forwards of the entire input batch.
        self.maybe_wait_for_kv_save()
        (finished_sending, finished_recving, kv_worker_meta, invalid_block_ids,
         invalid_block_group_index,
         kv_connector_stats) = self.get_finished_kv_transfers(scheduler_output)

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
                    [lp.logprob_token_ids for lp in combined_logprobs_lists]),
                logprobs=np.concatenate(
                    [lp.logprobs for lp in combined_logprobs_lists]),
                sampled_token_ranks=np.concatenate([
                    lp.sampled_token_ranks for lp in combined_logprobs_lists
                ]),
            )
            logprobs = logprobs_lists

        discard_sampled_tokens_req_indices = []
        num_reqs = self.input_batch.num_reqs
        req_ids = cast(list[str], self.input_batch.req_ids[:num_reqs])

        request_seq_lens = []
        for i, req_id in enumerate(req_ids):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
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

        kv_connector_output = (None if (
            finished_sending is None and finished_recving is None
            and kv_worker_meta is None and not invalid_block_ids
            and kv_connector_stats is None) else _build_kv_connector_output(
                finished_sending=finished_sending,
                finished_recving=finished_recving,
                kv_connector_worker_meta=kv_worker_meta,
                invalid_block_ids=invalid_block_ids,
                invalid_block_group_index=invalid_block_group_index,
                kv_connector_stats=kv_connector_stats,
            ))

        next_tokens_tpu = None
        copy_state = None
        if combined_selected_tokens:
            copy_state = AsyncTPUCopyState.from_device_chunks(
                combined_selected_tokens, combined_selected_tokens_real_lens)

        if self.scheduler_config.async_scheduling:
            self._modify_prev_results()

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
        if (self._is_async_drafter and combined_selected_tokens):
            if is_spec_step:
                eagle3_drafts = (
                    self.spec_decode_manager.propose_draft_token_ids(
                        sampled_token_ids=None,
                        discard_sampled_tokens_req_indices=
                        discard_sampled_tokens_req_indices,
                        num_rejected_tokens_np=None,
                        scheduler_output=scheduler_output,
                        return_device=is_async,
                        next_tokens_per_chunk=next_tokens_per_chunk))
            else:
                # Seed = the just-sampled tokens, kept on-device (no D2H):
                # propose takes the sync seed path but reads the seed from
                # `device_seed`.
                device_seed = torch.cat([
                    sel.view(-1)[:n]
                    for sel, n in zip(combined_selected_tokens,
                                      combined_selected_tokens_real_lens)
                ])
                eagle3_drafts = (
                    self.spec_decode_manager.propose_draft_token_ids(
                        sampled_token_ids=[],
                        discard_sampled_tokens_req_indices=
                        discard_sampled_tokens_req_indices,
                        num_rejected_tokens_np=None,
                        scheduler_output=scheduler_output,
                        return_device=is_async,
                        device_seed=device_seed))

        if self.scheduler_config.async_scheduling:
            # Build the async substitution source from the drafts proposed
            # above: the [bonus, draft_1..K] source + 1+K next_token_indices,
            # and the per-request rejected count to park.
            spec_num_rejected = None
            num_draft_per_req = None
            if is_spec_step:
                (next_tokens_tpu_chunks, next_token_indices, spec_num_rejected,
                 num_draft_per_req) = (self._assemble_async_spec_substitution(
                     eagle3_drafts, next_tokens_per_chunk, state))
            elif (self._is_async_drafter and next_tokens_tpu_chunks):
                # Prefill / pure-non-spec eagle3 bootstrap: the non-spec sampling
                # branch parked a stride-1 source (sampled token only). Rebuild
                # it as a stride-(1+K) [bonus, draft_1..K] source carrying real
                # prompt-context drafts so the NEXT step's verify gets real
                # drafts instead of placeholders.
                (next_tokens_tpu_chunks, next_token_indices, spec_num_rejected,
                 num_draft_per_req) = (self._assemble_async_prefill_bootstrap(
                     eagle3_drafts, combined_selected_tokens,
                     combined_selected_tokens_real_lens))
            req_id_to_index_copy = {}
            if not self.is_pooling_model:
                req_id_to_index_copy = self._update_placeholder(
                    discard_sampled_tokens_req_indices, request_seq_lens,
                    next_token_indices, num_draft_per_req)
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
                        1 + self.speculative_config.num_speculative_tokens)
                    if next_tokens_tpu.shape[0] < bound:
                        next_tokens_tpu = torch.nn.functional.pad(
                            next_tokens_tpu,
                            (0, bound - next_tokens_tpu.shape[0]))
                self._pre_async_results = AsyncPreResults(
                    req_ids=req_ids,
                    next_tokens_tpu=next_tokens_tpu,
                    request_seq_lens=request_seq_lens,
                    discard_sampled_tokens_req_indices=
                    discard_sampled_tokens_req_indices,
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
            sampled_token_ids=
            [],  # Filled in AsyncTPUModelRunnerOutput get_output
            logprobs=logprobs,
            prompt_logprobs_dict={req_id: None
                                  for req_id in req_ids},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
        )

        async_output = AsyncTPUModelRunnerOutput(
            model_runner_output=model_runner_output,
            copy_state=copy_state,
            discard_sampled_tokens_req_indices=
            discard_sampled_tokens_req_indices)

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
                    start_tok_idx = self.input_batch.num_tokens_no_spec[
                        req_idx]
                    end_tok_idx = start_tok_idx + len(valid_tokens)
                    self.input_batch.token_ids_cpu[
                        req_idx, start_tok_idx:end_tok_idx] = valid_tokens
                    self.input_batch.num_tokens_no_spec[req_idx] = end_tok_idx

                    req_state.output_token_ids.extend(valid_tokens)

            if (self.speculative_config
                    and self.speculative_config.method == "ngram"):
                # ngram drafts on the host from the committed token ids, so it
                # must run after materialization; eagle3/dflash were already proposed
                # from device state by the unified call above.
                self.spec_decode_manager.propose_draft_token_ids(
                    sampled_token_ids=final_output.sampled_token_ids,
                    discard_sampled_tokens_req_indices=
                    discard_sampled_tokens_req_indices,
                    num_rejected_tokens_np=None,
                    scheduler_output=scheduler_output)

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
        logger.info("Setting TPU quantization config for: %s",
                    self.model_config.quantization)
        self.vllm_config.quant_config = get_tpu_quantization_config(
            self.vllm_config)

        model_loader = get_model_loader(self.load_config)
        logger.info("Loading model from scratch...")
        with set_vllm_model_wrapper_context(mesh=self.mesh,
                                           vllm_config=self.vllm_config), \
             set_current_vllm_config(self.vllm_config):
            model = model_loader.load_model(vllm_config=self.vllm_config,
                                            model_config=self.model_config)
        self.model = model
        if envs.TPU_ROPE_CACHE_TRUNCATE:
            self._truncate_rope_caches()

        # If using eagle3/mtp or dflash speculative decoding, load the draft model and
        # share the target's embeddings / LM head (if the draft requires it).
        if self._is_async_drafter:
            self.drafter.load_model(self.model)

        # Ensure attention custom ops exist before any compile/inference path,
        self._initialize_pallas_kernels()

        if envs.TPU_MOE_SKIP_PADDED_TOKENS:
            self._token_padding_state = token_padding.TokenPaddingState.create(
                self.max_num_tokens, self.device)
            token_padding.set_padding_state(self._token_padding_state)

    def _initialize_pallas_kernels(self):
        self._initialize_attention_kernels()
        self._initialize_quantization_kernels()

    def _initialize_quantization_kernels(self):
        from vllm_torchtpu.layers.vllm.linear_common import (
            _get_quantized_matmul_fp4_op, _get_quantized_matmul_op)

        with set_current_vllm_config(self.vllm_config):
            # Pre-warm FP8 quantized-matmul lock; Dynamo can't trace Lock.
            _get_quantized_matmul_op()
            # Same for the NVFP4 W4A16 matmul op.
            _get_quantized_matmul_fp4_op()

    def _initialize_attention_kernels(self) -> None:
        """Pre-build Pallas RPA attention kernels before torch.compile.

        Must be called after model weights are loaded and before the first
        forward_model() call (which triggers torch.compile tracing).
        """
        if self._attention_kernels_initialized:
            return
        from vllm_torchtpu.layers.vllm.attention import (
            _DRAFT_KV_BLOCK_CAP, PallasAttentionBackendImpl,
            _pallas_rpa_kernel_local)

        # Only draft proposers expose _draft_attn_layer_names; NgramProposer
        # operates purely on token IDs. no draft attention layers to relocate.
        spec_drafter = self.drafter
        draft_attn_names = getattr(spec_drafter, "_draft_attn_layer_names",
                                   None) or set()

        layers = get_layers_from_vllm_config(self.vllm_config,
                                             AttentionLayerBase)
        initialized_count = 0
        with set_vllm_model_wrapper_context(mesh=self.mesh,
                                            vllm_config=self.vllm_config):
            for name, attn_layer in layers.items():
                if isinstance(getattr(attn_layer, "impl", None),
                              PallasAttentionBackendImpl):
                    # Relocate a REPLICATED (tp=1) draft's attention to the LOCAL
                    # (non-shard_map) kernel.
                    if (name in draft_attn_names and
                            self.speculative_config.draft_tensor_parallel_size
                            == 1):
                        # Instance attrs shadow the ClassVars; unique prefix
                        # keeps the local kernel op out of the sharded registry.
                        attn_layer.impl._kernel_entry = _pallas_rpa_kernel_local
                        attn_layer.impl._kernel_op_prefix = (
                            "pallas::rpa_kernel_local")
                        logger.info(
                            "Draft attn %s -> LOCAL (non-shard_map) RPA kernel"
                            " | DRAFT_KV_BLOCK_CAP=%d", name,
                            _DRAFT_KV_BLOCK_CAP)
                    attn_layer.impl.initialize_kernel(attn_layer)
                    initialized_count += 1
        logger.info("Pre-built RPA kernels for %d attention layers.",
                    initialized_count)
        self._attention_kernels_initialized = True

    @torch.no_grad()
    def _dummy_run(self,
                   num_tokens: int,
                   num_reqs: int,
                   num_blocks: int,
                   use_max_model_len: bool = True,
                   dp_lockstep: bool = False) -> None:
        kv_cache_initialized = self.kv_cache_config is not None

        if self.supports_mm_inputs:
            input_ids = None
            inputs_embeds = torch.zeros(
                (num_tokens, self.inputs_embeds_size),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            input_ids = torch.zeros((num_tokens),
                                    dtype=torch.int32).to(self.device)
            inputs_embeds = None
        actual_num_reqs = min(num_tokens, num_reqs)
        if self.uses_mrope:
            position_ids = torch.zeros((3, num_tokens),
                                       dtype=torch.int32).to(self.device)
        else:
            position_ids = torch.zeros(num_tokens,
                                       dtype=torch.int32).to(self.device)
        query_lens = [1] * num_reqs
        query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                    dtype=torch.int32),
                                       dim=0,
                                       dtype=torch.int32).to(self.device)
        seq_lens = torch.ones((num_reqs, ), dtype=torch.int32).to(self.device)
        # V3: request_distribution = [decode_end, prefill_end, mixed_end].
        # Dummy runs use one scheduled token per active request, so model
        # them as pure decode to match the real single-chip path.
        request_distribution = torch.tensor(
            [actual_num_reqs, actual_num_reqs, actual_num_reqs],
            dtype=torch.int32).to(self.device)
        dummy_layout_plan = _get_sequence_layout_planner_for_runner(
            self).prepare_dummy(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                kv_cache_initialized=kv_cache_initialized,
            )

        if self.kv_cache_config is not None:
            # Dummy compact-mamba slot ids (all null slot 0): the dummy run
            # only traces shapes/HBM, so the recurrent state read/written is
            # never consumed. Shape must match the GDN op's max_reqs
            # (= seq_lens length = num_reqs).
            dummy_mamba_state_indices = (torch.zeros(
                (num_reqs, ), dtype=torch.int32).to(self.device)
                                         if self._has_mamba_state else None)
            # Match _prepare_inputs: the spec-decode mamba fields are set iff
            # the model has mamba layers and spec decoding is enabled, so the
            # dummy runs compile the same GDN program signature — including
            # the two distributions being views of one [6] base tensor (see
            # _combined_request_distribution_cpu for why).
            dummy_mamba_request_distribution = None
            if self.mamba_slot_read_offsets is not None:
                combined_device = torch.tensor([actual_num_reqs] * 6,
                                               dtype=torch.int32).to(
                                                   self.device)
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
            per_layer_attn_metadata, _unused_spec_decode_common_attn_metadata = self._build_attention_metadata(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_query_len=1,
                num_tokens_padded=num_tokens,
                num_reqs_padded=num_reqs,
                slot_mappings=slot_mappings,
            )
        else:
            # Pre-init path (called before initialize_kv_cache): use the
            # caller-supplied num_blocks and a single shared metadata.
            if self._attn_layer_names is None:
                self._attn_layer_names = list(
                    get_layers_from_vllm_config(
                        self.vllm_config,
                        (AttentionLayerBase, MambaBase)).keys())
            block_tables = torch.zeros((num_reqs * num_blocks, ),
                                       dtype=torch.int32).to(self.device)
            attn_metadata = AttentionMetadata(
                input_positions=position_ids,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
            )

            per_layer_attn_metadata = {
                layer_name: attn_metadata
                for layer_name in self._attn_layer_names
            }

        if envs.TPU_MOE_SKIP_PADDED_TOKENS:
            assert self._token_padding_state is not None
            # Every row of a dummy step is padding.
            self._token_padding_state.update(0, num_tokens)
        with (
                self.maybe_select_dummy_loras(
                    self.lora_config, np.array([num_tokens], dtype=np.int32)),
                set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens if dp_lockstep else 0,
                    num_tokens_across_dp=self._dp_num_tokens_across_dp(
                        num_tokens) if dp_lockstep else None),
                set_vllm_model_wrapper_context(mesh=self.mesh,
                                               vllm_config=self.vllm_config),
        ):
            out, _ = self.forward_model(input_ids=input_ids,
                                        positions=position_ids,
                                        inputs_embeds=inputs_embeds)
            if dp_lockstep:
                # Idle DP engines must issue the same TP logits collective as
                # busy engines before the next DP synchronization.
                _idx = torch.zeros(num_reqs,
                                   dtype=torch.int32,
                                   device=out.device)
                _ = self.compute_selected_logits(out, _idx)
            sync.synchronize(out, wait=True)
        self._hidden_states_dtype = out.dtype

    @contextlib.contextmanager
    def _precompile_timed(self, name: str):
        """Common boilerplate for each AOT bucket precompile pass."""
        logger.info("Compiling %s with different input shapes.", name)
        start = time.perf_counter()
        yield
        logger.info("Compilation finished in %.2f [secs].",
                    time.perf_counter() - start)
        self._update_num_xla_graphs(name)

    def _dummy_logits(self, num_reqs: int) -> torch.Tensor:
        return torch.zeros((num_reqs, self.vocab_size),
                           device=self.device,
                           dtype=self._hidden_states_dtype)

    def _precompile_compute_selected_logits(self) -> None:
        hsize = self.model_config.get_hidden_size()
        with self._precompile_timed("compute_selected_logits"):
            for num_tokens in self.num_tokens_paddings:
                dummy_hidden = torch.zeros((num_tokens, hsize),
                                           device=self.device,
                                           dtype=self._hidden_states_dtype)
                for num_reqs in self.num_reqs_paddings:
                    indices = torch.zeros(num_reqs,
                                          dtype=torch.int32,
                                          device=self.device)
                    out = self.compute_selected_logits(dummy_hidden, indices)
                    sync.synchronize(out, wait=True)
                    logger.info("  -- num_tokens: %d, num_seqs: %d",
                                num_tokens, num_reqs)
                    if num_reqs >= min(num_tokens, self.max_num_reqs):
                        break

    def _precompile_compute_logits_from_hidden_states(self) -> None:
        hsize = self.model_config.get_hidden_size()
        with self._precompile_timed("compute_logits_from_hidden_states"):
            for num_reqs in self.num_reqs_paddings:
                dummy_hidden = torch.zeros((num_reqs, hsize),
                                           device=self.device,
                                           dtype=self._hidden_states_dtype)
                out = self.compute_logits_from_hidden_states(dummy_hidden)
                sync.synchronize(out, wait=True)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_structured_decoding(self) -> None:
        with self._precompile_timed("structured_decoding"):
            arange = self.structured_decode_arange.to(self.device)
            for num_reqs in self.num_reqs_paddings:
                out = self.structured_decode(
                    self.require_structured_out_cpu[:num_reqs].to(self.device),
                    self.grammar_bitmask_cpu[:num_reqs].to(self.device),
                    self._dummy_logits(num_reqs),
                    arange,
                )
                sync.synchronize(out, wait=True)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_sample_from_logits(self) -> None:
        with self._precompile_timed("sample_from_logits"):
            for num_reqs in self.num_reqs_paddings:
                dummy_logits = self._dummy_logits(num_reqs)
                dummy_temperatures = torch.ones(
                    (num_reqs, 1),
                    dtype=self._hidden_states_dtype,
                    device=self.device)
                dummy_u = torch.rand_like(dummy_logits)
                dummy_top_k = torch.zeros((num_reqs, 1),
                                          dtype=torch.int32,
                                          device=self.device)
                dummy_top_p = torch.ones((num_reqs, 1),
                                         dtype=torch.float32,
                                         device=self.device)
                for all_greedy in [False, True]:
                    out = self.sample_from_logits_func(dummy_logits,
                                                       dummy_temperatures,
                                                       dummy_u,
                                                       dummy_top_k,
                                                       dummy_top_p,
                                                       all_greedy=all_greedy)
                    sync.synchronize(out, wait=True)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_gather_logprobs(self) -> None:
        with self._precompile_timed("gather_logprobs"):
            for num_reqs in self.num_reqs_paddings:
                out = self.gather_logprobs(
                    self._dummy_logits(num_reqs),
                    torch.zeros((num_reqs, 1),
                                dtype=torch.int64).to(self.device),
                )
                sync.synchronize(out.logprobs, wait=True)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_substitute_placeholder_token(self) -> None:
        if not self.scheduler_config.async_scheduling:
            return
        # next_tokens source lengths the runtime produces: stride-1 num_reqs
        # buckets (non-spec / pure-async), plus the bounded stride-(1+K) span
        # async eagle3 parks (max_num_reqs*(1+K); see sample_tokens) so the spec
        # substitute doesn't recompile per real num_reqs.
        next_lens = list(self.num_reqs_paddings)
        if (self._is_async_drafter):
            next_lens.append(
                self.max_num_reqs *
                (1 + self.speculative_config.num_speculative_tokens))
        with self._precompile_timed("substitute_placeholder_token"):
            for num_tokens in self.num_tokens_paddings:
                input_ids = torch.zeros(num_tokens,
                                        dtype=torch.int32,
                                        device=self.device)
                cur_input_indices = torch.zeros(num_tokens,
                                                dtype=torch.int32,
                                                device=self.device)
                # -1 sentinel marks every slot as padding so the precompiled
                # program is byte-identical to the runtime JIT call.
                pre_next_tokens_indices = torch.full((num_tokens, ),
                                                     -1,
                                                     dtype=torch.int32,
                                                     device=self.device)
                for nlen in next_lens:
                    next_tokens = torch.zeros(nlen,
                                              dtype=torch.int64,
                                              device=self.device)
                    out = _substitute_placeholder_token(
                        input_ids, cur_input_indices, pre_next_tokens_indices,
                        next_tokens)
                    sync.synchronize(out, wait=True)
                    logger.info("  -- num_tokens: %d, next_len: %d",
                                num_tokens, nlen)

    def _precompile_rejection_sampler(self) -> None:
        """Warm the spec-decode verify path so it doesn't recompile at runtime.

        The rejection sampler is ``@torch.compile(backend="tpu",
        dynamic=False)`` -> a fresh graph per input shape, and the
        ``logits[...]`` gathers around it are eager. None are covered by
        ``_precompile_sampling_subgraphs`` (non-spec sampling only) or
        ``drafter.precompile()`` (draft forward only), so they recompile every
        time the verify length (``padded_logits_length``) lands in a new
        num-tokens bucket as requests finish and the batch shrinks. Replay the
        verify inner body (the ``use_spec`` branch in ``sample_tokens``) at
        every ``(padded_logits_length, padded_num_reqs)`` bucket the runtime can
        produce so the real call is a cache hit. Shared sync + async.

        Both verify sub-branches are distinct ``dynamic=False`` graphs, so warm
        each: greedy (``do_sampling=False`` ->
        ``_greedy_rejection_sample_with_segment``, prelude
        ``spec_bonus_and_target_logits``) and non-greedy (``do_sampling=True``
        -> ``_random_rejection_sample_with_segment``, prelude
        ``spec_gather_bonus_and_target_logits``). Which one runs is decided per
        step by ``input_batch.all_greedy``, so both are reachable and must be
        pre-compiled here.
        """
        if not (self._is_async_drafter):
            return
        k = self.speculative_config.num_speculative_tokens
        # The verify chunk samples at most num_reqs*(K+1) positions; bound the
        # padded_logits_length sweep to the bucket that covers the full batch.
        max_logits_len = _get_padded_token_len(self.num_tokens_paddings,
                                               self.max_num_reqs * (k + 1))
        with self._precompile_timed("rejection_sampler"):
            for num_tokens in self.num_tokens_paddings:
                if num_tokens > max_logits_len:
                    break
                # padded_logits_length-shaped inputs; dtypes mirror
                # get_spec_decode_metadata so the graph is byte-identical.
                dummy_logits = torch.zeros((num_tokens, self.vocab_size),
                                           device=self.device,
                                           dtype=self._hidden_states_dtype)
                draft_token_ids = torch.zeros(num_tokens,
                                              dtype=torch.int32,
                                              device=self.device)
                target_logits_indices = torch.zeros(num_tokens,
                                                    dtype=torch.int32,
                                                    device=self.device)
                segment_ids = torch.zeros(num_tokens,
                                          dtype=torch.int64,
                                          device=self.device)
                group_indices = torch.zeros(num_tokens,
                                            dtype=torch.int32,
                                            device=self.device)
                for num_reqs in self.num_reqs_paddings:
                    # padded_num_reqs-shaped inputs.
                    draft_lengths = torch.zeros(num_reqs,
                                                dtype=torch.int32,
                                                device=self.device)
                    bonus_logits_indices = torch.zeros(num_reqs,
                                                       dtype=torch.int32,
                                                       device=self.device)
                    # --- greedy verify path (do_sampling=False) ---
                    bonus_token_ids, target_logits_warm = (
                        self.spec_bonus_and_target_logits(
                            dummy_logits, bonus_logits_indices,
                            target_logits_indices))
                    out = self.rejection_sampler(
                        draft_token_ids=draft_token_ids,
                        num_draft_tokens=draft_lengths,
                        target_logits=target_logits_warm,
                        bonus_token_ids=bonus_token_ids,
                        segment_ids=segment_ids,
                        group_indices=group_indices,
                        max_draft_tokens=k,
                    )
                    sync.synchronize(out, wait=True)
                    # --- non-greedy verify path (do_sampling=True) ---
                    _, target_logits_ng = (
                        self.spec_gather_bonus_and_target_logits(
                            dummy_logits, bonus_logits_indices,
                            target_logits_indices))
                    warm_temps = torch.zeros((num_tokens, 1),
                                             dtype=self._hidden_states_dtype,
                                             device=self.device)
                    warm_top_k = torch.zeros((num_tokens, 1),
                                             dtype=torch.int32,
                                             device=self.device)
                    warm_top_p = torch.ones((num_tokens, 1),
                                            dtype=torch.float32,
                                            device=self.device)
                    warm_accept_u = torch.zeros(num_tokens,
                                                dtype=torch.float32,
                                                device=self.device)
                    warm_recover_u = torch.zeros((num_tokens, self.vocab_size),
                                                 dtype=torch.float32,
                                                 device=self.device)
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
                    sync.synchronize(out, wait=True)
                    logger.info("  -- padded_logits_length: %d, num_seqs: %d",
                                num_tokens, num_reqs)
                    if num_reqs >= min(num_tokens, self.max_num_reqs):
                        break

    def _precompile_sampling_subgraphs(self) -> None:
        """Compile sampling-path subgraphs so their bottom-HBM reservations
        are visible to vLLM's available-memory probe in profile_run."""
        self._precompile_compute_selected_logits()
        if (_get_sequence_layout_planner_for_runner(
                self).uses_selected_logits_hidden_states):
            self._precompile_compute_logits_from_hidden_states()
        self._precompile_structured_decoding()
        self._precompile_sample_from_logits()
        self._precompile_gather_logprobs()

    def capture_model(self) -> None:
        """Precompile every torch.compile subgraph across all input buckets."""
        if self.enforce_eager:
            return
        with self.maybe_setup_dummy_loras(self.lora_config):
            with self._precompile_timed("model backbone"):
                for num_tokens in self.num_tokens_paddings:
                    logger.info("  -- num_tokens: %d", num_tokens)
                    self._dummy_run(num_tokens,
                                    self.num_reqs_max_model_len,
                                    self.max_num_blocks_per_req,
                                    use_max_model_len=True)
                    if self.most_model_len is not None:
                        self._dummy_run(num_tokens,
                                        self.num_reqs_most_model_len,
                                        self.num_blocks_per_most_len_req,
                                        use_max_model_len=False)

            self._precompile_sampling_subgraphs()

            # Warm the drafter's forward + sampling subgraphs at every
            # bucket shape it may see at runtime.
            if (self._is_async_drafter):
                self.drafter.precompile()
                self._precompile_rejection_sampler()
                # The isolated precompile above emits standalone fused programs;
                # the real per-step propose+verify+sampling dispatch fuses them
                # differently, so the first real request would otherwise compile
                # a handful of new graphs. Warm those by running the REAL
                # two-phase spec-decode path once on a synthetic request.
                self._warmup_spec_decode()

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
        if os.environ.get("SPEC_WARMUP", "1") == "0":
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
        with _suspend_kv_transfer_group(), self._precompile_timed(
                "spec-decode real warmup"):
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
                            int(P), idx=i, quiet=True, greedy=greedy):
                        # First attempt cold-compiles the draft forward; a
                        # SymInt from that cold compile can leak into the
                        # dynamic=False gather wrapper on the first bucket. Retry
                        # once now that the forward is compiled and returns
                        # concrete-shaped tensors (this attempt logs loudly if it
                        # also fails).
                        self._warmup_one_spec_request(int(P),
                                                      idx=i,
                                                      greedy=greedy)
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
        logger.info("spec-decode warmup compiled %d graphs",
                    self.num_xla_graphs - n0)

    def _warmup_spec_batch(self, R: int, P: int, greedy: bool = True) -> None:
        """Warm an nr=R decode batch: prefill R synthetic requests then decode
        them together so the multi-request fused programs are compiled.

        greedy=False routes the non-greedy (do_sampling=True) verify+sample
        fusion, which is a distinct set of programs from the greedy path."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import (CachedRequestData,
                                               NewRequestData, SchedulerOutput)

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
            group_nblks.append(
                min(cdiv(max_pos, gbs), cdiv(self.max_model_len, gbs)))
        nblk = max(group_nblks)
        top = int(self.kv_cache_config.num_blocks)
        if top <= R * nblk + 1:
            logger.warning(
                "skip spec-decode warmup R=%d: not enough kv blocks", R)
            return
        rids = [f"__spec_warmup_b{R}_{j}__" for j in range(R)]
        sp = SamplingParams(temperature=0.0 if greedy else 1.0)
        try:
            # ---- batched prefill of R requests (disjoint top-of-pool blocks) --
            new_reqs = []
            for j, rid in enumerate(rids):
                lo = top - (j + 1) * nblk
                new_reqs.append(
                    NewRequestData(req_id=rid,
                                   prompt_token_ids=[0] * P,
                                   mm_features=[],
                                   sampling_params=sp,
                                   pooling_params=None,
                                   block_ids=tuple(
                                       list(range(lo, lo + g_nblk))
                                       for g_nblk in group_nblks),
                                   num_computed_tokens=0,
                                   lora_request=None))
            # Prefill in chunks that respect the pre-allocated host token buffers.
            # A single R*P prefill overflows positions_np ([max_num_tokens]) when
            # Decode below schedules R*(1+K) <= max_num_tokens, so only
            # the prefill needs chunking; the nr=R decode shape is unchanged.
            reqs_per_chunk = max(1, self.max_num_tokens // P)
            for c0 in range(0, R, reqs_per_chunk):
                chunk = new_reqs[c0:c0 + reqs_per_chunk]
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
                creq = CachedRequestData(req_ids=list(rids),
                                         resumed_req_ids=set(),
                                         new_token_ids=[],
                                         all_token_ids=all_token_ids,
                                         new_block_ids=[None] * R,
                                         num_computed_tokens=list(nct),
                                         num_output_tokens=[t + 1] * R)
                so = SchedulerOutput.make_empty()
                so.scheduled_cached_reqs = creq
                so.num_scheduled_tokens = {rid: 1 + K for rid in rids}
                so.total_num_scheduled_tokens = R * (1 + K)
                so.scheduled_spec_decode_tokens = {
                    rid: [0] * K
                    for rid in rids
                }
                assert self.execute_model(so) is None
                self.sample_tokens(None)
                self.take_draft_token_ids()
                nct = [x + 1 for x in nct]
        except Exception:
            logger.exception("spec-decode warmup R=%d failed; continuing", R)
        finally:
            self._warmup_spec_decode_cleanup(rids)

    def _warmup_one_spec_request(self,
                                 P: int,
                                 idx: int,
                                 quiet: bool = False,
                                 greedy: bool = True) -> bool:
        """Run one synthetic spec-decode request (prefill P + decodes).

        Returns True on success, False if the warmup raised (and was caught).
        The caller retries a failed bucket once: the first attempt cold-compiles
        the draft forward, and a SymInt from that cold compile can leak into the
        dynamic=False gather wrapper on the very first bucket; the retry runs with
        the forward already compiled (returning concrete-shaped tensors).
        """
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.output import (CachedRequestData,
                                               NewRequestData, SchedulerOutput)

        rid = f"__spec_warmup_{idx}__"
        K = self.speculative_config.num_speculative_tokens
        # The contexts (num_computed_tokens) to run decode steps at: a short walk
        # near the prefill.
        room = self.max_model_len - P
        n_dec = min(max(5, K + 2), max(0, room // (1 + K) - 1))
        decode_ncts = [P + t for t in range(n_dec)]
        # Blocks from the TOP of the pool so the first real allocations (low ids)
        # don't collide. Page CONTENTS are value-insensitive and KV writes use
        # empty_slot_mappings, so stale pages are safe; each request is fully torn
        # down before the next. Cover the highest context this request reaches.
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
            block_ids_per_group.append(list(range(top - g_nblk, top)))
            max_nblk = max(max_nblk, g_nblk)
        if top <= max_nblk + 1:
            logger.warning(
                "skip spec-decode warmup P=%d: not enough kv blocks", P)
            return
        block_ids = tuple(block_ids_per_group)
        sp = SamplingParams(temperature=0.0 if greedy else 1.0)
        try:
            # ---- prefill ----
            new_req = NewRequestData(req_id=rid,
                                     prompt_token_ids=[0] * P,
                                     mm_features=[],
                                     sampling_params=sp,
                                     pooling_params=None,
                                     block_ids=block_ids,
                                     num_computed_tokens=0,
                                     lora_request=None)
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
                creq = CachedRequestData(req_ids=[rid],
                                         resumed_req_ids=set(),
                                         new_token_ids=[],
                                         all_token_ids=all_token_ids,
                                         new_block_ids=[None],
                                         num_computed_tokens=[nct],
                                         num_output_tokens=[t + 1])
                so = SchedulerOutput.make_empty()
                so.scheduled_cached_reqs = creq
                so.num_scheduled_tokens = {rid: 1 + K}
                so.total_num_scheduled_tokens = 1 + K
                so.scheduled_spec_decode_tokens = {
                    rid: [int(x) for x in list(cur)[:K]]
                }
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
                    exc_info=True)
            else:
                logger.exception("spec-decode warmup P=%d failed; continuing",
                                 P)
            return False
        finally:
            self._warmup_spec_decode_cleanup(rid)
        return True

    def _warmup_spec_decode_cleanup(self, rids) -> None:
        """Tear down the synthetic warmup request(s) so serving starts clean."""
        from vllm.v1.core.sched.output import SchedulerOutput
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
                pass
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
            logger.exception("spec-decode warmup drain failed")
        # Belt-and-suspenders: drop any synthetic state defensively.
        for r in rids:
            try:
                self.requests.pop(r, None)
            except Exception:
                pass
        for attr in ("_draft_token_ids", ):
            try:
                if hasattr(self.spec_decode_manager, attr):
                    setattr(self.spec_decode_manager, attr, None)
            except Exception:
                pass
        try:
            if self.drafter is not None:
                self.drafter.draft_chunks = None
        except Exception:
            pass
        self._pre_async_results = None
        self.mm_embed_inputs = None
        if self.input_batch.num_reqs != 0:
            logger.warning("spec-decode warmup left batch non-empty (%d)",
                           self.input_batch.num_reqs)

    @contextmanager
    def _profile_isolated_cache(self) -> Iterator[None]:
        # Isolate the kv=0-specialized profile compile into a dedicated
        # subdirectory ("profile_cache") so it can be cached across warm
        # restarts without colliding with the real serving cache (kv > 0)
        # or across different models / TP sizes / dtypes.
        import hashlib

        from vllm.compilation.caching import aot_compile_hash_factors

        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash
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

        from vllm_torchtpu.layers.vllm.linear_common import (
            _get_quantized_matmul_fp4_op, _get_quantized_matmul_op)

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
                self._dummy_run(num_tokens, self.num_reqs_max_model_len,
                                self.max_num_blocks_per_req)
            compiled = (self.model.get_language_model().model if hasattr(
                self.model, "get_language_model") else self.model.model)
            reset_compile_wrapper(compiled)
            # reset_compile_wrapper wipes cache_dir; restore so
            # capture_model can read/write the persistent cache.
            cc.cache_dir, cc.local_cache_dir = saved_cache
            cc.compile_sizes = saved_sizes
            torch._dynamo.reset()
            sync.synchronize()

            # Sampling subgraphs (Phase A) — uses the restored cache_dir.
            if not self.enforce_eager:
                with self.maybe_setup_dummy_loras(self.lora_config):
                    self._precompile_sampling_subgraphs()

    def get_kv_prewarm_shapes(self) -> list[int]:
        if not has_kv_transfer_group():
            return []
        kv_connector = get_kv_transfer_group()
        cw = getattr(kv_connector, "connector_worker", None)
        if (cw is None or not hasattr(cw, "spec")
                or not hasattr(cw.spec, "prewarm_shapes")):
            return []
        return cw.spec.prewarm_shapes

    def prewarm_kv_offload_shape(self, p: int) -> None:
        if not has_kv_transfer_group():
            return
        kv_connector = get_kv_transfer_group()
        cw = getattr(kv_connector, "connector_worker", None)
        if (cw is not None and hasattr(cw, "spec")
                and hasattr(cw.spec, "prewarm_shape")):
            cw.spec.prewarm_shape(p)

    def _resolve_tpu_group_backend(
        self,
        layer_names: list[str],
        kv_cache_spec: KVCacheSpec,
    ) -> type[Any]:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(self.vllm_config, layer_type,
                                             layer_names)
        if layer_names and layer_names[0] in layers:
            return cast(type[Any], layers[layer_names[0]].get_attn_backend())
        if isinstance(kv_cache_spec, (AttentionSpec, MambaSpec)):
            return cast(type[Any], PallasAttentionBackend)
        raise NotImplementedError(
            f"Unsupported KV cache spec: {type(kv_cache_spec)!r}")

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig,
                                     kernel_block_sizes: list[int]) -> None:
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
        super().may_reinitialize_input_batch(kv_cache_config,
                                             kernel_block_sizes)
        if hasattr(self, "input_batch") and self.input_batch is not None:
            if hasattr(self.input_batch, "block_table"):
                try:
                    self.max_num_blocks_per_req = int(
                        self.input_batch.block_table[0].get_cpu_tensor(
                        ).shape[1])
                    if (self.block_table_cpu.shape[0] != self.max_num_reqs
                            or self.block_table_cpu.shape[1]
                            != self.max_num_blocks_per_req):
                        self.block_table_cpu = torch.zeros(
                            (self.max_num_reqs, self.max_num_blocks_per_req),
                            dtype=torch.int32,
                            device="cpu")
                except (IndexError, TypeError, AttributeError, KeyError):
                    pass

    def _initialize_unified_kv_cache(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        self.attn_groups = []
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            layer_names = list(group.layer_names)
            backend = self._resolve_tpu_group_backend(layer_names,
                                                      group.kv_cache_spec)
            if isinstance(group.kv_cache_spec,
                          AttentionSpec) and self.use_spmd:
                num_kv_heads = group.kv_cache_spec.num_kv_heads
                parallel_config = self.parallel_config
                tp_size = parallel_config.tensor_parallel_size
                assert num_kv_heads % tp_size == 0, (
                    f"num_kv_heads {num_kv_heads} must be divisible by "
                    f"tp_size {tp_size} under SPMD mode")
            self.attn_groups.append([
                AttentionGroup(
                    backend=backend,
                    layer_names=layer_names,
                    kv_cache_spec=group.kv_cache_spec,
                    kv_cache_group_id=gid,
                    metadata_builders=[],
                )
            ])

        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups)
        self._kernel_block_sizes = kernel_block_sizes
        kernel_block_size_by_gid = build_kernel_block_size_by_group_id(
            kv_cache_config=kv_cache_config,
            kernel_block_sizes=kernel_block_sizes,
        )
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)

        for group_list in self.attn_groups:
            for group in group_list:
                kv_cache_spec = group.kv_cache_spec
                kernel_block_size = kernel_block_size_by_gid.get(
                    group.kv_cache_group_id)
                if (isinstance(kv_cache_spec, AttentionSpec)
                        and kernel_block_size is not None):
                    kv_cache_spec = kv_cache_spec.copy_with_new_block_size(
                        kernel_block_size)
                builder = AttentionMetadataBuilder(
                    kv_cache_spec,
                    group.layer_names,
                    self.vllm_config,
                    self.device,
                    runner=self,
                    kv_cache_group_id=group.kv_cache_group_id,
                )
                group.metadata_builders.append(builder)

        for group_id in range(len(kv_cache_config.kv_cache_groups)):
            assert (self.block_table_cpu.dtype == self.input_batch.
                    block_table[group_id].get_cpu_tensor().dtype)

        materialized = materialize_kv_cache_tensors(
            kv_cache_config=kv_cache_config,
            attn_groups=self.attn_groups,
            kernel_block_sizes=kernel_block_sizes,
            device=self.device,
            cache_dtype=self.kv_cache_dtype,
        )
        kv_caches = materialized.kv_caches
        self.kv_cache_raw_tensors = materialized.raw_tensors
        if self._unified_kv_layout:
            self._build_mamba_copy_plan(kv_cache_config,
                                        materialized.raw_tensors)
        if kv_cache_config.has_mamba_layers:
            self._update_hybrid_attention_mamba_layout(kv_caches,
                                                       kernel_block_sizes)

        for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
        ):
            logger.debug("%s reuses KV cache of %s", layer_name,
                         target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        if self._unified_kv_layout:
            # Flush the pool zero-fill before any compiled execution. PJRT
            # only donates quiescent buffers: a pending write at enqueue time
            # makes every donated pool parameter fall back to a fresh
            # pool-sized output copy (a 2x-pool transient that OOMs at high
            # gpu_memory_utilization).
            for raw in self.kv_cache_raw_tensors:
                sync.synchronize(raw, wait=True)

        logger.info(
            "%s",
            format_kv_cache_layout_summary(
                kv_cache_config=kv_cache_config,
                kv_caches=kv_caches,
                raw_tensors=self.kv_cache_raw_tensors,
                attn_groups=self.attn_groups,
            ),
        )

        self.kv_caches = []
        utils.tpu_bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )

        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.register_kv_caches(kv_caches)
            if hasattr(kv_connector, "set_host_xfer_buffer_ops"):
                kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)
            if hasattr(kv_connector, "register_runner"):
                kv_connector.register_runner(self)

        # For hybrid models with spec decoding on the unified pool, keep a
        # per-block device buffer of mamba read offsets (num_accepted - 1
        # from each request's last verify step), indexed by manager block
        # id. The GDN kernel reads a request's initial state from checkpoint
        # `offset` of its state block, which is how rejected draft tokens
        # are rolled back (by selecting the checkpoint of the last accepted
        # token, never by copying state). The offsets follow the state
        # block through the align-mode seed copies (see
        # `_collect_mamba_state_seed_copies`).
        if (self._unified_kv_layout and kv_cache_config.has_mamba_layers
                and self.speculative_config is not None):
            self.mamba_slot_read_offsets = torch.zeros(
                kv_cache_config.num_blocks, dtype=torch.int32).to(self.device)

        if not self.enforce_eager:
            self._precompile_substitute_placeholder_token()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        assert kv_cache_config.num_blocks is not None, (
            "KVCacheConfig.num_blocks must be resolved by the scheduler")
        kv_cache_config = copy.deepcopy(kv_cache_config)
        # Mirror GPUModelRunner.initialize_kv_cache: needed by inherited
        # _update_states -> _may_reorder_batch which reads kv_cache_config.
        self.kv_cache_config = kv_cache_config
        self._maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)

        # Dummy slot mapping, not used anywhere in the TPU code flow. But needed
        #  for upstream `_build_attention_metadata` call.
        self.empty_slot_mappings = {
            gid: torch.empty(0, device=self.device)
            for gid in range(len(self.kv_cache_config.kv_cache_groups))
        }
        if self._unified_kv_layout:
            self._initialize_unified_kv_cache(kv_cache_config)
            return

        attn_block_size = None
        for group in kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                # We can safely ignore block size for Mamba layers since they only use a single cache state per sequence.
                continue
            if not isinstance(spec, AttentionSpec) and len(
                    kv_cache_config.kv_cache_groups) > 1:
                raise NotImplementedError(
                    "Only AttentionSpec and MambaSpec are supported in KV cache groups > 1."
                )

            block_size = spec.block_size
            if attn_block_size is None:
                attn_block_size = block_size
            assert attn_block_size == block_size, "Block size across attention groups must be the same."

        block_sizes = [
            group.kv_cache_spec.block_size
            for group in kv_cache_config.kv_cache_groups
        ]

        self.may_reinitialize_input_batch(kv_cache_config, block_sizes)

        # Populate self.attn_groups directly with a single shared TPU builder
        # per group; bypasses parent's initialize_attn_backend (per-backend
        # get_builder_cls dispatch, cudagraph-mode resolution, cp compat
        # checks) — all GPU-relevant and unused on TPU.
        self.attn_groups = []
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            builder = AttentionMetadataBuilder(group.kv_cache_spec,
                                               group.layer_names,
                                               self.vllm_config,
                                               self.device,
                                               runner=self,
                                               kv_cache_group_id=gid)
            self.attn_groups.append([
                AttentionGroup(
                    backend=None,
                    layer_names=list(group.layer_names),
                    kv_cache_spec=group.kv_cache_spec,
                    kv_cache_group_id=gid,
                    metadata_builders=[builder],
                )
            ])

        # Verify dtype compatibility between block_table_cpu and every
        # per-group block table that downstream code may index.
        for group_id in range(len(kv_cache_config.kv_cache_groups)):
            assert (self.block_table_cpu.dtype == self.input_batch.
                    block_table[group_id].get_cpu_tensor().dtype)

        layer_name_to_spec = {}
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.shared_kv_cache_layers:
                    continue
                if hasattr(group.kv_cache_spec, "kv_cache_specs"):
                    layer_name_to_spec[
                        layer_name] = group.kv_cache_spec.kv_cache_specs[
                            layer_name]
                else:
                    layer_name_to_spec[layer_name] = group.kv_cache_spec

        kv_caches: dict[str, torch.Tensor] = {}
        # Actual leading-dim block count used for mamba state arrays, captured
        # at allocation time so the slot pool below is sized to exactly what
        # was allocated (compact `_mamba_num_blocks`, or uniform `num_blocks`).
        allocated_mamba_num_blocks: int | None = None

        def _per_layer_spec(layer_name: str) -> KVCacheSpec:
            spec = layer_name_to_spec[layer_name]
            if isinstance(spec, UniformTypeKVCacheSpecs):
                return spec.kv_cache_specs[layer_name]
            return spec

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # If the KV cache tensor is shared by multiple layers, then we
            # duplicate cache for each layer and `num_blocks` is calculated
            # based on the total size of the shared cache.
            # Otherwise, `num_blocks` is calculated based on the size of the
            # single layer's KV cache spec.
            tensor_size = kv_cache_tensor.size
            shared_by = kv_cache_tensor.shared_by
            if len(shared_by) > 1:
                total_group_page_size = 0
                for name in shared_by:
                    spec = layer_name_to_spec[name]
                    # Use the per-layer *TPU-actual* per-block bytes so the
                    # sum equals the `page_size_padded` that
                    # `update_mamba_page_size_padded` installed on every
                    # spec (== attn_page + N × mamba_unpadded). For
                    # attention, the TPU-actual size includes dtype-
                    # specific packing (e.g., fp8 KV packs 4 elements per
                    # 32-bit word) which `spec.real_page_size_bytes`
                    # doesn't account for — on fp8 models they differ by
                    # 2×, which would break the num_blocks match here.
                    if isinstance(spec, MambaSpec):
                        total_group_page_size += dataclasses.replace(
                            spec, page_size_padded=None).page_size_bytes
                    elif isinstance(spec, AttentionSpec):
                        total_group_page_size += PallasAttentionBackend.get_kv_cache_page_size_bytes(
                            spec.block_size,
                            spec.num_kv_heads,
                            spec.head_size,
                            spec.dtype,
                        )
                    else:
                        raise NotImplementedError
                num_blocks = tensor_size // total_group_page_size
            else:
                num_blocks = kv_cache_config.num_blocks

            for layer_name in shared_by:
                kv_cache_spec = _per_layer_spec(layer_name)

                if isinstance(kv_cache_spec, MambaSpec):
                    # Compact-mamba: allocate only `_mamba_num_blocks`
                    # recurrent slots (= max_num_reqs + 1) when the override
                    # succeeded; otherwise fall back to the uniform
                    # `num_blocks`. Attention layers always keep `num_blocks`.
                    mamba_num_blocks = (self._mamba_num_blocks
                                        if self._mamba_num_blocks is not None
                                        else num_blocks)
                    allocated_mamba_num_blocks = mamba_num_blocks
                    mamba_states = []
                    for _, (shape, dtype) in enumerate(
                            zip(kv_cache_spec.shapes, kv_cache_spec.dtypes)):
                        cache_shape = (mamba_num_blocks, *shape)
                        mamba_states.append(
                            torch.zeros(cache_shape,
                                        dtype=dtype).to(self.device))
                    kv_caches[layer_name] = tuple(mamba_states)
                elif isinstance(kv_cache_spec, MLAAttentionSpec):
                    # SPMD Cache Invariance Details for Multi-Head Latent Attention (MLA):
                    # Because MLA maps all attention heads onto a single joint compressed latent key-value
                    # representation (`num_kv_heads=1`), the physical KV cache dimension never splits
                    # across tensor parallel ranks (`tp_size`) during SPMD graph execution (`self.use_spmd`).
                    # Each device partition consistently retains a complete, unsliced replication of the
                    # compressed latent cache structure across multi-chip execution loops.
                    kv_cache_shape = PallasMLAttentionBackend.get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        kv_cache_spec.dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    tpu_kv_cache = torch.zeros(kv_cache_shape,
                                               dtype=dtype).to(self.device)

                    kv_caches[layer_name] = tpu_kv_cache
                elif isinstance(kv_cache_spec, AttentionSpec):
                    if self.use_spmd:
                        num_kv_heads = kv_cache_spec.num_kv_heads
                        parallel_config = self.parallel_config
                        tp_size = parallel_config.tensor_parallel_size
                        # TODO: Handle kv cache duplication under SPMD mode.
                        assert num_kv_heads % tp_size == 0, (
                            f"num_kv_heads {num_kv_heads} must be divisible by "
                            f"tp_size {tp_size} under SPMD mode")
                    kv_cache_shape = PallasAttentionBackend.get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        kv_cache_spec.dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    tpu_kv_cache = torch.zeros(kv_cache_shape,
                                               dtype=dtype).to(self.device)

                    kv_caches[layer_name] = tpu_kv_cache
                else:
                    raise NotImplementedError

        self._add_shared_kv_cache_aliases(kv_caches)

        # Mark KV cache buffers as donation candidates outside torch.compile
        # regions to avoid Dynamo tracing through pybind calls.
        # TODO(geyuhao): Comment out for now as TorchTPU does not support this right now
        # for kv_cache in kv_caches.values():
        #     pallas.set_buffer_donor_(kv_cache, True)

        # Reset kv_caches list (tpu_bind_kv_cache expects empty list)
        self.kv_caches = []

        # Use tpu_bind_kv_cache to bind KV caches to attention layers using layer names
        # This is the native vLLM pattern and avoids the 'layer_id' attribute error
        utils.tpu_bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches,
        )

        if self.use_spmd:
            # Shard KV Cache
            for cache in self.kv_caches:
                continue
                # xs.mark_sharding(cache, self.mesh, (None, "x", None, None))

        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.register_kv_caches(kv_caches)
            # TPUConnector reads runner.kv_caches lazily and doesn't need
            # set_host_xfer_buffer_ops; only call it on connectors that
            # expose it.
            if hasattr(kv_connector, "set_host_xfer_buffer_ops"):
                kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)
            if hasattr(kv_connector, "register_runner"):
                kv_connector.register_runner(self)

        # Initialize the compact-mamba slot allocator now that the true mamba
        # block count is known. When compact sizing was skipped, mamba shares
        # the attention `num_blocks`, so the pool spans that range instead.
        if (allocated_mamba_num_blocks is not None
                and not self._uniform_mamba_layout):
            self._init_mamba_slot_pool(allocated_mamba_num_blocks)

        # Compact pool counterpart of the read-offset buffer allocated in
        # `initialize_kv_cache`, indexed over the mamba slot pool rather
        # than the unified pool's blocks: here checkpoint `offset` of a
        # request is the slot `base_slot + offset`.
        if (allocated_mamba_num_blocks is not None
                and self.speculative_config is not None):
            if self._uniform_mamba_layout:
                raise NotImplementedError(
                    "Speculative decoding with mamba layers requires the "
                    "compact mamba slot layout (unsupported with "
                    "kv_transfer_config / uniform mamba layout).")
            self.mamba_slot_read_offsets = torch.zeros(
                allocated_mamba_num_blocks, dtype=torch.int32).to(self.device)

        # Precompile after KV cache allocation so XLA's buffer assignment sees
        # the same HBM pressure as runtime.
        if not self.enforce_eager:
            self._precompile_substitute_placeholder_token()

    def _token_padding_update(self, num_tokens_padded: int) -> None:
        """Record which of this step's token rows are padding."""
        if self._token_padding_state is None:
            return
        plan = self._last_sequence_layout_plan
        if plan is not None and plan.kind is SequenceLayoutKind.ALL:
            num_valid_tokens = min(plan.local_num_tokens, num_tokens_padded)
        else:
            # Padding-mask support is currently limited to the ALL layout.
            num_valid_tokens = num_tokens_padded
        self._token_padding_state.update(num_valid_tokens, num_tokens_padded)

    def forward_model(self, input_ids, positions, inputs_embeds=None):
        # @support_torch_compile annotations will be put on the vLLM model if it
        # supports torch.compile
        out = self.model(input_ids=input_ids,
                         positions=positions,
                         inputs_embeds=inputs_embeds)
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
            self.input_batch.temperature_cpu_tensor[cur_start_idx:cur_end_idx])
        active_top_k = self.input_batch.top_k_cpu_tensor[
            cur_start_idx:cur_end_idx]
        top_k_cpu[:num_active_reqs].copy_(
            torch.where(active_top_k >= self.vocab_size,
                        torch.zeros_like(active_top_k), active_top_k))
        top_p_cpu[:num_active_reqs].copy_(
            self.input_batch.top_p_cpu_tensor[cur_start_idx:cur_end_idx])

        return (
            temps_cpu.unsqueeze(1).to(logits.device, non_blocking=True),
            top_k_cpu.unsqueeze(1).to(logits.device, non_blocking=True),
            top_p_cpu.unsqueeze(1).to(logits.device, non_blocking=True),
        )

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def compute_selected_logits(
            self, hidden_states: torch.Tensor,
            indices_do_sample: torch.Tensor) -> torch.Tensor:
        if self.is_pooling_model:
            return torch.empty((0, ), device=hidden_states.device)
        selected = torch.index_select(hidden_states, 0, indices_do_sample)
        return self.model.compute_logits(selected)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def compute_logits_from_hidden_states(
            self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def spec_bonus_and_target_logits(
        self, logits: torch.Tensor, bonus_logits_indices: torch.Tensor,
        target_logits_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Spec-decode verify prelude: gather the bonus-token logits + argmax, and
        # gather the target_logits rows for the rejection sampler, in ONE compiled
        # region keyed on [num_tokens, num_reqs] shapes. Run raw (the
        # logits[indices] gathers + argmax at the verify site), these are eager
        # ops torch-tpu's DEFER_AND_FUSE fuses with the live per-step dispatch
        # into per-context programs (the cold target-side recompiles). index_select
        # is value-identical to logits[indices]. Same pattern as the draft wraps.
        bonus_token_ids = torch.argmax(torch.index_select(
            logits, 0, bonus_logits_indices),
                                       dim=-1)
        target_logits = torch.index_select(logits, 0, target_logits_indices)
        return bonus_token_ids, target_logits

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def spec_gather_bonus_and_target_logits(
        self, logits: torch.Tensor, bonus_logits_indices: torch.Tensor,
        target_logits_indices: torch.Tensor
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

    def _apply_temperature(self, logits: torch.Tensor,
                           temperatures: torch.Tensor) -> torch.Tensor:
        safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
        return logits / safe_temperatures

    # TODO: Under SPMD mode, sample_from_logits has correctness issue.
    #       Re-enable the torch.compile once the issue is fixed in torchxla.
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def sample_from_logits(self,
                           logits: torch.Tensor,
                           temperatures: torch.Tensor,
                           u: torch.Tensor,
                           top_k: torch.Tensor,
                           top_p: torch.Tensor,
                           all_greedy: bool = False) -> torch.Tensor:
        """
        Sample with xla-friendly function. This function is to be traced
        separately from `forward` for lighter compilation overhead.
        """
        if all_greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        is_greedy = temperatures <= SAMPLING_EPS
        scaled_logits = self._apply_temperature(logits, temperatures)
        masked_logits = apply_top_k_top_p(scaled_logits, top_k, top_p)
        u_clamped = torch.clamp(u,
                                min=torch.finfo(u.dtype).tiny,
                                max=1.0 - torch.finfo(u.dtype).eps)
        gumbel_noise = -torch.log(-torch.log(u_clamped))
        noisy_logits = masked_logits + gumbel_noise
        final_logits = torch.where(is_greedy, logits, noisy_logits)
        return torch.argmax(final_logits, dim=-1, keepdim=True)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def gather_logprobs(self, logits: torch.Tensor,
                        sampled_tokens: torch.Tensor) -> LogprobsTensors:
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
            topk_logits, topk_indices = torch.topk(logits,
                                                   max_logprobs,
                                                   dim=-1)
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

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def structured_decode(
        self,
        require_struct_decoding: torch.Tensor,
        grammar_bitmask: torch.Tensor,
        logits: torch.Tensor,
        arange: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            require_struct_decoding,
            self.apply_grammar_bitmask(logits, grammar_bitmask, arange),
            logits,
        )

    def apply_grammar_bitmask(self, logits: torch.Tensor,
                              grammar_bitmask: torch.Tensor,
                              arange: torch.Tensor):
        assert logits.shape[0] == grammar_bitmask.shape[0]
        logits_cloned = logits.clone()
        for i in range(logits.shape[0]):
            unpacked_bitmask = (torch.bitwise_right_shift(
                grammar_bitmask[i][:, None], arange[None, :])
                                & 1) == 0
            unpacked_bitmask = unpacked_bitmask.reshape(-1)[:self.vocab_size]
            logits_cloned[i] = logits_cloned[i].masked_fill(
                unpacked_bitmask, -float("inf"))
        return logits_cloned

    def prepare_structured_decoding_input(
        self, logits: torch.Tensor, grammar_output: "GrammarOutput"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grammar_bitmask = grammar_output.grammar_bitmask
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.grammar_bitmask_cpu.zero_()
        self.require_structured_out_cpu.zero_()

        cumulative_mask_idx = 0
        for req_id in grammar_output.structured_output_request_ids:
            if req_id not in self.input_batch.req_id_to_index:
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            self.grammar_bitmask_cpu[batch_index] = torch.from_numpy(
                grammar_bitmask[cumulative_mask_idx])
            # It's not guaranteed that all requests in this batch require
            # structured output, so create a bool tensor to represent
            # the requests that need structured output.
            self.require_structured_out_cpu[batch_index] = True
            cumulative_mask_idx += 1

        return (
            self.require_structured_out_cpu[:num_reqs].to(logits.device),
            self.grammar_bitmask_cpu[:num_reqs].to(logits.device),
            self.structured_decode_arange.to(logits.device),
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


def _get_padded_num_kv_cache_update_slices(num_tokens: int, max_num_reqs: int,
                                           page_size: int) -> int:
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
