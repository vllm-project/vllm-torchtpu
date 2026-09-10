# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ensure environment overrides are applied before any other imports,
# especially torch_tpu which might read them at import time.
import vllm_torchtpu.env_override  # noqa: F401  # isort: skip

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# TODO: Remove this after jax dependency is removed
import jax
import numpy as np
import torch
import torch_tpu  # noqa: F401
import vllm.envs as vllm_envs
# TODO: Remove this after jax dependency is removed
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, DraftTokenIds,
                             ModelRunnerOutput)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_torchtpu import envs
from vllm_torchtpu.distributed import utils as dist_utils
from vllm_torchtpu.layers.common.sequence_layout import \
    create_sequence_layout_planner
from vllm_torchtpu.layers.vllm import token_padding
from vllm_torchtpu.layers.vllm.attention import (TPU_STR_DTYPE_TO_TORCH_DTYPE,
                                                 PallasAttentionBackend)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator
from vllm_torchtpu.platforms.tpu_block_size_utils import \
    unified_kv_layout_enabled
from vllm_torchtpu.runner import utils as runner_utils
from vllm_torchtpu.runner.async_substitution import (
    AsyncSubstitutionMixin, _substitute_placeholder_token)
from vllm_torchtpu.runner.compile_warmup import CompileWarmupMixin
from vllm_torchtpu.runner.dp_lockstep import DpLockstepMixin
from vllm_torchtpu.runner.ds_v4_kv_cache import (DsV4KvCacheMixin,
                                                 is_cache_for_ds_v4)
from vllm_torchtpu.runner.input_preparation import (
    InputPreparationMixin, _get_sequence_layout_planner_for_runner)
from vllm_torchtpu.runner.kv_cache_init import KvCacheInitMixin
from vllm_torchtpu.runner.kv_cache_spec import (KvCacheSpecMixin,
                                                _warn_if_kv_cache_is_padded)
from vllm_torchtpu.runner.kv_connector_glue import (
    _KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP, KvConnectorMixin,
    _build_kv_connector_output, _suspend_kv_transfer_group)
from vllm_torchtpu.runner.mamba_state import (
    MambaStateMixin, _reset_read_offsets, _reset_read_offsets_compiled,
    _reset_read_offsets_impl, _rollback_offsets_migrate,
    _rollback_offsets_migrate_compiled, _rollback_offsets_migrate_impl,
    _rollback_offsets_seed, _rollback_offsets_seed_compiled,
    _rollback_offsets_seed_impl, _tpu_compile)
from vllm_torchtpu.runner.model_loading import (ModelLoadingMixin,
                                                _torch_tpu_wrapper,
                                                _validate_libtpu_version)
from vllm_torchtpu.runner.padding import (
    MIN_NUM_SEQS, _get_num_slices_per_kv_cache_update_block,
    _get_padded_num_kv_cache_update_slices,
    _get_padded_num_reqs_with_upper_limit, _get_padded_token_len,
    _get_req_paddings, _prev_power_of_2)
from vllm_torchtpu.runner.phased_profiling import PhasedProfilingMixin
from vllm_torchtpu.runner.sampling import SAMPLING_EPS, SamplingMixin
from vllm_torchtpu.runner.spec_warmup import (SpecWarmupMixin,
                                              _spec_warmup_all_token_ids)
from vllm_torchtpu.runner.speculative_decoding_manager import (
    SpecDecodeMetadata, SpeculativeDecodingManager)
from vllm_torchtpu.runner.structured_decoding_manager import \
    StructuredDecodingManager
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncPreResults, AsyncTPUModelRunnerOutput)
from vllm_torchtpu.runner.xla_graph_tracker import XlaGraphTrackerMixin
from vllm_torchtpu.spec_decode.utils import (DraftChunkInputs,
                                             normalize_draft_config)
from vllm_torchtpu.tracing.annotation import TraceAnnotation
from vllm_torchtpu.tracing.utils import (extract_kv_lens_for_tracing,
                                         extract_request_ids_for_tracing)
from vllm_torchtpu.utils import synchronize_device, synchronize_tensors

__all__ = [
    'TPUModelRunner', 'ExecuteModelState', 'AsyncPreResults',
    'AsyncTPUModelRunnerOutput', 'MIN_NUM_SEQS', 'SAMPLING_EPS',
    '_KV_CONNECTOR_OUTPUT_SUPPORTS_INVALID_BLOCK_GROUP',
    '_build_kv_connector_output', '_get_num_slices_per_kv_cache_update_block',
    '_get_padded_num_kv_cache_update_slices',
    '_get_padded_num_reqs_with_upper_limit', '_get_padded_token_len',
    '_get_req_paddings', '_get_sequence_layout_planner_for_runner',
    '_prev_power_of_2', '_reset_read_offsets', '_reset_read_offsets_compiled',
    '_reset_read_offsets_impl', '_rollback_offsets_migrate',
    '_rollback_offsets_migrate_compiled', '_rollback_offsets_migrate_impl',
    '_rollback_offsets_seed', '_rollback_offsets_seed_compiled',
    '_rollback_offsets_seed_impl', '_spec_warmup_all_token_ids',
    '_substitute_placeholder_token', '_suspend_kv_transfer_group',
    '_torch_tpu_wrapper', '_tpu_compile', '_validate_libtpu_version',
    '_warn_if_kv_cache_is_padded', 'is_cache_for_ds_v4'
]

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


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


logger = init_logger(__name__)


# Recompilation-avoidance contract:
#   1. Input prep happens on CPU; H2D via `cpu_tensor.to(xla_device)`.
#   2. Forward is split into 4 `@torch.compile(backend="tpu")` subgraphs
#      (backbone, compute_selected_logits, sample_from_logits/structured_decode,
#      gather_logprobs) so _dummy_run and execute_model trace identically.
#   3. `_dummy_run` exercises every padding bucket so all shapes are AOT-
#      compiled before the first real request.
class TPUModelRunner(PhasedProfilingMixin, XlaGraphTrackerMixin,
                     KvConnectorMixin, DpLockstepMixin, DsV4KvCacheMixin,
                     ModelLoadingMixin, InputPreparationMixin, MambaStateMixin,
                     AsyncSubstitutionMixin, SamplingMixin, SpecWarmupMixin,
                     CompileWarmupMixin, KvCacheSpecMixin, KvCacheInitMixin,
                     GPUModelRunner):

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
        self._pcp_mtp_k1_enabled = (PcpStaticSupportValidator.from_vllm_config(
            vllm_config).pcp_mtp_k1_enabled)
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
        # DSpark/DFlash checkpoints may spell the mask token dspark_noise_token_id;
        # the parent __init__ transiently builds an upstream EagleProposer
        # that raises unless dflash_config.mask_token_id resolves.
        if (vllm_config.speculative_config is not None and
                vllm_config.speculative_config.method in ("dflash", "dspark")):
            normalize_draft_config(
                vllm_config.speculative_config.draft_model_config.hf_config)
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
        # DecodeBench fills local state instead of transferring manager blocks,
        # so it can retain the compact request-indexed Mamba layout.
        transfer_config = self.vllm_config.kv_transfer_config
        self._uniform_mamba_layout: bool = (transfer_config is not None
                                            and transfer_config.kv_connector
                                            != "DecodeBenchConnector")
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
        # Each entry seeds one set of raw pool buffers with the same
        # (src, dst) pairs in one program: (raws, src_ids, dst_ids).
        self._pending_mamba_state_copies: list[tuple[list[torch.Tensor],
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
        self._block_table_stage_cpu: torch.Tensor | None = None
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

        self.sample_from_logits_func = self.sample_from_logits

        # TPU async-scheduling state (passed between execute_model and
        # sample_tokens, mirroring vLLM's async path).
        self.mm_embed_inputs: tuple[list[torch.Tensor],
                                    torch.Tensor] | None = None
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
        self._combined_request_distribution_cpu = torch.zeros(
            6, dtype=torch.int32)
        self._decode_device_cache_key: tuple | None = None
        # Snapshot of the in-flight uploads from the host staging tensors
        # (input_ids_cpu, positions_cpu, seq_lens_cpu, ...); the next chunk
        # waits on it before rewriting them.
        self._input_staging_fence: Any = None
        self._cached_query_start_loc: torch.Tensor | None = None
        self._cached_logits_indices: torch.Tensor | None = None
        self._cached_request_distribution: torch.Tensor | None = None

        self.speculative_config = self.vllm_config.speculative_config
        self.spec_decode_manager = SpeculativeDecodingManager(self)
        self._init_speculative_decoding()
        self.structured_decoding_manager = StructuredDecodingManager(self)

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

    # ----- Backend hooks overridden from GPUModelRunner -----

    def _init_device_properties(self) -> None:
        # GPU sets self.num_sms from torch.cuda.get_device_properties; TPU
        # has no SM concept. The parent's call here is during __init__ but
        # we still want a no-op override so subclass invariants hold.
        pass

    def _sync_device(self) -> None:
        synchronize_device()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> Any:
        # Disable vLLM's native GPU async spec-decode num_computed_tokens
        # correction. The TPU runner does its own async rejection correction,
        # so letting the base also correct would double-count.
        for req_state in self.requests.values():
            req_state.prev_num_draft_len = 0
        self._install_spec_token_room_guard()
        return super()._update_states(scheduler_output)

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
            has_encoder_inputs = bool(
                scheduler_output.scheduled_encoder_inputs)
            if has_encoder_inputs:
                synchronize_tensors()
            self._execute_mm_encoder(scheduler_output)
            if has_encoder_inputs:
                synchronize_tensors()

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

        local_num_chunks, local_max_chunk_reqs = self._count_input_chunks(
            scheduler_output)
        self._dp_target_bucket, target_num_chunks = self._dp_coordinated_step(
            scheduler_output.total_num_scheduled_tokens,
            local_num_chunks,
            # Exact per-chunk request bound on this rank; the DP-wide MAX of
            # it sizes every rank's draft forwards.
            local_num_reqs=local_max_chunk_reqs)
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
             spec_decode_metadata, request_major_input_ids,
             draft_sequence_layout_plan) = (self._prepare_inputs(
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

            self._record_input_staging_fence()
            input_ids, inputs_embeds = self._get_model_inputs(
                input_ids, chunk_mm_inputs)
            # Run the decoder
            # set_forward_context: vLLM's native context for attention metadata
            # set_vllm_model_wrapper_context: TPU-specific context for mesh info
            # For multimodal models _get_model_inputs returns input_ids=None
            # and packs the tokens into inputs_embeds; use whichever exists.
            num_tokens_padded = (input_ids if input_ids is not None else
                                 inputs_embeds).shape[0]
            self._token_padding_update(num_tokens_padded)

            trace_kwargs = {}
            if TraceAnnotation.is_enabled():
                trace_kwargs = extract_request_ids_for_tracing(
                    self.input_batch.req_ids, start_index, num_reqs)
                trace_kwargs.update(
                    extract_kv_lens_for_tracing(
                        self.input_batch.num_computed_tokens_cpu, start_index,
                        num_reqs))

            if self.phase_based_profiler and 'batch_composition_stats' in locals(
            ):
                stats_map = {
                    "num_prefill_tokens": "num_prefill_tokens",
                    "num_decode_tokens": "num_decode_tokens",
                    "phase": "phase",
                    "batch_id": "batch_id",
                    "total_num_scheduled_tokens": "total_num_scheduled_tokens",
                    "padded_total_num_scheduled_tokens":
                    "padded_total_num_scheduled_tokens",
                    "min_kv_len": "min_kv_length",
                }
                for src_key, target_key in stats_map.items():
                    trace_kwargs[target_key] = batch_composition_stats.get(
                        src_key, "UNKNOWN" if src_key == "phase" else 0)

            trace_name = f"ModelForward: {num_reqs} reqs, {num_tokens_padded} toks"
            if "min_kv_len" in trace_kwargs:
                trace_name += (f", kv_len min={trace_kwargs['min_kv_len']} "
                               f"avg={trace_kwargs['avg_kv_len']} "
                               f"max={trace_kwargs['max_kv_len']}")

            with TraceAnnotation(
                    name=trace_name,
                    num_reqs=num_reqs,
                    **trace_kwargs,
            ), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=self._dp_num_tokens_across_dp(
                        num_tokens_padded),
            ), set_vllm_model_wrapper_context(
                    mesh=self.mesh,
                    vllm_config=self.vllm_config,
                    kv_cache_bundle=self._kv_cache_bundle,
            ):
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
                draft_chunk = DraftChunkInputs(
                    input_ids=(request_major_input_ids
                               if request_major_input_ids is not None else
                               draft_input_ids_src),
                    position_ids=self.position_ids,
                    query_start_loc_np=self.query_start_loc_np[:num_reqs +
                                                               1].copy(),
                    attn_ctx=self._attn_metadata_builder_ctx,
                    start_index=start_index,
                    num_reqs=num_reqs,
                    aux_hidden_states=(aux_hidden_states if aux_hidden_states
                                       is not None else []),
                    hidden_states=hidden_states,
                    draft_lengths=(spec_decode_metadata.draft_lengths if
                                   spec_decode_metadata is not None else None),
                    attn_metadata=attn_metadata,
                    sequence_layout_plan=draft_sequence_layout_plan,
                )
                draft_chunks.append(draft_chunk)

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
