# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Trimmed copy of vllm/v1/worker/gpu_model_runner.py from vLLM v0.33.0. It keeps
# only the GPUModelRunner methods TPUModelRunner (or vLLM's WorkerBase, by name)
# reaches. __init__, _update_states and get_model drop the GPU-only drafter,
# sampler, CUDA graph and n-gram GPU code; other bodies are unchanged. See
# vllm_torchtpu/runner/vllm_v1/__init__.py.


import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
)
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import (
    VllmConfig,
)
from vllm.config.cache import CacheConfig
from vllm.config.ec_manager_config import EncoderCacheManagerMetadata
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
)
from vllm.lora.layers import LoRAMapping, LoRAMappingType
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncsByType
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.interfaces import (
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    supports_mrope,
    supports_realtime,
    supports_transcription,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
    is_text_generation_model,
)
from vllm.model_executor.offloader import (
    create_offloader,
    set_offloader,
)
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import (
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import (
    copy_mm_embedding_modality,
    get_mm_features_in_window,
    group_and_batch_mm_kwargs,
    set_mm_embedding_modality,
)
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    PIN_MEMORY,
    async_tensor_h2d,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)
from vllm.v1.outputs import (
    KVConnectorOutput,
)
from vllm.v1.pool.late_interaction_runner import LateInteractionRunner
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import (
    AttentionGroup,
    EncoderTimingStats,
    copy_kv_cache_blocks_inplace,
    sanity_check_mm_encoder_outputs,
)

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.vllm_v1.utils import (
    reorder_batch_to_split_decodes_and_prefills,
)
from vllm_torchtpu.runner.vllm_v1.worker.block_table import SlotMappingMode
from vllm_torchtpu.runner.vllm_v1.worker.ec_connector_model_runner_mixin import (
    ECConnectorModelRunnerMixin,
)
from vllm_torchtpu.runner.vllm_v1.worker.gpu_input_batch import (
    CachedRequestState,
    InputBatch,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager


logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin, ECConnectorModelRunnerMixin):
    @JitWarmupRegistry.capture
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.offload_config = vllm_config.offload_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.dtype = self.model_config.dtype

        self.check_ep_fault = False
        if parallel_config.data_parallel_size > 1 and self.model_config.is_moe:
            self.check_ep_fault = get_ep_all2all_manager().support_fault_tolerance

        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # These will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False
        self.requires_sequential_video_encoding = False
        self.max_model_len = model_config.max_model_len

        # Always set to false after the first forward pass
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping micro-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 1
        )

        # Model-related.
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        self.is_mm_prefix_lm = self.model_config.is_mm_prefix_lm
        # Deprecated: cascade attention has been removed. Kept for
        # out-of-tree model runners.
        self.cascade_attn_enabled = False

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.mrope_num_dims = model_config.mrope_num_dims
        self.supports_mm_inputs = model_config.supports_multimodal_inputs

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Async scheduling
        self.use_async_scheduling = self.scheduler_config.async_scheduling

        # Async PP broadcast of sampled token ids, waited on in _prepare_input_ids.
        self._pp_recv_work: torch.distributed.Work | None = None

        self.eplb_state: EplbState | None = None
        self._moe_model: MixtureOfExperts | None = None
        # NOTE(yongji): flag to temporarily disable EPLB during scaling up/down
        self.eep_eplb_suppressed = False
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}
        self.late_interaction_runner = LateInteractionRunner()

        # Encoder CUDA graph manager (initialized after model load if enabled)
        self.encoder_cudagraph_manager: EncoderCudaGraphManager | None = None

        self.use_aux_hidden_state_outputs = False
        self.num_spec_tokens = 0
        self.prev_num_spec_tokens = 0
        self.valid_sampled_token_count_gpu: torch.Tensor | None = None
        if self.speculative_config:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            self.prev_num_spec_tokens = self.num_spec_tokens
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and draft_config.max_model_len is not None:
                self.effective_drafter_max_model_len = draft_config.max_model_len
            else:
                self.effective_drafter_max_model_len = self.max_model_len
        self.use_async_spec_decode = (
            self.use_async_scheduling and self.num_spec_tokens > 0
        )

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        logits_processors = model_config.logits_processors
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        placeholder_block_size = (
            self.cache_config.block_size or CacheConfig.DEFAULT_BLOCK_SIZE
        )
        placeholder_max_num_blocks = cdiv(
            max(self.max_model_len, self.max_encoder_len), placeholder_block_size
        )
        self._init_block_sizes = [placeholder_block_size]
        self._init_max_num_blocks = [placeholder_max_num_blocks]
        self._init_slot_mapping_modes = [SlotMappingMode.TOKEN_TO_KV_SLOT]
        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )
        # Capture warmup providers registered by the initial placeholder InputBatch
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoder
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[placeholder_block_size],
            max_num_blocks_per_req=[placeholder_max_num_blocks],
            num_spec_tokens=self.num_spec_tokens,
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                PIN_MEMORY,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively. Thinking-budget
            # tracking is requested dynamically when a budgeted request is in the
            # batch.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
            reasoning_config=self.vllm_config.reasoning_config,
            use_replayssm=self.cache_config.use_replayssm,
        )

        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            # Blocking (sleep) event to avoid busy-polling the CUDA driver lock;
            # under TP contention that spin can balloon and make the rank a straggler.
            self.prepare_inputs_event = torch.cuda.Event(blocking=True)

        # Encoder timing registry for observability
        self.encoder_timing_registry: dict[str, EncoderTimingStats] = {}
        self._encoder_timing_lock = threading.Lock()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        # [max_num_reqs, depth] prompt token ids preceding each request's chunk
        # start, for models exposing `token_lookback_depth`. Sized after load.
        self.lookback_token_ids: CpuGpuBuffer | None = None
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=self.device
        )
        self.is_padding = torch.zeros(
            self.max_num_tokens, dtype=torch.bool, device=self.device
        )
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.optimistic_seq_lens_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, pin_memory=PIN_MEMORY
        )
        self.num_computed_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.prev_num_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        # Maps current batch position -> previous batch position (-1 for new reqs)
        self.prev_positions = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
        self.num_scheduled_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )

        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, numpy=False
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        # The count includes the bonus token, so one is the minimum, and the
        # recurrent kernels pick their state slot as `accepted - 1`. Dummy runs
        # skip `_prepare_inputs`, so the zeros from the allocation would reach
        # those kernels as index -1 and fault on the out-of-bounds read.
        self.num_accepted_tokens.np.fill(1)
        self.num_accepted_tokens.copy_to_gpu()

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: For text-only inputs, each dimension has identical position
            # IDs, making M-RoPE functionally equivalent to 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(
                (self.mrope_num_dims, self.max_num_tokens + 1), dtype=torch.int64
            )

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None

        # OPTIMIZATION: Cache the arange tensors rather than creating them
        # every step. Keep in int64 to avoid overflow with long context.
        # - arange_np: immutable [0, 1, 2, ...] used as source for batched computation
        # - query_pos: CpuGpuBuffer for the computed batched arange result
        arange_size = max(self.max_num_reqs + 1, self.max_num_tokens)
        self.arange_np = np.arange(arange_size, dtype=np.int64)
        self.query_pos = self._make_buffer(arange_size, dtype=torch.int64)
        self._arange_scratch = np.empty(arange_size, dtype=np.int64)

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        self.mm_budget = (
            MultiModalBudget(self.vllm_config, self.mm_registry)
            if self.supports_mm_inputs
            else None
        )

        self.reorder_batch_threshold: int | None = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        self._draft_probs: torch.Tensor | None = None
        self._draft_prob_req_ids: list[str] | None = None
        self._draft_token_req_ids: list[str] | None = None
        self.transfer_event = torch.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )

        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        self.valid_sampled_token_count_event: torch.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None
        # We also copy the drafted tokens to the CPU asynchronously,
        # in case we need them for structured outputs.
        self.draft_token_ids_event: torch.Event | None = None
        self.draft_token_ids_copy_stream: torch.cuda.Stream | None = None
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None
        self.draft_token_ids_cpu: torch.Tensor | None = None
        self.num_accepted_tokens_event: torch.Event | None = None
        if self.num_spec_tokens:
            self.draft_token_ids_event = torch.Event()
            self.num_accepted_tokens_event = torch.Event()
            self.draft_token_ids_copy_stream = torch.cuda.Stream()
            self.draft_token_ids_cpu = torch.empty(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            if self.use_async_scheduling:
                self.valid_sampled_token_count_event = torch.Event()
                self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
                self.valid_sampled_token_count_cpu = torch.empty(
                    self.max_num_reqs,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )

        # Model weight offloader
        # Make sure this is called before any get_offloader call
        set_offloader(create_offloader(self.offload_config))

        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.kv_connector_output: KVConnectorOutput | None = None
        self.mamba_state_idx: dict[str, int] = {}
        self._mamba_bufs: mamba_utils.MambaBuffers | None = None
        self._mamba_state_copy_funcs: MambaStateCopyFuncsByType | None = None
        self.layerwise_nvtx_hooks_registered = False

    def reset_mm_cache(self) -> None:
        """Clear the multi-modal cache that was used during profiling,
        but no longer needed during inference.
        """
        if self.mm_budget:
            self.mm_budget.reset_cache()
        self.late_interaction_runner.clear()

    def reset_encoder_cache(self) -> None:
        """Clear the GPU-side encoder cache storing vision embeddings.

        This should be called when model weights are updated to ensure
        stale embeddings computed with old weights are not reused.
        """
        self.encoder_cache.clear()
        self.late_interaction_runner.clear()

    def release_late_interaction_query_cache(self, query_keys: list[str]) -> None:
        self.late_interaction_runner.release_queries(query_keys)

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            with_numpy=numpy,
        )

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.

        """
        # Attention free models have zero kv_cache_groups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if hasattr(self, "_kv_block_zeroer"):
            self._kv_block_zeroer.zero_block_ids(block_ids)

    def _on_request_state_removed(
        self,
        req_id: str,
        req_state: CachedRequestState | None,
    ) -> None:
        """Hook for platform runners to clean request-scoped side caches."""
        del req_id, req_state

    def _process_encoder_cache_scheduler_output(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Apply scheduler-side encoder cache lifecycle updates."""
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            req_state = self.requests.pop(req_id, None)
            self._on_request_state_removed(req_id, req_state)
            self.num_prompt_logprobs.pop(req_id, None)
        self.late_interaction_runner.on_requests_finished(
            scheduler_output.finished_req_ids
        )
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if scheduler_output.new_block_ids_to_zero:
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
        if scheduler_output.kv_cache_block_copies:
            copy_kv_cache_blocks_inplace(
                self.kv_caches,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
            )

        # Free the cached encoder outputs.
        self._process_encoder_cache_scheduler_output(scheduler_output)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        deferred_spec_decode_corrections = []

        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                prompt_is_token_ids=new_req_data.prompt_is_token_ids,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state
            self.late_interaction_runner.register_request(req_id, pooling_params)

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        if self.use_async_spec_decode:
            self.prev_num_draft_tokens.np.fill(0)

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                # prev_num_draft_len is used in async scheduling mode with
                # spec decode. it indicates if need to update num_computed_tokens
                # of the request. for example:
                # first step: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0.
                # second step: num_computed_tokens = 100(prompt length),
                # spec_tokens = [a,b], prev_num_draft_len = 0.
                # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2.
                # num_computed_tokens in first step and second step doesn't contain
                # the spec tokens length, but in third step it contains the
                # spec tokens length. we only need to update num_computed_tokens
                # when prev_num_draft_len > 0.
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    # Optimistically assume all accepted; queue up a correction
                    # to be called after the model forward to preserve async
                    # scheduling. Corrected on GPU in _prepare_inputs.
                    optimistic_num_accepted = req_state.prev_num_draft_len
                    req_state.output_token_ids.extend([-1] * optimistic_num_accepted)

                    deferred_spec_decode_corrections.append(
                        (req_id, optimistic_num_accepted, req_state)
                    )

                    prev_req_index = (
                        self.input_batch.prev_req_id_to_index.get(req_id)
                        if self.input_batch.prev_req_id_to_index
                        else None
                    )
                    if prev_req_index is not None:
                        self.prev_num_draft_tokens.np[prev_req_index] = (
                            optimistic_num_accepted
                        )

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                if not req_data.new_token_ids:
                    # Async scheduled PP: Sampled tokens propagated via GPU broadcast.
                    new_token_ids: list[int] = []
                else:
                    # Non-async scheduling with PP: The scheduler sends
                    # sampled token ids back because there's no direct communication
                    # between the first-stage worker and the last-stage worker.
                    new_token_ids = req_data.new_token_ids[i]
                    # Add the sampled token(s) from the previous step (if any).
                    # This doesn't include "unverified" tokens like spec tokens.
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        # Avoid slicing list in most common case.
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure, or output_token_ids was inflated by the optimistic
                # extend above (async spec decode). Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                start_token_index = self.input_batch.num_tokens_no_spec[req_index]
                # For chunked prefill, num_computed_tokens may less
                # than num_tokens_no_spec.
                # Async scheduled PP: no new_token_ids, advance num_tokens_no_spec
                # according to num_computed_tokens.
                end_token_index = max(
                    start_token_index,
                    num_computed_tokens + len(new_token_ids),
                )
                if end_token_index > start_token_index:
                    if new_token_ids:
                        # Add new_token_ids to token_ids_cpu.
                        num_new_tokens = end_token_index - start_token_index
                        tokens_to_append = new_token_ids[-num_new_tokens:]
                        self.input_batch.token_ids_cpu[
                            req_index, start_token_index:end_token_index
                        ] = tokens_to_append
                    self.input_batch.is_token_ids[
                        req_index, start_token_index:end_token_index
                    ] = True
                    self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

        if deferred_spec_decode_corrections:

            def correct_spec_decode_token_counts():
                valid_sampled_token_count = self._get_valid_sampled_token_count()
                if not valid_sampled_token_count:
                    return
                prev_req_id_to_index = self.input_batch.prev_req_id_to_index
                if not prev_req_id_to_index:
                    return
                for (
                    req_id,
                    optimistic_num_accepted,
                    req_state,
                ) in deferred_spec_decode_corrections:
                    prev_req_index = prev_req_id_to_index.get(req_id)
                    if prev_req_index is None:
                        continue
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    correction = optimistic_num_accepted - num_accepted
                    req_state.num_computed_tokens -= correction
                    cur_req_index = self.input_batch.req_id_to_index.get(req_id)
                    if cur_req_index is None:
                        continue
                    self.input_batch.num_computed_tokens_cpu[cur_req_index] -= (
                        correction
                    )

            return correct_spec_decode_token_counts
        else:
            return None

    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData
    ) -> CachedRequestState:
        """Updates streaming session request from `scheduled_new_reqs`.

        Removes the request from InputBatch (if present), updates the cached
        state, and prepares it for re-addition to the batch.

        NOTE: prompt_token_ids includes intermediate output tokens - tokens
        previously generated but now are input context (part of the prompt).
        """
        self.input_batch.remove_request(req_id)
        req_state = self.requests[req_id]

        req_state.prompt_token_ids = new_req_data.prompt_token_ids
        req_state.mm_features = new_req_data.mm_features
        req_state.prompt_embeds = new_req_data.prompt_embeds
        req_state.sampling_params = new_req_data.sampling_params
        req_state.pooling_params = new_req_data.pooling_params
        self.late_interaction_runner.register_request(req_id, req_state.pooling_params)
        req_state.block_ids = new_req_data.block_ids
        req_state.num_computed_tokens = new_req_data.num_computed_tokens
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Clear `output_token_ids` as previous output tokens are now part of
        # `prompt_token_ids`.
        req_state.output_token_ids.clear()

        if self.uses_mrope:
            self._init_mrope_positions(req_state)

        return req_state

    def _init_mrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        assert supports_mrope(model), "M-RoPE support is not implemented."
        mrope_model = cast(SupportsMRoPE, model)

        # `prompt_embeds` is a passthrough modality (no grid_thw), models'
        # M-RoPE code assumes per-feature grid info, so filter it out. The
        # prompt_embeds positions are treated as text positions for M-RoPE.
        mrope_features = [
            f for f in req_state.mm_features if f.modality != "prompt_embeds"
        ]

        if req_state.prompt_token_ids is not None:
            input_tokens = req_state.prompt_token_ids
        elif req_state.prompt_embeds is not None:
            # For embeddings-only inputs, get_mrope_input_positions only
            # needs the sequence length when mm_features is empty (which is
            # the case here since prompt_embeds are filtered out above).
            seq_len = req_state.prompt_embeds.shape[0]
            input_tokens = list(range(seq_len))
        else:
            raise ValueError(
                "M-RoPE requires either prompt_token_ids or prompt_embeds."
            )

        req_state.mrope_positions, req_state.mrope_position_delta = (
            mrope_model.get_mrope_input_positions(
                input_tokens,
                mrope_features,
            )
        )

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[
                    :, src_start:src_end
                ]
                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                assert req.mrope_position_delta is not None
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _batch_mm_inputs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[
        list[str],
        list[tuple[str, MultiModalKwargsItem]],
        list[tuple[str, PlaceholderRange]],
    ]:
        """Batch multimodal inputs from scheduled encoder inputs.

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
                inputs.

        Returns:
            A tuple of (mm_hashes, mm_kwargs, mm_lora_refs) where:
            - mm_hashes: List of multimodal hashes for each item
            - mm_kwargs: List of multimodal kwargs for each item
            - mm_lora_refs: List of (req_id, placeholder_range) for each item

        """
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], [], []

        mm_hashes = list[str]()
        mm_kwargs = list[tuple[str, MultiModalKwargsItem]]()
        # Multimodal LoRA reference info to map each multimodal item
        # back to its request & position
        mm_lora_refs = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue

                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))
                mm_lora_refs.append((req_id, mm_feature.mm_position))

        # Stable-sort by modality so each modality is encoded in as few batches
        # as possible. Encoder outputs are cached by mm_hash, so order is free.
        order = sorted(range(len(mm_kwargs)), key=lambda i: mm_kwargs[i][0])
        return (
            [mm_hashes[i] for i in order],
            [mm_kwargs[i] for i in order],
            [mm_lora_refs[i] for i in order],
        )

    def _cache_encoder_output(
        self,
        mm_hash: str,
        output: torch.Tensor,
        ec_manager_metadata: "EncoderCacheManagerMetadata | None",
        free_encoder_mm_hashes: list[str],
    ) -> None:
        """Store an encoder output for later multimodal embedding gather."""
        del ec_manager_metadata, free_encoder_mm_hashes
        self.encoder_cache[mm_hash] = output
        self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)

    def _execute_mm_encoder(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[torch.Tensor]:
        mm_hashes, mm_kwargs, mm_lora_refs = self._batch_mm_inputs_from_scheduler(
            scheduler_output
        )

        if not mm_kwargs:
            return []

        # `prompt_embeds` is a passthrough modality, the tensor is already in
        # the model embedding space, so no encoder runs. Inject each
        # `prompt_embeds` tensor directly into the encoder cache here so that
        # `_gather_mm_embeddings` can splice it via the standard `is_mm_embed`
        # path.
        pe_indices = [
            i
            for i, (modality, _) in enumerate(mm_kwargs)
            if modality == "prompt_embeds"
        ]
        if pe_indices:
            for i in pe_indices:
                pe_tensor = mm_kwargs[i][1]["embedding"].data
                assert isinstance(pe_tensor, torch.Tensor)

                self._cache_encoder_output(
                    mm_hashes[i],
                    async_tensor_h2d(pe_tensor, device=self.device),
                    scheduler_output.ec_manager_metadata,
                    scheduler_output.free_encoder_mm_hashes,
                )
            # Filter out `prompt_embeds` items from mm_kwargs/mm_hashes/mm_lora_refs
            # since they don't require further encoder processing.
            mm_hashes = [h for i, h in enumerate(mm_hashes) if i not in pe_indices]
            mm_kwargs = [k for i, k in enumerate(mm_kwargs) if i not in pe_indices]
            mm_lora_refs = [
                r for i, r in enumerate(mm_lora_refs) if i not in pe_indices
            ]
            if not mm_kwargs:
                return []  # nothing left to encode after filtering out `prompt_embeds`

        should_time = bool(
            self.observability_config
            and self.observability_config.enable_mm_processor_stats
            and scheduler_output.scheduled_encoder_inputs
        )

        model = cast(SupportsMultiModal, self.model)

        if self.lora_config and self.lora_manager.supports_tower_connector_lora():
            # Build LoRA mappings independently for encoder inputs
            # (encoder batch structure is different from main batch)
            prompt_lora_mapping = []
            token_lora_mapping = []
            lora_requests = set()
            encoder_token_counts = []
            connector_token_counts = []

            for (req_id, pos_info), (modality, mm_item) in zip(
                mm_lora_refs,
                mm_kwargs,
            ):
                req_idx = self.input_batch.req_id_to_index[req_id]
                lora_id = int(self.input_batch.request_lora_mapping[req_idx])

                tower_tokens, connector_tokens = self.model.get_mm_lora_token_counts(  # type: ignore[attr-defined]
                    modality=modality,
                    mm_kwargs=mm_item,
                    num_mm_embeds=pos_info.get_num_embeds(),
                )
                prompt_lora_mapping.append(lora_id)
                token_lora_mapping.extend([lora_id] * tower_tokens)
                encoder_token_counts.append(tower_tokens)
                connector_token_counts.append(connector_tokens)

                if lora_id > 0:
                    lora_request = self.input_batch.lora_id_to_lora_request.get(lora_id)
                    if lora_request is not None:
                        lora_requests.add(lora_request)

            # Set tower adapter mapping
            tower_mapping = LoRAMapping(
                tuple(token_lora_mapping),
                tuple(prompt_lora_mapping),
                is_prefill=True,
                type=LoRAMappingType.TOWER,
            )
            self.lora_manager.set_active_adapters(lora_requests, tower_mapping)

            # Only set connector mapping if the model actually has a connector
            # and reports connector token counts.
            mm_mapping = (
                self.model.get_mm_mapping()  # type: ignore[attr-defined]
                if hasattr(self.model, "get_mm_mapping")
                else None
            )
            if (
                mm_mapping is not None
                and mm_mapping.connector
                and all(count is not None for count in connector_token_counts)
            ):
                connector_token_mapping = np.repeat(
                    np.array(prompt_lora_mapping, dtype=np.int32),
                    np.array(connector_token_counts, dtype=np.int32),
                )
                connector_mapping = LoRAMapping(
                    index_mapping=tuple(connector_token_mapping.tolist()),
                    prompt_mapping=tuple(prompt_lora_mapping),
                    is_prefill=True,
                    type=LoRAMappingType.CONNECTOR,
                )

                self.lora_manager.set_active_adapters(
                    lora_requests,
                    connector_mapping,
                )

        encoder_outputs: list[torch.Tensor] = []
        # Track the current index in mm_kwargs/mm_lora_refs to map groups to request IDs
        current_item_idx = 0
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs, device=self.device, pin_memory=PIN_MEMORY
        ):
            batch_outputs: MultiModalEmbeddings

            # EVS and dynamic res video related change.
            # (ekhvedchenia): Temporary hack to limit peak memory usage when
            # processing multimodal data. This solves the issue with scheduler
            # putting too many video samples into a single batch. Scheduler
            # uses pruned vision tokens count to compare it versus compute
            # budget which is incorrect (Either input media size or non-pruned
            # output vision tokens count should be considered)
            # dynamic res video for nemotron temporarily uses this hack via
            # requires_sequential_video_encoding
            # because it doesn't yet support video batching.
            # TODO(ywang96): Fix memory profiling to take EVS into account and
            # remove this hack.
            if (
                (
                    self.is_multimodal_pruning_enabled
                    or self.requires_sequential_video_encoding
                )
                and modality == "video"
                and num_items > 1
            ):
                batch_outputs_lst = list[torch.Tensor]()
                for video_idx in range(num_items):
                    video_mm_kwargs_item = mm_kwargs[current_item_idx + video_idx]
                    with self.timed_encoder_operation(
                        should_time, mm_lora_refs, current_item_idx + video_idx, 1
                    ):
                        _, _, micro_batch_mm_inputs = next(
                            group_and_batch_mm_kwargs(
                                [video_mm_kwargs_item],
                                device=self.device,
                                pin_memory=PIN_MEMORY,
                            )
                        )

                        micro_batch_outputs = model.embed_multimodal(
                            **micro_batch_mm_inputs
                        )

                        batch_outputs_lst.extend(micro_batch_outputs)

                batch_outputs = batch_outputs_lst
            else:
                # Run the encoder.
                # `batch_outputs` is either of the following:
                # 1. A tensor of shape (num_items, feature_size, hidden_size)
                # in case feature_size is fixed across all multimodal items.
                # 2. A list or tuple (length: num_items) of tensors,
                # each of shape (feature_size, hidden_size) in case the feature
                # size is dynamic depending on the input multimodal items.

                with self.timed_encoder_operation(
                    should_time, mm_lora_refs, current_item_idx, num_items
                ):
                    cudagraph_output = None
                    if (
                        self.encoder_cudagraph_manager is not None
                        and self.encoder_cudagraph_manager.supports_modality(modality)
                    ):
                        cudagraph_output = self.encoder_cudagraph_manager.execute(
                            mm_kwargs_batch,
                        )

                    if cudagraph_output is not None:
                        batch_outputs = cudagraph_output
                    else:
                        batch_outputs = model.embed_multimodal(**mm_kwargs_batch)

            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
            encoder_outputs.extend(batch_outputs)

            current_item_idx += num_items

        # Cache the encoder outputs by mm_hash
        for mm_hash, output in zip(mm_hashes, encoder_outputs):
            self._cache_encoder_output(
                mm_hash,
                output,
                scheduler_output.ec_manager_metadata,
                scheduler_output.free_encoder_mm_hashes,
            )
            logger.debug("Finish execute for mm hash %s", mm_hash)

        return encoder_outputs

    def _get_encoder_output_from_cache(self, mm_hash: str) -> torch.Tensor | None:
        """Return a cached encoder output for multimodal
        embedding gather."""
        return self.encoder_cache.get(mm_hash, None)

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        mm_embeds = list[torch.Tensor]()
        is_mm_embed = torch.zeros(
            total_num_scheduled_tokens,
            dtype=torch.bool,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )

        req_start_idx = 0
        should_sync_mrope_positions = False

        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens

            mm_features = req_state.mm_features
            lo, hi = get_mm_features_in_window(
                mm_features,
                start=num_computed_tokens,
                end=num_computed_tokens + num_scheduled_tokens,
            )
            for i in range(lo, hi):
                mm_feature = mm_features[i]
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self._get_encoder_output_from_cache(mm_hash)
                if encoder_output is None:
                    # A feature starting at/after the processed boundary is only
                    # reached via the drafter's +1 look-ahead and might not be
                    # encoded yet; fall back to the token embedding for drafting.
                    if (
                        start_pos
                        >= req_state.num_computed_tokens + num_scheduled_tokens
                    ):
                        continue
                    raise RuntimeError(f"Encoder cache miss for {mm_hash}.")

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                # OR mask for overlapping mm_features (use_audio_in_video)
                if is_embed is None:
                    is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                        True
                    )
                else:
                    is_mm_embed[
                        req_start_pos + start_idx : req_start_pos + end_idx
                    ] |= is_embed
                set_mm_embedding_modality(mm_embeds_item, mm_feature.modality)
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                old_mm_embeds_req = mm_embeds_req
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                mm_embeds_req = [
                    copy_mm_embedding_modality(src, dst)
                    for src, dst in zip(old_mm_embeds_req, mm_embeds_req)
                ]
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def get_model(self) -> nn.Module:
        if not hasattr(self, "model"):
            raise ValueError("Cannot get model before model has been initialized")
        if isinstance(self.model, (CUDAGraphWrapper, BreakableCUDAGraphWrapper)):
            # get raw model out of the cudagraph wrapper.
            return self.model.unwrap()
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        if supports_realtime(model):
            supported_tasks.append("realtime")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        return list(model.pooler.get_supported_tasks())

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        sampled_count_event = self.valid_sampled_token_count_event
        if sampled_count_event is None or prev_sampled_token_ids is None:
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        assert counts_cpu is not None
        sampled_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig) -> None:
        """Re-initialize the input batch if the block sizes are different from
        what it was originally created with. This happens when the final
        block size (determined after model loading) differs from the
        placeholder used during __init__, or when there are multiple
        KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.

        """
        block_sizes = []
        max_num_blocks = []
        slot_mapping_modes = []
        max_model_len = max(self.max_model_len, self.max_encoder_len)
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            kv_cache_spec_kind = get_kv_cache_spec_kind(kv_cache_spec)
            if kv_cache_spec_kind == KVCacheSpecKind.ENCODER_ONLY_ATTENTION:
                continue
            block_size = kv_cache_spec.block_size
            block_sizes.append(block_size)
            if kv_cache_spec_kind == KVCacheSpecKind.MAMBA:
                slot_mapping_modes.append(SlotMappingMode.NONE)
            else:
                slot_mapping_modes.append(SlotMappingMode.TOKEN_TO_KV_SLOT)
            max_num_blocks_per_req = kv_cache_spec.max_num_blocks_per_req(
                self.vllm_config, max_model_len
            )
            max_num_blocks.append(max_num_blocks_per_req)

        if (
            block_sizes != self._init_block_sizes
            or max_num_blocks != self._init_max_num_blocks
            or slot_mapping_modes != self._init_slot_mapping_modes
            or self.cp_kv_cache_interleave_size
            != self.parallel_config.cp_kv_cache_interleave_size
        ):
            self._init_block_sizes = block_sizes
            self._init_max_num_blocks = max_num_blocks
            self._init_slot_mapping_modes = slot_mapping_modes
            self.cp_kv_cache_interleave_size = (
                self.parallel_config.cp_kv_cache_interleave_size
            )
            # Capture warmup providers registered after final KV-cache geometry is known
            with self.jit_warmup_registry.activate():  # type: ignore[attr-defined]
                self.input_batch = InputBatch(
                    max_num_reqs=self.max_num_reqs,
                    max_model_len=max_model_len,
                    max_num_batched_tokens=self.max_num_tokens,
                    device=self.device,
                    vocab_size=self.model_config.get_vocab_size(),
                    block_sizes=block_sizes,
                    max_num_blocks_per_req=max_num_blocks,
                    num_spec_tokens=self.num_spec_tokens,
                    logitsprocs=self.input_batch.logitsprocs,
                    logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                    is_pooling_model=self.is_pooling_model,
                    cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
                    reasoning_config=self.vllm_config.reasoning_config,
                    use_replayssm=self.cache_config.use_replayssm,
                    slot_mapping_modes=slot_mapping_modes,
                )

        assert self._init_block_sizes == block_sizes, (
            f"InputBatch block_sizes {self._init_block_sizes} != "
            f"kv_cache block_sizes {block_sizes}"
        )

    @contextmanager
    def timed_encoder_operation(
        self,
        should_time: bool,
        group_lora_refs: list[tuple[str, Any]],
        current_item_idx: int,
        num_items: int,
    ):
        """Context manager to time encoder forward operations.

        Args:
            should_time: Whether timing is enabled
            group_lora_refs: Full list of (request_id, pos_info) tuples
            current_item_idx: Starting index for this group
            num_items: Number of items in this group

        """
        if not should_time:
            yield
            return

        group_refs = group_lora_refs[current_item_idx : current_item_idx + num_items]
        group_request_ids = {req_id for req_id, _ in group_refs}

        torch.accelerator.synchronize()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            torch.accelerator.synchronize()
            elapsed = time.perf_counter() - start_time

            per_request_time = elapsed / max(len(group_request_ids), 1)

            with self._encoder_timing_lock:
                for req_id in group_request_ids:
                    if req_id not in self.encoder_timing_registry:
                        self.encoder_timing_registry[req_id] = EncoderTimingStats()

                    stats = self.encoder_timing_registry[req_id]
                    stats.encoder_forward_secs += per_request_time
                    stats.num_encoder_calls += 1
