# Copyright 2026 Google LLC
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
"""PCP manager

`PcpManager` delegates to a stateful `PcpShardingPlan` (`sharding_plan.py`),
selected via `TPU_PCP_SHARDING_PLAN` (`headtail` or `interleaved`).
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.pcp import get_pcp_group
from vllm_torchtpu.gdn_pool_layout import (
    QWEN_GDN_ARCHITECTURES,
    unified_kv_layout_enabled_for_architecture,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import WORD_BYTES
from vllm_torchtpu.layers.core.pcp_sequence_layout import (
    ensure_host_token_buffer_capacity,
)
from vllm_torchtpu.layers.core.sequence_layout import SequenceLayoutDescriptor
from vllm_torchtpu.runner.pcp.head_tail_sharding_plan import HeadTailPcpShardingPlan
from vllm_torchtpu.runner.pcp.interleaved_sharding_plan import (
    InterleavedPcpShardingPlan,
    validate_interleaved_pcp_params,
)
from vllm_torchtpu.runner.pcp.sharding_plan import PcpShardingPlan

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.pcp_metadata import PcpMetadata


def mla_kv_page_size(kv_cache_config: Any) -> int:
    """The single block size of the MLA KV cache groups (the sparse-MLA
    cache's page size), looking inside `UniformTypeKVCacheSpecs`."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs

    sizes: set[int] = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        specs = (
            spec.kv_cache_specs.values()
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )
        sizes.update(
            int(s.block_size) for s in specs if isinstance(s, MLAAttentionSpec)
        )
    if len(sizes) != 1:
        raise ValueError(
            f"PCP manager expects one MLA KV cache block size, got {sorted(sizes)}"
        )
    return sizes.pop()


class PcpShardingPlanName(StrEnum):
    HEADTAIL = "headtail"
    INTERLEAVED = "interleaved"


def validate_pcp_manager_config(vllm_config: Any) -> PcpShardingPlanName:
    try:
        plan = PcpShardingPlanName(envs.TPU_PCP_SHARDING_PLAN.lower())
    except ValueError as exc:
        raise ValueError(
            f"Unsupported TPU_PCP_SHARDING_PLAN: {envs.TPU_PCP_SHARDING_PLAN!r}."
        ) from exc

    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    pcp_size = int(parallel_config.prefill_context_parallel_size)
    dcp_size = int(parallel_config.decode_context_parallel_size)

    if plan is PcpShardingPlanName.HEADTAIL:
        tp_size = int(parallel_config.tensor_parallel_size)
        sparse_mla = bool(model_config.use_mla) and hasattr(
            model_config.hf_text_config, "index_topk"
        )
        if not sparse_mla:
            raise NotImplementedError("The PCP manager only supports sparse MLA.")
        if pcp_size <= 1 or dcp_size <= 1:
            raise NotImplementedError(
                "USE_TPU_PCP_MANAGER needs prefill and decode context parallel "
                f"sizes > 1, got pcp={pcp_size}, dcp={dcp_size}."
            )
        if dcp_size != tp_size * pcp_size:
            raise NotImplementedError(
                "PCP x DCP requires decode_context_parallel_size == "
                "tensor_parallel_size * prefill_context_parallel_size, got "
                f"dcp={dcp_size}, tp={tp_size}, pcp={pcp_size}."
            )
        max_model_len = int(model_config.max_model_len)
        if max_model_len > envs.TPU_PCP_KV_WORKSPACE_TOKENS:
            raise ValueError(
                f"max_model_len ({max_model_len}) exceeds "
                f"TPU_PCP_KV_WORKSPACE_TOKENS ({envs.TPU_PCP_KV_WORKSPACE_TOKENS}); "
                "raise TPU_PCP_KV_WORKSPACE_TOKENS or lower --max-model-len."
            )
        if model_config.runner_type == "pooling":
            raise NotImplementedError("PCP x DCP does not support pooling models.")
        if vllm_config.kv_transfer_config is not None:
            raise NotImplementedError("PCP x DCP does not support KV connectors.")
        interleave_size = int(parallel_config.cp_kv_cache_interleave_size)
        if interleave_size % WORD_BYTES:
            raise ValueError(
                "PCP x DCP needs --cp-kv-cache-interleave-size to be a multiple of "
                f"{WORD_BYTES}, got {interleave_size}."
            )
    else:
        interleave_size = int(parallel_config.cp_kv_cache_interleave_size)
        validate_interleaved_pcp_params(pcp_size, dcp_size, interleave_size)
        if (
            model_config.architecture in QWEN_GDN_ARCHITECTURES
            and unified_kv_layout_enabled_for_architecture(model_config.architecture)
            and interleave_size < 16
        ):
            # TODO(#2698): Remove once pooled GDN supports interleave_size < 16.
            raise NotImplementedError(
                "Interleaved PCP on pooled Qwen GDN models requires "
                f"cp_kv_cache_interleave_size >= 16, got {interleave_size}."
            )

    if model_config.is_multimodal_model and not (
        plan is PcpShardingPlanName.INTERLEAVED
        and model_config.multimodal_config is not None
        and model_config.multimodal_config.language_model_only
    ):
        raise NotImplementedError(
            "PCP manager does not support multimodal inputs"
            + (
                "; pass --language-model-only for text-only execution."
                if plan is PcpShardingPlanName.INTERLEAVED
                else "."
            )
        )

    if parallel_config.is_moe_model and not parallel_config.enable_expert_parallel:
        raise NotImplementedError(
            "PCP on an MoE model requires --enable-expert-parallel."
        )
    if parallel_config.pipeline_parallel_size != 1:
        raise NotImplementedError("PCP manager does not support pipeline parallelism.")
    if parallel_config.data_parallel_size != 1:
        raise NotImplementedError("PCP manager does not support data parallelism.")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "PCP manager does not support speculative decoding yet."
        )
    if vllm_config.lora_config is not None:
        raise NotImplementedError("PCP manager does not support LoRA.")
    return plan


def _layer_supports_pcp_manager(layer: Any, plan: PcpShardingPlanName) -> bool:
    if plan is PcpShardingPlanName.HEADTAIL:
        return bool(getattr(layer, "supports_tpu_pcp_manager", False))
    from vllm.model_executor.layers.attention import Attention

    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import (
        VllmGatedDeltaNetAttention,
    )

    return isinstance(layer, VllmGatedDeltaNetAttention) or (
        isinstance(layer, Attention)
        and isinstance(layer.impl, PallasAttentionBackendImpl)
    )


def validate_pcp_manager_layers(
    layers: dict[str, Any],
    plan: PcpShardingPlanName,
) -> None:
    """Rejects attention layers unsupported by the active PCP sharding plan."""
    unsupported = sorted(
        {
            type(layer).__name__
            for layer in layers.values()
            if not _layer_supports_pcp_manager(layer, plan)
        }
    )
    if unsupported:
        raise NotImplementedError(
            "USE_TPU_PCP_MANAGER=1 does not support these attention layers: "
            f"{', '.join(unsupported)}."
        )


class PcpManager:
    """PCP manager owned by `TPUModelRunner`. The runner reads the prepared
    step's state from `sharding_plan`."""

    uses_selected_logits_hidden_states: bool = True

    def __init__(self, sharding_plan: PcpShardingPlan) -> None:
        self.sharding_plan = sharding_plan

    @property
    def plan_name(self) -> PcpShardingPlanName:
        return (
            PcpShardingPlanName.INTERLEAVED
            if isinstance(self.sharding_plan, InterleavedPcpShardingPlan)
            else PcpShardingPlanName.HEADTAIL
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> PcpManager:
        plan = validate_pcp_manager_config(vllm_config)
        parallel_config = vllm_config.parallel_config
        group = get_pcp_group()
        pcp_rank = 0 if group is None else int(group.rank_in_group)
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if plan is PcpShardingPlanName.HEADTAIL:
            return cls(
                HeadTailPcpShardingPlan(
                    pcp_size=parallel_config.prefill_context_parallel_size,
                    pcp_rank=pcp_rank,
                    dcp_size=parallel_config.decode_context_parallel_size,
                    page_size=vllm_config.cache_config.block_size,
                    workspace_tokens=envs.TPU_PCP_KV_WORKSPACE_TOKENS,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                )
            )
        return cls(
            InterleavedPcpShardingPlan(
                pcp_size=parallel_config.prefill_context_parallel_size,
                pcp_rank=pcp_rank,
                interleave_size=parallel_config.cp_kv_cache_interleave_size,
                max_num_batched_tokens=max_num_batched_tokens,
            )
        )

    @property
    def local_reqs_per_req(self) -> int:
        return self.sharding_plan.local_reqs_per_req

    @property
    def max_local_num_tokens(self) -> int:
        return self.sharding_plan.max_local_num_tokens

    @property
    def sequence_layout_descriptor(self) -> SequenceLayoutDescriptor:
        return self.sharding_plan.sequence_layout_descriptor

    @property
    def requires_backend_preinit(self) -> bool:
        return self.sharding_plan.requires_backend_preinit

    @property
    def backend_preinit_world_size(self) -> int:
        return self.sharding_plan.pcp_size

    @property
    def local_compile_token_multiple(self) -> int:
        return self.sharding_plan.local_compile_token_multiple

    def num_reqs_in_chunk(self, q_lens: Any, seq_lens: Any, capacity: int) -> int:
        return self.sharding_plan.num_reqs_in_chunk(q_lens, seq_lens, capacity)

    def set_page_size(self, page_size: int) -> None:
        self.sharding_plan.set_page_size(page_size)

    def configure_kv_cache(
        self, kv_cache_config: Any, kernel_block_size: int | None = None
    ) -> None:
        self.sharding_plan.configure_kv_cache(kv_cache_config, kernel_block_size)

    def reserve_host_token_capacity(
        self, runner: Any, required_num_tokens: int
    ) -> None:
        ensure_host_token_buffer_capacity(runner, required_num_tokens)

    def prepare_real(self, **kwargs: Any) -> None:
        self.sharding_plan.prepare_real(**kwargs)

    def prepare_dummy(
        self, *, num_tokens: int, num_reqs: int, max_num_tokens: int
    ) -> None:
        self.sharding_plan.prepare_dummy(
            num_tokens=num_tokens, num_reqs=num_reqs, max_num_tokens=max_num_tokens
        )

    def build_metadata(self, **kwargs: Any) -> PcpMetadata | None:
        return self.sharding_plan.build_metadata(**kwargs)

    def maybe_select_logits_hidden_states(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        return self.sharding_plan.maybe_select_logits_hidden_states(hidden_states)

    def finalize_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.sharding_plan.finalize_hidden_states(hidden_states)
