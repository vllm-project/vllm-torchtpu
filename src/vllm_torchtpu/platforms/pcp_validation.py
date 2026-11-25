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

from dataclasses import dataclass
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.gdn_pool_layout import (
    QWEN_GDN_ARCHITECTURES,
    unified_kv_layout_enabled_for_architecture,
)
from vllm_torchtpu.kernels.experimental.gdn_pcp_fused import (
    fused_qkvz_pcp_dma_row_alignment,
)
from vllm_torchtpu.pcp_sharding_plan import PcpShardingPlanName


def pcp_streaming_local_token_multiple() -> int:
    """Rank-local compile token multiple required by streamed PCP attention."""
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.kernel import (
        PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    )

    return int(PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE)


def pcp_rank0_token_count(
    total_tokens: int, pcp_size: int, interleave_size: int
) -> int:
    """Return rank 0's token count under round-robin PCP chunk interleaving.

    O(1) closed form matching ``pcp_local_token_counts(...)[0]`` so startup
    budget checks do not import ``pcp_streaming_rpa``/JAX when streamed PCP is
    inactive and avoid ``pcp_local_token_counts``'s chunk loop.
    """
    tokens = max(0, int(total_tokens))
    if pcp_size <= 1:
        return tokens
    interleave = max(1, int(interleave_size))
    cycle = int(pcp_size) * interleave
    return (tokens // cycle) * interleave + min(tokens % cycle, interleave)


def effective_step_token_budget(scheduler_config: Any) -> tuple[str, int]:
    """Return ``(flag_name, tokens)`` for the tighter scheduler step budget."""
    budget = int(scheduler_config.max_num_batched_tokens)
    scheduled = scheduler_config.max_num_scheduled_tokens
    if scheduled is not None and int(scheduled) < budget:
        return "--max-num-scheduled-tokens", int(scheduled)
    return "--max-num-batched-tokens", budget


def interleaved_max_local_tokens(
    step_token_budget: int,
    pcp_size: int,
    interleave_size: int,
) -> int:
    """Upper bound on a rank's local tokens, rounded up to the streaming multiple."""
    rank0_max = pcp_rank0_token_count(
        max(1, int(step_token_budget)), pcp_size, interleave_size
    )
    multiple = pcp_streaming_local_token_multiple()
    return ((max(1, rank0_max) + multiple - 1) // multiple) * multiple


def is_interleaved_pcp_manager_enabled(vllm_config: Any) -> bool:
    """Return whether ``vllm_config`` selects interleaved ``PcpManager``."""
    return (
        envs.USE_TPU_PCP_MANAGER
        and int(vllm_config.parallel_config.prefill_context_parallel_size) > 1
        and vllm_config.model_config is not None
        and not vllm_config.model_config.use_mla
        and select_pcp_sharding_plan(vllm_config.model_config)
        is PcpShardingPlanName.INTERLEAVED
    )


def select_pcp_sharding_plan(model_config: Any) -> PcpShardingPlanName:
    if model_config.use_mla:
        if not hasattr(model_config.hf_text_config, "index_topk"):
            raise NotImplementedError(
                "The PCP manager does not support dense MLA "
                "(only sparse MLA or non-MLA attention)."
            )
        return PcpShardingPlanName.HEADTAIL
    return PcpShardingPlanName.INTERLEAVED


def _validate_hnd_interleave(kv_cache_layout: str, interleave_size: int) -> None:
    if kv_cache_layout == "HND" and interleave_size % 128 != 0:
        raise ValueError(
            "PCP with HND KV cache layout requires "
            "cp_kv_cache_interleave_size to be a multiple of 128, "
            f"got {interleave_size}."
        )


def _validate_moe_expert_parallel(
    is_moe: bool, expert_parallel: bool, pcp_size: int
) -> None:
    if is_moe and not expert_parallel:
        raise NotImplementedError(
            "PCP on an MoE model requires --enable-expert-parallel: "
            f"--prefill-context-parallel-size={pcp_size} without "
            "it would shard MoE experts across the PCP ranks, which the "
            "TPU MoE kernels do not implement. Add the flag, or drop "
            "--prefill-context-parallel-size."
        )


def validate_interleaved_pcp_params(
    pcp_size: int, dcp_size: int, interleave_size: int
) -> None:
    if pcp_size <= 1:
        raise NotImplementedError(
            "Interleaved PCP requires prefill_context_parallel_size > 1, "
            f"got pcp={pcp_size}."
        )
    if dcp_size != 1:
        raise NotImplementedError(
            "Interleaved PCP does not support DCP "
            f"(decode_context_parallel_size == 1 required), got dcp={dcp_size}."
        )
    multiple = pcp_streaming_local_token_multiple()
    if interleave_size <= 0 or multiple % interleave_size != 0:
        raise ValueError(
            "Interleaved PCP requires cp_kv_cache_interleave_size > 0 and it "
            f"must divide {multiple}, got {interleave_size}."
        )


def validate_pooled_qwen_gdn_pcp_interleave(
    architecture: object, interleave_size: int
) -> None:
    """Reject PCP interleave sizes unaligned to the pooled Qwen GDN DMA tile."""
    if not isinstance(architecture, str):
        return
    if (
        architecture in QWEN_GDN_ARCHITECTURES
        and unified_kv_layout_enabled_for_architecture(architecture)
    ):
        row_alignment = fused_qkvz_pcp_dma_row_alignment()
        if interleave_size < row_alignment or interleave_size % row_alignment != 0:
            raise ValueError(
                "PCP on pooled Qwen GDN models requires "
                "--cp-kv-cache-interleave-size to be a positive multiple of "
                f"{row_alignment} (for example, --cp-kv-cache-interleave-size "
                f"128), got {interleave_size}."
            )


def validate_pcp_manager_config(vllm_config: Any) -> PcpShardingPlanName:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    plan = select_pcp_sharding_plan(model_config)
    pcp_size = int(parallel_config.prefill_context_parallel_size)
    dcp_size = int(parallel_config.decode_context_parallel_size)

    if plan is PcpShardingPlanName.HEADTAIL:
        from vllm_torchtpu.kernels.mla.kv_cache_utils import WORD_BYTES

        tp_size = int(parallel_config.tensor_parallel_size)
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
        interleave_size = int(parallel_config.cp_kv_cache_interleave_size)
        if interleave_size % WORD_BYTES:
            raise ValueError(
                "PCP x DCP needs --cp-kv-cache-interleave-size to be a multiple of "
                f"{WORD_BYTES}, got {interleave_size}."
            )
    else:
        interleave_size = int(parallel_config.cp_kv_cache_interleave_size)
        if pcp_size > 1:
            validate_pooled_qwen_gdn_pcp_interleave(
                model_config.architecture, interleave_size
            )
        validate_interleaved_pcp_params(pcp_size, dcp_size, interleave_size)

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

    if model_config.runner_type == "pooling":
        raise NotImplementedError("PCP manager does not support pooling models.")
    if vllm_config.kv_transfer_config is not None:
        raise NotImplementedError("PCP manager does not support KV connectors.")
    _validate_moe_expert_parallel(
        parallel_config.is_moe_model is True,
        bool(parallel_config.enable_expert_parallel),
        pcp_size,
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


@dataclass(frozen=True)
class PcpStaticConfig:
    pcp_size: int
    interleave_size: int
    dcp_size: int
    pipeline_parallel_size: int
    async_scheduling: bool
    speculative_enabled: bool
    speculative_method: str | None
    num_speculative_tokens: int
    is_moe: bool
    expert_parallel: bool

    @property
    def enabled(self) -> bool:
        return self.pcp_size > 1


class PcpStaticSupportValidator:
    @staticmethod
    def from_vllm_config(vllm_config: Any) -> PcpStaticConfig:
        if vllm_config is None:
            return PcpStaticConfig(
                pcp_size=1,
                interleave_size=1,
                dcp_size=1,
                pipeline_parallel_size=1,
                async_scheduling=False,
                speculative_enabled=False,
                speculative_method=None,
                num_speculative_tokens=0,
                is_moe=False,
                expert_parallel=False,
            )

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        speculative_config = vllm_config.speculative_config

        speculative_method = None
        num_speculative_tokens = 0
        if speculative_config is not None:
            speculative_method = speculative_config.method
            num_speculative_tokens = speculative_config.num_speculative_tokens

        return PcpStaticConfig(
            pcp_size=parallel_config.prefill_context_parallel_size,
            interleave_size=parallel_config.cp_kv_cache_interleave_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            async_scheduling=bool(scheduler_config.async_scheduling),
            speculative_enabled=speculative_config is not None,
            speculative_method=speculative_method,
            num_speculative_tokens=num_speculative_tokens,
            is_moe=parallel_config.is_moe_model is True,
            expert_parallel=bool(parallel_config.enable_expert_parallel),
        )

    @staticmethod
    def validate_platform_config(
        vllm_config: Any,
        *,
        kv_cache_layout: str,
    ) -> PcpStaticConfig:
        config = PcpStaticSupportValidator.from_vllm_config(vllm_config)
        if envs.USE_TPU_PCP_MANAGER:
            if vllm_config is None or vllm_config.model_config is None:
                return config
            plan = validate_pcp_manager_config(vllm_config)
            if plan is PcpShardingPlanName.INTERLEAVED:
                _validate_hnd_interleave(kv_cache_layout, config.interleave_size)
            return config
        if not config.enabled:
            return config
        _validate_hnd_interleave(kv_cache_layout, config.interleave_size)
        if vllm_config.model_config is not None:
            validate_pooled_qwen_gdn_pcp_interleave(
                vllm_config.model_config.architecture,
                config.interleave_size,
            )

        validate_interleaved_pcp_params(
            config.pcp_size, config.dcp_size, config.interleave_size
        )
        _validate_moe_expert_parallel(
            config.is_moe, config.expert_parallel, config.pcp_size
        )
        if config.speculative_enabled:
            if config.speculative_method != "mtp":
                raise NotImplementedError(
                    "PCP speculative decoding currently requires method=mtp."
                )
            if config.num_speculative_tokens != 1:
                raise NotImplementedError(
                    "PCP MTP currently requires num_speculative_tokens=1."
                )
        return config
