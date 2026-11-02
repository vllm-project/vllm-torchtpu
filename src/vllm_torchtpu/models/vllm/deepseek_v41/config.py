# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Typed DSV41 serving policy at vLLM's out-of-tree configuration boundary."""

import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class ExecutionMode(StrEnum):
    FULL = "full"
    CED = "ced"


class MoEBackend(StrEnum):
    GMM = "gmm"
    FUSED_EP_W4A8 = "fused_ep_w4a8"


@dataclass(frozen=True)
class DeepseekV41Config:
    execution: ExecutionMode = ExecutionMode.FULL
    moe: MoEBackend = MoEBackend.GMM

    @classmethod
    def parse(cls, value: dict) -> "DeepseekV41Config":
        unknown = value.keys() - {"execution", "moe"}
        if unknown:
            raise ValueError(f"Unknown deepseek_v41 config fields: {sorted(unknown)}")
        return cls(
            execution=ExecutionMode(value.get("execution", "full")),
            moe=MoEBackend(value.get("moe", "gmm")),
        )

    @classmethod
    def from_vllm_config(cls, config: "VllmConfig") -> "DeepseekV41Config":
        additional = config.additional_config
        if not isinstance(additional, dict):
            raise TypeError("DSV41 requires a dictionary additional_config")
        value = additional.get("deepseek_v41")
        if value is None:
            # Migration only: recipes use the typed policy. Remove these aliases
            # after callers have migrated; consumers never resolve envs themselves.
            return cls.from_legacy_env()
        if not isinstance(value, dict):
            raise TypeError("additional_config.deepseek_v41 must be an object")
        return cls.parse(value)

    @classmethod
    def from_legacy_env(cls) -> "DeepseekV41Config":
        return cls(
            execution=ExecutionMode.CED
            if envs.TPU_DEEPSEEK_V41_CED
            else ExecutionMode.FULL,
            moe=MoEBackend.FUSED_EP_W4A8
            if envs.USE_MOE_FUSED_EP_KERNEL and envs.MOE_FUSED_EP_ENABLE_W4A8
            else MoEBackend.GMM,
        )

    def publish(self, config: "VllmConfig") -> None:
        # Canonical defaults participate in vLLM's config hash and worker copies.
        assert isinstance(config.additional_config, dict)
        config.additional_config["deepseek_v41"] = asdict(self)
        logger.info("[DSV41] resolved serving policy: %s", asdict(self))


def validate_environment(policy: DeepseekV41Config) -> None:
    aliases = {
        "TPU_DEEPSEEK_V41_CED": (
            envs.TPU_DEEPSEEK_V41_CED,
            policy.execution == ExecutionMode.CED,
        ),
        "USE_MOE_FUSED_EP_KERNEL": (
            envs.USE_MOE_FUSED_EP_KERNEL,
            policy.moe == MoEBackend.FUSED_EP_W4A8,
        ),
        "MOE_FUSED_EP_ENABLE_W4A8": (
            envs.MOE_FUSED_EP_ENABLE_W4A8,
            policy.moe == MoEBackend.FUSED_EP_W4A8,
        ),
    }
    for name, (legacy, selected) in aliases.items():
        if name in os.environ:
            if legacy != selected:
                raise ValueError(
                    f"{name} conflicts with deepseek_v41 serving policy; "
                    "remove the environment variable"
                )
            logger.warning_once(
                "[DSV41] %s is deprecated; use --additional-config "
                "deepseek_v41 instead",
                name,
            )
    if policy.moe == MoEBackend.FUSED_EP_W4A8:
        if envs.MOE_FUSED_EP_KERNEL_IMPL not in (None, "v2", "push"):
            raise ValueError(
                "DSV41 fused_ep_w4a8 uses v2 or push; remove MOE_FUSED_EP_KERNEL_IMPL"
            )
        if envs.MOE_REQUANTIZE_BLOCK_SIZE not in (None, 0, 512):
            raise ValueError(
                "DSV41 fused_ep_w4a8 uses block 512; remove MOE_REQUANTIZE_BLOCK_SIZE"
            )
        if envs.MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE:
            raise ValueError(
                "DSV41 uses kernel height tuning; remove "
                "MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE"
            )
        if (
            "MOE_FUSED_EP_V2_SHARDED_PLAN" in os.environ
            and not envs.MOE_FUSED_EP_V2_SHARDED_PLAN
        ):
            raise ValueError(
                "DSV41 fused_ep_w4a8 uses the sharded plan; remove "
                "MOE_FUSED_EP_V2_SHARDED_PLAN"
            )
    for name in os.environ:
        if (
            name.startswith("TPU_DEEPSEEK_V41_")
            and name not in aliases
            and name != "TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD"
        ):
            logger.warning_once(
                "[DSV41] %s no longer selects a serving route; "
                "remove it from the recipe",
                name,
            )


def verify_and_update_config(config: "VllmConfig") -> None:
    policy = DeepseekV41Config.from_vllm_config(config)
    if config.engram_config is None:
        from vllm.config import EngramConfig

        config.engram_config = EngramConfig(
            cpu_offload=envs.TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD
        )
    if "TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD" in os.environ:
        if config.engram_config.cpu_offload != envs.TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD:
            raise ValueError(
                "TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD conflicts with "
                "--engram-config; remove the environment variable"
            )
        logger.warning_once(
            "[DSV41] TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD is "
            "deprecated; use --engram-config instead"
        )
    parallel = config.parallel_config
    if envs.TPU_DEEPSEEK_V41_IMPL != "v2":
        raise ValueError("DSV41 serving uses v2; remove TPU_DEEPSEEK_V41_IMPL")
    if config.kernel_config.moe_backend != "auto":
        raise ValueError(
            "TPU DSV41 uses additional_config.deepseek_v41.moe; "
            "leave --moe-backend=auto"
        )
    if policy.moe == MoEBackend.FUSED_EP_W4A8:
        if not parallel.enable_expert_parallel:
            raise ValueError("fused_ep_w4a8 requires --enable-expert-parallel")
        assert config.model_config is not None
        if config.model_config.hf_config.expert_dtype != "fp4":
            raise ValueError("fused_ep_w4a8 requires FP4 expert weights")
    if policy.execution == ExecutionMode.CED:
        if parallel.prefill_context_parallel_size != 1:
            raise ValueError("CED requires prefill-context-parallel-size=1")
        if (
            config.cache_config.enable_prefix_caching
            and not config.cache_config.swa_bounded_replay
        ):
            raise ValueError("CED with prefix caching requires --swa-bounded-replay")
    validate_environment(policy)
    policy.publish(config)


def engram_cpu_offload(config: "VllmConfig") -> bool:
    # Compatibility for standalone callers without a resolved EngramConfig.
    if config.engram_config is None:
        return envs.TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD
    return config.engram_config.cpu_offload
