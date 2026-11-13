# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolve and read the DeepSeek-V4.1 model plan.

``resolve_plan`` runs once at configuration (``config.verify_and_update_config``)
and publishes the result into ``additional_config["model_plan"]``; workers
activate it from their config copy. Model code reads its slot parameters with
:func:`param`, which falls back to the SUPPORTED plan when no plan is active
(module-level tests that build a layer without a config).
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from functools import cache
from typing import TYPE_CHECKING, Any

from vllm_torchtpu import model_plan
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.model_plan import PlanContext, PlanError, ResolvedPlan
from vllm_torchtpu.models.deepseek_v41.plans import SUPPORTED_PLAN
from vllm_torchtpu.models.deepseek_v41.variants import SUPPORTED_VARIANTS

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

MODEL = "deepseek_v41"
# The only place outside experimental/ that reaches into it, lazily and only
# for a selected profile (the one-way import lint allows this module alone).
_EXPERIMENTAL = "vllm_torchtpu.models.deepseek_v41.experimental"

# Environment variables the plan replaced: name -> the slot that now owns it.
# Setting one is an error, so a recipe cannot silently lose a switch.
REMOVED_ENV: dict[str, str] = {
    "TPU_DEEPSEEK_V41_CSA2_DEEP_GATHER": "csa2_attention",
    "TPU_DEEPSEEK_V41_BLOCKED_ROPE_GATHER": "csa2_attention",
    "TPU_DEEPSEEK_V41_QB_ROPE_EXACT_ROWS": "csa2_attention",
    "TPU_DEEPSEEK_V41_INDEXER_ISSUE_DEPTH": "indexer",
    "TPU_DEEPSEEK_V41_MHC_RESIDENT_GATES": "mhc",
    "TPU_DEEPSEEK_V41_MHC_EARLY_PREP": "mhc",
    "TPU_DEEPSEEK_V41_MHC_SHIFTED_T16": "mhc",
    "TPU_DEEPSEEK_V41_ENGRAM_FUSED_HASH": "engram",
    "TPU_DEEPSEEK_V41_ENGRAM_MASKS_IN_HASH": "engram",
    "TPU_DEEPSEEK_V41_ENGRAM_INT32_SLOT_WRITE": "engram",
    "TPU_DEEPSEEK_V41_ENGRAM_TILE_SCATTER": "engram",
    "TPU_DEEPSEEK_V41_ENGRAM_FUSED_PREP": "engram",
    "TPU_DEEPSEEK_V41_CED_IDENTITY_REPLAY": "ced",
    "TPU_DEEPSEEK_V41_DSPARK_PALLAS_LOGITS": "dspark_draft",
    "TPU_DEEPSEEK_V41_REUSE_WINDOW_STARTS": "decode_step",
    "TPU_ONEHOT_TABLE_LOOKUP": "decode_step",
    "TPU_BLOCK_TABLE_DELTA_BUCKETS": "decode_step",
    "TPU_FUSED_SPEC_NEXT_INPUTS": "decode_step",
    "TPU_INLINE_DEDUPLICATED_CALLS": "decode_step",
    "TPU_GC_POLICY": "host_gc",
    "TPU_GC_COLLECT_EVERY": "host_gc",
    "MOE_FUSED_EP_PUSH_SC_PREPUSH": "moe_kernel",
    "MOE_FUSED_EP_PUSH_TC_EXCHANGE": "moe_kernel",
    "MOE_FUSED_EP_PUSH_TAIL_BARRIER": "moe_kernel",
    "MOE_FUSED_EP_PUSH_ONEHOT_SLOTS": "moe_kernel",
    "MOE_FUSED_EP_PUSH_HBM_STAGING": "moe_kernel",
    "MOE_FUSED_EP_PUSH_DISPATCH_AHEAD": "moe_kernel",
    "MOE_FUSED_EP_PUSH_MATMUL_SCALE_REPEAT": "moe_kernel",
    "MOE_FUSED_EP_PUSH_EXCHANGE_ROW_GATHER": "moe_kernel",
    "MOE_FUSED_EP_PUSH_BF16_SORT_COUNTS": "moe_kernel",
    "MOE_FUSED_EP_PUSH_SLIM_PREPUSH_BARRIER": "moe_kernel",
    "MOE_FUSED_EP_PUSH_FUSED_DISPATCH_META": "moe_kernel",
    "MOE_FUSED_EP_PUSH_TABLE_KERNEL": "moe_kernel",
    "MOE_FUSED_EP_PUSH_EXCHANGE_TABLES": "moe_kernel",
    "MOE_FUSED_EP_PUSH_KERNEL_PLAN_TABLES": "moe_kernel",
    "MOE_FUSED_EP_PUSH_LIGHT_PREPUSH": "moe_kernel",
    "MOE_FUSED_EP_PUSH_SINGLE_BARRIER": "moe_kernel",
    "MOE_FUSED_EP_PUSH_EARLY_SCALE_SCATTER": "moe_kernel",
    "MOE_FUSED_EP_PUSH_SHARED_EXPERT": "moe_kernel",
}


def experimental_tables() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """(EXPERIMENTAL_VARIANTS, PROFILES), imported on demand."""
    variants = importlib.import_module(f"{_EXPERIMENTAL}.variants")
    profiles = importlib.import_module(f"{_EXPERIMENTAL}.profiles")
    return variants.EXPERIMENTAL_VARIANTS, profiles.PROFILES


def resolve_plan(context: PlanContext, profile: str | None = None) -> ResolvedPlan:
    if profile is None:
        return model_plan.resolve(MODEL, context, SUPPORTED_VARIANTS, SUPPORTED_PLAN)
    experimental, profiles = experimental_tables()
    if profile not in profiles:
        raise PlanError(
            f"Unknown deepseek_v41 model_plan profile {profile!r}; "
            f"known: {sorted(profiles)}"
        )
    return model_plan.resolve(
        MODEL,
        context,
        SUPPORTED_VARIANTS,
        SUPPORTED_PLAN,
        profile=profile,
        profile_overrides=profiles[profile],
        experimental_variants=experimental,
    )


def plan_context(config: VllmConfig, moe: str) -> PlanContext:
    parallel = config.parallel_config
    tp = getattr(parallel, "tensor_parallel_size", 1)
    dp = getattr(parallel, "data_parallel_size", 1)
    return PlanContext(
        tp_size=tp,
        dp_size=dp,
        ep_size=tp * dp if getattr(parallel, "enable_expert_parallel", False) else 1,
        moe=moe,
    )


def requested_profile(config: VllmConfig) -> str | None:
    additional = config.additional_config
    value = additional.get(model_plan.ADDITIONAL_CONFIG_KEY) if additional else None
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("additional_config.model_plan must be an object")
    unknown = value.keys() - {"profile", "resolved"}
    if unknown:
        raise ValueError(f"Unknown model_plan fields: {sorted(unknown)}")
    profile = value.get("profile")
    if profile is not None and not isinstance(profile, str):
        raise TypeError("model_plan.profile must be a string or null")
    return profile


def reject_removed_env() -> None:
    present = sorted(name for name in REMOVED_ENV if name in os.environ)
    if present:
        slots = sorted({REMOVED_ENV[name] for name in present})
        raise ValueError(
            f"{', '.join(present)} no longer exist: DeepSeek-V4.1 selects these "
            f"through the model plan (slots {', '.join(slots)}). Remove them and "
            'pass --additional-config \'{"model_plan": {"profile": "<name>"}}\' '
            "(models/deepseek_v41/experimental/profiles.py) for a benchmark recipe."
        )


def publish(config: VllmConfig, plan: ResolvedPlan) -> None:
    assert isinstance(config.additional_config, dict)
    config.additional_config[model_plan.ADDITIONAL_CONFIG_KEY] = {
        "profile": plan.profile,
        "resolved": plan.to_config(),
    }
    model_plan.activate(plan)
    logger.info(
        "[DSV41] model plan%s: %s",
        f" (profile {plan.profile})" if plan.profile else "",
        plan.selection(),
    )


@cache
def _supported_default() -> ResolvedPlan:
    # The fallback for code built without a config: eligibility rules see a
    # single-chip context, which only affects slots no model module reads.
    return resolve_plan(PlanContext())


def param(slot: str, key: str) -> Any:
    """The active plan's ``slot.key``; the SUPPORTED plan's when none is active."""
    plan = model_plan.active()
    if plan is None or plan.model != MODEL:
        plan = _supported_default()
    return plan.params[slot][key]
