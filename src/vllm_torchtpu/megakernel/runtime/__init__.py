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
"""Framework-free runtime helpers for `megakernel`."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # eager imports for type checkers and IDEs
    from . import checks, contract, eligibility, layout_facts, policy, stats, step_meta
    from .checks import (
        PoolScheduleInfo,
        SentinelVariantPlan,
        WarmupStepPlan,
        build_global_avals,
        build_warmup_plan,
        check_compile_hlo_mesh,
        ir_required_buckets,
        pool_schedule_info,
        verify_compiled_step_ir,
        verify_ir_coverage,
        verify_pooled_step_hlo,
        verify_sentinel_step,
    )
    from .contract import (
        DOMAIN_TP,
        PARALLELISM_UNSUPPORTED,
        REQUIRES_KEYS,
        REQUIRES_UNMET,
        UNSUPPORTED_TP_SIZE,
        UNSUPPORTED_TPU_CHIP,
        Bucket,
        Decline,
        EligibilityResult,
        LayoutDecline,
        MonoProgram,
        RunnerShape,
        StepDecision,
        TpuMonoSpec,
        WarmupCase,
    )
    from .eligibility import (
        check_post_mesh_eligibility,
        check_startup_eligibility,
        check_step_eligibility,
    )
    from .layout_facts import (
        PoolLayoutFacts,
        compare_layout_facts,
        facts_from_pool_layout,
    )
    from .policy import (
        DEFAULT_BUCKETS,
        DEFAULT_FUSED_POOL_IO_MAX_B,
        DEFAULT_FUSED_POOL_IO_MIN_B_HI,
        DEFAULT_VERIFY_BUCKETS,
        MTP_SPECULATIVE_TOKENS,
        MTP_VERIFY_TOKENS,
        BucketPolicy,
        align_cache_exec,
        bucket_exec_spec,
        explicit_profile_settings,
        parse_mk_buckets,
        parse_mk_verify_buckets,
        parse_runtime_config,
        resolve_driver_profile,
        select_bucket,
    )
    from .stats import (
        BatchShape,
        MKABCalibrator,
        StepMixTracker,
        check_mrope_uniform,
        check_qsl_uniform,
        inspect_batch_shape,
    )
    from .step_meta import (
        JaxStepMetaOps,
        StepMeta,
        StepMetaOps,
        check_ckpt_write_cols_distinct,
        pack_step_meta,
        resolve_step_positions,
    )

# Public names resolve on first access (PEP 562): importing a submodule
# (e.g. `runtime.contract`) doesn't import jax through this package. The
# framework-free part is `contract` (stdlib) and `stats` (numpy); the rest is
# policy- / kernel-bound.
_SUBMODULES = frozenset(
    {
        "contract",
        "stats",
        "checks",
        "eligibility",
        "layout_facts",
        "policy",
        "step_meta",
    }
)
_LAZY: dict[str, str] = {
    "DOMAIN_TP": ".contract",
    "PARALLELISM_UNSUPPORTED": ".contract",
    "REQUIRES_KEYS": ".contract",
    "REQUIRES_UNMET": ".contract",
    "UNSUPPORTED_TP_SIZE": ".contract",
    "UNSUPPORTED_TPU_CHIP": ".contract",
    "Bucket": ".contract",
    "Decline": ".contract",
    "EligibilityResult": ".contract",
    "LayoutDecline": ".contract",
    "MonoProgram": ".contract",
    "RunnerShape": ".contract",
    "StepDecision": ".contract",
    "TpuMonoSpec": ".contract",
    "WarmupCase": ".contract",
    "BatchShape": ".stats",
    "MKABCalibrator": ".stats",
    "StepMixTracker": ".stats",
    "check_mrope_uniform": ".stats",
    "check_qsl_uniform": ".stats",
    "inspect_batch_shape": ".stats",
    "PoolScheduleInfo": ".checks",
    "SentinelVariantPlan": ".checks",
    "WarmupStepPlan": ".checks",
    "build_global_avals": ".checks",
    "build_warmup_plan": ".checks",
    "check_compile_hlo_mesh": ".checks",
    "ir_required_buckets": ".checks",
    "pool_schedule_info": ".checks",
    "verify_compiled_step_ir": ".checks",
    "verify_ir_coverage": ".checks",
    "verify_pooled_step_hlo": ".checks",
    "verify_sentinel_step": ".checks",
    "check_post_mesh_eligibility": ".eligibility",
    "check_startup_eligibility": ".eligibility",
    "check_step_eligibility": ".eligibility",
    "PoolLayoutFacts": ".layout_facts",
    "compare_layout_facts": ".layout_facts",
    "facts_from_pool_layout": ".layout_facts",
    "DEFAULT_BUCKETS": ".policy",
    "DEFAULT_FUSED_POOL_IO_MAX_B": ".policy",
    "DEFAULT_FUSED_POOL_IO_MIN_B_HI": ".policy",
    "DEFAULT_VERIFY_BUCKETS": ".policy",
    "MTP_SPECULATIVE_TOKENS": ".policy",
    "MTP_VERIFY_TOKENS": ".policy",
    "BucketPolicy": ".policy",
    "align_cache_exec": ".policy",
    "bucket_exec_spec": ".policy",
    "explicit_profile_settings": ".policy",
    "parse_mk_buckets": ".policy",
    "parse_mk_verify_buckets": ".policy",
    "parse_runtime_config": ".policy",
    "resolve_driver_profile": ".policy",
    "select_bucket": ".policy",
    "JaxStepMetaOps": ".step_meta",
    "StepMeta": ".step_meta",
    "StepMetaOps": ".step_meta",
    "check_ckpt_write_cols_distinct": ".step_meta",
    "pack_step_meta": ".step_meta",
    "resolve_step_positions": ".step_meta",
}

__all__ = [
    "DOMAIN_TP",
    "PARALLELISM_UNSUPPORTED",
    "REQUIRES_KEYS",
    "REQUIRES_UNMET",
    "UNSUPPORTED_TPU_CHIP",
    "UNSUPPORTED_TP_SIZE",
    "DEFAULT_BUCKETS",
    "DEFAULT_FUSED_POOL_IO_MAX_B",
    "DEFAULT_FUSED_POOL_IO_MIN_B_HI",
    "DEFAULT_VERIFY_BUCKETS",
    "MTP_SPECULATIVE_TOKENS",
    "MTP_VERIFY_TOKENS",
    "BatchShape",
    "Bucket",
    "BucketPolicy",
    "Decline",
    "EligibilityResult",
    "JaxStepMetaOps",
    "LayoutDecline",
    "MKABCalibrator",
    "MonoProgram",
    "PoolLayoutFacts",
    "PoolScheduleInfo",
    "RunnerShape",
    "SentinelVariantPlan",
    "StepDecision",
    "StepMeta",
    "StepMetaOps",
    "StepMixTracker",
    "TpuMonoSpec",
    "WarmupCase",
    "WarmupStepPlan",
    "align_cache_exec",
    "bucket_exec_spec",
    "build_global_avals",
    "build_warmup_plan",
    "check_ckpt_write_cols_distinct",
    "check_compile_hlo_mesh",
    "check_mrope_uniform",
    "check_post_mesh_eligibility",
    "check_qsl_uniform",
    "check_startup_eligibility",
    "check_step_eligibility",
    "checks",
    "contract",
    "compare_layout_facts",
    "eligibility",
    "explicit_profile_settings",
    "facts_from_pool_layout",
    "inspect_batch_shape",
    "ir_required_buckets",
    "layout_facts",
    "pack_step_meta",
    "parse_mk_buckets",
    "parse_mk_verify_buckets",
    "parse_runtime_config",
    "policy",
    "pool_schedule_info",
    "resolve_driver_profile",
    "resolve_step_positions",
    "select_bucket",
    "stats",
    "step_meta",
    "verify_compiled_step_ir",
    "verify_ir_coverage",
    "verify_pooled_step_hlo",
    "verify_sentinel_step",
]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        mod_obj = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod_obj
        return mod_obj
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
