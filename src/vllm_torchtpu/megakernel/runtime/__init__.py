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

from ..errors import EligibilityResult
from . import checks, eligibility, layout_facts, policy, step_meta
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
from .eligibility import (
    BatchShape,
    MKABCalibrator,
    StepDecision,
    StepMixTracker,
    check_mrope_uniform,
    check_post_mesh_eligibility,
    check_qsl_uniform,
    check_startup_eligibility,
    check_step_eligibility,
    inspect_batch_shape,
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
from .step_meta import (
    JaxStepMetaOps,
    StepMeta,
    StepMetaOps,
    check_ckpt_write_cols_distinct,
    pack_step_meta,
    resolve_step_positions,
)

__all__ = [
    "DEFAULT_BUCKETS",
    "DEFAULT_FUSED_POOL_IO_MAX_B",
    "DEFAULT_FUSED_POOL_IO_MIN_B_HI",
    "DEFAULT_VERIFY_BUCKETS",
    "MTP_SPECULATIVE_TOKENS",
    "MTP_VERIFY_TOKENS",
    "BatchShape",
    "BucketPolicy",
    "EligibilityResult",
    "JaxStepMetaOps",
    "MKABCalibrator",
    "PoolLayoutFacts",
    "PoolScheduleInfo",
    "SentinelVariantPlan",
    "StepDecision",
    "StepMeta",
    "StepMetaOps",
    "StepMixTracker",
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
    "step_meta",
    "verify_compiled_step_ir",
    "verify_ir_coverage",
    "verify_pooled_step_hlo",
    "verify_sentinel_step",
]
