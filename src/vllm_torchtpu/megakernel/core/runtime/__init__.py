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
"""Framework-free runtime helpers for `megakernel.core`."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # eager imports for type checkers and IDEs
    from . import contract, eligibility, policy, stats
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
    from .policy import (
        DEFAULT_BUCKETS,
        DEFAULT_VERIFY_BUCKETS,
        MTP_SPECULATIVE_TOKENS,
        MTP_VERIFY_TOKENS,
        BucketPolicy,
        parse_mk_buckets,
        parse_mk_verify_buckets,
        select_bucket,
    )
    from .stats import (
        BatchShape,
        MKABCalibrator,
        StepMixTracker,
        check_mrope_uniform,
        check_query_start_loc_uniform,
        inspect_batch_shape,
    )

# Public names resolve on first access (PEP 562): importing a submodule
# (e.g. `runtime.contract`) doesn't import jax through this package. The
# framework-free part is `contract` (stdlib) and `stats` (numpy); `policy` and
# `eligibility` load on first access.
_SUBMODULES = frozenset(
    {
        "contract",
        "stats",
        "eligibility",
        "policy",
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
    "check_query_start_loc_uniform": ".stats",
    "inspect_batch_shape": ".stats",
    "check_post_mesh_eligibility": ".eligibility",
    "check_startup_eligibility": ".eligibility",
    "check_step_eligibility": ".eligibility",
    "DEFAULT_BUCKETS": ".policy",
    "DEFAULT_VERIFY_BUCKETS": ".policy",
    "MTP_SPECULATIVE_TOKENS": ".policy",
    "MTP_VERIFY_TOKENS": ".policy",
    "BucketPolicy": ".policy",
    "parse_mk_buckets": ".policy",
    "parse_mk_verify_buckets": ".policy",
    "select_bucket": ".policy",
}

__all__ = [
    "DOMAIN_TP",
    "PARALLELISM_UNSUPPORTED",
    "REQUIRES_KEYS",
    "REQUIRES_UNMET",
    "UNSUPPORTED_TPU_CHIP",
    "UNSUPPORTED_TP_SIZE",
    "DEFAULT_BUCKETS",
    "DEFAULT_VERIFY_BUCKETS",
    "MTP_SPECULATIVE_TOKENS",
    "MTP_VERIFY_TOKENS",
    "BatchShape",
    "Bucket",
    "BucketPolicy",
    "Decline",
    "EligibilityResult",
    "LayoutDecline",
    "MKABCalibrator",
    "MonoProgram",
    "RunnerShape",
    "StepDecision",
    "StepMixTracker",
    "TpuMonoSpec",
    "WarmupCase",
    "check_mrope_uniform",
    "check_post_mesh_eligibility",
    "check_query_start_loc_uniform",
    "check_startup_eligibility",
    "check_step_eligibility",
    "contract",
    "eligibility",
    "inspect_batch_shape",
    "parse_mk_buckets",
    "parse_mk_verify_buckets",
    "policy",
    "select_bucket",
    "stats",
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
