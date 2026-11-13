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
"""Megakernel core capability limits and default-deny support gating."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .constants import (
    CHIP_KIND_MAP,
    CORE_LIMITS,
    DEFAULT_DECLARED_PHASES,
    CoreLimits,
)
from .models import DEPLOYMENT_PRESETS, DeploymentPreset, ModelProfile
from .specs import (
    SUPPORTED_ONCHIP_FAMILIES,
    FusionScope,
    SupportStatus,
    WorkloadSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .chips import ChipSpec
    from .roofline import _ScopeContext

__all__ = [
    "CORE_LIMITS",
    "CoreLimits",
    "evaluate_support",
    "exceeds_default_buckets",
    "find_declaration",
    "single_call_row_limit",
    "whole_step_row_limit",
]


def _find_deployment_preset(profile_name: str) -> DeploymentPreset | None:
    """Resolve a DeploymentPreset by exact or normalized model preset name."""
    dep = DEPLOYMENT_PRESETS.get(profile_name)
    if dep is not None:
        return dep
    norm = profile_name.lower().replace(".", "_").replace("-", "_")
    for key, val in DEPLOYMENT_PRESETS.items():
        if key.lower().replace(".", "_").replace("-", "_") == norm:
            return val
    return None


def _string_items(raw: object) -> tuple[str, ...]:
    """Return a JSON list as a tuple of strings, or an empty tuple."""
    match raw:
        case list() | tuple():
            return tuple(str(item) for item in raw)
        case _:
            return ()


def _declared_int(declaration: Mapping[str, object] | None, key: str) -> int:
    """Return an integer field of a declaration, or 0 when it is absent."""
    return int(str(declaration.get(key, 0))) if declaration else 0


def _declared_blocks(
    profile_name: str, scope: FusionScope, phase: str
) -> tuple[Mapping[str, object], ...]:
    """Return the deployments.json megakernel blocks that declare scope for phase.

    A block without `phases` covers decode and verify only, so a decode
    declaration never applies to a prefill step. A block needs a source and a
    rank count.
    """
    dep = _find_deployment_preset(profile_name)
    if dep is None:
        return ()
    blocks: list[Mapping[str, object]] = []
    for layout in (dep.decode, dep.prefill, dep.saturation):
        mk = layout.megakernel if layout is not None else None
        if not mk or not str(mk.get("source", "")):
            continue
        if _declared_int(mk, "ranks") <= 0:
            continue
        phases = _string_items(mk.get("phases")) or DEFAULT_DECLARED_PHASES
        if scope in _string_items(mk.get("scopes")) and phase in phases:
            blocks.append(mk)
    return tuple(blocks)


def find_declaration(
    profile_name: str, scope: FusionScope, eff_tp: int, phase: str
) -> Mapping[str, object] | None:
    """Return the megakernel block declaring scope at this rank count and phase."""
    for mk in _declared_blocks(profile_name, scope, phase):
        if eff_tp == _declared_int(mk, "ranks"):
            return mk
    return None


def whole_step_row_limit(
    declaration: Mapping[str, object] | None, workload: WorkloadSpec
) -> int:
    """Return the most rows a whole step may hold.

    A workload limit means a kernel that stops at that row count and does not
    split. A declaration allows `max_total_rows` through split calls, else its
    single-call `max_rows`. Without any of these, the core batch limit applies.
    """
    return (
        workload.max_rows_per_call
        or _declared_int(declaration, "max_total_rows")
        or _declared_int(declaration, "max_rows")
        or CORE_LIMITS.max_batch_rows
    )


def single_call_row_limit(
    profile_name: str, workload: WorkloadSpec, eff_tp: int
) -> int:
    """Return the megakernel's single-call row limit for a decode or verify step.

    The workload's own limit wins; otherwise a deployments.json declaration for
    `whole_step` at this rank count supplies it. 0 means no declared limit.
    """
    if workload.phase not in ("decode", "verify"):
        return 0
    return workload.max_rows_per_call or _declared_int(
        find_declaration(profile_name, "whole_step", eff_tp, workload.phase),
        "max_rows",
    )


def exceeds_default_buckets(workload: WorkloadSpec, rows_per_call: int) -> bool:
    """Return whether one call is larger than the core's default bucket list."""
    tops = {
        "decode": CORE_LIMITS.default_decode_bucket_top,
        "verify": CORE_LIMITS.default_verify_bucket_top,
    }
    return rows_per_call > tops.get(workload.phase, rows_per_call)


def evaluate_support(
    scope: FusionScope,
    profile: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    ctx: _ScopeContext,
) -> tuple[SupportStatus, tuple[str, ...]]:
    """Return `(support_status, blockers)` for `scope` under `CORE_LIMITS`.

    A deployments.json declaration for this scope, rank count, and phase states
    that a runtime exists for the model. It then supplies the row limit and the
    layer families that `CORE_LIMITS` checks otherwise.
    """
    if scope == "per_op" or workload.batch <= 0:
        return ("baseline", ())

    declared_blocks = _declared_blocks(profile.name, scope, workload.phase)
    declaration = next(
        (mk for mk in declared_blocks if ctx.eff_tp == _declared_int(mk, "ranks")),
        None,
    )
    has_unmodeled = declaration is None and any(
        fam not in SUPPORTED_ONCHIP_FAMILIES for fam in profile.layer_families
    )
    # Declared rank counts restrict a model. Without a declaration for this scope
    # and phase, the core rank grid applies.
    rank_bad = (
        declaration is None
        if declared_blocks
        else ctx.eff_tp not in CORE_LIMITS.verified_ranks
    )
    attention_dp_in_call = declaration is not None and bool(
        declaration.get("attention_dp_in_call", False)
    )
    planner_bad = (
        bool(ctx.plan_error)
        or (declaration is None and not profile.is_periodic)
        or profile.ar_slots_per_layer > CORE_LIMITS.max_ar_slots_per_layer
    )
    checks = (
        (
            scope in ("single_layer", "layer_group"),
            "scope_without_core_counterpart",
        ),
        (
            ctx.active_tokens > whole_step_row_limit(declaration, workload),
            "rows_exceed_core_max_batch_rows",
        ),
        (
            rank_bad,
            "rank_count_outside_core_grid",
        ),
        (
            CHIP_KIND_MAP.get(chip_spec.name.lower()) not in CORE_LIMITS.chip_kinds,
            "chip_without_core_budget",
        ),
        (ctx.ep_dp > 1 and not attention_dp_in_call, "all_to_all_not_in_core"),
        (has_unmodeled, "unmodeled_layer_family"),
        (profile.architecture_unverified, "model_type_unrecognized"),
        (planner_bad, "planner_rejected"),
    )
    blockers = tuple(tag for cond, tag in checks if cond)
    if blockers:
        return ("hypothetical", blockers)
    if declaration is not None and ctx.active_tokens > 0:
        return ("declared", ())
    return ("within_core_limits", ())
