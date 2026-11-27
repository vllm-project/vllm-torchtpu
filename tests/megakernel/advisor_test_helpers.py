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
"""Capability-driven shared helpers for megakernel advisor test suites."""

from __future__ import annotations

import dataclasses
import math
from itertools import pairwise
from typing import TYPE_CHECKING

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import constants

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_MAX_CORE_RANK = 16
DEFAULT_DECODE_REPLICA_BATCH = 16
DEFAULT_PREFILL_REPLICA_BATCH = 1


def minimal_valid_core_layout(
    layout: advisor.ParallelLayout,
    *,
    max_rank: int = DEFAULT_MAX_CORE_RANK,
) -> advisor.ParallelLayout:
    """Return a valid core-sized layout preserving EP/DP coupling and modes.

    Returns:
        `ParallelLayout` clamped to `max_rank` with valid DP when `tp % ep != 0`.
    """
    tp = min(layout.tp, max_rank)
    ep = min(layout.ep, max_rank)
    dp = layout.dp if tp % max(1, ep) != 0 else 1
    return advisor.ParallelLayout(
        tp=tp,
        ep=ep,
        dp=dp,
        pp=1,
        dcp=1,
        pcp=1,
        execution=layout.execution,
        host_table_hbm_overrides=layout.host_table_hbm_overrides,
    )


def build_preset_workload(
    model_slug: str,
    phase: advisor.Phase,
    *,
    per_replica_batch: int | None = None,
    use_decode_layout: bool = False,
    **overrides: object,
) -> advisor.WorkloadSpec:
    """Build a DP-aligned WorkloadSpec projected from a shipped deployment preset.

    Returns:
        Configured `WorkloadSpec` instance with layout and overrides applied.
    """
    preset = advisor.DEPLOYMENT_PRESETS[model_slug]
    layout = (
        preset.decode
        if use_decode_layout or phase in ("decode", "verify")
        else preset.prefill
    )
    replica_batch = (
        per_replica_batch
        if per_replica_batch is not None
        else (
            DEFAULT_PREFILL_REPLICA_BATCH
            if phase == "prefill"
            else DEFAULT_DECODE_REPLICA_BATCH
        )
    )
    spec_tokens = (
        (
            layout.spec_tokens
            or preset.decode.spec_tokens
            or constants.DEFAULT_SPEC_TOKENS
        )
        if phase in ("verify", "mixed")
        else 0
    )
    base = advisor.WorkloadSpec(
        phase=phase,
        batch=replica_batch * layout.dp,
        spec_tokens=spec_tokens,
    ).with_layout(layout)
    if not overrides:
        return base
    replace_fn: Callable[..., advisor.WorkloadSpec] = dataclasses.replace
    return replace_fn(base, **overrides)


def _expected_fusion_break_layers(
    profile: advisor.ModelProfile,
    workload: advisor.WorkloadSpec,
) -> tuple[int, ...]:
    """Return sorted 0-based fusion-break layers implied by profile and workload."""
    breaks: set[int] = set()
    for table in profile.host_tables:
        is_resident = next(
            (
                bool(is_hbm)
                for name, is_hbm in workload.effective_host_table_hbm_overrides
                if name == table.name
            ),
            table.hbm_resident,
        )
        if not is_resident:
            for layer_idx in table.layers:
                if 0 < layer_idx < profile.num_layers:
                    breaks.add(layer_idx)
    if (
        workload.effective_execution == "ced"
        and profile.ced_boundary > 0
        and workload.phase in ("prefill", "mixed")
        and 0 < profile.ced_boundary < profile.num_layers
    ):
        breaks.add(profile.ced_boundary)
    return tuple(sorted(breaks))


def _expected_break_layer_groups(
    num_layers: int, break_layers: tuple[int, ...]
) -> tuple[int, ...]:
    """Partition `num_layers` at sorted `break_layers` into contiguous group sizes."""
    if not break_layers or num_layers <= 1:
        return ()
    valid_cuts = sorted({lyr for lyr in break_layers if 0 < lyr < num_layers})
    if not valid_cuts:
        return ()
    bounds = (0, *valid_cuts, num_layers)
    return tuple(hi - lo for lo, hi in pairwise(bounds))


def expected_scope_feasibility(
    profile: advisor.ModelProfile,
    workload: advisor.WorkloadSpec,
    *,
    plan_error: str = "",
) -> dict[advisor.FusionScope, bool]:
    """Derive exact per-scope feasibility from profile and workload rules.

    Returns:
        Mapping from each `FusionScope` to expected structural feasibility.
    """
    layers_local = math.ceil(profile.num_layers / workload.pp)
    num_segments = (
        math.ceil(layers_local / workload.segment_layers)
        if 0 < workload.segment_layers < layers_local
        else 1
    )
    breaks = _expected_fusion_break_layers(profile, workload)
    has_fusion_break = bool(breaks)
    eff_layer_groups = workload.layer_groups or _expected_break_layer_groups(
        profile.num_layers, breaks
    )
    num_layer_groups = (
        math.ceil(len(eff_layer_groups) * layers_local / profile.num_layers)
        if eff_layer_groups
        else 0
    )
    layer_group_ok = not (
        num_layer_groups == 0 and workload.pp == 1 and num_segments == 1
    )
    whole_step_ok = not (
        workload.pp > 1
        or num_segments > 1
        or len(workload.layer_groups) > 1
        or has_fusion_break
        or profile.num_layers == 1
        or bool(plan_error)
    )
    return {
        "per_op": True,
        "single_layer": True,
        "layer_group": layer_group_ok,
        "whole_step": whole_step_ok,
    }
