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
"""TPU serving sanity contract tests separating physics and core readiness."""

from __future__ import annotations

import collections
import dataclasses
import itertools

import pytest

from vllm_torchtpu.megakernel import advisor
from vllm_torchtpu.megakernel.advisor import constants
from vllm_torchtpu.megakernel.core import errors

_CONTEXT_4K = 4096
_CONTEXT_32K = 32768
_QWEN_DECODE_BATCHES = (4, 8, 16)
_VERIFY_BATCH = 4
_VERIFY_SPEC_TOKENS = 3
_TP_EP_8 = 8
_KIMI_TP_EP_32 = 32
_SANITY_BATCH_8 = 8
_PRESET_SLUGS: tuple[str, ...] = ("qwen3.5-397b", "kimi-k3", "glm-5.2")
_MOE_SETTINGS: tuple[str, ...] = ("unpinned", "masked_on_chip", "grouped_on_chip")
_PINNED_MOE_MODES: dict[str, advisor.MoEMode] = {
    "masked_on_chip": "masked_on_chip",
    "grouped_on_chip": "grouped_on_chip",
}
_GRID_BATCHES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
_GRID_CONTEXTS: tuple[int, ...] = (1024, 4096, 32768, 131072)
_GRID_ISLS: tuple[int, ...] = (256, 1024, 4096, 8192)
_SAT_BATCH_600 = 600
_SAT_TP = 2
_SAT_DP = 4
_SAT_EP = 8
_HALF_DIVISOR = 2.0
_TIE_BREAK_BATCH = 32
_TIE_BREAK_TP = 4
_TIE_BREAK_EP = 8
_TIE_BREAK_DP = 2
_TIE_BREAK_SEG_LAYERS = 3
_TIE_BREAK_ETA_HBM = 0.70
_TIE_BREAK_ETA_DENSE = 0.85
_TIE_BREAK_ETA_MOE = 0.70
_TIE_BREAK_ETA_ICI = 0.90
_PP_8 = 8
_KIMI_MIXED_BATCH = 256
_KIMI_MIXED_ISL = 1024
_KIMI_MIXED_CHUNK = 256
_MAX_SHOWN_VIOLATIONS = 5


def _scope_by_name(
    report: advisor.AdvisorReport, name: advisor.FusionScope
) -> advisor.ScopeEstimate:
    """Return the `ScopeEstimate` entry matching `name` from `report`."""
    return next(scope_est for scope_est in report.scopes if scope_est.scope == name)


def test_qwen35_default_decode_and_verify_recommend_whole_step_within_core_limits() -> (
    None
):
    """Verify Qwen3.5 default decode and spec-verify recommend core-ready whole_step."""
    profile = advisor.resolve_model_profile("qwen3.5-397b")
    workloads = [
        advisor.WorkloadSpec(
            phase="decode",
            batch=batch,
            context=_CONTEXT_4K,
            tp=_TP_EP_8,
            ep=_TP_EP_8,
        )
        for batch in _QWEN_DECODE_BATCHES
    ]
    workloads.append(
        advisor.WorkloadSpec(
            phase="verify",
            batch=_VERIFY_BATCH,
            spec_tokens=_VERIFY_SPEC_TOKENS,
            context=_CONTEXT_4K,
            tp=_TP_EP_8,
            ep=_TP_EP_8,
        )
    )
    for workload in workloads:
        report = advisor.advise(profile, workload, "tpu7x")
        whole_step = _scope_by_name(report, "whole_step")
        per_op = _scope_by_name(report, "per_op")
        assert report.recommended_scope == "whole_step"
        assert report.core_support == "within_core_limits"
        assert report.core_ready is True
        assert report.core_blockers == ()
        assert report.core_ready_scope == "whole_step"
        assert report.build_megakernel is True
        assert whole_step.step_bound_us < per_op.step_bound_us
        serialized = report.to_dict()
        assert serialized["core_support"] == "within_core_limits"
        assert serialized["core_ready"] is True
        assert serialized["core_blockers"] == ()
        assert serialized["core_ready_scope"] == "whole_step"
        assert "hypothetical_best" not in serialized


def test_qwen35_fp32_state_keeps_physics_winner_and_flags_core_blocker() -> None:
    """Verify explicit FP32 state keeps whole_step physics winner and flags blocker."""
    profile = advisor.resolve_model_profile("qwen3.5-397b", state_dtype="fp32")
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=4,
        context=_CONTEXT_4K,
        tp=_TP_EP_8,
        ep=_TP_EP_8,
    )
    report = advisor.advise(profile, workload, "tpu7x")
    whole_step = _scope_by_name(report, "whole_step")
    per_op = _scope_by_name(report, "per_op")
    assert report.recommended_scope == "whole_step"
    assert report.core_support == "outside_core_limits"
    assert report.core_ready is False
    assert report.core_ready_scope == "per_op"
    assert "kv_or_state_dtype_not_in_core" in report.core_blockers
    assert "outside_core_limits" in report.rationale
    assert "kv_or_state_dtype_not_in_core" in report.rationale
    assert whole_step.step_bound_us < per_op.step_bound_us
    serialized = report.to_dict()
    assert serialized["core_support"] == "outside_core_limits"
    assert serialized["core_ready"] is False
    assert serialized["core_ready_scope"] == "per_op"
    assert serialized["core_blockers"] == ("kv_or_state_dtype_not_in_core",)
    assert "hypothetical_best" not in serialized


def test_kimi_k3_and_glm52_physics_winner_separated_from_core_readiness() -> None:
    """Verify Kimi-K3 and GLM-5.2 recommend whole_step while flagging core blockers."""
    cases = (
        (
            "kimi-k3",
            advisor.WorkloadSpec(
                phase="decode",
                batch=_SANITY_BATCH_8,
                context=_CONTEXT_4K,
                tp=_KIMI_TP_EP_32,
                ep=_KIMI_TP_EP_32,
            ),
        ),
        (
            "glm-5.2",
            advisor.WorkloadSpec(
                phase="decode",
                batch=_SANITY_BATCH_8,
                context=_CONTEXT_4K,
                tp=_TP_EP_8,
                ep=_TP_EP_8,
            ),
        ),
    )
    for model_slug, workload in cases:
        report = advisor.advise(model_slug, workload, "tpu7x")
        whole_step = _scope_by_name(report, "whole_step")
        per_op = _scope_by_name(report, "per_op")
        assert report.hbm_fits is True, model_slug
        assert report.recommended_scope == "whole_step", model_slug
        assert report.core_support == "outside_core_limits", model_slug
        assert report.core_ready is False, model_slug
        assert report.core_ready_scope == "per_op", model_slug
        assert len(report.core_blockers) >= 1, model_slug
        assert "hypothetical_best" not in report.to_dict(), model_slug
        assert "outside_core_limits" in report.rationale, model_slug
        assert whole_step.step_bound_us < per_op.step_bound_us, model_slug


def _apply_moe_setting(
    workload: advisor.WorkloadSpec, moe_setting: str
) -> advisor.WorkloadSpec:
    """Return `workload` unmodified when unpinned or pinned to `moe_setting`."""
    if moe_setting == "unpinned":
        return workload
    return dataclasses.replace(
        workload,
        moe_mode=_PINNED_MOE_MODES[moe_setting],
        moe_mode_explicit=True,
    )


def _named_cells_for_preset(
    preset_slug: str, moe_setting: str
) -> list[tuple[str, advisor.WorkloadSpec]]:
    """Return named must-cover regression cells for `preset_slug`."""
    if preset_slug == "qwen3.5-397b":
        return [
            (
                "named_qwen_sat_600",
                _apply_moe_setting(
                    advisor.WorkloadSpec(
                        phase="decode",
                        batch=_SAT_BATCH_600,
                        context=_CONTEXT_32K,
                        tp=2,
                        dp=4,
                        ep=8,
                    ),
                    moe_setting,
                ),
            ),
            (
                "named_qwen_pp8_4k",
                _apply_moe_setting(
                    advisor.WorkloadSpec(
                        phase="prefill",
                        batch=1,
                        isl=_CONTEXT_4K,
                        tp=1,
                        ep=1,
                        pp=_PP_8,
                    ),
                    moe_setting,
                ),
            ),
        ]
    if preset_slug == "kimi-k3":
        return [
            (
                "named_kimi_mixed_256",
                _apply_moe_setting(
                    advisor.WorkloadSpec(
                        phase="mixed",
                        batch=_KIMI_MIXED_BATCH,
                        isl=_KIMI_MIXED_ISL,
                        chunk_size=_KIMI_MIXED_CHUNK,
                        tp=_KIMI_TP_EP_32,
                        ep=_KIMI_TP_EP_32,
                        dcp=_KIMI_TP_EP_32,
                    ),
                    moe_setting,
                ),
            )
        ]
    return []


def _build_grid_workloads(
    preset_slug: str, moe_setting: str
) -> list[tuple[str, advisor.WorkloadSpec]]:
    """Build preset layout grid and named cells for `preset_slug` and `moe_setting`."""
    deployment = advisor.DEPLOYMENT_PRESETS[preset_slug]
    cells: list[tuple[str, advisor.WorkloadSpec]] = []
    for layout_name, layout in deployment.named_layouts():
        if layout_name == "prefill":
            for isl in _GRID_ISLS:
                workload = advisor.WorkloadSpec(
                    phase="prefill",
                    batch=layout.dp,
                    isl=isl,
                    tp=layout.tp,
                    ep=layout.ep,
                    dp=layout.dp,
                    pp=layout.pp,
                    dcp=layout.dcp,
                    pcp=layout.pcp,
                    layer_groups=layout.layer_groups,
                )
                cells.append((layout_name, _apply_moe_setting(workload, moe_setting)))
            continue
        spec_tokens = layout.spec_tokens or _VERIFY_SPEC_TOKENS
        for ctx in _GRID_CONTEXTS:
            for raw_batch in _GRID_BATCHES:
                batch = (
                    max(layout.dp, (raw_batch // layout.dp) * layout.dp)
                    if layout.dp > 1
                    else raw_batch
                )
                for phase_name, cell_spec_tokens in (
                    ("decode", 0),
                    ("verify", spec_tokens),
                ):
                    workload = advisor.WorkloadSpec(
                        phase=phase_name,
                        batch=batch,
                        context=ctx,
                        spec_tokens=cell_spec_tokens,
                        tp=layout.tp,
                        ep=layout.ep,
                        dp=layout.dp,
                        pp=layout.pp,
                        dcp=layout.dcp,
                        pcp=layout.pcp,
                        layer_groups=layout.layer_groups,
                    )
                    cells.append(
                        (layout_name, _apply_moe_setting(workload, moe_setting))
                    )
    cells.extend(_named_cells_for_preset(preset_slug, moe_setting))
    return cells


def _check_scope_envelope_invariants(
    report: advisor.AdvisorReport,
    tag: str,
    violations: list[str],
) -> None:
    """Append scope feasibility and bound-ordering violations for one report."""
    rec = _scope_by_name(report, report.recommended_scope)
    if not rec.feasible:
        violations.append(f"{tag} | rec_feasible | rec={rec.scope}")
    for scope_est in report.scopes:
        if scope_est.step_floor_strict_us > scope_est.step_bound_us:
            violations.append(
                f"{tag} | strict_le_floor | {scope_est.scope}: "
                f"{scope_est.step_floor_strict_us}>{scope_est.step_bound_us}"
            )
        if scope_est.step_bound_us > scope_est.step_upper_us:
            violations.append(
                f"{tag} | floor_le_upper | {scope_est.scope}: "
                f"{scope_est.step_bound_us}>{scope_est.step_upper_us}"
            )
        if scope_est.step_bound_us > scope_est.step_achievable_us:
            violations.append(
                f"{tag} | floor_le_est | {scope_est.scope}: "
                f"{scope_est.step_bound_us}>{scope_est.step_achievable_us}"
            )


def _check_verdict_and_core_invariants(
    report: advisor.AdvisorReport,
    tag: str,
    violations: list[str],
) -> None:
    """Append verdict, estimate-guard, core-readiness, and HBM OOM violations."""
    rec = _scope_by_name(report, report.recommended_scope)
    per_op = _scope_by_name(report, "per_op")
    fused = report.recommended_scope != "per_op"
    per_op_step_metric = report.metrics.get("per_op_step_us")
    measured_per_op_us = (
        float(per_op_step_metric["value"])
        if per_op_step_metric is not None
        and per_op_step_metric.get("tag") == "MEASURED"
        else None
    )
    checks = (
        (
            fused and rec.step_achievable_us > per_op.step_achievable_us,
            "fused_est_le_per_op",
            f"rec={rec.scope} est={rec.step_achievable_us}>{per_op.step_achievable_us}",
        ),
        (
            fused and rec.step_bound_us > per_op.step_bound_us,
            "fused_floor_le_per_op",
            f"rec={rec.scope} floor={rec.step_bound_us}>{per_op.step_bound_us}",
        ),
        (
            fused
            and measured_per_op_us is not None
            and rec.step_achievable_us > measured_per_op_us,
            "fused_est_le_measured_per_op",
            f"rec={rec.scope} est={rec.step_achievable_us}>{measured_per_op_us}",
        ),
        (
            report.build_megakernel
            and rec.step_achievable_us > per_op.step_achievable_us,
            "build_est_le_per_op",
            f"rec={rec.scope} est={rec.step_achievable_us}>{per_op.step_achievable_us}",
        ),
        (
            report.build_megakernel
            and measured_per_op_us is not None
            and rec.step_achievable_us > measured_per_op_us,
            "build_est_le_measured_per_op",
            f"rec={rec.scope} est={rec.step_achievable_us}>{measured_per_op_us}",
        ),
        (
            report.build_megakernel and report.verdict != "megakernel_likely",
            "build_verdict_likely",
            f"verdict={report.verdict}",
        ),
        (
            report.core_ready != (report.core_support == "within_core_limits"),
            "core_ready_matches_support",
            f"ready={report.core_ready} support={report.core_support}",
        ),
        (
            report.core_support == "within_core_limits"
            and report.core_ready_scope != report.recommended_scope,
            "within_limits_ready_scope",
            f"rec={report.recommended_scope} ready={report.core_ready_scope}",
        ),
        (
            report.core_support == "outside_core_limits" and not report.core_blockers,
            "outside_limits_blockers",
            f"blockers={report.core_blockers}",
        ),
        (
            not report.hbm_fits and (fused or report.build_megakernel),
            "hbm_oom_per_op_no_build",
            f"rec={report.recommended_scope} build={report.build_megakernel}",
        ),
    )
    for failed, check_name, values in checks:
        if failed:
            violations.append(f"{tag} | {check_name} | {values}")


def _check_measured_decode_variants(
    profile: advisor.ModelProfile,
    workload: advisor.WorkloadSpec,
    base_report: advisor.AdvisorReport,
    tag: str,
    violations: list[str],
) -> int:
    """Evaluate a decode cell at midpoint and modeled per_op measured baselines."""
    base_rec = _scope_by_name(base_report, base_report.recommended_scope)
    base_per_op = _scope_by_name(base_report, "per_op")
    midpoint_us = (base_rec.step_bound_us + base_rec.step_achievable_us) / _HALF_DIVISOR
    ran = 0
    for label, step_us in (
        ("mid", midpoint_us),
        ("per_op", base_per_op.step_achievable_us),
    ):
        measured_wl = dataclasses.replace(workload, per_op_step_us=step_us)
        measured_report = advisor.advise(profile, measured_wl, "tpu7x")
        ran += 1
        measured_tag = f"{tag} | measured={label}"
        if measured_report.metrics["per_op_step_us"]["tag"] != "MEASURED":
            violations.append(f"{measured_tag} | per_op_step_us_not_measured")
        _check_scope_envelope_invariants(measured_report, measured_tag, violations)
        _check_verdict_and_core_invariants(measured_report, measured_tag, violations)
    return ran


def _check_series_monotonicity(
    preset_slug: str,
    series: dict[
        tuple[str, advisor.Phase, int, advisor.FusionScope], list[tuple[int, float]]
    ],
    violations: list[str],
) -> None:
    """Append batch-monotonicity violations for HBM-fitting series points."""
    for (layout_name, phase_name, ctx_val, scope_name), points in sorted(
        series.items()
    ):
        points.sort()
        for (b0, f0), (b1, f1) in itertools.pairwise(points):
            if b1 > b0 and f1 < f0:
                violations.append(
                    f"{preset_slug} | {layout_name} | {phase_name} | "
                    f"{b0}->{b1} | {ctx_val} | floor_monotone | "
                    f"{scope_name}: {f0}->{f1}"
                )


@pytest.mark.parametrize("preset_slug", _PRESET_SLUGS)
@pytest.mark.parametrize("moe_setting", _MOE_SETTINGS)
def test_recommended_fused_scope_estimate_not_slower_than_per_op_on_grid(
    preset_slug: str,
    moe_setting: str,
) -> None:
    """Verify 13 report and series invariants across preset grids and MoE modes."""
    profile = advisor.resolve_model_profile(preset_slug)
    grid_workloads = _build_grid_workloads(preset_slug, moe_setting)
    ran_counts: collections.Counter[str] = collections.Counter()
    ran_phase_counts: collections.Counter[tuple[str, advisor.Phase]] = (
        collections.Counter()
    )
    rejected = 0
    violations: list[str] = []
    series: dict[
        tuple[str, advisor.Phase, int, advisor.FusionScope], list[tuple[int, float]]
    ] = collections.defaultdict(list)
    for layout_name, workload in grid_workloads:
        try:
            report = advisor.advise(profile, workload, "tpu7x")
        except errors.MegakernelConfigError:
            rejected += 1
            continue
        ran_counts[layout_name] += 1
        ran_phase_counts[(layout_name, workload.phase)] += 1
        ctx_label = (
            workload.context if workload.phase in ("decode", "verify") else workload.isl
        )
        tag = (
            f"{preset_slug} | {layout_name} | {workload.phase} | "
            f"{workload.batch} | {ctx_label}"
        )
        _check_scope_envelope_invariants(report, tag, violations)
        _check_verdict_and_core_invariants(report, tag, violations)
        if report.hbm_fits:
            for scope_est in report.scopes:
                series[
                    (layout_name, workload.phase, workload.context, scope_est.scope)
                ].append((workload.batch, scope_est.step_bound_us))
        if workload.phase == "decode":
            ran_counts[layout_name] += _check_measured_decode_variants(
                profile,
                workload,
                report,
                tag,
                violations,
            )
    expected_layouts = {name for name, _ in grid_workloads}
    for layout_name in sorted(expected_layouts):
        assert ran_counts[layout_name] > 0, (
            f"{preset_slug} ({moe_setting}) ran 0 cells for {layout_name}; "
            f"rejected={rejected}"
        )
    expected_pairs = {(name, wl.phase) for name, wl in grid_workloads}
    for layout_name, phase_name in sorted(expected_pairs):
        assert ran_phase_counts[(layout_name, phase_name)] > 0, (
            f"{preset_slug} ({moe_setting}) ran 0 {phase_name} cells for "
            f"{layout_name}; rejected={rejected}"
        )
    _check_series_monotonicity(preset_slug, series, violations)
    assert not violations, (
        f"{preset_slug} ({moe_setting}) {len(violations)} violations "
        f"(rejected={rejected}): {'; '.join(violations[:_MAX_SHOWN_VIOLATIONS])}"
    )


def test_unpinned_moe_mode_picks_best_admissible_mode() -> None:
    """Verify unpinned decode searches modes while explicit moe_mode pins selection."""
    sat_layout = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"].saturation
    assert sat_layout is not None
    base_workload = advisor.WorkloadSpec(
        phase="decode",
        batch=_SAT_BATCH_600,
        context=_CONTEXT_32K,
        tp=sat_layout.tp,
        dp=sat_layout.dp,
        ep=sat_layout.ep,
    )
    unpinned_report = advisor.advise("qwen3.5-397b", base_workload, "tpu7x")
    assert unpinned_report.recommended_scope == "whole_step"
    assert unpinned_report.recommended_moe_mode == "grouped_on_chip"
    assert (
        _scope_by_name(unpinned_report, "whole_step").step_achievable_us
        <= _scope_by_name(unpinned_report, "per_op").step_achievable_us
    )
    assert unpinned_report.build_megakernel is True

    masked_report = advisor.advise(
        "qwen3.5-397b",
        dataclasses.replace(
            base_workload, moe_mode="masked_on_chip", moe_mode_explicit=True
        ),
        "tpu7x",
    )
    masked_per_op = _scope_by_name(masked_report, "per_op")
    assert masked_report.recommended_scope == "single_layer"
    assert (
        _scope_by_name(masked_report, "single_layer").step_achievable_us
        <= masked_per_op.step_achievable_us
    )
    assert (
        _scope_by_name(masked_report, "whole_step").step_achievable_us
        > masked_per_op.step_achievable_us
    )
    assert masked_report.build_megakernel is False
    assert "whole_step" in masked_report.rationale

    grouped_report = advisor.advise(
        "qwen3.5-397b",
        dataclasses.replace(
            base_workload, moe_mode="grouped_on_chip", moe_mode_explicit=True
        ),
        "tpu7x",
    )
    assert grouped_report.recommended_scope == "whole_step"
    assert grouped_report.recommended_moe_mode == "grouped_on_chip"
    assert grouped_report.build_megakernel is True


def test_exact_floor_tie_breaks_on_estimate() -> None:
    """Verify exact floor tie on prefill PP8 rejects slower fused estimates."""
    prefill_layout = advisor.DEPLOYMENT_PRESETS["qwen3.5-397b"].prefill
    workload = advisor.WorkloadSpec(
        phase="prefill",
        batch=prefill_layout.dp,
        isl=_CONTEXT_4K,
        tp=prefill_layout.tp,
        ep=prefill_layout.ep,
        pp=prefill_layout.pp,
    )
    report = advisor.advise("qwen3.5-397b", workload, "tpu7x")
    per_op = _scope_by_name(report, "per_op")
    feasible_scopes = [scope_est for scope_est in report.scopes if scope_est.feasible]
    assert feasible_scopes
    assert all(
        scope_est.step_bound_us == per_op.step_bound_us for scope_est in feasible_scopes
    )
    assert report.recommended_scope == "per_op"
    assert report.build_megakernel is False


def test_measured_per_op_step_gates_fused_recommendation() -> None:
    """Verify measured per_op_step_us blocks slower fused recommendations."""
    base_workload = advisor.WorkloadSpec(
        phase="decode",
        batch=_SAT_BATCH_600,
        context=_CONTEXT_32K,
        tp=_SAT_TP,
        dp=_SAT_DP,
        ep=_SAT_EP,
    )
    unmeasured_report = advisor.advise("qwen3.5-397b", base_workload, "tpu7x")
    unmeasured_rec = _scope_by_name(
        unmeasured_report, unmeasured_report.recommended_scope
    )
    modeled_per_op = _scope_by_name(unmeasured_report, "per_op")
    midpoint_us = (
        unmeasured_rec.step_bound_us + unmeasured_rec.step_achievable_us
    ) / _HALF_DIVISOR

    fast_report = advisor.advise(
        "qwen3.5-397b",
        dataclasses.replace(base_workload, per_op_step_us=midpoint_us),
        "tpu7x",
    )
    assert fast_report.metrics["per_op_step_us"]["tag"] == "MEASURED"
    measured_fast_us = float(fast_report.metrics["per_op_step_us"]["value"])
    fast_rec = _scope_by_name(fast_report, fast_report.recommended_scope)
    if fast_report.recommended_scope != "per_op":
        assert fast_rec.step_achievable_us <= measured_fast_us
    assert not fast_report.build_megakernel or (
        fast_report.recommended_scope != "per_op"
        and fast_rec.step_achievable_us <= measured_fast_us
    )
    assert "measured per_op step" in fast_report.rationale

    meas_payload = {
        "schema": constants.MEASUREMENTS_SCHEMA,
        "source": {"kind": "xprof_xplane", "file": "sat.pb"},
        "per_op_step_us": midpoint_us,
    }
    meas_report = advisor.advise(
        "qwen3.5-397b", base_workload, "tpu7x", measurements=meas_payload
    )
    assert meas_report.metrics["per_op_step_us"]["tag"] == "MEASURED"
    assert meas_report.recommended_scope == fast_report.recommended_scope
    assert meas_report.build_megakernel is fast_report.build_megakernel

    slow_report = advisor.advise(
        "qwen3.5-397b",
        dataclasses.replace(
            base_workload, per_op_step_us=modeled_per_op.step_achievable_us
        ),
        "tpu7x",
    )
    assert slow_report.metrics["per_op_step_us"]["tag"] == "MEASURED"
    assert slow_report.recommended_scope == "whole_step"
    assert slow_report.build_megakernel is True


def test_rank_key_breaks_floor_upper_ops_tie_on_estimate() -> None:
    """Verify _scope_rank_key reaches step_achievable_us when floor/upper/ops tie."""
    eff = advisor.EfficiencySpec(
        eta_hbm=_TIE_BREAK_ETA_HBM,
        eta_dense=_TIE_BREAK_ETA_DENSE,
        eta_moe_grouped=_TIE_BREAK_ETA_MOE,
        eta_ici=_TIE_BREAK_ETA_ICI,
    )
    workload = advisor.WorkloadSpec(
        phase="decode",
        batch=_TIE_BREAK_BATCH,
        context=_CONTEXT_4K,
        tp=_TIE_BREAK_TP,
        ep=_TIE_BREAK_EP,
        dp=_TIE_BREAK_DP,
        segment_layers=_TIE_BREAK_SEG_LAYERS,
        moe_mode="grouped_on_chip",
        moe_mode_explicit=True,
        efficiency=eff,
    )
    report = advisor.advise("glm-5.2", workload, "tpu7x")
    per_op = _scope_by_name(report, "per_op")
    single_layer = _scope_by_name(report, "single_layer")
    layer_group = _scope_by_name(report, "layer_group")
    assert single_layer.feasible
    assert layer_group.feasible
    assert (
        single_layer.step_bound_us == layer_group.step_bound_us <= per_op.step_bound_us
    )
    assert single_layer.step_upper_us == layer_group.step_upper_us
    assert single_layer.ops_per_step == layer_group.ops_per_step
    assert (
        layer_group.step_achievable_us
        < single_layer.step_achievable_us
        <= per_op.step_achievable_us
    )
    assert report.recommended_scope == "layer_group"
