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
"""Physics-only fusion scope verdict engine and search utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import groupby
from math import ceil, log2
from typing import TYPE_CHECKING

from .chips import ChipSpec, resolve_chip_spec
from .constants import (
    BOUNDARY_SYNC_HOP_FACTOR,
    COLLECTIVE_PATH_IN_KERNEL,
    COLLECTIVE_PATH_XLA,
    COMPILED_XLA_GAP_P50_US,
    COMPILED_XLA_GAP_P90_US,
    CORE_LIMITS,
    DCP_COLLECTIVES_PER_ATTN_LAYER,
    DEFAULT_LAUNCH_GAP_FLOOR_US,
    DEFAULT_LAUNCH_GAP_UPPER_US,
    DEFAULT_LAUNCH_GAP_VAL_US,
    EQUAL_SPLIT_GROUP_COUNTS,
    FUSED_ELIGIBLE_PHASES,
    GAIN_BASIS_CALIBRATED_E2E,
    GAIN_BASIS_DEVICE_ONLY,
    LAYER_GROUP_PRICING_NOTE,
    PER_OP_COLLECTIVE_CALL_FACTOR,
    QUANTILE_DECIMALS,
    SUBLAYER_BOUNDARIES_PER_LAYER,
    UNFUSED_EAGER_GAP_US,
    XLA_COLLECTIVE_HBM_STAGING_PASSES,
)
from .models import (
    DEPLOYMENT_PRESETS,
    ModelProfile,
    ParallelLayout,
    resolve_model_profile,
)
from .roofline import (
    _ar_call_matches,
    _evaluate_scopes,
    _evaluate_scopes_with_context,
    _EvaluatedScopes,
    _is_calibrated_trace,
    _ScopeContext,
    can_group_moe_in_scope,
)
from .specs import (
    AdvisorReport,
    FusionScope,
    GainBasis,
    LayerGroupCandidate,
    LayerGroupSearchReport,
    MeasurementsData,
    MoEMode,
    ParallelismComparisonRow,
    ProvenanceMetric,
    ProvenanceTag,
    ScopeEstimate,
    WorkloadSpec,
    load_measurements,
)
from .vmem import (
    _mesh_divisibility_reasons,
    effective_pcp,
    effective_tp,
    exceeds_default_buckets,
    validate_advise_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

FLOOR_TIE_EPSILON_US = 1e-6
MICROSECONDS_PER_SECOND = 1e6
SECONDS_PER_MICROSECOND = 1e-6
BYTES_PER_GB = 1e9
BYTES_PER_MB = 1e6
MIB_BIT_SHIFT = 20
TFLOPS_DIVISOR = 1e12
PERCENT_SCALE = 100.0
STEP_DENOM_FLOOR_US = 1e-6
TIME_DENOM_FLOOR_S = 1e-9
ROUND_US_DECIMALS = 2
ROUND_GB_DECIMALS = 3
ROUND_RATIO_DECIMALS = 4
PAIR_COUNT = 2
A2A_COLLECTIVES_PER_LAYER = 2
STEP_SETTLE_HINT = "Supply calibrated step trace measurements for this chip and mesh"
ETA_SETTLE_HINT = "Calibrate achievable unit utilization from XProf"
DECODER_COVERAGE_STATEMENT = (
    "decoder layer stack; excludes embedding, LM head, sampling, "
    "draft heads and modality towers"
)
SUPPORTED_SCOPE_STATUSES = ("baseline", "within_core_limits")
SEARCHED_MOE_MODES: tuple[MoEMode, MoEMode] = ("masked_on_chip", "grouped_on_chip")
SCOPE_VERDICT_MAP: Mapping[FusionScope, str] = {
    "whole_step": "megakernel_likely",
    "layer_group": "megakernel_likely",
    "single_layer": "fused_layer_likely",
    "per_op": "separate_likely",
}


@dataclass(frozen=True, slots=True)
class _PriorInterval:
    """Resolved scalar prior, inclusive `[lo, hi]` interval, tag, and source."""

    value_us: float
    lo_us: float
    hi_us: float
    tag: ProvenanceTag
    source: str


@dataclass(frozen=True, slots=True)
class _StepCoveragePrior:
    """Resolved active-step coverage fraction, `[lo, hi]` band, tag, and source."""

    value: float
    band: tuple[float, float]
    tag: ProvenanceTag
    source: str


@dataclass(frozen=True, slots=True)
class _CalibratedPriors:
    """Resolved empirical or prior intervals used across roofline bands."""

    launch_gap: _PriorInterval
    collective_latency: _PriorInterval
    step_coverage: _StepCoveragePrior
    xla_all_reduce: _PriorInterval
    in_kernel_all_reduce: _PriorInterval


@dataclass(frozen=True, slots=True)
class _EvalInputs:
    """Resolved context, priors, and estimates handed to report assembly."""

    profile: ModelProfile
    effective_workload: WorkloadSpec
    eval_workload: WorkloadSpec
    chip_spec: ChipSpec
    scope_context: _ScopeContext
    measurements_data: MeasurementsData | None
    priors: _CalibratedPriors
    estimates: tuple[ScopeEstimate, ...]
    raw_step_map: Mapping[FusionScope, float]


@dataclass(frozen=True, slots=True)
class _ScopeSelection:
    """Fastest and core-ready scopes under HBM and mesh constraints."""

    hbm_fits: bool
    hbm_pool_overflow: bool
    divisibility_reasons: tuple[str, ...]
    fastest: ScopeEstimate
    core_ready: ScopeEstimate
    runner_up: ScopeEstimate | None
    upper_winner: ScopeEstimate
    filtered_lower_floor: ScopeEstimate | None = None
    measured_per_op_us: float | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedPerOpBaseline:
    """Resolved `per_op` baseline band, provenance, and un-gapped floor."""

    base_per_op_us: float
    band: tuple[float, float, float]
    tag: ProvenanceTag
    source: str


@dataclass(frozen=True, slots=True)
class _SaturationOutcome:
    """Updated verdict strings, flags, and rationale after saturation check."""

    verdict: str
    fastest_verdict: str
    flags: tuple[str, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class _GainSummary:
    """Fact vs assumed fusion gains, per-op baseline, and chip throughput band."""

    base_per_op_us: float
    pure_diff_us: float
    fact_diff_us: float
    assumed_diff_us: float
    ar_band_extra_us: float
    collectives_measured: bool
    device_gain_tag: ProvenanceTag
    boundary_band: tuple[float, float, float]
    boundary_tag: ProvenanceTag
    per_op_band: tuple[float, float, float]
    per_op_tag: ProvenanceTag
    per_op_source: str
    fusion_gain_band: tuple[float, float, float]
    expected_gain_band: tuple[float, float, float]
    gain_basis: GainBasis
    break_even_host_head_us: float
    throughput_low: float
    throughput_high: float
    recommended_upper_us: float


@dataclass(frozen=True, slots=True)
class _VerdictBundle:
    """Resolved verdict strings, confidence, flags, metrics, and assumptions."""

    verdict: str
    fastest_verdict: str
    confidence: str
    recommended_scope: FusionScope
    recommended: ScopeEstimate
    core_ready_scope: FusionScope
    flags: tuple[str, ...]
    metrics: dict[str, ProvenanceMetric]
    unresolved_assumptions: tuple[dict[str, object], ...]
    rationale: str
    divisibility_reasons: tuple[str, ...]


PROVENANCE_METRIC_META: Mapping[str, tuple[str, str]] = {
    "fits_hbm": ("bool", "hbm_used <= chip_spec.hbm_capacity_bytes"),
    "tokens_per_core": ("tok", "WorkloadSpec batch * phase tokens / (dp * pcp)"),
    "onchip_ceiling": (
        "tok",
        "(vmem_budget - static_vmem_bytes) // per_token_vmem_bytes",
    ),
    "compute_bound_at": ("tok", "w_step * ridge_flop_per_byte / flop_per_tok"),
    "gain_fact_us": (
        "us",
        "per_op_zero_hop_us - best_zero_hop_us (HBM weight/act/staging save)",
    ),
    "break_even_host_head_delta_us": (
        "us",
        "max(0, fused_device_bound_us - per_op_device_bound_us)",
    ),
    "throughput_tok_s_per_chip": (
        "tok/s/chip",
        "step_tokens * devices_per_chip / (eff_tp * pp * pcp) / floor",
    ),
    "boundary_cost_us": (
        "us",
        "(per_op_ops - scope_ops) * launch_gap_us + collective_saving_us + "
        "host_head_delta",
    ),
    "gain_assumed_us": (
        "us",
        "compiler_proof_gain_us - gain_fact_us (collective hop & call save)",
    ),
    "compiler_proof_gain_us": (
        "us",
        "per_op_physics_us - best_scope_physics_us (0 us boundary gap)",
    ),
    "compiler_proof_gain_pct": (
        "pct",
        "100 * compiler_proof_gain_us / per_op_physics_us",
    ),
    "achievable_gain_pct": (
        "pct",
        "100 * achievable_saving_us / per_op_achievable_us",
    ),
    "dispatch_bubble_gain_us": ("us", "(per_op_ops - best_scope_ops) * launch_gap_us"),
    "fusion_gain_max": ("ratio", "boundary_cost_us / per_op_step_us"),
    "expected_gain_max": ("ratio", "fusion_gain_max * step_coverage"),
    "gain_basis": ("mode", "per_op_step_us and step_coverage provenance"),
    "per_op_collective_call_factor": ("x", "constants.PER_OP_COLLECTIVE_CALL_FACTOR"),
    "boundary_sync_hop_factor": ("hops", "constants.BOUNDARY_SYNC_HOP_FACTOR"),
}

ASSUMPTION_META: Mapping[str, tuple[str, str, str]] = {
    "launch_gap_us": ("us", "scope", STEP_SETTLE_HINT),
    "collective_latency_us": ("us/hop", "scope", STEP_SETTLE_HINT),
    "prefill_attn_dot_dtype": (
        "TFLOP/s",
        "scope",
        "Inspect compiler profile HLO attention dot precision",
    ),
    "boundary_sync_hop_factor": (
        "hops",
        "scope",
        "Run an A/B step trace comparing per_op vs fused boundary fill/drain",
    ),
    "per_op_collective_call_factor": (
        "x",
        "scope",
        "Run an A/B step trace comparing per_op vs fused all-reduce latency",
    ),
    "decoding_sequences": (
        "seqs",
        "support",
        "Specify active decoding batch size matching the target serving regime",
    ),
    "context": (
        "tok",
        "gain",
        "Provide an explicit context length matching the serving KV window",
    ),
    "weight_dtype": (
        "dtype",
        "gain",
        "Provide an explicit weight dtype or resolve a quantized HF config",
    ),
    "max_rows_per_call": (
        "rows",
        "support",
        "Specify an explicit per-call row ceiling within core limits",
    ),
    "step_coverage": ("ratio", "gain", STEP_SETTLE_HINT),
    "per_op_step_us": (
        "us",
        "gain",
        "Provide a measured per_op baseline step latency from step traces",
    ),
    "host_head_us": (
        "us",
        "gain",
        "Measure pre-first-op host dispatch head from baseline step traces",
    ),
    "eta_hbm": ("ratio", "gain", ETA_SETTLE_HINT),
    "eta_dense": ("ratio", "gain", ETA_SETTLE_HINT),
    "eta_moe_grouped": ("ratio", "gain", ETA_SETTLE_HINT),
    "eta_ici": ("ratio", "gain", ETA_SETTLE_HINT),
}


def _make_assumption(
    name: str,
    value: float | str,
    band: list[float | str],
    sensitivity_us: float,
) -> dict[str, object]:
    """Serialize one unresolved ASSUMED parameter entry from `ASSUMPTION_META`."""
    unit, affects, hint = ASSUMPTION_META[name]
    return {
        "name": name,
        "value": value,
        "band": band,
        "unit": unit,
        "sensitivity_us": sensitivity_us,
        "affects": affects,
        "how_to_settle": hint,
    }


def _format_unapplied_collectives(
    measurements_data: MeasurementsData | None,
    *,
    exclude_path_group: tuple[str, int, float] | None = None,
) -> str:
    """Summarize loaded collective entries that were not applied to a path."""
    if measurements_data is None or not measurements_data.collective_calls:
        return ""
    unapplied = [
        call
        for call in measurements_data.collective_calls
        if exclude_path_group is None or not _ar_call_matches(call, *exclude_path_group)
    ]
    if not unapplied:
        return ""
    labels = list(
        dict.fromkeys(
            f"{call.op or 'unnamed'} (path={call.path}, group_size={call.group_size})"
            for call in unapplied
        )
    )
    prefix = (
        f"; not applied from {measurements_data.source_desc}"
        if exclude_path_group is None
        else "; not applied"
    )
    return f"{prefix}: {', '.join(labels)}"


def _resolve_collective_call_prior(
    scope_context: _ScopeContext,
    measurements_data: MeasurementsData | None,
    path: str,
    measured_s: float | None,
) -> _PriorInterval:
    """Return single-call all-reduce cost, band, tag, and source on `path`."""
    eff_tp = scope_context.eff_tp
    if eff_tp <= 1:
        return _PriorInterval(
            0.0,
            0.0,
            0.0,
            "DERIVED",
            "single-rank workload (no tensor-parallel all-reduce)",
        )
    expected_bytes = scope_context.token_residual_bytes
    if (
        measured_s is not None
        and measurements_data is not None
        and _is_calibrated_trace(measurements_data)
    ):
        matched = [
            call
            for call in measurements_data.collective_calls
            if _ar_call_matches(call, path, eff_tp, expected_bytes)
        ]
        total_n = sum(call.n for call in matched)
        stat_label = (
            "mean"
            if any(call.mean_latency_us is not None for call in matched)
            else "median"
        )
        rounded_us = round(measured_s * MICROSECONDS_PER_SECOND, QUANTILE_DECIMALS)
        note = _format_unapplied_collectives(
            measurements_data, exclude_path_group=(path, eff_tp, expected_bytes)
        )
        src = f"{measurements_data.source_desc} ({stat_label}, n={total_n}){note}"
        return _PriorInterval(rounded_us, rounded_us, rounded_us, "MEASURED", src)
    call_factor = PER_OP_COLLECTIVE_CALL_FACTOR if path == COLLECTIVE_PATH_XLA else 1
    hbm_staging_s = (
        (XLA_COLLECTIVE_HBM_STAGING_PASSES * expected_bytes) / scope_context.bw_hbm
        if path == COLLECTIVE_PATH_XLA
        else 0.0
    )
    lo_s = (
        scope_context.ar_transfer_time_s + scope_context.ar_base_lat_s + hbm_staging_s
    )
    extra_hop_s = max(0.0, scope_context.lat_coll_upper - scope_context.lat_coll)
    hi_s = (
        scope_context.ar_transfer_time_s
        + call_factor
        * (scope_context.ar_base_lat_s + scope_context.ar_intra_hops * extra_hop_s)
        + hbm_staging_s
    )
    lo_us = round(lo_s * MICROSECONDS_PER_SECOND, QUANTILE_DECIMALS)
    hi_us = round(hi_s * MICROSECONDS_PER_SECOND, QUANTILE_DECIMALS)
    note = _format_unapplied_collectives(measurements_data)
    src = f"modelled {path} all-reduce call prior at group_size={eff_tp}{note}"
    return _PriorInterval(lo_us, lo_us, hi_us, "ASSUMED", src)


def _resolve_launch_gap_prior(
    workload: WorkloadSpec,
    measurements_data: MeasurementsData | None,
) -> _PriorInterval:
    """Resolve launch gap bounds and provenance tag across dispatch regimes."""
    if workload.dispatch_regime == "zero_floor":
        return _PriorInterval(
            0.0,
            0.0,
            0.0,
            "DERIVED",
            "zero_floor compiler-proof physics regime (0.0 us/boundary)",
        )
    if workload.dispatch_regime == "unfused_eager":
        gap_us = workload.op_overhead_us or UNFUSED_EAGER_GAP_US
        return _PriorInterval(
            gap_us,
            gap_us,
            max(gap_us, DEFAULT_LAUNCH_GAP_UPPER_US),
            "ASSUMED",
            "unfused_eager exposed dispatch sensitivity regime",
        )
    if (
        measurements_data is not None
        and measurements_data.launch_gap_p50_us is not None
    ):
        gap_lo = measurements_data.launch_gap_p50_us
        gap_hi = max(gap_lo, measurements_data.launch_gap_p90_us or gap_lo)
        is_cal = _is_calibrated_trace(measurements_data)
        tag: ProvenanceTag = "MEASURED" if is_cal else "ASSUMED"
        count_note = f" (n={measurements_data.launch_gap_n})" if is_cal else ""
        return _PriorInterval(
            gap_lo, gap_lo, gap_hi, tag, f"{measurements_data.source_desc}{count_note}"
        )
    if workload.dispatch_regime == "compiled_xla":
        gap_us = workload.op_overhead_us or COMPILED_XLA_GAP_P50_US
        src = (
            "caller WorkloadSpec.op_overhead_us override"
            if workload.op_overhead_us > 0.0
            else "compiled-XLA sublayer boundary prior "
            "(uncalibrated; supply trace measurements)"
        )
        return _PriorInterval(
            gap_us, gap_us, max(gap_us, COMPILED_XLA_GAP_P90_US), "ASSUMED", src
        )
    if workload.op_overhead_us > 0.0:
        gap_us = workload.op_overhead_us
        return _PriorInterval(
            gap_us,
            gap_us,
            max(gap_us, DEFAULT_LAUNCH_GAP_UPPER_US),
            "ASSUMED",
            "caller WorkloadSpec.op_overhead_us override",
        )
    return _PriorInterval(
        DEFAULT_LAUNCH_GAP_VAL_US,
        DEFAULT_LAUNCH_GAP_FLOOR_US,
        DEFAULT_LAUNCH_GAP_UPPER_US,
        "ASSUMED",
        "uncalibrated TPU host/device launch gap prior [1.0, 8.0] us",
    )


def _build_calibrated_priors(
    evaluated: _EvaluatedScopes,
    chip_spec: ChipSpec,
    measurements_data: MeasurementsData | None,
) -> _CalibratedPriors:
    """Assemble provenance priors from evaluated scopes, chip spec, and traces."""
    workload, scope_context = evaluated.effective_workload, evaluated.scope_context
    coll_low_us, coll_high_us = chip_spec.collective_latency_us_bracket
    coll_source = (chip_spec.sources or {}).get(
        "collective_latency_us_bracket", "uncalibrated ICI hop latency bracket"
    )
    coll_val_us = round(
        evaluated.collective_latency_s * MICROSECONDS_PER_SECOND, ROUND_GB_DECIMALS
    )
    coll_info = _PriorInterval(
        coll_val_us,
        min(coll_low_us, coll_val_us),
        max(coll_high_us, coll_val_us),
        "ASSUMED",
        coll_source,
    )
    if workload.step_coverage is not None and evaluated.step_coverage is not None:
        coverage_value = evaluated.step_coverage
        coverage_prior = _StepCoveragePrior(
            coverage_value,
            (coverage_value, coverage_value),
            "MEASURED",
            "caller WorkloadSpec.step_coverage (workload step_coverage)",
        )
    elif (
        measurements_data is not None
        and measurements_data.step_coverage is not None
        and evaluated.step_coverage is not None
    ):
        coverage_value = evaluated.step_coverage
        coverage_tag: ProvenanceTag = (
            "MEASURED" if _is_calibrated_trace(measurements_data) else "ASSUMED"
        )
        coverage_prior = _StepCoveragePrior(
            coverage_value,
            (coverage_value, coverage_value),
            coverage_tag,
            f"{measurements_data.source_desc} step_coverage",
        )
    elif workload.phase == "mixed":
        decode_tokens = scope_context.batch_local * max(1, scope_context.verify_tokens)
        coverage_value = round(
            min(1.0, decode_tokens / max(1, scope_context.active_tokens)),
            ROUND_RATIO_DECIMALS,
        )
        coverage_prior = _StepCoveragePrior(
            coverage_value,
            (0.0, 1.0),
            "ASSUMED",
            "decode token share on mixed step (settle via calibrated measurements)",
        )
    else:
        coverage_prior = _StepCoveragePrior(
            1.0,
            (0.0, 1.0),
            "ASSUMED",
            "single-phase step coverage prior (settle via calibrated measurements)",
        )
    return _CalibratedPriors(
        launch_gap=_resolve_launch_gap_prior(workload, measurements_data),
        collective_latency=coll_info,
        step_coverage=coverage_prior,
        xla_all_reduce=_resolve_collective_call_prior(
            scope_context,
            measurements_data,
            COLLECTIVE_PATH_XLA,
            evaluated.measured_ar_xla_s,
        ),
        in_kernel_all_reduce=_resolve_collective_call_prior(
            scope_context,
            measurements_data,
            COLLECTIVE_PATH_IN_KERNEL,
            evaluated.measured_ar_in_kernel_s,
        ),
    )


def _round2(value: float) -> float:
    """Round a microsecond or percentage value to display precision."""
    return round(value, ROUND_US_DECIMALS)


def _metric(
    unit: str,
    tag: ProvenanceTag,
    source: str,
    band: tuple[float, float],
    *,
    value: float | str | bool,
) -> ProvenanceMetric:
    """Build a normalized `ProvenanceMetric` entry with ordered band bounds."""
    low, high = band
    return {
        "value": value,
        "unit": unit,
        "tag": tag,
        "source": source,
        "band": [min(low, high), max(low, high)],
    }


def _scope_rank_key(
    scope_estimate: ScopeEstimate,
    raw_step_map: Mapping[FusionScope, float],
) -> tuple[int, float, int, float]:
    """Order scopes by raw step time, upper bound, op count, and estimate."""
    raw_step_us = raw_step_map.get(scope_estimate.scope, scope_estimate.step_bound_us)
    return (
        round(raw_step_us / FLOOR_TIE_EPSILON_US),
        scope_estimate.step_upper_us,
        scope_estimate.ops_per_step,
        scope_estimate.step_achievable_us,
    )


def _select_scopes(
    inputs: _EvalInputs,
    *,
    measured_per_op_us: float | None = None,
) -> _ScopeSelection:
    """Select fastest scope beating modeled and measured per_op, plus core-ready."""
    estimates = inputs.estimates
    raw_step_map = inputs.raw_step_map
    scope_context = inputs.scope_context
    workload = inputs.effective_workload
    chip_spec = inputs.chip_spec
    hbm_used_bytes = scope_context.hbm_used
    hbm_fits = hbm_used_bytes <= chip_spec.hbm_capacity_bytes
    hbm_pool_overflow = (
        hbm_used_bytes > workload.hbm_utilization_cap * chip_spec.hbm_capacity_bytes
    )
    divisibility_reasons = tuple(
        _mesh_divisibility_reasons(inputs.profile, workload, chip_spec)
    )
    mesh_ok = hbm_fits and not divisibility_reasons
    per_op_estimate = estimates[0]
    if mesh_ok:
        candidates = [
            estimate
            for estimate in estimates
            if estimate.scope == "per_op"
            or (
                estimate.feasible
                and estimate.step_bound_us <= per_op_estimate.step_bound_us
                and estimate.step_achievable_us <= per_op_estimate.step_achievable_us
                and (
                    measured_per_op_us is None
                    or estimate.step_achievable_us <= measured_per_op_us
                )
            )
        ]
    else:
        candidates = [per_op_estimate]

    def _rank(estimate: ScopeEstimate) -> tuple[int, float, int, float]:
        """Rank candidate scope estimates by floor, upper bound, ops, and estimate."""
        return _scope_rank_key(estimate, raw_step_map)

    fastest = min(candidates, key=_rank) if mesh_ok else per_op_estimate
    candidate_scopes = {estimate.scope for estimate in candidates}
    filtered_lower = (
        [
            estimate
            for estimate in estimates
            if estimate.feasible
            and estimate.scope not in candidate_scopes
            and estimate.step_bound_us < fastest.step_bound_us
        ]
        if mesh_ok
        else []
    )
    filtered_lower_floor = min(filtered_lower, key=_rank) if filtered_lower else None
    core_ready_candidates = (
        [
            estimate
            for estimate in candidates
            if estimate.support in SUPPORTED_SCOPE_STATUSES
        ]
        or [per_op_estimate]
        if mesh_ok
        else [per_op_estimate]
    )
    core_ready = min(core_ready_candidates, key=_rank) if mesh_ok else per_op_estimate
    others = (
        [estimate for estimate in candidates if estimate.scope != fastest.scope]
        if mesh_ok
        else []
    )
    runner_up = min(others, key=_rank) if others else None
    upper_winner = (
        min(
            candidates,
            key=lambda estimate: (
                estimate.step_upper_us,
                estimate.step_bound_us,
                estimate.ops_per_step,
            ),
        )
        if mesh_ok
        else per_op_estimate
    )
    return _ScopeSelection(
        hbm_fits=hbm_fits,
        hbm_pool_overflow=hbm_pool_overflow,
        divisibility_reasons=divisibility_reasons,
        fastest=fastest,
        core_ready=core_ready,
        runner_up=runner_up,
        upper_winner=upper_winner,
        filtered_lower_floor=filtered_lower_floor,
        measured_per_op_us=measured_per_op_us,
    )


def _compute_asymptotic_intensity(
    workload: WorkloadSpec,
    scope_context: _ScopeContext,
    state_bytes: float,
) -> float | None:
    """Compute asymptotic long-context decode/verify arithmetic intensity."""
    if not state_bytes or workload.phase not in FUSED_ELIGIBLE_PHASES:
        return None
    return round(
        (scope_context.flops_dense + scope_context.flops_moe) / state_bytes,
        ROUND_US_DECIMALS,
    )


def _select_recommended_moe_mode(
    inputs: _EvalInputs,
    fastest: ScopeEstimate,
) -> str:
    """Select optimal MoE dispatch mode for the fastest scope."""
    scope_context = inputs.scope_context
    if scope_context.local_moe_layers == 0:
        return "dense_mlp"
    if fastest.scope == "per_op":
        return "grouped_in_hbm"
    workload = inputs.effective_workload
    can_group = can_group_moe_in_scope(
        fastest.scope, scope_context.eff_tp, scope_context.ep_dp, workload.moe_mode
    )
    tokens_per_ep_group = (
        ceil(scope_context.active_tokens / max(1, fastest.num_vmem_splits))
        * scope_context.ep_dp
    )
    meets_row_tile = tokens_per_ep_group >= inputs.chip_spec.matmul_row_tile(
        inputs.profile.expert_dtype or inputs.profile.weight_dtype
    )
    if can_group and (workload.moe_mode == "grouped_on_chip" or meets_row_tile):
        return "grouped_on_chip"
    return "masked_on_chip"


def _resolve_per_op_step(
    workload: WorkloadSpec,
    measurements_data: MeasurementsData | None,
    per_op_estimate: ScopeEstimate,
    base_per_op_us: float,
    launch_gap_us: float,
) -> tuple[tuple[float, float, float], ProvenanceTag, str]:
    """Resolve measured or modeled `per_op` baseline step time and band."""
    if workload.per_op_step_us is not None:
        step_us = round(workload.per_op_step_us, ROUND_US_DECIMALS)
        return (
            (step_us, step_us, step_us),
            "MEASURED",
            "caller WorkloadSpec.per_op_step_us",
        )
    if measurements_data is not None and _is_calibrated_trace(measurements_data):
        suffix = (
            f" [image={measurements_data.source_image}]"
            if measurements_data.source_image
            else ""
        )
        candidates = (
            (
                measurements_data.per_op_step_us,
                f"{measurements_data.source_desc} per_op_step_us{suffix}",
            ),
            (
                measurements_data.step_time_us
                if measurements_data.step_time_us is not None
                and measurements_data.step_time_us > 0.0
                and measurements_data.source_scope in ("", "per_op")
                else None,
                f"{measurements_data.source_desc} step_time_us{suffix}",
            ),
        )
        for raw_step, source in candidates:
            if raw_step is not None:
                step_us = round(raw_step, ROUND_US_DECIMALS)
                return (step_us, step_us, step_us), "MEASURED", source
    modeled_us = round(
        base_per_op_us
        + per_op_estimate.ops_per_step * launch_gap_us
        + per_op_estimate.host_head_us,
        ROUND_US_DECIMALS,
    )
    return (
        (
            modeled_us,
            per_op_estimate.step_bound_us,
            max(per_op_estimate.step_bound_us, per_op_estimate.step_upper_us),
        ),
        "ASSUMED",
        "modeled per_op step bound (settle via measured baseline step trace)",
    )


def _clamped_ratio_band(
    value_num: float,
    low_num: float,
    high_num: float,
    denom: float,
) -> tuple[float, float, float]:
    """Normalize `(value, low, high)` numerators into clamped `[0, 1]` ratios."""
    val = round(min(1.0, max(0.0, value_num / denom)), ROUND_RATIO_DECIMALS)
    low = round(min(val, max(0.0, low_num / denom)), ROUND_RATIO_DECIMALS)
    high = round(max(val, min(1.0, high_num / denom)), ROUND_RATIO_DECIMALS)
    return (val, low, high)


def _compute_fusion_gain_band(
    per_op_estimate: ScopeEstimate,
    fastest: ScopeEstimate,
    per_op_resolved: tuple[float, ProvenanceTag, float, float],
    boundary_band: tuple[float, float, float],
    launch_gap_us: float,
) -> tuple[float, float, float]:
    """Compute fractional step-time reduction band vs `per_op`."""
    per_op_value, per_op_tag, base_per_op_us, base_best_us = per_op_resolved
    if per_op_tag == "MEASURED":
        if fastest.scope == "per_op":
            return (0.0, 0.0, 0.0)
        denom = max(STEP_DENOM_FLOOR_US, per_op_value)
        best_at_gap = (
            base_best_us + fastest.ops_per_step * launch_gap_us + fastest.host_head_us
        )
        best_hi = max(fastest.step_bound_us, fastest.step_upper_us)
        return _clamped_ratio_band(
            per_op_value - best_at_gap,
            per_op_value - best_hi,
            per_op_value - fastest.step_bound_us,
            denom,
        )
    denom = max(
        STEP_DENOM_FLOOR_US,
        base_per_op_us
        + per_op_estimate.ops_per_step * launch_gap_us
        + per_op_estimate.host_head_us,
    )
    return _clamped_ratio_band(*boundary_band, denom)


def _zero_hop_gain_split(
    inputs: _EvalInputs,
    fastest: ScopeEstimate,
    pure_diff_us: float,
) -> tuple[float, float]:
    """Return `(fact_diff_us, assumed_diff_us)` with collective hops zeroed."""
    pp_zero = (
        inputs.scope_context.token_residual_bytes
        / inputs.chip_spec.ici_link_bw_bytes_per_s
        if inputs.effective_workload.pp > 1
        else 0.0
    )
    zero_hop_context = replace(
        inputs.scope_context,
        lat_coll=0.0,
        lat_coll_upper=0.0,
        ar_base_lat_s=0.0,
        ar_inter_latency_s=0.0,
        pp_stage_time=pp_zero,
        launch_gap_upper_us=0.0,
        measured_ar_xla_s=None,
        measured_ar_in_kernel_s=None,
    )
    zero_hop_workload = replace(
        inputs.effective_workload,
        op_overhead_us=0.0,
        host_head_per_op_us=0.0,
        host_head_fused_us=0.0,
    )
    zero_est, _ = _evaluate_scopes(zero_hop_context, inputs.profile, zero_hop_workload)
    zero_by_scope = {estimate.scope: estimate for estimate in zero_est}
    pure_rounded = round(pure_diff_us, ROUND_US_DECIMALS)
    zero_diff_us = round(
        max(
            0.0,
            zero_by_scope["per_op"].step_bound_us
            - zero_by_scope[fastest.scope].step_bound_us,
        ),
        ROUND_US_DECIMALS,
    )
    factual_diff_us = min(pure_rounded, zero_diff_us)
    assumed_diff_us = round(max(0.0, pure_rounded - factual_diff_us), ROUND_US_DECIMALS)
    return factual_diff_us, assumed_diff_us


def _compute_throughput_band(
    inputs: _EvalInputs,
    recommended: ScopeEstimate,
    *,
    hbm_fits: bool,
) -> tuple[float, float, float]:
    """Return `(throughput_low, throughput_high, recommended_upper_us)`."""
    workload = inputs.effective_workload
    scope_context = inputs.scope_context
    eff_pcp = effective_pcp(workload)
    decode_tokens = scope_context.batch_local * scope_context.verify_tokens
    step_tokens = (
        decode_tokens + max(0, scope_context.active_tokens - decode_tokens) * eff_pcp
        if workload.phase == "mixed"
        else scope_context.active_tokens * eff_pcp
    )
    tokens_per_chip = (
        step_tokens
        * inputs.chip_spec.devices_per_chip
        / max(1, scope_context.eff_tp * workload.pp * eff_pcp)
    )
    recommended_upper_us = max(recommended.step_bound_us, recommended.step_upper_us)
    if not (hbm_fits and recommended.feasible):
        return 0.0, 0.0, recommended_upper_us
    high = round(
        tokens_per_chip
        / max(TIME_DENOM_FLOOR_S, recommended.step_bound_us * SECONDS_PER_MICROSECOND),
        ROUND_US_DECIMALS,
    )
    low = round(
        tokens_per_chip
        / max(TIME_DENOM_FLOOR_S, recommended_upper_us * SECONDS_PER_MICROSECOND),
        ROUND_US_DECIMALS,
    )
    return low, high, recommended_upper_us


def _resolve_per_op_baseline(inputs: _EvalInputs) -> _ResolvedPerOpBaseline:
    """Resolve the `per_op` baseline once from inputs and launch priors."""
    per_op_estimate = inputs.estimates[0]
    gap_floor = inputs.eval_workload.op_overhead_us
    base_per_op_us = (
        per_op_estimate.step_bound_us
        - per_op_estimate.ops_per_step * gap_floor
        - per_op_estimate.host_head_us
    )
    band, tag, source = _resolve_per_op_step(
        inputs.effective_workload,
        inputs.measurements_data,
        per_op_estimate,
        base_per_op_us,
        inputs.priors.launch_gap.value_us,
    )
    return _ResolvedPerOpBaseline(
        base_per_op_us=base_per_op_us,
        band=band,
        tag=tag,
        source=source,
    )


def _compute_gains_and_throughput(
    inputs: _EvalInputs,
    selection: _ScopeSelection,
    per_op_baseline: _ResolvedPerOpBaseline,
) -> _GainSummary:
    """Compute zero-hop device gains, boundary savings, and chip throughput."""
    per_op_estimate, fastest = inputs.estimates[0], selection.fastest
    gap_floor = inputs.eval_workload.op_overhead_us
    base_per_op_us = per_op_baseline.base_per_op_us
    base_fastest_us = (
        fastest.step_bound_us - fastest.ops_per_step * gap_floor - fastest.host_head_us
    )
    raw_diff_us = base_per_op_us - base_fastest_us
    pure_diff_us = max(0.0, raw_diff_us)
    factual_diff_us, assumed_diff_us = _zero_hop_gain_split(
        inputs, fastest, pure_diff_us
    )

    priors = inputs.priors
    collectives_measured = (
        priors.xla_all_reduce.tag == "MEASURED"
        and priors.in_kernel_all_reduce.tag == "MEASURED"
    )
    collective_span_us = max(
        0.0, priors.collective_latency.hi_us - priors.collective_latency.lo_us
    )
    all_reduce_extra_us = (
        0.0
        if (collectives_measured or fastest.scope == "per_op")
        else inputs.scope_context.num_ar
        * inputs.scope_context.ar_hops
        * collective_span_us
    )
    device_gain_tag: ProvenanceTag = (
        "ASSUMED" if (assumed_diff_us > 0.0 and not collectives_measured) else "DERIVED"
    )
    delta_ops = max(0, per_op_estimate.ops_per_step - fastest.ops_per_step)
    host_head_gain_us = per_op_estimate.host_head_us - fastest.host_head_us
    launch_prior = priors.launch_gap
    boundary_band = (
        _round2(
            max(
                0.0, delta_ops * launch_prior.value_us + raw_diff_us + host_head_gain_us
            )
        ),
        _round2(
            max(0.0, delta_ops * launch_prior.lo_us + raw_diff_us + host_head_gain_us)
        ),
        _round2(
            max(
                0.0,
                delta_ops * launch_prior.hi_us
                + raw_diff_us
                + host_head_gain_us
                + all_reduce_extra_us,
            )
        ),
    )
    boundary_tag: ProvenanceTag = (
        "MEASURED"
        if launch_prior.tag == "MEASURED" and collectives_measured
        else "ASSUMED"
    )
    per_op_band = per_op_baseline.band
    per_op_tag = per_op_baseline.tag
    per_op_source = per_op_baseline.source
    fusion_gain_band = _compute_fusion_gain_band(
        per_op_estimate,
        fastest,
        (per_op_band[0], per_op_tag, base_per_op_us, base_fastest_us),
        boundary_band,
        launch_prior.value_us,
    )
    step_coverage_prior = priors.step_coverage
    expected_gain_band = (
        round(fusion_gain_band[0] * step_coverage_prior.value, ROUND_RATIO_DECIMALS),
        round(fusion_gain_band[1] * step_coverage_prior.band[0], ROUND_RATIO_DECIMALS),
        round(fusion_gain_band[2] * step_coverage_prior.band[1], ROUND_RATIO_DECIMALS),
    )
    gain_basis: GainBasis = (
        GAIN_BASIS_CALIBRATED_E2E
        if (per_op_tag == "MEASURED" and step_coverage_prior.tag == "MEASURED")
        else GAIN_BASIS_DEVICE_ONLY
    )
    mesh_ok = selection.hbm_fits and not selection.divisibility_reasons
    recommended = selection.fastest
    if not mesh_ok:
        break_even_host_head_us = 0.0
    elif recommended.scope != "per_op":
        break_even_host_head_us = recommended.break_even_host_head_delta_us
    else:
        feasible_fused = [
            estimate
            for estimate in inputs.estimates
            if estimate.scope != "per_op" and estimate.feasible
        ]
        pool = [
            estimate
            for estimate in feasible_fused
            if estimate.support == "within_core_limits"
        ] or (feasible_fused if fastest.scope == "per_op" else [])
        break_even_host_head_us = (
            min(estimate.break_even_host_head_delta_us for estimate in pool)
            if pool
            else 0.0
        )
    throughput_low, throughput_high, recommended_upper_us = _compute_throughput_band(
        inputs, recommended, hbm_fits=mesh_ok
    )
    return _GainSummary(
        base_per_op_us=base_per_op_us,
        pure_diff_us=pure_diff_us,
        fact_diff_us=factual_diff_us,
        assumed_diff_us=assumed_diff_us,
        ar_band_extra_us=all_reduce_extra_us,
        collectives_measured=collectives_measured,
        device_gain_tag=device_gain_tag,
        boundary_band=boundary_band,
        boundary_tag=boundary_tag,
        per_op_band=per_op_band,
        per_op_tag=per_op_tag,
        per_op_source=per_op_source,
        fusion_gain_band=fusion_gain_band,
        expected_gain_band=expected_gain_band,
        gain_basis=gain_basis,
        break_even_host_head_us=break_even_host_head_us,
        throughput_low=throughput_low,
        throughput_high=throughput_high,
        recommended_upper_us=recommended_upper_us,
    )


def _build_provenance_metrics(
    inputs: _EvalInputs,
    selection: _ScopeSelection,
    ridge_tokens: int,
    gains: _GainSummary,
) -> dict[str, ProvenanceMetric]:
    """Build the full metric provenance map across geometry, priors, and gains."""
    priors = inputs.priors
    recommended, fastest, per_op_estimate = (
        selection.fastest,
        selection.fastest,
        inputs.estimates[0],
    )
    launch_prior, collective_prior, step_coverage_prior = (
        priors.launch_gap,
        priors.collective_latency,
        priors.step_coverage,
    )
    delta_ops = max(0, per_op_estimate.ops_per_step - fastest.ops_per_step)
    pure_gain_low_us = round(gains.pure_diff_us, ROUND_US_DECIMALS)
    pure_gain_high_us = round(
        gains.pure_diff_us + gains.ar_band_extra_us, ROUND_US_DECIMALS
    )
    base_denom = max(STEP_DENOM_FLOOR_US, gains.base_per_op_us)
    proof_pct_low = round(
        PERCENT_SCALE * gains.pure_diff_us / base_denom, ROUND_US_DECIMALS
    )
    proof_pct_high = round(
        PERCENT_SCALE * (gains.pure_diff_us + gains.ar_band_extra_us) / base_denom,
        ROUND_US_DECIMALS,
    )
    achievable_gain_pct = round(
        PERCENT_SCALE
        * (per_op_estimate.step_achievable_us - fastest.step_achievable_us)
        / max(STEP_DENOM_FLOOR_US, per_op_estimate.step_achievable_us),
        ROUND_US_DECIMALS,
    )
    assumed_tag: ProvenanceTag = (
        "MEASURED"
        if gains.collectives_measured
        else ("ASSUMED" if gains.assumed_diff_us > 0.0 else "DERIVED")
    )
    is_e2e = gains.gain_basis == GAIN_BASIS_CALIBRATED_E2E
    workload = inputs.effective_workload
    efficiency = workload.efficiency
    out: dict[str, ProvenanceMetric] = {}
    point_vals: tuple[tuple[str, float | bool], ...] = (
        ("fits_hbm", selection.hbm_fits),
        ("tokens_per_core", inputs.scope_context.active_tokens),
        ("onchip_ceiling", fastest.max_vmem_tokens),
        ("compute_bound_at", ridge_tokens),
        ("gain_fact_us", gains.fact_diff_us),
        ("break_even_host_head_delta_us", gains.break_even_host_head_us),
    )
    for name, val in point_vals:
        unit, src = PROVENANCE_METRIC_META[name]
        scalar_value = float(val)
        out[name] = _metric(
            unit, "DERIVED", src, (scalar_value, scalar_value), value=val
        )
    for name, unit, interval in (
        ("launch_gap_us", "us", launch_prior),
        ("collective_latency_us", "us/hop", collective_prior),
        ("collective_call_us_xla", "us", priors.xla_all_reduce),
        ("collective_call_us_in_kernel", "us", priors.in_kernel_all_reduce),
    ):
        out[name] = _metric(
            unit,
            interval.tag,
            interval.source,
            (interval.lo_us, interval.hi_us),
            value=interval.value_us,
        )

    boundary_band = gains.boundary_band
    fusion_gain_band = gains.fusion_gain_band
    expected_gain_band = gains.expected_gain_band
    assumed_gain_high_us = round(
        gains.assumed_diff_us + gains.ar_band_extra_us, ROUND_US_DECIMALS
    )
    dispatch_low_us = round(delta_ops * launch_prior.lo_us, ROUND_US_DECIMALS)
    dispatch_high_us = round(delta_ops * launch_prior.hi_us, ROUND_US_DECIMALS)
    dispatch_value_us = round(delta_ops * launch_prior.value_us, ROUND_US_DECIMALS)
    gain_tag, all_reduce_factor, boundary_sync_factor = (
        gains.device_gain_tag,
        float(PER_OP_COLLECTIVE_CALL_FACTOR),
        float(BOUNDARY_SYNC_HOP_FACTOR),
    )
    e2e_tag: ProvenanceTag = "MEASURED" if is_e2e else "ASSUMED"
    e2e_band = (1.0, 1.0) if is_e2e else (0.0, 0.0)
    band_rows: tuple[
        tuple[str, ProvenanceTag, tuple[float, float], float | str], ...
    ] = (
        (
            "throughput_tok_s_per_chip",
            "DERIVED",
            (gains.throughput_low, gains.throughput_high),
            gains.throughput_high,
        ),
        (
            "boundary_cost_us",
            gains.boundary_tag,
            (boundary_band[1], boundary_band[2]),
            boundary_band[0],
        ),
        (
            "gain_assumed_us",
            assumed_tag,
            (gains.assumed_diff_us, assumed_gain_high_us),
            gains.assumed_diff_us,
        ),
        (
            "compiler_proof_gain_us",
            gain_tag,
            (pure_gain_low_us, pure_gain_high_us),
            pure_gain_low_us,
        ),
        (
            "compiler_proof_gain_pct",
            gain_tag,
            (proof_pct_low, proof_pct_high),
            proof_pct_low,
        ),
        (
            "achievable_gain_pct",
            "ASSUMED",
            (achievable_gain_pct, achievable_gain_pct),
            achievable_gain_pct,
        ),
        (
            "dispatch_bubble_gain_us",
            launch_prior.tag,
            (dispatch_low_us, dispatch_high_us),
            dispatch_value_us,
        ),
        (
            "fusion_gain_max",
            "DERIVED",
            (fusion_gain_band[1], fusion_gain_band[2]),
            fusion_gain_band[0],
        ),
        (
            "expected_gain_max",
            "DERIVED",
            (expected_gain_band[1], expected_gain_band[2]),
            expected_gain_band[0],
        ),
        ("gain_basis", e2e_tag, e2e_band, gains.gain_basis),
        (
            "per_op_collective_call_factor",
            "ASSUMED",
            (1.0, all_reduce_factor),
            all_reduce_factor,
        ),
        (
            "boundary_sync_hop_factor",
            "ASSUMED",
            (0.0, boundary_sync_factor),
            boundary_sync_factor,
        ),
    )
    for name, tag, band, val in band_rows:
        unit, src = PROVENANCE_METRIC_META[name]
        out[name] = _metric(unit, tag, src, band, value=val)

    head_src = (
        (
            "WorkloadSpec explicit host dispatch head (per_op; band spans "
            "[min, max] across per_op and fastest scope)"
        )
        if workload.host_head_explicit
        else "default 0.0 us host dispatch head"
    )
    per_op_host_us, fastest_host_us, per_op_band = (
        per_op_estimate.host_head_us,
        fastest.host_head_us,
        gains.per_op_band,
    )
    host_head_band = (
        _round2(min(per_op_host_us, fastest_host_us)),
        _round2(max(per_op_host_us, fastest_host_us)),
    )
    host_head_tag: ProvenanceTag = (
        "MEASURED" if workload.host_head_explicit else "ASSUMED"
    )
    extra_rows: tuple[
        tuple[str, str, ProvenanceTag, str, tuple[float, float], float], ...
    ] = (
        (
            "step_floor_us",
            "us",
            "DERIVED",
            f"roofline floor for scope={recommended.scope}",
            (recommended.step_bound_us, gains.recommended_upper_us),
            recommended.step_bound_us,
        ),
        (
            "step_coverage",
            "ratio",
            step_coverage_prior.tag,
            step_coverage_prior.source,
            step_coverage_prior.band,
            step_coverage_prior.value,
        ),
        (
            "per_op_step_us",
            "us",
            gains.per_op_tag,
            gains.per_op_source,
            (per_op_band[1], per_op_band[2]),
            per_op_band[0],
        ),
        (
            "host_head_us",
            "us",
            host_head_tag,
            head_src,
            host_head_band,
            _round2(per_op_host_us),
        ),
    )
    for name, unit, tag, src, band, val in extra_rows:
        out[name] = _metric(unit, tag, src, band, value=val)
    for eta_name, eta_val in (
        ("eta_hbm", efficiency.eta_hbm),
        ("eta_dense", efficiency.eta_dense),
        ("eta_moe_grouped", efficiency.eta_moe_grouped),
        ("eta_ici", efficiency.eta_ici),
    ):
        eta_source = f"WorkloadSpec.efficiency.{eta_name}"
        eta_lo, eta_hi = efficiency.BRACKETS[eta_name]
        out[eta_name] = _metric(
            "ratio",
            "ASSUMED",
            eta_source,
            (min(eta_lo, eta_val), max(eta_hi, eta_val)),
            value=eta_val,
        )
    measurements_data = inputs.measurements_data
    if measurements_data is not None and measurements_data.host_head_p50_us is not None:
        host_head_low_us = _round2(measurements_data.host_head_p50_us)
        host_head_high_us = _round2(
            max(
                host_head_low_us, measurements_data.host_head_p90_us or host_head_low_us
            )
        )
        measured_source = (
            f"{measurements_data.source_desc} XLA Modules pre-first-op head "
            f"(n={measurements_data.host_head_n})"
        )
        measured_tag: ProvenanceTag = (
            "MEASURED" if _is_calibrated_trace(measurements_data) else "ASSUMED"
        )
        out["calibrated_host_head_us"] = _metric(
            "us",
            measured_tag,
            measured_source,
            (host_head_low_us, host_head_high_us),
            value=host_head_low_us,
        )
    return out


def _scope_sync_boundaries(
    estimate: ScopeEstimate,
    scope_context: _ScopeContext,
    workload: WorkloadSpec,
) -> int:
    """Return total boundary/tile sync events matching roofline upper fill/drain."""
    has_any_collectives = (
        (scope_context.num_ar > 0 and scope_context.eff_tp > 1)
        or scope_context.a2a_time > 0
        or scope_context.pcp_ring_time > 0
        or scope_context.pp_stage_time > 0
        or scope_context.dcp_hops > 0
        or (
            scope_context.local_attn_layers > 0
            and workload.dcp > 1
            and (workload.dcp & (workload.dcp - 1)) == 0
            and workload.phase in FUSED_ELIGIBLE_PHASES
        )
    )
    if not has_any_collectives:
        return 0
    if (
        estimate.scope == "whole_step"
        and 0 < scope_context.single_call_rows < scope_context.active_tokens
    ):
        num_calls = ceil(scope_context.active_tokens / scope_context.single_call_rows)
        base_boundaries = max(0, estimate.ops_per_step - num_calls)
        per_call_splits = max(1, estimate.num_vmem_splits // num_calls)
        tiled_sync = (
            (per_call_splits - 1)
            * SUBLAYER_BOUNDARIES_PER_LAYER
            * scope_context.layers_local
            * num_calls
            if per_call_splits > 1
            else 0
        )
        return base_boundaries + tiled_sync
    tiled_sync = (
        (estimate.num_vmem_splits - 1)
        * SUBLAYER_BOUNDARIES_PER_LAYER
        * scope_context.layers_local
        if estimate.scope in ("whole_step", "layer_group")
        and estimate.num_vmem_splits > 1
        else 0
    )
    return max(0, estimate.ops_per_step - 1) + tiled_sync


def _collective_hop_counts(
    scope_context: _ScopeContext,
    workload: WorkloadSpec,
) -> tuple[int, int]:
    """Return `(total_hops, factor_hops)` for collective latency and factor."""
    ar_total_hops = scope_context.num_ar * scope_context.ar_hops
    ar_intra_hops = scope_context.num_ar * scope_context.ar_intra_hops
    a2a_hops = (
        A2A_COLLECTIVES_PER_LAYER * scope_context.local_moe_layers
        if scope_context.local_moe_layers > 0 and scope_context.ep_dp > 1
        else 0
    )
    a2a_factor_hops = a2a_hops if scope_context.eff_tp > 1 else 0
    if scope_context.has_dsa:
        dcp_total_hops = (
            scope_context.dcp_post_attn_hops + scope_context.dcp_query_all_gather_hops
        )
        dcp_factor_hops = dcp_total_hops if scope_context.eff_tp > 1 else 0
    elif (
        scope_context.local_attn_layers > 0
        and workload.dcp > 1
        and (workload.dcp & (workload.dcp - 1)) == 0
        and workload.phase in FUSED_ELIGIBLE_PHASES
    ):
        dcp_total_hops = (
            scope_context.local_attn_layers
            * DCP_COLLECTIVES_PER_ATTN_LAYER
            * round(log2(workload.dcp))
        )
        dcp_factor_hops = dcp_total_hops
    else:
        dcp_total_hops = 0
        dcp_factor_hops = 0
    pp_hops = 1 if workload.pp > 1 else 0
    total_hops = ar_total_hops + a2a_hops + dcp_total_hops + pp_hops
    factor_hops = ar_intra_hops + a2a_factor_hops + dcp_factor_hops
    return total_hops, factor_hops


def _build_scope_assumptions(
    inputs: _EvalInputs,
    selection: _ScopeSelection,
    gains: _GainSummary,
) -> list[dict[str, object]]:
    """Build unresolved ASSUMED inputs that can flip fastest scope choice."""
    workload, scope_context, fastest = (
        inputs.effective_workload,
        inputs.scope_context,
        selection.fastest,
    )
    challenger = (
        selection.upper_winner
        if selection.upper_winner.scope != fastest.scope
        else selection.runner_up
    )
    compare_est = challenger if challenger is not None else inputs.estimates[0]
    delta_ops = abs(compare_est.ops_per_step - fastest.ops_per_step)
    launch_prior, collective_prior = (
        inputs.priors.launch_gap,
        inputs.priors.collective_latency,
    )
    entries: list[dict[str, object]] = []
    if launch_prior.tag == "ASSUMED":
        launch_sensitivity_us = round(
            delta_ops * max(0.0, launch_prior.hi_us - launch_prior.lo_us),
            ROUND_US_DECIMALS,
        )
        entries.append(
            _make_assumption(
                "launch_gap_us",
                launch_prior.value_us,
                [launch_prior.lo_us, launch_prior.hi_us],
                launch_sensitivity_us,
            )
        )
    total_hops, factor_hops = _collective_hop_counts(scope_context, workload)
    if (
        not gains.collectives_measured
        and collective_prior.tag == "ASSUMED"
        and total_hops > 0
    ):
        collective_sensitivity_us = round(
            total_hops * max(0.0, collective_prior.hi_us - collective_prior.lo_us),
            ROUND_US_DECIMALS,
        )
        entries.append(
            _make_assumption(
                "collective_latency_us",
                collective_prior.value_us,
                [collective_prior.lo_us, collective_prior.hi_us],
                collective_sensitivity_us,
            )
        )
    if (
        not scope_context.attn_is_bf16
        and workload.phase in ("prefill", "mixed")
        and inputs.chip_spec.native_ocp_fp8
        and scope_context.flops_kv > 0
    ):
        attn_sensitivity_us = max(
            0.0,
            (
                scope_context.flops_kv / scope_context.bf16_peak
                - scope_context.flops_kv / scope_context.peak_w
            )
            * MICROSECONDS_PER_SECOND,
        )
        bf16_tflops, peak_tflops = (
            round(scope_context.bf16_peak / TFLOPS_DIVISOR, 1),
            round(scope_context.peak_w / TFLOPS_DIVISOR, 1),
        )
        entries.append(
            _make_assumption(
                "prefill_attn_dot_dtype",
                bf16_tflops,
                [bf16_tflops, peak_tflops],
                round(attn_sensitivity_us, ROUND_US_DECIMALS),
            )
        )
    boundary_sync_factor = float(BOUNDARY_SYNC_HOP_FACTOR)
    sync_delta = abs(
        _scope_sync_boundaries(compare_est, scope_context, workload)
        - _scope_sync_boundaries(fastest, scope_context, workload)
    )
    sync_sensitivity_us = round(
        sync_delta * boundary_sync_factor * collective_prior.value_us, ROUND_US_DECIMALS
    )
    entries.append(
        _make_assumption(
            "boundary_sync_hop_factor",
            boundary_sync_factor,
            [0.0, boundary_sync_factor],
            sync_sensitivity_us,
        )
    )
    all_reduce_factor = float(PER_OP_COLLECTIVE_CALL_FACTOR)
    call_factor_sensitivity_us = round(
        max(0.0, all_reduce_factor - 1.0) * factor_hops * collective_prior.hi_us,
        ROUND_US_DECIMALS,
    )
    if call_factor_sensitivity_us > 0.0:
        entries.append(
            _make_assumption(
                "per_op_collective_call_factor",
                all_reduce_factor,
                [1.0, all_reduce_factor],
                call_factor_sensitivity_us,
            )
        )
    return entries


def _build_eta_and_context_assumptions(inputs: _EvalInputs) -> list[dict[str, object]]:
    """Build utilization-efficiency, batch, context, dtype, and row assumptions."""
    workload, scope_context = inputs.effective_workload, inputs.scope_context
    efficiency = workload.efficiency
    eta_items: list[tuple[str, float]] = [
        ("eta_hbm", efficiency.eta_hbm),
        ("eta_dense", efficiency.eta_dense),
    ]
    if scope_context.local_moe_layers > 0:
        eta_items.append(("eta_moe_grouped", efficiency.eta_moe_grouped))
    if (
        scope_context.eff_tp
        * workload.ep
        * workload.dcp
        * workload.pp
        * effective_pcp(workload)
        > 1
    ):
        eta_items.append(("eta_ici", efficiency.eta_ici))
    entries: list[dict[str, object]] = []
    for eta_name, eta_val in eta_items:
        eta_lo, eta_hi = efficiency.BRACKETS[eta_name]
        entries.append(
            _make_assumption(
                eta_name,
                eta_val,
                [min(eta_lo, eta_val), max(eta_hi, eta_val)],
                0.0,
            )
        )
    bucket_top = (
        CORE_LIMITS.default_verify_bucket_top
        if workload.phase == "verify"
        else CORE_LIMITS.default_decode_bucket_top
    )
    row_oom = any(
        estimate.scope == "whole_step" and "row_limit_not_stated" in estimate.blockers
        for estimate in inputs.estimates
    )
    optional_rows: tuple[tuple[bool, str, float | str, list[float | str]], ...] = (
        (
            not workload.batch_explicit and workload.phase in FUSED_ELIGIBLE_PHASES,
            "decoding_sequences",
            workload.batch,
            [1.0, float(max(workload.batch, CORE_LIMITS.max_batch_rows))],
        ),
        (
            workload.context == 0,
            "context",
            scope_context.eff_ctx,
            [float(workload.isl), float(workload.isl + workload.osl)],
        ),
        (
            inputs.profile.dtype_source.startswith("default"),
            "weight_dtype",
            inputs.profile.weight_dtype,
            [inputs.profile.weight_dtype, "bf16"],
        ),
        (
            row_oom,
            "max_rows_per_call",
            0,
            [float(bucket_top), float(CORE_LIMITS.max_batch_rows)],
        ),
    )
    entries.extend(
        _make_assumption(name, val, band, 0.0)
        for cond, name, val, band in optional_rows
        if cond
    )
    return entries


def _build_gain_assumptions(
    inputs: _EvalInputs, gains: _GainSummary
) -> list[dict[str, object]]:
    """Build unresolved ASSUMED inputs affecting step gain and host head."""
    workload, step_coverage_prior = (
        inputs.effective_workload,
        inputs.priors.step_coverage,
    )
    entries: list[dict[str, object]] = []
    if step_coverage_prior.tag == "ASSUMED":
        coverage_sensitivity_us = round(gains.boundary_band[0], ROUND_US_DECIMALS)
        entries.append(
            _make_assumption(
                "step_coverage",
                step_coverage_prior.value,
                [0.0, 1.0],
                coverage_sensitivity_us,
            )
        )
    if gains.per_op_tag == "ASSUMED":
        prior_value, prior_low, prior_high = gains.per_op_band
        entries.append(
            _make_assumption(
                "per_op_step_us",
                prior_value,
                [prior_low, prior_high],
                round(abs(prior_high - prior_low), ROUND_US_DECIMALS),
            )
        )
    if not workload.host_head_explicit:
        measurements_data = inputs.measurements_data
        head_hi = (
            round(measurements_data.host_head_p90_us, ROUND_US_DECIMALS)
            if measurements_data is not None
            and measurements_data.host_head_p90_us is not None
            else round(gains.break_even_host_head_us, ROUND_US_DECIMALS)
        )
        host_head_sensitivity_us = round(
            gains.break_even_host_head_us, ROUND_US_DECIMALS
        )
        entries.append(
            _make_assumption(
                "host_head_us",
                0.0,
                [0.0, max(0.0, head_hi)],
                host_head_sensitivity_us,
            )
        )
    return entries


def _build_assumptions(
    inputs: _EvalInputs,
    selection: _ScopeSelection,
    gains: _GainSummary,
) -> list[dict[str, object]]:
    """Build sorted ASSUMED input table across scope, gain, and support."""
    entries = _build_scope_assumptions(inputs, selection, gains)
    entries.extend(_build_gain_assumptions(inputs, gains))
    entries.extend(_build_eta_and_context_assumptions(inputs))
    entries.sort(key=lambda item: float(str(item["sensitivity_us"])), reverse=True)
    return entries


def _resolve_flags(inputs: _EvalInputs, selection: _ScopeSelection) -> tuple[str, ...]:
    """Resolve diagnostic warning flags from HBM limits, microbatching, and dtypes."""
    scope_context = inputs.scope_context
    workload = inputs.effective_workload
    microbatched = 0 < scope_context.single_call_rows < scope_context.active_tokens
    rows_per_call = min(
        scope_context.active_tokens,
        scope_context.single_call_rows or scope_context.active_tokens,
    )
    summed_calls = (
        ceil(scope_context.active_tokens / scope_context.single_call_rows)
        if selection.fastest.scope == "whole_step" and microbatched
        else 1
    )
    whole_step_ok = any(
        estimate.scope == "whole_step" and estimate.feasible
        for estimate in inputs.estimates
    )
    candidates = (
        (selection.fastest.num_vmem_splits > summed_calls, "multi_call_unmodeled"),
        (microbatched and whole_step_ok, "microbatch_rerun"),
        (
            whole_step_ok and exceeds_default_buckets(workload, rows_per_call),
            "above_default_buckets",
        ),
        (selection.hbm_pool_overflow, "hbm_pool_overflow"),
        (
            inputs.profile.dtype_source.startswith("default"),
            "unquantized_default_dtype",
        ),
        (True, "core_limits_pinned"),
    )
    return tuple(flag for condition, flag in candidates if condition)


def _slower_baseline_clause(
    excluded: ScopeEstimate,
    modeled_us: float,
    measured_us: float | None,
) -> str:
    """Describe which per_op baseline caused `excluded` to be filtered out."""
    lost_modeled = excluded.step_achievable_us > modeled_us
    lost_measured = (
        measured_us is not None and excluded.step_achievable_us > measured_us
    )
    if lost_modeled and lost_measured and measured_us is not None:
        baseline = (
            f"modeled per_op estimate ({modeled_us:.1f} us) and "
            f"measured per_op step ({measured_us:.1f} us)"
        )
    elif lost_measured and measured_us is not None:
        baseline = f"measured per_op step ({measured_us:.1f} us)"
    else:
        baseline = f"modeled per_op estimate ({modeled_us:.1f} us)"
    return (
        f"; {excluded.scope} skipped despite lower floor "
        f"({excluded.step_bound_us:.1f} us) because estimate "
        f"({excluded.step_achievable_us:.1f} us) is slower than {baseline}"
    )


def _format_advise_rationale(
    inputs: _EvalInputs,
    selection: _ScopeSelection,
    ridge_tokens: int,
) -> str:
    """Format concise rationale comparing best scope, runner-up, and upper flip."""
    if selection.divisibility_reasons:
        reasons = ", ".join(selection.divisibility_reasons)
        return f"Invalid mesh configuration: {reasons}"
    used_gb = inputs.scope_context.hbm_used / BYTES_PER_GB
    cap_gb = inputs.chip_spec.hbm_capacity_bytes / BYTES_PER_GB
    if not selection.hbm_fits:
        return f"Model+cache ({used_gb:.2f} GB) > HBM ({cap_gb:.2f} GB)"
    fastest = selection.fastest
    parts = [f"{fastest.scope} ({fastest.step_bound_us:.1f} us)"]
    runner_up = selection.runner_up
    if runner_up is not None:
        delta_us = runner_up.step_bound_us - fastest.step_bound_us
        pct = PERCENT_SCALE * delta_us / max(TIME_DENOM_FLOOR_S, fastest.step_bound_us)
        parts.append(
            f" vs runner-up {runner_up.scope} ({runner_up.step_bound_us:.1f} us, "
            f"+{delta_us:.1f} us, {pct:.1f}%)"
        )
        if fastest.scope != "per_op" and runner_up.scope != "per_op":
            parts.append(f"; per_op {inputs.estimates[0].step_bound_us:.1f} us")
    if fastest.scope == "per_op":
        parts.append(f"; compute-bound at T={ridge_tokens}")
    if selection.upper_winner.scope != fastest.scope:
        upper = selection.upper_winner
        parts.append(
            f"; {upper.scope} wins at upper bound ({upper.step_upper_us:.1f} us)"
        )
    if selection.filtered_lower_floor is not None:
        parts.append(
            _slower_baseline_clause(
                selection.filtered_lower_floor,
                inputs.estimates[0].step_achievable_us,
                selection.measured_per_op_us,
            )
        )
    if fastest.support == "outside_core_limits":
        blocker_text = ", ".join(fastest.blockers) or "none"
        parts.append(
            f"; core_support={fastest.support} "
            f"(core_ready_scope={selection.core_ready.scope}; "
            f"blockers={blocker_text})"
        )
    return "".join(parts)


def _apply_saturation_comparison(
    inputs: _EvalInputs,
    outcome: _SaturationOutcome,
) -> _SaturationOutcome:
    """Append throughput rationale and decode saturation layout comparison."""
    verdict_str = outcome.verdict
    fastest_verdict = outcome.fastest_verdict
    flags = outcome.flags
    rationale = outcome.rationale
    workload = inputs.effective_workload
    preset = DEPLOYMENT_PRESETS.get(inputs.profile.name)
    if preset is None or preset.saturation is None or workload.phase != "decode":
        return outcome
    current_layout = ParallelLayout(
        tp=workload.tp,
        ep=workload.ep,
        dp=workload.dp,
        pp=workload.pp,
        dcp=workload.dcp,
        pcp=workload.pcp,
        layer_groups=workload.layer_groups,
    )
    if current_layout.label() == preset.saturation.label():
        return outcome
    comparison = compare_parallelism_candidates(
        inputs.profile,
        (current_layout, preset.saturation),
        chip=inputs.chip_spec,
        base_workload=replace(
            workload,
            batch=max(1, workload.batch // max(1, workload.dp)),
            context=inputs.scope_context.eff_ctx,
        ),
        measurements=inputs.measurements_data,
    )
    comp_verdict = str(comparison["verdict"])
    if workload.objective == "both" and comp_verdict.startswith("tie(objective_split:"):
        flags = (*flags, "objective_split")
        fastest_verdict = comp_verdict
        verdict_str = comp_verdict
        rationale += f"; {comp_verdict}"
    elif workload.objective == "both" and comp_verdict == "not comparable":
        flags = (*flags, "layout_not_comparable")
        candidates = comparison["candidates"]
        if isinstance(candidates, list) and len(candidates) == PAIR_COUNT:
            cur_row, preset_row = candidates
            best_layout = str(comparison["best_throughput_layout"])
            best_suffix = (
                f"; best_throughput_layout={best_layout}" if best_layout else ""
            )
            rationale += (
                f"; step not comparable: {cur_row['layout']} uses "
                f"{cur_row['chips']} chips, {preset_row['layout']} uses "
                f"{preset_row['chips']} chips{best_suffix}"
            )
    elif workload.objective == "throughput":
        best_layout = str(comparison["best_throughput_layout"])
        if best_layout:
            rationale += f"; best_throughput_layout={best_layout}"
    return _SaturationOutcome(
        verdict=verdict_str,
        fastest_verdict=fastest_verdict,
        flags=flags,
        rationale=rationale,
    )


def _apply_throughput_and_divisibility(
    inputs: _EvalInputs,
    selection: _ScopeSelection,
    gains: _GainSummary,
    ridge_tokens: int,
) -> _VerdictBundle:
    """Apply mesh divisibility check, endpoint tie check, and objective splits."""
    metrics = _build_provenance_metrics(inputs, selection, ridge_tokens, gains)
    unresolved = _build_assumptions(inputs, selection, gains)
    flags = _resolve_flags(inputs, selection)
    measurements_data = inputs.measurements_data
    is_cal = _is_calibrated_trace(measurements_data)
    has_e2e = (
        measurements_data is not None
        and is_cal
        and measurements_data.step_time_us is not None
    )
    confidence = (
        "validated_end_to_end"
        if gains.per_op_tag == "MEASURED" or has_e2e
        else ("bounded" if is_cal else "prior_only")
    )
    rationale = _format_advise_rationale(inputs, selection, ridge_tokens)

    divisibility_reasons = selection.divisibility_reasons
    if divisibility_reasons:
        invalid_verdict = f"invalid_config({','.join(divisibility_reasons)})"
        return _VerdictBundle(
            invalid_verdict,
            invalid_verdict,
            confidence,
            "per_op",
            inputs.estimates[0],
            "per_op",
            (*flags, *divisibility_reasons),
            metrics,
            tuple(unresolved),
            rationale,
            divisibility_reasons,
        )

    fastest_scope: FusionScope = (
        selection.fastest.scope if selection.hbm_fits else "per_op"
    )
    recommended_scope: FusionScope = fastest_scope
    recommended = selection.fastest if selection.hbm_fits else inputs.estimates[0]
    core_ready_scope: FusionScope = (
        selection.core_ready.scope if selection.hbm_fits else "per_op"
    )
    scope_assumptions = [item for item in unresolved if item["affects"] == "scope"]
    has_upper_flip = (
        selection.hbm_fits
        and selection.fastest.feasible
        and selection.upper_winner.scope != selection.fastest.scope
        and bool(scope_assumptions)
    )
    fastest_verdict = (
        f"tie(measure: {scope_assumptions[0]['name']})"
        if has_upper_flip
        else SCOPE_VERDICT_MAP[
            fastest_scope if selection.fastest.feasible else "per_op"
        ]
    )
    verdict_str = fastest_verdict
    if inputs.effective_workload.objective in ("throughput", "both"):
        tp_val = float(metrics["throughput_tok_s_per_chip"]["value"])
        rationale = f"{rationale}; throughput={tp_val:.1f} tok/s/chip"
        outcome = _apply_saturation_comparison(
            inputs,
            _SaturationOutcome(
                verdict=verdict_str,
                fastest_verdict=fastest_verdict,
                flags=flags,
                rationale=rationale,
            ),
        )
        verdict_str = outcome.verdict
        fastest_verdict = outcome.fastest_verdict
        flags = outcome.flags
        rationale = outcome.rationale
    return _VerdictBundle(
        verdict_str,
        fastest_verdict,
        confidence,
        recommended_scope,
        recommended,
        core_ready_scope,
        flags,
        metrics,
        tuple(unresolved),
        rationale,
        (),
    )


def _recommended_rank_key(report: AdvisorReport) -> tuple[float, float]:
    """Return `(step_bound_us, step_achievable_us)` of the recommended scope."""
    for scope_est in report.scopes:
        if scope_est.scope == report.recommended_scope:
            return (scope_est.step_bound_us, scope_est.step_achievable_us)
    return (report.scopes[0].step_bound_us, report.scopes[0].step_achievable_us)


def _advise_single(
    profile_obj: ModelProfile,
    workload: WorkloadSpec,
    chip_spec: ChipSpec,
    *,
    plan_error: str = "",
    measurements_data: MeasurementsData | None = None,
) -> AdvisorReport:
    """Evaluate fusion scopes and assemble one report for a single workload."""
    evaluated = _evaluate_scopes_with_context(
        profile_obj,
        workload,
        chip_spec,
        plan_error=plan_error,
        measurements=measurements_data,
    )
    inputs = _EvalInputs(
        profile=profile_obj,
        effective_workload=evaluated.effective_workload,
        eval_workload=evaluated.eval_workload,
        chip_spec=chip_spec,
        scope_context=evaluated.scope_context,
        measurements_data=measurements_data,
        priors=_build_calibrated_priors(evaluated, chip_spec, measurements_data),
        estimates=evaluated.estimates,
        raw_step_map=evaluated.raw_step_map,
    )
    return _assemble_advisor_report(inputs)


def advise(
    profile: ModelProfile | str | Mapping[str, object],
    workload: WorkloadSpec,
    chip: ChipSpec | str = "tpu7x",
    plan_error: str = "",
    measurements: MeasurementsData | str | Path | Mapping[str, object] | None = None,
) -> AdvisorReport:
    """Evaluate fusion scopes from first-principles hardware and model bounds.

    Returns:
        Completed `AdvisorReport` with scope bounds, provenance, and verdict.
    """
    profile_obj = resolve_model_profile(profile)
    chip_spec = resolve_chip_spec(chip)
    measurements_data = (
        load_measurements(measurements, chip_name=chip_spec.name)
        if measurements is not None
        else None
    )
    first = _advise_single(
        profile_obj,
        workload,
        chip_spec,
        plan_error=plan_error,
        measurements_data=measurements_data,
    )
    has_routed_experts = profile_obj.num_moe_layers > 0 and profile_obj.num_experts > 0
    if (
        not workload.moe_mode_explicit
        and has_routed_experts
        and workload.moe_mode in SEARCHED_MOE_MODES
    ):
        other_mode: MoEMode = (
            "grouped_on_chip"
            if workload.moe_mode == "masked_on_chip"
            else "masked_on_chip"
        )
        second = _advise_single(
            profile_obj,
            replace(workload, moe_mode=other_mode),
            chip_spec,
            plan_error=plan_error,
            measurements_data=measurements_data,
        )
        if _recommended_rank_key(second) < _recommended_rank_key(first):
            return second
    return first


def _assemble_advisor_report(inputs: _EvalInputs) -> AdvisorReport:
    """Assemble final `AdvisorReport` from evaluated scope bounds and priors."""
    per_op_baseline = _resolve_per_op_baseline(inputs)
    measured_per_op_us = (
        per_op_baseline.band[0] if per_op_baseline.tag == "MEASURED" else None
    )
    selection = _select_scopes(inputs, measured_per_op_us=measured_per_op_us)
    scope_context = inputs.scope_context
    fastest = selection.fastest
    ridge_flop_per_byte = scope_context.peak_w / scope_context.bw_hbm
    weight_and_act_bytes = (
        fastest.step_weight_gb * BYTES_PER_GB + fastest.act_hbm_mb * BYTES_PER_MB
    )
    state_bytes = (
        scope_context.kv_cache_read_bytes + scope_context.recurrent_state_rw_bytes
    )
    total_flops = (
        scope_context.flops_dense + fastest.moe_flop_factor * scope_context.flops_moe
    )
    step_intensity = round(
        total_flops / max(1.0, weight_and_act_bytes + state_bytes), ROUND_US_DECIMALS
    )
    asymptotic_intensity = _compute_asymptotic_intensity(
        inputs.effective_workload, scope_context, state_bytes
    )
    flop_per_token = max(
        1.0,
        (scope_context.flops_dense + scope_context.flops_moe)
        / max(1, scope_context.active_tokens),
    )
    ridge_tokens = max(
        1,
        ceil(
            inputs.estimates[0].step_weight_gb
            * BYTES_PER_GB
            * ridge_flop_per_byte
            / flop_per_token
        ),
    )
    recommended_moe_mode = _select_recommended_moe_mode(inputs, fastest)

    gains = _compute_gains_and_throughput(inputs, selection, per_op_baseline)
    bundle = _apply_throughput_and_divisibility(inputs, selection, gains, ridge_tokens)
    fastest_scope: FusionScope = (
        fastest.scope
        if (selection.hbm_fits and not bundle.divisibility_reasons)
        else "per_op"
    )
    core_support = bundle.recommended.support
    core_blockers = bundle.recommended.blockers
    core_ready = core_support == "within_core_limits"
    return AdvisorReport(
        model=inputs.profile.name,
        chip=inputs.chip_spec.name,
        phase=inputs.effective_workload.phase,
        active_tokens=scope_context.active_tokens,
        effective_context=scope_context.eff_ctx,
        vmem_budget_mib=round(
            scope_context.vmem_budget / (1 << MIB_BIT_SHIFT), ROUND_GB_DECIMALS
        ),
        hbm_fits=selection.hbm_fits,
        hbm_pool_overflow=selection.hbm_pool_overflow,
        hbm_utilization_cap=inputs.effective_workload.hbm_utilization_cap,
        hbm_used_gb=round(scope_context.hbm_used / BYTES_PER_GB, ROUND_GB_DECIMALS),
        hbm_capacity_gb=round(
            inputs.chip_spec.hbm_capacity_bytes / BYTES_PER_GB, ROUND_GB_DECIMALS
        ),
        ridge_dense_flop_per_byte=round(ridge_flop_per_byte, ROUND_US_DECIMALS),
        step_arithmetic_intensity=step_intensity,
        asymptotic_decode_intensity=asymptotic_intensity,
        recommended_scope=bundle.recommended_scope,
        build_megakernel=(
            selection.hbm_fits
            and (not bundle.divisibility_reasons)
            and bundle.verdict == "megakernel_likely"
        ),
        rationale=bundle.rationale,
        scopes=inputs.estimates,
        per_token_vmem_bytes=scope_context.per_token_fused_bytes,
        ridge_tokens=ridge_tokens,
        recommended_moe_mode=recommended_moe_mode,
        dma_buffers=inputs.effective_workload.dma_buffers,
        verdict=bundle.verdict,
        confidence=bundle.confidence,
        flags=bundle.flags,
        metrics=bundle.metrics,
        unresolved_assumptions=bundle.unresolved_assumptions,
        fastest_scope=fastest_scope,
        fastest_verdict=bundle.fastest_verdict,
        evidence=core_support,
        gain_basis=gains.gain_basis,
        coverage=DECODER_COVERAGE_STATEMENT,
        ranking_metric=bundle.recommended.ranking_metric or "step_bound_us",
        binding_unit=bundle.recommended.binding_unit or "hbm",
        core_support=core_support,
        core_blockers=core_blockers,
        core_ready=core_ready,
        core_ready_scope=bundle.core_ready_scope,
    )


def _evaluate_layout_row(
    profile_obj: ModelProfile,
    chip_spec: ChipSpec,
    layout: ParallelLayout,
    template: WorkloadSpec,
    measurements_data: MeasurementsData | None,
) -> ParallelismComparisonRow:
    """Evaluate one `ParallelLayout` for `compare_parallelism_candidates`."""
    per_replica_batch = max(1, template.batch)
    workload = replace(
        template,
        batch=per_replica_batch * layout.dp,
        isl=template.isl or template.context or WorkloadSpec().isl,
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        layer_groups=layout.layer_groups,
        objective="step",
    )
    report = advise(profile_obj, workload, chip_spec, measurements=measurements_data)
    valid = not _mesh_divisibility_reasons(profile_obj, workload, chip_spec)
    eff_tp = effective_tp(workload)
    eff_pcp = effective_pcp(workload)
    mesh_cores = eff_tp * layout.dp * layout.pp * eff_pcp
    return {
        "layout": layout.label(),
        "chips": max(1, ceil(mesh_cores / chip_spec.devices_per_chip)),
        "step_floor_us": float(report.metrics["step_floor_us"]["value"]),
        "throughput_tok_s_per_chip": float(
            report.metrics["throughput_tok_s_per_chip"]["value"]
        ),
        "recommended_scope": report.recommended_scope,
        "fastest_scope": report.fastest_scope,
        "hbm_fits": report.hbm_fits,
        "valid": valid,
    }


def _comparison_dict(
    profile_name: str,
    chip_name: str,
    objective: str,
    summary: tuple[str, str, str],
    candidates: list[ParallelismComparisonRow],
) -> dict[str, object]:
    """Build uniform return dictionary for `compare_parallelism_candidates`."""
    step_label, tp_label, verdict_str = summary
    return {
        "model": profile_name,
        "chip": chip_name,
        "objective": objective,
        "best_step_layout": step_label,
        "best_throughput_layout": tp_label,
        "verdict": verdict_str,
        "candidates": candidates,
    }


def compare_parallelism_candidates(
    profile: ModelProfile | str | Mapping[str, object],
    layouts: Sequence[ParallelLayout],
    *,
    chip: ChipSpec | str = "tpu7x",
    base_workload: WorkloadSpec | None = None,
    measurements: MeasurementsData | str | Path | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compare candidate `ParallelLayout` meshes on step latency and throughput.

    Returns:
        Dictionary summarizing per-layout rows and winning objective labels.
    """
    profile_obj, chip_spec = resolve_model_profile(profile), resolve_chip_spec(chip)
    measurements_data = (
        load_measurements(measurements, chip_name=chip_spec.name)
        if measurements is not None
        else None
    )
    template = base_workload or WorkloadSpec(
        phase="decode", batch=16, context=4096, objective="both"
    )
    objective = template.objective
    if not layouts:
        return _comparison_dict(
            profile_obj.name,
            chip_spec.name,
            objective,
            ("", "", "no_layouts_provided"),
            [],
        )
    rows = [
        _evaluate_layout_row(
            profile_obj, chip_spec, layout, template, measurements_data
        )
        for layout in layouts
    ]
    feasible = [row for row in rows if row["hbm_fits"] and row["valid"]]
    if not feasible:
        verdict = (
            "no_valid_feasible_layout"
            if any(row["hbm_fits"] for row in rows)
            else "no_feasible_layout_hbm_oom"
        )
        return _comparison_dict(
            profile_obj.name, chip_spec.name, objective, ("", "", verdict), rows
        )
    best_step = min(feasible, key=lambda row: row["step_floor_us"])
    best_tp = max(feasible, key=lambda row: row["throughput_tok_s_per_chip"])
    equal_chips = len({row["chips"] for row in feasible}) == 1
    step_label, tp_label = (
        (best_step["layout"] if equal_chips else ""),
        best_tp["layout"],
    )
    if objective == "throughput":
        verdict_str = tp_label
    elif not equal_chips:
        verdict_str = "not comparable"
    elif objective == "step" or step_label == tp_label:
        verdict_str = step_label
    else:
        verdict_str = f"tie(objective_split: step={step_label}, throughput={tp_label})"
    return _comparison_dict(
        profile_obj.name,
        chip_spec.name,
        objective,
        (step_label, tp_label, verdict_str),
        rows,
    )


def _format_partition_runs(groups: tuple[int, ...]) -> str:
    """Compress contiguous equal group sizes as `4 + 8x11 + 1`."""
    if not groups:
        return "none"
    return " + ".join(
        f"{size}x{count}" if count > 1 else str(size)
        for size, group in groupby(groups)
        for count in (sum(1 for _ in group),)
    )


def _candidate_layer_partitions(
    model_name: str,
    num_layers: int,
    workload: WorkloadSpec,
) -> tuple[tuple[int, ...], list[tuple[str, tuple[int, ...], bool]]]:
    """Return `(listed_groups, [(label, groups, is_listed), ...])` for `num_layers`."""
    if workload.layer_groups:
        listed_groups = workload.layer_groups
    else:
        preset = DEPLOYMENT_PRESETS.get(model_name.lower())
        if preset is None:
            listed_groups = ()
        elif workload.phase == "prefill":
            listed_groups = preset.prefill.layer_groups
        elif workload.dp > 1 and preset.saturation is not None:
            listed_groups = preset.saturation.layer_groups
        else:
            listed_groups = preset.decode.layer_groups

    partitions: list[tuple[str, tuple[int, ...], bool]] = []
    seen: set[tuple[int, ...]] = set()

    def _add(label_prefix: str, groups: tuple[int, ...]) -> None:
        """Append unique layer partition labeled with its run-length summary."""
        if not groups or groups in seen:
            return
        seen.add(groups)
        partitions.append(
            (
                f"{label_prefix} ({_format_partition_runs(groups)})",
                groups,
                groups == listed_groups,
            )
        )

    for count in EQUAL_SPLIT_GROUP_COUNTS:
        if count <= num_layers:
            base_size, rem = divmod(num_layers, count)
            groups = tuple(base_size + (1 if idx < rem else 0) for idx in range(count))
            _add(f"{count} {'group' if count == 1 else 'groups'}", groups)
    if sum(listed_groups) == num_layers:
        _add(f"listed {len(listed_groups)} groups", listed_groups)
    _add("1 per layer", (1,) * num_layers)
    return listed_groups, partitions


def _pct_gap(ref: float, candidate: float) -> float:
    """Compute rounded relative percentage savings vs `ref`."""
    return (
        0.0
        if ref <= 0.0
        else round(PERCENT_SCALE * (ref - candidate) / ref, ROUND_US_DECIMALS)
    )


def search_layer_groups(
    profile: ModelProfile | str | Mapping[str, object],
    workload: WorkloadSpec,
    chip: ChipSpec | str = "tpu7x",
    *,
    measurements: MeasurementsData | Mapping[str, object] | Path | str | None = None,
) -> LayerGroupSearchReport:
    """Price `layer_group` across equal splits, 1-per-layer, and listed partitions.

    Returns:
        `LayerGroupSearchReport` comparing candidate layer-group schedules.
    """
    model_profile, chip_spec = resolve_model_profile(profile), resolve_chip_spec(chip)
    measurements_data = (
        load_measurements(measurements, chip_name=chip_spec.name)
        if measurements is not None
        else None
    )
    effective_workload = workload
    if (
        measurements_data is not None
        and measurements_data.ring_plan is not None
        and effective_workload.ring_plan is None
    ):
        effective_workload = replace(
            effective_workload, ring_plan=measurements_data.ring_plan
        )
    validate_advise_inputs(model_profile, effective_workload, chip_spec)
    mesh_valid = not _mesh_divisibility_reasons(
        model_profile, effective_workload, chip_spec
    )
    listed_groups, partitions = _candidate_layer_partitions(
        model_profile.name, model_profile.num_layers, effective_workload
    )
    base_eval = _evaluate_scopes_with_context(
        model_profile,
        replace(effective_workload, layer_groups=(), objective="step"),
        chip_spec,
        measurements=measurements_data,
    )
    per_op_estimate = next(
        estimate for estimate in base_eval.estimates if estimate.scope == "per_op"
    )
    layout_label = ParallelLayout(
        tp=effective_workload.tp,
        ep=effective_workload.ep,
        dp=effective_workload.dp,
        pp=effective_workload.pp,
        dcp=effective_workload.dcp,
        pcp=effective_workload.pcp,
    ).label()
    candidates: list[LayerGroupCandidate] = []
    for label, groups, is_listed in partitions:
        cand_eval = _evaluate_scopes_with_context(
            model_profile,
            replace(effective_workload, layer_groups=groups, objective="step"),
            chip_spec,
            measurements=measurements_data,
        )
        layer_group_estimate = next(
            estimate
            for estimate in cand_eval.estimates
            if estimate.scope == "layer_group"
        )
        cand_feasible = (
            mesh_valid
            and (cand_eval.scope_context.hbm_used <= chip_spec.hbm_capacity_bytes)
            and layer_group_estimate.feasible
        )
        candidates.append(
            LayerGroupCandidate(
                label=label,
                layer_groups=groups,
                num_groups=len(groups),
                feasible=cand_feasible,
                max_vmem_tokens=layer_group_estimate.max_vmem_tokens,
                num_vmem_splits=layer_group_estimate.num_vmem_splits,
                step_bound_us=layer_group_estimate.step_bound_us,
                step_achievable_us=layer_group_estimate.step_achievable_us,
                step_upper_us=layer_group_estimate.step_upper_us,
                is_listed=is_listed,
            )
        )
    ordered = tuple(
        sorted(candidates, key=lambda item: (item.num_groups, item.step_bound_us))
    )
    feasible = [item for item in ordered if item.feasible]
    best = (
        min(feasible, key=lambda item: (item.step_bound_us, item.num_groups))
        if feasible
        else None
    )
    listed_match = next((item for item in ordered if item.is_listed), None)
    listed_bound = (
        listed_match.step_bound_us
        if (listed_match is not None and listed_match.feasible)
        else None
    )
    has_listed_comparison = best is not None and listed_bound is not None
    return LayerGroupSearchReport(
        model=model_profile.name,
        chip=chip_spec.name,
        phase=workload.phase,
        layout=layout_label,
        active_tokens=base_eval.scope_context.active_tokens,
        per_op_step_bound_us=per_op_estimate.step_bound_us,
        per_op_step_achievable_us=per_op_estimate.step_achievable_us,
        per_op_step_upper_us=per_op_estimate.step_upper_us,
        listed_layer_groups=listed_groups if listed_match is not None else (),
        listed_step_bound_us=listed_bound,
        best=best,
        gap_vs_per_op_us=_round2(per_op_estimate.step_bound_us - best.step_bound_us)
        if best is not None
        else 0.0,
        gap_vs_per_op_pct=_pct_gap(per_op_estimate.step_bound_us, best.step_bound_us)
        if best is not None
        else 0.0,
        gap_vs_listed_us=_round2(listed_bound - best.step_bound_us)
        if has_listed_comparison and best is not None and listed_bound is not None
        else None,
        gap_vs_listed_pct=_pct_gap(listed_bound, best.step_bound_us)
        if has_listed_comparison and best is not None and listed_bound is not None
        else None,
        candidates=ordered,
        note=LAYER_GROUP_PRICING_NOTE,
    )
