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
"""Plain-text terminal formatters for advisor verdicts, sweeps, and searches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import chips
from .constants import (
    DECODE_MED_REF_INDEX,
    GAIN_BASIS_CALIBRATED_E2E,
    MIN_PROVENANCE_BAND_LEN,
    MOE_MODE_SHORT_HINTS,
    SCOPE_HINTS,
)
from .specs import as_scalar_number

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .specs import (
        AdvisorReport,
        AdvisorSweepReport,
        LayerGroupSearchReport,
        RegimeEntry,
        ScopeEstimate,
    )

__all__ = [
    "format_advisor_summary",
    "format_layer_group_search",
    "format_regime_table",
    "format_ring_slots",
    "gain_basis_label",
    "recommended_scope_dict",
    "vector_register_load_cap_line",
]


MAX_TABLE_BLOCKERS = 2


def _format_table_blockers(core_blockers: tuple[str, ...]) -> str:
    """Format core blockers compactly for table cells, preserving overflow count."""
    if not core_blockers:
        return "-"
    if len(core_blockers) <= MAX_TABLE_BLOCKERS:
        return ", ".join(core_blockers)
    shown = ", ".join(core_blockers[:MAX_TABLE_BLOCKERS])
    return f"{shown} (+{len(core_blockers) - MAX_TABLE_BLOCKERS})"


def _mapping_float(
    mapping: Mapping[str, object], key: str, default: float = 0.0
) -> float:
    """Extract a finite numeric scalar from `mapping[key]` or return `default`."""
    num = as_scalar_number(mapping.get(key))
    return default if num is None else num


def format_ring_slots(dense_slots: int, expert_slots: int) -> str:
    """Format `(dense_slots,expert_slots)` pair, or `"-"` when unbuffered."""
    return f"({dense_slots},{expert_slots})" if dense_slots > 0 else "-"


def gain_basis_label(gain_basis: str) -> str:
    """Return human-readable label for a gain-basis enum string."""
    return (
        "calibrated whole-server E2E"
        if gain_basis == GAIN_BASIS_CALIBRATED_E2E
        else "fused-scope device estimate"
    )


def recommended_scope_dict(regime_entry: RegimeEntry) -> Mapping[str, object]:
    """Return serialized scope dict matching `regime_entry.recommended_scope`."""
    return _scope_dict_by_name(regime_entry, regime_entry.recommended_scope)


def _format_ring_pair(scope_dict: Mapping[str, object]) -> str:
    """Format `(dense_slots, expert_slots)` ring depths when present."""
    dense = int(_mapping_float(scope_dict, "selected_dense_slots"))
    expert = int(_mapping_float(scope_dict, "selected_expert_slots"))
    return format_ring_slots(dense, expert)


def metric_float(
    metrics: Mapping[str, object],
    key: str,
    default: float = 0.0,
) -> float:
    """Extract a numeric value from a provenance metric mapping."""
    raw = metrics.get(key)
    if type(raw) is dict and "value" in raw:
        return _mapping_float(raw, "value", default)
    return default


def _gain_band_text(metric: Mapping[str, object] | None) -> str:
    """Return a gain metric provenance tag and band for single-line output."""
    if metric is None:
        return "no band"
    band, tag = metric.get("band"), str(metric.get("tag", "DERIVED"))
    if type(band) is not list or len(band) < MIN_PROVENANCE_BAND_LEN:
        return tag
    band_low, band_high = as_scalar_number(band[0]), as_scalar_number(band[-1])
    if band_low is None or band_high is None:
        return tag
    return f"{tag}, band {band_low:+.1f} to {band_high:+.1f} us"


def _format_unresolved_lines(assumptions: tuple[dict[str, object], ...]) -> list[str]:
    """Render `(metric, assumed, affects, settle)` lines for unresolved priors."""
    lines: list[str] = []
    for item in assumptions:
        band, assumed_value = item.get("band", []), item.get("value")
        assumed = (
            f"{assumed_value} [band={band}]"
            if assumed_value is not None
            else f"band={band}"
        )
        lines.append(
            f"  * {item['name']}: assumed={assumed} ({item['unit']}) | "
            f"affects={item.get('affects', 'gain')} | settle: {item['how_to_settle']}"
        )
    return lines


def _format_hbm_status_line(report: AdvisorReport) -> str:
    """Format physical OOM and pool-headroom status for an `AdvisorReport`."""
    if not report.hbm_fits:
        note = "physical OOM (exceeds 1.00x HBM capacity)"
    elif report.hbm_pool_overflow:
        cap_pct = round(report.hbm_utilization_cap * 100.0)
        note = f"hbm_pool_overflow (fits 1.00x HBM; exceeds {cap_pct}% pool cap)"
    else:
        note = "fits physical HBM and pool cap"
    return (
        f"HBM: {report.hbm_used_gb:.2f} / {report.hbm_capacity_gb:.1f} GB/core "
        f"(hbm_fits={report.hbm_fits}, hbm_pool_overflow={report.hbm_pool_overflow}; "
        f"{note})"
    )


def _format_gain_summary_lines(report: AdvisorReport) -> list[str]:
    """Format physics fusion gain, dispatch bubble gain, and break-even lines."""
    lines: list[str] = []
    physics_gain_metric = report.metrics.get("compiler_proof_gain_us")
    dispatch_gain_metric = report.metrics.get("dispatch_bubble_gain_us")
    if physics_gain_metric is not None and dispatch_gain_metric is not None:
        calibrated = report.gain_basis == GAIN_BASIS_CALIBRATED_E2E
        expected_gain_pct = metric_float(report.metrics, "expected_gain_max") * 100.0
        prefix = (
            f"Whole-server E2E gain (calibrated): {expected_gain_pct:+.2f}% | "
            if calibrated
            else ""
        )
        basis = gain_basis_label(report.gain_basis)
        lines.append(
            f"{prefix}Fusion gain (estimate): "
            f"{_mapping_float(physics_gain_metric, 'value'):+.2f} us "
            f"[{_gain_band_text(physics_gain_metric)}] | Dispatch Bubble Gain: "
            f"{_mapping_float(dispatch_gain_metric, 'value'):+.2f} us "
            f"[basis: {basis}]"
        )
    break_even_us = metric_float(report.metrics, "break_even_host_head_delta_us")
    if report.recommended_scope == "per_op" and break_even_us > 0.0:
        lines.append(
            f"Break-even host-head delta: +{break_even_us:.1f} us "
            "(per_op - fused host dispatch head needed for fusion to tie)"
        )
    return lines


def _format_scope_line(scope_est: ScopeEstimate) -> str:
    """Format one evaluated `ScopeEstimate` row for terminal output."""
    upper_us = max(scope_est.step_bound_us, scope_est.step_upper_us)
    critical_weight_clause = (
        f", crit_weights={scope_est.critical_weight_gb:.2f} GB"
        if scope_est.critical_weight_gb is not None
        else ""
    )
    ring = f"({scope_est.selected_dense_slots},{scope_est.selected_expert_slots})"
    return (
        f"  - {scope_est.scope:12s} "
        f"[floor {scope_est.step_bound_us:7.1f} .. {upper_us:7.1f} us] "
        f"(Derated Achievable Step={scope_est.step_achievable_us:9.1f} us) "
        f"[rows/pass<={scope_est.max_vmem_tokens:4d}, "
        f"passes={scope_est.num_vmem_splits:2d}, "
        f"ring={ring}, static_vmem={scope_est.vmem_static_mib:5.2f} MiB, "
        f"weights={scope_est.step_weight_gb:.2f} GB{critical_weight_clause}, "
        f"metric={scope_est.ranking_metric}, "
        f"bind={scope_est.binding_unit}]"
    )


def _format_core_blockers_and_flags_lines(report: AdvisorReport) -> list[str]:
    """Format blocker diagnostics and active warning flags for `AdvisorReport`."""
    lines: list[str] = []
    if report.core_support == "outside_core_limits" or report.core_blockers:
        blockers_text = ", ".join(report.core_blockers) or "none"
        lines.append(
            f"Blocked by: {blockers_text} "
            f"(recommended_scope={report.recommended_scope}, "
            f"core_support={report.core_support}, "
            f"core_ready_scope={report.core_ready_scope})"
        )
    if report.flags:
        lines.append(f"Flags: {', '.join(report.flags)}")
    return lines


def _format_explain_lines(report: AdvisorReport) -> list[str]:
    """Format provenance breakdown lines when `--explain` is set."""
    if not report.metrics:
        return []
    lines = ["Provenance Breakdown:"]
    for key, item in report.metrics.items():
        band = item.get("band")
        band_prefix = (
            f"band=[{band[0]}, {band[1]}], "
            if type(band) is list and len(band) >= MIN_PROVENANCE_BAND_LEN
            else ""
        )
        lines.append(
            f"  * [{item['tag']}] {key}: {item['value']} {item['unit']} "
            f"({band_prefix}source={item['source']})"
        )
    return lines


def format_advisor_summary(
    advisor_report: AdvisorReport, *, explain: bool = False, unresolved: bool = False
) -> str:
    """Format single-workload `AdvisorReport` for plain-text terminal output.

    Returns:
        Multi-line plain-text summary of recommendation, bounds, and flags.
    """
    report = advisor_report
    core_blockers_suffix = (
        f", core_blockers={', '.join(report.core_blockers)}"
        if report.core_blockers
        else ""
    )
    lines = [
        (
            f"Model: {report.model} | Chip: {report.chip} | "
            f"Phase: {report.phase} (T={report.active_tokens})"
        ),
        (
            f"Recommended: {report.recommended_scope} "
            f"[moe={report.recommended_moe_mode}] "
            f"(build={report.build_megakernel}, verdict={report.verdict}, "
            f"ranking={report.ranking_metric}, "
            f"bind={report.binding_unit})"
        ),
        (
            f"Core Readiness: {report.core_support} "
            f"(core_ready={report.core_ready}, "
            f"core_ready_scope={report.core_ready_scope}{core_blockers_suffix})"
        ),
        (
            f"Evidence: {report.evidence} (confidence={report.confidence}; "
            f"coverage: {report.coverage})"
        ),
        _format_hbm_status_line(report),
        f"Rationale: {report.rationale}",
        (
            f"VMEM Budget: {report.vmem_budget_mib:.1f} MiB "
            f"({report.dma_buffers}-slot async DMA)"
            f" | Per-Token Slope: {report.per_token_vmem_bytes / 1024.0:.1f} "
            f"KiB/tok | Ridge: {report.ridge_dense_flop_per_byte:.1f} FLOP/B "
            f"(T_ridge={report.ridge_tokens})"
        ),
        *_format_gain_summary_lines(report),
        *(_format_scope_line(scope_est) for scope_est in report.scopes),
    ]
    if explain:
        lines.extend(_format_explain_lines(report))
    if unresolved and report.unresolved_assumptions:
        lines.extend(
            [
                "Unresolved Assumptions:",
                *_format_unresolved_lines(report.unresolved_assumptions),
            ]
        )
    lines.extend(_format_core_blockers_and_flags_lines(report))
    if explain:
        lines.extend(
            [
                "Inputs:",
                (
                    f"  * [CHOSEN] phase={report.phase}, "
                    f"active_tokens={report.active_tokens}, "
                    f"effective_context={report.effective_context}"
                ),
                (
                    f"  * [FACT] chip={report.chip}, model={report.model}, "
                    f"hbm_capacity_gb={report.hbm_capacity_gb}, "
                    f"vmem_budget_mib={report.vmem_budget_mib}"
                ),
            ]
        )
    return "\n".join(lines)


def vector_register_load_cap_line(chip_spec: chips.ChipSpec) -> str:
    """Return the vector register load cap line, or empty string when unset.

    Returns:
        Formatted vector register load bandwidth summary line or `""`.
    """
    cap_bps = chip_spec.vector_load_cap_bytes_per_s
    if cap_bps <= 0.0:
        return ""
    relation = (
        "It is above the HBM rate, so it does not bind while weights stream from HBM."
        if cap_bps > chip_spec.hbm_bw_bytes_per_s
        else "It is at or below the HBM rate, so it can bind while weights stream "
        "from HBM."
    )
    return (
        f"Vector register load cap: {chip_spec.vector_loads_per_cycle} x "
        f"{chip_spec.vector_register_bytes:,} B x "
        f"{chip_spec.clock_hz / 1e9:.1f}e9 Hz = "
        f"{cap_bps / 1e12:.2f} TB/s per core. {relation}"
    )


def _scope_dict_by_name(
    regime_entry: RegimeEntry, scope_name: str
) -> Mapping[str, object]:
    """Return the serialized scope mapping matching `scope_name`."""
    for scope_dict in regime_entry.scopes:
        if scope_dict.get("scope") == scope_name:
            return scope_dict
    return regime_entry.scopes[0] if regime_entry.scopes else {}


def _format_triage_entry_lines(regime_entry: RegimeEntry) -> list[str]:
    """Format peak-floor band and derated achievable triage lines for one regime."""
    scope_dict = _scope_dict_by_name(regime_entry, regime_entry.recommended_scope)
    upper_us = max(regime_entry.step_bound_us, regime_entry.step_upper_us)
    achievable_low_us = _mapping_float(
        scope_dict, "step_achievable_us", regime_entry.step_achievable_us
    )
    achievable_high_us = _mapping_float(
        scope_dict, "step_achievable_upper_us", regime_entry.step_achievable_upper_us
    )
    per_op_achievable_us = _mapping_float(
        _scope_dict_by_name(regime_entry, "per_op"),
        "step_achievable_us",
        achievable_low_us,
    )
    recommended_gain_pct = (
        100.0 * (per_op_achievable_us - achievable_low_us) / per_op_achievable_us
        if per_op_achievable_us > 0.0
        else 0.0
    )
    physics_gain_us = metric_float(regime_entry.metrics, "compiler_proof_gain_us")
    dispatch_gain_us = metric_float(regime_entry.metrics, "dispatch_bubble_gain_us")
    physics_gain_pct = metric_float(regime_entry.metrics, "compiler_proof_gain_pct")
    core_ready_note = ""
    if regime_entry.core_ready_scope != regime_entry.recommended_scope:
        core_ready_achievable_us = _mapping_float(
            _scope_dict_by_name(regime_entry, regime_entry.core_ready_scope),
            "step_achievable_us",
            per_op_achievable_us,
        )
        core_ready_note = (
            f" | core_ready {regime_entry.core_ready_scope}: "
            f"{core_ready_achievable_us:.1f} us"
        )
    gain_band_text = _gain_band_text(regime_entry.metrics.get("compiler_proof_gain_us"))
    break_even_us = metric_float(regime_entry.metrics, "break_even_host_head_delta_us")
    break_even_suffix = (
        f" | Break-even host-head=+{break_even_us:.1f} us"
        if regime_entry.recommended_scope == "per_op" and break_even_us > 0.0
        else ""
    )
    low_gain_tag = " [LOW PHYSICS GAIN <1%]" if 0.0 < physics_gain_pct < 1.0 else ""
    oom_tag = " [PHYSICAL OOM]" if not regime_entry.hbm_fits else ""
    invalid_tag = (
        f" [{regime_entry.verdict}]"
        if regime_entry.verdict.startswith("invalid_config(")
        else ""
    )
    core_tag = (
        f" [core={regime_entry.core_support}, "
        f"core_ready={regime_entry.core_ready_scope}]"
    )
    triage_lines = [
        (
            f"  * {regime_entry.label}: Peak Floor Band="
            f"[{regime_entry.step_bound_us:.1f} .. {upper_us:.1f}] "
            f"us (bind={regime_entry.binding_unit})"
            f"{core_tag}{oom_tag}{invalid_tag}"
        ),
        (
            f"    Derated Achievable=[{achievable_low_us:.1f} .. "
            f"{achievable_high_us:.1f}] us | "
            f"Derated Achievable Step={achievable_low_us:.1f} us "
            f"({recommended_gain_pct:+.1f}% vs per_op)"
            f"{core_ready_note} | ring={_format_ring_pair(scope_dict)} | "
            f"Fusion gain (estimate)={physics_gain_us:+.1f} us "
            f"({physics_gain_pct:+.1f}%; {gain_band_text}) | "
            f"Dispatch Bubble Gain={dispatch_gain_us:+.1f} us"
            f"{break_even_suffix}{low_gain_tag}"
        ),
    ]
    if regime_entry.core_support == "outside_core_limits" or regime_entry.core_blockers:
        blockers_text = ", ".join(regime_entry.core_blockers) or "none"
        triage_lines.append(
            f"    Core Blockers: {blockers_text} "
            f"(core_ready_scope={regime_entry.core_ready_scope})"
        )
    return triage_lines


def _format_sweep_layout_lines(sweep: AdvisorSweepReport) -> list[str]:
    """Format decode, prefill, saturation, sublayer, and DCN handoff lines."""
    decode_layout = sweep.parallelism
    mode_tag = decode_layout.source or (
        "auto-recommended" if decode_layout.auto_selected else "user-specified"
    )
    lines = [
        f"Decode Parallelism: {decode_layout.label()} "
        f"[{decode_layout.topology}] ({mode_tag})"
    ]
    if (prefill := sweep.prefill_parallelism) is not None:
        layer_groups_suffix = (
            f", {len(prefill.layer_groups)} layer groups"
            if prefill.layer_groups
            else ""
        )
        lines.append(
            f"Prefill Parallelism: {prefill.label()} [{prefill.topology}] "
            f"({prefill.source}{layer_groups_suffix})"
        )
    if (saturation_layout := sweep.saturation_parallelism) is not None:
        lines.append(
            f"Decode Saturation Parallelism: {saturation_layout.label()} "
            f"[{saturation_layout.topology}] ({saturation_layout.source})"
        )
    roles = [
        f"{role} {sweep.submodule_labels[role]}"
        for role in ("decode", "prefill", "saturation")
        if role in sweep.submodule_labels
    ]
    if roles:
        lines.append(f"Sublayer meshes: {'; '.join(roles)}")
    if (handoff := sweep.disagg_handoff) is not None:
        xfer_low_ms, xfer_high_ms = handoff.dcn_xfer_bracket_ms
        lines.append(
            f"Disaggregated Prefill/Decode DCN Handoff (ISL={handoff.isl}): "
            f"{handoff.total_wire_mb:.2f} MB/req "
            f"(KV={handoff.kv_wire_mb:.2f} MB + "
            f"state={handoff.state_wire_mb:.2f} MB) -> {handoff.dcn_xfer_ms:.2f} ms "
            f"[{xfer_low_ms:.2f}..{xfer_high_ms:.2f} ms] at "
            f"{handoff.dcn_bw_gbps:.0f} Gbps/chip "
            f"[{handoff.dcn_provenance}]"
        )
    return lines


def _format_sweep_header_lines(sweep: AdvisorSweepReport) -> list[str]:
    """Format sweep HBM, VMEM ceiling, and roofline ridge header lines."""
    decode_layout, vmem_breakdown = sweep.parallelism, sweep.vmem_breakdown_bytes
    chip_spec = sweep.chip_spec or chips.resolve_chip_spec(sweep.chip)
    load_cap = vector_register_load_cap_line(chip_spec)
    vmem_limit = chip_spec.vmem_usable_bytes / (1 << 20)
    lines = [
        f"=== Megakernel Advisor Regime Report: {sweep.model} on {sweep.chip} ===",
        *_format_sweep_layout_lines(sweep),
        f"Dtypes: {sweep.model_sources.get('dtypes', '')}",
    ]
    if sweep.model_profile is not None and sweep.model_profile.dtype_source.startswith(
        "default"
    ):
        lines.append(
            "WARNING: Config has no quantization_config block; using default dtype."
        )
    lines.extend(
        [
            f"HBM Footprint: {decode_layout.hbm_used_gb:.2f} / "
            f"{decode_layout.hbm_capacity_gb:.1f} "
            f"GB/core (active weights/step: "
            f"{decode_layout.step_weight_gb:.2f} GB/core)",
            *([load_cap] if load_cap else []),
            (
                f"Ideal-kernel VMEM ceiling ({sweep.vmem_reference_scope}, "
                f"{sweep.dma_buffers}-slot DMA): max_vmem_tokens = "
                f"{sweep.max_vmem_tokens} tokens, an upper bound capped by "
                f"max_rows_per_call when set (limit={vmem_limit:.1f} MiB, "
                f"fit={sweep.vmem_budget_mib:.1f} MiB, "
                f"static={sweep.vmem_static_mib:.2f} MiB, "
                f"slope={sweep.per_token_vmem_bytes / 1024.0:.1f} KiB/tok "
                f"[residual={vmem_breakdown['residual_buffer_bytes'] // 1024}K, "
                f"attn={vmem_breakdown['attn_scratch_bytes'] // 1024}K, "
                f"linear_attn={vmem_breakdown['linear_attn_scratch_bytes'] // 1024}K, "
                f"moe={vmem_breakdown['moe_scratch_bytes'] // 1024}K])"
            ),
            f"Roofline Ridge Crossover: {sweep.ridge_dense_flop_per_byte:.1f} FLOP/B "
            f"(T_ridge = {sweep.ridge_tokens} active tokens/step)",
        ]
    )
    return lines


def format_regime_table(sweep: AdvisorSweepReport, *, unresolved: bool = False) -> str:
    """Format multi-regime sweep table and tailoring guide for terminal output.

    Returns:
        Multi-line plain-text regime sweep report.
    """
    blocker_cells = [
        _format_table_blockers(entry.core_blockers) for entry in sweep.regimes
    ]
    blocker_col_width = max(
        [len("Core Blockers"), *(len(cell) for cell in blocker_cells)]
    )
    header = (
        f"{'Request Regime':36s} | {'Layout':22s} | {'Tokens':>6s} | "
        f"{'Intensity (FLOP/B)':>18s} | {'Passes':>6s} | {'Scope':12s} | "
        f"{'Core Support':19s} | {'Core-Ready':12s} | {'MoE Mode':15s} | "
        f"{'Peak Floor Band (eta=1.0, us)':>29s} | "
        f"{'Core Blockers':{blocker_col_width}s}"
    )
    lines = [
        *_format_sweep_header_lines(sweep),
        "",
        header,
        "-" * len(header),
        *(
            f"{entry.label:36s} | {entry.layout:22s} | {entry.active_tokens:6d} | "
            f"{entry.step_arithmetic_intensity:18.2f} | "
            f"{entry.num_vmem_splits:6d} | "
            f"{entry.recommended_scope:12s} | "
            f"{entry.core_support:19s} | "
            f"{entry.core_ready_scope:12s} | "
            f"{entry.recommended_moe_mode:15s} | "
            f"[floor {entry.step_bound_us:7.1f}.."
            f"{max(entry.step_bound_us, entry.step_upper_us):7.1f}] | "
            f"{blocker_cell:{blocker_col_width}s}"
            for entry, blocker_cell in zip(sweep.regimes, blocker_cells)
        ),
        "eta is the efficiency factor applied to peak hardware rates; "
        "eta=1.0 means peak rates.",
    ]
    if sweep.regimes:
        lines.extend(
            [
                "",
                "Investment Triage (Peak Floor Band, Derated Achievable, Gain Split):",
                *(
                    line
                    for entry in sweep.regimes
                    for line in _format_triage_entry_lines(entry)
                ),
            ]
        )
        if unresolved:
            fallback = (
                sweep.regimes[DECODE_MED_REF_INDEX]
                if len(sweep.regimes) > DECODE_MED_REF_INDEX
                else sweep.regimes[0]
            )
            decode_ref = next(
                (
                    entry
                    for entry in sweep.regimes
                    if entry.label.startswith("Decode Med")
                ),
                fallback,
            )
            if decode_ref.unresolved_assumptions:
                lines.extend(
                    [
                        "",
                        "Unresolved Assumptions (Decode Reference):",
                        *_format_unresolved_lines(decode_ref.unresolved_assumptions),
                    ]
                )
    lines.extend(
        [
            "",
            "Scope (where the kernel boundary falls):",
            *(f"  * {name:14s}: {hint}." for name, hint in SCOPE_HINTS.items()),
            "MoE Mode (where tokens are grouped by expert):",
            *(
                f"  * {name:14s}: {hint}."
                for name, hint in MOE_MODE_SHORT_HINTS.items()
            ),
            "",
            "Tailoring Guide:",
            *(f"  * {rule}" for rule in sweep.tailoring_guide),
        ]
    )
    if sweep.layer_group_search is not None:
        lines.extend(["", format_layer_group_search(sweep.layer_group_search)])
    return "\n".join(lines)


def format_layer_group_search(search: LayerGroupSearchReport) -> str:
    """Format a plain-text terminal table for `search_layer_groups`.

    Returns:
        Formatted table comparing candidate `layer_group` partitions.
    """
    header = (
        f"{'Partition':28s} | {'Calls':>5s} | {'Feasible':>8s} | "
        f"{'Peak Floor (us)':>15s} | {'Peak Upper (us)':>15s} | "
        f"{'Derated Achievable (us)':>23s}"
    )
    lines = [
        f"=== Layer-Group Partition Search: {search.model} on {search.chip} "
        f"({search.phase}, {search.layout}, active_tokens={search.active_tokens}) ===",
        f"Note: {search.note}",
        (
            f"per_op baseline: bound={search.per_op_step_bound_us:.1f} us | "
            f"achievable={search.per_op_step_achievable_us:.1f} us | "
            f"upper={search.per_op_step_upper_us:.1f} us"
        ),
        header,
        "-" * len(header),
        *(
            f"{candidate.label:28s} | {candidate.num_groups:5d} | "
            f"{('yes' if candidate.feasible else 'no'):>8s} | "
            f"{candidate.step_bound_us:15.1f} | {candidate.step_upper_us:15.1f} | "
            f"{candidate.step_achievable_us:23.1f}"
            for candidate in search.candidates
        ),
    ]
    if (best := search.best) is not None:
        listed_clause = (
            f"; vs listed: {search.gap_vs_listed_us:+.1f} us "
            f"({search.gap_vs_listed_pct:+.2f}%)"
            if search.gap_vs_listed_us is not None
            and search.gap_vs_listed_pct is not None
            else ""
        )
        lines.append(
            f"Best feasible partition: {best.label} "
            f"(bound {best.step_bound_us:.1f} us; "
            f"vs per_op: {search.gap_vs_per_op_us:+.1f} us "
            f"[{search.gap_vs_per_op_pct:+.2f}%]{listed_clause})"
        )
    else:
        lines.append("Best feasible partition: none (all partitions infeasible or OOM)")
    return "\n".join(lines)
