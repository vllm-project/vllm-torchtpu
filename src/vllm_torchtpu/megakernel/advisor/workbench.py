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
"""Thin Python-backed HTML shell and seed builder for the Fusion Workbench."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from .assets import load_report_css, load_workbench_css, load_workbench_js
from .constants import (
    BYTES_PER_MIB,
    DEFAULT_PREFILL_ISL,
    MIN_VMEM_MIB,
    RECOMMEND_DEFAULT_BATCH,
    TP_CANDIDATE_DEGREES,
)
from .report import build_bucket_schedule, render_advisor_html
from .specs import (
    DispatchRegime,
    EfficiencySpec,
    MoEMode,
    ObjectiveMode,
    Phase,
    RingPlan,
    WorkloadSpec,
    as_mapping,
    as_scalar_number,
    as_sequence,
    scalar_bool,
    scalar_float,
    scalar_int,
    scalar_str,
)
from .verdict import advise

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .chips import ChipSpec
    from .models import ModelProfile
    from .specs import AdvisorSweepReport, MeasurementsData

__all__ = [
    "PHASE_PREFILL_TOKEN_GRID",
    "PHASE_S_GRID",
    "PHASE_T_GRID",
    "PhaseMapOptions",
    "build_phase_map_payload",
    "build_workbench_seed",
    "evaluate_workbench_payload",
    "render_workbench_html",
    "write_workbench_html",
]

WorkbenchMode = Literal["live", "static"]
VMEM_SLIDER_STEP_MIB = 0.5
PHASE_T_GRID: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
PHASE_PREFILL_TOKEN_GRID: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)
PHASE_S_GRID: tuple[int, ...] = (1024, 4096, 16384, 65536, 262144)


@dataclass(frozen=True)
class PhaseMapOptions:
    """Optional overrides for 2D phase-map evaluation."""

    phase: Phase = "decode"
    base_workload: WorkloadSpec | None = None
    measurements: MeasurementsData | None = None


def _esc_attr(value: object) -> str:
    """Escape a scalar value for safe HTML text and attribute interpolation."""
    return "" if value is None else html.escape(str(value), quote=True)


def _range_knob(
    key: str, title: str, init_label: str, bounds: tuple[float, float, float, float]
) -> str:
    """Render one range slider control row bound to `wbOnControl()`."""
    low, high, step, initial = bounds
    return (
        f'<div class="wb-knob-row"><div class="wb-knob-head"><span>{title}</span>'
        f'<span class="tabular" id="lbl-{key}">{init_label}</span></div>'
        f'<input id="ctl-{key}" type="range" min="{low}" max="{high}" '
        f'step="{step}" value="{initial}" oninput="wbOnControl()"/></div>'
    )


def _select_knob(
    key: str, title: str, init_label: str, options: tuple[tuple[str, str], ...]
) -> str:
    """Render one select dropdown control row bound to `wbOnControl()`."""
    options_html = "".join(
        f'<option value="{option_value}">{option_label}</option>'
        for option_value, option_label in options
    )
    return (
        f'<div class="wb-knob-row"><div class="wb-knob-head"><span>{title}</span>'
        f'<span class="tabular" id="lbl-{key}">{init_label}</span></div>'
        f'<select id="ctl-{key}" onchange="wbOnControl()">{options_html}</select></div>'
    )


def _map_opt_int(mapping: Mapping[str, object], key: str, default: int) -> int:
    """Read an optional strict integer field from a serialized workload dict."""
    raw = mapping.get(key)
    return default if raw is None else scalar_int(raw, key)


def _map_opt_float(mapping: Mapping[str, object], key: str, default: float) -> float:
    """Read an optional finite float field from a serialized workload dict."""
    raw = mapping.get(key)
    return default if raw is None else scalar_float(raw, key)


def _map_opt_nullable_float(mapping: Mapping[str, object], key: str) -> float | None:
    """Read an optional nullable finite float from a serialized workload dict."""
    raw = mapping.get(key)
    return None if raw is None else scalar_float(raw, key)


def _coerce_efficiency(raw: object) -> EfficiencySpec:
    """Convert serialized efficiency mapping or return default `EfficiencySpec`."""
    default_efficiency = EfficiencySpec()
    eff_map = as_mapping(raw)
    if eff_map is None:
        return default_efficiency
    return EfficiencySpec(
        eta_hbm=_map_opt_float(eff_map, "eta_hbm", default_efficiency.eta_hbm),
        eta_dense=_map_opt_float(eff_map, "eta_dense", default_efficiency.eta_dense),
        eta_moe_grouped=_map_opt_float(
            eff_map, "eta_moe_grouped", default_efficiency.eta_moe_grouped
        ),
        eta_ici=_map_opt_float(eff_map, "eta_ici", default_efficiency.eta_ici),
    )


def _coerce_ring_plan(raw: object) -> RingPlan | None:
    """Convert serialized ring-plan mapping into `RingPlan` when present."""
    ring_map = as_mapping(raw)
    if ring_map is None:
        return None
    defaults = RingPlan()
    return RingPlan(
        dense_slots=_map_opt_int(ring_map, "dense_slots", defaults.dense_slots),
        expert_slots=_map_opt_int(ring_map, "expert_slots", defaults.expert_slots),
        dense_slot_bytes=_map_opt_int(
            ring_map, "dense_slot_bytes", defaults.dense_slot_bytes
        ),
        expert_slot_bytes=_map_opt_int(
            ring_map, "expert_slot_bytes", defaults.expert_slot_bytes
        ),
        compiler_spill_reserve_bytes=_map_opt_int(
            ring_map,
            "compiler_spill_reserve_bytes",
            defaults.compiler_spill_reserve_bytes,
        ),
    )


def _coerce_workload(workload: WorkloadSpec | Mapping[str, object]) -> WorkloadSpec:
    """Convert a serialized `RegimeEntry.workload` mapping into `WorkloadSpec`."""
    if type(workload) is WorkloadSpec:
        return workload
    workload_map = as_mapping(workload)
    if workload_map is None:
        msg = f"Expected WorkloadSpec or Mapping, got {type(workload).__name__}"
        raise TypeError(msg)
    defaults = WorkloadSpec()
    raw_groups = as_sequence(workload_map.get("layer_groups"))
    layer_groups = (
        tuple(scalar_int(item, "layer_groups") for item in raw_groups)
        if raw_groups is not None
        else ()
    )
    return WorkloadSpec(
        phase=cast(
            "Phase",
            scalar_str(workload_map.get("phase", defaults.phase), "phase"),
        ),
        batch=_map_opt_int(workload_map, "batch", defaults.batch),
        isl=_map_opt_int(workload_map, "isl", defaults.isl),
        osl=_map_opt_int(workload_map, "osl", defaults.osl),
        context=_map_opt_int(workload_map, "context", defaults.context),
        prefix_hit_rate=_map_opt_float(
            workload_map, "prefix_hit_rate", defaults.prefix_hit_rate
        ),
        spec_tokens=_map_opt_int(workload_map, "spec_tokens", defaults.spec_tokens),
        chunk_size=_map_opt_int(workload_map, "chunk_size", defaults.chunk_size),
        microbatches=_map_opt_int(workload_map, "microbatches", defaults.microbatches),
        tp=_map_opt_int(workload_map, "tp", defaults.tp),
        pp=_map_opt_int(workload_map, "pp", defaults.pp),
        dp=_map_opt_int(workload_map, "dp", defaults.dp),
        ep=_map_opt_int(workload_map, "ep", defaults.ep),
        kv_split=_map_opt_int(workload_map, "kv_split", defaults.kv_split),
        dcp=_map_opt_int(workload_map, "dcp", defaults.dcp),
        pcp=_map_opt_int(workload_map, "pcp", defaults.pcp),
        segment_layers=_map_opt_int(
            workload_map, "segment_layers", defaults.segment_layers
        ),
        layer_groups=layer_groups,
        moe_mode=cast(
            "MoEMode",
            scalar_str(workload_map.get("moe_mode", defaults.moe_mode), "moe_mode"),
        ),
        vmem_alias=scalar_bool(
            workload_map.get("vmem_alias", defaults.vmem_alias), "vmem_alias"
        ),
        vmem_fit_headroom=scalar_bool(
            workload_map.get("vmem_fit_headroom", defaults.vmem_fit_headroom),
            "vmem_fit_headroom",
        ),
        op_overhead_us=_map_opt_float(
            workload_map, "op_overhead_us", defaults.op_overhead_us
        ),
        vmem_budget_mib=_map_opt_nullable_float(workload_map, "vmem_budget_mib"),
        dma_buffers=_map_opt_int(workload_map, "dma_buffers", defaults.dma_buffers),
        ring_plan=_coerce_ring_plan(workload_map.get("ring_plan")),
        objective=cast(
            "ObjectiveMode",
            scalar_str(workload_map.get("objective", defaults.objective), "objective"),
        ),
        hbm_utilization_cap=_map_opt_float(
            workload_map, "hbm_utilization_cap", defaults.hbm_utilization_cap
        ),
        dispatch_regime=cast(
            "DispatchRegime",
            scalar_str(
                workload_map.get("dispatch_regime", defaults.dispatch_regime),
                "dispatch_regime",
            ),
        ),
        efficiency=_coerce_efficiency(workload_map.get("efficiency")),
        max_rows_per_call=_map_opt_int(
            workload_map, "max_rows_per_call", defaults.max_rows_per_call
        ),
        host_head_per_op_us=_map_opt_float(
            workload_map, "host_head_per_op_us", defaults.host_head_per_op_us
        ),
        host_head_fused_us=_map_opt_float(
            workload_map, "host_head_fused_us", defaults.host_head_fused_us
        ),
        host_head_explicit=scalar_bool(
            workload_map.get("host_head_explicit", defaults.host_head_explicit),
            "host_head_explicit",
        ),
        per_op_step_us=_map_opt_nullable_float(workload_map, "per_op_step_us"),
        step_coverage=_map_opt_nullable_float(workload_map, "step_coverage"),
        batch_explicit=scalar_bool(
            workload_map.get("batch_explicit", defaults.batch_explicit),
            "batch_explicit",
        ),
        moe_mode_explicit=scalar_bool(
            workload_map.get("moe_mode_explicit", defaults.moe_mode_explicit),
            "moe_mode_explicit",
        ),
        latent_proj_tp=_map_opt_int(
            workload_map, "latent_proj_tp", defaults.latent_proj_tp
        ),
    )


def evaluate_workbench_payload(
    profile: ModelProfile,
    workload: WorkloadSpec | Mapping[str, object],
    chip_spec: ChipSpec,
    *,
    measurements: MeasurementsData | None = None,
) -> dict[str, object]:
    """Evaluate `advise()` plus a zero-gap comparison directly in Python."""
    resolved_workload = _coerce_workload(workload)
    report = advise(profile, resolved_workload, chip_spec, measurements=measurements)
    zero_report = advise(
        profile,
        replace(resolved_workload, dispatch_regime="zero_floor"),
        chip_spec,
        measurements=measurements,
    )
    return {"report": report.to_dict(), "zero_floor_report": zero_report.to_dict()}


def build_phase_map_payload(
    profile: ModelProfile,
    chip_spec: ChipSpec,
    sweep: AdvisorSweepReport,
    *,
    options: PhaseMapOptions | None = None,
) -> dict[str, object]:
    """Evaluate the 2D (tokens x context) winning-scope grid and schedule."""
    map_options = options or PhaseMapOptions()
    phase, parallelism = map_options.phase, sweep.parallelism
    reference_workload = map_options.base_workload or WorkloadSpec(
        phase=phase,
        batch=RECOMMEND_DEFAULT_BATCH,
        tp=parallelism.tp,
        ep=parallelism.ep,
        dp=parallelism.dp,
        pp=parallelism.pp,
        dcp=parallelism.dcp,
        pcp=parallelism.pcp,
        layer_groups=parallelism.layer_groups,
    )
    dp_size = max(1, reference_workload.dp)
    t_grid = PHASE_PREFILL_TOKEN_GRID if phase == "prefill" else PHASE_T_GRID
    cells: list[dict[str, object]] = []
    for context_tokens in PHASE_S_GRID:
        for token_count in t_grid:
            if phase == "prefill":
                cell_batch = dp_size
                cell_isl = max(token_count, context_tokens)
                cell_chunk = token_count
                cell_spec = 0
            elif phase == "verify":
                cell_spec = max(1, reference_workload.spec_tokens)
                group_width = cell_spec + 1
                seqs_per_dp = max(1, (token_count + group_width - 1) // group_width)
                cell_batch = seqs_per_dp * dp_size
                cell_isl = DEFAULT_PREFILL_ISL
                cell_chunk = 0
            else:
                cell_batch = token_count * dp_size
                cell_isl = DEFAULT_PREFILL_ISL
                cell_chunk = 0
                cell_spec = 0
            cell_workload = replace(
                reference_workload,
                phase=phase,
                batch=cell_batch,
                isl=cell_isl,
                context=context_tokens,
                chunk_size=cell_chunk,
                spec_tokens=cell_spec,
            )
            report = advise(
                profile,
                cell_workload,
                chip_spec,
                measurements=map_options.measurements,
            )
            cells.append(
                {
                    "tokens": token_count,
                    "context": context_tokens,
                    "recommended_scope": report.recommended_scope,
                    "fastest_scope": report.fastest_scope,
                    "core_ready_scope": report.core_ready_scope,
                    "core_support": report.core_support,
                    "core_blockers": list(report.core_blockers),
                    "hbm_fits": report.hbm_fits,
                    "verdict": report.verdict,
                }
            )
    return {
        "phase": phase,
        "t_grid": list(t_grid),
        "s_grid": list(PHASE_S_GRID),
        "cells": cells,
        "bucket_schedule": build_bucket_schedule(sweep).to_dict(),
    }


def build_workbench_seed(
    profile: ModelProfile,
    chip_spec: ChipSpec,
    sweep: AdvisorSweepReport,
    *,
    mode: WorkbenchMode = "static",
    measurements: MeasurementsData | None = None,
) -> dict[str, object]:
    """Assemble serialized JSON seed embedded inside `advisor-data`."""
    parallelism = sweep.parallelism
    regime_evals = [
        evaluate_workbench_payload(
            profile, entry.workload, chip_spec, measurements=measurements
        )
        for entry in sweep.regimes
    ]
    if regime_evals:
        default_eval = regime_evals[0]
    else:
        fallback_workload = WorkloadSpec(
            phase="decode",
            batch=RECOMMEND_DEFAULT_BATCH,
            tp=parallelism.tp,
            ep=parallelism.ep,
            dp=parallelism.dp,
            pp=parallelism.pp,
            dcp=parallelism.dcp,
            pcp=parallelism.pcp,
            layer_groups=parallelism.layer_groups,
        )
        default_eval = evaluate_workbench_payload(
            profile, fallback_workload, chip_spec, measurements=measurements
        )
    phase_map = build_phase_map_payload(
        profile, chip_spec, sweep, options=PhaseMapOptions(measurements=measurements)
    )
    return {
        "mode": mode,
        "model": sweep.model,
        "chip": sweep.chip,
        "layout": parallelism.label(),
        "vmem_budget_mib": sweep.vmem_budget_mib,
        "default_efficiency": asdict(EfficiencySpec()),
        "regimes": [entry.to_dict() for entry in sweep.regimes],
        "regime_evals": regime_evals,
        "default_eval": default_eval,
        "phase_map": phase_map,
        "bucket_schedule": phase_map["bucket_schedule"],
    }


def _render_sidebar_controls(sweep: AdvisorSweepReport, vmem_max_mib: float) -> str:
    """Render sidebar VMEM, efficiency, workload, mesh, and upload controls."""
    mesh_options = tuple((str(val), str(val)) for val in TP_CANDIDATE_DEGREES)
    mesh_grid = "".join(
        _select_knob(axis_name, axis_name.upper(), "1", mesh_options)
        for axis_name in ("tp", "ep", "dp", "pp", "dcp", "pcp")
    )
    default_efficiency = EfficiencySpec()
    eta_rows = "".join(
        _range_knob(
            knob_key,
            knob_label,
            f"{initial_value:.2f}",
            (0.10, 1.0, 0.02, initial_value),
        )
        for knob_key, knob_label, initial_value in (
            ("eta-dense", "Dense eta", default_efficiency.eta_dense),
            ("eta-moe", "MoE eta", default_efficiency.eta_moe_grouped),
            ("eta-hbm", "HBM eta", default_efficiency.eta_hbm),
            ("eta-ici", "ICI eta", default_efficiency.eta_ici),
        )
    )
    hardware_knobs_html = (
        '<div class="wb-knob-head"><span>Dispatch Regime</span>'
        '<span class="tabular" id="lbl-dispatch-regime">compiled_xla</span></div>'
        '<div class="wb-dr-group">'
        '<button class="wb-mini-btn" type="button" data-regime="zero_floor" '
        "onclick=\"wbSetDispatchRegime('zero_floor')\">Zero-Gap</button>"
        '<button class="wb-mini-btn active-a" type="button" '
        'data-regime="compiled_xla" '
        "onclick=\"wbSetDispatchRegime('compiled_xla')\">Compiled XLA</button>"
        '<button class="wb-mini-btn" type="button" data-regime="unfused_eager" '
        "onclick=\"wbSetDispatchRegime('unfused_eager')\">Exposed</button></div>"
        + _range_knob(
            "vmem",
            "VMEM Budget",
            f"{sweep.vmem_budget_mib:.1f} MiB",
            (
                float(MIN_VMEM_MIB),
                vmem_max_mib,
                VMEM_SLIDER_STEP_MIB,
                sweep.vmem_budget_mib,
            ),
        )
        + '<div class="wb-knob-pair">'
        + _select_knob(
            "dma", "DMA Slots", "2 slots", (("2", "2 slots"), ("3", "3 slots"))
        )
        + _select_knob("alias", "Scratch Alias", "on", (("1", "on"), ("0", "off")))
        + "</div>"
        f'<details class="wb-eta-details"><summary>Efficiency (&eta;)</summary>'
        f"{eta_rows}</details>"
    )
    workload_knobs_html = (
        _select_knob(
            "phase",
            "Phase",
            "decode",
            (
                ("decode", "decode"),
                ("verify", "verify"),
                ("prefill", "prefill"),
                ("mixed", "mixed"),
            ),
        )
        + _range_knob(
            "batch",
            "Decoding Seqs (B)",
            str(RECOMMEND_DEFAULT_BATCH),
            (1, 1024, 1, RECOMMEND_DEFAULT_BATCH),
        )
        + '<details class="wb-eta-details"><summary>Little\'s Law</summary>'
        '<div class="wb-knob-pair">'
        '<input id="ctl-out-tps" type="number" min="0.1" step="1" '
        'placeholder="Out tok/s"/>'
        '<input id="ctl-mean-tpot" type="number" min="0.1" step="0.1" '
        'placeholder="TPOT ms"/></div>'
        '<button class="wb-mini-btn" type="button" '
        'onclick="wbApplyLittlesLaw()">Apply</button></details>'
        + _range_knob("seq-tokens", "Tokens/Seq", "1", (1, 8192, 1, 1))
        + _range_knob("context", "Context (S)", "4K", (512, 1048576, 512, 4096))
        + '<details class="wb-eta-details"><summary>6-Axis Mesh</summary>'
        f'<div class="wb-mesh-grid">{mesh_grid}</div></details>'
    )
    return (
        '<div class="wb-sec-title">2. VMEM &amp; Runtime Priors</div>'
        f'<div class="wb-knob-group">{hardware_knobs_html}</div>'
        '<div class="wb-sec-title">3. Workload &amp; Mesh</div>'
        f'<div class="wb-knob-group">{workload_knobs_html}</div>'
        '<div class="wb-sec-title">4. Trace Calibration (.xplane.pb / .json)</div>'
        '<div class="wb-knob-group" id="wb-cal-dropzone" '
        'ondragover="event.preventDefault()" ondrop="wbOnDropCalibration(event)">'
        '<input id="cal-file-input" type="file" accept=".json,.gz,.pb" '
        'onchange="wbOnPickCalibration(event)"/>'
        '<div id="wb-cal-status" class="wb-sub">Status: hardware priors</div></div>'
    )


def _initial_shell_fields(
    init_report: Mapping[str, object] | None,
) -> tuple[dict[str, str], str, str]:
    """Build initial top-bar KPIs, pill text, and summary HTML from `init_report`."""
    if init_report is None:
        empty_kpis = {
            "scope": "-",
            "core": "-",
            "floor": "-",
            "tmax": "-",
            "tps": "-",
            "bind": "-",
        }
        return empty_kpis, "-", "-"
    rec_scope = str(init_report.get("recommended_scope") or "-")
    verdict_text = str(init_report.get("verdict") or "")
    pill_text = (
        "INVALID CONFIG"
        if verdict_text.startswith("invalid_config(")
        else _esc_attr(rec_scope.upper())
    )
    scope_dicts = [
        item
        for raw in (as_sequence(init_report.get("scopes")) or ())
        if (item := as_mapping(raw)) is not None
    ]
    rec_dict = next(
        (item for item in scope_dicts if item.get("scope") == rec_scope),
        scope_dicts[0] if scope_dicts else None,
    )
    floor_val = (
        as_scalar_number(rec_dict.get("step_bound_us"))
        if rec_dict is not None
        else as_scalar_number(init_report.get("step_bound_us"))
    )
    tmax_val = (
        as_scalar_number(rec_dict.get("max_vmem_tokens"))
        if rec_dict is not None
        else as_scalar_number(init_report.get("max_vmem_tokens"))
    )
    metrics_map = as_mapping(init_report.get("metrics")) or {}
    tps_map = as_mapping(metrics_map.get("throughput_tok_s_per_chip"))
    tps_val = as_scalar_number(tps_map.get("value")) if tps_map is not None else None
    bind_raw = init_report.get("binding_unit") or (
        rec_dict.get("binding_unit") if rec_dict is not None else ""
    )
    core_support = str(
        init_report.get("core_support")
        or (rec_dict.get("support") if rec_dict is not None else "")
        or "baseline"
    )
    core_ready_scope = str(init_report.get("core_ready_scope") or "per_op")
    blocker_items = [
        str(item)
        for item in (as_sequence(init_report.get("core_blockers")) or ())
        if str(item)
    ]
    if core_support == "outside_core_limits" or blocker_items:
        blk_text = ", ".join(blocker_items) or "none"
        blk_html = f" &middot; Blocked: <code>{_esc_attr(blk_text)}</code>"
    else:
        blk_html = ""
    init_summary = (
        f"Core: <code>{_esc_attr(core_support)}</code> "
        f"(ready: <code>{_esc_attr(core_ready_scope)}</code>)"
        f"{blk_html}"
    )
    init_kpis = {
        "scope": _esc_attr(rec_scope),
        "core": _esc_attr(core_support),
        "floor": f"{floor_val:.1f} us" if floor_val is not None else "-",
        "tmax": f"{int(tmax_val)} tok" if tmax_val is not None else "-",
        "tps": f"{tps_val:.1f}" if tps_val is not None else "-",
        "bind": _esc_attr(str(bind_raw or "-")),
    }
    return init_kpis, pill_text, init_summary


def _render_workbench_shell(
    sweep: AdvisorSweepReport,
    vmem_max_mib: float,
    *,
    default_eval: Mapping[str, object] | None = None,
) -> str:
    """Render the 3-pane Fusion Workbench HTML layout shell."""
    model_esc = _esc_attr(sweep.model)
    chip_esc = _esc_attr(sweep.chip)
    layout_esc = _esc_attr(sweep.parallelism.label())
    controls_html = _render_sidebar_controls(sweep, vmem_max_mib)
    init_report = as_mapping((default_eval or {}).get("report"))
    if init_report is None and sweep.regimes:
        init_report = sweep.regimes[0].to_dict()
    init_kpis, init_pill, init_summary = _initial_shell_fields(init_report)
    kpis = "".join(
        f'<div class="wb-pill-kpi">{kpi_label}: '
        f'<strong class="tabular" id="wb-kpi-{kpi_key}">'
        f"{init_kpis[kpi_key]}</strong></div>"
        for kpi_key, kpi_label in (
            ("scope", "SCOPE"),
            ("core", "CORE"),
            ("floor", "FLOOR"),
            ("tmax", "MAX TOKENS"),
            ("tps", "TOK/S/CHIP"),
            ("bind", "BIND"),
        )
    )
    scope_options_html = "".join(
        f'<option value="{scope_name}">{scope_name}</option>'
        for scope_name in ("whole_step", "layer_group", "single_layer", "per_op")
    )
    return (
        f'<header class="wb-topbar"><div class="wb-brand">'
        f"<h1>Megakernel Fusion Workbench &middot; {model_esc}</h1>"
        f'<div class="wb-sub">Chip: <strong>{chip_esc}</strong> &middot; '
        f"Mesh: <strong>{layout_esc}</strong></div></div>"
        '<div class="wb-verdict-wrap">'
        f'<span class="wb-verdict-pill" id="wb-verdict-pill">{init_pill}</span>'
        f'<div class="wb-verdict-meta tabular" id="wb-verdict-summary">'
        f"{init_summary}</div></div>"
        f'<div class="wb-top-kpis">{kpis}</div><div class="wb-actions">'
        '<button class="wb-tab-btn active" id="btn-view-workbench" type="button" '
        "onclick=\"wbSetView('workbench')\">Workbench</button>"
        '<button class="wb-tab-btn" id="btn-view-audit" type="button" '
        "onclick=\"wbSetView('audit')\">Audit Report</button>"
        '<button class="wb-tab-btn" type="button" '
        'onclick="toggleTheme()">Theme</button></div></header>'
        '<div id="wb-static-banner" class="wb-static-banner view-hidden"></div>'
        '<div id="wb-invalid-banner" class="wb-invalid-banner view-hidden" '
        'role="alert"></div>'
        '<main class="wb-grid" id="view-workbench"><aside class="wb-rail">'
        '<div class="wb-sec-title">1. Request Regimes ([A] vs [B])</div>'
        '<div style="display:flex;gap:5px;margin-bottom:4px;">'
        '<button class="wb-mini-btn" type="button" '
        'onclick="wbSwapAB()">Swap</button>'
        '<button class="wb-mini-btn" type="button" '
        'onclick="wbClearB()">vs per_op</button>'
        '<button class="wb-mini-btn" type="button" '
        'onclick="wbResetSliders()">Reset</button></div>'
        f'<div class="wb-regime-list" id="wb-regime-list"></div>'
        f"{controls_html}</aside>"
        '<section class="wb-stage"><div style="display:flex;gap:8px;flex-wrap:wrap;">'
        '<button class="wb-tab-btn active" id="btn-stage-arena" type="button" '
        "onclick=\"wbSetStageTab('arena')\">VMEM Floorplan &amp; Waterfall</button>"
        '<button class="wb-tab-btn" id="btn-stage-phase" type="button" '
        "onclick=\"wbSetStageTab('phase')\">2D Phase Map &amp; Buckets</button></div>"
        '<div id="wb-stage-arena" '
        'style="display:flex;flex-direction:column;gap:12px;">'
        '<div class="wb-panel"><div class="wb-panel-head">'
        "<h2>VMEM Spatial Floorplan</h2>"
        '<select class="wb-inline-sel" id="ctl-arena-scope" '
        f'onchange="wbOnControl()">{scope_options_html}</select></div>'
        '<div id="wb-vmem-arena-svg"></div></div>'
        '<div class="wb-panel"><div class="wb-panel-head">'
        "<h2>4-Scope Step Timeline</h2></div>"
        '<div id="wb-scope-timeline-svg"></div></div>'
        '<div class="wb-panel"><div class="wb-panel-head">'
        '<h2 id="wb-waterfall-title">Attribution</h2>'
        '<span class="kpi-sub" id="wb-waterfall-summary"></span></div>'
        '<div id="wb-waterfall-svg"></div></div></div>'
        '<div id="wb-stage-phase" '
        'style="display:none;flex-direction:column;gap:12px;">'
        '<div class="wb-panel"><div class="wb-panel-head">'
        "<h2>2D Winning-Scope Phase Map</h2>"
        '<select class="wb-inline-sel" id="wb-phase-map-mode" '
        'onchange="wbSetPhaseMapMode(this.value)">'
        '<option value="recommended">Physics Winner</option>'
        '<option value="core_ready">Core-Ready Today</option></select></div>'
        '<div id="wb-phase-map-svg"></div></div>'
        '<div class="wb-panel"><div class="wb-panel-head">'
        "<h2>Bucket Schedule JSON</h2>"
        '<button class="wb-mini-btn" type="button" '
        'onclick="wbCopyBucketSchedule()">Copy JSON</button></div>'
        '<pre class="wb-formula" id="wb-bucket-schedule-pre"></pre>'
        "</div></div></section>"
        '<aside class="wb-inspector">'
        '<div class="wb-sec-title">Proof-Ladder Inspector</div>'
        '<div id="wb-deriv-drawer"></div></aside></main>'
    )


def render_workbench_html(
    profile: ModelProfile,
    chip_spec: ChipSpec,
    sweep: AdvisorSweepReport,
    *,
    mode: WorkbenchMode = "static",
    measurements: MeasurementsData | None = None,
) -> str:
    """Render the Python-backed workbench HTML embedding the zero-JS audit view."""
    seed = build_workbench_seed(
        profile, chip_spec, sweep, mode=mode, measurements=measurements
    )
    safe_seed_json = (
        json.dumps(seed)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    audit_full = render_advisor_html(sweep)
    body_start, body_end = audit_full.find("<body>"), audit_full.rfind("</body>")
    audit_inner = (
        audit_full[body_start + len("<body>") : body_end]
        if body_start >= 0 and body_end > body_start
        else ""
    )
    vmem_max_mib = round(chip_spec.vmem_usable_bytes / BYTES_PER_MIB, 1)
    shell_html = _render_workbench_shell(
        sweep,
        max(vmem_max_mib, sweep.vmem_budget_mib),
        default_eval=as_mapping(seed.get("default_eval")),
    )
    title = _esc_attr(f"Megakernel Fusion Workbench - {sweep.model} ({sweep.chip})")
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        f'<head><meta charset="utf-8"/><title>{title}</title>\n'
        f"<style>\n{load_report_css()}\n{load_workbench_css()}\n</style></head>\n"
        f'<body class="wb-mode-active">\n{shell_html}\n'
        '<div id="view-audit" class="audit-scroll-wrap view-hidden">'
        f"{audit_inner}</div>\n"
        f'<script id="advisor-data" type="application/json">{safe_seed_json}</script>\n'
        f"<script>\n{load_workbench_js()}\n</script>\n</body></html>\n"
    )


def write_workbench_html(
    profile: ModelProfile,
    chip_spec: ChipSpec,
    sweep: AdvisorSweepReport,
    path: str | Path,
    *,
    measurements: MeasurementsData | None = None,
) -> Path:
    """Write a static-snapshot Python-backed HTML workbench to `path`."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        render_workbench_html(
            profile, chip_spec, sweep, mode="static", measurements=measurements
        ),
        encoding="utf-8",
    )
    return dest
