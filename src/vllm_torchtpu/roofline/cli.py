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
"""CLI entrypoint for physics lower bounds, ceiling export, and explain."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from .chips import (
    CHIPS,
    ChipSpec,
    SliceTopology,
    load_chip_spec,
    override_chip_ici_bw,
)
from .constants import (
    DEFAULT_HBM_RESERVE_GB,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SLO_TIERS,
    EXPERT_DTYPE_BYTES,
    KV_DTYPE_BYTES,
    MIN_INSTANCES,
    MIN_TOP_K,
    NO_FEASIBLE_LAYOUT_WARNING,
    STATE_DTYPE_BYTES,
    UNIT_GIGA,
    UNIT_KILO,
    UNIT_MEGA,
    UNIT_MICRO,
    WEIGHT_DTYPE_BYTES,
)
from .emit import (
    build_solve_payload,
    emit_ceiling_rows,
    format_solve_json,
    slice_per_concurrency,
    top_layouts,
)
from .families import (
    attach_reference_report,
    parse_layout,
    reference_layout_report,
)
from .geometry import ModelGeometry, UnsupportedArchitectureError, from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config_with_identity
from .modules import Dtypes, Parallel, StepShape
from .report import write_html
from .schema import ReferenceLayoutReport
from .serving import ServingRequest, plan
from .solver import (
    EXAMPLE_SCENARIOS,
    PoolMeshPlan,
    ScenarioOverrides,
    ServingPlan,
    SolveOptions,
    TrafficScenario,
    _resolve_solve_topologies,
    build_example_scenario,
    explain_layout_rejection,
    solve_serving_plan,
)
from .step import FloorOptions, SpecShape, spec_cycle_floor, step_floor

if TYPE_CHECKING:
    from collections.abc import Sequence


def _add_model_source(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive --config or --model flags plus --hf-overrides."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Path to HF config.json")
    group.add_argument("--model", help="HF model ID, directory, or config.json")
    parser.add_argument(
        "--hf-overrides",
        default=None,
        help="JSON dict merged into the loaded HF config before validation.",
    )
    parser.add_argument(
        "--chip-spec",
        default=None,
        help="Path to declarative JSON chip specification overriding --chip.",
    )


def _add_dtype_override_args(parser: argparse.ArgumentParser) -> None:
    """Add optional dtype overrides (default None = derived from HF config)."""
    parser.add_argument(
        "--weight-dtype", default=None, choices=tuple(sorted(WEIGHT_DTYPE_BYTES))
    )
    parser.add_argument(
        "--expert-weight-dtype",
        default=None,
        choices=tuple(sorted(EXPERT_DTYPE_BYTES)),
    )
    parser.add_argument(
        "--aux-dtype", default=None, choices=tuple(sorted(WEIGHT_DTYPE_BYTES))
    )
    parser.add_argument(
        "--kv-dtype", default=None, choices=tuple(sorted(KV_DTYPE_BYTES))
    )
    parser.add_argument(
        "--state-dtype", default=None, choices=tuple(sorted(STATE_DTYPE_BYTES))
    )
    parser.add_argument(
        "--act",
        "--act-dtype",
        dest="act",
        default=None,
        choices=("a4", "a8", "a16", "fp4", "nvfp4", "fp8", "bf16", "bfloat16"),
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common serving and parallel execution flags to parser."""
    _add_model_source(parser)
    parser.add_argument("--chip", default="tpu7x", choices=sorted(CHIPS))
    parser.add_argument("--tp", type=int, required=True, default=None)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--kv-split", type=int, default=1)
    parser.add_argument("--kv-merge", default="ag_rs", choices=("ag_rs", "exchange"))
    parser.add_argument("--moe", default="tp_slice", choices=("tp_slice", "ep"))
    parser.add_argument(
        "--moe-dispatch",
        default="allreduce",
        choices=("allreduce", "a2a_push", "ag_ep", "fused_rs"),
    )
    parser.add_argument(
        "--residual", default="replicated", choices=("replicated", "sp")
    )
    parser.add_argument(
        "--ar-algo",
        default="one_shot",
        choices=("one_shot", "two_shot", "ring"),
    )
    _add_dtype_override_args(parser)
    parser.add_argument("--ici-bw-gbps", type=float, default=None)
    parser.add_argument("--collective-latency-us", type=float, default=None)
    parser.add_argument("--json", action="store_true")


def _add_scenario_args(parser: argparse.ArgumentParser) -> None:
    """Add traffic scenario, SLO, and instance constraint flags."""
    _add_model_source(parser)
    parser.add_argument("--chip", default="tpu7x", choices=sorted(CHIPS))
    parser.add_argument(
        "--example-scenario", default=None, choices=sorted(EXAMPLE_SCENARIOS)
    )
    parser.add_argument("--mode", default="auto", choices=("auto", "mixed", "disagg"))
    parser.add_argument(
        "--max-chips", "--num-chips", dest="max_chips", type=int, default=None
    )
    parser.add_argument("--prefill-topology", type=str, default=None)
    parser.add_argument("--decode-topology", type=str, default=None)
    parser.add_argument("--pd", type=str, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "--example-slo-tier", default="none", choices=sorted(EXAMPLE_SLO_TIERS)
    )
    parser.add_argument("--ttft-slo-ms", type=float, default=None)
    parser.add_argument("--tpot-slo-ms", type=float, default=None)
    parser.add_argument("--isl", type=int, default=None)
    parser.add_argument("--osl", type=int, default=None)
    parser.add_argument("--concurrency", default=None)
    parser.add_argument("--prefix-hit-rate", type=float, default=None)
    parser.add_argument("--accept-len", type=float, default=None)
    parser.add_argument("--think-time-s", type=float, default=None)
    _add_dtype_override_args(parser)
    parser.add_argument("--ici-bw-gbps", type=float, default=None)
    parser.add_argument(
        "--hbm-reserve-gb",
        type=float,
        default=DEFAULT_HBM_RESERVE_GB,
    )
    parser.add_argument(
        "--sort-by",
        default="throughput",
        choices=("throughput", "interactivity"),
    )
    parser.add_argument("--pairs", action="store_true")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for physics bounds, ceiling export, and explain."""
    parser = argparse.ArgumentParser(
        prog="python -m vllm_torchtpu.roofline",
        description=(
            "Compute physical lower bounds and uncalibrated serial-phase "
            "layout rankings across TPU parallel configurations."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    solve_parser = sub.add_parser(
        "solve",
        help=(
            "Prune with physical HBM bounds and order all physically legal layouts "
            "against SLOs."
        ),
        description=(
            "Prune parallel layouts via physical HBM bounds and order all "
            "uncalibrated serial-phase candidates against traffic SLOs."
        ),
    )
    _add_scenario_args(solve_parser)
    solve_parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Write an offline HTML report to PATH.",
    )
    solve_parser.add_argument(
        "--reference-layout",
        default=None,
        help=(
            "Compare a reference layout (e.g. 'TP8 x EP8' or "
            "'P: PP4 / D: TP8') against solver candidates."
        ),
    )

    ceiling_parser = sub.add_parser(
        "ceiling",
        help="Export uncalibrated physics candidate ceiling rows from solve.",
        description=(
            "Export uncalibrated serial-phase physics candidate rows along the "
            "estimated throughput-interactivity envelope."
        ),
    )
    _add_scenario_args(ceiling_parser)

    plan_parser = sub.add_parser(
        "plan",
        help=(
            "Compute hard physical step floors and uncalibrated serial "
            "decode estimates."
        ),
        description=(
            "Compute per-concurrency hard physical step floors and "
            "uncalibrated serial-phase decode estimates for one parallel layout."
        ),
    )
    _add_common_args(plan_parser)
    plan_parser.add_argument("--concurrency", default="1,4,16,64")
    plan_parser.add_argument("--spec-window", type=int, default=0)
    plan_parser.add_argument("--accept-len", type=float, default=None)

    explain_parser = sub.add_parser(
        "explain",
        help=(
            "Break down per-module physical HBM, FLOP, and ICI volumes for one step."
        ),
        description=(
            "Report per-module physical HBM, FLOP, and ICI collective "
            "volumes for one decode or speculative cycle step."
        ),
    )
    _add_common_args(explain_parser)
    explain_parser.add_argument("--batch", type=int, default=1)
    explain_parser.add_argument("--rows-per-seq", type=int, default=1)
    explain_parser.add_argument("--spec-window", type=int, default=0)
    return parser


def _require_positive_finite(name: str, value: float | None) -> None:
    """Raise ValueError when an optional float flag is non-positive or non-finite."""
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        msg = f"{name} must be finite and > 0, got {value}"
        raise ValueError(msg)


def _require_non_negative_finite(name: str, value: float | None) -> None:
    """Raise ValueError when an optional float flag is negative or non-finite."""
    if value is not None and (not math.isfinite(value) or value < 0.0):
        msg = f"{name} must be finite and >= 0, got {value}"
        raise ValueError(msg)


def _require_positive_int(name: str, value: int | None) -> None:
    """Raise ValueError when an optional integer flag is < 1."""
    if value is not None and value < 1:
        msg = f"{name} must be >= 1, got {value}"
        raise ValueError(msg)


def _parse_concurrencies(raw: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated list of positive concurrency ints."""
    values = tuple(int(item) for item in raw.split(",") if item.strip())
    if not values or any(val < 1 for val in values):
        msg = f"--concurrency must contain positive integers (>= 1), got {raw!r}"
        raise ValueError(msg)
    return values


def _resolve_scenario(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> TrafficScenario:
    """Validate CLI flags and build the target TrafficScenario."""
    _require_positive_finite("--ttft-slo-ms", args.ttft_slo_ms)
    _require_positive_finite("--tpot-slo-ms", args.tpot_slo_ms)
    _require_non_negative_finite("--think-time-s", args.think_time_s)
    if args.example_scenario is None and args.example_slo_tier != "none":
        msg = "--example-slo-tier requires --example-scenario"
        raise ValueError(msg)
    concurrencies = _parse_concurrencies(args.concurrency) if args.concurrency else None
    if args.example_scenario is not None:
        return build_example_scenario(
            args.example_scenario,
            slo_tier=args.example_slo_tier,
            overrides=ScenarioOverrides(
                isl=args.isl,
                osl=args.osl,
                concurrencies=concurrencies,
                prefix_hit_rate=args.prefix_hit_rate,
                accept_len=args.accept_len,
                think_time_s=args.think_time_s,
                ttft_slo_ms=args.ttft_slo_ms,
                tpot_slo_ms=args.tpot_slo_ms,
            ),
        )
    if (
        args.isl is None
        or args.osl is None
        or args.prefix_hit_rate is None
        or args.accept_len is None
    ):
        parser.error(
            "Either --example-scenario or all of --isl, --osl, "
            "--prefix-hit-rate, and --accept-len are required."
        )
    return TrafficScenario(
        name="custom",
        isl=args.isl,
        osl=args.osl,
        prefix_hit_rate=args.prefix_hit_rate,
        accept_len=args.accept_len,
        think_time_s=args.think_time_s,
        concurrencies=(
            concurrencies if concurrencies is not None else EXAMPLE_CONCURRENCIES
        ),
        kv_working_set_turns=1.0,
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
    )


def _format_pool_axes(pool: PoolMeshPlan) -> str:
    """Format non-trivial parallelism axes for a pool summary label."""
    axes: list[str] = []
    if pool.tp > 1:
        axes.append(f"TP{pool.tp}")
    if pool.dp > 1:
        axes.append(f"DP{pool.dp}")
    if pool.ep > 1:
        axes.append(f"EP{pool.ep}")
    if pool.pp > 1:
        axes.append(f"PP{pool.pp}")
    if pool.pcp > 1:
        axes.append(f"PCP{pool.pcp}")
    if pool.kv_split > 1:
        axes.append(f"DCP{pool.kv_split}")
    return " x ".join(axes) if axes else "TP1"


def _write_warnings(warnings: Sequence[str]) -> None:
    """Write formatted warning lines to stderr."""
    for warning in warnings:
        sys.stderr.write(f"WARNING: {warning}\n")


def _dispatch_ceiling(args: argparse.Namespace, serving_plan: ServingPlan) -> int:
    """Write uncalibrated candidate ceiling rows as JSON array or JSON lines."""
    ceiling_rows = emit_ceiling_rows(serving_plan, top_k=args.top_k)
    if serving_plan.warnings:
        _write_warnings(serving_plan.warnings)
    if not ceiling_rows:
        if not serving_plan.warnings:
            _write_warnings((NO_FEASIBLE_LAYOUT_WARNING,))
        if args.json:
            sys.stdout.write("[]\n")
        return 1
    if args.json:
        sys.stdout.write(json.dumps(ceiling_rows, allow_nan=False) + "\n")
        return 0
    for ceiling_row in ceiling_rows:
        sys.stdout.write(json.dumps(ceiling_row, allow_nan=False) + "\n")
    return 0


def _parse_pd_ratio(
    raw_pd: str | None,
    parser: argparse.ArgumentParser,
) -> tuple[int, int] | None:
    """Parse optional --pd PREFILL:DECODE string into positive integer pair."""
    if raw_pd is None:
        return None
    parts = raw_pd.split(":")
    expected_parts = 2
    if len(parts) != expected_parts:
        parser.error(f"--pd must be PREFILL:DECODE, got {raw_pd!r}")
    try:
        prefill_instances, decode_instances = int(parts[0]), int(parts[1])
    except ValueError:
        parser.error(f"--pd must contain integer counts, got {raw_pd!r}")
    if prefill_instances < 1 or decode_instances < 1:
        parser.error(f"--pd counts must be >= 1, got {raw_pd!r}")
    return (prefill_instances, decode_instances)


def _write_solve_text_output(
    serving_plan: ServingPlan,
    ref_report: ReferenceLayoutReport | None = None,
    *,
    top_k: int | None = None,
) -> int:
    """Write solve warnings to stderr and formatted table rows to stdout."""
    _write_warnings(serving_plan.warnings)
    visible_candidates = slice_per_concurrency(serving_plan.candidates, top_k)
    if visible_candidates:
        sys.stdout.write(
            f"{'mode':<7}  {'c':>5}  {'chips':>5}  {'pareto':>6}  {'pools':<32}  "
            f"{'est_int_tok_s':>13}  {'tok_tput_chip':>13}  {'est_ttft_ms':>11}  "
            f"{'est_tpot_ms':>11}\n"
        )
        for point in visible_candidates:
            if point.prefill_pool is not None:
                pools = (
                    f"{point.prefill_pool.instances}x{point.prefill_pool.topology}"
                    f"({_format_pool_axes(point.prefill_pool)})P+"
                    f"{point.decode_pool.instances}x{point.decode_pool.topology}"
                    f"({_format_pool_axes(point.decode_pool)})D"
                )
            else:
                pools = (
                    f"1x{point.decode_pool.topology}"
                    f"({_format_pool_axes(point.decode_pool)})"
                )
            mark = "*" if point.is_pareto else ""
            sys.stdout.write(
                f"{point.mode:<7}  {point.concurrency:>5}  {point.total_chips:>5}  "
                f"{mark:>6}  {pools:<32}  {point.est_interactivity_tok_s:>13.2f}  "
                f"{point.token_throughput_per_chip:>13.1f}  "
                f"{point.est_ttft_ms:>11.1f}  {point.est_tpot_ms:>11.2f}\n"
            )
    else:
        sys.stdout.write(
            f"{'role':<8}  {'c':>5}  {'chips':>5}  {'pareto':>6}  {'pool':<28}  "
            f"{'tok_tput_chip':>13}  {'ttft_ms':>10}  {'tpot_ms':>10}\n"
        )
        visible_prefill = slice_per_concurrency(serving_plan.prefill_pools, top_k)
        visible_decode = slice_per_concurrency(serving_plan.decode_pools, top_k)
        for record in (*visible_prefill, *visible_decode):
            mark = "*" if record.is_pareto else ""
            pool_lbl = f"{record.pool.topology}({_format_pool_axes(record.pool)})"
            ttft_text = (
                f"{record.ttft_ms:>10.1f}"
                if record.ttft_ms is not None
                else f"{'-':>10}"
            )
            tpot_text = (
                f"{record.tpot_ms:>10.2f}"
                if record.tpot_ms is not None
                else f"{'-':>10}"
            )
            sys.stdout.write(
                f"{record.role:<8}  {record.concurrency:>5}  "
                f"{record.pool.chips_per_instance:>5}  {mark:>6}  {pool_lbl:<28}  "
                f"{record.token_throughput_per_chip:>13.1f}  "
                f"{ttft_text}  {tpot_text}\n"
            )
    _write_reference_summary(ref_report)
    return 0


def _dispatch_solve(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config: dict[str, object],
    *,
    model_ref: str | None = None,
) -> int:
    """Execute the solve subcommand and write JSON or table output."""
    scenario = _resolve_scenario(args, parser)
    pd_ratio = _parse_pd_ratio(args.pd, parser)
    has_pins = bool(args.prefill_topology and args.decode_topology and pd_ratio)
    if args.max_chips is None and args.max_instances is None and not has_pins:
        parser.error(
            "Either --max-chips, complete pins (--prefill-topology, "
            "--decode-topology, --pd), or --max-instances is required."
        )
    _require_positive_int("--max-chips", args.max_chips)
    if args.max_instances is not None and args.max_instances < MIN_INSTANCES:
        msg = f"--max-instances must be >= {MIN_INSTANCES}, got {args.max_instances}"
        raise ValueError(msg)
    if args.top_k is not None and args.top_k < MIN_TOP_K:
        msg = f"--top-k must be >= {MIN_TOP_K}, got {args.top_k}"
        raise ValueError(msg)
    _require_non_negative_finite("--hbm-reserve-gb", args.hbm_reserve_gb)
    resolved_dtypes = resolve_dtypes_from_args(args, config)
    chip, custom_topos = _resolve_chip_and_topos(args)
    chip = _apply_chip_overrides(chip, args)
    include_pairs = args.pairs or (
        args.cmd == "ceiling"
        and args.mode in ("auto", "disagg")
        and (args.max_chips is not None or has_pins)
    )
    parsed_ref: dict[str, dict[str, int]] | None = None
    if args.cmd == "solve" and args.reference_layout is not None:
        try:
            parsed_ref = parse_layout(args.reference_layout)
        except ValueError as err:
            parser.error(f"invalid --reference-layout: {err}")
    geometry = from_hf_config(config, model_ref=model_ref)
    solve_opts = SolveOptions(
        mode=args.mode,
        max_chips=args.max_chips,
        max_instances=args.max_instances,
        hbm_reserve_gb=args.hbm_reserve_gb,
        topologies=tuple(custom_topos.values()) if custom_topos else None,
        sort_by=args.sort_by,
        include_pairs=include_pairs,
        top_k=args.top_k,
        prefill_topology=args.prefill_topology,
        decode_topology=args.decode_topology,
        pd_ratio=pd_ratio,
    )
    serving_plan = solve_serving_plan(
        geometry, resolved_dtypes, chip, scenario, solve_opts
    )
    if args.cmd == "ceiling":
        return _dispatch_ceiling(args, serving_plan)
    src = args.model if args.model is not None else (model_ref or args.config)
    ref_report = _build_reference_report_for_cli(
        args, geometry, serving_plan, chip, solve_opts, parsed_ref=parsed_ref
    )
    payload = (
        build_solve_payload(
            serving_plan,
            chip=chip,
            dtypes=resolved_dtypes,
            model=src,
            top_k=args.top_k,
            topologies=custom_topos,
        )
        if (args.html is not None or args.json)
        else None
    )
    if payload is not None and ref_report is not None:
        payload = attach_reference_report(payload, ref_report)
    if args.html is not None and payload is not None:
        write_html(payload, args.html)
    if args.json and payload is not None:
        sys.stdout.write(format_solve_json(payload) + "\n")
        return 0
    return _write_solve_text_output(serving_plan, ref_report, top_k=args.top_k)


def _build_reference_report_for_cli(
    args: argparse.Namespace,
    geometry: ModelGeometry,
    serving_plan: ServingPlan,
    chip: ChipSpec,
    solve_opts: SolveOptions,
    *,
    parsed_ref: dict[str, dict[str, int]] | None = None,
) -> ReferenceLayoutReport | None:
    """Build optional reference-layout diagnostic report for solve CLI."""
    ref_layout = args.reference_layout
    if ref_layout is None:
        return None
    resolved_ref = parsed_ref if parsed_ref is not None else parse_layout(ref_layout)
    topos = _resolve_solve_topologies(solve_opts, chip=chip)
    rejected_by = explain_layout_rejection(geometry, resolved_ref, topos, solve_opts)
    return reference_layout_report(
        serving_plan,
        top_layouts(serving_plan, args.top_k),
        ref_layout,
        rejected_by,
    )


def _write_reference_summary(ref_report: ReferenceLayoutReport | None) -> None:
    """Write reference-layout summary line to stdout in text mode."""
    if ref_report is None:
        return
    if ref_report["found"]:
        concurrency = ref_report["concurrency"]
        concurrency_part = f"c={concurrency}, " if concurrency is not None else ""
        sys.stdout.write(
            f"reference {ref_report['layout']}: "
            f"{concurrency_part}"
            f"tpot_bound_rank={ref_report['rank_by_tpot_bound']}/"
            f"{ref_report['candidates']}, "
            f"tput_rank={ref_report['rank_by_estimated_throughput']}/"
            f"{ref_report['candidates']}\n"
        )
        return
    sys.stdout.write(
        f"reference {ref_report['layout']}: not found ({ref_report['rejected_by']})\n"
    )


def _apply_chip_overrides(chip: ChipSpec, args: argparse.Namespace) -> ChipSpec:
    """Apply explicit CLI bandwidth overrides to a ChipSpec."""
    if args.ici_bw_gbps is not None:
        return override_chip_ici_bw(chip, args.ici_bw_gbps * UNIT_GIGA)
    return chip


def _resolve_chip_and_topos(
    args: argparse.Namespace,
) -> tuple[ChipSpec, dict[str, SliceTopology] | None]:
    """Resolve (ChipSpec, optional topologies) from --chip-spec or --chip."""
    if args.chip_spec is not None:
        chip, topos = load_chip_spec(args.chip_spec)
        return chip, topos
    return CHIPS[args.chip], None


def _resolve_chip_spec(args: argparse.Namespace) -> ChipSpec:
    """Resolve ChipSpec from --chip-spec JSON path or built-in --chip name."""
    chip, _ = _resolve_chip_and_topos(args)
    return chip


def _parse_hf_overrides(raw: str | None) -> dict[str, object] | None:
    """Parse optional --hf-overrides JSON string into a dict."""
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        msg = f"--hf-overrides must be a JSON object, got {type(parsed).__name__}"
        raise TypeError(msg)
    return parsed


def resolve_dtypes_from_args(
    args: argparse.Namespace,
    config: dict[str, object] | None = None,
) -> Dtypes:
    """Apply optional CLI dtype overrides onto HF-derived defaults via replace."""
    dtypes = dtypes_from_hf_config(config) if config is not None else Dtypes()
    if args.weight_dtype is not None:
        dtypes = replace(
            dtypes,
            w_dense=WEIGHT_DTYPE_BYTES[args.weight_dtype],
            w_dense_source="flag",
            w_dense_dtype=args.weight_dtype,
        )
    if args.expert_weight_dtype is not None:
        dtypes = replace(
            dtypes,
            w_expert=EXPERT_DTYPE_BYTES[args.expert_weight_dtype],
            w_expert_source="flag",
            w_expert_dtype=args.expert_weight_dtype,
        )
    if args.aux_dtype is not None:
        dtypes = replace(dtypes, w_aux=WEIGHT_DTYPE_BYTES[args.aux_dtype])
    if args.kv_dtype is not None:
        dtypes = replace(
            dtypes,
            kv=KV_DTYPE_BYTES[args.kv_dtype],
            kv_source="flag",
            kv_dtype=args.kv_dtype,
        )
    if args.state_dtype is not None:
        dtypes = replace(
            dtypes,
            state=STATE_DTYPE_BYTES[args.state_dtype],
            state_source="flag",
            state_dtype=args.state_dtype,
        )
    if args.act is not None:
        if args.act in ("a4", "fp4", "nvfp4"):
            dtypes = replace(dtypes, act="a4")
        elif args.act in ("a8", "fp8"):
            dtypes = replace(dtypes, act="a8")
        else:
            dtypes = replace(dtypes, act="a16")
    return dtypes


def _build_common_plan_inputs(
    args: argparse.Namespace,
    config: dict[str, object] | None = None,
) -> tuple[Parallel, Dtypes, float]:
    """Build Parallel, Dtypes, and latency override from common CLI flags."""
    _require_non_negative_finite("--collective-latency-us", args.collective_latency_us)
    if args.context < 0:
        msg = f"--context must be >= 0, got {args.context}"
        raise ValueError(msg)
    if args.spec_window < 0:
        msg = f"--spec-window must be >= 0, got {args.spec_window}"
        raise ValueError(msg)
    parallel = Parallel(
        tp=args.tp,
        kv_split=args.kv_split,
        kv_merge=args.kv_merge,
        moe=args.moe,
        moe_dispatch=args.moe_dispatch,
        residual=args.residual,
        ar_algo=args.ar_algo,
    )
    dtypes = resolve_dtypes_from_args(args, config)
    latency_s = (args.collective_latency_us or 0.0) * UNIT_MICRO
    return parallel, dtypes, latency_s


def _dispatch_plan(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    config: dict[str, object],
    *,
    model_ref: str | None = None,
) -> int:
    """Execute the plan subcommand across target concurrencies."""
    if args.spec_window > 0 and args.accept_len is None:
        parser.error("--accept-len is required when --spec-window > 0")
    chip, topologies = _resolve_chip_and_topos(args)
    chip = _apply_chip_overrides(chip, args)
    parallel, dtypes, _ = _build_common_plan_inputs(args, config)
    concurrencies_list = list(_parse_concurrencies(args.concurrency))
    rows = plan(
        geometry=from_hf_config(config, model_ref=model_ref),
        chip=chip,
        parallel=parallel,
        dtypes=dtypes,
        request=ServingRequest(
            context=args.context,
            concurrencies=concurrencies_list,
            spec_window=args.spec_window,
            acceptance_length=args.accept_len,
            collective_latency_us=args.collective_latency_us,
        ),
        topologies=topologies,
    )
    if args.json:
        sys.stdout.write(json.dumps([asdict(row_item) for row_item in rows]) + "\n")
        return 0
    header = (
        f"{'c':>4}  {'rows':>6}  {'binding':<7}  "
        f"{'hard_floor_ms':>13}  {'serial_estimate_ms':>18}  "
        f"{'tpot_floor_ms':>13}  {'interactivity_ceil_tok_s':>24}  "
        f"{'output_tput_ceil_per_chip':>25}  {'flag':<7}\n"
    )
    sys.stdout.write(header)
    for row in rows:
        sys.stdout.write(
            f"{row.concurrency:>4}  {row.rows:>6}  {row.binding:<7}  "
            f"{row.hard_floor_ms:>13.3f}  {row.serial_estimate_ms:>18.3f}  "
            f"{row.tpot_floor_ms:>13.3f}  {row.interactivity_ceil_tok_s:>24.1f}  "
            f"{row.output_tput_ceil_per_chip:>25.1f}  {row.flag:<7}\n"
        )
    return 0


def _dispatch_explain(
    args: argparse.Namespace,
    config: dict[str, object],
    *,
    model_ref: str | None = None,
) -> int:
    """Execute the explain subcommand for per-module step breakdowns."""
    _require_positive_int("--batch", args.batch)
    _require_positive_int("--rows-per-seq", args.rows_per_seq)
    geometry = from_hf_config(config, model_ref=model_ref)
    chip = _resolve_chip_spec(args)
    chip = _apply_chip_overrides(chip, args)
    parallel, dtypes, latency_s = _build_common_plan_inputs(args, config)
    opts = FloorOptions(dtypes=dtypes, collective_latency_s=latency_s)
    floor_result = (
        spec_cycle_floor(
            geometry,
            chip,
            parallel,
            SpecShape(batch=args.batch, context=args.context, spec_k=args.spec_window),
            opts,
        )
        if args.spec_window > 0
        else step_floor(
            geometry=geometry,
            chip=chip,
            parallel=parallel,
            shape=StepShape(
                batch=args.batch,
                rows_per_seq=args.rows_per_seq,
                context=args.context,
            ),
            options=opts,
        )
    )
    if args.json:
        payload = [
            {
                **asdict(module_item),
                "ici_bytes": module_item.ici_bytes,
                "n_collectives": module_item.n_collectives,
                "chain_hops": module_item.chain_hops,
            }
            for module_item in floor_result.modules
        ]
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0
    sys.stdout.write(
        f"{'module':<10}  {'hbm_mb':>10}  {'fp8_gflops':>10}  "
        f"{'bf16_gflops':>11}  {'int8_gflops':>11}  {'fp4_gflops':>10}  "
        f"{'int4_gflops':>11}  {'vector_gflops':>13}  "
        f"{'collectives':>11}  {'wire_kb':>10}  {'hops':>6}\n"
    )
    for module in floor_result.modules:
        wire_kb = (
            sum(collective.wire_bytes for collective in module.collectives) / UNIT_KILO
        )
        sys.stdout.write(
            f"{module.name:<10}  {module.hbm_bytes / UNIT_MEGA:>10.2f}  "
            f"{module.fp8_flops / UNIT_GIGA:>10.2f}  "
            f"{module.bf16_flops / UNIT_GIGA:>11.2f}  "
            f"{module.int8_flops / UNIT_GIGA:>11.2f}  "
            f"{module.fp4_flops / UNIT_GIGA:>10.2f}  "
            f"{module.int4_flops / UNIT_GIGA:>11.2f}  "
            f"{module.vector_flops / UNIT_GIGA:>13.2f}  "
            f"{module.n_collectives:>11}  {wire_kb:>10.2f}  {module.chain_hops:>6}\n"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the roofline CLI subcommand."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    if args.cmd in ("plan", "explain") and (args.tp is None or args.tp <= 0):
        parser.error(f"--tp must be a positive integer (>= 1), got {args.tp}")
    src = args.model if args.model is not None else args.config
    try:
        overrides = _parse_hf_overrides(args.hf_overrides)
        config, identity = load_hf_config_with_identity(src, hf_overrides=overrides)
        model_ref = identity.hf_repo
        if args.cmd in ("solve", "ceiling"):
            return _dispatch_solve(args, parser, config, model_ref=model_ref)
        if args.cmd == "plan":
            return _dispatch_plan(args, parser, config, model_ref=model_ref)
        return _dispatch_explain(args, config, model_ref=model_ref)
    except (
        UnsupportedArchitectureError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
