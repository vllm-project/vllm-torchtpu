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
"""CLI entrypoint for solve, plan, and explain subcommands."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace

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
    STATE_DTYPE_BYTES,
    UNIT_GIGA,
    UNIT_KILO,
    UNIT_MEGA,
    UNIT_MICRO,
    WEIGHT_DTYPE_BYTES,
)
from .geometry import UnsupportedArchitectureError, from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config_with_identity
from .modules import Dtypes, Parallel, StepShape
from .serving import ServingRequest, plan
from .solver import (
    EXAMPLE_SCENARIOS,
    PoolMeshPlan,
    ScenarioOverrides,
    ServingPlan,
    SolveOptions,
    TrafficScenario,
    build_example_scenario,
    solve_serving_plan,
)
from .step import FloorOptions, SpecShape, spec_cycle_floor, step_floor


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
    parser.add_argument("--max-chips", type=int, default=None)
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
    """Build argument parser for solve, plan, and explain subcommands."""
    parser = argparse.ArgumentParser(prog="python -m vllm_torchtpu.roofline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    solve_parser = sub.add_parser("solve")
    _add_scenario_args(solve_parser)

    plan_parser = sub.add_parser("plan")
    _add_common_args(plan_parser)
    plan_parser.add_argument("--concurrency", default="1,4,16,64")
    plan_parser.add_argument("--spec-window", type=int, default=0)
    plan_parser.add_argument("--accept-len", type=float, default=None)

    explain_parser = sub.add_parser("explain")
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
        or args.think_time_s is None
    ):
        parser.error(
            "Either --example-scenario or all of --isl, --osl, "
            "--prefix-hit-rate, --accept-len, and --think-time-s "
            "are required."
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


def _write_solve_text_output(serving_plan: ServingPlan) -> int:
    """Write solve warnings to stderr and formatted table rows to stdout."""
    for warning in serving_plan.warnings:
        sys.stderr.write(f"WARNING: {warning}\n")
    if serving_plan.candidates:
        sys.stdout.write(
            f"{'mode':<7}  {'c':>5}  {'chips':>5}  {'pareto':>6}  {'pools':<32}  "
            f"{'est_int_tok_s':>13}  {'tok_tput_chip':>13}  {'est_ttft_ms':>11}  "
            f"{'est_tpot_ms':>11}\n"
        )
        for point in serving_plan.candidates:
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
        for record in (*serving_plan.prefill_pools, *serving_plan.decode_pools):
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
    if args.max_chips is None and args.max_instances is None:
        parser.error("Either --max-chips or --max-instances is required.")
    _require_positive_int("--max-chips", args.max_chips)
    _require_positive_int("--max-instances", args.max_instances)
    _require_positive_int("--top-k", args.top_k)
    _require_non_negative_finite("--hbm-reserve-gb", args.hbm_reserve_gb)
    resolved_dtypes = resolve_dtypes_from_args(args, config)
    chip, custom_topos = _resolve_chip_and_topos(args)
    chip = _apply_chip_overrides(chip, args)
    serving_plan = solve_serving_plan(
        from_hf_config(config, model_ref=model_ref),
        resolved_dtypes,
        chip,
        scenario,
        SolveOptions(
            mode=args.mode,
            max_chips=args.max_chips,
            max_instances=args.max_instances,
            hbm_reserve_gb=args.hbm_reserve_gb,
            topologies=tuple(custom_topos.values()) if custom_topos else None,
            sort_by=args.sort_by,
            include_pairs=args.pairs,
            top_k=args.top_k,
        ),
    )
    if args.json:
        sys.stdout.write(json.dumps(asdict(serving_plan), allow_nan=False) + "\n")
        return 0
    return _write_solve_text_output(serving_plan)


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
        dtypes = replace(dtypes, state=STATE_DTYPE_BYTES[args.state_dtype])
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


def main(argv: list[str] | None = None) -> int:
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
        if args.cmd == "solve":
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
