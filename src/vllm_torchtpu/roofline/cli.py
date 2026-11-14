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
"""CLI entrypoint for solve, emit, ceiling, plan, and explain subcommands."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

from .chips import CHIPS
from .constants import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SLO_TIERS,
    EXPERT_DTYPE_BYTES,
    KV_DTYPE_BYTES,
    STATE_DTYPE_BYTES,
    UNIT_GIGA,
    UNIT_KILO,
    UNIT_MEGA,
    UNIT_MICROS_PER_SECOND,
    WEIGHT_DTYPE_BYTES,
)
from .emit import (
    build_solve_payload,
    emit_ceiling_rows,
    emit_launch_args,
)
from .geometry import from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .modules import Dtypes, Parallel, StepShape
from .serving import ServingRequest, plan
from .solver import (
    EXAMPLE_SCENARIOS,
    ScenarioOverrides,
    ServingPlan,
    SolveOptions,
    TrafficScenario,
    build_example_scenario,
    solve_serving_plan,
)
from .step import FloorOptions, SpecShape, spec_cycle_floor, step_floor

if TYPE_CHECKING:
    from collections.abc import Sequence


def _add_model_source(p: argparse.ArgumentParser) -> None:
    """Add mutually exclusive --config or --model flags to parser."""
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--config", help="Path to HF config.json")
    grp.add_argument("--model", help="HF model ID, directory, or config.json")


def _add_common_args(p: argparse.ArgumentParser, *, tp_required: bool = True) -> None:
    """Add common serving and parallel execution flags to parser."""
    _add_model_source(p)
    p.add_argument("--chip", default="tpu7x", choices=sorted(CHIPS))
    p.add_argument("--tp", type=int, required=tp_required, default=None)
    p.add_argument("--context", type=int, default=0)
    p.add_argument("--kv-split", type=int, default=1)
    p.add_argument("--kv-merge", default="ag_rs", choices=("ag_rs", "exchange"))
    p.add_argument("--moe", default="tp_slice", choices=("tp_slice", "ep"))
    p.add_argument(
        "--moe-dispatch",
        default="allreduce",
        choices=("allreduce", "a2a_push", "ag_ep", "fused_rs"),
    )
    p.add_argument("--residual", default="replicated", choices=("replicated", "sp"))
    p.add_argument(
        "--ar-algo",
        default="one_shot",
        choices=("one_shot", "two_shot", "ring"),
    )
    p.add_argument("--weight-dtype", default=None, choices=("bf16", "fp8"))
    p.add_argument(
        "--expert-dtype",
        default=None,
        choices=("bf16", "fp8", "nvfp4", "mxfp4"),
    )
    p.add_argument("--aux-dtype", default=None, choices=("bf16", "fp8"))
    p.add_argument("--kv-dtype", default=None, choices=("bf16", "fp8"))
    p.add_argument("--state-dtype", default=None, choices=("fp32", "bf16"))
    p.add_argument(
        "--act",
        "--act-dtype",
        dest="act",
        default="a8",
        choices=("a8", "a16", "fp8", "bf16", "bfloat16"),
    )
    p.add_argument("--ici-bw-gbps", type=float, default=None)
    p.add_argument("--collective-latency-us", type=float, default=None)
    p.add_argument("--json", action="store_true")


def _add_scenario_args(p: argparse.ArgumentParser) -> None:
    """Add traffic scenario, SLO, and instance constraint flags."""
    _add_model_source(p)
    p.add_argument("--chip", default="tpu7x", choices=sorted(CHIPS))
    p.add_argument(
        "--example-scenario", default=None, choices=sorted(EXAMPLE_SCENARIOS)
    )
    p.add_argument("--mode", default="auto", choices=("auto", "mixed", "disagg"))
    p.add_argument("--max-chips", type=int, default=None)
    p.add_argument("--max-instances", type=int, default=None)
    p.add_argument(
        "--example-slo-tier", default="none", choices=sorted(EXAMPLE_SLO_TIERS)
    )
    p.add_argument("--ttft-slo-ms", type=float, default=None)
    p.add_argument("--tpot-slo-ms", type=float, default=None)
    p.add_argument("--isl", type=int, default=None)
    p.add_argument("--osl", type=int, default=None)
    p.add_argument("--concurrency", default=None)
    p.add_argument("--prefix-hit-rate", type=float, default=None)
    p.add_argument("--accept-len", type=float, default=None)
    p.add_argument("--think-time-s", type=float, default=None)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    p.add_argument("--json", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser for solve, emit, ceiling, plan, and explain."""
    parser = argparse.ArgumentParser(prog="python -m vllm_torchtpu.roofline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_solve = sub.add_parser("solve")
    _add_scenario_args(p_solve)

    p_emit = sub.add_parser("emit")
    _add_scenario_args(p_emit)

    p_ceil = sub.add_parser("ceiling")
    _add_scenario_args(p_ceil)

    p_plan = sub.add_parser("plan")
    _add_common_args(p_plan)
    p_plan.add_argument("--concurrency", default="1,4,16,64")
    p_plan.add_argument("--spec-window", type=int, default=0)
    p_plan.add_argument("--acceptance-length", type=float, default=None)

    p_exp = sub.add_parser("explain")
    _add_common_args(p_exp)
    p_exp.add_argument("--batch", type=int, default=1)
    p_exp.add_argument("--rows-per-seq", type=int, default=1)
    p_exp.add_argument("--spec-window", type=int, default=0)
    return parser


def _resolve_scenario(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> TrafficScenario:
    """Validate CLI flags and build the target TrafficScenario."""
    concs = (
        tuple(int(x) for x in args.concurrency.split(",") if x.strip())
        if args.concurrency
        else None
    )
    if args.example_scenario is not None:
        return build_example_scenario(
            args.example_scenario,
            slo_tier=args.example_slo_tier,
            overrides=ScenarioOverrides(
                isl=args.isl,
                osl=args.osl,
                concurrencies=concs,
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
        concurrencies=concs if concs is not None else EXAMPLE_CONCURRENCIES,
        kv_working_set_turns=1.0,
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
    )


def _dispatch_ceiling(args: argparse.Namespace, splan: ServingPlan) -> int:
    """Write Pareto ceiling rows as JSON array or JSON lines."""
    rows_ceil = emit_ceiling_rows(splan)
    if not rows_ceil:
        for w in splan.warnings:
            sys.stderr.write(f"WARNING: {w}\n")
        if args.json:
            sys.stdout.write("[]\n")
        return 1
    if args.json:
        sys.stdout.write(json.dumps(rows_ceil) + "\n")
        return 0
    for rc_row in rows_ceil:
        sys.stdout.write(json.dumps(rc_row) + "\n")
    return 0


def _dispatch_emit(
    args: argparse.Namespace, splan: ServingPlan, dt_auto: Dtypes
) -> int:
    """Write emitted vLLM serve launch commands or JSON payload."""
    emitted = emit_launch_args(splan, dtypes=dt_auto, model=args.model)
    if not emitted["commands"]:
        for w in emitted["warnings"]:
            sys.stderr.write(f"WARNING: {w}\n")
        if args.json:
            sys.stdout.write(json.dumps(emitted) + "\n")
        return 1
    if args.json:
        sys.stdout.write(json.dumps(emitted) + "\n")
        return 0
    for cmd_line in emitted["commands"]:
        sys.stdout.write(cmd_line + "\n")
    return 0


def _dispatch_solve(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    cfg: dict[str, object],
) -> int:
    """Execute the solve, emit, or ceiling subcommands."""
    scen = _resolve_scenario(args, parser)
    if args.max_chips is None and args.max_instances is None:
        parser.error("Either --max-chips or --max-instances is required.")
    src = args.model if args.model is not None else args.config
    dt_auto = dtypes_from_hf_config(cfg)
    splan = solve_serving_plan(
        from_hf_config(cfg),
        dt_auto,
        CHIPS[args.chip],
        scen,
        SolveOptions(
            mode=args.mode,
            max_chips=args.max_chips,
            max_instances=args.max_instances,
            gpu_memory_utilization=args.gpu_memory_utilization,
        ),
    )
    if args.cmd == "ceiling":
        return _dispatch_ceiling(args, splan)
    if args.cmd == "emit":
        return _dispatch_emit(args, splan, dt_auto)
    if args.json:
        payload = build_solve_payload(
            splan,
            chip=CHIPS[args.chip],
            dtypes=dt_auto,
            model=src,
        )
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0
    for w in splan.warnings:
        sys.stdout.write(f"WARNING: {w}\n")
    sys.stdout.write(
        f"{'mode':<7}  {'c':>5}  {'chips':>5}  {'pools':<24}  "
        f"{'p90_int_tok_s':>13}  {'tok_tput_chip':>13}  {'p90_ttft_ms':>11}  "
        f"{'p90_tpot_ms':>11}\n"
    )
    for pt in splan.frontier:
        pools = (
            f"{pt.prefill_pool.instances}x{pt.prefill_pool.topology}P+"
            f"{pt.decode_pool.instances}x{pt.decode_pool.topology}D"
            if pt.prefill_pool is not None
            else f"1x{pt.decode_pool.topology}(TP{pt.decode_pool.tp})"
        )
        sys.stdout.write(
            f"{pt.mode:<7}  {pt.concurrency:>5}  {pt.total_chips:>5}  "
            f"{pools:<24}  {pt.p90_interactivity_tok_s:>13.2f}  "
            f"{pt.token_throughput_per_chip:>13.1f}  "
            f"{pt.p90_ttft_ms:>11.1f}  {pt.p90_tpot_ms:>11.2f}\n"
        )
    return 0


def _build_common_plan_inputs(
    args: argparse.Namespace,
    cfg: dict[str, object] | None = None,
) -> tuple[Parallel, Dtypes, float]:
    """Build Parallel, Dtypes, and latency override from common CLI flags."""
    if args.tp is None or args.tp <= 0:
        msg = f"--tp must be >= 1, got {args.tp}"
        raise ValueError(msg)
    par = Parallel(
        tp=args.tp,
        kv_split=args.kv_split,
        kv_merge=args.kv_merge,
        moe=args.moe,
        moe_dispatch=args.moe_dispatch,
        residual=args.residual,
        ar_algo=args.ar_algo,
    )
    base_dt = dtypes_from_hf_config(cfg) if cfg is not None else Dtypes()
    dt = Dtypes(
        w_dense=(
            WEIGHT_DTYPE_BYTES[args.weight_dtype]
            if args.weight_dtype is not None
            else base_dt.w_dense
        ),
        w_expert=(
            EXPERT_DTYPE_BYTES[args.expert_dtype]
            if args.expert_dtype is not None
            else base_dt.w_expert
        ),
        w_aux=(
            WEIGHT_DTYPE_BYTES[args.aux_dtype]
            if args.aux_dtype is not None
            else base_dt.w_aux
        ),
        kv=(KV_DTYPE_BYTES[args.kv_dtype] if args.kv_dtype is not None else base_dt.kv),
        state=(
            STATE_DTYPE_BYTES[args.state_dtype]
            if args.state_dtype is not None
            else base_dt.state
        ),
        act="a8" if args.act in ("a8", "fp8") else "a16",
    )
    lat_s = (args.collective_latency_us or 0.0) * UNIT_MICROS_PER_SECOND
    return par, dt, lat_s


def _dispatch_plan(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    cfg: dict[str, object],
) -> int:
    """Execute the plan subcommand across target concurrencies."""
    if args.spec_window > 0 and args.acceptance_length is None:
        parser.error("--acceptance-length is required when --spec-window > 0")
    chip = CHIPS[args.chip]
    if args.ici_bw_gbps is not None:
        chip = replace(chip, ici_bw=args.ici_bw_gbps * UNIT_GIGA)
    par, dt, _ = _build_common_plan_inputs(args, cfg)
    concs_list = [int(x) for x in args.concurrency.split(",") if x.strip()]
    rows = plan(
        geom=from_hf_config(cfg),
        chip=chip,
        parallel=par,
        dt=dt,
        request=ServingRequest(
            context=args.context,
            concurrencies=concs_list,
            spec_window=args.spec_window,
            acceptance_length=args.acceptance_length,
            collective_latency_us=args.collective_latency_us,
        ),
    )
    if args.json:
        sys.stdout.write(json.dumps([asdict(r) for r in rows]) + "\n")
        return 0
    hdr = (
        f"{'c':>4}  {'rows':>6}  {'binding':<7}  "
        f"{'hard_floor_ms':>13}  {'serial_estimate_ms':>15}  "
        f"{'tpot_floor_ms':>13}  {'interactivity_ceil_tok_s':>24}  "
        f"{'output_tput_ceil_per_chip':>25}  {'flag':<7}\n"
    )
    sys.stdout.write(hdr)
    for r in rows:
        sys.stdout.write(
            f"{r.c:>4}  {r.rows:>6}  {r.binding:<7}  "
            f"{r.hard_floor_ms:>13.3f}  {r.serial_estimate_ms:>15.3f}  "
            f"{r.tpot_floor_ms:>13.3f}  {r.interactivity_ceil_tok_s:>24.1f}  "
            f"{r.output_tput_ceil_per_chip:>25.1f}  {r.flag:<7}\n"
        )
    return 0


def _dispatch_explain(
    args: argparse.Namespace,
    cfg: dict[str, object],
) -> int:
    """Execute the explain subcommand for per-module step breakdowns."""
    geom = from_hf_config(cfg)
    chip = CHIPS[args.chip]
    if args.ici_bw_gbps is not None:
        chip = replace(chip, ici_bw=args.ici_bw_gbps * UNIT_GIGA)
    par, dt, lat_s = _build_common_plan_inputs(args, cfg)
    opts = FloorOptions(dt=dt, collective_latency_s=lat_s)
    sf = (
        spec_cycle_floor(
            geom,
            chip,
            par,
            SpecShape(batch=args.batch, context=args.context, k=args.spec_window),
            opts,
        )
        if args.spec_window > 0
        else step_floor(
            geom=geom,
            chip=chip,
            parallel=par,
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
                **asdict(m),
                "ici_bytes": m.ici_bytes,
                "n_collectives": m.n_collectives,
                "chain_hops": m.chain_hops,
            }
            for m in sf.modules
        ]
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0
    sys.stdout.write(
        f"{'module':<10}  {'hbm_mb':>10}  {'fp8_gflops':>10}  "
        f"{'bf16_gflops':>11}  {'collectives':>11}  {'wire_kb':>10}  {'hops':>6}\n"
    )
    for m in sf.modules:
        wire_kb = sum(c.wire_bytes for c in m.collectives) / UNIT_KILO
        sys.stdout.write(
            f"{m.name:<10}  {m.hbm_bytes / UNIT_MEGA:>10.2f}  "
            f"{m.fp8_flops / UNIT_GIGA:>10.2f}  {m.bf16_flops / UNIT_GIGA:>11.2f}  "
            f"{m.n_collectives:>11}  {wire_kb:>10.2f}  {m.chain_hops:>6}\n"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the roofline CLI subcommand."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(raw_argv)
    src = args.model if args.model is not None else args.config
    cfg = load_hf_config(src)
    if args.cmd in ("solve", "emit", "ceiling"):
        return _dispatch_solve(args, parser, cfg)
    if args.cmd == "plan":
        return _dispatch_plan(args, parser, cfg)
    return _dispatch_explain(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
