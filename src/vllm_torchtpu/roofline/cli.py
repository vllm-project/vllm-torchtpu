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

from .chips import CHIPS
from .emit import emit_ceiling_rows, emit_launch_args
from .geometry import from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .modules import Dtypes, Parallel, StepShape
from .serving import plan
from .solver import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SCENARIOS,
    EXAMPLE_SLO_TIERS,
    TrafficScenario,
    build_example_scenario,
    solve_serving_plan,
)
from .step import spec_cycle_floor, step_floor

_WEIGHT_BYTES: dict[str, float] = {"bf16": 2.0, "fp8": 1.0}
_EXPERT_BYTES: dict[str, float] = {
    "bf16": 2.0,
    "fp8": 1.0,
    "nvfp4": 0.5625,
    "mxfp4": 0.53125,
}
_KV_BYTES: dict[str, float] = {"bf16": 2.0, "fp8": 1.0}
_STATE_BYTES: dict[str, float] = {"fp32": 4.0, "bf16": 2.0}


def _add_model_source(p: argparse.ArgumentParser) -> None:
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--config", help="Path to HF config.json")
    grp.add_argument("--model", help="HF model ID, directory, or config.json")


def _add_common_args(p: argparse.ArgumentParser, *, tp_required: bool = True) -> None:
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
    p.add_argument("--weight-dtype", default="fp8", choices=("bf16", "fp8"))
    p.add_argument(
        "--expert-dtype",
        default="fp8",
        choices=("bf16", "fp8", "nvfp4", "mxfp4"),
    )
    p.add_argument("--aux-dtype", default="bf16", choices=("bf16", "fp8"))
    p.add_argument("--kv-dtype", default="fp8", choices=("bf16", "fp8"))
    p.add_argument("--state-dtype", default="fp32", choices=("fp32", "bf16"))
    p.add_argument("--act", default="a8", choices=("a8", "a16"))
    p.add_argument("--ici-bw-gbps", type=float, default=None)
    p.add_argument("--collective-latency-us", type=float, default=None)
    p.add_argument("--json", action="store_true")


def _add_scenario_args(p: argparse.ArgumentParser) -> None:
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


def main(argv: list[str] | None = None) -> int:
    """Run the roofline CLI subcommand."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
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

    args = parser.parse_args(raw_argv)
    src = args.model if args.model is not None else args.config
    cfg = load_hf_config(src)
    geom = from_hf_config(cfg)
    chip = CHIPS[args.chip]

    if args.cmd in ("solve", "emit", "ceiling"):
        concs = (
            tuple(int(x) for x in args.concurrency.split(",") if x.strip())
            if args.concurrency
            else None
        )
        if args.example_scenario is not None:
            scen = build_example_scenario(
                args.example_scenario,
                isl=args.isl,
                osl=args.osl,
                concurrencies=concs,
                prefix_hit_rate=args.prefix_hit_rate,
                accept_len=args.accept_len,
                think_time_s=args.think_time_s,
                slo_tier=args.example_slo_tier,
                ttft_slo_ms=args.ttft_slo_ms,
                tpot_slo_ms=args.tpot_slo_ms,
            )
        else:
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
            scen = TrafficScenario(
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
        if args.max_chips is None and args.max_instances is None:
            parser.error("Either --max-chips or --max-instances is required.")
        dt_auto = dtypes_from_hf_config(cfg)
        splan = solve_serving_plan(
            geom,
            dt_auto,
            chip,
            scen,
            mode=args.mode,
            max_chips=args.max_chips,
            max_instances=args.max_instances,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        if args.cmd == "ceiling":
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

        if args.cmd == "emit":
            emitted = emit_launch_args(
                splan,
                dtypes=dt_auto,
                model=args.model,
            )
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

        if args.json:
            sys.stdout.write(json.dumps(asdict(splan)) + "\n")
            return 0
        for w in splan.warnings:
            sys.stdout.write(f"WARNING: {w}\n")
        sys.stdout.write(
            f"{'mode':<7}  {'c':>5}  {'chips':>5}  {'pools':<24}  "
            f"{'p90_int_tok_s':>13}  {'tok_tput_chip':>13}  {'p90_ttft_ms':>11}  "
            f"{'p90_tpot_ms':>11}\n"
        )
        for pt in splan.frontier:
            if pt.prefill_pool is not None:
                pools = (
                    f"{pt.prefill_pool.instances}x{pt.prefill_pool.topology}P+"
                    f"{pt.decode_pool.instances}x{pt.decode_pool.topology}D"
                )
            else:
                pools = f"1x{pt.decode_pool.topology}(TP{pt.decode_pool.tp})"
            sys.stdout.write(
                f"{pt.mode:<7}  {pt.concurrency:>5}  {pt.total_chips:>5}  "
                f"{pools:<24}  {pt.p90_interactivity_tok_s:>13.2f}  "
                f"{pt.token_throughput_per_chip:>13.1f}  "
                f"{pt.p90_ttft_ms:>11.1f}  {pt.p90_tpot_ms:>11.2f}\n"
            )
        return 0

    if args.ici_bw_gbps is not None:
        chip = replace(chip, ici_bw=args.ici_bw_gbps * 1e9)
    lat_s = (args.collective_latency_us or 0.0) * 1e-6

    if args.tp is None or args.tp <= 0:
        raise ValueError(f"--tp must be >= 1, got {args.tp}")
    par = Parallel(
        tp=args.tp,
        kv_split=args.kv_split,
        kv_merge=args.kv_merge,
        moe=args.moe,
        moe_dispatch=args.moe_dispatch,
        residual=args.residual,
        ar_algo=args.ar_algo,
    )
    dt = Dtypes(
        w_dense=_WEIGHT_BYTES[args.weight_dtype],
        w_expert=_EXPERT_BYTES[args.expert_dtype],
        w_aux=_WEIGHT_BYTES[args.aux_dtype],
        kv=_KV_BYTES[args.kv_dtype],
        state=_STATE_BYTES[args.state_dtype],
        act=args.act,
    )

    if args.cmd == "plan":
        if args.spec_window > 0 and args.acceptance_length is None:
            parser.error("--acceptance-length is required when --spec-window > 0")
        concs_list = [int(x) for x in args.concurrency.split(",") if x.strip()]
        rows = plan(
            geom=geom,
            chip=chip,
            par=par,
            dt=dt,
            context=args.context,
            concurrencies=concs_list,
            spec_window=args.spec_window,
            acceptance_length=args.acceptance_length,
            collective_latency_us=args.collective_latency_us,
        )
        if args.json:
            sys.stdout.write(json.dumps([asdict(r) for r in rows]) + "\n")
            return 0
        hdr = (
            f"{'c':>4}  {'rows':>6}  {'binding':<7}  "
            f"{'hard_floor_ms':>13}  {'serial_floor_ms':>15}  "
            f"{'tpot_floor_ms':>13}  {'interactivity_ceil_tok_s':>24}  "
            f"{'output_tput_ceil_per_chip':>25}  {'flag':<7}\n"
        )
        sys.stdout.write(hdr)
        for r in rows:
            line = (
                f"{r.c:>4}  {r.rows:>6}  {r.binding:<7}  "
                f"{r.hard_floor_ms:>13.3f}  {r.serial_floor_ms:>15.3f}  "
                f"{r.tpot_floor_ms:>13.3f}  {r.interactivity_ceil_tok_s:>24.1f}  "
                f"{r.output_tput_ceil_per_chip:>25.1f}  {r.flag:<7}\n"
            )
            sys.stdout.write(line)
        return 0

    if args.spec_window > 0:
        sf = spec_cycle_floor(
            geom,
            chip,
            par,
            args.batch,
            args.context,
            args.spec_window,
            dt,
            collective_latency_s=lat_s,
        )
    else:
        sf = step_floor(
            geom=geom,
            chip=chip,
            par=par,
            shape=StepShape(
                batch=args.batch,
                rows_per_seq=args.rows_per_seq,
                context=args.context,
            ),
            dt=dt,
            collective_latency_s=lat_s,
        )
    if args.json:
        sys.stdout.write(
            json.dumps(
                [
                    {
                        **asdict(m),
                        "ici_bytes": m.ici_bytes,
                        "n_collectives": m.n_collectives,
                        "chain_hops": m.chain_hops,
                    }
                    for m in sf.modules
                ]
            )
            + "\n"
        )
        return 0
    sys.stdout.write(
        f"{'module':<10}  {'hbm_mb':>10}  {'fp8_gflops':>10}  "
        f"{'bf16_gflops':>11}  {'collectives':>11}  {'wire_kb':>10}  {'hops':>6}\n"
    )
    for m in sf.modules:
        wire_kb = sum(c.wire_bytes for c in m.collectives) / 1e3
        sys.stdout.write(
            f"{m.name:<10}  {m.hbm_bytes / 1e6:>10.2f}  "
            f"{m.fp8_flops / 1e9:>10.2f}  {m.bf16_flops / 1e9:>11.2f}  "
            f"{m.n_collectives:>11}  {wire_kb:>10.2f}  {m.chain_hops:>6}\n"
        )
    return 0
