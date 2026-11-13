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
"""CLI entrypoint for plan and explain subcommands."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace

from .chips import CHIPS
from .constants import (
    EXPERT_DTYPE_BYTES,
    KV_DTYPE_BYTES,
    STATE_DTYPE_BYTES,
    UNIT_GIGA,
    UNIT_KILO,
    UNIT_MEGA,
    UNIT_MICROS_PER_SECOND,
    WEIGHT_DTYPE_BYTES,
)
from .geometry import from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .modules import Dtypes, Parallel, StepShape
from .serving import ServingRequest, plan
from .step import FloorOptions, SpecShape, spec_cycle_floor, step_floor


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


def main(argv: list[str] | None = None) -> int:
    """Run the roofline CLI subcommand."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m vllm_torchtpu.roofline")
    sub = parser.add_subparsers(dest="cmd", required=True)

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

    if args.ici_bw_gbps is not None:
        chip = replace(chip, ici_bw=args.ici_bw_gbps * UNIT_GIGA)
    lat_s = (args.collective_latency_us or 0.0) * UNIT_MICROS_PER_SECOND

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

    if args.cmd == "plan":
        if args.spec_window > 0 and args.acceptance_length is None:
            parser.error("--acceptance-length is required when --spec-window > 0")
        concs_list = [int(x) for x in args.concurrency.split(",") if x.strip()]
        rows = plan(
            geom=geom,
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
            line = (
                f"{r.c:>4}  {r.rows:>6}  {r.binding:<7}  "
                f"{r.hard_floor_ms:>13.3f}  {r.serial_estimate_ms:>15.3f}  "
                f"{r.tpot_floor_ms:>13.3f}  {r.interactivity_ceil_tok_s:>24.1f}  "
                f"{r.output_tput_ceil_per_chip:>25.1f}  {r.flag:<7}\n"
            )
            sys.stdout.write(line)
        return 0

    opts = FloorOptions(dt=dt, collective_latency_s=lat_s)
    if args.spec_window > 0:
        sf = spec_cycle_floor(
            geom,
            chip,
            par,
            SpecShape(batch=args.batch, context=args.context, k=args.spec_window),
            opts,
        )
    else:
        sf = step_floor(
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
        wire_kb = sum(c.wire_bytes for c in m.collectives) / UNIT_KILO
        sys.stdout.write(
            f"{m.name:<10}  {m.hbm_bytes / UNIT_MEGA:>10.2f}  "
            f"{m.fp8_flops / UNIT_GIGA:>10.2f}  {m.bf16_flops / UNIT_GIGA:>11.2f}  "
            f"{m.n_collectives:>11}  {wire_kb:>10.2f}  {m.chain_hops:>6}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
