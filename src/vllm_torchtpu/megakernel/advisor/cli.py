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
"""Command-line interface for the TPU megakernel fusion-scope advisor."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from vllm_torchtpu.megakernel import errors
from vllm_torchtpu.megakernel.advisor import (
    MoEMode,
    Phase,
    WorkloadSpec,
    advise,
    advise_regimes,
    recommend_parallelism,
    resolve_chip_spec,
)
from vllm_torchtpu.megakernel.advisor.models import resolve_model_profile
from vllm_torchtpu.megakernel.advisor.report import (
    format_advisor_summary,
    format_regime_table,
    write_advisor_html,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for single-phase and regime-sweep queries."""
    parser = argparse.ArgumentParser(prog="python -m vllm_torchtpu.megakernel.advisor")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Path to local HuggingFace config.json")
    source.add_argument(
        "--model", help="Registered model preset, HF repo ID, or local directory"
    )

    for fld in fields(WorkloadSpec):
        if fld.type in ("int", "float", "float | None"):
            arg_type = int if fld.type == "int" else float
            default_val = None if fld.name in ("batch", "tp", "ep") else fld.default
            parser.add_argument(
                f"--{fld.name.replace('_', '-')}",
                type=arg_type,
                default=default_val,
            )

    for dtype_flag in (
        "weight_dtype",
        "expert_dtype",
        "kv_dtype",
        "state_dtype",
    ):
        parser.add_argument(f"--{dtype_flag.replace('_', '-')}", type=str, default=None)

    parser.add_argument("--chip", type=str, default="tpu7x")
    parser.add_argument("--ici-bw-gbps", type=float, default=None)
    parser.add_argument("--collective-latency-us", type=float, default=None)
    parser.add_argument("--phase", choices=get_args(Phase), default="decode")
    parser.add_argument(
        "--moe-mode", choices=get_args(MoEMode), default="token_masked_vmem"
    )
    parser.add_argument("--no-vmem-alias", dest="vmem_alias", action="store_false")
    for bool_flag in (
        "--megacore",
        "--vmem-fit-headroom",
        "--json",
        "--regime-sweep",
    ):
        parser.add_argument(bool_flag, action="store_true")
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Write a self-contained inline-SVG HTML regime report to this path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the megakernel advisor CLI."""
    args = build_arg_parser().parse_args(argv)
    if args.config and not Path(args.config).is_file():
        msg = f"Missing config file: {args.config!r}"
        raise errors.MegakernelConfigError(msg)

    dtypes = (
        args.weight_dtype,
        args.expert_dtype,
        args.kv_dtype,
        args.state_dtype,
    )
    target = Path(args.config) if args.config else args.model
    profile = resolve_model_profile(target, *dtypes)
    chip_spec = resolve_chip_spec(
        args.chip,
        megacore=args.megacore,
        ici_bw_gbps=args.ici_bw_gbps,
        collective_latency_us=args.collective_latency_us,
    )

    auto_mesh = args.tp is None
    if auto_mesh:
        rec = recommend_parallelism(profile, chip_spec, dp=args.dp, pp=args.pp)
        tp_val = rec.tp
        ep_val = args.ep if args.ep is not None else rec.ep
    else:
        tp_val = args.tp
        ep_val = args.ep if args.ep is not None else 1

    if args.html or args.regime_sweep:
        sweep = advise_regimes(
            profile,
            chip_spec,
            tp=tp_val if not auto_mesh else None,
            ep=args.ep,
            dp=args.dp,
            pp=args.pp,
            dcp=args.dcp,
            pcp=args.pcp,
            vmem_alias=args.vmem_alias,
            vmem_fit_headroom=args.vmem_fit_headroom,
            op_overhead_us=args.op_overhead_us,
            vmem_budget_mib=args.vmem_budget_mib,
            dma_buffers=args.dma_buffers,
        )
        if args.html:
            write_advisor_html(sweep, args.html)
        if args.regime_sweep:
            out = (
                json.dumps(sweep.to_dict(), indent=2)
                if args.json
                else format_regime_table(sweep)
            )
            sys.stdout.write(out + "\n")
            return 0

    batch_val = (
        args.batch if args.batch is not None else (1 if args.phase == "prefill" else 16)
    )
    wl_kwargs = {
        fld.name: getattr(args, fld.name)
        for fld in fields(WorkloadSpec)
        if hasattr(args, fld.name)
    }
    wl_kwargs.update({"batch": batch_val, "tp": tp_val, "ep": ep_val})
    report = advise(profile, WorkloadSpec(**wl_kwargs), chip_spec)
    out = (
        json.dumps(report.to_dict(), indent=2)
        if args.json
        else format_advisor_summary(report)
    )
    sys.stdout.write(out + "\n")
    return 0
