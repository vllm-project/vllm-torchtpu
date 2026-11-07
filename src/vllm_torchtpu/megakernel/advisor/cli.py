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

from .. import errors
from ..advisor.models import resolve_model_profile
from . import (
    DEPLOYMENT_PRESETS,
    MoEMode,
    ObjectiveMode,
    Phase,
    WorkloadSpec,
    advise,
    advise_regimes,
    calibrate_trace,
    recommend_parallelism,
    resolve_chip_spec,
)
from .report import (
    format_advisor_summary,
    format_regime_table,
    write_advisor_html,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _parse_layer_groups(raw: str | None) -> tuple[int, ...]:
    """Parse a comma-separated layer-group string into a tuple of ints."""
    if not raw:
        return ()
    try:
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        msg = f"Invalid --layer-groups value {raw!r}; expected comma-separated integers"
        raise errors.MegakernelConfigError(msg) from exc


def _emit_json(payload: dict[str, object], dest: str) -> None:
    """Write formatted JSON to stdout when `dest == '-'`, else to file `dest`."""
    rendered = json.dumps(payload, indent=2) + "\n"
    if dest == "-":
        sys.stdout.write(rendered)
        return
    out_path = Path(dest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")


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
    parser.add_argument(
        "--chip-spec",
        type=str,
        default=None,
        help="Path to JSON hardware chip specification file",
    )
    parser.add_argument(
        "--measurements",
        type=str,
        default=None,
        help="Path to calibrated advisor.measurements.v1 JSON file",
    )
    parser.add_argument("--ici-bw-gbps", type=float, default=None)
    parser.add_argument("--collective-latency-us", type=float, default=None)
    parser.add_argument("--phase", choices=get_args(Phase), default="decode")
    parser.add_argument(
        "--moe-mode", choices=get_args(MoEMode), default="masked_on_chip"
    )
    parser.add_argument("--objective", choices=get_args(ObjectiveMode), default="step")
    parser.add_argument(
        "--layer-groups",
        type=str,
        default=None,
        help="Comma-separated layer counts per fused call group",
    )
    parser.add_argument("--no-vmem-alias", dest="vmem_alias", action="store_false")
    parser.add_argument(
        "--no-deployment-preset",
        dest="deployment_preset",
        action="store_false",
        help="Ignore the model's shipped layouts and auto-pick parallelism",
    )
    for bool_flag in (
        "--megacore",
        "--vmem-fit-headroom",
        "--regime-sweep",
        "--explain",
        "--unresolved",
    ):
        parser.add_argument(bool_flag, action="store_true")
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        help="Emit JSON to stdout or to the given file path",
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Write a self-contained inline-SVG HTML regime report to this path",
    )
    return parser


def build_calibrate_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the `advisor calibrate` subcommand."""
    parser = argparse.ArgumentParser(
        prog="python -m vllm_torchtpu.megakernel.advisor calibrate"
    )
    parser.add_argument(
        "--trace",
        required=True,
        type=str,
        help="Path to .xplane.pb, profile directory, or Perfetto .json/.json.gz",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Destination path for advisor.measurements.v1 JSON",
    )
    parser.add_argument("--chip", type=str, default="tpu7x")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--phase", choices=get_args(Phase), default="decode")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--tp", type=int, default=None)
    return parser


def _run_calibrate_cli(argv: Sequence[str]) -> int:
    """Execute the `advisor calibrate` subcommand."""
    args = build_calibrate_arg_parser().parse_args(list(argv))
    calibrate_trace(
        args.trace,
        args.out,
        chip=args.chip,
        model=args.model,
        phase=args.phase,
        batch=args.batch,
        tp=args.tp,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the megakernel advisor CLI."""
    argv_list = list(argv) if argv is not None else list(sys.argv[1:])
    if argv_list and argv_list[0] == "calibrate":
        return _run_calibrate_cli(argv_list[1:])
    if argv_list and argv_list[0] == "estimate":
        argv_list = argv_list[1:]
    args = build_arg_parser().parse_args(argv_list)
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
        chip_spec_path=args.chip_spec,
    )

    auto_mesh = args.tp is None
    dep = DEPLOYMENT_PRESETS.get(profile.name) if args.deployment_preset else None
    dp_val = int(args.dp)
    dcp_val = int(args.dcp)
    if (
        auto_mesh
        and args.objective == "throughput"
        and args.phase == "decode"
        and dep is not None
        and dep.saturation is not None
    ):
        tp_val = dep.saturation.tp
        ep_val = args.ep if args.ep is not None else dep.saturation.ep
        if dp_val == 1:
            dp_val = dep.saturation.dp
        if dcp_val == 1:
            dcp_val = dep.saturation.dcp
    elif auto_mesh:
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
            deployment_preset=args.deployment_preset,
            measurements=args.measurements,
            objective=args.objective,
        )
        if args.html:
            write_advisor_html(sweep, args.html)
        if args.json is not None:
            _emit_json(sweep.to_dict(), args.json)
        elif args.regime_sweep:
            sys.stdout.write(format_regime_table(sweep) + "\n")
        return 0

    batch_val = (
        args.batch if args.batch is not None else (1 if args.phase == "prefill" else 16)
    )
    ns = vars(args)
    wl_kwargs = {
        fld.name: ns[fld.name] for fld in fields(WorkloadSpec) if fld.name in ns
    }
    wl_kwargs.update(
        {
            "batch": batch_val,
            "tp": tp_val,
            "ep": ep_val,
            "dp": dp_val,
            "dcp": dcp_val,
            "layer_groups": _parse_layer_groups(args.layer_groups),
            "objective": args.objective,
        }
    )
    report = advise(
        profile,
        WorkloadSpec(**wl_kwargs),
        chip_spec,
        measurements=args.measurements,
    )
    if args.json is not None:
        _emit_json(report.to_dict(), args.json)
        return 0
    out = format_advisor_summary(
        report, explain=args.explain, unresolved=args.unresolved
    )
    sys.stdout.write(out + "\n")
    return 0
