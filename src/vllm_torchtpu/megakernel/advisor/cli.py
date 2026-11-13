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
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from vllm_torchtpu.megakernel.core import errors

from .chips import ChipSpec, resolve_chip_spec
from .constants import (
    DEFAULT_CHIP,
    DEFAULT_PREFILL_ISL,
    INVALID_CONFIG_EXIT_CODE,
    MIN_VMEM_MIB,
    RECOMMEND_DEFAULT_BATCH,
    VALID_DMA_BUFFERS,
)
from .models import (
    ModelProfile,
    ParallelLayout,
    resolve_model_profile,
)
from .regimes import (
    RegimeSweepOptions,
    advise_regimes,
    recommend_parallelism,
)
from .report import (
    format_advisor_summary,
    format_regime_table,
    write_advisor_html,
)
from .specs import (
    DispatchRegime,
    MoEMode,
    ObjectiveMode,
    Phase,
    WorkloadSpec,
)
from .verdict import advise

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
    blob = json.dumps(payload, indent=2) + "\n"
    if dest == "-":
        sys.stdout.write(blob)
    else:
        out_path = Path(dest)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(blob, encoding="utf-8")


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    """Register model and chip source selection flags."""
    m_grp = parser.add_mutually_exclusive_group(required=True)
    m_grp.add_argument(
        "--model",
        type=str,
        help="Built-in deployment preset name, HF repo ID, or local directory.",
    )
    m_grp.add_argument(
        "--config",
        type=Path,
        help="Path to HuggingFace config.json or repo directory.",
    )
    m_grp.add_argument(
        "--hf-repo",
        type=str,
        help="HuggingFace repository ID.",
    )
    parser.add_argument(
        "--chip",
        default=DEFAULT_CHIP,
        help="Target TPU generation ('tpu7x', 'v6e', 'auto').",
    )
    parser.add_argument(
        "--chip-spec",
        type=Path,
        dest="chip_spec_path",
        help="Path to a custom JSON chip specification.",
    )
    parser.add_argument(
        "--act-dtype",
        type=str,
        default=None,
        help="Activation precision override (e.g. 'fp8', 'bf16').",
    )


def _add_workload_args(parser: argparse.ArgumentParser) -> None:
    """Register phase, sequence length, and batch geometry flags."""
    parser.add_argument(
        "--phase",
        choices=get_args(Phase),
        default="decode",
        help="Workload serving phase ('decode', 'prefill', 'verify', 'mixed').",
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Batch size (default: 1 for prefill, 16 for decode/verify/mixed).",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=DEFAULT_PREFILL_ISL,
        help="Input sequence length in tokens.",
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=1024,
        help="Output sequence length in tokens (defaults to 0 for prefill).",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="Full KV cache context length in tokens.",
    )
    parser.add_argument(
        "--prefix-hit-rate",
        type=float,
        default=0.0,
        help="Fraction of input tokens cached in prefix (0.0 .. <1.0).",
    )
    parser.add_argument(
        "--spec-tokens",
        type=int,
        default=0,
        help="Number of speculative draft tokens per step for verify.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Chunked prefill token budget per forward call.",
    )
    parser.add_argument(
        "--microbatches",
        type=int,
        default=0,
        help="Pipeline microbatches per step.",
    )
    parser.add_argument(
        "--segment-layers",
        type=int,
        default=0,
        help="Layers per pipeline/fused segment.",
    )


def _add_mesh_and_tuning_args(parser: argparse.ArgumentParser) -> None:
    """Register TPU parallelism, VMEM buffer, and dispatch regime flags."""
    parser.add_argument(
        "--tp",
        type=int,
        help="Tensor-parallel degree over intra-slice ICI (must be power of two).",
    )
    parser.add_argument(
        "--ep",
        type=int,
        help="Expert-parallel degree (must divide num_experts).",
    )
    parser.add_argument("--dp", type=int, default=1, help="Data-parallel degree.")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline-parallel degree.")
    parser.add_argument(
        "--dcp",
        type=int,
        default=1,
        help="Decode context parallelism degree.",
    )
    parser.add_argument(
        "--pcp",
        type=int,
        default=1,
        help="Prefill context parallelism degree.",
    )
    parser.add_argument(
        "--kv-split",
        type=int,
        default=1,
        help="KV-cache head sharding split factor.",
    )
    parser.add_argument(
        "--layer-groups",
        type=str,
        help="Comma-separated layer counts per group (e.g. '8,8,8,8,8').",
    )
    parser.add_argument(
        "--moe-mode",
        choices=get_args(MoEMode),
        default="masked_on_chip",
        help="Expert execution mode.",
    )
    parser.add_argument(
        "--no-vmem-alias",
        action="store_false",
        dest="vmem_alias",
        help="Disable GQA and MoE scratch buffer aliasing.",
    )
    parser.add_argument(
        "--vmem-fit-headroom",
        action="store_true",
        help="Require 2 MiB scratchpad alignment headroom when fitting tokens.",
    )
    parser.add_argument(
        "--vmem-budget-mib",
        type=float,
        help="Override total usable VMEM in MiB.",
    )
    parser.add_argument(
        "--dma-buffers",
        type=int,
        default=2,
        help="Weight double-buffering depth (2 or 3).",
    )
    parser.add_argument(
        "--max-rows-per-call",
        type=int,
        default=0,
        help=(
            "Most decode or verify rows the megakernel serves in one call; a "
            "larger batch is priced as separate calls (0: no declared limit)."
        ),
    )
    parser.add_argument(
        "--hbm-utilization-cap",
        type=float,
        default=0.80,
        help="Serving HBM pool headroom cap fraction (default: 0.80).",
    )
    parser.add_argument(
        "--op-overhead-us",
        type=float,
        default=0.0,
        help="Per-operator launch overhead in microseconds.",
    )
    parser.add_argument(
        "--dispatch-regime",
        choices=get_args(DispatchRegime),
        default="auto",
        help=(
            "Dispatch gap regime ('auto', 'zero_floor', 'compiled_xla', "
            "'unfused_eager')."
        ),
    )
    parser.add_argument(
        "--objective",
        choices=get_args(ObjectiveMode),
        default="step",
        help="Optimization objective ('step', 'throughput', 'both').",
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Register calibration, sweep, and output formatting flags."""
    parser.add_argument(
        "--measurements",
        type=str,
        help="Path to an advisor.measurements.v1 JSON file or bundled/auto.",
    )
    parser.add_argument(
        "--regime-sweep",
        action="store_true",
        help="Evaluate the full 12-regime matrix across batch and phase.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        help="Write self-contained HTML regime report to disk.",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        help="Emit machine-readable JSON output (to stdout or file).",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Surface per-metric provenance tags in terminal summary.",
    )
    parser.add_argument(
        "--unresolved",
        action="store_true",
        help="List unresolved ASSUMED inputs ranked by step-time sensitivity.",
    )
    parser.add_argument(
        "--no-deployment-preset",
        action="store_false",
        dest="deployment_preset",
        help="Ignore deployment presets during regime sweep.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the fusion-scope advisor."""
    parser = argparse.ArgumentParser(
        prog="python -m vllm_torchtpu.megakernel.advisor",
        description="Estimate megakernel fit and fusion gain across TPU scopes.",
    )
    _add_source_args(parser)
    _add_workload_args(parser)
    _add_mesh_and_tuning_args(parser)
    _add_output_args(parser)
    return parser


def _resolve_cli_mesh(
    args: argparse.Namespace,
) -> tuple[int | None, int | None, int, int, int]:
    """Extract parallel layout degrees (tp, ep, dp, pp, dcp) from CLI namespace."""
    return (args.tp, args.ep, args.dp, args.pp, args.dcp)


def _build_single_workload(
    args: argparse.Namespace,
    *,
    batch: int,
    layout: ParallelLayout,
) -> WorkloadSpec:
    """Construct WorkloadSpec from parsed CLI arguments and layout."""
    return WorkloadSpec(
        phase=args.phase,
        batch=batch,
        isl=args.isl,
        osl=args.osl,
        context=args.context,
        prefix_hit_rate=args.prefix_hit_rate,
        spec_tokens=args.spec_tokens,
        chunk_size=args.chunk_size,
        microbatches=args.microbatches,
        segment_layers=args.segment_layers,
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        kv_split=args.kv_split,
        layer_groups=layout.layer_groups,
        moe_mode=args.moe_mode,
        vmem_alias=args.vmem_alias,
        vmem_fit_headroom=args.vmem_fit_headroom,
        op_overhead_us=args.op_overhead_us,
        vmem_budget_mib=args.vmem_budget_mib,
        dma_buffers=args.dma_buffers,
        max_rows_per_call=args.max_rows_per_call,
        hbm_utilization_cap=args.hbm_utilization_cap,
        dispatch_regime=args.dispatch_regime,
        objective=args.objective,
    )


def _resolve_cli_model_and_chip(
    args: argparse.Namespace,
) -> tuple[ModelProfile, ChipSpec]:
    """Resolve ModelProfile and ChipSpec from CLI arguments."""
    target: str | Path = args.model or args.config or args.hf_repo or "qwen3.5-397b"
    act_dt = getattr(args, "act_dtype", None)
    profile = resolve_model_profile(target, act_dtype=act_dt)
    chip_spec = resolve_chip_spec(
        args.chip,
        chip_spec_path=args.chip_spec_path,
    )
    return profile, chip_spec


def _validate_cli_ints(args: argparse.Namespace) -> None:
    """Validate CLI integer bounds before running recommendations or sweeps."""
    for name, val in (
        ("isl", args.isl),
        ("dp", args.dp),
        ("pp", args.pp),
        ("dcp", args.dcp),
        ("pcp", args.pcp),
        ("kv_split", args.kv_split),
    ):
        if val < 1:
            msg = f"--{name.replace('_', '-')} must be >= 1, got {val}"
            raise errors.MegakernelConfigError(msg)
    for name, opt_val in (("batch", args.batch), ("tp", args.tp), ("ep", args.ep)):
        if opt_val is not None and opt_val < 1:
            msg = f"--{name} must be >= 1, got {opt_val}"
            raise errors.MegakernelConfigError(msg)
    for name, val in (
        ("osl", args.osl),
        ("context", args.context),
        ("spec_tokens", args.spec_tokens),
        ("chunk_size", args.chunk_size),
        ("microbatches", args.microbatches),
        ("segment_layers", args.segment_layers),
        ("max_rows_per_call", args.max_rows_per_call),
    ):
        if val < 0:
            msg = f"--{name.replace('_', '-')} must be >= 0, got {val}"
            raise errors.MegakernelConfigError(msg)
    if args.dma_buffers not in VALID_DMA_BUFFERS:
        msg = f"--dma-buffers must be 2 or 3, got {args.dma_buffers!r}"
        raise errors.MegakernelConfigError(msg)


def _validate_cli_floats(args: argparse.Namespace) -> None:
    """Validate CLI floating-point bounds before running recommendations or sweeps."""
    if not math.isfinite(args.prefix_hit_rate) or not (
        0.0 <= args.prefix_hit_rate < 1.0
    ):
        msg = f"--prefix-hit-rate must be in [0, 1), got {args.prefix_hit_rate!r}"
        raise errors.MegakernelConfigError(msg)
    if not math.isfinite(args.hbm_utilization_cap) or not (
        0.0 < args.hbm_utilization_cap <= 1.0
    ):
        msg = (
            f"--hbm-utilization-cap must be in (0, 1], got {args.hbm_utilization_cap!r}"
        )
        raise errors.MegakernelConfigError(msg)
    if not math.isfinite(args.op_overhead_us) or args.op_overhead_us < 0.0:
        msg = f"--op-overhead-us must be finite and >= 0, got {args.op_overhead_us!r}"
        raise errors.MegakernelConfigError(msg)
    if args.vmem_budget_mib is not None and (
        not math.isfinite(args.vmem_budget_mib) or args.vmem_budget_mib < MIN_VMEM_MIB
    ):
        msg = (
            f"--vmem-budget-mib must be >= {MIN_VMEM_MIB}, got {args.vmem_budget_mib!r}"
        )
        raise errors.MegakernelConfigError(msg)


def _validate_cli_args(args: argparse.Namespace) -> None:
    """Validate CLI scalar bounds before running recommendations or sweeps."""
    _validate_cli_ints(args)
    _validate_cli_floats(args)


def _run_sweep_command(
    args: argparse.Namespace,
    profile: ModelProfile,
    chip_spec: ChipSpec,
) -> int:
    """Execute `--regime-sweep` / `--html` and emit stdout/HTML/JSON outputs."""
    base_wl = WorkloadSpec(
        tp=args.tp or 1,
        ep=args.ep or 1,
        dp=args.dp,
        pp=args.pp,
        dcp=args.dcp,
        pcp=args.pcp,
        vmem_alias=args.vmem_alias,
        vmem_fit_headroom=args.vmem_fit_headroom,
        op_overhead_us=args.op_overhead_us,
        vmem_budget_mib=args.vmem_budget_mib,
        dma_buffers=args.dma_buffers,
        max_rows_per_call=args.max_rows_per_call,
        hbm_utilization_cap=args.hbm_utilization_cap,
        objective=args.objective,
        dispatch_regime=args.dispatch_regime,
    )
    opts = RegimeSweepOptions(
        base_workload=base_wl,
        tp=args.tp,
        ep=args.ep,
        dp=args.dp,
        pp=args.pp,
        dcp=args.dcp,
        pcp=args.pcp,
        deployment_preset=args.deployment_preset,
        measurements=args.measurements,
    )
    sweep = advise_regimes(profile, chip_spec, options=opts)
    if args.html:
        write_advisor_html(sweep, args.html)
    if args.json is not None:
        _emit_json(sweep.to_dict(), args.json)
    elif args.regime_sweep:
        sys.stdout.write(format_regime_table(sweep) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fusion-scope advisor command-line interface."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(args)

    if args.config and not Path(args.config).is_file():
        msg = f"Missing config file: {args.config!r}"
        raise errors.MegakernelConfigError(msg)

    profile, chip_spec = _resolve_cli_model_and_chip(args)

    if args.regime_sweep or args.html:
        return _run_sweep_command(args, profile, chip_spec)

    tp, ep, dp, pp, dcp = _resolve_cli_mesh(args)
    if tp is None:
        rec = recommend_parallelism(profile, chip_spec, dp=dp, pp=pp)
        tp = rec.tp
        if ep is None:
            ep = rec.ep

    layout = ParallelLayout(
        tp=1 if tp is None else tp,
        ep=1 if ep is None else ep,
        dp=dp,
        pp=pp,
        dcp=dcp,
        pcp=args.pcp,
        layer_groups=_parse_layer_groups(args.layer_groups),
    )

    batch_val = (
        args.batch
        if args.batch is not None
        else (1 if args.phase == "prefill" else RECOMMEND_DEFAULT_BATCH)
    )
    workload = _build_single_workload(args, batch=batch_val, layout=layout)
    report = advise(
        profile,
        workload,
        chip_spec,
        measurements=args.measurements,
    )
    exit_code = (
        INVALID_CONFIG_EXIT_CODE if report.verdict.startswith("invalid_config(") else 0
    )
    if args.json is not None:
        _emit_json(report.to_dict(), args.json)
        return exit_code
    out = format_advisor_summary(
        report, explain=args.explain, unresolved=args.unresolved
    )
    sys.stdout.write(out + "\n")
    return exit_code
