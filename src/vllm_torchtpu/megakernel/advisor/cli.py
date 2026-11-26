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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, get_args

from vllm_torchtpu.megakernel.core import errors

from .calibrate import CalibrateOptions, calibrate_trace
from .chips import BUILTIN_CHIP_SPECS, ChipSpec, resolve_chip_spec
from .constants import (
    CHIP_KIND_MAP,
    CORE_LIMITS,
    DEFAULT_CHIP,
    DEFAULT_DECODE_OSL,
    DEFAULT_PHASE,
    DEFAULT_PREFILL_ISL,
    DEFAULT_PREFILL_OSL,
    HBM_UTILIZATION_CAP,
    INVALID_CONFIG_EXIT_CODE,
    MIN_VMEM_MIB,
    PUBLIC_MEASUREMENT_ALIASES,
    RECOMMEND_DEFAULT_BATCH,
    VALID_DMA_BUFFERS,
)
from .models import (
    DEPLOYMENT_PRESETS,
    DTYPE_BASIS_PHRASES,
    MODEL_PRESET_DTYPE_BASIS,
    MODEL_PRESET_SOURCES,
    MODEL_PRESETS,
    ModelProfile,
    ParallelLayout,
    match_preset_by_geometry,
    resolve_model_profile,
)
from .regimes import (
    ParallelismSearchOptions,
    RegimeSweepOptions,
    advise_regimes,
    recommend_parallelism,
)
from .report import (
    build_bucket_schedule,
    write_advisor_html,
    write_bucket_schedule,
)
from .specs import DispatchRegime, MoEMode, ObjectiveMode, Phase, WorkloadSpec
from .text_format import (
    format_advisor_summary,
    format_layer_group_search,
    format_regime_table,
)
from .verdict import advise, search_layer_groups
from .vmem import _mesh_divisibility_reasons

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "build_calibrate_arg_parser",
    "build_parser",
    "derive_decoding_sequences",
    "format_model_listing",
    "main",
]


@dataclass(frozen=True, slots=True)
class _MinIntArg:
    """Argparse type converter enforcing an integer lower bound."""

    flag: str
    min_value: int
    __name__: ClassVar[str] = "int"

    def __call__(self, raw: str) -> int:
        """Parse integer CLI value and enforce minimum bound."""
        parsed_value = int(raw)
        if parsed_value < self.min_value:
            msg = f"{self.flag} must be >= {self.min_value}, got {parsed_value}"
            raise errors.MegakernelConfigError(msg)
        return parsed_value


@dataclass(frozen=True, slots=True)
class _FiniteFloatArg:
    """Argparse type converter enforcing finite bounds on float CLI arguments."""

    flag: str
    low: float | None = None
    high: float | None = None
    closed: str = "[]"
    __name__: ClassVar[str] = "float"

    def __call__(self, raw: str) -> float:
        """Parse finite float CLI value within configured interval bounds."""
        parsed_value = float(raw)
        if not math.isfinite(parsed_value):
            msg = f"{self.flag} must be finite, got {raw!r}"
            raise errors.MegakernelConfigError(msg)
        lo_ok = self.low is None or (
            parsed_value >= self.low
            if self.closed[0] == "["
            else parsed_value > self.low
        )
        hi_ok = self.high is None or (
            parsed_value <= self.high
            if self.closed[1] == "]"
            else parsed_value < self.high
        )
        if not lo_ok or not hi_ok:
            msg = f"{self.flag} out of bounds {self.closed}, got {parsed_value!r}"
            raise errors.MegakernelConfigError(msg)
        return parsed_value


def _parse_dma_buffers(raw: str) -> int:
    """Parse `--dma-buffers` and require an allowed buffer depth."""
    parsed_value = int(raw)
    if parsed_value not in VALID_DMA_BUFFERS:
        msg = f"--dma-buffers must be 2 or 3, got {parsed_value!r}"
        raise errors.MegakernelConfigError(msg)
    return parsed_value


def _parse_host_head_arg(raw: str) -> tuple[float, float, bool]:
    """Parse `per_op=<us>,fused=<us>` into `(per_op_us, fused_us, explicit)`."""
    stripped = raw.strip()
    if not stripped:
        msg = "Invalid --host-head-us: expected non-empty per_op=<us>,fused=<us>"
        raise errors.MegakernelConfigError(msg)
    parsed: dict[str, float] = {}
    for segment in stripped.split(","):
        part = segment.strip()
        if "=" not in part:
            msg = f"Invalid --host-head-us {part!r}; expected per_op/fused=<us>"
            raise errors.MegakernelConfigError(msg)
        key_raw, raw_value = part.split("=", 1)
        key = key_raw.strip()
        if key not in {"per_op", "fused"} or key in parsed:
            msg = f"Invalid or duplicate --host-head-us key {key!r}"
            raise errors.MegakernelConfigError(msg)
        try:
            parsed_value = float(raw_value.strip())
        except ValueError as exc:
            msg = f"Invalid float in --host-head-us {part!r}"
            raise errors.MegakernelConfigError(msg) from exc
        if not math.isfinite(parsed_value) or parsed_value < 0.0:
            msg = f"--host-head-us {key} must be finite >= 0, got {parsed_value!r}"
            raise errors.MegakernelConfigError(msg)
        parsed[key] = parsed_value
    return parsed.get("per_op", 0.0), parsed.get("fused", 0.0), True


def _parse_layer_groups(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated `--layer-groups` string into integer group sizes."""
    parts = [part.strip() for part in raw.split(",")]
    if not raw.strip() or any(not part for part in parts):
        msg = f"Invalid --layer-groups value {raw!r}; expected comma-separated ints"
        raise errors.MegakernelConfigError(msg)
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        msg = f"Invalid --layer-groups value {raw!r}; expected comma-separated ints"
        raise errors.MegakernelConfigError(msg) from exc


def _format_chip_budget_lines() -> list[str]:
    """Build `--list-models` chip core-budget lines from `CORE_LIMITS`."""
    budget_chips = [
        chip_name
        for chip_name in BUILTIN_CHIP_SPECS
        if CHIP_KIND_MAP.get(chip_name) in CORE_LIMITS.chip_kinds
    ]
    rate_only_chips = [
        chip_name
        for chip_name in BUILTIN_CHIP_SPECS
        if CHIP_KIND_MAP.get(chip_name) not in CORE_LIMITS.chip_kinds
    ]
    if budget_chips:
        verb = "has" if len(budget_chips) == 1 else "have"
        lines = [f"Chips: {', '.join(budget_chips)} {verb} verified core budgets."]
    else:
        lines = [
            f"Chips with verified core budgets: {', '.join(CORE_LIMITS.chip_kinds)}."
        ]
    if rate_only_chips:
        verb = "is" if len(rate_only_chips) == 1 else "are"
        lines.append(
            f"{', '.join(rate_only_chips)} {verb} priced from hardware rates only: "
            "fused scopes there report outside_core_limits "
            "(chip_without_core_budget)."
        )
    return lines


def format_model_listing() -> str:
    """Format validated physics-only model presets and 6-axis reference layouts.

    Returns:
        Multi-line plain-text summary for `--list-models`.
    """
    lines = [
        "=== Megakernel Advisor: validated model presets (physics-only) ===",
        *_format_chip_budget_lines(),
        "",
    ]
    for preset_name in MODEL_PRESETS:
        profile = resolve_model_profile(preset_name)
        families_text = ",".join(profile.layer_families) or "none"
        basis_phrase = DTYPE_BASIS_PHRASES[MODEL_PRESET_DTYPE_BASIS[preset_name]]
        lines.append(
            f"  {preset_name}: families={families_text} | "
            f"dtypes={profile.weight_dtype}/{profile.expert_dtype} ({basis_phrase})"
        )
        lines.append(f"    source: {MODEL_PRESET_SOURCES.get(preset_name, '')}")
        if (deployment := DEPLOYMENT_PRESETS.get(preset_name)) is not None:
            lines.extend(
                f"    {role}: {layout.label()} ({layout.submodule_label()})"
                for role, layout in deployment.named_layouts()
            )
    return "\n".join(lines)


def _require_positive_finite(name: str, parsed_value: float) -> None:
    """Validate that a Python float argument is non-bool, finite, and > 0."""
    if (
        type(parsed_value) is bool
        or not math.isfinite(parsed_value)
        or parsed_value <= 0
    ):
        msg = f"{name} must be finite and > 0, got {parsed_value!r}"
        raise errors.MegakernelConfigError(msg)


def derive_decoding_sequences(
    output_tok_s: float, mean_tpot_ms: float
) -> tuple[float, int]:
    """Derive active decoding sequences `B = output_tok_s * (mean_tpot_ms / 1000)`.

    Returns:
        Tuple `(exact_sequences, rounded_positive_batch)`.
    """
    _require_positive_finite("output_tok_s", output_tok_s)
    _require_positive_finite("mean_tpot_ms", mean_tpot_ms)
    exact_seqs = round(output_tok_s * (mean_tpot_ms / 1000.0), 6)
    return exact_seqs, max(1, round(exact_seqs))


def _validate_json_dest(dest: str) -> str:
    """Validate `--json` destination is non-empty and not an existing directory."""
    cleaned = dest.strip()
    if not cleaned:
        msg = "--json path must not be empty"
        raise errors.MegakernelConfigError(msg)
    if cleaned != "-" and Path(cleaned).is_dir():
        msg = f"--json path is a directory: {dest!r}"
        raise errors.MegakernelConfigError(msg)
    return cleaned


def _emit_json(payload: dict[str, object], dest: str) -> None:
    """Write formatted JSON to stdout when `dest == '-'`, else to file `dest`."""
    cleaned = _validate_json_dest(dest)
    blob = json.dumps(payload, indent=2) + "\n"
    if cleaned == "-":
        sys.stdout.write(blob)
    else:
        out_path = Path(cleaned)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(blob, encoding="utf-8")


def _add_int_flags(
    parser: argparse.ArgumentParser,
    min_value: int,
    default: int | None,
    specs: tuple[tuple[str, str], ...],
) -> None:
    """Register integer flags sharing `(min_value, default)` onto `parser`."""
    for flag, help_text in specs:
        parser.add_argument(
            flag, type=_MinIntArg(flag, min_value), default=default, help=help_text
        )


def _add_float_flags(
    parser: argparse.ArgumentParser,
    specs: tuple[tuple[str, float | None, float | None, str, float | None, str], ...],
) -> None:
    """Register finite float flags with interval bounds onto `parser`."""
    for flag, lo, hi, closed, default, help_text in specs:
        parser.add_argument(
            flag,
            type=_FiniteFloatArg(flag, lo, hi, closed),
            default=default,
            help=help_text,
        )


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    """Register model and chip selection flags on `parser`."""
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--model", type=str, help="Preset name, HF repo, or dir.")
    source_group.add_argument("--config", type=str, help="HF config.json or repo dir.")
    source_group.add_argument("--hf-repo", type=str, help="HuggingFace repository ID.")
    source_group.add_argument(
        "--list-models", action="store_true", help="List presets."
    )
    parser.add_argument("--chip", default=DEFAULT_CHIP, help="TPU generation.")
    parser.add_argument("--act-dtype", type=str, default=None, help="Act dtype.")
    parser.add_argument("--kv-dtype", type=str, default=None, help="KV cache dtype.")
    parser.add_argument("--state-dtype", type=str, default=None, help="State dtype.")


def _add_workload_args(parser: argparse.ArgumentParser) -> None:
    """Register phase, sequence length, and batch geometry flags on `parser`."""
    parser.add_argument("--phase", choices=get_args(Phase), default=None, help="Phase.")
    _add_int_flags(
        parser,
        1,
        None,
        (
            ("--batch", "Active sequences per step (1 prefill, 16 decode)."),
            ("--decoding-seqs", "Active decoding sequences (--batch alias)."),
            ("--isl", "Input sequence length in tokens."),
        ),
    )
    _add_int_flags(
        parser,
        0,
        None,
        (
            ("--osl", "Output sequence length in tokens (0 for prefill)."),
            ("--context", "Full KV cache context length in tokens."),
            ("--spec-tokens", "Speculative draft tokens for verify."),
            ("--chunk-size", "Chunked prefill token budget per step."),
            ("--microbatches", "Pipeline microbatches per step."),
            ("--segment-layers", "Layers per pipeline/fused segment."),
        ),
    )
    _add_float_flags(
        parser,
        (
            ("--output-tok-s", 0.0, None, "()", None, "Output throughput (tok/s)."),
            ("--mean-tpot-ms", 0.0, None, "()", None, "Mean TPOT (ms)."),
            ("--prefix-hit-rate", 0.0, 1.0, "[)", None, "Prefix hit rate [0, 1)."),
        ),
    )


def _add_mesh_and_tuning_args(parser: argparse.ArgumentParser) -> None:
    """Register TPU parallelism, VMEM buffer, and dispatch regime flags."""
    _add_int_flags(
        parser,
        1,
        None,
        (
            ("--tp", "Tensor-parallel degree."),
            ("--ep", "Expert-parallel degree."),
            ("--dcp", "Decode context parallelism degree."),
            ("--kv-split", "KV head split factor."),
        ),
    )
    _add_int_flags(
        parser,
        1,
        1,
        (
            ("--dp", "Data-parallel degree."),
            ("--pp", "Pipeline-parallel degree."),
            ("--pcp", "Prefill context parallelism degree."),
        ),
    )
    _add_int_flags(parser, 0, 0, (("--max-rows-per-call", "Max rows per call."),))
    parser.add_argument(
        "--dma-buffers", type=_parse_dma_buffers, default=2, help="2/3."
    )
    parser.add_argument(
        "--layer-groups", type=_parse_layer_groups, default=None, help="CSV sizes."
    )
    parser.add_argument(
        "--moe-mode", choices=get_args(MoEMode), default=None, help="MoE mode."
    )
    parser.add_argument(
        "--no-vmem-alias",
        action="store_false",
        dest="vmem_alias",
        help="Disable VMEM aliasing.",
    )
    parser.add_argument(
        "--vmem-fit-headroom", action="store_true", help="Require fit headroom."
    )
    _add_float_flags(
        parser,
        (
            ("--vmem-budget-mib", float(MIN_VMEM_MIB), None, "[)", None, "VMEM MiB."),
            ("--hbm-utilization-cap", 0.0, 1.0, "(]", HBM_UTILIZATION_CAP, "HBM cap."),
            ("--op-overhead-us", 0.0, None, "[)", 0.0, "Per-op overhead (us)."),
            ("--per-op-step-us", 0.0, None, "()", None, "Baseline per_op latency."),
            ("--step-coverage", 0.0, 1.0, "(]", None, "Step coverage in (0, 1]."),
        ),
    )
    parser.add_argument(
        "--host-head-us",
        type=_parse_host_head_arg,
        default=(0.0, 0.0, False),
        help="per_op=<us>,fused=<us>.",
    )
    parser.add_argument(
        "--dispatch-regime",
        choices=get_args(DispatchRegime),
        default="auto",
        help="Dispatch regime.",
    )
    parser.add_argument(
        "--objective",
        choices=get_args(ObjectiveMode),
        default="step",
        help="Optimization objective.",
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Register measurement inputs, sweep flags, and plain-text/JSON outputs."""
    parser.add_argument("--measurements", type=str, help="Measurements JSON/alias.")
    parser.add_argument("--regime-sweep", action="store_true", help="Regime matrix.")
    parser.add_argument(
        "--search-layer-groups", action="store_true", help="Layer-group search."
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Write self-contained HTML audit report to disk.",
    )
    parser.add_argument("--json", nargs="?", const="-", help="JSON stdout or file.")
    parser.add_argument("--explain", action="store_true", help="Metric provenance.")
    parser.add_argument("--unresolved", action="store_true", help="Assumed inputs.")
    parser.add_argument(
        "--no-deployment-preset",
        action="store_false",
        dest="deployment_preset",
        help="Ignore presets.",
    )
    parser.add_argument(
        "--bucket-schedule",
        type=str,
        default=None,
        help=(
            "Write advisory advisor.bucket_schedule.v1 JSON schedule to this path "
            "(or '-' for stdout)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the fusion-scope advisor.

    Returns:
        Configured `argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="python -m vllm_torchtpu.megakernel.advisor",
        description="Estimate megakernel fit and fusion gain across TPU scopes.",
    )
    for add_fn in (
        _add_source_args,
        _add_workload_args,
        _add_mesh_and_tuning_args,
        _add_output_args,
    ):
        add_fn(parser)
    return parser


def _validate_batch_exclusivity(args: argparse.Namespace) -> None:
    """Enforce mutual exclusion and completeness on CLI sequence-count flags."""
    has_batch = args.batch is not None
    has_dec = args.decoding_seqs is not None
    has_littles_law_pair = (
        args.output_tok_s is not None or args.mean_tpot_ms is not None
    )
    if sum((has_batch, has_dec, has_littles_law_pair)) > 1:
        msg = (
            "Specify at most one of --batch, --decoding-seqs, or "
            "(--output-tok-s with --mean-tpot-ms)"
        )
        raise errors.MegakernelConfigError(msg)
    if has_littles_law_pair and (
        args.output_tok_s is None or args.mean_tpot_ms is None
    ):
        msg = "--output-tok-s and --mean-tpot-ms must be provided together"
        raise errors.MegakernelConfigError(msg)


def _validate_non_empty_cli_values(args: argparse.Namespace) -> None:
    """Reject empty/whitespace name or path strings and directory output paths."""
    if not args.chip.strip():
        msg = "--chip must not be empty"
        raise errors.MegakernelConfigError(msg)
    for flag, raw_val in (
        ("--act-dtype", args.act_dtype),
        ("--kv-dtype", args.kv_dtype),
        ("--state-dtype", args.state_dtype),
    ):
        if raw_val is not None and not raw_val.strip():
            msg = f"{flag} must not be empty"
            raise errors.MegakernelConfigError(msg)
    if args.measurements is not None:
        cleaned_meas = args.measurements.strip()
        if not cleaned_meas:
            msg = "--measurements must not be empty"
            raise errors.MegakernelConfigError(msg)
        if (
            cleaned_meas not in PUBLIC_MEASUREMENT_ALIASES
            and Path(cleaned_meas).is_dir()
        ):
            msg = f"--measurements path is a directory: {args.measurements!r}"
            raise errors.MegakernelConfigError(msg)
    if args.json is not None:
        _validate_json_dest(args.json)


def _validate_cli_combinations(args: argparse.Namespace) -> None:
    """Enforce fail-closed CLI mode and flag combinations."""
    _validate_batch_exclusivity(args)
    _validate_non_empty_cli_values(args)
    parsed_flags = vars(args)
    single_pairs = (
        ("--phase", "phase"),
        ("--batch", "batch"),
        ("--decoding-seqs", "decoding_seqs"),
        ("--output-tok-s", "output_tok_s"),
        ("--mean-tpot-ms", "mean_tpot_ms"),
        ("--isl", "isl"),
        ("--osl", "osl"),
        ("--context", "context"),
        ("--prefix-hit-rate", "prefix_hit_rate"),
        ("--spec-tokens", "spec_tokens"),
        ("--chunk-size", "chunk_size"),
        ("--microbatches", "microbatches"),
        ("--segment-layers", "segment_layers"),
        ("--kv-split", "kv_split"),
        ("--layer-groups", "layer_groups"),
        ("--moe-mode", "moe_mode"),
    )
    single_only = [
        flag_name
        for flag_name, attr_name in single_pairs
        if parsed_flags[attr_name] is not None
    ]
    if args.list_models:
        bool_flags = (
            ("--regime-sweep", args.regime_sweep),
            ("--search-layer-groups", args.search_layer_groups),
            ("--explain", args.explain),
            ("--unresolved", args.unresolved),
            ("--no-deployment-preset", not args.deployment_preset),
        )
        opt_flags = (
            ("--html", "html"),
            ("--bucket-schedule", "bucket_schedule"),
            ("--json", "json"),
            ("--measurements", "measurements"),
            ("--tp", "tp"),
            ("--ep", "ep"),
            ("--dcp", "dcp"),
            ("--act-dtype", "act_dtype"),
            ("--kv-dtype", "kv_dtype"),
            ("--state-dtype", "state_dtype"),
        )
        conflicting_flags = [
            *single_only,
            *(flag_name for flag_name, enabled in bool_flags if enabled),
            *(
                flag_name
                for flag_name, attr_name in opt_flags
                if parsed_flags[attr_name] is not None
            ),
        ]
        if conflicting_flags:
            msg = (
                f"--list-models cannot be combined with {', '.join(conflicting_flags)}"
            )
            raise errors.MegakernelConfigError(msg)
    sweep_mode = args.regime_sweep or args.bucket_schedule is not None
    if sweep_mode and (args.search_layer_groups or args.explain or single_only):
        mode_flag = "--regime-sweep" if args.regime_sweep else "--bucket-schedule"
        bad_sweep = (
            *(["--search-layer-groups"] if args.search_layer_groups else []),
            *(["--explain"] if args.explain else []),
            *single_only,
        )
        msg = f"{mode_flag} cannot be combined with {', '.join(bad_sweep)}"
        raise errors.MegakernelConfigError(msg)
    if args.search_layer_groups and (args.explain or args.unresolved):
        bad_search = (
            *(["--explain"] if args.explain else []),
            *(["--unresolved"] if args.unresolved else []),
        )
        msg = f"--search-layer-groups cannot be combined with {', '.join(bad_search)}"
        raise errors.MegakernelConfigError(msg)
    if (args.phase or "decode") == "prefill" and any(
        seq_arg is not None
        for seq_arg in (args.decoding_seqs, args.output_tok_s, args.mean_tpot_ms)
    ):
        msg = (
            "--decoding-seqs, --output-tok-s, and --mean-tpot-ms are not valid "
            "with --phase prefill"
        )
        raise errors.MegakernelConfigError(msg)


def _validate_output_file_arg(
    raw_path: str | None, flag_name: str, *, allow_stdout: bool = False
) -> str | None:
    """Validate that an optional file output flag is non-empty and not a dir."""
    if raw_path is None:
        return None
    cleaned = raw_path.strip()
    if not cleaned or (not allow_stdout and cleaned == "-"):
        msg = f"{flag_name} requires a non-empty file path, not '-' (stdout)."
        raise errors.MegakernelConfigError(msg)
    if cleaned != "-" and Path(cleaned).is_dir():
        msg = f"{flag_name} path is a directory: {raw_path!r}"
        raise errors.MegakernelConfigError(msg)
    return cleaned


def _validate_output_exclusivity(args: argparse.Namespace) -> None:
    """Enforce valid CLI output paths and flag combinations before evaluation."""
    args.html = _validate_output_file_arg(args.html, "--html")
    args.bucket_schedule = _validate_output_file_arg(
        args.bucket_schedule, "--bucket-schedule", allow_stdout=True
    )
    if args.bucket_schedule is not None and args.unresolved and not args.regime_sweep:
        msg = (
            "--unresolved requires --regime-sweep or single-run output "
            "and cannot be combined with --bucket-schedule alone."
        )
        raise errors.MegakernelConfigError(msg)
    if args.search_layer_groups and args.html is not None:
        msg = "--search-layer-groups cannot be combined with --html."
        raise errors.MegakernelConfigError(msg)
    if args.bucket_schedule != "-":
        return
    if args.json == "-":
        msg = "Cannot output both --bucket-schedule and --json to stdout ('-')."
        raise errors.MegakernelConfigError(msg)
    if args.regime_sweep:
        msg = (
            "Cannot output --bucket-schedule to stdout ('-') when --regime-sweep "
            "text table is requested."
        )
        raise errors.MegakernelConfigError(msg)


def _resolve_cli_batch(args: argparse.Namespace) -> tuple[int, bool]:
    """Resolve `(batch, batch_explicit)` from CLI sequence-count flags."""
    if args.batch is not None:
        return args.batch, True
    if args.decoding_seqs is not None:
        return args.decoding_seqs, True
    if args.output_tok_s is not None and args.mean_tpot_ms is not None:
        return derive_decoding_sequences(args.output_tok_s, args.mean_tpot_ms)[1], True
    phase = args.phase or "decode"
    return (1 if phase == "prefill" else RECOMMEND_DEFAULT_BATCH), False


def _build_base_workload(args: argparse.Namespace) -> WorkloadSpec:
    """Construct shared `WorkloadSpec` defaults for sweep and single evaluation."""
    host_per_op_us, host_fused_us, host_explicit = args.host_head_us
    return WorkloadSpec(
        vmem_alias=args.vmem_alias,
        vmem_fit_headroom=args.vmem_fit_headroom,
        op_overhead_us=args.op_overhead_us,
        vmem_budget_mib=args.vmem_budget_mib,
        dma_buffers=args.dma_buffers,
        max_rows_per_call=args.max_rows_per_call,
        host_head_per_op_us=host_per_op_us,
        host_head_fused_us=host_fused_us,
        host_head_explicit=host_explicit,
        per_op_step_us=args.per_op_step_us,
        step_coverage=args.step_coverage,
        hbm_utilization_cap=args.hbm_utilization_cap,
        dispatch_regime=args.dispatch_regime,
        objective=args.objective,
    )


def _build_single_workload(
    args: argparse.Namespace,
    *,
    batch: int,
    batch_explicit: bool,
    layout: ParallelLayout,
) -> WorkloadSpec:
    """Construct `WorkloadSpec` from parsed CLI arguments and resolved layout."""
    phase: Phase = args.phase or "decode"
    default_osl = DEFAULT_PREFILL_OSL if phase == "prefill" else DEFAULT_DECODE_OSL
    return replace(
        _build_base_workload(args),
        phase=phase,
        batch=batch,
        isl=DEFAULT_PREFILL_ISL if args.isl is None else args.isl,
        osl=default_osl if args.osl is None else args.osl,
        context=0 if args.context is None else args.context,
        prefix_hit_rate=0.0 if args.prefix_hit_rate is None else args.prefix_hit_rate,
        spec_tokens=0 if args.spec_tokens is None else args.spec_tokens,
        chunk_size=0 if args.chunk_size is None else args.chunk_size,
        microbatches=0 if args.microbatches is None else args.microbatches,
        segment_layers=0 if args.segment_layers is None else args.segment_layers,
        tp=layout.tp,
        ep=layout.ep,
        dp=layout.dp,
        pp=layout.pp,
        dcp=layout.dcp,
        pcp=layout.pcp,
        kv_split=1 if args.kv_split is None else args.kv_split,
        layer_groups=layout.layer_groups,
        moe_mode="masked_on_chip" if args.moe_mode is None else args.moe_mode,
        batch_explicit=batch_explicit,
        moe_mode_explicit=args.moe_mode is not None,
    )


def _resolve_cli_model_and_chip(
    args: argparse.Namespace,
) -> tuple[ModelProfile, ChipSpec]:
    """Resolve `ModelProfile` and `ChipSpec` from CLI source arguments."""
    target: str | Path | None = None
    if args.config is not None:
        raw_config = args.config.strip()
        if not raw_config:
            msg = "--config path must not be empty"
            raise errors.MegakernelConfigError(msg)
        config_path = Path(raw_config)
        cfg_file = config_path / "config.json" if config_path.is_dir() else config_path
        if not cfg_file.is_file():
            msg = f"Missing config file: {args.config!r}"
            raise errors.MegakernelConfigError(msg)
        target = config_path
    else:
        for flag, raw_str in (("--model", args.model), ("--hf-repo", args.hf_repo)):
            if raw_str is not None:
                target = raw_str.strip()
                if not target:
                    msg = f"{flag} must not be empty"
                    raise errors.MegakernelConfigError(msg)
                break
    if target is None:
        msg = "One of --model, --config, --hf-repo, or --list-models is required"
        raise errors.MegakernelConfigError(msg)
    profile = resolve_model_profile(
        target,
        act_dtype=args.act_dtype,
        kv_dtype=args.kv_dtype,
        state_dtype=args.state_dtype,
    )
    if (
        args.config is not None
        and (preset := match_preset_by_geometry(profile)) is not None
    ):
        sys.stderr.write(
            f"Config matched validated preset {preset}: "
            "its 6-axis reference topologies apply.\n"
        )
    return profile, resolve_chip_spec(args.chip)


def _finish_output(
    args: argparse.Namespace,
    payload: dict[str, object],
    render_text: Callable[[], str],
    exit_code: int,
) -> int:
    """Write either JSON or rendered plain text and return `exit_code`."""
    if args.json is not None:
        _emit_json(payload, args.json)
    else:
        sys.stdout.write(render_text() + "\n")
    return exit_code


def _run_sweep_command(
    args: argparse.Namespace,
    profile: ModelProfile,
    chip_spec: ChipSpec,
    measurements_path: str | None,
) -> int:
    """Execute `--regime-sweep` and emit plain-text or JSON outputs."""
    base_workload = replace(
        _build_base_workload(args),
        tp=args.tp or 1,
        ep=args.ep or 1,
        dp=args.dp,
        pp=args.pp,
        dcp=1 if args.dcp is None else args.dcp,
        pcp=args.pcp,
        batch_explicit=True,
    )
    sweep_options = RegimeSweepOptions(
        base_workload=base_workload,
        tp=args.tp,
        ep=args.ep,
        dp=args.dp,
        pp=args.pp,
        dcp=args.dcp,
        pcp=args.pcp,
        deployment_preset=args.deployment_preset,
        measurements=measurements_path,
    )
    sweep = advise_regimes(profile, chip_spec, options=sweep_options)
    invalid = any(
        entry.verdict.startswith("invalid_config(") for entry in sweep.regimes
    )
    exit_code = INVALID_CONFIG_EXIT_CODE if invalid else 0
    if args.html is not None:
        write_advisor_html(sweep, args.html)
    if args.bucket_schedule is not None:
        if args.bucket_schedule == "-":
            _emit_json(build_bucket_schedule(sweep).to_dict(), "-")
        else:
            write_bucket_schedule(sweep, args.bucket_schedule)
    if args.json is None and not args.regime_sweep:
        return exit_code
    return _finish_output(
        args,
        sweep.to_dict(),
        lambda: format_regime_table(sweep, unresolved=args.unresolved),
        exit_code,
    )


def _run_single_command(
    args: argparse.Namespace,
    profile: ModelProfile,
    chip_spec: ChipSpec,
    measurements_path: str | None,
) -> int:
    """Evaluate a single workload or layer-group search and write output."""
    phase: Phase = args.phase or "decode"
    tp, ep, dcp = args.tp, args.ep, args.dcp
    if tp is None:
        search_opts = ParallelismSearchOptions(
            dp=args.dp,
            pp=args.pp,
            dcp=0 if dcp is None else dcp,
            pcp=args.pcp,
            ep=ep,
            phase=phase,
        )
        recommendation = recommend_parallelism(profile, chip_spec, options=search_opts)
        tp, dcp = recommendation.tp, recommendation.dcp
        if ep is None:
            ep = recommendation.ep
    layout = ParallelLayout(
        tp=tp,
        ep=1 if ep is None else ep,
        dp=args.dp,
        pp=args.pp,
        dcp=1 if dcp is None else dcp,
        pcp=args.pcp,
        layer_groups=() if args.layer_groups is None else args.layer_groups,
    )
    batch_size, batch_explicit = _resolve_cli_batch(args)
    workload = _build_single_workload(
        args, batch=batch_size, batch_explicit=batch_explicit, layout=layout
    )
    if args.search_layer_groups:
        search_report = search_layer_groups(
            profile, workload, chip_spec, measurements=measurements_path
        )
        invalid = bool(_mesh_divisibility_reasons(profile, workload, chip_spec))
        return _finish_output(
            args,
            search_report.to_dict(),
            lambda: format_layer_group_search(search_report),
            INVALID_CONFIG_EXIT_CODE if invalid else 0,
        )
    report = advise(profile, workload, chip_spec, measurements=measurements_path)
    if args.html is not None:
        write_advisor_html(report, args.html)
    return _finish_output(
        args,
        report.to_dict(),
        lambda: format_advisor_summary(
            report, explain=args.explain, unresolved=args.unresolved
        ),
        INVALID_CONFIG_EXIT_CODE if report.verdict.startswith("invalid_config(") else 0,
    )


def _dispatch(args: argparse.Namespace) -> int:
    """Dispatch parsed CLI arguments to listing, regime sweep, or estimate."""
    _validate_cli_combinations(args)
    _validate_output_exclusivity(args)
    if args.list_models:
        sys.stdout.write(format_model_listing() + "\n")
        return 0
    profile, chip_spec = _resolve_cli_model_and_chip(args)
    if args.regime_sweep or args.bucket_schedule is not None:
        return _run_sweep_command(args, profile, chip_spec, args.measurements)
    return _run_single_command(args, profile, chip_spec, args.measurements)


def build_calibrate_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `advisor calibrate`."""
    parser = argparse.ArgumentParser(
        prog="python -m vllm_torchtpu.megakernel.advisor calibrate",
        description="Calibrate TPU megakernel advisor priors from XPlane/Perfetto.",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to .xplane.pb, profile directory, or Perfetto .json/.json.gz",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Destination path for advisor.measurements.v1 JSON",
    )
    parser.add_argument("--chip", default=DEFAULT_CHIP)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--num-layers",
        type=_MinIntArg("--num-layers", 1),
        default=None,
        help=(
            "Per-rank decoder layers for conserved boundary normalization "
            "(required override over --model when pp > 1)."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=get_args(Phase),
        default=DEFAULT_PHASE,
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--tp", type=int, default=None)
    return parser


def _validate_calibrate_cli_values(args: argparse.Namespace) -> tuple[str, str]:
    """Validate non-empty paths/names and output destination for `calibrate`."""
    trace_str = args.trace.strip()
    if not trace_str:
        msg = "--trace path must not be empty"
        raise errors.MegakernelConfigError(msg)
    out_str = args.out.strip()
    if not out_str or out_str == "-":
        msg = "--out must be a non-empty file path"
        raise errors.MegakernelConfigError(msg)
    if Path(out_str).is_dir():
        msg = f"--out path is a directory: {args.out!r}"
        raise errors.MegakernelConfigError(msg)
    if not args.chip.strip():
        msg = "--chip must not be empty"
        raise errors.MegakernelConfigError(msg)
    if args.model is not None and not args.model.strip():
        msg = "--model must not be empty"
        raise errors.MegakernelConfigError(msg)
    return trace_str, out_str


def _run_calibrate_cli(argv: Sequence[str]) -> int:
    """Parse arguments and execute the `calibrate` subcommand."""
    args = build_calibrate_arg_parser().parse_args(list(argv))
    trace_str, out_str = _validate_calibrate_cli_values(args)
    calibrate_options = CalibrateOptions(
        chip=args.chip,
        model=args.model,
        phase=args.phase,
        batch=args.batch,
        tp=args.tp,
        num_layers=args.num_layers,
    )
    calibrate_trace(trace_str, out_str, options=calibrate_options)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fusion-scope advisor command-line interface.

    Returns:
        Process exit status code.
    """
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list and argv_list[0] == "calibrate":
        return _run_calibrate_cli(argv_list[1:])
    if argv_list and argv_list[0] == "estimate":
        argv_list = argv_list[1:]
    return _dispatch(build_parser().parse_args(argv_list))
