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
"""CLI entrypoint for physics hardware floors, ordering, inspect, and explain."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .chips import (
    CHIPS,
    JAX_VERSION,
    ChipSpec,
    SliceTopology,
    chip_catalog,
    get_solver_assumptions,
    load_chip_spec,
    override_chip_ici_bw,
    topologies_for_chip,
)
from .constants import (
    BITS_PER_BYTE,
    DEFAULT_HBM_RESERVE_GB,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SLO_TIERS,
    EXPERT_DTYPE_BYTES,
    INSPECT_SCHEMA,
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
    top_layout_roles,
    top_layouts,
)
from .families import (
    _candidate_matches_layout,
    attach_reference_report,
    ordered_candidates_by_concurrency,
    parse_layout,
    reference_layout_report,
)
from .geometry import (
    UnsupportedArchitectureError,
    from_hf_config,
    resolve_model_identity,
)
from .hf_loader import (
    apply_hf_overrides,
    audit_hf_config,
    dtypes_from_hf_config,
    load_hf_config,
    load_hf_config_with_identity,
)
from .modules import Dtypes, Parallel, StepShape
from .occupancy import ExpertActivationModel, parse_activation_spec
from .report import write_html
from .schema import format_parallelism
from .serving import ServingRequest, plan
from .solver import (
    EXAMPLE_SCENARIOS,
    ParetoPoint,
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

    from .geometry import ModelGeometry
    from .schema import PoolMeshPlan, ReferenceLayoutReport

_EXIT_EMPTY_FRONTIER: Final[int] = 1
EXIT_ERROR: Final[int] = 2


@dataclass(frozen=True, slots=True)
class _CliModelContext:
    """Resolved model source, config, geometry, chip, and topologies."""

    source: str
    config: dict[str, object]
    geometry: ModelGeometry
    chip: ChipSpec
    topologies: dict[str, SliceTopology]


@dataclass(frozen=True, slots=True)
class _SolveRenderContext:
    """Chip, dtypes, model label, and topologies for solve output rendering."""

    chip: ChipSpec
    dtypes: Dtypes
    source: str
    topologies: Mapping[str, SliceTopology] | None = None


def _parse_hf_overrides(raw: str | None) -> dict[str, object] | None:
    """Parse optional ``--hf-overrides`` JSON object string into a dict."""
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        msg = f"--hf-overrides must be a JSON object, got {type(parsed).__name__}"
        raise TypeError(msg)
    return parsed


def _add_model_source(parser: argparse.ArgumentParser) -> None:
    """Attach optional ``--config`` / ``--model`` and ``--hf-overrides`` inputs."""
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to local Hugging Face config.json file (unit: path, scope: model).",
    )
    source_group.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Hugging Face model ID, local directory, or config.json path "
            "(unit: id_or_path, scope: model)."
        ),
    )
    parser.add_argument(
        "--hf-overrides",
        type=str,
        default=None,
        help=(
            "JSON object merged into loaded HF config before validation "
            "(unit: json_obj, scope: model)."
        ),
    )


def _add_shared_chip_flags(
    parser: argparse.ArgumentParser,
    *,
    is_root: bool,
) -> None:
    """Attach chip selection, interconnect overrides, assumptions, and JSON flags."""
    sup = argparse.SUPPRESS
    parser.add_argument(
        "--chip",
        default="tpu7x" if is_root else sup,
        choices=sorted(CHIPS),
        help="Target TPU chip specification key (unit: enum, scope: chip).",
    )
    parser.add_argument(
        "--chip-spec",
        type=Path,
        default=None if is_root else sup,
        help=(
            "Path to a JSON ChipSpec specification overriding --chip "
            "(unit: path, scope: chip)."
        ),
    )
    parser.add_argument(
        "--ici-bw-gbytes-per-s",
        "--ici-bw-gbps",
        dest="ici_bw_gbps",
        type=float,
        default=None if is_root else sup,
        help=(
            "Override per-core ICI egress bandwidth in gigabytes per second "
            "(unit: GB/s, scope: core)."
        ),
    )
    parser.add_argument(
        "--dcn-gbit-per-s",
        "--dcn-gbps",
        dest="dcn_gbps",
        type=float,
        default=None if is_root else sup,
        help=(
            "Override per-chip DCN bandwidth in gigabits per second "
            "(unit: Gbit/s, scope: chip)."
        ),
    )
    parser.add_argument(
        "--assumptions",
        action="store_true",
        default=False if is_root else sup,
        help=(
            "Print the hardware and solver constants catalog and exit "
            "(unit: flag, scope: chip)."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False if is_root else sup,
        help=(
            "Emit machine-readable JSON output to stdout (unit: flag, scope: command)."
        ),
    )


def _add_dtype_override_args(parser: argparse.ArgumentParser) -> None:
    """Attach optional weight, KV, state, activation, and latency override flags."""
    parser.add_argument(
        "--weight-dtype",
        choices=sorted(WEIGHT_DTYPE_BYTES),
        default=None,
        help="Dense linear projection weight storage dtype (unit: dtype, scope: core).",
    )
    parser.add_argument(
        "--expert-weight-dtype",
        choices=sorted(EXPERT_DTYPE_BYTES),
        default=None,
        help="Routed MoE expert weight storage dtype (unit: dtype, scope: core).",
    )
    parser.add_argument(
        "--aux-dtype",
        choices=sorted(WEIGHT_DTYPE_BYTES),
        default=None,
        help="Router, embedding, and LM-head weight dtype (unit: dtype, scope: core).",
    )
    parser.add_argument(
        "--kv-dtype",
        choices=sorted(KV_DTYPE_BYTES),
        default=None,
        help="KV cache storage dtype in HBM (unit: dtype, scope: core).",
    )
    parser.add_argument(
        "--state-dtype",
        choices=sorted(STATE_DTYPE_BYTES),
        default=None,
        help=(
            "Recurrent linear-attention state storage dtype (unit: dtype, scope: core)."
        ),
    )
    parser.add_argument(
        "--act-dtype",
        "--act",
        dest="act_dtype",
        choices=("a4", "a8", "a16", "fp4", "nvfp4", "fp8", "bf16", "bfloat16"),
        default=None,
        help=(
            "Matmul activation precision mode a4/fp4/nvfp4, a8/fp8, or a16/bf16 "
            "(unit: enum, scope: core)."
        ),
    )
    parser.add_argument(
        "--collective-latency-us",
        type=float,
        default=None,
        help=(
            "Per-collective communication latency applied to estimates only "
            "(unit: us, scope: collective)."
        ),
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attach flags shared across the ``plan`` and ``explain`` subcommands."""
    _add_model_source(parser)
    _add_shared_chip_flags(parser, is_root=False)
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help=(
            "Tensor parallel rank count across TensorCores (unit: ranks, scope: pool)."
        ),
    )
    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help=(
            "Active KV context length per sequence in tokens "
            "(unit: tokens, scope: request)."
        ),
    )
    parser.add_argument(
        "--kv-split",
        type=int,
        default=1,
        help=(
            "Decode context parallel KV split factor across ranks "
            "(unit: ranks, scope: pool)."
        ),
    )
    parser.add_argument(
        "--kv-merge",
        choices=("ag_rs", "exchange"),
        default="ag_rs",
        help=(
            "Context parallel attention merge collective schedule "
            "(unit: enum, scope: pool)."
        ),
    )
    parser.add_argument(
        "--moe",
        choices=("tp_slice", "ep"),
        default="tp_slice",
        help="Routed MoE weight sharding mode across ranks (unit: enum, scope: pool).",
    )
    parser.add_argument(
        "--moe-dispatch",
        choices=("allreduce", "ag_ep", "a2a_push", "fused_rs"),
        default="allreduce",
        help=(
            "MoE token dispatch and combine collective schedule "
            "(unit: enum, scope: pool)."
        ),
    )
    parser.add_argument(
        "--residual",
        choices=("replicated", "sp"),
        default="replicated",
        help=(
            "Residual stream layout across tensor parallel ranks "
            "(unit: enum, scope: pool)."
        ),
    )
    parser.add_argument(
        "--ar-algo",
        choices=("one_shot", "two_shot", "ring"),
        default="one_shot",
        help="All-reduce collective algorithm on ICI links (unit: enum, scope: link).",
    )
    _add_dtype_override_args(parser)


def _add_scenario_budget_args(parser: argparse.ArgumentParser) -> None:
    """Attach cluster mode, chip/instance budget, SLO, and pin flags."""
    parser.add_argument(
        "--example-scenario",
        choices=sorted(EXAMPLE_SCENARIOS),
        default=None,
        help="Predefined traffic scenario preset name (unit: enum, scope: workload).",
    )
    parser.add_argument(
        "--mode",
        choices=("mixed", "disagg", "auto"),
        default="auto",
        help=(
            "Pool serving architecture mode: mixed, disagg, or auto "
            "(unit: enum, scope: cluster)."
        ),
    )
    parser.add_argument(
        "--max-chips",
        "--num-chips",
        dest="max_chips",
        type=int,
        default=None,
        help=(
            "Maximum physical TPU chips available across all pools "
            "(unit: chips, scope: cluster)."
        ),
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help=(
            "Maximum total serving instances across prefill and decode pools "
            "(unit: instances, scope: cluster)."
        ),
    )
    parser.add_argument(
        "--example-slo-tier",
        choices=sorted(EXAMPLE_SLO_TIERS),
        default="none",
        help="Named TTFT/TPOT latency SLO preset tier (unit: enum, scope: workload).",
    )
    parser.add_argument(
        "--ttft-slo-ms",
        type=float,
        default=None,
        help=(
            "Target maximum Time To First Token SLO in milliseconds "
            "(unit: ms, scope: request)."
        ),
    )
    parser.add_argument(
        "--tpot-slo-ms",
        type=float,
        default=None,
        help=(
            "Target maximum Time Per Output Token SLO in milliseconds "
            "(unit: ms, scope: request)."
        ),
    )
    parser.add_argument(
        "--prefill-topology",
        type=str,
        default=None,
        help="Target prefill pool slice mesh topology (unit: mesh_str, scope: pool).",
    )
    parser.add_argument(
        "--decode-topology",
        type=str,
        default=None,
        help="Target decode pool slice mesh topology (unit: mesh_str, scope: pool).",
    )
    parser.add_argument(
        "--pd",
        type=str,
        default=None,
        help=(
            "Prefill:decode instance ratio (e.g. 1:1, 1:2) "
            "(unit: ratio_str, scope: cluster)."
        ),
    )


def _add_scenario_workload_args(parser: argparse.ArgumentParser) -> None:
    """Attach sequence length, prefix caching, offload, and acceptance flags."""
    parser.add_argument(
        "--isl",
        type=int,
        default=None,
        help=(
            "Input sequence length per request in tokens "
            "(unit: tokens, scope: request)."
        ),
    )
    parser.add_argument(
        "--osl",
        type=int,
        default=None,
        help=(
            "Output sequence length per request in tokens "
            "(unit: tokens, scope: request)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default=None,
        help=(
            "Comma-separated list of concurrent request counts to evaluate "
            "(unit: requests, scope: workload)."
        ),
    )
    parser.add_argument(
        "--prefix-hit-rate",
        type=float,
        default=None,
        help=(
            "Cached prefix token hit ratio in [0.0, 1.0] (unit: ratio, scope: request)."
        ),
    )
    parser.add_argument(
        "--kv-offload-gb",
        type=float,
        default=None,
        help=(
            "Host DRAM KV cache offloading capacity per instance in GiB "
            "(unit: GiB, scope: host)."
        ),
    )
    parser.add_argument(
        "--accept-len",
        type=float,
        default=None,
        help=(
            "Mean accepted tokens per speculative verification step >= 1.0 "
            "(unit: tokens/step, scope: request)."
        ),
    )
    parser.add_argument(
        "--think-time-s",
        type=float,
        default=None,
        help=(
            "Optional client think time between turns in seconds; omit for "
            "open-loop mode (unit: s, scope: request)."
        ),
    )


def _add_scenario_override_args(parser: argparse.ArgumentParser) -> None:
    """Attach optional memory reserve, dtype, latency, and MoE activation overrides."""
    parser.add_argument(
        "--hbm-reserve-gb",
        type=float,
        default=DEFAULT_HBM_RESERVE_GB,
        help=(
            "HBM reserved per core excluded from serving limits "
            "(unit: GB, scope: core)."
        ),
    )
    _add_dtype_override_args(parser)
    parser.add_argument(
        "--moe-activation",
        type=str,
        default=ExpertActivationModel().regime,
        help=(
            "Routed MoE expert activation envelope mode: pigeonhole, bernoulli, "
            "or hull:<path> (unit: spec, scope: workload)."
        ),
    )


def _add_scenario_args(parser: argparse.ArgumentParser) -> None:
    """Attach scenario, budget, offload, and override flags."""
    _add_model_source(parser)
    _add_shared_chip_flags(parser, is_root=False)
    _add_scenario_budget_args(parser)
    _add_scenario_workload_args(parser)
    _add_scenario_override_args(parser)


def _add_solve_ordering_args(target: argparse.ArgumentParser) -> None:
    """Register ordering and disagg-pairing flags shared by solve and ceiling."""
    target.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Optional cap on ordered layouts returned per concurrency; "
            "omit to return all (unit: count, scope: command)."
        ),
    )
    target.add_argument(
        "--sort-by",
        choices=("throughput", "interactivity"),
        default="throughput",
        help=(
            "Primary ordering metric within each SLO group: throughput or "
            "interactivity (unit: enum, scope: command)."
        ),
    )
    target.add_argument(
        "--pairs",
        action="store_true",
        default=False,
        help=(
            "Evaluate full disaggregated prefill-decode instance pairs in "
            "addition to per-pool tables (unit: flag, scope: command)."
        ),
    )


def _add_solve_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register solve and ceiling subcommands."""
    solve_parser = subparsers.add_parser(
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
    _add_solve_ordering_args(solve_parser)
    solve_parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help=(
            "Write a self-contained offline HTML report to file path "
            "(unit: path, scope: command)."
        ),
    )
    solve_parser.add_argument(
        "--reference-layout",
        type=str,
        default=None,
        help=(
            "Reference layout to rank against search space "
            "(unit: layout specification, scope: command)."
        ),
    )

    ceil_parser = subparsers.add_parser(
        "ceiling",
        help="Export uncalibrated physics candidate ceiling rows from solve.",
        description=(
            "Export uncalibrated serial-phase physics candidate rows along the "
            "estimated throughput-interactivity envelope."
        ),
    )
    _add_scenario_args(ceil_parser)
    _add_solve_ordering_args(ceil_parser)


def _add_plan_explain_inspect_subparsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register plan, explain, and inspect subcommands."""
    plan_parser = subparsers.add_parser(
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
    plan_parser.add_argument(
        "--concurrency",
        type=str,
        default="1,4,8,16,32,64",
        help=(
            "Comma-separated decode batch concurrencies to evaluate "
            "(unit: requests, scope: pool)."
        ),
    )
    plan_parser.add_argument(
        "--spec-window",
        type=int,
        default=0,
        help=(
            "Speculative draft token window k per verification step "
            "(unit: tokens, scope: step)."
        ),
    )
    plan_parser.add_argument(
        "--accept-len",
        type=float,
        default=None,
        help=(
            "Mean accepted tokens per verification step when spec-window > 0 "
            "(unit: tokens/step, scope: step)."
        ),
    )

    explain_parser = subparsers.add_parser(
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
    explain_parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Active sequence batch size per step (unit: requests, scope: step).",
    )
    explain_parser.add_argument(
        "--rows-per-seq",
        type=int,
        default=1,
        help=(
            "Token rows processed per sequence in the step "
            "(unit: tokens/seq, scope: step)."
        ),
    )
    explain_parser.add_argument(
        "--spec-window",
        type=int,
        default=0,
        help=(
            "Speculative draft token window k per verification step "
            "(unit: tokens, scope: step)."
        ),
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Audit HF config onboarding readiness and parameter counts.",
    )
    _add_model_source(inspect_parser)
    _add_shared_chip_flags(inspect_parser, is_root=False)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for physics floors, ordering, inspect, and explain."""
    parser = argparse.ArgumentParser(
        prog="vllm_torchtpu.roofline",
        description=(
            "Compute physical lower bounds and uncalibrated serial-phase "
            "layout rankings across TPU parallel configurations."
        ),
    )
    parser.set_defaults(
        weight_dtype=None,
        expert_weight_dtype=None,
        aux_dtype=None,
        kv_dtype=None,
        state_dtype=None,
        act_dtype=None,
        pd=None,
        max_instances=None,
        prefill_topology=None,
        decode_topology=None,
        reference_layout=None,
        hf_overrides=None,
        top_k=None,
        sort_by="throughput",
        pairs=False,
    )
    _add_shared_chip_flags(parser, is_root=True)
    subparsers = parser.add_subparsers(dest="cmd", required=False)
    _add_solve_subparsers(subparsers)
    _add_plan_explain_inspect_subparsers(subparsers)
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


def _apply_chip_overrides(chip: ChipSpec, args: argparse.Namespace) -> ChipSpec:
    """Apply CLI ICI and DCN interconnect overrides cleanly onto ``chip``."""
    out = chip
    if args.ici_bw_gbps is not None:
        out = override_chip_ici_bw(out, args.ici_bw_gbps * UNIT_GIGA)
    if args.dcn_gbps is not None:
        _require_positive_finite("--dcn-gbit-per-s", args.dcn_gbps)
        out = replace(
            out,
            dcn_bw=(args.dcn_gbps * UNIT_GIGA) / BITS_PER_BYTE,
            dcn_host_gbit_per_s=args.dcn_gbps * float(max(1, out.chips_per_host)),
            dcn_bw_overridden=True,
        )
    return out


def _render_assumptions_view(chip: ChipSpec, *, as_json: bool) -> int:
    """Render the live hardware and solver constants catalog."""
    catalog = chip_catalog(chip)
    solver_assumptions = get_solver_assumptions(chip)
    if as_json:
        payload = {
            "chip": chip.name,
            "jax_version": JAX_VERSION,
            "catalog": catalog,
            "solver_assumptions": solver_assumptions,
        }
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0
    sys.stdout.write(
        f"Constants catalog for chip={chip.name} (jax_version={JAX_VERSION}):\n"
    )
    sys.stdout.write(
        f"{'name':<30}  {'display':<20}  {'raw_value':>14}  {'unit':<8}  "
        f"{'scope':<6}  {'kind':<8}  {'description':<36}  source\n"
    )
    for name, entry in catalog.items():
        val = entry["value"]
        val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
        display_str = str(entry.get("display") or f"{val_str} {entry['unit']}")
        desc_str = str(entry.get("description") or "")
        sys.stdout.write(
            f"{name:<30}  {display_str:<20}  {val_str:>14}  {entry['unit']:<8}  "
            f"{entry['scope']:<6}  {entry['kind']:<8}  {desc_str:<32}  "
            f"{entry['source']}\n"
        )
    sys.stdout.write("\nSolver assumptions:\n")
    sys.stdout.write(
        f"{'name':<28}  {'flag':<26}  {'kind':<11}  {'value':<12}  "
        f"{'unit':<8}  {'source':<10}  description\n"
    )
    for item in solver_assumptions:
        raw_val = item.get("value")
        val_txt = f"{raw_val:.4g}" if isinstance(raw_val, float) else str(raw_val)
        sys.stdout.write(
            f"{item.get('name', '')!s:<28}  {item.get('flag', '')!s:<26}  "
            f"{item.get('kind', '')!s:<11}  {val_txt:<12}  "
            f"{item.get('unit', '')!s:<8}  {item.get('source', '')!s:<10}  "
            f"{item.get('description', '')!s}\n"
        )
    return 0


def _parse_concurrencies(raw_conc: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated positive concurrency integers."""
    if raw_conc is None:
        return None
    concs = tuple(int(tok) for tok in raw_conc.split(",") if tok.strip())
    if not concs or any(val < 1 for val in concs):
        msg = f"--concurrency values must be positive integers, got {raw_conc!r}"
        raise ValueError(msg)
    return concs


def _build_cli_scenario(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> TrafficScenario:
    """Build TrafficScenario directly from CLI flags or named scenario preset."""
    _require_positive_finite("--ttft-slo-ms", args.ttft_slo_ms)
    _require_positive_finite("--tpot-slo-ms", args.tpot_slo_ms)
    _require_non_negative_finite("--think-time-s", args.think_time_s)
    if args.example_scenario is None and args.example_slo_tier != "none":
        msg = "--example-slo-tier requires --example-scenario"
        raise ValueError(msg)
    concs = _parse_concurrencies(args.concurrency)
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
                kv_offload_gb=args.kv_offload_gb,
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
        concurrencies=concs if concs is not None else EXAMPLE_CONCURRENCIES,
        kv_working_set_turns=1.0,
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
        kv_offload_gb=args.kv_offload_gb if args.kv_offload_gb is not None else 0.0,
    )


def _write_warnings(warnings: Sequence[str]) -> None:
    """Write formatted warning lines to stderr."""
    for warning in warnings:
        sys.stderr.write(f"WARNING: {warning}\n")


def _render_ceiling_cmd(args: argparse.Namespace, serving_plan: ServingPlan) -> int:
    """Write uncalibrated candidate ceiling rows as JSON array or JSON lines."""
    _write_warnings(serving_plan.warnings)
    ceiling_rows = emit_ceiling_rows(serving_plan, top_k=args.top_k)
    if not ceiling_rows:
        if not serving_plan.warnings:
            _write_warnings((NO_FEASIBLE_LAYOUT_WARNING,))
        if args.json:
            sys.stdout.write("[]\n")
        return _EXIT_EMPTY_FRONTIER
    if args.json:
        sys.stdout.write(json.dumps(ceiling_rows, allow_nan=False) + "\n")
        return 0
    for row in ceiling_rows:
        sys.stdout.write(json.dumps(row, allow_nan=False) + "\n")
    return 0


def _format_pool_label(pool: PoolMeshPlan) -> str:
    """Format one pool's instance count, topology, and parallelism axes."""
    prefix = f"{pool.instances}x " if pool.instances > 1 else ""
    return f"{prefix}{pool.topology} {format_parallelism(asdict(pool))}"


def _format_point_pools_label(point: ParetoPoint) -> str:
    """Format prefill/decode pool label pair for one ParetoPoint."""
    if point.prefill_pool is not None:
        return (
            f"P {_format_pool_label(point.prefill_pool)} + "
            f"D {_format_pool_label(point.decode_pool)}"
        )
    return _format_pool_label(point.decode_pool)


def _find_reference_candidate(
    serving_plan: ServingPlan,
    ref_report: Mapping[str, object] | None,
) -> tuple[ParetoPoint | None, str]:
    """Return reference candidate outside top-k and its pool label when found."""
    if (
        ref_report is None
        or not bool(ref_report.get("found"))
        or bool(ref_report.get("in_top_k"))
    ):
        return (None, "")
    point_dict = ref_report.get("point")
    if not point_dict:
        return (None, "")
    candidate = next(
        (cand for cand in serving_plan.candidates if asdict(cand) == point_dict),
        None,
    )
    if candidate is None:
        return (None, "")
    return (candidate, _format_point_pools_label(candidate))


def _render_solve_ascii_table(
    serving_plan: ServingPlan,
    top_k: int | None,
    render_ctx: _SolveRenderContext,
    ref_report: Mapping[str, object] | None = None,
) -> None:
    """Write ordered candidate layouts as an aligned terminal table."""
    dtypes = render_ctx.dtypes
    sys.stdout.write(
        f"# Dtypes: dense={dtypes.w_dense_dtype} ({dtypes.w_dense:.4g}B, "
        f"{dtypes.w_dense_source}), expert={dtypes.w_expert_dtype} "
        f"({dtypes.w_expert:.4g}B, {dtypes.w_expert_source}), "
        f"kv={dtypes.kv_dtype} ({dtypes.kv:.4g}B, {dtypes.kv_source}), "
        f"state={dtypes.state_dtype} ({dtypes.state:.4g}B, {dtypes.state_source}), "
        f"act={dtypes.act}\n"
    )
    selected_points = top_layouts(serving_plan, top_k=top_k)
    annotated = top_layout_roles(
        selected_points, serving_plan.frontier, sort_by=serving_plan.sort_by
    )
    pool_labels = [_format_point_pools_label(point) for point, _, _ in annotated]
    reference_candidate, reference_pools_label = _find_reference_candidate(
        serving_plan, ref_report
    )
    all_labels = pool_labels + (
        [reference_pools_label] if reference_pools_label else []
    )
    pool_col_width = max([len("pools"), *(len(label) for label in all_labels)])
    sys.stdout.write(
        f"{'rank':>4}  {'role':<17}  {'mode':<7}  {'concurrency':>11}  "
        f"{'chips':>5}  {'pools':<{pool_col_width}}  {'est_int_tok_s':>13}  "
        f"{'tok_tput_chip':>13}  {'bound_tpot_ms':>13}  {'est_ttft_ms':>11}  "
        f"{'est_tpot_ms':>11}\n",
    )
    for rank, ((point, role, _margin), pools) in enumerate(
        zip(annotated, pool_labels, strict=True), 1
    ):
        sys.stdout.write(
            f"{rank:>4}  {role:<17}  {point.mode:<7}  "
            f"{point.concurrency:>11}  {point.total_chips:>5}  "
            f"{pools:<{pool_col_width}}  "
            f"{point.est_interactivity_tok_s:>13.2f}  "
            f"{point.token_throughput_per_chip:>13.1f}  "
            f"{point.bound_tpot_ms:>13.2f}  "
            f"{point.est_ttft_ms:>11.1f}  {point.est_tpot_ms:>11.2f}\n",
        )
    if reference_candidate is not None:
        sys.stdout.write(
            f"{'-':>4}  {'reference':<17}  {reference_candidate.mode:<7}  "
            f"{reference_candidate.concurrency:>11}  "
            f"{reference_candidate.total_chips:>5}  "
            f"{reference_pools_label:<{pool_col_width}}  "
            f"{reference_candidate.est_interactivity_tok_s:>13.2f}  "
            f"{reference_candidate.token_throughput_per_chip:>13.1f}  "
            f"{reference_candidate.bound_tpot_ms:>13.2f}  "
            f"{reference_candidate.est_ttft_ms:>11.1f}  "
            f"{reference_candidate.est_tpot_ms:>11.2f}\n",
        )


def _write_reference_summary(ref_report: Mapping[str, object] | None) -> None:
    """Write reference-layout diagnostic summary line in text mode."""
    if ref_report is None:
        return
    if bool(ref_report.get("ruled_out")):
        sys.stdout.write(
            f"reference {ref_report['layout']}: "
            f"enumerated, ruled out by SLO ({ref_report['rejected_by']})\n"
        )
    elif bool(ref_report.get("found")):
        concurrency = ref_report.get("concurrency")
        concurrency_part = f"c={concurrency}, " if concurrency is not None else ""
        sys.stdout.write(
            f"reference {ref_report['layout']}: "
            f"{concurrency_part}"
            f"tpot_bound_rank={ref_report['rank_by_tpot_bound']}/"
            f"{ref_report['candidates']}, "
            f"tput_rank={ref_report['rank_by_estimated_throughput']}/"
            f"{ref_report['candidates']}\n"
        )
    else:
        sys.stdout.write(
            f"reference {ref_report['layout']}: "
            f"not found ({ref_report['rejected_by']})\n"
        )


def _render_pool_records_table(
    serving_plan: ServingPlan,
    top_k: int | None = None,
) -> None:
    """Write per-pool prefill and decode floor records when pairs are omitted."""
    sys.stdout.write(
        f"{'role':<8}  {'concurrency':>11}  {'chips':>5}  {'pareto':>6}  "
        f"{'pool':<28}  {'tok_tput_chip':>13}  {'ttft_ms':>10}  {'tpot_ms':>10}\n"
    )
    visible_prefill = slice_per_concurrency(serving_plan.prefill_pools, top_k)
    visible_decode = slice_per_concurrency(serving_plan.decode_pools, top_k)
    for record in (*visible_prefill, *visible_decode):
        mark = "*" if record.is_pareto else ""
        pool_lbl = _format_pool_label(record.pool)
        ttft_s = (
            f"{record.ttft_ms:>10.1f}" if record.ttft_ms is not None else f"{'-':>10}"
        )
        tpot_s = (
            f"{record.tpot_ms:>10.2f}" if record.tpot_ms is not None else f"{'-':>10}"
        )
        sys.stdout.write(
            f"{record.role:<8}  {record.concurrency:>11}  "
            f"{record.pool.chips_per_instance:>5}  {mark:>6}  {pool_lbl:<28}  "
            f"{record.token_throughput_per_chip:>13.1f}  {ttft_s}  {tpot_s}\n"
        )


def _render_concurrency_ordered_lines(
    serving_plan: ServingPlan,
    top_k: int | None,
) -> None:
    """Write one-rank-per-line ordered candidate summaries per concurrency."""
    ordered = ordered_candidates_by_concurrency(serving_plan, top_k=top_k)
    if not ordered:
        return
    sys.stdout.write("# Ordered Candidates by Concurrency\n")
    for concurrency, points in ordered.items():
        sys.stdout.write(f"# c={concurrency}:\n")
        for rank, point in enumerate(points, 1):
            label = _format_point_pools_label(point)
            sys.stdout.write(
                f"#   #{rank} {label} | bound_tpot_ms={point.bound_tpot_ms:.2f} "
                f"| interactivity={point.est_interactivity_tok_s:.1f} tok/s "
                f"| throughput={point.token_throughput_per_chip:.1f} tok/s/chip "
                f"| min_decode_seqs={point.min_decode_seqs:.1f}\n"
            )


def _render_solve_output(
    args: argparse.Namespace,
    serving_plan: ServingPlan,
    render_ctx: _SolveRenderContext,
    ref_report: Mapping[str, object] | None = None,
) -> int:
    """Render solve subcommand HTML, JSON, or text table and return exit code."""
    payload = None
    if args.html is not None or args.json:
        payload = build_solve_payload(
            serving_plan,
            chip=render_ctx.chip,
            dtypes=render_ctx.dtypes,
            model=render_ctx.source,
            top_k=args.top_k,
            topologies=render_ctx.topologies,
        )
        if ref_report is not None:
            payload = attach_reference_report(payload, ref_report)
    if args.html is not None and payload is not None:
        write_html(payload, args.html)
    _write_warnings(serving_plan.warnings)
    has_results = bool(
        serving_plan.candidates
        or serving_plan.prefill_pools
        or serving_plan.decode_pools
    )
    if not has_results:
        if args.json and payload is not None:
            sys.stdout.write(format_solve_json(payload) + "\n")
        return _EXIT_EMPTY_FRONTIER
    if args.json and payload is not None:
        sys.stdout.write(format_solve_json(payload) + "\n")
        return 0
    if serving_plan.candidates:
        _render_solve_ascii_table(
            serving_plan, args.top_k, render_ctx, ref_report=ref_report
        )
        _write_reference_summary(ref_report)
        _render_concurrency_ordered_lines(serving_plan, args.top_k)
        if serving_plan.prefill_pools or serving_plan.decode_pools:
            _render_pool_records_table(serving_plan, top_k=args.top_k)
    else:
        _render_pool_records_table(serving_plan, top_k=args.top_k)
        _write_reference_summary(ref_report)
    sys.stdout.write(
        "# Note: bound_* columns are hard physical lower bounds; "
        "est_* columns are model estimates.\n",
    )
    return 0


def resolve_dtypes_from_args(
    args: argparse.Namespace,
    config: dict[str, object] | None = None,
) -> Dtypes:
    """Apply optional CLI dtype overrides onto HF-derived defaults via replace."""
    dtypes = dtypes_from_hf_config(config) if config is not None else Dtypes()
    if args.weight_dtype is not None:
        weight_dtype = str(args.weight_dtype)
        dtypes = replace(
            dtypes,
            w_dense=WEIGHT_DTYPE_BYTES[weight_dtype],
            w_dense_dtype=weight_dtype,
            w_dense_source="flag",
        )
    if args.expert_weight_dtype is not None:
        expert_dtype = str(args.expert_weight_dtype)
        dtypes = replace(
            dtypes,
            w_expert=EXPERT_DTYPE_BYTES[expert_dtype],
            w_expert_dtype=expert_dtype,
            w_expert_source="flag",
        )
    if args.aux_dtype is not None:
        dtypes = replace(dtypes, w_aux=WEIGHT_DTYPE_BYTES[str(args.aux_dtype)])
    if args.kv_dtype is not None:
        kv_dtype = str(args.kv_dtype)
        dtypes = replace(
            dtypes,
            kv=KV_DTYPE_BYTES[kv_dtype],
            kv_dtype=kv_dtype,
            kv_source="flag",
        )
    if args.state_dtype is not None:
        state_dtype = str(args.state_dtype)
        dtypes = replace(
            dtypes,
            state=STATE_DTYPE_BYTES[state_dtype],
            state_dtype=state_dtype,
            state_source="flag",
        )
    if args.act_dtype is not None:
        act_str = str(args.act_dtype)
        if act_str in ("a4", "fp4", "nvfp4"):
            dtypes = replace(dtypes, act="a4")
        elif act_str in ("a8", "fp8"):
            dtypes = replace(dtypes, act="a8")
        else:
            dtypes = replace(dtypes, act="a16")
    return dtypes


def _parse_pd_ratio(
    raw_pd: str,
    parser: argparse.ArgumentParser,
) -> tuple[int, int]:
    """Parse --pd PREFILL:DECODE ratio as two positive integers or exit with usage."""
    usage_msg = (
        "--pd must be PREFILL:DECODE with two positive integers "
        f"(for example, 1:2), got {raw_pd!r}"
    )
    prefill_instances = 0
    decode_instances = 0
    prefill_part, separator, decode_part = raw_pd.partition(":")
    if separator and ":" not in decode_part:
        try:
            prefill_instances = int(prefill_part)
            decode_instances = int(decode_part)
        except ValueError:
            prefill_instances = 0
    if prefill_instances < 1 or decode_instances < 1:
        parser.error(usage_msg)
    return (prefill_instances, decode_instances)


def _parse_cli_reference_layout(
    reference_layout: str | None,
    parser: argparse.ArgumentParser,
) -> dict[str, dict[str, int]] | None:
    """Parse --reference-layout CLI argument or raise parser error."""
    if reference_layout is None:
        return None
    try:
        return parse_layout(reference_layout)
    except ValueError as err:
        parser.error(f"Invalid --reference-layout: {err}")


def _build_cli_reference_report(
    serving_plan: ServingPlan,
    model_ctx: _CliModelContext,
    parsed_ref: dict[str, dict[str, int]] | None,
    args: argparse.Namespace,
    solve_opts: SolveOptions,
) -> ReferenceLayoutReport | None:
    """Build reference layout comparison report when requested."""
    if parsed_ref is None or args.reference_layout is None:
        return None
    effective_solve_opts = (
        replace(solve_opts, max_chips=serving_plan.budget)
        if solve_opts.max_chips is None and serving_plan.budget > 0
        else solve_opts
    )
    solve_topologies = _resolve_solve_topologies(
        effective_solve_opts, chip=model_ctx.chip
    )
    is_found = any(
        _candidate_matches_layout(candidate, parsed_ref)
        for candidate in serving_plan.candidates
    )
    rejected_by = (
        ""
        if is_found
        else explain_layout_rejection(
            model_ctx.geometry,
            parsed_ref,
            solve_topologies,
            effective_solve_opts,
            concurrencies=serving_plan.scenario.concurrencies,
        )
    )
    top_candidates = top_layouts(serving_plan, top_k=args.top_k)
    return reference_layout_report(
        serving_plan,
        top_candidates,
        args.reference_layout,
        rejected_by,
        top_k=args.top_k,
    )


def _parse_cli_activation_model(
    spec: str,
    geometry: ModelGeometry,
    parser: argparse.ArgumentParser,
) -> ExpertActivationModel:
    """Parse CLI --moe-activation spec or exit via parser.error on failure."""
    try:
        return parse_activation_spec(spec, geometry)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(f"invalid --moe-activation {spec!r}: {exc}")


def _resolve_eff_max_chips(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int | None:
    """Resolve chip budget from CLI flags, pins, or instance cap."""
    has_pins = bool(args.prefill_topology and args.decode_topology and args.pd)
    if args.max_chips is None and args.max_instances is None and not has_pins:
        parser.error(
            "Either --max-chips, complete pins "
            "(--prefill-topology, --decode-topology, --pd), "
            "or --max-instances is required."
        )
    return args.max_chips


def _dispatch_solve_family(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    model_ctx: _CliModelContext,
) -> int:
    """Execute solve or ceiling subcommands."""
    parsed_ref = _parse_cli_reference_layout(args.reference_layout, parser)
    pd_ratio = _parse_pd_ratio(args.pd, parser) if args.pd else None
    eff_max_chips = _resolve_eff_max_chips(args, parser)
    activation_model = _parse_cli_activation_model(
        args.moe_activation, model_ctx.geometry, parser
    )
    try:
        _require_positive_int("--max-chips", eff_max_chips)
        if args.max_instances is not None and args.max_instances < MIN_INSTANCES:
            msg = (
                f"--max-instances must be >= {MIN_INSTANCES}, got {args.max_instances}"
            )
            raise ValueError(msg)
        if args.top_k is not None and args.top_k < MIN_TOP_K:
            msg = f"--top-k must be >= {MIN_TOP_K}, got {args.top_k}"
            raise ValueError(msg)
        _require_non_negative_finite("--hbm-reserve-gb", args.hbm_reserve_gb)
        _require_non_negative_finite(
            "--collective-latency-us", args.collective_latency_us
        )
        scenario = _build_cli_scenario(args, parser)
        dtypes = resolve_dtypes_from_args(args, model_ctx.config)
        has_pins = bool(args.prefill_topology and args.decode_topology and pd_ratio)
        include_pairs = bool(args.pairs) or (
            args.cmd == "ceiling"
            and args.mode in ("auto", "disagg")
            and (eff_max_chips is not None or has_pins)
        )
        solve_opts = SolveOptions(
            mode=args.mode,
            max_chips=eff_max_chips,
            max_instances=args.max_instances,
            collective_latency_s=(args.collective_latency_us or 0.0) * UNIT_MICRO,
            topologies=tuple(model_ctx.topologies.values()),
            hbm_reserve_gb=args.hbm_reserve_gb,
            sort_by=args.sort_by,
            include_pairs=include_pairs,
            top_k=args.top_k,
            prefill_topology=args.prefill_topology,
            decode_topology=args.decode_topology,
            pd_ratio=pd_ratio,
            activation_model=activation_model,
        )
        serving_plan = solve_serving_plan(
            model_ctx.geometry,
            dtypes,
            model_ctx.chip,
            scenario,
            solve_opts,
        )
    except (ValueError, TypeError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return EXIT_ERROR
    if args.cmd == "ceiling":
        return _render_ceiling_cmd(args, serving_plan)
    ref_report = _build_cli_reference_report(
        serving_plan, model_ctx, parsed_ref, args, solve_opts
    )
    render_ctx = _SolveRenderContext(
        chip=model_ctx.chip,
        dtypes=dtypes,
        source=model_ctx.source,
        topologies=model_ctx.topologies,
    )
    return _render_solve_output(args, serving_plan, render_ctx, ref_report=ref_report)


def _build_plan_parallel_dtypes(
    args: argparse.Namespace,
    config: dict[str, object] | None = None,
) -> tuple[Parallel, Dtypes, float]:
    """Build Parallel, Dtypes, and latency override from plan/explain flags."""
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
    return (parallel, dtypes, latency_s)


def _dispatch_plan_cmd(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    model_ctx: _CliModelContext,
) -> int:
    """Execute plan for hard physics floors and uncalibrated serial decode estimates."""
    if args.spec_window > 0 and args.accept_len is None:
        parser.error("--accept-len is required when --spec-window > 0")
    parallel, dtypes, _ = _build_plan_parallel_dtypes(args, model_ctx.config)
    concs_list = list(_parse_concurrencies(args.concurrency) or ())
    rows = plan(
        geometry=model_ctx.geometry,
        chip=model_ctx.chip,
        parallel=parallel,
        dtypes=dtypes,
        request=ServingRequest(
            context=args.context,
            concurrencies=concs_list,
            spec_window=args.spec_window,
            acceptance_length=args.accept_len,
            collective_latency_us=args.collective_latency_us,
        ),
        topologies=model_ctx.topologies,
    )
    if args.json:
        sys.stdout.write(json.dumps([asdict(row) for row in rows]) + "\n")
        return 0
    header = (
        f"{'concurrency':>11}  {'rows':>6}  {'binding':<7}  "
        f"{'hard_floor_ms':>13}  {'serial_estimate_ms':>18}  "
        f"{'tpot_floor_ms':>13}  {'interactivity_ceil_tok_s':>24}  "
        f"{'output_tput_ceil_per_chip':>25}  {'flag':<7}\n"
    )
    sys.stdout.write(header)
    for row in rows:
        sys.stdout.write(
            f"{row.concurrency:>11}  {row.rows:>6}  {row.binding:<7}  "
            f"{row.hard_floor_ms:>13.3f}  {row.serial_estimate_ms:>18.3f}  "
            f"{row.tpot_floor_ms:>13.3f}  {row.interactivity_ceil_tok_s:>24.1f}  "
            f"{row.output_tput_ceil_per_chip:>25.1f}  {row.flag:<7}\n"
        )
    return 0


def _dispatch_explain_cmd(
    args: argparse.Namespace,
    model_ctx: _CliModelContext,
) -> int:
    """Execute explain for per-module physical step breakdowns."""
    _require_positive_int("--batch", args.batch)
    _require_positive_int("--rows-per-seq", args.rows_per_seq)
    parallel, dtypes, latency_s = _build_plan_parallel_dtypes(args, model_ctx.config)
    opts = FloorOptions(dtypes=dtypes, collective_latency_s=latency_s)
    floor_result = (
        spec_cycle_floor(
            model_ctx.geometry,
            model_ctx.chip,
            parallel,
            SpecShape(batch=args.batch, context=args.context, spec_k=args.spec_window),
            opts,
        )
        if args.spec_window > 0
        else step_floor(
            geometry=model_ctx.geometry,
            chip=model_ctx.chip,
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
                **asdict(module),
                "ici_bytes": module.ici_bytes,
                "n_collectives": module.n_collectives,
                "chain_hops": module.chain_hops,
            }
            for module in floor_result.modules
        ]
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0
    sys.stdout.write(
        f"{'module':<10}  {'hbm_mb':>10}  {'fp8_gflops':>10}  "
        f"{'bf16_gflops':>11}  {'int8_gflops':>11}  {'fp4_gflops':>10}  "
        f"{'int4_gflops':>11}  {'vector_gflops':>13}  {'collectives':>11}  "
        f"{'wire_kb':>10}  {'hops':>6}\n"
    )
    for module in floor_result.modules:
        wire_kb = sum(coll.wire_bytes for coll in module.collectives) / UNIT_KILO
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


def _resolve_model_source(args: argparse.Namespace) -> str:
    """Resolve model directory, HF repo ID, or local config path string."""
    if args.model is not None:
        return str(args.model)
    if args.config is not None:
        return str(args.config)
    return ""


def _write_inspect_text(payload: Mapping[str, object]) -> None:
    """Write human-readable config inspection summary to stdout."""
    lines = [
        f"schema_version: {payload['schema_version']}",
        f"chip: {payload['chip']}",
        f"onboarding_tier: {payload['onboarding_tier']}",
        f"canonical_id: {payload['canonical_id'] or '<unlisted>'}",
        f"hf_repo: {payload['hf_repo'] or '<local>'}",
        f"model_type: {payload['model_type'] or '<omitted>'}",
        f"model_type_status: {payload['model_type_status']}",
    ]
    params = payload.get("param_counts")
    if isinstance(params, Mapping):
        lines.append(
            "param_counts: "
            f"total={params['total_params']:.0f}, "
            f"active={params['active_params_per_token']:.0f}"
        )
    lines.append(f"unmapped_structural_keys: {payload.get('unmapped_structural_keys')}")
    lines.append(f"ignored_non_text_keys: {payload.get('ignored_non_text_keys')}")
    reasons = payload.get("blocking_reasons")
    if reasons:
        lines.append(f"blocking_reasons: {reasons}")
    sys.stdout.write("\n".join(lines) + "\n")


def _dispatch_inspect_cmd(
    args: argparse.Namespace,
    source: str,
    chip: ChipSpec,
) -> int:
    """Audit HF config onboarding readiness before geometry loading fails."""
    try:
        overrides = _parse_hf_overrides(args.hf_overrides)
        config = load_hf_config(source)
        if overrides:
            config = apply_hf_overrides(config, overrides)
    except (FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return EXIT_ERROR
    model_ref = None if Path(source).exists() else source
    audit = audit_hf_config(config, model_ref=model_ref)
    canonical_id: str | None = None
    hf_repo: str | None = None
    try:
        identity = resolve_model_identity(config, model_ref=model_ref)
        canonical_id = identity.canonical_id
        hf_repo = identity.hf_repo
    except UnsupportedArchitectureError:
        canonical_id = None
        hf_repo = None
    payload: dict[str, object] = {
        "schema_version": INSPECT_SCHEMA,
        "chip": chip.name,
        "onboarding_tier": audit.onboarding_tier,
        "canonical_id": canonical_id,
        "hf_repo": hf_repo,
        "model_type": audit.model_type,
        "model_type_status": audit.model_type_status,
        "layer_counts": (
            asdict(audit.layer_counts) if audit.layer_counts is not None else None
        ),
        "param_counts": (
            asdict(audit.param_counts) if audit.param_counts is not None else None
        ),
        "dtypes": asdict(audit.dtypes) if audit.dtypes is not None else None,
        "unmapped_structural_keys": list(audit.unmapped_structural_keys),
        "ignored_non_text_keys": list(audit.ignored_non_text_keys),
        "blocking_reasons": list(audit.blocking_reasons),
    }
    if args.json:
        sys.stdout.write(json.dumps(payload) + "\n")
    else:
        _write_inspect_text(payload)
    return 0 if audit.onboarding_tier == "tier0_ready" else EXIT_ERROR


def _load_cli_geometry(
    source: str,
    *,
    hf_overrides: dict[str, object] | None = None,
) -> tuple[dict[str, object], ModelGeometry, str | None]:
    """Load HF config and parse ModelGeometry with canonical identity validation."""
    resolved_src = str(Path(source).resolve()) if Path(source).exists() else source
    config, identity = load_hf_config_with_identity(
        resolved_src,
        hf_overrides=hf_overrides,
    )
    geometry = from_hf_config(config, model_ref=identity.hf_repo)
    return (config, geometry, identity.hf_repo)


def _resolve_cli_chip(
    args: argparse.Namespace,
) -> tuple[ChipSpec, dict[str, SliceTopology]] | None:
    """Load --chip-spec or catalog chip and apply command-line overrides."""
    chip_spec_path = args.chip_spec
    if chip_spec_path is not None:
        try:
            chip, topo_map = load_chip_spec(chip_spec_path)
        except (OSError, ValueError, TypeError) as exc:
            sys.stderr.write(f"ERROR: failed to load --chip-spec: {exc}\n")
            return None
    else:
        chip = CHIPS[args.chip]
        topo_map = topologies_for_chip(chip)
    try:
        return _apply_chip_overrides(chip, args), topo_map
    except (ValueError, TypeError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return None


def _validate_subcommand_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Validate required subcommand arguments."""
    if not args.cmd:
        parser.error("the following arguments are required: cmd")
    if args.cmd in {"plan", "explain"}:
        if args.tp is None:
            parser.error("the following arguments are required: --tp")
        if args.tp <= 0:
            parser.error(f"--tp must be a positive integer (>= 1), got {args.tp}")


def _dispatch_loaded_subcommand(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    model_ctx: _CliModelContext,
) -> int:
    """Dispatch solve, ceiling, plan, or explain on a loaded model context."""
    if args.cmd in {"solve", "ceiling"}:
        return _dispatch_solve_family(args, parser, model_ctx)
    if args.cmd == "plan":
        return _dispatch_plan_cmd(args, parser, model_ctx)
    return _dispatch_explain_cmd(args, model_ctx)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the roofline CLI subcommand."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    resolved_chip = _resolve_cli_chip(args)
    if resolved_chip is None:
        return EXIT_ERROR
    chip, chip_topologies = resolved_chip
    if args.assumptions:
        return _render_assumptions_view(chip, as_json=args.json)
    _validate_subcommand_args(args, parser)
    source = _resolve_model_source(args)
    if not source:
        parser.error("One of --config or --model is required.")
    if args.cmd == "inspect":
        return _dispatch_inspect_cmd(args, source, chip)
    try:
        overrides = _parse_hf_overrides(args.hf_overrides)
        config, geometry, _ = _load_cli_geometry(
            source,
            hf_overrides=overrides,
        )
        model_ctx = _CliModelContext(
            source=source,
            config=config,
            geometry=geometry,
            chip=chip,
            topologies=chip_topologies,
        )
        return _dispatch_loaded_subcommand(args, parser, model_ctx)
    except (
        UnsupportedArchitectureError,
        FileNotFoundError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
