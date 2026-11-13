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
"""Command-line interface for the TPU serving roofline solver."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

from .chips import CHIPS, JAX_VERSION, ChipSpec, chip_catalog
from .constants import (
    BF16_BYTES,
    BITS_PER_BYTE,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_HBM_RESERVE_GB,
    DEFAULT_PREFILL_CHUNK,
    DEFAULT_TOP_K,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SLO_TIERS,
    EXPERT_DTYPE_BYTES,
    FP8_BYTES,
    KV_DTYPE_BYTES,
    MIN_INSTANCES,
    MIN_TOP_K,
    STATE_DTYPE_BYTES,
    UNIT_GIGA,
    UNIT_KILO,
    UNIT_MEGA,
    UNIT_MICROS_PER_SECOND,
    WEIGHT_DTYPE_BYTES,
)
from .emit import (
    _candidate_matches_layout,
    build_solve_payload,
    emit_ceiling_rows,
    emit_launch_args,
    parse_layout,
    pool_label,
    reference_layout_report,
    top_layout_roles,
    top_layouts,
)
from .geometry import from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .modules import Dtypes, Parallel, StepShape
from .report import write_html
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

_SERVE_STR_FLAGS: Final[dict[str, str]] = {
    "--model": "model",
    "--quantization": "quantization",
    "-q": "quantization",
    "--kv-cache-dtype": "kv_cache_dtype",
}

_SERVE_INT_FLAGS: Final[dict[str, str]] = {
    "--tensor-parallel-size": "tp",
    "-tp": "tp",
    "--data-parallel-size": "dp",
    "-dp": "dp",
    "--pipeline-parallel-size": "pp",
    "-pp": "pp",
    "--prefill-context-parallel-size": "pcp",
    "--max-num-seqs": "max_num_seqs",
    "--max-model-len": "max_model_len",
    "--max-num-batched-tokens": "max_num_batched_tokens",
}

_SERVE_FLOAT_FLAGS: Final[dict[str, str]] = {
    "--gpu-memory-utilization": "gpu_memory_utilization",
    "--kv-offloading-size": "kv_offload_gb",
}

_BENCH_STR_FLAGS: Final[dict[str, str]] = {
    "--model": "model",
}

_BENCH_INT_FLAGS: Final[dict[str, str]] = {
    "--random-input-len": "random_input_len",
    "--sonnet-input-len": "sonnet_input_len",
    "--input-len": "generic_input_len",
    "--random-output-len": "osl",
    "--output-len": "osl",
    "--sonnet-output-len": "osl",
    "--random-prefix-len": "shared_prefix_len",
    "--prefix-len": "shared_prefix_len",
    "--sonnet-prefix-len": "shared_prefix_len",
    "--max-model-len": "max_model_len",
    "--max-num-batched-tokens": "max_num_batched_tokens",
}

_DEFAULT_CMD_ISL: Final[int] = 1024
_DEFAULT_CMD_OSL: Final[int] = 128
_EXIT_EMPTY_FRONTIER: Final[int] = 1
_EXIT_ERROR: Final[int] = 2


def _normalize_tokens(raw_cmd: str) -> list[str]:
    """Split a shell command string and expand ``--flag=value`` tokens."""
    out: list[str] = []
    for tok in shlex.split(raw_cmd):
        if tok.startswith("--") and "=" in tok:
            flag, val = tok.split("=", 1)
            out.extend([flag, val])
        else:
            out.append(tok)
    return out


def _parse_spec_config_into(raw_spec: str, parsed: dict[str, object]) -> None:
    """Extract speculative decoding fields from JSON ``--speculative-config``."""
    parsed["speculative_config"] = raw_spec
    if not raw_spec.strip().startswith("{"):
        return
    try:
        spec_obj = json.loads(raw_spec)
    except json.JSONDecodeError:
        return
    if isinstance(spec_obj, Mapping):
        if "num_speculative_tokens" in spec_obj:
            parsed["spec_tokens"] = int(str(spec_obj["num_speculative_tokens"]))
        if "method" in spec_obj:
            parsed["spec_method"] = str(spec_obj["method"]).lower()
        if "synthetic_acceptance_length" in spec_obj:
            parsed["spec_accept_len"] = float(
                str(spec_obj["synthetic_acceptance_length"])
            )


def _parse_serve_cmd(raw_cmd: str) -> dict[str, object]:
    """Parse model, parallelism, memory, and spec flags from ``vllm serve``."""
    toks = _normalize_tokens(raw_cmd)
    parsed: dict[str, object] = {}
    i = 0
    while i < len(toks):
        tok = toks[i]
        has_val = i + 1 < len(toks)
        if tok == "serve" and has_val and not toks[i + 1].startswith("-"):
            parsed.setdefault("model", toks[i + 1])
            i += 2
            continue
        if has_val and tok == "--speculative-config":
            _parse_spec_config_into(toks[i + 1], parsed)
            i += 2
            continue
        if has_val and tok in _SERVE_STR_FLAGS:
            parsed[_SERVE_STR_FLAGS[tok]] = toks[i + 1]
            i += 2
            continue
        if has_val and tok in _SERVE_INT_FLAGS:
            parsed[_SERVE_INT_FLAGS[tok]] = int(toks[i + 1])
            i += 2
            continue
        if has_val and tok in _SERVE_FLOAT_FLAGS:
            parsed[_SERVE_FLOAT_FLAGS[tok]] = float(toks[i + 1])
            i += 2
            continue
        i += 1
    return parsed


def _parse_bench_cmd(raw_cmd: str) -> dict[str, object]:
    """Parse dataset, length, and concurrency flags from ``vllm bench serve``."""
    toks = _normalize_tokens(raw_cmd)
    parsed: dict[str, object] = {}
    i = 0
    while i < len(toks):
        tok = toks[i]
        has_val = i + 1 < len(toks)
        if has_val and tok == "--dataset-name":
            parsed["dataset_name"] = toks[i + 1].strip().lower()
            i += 2
            continue
        if has_val and tok == "--max-concurrency":
            parsed["concurrency"] = toks[i + 1]
            i += 2
            continue
        if has_val and tok in _BENCH_STR_FLAGS:
            parsed[_BENCH_STR_FLAGS[tok]] = toks[i + 1]
            i += 2
            continue
        if has_val and tok in _BENCH_INT_FLAGS:
            parsed[_BENCH_INT_FLAGS[tok]] = int(toks[i + 1])
            i += 2
            continue
        i += 1
    return parsed


def _add_model_source(p: argparse.ArgumentParser) -> None:
    """Attach optional ``--config`` / ``--model`` mutually exclusive inputs."""
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to local Hugging Face config.json file (unit: path, scope: model).",
    )
    src.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Hugging Face model ID, local directory, or config.json path "
            "(unit: id_or_path, scope: model)."
        ),
    )


def _add_shared_chip_flags(
    p: argparse.ArgumentParser,
    *,
    is_root: bool,
) -> None:
    """Attach chip selection, interconnect overrides, assumptions, and JSON flags."""
    sup = argparse.SUPPRESS
    p.add_argument(
        "--chip",
        default="tpu7x" if is_root else sup,
        choices=sorted(CHIPS),
        help="Target TPU chip specification key (unit: enum, scope: chip).",
    )
    p.add_argument(
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
    p.add_argument(
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
    p.add_argument(
        "--assumptions",
        action="store_true",
        default=False if is_root else sup,
        help=(
            "Print the hardware and solver constants catalog and exit "
            "(unit: flag, scope: chip)."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=False if is_root else sup,
        help=(
            "Emit machine-readable JSON output to stdout (unit: flag, scope: command)."
        ),
    )


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Attach flags shared across the ``plan`` and ``explain`` subcommands."""
    _add_model_source(p)
    _add_shared_chip_flags(p, is_root=False)
    p.add_argument(
        "--tp",
        type=int,
        default=None,
        help=(
            "Tensor parallel rank count across TensorCores (unit: ranks, scope: pool)."
        ),
    )
    p.add_argument(
        "--context",
        type=int,
        default=0,
        help=(
            "Active KV context length per sequence in tokens "
            "(unit: tokens, scope: request)."
        ),
    )
    p.add_argument(
        "--kv-split",
        type=int,
        default=1,
        help=(
            "Decode context parallel KV split factor across ranks "
            "(unit: ranks, scope: pool)."
        ),
    )
    p.add_argument(
        "--kv-merge",
        choices=("ag_rs", "exchange"),
        default="ag_rs",
        help=(
            "Context parallel attention merge collective schedule "
            "(unit: enum, scope: pool)."
        ),
    )
    p.add_argument(
        "--moe",
        choices=("tp_slice", "ep"),
        default="tp_slice",
        help="Routed MoE weight sharding mode across ranks (unit: enum, scope: pool).",
    )
    p.add_argument(
        "--moe-dispatch",
        choices=("allreduce", "ag_ep", "a2a_push", "fused_rs"),
        default="allreduce",
        help=(
            "MoE token dispatch and combine collective schedule "
            "(unit: enum, scope: pool)."
        ),
    )
    p.add_argument(
        "--residual",
        choices=("replicated", "sp"),
        default="replicated",
        help=(
            "Residual stream layout across tensor parallel ranks "
            "(unit: enum, scope: pool)."
        ),
    )
    p.add_argument(
        "--ar-algo",
        choices=("one_shot", "two_shot", "ring"),
        default="one_shot",
        help="All-reduce collective algorithm on ICI links (unit: enum, scope: link).",
    )
    p.add_argument(
        "--weight-dtype",
        choices=sorted(WEIGHT_DTYPE_BYTES),
        default=None,
        help="Dense linear projection weight storage dtype (unit: dtype, scope: core).",
    )
    p.add_argument(
        "--expert-dtype",
        choices=sorted(EXPERT_DTYPE_BYTES),
        default=None,
        help="Routed MoE expert weight storage dtype (unit: dtype, scope: core).",
    )
    p.add_argument(
        "--aux-dtype",
        choices=sorted(WEIGHT_DTYPE_BYTES),
        default=None,
        help="Router, embedding, and LM-head weight dtype (unit: dtype, scope: core).",
    )
    p.add_argument(
        "--kv-dtype",
        choices=sorted(KV_DTYPE_BYTES),
        default=None,
        help="Paged KV cache storage dtype in HBM (unit: dtype, scope: core).",
    )
    p.add_argument(
        "--state-dtype",
        choices=sorted(STATE_DTYPE_BYTES),
        default=None,
        help=(
            "Recurrent linear-attention state storage dtype (unit: dtype, scope: core)."
        ),
    )
    p.add_argument(
        "--act",
        "--act-dtype",
        dest="act",
        choices=("a8", "a16", "fp8", "bf16", "bfloat16"),
        default="a8",
        help=(
            "Matmul activation precision mode a8/fp8 or a16/bf16 "
            "(unit: enum, scope: core)."
        ),
    )
    p.add_argument(
        "--collective-latency-us",
        type=float,
        default=None,
        help=(
            "Override flat per-collective startup latency in microseconds "
            "(unit: us, scope: link)."
        ),
    )


def _add_scenario_args(p: argparse.ArgumentParser) -> None:
    """Attach scenario, budget, offload, and command import flags."""
    _add_model_source(p)
    _add_shared_chip_flags(p, is_root=False)
    p.add_argument(
        "--example-scenario",
        choices=sorted(EXAMPLE_SCENARIOS),
        default=None,
        help="Predefined traffic scenario preset name (unit: enum, scope: workload).",
    )
    p.add_argument(
        "--mode",
        choices=("mixed", "disagg", "auto"),
        default="auto",
        help=(
            "Pool serving architecture mode: mixed, disagg, or auto "
            "(unit: enum, scope: cluster)."
        ),
    )
    p.add_argument(
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
    p.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help=(
            "Maximum total serving instances across prefill and decode pools "
            "(unit: instances, scope: cluster)."
        ),
    )
    p.add_argument(
        "--example-slo-tier",
        choices=sorted(EXAMPLE_SLO_TIERS),
        default="none",
        help="Named TTFT/TPOT latency SLO preset tier (unit: enum, scope: workload).",
    )
    p.add_argument(
        "--ttft-slo-ms",
        type=float,
        default=None,
        help=(
            "Target maximum Time To First Token SLO in milliseconds "
            "(unit: ms, scope: request)."
        ),
    )
    p.add_argument(
        "--tpot-slo-ms",
        type=float,
        default=None,
        help=(
            "Target maximum Time Per Output Token SLO in milliseconds "
            "(unit: ms, scope: request)."
        ),
    )
    p.add_argument(
        "--isl",
        type=int,
        default=None,
        help=(
            "Input sequence length per request in tokens "
            "(unit: tokens, scope: request)."
        ),
    )
    p.add_argument(
        "--osl",
        type=int,
        default=None,
        help=(
            "Output sequence length per request in tokens "
            "(unit: tokens, scope: request)."
        ),
    )
    p.add_argument(
        "--hbm-reserve-gb",
        type=float,
        default=DEFAULT_HBM_RESERVE_GB,
        help=(
            "HBM reserved per core excluded from serving limits "
            "(unit: GB, scope: core)."
        ),
    )
    p.add_argument(
        "--kv-dtype",
        choices=sorted(KV_DTYPE_BYTES),
        default=None,
        help="Override paged KV cache storage dtype in HBM (unit: dtype, scope: core).",
    )
    p.add_argument(
        "--weight-dtype",
        choices=sorted(WEIGHT_DTYPE_BYTES),
        default=None,
        help="Dense linear projection weight storage dtype (unit: dtype, scope: core).",
    )
    p.add_argument(
        "--expert-weight-dtype",
        choices=sorted(EXPERT_DTYPE_BYTES),
        default=None,
        help="Routed MoE expert weight storage dtype (unit: dtype, scope: core).",
    )
    p.add_argument(
        "--act-dtype",
        "--act",
        dest="act_dtype",
        choices=("a8", "a16", "fp8", "bf16", "bfloat16"),
        default=None,
        help=(
            "Override matmul activation precision mode a8/fp8 or a16/bf16 "
            "(unit: enum, scope: core)."
        ),
    )
    p.add_argument(
        "--prefill-topology",
        type=str,
        default=None,
        help="Target prefill pool slice mesh topology (unit: mesh_str, scope: pool).",
    )
    p.add_argument(
        "--decode-topology",
        type=str,
        default=None,
        help="Target decode pool slice mesh topology (unit: mesh_str, scope: pool).",
    )
    p.add_argument(
        "--pd",
        type=str,
        default=None,
        help=(
            "Prefill:decode instance ratio (e.g. 1:1, 1:2) "
            "(unit: ratio_str, scope: cluster)."
        ),
    )
    p.add_argument(
        "--concurrency",
        type=str,
        default=None,
        help=(
            "Comma-separated list of concurrent request counts to evaluate "
            "(unit: requests, scope: workload)."
        ),
    )
    p.add_argument(
        "--prefix-hit-rate",
        type=float,
        default=None,
        help=(
            "Cached prefix token hit ratio in [0.0, 1.0] (unit: ratio, scope: request)."
        ),
    )
    p.add_argument(
        "--shared-prefix-len",
        type=int,
        default=None,
        help="Shared cached prefix length in tokens (unit: tokens, scope: request).",
    )
    p.add_argument(
        "--random-prefix-len",
        type=int,
        default=None,
        help=(
            "Benchmark random cached prefix length in tokens "
            "(unit: tokens, scope: request)."
        ),
    )
    p.add_argument(
        "--kv-offload-gb",
        type=float,
        default=None,
        help=(
            "Host DRAM KV cache offloading capacity per instance in GiB "
            "(unit: GiB, scope: host)."
        ),
    )
    p.add_argument(
        "--from-serve-cmd",
        type=str,
        default=None,
        help=(
            "Extract model and parallelism defaults from a vllm serve command string "
            "(unit: cmd_str, scope: command)."
        ),
    )
    p.add_argument(
        "--from-bench-cmd",
        type=str,
        default=None,
        help=(
            "Extract traffic scenario defaults from a vllm bench serve command "
            "string (unit: cmd_str, scope: command)."
        ),
    )
    p.add_argument(
        "--accept-len",
        type=float,
        default=None,
        help=(
            "Mean accepted tokens per speculative verification step >= 1.0 "
            "(unit: tokens/step, scope: request)."
        ),
    )
    p.add_argument(
        "--think-time-s",
        type=float,
        default=None,
        help="Client think time between turns in seconds (unit: s, scope: request).",
    )
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "Fraction of per-core HBM capacity usable for weights, state, "
            "and KV cache (unit: ratio, scope: core)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser across all roofline subcommands."""
    parser = argparse.ArgumentParser(prog="vllm_torchtpu.roofline")
    parser.set_defaults(
        weight_dtype=None,
        expert_weight_dtype=None,
        kv_dtype=None,
        pd=None,
        max_instances=None,
        prefill_topology=None,
        decode_topology=None,
        reference_layout=None,
    )
    _add_shared_chip_flags(parser, is_root=True)
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_solve = sub.add_parser("solve", help="Find Pareto serving layouts.")
    _add_scenario_args(p_solve)
    p_solve.add_argument(
        "--html",
        type=Path,
        default=None,
        help=(
            "Write a self-contained offline HTML report to file path "
            "(unit: path, scope: command)."
        ),
    )
    p_solve.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Maximum diverse Pareto layouts to display and emit "
            "(unit: count, scope: command)."
        ),
    )
    p_solve.add_argument(
        "--reference-layout",
        type=str,
        default=None,
        help=(
            "Reference layout to rank against search space "
            "(unit: layout specification, scope: command)."
        ),
    )

    p_emit = sub.add_parser("emit", help="Emit vllm serve flags.")
    _add_scenario_args(p_emit)

    p_ceil = sub.add_parser("ceiling", help="Emit Pareto ceiling rows.")
    _add_scenario_args(p_ceil)

    p_plan = sub.add_parser("plan", help="Sweep step floors.")
    _add_common_args(p_plan)
    p_plan.add_argument(
        "--concurrency",
        type=str,
        default="1,4,8,16,32,64",
        help=(
            "Comma-separated decode batch concurrencies to evaluate "
            "(unit: requests, scope: pool)."
        ),
    )
    p_plan.add_argument(
        "--spec-window",
        type=int,
        default=0,
        help=(
            "Speculative draft token window k per verification step "
            "(unit: tokens, scope: step)."
        ),
    )
    p_plan.add_argument(
        "--acceptance-length",
        type=float,
        default=None,
        help=(
            "Mean accepted tokens per verification step when spec-window > 0 "
            "(unit: tokens/step, scope: step)."
        ),
    )

    p_exp = sub.add_parser("explain", help="Print per-module breakdown.")
    _add_common_args(p_exp)
    p_exp.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Active sequence batch size per step (unit: requests, scope: step).",
    )
    p_exp.add_argument(
        "--rows-per-seq",
        type=int,
        default=1,
        help=(
            "Token rows processed per sequence in the step "
            "(unit: tokens/seq, scope: step)."
        ),
    )
    p_exp.add_argument(
        "--spec-window",
        type=int,
        default=0,
        help=(
            "Speculative draft token window k per verification step "
            "(unit: tokens, scope: step)."
        ),
    )
    return parser


def _apply_chip_overrides(chip: ChipSpec, args: argparse.Namespace) -> ChipSpec:
    """Apply CLI ICI and DCN interconnect overrides cleanly onto ``chip``."""
    out = chip
    if args.ici_bw_gbps is not None:
        ici_val = args.ici_bw_gbps * UNIT_GIGA
        out = replace(
            out,
            ici_bw=ici_val,
            d2d_bw=ici_val * float(out.cores_per_chip),
        )
    if args.dcn_gbps is not None:
        out = replace(out, dcn_bw=(args.dcn_gbps * UNIT_GIGA) / BITS_PER_BYTE)
    return out


def _render_assumptions_view(chip: ChipSpec, *, as_json: bool) -> int:
    """Render the live hardware and solver constants catalog."""
    cat = chip_catalog(chip)
    if as_json:
        payload = {"chip": chip.name, "jax_version": JAX_VERSION, "catalog": cat}
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0
    sys.stdout.write(
        f"Constants catalog for chip={chip.name} (jax_version={JAX_VERSION}):\n"
    )
    sys.stdout.write(
        f"{'name':<30}  {'display':<20}  {'raw_value':>14}  {'unit':<8}  "
        f"{'scope':<6}  {'kind':<8}  {'description':<36}  source\n"
    )
    for name, entry in cat.items():
        val = entry["value"]
        val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
        display_str = str(entry.get("display") or f"{val_str} {entry['unit']}")
        desc_str = str(entry.get("description") or "")
        sys.stdout.write(
            f"{name:<30}  {display_str:<20}  {val_str:>14}  {entry['unit']:<8}  "
            f"{entry['scope']:<6}  {entry['kind']:<8}  {desc_str:<32}  "
            f"{entry['source']}\n"
        )
    return 0


def _resolve_bench_isl_and_hit(
    args: argparse.Namespace,
    bench_parsed: dict[str, object],
) -> tuple[int | None, float | None, int | None]:
    """Resolve effective ISL and prefix hit rate from CLI and bench flags."""
    cli_prefix_len = (
        args.shared_prefix_len
        if args.shared_prefix_len is not None
        else args.random_prefix_len
    )
    shared_p = (
        cli_prefix_len
        if cli_prefix_len is not None
        else (
            int(str(bench_parsed["shared_prefix_len"]))
            if "shared_prefix_len" in bench_parsed
            else None
        )
    )
    eff_isl = args.isl
    if eff_isl is None:
        ds_name = str(bench_parsed.get("dataset_name", "random"))
        if "sonnet_input_len" in bench_parsed:
            eff_isl = int(str(bench_parsed["sonnet_input_len"]))
        elif "random_input_len" in bench_parsed:
            eff_isl = int(str(bench_parsed["random_input_len"])) + (shared_p or 0)
        elif "generic_input_len" in bench_parsed:
            base_in = int(str(bench_parsed["generic_input_len"]))
            eff_isl = base_in if ds_name == "sonnet" else base_in + (shared_p or 0)
    eff_hit = args.prefix_hit_rate
    if eff_hit is None:
        if shared_p is not None and eff_isl is not None and eff_isl > 0:
            eff_hit = min(1.0, max(0.0, float(shared_p) / float(eff_isl)))
        elif bench_parsed:
            eff_hit = 0.0
    return (eff_isl, eff_hit, cli_prefix_len)


def _stated_accept_len(
    args: argparse.Namespace, serve_parsed: dict[str, object]
) -> float | None:
    """Return the acceptance length stated by the user, else None.

    ``--accept-len`` wins over ``synthetic_acceptance_length`` in
    ``--speculative-config``.
    """
    if args.accept_len is not None:
        return float(args.accept_len)
    if "spec_accept_len" in serve_parsed:
        return float(str(serve_parsed["spec_accept_len"]))
    return None


def _resolve_cmd_rates(
    args: argparse.Namespace,
    serve_parsed: dict[str, object],
    *,
    from_cmd_mode: bool,
) -> tuple[int, float | None, float | None, float, float]:
    """Resolve speculative window, acceptance, think time, offload, and GMU."""
    eff_spec = int(str(serve_parsed.get("spec_tokens", 0)))
    eff_accept = _stated_accept_len(args, serve_parsed)
    if eff_accept is None and eff_spec > 0:
        # Every draft accepted is the largest acceptance length, so the
        # time-per-token bound stays valid when no length is stated.
        eff_accept = float(eff_spec + 1)
        if args.example_scenario is None:
            serve_parsed["spec_accept_assumed"] = True
    elif eff_accept is None and from_cmd_mode:
        eff_accept = 1.0
    eff_think = (
        args.think_time_s
        if args.think_time_s is not None
        else (0.0 if from_cmd_mode else None)
    )
    eff_offload = (
        args.kv_offload_gb
        if args.kv_offload_gb is not None
        else (
            float(str(serve_parsed["kv_offload_gb"]))
            if "kv_offload_gb" in serve_parsed
            else 0.0
        )
    )
    eff_gmu = (
        args.gpu_memory_utilization
        if args.gpu_memory_utilization is not None
        else (
            float(str(serve_parsed["gpu_memory_utilization"]))
            if "gpu_memory_utilization" in serve_parsed
            else DEFAULT_GPU_MEMORY_UTILIZATION
        )
    )
    return (eff_spec, eff_accept, eff_think, eff_offload, eff_gmu)


def _apply_example_overrides(
    scen: TrafficScenario,
    overrides: tuple[float, int | None, int, int],
) -> TrafficScenario:
    """Apply parsed serve/bench overrides onto an example TrafficScenario."""
    eff_offload, eff_max_seqs, eff_spec, eff_chunk = overrides
    if eff_offload > 0.0:
        scen = replace(scen, kv_offload_gb=eff_offload)
    if eff_max_seqs is not None:
        scen = replace(scen, max_num_seqs=eff_max_seqs)
    if eff_spec > 0:
        scen = replace(scen, spec_tokens=eff_spec)
    if eff_chunk > 0:
        scen = replace(scen, prefill_chunk=eff_chunk)
    return scen


def _build_cli_scenario(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    parsed: tuple[dict[str, object], dict[str, object]],
) -> tuple[TrafficScenario, float, int]:
    """Build TrafficScenario, GMU, and max_model_len cap from CLI and cmds."""
    serve_parsed, bench_parsed = parsed
    if (
        args.shared_prefix_len is not None
        and args.random_prefix_len is not None
        and args.shared_prefix_len != args.random_prefix_len
    ):
        parser.error(
            "--shared-prefix-len and --random-prefix-len both specify "
            "cached prefix tokens and cannot differ."
        )
    eff_conc_str = args.concurrency or (
        str(bench_parsed["concurrency"]) if "concurrency" in bench_parsed else None
    )
    concs = (
        tuple(int(x) for x in eff_conc_str.split(",") if x.strip())
        if eff_conc_str
        else None
    )
    eff_osl = (
        args.osl
        if args.osl is not None
        else (int(str(bench_parsed["osl"])) if "osl" in bench_parsed else None)
    )
    eff_isl, eff_hit, cli_prefix_len = _resolve_bench_isl_and_hit(args, bench_parsed)
    from_cmd_mode = bool(serve_parsed or bench_parsed or cli_prefix_len is not None)
    eff_spec, eff_accept, eff_think, eff_offload, eff_gmu = _resolve_cmd_rates(
        args, serve_parsed, from_cmd_mode=from_cmd_mode
    )
    eff_chunk = int(
        str(
            serve_parsed.get("max_num_batched_tokens")
            or bench_parsed.get("max_num_batched_tokens")
            or 0
        )
    )
    cmd_max_len = int(
        str(serve_parsed.get("max_model_len") or bench_parsed.get("max_model_len") or 0)
    )
    eff_max_seqs = (
        int(str(serve_parsed["max_num_seqs"]))
        if "max_num_seqs" in serve_parsed
        else None
    )
    if from_cmd_mode:
        eff_isl = _DEFAULT_CMD_ISL if eff_isl is None else eff_isl
        eff_osl = _DEFAULT_CMD_OSL if eff_osl is None else eff_osl
        eff_hit = 0.0 if eff_hit is None else eff_hit
    if args.example_scenario is not None:
        stated_accept_len = _stated_accept_len(args, serve_parsed)
        scen = build_example_scenario(
            args.example_scenario,
            slo_tier=args.example_slo_tier,
            overrides=ScenarioOverrides(
                isl=eff_isl,
                osl=eff_osl,
                concurrencies=concs,
                prefix_hit_rate=eff_hit,
                accept_len=stated_accept_len,
                think_time_s=args.think_time_s,
                ttft_slo_ms=args.ttft_slo_ms,
                tpot_slo_ms=args.tpot_slo_ms,
            ),
        )
        example_spec_tokens = scen.spec_tokens or 0
        scen = _apply_example_overrides(
            scen, (eff_offload, eff_max_seqs, eff_spec, eff_chunk)
        )
        if (
            stated_accept_len is None
            and eff_spec > 0
            and eff_spec != example_spec_tokens
        ):
            # The example's acceptance length describes its own draft count. With
            # another draft count and no stated length, every draft is assumed
            # accepted, which keeps the time-per-token bound valid.
            scen = replace(scen, accept_len=float(eff_spec + 1))
            serve_parsed["spec_accept_assumed"] = True
        return (scen, eff_gmu, cmd_max_len)
    if (
        eff_isl is None
        or eff_osl is None
        or eff_hit is None
        or eff_accept is None
        or eff_think is None
    ):
        parser.error(
            "Either --example-scenario or all of --isl, --osl, "
            "--prefix-hit-rate, --accept-len, and --think-time-s "
            "are required."
        )
    scen = TrafficScenario(
        name="custom",
        isl=eff_isl,
        osl=eff_osl,
        prefix_hit_rate=eff_hit,
        accept_len=eff_accept,
        think_time_s=eff_think,
        concurrencies=concs if concs is not None else EXAMPLE_CONCURRENCIES,
        kv_working_set_turns=1.0,
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
        kv_offload_gb=eff_offload,
        max_num_seqs=eff_max_seqs,
        spec_tokens=eff_spec if eff_spec > 0 else None,
        prefill_chunk=eff_chunk if eff_chunk > 0 else DEFAULT_PREFILL_CHUNK,
    )
    return (scen, eff_gmu, cmd_max_len)


def _cap_point_max_len(pt: ParetoPoint, max_len: int) -> ParetoPoint:
    """Cap ParetoPoint pool ``max_model_len`` values from CLI serve cmd."""
    p_pool = (
        replace(pt.prefill_pool, max_model_len=max_len)
        if pt.prefill_pool is not None
        else None
    )
    return replace(
        pt,
        decode_pool=replace(pt.decode_pool, max_model_len=max_len),
        prefill_pool=p_pool,
    )


def _apply_cmd_postprocess(
    splan: ServingPlan,
    scen: TrafficScenario,
    serve_parsed: dict[str, object],
    cmd_max_model_len: int,
) -> ServingPlan:
    """Apply ``--max-model-len`` caps and speculative warnings to ServingPlan."""
    cmd_warnings: list[str] = []
    if serve_parsed.get("spec_method") and serve_parsed["spec_method"] != "mtp":
        cmd_warnings.append(
            f"Speculative method '{serve_parsed['spec_method']}' is not MTP; "
            "roofline assumes MTP draft geometry."
        )
    if serve_parsed.get("spec_accept_assumed"):
        cmd_warnings.append(
            "No acceptance length was stated (--accept-len, or "
            "synthetic_acceptance_length in --speculative-config); every draft "
            "token is assumed accepted. The time-per-token estimate needs a "
            "stated acceptance length."
        )
    if cmd_max_model_len > 0:
        if cmd_max_model_len < (scen.isl + scen.osl):
            cmd_warnings.append(
                f"--max-model-len {cmd_max_model_len} is smaller than isl + osl "
                f"({scen.isl + scen.osl})."
            )
        new_rec = (
            _cap_point_max_len(splan.recommended, cmd_max_model_len)
            if splan.recommended is not None
            else None
        )
        splan = replace(
            splan,
            recommended=new_rec,
            frontier=tuple(
                _cap_point_max_len(pt, cmd_max_model_len) for pt in splan.frontier
            ),
            candidates=tuple(
                _cap_point_max_len(pt, cmd_max_model_len) for pt in splan.candidates
            ),
        )
    if cmd_warnings:
        splan = replace(splan, warnings=splan.warnings + tuple(cmd_warnings))
    return splan


def _render_ceiling_cmd(args: argparse.Namespace, splan: ServingPlan) -> int:
    """Render ceiling subcommand output and return exit code."""
    rows_ceil = emit_ceiling_rows(splan)
    if not rows_ceil:
        for w in splan.warnings:
            sys.stderr.write(f"WARNING: {w}\n")
        if args.json:
            sys.stdout.write("[]\n")
        return _EXIT_EMPTY_FRONTIER
    if args.json:
        sys.stdout.write(json.dumps(rows_ceil) + "\n")
        return 0
    for rc_row in rows_ceil:
        sys.stdout.write(json.dumps(rc_row) + "\n")
    return 0


def _render_emit_cmd(
    args: argparse.Namespace,
    splan: ServingPlan,
    dt_auto: Dtypes,
    model_arg: str | None,
) -> int:
    """Render emit subcommand output and return exit code."""
    emitted = emit_launch_args(splan, dtypes=dt_auto, model=model_arg)
    if not emitted["commands"]:
        for w in emitted["warnings"]:
            sys.stderr.write(f"WARNING: {w}\n")
        if args.json:
            sys.stdout.write(json.dumps(emitted) + "\n")
        return _EXIT_EMPTY_FRONTIER
    if args.json:
        sys.stdout.write(json.dumps(emitted) + "\n")
        return 0
    for cmd_line in emitted["commands"]:
        sys.stdout.write(cmd_line + "\n")
    return 0


def _render_solve_ascii_table(
    splan: ServingPlan,
    top_k: int,
    dt: Dtypes | None = None,
    ref_report: Mapping[str, object] | None = None,
) -> None:
    """Write Top-K diverse Pareto layouts as an aligned terminal table."""
    if dt is not None:
        sys.stdout.write(
            f"# Dtypes: dense={dt.w_dense_dtype} ({dt.w_dense:.4g}B, "
            f"{dt.w_dense_source}), expert={dt.w_expert_dtype} "
            f"({dt.w_expert:.4g}B, {dt.w_expert_source}), "
            f"kv={dt.kv_dtype} ({dt.kv:.4g}B, {dt.kv_source}), act={dt.act}\n"
        )
    top_pts = top_layouts(splan, k=top_k)
    annotated = top_layout_roles(top_pts)
    pool_labels = [
        (
            f"P {pool_label(asdict(pt.prefill_pool))} + "
            f"D {pool_label(asdict(pt.decode_pool))}"
            if pt.prefill_pool is not None
            else pool_label(asdict(pt.decode_pool))
        )
        for pt, _, _ in annotated
    ]

    reference_candidate: ParetoPoint | None = None
    reference_pools_label = ""
    if (
        ref_report is not None
        and bool(ref_report.get("found"))
        and not bool(ref_report.get("in_top_k"))
    ):
        point_dict = ref_report.get("point")
        if point_dict:
            reference_candidate = next(
                (
                    candidate
                    for candidate in splan.candidates
                    if asdict(candidate) == point_dict
                ),
                None,
            )
            if reference_candidate is not None:
                reference_pools_label = (
                    f"P {pool_label(asdict(reference_candidate.prefill_pool))} + "
                    f"D {pool_label(asdict(reference_candidate.decode_pool))}"
                    if reference_candidate.prefill_pool is not None
                    else pool_label(asdict(reference_candidate.decode_pool))
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
    for rank, ((pt, role, _margin), pools) in enumerate(
        zip(annotated, pool_labels, strict=True), 1
    ):
        sys.stdout.write(
            f"{rank:>4}  {role:<17}  {pt.mode:<7}  "
            f"{pt.concurrency:>11}  {pt.total_chips:>5}  {pools:<{pool_col_width}}  "
            f"{pt.est_interactivity_tok_s:>13.2f}  "
            f"{pt.token_throughput_per_chip:>13.1f}  "
            f"{pt.bound_tpot_ms:>13.2f}  "
            f"{pt.est_ttft_ms:>11.1f}  {pt.est_tpot_ms:>11.2f}\n",
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
    if ref_report is not None and not bool(ref_report.get("found")):
        sys.stdout.write(
            f"# reference layout {ref_report['layout']}: "
            f"not enumerated ({ref_report['rejected_by']})\n"
        )


def _render_solve_output(
    args: argparse.Namespace,
    splan: ServingPlan,
    meta: tuple[ChipSpec, Dtypes, str],
    ref_report: Mapping[str, object] | None = None,
) -> int:
    """Render solve subcommand HTML, JSON, or text table and return exit code."""
    chip, dt_auto, src = meta
    if args.html is not None:
        payload_html = build_solve_payload(
            splan,
            chip=chip,
            dtypes=dt_auto,
            model=src,
            top_k=args.top_k,
            reference=ref_report,
        )
        write_html(payload_html, args.html)
    warnings_to_emit = list(splan.warnings)
    if not splan.frontier and not warnings_to_emit:
        warnings_to_emit.append(
            "No feasible layout met memory and latency constraints."
        )
    for w in warnings_to_emit:
        sys.stderr.write(f"WARNING: {w}\n")
    if not splan.frontier:
        if args.json:
            payload = build_solve_payload(
                splan,
                chip=chip,
                dtypes=dt_auto,
                model=src,
                top_k=args.top_k,
                reference=ref_report,
            )
            sys.stdout.write(json.dumps(payload) + "\n")
        return _EXIT_EMPTY_FRONTIER
    if args.json:
        payload = build_solve_payload(
            splan,
            chip=chip,
            dtypes=dt_auto,
            model=src,
            top_k=args.top_k,
            reference=ref_report,
        )
        sys.stdout.write(json.dumps(payload) + "\n")
        return 0
    _render_solve_ascii_table(splan, args.top_k, dt_auto, ref_report=ref_report)
    sys.stdout.write(
        "# Note: bound_* columns are hard physical lower bounds; "
        "est_* columns are model estimates.\n",
    )
    return 0


def _resolve_solve_dtypes(
    args: argparse.Namespace,
    cfg: dict[str, object],
    serve_parsed: dict[str, object],
) -> Dtypes:
    """Resolve effective Dtypes for solve/emit/ceiling with CLI priority."""
    dt_auto = dtypes_from_hf_config(cfg)
    dense_val = dt_auto.w_dense
    dense_dt = dt_auto.w_dense_dtype
    dense_src = dt_auto.w_dense_source
    if args.weight_dtype is not None:
        dense_val = WEIGHT_DTYPE_BYTES[args.weight_dtype]
        dense_dt = args.weight_dtype
        dense_src = "flag"

    exp_val = dt_auto.w_expert
    exp_dt = dt_auto.w_expert_dtype
    exp_src = dt_auto.w_expert_source
    if args.expert_weight_dtype is not None:
        exp_val = EXPERT_DTYPE_BYTES[args.expert_weight_dtype]
        exp_dt = args.expert_weight_dtype
        exp_src = "flag"

    kv_val = dt_auto.kv
    kv_dt = dt_auto.kv_dtype
    kv_src = dt_auto.kv_source
    if args.kv_dtype is not None:
        kv_val = KV_DTYPE_BYTES[args.kv_dtype]
        kv_dt = args.kv_dtype
        kv_src = "flag"
    elif "kv_cache_dtype" in serve_parsed:
        s_dt = str(serve_parsed["kv_cache_dtype"]).lower()
        if s_dt.startswith("fp8"):
            kv_val = FP8_BYTES
            kv_dt = "fp8"
            kv_src = "flag"
        elif s_dt in ("bf16", "bfloat16", "fp16", "float16"):
            kv_val = BF16_BYTES
            kv_dt = "bfloat16"
            kv_src = "flag"

    act_val = dt_auto.act
    if getattr(args, "act_dtype", None) is not None:
        act_val = "a8" if args.act_dtype in ("a8", "fp8") else "a16"

    return replace(
        dt_auto,
        w_dense=dense_val,
        w_dense_dtype=dense_dt,
        w_dense_source=dense_src,
        w_expert=exp_val,
        w_expert_dtype=exp_dt,
        w_expert_source=exp_src,
        kv=kv_val,
        kv_dtype=kv_dt,
        kv_source=kv_src,
        act=act_val,
    )


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
    splan: ServingPlan,
    geom: ModelGeometry,
    parsed_ref: dict[str, dict[str, int]] | None,
    args: argparse.Namespace,
    solve_opts: SolveOptions,
) -> dict[str, object] | None:
    """Build reference layout comparison report when requested."""
    if parsed_ref is None or getattr(args, "reference_layout", None) is None:
        return None
    effective_solve_opts = (
        replace(solve_opts, max_chips=splan.budget)
        if solve_opts.max_chips is None
        else solve_opts
    )
    solve_topologies = _resolve_solve_topologies(effective_solve_opts)
    is_found = any(
        _candidate_matches_layout(candidate, parsed_ref)
        for candidate in splan.candidates
    )
    rejected_by = (
        ""
        if is_found
        else explain_layout_rejection(
            geom,
            parsed_ref,
            solve_topologies,
            effective_solve_opts,
        )
    )
    top_candidates = top_layouts(splan, k=args.top_k)
    return reference_layout_report(
        splan,
        top_candidates,
        args.reference_layout,
        rejected_by,
    )


def _dispatch_solve_family(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    model_ctx: tuple[str, dict[str, object], ModelGeometry, ChipSpec],
    parsed: tuple[dict[str, object], dict[str, object]],
) -> int:
    """Execute solve, emit, or ceiling subcommands."""
    if args.cmd == "solve" and args.top_k < MIN_TOP_K:
        parser.error(f"--top-k must be >= {MIN_TOP_K}, got {args.top_k}")
    if args.max_instances is not None and args.max_instances < MIN_INSTANCES:
        parser.error(
            f"--max-instances must be >= {MIN_INSTANCES}, got {args.max_instances}"
        )
    parsed_ref = _parse_cli_reference_layout(
        getattr(args, "reference_layout", None), parser
    )
    src, cfg, geom, chip = model_ctx
    serve_parsed, _ = parsed
    scen, eff_gmu, cmd_max_model_len = _build_cli_scenario(args, parser, parsed)
    pd_ratio = None
    if args.pd:
        parts = args.pd.split(":")
        pd_ratio = (int(parts[0]), int(parts[1]))

    eff_max_chips = args.max_chips
    if eff_max_chips is None and serve_parsed:
        s_tp = int(str(serve_parsed.get("tp", 1)))
        s_dp = int(str(serve_parsed.get("dp", 1)))
        s_pp = int(str(serve_parsed.get("pp", 1)))
        s_pcp = int(str(serve_parsed.get("pcp", 1)))
        chips_per_instance = max(
            1,
            math.ceil((s_tp * s_dp * s_pp * s_pcp) / max(1, chip.cores_per_chip)),
        )
        inst_cap = args.max_instances or 1
        eff_max_chips = chips_per_instance * inst_cap

    has_pins = bool(args.prefill_topology and args.decode_topology and args.pd)
    if eff_max_chips is None and not has_pins:
        parser.error(
            "Either --max-chips, complete pins "
            "(--prefill-topology, --decode-topology, --pd), "
            "or a serve command with --max-instances is required."
        )

    budget_source = (
        "flag"
        if args.max_chips is not None
        else ("pins" if has_pins else "serve_command")
    )
    dt_auto = _resolve_solve_dtypes(args, cfg, serve_parsed)
    solve_opts = SolveOptions(
        mode=args.mode,
        max_chips=eff_max_chips,
        max_instances=args.max_instances,
        gpu_memory_utilization=eff_gmu,
        hbm_reserve_gb=args.hbm_reserve_gb,
        prefill_topology=args.prefill_topology,
        decode_topology=args.decode_topology,
        pd_ratio=pd_ratio,
        budget_source=budget_source,
    )
    try:
        splan = solve_serving_plan(
            geom,
            dt_auto,
            chip,
            scen,
            solve_opts,
        )
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return _EXIT_ERROR
    splan = _apply_cmd_postprocess(splan, scen, serve_parsed, cmd_max_model_len)
    ref_report = _build_cli_reference_report(splan, geom, parsed_ref, args, solve_opts)
    if args.cmd == "ceiling":
        return _render_ceiling_cmd(args, splan)
    if args.cmd == "emit":
        model_arg = (
            args.model if args.model is not None else (src if serve_parsed else None)
        )
        return _render_emit_cmd(args, splan, dt_auto, model_arg)
    return _render_solve_output(
        args, splan, (chip, dt_auto, src), ref_report=ref_report
    )


def _build_plan_par_dtypes(
    args: argparse.Namespace,
    cfg: dict[str, object] | None = None,
) -> tuple[Parallel, Dtypes, float]:
    """Build Parallel, Dtypes, and latency override from plan/explain flags."""
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
    return (par, dt, lat_s)


def _dispatch_plan_cmd(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    geom: ModelGeometry,
    chip: ChipSpec,
    cfg: dict[str, object] | None = None,
) -> int:
    """Execute the plan subcommand across target concurrencies."""
    if args.spec_window > 0 and args.acceptance_length is None:
        parser.error("--acceptance-length is required when --spec-window > 0")
    par, dt, _ = _build_plan_par_dtypes(args, cfg)
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
        f"{'concurrency':>11}  {'rows':>6}  {'binding':<7}  "
        f"{'hard_floor_ms':>13}  {'serial_estimate_ms':>18}  "
        f"{'tpot_floor_ms':>13}  {'interactivity_ceil_tok_s':>24}  "
        f"{'output_tput_ceil_per_chip':>25}  {'flag':<7}\n"
    )
    sys.stdout.write(hdr)
    for r in rows:
        sys.stdout.write(
            f"{r.c:>11}  {r.rows:>6}  {r.binding:<7}  "
            f"{r.hard_floor_ms:>13.3f}  {r.serial_estimate_ms:>18.3f}  "
            f"{r.tpot_floor_ms:>13.3f}  {r.interactivity_ceil_tok_s:>24.1f}  "
            f"{r.output_tput_ceil_per_chip:>25.1f}  {r.flag:<7}\n"
        )
    return 0


def _dispatch_explain_cmd(
    args: argparse.Namespace,
    geom: ModelGeometry,
    chip: ChipSpec,
    cfg: dict[str, object] | None = None,
) -> int:
    """Execute the explain subcommand for per-module step breakdowns."""
    par, dt, lat_s = _build_plan_par_dtypes(args, cfg)
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


def _resolve_model_src(
    args: argparse.Namespace,
    serve_parsed: dict[str, object],
    bench_parsed: dict[str, object],
) -> str:
    """Resolve model directory, HF repo ID, or local config path string."""
    if args.model is not None:
        return str(args.model)
    if args.config is not None:
        return str(args.config)
    return str(serve_parsed.get("model") or bench_parsed.get("model") or "")


def _load_cli_geometry(
    src: str,
    serve_parsed: dict[str, object],
) -> tuple[dict[str, object], ModelGeometry]:
    """Load HF config and parse ModelGeometry with CLI quantization fallback."""
    cfg = load_hf_config(src)
    q_sub = cfg.get("text_config")
    has_qcfg = bool(
        cfg.get("quantization_config")
        or (isinstance(q_sub, Mapping) and q_sub.get("quantization_config"))
    )
    if "quantization" in serve_parsed and not has_qcfg:
        q_val = str(serve_parsed["quantization"]).lower()
        if q_val and q_val != "none":
            cfg = {**cfg, "quantization_config": {"quant_method": q_val}}
    return (cfg, from_hf_config(cfg))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the roofline CLI subcommand."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    chip = _apply_chip_overrides(CHIPS[args.chip], args)
    if args.assumptions:
        return _render_assumptions_view(chip, as_json=args.json)
    if not args.cmd:
        parser.error("the following arguments are required: cmd")
    if args.cmd in ("plan", "explain") and args.tp is None:
        parser.error("the following arguments are required: --tp")
    is_solve_cmd = args.cmd in ("solve", "emit", "ceiling")
    serve_parsed = (
        _parse_serve_cmd(args.from_serve_cmd)
        if is_solve_cmd and args.from_serve_cmd
        else {}
    )
    bench_parsed = (
        _parse_bench_cmd(args.from_bench_cmd)
        if is_solve_cmd and args.from_bench_cmd
        else {}
    )
    src = _resolve_model_src(args, serve_parsed, bench_parsed)
    if not src:
        parser.error(
            "One of --config, --model, or --from-serve-cmd with a model is required."
        )
    try:
        cfg, geom = _load_cli_geometry(src, serve_parsed)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return _EXIT_ERROR
    if is_solve_cmd:
        return _dispatch_solve_family(
            args, parser, (src, cfg, geom, chip), (serve_parsed, bench_parsed)
        )
    if args.cmd == "plan":
        return _dispatch_plan_cmd(args, parser, geom, chip, cfg=cfg)
    return _dispatch_explain_cmd(args, geom, chip, cfg=cfg)


if __name__ == "__main__":
    raise SystemExit(main())
