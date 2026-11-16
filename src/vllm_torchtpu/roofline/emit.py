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
"""Render vLLM serve CLI flags, top-K layouts, and versioned solve payloads."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, TypedDict

from .constants import (
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    KV_DTYPE_BYTES,
    MIN_TOP_LAYOUTS_FOR_FASTEST,
    SOLVE_SCHEMA,
    TOPOLOGY_AXES,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .chips import ChipSpec
    from .modules import Dtypes
    from .solver import ParetoPoint, PoolMeshPlan, ServingPlan

_POOL_AXES: tuple[tuple[str, str], ...] = (
    ("TP", "tp"),
    ("DP", "dp"),
    ("EP", "ep"),
    ("PP", "pp"),
    ("PCP", "pcp"),
    ("DCP", "kv_split"),
)


class PointLaunchPayload(TypedDict):
    """Launch CLI arguments and formatted commands for a single Pareto point."""

    mode: str
    cli_args: list[str]
    prefill_cli_args: list[str]
    commands: list[str]


class LaunchPayload(TypedDict):
    """Structured vLLM serve CLI payload emitted for one serving plan."""

    mode: str
    cli_args: list[str]
    prefill_cli_args: list[str]
    commands: list[str]
    warnings: list[str]
    serving_plan: Mapping[str, object]


class ProvenancePayload(TypedDict):
    """Analytical model provenance metadata in solve JSON payloads."""

    calibrated: bool
    basis: str


class TopKEntry(TypedDict):
    """Ranked top layout entry with role label, throughput margin, and launch."""

    frontier_role: str
    margin_vs_rank1_pct: float
    point: Mapping[str, object]
    launch: PointLaunchPayload


class SolvePayload(TypedDict):
    """Versioned ``solve/v1`` JSON payload with provenance and Top-K launches."""

    model_layers: int
    chip: str
    scenario: Mapping[str, object]
    mode: str
    recommended: Mapping[str, object] | None
    frontier: tuple[Mapping[str, object], ...]
    warnings: tuple[str, ...]
    candidates: tuple[Mapping[str, object], ...]
    gpu_memory_utilization: float
    search: Mapping[str, object] | None
    schema: str
    provenance: ProvenancePayload
    chip_spec: Mapping[str, object]
    model: str | None
    top_k: list[TopKEntry]


class CeilingRow(TypedDict):
    """Single Pareto ceiling row exported by ``emit_ceiling_rows``."""

    p90_interactivity_tok_s: float
    token_throughput_per_chip: float
    output_throughput_per_chip: float
    p90_ttft_ms: float
    p90_tpot_ms: float
    concurrency: int
    total_chips: int
    mode: str
    binding_unit: str


type _PoolStructKey = tuple[str, int, int, bool, int, int, int]


def pool_cli_args(
    pool: PoolMeshPlan,
    *,
    kv_dtype: str = "auto",
    gpu_memory_utilization: float | None = None,
) -> list[str]:
    """Translate one ``PoolMeshPlan`` into ``vllm serve`` CLI flags."""
    args = [
        "--tensor-parallel-size",
        str(pool.tp),
        "--data-parallel-size",
        str(pool.dp),
        "--kv-cache-dtype",
        kv_dtype,
    ]
    if pool.pp > 1:
        args.extend(["--pipeline-parallel-size", str(pool.pp)])
    if pool.pcp > 1:
        args.extend(["--prefill-context-parallel-size", str(pool.pcp)])
    if pool.kv_split > 1:
        args.extend(["--decode-context-parallel-size", str(pool.kv_split)])
    if pool.ep > 1:
        args.append("--enable-expert-parallel")
    if pool.max_model_len is not None:
        args.extend(["--max-model-len", str(pool.max_model_len)])
    if pool.max_num_seqs is not None:
        args.extend(["--max-num-seqs", str(pool.max_num_seqs)])
    if gpu_memory_utilization is not None:
        args.extend(["--gpu-memory-utilization", f"{gpu_memory_utilization:g}"])
    return args


def _pool_int(val: object, default: int = 1) -> int:
    """Extract positive integer from dictionary values safely."""
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def format_parallelism(pool: Mapping[str, object]) -> str:
    """Format active parallelism axes joined by ' x ' in canonical order, or TP1."""
    parts = [
        f"{name}{_pool_int(pool.get(key))}"
        for name, key in _POOL_AXES
        if _pool_int(pool.get(key)) > 1
    ]
    return " x ".join(parts) if parts else "TP1"


def pool_label(pool: Mapping[str, object]) -> str:
    """Return a concise pool label.

    Example: ``4x 2x2x2 (32 chips, 8 hosts, 64 cores): TP16 x EP16 x DCP2``.
    """
    instance_count = _pool_int(pool.get("instances"))
    instance_prefix = f"{instance_count}x "
    topology_name = str(pool.get("topology", ""))
    chips_per_instance = _pool_int(pool.get("chips_per_instance"), default=0)
    if chips_per_instance <= 0 and topology_name in TOPOLOGY_AXES:
        topology_dimensions = TOPOLOGY_AXES[topology_name]
        chips_per_instance = (
            topology_dimensions[0] * topology_dimensions[1] * topology_dimensions[2]
        )
    total_chips = chips_per_instance * max(1, instance_count)
    layout_text = format_parallelism(pool)
    if total_chips <= 0:
        return f"{instance_prefix}{topology_name}: {layout_text}"
    hosts = max(1, total_chips // DEFAULT_CHIPS_PER_HOST)
    host_description = "1 host" if hosts == 1 else f"{hosts} hosts"
    cores = total_chips * DEFAULT_CORES_PER_CHIP
    pool_description = f" ({total_chips} chips, {host_description}, {cores} cores)"
    return f"{instance_prefix}{topology_name}{pool_description}: {layout_text}"


def _format_server_cmd(cli_args: list[str], model: str | None) -> str:
    """Format launch CLI arguments as a standalone bash command string."""
    joined = " ".join(cli_args)
    prefix = f"vllm serve {model}" if model else "vllm serve"
    return f"{prefix} {joined}".strip()


def _pool_key(pool: PoolMeshPlan | None) -> _PoolStructKey | None:
    """Group layouts by structural parallelism axes that emit distinct serve CLIs."""
    if pool is None:
        return None
    return (
        pool.topology,
        pool.tp,
        pool.dp,
        pool.ep > 1,
        pool.pp,
        pool.pcp,
        pool.kv_split,
    )


def _point_struct_key(
    pt: ParetoPoint,
) -> tuple[str, _PoolStructKey | None, _PoolStructKey | None]:
    """Build composite structural identity key across mixed or prefill/decode pools."""
    return (pt.mode, _pool_key(pt.prefill_pool), _pool_key(pt.decode_pool))


def top_layouts(plan: ServingPlan, k: int = 3) -> tuple[ParetoPoint, ...]:
    """Select up to k structurally distinct SLO-met layouts spanning Pareto roles."""
    if k <= 0:
        return ()
    best: dict[
        tuple[str, _PoolStructKey | None, _PoolStructKey | None], ParetoPoint
    ] = {}
    slo_cands: list[ParetoPoint] = []
    for pt in plan.candidates:
        if not pt.slo_met:
            continue
        slo_cands.append(pt)
        key = _point_struct_key(pt)
        cur = best.get(key)
        if cur is None or (
            pt.token_throughput_per_chip,
            pt.p90_interactivity_tok_s,
        ) > (
            cur.token_throughput_per_chip,
            cur.p90_interactivity_tok_s,
        ):
            best[key] = pt
    ranked = sorted(best.values(), key=lambda p: -p.token_throughput_per_chip)
    if len(ranked) <= 1 or k < MIN_TOP_LAYOUTS_FOR_FASTEST:
        return tuple(ranked[:k])
    selected = list(ranked[:k])
    fastest = max(
        slo_cands,
        key=lambda p: (p.p90_interactivity_tok_s, p.token_throughput_per_chip),
    )
    f_key = _point_struct_key(fastest)
    sel_keys = [_point_struct_key(p) for p in selected]
    if f_key not in sel_keys:
        selected[-1] = fastest
    elif sel_keys.index(f_key) > 0:
        selected[sel_keys.index(f_key)] = fastest
    selected.sort(key=lambda p: -p.token_throughput_per_chip)
    return tuple(selected)


def point_launch_args(
    pt: ParetoPoint,
    *,
    kv_dtype: str = "auto",
    gpu_memory_utilization: float | None = None,
    model: str | None = None,
) -> PointLaunchPayload:
    """Emit ``vllm serve`` CLI flags and commands for a single ParetoPoint."""
    decode_args = pool_cli_args(
        pt.decode_pool,
        kv_dtype=kv_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    prefill_args = (
        pool_cli_args(
            pt.prefill_pool,
            kv_dtype=kv_dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        if pt.prefill_pool is not None
        else []
    )
    decode_cmd = _format_server_cmd(decode_args, model)
    if pt.prefill_pool is not None:
        p_pool = pt.prefill_pool
        d_pool = pt.decode_pool
        prefill_cmd = _format_server_cmd(prefill_args, model)
        commands = [
            f"# prefill ({p_pool.instances}x {p_pool.topology}): {prefill_cmd}",
            f"# decode ({d_pool.instances}x {d_pool.topology}): {decode_cmd}",
        ]
    else:
        commands = [decode_cmd]

    return {
        "mode": pt.mode,
        "cli_args": decode_args,
        "prefill_cli_args": prefill_args,
        "commands": commands,
    }


def emit_launch_args(
    plan: ServingPlan,
    *,
    dtypes: Dtypes | None = None,
    model: str | None = None,
) -> LaunchPayload:
    """Emit ``vllm serve`` CLI flags and commands for ``plan``."""
    kv_str = (
        "fp8" if (dtypes is not None and dtypes.kv <= KV_DTYPE_BYTES["fp8"]) else "auto"
    )
    warnings = list(plan.warnings)
    rec = plan.recommended
    if rec is None or not rec.slo_met:
        if not warnings:
            warnings.append("No feasible layout met memory and latency constraints.")
        return {
            "mode": plan.mode,
            "cli_args": [],
            "prefill_cli_args": [],
            "commands": [],
            "warnings": warnings,
            "serving_plan": asdict(plan),
        }

    launch = point_launch_args(
        rec,
        kv_dtype=kv_str,
        gpu_memory_utilization=plan.gpu_memory_utilization,
        model=model,
    )
    return {
        "mode": launch["mode"],
        "cli_args": launch["cli_args"],
        "prefill_cli_args": launch["prefill_cli_args"],
        "commands": launch["commands"],
        "warnings": warnings,
        "serving_plan": asdict(plan),
    }


def build_solve_payload(
    plan: ServingPlan,
    *,
    chip: ChipSpec,
    dtypes: Dtypes | None = None,
    model: str | None = None,
    top_k: int = 3,
) -> SolvePayload:
    """Build ``solve/v1`` JSON payload with schema, provenance, chip, and Top-K."""
    kv_str = (
        "fp8" if (dtypes is not None and dtypes.kv <= KV_DTYPE_BYTES["fp8"]) else "auto"
    )
    top_pts = top_layouts(plan, top_k)
    top1_tput = top_pts[0].token_throughput_per_chip if top_pts else 0.0
    fastest_idx = (
        max(
            range(len(top_pts)),
            key=lambda i: (
                top_pts[i].p90_interactivity_tok_s,
                top_pts[i].token_throughput_per_chip,
            ),
        )
        if len(top_pts) > 1
        else -1
    )
    top_entries: list[TopKEntry] = [
        {
            "frontier_role": (
                "max_throughput"
                if idx == 0
                else ("max_interactivity" if idx == fastest_idx else "balanced_knee")
            ),
            "margin_vs_rank1_pct": round(
                100.0
                * (pt.token_throughput_per_chip - top1_tput)
                / max(1e-9, top1_tput),
                2,
            ),
            "point": asdict(pt),
            "launch": point_launch_args(
                pt,
                kv_dtype=kv_str,
                gpu_memory_utilization=plan.gpu_memory_utilization,
                model=model,
            ),
        }
        for idx, pt in enumerate(top_pts)
    ]
    return {
        "model_layers": plan.model_layers,
        "chip": plan.chip,
        "scenario": asdict(plan.scenario),
        "mode": plan.mode,
        "recommended": asdict(plan.recommended)
        if plan.recommended is not None
        else None,
        "frontier": tuple(asdict(p) for p in plan.frontier),
        "warnings": plan.warnings,
        "candidates": tuple(asdict(p) for p in plan.candidates),
        "gpu_memory_utilization": plan.gpu_memory_utilization,
        "search": asdict(plan.search) if plan.search is not None else None,
        "schema": SOLVE_SCHEMA,
        "provenance": {
            "calibrated": False,
            "basis": "analytical roofline floors; p90 values are modeled, not measured",
        },
        "chip_spec": asdict(chip),
        "model": model,
        "top_k": top_entries,
    }


def emit_ceiling_rows(plan: ServingPlan) -> list[CeilingRow]:
    """Export Pareto ceiling rows leading with p90 interactivity and throughput."""
    return [
        {
            "p90_interactivity_tok_s": pt.p90_interactivity_tok_s,
            "token_throughput_per_chip": pt.token_throughput_per_chip,
            "output_throughput_per_chip": pt.output_throughput_per_chip,
            "p90_ttft_ms": pt.p90_ttft_ms,
            "p90_tpot_ms": pt.p90_tpot_ms,
            "concurrency": pt.concurrency,
            "total_chips": pt.total_chips,
            "mode": pt.mode,
            "binding_unit": pt.binding,
        }
        for pt in plan.frontier
        if pt.slo_met
    ]
