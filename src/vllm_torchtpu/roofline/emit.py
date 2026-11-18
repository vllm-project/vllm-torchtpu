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
"""Render vLLM serve CLI flags and Pareto ceiling curves from solved plans."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, TypedDict

from .constants import (
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    KV_DTYPE_BYTES,
    TOPOLOGY_AXES,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .modules import Dtypes
    from .solver import PoolMeshPlan, ServingPlan

_POOL_AXES: tuple[tuple[str, str], ...] = (
    ("TP", "tp"),
    ("DP", "dp"),
    ("EP", "ep"),
    ("PP", "pp"),
    ("PCP", "pcp"),
    ("DCP", "kv_split"),
)


class LaunchPayload(TypedDict):
    """Structured vLLM serve CLI payload emitted for one serving plan."""

    mode: str
    cli_args: list[str]
    prefill_cli_args: list[str]
    commands: list[str]
    warnings: list[str]
    serving_plan: Mapping[str, object]


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

    decode_args = pool_cli_args(
        rec.decode_pool,
        kv_dtype=kv_str,
        gpu_memory_utilization=plan.gpu_memory_utilization,
    )
    prefill_args = (
        pool_cli_args(
            rec.prefill_pool,
            kv_dtype=kv_str,
            gpu_memory_utilization=plan.gpu_memory_utilization,
        )
        if rec.prefill_pool is not None
        else []
    )
    decode_cmd = _format_server_cmd(decode_args, model)
    if rec.prefill_pool is not None:
        p_pool = rec.prefill_pool
        d_pool = rec.decode_pool
        prefill_cmd = _format_server_cmd(prefill_args, model)
        commands = [
            f"# prefill ({p_pool.instances}x {p_pool.topology}): {prefill_cmd}",
            f"# decode ({d_pool.instances}x {d_pool.topology}): {decode_cmd}",
        ]
    else:
        commands = [decode_cmd]

    return {
        "mode": rec.mode,
        "cli_args": decode_args,
        "prefill_cli_args": prefill_args,
        "commands": commands,
        "warnings": warnings,
        "serving_plan": asdict(plan),
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
