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
"""Render vLLM serve CLI flags and uncalibrated L1 candidate curves from plans."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, TypedDict

from .constants import (
    KV_DTYPE_BYTES,
    NO_FEASIBLE_LAYOUT_WARNING,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .modules import Dtypes
    from .solver import PoolMeshPlan, ServingPlan

# Printed in a command in place of a model that ``vllm serve`` cannot load.
_MODEL_PLACEHOLDER = "<MODEL>"
_UNKNOWN_MODEL_WARNING = (
    f"The model to serve is not known: replace {_MODEL_PLACEHOLDER} in each "
    "command with a model ID or a model directory."
)


class LaunchPayload(TypedDict):
    """Structured vLLM serve CLI payload emitted for the top-ranked L1 candidate."""

    mode: str
    cli_args: list[str]
    prefill_cli_args: list[str]
    commands: list[str]
    warnings: list[str]
    serving_plan: Mapping[str, object]


class CeilingRow(TypedDict):
    """Single uncalibrated L1 candidate row exported by ``emit_ceiling_rows``."""

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


def _servable_model(source: str | None) -> str | None:
    """Return ``source`` unless it is None or names a ``.json`` config file."""
    if source is None or source.endswith(".json"):
        return None
    return source


def _format_server_cmd(cli_args: list[str], model: str | None) -> str:
    """Format launch CLI arguments as a standalone bash command string."""
    joined = " ".join(cli_args)
    prefix = f"vllm serve {_servable_model(model) or _MODEL_PLACEHOLDER}"
    return f"{prefix} {joined}".strip()


def emit_launch_args(
    plan: ServingPlan,
    *,
    dtypes: Dtypes | None = None,
    model: str | None = None,
) -> LaunchPayload:
    """Emit ``vllm serve`` CLI flags and commands for the top-ranked L1 candidate."""
    kv_str = (
        "fp8" if (dtypes is not None and dtypes.kv <= KV_DTYPE_BYTES["fp8"]) else "auto"
    )
    warnings = list(plan.warnings)
    rec = plan.recommended
    if rec is None or not rec.slo_met:
        if not warnings:
            warnings.append(NO_FEASIBLE_LAYOUT_WARNING)
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

    if _servable_model(model) is None:
        warnings.append(_UNKNOWN_MODEL_WARNING)
    return {
        "mode": rec.mode,
        "cli_args": decode_args,
        "prefill_cli_args": prefill_args,
        "commands": commands,
        "warnings": warnings,
        "serving_plan": asdict(plan),
    }


def emit_ceiling_rows(plan: ServingPlan) -> list[CeilingRow]:
    """Export uncalibrated L1 candidate rows by p90 interactivity and throughput."""
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
