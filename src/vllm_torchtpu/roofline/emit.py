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
from typing import Any

from .chips import ChipSpec
from .modules import Dtypes
from .solver import ParetoPoint, PoolMeshPlan, ServingPlan

SOLVE_SCHEMA: str = "vllm_torchtpu.roofline.solve/v1"


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
    if gpu_memory_utilization is not None:
        args.extend(["--gpu-memory-utilization", f"{gpu_memory_utilization:g}"])
    return args


def _format_server_cmd(cli_args: list[str], model: str | None) -> str:
    joined = " ".join(cli_args)
    prefix = f"vllm serve {model}" if model else "vllm serve"
    return f"{prefix} {joined}".strip()


def _pool_key(pool: PoolMeshPlan | None) -> tuple | None:
    if pool is None:
        return None
    return (
        pool.topology,
        pool.instances,
        pool.tp,
        pool.dp,
        pool.ep,
        pool.pp,
        pool.pcp,
        pool.kv_split,
        pool.moe_dispatch,
    )


def top_layouts(plan: ServingPlan, k: int = 3) -> tuple[ParetoPoint, ...]:
    """Best SLO-met point per distinct layout, ranked by token throughput per chip."""
    best: dict[tuple, ParetoPoint] = {}
    for pt in plan.candidates:
        if not pt.slo_met:
            continue
        key = (pt.mode, _pool_key(pt.prefill_pool), _pool_key(pt.decode_pool))
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
    return tuple(ranked[:k])


def point_launch_args(
    pt: ParetoPoint,
    *,
    kv_dtype: str = "auto",
    gpu_memory_utilization: float | None = None,
    model: str | None = None,
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    """Emit ``vllm serve`` CLI flags and commands for ``plan``."""
    kv_str = "fp8" if (dtypes is not None and dtypes.kv <= 1.0) else "auto"
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
        **launch,
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
) -> dict[str, Any]:
    """`solve/v1` JSON: every ServingPlan key plus schema, provenance, chip, and Top-K."""
    kv_str = "fp8" if (dtypes is not None and dtypes.kv <= 1.0) else "auto"
    payload = asdict(plan)
    payload["schema"] = SOLVE_SCHEMA
    payload["provenance"] = {
        "calibrated": False,
        "basis": "analytical roofline floors; p90 values are modeled, not measured",
    }
    payload["chip_spec"] = asdict(chip)
    payload["model"] = model
    payload["top_k"] = [
        {
            "point": asdict(pt),
            "launch": point_launch_args(
                pt,
                kv_dtype=kv_str,
                gpu_memory_utilization=plan.gpu_memory_utilization,
                model=model,
            ),
        }
        for pt in top_layouts(plan, top_k)
    ]
    return payload


def emit_ceiling_rows(plan: ServingPlan) -> list[dict[str, Any]]:
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
