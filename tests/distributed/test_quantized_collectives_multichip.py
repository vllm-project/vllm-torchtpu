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
"""Verify quantized_all_gather over real TPU collectives (8 ranks, one host).

`test_quantized_collectives.py` tests wire format and FP8 bounds against CPU
stand-ins; this runs `quantized_all_gather(x, group)` on a real 8-rank
`tpu_dist` GroupCoordinator, eager and under `torch.compile(backend="tpu")`,
enforces int8/float32 wire dtypes, and compares every rank's output against
`group.all_gather(x, dim=0)`.

Run on one host (4 chips x 2 cores = 8 ranks).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from vllm.distributed import parallel_state

from tests.distributed.test_quantized_collectives import fp8_round_trip_tol
from tests.distributed.tpu_test_utils import (
    launch_torchrun_workers,
    read_worker_results,
    run_torchrun_worker,
    torchrun_worker_env,
)
from vllm_torchtpu.distributed.collectives import quantized_all_gather

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.distributed.parallel_state import GroupCoordinator

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

REPO_ROOT = Path(__file__).resolve().parents[2]

WORLD_SIZE = 8
# Per-rank (rows, hidden) matching Qwen3.5-397B hidden_size=4096 at 512 and
# 8192 gathered tokens across 8 ranks.
SHAPES = ((64, 4096), (1024, 4096))
CALLS_PER_SHAPE = 2
# Stays under Buildkite's multichip lane pytest --timeout=600.
LAUNCH_TIMEOUT_SECONDS = 480
_MAX_TOL_RATIO = 1.0
# Up to 1 ULP for TPU reciprocal (1.0 / scale).
_TPU_RECIPROCAL_ULPS = 1.0


class _QuantizedWireGroup:
    """Forwarding group that rejects unquantized payloads on all_gather."""

    def __init__(self, inner: GroupCoordinator) -> None:
        """Wrap the real collective group."""
        self._inner = inner

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Require int8 rows or float32 scales before forwarding."""
        if tensor.dtype not in (torch.int8, torch.float32):
            msg = f"expected int8 or float32 wire dtype, got {tensor.dtype}"
            raise AssertionError(msg)
        return self._inner.all_gather(tensor, dim=dim)


def _check_shape(
    rank: int,
    fn: Callable[[torch.Tensor], torch.Tensor],
    group: GroupCoordinator,
    shape: tuple[int, int],
) -> list[dict[str, object]]:
    """Run CALLS_PER_SHAPE gathers for one shape and return per-call metrics."""
    rows, hidden = shape
    calls: list[dict[str, object]] = []
    for call_idx in range(CALLS_PER_SHAPE):
        gen = torch.Generator().manual_seed(rows * 7919 + hidden * 31 + call_idx)
        # Distinct per-rank amplitudes ensure any cross-rank scale mismatch fails.
        inputs = [
            (torch.randn(rows, hidden, generator=gen) * (0.5 * (r + 1))).to(
                torch.bfloat16
            )
            for r in range(WORLD_SIZE)
        ]
        inputs[0][0].zero_()
        x = inputs[rank].to("tpu")
        got = fn(x).cpu()
        group_got = group.all_gather(x, dim=0).cpu()
        want = torch.cat(inputs, dim=0)

        rtol, atol = fp8_round_trip_tol(group_got, reciprocal_ulps=_TPU_RECIPROCAL_ULPS)
        diff = (got.float() - group_got.float()).abs()
        tol = atol + rtol * group_got.float().abs()
        max_tol_ratio = float((diff / tol).max().item())
        rel_err = float(
            (got.float() - group_got.float()).norm() / group_got.float().norm()
        )
        calls.append(
            {
                "shape": list(got.shape),
                "dtype_ok": got.dtype == torch.bfloat16,
                "all_finite": bool(torch.isfinite(got).all().item()),
                "zero_row_exact": bool(torch.equal(got[0], torch.zeros_like(got[0]))),
                "group_matches_cpu": bool(torch.equal(group_got, want)),
                "max_tol_ratio": max_tol_ratio,
                "rel_err": rel_err,
            }
        )
    return calls


def _check(rank: int) -> dict[str, list[dict[str, object]]]:
    """Worker check: run eager and compiled quantized_all_gather on real ranks."""
    world = parallel_state.get_world_group()
    backend = torch.distributed.get_backend(world.device_group)
    group = parallel_state.init_model_parallel_group(
        [list(range(WORLD_SIZE))],
        world.local_rank,
        backend,
        group_name="quantized_ag",
    )
    wire_group = _QuantizedWireGroup(group)

    def run_gather(tensor: torch.Tensor) -> torch.Tensor:
        """Call quantized_all_gather over the wire-checked real 8-rank group."""
        return quantized_all_gather(tensor, wire_group)

    out: dict[str, list[dict[str, object]]] = {}
    for how in ("eager", "compiled"):
        for rows, hidden in SHAPES:
            fn = run_gather
            if how == "compiled":
                torch.compiler.reset()
                fn = torch.compile(
                    run_gather, backend="tpu", fullgraph=True, dynamic=False
                )
            out[f"{how}/{rows}x{hidden}"] = _check_shape(
                rank, fn, group, (rows, hidden)
            )
    return out


def test_quantized_all_gather_on_real_collectives(tmp_path: Path) -> None:
    """Verify quantized_all_gather against group.all_gather on 8 real TPU ranks."""
    result_dir = tmp_path / "quantized_ag"
    result_dir.mkdir()

    env = torchrun_worker_env(WORLD_SIZE)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT}{os.pathsep}{pythonpath}" if pythonpath else str(REPO_ROOT)
    )
    completed = launch_torchrun_workers(
        Path(__file__).resolve(),
        result_dir,
        world_size=WORLD_SIZE,
        env=env,
        cwd=REPO_ROOT,
        timeout=LAUNCH_TIMEOUT_SECONDS,
    )
    results = read_worker_results(result_dir, WORLD_SIZE, completed)
    for r in results:
        check = r["check"]
        for how in ("eager", "compiled"):
            for rows, hidden in SHAPES:
                calls = check[f"{how}/{rows}x{hidden}"]
                for call in calls:
                    assert call["shape"] == [WORLD_SIZE * rows, hidden], (
                        r["rank"],
                        how,
                        call,
                    )
                    assert call["dtype_ok"], (r["rank"], how, call)
                    assert call["all_finite"], (r["rank"], how, call)
                    assert call["zero_row_exact"], (r["rank"], how, call)
                    assert call["group_matches_cpu"], (r["rank"], how, call)
                    assert call["rel_err"] > 0, (r["rank"], how, call)
                    assert call["max_tol_ratio"] <= _MAX_TOL_RATIO, (
                        r["rank"],
                        how,
                        call,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        run_torchrun_worker(args.result_dir, _check)
