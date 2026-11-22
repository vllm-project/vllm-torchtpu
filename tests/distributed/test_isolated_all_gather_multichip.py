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
"""`isolated_all_gather` over a real TP group when the ranks' input work differs.

This is the data-parallel ViT pattern: a rank with images runs the vision
tower and a rank without only pads, then the outputs are all-gathered. Under
torch_tpu's DEFER_AND_FUSE eager mode a plain all-gather is fused into a
different program on each rank; one rank returns wrong rows and the other
hangs, so this launch times out without the isolation.

Run on one host (2 ranks).
"""

import argparse
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

REPO_ROOT = Path(__file__).resolve().parents[2]

WORLD_SIZE = 2
ROWS, HIDDEN = 256, 1024
LAUNCH_TIMEOUT_SECONDS = 300


def _tower_reference():
    import torch

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(ROWS, HIDDEN, generator=gen)
    w = torch.randn(HIDDEN, HIDDEN, generator=gen) / HIDDEN**0.5
    return x, w


def _check(rank: int) -> dict:
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import initialize_model_parallel

    from vllm_torchtpu.distributed.collectives import isolated_all_gather
    from vllm_torchtpu.platforms.tpu_platform import _configure_torchtpu_eager_mode

    _configure_torchtpu_eager_mode()
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=WORLD_SIZE)
    device = torch.device("tpu")
    x, w = _tower_reference()
    if rank == 0:
        local, w_dev = x.to(device), w.to(device)
        for _ in range(4):
            local = torch.relu(local @ w_dev)
    else:
        empty = torch.zeros((0, HIDDEN), device=device)
        local = torch.cat([empty, torch.zeros((ROWS, HIDDEN), device=device)])
    gathered = isolated_all_gather(local, dim=0).cpu()
    for _ in range(4):
        x = torch.relu(x @ w)
    return {
        "shape": list(gathered.shape),
        "rank0_rows_max_abs_err": float((gathered[:ROWS] - x).abs().max()),
        "rank0_rows_max_abs": float(x.abs().max()),
        "rank1_rows_abs_sum": float(gathered[ROWS:].abs().sum()),
    }


def test_isolated_all_gather_pairs_ranks_with_different_input_work(tmp_path):
    from tests.distributed.tpu_test_utils import (
        launch_torchrun_workers,
        read_worker_results,
        torchrun_worker_env,
    )

    result_dir = tmp_path / "isolated_all_gather"
    result_dir.mkdir()
    completed = launch_torchrun_workers(
        Path(__file__).resolve(),
        result_dir,
        world_size=WORLD_SIZE,
        env=torchrun_worker_env(WORLD_SIZE),
        cwd=REPO_ROOT,
        timeout=LAUNCH_TIMEOUT_SECONDS,
    )
    for r in read_worker_results(result_dir, WORLD_SIZE, completed):
        check = r["check"]
        assert check["shape"] == [WORLD_SIZE * ROWS, HIDDEN], (r["rank"], check)
        # Default TPU matmul precision against an f32 CPU reference; a
        # mispaired collective is off by the activations themselves.
        assert check["rank0_rows_max_abs_err"] <= 0.05 * check["rank0_rows_max_abs"], (
            r["rank"],
            check,
        )
        assert check["rank1_rows_abs_sum"] == 0.0, (r["rank"], check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        # Run as a script: the repo root is not on sys.path.
        sys.path.insert(0, str(REPO_ROOT))
        from tests.distributed.tpu_test_utils import run_torchrun_worker

        run_torchrun_worker(args.result_dir, _check)
