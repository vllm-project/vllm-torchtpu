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
"""vLLM's data-parallel vision path on TPU workers when ranks have no image.

run_dp_sharded_mrope_vision_model all-gathers each rank's encoder output right
after rank-local work: the rank holding the image runs the vision model, the
others only pad. Under torch_tpu's DEFER_AND_FUSE eager mode a plain all-gather
is fused into a different device program on each rank, so the ranks do not
pair and this launch times out.

Run on one host (8 ranks, like the other multichip launches, since the
launch environment of the first launch in a pytest process is reused).
"""

import argparse
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

REPO_ROOT = Path(__file__).resolve().parents[2]

WORLD_SIZE = 8
GRID = [1, 32, 32]
HIDDEN = 1024
LAUNCH_TIMEOUT_SECONDS = 300


class _VisionModel(nn.Module):
    """Merges every 4 patches into one row, then runs a few dense layers."""

    spatial_merge_size = 2
    out_hidden_size = HIDDEN

    def __init__(self) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.weights = nn.ParameterList(
            torch.randn(HIDDEN, HIDDEN, generator=gen) / HIDDEN**0.5 for _ in range(4)
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw_list) -> torch.Tensor:
        x = pixel_values.reshape(-1, HIDDEN)
        for w in self.weights:
            x = torch.relu(x @ w)
        return x


def _check(rank: int) -> dict:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import initialize_model_parallel
    from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model

    from vllm_torchtpu import patch_registry

    del rank
    patch_registry.apply("worker_init")
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=WORLD_SIZE)
    model = _VisionModel()
    pixels = torch.randn(
        math.prod(GRID), HIDDEN // 4, generator=torch.Generator().manual_seed(1)
    )
    with torch.no_grad():
        want = model(pixels, [GRID])
        (got,) = run_dp_sharded_mrope_vision_model(
            model.to("tpu"), pixels.to("tpu"), [GRID], rope_type="rope_3d"
        )
        got = got.cpu()
    return {
        "shape": list(got.shape),
        "max_abs_err": float((got - want).abs().max()),
        "max_abs": float(want.abs().max()),
    }


def test_dp_vision_all_gather_pairs_ranks_without_an_image(tmp_path):
    from tests.distributed.tpu_test_utils import (
        launch_torchrun_workers,
        read_worker_results,
        torchrun_worker_env,
    )

    completed = launch_torchrun_workers(
        Path(__file__).resolve(),
        tmp_path,
        world_size=WORLD_SIZE,
        env=torchrun_worker_env(WORLD_SIZE),
        cwd=REPO_ROOT,
        timeout=LAUNCH_TIMEOUT_SECONDS,
    )
    for r in read_worker_results(tmp_path, WORLD_SIZE, completed):
        check = r["check"]
        assert check["shape"] == [math.prod(GRID) // 4, HIDDEN], (r["rank"], check)
        # Default TPU matmul precision against an f32 CPU reference; rows from
        # a mispaired collective are off by the activations themselves.
        assert check["max_abs_err"] <= 0.05 * check["max_abs"], (r["rank"], check)


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
