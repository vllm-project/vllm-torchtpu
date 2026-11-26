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
"""TPU_MOE_EP_HIER_RS=bf16|fp8 over real collectives (EP = DP = 8, one tpu7x host).

`test_hier_reduce_scatter.py` pins arming and dispatch against stand-ins; this
arms the real op through `prebuild_hier_reduce_scatter()` and checks that
`moe_combine_reduce_scatter(group, x)` gives every rank the block
`group.reduce_scatter(x, dim=0)` gives it, eager and under `torch.compile`,
through a group whose own reduce-scatter raises, so the kernel must serve it.
That is the rank-to-mesh-index mapping, which only real ranks exercise.

Run on one host (4 chips x 2 cores = 8 ranks).
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

REPO_ROOT = Path(__file__).resolve().parents[2]

WORLD_SIZE = 8
# (rows per rank, hidden): the padded micro-batched path (64, 160), the
# micro-batched kernel (1024), the row-streaming kernel at the default
# threshold (3072 x 4096 = 24 MiB), and a hidden too wide for its VMEM, which
# must fall back to the micro-batched kernel (2048 x 7168 = 28 MiB).
SHAPES = ((64, 4096), (160, 4096), (1024, 4096), (3072, 4096), (2048, 7168))
CALLS_PER_SHAPE = 2  # serving calls the op once per MoE layer on new data
# Both sides round a sum of 8 bf16 values to bf16, in different orders. On the
# fp8 wire, calls of 16 MiB or more also round partial sums to e4m3 (~30 dB).
MAX_REL_ERR = {"bf16": 1e-2, "fp8": 6e-2}
FP8_STATIC_SCALE = "16"
# Under the per-PR multichip lane's pytest --timeout=600.
LAUNCH_TIMEOUT_SECONDS = 480


def _rel_err(got, want) -> float:
    return float((got - want).norm() / want.norm())


def _check(rank: int) -> dict:
    import torch
    from vllm.distributed import parallel_state

    from vllm_torchtpu.distributed import hier_reduce_scatter as hrs

    world = parallel_state.get_world_group()
    backend = torch.distributed.get_backend(world.device_group)
    ep = parallel_state.init_model_parallel_group(
        [list(range(WORLD_SIZE))], world.local_rank, backend, group_name="hier_rs_ep"
    )
    # The serving layout the op arms for: EP = DP = the whole world.
    parallel_state._EP = ep
    parallel_state._DP = ep

    start = time.monotonic()

    def progress(what: str) -> None:
        # Shown when the launch fails: how far each rank got, and how fast.
        print(f"rank {rank}: {what} at {time.monotonic() - start:.1f}s", flush=True)

    out: dict = {"armed": hrs.prebuild_hier_reduce_scatter()}
    progress("armed")
    if not out["armed"]:
        return out
    armed = hrs._ARMED
    assert armed is not None
    kernel_calls = [0]

    def counted(x):
        kernel_calls[0] += 1
        return armed.op(x)

    hrs._ARMED = hrs._Armed(op=counted, ranks=armed.ranks)

    class _NoFallback:
        """The EP group's ranks, with no reduce-scatter to fall back to: a
        call, eager or compiled, that does not take the kernel raises."""

        ranks = ep.ranks
        world_size = ep.world_size

        def reduce_scatter(self, x, dim):
            raise AssertionError("fell back to the group's reduce-scatter")

    no_fallback = _NoFallback()

    def combine(x):
        return hrs.moe_combine_reduce_scatter(no_fallback, x)

    for how in ("eager", "compiled"):
        for rows, hidden in SHAPES:
            # One graph per shape, as serving compiles one per bucket. Dynamo
            # counts recompiles per code object, so start each shape afresh.
            fn = combine
            if how == "compiled":
                torch._dynamo.reset()
                fn = torch.compile(
                    combine, backend="tpu", fullgraph=True, dynamic=False
                )
            key = f"{how}/{rows}x{hidden}"
            try:
                before = kernel_calls[0]
                errs = []
                for call in range(CALLS_PER_SHAPE):
                    # Every rank builds every rank's input, for the f32 reference.
                    gen = torch.Generator().manual_seed(
                        rows * 7919 + hidden * 31 + call
                    )
                    inputs = [
                        torch.randn(rows, hidden, generator=gen).to(torch.bfloat16)
                        for _ in range(WORLD_SIZE)
                    ]
                    block = rows // WORLD_SIZE
                    want = sum(x.float() for x in inputs)[
                        rank * block : (rank + 1) * block
                    ]
                    x = inputs[rank].to("tpu")
                    got = fn(x).cpu().float()
                    group_got = ep.reduce_scatter(x, dim=0).cpu().float()
                    errs.append(
                        {
                            "shape": list(got.shape),
                            "vs_f32": _rel_err(got, want),
                            "vs_group": _rel_err(got, group_got),
                        }
                    )
                out[key] = {"calls": errs, "kernel_calls": kernel_calls[0] - before}
                progress(key)
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                out[key] = {"error": repr(exc)}
    return out


@pytest.mark.parametrize("wire", ["bf16", "fp8"])
def test_hier_reduce_scatter_on_real_collectives(tmp_path, wire):
    result_dir = tmp_path / "hier_rs"
    result_dir.mkdir()
    from tests.distributed.tpu_test_utils import (
        launch_torchrun_workers,
        read_worker_results,
        torchrun_worker_env,
    )

    env = torchrun_worker_env(WORLD_SIZE)
    env["TPU_MOE_EP_HIER_RS"] = wire
    if wire == "fp8":
        env["TPU_MOE_EP_HIER_RS_FP8_STATIC_SCALE"] = FP8_STATIC_SCALE
    completed = launch_torchrun_workers(
        Path(__file__).resolve(),
        result_dir,
        world_size=WORLD_SIZE,
        env=env,
        cwd=REPO_ROOT,
        timeout=LAUNCH_TIMEOUT_SECONDS,
    )
    # Rank 0's timings, also on success: how close the launch came to timing out.
    print(
        "".join(
            f"\n  {line}"
            for line in completed.stdout.splitlines()
            if line.startswith("rank 0:")
        )
    )
    results = read_worker_results(result_dir, WORLD_SIZE, completed)
    for r in results:
        check = r["check"]
        assert check["armed"], (r["rank"], check)
        for how in ("eager", "compiled"):
            for rows, hidden in SHAPES:
                case = check[f"{how}/{rows}x{hidden}"]
                assert "error" not in case, (r["rank"], how, rows, hidden, case)
                if how == "eager":
                    assert case["kernel_calls"] == CALLS_PER_SHAPE, (r["rank"], case)
                for call in case["calls"]:
                    assert call["shape"] == [rows // WORLD_SIZE, hidden], case
                    assert call["vs_f32"] < MAX_REL_ERR[wire], (r["rank"], how, case)
                    assert call["vs_group"] < MAX_REL_ERR[wire], (r["rank"], how, case)


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
