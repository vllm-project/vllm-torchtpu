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
`group.reduce_scatter(x, dim=0)` gives it, eager and under `torch.compile`.
That is the rank-to-mesh-index mapping, which only real ranks exercise.

Run on one host (4 chips x 2 cores = 8 ranks).
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
# (rows per rank, hidden): the padded micro-batched path (64, 160), the
# micro-batched kernel (1024), the row-streaming kernel at the default
# threshold (2048 x 4096 = 16 MiB), and a hidden too wide for its VMEM, which
# must fall back to the micro-batched kernel (2048 x 7168).
SHAPES = ((64, 4096), (160, 4096), (1024, 4096), (2048, 4096), (2048, 7168))
CALLS_PER_SHAPE = 2  # serving calls the op once per MoE layer on new data
# Both sides round a sum of 8 bf16 values to bf16, in different orders. On the
# fp8 wire, calls of 16 MiB or more also round partial sums to e4m3 (~30 dB).
MAX_REL_ERR = {"bf16": 1e-2, "fp8": 6e-2}
FP8_STATIC_SCALE = "16"
# Under the per-PR multichip lane's pytest --timeout=600, so this limit, not
# pytest's, ends a hung launch and the finally below can reap the workers.
LAUNCH_TIMEOUT_SECONDS = 480


def _worker_command(result_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--result-dir",
        str(result_dir),
    ]


def _prepare_worker_env(wire: str) -> dict[str, str]:
    try:
        from tests.distributed.tpu_test_utils import prepare_tpu_environment
    except ImportError as exc:
        pytest.skip(f"TorchTPU is unavailable: {exc}")
    keys = (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    )
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        prepare_tpu_environment(world_size=WORLD_SIZE)
        env = os.environ.copy()
    except ValueError as exc:
        pytest.skip(f"Eight TPU devices are unavailable: {exc}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    env["TPU_MOE_EP_HIER_RS"] = wire
    if wire == "fp8":
        env["TPU_MOE_EP_HIER_RS_FP8_STATIC_SCALE"] = FP8_STATIC_SCALE
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    return env


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

    out: dict = {"armed": hrs.prebuild_hier_reduce_scatter()}
    if not out["armed"]:
        return out
    armed = hrs._ARMED
    assert armed is not None
    kernel_calls = [0]

    def counted(x):
        kernel_calls[0] += 1
        return armed.op(x)

    hrs._ARMED = hrs._Armed(op=counted, ranks=armed.ranks)

    def combine(x):
        return hrs.moe_combine_reduce_scatter(ep, x)

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
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                out[key] = {"error": repr(exc)}
    return out


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {"rank": rank, "world": world}
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import init_distributed_environment

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        result["check"] = _check(rank)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=2))
        if rank == 0:
            print("RESULT rank0", json.dumps(result), flush=True)
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                traceback.print_exc()
    if "error" in result:
        raise SystemExit(1)


def _launch(result_dir: Path, wire: str) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        _worker_command(result_dir),
        cwd=Path(__file__).resolve().parents[2],
        env=_prepare_worker_env(wire),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    output = ""
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        output = exc.output or ""
    finally:
        # Also on pytest's own timeout, which raises out of communicate(): the
        # workers run in their own session and would keep holding the chips.
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=60)
    return subprocess.CompletedProcess(process.args, process.returncode, output, "")


@pytest.mark.parametrize("wire", ["bf16", "fp8"])
def test_hier_reduce_scatter_on_real_collectives(tmp_path, wire):
    result_dir = tmp_path / "hier_rs"
    result_dir.mkdir()
    completed = _launch(result_dir, wire)
    results = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
        results.append(
            json.loads(path.read_text())
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    errors = [r for r in results if "error" in r]
    assert not errors, (errors, (completed.stdout or "")[-4000:])
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
        _run_worker(args.result_dir)
