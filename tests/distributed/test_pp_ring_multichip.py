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
"""The pipeline ring's hand-off over the runtime's point-to-point path.

`test_pp_ring.py` drives the ring on CPU with a fake transport; this runs a
two-stage ring on two TPU devices with `torch.distributed` isend/irecv, the
way a pipeline does: serving steps that post their receives ahead, warmup
forwards that post their own, forwards of every bucket size, and the sum of
hidden states and residual arriving as the next stage's hidden states.
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

WORLD_SIZE = 2
HIDDEN = 256
MAX_ROWS = 64
BUCKETS = (8, 16, 32, 64)
LAUNCH_TIMEOUT_SECONDS = 900


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


def _prepare_worker_env() -> dict[str, str]:
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
        pytest.skip(f"Two TPU devices are unavailable: {exc}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    env["TPU_PP_HANDOFF"] = "ring"
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    return env


# --------------------------------------------------------------------- worker
def _forwards() -> list[tuple[int, bool]]:
    """(rows, posted ahead) of every forward: each bucket twice as a serving
    step and once as a warmup forward, then nine warmup forwards in a row,
    more than the receives a device may have posted."""
    plan = []
    for rows in BUCKETS:
        plan += [(rows, True), (rows, True), (rows, False)]
    plan += [(8, False)] * 9
    return plan


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result_path = result_dir / f"rank_{rank}.json"
    result: dict[str, object] = {"rank": rank, "world": world}
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        device = torch.device("tpu")
        torch.empty((1,), device=device).cpu()
        from vllm.distributed import parallel_state
        from vllm.distributed.parallel_state import (
            get_world_group,
            init_distributed_environment,
            init_model_parallel_group,
        )

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        # The ring reads the pipeline group; one group over both ranks is
        # the two-stage pipeline, without a model config to initialize.
        world_group = get_world_group()
        parallel_state._PP = init_model_parallel_group(
            [list(range(world))],
            world_group.local_rank,
            dist.get_backend(world_group.device_group),
            group_name="pp_ring_test",
        )
        from vllm_torchtpu.distributed import pp_ring

        template = {
            "hidden_states": ((HIDDEN,), torch.bfloat16),
            "residual": ((HIDDEN,), torch.bfloat16),
        }
        ring = pp_ring.PPRing(device, MAX_ROWS, template, depth=2)
        ring.warmup()
        seen = []
        for g, (rows, ahead) in enumerate(_forwards()):
            if ahead:
                ring.begin_step([rows])
            received = ring.start_forward(rows)
            if ring.rank == 0:
                assert received is None
                hidden = torch.full((rows, HIDDEN), float(g + 1), dtype=torch.bfloat16)
                residual = torch.full((rows, HIDDEN), 0.5, dtype=torch.bfloat16)
                ring.end_forward(
                    {
                        "hidden_states": hidden.to(device),
                        "residual": residual.to(device),
                    },
                    rows,
                )
            else:
                # Read on the host once the data has landed, as the
                # self-test does; a stage's forward would read it in place.
                hidden = received["hidden_states"]
                residual = received["residual"]
                seen.append(
                    {
                        "rows": rows,
                        "hidden": pp_ring._first_and_last(hidden),
                        "residual": pp_ring._first_and_last(residual),
                        "shape": tuple(hidden.shape),
                    }
                )
                ring.end_forward(None, rows)
        ring.close()
        result["forwards"] = len(_forwards())
        result["posted_through"] = ring.posted_through
        result["landed_through"] = ring.landed_through
        result["sent"] = ring.sent
        result["seen"] = seen
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result_path.write_text(json.dumps(result), encoding="utf-8")
        if initialized:
            dist.destroy_process_group()


# --------------------------------------------------------------------- parent
def _launch(result_dir: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        _worker_command(result_dir),
        cwd=Path(__file__).resolve().parents[2],
        env=_prepare_worker_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # libtpu logs can carry non-UTF-8 bytes; they must not fail the test.
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        output = exc.output or ""
        process.wait(timeout=60)
    return subprocess.CompletedProcess(process.args, process.returncode, output, "")


def _results(result_dir: Path) -> list[dict]:
    out = []
    for rank in range(WORLD_SIZE):
        path = result_dir / f"rank_{rank}.json"
        out.append(
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    return out


def assert_ring_results(results: list[dict]) -> None:
    errors = [r for r in results if "error" in r]
    assert not errors, f"worker failures: {errors}"
    first, last = results
    plan = _forwards()
    assert last["forwards"] == len(plan)
    assert first["sent"] == len(plan)
    # every forward's transfer, after the warm-up self-test's
    assert last["posted_through"] == last["landed_through"] == len(plan) + 1
    assert len(last["seen"]) == len(plan)
    for g, ((rows, _), seen) in enumerate(zip(plan, last["seen"])):
        # the sum arrives as the hidden states, over a zero residual
        assert seen["rows"] == rows and tuple(seen["shape"]) == (rows, HIDDEN)
        assert seen["hidden"] == [g + 1.5, g + 1.5], (g, seen)
        assert seen["residual"] == [0.0, 0.0], (g, seen)


def test_two_stages_hand_off_on_real_devices(tmp_path):
    result_dir = tmp_path / "pp_ring"
    result_dir.mkdir()
    completed = _launch(result_dir)
    results = _results(result_dir)
    assert completed.returncode == 0 or any("error" in r for r in results), (
        completed.stdout[-12000:]
    )
    assert_ring_results(results)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.worker:
        raise ValueError("expected --worker")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    _run_worker(args.result_dir)


if __name__ == "__main__":
    _main()
