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
"""The buffered pipeline hand-off over the runtime's point-to-point path.

`test_pp_buffered.py` drives it on CPU with a fake transport; this runs a
two-stage hand-off on two TPU devices with `torch.distributed` isend/irecv, the
way a pipeline does: serving steps that post their receives ahead, warmup
forwards that post their own, forwards of every bucket size, and the sum of
hidden states and residual arriving as the next stage's hidden states.

A launcher child prepares the two-device TPU environment and runs torchrun,
so that the pytest process keeps no memory of a two-device topology: the
runtime's launch-env helper reuses the first value it generated for a
variable in a process, which would otherwise hand later full-host tests two
devices.
"""

import argparse
import importlib.util
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
# Stays below the multichip lane's pytest timeout, so _launch stops the group.
LAUNCH_TIMEOUT_SECONDS = 480


SKIP_EXIT_CODE = 75


def _launch_command(result_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--launch",
        "--result-dir",
        str(result_dir),
    ]


def _torchrun_args(result_dir: Path) -> list[str]:
    return [
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--result-dir",
        str(result_dir),
    ]


def _require_torchtpu() -> None:
    if importlib.util.find_spec("torch_tpu") is None:
        pytest.skip("TorchTPU is unavailable")


def _launch_torchrun(result_dir: Path) -> None:
    """Runs in the launcher child: a fresh two-device environment, then torchrun."""
    for key in (
        "TORCH_TPU_XPROF_SESSION_ID",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
    ):
        os.environ.pop(key, None)
    try:
        from tests.distributed.tpu_test_utils import prepare_tpu_environment

        prepare_tpu_environment(world_size=WORLD_SIZE)
    except (ImportError, ValueError) as exc:
        print(f"Two TPU devices are unavailable: {exc}", flush=True)
        sys.exit(SKIP_EXIT_CODE)
    # The workers run on the default hand-off.
    os.environ.pop("TPU_PP_HANDOFF", None)
    os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")
    from torch.distributed.run import main as torchrun_main

    torchrun_main(_torchrun_args(result_dir))


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
        # The hand-off reads the pipeline group; one group over both ranks is
        # the two-stage pipeline, without a model config to initialize.
        world_group = get_world_group()
        parallel_state._PP = init_model_parallel_group(
            [list(range(world))],
            world_group.local_rank,
            dist.get_backend(world_group.device_group),
            group_name="pp_buffered_test",
        )
        from vllm_torchtpu.distributed import pp_buffered
        from vllm_torchtpu.distributed.pp_push import buffered_handoff

        # Built directly below; the default must select it.
        result["default_is_buffered"] = buffered_handoff()
        template = {
            "hidden_states": ((HIDDEN,), torch.bfloat16),
            "residual": ((HIDDEN,), torch.bfloat16),
        }
        handoff = pp_buffered.PPBuffered(device, MAX_ROWS, template, depth=2)
        handoff.warmup()
        seen = []
        for g, (rows, ahead) in enumerate(_forwards()):
            if ahead:
                handoff.begin_step([rows])
            received = handoff.start_forward(rows)
            if handoff.rank == 0:
                assert received is None
                hidden = torch.full((rows, HIDDEN), float(g + 1), dtype=torch.bfloat16)
                residual = torch.full((rows, HIDDEN), 0.5, dtype=torch.bfloat16)
                handoff.end_forward(
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
                        "hidden": pp_buffered._first_and_last(hidden),
                        "residual": pp_buffered._first_and_last(residual),
                        "shape": tuple(hidden.shape),
                    }
                )
                handoff.end_forward(None, rows)
        handoff.close()
        result["forwards"] = len(_forwards())
        result["posted_through"] = handoff.posted_through
        result["landed_through"] = handoff.landed_through
        result["sent"] = handoff.sent
        result["seen"] = seen
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result_path.write_text(json.dumps(result), encoding="utf-8")
        if initialized:
            dist.destroy_process_group()


# --------------------------------------------------------------------- parent
def _launch(result_dir: Path) -> subprocess.CompletedProcess[str]:
    _require_torchtpu()
    process = subprocess.Popen(
        _launch_command(result_dir),
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(),
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
        output = exc.output or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
    finally:
        # A timeout, or an error such as pytest's own timeout firing inside
        # communicate, leaves the launcher and its torchrun workers holding
        # the devices: stop the whole group.
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
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


def assert_handoff_results(results: list[dict]) -> None:
    errors = [r for r in results if "error" in r]
    assert not errors, f"worker failures: {errors}"
    first, last = results
    assert first["default_is_buffered"] and last["default_is_buffered"]
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
    result_dir = tmp_path / "pp_buffered"
    result_dir.mkdir()
    completed = _launch(result_dir)
    if completed.returncode == SKIP_EXIT_CODE:
        pytest.skip(completed.stdout.strip().splitlines()[-1])
    results = _results(result_dir)
    assert completed.returncode == 0 or any("error" in r for r in results), (
        completed.stdout[-12000:]
    )
    assert_handoff_results(results)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.launch == args.worker:
        raise ValueError("expected exactly one of --launch and --worker")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.launch:
        _launch_torchrun(args.result_dir)
    else:
        _run_worker(args.result_dir)


if __name__ == "__main__":
    _main()
