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
"""Helpers for tests that must not retain TPU runtime state in pytest."""

import multiprocessing
import queue
import traceback
from collections.abc import Callable
from typing import Any

import pytest


def _isolated_test_worker(target: Callable[[], None], result_queue: Any) -> None:
    try:
        target()
    except pytest.skip.Exception as exc:
        result_queue.put(("SKIP", str(exc)))
    except BaseException:
        result_queue.put(("ERROR", traceback.format_exc()))
    else:
        result_queue.put(("OK", None))


def run_in_isolated_process(
    target: Callable[[], None], *, timeout: float = 600
) -> None:
    """Run a TPU test in a spawned process and propagate its result.

    PJRT keeps TPU device handles alive for the lifetime of its process. Running
    device tests in a child prevents those handles from blocking later tests
    that start their own engine processes.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=_isolated_test_worker, args=(target, result_queue))
    try:
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            raise AssertionError(f"Isolated TPU test timed out after {timeout} seconds")
        try:
            status, payload = result_queue.get(timeout=5)
        except queue.Empty as exc:
            raise AssertionError(
                "Isolated TPU test exited without reporting a result "
                f"(exitcode={process.exitcode})"
            ) from exc
    finally:
        result_queue.close()

    if status == "SKIP":
        pytest.skip(payload)
    if status == "ERROR":
        raise AssertionError(f"Isolated TPU test failed:\n{payload}")
    if process.exitcode != 0:
        raise AssertionError(f"Isolated TPU test crashed (exitcode={process.exitcode})")


def prepare_tpu_environment(world_size: int | None = None) -> None:
    """Prepare single-host TPU environment variables."""
    import torch
    import torch_tpu  # noqa: F401

    torch.tpu.distributed.set_tpu_launch_env(nproc_per_node=world_size)


# Multichip tests that launch one worker per device with torchrun. The test
# file runs as the worker script: `<file> --worker --result-dir DIR`.

_STACK_DUMP_ENV = "TPU_TEST_WORKER_STACK_DUMP_SECONDS"

_LAUNCH_ENV_KEYS = (
    "TORCH_TPU_XPROF_SESSION_ID",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_TOPOLOGY",
)


def torchrun_worker_env(world_size: int) -> dict[str, str]:
    """The environment for `world_size` single-host TPU workers; skips the test
    when TorchTPU or the devices are unavailable."""
    import os

    saved = {k: os.environ.get(k) for k in _LAUNCH_ENV_KEYS}
    try:
        for k in _LAUNCH_ENV_KEYS:
            os.environ.pop(k, None)
        prepare_tpu_environment(world_size=world_size)
        env = os.environ.copy()
    except ImportError as exc:
        pytest.skip(f"TorchTPU is unavailable: {exc}")
    except ValueError as exc:
        pytest.skip(f"{world_size} TPU devices are unavailable: {exc}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    env.setdefault("TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS", "false")
    env.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "0")
    env.setdefault("VLLM_USE_AOT_COMPILE", "0")
    return env


# On the CI hosts, a launch that starts right after another one ended sometimes
# stalls in TPU initialization: a rank's TPU runtime cannot start a session on
# the host (it reports _SESSION_FAILURE after its retries, which can take most
# of the launch timeout) and the other ranks wait for it. Every rank marks the
# end of its TPU initialization (rank_<rank>.started); a launch in which a rank
# reports _SESSION_FAILURE, or not every rank has started within
# _STARTUP_TIMEOUT_S, is ended and run again after a pause.
_SESSION_FAILURE = "START_SESSION failed"
_STARTUP_TIMEOUT_S = 120.0
# The workers of the ended launch release the TPUs asynchronously; the same
# wait test_latent_proj_intra_chip_multichip.py uses for that race.
_STARTUP_RETRY_PAUSE_S = 60.0
# Time a relaunch needs, start-up included, for a retry to be worth it.
_STARTUP_RETRY_MIN_S = 120.0
# How long the workers of an ended launch get to exit after SIGTERM, and then
# after SIGKILL.
_REAP_S = 30.0
_POLL_S = 2.0


def launch_torchrun_workers(
    script: Any,
    result_dir: Any,
    *,
    world_size: int,
    env: dict[str, str],
    cwd: Any,
    timeout: float,
) -> Any:
    """Runs `script --worker` on `world_size` ranks; returns the completed
    process. Keep `timeout` under pytest's, so that this, not pytest, ends a
    hung launch and the workers are reaped.

    A launch that stalls in TPU initialization (see _STARTUP_TIMEOUT_S) is
    ended and, while enough of `timeout` is left, run again after a pause;
    each launch's output is kept, in order."""
    import time

    deadline = time.monotonic() + timeout
    outputs = []
    while True:
        completed, stalled = _launch_torchrun_once(
            script, result_dir, world_size, env, cwd, deadline
        )
        outputs.append(completed.stdout)
        remaining = deadline - time.monotonic()
        if stalled is None or remaining < _STARTUP_RETRY_PAUSE_S + _STARTUP_RETRY_MIN_S:
            break
        outputs.append(
            f"\n=== TPU start-up failed ({stalled}); "
            f"relaunching in {_STARTUP_RETRY_PAUSE_S:.0f}s ===\n"
        )
        print(outputs[-1], flush=True)
        for stale in [
            *result_dir.glob("rank_*.json"),
            *result_dir.glob("rank_*.started"),
        ]:
            stale.unlink()
        time.sleep(_STARTUP_RETRY_PAUSE_S)
    completed.stdout = "".join(outputs)
    return completed


def _startup_stall(result_dir: Any, world_size: int, elapsed: float) -> str | None:
    """Why the launch is stuck in TPU initialization, or None if it is not."""
    for path in result_dir.glob("rank_*.json"):
        try:
            if _SESSION_FAILURE in path.read_text():
                return f"a rank reported {_SESSION_FAILURE}"
        except OSError:
            pass
    started = len(list(result_dir.glob("rank_*.started")))
    if elapsed >= _STARTUP_TIMEOUT_S and started < world_size:
        return (
            f"{started} of {world_size} ranks initialized TPU in "
            f"{_STARTUP_TIMEOUT_S:.0f}s"
        )
    return None


def _group_alive(pgid: int) -> bool:
    """Whether a process of process group `pgid` still exists."""
    import os

    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _launch_torchrun_once(
    script: Any,
    result_dir: Any,
    world_size: int,
    env: dict[str, str],
    cwd: Any,
    deadline: float,
) -> tuple[Any, bool]:
    """One torchrun launch, ended at `deadline` or as soon as it stalls in TPU
    initialization; returns the completed process and why it stalled, or None."""
    import contextlib
    import os
    import signal
    import subprocess
    import sys
    import time

    # Workers dump their Python stacks shortly before the launch times out, so
    # a hang shows where each rank is stuck (see run_torchrun_worker).
    remaining = deadline - time.monotonic()
    env = {**env, _STACK_DUMP_ENV: str(max(remaining - 45, 1))}
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={world_size}",
            str(script),
            "--worker",
            "--result-dir",
            str(result_dir),
        ],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    output = ""
    stalled = None
    start = time.monotonic()
    try:
        while True:
            try:
                # communicate() can be called again after a timeout; it keeps
                # what it has read so far.
                output, _ = process.communicate(
                    timeout=max(0.0, min(_POLL_S, deadline - time.monotonic()))
                )
                break
            except subprocess.TimeoutExpired as exc:
                stalled = _startup_stall(
                    result_dir, world_size, time.monotonic() - start
                )
                if time.monotonic() < deadline and stalled is None:
                    continue
                # Bytes even in text mode: the partial output is not decoded.
                output = exc.output or ""
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                break
    finally:
        # Also on pytest's own timeout, which raises out of communicate(): the
        # workers run in their own session and would keep holding the chips.
        # Every worker, not just torchrun: a worker stuck in TPU
        # initialization keeps holding its chip until it is gone.
        if process.poll() is None or _group_alive(process.pid):
            for sig in (signal.SIGTERM, signal.SIGKILL):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, sig)
                reap_deadline = time.monotonic() + _REAP_S
                while time.monotonic() < reap_deadline:
                    process.poll()
                    if process.returncode is not None and not _group_alive(process.pid):
                        break
                    time.sleep(0.5)
                else:
                    continue
                break
    completed = subprocess.CompletedProcess(
        process.args, process.returncode, output, ""
    )
    return completed, stalled


def read_worker_results(result_dir: Any, world_size: int, completed: Any) -> list:
    """Every rank's result; fails the test if a rank reported an error."""
    import json

    results = []
    for rank in range(world_size):
        path = result_dir / f"rank_{rank}.json"
        results.append(
            json.loads(path.read_text())
            if path.exists()
            else {"rank": rank, "error": "missing worker result"}
        )
    errors = [r for r in results if "error" in r]
    assert not errors, (errors, _host_cpus(), (completed.stdout or "")[-24000:])
    return results


def _host_cpus() -> str:
    """The CPUs the workers could use: a starved host compiles slowly."""
    import os
    from pathlib import Path

    quota = Path("/sys/fs/cgroup/cpu.max")
    return (
        f"cpu_count={os.cpu_count()} affinity={len(os.sched_getaffinity(0))} "
        f"cgroup cpu.max={quota.read_text().strip() if quota.exists() else None}"
    )


def run_torchrun_worker(result_dir: Any, check: Callable[[int], dict]) -> None:
    """Worker side: joins torch.distributed and vLLM's world, runs
    `check(rank)` and writes the rank's result for read_worker_results."""
    import faulthandler
    import json
    import os

    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if os.environ.get(_STACK_DUMP_ENV):
        faulthandler.dump_traceback_later(float(os.environ[_STACK_DUMP_ENV]))
    result: dict[str, object] = {"rank": rank, "world": world}
    initialized = False
    try:
        dist.init_process_group(backend="tpu_dist")
        initialized = True
        torch.empty((1,), device="tpu").cpu()
        # TPU initialization is done (see _startup_stall).
        (result_dir / f"rank_{rank}.started").touch()
        from vllm.distributed.parallel_state import init_distributed_environment

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        result["check"] = check(rank)
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
