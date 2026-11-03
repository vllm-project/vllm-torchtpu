# SPDX-License-Identifier: Apache-2.0
"""Helpers for tests that must not retain TPU runtime state in pytest."""

import contextlib
import multiprocessing
import os
import queue
import signal
import subprocess
import time
import traceback
from collections.abc import Callable, Mapping
from typing import Any

import pytest

TPU_LAUNCH_ENV_KEYS: tuple[str, ...] = (
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_TOPOLOGY",
    "TORCH_TPU_XPROF_SESSION_ID",
    "MASTER_ADDR",
    "MASTER_PORT",
    "JOB_COMPLETION_INDEX",
    "TPU_VISIBLE_CHIPS",
    "TPU_VISIBLE_DEVICES",
)

TPU_TRANSIENT_SESSION_ERROR_MARKERS: tuple[str, ...] = (
    "START_SESSION failed",
    "SLICE_FAILURE_SW_INJECT_ERROR",
    "failed to retrieve the PjRt client: TPU initialization failed",
    "TPU_WORKER_BOOTSTRAP_TIMEOUT:",
)


def has_tpu_vfio_devices(
    *,
    vfio_dir: str = "/dev/vfio",
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return True iff numeric TPU IOMMU groups exist and CPU mode is not forced.

    Linux CPU nodes frequently expose ``/dev/vfio/vfio`` without numeric IOMMU
    group devices (``/dev/vfio/0``, ...), so ``os.path.exists('/dev/vfio')``
    false-positives on CPU containers.
    """
    env = os.environ if environ is None else environ
    if env.get("JAX_PLATFORMS", "").strip().lower() == "cpu":
        return False
    if env.get("SHAPE", "").strip().lower() == "cpu":
        return False
    try:
        entries = os.listdir(vfio_dir)
    except OSError:
        return False
    return any(entry.isdigit() for entry in entries)


def clear_tpu_launch_env(environ: Any | None = None) -> None:
    """Remove inherited single-slice launch variables before bootstrapping workers."""
    target = os.environ if environ is None else environ
    for key in TPU_LAUNCH_ENV_KEYS:
        target.pop(key, None)


def is_transient_tpu_session_error(log_text: str) -> bool:
    """Return True if worker logs show a transient GCE/libtpu session-init race."""
    for line in log_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(marker in stripped for marker in TPU_TRANSIENT_SESSION_ERROR_MARKERS):
            return True
    return False


def find_vfio_holder_pids(
    *,
    proc_root: str = "/proc",
    vfio_prefix: str = "/dev/vfio",
    ignore_pid: int | None = None,
) -> list[int]:
    """Return sorted PIDs holding open file descriptors under ``vfio_prefix``."""
    self_pid = os.getpid() if ignore_pid is None else ignore_pid
    holders: list[int] = []
    norm_prefix = vfio_prefix.rstrip("/")
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return holders

    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == self_pid:
            continue
        fd_dir = os.path.join(proc_root, entry, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd_name in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd_name))
            except OSError:
                continue
            if target == norm_prefix or target.startswith(norm_prefix + "/"):
                holders.append(pid)
                break
    holders.sort()
    return holders


def wait_for_vfio_idle(
    *,
    proc_root: str = "/proc",
    vfio_prefix: str = "/dev/vfio",
    timeout_s: float = 30.0,
    settle_s: float = 3.0,
    poll_s: float = 0.25,
) -> list[int]:
    """Wait until no other process holds ``/dev/vfio/*``, then sleep ``settle_s``."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    holders = find_vfio_holder_pids(proc_root=proc_root, vfio_prefix=vfio_prefix)
    while holders and time.monotonic() < deadline:
        time.sleep(min(poll_s, max(0.01, deadline - time.monotonic())))
        holders = find_vfio_holder_pids(proc_root=proc_root, vfio_prefix=vfio_prefix)
    if settle_s > 0 and has_tpu_vfio_devices(vfio_dir=vfio_prefix):
        time.sleep(settle_s)
    return holders


def kill_process_group(
    child: subprocess.Popen[Any],
    *,
    sigterm_timeout_s: float = 30.0,
) -> None:
    """Terminate and unconditionally reap an entire ``start_new_session`` group."""
    if child.poll() is None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=sigterm_timeout_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(child.pid, signal.SIGKILL)


def run_tpu_worker_session[T](
    run_attempt: Callable[[int], tuple[int, str, T]],
    *,
    max_attempts: int = 3,
    backoff_s: float = 12.0,
    settle_s: float = 10.0,
) -> tuple[int, str, T]:
    """Run a whole-slice worker launcher with VFIO teardown waits and session retry."""
    if max_attempts < 1:
        raise ValueError(
            f"max_attempts must be >= 1, got {max_attempts}; pass e.g. max_attempts=3."
        )
    last_result: tuple[int, str, T] | None = None
    for attempt in range(max_attempts):
        wait_for_vfio_idle(settle_s=0.0)
        try:
            rc, log_text, payload = run_attempt(attempt)
        finally:
            wait_for_vfio_idle(settle_s=settle_s)
        last_result = (rc, log_text, payload)
        if rc == 0:
            return last_result
        if attempt + 1 < max_attempts and is_transient_tpu_session_error(log_text):
            if backoff_s > 0:
                time.sleep(backoff_s * (attempt + 1))
            continue
        return last_result
    assert last_result is not None
    return last_result


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
    """Prepare single-host TPU environment variables across torch_tpu versions.

    Backwards-compatible bridge supporting:
      - torch_tpu >= 0.1.2: torch.tpu.distributed.set_tpu_launch_env (public API)
      - torch_tpu <= 0.1.1:
        torch_tpu._internal.distributed.launchers.singlehost_wrapper.prepare_tpu_environment
    """
    import torch

    tpu = getattr(torch, "tpu", None)
    tpu_dist = getattr(tpu, "distributed", None) if tpu is not None else None
    if tpu_dist is not None and hasattr(tpu_dist, "set_tpu_launch_env"):
        tpu_dist.set_tpu_launch_env(nproc_per_node=world_size)
        return

    # TODO: Remove legacy singlehost_wrapper fallback once torch_tpu >= 0.1.2
    # is baseline.
    from torch_tpu._internal.distributed.launchers.singlehost_wrapper import (
        prepare_tpu_environment as _legacy_prepare,
    )

    _legacy_prepare(world_size=world_size)
