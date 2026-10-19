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
