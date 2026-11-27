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
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from .tpu_test_utils import prepare_tpu_environment, torchrun_worker_env


def test_prepare_tpu_environment(monkeypatch):
    """Verify prepare_tpu_environment dispatches to
    torch.tpu.distributed.set_tpu_launch_env.
    """
    mock_set_tpu_launch_env = MagicMock()
    mock_dist = SimpleNamespace(set_tpu_launch_env=mock_set_tpu_launch_env)
    mock_tpu = SimpleNamespace(distributed=mock_dist)

    monkeypatch.setattr(torch, "tpu", mock_tpu, raising=False)

    prepare_tpu_environment(world_size=8)
    mock_set_tpu_launch_env.assert_called_once_with(nproc_per_node=8)


def test_torchrun_worker_env_skips_without_torch_tpu(monkeypatch):
    """A missing TorchTPU skips the test instead of failing it."""
    # A None entry makes `import torch_tpu` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch_tpu", None)

    with pytest.raises(pytest.skip.Exception, match="TorchTPU is unavailable"):
        torchrun_worker_env(world_size=8)


# A torchrun worker, run with `--worker --result-dir DIR`, that needs no TPU.
# In the first launch it fails as FAKE_MODE says and hangs, as on the CI hosts:
#   session: rank 0 reports that its TPU runtime could not start a session;
#   stall:   no rank finishes TPU initialization;
#   check:   both ranks start, then rank 0 reports another error.
# In a later launch, which a new torchrun agent (another parent pid) runs, both
# ranks start and succeed.
_FAKE_WORKER = """
import json, os, sys, time
from pathlib import Path

result_dir = Path(sys.argv[sys.argv.index("--result-dir") + 1])
rank = int(os.environ["RANK"])
mode = os.environ["FAKE_MODE"]
marker = result_dir.parent / "first_launch"
if rank == 0 and not marker.exists():
    marker.write_text(str(os.getppid()))
while not marker.exists():
    time.sleep(0.05)
if marker.read_text() == str(os.getppid()):
    if mode == "check":
        (result_dir / f"rank_{rank}.started").touch()
    if rank == 0 and mode != "stall":
        error = (
            "RuntimeError('TPU initialization failed: GRPC_ERROR error message: "
            "START_SESSION failed. Max # of tries(=3) exhausted')"
            if mode == "session"
            else "RuntimeError('boom')"
        )
        (result_dir / "rank_0.json").write_text(json.dumps({"rank": 0, "error": error}))
    time.sleep(600)
(result_dir / f"rank_{rank}.started").touch()
(result_dir / f"rank_{rank}.json").write_text(json.dumps({"rank": rank, "check": {}}))
"""


def _launch_fake_workers(tmp_path, monkeypatch, mode, timeout):
    import os
    import time

    from . import tpu_test_utils

    monkeypatch.setattr(tpu_test_utils, "_STARTUP_TIMEOUT_S", 20.0)
    monkeypatch.setattr(tpu_test_utils, "_STARTUP_RETRY_PAUSE_S", 0.1)
    monkeypatch.setattr(tpu_test_utils, "_STARTUP_RETRY_MIN_S", 1.0)
    script = tmp_path / "worker.py"
    script.write_text(_FAKE_WORKER)
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    env = {**os.environ, "FAKE_MODE": mode}
    start = time.monotonic()
    completed = tpu_test_utils.launch_torchrun_workers(
        script, result_dir, world_size=2, env=env, cwd=tmp_path, timeout=timeout
    )
    return completed, result_dir, time.monotonic() - start


@pytest.mark.parametrize(
    "mode, reason",
    [
        ("session", "a rank reported START_SESSION failed"),
        ("stall", "0 of 2 ranks initialized TPU"),
    ],
)
def test_a_launch_stalled_in_tpu_startup_is_retried(
    tmp_path, monkeypatch, mode, reason
):
    """A launch whose rank cannot start a TPU session, or whose ranks do not
    all finish TPU initialization in time, is ended without waiting out the
    timeout and run again; the retry's results are the ones read."""
    from .tpu_test_utils import read_worker_results

    completed, result_dir, elapsed = _launch_fake_workers(
        tmp_path, monkeypatch, mode, timeout=120
    )
    assert elapsed < 90, completed.stdout
    assert f"TPU start-up failed ({reason}" in completed.stdout
    results = read_worker_results(result_dir, 2, completed)
    assert [r["rank"] for r in results] == [0, 1]


def test_a_launch_failing_after_startup_is_not_retried(tmp_path, monkeypatch):
    """A failure after every rank has started is not retried: the launch runs
    until its timeout and the test fails on that error."""
    from .tpu_test_utils import read_worker_results

    completed, result_dir, elapsed = _launch_fake_workers(
        tmp_path, monkeypatch, "check", timeout=30
    )
    assert 30 <= elapsed < 90
    assert "TPU start-up failed" not in completed.stdout
    with pytest.raises(AssertionError, match="boom"):
        read_worker_results(result_dir, 2, completed)
