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
# Rank 0 of the first launch reports that its TPU runtime could not start a
# session (or, with FAKE_ERROR set, another error) and then hangs, as on the CI
# hosts; rank 1 hangs with it. In a later launch, which a new torchrun agent
# (another parent pid) runs, both ranks succeed.
_FAKE_WORKER = """
import json, os, sys, time
from pathlib import Path

result_dir = Path(sys.argv[sys.argv.index("--result-dir") + 1])
rank = int(os.environ["RANK"])
marker = result_dir.parent / "first_launch"
if rank == 0 and not marker.exists():
    marker.write_text(str(os.getppid()))
while not marker.exists():
    time.sleep(0.05)
if marker.read_text() == str(os.getppid()):
    if rank == 0:
        error = os.environ.get("FAKE_ERROR") or (
            "RuntimeError('TPU initialization failed: GRPC_ERROR error message: "
            "START_SESSION failed. Max # of tries(=3) exhausted')"
        )
        (result_dir / "rank_0.json").write_text(json.dumps({"rank": 0, "error": error}))
    time.sleep(600)
(result_dir / f"rank_{rank}.json").write_text(json.dumps({"rank": rank, "check": {}}))
"""


def _launch_fake_workers(tmp_path, monkeypatch, timeout, error=None):
    import os
    import time

    from . import tpu_test_utils

    monkeypatch.setattr(tpu_test_utils, "_SESSION_RETRY_PAUSE_S", 0.1)
    monkeypatch.setattr(tpu_test_utils, "_SESSION_RETRY_MIN_S", 1.0)
    script = tmp_path / "worker.py"
    script.write_text(_FAKE_WORKER)
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    env = dict(os.environ)
    if error is not None:
        env["FAKE_ERROR"] = error
    start = time.monotonic()
    completed = tpu_test_utils.launch_torchrun_workers(
        script, result_dir, world_size=2, env=env, cwd=tmp_path, timeout=timeout
    )
    return completed, result_dir, time.monotonic() - start


def test_a_launch_whose_tpu_session_fails_is_retried(tmp_path, monkeypatch):
    """A rank that cannot start a TPU session ends the launch at once, without
    waiting out the timeout for the ranks it holds up, and the launch is run
    again; the retry's results are the ones read."""
    from .tpu_test_utils import read_worker_results

    completed, result_dir, elapsed = _launch_fake_workers(
        tmp_path, monkeypatch, timeout=120
    )
    assert elapsed < 60, completed.stdout
    assert "could not start a TPU session" in completed.stdout
    results = read_worker_results(result_dir, 2, completed)
    assert [r["rank"] for r in results] == [0, 1]


def test_a_launch_failing_otherwise_is_not_retried(tmp_path, monkeypatch):
    """Any other error is not retried: the launch runs until its timeout and
    the test fails on that error."""
    from .tpu_test_utils import read_worker_results

    completed, result_dir, elapsed = _launch_fake_workers(
        tmp_path, monkeypatch, timeout=8, error="RuntimeError('boom')"
    )
    assert 8 <= elapsed < 60
    assert "could not start a TPU session" not in completed.stdout
    with pytest.raises(AssertionError, match="boom"):
        read_worker_results(result_dir, 2, completed)
