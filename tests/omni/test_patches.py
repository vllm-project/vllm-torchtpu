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
import subprocess
import sys
import textwrap
import types

import pytest
import torch

from vllm_torchtpu.omni import patches

pytestmark = pytest.mark.omni

_DIFFUSION_WORKER = "vllm_omni.diffusion.worker.diffusion_worker"


class _FakeWorkerProc:
    """Builds its stream device from self.gpu_id like the upstream loop."""

    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.queue = "queue"

    def _async_output_loop(self):
        self.stream_device = torch.device("cpu", self.gpu_id)
        self.seen_queue = self.queue


def _install_worker_proc(monkeypatch, cls):
    module = types.ModuleType(_DIFFUSION_WORKER)
    module.WorkerProc = cls
    monkeypatch.setitem(sys.modules, _DIFFUSION_WORKER, module)


@pytest.fixture
def worker_proc(monkeypatch):
    # A fresh subclass per test, so the patch never edits a shared class.
    cls = type("WorkerProc", (_FakeWorkerProc,), {})
    _install_worker_proc(monkeypatch, cls)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 1)
    return cls


def test_d2h_stream_uses_current_device_index(worker_proc):
    patches._patch_diffusion_worker_d2h_stream()

    worker = worker_proc(gpu_id=3)
    worker._async_output_loop()

    assert worker.stream_device == torch.device("cpu", 1)
    assert worker.seen_queue == "queue"
    assert worker.gpu_id == 3


def test_d2h_stream_patch_is_idempotent(worker_proc):
    orig_loop = worker_proc._async_output_loop

    patches._patch_diffusion_worker_d2h_stream()
    patched_loop = worker_proc._async_output_loop
    patches._patch_diffusion_worker_d2h_stream()

    assert patched_loop is not orig_loop
    assert worker_proc._async_output_loop is patched_loop
    assert worker_proc._tpu_orig_async_output_loop is orig_loop


def test_d2h_stream_patch_skips_missing_loop(monkeypatch):
    cls = type("WorkerProc", (), {})
    _install_worker_proc(monkeypatch, cls)
    warnings = []
    monkeypatch.setattr(patches.logger, "warning", warnings.append)

    patches._patch_diffusion_worker_d2h_stream()

    assert len(warnings) == 1
    assert not hasattr(cls, "_async_output_loop")
    assert not hasattr(cls, "_tpu_orig_async_output_loop")


def test_d2h_stream_patch_targets_real_worker_proc():
    # Runs in a fresh interpreter because tests/omni mock vllm_omni in
    # sys.modules. Skipped when no importable vLLM-Omni is on the path.
    code = textwrap.dedent(
        """
        import sys

        try:
            from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc
        except ImportError as e:
            print(e)
            sys.exit(77)

        from vllm_torchtpu.omni.patches import _patch_diffusion_worker_d2h_stream

        orig_loop = WorkerProc._async_output_loop
        _patch_diffusion_worker_d2h_stream()
        assert WorkerProc._tpu_orig_async_output_loop is orig_loop
        assert WorkerProc._async_output_loop is not orig_loop
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode == 77:
        pytest.skip(f"vllm_omni is not importable: {result.stdout.strip()}")
    assert result.returncode == 0, result.stdout + result.stderr
