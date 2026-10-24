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
"""Host tensors uploaded outside the runner are held by it, and vLLM's
upload helpers route through `host_uploads.upload` once patched."""

import gc
import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.multimodal import inputs as mm_inputs
from vllm.utils import torch_utils
from vllm.v1.sample.logits_processor import builtin as logits_builtin
from vllm.v1.worker import dp_utils
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_torchtpu.runner import host_uploads


@pytest.fixture
def isolated(monkeypatch):
    """Restore the module's registration after the test."""
    for name in ("_uploader", "_holder", "_device_type"):
        monkeypatch.setattr(host_uploads, name, getattr(host_uploads, name))


@pytest.fixture
def held(isolated):
    """Register a stand-in runner uploading to `meta` and record what it holds."""
    sources = []

    def uploader(host):
        sources.append(host)
        return host.to("meta")

    host_uploads.set_uploader(uploader, torch.device("meta"), sources.append)
    return sources


def test_upload_goes_through_the_registered_runner(held):
    host = torch.tensor([1, 2, 3])
    dev = host_uploads.upload(host, "meta")
    assert dev.device.type == "meta"
    assert [id(t) for t in held] == [id(host)]


def test_upload_copies_synchronously_without_a_runner(isolated):
    host_uploads.set_uploader(None, None)
    assert host_uploads.upload(torch.tensor([1]), "meta").device.type == "meta"
    host_uploads.hold(torch.tensor([1]))  # nothing registered: a no-op


def test_a_released_runner_is_not_kept_alive(isolated):
    """The registration holds a runner's methods weakly; once the runner is
    gone, uploads fall back to synchronous copies and holds do nothing."""

    class Runner:
        def __init__(self):
            self.held = []

        def upload(self, host):
            self.held.append(host)
            return host.to("meta")

        def hold(self, host):
            self.held.append(host)

    runner = Runner()
    host_uploads.set_uploader(runner.upload, torch.device("meta"), runner.hold)
    sink, alive = runner.held, weakref.ref(runner)
    host_uploads.upload(torch.tensor([1]), "meta")
    assert len(sink) == 1
    del runner
    gc.collect()
    assert alive() is None
    assert host_uploads.upload(torch.tensor([2]), "meta").device.type == "meta"
    host_uploads.hold(torch.tensor([3]))
    assert len(sink) == 1


def test_hold_reaches_the_registered_runner(held):
    host = torch.tensor([5])
    host_uploads.hold(host)
    assert [id(t) for t in held] == [id(host)]


def test_cpu_targets_and_device_sources_are_not_held(held):
    host = torch.tensor([1])
    assert host_uploads.upload(host, "cpu") is host
    on_meta = torch.empty(2, device="meta")
    assert host_uploads.upload(on_meta, "meta").device.type == "meta"
    assert held == []


@pytest.fixture
def patched(monkeypatch):
    """Apply `patch_host_uploads`, restoring everything it rebinds afterwards."""
    original = torch_utils.async_tensor_h2d
    for module in list(sys.modules.values()):
        if getattr(module, "async_tensor_h2d", None) is original:
            monkeypatch.setattr(module, "async_tensor_h2d", original)
    monkeypatch.setattr(mm_inputs, "_nested_tensors_h2d", mm_inputs._nested_tensors_h2d)
    monkeypatch.setattr(
        InputBatch,
        "_make_prompt_token_ids_cpu_tensor",
        InputBatch._make_prompt_token_ids_cpu_tensor,
    )
    monkeypatch.setattr(dp_utils, "_run_ar", dp_utils._run_ar)
    host_uploads.patch_host_uploads()


def test_patched_helpers_route_through_upload(held, patched):
    out = torch_utils.async_tensor_h2d([[1, 2]], device="meta", dtype=torch.int64)
    assert out.device.type == "meta" and len(held) == 1
    assert held[0].dtype == torch.int64

    batch = {"pixels": torch.ones(2, 3), "lens": [torch.tensor([1])], "n": 2}
    moved = mm_inputs._nested_tensors_h2d(batch, device="meta")
    assert moved["pixels"].device.type == "meta" and moved["n"] == 2
    assert len(held) == 3
    # keep_on_cpu fields reach this function with device="cpu".
    kept = mm_inputs._nested_tensors_h2d({"lens": torch.tensor([4])}, device="cpu")
    assert kept["lens"].device.type == "cpu" and len(held) == 3

    # Modules that imported the helper by name before the patch are rebound.
    logits_builtin.async_tensor_h2d([0.5], device="meta", dtype=torch.float32)
    assert len(held) == 4

    # The penalty prompt-token matrix is held as built.
    batch = SimpleNamespace(
        num_reqs=2,
        num_prompt_tokens=np.array([2, 3]),
        token_ids_cpu=np.arange(8).reshape(2, 4),
        vocab_size=99,
    )
    prompt = InputBatch._make_prompt_token_ids_cpu_tensor(batch)
    assert len(held) == 5 and held[-1] is prompt
    assert prompt.tolist() == [[0, 1, 99], [4, 5, 6]]


def test_dp_token_count_upload_is_held(held, patched, monkeypatch):
    """The data-parallel token-count all-reduce uploads a temporary host
    tensor; it is held until the runner's fence."""
    monkeypatch.setattr(dp_utils, "_get_device_and_group", lambda _: ("meta", None))
    monkeypatch.setattr(dp_utils.dist, "all_reduce", lambda tensor, group: None)
    config = SimpleNamespace(data_parallel_size=2, data_parallel_rank=1)
    out = dp_utils._run_ar(
        should_ubatch=False,
        orig_num_tokens_per_ubatch=3,
        padded_num_tokens_per_ubatch=4,
        cudagraph_mode=0,
        parallel_config=config,
    )
    assert out.device.type == "meta"
    assert len(held) == 1
    assert held[0].tolist() == [[0, 3], [0, 4], [0, 0], [0, 0]]


def test_encoder_metadata_uploads_are_held(held):
    """Host tensors a vision tower moves to the device while building its
    encoder metadata are held, strided ones as contiguous copies."""

    class Tower(torch.nn.Module):
        def prepare_encoder_metadata(self, n):
            cos = torch.arange(n, dtype=torch.float32)
            rows = torch.arange(2 * n).reshape(2, n)
            out = torch.empty(n, device="meta")
            out.copy_(torch.arange(2 * n, dtype=torch.float32)[::2])
            return {
                "cos": cos.to(device="meta", non_blocking=True),
                "rev": rows[:, 0].to("meta", non_blocking=True),
                "out": out,
                "stay": torch.zeros(1).to(torch.float64),
            }

    model = torch.nn.Sequential(Tower())
    host_uploads.hold_encoder_metadata_uploads(model)
    host_uploads.hold_encoder_metadata_uploads(model)  # idempotent
    metadata = model[0].prepare_encoder_metadata(3)
    assert metadata["cos"].device.type == "meta"
    assert metadata["stay"].device.type == "cpu"
    assert len(held) == 3
    assert all(t.device.type == "cpu" and t.is_contiguous() for t in held)
    # In call order: the copy_ source, the rotary table, the strided column.
    assert [h.tolist() for h in held] == [[0.0, 2.0, 4.0], [0.0, 1.0, 2.0], [0, 3]]
