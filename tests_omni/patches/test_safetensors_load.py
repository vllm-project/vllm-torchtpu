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
"""Unit tests for src/vllm_torchtpu/omni/patches/safetensors_load.py."""

import inspect
import re

import pytest
import safetensors.torch as safetensors_torch
import torch
from safetensors import SafetensorError

from vllm_torchtpu.omni.patches import apply_omni_tpu_patches
from vllm_torchtpu.omni.patches.safetensors_load import (
    patch_safetensors_load_file_tpu_device,
)

pytestmark = [pytest.mark.omni]

TPU_DEVICES = ["tpu", "tpu:0", torch.device("tpu:0")]

# The unpatched function, even if the platform already installed the wrapper.
_REAL_LOAD_FILE = inspect.unwrap(safetensors_torch.load_file)


def _apply_patch_over(monkeypatch, load_file):
    """Make `load_file` the current safetensors function and patch over it."""
    monkeypatch.setattr(safetensors_torch, "load_file", load_file)
    patch_safetensors_load_file_tpu_device.cache_clear()
    patch_safetensors_load_file_tpu_device()


@pytest.fixture
def patched(monkeypatch):
    """The real safetensors load_file with the patch applied over it."""
    _apply_patch_over(monkeypatch, _REAL_LOAD_FILE)


@pytest.fixture
def checkpoint(tmp_path):
    """A real .safetensors file with a few dtypes, and its tensors."""
    tensors = {
        "f32": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bf16": torch.linspace(-1, 1, 4).to(torch.bfloat16),
        "i64": torch.tensor([1, -2, 3], dtype=torch.int64),
    }
    path = tmp_path / "w.safetensors"
    safetensors_torch.save_file(tensors, str(path))
    return path, tensors


def test_safetensors_still_rejects_tpu_device(checkpoint):
    """Canary: the real safetensors still cannot load straight to TPU.

    This test is meant to fail once safetensors accepts torch_tpu devices. When
    it does, remove the patch and this test file (see the TODO in the patch
    module).
    """
    path, _ = checkpoint
    with pytest.raises(SafetensorError, match="device"):
        _REAL_LOAD_FILE(str(path), device="tpu:0")


def test_platform_patches_install_the_wrapper(monkeypatch):
    """apply_omni_tpu_patches(), run at platform construction, installs it."""
    monkeypatch.delenv("VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT", raising=False)
    monkeypatch.setattr(safetensors_torch, "load_file", _REAL_LOAD_FILE)
    patch_safetensors_load_file_tpu_device.cache_clear()
    apply_omni_tpu_patches.cache_clear()
    apply_omni_tpu_patches()
    assert safetensors_torch.load_file is not _REAL_LOAD_FILE
    assert safetensors_torch.load_file.__wrapped__ is _REAL_LOAD_FILE


def test_patch_installs_wrapper_once(patched):
    wrapper = safetensors_torch.load_file
    assert wrapper.__wrapped__ is _REAL_LOAD_FILE
    patch_safetensors_load_file_tpu_device()  # @run_once: no second wrapper
    assert safetensors_torch.load_file is wrapper


@pytest.mark.parametrize("device", TPU_DEVICES)
def test_real_file_loads_to_tpu(patched, checkpoint, device):
    path, tensors = checkpoint
    expected = _REAL_LOAD_FILE(str(path), device="cpu")
    state = safetensors_torch.load_file(str(path), device=device)
    assert state.keys() == tensors.keys()
    for name, tensor in state.items():
        assert tensor.device.type == "tpu"
        assert tensor.dtype == expected[name].dtype
        assert torch.equal(tensor.cpu(), expected[name])


def _cpu_error(path):
    with pytest.raises(Exception) as excinfo:
        _REAL_LOAD_FILE(str(path), device="cpu")
    return excinfo.value


def test_missing_file_raises_the_cpu_error(patched, tmp_path):
    """A missing file on TPU fails as it does on CPU, not as a device error."""
    path = tmp_path / "missing.safetensors"
    cpu_error = _cpu_error(path)
    with pytest.raises(type(cpu_error), match=re.escape(str(cpu_error))):
        safetensors_torch.load_file(str(path), device="tpu:0")


def test_damaged_file_raises_the_cpu_error(patched, tmp_path):
    """A damaged file on TPU fails as it does on CPU, not as a device error."""
    path = tmp_path / "damaged.safetensors"
    path.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00not json")
    cpu_error = _cpu_error(path)
    with pytest.raises(type(cpu_error), match=re.escape(str(cpu_error))):
        safetensors_torch.load_file(str(path), device="tpu:0")


@pytest.fixture
def recording_load_file(monkeypatch):
    """Patch over a fake load_file that records the (device, kwargs) it gets."""
    calls: list[tuple[str, dict]] = []

    def _load_file(filename, device="cpu", **kwargs):
        calls.append((str(device), kwargs))
        return {"w": torch.zeros(2)}

    _apply_patch_over(monkeypatch, _load_file)
    return calls


@pytest.mark.parametrize("device", ["cpu", "cuda:0", torch.device("cpu")])
def test_non_tpu_devices_go_straight_to_the_original(recording_load_file, device):
    safetensors_torch.load_file("f", device=device)
    assert recording_load_file == [(str(device), {})]


def test_default_device_is_cpu(recording_load_file):
    safetensors_torch.load_file("f")
    assert recording_load_file == [("cpu", {})]


def test_tpu_reads_once_on_cpu_and_passes_kwargs(recording_load_file):
    """safetensors >= 0.8 has a keyword-only `backend=`; it must reach the read."""
    safetensors_torch.load_file("f", device="tpu:0", backend="pread")
    assert recording_load_file == [("cpu", {"backend": "pread"})]
