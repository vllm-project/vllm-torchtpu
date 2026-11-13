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
"""Tests for the k2-fsa/OmniVoice TPU patch in vllm_torchtpu.omni.patches."""

import inspect

import pytest
import safetensors.torch as safetensors_torch
import torch
from safetensors import SafetensorError

from vllm_torchtpu.omni.patches import _patch_omnivoice_cpu_safetensors_load

pytestmark = [pytest.mark.omni]

_WEIGHTS = {"w": torch.arange(4, dtype=torch.float32)}


@pytest.fixture
def fake_load_file(monkeypatch):
    """Install a fake safetensors load_file, then apply the patch on top of it.

    The fake records (device, kwargs) for each call and raises whatever the test
    puts in `fail_on` for that device.
    """
    calls: list[tuple[str, dict]] = []
    fail_on: dict[str, Exception] = {}

    def _load_file(filename, device="cpu", **kwargs):
        calls.append((str(device), kwargs))
        if str(device) in fail_on:
            raise fail_on[str(device)]
        return {k: v.clone() for k, v in _WEIGHTS.items()}

    monkeypatch.setattr(safetensors_torch, "load_file", _load_file)
    _patch_omnivoice_cpu_safetensors_load.cache_clear()
    _patch_omnivoice_cpu_safetensors_load()
    yield _load_file, calls, fail_on
    # The next test applies the patch on its own fake again.
    _patch_omnivoice_cpu_safetensors_load.cache_clear()


def test_torch_tpu_safetensors_still_rejects_tpu_device(tmp_path):
    """Canary: the real safetensors still cannot load straight to TPU.

    Informational only: once this fails, the fallback in
    _patch_omnivoice_cpu_safetensors_load never runs and the patch can go.
    """
    path = tmp_path / "w.safetensors"
    safetensors_torch.save_file({"w": torch.ones(2)}, str(path))
    real_load_file = inspect.unwrap(safetensors_torch.load_file)
    with pytest.raises(SafetensorError):
        real_load_file(str(path), device="tpu:0")


def test_patch_installs_wrapper_once_without_touching_original(fake_load_file):
    original, _, _ = fake_load_file
    patched = safetensors_torch.load_file
    assert patched is not original
    assert patched.__wrapped__ is original
    assert vars(original) == {}  # no attribute was set on the original function

    _patch_omnivoice_cpu_safetensors_load()  # @run_once: no second wrapper
    assert safetensors_torch.load_file is patched


def test_direct_load_is_used_when_it_works(fake_load_file):
    _, calls, _ = fake_load_file
    state = safetensors_torch.load_file("f", device="tpu:0")
    assert calls == [("tpu:0", {})]  # no CPU detour
    torch.testing.assert_close(state["w"], _WEIGHTS["w"])
    assert safetensors_torch.load_file("f")["w"].device.type == "cpu"  # filename only


def test_falls_back_to_cpu_for_unsupported_device(fake_load_file):
    _, calls, fail_on = fake_load_file
    fail_on["tpu:0"] = SafetensorError("device tpu is invalid")
    state = safetensors_torch.load_file("f", device="tpu:0")
    assert calls == [("tpu:0", {}), ("cpu", {})]
    assert state["w"].device.type == "tpu"
    torch.testing.assert_close(state["w"].cpu(), _WEIGHTS["w"])


def test_extra_arguments_are_passed_through(fake_load_file):
    """safetensors >= 0.8 has a keyword-only `backend=`; it must reach the original."""
    _, calls, fail_on = fake_load_file
    safetensors_torch.load_file("f", "cpu", backend="pread")
    fail_on["tpu:0"] = SafetensorError("device tpu is invalid")
    safetensors_torch.load_file("f", device="tpu:0", backend="pread")
    assert calls == [
        ("cpu", {"backend": "pread"}),
        ("tpu:0", {"backend": "pread"}),
        ("cpu", {"backend": "pread"}),  # fallback keeps the same options
    ]


def test_cpu_load_errors_surface_the_original_error(fake_load_file):
    """A failing CPU load is retried once on CPU, then its own error is raised."""
    _, calls, fail_on = fake_load_file
    fail_on["cpu"] = SafetensorError("bad header")
    with pytest.raises(SafetensorError, match="bad header"):
        safetensors_torch.load_file("f", device="cpu")
    assert calls == [("cpu", {}), ("cpu", {})]


def test_file_errors_surface_the_original_error(fake_load_file):
    _, calls, fail_on = fake_load_file
    device_error = SafetensorError("device tpu is invalid")
    cpu_error = SafetensorError("bad header")
    fail_on["tpu:0"] = device_error
    fail_on["cpu"] = cpu_error
    with pytest.raises(SafetensorError) as excinfo:
        safetensors_torch.load_file("f", device="tpu:0")
    assert excinfo.value is device_error
    assert excinfo.value.__cause__ is cpu_error
    assert calls == [("tpu:0", {}), ("cpu", {})]


def test_non_safetensors_errors_propagate(fake_load_file):
    _, calls, fail_on = fake_load_file
    fail_on["tpu:0"] = FileNotFoundError("missing")
    with pytest.raises(FileNotFoundError):
        safetensors_torch.load_file("f", device="tpu:0")
    assert calls == [("tpu:0", {})]
