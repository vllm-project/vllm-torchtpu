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
"""Tests for skipping unread AOTAutogradCache.save on synthetic `torchtpu_` keys."""

import os
from functools import partial

import pytest
import torch
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache

import vllm_torchtpu
from vllm_torchtpu import patch_registry


def test_patch_registered_for_platform_and_worker_init():
    target = "vllm_torchtpu:_patch_torchtpu_skip_unread_aot_autograd_save"
    matches = [p for p in patch_registry.PATCHES if p.target == target]
    assert len(matches) == 1
    assert set(matches[0].stages) == {"platform_activation", "worker_init"}


def test_skip_torchtpu_save_filters_only_synthetic_keys(monkeypatch):
    saved_calls: list[tuple[str, object, bool]] = []

    def fake_orig_save(key: str, entry: object, remote: bool = False) -> None:
        saved_calls.append((key, entry, remote))

    monkeypatch.setattr(AOTAutogradCache, "save", staticmethod(fake_orig_save))

    vllm_torchtpu._patch_torchtpu_skip_unread_aot_autograd_save()
    first_patched = AOTAutogradCache.save
    # Idempotent when applied again at a second lifecycle boundary.
    vllm_torchtpu._patch_torchtpu_skip_unread_aot_autograd_save()
    assert AOTAutogradCache.save is first_patched

    AOTAutogradCache.save("torchtpu_deadbeef", "synthetic_entry", remote=False)
    assert saved_calls == []

    AOTAutogradCache.save("inductor_real_key", "real_entry", remote=True)
    assert saved_calls == [("inductor_real_key", "real_entry", True)]


def _verify_skip_save_on_tpu(root: str) -> None:
    if not os.path.exists("/dev/vfio"):
        pytest.skip("TPU hardware required")
    os.environ["TORCH_TPU_TIER2_COMPILATION_CACHE"] = "disabled"
    os.environ.pop("TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT", None)
    inductor_dir = os.path.join(root, "inductor")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_dir
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "0"

    from vllm.config.utils import Range

    from vllm_torchtpu.compilation import tpu_compiler
    from vllm_torchtpu.compilation.tpu_compiler import TpuCompilerAdaptor

    patch_registry.apply("worker_init")

    # 1. Standalone @torch.compile(backend="tpu") helper writes no aotautograd file.
    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def helper(x: torch.Tensor) -> torch.Tensor:
        return x + 3.0

    x = torch.arange(8, dtype=torch.float32, device="tpu")
    assert torch.equal(helper(x).cpu(), torch.arange(8, dtype=torch.float32) + 3.0)
    assert not os.path.exists(os.path.join(inductor_dir, "aotautograd"))

    # 2. TpuCompilerAdaptor.compile writes no aotautograd file and preserves
    #    identity between entry.compiled_fw.result and the closure's artifact.
    real_backend = tpu_compiler._tpu_backend
    checked = 0

    def check_identity_backend(graph, example_inputs):
        nonlocal checked
        compiled_fn = real_backend(graph, example_inputs)
        freevars = dict(
            zip(
                compiled_fn.__code__.co_freevars,
                (c.cell_contents for c in compiled_fn.__closure__ or ()),
            )
        )
        assert "compiled_artifacts" in freevars, (
            "TpuBackend.__call__ closure no longer exposes 'compiled_artifacts'; "
            "re-validate entry.compiled_fw.result identity against the new torch_tpu."
        )
        assert (
            compiled_fn.serialize().compiled_fw.result
            is freevars["compiled_artifacts"][0]
        )
        checked += 1
        return compiled_fn

    tpu_compiler._tpu_backend = check_identity_backend
    compiler = TpuCompilerAdaptor()
    compiler.initialize_cache(cache_dir=os.path.join(root, "adaptor_cache"))

    def adaptor_backend(graph, example_inputs):
        fn, handle = compiler.compile(
            graph, example_inputs, {}, Range(start=8, end=8), key="b8"
        )
        assert fn is not None and handle is not None
        return fn

    @torch.compile(backend=adaptor_backend, fullgraph=True, dynamic=False)
    def step(y: torch.Tensor) -> torch.Tensor:
        return y * 2.0

    assert torch.equal(step(x).cpu(), torch.arange(8, dtype=torch.float32) * 2.0)
    assert checked == 1
    assert not os.path.exists(os.path.join(inductor_dir, "aotautograd"))


@pytest.mark.tpu_test
@pytest.mark.spawns_tpu_workers
def test_skip_torchtpu_aot_autograd_save_on_tpu(tmp_path):
    from tests.distributed.tpu_test_utils import run_in_isolated_process

    run_in_isolated_process(
        partial(_verify_skip_save_on_tpu, str(tmp_path)), timeout=180
    )
