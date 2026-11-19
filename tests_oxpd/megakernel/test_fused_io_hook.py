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
"""Fused pool I/O selection with and without a registered implementation."""

from __future__ import annotations

import pytest

from vllm_torchtpu import megakernel as mk
from vllm_torchtpu.megakernel import fused_io_hook
from vllm_torchtpu.megakernel.models import qwen35


def _spec_and_exec() -> tuple[mk.ModelSpec, mk.ExecSpec]:
    spec = qwen35.qwen35_spec(qwen35.TINY_EP)
    return spec, mk.ExecSpec(
        R=2, B=8, T=1, ctx=256, cache=mk.CacheExec(paged_kv=True, num_pages=512)
    )


def test_fused_io_registered_by_experimental_package():
    spec, ex = _spec_and_exec()
    assert fused_io_hook.fused_io_available()
    assert mk.should_fuse_pool_io(spec, ex, fused_pool_io="auto") is True
    assert mk.should_fuse_pool_io(spec, ex, fused_pool_io=True) is True


def test_auto_fused_falls_back_to_split_without_implementation(monkeypatch):
    spec, ex = _spec_and_exec()
    monkeypatch.setattr(fused_io_hook, "_REGISTRY", {})
    assert not fused_io_hook.fused_io_available()
    assert mk.should_fuse_pool_io(spec, ex, fused_pool_io="auto") is False
    assert mk.should_fuse_pool_io(spec, ex, fused_pool_io=False) is False


def test_explicit_fused_without_implementation_raises(monkeypatch):
    spec, ex = _spec_and_exec()
    monkeypatch.setattr(fused_io_hook, "_REGISTRY", {})
    with pytest.raises(mk.MegakernelConfigError, match="fused pool I/O"):
        mk.should_fuse_pool_io(spec, ex, fused_pool_io=True)
    with pytest.raises(mk.MegakernelConfigError) as err:
        fused_io_hook.fused_io()
    # Framework-neutral message: no package path of the hosting framework.
    assert "vllm" not in str(err.value)
