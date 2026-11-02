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
"""Gemma4 top-k softmax routing matches upstream's torch routing."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models import gemma4

from vllm_torchtpu.models.vllm import gemma4_routing_patch

pytestmark = pytest.mark.usefixtures("default_on_knobs_off")

_UPSTREAM = gemma4.gemma4_routing_function_torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 16, 1000])
def test_matches_upstream_routing(dtype, num_tokens):
    gen = torch.Generator().manual_seed(num_tokens)
    logits = (torch.randn(num_tokens, 128, generator=gen) * 3).to(dtype)
    scale = (torch.rand(128, generator=gen) + 0.5).to(torch.bfloat16)

    ref_w, ref_ids = _UPSTREAM(logits, 8, scale)
    new_w, new_ids = gemma4_routing_patch.gemma4_topk_softmax_routing(logits, 8, scale)

    assert new_ids.dtype == ref_ids.dtype == torch.int32
    assert new_w.dtype == ref_w.dtype == torch.float32
    torch.testing.assert_close(new_ids, ref_ids, rtol=0, atol=0)
    # Upstream normalizes in the logits dtype; this computes in float32.
    tol = 1e-6 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(new_w, ref_w, rtol=tol, atol=tol)


def _config(model_type):
    return SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type))


def test_patch_installs_only_when_enabled_for_gemma4(monkeypatch):
    patch = gemma4_routing_patch.maybe_patch_gemma4_routing
    monkeypatch.setattr(gemma4, "gemma4_routing_function_torch", _UPSTREAM)
    monkeypatch.setenv("TPU_GEMMA4_TOPK_SOFTMAX_ROUTING", "0")
    assert patch(_config("gemma4_text")) is None
    assert gemma4.gemma4_routing_function_torch is _UPSTREAM

    monkeypatch.setenv("TPU_GEMMA4_TOPK_SOFTMAX_ROUTING", "1")
    assert patch(None) is False
    assert patch(_config("qwen3_5_moe_text")) is False
    assert gemma4.gemma4_routing_function_torch is _UPSTREAM

    assert patch(_config("gemma4_text")) is None
    routed = gemma4.gemma4_routing_function_torch
    assert routed is gemma4_routing_patch.gemma4_topk_softmax_routing
