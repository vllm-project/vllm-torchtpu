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
"""A v3 request is served by v3 or refused, never by the v2 kernel.

With MOE_FUSED_EP_KERNEL_IMPL=v3 the FP8 MoE method builds the v3 kernel and
never calls v2's prebuild. The NVFP4 and DeepSeek-V4 MXFP4 methods call v2's
prebuild when they would build the W4A8 fused EP kernel
(MOE_FUSED_EP_ENABLE_W4A8=1 under expert parallelism), so that call is where a
v3 request would otherwise turn into v2 without a word.
"""

from types import SimpleNamespace

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as v2

pytestmark = pytest.mark.cpu_test


def _layer(use_ep):
    """With use_ep=False, a layer v2 refuses early for its own reason
    (replicated experts)."""
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=use_ep, pcp_size=1)
        )
    )


@pytest.fixture
def fused_ep_on(monkeypatch):
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_ENABLE_W4A8", False)


@pytest.mark.parametrize("weight_format", ["fp8", "fp4"])
def test_a_v3_request_refuses_to_build_v2(fused_ep_on, monkeypatch, weight_format):
    """fp4 is the call the NVFP4 and MXFP4 methods make: W4A8 on and an
    expert-parallel layer, which v2 would otherwise go on to build."""
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_IMPL", "v3")
    monkeypatch.setattr(envs, "MOE_FUSED_EP_ENABLE_W4A8", weight_format == "fp4")
    with pytest.raises(ValueError, match="MOE_FUSED_EP_KERNEL_IMPL=v3"):
        v2.prebuild_fused_moe_ep(
            _layer(use_ep=weight_format == "fp4"),
            topk=2,
            renormalize=True,
            activation="silu",
            weight_format=weight_format,
        )


@pytest.mark.parametrize("impl", [None, "v2"])
def test_other_requests_reach_v2_as_before(fused_ep_on, monkeypatch, impl):
    """The control: without a v3 request, v2 decides for itself (here it
    refuses the replicated layer and returns None)."""
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_IMPL", impl)
    assert (
        v2.prebuild_fused_moe_ep(
            _layer(use_ep=False), topk=2, renormalize=True, activation="silu"
        )
        is None
    )
