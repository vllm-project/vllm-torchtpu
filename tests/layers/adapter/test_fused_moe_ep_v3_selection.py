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
never calls v2's prebuild. Every other method that builds a fused EP kernel
(NVFP4, MXFP4) calls v2's prebuild, so that call is where a v3 request would
otherwise turn into v2 without a word.
"""

from types import SimpleNamespace

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as v2

pytestmark = pytest.mark.cpu_test


def _replicated_layer():
    """A layer v2 refuses early for its own reason (replicated experts)."""
    return SimpleNamespace(
        moe_config=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_ep=False, pcp_size=1)
        )
    )


@pytest.fixture
def fused_ep_on(monkeypatch):
    monkeypatch.setattr(envs, "USE_MOE_FUSED_EP_KERNEL", True)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_ENABLE_W4A8", False)


@pytest.mark.parametrize("weight_format", ["fp8", "fp4"])
def test_a_v3_request_refuses_to_build_v2(fused_ep_on, monkeypatch, weight_format):
    """fp4 is the NVFP4/MXFP4 case; without W4A8 v2 would otherwise return
    None and leave the layer on the grouped-matmul path silently."""
    monkeypatch.setattr(envs, "MOE_FUSED_EP_KERNEL_IMPL", "v3")
    with pytest.raises(ValueError, match="MOE_FUSED_EP_KERNEL_IMPL=v3"):
        v2.prebuild_fused_moe_ep(
            _replicated_layer(),
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
            _replicated_layer(), topk=2, renormalize=True, activation="silu"
        )
        is None
    )
