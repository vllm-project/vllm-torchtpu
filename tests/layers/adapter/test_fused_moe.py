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

from enum import Enum
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter.fused_moe import (
    TpuMoEActivationMixin,
    get_fused_moe_activation,
)


class _Activation(Enum):
    SILU = "silu"


@pytest.mark.parametrize("activation", ["silu", _Activation.SILU])
@pytest.mark.parametrize("swiglu_limit", [None, 0.0])
def test_get_fused_moe_activation_preserves_standard_activations(
    activation, swiglu_limit
):
    moe_config = SimpleNamespace(swiglu_limit=swiglu_limit)
    assert get_fused_moe_activation(activation, moe_config) == "silu"


@pytest.mark.parametrize(
    "swiglu_limit,expected",
    [
        # DeepSeek-V4 and Hy4-preview: the name the fused EP kernel matches.
        (10.0, "silu_and_mul_with_clamp"),
        (4.0, "silu_and_mul_with_clamp:4.0"),
    ],
)
def test_get_fused_moe_activation_encodes_swiglu_limit(swiglu_limit, expected):
    moe_config = SimpleNamespace(swiglu_limit=swiglu_limit)

    assert get_fused_moe_activation("silu", moe_config) == expected
    assert get_fused_moe_activation(_Activation.SILU, moe_config) == expected


def test_swiglu_limit_reaches_the_quant_method_activation():
    """The fp8 / unquantized / nvfp4 methods resolve through this mixin."""
    layer = SimpleNamespace(
        activation="silu", moe_config=SimpleNamespace(swiglu_limit=10.0)
    )

    assert (
        TpuMoEActivationMixin()._resolve_tpu_activation(layer)
        == "silu_and_mul_with_clamp"
    )


@pytest.mark.parametrize(
    "linear_beta,expected",
    [
        (None, "situ:4.0:none"),
        (25.0, "situ:4.0:25.0"),
    ],
)
def test_get_fused_moe_activation_encodes_situ(linear_beta, expected):
    moe_config = SimpleNamespace(
        activation_situ_beta=4.0,
        activation_situ_linear_beta=linear_beta,
    )

    assert get_fused_moe_activation("situ", moe_config) == expected


def test_get_fused_moe_activation_requires_situ_beta():
    moe_config = SimpleNamespace(
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )

    with pytest.raises(AssertionError):
        get_fused_moe_activation("situ", moe_config)


_NO_CLAMP = SimpleNamespace(swiglu_limit=None)


@pytest.mark.tpu_test
def test_get_fused_moe_activation_compiles():
    @torch.compile(fullgraph=True)
    def compiled_fn(x, activation):
        act_str = get_fused_moe_activation(activation, _NO_CLAMP)
        if act_str == "silu":
            return x * 2
        return x

    res = compiled_fn(torch.ones(1).to("tpu"), _Activation.SILU)
    assert res.cpu() == 2.0
