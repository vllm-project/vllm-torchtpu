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

from vllm_torchtpu.layers.adapter import fused_moe
from vllm_torchtpu.layers.adapter.fused_moe import get_fused_moe_activation


class _Activation(Enum):
    SILU = "silu"


@pytest.mark.parametrize("activation", ["silu", _Activation.SILU])
def test_get_fused_moe_activation_preserves_standard_activations(activation):
    assert get_fused_moe_activation(activation, object()) == "silu"


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


def test_get_fused_moe_activation_compiles():
    @torch.compile(fullgraph=True)
    def compiled_fn(x, activation):
        act_str = get_fused_moe_activation(activation, None)
        if act_str == "silu":
            return x * 2
        return x

    res = compiled_fn(torch.ones(1).to("tpu"), _Activation.SILU)
    assert res.cpu() == 2.0


def test_custom_op_cache_separates_primary_counts(monkeypatch):
    built = []

    def register(name, function):
        op = SimpleNamespace(register_fake=lambda fn: None, function=function)
        built.append(op)
        return op

    monkeypatch.setattr(fused_moe, "_fused_moe_kernel_cache", {})
    monkeypatch.setattr(fused_moe.pallas, "jax_op", register)
    opts = dict(topk=10, activation="silu", use_ep=True, skip_padded_tokens=True)
    plain = fused_moe._get_fused_moe_custom_op(**opts)
    copies = fused_moe._get_fused_moe_custom_op(**opts, num_primary_experts=64)
    assert plain is not copies
    assert copies is fused_moe._get_fused_moe_custom_op(**opts, num_primary_experts=64)
    assert len(built) == 2
    assert plain.function.keywords["num_primary_experts"] is None
    assert copies.function.keywords["num_primary_experts"] == 64
