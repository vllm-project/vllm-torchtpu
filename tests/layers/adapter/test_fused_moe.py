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
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import (
    VllmDeepseekV4Mxfp4MoEMethod,
    VllmMxfp4MoEMethod,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


class _Activation(Enum):
    SILU = "silu"


@pytest.mark.parametrize("activation", ["silu", _Activation.SILU])
def test_get_fused_moe_activation_preserves_standard_activations(activation):
    assert get_fused_moe_activation(activation, object()) == "silu"


@pytest.mark.parametrize("silu_bf16", [False, True])
@pytest.mark.parametrize(
    "activation", ["silu", _Activation.SILU, "gelu", "swigluoai", "situ"]
)
def test_get_fused_moe_activation_resolves_silu_bf16(
    monkeypatch, silu_bf16, activation
):
    monkeypatch.setattr(fused_moe.envs, "TPU_MOE_SILU_BF16", silu_bf16)
    moe_config = SimpleNamespace(
        activation_situ_beta=4.0, activation_situ_linear_beta=None
    )
    expected = {
        "silu": "silu_bf16" if silu_bf16 else "silu",
        _Activation.SILU: "silu_bf16" if silu_bf16 else "silu",
        "gelu": "gelu",
        "swigluoai": "swigluoai",
        "situ": "situ:4.0:none",
    }[activation]
    assert get_fused_moe_activation(activation, moe_config) == expected


@pytest.mark.parametrize("threshold", [0, 64])
def test_silu_bf16_warns_when_it_turns_off_dense_expert(monkeypatch, threshold):
    warnings = []
    monkeypatch.setattr(fused_moe.envs, "TPU_MOE_SILU_BF16", True)
    monkeypatch.setattr(fused_moe.envs, "TPU_MOE_DENSE_EXPERT_THRESHOLD", threshold)
    monkeypatch.setattr(
        fused_moe.logger,
        "warning_once",
        lambda msg, *args: warnings.append(msg % args),
    )

    assert fused_moe.resolve_silu_bf16("silu") == "silu_bf16"
    if threshold:
        assert len(warnings) == 1
        assert "TPU_MOE_SILU_BF16=1" in warnings[0]
        assert "TPU_MOE_DENSE_EXPERT_THRESHOLD=64" in warnings[0]
    else:
        assert warnings == []


@pytest.mark.parametrize("silu_bf16", [False, True])
def test_mxfp4_resolves_silu_bf16_but_keeps_clamped_kernel(monkeypatch, silu_bf16):
    monkeypatch.setattr(fused_moe.envs, "TPU_MOE_SILU_BF16", silu_bf16)
    layer = SimpleNamespace(activation=_Activation.SILU)

    assert VllmMxfp4MoEMethod._resolve_tpu_activation(None, layer) == (
        "silu_bf16" if silu_bf16 else "silu"
    )
    # The clamped kernel keeps float32 intermediates in both settings.
    assert (
        VllmDeepseekV4Mxfp4MoEMethod._resolve_tpu_activation(None, layer)
        == "silu_and_mul_with_clamp"
    )


def test_moe_numerics_env_vars_reach_ray_workers():
    assert "TPU_MOE_TOPK_WEIGHTS_FLOAT32" in TpuPlatform.additional_env_vars
    assert "TPU_MOE_SILU_BF16" in TpuPlatform.additional_env_vars


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


@pytest.mark.tpu_test
def test_get_fused_moe_activation_compiles():
    @torch.compile(fullgraph=True)
    def compiled_fn(x, activation):
        act_str = get_fused_moe_activation(activation, None)
        if act_str == "silu":
            return x * 2
        return x

    res = compiled_fn(torch.ones(1).to("tpu"), _Activation.SILU)
    assert res.cpu() == 2.0
