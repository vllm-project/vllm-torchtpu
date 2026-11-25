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
from vllm_torchtpu.layers.adapter.fused_moe import (
    TpuMoEActivationMixin,
    get_fused_moe_activation,
    resolve_silu_bf16,
)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16 import (  # noqa: E501
    VllmCompressedTensorsW4A16MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import (
    VllmDeepseekV4Mxfp4MoEMethod,
    VllmMxfp4MoEMethod,
)


class _Activation(Enum):
    SILU = "silu"


@pytest.mark.parametrize("activation", ["silu", _Activation.SILU])
def test_get_fused_moe_activation_preserves_standard_activations(activation):
    assert get_fused_moe_activation(activation, object()) == "silu"


def _vllm_config(**additional_config):
    return SimpleNamespace(additional_config=additional_config)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "activation",
    ["silu", "gelu", "swigluoai", "silu_and_mul_with_clamp", "situ:4.0:none"],
)
def test_resolve_silu_bf16_maps_only_silu(enabled, activation):
    expected = "silu_bf16" if enabled and activation == "silu" else activation
    assert resolve_silu_bf16(activation, enabled) == expected


class _Method(TpuMoEActivationMixin):
    pass


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("threshold", [0, 64])
def test_set_tpu_activation_reads_additional_config(monkeypatch, enabled, threshold):
    warnings = []
    monkeypatch.setattr(
        fused_moe,
        "get_current_vllm_config_or_none",
        lambda: _vllm_config(tpu_moe_silu_bf16=enabled),
    )
    monkeypatch.setattr(fused_moe.envs, "TPU_MOE_DENSE_EXPERT_THRESHOLD", threshold)
    monkeypatch.setattr(
        fused_moe.logger, "warning_once", lambda msg, *a: warnings.append(msg % a)
    )
    method = _Method()
    layer = SimpleNamespace(activation=_Activation.SILU, moe_config=None)

    expected = "silu_bf16" if enabled else "silu"
    assert method._set_tpu_activation(layer) == expected
    assert method._tpu_activation_str == expected
    assert method._tpu_topk_weights_float32 is False
    if enabled and threshold:
        assert len(warnings) == 1
        assert "tpu_moe_silu_bf16" in warnings[0]
        assert "TPU_MOE_DENSE_EXPERT_THRESHOLD=64" in warnings[0]
    else:
        assert warnings == []


@pytest.mark.parametrize("enabled", [False, True])
def test_set_tpu_activation_configures_topk_weights_float32(monkeypatch, enabled):
    monkeypatch.setattr(
        fused_moe,
        "get_current_vllm_config_or_none",
        lambda: _vllm_config(tpu_moe_topk_weights_float32=enabled),
    )
    layer = SimpleNamespace(activation=_Activation.SILU, moe_config=None)
    method, other = _Method(), _Method()

    # Only the silu_bf16 setting changes the activation.
    assert method._set_tpu_activation(layer) == "silu"
    # Stored per quant method, so another layer's method is unaffected.
    assert method._tpu_topk_weights_float32 is enabled
    assert other._tpu_topk_weights_float32 is False


@pytest.mark.parametrize("enabled", [False, True])
def test_mxfp4_methods_resolve_silu_bf16(monkeypatch, enabled):
    monkeypatch.setattr(
        fused_moe,
        "get_current_vllm_config_or_none",
        lambda: _vllm_config(tpu_moe_silu_bf16=enabled),
    )
    layer = SimpleNamespace(activation=_Activation.SILU, moe_config=None)

    plain = VllmMxfp4MoEMethod.__new__(VllmMxfp4MoEMethod)
    assert plain._set_tpu_activation(layer) == ("silu_bf16" if enabled else "silu")
    # DeepSeek-V4's clamped SiLU keeps float32 intermediates in both settings.
    clamped = VllmDeepseekV4Mxfp4MoEMethod.__new__(VllmDeepseekV4Mxfp4MoEMethod)
    assert clamped._set_tpu_activation(layer) == "silu_and_mul_with_clamp"


def test_w4a16_resolves_activation_at_weight_load():
    # Resolution reads additional_config and logs, which the compiled forward
    # must not do; the method stores the string at load like the others.
    assert issubclass(VllmCompressedTensorsW4A16MoEMethod, TpuMoEActivationMixin)


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
