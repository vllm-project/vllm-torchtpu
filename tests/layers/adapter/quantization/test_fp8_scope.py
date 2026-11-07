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
"""Layer-kind scoping of TPU FP8 via upstream `--quantization-config`.

`--quantization online --quantization-config '{"moe": "fp8_per_channel_static"}'`
quantizes only routed experts; `--quantization fp8_per_channel` with no config
quantizes linear and MoE (same as `fp8` on TPU).
"""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.layers.adapter.quantization import (
    _apply_online_fp8_scope,
    get_tpu_quantization_config,
)
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
    VllmFp8MoEMethodTPU,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core import quant_methods

EXPERTS = "model.layers.0.mlp.experts"
QKV = "model.layers.0.self_attn.qkv_proj"


class FakeRoutedExperts(RoutedExperts):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock()
        self.moe_config.moe_parallel_config = MagicMock()


@pytest.fixture
def linear_layer():
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.output_size = 32
    layer.input_size = 64
    return layer


@pytest.fixture(autouse=True)
def mock_vllm_config(monkeypatch):
    """VllmQuantConfig.vllm_config is read by get_moe_config / set_configs."""
    mock_cfg = MagicMock()
    mock_cfg.parallel_config.enable_expert_parallel = False
    mock_comp = MagicMock()
    mock_comp.custom_ops = ["all"]
    mock_comp.enabled_custom_ops = set()
    mock_comp.disabled_custom_ops = set()
    mock_cfg.compilation_config = mock_comp
    monkeypatch.setattr(VllmQuantConfig, "vllm_config", mock_cfg)
    monkeypatch.setattr("vllm.config.vllm._current_vllm_config", mock_cfg)


def _vllm_config(quantization, quantization_config):
    """A VllmConfig stand-in as seen after EngineArgs resolution.

    `quantization_config` is passed as the dict a user would give on the CLI
    and resolved exactly like `EngineArgs.__post_init__` does.
    """
    from vllm.config.quantization import resolve_quantization_config

    cfg = MagicMock()
    cfg.model_config.quantization = quantization
    cfg.model_config.quantization_config = resolve_quantization_config(
        quantization, quantization_config
    )
    cfg.additional_config = {}
    return cfg


def _get_config(monkeypatch, quantization, quantization_config):
    """Run get_tpu_quantization_config with a BF16 (no quant metadata) model."""
    monkeypatch.setattr(
        "vllm.config.VllmConfig.get_quantization_config",
        lambda model_config, load_config: VllmFp8Config(),
    )
    return get_tpu_quantization_config(_vllm_config(quantization, quantization_config))


class TestScopeResolution:
    def test_online_moe_only(self, monkeypatch):
        cfg = _get_config(
            monkeypatch, quant_methods.ONLINE, {"moe": "fp8_per_channel_static"}
        )
        assert isinstance(cfg, VllmFp8Config)
        assert cfg.quantize_moe is True
        assert cfg.quantize_linear is False

    def test_online_linear_only(self, monkeypatch):
        cfg = _get_config(
            monkeypatch, quant_methods.ONLINE, {"linear": "fp8_per_channel_static"}
        )
        assert cfg.quantize_moe is False
        assert cfg.quantize_linear is True

    def test_fp8_per_channel_default_is_full_scope(self, monkeypatch):
        cfg = _get_config(monkeypatch, quant_methods.FP8_PER_CHANNEL, None)
        assert isinstance(cfg, VllmFp8Config)
        assert cfg.quantize_moe is True
        assert cfg.quantize_linear is True

    def test_fp8_per_channel_linear_null_keeps_shorthand_default(self, monkeypatch):
        """Upstream merges `linear: null` back to the shorthand's spec."""
        cfg = _get_config(monkeypatch, quant_methods.FP8_PER_CHANNEL, {"linear": None})
        assert cfg.quantize_linear is True

    def test_fp8_per_channel_explicit_empty_linear_spec(self, monkeypatch):
        cfg = _get_config(
            monkeypatch,
            quant_methods.FP8_PER_CHANNEL,
            {"linear": {"weight": None, "activation": None}},
        )
        assert cfg.quantize_linear is False
        assert cfg.quantize_moe is True

    def test_online_requires_config(self, monkeypatch):
        with pytest.raises(ValueError, match="requires --quantization-config"):
            _get_config(monkeypatch, quant_methods.ONLINE, None)

    def test_ignore_list_is_forwarded(self, monkeypatch):
        cfg = _get_config(
            monkeypatch,
            quant_methods.ONLINE,
            {"moe": "fp8_per_channel_static", "ignore": ["model.layers.0.mlp"]},
        )
        assert cfg.ignored_layers == ["model.layers.0.mlp"]

    def test_quantization_config_is_cleared_before_upstream(self, monkeypatch):
        seen = {}

        def fake(model_config, load_config):
            seen["quantization_config"] = model_config.quantization_config
            seen["quantization"] = model_config.quantization
            return VllmFp8Config()

        monkeypatch.setattr("vllm.config.VllmConfig.get_quantization_config", fake)
        get_tpu_quantization_config(
            _vllm_config(quant_methods.ONLINE, {"moe": "fp8_per_channel_static"})
        )
        assert seen["quantization_config"] is None
        assert seen["quantization"] == "tpu-fp8"


class TestScopeValidation:
    @pytest.mark.parametrize(
        "weight", ["fp8_per_tensor_static", "fp8_per_block_static"]
    )
    def test_non_per_channel_weight_rejected(self, weight):
        args = QuantizationConfigArgs(moe=weight)
        with pytest.raises(NotImplementedError, match="fp8_per_channel_static"):
            _apply_online_fp8_scope(VllmFp8Config(), args)

    def test_activation_override_rejected(self):
        args = QuantizationConfigArgs(
            moe={"weight": "fp8_per_channel_static", "activation": "fp8_per_token"}
        )
        with pytest.raises(NotImplementedError, match="activation"):
            _apply_online_fp8_scope(VllmFp8Config(), args)

    def test_targets_rejected(self):
        args = QuantizationConfigArgs(targets={"model.layers.*": "fp8_per_channel"})
        with pytest.raises(NotImplementedError, match="targets"):
            _apply_online_fp8_scope(VllmFp8Config(), args)

    def test_serialized_checkpoint_rejected(self):
        """Scoping is for BF16 checkpoints; FP8 checkpoints carry their own."""
        cfg = VllmFp8Config(is_checkpoint_fp8_serialized=True)
        args = QuantizationConfigArgs(moe="fp8_per_channel_static")
        with pytest.raises(NotImplementedError, match="FP8-serialized"):
            _apply_online_fp8_scope(cfg, args)


class TestScopeDispatch:
    def test_moe_only_dispatch(self, linear_layer):
        cfg = VllmFp8Config()
        _apply_online_fp8_scope(
            cfg, QuantizationConfigArgs(moe="fp8_per_channel_static")
        )

        moe = cfg.get_quant_method(FakeRoutedExperts(), prefix=EXPERTS)
        assert isinstance(moe, VllmFp8MoEMethodTPU)

        lin = cfg.get_quant_method(linear_layer, prefix=QKV)
        assert isinstance(lin, VllmUnquantizedLinearMethod)

        assert cfg.get_quant_method(torch.nn.Identity(), prefix="attn") is None

    def test_full_scope_dispatch(self, linear_layer):
        cfg = VllmFp8Config()
        assert isinstance(
            cfg.get_quant_method(FakeRoutedExperts(), prefix=EXPERTS),
            VllmFp8MoEMethodTPU,
        )
        assert isinstance(
            cfg.get_quant_method(linear_layer, prefix=QKV),
            VllmFp8LinearMethodTPU,
        )

    def test_linear_only_dispatch(self, linear_layer):
        cfg = VllmFp8Config()
        _apply_online_fp8_scope(
            cfg, QuantizationConfigArgs(linear="fp8_per_channel_static")
        )
        assert isinstance(
            cfg.get_quant_method(FakeRoutedExperts(), prefix=EXPERTS),
            VllmUnquantizedFusedMoEMethod,
        )
        assert isinstance(
            cfg.get_quant_method(linear_layer, prefix=QKV),
            VllmFp8LinearMethodTPU,
        )
