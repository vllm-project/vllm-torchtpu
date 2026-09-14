# Copyright 2025 Google LLC
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
"""Unit tests for QUANTIZE_ON_LOAD_PREFIXES and should_quantize_on_load."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.layers.adapter.quantization.configs import (
    VllmQuantConfig, should_quantize_on_load)
from vllm_torchtpu.layers.adapter.quantization.fp8 import \
    VllmFp8LinearMethodTPU
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig, VllmUnquantizedLinearMethod)


@pytest.fixture
def linear_layer():
    """A bare LinearBase instance for testing get_quant_method dispatch."""
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.output_size = 32
    layer.input_size = 64
    return layer


@pytest.fixture(autouse=True)
def mock_vllm_config(monkeypatch):
    """Ensure VllmQuantConfig.vllm_config is set for get_linear_config."""
    monkeypatch.setattr(VllmQuantConfig, "vllm_config", MagicMock())


class TestShouldQuantizeOnLoad:

    def test_empty_env_returns_false(self, monkeypatch):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "")
        assert not should_quantize_on_load("model.layers.0.self_attn.q_proj")

    def test_single_prefix_match(self, monkeypatch):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "self_attn")
        assert should_quantize_on_load("model.layers.0.self_attn.q_proj")
        assert should_quantize_on_load("model.layers.0.self_attn.attn")
        assert not should_quantize_on_load("model.layers.0.mlp.gate_proj")

    def test_dot_bounded_prevents_partial_substring_match(self, monkeypatch):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "attn")
        # "attn" should not match "self_attn" as a partial token
        assert not should_quantize_on_load("model.layers.0.self_attn.q_proj")
        # "attn" should match ".attn." as a dot-bounded token
        assert should_quantize_on_load("model.layers.0.self_attn.attn")

    def test_multiple_prefixes(self, monkeypatch):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES",
                           "self_attn,shared_experts,layers.0.mlp")
        assert should_quantize_on_load("model.layers.2.self_attn.o_proj")
        assert should_quantize_on_load(
            "model.layers.5.shared_experts.down_proj")
        assert should_quantize_on_load("model.layers.0.mlp.gate_up_proj")
        assert not should_quantize_on_load("model.layers.1.mlp.gate_up_proj")


class TestUnquantizedConfigQuantizeOnLoadDispatch:

    def test_matching_linear_layer_uses_fp8_method(self, monkeypatch,
                                                   linear_layer):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "self_attn")
        cfg = VllmUnquantizedConfig()
        method = cfg.get_quant_method(linear_layer,
                                      prefix="model.layers.0.self_attn.q_proj")
        assert isinstance(method, VllmFp8LinearMethodTPU)

    def test_kv_b_proj_is_skipped_for_linear_method(self, monkeypatch,
                                                    linear_layer):
        """kv_b_proj must remain unquantized here because MLAAttention slices W_UK_T/W_UV and quantizes them separately."""
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "self_attn")
        cfg = VllmUnquantizedConfig()
        method = cfg.get_quant_method(
            linear_layer, prefix="model.layers.0.self_attn.kv_b_proj")
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_non_matching_linear_layer_uses_unquantized_method(
            self, monkeypatch, linear_layer):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "self_attn")
        cfg = VllmUnquantizedConfig()
        method = cfg.get_quant_method(linear_layer,
                                      prefix="model.layers.0.mlp.down_proj")
        assert isinstance(method, VllmUnquantizedLinearMethod)
