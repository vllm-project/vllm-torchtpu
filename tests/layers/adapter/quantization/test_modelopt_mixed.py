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
"""Tests for ModelOpt MIXED_PRECISION dispatch on TPU.

The config below has the shape of ``quantization_config`` in the
``nvidia/Qwen3.8-2.4T-A95B-NVFP4`` ``config.json``: FP8 attention and
linear-attention projections, NVFP4 routed experts, an unquantized shared
expert, and an excluded MTP head. The tests pin which TPU method each layer
gets; they do not load weights.
"""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import get_quantization_config

from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.layers.adapter.quantization.modelopt_mixed import (
    VllmModelOptMixedConfig,
)
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import (
    VllmNvfp4LinearMethod,
    VllmNvfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

LAYER = "model.layers.0"

# Qwen3_5ForCausalLMBase.packed_modules_mapping: vLLM's fused layer name to the
# checkpoint names that quantized_layers is keyed by.
PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


def _quant_config(extra_layers: dict | None = None) -> dict:
    quantized_layers = {
        f"{LAYER}.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
        f"{LAYER}.linear_attn.in_proj_z": {"quant_algo": "FP8"},
        f"{LAYER}.linear_attn.out_proj": {"quant_algo": "FP8"},
        f"{LAYER}.self_attn.q_proj": {"quant_algo": "FP8"},
        f"{LAYER}.self_attn.k_proj": {"quant_algo": "FP8"},
        f"{LAYER}.self_attn.v_proj": {"quant_algo": "FP8"},
        f"{LAYER}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
    }
    quantized_layers.update(extra_layers or {})
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": quantized_layers,
        "ignore": ["mtp*"],
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    }


def _make_config(extra_layers: dict | None = None) -> VllmModelOptMixedConfig:
    cfg = VllmModelOptMixedConfig.from_config(_quant_config(extra_layers))
    assert isinstance(cfg, VllmModelOptMixedConfig)
    cfg.packed_modules_mapping = PACKED_MODULES
    cfg.vllm_config = MagicMock()
    cfg.vllm_config.parallel_config.enable_expert_parallel = False
    return cfg


def _linear() -> LinearBase:
    """A bare LinearBase: dispatch only needs the type and output_size."""
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.input_size = 64
    layer.output_size = 32
    return layer


class _FakeRoutedExperts(RoutedExperts):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock()
        self.moe_config.has_bias = False


class TestRegistration:
    def test_name_matches_vllm_detection(self):
        assert VllmModelOptMixedConfig.get_name() == "modelopt_mixed"

    def test_registered_under_tpu_name(self):
        assert get_quantization_config("tpu-modelopt_mixed") is VllmModelOptMixedConfig

    def test_platform_admits_method(self):
        assert "modelopt_mixed" in TpuPlatform.supported_quantization


class TestDispatch:
    @pytest.mark.parametrize(
        "prefix",
        [
            f"{LAYER}.linear_attn.out_proj",
            # Fused layers: every checkpoint shard is FP8.
            f"{LAYER}.linear_attn.in_proj_qkvz",
            f"{LAYER}.self_attn.qkv_proj",
        ],
    )
    def test_fp8_linear(self, prefix):
        cfg = _make_config()
        method = cfg.get_quant_method(_linear(), prefix)
        assert isinstance(method, VllmFp8LinearMethodTPU)
        # ModelOpt FP8 linears carry a per-tensor weight scale and a static
        # input scale; both need a parameter to load into.
        assert method.act_q_static
        assert not method.block_quant
        assert method.quant_config.is_checkpoint_fp8_serialized

    def test_nvfp4_experts(self):
        cfg = _make_config()
        method = cfg.get_quant_method(_FakeRoutedExperts(), f"{LAYER}.mlp.experts")
        assert isinstance(method, VllmNvfp4MoEMethod)
        assert method.group_size == 16

    def test_nvfp4_linear(self):
        cfg = _make_config({f"{LAYER}.mlp.down": {"quant_algo": "W4A16_NVFP4"}})
        method = cfg.get_quant_method(_linear(), f"{LAYER}.mlp.down")
        assert isinstance(method, VllmNvfp4LinearMethod)
        assert method.group_size == 16

    @pytest.mark.parametrize(
        "prefix",
        [
            # In the checkpoint but absent from quantized_layers.
            f"{LAYER}.mlp.shared_expert.down_proj",
            f"{LAYER}.mlp.shared_expert.gate_up_proj",
            f"{LAYER}.mlp.gate",
            # Matched by the "mtp*" ignore pattern.
            "mtp.layers.0.self_attn.o_proj",
        ],
    )
    def test_unquantized_linear(self, prefix):
        cfg = _make_config()
        method = cfg.get_quant_method(_linear(), prefix)
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_excluded_experts_are_unquantized(self, vllm_config_context):
        cfg = _make_config()
        method = cfg.get_quant_method(_FakeRoutedExperts(), "mtp.layers.0.mlp.experts")
        assert isinstance(method, VllmUnquantizedFusedMoEMethod)

    def test_attention_has_no_method(self):
        cfg = _make_config()
        layer = Attention.__new__(Attention)
        torch.nn.Module.__init__(layer)
        assert cfg.get_quant_method(layer, f"{LAYER}.self_attn.attn") is None

    def test_unsupported_linear_algo_raises(self):
        cfg = _make_config({f"{LAYER}.mlp.down": {"quant_algo": "MXFP8"}})
        with pytest.raises(NotImplementedError, match="MXFP8"):
            cfg.get_quant_method(_linear(), f"{LAYER}.mlp.down")

    def test_unsupported_experts_algo_raises(self):
        cfg = _make_config({f"{LAYER}.mlp.experts": {"quant_algo": "FP8"}})
        with pytest.raises(NotImplementedError, match="FP8"):
            cfg.get_quant_method(_FakeRoutedExperts(), f"{LAYER}.mlp.experts")
