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
"""Unit tests for DeepSeek-V4 linear weight loading and quantization config resolution."""

import torch
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_torchtpu.layers.common.quant_methods import DEEPSEEK_V4_FP8
from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8LinearMethodTPU


class _DummyDeepseekV4Fp8Config(Fp8Config):
    """Stub DeepSeek-V4 FP8 config for testing weight block size resolution."""

    @classmethod
    def get_name(cls) -> str:
        return DEEPSEEK_V4_FP8


def test_linear_weight_loading_quant_config_resolving():
    """Verify weight block size resolution for dense and MoE prefixes."""
    dummy_quant_config = _DummyDeepseekV4Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[1, 32],
    )

    for prefix in ("model.layers.0.self_attn.fused_wqa_wkv",
                   "model.layers.0.mlp.experts.0.w13"):
        method = VllmFp8LinearMethodTPU(
            quant_config=dummy_quant_config,
            prefix=prefix,
        )
        assert method.block_quant is True
        assert method.weight_block_size == [1, 32]


def test_linear_weight_loading_unquantized_fallback():
    """Verify dynamic quantization fallback when checkpoint is unquantized."""
    dummy_quant_config = _DummyDeepseekV4Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
        weight_block_size=None,
    )
    dense_method = VllmFp8LinearMethodTPU(
        quant_config=dummy_quant_config,
        prefix="model.layers.0.self_attn.fused_wqa_wkv",
    )
    layer = torch.nn.Linear(64, 32, bias=False)
    dense_method.process_weights_after_loading(layer)
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_scale")
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_block_size == (1, 64)
