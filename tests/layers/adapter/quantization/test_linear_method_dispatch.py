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
"""
Tests that unquantized dense-linear fallbacks use the TPU (k, n) method.

Several quantization schemes hand plain dense layers back to an unquantized
linear method: layers listed as ignored/excluded in the checkpoint, and -- for
MXFP4, which quantizes only MoE -- every linear layer in the model. Those
fallbacks must resolve to `VllmUnquantizedLinearMethod` (canonical (k, n)
layout), not vLLM's stock N-major `UnquantizedLinearMethod`, or the model runs
its attention projections on the slower transposed contraction.
"""

import pytest
import torch
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.layers.adapter.quantization.unquantized import \
    VllmUnquantizedLinearMethod

SKIPPED = "model.layers.0.skipme"


@pytest.fixture
def linear_layer():
    """A bare LinearBase instance.

    get_quant_method only needs `isinstance(layer, LinearBase)` to hold for
    these fallback branches, and constructing a real LinearBase would require
    an initialized tensor-parallel group. __new__ gives an instance that
    satisfies the isinstance check without that dependency.
    """
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    # Some schemes build a VllmQuantLinearConfig before the skip check, which
    # reads output_size; nothing else on the layer is touched.
    layer.output_size = 32
    layer.input_size = 64
    return layer


def _fp8_config(cls):
    return cls.from_config({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "ignored_layers": [SKIPPED],
    })


class TestUnquantizedFallbackDispatch:

    def test_fp8_skipped_layer(self, linear_layer):
        from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
        cfg = _fp8_config(VllmFp8Config)
        method = cfg.get_quant_method(linear_layer, prefix=SKIPPED)
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_deepseek_v4_fp8_skipped_layer(self, linear_layer):
        from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import \
            VllmDeepseekV4Fp8Config
        cfg = _fp8_config(VllmDeepseekV4Fp8Config)
        method = cfg.get_quant_method(linear_layer, prefix=SKIPPED)
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_nvfp4_excluded_layer(self, linear_layer):
        from vllm_torchtpu.layers.adapter.quantization.nvfp4 import \
            VllmNvfp4Config
        cfg = VllmNvfp4Config.from_config({"quant_algo": "NVFP4"})
        # ModelOpt nests exclude_modules in the checkpoint config; set it
        # directly so this test pins the dispatch branch, not config parsing.
        cfg.exclude_modules = [SKIPPED]
        assert cfg.is_layer_excluded(
            SKIPPED), "fixture must hit the excluded path"
        method = cfg.get_quant_method(linear_layer, prefix=SKIPPED)
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_mxfp4_uses_kn_for_every_linear_layer(self, linear_layer):
        """MXFP4 quantizes only MoE, so all linear layers take this path."""
        from vllm_torchtpu.layers.adapter.quantization.mxfp4 import \
            VllmMxfp4Config
        cfg = VllmMxfp4Config.from_config({})
        method = cfg.get_quant_method(
            linear_layer, prefix="model.layers.0.self_attn.qkv_proj")
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_unquantized_config_uses_kn(self, linear_layer):
        from vllm_torchtpu.layers.adapter.quantization.unquantized import \
            VllmUnquantizedConfig
        cfg = VllmUnquantizedConfig.from_config({})
        method = cfg.get_quant_method(
            linear_layer, prefix="model.layers.0.self_attn.qkv_proj")
        assert isinstance(method, VllmUnquantizedLinearMethod)
