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
"""MiniMax M3 registers through its model directory at plugin load."""

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP

from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.layers.adapter.quantization import _MODEL_QUANT_CONFIGS
from vllm_torchtpu.models.minimax_m3.experimental.dspark import (
    ARCHITECTURE,
    MiniMaxM3DSparkForCausalLM,
)
from vllm_torchtpu.models.minimax_m3.experimental.hooks import TPUMiniMaxM3Config
from vllm_torchtpu.models.minimax_m3.experimental.model import (
    MiniMaxM3SparseForCausalLM,
)
from vllm_torchtpu.models.minimax_m3.experimental.quant import MiniMaxM3QuantConfig


def test_minimax_m3_is_registered() -> None:
    register_layers()
    for arch in (
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ):
        assert ModelRegistry.models[arch].load_model_cls() is (
            MiniMaxM3SparseForCausalLM
        )
        assert MODELS_CONFIG_MAP[arch] is TPUMiniMaxM3Config
    assert _MODEL_QUANT_CONFIGS["mxfp8"] is MiniMaxM3QuantConfig


def test_dspark_is_registered() -> None:
    register_layers()
    entry = ModelRegistry.models[ARCHITECTURE]
    assert entry.load_model_cls() is MiniMaxM3DSparkForCausalLM
    # The draft's MXFP8 linears resolve to the TPU MiniMax config.
    assert get_quantization_config("tpu-mxfp8") is MiniMaxM3QuantConfig
