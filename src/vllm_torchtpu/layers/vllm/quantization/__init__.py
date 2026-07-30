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
TPU Quantization Module for vLLM.

This module provides TPU-specific quantization configurations and methods,
allowing vLLM to use custom quantization backends on TPU hardware.

Flow:
1. vLLM loads a model with quantization config (e.g., "mxfp4")
2. get_tpu_quantization_config() is called to get a TPU-specific config
3. The config returns TPU-specific quant methods for each layer type
4. During model loading, these methods handle weight processing
5. During inference, the apply() method runs TPU-optimized kernels
"""

import copy
from typing import Dict, Optional, Type

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig

from vllm_torchtpu.layers.common import quant_methods
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors import \
    VllmCompressedTensorsConfig
from vllm_torchtpu.layers.vllm.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.layers.vllm.quantization.mxfp4 import VllmMxfp4Config
from vllm_torchtpu.layers.vllm.quantization.nvfp4 import VllmNvfp4Config
from vllm_torchtpu.layers.vllm.quantization.unquantized import \
    VllmUnquantizedConfig
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def get_tpu_quantization_config(
    vllm_config: VllmConfig, ) -> QuantizationConfig:
    """
    Get TPU-specific quantization configuration.

    This function is called during model initialization to replace the default
    quantization config with a TPU-compatible version.

    Args:
        vllm_config: The vLLM configuration containing model and quant settings.

    Returns:
        A TPU-compatible QuantizationConfig.

    Raises:
        NotImplementedError: If the quantization method is not supported on TPU.
    """
    model_config = copy.deepcopy(vllm_config.model_config)

    # Map from quant method name to TPU config class
    method_to_config: Dict[Optional[str], Type[VllmQuantConfig]] = {
        None: VllmUnquantizedConfig,
        quant_methods.FP8: VllmFp8Config,
        quant_methods.MXFP4: VllmMxfp4Config,
        quant_methods.NVFP4: VllmNvfp4Config,
        quant_methods.COMPRESSED_TENSORS: VllmCompressedTensorsConfig,
        # TODO: Add more quantization methods as needed
        # quant_methods.AWQ: VllmAWQConfig,
    }

    if model_config.quantization not in method_to_config:
        raise NotImplementedError(
            f"{model_config.quantization} quantization method not supported on TPU. "
            f"Supported methods are: {list(method_to_config.keys())}")

    quant_config_cls = method_to_config[model_config.quantization]
    assert issubclass(quant_config_cls, VllmQuantConfig)

    # Set global config for this quantization class
    quant_config_cls.set_configs(vllm_config)

    # Register the TPU quant method name so vLLM uses our custom config
    model_config.quantization = quant_methods.get_tpu_quant_method(
        quant_config_cls.get_name())
    return VllmConfig.get_quantization_config(model_config,
                                              vllm_config.load_config)
