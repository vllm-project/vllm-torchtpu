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
"""ModelOpt MIXED_PRECISION quantization support for TPU.

A ModelOpt mixed-precision checkpoint (``quant_algo: MIXED_PRECISION``, e.g.
``nvidia/Qwen3.8-2.4T-A95B-NVFP4``) names a quantization algorithm per layer in
``quantized_layers``. vLLM resolves it to the ``modelopt_mixed`` method.

This config reuses upstream's checkpoint parsing and per-prefix algorithm
lookup, and dispatches each layer to the TPU method that already serves that
algorithm in a uniform checkpoint:

  - ``FP8`` linear layers: ``VllmFp8LinearMethodTPU`` (per-tensor weight scale
    and a static ``input_scale`` in the checkpoint).
  - ``NVFP4`` / ``W4A16_NVFP4`` linear layers: ``VllmNvfp4LinearMethod``.
  - ``NVFP4`` / ``W4A16_NVFP4`` routed experts: ``VllmNvfp4MoEMethod``.
  - Layers that are excluded or absent from ``quantized_layers``: the
    unquantized TPU methods.

Any other layer/algorithm pair raises, so a checkpoint never loads through a
path that has not been written for it.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)

from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
)
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import (
    VllmNvfp4Config,
    VllmNvfp4LinearMethod,
    VllmNvfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import VllmUnquantizedConfig
from vllm_torchtpu.layers.core.quant_methods import (
    MODELOPT_MIXED,
    get_tpu_quant_method,
)

# Both names store the same packed E2M1 weights and block scales. They differ
# only in the activation precision the checkpoint was calibrated for, which the
# TPU methods choose themselves (see nvfp4.py).
_NVFP4_ALGOS = ("NVFP4", "W4A16_NVFP4")


@register_quantization_config(get_tpu_quant_method(MODELOPT_MIXED))
class VllmModelOptMixedConfig(ModelOptMixedPrecisionConfig, VllmQuantConfig):
    """ModelOpt MIXED_PRECISION config for TPU."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The TPU methods read their checkpoint format from a config of their
        # own type, so build one per algorithm from what upstream parsed.
        # ModelOpt FP8 linears store a per-tensor weight scale and a static
        # input scale; "static" makes the method register both for loading.
        self.tpu_fp8_config = VllmFp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
        )
        self.tpu_nvfp4_config = VllmNvfp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=self.kv_cache_quant_method,
            exclude_modules=[],
            group_size=self.nvfp4_config.group_size,
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return MODELOPT_MIXED

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, Attention):
            # KV-cache quantization on TPU follows --kv-cache-dtype.
            return None
        if self.is_layer_excluded(prefix):
            return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)
        quant_algo = self._resolve_quant_algo(prefix)
        if quant_algo is None:
            return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)

        if isinstance(layer, LinearBase):
            if quant_algo == "FP8":
                return VllmFp8LinearMethodTPU(
                    self.tpu_fp8_config, self.get_linear_config(layer), prefix=prefix
                )
            if quant_algo in _NVFP4_ALGOS:
                return VllmNvfp4LinearMethod(
                    self.tpu_nvfp4_config, self.get_linear_config(layer)
                )
        elif isinstance(layer, RoutedExperts):
            if quant_algo in _NVFP4_ALGOS:
                return VllmNvfp4MoEMethod(
                    self.tpu_nvfp4_config, self.get_moe_config(layer)
                )
        raise NotImplementedError(
            f"ModelOpt MIXED_PRECISION on TPU does not support quant_algo "
            f"{quant_algo!r} for {type(layer).__name__} at {prefix!r}."
        )
