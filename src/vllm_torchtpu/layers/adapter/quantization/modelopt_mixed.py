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

A ModelOpt mixed-precision checkpoint (``quant_algo: MIXED_PRECISION``) names a
quantization algorithm per layer in ``quantized_layers``. vLLM resolves it to
the ``modelopt_mixed`` method.

This config reuses upstream's checkpoint parsing and per-prefix algorithm
lookup, and dispatches each layer to the TPU method that already serves that
algorithm in a uniform checkpoint:

  - ``FP8`` linear layers: ``VllmFp8LinearMethodTPU`` (per-tensor weight scale
    and a static ``input_scale`` in the checkpoint).
  - ``NVFP4`` / ``W4A16_NVFP4`` linear layers: ``VllmNvfp4LinearMethod``.
  - ``NVFP4`` / ``W4A16_NVFP4`` routed experts: ``VllmNvfp4MoEMethod``.
  - Layers that are excluded or absent from ``quantized_layers``: the
    unquantized TPU methods (with ``QUANTIZE_ON_LOAD_PREFIXES`` support via
    ``resolve_online_fp8``).

Any other layer/algorithm pair raises ``NotImplementedError``, so a checkpoint
never loads through a path that has not been written for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    ModelOptFp8Config,
    ModelOptMixedPrecisionConfig,
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
)

from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
    resolve_online_fp8,
)
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import (
    VllmNvfp4Config,
    VllmNvfp4LinearMethod,
    VllmNvfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import map_online_fp8
from vllm_torchtpu.layers.adapter.quantization.unquantized import VllmUnquantizedConfig
from vllm_torchtpu.layers.core.quant_methods import (
    MODELOPT_MIXED,
    get_tpu_quant_method,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


@register_quantization_config(get_tpu_quant_method(MODELOPT_MIXED))
class VllmModelOptMixedConfig(ModelOptMixedPrecisionConfig, VllmQuantConfig):
    """ModelOpt MIXED_PRECISION config for TPU."""

    def __init__(
        self,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        quantized_layers: dict[str, dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        nvfp4_config: ModelOptNvFp4Config,
        w4a16_nvfp4_config: ModelOptNvFp4Config,
        mxfp8_config: ModelOptMxFp8Config,
    ) -> None:
        super().__init__(
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            quantized_layers=quantized_layers,
            fp8_config=fp8_config,
            nvfp4_config=nvfp4_config,
            w4a16_nvfp4_config=w4a16_nvfp4_config,
            mxfp8_config=mxfp8_config,
        )
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
        self.tpu_w4a16_nvfp4_config = VllmNvfp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=self.kv_cache_quant_method,
            exclude_modules=[],
            group_size=self.w4a16_nvfp4_config.group_size,
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return MODELOPT_MIXED

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        map_online_fp8(self, hf_to_vllm_mapper)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        base_method = self._get_checkpoint_quant_method(layer, prefix)
        return resolve_online_fp8(self, layer, prefix, base_method)

    def _nvfp4_config_for(self, quant_algo: str) -> VllmNvfp4Config | None:
        if quant_algo == "NVFP4":
            return self.tpu_nvfp4_config
        if quant_algo == "W4A16_NVFP4":
            return self.tpu_w4a16_nvfp4_config
        return None

    def _get_checkpoint_quant_method(
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

        nvfp4_cfg = self._nvfp4_config_for(quant_algo)
        if isinstance(layer, LinearBase):
            if quant_algo == "FP8":
                return VllmFp8LinearMethodTPU(
                    self.tpu_fp8_config, self.get_linear_config(layer)
                )
            if nvfp4_cfg is not None:
                return VllmNvfp4LinearMethod(nvfp4_cfg, self.get_linear_config(layer))
        elif isinstance(layer, RoutedExperts):
            if nvfp4_cfg is not None:
                return VllmNvfp4MoEMethod(nvfp4_cfg, self.get_moe_config(layer))
        raise NotImplementedError(
            f"ModelOpt MIXED_PRECISION on TPU does not support quant_algo "
            f"{quant_algo!r} for {type(layer).__name__} at {prefix!r}."
        )
