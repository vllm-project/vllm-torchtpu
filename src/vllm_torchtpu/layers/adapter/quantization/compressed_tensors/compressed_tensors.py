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
"""Compressed Tensors Quantization Config for TPU."""

from typing import Optional

import torch
from compressed_tensors.quantization import (QuantizationArgs,
                                             QuantizationStrategy,
                                             QuantizationType)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig, CompressedTensorsLinearMethod,
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target, should_ignore_layer)

from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import \
    VllmCompressedTensorsMoEMethod
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config, VllmFp8LinearMethodTPU)
from vllm_torchtpu.layers.adapter.quantization.unquantized import \
    VllmUnquantizedConfig
from vllm_torchtpu.layers.core.quant_methods import (COMPRESSED_TENSORS,
                                                     get_tpu_quant_method)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _is_weight_fp8(weight_quant: Optional[QuantizationArgs]) -> bool:
    return (weight_quant is not None
            and weight_quant.type == QuantizationType.FLOAT
            and weight_quant.num_bits == 8)


def _is_int4_w4aN(weight_quant: Optional[QuantizationArgs]) -> bool:
    if weight_quant is None:
        return False
    is_int4_weight = (int(weight_quant.num_bits) == 4
                      and weight_quant.type == QuantizationType.INT)
    is_group_or_channel_weight = weight_quant.strategy in [
        QuantizationStrategy.GROUP, QuantizationStrategy.CHANNEL
    ]
    is_static_weight = not weight_quant.dynamic

    return is_int4_weight and is_group_or_channel_weight and is_static_weight


def _build_fp8_config(
    weight_quant: QuantizationArgs,
    input_quant: Optional[QuantizationArgs],
) -> VllmFp8Config:
    """Adapt a matched compressed-tensors FP8 scheme into a VllmFp8Config.

    This reuses VllmFp8Config/VllmFp8LinearMethodTPU/VllmFp8MoEMethodTPU's
    existing dequant/requant runtime path instead of duplicating it.
    """
    weight_block_size = (weight_quant.block_structure if weight_quant.strategy
                         == QuantizationStrategy.BLOCK else None)
    activation_scheme = ("dynamic" if (input_quant is None
                                       or input_quant.dynamic) else "static")
    fp8_config = VllmFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme=activation_scheme,
        weight_block_size=weight_block_size,
    )
    fp8_config.is_channel_quant = (
        weight_quant.strategy == QuantizationStrategy.CHANNEL)
    return fp8_config


def _build_fp8_linear_method(
    layer: LinearBase,
    weight_quant: QuantizationArgs,
    input_quant: Optional[QuantizationArgs],
) -> VllmFp8LinearMethodTPU:
    fp8_config = _build_fp8_config(weight_quant, input_quant)
    return VllmFp8LinearMethodTPU(fp8_config,
                                  fp8_config.get_linear_config(layer))


def _raise_not_implemented(scheme_class: str) -> None:
    raise NotImplementedError(
        f"Compressed-tensors scheme '{scheme_class}' is not implemented yet in vllm-torchtpu."
    )


@register_quantization_config(get_tpu_quant_method(COMPRESSED_TENSORS))
class VllmCompressedTensorsConfig(CompressedTensorsConfig, VllmQuantConfig):

    @classmethod
    def get_name(cls) -> str:
        return COMPRESSED_TENSORS

    @classmethod
    def set_configs(cls, vllm_config) -> None:
        super().set_configs(vllm_config)
        # get_quant_method() below builds VllmFp8Config instances and calls
        # their (inherited from VllmQuantConfig) get_linear_config() /
        # get_moe_config(), which read VllmFp8Config's own class-level
        # vllm_config -- so it must be populated too, not just ours.
        VllmFp8Config.set_configs(vllm_config)

    def get_scheme(self,
                   layer: torch.nn.Module,
                   layer_name: Optional[str] = None
                   ) -> Optional["CompressedTensorsScheme"]:
        """
        compressed-tensors supports non uniform in the following way:

        targets of config_groups: There can be N config_groups which each
            have a quantization scheme. Each config_group has a list of targets
            which can be a full layer_name, a regex for a layer_name, or
            an nn.Module name.

        Detect whether a layer_name is found in any target and
        use the quantization scheme corresponding to the matched target
        to select the CompressedTensorsScheme used for inference.
        """

        # Will be empty for models with only sparsity
        weight_quant = input_quant = None
        if self.target_scheme_map:
            matched_target = find_matched_target(
                layer_name=layer_name,
                module=layer,
                targets=self.target_scheme_map.keys(),
                fused_mapping=self.packed_modules_mapping,
            )
            if matched_target is not None:
                scheme_dict = self.target_scheme_map[matched_target]
                weight_quant = scheme_dict.get("weights")
                input_quant = scheme_dict.get("input_activations")

        if weight_quant is None:
            logger.warning_once("Acceleration for non-quantized schemes is "
                                "not supported by Compressed Tensors. "
                                "Falling back to UnquantizedLinearMethod")
            return None

        # TODO: Add support for the unsupported format
        # We raise NotImplementedError for all schemes in this package since
        # we don't have the VllmCompressedTensors... scheme classes ported yet.
        if self._is_fp8_w4a8(weight_quant, input_quant):
            _raise_not_implemented("VllmCompressedTensorsW4A8Fp8")

        if self._is_nvfp4_format(weight_quant):
            _raise_not_implemented("VllmCompressedTensorsW4A4Fp4")

        if self._is_fp8_w8a8(weight_quant, input_quant):
            _raise_not_implemented("VllmCompressedTensorsW8A8Fp8")

        if input_quant is not None and self._is_dynamic_token_w8a8(
                weight_quant, input_quant):
            _raise_not_implemented("VllmCompressedTensorsW8A8Int8")

        raise NotImplementedError(
            "No compressed-tensors compatible scheme was found for layer "
            f"{layer_name}.")

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if should_ignore_layer(prefix,
                               ignore=self.ignore,
                               fused_mapping=self.packed_modules_mapping):
            return VllmUnquantizedConfig.get_quant_method(self, layer, prefix)

        match layer:
            case LinearBase():
                scheme_dict = self.get_scheme_dict(layer, prefix)
                weight_quant = scheme_dict.get(
                    "weights") if scheme_dict else None
                input_quant = scheme_dict.get(
                    "input_activations") if scheme_dict else None

                if weight_quant is None:
                    return VllmUnquantizedConfig.get_quant_method(
                        self, layer, prefix)

                # This is a bypass way of handling FP8 as a compressed tensors method for this is
                # not implemented yet in vllm-torchtpu. Since this was an
                # existing code, it has been moved here to ensure backward compatibility
                if _is_weight_fp8(weight_quant):
                    return _build_fp8_linear_method(layer, weight_quant,
                                                    input_quant)

                # get_scheme will raise NotImplementedError since custom CompressedTensorsScheme subclasses (like
                # W4A8, W8A8, NVFP4) have not been ported to vllm-torchtpu yet (except for non-quantized layouts).
                scheme = self.get_scheme(layer=layer, layer_name=prefix)
                if scheme is None:
                    return VllmUnquantizedConfig.get_quant_method(
                        self, layer, prefix)
                layer.scheme = scheme
                return CompressedTensorsLinearMethod(self)

            case RoutedExperts():
                layer.moe_config = self.get_moe_config(layer)
                return VllmCompressedTensorsMoEMethod.get_moe_method(
                    self, layer, layer_name=prefix)

            case Attention():
                # TODO: KV-cache quantization for compressed-tensors checkpoints
                # is not implemented on TPU yet.
                return None

            case _:
                return None
