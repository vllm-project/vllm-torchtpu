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
"""Compressed Tensors MoE Dispatcher for TPU."""

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import \
    CompressedTensorsMoEMethod

from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16 import \
    VllmCompressedTensorsW4A16MoEMethod
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import \
    VllmCompressedTensorsW4ANMxfp4MoEMethod
from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8MoEMethodTPU
from vllm_torchtpu.layers.vllm.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod

if TYPE_CHECKING:
    from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors import \
        VllmCompressedTensorsConfig


class VllmCompressedTensorsMoEMethod(CompressedTensorsMoEMethod):
    """TPU dispatcher for Compressed Tensors MoE schemes."""

    @staticmethod
    def get_moe_method(
        quant_config: "VllmCompressedTensorsConfig",  # type: ignore
        layer: torch.nn.Module,
        layer_name: str,
    ) -> CompressedTensorsMoEMethod:
        assert isinstance(layer, RoutedExperts)

        # RoutedExperts was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        quant_config._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        # TODO: refactor this to use expert_mapping and check all layer numbers
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()

        # multiple schemes found
        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError("All MoE projections need to have same "
                             "quantization scheme but found multiple")

        if scheme_dict is None:  # ignored layer
            return VllmUnquantizedFusedMoEMethod(
                quant_config.get_moe_config(layer))

        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")

        # Have to keep the imports here to prevent circular import
        from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors import (
            _build_fp8_config, _is_int4_w4aN, _is_weight_fp8)

        # 1. Dispatch W4A16 MoE
        if _is_int4_w4aN(weight_quant):
            # NOTE: We route both W4AN configurations to the same runner.
            # Under the hood, the GMM kernel (gmm_v2.py) always receives maybe_quantize_lhs=True
            # for INT4 weights. If group_size < 128 (e.g. 16 or 32 for DeepSeek), the kernel
            # will automatically fallback to dequantize-before-matmul (running as W4A16).
            # If group_size >= 128, it will dynamically quantize activations (running as W4A8).
            return VllmCompressedTensorsW4A16MoEMethod(
                weight_quant, input_quant, quant_config.get_moe_config(layer))

        # 2. Dispatch MXFP4 W4 MoE
        if quant_config._is_mxfp4(weight_quant):
            return VllmCompressedTensorsW4ANMxfp4MoEMethod(
                quant_config.get_moe_config(layer))

        # 3. Dispatch FP8 MoE
        if _is_weight_fp8(weight_quant):
            fp8_config = _build_fp8_config(weight_quant, input_quant)
            return VllmFp8MoEMethodTPU(fp8_config,
                                       fp8_config.get_moe_config(layer))
        # Fallback
        raise RuntimeError(
            f"Unsupported TPU FusedMoe scheme: {weight_quant}, {input_quant}")
