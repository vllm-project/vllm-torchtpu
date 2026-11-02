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
"""ModelOpt MXFP8 quantization support for TPU.

Loads NVIDIA ModelOpt MXFP8 checkpoints (`quant_algo: MXFP8`): FP8 E4M3
weights with one E8M0 scale per 32 values along the input dimension. TPU has
no MXFP8 kernel, so the weights are converted to the existing FP8 runtime:

  - Parameter registration reuses vLLM's MXFP8 weight scheme `KMxfp8Static`
    (FP8 `weight`, uint8 E8M0 `weight_scale`).
  - `process_weights_after_loading` decodes the E8M0 scale to FP32 and hands
    the layer to `VllmFp8LinearMethodTPU` as an FP8 checkpoint with
    `weight_block_size = (1, 32)`, which dequantizes, requantizes to the
    configured runtime block, and lays the weight out for the matmul.

The runtime block follows the FP8 runtime's settings. Its default requantizes
per output channel and quantizes activations to FP8, which can cost accuracy;
`REQUANTIZE_BLOCK_SIZE=32` with `ENABLE_QUANTIZED_MATMUL_KERNEL=1` keeps the
checkpoint's own blocks, at the speed of the blockwise kernel.

MoE layers are not supported yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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
    WEIGHT,
    CkptCtx,
    KMxfp8Static,
    ModelOptMxFp8Config,
    Shapes,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)

from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8LinearMethodTPU,
    resolve_online_fp8,
)
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import map_online_fp8
from vllm_torchtpu.layers.core.quant_methods import (
    MODELOPT_MXFP8,
    get_tpu_quant_method,
)
from vllm_torchtpu.layers.core.quantization import e8m0_to_fp32

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_torchtpu.layers.adapter.quantization.configs import TpuLinear
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config


class VllmModelOptMxfp8LinearMethod(VllmFp8LinearMethodTPU):
    """MXFP8 linear for TPU: E8M0 block scales decoded, then the FP8 path."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        tpu_layer = cast("TpuLinear", layer)
        tpu_layer.logical_widths = output_partition_sizes
        tpu_layer.input_size_per_partition = input_size_per_partition
        tpu_layer.output_size_per_partition = sum(output_partition_sizes)
        # vLLM's MXFP8 weight scheme: FP8 `weight` and uint8 E8M0 `weight_scale`
        # with one scale per 32 inputs; it also sets `layer.weight_block_size`
        # to (1, 32), the config's block.
        KMxfp8Static().create_weights(
            layer,
            WEIGHT,
            CkptCtx(),
            Shapes(output_partition_sizes, input_size_per_partition, params_dtype),
            extra_weight_attrs.get("weight_loader"),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, WEIGHT_FLIPPED_ATTR, False):
            return  # already converted; reload paths re-enter this hook
        scale_u8 = cast("torch.Tensor", layer.weight_scale).data
        if scale_u8.dtype != torch.uint8:
            raise ValueError(
                f"MXFP8 expects uint8 E8M0 weight scales, got {scale_u8.dtype}"
            )
        # Decoded on CPU: `torch.ldexp` overflows on TPU (see `e8m0_to_fp32`).
        # The FP8 path reads a block scale from `weight_scale_inv`, multiplies,
        # and replaces both scale parameters with the runtime scale.
        scale = e8m0_to_fp32(scale_u8.to("cpu")).to(scale_u8.device)
        layer.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
        super().process_weights_after_loading(layer)


@register_quantization_config(get_tpu_quant_method(MODELOPT_MXFP8))
class VllmModelOptMxfp8Config(ModelOptMxFp8Config, VllmQuantConfig):
    """MXFP8 config for TPU. Inherits ModelOpt checkpoint parsing from upstream;
    dispatches linear layers to the FP8 runtime."""

    # Read by `VllmFp8LinearMethodTPU`: an FP8 checkpoint with (1, 32) block
    # scales and dynamically quantized activations.
    is_checkpoint_fp8_serialized = True
    weight_block_size = [1, MXFP8_BLOCK_SIZE]
    activation_scheme = "dynamic"

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return MODELOPT_MXFP8

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        map_online_fp8(self, hf_to_vllm_mapper)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        base_method = self._get_checkpoint_quant_method(layer, prefix)
        # The FP8 helpers read only the FP8-config attributes set above.
        return resolve_online_fp8(
            cast("VllmFp8Config", self), layer, prefix, base_method
        )

    def _get_checkpoint_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, Attention):
            # MXFP8 attention/kv-cache quantization is not implemented on TPU.
            return None
        if isinstance(layer, LinearBase):
            if self.is_layer_excluded(prefix):
                from vllm_torchtpu.layers.adapter.quantization.unquantized import (
                    VllmUnquantizedLinearMethod,
                )

                return VllmUnquantizedLinearMethod()
            return VllmModelOptMxfp8LinearMethod(
                cast("Fp8Config", self), self.get_linear_config(layer), prefix
            )
        if isinstance(layer, RoutedExperts):
            if self.is_layer_excluded(prefix):
                from vllm_torchtpu.layers.adapter.quantization.unquantized import (
                    VllmUnquantizedFusedMoEMethod,
                )

                return VllmUnquantizedFusedMoEMethod(self.get_moe_config(layer))
            raise NotImplementedError(
                f"MXFP8 MoE layers are not supported on TPU yet ({prefix})"
            )
        return None
