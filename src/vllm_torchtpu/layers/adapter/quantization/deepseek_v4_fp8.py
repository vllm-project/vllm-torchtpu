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

import torch
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.utils import replace_parameter
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

from vllm_torchtpu.layers.adapter.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8LinearMethodTPU,
    VllmFp8MoEMethodTPU,
)
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import VllmDeepseekV4Mxfp4MoEMethod
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
)
from vllm_torchtpu.layers.core.quant_methods import (
    DEEPSEEK_V4_FP8,
    get_tpu_quant_method,
)
from vllm_torchtpu.layers.core.quantization import e8m0_to_fp32
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class VllmDeepseekV41FlashFp8LinearMethod(VllmFp8LinearMethodTPU):
    """FP8 linear method for DeepSeek-V4.1-Flash's 32x32 ue8m0 block-scaled layers."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        scale_inv = getattr(layer, "weight_scale_inv", None)
        if scale_inv is not None and scale_inv.dtype == torch.float8_e8m0fnu:
            # Decode the exponent bytes exactly instead of a dtype cast.
            replace_parameter(
                layer,
                "weight_scale_inv",
                e8m0_to_fp32(scale_inv.data.view(torch.uint8)),
            )
        dtype_name, dtype, block_size, _ = self._linear_quant_config
        out_dim, in_dim = layer.weight.shape
        if block_size is not None and (
            # The fused o-projection reads a per-channel scale.
            self.prefix.endswith(".wo_a")
            # Blockwise requantization needs whole blocks.
            or in_dim % block_size
            or out_dim % 128
        ):
            self._linear_quant_config = (dtype_name, dtype, None, False)
        super().process_weights_after_loading(layer)


@register_quantization_config(DEEPSEEK_V4_FP8)
@register_quantization_config(get_tpu_quant_method(DEEPSEEK_V4_FP8))
class VllmDeepseekV4Fp8Config(DeepseekV4FP8Config, VllmQuantConfig):
    """TPU quantization config for "deepseek_v4_fp8" format.

    Registered under "tpu-deepseek_v4_fp8".
    Dispatches MoE quant methods based on the checkpoint's expert_dtype:
      - "fp4" → VllmDeepseekV4Mxfp4MoEMethod
      - "fp8" → VllmFp8MoEMethodTPU
    Linear layers use VllmFp8LinearMethodTPU (FP8 block-quant), or
    VllmDeepseekV41FlashFp8LinearMethod for V4.1-Flash's 32x32 ue8m0 layout.
    """

    @property
    def is_scale_e8m0(self) -> bool:
        try:
            from vllm.config import get_current_vllm_config

            hf_config = get_current_vllm_config().model_config.hf_config
            quant_cfg = getattr(hf_config, "quantization_config", None) or {}
            if isinstance(quant_cfg, dict) and quant_cfg.get("scale_fmt") == "ue8m0":
                return True
        except Exception:  # noqa: S110
            pass
        return super().is_scale_e8m0

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return DEEPSEEK_V4_FP8

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        # This class replaces the fork's registration, but the V4 base claims
        # model_type deepseek_v4 only; V4.1 checkpoints use the same format.
        model_type = getattr(hf_config, "model_type", None)
        if model_type in ("deepseek_v41", "deepseek_v41_text"):
            user_quant = DEEPSEEK_V4_FP8
        return super().override_quantization_method(hf_quant_cfg, user_quant, hf_config)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> vllm_linear.LinearMethodBase | QuantizeMethodBase | None:
        if isinstance(layer, vllm_linear.LinearBase):
            linear_config = self.get_linear_config(layer)
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                from vllm_torchtpu.layers.adapter.quantization.unquantized import (
                    VllmUnquantizedLinearMethod,
                )

                return VllmUnquantizedLinearMethod()
            # Same layout test as the fork's V4.1 quant config.
            if self.weight_block_size == [32, 32] and self.is_scale_e8m0:
                return VllmDeepseekV41FlashFp8LinearMethod(
                    self, linear_config, prefix=prefix
                )
            return VllmFp8LinearMethodTPU(self, linear_config, prefix=prefix)
        elif isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return VllmUnquantizedFusedMoEMethod(layer.moe_config)

            if self.expert_dtype == "fp4":
                if self.moe_quant_algo == "NVFP4":
                    raise NotImplementedError("NVFP4 is not supported yet.")

                moe_config = self.get_moe_config(layer)
                return VllmDeepseekV4Mxfp4MoEMethod(moe_config)
            else:
                return VllmFp8MoEMethodTPU(self, self.get_moe_config(layer))

        return None
