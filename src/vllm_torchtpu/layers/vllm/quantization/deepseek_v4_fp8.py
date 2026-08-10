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

from typing import Optional, Union

import torch
from vllm.model_executor.layers import linear as vllm_linear
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    is_layer_skipped
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config

from vllm_torchtpu.layers.common.quant_methods import (DEEPSEEK_V4_FP8,
                                                       get_tpu_quant_method)
from vllm_torchtpu.layers.vllm.quantization.configs import VllmQuantConfig
from vllm_torchtpu.layers.vllm.quantization.fp8 import (VllmFp8LinearMethodTPU,
                                                        VllmFp8MoEMethodTPU)
from vllm_torchtpu.layers.vllm.quantization.mxfp4 import \
    VllmDeepseekV4Mxfp4MoEMethod
from vllm_torchtpu.layers.vllm.quantization.unquantized import \
    VllmUnquantizedFusedMoEMethod
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@register_quantization_config(DEEPSEEK_V4_FP8)
@register_quantization_config(get_tpu_quant_method(DEEPSEEK_V4_FP8))
class VllmDeepseekV4Fp8Config(DeepseekV4FP8Config, VllmQuantConfig):
    """TPU quantization config for "deepseek_v4_fp8" format.

    Registered under "tpu-deepseek_v4_fp8".
    Dispatches MoE quant methods based on the checkpoint's expert_dtype:
      - "fp4" → VllmDeepseekV4Mxfp4MoEMethod
      - "fp8" → VllmFp8MoEMethodTPU
    Linear layers always use VllmFp8LinearMethodTPU (FP8 block-quant).
    """

    @property
    def is_scale_e8m0(self) -> bool:
        try:
            from vllm.config import get_current_vllm_config
            hf_config = get_current_vllm_config().model_config.hf_config
            quant_cfg = getattr(hf_config, "quantization_config", None) or {}
            if isinstance(quant_cfg,
                          dict) and quant_cfg.get("scale_fmt") == "ue8m0":
                return True
        except Exception:
            pass
        return super().is_scale_e8m0

    @classmethod
    def get_name(cls) -> str:
        return DEEPSEEK_V4_FP8

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[Union[vllm_linear.LinearMethodBase, QuantizeMethodBase]]:
        if isinstance(layer, vllm_linear.LinearBase):
            linear_config = self.get_linear_config(layer)
            if is_layer_skipped(
                    prefix=prefix,
                    ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping,
            ):
                return vllm_linear.UnquantizedLinearMethod()
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
                return VllmFp8MoEMethodTPU(self, layer)

        return None
