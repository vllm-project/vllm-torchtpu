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
"""MiniMax M3 checkpoint quantization: MXFP8 linears, NVFP4 routed experts.

The checkpoint's `quant_method: mxfp8` config also carries the routed experts'
compressed-tensors NVFP4 scheme (`moe_quant_algo`). MXFP8 linears are
dequantized to bf16 at load: an e4m3 value times a power-of-two scale is exact
in bf16, so this keeps the checkpoint's numerics. The experts reuse the TPU
NVFP4 MoE path after renaming the compressed-tensors tensors to ModelOpt's.
"""

from typing import Any

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_torchtpu.layers.adapter.quantization.nvfp4 import (
    VllmNvfp4Config,
    VllmNvfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core.quant_methods import get_tpu_quant_method
from vllm_torchtpu.utils import synchronize_tensors

QUANT_METHOD = "mxfp8"
MXFP8_BLOCK = 32
_E8M0_BIAS = 127
_NVFP4_GROUP = 16
# The TPU MoE kernels hard-code SwiGLU-OAI's (alpha, limit, beta).
_KERNEL_SWIGLU = (1.702, 7.0, 1.0)

# The DSpark draft's quantization config names its modules
# `language_model.model.dspark.<module>`; the draft model builds them as
# `model.<module>`.
_DSPARK_PREFIX = ("language_model.model.dspark.", "model.")

# Compressed-tensors NVFP4 expert tensor suffix -> ModelOpt parameter suffix.
CT_TO_MODELOPT_EXPERT_SUFFIX = {
    ".weight_packed": ".weight",
    ".weight_global_scale": ".weight_scale_2",
    ".input_global_scale": ".input_scale",
}


def dequantize_mxfp8(weight: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """``weight[n, k] * 2 ** (scale[n, k // 32] - 127)`` in bf16."""
    out_dim, in_dim = weight.shape
    scale = torch.exp2(scale_e8m0.to(torch.float32) - _E8M0_BIAS)
    w = weight.to(torch.float32).view(out_dim, in_dim // MXFP8_BLOCK, MXFP8_BLOCK)
    return (w * scale.unsqueeze(-1)).view(out_dim, in_dim).to(torch.bfloat16)


@register_weight_loader_v2_supported_method
class MiniMaxM3Mxfp8LinearMethod(LinearMethodBase):
    """Loads e4m3 weights with ``[1, 32]`` e8m0 scales, runs in bf16."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        if input_size_per_partition % MXFP8_BLOCK:
            raise ValueError(
                f"MXFP8 needs the per-rank input size ({input_size_per_partition}) "
                f"to be a multiple of {MXFP8_BLOCK}; use a smaller tensor parallel "
                "size."
            )
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs["weight_loader"]
        layer.logical_widths = output_partition_sizes
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=torch.float8_e4m3fn,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // MXFP8_BLOCK,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = dequantize_mxfp8(layer.weight.data, layer.weight_scale.data)
        del layer.weight_scale
        replace_parameter(
            layer, "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        VllmUnquantizedLinearMethod().process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return VllmUnquantizedLinearMethod().apply(layer, x, bias)


class MiniMaxM3Nvfp4MoEMethod(VllmNvfp4MoEMethod):
    """Compressed-tensors NVFP4 experts on the TPU NVFP4 MoE path.

    The model loader renames the tensors with `CT_TO_MODELOPT_EXPERT_SUFFIX`.
    Compressed-tensors stores the reciprocal of ModelOpt's per-tensor weight
    scale, so it is inverted before the shared preparation runs.
    """

    def _resolve_tpu_activation(self, layer: torch.nn.Module) -> str:
        if layer.activation.value != "swigluoai_uninterleave":
            raise NotImplementedError(
                f"MiniMax M3 experts use swigluoai_uninterleave, got {layer.activation}"
            )
        moe = layer.moe_config
        swiglu = (moe.swiglu_alpha, moe.swiglu_limit, moe.swiglu_beta)
        if swiglu != _KERNEL_SWIGLU:
            raise NotImplementedError(
                f"TPU MoE kernels implement SwiGLU-OAI (alpha, limit, beta)="
                f"{_KERNEL_SWIGLU}, got {swiglu}."
            )
        # The kernels' "swigluoai" reads w13 as [gate half; up half], which is
        # this checkpoint's layout already.
        return "swigluoai"

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
            param = getattr(layer, name)
            param.data = torch.reciprocal(param.data)
        if layer.w13_weight_scale_2.device.type == "tpu":
            synchronize_tensors([layer.w13_weight_scale_2, layer.w2_weight_scale_2])
        super().process_weights_after_loading(layer)


@register_quantization_config(get_tpu_quant_method(QUANT_METHOD))
class MiniMaxM3QuantConfig(VllmNvfp4Config):
    """TPU config for the checkpoint's ``quant_method: mxfp8`` block."""

    @classmethod
    def get_name(cls) -> str:
        return QUANT_METHOD

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MiniMaxM3QuantConfig":
        expected = {
            "weight_block_size": [1, MXFP8_BLOCK],
            "activation_scheme": "dynamic",
            "moe_quant_algo": "NVFP4",
            "moe_quant_format": "nvfp4-pack-quantized",
            "group_size": _NVFP4_GROUP,
        }
        got = {key: config.get(key) for key in expected}
        if got != expected:
            raise NotImplementedError(
                f"MiniMax M3 TPU quantization supports {expected}, got {got}."
            )
        old, new = _DSPARK_PREFIX
        exclude_modules = [
            new + name.removeprefix(old) if name.startswith(old) else name
            for name in config.get("ignored_layers") or []
        ]
        return cls(
            quant_method="NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            exclude_modules=exclude_modules,
            group_size=_NVFP4_GROUP,
        )

    def _get_checkpoint_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        excluded = self.is_layer_excluded(prefix)
        if isinstance(layer, LinearBase):
            if excluded:
                return VllmUnquantizedLinearMethod()
            return MiniMaxM3Mxfp8LinearMethod()
        if isinstance(layer, RoutedExperts):
            if excluded:
                return VllmUnquantizedFusedMoEMethod(self.get_moe_config(layer))
            return MiniMaxM3Nvfp4MoEMethod(self, self.get_moe_config(layer))
        return None
