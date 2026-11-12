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

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import (
    VllmNvfp4Config,
    VllmNvfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core.quant_methods import get_tpu_quant_method
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.kernels.mxfp8_w8a16 import (
    mxfp8_w8a16_linear,
    mxfp8_w8a16_op,
    supports_shape,
)
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

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


def mxfp8_to_w8a16(
    weight: torch.Tensor, scale_e8m0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[n, k]`` e4m3 + ``[n, k // 32]`` e8m0 -> K-major ``[k, n]`` e4m3 and the
    ``[k // 32, n]`` bf16 power-of-two block scale.

    ``w[k, n] * scale[k // 32, n]`` in bf16 is bit-identical to
    `dequantize_mxfp8`'s weight: an e4m3 value times ``2 ** j`` is exact.
    """
    w_kn = weight.view(torch.uint8).t().contiguous().view(torch.float8_e4m3fn)
    scale = torch.exp2(scale_e8m0.to(torch.float32) - _E8M0_BIAS)
    return w_kn, scale.t().contiguous().to(torch.bfloat16)


@register_weight_loader_v2_supported_method
class MiniMaxM3Mxfp8W8A16LinearMethod(MiniMaxM3Mxfp8LinearMethod):
    """bf16 weight (as `MiniMaxM3Mxfp8LinearMethod`) plus an e4m3 copy.

    Token buckets of at most ``max_rows`` rows run `mxfp8_w8a16_matmul` on the
    e4m3 copy: the kernel dequantizes each chunk to the same bf16 weight in
    VMEM, so decode reads half the weight bytes. Larger buckets (prefill) keep
    the bf16 matmul unchanged.
    """

    _logged = False

    def __init__(self, max_rows: int):
        super().__init__()
        self.max_rows = max_rows

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(layer, "weight_w8"):
            return  # reload paths re-enter this hook
        n, k = layer.weight.shape
        w8 = scale = None
        if supports_shape(k, n):
            w8, scale = mxfp8_to_w8a16(layer.weight.data, layer.weight_scale.data)
        super().process_weights_after_loading(layer)
        if w8 is None or not getattr(layer, WEIGHT_FLIPPED_ATTR, False):
            return  # the bf16 weight must be K-major [k, n] like the copy
        # Build the torch op outside the compiled forward (its cache lock
        # cannot be traced).
        mxfp8_w8a16_op(self.max_rows)
        layer.register_parameter(
            "weight_w8", torch.nn.Parameter(w8.to(layer.weight.device), requires_grad=False)
        )
        layer.register_parameter(
            "weight_w8_scale",
            torch.nn.Parameter(scale.to(layer.weight.device), requires_grad=False),
        )
        if layer.weight.device.type == "tpu":
            synchronize_tensors([layer.weight_w8, layer.weight_w8_scale])
        if not MiniMaxM3Mxfp8W8A16LinearMethod._logged:
            MiniMaxM3Mxfp8W8A16LinearMethod._logged = True
            logger.info(
                "TPU_MINIMAX_M3_MXFP8_W8A16_MAX_ROWS=%d: MXFP8 dense linears add an "
                "e4m3 copy (W8A16 kernel for buckets <= %d rows); first weight "
                "[k, n]=%s",
                self.max_rows,
                self.max_rows,
                [k, n],
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None or not hasattr(layer, "weight_w8"):
            return super().apply(layer, x, bias)
        return mxfp8_w8a16_linear(
            x, layer.weight, layer.weight_w8, layer.weight_w8_scale, self.max_rows
        )


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
        quant_config = cls(
            quant_method="NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            exclude_modules=exclude_modules,
            group_size=_NVFP4_GROUP,
        )
        # The DSpark draft's config names its modules under the draft prefix;
        # the draft reads its qkv_proj weight raw, so it stays bf16.
        is_draft = any(
            name.startswith(old) for name in config.get("ignored_layers") or []
        )
        quant_config.w8a16_max_rows = (
            0 if is_draft else envs.TPU_MINIMAX_M3_MXFP8_W8A16_MAX_ROWS
        )
        return quant_config

    def _get_checkpoint_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        excluded = self.is_layer_excluded(prefix)
        if isinstance(layer, LinearBase):
            if excluded:
                return VllmUnquantizedLinearMethod()
            max_rows = getattr(self, "w8a16_max_rows", 0)
            skip = [s for s in envs.TPU_MINIMAX_M3_MXFP8_W8A16_SKIP.split(",") if s]
            if max_rows and not any(s in prefix for s in skip):
                return MiniMaxM3Mxfp8W8A16LinearMethod(max_rows)
            return MiniMaxM3Mxfp8LinearMethod()
        if isinstance(layer, RoutedExperts):
            if excluded:
                return VllmUnquantizedFusedMoEMethod(self.get_moe_config(layer))
            return MiniMaxM3Nvfp4MoEMethod(self, self.get_moe_config(layer))
        return None
