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
compressed-tensors NVFP4 scheme (`moe_quant_algo`). The target model's MXFP8
linears stay e4m3 with one power-of-two scale per output channel
(`requantize_mxfp8_per_channel`), which halves the weight bytes every decode
step streams; so do the DSpark draft's MLP linears (its fc and heads are
bf16 in the checkpoint). The experts reuse the TPU NVFP4 MoE path
after renaming the compressed-tensors tensors to ModelOpt's.
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
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

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels import fp8_linear
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)
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
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

logger = init_logger(__name__)

# From this many rows the matmul is MXU-bound and XLA's dot runs slower on an
# in-fusion e4m3 -> bf16 convert (o_proj +7%, dense up +16% at 8192 rows,
# measured) than on a bf16 copy materialized first.
_UPCAST_MIN_ROWS = 4096


def _upcast_e4m3(w: jax.Array) -> jax.Array:
    return jax.lax.optimization_barrier(w.astype(jnp.bfloat16))


# Built at import: torch_tpu cannot create a jax_op inside a traced forward.
_materialized_bf16 = pallas.jax_op("pallas::minimax_m3_upcast_e4m3", _upcast_e4m3)
_materialized_bf16.register_fake(lambda w: torch.empty_like(w, dtype=torch.bfloat16))

_fp8_linear = pallas.jax_op("pallas::minimax_m3_fp8_linear", fp8_linear.fp8_linear)
_fp8_linear.register_fake(
    lambda x, w, s: x.new_empty((x.shape[0], w.shape[1]), dtype=x.dtype)
)


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


def requantize_mxfp8_per_channel(
    weight: torch.Tensor, scale_e8m0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """MXFP8 ``weight[n, k]`` (e4m3, e8m0 scale per ``[1, 32]`` block) as e4m3
    ``[k, n]`` times a power-of-two scale per output channel, plus how many
    elements changed value.

    Each block is shifted by a power of two onto its channel's largest block
    scale, so an element is exact unless the shift takes it below e4m3's
    normal range; a matmul against it, rounded once and scaled by a power of
    two, then equals the bf16 dequantized matmul."""
    out_dim, in_dim = weight.shape
    exp = scale_e8m0.to(torch.int32)
    top = exp.amax(dim=1, keepdim=True)
    shift = torch.exp2((exp - top).to(torch.float32))
    w = weight.to(torch.float32).view(out_dim, in_dim // MXFP8_BLOCK, MXFP8_BLOCK)
    w = (w * shift.unsqueeze(-1)).view(out_dim, in_dim)
    w8 = w.to(torch.float8_e4m3fn)
    changed = int((w8.to(torch.float32) != w).sum().item())
    channel_scale = torch.exp2(top.squeeze(1).to(torch.float32) - _E8M0_BIAS)
    return w8.transpose(0, 1).contiguous(), channel_scale, changed


@register_weight_loader_v2_supported_method
class MiniMaxM3Mxfp8LinearMethod(LinearMethodBase):
    """Loads e4m3 weights with ``[1, 32]`` e8m0 scales. With ``keep_fp8`` a
    bias-free linear runs on e4m3 weights with a per-channel power-of-two
    scale (``requantize_mxfp8_per_channel``); otherwise on bf16."""

    def __init__(self, keep_fp8: bool = False) -> None:
        super().__init__()
        self.keep_fp8 = keep_fp8

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

    def _runs_fp8(self, layer: torch.nn.Module) -> bool:
        return (
            self.keep_fp8
            and getattr(layer, "bias", None) is None
            and not getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False)
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not self._runs_fp8(layer):
            weight = dequantize_mxfp8(layer.weight.data, layer.weight_scale.data)
            del layer.weight_scale
            replace_parameter(
                layer, "weight", torch.nn.Parameter(weight, requires_grad=False)
            )
            VllmUnquantizedLinearMethod().process_weights_after_loading(layer)
            return
        weight, scale, changed = requantize_mxfp8_per_channel(
            layer.weight.data, layer.weight_scale.data
        )
        replace_parameter(
            layer, "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        replace_parameter(
            layer,
            "weight_scale",
            torch.nn.Parameter(scale.to(torch.bfloat16), requires_grad=False),
        )
        # apply reads the scale as one [1, N] row (the kernel's layout) on every
        # path, so traces of all token buckets take the same inputs. It is a
        # copy of weight_scale, set only here; weight_scale stays 1-D for the
        # MoE kernels that read the shared expert's.
        layer.weight_scale_row = fresh_buffer(layer.weight_scale.data.view(1, -1))
        layer.fp8_linear_max_rows = 0
        setattr(layer, WEIGHT_FLIPPED_ATTR, True)
        if weight.device.type == "tpu":
            layer.fp8_linear_max_rows = fp8_linear.max_rows(*weight.shape)
            synchronize_tensors(
                [layer.weight, layer.weight_scale, layer.weight_scale_row]
            )
        logger.info_once(
            "MXFP8 linears run on e4m3 weights with per-channel power-of-two scales."
        )
        logger.debug(
            "MXFP8 linear %s kept e4m3: %d of %d elements below e4m3's normal "
            "range after the block shift.",
            str(tuple(weight.shape)),
            changed,
            weight.numel(),
        )

    def vmem_weight(self, layer: torch.nn.Module, rows: int) -> torch.Tensor | None:
        """The weight `apply` hands `fp8_linear` as a VMEM operand at every
        row count up to `rows`, or None when it does not."""
        if layer.weight.dtype != torch.float8_e4m3fn:
            return None
        return layer.weight if 0 < rows <= layer.fp8_linear_max_rows else None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``weight``: a copy of `vmem_weight` already in VMEM (carried there
        by the all-reduce before this call), for the `fp8_linear` path."""
        if layer.weight.dtype != torch.float8_e4m3fn:
            return VllmUnquantizedLinearMethod().apply(layer, x, bias)
        rows = math.prod(x.shape[:-1])
        if x.dtype == torch.bfloat16 and 0 < rows <= layer.fp8_linear_max_rows:
            out = _fp8_linear(
                x.reshape(rows, x.shape[-1]),
                layer.weight if weight is None else weight,
                layer.weight_scale_row,
            )
            return out.view(*x.shape[:-1], out.shape[-1])
        if rows >= _UPCAST_MIN_ROWS and x.dtype == torch.bfloat16:
            weight = _materialized_bf16(layer.weight)
        else:
            weight = layer.weight.to(x.dtype)
        # The scale is a power of two, so this rounds once, like x @ bf16(W).
        return torch.matmul(x, weight) * layer.weight_scale_row


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

    # Whether bias-free MXFP8 linears stay e4m3 (set by from_config).
    fp8_linears: bool = False
    # The DSpark draft's config (set by from_config): only its MLP linears
    # stay e4m3. At its few rows the in-fusion e4m3 -> bf16 convert bounds
    # the matmul, so the small qkv/o_proj ran slower in e4m3 (1.9 -> 3.5,
    # 1.5 -> 4.3 us) while gate_up/down ran 1.3x/2.3x faster (c4 b1 twin
    # j-7e996ad9 vs j-44c4235f).
    is_draft: bool = False

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
        ignored = config.get("ignored_layers") or []
        exclude_modules = [
            new + name.removeprefix(old) if name.startswith(old) else name
            for name in ignored
        ]
        quant_config = cls(
            quant_method="NVFP4",
            is_checkpoint_nvfp4_serialized=True,
            exclude_modules=exclude_modules,
            group_size=_NVFP4_GROUP,
        )
        # The draft checkpoint names its modules under the dspark prefix.
        quant_config.is_draft = any(name.startswith(old) for name in ignored)
        quant_config.fp8_linears = envs.TPU_MINIMAX_M3_FP8_LINEARS and (
            envs.TPU_MINIMAX_M3_FP8_DRAFT_LINEARS or not quant_config.is_draft
        )
        return quant_config

    def _get_checkpoint_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        excluded = self.is_layer_excluded(prefix)
        if isinstance(layer, LinearBase):
            if excluded:
                return VllmUnquantizedLinearMethod()
            keep_fp8 = self.fp8_linears and (not self.is_draft or ".mlp." in prefix)
            return MiniMaxM3Mxfp8LinearMethod(keep_fp8)
        if isinstance(layer, RoutedExperts):
            if excluded:
                return VllmUnquantizedFusedMoEMethod(self.get_moe_config(layer))
            return MiniMaxM3Nvfp4MoEMethod(self, self.get_moe_config(layer))
        return None
