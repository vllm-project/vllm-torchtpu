# Copyright 2025 Google LLC
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
"""
FP8 quantization support for TPU.

This module provides TPU-compatible FP8 quantization for MoE models like
Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8. The target model uses FP8 e4m3fn
weights with block-wise scales (weight_block_size: [128, 128]).

== MoE Weights ==

The checkpoint stores FP8 weights with 2D block scales. At load time we
dequantize those weights, then requantize them into the runtime format used by
the TPU GMM kernel. By default the runtime rhs uses per-channel scales over the
contracting dimension.

== Linear Weights (attention projections) ==

The checkpoint stores FP8 weights with either block scales or tensor/channel
scales. At load time we dequantize those weights to float32, then requantize
them into the runtime FP8 layout consumed by the Pallas matmul bridge in
`vllm_torchtpu/layers/adapter/linear_common.py`. Activations are quantized
dynamically inside the bridge. By default the bridge dispatches to a pure-JAX
`dot_general`-based per-channel FP8 matmul (`xla_quantized_matmul`); setting
ENABLE_QUANTIZED_MATMUL_KERNEL=1 together with REQUANTIZE_BLOCK_SIZE selects
the blockwise Pallas kernel path instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.layers.quantization import fp8 as vllm_fp8
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
    Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import ChannelQuantScaleParameter
from vllm.model_executor.utils import replace_parameter, set_weight_attrs

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import moe_routing, token_padding
from vllm_torchtpu.layers.adapter.fused_moe import (
    TpuMoEActivationMixin,
    fused_moe_gmm,
    prebuild_fused_moe_kernel,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
    quantized_matmul,
)
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute,
    pipelined_fused_moe_gmm,
)
from vllm_torchtpu.layers.adapter.quantization.configs import (
    VllmQuantConfig,
    VllmQuantLinearConfig,
)
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import (
    OnlineFp8Policy,
    map_online_fp8,
    quantize_online_fp8,
)
from vllm_torchtpu.layers.core.quant_methods import FP8, get_tpu_quant_method
from vllm_torchtpu.layers.core.quantization import dequantize_tensor, quantize_tensor
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4Config

logger = init_logger(__name__)

_MOE_REQUANT_WEIGHT_DTYPES = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "int8": torch.int8,
}
if hasattr(torch, "float8_e5m2"):
    _MOE_REQUANT_WEIGHT_DTYPES["float8_e5m2"] = torch.float8_e5m2

LinearQuantConfig = tuple[str, torch.dtype, int | None, bool]


def _dequantize_fp8_linear(
    weight: torch.Tensor,
    *,
    weight_scale: torch.Tensor | None,
    weight_scale_inv: torch.Tensor | None,
    block_quant: bool,
    weight_block_size: tuple[int, int] | None,
    logical_widths: list[int] | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 linear weights."""
    weight_f = weight.to(out_dtype)

    if block_quant:
        assert weight_scale_inv is not None
        assert weight_block_size is not None
        block_h, block_w = weight_block_size

        out_dim, in_dim = weight_f.shape
        if out_dim % block_h == 0 and in_dim % block_w == 0:
            weight_f = weight_f.reshape(
                out_dim // block_h, block_h, in_dim // block_w, block_w
            )
            weight_f = weight_f * weight_scale_inv.to(out_dtype).unsqueeze(1).unsqueeze(
                3
            )
            weight_f = weight_f.reshape(out_dim, in_dim)
        else:
            scales = (
                weight_scale_inv.to(out_dtype)
                .repeat_interleave(block_h, dim=0)
                .repeat_interleave(block_w, dim=1)
            )
            weight_f = weight_f * scales[:out_dim, :in_dim]
    else:
        assert weight_scale is not None
        scales = weight_scale.to(out_dtype)
        if (
            scales.numel() > 1
            and logical_widths is not None
            and scales.numel() == len(logical_widths)
        ):
            scales = (
                scales.cpu()
                .repeat_interleave(torch.tensor(logical_widths))
                .unsqueeze(1)
                .to(scales.device)
            )
        weight_f = weight_f * scales

    return weight_f


def _get_moe_quant_config(
    log_prefix: str,
) -> tuple[str, torch.dtype, int | None]:
    """Parse MoE quantization env vars and return (dtype_name, dtype, block_size)."""
    if desired_quant_dtype_from_env := envs.MOE_REQUANTIZE_WEIGHT_DTYPE:
        desired_quant_dtype = desired_quant_dtype_from_env
    else:
        desired_quant_dtype = "float8_e4m3fn"

    requant_block_size = None
    if requant_block_size_from_env := envs.MOE_REQUANTIZE_BLOCK_SIZE:
        requant_block_size = (
            int(requant_block_size_from_env) if requant_block_size_from_env else None
        )

    requant_dtype = _MOE_REQUANT_WEIGHT_DTYPES.get(desired_quant_dtype)
    if requant_dtype is None:
        supported = ", ".join(sorted(_MOE_REQUANT_WEIGHT_DTYPES))
        raise ValueError(
            "Unsupported MOE_REQUANTIZE_WEIGHT_DTYPE="
            f"{desired_quant_dtype!r}. Supported values: {supported}."
        )

    msg = f"{log_prefix} to {desired_quant_dtype}"
    if requant_block_size is not None:
        msg += f" with block size {requant_block_size}"
    logger.info_once(msg)

    return desired_quant_dtype, requant_dtype, requant_block_size


def _get_linear_quant_config(
    linear_config: VllmQuantLinearConfig | None,
) -> LinearQuantConfig:
    """Parse dense-linear quantization settings from env-gated overrides.

    Reads ENABLE_QUANTIZED_MATMUL_KERNEL, REQUANTIZE_BLOCK_SIZE, and
    REQUANTIZE_WEIGHT_DTYPE (either from `linear_config` or `envs`) and
    validates that the kernel-gate / block-size combination is coherent.
    """
    enable_quantized_matmul_kernel = (
        linear_config.enable_quantized_matmul_kernel
        if linear_config is not None
        else envs.ENABLE_QUANTIZED_MATMUL_KERNEL
    )
    requant_block_size = (
        linear_config.requant_block_size
        if linear_config is not None
        else envs.REQUANTIZE_BLOCK_SIZE
    )
    desired_quant_dtype = (
        linear_config.requant_weight_dtype
        if linear_config is not None
        else envs.REQUANTIZE_WEIGHT_DTYPE
    )

    if enable_quantized_matmul_kernel and not requant_block_size:
        raise ValueError(
            "You should set REQUANTIZE_BLOCK_SIZE to enable quantized matmul "
            "kernel. Please set the value or disable the quantized matmul "
            "kernel."
        )
    if not enable_quantized_matmul_kernel and requant_block_size:
        raise ValueError(
            "Blockwise quantization is supported by quantized matmul kernel. "
            "Please enable quantized_matmul_kernel or unset the quantize "
            "block size to trigger XLA per-channel quantization."
        )

    requant_dtype = _MOE_REQUANT_WEIGHT_DTYPES.get(desired_quant_dtype)
    if requant_dtype is None:
        supported = ", ".join(sorted(_MOE_REQUANT_WEIGHT_DTYPES))
        raise ValueError(
            "Unsupported REQUANTIZE_WEIGHT_DTYPE="
            f"{desired_quant_dtype!r}. Supported values: {supported}."
        )

    return (
        desired_quant_dtype,
        requant_dtype,
        requant_block_size,
        enable_quantized_matmul_kernel,
    )


def _format_linear_scale_for_runtime(
    weight_scale: torch.Tensor,
    *,
    blockwise_kernel: bool,
) -> torch.Tensor:
    """Reshape `weight_scale` to the layout expected by the matmul bridge.

    - Blockwise path: gmm_v2 expects `[1, n_in_blocks, 1, n_out]` (its
      `[size_group, num_blocks, 1, out_size]` rhs_scale, single group). The
      requantizer produces `[n_out, n_in_blocks]`, so transpose and add both
      the singleton middle axis and the leading group axis.
    - XLA per-channel path: kernel expects a 1-D `[n_out]` scale, so drop
      a trailing singleton dim if present. Layout independent -- the scale is
      per output channel either way.
    """
    weight_scale = weight_scale.to(torch.float32).contiguous()
    if blockwise_kernel:
        # quantize_tensor returns [n_out, n_blocks]; gmm_v2 wants
        # [1, n_blocks, 1, n_out].
        weight_scale = (
            weight_scale.transpose(0, 1).contiguous().unsqueeze(1).unsqueeze(0)
        )
    elif weight_scale.ndim == 2 and weight_scale.shape[-1] == 1:
        weight_scale = weight_scale.squeeze(-1).contiguous()
    return weight_scale


def _process_fp8_linear_weights(
    layer: torch.nn.Module,
    *,
    block_quant: bool,
    weight_block_size: tuple[int, int] | None,
    linear_config: VllmQuantLinearConfig | None,
    linear_quant_config: LinearQuantConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str, int | None]:
    """Convert checkpoint FP8 dense linear weights to the runtime FP8 layout.

    Dequantizes the checkpoint weight to float32 (collapsing per-block or
    per-tensor scales), then requantizes to the runtime dtype/block-size
    chosen by the env gates, and formats the scale tensor to the shape the
    matmul bridge expects.
    """
    if linear_quant_config is None:
        linear_quant_config = _get_linear_quant_config(linear_config)
    desired_quant_dtype, requant_dtype, requant_block_size, blockwise_kernel = (
        linear_quant_config
    )

    weight_scale = layer.weight_scale.data if hasattr(layer, "weight_scale") else None
    weight_scale_inv = (
        layer.weight_scale_inv.data if hasattr(layer, "weight_scale_inv") else None
    )

    weight_f = _dequantize_fp8_linear(
        layer.weight.data,
        weight_scale=weight_scale,
        weight_scale_inv=weight_scale_inv,
        block_quant=block_quant,
        weight_block_size=weight_block_size,
        logical_widths=layer.logical_widths,
        out_dtype=torch.float32,
    )
    weight, weight_scale = quantize_tensor(
        weight_f,
        quant_dtype=requant_dtype,
        axis=-1,
        block_size=requant_block_size,
    )
    weight_scale = _format_linear_scale_for_runtime(
        weight_scale,
        blockwise_kernel=blockwise_kernel,
    )
    return weight.contiguous(), weight_scale, desired_quant_dtype, requant_block_size


def _quantize_and_format_single_moe_weight(
    w: torch.Tensor,
    *,
    name: str,
    quant_dtype: torch.dtype,
    block_size: int | None,
    activation: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float32 MoE weight tensor to FP8 and format for the GMM kernel.

    Handles swigluoai deinterleaving (for w13), 128-alignment padding,
    block alignment validation, quantization, transpose, and scale reshaping.
    """
    if activation == "swigluoai" and name == "w13":
        w1 = w[:, ::2, :]
        w3 = w[:, 1::2, :]
        w = torch.cat([w1, w3], dim=1)

    if name == "w13":
        half = w.shape[1] // 2
        aligned_half = (half + 127) // 128 * 128
        if aligned_half != half:
            pad_w = w.new_zeros(w.shape[0], aligned_half - half, w.shape[2])
            w = torch.cat(
                [w[:, :half, :], pad_w, w[:, half:, :], pad_w], dim=1
            ).contiguous()
    elif name == "w2":
        inter = w.shape[2]
        aligned_inter = (inter + 127) // 128 * 128
        if aligned_inter != inter:
            pad_w = w.new_zeros(w.shape[0], w.shape[1], aligned_inter - inter)
            w = torch.cat([w, pad_w], dim=2).contiguous()

    if block_size is not None and w.shape[-1] % block_size != 0:
        raise ValueError(
            f"Unsupported MoE quantization: {name} contracting "
            f"dimension {w.shape[-1]} is not divisible by "
            f"block_size={block_size}."
        )

    w, w_scale = quantize_tensor(
        w, quant_dtype=quant_dtype, axis=-1, block_size=block_size
    )

    w = w.transpose(1, 2).contiguous()

    # Reshape scales for GMM kernel.
    # An `unsqueeze(2)` view on a TPU tensor leaves the persistent
    # nn.Parameter carrying ambiguous strides (stride[1] == stride[2]) that
    # torch_tpu's per-op JIT re-emits as a per-step `tt_jit_as_strided`
    # program at the GMM `pallas.jax_op` boundary. Allocate a fresh 4D
    # contiguous device buffer with `torch.empty + .copy_` to break the
    # view chain so the persistent storage is a plain 4D buffer that PJRT
    # ships to the kernel with row-major layout.
    if w_scale is not None:
        s = w_scale.transpose(1, 2).contiguous().to(torch.float32)
        w_scale = torch.empty(
            s.shape[0], s.shape[1], 1, s.shape[2], dtype=s.dtype, device=s.device
        ).copy_(s.unsqueeze(2))

    return w, w_scale


# Bytes of the float32 copy one requantization chunk dequantizes at a time.
# The quantized chunk, its scales and the formatting temporaries come on top.
_REQUANT_CHUNK_BYTES = 1 << 30


def _requantize_expert_chunks(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    dequantize: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    zero_padding: Callable[[torch.Tensor], None] | None,
    **format_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dequantize, requantize and format ``weight`` a few experts at a time.

    The float32 copy of a whole expert tensor can be several times the
    device memory left after loading (one layer of a 160-expert model with
    no expert parallelism is 4.7 GB in FP8), so every chunk is materialized
    before the next one starts and only the quantized results are
    concatenated.
    One concatenation joins the chunks rather than slice writes into a
    preallocated tensor: on TPU an in-place slice write is a program that
    rewrites the whole tensor, once per chunk.
    """
    num_experts = weight.shape[0]
    per_expert = weight[0].numel() * 4
    per_chunk = max(1, _REQUANT_CHUNK_BYTES // per_expert)
    weights, scales = [], []
    for start in range(0, num_experts, per_chunk):
        end = min(start + per_chunk, num_experts)
        fp32 = dequantize(
            weight[start:end], None if scale is None else scale[start:end]
        )
        if zero_padding is not None:
            zero_padding(fp32)
        w, w_scale = _quantize_and_format_single_moe_weight(fp32, **format_kwargs)
        del fp32
        if per_chunk < num_experts and w.device.type == "tpu":
            synchronize_tensors([w] if w_scale is None else [w, w_scale])
        weights.append(w)
        scales.append(w_scale)
    if len(weights) == 1:
        return weights[0], scales[0]
    w = torch.cat(weights, dim=0)
    w_scale = None if scales[0] is None else torch.cat(scales, dim=0)
    return w, w_scale


def _quantize_and_format_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    quant_dtype: torch.dtype,
    block_size: int | None,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize float32 MoE weights to FP8 and format for the GMM kernel.

    Handles block alignment validation, quantization, swigluoai
    deinterleaving, transpose, and scale reshaping.
    """
    w13, w13_scale = _quantize_and_format_single_moe_weight(
        w13,
        name="w13",
        quant_dtype=quant_dtype,
        block_size=block_size,
        activation=activation,
    )
    w2, w2_scale = _quantize_and_format_single_moe_weight(
        w2,
        name="w2",
        quant_dtype=quant_dtype,
        block_size=block_size,
        activation=activation,
    )
    return w13, w13_scale, w2, w2_scale


def _process_fp8_moe_weights(
    layer: RoutedExperts,
    *,
    weight_block_size: tuple[int, int] | None = None,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int | None]:
    """Dequantize and requantize FP8 MoE weights for the TPU GMM kernel.

    Returns (w13_weight, w13_scale, w2_weight, w2_scale,
             requant_dtype_name, requant_block_size).
    """
    desired_quant_dtype, requant_dtype, requant_block_size = _get_moe_quant_config(
        "[MoE requantization]: re-quantizing MoE weights"
    )

    # Dequantize from checkpoint FP8 to float32
    w13_scale_param = getattr(
        layer, "w13_weight_scale_inv", getattr(layer, "w13_weight_scale", None)
    )
    w2_scale_param = getattr(
        layer, "w2_weight_scale_inv", getattr(layer, "w2_weight_scale", None)
    )
    if w13_scale_param is None or w2_scale_param is None:
        raise ValueError(
            "Missing MoE weight scale parameters (expected w13_weight_scale_inv or w13_weight_scale)"
        )

    padded_intermediate = layer.moe_config.intermediate_size_per_partition
    unpadded_intermediate = layer.moe_config.intermediate_size_per_partition_unpadded
    needs_padding = padded_intermediate != unpadded_intermediate
    if needs_padding:
        full_intermediate = unpadded_intermediate * layer.moe_config.tp_size
        local_start = padded_intermediate * layer.moe_config.tp_rank
        local_real = max(0, min(padded_intermediate, full_intermediate - local_start))

    if weight_block_size is not None:
        # Validate block alignment
        block_h, block_w = weight_block_size
        for name, weight in (
            ("w13_weight", layer.w13_weight.data),
            ("w2_weight", layer.w2_weight.data),
        ):
            _, out_dim, in_dim = weight.shape
            if out_dim % block_h != 0 or in_dim % block_w != 0:
                raise ValueError(
                    "FP8 block quantized MoE weights must be divisible by the checkpoint "
                    f"block size, got {name}.shape={tuple(weight.shape)} and "
                    f"block_size={weight_block_size}."
                )

    zero_w13 = zero_w2 = None
    if needs_padding and local_real < padded_intermediate:

        def zero_w13(w13_fp32: torch.Tensor) -> None:
            w13_fp32[:, local_real:padded_intermediate, :] = 0
            w13_fp32[
                :, padded_intermediate + local_real : 2 * padded_intermediate, :
            ] = 0

        def zero_w2(w2_fp32: torch.Tensor) -> None:
            w2_fp32[:, :, local_real:padded_intermediate] = 0

    if weight_block_size is not None:

        def dequantize_w13(w_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            return dequantize_tensor(w_q, scale, axis=(1, 2), out_dtype=torch.float32)

        dequantize_w2 = dequantize_w13
    else:

        def dequantize_w13(w_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            w_q = w_q.to(torch.float32)
            s13 = scale.to(torch.float32)
            if s13.ndim == 2 and s13.shape[1] == 2:
                half = w_q.shape[1] // 2
                s13 = s13.repeat_interleave(half, dim=1).unsqueeze(-1)
            elif s13.ndim == 2 and s13.shape[1] == w_q.shape[1]:
                s13 = s13.unsqueeze(-1)
            elif s13.ndim == 1:
                s13 = s13.view(-1, 1, 1)
            return w_q * s13

        def dequantize_w2(w_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            w_q = w_q.to(torch.float32)
            s2 = scale.to(torch.float32)
            if s2.ndim == 2 and s2.shape[1] == w_q.shape[1]:
                s2 = s2.unsqueeze(-1)
            elif s2.ndim == 1:
                s2 = s2.view(-1, 1, 1)
            return w_q * s2

    w13, w13_scale = _requantize_expert_chunks(
        layer.w13_weight.data,
        w13_scale_param.data,
        dequantize_w13,
        zero_w13,
        name="w13",
        quant_dtype=requant_dtype,
        block_size=requant_block_size,
        activation=activation,
    )
    w2, w2_scale = _requantize_expert_chunks(
        layer.w2_weight.data,
        w2_scale_param.data,
        dequantize_w2,
        zero_w2,
        name="w2",
        quant_dtype=requant_dtype,
        block_size=requant_block_size,
        activation=activation,
    )

    return w13, w13_scale, w2, w2_scale, desired_quant_dtype, requant_block_size


def _quantize_bf16_moe_weights(
    layer: RoutedExperts,
    *,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int | None]:
    """Quantize BF16 MoE weights to FP8 for the TPU GMM kernel.

    Same output format as _process_fp8_moe_weights but starts from BF16
    weights instead of dequantizing from FP8 first.
    """
    desired_quant_dtype, requant_dtype, requant_block_size = _get_moe_quant_config(
        "[MoE online FP8]: quantizing BF16 MoE weights"
    )

    def to_fp32(w: torch.Tensor, _scale: torch.Tensor | None) -> torch.Tensor:
        return w.to(torch.float32)

    w13, w13_scale = _requantize_expert_chunks(
        layer.w13_weight.data,
        None,
        to_fp32,
        None,
        name="w13",
        quant_dtype=requant_dtype,
        block_size=requant_block_size,
        activation=activation,
    )
    w2, w2_scale = _requantize_expert_chunks(
        layer.w2_weight.data,
        None,
        to_fp32,
        None,
        name="w2",
        quant_dtype=requant_dtype,
        block_size=requant_block_size,
        activation=activation,
    )

    return w13, w13_scale, w2, w2_scale, desired_quant_dtype, requant_block_size


@register_quantization_config(get_tpu_quant_method(FP8))
class VllmFp8Config(Fp8Config, VllmQuantConfig):
    """Native "fp8" quant_method config.

    Also constructed directly (bypassing from_config) by
    VllmCompressedTensorsConfig in compressed_tensors.py, which adapts a
    matched compressed-tensors FP8 scheme into an instance of this class to
    reuse its dequant/requant runtime path.
    """

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        store_dtype: str | None = None,
    ) -> None:
        if store_dtype is not None:
            raise NotImplementedError(
                "FP8 store_dtype is not supported by the TPU quantization path"
            )
        super().__init__(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            store_dtype=store_dtype,
        )

    @classmethod
    def get_name(cls) -> str:
        return FP8

    @classmethod
    def from_config(cls, config: dict) -> VllmFp8Config:
        weight_block_size = config.get("weight_block_size")
        activation_scheme = config.get("activation_scheme", "dynamic")
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        store_dtype = cls.get_from_keys_or(config, ["store_dtype"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=activation_scheme,
            weight_block_size=weight_block_size,
            ignored_layers=ignored_layers,
            store_dtype=store_dtype,
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        map_online_fp8(self, hf_to_vllm_mapper)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        base_method = self._get_checkpoint_quant_method(layer, prefix)
        return resolve_online_fp8(self, layer, prefix, base_method)

    def _get_checkpoint_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                from vllm_torchtpu.layers.adapter.quantization.unquantized import (
                    VllmUnquantizedFusedMoEMethod,
                )

                moe_config = self.get_moe_config(layer)
                return VllmUnquantizedFusedMoEMethod(moe_config)
            moe_config = self.get_moe_config(layer)
            return VllmFp8MoEMethodTPU(self, moe_config)

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                from vllm_torchtpu.layers.adapter.quantization.unquantized import (
                    VllmUnquantizedLinearMethod,
                )

                return VllmUnquantizedLinearMethod()
            return VllmFp8LinearMethodTPU(self, self.get_linear_config(layer))

        if isinstance(layer, Attention):
            return None

        return None


class VllmFp8MoEMethodTPU(TpuMoEActivationMixin, Fp8MoEMethod):
    """
    TPU-native FP8 MoE method.

    Supports both FP8-serialized checkpoints (dequant → requant) and BF16
    checkpoints (online quantization to FP8 at load time).

    Uses is_monolithic=True so vLLM's DefaultMoERunner calls
    apply_monolithic(layer, x, router_logits) directly, bypassing the
    CUDA-only router.select_experts() path.
    """

    def __init__(self, quant_config: Fp8Config, moe_config):
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )
        # Native allocation/loading reads these states in vLLM 0.29. TPU
        # pads to the checkpoint block grid instead of refining GPU scales.
        self.moe_block_shape = self.weight_block_size
        self.weight_scale_refine = None
        self.fp8_backend = None

    @property
    def supports_internal_mk(self) -> bool:
        """Whether this method owns the DP/EP dispatch and combine.

        vLLM brackets `apply_monolithic` with an all-gather and a
        reduce-scatter unless the method claims them
        (`MoERunner.do_naive_dispatch_combine`). Two independent paths claim
        them, and either one alone has to suppress vLLM's pair:

        - chunk pipelining stages the same two collectives itself so they
          overlap the compute;
        - the fused EP kernel does the exchange inside its own program.

        The fused half asks `fused_moe_ep_supported` about this method, not
        `envs.USE_MOE_FUSED_EP_KERNEL`: the env var only says the operator
        asked for the kernel, while `prebuild_fused_moe_ep` is what decides
        whether this layer was actually armed. Claiming ownership on the
        request would strip vLLM's collectives in every configuration prebuild
        refuses -- TP, over the SMEM bound, under the token threshold, or
        a weight or routing layout the kernel cannot serve.
        """
        from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported

        return enable_pipelined_collective_and_compute() or fused_moe_ep_supported(self)

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config,
    ) -> tuple[int, int]:
        hidden_size, intermediate_size_per_partition = Fp8MoEMethod.maybe_roundup_sizes(
            self,
            hidden_size,
            intermediate_size_per_partition,
            act_dtype,
            moe_parallel_config,
        )
        if self.quant_config.is_checkpoint_fp8_serialized and self.block_quant:
            assert self.weight_block_size is not None
            block_n, block_k = self.weight_block_size
            block_size = max(block_n, block_k)
            intermediate_size_per_partition = (
                (intermediate_size_per_partition + block_size - 1)
                // block_size
                * block_size
            )
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if not hasattr(layer, "has_bias") and hasattr(layer, "moe_config"):
            layer.has_bias = layer.moe_config.has_bias
        if self.quant_config.is_checkpoint_fp8_serialized:
            Fp8MoEMethod.create_weights(
                self,
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )
            if (
                getattr(self.quant_config, "is_channel_quant", False)
                and not self.block_quant
            ):
                # register_parameter() safely replaces the base class's
                # differently-shaped w13_weight_scale/w2_weight_scale below;
                # no need to delete them first.
                w13_scale_data = torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    1,
                    dtype=torch.float32,
                )
                w2_scale_data = torch.ones(
                    num_experts,
                    hidden_size,
                    1,
                    dtype=torch.float32,
                )
                w13_weight_scale = torch.nn.Parameter(
                    w13_scale_data, requires_grad=False
                )
                w2_weight_scale = torch.nn.Parameter(w2_scale_data, requires_grad=False)
                layer.register_parameter(
                    f"w13_{self.weight_scale_name}", w13_weight_scale
                )
                layer.register_parameter(
                    f"w2_{self.weight_scale_name}", w2_weight_scale
                )
                from vllm.model_executor.layers.fused_moe import (
                    FusedMoeWeightScaleSupported,
                )

                attrs = dict(extra_weight_attrs)
                attrs["quant_method"] = FusedMoeWeightScaleSupported.CHANNEL.value
                set_weight_attrs(w13_weight_scale, attrs)
                set_weight_attrs(w2_weight_scale, attrs)

            for param in [
                getattr(layer, "w13_weight", None),
                getattr(layer, "w2_weight", None),
                getattr(layer, "w13_weight_scale", None),
                getattr(layer, "w2_weight_scale", None),
                getattr(layer, "w13_weight_scale_inv", None),
                getattr(layer, "w2_weight_scale_inv", None),
            ]:
                if (
                    param is not None
                    and getattr(param, "weight_loader", None) is None
                    and hasattr(layer, "weight_loader")
                ):
                    param.weight_loader = layer.weight_loader
        else:
            from vllm_torchtpu.layers.adapter.quantization.unquantized import (
                VllmUnquantizedFusedMoEMethod,
            )

            VllmUnquantizedFusedMoEMethod.create_weights(
                self,
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )

    @property
    def is_monolithic(self) -> bool:
        return True

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Process MoE weights into the runtime FP8 layout used by the GMM kernel.

        For FP8 checkpoints: dequantize block FP8 → float32, then requantize
        to the GMM kernel's expected per-channel FP8 format.

        For BF16 checkpoints (online quantization): quantize BF16 → FP8
        directly using the same requantization path.
        """
        assert isinstance(layer, RoutedExperts)
        assert not self.moe.has_bias, "TPU FP8 MoE path does not support bias."

        activation_str = self._set_tpu_activation(layer)

        is_fp8_serialized = self.quant_config.is_checkpoint_fp8_serialized
        if is_fp8_serialized:
            weight_block_size = (
                tuple(self.weight_block_size)
                if self.weight_block_size is not None
                else None
            )
            (w13, w13_scale, w2, w2_scale, requant_dtype_name, requant_block_size) = (
                _process_fp8_moe_weights(
                    layer,
                    weight_block_size=weight_block_size,
                    activation=activation_str,
                )
            )
        else:
            (w13, w13_scale, w2, w2_scale, requant_dtype_name, requant_block_size) = (
                _quantize_bf16_moe_weights(
                    layer,
                    activation=activation_str,
                )
            )

        # Remove placeholder checkpoint scales before assigning runtime inverse scales
        # to prevent nn.Module parameter re-assignment errors and free state dict memory.
        # Note: RoutedExperts universally names its parameters w13 (gate+up) and w2 (down);
        # hasattr is True for serialized FP8 checkpoints and False for unquantized BF16.
        if hasattr(layer, f"w13_{self.weight_scale_name}"):
            delattr(layer, f"w13_{self.weight_scale_name}")
        if hasattr(layer, f"w2_{self.weight_scale_name}"):
            delattr(layer, f"w2_{self.weight_scale_name}")

        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale_inv = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale_inv = torch.nn.Parameter(w2_scale, requires_grad=False)

        # Eagerly materialize weights to avoid OOM during vLLM memory probing.
        # Without this, lazy tensors accumulate and the profiling run OOMs.
        if layer.w13_weight.device.type == "tpu":
            synchronize_tensors(
                [
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale_inv,
                    layer.w2_weight_scale_inv,
                ]
            )

        scale_desc = (
            "per-channel" if requant_block_size is None else str(requant_block_size)
        )
        logger.info_once(
            "FP8 weights transposed for GMM kernel: "
            f"w13={list(layer.w13_weight.shape)}, "
            f"w2={list(layer.w2_weight.shape)}, "
            f"w13_scale={list(layer.w13_weight_scale_inv.shape)}, "
            f"w2_scale={list(layer.w2_weight_scale_inv.shape)}, "
            f"requant_block_size={scale_desc}"
        )
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_routing.validate_linear_ep_placement(layer)
        moe_routing.register_experts_start_buffer(layer, device=layer.w13_weight.device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
        )
        # Resolve the EP mesh here, not on the first forward: it maps ranks to
        # TPU device ids with an all_gather_object, and Dynamo cannot trace a
        # collective inside the compiled graph. The result is recorded on this
        # method, which vLLM builds one of per MoE layer, so a layer the kernel
        # cannot serve is refused on its own rather than for the process.
        from vllm_torchtpu.layers.adapter.fused_moe_ep import (
            FUSED_MOE_EP_OP_ATTR,
            prebuild_fused_moe_ep,
        )

        setattr(
            self,
            FUSED_MOE_EP_OP_ATTR,
            prebuild_fused_moe_ep(
                layer,
                topk=layer.moe_config.experts_per_token,
                renormalize=layer.renormalize,
                activation=activation_str,
            ),
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass using TPU-native GMM kernel with FP8 weights."""
        activation_str = self._tpu_activation_str
        assert activation_str is not None, (
            "[moe] process_weights_after_loading did not run for this layer"
        )
        # The fused EP kernel takes this rank's own tokens and returns this
        # rank's own tokens: it routes, exchanges, computes and combines inside
        # one program. Keyed on the same predicate `supports_internal_mk` uses,
        # so ownership and the path that exercises it cannot disagree.
        from vllm_torchtpu.layers.adapter.fused_moe_ep import (
            fused_moe_ep,
            fused_moe_ep_supported,
        )

        if fused_moe_ep_supported(self):
            return fused_moe_ep(
                self,
                x,
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale_inv,
                layer.w2_weight_scale_inv,
                router_logits,
            )
        # Step 1: Routing
        # Quantization-independent routing decision (simulation override ->
        # custom_routing_function -> select_experts); shared across all TPU MoE
        # methods so the routing-simulation hook lives in exactly one place.
        topk_weights, topk_ids = moe_routing.route(layer, x, router_logits)

        if envs.TPU_MOE_SKIP_PADDED_TOKENS:
            topk_ids, topk_weights = token_padding.zero_routing_weights_for_padding(
                topk_ids,
                topk_weights,
                is_local_tensor=enable_pipelined_collective_and_compute(),
            )

        # Step 2: EP global->local remap happens inside fused_moe_gmm via an
        # elementwise subtract from `experts_start` (scalar).
        kwargs = {
            "hidden_states": x,
            "w1": layer.w13_weight,
            "w2": layer.w2_weight,
            "w1_scale": layer.w13_weight_scale_inv,
            "w2_scale": layer.w2_weight_scale_inv,
            "w1_bias": getattr(layer, "w13_bias", None),
            "w2_bias": getattr(layer, "w2_bias", None),
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "experts_start": layer._experts_start,
            "topk": layer.moe_config.experts_per_token,
            "activation": activation_str,
        }
        if enable_pipelined_collective_and_compute():
            return pipelined_fused_moe_gmm(**kwargs)

        return fused_moe_gmm(**kwargs)


class ChannelQuantScaleParameterTPU(ChannelQuantScaleParameter):
    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        if loaded_weight.ndim == 1:
            loaded_weight = loaded_weight.view(-1, 1)
        super().load_column_parallel_weight(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs):
        if loaded_weight.ndim == 1:
            loaded_weight = loaded_weight.view(-1, 1)
        super().load_merged_column_weight(loaded_weight, **kwargs)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):
        if loaded_weight.ndim == 1:
            loaded_weight = loaded_weight.view(-1, 1)
        super().load_qkv_weight(loaded_weight, **kwargs)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor, **kwargs):
        if loaded_weight.ndim == 1:
            loaded_weight = loaded_weight.view(-1, 1)
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)


@register_weight_loader_v2_supported_method
class VllmFp8LinearMethodTPU(Fp8LinearMethod):
    """
    TPU FP8 linear method that keeps weights in FP8 at runtime and quantizes
    activations dynamically inside a Pallas matmul bridge.

    Reuses vLLM's Fp8LinearMethod.create_weights() for correct FP8 weight and
    scale parameter allocation (needed by the weight loader). After loading,
    checkpoint FP8 weights are converted to the runtime FP8 layout and applied
    with dynamically quantized activations through `quantized_matmul`.

    Inherits from Fp8LinearMethod but skips GPU-specific __init__. Only
    sets attributes needed by create_weights().
    """

    def __init__(
        self,
        quant_config: Fp8Config,
        linear_config: VllmQuantLinearConfig | None = None,
        prefix: str = "",
    ):
        # Skip Fp8LinearMethod.__init__ which has GPU-specific code
        # (CUDA capability, Marlin, cutlass, W8A8BlockFp8LinearOp).
        # Set only the attributes that create_weights() reads.
        self.prefix = prefix
        self.online_fp8 = False
        self.online_fp8_policy: OnlineFp8Policy | None = None
        self.quant_config = quant_config
        self.weight_block_size = quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = quant_config.activation_scheme == "static"
        # These are needed by create_weights but are GPU-specific.
        # Set safe defaults so the parent's create_weights doesn't crash.
        self.use_marlin = False
        self.use_deep_gemm = False
        self.cutlass_block_fp8_supported = False
        self.is_scale_e8m0 = getattr(quant_config, "is_scale_e8m0", False)
        self.activation_quant_key = None
        self.weight_quant_key = None
        self.input_dtype = torch.get_default_dtype()
        self.out_dtype = torch.get_default_dtype()
        vllm_fp8.init_fp8_linear_kernel = lambda *args, **kwargs: None
        self.linear_config = linear_config

        if quant_config.is_checkpoint_fp8_serialized:
            self._linear_quant_config = _get_linear_quant_config(self.linear_config)
        else:
            # Load-time quantization has its own fixed per-channel contract.
            self._linear_quant_config = (
                "float8_e4m3fn",
                torch.float8_e4m3fn,
                None,
                False,
            )

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if self.online_fp8:
            # Define the per-layer state before the post-load hook can run.
            layer._tpu_online_fp8_processed = False
        if self.quant_config.is_checkpoint_fp8_serialized:
            Fp8LinearMethod.create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )
            if getattr(self.quant_config, "is_channel_quant", False) or getattr(
                self, "is_channel_quant", False
            ):
                weight_loader = extra_weight_attrs.get(
                    "weight_loader", getattr(layer, "weight_loader", None)
                )
                scale = ChannelQuantScaleParameterTPU(
                    data=torch.empty(
                        (sum(output_partition_sizes), 1), dtype=torch.float32
                    ),
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)
            if hasattr(layer, "weight_scale_inv") and not hasattr(
                layer, "weight_scale"
            ):
                from vllm.model_executor.parameter import BlockQuantScaleParameter

                scale_inv = layer.weight_scale_inv
                weight_scale = BlockQuantScaleParameter(
                    data=scale_inv.data,
                    input_dim=getattr(scale_inv, "input_dim", 1),
                    output_dim=getattr(scale_inv, "output_dim", 0),
                    weight_loader=getattr(
                        scale_inv,
                        "weight_loader",
                        getattr(layer, "weight_loader", None),
                    ),
                )
                layer.register_parameter("weight_scale", weight_scale)
            for param in [
                getattr(layer, "weight", None),
                getattr(layer, "weight_scale_inv", None),
                getattr(layer, "weight_scale", None),
            ]:
                if (
                    param is not None
                    and getattr(param, "weight_loader", None) is None
                    and hasattr(layer, "weight_loader")
                ):
                    param.weight_loader = layer.weight_loader
        else:
            from vllm.model_executor.layers.linear import UnquantizedLinearMethod

            UnquantizedLinearMethod().create_weights(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Convert dense linear checkpoint weights to runtime FP8."""
        if self.online_fp8:
            if layer._tpu_online_fp8_processed:
                return
            if layer.weight.dtype != torch.bfloat16:
                raise ValueError(
                    f"Online FP8 requires BF16 checkpoint weights: "
                    f"{self.prefix} has {layer.weight.dtype}"
                )
        if self.quant_config.is_checkpoint_fp8_serialized:
            weight, weight_scale, requant_dtype_name, requant_block_size = (
                _process_fp8_linear_weights(
                    layer,
                    block_quant=self.block_quant,
                    weight_block_size=tuple(self.weight_block_size)
                    if self.weight_block_size is not None
                    else None,
                    linear_config=self.linear_config,
                    linear_quant_config=self._linear_quant_config,
                )
            )
        elif self.online_fp8:
            weight, weight_scale, requant_block_size = quantize_online_fp8(
                layer.weight.data, self.online_fp8_policy
            )
            requant_dtype_name = "fp8"
        else:
            # Respect configured TPU quantization settings (dtype, block size, kernel flags) on load.
            linear_quant_config = getattr(self, "_linear_quant_config", None)
            if linear_quant_config is None:
                linear_quant_config = _get_linear_quant_config(self.linear_config)
            desired_quant_dtype, requant_dtype, requant_block_size, blockwise_kernel = (
                linear_quant_config
            )
            weight, weight_scale = quantize_tensor(
                layer.weight.data.to(torch.float32),
                quant_dtype=requant_dtype,
                axis=-1,
                block_size=requant_block_size,
            )
            weight_scale = _format_linear_scale_for_runtime(
                weight_scale,
                blockwise_kernel=blockwise_kernel,
            )
            requant_dtype_name = desired_quant_dtype

        # Canonical (k, n) layout: the runtime matmul is (m, k) @ (k, n), so
        # the checkpoint's [n_out, n_in] weight is transposed once here at load
        # instead of being transposed per step inside the contraction. Placed
        # where both branches above converge (fp8-serialized checkpoint and
        # load-time quantization) so neither can reach the matmul N-major, and
        # after quantization so the requantizer still sees the contracting dim
        # last (axis=-1). The reallocation below materializes the view.
        ckpt_n_out, ckpt_n_in = weight.shape
        # Layers whose weight is read raw by code expecting vLLM's
        # [n_out, n_in] keep that layout; `apply` below contracts accordingly.
        if getattr(layer, WEIGHT_FLIPPED_ATTR, False):
            return  # already flipped; reload paths re-enter this hook
        keep_vllm_layout = getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False)
        if not keep_vllm_layout:
            weight = weight.transpose(0, 1)

        # The dequant/requant chain may leave views backed by higher-rank PJRT
        # buffers. Allocate fresh contiguous persistent parameters so the
        # matmul boundary sees plain row-major buffers.
        weight = torch.empty(
            weight.shape[0], weight.shape[1], dtype=weight.dtype, device=weight.device
        ).copy_(weight)
        weight_scale = torch.empty(
            *weight_scale.shape,
            dtype=weight_scale.dtype,
            device=weight_scale.device,
        ).copy_(weight_scale)

        replace_parameter(
            layer, "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        if not keep_vllm_layout:
            setattr(layer, WEIGHT_FLIPPED_ATTR, True)
        if hasattr(layer, "weight_scale_inv"):
            delattr(layer, "weight_scale_inv")
        replace_parameter(layer, "weight_scale", weight_scale)
        # Derive weight_block_size explicitly from weight_scale layout when requant_block_size is unspecified.
        # Setting (1, weight.shape[1]) vs (weight.shape[0], 1) guarantees downstream linear kernels and dequantization
        # pipelines properly recognize scale broadcast granularity right across tensor-parallel shards.
        if requant_block_size is None:
            # Stated in checkpoint [n_out, n_in] terms, which is what the
            # downstream dequantization pipelines read; the transpose above
            # changes only the runtime matmul layout, not scale granularity.
            if weight_scale.ndim == 1:
                if weight_scale.shape[0] == ckpt_n_out:
                    layer.weight_block_size = (1, ckpt_n_in)
                else:
                    layer.weight_block_size = (ckpt_n_out, 1)
            else:
                layer.weight_block_size = (ckpt_n_out, ckpt_n_in)
        else:
            layer.weight_block_size = requant_block_size

        if layer.weight.device.type == "tpu":
            synchronize_tensors([layer.weight, layer.weight_scale])

        if self.online_fp8:
            layer._tpu_online_fp8_processed = True

        scale_desc = (
            "per-channel" if requant_block_size is None else str(requant_block_size)
        )
        logger.info_once(
            "FP8 linear weights requantized for runtime: "
            f"shape={list(layer.weight.shape)}, "
            f"weight_dtype={requant_dtype_name}, "
            f"scale={list(layer.weight_scale.shape)}, "
            f"requant_block_size={scale_desc}"
        )

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        weight = layer.weight
        if getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False):
            # Stored [n_out, n_in] for a raw consumer; `quantized_matmul`
            # takes the canonical (k, n), so flip the view for the contraction.
            weight = weight.transpose(0, 1)
        out = quantized_matmul(x, weight, layer.weight_scale)
        if bias is not None:
            out = out + bias
        return out


def resolve_online_fp8(
    config: VllmFp8Config | VllmNvfp4Config,
    layer: torch.nn.Module,
    prefix: str,
    base_method: QuantizeMethodBase | None,
) -> QuantizeMethodBase | None:
    policy = config.online_fp8_policy
    if policy is None:
        return base_method
    selected, aliases = policy.select(prefix, config.packed_modules_mapping)
    if not selected:
        return base_method
    if not isinstance(layer, LinearBase):
        raise ValueError(f"Online FP8 target is not a supported Linear: {prefix}")
    if not isinstance(base_method, UnquantizedLinearMethod):
        raise ValueError(
            f"Online FP8 target {prefix} already has checkpoint "
            f"method {type(base_method).__name__}"
        )
    online_config = VllmFp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
        weight_block_size=None,
    )
    method = VllmFp8LinearMethodTPU(online_config, prefix=prefix)
    method.online_fp8 = True
    method.online_fp8_policy = policy
    policy.selected[prefix] = aliases
    return method
