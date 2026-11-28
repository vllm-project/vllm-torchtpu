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
"""Inkling 2 checkpoint quantization metadata and pure-PyTorch weight codecs.

Inkling 2 checkpoints declare per-tensor compute precision in
``hf_quant_config.json`` under ``model.parameter_groups``: routed experts use
mixed ``nvfp4`` (block-16 E4M3 + per-expert ``amax / outer_scale_denominator``)
and ``mxfp8`` (block-32 UE8M0 power-of-two scales), while attention, routers,
loop controllers, norms, and the LM head remain unquantized BF16/FP32.

Dequantizing MXFP8 (3-bit E4M3 mantissa times a power-of-two scale) to BF16
(7-bit mantissa) is numerically exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

INKLING2_QUANT_METHOD = "inkling2"
_UE8M0_EXPONENT_BIAS = 127
_SUPPORTED_COMPUTE_DTYPES = frozenset({"float32", "bfloat16", "mxfp8", "nvfp4"})

ExpertQuantFormat = Literal["nvfp4", "mxfp8"]


@dataclass(frozen=True)
class Inkling2QuantMap:
    """Per-tensor precision map parsed from an Inkling 2 ``hf_quant_config.json``."""

    mxfp8_weights: frozenset[str]
    nvfp4_weights: frozenset[str]

    @classmethod
    def from_hf_quant_config(cls, config: dict[str, Any]) -> Inkling2QuantMap:
        """Parse ``model.parameter_groups`` and validate every compute dtype."""
        groups = config.get("model", {}).get("parameter_groups")
        if not isinstance(groups, list):
            raise ValueError(
                "Inkling2 hf_quant_config.json must define model.parameter_groups"
            )
        mxfp8: set[str] = set()
        nvfp4: set[str] = set()
        for group in groups:
            compute_dtype = group.get("compute_dtype")
            if compute_dtype not in _SUPPORTED_COMPUTE_DTYPES:
                raise ValueError(
                    f"Unsupported Inkling2 compute_dtype: {compute_dtype!r}"
                )
            tensors = group.get("tensors", ())
            if compute_dtype == "mxfp8":
                mxfp8.update(tensors)
            elif compute_dtype == "nvfp4":
                nvfp4.update(tensors)
        return cls(
            mxfp8_weights=frozenset(mxfp8),
            nvfp4_weights=frozenset(nvfp4),
        )

    def is_mxfp8_layer(self, layer_id: int) -> bool:
        """Whether routed experts in ``layer_id`` are stored in MXFP8."""
        return (
            f"model.llm.layers.{layer_id}.mlp.experts.w13_weight" in self.mxfp8_weights
        )

    def expert_format(self, layer_id: int) -> ExpertQuantFormat:
        """Return ``'mxfp8'`` or ``'nvfp4'`` for routed experts in ``layer_id``."""
        w13_key = f"model.llm.layers.{layer_id}.mlp.experts.w13_weight"
        if w13_key in self.mxfp8_weights:
            return "mxfp8"
        if w13_key in self.nvfp4_weights:
            return "nvfp4"
        raise KeyError(
            f"Layer {layer_id} routed expert tensor {w13_key!r} not found in "
            "NVFP4 or MXFP8 parameter groups"
        )


def _derive_group_size(
    weight_shape: tuple[int, ...], scale_shape: tuple[int, ...]
) -> int:
    if scale_shape[:-1] != weight_shape[:-1]:
        raise ValueError(
            f"Scale leading shape {scale_shape[:-1]} does not match "
            f"weight leading shape {weight_shape[:-1]}"
        )
    in_dim, num_blocks = weight_shape[-1], scale_shape[-1]
    if num_blocks <= 0 or in_dim % num_blocks != 0:
        raise ValueError(
            f"Weight contracting dim {in_dim} must be a positive multiple of "
            f"scale block count {num_blocks}"
        )
    return in_dim // num_blocks


def dequantize_mxfp8(
    weight: torch.Tensor,
    scale_e8m0: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize MXFP8 weights ``[..., N, K]`` with UE8M0 scales ``[..., N, B]``.

    The block size ``K // B`` is derived from the tensor shapes.

    Args:
        weight: ``float8_e4m3fn`` tensor of shape ``[..., N, K]``.
        scale_e8m0: ``uint8`` UE8M0 exponent tensor of shape ``[..., N, B]``.
        out_dtype: Destination dtype (default ``bfloat16``, exact for E4M3 * 2^e).

    Returns:
        Dequantized tensor of shape ``[..., N, K]`` in ``out_dtype``.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Expected float8_e4m3fn MXFP8 weights, got {weight.dtype}")
    if scale_e8m0.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 UE8M0 scales, got {scale_e8m0.dtype}")
    group_size = _derive_group_size(tuple(weight.shape), tuple(scale_e8m0.shape))
    *lead, out_dim, in_dim = weight.shape
    scale = torch.exp2(scale_e8m0.to(torch.float32) - _UE8M0_EXPONENT_BIAS)
    grouped = weight.to(torch.float32).reshape(
        *lead, out_dim, in_dim // group_size, group_size
    )
    return (grouped * scale.unsqueeze(-1)).reshape(*lead, out_dim, in_dim).to(out_dtype)


def unpack_uint8_to_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 E2M1 nibbles (low nibble first) ``[..., K//2]`` -> fp32.

    Uses branchless bitwise arithmetic rather than LUT gather indexing so the
    decoding vectorizes cleanly on TPU VPU lanes under ``torch.compile``.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 packed FP4 tensor, got {packed.dtype}")
    nibbles = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1)
    sign_bit = (nibbles >> 3) & 0x01
    exp = (nibbles >> 1) & 0x03
    mant = (nibbles & 0x01).to(torch.float32)
    normal_mag = (1.0 + 0.5 * mant) * torch.exp2(
        (exp.to(torch.float32) - 1.0).clamp_min(0.0)
    )
    mag = torch.where(exp != 0, normal_mag, 0.5 * mant)
    vals = torch.where(sign_bit != 0, -mag, mag)
    return vals.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequantize_nvfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_amax: torch.Tensor,
    weight_outer_scale_denominator: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize Inkling 2 NVFP4 weights ``[..., N, K//2]`` to ``out_dtype``.

    The block size ``K // weight_scale.shape[-1]`` is derived from tensor shapes.

    Args:
        weight_packed: ``uint8`` packed E2M1 weights ``[..., N, K // 2]``.
        weight_scale: ``float8_e4m3fn`` (or fp32) block scales ``[..., N, B]``.
        weight_amax: Per-expert fp32 amax ``[...]``.
        weight_outer_scale_denominator: Single-element fp32 outer scale denominator.
        out_dtype: Destination dtype.

    Returns:
        Dequantized tensor of shape ``[..., N, K]`` in ``out_dtype``.
    """
    if weight_outer_scale_denominator.numel() != 1:
        raise ValueError(
            "Expected single-element weight_outer_scale_denominator, got shape "
            f"{tuple(weight_outer_scale_denominator.shape)}"
        )
    fp4_f32 = unpack_uint8_to_fp4(weight_packed)
    group_size = _derive_group_size(tuple(fp4_f32.shape), tuple(weight_scale.shape))
    *lead, out_dim, in_dim = fp4_f32.shape
    global_scale = weight_amax.to(torch.float32) / weight_outer_scale_denominator.to(
        torch.float32
    )
    scales = weight_scale.to(torch.float32)
    while global_scale.ndim < scales.ndim:
        global_scale = global_scale.unsqueeze(-1)
    grouped = fp4_f32.reshape(*lead, out_dim, in_dim // group_size, group_size)
    return (
        (grouped * (scales * global_scale).unsqueeze(-1))
        .reshape(*lead, out_dim, in_dim)
        .to(out_dtype)
    )
