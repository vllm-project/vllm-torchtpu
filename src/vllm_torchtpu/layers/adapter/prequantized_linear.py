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
"""Dynamic-FP8 dense linear split into row quantization and pre-quantized matmul.

``xla_quantized_matmul`` quantizes each activation row dynamically with its own
scale and then contracts against ``[n_in, n_out]`` FP8 weights. Splitting that
operation at the quantized activation allows TP ranks to quantize only their
local token shard, all-gather 1-byte FP8 rows plus f32 row scales instead of
bf16 activations, and run the column-parallel matmul without re-quantizing.

Row quantization is strictly per token, so quantizing row shards and gathering
produces the exact same quantized bytes and scales as quantizing the full
activation tensor on every rank.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax import lax
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
)

_FP8 = jnp.float8_e4m3fn

_ops_lock = Lock()
_quantize_rows_op: (
    Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
) = None
_prequantized_matmul_op: (
    Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]
    | None
) = None


def quantize_rows_jax(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Quantize ``[m, k]`` activations into ``([m, k] uint8, [m, 1] f32)``.

    Uses ``quantize_tensor`` so the quantized values and scales match the
    activation half of ``xla_quantized_matmul`` bit for bit. The FP8 payload is
    bitcast to ``uint8`` so PyTorch collectives can transport it cleanly.
    """
    x_q, x_scale = quantize_tensor(x, _FP8)
    return lax.bitcast_convert_type(x_q, jnp.uint8), x_scale


def prequantized_matmul_jax(
    x_q_u8: jax.Array,
    x_scale: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
) -> jax.Array:
    """Contract ``(m, k) @ (k, n)`` on rows from ``quantize_rows_jax``.

    Mirrors the contraction half of ``xla_quantized_matmul``: f32 accumulation,
    per-row activation scale, per-channel weight scale, and bf16 output cast.
    """
    if x_q_u8.shape[1] != w_q.shape[0]:
        raise ValueError(
            f"Input hidden dim {x_q_u8.shape[1]} must match weight hidden dim "
            f"{w_q.shape[0]}."
        )
    if w_scale.ndim != 1 or w_scale.shape[0] not in (1, w_q.shape[1]):
        raise ValueError(
            f"Expected 1-D weight scale of length 1 or {w_q.shape[1]}, "
            f"got {w_scale.shape=}"
        )
    if not jnp.issubdtype(w_q.dtype, jnp.floating):
        raise ValueError(f"Expected FP8 weights, got {w_q.dtype=}")
    x_q = lax.bitcast_convert_type(x_q_u8, _FP8)
    out = lax.dot_general(
        x_q,
        w_q,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    out *= x_scale
    out *= jnp.expand_dims(w_scale, 0)
    return out.astype(jnp.bfloat16)


def ensure_ops() -> None:
    """Register the ``pallas.jax_op`` custom ops at model construction time."""
    global _quantize_rows_op, _prequantized_matmul_op
    if _quantize_rows_op is not None and _prequantized_matmul_op is not None:
        return

    with _ops_lock:
        if _quantize_rows_op is not None and _prequantized_matmul_op is not None:
            return

        quant = pallas.jax_op("pallas::fp8_quantize_rows", quantize_rows_jax)

        def _fake_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.empty(x.shape, dtype=torch.uint8, device=x.device),
                torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device),
            )

        quant.register_fake(_fake_quant)

        matmul = pallas.jax_op(
            "pallas::fp8_prequantized_matmul", prequantized_matmul_jax
        )

        def _fake_matmul(
            x_q_u8: torch.Tensor,
            x_scale: torch.Tensor,
            w_q: torch.Tensor,
            w_scale: torch.Tensor,
        ) -> torch.Tensor:
            del x_scale, w_scale
            return torch.empty(
                x_q_u8.shape[0],
                w_q.shape[-1],
                dtype=torch.bfloat16,
                device=x_q_u8.device,
            )

        matmul.register_fake(_fake_matmul)
        _prequantized_matmul_op = matmul
        _quantize_rows_op = quant


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2-D bf16 activations into ``(uint8 FP8 bytes, f32 row scales)``."""
    if x.ndim != 2:
        raise ValueError(
            f"quantize_rows expects 2-D activations, got shape {tuple(x.shape)}"
        )
    if x.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"quantize_rows requires bfloat16 activations, got {x.dtype}"
        )
    if _quantize_rows_op is None:
        raise RuntimeError(
            "ensure_ops() must be called at model construction before quantize_rows()"
        )
    return _quantize_rows_op(x)


def check_prequantized_linear(layer: torch.nn.Module) -> None:
    """Validate that ``layer`` matches ``VllmFp8LinearMethodTPU`` per-channel FP8."""
    name = getattr(layer, "prefix", type(layer).__name__)
    if getattr(layer, "bias", None) is not None:
        raise NotImplementedError(f"{name}: prequantized FP8 input with a bias")
    weight: Any = getattr(layer, "weight", None)
    if (
        weight is None
        or not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.float8_e4m3fn
    ):
        dtype = getattr(weight, "dtype", None)
        raise NotImplementedError(
            f"{name}: prequantized FP8 input requires float8_e4m3fn weights, "
            f"got {dtype}"
        )
    weight_scale: Any = getattr(layer, "weight_scale", None)
    if (
        weight_scale is None
        or not isinstance(weight_scale, torch.Tensor)
        or weight_scale.ndim != 1
        or weight_scale.shape[0] not in (1, weight.shape[1])
    ):
        shape = (
            tuple(weight_scale.shape)
            if weight_scale is not None and isinstance(weight_scale, torch.Tensor)
            else None
        )
        raise NotImplementedError(
            f"{name}: prequantized FP8 input requires a 1-D per-channel weight_scale, "
            f"got {shape}"
        )
    if getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False) or not getattr(
        layer, WEIGHT_FLIPPED_ATTR, False
    ):
        raise NotImplementedError(
            f"{name}: prequantized FP8 input requires [n_in, n_out] flipped weights"
        )
    quant_method = getattr(layer, "quant_method", None)
    if getattr(quant_method, "w8a16_max_rows", 0):
        raise NotImplementedError(
            f"{name}: prequantized FP8 input does not support "
            "TPU_FP8_LINEAR_W8A16_MAX_ROWS"
        )


def prequantized_linear(
    layer: Any, x_q_u8: torch.Tensor, x_scale: torch.Tensor
) -> torch.Tensor:
    """Run ``layer``'s FP8 matmul on rows already quantized by ``quantize_rows``."""
    if _prequantized_matmul_op is None:
        raise RuntimeError(
            "ensure_ops() must be called at model construction before "
            "prequantized_linear()"
        )
    weight: torch.Tensor = layer.weight
    weight_scale: torch.Tensor = layer.weight_scale
    if x_q_u8.shape[1] != weight.shape[0]:
        raise ValueError(
            f"Input hidden dim {x_q_u8.shape[1]} must match weight hidden dim "
            f"{weight.shape[0]}."
        )
    return _prequantized_matmul_op(x_q_u8, x_scale, weight, weight_scale)
