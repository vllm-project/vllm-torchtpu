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
"""Custom-op bridges for dynamic per-row FP8 quantization and pre-quantized matmul.

Splitting `xla_quantized_matmul` at the quantized activation lets each TP rank
quantize only its own token shard and all-gather 1-byte FP8 activations plus
per-row f32 scales instead of 2-byte bf16 activations.

Unlike `pipelined_fused_moe._quantized_all_gather` (which dequantizes gathered
rows back to bf16 on arrival), these two ops keep `(x_q, x_scale)` quantized
into the linear contraction via `util.quantize_tensor`, avoiding a redundant
dequantize + requantize round-trip and matching `xla_quantized_matmul` bit for
bit.
"""

from collections.abc import Callable
from threading import Lock
from typing import NamedTuple

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_prequantized_matmul,
)


class _Fp8LinearOps(NamedTuple):
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    matmul: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]


def _quantize_fp8_activations_jax(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`[m, k]` bf16 -> (`[m, k]` FP8 bits as uint8, `[m, 1]` f32 row scales).

    Returned as uint8 because torch collectives support uint8 rather than FP8.
    """
    x_q, x_scale = quantize_tensor(x, jnp.float8_e4m3fn)
    return jax.lax.bitcast_convert_type(x_q, jnp.uint8), x_scale


def _prequantized_fp8_matmul_jax(
    x_q: jax.Array, x_scale: jax.Array, w_q: jax.Array, w_scale: jax.Array
) -> jax.Array:
    x_q_fp8 = jax.lax.bitcast_convert_type(x_q, jnp.float8_e4m3fn)
    return xla_prequantized_matmul(
        x_q_fp8, x_scale, w_q, w_scale, out_dtype=jnp.bfloat16
    )


_ops: _Fp8LinearOps | None = None
_ops_lock = Lock()


def _get_ops() -> _Fp8LinearOps:
    global _ops
    if _ops is not None:
        return _ops

    with _ops_lock:
        if _ops is not None:
            return _ops

        quantize_op = pallas.jax_op(
            "pallas::fp8_quantize_activations", _quantize_fp8_activations_jax
        )

        def _fake_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                x.new_empty(x.shape, dtype=torch.uint8),
                x.new_empty((x.shape[0], 1), dtype=torch.float32),
            )

        quantize_op.register_fake(_fake_quantize)

        matmul_op = pallas.jax_op(
            "pallas::fp8_prequantized_matmul", _prequantized_fp8_matmul_jax
        )

        def _fake_matmul(
            x_q: torch.Tensor,
            x_scale: torch.Tensor,
            w_q: torch.Tensor,
            w_scale: torch.Tensor,
        ) -> torch.Tensor:
            del x_scale, w_scale
            return x_q.new_empty((x_q.shape[0], w_q.shape[-1]), dtype=torch.bfloat16)

        matmul_op.register_fake(_fake_matmul)
        _ops = _Fp8LinearOps(quantize=quantize_op, matmul=matmul_op)
        return _ops


def register_fp8_linear_ops() -> None:
    """Registers the FP8 activation-quantization and matmul ops before tracing."""
    _get_ops()


def quantize_fp8_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantizes `[m, k]` bf16 activations to (`[m, k]` uint8, `[m, 1]` f32 scales)."""
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"quantize_fp8_activations expects 2-D bfloat16 activations, got "
            f"{x.dtype} {tuple(x.shape)}"
        )
    return _get_ops().quantize(x)


def prequantized_fp8_matmul(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Contracts pre-quantized `uint8[m, k]` rows with `(k, n)` FP8 weights."""
    if x_q.ndim != 2 or x_q.dtype != torch.uint8:
        raise NotImplementedError(
            f"prequantized_fp8_matmul expects 2-D uint8 x_q, got "
            f"{x_q.dtype} {tuple(x_q.shape)}"
        )
    if (
        x_scale.ndim != 2
        or x_scale.dtype != torch.float32
        or x_scale.shape[0] != x_q.shape[0]
        or x_scale.shape[1] != 1
    ):
        raise NotImplementedError(
            f"prequantized_fp8_matmul expects float32 x_scale of shape "
            f"({x_q.shape[0]}, 1), got {x_scale.dtype} {tuple(x_scale.shape)}"
        )
    if weight.ndim != 2 or x_q.shape[1] != weight.shape[0]:
        raise NotImplementedError(
            f"prequantized_fp8_matmul expects 2-D weight with shape[0] == "
            f"x_q.shape[1] ({x_q.shape[1]}), got {tuple(weight.shape)}"
        )
    return _get_ops().matmul(x_q, x_scale, weight, weight_scale)
