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
"""Custom-op boundary for the Qwen4Exp FP8 projections (``quant.py``).

``fp8_linear_jax`` is the JAX body of the ``qwen4_exp::fp8_linear`` op: one
FP8 x FP8 ``dot_general`` (W8A8) above the W8A16 row cutoff, or BF16
activations against the FP8 weight (W8A16) at or below it.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
# Smallest absmax a scale is computed from (weight rows and activation
# tokens). Below it, absmax / FP8_MAX is a float32 subnormal, which the TPU
# flushes to zero, and FP8_MAX / absmax overflows float32.
FP8_MIN_AMAX = FP8_MAX * float(jnp.finfo(jnp.float32).tiny)


def fp8_linear_jax(
    x: jax.Array, w_q: jax.Array, w_scale: jax.Array, w8a16_max_rows: int = 0
) -> jax.Array:
    """``x @ dequant(w_q)`` for ``x`` [m, k], ``w_q`` FP8 [k, n], ``w_scale`` [n].

    ``m`` is static at trace time, so each token bucket compiles one branch.
    """
    if x.shape[1] != w_q.shape[0]:
        raise ValueError(
            f"Input hidden dim {x.shape[1]} must match weight hidden dim "
            f"{w_q.shape[0]}."
        )
    dims = (((1,), (0,)), ((), ()))
    if 0 < x.shape[0] <= w8a16_max_rows:
        out = jax.lax.dot_general(
            x, w_q.astype(x.dtype), dims, preferred_element_type=jnp.float32
        )
        return (out * w_scale[None, :]).astype(x.dtype)
    x32 = x.astype(jnp.float32)
    amax = jnp.max(jnp.abs(x32), axis=1, keepdims=True)
    amax = jnp.maximum(amax, FP8_MIN_AMAX)
    x_q = jnp.clip(x32 * (FP8_MAX / amax), -FP8_MAX, FP8_MAX).astype(w_q.dtype)
    out = jax.lax.dot_general(x_q, w_q, dims, preferred_element_type=jnp.float32)
    return (out * (amax / FP8_MAX) * w_scale[None, :]).astype(x.dtype)


_fp8_linear_op: Callable[..., Any] | None = None
_fp8_linear_op_lock = Lock()


def _get_fp8_linear_op() -> Callable[..., Any]:
    global _fp8_linear_op
    op = _fp8_linear_op
    if op is not None:
        return op
    with _fp8_linear_op_lock:
        op = _fp8_linear_op
        if op is None:
            op = pallas.jax_op("qwen4_exp::fp8_linear", fp8_linear_jax)

            def _fake(
                x: torch.Tensor,
                w_q: torch.Tensor,
                w_scale: torch.Tensor,
                w8a16_max_rows: int = 0,
            ) -> torch.Tensor:
                del w_scale, w8a16_max_rows
                return x.new_empty(x.shape[0], w_q.shape[1])

            op.register_fake(_fake)
            _fp8_linear_op = op
    return op


def _call_fp8_linear(
    op: Callable[..., Any],
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    w8a16_max_rows: int,
) -> torch.Tensor:
    x_2d = x.reshape(-1, x.shape[-1])
    out = op(x_2d, w_q, w_scale, w8a16_max_rows)
    return out.reshape(*x.shape[:-1], w_q.shape[1])
