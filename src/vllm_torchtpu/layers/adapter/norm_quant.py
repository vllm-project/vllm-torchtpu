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
"""Torch ops of `kernels.norm_quant.add_rms_norm_quantize`.

The kernel's epsilon and FP8 dtype are static, so there is one op per pair.
Building an op cannot be traced: callers build it at model construction.
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.norm_quant import (
    add_rms_norm_quantize,
    moe_head_rms_norm_quantize,
    moe_tail_rms_norm_quantize,
)

_ops: dict[tuple[float, torch.dtype], Callable] = {}


def _add_rms_norm_quantize_jax(
    x: jax.Array,
    residual: jax.Array,
    w_post: jax.Array,
    scalar: jax.Array,
    w_next: jax.Array,
    *,
    eps: float,
    quant_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return add_rms_norm_quantize(
        x,
        residual,
        w_post,
        w_next,
        scalar,
        eps=eps,
        quant_dtype=quant_dtype,
        bytes_out=True,
    )


def add_rms_norm_quantize_op(eps: float, quant_dtype: torch.dtype) -> Callable:
    """`(x, residual, w_post, scalar, w_next) -> (h, q, scale)` for this epsilon
    and FP8 dtype (kernels/norm_quant.py), `q` holding the FP8 rows' bytes as
    int8 (collectives move them raw); `scalar` is a one-element tensor. The
    argument order is the order an unfused layer first uses them in."""
    key = (eps, quant_dtype)
    op = _ops.get(key)
    if op is not None:
        return op
    op = pallas.jax_op(
        f"pallas::add_rms_norm_quantize_{len(_ops)}",
        functools.partial(
            _add_rms_norm_quantize_jax,
            eps=eps,
            quant_dtype=jnp.dtype(str(quant_dtype).removeprefix("torch.")),
        ),
    )

    def _fake(x, residual, w_post, scalar, w_next):
        return (
            torch.empty_like(x),
            torch.empty(x.shape, dtype=torch.int8, device=x.device),
            torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
        )

    op.register_fake(_fake)
    _ops[key] = op
    return op


_moe_tail_ops: dict[tuple[float, torch.dtype], Callable] = {}


def _moe_tail_rms_norm_quantize_jax(
    mlp: jax.Array,
    moe: jax.Array,
    residual: jax.Array,
    w_mlp: jax.Array,
    w_moe: jax.Array,
    w_post: jax.Array,
    scalar: jax.Array,
    w_next: jax.Array,
    *,
    eps: float,
    quant_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # A fixed-length tuple: pallas.jax_op derives the op's return schema from
    # this annotation, and a variadic one would make it a list.
    return moe_tail_rms_norm_quantize(
        mlp,
        moe,
        residual,
        w_mlp,
        w_moe,
        w_post,
        scalar,
        w_next,
        eps=eps,
        quant_dtype=quant_dtype,
        bytes_out=True,
    )


def moe_tail_rms_norm_quantize_op(eps: float, quant_dtype: torch.dtype) -> Callable:
    """`(mlp, moe, residual, w_mlp, w_moe, w_post, scalar, w_next) -> (h, q, scale)`
    of `kernels.norm_quant.moe_tail_rms_norm_quantize` for this epsilon and FP8
    dtype, `q` holding the FP8 rows' bytes as int8; `scalar` is a one-element
    tensor. The argument order is the order an unfused layer first uses them in."""
    key = (eps, quant_dtype)
    op = _moe_tail_ops.get(key)
    if op is not None:
        return op
    op = pallas.jax_op(
        f"pallas::moe_tail_rms_norm_quantize_{len(_moe_tail_ops)}",
        functools.partial(
            _moe_tail_rms_norm_quantize_jax,
            eps=eps,
            quant_dtype=jnp.dtype(str(quant_dtype).removeprefix("torch.")),
        ),
    )

    def _fake(mlp, moe, residual, w_mlp, w_moe, w_post, scalar, w_next):
        return (
            torch.empty_like(mlp),
            torch.empty(mlp.shape, dtype=torch.int8, device=mlp.device),
            torch.empty((mlp.shape[0], 1), dtype=torch.float32, device=mlp.device),
        )

    op.register_fake(_fake)
    _moe_tail_ops[key] = op
    return op


_moe_head_ops: dict[tuple[float, torch.dtype, float], Callable] = {}


def _moe_head_rms_norm_quantize_jax(
    x: jax.Array,
    residual: jax.Array,
    w_post: jax.Array,
    w_next: jax.Array,
    w_moe: jax.Array,
    w_router: jax.Array,
    router_scale: jax.Array,
    *,
    eps: float,
    router_root: float,
    quant_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    # A fixed-length tuple: see _moe_tail_rms_norm_quantize_jax.
    return moe_head_rms_norm_quantize(
        x,
        residual,
        w_post,
        w_next,
        w_moe,
        w_router,
        router_scale,
        eps=eps,
        router_root=router_root,
        quant_dtype=quant_dtype,
        bytes_out=True,
    )


def moe_head_rms_norm_quantize_op(
    eps: float, quant_dtype: torch.dtype, router_root: float
) -> Callable:
    """`(x, residual, w_post, w_next, w_moe, w_router, router_scale) -> (h, q,
    scale, m, logits)` of `kernels.norm_quant.moe_head_rms_norm_quantize` for
    this epsilon, FP8 dtype and router root scale, `q` holding the FP8 rows'
    bytes as int8."""
    key = (eps, quant_dtype, router_root)
    op = _moe_head_ops.get(key)
    if op is not None:
        return op
    op = pallas.jax_op(
        f"pallas::moe_head_rms_norm_quantize_{len(_moe_head_ops)}",
        functools.partial(
            _moe_head_rms_norm_quantize_jax,
            eps=eps,
            router_root=router_root,
            quant_dtype=jnp.dtype(str(quant_dtype).removeprefix("torch.")),
        ),
    )

    def _fake(x, residual, w_post, w_next, w_moe, w_router, router_scale):
        rows = x.shape[0]
        return (
            torch.empty_like(x),
            torch.empty(x.shape, dtype=torch.int8, device=x.device),
            torch.empty((rows, 1), dtype=torch.float32, device=x.device),
            torch.empty_like(x),
            torch.empty(
                (rows, w_router.shape[0]), dtype=torch.float32, device=x.device
            ),
        )

    op.register_fake(_fake)
    _moe_head_ops[key] = op
    return op
