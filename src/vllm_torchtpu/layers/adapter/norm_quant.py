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
"""Torch ops wrapping ``kernels.norm_quant``."""

import functools
import inspect
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.norm_quant import (
    add_rms_norm_quantize,
    moe_head_rms_norm_quantize,
    moe_tail_rms_norm_quantize,
)

_ops: dict[tuple[Any, ...], Callable] = {}


def _float_tag(value: float) -> str:
    return f"{float(value):.6e}".replace(".", "p").replace("-", "m").replace("+", "")


def _dtype_name(quant_dtype: torch.dtype) -> str:
    return str(quant_dtype).removeprefix("torch.")


def _fake_3out(x: torch.Tensor, *_rest: Any):
    return (
        torch.empty_like(x),
        torch.empty(x.shape, dtype=torch.int8, device=x.device),
        torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device),
    )


def _cached_op(
    key: tuple[Any, ...],
    name: str,
    arg_names: tuple[str, ...],
    fn: Callable[..., Any],
    fake: Callable[..., Any],
    return_annotation: Any = tuple[jax.Array, jax.Array, jax.Array],
) -> Callable:
    op = _ops.get(key)
    if op is not None:
        return op

    def wrapped(*args: jax.Array):
        return fn(*args)

    params = [
        inspect.Parameter(
            n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array
        )
        for n in arg_names
    ]
    wrapped.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        params, return_annotation=return_annotation
    )
    op = pallas.jax_op(name, wrapped)
    op.register_fake(fake)
    _ops[key] = op
    return op


def add_rms_norm_quantize_op(eps: float, quant_dtype: torch.dtype) -> Callable:
    dname = _dtype_name(quant_dtype)
    qdtype = jnp.dtype(dname)
    return _cached_op(
        ("add", float(eps), quant_dtype),
        f"pallas::add_rms_norm_quantize_{dname}_{_float_tag(eps)}",
        ("x", "residual", "w_post", "scalar", "w_next"),
        lambda x, res, wp, sc, wn: add_rms_norm_quantize(
            x, res, wp, wn, sc, eps=float(eps), quant_dtype=qdtype, bytes_out=True
        ),
        _fake_3out,
    )


def moe_tail_rms_norm_quantize_op(eps: float, quant_dtype: torch.dtype) -> Callable:
    dname = _dtype_name(quant_dtype)
    return _cached_op(
        ("tail", float(eps), quant_dtype),
        f"pallas::moe_tail_rms_norm_quantize_{dname}_{_float_tag(eps)}",
        ("mlp", "moe", "residual", "w_mlp", "w_moe", "w_post", "scalar", "w_next"),
        functools.partial(
            moe_tail_rms_norm_quantize,
            eps=float(eps),
            quant_dtype=jnp.dtype(dname),
            bytes_out=True,
        ),
        _fake_3out,
    )


def moe_head_rms_norm_quantize_op(
    eps: float, quant_dtype: torch.dtype, router_root: float
) -> Callable:
    dname = _dtype_name(quant_dtype)

    def _fake(x, _res, _wp, _wn, _wm, w_router, _rs):
        rows = x.shape[0]
        return (
            *_fake_3out(x),
            torch.empty_like(x),
            torch.empty(
                (rows, w_router.shape[0]), dtype=torch.float32, device=x.device
            ),
        )

    return _cached_op(
        ("head", float(eps), quant_dtype, float(router_root)),
        f"pallas::moe_head_rms_norm_quantize_{dname}_{_float_tag(eps)}_{_float_tag(router_root)}",
        ("x", "residual", "w_post", "w_next", "w_moe", "w_router", "router_scale"),
        functools.partial(
            moe_head_rms_norm_quantize,
            eps=float(eps),
            router_root=float(router_root),
            quant_dtype=jnp.dtype(dname),
            bytes_out=True,
        ),
        _fake,
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    )
