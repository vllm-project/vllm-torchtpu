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
"""Fused Pallas AdaLN modulation and gated residual kernels for MiniMax-H3 DiT.

MiniMax-H3 selects one of ``R = M * modalities`` modulation rows per token via
``combined_indices`` (``[S]``). Standard XLA gather (`table[indices]`)
materializes multiple ``[S, D]`` FP32 buffers in HBM and runs RMSNorm, affine
modulation, and gated residual updates in separate HBM passes.

These Pallas kernels keep the small ``[R_pad, D]`` modulation tables resident
in VMEM and select rows on the MXU via an exact FP32 one-hot ``dot_general``
(``[block_rows, R_pad] @ [R_pad, D]``, single nonzero entry per row), matching
the FP32 intermediate and BF16 residual rounding contract of
``vllm_omni.diffusion.layers.indexed_modulation``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BF16 = jnp.bfloat16
F32 = jnp.float32
SUBLANES = 8
LANES = 128
DEFAULT_ROWS = 256


def _select_2d(onehot: jax.Array, tab_ref: jax.Ref) -> jax.Array:
    """Exact ``[rows, R_pad] @ [R_pad, D] -> [rows, D]`` FP32 row selection on MXU."""
    return jax.lax.dot_general(
        onehot,
        tab_ref[...].astype(F32),
        (((1,), (0,)), ((), ())),
        preferred_element_type=F32,
        precision=jax.lax.Precision.HIGHEST,
    )


def _onehot(idx_ref: jax.Ref, r_pad: int) -> jax.Array:
    idx = idx_ref[:, :1]  # [rows, 1] int32
    return (
        idx == jax.lax.broadcasted_iota(jnp.int32, (idx.shape[0], r_pad), 1)
    ).astype(F32)


def _rms_norm(x_f32: jax.Array, w_f32: jax.Array, eps: float) -> jax.Array:
    return (
        x_f32
        * jax.lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
        * w_f32
    )


def _rows_for(s: int) -> int:
    rows = DEFAULT_ROWS
    while s % rows and rows > SUBLANES:
        rows //= 2
    return max(rows, SUBLANES)


def _pad_rows_and_indices(
    s0: int, rows: int, idx: jax.Array, *tensors: jax.Array
) -> tuple[int, jax.Array, list[jax.Array]]:
    s_pad = -(-s0 // rows) * rows
    pad_s = s_pad - s0
    idx_i32 = idx.astype(jnp.int32)
    if pad_s:
        idx_i32 = jnp.pad(idx_i32, (0, pad_s), constant_values=0)
        padded = [jnp.pad(t, ((0, pad_s), (0, 0))) for t in tensors]
    else:
        padded = list(tensors)
    idx_2d = jnp.broadcast_to(idx_i32[:, None], (s_pad, LANES))
    return s_pad, idx_2d, padded


def _pad_tables(*tables: jax.Array) -> tuple[int, list[jax.Array]]:
    r0 = tables[0].shape[0]
    r_pad = max(SUBLANES, -(-r0 // SUBLANES) * SUBLANES)
    pad_r = r_pad - r0
    out = [
        jnp.pad(t.astype(F32), ((0, pad_r), (0, 0))) if pad_r else t.astype(F32)
        for t in tables
    ]
    return r_pad, out


def _compiler_params() -> pltpu.CompilerParams:
    return pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=48 * 1024 * 1024,
    )


def _rms_norm_scale_shift_kernel(
    x_ref: jax.Ref,
    w_ref: jax.Ref,
    shift_ref: jax.Ref,
    scale_ref: jax.Ref,
    idx_ref: jax.Ref,
    out_ref: jax.Ref,
    *,
    eps: float,
) -> None:
    oh = _onehot(idx_ref, shift_ref.shape[0])
    normed = _rms_norm(x_ref[...].astype(F32), w_ref[...][:1, :], eps)
    out = normed * (1.0 + _select_2d(oh, scale_ref)) + _select_2d(oh, shift_ref)
    out_ref[...] = out.astype(out_ref.dtype)


def _gate_rms_norm_scale_shift_kernel(
    res_ref: jax.Ref,
    gate_ref: jax.Ref,
    branch_ref: jax.Ref,
    w_ref: jax.Ref,
    shift_ref: jax.Ref,
    scale_ref: jax.Ref,
    idx_ref: jax.Ref,
    res_out_ref: jax.Ref,
    mod_out_ref: jax.Ref,
    *,
    eps: float,
) -> None:
    oh = _onehot(idx_ref, gate_ref.shape[0])
    # Match vllm-omni's indexed_gate_rms_norm_scale_shift rounding contract:
    # residual stream rounds to BF16 before feeding the MLP RMSNorm in FP32.
    res_bf16 = (
        res_ref[...].astype(F32)
        + _select_2d(oh, gate_ref) * branch_ref[...].astype(F32)
    ).astype(res_out_ref.dtype)
    res_out_ref[...] = res_bf16
    normed = _rms_norm(res_bf16.astype(F32), w_ref[...][:1, :], eps)
    mod = normed * (1.0 + _select_2d(oh, scale_ref)) + _select_2d(oh, shift_ref)
    mod_out_ref[...] = mod.astype(mod_out_ref.dtype)


def _gate_kernel(
    x_ref: jax.Ref,
    gate_ref: jax.Ref,
    other_ref: jax.Ref,
    idx_ref: jax.Ref,
    out_ref: jax.Ref,
) -> None:
    oh = _onehot(idx_ref, gate_ref.shape[0])
    out = x_ref[...].astype(F32) + _select_2d(oh, gate_ref) * other_ref[...].astype(F32)
    out_ref[...] = out.astype(out_ref.dtype)


def _scale_shift_kernel(
    x_ref: jax.Ref,
    shift_ref: jax.Ref,
    scale_ref: jax.Ref,
    idx_ref: jax.Ref,
    out_ref: jax.Ref,
) -> None:
    oh = _onehot(idx_ref, shift_ref.shape[0])
    out = x_ref[...].astype(F32) * (1.0 + _select_2d(oh, scale_ref)) + _select_2d(
        oh, shift_ref
    )
    out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("eps", "interpret"))
def rms_norm_indexed_scale_shift_pallas(
    x: jax.Array,
    weight: jax.Array,
    shift: jax.Array,
    scale: jax.Array,
    indices: jax.Array,
    *,
    eps: float,
    interpret: bool = False,
) -> jax.Array:
    """Fused ``RMSNorm(x, weight, eps) * (1 + scale[indices]) + shift[indices]``."""
    s0, d = x.shape
    rows = _rows_for(s0)
    s_pad, idx_2d, (xp,) = _pad_rows_and_indices(s0, rows, indices, x)
    r_pad, (shift_p, scale_p) = _pad_tables(shift, scale)
    wb = jnp.broadcast_to(weight.astype(F32)[None, :], (SUBLANES, d))

    row_spec = pl.BlockSpec((rows, d), lambda i: (i, 0))
    w_spec = pl.BlockSpec((SUBLANES, d), lambda i: (0, 0))
    tab_spec = pl.BlockSpec((r_pad, d), lambda i: (0, 0))
    idx_spec = pl.BlockSpec((rows, LANES), lambda i: (i, 0))

    out = pl.pallas_call(
        functools.partial(_rms_norm_scale_shift_kernel, eps=eps),
        grid=(s_pad // rows,),
        in_specs=[row_spec, w_spec, tab_spec, tab_spec, idx_spec],
        out_specs=row_spec,
        out_shape=jax.ShapeDtypeStruct((s_pad, d), x.dtype),
        compiler_params=_compiler_params(),
        interpret=interpret,
        name="diffusion_adaln_rms_norm_scale_shift",
    )(xp, wb, shift_p, scale_p, idx_2d)
    return out[:s0]


@functools.partial(jax.jit, static_argnames=("eps", "interpret"))
def indexed_gate_rms_norm_scale_shift_pallas(
    residual: jax.Array,
    gate: jax.Array,
    branch: jax.Array,
    weight: jax.Array,
    shift: jax.Array,
    scale: jax.Array,
    indices: jax.Array,
    *,
    eps: float,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused gated residual update + next-sublayer AdaLN RMSNorm modulation."""
    s0, d = residual.shape
    rows = _rows_for(s0)
    s_pad, idx_2d, (res_p, br_p) = _pad_rows_and_indices(
        s0, rows, indices, residual, branch
    )
    r_pad, (gate_p, shift_p, scale_p) = _pad_tables(gate, shift, scale)
    wb = jnp.broadcast_to(weight.astype(F32)[None, :], (SUBLANES, d))

    row_spec = pl.BlockSpec((rows, d), lambda i: (i, 0))
    w_spec = pl.BlockSpec((SUBLANES, d), lambda i: (0, 0))
    tab_spec = pl.BlockSpec((r_pad, d), lambda i: (0, 0))
    idx_spec = pl.BlockSpec((rows, LANES), lambda i: (i, 0))
    out_struct = jax.ShapeDtypeStruct((s_pad, d), residual.dtype)

    res_out, mod_out = pl.pallas_call(
        functools.partial(_gate_rms_norm_scale_shift_kernel, eps=eps),
        grid=(s_pad // rows,),
        in_specs=[
            row_spec,
            tab_spec,
            row_spec,
            w_spec,
            tab_spec,
            tab_spec,
            idx_spec,
        ],
        out_specs=[row_spec, row_spec],
        out_shape=[out_struct, out_struct],
        compiler_params=_compiler_params(),
        interpret=interpret,
        name="diffusion_adaln_gate_rms_norm_scale_shift",
    )(res_p, gate_p, br_p, wb, shift_p, scale_p, idx_2d)
    return res_out[:s0], mod_out[:s0]


@functools.partial(jax.jit, static_argnames=("interpret",))
def indexed_gate_pallas(
    x: jax.Array,
    gate: jax.Array,
    other: jax.Array,
    indices: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Fused ``x + gate[indices] * other`` keeping ``gate`` in VMEM."""
    s0, d = x.shape
    rows = _rows_for(s0)
    s_pad, idx_2d, (xp, op) = _pad_rows_and_indices(s0, rows, indices, x, other)
    r_pad, (gate_p,) = _pad_tables(gate)

    row_spec = pl.BlockSpec((rows, d), lambda i: (i, 0))
    tab_spec = pl.BlockSpec((r_pad, d), lambda i: (0, 0))
    idx_spec = pl.BlockSpec((rows, LANES), lambda i: (i, 0))

    out = pl.pallas_call(
        _gate_kernel,
        grid=(s_pad // rows,),
        in_specs=[row_spec, tab_spec, row_spec, idx_spec],
        out_specs=row_spec,
        out_shape=jax.ShapeDtypeStruct((s_pad, d), x.dtype),
        compiler_params=_compiler_params(),
        interpret=interpret,
        name="diffusion_adaln_indexed_gate",
    )(xp, gate_p, op, idx_2d)
    return out[:s0]


@functools.partial(jax.jit, static_argnames=("interpret",))
def indexed_scale_shift_pallas(
    x: jax.Array,
    shift: jax.Array,
    scale: jax.Array,
    indices: jax.Array,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Fused ``x * (1 + scale[indices]) + shift[indices]`` keeping tables in VMEM."""
    s0, d = x.shape
    rows = _rows_for(s0)
    s_pad, idx_2d, (xp,) = _pad_rows_and_indices(s0, rows, indices, x)
    r_pad, (shift_p, scale_p) = _pad_tables(shift, scale)

    row_spec = pl.BlockSpec((rows, d), lambda i: (i, 0))
    tab_spec = pl.BlockSpec((r_pad, d), lambda i: (0, 0))
    idx_spec = pl.BlockSpec((rows, LANES), lambda i: (i, 0))

    out = pl.pallas_call(
        _scale_shift_kernel,
        grid=(s_pad // rows,),
        in_specs=[row_spec, tab_spec, tab_spec, idx_spec],
        out_specs=row_spec,
        out_shape=jax.ShapeDtypeStruct((s_pad, d), x.dtype),
        compiler_params=_compiler_params(),
        interpret=interpret,
        name="diffusion_adaln_indexed_scale_shift",
    )(xp, shift_p, scale_p, idx_2d)
    return out[:s0]
