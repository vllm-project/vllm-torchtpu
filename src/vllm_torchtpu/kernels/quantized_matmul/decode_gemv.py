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
"""Weight-streaming matmuls for decode-sized token buckets."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.row_blocks import vmem_limit_bytes

_TILE_BYTES = 3 * 1024 * 1024
_NT_TILE_BYTES = 2 * _TILE_BYTES
_DEPTH = 4
_LANES = 128
_ROW_TILE = 32
_MAX_FP8_VMEM_BYTES = 32 * 1024 * 1024
_MAX_NT_VMEM_BYTES = 48 * 1024 * 1024
_VMEM_BUDGET = 0.75
_VMEM = pl.BlockSpec(memory_space=pltpu.VMEM)
_HBM = pl.BlockSpec(memory_space=pltpu.HBM)


def tile_config(
    k: int, n: int, itemsize: int = 1, tile_bytes: int = _TILE_BYTES
) -> tuple[int, int, int]:
    lanes = n // _LANES
    tn = _LANES
    for d in range(lanes, 0, -1):
        if lanes % d == 0 and k * d * _LANES * itemsize <= tile_bytes:
            tn = d * _LANES
            break
    splits = 2 if k % (2 * _ROW_TILE) == 0 else 1
    return tn, min(_DEPTH, n // tn), splits


def _fits_vmem(buffer_bytes: int, limit: int) -> bool:
    return buffer_bytes <= _VMEM_BUDGET * limit


def supports(
    rows: int,
    k: int,
    n: int,
    w_dtype: jnp.dtype,
    w_scale_ndim: int,
    row_multiple: int = 16,
) -> bool:
    if not (
        jnp.issubdtype(w_dtype, jnp.floating)
        and jnp.dtype(w_dtype).itemsize == 1
        and w_scale_ndim == 1
        and rows % row_multiple == 0
        and row_multiple % 8 == 0
        and n % _LANES == 0
        and k % _ROW_TILE == 0
    ):
        return False
    tn, depth, _ = tile_config(k, n)
    mp = -(-rows // _ROW_TILE) * _ROW_TILE
    rows_bytes = rows * k * 2 + mp * k * (1 + 4)
    out_bytes = rows * n * 2 + n * 4 + mp * tn * 4
    return _fits_vmem(depth * k * tn + rows_bytes + out_bytes, _MAX_FP8_VMEM_BYTES)


def _quantize_rows(x: jax.Array, quant_dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
    info = jnp.finfo(quant_dtype)
    xf = x.astype(jnp.float32)
    scale = jnp.max(jnp.abs(xf), axis=-1, keepdims=True) / float(info.max)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    q = jnp.clip(xf / scale, float(info.min), float(info.max))
    return q.astype(quant_dtype), scale


def _stream_matmul(
    xq_ref, xs, ws_vmem, w_hbm, o_ref, wbuf, sems, scales, *, rows, tn, depth, splits
):
    k = wbuf.shape[1]
    kc = k // splits
    nt = w_hbm.shape[1] // tn

    def copy(t, slot, s):
        return pltpu.make_async_copy(
            w_hbm.at[pl.ds(s * kc, kc), pl.ds(t * tn, tn)],
            wbuf.at[slot, pl.ds(s * kc, kc)],
            sems.at[slot, s],
        )

    def body(t, carry):
        slot = lax.rem(t, depth)
        for s in range(splits):
            copy(t, slot, s).wait()
        pl.when(t == 0)(scales.wait)
        acc = jnp.dot(xq_ref[...], wbuf[slot], preferred_element_type=jnp.float32)
        col = pl.multiple_of(t * tn, _LANES)
        out = acc * xs * ws_vmem[pl.ds(col, tn)].reshape(1, tn)
        o_ref[:, pl.ds(col, tn)] = out[:rows].astype(o_ref.dtype)

        @pl.when(t + depth < nt)
        def _():
            for s in range(splits):
                copy(t + depth, slot, s).start()

        return carry

    lax.fori_loop(0, nt, body, 0)


def _start_copies(w_hbm, wbuf, sems, ws_hbm, ws_vmem, ws_sem, *, tn, depth, splits):
    scales = pltpu.make_async_copy(ws_hbm, ws_vmem, ws_sem.at[0])
    scales.start()
    kc = wbuf.shape[1] // splits
    for t in range(depth):
        for s in range(splits):
            pltpu.make_async_copy(
                w_hbm.at[pl.ds(s * kc, kc), pl.ds(t * tn, tn)],
                wbuf.at[t, pl.ds(s * kc, kc)],
                sems.at[t, s],
            ).start()
    return scales


def _pad_rows(x: jax.Array, rows: int) -> jax.Array:
    if x.shape[0] == rows:
        return x
    pad = jnp.zeros((rows - x.shape[0], x.shape[1]), x.dtype)
    return jnp.concatenate([x, pad], axis=0)


def _hbm(x: jax.Array) -> jax.Array:
    return pltpu.with_memory_space_constraint(x, pltpu.HBM)


def _fp8_run(prep_fn, x_args, w_q, w_scale, out_dtype, tile_bytes, name, interpret):
    m, k = x_args[0].shape
    n = w_q.shape[1]
    tn, depth, splits = tile_config(k, n, tile_bytes=tile_bytes)
    mp = -(-m // _ROW_TILE) * _ROW_TILE
    nx = len(x_args)

    def kernel(*refs):
        in_refs = refs[:nx]
        w_hbm, ws_hbm, o_ref, wbuf, sems, ws_vmem, ws_sem, xq_ref = refs[nx:]
        scales = _start_copies(
            w_hbm,
            wbuf,
            sems,
            ws_hbm,
            ws_vmem,
            ws_sem,
            tn=tn,
            depth=depth,
            splits=splits,
        )
        xs = prep_fn(*in_refs, xq_ref, m, mp, k, w_q.dtype)
        _stream_matmul(
            xq_ref,
            xs,
            ws_vmem,
            w_hbm,
            o_ref,
            wbuf,
            sems,
            scales,
            rows=m,
            tn=tn,
            depth=depth,
            splits=splits,
        )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        in_specs=[_VMEM] * nx + [_HBM, _HBM],
        out_specs=_VMEM,
        scratch_shapes=[
            pltpu.VMEM((depth, k, tn), w_q.dtype),
            pltpu.SemaphoreType.DMA((depth, splits)),
            pltpu.VMEM((n,), jnp.float32),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.VMEM((mp, k), w_q.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes(_MAX_FP8_VMEM_BYTES)
        ),
        name=f"{name}_tn{tn}_d{depth}_s{splits}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*x_args, _hbm(w_q), _hbm(w_scale.astype(jnp.float32)))


@functools.partial(jax.jit, static_argnames=("tile_bytes", "interpret"))
def fp8_decode_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    tile_bytes: int = _TILE_BYTES,
    interpret: bool = False,
) -> jax.Array:
    def prep(x_ref, xq_ref, _m, mp, _k, w_dtype):
        q, xs = _quantize_rows(_pad_rows(x_ref[...], mp), w_dtype)
        xq_ref[...] = q
        return xs

    return _fp8_run(
        prep, (x,), w_q, w_scale, x.dtype, tile_bytes, "fp8_decode_matmul", interpret
    )


@functools.partial(jax.jit, static_argnames=("out_dtype", "tile_bytes", "interpret"))
def fp8_decode_prequantized_matmul(
    x_q: jax.Array,
    x_scale: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    out_dtype: jnp.dtype,
    tile_bytes: int = _TILE_BYTES,
    interpret: bool = False,
) -> jax.Array:
    def prep(xq_in, xs_ref, xq_ref, m, mp, k, _w_dtype):
        xq_ref[pl.ds(0, m)] = xq_in[...]
        if mp > m:
            xq_ref[pl.ds(m, mp - m)] = jnp.zeros((mp - m, k), xq_ref.dtype)
        xs = xs_ref[...].astype(jnp.float32)
        return (
            jnp.concatenate([xs, jnp.ones((mp - m, 1), jnp.float32)], axis=0)
            if mp > m
            else xs
        )

    return _fp8_run(
        prep,
        (x_q, x_scale),
        w_q,
        w_scale,
        out_dtype,
        tile_bytes,
        "fp8_decode_prequantized_matmul",
        interpret,
    )


def supports_nt(rows: int, k: int, n: int, itemsize: int = 2) -> bool:
    if not (rows % 8 == 0 and n % _LANES == 0 and k % _LANES == 0):
        return False
    tn, depth, _ = tile_config(k, n, itemsize=itemsize, tile_bytes=_NT_TILE_BYTES)
    ring = depth * tn * k * itemsize
    out_bytes = rows * n * itemsize + rows * tn * 4
    return _fits_vmem(ring + rows * k * itemsize + out_bytes, _MAX_NT_VMEM_BYTES)


@functools.partial(jax.jit, static_argnames=("tile_bytes", "interpret"))
def decode_matmul_nt(
    x: jax.Array,
    w: jax.Array,
    tile_bytes: int = _NT_TILE_BYTES,
    interpret: bool = False,
) -> jax.Array:
    m, k = x.shape
    n = w.shape[0]
    tn, depth, _ = tile_config(k, n, itemsize=w.dtype.itemsize, tile_bytes=tile_bytes)
    nt = n // tn

    def kernel(x_ref, w_hbm, o_ref, wbuf, sems):
        def copy(t, slot):
            return pltpu.make_async_copy(
                w_hbm.at[pl.ds(t * tn, tn)], wbuf.at[slot], sems.at[slot]
            )

        for t in range(depth):
            copy(t, t).start()

        def body(t, carry):
            slot = lax.rem(t, depth)
            copy(t, slot).wait()
            acc = lax.dot_general(
                x_ref[...],
                wbuf[slot],
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            o_ref[:, pl.ds(pl.multiple_of(t * tn, _LANES), tn)] = acc.astype(
                o_ref.dtype
            )
            pl.when(t + depth < nt)(lambda: copy(t + depth, slot).start())
            return carry

        lax.fori_loop(0, nt, body, 0)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        in_specs=[_VMEM, _HBM],
        out_specs=_VMEM,
        scratch_shapes=[
            pltpu.VMEM((depth, tn, k), w.dtype),
            pltpu.SemaphoreType.DMA((depth,)),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes(_MAX_NT_VMEM_BYTES)
        ),
        name=f"decode_matmul_nt_tn{tn}_d{depth}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(x, _hbm(w))
