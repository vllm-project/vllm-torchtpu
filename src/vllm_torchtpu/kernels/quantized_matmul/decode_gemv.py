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
"""Weight-streaming matmuls for decode-sized token buckets.

`fp8_decode_matmul` / `fp8_decode_prequantized_matmul`: per-channel W8A8 FP8
projections on the canonical `(k, n)` weight. `decode_matmul_nt`: `x @ w.T`
on an `(n, k)` weight, the layout vLLM keeps for a (tied) LM head.

A decode bucket's FP8 projection reads a weight of tens of MB for a few rows,
so its time is the weight's HBM stream. XLA runs it as two activation
quantization fusions and a dot fusion whose weight pipeline fills and drains
on every call. This kernel quantizes the rows (when given bf16 rows) while the
first weight tiles are in flight, and keeps `depth` `[K, tn]` column tiles of
the weight in flight, each copied as `splits` row-chunk DMAs, until the last.

Arithmetic: `quantized_matmul.util.quantize_tensor` per row (f32 scale =
row absmax / dtype max, 1 for a zero row; rows divided by it in f32, clipped,
cast), an FP8 x FP8 dot accumulated in f32, then `* x_scale * w_scale` and a
cast, as `xla_prequantized_matmul`. The dot's accumulation order is Mosaic's,
not XLA's, so on some shapes the outputs round differently.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Weight bytes per tile: large enough that `depth` tiles keep the HBM busy,
# small enough that the ring of tiles stays a fraction of VMEM.
_TILE_BYTES = 3 * 1024 * 1024
_DEPTH = 4
_LANES = 128
# FP8 rows pack 32 to a sublane tile; the dot runs on a whole tile of rows.
_ROW_TILE = 32
_VMEM_LIMIT_BYTES = 32 * 1024 * 1024
# The LM head's 5.5 MB bf16 row tiles and its [rows, vocab shard] output.
_NT_VMEM_LIMIT_BYTES = 48 * 1024 * 1024
# Share of a limit the estimated buffers may take; the rest is left for the
# compiler's temporaries. Checked by deviceless tpu7x compiles (jax 0.11.2,
# libtpu 0.0.50) of the largest buckets it admits at Gemma4-31B TP2; recheck
# for shapes far from those.
_VMEM_BUDGET = 0.75


def tile_config(
    k: int, n: int, itemsize: int = 1, tile_bytes: int = _TILE_BYTES
) -> tuple[int, int, int]:
    """`(tn, depth, splits)` for a `[k, n]` weight: the widest whole-lane
    divisor of `n` whose tile fits `tile_bytes`, and two row chunks per tile
    when `k` halves into whole FP8 row tiles."""
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


def supports(rows: int, k: int, n: int, w_dtype: jnp.dtype, w_scale_ndim: int) -> bool:
    """Whether `[rows, k] @ [k, n]` with FP8 weights and a per-channel scale
    can run here: whole lanes of output columns, whole FP8 row tiles of k, and
    the weight ring, the rows, their f32 copy and the whole output in VMEM."""
    if not (
        jnp.issubdtype(w_dtype, jnp.floating)
        and jnp.dtype(w_dtype).itemsize == 1
        and w_scale_ndim == 1
        and rows % 16 == 0
        and n % _LANES == 0
        and k % _ROW_TILE == 0
    ):
        return False
    tn, depth, _ = tile_config(k, n)
    mp = -(-rows // _ROW_TILE) * _ROW_TILE
    rows_bytes = rows * k * 2 + mp * k * (1 + 4)  # bf16 in, FP8 + f32 copies
    out_bytes = rows * n * 2 + n * 4 + mp * tn * 4  # bf16 out, scales, f32 tile
    return _fits_vmem(depth * k * tn + rows_bytes + out_bytes, _VMEM_LIMIT_BYTES)


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
    """o = ((xq @ w) * xs * ws)[:rows], one column tile at a time; the first
    `depth` tiles' copies and the `scales` copy were issued by
    `_start_copies`."""
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
        # Issued ahead of the tiles, the scales have landed with the first.
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
    """Issue the column scales, then the first `depth` weight tiles."""
    # The scales go first: queued behind the tiles they would land last, and
    # the first tile's epilogue reads them.
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
    # Left unconstrained, XLA may stage a whole weight in VMEM before the call
    # and wait for it there; the kernel streams it from HBM itself.
    return pltpu.with_memory_space_constraint(x, pltpu.HBM)


def _fp8_call(
    kernel, args, *, m, k, n, tn, depth, splits, w_dtype, out_dtype, name, interpret
):
    """The FP8 kernels' pallas_call. `args` ends with the weight and its
    column scales, which stay in HBM; the operands before them are VMEM."""
    mp = -(-m // _ROW_TILE) * _ROW_TILE
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        in_specs=[vmem] * (len(args) - 2) + [hbm, hbm],
        out_specs=vmem,
        scratch_shapes=[
            pltpu.VMEM((depth, k, tn), w_dtype),
            pltpu.SemaphoreType.DMA((depth, splits)),
            pltpu.VMEM((n,), jnp.float32),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.VMEM((mp, k), w_dtype),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        name=f"{name}_tn{tn}_d{depth}_s{splits}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(*args)


@functools.partial(jax.jit, static_argnames=("tile_bytes", "interpret"))
def fp8_decode_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    tile_bytes: int = _TILE_BYTES,
    interpret: bool = False,
) -> jax.Array:
    """`xla_quantized_matmul(x, w_q, w_scale)` for few rows: `x` `[m, k]`
    (m a multiple of 16), `w_q` FP8 `[k, n]`, `w_scale` f32 `[n]`."""
    m, k = x.shape
    n = w_q.shape[1]
    tn, depth, splits = tile_config(k, n, tile_bytes=tile_bytes)
    mp = -(-m // _ROW_TILE) * _ROW_TILE

    def kernel(x_ref, w_hbm, ws_hbm, o_ref, wbuf, sems, ws_vmem, ws_sem, xq_ref):
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
        q, xs = _quantize_rows(_pad_rows(x_ref[...], mp), w_q.dtype)
        xq_ref[...] = q
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

    return _fp8_call(
        kernel,
        (x, _hbm(w_q), _hbm(w_scale.astype(jnp.float32))),
        m=m,
        k=k,
        n=n,
        tn=tn,
        depth=depth,
        splits=splits,
        w_dtype=w_q.dtype,
        out_dtype=x.dtype,
        name="fp8_decode_matmul",
        interpret=interpret,
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
    """`xla_prequantized_matmul(x_q, x_scale, w_q, w_scale, out_dtype)` for few
    rows: `x_q` FP8 `[m, k]` with f32 `[m, 1]` scales."""
    m, k = x_q.shape
    n = w_q.shape[1]
    tn, depth, splits = tile_config(k, n, tile_bytes=tile_bytes)
    mp = -(-m // _ROW_TILE) * _ROW_TILE

    def kernel(
        xq_in, xs_ref, w_hbm, ws_hbm, o_ref, wbuf, sems, ws_vmem, ws_sem, xq_ref
    ):
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
        xq_ref[pl.ds(0, m)] = xq_in[...]
        if mp > m:
            xq_ref[pl.ds(m, mp - m)] = jnp.zeros((mp - m, k), xq_ref.dtype)
        xs = xs_ref[...].astype(jnp.float32)
        if mp > m:
            xs = jnp.concatenate([xs, jnp.ones((mp - m, 1), jnp.float32)], axis=0)
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

    return _fp8_call(
        kernel,
        (x_q, x_scale, _hbm(w_q), _hbm(w_scale.astype(jnp.float32))),
        m=m,
        k=k,
        n=n,
        tn=tn,
        depth=depth,
        splits=splits,
        w_dtype=w_q.dtype,
        out_dtype=out_dtype,
        name="fp8_decode_prequantized_matmul",
        interpret=interpret,
    )


def supports_nt(rows: int, k: int, n: int, itemsize: int = 2) -> bool:
    """Whether `[rows, k] @ [n, k].T` can run in `decode_matmul_nt`: whole
    lanes, and the weight ring, the rows and the whole output in VMEM."""
    if not (rows % 8 == 0 and n % _LANES == 0 and k % _LANES == 0):
        return False
    tn, depth, _ = tile_config(k, n, itemsize=itemsize, tile_bytes=_NT_TILE_BYTES)
    ring = depth * tn * k * itemsize
    out_bytes = rows * n * itemsize + rows * tn * 4
    return _fits_vmem(ring + rows * k * itemsize + out_bytes, _NT_VMEM_LIMIT_BYTES)


# Row tiles of a bf16 LM head: 512 rows of a 5376-wide hidden size.
_NT_TILE_BYTES = 2 * _TILE_BYTES


@functools.partial(jax.jit, static_argnames=("tile_bytes", "interpret"))
def decode_matmul_nt(
    x: jax.Array,
    w: jax.Array,
    tile_bytes: int = _NT_TILE_BYTES,
    interpret: bool = False,
) -> jax.Array:
    """`x @ w.T` accumulated in f32 and cast to `x.dtype`, streaming `w`
    (`[n, k]`) as `[tn, k]` row tiles: `x` `[m, k]` (m a multiple of 8)."""
    m, k = x.shape
    n = w.shape[0]
    # A [tn, k] row tile holds as many bytes as a [k, tn] column tile.
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
            col = pl.multiple_of(t * tn, _LANES)
            o_ref[:, pl.ds(col, tn)] = acc.astype(o_ref.dtype)

            @pl.when(t + depth < nt)
            def _():
                copy(t + depth, slot).start()

            return carry

        lax.fori_loop(0, nt, body, 0)

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        in_specs=[vmem, pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=vmem,
        scratch_shapes=[
            pltpu.VMEM((depth, tn, k), w.dtype),
            pltpu.SemaphoreType.DMA((depth,)),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=_NT_VMEM_LIMIT_BYTES),
        name=f"decode_matmul_nt_tn{tn}_d{depth}",
        interpret=pltpu.InterpretParams() if interpret else False,
    )(x, _hbm(w))
