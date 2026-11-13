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
"""FP8 matmul for a few activation rows, with the weight streamed from HBM.

The kernel computes ``x @ (w_q * w_s)`` for a per-channel FP8 weight: ``x`` is a
bf16 activation of at most 16 rows (``MAX_M``), ``w_q`` is an FP8 E4M3 weight in
K-major ``[k, n]`` layout, and ``w_s`` is one float32 scale per output column.
With so few rows, the call is bound by the time to read the weight from HBM. A
compiler that copies the whole weight into VMEM before the first MXU push, or
that streams it in a few windows with a fixed cost each, can take longer than
that read time. This kernel keeps the weight in HBM and reads it through a
small VMEM ring with several DMAs in flight, in one grid step. It computes each
weight tile as soon as the tile arrives.

Numerics: the kernel quantizes the activation per row with the formula of
``util.quantize_tensor``, as XLA compiles it:
scale = f32(amax|x|) * f32(1 / fp8_max), with scale = 1 where it is 0, and
x_q = fp8(clip(f32(x) * (1 / scale))). The x_q and scale values are therefore
bit for bit those of ``util.quantize_tensor``. The kernel then holds x_q as
bf16, which represents every E4M3 value exactly, and runs a bf16 x fp8 MXU dot
with float32 accumulation. XLA's fp8 x fp8 dot accumulates with less precision
than float32. In the accuracy test against XLA on TPU v7x (three weight shapes,
1 to 16 rows), the largest per-row error relative to the row maximum, against
the float64 product of the same quantized operands, was 0.66 % to 1.1 % for XLA
and 0.22 % to 0.37 % for this kernel. The output of this kernel therefore
differs from ``util.xla_quantized_matmul`` and is closer to the float64 product.

Tilings. Both multiply in NN form on the K-major ``[k, n]`` weight, which needs
``n % 128 == 0``. A weight whose width is not a multiple of 128 is padded once
with zero columns and unit scales (``padded_n``), and the caller cuts the
output back with ``n_logical``.

* ``'n'``: tile j is the column window ``W[:, j*t:(j+1)*t]``. Each tile writes
  its own output columns once.
* ``'k'``: tile j is the row window ``W[j*t:(j+1)*t, :]``, which is contiguous
  in HBM. A float32 accumulator in VMEM sums the tiles.

An N-major ``[n, k]`` weight would need an NT dot. Its transposed tile goes
through the XLU before the MXU push, which costs about as much as the weight
read itself, so the kernel takes only the K-major layout.

The weight ring follows the pattern of tokamax ``fused_moe_rs``. The prologue
starts ``nbuf`` tile copies. In the steady state, step j waits for slot
j % nbuf, computes tile j, and starts the copy of tile j + nbuf into the same
slot. The tile loop is unrolled in Python, so the slots and the lane offsets are
static.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MAX_M = 16
LANE = 128
# Fraction of the chip's VMEM that one call may plan to use. The rest stays free
# for Mosaic's internal scratch and for the XLA ops around the call. The kernels
# in kernels/experimental/moe_tp_decode use the same fraction.
VMEM_FRACTION = 0.75
# Sublane tile of the narrowest dtype here: 8 sublanes x 4 FP8 values per 32-bit
# word. Rounding every row count up to it gives an upper bound of the VMEM bytes.
_ROW_TILE = 32

# Per-shape setting: (k, n) -> (split, tile, nbuf). `split` is the tiling ('n' or
# 'k', see the module docstring), `tile` the tile extent along that axis, and
# `nbuf` the number of tile copies in flight. Each entry is the fastest setting
# measured on TPU v7x for that shape with a weight that is not in any cache. No
# other TPU generation was measured. When two settings tie, the entry is the one
# with the smaller ring. `supports` routes only these shapes to the kernel; other
# shapes stay on XLA.
TUNED = {
    (6144, 2688): ("k", 1024, 3),
    (2048, 16384): ("n", 1024, 4),
    (1024, 6144): ("n", 2048, 3),
    (6144, 1536): ("k", 1536, 3),
}


def pick_config(k: int, n: int) -> tuple[str, int, int]:
    """Returns the tiling for a ``[k, n]`` weight.

    Args:
      k: Input features (rows of the K-major weight).
      n: Output features (columns of the weight), a multiple of 128.

    Returns:
      ``(split, tile, nbuf)``. ``TUNED[(k, n)]`` when the shape is in the
      table. Otherwise an ``'n'`` tiling with 3 copies in flight and the widest
      tile of 2048, 1024, 512, 256 or 128 columns that divides ``n`` into at
      least 3 tiles. When no tile qualifies, one tile of all ``n`` columns.
      Example: ``pick_config(512, 640)`` returns ``('n', 128, 3)``, because
      640 is not a multiple of 256 or wider tiles and 640 / 128 = 5 tiles.
    """
    if (k, n) in TUNED:
        return TUNED[(k, n)]
    for t in (2048, 1024, 512, 256, 128):
        if n % t == 0 and n // t >= 3:
            return ("n", t, 3)
    return ("n", n, 1)


def supports(m: int, w_q: jax.Array, w_s: jax.Array) -> bool:
    """Returns whether the kernel takes this call.

    A caller that falls back to XLA otherwise uses this check. The kernel
    computes any shape that ``pick_config`` gives a setting for, but this check
    accepts only shapes in ``TUNED``: the default ring of an untuned shape can
    exceed VMEM, and it is not measured to be faster than XLA.

    ``TUNED`` was measured on TPU v7x only, and this check does not read the TPU
    generation. A caller on another generation must gate the call itself.

    Args:
      m: Number of activation rows.
      w_q: Weight, expected FP8 E4M3 ``[k, n]``.
      w_s: Weight scale, expected float32 ``[n]`` (one per output column).

    Returns:
      True when ``m <= MAX_M``, ``w_q`` is a 2-D FP8 E4M3 array whose shape is
      in ``TUNED``, and ``w_s`` is 1-D with one value per column of ``w_q``.
    """
    return (
        m <= MAX_M
        and w_q.ndim == 2
        and w_s.ndim == 1
        and w_q.dtype == jnp.float8_e4m3fn
        and tuple(w_q.shape) in TUNED
        and w_s.shape[0] == w_q.shape[1]
    )


def _kernel(
    idx_ref,
    x_ref,
    ws_ref,
    w_hbm,
    o_ref,
    xq_ref,
    xs_ref,
    wbuf,
    sem,
    *acc,
    split,
    t,
    n_tiles,
    nbuf,
    q_dtype,
):
    # Index of the weight in the stack: 0 for a single weight (a [1, k, n] view).
    s = idx_ref[0]

    def copy(j):
        if split == "n":
            src = w_hbm.at[s, :, pl.ds(j * t, t)]
        else:
            src = w_hbm.at[s, pl.ds(j * t, t), :]
        return pltpu.make_async_copy(src, wbuf.at[j % nbuf], sem.at[j % nbuf])

    for j in range(min(nbuf, n_tiles)):
        copy(j).start()

    q_max = float(jnp.finfo(q_dtype).max)
    x = x_ref[...]
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(
        jnp.float32
    ) * jnp.float32(1.0 / q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    x_q = jnp.clip(
        x.astype(jnp.float32) * (jnp.float32(1.0) / scale), -q_max, q_max
    ).astype(q_dtype)
    xq_ref[...] = x_q.astype(xq_ref.dtype)
    xs_ref[...] = scale

    dn = (((1,), (0,)), ((), ()))
    for j in range(n_tiles):
        copy(j).wait()
        w = wbuf[j % nbuf]
        if split == "n":
            cols = slice(j * t, (j + 1) * t)
            y = jax.lax.dot_general(
                xq_ref[...], w, dn, preferred_element_type=jnp.float32
            )
            o_ref[:, cols] = (y * xs_ref[...] * ws_ref[:, cols]).astype(o_ref.dtype)
        else:
            y = jax.lax.dot_general(
                xq_ref[:, j * t : (j + 1) * t],
                w,
                dn,
                preferred_element_type=jnp.float32,
            )
            if j == 0:
                acc[0][...] = y
            else:
                acc[0][...] += y
        if j + nbuf < n_tiles:
            copy(j + nbuf).start()
    if split == "k":
        o_ref[...] = (acc[0][...] * xs_ref[...] * ws_ref[...]).astype(o_ref.dtype)


def _vmem_bytes(m, k, n, out_dtype, split, t, nbuf) -> int:
    """Returns an upper bound of the VMEM bytes that one call allocates.

    Each buffer is counted with its rows rounded up to ``_ROW_TILE`` and its
    columns rounded up to 128 lanes. The input and output blocks count twice,
    because the Pallas pipeline allocates two buffers per block. Example: the
    ``TUNED`` setting ``('k', 1024, 3)`` of ``k, n = 6144, 2688`` with 16 bf16
    rows needs 10,829,824 bytes (about 10.3 MiB), 8,257,536 of them for the
    three weight tiles.

    Args:
      m: Activation rows.
      k: Input features.
      n: Output features, a multiple of 128.
      out_dtype: Output dtype.
      split: Tiling, ``'n'`` or ``'k'``.
      t: Tile extent along the split axis.
      nbuf: Number of weight tiles in VMEM at the same time.

    Returns:
      The estimated number of bytes.
    """

    def size(rows, cols, itemsize):
        return -(-rows // _ROW_TILE) * _ROW_TILE * padded_n(cols) * itemsize

    tile = (k, t) if split == "n" else (t, n)
    need = (
        2 * size(m, k, 2)  # x block
        + 2 * size(1, n, 4)  # weight scale block
        + 2 * size(m, n, jnp.dtype(out_dtype).itemsize)  # output block
        + size(m, k, 2)  # x_q as bf16
        + size(m, 1, 4)  # activation scale
        + nbuf * size(*tile, 1)  # weight ring
    )
    if split == "k":
        need += size(m, n, 4)  # float32 accumulator
    return need


@functools.cache
def _build(m, k, n, n_stack, w_dtype, out_dtype, split, t, nbuf):
    full = k if split == "k" else n
    if full % t or t % LANE:
        raise ValueError(
            f"tile {t} does not divide {split}-extent {full} in lane multiples"
        )
    n_tiles = full // t
    nbuf = max(1, min(nbuf, n_tiles))
    need = _vmem_bytes(m, k, n, out_dtype, split, t, nbuf)
    capacity = pltpu.get_tpu_info().vmem_capacity_bytes
    if need > int(capacity * VMEM_FRACTION):
        raise ValueError(
            f"setting ({split!r}, {t}, {nbuf}) for weight [{k}, {n}] needs about "
            f"{need} bytes of VMEM, more than {VMEM_FRACTION} of the {capacity} "
            "bytes of this chip; pass a config with a smaller tile or fewer copies"
        )
    tile = (k, t) if split == "n" else (t, n)
    scratch = [
        pltpu.VMEM((m, k), jnp.bfloat16),
        pltpu.VMEM((m, 1), jnp.float32),
        pltpu.VMEM((nbuf, *tile), w_dtype),
        pltpu.SemaphoreType.DMA((nbuf,)),
    ]
    if split == "k":
        scratch.append(pltpu.VMEM((m, n), jnp.float32))
    kern = functools.partial(
        _kernel, split=split, t=t, n_tiles=n_tiles, nbuf=nbuf, q_dtype=w_dtype
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(1,),
        in_specs=[
            pl.BlockSpec((m, k), lambda g, idx: (0, 0)),
            pl.BlockSpec((1, n), lambda g, idx: (0, 0)),
            # The weight stack [s, k, n] stays in HBM, so the compiler cannot
            # move it into VMEM; the kernel copies its tiles.
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec((m, n), lambda g, idx: (0, 0)),
        scratch_shapes=scratch,
    )
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n), out_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"fp8_skinny_{split}{t}_b{nbuf}_m{m}_k{k}_n{n}",
    )


def fp8_skinny_matmul(
    x: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    *,
    n_logical: int | None = None,
    config: tuple[str, int, int] | None = None,
) -> jax.Array:
    """Computes ``quantize(x) @ (w_q * w_s)`` with the weight streamed from HBM.

    Example: ``x`` bf16 ``[4, 1024]``, ``w_q`` FP8 ``[1024, 6144]`` and ``w_s``
    float32 ``[6144]`` take the ``TUNED`` setting ``('n', 2048, 3)``: three
    column tiles of ``[1024, 2048]`` (2 MiB each), all three copies in flight.
    The result is bf16 ``[4, 6144]``. For a weight of 2624 columns, the caller
    passes the copy padded to ``padded_n(2624) = 2688`` columns with
    ``n_logical=2624`` and gets ``[4, 2624]``.

    Args:
      x: Activation ``[m, k]``, bf16, with ``m <= MAX_M``.
      w_q: FP8 E4M3 weight ``[k, n]`` (K-major), ``n % 128 == 0``.
      w_s: Float32 weight scale ``[n]``, one value per output column.
      n_logical: Number of output columns to return. Use it with a weight that
        was padded by ``padded_n``. None returns all ``n`` columns.
      config: ``(split, tile, nbuf)`` to use instead of ``pick_config(k, n)``.

    Returns:
      ``[m, n_logical or n]`` in ``x.dtype``. A row of ``x`` that is all zeros
      gives a row of zeros.

    Raises:
      ValueError: If ``x`` is not bf16 or does not have 1 to ``MAX_M`` rows, if
        ``w_q`` is not FP8 E4M3, if the ``k`` of ``x`` and ``w_q`` differ, if
        ``n`` is not a multiple of 128, or if ``w_s`` is not of shape ``[n]``.
        Also if the tile does not divide the split extent (``n`` for ``'n'``,
        ``k`` for ``'k'``) or is not a multiple of 128, or if the setting needs
        more than ``VMEM_FRACTION`` of the chip's VMEM (for example the default
        setting of a shape outside ``TUNED``).
    """
    return fp8_skinny_matmul_stacked(
        x,
        w_q[None],
        jnp.zeros((1,), jnp.int32),
        w_s,
        n_logical=n_logical,
        config=config,
    )


def fp8_skinny_matmul_stacked(x, w_stack, idx, w_s, *, n_logical=None, config=None):
    """Same as ``fp8_skinny_matmul``, with the weight selected inside the kernel.

    The kernel reads the weight ``w_stack[idx[0]]`` directly from HBM. A
    benchmark can therefore cycle through several weights, so that each call
    reads a weight that is not in any cache.

    Args:
      x: Activation ``[m, k]``, bf16, with ``m <= MAX_M``.
      w_stack: FP8 E4M3 weights ``[s, k, n]``, ``n % 128 == 0``.
      idx: int32 ``[1]``, the index of the weight in ``w_stack``.
      w_s: Float32 weight scale ``[n]`` of the selected weight.
      n_logical: As in ``fp8_skinny_matmul``.
      config: As in ``fp8_skinny_matmul``.

    Returns:
      ``[m, n_logical or n]`` in ``x.dtype``.

    Raises:
      ValueError: As in ``fp8_skinny_matmul``.
    """
    m, k = x.shape
    n_stack, k_w, n = w_stack.shape
    if not 1 <= m <= MAX_M:
        raise ValueError(f"x has {m} rows; the kernel takes 1 to {MAX_M} rows")
    if x.dtype != jnp.bfloat16:
        raise ValueError(f"x must be bfloat16, got {x.dtype}")
    if w_stack.dtype != jnp.float8_e4m3fn:
        raise ValueError(f"weight must be float8_e4m3fn, got {w_stack.dtype}")
    if k_w != k:
        raise ValueError(f"x has k = {k} columns, but the weight has {k_w} rows")
    if n % LANE:
        raise ValueError(
            f"weight width {n} is not a multiple of {LANE}; pass a copy padded "
            "to padded_n(n) columns with n_logical"
        )
    if tuple(w_s.shape) != (n,):
        raise ValueError(
            f"weight scale must have shape ({n},), one value per output column, "
            f"got {tuple(w_s.shape)}"
        )
    split, t, nbuf = config or pick_config(k, n)
    call = _build(m, k, n, n_stack, w_stack.dtype, x.dtype, split, t, nbuf)
    out = call(idx, x, w_s.reshape(1, n).astype(jnp.float32), w_stack)
    if n_logical is not None and n_logical != n:
        out = out[:, :n_logical]
    return out


def padded_n(n: int, multiple: int = LANE) -> int:
    """Returns the width to pad a weight of ``n`` columns to for this kernel.

    The padded columns hold zero weights and unit scales, so their outputs are
    zero. Example: ``padded_n(2624) == 2688`` and ``padded_n(6144) == 6144``.

    Args:
      n: Output columns of the weight.
      multiple: Column multiple to pad to; 128 lanes by default.

    Returns:
      The smallest multiple of ``multiple`` that is at least ``n``.
    """
    return -(-n // multiple) * multiple
