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
"""Greedy draft tokens with the LM head's columns split over the DP group.

Every DP engine holds the same vocab-parallel LM head, so each draft propose
reads the rank's whole TP shard [hidden, cols] for its engine's few rows. Here
the ranks of one TP column (one per engine) share that read: DP rank ``d``
scores columns ``[d * cols / dp, (d + 1) * cols / dp)`` of its shard for the
rows of every engine (one DP all-gather of the rows) and keeps each row's best
logit and its first column. One DP all-to-all returns each engine its rows'
pairs, the TP all-gather of the vocab-parallel argmax adds the other shards,
and the first maximum in vocabulary order is the token.

The kernel sums the hidden dimension the way XLA's dot does for the whole shard
in the vocab-parallel argmax program at the same row count: consecutive
windows of input features, each window's MXU sum added in order in FP32, then
one bf16 rounding (``xla_head_window``), so that each logit keeps the bits of
the vocab-parallel head's; ties resolve to the lowest id, as there.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

DP_AXIS = "dp"
TP_AXIS = "tp"
_LANES = 128
# Columns of one weight block: 4 KiB of bf16 per hidden row and DMA.
_BLOCK_COLS = 2048
_NO_COLUMN = 2**30
# Scoped VMEM the kernel may take; row counts above it keep the vocab-parallel
# argmax.
_VMEM_BUDGET = 40 * 1024 * 1024

# Input features per window of XLA's dot in the vocab-parallel argmax program
# (hidden-major projection fused with the masked maximum and first index) on
# tpu7x with libtpu 0.0.47, by (rows, hidden, shard columns).
_XLA_HEAD_WINDOWS = {
    **{(rows, 8192, 62080): 1024 for rows in range(1, 41)},
    **{(rows, 8192, 62080): 512 for rows in range(41, 81)},
    **{(rows, 8192, 62080): 1024 for rows in range(81, 129)},
}


def xla_head_window(rows: int, hidden: int, cols: int) -> int | None:
    """Input features per window of XLA's head dot at this shape, or None
    when it is not known."""
    return _XLA_HEAD_WINDOWS.get((rows, hidden, cols))


def read_columns(cols: int, dp_size: int) -> int:
    """Columns each rank reads: its slice plus up to 127 columns that align
    the first one to a lane tile, in whole weight blocks."""
    return pl.cdiv(cols // dp_size + _LANES - 1, _BLOCK_COLS) * _BLOCK_COLS


def _vmem_bytes(rows: int, hidden: int, window: int) -> int:
    weights = 2 * window * _BLOCK_COLS * 2
    inputs = 2 * rows * hidden * 2
    partials = 2 * rows * _BLOCK_COLS * 4
    return weights + inputs + partials + 4 * rows * _LANES * 4


def supported(rows: int, hidden: int, cols: int, dp_size: int) -> bool:
    """Whether a propose of ``rows`` draft rows per engine takes the split."""
    window = xla_head_window(rows, hidden, cols)
    return (
        window is not None
        and dp_size > 1
        and cols % dp_size == 0
        and cols % _LANES == 0
        and read_columns(cols, dp_size) <= cols
        and _vmem_bytes(dp_size * rows, hidden, window) <= _VMEM_BUDGET
    )


def _slice_argmax_kernel(
    meta_ref, rows_ref, w_ref, out_ref, acc_ref, best_ref, col_ref, *, num_windows
):
    """Grid (column block n, input window k). meta: first column read, the
    slice's start and end and the vocabulary end, relative to the first."""
    n, k = pl.program_id(0), pl.program_id(1)
    part = jnp.dot(rows_ref[k], w_ref[...], preferred_element_type=jnp.float32)

    @pl.when(k == 0)
    def _():
        acc_ref[...] = part

    @pl.when(k > 0)
    def _():
        acc_ref[...] = acc_ref[...] + part

    @pl.when(k == num_windows - 1)
    def _():
        @pl.when(n == 0)
        def _():
            best_ref[...] = jnp.full(best_ref.shape, -jnp.inf, jnp.float32)
            col_ref[...] = jnp.full(col_ref.shape, _NO_COLUMN, jnp.int32)

        lo, hi, vocab_end = meta_ref[1], meta_ref[2], meta_ref[3]
        lane = jax.lax.broadcasted_iota(jnp.int32, best_ref.shape, 1)
        best, best_col = best_ref[...], col_ref[...]
        # Per lane, the first maximum in column order; vocabulary columns at
        # or past the end count as -inf, as in the vocab-parallel argmax.
        for j in range(_BLOCK_COLS // _LANES):
            tile = acc_ref[:, j * _LANES : (j + 1) * _LANES]
            value = tile.astype(jnp.bfloat16).astype(jnp.float32)
            col = lane + (n * _BLOCK_COLS + j * _LANES)
            value = jnp.where(col < vocab_end, value, -jnp.inf)
            take = (
                (col >= lo) & (col < hi) & ((value > best) | (best_col == _NO_COLUMN))
            )
            best = jnp.where(take, value, best)
            best_col = jnp.where(take, col, best_col)
        best_ref[...] = best
        col_ref[...] = best_col

        @pl.when(n == pl.num_programs(0) - 1)
        def _():
            seen = best_col < _NO_COLUMN
            top = jnp.max(jnp.where(seen, best, -jnp.inf), axis=1, keepdims=True)
            first = jnp.min(
                jnp.where(seen & (best == top), best_col, _NO_COLUMN),
                axis=1,
                keepdims=True,
            )
            out_ref[...] = jnp.concatenate([top, first.astype(jnp.float32)], axis=1)


def slice_argmax(
    rows: jax.Array,
    w: jax.Array,
    meta: jax.Array,
    *,
    read_cols: int,
    interpret: bool = False,
) -> jax.Array:
    """Each row's best bf16 logit over one rank's columns of an LM head shard.

    ``rows`` [num_windows, R, window] are the rows split into input-feature
    windows, ``w`` [hidden, cols] the hidden-major shard, ``meta`` int32 [4]:
    the first column read (a multiple of 128, ``read_cols`` columns from
    there), then the slice's start and end and the vocabulary end relative to
    it. Returns [R, 2] float32: the maximum over the slice and its first
    column, relative to the first column read.
    """
    num_windows, num_rows, window = rows.shape
    num_blocks = read_cols // _BLOCK_COLS
    return pl.pallas_call(
        functools.partial(_slice_argmax_kernel, num_windows=num_windows),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_blocks, num_windows),
            in_specs=[
                pl.BlockSpec(rows.shape, lambda n, k, meta: (0, 0, 0)),
                pl.BlockSpec(
                    (pl.Element(window), pl.Element(_BLOCK_COLS)),
                    lambda n, k, meta: (
                        k * window,
                        pl.multiple_of(meta[0] + n * _BLOCK_COLS, _LANES),
                    ),
                ),
            ],
            out_specs=pl.BlockSpec((num_rows, 2), lambda n, k, meta: (0, 0)),
            scratch_shapes=[
                pltpu.VMEM((num_rows, _BLOCK_COLS), jnp.float32),
                pltpu.VMEM((num_rows, _LANES), jnp.float32),
                pltpu.VMEM((num_rows, _LANES), jnp.int32),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((num_rows, 2), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=_vmem_bytes(num_rows, num_windows * window, window)
            + 8 * 1024 * 1024,
        ),
        interpret=interpret,
        name="draft_slice_argmax",
    )(meta, rows, w)


def draft_argmax_local(
    x: jax.Array,
    w: jax.Array,
    slice_start: jax.Array,
    vocab_start: jax.Array,
    vocab_end: jax.Array,
    *,
    dp_size: int,
    tp_size: int,
    interpret: bool = False,
) -> jax.Array:
    """One rank's part: its engine's draft rows ``x`` [p, hidden] -> their
    greedy tokens [p] int32.

    ``w`` [hidden, cols] is the rank's TP shard (global ids ``vocab_start +
    column``, valid below ``vocab_end``), ``slice_start`` [1] its first column
    ``dp_rank * cols / dp``, passed as data: the multi-process SPMD compile
    does not lower a partition id.
    """
    num_rows, hidden = x.shape
    cols = w.shape[1]
    window = xla_head_window(num_rows, hidden, cols)
    assert window is not None and hidden % window == 0, (num_rows, hidden, cols)
    # [windows, rows, window], every engine's rows in DP-rank order.
    rows = x.reshape(num_rows, hidden // window, window).transpose(1, 0, 2)
    if dp_size > 1:
        rows = jax.lax.all_gather(rows, DP_AXIS, axis=1, tiled=True)
    read_cols = read_columns(cols, dp_size)
    start = slice_start[0]
    first = jnp.minimum(start // _LANES * _LANES, cols - read_cols)
    meta = jnp.stack(
        [
            first,
            start - first,
            start - first + cols // dp_size,
            vocab_end[0] - vocab_start[0] - first,
        ]
    ).astype(jnp.int32)
    best = slice_argmax(rows, w, meta, read_cols=read_cols, interpret=interpret)
    ids = best[:, 1].astype(jnp.int32) + first + vocab_start[0]
    pairs = jnp.stack([best[:, 0], ids.astype(jnp.float32)], axis=-1)
    pairs = pairs.reshape(dp_size, num_rows, 2)
    if dp_size > 1:
        # Engine e's rows go to DP rank e; slice d's pairs arrive as chunk d.
        pairs = jax.lax.all_to_all(pairs, DP_AXIS, 0, 0, tiled=True)
    pairs = jax.lax.all_gather(pairs, TP_AXIS) if tp_size > 1 else pairs[None]
    # [rows, tp * dp, 2] in vocabulary order: the first maximum is the
    # lowest id among the tied ones.
    pairs = pairs.transpose(2, 0, 1, 3).reshape(num_rows, tp_size * dp_size, 2)
    pick = jnp.argmax(pairs[..., 0], axis=-1)
    token = jnp.take_along_axis(pairs[..., 1], pick[:, None], axis=-1)[:, 0]
    return token.astype(jnp.int32)


def draft_argmax(
    x: jax.Array,
    w: jax.Array,
    slice_start: jax.Array,
    vocab_start: jax.Array,
    vocab_end: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    interpret: bool = False,
) -> jax.Array:
    """Global-view entry: the ranks' local arrays stacked on two leading
    axes, x [dp, tp, p, hidden], w [dp, tp, hidden, cols] and the three
    int32 [dp, tp, 1]; the result is [dp, tp, p] (equal over each engine's
    TP ranks)."""
    spec = P(DP_AXIS, TP_AXIS)
    local = functools.partial(
        draft_argmax_local,
        dp_size=mesh.shape[DP_AXIS],
        tp_size=mesh.shape[TP_AXIS],
        interpret=interpret,
    )

    def body(x, w, s, a, b):
        return local(x[0, 0], w[0, 0], s[0, 0], a[0, 0], b[0, 0])[None, None]

    mapped = jax.shard_map(
        body, mesh=mesh, in_specs=(spec,) * 5, out_specs=spec, check_vma=False
    )
    return mapped(x, w, slice_start, vocab_start, vocab_end)
