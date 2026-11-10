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
"""In-place row writes into the packed MSA caches.

An XLA scatter into the packed layout ``[pages, S / p, p, W]`` cannot index
the packed axis in place: XLA relayouts the whole cache to ``[rows, W]`` and
back around it, two full-cache copies per cache per layer. This kernel
instead rewrites only the pages that receive rows. A step of a token run
(consecutive tokens of one request landing in one page) reads the page as
32-bit words, merges the run's rows into their packing slots, and writes the
page back over the aliased cache.

All the caches a step writes at the same slots (K/V rows, index rows, the
4-bit scale page) go through one kernel: one grid over the runs, one page
of each cache per step (`write_fp4_caches`, `write_rows_many`).
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4

_UINT_BY_BITS = {16: jnp.uint16, 8: jnp.uint8}

# Steps of at most this many tokens (decode, MTP verify, small prefill chunks)
# write token by token, rows and 4-bit scale pages alike
# (`_rows_by_token`): gathering max_runs x page_size rows per layer cost
# 6 us per layer for 8 tokens in the b1 decode twin (j-3847708e), whatever
# their width, and 0.12-0.17 ms per layer for the 312-456-token decode
# batches of c48 / c64 (j-fd793d3b, j-7df0161c). A run then costs one merge
# per token instead of one per page.
_BY_TOKEN_MAX_TOKENS = 512
# Lanes of the per-row scale bits `_rows_by_page` reads (one f32 tile).
_SCALE_BLOCK_LANES = 128
# Rows of a 32-bit sublane tile: by-page row blocks start at a multiple.
_ROW_TILE = 8


def _strided_rows(ref, start, count: int, stride: int, folds: int) -> jax.Array:
    """Rows ``start, start + stride, ...`` (`count` of them) of a 32-bit
    ``[R, 128 * folds]`` array held as ``[R * folds, 128]`` in `ref` (row
    ``r``'s lanes ``[128 i, 128 (i + 1))`` at ref row ``r * folds + i``):
    Mosaic's strided loads need 128-lane rows, as in `ragged_paged_attention` v3."""
    return jnp.concatenate(
        [
            ref[pl.ds(start * folds + i, count, stride=stride * folds)]
            for i in range(folds)
        ],
        axis=1,
    )


# Merges of one run into one page. Each gets the run's first and end slot,
# its first token (by token) or first token - first slot (by page), the
# cache's data refs, the page before the write and the aliased output page.


def _rows_by_page(lo, hi, base, new_ref, *refs, bits: int, pad: int):
    """``new_ref [(S + 8) * W / 128, 128]`` uint32: the run page's slot s at
    row ``(base + pad) % 8 + s`` of the ``[S + 8, W]`` rows (`_strided_rows`);
    with folded scales then ``[S + 8, lanes]`` uint32 scale bits of slot s on
    every lane of the same row."""
    scale_ref = refs[0] if len(refs) == 3 else None
    page_ref, o_ref = refs[-2:]
    rows, _, width = page_ref.shape[1:]
    packing = 32 // bits
    words = page_ref.bitcast(jnp.uint32)[0, :, 0, :]
    row = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 0)
    data_rows = rows - 1 if scale_ref is not None else rows
    off = (base + pad) % _ROW_TILE
    for j in range(packing):
        slot = row * packing + j
        take = (slot >= lo) & (slot < hi)
        keep = jnp.uint32(~(((1 << bits) - 1) << (bits * j)) & 0xFFFFFFFF)
        # Slots j, p + j, ...: rows of the run's page at stride p.
        new = _strided_rows(new_ref, off + j, data_rows, packing, width // 128)
        if data_rows < rows:
            new = jnp.pad(new, ((0, rows - data_rows), (0, 0)))
        merged = (words & keep) | (new << jnp.uint32(bits * j))
        words = jnp.where(take, merged, words)
    if scale_ref is not None:  # last word row: one f32 scale per slot (lane)
        lanes = scale_ref.shape[1]
        # [1, lanes]: slot s's bits at lane s.
        by_slot = scale_ref[pl.ds(off, lanes), :].T[:1]
        if width > lanes:
            by_slot = jnp.pad(by_slot, ((0, 0), (0, width - lanes)))
        lane = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)
        take = (row == rows - 1) & (lane >= lo) & (lane < hi)
        words = jnp.where(take, by_slot[:, :width], words)
    o_ref.bitcast(jnp.uint32)[0, :, 0, :] = words


def _rows_by_token(lo, hi, tok0, rows_ref, *refs, bits: int):
    """`_rows_by_page` that merges a run's tokens one by one from the batch's
    rows (``rows_ref [T, W]`` uint32, one value per word; with folded scales
    then ``[T, W]`` uint32 scale bits per token), so no per-run page of rows
    is gathered first."""
    scale_ref = refs[0] if len(refs) == 3 else None
    page_ref, o_ref = refs[-2:]
    rows, _, width = page_ref.shape[1:]
    packing = 32 // bits
    row = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)

    def put(i, words):
        slot = lo + i
        shift = (slot % packing * bits).astype(jnp.uint32)
        keep = ~(jnp.uint32((1 << bits) - 1) << shift)
        new = rows_ref[pl.ds(tok0 + i, 1), :]
        words = jnp.where(
            row == slot // packing, (words & keep) | (new << shift), words
        )
        if scale_ref is not None:  # last word row: the scale at lane = slot
            take = (row == rows - 1) & (lane == slot)
            words = jnp.where(take, scale_ref[pl.ds(tok0 + i, 1), :], words)
        return words

    words = page_ref.bitcast(jnp.uint32)[0, :, 0, :]
    words = jax.lax.fori_loop(0, hi - lo, put, words)
    o_ref.bitcast(jnp.uint32)[0, :, 0, :] = words


def _slot_scales_by_token(lo, hi, tok0, values_ref, page_ref, o_ref, *, num_values):
    """Per-slot values (lane = slot) of a run's tokens, one by one, into rows
    ``[0, num_values)`` of its ``[1, rows, S]`` scale page; ``values_ref [T,
    rows, S]`` f32: every token's values, broadcast over lanes."""
    shape = page_ref.shape[1:]
    row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)

    def put(i, page):
        take = (lane == lo + i) & (row < num_values)
        return jnp.where(take, values_ref[tok0 + i], page)

    o_ref[0] = jax.lax.fori_loop(0, hi - lo, put, page_ref[0])


def _slot_scales_by_page(lo, hi, base, values_ref, page_ref, o_ref, *, num_values, pad):
    """A run's per-slot values into rows ``[0, num_values)`` of its scale
    page, all at once: ``values_ref [S + 8, 128]`` f32 holds slot s's values
    in lanes ``[0, n)`` of row ``(base + pad) % 8 + s``, transposed."""
    rows, lanes = page_ref.shape[1:]
    off = (base + pad) % _ROW_TILE
    by_slot = values_ref[pl.ds(off, lanes), :].T[:rows]  # [rows, S]
    row = jax.lax.broadcasted_iota(jnp.int32, (rows, lanes), 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, lanes), 1)
    take = (row < num_values) & (lane >= lo) & (lane < hi)
    o_ref[0] = jnp.where(take, by_slot, page_ref[0])


def _kernel(
    # Scalar prefetch
    num_runs_ref,  # [1]
    run_page_ref,  # [R + 1] page of each run (padding runs repeat the last one)
    run_lo_ref,  # [R + 1] first slot of each run within its page
    run_hi_ref,  # [R + 1] one past the last slot
    run_x_ref,  # [R + 1] by token: first token; by page: first token - first slot
    # Per cache: its data inputs, then its page before the write; then the
    # output pages, aliased with the caches.
    *refs,
    merges: tuple,  # per cache: (number of data inputs, merge)
):
    num_in = sum(n + 1 for n, _ in merges)
    ins, outs = refs[:num_in], refs[num_in:]
    run = pl.program_id(0)
    active = run < num_runs_ref[0]

    @pl.when(active)
    def _():
        lo, hi, x = run_lo_ref[run], run_hi_ref[run], run_x_ref[run]
        i = 0
        for (n, merge), o_ref in zip(merges, outs):
            merge(lo, hi, x, *ins[i : i + n + 1], o_ref)
            i += n + 1

    # With no run at all the output pages must still be written back
    # unchanged (a static grid's later padding steps revisit the last run's
    # pages and leave their already written output blocks alone).
    @pl.when(jnp.logical_and(jnp.logical_not(active), run == 0))
    def _():
        i = 0
        for (n, _), o_ref in zip(merges, outs):
            o_ref[...] = ins[i + n][...]
            i += n + 1


class _Write(NamedTuple):
    """One cache a step writes: rows (``bits`` per slot, optionally folded
    f32 `scales [T]`) or, with ``bits=None``, f32 slot values ``[T, n]``
    into rows ``[0, n)`` of a ``[pages, rows, S]`` scale page."""

    cache: jax.Array
    data: jax.Array
    bits: int | None = None
    scales: jax.Array | None = None


def _rows_operands(w: _Write, by_token: bool):
    """Data inputs, their specs and the merge of a rows write."""
    _, rows_per_page, _, width = w.cache.shape
    bits = w.bits
    packing = 32 // bits
    folded = w.scales is not None
    if folded and not (
        bits == 4 and msa_fp4.scales_in_page(rows_per_page, w.cache.dtype)
    ):
        raise ValueError(
            f"scales need a 4-bit cache with a scale row, got {w.cache.shape}"
        )
    page_size = (rows_per_page - 1 if folded else rows_per_page) * packing
    num_tokens = w.data.shape[0]
    if bits == 4:
        word_bits = w.data.astype(jnp.uint32) & 0xF
    else:
        word_bits = jax.lax.bitcast_convert_type(
            w.data, _UINT_BY_BITS[bits]
        ).astype(jnp.uint32)
    scale_bits = None
    if folded:
        scale_bits = jax.lax.bitcast_convert_type(
            w.scales.astype(jnp.float32), jnp.uint32
        )
    if by_token:
        tokens_block = pl.BlockSpec((num_tokens, width), lambda r, *_: (0, 0))
        data = [word_bits]
        if folded:  # each token's scale bits on every lane
            data.append(jnp.broadcast_to(scale_bits[:, None], (num_tokens, width)))
        merge = functools.partial(_rows_by_token, bits=bits)
        return data, [tokens_block] * len(data), merge
    # Each run reads its page's worth of rows straight from the batch, padded
    # by a page each side so that slot s of the run's page is row
    # tok0 - lo + s + S: no [runs, S, W] gather of rows (and of the scale
    # bits) is built first (XLA gathers of 0.06-0.16 ms per write at 8K
    # tokens, the 4-bit scale-bit element gather 0.1 ms, c48 twin j-b0a2e4d7).
    # Blocks start at the row tile below; the merge reads from row0 % tile.
    tile = _ROW_TILE
    padded = jnp.pad(word_bits, ((page_size, page_size + tile), (0, 0)))
    data = [padded.reshape(-1, 128)]  # row r's 128-lane blocks consecutive

    def block(lanes, folds=1):
        return pl.BlockSpec(
            (pl.Element((page_size + tile) * folds), pl.Element(lanes)),
            lambda r, n, run_page, lo, hi, base: (
                pl.multiple_of(
                    (_at(base, r) + page_size) // tile * tile * folds, tile
                ),
                0,
            ),
        )

    specs = [block(128, width // 128)]
    if folded:  # each token's scale bits on every lane, read as the rows
        # The merge reads a lane tile of rows from within the page's block.
        assert page_size >= _SCALE_BLOCK_LANES, page_size
        data.append(
            jnp.broadcast_to(
                jnp.pad(scale_bits, (page_size, page_size + tile))[:, None],
                (num_tokens + 2 * page_size + tile, _SCALE_BLOCK_LANES),
            )
        )
        specs.append(block(_SCALE_BLOCK_LANES))
    return data, specs, functools.partial(_rows_by_page, bits=bits, pad=page_size)


def _slot_scales_operands(w: _Write, by_token: bool):
    """Data inputs, their specs and the merge of a scale-page write. Small
    batches go token by token: XLA's scatter cost 0.1 ms per layer for a c48
    decode batch (j-fd793d3b); larger ones page by page, each run reading
    its page's worth of value rows straight from the batch: XLA's scatter
    there passed over the whole scale cache, 0.11-0.18 ms per layer at 8K
    tokens (c48 / c64 twins, j-b0a2e4d7, j-f1b20471)."""
    num_tokens, num_values = w.data.shape
    _, rows, lanes = w.cache.shape
    values = w.data.astype(jnp.float32)
    if by_token:
        padded = jnp.pad(values, ((0, 0), (0, rows - num_values)))
        broadcast = jnp.broadcast_to(padded[:, :, None], (num_tokens, rows, lanes))
        spec = pl.BlockSpec((num_tokens, rows, lanes), lambda r, *_: (0, 0, 0))
        merge = functools.partial(_slot_scales_by_token, num_values=num_values)
        return [broadcast], [spec], merge
    tile = _ROW_TILE
    padded = jnp.pad(
        values, ((lanes, lanes + tile), (0, _SCALE_BLOCK_LANES - num_values))
    )
    spec = pl.BlockSpec(
        (pl.Element(lanes + tile), pl.Element(_SCALE_BLOCK_LANES)),
        lambda r, n, run_page, lo, hi, base: (
            pl.multiple_of((_at(base, r) + lanes) // tile * tile, tile),
            0,
        ),
    )
    merge = functools.partial(_slot_scales_by_page, num_values=num_values, pad=lanes)
    return [padded], [spec], merge


def _page_block(cache: jax.Array) -> pl.BlockSpec:
    trailing = (0,) * (cache.ndim - 1)
    return pl.BlockSpec(
        (1, *cache.shape[1:]),
        lambda r, n, run_page, *_: (_at(run_page, r), *trailing),
    )


def _grid(num_runs: jax.Array, max_runs: int, traced_grid: bool) -> tuple:
    """One grid step per run (`traced_grid`), else `max_runs` steps of which
    those past `num_runs` write nothing. A traced grid keeps one step when
    there is no run, which writes the pages back unchanged (`_kernel`'s
    ``run == 0`` branch)."""
    return (jnp.maximum(num_runs, 1),) if traced_grid else (max_runs,)


def _at(ref, r):
    """Entry `r` of a run array, clamped to its last (padding) entry: the
    pipeline evaluates the index maps of the step after the last one."""
    return ref[jnp.minimum(r, ref.shape[0] - 1)]


def _write(
    writes: list[_Write],
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    traced_grid: bool,
    interpret: bool,
    name: str,
) -> list[jax.Array]:
    """Every write of `writes` at ``(pages[t], slots[t])`` in one kernel."""
    num_pages = writes[0].cache.shape[0]
    num_runs, run_page, run_lo, run_hi, run_tok0 = token_runs(
        pages, slots, num_pages, max_runs
    )
    by_token = pages.shape[0] <= _BY_TOKEN_MAX_TOKENS
    run_x = run_tok0 if by_token else run_tok0 - run_lo
    inputs, in_specs, merges, aliases = [], [], [], {}
    for k, w in enumerate(writes):
        if w.cache.shape[0] != num_pages:
            raise ValueError(f"caches of {num_pages} and {w.cache.shape[0]} pages")
        operands = _slot_scales_operands if w.bits is None else _rows_operands
        data, specs, merge = operands(w, by_token)
        inputs += data
        in_specs += specs
        aliases[5 + len(inputs)] = k  # after the 5 scalar-prefetch operands
        inputs.append(w.cache)
        in_specs.append(_page_block(w.cache))
        merges.append((len(data), merge))
    return pl.pallas_call(
        functools.partial(_kernel, merges=tuple(merges)),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=_grid(num_runs, max_runs, traced_grid),
            in_specs=in_specs,
            out_specs=[_page_block(w.cache) for w in writes],
        ),
        out_shape=[jax.ShapeDtypeStruct(w.cache.shape, w.cache.dtype) for w in writes],
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name=name,
    )(num_runs[None], run_page, run_lo, run_hi, run_x, *inputs)


@functools.partial(
    jax.jit, static_argnames=("max_runs", "traced_grid", "interpret")
)
def write_slot_scales(
    scale_cache: jax.Array,
    values: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    traced_grid: bool = True,
    interpret: bool = False,
) -> jax.Array:
    """Write f32 ``values [T, n]`` into rows ``[0, n)`` of the ``[pages, rows,
    S]`` scale cache at ``(pages[t], lane slots[t])``; runs as `write_rows`."""
    (out,) = _write(
        [_Write(scale_cache, values)],
        pages,
        slots,
        max_runs=max_runs,
        traced_grid=traced_grid,
        interpret=interpret,
        name="minimax_m3_msa_write_slot_scales",
    )
    return out


def token_runs(pages: jax.Array, slots: jax.Array, num_pages: int, max_runs: int):
    """Token runs: maximal spans of consecutive tokens writing one page.

    Returns ``num_runs`` and ``[max_runs + 1]`` arrays of each run's page,
    first slot, end slot and first token. Entries from ``num_runs`` on
    repeat the last run, so that the pipeline's look-ahead past the last grid
    step (entry ``num_runs``, at most ``max_runs``) neither reads out of
    bounds nor fetches a new block."""
    num_tokens = pages.shape[0]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = (pages >= 0) & (pages < num_pages)
    prev_page = jnp.concatenate([jnp.full((1,), -1, pages.dtype), pages[:-1]])
    prev_slot = jnp.concatenate([jnp.full((1,), -2, slots.dtype), slots[:-1]])
    start = valid & ((pages != prev_page) | (slots != prev_slot + 1))
    run_id = jnp.cumsum(start.astype(jnp.int32)) - 1
    num_runs = jnp.minimum(jnp.sum(start.astype(jnp.int32)), max_runs)
    size = max_runs + 1
    at_start = jnp.where(start & (run_id < max_runs), run_id, size)
    run_page = jnp.zeros(size, jnp.int32).at[at_start].set(pages, mode="drop")
    run_lo = jnp.zeros(size, jnp.int32).at[at_start].set(slots, mode="drop")
    run_tok0 = jnp.zeros(size, jnp.int32).at[at_start].set(tok, mode="drop")
    count = (
        jnp.zeros(size, jnp.int32)
        .at[jnp.where(valid & (run_id < max_runs), run_id, size)]
        .add(1, mode="drop")
    )
    run_hi = run_lo + count
    last = jnp.maximum(num_runs - 1, 0)
    padding = jnp.arange(size) >= num_runs
    return (
        num_runs,
        *(jnp.where(padding, a[last], a) for a in (run_page, run_lo, run_hi, run_tok0)),
    )


def _default_bits(cache: jax.Array, bits: int | None) -> int:
    """`bits` per slot, by default the cache dtype's width."""
    return 32 // cache.shape[2] if bits is None else bits


@functools.partial(
    jax.jit, static_argnames=("max_runs", "bits", "traced_grid", "interpret")
)
def write_rows(
    cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    bits: int | None = None,
    scales: jax.Array | None = None,
    traced_grid: bool = True,
    interpret: bool = False,
) -> jax.Array:
    """Write ``rows [T, W]`` (cache dtype) at ``(pages[t], slots[t])``.

    Tokens with a page outside the cache are dropped. Consecutive tokens of a
    run must take consecutive slots of one page, as a request's new tokens do;
    a run boundary is wherever that breaks. `max_runs` bounds the number of
    runs (see `max_write_runs`). The grid has one step per run
    (`traced_grid`); ``traced_grid=False`` runs all `max_runs` steps, the
    reference the traced grid is checked against.

    `bits` per slot defaults to the cache dtype's width. With ``bits=4`` (the
    uint8 4-bit caches of `msa_fp4`) `rows` are uint8 values below 16, and
    each 32-bit word holds 8 slots. `scales` (f32 ``[T]``) go to the last word
    row of a page with folded scales (`msa_fp4.scales_in_page`), lane = slot.
    """
    (out,) = _write(
        [_Write(cache, rows, _default_bits(cache, bits), scales)],
        pages,
        slots,
        max_runs=max_runs,
        traced_grid=traced_grid,
        interpret=interpret,
        name="minimax_m3_msa_write_rows",
    )
    return out


@functools.partial(
    jax.jit, static_argnames=("max_runs", "bits", "traced_grid", "interpret")
)
def write_rows_many(
    caches: tuple[jax.Array, ...],
    rows: tuple[jax.Array, ...],
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    bits: int | None = None,
    traced_grid: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """`write_rows` of ``rows[i]`` into ``caches[i]`` at the same pages and
    slots (the fp8 K/V and index caches), in one kernel."""
    return tuple(
        _write(
            [_Write(c, r, _default_bits(c, bits)) for c, r in zip(caches, rows)],
            pages,
            slots,
            max_runs=max_runs,
            traced_grid=traced_grid,
            interpret=interpret,
            name="minimax_m3_msa_write_caches",
        )
    )


@functools.partial(jax.jit, static_argnames=("max_runs", "traced_grid", "interpret"))
def write_fp4_caches(
    kv_cache: jax.Array,
    index_cache: jax.Array,
    scale_cache: jax.Array,
    kv_rows: jax.Array,
    index_rows: jax.Array,
    index_scales: jax.Array,
    kv_scales: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    traced_grid: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A step's 4-bit writes at ``(pages[t], slots[t])`` in one kernel: K/V
    nibble rows ``[T, W]``, index nibble rows ``[T, D]`` with their folded f32
    scales ``[T]``, and f32 K/V scales ``[T, n]`` into rows ``[0, n)`` of the
    scale page; as `write_rows` twice and `write_slot_scales`."""
    kv_cache, index_cache, scale_cache = _write(
        [
            _Write(kv_cache, kv_rows, 4),
            _Write(index_cache, index_rows, 4, index_scales),
            _Write(scale_cache, kv_scales),
        ],
        pages,
        slots,
        max_runs=max_runs,
        traced_grid=traced_grid,
        interpret=interpret,
        name="minimax_m3_msa_write_caches",
    )
    return kv_cache, index_cache, scale_cache


def max_write_runs(num_tokens: int, max_num_seqs: int, page_size: int) -> int:
    """Runs of a batch: a request's q tokens span at most
    ``(q - 1) // page_size + 2`` pages."""
    return max(1, min(num_tokens, num_tokens // page_size + 2 * max_num_seqs))
