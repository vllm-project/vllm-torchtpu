# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Kimi-K3.1's sparse MLA prefill attention in latent form (absorbed MQA):
each query row attends exactly its selected keys, gathered token by token.

A row's queries at ``H`` heads become ``[q_nope W_UK^T | q_rope | 0]`` (640
lanes, pre-scaled by ``SCALE_LOG2``; :func:`absorb`). Its selection words
(bit ``c % 32`` of word ``c // 32`` over the ``[arena | window]`` columns,
``mla_flash_prefill_packed``'s ``select``) compact into its selected
columns (:func:`selected_columns`, or :func:`sparse_core_table_rows` on the
SparseCore), which name rows of one gather table (:func:`gather_table`):
the gathered FP8 prefix pages' tokens (an arena column finds its page by the
arena order's runs, :func:`token_rows`), the window's latents rounded to
FP8 as the pool stores them, then zero rows that every padded slot reads.
The SparseCore gathers each row's tokens, and :func:`mla_sparse_rows`
attends each row's heads over its keys in one softmax pass; ``W_UV`` then
maps the latent outputs to the heads' values (:func:`attend_rows`). Real
selections are scattered token by token (a row's 2048 keys touch about 860
of the 32-token chunks at 138K context), so the gather is per token.

Consecutive rows share many of their keys (8 rows select about 6.2K distinct
tokens at 114K context, against 16K row-keys), so :func:`attend_blocks`
gathers each block's union once: the gather table's spare lanes carry each
row's membership (:func:`member_lanes`), and a row's absorbed query holds
:data:`BIAS` on its own lane, so the scores of keys the row did not select
fall below ``BIAS / 2`` and drop out while its own keys' scores are exact
(:func:`mla_sparse_blocks`).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

CHUNK = 32  # key columns per selection word
LANES = 640  # latent 512 | rope 64 | zero 64
LATENT = 512
USED = 576
PAGE_CHUNKS = 16  # chunks per 512-token FP8 page


def valid_words(segments, length, *, count: int, capacity: int, tokens: int, rows=None):
    """int32 ``[R, (capacity + tokens) / 32]``: every candidate of each row
    (``dsa_prefill.candidate_valid``'s rule, packed): a row of sequence ``s``
    below ``length`` keeps arena columns ``[base_s, base_s + cached_s)`` and
    window columns ``capacity + [offset_s, row]``. ``rows``: the window rows
    (``[R]``, default all). For windows whose rows all keep every candidate
    (the selection is skipped there)."""
    rows = (jnp.arange(tokens) if rows is None else rows).astype(jnp.int32)[:, None]
    first = jnp.full(rows.shape, capacity + tokens, jnp.int32)
    arena_lo = arena_hi = window_lo = window_hi = first
    for s in range(count):
        offset, end = segments[s], segments[s] + segments[count + s]
        base, cached = segments[3 * count + s], segments[4 * count + s]
        inside = (rows >= offset) & (rows < end) & (rows < length[0])
        arena_lo = jnp.where(inside, base, arena_lo)
        arena_hi = jnp.where(inside, base + cached, arena_hi)
        window_lo = jnp.where(inside, capacity + offset, window_lo)
        window_hi = jnp.where(inside, capacity + rows + 1, window_hi)
    start = jnp.arange((capacity + tokens) // CHUNK, dtype=jnp.int32)[None] * CHUNK

    def span(lo, hi):  # bits of [lo, hi) within each word
        a = jnp.clip(lo - start, 0, CHUNK)
        b = jnp.clip(hi - start, 0, CHUNK)

        def ones(n):
            return jnp.where(n >= CHUNK, -1, (jnp.int32(1) << jnp.minimum(n, CHUNK - 1)) - 1)

        return jnp.where(b > a, ones(b) & ~ones(a), 0)

    return span(arena_lo, arena_hi) | span(window_lo, window_hi)


def selection_words(select, dense, desc, length, *, count: int, capacity: int, tokens: int, transposed: bool,
                    rows=None):
    """The words the sparse attention reads: the selection's (``[R, W]``, or
    ``[W, R]`` with ``transposed``), or in a dense window (``dense[0]``
    nonzero: its rows all keep every candidate, and the selection was
    skipped) every candidate's (:func:`valid_words`) of the rows ``rows``
    (default all ``tokens``), padded to the selection's width."""

    def every(_):
        valid = jax.lax.bitcast_convert_type(
            valid_words(desc, length, count=count, capacity=capacity, tokens=tokens, rows=rows), select.dtype)
        width = select.shape[0] if transposed else select.shape[1]
        valid = jnp.pad(valid, ((0, 0), (0, width - valid.shape[1])))
        return valid.T if transposed else valid

    return jax.lax.cond(dense[0] > 0, every, lambda _: select, None)


def absorb(query, key_value_up):
    """``[R, H, 640]`` BF16 absorbed queries ``[q_nope W_UK^T | q_rope | 0]``
    from the heads' queries ``[R, H, 256]`` (``mla_heads``' layout: nope 128,
    rope 64, zero 64; already scaled) and their up-projection ``[512, H *
    256]`` (per head: W_UK 128 columns, then W_UV 128)."""
    nope, rope = absorb_parts(query, key_value_up)
    return jnp.concatenate((nope.astype(jnp.bfloat16), rope), axis=2)


def absorb_parts(query, key_value_up):
    """:func:`absorb`'s two parts: ``q_nope W_UK^T`` in FP32 ``[R, H, 512]``
    (unrounded, for a two-part BF16 split in the kernel) and ``[q_rope | 0]``
    BF16 ``[R, H, 128]``."""
    rows, heads, _ = query.shape
    w_uk = key_value_up.reshape(LATENT, heads, 256)[:, :, :128]
    nope = jnp.einsum("rhd,khd->rhk", query[:, :, :128], w_uk, preferred_element_type=jnp.float32)
    return nope, jnp.concatenate((query[:, :, 128:192], jnp.zeros((rows, heads, LANES - USED), jnp.bfloat16)), axis=2)


def selected_columns(select, *, topk: int = 2048, transposed: bool = False, capacity: int = 0, cached=None):
    """``(columns, counts)``: each row's selected columns in ascending order,
    int32 ``[R, topk]`` (zero past its count), and its count ``[R]``, from its
    selection words (``[R, W]``, or ``[W, R]`` with ``transposed``; at most
    ``topk`` bits per row). ``cached`` (with ``capacity``, the arena's
    columns): arena words at and above ``cached / 32`` are ignored (a
    selection kernel may leave them unwritten). Slot ``s`` of a row finds its
    word by a binary search over the words' running bit counts, then its bit
    by a binary search over the word's low-bit counts."""
    words = jax.lax.bitcast_convert_type(select, jnp.uint32)
    if transposed:
        words = words.T
    rows, width = words.shape
    if cached is not None:
        column = jnp.arange(width, dtype=jnp.int32)[None]
        live = (column >= capacity // CHUNK) | (column < (cached + CHUNK - 1) // CHUNK)
        words = jnp.where(live, words, jnp.uint32(0))
    counts = jax.lax.population_count(words).astype(jnp.int32)
    ends = jnp.cumsum(counts, axis=1)  # [R, W]: bits in words 0..w
    total = ends[:, -1]
    slot = jnp.arange(topk, dtype=jnp.int32)[None]
    lo = jnp.zeros((rows, topk), jnp.int32)
    hi = jnp.full((rows, topk), width - 1, jnp.int32)
    for _ in range(max(1, (width - 1).bit_length())):  # first word whose end exceeds the slot
        mid = (lo + hi) // 2
        below = jnp.take_along_axis(ends, mid, axis=1) <= slot
        lo, hi = jnp.where(below, mid + 1, lo), jnp.where(below, hi, mid)
    word = jnp.take_along_axis(words, lo, axis=1)
    rank = slot - (jnp.take_along_axis(ends, lo, axis=1) - jnp.take_along_axis(counts, lo, axis=1))
    bit = jnp.zeros((rows, topk), jnp.int32)
    for width_bits in (16, 8, 4, 2, 1):  # the rank-th set bit: narrow from the low end
        low = jax.lax.population_count(
            (word >> bit.astype(jnp.uint32)) & jnp.uint32((1 << width_bits) - 1)).astype(jnp.int32)
        step = low <= rank
        rank = jnp.where(step, rank - low, rank)
        bit = jnp.where(step, bit + width_bits, bit)
    live = slot < total[:, None]
    return jnp.where(live, lo * CHUNK + bit, 0), total


SC_LANES = 16  # a SparseCore vector


def token_rows(columns, order, *, capacity: int, runs: int):
    """Rows of :func:`gather_table`'s arena part for arena ``columns`` (below
    ``capacity``): column ``c`` is token ``c % 512`` of gathered page
    ``order[1 + c // 512]``. The arena order is a few runs of consecutive
    gathered pages (a sequence's pages on one rank), so each column finds
    its run by comparisons rather than a gather; an order of more than
    ``runs`` runs over its used pages takes the gather."""
    page = PAGE_CHUNKS * CHUNK
    pages = order.shape[0] - 1
    slots = order[1:]
    arena_page = jnp.minimum(columns, capacity - 1) // page
    first = jnp.concatenate((jnp.ones((1,), bool), slots[1:] != slots[:-1] + 1))
    used = jnp.sum(first & (jnp.arange(pages) < order[0]))

    def by_runs(arena_page):
        lo = jnp.nonzero(first, size=runs, fill_value=pages)[0].astype(jnp.int32)
        hi = jnp.concatenate((lo[1:], jnp.full((1,), pages, jnp.int32)))
        base = slots[jnp.minimum(lo, pages - 1)] - lo
        slot = jnp.zeros_like(arena_page)
        for k in range(runs):
            slot = jnp.where((arena_page >= lo[k]) & (arena_page < hi[k]), base[k] + arena_page, slot)
        return slot

    def by_gather(arena_page):
        return slots[arena_page]

    slot = jax.lax.cond(used <= runs, by_runs, by_gather, arena_page)
    return slot * page + columns % page


ZERO_ROWS = 32  # zero token rows after the window in the gather table: every padded slot reads them


def table_extra(tokens: int) -> int:
    """The spare rows a flat prefix arena needs after its pages
    (``gathered_prefix_pages(flat_extra=...)``): the window's ``tokens``,
    then :data:`ZERO_ROWS`."""
    return tokens + ZERO_ROWS


def gather_table(pages, window_latent, *, fp8: bool = False):
    """``(table, window_base)``: int8 ``[S * 512 + T + ZERO_ROWS, 640]`` token
    rows, the gathered FP8 pages' tokens (row ``s * 512 + t`` is token ``t``
    of page ``s``, the pages' own packing), the window's latents as FP8 (the
    pool writer's rounding), then :data:`ZERO_ROWS` zero rows (from row
    ``window_base + T``), and the window's first row. ``pages`` is ``[S,
    128, 4, 640]``, or already token rows ``[S * 512 + table_extra(T), 640]``
    (``gathered_prefix_pages(flat_extra=table_extra(T))``) whose last rows
    take the window and the zeros in place."""
    window = jnp.pad(window_latent, ((0, ZERO_ROWS), (0, LANES - window_latent.shape[1])))
    window = window.astype(jnp.float8_e4m3fn)
    view = (lambda x: x) if fp8 else (lambda x: jax.lax.bitcast_convert_type(x, jnp.int8))
    if pages.ndim == 2:
        base = pages.shape[0] - window.shape[0]
        return view(jax.lax.dynamic_update_slice(pages, window, (base, 0))), base
    slots = pages.shape[0]
    arena = pages.reshape(slots * 512, LANES)
    return view(jnp.concatenate((arena, window), axis=0)), slots * 512


def _row_kernel(count_ref, *refs):
    """``rows`` rows per step: each row's queries over its gathered keys
    ``[topk, 640]`` (FP8), its first ``count`` valid, in one softmax pass
    (``S = Q K^T``; the denominators from ones in lanes 576:). Queries
    ``[H, 640]`` BF16, or (split) ``q_nope W_UK^T`` FP32 ``[H, 512]`` and
    ``[q_rope | 0]`` ``[H, 128]``: the nope part in two BF16 parts (its
    rounding, then the FP32 remainder), so the scores lose no BF16 rounding
    of it. A row of no keys (``count`` zero: past the window's tokens) gives
    zeros without reading its keys, which the gather may leave unwritten."""
    split = len(refs) == 4
    keys_ref, out_ref = refs[-2:]
    step = pl.program_id(0)
    rows = out_ref.shape[0]
    topk = keys_ref.shape[0] // rows
    for r in range(rows):
        count = count_ref[step * rows + r]

        @pl.when(count == 0)
        def _(r=r):
            out_ref[r] = jnp.zeros(out_ref.shape[1:], out_ref.dtype)

        @pl.when(count > 0)
        def _(r=r, count=count):
            _attend_row(refs, r, count, topk, split)


def _attend_row(refs, r, count, topk, split):
    """:func:`_row_kernel`'s row ``r`` of its step: its first ``count`` keys."""
    keys_ref, out_ref = refs[-2:]
    contract = (((1,), (1,)), ((), ()))
    keys = keys_ref[pl.ds(r * topk, topk)].astype(jnp.bfloat16)
    lane = jax.lax.broadcasted_iota(jnp.int32, keys.shape, 1)
    keys = jnp.where(lane < USED, keys, jnp.ones((), keys.dtype))
    if split:
        nope = refs[0][r]
        high = nope.astype(jnp.bfloat16)
        low = (nope - high.astype(jnp.float32)).astype(jnp.bfloat16)
        latent = keys[:, :LATENT]
        scores = (jax.lax.dot_general(high, latent, contract, preferred_element_type=jnp.float32)
                  + jax.lax.dot_general(low, latent, contract, preferred_element_type=jnp.float32)
                  + jax.lax.dot_general(refs[1][r], keys[:, LATENT:], contract,
                                        preferred_element_type=jnp.float32))
    else:
        scores = jax.lax.dot_general(refs[0][r], keys, contract, preferred_element_type=jnp.float32)
    valid = jax.lax.broadcasted_iota(jnp.int32, scores.shape, 1) < count
    scores = jnp.where(valid, scores, -jnp.inf)
    top = jnp.max(scores, axis=1, keepdims=True)
    top = jnp.where(top == -jnp.inf, 0.0, top)
    probabilities = jnp.exp2(scores - top).astype(jnp.bfloat16)
    weighted = jax.lax.dot_general(probabilities, keys, (((1,), (0,)), ((), ())),
                                   preferred_element_type=jnp.float32)  # [H, 640]
    denominator = weighted[:, USED:USED + 1]
    out_ref[r] = weighted[:, :LATENT] / jnp.where(denominator > 0, denominator, 1.0)


def mla_sparse_rows(query, keys, counts, *, rows: int = 2):
    """``[R, H, 512]`` FP32: each row's queries over its gathered keys (``[R
    topk, 640]`` FP8, the gather's rows as they come: row ``r``'s from ``r
    topk``, its first ``counts[r]`` valid). ``query``: ``[R, H, 640]`` BF16
    absorbed and scaled (:func:`absorb`), or the pair of :func:`absorb_parts`
    (scored in two BF16 parts)."""
    parts = query if isinstance(query, tuple) else (query,)
    total, heads, _ = parts[0].shape
    topk = keys.shape[0] // total
    if total % rows:
        raise ValueError("rows must divide the query rows")
    specs = [pl.BlockSpec((rows, heads, part.shape[-1]), lambda i, *_: (i, 0, 0)) for part in parts]
    return pl.pallas_call(
        _row_kernel,
        out_shape=jax.ShapeDtypeStruct((total, heads, LATENT), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(total // rows,),
            in_specs=[*specs, pl.BlockSpec((rows * topk, LANES), lambda i, *_: (i, 0))],
            out_specs=pl.BlockSpec((rows, heads, LATENT), lambda i, *_: (i, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",),
                                             vmem_limit_bytes=64 * 1024**2),
        name="kimi_mla_sparse_rows",
    )(counts.astype(jnp.int32), *parts, keys)


def attend_rows(query, window_latent, key_value_up, pages, order, select, *, capacity: int, gather,
                runs: int = 64, transposed: bool = False, topk: int = 2048, rows: int = 2, groups: int = 1,
                compact: str = "xla", fp8_gather: bool = False, split: bool = False):
    """:func:`attend` by per-row token gathers: ``[R, H, 128]`` BF16.
    ``gather(table, indices, end)`` gathers the int8 table rows of
    ``indices[:end]`` (``fp8_gather``: FP8; the SparseCore's
    ``ragged_gather_v2``; the rows past ``end`` may stay unwritten): a group
    gathers up to its last row with keys, so the rows past the window's
    tokens (no keys) read nothing. Window keys are their FP8 rounding (as the pool
    stores them). ``runs``: the most runs of the arena order (sequences
    times ranks). ``groups``: row groups gathered and attended in turn, so
    the gather of one group can overlap the attention of the previous.
    ``compact``: the column compaction, ``xla`` (:func:`selected_columns`,
    :func:`token_rows`) or ``sparse_core`` (:func:`sparse_core_table_rows`:
    no XLA gathers, which the SparseCore serves slowly when indices repeat).
    ``split``: the absorbed queries scored in two BF16 parts
    (:func:`absorb_parts`)."""
    count, heads, _ = query.shape
    if count % (groups * rows):
        raise ValueError("groups of whole kernel steps must divide the rows")
    q = absorb_parts(query, key_value_up) if split else absorb(query, key_value_up)
    cached = order[0] * PAGE_CHUNKS * CHUNK
    table, base = gather_table(pages, window_latent, fp8=fp8_gather)
    zero = base + window_latent.shape[0]
    if compact == "sparse_core":
        words = jax.lax.bitcast_convert_type(select, jnp.uint32)
        words = words.T if transposed else words
        column = jnp.arange(words.shape[1], dtype=jnp.int32)[None]
        live = (column >= capacity // CHUNK) | (column < (cached + CHUNK - 1) // CHUNK)
        words = jnp.where(live, words, jnp.uint32(0))
        counts = jnp.minimum(jnp.sum(jax.lax.population_count(words), axis=1).astype(jnp.int32), topk)
        index = sparse_core_table_rows(words, order, capacity=capacity, base=base, zero=zero, tile=topk,
                                       whole=True)[0].reshape(count, topk)
    else:
        columns, counts = selected_columns(select, topk=topk, transposed=transposed, capacity=capacity,
                                           cached=cached)
        index = jnp.where(columns < capacity, token_rows(columns, order, capacity=capacity, runs=runs),
                          base + columns - capacity)
        # A row's slots past its count read zero rows: their probabilities are zero, and the arena row a padded
        # column would name may be unwritten (stale bytes, NaN codes) where 0 x NaN is NaN in P V. They take the
        # zero rows in turn: the SparseCore serializes repeated reads of one row (about 37 ns each).
        slot = jnp.arange(topk, dtype=jnp.int32)[None]
        index = jnp.where(slot < counts[:, None], index, zero + slot % ZERO_ROWS)
    size = count // groups
    outs = []
    for g in range(groups):
        part = slice(g * size, (g + 1) * size)
        live = jnp.max(jnp.where(counts[part] > 0, jnp.arange(1, size + 1, dtype=jnp.int32), 0))
        keys = gather(table, index[part].reshape(-1), live * topk)
        if not fp8_gather:
            keys = jax.lax.bitcast_convert_type(keys, jnp.float8_e4m3fn)
        outs.append(mla_sparse_rows(tuple(x[part] for x in q) if split else q[part], keys, counts[part], rows=rows))
    out = outs[0] if groups == 1 else jnp.concatenate(outs, axis=0)
    w_uv = key_value_up.reshape(LATENT, heads, 256)[:, :, 128:].astype(jnp.float32)
    attended = jnp.einsum("rhd,dhv->rhv", out, w_uv, precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32)
    return attended.astype(jnp.bfloat16)


MEMBERS = LANES - USED  # rows whose membership a gather table's spare lanes carry at once
OUTSIDE = 0x38  # FP8 E4M3 1.0: a membership lane whose row did not select the table row's token
BIAS = -32768.0  # a row's absorbed query on its own membership lane


def union_columns(words, *, block: int, limit: int):
    """``(columns, counts)``: the selected columns of each ``block``
    consecutive rows together (the OR of their words ``[R, W]``), ascending,
    ``[R / block, limit]`` and ``[R / block]``."""
    rows, width = words.shape
    union = jax.lax.bitcast_convert_type(words, jnp.uint32).reshape(rows // block, block, width)
    union = jax.lax.reduce(union, jnp.uint32(0), jax.lax.bitwise_or, (1,))
    return selected_columns(union, topk=limit)


def member_lanes(words, order, *, capacity: int, slots: int, tokens: int):
    """int8 ``[slots * 512 + tokens + ZERO_ROWS, 64]``, a gather table's
    (:func:`gather_table`) spare lanes: lane ``j`` of the row of column ``c``
    is FP8 zero if row ``j`` of ``words`` (``[<= 64, W]``) selected ``c``,
    else FP8 one (:data:`OUTSIDE`; zero rows and pages outside the arena
    order: every lane). Each slot takes its arena page's 16 words."""
    rows = words.shape[0]
    words = jax.lax.bitcast_convert_type(words, jnp.uint32)
    pages = capacity // (PAGE_CHUNKS * CHUNK)
    arena = words[:, :pages * PAGE_CHUNKS].reshape(rows, pages, PAGE_CHUNKS)
    # Each slot's arena position by comparisons (no scatter or repeated gather index: XLA serves both on the
    # SparseCore, which serializes repeated rows); slots outside the arena order take the zero page.
    match = ((order[1:1 + pages][None, :] == jnp.arange(slots, dtype=order.dtype)[:, None])
             & (jnp.arange(pages) < order[0])[None, :])
    found = jnp.any(match, axis=1)
    inverse = jnp.where(found, jnp.argmax(match, axis=1), jnp.arange(slots) % pages)
    picked = jnp.where(found[None, :, None], arena[:, inverse], jnp.uint32(0))
    table = jnp.concatenate((picked.reshape(rows, slots * PAGE_CHUNKS),
                             words[:, pages * PAGE_CHUNKS:(capacity + tokens) // CHUNK],
                             jnp.zeros((rows, ZERO_ROWS // CHUNK), jnp.uint32)), axis=1)
    bits = (table.T[:, None, :] >> jnp.arange(CHUNK, dtype=jnp.uint32)[None, :, None]) & 1
    codes = jnp.where(bits == 0, jnp.int8(OUTSIDE), jnp.int8(0)).reshape(-1, rows)
    return jnp.pad(codes, ((0, 0), (0, MEMBERS - rows)), constant_values=OUTSIDE)


def block_rows(index, counts, *, tile: int, zero):
    """The blocks' gather-table rows (``index`` ``[B, limit]``, the first
    ``counts[b]`` of block ``b``) laid end to end, each block padded to whole
    ``tile``-row tiles with the :data:`ZERO_ROWS` zero rows from ``zero`` in
    turn: ``(rows [B * ceil(limit / tile) * tile], offsets [B], tiles [B],
    total)``."""
    blocks, limit = index.shape
    tiles = (counts + tile - 1) // tile
    ends = jnp.cumsum(tiles) * tile
    offsets = ends - tiles * tile
    flat = jnp.arange(blocks * (-(-limit // tile)) * tile, dtype=jnp.int32)
    owner = jnp.sum(flat[:, None] >= ends[None, :], axis=1)
    mine = jnp.minimum(owner, blocks - 1)
    slot = flat - offsets[mine]
    live = (owner < blocks) & (slot < counts[mine])
    rows = jnp.where(live, index[mine, jnp.minimum(slot, limit - 1)], zero + flat % ZERO_ROWS)
    return rows, offsets, tiles, ends[-1]


def _table_rows_kernel(words_hbm, offsets_hbm, slots_hbm, out_hbm, words_vmem, slots_vmem, out_vmem, scalar_vmem, *,
                       blocks: int, width: int, tile: int, whole: bool, capacity: int, base: int, zero: int,
                       workers: int, subcores: int):
    """:func:`sparse_core_table_rows` on one vector subcore: whole blocks
    (or rows); a block's words into VMEM, every set bit's gather-table row
    appended (lowest set bit first, so a wave of 16 words takes as many
    steps as its fullest word has bits; an arena column's page slot from the
    arena order in VMEM), padded with the zero rows in turn to whole tiles
    (``whole``: to one tile) and written from the block's offset."""
    worker = jax.lax.axis_index("core") * subcores + jax.lax.axis_index("subcore")
    lane = jax.lax.iota(jnp.int32, SC_LANES)
    page = PAGE_CHUNKS * CHUNK
    pltpu.sync_copy(slots_hbm, slots_vmem)

    @pl.loop(0, (blocks + workers - 1) // workers)
    def wave(w):
        b = w * workers + worker

        @pl.when(b < blocks)
        def one():
            pltpu.sync_copy(offsets_hbm.at[pl.ds(b * SC_LANES, SC_LANES)], scalar_vmem)
            offset = scalar_vmem[...][0]
            pltpu.sync_copy(words_hbm.at[pl.ds(b * width, width)], words_vmem)

            def chunk(i, count):
                first = (i * SC_LANES + lane) * CHUNK

                def more(state):
                    return plsc.all_reduce_population_count(state[0] != 0)[0] > 0

                def take(state):
                    word, count = state
                    live = word != 0
                    low = word & -word
                    bit = jnp.zeros_like(word)
                    for shift, mask in ((1, -0x55555556), (2, -0x33333334), (4, -0x0F0F0F10), (8, -0x00FF0100),
                                        (16, -0x00010000)):  # 0xAAAAAAAA, 0xCCCCCCCC, 0xF0F0F0F0, 0xFF00FF00, 0xFFFF0000
                        bit = bit | jnp.where((low & mask) != 0, shift, 0)
                    column = first + bit
                    arena = column < capacity
                    slot = plsc.load_gather(slots_vmem, [jnp.where(arena, column // page, 0)], mask=live & arena)
                    row = jnp.where(arena, slot * page + column % page, base + column - capacity)
                    plsc.store_compressed(out_vmem.at[pl.ds(count, SC_LANES)], row, mask=live)
                    return word & (word - 1), count + plsc.all_reduce_population_count(live)[0]

                return jax.lax.while_loop(more, take, (words_vmem[pl.ds(i * SC_LANES, SC_LANES)], count))[1]

            count = jax.lax.fori_loop(0, width // SC_LANES, chunk, jnp.int32(0))
            padded = jnp.int32(tile) if whole else (count + tile - 1) // tile * tile

            @pl.loop(count // SC_LANES, padded // SC_LANES)
            def fill(k):
                at = pl.ds(k * SC_LANES, SC_LANES)
                position = k * SC_LANES + lane
                out_vmem[at] = jnp.where(position >= count, zero + position % ZERO_ROWS, out_vmem[at])

            @pl.loop(0, padded // tile)
            def flush(t):
                pltpu.sync_copy(out_vmem.at[pl.ds(t * tile, tile)], out_hbm.at[pl.ds(offset + t * tile, tile)])


def sparse_core_table_rows(words, order, *, capacity: int, base: int, zero: int, tile: int, whole: bool = False,
                           limit: int = 0):
    """Gather-table rows (:func:`gather_table`) of each row's (or block's)
    selected columns (``words`` ``[B, W]``, ``W`` a multiple of 16, at most
    ``limit`` bits per row) on the SparseCores, without XLA gathers: laid end
    to end in whole ``tile``-row tiles (``whole``: exactly one tile per row,
    ``tile`` >= its bits), padded with the :data:`ZERO_ROWS` zero rows from
    ``zero`` in turn, an arena column ``c`` at row ``order[1 + c // 512] *
    512 + c % 512`` and a window column at ``base + c - capacity``: ``(rows
    [B * ceil(limit / tile) * tile], offsets [B], tiles [B], total)``, a
    block's rows in an unspecified order, the rows past ``total``
    unwritten."""
    words = jnp.pad(words, ((0, 0), (0, -words.shape[1] % SC_LANES)))
    blocks, width = words.shape
    if whole:
        tiles = jnp.ones((blocks,), jnp.int32)
        size = tile
    else:
        counts = jnp.sum(jax.lax.population_count(jax.lax.bitcast_convert_type(words, jnp.uint32)), axis=1)
        tiles = (counts.astype(jnp.int32) + tile - 1) // tile
        size = -(-limit // tile) * tile
    ends = jnp.cumsum(tiles) * tile
    offsets = ends - tiles * tile
    pages = capacity // (PAGE_CHUNKS * CHUNK)
    slots = jnp.pad(order[1:1 + pages].astype(jnp.int32), (0, -pages % SC_LANES))
    info = pltpu.get_tpu_info().sparse_core
    mesh = plsc.VectorSubcoreMesh(num_cores=info.num_cores, num_subcores=info.num_subcores,
                                  core_axis_name="core", subcore_axis_name="subcore")
    rows = pl.kernel(
        functools.partial(_table_rows_kernel, blocks=blocks, width=width, tile=tile, whole=whole, capacity=capacity,
                          base=base, zero=zero, workers=info.num_cores * info.num_subcores,
                          subcores=info.num_subcores),
        out_type=jax.ShapeDtypeStruct((blocks * size,), jnp.int32),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, needs_layout_passes=False),
        scratch_types=(pltpu.VMEM((width,), jnp.int32), pltpu.VMEM((slots.shape[0],), jnp.int32),
                       pltpu.VMEM((size + SC_LANES,), jnp.int32), pltpu.VMEM((SC_LANES,), jnp.int32)),
        mesh=mesh,
        name="kimi_sparse_table_rows",
    )(jax.lax.bitcast_convert_type(words, jnp.int32).reshape(-1), jnp.repeat(offsets, SC_LANES), slots)
    return rows, offsets, tiles, ends[-1]


def _union_kernel(offset_ref, tiles_ref, q_ref, keys_hbm, out_ref, buf, sems, top_ref, sum_ref, acc_ref, *,
                  tile: int):
    """One block of rows: its queries ``[Q, H, 640]`` (each row's own
    membership lane :data:`BIAS`) over its union's key tiles (FP8 rows in
    HBM from ``offset``, double-buffered), online softmax with the
    denominators on the MXU. A row's non-members score near ``BIAS``: once
    a tile holds one of its keys they underflow, and what they added before
    is rescaled to zero; a row with no key in the block (past the length)
    gives zeros."""
    b = pl.program_id(0)
    block, heads, _ = q_ref.shape
    q = q_ref[...].reshape(block * heads, LANES)
    start, count = offset_ref[b], tiles_ref[b]

    def copy(t, slot):
        return pltpu.make_async_copy(keys_hbm.at[pl.ds(pl.multiple_of(start + t * tile, tile), tile)], buf.at[slot],
                                     sems.at[slot])

    ones = jnp.ones((tile, 128), jnp.bfloat16)
    top_ref[...] = jnp.full(top_ref.shape, 2 * BIAS, jnp.float32)
    sum_ref[...] = jnp.zeros(sum_ref.shape, jnp.float32)
    acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    @pl.when(count > 0)
    def first():
        copy(0, 0).start()

    @pl.loop(0, count)
    def step(t):
        slot = t % 2

        @pl.when(t + 1 < count)
        def ahead():
            copy(t + 1, 1 - slot).start()

        copy(t, slot).wait()
        keys = buf[slot].astype(jnp.bfloat16)
        scores = jax.lax.dot_general(q, keys, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        old = top_ref[...]
        new = jnp.maximum(old, jnp.max(scores, axis=1, keepdims=True))
        scale = jnp.exp2(old - new)
        probabilities = jnp.exp2(scores - new).astype(jnp.bfloat16)
        sum_ref[...] = scale * sum_ref[...] + jax.lax.dot_general(
            probabilities, ones, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)[:, :1]
        acc_ref[...] = scale * acc_ref[...] + jax.lax.dot_general(
            probabilities, keys[:, :LATENT], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        top_ref[...] = new

    total = sum_ref[...]
    member = (top_ref[...] > BIAS / 2) & (total > 0)
    out_ref[...] = jnp.where(member, acc_ref[...] / jnp.where(member, total, 1.0), 0.0).reshape(block, heads, LATENT)


def mla_sparse_blocks(query, keys, offsets, tiles, *, block: int, tile: int = 512):
    """``[R, H, 512]`` FP32: each block of ``block`` rows' queries (``[R, H,
    640]`` BF16 absorbed, scaled, own membership lanes set) over its keys,
    ``tiles[b]`` tiles of ``tile`` FP8 rows of ``keys`` from ``offsets[b]``."""
    total, heads, _ = query.shape
    blocks = total // block
    return pl.pallas_call(
        functools.partial(_union_kernel, tile=tile),
        out_shape=jax.ShapeDtypeStruct((total, heads, LATENT), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(blocks,),
            in_specs=[pl.BlockSpec((block, heads, LANES), lambda i, *_: (i, 0, 0)),
                      pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec((block, heads, LATENT), lambda i, *_: (i, 0, 0)),
            scratch_shapes=(pltpu.VMEM((2, tile, LANES), keys.dtype), pltpu.SemaphoreType.DMA((2,)),
                            pltpu.VMEM((block * heads, 1), jnp.float32),
                            pltpu.VMEM((block * heads, 1), jnp.float32),
                            pltpu.VMEM((block * heads, LATENT), jnp.float32)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",),
                                             vmem_limit_bytes=64 * 1024**2),
        name="kimi_mla_sparse_blocks",
    )(offsets.astype(jnp.int32), tiles.astype(jnp.int32), query, keys)


def attend_blocks(query, window_latent, key_value_up, pages, order, select, *, capacity: int, gather,
                  runs: int = 64, transposed: bool = False, topk: int = 2048, block: int = 8, tile: int = 512,
                  compact: str = "xla", fp8_gather: bool = False):
    """:func:`attend_rows` by block unions: each ``block`` consecutive rows
    attend the union of their selected tokens, gathered once, each row
    masked to its own by its membership lane of the gather table (rows in
    groups of up to 64, the table's spare lanes; :func:`member_lanes`).
    ``gather(table, rows, end)`` gathers ``table[rows[:end]]`` (rows past
    ``end`` unspecified). ``compact``: the union compaction, ``xla``
    (:func:`union_columns`, :func:`block_rows`) or ``sparse_core``
    (:func:`sparse_core_table_rows`). ``[R, H, 128]`` BF16."""
    count, heads, _ = query.shape
    group = min(count, MEMBERS)
    if count % group or group % block:
        raise ValueError("blocks must tile groups of up to 64 rows")
    words = jax.lax.bitcast_convert_type(select, jnp.uint32)
    words = words.T if transposed else words
    cached = order[0] * PAGE_CHUNKS * CHUNK
    column = jnp.arange(words.shape[1], dtype=jnp.int32)[None]
    live = (column >= capacity // CHUNK) | (column < (cached + CHUNK - 1) // CHUNK)
    words = jnp.where(live, words, jnp.uint32(0))
    tokens = window_latent.shape[0]
    table, base = gather_table(pages, window_latent, fp8=fp8_gather)
    slots = base // (PAGE_CHUNKS * CHUNK)
    q = absorb(query, key_value_up)
    lane = jnp.arange(MEMBERS, dtype=jnp.int32)[None, None]
    own = (jnp.arange(count, dtype=jnp.int32) % group)[:, None, None]
    q = q.at[:, :, USED:].set(jnp.where(lane == own, BIAS, 0.0).astype(q.dtype))
    limit = min(block * topk, capacity + tokens)
    outs = []
    for g in range(count // group):
        part = slice(g * group, (g + 1) * group)
        lanes = member_lanes(words[part], order, capacity=capacity, slots=slots, tokens=tokens)
        marked = jax.lax.dynamic_update_slice(
            table, jax.lax.bitcast_convert_type(lanes, jnp.float8_e4m3fn) if fp8_gather else lanes, (0, USED))
        if compact == "sparse_core":
            union = words[part].reshape(group // block, block, -1)
            union = jax.lax.reduce(union, jnp.uint32(0), jax.lax.bitwise_or, (1,))
            rows, offsets, tiles, end = sparse_core_table_rows(union, order, capacity=capacity, base=base,
                                                               zero=base + tokens, tile=tile, limit=limit)
            # The SparseCore gather reads whole blocks of indices past ``end``: rows the kernel left unwritten
            # would name arbitrary table rows.
            position = jnp.arange(rows.shape[0], dtype=jnp.int32)
            rows = jnp.where(position < end, rows, base + tokens + position % ZERO_ROWS)
        else:
            columns, counts = union_columns(words[part], block=block, limit=limit)
            index = jnp.where(columns < capacity, token_rows(columns, order, capacity=capacity, runs=runs),
                              base + columns - capacity)
            rows, offsets, tiles, end = block_rows(index, counts, tile=tile, zero=base + tokens)
        keys = gather(marked, rows, end)
        keys = keys if fp8_gather else jax.lax.bitcast_convert_type(keys, jnp.float8_e4m3fn)
        outs.append(mla_sparse_blocks(q[part], keys, offsets, tiles, block=block, tile=tile))
    out = outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=0)
    w_uv = key_value_up.reshape(LATENT, heads, 256)[:, :, 128:].astype(jnp.float32)
    attended = jnp.einsum("rhd,dhv->rhv", out, w_uv, precision=jax.lax.Precision.HIGHEST,
                          preferred_element_type=jnp.float32)
    return attended.astype(jnp.bfloat16)
