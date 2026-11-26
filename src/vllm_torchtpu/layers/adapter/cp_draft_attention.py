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
"""Sliding-window draft attention over a DCP-sharded KV cache, in XLA.

A drafter's KV cache (e.g. the DSpark draft of MiniMax-M3) is sharded over
the decode-context-parallel group with the target's block tables: vLLM's
logical block ``j`` covers the ``dcp`` global blocks of 128 tokens
``j * dcp .. j * dcp + dcp - 1`` and has page ``block_tables[seq, j]`` on
every rank. Each rank stores only its own KV head. The ``head_ranks`` ranks
of the group holding one KV head (its tensor-parallel replicas, at most
``dcp``) split its tokens: global block ``b`` lives on the head's rank
``b % head_ranks``, as the ``(b % dcp) // head_ranks``-th of the
``dcp // head_ranks`` blocks its page holds (`_page_slot`). So a K/V write
moves nothing between ranks, and attention exchanges queries and partial
outputs only within the ranks of one head; with one rank per head
(``head_ranks == 1``) nothing crosses at all.

The draft attends a short query block (its draft tokens) over a window of
cached context, so this runs without the RPA kernels (the batched RPA DCP
kernels reject sliding windows):

1. `write_kv`: each rank writes its KV head's K/V of the tokens whose block
   it holds, the other tokens are dropped.
2. `attend_cached`: each rank attends its head's query heads (gathered from
   the head's ranks) over the window's blocks it holds and returns the
   partial output and its log-sum-exp; the caller merges the head's ranks'
   partials (`dcp_comm.cp_lse_ag_out_rs`) back to its own query heads.
3. `attend_block`: the query block over its own new K/V (causal or not, as
   the layer; the block is shorter than the window), on every rank with its
   own heads; merged with step 2 by LSE.

Together they equal the RPA kernel's sliding-window attention over the full
context (``q - k < window``, plus ``q >= k`` if causal; DSpark's query block
is non-causal), with the block written to the cache.

Pages keep the RPA shape ``[pages, S, words, p, D]`` (``p`` entries per
32-bit word) and the byte size of ``dcp // head_ranks`` KV heads. Read as
``S * words`` rows of one word each, a page holds its blocks' token-heads
(a token's K and V, adjacent entries) in position order, `_halves` of them
per word.
"""

import functools
import inspect
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import tight_vmem
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


_NEG_INF = float("-inf")


def _token_layout(seq_lens: jax.Array, cu_q_lens: jax.Array, num_tokens: int):
    """Per token: (sequence, global position, is-real); padding tokens get
    sequence 0 and position 0.

    Sums over a ``[sequences, tokens]`` membership mask (sequences on
    sublanes): a search plus gathers of per-sequence values by token cost
    ~50 us per 512 tokens and ~160 us per 8K on tpu7x."""
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    starts, ends = cu_q_lens[:-1, None], cu_q_lens[1:, None]
    inside = (starts <= tok[None, :]) & (tok[None, :] < ends)
    index = jnp.arange(seq_lens.shape[0], dtype=jnp.int32)[:, None]
    seq = jnp.sum(jnp.where(inside, index, 0), axis=0)
    # A token's position: its index past its sequence's end, from the length.
    shift = (seq_lens - cu_q_lens[1:])[:, None]
    pos = tok + jnp.sum(jnp.where(inside, shift, 0), axis=0)
    valid = tok < cu_q_lens[-1]
    return seq, jnp.where(valid, pos, 0), valid


def _page_slot(pos, *, page_size: int, dcp: int, head_ranks: int):
    """Global position -> (rank of its KV head's ``head_ranks`` holding it,
    token-head index in the page of its logical block)."""
    block = pos // page_size
    slot = (block % dcp) // head_ranks * page_size + pos % page_size
    return block % head_ranks, slot


def _halves(page_shape, page_blocks: int) -> int:
    """Token-heads per 32-bit word of a page (``[S, words, p, D]``) holding
    ``page_blocks`` blocks of one KV head."""
    _, words, packing, _ = page_shape
    if packing < 2:
        raise NotImplementedError("a token's K and V must share a 32-bit word")
    halves = min(packing // 2, page_blocks)
    # Every slot of a page must be one of its blocks' (the readers treat them so).
    if words * halves != page_blocks:
        raise NotImplementedError(f"{page_blocks} blocks do not fill page {page_shape}")
    return halves


def token_head_rows(key: jax.Array, value: jax.Array, halves: int, page_shape):
    """``[T, 1, D]`` K and V of one KV head -> ``[T, p / halves, D']`` half-word
    rows of pages ``[S, words, p, D']`` (K, V, zero padding)."""
    *_, packing, width = page_shape
    entries = packing // halves
    num_tokens, _, head_dim = key.shape
    # Equal values, laid out differently by XLA: stacked pairs feed
    # `write_pairs`' word views without a relayout (~300 us per 8K-token write
    # at TP8/DCP8), the concatenation keeps the one-head pages' write as fast.
    if halves == 2:
        kv = jnp.stack([key[:, 0], value[:, 0]], axis=1)
    else:
        kv = jnp.concatenate([key, value], axis=-1).reshape(num_tokens, 2, head_dim)
    return jnp.pad(kv, ((0, 0), (0, entries - 2), (0, width - head_dim)))


def _quantize(x: jax.Array, dtype, scale: float | None) -> jax.Array:
    if scale is not None:
        x = x.astype(jnp.float32) / scale
    if jnp.issubdtype(dtype, jnp.floating) and jnp.finfo(dtype).bits == 8:
        limit = float(jnp.finfo(dtype).max)
        x = jnp.clip(x, -limit, limit)
    return x.astype(dtype)


def write_kv(
    kv_cache: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    dcp: int,
    head_ranks: int,
    rank: jax.Array,
    k_scale: float | None = None,
    v_scale: float | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Write this step's K/V of this rank's KV head (``[T, 1, D]``) at the
    tokens' positions, keeping only the tokens of blocks this rank holds."""
    from vllm_torchtpu.models.minimax_m3.kernels.msa_write import max_write_runs

    num_tokens = key.shape[0]
    num_pages, page_size, words, packing, width = kv_cache.shape
    max_seqs = seq_lens.shape[0]
    tables = block_tables.reshape(max_seqs, -1)
    seq, pos, valid = _token_layout(seq_lens, cu_q_lens, num_tokens)
    holder, slot = _page_slot(pos, page_size=page_size, dcp=dcp, head_ranks=head_ranks)
    mine = valid & (holder == rank % head_ranks)
    page = jnp.where(mine, tables[seq, pos // page_size // dcp], num_pages)
    halves = _halves(kv_cache.shape[1:], dcp // head_ranks)
    rows = token_head_rows(
        _quantize(key, kv_cache.dtype, k_scale),
        _quantize(value, kv_cache.dtype, v_scale),
        halves,
        kv_cache.shape[1:],
    )
    # A request's tokens on this rank break at each of its blocks, so into at
    # most q // page_size + 2 runs, the bound `max_write_runs` takes.
    max_runs = max_write_runs(num_tokens, max_seqs, page_size)
    if halves == 2:
        word_rows = kv_cache.reshape(num_pages, page_size * words, packing, width)
        written = write_pairs(
            word_rows, rows, page, slot, max_runs=max_runs, interpret=interpret
        )
    else:
        # One token-head per word row ([.., 1, p, D], the page's own shape
        # when it holds one block).
        word_rows = kv_cache.reshape(num_pages, page_size * words, 1, packing, width)
        write = (
            write_runs if num_tokens >= _RUN_WRITE_MIN_TOKENS else write_runs_batched
        )
        rows = rows.reshape(num_tokens, 1, packing, width)
        written = write(
            word_rows, rows, page, slot, max_runs=max_runs, interpret=interpret
        )
    return written.reshape(kv_cache.shape)


# Prefill-sized writes take one grid step per run (`write_runs`); a decode
# batch's many short runs are all issued from one step (`write_runs_batched`),
# where an XLA scatter cost ~0.12 us per token (65 us per draft layer at 512
# tokens, c64 twin j-ded092da).
_RUN_WRITE_MIN_TOKENS = 1024


def write_runs_batched(
    kv_cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    interpret: bool = False,
) -> jax.Array:
    """`write_runs` from a single grid step: every run's power-of-two DMA
    pieces are started back to back, then all are waited for."""
    from vllm_torchtpu.models.minimax_m3.kernels.msa_write import token_runs

    num_pages, page_size = kv_cache.shape[:2]
    # A run fits a page and the rows.
    longest = min(page_size, rows.shape[0])
    pieces = [1 << b for b in reversed(range(longest.bit_length()))]
    num_runs, run_page, run_lo, run_hi, run_tok0 = token_runs(
        pages.astype(jnp.int32), slots.astype(jnp.int32), num_pages, max_runs
    )

    def kernel(
        n_ref, page_ref, lo_ref, hi_ref, tok0_ref, rows_hbm, cache_in, cache_out, sem
    ):
        del cache_in

        def run_copies(r):
            count = hi_ref[r] - lo_ref[r]
            copies = []
            for size in pieces:
                off = count & ~(2 * size - 1)  # after the larger pieces
                copy = pltpu.make_async_copy(
                    rows_hbm.at[pl.ds(tok0_ref[r] + off, size)],
                    cache_out.at[page_ref[r], pl.ds(lo_ref[r] + off, size)],
                    sem.at[0],
                )
                copies.append(((count & size) != 0, copy))
            return copies

        def start(r, carry):
            for todo, copy in run_copies(r):
                pl.when(todo)(copy.start)
            return carry

        def wait(r, carry):
            for todo, copy in run_copies(r):
                pl.when(todo)(copy.wait)
            return carry

        jax.lax.fori_loop(0, n_ref[0], start, 0)
        jax.lax.fori_loop(0, n_ref[0], wait, 0)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1,),
            in_specs=[hbm, hbm],
            out_specs=hbm,
            scratch_shapes=[pltpu.SemaphoreType.DMA((1,))],
        ),
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        # Operand 6 = the cache (after 5 scalar-prefetch operands and the rows).
        input_output_aliases={6: 0},
        interpret=interpret,
        name="dcp_draft_write_runs_batched",
    )(num_runs[None], run_page, run_lo, run_hi, run_tok0, rows, kv_cache)


def write_runs(
    kv_cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    interpret: bool = False,
) -> jax.Array:
    """``kv_cache.at[pages, slots].set(rows, mode="drop")`` by DMA: one grid
    step per token run (consecutive tokens taking consecutive slots of one
    page), copying the run's rows from HBM in power-of-two pieces."""
    from vllm_torchtpu.models.minimax_m3.kernels.msa_write import token_runs

    num_pages, page_size = kv_cache.shape[:2]
    # A run fits a page and the rows.
    longest = min(page_size, rows.shape[0])
    pieces = [1 << b for b in reversed(range(longest.bit_length()))]
    num_runs, run_page, run_lo, run_hi, run_tok0 = token_runs(
        pages.astype(jnp.int32), slots.astype(jnp.int32), num_pages, max_runs
    )

    def kernel(
        n_ref, page_ref, lo_ref, hi_ref, tok0_ref, rows_hbm, cache_in, cache_out, sem
    ):
        del cache_in
        r = pl.program_id(0)
        count = hi_ref[r] - lo_ref[r]
        copies = []
        for i, size in enumerate(pieces):
            off = count & ~(2 * size - 1)  # after the larger pieces
            copy = pltpu.make_async_copy(
                rows_hbm.at[pl.ds(tok0_ref[r] + off, size)],
                cache_out.at[page_ref[r], pl.ds(lo_ref[r] + off, size)],
                sem.at[i],
            )
            todo = (r < n_ref[0]) & ((count & size) != 0)
            pl.when(todo)(copy.start)
            copies.append((todo, copy))
        for todo, copy in copies:
            pl.when(todo)(copy.wait)

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(max_runs,),
            in_specs=[hbm, hbm],
            out_specs=hbm,
            scratch_shapes=[pltpu.SemaphoreType.DMA((len(pieces),))],
        ),
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        # Operand 6 = the cache (after 5 scalar-prefetch operands and the rows).
        input_output_aliases={6: 0},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="dcp_draft_write_runs",
    )(num_runs[None], run_page, run_lo, run_hi, run_tok0, rows, kv_cache)


def write_pairs(
    cache: jax.Array,
    rows: jax.Array,
    pages: jax.Array,
    slots: jax.Array,
    *,
    max_runs: int,
    interpret: bool = False,
) -> jax.Array:
    """Write token ``t``'s half-word ``rows[t]`` (``[p / 2, D]``) at half
    ``slots[t]`` of page ``pages[t]`` (dropped out of range) of a cache of
    two-half words (``[pages, W, p, D]`` of 8-bit entries), by DMA from one
    grid step, run by run (consecutive tokens taking consecutive halves of one
    page) in whole words.

    A run starting at an odd half keeps its first word's first half from the
    cache (read and merged in VMEM); a run ending at an odd half fills its
    last word's second half with the next row. That half is the position after
    the run, past what its request has written: cache positions are written in
    order, so no earlier write holds data there.
    """
    from vllm_torchtpu.models.minimax_m3.kernels.msa_write import token_runs

    num_pages, page_words, packing, width = cache.shape
    num_tokens = rows.shape[0]
    assert jax.dtypes.itemsize_bits(cache.dtype) == 8 and rows.shape[1] * 2 == packing
    num_runs, run_page, run_lo, run_hi, run_tok0 = token_runs(
        pages.astype(jnp.int32), slots.astype(jnp.int32), num_pages, max_runs
    )
    # Two word views of the rows: view 0 pairs rows (2w, 2w + 1), view 1 rows
    # (2w - 1, 2w). A run's words start at row tok0 - (lo odd) in one of them.
    view_words = num_tokens // 2 + 1
    views = jnp.concatenate(
        [
            jnp.pad(rows, ((0, 2 * view_words - num_tokens), (0, 0), (0, 0))),
            jnp.pad(rows, ((1, 2 * view_words - num_tokens - 1), (0, 0), (0, 0))),
        ]
    ).reshape(2 * view_words, packing, width)
    first_row = run_tok0 - run_lo % 2
    # A run's words fit a page and a view.
    longest = min(page_words, view_words)
    pieces = [1 << b for b in reversed(range(longest.bit_length()))]
    run_src = (first_row % 2) * view_words + (first_row + 1) // 2

    def kernel(
        n_ref,
        page_ref,
        lo_ref,
        hi_ref,
        src_ref,
        views_hbm,
        cache_in,
        cache_out,
        *scratch,
    ):
        old, new, sem = scratch
        del cache_in

        def edge_copies(r):
            word = (page_ref[r], pl.ds(lo_ref[r] // 2, 1))
            return [
                pltpu.make_async_copy(
                    cache_out.at[word], old.at[pl.ds(r, 1)], sem.at[0]
                ),
                pltpu.make_async_copy(
                    views_hbm.at[pl.ds(src_ref[r], 1)], new.at[pl.ds(r, 1)], sem.at[0]
                ),
            ]

        def run_copies(r):
            odd = lo_ref[r] % 2
            first = lo_ref[r] // 2 + odd
            count = (hi_ref[r] + 1) // 2 - first
            copies = [
                (
                    odd == 1,
                    pltpu.make_async_copy(
                        new.at[pl.ds(r, 1)],
                        cache_out.at[page_ref[r], pl.ds(lo_ref[r] // 2, 1)],
                        sem.at[0],
                    ),
                )
            ]
            for size in pieces:
                off = count & ~(2 * size - 1)  # after the larger pieces
                copy = pltpu.make_async_copy(
                    views_hbm.at[pl.ds(src_ref[r] + odd + off, size)],
                    cache_out.at[page_ref[r], pl.ds(first + off, size)],
                    sem.at[0],
                )
                copies.append(((count & size) != 0, copy))
            return copies

        def each_edge(method):
            def body(r, carry):
                @pl.when(lo_ref[r] % 2 == 1)
                def _():
                    for copy in edge_copies(r):
                        getattr(copy, method)()

                return carry

            jax.lax.fori_loop(0, n_ref[0], body, 0)

        each_edge("start")
        each_edge("wait")
        old_words = old.bitcast(jnp.uint32)
        new_words = new.bitcast(jnp.uint32)

        def merge(r, carry):
            # Entries fill a word from its lowest bits: half 0 is bits 0-15.
            @pl.when(lo_ref[r] % 2 == 1)
            def _():
                new_words[r] = (old_words[r] & low) | (new_words[r] & ~low)

            return carry

        low = jnp.uint32(0xFFFF)
        jax.lax.fori_loop(0, n_ref[0], merge, 0)

        def each_run(method):
            def body(r, carry):
                for todo, copy in run_copies(r):
                    pl.when(todo)(getattr(copy, method))
                return carry

            jax.lax.fori_loop(0, n_ref[0], body, 0)

        each_run("start")
        each_run("wait")

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    edge = pltpu.VMEM((max_runs, packing, width), cache.dtype)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1,),
            in_specs=[hbm, hbm],
            out_specs=hbm,
            scratch_shapes=[edge, edge, pltpu.SemaphoreType.DMA((1,))],
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        # Operand 6 = the cache (after 5 scalar-prefetch operands and the rows).
        input_output_aliases={6: 0},
        interpret=interpret,
        name="dcp_draft_write_pairs",
    )(num_runs[None], run_page, run_lo, run_hi, run_src, views, cache)


def _window_pages(
    seq_lens,
    block_tables,
    cu_q_lens,
    *,
    max_query_len,
    sliding_window,
    page_size,
    dcp,
    head_ranks,
    rank,
):
    """Per request: the cached length, and the pages of the window's logical
    blocks with the global block each page's first slot holds on this rank
    (``[R, L]`` each; slot ``k`` holds that block ``+ k * head_ranks``)."""
    max_seqs = seq_lens.shape[0]
    tables = block_tables.reshape(max_seqs, -1)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    cache_lens = seq_lens - q_lens
    # Blocks a request's tokens can see: the window of its first query token
    # through its last cached block; this rank's are every head_ranks-th.
    window_blocks = -(-(sliding_window + max_query_len) // page_size) + 1
    num_pages = -(-window_blocks // dcp) + 1
    first = jnp.maximum(cache_lens - sliding_window + 1, 0) // page_size
    holder = rank % head_ranks
    first_page = (first + (holder - first) % head_ranks) // dcp
    logical = first_page[:, None] + jnp.arange(num_pages)[None, :]
    pages = tables[
        jnp.arange(max_seqs)[:, None], jnp.minimum(logical, tables.shape[1] - 1)
    ]
    return q_lens, cache_lens, logical * dcp + holder, pages


def attend_cached(
    q: jax.Array,
    kv_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    head_ranks: int,
    max_query_len: int,
    sliding_window: int,
    sm_scale: float,
    dcp: int,
    rank: jax.Array,
    k_scale: float | None = None,
    v_scale: float | None = None,
    interpret: bool = False,
    uniform_blocks: bool = False,
    raw: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Partial attention of ``q [T, H, D]`` (every query head of this rank's
    KV head) over the cached tokens of the window held by this rank. Returns
    ``(out [T, H, D], lse [T, H] f32)``;
    ``lse`` is -inf where the rank holds none of a token's window.

    A request's query tokens are at most `max_query_len` (the draft block);
    with ``uniform_blocks`` every request has exactly that many, request
    ``r`` at rows ``r * max_query_len`` on (`_uniform_rows`).
    """
    num_tokens = q.shape[0]
    q_lens, cache_lens, blocks, pages = _window_pages(
        seq_lens,
        block_tables,
        cu_q_lens,
        max_query_len=max_query_len,
        sliding_window=sliding_window,
        page_size=kv_cache.shape[1],
        dcp=dcp,
        head_ranks=head_ranks,
        rank=rank,
    )
    out, lse = attend_pages(
        q,
        kv_cache,
        pages,
        blocks,
        cache_lens,
        q_lens,
        cu_q_lens[:-1],
        max_query_len=max_query_len,
        dcp=dcp,
        head_ranks=head_ranks,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        interpret=interpret,
        native_lse=raw,
    )
    if raw:
        # The kernel's own layout (`attend_pages` ``native_lse``), for
        # consumers that take it as it is (`pair_lse_merge.exchange_block`).
        return out, lse
    # Back to token order.
    if uniform_blocks:
        valid = _uniform_rows(seq_lens.shape[0], max_query_len, num_tokens)
        lse_t = _pad_rows(lse.reshape(-1, lse.shape[-1]), num_tokens, _NEG_INF)
    else:
        seq, _, valid = _token_layout(seq_lens, cu_q_lens, num_tokens)
        col = jnp.clip(jnp.arange(num_tokens) - cu_q_lens[seq], 0, max_query_len - 1)
        lse_t = jnp.where(valid[:, None], lse[seq, col], _NEG_INF)
    out_t = jnp.where(valid[:, None, None], out, 0.0)
    return out_t.astype(q.dtype), lse_t


def _uniform_rows(num_reqs: int, block: int, num_tokens: int) -> jax.Array:
    """Which of ``num_tokens`` rows belong to a request when ``num_reqs``
    requests of exactly ``block`` tokens lead them (the rest is padding)."""
    assert num_reqs * block <= num_tokens, (num_reqs, block, num_tokens)
    return jnp.arange(num_tokens) < num_reqs * block


def _pad_rows(x: jax.Array, num_rows: int, value: float) -> jax.Array:
    """``x`` with rows of ``value`` appended up to ``num_rows``."""
    pad = [(0, num_rows - x.shape[0])] + [(0, 0)] * (x.ndim - 1)
    return jnp.pad(x, pad, constant_values=value)


def attend_pages(
    q: jax.Array,
    kv_cache: jax.Array,
    pages: jax.Array,
    blocks: jax.Array,
    cache_lens: jax.Array,
    q_lens: jax.Array,
    q_starts: jax.Array,
    *,
    max_query_len: int,
    dcp: int,
    head_ranks: int,
    sliding_window: int,
    sm_scale: float,
    k_scale: float | None = None,
    v_scale: float | None = None,
    interpret: bool = False,
    native_lse: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Each request's query tokens (``q [T, H, D]`` rows ``q_starts[r]`` on,
    ``q_lens[r]`` of at most ``max_query_len``) over its window pages
    ``pages [R, L]`` (first slots' global blocks ``blocks [R, L]``,
    `_window_pages`), read straight from the paged cache: one grid step per
    request DMAs its L pages and query rows into VMEM (the next request's
    while this one computes), attends every query head to the pages'
    token-heads and DMAs the output rows back. Returns
    ``(out [T, H, D] in q's dtype, lse [R, Q, H] f32)``; rows of ``out`` no request
    owns are undefined.

    Replaces a page gather to HBM plus XLA attention over the gathered,
    re-laid-out and f32-converted window (~0.35 us per page of grid overhead
    and several full-window copies per request and layer).
    """
    num_tokens, num_heads, head_dim = q.shape
    num_reqs, num_cols = pages.shape
    q_block = max_query_len
    page_size, words, packing, width = kv_cache.shape[1:]
    halves = _halves(kv_cache.shape[1:], dcp // head_ranks)
    page_rows = page_size * words
    rows = q_block * num_heads
    # Word rows of the window; each holds `halves` keys.
    keys = num_cols * page_rows
    bits = jax.dtypes.itemsize_bits(kv_cache.dtype)
    # 8- and 16-bit caches (`_halves`) hold bf16-exact values: QK^T takes them
    # as bf16 with the scales applied to the f32 products; PV stays f32 (P is
    # not bf16-exact, and the kernel is VPU-bound, not MXU-bound).
    mxu_dtype = jnp.bfloat16
    k_mult = sm_scale * (1.0 if k_scale is None else k_scale)
    v_mult = 1.0 if v_scale is None else v_scale

    def unpack(raw, slot):
        part = (raw >> (bits * slot)) & ((1 << bits) - 1)
        return pltpu.bitcast(part.astype(jnp.dtype(f"uint{bits}")), kv_cache.dtype)

    def kernel(
        pages_ref,
        blocks_ref,
        cache_ref,
        qlen_ref,
        qstart_ref,
        q_hbm,
        kv_hbm,
        o_hbm,
        lse_ref,
        buf,
        q_buf,
        o_buf,
        sem,
        q_sem,
        o_sem,
    ):
        r = pl.program_id(0)
        slot = r % 2

        def copies(req, dst):
            """(condition or None for always, copy) pairs fetching a request's
            pages and query rows."""
            # A request with no cached tokens (a draft's padding rows) has
            # nothing to fetch or attend.
            live = cache_ref[req] > 0
            pages = [
                (
                    live,
                    pltpu.make_async_copy(
                        kv_hbm.at[pages_ref[req * num_cols + c]],
                        buf.at[dst, c],
                        sem.at[dst],
                    ),
                )
                for c in range(num_cols)
            ]
            # A full block is one copy; a shorter request copies its rows one
            # by one (rows past its tokens stay stale and are masked).
            full = qlen_ref[req] == q_block
            rows_in = [
                (
                    full,
                    pltpu.make_async_copy(
                        q_hbm.at[pl.ds(qstart_ref[req], q_block)],
                        q_buf.at[dst],
                        q_sem.at[dst],
                    ),
                )
            ] + [
                (
                    jnp.logical_not(full) & (j < qlen_ref[req]),
                    pltpu.make_async_copy(
                        q_hbm.at[pl.ds(qstart_ref[req] + j, 1)],
                        q_buf.at[dst, pl.ds(j, 1)],
                        q_sem.at[dst],
                    ),
                )
                for j in range(q_block)
            ]
            return pages + rows_in

        def run(pairs, method):
            for todo, cp in pairs:
                op = getattr(cp, method)
                if todo is None:
                    op()
                else:
                    pl.when(todo)(op)

        @pl.when(r == 0)
        def _():
            run(copies(0, 0), "start")

        @pl.when(r + 1 < pl.num_programs(0))
        def _():
            run(copies(r + 1, 1 - slot), "start")

        run(copies(r, slot), "wait")

        def rows_out(req, src):
            full = qlen_ref[req] == q_block
            return [
                (
                    full,
                    pltpu.make_async_copy(
                        o_buf.at[src],
                        o_hbm.at[pl.ds(qstart_ref[req], q_block)],
                        o_sem.at[src],
                    ),
                )
            ] + [
                (
                    jnp.logical_not(full) & (j < qlen_ref[req]),
                    pltpu.make_async_copy(
                        o_buf.at[src, pl.ds(j, 1)],
                        o_hbm.at[pl.ds(qstart_ref[req] + j, 1)],
                        o_sem.at[src],
                    ),
                )
                for j in range(q_block)
            ]

        @pl.when(r > 1)
        def _():
            run(rows_out(r - 2, slot), "wait")

        def attend():
            cache_len = cache_ref[r]
            col = jax.lax.broadcasted_iota(jnp.int32, (1, keys), 1)
            row_k = jax.lax.broadcasted_iota(jnp.int32, (keys, 1), 0)
            q_tok = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0) // num_heads
            # [L * S * words, width] u32: one word row each, `packing` entries
            # (token-head h's K and V at 2h, 2h + 1), lowest bits first.
            kv_u32 = buf.at[slot].bitcast(jnp.uint32).reshape(keys, width)[...]
            # [Q, H, D] -> [Q * H, D]: row q * H + h is token q, head h.
            qv = q_buf[slot].reshape(rows, head_dim).astype(mxu_dtype)

            def key_positions(index, half):
                """Global position of half `half` of each word row ``index``."""
                pos = jnp.zeros(index.shape, jnp.int32)
                for c in range(num_cols):
                    token_head = (index - c * page_rows) * halves + half
                    block = (
                        blocks_ref[r * num_cols + c]
                        + token_head // page_size * head_ranks
                    )
                    pos = jnp.where(
                        index // page_rows == c,
                        block * page_size + token_head % page_size,
                        pos,
                    )
                return pos

            scores, values = [], []
            for half in range(halves):
                k_pos = key_positions(col, half)
                visible = (
                    (k_pos < cache_len)
                    & (cache_len + q_tok - k_pos < sliding_window)
                    & (q_tok < qlen_ref[r])
                )
                # Masked keys must hold finite values (the cache is
                # zero-initialised and writes clip): an add, unlike a select,
                # would carry a NaN score through.
                bias = jnp.where(visible, 0.0, _NEG_INF)
                k = unpack(kv_u32, 2 * half)[:, :head_dim].astype(mxu_dtype)
                v = unpack(kv_u32, 2 * half + 1)[:, :head_dim].astype(jnp.float32)
                values.append(jnp.where(key_positions(row_k, half) < cache_len, v, 0.0))
                s = jax.lax.dot_general(
                    qv, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
                )
                scores.append(s * k_mult + bias)
            m = functools.reduce(
                jnp.maximum, [jnp.max(s, axis=-1, keepdims=True) for s in scores]
            )
            m_safe = jnp.where(m > _NEG_INF, m, 0.0)
            probs = [jnp.exp(s - m_safe) for s in scores]
            denom = functools.reduce(
                jnp.add, [jnp.sum(p, axis=-1, keepdims=True) for p in probs]
            )
            safe = jnp.where(denom > 0, denom, 1.0)
            out = functools.reduce(
                jnp.add,
                [
                    jnp.dot(p, v, preferred_element_type=jnp.float32)
                    for p, v in zip(probs, values)
                ],
            )
            out = out * (v_mult / safe)
            o_buf[slot] = out.reshape(q_block, num_heads, head_dim).astype(o_buf.dtype)
            lse = jnp.where(denom > 0, m_safe + jnp.log(safe), _NEG_INF)
            # Moves the [rows, 1] column to a [1, rows] row (sublanes to lanes).
            diag = jax.lax.broadcasted_iota(
                jnp.int32, (rows, rows), 0
            ) == jax.lax.broadcasted_iota(jnp.int32, (rows, rows), 1)
            lse_ref[0] = jnp.sum(jnp.where(diag, lse, 0.0), axis=0, keepdims=True)

        pl.when(cache_ref[r] > 0)(attend)

        @pl.when(cache_ref[r] == 0)
        def _():
            o_buf[slot] = jnp.zeros(o_buf.shape[1:], o_buf.dtype)
            lse_ref[...] = jnp.full(lse_ref.shape, _NEG_INF, lse_ref.dtype)

        # The rows go out while the next requests compute; a slot's copies are
        # waited for before it is refilled, the last two at the end.
        run(rows_out(r, slot), "start")

        @pl.when(r + 1 == pl.num_programs(0))
        def _():
            @pl.when(r > 0)
            def _():
                run(rows_out(r - 1, 1 - slot), "wait")

            run(rows_out(r, slot), "wait")

    out, lse = tight_vmem.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(num_reqs,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((1, 1, rows), lambda r, *_: (r, 0, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM(
                    (2, num_cols, page_size, words, packing, width), kv_cache.dtype
                ),
                pltpu.VMEM((2, q_block, num_heads, head_dim), q.dtype),
                pltpu.VMEM((2, q_block, num_heads, head_dim), q.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_tokens, num_heads, head_dim), q.dtype),
            jax.ShapeDtypeStruct((num_reqs, 1, rows), jnp.float32),
        ],
        # The next request's pages are fetched in the current step.
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=48 * 1024 * 1024
        ),
        interpret=interpret,
        name="dcp_draft_attend_pages",
    )(
        pages.reshape(-1).astype(jnp.int32),
        blocks.reshape(-1).astype(jnp.int32),
        cache_lens.astype(jnp.int32),
        q_lens.astype(jnp.int32),
        q_starts.astype(jnp.int32),
        q,
        kv_cache,
    )
    if native_lse:
        # [R, 1, Q * H] as the kernel writes it (request r's rows lane-major):
        # the reshape below is a relayout XLA runs as its own op.
        return out, lse
    return out, lse.reshape(num_reqs, q_block, num_heads)


def attend_cached_reference(
    q: jax.Array,
    kv_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    head_ranks: int,
    max_query_len: int,
    sliding_window: int,
    sm_scale: float,
    dcp: int,
    rank: jax.Array,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """`attend_cached` in XLA over gathered pages (the kernel's reference)."""
    num_tokens, num_heads, head_dim = q.shape
    page_size, words, packing, width = kv_cache.shape[1:]
    max_seqs = seq_lens.shape[0]
    q_lens, cache_lens, blocks, pages = _window_pages(
        seq_lens,
        block_tables,
        cu_q_lens,
        max_query_len=max_query_len,
        sliding_window=sliding_window,
        page_size=page_size,
        dcp=dcp,
        head_ranks=head_ranks,
        rank=rank,
    )
    halves = _halves(kv_cache.shape[1:], dcp // head_ranks)
    # [R, L, token-head, entries, D]: a page's token-heads in slot order.
    kv = kv_cache[pages].reshape(
        max_seqs, pages.shape[1], page_size * words * halves, packing // halves, width
    )
    k = kv[..., 0, :head_dim].reshape(max_seqs, -1, head_dim).astype(jnp.float32)
    v = kv[..., 1, :head_dim].reshape(max_seqs, -1, head_dim).astype(jnp.float32)
    if k_scale is not None:
        k = k * k_scale
    if v_scale is not None:
        v = v * v_scale
    token_head = jnp.arange(page_size * words * halves)
    block = blocks[:, :, None] + token_head // page_size * head_ranks
    k_pos = (block * page_size + token_head % page_size).reshape(max_seqs, -1)

    # Query tokens padded per request: [R, Q].
    slot = jnp.arange(max_query_len)
    tok = cu_q_lens[:-1, None] + slot[None, :]
    tok_valid = slot[None, :] < q_lens[:, None]
    q_pos = cache_lens[:, None] + slot[None, :]
    qr = q[jnp.minimum(tok, num_tokens - 1)].astype(jnp.float32)  # [R, Q, H, D]
    s = jnp.einsum("rqhd,rkd->rqhk", qr, k) * sm_scale
    visible = (
        (k_pos[:, None, :] < cache_lens[:, None, None])
        & (q_pos[:, :, None] - k_pos[:, None, :] < sliding_window)
        & tok_valid[:, :, None]
    )  # [R, Q, K]
    s = jnp.where(visible[:, :, None, :], s, _NEG_INF)
    m = jnp.max(s, axis=-1, keepdims=True)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.exp(s - m_safe)
    l = jnp.sum(p, axis=-1, keepdims=True)
    out = jnp.einsum("rqhk,rkd->rqhd", p, v) / jnp.where(l > 0, l, 1.0)
    lse = jnp.where(l > 0, m_safe + jnp.log(jnp.where(l > 0, l, 1.0)), _NEG_INF)[..., 0]
    # Back to token order.
    seq, _, valid = _token_layout(seq_lens, cu_q_lens, num_tokens)
    col = jnp.arange(num_tokens) - cu_q_lens[seq]
    col = jnp.clip(col, 0, max_query_len - 1)
    out_t = jnp.where(valid[:, None, None], out[seq, col], 0.0)
    lse_t = jnp.where(valid[:, None], lse[seq, col], _NEG_INF)
    return out_t.astype(q.dtype), lse_t


def attend_block(
    q: jax.Array,
    key: jax.Array,
    value: jax.Array,
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    *,
    max_query_len: int,
    sm_scale: float,
    causal: bool = True,
    uniform_blocks: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Attention of each request's query tokens (``q [T, H, D]``) over its
    own new K/V (``[T, G, D]``), causal or over the whole block. The block is
    shorter than the window. Returns ``(out, lse)``. ``uniform_blocks`` as in
    `attend_cached`: the requests' blocks are reshapes of the leading rows."""
    num_tokens, num_heads, head_dim = q.shape
    kv_heads = key.shape[1]
    max_seqs = seq_lens.shape[0]
    slot = jnp.arange(max_query_len)
    group = num_heads // kv_heads
    if uniform_blocks:
        valid = _uniform_rows(max_seqs, max_query_len, num_tokens)
        lead = max_seqs * max_query_len
        blocks = (max_seqs, max_query_len)
        q_blocks = q[:lead].reshape(*blocks, num_heads, head_dim)
        k_blocks = key[:lead].reshape(*blocks, kv_heads, head_dim)
        v_blocks = value[:lead].reshape(*blocks, kv_heads, head_dim)
        tok_valid = jnp.ones(blocks, dtype=bool)
    else:
        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        tok = jnp.minimum(cu_q_lens[:-1, None] + slot[None, :], num_tokens - 1)
        tok_valid = slot[None, :] < q_lens[:, None]
        q_blocks, k_blocks, v_blocks = q[tok], key[tok], value[tok]
    qr = q_blocks.astype(jnp.float32).reshape(
        max_seqs, max_query_len, kv_heads, group, head_dim
    )
    k = k_blocks.astype(jnp.float32)  # [R, Q, G, D]
    v = v_blocks.astype(jnp.float32)
    s = jnp.einsum("rqgxd,rkgd->rqgxk", qr, k) * sm_scale
    visible = tok_valid[:, None, :] & tok_valid[:, :, None]
    if causal:
        visible = visible & (slot[None, :, None] >= slot[None, None, :])
    s = jnp.where(visible[:, :, None, None, :], s, _NEG_INF)
    m = jnp.max(s, axis=-1, keepdims=True)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.exp(s - m_safe)
    l = jnp.sum(p, axis=-1, keepdims=True)
    out = jnp.einsum("rqgxk,rkgd->rqgxd", p, v) / jnp.where(l > 0, l, 1.0)
    lse = jnp.where(l > 0, m_safe + jnp.log(jnp.where(l > 0, l, 1.0)), _NEG_INF)
    out = out.reshape(max_seqs, max_query_len, num_heads, head_dim)
    lse = lse.reshape(max_seqs, max_query_len, num_heads)
    if uniform_blocks:
        out_t = _pad_rows(out.reshape(-1, num_heads, head_dim), num_tokens, 0.0)
        lse_t = _pad_rows(lse.reshape(-1, num_heads), num_tokens, _NEG_INF)
        out_t = jnp.where(valid[:, None, None], out_t, 0.0)
        return out_t.astype(q.dtype), lse_t
    seq, _, valid = _token_layout(seq_lens, cu_q_lens, num_tokens)
    col = jnp.clip(jnp.arange(num_tokens) - cu_q_lens[seq], 0, max_query_len - 1)
    out_t = jnp.where(valid[:, None, None], out[seq, col], 0.0)
    lse_t = jnp.where(valid[:, None], lse[seq, col], _NEG_INF)
    return out_t.astype(q.dtype), lse_t


def merge(out_a, lse_a, out_b, lse_b):
    """LSE-weighted merge of two partial attentions (f32 LSE)."""
    m = jnp.maximum(lse_a, lse_b)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    wa = jnp.exp(lse_a - m_safe)
    wb = jnp.exp(lse_b - m_safe)
    norm = wa + wb
    norm_safe = jnp.where(norm > 0, norm, 1.0)
    out = (
        wa[..., None] * out_a.astype(jnp.float32) + wb[..., None] * out_b.astype(jnp.float32)
    ) / norm_safe[..., None]
    lse = jnp.where(norm > 0, m_safe + jnp.log(norm_safe), _NEG_INF)
    return out.astype(out_a.dtype), lse


# ---------------------------------------------------------------------------
# Torch entry points (one jax_op per static configuration)
#
# Build them eagerly with `build_ops` (from module setup): `pallas.jax_op`
# inspects the function signature, which Dynamo cannot trace inside a
# compiled forward.
# ---------------------------------------------------------------------------


def build_ops(
    *,
    dcp: int,
    head_ranks: int,
    max_query_len: int,
    sliding_window: int,
    sm_scale: float,
    causal: bool,
    k_scale: float | None,
    v_scale: float | None,
    uniform_blocks: bool = False,
) -> dict[str, Any]:
    """The write, cached-attention and block ops of one draft layer.
    ``uniform_blocks``: every forward's query rows are ``max_query_len``
    tokens per request (`attend_cached`)."""
    extra: dict[str, Any] = {}
    if envs.TPU_DRAFT_DCP_PAIR_MERGE and envs.TPU_DRAFT_DCP_PAIR_MERGE_BLOCK:
        # The attention with the kernel's own LSE layout, for the pair merge
        # op that folds the block op in (`arm_pair_merge`).
        extra["attend_raw"] = _attend_op(
            head_ranks,
            max_query_len,
            sliding_window,
            sm_scale,
            dcp,
            k_scale,
            v_scale,
            uniform_blocks,
            raw=True,
        )
    return dict(
        write=_write_op(dcp, head_ranks, k_scale, v_scale),
        attend=_attend_op(
            head_ranks,
            max_query_len,
            sliding_window,
            sm_scale,
            dcp,
            k_scale,
            v_scale,
            uniform_blocks,
        ),
        block=_block_op(max_query_len, sm_scale, causal, uniform_blocks),
        # The block op's static arguments (`_block_key`).
        block_args=dict(
            max_query_len=max_query_len,
            sm_scale=sm_scale,
            causal=causal,
            uniform_blocks=uniform_blocks,
            k_scale=k_scale,
            v_scale=v_scale,
        ),
        **extra,
    )


@functools.cache
def _write_op(dcp: int, head_ranks: int, k_scale: float | None, v_scale: float | None):
    from torch_tpu._internal import pallas

    def _write(
        kv_cache: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        dcp_rank: jax.Array,
    ) -> jax.Array:
        return write_kv(
            kv_cache,
            key,
            value,
            seq_lens,
            block_tables,
            query_start_loc,
            dcp=dcp,
            head_ranks=head_ranks,
            rank=dcp_rank[0],
            k_scale=k_scale,
            v_scale=v_scale,
        )

    name = f"pallas::dcp_draft_write_d{dcp}_h{head_ranks}_k{k_scale}_v{v_scale}"
    name = name.replace(".", "p")
    op = pallas.jax_op(name, _write, donate_argnums=(0,))
    op.register_fake(lambda kv_cache, *args: torch.empty_like(kv_cache))
    return op


@functools.cache
def _attend_op(
    head_ranks: int,
    max_query_len: int,
    sliding_window: int,
    sm_scale: float,
    dcp: int,
    k_scale: float | None,
    v_scale: float | None,
    uniform_blocks: bool,
    raw: bool = False,
):
    from torch_tpu._internal import pallas

    def _attend(
        q_all: jax.Array,
        kv_cache: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        dcp_rank: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return attend_cached(
            q_all,
            kv_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            head_ranks=head_ranks,
            max_query_len=max_query_len,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            dcp=dcp,
            rank=dcp_rank[0],
            k_scale=k_scale,
            v_scale=v_scale,
            uniform_blocks=uniform_blocks,
            raw=raw,
        )

    name = (
        f"pallas::dcp_draft_attend_h{head_ranks}_q{max_query_len}_w{sliding_window}"
        f"_s{sm_scale:.8f}_d{dcp}_k{k_scale}_v{v_scale}_u{int(uniform_blocks)}"
        f"{'_raw' if raw else ''}"
    ).replace(".", "p")
    op = pallas.jax_op(name, _attend)
    if raw:
        op.register_fake(
            lambda q, kv_cache, seq_lens, *args: (
                torch.empty_like(q),
                torch.empty(
                    (seq_lens.shape[0], 1, max_query_len * q.shape[1]),
                    dtype=torch.float32,
                    device=q.device,
                ),
            )
        )
        return op
    op.register_fake(
        lambda q, *args: (
            torch.empty_like(q),
            torch.empty(q.shape[:2], dtype=torch.float32, device=q.device),
        )
    )
    return op


@functools.cache
def _block_op(
    max_query_len: int, sm_scale: float, causal: bool, uniform_blocks: bool
):
    from torch_tpu._internal import pallas

    def _block(
        q: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_lens: jax.Array,
        query_start_loc: jax.Array,
        ctx_out: jax.Array,
        ctx_lse: jax.Array,
    ) -> jax.Array:
        out, lse = attend_block(
            q, key, value, seq_lens, query_start_loc,
            max_query_len=max_query_len, sm_scale=sm_scale, causal=causal,
            uniform_blocks=uniform_blocks,
        )
        return merge(ctx_out.astype(out.dtype), ctx_lse, out, lse)[0]

    name = (
        f"pallas::dcp_draft_block_q{max_query_len}_s{sm_scale:.8f}_c{int(causal)}"
        f"_u{int(uniform_blocks)}"
    ).replace(".", "p")
    op = pallas.jax_op(name, _block)
    op.register_fake(lambda q, *args: torch.empty_like(q))
    return op


def write_layers(
    caches,
    keys: jax.Array,
    values: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    *,
    dcp: int,
    head_ranks: int,
    rank: jax.Array,
    k_scale: float | None,
    v_scale: float | None,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """`write_kv` of every layer (``keys [L, T, 1, D]``) in one trace, so
    XLA computes the shared token layout, slots and runs once."""
    return tuple(
        write_kv(
            cache,
            keys[i],
            values[i],
            seq_lens,
            block_tables,
            query_start_loc,
            dcp=dcp,
            head_ranks=head_ranks,
            rank=rank,
            k_scale=k_scale,
            v_scale=v_scale,
            interpret=interpret,
        )
        for i, cache in enumerate(caches)
    )


@functools.cache
def write_layers_op(
    dcp: int,
    head_ranks: int,
    k_scale: float | None,
    v_scale: float | None,
    num_layers: int,
):
    """`write_kv` of ``num_layers`` draft layers in one op:
    ``(kv_cache_0, .., kv_cache_{L-1}, keys [L, T, 1, D], values, seq_lens,
    block_tables, query_start_loc, dcp_rank) -> new caches``. Build it outside
    a compiled forward (see `build_ops`)."""
    from torch_tpu._internal import pallas

    def _write_layers(*args: jax.Array) -> tuple[jax.Array, ...] | jax.Array:
        caches = args[:num_layers]
        keys, values, seq_lens, block_tables, query_start_loc, dcp_rank = args[
            num_layers:
        ]
        new_caches = write_layers(
            caches,
            keys,
            values,
            seq_lens,
            block_tables,
            query_start_loc,
            dcp=dcp,
            head_ranks=head_ranks,
            rank=dcp_rank[0],
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return new_caches[0] if num_layers == 1 else new_caches

    names = [f"kv_cache_{i}" for i in range(num_layers)] + [
        "keys",
        "values",
        "seq_lens",
        "block_tables",
        "query_start_loc",
        "dcp_rank",
    ]
    # jax_op reads the arguments (all arrays) from the signature and the
    # schema's outputs from the return annotation: a one-tuple reads as one
    # Tensor (not a tuple of one), so one layer returns its bare cache.
    _write_layers.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=jax.Array
            )
            for n in names
        ],
        return_annotation=(
            jax.Array if num_layers == 1 else tuple[(jax.Array,) * num_layers]
        ),
    )
    name = (
        f"pallas::dcp_draft_write_l{num_layers}_d{dcp}_h{head_ranks}"
        f"_k{k_scale}_v{v_scale}"
    )
    op = pallas.jax_op(
        name.replace(".", "p"), _write_layers, donate_argnums=tuple(range(num_layers))
    )

    def _fake(*args):
        caches = tuple(torch.empty_like(c) for c in args[:num_layers])
        return caches[0] if num_layers == 1 else caches

    op.register_fake(_fake)
    return op


def write_layers_with_dcp(
    *,
    keys: torch.Tensor,
    values: torch.Tensor,
    kv_caches: list[torch.Tensor],
    attn_metadata: Any,
    dcp_rank: torch.Tensor,
    op: Any,
) -> None:
    """`write_with_dcp` of every draft layer (``keys`` ``[L, T, 1, D]``) at
    once, for layers sharing one attention metadata (`write_layers_op`)."""
    assert keys.shape[2] == 1, keys.shape
    new_caches = op(
        *kv_caches,
        keys,
        values,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        dcp_rank,
    )
    if isinstance(new_caches, torch.Tensor):  # one layer: a bare cache
        new_caches = (new_caches,)
    for cache, new_cache in zip(kv_caches, new_caches, strict=True):
        cache.copy_(new_cache)


def write_with_dcp(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    dcp_rank: torch.Tensor,
    ops: dict[str, Any],
) -> None:
    """Write the drafter's context K/V (``[T, 1, D]``, own KV head) into the
    DCP-sharded cache: each rank keeps the tokens of the blocks it holds."""
    assert key.shape[1] == 1, key.shape
    new_cache = ops["write"](
        kv_cache,
        key,
        value,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        dcp_rank,
    )
    kv_cache.copy_(new_cache)


def forward_with_dcp(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    dcp_rank: torch.Tensor,
    dcp: int,
    kv_head_replicas: int,
    head_group: Any | None,
    k_scale: float | None,
    v_scale: float | None,
    ops: dict[str, Any],
    pair_meta: torch.Tensor | None = None,
) -> torch.Tensor:
    """The drafter's attention of its query block (``query [T, H, D]``, own
    heads; ``key``/``value [T, 1, D]``, own KV head) over the window of a
    DCP-sharded cache plus the block itself. The block is not written.
    ``pair_meta``: set by `arm_pair_merge`, the partials merge through
    ``ops["pair_merge"]`` (one exchange with the head's other rank), or with
    ``ops["pair_merge_block"]`` the exchange, the block's own attention and
    the final merge run in one op."""
    from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs

    del dcp, kv_head_replicas  # shared layer args (the cache spec uses them)
    # The window's K/V of this rank's KV head are split over `head_group`
    # (the head's ranks), so only their query heads and partials cross.
    q_all = query
    if head_group is not None:
        q_all = head_group.all_gather(query.contiguous(), dim=1)
    folded = pair_meta is not None and "pair_merge_block" in ops
    ctx_out, ctx_lse = ops["attend_raw" if folded else "attend"](
        q_all,
        kv_cache,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        dcp_rank,
    )
    if folded:
        # The kernel's own LSE layout and the block's K/V as they are: the op
        # quantizes them like the cache.
        return ops["pair_merge_block"](
            ctx_out,
            ctx_lse,
            pair_meta,
            query,
            key.to(query.dtype),
            value.to(query.dtype),
            attn_metadata.seq_lens,
        )
    if pair_meta is not None:
        ctx_out, ctx_lse = ops["pair_merge"](ctx_out, ctx_lse, pair_meta)
    else:
        ctx_out, ctx_lse = cp_lse_ag_out_rs(
            ctx_out, ctx_lse, head_group, return_lse=True
        )
    # The block's K/V as the cache would hold them (quantized), own KV head.
    if kv_cache.dtype != key.dtype and k_scale is not None:
        key = (key.float() / k_scale).to(kv_cache.dtype).float() * k_scale
        value = (value.float() / v_scale).to(kv_cache.dtype).float() * v_scale
    return ops["block"](
        query,
        key.to(query.dtype),
        value.to(query.dtype),
        attn_metadata.seq_lens,
        attn_metadata.query_start_loc,
        ctx_out,
        ctx_lse,
    )


# ---------------------------------------------------------------------------
# Pair merge (TPU_DRAFT_DCP_PAIR_MERGE): the partials of a two-rank head group
# merge in one exchange (kernels/collectives/pair_lse_merge.py) instead of
# cp_lse_ag_out_rs's all-gather + reduce-scatter. Armed once after load, over
# the mesh of the armed Pallas all-reduce.
# ---------------------------------------------------------------------------
SELFTEST_TIMEOUT_S = 300


@functools.cache
def _pair_merge_op(mesh: Any, perm: tuple[tuple[int, int], ...]):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.collectives import pair_lse_merge

    axis = mesh.axis_names[0]

    def body(out, lse, meta):
        if pair_lse_merge.supported(out.shape, out.dtype):
            return pair_lse_merge.exchange(out, lse, meta)
        block = meta.reshape(-1)[1]
        return pair_lse_merge.exchange_xla(out, lse, block, axis, list(perm))

    specs = (P(axis),) * 3
    out_specs = (P(axis), P(axis))
    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=out_specs, check_vma=False
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(
        out: jax.Array, lse: jax.Array, meta: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        args = (out, lse, meta)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        "pallas::dcp_draft_pair_merge",
        fn,
        mesh=mesh,
        input_partition_specs=specs,
        output_partition_specs=out_specs,
    )
    op.register_fake(
        lambda out, lse, meta: (
            out.new_empty((out.shape[0], out.shape[1] // 2, out.shape[2])),
            lse.new_empty((lse.shape[0], lse.shape[1] // 2), dtype=torch.float32),
        )
    )
    return op


def _pair_selftest(
    op, meta: torch.Tensor, head_group: Any, shape, shape_rows, device
) -> bool:
    """Whether the pair merge gives this rank cp_lse_ag_out_rs's result to a
    bf16 rounding, at each of ``shape_rows`` rows of ``[2H, D]``. Collective;
    a wrong peer hangs, so a timeout dumps every stack and exits."""
    import faulthandler

    from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs

    gen = torch.Generator().manual_seed(7 + int(head_group.rank_in_group))
    ok = True
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for rows in shape_rows:
            dims = (rows, *shape)
            out = torch.randn(dims, generator=gen).to(torch.bfloat16).to(device)
            lse = (torch.randn(dims[:2], generator=gen) * 3).to(device)
            want, want_lse = cp_lse_ag_out_rs(out, lse, head_group, return_lse=True)
            got, got_lse = op(out, lse, meta)
            ok &= bool(
                torch.allclose(
                    got.float().cpu(), want.float().cpu(), rtol=2**-7, atol=1e-3
                )
                and torch.allclose(got_lse.cpu(), want_lse.float().cpu(), atol=1e-5)
            )
    finally:
        faulthandler.cancel_dump_traceback_later()
    return ok


def _round_trip(x: jax.Array, scale: float, quantized) -> jax.Array:
    """x as the draft cache holds it (`forward_with_dcp`'s block K/V)."""
    stored = (x.astype(jnp.float32) / scale).astype(quantized).astype(jnp.float32)
    return (stored * scale).astype(x.dtype)


@functools.cache
def _pair_merge_block_op(
    mesh: Any,
    perm: tuple[tuple[int, int], ...],
    max_query_len: int,
    sm_scale: float,
    causal: bool,
    k_scale: float | None,
    v_scale: float | None,
    cache_dtype: str | None,
):
    """`_pair_merge_op` with the draft block's own attention and the final
    merge (`_block_op`, uniform blocks) folded into the exchange kernel
    (`pair_lse_merge.exchange_block`); the unfused ops past its cap. Takes
    the attention's `attend_raw` outputs and the block's K/V before the
    cache quantization (``k_scale``, ``v_scale`` into ``cache_dtype``)."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.collectives import pair_lse_merge

    axis = mesh.axis_names[0]

    quantized = jnp.dtype(cache_dtype) if k_scale is not None else None

    def body(out, lse, meta, q, key, value, seq_lens):
        if pair_lse_merge.supported(out.shape, out.dtype):
            return pair_lse_merge.exchange_block(
                out,
                lse,
                meta,
                q,
                key,
                value,
                q_block=max_query_len,
                sm_scale=sm_scale,
                causal=causal,
                k_scale=k_scale,
                v_scale=v_scale,
                cache_dtype=quantized,
            )
        # The unfused ops on the attention's token-order outputs
        # (`attend_cached`'s uniform-block layout).
        num_tokens, two_h, _ = out.shape
        valid = jnp.arange(num_tokens) < seq_lens.shape[0] * max_query_len
        out = jnp.where(valid[:, None, None], out, 0).astype(out.dtype)
        lse = _pad_rows(lse.reshape(-1, two_h), num_tokens, _NEG_INF)
        if quantized is not None:
            key = _round_trip(key, k_scale, quantized)
            value = _round_trip(value, v_scale, quantized)
        block = meta.reshape(-1)[1]
        ctx, ctx_lse = pair_lse_merge.exchange_xla(out, lse, block, axis, list(perm))
        out_b, lse_b = attend_block(
            q,
            key,
            value,
            seq_lens,
            None,
            max_query_len=max_query_len,
            sm_scale=sm_scale,
            causal=causal,
            uniform_blocks=True,
        )
        return merge(ctx.astype(out_b.dtype), ctx_lse, out_b, lse_b)[0]

    specs = (P(axis),) * 7
    sharded = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P(axis), check_vma=False
    )

    # torch_tpu reads the operand types from these annotations.
    def fn(
        out: jax.Array,
        lse: jax.Array,
        meta: jax.Array,
        q: jax.Array,
        key: jax.Array,
        value: jax.Array,
        seq_lens: jax.Array,
    ) -> jax.Array:
        args = (out, lse, meta, q, key, value, seq_lens)
        return sharded(
            *(jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(args, specs))
        )

    name = (
        f"pallas::dcp_draft_pair_merge_block_q{max_query_len}_s{sm_scale:.8f}"
        f"_c{int(causal)}_k{k_scale}_v{v_scale}_{cache_dtype}"
    ).replace(".", "p")
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        name, fn, mesh=mesh, input_partition_specs=specs, output_partition_specs=P(axis)
    )
    op.register_fake(lambda out, lse, meta, q, *args: torch.empty_like(q))
    return op


def _block_key(layer: Any) -> tuple | None:
    """`_pair_merge_block_op`'s static arguments for a layer whose block op
    the pair merge can fold in (uniform blocks), else None."""
    ops = layer.impl.sharded_draft_dcp["ops"]
    block_args = ops.get("block_args")
    if block_args is None or "attend_raw" not in ops:
        return None
    if not block_args["uniform_blocks"]:
        return None
    quantized = layer.impl.kv_cache_quantized_dtype
    return (
        block_args["max_query_len"],
        block_args["sm_scale"],
        block_args["causal"],
        block_args["k_scale"],
        block_args["v_scale"],
        None if quantized is None else str(quantized).removeprefix("torch."),
    )


def _arm_pair_merge_block(layers, mesh, perm, pair_op, meta, shape, tp) -> dict:
    """`TPU_DRAFT_DCP_PAIR_MERGE_BLOCK`: the folded block op per distinct
    block configuration of ``layers``, each self-tested against the unfused
    pair merge + block op on a decode block; {} when off or a test fails on
    any rank. Collective over the TP group."""
    if not envs.TPU_DRAFT_DCP_PAIR_MERGE_BLOCK:
        return {}
    keys = {_block_key(m) for m in layers}
    keys.discard(None)
    import faulthandler

    device = meta.device
    gen = torch.Generator().manual_seed(11 + int(tp.rank_in_group))
    built, ok = {}, True
    # A wrong peer or semaphore hangs the exchange instead of failing.
    faulthandler.dump_traceback_later(SELFTEST_TIMEOUT_S, exit=True)
    try:
        for key in sorted(keys, key=repr):
            try:
                built[key], passed = _selftest_block_op(
                    key, mesh, perm, pair_op, meta, shape, gen, device
                )
            except Exception as e:  # noqa: BLE001 - any failure keeps it off
                logger.warning("DCP draft pair merge + block self-test raised: %r", e)
                passed = False
            ok &= passed
    finally:
        faulthandler.cancel_dump_traceback_later()
    seen = [None] * tp.world_size
    torch.distributed.all_gather_object(seen, ok, group=tp.cpu_group)
    if not all(seen):
        logger.warning("DCP draft pair merge + block not armed: self-test %s", seen)
        return {}
    logger.info("DCP draft pair merge folds the block op (%s).", sorted(built))
    return built


def _selftest_block_op(key, mesh, perm, pair_op, meta, shape, gen, device):
    """The folded op of block configuration ``key`` and whether it matches the
    unfused path (`forward_with_dcp`: the attention's token-order outputs,
    the pair merge, the cache round trip of the block's K/V, the block op) on
    a decode block of two requests and two padding rows."""
    max_query_len, sm_scale, causal, k_scale, v_scale, cache_dtype = key
    block_op = _pair_merge_block_op(mesh, perm, *key)
    rows, (two_h, d) = 2 * max_query_len + 2, shape
    heads, valid = two_h // 2, 2 * max_query_len

    def t(*dims, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(dims, generator=gen) * scale).to(dtype).to(device)

    out = t(rows, two_h, d)
    lse = t(2, 1, max_query_len * two_h, dtype=torch.float32, scale=3)
    q, key_, value = t(rows, heads, d), t(rows, 1, d), t(rows, 1, d)
    seq_lens = torch.zeros(2, dtype=torch.int32, device=device)
    starts = torch.arange(3, dtype=torch.int32, device=device) * max_query_len
    got = block_op(out, lse, meta, q, key_, value, seq_lens)
    out_t = out.clone()
    out_t[valid:] = 0
    lse_t = torch.full((rows, two_h), float("-inf"), device=device)
    lse_t[:valid] = lse.reshape(valid, two_h)
    ctx, ctx_lse = pair_op(out_t, lse_t, meta)
    if k_scale is not None:
        quantized = getattr(torch, cache_dtype)
        key_ = ((key_.float() / k_scale).to(quantized).float() * k_scale).to(q.dtype)
        value = ((value.float() / v_scale).to(quantized).float() * v_scale).to(
            q.dtype
        )
    want = _block_op(max_query_len, sm_scale, causal, True)(
        q, key_, value, seq_lens, starts, ctx, ctx_lse
    )
    passed = torch.allclose(
        got.float().cpu(), want.float().cpu(), rtol=2**-6, atol=1e-2
    )
    return block_op, bool(passed)


def _pair_perm(pairs, tp_ranks) -> tuple[tuple[int, int], ...]:
    """The collective-permute pairs (mesh indices, both directions) of the
    two-rank head groups ``pairs`` (global ranks) over the TP group whose
    global ranks are ``tp_ranks`` (mesh index = TP rank)."""
    index = {int(g): i for i, g in enumerate(tp_ranks)}
    return tuple(
        (index[pair[k]], index[pair[1 - k]])
        for pair in sorted(set(pairs))
        for k in range(2)
    )


def arm_pair_merge(model: torch.nn.Module) -> int:
    """Route every DCP-sharded draft attention layer of `model` whose head
    group has two ranks through the pair merge; returns how many. Collective
    over the TP group (every rank must call it)."""
    from vllm.distributed import get_tp_group

    from vllm_torchtpu.distributed import pallas_all_reduce
    from vllm_torchtpu.kernels.collectives import pair_lse_merge

    layers = [
        m
        for m in model.modules()
        if getattr(getattr(m, "impl", None), "sharded_draft_dcp", None) is not None
    ]
    tp = get_tp_group()
    groups = {id(m.impl.sharded_draft_dcp["head_group"]) for m in layers}
    head_group = layers[0].impl.sharded_draft_dcp["head_group"] if layers else None
    armed = pallas_all_reduce.armed_for(tp.ranks)
    refusal = None
    if not layers:
        refusal = "no DCP-sharded draft layer"
    elif armed is None or armed.mesh is None or not armed.rank_ordered:
        refusal = "the Pallas all-reduce is not armed in rank order"
    elif len(groups) != 1 or head_group is None or head_group.world_size != 2:
        refusal = "the draft head group does not have two ranks"
    seen = [None] * tp.world_size
    mine = None if refusal else tuple(int(r) for r in head_group.ranks)
    torch.distributed.all_gather_object(seen, (refusal, mine), group=tp.cpu_group)
    refusals = {r for r, _ in seen if r}
    if refusals:
        logger.info("DCP draft pair merge not armed: %s", sorted(refusals))
        return 0
    perm = _pair_perm([g for _, g in seen], tp.ranks)
    index = {int(g): i for i, g in enumerate(tp.ranks)}
    me = int(tp.rank_in_group)
    peer = index[mine[1 - int(head_group.rank_in_group)]]
    device = layers[0].draft_dcp_rank.device
    meta = torch.tensor([[peer, int(head_group.rank_in_group)]], dtype=torch.int32)
    meta = meta.to(device)
    op = _pair_merge_op(armed.mesh, perm)
    impl = layers[0].impl
    shape = (impl.num_heads * 2, impl.head_size)
    # A decode block on the kernel, and one past its cap on the permute.
    over = pair_lse_merge.MAX_BYTES // (shape[0] * shape[1] * 2) + 8
    selftest = _pair_selftest(op, meta, head_group, shape, (24, over), device)
    ok = [None] * tp.world_size
    torch.distributed.all_gather_object(ok, selftest, group=tp.cpu_group)
    if not all(ok):
        logger.warning("DCP draft pair merge not armed: self-test failed %s", ok)
        return 0
    block_ops = _arm_pair_merge_block(layers, armed.mesh, perm, op, meta, shape, tp)
    for layer in layers:
        ops = dict(layer.impl.sharded_draft_dcp["ops"], pair_merge=op)
        key = _block_key(layer)
        if key in block_ops:
            ops["pair_merge_block"] = block_ops[key]
        layer.impl.sharded_draft_dcp["ops"] = ops
        layer.register_buffer("draft_pair_meta", meta, persistent=False)
    logger.info(
        "DCP draft pair merge armed on %d layers (rank %d, peer %d).",
        len(layers),
        me,
        peer,
    )
    return len(layers)
