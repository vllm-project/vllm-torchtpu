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

A drafter's KV cache (e.g. the DSpark draft of MiniMax-M3) is sharded like
the target's under decode context parallelism: global block ``b`` of 128
tokens lives on DCP rank ``b % dcp`` in page ``block_tables[seq, b // dcp]``
(vLLM's logical block ``j`` covers ``dcp`` blocks). A DCP group wider than
a KV head's tensor-parallel replicas stores all ``G`` KV heads of the group
per row (``kv_heads``), so every rank serves every query head of the group.

The draft attends a short query block (its draft tokens) over a window of
cached context, so this runs without a Pallas kernel (the batched RPA DCP
kernels reject sliding windows):

1. `write_kv`: each rank writes the K/V rows of the tokens whose block it
   owns, the other tokens are dropped.
2. `attend_cached`: each rank attends every group query head over the
   window's blocks it holds and returns the partial output and its
   log-sum-exp; the caller merges the ranks' partials
   (`dcp_comm.cp_lse_ag_out_rs`) back to its own heads.
3. `attend_block`: the query block over its own new K/V (causal or not, as
   the layer; the block is shorter than the window), on every rank with its
   own heads; merged with step 2 by LSE.

Together they equal the RPA kernel's sliding-window attention over the full
context (``q - k < window``, plus ``q >= k`` if causal; DSpark's query block
is non-causal), with the block written to the cache.

Cache rows use the RPA page layout ``[pages, S, 2G' / p, p, D]`` (``G'``
the KV heads padded to the packing), K and V of head ``h`` interleaved.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NEG_INF = float("-inf")


def _token_layout(seq_lens: jax.Array, cu_q_lens: jax.Array, num_tokens: int):
    """Per token: (sequence, global position, is-real)."""
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.searchsorted(cu_q_lens[1:], tok, side="right", method="compare_all")
    seq = jnp.minimum(seq.astype(jnp.int32), seq_lens.shape[0] - 1)
    q_len = cu_q_lens[seq + 1] - cu_q_lens[seq]
    pos = seq_lens[seq] - q_len + (tok - cu_q_lens[seq])
    valid = tok < cu_q_lens[-1]
    return seq, jnp.where(valid, pos, 0), valid


def merge_rows(key: jax.Array, value: jax.Array, page_shape) -> jax.Array:
    """``[T, G, D]`` K and V -> cache rows ``[T, 2G' / p, p, D]`` (K_h, V_h
    interleaved, zero-padded to the packing)."""
    num_tokens, kv_heads, head_dim = key.shape
    words, packing, width = page_shape
    kv = jnp.concatenate([key, value], axis=-1).reshape(num_tokens, 2 * kv_heads, head_dim)
    pad_heads = words * packing - 2 * kv_heads
    kv = jnp.pad(kv, ((0, 0), (0, pad_heads), (0, width - head_dim)))
    return kv.reshape(num_tokens, words, packing, width)


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
    rank: jax.Array,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> jax.Array:
    """Write this step's K/V (``[T, G, D]``, every head of the group) at the
    tokens' positions, keeping only the tokens of blocks this rank owns."""
    num_tokens = key.shape[0]
    num_pages, page_size = kv_cache.shape[:2]
    max_seqs = seq_lens.shape[0]
    tables = block_tables.reshape(max_seqs, -1)
    seq, pos, valid = _token_layout(seq_lens, cu_q_lens, num_tokens)
    block = pos // page_size
    mine = valid & (block % dcp == rank)
    page = jnp.where(mine, tables[seq, block // dcp], num_pages)
    rows = merge_rows(
        _quantize(key, kv_cache.dtype, k_scale),
        _quantize(value, kv_cache.dtype, v_scale),
        kv_cache.shape[2:],
    )
    from vllm_torchtpu.models.minimax_m3.kernels.msa_write import max_write_runs

    # A rank owns whole pages, so a request's tokens break into at most
    # q // page_size + 2 runs, the bound `max_write_runs` takes.
    write = write_runs if num_tokens >= _RUN_WRITE_MIN_TOKENS else write_runs_batched
    return write(
        kv_cache,
        rows,
        page,
        pos % page_size,
        max_runs=max_write_runs(num_tokens, max_seqs, page_size),
    )


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
    pieces = [1 << b for b in reversed(range(page_size.bit_length()))]
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
    pieces = [1 << b for b in reversed(range(page_size.bit_length()))]
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


def gather_pages(kv_cache: jax.Array, pages: jax.Array, *, interpret: bool = False) -> jax.Array:
    """``kv_cache[pages]`` for ``pages [R, L]`` by one page DMA per grid step.

    An XLA gather of whole pages from the paged cache converts the cache's
    layout on SparseCore on every call (~1 ms per draft layer at c16 DCP2,
    scaling with the cache), so copy the pages with a Pallas kernel instead.
    """
    num_rows, num_cols = pages.shape
    page_shape = kv_cache.shape[1:]
    zeros = (0,) * len(page_shape)

    def kernel(pages_ref, page_ref, o_ref):
        del pages_ref
        o_ref[0] = page_ref[...]

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_rows, num_cols),
            in_specs=[
                pl.BlockSpec(
                    (1, *page_shape),
                    lambda r, c, pages_ref: (pages_ref[r * num_cols + c], *zeros),
                )
            ],
            out_specs=pl.BlockSpec(
                (1, 1, *page_shape), lambda r, c, pages_ref: (r, c, *zeros)
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((num_rows, num_cols, *page_shape), kv_cache.dtype),
        interpret=interpret,
        name="dcp_draft_gather_pages",
    )(pages.reshape(-1).astype(jnp.int32), kv_cache)


def _read_heads(pages: jax.Array, kv_heads: int, head_dim: int):
    """Gathered pages ``[..., S, 2G' / p, p, D]`` -> K, V ``[..., S, G, D]``."""
    shape = pages.shape
    kv = pages.reshape(*shape[:-3], -1, shape[-1])[..., : 2 * kv_heads, :head_dim]
    kv = kv.reshape(*shape[:-3], kv_heads, 2, head_dim)
    return kv[..., 0, :], kv[..., 1, :]


def _window_pages(
    seq_lens,
    block_tables,
    cu_q_lens,
    *,
    max_query_len,
    sliding_window,
    page_size,
    dcp,
    rank,
):
    """Per request: the cached length, and the global blocks of the window this
    rank holds with their pages (``[R, L]`` each)."""
    max_seqs = seq_lens.shape[0]
    tables = block_tables.reshape(max_seqs, -1)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    cache_lens = seq_lens - q_lens
    # Blocks a request's tokens can see: the window of its first query token
    # through its last cached block; this rank's are every dcp-th.
    window_blocks = -(-(sliding_window + max_query_len) // page_size) + 1
    local_blocks = -(-window_blocks // dcp) + 1
    first = jnp.maximum(cache_lens - sliding_window + 1, 0) // page_size
    first_mine = first + (rank - first) % dcp
    blocks = first_mine[:, None] + dcp * jnp.arange(local_blocks)[None, :]
    pages = tables[
        jnp.arange(max_seqs)[:, None], jnp.minimum(blocks // dcp, tables.shape[1] - 1)
    ]
    return q_lens, cache_lens, blocks, pages


def attend_cached(
    q: jax.Array,
    kv_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    kv_heads: int,
    max_query_len: int,
    sliding_window: int,
    sm_scale: float,
    dcp: int,
    rank: jax.Array,
    k_scale: float | None = None,
    v_scale: float | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Partial attention of ``q [T, H, D]`` (every query head of the group,
    head ``i`` reading KV head ``i * G // H``) over the cached tokens of the
    window held by this rank. Returns ``(out [T, H, D], lse [T, H] f32)``;
    ``lse`` is -inf where the rank holds none of a token's window.

    A request's query tokens are at most `max_query_len` (the draft block).
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
        kv_heads=kv_heads,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        interpret=interpret,
    )
    # Back to token order.
    seq, _, valid = _token_layout(seq_lens, cu_q_lens, num_tokens)
    col = jnp.clip(jnp.arange(num_tokens) - cu_q_lens[seq], 0, max_query_len - 1)
    out_t = jnp.where(valid[:, None, None], out, 0.0)
    lse_t = jnp.where(valid[:, None], lse[seq, col], _NEG_INF)
    return out_t.astype(q.dtype), lse_t


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
    kv_heads: int,
    sliding_window: int,
    sm_scale: float,
    k_scale: float | None = None,
    v_scale: float | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Each request's query tokens (``q [T, H, D]`` rows ``q_starts[r]`` on,
    ``q_lens[r]`` of at most ``max_query_len``) over its window pages
    ``pages [R, L]`` (global blocks ``blocks [R, L]``), read straight from the
    paged cache: one grid step per request DMAs its L pages and query rows into
    VMEM (the next request's while this one computes), attends every KV
    head's query group to them and DMAs the output rows back. Returns
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
    group = num_heads // kv_heads
    rows = q_block * group
    keys = num_cols * page_size
    bits = jax.dtypes.itemsize_bits(kv_cache.dtype)
    # 8- and 16-bit caches hold bf16-exact values: QK^T takes them as bf16
    # with the scales applied to the f32 products; PV stays f32 (P is not
    # bf16-exact, and the kernel is VPU-bound, not MXU-bound).
    mxu_dtype = jnp.bfloat16 if bits <= 16 else jnp.float32
    k_mult = sm_scale * (1.0 if k_scale is None else k_scale)
    v_mult = 1.0 if v_scale is None else v_scale

    def unpack(raw, slot):
        if bits == 32:
            return pltpu.bitcast(raw, kv_cache.dtype)
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
            pages = [
                (
                    None,
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

        cache_len = cache_ref[r]
        col = jax.lax.broadcasted_iota(jnp.int32, (1, keys), 1)
        row_k = jax.lax.broadcasted_iota(jnp.int32, (keys, 1), 0)
        k_pos = jnp.zeros((1, keys), jnp.int32)
        k_pos_col = jnp.zeros((keys, 1), jnp.int32)
        for c in range(num_cols):
            base = blocks_ref[r * num_cols + c] * page_size - c * page_size
            k_pos = jnp.where(col // page_size == c, col + base, k_pos)
            k_pos_col = jnp.where(row_k // page_size == c, row_k + base, k_pos_col)
        q_tok = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0) // group
        visible = (
            (k_pos < cache_len)
            & (cache_len + q_tok - k_pos < sliding_window)
            & (q_tok < qlen_ref[r])
        )
        cached = k_pos_col < cache_len
        # One additive mask per request, shared by every KV head. Masked keys
        # must hold finite values (the cache is zero-initialised and writes
        # clip): an add, unlike a select, would carry a NaN score through.
        bias = jnp.where(visible, 0.0, _NEG_INF)
        # Moves a [rows, 1] column to a [1, rows] row (sublanes to lanes).
        diag = jax.lax.broadcasted_iota(
            jnp.int32, (rows, rows), 0
        ) == jax.lax.broadcasted_iota(jnp.int32, (rows, rows), 1)
        # [L * S * words, width] u32: row (token, word), each word `packing`
        # K/V entries (K and V of head h at 2h, 2h + 1), lowest bits first.
        kv_u32 = buf.at[slot].bitcast(jnp.uint32).reshape(keys * words, width)
        loaded = {}

        def load_word(w):
            if w not in loaded:
                loaded[w] = (
                    kv_u32[pl.ds(w, keys, stride=words)] if words > 1 else kv_u32[...]
                )
            return loaded[w]

        for g in range(kv_heads):
            k_word, k_slot = divmod(2 * g, packing)
            v_word, v_slot = divmod(2 * g + 1, packing)
            k = unpack(load_word(k_word), k_slot)[:, :head_dim].astype(mxu_dtype)
            v = unpack(load_word(v_word), v_slot)[:, :head_dim].astype(jnp.float32)
            v = jnp.where(cached, v, 0.0)
            # [Q, X, D] -> [Q * X, D]: row q * X + x is token q, head g * X + x.
            qv = q_buf[slot, :, g * group : (g + 1) * group, :].reshape(rows, head_dim)
            s = jax.lax.dot_general(
                qv.astype(mxu_dtype),
                k,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            s = s * k_mult + bias
            m = jnp.max(s, axis=-1, keepdims=True)
            m_safe = jnp.where(m > _NEG_INF, m, 0.0)
            p = jnp.exp(s - m_safe)
            denom = jnp.sum(p, axis=-1, keepdims=True)
            safe = jnp.where(denom > 0, denom, 1.0)
            out = jnp.dot(p, v, preferred_element_type=jnp.float32)
            out = out * (v_mult / safe)
            o_buf[slot, :, g * group : (g + 1) * group, :] = out.reshape(
                q_block, group, head_dim
            ).astype(o_buf.dtype)
            lse = jnp.where(denom > 0, m_safe + jnp.log(safe), _NEG_INF)
            lse_ref[0, g] = jnp.sum(jnp.where(diag, lse, 0.0), axis=0, keepdims=True)
        # The rows go out while the next requests compute; a slot's copies are
        # waited for before it is refilled, the last two at the end.
        run(rows_out(r, slot), "start")

        @pl.when(r + 1 == pl.num_programs(0))
        def _():
            @pl.when(r > 0)
            def _():
                run(rows_out(r - 1, 1 - slot), "wait")

            run(rows_out(r, slot), "wait")

    out, lse = pl.pallas_call(
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
                pl.BlockSpec((1, kv_heads, 1, rows), lambda r, *_: (r, 0, 0, 0)),
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
            jax.ShapeDtypeStruct((num_reqs, kv_heads, 1, rows), jnp.float32),
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
    lse = lse.reshape(num_reqs, kv_heads, q_block, group).transpose(0, 2, 1, 3)
    return out, lse.reshape(num_reqs, q_block, num_heads)


def attend_cached_reference(
    q: jax.Array,
    kv_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    kv_heads: int,
    max_query_len: int,
    sliding_window: int,
    sm_scale: float,
    dcp: int,
    rank: jax.Array,
    k_scale: float | None = None,
    v_scale: float | None = None,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """`attend_cached` in XLA over gathered pages (the kernel's reference)."""
    num_tokens, num_heads, head_dim = q.shape
    num_pages, page_size = kv_cache.shape[:2]
    max_seqs = seq_lens.shape[0]
    tables = block_tables.reshape(max_seqs, -1)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    cache_lens = seq_lens - q_lens
    # Blocks a request's tokens can see: the window of its first query token
    # through its last cached block; this rank's are every dcp-th.
    window_blocks = -(-(sliding_window + max_query_len) // page_size) + 1
    local_blocks = -(-window_blocks // dcp) + 1
    first = jnp.maximum(cache_lens - sliding_window + 1, 0) // page_size
    first_mine = first + (rank - first) % dcp
    blocks = first_mine[:, None] + dcp * jnp.arange(local_blocks)[None, :]  # [R, L]
    pages = tables[jnp.arange(max_seqs)[:, None], jnp.minimum(blocks // dcp, tables.shape[1] - 1)]
    k, v = _read_heads(gather_pages(kv_cache, pages, interpret=interpret), kv_heads, head_dim)
    k = k.reshape(max_seqs, -1, kv_heads, head_dim).astype(jnp.float32)
    v = v.reshape(max_seqs, -1, kv_heads, head_dim).astype(jnp.float32)
    if k_scale is not None:
        k = k * k_scale
    if v_scale is not None:
        v = v * v_scale
    k_pos = (blocks[:, :, None] * page_size + jnp.arange(page_size)).reshape(max_seqs, -1)

    # Query tokens padded per request: [R, Q].
    slot = jnp.arange(max_query_len)
    tok = cu_q_lens[:-1, None] + slot[None, :]
    tok_valid = slot[None, :] < q_lens[:, None]
    q_pos = cache_lens[:, None] + slot[None, :]
    qr = q[jnp.minimum(tok, num_tokens - 1)].astype(jnp.float32)  # [R, Q, H, D]
    group = num_heads // kv_heads
    qr = qr.reshape(max_seqs, max_query_len, kv_heads, group, head_dim)
    s = jnp.einsum("rqgxd,rkgd->rqgxk", qr, k) * sm_scale
    visible = (
        (k_pos[:, None, :] < cache_lens[:, None, None])
        & (q_pos[:, :, None] - k_pos[:, None, :] < sliding_window)
        & tok_valid[:, :, None]
    )  # [R, Q, K]
    s = jnp.where(visible[:, :, None, None, :], s, _NEG_INF)
    m = jnp.max(s, axis=-1, keepdims=True)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.exp(s - m_safe)
    l = jnp.sum(p, axis=-1, keepdims=True)
    out = jnp.einsum("rqgxk,rkgd->rqgxd", p, v) / jnp.where(l > 0, l, 1.0)
    lse = jnp.where(l > 0, m_safe + jnp.log(jnp.where(l > 0, l, 1.0)), _NEG_INF)
    out = out.reshape(max_seqs, max_query_len, num_heads, head_dim)
    lse = lse.reshape(max_seqs, max_query_len, num_heads)
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
) -> tuple[jax.Array, jax.Array]:
    """Attention of each request's query tokens (``q [T, H, D]``) over its
    own new K/V (``[T, G, D]``), causal or over the whole block. The block is
    shorter than the window. Returns ``(out, lse)``."""
    num_tokens, num_heads, head_dim = q.shape
    kv_heads = key.shape[1]
    max_seqs = seq_lens.shape[0]
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    slot = jnp.arange(max_query_len)
    tok = jnp.minimum(cu_q_lens[:-1, None] + slot[None, :], num_tokens - 1)
    tok_valid = slot[None, :] < q_lens[:, None]
    group = num_heads // kv_heads
    qr = q[tok].astype(jnp.float32).reshape(max_seqs, max_query_len, kv_heads, group, head_dim)
    k = key[tok].astype(jnp.float32)  # [R, Q, G, D]
    v = value[tok].astype(jnp.float32)
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
    kv_heads: int,
    max_query_len: int,
    sliding_window: int,
    sm_scale: float,
    causal: bool,
    k_scale: float | None,
    v_scale: float | None,
) -> dict[str, Any]:
    """The write, cached-attention and block ops of one draft layer."""
    return dict(
        write=_write_op(dcp, k_scale, v_scale),
        attend=_attend_op(
            kv_heads, max_query_len, sliding_window, sm_scale, dcp, k_scale, v_scale
        ),
        block=_block_op(max_query_len, sm_scale, causal),
    )


@functools.cache
def _write_op(dcp: int, k_scale: float | None, v_scale: float | None):
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
            rank=dcp_rank[0],
            k_scale=k_scale,
            v_scale=v_scale,
        )

    name = f"pallas::dcp_draft_write_d{dcp}_k{k_scale}_v{v_scale}".replace(".", "p")
    op = pallas.jax_op(name, _write, donate_argnums=(0,))
    op.register_fake(lambda kv_cache, *args: torch.empty_like(kv_cache))
    return op


@functools.cache
def _attend_op(
    kv_heads: int,
    max_query_len: int,
    sliding_window: int,
    sm_scale: float,
    dcp: int,
    k_scale: float | None,
    v_scale: float | None,
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
            kv_heads=kv_heads,
            max_query_len=max_query_len,
            sliding_window=sliding_window,
            sm_scale=sm_scale,
            dcp=dcp,
            rank=dcp_rank[0],
            k_scale=k_scale,
            v_scale=v_scale,
        )

    name = (
        f"pallas::dcp_draft_attend_g{kv_heads}_q{max_query_len}_w{sliding_window}"
        f"_s{sm_scale:.8f}_d{dcp}_k{k_scale}_v{v_scale}"
    ).replace(".", "p")
    op = pallas.jax_op(name, _attend)
    op.register_fake(
        lambda q, *args: (
            torch.empty_like(q),
            torch.empty(q.shape[:2], dtype=torch.float32, device=q.device),
        )
    )
    return op


@functools.cache
def _block_op(max_query_len: int, sm_scale: float, causal: bool):
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
        )
        return merge(ctx_out.astype(out.dtype), ctx_lse, out, lse)[0]

    name = f"pallas::dcp_draft_block_q{max_query_len}_s{sm_scale:.8f}_c{int(causal)}".replace(
        ".", "p"
    )
    op = pallas.jax_op(name, _block)
    op.register_fake(lambda q, *args: torch.empty_like(q))
    return op


def write_with_dcp(
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: Any,
    dcp_rank: torch.Tensor,
    dcp: int,
    kv_head_replicas: int,
    ops: dict[str, Any],
) -> None:
    """Write the drafter's context K/V (own KV head per rank) into the
    DCP-sharded cache: each rank keeps the tokens of its blocks."""
    from vllm_torchtpu.distributed.dcp import get_dcp_group

    group = get_dcp_group()
    if dcp > kv_head_replicas:
        gathered = group.all_gather(torch.cat([key, value], dim=1).contiguous(), dim=1)
        num_tokens, _, head_dim = gathered.shape
        per_rank = gathered.view(num_tokens, dcp, 2, head_dim)[:, ::kv_head_replicas]
        key, value = per_rank[:, :, 0].contiguous(), per_rank[:, :, 1].contiguous()
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
    k_scale: float | None,
    v_scale: float | None,
    ops: dict[str, Any],
) -> torch.Tensor:
    """The drafter's attention of its query block (``query [T, H, D]``, own
    heads; ``key``/``value [T, 1, D]``, own KV head) over the window of a
    DCP-sharded cache plus the block itself. The block is not written."""
    from vllm_torchtpu.distributed.dcp import get_dcp_group
    from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs

    del dcp, kv_head_replicas  # shared layer args (`write_with_dcp` uses them)
    group = get_dcp_group()
    # Every query head of the group; the window's K/V come from the cache and
    # the block's own K/V stay on this rank, so only the queries cross.
    q_all = group.all_gather(query.contiguous(), dim=1)
    ctx_out, ctx_lse = ops["attend"](
        q_all,
        kv_cache,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        dcp_rank,
    )
    ctx_out, ctx_lse = cp_lse_ag_out_rs(ctx_out, ctx_lse, group, return_lse=True)
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
