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
    return kv_cache.at[page, pos % page_size].set(rows, mode="drop")


def pack_rows(
    key: jax.Array,
    value: jax.Array,
    page_shape,
    dtype,
    *,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> jax.Array:
    """``[T, G, D]`` K and V -> cache rows ``[T, 2G' / p, p, D]`` in the cache
    dtype, the bytes `write_kv` stores for them."""
    return merge_rows(
        _quantize(key, dtype, k_scale), _quantize(value, dtype, v_scale), page_shape
    )


def max_owned_runs(num_tokens: int, max_seqs: int, page_size: int, dcp: int) -> int:
    """Bound on the runs of `owned_runs`. A request of ``q`` tokens touches at
    most ``(q - 1) // page_size + 2`` blocks, of which at most
    ``q // (dcp * page_size) + 2`` are one rank's; each owned block is one run."""
    return max(1, min(num_tokens, num_tokens // (dcp * page_size) + 2 * max_seqs))


def owned_runs(
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    num_tokens: int,
    num_pages: int,
    page_size: int,
    dcp: int,
    rank: jax.Array,
    max_runs: int,
):
    """The tokens `write_kv` keeps on `rank`, as runs: maximal spans of
    consecutive tokens writing consecutive slots of one page.

    Returns ``(num_runs [1], tok0, page, slot, length)`` (each ``[max_runs]``).
    Pages are resolved as `write_kv`'s ``.at[page, slot].set(mode="drop")``
    resolves them: negative pages wrap once, pages outside the cache drop.
    Built from cumulative sums and binary searches, without a scatter.
    """
    max_seqs = seq_lens.shape[0]
    tables = block_tables.reshape(max_seqs, -1)
    seq, pos, valid = _token_layout(seq_lens, cu_q_lens, num_tokens)
    block = pos // page_size
    mine = valid & (block % dcp == rank)
    page = jnp.where(mine, tables[seq, block // dcp], num_pages)
    page = jnp.where(page < 0, page + num_pages, page)
    keep = (page >= 0) & (page < num_pages)
    slot = pos % page_size
    cont = (
        keep[1:] & keep[:-1] & (page[1:] == page[:-1]) & (slot[1:] == slot[:-1] + 1)
    )
    no = jnp.zeros((1,), jnp.bool_)
    start = keep & ~jnp.concatenate([no, cont])
    end = keep & ~jnp.concatenate([cont, no])
    starts = jnp.cumsum(start.astype(jnp.int32))
    ends = jnp.cumsum(end.astype(jnp.int32))
    nth = jnp.arange(1, max_runs + 1, dtype=jnp.int32)
    tok0 = jnp.searchsorted(starts, nth, side="left").astype(jnp.int32)
    last = jnp.searchsorted(ends, nth, side="left").astype(jnp.int32)
    first = jnp.minimum(tok0, num_tokens - 1)
    num_runs = jnp.minimum(starts[-1], max_runs)
    return (
        num_runs[None].astype(jnp.int32),
        tok0,
        page[first].astype(jnp.int32),
        slot[first].astype(jnp.int32),
        (last - tok0 + 1).astype(jnp.int32),
    )


# Runs whose DMAs are in flight before the oldest is waited on.
_DMA_LAG = 8


def _copy_runs_kernel(
    num_runs_ref, tok0_ref, page_ref, slot_ref, len_ref,  # scalar prefetch
    rows_hbm,  # [T, 2G' / p, p, D]
    cache_hbm,  # aliased with o_hbm
    o_hbm,  # [pages, S, 2G' / p, p, D]
    sems,  # one DMA semaphore per copy size
    *,
    sizes: tuple[int, ...],
):
    del cache_hbm

    def copies(i, method):
        tok, page, slot, n = tok0_ref[i], page_ref[i], slot_ref[i], len_ref[i]
        for c, size in enumerate(sizes):
            # A run of n rows is copied as the power-of-two pieces of n,
            # largest first; this piece starts after the larger ones.
            off = n - n % (2 * size)
            copy = pltpu.make_async_copy(
                rows_hbm.at[pl.ds(tok + off, size)],
                o_hbm.at[page, pl.ds(slot + off, size)],
                sems.at[c],
            )
            pl.when(n % (2 * size) >= size)(getattr(copy, method))

    num_runs = num_runs_ref[0]

    def body(i, carry):
        copies(i, "start")
        pl.when(i >= _DMA_LAG)(lambda: copies(i - _DMA_LAG, "wait"))
        return carry

    jax.lax.fori_loop(0, num_runs, body, 0)

    def drain(i, carry):
        copies(i, "wait")
        return carry

    jax.lax.fori_loop(jnp.maximum(num_runs - _DMA_LAG, 0), num_runs, drain, 0)


def write_owned_rows(
    kv_cache: jax.Array,
    rows: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    dcp: int,
    rank: jax.Array,
    interpret: bool = False,
) -> jax.Array:
    """`write_kv` from packed rows (``[T, 2G' / p, p, D]``, `pack_rows`, one
    per token): copy only this rank's tokens into their pages, one DMA per
    power-of-two piece of each run, over the aliased cache. The bytes are
    copied as they are, so the cache equals `write_kv`'s whenever the kept
    tokens' (page, slot) are distinct, as real metadata's are."""
    num_tokens = rows.shape[0]
    num_pages, page_size = kv_cache.shape[:2]
    max_runs = max_owned_runs(num_tokens, seq_lens.shape[0], page_size, dcp)
    runs = owned_runs(
        seq_lens,
        block_tables,
        cu_q_lens,
        num_tokens=num_tokens,
        num_pages=num_pages,
        page_size=page_size,
        dcp=dcp,
        rank=rank,
        max_runs=max_runs,
    )
    # A run is at most min(page_size, T) rows: its power-of-two pieces.
    sizes = tuple(1 << b for b in reversed(range(min(page_size, num_tokens).bit_length())))
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        functools.partial(_copy_runs_kernel, sizes=sizes),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1,),
            in_specs=[hbm, hbm],
            out_specs=hbm,
            scratch_shapes=[pltpu.SemaphoreType.DMA((len(sizes),))],
        ),
        out_shape=jax.ShapeDtypeStruct(kv_cache.shape, kv_cache.dtype),
        # Operand 6 = the cache (after 5 scalar-prefetch operands and the rows).
        input_output_aliases={6: 0},
        interpret=interpret,
        name="dcp_draft_write_owned_rows",
    )(*runs, rows.astype(kv_cache.dtype), kv_cache)


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
        pack=_pack_op(k_scale, v_scale),
        write_rows=_write_rows_op(dcp),
    )


@functools.cache
def _pack_op(k_scale: float | None, v_scale: float | None):
    from torch_tpu._internal import pallas

    def _pack(kv_cache: jax.Array, key: jax.Array, value: jax.Array) -> jax.Array:
        # The cache only gives the row shape and dtype.
        return pack_rows(
            key, value, kv_cache.shape[2:], kv_cache.dtype, k_scale=k_scale, v_scale=v_scale
        )

    name = f"pallas::dcp_draft_pack_k{k_scale}_v{v_scale}".replace(".", "p")
    op = pallas.jax_op(name, _pack)
    op.register_fake(
        lambda kv_cache, key, *args: torch.empty(
            (key.shape[0], *kv_cache.shape[2:]), dtype=kv_cache.dtype, device=key.device
        )
    )
    return op


@functools.cache
def _write_rows_op(dcp: int):
    from torch_tpu._internal import pallas

    def _write_rows(
        kv_cache: jax.Array,
        rows: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        dcp_rank: jax.Array,
    ) -> jax.Array:
        return write_owned_rows(
            kv_cache,
            rows,
            seq_lens,
            block_tables,
            query_start_loc,
            dcp=dcp,
            rank=dcp_rank[0],
        )

    op = pallas.jax_op(f"pallas::dcp_draft_write_rows_d{dcp}", _write_rows, donate_argnums=(0,))
    op.register_fake(lambda kv_cache, *args: torch.empty_like(kv_cache))
    return op


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


def gather_group(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    group: Any,
    dcp: int,
    kv_head_replicas: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every rank's query heads, and the K/V of the group's KV heads.

    A group no wider than a KV head's replicas shares one KV head, so only
    the queries are gathered; a wider one gathers ``[q | k | v]`` once.
    """
    from vllm_torchtpu.distributed.dcp import split_group_heads

    if dcp <= kv_head_replicas:
        return group.all_gather(query.contiguous(), dim=1), key, value
    gathered = group.all_gather(torch.cat([query, key, value], dim=1).contiguous(), dim=1)
    q_all, (k_all, v_all) = split_group_heads(
        gathered, dcp, query.shape[1], kv_head_replicas, per_kv_head=2
    )
    return q_all, k_all, v_all


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


def gather_rows(rows: torch.Tensor, group: Any) -> torch.Tensor:
    """All-gather packed rows ``[n, ...]`` over the DCP group (rank-major),
    moved as 32-bit words (rows are whole 128-lane words)."""
    flat = rows.contiguous().reshape(rows.shape[0], -1)
    if flat.dtype.itemsize == 1:
        flat = flat.view(torch.uint8)
    words = flat.view(torch.int32)
    gathered = group.all_gather(words, dim=0)
    return gathered.view(flat.dtype).view(rows.dtype).view(-1, *rows.shape[1:])


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

    group = get_dcp_group()
    q_all, k_all, _ = gather_group(query, key, value, group, dcp, kv_head_replicas)
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
