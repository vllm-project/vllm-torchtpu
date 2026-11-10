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
"""The attention half of the MSA DCP forward for a DCP group that spans
several KV heads (`msa_dcp`, ``G > 1``), as one SPMD program over the DCP
mesh axis: the per-rank body of a ``shard_map``.

The DCP attention gathers every query head of the group to every rank and
reduce-scatters the ranks' partial outputs back: ``2 * T * G * Hg * D``
bf16 per rank and layer, whatever the context. A prefill chunk's rows are
many and its context comparatively short (c64, DCP 8: 8K rows over a
149K-token context move 235 MB of queries and partials per rank, the
context's pages of one KV head 19 MB). So a long step attends prefill
requests head-parallel: each rank brings its own KV head's pages of those
requests from the other ranks (an all-to-all of whole pages) and attends
its own query heads over them, which needs no query gather and no merge.
Decode-sized requests (at most ``decode_rows`` query rows) keep the DCP
attention on their rows only, compacted to a static ``max_num_seqs *
decode_rows`` rows.

Both paths attend the same selection (`msa_dcp.merge_head_candidates` of
the ranks' candidates) over the same 4-bit pages; the head-parallel one
normalizes in one pass instead of merging bf16 partials.
"""

import functools

import jax
import jax.numpy as jnp

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa,
    msa_attention,
    msa_attention_lse,
    msa_dcp,
    msa_fp4,
)

# Steps of at least this many tokens attend their prefill requests
# head-parallel. Each round moves a static page buffer (~35 MB per rank at
# max_model_len 256K), which a step's query gather and merge exceed from
# about 2K tokens (~0.16 us per token at c64 DCP 8, j-0a66fa2e).
HEAD_PARALLEL_MIN_TOKENS = 2048


def merge_partials(
    out: jax.Array, lse: jax.Array, axis: str, *, rs_out_precision: bool
) -> jax.Array:
    """`ops.msa.merge_head_partials` inside the SPMD program: head-major
    partials ``out [H, T, D]``, ``lse [H, T]`` -> this rank's heads,
    unrotated, ``[T, H / dcp, D]``."""
    lse = lse.astype(jnp.float32).T  # token-major across the gather
    lses = jax.lax.all_gather(lse, axis, axis=0)
    weight = jnp.exp(lse - jax.nn.logsumexp(lses, axis=0)).T[..., None]
    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    weighted = jnp.where(weight > 0, out.astype(jnp.float32) * weight, 0.0)
    if rs_out_precision:
        weighted = weighted.astype(out.dtype)
    merged = jax.lax.psum_scatter(weighted, axis, scatter_dimension=0, tiled=True)
    return msa_fp4.unrotate_output(merged.astype(out.dtype)).transpose(1, 0, 2)


def _dcp_attend(
    q,
    kv_cache,
    scale_cache,
    candidates,
    seq_lens,
    block_tables,
    cu,
    *,
    axis,
    rank,
    dcp,
    sm_scale,
    topk,
    rs_out_precision,
    interpret,
):
    """The DCP attention of rows ``q [R, Hl, D]`` (this rank's rotated query
    heads) with this rank's packed ``candidates [R, G * 2 * topk]``."""
    q_all = jax.lax.all_gather(q, axis, axis=1, tiled=True)
    cand_all = jax.lax.all_gather(candidates, axis, axis=0)
    out, lse = msa_dcp.msa_attend_dcp(
        q_all,
        kv_cache,
        cand_all,
        seq_lens,
        block_tables,
        cu,
        sm_scale=sm_scale,
        topk=topk,
        rank=rank,
        dcp=dcp,
        scale_cache=scale_cache,
        interpret=interpret,
    )
    return merge_partials(out, lse, axis, rs_out_precision=rs_out_precision)


def _per_token(seq_id: jax.Array, table: jax.Array) -> jax.Array:
    """``table[seq_id]`` for a per-request table, by a one-hot over the
    requests: a gather of 8K tokens from a 64-entry table cost ~70 us."""
    onehot = seq_id[:, None] == jnp.arange(table.shape[0], dtype=jnp.int32)
    return jnp.sum(jnp.where(onehot, table[None, :].astype(jnp.int32), 0), axis=1)


def _exchange_pages(kv_cache, scale_cache, pages, *, dcp, replicas, head_width, axis):
    """Every rank's ``pages [L]`` of the KV head each rank owns:
    ``(kv [L * dcp, S / 8, 4, 2 * D], scales [L * dcp, SCALE_ROWS, 128])``,
    rank ``src``'s page ``pages[j * L / replicas + i]`` at slot ``(j * dcp +
    src) * L / replicas + i``.

    The ``replicas`` ranks of a KV head each receive a ``1 / replicas`` part
    of the pages over the all-to-all and share them with an all-gather
    inside the replica group (one chip's cores at c64)."""
    num_pages = pages.shape[0]
    part = num_pages // replicas
    _, rows, packing, _ = kv_cache.shape
    heads = dcp // replicas
    if kv_cache.shape[3] != heads * head_width:
        raise ValueError(
            f"cache rows {kv_cache.shape} do not hold {heads} KV heads of "
            f"{head_width} columns"
        )
    kv = kv_cache[pages].reshape(replicas, part, rows, packing, heads, head_width)
    # Destination rank d = h * replicas + j gets part j of head h.
    send = jnp.moveaxis(kv, 4, 0).reshape(dcp, part, rows, packing, head_width)
    sc = scale_cache[pages][:, : 2 * heads]  # K/V rows of every head
    sc = sc.reshape(replicas, part, heads, 2, sc.shape[-1])
    send_sc = jnp.moveaxis(sc, 2, 0).reshape(dcp, part, 2, sc.shape[-1])
    recv = jax.lax.all_to_all(send, axis, 0, 0, tiled=True)
    recv_sc = jax.lax.all_to_all(send_sc, axis, 0, 0, tiled=True)
    if replicas > 1:
        groups = [list(range(h * replicas, (h + 1) * replicas)) for h in range(heads)]
        recv = jax.lax.all_gather(recv, axis, axis_index_groups=groups, tiled=True)
        recv_sc = jax.lax.all_gather(
            recv_sc, axis, axis_index_groups=groups, tiled=True
        )
    recv = recv.reshape(num_pages * dcp, rows, packing, head_width)
    recv_sc = recv_sc.reshape(num_pages * dcp, 2, recv_sc.shape[-1])
    recv_sc = jnp.pad(recv_sc, ((0, 0), (0, msa_fp4.SCALE_ROWS - 2), (0, 0)))
    return recv, recv_sc


def _head_parallel_attend(
    q,
    kv_cache,
    scale_cache,
    candidates,
    seq_lens,
    block_tables,
    cu,
    hp_req,
    tokens,
    *,
    axis,
    rank,
    dcp,
    kv_head_replicas,
    sm_scale,
    topk,
    interpret,
):
    """Head-major ``[Hl, T, D]`` attention (rotated basis) of the rows of the
    requests in ``hp_req`` over their pages of this rank's KV head; other
    rows are zero. ``tokens`` is `msa.token_positions` of the step.

    Requests are attended in rounds of at most ``pages_per_seq`` logical
    blocks (one max-length request): each round exchanges one static page
    buffer and attends the round's rows in windows of one kernel call."""
    del rank  # every rank runs the same exchange
    num_tokens, num_q_heads, head_dim = q.shape
    block_size = msa.page_size(kv_cache)
    max_num_seqs = seq_lens.shape[0]
    block_tables_2d = block_tables.reshape(max_num_seqs, -1)
    capacity = block_tables_2d.shape[1]  # logical blocks per round
    replicas = kv_head_replicas
    part = -(-capacity // replicas)  # each replica's share of the buffer
    buffer_pages = part * replicas
    num_heads = candidates.shape[1] // (2 * topk)

    # This rank's KV head's candidates from every rank -> its selection.
    per_rank = candidates.reshape(num_tokens, num_heads, 2 * topk).transpose(1, 0, 2)
    send = jnp.repeat(per_rank, replicas, axis=0)  # rank d gets head d // replicas
    own = jax.lax.all_to_all(send, axis, 0, 0, tiled=True)  # [dcp, T, 2 * topk]
    selected = msa_dcp.merge_head_candidates(own, topk, interpret)[:, 0]  # [T, topk]

    seq_id, pos, valid = tokens
    seq_blocks = -(-seq_lens // (block_size * dcp))
    blocks = jnp.where(hp_req, seq_blocks, 0)
    seq_idx = jnp.arange(max_num_seqs, dtype=jnp.int32)
    page_table = jnp.arange(dcp * buffer_pages, dtype=jnp.int32)
    # Rows per kernel call (the caller pads tokens to a block_q multiple).
    window = min(num_tokens, msa_attention.MAX_TOKENS_PER_CALL)
    sel_block = jnp.maximum(selected, 0)
    own_block = pos // block_size
    limit = pos % block_size + 1

    def slot(base, block):
        # Logical block l = base + block // dcp of rank block % dcp.
        logical = base + block // dcp
        return ((logical // part) * dcp + block % dcp) * part + logical % part

    def attend_round(first, acc, merge):
        csum = jnp.cumsum(jnp.where(seq_idx >= first, blocks, 0))
        # A request never exceeds the buffer (the block table's width); always
        # admitting `first` keeps the rounds moving if that ever broke.
        in_round = (
            hp_req & (seq_idx >= first) & ((csum <= capacity) | (seq_idx == first))
        )
        base = csum - blocks  # the request's first logical block in the buffer
        total = jnp.max(jnp.where(in_round, csum, 0))
        nxt = jnp.min(
            jnp.where((seq_idx >= first) & ~in_round & hp_req, seq_idx, max_num_seqs)
        )
        # Buffer page s: logical block s - base[q] of request q.
        page = jnp.arange(buffer_pages, dtype=jnp.int32)
        q_of = jnp.minimum(jnp.searchsorted(csum, page, side="right"), max_num_seqs - 1)
        j = jnp.clip(page - base[q_of], 0, capacity - 1)
        pages = jnp.where(page < total, block_tables_2d[q_of, j], 0)
        kv, sc = _exchange_pages(
            kv_cache,
            scale_cache,
            pages,
            dcp=dcp,
            replicas=replicas,
            head_width=2 * head_dim,
            axis=axis,
        )

        mine = valid & (_per_token(seq_id, in_round) > 0)
        tok_base = _per_token(seq_id, base)
        keys = jnp.where(
            mine[:, None] & (selected >= 0), slot(tok_base[:, None], sel_block), -1
        )
        local = jnp.where(mine, slot(tok_base, own_block), -1)
        lo = cu[jnp.min(jnp.where(in_round, seq_idx, max_num_seqs - 1))]
        hi = cu[jnp.max(jnp.where(in_round, seq_idx + 1, 0))]

        attend = functools.partial(
            msa_attention_lse.msa_sparse_attention_lse,
            sm_scale=sm_scale,
            block_q=msa_attention.DEFAULT_BLOCK_Q,
            pages_per_step=msa_attention.DEFAULT_PAGES_PER_STEP,
            head=0,
            scale_cache=sc,
            k_scale_row=0,
            v_scale_row=1,
            head_major=True,
            interpret=interpret,
        )
        if not merge:
            # One call over every row: the kernel chunks the rows and writes
            # its output in place; rows outside the round see no page and
            # come back zero (cheap empty tiles).
            out, _ = attend(q, kv, keys, local, limit, page_table)
            return nxt, out

        def window_body(i, acc):
            start = jnp.minimum(lo + i * window, num_tokens - window)

            def sl(x):
                return jax.lax.dynamic_slice_in_dim(x, start, window, 0)

            out, _ = attend(sl(q), kv, sl(keys), sl(local), sl(limit), page_table)
            old = jax.lax.dynamic_slice_in_dim(acc, start, window, 1)
            out = jnp.where(sl(mine)[None, :, None], out, old)  # earlier rounds
            return jax.lax.dynamic_update_slice_in_dim(acc, out, start, 1)

        acc = jax.lax.fori_loop(0, -(-(hi - lo) // window), window_body, acc)
        return nxt, acc

    first = jnp.min(jnp.where(hp_req, seq_idx, max_num_seqs))
    first, acc = attend_round(first, None, merge=False)

    # Further rounds sit under a cond: a while loop that reads the cache
    # makes XLA copy the whole cache ahead of it (0.45 ms per layer at c64).
    def more_rounds(state):
        return jax.lax.while_loop(
            lambda s: s[0] < max_num_seqs,
            lambda s: attend_round(*s, merge=True),
            state,
        )

    _, acc = jax.lax.cond(first < max_num_seqs, more_rounds, lambda s: s, (first, acc))
    return acc


def attend_and_merge(
    rank: jax.Array,
    q: jax.Array,
    kv_cache: jax.Array,
    scale_cache: jax.Array,
    candidates: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    axis: str,
    dcp: int,
    kv_head_replicas: int,
    sm_scale: float,
    topk: int,
    decode_rows: int,
    rs_out_precision: bool,
    head_parallel_min_tokens: int = HEAD_PARALLEL_MIN_TOKENS,
    interpret: bool = False,
) -> jax.Array:
    """This rank's (``rank``, ``int32[1]``) attention output ``[T, Hl, D]``
    (unrotated) for its rotated query heads ``q [T, Hl, D]`` and its packed
    ``candidates [T, G * 2 * topk]`` (`msa_dcp.msa_write_and_candidates_dcp`);
    4-bit caches.
    Runs inside a ``shard_map`` over `axis` (the DCP group)."""
    num_tokens = q.shape[0]
    # The rank is an input: torch_tpu's SPMD partitioner refuses the
    # partition-id that `lax.axis_index` lowers to.
    rank = msa_dcp.rank_array(rank)[0]
    common = dict(
        axis=axis, rank=rank, dcp=dcp, sm_scale=sm_scale, topk=topk, interpret=interpret
    )
    max_num_seqs = seq_lens.shape[0]
    rows = max_num_seqs * decode_rows
    if num_tokens < max(head_parallel_min_tokens, 2 * rows):
        return _dcp_attend(
            q,
            kv_cache,
            scale_cache,
            candidates,
            seq_lens,
            block_tables,
            cu_q_lens,
            rs_out_precision=rs_out_precision,
            **common,
        )
    seq_idx = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    live = (seq_idx < distribution[2]) & (q_lens > 0)
    dcp_req = live & (q_lens <= decode_rows)
    hp_req = live & (q_lens > decode_rows)

    # The decode requests' rows, compacted: their own cu_q_lens.
    cu_d = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(jnp.where(dcp_req, q_lens, 0))]
    ).astype(jnp.int32)
    p = jnp.arange(rows, dtype=jnp.int32)
    req = jnp.minimum(jnp.searchsorted(cu_d[1:], p, side="right"), max_num_seqs - 1)
    src_row = jnp.where(p < cu_d[-1], cu_q_lens[req] + p - cu_d[req], 0)
    dcp_out = _dcp_attend(
        q[src_row],
        kv_cache,
        scale_cache,
        candidates[src_row],
        seq_lens,
        block_tables,
        cu_d,
        rs_out_precision=rs_out_precision,
        **common,
    )
    tokens = msa.token_positions(num_tokens, seq_lens, cu_q_lens)
    hp_out = _head_parallel_attend(
        q,
        kv_cache,
        scale_cache,
        candidates,
        seq_lens,
        block_tables,
        cu_q_lens,
        hp_req,
        tokens,
        kv_head_replicas=kv_head_replicas,
        **common,
    )
    hp_out = msa_fp4.unrotate_output(hp_out.transpose(1, 0, 2))
    # The decode rows go back to their places (the others are dropped).
    dest = jnp.where(p < cu_d[-1], src_row, num_tokens)
    return hp_out.at[dest].set(dcp_out, mode="drop")
