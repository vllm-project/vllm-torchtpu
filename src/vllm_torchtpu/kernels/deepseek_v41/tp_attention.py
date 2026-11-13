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
"""Token-parallel attention core for DeepSeek-V4.1 under TP (VLLM_DSV41_TP_ATTN).

Head-parallel TP attention makes every rank read the same SWA window pages and
CSA top-k rows for all T tokens, for its H/tp heads. Here every rank attends
with all H heads for its contiguous token slice [rank * T/tp, (rank+1) * T/tp)
instead, so those cache reads are split across ranks, and the attention output
reduce-scatter disappears (the caller projects with replicated wq_b/wo_a/wo_b).

A request's slice on rank r reads earlier tokens of the same chunk that belong
to ranks < r from the cache, so every rank first writes all T new SWA rows to
its cache replica (`swa_store_all_xla`). The attention kernel's own write of the
local rows then rewrites identical bytes. CED decoder layers are not supported
(their q_len > 1 window start depends on the whole chunk).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import (
    prepare_kv_inputs,
)


def rebase_token_slice(
    cu_q_lens: jax.Array,  # i32[S + 1]
    kv_lens: jax.Array | None,  # i32[S]
    page_indices: jax.Array | None,  # i32[S * P] (flat) or i32[S, P]
    distribution: jax.Array,  # i32[3]
    token_start: jax.Array | int,
    *,
    local_tokens: int,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None, jax.Array]:
    """Per-request metadata for the token slice [token_start, +local_tokens).

    Requests overlapping the slice are packed to local indices 0..n_local-1
    (`_run_swa` trims to the first align(num_tokens, 8) requests), cu_q_lens is
    clipped to the slice, kv_lens drops a request's tokens past the slice end
    (so every local token keeps its absolute position), page-table rows move
    with their request, and distribution shifts by the requests that end before
    the slice. Static shapes are unchanged.
    """
    num_seq_slots = cu_q_lens.shape[0] - 1
    ts = jnp.asarray(token_start, jnp.int32)
    te = ts + local_tokens
    num_seqs = distribution[2]

    seq = jnp.arange(num_seq_slots, dtype=jnp.int32)
    real = seq < num_seqs
    starts, ends = cu_q_lens[:-1], cu_q_lens[1:]
    s0 = jnp.sum((real & (ends <= ts)).astype(jnp.int32))
    n_local = jnp.sum((real & (starts < te) & (ends > ts)).astype(jnp.int32))

    src = jnp.minimum(s0 + seq, num_seq_slots - 1)
    lvalid = seq < n_local
    g_end = ends[src]
    total_local = jnp.clip(cu_q_lens[num_seqs], ts, te) - ts
    cu_local = jnp.concatenate(
        [
            jnp.zeros((1,), jnp.int32),
            jnp.where(lvalid, jnp.clip(g_end, ts, te) - ts, total_local),
        ]
    ).astype(cu_q_lens.dtype)

    kv_local = None
    if kv_lens is not None:
        dropped = g_end - jnp.minimum(g_end, te)
        raw_kv = jnp.where(lvalid, kv_lens[src] - dropped, 0).astype(kv_lens.dtype)
        full_q_len = g_end - starts[src]
        prior_in_step = jnp.where(
            lvalid & (full_q_len > 1), jnp.maximum(ts - starts[src], 0), 0
        )
        eff_span = jnp.where(
            prior_in_step > 0, jnp.minimum(g_end, te) - starts[src], 0
        ).astype(kv_lens.dtype)
        kv_local = raw_kv | (eff_span << 20)

    table_local = None
    if page_indices is not None:
        table = page_indices.reshape(num_seq_slots, -1)
        table_local = jnp.where(lvalid[:, None], table[src], 0)
        table_local = table_local.astype(page_indices.dtype).reshape(page_indices.shape)

    dist_local = jnp.stack(
        [
            jnp.clip(distribution[0] - s0, 0, n_local),
            jnp.clip(distribution[1] - s0, 0, n_local),
            n_local,
        ]
    ).astype(distribution.dtype)
    return cu_local, kv_local, table_local, dist_local


def _swa_token_rows_for_indices(
    t: jax.Array,  # i32[K]
    kv_lens: jax.Array,  # i32[S]
    table: jax.Array,  # i32[S, P], raw (un-ringed) or pre-ringed page ids
    cu_q_lens: jax.Array,  # i32[S + 1]
    n_valid_tokens: jax.Array,  # i32[]
    *,
    num_cache_pages: int,
    phys_tokens_per_page: int,
    logical_page_size: int,
    table_ring_mapped: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    num_seq_slots = kv_lens.shape[0]
    req = jnp.searchsorted(cu_q_lens[1:], t, side="right", method="compare_all")
    req = jnp.minimum(req, num_seq_slots - 1).astype(jnp.int32)
    pos = kv_lens[req] - (cu_q_lens[req + 1] - t)
    col = jnp.clip(pos // logical_page_size, 0, table.shape[1] - 1)
    page = table[req, col]
    if num_cache_pages > 1 and not table_ring_mapped:
        page = jnp.where(page > 0, ((page - 1) % (num_cache_pages - 1)) + 1, 0)
    row = page * phys_tokens_per_page + pos % logical_page_size
    valid = t < n_valid_tokens
    row = jnp.where(valid, row, num_cache_pages * phys_tokens_per_page)
    return req, pos, row


def swa_token_rows(
    kv_lens: jax.Array,  # i32[S]
    page_indices: jax.Array,  # i32[S * P] or i32[S, P], raw (un-ringed) page ids
    cu_q_lens: jax.Array,  # i32[S + 1]
    distribution: jax.Array,  # i32[3]
    *,
    num_tokens: int,
    num_cache_pages: int,
    phys_tokens_per_page: int,
    logical_page_size: int,
) -> jax.Array:
    """Cache token row of every new token, as the SWA kernel computes it.

    Row = ring(page_indices[req, pos // P]) * phys_tokens_per_page + pos % P,
    with ring(p) = ((p - 1) % (N - 1)) + 1 for p > 0 (`_prepare_swa_metadata`).
    Padding tokens get the out-of-range row N * phys_tokens_per_page.
    """
    num_seq_slots = kv_lens.shape[0]
    table = page_indices.reshape(num_seq_slots, -1)
    t = jnp.arange(num_tokens, dtype=jnp.int32)
    _, _, row = _swa_token_rows_for_indices(
        t,
        kv_lens,
        table,
        cu_q_lens,
        cu_q_lens[distribution[2]],
        num_cache_pages=num_cache_pages,
        phys_tokens_per_page=phys_tokens_per_page,
        logical_page_size=logical_page_size,
    )
    return row


def _pack_kv_words(
    new_kv: jax.Array,
    sw_cache_dtype: jnp.dtype,
    slots_per_token: int,
    kv_packing: int,
    lanes: int,
) -> jax.Array:
    T = new_kv.shape[0]
    if csa_cache_layout.is_word_array(sw_cache_dtype) and kv_packing == 4:
        u16 = lax.bitcast_convert_type(
            new_kv.reshape(T * slots_per_token, 2 * lanes), jnp.uint16
        )
        lo = u16[:, :lanes].astype(jnp.uint32)
        hi = u16[:, lanes:].astype(jnp.uint32)
        return lax.bitcast_convert_type(lo | (hi << 16), sw_cache_dtype).reshape(
            T, slots_per_token, lanes
        )
    tok = prepare_kv_inputs(new_kv).reshape(T, slots_per_token, kv_packing, lanes)
    if csa_cache_layout.is_word_array(sw_cache_dtype):
        # Byte b of word w is slab sub-row b (`csa_cache_layout.slabs_to_words`).
        tok = lax.bitcast_convert_type(jnp.moveaxis(tok, 2, -1), sw_cache_dtype)
    return tok


def swa_store_all_xla(
    sw_cache: jax.Array,  # int32 [N, rows, 128] or uint8 [N, rows, packing, 128]
    new_kv: jax.Array,  # bf16 [T, head_dim]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    logical_page_size: int,
) -> jax.Array:
    """Writes every new token's SWA row with one XLA row scatter."""
    num_pages, rows, kv_packing, lanes = csa_cache_layout.u8_slab_shape(
        sw_cache.shape, sw_cache.dtype
    )
    slots_per_token = new_kv.shape[1] * 2 // (kv_packing * lanes)
    phys_tokens_per_page = rows // slots_per_token
    dst = swa_token_rows(
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        num_tokens=new_kv.shape[0],
        num_cache_pages=num_pages,
        phys_tokens_per_page=phys_tokens_per_page,
        logical_page_size=logical_page_size,
    )
    tok = _pack_kv_words(new_kv, sw_cache.dtype, slots_per_token, kv_packing, lanes)
    flat = sw_cache.reshape(num_pages * phys_tokens_per_page, *tok.shape[1:])
    return flat.at[dst].set(tok, mode="drop").reshape(sw_cache.shape)


def _swa_store_pallas_coalesced_kernel(
    dst_ref,        # SMEM: i32[T] (packed: (run_len << 24) | row_start for has_rem != 0)
    blk_meta_ref,   # SMEM: i32[T // blk_sz, 5] -> (len0, row0, len1, row1, has_rem)
    sw_cache_ref,   # HBM: [total_rows, ...]
    tok_ref,        # HBM: [T, ...]
    out_cache_ref,  # HBM aliased to sw_cache_ref
    sem,
):
    del sw_cache_ref
    T = tok_ref.shape[0]
    total_rows = out_cache_ref.shape[0]
    blk_sz = 64 if T % 64 == 0 else (16 if T % 16 == 0 else 1)
    num_blks = T // blk_sz

    def blk_body(b, total_rows_copied):
        len0 = blk_meta_ref[b, 0]
        row0 = blk_meta_ref[b, 1]
        len1 = blk_meta_ref[b, 2]
        row1 = blk_meta_ref[b, 3]
        has_rem = blk_meta_ref[b, 4] != 0
        t0 = b * blk_sz

        @pl.when(len0 > 0)
        def _cp0():
            pltpu.make_async_copy(
                src_ref=tok_ref.at[pl.ds(t0, len0)],
                dst_ref=out_cache_ref.at[pl.ds(row0, len0)],
                sem=sem.at[0],
            ).start()

        @pl.when(len1 > 0)
        def _cp1():
            pltpu.make_async_copy(
                src_ref=tok_ref.at[pl.ds(t0 + len0, len1)],
                dst_ref=out_cache_ref.at[pl.ds(row1, len1)],
                sem=sem.at[0],
            ).start()

        copied = total_rows_copied + len0 + len1

        def do_rem(c):
            def run_cond(state):
                i, _ = state
                return i < blk_sz

            def run_body(state):
                i, cnt = state
                t = t0 + i
                packed = dst_ref[t]
                row_start = packed & 0xFFFFFF
                run_len = packed >> 24
                valid = row_start < total_rows

                def copy_run():
                    pltpu.make_async_copy(
                        src_ref=tok_ref.at[pl.ds(t, run_len)],
                        dst_ref=out_cache_ref.at[pl.ds(row_start, run_len)],
                        sem=sem.at[0],
                    ).start()
                    return (i + run_len, cnt + run_len)

                return lax.cond(valid, copy_run, lambda: (i + run_len, cnt))

            _, final_cnt = lax.while_loop(run_cond, run_body, (jnp.int32(0), c))
            return final_cnt

        return lax.cond(has_rem, do_rem, lambda c: c, copied)

    n_copied = lax.fori_loop(0, num_blks, blk_body, jnp.int32(0))

    @pl.when(n_copied > 0)
    def _wait_all():
        pltpu.make_async_copy(
            src_ref=out_cache_ref.at[pl.ds(0, n_copied)],
            dst_ref=out_cache_ref.at[pl.ds(0, n_copied)],
            sem=sem.at[0],
        ).wait()


def _pack_blk_run_len_fast(rows_batch: jax.Array, total_rows: int) -> jax.Array:
    """Packs (run_len << 24) | row for each [B_BATCH, blk_sz] row using log2 shift-min."""
    b_batch, blk_sz = rows_batch.shape
    if blk_sz == 1:
        return (jnp.int32(1) << 24) | (rows_batch & 0xFFFFFF)
    r0 = rows_batch[:, :-1]
    r1 = rows_batch[:, 1:]
    v0 = r0 < total_rows
    v1 = r1 < total_rows
    same_run = (v0 & v1 & (r1 == r0 + 1)) | (~v0 & ~v1)
    is_break = jnp.concatenate(
        [~same_run, jnp.ones((b_batch, 1), dtype=jnp.bool_)], axis=1
    )
    idx = jnp.arange(blk_sz, dtype=jnp.int32)[None, :]
    run_end = jnp.where(is_break, idx + 1, blk_sz)
    d = 1
    while d < blk_sz:
        run_end = jnp.minimum(
            run_end,
            jnp.pad(run_end[:, d:], ((0, 0), (0, d)), constant_values=blk_sz),
        )
        d *= 2
    run_len = run_end - idx
    return (run_len << 24) | (rows_batch & 0xFFFFFF)


def _compute_swa_store_coalesced_metadata(
    kv_lens: jax.Array,
    table: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    num_tokens: int,
    num_cache_pages: int,
    phys_tokens_per_page: int,
    logical_page_size: int,
    table_ring_mapped: bool = False,
) -> tuple[jax.Array, jax.Array]:
    total_rows = num_cache_pages * phys_tokens_per_page
    blk_sz = 64 if num_tokens % 64 == 0 else (16 if num_tokens % 16 == 0 else 1)
    num_blks = num_tokens // blk_sz
    b_idx = jnp.arange(num_blks, dtype=jnp.int32)
    n_valid_tokens = cu_q_lens[distribution[2]]
    n_blk = jnp.clip(n_valid_tokens - b_idx * blk_sz, 0, blk_sz)
    t_start = b_idx * blk_sz
    t_end = t_start + jnp.maximum(n_blk - 1, 0)
    t_ends = jnp.stack([t_start, t_end], axis=1)  # [num_blks, 2]
    t_flat = t_ends.reshape(-1)  # [2 * num_blks]

    req_flat, pos_flat, row_flat = _swa_token_rows_for_indices(
        t_flat,
        kv_lens,
        table,
        cu_q_lens,
        n_valid_tokens,
        num_cache_pages=num_cache_pages,
        phys_tokens_per_page=phys_tokens_per_page,
        logical_page_size=logical_page_size,
        table_ring_mapped=table_ring_mapped,
    )
    req_ends = req_flat.reshape(num_blks, 2)
    pos_ends = pos_flat.reshape(num_blks, 2)
    row_ends = row_flat.reshape(num_blks, 2)

    can_split2 = (
        (n_blk > 0)
        & (req_ends[:, 0] == req_ends[:, 1])
        & (pos_ends[:, 0] >= 0)
        & (n_blk <= logical_page_size)
    )
    pos0 = pos_ends[:, 0]
    rem_in_page = logical_page_size - (pos0 % logical_page_size)
    len0 = jnp.where(can_split2, jnp.minimum(n_blk, rem_in_page), 0)
    row0 = row_ends[:, 0]
    len1 = jnp.where(can_split2, n_blk - len0, 0)
    row1 = row_ends[:, 1] - len1 + 1
    has_rem = (~can_split2 & (n_blk > 0)).astype(jnp.int32)
    blk_meta = jnp.stack([len0, row0, len1, row1, has_rem], axis=1)

    n_rem = jnp.sum(has_rem)
    B_BATCH = min(16, num_blks)
    rem_pos = jnp.cumsum(has_rem) - 1
    pad_len = ((num_blks + B_BATCH - 1) // B_BATCH) * B_BATCH
    rem_blk_ids = (
        jnp.zeros((pad_len,), dtype=jnp.int32)
        .at[jnp.where(has_rem != 0, rem_pos, pad_len)]
        .set(b_idx, mode="drop")
    )

    init_dst = jnp.zeros((num_blks, blk_sz), dtype=jnp.int32)
    iota_blk = jnp.arange(blk_sz, dtype=jnp.int32)[None, :]
    iota_batch = jnp.arange(B_BATCH, dtype=jnp.int32)

    def rem_body(state):
        k, acc = state
        blks = rem_blk_ids[k + iota_batch]
        t_batch = (blks[:, None] * blk_sz + iota_blk).reshape(-1)
        _, _, rows_batch = _swa_token_rows_for_indices(
            t_batch,
            kv_lens,
            table,
            cu_q_lens,
            n_valid_tokens,
            num_cache_pages=num_cache_pages,
            phys_tokens_per_page=phys_tokens_per_page,
            logical_page_size=logical_page_size,
            table_ring_mapped=table_ring_mapped,
        )
        rows_batch = rows_batch.reshape(B_BATCH, blk_sz)
        packed_batch = _pack_blk_run_len_fast(rows_batch, total_rows)
        valid_k = (k + iota_batch) < n_rem
        target_blks = jnp.where(valid_k, blks, num_blks)
        acc = acc.at[target_blks].set(packed_batch, mode="drop")
        return (k + B_BATCH, acc)

    _, dst_2d = lax.while_loop(
        lambda s: s[0] < n_rem,
        rem_body,
        (jnp.int32(0), init_dst),
    )
    dst = dst_2d.reshape(num_tokens)
    return dst, blk_meta


def _compute_tp_swa_halo_metadata(
    kv_lens: jax.Array,
    table: jax.Array,
    cu_q_lens: jax.Array,
    rank: jax.Array,
    *,
    num_tokens: int,
    local_tokens: int,
    sliding_window: int,
    num_cache_pages: int,
    phys_tokens_per_page: int,
    logical_page_size: int,
) -> tuple[jax.Array, jax.Array]:
    total_rows = num_cache_pages * phys_tokens_per_page
    w = sliding_window
    blk_sz = 64 if (2 * w) % 64 == 0 else (16 if (2 * w) % 16 == 0 else 1)
    num_halo_blks = (2 * w) // blk_sz
    iota_w = jnp.arange(w, dtype=jnp.int32)
    r = rank[0]
    t_prev = jnp.maximum(r, 1) * local_tokens - w + iota_w
    t_last = (num_tokens - w) + iota_w
    t_halo = jnp.concatenate([t_prev, t_last], axis=0)
    _, pos_halo, rows_halo = _swa_token_rows_for_indices(
        t_halo,
        kv_lens,
        table,
        cu_q_lens,
        jnp.int32(num_tokens),
        num_cache_pages=num_cache_pages,
        phys_tokens_per_page=phys_tokens_per_page,
        logical_page_size=logical_page_size,
        table_ring_mapped=True,
    )
    rows_2d = rows_halo.reshape(num_halo_blks, blk_sz)
    pos_2d = pos_halo.reshape(num_halo_blks, blk_sz)
    b_idx = jnp.arange(num_halo_blks, dtype=jnp.int32)
    blk_active = (r > 0) | (b_idx >= (w // blk_sz))
    pos0 = pos_2d[:, 0]
    rem_in_page = logical_page_size - (pos0 % logical_page_size)
    can_split2 = blk_active & (pos0 >= 0) & (blk_sz <= logical_page_size)
    len0 = jnp.where(can_split2, jnp.minimum(blk_sz, rem_in_page), 0)
    row0 = rows_2d[:, 0]
    len1 = jnp.where(can_split2, blk_sz - len0, 0)
    row1 = rows_2d[:, -1] - len1 + 1
    has_rem = (~can_split2 & blk_active).astype(jnp.int32)
    blk_meta_halo = jnp.stack([len0, row0, len1, row1, has_rem], axis=1)
    dst_halo = _pack_blk_run_len_fast(rows_2d, total_rows).reshape(2 * w)
    return dst_halo, blk_meta_halo


def prepare_tp_swa_step_metadata(
    new_kv: jax.Array,  # bf16 [T, head_dim] or sharded [T / tp, head_dim]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    rank: jax.Array,
    *,
    num_cache_pages: int,
    phys_tokens_per_page: int,
    logical_page_size: int,
    tp_size: int,
    sliding_window: int = 0,
    kv_is_sharded: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Precomputes stage-invariant SWA coalesced store and token-slice metadata."""
    kv_has_packed_halo = (
        kv_is_sharded
        and tp_size > 1
        and sliding_window > 0
        and new_kv.shape[0] > tp_size * sliding_window
    )
    T = (
        (new_kv.shape[0] - tp_size * sliding_window) * tp_size
        if kv_has_packed_halo
        else new_kv.shape[0]
    )
    local_tokens = T // tp_size
    token_start = rank[0] * local_tokens
    num_seq_slots = kv_lens.shape[0]
    table = page_indices.reshape(num_seq_slots, -1)
    if num_cache_pages > 1:
        table = jnp.where(
            table > 0, ((table - 1) % (num_cache_pages - 1)) + 1, 0
        )

    def _full_meta(_):
        d, bm = _compute_swa_store_coalesced_metadata(
            kv_lens,
            table,
            cu_q_lens,
            distribution,
            num_tokens=T,
            num_cache_pages=num_cache_pages,
            phys_tokens_per_page=phys_tokens_per_page,
            logical_page_size=logical_page_size,
            table_ring_mapped=True,
        )
        if bm.shape[0] < T:
            bm = jnp.pad(bm, ((0, T - bm.shape[0]), (0, 0)))
        return d, bm

    if kv_has_packed_halo and local_tokens >= sliding_window and (2 * sliding_window) <= T:
        single_req = (distribution[2] == 1) & (cu_q_lens[1] == T)

        def _halo_meta(_):
            d_h, bm_h = _compute_tp_swa_halo_metadata(
                kv_lens,
                table,
                cu_q_lens,
                rank,
                num_tokens=T,
                local_tokens=local_tokens,
                sliding_window=sliding_window,
                num_cache_pages=num_cache_pages,
                phys_tokens_per_page=phys_tokens_per_page,
                logical_page_size=logical_page_size,
            )
            d_pad = (
                jnp.pad(d_h, (0, T - d_h.shape[0]))
                if d_h.shape[0] < T
                else d_h
            )
            bm_pad = (
                jnp.pad(bm_h, ((0, T - bm_h.shape[0]), (0, 0)))
                if bm_h.shape[0] < T
                else bm_h
            )
            bm_pad = bm_pad.at[-1, 0].set(jnp.int32(1))
            return d_pad, bm_pad

        dst, blk_meta = lax.cond(single_req, _halo_meta, _full_meta, None)
    else:
        dst, blk_meta = _full_meta(None)

    swa_cu, swa_kv, swa_pages, swa_dist = rebase_token_slice(
        cu_q_lens,
        kv_lens,
        table.reshape(page_indices.shape),
        distribution,
        token_start,
        local_tokens=local_tokens,
    )
    return dst, blk_meta, swa_kv, swa_pages, swa_cu, swa_dist


def swa_store_all_pallas_precomputed(
    sw_cache: jax.Array,
    new_kv: jax.Array,
    dst: jax.Array,
    blk_meta: jax.Array,
) -> jax.Array:
    """Writes every new token's SWA row using precomputed (dst, blk_meta)."""
    num_pages, rows, kv_packing, lanes = csa_cache_layout.u8_slab_shape(
        sw_cache.shape, sw_cache.dtype
    )
    slots_per_token = new_kv.shape[1] * 2 // (kv_packing * lanes)
    phys_tokens_per_page = rows // slots_per_token
    total_rows = num_pages * phys_tokens_per_page
    if total_rows == 0:
        return sw_cache
    T = new_kv.shape[0]
    blk_sz = 64 if T % 64 == 0 else (16 if T % 16 == 0 else 1)
    num_blks = T // blk_sz
    tok = _pack_kv_words(new_kv, sw_cache.dtype, slots_per_token, kv_packing, lanes)
    flat = sw_cache.reshape(total_rows, *tok.shape[1:])
    out_flat = pl.pallas_call(
        _swa_store_pallas_coalesced_kernel,
        out_shape=jax.ShapeDtypeStruct(flat.shape, flat.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            grid=(1,),
            scratch_shapes=[pltpu.SemaphoreType.DMA((1,))],
        ),
        input_output_aliases={2: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="swa_store_all_pallas",
    )(dst, blk_meta[:num_blks], flat, tok)
    return out_flat.reshape(sw_cache.shape)


def swa_store_all_pallas(
    sw_cache: jax.Array,  # int32 [N, rows, 128] or uint8 [N, rows, packing, 128]
    new_kv: jax.Array,  # bf16 [T, head_dim]
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    logical_page_size: int,
) -> jax.Array:
    """Writes every new token's SWA row in-place via page-coalesced Pallas HBM->HBM DMA."""
    num_pages, rows, kv_packing, lanes = csa_cache_layout.u8_slab_shape(
        sw_cache.shape, sw_cache.dtype
    )
    slots_per_token = new_kv.shape[1] * 2 // (kv_packing * lanes)
    phys_tokens_per_page = rows // slots_per_token
    total_rows = num_pages * phys_tokens_per_page
    if total_rows == 0:
        return sw_cache
    T = new_kv.shape[0]
    num_seq_slots = kv_lens.shape[0]
    table = page_indices.reshape(num_seq_slots, -1)
    dst, blk_meta = _compute_swa_store_coalesced_metadata(
        kv_lens,
        table,
        cu_q_lens,
        distribution,
        num_tokens=T,
        num_cache_pages=num_pages,
        phys_tokens_per_page=phys_tokens_per_page,
        logical_page_size=logical_page_size,
    )
    return swa_store_all_pallas_precomputed(sw_cache, new_kv, dst, blk_meta)


def attention_tp_local(
    q: jax.Array,  # [T / tp, H, D]: all heads of this rank's token slice
    new_kv: jax.Array,  # [T, D] full gathered or [local_tokens + tp_size * w, D] packed halo
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,  # [H]
    rank: jax.Array,  # i32[1]
    a0: jax.Array | None = None,
    a1: jax.Array | None = None,
    a2: jax.Array | None = None,
    a3: jax.Array | None = None,
    a4: jax.Array | None = None,
    a5: jax.Array | None = None,
    a6: jax.Array | None = None,
    a7: jax.Array | None = None,
    *,
    base_fn: Callable,
    logical_page_size: int,
    store_fn: Callable = swa_store_all_pallas,
    sliding_window: int = 0,
    tp_size: int = 1,
    ced_decoder: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Stores SWA rows needed across ranks, then runs `base_fn` on this rank's slice."""
    trailing_args = tuple(
        x for x in (a0, a1, a2, a3, a4, a5, a6, a7) if x is not None
    )
    main_cache_rope: jax.Array | None = None
    union_gather_indices: jax.Array | None = None
    cand_bits_tiled: jax.Array | None = None
    n_unique: jax.Array | None = None
    pre_gathered_nope: jax.Array | None = None
    pre_gathered_rope: jax.Array | None = None
    swa_store_dst: jax.Array | None = None
    swa_store_blk_meta: jax.Array | None = None
    if len(trailing_args) == 1:
        (main_cache_rope,) = trailing_args
    elif len(trailing_args) == 2:
        swa_store_dst, swa_store_blk_meta = trailing_args
    elif len(trailing_args) == 3:
        main_cache_rope, swa_store_dst, swa_store_blk_meta = trailing_args
    elif len(trailing_args) == 4:
        (
            main_cache_rope,
            union_gather_indices,
            cand_bits_tiled,
            n_unique,
        ) = trailing_args
    elif len(trailing_args) == 6 and trailing_args[5].shape[-1] == 5:
        (
            main_cache_rope,
            union_gather_indices,
            cand_bits_tiled,
            n_unique,
            swa_store_dst,
            swa_store_blk_meta,
        ) = trailing_args
    elif len(trailing_args) == 6:
        (
            main_cache_rope,
            union_gather_indices,
            cand_bits_tiled,
            n_unique,
            pre_gathered_nope,
            pre_gathered_rope,
        ) = trailing_args
    elif len(trailing_args) == 8:
        (
            main_cache_rope,
            union_gather_indices,
            cand_bits_tiled,
            n_unique,
            pre_gathered_nope,
            pre_gathered_rope,
            swa_store_dst,
            swa_store_blk_meta,
        ) = trailing_args
    elif len(trailing_args) != 0:
        raise ValueError(f"Unexpected trailing_args count: {len(trailing_args)}")

    local_tokens = q.shape[0]
    token_start = rank[0] * local_tokens
    w = sliding_window
    is_packed_halo = (
        tp_size > 1
        and w > 0
        and new_kv.shape[0] == local_tokens + tp_size * w
    )
    is_csa = main_cache_rope is not None
    if is_csa:
        main_cu, _, _, main_dist = rebase_token_slice(
            main_cu_q_lens,
            None,
            None,
            main_distribution,
            token_start,
            local_tokens=local_tokens,
        )
    else:
        main_cu, main_dist = main_cu_q_lens, main_distribution

    def local(x: jax.Array) -> jax.Array:
        if x.shape[0] == local_tokens:
            return x
        return lax.dynamic_slice_in_dim(x, token_start, local_tokens, axis=0)

    my_local_kv = new_kv[:local_tokens] if is_packed_halo else local(new_kv)

    has_union = (
        is_csa
        and union_gather_indices is not None
        and cand_bits_tiled is not None
        and n_unique is not None
    )

    def _call_with_local_pad(
        sw_c: jax.Array,
        swa_cu_in: jax.Array,
        swa_kv_in: jax.Array,
        swa_pages_in: jax.Array,
        swa_dist_in: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        loc_extra = local(extra) if is_csa else swa_kv_in
        if local_tokens % 8 != 0:
            q_per_req = 6 if local_tokens % 6 == 0 else 1
            pad_reqs = (
                (24 - local_tokens) // 6
                if q_per_req == 6
                else ((local_tokens + 7) & ~7) - local_tokens
            )
            pad_t = pad_reqs * q_per_req
            swa_cu_iota = jnp.arange(swa_cu_in.shape[0], dtype=jnp.int32)
            main_cu_iota = jnp.arange(main_cu.shape[0], dtype=jnp.int32)
            seq_iota = jnp.arange(swa_kv_in.shape[0], dtype=jnp.int32)
            swa_cu_p = (
                swa_cu_in
                + jnp.clip(swa_cu_iota - swa_dist_in[2], 0, pad_reqs) * q_per_req
            )
            swa_dist_p = jnp.stack(
                [
                    jnp.where(
                        (swa_dist_in[0] == swa_dist_in[2]) & (q_per_req == 1),
                        swa_dist_in[0] + pad_reqs,
                        swa_dist_in[0],
                    ),
                    jnp.where(
                        swa_dist_in[1] == swa_dist_in[2],
                        swa_dist_in[1] + pad_reqs,
                        swa_dist_in[1],
                    ),
                    swa_dist_in[2] + pad_reqs,
                ]
            ).astype(swa_dist_in.dtype)
            swa_kv_p = jnp.where(
                (seq_iota >= swa_dist_in[2]) & (seq_iota < swa_dist_in[2] + pad_reqs),
                q_per_req,
                swa_kv_in,
            )
            main_cu_p = (
                main_cu
                + jnp.clip(main_cu_iota - main_dist[2], 0, pad_reqs) * q_per_req
            )
            main_dist_p = jnp.stack(
                [
                    jnp.where(
                        (main_dist[0] == main_dist[2]) & (q_per_req == 1),
                        main_dist[0] + pad_reqs,
                        main_dist[0],
                    ),
                    jnp.where(
                        main_dist[1] == main_dist[2],
                        main_dist[1] + pad_reqs,
                        main_dist[1],
                    ),
                    main_dist[2] + pad_reqs,
                ]
            ).astype(main_dist.dtype)
            q_pad = jnp.pad(q, ((0, pad_t), (0, 0), (0, 0)))
            kv_pad = jnp.pad(my_local_kv, ((0, pad_t), (0, 0)))
            extra_pad = (
                jnp.pad(loc_extra, ((0, pad_t), (0, 0)), constant_values=-1)
                if is_csa
                else swa_kv_p
            )
            union_args: tuple = ()
            if has_union:
                u_per_grp = union_gather_indices.shape[0] // n_unique.shape[0]
                union_args = (
                    jnp.pad(
                        union_gather_indices,
                        (0, pad_reqs * u_per_grp),
                        constant_values=-1,
                    ),
                    jnp.pad(cand_bits_tiled, ((0, pad_reqs), (0, 0), (0, 0))),
                    jnp.pad(n_unique, (0, pad_reqs)),
                )
                if pre_gathered_nope is not None and pre_gathered_rope is not None:
                    union_args += (
                        jnp.pad(
                            pre_gathered_nope,
                            ((0, pad_reqs * u_per_grp), (0, 0)),
                        ),
                        jnp.pad(
                            pre_gathered_rope,
                            ((0, pad_reqs * (u_per_grp // 4)), (0, 0)),
                        ),
                    )
            out_pad, sw_out = base_fn(
                q_pad,
                kv_pad,
                sw_c,
                swa_kv_p,
                swa_pages_in,
                swa_cu_p,
                swa_dist_p,
                main_cache_kv,
                main_kv_lens,
                extra_pad,
                main_page_indices,
                main_cu_p,
                main_dist_p,
                attention_sinks,
                *((main_cache_rope,) if is_csa else ()),
                *union_args,
            )
            return out_pad[:local_tokens], sw_out
        return base_fn(
            q,
            my_local_kv,
            sw_c,
            swa_kv_in,
            swa_pages_in,
            swa_cu_in,
            swa_dist_in,
            main_cache_kv,
            main_kv_lens,
            loc_extra,
            main_page_indices,
            main_cu,
            main_dist,
            attention_sinks,
            *((main_cache_rope,) if is_csa else ()),
            *(
                (union_gather_indices, cand_bits_tiled, n_unique)
                + (
                    (pre_gathered_nope, pre_gathered_rope)
                    if pre_gathered_nope is not None and pre_gathered_rope is not None
                    else ()
                )
                if has_union
                else ()
            ),
        )

    del ced_decoder
    if (
        swa_store_dst is not None
        and swa_store_blk_meta is not None
        and new_kv.shape[0] >= 64
    ):
        swa_cu, swa_kv, swa_pages, swa_dist = (
            swa_cu_q_lens,
            swa_kv_lens,
            swa_page_indices,
            swa_distribution,
        )

        if is_packed_halo:
            T = local_tokens * tp_size
            kv_tails = new_kv[local_tokens:]
            num_pages, _, _, _ = csa_cache_layout.u8_slab_shape(
                sw_cache.shape, sw_cache.dtype
            )
            ring_wraps = (num_pages > 1) and (
                (num_pages - 1) * logical_page_size < T
            )
            r_prev = jnp.maximum(rank[0] - 1, 0)
            halo_prev = lax.dynamic_slice_in_dim(
                kv_tails, r_prev * w, w, axis=0
            )
            halo_last = lax.dynamic_slice_in_dim(
                kv_tails, (tp_size - 1) * w, w, axis=0
            )
            if not ring_wraps:
                halo_kv = jnp.concatenate([halo_prev, halo_last], axis=0)
                sw_c = swa_store_all_pallas_precomputed(
                    sw_cache, halo_kv, swa_store_dst[: 2 * w], swa_store_blk_meta
                )
                return _call_with_local_pad(
                    sw_c, swa_cu, swa_kv, swa_pages, swa_dist
                )
            blk_sz_w = (
                64 if (2 * w) % 64 == 0 else (16 if (2 * w) % 16 == 0 else 1)
            )
            w_blks = w // blk_sz_w
            sw_c = swa_store_all_pallas_precomputed(
                sw_cache,
                halo_prev,
                swa_store_dst[:w],
                swa_store_blk_meta[:w_blks],
            )
            out, sw_c = _call_with_local_pad(
                sw_c, swa_cu, swa_kv, swa_pages, swa_dist
            )
            sw_c = swa_store_all_pallas_precomputed(
                sw_c,
                halo_last,
                swa_store_dst[w : 2 * w],
                swa_store_blk_meta[w_blks : 2 * w_blks],
            )
            return out, sw_c

        sw_cache = swa_store_all_pallas_precomputed(
            sw_cache, new_kv, swa_store_dst, swa_store_blk_meta
        )
        return _call_with_local_pad(
            sw_cache, swa_cu, swa_kv, swa_pages, swa_dist
        )

    sw_cache = store_fn(
        sw_cache,
        new_kv,
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        swa_distribution,
        logical_page_size=logical_page_size,
    )
    swa_cu, swa_kv, swa_pages, swa_dist = rebase_token_slice(
        swa_cu_q_lens,
        swa_kv_lens,
        swa_page_indices,
        swa_distribution,
        token_start,
        local_tokens=local_tokens,
    )
    return _call_with_local_pad(
        sw_cache, swa_cu, swa_kv, swa_pages, swa_dist
    )


def attention_tp_local_swa_op(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    rank: jax.Array,
    swa_store_dst: jax.Array,
    swa_store_blk_meta: jax.Array,
    *,
    base_fn: Callable,
    logical_page_size: int,
    sliding_window: int = 0,
    tp_size: int = 1,
    ced_decoder: bool = False,
) -> tuple[jax.Array, jax.Array]:
    return attention_tp_local(
        q,
        new_kv,
        sw_cache,
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        swa_distribution,
        main_cache_kv,
        main_kv_lens,
        extra,
        main_page_indices,
        main_cu_q_lens,
        main_distribution,
        attention_sinks,
        rank,
        swa_store_dst,
        swa_store_blk_meta,
        base_fn=base_fn,
        logical_page_size=logical_page_size,
        sliding_window=sliding_window,
        tp_size=tp_size,
        ced_decoder=ced_decoder,
    )


def attention_tp_local_csa_op(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    rank: jax.Array,
    main_cache_rope: jax.Array,
    swa_store_dst: jax.Array,
    swa_store_blk_meta: jax.Array,
    *,
    base_fn: Callable,
    logical_page_size: int,
    sliding_window: int = 0,
    tp_size: int = 1,
    ced_decoder: bool = False,
) -> tuple[jax.Array, jax.Array]:
    return attention_tp_local(
        q,
        new_kv,
        sw_cache,
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        swa_distribution,
        main_cache_kv,
        main_kv_lens,
        extra,
        main_page_indices,
        main_cu_q_lens,
        main_distribution,
        attention_sinks,
        rank,
        main_cache_rope,
        swa_store_dst,
        swa_store_blk_meta,
        base_fn=base_fn,
        logical_page_size=logical_page_size,
        sliding_window=sliding_window,
        tp_size=tp_size,
        ced_decoder=ced_decoder,
    )


def attention_tp_local_csa_union_op(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    rank: jax.Array,
    main_cache_rope: jax.Array,
    union_gather_indices: jax.Array,
    cand_bits_tiled: jax.Array,
    n_unique: jax.Array,
    swa_store_dst: jax.Array,
    swa_store_blk_meta: jax.Array,
    *,
    base_fn: Callable,
    logical_page_size: int,
    sliding_window: int = 0,
    tp_size: int = 1,
    ced_decoder: bool = False,
) -> tuple[jax.Array, jax.Array]:
    return attention_tp_local(
        q,
        new_kv,
        sw_cache,
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        swa_distribution,
        main_cache_kv,
        main_kv_lens,
        extra,
        main_page_indices,
        main_cu_q_lens,
        main_distribution,
        attention_sinks,
        rank,
        main_cache_rope,
        union_gather_indices,
        cand_bits_tiled,
        n_unique,
        swa_store_dst,
        swa_store_blk_meta,
        base_fn=base_fn,
        logical_page_size=logical_page_size,
        sliding_window=sliding_window,
        tp_size=tp_size,
        ced_decoder=ced_decoder,
    )


def attention_tp_local_csa_union_pregather_op(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    rank: jax.Array,
    main_cache_rope: jax.Array,
    union_gather_indices: jax.Array,
    cand_bits_tiled: jax.Array,
    n_unique: jax.Array,
    pre_gathered_nope: jax.Array,
    pre_gathered_rope: jax.Array,
    swa_store_dst: jax.Array,
    swa_store_blk_meta: jax.Array,
    *,
    base_fn: Callable,
    logical_page_size: int,
    sliding_window: int = 0,
    tp_size: int = 1,
    ced_decoder: bool = False,
) -> tuple[jax.Array, jax.Array]:
    return attention_tp_local(
        q,
        new_kv,
        sw_cache,
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        swa_distribution,
        main_cache_kv,
        main_kv_lens,
        extra,
        main_page_indices,
        main_cu_q_lens,
        main_distribution,
        attention_sinks,
        rank,
        main_cache_rope,
        union_gather_indices,
        cand_bits_tiled,
        n_unique,
        pre_gathered_nope,
        pre_gathered_rope,
        swa_store_dst,
        swa_store_blk_meta,
        base_fn=base_fn,
        logical_page_size=logical_page_size,
        sliding_window=sliding_window,
        tp_size=tp_size,
        ced_decoder=ced_decoder,
    )
