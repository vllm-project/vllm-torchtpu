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
"""Decode-mode RPA schedule built with array ops.

`schedule.generate_rpa_metadata` walks sequence -> query block -> KV block on
the scalar core and stores every task's DMA descriptors one word at a time.
For a decode batch each sequence holds one query block, so task t's sequence
and KV block follow from a prefix sum over the sequences' KV block counts,
and every descriptor is an elementwise function of them. This module computes
the same descriptors as vector ops, laid out exactly as the kernel lays them
out in HBM (task t in step t // batch_size, lane t % batch_size; full chunks
back to back), with the same step totals and step count. Entries past the
step count, which the attention kernel never reads, hold masked values.
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, schedule


def supported(
    cfgs: configs.RpaConfigs,
    computer_cls: type[schedule.BaseMetadataComputer] = schedule.BaseMetadataComputer,
    extra_scalars=(),
) -> bool:
    """Whether `generate_rpa_metadata` reproduces the kernel for `cfgs`: decode
    mode with one query block per sequence, the whole page table in SMEM, no
    sliding window, context parallelism or attention scope."""
    return (
        computer_cls is schedule.BaseMetadataComputer
        and not extra_scalars
        and cfgs.mode == configs.RpaCase.DECODE
        and cfgs.bq_sz >= cfgs.serve.decode_query_size
        and not cfgs.serve.stream_page_indices
        and cfgs.serve.cp_group_size is None
        and cfgs.model.sliding_window is None
        and cfgs.serve.attention_scope == configs.AttentionScope.FULL
    )


def _cdiv(a, b):
    return (a + b - 1) // b


def generate_rpa_metadata(
    cu_q_lens: jax.Array,
    q_offsets: jax.Array,
    kv_cache_lens: jax.Array,
    kv_new_lens: jax.Array,
    distribution: jax.Array,
    cfgs: configs.RpaConfigs,
) -> schedule.RpaSchedule:
    """Same consumed contents as `schedule.generate_rpa_metadata` for `supported`
    configs, given the kernel's own preconditions: the decode range's sequences
    carry at most bq_sz queries and their KV fits their page table row."""
    assert supported(cfgs)
    serve = cfgs.serve
    out = schedule.BaseMetadataComputer.get_rpa_schedule(
        cfgs, multiplier=cfgs.max_schedule_size_multiplier
    )
    batch = cfgs.batch_size
    num_seqs = serve.num_seqs
    bkv, bq, page = cfgs.bkv_sz, cfgs.bq_sz, serve.page_size
    i32 = jnp.int32
    # A sequence spans at most max_k KV blocks of its page table row, so only
    # the first tasks_ub task slots (whole steps) can be written.
    max_k = _cdiv(serve.pages_per_seq * page, bkv)
    tasks_ub = min(out.s_idx.data.shape[0], _cdiv(num_seqs * max_k, batch) * batch)

    # Per sequence: the decode range holds one query block per live sequence
    # (q_len <= bq_sz), attending KV blocks [0, end_k).
    seq = jnp.arange(num_seqs, dtype=i32)
    q_start = cu_q_lens[:-1].astype(i32)
    q_end = cu_q_lens[1:].astype(i32)
    q_len = q_end - q_start
    k_len = kv_cache_lens + kv_new_lens
    num_k = _cdiv(k_len, bkv)
    q_sz = jnp.clip(q_len, 0, bq)
    end_k = jnp.minimum(num_k, (q_offsets + q_sz - 1) // bkv + 1)
    live = (seq < distribution[0]) & (q_len > 0)
    tasks = jnp.where(live, jnp.maximum(end_k, 0), 0)
    ends = jnp.cumsum(tasks)
    starts = ends - tasks
    total = ends[-1]

    # Task t: its sequence s (first one whose tasks end after t) and KV block.
    # Sequences run along the major axis, so each lookup sums across rows.
    t = jnp.arange(tasks_ub, dtype=i32)
    owner = jnp.sum((ends[:, None] <= t[None, :]).astype(i32), axis=0)
    owner = jnp.minimum(owner, num_seqs - 1)
    pick = seq[:, None] == owner[None, :]

    def at(v):
        return jnp.sum(jnp.where(pick, v[:, None].astype(i32), 0), axis=0)

    s = owner
    k = t - at(starts)
    valid = t < total
    t_q_start, t_q_end = at(q_start), at(q_end)
    t_cache, t_new, t_end_k = at(kv_cache_lens), at(kv_new_lens), at(end_k)

    def keep(v):
        return jnp.where(valid.reshape(valid.shape + (1,) * (v.ndim - 1)), v, 0)

    q_sz_task = jnp.clip(t_q_end - t_q_start, 0, bq)
    kv_len_start = k * bkv
    kv_left = t_cache + t_new - kv_len_start
    if serve.skip_kv_update:
        kv_left_frm_cache = kv_left
    else:
        kv_left_frm_cache = jnp.maximum(t_cache - kv_len_start, 0)
    p_offset = s * serve.pages_per_seq + k * cfgs.bkv_p
    seq_lane = serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE

    # Cached-page DMAs, one row per task with the (src, dst, size) words of its
    # pages side by side: [tasks, 3 * pages].
    col = jnp.arange(3 * cfgs.bkv_p_cache, dtype=i32)
    col_page, col_field = col // 3, col % 3
    col_dst = col_page << serve.page_size_log2
    cache_sz = jnp.clip(kv_left_frm_cache[:, None] - col_dst[None, :], 0, page)
    cache_third = (cache_sz > 0).astype(i32) if seq_lane else cache_sz
    cache_src = jnp.minimum(
        p_offset[:, None] + col_page[None, :], serve.num_page_indices - 1
    )
    dma_kv_cache = keep(
        jnp.where(
            col_field == 0, cache_src, jnp.where(col_field == 1, col_dst, cache_third)
        )
    )
    # Tokens the step totals count per task (`schedule._compute_waits`): the
    # cached pages fill in order, so their sizes sum to the clipped length, and
    # cdiv(length, page) of them are non-empty.
    cache_span = cfgs.bkv_p_cache * page
    if seq_lane:
        kv_in = jnp.clip(_cdiv(kv_left_frm_cache, page), 0, cfgs.bkv_p_cache) * page
    else:
        kv_in = jnp.clip(kv_left_frm_cache, 0, cache_span)

    kv_left_frm_new = kv_left - kv_left_frm_cache
    bkv_sz_cache = jnp.minimum(kv_left_frm_cache, bkv)
    new_sz = jnp.minimum(bkv - bkv_sz_cache, kv_left_frm_new)
    q_wb = jnp.maximum(0, kv_len_start - t_cache) // bq
    do_writeback = ((new_sz > 0) & (q_wb == 0)).astype(i32)
    new_src = t_q_end - kv_left_frm_new

    # New-token DMAs: (dst_vmem, dma_sz, slot_start) per entry, as the kernel
    # walks them.
    entries = []
    if cfgs.bkv_p_new < cfgs.bkv_p:
        curr_vmem, curr_rem = bkv_sz_cache, new_sz
        for _ in range(cfgs.bkv_p_new):
            slot_start = (curr_vmem // page) * page
            dma_sz = jnp.minimum(curr_rem, slot_start + page - curr_vmem)
            dma_sz = jnp.where(curr_rem > 0, dma_sz, 0)
            entries.append((curr_vmem, dma_sz, slot_start))
            curr_vmem = curr_vmem + dma_sz
            curr_rem = jnp.maximum(0, curr_rem - dma_sz)
    else:
        for i in range(max(cfgs.bkv_p, cfgs.bkv_p_new)):
            slot_start = jnp.full_like(new_sz, i * page)
            dst = jnp.maximum(slot_start, bkv_sz_cache)
            end_in_slot = jnp.minimum(slot_start + page, bkv_sz_cache + new_sz)
            entries.append((dst, jnp.maximum(0, end_in_slot - dst), slot_start))
    assert len(entries) == cfgs.bkv_p_new
    words = []  # each entry's struct words in offset order, entries side by side
    kv_out = jnp.zeros_like(new_sz)
    if seq_lane:
        cache_pages = _cdiv(bkv_sz_cache, page)
        base = t_q_end - kv_left_frm_new
        new_tok_offset = base % page
        pages_to_fetch = jnp.where(
            new_sz > 0, (new_tok_offset + new_sz - 1) // page + 1, 0
        )
        for i, (_, dma_sz, slot_start) in enumerate(entries):
            fetch_val = (i < pages_to_fetch).astype(i32)
            wb_val = (dma_sz > 0).astype(i32)
            p_idx = jnp.minimum(
                (kv_len_start + slot_start) >> serve.page_size_log2,
                serve.pages_per_seq - 1,
            )
            words += [
                base - new_tok_offset + i * page,  # fetch_hbm
                (cache_pages + i) * page,  # fetch_vmem
                s * serve.pages_per_seq + p_idx,  # wb_hbm
                slot_start,  # wb_vmem
                fetch_val | (wb_val << 1),  # flags
            ]
            kv_in = kv_in + jnp.where(fetch_val > 0, page, 0)
            kv_out = kv_out + jnp.where((do_writeback == 1) & (wb_val > 0), page, 0)
    else:
        for dst, dma_sz, _ in entries:
            p_idx = jnp.minimum(
                (kv_len_start + dst) >> serve.page_size_log2, serve.pages_per_seq - 1
            )
            p_off = (kv_len_start + dst) & serve.page_size_mask
            dst_hbm = (
                (s * serve.pages_per_seq + p_idx) << serve.page_size_log2
            ) | p_off
            # Struct words: wb_hbm, fetch_hbm, fetch_vmem (= wb_vmem), flags.
            words += [dst_hbm, new_src, dst, dma_sz]
            kv_in = kv_in + dma_sz
            kv_out = kv_out + jnp.where(do_writeback == 1, dma_sz, 0)
    assert len(words) == cfgs.dma_kv_new_size * cfgs.bkv_p_new
    dma_kv_new = keep(jnp.stack(words, axis=-1))
    dma_q = keep(jnp.stack([t_q_start, q_sz_task], axis=-1))
    is_last_k = keep((k == t_end_k - 1).astype(i32))
    q_sz_kept = keep(q_sz_task)

    # Step totals over each step's lanes.
    chunk = pltpu.get_tpu_info().num_lanes * 4

    def per_step(v):
        return jnp.sum(keep(v).reshape(tasks_ub // batch, batch), axis=1)

    o_out = per_step(jnp.where(is_last_k == 1, q_sz_kept, 0))
    lse_bytes_per_token = (
        cfgs.model.num_kv_heads
        * cfgs.aligned_num_q_heads_per_kv_head
        * 128
        * serve.dtype_out.itemsize
    )

    def leaf(spec, value, fill=0):
        """The leaf's buffer: these task slots' words, then `fill` (never read)."""
        flat = value.reshape(-1).astype(i32)
        pad = spec.data.shape[0] - flat.shape[0]
        return dataclasses.replace(
            spec, data=jnp.pad(flat, (0, pad), constant_values=fill)
        )

    return schedule.RpaSchedule(
        s_idx=leaf(out.s_idx, jnp.where(valid, s, -1), fill=-1),
        q_idx=leaf(out.q_idx, jnp.zeros_like(s)),
        k_idx=leaf(out.k_idx, keep(k)),
        is_last_k=leaf(out.is_last_k, is_last_k),
        do_writeback=leaf(out.do_writeback, keep(do_writeback)),
        dma_q=leaf(out.dma_q, dma_q),
        dma_kv_cache=leaf(out.dma_kv_cache, dma_kv_cache),
        dma_kv_new=leaf(out.dma_kv_new, dma_kv_new),
        total_wait_kv_in=leaf(
            out.total_wait_kv_in, per_step(kv_in) * cfgs.kv_bytes_per_token // chunk
        ),
        total_wait_kv_out=leaf(
            out.total_wait_kv_out, per_step(kv_out) * cfgs.kv_bytes_per_token // chunk
        ),
        total_wait_q_in=leaf(
            out.total_wait_q_in, per_step(q_sz_kept) * cfgs.q_bytes_per_token // chunk
        ),
        total_wait_o_out=leaf(
            out.total_wait_o_out, o_out * cfgs.o_bytes_per_token // chunk
        ),
        total_wait_lse_out=leaf(
            out.total_wait_lse_out, o_out * lse_bytes_per_token // chunk
        ),
        actual_steps=_cdiv(total, batch).astype(i32).reshape(1),
        chunk_steps=None,
        cfgs=cfgs,
    )
