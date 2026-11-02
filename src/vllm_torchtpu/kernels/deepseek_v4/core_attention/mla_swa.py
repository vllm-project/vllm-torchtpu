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
"""TPU-Friendly MLA Ragged Paged Attention kernel."""

import functools
from enum import Enum

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


class MlaCase(Enum):
    """Represents the different cases for MLA.

    - DECODE: Sequences are in decode-only mode (q_len = 1).
    - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
    - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
    """

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
        }[self]


def _mla_sliding_window_ragged_paged_attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bo_ids_ref,
    # [6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset,
    #     bkv_sem_0_sz, bkv_sem_1_sz)
    bkv_update_ids_ref,
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    new_kv_hbm_ref,  # [max_num_tokens, token_bytes // 128, 128] uint8
    # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    cache_kv_hbm_ref,
    in_output_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    in_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    in_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    updated_cache_kv_hbm_ref,
    l_hbm_ref,  # [max_num_tokens, num_l_heads]
    m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Scratch
    bkv_x2_ref,  # [2, bkv_sz, token_bytes // 128, 128] uint8
    bq_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bo_x2_ref,  # [2, bq_sz, num_q_heads, head_dim]
    bl_x2_ref,  # [2, bq_sz, num_l_heads]
    bm_x2_ref,  # [2, bq_sz, num_l_heads]
    sems,  # [6, 2]
    l_ref,  # [bq_sz * num_q_heads],
    m_ref,  # [bq_sz * num_q_heads],
    acc_ref,  # [bq_sz * num_q_heads, head_dim],
    *,
    static_q_len: int,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    unnormalized_output: bool,
    non_causal_block: bool,
    q_compute_block_size: int | None,
    bkv_p,
    bq_sz,
    ced_decoder: bool = False,
    skip_kv_update: bool = False,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert sliding_window > 0

    _, num_q_heads, head_dim = q_hbm_ref.shape

    # An int32 CSA host array is viewed as the uint8 slabs it holds.
    if csa_cache_layout.is_word_array(new_kv_hbm_ref.dtype):
        new_kv_hbm_ref = new_kv_hbm_ref.bitcast(jnp.uint8).reshape(
            new_kv_hbm_ref.shape[0],
            new_kv_hbm_ref.shape[1] * csa_cache_layout.SLAB_ROWS,
            new_kv_hbm_ref.shape[2],
        )
    cache_kv_hbm_ref = csa_cache_layout.as_u8_slabs(cache_kv_hbm_ref)
    updated_cache_kv_hbm_ref = csa_cache_layout.as_u8_slabs(updated_cache_kv_hbm_ref)
    total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        cache_kv_hbm_ref.shape
    )
    q_dtype = q_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert head_dim % 128 == 0
    token_bytes = head_dim * get_dtype_bitwidth(q_dtype) // 8
    slot_bytes = kv_packing * lkv_dim
    assert token_bytes % slot_bytes == 0
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token
    ring_cap_tokens = jnp.maximum(
        logical_page_size, (total_num_pages - 1) * logical_page_size
    )

    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    bkv_sz = bkv_p * logical_page_size
    page_size = logical_page_size

    # Static upper bound on the number of bkv blocks any (seq, bq) block can
    # need
    max_bkv_blocks = cdiv(bq_sz + sliding_window - 1, bkv_sz)
    single_bkv_block = max_bkv_blocks == 1

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    seq_idx = pl.program_id(0) + start_seq_idx
    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx] & 0xFFFFF

    def flash_attention(
        bq_vmem_ref,  # [bq_sz, num_q_heads, head_dim] VMEM ref
        kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_x2_ref
        *,
        bq_idx,
        bkv_idx,
        start_offset,
        bo_sem_idx=0,
    ):
        assert len(kv.shape) == 2
        assert kv.shape == (bkv_sz, head_dim)
        n = bq_sz

        # split bq into chunks of size q_compute_block_size
        if q_compute_block_size is None:
            chunk_sz = n
        else:
            chunk_sz = q_compute_block_size if n % q_compute_block_size == 0 else n
        num_chunks = n // chunk_sz

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val), ref[...])

        # k_span is shared across all chunks (only depends on the bkv column).
        k_span = (
            start_offset
            + bkv_idx * bkv_sz
            + lax.broadcasted_iota(jnp.int32, (1, bkv_sz), 1)
        )
        q_base_pos = kv_len - q_len + bq_idx * bq_sz

        # Software-pipeline across `num_chunks` slices so MXU `Q_{c+1} K^T`
        # overlaps with VPU `exp(S_c - m_c)` and MXU `P_c V`.
        chunk_size = chunk_sz * num_q_heads
        num_l_heads = bl_x2_ref.shape[2]
        use_2d_sinks = attention_sinks_ref.ndim == 2
        sinks_chunk = (
            None
            if unnormalized_output
            else (
                attention_sinks_ref.at[0:chunk_size, 0:1][...]
                if use_2d_sinks
                else (
                    attention_sinks_ref[...]
                    if attention_sinks_ref.shape[0] == chunk_size
                    else attention_sinks_ref[:chunk_size]
                )
            )
        )
        q_sub_iota = (
            q_base_pos
            + lax.broadcasted_iota(jnp.int32, (chunk_size, 1), 0) // num_q_heads
        )

        def _compute_qk_and_mask(q_mat, c):
            start_row = c * chunk_size
            qc = q_mat[start_row : start_row + chunk_size]
            q_span = q_sub_iota if c == 0 else (q_sub_iota + c * chunk_sz)
            keep = (q_span - k_span).astype(jnp.uint32) < jnp.uint32(sliding_window)
            if non_causal_block:
                keep = keep | ((k_span >= kv_len - q_len) & (k_span < kv_len))
            s = jnp.einsum("nd,md->nm", qc, kv, preferred_element_type=jnp.float32)
            s *= sm_scale
            return jnp.where(keep, s, jnp.finfo(s.dtype).min)

        def _run_chunk_softmax_pv(s_in, c_sz, c_size, s_sinks, q_row_start, q_row_end, next_fn=None):
            if use_2d_sinks:
                m_curr = jnp.max(s_in, axis=1, keepdims=True)
                p = jnp.exp(s_in - m_curr)
                s_next = next_fn() if next_fn is not None else None
                pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
                l_curr = jnp.sum(p, axis=1, keepdims=True)
                if unnormalized_output:
                    out_c = pv.astype(q_dtype)
                else:
                    l_tot = l_curr + jnp.exp(s_sinks - m_curr)
                    out_c = (
                        lax.div(pv, l_tot)
                        if q_dtype == jnp.float32
                        else (pv * pl.reciprocal(l_tot, approx=True)).astype(q_dtype)
                    )
            else:
                m_curr = jnp.max(s_in, axis=1)
                p = jnp.exp(s_in - m_curr[:, None])
                s_next = next_fn() if next_fn is not None else None
                pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
                l_curr = jnp.sum(p, axis=1)
                if unnormalized_output:
                    out_c = pv.astype(q_dtype)
                else:
                    l_tot = l_curr + jnp.exp(s_sinks - m_curr)
                    out_c = (
                        lax.div(pv, l_tot[:, None])
                        if q_dtype == jnp.float32
                        else (pv * pl.reciprocal(l_tot[:, None], approx=True)).astype(
                            q_dtype
                        )
                    )
            bo_x2_ref.at[bo_sem_idx, q_row_start:q_row_end][...] = out_c.reshape(
                c_sz, num_q_heads, head_dim
            )
            if unnormalized_output:
                l_2d = l_curr.reshape(c_sz, num_q_heads)
                m_2d = m_curr.reshape(c_sz, num_q_heads)
                if num_q_heads < num_l_heads:
                    pad_cfg = ((0, 0), (0, num_l_heads - num_q_heads))
                    l_2d = jnp.pad(l_2d, pad_cfg)
                    m_2d = jnp.pad(m_2d, pad_cfg)
                bl_x2_ref.at[bo_sem_idx, q_row_start:q_row_end][...] = l_2d
                bm_x2_ref.at[bo_sem_idx, q_row_start:q_row_end][...] = m_2d
            return s_next

        short_chunk_sz = 8
        use_short_fastpath = (
            single_bkv_block
            and bq_sz > short_chunk_sz
            and (short_chunk_sz * num_q_heads) % 128 == 0
        )

        if single_bkv_block:
            cur_q_len = q_len - bq_idx * bq_sz
            if use_short_fastpath:
                @pl.when(cur_q_len <= short_chunk_sz)
                def _short_q_path():
                    short_size = short_chunk_sz * num_q_heads
                    qc_short = bq_vmem_ref.at[0:short_chunk_sz][...].reshape(
                        short_size, head_dim
                    )
                    q_span_short = (
                        q_base_pos
                        + lax.broadcasted_iota(jnp.int32, (short_size, 1), 0)
                        // num_q_heads
                    )
                    keep_short = (q_span_short - k_span).astype(
                        jnp.uint32
                    ) < jnp.uint32(sliding_window)
                    if non_causal_block:
                        keep_short = keep_short | (
                            (k_span >= kv_len - q_len) & (k_span < kv_len)
                        )
                    s_short = jnp.einsum(
                        "nd,md->nm",
                        qc_short,
                        kv,
                        preferred_element_type=jnp.float32,
                    )
                    s_short *= sm_scale
                    s_short = jnp.where(
                        keep_short, s_short, jnp.finfo(s_short.dtype).min
                    )
                    sinks_short = (
                        None
                        if unnormalized_output
                        else (
                            attention_sinks_ref.at[0:short_size, 0:1][...]
                            if use_2d_sinks
                            else attention_sinks_ref[:short_size]
                        )
                    )
                    _run_chunk_softmax_pv(
                        s_short,
                        short_chunk_sz,
                        short_size,
                        sinks_short,
                        0,
                        short_chunk_sz,
                        None,
                    )

                @pl.when(cur_q_len > short_chunk_sz)
                def _full_q_path():
                    q_full = bq_vmem_ref[...].reshape(bq_sz * num_q_heads, head_dim)
                    s_curr = _compute_qk_and_mask(q_full, 0)
                    for c in range(num_chunks):
                        q_row_start = c * chunk_sz
                        q_row_end = q_row_start + chunk_sz
                        next_fn = (
                            (lambda c_next=c + 1: _compute_qk_and_mask(q_full, c_next))
                            if c + 1 < num_chunks
                            else None
                        )
                        s_curr = _run_chunk_softmax_pv(
                            s_curr,
                            chunk_sz,
                            chunk_size,
                            sinks_chunk,
                            q_row_start,
                            q_row_end,
                            next_fn,
                        )
            else:
                q_full = bq_vmem_ref[...].reshape(bq_sz * num_q_heads, head_dim)
                s_curr = _compute_qk_and_mask(q_full, 0)
                for c in range(num_chunks):
                    q_row_start = c * chunk_sz
                    q_row_end = q_row_start + chunk_sz
                    next_fn = (
                        (lambda c_next=c + 1: _compute_qk_and_mask(q_full, c_next))
                        if c + 1 < num_chunks
                        else None
                    )
                    s_curr = _run_chunk_softmax_pv(
                        s_curr,
                        chunk_sz,
                        chunk_size,
                        sinks_chunk,
                        q_row_start,
                        q_row_end,
                        next_fn,
                    )
        else:
            q_full = bq_vmem_ref[...].reshape(bq_sz * num_q_heads, head_dim)
            s_curr = _compute_qk_and_mask(q_full, 0)
            for c in range(num_chunks):
                start_row = c * chunk_size
                s_rowmax = jnp.max(s_curr, axis=1)
                cl_ref = l_ref.at[start_row : start_row + chunk_size]
                cm_ref = m_ref.at[start_row : start_row + chunk_size]
                cacc_ref = acc_ref.at[start_row : start_row + chunk_size]
                m_prev = load_with_init(cm_ref, jnp.finfo(jnp.float32).min)
                m_curr = jnp.maximum(m_prev, s_rowmax)
                cm_ref[...] = m_curr
                p = jnp.exp(s_curr - m_curr[:, None])
                s_next = _compute_qk_and_mask(q_full, c + 1) if c + 1 < num_chunks else None
                pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
                p_rowsum = jnp.sum(p, axis=1)
                exp_m_diff = jnp.exp(m_prev - m_curr)
                l_prev = load_with_init(cl_ref, 0.0)
                l_curr = exp_m_diff * l_prev + p_rowsum
                cl_ref[...] = l_curr
                o_prev = load_with_init(cacc_ref, 0.0)
                o_curr = exp_m_diff[:, None] * o_prev + pv
                cacc_ref[...] = o_curr
                s_curr = s_next

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _get_kv_len(seq_idx):
        return jnp.where(seq_idx < end_seq_idx, kv_lens_ref[seq_idx] & 0xFFFFF, 0)

    def _get_q_len(seq_idx):
        return jnp.where(
            seq_idx < end_seq_idx,
            cu_q_lens_ref[seq_idx + 1] - cu_q_lens_ref[seq_idx],
            0,
        )

    def _start_offset(seq_idx, bq_idx):
        # Exact start of the sliding window for this (seq, bq) block, in tokens.
        # The first query row of this bq block sits at position
        #   q_pos_min = kv_len - q_len + bq_idx * bq_sz
        # and attends to keys k with 0 <= q_pos_min - k < sliding_window, so the
        # first key it can ever need is q_pos_min - sliding_window + 1.
        kv_len_s = _get_kv_len(seq_idx)
        q_len_s = _get_q_len(seq_idx)
        eff_q_span_s = jnp.where(
            seq_idx < end_seq_idx,
            jnp.maximum(q_len_s, kv_lens_ref[seq_idx] >> 20),
            0,
        )
        min_start = (
            jnp.where(eff_q_span_s > 8, kv_len_s - eff_q_span_s, 0)
            if ced_decoder
            else 0
        )
        return jnp.maximum(
            kv_len_s - q_len_s + bq_idx * bq_sz - sliding_window + 1,
            min_start,
        )

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        # bkv_x2_ref shape: [2, bkv_sz, slots_per_token * kv_packing, lkv_dim]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        # View the physical uint8 cache [total_num_pages, slots, kv_packing, lkv_dim]
        # as one contiguous row per token: [total_num_pages * phys_tokens_per_page,
        # token_bytes].
        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * phys_tokens_per_page,
            slots_per_token * kv_packing,
            lkv_dim,
        )

        kv_len = kv_lens_ref[seq_idx] & 0xFFFFF
        # Window base for this (seq, bq) block plus the relative block. `start_offset`
        # is an exact token position, so `kv_len_start` is generally not a multiple
        # of page_size; `page_off` is where inside its page the window starts.
        kv_len_start = start_offset + bkv_idx * bkv_sz
        kv_p_start = kv_len_start // page_size
        page_off = kv_len_start - kv_p_start * page_size

        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = jnp.maximum(kv_len - kv_len_start, 0)
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        new_kv_len_start = q_end - kv_left_frm_new
        dma_bkv_sz = bkv_sz_frm_cache + bkv_sz_frm_new

        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

            # Fetch effective kv from kv cache. To pipeline multiple DMA calls, we
            # utilize static for loop instead of dynamic for loop.
            #
            # The window is not page-aligned, so it spans bkv_p + 1 source pages: a
            # partial head page (starting at `page_off`), bkv_p - 1 full pages, and a
            # partial tail page contributing `page_off` tokens. When page_off == 0
            # the tail copy degenerates to size 0.
            @pl.when(bkv_sz_frm_cache > 0)
            def _fetch_from_cache():
                for i in range(bkv_p + 1):
                    if i == 0:
                        # Head page: skip the first `page_off` tokens of the page.
                        in_page_off = page_off
                        vmem_off = jnp.int32(0)
                        avail = page_size - page_off
                    else:
                        in_page_off = jnp.int32(0)
                        vmem_off = i * page_size - page_off
                        avail = jnp.int32(page_size)
                    # Ensure only effective kvs are copied, that we don't go negative, and
                    # that we never write past the end of the bkv buffer.
                    sz = jnp.clip(bkv_sz_frm_cache - vmem_off, 0, avail)
                    # If the page index is out of bound, we set page_idx to the last page.
                    # And there will be no copy since sz will be 0.
                    page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                    _async_copy(
                        reshaped_cache_hbm_ref.at[
                            pl.ds(
                                page_indices_ref[page_idx] * phys_tokens_per_page
                                + in_page_off,
                                sz,
                            ),
                        ],
                        bkv_vmem_ref.at[pl.ds(vmem_off, sz)],
                        sem,
                        wait,
                    )

            # Fetch new KVs by appending them right after the last cache token. Token
            # rows are self-contained, so new KV begins exactly where cache KV ends.
            _async_copy(
                new_kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                bkv_vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                sem,
                wait,
            )

        else:
            # When we wait, we can use a dummy copy to wait for DMAs to complete where
            # src == dst. However, the dma size must be correct.
            dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
            _async_copy(
                src=dst_kv,
                dst=dst_kv,
                sem=sem,
                wait=True,
            )

        # This returns (offset, size, vmem_start) in units of tokens:
        #   offset: starting token index where the new KV should be stored
        #   size: number of tokens of the new KV, which is 1 in decode.
        #   vmem_start: where the new KV starts within the bkv VMEM buffer.
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new, bkv_sz_frm_cache

    def _update_kv_cache(
        seq_idx,
        bkv_sem_idx,
        offset,  # In units of tokens.
        update_sz,  # In units of tokens.
        *,
        in_vmem_start=0,  # New-KV start within the bkv VMEM buffer.
        wait=False,
    ):
        if skip_kv_update:
            return
        sem = sems.at[3, bkv_sem_idx]
        bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        update_kv_packing_iters = update_sz

        # View the cache as one contiguous row per token (see _fetch_bkv).
        reshaped_cache_kv_hbm_ref = updated_cache_kv_hbm_ref.reshape(
            total_num_pages * phys_tokens_per_page,
            slots_per_token * kv_packing,
            lkv_dim,
        )

        if not wait:
            # Issue pipelined static DMA copies page by page without dynamic fori_loop.
            kv_p_start = offset // page_size
            start_word_in_page = offset - kv_p_start * page_size
            page_indices_offset = seq_idx * pages_per_seq + kv_p_start
            max_update_pages = cdiv(min(bq_sz, bkv_sz), page_size) + 1

            for i in range(max_update_pages):
                if i == 0:
                    curr_word_in_page = start_word_in_page
                    vmem_delta = jnp.int32(0)
                    avail = page_size - start_word_in_page
                else:
                    curr_word_in_page = jnp.int32(0)
                    vmem_delta = i * page_size - start_word_in_page
                    avail = jnp.int32(page_size)
                sz = jnp.clip(update_kv_packing_iters - vmem_delta, 0, avail)
                page_table_idx = jnp.minimum(
                    page_indices_offset + i, num_page_indices - 1
                )
                page_idx = page_indices_ref[page_table_idx]

                _async_copy(
                    bkv_vmem_ref.at[pl.ds(in_vmem_start + vmem_delta, sz)],
                    reshaped_cache_kv_hbm_ref.at[
                        pl.ds(
                            page_idx * phys_tokens_per_page + curr_word_in_page,
                            sz,
                        ),
                    ],
                    sem,
                    wait=False,
                )
        else:  # Wait
            dma_sz_words = update_kv_packing_iters
            dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_sz_words)]
            _async_copy(
                src=dst_kv,
                dst=dst_kv,
                sem=sem,
                wait=True,
            )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            q_hbm_ref.at[pl.ds(q_len_start, sz)],
            bq_vmem_ref.at[pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            o_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def _send_l(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[4, bo_sem_idx]
        vmem_ref = bl_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            l_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def _send_m(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[5, bo_sem_idx]
        vmem_ref = bm_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            m_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, start_offset, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)
        if unnormalized_output:
            _send_l(seq_idx, bo_idx, bo_sem_idx)
            _send_m(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(old_seq_idx >= 0, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)
            if unnormalized_output:
                _send_l(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)
                _send_m(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, vmem_start):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(
            seq_idx, bkv_sem_idx, offset, update_sz, in_vmem_start=vmem_start
        )

    def wait_update_kv_cache(bkv_sem_idx):
        if skip_kv_update:
            return
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)

    def load_bkv(bkv_sem_idx, bkv_idx, start_offset):
        bkv_u8 = bkv_x2_ref.at[bkv_sem_idx][...]
        # [bkv_sz, slots_per_token * kv_packing, lkv_dim] -> [bkv_sz, token_bytes]
        # bkv_u8: [bkv_sz, slots_per_token * kv_packing, lkv_dim] uint8. The bytes
        # were stored (see `prepare_kv_inputs`) so that each pair of consecutive
        # sublanes already holds one bf16 value in little-endian order (lo byte then
        # hi byte). `pltpu.bitcast` fuses consecutive sublanes little-endian, so the
        # reconstruction is a plain bitcast + reshape.
        bkv = pltpu.bitcast(bkv_u8, jnp.bfloat16).reshape(bkv_sz, head_dim)
        return bkv

    def process():
        if static_q_len is None:
            num_bq = jnp.maximum(1, cdiv(q_len, bq_sz))
        else:
            num_bq = jnp.maximum(1, cdiv(static_q_len, bq_sz))

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def compute_with_bq(bq_idx, _):
            # given it's sliding window attention, we only need to scan over the
            # needed bkv indices within the window of the current bq block.
            # Block indices are relative to the page-aligned window base
            # (_start_offset), so they always start at 0 for each bq block.
            #
            # _start_offset only depends on (seq_idx, bq_idx), so it is constant for
            # the whole current bq block. Compute it once here.
            cur_start_offset = _start_offset(seq_idx, bq_idx)
            start_bkv_idx = 0
            if single_bkv_block:
                end_bkv_idx = 1
            else:
                if non_causal_block:
                    # Draft block mode: every query in the block may attend to the
                    # whole block, which ends at kv_len, not at this bq block's last
                    # query.
                    q_pos_end = kv_len
                else:
                    # Causal: no query sees past its own position, so the last query
                    # of this bq block bounds the keys needed.
                    q_pos_end = jnp.minimum(
                        kv_len - q_len + (bq_idx + 1) * bq_sz, kv_len
                    )
                end_bkv_idx = jnp.maximum(cdiv(q_pos_end - cur_start_offset, bkv_sz), 1)

            def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
                next_bkv_idx = bkv_idx + 1
                is_last_bkv = next_bkv_idx == end_bkv_idx
                next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
                is_last_bq = next_bq_idx == num_bq
                next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
                next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
                # Next bq block restarts its relative block index at 0.
                next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
                next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
                return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                # Get next bkv ids.
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx = (
                    get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)
                )

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    # The next block may belong to a different (seq, bq), so it needs
                    # its own start offset.
                    next_start_offset = _start_offset(next_seq_idx, next_bq_idx)
                    start_fetch_bkv(
                        next_seq_idx,
                        next_bkv_idx,
                        next_bkv_sem_idx,
                        next_start_offset,
                    )

                # Wait for cur bkv
                offset, update_sz, vmem_start = wait_fetch_bkv(
                    seq_idx, bkv_idx, bkv_sem_idx, cur_start_offset
                )

                # Start updating bkv to kv cache if applicable.
                # Only write tokens within the surviving ring capacity [kv_len - ring_cap_tokens, kv_len).
                if not skip_kv_update:
                    q_pos_start = kv_len - q_len + bq_idx * bq_sz
                    q_pos_end = jnp.minimum(
                        kv_len - q_len + (bq_idx + 1) * bq_sz, kv_len
                    )
                    write_start = jnp.maximum(
                        jnp.maximum(offset, q_pos_start), kv_len - ring_cap_tokens
                    )
                    write_end = jnp.minimum(offset + update_sz, q_pos_end)
                    actual_update_sz = jnp.maximum(0, write_end - write_start)
                    actual_vmem_start = vmem_start + (write_start - offset)

                    @pl.when(actual_update_sz > 0)
                    def update_cur_bkv_to_cache():
                        start_update_kv_cache(
                            seq_idx,
                            bkv_sem_idx,
                            write_start,
                            actual_update_sz,
                            actual_vmem_start,
                        )

                # Load bkv into vreg. There is no need to mask out invalid k/v entries,
                # because the score of invalid Q.K^T pairs are masked (to be zero) in
                # flash attention, so that the invalid kv entries
                # (as long as they are not NaN or inf) won't affect to the output.
                bkv = load_bkv(
                    bkv_sem_idx,
                    bkv_idx,
                    cur_start_offset,
                )

                if single_bkv_block:
                    wait_send_bo(bo_sem_idx)

                flash_attention(
                    bq_x2_ref.at[bq_sem_idx],
                    bkv,
                    bq_idx=bq_idx,
                    bkv_idx=bkv_idx,
                    start_offset=cur_start_offset,
                    bo_sem_idx=bo_sem_idx,
                )

            # Wait for cur bq if not ready yet
            wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)
            if single_bkv_block:
                bo_sem_idx = sem_ids_ref[2]
                sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
                compute_with_bkv(0, None)
                start_send_bo(seq_idx, bq_idx, bo_sem_idx)
            else:
                bo_sem_idx = 0
                lax.fori_loop(
                    start_bkv_idx,
                    end_bkv_idx,
                    compute_with_bkv,
                    None,
                    unroll=False,
                )

                # Load acc and calculate final output.
                acc = acc_ref[...]

                if unnormalized_output:
                    out = acc.astype(q_dtype)
                else:
                    attention_sinks = (
                        jnp.concat([attention_sinks_ref[:, 0] for _ in range(bq_sz)])
                        if attention_sinks_ref.ndim == 2
                        else attention_sinks_ref[...]
                    )
                    exp_attention_sinks = jnp.exp(attention_sinks - m_ref[...])
                    l = l_ref[...] + exp_attention_sinks  # noqa: E741
                    out = (
                        lax.div(acc, l[:, None])
                        if q_dtype == jnp.float32
                        else (acc * pl.reciprocal(l[:, None], approx=True)).astype(
                            q_dtype
                        )
                    )

                # Wait for previous bo to be fully sent before storing new bo.
                bo_sem_idx = sem_ids_ref[2]
                sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
                wait_send_bo(bo_sem_idx)

                # Store output from acc to bo.
                bo_x2_ref.at[bo_sem_idx][...] = out.reshape(bq_sz, num_q_heads, head_dim)
                if unnormalized_output:
                    if num_q_heads % 128 == 0:
                        bl_x2_ref.at[bo_sem_idx][...] = l_ref[...].reshape(
                            bq_sz, num_q_heads
                        )
                        bm_x2_ref.at[bo_sem_idx][...] = m_ref[...].reshape(
                            bq_sz, num_q_heads
                        )
                    else:
                        for i in range(bq_sz):
                            start = i * num_q_heads
                            end = start + num_q_heads
                            bl_x2_ref.at[bo_sem_idx, i][:num_q_heads] = l_ref[start:end]
                            bm_x2_ref.at[bo_sem_idx, i][:num_q_heads] = m_ref[start:end]

                # Send cur bo
                start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)

        # Initialize bkv_x2_ref to avoid NaN issues from accessing uninitialized
        # memory
        bkv_zeros = jnp.zeros(bkv_x2_ref.shape[1:], bkv_x2_ref.dtype)
        bkv_x2_ref[0] = bkv_zeros
        # bq_idx=0, relative bkv_idx=0 (page-aligned base via _start_offset).
        start_fetch_bkv(start_seq_idx, 0, 0, _start_offset(start_seq_idx, 0))
        bkv_x2_ref[1] = bkv_zeros

    process()

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(i)
            wait_update_kv_cache(i)

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    if num_q_heads != actual_num_q_heads or head_dim != actual_head_dim:
        q = jnp.pad(
            q,
            (
                (0, 0),
                (0, num_q_heads - actual_num_q_heads),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        )
    return q


def prepare_kv_inputs(kv: jax.Array):
    """Reinterprets bf16 new KV as one raw uint8 byte row per token.

    Each token's `actual_head_dim` bf16 values become `actual_head_dim * 2`
    contiguous bytes. The bytes are laid out so that, once a token row is viewed
    as ``[slots_per_token * kv_packing, 128]`` (128 = the kernel's lane width),
    every pair of consecutive sublanes holds one bf16 value in little-endian byte
    order (lo byte then hi byte).
    """
    assert kv.dtype == jnp.bfloat16
    tokens, head_dim = kv.shape
    assert head_dim % 128 == 0
    kv_u16 = jax.lax.bitcast_convert_type(kv, jnp.uint16)
    kv_hi = ((kv_u16 >> 8) & 0xFF).astype(jnp.uint8)
    kv_lo = (kv_u16 & 0xFF).astype(jnp.uint8)
    nb = head_dim // 128
    kv_hi = kv_hi.reshape(tokens, nb, 128)
    kv_lo = kv_lo.reshape(tokens, nb, 128)
    # (tokens, nb, 2, 128)
    interleaved = jnp.stack([kv_lo, kv_hi], axis=2)
    return interleaved.reshape(tokens, head_dim * 2)


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    return out[:, :actual_num_q_heads, :actual_head_dim]


def get_physical_layout(
    cache_kv_shape, q_dtype, head_dim, max_num_seqs, num_page_indices
):
    total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        cache_kv_shape
    )
    token_bytes = head_dim * get_dtype_bitwidth(q_dtype) // 8
    slot_bytes = kv_packing * lkv_dim
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token
    pages_per_seq = num_page_indices // max_num_seqs

    return (
        total_num_pages,
        kv_packing,
        lkv_dim,
        slots_per_token,
        phys_tokens_per_page,
        pages_per_seq,
    )


def _mla_batched_decode_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2]
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    new_kv_hbm_ref,  # [max_num_tokens, token_bytes // 128, 128] uint8
    # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    cache_kv_hbm_ref,
    in_output_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    in_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    in_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    updated_cache_kv_hbm_ref,
    l_hbm_ref,  # [max_num_tokens, num_l_heads]
    m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Scratch (x2 for double buffering)
    # [2, seq_batch_sz, bkv_sz, slots_per_token * kv_packing, lkv_dim] uint8
    bkv_batch_x2_vmem,
    q_batch_x2_vmem,  # [2, seq_batch_sz, num_q_heads, head_dim]
    o_batch_x2_vmem,  # [2, seq_batch_sz, num_q_heads, head_dim]
    l_batch_x2_vmem,  # [2, seq_batch_sz, num_l_heads]
    m_batch_x2_vmem,  # [2, seq_batch_sz, num_l_heads]
    new_kv_batch_x2_vmem,  # [2, seq_batch_sz, ...]
    sems,  # [2, 7]
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    unnormalized_output: bool,
    bkv_p: int,
    seq_batch_sz: int,
):
    del in_output_hbm_ref, in_l_hbm_ref, in_m_hbm_ref  # Aliased with out, l, m.

    def async_copy(src, dst, sem, wait=False):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    _, num_q_heads, head_dim = q_hbm_ref.shape
    new_kv_raw_hbm_ref = new_kv_hbm_ref
    cache_kv_hbm_ref = csa_cache_layout.as_u8_slabs(cache_kv_hbm_ref)
    updated_cache_kv_hbm_ref = csa_cache_layout.as_u8_slabs(updated_cache_kv_hbm_ref)
    total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        cache_kv_hbm_ref.shape
    )
    q_dtype = q_hbm_ref.dtype
    token_bytes = head_dim * get_dtype_bitwidth(q_dtype) // 8
    slot_bytes = kv_packing * lkv_dim
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token

    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs

    bkv_sz = bkv_p * logical_page_size

    reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
        total_num_pages * phys_tokens_per_page,
        slots_per_token * kv_packing,
        lkv_dim,
    )
    reshaped_updated_cache_hbm_ref = updated_cache_kv_hbm_ref.reshape(
        total_num_pages * phys_tokens_per_page,
        slots_per_token * kv_packing,
        lkv_dim,
    )

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]

    num_decode_seqs = end_seq_idx - start_seq_idx
    num_batches = cdiv(num_decode_seqs, seq_batch_sz)

    def get_bkv_sz_frm_cache(kv_len):
        start_offset = jnp.maximum(kv_len - sliding_window, jnp.int32(0))
        kv_left = jnp.maximum(kv_len - start_offset, jnp.int32(0))
        kv_left_frm_cache = jnp.maximum(kv_left - 1, jnp.int32(0))
        return start_offset, jnp.minimum(kv_left_frm_cache, bkv_sz)

    def _fetch_bq(batch_idx, sem_idx, wait=False):
        if not wait:
            wait_update_kv_cache(batch_idx - 2, sem_idx)
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        q_start_seq = jnp.minimum(s_start, max_num_seqs - 1)
        q_token_idx = jnp.where(
            s_start < end_seq_idx,
            cu_q_lens_ref[q_start_seq],
            0,
        )
        batch_sz = jnp.clip(end_seq_idx - s_start, 0, seq_batch_sz)
        async_copy(
            q_hbm_ref.at[pl.ds(q_token_idx, batch_sz)],
            q_batch_x2_vmem.at[sem_idx, pl.ds(0, batch_sz)],
            sems.at[sem_idx, 0],
            wait=wait,
        )
        async_copy(
            new_kv_raw_hbm_ref.at[pl.ds(q_token_idx, batch_sz)],
            new_kv_batch_x2_vmem.at[sem_idx, pl.ds(0, batch_sz)],
            sems.at[sem_idx, 5],
            wait=wait,
        )

    def _fetch_bkv(batch_idx, sem_idx, wait=False):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        sem_kv = sems.at[sem_idx, 4]
        if wait:
            total_rows = jnp.int32(0)
            for s in range(seq_batch_sz):
                seq_idx = s_start + s
                seq_idx_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
                kv_len = jnp.where(
                    seq_idx < end_seq_idx,
                    kv_lens_ref[seq_idx_clamped],
                    0,
                )
                _, bkv_sz_frm_cache = get_bkv_sz_frm_cache(kv_len)
                total_rows = total_rows + jnp.where(
                    seq_idx < end_seq_idx, bkv_sz_frm_cache, 0
                )
            dst_all = bkv_batch_x2_vmem.at[sem_idx, pl.ds(0, total_rows)]
            async_copy(dst_all, dst_all, sem_kv, wait=True)
            return

        for s in range(seq_batch_sz):
            seq_idx = s_start + s
            seq_idx_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
            kv_len = jnp.where(
                seq_idx < end_seq_idx,
                kv_lens_ref[seq_idx_clamped],
                0,
            )

            start_offset, bkv_sz_frm_cache = get_bkv_sz_frm_cache(kv_len)
            kv_p_start = start_offset // logical_page_size
            page_off = start_offset - kv_p_start * logical_page_size
            page_indices_offset = jnp.minimum(
                seq_idx_clamped * pages_per_seq + kv_p_start,
                num_page_indices - 1,
            )
            bkv_s_base = s * bkv_sz

            vmem_off = 0
            for i in range(bkv_p + 1):
                avail = logical_page_size - page_off
                chunk_sz = jnp.where(
                    seq_idx < end_seq_idx,
                    jnp.clip(bkv_sz_frm_cache - vmem_off, jnp.int32(0), avail),
                    0,
                )
                page_idx = page_indices_ref[
                    jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                ]
                async_copy(
                    reshaped_cache_hbm_ref.at[
                        pl.ds(
                            page_idx * phys_tokens_per_page + page_off,
                            chunk_sz,
                        )
                    ],
                    bkv_batch_x2_vmem.at[
                        sem_idx, pl.ds(bkv_s_base + vmem_off, chunk_sz)
                    ],
                    sem_kv,
                    wait=wait,
                )
                page_off = 0
                vmem_off += chunk_sz

    compute_sub_sz = min(seq_batch_sz, 8)
    m_vec_sub = lax.broadcasted_iota(jnp.int32, (1, 1, bkv_sz), 2)

    def compute_batch(batch_idx, sem_idx):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        if not unnormalized_output:
            attention_sinks = attention_sinks_ref[...][None, :]
        new_kv_u8_vmem = new_kv_batch_x2_vmem.at[sem_idx].bitcast(jnp.uint8).reshape(
            seq_batch_sz, slots_per_token * kv_packing, lkv_dim
        )

        for sub_off in range(0, seq_batch_sz, compute_sub_sz):
            kv_lens_sub = jnp.stack(
                [
                    jnp.where(
                        (s_start + sub_off + u) < end_seq_idx,
                        kv_lens_ref[
                            jnp.minimum(s_start + sub_off + u, max_num_seqs - 1)
                        ],
                        0,
                    )
                    for u in range(compute_sub_sz)
                ]
            )
            for u in range(compute_sub_sz):
                _, bkv_sz_u = get_bkv_sz_frm_cache(kv_lens_sub[u])
                bkv_batch_x2_vmem[
                    sem_idx, (sub_off + u) * bkv_sz + bkv_sz_u, ...
                ] = new_kv_u8_vmem[sub_off + u, ...]

            bkv_sub = pltpu.bitcast(
                bkv_batch_x2_vmem[
                    sem_idx,
                    sub_off * bkv_sz : (sub_off + compute_sub_sz) * bkv_sz,
                    ...,
                ],
                jnp.bfloat16,
            ).reshape(compute_sub_sz, bkv_sz, head_dim)
            q_sub = q_batch_x2_vmem[sem_idx, sub_off : sub_off + compute_sub_sz, ...]

            cur_start_offset_sub = jnp.maximum(
                kv_lens_sub - sliding_window, jnp.int32(0)
            )
            k_span_sub = cur_start_offset_sub[:, None, None] + m_vec_sub
            q_span_sub = (kv_lens_sub - 1)[:, None, None]
            keep_sub = (
                (q_span_sub - k_span_sub).astype(jnp.uint32)
                < jnp.uint32(sliding_window)
            ) | (m_vec_sub == 0)
            bkv_w = jnp.where(jnp.swapaxes(keep_sub, 1, 2), bkv_sub, 0)
            s_mat = (
                jnp.einsum(
                    "snd,smd->snm",
                    q_sub,
                    bkv_w,
                    preferred_element_type=jnp.float32,
                )
                * sm_scale
            )
            s_mat = jnp.where(keep_sub, s_mat, jnp.finfo(jnp.float32).min)

            s_max = jnp.max(s_mat, axis=2, keepdims=True)
            p_mat = jnp.exp(s_mat - s_max)
            pv_sub = jnp.einsum(
                "snm,smd->snd", p_mat, bkv_w, preferred_element_type=jnp.float32
            )

            m_val = s_max[:, :, 0]
            l_val = jnp.sum(p_mat, axis=2)

            if unnormalized_output:
                out = pv_sub.astype(q_dtype)
            else:
                exp_attention_sinks = jnp.exp(attention_sinks - m_val)
                l_tile = (l_val + exp_attention_sinks)[:, :, None]
                out = (
                    lax.div(pv_sub, l_tile)
                    if q_dtype == jnp.float32
                    else (pv_sub * pl.reciprocal(l_tile, approx=True)).astype(q_dtype)
                )

            o_batch_x2_vmem[sem_idx, sub_off : sub_off + compute_sub_sz, ...] = out
            if unnormalized_output:
                num_l_heads = l_batch_x2_vmem.shape[2]
                if num_l_heads == num_q_heads:
                    l_batch_x2_vmem[
                        sem_idx, sub_off : sub_off + compute_sub_sz, ...
                    ] = l_val
                    m_batch_x2_vmem[
                        sem_idx, sub_off : sub_off + compute_sub_sz, ...
                    ] = m_val
                else:
                    l_batch_x2_vmem[
                        sem_idx, sub_off : sub_off + compute_sub_sz, ...
                    ] = jnp.pad(l_val, ((0, 0), (0, num_l_heads - num_q_heads)))
                    m_batch_x2_vmem[
                        sem_idx, sub_off : sub_off + compute_sub_sz, ...
                    ] = jnp.pad(m_val, ((0, 0), (0, num_l_heads - num_q_heads)))

    def _send_bo(batch_idx, sem_idx, wait=False):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        q_start_seq = jnp.minimum(s_start, max_num_seqs - 1)
        q_token_idx = jnp.where(
            s_start < end_seq_idx,
            cu_q_lens_ref[q_start_seq],
            0,
        )
        batch_sz = jnp.clip(end_seq_idx - s_start, 0, seq_batch_sz)
        async_copy(
            o_batch_x2_vmem.at[sem_idx, pl.ds(0, batch_sz)],
            o_hbm_ref.at[pl.ds(q_token_idx, batch_sz)],
            sems.at[sem_idx, 1],
            wait=wait,
        )
        if unnormalized_output:
            async_copy(
                l_batch_x2_vmem.at[sem_idx, pl.ds(0, batch_sz)],
                l_hbm_ref.at[pl.ds(q_token_idx, batch_sz)],
                sems.at[sem_idx, 2],
                wait=wait,
            )
            async_copy(
                m_batch_x2_vmem.at[sem_idx, pl.ds(0, batch_sz)],
                m_hbm_ref.at[pl.ds(q_token_idx, batch_sz)],
                sems.at[sem_idx, 3],
                wait=wait,
            )

    def _update_kv_cache(batch_idx, sem_idx, wait=False):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        sem_upd = sems.at[sem_idx, 6]
        new_kv_u8_vmem_ref = new_kv_batch_x2_vmem.at[sem_idx].bitcast(
            jnp.uint8
        ).reshape(seq_batch_sz, slots_per_token * kv_packing, lkv_dim)
        if wait:
            batch_sz = jnp.clip(end_seq_idx - s_start, 0, seq_batch_sz)
            dst_upd = new_kv_u8_vmem_ref.at[pl.ds(0, batch_sz)]
            async_copy(dst_upd, dst_upd, sem_upd, wait=True)
            return

        for s in range(seq_batch_sz):
            seq_idx = s_start + s
            seq_idx_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
            sz = jnp.where(seq_idx < end_seq_idx, 1, 0)

            kv_len = jnp.where(
                seq_idx < end_seq_idx,
                kv_lens_ref[seq_idx_clamped],
                0,
            )
            new_token_idx = jnp.maximum(kv_len - 1, 0)
            new_token_p = new_token_idx // logical_page_size
            new_token_off = new_token_idx - new_token_p * logical_page_size

            new_token_page_idx = page_indices_ref[
                jnp.minimum(
                    seq_idx_clamped * pages_per_seq + new_token_p,
                    num_page_indices - 1,
                )
            ]

            async_copy(
                new_kv_u8_vmem_ref.at[pl.ds(s, sz)],
                reshaped_updated_cache_hbm_ref.at[
                    pl.ds(
                        new_token_page_idx * phys_tokens_per_page + new_token_off,
                        sz,
                    )
                ],
                sem_upd,
                wait=wait,
            )

    def fetch_batch(batch_idx, sem_idx):
        _fetch_bq(batch_idx, sem_idx, wait=False)
        _fetch_bkv(batch_idx, sem_idx, wait=False)

    def wait_fetch(batch_idx, sem_idx):
        _fetch_bq(batch_idx, sem_idx, wait=True)
        _fetch_bkv(batch_idx, sem_idx, wait=True)

    def wait_update_kv_cache(batch_idx, sem_idx):
        @pl.when(batch_idx >= 0)
        def _():
            _update_kv_cache(batch_idx, sem_idx, wait=True)

    def wait_send_bo(batch_idx, sem_idx):
        @pl.when(batch_idx >= 0)
        def _():
            _send_bo(batch_idx, sem_idx, wait=True)

    start_batch_idx = 0

    @pl.when(start_batch_idx < num_batches)
    def _():
        fetch_batch(start_batch_idx, 0)

    def loop_body(i, _):
        batch_idx = i

        sem_idx = i % 2
        next_sem_idx = (i + 1) % 2

        next_batch_idx = batch_idx + 1

        @pl.when(next_batch_idx < num_batches)
        def _():
            fetch_batch(next_batch_idx, next_sem_idx)

        wait_fetch(batch_idx, sem_idx)

        wait_send_bo(batch_idx - 2, sem_idx)

        compute_batch(batch_idx, sem_idx)

        _send_bo(batch_idx, sem_idx, wait=False)
        _update_kv_cache(batch_idx, sem_idx, wait=False)

        return ()

    loop_cnt = num_batches
    lax.fori_loop(0, loop_cnt, loop_body, ())

    @pl.when(loop_cnt > 0)
    def _():
        last_i = loop_cnt - 1
        last_batch_idx = last_i
        wait_update_kv_cache(last_batch_idx, last_i % 2)
        wait_send_bo(last_batch_idx, last_i % 2)

        @pl.when(loop_cnt > 1)
        def _():
            wait_send_bo(last_batch_idx - 1, (last_i - 1) % 2)
            wait_update_kv_cache(last_batch_idx - 1, (last_i - 1) % 2)


def run_mla_batched_decode_kernel(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    new_kv: jax.Array,  # [max_num_tokens, lkv_dim]
    # [total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim]
    cache_kv: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_kv_pages_per_block: int,
    num_queries_per_block: int,
    start_seq_idx: jax.Array,  # i32
    end_seq_idx: jax.Array,  # i32
    in_output: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    in_l: jax.Array,  # [max_num_tokens, num_l_heads]
    in_m: jax.Array,  # [max_num_tokens, num_l_heads]
    attention_sinks: jax.Array,  # float32[num_q_heads]
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    vmem_limit_bytes: int,
    unnormalized_output: bool,
    seq_batch_sz: int = 32,
):
    num_decode_seqs = end_seq_idx - start_seq_idx
    bkv_p = num_kv_pages_per_block
    bkv_sz = bkv_p * logical_page_size

    grid = (jnp.where(num_decode_seqs > 0, 1, 0),)

    _, num_q_heads, head_dim = q.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    num_l_heads = align_to(num_q_heads, 128)
    (
        total_num_pages,
        kv_packing,
        lkv_dim,
        slots_per_token,
        phys_tokens_per_page,
        pages_per_seq,
    ) = get_physical_layout(
        csa_cache_layout.u8_slab_shape(cache_kv.shape, cache_kv.dtype),
        q.dtype,
        head_dim,
        max_num_seqs,
        num_page_indices,
    )

    scalar_prefetches = (
        kv_lens,
        page_indices,
        cu_q_lens,
        jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
    )

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
        pl.BlockSpec(memory_space=pltpu.HBM),  # q
        pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
        pl.BlockSpec(memory_space=pltpu.HBM),  # in_output
        pl.BlockSpec(memory_space=pltpu.HBM),  # in_l
        pl.BlockSpec(memory_space=pltpu.HBM),  # in_m
    ]
    out_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),  # out
        pl.BlockSpec(memory_space=pltpu.HBM),  # updated_cache_kv
        pl.BlockSpec(memory_space=pltpu.HBM),  # l
        pl.BlockSpec(memory_space=pltpu.HBM),  # m
    ]
    out_shape = [
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
        jax.ShapeDtypeStruct(shape=cache_kv.shape, dtype=cache_kv.dtype),
        jax.ShapeDtypeStruct(shape=(q.shape[0], num_l_heads), dtype=jnp.float32),
        jax.ShapeDtypeStruct(shape=(q.shape[0], num_l_heads), dtype=jnp.float32),
    ]
    input_output_aliases = {
        len(scalar_prefetches) + 4: 0,  # Alias in_output with out
        len(scalar_prefetches) + 3: 1,  # Alias cache_kv with updated_cache_kv
        len(scalar_prefetches) + 5: 2,  # Alias in_l with out_l
        len(scalar_prefetches) + 6: 3,  # Alias in_m with out_m
    }

    # Double buffered VMEM allocations
    bkv_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz * bkv_sz, slots_per_token * kv_packing, lkv_dim),
        jnp.uint8,
    )
    q_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz, num_q_heads, head_dim),
        q.dtype,
    )
    o_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz, num_q_heads, head_dim),
        q.dtype,
    )
    l_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz, num_l_heads),
        jnp.float32,
    )
    m_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz, num_l_heads),
        jnp.float32,
    )
    new_kv_batch_x2_vmem = pltpu.VMEM(
        (2, seq_batch_sz) + new_kv.shape[1:],
        new_kv.dtype,
    )
    sems = pltpu.SemaphoreType.DMA((2, 7))

    scratch_shapes = [
        bkv_batch_x2_vmem,
        q_batch_x2_vmem,
        o_batch_x2_vmem,
        l_batch_x2_vmem,
        m_batch_x2_vmem,
        new_kv_batch_x2_vmem,
        sems,
    ]

    scope_name = f"SWA-d-bq_1-bkvp_{bkv_p}"
    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _mla_batched_decode_kernel,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                logical_page_size=logical_page_size,
                unnormalized_output=unnormalized_output,
                bkv_p=bkv_p,
                seq_batch_sz=seq_batch_sz,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=out_shape,
            input_output_aliases=input_output_aliases,
            name=scope_name,
        )
    )
    return kernel(
        *scalar_prefetches,
        attention_sinks,
        q,
        new_kv,
        cache_kv,
        in_output,
        in_l,
        in_m,
    )


def _mla_batched_verify_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    start_end_seq_idx_ref,
    attention_sinks_ref,
    q_hbm_ref,
    new_kv_hbm_ref,
    cache_kv_hbm_ref,
    o_hbm_ref,
    updated_cache_kv_hbm_ref,
    l_hbm_ref,
    m_hbm_ref,
    bkv_batch_x2_vmem,
    q_batch_x2_vmem,
    o_batch_x2_vmem,
    l_batch_x2_vmem,
    m_batch_x2_vmem,
    new_kv_batch_x2_vmem,
    sems,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    unnormalized_output: bool,
    bkv_p: int,
    seq_batch_sz: int,
    static_q_len: int,
    ced_decoder: bool,
    skip_kv_update: bool,
    non_causal_block: bool = False,
):
    def async_copy(src, dst, sem, wait=False):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    _, num_q_heads, head_dim = q_hbm_ref.shape
    cache_kv_hbm_ref = csa_cache_layout.as_u8_slabs(cache_kv_hbm_ref)
    updated_cache_kv_hbm_ref = csa_cache_layout.as_u8_slabs(updated_cache_kv_hbm_ref)
    total_num_pages, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        cache_kv_hbm_ref.shape
    )
    q_dtype = q_hbm_ref.dtype
    token_bytes = head_dim * get_dtype_bitwidth(q_dtype) // 8
    slot_bytes = kv_packing * lkv_dim
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token

    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    bkv_sz = bkv_p * logical_page_size
    tok_per_batch = seq_batch_sz * static_q_len

    reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
        total_num_pages * phys_tokens_per_page,
        slots_per_token * kv_packing,
        lkv_dim,
    )
    reshaped_updated_cache_hbm_ref = updated_cache_kv_hbm_ref.reshape(
        total_num_pages * phys_tokens_per_page,
        slots_per_token * kv_packing,
        lkv_dim,
    )

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    num_seqs = end_seq_idx - start_seq_idx
    num_batches = cdiv(num_seqs, seq_batch_sz)

    def get_cache_span(kv_len):
        start_offset = jnp.maximum(
            kv_len - static_q_len - sliding_window + 1, jnp.int32(0)
        )
        kv_left = jnp.maximum(kv_len - start_offset, jnp.int32(0))
        kv_left_frm_cache = jnp.maximum(kv_left - static_q_len, jnp.int32(0))
        return start_offset, jnp.minimum(kv_left_frm_cache, bkv_sz - static_q_len)

    def _fetch_bq(batch_idx, sem_idx, wait=False):
        if not wait:
            wait_update_kv_cache(batch_idx - 2, sem_idx)
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        q_start_seq = jnp.minimum(s_start, max_num_seqs - 1)
        q_token_idx = jnp.where(s_start < end_seq_idx, cu_q_lens_ref[q_start_seq], 0)
        n_tok = jnp.clip((end_seq_idx - s_start) * static_q_len, 0, tok_per_batch)
        async_copy(
            q_hbm_ref.at[pl.ds(q_token_idx, n_tok)],
            q_batch_x2_vmem.at[sem_idx, pl.ds(0, n_tok)],
            sems.at[sem_idx, 0],
            wait=wait,
        )
        async_copy(
            new_kv_hbm_ref.at[pl.ds(q_token_idx, n_tok)],
            new_kv_batch_x2_vmem.at[sem_idx, pl.ds(0, n_tok)],
            sems.at[sem_idx, 5],
            wait=wait,
        )

    def _fetch_bkv(batch_idx, sem_idx, wait=False):
        if ced_decoder:
            return
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        sem_kv = sems.at[sem_idx, 4]
        if wait:
            total_rows = jnp.int32(0)
            for s in range(seq_batch_sz):
                seq_idx = s_start + s
                seq_idx_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
                kv_len = jnp.where(
                    seq_idx < end_seq_idx, kv_lens_ref[seq_idx_clamped], 0
                )
                _, bkv_sz_frm_cache = get_cache_span(kv_len)
                total_rows = total_rows + jnp.where(
                    seq_idx < end_seq_idx, bkv_sz_frm_cache, 0
                )
            dst_all = bkv_batch_x2_vmem.at[sem_idx, pl.ds(0, total_rows)]
            async_copy(dst_all, dst_all, sem_kv, wait=True)
            return

        for s in range(seq_batch_sz):
            seq_idx = s_start + s
            seq_idx_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
            kv_len = jnp.where(
                seq_idx < end_seq_idx, kv_lens_ref[seq_idx_clamped], 0
            )
            start_offset, bkv_sz_frm_cache = get_cache_span(kv_len)
            kv_p_start = start_offset // logical_page_size
            page_off = start_offset - kv_p_start * logical_page_size
            page_indices_offset = jnp.minimum(
                seq_idx_clamped * pages_per_seq + kv_p_start,
                num_page_indices - 1,
            )
            bkv_s_base = s * bkv_sz
            vmem_off = 0
            for i in range(bkv_p + 1):
                avail = logical_page_size - page_off
                chunk_sz = jnp.where(
                    seq_idx < end_seq_idx,
                    jnp.clip(bkv_sz_frm_cache - vmem_off, jnp.int32(0), avail),
                    0,
                )
                page_idx = page_indices_ref[
                    jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                ]
                async_copy(
                    reshaped_cache_hbm_ref.at[
                        pl.ds(page_idx * phys_tokens_per_page + page_off, chunk_sz)
                    ],
                    bkv_batch_x2_vmem.at[
                        sem_idx, pl.ds(bkv_s_base + vmem_off, chunk_sz)
                    ],
                    sem_kv,
                    wait=False,
                )
                page_off = 0
                vmem_off += chunk_sz

    num_l_heads = l_batch_x2_vmem.shape[2]
    q_idx_vec = lax.broadcasted_iota(
        jnp.int32, (static_q_len * num_q_heads, 1), 0
    ) // num_q_heads

    def compute_batch(batch_idx, sem_idx):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        if not unnormalized_output:
            attention_sinks = attention_sinks_ref[...]
        new_kv_u8_vmem = new_kv_batch_x2_vmem.at[sem_idx].bitcast(jnp.uint8).reshape(
            tok_per_batch, slots_per_token * kv_packing, lkv_dim
        )

        if ced_decoder:
            q_all = q_batch_x2_vmem[sem_idx, ...].reshape(
                seq_batch_sz, static_q_len * num_q_heads, head_dim
            )
            kv_all = pltpu.bitcast(new_kv_u8_vmem[...], jnp.bfloat16).reshape(
                seq_batch_sz, static_q_len, head_dim
            )
            kv_pad = jnp.pad(kv_all, ((0, 0), (0, 128 - static_q_len), (0, 0)))
            k_iota_128 = lax.broadcasted_iota(jnp.int32, (1, 128), 1)
            if non_causal_block:
                keep_mask = jnp.broadcast_to(
                    k_iota_128 < static_q_len, (static_q_len * num_q_heads, 128)
                )
            else:
                keep_mask = (k_iota_128 < static_q_len) & (q_idx_vec >= k_iota_128)
            s_all = (
                jnp.einsum(
                    "snd,smd->snm",
                    q_all,
                    kv_pad,
                    preferred_element_type=jnp.float32,
                )
                * sm_scale
            )
            s_all = jnp.where(keep_mask[None, :, :], s_all, jnp.finfo(jnp.float32).min)
            m_curr = jnp.max(s_all, axis=2, keepdims=True)
            p_all = jnp.exp(s_all - m_curr)
            pv_all = jnp.einsum(
                "snm,smd->snd", p_all, kv_pad, preferred_element_type=jnp.float32
            )
            l_curr = jnp.sum(p_all, axis=2)
            m_flat = m_curr[:, :, 0]
            if unnormalized_output:
                out_all = pv_all.astype(q_dtype)
            else:
                sinks_t = jnp.tile(attention_sinks, static_q_len)[None, :]
                l_tot = (l_curr + jnp.exp(sinks_t - m_flat))[:, :, None]
                out_all = (pv_all * pl.reciprocal(l_tot, approx=True)).astype(q_dtype)
            o_batch_x2_vmem[sem_idx, ...] = out_all.reshape(
                tok_per_batch, num_q_heads, head_dim
            )
            if unnormalized_output:
                l_2d = l_curr.reshape(tok_per_batch, num_q_heads)
                m_2d = m_flat.reshape(tok_per_batch, num_q_heads)
                if num_q_heads < num_l_heads:
                    pad_cfg = ((0, 0), (0, num_l_heads - num_q_heads))
                    l_2d = jnp.pad(l_2d, pad_cfg)
                    m_2d = jnp.pad(m_2d, pad_cfg)
                l_batch_x2_vmem[sem_idx, ...] = l_2d
                m_batch_x2_vmem[sem_idx, ...] = m_2d
            return

        m_vec_1d = lax.broadcasted_iota(jnp.int32, (1, bkv_sz), 1)
        for s in range(seq_batch_sz):
            seq_idx = s_start + s
            kv_len = jnp.where(
                seq_idx < end_seq_idx,
                kv_lens_ref[jnp.minimum(seq_idx, max_num_seqs - 1)],
                0,
            )
            start_offset, bkv_sz_u = get_cache_span(kv_len)
            for t in range(static_q_len):
                bkv_batch_x2_vmem[
                    sem_idx, s * bkv_sz + bkv_sz_u + t, ...
                ] = new_kv_u8_vmem[s * static_q_len + t, ...]
            bkv_s = pltpu.bitcast(
                bkv_batch_x2_vmem[sem_idx, s * bkv_sz : (s + 1) * bkv_sz, ...],
                jnp.bfloat16,
            ).reshape(bkv_sz, head_dim)
            k_span = start_offset + m_vec_1d
            q_span = (kv_len - static_q_len) + q_idx_vec
            keep = (q_span - k_span).astype(jnp.uint32) < jnp.uint32(sliding_window)
            if non_causal_block:
                keep = keep | ((k_span >= kv_len - static_q_len) & (k_span < kv_len))
            valid_kv = m_vec_1d.T < (bkv_sz_u + static_q_len)
            bkv_w = jnp.where(valid_kv, bkv_s, 0)
            q_s = q_batch_x2_vmem[
                sem_idx, s * static_q_len : (s + 1) * static_q_len, ...
            ].reshape(static_q_len * num_q_heads, head_dim)
            s_mat = (
                jnp.einsum("nd,md->nm", q_s, bkv_w, preferred_element_type=jnp.float32)
                * sm_scale
            )
            s_mat = jnp.where(keep, s_mat, jnp.finfo(jnp.float32).min)
            m_curr = jnp.max(s_mat, axis=1, keepdims=True)
            p_mat = jnp.exp(s_mat - m_curr)
            pv_s = jnp.einsum(
                "nm,md->nd", p_mat, bkv_w, preferred_element_type=jnp.float32
            )
            l_curr = jnp.sum(p_mat, axis=1)
            m_1d = m_curr[:, 0]
            if unnormalized_output:
                out_s = pv_s.astype(q_dtype)
            else:
                sinks_t = jnp.tile(attention_sinks, static_q_len)
                l_tot = (l_curr + jnp.exp(sinks_t - m_1d))[:, None]
                out_s = (pv_s * pl.reciprocal(l_tot, approx=True)).astype(q_dtype)
            o_batch_x2_vmem[
                sem_idx, s * static_q_len : (s + 1) * static_q_len, ...
            ] = out_s.reshape(static_q_len, num_q_heads, head_dim)
            if unnormalized_output:
                l_2d = l_curr.reshape(static_q_len, num_q_heads)
                m_2d = m_1d.reshape(static_q_len, num_q_heads)
                if num_q_heads < num_l_heads:
                    pad_cfg = ((0, 0), (0, num_l_heads - num_q_heads))
                    l_2d = jnp.pad(l_2d, pad_cfg)
                    m_2d = jnp.pad(m_2d, pad_cfg)
                l_batch_x2_vmem[
                    sem_idx, s * static_q_len : (s + 1) * static_q_len, ...
                ] = l_2d
                m_batch_x2_vmem[
                    sem_idx, s * static_q_len : (s + 1) * static_q_len, ...
                ] = m_2d

    def _send_bo(batch_idx, sem_idx, wait=False):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        q_start_seq = jnp.minimum(s_start, max_num_seqs - 1)
        q_token_idx = jnp.where(s_start < end_seq_idx, cu_q_lens_ref[q_start_seq], 0)
        n_tok = jnp.clip((end_seq_idx - s_start) * static_q_len, 0, tok_per_batch)
        async_copy(
            o_batch_x2_vmem.at[sem_idx, pl.ds(0, n_tok)],
            o_hbm_ref.at[pl.ds(q_token_idx, n_tok)],
            sems.at[sem_idx, 1],
            wait=wait,
        )
        if unnormalized_output:
            async_copy(
                l_batch_x2_vmem.at[sem_idx, pl.ds(0, n_tok)],
                l_hbm_ref.at[pl.ds(q_token_idx, n_tok)],
                sems.at[sem_idx, 2],
                wait=wait,
            )
            async_copy(
                m_batch_x2_vmem.at[sem_idx, pl.ds(0, n_tok)],
                m_hbm_ref.at[pl.ds(q_token_idx, n_tok)],
                sems.at[sem_idx, 3],
                wait=wait,
            )

    def _update_kv_cache(batch_idx, sem_idx, wait=False):
        if skip_kv_update:
            return
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        sem_upd = sems.at[sem_idx, 6]
        new_kv_u8_vmem_ref = new_kv_batch_x2_vmem.at[sem_idx].bitcast(
            jnp.uint8
        ).reshape(tok_per_batch, slots_per_token * kv_packing, lkv_dim)
        if wait:
            n_tok = jnp.clip((end_seq_idx - s_start) * static_q_len, 0, tok_per_batch)
            dst_upd = new_kv_u8_vmem_ref.at[pl.ds(0, n_tok)]
            async_copy(dst_upd, dst_upd, sem_upd, wait=True)
            return

        for s in range(seq_batch_sz):
            seq_idx = s_start + s
            seq_idx_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
            kv_len = jnp.where(
                seq_idx < end_seq_idx, kv_lens_ref[seq_idx_clamped], 0
            )
            upd_offset = jnp.maximum(kv_len - static_q_len, 0)
            upd_sz = jnp.where(seq_idx < end_seq_idx, static_q_len, 0)
            kv_p_start = upd_offset // logical_page_size
            start_in_page = upd_offset - kv_p_start * logical_page_size
            page_indices_off = seq_idx_clamped * pages_per_seq + kv_p_start
            for i in range(2):
                in_p = start_in_page if i == 0 else jnp.int32(0)
                v_delta = (
                    jnp.int32(0) if i == 0 else (logical_page_size - start_in_page)
                )
                avail = (
                    (logical_page_size - start_in_page)
                    if i == 0
                    else jnp.int32(logical_page_size)
                )
                sz = jnp.clip(upd_sz - v_delta, 0, avail)
                page_idx = page_indices_ref[
                    jnp.minimum(page_indices_off + i, num_page_indices - 1)
                ]
                async_copy(
                    new_kv_u8_vmem_ref.at[pl.ds(s * static_q_len + v_delta, sz)],
                    reshaped_updated_cache_hbm_ref.at[
                        pl.ds(page_idx * phys_tokens_per_page + in_p, sz)
                    ],
                    sem_upd,
                    wait=False,
                )

    def wait_update_kv_cache(batch_idx, sem_idx):
        @pl.when(batch_idx >= 0)
        def _():
            _update_kv_cache(batch_idx, sem_idx, wait=True)

    def wait_send_bo(batch_idx, sem_idx):
        @pl.when(batch_idx >= 0)
        def _():
            _send_bo(batch_idx, sem_idx, wait=True)

    @pl.when(num_batches > 0)
    def _():
        _fetch_bq(0, 0, wait=False)
        _fetch_bkv(0, 0, wait=False)

    def loop_body(i, _):
        sem_idx = i % 2
        next_sem_idx = (i + 1) % 2

        @pl.when(i + 1 < num_batches)
        def _():
            _fetch_bq(i + 1, next_sem_idx, wait=False)
            _fetch_bkv(i + 1, next_sem_idx, wait=False)

        _fetch_bq(i, sem_idx, wait=True)
        _fetch_bkv(i, sem_idx, wait=True)
        wait_send_bo(i - 2, sem_idx)
        compute_batch(i, sem_idx)
        _send_bo(i, sem_idx, wait=False)
        _update_kv_cache(i, sem_idx, wait=False)
        return ()

    lax.fori_loop(0, num_batches, loop_body, ())

    @pl.when(num_batches > 0)
    def _():
        last_i = num_batches - 1
        wait_update_kv_cache(last_i, last_i % 2)
        wait_send_bo(last_i, last_i % 2)

        @pl.when(num_batches > 1)
        def _():
            wait_send_bo(last_i - 1, (last_i - 1) % 2)
            wait_update_kv_cache(last_i - 1, (last_i - 1) % 2)


def run_mla_batched_verify_kernel(
    q: jax.Array,
    new_kv: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    start_seq_idx: jax.Array,
    end_seq_idx: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    vmem_limit_bytes: int,
    unnormalized_output: bool,
    static_q_len: int,
    ced_decoder: bool,
    skip_kv_update: bool,
    non_causal_block: bool = False,
):
    num_tokens, num_q_heads, head_dim = q.shape
    max_seqs_cap = max(1, num_tokens // static_q_len)
    seq_batch_sz = min(max_seqs_cap, 16 if num_q_heads <= 16 else 8)
    bkv_p = (
        1
        if ced_decoder
        else max(2, cdiv(sliding_window + static_q_len - 1, logical_page_size))
    )
    bkv_sz = bkv_p * logical_page_size
    tok_b = seq_batch_sz * static_q_len
    num_l_heads = align_to(num_q_heads, 128)
    _, kv_packing, lkv_dim, slots_per_token, _, _ = get_physical_layout(
        csa_cache_layout.u8_slab_shape(cache_kv.shape, cache_kv.dtype),
        q.dtype,
        head_dim,
        kv_lens.shape[0],
        page_indices.shape[0],
    )
    bkv_vmem = pltpu.VMEM(
        (2, 1 if ced_decoder else seq_batch_sz * bkv_sz, slots_per_token * kv_packing, lkv_dim),
        jnp.uint8,
    )
    q_vmem = pltpu.VMEM((2, tok_b, num_q_heads, head_dim), q.dtype)
    lm_vmem = pltpu.VMEM((2, tok_b, num_l_heads), jnp.float32)
    nkv_vmem = pltpu.VMEM((2, tok_b) + new_kv.shape[1:], new_kv.dtype)
    sems = pltpu.SemaphoreType.DMA((2, 7))
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scope_name = (
        f"SWA-v-bq_{static_q_len}-bkvp_{bkv_p}"
        f"{'-ced' if ced_decoder else ''}"
        f"{'-ncb' if non_causal_block else ''}"
    )
    return jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _mla_batched_verify_kernel,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                logical_page_size=logical_page_size,
                unnormalized_output=unnormalized_output,
                bkv_p=bkv_p,
                seq_batch_sz=seq_batch_sz,
                static_q_len=static_q_len,
                ced_decoder=ced_decoder,
                skip_kv_update=skip_kv_update,
                non_causal_block=non_causal_block,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM), hbm, hbm, hbm],
                out_specs=[hbm, hbm, hbm, hbm],
                grid=(1,),
                scratch_shapes=[
                    bkv_vmem,
                    q_vmem,
                    q_vmem,
                    lm_vmem,
                    lm_vmem,
                    nkv_vmem,
                    sems,
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=[
                jax.ShapeDtypeStruct(q.shape, q.dtype),
                jax.ShapeDtypeStruct(cache_kv.shape, cache_kv.dtype),
                jax.ShapeDtypeStruct((num_tokens, num_l_heads), jnp.float32),
                jax.ShapeDtypeStruct((num_tokens, num_l_heads), jnp.float32),
            ],
            input_output_aliases={7: 1},
            name=scope_name,
        )
    )(
        kv_lens,
        page_indices,
        cu_q_lens,
        jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
        attention_sinks,
        q,
        new_kv,
        cache_kv,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "logical_page_size",
        "unnormalized_output",
        "non_causal_block",
        "q_compute_block_size",
        "decode_seq_batch_size",
        "ced_decoder",
        "keep_lm_aligned",
        "skip_kv_update",
        "verify_q_len",
    ),
    donate_argnames=("cache_kv",),
)
def mla_sliding_window_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    new_kv: jax.Array,  # [max_num_tokens, actual_head_dim]
    # [total_num_pages, physical_page_size_per_kv_packing, 4, 128] uint8
    cache_kv: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    *,
    sm_scale: float = 1.0,
    sliding_window: int,
    logical_page_size: int,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # If passsed in as int, all cases are the same.
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    q_compute_block_size: int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    unnormalized_output: bool = False,
    # Draft (DSpark/DFlash) block mode: the last q_len positions of each
    # sequence attend bidirectionally within themselves (window still bounds
    # context keys). Requires q_len <= sliding_window.
    non_causal_block: bool = False,
    decode_seq_batch_size: int = 32,
    ced_decoder: bool = False,
    keep_lm_aligned: bool = False,
    skip_kv_update: bool = False,
    verify_q_len: int = 0,
) -> tuple[
    jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    jax.Array,  # [total_num_pages, physical_page_size_per_kv_packing, 4, 128] uint8
    jax.Array,  # [max_num_tokens, actual_num_q_heads]
    jax.Array,  # [max_num_tokens, actual_num_q_heads]
]:
    """MLA Ragged paged attention that supports mixed prefill and decode."""
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError(
            "num_kv_pages_per_block and num_queries_per_block must be specified."
        )
    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block

    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
    else:
        num_queries_per_blocks = num_queries_per_block

    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    if q.shape[1] != actual_num_q_heads:
        attention_sinks = jnp.pad(
            attention_sinks,
            (0, q.shape[1] - actual_num_q_heads),
            constant_values=jnp.finfo(attention_sinks.dtype).min,
        )
    assert new_kv.dtype == jnp.bfloat16
    assert cache_kv.dtype in (jnp.uint8, jnp.int32)
    head_dim = q.shape[-1]
    _, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        csa_cache_layout.u8_slab_shape(cache_kv.shape, cache_kv.dtype)
    )

    slot_bytes = kv_packing * lkv_dim
    token_bytes = head_dim * get_dtype_bitwidth(new_kv.dtype) // 8
    assert token_bytes % slot_bytes == 0
    slots_per_token = token_bytes // slot_bytes
    phys_tokens_per_page = physical_page_size_per_kv_packing // slots_per_token

    if (
        kv_packing == csa_cache_layout.SLAB_ROWS
        and lkv_dim == csa_cache_layout.ROW_WORDS
    ):
        u16 = lax.bitcast_convert_type(
            new_kv.reshape(new_kv.shape[0] * slots_per_token, 2 * lkv_dim),
            jnp.uint16,
        )
        lo = u16[:, :lkv_dim].astype(jnp.uint32)
        hi = u16[:, lkv_dim:].astype(jnp.uint32)
        new_kv = lax.bitcast_convert_type(
            lo | (hi << 16), jnp.int32
        ).reshape(new_kv.shape[0], slots_per_token, lkv_dim)
    else:
        new_kv = prepare_kv_inputs(new_kv)  # [max_num_tokens, token_bytes] uint8
        new_kv = new_kv.reshape(new_kv.shape[0], slots_per_token * kv_packing, lkv_dim)
    assert logical_page_size <= phys_tokens_per_page

    _, num_q_heads, _ = q.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0

    def run_mla_kernel(
        q: jax.Array,
        new_kv: jax.Array,
        cache_kv: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        start_seq_idx: jax.Array,
        end_seq_idx: jax.Array,
        in_output: jax.Array,
        in_l: jax.Array,
        in_m: jax.Array,
        attention_sinks: jax.Array,
        static_q_len: int | None,
        unnormalized_output: bool,
        num_kv_pages_per_block: int,
        num_queries_per_block: int,
        case: MlaCase = MlaCase.MIXED,
    ):
        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz = bkv_p * logical_page_size
        grid = (end_seq_idx - start_seq_idx,)

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # in_output
            pl.BlockSpec(memory_space=pltpu.HBM),  # in_l
            pl.BlockSpec(memory_space=pltpu.HBM),  # in_m
        ]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # o
            pl.BlockSpec(memory_space=pltpu.HBM),  # updated_cache_kv
            pl.BlockSpec(memory_space=pltpu.HBM),  # l
            pl.BlockSpec(memory_space=pltpu.HBM),  # m
        ]

        bkv_double_buf = pltpu.VMEM(
            (2, bkv_sz, slots_per_token * kv_packing, lkv_dim),
            jnp.uint8,
        )

        bq_double_bufq = pltpu.VMEM(
            (2, bq_sz, num_q_heads, head_dim),
            q.dtype,
        )

        bo_double_buf = bq_double_bufq

        num_l_heads = align_to(num_q_heads, 128)
        bl_double_buf = pltpu.VMEM(
            (2, bq_sz, num_l_heads),
            jnp.float32,
        )
        bm_double_buf = bl_double_buf

        single_bkv = cdiv(bq_sz + sliding_window - 1, bkv_sz) == 1
        scratch_rows = 128 if single_bkv else bq_sz * num_q_heads
        l_scratch = pltpu.VMEM(
            (scratch_rows,),
            jnp.float32,
        )
        m_scratch = l_scratch

        acc_scratch = pltpu.VMEM(
            (1 if single_bkv else bq_sz * num_q_heads, head_dim),
            jnp.float32,
        )

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bo_double_buf,  # Double buffering for output block.
            bl_double_buf,  # Double buffering for l output.
            bm_double_buf,  # Double buffering for m output.
            # Semaphores for double buffering of bkv, bq, bo, bkv_update, l, m.
            pltpu.SemaphoreType.DMA((6, 2)),
            # Intermediate buffers per kv head for flash attention.
            l_scratch,
            m_scratch,
            acc_scratch,
        ]

        scalar_prefetches = (
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            # (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
            jnp.zeros((3,), jnp.int32),
            # (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
            jnp.full((4,), -1, jnp.int32),
            # (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset,
            #     bkv_sem_0_sz, bkv_sem_1_sz)
            jnp.full((6,), -1, jnp.int32),
        )

        scope_name = (
            f"SWA-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
            f"{'-ncb' if non_causal_block else ''}"
            f"{'-ced' if ced_decoder else ''}"
        )
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _mla_sliding_window_ragged_paged_attention_kernel,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    static_q_len=static_q_len,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    logical_page_size=logical_page_size,
                    unnormalized_output=unnormalized_output,
                    non_causal_block=non_causal_block,
                    q_compute_block_size=q_compute_block_size,
                    ced_decoder=ced_decoder,
                    skip_kv_update=skip_kv_update,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=[
                    jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                    jax.ShapeDtypeStruct(shape=cache_kv.shape, dtype=cache_kv.dtype),
                    jax.ShapeDtypeStruct(
                        shape=(q.shape[0], num_l_heads), dtype=jnp.float32
                    ),
                    jax.ShapeDtypeStruct(
                        shape=(q.shape[0], num_l_heads), dtype=jnp.float32
                    ),
                ],
                input_output_aliases={
                    11: 0,  # Alias output activation with in_output
                    10: 1,  # Aliasing cache_kv with updated_cache_kv
                    12: 2,  # Alias l with in_l
                    13: 3,  # Alias m with in_m
                },
                name=scope_name,
            )
        )
        eff_chunk_sz = (
            (q_compute_block_size if bq_sz % q_compute_block_size == 0 else bq_sz)
            if q_compute_block_size is not None
            else bq_sz
        )
        sinks_arg = (
            jnp.broadcast_to(
                jnp.tile(attention_sinks, eff_chunk_sz)[:, None],
                (eff_chunk_sz * num_q_heads, 128),
            )
            if single_bkv
            else jnp.tile(attention_sinks, bq_sz)
        )
        return kernel(
            *scalar_prefetches,
            sinks_arg,
            q,
            new_kv,
            cache_kv,
            in_output,
            in_l,
            in_m,
        )

    num_l_heads = align_to(num_q_heads, 128)
    alloc_zero_lm = jnp.zeros if unnormalized_output else jnp.empty

    def _run_general_swa(kv_in):
        out_a = jnp.zeros_like(q)
        l_a = alloc_zero_lm((q.shape[0], num_l_heads), dtype=jnp.float32)
        m_a = alloc_zero_lm((q.shape[0], num_l_heads), dtype=jnp.float32)
        bkv_s = num_kv_pages_per_blocks[0] * logical_page_size
        if sliding_window <= bkv_s:
            out_a, kv_a, l_a, m_a = lax.cond(
                distribution[0] > 0,
                lambda a: run_mla_batched_decode_kernel(
                    q,
                    new_kv,
                    a[1],
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    num_kv_pages_per_block=num_kv_pages_per_blocks[0],
                    num_queries_per_block=num_queries_per_blocks[0],
                    start_seq_idx=jnp.array(0),
                    end_seq_idx=distribution[0],
                    in_output=a[0],
                    in_l=a[2],
                    in_m=a[3],
                    attention_sinks=attention_sinks,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    logical_page_size=logical_page_size,
                    vmem_limit_bytes=vmem_limit_bytes,
                    unnormalized_output=unnormalized_output,
                    seq_batch_sz=decode_seq_batch_size,
                ),
                lambda a: a,
                (out_a, kv_in, l_a, m_a),
            )
        else:
            out_a, kv_a, l_a, m_a = run_mla_kernel(
                q,
                new_kv,
                kv_in,
                kv_lens,
                page_indices,
                cu_q_lens,
                start_seq_idx=jnp.array(0),
                end_seq_idx=distribution[0],
                in_output=out_a,
                in_l=l_a,
                in_m=m_a,
                attention_sinks=attention_sinks,
                static_q_len=1,
                unnormalized_output=unnormalized_output,
                num_kv_pages_per_block=num_kv_pages_per_blocks[0],
                num_queries_per_block=num_queries_per_blocks[0],
                case=MlaCase.DECODE,
            )
        return lax.cond(
            distribution[2] > distribution[1],
            lambda a: run_mla_kernel(
                q,
                new_kv,
                a[1],
                kv_lens,
                page_indices,
                cu_q_lens,
                num_kv_pages_per_block=num_kv_pages_per_blocks[2],
                num_queries_per_block=num_queries_per_blocks[2],
                start_seq_idx=distribution[1],
                end_seq_idx=distribution[2],
                in_output=a[0],
                in_l=a[2],
                in_m=a[3],
                attention_sinks=attention_sinks,
                static_q_len=None,
                unnormalized_output=unnormalized_output,
                case=MlaCase.MIXED,
            ),
            lambda a: a,
            (out_a, kv_a, l_a, m_a),
        )

    eff_verify_q_len = verify_q_len
    if (
        eff_verify_q_len == 0
        and non_causal_block
        and chunk_prefill_size is None
        and q.shape[0] <= 1536
    ):
        for _cand in (5, 3, 4):
            if q.shape[0] % _cand == 0:
                eff_verify_q_len = _cand
                break

    if (
        2 <= eff_verify_q_len <= 8
        and chunk_prefill_size is None
        and q.shape[0] % eff_verify_q_len == 0
    ):
        seq_q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        active_seq = jnp.arange(max_num_seqs, dtype=jnp.int32) < distribution[2]
        is_uniform_verify = (
            (distribution[0] == 0)
            & (distribution[1] == 0)
            & (distribution[2] > 0)
            & (cu_q_lens[distribution[2]] == distribution[2] * eff_verify_q_len)
            & jnp.all(
                ~active_seq
                | ((seq_q_lens == eff_verify_q_len) & ((kv_lens >> 20) == 0))
            )
        )

        def _run_uniform_verify(kv_in):
            return run_mla_batched_verify_kernel(
                q,
                new_kv,
                kv_in,
                kv_lens,
                page_indices,
                cu_q_lens,
                jnp.array(0, jnp.int32),
                distribution[2],
                attention_sinks,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                logical_page_size=logical_page_size,
                vmem_limit_bytes=vmem_limit_bytes,
                unnormalized_output=unnormalized_output,
                static_q_len=eff_verify_q_len,
                ced_decoder=False,
                skip_kv_update=skip_kv_update,
                non_causal_block=non_causal_block,
            )

        output, updated_kv, out_l, out_m = lax.cond(
            is_uniform_verify,
            _run_uniform_verify,
            _run_general_swa,
            cache_kv,
        )
        output = prepare_outputs(output, actual_num_q_heads, actual_head_dim)
        if not keep_lm_aligned:
            out_l = out_l[:, :actual_num_q_heads]
            out_m = out_m[:, :actual_num_q_heads]
        return output, updated_kv, out_l, out_m

    if chunk_prefill_size is None and q.shape[0] > 128:
        is_full_mixed = (
            (distribution[0] == 0)
            & (distribution[1] == 0)
            & (cu_q_lens[distribution[2]] == q.shape[0])
        )

        def _run_full_mixed(kv_in):
            return run_mla_kernel(
                q,
                new_kv,
                kv_in,
                kv_lens,
                page_indices,
                cu_q_lens,
                num_kv_pages_per_block=num_kv_pages_per_blocks[2],
                num_queries_per_block=num_queries_per_blocks[2],
                start_seq_idx=jnp.array(0, jnp.int32),
                end_seq_idx=distribution[2],
                in_output=jnp.empty_like(q),
                in_l=jnp.empty((q.shape[0], num_l_heads), dtype=jnp.float32),
                in_m=jnp.empty((q.shape[0], num_l_heads), dtype=jnp.float32),
                attention_sinks=attention_sinks,
                static_q_len=None,
                unnormalized_output=unnormalized_output,
                case=MlaCase.MIXED,
            )

        output, updated_kv, out_l, out_m = lax.cond(
            is_full_mixed,
            _run_full_mixed,
            _run_general_swa,
            cache_kv,
        )
        output = prepare_outputs(output, actual_num_q_heads, actual_head_dim)
        if not keep_lm_aligned:
            out_l = out_l[:, :actual_num_q_heads]
            out_m = out_m[:, :actual_num_q_heads]
        return output, updated_kv, out_l, out_m

    l = alloc_zero_lm((q.shape[0], num_l_heads), dtype=jnp.float32)  # noqa: E741
    m = alloc_zero_lm((q.shape[0], num_l_heads), dtype=jnp.float32)
    in_output = jnp.zeros_like(q)

    bkv_sz = num_kv_pages_per_blocks[0] * logical_page_size
    if sliding_window <= bkv_sz:
        def _run_dec_batch(args):
            return run_mla_batched_decode_kernel(
                q,
                new_kv,
                args[1],
                kv_lens,
                page_indices,
                cu_q_lens,
                num_kv_pages_per_block=num_kv_pages_per_blocks[0],
                num_queries_per_block=num_queries_per_blocks[0],
                start_seq_idx=jnp.array(0),
                end_seq_idx=distribution[0],
                in_output=args[0],
                in_l=args[2],
                in_m=args[3],
                attention_sinks=attention_sinks,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                logical_page_size=logical_page_size,
                vmem_limit_bytes=vmem_limit_bytes,
                unnormalized_output=unnormalized_output,
                seq_batch_sz=decode_seq_batch_size,
            )

        if q.shape[0] > 128:
            output, updated_kv, out_l, out_m = lax.cond(
                distribution[0] > 0,
                _run_dec_batch,
                lambda args: args,
                (in_output, cache_kv, l, m),
            )
        else:
            output, updated_kv, out_l, out_m = _run_dec_batch(
                (in_output, cache_kv, l, m)
            )
    else:
        output, updated_kv, out_l, out_m = run_mla_kernel(
            q,
            new_kv,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q_lens,
            start_seq_idx=jnp.array(0),
            end_seq_idx=distribution[0],
            in_output=in_output,
            in_l=l,
            in_m=m,
            attention_sinks=attention_sinks,
            static_q_len=1,
            unnormalized_output=unnormalized_output,
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            case=MlaCase.DECODE,
        )

    if chunk_prefill_size is not None:
        # Handle prefill where the query length is fixed per sequence.
        def _run_prefill_batch(args):
            return run_mla_kernel(
                q,
                new_kv,
                args[1],
                kv_lens,
                page_indices,
                cu_q_lens,
                num_kv_pages_per_block=num_kv_pages_per_blocks[1],
                num_queries_per_block=num_queries_per_blocks[1],
                start_seq_idx=distribution[0],
                end_seq_idx=distribution[1],
                in_output=args[0],
                in_l=args[2],
                in_m=args[3],
                attention_sinks=attention_sinks,
                static_q_len=chunk_prefill_size,
                unnormalized_output=unnormalized_output,
                case=MlaCase.PREFILL,
            )

        output, updated_kv, out_l, out_m = lax.cond(
            distribution[1] > distribution[0],
            _run_prefill_batch,
            lambda args: args,
            (output, updated_kv, out_l, out_m),
        )

    # Handle mixed case where the query length per sequence is variable.
    def _run_mixed_batch(args):
        return run_mla_kernel(
            q,
            new_kv,
            args[1],
            kv_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[2],
            num_queries_per_block=num_queries_per_blocks[2],
            start_seq_idx=distribution[1],
            end_seq_idx=distribution[2],
            in_output=args[0],
            in_l=args[2],
            in_m=args[3],
            attention_sinks=attention_sinks,
            static_q_len=None,
            unnormalized_output=unnormalized_output,
            case=MlaCase.MIXED,
        )

    output, updated_kv, out_l, out_m = lax.cond(
        distribution[2] > distribution[1],
        _run_mixed_batch,
        lambda args: args,
        (output, updated_kv, out_l, out_m),
    )
    output = prepare_outputs(
        output, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    if not keep_lm_aligned:
        out_l = out_l[:, :actual_num_q_heads]
        out_m = out_m[:, :actual_num_q_heads]

    return output, updated_kv, out_l, out_m
