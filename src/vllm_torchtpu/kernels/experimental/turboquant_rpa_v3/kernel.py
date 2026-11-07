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
"""Experimental Ragged Paged Attention v3 kernel for TurboQuant KV cache.

Adapted from `vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel` at commit
`132a64aee` (2026-10-09), sharing `RpaCase`, `prepare_inputs`, `prepare_outputs`,
`get_default_block_sizes`, and `has_bank_conflicts`. Kept in
`kernels/experimental/turboquant_rpa_v3` rather than branching inside `v3/kernel.py`
because the HBM/VMEM page contract changes from 5D `kv_dtype` pages to 4D
`uint8[num_pages, num_kv_heads, page_words, 128]` pages with an extra `tq_new_kv`
DMA/VMEM double-buffer and VREG callbacks (`tq_update_page_fn`, `tq_dequant_page_fn`)
inside the `RPAd` / `RPAm` loops.
"""

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    RpaCase,
    get_default_block_sizes,
    has_bank_conflicts,
    prepare_inputs,
    prepare_outputs,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import (
    align_to,
    cdiv,
    get_dtype_packing,
    get_tpu_version,
)


def _ragged_paged_attention_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    distribution_ref,
    sem_ids_ref,
    bo_ids_ref,
    bkv_update_ids_ref,
    q_hbm_ref,
    kv_hbm_ref,
    kv_cache_hbm_ref,
    *rest_args,
    has_tq_new_kv: bool = False,
    **kwargs,
):
    start_seq_idx, end_seq_idx = kwargs["case"].get_range(distribution_ref)
    if not has_tq_new_kv:
        rest_args = (None, *rest_args)

    @pl.loop(start_seq_idx, end_seq_idx)
    def _(seq_idx):
        return _ragged_paged_attention_kernel_loop(
            seq_idx,
            kv_lens_ref,
            page_indices_ref,
            cu_q_lens_ref,
            distribution_ref,
            sem_ids_ref,
            bo_ids_ref,
            bkv_update_ids_ref,
            q_hbm_ref,
            kv_hbm_ref,
            kv_cache_hbm_ref,
            *rest_args,
            **kwargs,
        )


def _ragged_paged_attention_kernel_loop(
    seq_idx,
    # Scalar prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [4] (seq_0, seq_1, bo_idx_0, bo_idx_1)
    bkv_update_ids_ref,  # [6] (seq_0, seq_1, offset_0, offset_1, sz_0, sz_1)
    # HBM inputs
    # [actual_num_kv_heads, max_num_tokens, num_q_per_kv // q_packing, q_packing, hd]
    q_hbm_ref,
    kv_hbm_ref,  # [max_num_tokens, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache_hbm_ref,  # u8[total_num_pages, actual_num_kv_heads, page_words, 128]
    tq_new_kv_hbm_ref,  # u8[max_num_tokens, num_new_words, actual_num_kv_heads, 128]
    # HBM outputs
    o_hbm_ref,
    updated_kv_cache_hbm_ref,
    # VMEM scratch
    bkv_x2_ref,  # [2, raw_bkv_vmem_sz, bkv_stride, kv_packing, head_dim]
    bq_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_per_kv // q_packing, q_pack, hd]
    bo_x2_ref,  # [2, actual_num_kv_heads, bq_sz, num_q_per_kv // q_packing, q_pack, hd]
    sems,  # [4, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128]
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128]
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim]
    btq_x2_ref,  # u8[2, bkv_p, actual_num_kv_heads, page_words, 128]
    btq_new_x2_ref=None,  # u8[2, 1 | bkv_sz, num_new_words, actual_num_kv_heads, 128]
    *,
    skip_kv_update: bool = False,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float,
    bq_sz: int,
    bkv_sz: int,
    bq_csz: int,
    bkv_csz: int,
    case: RpaCase = RpaCase.MIXED,
    tq_page_size: int,
    tq_dequant_page_fn: Callable[..., Any],
    tq_update_page_fn: Callable[..., Any] | None = None,
):
    is_decode = case == RpaCase.DECODE
    out_dtype = acc_ref.dtype
    (
        actual_num_kv_heads,
        _,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    page_size = tq_page_size
    _, num_kv_heads_x2_per_kv_packing, kv_packing, _ = kv_hbm_ref.shape
    kv_dtype = kv_hbm_ref.dtype
    bkv_stride = bkv_x2_ref.shape[2]
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    bkv_p = bkv_sz // page_size
    span_int_ty = (
        jnp.int16
        if (
            q_packing != 1
            and get_tpu_version() >= 6
            and pages_per_seq * page_size <= jnp.iinfo(jnp.int16).max
        )
        else jnp.int32
    )
    start_seq_idx, end_seq_idx = case.get_range(distribution_ref)

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]
    kv_q_gap = kv_len - q_len
    cur_seq_start_bkv_idx = 0
    next_seq_start_bkv_idx = 0

    if sliding_window is not None:
        cur_seq_start_bkv_idx = jnp.maximum(kv_q_gap - sliding_window, 0) // bkv_sz
        next_seq_idx = jnp.minimum(seq_idx + 1, end_seq_idx - 1)
        next_q_start = cu_q_lens_ref[next_seq_idx]
        next_q_end = cu_q_lens_ref[next_seq_idx + 1]
        next_kv_q_gap = kv_lens_ref[next_seq_idx] - (next_q_end - next_q_start)
        next_seq_start_bkv_idx = (
            jnp.maximum(next_kv_q_gap - sliding_window, 0) // bkv_sz
        )

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        target_minor = align_to(shape[-1], src.shape[-1])
        return jnp.concatenate(
            [src for _ in range(target_minor // src.shape[-1])], axis=-1
        )[..., : shape[-1]]

    def flash_attention_step1_qk_softmax(
        q,  # [actual_bq_csz * num_q_heads_per_kv_head, head_dim]
        k,  # [bkv_csz, head_dim]
        l_slice_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        m_slice_ref,  # [actual_bq_csz * num_q_heads_per_kv_head, 128]
        *,
        processed_q_len,
        processed_kv_len,
        effective_kv_len,
        skip_tile_mask: bool = False,
        k_norm_row=None,
    ):
        s_f32 = jnp.matmul(q, k.T, preferred_element_type=jnp.float32)
        if k_norm_row is not None:
            s = (s_f32 * (k_norm_row * sm_scale)).astype(out_dtype)
        else:
            s = s_f32.astype(out_dtype) * sm_scale
        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)

        if not skip_tile_mask:
            processed_kv_len_int = processed_kv_len.astype(span_int_ty)
            effective_kv_len_int = effective_kv_len.astype(span_int_ty)
            k_span = processed_kv_len_int + lax.broadcasted_iota(
                span_int_ty, s.shape, 1
            )
            mask = k_span < effective_kv_len_int
            if not is_decode or sliding_window is not None:
                processed_q_len_int = processed_q_len.astype(span_int_ty)
                q_span = processed_q_len_int + (
                    lax.broadcasted_iota(jnp.int32, s.shape, 0)
                    // num_q_heads_per_kv_head
                ).astype(span_int_ty)
                if not is_decode:
                    mask = mask & (q_span >= k_span)
                if sliding_window is not None:
                    mask = mask & (q_span < k_span + sliding_window)
            s = jnp.where(mask, s, mask_value)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = m_slice_ref[...]
        m_curr = jnp.maximum(m_prev, s_rowmax)
        m_slice_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        p_rowsum = jnp.sum(p, axis=1, keepdims=True, dtype=out_dtype)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_slice_ref[...] = exp_m_diff * l_slice_ref[...] + p_rowsum
        return p, exp_m_diff

    def flash_attention_step2_pv(p, v, exp_m_diff, o_ref):
        pv = jnp.matmul(p, v, preferred_element_type=jnp.float32).astype(out_dtype)
        o_prev = o_ref[...]
        o_ref[...] = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv

    def _async_copy(src, dst, sem, wait: bool):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait: bool = False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]

        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        kv_left_frm_cache = (
            kv_left if skip_kv_update else jnp.maximum(kv_left - q_len, 0)
        )
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        has_new = btq_new_x2_ref is not None
        cached_bkv_tokens = (
            bkv_sz_frm_cache + bkv_sz_frm_new
            if (has_new and is_decode)
            else bkv_sz_frm_cache
        )
        new_off = jnp.int32(0) if is_decode else bkv_sz_frm_cache
        if not wait:
            wait_update_kv_cache(bkv_sem_idx)
            for i in range(bkv_p):
                has_page = jnp.where(cached_bkv_tokens > i * page_size, 1, 0)
                p_idx = page_indices_ref[
                    jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                ]
                _async_copy(
                    kv_cache_hbm_ref.at[pl.ds(p_idx, has_page)],
                    btq_x2_ref.at[bkv_sem_idx, pl.ds(i, has_page)],
                    sem,
                    wait=False,
                )
            new_start = q_end - kv_left_frm_new
            if not is_decode and has_new:
                _async_copy(
                    kv_hbm_ref.at[pl.ds(new_start, bkv_sz_frm_new)],
                    vmem_ref.at[pl.ds(new_off, bkv_sz_frm_new)],
                    sem,
                    wait=False,
                )
            if has_new:
                assert tq_new_kv_hbm_ref is not None
                _async_copy(
                    tq_new_kv_hbm_ref.at[pl.ds(new_start, bkv_sz_frm_new)],
                    btq_new_x2_ref.at[bkv_sem_idx, pl.ds(new_off, bkv_sz_frm_new)],
                    sem,
                    wait=False,
                )
        else:
            tq_dst = btq_x2_ref.at[
                bkv_sem_idx, pl.ds(0, cdiv(cached_bkv_tokens, page_size))
            ]
            _async_copy(src=tq_dst, dst=tq_dst, sem=sem, wait=True)
            if not is_decode and has_new:
                new_dst = vmem_ref.at[pl.ds(new_off, bkv_sz_frm_new)]
                _async_copy(src=new_dst, dst=new_dst, sem=sem, wait=True)
            if has_new:
                tq_new_dst = btq_new_x2_ref.at[
                    bkv_sem_idx, pl.ds(new_off, bkv_sz_frm_new)
                ]
                _async_copy(src=tq_new_dst, dst=tq_new_dst, sem=sem, wait=True)
        return (
            kv_len_start + bkv_sz_frm_cache,
            bkv_sz_frm_new if has_new else jnp.int32(0),
        )

    def _update_kv_cache(
        seq_idx, bkv_sem_idx, offset, update_sz, *, wait: bool = False
    ):
        if btq_new_x2_ref is None or tq_update_page_fn is None:
            return
        sem = sems.at[3, bkv_sem_idx]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        num_new_words = btq_new_x2_ref.shape[2]
        slot_iota = lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
        if is_decode:
            if not wait:
                valid_mask = slot_iota == ignore
                new_words = [
                    btq_new_x2_ref.at[bkv_sem_idx, 0, w][...]
                    for w in range(num_new_words)
                ]
                for h in range(actual_num_kv_heads):
                    old_p = btq_x2_ref.at[bkv_sem_idx, p_ignore, h][...]
                    nw_h = [w_arr[h : h + 1, :] for w_arr in new_words]
                    btq_x2_ref.at[bkv_sem_idx, p_ignore, h][...] = tq_update_page_fn(
                        old_p, nw_h, valid_mask, slot_idx=ignore
                    )
                _async_copy(
                    btq_x2_ref.at[bkv_sem_idx, pl.ds(p_ignore, 1)],
                    updated_kv_cache_hbm_ref.at[
                        pl.ds(page_indices_ref[page_indices_offset], 1)
                    ],
                    sem,
                    wait=False,
                )
            else:
                dst = updated_kv_cache_hbm_ref.at[pl.ds(0, 1)]
                _async_copy(src=dst, dst=dst, sem=sem, wait=True)
            return

        num_touched_pages = kv_p_end - kv_p_start
        if not wait:
            local_new_start = offset - bkv_id * bkv_sz
            local_new_end = local_new_start + update_sz

            def tq_update_body(i, _):
                assert btq_new_x2_ref is not None
                p_local = p_ignore + i
                tok_pos = p_local * page_size + slot_iota
                valid_mask = (tok_pos >= local_new_start) & (tok_pos < local_new_end)
                p_ds = pl.ds(p_local * page_size, page_size)
                for h in range(actual_num_kv_heads):
                    old_p = jnp.where(
                        (p_local * page_size) < local_new_start,
                        btq_x2_ref.at[bkv_sem_idx, p_local, h][...],
                        jnp.uint8(0),
                    )
                    nw_h = [
                        btq_new_x2_ref.at[bkv_sem_idx, p_ds, w][...][:, h, :]
                        for w in range(num_new_words)
                    ]
                    btq_x2_ref.at[bkv_sem_idx, p_local, h][...] = tq_update_page_fn(
                        old_p, nw_h, valid_mask
                    )
                _async_copy(
                    btq_x2_ref.at[bkv_sem_idx, pl.ds(p_local, 1)],
                    updated_kv_cache_hbm_ref.at[
                        pl.ds(page_indices_ref[page_indices_offset + i], 1)
                    ],
                    sem,
                    wait=False,
                )
                return None

            lax.fori_loop(0, num_touched_pages, tq_update_body, None, unroll=False)
        else:
            dst = updated_kv_cache_hbm_ref.at[pl.ds(0, num_touched_pages)]
            _async_copy(src=dst, dst=dst, sem=sem, wait=True)

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait: bool = False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)
        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            vmem_ref.at[:, pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait: bool = False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)
        _async_copy(
            vmem_ref.at[:, pl.ds(0, sz)],
            o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(start_seq_idx <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)

    def strided_load(ref, start, sz, step, *, dtype=None):
        rows, lanes = ref.shape
        folds = lanes // 128
        ref = ref.reshape(rows * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        vec = jnp.concat(
            [ref[pl.ds(start + i, sz // step, step)] for i in range(folds)], axis=1
        )
        return pltpu.bitcast(vec, dtype) if dtype is not None else vec

    def strided_store(ref, start, sz, step, val):
        rows, lanes = ref.shape
        folds = lanes // 128
        ref = ref.reshape(rows * folds, 128)
        start *= folds
        sz *= folds
        step *= folds
        for i in range(folds):
            ref[pl.ds(start + i, sz // step, step)] = val[:, i * 128 : (i + 1) * 128]

    def load_bq(bq_sem_idx, kv_head_idx, start, sz):
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx, kv_head_idx]
            .reshape(bq_sz * num_q_heads_per_kv_head_per_packing, head_dim)
        )
        start *= num_q_heads_per_kv_head_per_packing
        sz *= num_q_heads_per_kv_head_per_packing
        return strided_load(q_ref, start, sz, 1, dtype=q_dtype)

    def _load_bkv_raw(bkv_sem_idx, kv_head_idx, start, sz):
        start *= bkv_stride
        sz *= bkv_stride
        step = bkv_stride
        kv_ref = (
            bkv_x2_ref.bitcast(jnp.uint32)
            .at[bkv_sem_idx]
            .reshape(bkv_sz * step, head_dim)
        )
        if kv_packing == 1:
            start += kv_head_idx * 2
            k = strided_load(kv_ref, start, sz, step, dtype=kv_dtype)
            v = strided_load(kv_ref, start + 1, sz, step, dtype=kv_dtype)
            return k, v

        num_kv_per_load = kv_packing // 2
        offset = kv_head_idx // num_kv_per_load
        kv_idx_in_load = kv_head_idx % num_kv_per_load
        kv = strided_load(kv_ref, start + offset, sz, step)
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        k = kv >> (kv_idx_in_load * 2 * bitwidth)
        v = k >> bitwidth
        return (
            pltpu.bitcast(k.astype(repack_ty), kv_dtype),
            pltpu.bitcast(v.astype(repack_ty), kv_dtype),
        )

    def _load_bkv_dequant(
        bkv_sem_idx, kv_head_idx, start, sz, *, return_k_norm: bool = False
    ):
        start_page = start // page_size
        refs = [
            btq_x2_ref.at[bkv_sem_idx, start_page + p_off, kv_head_idx]
            for p_off in range(sz // page_size)
        ]
        return tq_dequant_page_fn(refs, kv_dtype, return_k_norm=return_k_norm)

    def _load_bkv_mixed(bkv_sem_idx, kv_head_idx, start, sz, bkv_sz_frm_cache):
        row_idx_128 = start + lax.broadcasted_iota(jnp.int32, (sz, 128), 0)
        from_cache = broadcast_minor(row_idx_128 < bkv_sz_frm_cache, (sz, head_dim))
        bk_new, bv_new = _load_bkv_raw(bkv_sem_idx, kv_head_idx, start, sz)
        bk_tq, bv_tq = _load_bkv_dequant(bkv_sem_idx, kv_head_idx, start, sz)
        return (
            jnp.where(from_cache, bk_tq, bk_new),
            jnp.where(from_cache, bv_tq, bv_new),
        )

    def process():
        actual_bq_sz = 1 if is_decode else bq_sz
        num_bq = 1 if is_decode else cdiv(q_len, actual_bq_sz)
        actual_bq_csz = min(bq_csz, actual_bq_sz)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx, *, num_bkv):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)

            next_bq_start_bkv_idx = 0
            if sliding_window is not None:
                next_bq_start_bkv_idx = (
                    jnp.maximum(
                        kv_q_gap + (bq_idx + 1) * actual_bq_sz - sliding_window, 0
                    )
                    // bkv_sz
                )
            next_bkv_idx = lax.select(is_last_bkv, next_bq_start_bkv_idx, next_bkv_idx)
            next_bkv_idx = lax.select(is_last_bq, next_seq_start_bkv_idx, next_bkv_idx)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        @pl.loop(0, num_bq, unroll=False)
        def compute_with_bq(bq_idx):
            l_ref[...] = jnp.full_like(l_ref, 0.0)
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            acc_ref[...] = jnp.full_like(acc_ref, 0.0)

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                seq_idx, bq_idx, bq_sem_idx
            )

            processed_q_len = kv_q_gap + bq_idx * actual_bq_sz
            start_bkv_idx = (
                0
                if sliding_window is None
                else jnp.maximum(processed_q_len - sliding_window, 0) // bkv_sz
            )
            effective_kv_len = (
                kv_len
                if is_decode
                else jnp.minimum(kv_len, processed_q_len + actual_bq_sz)
            )
            end_bkv_idx = cdiv(effective_kv_len, bkv_sz)

            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                _fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            @pl.loop(start_bkv_idx, end_bkv_idx, unroll=False)
            def compute_with_bkv(bkv_idx):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx, bq_idx, bkv_idx, bkv_sem_idx, num_bkv=end_bkv_idx
                )
                processed_kv_len = bkv_idx * bkv_sz

                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    _fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)

                @pl.when(bkv_idx == start_bkv_idx)
                def wait_cur_bq():
                    _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

                offset, update_sz = _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == num_bq - 1))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                effective_bkv_sz = jnp.maximum(
                    jnp.minimum(effective_kv_len - bkv_idx * bkv_sz, bkv_sz), 0
                )
                num_loops = cdiv(effective_bkv_sz, bkv_csz)
                preloaded_bq = (
                    [
                        load_bq(bq_sem_idx, kv_head_idx, 0, actual_bq_csz)
                        for kv_head_idx in range(actual_num_kv_heads)
                    ]
                    if is_decode
                    else None
                )

                def _mask_v_tile(v, *, cur_processed_kv_len):
                    v_span_128 = cur_processed_kv_len.astype(
                        span_int_ty
                    ) + lax.broadcasted_iota(span_int_ty, (v.shape[0], 128), 0)
                    v_valid = broadcast_minor(
                        v_span_128 < effective_kv_len.astype(span_int_ty), v.shape
                    )
                    return jnp.where(v_valid, v, 0.0)

                def _run_tq_tile(idx, load_fn, *, skip_tile_mask: bool = False):
                    bkv_start = idx * bkv_csz
                    cur_processed_kv_len = processed_kv_len + bkv_start
                    for kv_head_idx in range(actual_num_kv_heads):
                        if load_fn is _load_bkv_dequant:
                            bk_c, bv_c, k_norm_row = _load_bkv_dequant(
                                bkv_sem_idx,
                                kv_head_idx,
                                bkv_start,
                                bkv_csz,
                                return_k_norm=True,
                            )
                        else:
                            bk_c, bv_c = load_fn(
                                bkv_sem_idx,
                                kv_head_idx,
                                bkv_start,
                                bkv_csz,
                            )
                            k_norm_row = None
                        if not skip_tile_mask:
                            bv_c = _mask_v_tile(
                                bv_c, cur_processed_kv_len=cur_processed_kv_len
                            )
                        for bq_start in range(0, actual_bq_sz, actual_bq_csz):
                            bq_c = (
                                preloaded_bq[kv_head_idx]
                                if preloaded_bq is not None
                                else load_bq(
                                    bq_sem_idx,
                                    kv_head_idx,
                                    bq_start,
                                    actual_bq_csz,
                                )
                            )
                            lm_slice_start = bq_start * num_q_heads_per_kv_head
                            lm_slice_size = actual_bq_csz * num_q_heads_per_kv_head
                            lm_slice = (
                                kv_head_idx,
                                pl.ds(lm_slice_start, lm_slice_size),
                            )
                            cur_p, cur_exp_m_diff = flash_attention_step1_qk_softmax(
                                bq_c,
                                bk_c,
                                l_ref.at[*lm_slice],
                                m_ref.at[*lm_slice],
                                processed_q_len=processed_q_len + bq_start,
                                processed_kv_len=cur_processed_kv_len,
                                effective_kv_len=effective_kv_len,
                                skip_tile_mask=skip_tile_mask,
                                k_norm_row=k_norm_row,
                            )
                            flash_attention_step2_pv(
                                cur_p,
                                bv_c,
                                cur_exp_m_diff,
                                acc_ref.at[*lm_slice],
                            )

                can_skip_full_tile_mask = sliding_window is None
                if not (skip_kv_update or is_decode):
                    bkv_sz_frm_cache = jnp.maximum(offset - bkv_idx * bkv_sz, 0)
                    num_dequant_loops = jnp.minimum(
                        bkv_sz_frm_cache // bkv_csz, num_loops
                    )
                    raw_start_loop = jnp.minimum(
                        cdiv(bkv_sz_frm_cache, bkv_csz), num_loops
                    )

                    @pl.loop(0, num_dequant_loops, unroll=False)
                    def attention_loop_dequant(idx):
                        _run_tq_tile(
                            idx,
                            _load_bkv_dequant,
                            skip_tile_mask=can_skip_full_tile_mask,
                        )

                    @pl.when(num_dequant_loops < raw_start_loop)
                    def attention_step_mixed():
                        _run_tq_tile(
                            num_dequant_loops,
                            lambda sem, h, s, sz: _load_bkv_mixed(
                                sem, h, s, sz, bkv_sz_frm_cache
                            ),
                        )

                    @pl.loop(raw_start_loop, num_loops, unroll=False)
                    def attention_loop_raw(idx):
                        _run_tq_tile(idx, _load_bkv_raw)
                else:
                    num_full_loops = (
                        effective_bkv_sz // bkv_csz
                        if is_decode
                        else jnp.minimum(
                            jnp.maximum(kv_q_gap - bkv_idx * bkv_sz, 0) // bkv_csz,
                            effective_bkv_sz // bkv_csz,
                        )
                    )

                    @pl.loop(0, num_full_loops, unroll=False)
                    def attention_loop_tq_full(idx):
                        _run_tq_tile(
                            idx,
                            _load_bkv_dequant,
                            skip_tile_mask=can_skip_full_tile_mask,
                        )

                    if is_decode:

                        @pl.when(num_full_loops < num_loops)
                        def attention_step_tq_tail():
                            _run_tq_tile(num_full_loops, _load_bkv_dequant)
                    else:

                        @pl.loop(num_full_loops, num_loops, unroll=False)
                        def attention_loop_tq_tail(idx):
                            _run_tq_tile(idx, _load_bkv_dequant)

            acc = acc_ref[...]
            denom = broadcast_minor(l_ref[...], acc.shape)
            out = (
                acc * pl.reciprocal(denom, approx=True)
                if (denom.dtype == jnp.float32 and out_dtype != jnp.float32)
                else lax.div(acc, denom)
            ).astype(out_dtype)

            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            out_ref = (
                bo_x2_ref.at[bo_sem_idx]
                .bitcast(jnp.int32)
                .reshape(
                    actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head_per_packing,
                    head_dim,
                )
            )
            out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
            strided_store(out_ref, 0, out_ref.shape[0], 1, out)
            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        _fetch_bq(seq_idx=start_seq_idx, bq_idx=0, bq_sem_idx=0)
        _fetch_bkv(seq_idx=start_seq_idx, bkv_idx=cur_seq_start_bkv_idx, bkv_sem_idx=0)

    @pl.when(jnp.logical_and(start_seq_idx <= seq_idx, seq_idx < end_seq_idx))
    def pipeline():
        process()

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(bo_sem_idx=i)
            wait_update_kv_cache(bkv_sem_idx=i)


def static_validate_inputs(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    tq_page_size: int | None,
    tq_dequant_page_fn: Callable[..., Any] | None,
    tq_new_kv: jax.Array | None = None,
    tq_update_page_fn: Callable[..., Any] | None = None,
    skip_kv_update: bool = False,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
) -> None:
    """Validate inputs to the TurboQuant RPA v3 kernel at trace/compile time."""
    if tq_page_size is None or tq_page_size <= 0:
        raise ValueError(f"Expected positive {tq_page_size=}.")
    if tq_dequant_page_fn is None:
        raise ValueError("Expected non-None tq_dequant_page_fn.")

    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape or (q.shape[0], q.shape[2]) != (k.shape[0], k.shape[2]):
        raise ValueError(
            f"Mismatched Q/K/V shapes: {q.shape=}, {k.shape=}, {v.shape=}."
        )
    max_num_tokens, actual_num_q_heads, _ = q.shape
    actual_num_kv_heads = k.shape[1]
    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(
            f"Expected {actual_num_q_heads=} to be divisible by {actual_num_kv_heads=}."
        )

    if (
        kv_cache.dtype != jnp.uint8
        or len(kv_cache.shape) != 4
        or kv_cache.shape[1] != actual_num_kv_heads
        or kv_cache.shape[2] <= 0
        or kv_cache.shape[3] != 128
    ):
        raise ValueError(
            f"Expected 4D uint8 TurboQuant kv_cache (..., {actual_num_kv_heads},"
            f" page_words, 128), got {kv_cache.shape=}, {kv_cache.dtype=}."
        )

    if not skip_kv_update:
        if tq_new_kv is None or tq_update_page_fn is None:
            raise ValueError(
                "tq_new_kv and tq_update_page_fn are required when"
                " skip_kv_update=False."
            )
        if (
            tq_new_kv.dtype != jnp.uint8
            or len(tq_new_kv.shape) != 4
            or tq_new_kv.shape[0] != max_num_tokens
            or tq_new_kv.shape[1] <= 0
            or tq_new_kv.shape[2:] != (actual_num_kv_heads, 128)
        ):
            raise ValueError(
                f"Expected 4D uint8 tq_new_kv ({max_num_tokens}, num_new_words,"
                f" {actual_num_kv_heads}, 128), got {tq_new_kv.shape=},"
                f" {tq_new_kv.dtype=}."
            )

    if any(
        arr.dtype != jnp.int32 or len(arr.shape) != 1
        for arr in (kv_lens, page_indices, cu_q_lens, distribution)
    ):
        raise ValueError("Metadata arrays must be 1D int32.")
    max_num_seqs = kv_lens.shape[0]
    if (
        page_indices.shape[0] % max_num_seqs != 0
        or cu_q_lens.shape != (max_num_seqs + 1,)
        or distribution.shape != (3,)
    ):
        raise ValueError(
            f"Invalid metadata shapes: {kv_lens.shape=}, {page_indices.shape=},"
            f" {cu_q_lens.shape=}, {distribution.shape=}."
        )

    kv_packing = get_dtype_packing(k.dtype)
    if tq_page_size % kv_packing != 0:
        raise ValueError(f"{tq_page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    for prefix, bs in (("decode", d_block_sizes), ("mixed", m_block_sizes)):
        if bs is None:
            continue
        bq_sz, bkv_sz, bq_csz, bkv_csz = bs
        if not (
            bq_csz > 0 and bq_sz % bq_csz == 0 and bkv_csz > 0 and bkv_sz % bkv_csz == 0
        ):
            raise ValueError(f"Invalid {prefix} block sizes {bs}.")
        if bkv_sz % tq_page_size != 0 or bkv_csz % tq_page_size != 0:
            raise ValueError(
                f"{prefix} {bkv_sz=} and {bkv_csz=} must be divisible by"
                f" {tq_page_size=}."
            )


def prepare_tq_block_sizes(
    block_sizes: tuple[int, int, int, int] | None,
    *,
    q_dtype,
    kv_dtype,
    actual_num_q_heads: int,
    actual_num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_num_tokens: int,
    max_num_seqs: int,
    pages_per_seq: int,
    case: RpaCase,
    tpu_version: int | None = None,
) -> dict[str, int]:
    """Select `(bq_sz, bkv_sz, bq_csz, bkv_csz)` bounded for compressed-page VMEM."""
    if block_sizes is not None:
        return dict(zip(("bq_sz", "bkv_sz", "bq_csz", "bkv_csz"), block_sizes))
    if tpu_version is None:
        tpu_version = get_tpu_version()
    sizes = get_default_block_sizes(
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size,
        max_num_tokens,
        max_num_seqs,
        pages_per_seq,
        case=case,
    )
    if tpu_version >= 7 and page_size >= 64:
        target_csz = 256
    else:
        target_csz = 128 if (head_dim <= 128 or page_size >= 64) else 64
    bkv_csz = align_to(min(sizes["bkv_csz"], max(page_size, target_csz)), page_size)
    bkv_sz = align_to(min(sizes["bkv_sz"], max(page_size * 32, 512)), bkv_csz)
    bq_sz = sizes["bq_sz"]
    return {
        "bq_sz": bq_sz,
        "bkv_sz": bkv_sz,
        "bq_csz": min(sizes["bq_csz"], bq_sz),
        "bkv_csz": bkv_csz,
    }


@jax.jit(
    static_argnames=(
        "skip_kv_update",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "d_block_sizes",
        "m_block_sizes",
        "vmem_limit_bytes",
        "tq_page_size",
        "tq_dequant_page_fn",
        "tq_update_page_fn",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
)
def ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    kv_cache: jax.Array,  # u8[total_num_pages, actual_num_kv_heads, page_words, 128]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    tq_page_size: int,
    tq_dequant_page_fn: Callable[..., Any],
    tq_new_kv: jax.Array | None = None,
    tq_update_page_fn: Callable[..., Any] | None = None,
    skip_kv_update: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Fused Pallas RPA v3 attention over a compressed TurboQuant `uint8` KV cache."""
    q, k, v = queries, keys, values
    tpu_version = get_tpu_version()
    out_dtype = jnp.float32 if q.dtype == jnp.float32 else jnp.bfloat16
    mask_value = float(jnp.finfo(out_dtype).min)
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    static_validate_inputs(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        tq_page_size=tq_page_size,
        tq_dequant_page_fn=tq_dequant_page_fn,
        tq_new_kv=tq_new_kv,
        tq_update_page_fn=tq_update_page_fn,
        skip_kv_update=skip_kv_update,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        d_block_sizes=d_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    actual_num_q_heads, actual_head_dim = q.shape[1], q.shape[2]
    actual_num_kv_heads = k.shape[1]
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q, kv = prepare_inputs(q, k, v, fuse_non_tiling_axis_swap=False)
    _, max_num_tokens, num_q_per_kv_packed, q_packing, head_dim = q.shape
    page_size = tq_page_size
    num_kv_heads_x2_per_kv_packing = kv.shape[1]
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_q_heads_per_kv_head = num_q_per_kv_packed * q_packing

    scalar_prefetches = (
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        jnp.zeros((3,), jnp.int32),
        jnp.full((4,), -1, jnp.int32),
        jnp.full((6,), -1, jnp.int32),
    )

    def run_rpa_kernel(
        q,
        kv_cache,
        *,
        bq_sz: int,
        bkv_sz: int,
        bq_csz: int,
        bkv_csz: int,
        case: RpaCase,
    ):
        is_decode = case == RpaCase.DECODE
        has_tq_new_kv = tq_new_kv is not None and not skip_kv_update
        bkv_stride = num_kv_heads_x2_per_kv_packing + int(
            has_bank_conflicts(num_kv_heads_x2_per_kv_packing)
        )
        raw_bkv_vmem_sz = 1 if (is_decode or skip_kv_update) else bkv_sz
        bq_double_buf = pltpu.VMEM(
            (2, actual_num_kv_heads, bq_sz, *q.shape[2:]), q.dtype
        )
        l_scratch = pltpu.VMEM(
            (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128), out_dtype
        )
        scratch_shapes: list[Any] = [
            pltpu.VMEM((2, raw_bkv_vmem_sz, bkv_stride, *kv.shape[2:]), kv.dtype),
            bq_double_buf,
            bq_double_buf,
            pltpu.SemaphoreType.DMA((4, 2)),
            l_scratch,
            l_scratch,
            pltpu.VMEM(
                (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
                out_dtype,
            ),
            pltpu.VMEM(
                (2, bkv_sz // page_size, actual_num_kv_heads, kv_cache.shape[2], 128),
                kv_cache.dtype,
            ),
        ]
        if has_tq_new_kv:
            assert tq_new_kv is not None
            scratch_shapes.append(
                pltpu.VMEM(
                    (
                        2,
                        1 if is_decode else bkv_sz,
                        tq_new_kv.shape[1],
                        actual_num_kv_heads,
                        128,
                    ),
                    tq_new_kv.dtype,
                )
            )

        scope_name = (
            f"TQ_RPA{case.symbol}-p_{page_size}"
            f"-bq_{bq_sz}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}"
        )
        if sliding_window is not None:
            scope_name += f"-sw_{sliding_window}"
        hbm_cls = pltpu.HBM if tpu_version >= 7 else jax.ShapeDtypeStruct
        kernel = pl.pallas_call(
            functools.partial(
                _ragged_paged_attention_kernel,
                has_tq_new_kv=has_tq_new_kv,
                skip_kv_update=skip_kv_update,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                mask_value=mask_value,
                bq_sz=bq_sz,
                bkv_sz=bkv_sz,
                bq_csz=bq_csz,
                bkv_csz=bkv_csz,
                case=case,
                tq_page_size=page_size,
                tq_dequant_page_fn=tq_dequant_page_fn,
                tq_update_page_fn=tq_update_page_fn,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)]
                * (4 if has_tq_new_kv else 3),
                out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 2,
                grid=(1,),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
                disable_semaphore_checks=True,
            ),
            out_shape=[
                hbm_cls(shape=q.shape, dtype=q.dtype),
                hbm_cls(shape=kv_cache.shape, dtype=kv_cache.dtype),
            ],
            input_output_aliases={7: 0, 9: 1},
            name=scope_name,
        )

        hbm_args = (q, kv, kv_cache) + ((tq_new_kv,) if has_tq_new_kv else ())
        if tpu_version >= 7:

            @jax.jit
            def run(scalar_prefetches, *args):
                return kernel(
                    *scalar_prefetches,
                    *(pltpu.with_memory_space_constraint(a, pltpu.HBM) for a in args),
                )

            return run(scalar_prefetches, *hbm_args)
        return kernel(*scalar_prefetches, *hbm_args)

    block_size_kw = dict(
        q_dtype=q.dtype,
        kv_dtype=kv.dtype,
        actual_num_q_heads=actual_num_q_heads,
        actual_num_kv_heads=actual_num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_num_tokens=max_num_tokens,
        max_num_seqs=max_num_seqs,
        pages_per_seq=pages_per_seq,
        tpu_version=tpu_version,
    )
    for case, bs in ((RpaCase.DECODE, d_block_sizes), (RpaCase.MIXED, m_block_sizes)):
        q, kv_cache = run_rpa_kernel(
            q,
            kv_cache,
            **prepare_tq_block_sizes(bs, case=case, **block_size_kw),
            case=case,
        )

    return (
        prepare_outputs(
            q,
            actual_num_q_heads_per_kv_head,
            actual_head_dim,
            fuse_non_tiling_axis_swap=False,
        ),
        kv_cache,
    )
