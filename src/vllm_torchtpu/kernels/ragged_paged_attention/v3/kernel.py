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
"""TPU-Friendly Ragged Paged Attention kernel.

This kernel offers a highly optimized implementation of ragged paged attention,
specifically designed for TPU and compatible with a wide range of model
specifications. It supports mixed prefill and decoding, enhancing throughput
during inference.
"""

import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import flash_attention, staging
from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import (
    align_to,
    async_copy,
    cdiv,
    get_dtype_bitwidth,
    get_dtype_packing,
    get_span_int_dtype,
    get_tpu_version,
    has_bank_conflicts,
    next_power_of_2,
    validate_common_inputs,
)

RpaCase = staging.RpaCase
get_smem_estimate_bytes = staging.get_smem_estimate_bytes


def ref_ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    skip_kv_update: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    if out_dtype is None:
        out_dtype = jnp.float32 if queries.dtype == jnp.float32 else jnp.bfloat16

    if mask_value is None:
        # We do not set to -inf directly because (-inf) - (-inf) is nan.
        mask_value = cast(float, jnp.finfo(out_dtype).min)
    dynamic_validate_inputs(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=use_causal_mask,
        skip_kv_mask=skip_kv_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]
    merged_kv = merge_kv(keys, values)
    assert merged_kv.shape[-3:] == kv_cache.shape[-3:]

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = kv_cache.shape
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    assert num_kv_heads_x2 % 2 == 0
    assert actual_num_q_heads % actual_num_kv_heads == 0
    assert head_dim % 128 == 0
    assert get_dtype_packing(kv_cache.dtype) == kv_packing
    assert num_kv_heads_x2 == align_to(actual_num_kv_heads * 2, kv_packing)
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    outputs = []

    for i in range(distribution[-1]):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start

        kv_len = kv_lens[i]
        indices_start = i * pages_per_seq
        indices_end = indices_start + cdiv(kv_len, page_size)
        indices = page_indices[indices_start:indices_end]
        q = queries[q_start:q_end, :, :actual_head_dim]

        # Update the kv cache.
        assert kv_len - q_len >= 0
        gathered_kv = kv_cache[indices]
        gathered_shape = gathered_kv.shape
        gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])
        if not skip_kv_update:
            # KV-sharing (cross-layer) layers reuse the target layer's K/V that
            # is already present in the (shared) cache, so they must neither
            # insert their own K/V nor write the cache back.
            gathered_kv = gathered_kv.at[kv_len - q_len : kv_len].set(
                merged_kv[q_start:q_end]
            )
            kv_cache = kv_cache.at[indices].set(gathered_kv.reshape(gathered_shape))

        kv = gathered_kv.reshape(-1, num_kv_heads_x2, head_dim)[
            :, : actual_num_kv_heads * 2, :
        ].reshape(-1, actual_num_kv_heads, head_dim * 2)
        k = kv[:kv_len, :, :head_dim][:, :, :actual_head_dim]
        v = kv[:kv_len, :, head_dim:][:, :, :actual_head_dim]
        k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
        v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(k.dtype, jnp.floating):
                dtype_info = jnp.finfo(k.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(k.dtype)

        attn = jnp.einsum(
            "qhd,khd->hqk", q, k, preferred_element_type=jnp.float32
        ).astype(out_dtype)
        attn *= sm_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale
        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        if use_causal_mask:
            q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(
                jnp.int32, attn.shape, 1
            )
            kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
            mask = q_span >= kv_span
            if sliding_window is not None:
                mask = jnp.logical_and(mask, q_span < kv_span + sliding_window)
            attn = jnp.where(mask, attn, mask_value)
        attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(out_dtype)
        if v_scale is not None:
            out *= v_scale

        outputs.append(out)

    result = jnp.concatenate(outputs, axis=0)
    return result, kv_cache


def get_vmem_estimate_bytes(
    actual_num_kv_heads,
    actual_num_q_heads_per_kv_head,
    actual_head_dim,
    bq_sz,
    bkv_sz,
    q_dtype,
    kv_dtype,
    *,
    out_dtype=jnp.float32,
):
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    out_bits = get_dtype_bitwidth(out_dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    bkv_stride = cdiv(actual_num_kv_heads * 2, kv_packing)
    if has_bank_conflicts(bkv_stride):
        bkv_stride += 1
    head_dim = align_to(actual_head_dim, 128)

    total_bits = (
        # bkv_x2_ref
        (2 * bkv_sz * bkv_stride * kv_packing * head_dim) * (32 // kv_packing)
        +
        # bq_x2_ref + bo_x2_ref
        2
        * (2 * actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim)
        * (32 // q_packing)
        +
        # l_ref + m_ref
        2 * (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * out_bits
        +
        # acc_ref
        (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) * out_bits
    )
    return cdiv(total_bits, 8)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        align_to(actual_head_dim, 128),
    )


def _ragged_paged_attention_kernel(*args, **kwargs):
    distribution_ref = args[3]
    has_prefill_pass = kwargs.pop("has_prefill_pass", True)
    start_seq_idx, end_seq_idx = kwargs["case"].get_range(
        distribution_ref, has_prefill_pass=has_prefill_pass
    )

    @pl.loop(start_seq_idx, end_seq_idx)
    def _(seq_idx):
        return _ragged_paged_attention_kernel_loop(
            seq_idx,
            start_seq_idx,
            end_seq_idx,
            *args,
            **kwargs,
        )


def _ragged_paged_attention_kernel_loop(
    seq_idx,
    start_seq_idx,
    end_seq_idx,
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    # TODO(jevinjiang): merge these into one so we can save SMEM.
    distribution_ref,  # [3] (decode_end, prefill_end, mixed_end)
    sem_ids_ref,  # [3] (bq_sem_idx, bkv_sem_idx, bo_sem_idx)
    # [4] (bo_sem_0_seq_idx, bo_sem_1_seq_idx, bo_sem_0_bo_idx, bo_sem_1_bo_idx)
    bo_ids_ref,
    # [6] (bkv_sem_0_seq_idx, bkv_sem_1_seq_idx, bkv_sem_0_offset, bkv_sem_1_offset,
    #     bkv_sem_0_sz, bkv_sem_1_sz)
    bkv_update_ids_ref,
    # Input
    # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing,
    #     q_packing, head_dim]
    q_hbm_ref,
    kv_hbm_ref,  # [max_num_tokens, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache_hbm_ref,
    # Output
    # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing,
    #     q_packing, head_dim]
    o_hbm_ref,
    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    updated_kv_cache_hbm_ref,
    # Scratch
    ## Add one extra to handle bank conflicts for strided load if needed.
    # [2, bkv_sz, num_kv_heads_x2 // kv_packing (+ 1), kv_packing, head_dim]
    bkv_x2_ref,
    # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing,
    #     head_dim]
    bq_x2_ref,
    # [2, actual_num_kv_heads, bq_sz, num_q_heads_per_kv_head // q_packing, q_packing,
    #     head_dim]
    bo_x2_ref,
    sems,  # [4, 2]
    l_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    m_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128],
    acc_ref,  # [actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim],
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    skip_kv_update: bool = False,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    static_q_len: int | None = None,
    bq_sz,  # bq fetch size
    bkv_sz,  # bkv prefetch size
    bq_csz,  # bq compute size
    bkv_csz,  # bkv compute size
    case: RpaCase = RpaCase.MIXED,
    debug_mode: bool = False,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert q_hbm_ref.shape[-1] == kv_cache_hbm_ref.shape[-1]

    if case == RpaCase.DECODE:
        use_causal_mask = False

    out_dtype = acc_ref.dtype
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        head_dim,
    ) = q_hbm_ref.shape
    (
        total_num_pages,
        page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        _,
    ) = kv_cache_hbm_ref.shape
    bkv_stride = bkv_x2_ref.shape[2]
    assert bkv_stride in (
        num_kv_heads_x2_per_kv_packing,
        num_kv_heads_x2_per_kv_packing + 1,
    )
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    # num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert head_dim % 128 == 0
    assert bkv_sz % page_size == 0
    bkv_p = bkv_sz // page_size
    del distribution_ref
    max_kv_len = pages_per_seq * page_size
    span_int_ty = get_span_int_dtype(q_dtype, pages_per_seq, page_size, bkv_csz)

    def debug_print(msg, *args):
        if debug_mode:
            pl.debug_print(msg, *args)

    def _async_copy(src, dst, sem, wait):
        async_copy(src, dst, sem, wait, debug_mode=debug_mode)

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[staging.SEM_BKV_FETCH, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]

        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )
        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        if skip_kv_update:
            # KV-sharing layer: the target layer already wrote all kv (including
            # the current step's tokens) into the shared cache, so fetch every
            # token from the cache and write nothing back (bkv_sz_frm_new == 0,
            # so update_sz == 0 and start_update_kv_cache is never triggered).
            kv_left_frm_cache = kv_left
        else:
            kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        debug_print(
            f"[RPA debug] -----------{'wait' if wait else 'start'}_fetch_bkv-----------"
        )
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bkv_idx={}", bkv_idx)
        debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
        debug_print("[RPA debug] kv_len_start={}", kv_len_start)
        debug_print("[RPA debug] kv_p_start={}", kv_p_start)
        debug_print("[RPA debug] kv_left={}", kv_left)
        debug_print("[RPA debug] kv_left_frm_cache={}", kv_left_frm_cache)
        debug_print("[RPA debug] kv_left_frm_new={}", kv_left_frm_new)
        debug_print("[RPA debug] bkv_sz_frm_cache={}", bkv_sz_frm_cache)
        debug_print("[RPA debug] bkv_sz_frm_new={}", bkv_sz_frm_new)
        debug_print("[RPA debug] page_indices_offset={}", page_indices_offset)

        if not wait:
            # Make sure the current bkv buffer is safe to overwrite.
            wait_update_kv_cache(bkv_sem_idx)

            # Fetch effective kv from kv cache. To pipeline multiple DMA calls, we
            # utilize static for loop instead of dynamic for loop.
            for i in range(bkv_p):
                # Ensure only effective kvs are copied.
                sz = jnp.clip(kv_left_frm_cache - i * page_size, 0, page_size)
                # If the page index is out of bound, we set page_idx to the last page.
                # And there will be no copy since sz will be 0.
                page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                _async_copy(
                    cache_hbm_ref.at[pl.ds(page_indices_ref[page_idx] * page_size, sz)],
                    vmem_ref.at[pl.ds(i * page_size, sz)],
                    sem,
                    wait=False,
                )
                debug_print("[RPA debug] loop_body i={}, sz={}", i, sz)

            new_kv_len_start = q_end - kv_left_frm_new
            debug_print("[RPA debug] new_kv_len_start={}", new_kv_len_start)
            _async_copy(
                kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                vmem_ref.at[pl.ds(bkv_sz_frm_cache, bkv_sz_frm_new)],
                sem,
                wait,
            )
        else:
            dst = vmem_ref.at[pl.ds(0, bkv_sz_frm_cache + bkv_sz_frm_new)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )
        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    def _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False):
        sem = sems.at[staging.SEM_BKV_UPDATE, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        cache_hbm_shape = updated_kv_cache_hbm_ref.shape
        cache_hbm_ref = updated_kv_cache_hbm_ref.reshape(
            cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
        )

        debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_update_kv_cache-----------"
        )
        debug_print("[RPA debug] seq_idx={}", seq_idx)
        debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
        debug_print("[RPA debug] offset={}", offset)
        debug_print("[RPA debug] update_sz={}", update_sz)
        debug_print("[RPA debug] bkv_id={}", bkv_id)
        debug_print("[RPA debug] kv_p_start={}", kv_p_start)
        debug_print("[RPA debug] kv_p_end={}", kv_p_end)
        debug_print("[RPA debug] ignore={}", ignore)
        debug_print("[RPA debug] p_ignore={}", p_ignore)
        debug_print("[RPA debug] page_indices_offset={}", page_indices_offset)

        def loop_body(i, states):
            update_sz, ignore = states
            sz = jnp.minimum(page_size - ignore, update_sz)

            _async_copy(
                vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                cache_hbm_ref.at[
                    pl.ds(
                        page_indices_ref[page_indices_offset + i] * page_size + ignore,
                        sz,
                    )
                ],
                sem,
                wait,
            )
            debug_print("[RPA debug] loop_body i={}, sz={}", i, sz)
            return update_sz - sz, 0

        if not wait:
            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (update_sz, ignore),  # total transfer size
                unroll=False,
            )
        else:
            dst = cache_hbm_ref.at[pl.ds(0, update_sz)]
            _async_copy(
                src=dst,
                dst=dst,
                sem=sem,
                wait=True,
            )

    staging_refs = staging.StagingRefs(
        q_hbm_ref=q_hbm_ref,
        o_hbm_ref=o_hbm_ref,
        bq_x2_ref=bq_x2_ref,
        bo_x2_ref=bo_x2_ref,
        cu_q_lens_ref=cu_q_lens_ref,
        bo_ids_ref=bo_ids_ref,
        sems=sems,
    )
    kv_update_ring = staging.KvUpdateRing(bkv_update_ids_ref, _update_kv_cache)
    wait_update_kv_cache = kv_update_ring.wait

    def compute_bkv_attention(ctx: staging.BkvStepContext) -> None:
        assert bkv_sz % kv_packing == 0
        num_loops = cdiv(ctx.effective_bkv_sz, bkv_csz)

        @pl.loop(0, num_loops, unroll=False)
        def attention_loop(idx):
            prev_lm_slice = None
            prev_p = None
            prev_v = None
            prev_exp_m_diff = None
            bkv_start = idx * bkv_csz

            for bq_start in range(0, ctx.actual_bq_sz, ctx.actual_bq_csz):
                for kv_head_idx in range(actual_num_kv_heads):
                    bk_c, bv_c = staging.load_bkv(
                        bkv_x2_ref, ctx.bkv_sem_idx, kv_head_idx, bkv_start, bkv_csz
                    )
                    bq_c = staging.load_bq(
                        bq_x2_ref,
                        ctx.bq_sem_idx,
                        kv_head_idx,
                        bq_start,
                        ctx.actual_bq_csz,
                    )
                    lm_slice = (
                        kv_head_idx,
                        pl.ds(
                            bq_start * num_q_heads_per_kv_head,
                            ctx.actual_bq_csz * num_q_heads_per_kv_head,
                        ),
                    )
                    cur_p, cur_v, cur_exp_m_diff = (
                        flash_attention.flash_attention_step1_qk_softmax(
                            bq_c,
                            bk_c,
                            bv_c,
                            l_ref.at[*lm_slice],
                            m_ref.at[*lm_slice],
                            processed_q_len=ctx.processed_q_len + bq_start,
                            processed_kv_len=ctx.processed_kv_len + bkv_start,
                            effective_kv_len=ctx.effective_kv_len,
                            num_q_heads_per_kv_head=num_q_heads_per_kv_head,
                            sm_scale=sm_scale,
                            out_dtype=out_dtype,
                            mask_value=mask_value,
                            span_int_ty=span_int_ty,
                            max_kv_len=max_kv_len,
                            use_causal_mask=use_causal_mask,
                            skip_kv_mask=skip_kv_mask,
                            sliding_window=sliding_window,
                            soft_cap=soft_cap,
                            q_scale=q_scale,
                            k_scale=k_scale,
                        )
                    )
                    if prev_lm_slice is not None:
                        flash_attention.flash_attention_step2_pv(
                            prev_p,
                            prev_v,
                            prev_exp_m_diff,
                            acc_ref.at[*prev_lm_slice],
                            out_dtype=out_dtype,
                            v_scale=v_scale,
                        )
                    prev_lm_slice = lm_slice
                    prev_p = cur_p
                    prev_v = cur_v
                    prev_exp_m_diff = cur_exp_m_diff

            assert prev_lm_slice is not None
            flash_attention.flash_attention_step2_pv(
                prev_p,
                prev_v,
                prev_exp_m_diff,
                acc_ref.at[*prev_lm_slice],
                out_dtype=out_dtype,
                v_scale=v_scale,
            )

    staging.run_rpa_sequence_pipeline(
        seq_idx,
        start_seq_idx=start_seq_idx,
        end_seq_idx=end_seq_idx,
        staging_refs=staging_refs,
        kv_lens_ref=kv_lens_ref,
        sem_ids_ref=sem_ids_ref,
        kv_update_ring=kv_update_ring,
        l_ref=l_ref,
        m_ref=m_ref,
        acc_ref=acc_ref,
        fetch_bkv_fn=_fetch_bkv,
        compute_bkv_fn=compute_bkv_attention,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        bq_csz=bq_csz,
        use_causal_mask=use_causal_mask,
        sliding_window=sliding_window,
        static_q_len=static_q_len,
        debug_mode=debug_mode,
    )


def merge_kv(
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
):
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)

    head_dim = align_to(actual_head_dim, 128)
    kv = jnp.pad(
        jnp.concat([k, v], axis=-1).reshape(
            max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
        ),
        (
            (0, 0),
            (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads_x2 // kv_packing,
        kv_packing,
        head_dim,
    )
    return kv


def prepare_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
    k: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    v: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim],
    *,
    fuse_non_tiling_axis_swap: bool | None = None,
):
    if fuse_non_tiling_axis_swap is None:
        fuse_non_tiling_axis_swap = envs.USE_RPA_PIPELINED_DMA_STAGING
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    actual_num_kv_heads = k.shape[1]
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q_reshaped = jnp.pad(
        q.reshape(
            max_num_tokens,
            actual_num_kv_heads,
            actual_num_q_heads_per_kv_head,
            actual_head_dim,
        ),
        (
            (0, 0),
            (0, 0),
            (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        actual_num_kv_heads,
        num_q_heads_per_kv_head // q_packing,
        q_packing,
        head_dim,
    )
    if not fuse_non_tiling_axis_swap:  # noqa: SIM108 (one comment per path)
        # Legacy path: Separate query axis swap emitting
        # xlu.transpose (Transpose::Execute).
        q = q_reshaped.swapaxes(0, 1)
    else:
        # Fused DMA layout path: Leave query in token-major layout
        # [max_num_tokens, actual_num_kv_heads, ...] and carry axis permutation
        # directly in Pallas strided DMA descriptors.
        q = q_reshaped
    # TODO(kyuyeunk, chengjiyao): Add kv quantization here.
    kv = merge_kv(k, v)
    return q, kv


def prepare_outputs(
    # [actual_num_kv_heads, max_num_tokens, num_q_heads_per_kv_head // q_packing,
    #     q_packing, head_dim]
    out,
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
    *,
    fuse_non_tiling_axis_swap: bool | None = None,
):
    if fuse_non_tiling_axis_swap is None:
        fuse_non_tiling_axis_swap = envs.USE_RPA_PIPELINED_DMA_STAGING
    if not fuse_non_tiling_axis_swap:
        (
            actual_num_kv_heads,
            max_num_tokens,
            num_q_heads_per_kv_head_per_q_packing,
            q_packing,
            head_dim,
        ) = out.shape
        actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
        return (
            out.swapaxes(0, 1)
            .reshape(
                max_num_tokens,
                actual_num_kv_heads,
                num_q_heads_per_kv_head_per_q_packing * q_packing,
                head_dim,
            )[:, :, :actual_num_q_heads_per_kv_head, :actual_head_dim]
            .reshape(max_num_tokens, actual_num_q_heads, actual_head_dim)
        )
    else:
        # Direct reshape without redundant inverse axis swap
        (
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head_per_q_packing,
            q_packing,
            head_dim,
        ) = out.shape
        actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
        return out.reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head_per_q_packing * q_packing,
            head_dim,
        )[:, :, :actual_num_q_heads_per_kv_head, :actual_head_dim].reshape(
            max_num_tokens, actual_num_q_heads, actual_head_dim
        )


# Expect to run this validation during runtime.
def dynamic_validate_inputs(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
):
    q, k, v = queries, keys, values
    static_validate_inputs(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        use_causal_mask=use_causal_mask,
        skip_kv_mask=skip_kv_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        d_block_sizes=d_block_sizes,
        p_block_sizes=p_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    max_num_tokens = q.shape[0]
    total_num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    i, j, k = distribution
    if not (i <= j <= k):
        raise ValueError(f"Invalid distribution: {distribution=}")

    if k > max_num_seqs:
        raise ValueError(f"num_seqs={k} must be <= {max_num_seqs=}")

    if cu_q_lens[k] > max_num_tokens:
        raise ValueError(f"Total q tokens {cu_q_lens[k]} must be <= {max_num_tokens=}.")
    for i in range(k):
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        kv_len = kv_lens[i]
        if not (0 < q_len <= kv_len):
            raise ValueError(f"Require 0 < {q_len=} <= {kv_len=} at sequence {i}.")
        page_cnt = cdiv(kv_len, page_size)
        if page_cnt > pages_per_seq:
            raise ValueError(
                f"Require {page_cnt=} <= {pages_per_seq=} at sequence {i} where"
                f" {kv_len=} and {page_size=}."
            )
        for p in range(page_cnt):
            page_idx = page_indices[i * pages_per_seq + p]
            if not (0 <= page_idx < total_num_pages):
                raise ValueError(
                    f"Require 0 <= {page_idx=} < {total_num_pages=} at sequence"
                    f" {i} where {kv_len=} and {page_size=}."
                )


# Expect to run this validation during compile time.
def static_validate_inputs(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params.
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Validate inputs to the RPA kernel statically."""
    if use_causal_mask and skip_kv_mask:
        raise ValueError("Can not skip kv mask when using causal mask.")

    q, k, v = queries, keys, values
    if len(kv_cache.shape) != 5:
        raise ValueError(f"Expected 5D kv_cache, got {kv_cache.shape=}.")
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")

    (
        total_num_pages,
        page_size,
        num_kv_heads_x2_per_kv_packing,
        kv_packing,
        head_dim,
    ) = kv_cache.shape
    if kv_packing != get_dtype_packing(kv_cache.dtype):
        raise ValueError(f"{kv_packing=} does not match with {kv_cache.dtype=}")

    validate_common_inputs(
        q,
        k,
        v,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        page_size=page_size,
        kv_packing=kv_packing,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        chunk_prefill_size=chunk_prefill_size,
        block_sizes_by_name=(
            ("decode", d_block_sizes),
            ("prefill", p_block_sizes),
            ("mixed", m_block_sizes),
        ),
        vmem_limit_bytes=vmem_limit_bytes,
    )

    actual_head_dim = q.shape[2]
    actual_num_kv_heads = k.shape[1]
    expected_kv_cache_shape = get_kv_cache_shape(
        total_num_pages,
        page_size,
        actual_num_kv_heads,
        actual_head_dim,
        kv_cache.dtype,
    )
    if kv_cache.shape != expected_kv_cache_shape:
        raise ValueError(
            f"Expected {kv_cache.shape=} to be equal to {expected_kv_cache_shape=}"
        )
    if head_dim != align_to(actual_head_dim, 128):
        raise ValueError(
            f"Expected {head_dim=} is equal to {align_to(actual_head_dim, 128)=}"
        )
    if not (kv_cache.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"Expected {kv_cache.dtype=} to be equal to {k.dtype=} and {v.dtype=}."
        )

    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    if num_kv_heads_x2 % 2 != 0:
        raise ValueError(
            f"Combined KV heads must be divisible by 2, but got {num_kv_heads_x2}"
        )
    if num_kv_heads_x2 % kv_packing != 0 or num_kv_heads_x2 // 2 < actual_num_kv_heads:
        raise ValueError(
            f"Invalid {num_kv_heads_x2=}, {actual_num_kv_heads=}, {kv_packing=}"
        )

    del sm_scale
    del mask_value
    del out_dtype
    del q_scale
    del k_scale
    del v_scale


def get_default_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    max_num_seqs,
    pages_per_seq,
    *,
    case: RpaCase = RpaCase.MIXED,
    tpu_version: int | None = None,
):
    """Get (bq, bkv_sz, bq_csz, bkv_csz) by some heuristic formulas.

    Note the default block sizes are not necessarily optimal.
    """
    if tpu_version is None:
        tpu_version = get_tpu_version()

    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = next_power_of_2(align_to(actual_num_kv_heads * 2, kv_packing))
    head_dim = align_to(head_dim, 128)
    num_q_heads_per_kv_head = next_power_of_2(actual_num_q_heads // actual_num_kv_heads)

    max_q = next_power_of_2(max_num_tokens)
    max_kv = pages_per_seq * page_size

    min_bkv_sz_to_peak = (
        16 * 1024 * 1024 * kv_packing // 4 // head_dim // num_kv_heads_x2
    )

    match tpu_version:
        case 5 | 6:
            if case == RpaCase.DECODE:
                bq_sz = 1
                bkv_sz = min(min_bkv_sz_to_peak, max_kv)
                bq_csz = 1
                bkv_csz = min(min_bkv_sz_to_peak, max_kv)
            else:
                bq_sz = min(1024 // num_q_heads_per_kv_head, max_q // 2)
                bkv_sz = min(1024, max_kv)
                bq_csz = min(512 // num_q_heads_per_kv_head, max_q)
                bkv_csz = min(512, align_to(max_kv // 2, page_size))
        case 7:
            if case == RpaCase.DECODE:
                bq_sz = 1
                bkv_sz = min(min_bkv_sz_to_peak, max_kv)
                bq_csz = 1
                bkv_csz = min(min_bkv_sz_to_peak, max_kv)
            else:
                bq_sz = min(2048 // num_q_heads_per_kv_head, max_q // 2)
                bkv_sz = min(2048, max_kv)
                bq_csz = min(1024 // num_q_heads_per_kv_head, max_q // 2)
                bkv_csz = min(512, align_to(max_kv // 2, page_size))
        case _:
            raise NotImplementedError(f"Unsupported {tpu_version=}.")

    # The query compute block should never exceed the query fetch block.
    bq_csz = min(bq_csz, bq_sz)

    return {
        "bq_sz": max(1, bq_sz),
        "bkv_sz": align_to(bkv_sz, page_size),
        "bq_csz": max(1, bq_csz),
        "bkv_csz": align_to(bkv_csz, page_size),
    }


@jax.jit(
    static_argnames=(
        "use_causal_mask",
        "skip_kv_mask",
        "skip_kv_update",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "out_dtype",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "d_block_sizes",
        "p_block_sizes",
        "m_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "disable_bounds_checks",
        "disable_semaphore_checks",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache"),
)
def ragged_paged_attention(
    queries: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    keys: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    values: jax.Array,  # [max_num_tokens, actual_num_kv_heads, actual_head_dim]
    # [total_num_pages, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]
    kv_cache: jax.Array,
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    use_causal_mask: bool = True,
    skip_kv_mask: bool = False,
    skip_kv_update: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    out_dtype: Any = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    # Kernel optimization params.
    chunk_prefill_size: int | None = None,
    # Kernel tuning params for decode, prefill, and mixed cases.
    # Each case takes a tuple of (bq_sz, bkv_sz, bq_csz, bkv_csz).
    # - bq_sz: the block size for the query fetching.
    # - bkv_sz: the block size for the kv fetching.
    # - bq_csz: the compute size of the block query.
    # - bkv_csz: the compute size of the block kv.
    d_block_sizes: tuple[int, int, int, int] | None = None,
    p_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    # Debug params.
    debug_mode: bool = False,
    disable_bounds_checks: bool = True,
    disable_semaphore_checks: bool = True,
):
    """Ragged paged attention that supports mixed prefill and decode.

    Args:
      queries: concatenated all sequences' queries.
      keys: concatenated all sequences' keys (quantized).
      values: concatenated all sequences' values (quantized).
      kv_cache: paged KV cache with TPU-friendly shape.
      kv_lens: padded kv lengths. Only the first num_seqs values are valid.
      page_indices: flattened page indices look-up table by (seq_id, page_id).
      cu_q_lens: the cumulative sum of the effective query lengths. Similar to
        kv_lens, only the first num_seqs+1 values are valid.
      distribution: (i, j, k) represents that sequences[0:i] are decode-only,
        sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
        k is also the total number of sequences.
      use_causal_mask: if true, use causal mask.
      skip_kv_mask: only set to true if use_causal_mask=False and each dynamic
        kv_len % bkv_csz == 0. Set to true can improve performance.
      sm_scale: the softmax scale which will be applied to the Q@K^T.
      sliding_window: the sliding window size for the attention.
      soft_cap: the logit soft cap for the attention.
      out_dtype: the dtype of the output and the accumulator for matmul. Set
        lower for better performance, set higher for better accuracy. If None, it
        uses q.dtype.
      mask_value: mask value for causal mask.
      q_scale: the scale for the query.
      k_scale: the scale for the key.
      v_scale: the scale for the value.
      chunk_prefill_size: the chunk prefill size for the attention.
      d_block_sizes: the block sizes for the decode case.
      p_block_sizes: the block sizes for the prefill case.
      m_block_sizes: the block sizes for the mixed case.
      vmem_limit_bytes: the vmem limit for the pallas kernel.
      debug_mode: if true, RPA does not issue any DMAs or run flash attention but
        print debug info. Need to compile with `--xla_tpu_enable_log_recorder`.
      disable_bounds_checks: if true, disable bounds checks.
      disable_semaphore_checks: if true, disable semaphore checks.

    Returns:
      The output of the attention.
    """
    q, k, v = queries, keys, values
    tpu_version = get_tpu_version()

    if out_dtype is None:
        out_dtype = jnp.float32 if q.dtype == jnp.float32 else jnp.bfloat16

    if mask_value is None:
        # We do not set to -inf directly because (-inf) - (-inf) is nan.
        mask_value = cast(float, jnp.finfo(out_dtype).min)

    if vmem_limit_bytes is None:
        # TODO(jevinjiang, jacobplatin): change this to use
        # `get_vmem_estimate_bytes` when VREG spilling is fixed.
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
        use_causal_mask=use_causal_mask,
        skip_kv_mask=skip_kv_mask,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        out_dtype=out_dtype,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        d_block_sizes=d_block_sizes,
        p_block_sizes=p_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    actual_num_q_heads = q.shape[1]
    actual_head_dim = q.shape[2]
    actual_num_kv_heads = k.shape[1]

    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q, kv = prepare_inputs(q, k, v, fuse_non_tiling_axis_swap=False)
    (
        _,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    page_size = kv_cache.shape[1]
    num_kv_heads_x2_per_kv_packing = kv_cache.shape[2]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing

    scalar_prefetches = staging.make_scalar_prefetches(
        kv_lens, page_indices, cu_q_lens, distribution
    )
    num_sp = len(scalar_prefetches)
    has_prefill_pass = chunk_prefill_size is not None

    def run_rpa_kernel(
        q,
        kv_cache,
        *,
        bq_sz,
        bkv_sz,
        bq_csz,
        bkv_csz,
        static_q_len=None,
        case: RpaCase = RpaCase.MIXED,
    ):
        scratch_shapes = staging.make_base_rpa_scratch_shapes(
            q=q,
            kv_tail_shape=kv_cache.shape[3:],
            kv_dtype=kv_cache.dtype,
            out_dtype=out_dtype,
            actual_num_kv_heads=actual_num_kv_heads,
            num_q_heads_per_kv_head=num_q_heads_per_kv_head,
            head_dim=head_dim,
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            num_kv_heads_x2_per_kv_packing=num_kv_heads_x2_per_kv_packing,
        )
        scope_name = (
            f"RPA{case.symbol}-p_{page_size}-bq_{bq_sz}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}"
        )
        if sliding_window is not None:
            scope_name += f"-sw_{sliding_window}"
        return staging.invoke_rpa_pallas_kernel(
            kernel_fn=functools.partial(
                _ragged_paged_attention_kernel,
                use_causal_mask=use_causal_mask,
                skip_kv_mask=skip_kv_mask,
                skip_kv_update=skip_kv_update,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
                mask_value=mask_value,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_sz=bkv_sz,
                bq_csz=bq_csz,
                bkv_csz=bkv_csz,
                case=case,
                has_prefill_pass=has_prefill_pass,
                debug_mode=debug_mode,
            ),
            scalar_prefetches=scalar_prefetches,
            hbm_inputs=(q, kv, kv_cache),
            scratch_shapes=scratch_shapes,
            out_arrays=(q, kv_cache),
            input_output_aliases={num_sp: 0, num_sp + 2: 1},
            vmem_limit_bytes=cast(int, vmem_limit_bytes),
            scope_name=scope_name,
            tpu_version=tpu_version,
            disable_bounds_checks=disable_bounds_checks,
            disable_semaphore_checks=disable_semaphore_checks,
        )

    def _prepare_block_sizes(block_sizes, case):
        if block_sizes is None:
            return get_default_block_sizes(
                q.dtype,
                kv_cache.dtype,
                actual_num_q_heads,
                actual_num_kv_heads,
                head_dim,
                page_size,
                max_num_tokens,
                max_num_seqs,
                pages_per_seq,
                case=case,
            )
        return {
            "bq_sz": block_sizes[0],
            "bkv_sz": block_sizes[1],
            "bq_csz": block_sizes[2],
            "bkv_csz": block_sizes[3],
        }

    # Decode-only
    q, kv_cache = run_rpa_kernel(
        q,
        kv_cache,
        **_prepare_block_sizes(d_block_sizes, RpaCase.DECODE),
        static_q_len=1,
        case=RpaCase.DECODE,
    )
    if chunk_prefill_size is not None:
        # Prefill-only
        q, kv_cache = run_rpa_kernel(
            q,
            kv_cache,
            **_prepare_block_sizes(p_block_sizes, RpaCase.PREFILL),
            static_q_len=chunk_prefill_size,
            case=RpaCase.PREFILL,
        )
    # Mixed
    q, kv_cache = run_rpa_kernel(
        q,
        kv_cache,
        **_prepare_block_sizes(m_block_sizes, RpaCase.MIXED),
        static_q_len=None,
        case=RpaCase.MIXED,
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


@jax.jit(
    static_argnames=(
        "use_causal_mask",
        "skip_kv_mask",
        "skip_kv_update",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "out_dtype",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "d_block_sizes",
        "p_block_sizes",
        "m_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "disable_bounds_checks",
        "disable_semaphore_checks",
    ),
    donate_argnames=("queries", "keys", "values", "kv_cache_bundle"),
)
def ragged_paged_attention_bundled(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache_bundle: jax.Array,  # [num_pages, num_layers, page_size, K2/p, p, H]
    layer_idx: jax.Array,  # i32 scalar, dynamic
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    **kwargs,
):
    """Executes ragged paged attention directly against a bundled block-major
    KV cache in place.

    Flattens the multi-layer bundle (shape `[num_pages, num_layers, page_size, ...]`)
    into a flat page view (`[num_pages * num_layers, page_size, ...]`) via zero-copy
    metadata reshape. Logical page indices are remapped to target the specific
    layer's contiguous slots:

      flat_page = page * num_layers + layer_idx

    Because the bundle is laid out block-major (blocks outermost, layers second),
    element `bundle[page, layer]` maps directly to flat row `page * num_layers + layer`.
    The inner Pallas kernel writes exclusively into the target layer's slots.

    JAX buffer donation (`donate_argnames`) aliases the output directly to the input
    buffer, ensuring in-place updates without emitting `dynamic_update_slice` or
    incurring intermediate HBM memory traffic.
    """
    return staging.run_bundled_rpa(
        ragged_paged_attention,
        queries,
        keys,
        values,
        kv_cache_bundle,
        layer_idx,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        **kwargs,
    )
