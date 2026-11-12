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
"""Ragged Paged Attention v3 kernel for byte-packed compressed KV caches.

Shares `run_rpa_sequence_pipeline`, `StagingRefs`, `KvUpdateRing`, FlashAttention
step 1/2, scratch construction, and bundled dispatch with `v3.kernel` via
`v3.staging` and `v3.util`, specializing the KV cache to byte-packed pages
(`uint8[num_pages, num_kv_heads, page_words, 128]` or 5D block-major
`uint8[num_pages, num_layers, num_kv_heads, page_words, 128]`) with codec-agnostic
VREG hooks (`DequantPageFn`, `UpdatePageFn`).

Numeric contract (`DECODE` vs `MIXED`):
  - `DECODE` (`[0, distribution[0])`) updates the compressed VMEM page first and
    attends through `dequant_page_fn`, avoiding a raw-KV HBM fetch/branch per token.
  - `MIXED` (`[distribution[0], distribution[2])`, prefill + spec-decode verify)
    attends to historical tokens through `dequant_page_fn` and to the current step's
    `[kv_len - q_len, kv_len)` tokens from raw `keys`/`values` while writing updated
    compressed pages back, so `q_len == 1` decode and `q_len > 1` verify differ by
    quantization noise on the newest token.

Tensor-parallel `shard_map` integration (`layers/adapter/turboquant_attention.py`):
  Partition `queries`/`keys`/`values` on axis 1, `packed_new_kv` on axis 2, 4D
  `kv_cache` on axis 1 (5D `kv_cache_bundle` on axis 2), and replicate `kv_lens`,
  `page_indices`, `cu_q_lens`, `distribution`, and `layer_idx`.
"""

import functools
from collections.abc import Sequence
from typing import Any, Protocol, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import flash_attention, staging
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    get_default_block_sizes,
    get_vmem_estimate_bytes,
    prepare_inputs,
    prepare_outputs,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import (
    align_to,
    async_copy,
    broadcast_minor,
    cdiv,
    get_dtype_packing,
    get_span_int_dtype,
    get_tpu_version,
    next_power_of_2,
    validate_common_inputs,
)
from vllm_torchtpu.logger import init_logger

RpaCase = staging.RpaCase
logger = init_logger(__name__)

# Reserve 25% of VMEM for compiler spill slots, DMA/scalar-prefetch descriptors,
# and alignment slack across the 8-9 live VMEM scratch buffers.
_VMEM_HEADROOM_RATIO = 0.75

_STATIC_ARGNAMES = (
    "skip_kv_update",
    "sm_scale",
    "sliding_window",
    "soft_cap",
    "d_block_sizes",
    "m_block_sizes",
    "vmem_limit_bytes",
    "fuse_non_tiling_axis_swap",
    "page_size",
    "dequant_page_fn",
    "update_page_fn",
)


class DequantPageFn(Protocol):
    """VREG hook dequantizing `bkv_csz // page_size` `uint8[page_words, 128]` page refs
    into `(k, v)` (or `(k, v, k_norm_row)` when `return_k_norm=True`)."""

    def __call__(
        self,
        page_refs: Sequence[Any],
        kv_dtype: Any,
        *,
        return_k_norm: bool = False,
    ) -> tuple[jax.Array, ...]: ...


class UpdatePageFn(Protocol):
    """VREG hook merging `num_new_words` `uint8` slices into `old_page`."""

    def __call__(
        self,
        old_page: jax.Array,
        new_words_for_head: Sequence[jax.Array],
        valid_mask: jax.Array,
        *,
        slot_idx: jax.Array | None = None,
    ) -> jax.Array: ...


def align_second_minor_sublanes(rows: int, dtype) -> int:
    """Align `shape[-2]` to Mosaic's TPU VMEM sublane tile for `dtype`."""
    assert rows > 0
    packing = get_dtype_packing(dtype)
    full_tile = 8 * packing
    tile = full_tile if rows >= full_tile else max(packing, next_power_of_2(rows))
    return align_to(rows, tile)


def _mask_v_tile(v, *, processed_kv_len, effective_kv_len, span_int_ty):
    """Zero out value rows at or past `effective_kv_len` across 128-lane folds."""
    v_span_128 = processed_kv_len.astype(span_int_ty) + lax.broadcasted_iota(
        span_int_ty, (v.shape[0], 128), 0
    )
    v_valid = broadcast_minor(
        v_span_128 < effective_kv_len.astype(span_int_ty), v.shape
    )
    return jnp.where(v_valid, v, 0.0)


def _compressed_step1_qk_softmax(
    q,
    k,
    l_ref,
    m_ref,
    *,
    processed_q_len,
    processed_kv_len,
    effective_kv_len,
    num_q_heads_per_kv_head: int,
    sm_scale: float,
    out_dtype,
    mask_value: float,
    span_int_ty,
    max_kv_len: int,
    use_causal_mask: bool,
    skip_tile_mask: bool,
    sliding_window: int | None,
    soft_cap: float | None,
    k_norm_row=None,
):
    """Compute `Q @ K^T` with optional per-token `k_norm_row` and `skip_tile_mask`."""
    s_f32 = jnp.matmul(q, k.T, preferred_element_type=jnp.float32)
    s = (
        (s_f32 * (k_norm_row * sm_scale)).astype(out_dtype)
        if k_norm_row is not None
        else s_f32.astype(out_dtype) * sm_scale
    )
    if soft_cap is not None:
        s = soft_cap * jnp.tanh(s / soft_cap)

    if not skip_tile_mask:
        k_span = processed_kv_len.astype(span_int_ty) + lax.broadcasted_iota(
            span_int_ty, s.shape, 1
        )
        mask = k_span < effective_kv_len.astype(span_int_ty)
        apply_sw = (
            sliding_window is not None
            and sliding_window < max_kv_len
            and sliding_window <= jnp.iinfo(span_int_ty).max
        )
        if use_causal_mask or apply_sw:
            q_span = processed_q_len.astype(span_int_ty) + (
                lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
            ).astype(span_int_ty)
            if use_causal_mask:
                mask = mask & (q_span >= k_span)
            if apply_sw:
                assert sliding_window is not None
                mask = mask & ((q_span - k_span) < sliding_window)
        s = jnp.where(mask, s, mask_value)

    s_rowmax = jnp.max(s, axis=1, keepdims=True)
    m_prev = m_ref[...]
    m_curr = jnp.maximum(m_prev, s_rowmax)
    m_ref[...] = m_curr
    p = jnp.exp(s - broadcast_minor(m_curr, s.shape))
    p_rowsum = jnp.sum(p, axis=1, keepdims=True, dtype=out_dtype)
    exp_m_diff = jnp.exp(m_prev - m_curr)
    l_ref[...] = exp_m_diff * l_ref[...] + p_rowsum
    return p, exp_m_diff


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
    has_packed_new_kv: bool = False,
    **kwargs,
):
    start_seq_idx, end_seq_idx = kwargs["case"].get_range(
        distribution_ref, has_prefill_pass=False
    )
    if not has_packed_new_kv:
        rest_args = (None, *rest_args)

    @pl.loop(start_seq_idx, end_seq_idx)
    def _(seq_idx):
        return _ragged_paged_attention_kernel_loop(
            seq_idx,
            start_seq_idx,
            end_seq_idx,
            kv_lens_ref,
            page_indices_ref,
            cu_q_lens_ref,
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
    start_seq_idx,
    end_seq_idx,
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    sem_ids_ref,
    bo_ids_ref,
    bkv_update_ids_ref,
    q_hbm_ref,
    kv_hbm_ref,
    kv_cache_hbm_ref,  # u8[total_num_pages, actual_num_kv_heads, page_words, 128]
    packed_new_kv_hbm_ref,  # u8[max_num_tokens, num_new_words, num_kv_heads, 128]
    o_hbm_ref,
    updated_kv_cache_hbm_ref,
    bkv_x2_ref,
    bq_x2_ref,
    bo_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    bcompressed_x2_ref,  # u8[2, bkv_p, actual_num_kv_heads, page_words, 128]
    bpacked_new_x2_ref=None,
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
    page_size: int,
    dequant_page_fn: DequantPageFn,
    update_page_fn: UpdatePageFn | None = None,
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
    _, num_kv_heads_x2_per_kv_packing, _, _ = kv_hbm_ref.shape
    q_dtype, kv_dtype = q_hbm_ref.dtype, kv_hbm_ref.dtype
    num_page_indices = page_indices_ref.shape[0]
    pages_per_seq = num_page_indices // kv_lens_ref.shape[0]
    max_kv_len = pages_per_seq * page_size
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    bkv_p = bkv_sz // page_size
    span_int_ty = get_span_int_dtype(q_dtype, pages_per_seq, page_size, bkv_csz)

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait: bool = False):
        sem = sems.at[staging.SEM_BKV_FETCH, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx, :, :num_kv_heads_x2_per_kv_packing]
        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - cu_q_lens_ref[seq_idx]

        kv_left = kv_len - kv_len_start
        kv_left_frm_cache = (
            kv_left if skip_kv_update else jnp.maximum(kv_left - q_len, 0)
        )
        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        page_indices_offset = seq_idx * pages_per_seq + bkv_idx * bkv_p

        has_new = bpacked_new_x2_ref is not None
        cached_bkv_tokens = (
            bkv_sz_frm_cache + bkv_sz_frm_new
            if (has_new and is_decode)
            else bkv_sz_frm_cache
        )
        new_off = jnp.int32(0) if is_decode else bkv_sz_frm_cache
        if not wait:
            kv_update_ring.wait(bkv_sem_idx)
            for i in range(bkv_p):
                has_page = jnp.where(cached_bkv_tokens > i * page_size, 1, 0)
                p_idx = page_indices_ref[
                    jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                ]
                async_copy(
                    kv_cache_hbm_ref.at[pl.ds(p_idx, has_page)],
                    bcompressed_x2_ref.at[bkv_sem_idx, pl.ds(i, has_page)],
                    sem,
                    wait=False,
                )
            new_start = q_end - kv_left_frm_new
            if not is_decode and has_new:
                async_copy(
                    kv_hbm_ref.at[pl.ds(new_start, bkv_sz_frm_new)],
                    vmem_ref.at[pl.ds(new_off, bkv_sz_frm_new)],
                    sem,
                    wait=False,
                )
            if has_new:
                assert packed_new_kv_hbm_ref is not None
                async_copy(
                    packed_new_kv_hbm_ref.at[pl.ds(new_start, bkv_sz_frm_new)],
                    bpacked_new_x2_ref.at[bkv_sem_idx, pl.ds(new_off, bkv_sz_frm_new)],
                    sem,
                    wait=False,
                )
        else:
            comp_dst = bcompressed_x2_ref.at[
                bkv_sem_idx, pl.ds(0, cdiv(cached_bkv_tokens, page_size))
            ]
            async_copy(src=comp_dst, dst=comp_dst, sem=sem, wait=True)
            if not is_decode and has_new:
                new_dst = vmem_ref.at[pl.ds(new_off, bkv_sz_frm_new)]
                async_copy(src=new_dst, dst=new_dst, sem=sem, wait=True)
            if has_new:
                packed_new_dst = bpacked_new_x2_ref.at[
                    bkv_sem_idx, pl.ds(new_off, bkv_sz_frm_new)
                ]
                async_copy(src=packed_new_dst, dst=packed_new_dst, sem=sem, wait=True)
        return (
            kv_len_start + bkv_sz_frm_cache,
            bkv_sz_frm_new if has_new else jnp.int32(0),
        )

    def _update_kv_cache(
        seq_idx, bkv_sem_idx, offset, update_sz, *, wait: bool = False
    ):
        if bpacked_new_x2_ref is None or update_page_fn is None:
            return
        sem = sems.at[staging.SEM_BKV_UPDATE, bkv_sem_idx]
        kv_p_start = offset // page_size
        num_touched = (
            1 if is_decode else (cdiv(offset + update_sz, page_size) - kv_p_start)
        )
        if wait:
            dst = updated_kv_cache_hbm_ref.at[pl.ds(0, num_touched)]
            async_copy(src=dst, dst=dst, sem=sem, wait=True)
            return

        bkv_id = offset // bkv_sz
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start
        num_new_words = bpacked_new_x2_ref.shape[2]
        slot_iota = lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)

        if is_decode:
            valid_mask = slot_iota == ignore
            new_words = [
                bpacked_new_x2_ref.at[bkv_sem_idx, 0, w][...]
                for w in range(num_new_words)
            ]
            for h in range(actual_num_kv_heads):
                old_p = bcompressed_x2_ref.at[bkv_sem_idx, p_ignore, h][...]
                bcompressed_x2_ref.at[bkv_sem_idx, p_ignore, h][...] = update_page_fn(
                    old_p,
                    [w_arr[h : h + 1, :] for w_arr in new_words],
                    valid_mask,
                    slot_idx=ignore,
                )
            async_copy(
                bcompressed_x2_ref.at[bkv_sem_idx, pl.ds(p_ignore, 1)],
                updated_kv_cache_hbm_ref.at[
                    pl.ds(page_indices_ref[page_indices_offset], 1)
                ],
                sem,
                wait=False,
            )
            return

        local_new_start = offset - bkv_id * bkv_sz
        local_new_end = local_new_start + update_sz

        def update_body(i, _):
            assert bpacked_new_x2_ref is not None
            p_local = p_ignore + i
            tok_pos = p_local * page_size + slot_iota
            valid_mask = (tok_pos >= local_new_start) & (tok_pos < local_new_end)
            p_ds = pl.ds(p_local * page_size, page_size)
            # Load each `[page_size, actual_num_kv_heads, 128]` word block once.
            new_words = [
                bpacked_new_x2_ref.at[bkv_sem_idx, p_ds, w][...]
                for w in range(num_new_words)
            ]
            for h in range(actual_num_kv_heads):
                old_p = jnp.where(
                    (p_local * page_size) < local_new_start,
                    bcompressed_x2_ref.at[bkv_sem_idx, p_local, h][...],
                    jnp.uint8(0),
                )
                bcompressed_x2_ref.at[bkv_sem_idx, p_local, h][...] = update_page_fn(
                    old_p, [w_arr[:, h, :] for w_arr in new_words], valid_mask
                )
            async_copy(
                bcompressed_x2_ref.at[bkv_sem_idx, pl.ds(p_local, 1)],
                updated_kv_cache_hbm_ref.at[
                    pl.ds(page_indices_ref[page_indices_offset + i], 1)
                ],
                sem,
                wait=False,
            )
            return None

        lax.fori_loop(0, num_touched, update_body, None, unroll=False)

    staging_refs = staging.StagingRefs(
        q_hbm_ref, o_hbm_ref, bq_x2_ref, bo_x2_ref, cu_q_lens_ref, bo_ids_ref, sems
    )
    kv_update_ring = staging.KvUpdateRing(bkv_update_ids_ref, _update_kv_cache)

    def _load_bkv_raw(bkv_sem_idx, kv_head_idx, start, sz):
        return staging.load_bkv(bkv_x2_ref, bkv_sem_idx, kv_head_idx, start, sz)

    def _load_bkv_dequant(
        bkv_sem_idx, kv_head_idx, start, sz, *, return_k_norm: bool = False
    ):
        start_page = start // page_size
        refs = [
            bcompressed_x2_ref.at[bkv_sem_idx, start_page + p_off, kv_head_idx]
            for p_off in range(sz // page_size)
        ]
        return dequant_page_fn(refs, kv_dtype, return_k_norm=return_k_norm)

    def _load_bkv_mixed(bkv_sem_idx, kv_head_idx, start, sz, bkv_sz_frm_cache):
        row_idx_128 = start + lax.broadcasted_iota(jnp.int32, (sz, 128), 0)
        from_cache = broadcast_minor(row_idx_128 < bkv_sz_frm_cache, (sz, head_dim))
        bk_new, bv_new = _load_bkv_raw(bkv_sem_idx, kv_head_idx, start, sz)
        bk_comp, bv_comp = _load_bkv_dequant(bkv_sem_idx, kv_head_idx, start, sz)
        return (
            jnp.where(from_cache, bk_comp, bk_new),
            jnp.where(from_cache, bv_comp, bv_new),
        )

    def compute_bkv_attention(ctx: staging.BkvStepContext) -> None:
        num_loops = cdiv(ctx.effective_bkv_sz, bkv_csz)
        preloaded_bq = (
            [
                staging.load_bq(bq_x2_ref, ctx.bq_sem_idx, h, 0, ctx.actual_bq_csz)
                for h in range(actual_num_kv_heads)
            ]
            if is_decode
            else None
        )

        def _run_compressed_tile(idx, load_fn, *, skip_tile_mask: bool = False):
            bkv_start = idx * bkv_csz
            cur_processed_kv_len = ctx.processed_kv_len + bkv_start
            prev_lm_slice = None
            prev_p = None
            prev_v = None
            prev_exp_m_diff = None

            # Keep `kv_head_idx` outer to amortize VREG page dequant across `bq_start`,
            # deferring `step2_pv` by one inner step to overlap MXU `P @ V` with VPU.
            for kv_head_idx in range(actual_num_kv_heads):
                if load_fn is _load_bkv_dequant:
                    bk_c, bv_c, k_norm_row = _load_bkv_dequant(
                        ctx.bkv_sem_idx,
                        kv_head_idx,
                        bkv_start,
                        bkv_csz,
                        return_k_norm=True,
                    )
                else:
                    bk_c, bv_c = load_fn(
                        ctx.bkv_sem_idx, kv_head_idx, bkv_start, bkv_csz
                    )
                    k_norm_row = None
                if not skip_tile_mask:
                    bv_c = _mask_v_tile(
                        bv_c,
                        processed_kv_len=cur_processed_kv_len,
                        effective_kv_len=ctx.effective_kv_len,
                        span_int_ty=span_int_ty,
                    )

                for bq_start in range(0, ctx.actual_bq_sz, ctx.actual_bq_csz):
                    bq_c = (
                        preloaded_bq[kv_head_idx]
                        if preloaded_bq is not None
                        else staging.load_bq(
                            bq_x2_ref,
                            ctx.bq_sem_idx,
                            kv_head_idx,
                            bq_start,
                            ctx.actual_bq_csz,
                        )
                    )
                    lm_slice = (
                        kv_head_idx,
                        pl.ds(
                            bq_start * num_q_heads_per_kv_head,
                            ctx.actual_bq_csz * num_q_heads_per_kv_head,
                        ),
                    )
                    cur_p, cur_exp_m_diff = _compressed_step1_qk_softmax(
                        bq_c,
                        bk_c,
                        l_ref.at[*lm_slice],
                        m_ref.at[*lm_slice],
                        processed_q_len=ctx.processed_q_len + bq_start,
                        processed_kv_len=cur_processed_kv_len,
                        effective_kv_len=ctx.effective_kv_len,
                        num_q_heads_per_kv_head=num_q_heads_per_kv_head,
                        sm_scale=sm_scale,
                        out_dtype=out_dtype,
                        mask_value=mask_value,
                        span_int_ty=span_int_ty,
                        max_kv_len=max_kv_len,
                        use_causal_mask=not is_decode,
                        skip_tile_mask=skip_tile_mask,
                        sliding_window=sliding_window,
                        soft_cap=soft_cap,
                        k_norm_row=k_norm_row,
                    )
                    if prev_lm_slice is not None:
                        flash_attention.flash_attention_step2_pv(
                            prev_p,
                            prev_v,
                            prev_exp_m_diff,
                            acc_ref.at[*prev_lm_slice],
                            out_dtype=out_dtype,
                        )
                    prev_lm_slice, prev_p, prev_v, prev_exp_m_diff = (
                        lm_slice,
                        cur_p,
                        bv_c,
                        cur_exp_m_diff,
                    )

            assert prev_lm_slice is not None
            flash_attention.flash_attention_step2_pv(
                prev_p,
                prev_v,
                prev_exp_m_diff,
                acc_ref.at[*prev_lm_slice],
                out_dtype=out_dtype,
            )

        can_skip_full_mask = sliding_window is None
        if not (skip_kv_update or is_decode):
            bkv_sz_frm_cache = jnp.maximum(ctx.offset - ctx.bkv_idx * bkv_sz, 0)
            num_dequant_loops = jnp.minimum(bkv_sz_frm_cache // bkv_csz, num_loops)
            raw_start_loop = jnp.minimum(cdiv(bkv_sz_frm_cache, bkv_csz), num_loops)

            @pl.loop(0, num_dequant_loops, unroll=False)
            def _(idx):
                _run_compressed_tile(
                    idx, _load_bkv_dequant, skip_tile_mask=can_skip_full_mask
                )

            @pl.when(num_dequant_loops < raw_start_loop)
            def _():
                _run_compressed_tile(
                    num_dequant_loops,
                    lambda sem, h, s, sz: _load_bkv_mixed(
                        sem, h, s, sz, bkv_sz_frm_cache
                    ),
                )

            @pl.loop(raw_start_loop, num_loops, unroll=False)
            def _(idx):
                _run_compressed_tile(idx, _load_bkv_raw)
        else:
            num_full_loops = (
                ctx.effective_bkv_sz // bkv_csz
                if is_decode
                else jnp.minimum(
                    jnp.maximum(ctx.kv_q_gap - ctx.bkv_idx * bkv_sz, 0) // bkv_csz,
                    ctx.effective_bkv_sz // bkv_csz,
                )
            )

            @pl.loop(0, num_full_loops, unroll=False)
            def _(idx):
                _run_compressed_tile(
                    idx, _load_bkv_dequant, skip_tile_mask=can_skip_full_mask
                )

            if is_decode:

                @pl.when(num_full_loops < num_loops)
                def _():
                    _run_compressed_tile(num_full_loops, _load_bkv_dequant)
            else:

                @pl.loop(num_full_loops, num_loops, unroll=False)
                def _(idx):
                    _run_compressed_tile(idx, _load_bkv_dequant)

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
        use_causal_mask=not is_decode,
        sliding_window=sliding_window,
        static_q_len=1 if is_decode else None,
    )


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
    page_size: int | None,
    dequant_page_fn: DequantPageFn | None,
    packed_new_kv: jax.Array | None = None,
    update_page_fn: UpdatePageFn | None = None,
    skip_kv_update: bool = False,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    fuse_non_tiling_axis_swap: bool = False,
) -> None:
    """Validate inputs to the compressed-KV RPA v3 kernel at trace/compile time."""
    if fuse_non_tiling_axis_swap:
        raise NotImplementedError(
            "fuse_non_tiling_axis_swap=True is not enabled for compressed KV RPA v3."
        )
    if page_size is None or page_size <= 0:
        raise ValueError(f"Expected positive {page_size=}.")
    if dequant_page_fn is None:
        raise ValueError("Expected non-None dequant_page_fn.")

    q, k, v = queries, keys, values
    kv_packing = (
        get_dtype_packing(k.dtype) if jnp.issubdtype(k.dtype, jnp.floating) else 1
    )
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
        block_sizes_by_name=(("decode", d_block_sizes), ("mixed", m_block_sizes)),
        vmem_limit_bytes=vmem_limit_bytes,
    )

    max_num_tokens, _, actual_head_dim = q.shape
    actual_num_kv_heads = k.shape[1]
    if actual_head_dim % 128 != 0:
        raise ValueError(
            f"Expected {actual_head_dim=} to be a multiple of 128 for compressed KV."
        )
    if (
        kv_cache.dtype != jnp.uint8
        or kv_cache.ndim != 4
        or kv_cache.shape[1] != actual_num_kv_heads
        or kv_cache.shape[2] <= 0
        or kv_cache.shape[3] != 128
    ):
        raise ValueError(
            f"Expected 4D uint8 compressed kv_cache (..., {actual_num_kv_heads},"
            f" page_words, 128), got {kv_cache.shape=}, {kv_cache.dtype=}."
        )
    if not skip_kv_update:
        if packed_new_kv is None or update_page_fn is None:
            raise ValueError(
                "packed_new_kv and update_page_fn are required unless skip_kv_update."
            )
        if (
            packed_new_kv.dtype != jnp.uint8
            or packed_new_kv.ndim != 4
            or packed_new_kv.shape[0] != max_num_tokens
            or packed_new_kv.shape[1] <= 0
            or packed_new_kv.shape[2:] != (actual_num_kv_heads, 128)
        ):
            raise ValueError(
                f"Expected 4D uint8 packed_new_kv ({max_num_tokens}, num_new_words,"
                f" {actual_num_kv_heads}, 128), got {packed_new_kv.shape=},"
                f" {packed_new_kv.dtype=}."
            )


def get_compressed_kv_vmem_estimate_bytes(
    *,
    bq_sz: int,
    bkv_sz: int,
    q_dtype,
    kv_dtype,
    actual_num_q_heads: int,
    actual_num_kv_heads: int,
    head_dim: int,
    page_size: int,
    page_words: int,
    num_new_words: int,
    case: RpaCase,
    skip_kv_update: bool = False,
) -> int:
    """Estimate total VMEM scratch bytes for `(bq_sz, bkv_sz)` with sublane padding."""
    is_decode = case == RpaCase.DECODE
    out_dtype = jnp.float32 if q_dtype == jnp.float32 else jnp.bfloat16
    raw_bkv_sz = 1 if (is_decode or skip_kv_update) else bkv_sz
    base_vmem = get_vmem_estimate_bytes(
        actual_num_kv_heads,
        actual_num_q_heads // actual_num_kv_heads,
        head_dim,
        bq_sz,
        raw_bkv_sz,
        q_dtype,
        kv_dtype,
        out_dtype=out_dtype,
    )
    # Mosaic pads `shape[-2]` of each VMEM scratch buffer to its sublane tile.
    aligned_words = align_second_minor_sublanes(page_words, jnp.uint8)
    bcompressed_vmem = (
        2 * (bkv_sz // page_size) * actual_num_kv_heads * aligned_words * 128
    )
    bpacked_new_vmem = (
        0
        if (skip_kv_update or num_new_words <= 0)
        else 2
        * (1 if is_decode else bkv_sz)
        * num_new_words
        * align_second_minor_sublanes(actual_num_kv_heads, jnp.uint8)
        * 128
    )
    return base_vmem + bcompressed_vmem + bpacked_new_vmem


def prepare_compressed_kv_block_sizes(
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
    vmem_limit_bytes: int,
    page_words: int = 0,
    num_new_words: int = 0,
    skip_kv_update: bool = False,
    tpu_version: int | None = None,
) -> dict[str, int]:
    """Select `(bq_sz, bkv_sz, bq_csz, bkv_csz)` bounded by `vmem_limit_bytes`."""
    if block_sizes is not None:
        return dict(zip(("bq_sz", "bkv_sz", "bq_csz", "bkv_csz"), block_sizes))
    if vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")
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
        tpu_version=tpu_version,
    )
    # VREG budget of one dequantized KV tile (`bk_c` + `bv_c` + unpacked nibbles +
    # codebook + softmax): TPU v7x fits `bkv_csz = 256` (`128 KiB` at `head_dim=128`)
    # when `page_size >= 64`; TPU v5/v6 caps at `128` (`64` if `head_dim > 128` and
    # `page_size < 64`). Cap `bkv_sz` at `max(32 pages, 512)` since compressed `uint8`
    # pages are ~4x smaller in HBM and issue one DMA descriptor per page.
    target_csz = (
        256
        if (tpu_version >= 7 and page_size >= 64)
        else (128 if (head_dim <= 128 or page_size >= 64) else 64)
    )
    bkv_csz = align_to(min(sizes["bkv_csz"], max(page_size, target_csz)), page_size)
    bkv_sz = align_to(min(sizes["bkv_sz"], max(page_size * 32, 512)), bkv_csz)
    bq_sz, bq_csz = sizes["bq_sz"], min(sizes["bq_csz"], sizes["bq_sz"])

    limit_bytes = int(vmem_limit_bytes * _VMEM_HEADROOM_RATIO)
    est_kw = dict(
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        actual_num_q_heads=actual_num_q_heads,
        actual_num_kv_heads=actual_num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        page_words=page_words,
        num_new_words=num_new_words,
        case=case,
        skip_kv_update=skip_kv_update,
    )
    while (
        est_bytes := get_compressed_kv_vmem_estimate_bytes(
            bq_sz=bq_sz, bkv_sz=bkv_sz, **est_kw
        )
    ) > limit_bytes:
        bkv_only = get_compressed_kv_vmem_estimate_bytes(
            bq_sz=1, bkv_sz=bkv_sz, **est_kw
        )
        bq_only = get_compressed_kv_vmem_estimate_bytes(
            bq_sz=bq_sz, bkv_sz=page_size, **est_kw
        )
        if bkv_sz > page_size and (bkv_only >= bq_only or bq_sz == 1):
            bkv_sz = max(page_size, align_to(bkv_sz // 2, page_size))
        elif bq_sz > 1:
            bq_sz = max(1, bq_sz // 2)
        else:
            lookup_key = (
                f"tpu_v{tpu_version}",
                case.symbol,
                str(jnp.dtype(q_dtype)),
                str(jnp.dtype(kv_dtype)),
                actual_num_q_heads,
                actual_num_kv_heads,
                head_dim,
                page_size,
                page_words,
                num_new_words,
            )
            logger.warning_once(
                "Minimum compressed-KV RPA v3 block sizes (bq_sz=%d, bkv_sz=%d)"
                " estimate %d B > %d B VMEM budget (limit=%d B) for %s.",
                bq_sz,
                bkv_sz,
                est_bytes,
                limit_bytes,
                vmem_limit_bytes,
                lookup_key,
            )
            break

    while bkv_csz > bkv_sz or bkv_sz % bkv_csz != 0:
        bkv_csz = max(page_size, align_to(bkv_csz // 2, page_size))
    while bq_csz > bq_sz or bq_sz % bq_csz != 0:
        bq_csz = max(1, bq_csz // 2)
    return {"bq_sz": bq_sz, "bkv_sz": bkv_sz, "bq_csz": bq_csz, "bkv_csz": bkv_csz}


@jax.jit(static_argnames=_STATIC_ARGNAMES, donate_argnames=("queries", "kv_cache"))
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
    page_size: int,
    dequant_page_fn: DequantPageFn,
    packed_new_kv: jax.Array | None = None,
    update_page_fn: UpdatePageFn | None = None,
    skip_kv_update: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    fuse_non_tiling_axis_swap: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused Pallas RPA v3 attention over a byte-packed compressed KV cache.

    `queries` and `kv_cache` are marked in `donate_argnames` (`kv_cache` is updated
    in place via `input_output_aliases`; `keys`/`values` are never donated). See
    module docstring for the `DECODE` vs `MIXED` numeric contract and `shard_map`.

    Args:
      queries: `floating[max_num_tokens, num_q_heads, head_dim]`, `head_dim % 128 == 0`.
      keys: `floating[max_num_tokens, num_kv_heads, head_dim]`.
      values: `floating[max_num_tokens, num_kv_heads, head_dim]`, matching `keys.dtype`.
      kv_cache: `uint8[total_num_pages, num_kv_heads, page_words, 128]`.
      kv_lens: `int32[max_num_seqs]` effective KV lengths for `[0, distribution[2])`.
      page_indices: `int32[max_num_seqs * pages_per_seq]` flattened block table.
      cu_q_lens: `int32[max_num_seqs + 1]` cumulative query token offsets.
      distribution: `int32[3]` `(decode_end, prefill_end, total_seqs)`.
      page_size: Tokens stored per compressed page (`page_size % kv_packing == 0`).
      dequant_page_fn: Hashable static `DequantPageFn` dequantizing VMEM page refs.
      packed_new_kv: `uint8[max_num_tokens, num_new_words, num_kv_heads, 128]`
        per-token compressed payload (required when `skip_kv_update=False`).
      update_page_fn: Hashable static `UpdatePageFn` merging `packed_new_kv` into a
        page in VREGs (required when `skip_kv_update=False`).
      skip_kv_update: If `True`, reads all `[0, kv_len)` tokens from `kv_cache`
        without reading `keys`/`values` or writing back.
      sm_scale: Softmax scale applied to `Q @ K^T`.
      sliding_window: Optional causal sliding-window span in tokens.
      soft_cap: Optional `tanh` logit soft cap.
      d_block_sizes: Optional `(bq_sz, bkv_sz, bq_csz, bkv_csz)` override for `DECODE`.
      m_block_sizes: Optional `(bq_sz, bkv_sz, bq_csz, bkv_csz)` override for `MIXED`.
      vmem_limit_bytes: Per-core VMEM limit override; defaults to chip capacity.
      fuse_non_tiling_axis_swap: Reserved token-major DMA flag (must be `False`).

    Returns:
      `(outputs, updated_kv_cache)` matching `(queries, kv_cache)` shapes and dtypes.

    Raises:
      ValueError: On invalid ranks, dtypes, `head_dim % 128`, metadata, or block sizes.
      NotImplementedError: If `fuse_non_tiling_axis_swap=True` or `tpu_version < 5`.
    """
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
        page_size=page_size,
        dequant_page_fn=dequant_page_fn,
        packed_new_kv=packed_new_kv,
        update_page_fn=update_page_fn,
        skip_kv_update=skip_kv_update,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        d_block_sizes=d_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        fuse_non_tiling_axis_swap=fuse_non_tiling_axis_swap,
    )

    actual_num_q_heads, actual_head_dim = q.shape[1], q.shape[2]
    actual_num_kv_heads = k.shape[1]
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q, kv = prepare_inputs(q, k, v, fuse_non_tiling_axis_swap=False)
    _, max_num_tokens, num_q_per_kv_packed, q_packing, head_dim = q.shape
    num_kv_heads_x2_per_kv_packing = kv.shape[1]
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_q_heads_per_kv_head = num_q_per_kv_packed * q_packing
    page_words = kv_cache.shape[2]
    has_packed_new_kv = (not skip_kv_update) and (packed_new_kv is not None)
    num_new_words = (
        packed_new_kv.shape[1]
        if (has_packed_new_kv and packed_new_kv is not None)
        else 0
    )
    scalar_prefetches = staging.make_scalar_prefetches(
        kv_lens, page_indices, cu_q_lens, distribution
    )
    num_sp = len(scalar_prefetches)

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
        raw_bkv_sz = 1 if (is_decode or skip_kv_update) else bkv_sz
        scratch_shapes = staging.make_base_rpa_scratch_shapes(
            q=q,
            kv_tail_shape=kv.shape[2:],
            kv_dtype=kv.dtype,
            out_dtype=out_dtype,
            actual_num_kv_heads=actual_num_kv_heads,
            num_q_heads_per_kv_head=num_q_heads_per_kv_head,
            head_dim=head_dim,
            bq_sz=bq_sz,
            bkv_sz=raw_bkv_sz,
            num_kv_heads_x2_per_kv_packing=num_kv_heads_x2_per_kv_packing,
        )
        scratch_shapes.append(
            pltpu.VMEM(
                (2, bkv_sz // page_size, actual_num_kv_heads, page_words, 128),
                jnp.uint8,
            )
        )
        hbm_inputs: tuple[jax.Array, ...] = (q, kv, kv_cache)
        if has_packed_new_kv:
            assert packed_new_kv is not None
            hbm_inputs = (*hbm_inputs, packed_new_kv)
            scratch_shapes.append(
                pltpu.VMEM(
                    (
                        2,
                        1 if is_decode else bkv_sz,
                        num_new_words,
                        actual_num_kv_heads,
                        128,
                    ),
                    jnp.uint8,
                )
            )
        scope_name = (
            f"RPA-CKV-{case.symbol}-p_{page_size}-bq_{bq_sz}_{bq_csz}-bkv_{bkv_sz}_{bkv_csz}"
            + (f"-sw_{sliding_window}" if sliding_window is not None else "")
        )
        return staging.invoke_rpa_pallas_kernel(
            kernel_fn=functools.partial(
                _ragged_paged_attention_kernel,
                has_packed_new_kv=has_packed_new_kv,
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
                page_size=page_size,
                dequant_page_fn=dequant_page_fn,
                update_page_fn=update_page_fn,
            ),
            scalar_prefetches=scalar_prefetches,
            hbm_inputs=hbm_inputs,
            scratch_shapes=scratch_shapes,
            out_arrays=(q, kv_cache),
            input_output_aliases={num_sp: 0, num_sp + 2: 1},
            vmem_limit_bytes=cast(int, vmem_limit_bytes),
            scope_name=scope_name,
            tpu_version=tpu_version,
        )

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
        vmem_limit_bytes=cast(int, vmem_limit_bytes),
        page_words=page_words,
        num_new_words=num_new_words,
        skip_kv_update=skip_kv_update,
        tpu_version=tpu_version,
    )
    for case, bs in ((RpaCase.DECODE, d_block_sizes), (RpaCase.MIXED, m_block_sizes)):
        q, kv_cache = run_rpa_kernel(
            q,
            kv_cache,
            **prepare_compressed_kv_block_sizes(bs, case=case, **block_size_kw),
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


@jax.jit(
    static_argnames=_STATIC_ARGNAMES, donate_argnames=("queries", "kv_cache_bundle")
)
def ragged_paged_attention_bundled(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache_bundle: jax.Array,  # u8[num_pages, num_layers, kv_heads, page_words, 128]
    layer_idx: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    page_size: int,
    dequant_page_fn: DequantPageFn,
    packed_new_kv: jax.Array | None = None,
    update_page_fn: UpdatePageFn | None = None,
    skip_kv_update: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    fuse_non_tiling_axis_swap: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Execute compressed-KV RPA v3 against a 5D block-major bundled KV cache."""
    if kv_cache_bundle.ndim != 5:
        raise ValueError(
            "Expected 5D uint8 compressed kv_cache_bundle"
            " (num_pages, num_layers, num_kv_heads, page_words, 128),"
            f" got {kv_cache_bundle.shape=}, {kv_cache_bundle.dtype=}."
        )
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
        page_size=page_size,
        dequant_page_fn=dequant_page_fn,
        packed_new_kv=packed_new_kv,
        update_page_fn=update_page_fn,
        skip_kv_update=skip_kv_update,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        d_block_sizes=d_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        fuse_non_tiling_axis_swap=fuse_non_tiling_axis_swap,
    )
