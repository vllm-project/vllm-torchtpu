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
`v3.staging`, `v3.pipeline`, `v3.launch`, `v3.flash_attention`, and `v3.util`,
specializing the KV cache to byte-packed pages
(`uint8[num_pages, num_kv_heads, page_words, 128]` or 5D block-major
`uint8[num_pages, num_layers, num_kv_heads, page_words, 128]`) parameterized by
a frozen `CompressedKvCodec` (`DequantPageFn`, `UpdatePageFn`).

Numeric contract (`DECODE` and `MIXED`):
  Both `DECODE` (`[0, distribution[0])`) and `MIXED` (`[distribution[0],
  distribution[2])`, covering prefill and speculative-decoding verification)
  merge `packed_new_kv` into `bcompressed_x2_ref` in VMEM immediately after the
  fetch DMA completes (`fetch_bkv_fn(..., wait=True)`) and attend to all
  `[0, kv_len)` tokens exclusively through `codec.dequant_page_fn`. Decode and
  spec-decode verify steps therefore observe identical quantized K/V representations.

Tensor-parallel `shard_map` partitioning:
  Partition `queries` on axis 1, `packed_new_kv` on axis 2, 4D `kv_cache` on
  axis 1 (5D `kv_cache_bundle` on axis 2), and replicate `kv_lens`,
  `page_indices`, `cu_q_lens`, `distribution`, and `layer_idx`.
"""

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.pallas_util import (
    align_to,
    async_copy,
    cdiv,
    get_dtype_packing,
    get_tpu_info_for_version,
    get_tpu_version,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3 import (
    flash_attention,
    launch,
    pipeline,
    staging,
    util,
)
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    get_default_block_sizes,
    prepare_outputs,
    prepare_q,
)
from vllm_torchtpu.logger import init_logger

RpaCase = util.RpaCase
logger = init_logger(__name__)

# Reserve 25% of VMEM for compiler spill slots, DMA/scalar-prefetch descriptors,
# and alignment slack across the live VMEM scratch buffers.
_VMEM_HEADROOM_RATIO = 0.75

_STATIC_ARGNAMES = (
    "codec",
    "kv_dtype",
    "skip_kv_update",
    "sm_scale",
    "sliding_window",
    "soft_cap",
    "d_block_sizes",
    "m_block_sizes",
    "vmem_limit_bytes",
    "disable_bounds_checks",
    "disable_semaphore_checks",
)


class DequantPageFn(Protocol):
    """VREG hook dequantizing `bkv_csz // page_size` `uint8[page_words, 128]` page refs.

    Output contract:
      Returns `(k, v, k_norm_row)` where `k` and `v` have shape
      `(bkv_csz, head_dim)` and dtype `kv_dtype`, and `k_norm_row` is either
      `None` (when `k` already incorporates its scale) or a per-token row of
      shape `(1, bkv_csz)`. When `k_norm_row` is not `None`, `k` MUST NOT
      already include the norm, because the kernel multiplies `Q @ K^T` scores
      by `k_norm_row`.
    """

    def __call__(
        self,
        page_refs: Sequence[Any],
        kv_dtype: Any,
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]: ...


class UpdatePageFn(Protocol):
    """VREG hook merging `num_new_words` `uint8` slices into `old_page`.

    Dual-mode calling contract:
      - `old_page`: `uint8[page_words, 128]` existing compressed page for one
        KV head (zeroed by the caller when initializing a fresh page).
      - `new_words_for_head`: Sequence of `num_new_words` `uint8` slices for the
        target KV head:
          * `DECODE` (`slot_idx` provided as an `int32` scalar in
            `[0, page_size)`): each element has shape `uint8[1, 128]` holding
            the single new token's packed words.
          * `MIXED` (`slot_idx is None`): each element has shape
            `uint8[page_size, 128]` aligned to the page's `[0, page_size)` token
            slots.
      - `valid_mask`: `bool[page_size, 1]` mask indicating which page slots are
        being updated in this call (`slot_iota == slot_idx` in `DECODE`, or the
        in-range new-token span in `MIXED`).
      - Invariant: Implementations MUST preserve all bytes belonging to page
        slots where `valid_mask` is `False` unchanged from `old_page`, and
        return the merged `uint8[page_words, 128]` page.
    """

    def __call__(
        self,
        old_page: jax.Array,
        new_words_for_head: Sequence[jax.Array],
        valid_mask: jax.Array,
        *,
        slot_idx: jax.Array | None = None,
    ) -> jax.Array: ...


@dataclasses.dataclass(frozen=True)
class CompressedKvCodec:
    """Frozen, hashable codec bundle for compressed-KV ragged paged attention.

    Attributes:
      dequant_page_fn: Hashable `DequantPageFn` VREG hook returning
        `(k, v, k_norm_row)`.
      update_page_fn: Hashable `UpdatePageFn` VREG hook merging `packed_new_kv`
        into a page (required when `skip_kv_update=False`; may be `None` when
        `skip_kv_update=True`).
      page_size_from_words: Hashable callable deriving `page_size` from
        `kv_cache`'s `page_words` dimension (`kv_cache.shape[-2]`).
      max_bkv_csz_fn: Optional hashable callable `(head_dim, page_size, *,
        tpu_version=...) -> int` returning the codec's VREG-calibrated `bkv_csz`
        upper bound.
    """

    dequant_page_fn: DequantPageFn
    update_page_fn: UpdatePageFn | None
    page_size_from_words: Callable[[int], int]
    max_bkv_csz_fn: Callable[..., int] | None = None


def _ragged_paged_attention_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    distribution_ref,
    sem_ids_ref,
    bo_ids_ref,
    bkv_update_ids_ref,
    q_hbm_ref,
    kv_cache_hbm_ref,
    *rest_args,
    skip_kv_update: bool = False,
    **kwargs,
):
    start_seq_idx, end_seq_idx = kwargs["case"].get_range(
        distribution_ref, has_prefill_pass=False
    )
    if skip_kv_update:
        packed_new_kv_hbm_ref = None
        tail_args = rest_args
    else:
        packed_new_kv_hbm_ref, *tail_list = rest_args
        tail_args = tuple(tail_list)

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
            kv_cache_hbm_ref,
            packed_new_kv_hbm_ref,
            *tail_args,
            skip_kv_update=skip_kv_update,
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
    kv_cache_hbm_ref,  # u8[total_num_pages, actual_num_kv_heads, page_words, 128]
    packed_new_kv_hbm_ref,  # u8[max_num_tokens, num_new_words, num_kv_heads, 128]
    o_hbm_ref,
    updated_kv_cache_hbm_ref,
    bcompressed_x2_ref,  # u8[2, bkv_p, actual_num_kv_heads, page_words, 128]
    bq_x2_ref,
    bo_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    bpacked_new_x2_ref=None,
    *,
    codec: CompressedKvCodec,
    kv_dtype,
    page_size: int,
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
    tpu_version: int | None = None,
):
    is_decode = case == RpaCase.DECODE
    out_dtype = acc_ref.dtype
    (
        actual_num_kv_heads,
        _,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        _,
    ) = q_hbm_ref.shape
    q_dtype = q_hbm_ref.dtype
    num_page_indices = page_indices_ref.shape[0]
    pages_per_seq = num_page_indices // kv_lens_ref.shape[0]
    max_kv_len = pages_per_seq * page_size
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    bkv_p = bkv_sz // page_size
    span_int_ty = util.get_span_int_dtype(
        q_dtype, pages_per_seq, page_size, bkv_csz, tpu_version=tpu_version
    )

    def _merge_bkv_pages_in_vmem(bkv_sem_idx, bkv_idx, offset, update_sz):
        assert bpacked_new_x2_ref is not None
        assert codec.update_page_fn is not None
        kv_p_start = offset // page_size
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_idx * bkv_p
        num_new_words = bpacked_new_x2_ref.shape[2]
        slot_iota = lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)

        if is_decode:
            valid_mask = slot_iota == ignore
            new_words = [
                bpacked_new_x2_ref.at[bkv_sem_idx, 0, w][...]
                for w in range(num_new_words)
            ]
            for h in range(actual_num_kv_heads):
                old_p = jnp.where(
                    ignore == 0,
                    jnp.uint8(0),
                    bcompressed_x2_ref.at[bkv_sem_idx, p_ignore, h][...],
                )
                bcompressed_x2_ref.at[bkv_sem_idx, p_ignore, h][...] = (
                    codec.update_page_fn(
                        old_p,
                        [w_arr[h : h + 1, :] for w_arr in new_words],
                        valid_mask,
                        slot_idx=ignore,
                    )
                )
            return

        num_touched = cdiv(offset + update_sz, page_size) - kv_p_start
        local_new_start = offset - bkv_idx * bkv_sz
        local_new_end = local_new_start + update_sz

        def merge_body(i, _):
            assert bpacked_new_x2_ref is not None
            assert codec.update_page_fn is not None
            p_local = p_ignore + i
            tok_pos = p_local * page_size + slot_iota
            valid_mask = (tok_pos >= local_new_start) & (tok_pos < local_new_end)
            p_ds = pl.ds(p_local * page_size, page_size)
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
                bcompressed_x2_ref.at[bkv_sem_idx, p_local, h][...] = (
                    codec.update_page_fn(
                        old_p,
                        [w_arr[:, h, :] for w_arr in new_words],
                        valid_mask,
                    )
                )
            return None

        lax.fori_loop(0, num_touched, merge_body, None, unroll=False)

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait: bool = False):
        sem = sems.at[staging.SEM_BKV_FETCH, bkv_sem_idx]
        kv_len = kv_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - cu_q_lens_ref[seq_idx]

        (
            kv_len_start,
            _,
            kv_left_frm_new,
            bkv_sz_frm_cache,
            bkv_sz_frm_new,
        ) = staging.compute_bkv_cache_and_new_sizes(
            kv_len,
            q_len,
            bkv_idx,
            bkv_sz=bkv_sz,
            skip_kv_update=skip_kv_update,
        )
        offset = kv_len_start + bkv_sz_frm_cache
        page_indices_offset = seq_idx * pages_per_seq + bkv_idx * bkv_p
        new_off = jnp.int32(0) if is_decode else bkv_sz_frm_cache

        if not wait:
            for i in range(bkv_p):
                has_page = jnp.where(bkv_sz_frm_cache > i * page_size, 1, 0)
                p_idx = page_indices_ref[
                    jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                ]
                async_copy(
                    kv_cache_hbm_ref.at[pl.ds(p_idx, has_page)],
                    bcompressed_x2_ref.at[bkv_sem_idx, pl.ds(i, has_page)],
                    sem,
                    wait=False,
                )
            if not skip_kv_update:
                assert packed_new_kv_hbm_ref is not None
                assert bpacked_new_x2_ref is not None
                new_start = q_end - kv_left_frm_new
                async_copy(
                    packed_new_kv_hbm_ref.at[pl.ds(new_start, bkv_sz_frm_new)],
                    bpacked_new_x2_ref.at[bkv_sem_idx, pl.ds(new_off, bkv_sz_frm_new)],
                    sem,
                    wait=False,
                )
        else:
            comp_dst = bcompressed_x2_ref.at[
                bkv_sem_idx, pl.ds(0, cdiv(bkv_sz_frm_cache, page_size))
            ]
            async_copy(src=comp_dst, dst=comp_dst, sem=sem, wait=True)
            if not skip_kv_update:
                assert bpacked_new_x2_ref is not None
                packed_new_dst = bpacked_new_x2_ref.at[
                    bkv_sem_idx, pl.ds(new_off, bkv_sz_frm_new)
                ]
                async_copy(src=packed_new_dst, dst=packed_new_dst, sem=sem, wait=True)

                @pl.when(bkv_sz_frm_new > 0)
                def _():
                    _merge_bkv_pages_in_vmem(
                        bkv_sem_idx, bkv_idx, offset, bkv_sz_frm_new
                    )

        return offset, bkv_sz_frm_new

    def _update_kv_cache(
        seq_idx, bkv_sem_idx, offset, update_sz, *, wait: bool = False
    ):
        if skip_kv_update:
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
        p_ignore = kv_p_start - bkv_id * bkv_p
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start
        if is_decode:
            async_copy(
                bcompressed_x2_ref.at[bkv_sem_idx, pl.ds(p_ignore, 1)],
                updated_kv_cache_hbm_ref.at[
                    pl.ds(page_indices_ref[page_indices_offset], 1)
                ],
                sem,
                wait=False,
            )
            return

        @pl.loop(0, num_touched, unroll=False)
        def _(i):
            async_copy(
                bcompressed_x2_ref.at[bkv_sem_idx, pl.ds(p_ignore + i, 1)],
                updated_kv_cache_hbm_ref.at[
                    pl.ds(page_indices_ref[page_indices_offset + i], 1)
                ],
                sem,
                wait=False,
            )

    staging_refs = staging.StagingRefs(
        q_hbm_ref, o_hbm_ref, bq_x2_ref, bo_x2_ref, cu_q_lens_ref, bo_ids_ref, sems
    )
    kv_update_ring = staging.KvUpdateRing(bkv_update_ids_ref, _update_kv_cache)
    fa_cfg_masked = flash_attention.FlashAttentionConfig(
        num_q_heads_per_kv_head=num_q_heads_per_kv_head,
        sm_scale=sm_scale,
        out_dtype=out_dtype,
        mask_value=mask_value,
        span_int_ty=span_int_ty,
        max_kv_len=max_kv_len,
        use_causal_mask=not is_decode,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    fa_cfg_unmasked = dataclasses.replace(fa_cfg_masked, use_causal_mask=False)

    def _load_bkv_dequant(
        bkv_sem_idx, kv_head_idx, start, sz
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        start_page = start // page_size
        refs = [
            bcompressed_x2_ref.at[bkv_sem_idx, start_page + p_off, kv_head_idx]
            for p_off in range(sz // page_size)
        ]
        return codec.dequant_page_fn(refs, kv_dtype)

    def compute_bkv_attention(ctx: pipeline.BkvStepContext) -> None:
        num_loops = cdiv(ctx.effective_bkv_sz, bkv_csz)
        preloaded_bq = (
            [
                staging.load_bq(bq_x2_ref, ctx.bq_sem_idx, h, 0, ctx.actual_bq_csz)
                for h in range(actual_num_kv_heads)
            ]
            if is_decode
            else None
        )

        def _run_compressed_tile(idx, *, skip_tile_mask: bool = False):
            bkv_start = idx * bkv_csz
            cur_processed_kv_len = ctx.processed_kv_len + bkv_start
            tile_cfg = fa_cfg_unmasked if skip_tile_mask else fa_cfg_masked
            prev_lm_slice = None
            prev_p = None
            prev_v = None
            prev_exp_m_diff = None

            # Keep `kv_head_idx` outer to amortize VREG page dequant across `bq_start`,
            # deferring `step2_pv` by one inner step to overlap MXU `P @ V` with VPU.
            for kv_head_idx in range(actual_num_kv_heads):
                bk_c, bv_c, k_norm_row = _load_bkv_dequant(
                    ctx.bkv_sem_idx, kv_head_idx, bkv_start, bkv_csz
                )
                if not skip_tile_mask:
                    bv_c = flash_attention.mask_v_tile(
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
                    cur_p, _, cur_exp_m_diff = (
                        flash_attention.flash_attention_step1_qk_softmax(
                            bq_c,
                            bk_c,
                            None,
                            l_ref.at[*lm_slice],
                            m_ref.at[*lm_slice],
                            processed_q_len=ctx.processed_q_len + bq_start,
                            processed_kv_len=cur_processed_kv_len,
                            effective_kv_len=ctx.effective_kv_len,
                            cfg=tile_cfg,
                            skip_kv_mask=skip_tile_mask,
                            k_norm_row=k_norm_row,
                        )
                    )
                    if prev_lm_slice is not None:
                        flash_attention.flash_attention_step2_pv(
                            prev_p,
                            prev_v,
                            prev_exp_m_diff,
                            acc_ref.at[*prev_lm_slice],
                            cfg=tile_cfg,
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
                cfg=fa_cfg_masked,
            )

        if sliding_window is not None:

            @pl.loop(0, num_loops, unroll=False)
            def _(idx):
                _run_compressed_tile(idx, skip_tile_mask=False)

        elif is_decode:
            num_full_loops = ctx.effective_bkv_sz // bkv_csz

            @pl.loop(0, num_full_loops, unroll=False)
            def _(idx):
                _run_compressed_tile(idx, skip_tile_mask=True)

            @pl.when(num_full_loops < num_loops)
            def _():
                _run_compressed_tile(num_full_loops, skip_tile_mask=False)

        else:
            num_full_loops = jnp.minimum(
                jnp.maximum(ctx.processed_q_len - ctx.bkv_idx * bkv_sz, 0) // bkv_csz,
                ctx.effective_bkv_sz // bkv_csz,
            )

            @pl.loop(0, num_full_loops, unroll=False)
            def _(idx):
                _run_compressed_tile(idx, skip_tile_mask=True)

            @pl.loop(num_full_loops, num_loops, unroll=False)
            def _(idx):
                _run_compressed_tile(idx, skip_tile_mask=False)

    pipeline.run_rpa_sequence_pipeline(
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
        cfg=pipeline.PipelineConfig(
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            bq_csz=bq_csz,
            use_causal_mask=not is_decode,
            sliding_window=sliding_window,
            static_q_len=1 if is_decode else None,
        ),
    )


def static_validate_inputs(
    queries: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    codec: CompressedKvCodec,
    kv_dtype: Any,
    packed_new_kv: jax.Array | None = None,
    skip_kv_update: bool = False,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
) -> None:
    """Validate inputs to the compressed-KV RPA v3 kernel at trace/compile time."""
    if not isinstance(codec, CompressedKvCodec):
        raise ValueError(f"Expected CompressedKvCodec, got {type(codec).__name__}.")
    if codec.dequant_page_fn is None:
        raise ValueError("Expected non-None codec.dequant_page_fn.")
    if not jnp.issubdtype(kv_dtype, jnp.floating):
        raise ValueError(f"Expected floating-point kv_dtype, got {kv_dtype=}.")

    if (
        kv_cache.dtype != jnp.uint8
        or kv_cache.ndim != 4
        or kv_cache.shape[1] <= 0
        or kv_cache.shape[2] <= 0
        or kv_cache.shape[3] != 128
    ):
        raise ValueError(
            "Expected 4D uint8 compressed kv_cache (num_pages, num_kv_heads,"
            f" page_words, 128), got {kv_cache.shape=}, {kv_cache.dtype=}."
        )

    actual_num_kv_heads = kv_cache.shape[1]
    page_words = kv_cache.shape[2]
    page_size = codec.page_size_from_words(page_words)
    if page_size <= 0:
        raise ValueError(f"Expected positive {page_size=} from {page_words=}.")

    util.validate_common_inputs(
        queries,
        None,
        None,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        page_size=page_size,
        kv_packing=get_dtype_packing(kv_dtype),
        actual_num_kv_heads=actual_num_kv_heads,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        block_sizes_by_name=(("decode", d_block_sizes), ("mixed", m_block_sizes)),
        vmem_limit_bytes=vmem_limit_bytes,
    )

    max_num_tokens, _, actual_head_dim = queries.shape
    if actual_head_dim % 128 != 0:
        raise ValueError(
            f"Expected {actual_head_dim=} to be a multiple of 128 for compressed KV."
        )
    if not skip_kv_update:
        if packed_new_kv is None or codec.update_page_fn is None:
            raise ValueError(
                "packed_new_kv and codec.update_page_fn are required unless"
                " skip_kv_update."
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


def _make_compressed_kv_scratch_shapes(
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
) -> list[Any]:
    """Build the VMEM/DMA scratch spec list shared by the estimator and kernel."""
    if page_words <= 0:
        raise ValueError(f"Expected positive {page_words=}.")
    if not skip_kv_update and num_new_words <= 0:
        raise ValueError(
            f"Expected positive {num_new_words=} when skip_kv_update is False."
        )
    is_decode = case == RpaCase.DECODE
    out_dtype = jnp.float32 if q_dtype == jnp.float32 else jnp.bfloat16
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_q_heads_per_kv_head = align_to(
        actual_num_q_heads // actual_num_kv_heads, q_packing
    )
    aligned_head_dim = align_to(head_dim, 128)
    base_shapes = launch.make_base_rpa_scratch_shapes(
        q_tail_shape=(
            num_q_heads_per_kv_head // q_packing,
            q_packing,
            aligned_head_dim,
        ),
        q_dtype=q_dtype,
        kv_tail_shape=(kv_packing, aligned_head_dim),
        kv_dtype=kv_dtype,
        out_dtype=out_dtype,
        actual_num_kv_heads=actual_num_kv_heads,
        bq_sz=bq_sz,
        bkv_sz=1,
        kv_rows_per_token=cdiv(actual_num_kv_heads * 2, kv_packing),
        pad_bkv_bank_conflicts=False,
    )
    bcompressed_double_buf = pltpu.VMEM(
        (2, bkv_sz // page_size, actual_num_kv_heads, page_words, 128),
        jnp.uint8,
    )
    scratch_shapes = [bcompressed_double_buf, *base_shapes[1:]]
    if not skip_kv_update:
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
    return scratch_shapes


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
    tpu_version: int | None = None,
) -> int:
    """Estimate total VMEM scratch bytes for `(bq_sz, bkv_sz)` with sublane padding."""
    scratch_shapes = _make_compressed_kv_scratch_shapes(
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
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
    if tpu_version is None and jax.default_backend() == "tpu":
        tpu_info = pltpu.get_tpu_info()
    else:
        resolved_ver = get_tpu_version() if tpu_version is None else tpu_version
        tpu_info = get_tpu_info_for_version(resolved_ver)
    return launch.get_scratch_vmem_bytes(scratch_shapes, tpu_info=tpu_info)


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
    page_words: int,
    num_new_words: int,
    max_bkv_csz: int | None = None,
    skip_kv_update: bool = False,
    tpu_version: int | None = None,
) -> dict[str, int]:
    """Select `(bq_sz, bkv_sz, bq_csz, bkv_csz)` bounded by `vmem_limit_bytes`."""
    if page_words <= 0:
        raise ValueError(f"Expected positive {page_words=}.")
    if not skip_kv_update and num_new_words <= 0:
        raise ValueError(
            f"Expected positive {num_new_words=} when skip_kv_update is False."
        )
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
    raw_csz = (
        sizes["bkv_csz"]
        if max_bkv_csz is None
        else min(sizes["bkv_csz"], max(page_size, max_bkv_csz))
    )
    bkv_csz = align_to(raw_csz, page_size)
    # Cap `bkv_sz` at `max(32 pages, 512)` since compressed `uint8` pages are
    # ~4x smaller in HBM and issue one DMA descriptor per page.
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
        tpu_version=tpu_version,
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
    kv_cache: jax.Array,  # u8[total_num_pages, actual_num_kv_heads, page_words, 128]
    kv_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    codec: CompressedKvCodec,
    packed_new_kv: jax.Array | None = None,
    kv_dtype: Any = None,
    skip_kv_update: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    disable_bounds_checks: bool = True,
    disable_semaphore_checks: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Fused Pallas RPA v3 attention over a byte-packed compressed KV cache.

    `queries` and `kv_cache` are marked in `donate_argnames` (`kv_cache` is
    updated in place via `input_output_aliases`). Both `DECODE` and `MIXED`
    merge `packed_new_kv` into `bcompressed_x2_ref` before compute and read all
    tokens via `codec.dequant_page_fn`.

    Args:
      queries: `floating[max_num_tokens, num_q_heads, head_dim]`, `head_dim % 128 == 0`.
      kv_cache: `uint8[total_num_pages, num_kv_heads, page_words, 128]`.
      kv_lens: `int32[max_num_seqs]` effective KV lengths for `[0, distribution[2])`.
      page_indices: `int32[max_num_seqs * pages_per_seq]` flattened block table.
      cu_q_lens: `int32[max_num_seqs + 1]` cumulative query token offsets.
      distribution: `int32[3]` `(decode_end, prefill_end, total_seqs)`.
      codec: Frozen, hashable `CompressedKvCodec` defining `dequant_page_fn`,
        `update_page_fn`, `page_size_from_words`, and optional `max_bkv_csz_fn`.
      packed_new_kv: `uint8[max_num_tokens, num_new_words, num_kv_heads, 128]`
        per-token compressed payload (required when `skip_kv_update=False`).
      kv_dtype: Floating-point dtype produced by `codec.dequant_page_fn`
        (defaults to `queries.dtype`).
      skip_kv_update: If `True`, reads all `[0, kv_len)` tokens from `kv_cache`
        without reading `packed_new_kv` or writing back.
      sm_scale: Softmax scale applied to `Q @ K^T`.
      sliding_window: Optional causal sliding-window span in tokens.
      soft_cap: Optional `tanh` logit soft cap.
      d_block_sizes: Optional `(bq_sz, bkv_sz, bq_csz, bkv_csz)` override for `DECODE`.
      m_block_sizes: Optional `(bq_sz, bkv_sz, bq_csz, bkv_csz)` override for `MIXED`.
      vmem_limit_bytes: Per-core VMEM limit override; defaults to chip capacity.
      disable_bounds_checks: Passed through to `pltpu.CompilerParams`.
      disable_semaphore_checks: Passed through to `pltpu.CompilerParams`.

    Returns:
      `(outputs, updated_kv_cache)` matching `(queries, kv_cache)` shapes and dtypes.
    """
    q = queries
    resolved_kv_dtype = q.dtype if kv_dtype is None else kv_dtype
    tpu_version = get_tpu_version()
    out_dtype = jnp.float32 if q.dtype == jnp.float32 else jnp.bfloat16
    mask_value = float(jnp.finfo(out_dtype).min)
    if vmem_limit_bytes is None:
        vmem_limit_bytes = pltpu.get_tpu_info().vmem_capacity_bytes

    static_validate_inputs(
        q,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        codec=codec,
        kv_dtype=resolved_kv_dtype,
        packed_new_kv=packed_new_kv,
        skip_kv_update=skip_kv_update,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        d_block_sizes=d_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    actual_num_q_heads, actual_head_dim = q.shape[1], q.shape[2]
    actual_num_kv_heads = kv_cache.shape[1]
    page_words = kv_cache.shape[2]
    page_size = codec.page_size_from_words(page_words)
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q = prepare_q(q, actual_num_kv_heads, fuse_non_tiling_axis_swap=False)
    _, max_num_tokens, _, _, head_dim = q.shape
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_new_words = 0 if skip_kv_update else cast(jax.Array, packed_new_kv).shape[1]
    max_bkv_csz = (
        None
        if codec.max_bkv_csz_fn is None
        else codec.max_bkv_csz_fn(head_dim, page_size, tpu_version=tpu_version)
    )
    scalar_prefetches = launch.make_scalar_prefetches(
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
        scratch_shapes = _make_compressed_kv_scratch_shapes(
            bq_sz=bq_sz,
            bkv_sz=bkv_sz,
            q_dtype=q.dtype,
            kv_dtype=resolved_kv_dtype,
            actual_num_q_heads=actual_num_q_heads,
            actual_num_kv_heads=actual_num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            page_words=page_words,
            num_new_words=num_new_words,
            case=case,
            skip_kv_update=skip_kv_update,
        )
        hbm_inputs: tuple[Any, ...] = (
            (q, kv_cache) if skip_kv_update else (q, kv_cache, packed_new_kv)
        )
        scope_name = (
            f"RPA-CKV-{case.symbol}-p_{page_size}-bq_{bq_sz}_{bq_csz}"
            f"-bkv_{bkv_sz}_{bkv_csz}"
            + (f"-sw_{sliding_window}" if sliding_window is not None else "")
        )
        return launch.invoke_rpa_pallas_kernel(
            kernel_fn=functools.partial(
                _ragged_paged_attention_kernel,
                codec=codec,
                kv_dtype=resolved_kv_dtype,
                page_size=page_size,
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
                tpu_version=tpu_version,
            ),
            scalar_prefetches=scalar_prefetches,
            inputs=hbm_inputs,
            scratch_shapes=scratch_shapes,
            out_arrays=(q, kv_cache),
            input_output_aliases={num_sp: 0, num_sp + 1: 1},
            vmem_limit_bytes=cast(int, vmem_limit_bytes),
            scope_name=scope_name,
            tpu_version=tpu_version,
            disable_bounds_checks=disable_bounds_checks,
            disable_semaphore_checks=disable_semaphore_checks,
        )

    block_size_kw = dict(
        q_dtype=q.dtype,
        kv_dtype=resolved_kv_dtype,
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
        max_bkv_csz=max_bkv_csz,
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
    kv_cache_bundle: jax.Array,  # u8[num_pages, num_layers, kv_heads, page_words, 128]
    layer_idx: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    codec: CompressedKvCodec,
    packed_new_kv: jax.Array | None = None,
    kv_dtype: Any = None,
    skip_kv_update: bool = False,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    d_block_sizes: tuple[int, int, int, int] | None = None,
    m_block_sizes: tuple[int, int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    disable_bounds_checks: bool = True,
    disable_semaphore_checks: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Execute compressed-KV RPA v3 against a 5D block-major bundled KV cache."""
    if kv_cache_bundle.ndim != 5 or kv_cache_bundle.dtype != jnp.uint8:
        raise ValueError(
            "Expected 5D uint8 compressed kv_cache_bundle"
            " (num_pages, num_layers, num_kv_heads, page_words, 128),"
            f" got {kv_cache_bundle.shape=}, {kv_cache_bundle.dtype=}."
        )
    return util.run_bundled_rpa(
        ragged_paged_attention,
        queries,
        None,
        None,
        kv_cache_bundle,
        layer_idx,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        codec=codec,
        packed_new_kv=packed_new_kv,
        kv_dtype=kv_dtype,
        skip_kv_update=skip_kv_update,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        d_block_sizes=d_block_sizes,
        m_block_sizes=m_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        disable_bounds_checks=disable_bounds_checks,
        disable_semaphore_checks=disable_semaphore_checks,
    )
