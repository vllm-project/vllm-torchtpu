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
"""FlashMLA (Split-KV across TP=8 ranks) for DeepSeek-V4.1 SWA and CSA.

Architecture:
1. Each TP rank r in {0..7} holds all H = 64 query heads ([B, 64, 512], 50% MXU
   M-occupancy per query token, 100% across >= 2 tokens) via all_gather on Q.
2. Each rank r scans only 1/8th of the KV blocks/candidates (L / 8 per rank),
   emitting partial online-softmax accumulators:
     - Unnormalized output O_r in R^[B, 64, 512] (float32)
     - Row max m_r in R^[B, 64] (float32)
     - Sum-exp l_r in R^[B, 64] (float32)
3. Across the 8 TP ranks, exact online-softmax rescaling + head scatter (64 -> 8):
     m = max_r m_r
     l = sum_r l_r * exp(m_r - m) + exp(sinks - m)
     O_local = psum_scatter(O_r * exp(m_r - m), scatter_dimension=1) / l_local
   outputs [B, H_local=8, 512] directly into head-sharded o_proj_op + wo_b.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather
from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import (
    DEFAULT_VMEM_LIMIT_BYTES,
    align_to,
    cdiv,
    get_dtype_bitwidth,
    get_dtype_packing,
    prepare_kv_inputs,
    prepare_q_inputs,
)
from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
    _dequant_dsv4_fp8,
    _largest_divisor,
)

_MIN_F32 = jnp.finfo(jnp.float32).min


def _split_kv_swa_decode_kernel(
    # Scalar prefetches
    kv_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    rank_info_ref,  # [2] (tp_rank, tp_size)
    # HBM inputs
    q_hbm_ref,  # [max_num_tokens, num_q_heads=64, head_dim=512] bf16
    new_kv_hbm_ref,  # [max_num_tokens, token_bytes // 128, 128] uint8
    cache_kv_hbm_ref,  # [total_num_pages, phys_page_per_pack, kv_pack, lkv_dim]
    in_output_hbm_ref,  # [max_num_tokens, num_q_heads=64, head_dim=512] f32
    in_l_hbm_ref,  # [max_num_tokens, 128] f32
    in_m_hbm_ref,  # [max_num_tokens, 128] f32
    # HBM outputs
    o_hbm_ref,  # [max_num_tokens, num_q_heads=64, head_dim=512] f32
    updated_cache_kv_hbm_ref,
    l_hbm_ref,  # [max_num_tokens, 128] f32
    m_hbm_ref,  # [max_num_tokens, 128] f32
    # Double-buffered VMEM scratch
    bkv_batch_x2_vmem,  # [2, seq_batch_sz, bkv_sz, slots_per_token * kv_pack, lkv_dim] u8
    q_batch_x2_vmem,  # [2, seq_batch_sz, num_q_heads, head_dim] bf16
    o_batch_x2_vmem,  # [2, seq_batch_sz, num_q_heads, head_dim] f32
    l_batch_x2_vmem,  # [2, seq_batch_sz, 128] f32
    m_batch_x2_vmem,  # [2, seq_batch_sz, 128] f32
    sems,  # [2, seq_batch_sz + 4]
    *,
    sm_scale: float,
    sliding_window: int,
    local_window: int,
    logical_page_size: int,
    bkv_p: int,
    seq_batch_sz: int,
):
    del in_output_hbm_ref, in_l_hbm_ref, in_m_hbm_ref

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
    tp_rank = rank_info_ref[0]

    num_decode_seqs = end_seq_idx - start_seq_idx
    num_batches = cdiv(num_decode_seqs, seq_batch_sz)

    def _slice_bounds(kv_len):
        global_start = jnp.maximum(kv_len - sliding_window, jnp.int32(0))
        rank_start = global_start + tp_rank * local_window
        rank_end = jnp.minimum(kv_len, rank_start + local_window)
        from_cache = jnp.maximum(
            jnp.minimum(rank_end, kv_len - 1) - rank_start, jnp.int32(0)
        )
        has_new = (rank_start <= kv_len - 1) & (kv_len - 1 < rank_end)
        return rank_start, rank_end, from_cache, has_new

    def _fetch_bq(batch_idx, sem_idx, wait=False):
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

    def _fetch_bkv(batch_idx, sem_idx, wait=False):
        if not wait:
            wait_update_kv_cache(batch_idx - 2, sem_idx)
        s_start = start_seq_idx + batch_idx * seq_batch_sz

        for s in range(seq_batch_sz):
            seq_idx = s_start + s
            seq_idx_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
            active_seq = seq_idx < end_seq_idx

            sem_kv = sems.at[sem_idx, 4 + s]
            bkv_s_ref = bkv_batch_x2_vmem.at[sem_idx, s]

            kv_len = jnp.where(active_seq, kv_lens_ref[seq_idx_clamped], 0)
            rank_start, _, from_cache, has_new = _slice_bounds(kv_len)

            kv_p_start = rank_start // logical_page_size
            page_off = rank_start - kv_p_start * logical_page_size
            page_indices_offset = jnp.minimum(
                seq_idx_clamped * pages_per_seq + kv_p_start,
                num_page_indices - 1,
            )

            vmem_off = 0
            for i in range(bkv_p + 1):
                avail = logical_page_size - page_off
                chunk_sz = jnp.where(
                    active_seq,
                    jnp.clip(from_cache - vmem_off, jnp.int32(0), avail),
                    0,
                )
                page_idx = page_indices_ref[
                    jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                ]
                async_copy(
                    reshaped_cache_hbm_ref.at[
                        pl.ds(page_idx * phys_tokens_per_page + page_off, chunk_sz)
                    ],
                    bkv_s_ref.at[pl.ds(vmem_off, chunk_sz)],
                    sem_kv,
                    wait=wait,
                )
                page_off = 0
                vmem_off += chunk_sz

            sz_new = jnp.where(active_seq, 1, 0)
            new_kv_slot = jnp.where(has_new, from_cache, local_window)
            new_kv_vmem_ref = bkv_s_ref.bitcast(jnp.uint8).reshape(
                bkv_sz, token_bytes // 128, 128
            )
            q_token_idx = jnp.where(active_seq, cu_q_lens_ref[seq_idx_clamped], 0)
            async_copy(
                new_kv_hbm_ref.at[pl.ds(q_token_idx, sz_new)],
                new_kv_vmem_ref.at[pl.ds(new_kv_slot, sz_new)],
                sem_kv,
                wait=wait,
            )

    def compute_batch(batch_idx, sem_idx):
        s_start = start_seq_idx + batch_idx * seq_batch_sz
        kv_lens_batch = jnp.stack(
            [
                jnp.where(
                    (s_start + s) < end_seq_idx,
                    kv_lens_ref[jnp.minimum(s_start + s, max_num_seqs - 1)],
                    0,
                )
                for s in range(seq_batch_sz)
            ]
        )
        rank_start_batch, rank_end_batch, _, _ = _slice_bounds(kv_lens_batch)
        rank_count_batch = jnp.maximum(rank_end_batch - rank_start_batch, jnp.int32(0))

        bkv_batch_vmem = bkv_batch_x2_vmem[sem_idx]
        q_batch_vmem = q_batch_x2_vmem[sem_idx]

        bkv_batch = pltpu.bitcast(
            bkv_batch_vmem[:, :bkv_sz, ...], jnp.bfloat16
        ).reshape(seq_batch_sz, bkv_sz, head_dim)

        m_vec = lax.broadcasted_iota(jnp.int32, (1, 1, bkv_sz), 2)
        keep_batch = m_vec < rank_count_batch[:, None, None]

        bkv_batch = jnp.where(jnp.swapaxes(keep_batch, 1, 2), bkv_batch, 0)

        s_mat = jnp.einsum(
            "snd,smd->snm", q_batch_vmem, bkv_batch, preferred_element_type=jnp.float32
        )
        s_mat *= sm_scale
        s_mat = jnp.where(keep_batch, s_mat, _MIN_F32)

        s_max = jnp.max(s_mat, axis=2, keepdims=True)
        p_mat = jnp.where(keep_batch, jnp.exp(s_mat - s_max), 0.0)
        pv_batch = jnp.einsum(
            "snm,smd->snd", p_mat, bkv_batch, preferred_element_type=jnp.float32
        )

        m_val = s_max[:, :, 0]
        l_val = jnp.sum(p_mat, axis=2)

        o_batch_x2_vmem[sem_idx, ...] = pv_batch
        num_l_heads = l_batch_x2_vmem.shape[2]
        if num_l_heads == num_q_heads:
            l_batch_x2_vmem[sem_idx, ...] = l_val
            m_batch_x2_vmem[sem_idx, ...] = m_val
        else:
            l_batch_x2_vmem[sem_idx, ...] = jnp.pad(
                l_val, ((0, 0), (0, num_l_heads - num_q_heads))
            )
            m_batch_x2_vmem[sem_idx, ...] = jnp.pad(
                m_val,
                ((0, 0), (0, num_l_heads - num_q_heads)),
                constant_values=_MIN_F32,
            )

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
        for s in range(seq_batch_sz):
            seq_idx = s_start + s
            seq_idx_clamped = jnp.minimum(seq_idx, max_num_seqs - 1)
            active_seq = seq_idx < end_seq_idx
            sz_write = jnp.where(active_seq, 1, 0)

            sem_kv = sems.at[sem_idx, 4 + s]
            bkv_s_ref = bkv_batch_x2_vmem.at[sem_idx, s]

            kv_len = jnp.where(active_seq, kv_lens_ref[seq_idx_clamped], 0)
            _, _, from_cache, has_new = _slice_bounds(kv_len)
            new_kv_slot = jnp.where(has_new, from_cache, local_window)

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
                bkv_s_ref.at[pl.ds(new_kv_slot, sz_write)],
                reshaped_updated_cache_hbm_ref.at[
                    pl.ds(
                        new_token_page_idx * phys_tokens_per_page + new_token_off,
                        sz_write,
                    )
                ],
                sem_kv,
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

    @pl.when(num_batches > 0)
    def _():
        fetch_batch(0, 0)

    def loop_body(i, _):
        sem_idx = i % 2
        next_sem_idx = (i + 1) % 2
        next_batch_idx = i + 1

        @pl.when(next_batch_idx < num_batches)
        def _():
            fetch_batch(next_batch_idx, next_sem_idx)

        wait_fetch(i, sem_idx)
        wait_send_bo(i - 2, sem_idx)
        compute_batch(i, sem_idx)
        _send_bo(i, sem_idx, wait=False)
        _update_kv_cache(i, sem_idx, wait=False)
        return ()

    lax.fori_loop(0, num_batches, loop_body, ())

    @pl.when(num_batches > 0)
    def _():
        @pl.when(num_batches > 1)
        def _():
            wait_send_bo(num_batches - 2, (num_batches - 2) % 2)
            wait_update_kv_cache(num_batches - 2, (num_batches - 2) % 2)

        wait_send_bo(num_batches - 1, (num_batches - 1) % 2)
        wait_update_kv_cache(num_batches - 1, (num_batches - 1) % 2)


def run_split_kv_swa_rank(
    q_full: jax.Array,  # [num_tokens, 64, 512] bf16
    new_kv: jax.Array,  # [num_tokens, 512] bf16
    sw_cache: jax.Array,  # [total_pages, ...] uint8 / int32
    kv_lens: jax.Array,  # [max_num_seqs] int32
    page_indices: jax.Array,  # [max_num_seqs * pages_per_seq] int32
    cu_q_lens: jax.Array,  # [max_num_seqs + 1] int32
    distribution: jax.Array,  # [3] int32
    *,
    tp_rank: jax.Array,  # scalar int32 in {0..tp_size-1}
    tp_size: int = 8,
    sm_scale: float,
    sliding_window: int = 128,
    logical_page_size: int = 16,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Runs Split-KV SWA on rank `tp_rank`, returning (O_r [B,64,512] f32, updated_cache, l_r [B,64] f32, m_r [B,64] f32)."""
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_flash_attention_op import (
        _prepare_swa_metadata,
    )

    num_tokens, actual_num_q_heads, actual_head_dim = q_full.shape
    max_num_seqs = kv_lens.shape[0]
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    if eff_max_seqs < max_num_seqs:
        pages_per_seq = page_indices.size // max_num_seqs
        kv_lens = kv_lens[:eff_max_seqs]
        page_indices = page_indices[: eff_max_seqs * pages_per_seq]
        cu_q_lens = cu_q_lens[: eff_max_seqs + 1]
        distribution = jnp.minimum(distribution, eff_max_seqs)

    kv_lens, page_indices = _prepare_swa_metadata(
        kv_lens,
        page_indices,
        cu_q_lens,
        num_tokens=num_tokens,
        sw_cache_num_pages=sw_cache.shape[0],
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
    )

    q_padded = prepare_q_inputs(q_full)
    _, num_q_heads, head_dim = q_padded.shape

    _, physical_page_size_per_kv_packing, kv_packing, lkv_dim = (
        csa_cache_layout.u8_slab_shape(sw_cache.shape, sw_cache.dtype)
    )
    slot_bytes = kv_packing * lkv_dim
    token_bytes = head_dim * get_dtype_bitwidth(new_kv.dtype) // 8
    slots_per_token = token_bytes // slot_bytes

    new_kv_u8 = prepare_kv_inputs(new_kv).reshape(
        num_tokens, slots_per_token * kv_packing, lkv_dim
    )

    local_window = cdiv(sliding_window, tp_size)
    # Allocate +1 extra page in VMEM so ranks whose local slice excludes kv_len-1
    # still have a dedicated VMEM slot at `local_window` to stage `new_kv` writeback.
    bkv_p = max(1, cdiv(local_window, logical_page_size) + 1)
    seq_batch_sz = 8 if num_tokens <= 16 else (16 if num_tokens <= 128 else 32)
    num_l_heads = align_to(num_q_heads, 128)

    in_output = jnp.zeros((num_tokens, num_q_heads, head_dim), dtype=jnp.float32)
    in_l = jnp.zeros((num_tokens, num_l_heads), dtype=jnp.float32)
    in_m = jnp.full((num_tokens, num_l_heads), _MIN_F32, dtype=jnp.float32)

    bkv_sz = bkv_p * logical_page_size
    scratch_shapes = [
        pltpu.VMEM(
            (2, seq_batch_sz, bkv_sz, slots_per_token * kv_packing, lkv_dim),
            jnp.uint8,
        ),
        pltpu.VMEM((2, seq_batch_sz, num_q_heads, head_dim), q_padded.dtype),
        pltpu.VMEM((2, seq_batch_sz, num_q_heads, head_dim), jnp.float32),
        pltpu.VMEM((2, seq_batch_sz, num_l_heads), jnp.float32),
        pltpu.VMEM((2, seq_batch_sz, num_l_heads), jnp.float32),
        pltpu.SemaphoreType.DMA((2, seq_batch_sz + 4)),
    ]

    scalar_prefetches = (
        kv_lens,
        page_indices,
        cu_q_lens,
        jnp.array([0, distribution[0]], jnp.int32),
        jnp.array([tp_rank, tp_size], jnp.int32),
    )

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out_shape = [
        jax.ShapeDtypeStruct((num_tokens, num_q_heads, head_dim), jnp.float32),
        jax.ShapeDtypeStruct(sw_cache.shape, sw_cache.dtype),
        jax.ShapeDtypeStruct((num_tokens, num_l_heads), jnp.float32),
        jax.ShapeDtypeStruct((num_tokens, num_l_heads), jnp.float32),
    ]

    o_r, updated_cache, l_r, m_r = pl.pallas_call(
        functools.partial(
            _split_kv_swa_decode_kernel,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            local_window=local_window,
            logical_page_size=logical_page_size,
            bkv_p=bkv_p,
            seq_batch_sz=seq_batch_sz,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=[hbm] * 6,
            out_specs=[hbm] * 4,
            grid=(1,),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        out_shape=out_shape,
        input_output_aliases={
            len(scalar_prefetches) + 2: 1,
            len(scalar_prefetches) + 3: 0,
            len(scalar_prefetches) + 4: 2,
            len(scalar_prefetches) + 5: 3,
        },
        name=f"FlashMLA-SplitKV-SWA-bkvp_{bkv_p}-sb_{seq_batch_sz}",
    )(
        *scalar_prefetches,
        q_padded,
        new_kv_u8,
        sw_cache,
        in_output,
        in_l,
        in_m,
    )

    return (
        o_r[:, :actual_num_q_heads, :actual_head_dim],
        updated_cache,
        l_r[:, :actual_num_q_heads],
        m_r[:, :actual_num_q_heads],
    )


def _split_kv_sparse_mla_kernel(
    # Scalar prefetches
    kv_lens_ref,  # [max_num_seqs]
    start_end_seq_idx_ref,  # [3] (start_seq_idx, end_seq_idx, num_valid_tokens)
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [2]
    # HBM inputs
    q_hbm_ref,  # [max_num_tokens, num_q_heads=64, head_dim=512] bf16
    cache_kv_nope_hbm_ref,  # int32[chunk_size, topk_local, 128]
    cache_kv_rope_hbm_ref,  # int32[chunk_size, topk_local // 4, 128]
    swa_acc_hbm_ref,  # [max_num_tokens, num_q_heads=64, head_dim=512] f32
    swa_l_hbm_ref,  # [max_num_tokens, 128] f32
    swa_m_hbm_ref,  # [max_num_tokens, 128] f32
    in_out_l_hbm_ref,  # [max_num_tokens, 128] f32
    in_out_m_hbm_ref,  # [max_num_tokens, 128] f32
    # HBM outputs
    o_hbm_ref,  # [max_num_tokens, num_q_heads=64, head_dim=512] f32
    out_l_hbm_ref,  # [max_num_tokens, 128] f32
    out_m_hbm_ref,  # [max_num_tokens, 128] f32
    # Double-buffered VMEM scratch
    bkv_nope_x2_ref,  # [2, batch_size, topk_local * 4, 128] u8
    bkv_rope_x2_ref,  # [2, batch_size, (topk_local // 4) * 2, 128] bf16
    bq_x2_ref,  # [2, batch_size, num_q_heads, head_dim] bf16
    bo_x2_ref,  # [2, batch_size, num_q_heads, head_dim] f32
    bl_x2_ref,  # [2, batch_size, 128] f32
    bm_x2_ref,  # [2, batch_size, 128] f32
    swa_acc_x2_ref,  # [2, batch_size, num_q_heads, head_dim] f32
    bl_out_x2_ref,  # [2, batch_size, 128] f32
    bm_out_x2_ref,  # [2, batch_size, 128] f32
    sems,  # [9, 2]
    *,
    sm_scale: float,
    batch_size: int = 1,
):
    del in_out_l_hbm_ref, in_out_m_hbm_ref
    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    cache_kv_nope_hbm_ref = cache_kv_nope_hbm_ref.bitcast(jnp.uint8)
    cache_kv_rope_hbm_ref = cache_kv_rope_hbm_ref.bitcast(jnp.bfloat16)
    bkv_sz = cache_kv_rope_hbm_ref.shape[1] * 2

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    num_valid_tokens = start_end_seq_idx_ref[2]

    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
        sem_nope = sems.at[0, bkv_sem_idx]
        sem_rope = sems.at[6, bkv_sem_idx]
        bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, :]
        bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx, :]
        page_idx_start = seq_idx_start - start_seq_idx
        if not wait:
            _async_copy(
                cache_kv_nope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                bkv_nope_vmem_ref,
                sem_nope,
                False,
            )
            _async_copy(
                cache_kv_rope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                bkv_rope_vmem_ref,
                sem_rope,
                False,
            )
        else:
            _async_copy(src=bkv_nope_vmem_ref, dst=bkv_nope_vmem_ref, sem=sem_nope, wait=True)
            _async_copy(src=bkv_rope_vmem_ref, dst=bkv_rope_vmem_ref, sem=sem_rope, wait=True)

    def _fetch_bq_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, :]
        if not wait:
            _async_copy(q_hbm_ref.at[pl.ds(seq_idx_start, batch_size)], bq_vmem_ref, sem, False)
        else:
            _async_copy(src=bq_vmem_ref, dst=bq_vmem_ref, sem=sem, wait=True)

    def _fetch_swa_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem_acc = sems.at[3, bq_sem_idx]
        sem_l = sems.at[4, bq_sem_idx]
        sem_m = sems.at[5, bq_sem_idx]
        if not wait:
            _async_copy(
                swa_acc_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                swa_acc_x2_ref.at[bq_sem_idx, :],
                sem_acc,
                False,
            )
            _async_copy(
                swa_l_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bl_x2_ref.at[bq_sem_idx, :],
                sem_l,
                False,
            )
            _async_copy(
                swa_m_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bm_x2_ref.at[bq_sem_idx, :],
                sem_m,
                False,
            )
        else:
            _async_copy(src=swa_acc_x2_ref.at[bq_sem_idx, :], dst=swa_acc_x2_ref.at[bq_sem_idx, :], sem=sem_acc, wait=True)
            _async_copy(src=bl_x2_ref.at[bq_sem_idx, :], dst=bl_x2_ref.at[bq_sem_idx, :], sem=sem_l, wait=True)
            _async_copy(src=bm_x2_ref.at[bq_sem_idx, :], dst=bm_x2_ref.at[bq_sem_idx, :], sem=sem_m, wait=True)

    def _send_bo_batch(seq_idx_start, bo_sem_idx, *, wait=False):
        sem_o = sems.at[2, bo_sem_idx]
        sem_l = sems.at[7, bo_sem_idx]
        sem_m = sems.at[8, bo_sem_idx]
        sz = jnp.clip(num_valid_tokens - seq_idx_start, 1, batch_size)
        v_o = bo_x2_ref.at[bo_sem_idx, pl.ds(0, sz)]
        v_l = bl_out_x2_ref.at[bo_sem_idx, pl.ds(0, sz)]
        v_m = bm_out_x2_ref.at[bo_sem_idx, pl.ds(0, sz)]
        if not wait:
            bo_ids_ref[bo_sem_idx] = seq_idx_start
            _async_copy(v_o, o_hbm_ref.at[pl.ds(seq_idx_start, sz)], sem_o, False)
            _async_copy(v_l, out_l_hbm_ref.at[pl.ds(seq_idx_start, sz)], sem_l, False)
            _async_copy(v_m, out_m_hbm_ref.at[pl.ds(seq_idx_start, sz)], sem_m, False)
        else:
            _async_copy(src=v_o, dst=v_o, sem=sem_o, wait=True)
            _async_copy(src=v_l, dst=v_l, sem=sem_l, wait=True)
            _async_copy(src=v_m, dst=v_m, sem=sem_m, wait=True)

    def wait_send_bo_batch(bo_sem_idx):
        old_seq_idx_start = bo_ids_ref[bo_sem_idx]

        @pl.when(old_seq_idx_start >= 0)
        def _():
            _send_bo_batch(old_seq_idx_start, bo_sem_idx, wait=True)

    def start_fetch_bi(seq_idx_start, bi_sem_idx):
        _fetch_bkv_batch(seq_idx_start, bi_sem_idx, wait=False)
        _fetch_bq_batch(seq_idx_start, bi_sem_idx, wait=False)
        _fetch_swa_batch(seq_idx_start, bi_sem_idx, wait=False)

    def wait_fetch_bi(seq_idx_start, bi_sem_idx):
        _fetch_bkv_batch(seq_idx_start, bi_sem_idx, wait=True)
        _fetch_bq_batch(seq_idx_start, bi_sem_idx, wait=True)
        _fetch_swa_batch(seq_idx_start, bi_sem_idx, wait=True)

    def load_bkv_batch(bkv_sem_idx):
        bkv_nope = bkv_nope_x2_ref.at[bkv_sem_idx][...].reshape(batch_size * bkv_sz, -1)
        bkv_nope = _dequant_dsv4_fp8(bkv_nope).reshape(batch_size, bkv_sz, -1)
        bkv_rope_pairs = bkv_rope_x2_ref.at[bkv_sem_idx][...].reshape(
            batch_size // 2, bkv_sz, 128
        )
        bkvs_list = []
        for p in range(batch_size // 2):
            rope_even = bkv_rope_pairs[p, :, :64]
            rope_odd = bkv_rope_pairs[p, :, 64:]
            bkvs_list.append(jnp.concatenate([bkv_nope[2 * p], rope_even], axis=-1))
            bkvs_list.append(jnp.concatenate([bkv_nope[2 * p + 1], rope_odd], axis=-1))
        return bkvs_list

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bi(start_seq_idx, 0)

    bi_sem_idx = sem_ids_ref[0]
    next_batch_start = batch_start_seq_idx + batch_size
    next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)

    @pl.when(next_batch_start < end_seq_idx)
    def _():
        sem_ids_ref[0] = next_bi_sem_idx
        start_fetch_bi(next_batch_start, next_bi_sem_idx)

    wait_fetch_bi(batch_start_seq_idx, bi_sem_idx)

    bo_sem_idx = sem_ids_ref[1]
    next_bo_sem_idx = lax.select(bo_sem_idx == 0, 1, 0)
    sem_ids_ref[1] = next_bo_sem_idx
    wait_send_bo_batch(bo_sem_idx)

    bkvs = load_bkv_batch(bi_sem_idx)
    bqs = [
        bq_x2_ref.at[bi_sem_idx, i][...].reshape(num_q_heads, head_dim)
        for i in range(batch_size)
    ]
    kv_lens = [kv_lens_ref[batch_start_seq_idx + i] for i in range(batch_size)]
    swa_ms = [bm_x2_ref.at[bi_sem_idx, i, :num_q_heads][...] for i in range(batch_size)]
    swa_ls = [bl_x2_ref.at[bi_sem_idx, i, :num_q_heads][...] for i in range(batch_size)]
    swa_accs = [swa_acc_x2_ref.at[bi_sem_idx, i][...] for i in range(batch_size)]

    num_l_heads = bl_out_x2_ref.shape[2]
    for b_idx in range(batch_size):
        q_i = bqs[b_idx]  # [64, 512]
        kv_i = bkvs[b_idx]  # [64, 512]
        kv_len_i = kv_lens[b_idx]
        m_prev = swa_ms[b_idx][:, None]  # [64, 1]
        l_prev = swa_ls[b_idx][:, None]  # [64, 1]
        o_prev = swa_accs[b_idx]  # [64, 512]

        s = jnp.einsum("nd,md->nm", q_i, kv_i, preferred_element_type=jnp.float32)
        s *= sm_scale
        valid_mask = lax.broadcasted_iota(jnp.int32, s.shape, 1) < kv_len_i
        s = jnp.where(valid_mask, s, _MIN_F32)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        p = jnp.where(valid_mask, jnp.exp(s - m_curr), 0.0)
        exp_m_diff = jnp.where(m_prev > _MIN_F32, jnp.exp(m_prev - m_curr), 0.0)
        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        l_curr = exp_m_diff * l_prev + p_rowsum

        pv = jnp.einsum("nm,md->nd", p, kv_i, preferred_element_type=jnp.float32)
        acc = exp_m_diff * o_prev + pv

        bo_x2_ref.at[bo_sem_idx, b_idx][...] = acc
        l_1d = l_curr[:, 0]
        m_1d = m_curr[:, 0]
        if num_l_heads == num_q_heads:
            bl_out_x2_ref.at[bo_sem_idx, b_idx][...] = l_1d
            bm_out_x2_ref.at[bo_sem_idx, b_idx][...] = m_1d
        else:
            bl_out_x2_ref.at[bo_sem_idx, b_idx][...] = jnp.pad(
                l_1d, (0, num_l_heads - num_q_heads)
            )
            bm_out_x2_ref.at[bo_sem_idx, b_idx][...] = jnp.pad(
                m_1d, (0, num_l_heads - num_q_heads), constant_values=_MIN_F32
            )

    _send_bo_batch(batch_start_seq_idx, bo_sem_idx, wait=False)

    @pl.when(batch_end_seq_idx >= end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo_batch(i)


def run_split_kv_csa_rank(
    q_full: jax.Array,  # [num_tokens, 64, 512] bf16
    new_kv: jax.Array,  # [num_tokens, 512] bf16
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    cache_kv_nope: jax.Array,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # int32[total_num_pages, page_size // 4, 128]
    topk_indices_local: jax.Array,  # int32[num_tokens, topk // tp_size] (e.g. [B, 64])
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    *,
    tp_rank: jax.Array,
    tp_size: int = 8,
    sm_scale: float,
    sliding_window: int = 128,
    logical_page_size: int = 16,
    two_caches_same_buffer: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Runs Split-KV SWA + Split-KV CSA on rank `tp_rank`, returning (O_r [B,64,512] f32, updated_sw_cache, l_r [B,64] f32, m_r [B,64] f32)."""
    o_swa_r, updated_sw_cache, l_swa_r, m_swa_r = run_split_kv_swa_rank(
        q_full=q_full,
        new_kv=new_kv,
        sw_cache=sw_cache,
        kv_lens=swa_kv_lens,
        page_indices=swa_page_indices,
        cu_q_lens=swa_cu_q_lens,
        distribution=swa_distribution,
        tp_rank=tp_rank,
        tp_size=tp_size,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
    )
    if two_caches_same_buffer:
        cache_kv_nope = updated_sw_cache

    orig_num_tokens, actual_num_q_heads, actual_head_dim = q_full.shape
    num_tokens = align_to(orig_num_tokens, 2)
    pad_toks = num_tokens - orig_num_tokens
    if pad_toks > 0:
        q_work = jnp.pad(q_full, ((0, pad_toks), (0, 0), (0, 0)))
        o_swa_work = jnp.pad(o_swa_r, ((0, pad_toks), (0, 0), (0, 0)))
        l_swa_work = jnp.pad(l_swa_r, ((0, pad_toks), (0, 0)))
        m_swa_work = jnp.pad(
            m_swa_r, ((0, pad_toks), (0, 0)), constant_values=_MIN_F32
        )
        topk_work = jnp.pad(
            topk_indices_local, ((0, pad_toks), (0, 0)), constant_values=-1
        )
    else:
        q_work = q_full
        o_swa_work = o_swa_r
        l_swa_work = l_swa_r
        m_swa_work = m_swa_r
        topk_work = topk_indices_local

    q_padded = prepare_q_inputs(q_work)
    _, num_q_heads, head_dim = q_padded.shape
    topk_local = topk_work.shape[-1]  # 64
    num_valid_tokens = jnp.full((), num_tokens, dtype=jnp.int32)

    kv_lens_csa = jnp.sum(topk_work >= 0, axis=-1)
    gather_indices = jnp.where(topk_work < 0, ~topk_work, topk_work).ravel()

    eff_chunk_size = num_tokens if num_tokens <= 512 else min(num_tokens, 1024)
    eff_batch_size = 8 if num_tokens <= 32 else 16
    num_l_heads = align_to(num_q_heads, 128)

    swa_l_pad = jnp.pad(l_swa_work, ((0, 0), (0, num_l_heads - num_q_heads)))
    swa_m_pad = jnp.pad(
        m_swa_work, ((0, 0), (0, num_l_heads - num_q_heads)), constant_values=_MIN_F32
    )
    out_o = o_swa_work
    out_l = swa_l_pad
    out_m = swa_m_pad

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    for i in range(cdiv(num_tokens, eff_chunk_size)):
        start_pos = i * eff_chunk_size
        end_pos = min(start_pos + eff_chunk_size, num_tokens)
        chunk_size = end_pos - start_pos
        indices = gather_indices[start_pos * topk_local : end_pos * topk_local]

        batch_size = _largest_divisor(chunk_size, eff_batch_size)
        num_valid_indices = chunk_size * topk_local

        gathered_nope, gathered_rope = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices,
            num_valid_indices=num_valid_indices,
            top_k=topk_local * 2,
        )
        gathered_nope = gathered_nope.reshape(chunk_size, topk_local, -1)
        gathered_rope = gathered_rope.reshape(chunk_size, topk_local // 4, -1)
        lanes = gathered_nope.shape[-1]
        end_seq_idx = end_pos
        grid = (cdiv(end_seq_idx - start_pos, batch_size),)

        bq_f16_buf = pltpu.VMEM((2, batch_size, num_q_heads, head_dim), q_padded.dtype)
        bo_f32_buf = pltpu.VMEM((2, batch_size, num_q_heads, head_dim), jnp.float32)
        bl_buf = pltpu.VMEM((2, batch_size, num_l_heads), jnp.float32)
        scratch_shapes = [
            pltpu.VMEM((2, batch_size, topk_local * 4, lanes), jnp.uint8),
            pltpu.VMEM((2, batch_size, (topk_local // 4) * 2, lanes), jnp.bfloat16),
            bq_f16_buf,
            bo_f32_buf,
            bl_buf,
            bl_buf,
            bo_f32_buf,
            bl_buf,
            bl_buf,
            pltpu.SemaphoreType.DMA((9, 2)),
        ]
        scalar_prefetches = (
            kv_lens_csa,
            jnp.array([start_pos, end_seq_idx, num_valid_tokens], jnp.int32),
            jnp.zeros((2,), jnp.int32),
            jnp.full((2,), -1, jnp.int32),
        )
        out_o, out_l, out_m = pl.pallas_call(
            functools.partial(
                _split_kv_sparse_mla_kernel,
                sm_scale=sm_scale,
                batch_size=batch_size,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=[hbm] * 8,
                out_specs=[hbm] * 3,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                disable_bounds_checks=True,
            ),
            out_shape=[
                jax.ShapeDtypeStruct((num_tokens, num_q_heads, head_dim), jnp.float32),
                jax.ShapeDtypeStruct((num_tokens, num_l_heads), jnp.float32),
                jax.ShapeDtypeStruct((num_tokens, num_l_heads), jnp.float32),
            ],
            input_output_aliases={
                len(scalar_prefetches) + 3: 0,
                len(scalar_prefetches) + 6: 1,
                len(scalar_prefetches) + 7: 2,
            },
            name=f"FlashMLA-SplitKV-CSA-k_{topk_local}-bz_{batch_size}",
        )(
            *scalar_prefetches,
            q_padded,
            gathered_nope,
            gathered_rope,
            out_o,
            swa_l_pad,
            swa_m_pad,
            out_l,
            out_m,
        )

    return (
        out_o[:orig_num_tokens, :actual_num_q_heads, :actual_head_dim],
        updated_sw_cache,
        out_l[:orig_num_tokens, :actual_num_q_heads],
        out_m[:orig_num_tokens, :actual_num_q_heads],
    )


def combine_split_kv_tp8(
    o_r: jax.Array,  # [B, H=64, D=512] float32
    m_r: jax.Array,  # [B, H=64] float32
    l_r: jax.Array,  # [B, H=64] float32
    attention_sinks_local: jax.Array,  # [H_local=8] float32
    *,
    axis_name: str = "model",
    out_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Combines (O_r, m_r, l_r) across TP=8 ranks via exact online-softmax rescaling
    and scatters heads (64 -> 8) so each rank outputs [B, H_local=8, 512] directly."""
    tp_rank = lax.axis_index(axis_name)
    h_local = attention_sinks_local.shape[0]

    # Gather tiny (m_r, l_r) statistics [B, 64, 2] across all 8 TP ranks -> [8, B, 64, 2]
    ml_r = jnp.stack([m_r, l_r], axis=-1)
    ml_all = lax.all_gather(ml_r, axis_name=axis_name, axis=0, tiled=False)
    m_all = ml_all[..., 0]  # [8, B, 64]
    l_all = ml_all[..., 1]  # [8, B, 64]

    # Global max m = max_r m_r ([B, 64])
    m_global = jnp.max(m_all, axis=0)

    # Rescale this rank's unnormalized output O_r in float32, then cast to bf16 before ICI psum_scatter
    alpha_r = jnp.where(m_r > _MIN_F32, jnp.exp(m_r - m_global), 0.0)
    o_scaled = (o_r * alpha_r[:, :, None]).astype(out_dtype)

    # Hardware ICI ring reduce-scatter along heads: [B, 64, 512] -> [B, 8, 512]
    o_num_local = lax.psum_scatter(
        o_scaled, axis_name=axis_name, scatter_dimension=1, tiled=True
    ).astype(jnp.float32)

    # Compute local denominator for this rank's 8 heads: sum_r l_r * exp(m_r - m) + exp(sink - m)
    m_local = lax.dynamic_slice_in_dim(m_global, tp_rank * h_local, h_local, axis=1)
    m_all_local = lax.dynamic_slice_in_dim(m_all, tp_rank * h_local, h_local, axis=2)
    l_all_local = lax.dynamic_slice_in_dim(l_all, tp_rank * h_local, h_local, axis=2)
    alpha_all_local = jnp.where(
        m_all_local > _MIN_F32, jnp.exp(m_all_local - m_local[None, :, :]), 0.0
    )
    l_sum_local = jnp.sum(l_all_local * alpha_all_local, axis=0)
    sink_term = jnp.exp(attention_sinks_local[None, :] - m_local)
    l_tot_local = l_sum_local + sink_term

    return (o_num_local / l_tot_local[:, :, None]).astype(out_dtype)


def flashmla_attention_swa_tp8(
    q_local: jax.Array,  # [num_tokens, 8, 512] bf16 (sharded P(None, "model", None))
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
    attention_sinks_local: jax.Array,  # [8] float32 (sharded P("model"))
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    ced_decoder: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """TP=8 FlashMLA Split-KV entry point for DeepSeek-V4.1 SWA layers."""
    del main_cache_kv, main_kv_lens, extra, main_page_indices, main_cu_q_lens, main_distribution, ced_decoder
    tp_rank = lax.axis_index("model")
    q_full = lax.all_gather(q_local, axis_name="model", axis=1, tiled=True)

    o_r, updated_sw_cache, l_r, m_r = run_split_kv_swa_rank(
        q_full=q_full,
        new_kv=new_kv,
        sw_cache=sw_cache,
        kv_lens=swa_kv_lens,
        page_indices=swa_page_indices.flatten(),
        cu_q_lens=swa_cu_q_lens,
        distribution=swa_distribution,
        tp_rank=tp_rank,
        tp_size=8,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
    )
    o_local = combine_split_kv_tp8(
        o_r, m_r, l_r, attention_sinks_local, axis_name="model", out_dtype=q_local.dtype
    )
    return o_local, updated_sw_cache


def flashmla_attention_csa_tp8(
    q_local: jax.Array,  # [num_tokens, 8, 512] bf16 (sharded P(None, "model", None))
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,  # [num_tokens, 512] int32 encoded top-k indices
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks_local: jax.Array,  # [8] float32 (sharded P("model"))
    main_cache_rope: jax.Array,
    union_gather_indices: jax.Array | None = None,
    cand_bits_tiled: jax.Array | None = None,
    n_unique: jax.Array | None = None,
    *,
    precomputed_union: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    two_caches_same_buffer: bool,
    ced_decoder: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """TP=8 FlashMLA Split-KV entry point for DeepSeek-V4.1 CSA layers."""
    del main_kv_lens, main_page_indices, union_gather_indices, cand_bits_tiled, n_unique, precomputed_union, ced_decoder
    if main_cache_kv.shape[0] == 0 or main_cache_rope.shape[0] == 0:
        return jnp.zeros_like(q_local), sw_cache

    tp_rank = lax.axis_index("model")
    q_full = lax.all_gather(q_local, axis_name="model", axis=1, tiled=True)

    topk_total = extra.shape[-1]
    topk_per_rank = topk_total // 8
    topk_local = lax.dynamic_slice_in_dim(
        extra, tp_rank * topk_per_rank, topk_per_rank, axis=1
    )

    o_r, updated_sw_cache, l_r, m_r = run_split_kv_csa_rank(
        q_full=q_full,
        new_kv=new_kv,
        sw_cache=sw_cache,
        swa_kv_lens=swa_kv_lens,
        swa_page_indices=swa_page_indices.flatten(),
        swa_cu_q_lens=swa_cu_q_lens,
        swa_distribution=swa_distribution,
        cache_kv_nope=main_cache_kv,
        cache_kv_rope=main_cache_rope,
        topk_indices_local=topk_local,
        main_cu_q_lens=main_cu_q_lens,
        main_distribution=main_distribution,
        tp_rank=tp_rank,
        tp_size=8,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        two_caches_same_buffer=two_caches_same_buffer,
    )
    o_local = combine_split_kv_tp8(
        o_r, m_r, l_r, attention_sinks_local, axis_name="model", out_dtype=q_local.dtype
    )
    return o_local, updated_sw_cache
