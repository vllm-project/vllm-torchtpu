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

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather

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


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


def _largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of ``x`` that is <= ``cap``."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


_GATHER_PAGE_CHUNK = 128
# Query tokens per grid step of `gather_row_indices`.
GATHER_INDEX_BLOCK_TOKENS = 8


def _gather_row_indices_kernel(
    # Prefetch
    block_ids_ref,  # [num_blocks] token blocks to fill; the grid ends before the rest
    seq_ids_ref,  # [num_tokens] request of each token
    # Input
    page_table_ref,  # i32[max_num_seqs, padded_pps]
    topk_ref,  # i32[block_tokens, topk]
    # Output
    out_ref,  # i32[block_tokens * topk // 128, 128]
    *,
    page_size: int,
    pages_per_seq: int,
    total_num_pages: int,
):
    block_tokens, topk = topk_ref.shape
    num_tokens = seq_ids_ref.shape[0]
    first = block_ids_ref[pl.program_id(0)] * block_tokens
    # Per-token page-table window: row t is the page table of token t's
    # request. A partial last block repeats the last token's request.
    seqs = [
        seq_ids_ref[jnp.minimum(first + t, num_tokens - 1)] for t in range(block_tokens)
    ]
    windows = jnp.concatenate(
        [page_table_ref[pl.ds(seq, 1), :] for seq in seqs], axis=0
    )  # i32[block_tokens, padded_pps]
    topk_indices = topk_ref[...]
    logical = jnp.clip(topk_indices // page_size, 0, pages_per_seq - 1)
    page_ids = jnp.zeros_like(logical)
    for c in range(page_table_ref.shape[1] // _GATHER_PAGE_CHUNK):
        window_chunk = windows[:, c * _GATHER_PAGE_CHUNK : (c + 1) * _GATHER_PAGE_CHUNK]
        local = logical - c * _GATHER_PAGE_CHUNK
        gathered = jnp.take_along_axis(
            window_chunk, jnp.clip(local, 0, _GATHER_PAGE_CHUNK - 1), axis=1
        )
        page_ids = jnp.where(
            (local >= 0) & (local < _GATHER_PAGE_CHUNK), gathered, page_ids
        )
    token_offset = topk_indices % page_size

    # For the "-1" padding elements in topk_indices, we scatter the
    # corresponding page_ids and token_offset to avoid gather memory access
    # hotspotting. 104729 and 15485863 are randomly chosen large prime numbers.
    flat_element_index = (
        first + lax.broadcasted_iota(jnp.int32, topk_indices.shape, 0)
    ) * topk + lax.broadcasted_iota(jnp.int32, topk_indices.shape, 1)
    is_padding = topk_indices == -1
    page_ids = jnp.where(
        is_padding, (flat_element_index * 104729) % total_num_pages, page_ids
    )
    token_offset = jnp.where(
        is_padding, (flat_element_index * 15485863) % page_size, token_offset
    )
    indices = page_ids * page_size + token_offset
    # Row-major [num_tokens * topk] as 128-lane rows: token t, lanes
    # [128 c, 128 c + 128) go to row t * topk // 128 + c.
    row_chunks = topk // 128
    for c in range(row_chunks):
        out_ref[pl.ds(c, block_tokens, stride=row_chunks), :] = indices[
            :, c * 128 : (c + 1) * 128
        ]


def gather_row_indices(
    topk_indices: jax.Array,  # i32[num_tokens, topk]  (position within seq)
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    seq_ids_segment: jax.Array,  # i32[num_tokens]  (token -> seq id)
    range_ends: jax.Array,  # i32[num_ranges]
    *,
    max_num_seqs: int,
    page_size: int,
    total_num_pages: int,
    range_size: int,
) -> jax.Array:
    """Cache row of every top-k entry of the tokens the gather path reads.

    Range r covers tokens [r * range_size, range_ends[r]); the grid walks the
    token blocks those ranges touch only. Returns the row-major
    [num_tokens * topk] row indices as i32[num_tokens * topk // 128, 128];
    rows of tokens outside every range are left unwritten.
    """
    num_tokens, topk = topk_indices.shape
    block_tokens = min(GATHER_INDEX_BLOCK_TOKENS, num_tokens)
    # Every range starts on a block.
    assert num_tokens <= range_size or range_size % block_tokens == 0
    assert topk % 128 == 0
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    padded_pps = align_to(pages_per_seq, _GATHER_PAGE_CHUNK)
    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))

    # The touched blocks of every range, in order, at the front.
    num_blocks = cdiv(num_tokens, block_tokens)
    range_starts = jnp.arange(range_ends.shape[0], dtype=jnp.int32) * range_size
    range_blocks = cdiv(jnp.maximum(range_ends - range_starts, 0), block_tokens)
    range_offsets = jnp.cumsum(range_blocks) - range_blocks
    slot = jnp.arange(num_blocks, dtype=jnp.int32)[:, None]
    in_range = (slot >= range_offsets) & (slot < range_offsets + range_blocks)
    block_ids = jnp.sum(
        jnp.where(in_range, range_starts // block_tokens + slot - range_offsets, 0),
        axis=1,
    )
    num_touched = jnp.sum(range_blocks)

    row_chunks = topk // 128
    return pl.pallas_call(
        functools.partial(
            _gather_row_indices_kernel,
            page_size=page_size,
            pages_per_seq=pages_per_seq,
            total_num_pages=total_num_pages,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(
                    (block_tokens, topk), lambda i, block_ids, *_: (block_ids[i], 0)
                ),
            ],
            out_specs=pl.BlockSpec(
                (block_tokens * row_chunks, 128),
                lambda i, block_ids, *_: (block_ids[i], 0),
            ),
            grid=(num_touched,),
        ),
        out_shape=jax.ShapeDtypeStruct((num_tokens * row_chunks, 128), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        name="gather_row_indices",
    )(block_ids, seq_ids_segment.astype(jnp.int32), page_table, topk_indices)


# DSV4 FP8 nope layout: 448 e4m3 values, then one e8m0 scale per lane of a
# 64-lane period -- 64 quantization groups of 7 dims, group j holding the dims
# {j, j + 64, ... j + 384}. The 64 scale bytes fill the row's padding, so the
# record is still 448 + 64 = 512 bytes.
NOPE_DIM = 448
NOPE_SCALE_PERIOD = 64


def _dequant_dsv4_fp8(bkv_nope: jax.Array):
    """Dequantize FP8 values to BF16."""
    nope_fp8 = pltpu.bitcast(bkv_nope[:, :NOPE_DIM], jnp.float8_e4m3fn).astype(
        jnp.bfloat16
    )
    # scales[d] = scales[d % 64]
    tail = pltpu.bitcast(bkv_nope[:, -128:], jnp.float8_e8m0fnu).astype(jnp.bfloat16)
    lane = lax.broadcasted_iota(jnp.int32, tail.shape, 1)
    tile = jnp.where(
        lane < NOPE_SCALE_PERIOD, pltpu.roll(tail, NOPE_SCALE_PERIOD, 1), tail
    )
    nope_scales = jnp.concatenate([tile] * 4, axis=1)[:, :NOPE_DIM]
    return (nope_fp8 * nope_scales).astype(jnp.bfloat16)


def _attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    start_end_seq_idx_ref,  # [3] (start_seq_idx, end_seq_idx, num_valid_tokens)
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [2] seq_idx_start of the in-flight bo DMA per bo sem
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_nope_hbm_ref,  # int32[chunk_size, topk, 128]
    cache_kv_rope_hbm_ref,  # int32[chunk_size, topk // 4, 128]
    swa_accumution_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    swa_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    bkv_nope_x2_ref,  # [2, batch_size, page_size * 4, 128]
    bkv_rope_x2_ref,  # [2, batch_size, page_size, rope_dim]
    bq_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bo_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bl_x2_ref,  # [2, batch_size, num_l_heads]
    bm_x2_ref,  # [2, batch_size, num_l_heads]
    swa_acc_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    sems,  # [7, 2]
    *,
    sm_scale: float,
    batch_size: int = 1,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape

    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    assert kv_lens_ref.shape[0] == num_tokens
    # csa_gather emits int32 words; these are the uint8 nope slabs
    # [.., topk * 4, 128] and bf16 rope rows [.., topk // 2, 128] they encode,
    # a free bitcast of the same bytes.
    cache_kv_nope_hbm_ref = cache_kv_nope_hbm_ref.bitcast(jnp.uint8)
    cache_kv_rope_hbm_ref = cache_kv_rope_hbm_ref.bitcast(jnp.bfloat16)
    # The gathered rope buffer carries two top-k entries per row (csa_gather).
    bkv_sz = cache_kv_rope_hbm_ref.shape[1] * 2

    q_dtype = q_hbm_ref.dtype
    q_packing = get_dtype_packing(q_dtype)
    # Validate against the KV dtype.
    assert o_hbm_ref.dtype == q_dtype

    assert head_dim % 128 == 0
    assert num_q_heads % q_packing == 0

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    num_valid_tokens = start_end_seq_idx_ref[2]

    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def flash_attention_step1_qk_softmax(
        q,  # [bq_sz * num_q_heads, head_dim]
        kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
        kv_len,  # scalar
        swa_m,  # [bq_sz * num_q_heads],
        swa_l,  # [bq_sz * num_q_heads],
        attention_sinks,  # [num_q_heads]
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape == (bkv_sz, head_dim)

        # Follow FlashAttention-2 forward pass.
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        k_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(k_span < kv_len, s, jnp.finfo(s.dtype).min)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = swa_m
        m_curr = jnp.maximum(m_prev, s_rowmax)
        p = jnp.exp(s - m_curr)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        l_prev = swa_l
        l_curr = exp_m_diff * l_prev + p_rowsum
        exp_attention_sinks = jnp.exp(attention_sinks - m_curr)
        l_total = l_curr + exp_attention_sinks

        return p, exp_m_diff, l_total

    def flash_attention_step2_pv(
        p,
        kv,
        exp_m_diff,
        swa_acc,
        l_total,
    ):
        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)

        o_prev = swa_acc
        acc = exp_m_diff * o_prev + pv
        out = (
            lax.div(acc, l_total)
            if q_dtype == jnp.float32
            else (acc * pl.reciprocal(l_total, approx=True)).astype(q_dtype)
        )
        return out

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

        # The index into cache_kv_hbm_ref should be relative to the current
        # chunk.
        page_idx_start = seq_idx_start - start_seq_idx
        if not wait:
            _async_copy(
                cache_kv_nope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                bkv_nope_vmem_ref,
                sem_nope,
                wait,
            )
            _async_copy(
                cache_kv_rope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                bkv_rope_vmem_ref,
                sem_rope,
                wait,
            )
        else:
            dst_nope = bkv_nope_vmem_ref
            _async_copy(src=dst_nope, dst=dst_nope, sem=sem_nope, wait=True)
            dst_rope = bkv_rope_vmem_ref
            _async_copy(src=dst_rope, dst=dst_rope, sem=sem_rope, wait=True)

    def _fetch_bq_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, :]

        if not wait:
            _async_copy(
                q_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bq_vmem_ref,
                sem,
                wait,
            )
        else:
            _async_copy(src=bq_vmem_ref, dst=bq_vmem_ref, sem=sem, wait=True)

    def _send_bo_batch(seq_idx_start, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        sz = jnp.clip(num_valid_tokens - seq_idx_start, 1, batch_size)
        vmem_ref = bo_x2_ref.at[bo_sem_idx, pl.ds(0, sz)]

        if not wait:
            # Remember where this DMA started so its wait can rebuild the same
            # `sz`; a mismatch here deadlocks on the DMA semaphore.
            bo_ids_ref[bo_sem_idx] = seq_idx_start
            _async_copy(
                vmem_ref,
                o_hbm_ref.at[pl.ds(seq_idx_start, sz)],
                sem,
                wait,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def _fetch_swa_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem_acc = sems.at[3, bq_sem_idx]
        sem_l = sems.at[4, bq_sem_idx]
        sem_m = sems.at[5, bq_sem_idx]

        if not wait:
            _async_copy(
                swa_accumution_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                swa_acc_x2_ref.at[bq_sem_idx, :],
                sem_acc,
                wait=False,
            )
            _async_copy(
                swa_l_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bl_x2_ref.at[bq_sem_idx, :],
                sem_l,
                wait=False,
            )
            _async_copy(
                swa_m_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bm_x2_ref.at[bq_sem_idx, :],
                sem_m,
                wait=False,
            )

        else:
            dst_acc = swa_acc_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_acc, dst=dst_acc, sem=sem_acc, wait=True)

            dst_l = bl_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_l, dst=dst_l, sem=sem_l, wait=True)

            dst_m = bm_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_m, dst=dst_m, sem=sem_m, wait=True)

    def start_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx)

    def wait_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, wait=True)

    def start_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx)

    def wait_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx, wait=True)

    def start_fetch_swa_batch(seq_idx_start, bq_sem_idx):
        return _fetch_swa_batch(seq_idx_start, bq_sem_idx)

    def wait_fetch_swa_batch(seq_idx_start, bq_sem_idx):
        return _fetch_swa_batch(seq_idx_start, bq_sem_idx, wait=True)

    def start_send_bo_batch(seq_idx_start, bo_sem_idx):
        return _send_bo_batch(seq_idx_start, bo_sem_idx)

    def wait_send_bo_batch(bo_sem_idx):
        # `sz` must match the started DMA exactly, so recover its
        # seq_idx_start rather than using the current step's.
        old_seq_idx_start = bo_ids_ref[bo_sem_idx]

        @pl.when(old_seq_idx_start >= 0)
        def _():
            _send_bo_batch(old_seq_idx_start, bo_sem_idx, wait=True)

    def load_bq(bq_sem_idx, batch_idx):
        q = bq_x2_ref.at[bq_sem_idx, batch_idx][...]
        return q

    def load_bkv(bkv_sem_idx, batch_idx):
        bkv_nope = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx][...]
        # The gather nope is (4, 128) u8, reshape it to (1, 512) u8.
        bkv_nope = bkv_nope.reshape(bkv_sz, -1)
        bkv_nope = _dequant_dsv4_fp8(bkv_nope)

        # The gathered rope block is [bkv_sz // 2, 2 * rope_dim]: entry i in
        # lanes 0:rope_dim, entry i + bkv_sz // 2 in the upper half (see
        # csa_gather). Mosaic has no [m, 2n] -> [2m, n] shape cast, but the
        # lane slice + row concatenate below is one lane rotate per vreg.
        bkv_rope = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx][...]
        bkv_rope = bkv_rope.reshape(bkv_sz // 2, -1)
        rope_dim = bkv_rope.shape[-1] // 2
        bkv_rope = jnp.concatenate(
            [bkv_rope[:, :rope_dim], bkv_rope[:, rope_dim:]], axis=0
        )
        bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)

        # In vLLM, multiple caches may overlay on the same KV Tensor. For example,
        # compressor state cache write data in bfloat16 / float32 format, certain
        # byte pattern are interpreted as NaN in FP8, e.g. float8_e8m0fnu byte 0xFF
        # decodes to NaN.
        # We need to mask out the data by the actual kv_len to avoid NaN propagting
        # to the downstream computation.
        kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
        k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
        bkv = jnp.where(k_span < kv_len, bkv, 0)
        return bkv

    def load_swa_output(bq_sem_idx, batch_idx):
        swa_acc = swa_acc_x2_ref[bq_sem_idx, batch_idx, ...]
        swa_l = bl_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        swa_m = bm_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        return swa_acc, swa_l, swa_m

    def process():
        def get_next_seq_ids(seq_idx, bi_sem_idx):
            next_seq_idx = seq_idx + batch_size
            next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bi_sem_idx

        bi_sem_idx = sem_ids_ref[0]
        next_seq_idx, next_bi_sem_idx = get_next_seq_ids(
            batch_start_seq_idx, bi_sem_idx
        )

        # Prefetch next seq
        @pl.when(next_seq_idx < end_seq_idx)
        def prefetch_next_seq():
            sem_ids_ref[0] = next_bi_sem_idx
            start_fetch_bq_batch(next_seq_idx, next_bi_sem_idx)
            start_fetch_swa_batch(next_seq_idx, next_bi_sem_idx)
            start_fetch_bkv_batch(next_seq_idx, next_bi_sem_idx)

        bo_sem_idx = sem_ids_ref[1]
        sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)
        attention_sinks = attention_sinks_ref[...][..., None]

        prev_p = None
        prev_bkv = None
        prev_exp_m_diff = None
        prev_l = None
        prev_swa_acc = None
        prev_out = None

        # Wait for cur blocks if not ready yet
        wait_fetch_bq_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_swa_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx)

        @pl.when(pl.program_id(0) >= 2)
        def _wait_send():
            wait_send_bo_batch(bo_sem_idx)

        for batch_idx in range(batch_size):
            bkv = load_bkv(bi_sem_idx, batch_idx)
            bq = load_bq(bi_sem_idx, batch_idx)

            if prev_out is not None:
                # Artificial dependency to force MXU/VPU interleaving by limiting
                # LLO's QK runahead
                # We use jnp.where to prevent XLA from optimizing the dependency away.
                # prev_out won't be inf in practice.
                bq = jnp.where(prev_out == jnp.inf, prev_out, bq)

            swa_acc, swa_l, swa_m = load_swa_output(bi_sem_idx, batch_idx)

            p, exp_m_diff, l_total = flash_attention_step1_qk_softmax(
                bq,
                bkv,
                kv_lens_ref[batch_start_seq_idx + batch_idx],
                swa_m,
                swa_l,
                attention_sinks,
            )

            if prev_p is not None:
                assert prev_bkv is not None
                assert prev_exp_m_diff is not None
                assert prev_l is not None
                out = flash_attention_step2_pv(
                    prev_p,
                    prev_bkv,
                    prev_exp_m_diff,
                    prev_swa_acc,
                    prev_l,
                )

                # Store output from acc to bo.
                bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out
                prev_out = out

            prev_p = p
            prev_bkv = bkv
            prev_exp_m_diff = exp_m_diff
            prev_l = l_total
            prev_swa_acc = swa_acc

        # end of pipelining loop
        out = flash_attention_step2_pv(
            prev_p, prev_bkv, prev_exp_m_diff, prev_swa_acc, prev_l
        )
        bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out
        start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq_batch(batch_start_seq_idx, 0)
        start_fetch_swa_batch(batch_start_seq_idx, 0)
        start_fetch_bkv_batch(batch_start_seq_idx, 0)

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        wait_send_bo_batch(0)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_1():
            wait_send_bo_batch(1)

    ### ------- Kernel end ------- ###


# Dense path for chunks whose valid tokens all belong to one request: attend
# over that request's contiguous compressed KV pages with the top-k selection
# as a mask, instead of gathering top-k rows per token. Query tokens per grid
# step; every KV page pushed to the MXU serves DENSE_BLOCK_TOKENS *
# num_q_heads query rows.
DENSE_BLOCK_TOKENS = 32
# The dense path reads every KV page up to the block's largest selected
# position, so its cost grows with the request's compressed length while the
# gather path's does not. Past this many pages a block keeps the gather path.
# Empirical, to be tuned by measurement: by static LLO cycle counts a dense
# block over 8 pages of 256 rows costs about what the gather path measures.
DENSE_MAX_KV_PAGES = 8
# KV pages per KV step: the f32 PV output and the running max, sum and
# accumulator are read and written once per step, whatever its key count.
DENSE_KV_PAGES_PER_STEP = 2
# Unroll of the per-token mask build, whose iterations are independent.
DENSE_MASK_UNROLL = 32
# Query tokens per softmax group in a KV step. The step's score tile does not
# fit in vector registers either way; groups let the MXU run one group's QK
# while the VPU runs another's softmax, at the cost of pushing the KV tile to
# the MXU once per group.
DENSE_GROUP_TOKENS = 4
# Smaller chunks are the op's decode-bucket tiling and keep the gather path.
DENSE_MIN_CHUNK_TOKENS = 128
# The membership mask holds KV position x at row x >> 7, lane x & 127.
_MASK_LANE_BITS = 7
_MASK_LANES = 1 << _MASK_LANE_BITS


def _dense_attention_kernel(
    # Prefetch
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    num_valid_ref,  # [1] valid tokens of the step
    block_ids_ref,  # [num_blocks] the dense blocks, then unused entries
    block_seqs_ref,  # [num_blocks] request of each block
    block_pages_ref,  # [num_blocks] KV pages each block reads
    kv_bounds_ref,  # [num_blocks] 1 + largest selected KV position per block
    prefix_ref,  # [num_blocks] 1 if every token selected all of [0, its bound)
    seq_bounds_ref,  # [num_blocks] largest kv_bounds of the block's request
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_ref,  # [bq_sz, num_q_heads, head_dim]
    topk_ref,  # i32[bq_sz, topk]
    nope_hbm_ref,  # int32[total_num_pages, page_size, 128]
    rope_hbm_ref,  # int32[total_num_pages, page_size // 4, 128]
    swa_acc_ref,  # [bq_sz, num_q_heads, head_dim]
    swa_l_ref,  # float32[bq_sz, num_l_heads]
    swa_m_ref,  # float32[bq_sz, num_l_heads]
    # Output
    o_ref,  # [bq_sz, num_q_heads, head_dim]
    # Scratch
    mask_ref,  # float32[mask_rows, bq_sz, 128] top-k membership counts
    rope_kv_ref,  # float32[page_size, 128] decoded RoPE rows of one page
    m_ref,  # float32[bq_sz * num_q_heads, 128]
    l_ref,  # float32[bq_sz * num_q_heads, 128]
    acc_ref,  # float32[bq_sz * num_q_heads, head_dim]
    kv_cache_ref,  # [cache_pages * page_size, head_dim] decoded KV pages
    nope_stage_ref,  # int32[cache_pages, page_size, 128] fetched NoPE pages
    rope_stage_ref,  # int32[cache_pages, page_size // 4, 128] fetched RoPE pages
    page_sems,  # DMA semaphores [2, cache_pages]
    cache_state_ref,  # i32[4] (decoded request, decoded pages, fetch range)
    *,
    sm_scale: float,
    pages_per_seq: int,
):
    """Grid (dense block,): FlashAttention over one request's pages.

    Each grid step runs all of its block's KV steps, up to the block's largest
    selected position: steps of DENSE_KV_PAGES_PER_STEP pages, the last one
    shorter if the block's page count is not a multiple. The running max, sum
    and accumulator start from the SWA partials, as the gather kernel's do,
    and a KV position contributes to a token only if the token's top-k
    selected it. Same keys as the gather path, so the output differs from it
    only by the order of the softmax reductions.

    A request's pages are fetched and decoded once, into `kv_cache_ref`, and
    reused by its other blocks; the grid runs sequentially, so the cache
    persists. Each grid step fetches the pages the next block adds while it
    computes.
    """
    kv_group = DENSE_KV_PAGES_PER_STEP
    bq_sz, num_q_heads, head_dim = q_ref.shape
    page_size = nope_stage_ref.shape[1]
    topk = topk_ref.shape[1]
    mask_rows = mask_ref.shape[0]
    num_rows = bq_sz * num_q_heads
    rope_rows = rope_stage_ref.shape[1]
    lanes = rope_stage_ref.shape[2]
    rope_words = csa_cache_layout.ROPE_WORDS
    rope_tokens_per_row = csa_cache_layout.ROPE_TOKENS_PER_ROW
    halves = page_size // _MASK_LANES

    block = pl.program_id(0)
    bq_idx = block_ids_ref[block]
    kv_bound = kv_bounds_ref[bq_idx]
    seq = block_seqs_ref[bq_idx]
    seq_bound = seq_bounds_ref[bq_idx]

    def broadcast_minor(src, shape):
        # src is lane-replicated [rows, 128]; tile it to `shape`'s minor dim.
        return jnp.concatenate([src] * (shape[-1] // src.shape[-1]), axis=-1)

    def page_copies(req, page):
        phys = page_indices_ref[req * pages_per_seq + page]
        return (
            pltpu.make_async_copy(
                nope_hbm_ref.at[phys], nope_stage_ref.at[page], page_sems.at[0, page]
            ),
            pltpu.make_async_copy(
                rope_hbm_ref.at[phys], rope_stage_ref.at[page], page_sems.at[1, page]
            ),
        )

    def start_fetch(req, lo, hi):
        cache_state_ref[2] = lo
        cache_state_ref[3] = hi

        @pl.loop(lo, hi)
        def _(page):
            for copy in page_copies(req, page):
                copy.start()

    @pl.when(block == 0)
    def init_cache():
        cache_state_ref[0] = -1
        cache_state_ref[1] = 0
        start_fetch(seq, 0, block_pages_ref[bq_idx])

    # The accumulator starts from swa_acc in the first KV step (`compute`).
    def init_max_sum():
        for t in range(bq_sz):
            rows = pl.ds(t * num_q_heads, num_q_heads)
            swa_m = swa_m_ref[t, :num_q_heads][:, None]
            swa_l = swa_l_ref[t, :num_q_heads][:, None]
            m_ref[rows, :] = jnp.broadcast_to(swa_m, (num_q_heads, 128))
            l_ref[rows, :] = jnp.broadcast_to(swa_l, (num_q_heads, 128))

    # counts[x >> 7, x & 127] = how often the token selected position x,
    # as a one-hot product on the MXU; -1 entries match no row.
    def build_mask(t, carry):
        row = topk_ref[pl.ds(t, 1), :]  # [1, topk]
        hi = jnp.right_shift(row, _MASK_LANE_BITS)
        lo = jnp.bitwise_and(row, _MASK_LANES - 1)
        hi_onehot = lax.broadcasted_iota(jnp.int32, (mask_rows, topk), 0) == hi
        lo_onehot = lax.broadcasted_iota(jnp.int32, (_MASK_LANES, topk), 0) == lo
        counts = lax.dot_general(
            hi_onehot.astype(jnp.float32),
            lo_onehot.astype(jnp.float32),
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        mask_ref[:, pl.ds(t, 1), :] = counts[:, None, :]
        return carry

    # Each branch initializes the running max and sum itself, so that it
    # overlaps with the mask build in one basic block.
    is_prefix = prefix_ref[bq_idx] != 0

    @pl.when(jnp.logical_not(is_prefix))
    def init_topk_mask():
        init_max_sum()
        lax.fori_loop(0, bq_sz, build_mask, None, unroll=DENSE_MASK_UNROLL)

    # Where each token's selection is a whole prefix of the KV positions (the
    # request is no longer than top-k), its mask is just its bound.
    @pl.when(is_prefix)
    def init_prefix_mask():
        init_max_sum()
        bound = jnp.max(topk_ref[...], axis=1, keepdims=True) + 1
        lane = lax.broadcasted_iota(jnp.int32, (bq_sz, _MASK_LANES), 1)
        for h in range(mask_rows):
            mask_ref[h] = (lane < bound - h * _MASK_LANES).astype(jnp.float32)

    def load_page(page):
        nope_ref, rope_ref = nope_stage_ref.at[page], rope_stage_ref.at[page]
        bkv_nope = _dequant_dsv4_fp8(
            pltpu.bitcast(nope_ref[...], jnp.uint8).reshape(page_size, -1)
        )

        # RoPE row r holds tokens 4r..4r+3, token k in words [32k, 32k+32),
        # word w = bf16 w (low half) | bf16 w+32 (high half) (see
        # `csa_cache_layout`). Move token k's 64 values to lanes 0..63 and
        # store them to row 4r+k.
        words = rope_ref[...]
        rope_lo = pltpu.bitcast(jnp.left_shift(words, 16), jnp.float32)
        rope_hi = pltpu.bitcast(jnp.bitwise_and(words, jnp.int32(-65536)), jnp.float32)
        lane = lax.broadcasted_iota(jnp.int32, words.shape, 1)

        def rotate(x, shift):
            shift %= lanes
            return pltpu.roll(x, shift, 1) if shift else x

        for k in range(rope_tokens_per_row):
            token = jnp.where(
                lane < rope_words,
                rotate(rope_lo, -rope_words * k),
                rotate(rope_hi, rope_words - rope_words * k),
            )
            rope_kv_ref[pl.ds(k, rope_rows, stride=rope_tokens_per_row), :] = token
        bkv_rope = rope_kv_ref[:, : 2 * rope_words].astype(jnp.bfloat16)
        bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)

        # Rows past every position the request selected may hold another
        # cache's bytes, which can decode to NaN (see the gather kernel's
        # `load_bkv`). Rows below are the request's own; a block masks those
        # it did not select.
        k_pos = page * page_size + lax.broadcasted_iota(jnp.int32, (page_size, 1), 0)
        return jnp.where(k_pos < seq_bound, bkv, 0)

    @pl.when(cache_state_ref[0] != seq)
    def new_request():
        cache_state_ref[0] = seq
        cache_state_ref[1] = 0

    # Decode the pages this block adds, which the previous grid step fetched.
    @pl.loop(cache_state_ref[2], cache_state_ref[3])
    def _(page):
        for copy in page_copies(seq, page):
            copy.wait()
        rows = pl.ds(pl.multiple_of(page * page_size, page_size), page_size)
        kv_cache_ref[rows, :] = load_page(page)

    decoded_pages = jnp.maximum(cache_state_ref[1], cache_state_ref[3])
    cache_state_ref[1] = decoded_pages

    @pl.when(block + 1 < pl.num_programs(0))
    def fetch_next_block():
        next_idx = block_ids_ref[block + 1]
        next_seq = block_seqs_ref[next_idx]
        have = jnp.where(next_seq == seq, decoded_pages, 0)
        start_fetch(next_seq, have, jnp.maximum(have, block_pages_ref[next_idx]))

    def compute(first_page, num_pages=kv_group, from_swa=False):
        step_len = num_pages * page_size
        bkv = kv_cache_ref[
            pl.ds(pl.multiple_of(first_page * page_size, page_size), step_len), :
        ]

        selected = jnp.concatenate(
            [mask_ref[first_page * halves + h] for h in range(num_pages * halves)],
            axis=-1,
        )  # [bq_sz, step_len]
        group_rows = DENSE_GROUP_TOKENS * num_q_heads

        def scores(g):
            tokens = pl.ds(g * DENSE_GROUP_TOKENS, DENSE_GROUP_TOKENS)
            q = q_ref[tokens].reshape(group_rows, head_dim)
            s = jnp.einsum("nd,md->nm", q, bkv, preferred_element_type=jnp.float32)
            s *= sm_scale
            sel = selected[g * DENSE_GROUP_TOKENS : (g + 1) * DENSE_GROUP_TOKENS]
            return jnp.where(
                sel[:, None, :] > 0,
                s.reshape(DENSE_GROUP_TOKENS, num_q_heads, step_len),
                jnp.finfo(s.dtype).min,
            ).reshape(group_rows, step_len)

        # Group g + 1's QK is issued before group g's PV, so the MXU works on
        # one group while the VPU and EUP run the other's softmax.
        assert bq_sz % DENSE_GROUP_TOKENS == 0
        num_groups = bq_sz // DENSE_GROUP_TOKENS
        s_next = scores(0)
        for g in range(num_groups):
            s = s_next
            if g + 1 < num_groups:
                s_next = scores(g + 1)
            rows = pl.ds(g * group_rows, group_rows)
            m_prev, l_prev = m_ref[rows, :], l_ref[rows, :]
            m_curr = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
            p = jnp.exp(s - broadcast_minor(m_curr, s.shape))
            exp_m_diff = jnp.exp(m_prev - m_curr)
            l_sum = jnp.sum(p, axis=1, keepdims=True)
            l_ref[rows, :] = exp_m_diff * l_prev + l_sum
            m_ref[rows, :] = m_curr
            pv = jnp.einsum("nm,md->nd", p, bkv, preferred_element_type=jnp.float32)
            exp_m_diff = broadcast_minor(exp_m_diff, pv.shape)
            if from_swa:
                tokens = pl.ds(g * DENSE_GROUP_TOKENS, DENSE_GROUP_TOKENS)
                acc_prev = swa_acc_ref[tokens].reshape(group_rows, head_dim)
                acc_prev = acc_prev.astype(jnp.float32)
            else:
                acc_prev = acc_ref[rows, :]
            acc_ref[rows, :] = exp_m_diff * acc_prev + pv

    # Steps of kv_group pages, then one page if the block reads an odd count:
    # a step over a page past the block's last would do its work all masked.
    num_pages = cdiv(kv_bound, page_size)
    num_full_steps = num_pages // kv_group
    tail_pages = num_pages % kv_group

    # No token of the block selected any key.
    @pl.when(num_pages == 0)
    def keep_swa_acc():
        acc_ref[...] = swa_acc_ref[...].reshape(num_rows, head_dim).astype(jnp.float32)

    @pl.when(num_full_steps > 0)
    def first_kv_step():
        compute(0, from_swa=True)

    @pl.loop(1, num_full_steps)
    def _(kv_step):
        compute(kv_step * kv_group)

    tail_page = num_full_steps * kv_group
    for pages in range(1, kv_group):

        @pl.when((tail_pages == pages) & (num_full_steps == 0))
        def first_tail_step(*, pages=pages):
            compute(tail_page, pages, from_swa=True)

        @pl.when((tail_pages == pages) & (num_full_steps > 0))
        def tail_step(*, pages=pages):
            compute(tail_page, pages)

    sinks = jnp.broadcast_to(attention_sinks_ref[...][:, None], (num_q_heads, 128))
    # Token by token, so no [num_rows, head_dim] temporary is live at once.
    for t in range(bq_sz):
        rows = pl.ds(t * num_q_heads, num_q_heads)
        l_total = l_ref[rows, :] + jnp.exp(sinks - m_ref[rows, :])
        l_total = broadcast_minor(l_total, (num_q_heads, head_dim))
        acc = acc_ref[rows, :]
        out = (
            lax.div(acc, l_total)
            if o_ref.dtype == jnp.float32
            else (acc * pl.reciprocal(l_total, approx=True)).astype(o_ref.dtype)
        )
        o_ref[t] = out

    # Padding rows keep q, as the gather kernel leaves them. Only the last
    # valid block can hold any.
    num_valid = num_valid_ref[0] - bq_idx * bq_sz

    @pl.when(num_valid < bq_sz)
    def keep_padding_rows():
        for t in range(bq_sz):
            o_ref[t] = jnp.where(t < num_valid, o_ref[t], q_ref[t])


def _dense_attention(
    q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # int32[total_num_pages, page_size // 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    seq_ids_segment: jax.Array,  # i32[max_num_tokens]
    attention_sinks: jax.Array,  # float32[num_q_heads]
    swa_accumution: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[max_num_tokens, num_l_heads]
    swa_m: jax.Array,  # float32[max_num_tokens, num_l_heads]
    num_valid_tokens: jax.Array,  # i32
    *,
    chunk_size: int,
    max_num_seqs: int,
    sm_scale: float,
    vmem_limit_bytes: int,
) -> tuple[jax.Array, jax.Array]:
    """Runs every dense block of the step in one call.

    A 32-token block is eligible when its valid tokens all belong to one
    request and read at most DENSE_MAX_KV_PAGES pages. In each chunk of
    `chunk_size` tokens the valid blocks after the last ineligible one are
    dense, and the grid walks those only. Returns q with their rows
    attended, and per chunk the token where its gather-path range ends.
    """
    bq_sz = DENSE_BLOCK_TOKENS
    kv_group = DENSE_KV_PAGES_PER_STEP
    num_tokens, num_q_heads, head_dim = q.shape
    _, page_size, lanes = cache_kv_nope.shape
    _, rope_rows, _ = cache_kv_rope.shape
    topk = topk_indices.shape[1]
    num_l_heads = swa_l.shape[1]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    max_kv_pages = min(DENSE_MAX_KV_PAGES, pages_per_seq)
    cache_pages = max_kv_pages
    mask_rows = align_to(cache_pages * page_size // _MASK_LANES, 8)
    num_chunks = num_tokens // chunk_size
    chunk_blocks = chunk_size // bq_sz
    num_blocks = num_tokens // bq_sz

    num_valid_tokens = jnp.clip(num_valid_tokens, 0, num_tokens)
    token_valid = jnp.arange(num_tokens) < num_valid_tokens
    token_bound = jnp.where(token_valid, jnp.max(topk_indices, axis=-1) + 1, 0)
    kv_bounds = jnp.max(token_bound.reshape(num_blocks, bq_sz), axis=1)
    # Top-k positions are distinct, so as many of them as the bound means the
    # token selected every position below it.
    token_prefix = jnp.sum(topk_indices >= 0, axis=-1) == token_bound
    block_prefix = jnp.all((token_prefix | ~token_valid).reshape(num_blocks, bq_sz), 1)
    block_pages = jnp.maximum(1, cdiv(kv_bounds, page_size))

    block_start = jnp.arange(num_blocks) * bq_sz
    block_valid = block_start < num_valid_tokens
    block_last = jnp.clip(num_valid_tokens - 1, block_start, block_start + bq_sz - 1)
    one_seq = seq_ids_segment[block_start] == seq_ids_segment[block_last]
    eligible = one_seq & (block_pages <= max_kv_pages)
    # The gather path walks a prefix of each chunk, up to the end of the
    # chunk's last valid block that is not eligible; the blocks after it
    # are dense.
    gather_blocks = jnp.max(
        jnp.where(
            (block_valid & ~eligible).reshape(num_chunks, chunk_blocks),
            jnp.arange(1, chunk_blocks + 1),
            0,
        ),
        axis=1,
    )
    block_in_chunk = jnp.arange(num_blocks) % chunk_blocks
    is_dense = block_valid & (block_in_chunk >= jnp.repeat(gather_blocks, chunk_blocks))
    # The dense blocks' ids, in order, at the front.
    slot = jnp.where(is_dense, jnp.cumsum(is_dense) - 1, num_blocks)
    block_ids = (
        jnp.zeros(num_blocks, jnp.int32)
        .at[slot]
        .set(jnp.arange(num_blocks, dtype=jnp.int32), mode="drop")
    )
    num_dense = jnp.sum(is_dense)
    block_seqs = seq_ids_segment[::bq_sz].astype(jnp.int32)
    seq_bounds = jax.ops.segment_max(
        jnp.where(is_dense, kv_bounds, 0), block_seqs, num_segments=max_num_seqs
    )

    def q_map(i, page_indices_ref, num_valid_ref, block_ids_ref, *_):
        return (block_ids_ref[i], 0, 0)

    def row_map(i, page_indices_ref, num_valid_ref, block_ids_ref, *_):
        return (block_ids_ref[i], 0)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
        pl.BlockSpec((bq_sz, num_q_heads, head_dim), q_map),  # q
        pl.BlockSpec((bq_sz, topk), row_map),  # topk_indices
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
        pl.BlockSpec((bq_sz, num_q_heads, head_dim), q_map),  # swa_accumution
        pl.BlockSpec((bq_sz, num_l_heads), row_map),  # swa_l
        pl.BlockSpec((bq_sz, num_l_heads), row_map),  # swa_m
    ]
    num_rows = bq_sz * num_q_heads
    scratch_shapes = [
        pltpu.VMEM((mask_rows, bq_sz, _MASK_LANES), jnp.float32),
        pltpu.VMEM((page_size, lanes), jnp.float32),
        pltpu.VMEM((num_rows, 128), jnp.float32),
        pltpu.VMEM((num_rows, 128), jnp.float32),
        pltpu.VMEM((num_rows, head_dim), jnp.float32),
        pltpu.VMEM((cache_pages * page_size, head_dim), jnp.bfloat16),
        pltpu.VMEM((cache_pages, page_size, lanes), jnp.int32),
        pltpu.VMEM((cache_pages, rope_rows, lanes), jnp.int32),
        pltpu.SemaphoreType.DMA((2, cache_pages)),
        pltpu.SMEM((4,), jnp.int32),
    ]
    scalar_prefetches = (
        page_indices,
        num_valid_tokens.reshape(1).astype(jnp.int32),
        block_ids,
        block_seqs,
        block_pages.astype(jnp.int32),
        kv_bounds.astype(jnp.int32),
        block_prefix.astype(jnp.int32),
        seq_bounds[block_seqs].astype(jnp.int32),
    )
    scope_name = f"SparseMLA-dense-bq_{bq_sz}-kvp_{kv_group}"
    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _dense_attention_kernel,
                sm_scale=sm_scale,
                pages_per_seq=pages_per_seq,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=pl.BlockSpec((bq_sz, num_q_heads, head_dim), q_map),
                grid=(num_dense,),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                # Sequential on one core: blocks reuse the decoded-page cache.
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
            input_output_aliases={len(scalar_prefetches) + 1: 0},  # q
            name=scope_name,
        )
    )
    q = kernel(
        *scalar_prefetches,
        attention_sinks,
        q,
        topk_indices,
        cache_kv_nope,
        cache_kv_rope,
        swa_accumution,
        swa_l,
        swa_m,
    )
    gather_end = jnp.arange(num_chunks) * chunk_size + gather_blocks * bq_sz
    return q, gather_end


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
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


def prepare_swa_inputs(
    swa_accumution: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # [max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # [max_num_tokens, num_q_heads]
):
    _, actual_num_q_heads, actual_head_dim = swa_accumution.shape
    swa_packing = get_dtype_packing(swa_accumution.dtype)
    num_q_heads = align_to(actual_num_q_heads, swa_packing)
    head_dim = align_to(actual_head_dim, 128)
    swa_accumution = jnp.pad(
        swa_accumution,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    num_l_heads = align_to(num_q_heads, 128)
    swa_l = jnp.pad(
        swa_l,
        (
            (0, 0),
            (0, num_l_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    swa_m = jnp.pad(
        swa_m,
        (
            (0, 0),
            (0, num_l_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    return swa_accumution, swa_l, swa_m


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    return out[:, :actual_num_q_heads, :actual_head_dim]


# Main Attention kernel for DeepSeek V4 CSA (gather and attention)
# Note that the compressed kv tokens of current batch (current forward pass)
# have been written to the `cache_kv` by the compressor module before calling
# this function, `kv_lens` reflects the length after compressed kv cache write.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "vmem_limit_bytes",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # int32[total_num_pages, page_size // 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # float32[max_num_tokens, num_q_heads]
    *,
    sm_scale: float = 1.0,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int = 128,
    attention_kernel_batch_size: int = 16,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """MLA Ragged paged attention that supports mixed prefill and decode.

    Args:
      q: concatenated all sequences' queries.
      cache_kv_nope: the current kv cache for nope, one row per token (see
        `csa_cache_layout`).
      cache_kv_rope: the current kv cache for rope, 32 words per token.
      topk_indices: for each query token, the indices of the top k key tokens to
        attend to.
      page_indices: flattened page indices look-up table by (seq_id, page_id).
      cu_q_lens: the cumulative sum of the effective query lengths. Similar to
        kv_lens, only the first num_seqs+1 values are valid.
      distribution: (i, j, k) represents that sequences[0:i] are decode-only,
        sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
        k is also the total number of sequences.
      sm_scale: the softmax scale which will be applied to the Q@K^T.
      vmem_limit_bytes: the vmem limit for the pallas kernel.

    Returns:
      The output of attention.
    """
    # The cache is DSV4 FP8 format.
    # nope_cache contains 448 fp8 + 7 fp8 scales,
    # rope_cache contains 64 bf16
    assert cache_kv_nope.dtype == jnp.int32
    assert cache_kv_rope.dtype == jnp.int32

    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    head_dim = q.shape[-1]
    attention_sinks = jnp.pad(
        attention_sinks,
        (0, q.shape[1] - actual_num_q_heads),
        constant_values=jnp.finfo(attention_sinks.dtype).min,
    )
    assert swa_accumution.dtype == q.dtype
    swa_accumution, swa_l, swa_m = prepare_swa_inputs(swa_accumution, swa_l, swa_m)

    _, page_size, _ = cache_kv_nope.shape

    _, num_q_heads, _ = q.shape
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0

    def run_mla_kernel(
        q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
        cache_kv_nope: jax.Array,  # int32[chunk_size, topk, 128]
        cache_kv_rope: jax.Array,  # int32[chunk_size, topk // 4, 128]
        kv_lens: jax.Array,  # i32[max_num_seqs]
        attention_sinks: jax.Array,  # float32[num_q_heads]
        swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
        swa_l: jax.Array,  # float32[max_num_tokens, num_l_heads]
        swa_m: jax.Array,  # float32[max_num_tokens, num_l_heads]
        start_seq_idx: int,
        end_seq_idx: jax.Array,  # i32
        num_valid_tokens: jax.Array,  # i32
        kernel_batch_size: int,
    ):
        batch_size = kernel_batch_size
        end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
        grid = (cdiv(end_seq_idx - start_seq_idx, batch_size),)
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.VMEM),  # attention_sinks
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_accumution
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_l
            pl.BlockSpec(memory_space=pltpu.HBM),  # swa_m
        ]

        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o

        # One batch entry's worth of gathered top-k rows, per cache, in the
        # uint8 / bf16 views the kernel bitcasts the int32 words to.
        _, topk_rows, lanes = cache_kv_nope.shape
        bkv_nope_double_buf = pltpu.VMEM(
            (2, batch_size, topk_rows * 4, lanes),
            jnp.uint8,
        )
        _, rope_rows, lanes = cache_kv_rope.shape
        bkv_rope_double_buf = pltpu.VMEM(
            (2, batch_size, rope_rows * 2, lanes),
            jnp.bfloat16,
        )

        bq_double_bufq = pltpu.VMEM(
            (2, batch_size, num_q_heads, head_dim),
            q.dtype,
        )

        bo_double_buf = bq_double_bufq

        num_l_heads = align_to(num_q_heads, 128)
        bl_double_buf = pltpu.VMEM(
            (2, batch_size, num_l_heads),
            jnp.float32,
        )
        bm_double_buf = bl_double_buf

        swa_acc_double_buf = pltpu.VMEM(
            (2, batch_size, num_q_heads, head_dim),
            q.dtype,
        )

        scratch_shapes = [
            bkv_nope_double_buf,
            bkv_rope_double_buf,
            bq_double_bufq,
            bo_double_buf,  # Double buffering for output block.
            bl_double_buf,  # Double buffering for l output.
            bm_double_buf,  # Double buffering for m output.
            swa_acc_double_buf,  # Buffer for swa_accumution.
            # Semaphores for double buffering of
            # bkv_nope, bq, bo, swa_acc, swa_l, swa_m, bkv_rope
            pltpu.SemaphoreType.DMA((7, 2)),
        ]

        scalar_prefetches = (
            kv_lens,
            jnp.array([start_seq_idx, end_seq_idx, num_valid_tokens], jnp.int32),
            # (bi_sem_idx, bo_sem_idx)
            jnp.zeros((2,), jnp.int32),
            # seq_idx_start of the in-flight bo DMA per bo sem; -1 = none.
            jnp.full((2,), -1, jnp.int32),
        )

        gathered_topk = cache_kv_nope.shape[1]
        scope_name = (
            f"SparseMLA-p_{gathered_topk}-bz_{batch_size}-gcz_{cache_kv_nope.shape[0]}"
        )
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _attention_kernel,
                    sm_scale=sm_scale,
                    batch_size=batch_size,
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
                out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                input_output_aliases={
                    5: 0,  # Alias output activation with q
                },
                name=scope_name,
            )
        )
        return kernel(
            *scalar_prefetches,
            attention_sinks,
            q,
            cache_kv_nope,
            cache_kv_rope,
            swa_accumution,
            swa_l,
            swa_m,
        )

    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(
        jnp.arange(max_num_seqs), tokens_per_seq, total_repeat_length=q.shape[0]
    )
    assert topk_indices is not None
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)

    topk = topk_indices.shape[-1]
    num_chunks = cdiv(q.shape[0], gather_and_attention_chunk_size)
    dense_path = (
        gather_and_attention_chunk_size >= DENSE_MIN_CHUNK_TOKENS
        and gather_and_attention_chunk_size % DENSE_BLOCK_TOKENS == 0
        # The gather range ends on a block boundary and its grid walks
        # whole kernel batches.
        and DENSE_BLOCK_TOKENS
        % _largest_divisor(gather_and_attention_chunk_size, attention_kernel_batch_size)
        == 0
        and q.shape[0] % gather_and_attention_chunk_size == 0
        and page_size % _MASK_LANES == 0
    )
    if dense_path:
        q, gather_ends = _dense_attention(
            q,
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            page_indices,
            seq_ids_segment,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            cu_q_lens[distribution[2]],
            chunk_size=gather_and_attention_chunk_size,
            max_num_seqs=max_num_seqs,
            sm_scale=sm_scale,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    # Each chunk's gather range [start_pos, batch_end).
    chunks = []
    for i in range(num_chunks):
        start_pos = i * gather_and_attention_chunk_size
        end_pos = min(start_pos + gather_and_attention_chunk_size, q.shape[0])
        chunk_size = end_pos - start_pos

        # Batching
        kernel_batch_size = _largest_divisor(chunk_size, attention_kernel_batch_size)
        assert chunk_size % kernel_batch_size == 0
        # The kernel grid walks [start_pos, batch_end) in `kernel_batch_size`
        # steps, so `batch_end - start_pos` MUST be a multiple of
        # `kernel_batch_size`.
        batch_end = start_pos + (
            cdiv(
                jnp.maximum(
                    0,
                    jnp.minimum(
                        cu_q_lens[distribution[2]],
                        end_pos,
                    )
                    - start_pos,
                ),
                kernel_batch_size,
            )
            * kernel_batch_size
        )
        if dense_path:
            # Rows past the gather range were attended by the dense path.
            batch_end = jnp.minimum(batch_end, gather_ends[i])
        chunks.append((start_pos, chunk_size, kernel_batch_size, batch_end))

    # Only the gather ranges' indices are computed; the SparseCore gather
    # skips every other token.
    gather_indices = gather_row_indices(
        topk_indices,
        page_indices,
        seq_ids_segment,
        jnp.stack([batch_end for *_, batch_end in chunks]),
        max_num_seqs=max_num_seqs,
        page_size=page_size,
        total_num_pages=cache_kv_nope.shape[0],
        range_size=gather_and_attention_chunk_size,
    ).reshape(-1)

    for start_pos, chunk_size, kernel_batch_size, batch_end in chunks:
        num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk

        # For prefilling of short sequences (or early in the sequence), there are
        # very few number of KVs in the sequence, so different qs' selected topk
        # would have large overlap. This causes gather read hotspotting. We've seen
        # 30%+ performance degradation compared to the no-duplicate-indices case.
        #
        # Blocks of a single request with few KV pages take the dense path
        # above (`_dense_attention`) and skip this gather.
        gathered_nope_buffer, gathered_rope_buffer = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            gather_indices,
            num_valid_indices=num_valid_indices,
            top_k=topk,
            offset=start_pos * topk,
            size=chunk_size * topk,
        )
        gathered_nope_buffer = gathered_nope_buffer.reshape(chunk_size, topk, -1)
        gathered_rope_buffer = gathered_rope_buffer.reshape(chunk_size, topk // 4, -1)

        q = run_mla_kernel(
            q,
            gathered_nope_buffer,
            gathered_rope_buffer,
            kv_lens,
            attention_sinks,
            swa_accumution,
            swa_l,
            swa_m,
            start_seq_idx=start_pos,
            end_seq_idx=batch_end,
            num_valid_tokens=cu_q_lens[distribution[2]],
            kernel_batch_size=kernel_batch_size,
        )
    return prepare_outputs(
        q, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
