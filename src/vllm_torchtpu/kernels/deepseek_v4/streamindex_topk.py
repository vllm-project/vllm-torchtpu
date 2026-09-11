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
"""TPU-Friendly StreamIndex Top-K kernel."""

import enum
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk

Enum = enum.Enum
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
    """Represents the different cases for MLA."""

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


def kernel(
    # Prefetch
    seq_lens_ref,  # Shape: [max_num_seqs], Memory: SMEM
    page_indices_ref,  # Shape: [max_num_seqs * pages_per_seq], Memory: SMEM
    cu_q_lens_ref,  # Shape: [max_num_seqs + 1], Memory: SMEM
    start_end_seq_idx_ref,  # Shape: [2], Memory: SMEM
    sem_ids_ref,  # Shape: [3], Memory: SMEM
    bo_sz_ref,  # Shape: [2], Memory: SMEM
    # Input
    q_hbm_ref,  # Shape: [max_num_tokens, num_q_heads, head_dim], Memory: HBM
    indexer_weights_hbm_ref,  # Shape: [max_num_tokens, num_q_heads], Memory: HBM
    cache_kv_hbm_ref,  # Shape: [total_num_pages, page_size_per_kv_packing, kv_packing, lkv_dim], Memory: HBM
    scores_in_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM (aliased to output)
    # Output
    scores_hbm_ref,  # Shape: [max_num_tokens, num_sublanes_total, 128], Memory: HBM
    # Scratch
    bkv_x2_ref,  # Shape: [2, seq_batch_size, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim], Memory: VMEM
    bq_x2_ref,  # Shape: [2, seq_batch_size, bq_sz, num_q_heads, head_dim], Memory: VMEM
    bq_weights_x2_ref,  # Shape: [2, seq_batch_size, bq_sz, num_q_heads], Memory: VMEM
    scores_block_x2_ref,  # Shape: [2, seq_batch_size * bq_sz, num_sublanes_bkv, 128], Memory: VMEM
    sems,  # Shape: [4, 2, seq_batch_size], Memory: Semaphore
    *,
    compression_ratio: int,
    static_q_len: int,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
):
    """Core kernel logic that operates on memory references."""

    _, num_q_heads, head_dim = q_hbm_ref.shape
    lkv_dim = cache_kv_hbm_ref.shape[-1]

    total_num_pages, page_size_per_kv_packing, kv_packing, _ = (
        cache_kv_hbm_ref.shape)

    max_num_seqs = seq_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    pages_per_seq = num_page_indices // max_num_seqs

    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing
    num_sublanes_bkv = bkv_sz // 128

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * seq_batch_size
    batch_end_seq_idx = batch_start_seq_idx + seq_batch_size - 1

    q_lens = []
    kv_lens = []
    seq_lens = []
    for batch_idx in range(seq_batch_size):
        q_start = cu_q_lens_ref[batch_start_seq_idx + batch_idx]
        q_end = cu_q_lens_ref[batch_start_seq_idx + batch_idx + 1]
        q_len = q_end - q_start
        q_lens.append(q_len)
        seq_len = seq_lens_ref[batch_start_seq_idx + batch_idx]
        seq_lens.append(seq_len)
        kv_len = seq_len // compression_ratio
        kv_lens.append(kv_len)

    def wait_send_scores(bo_sem_idx):
        old_sz = bo_sz_ref[bo_sem_idx]

        @pl.when(old_sz >= 0)
        def _():
            dst = scores_block_x2_ref.at[bo_sem_idx,
                                         pl.ds(0, seq_batch_size * old_sz)]
            _async_copy(dst, dst, sems.at[2, bo_sem_idx, 0], wait=True)

    def start_send_scores(bo_sem_idx, sz, token_start, bkv_idx):
        # All sequences in the batch have the same sz. Issue a single DMA for the
        # entire batch.
        bo_sz_ref[bo_sem_idx] = sz
        sublane_start = bkv_idx * num_sublanes_bkv
        _async_copy(
            scores_block_x2_ref.at[bo_sem_idx,
                                   pl.ds(0, seq_batch_size * sz)],
            scores_hbm_ref.at[
                pl.ds(token_start, seq_batch_size * sz),
                pl.ds(sublane_start, num_sublanes_bkv),
            ],
            sems.at[2, bo_sem_idx, 0],
            wait=False,
        )

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
            kv_packing,
            lkv_dim,
        )
        max_hbm_pages = reshaped_cache_hbm_ref.shape[0]

        for batch_idx in range(seq_batch_size):
            sem = sems.at[0, bkv_sem_idx, batch_idx]
            bkv_vmem_ref = bkv_x2_ref.at[bkv_sem_idx, batch_idx]

            kv_p_start = bkv_idx * bkv_p
            page_indices_offset = (seq_idx +
                                   batch_idx) * pages_per_seq + kv_p_start

            if not wait:
                for i in range(bkv_p):
                    sz_per_kv_packing = page_size_per_kv_packing
                    page_idx = jnp.minimum(page_indices_offset + i,
                                           num_page_indices - 1)
                    safe_page_offset = jnp.minimum(
                        page_indices_ref[page_idx] * page_size_per_kv_packing,
                        jnp.maximum(0,
                                    max_hbm_pages - page_size_per_kv_packing),
                    )

                    _async_copy(
                        reshaped_cache_hbm_ref.at[pl.ds(
                            safe_page_offset, sz_per_kv_packing)],
                        bkv_vmem_ref.at[pl.ds(i * page_size_per_kv_packing,
                                              sz_per_kv_packing)],
                        sem,
                        wait=False,
                    )
            else:
                dma_bkv_sz = bkv_p * page_size_per_kv_packing
                dst_kv = bkv_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
                _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        for batch_idx in range(seq_batch_size):
            sem = sems.at[1, bq_sem_idx, batch_idx]
            weights_sem = sems.at[3, bq_sem_idx, batch_idx]
            bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, batch_idx]
            bq_weights_vmem_ref = bq_weights_x2_ref.at[bq_sem_idx, batch_idx]

            q_len_start = cu_q_lens_ref[seq_idx + batch_idx] + bq_idx * bq_sz
            curr_q_end = cu_q_lens_ref[seq_idx + batch_idx + 1]
            sz = jnp.maximum(0, jnp.minimum(bq_sz, curr_q_end - q_len_start))

            if not wait:
                _async_copy(
                    q_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_vmem_ref.at[pl.ds(0, sz)],
                    sem,
                    wait=False,
                )
                _async_copy(
                    indexer_weights_hbm_ref.at[pl.ds(q_len_start, sz)],
                    bq_weights_vmem_ref.at[pl.ds(0, sz)],
                    weights_sem,
                    wait=False,
                )
            else:
                dst = bq_vmem_ref.at[pl.ds(0, sz)]
                _async_copy(dst, dst, sem, wait=True)
                dst_w = bq_weights_vmem_ref.at[pl.ds(0, sz)]
                _async_copy(dst_w, dst_w, weights_sem, wait=True)

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def load_bq(bq_sem_idx):
        data = bq_x2_ref.at[bq_sem_idx, :, :bq_sz][...].reshape(
            seq_batch_size, bq_sz * num_q_heads, head_dim)
        bqs = []
        for batch_idx in range(seq_batch_size):
            bqs.append(data[batch_idx])
        return bqs

    def load_bq_weights(bq_sem_idx):
        data = bq_weights_x2_ref.at[bq_sem_idx, :, :bq_sz][...]
        bq_weights = []
        for batch_idx in range(seq_batch_size):
            bq_weights.append(data[batch_idx])
        return bq_weights

    def load_bkv(bkv_sem_idx):
        bkvs = []
        bkv_scales = []
        for batch_idx in range(seq_batch_size):
            bkv = bkv_x2_ref.at[bkv_sem_idx,
                                batch_idx, :bkv_sz_per_kv_packing][...]

            flat_bkv = bkv.reshape(-1, bkv.shape[-1])
            fp8_val = flat_bkv[:, :head_dim]
            fp8_val = pltpu.bitcast(fp8_val, jnp.float8_e4m3fn)
            scale_val = pltpu.bitcast(flat_bkv[:, head_dim:head_dim + 1].T,
                                      jnp.float8_e8m0fnu).astype(jnp.bfloat16)

            bkvs.append(fp8_val.reshape(bkv_sz, head_dim))
            bkv_scales.append(scale_val)
        return bkvs, bkv_scales

    def process():
        kv_len_max = jnp.max(jnp.stack(kv_lens))
        num_bkv = jnp.maximum(1, cdiv(kv_len_max, bkv_sz))
        if static_q_len is None:
            assert seq_batch_size == 1
            num_bq = jnp.maximum(1, cdiv(q_lens[0], bq_sz))
        else:
            num_bq = jnp.maximum(1, cdiv(static_q_len, bq_sz))

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + seq_batch_size,
                                      seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + seq_batch_size,
                                      seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_scores(
            bq_vec,
            bkv_vec,
            scale_val_vec,
            bq_weights_vec,
            bq_pos_compressed_vec,
            bkv_idx,
        ):
            # Vectorized batched matmul
            bq_stacked = jnp.stack(
                bq_vec
            )  # Shape: (seq_batch_size, bq_sz * num_q_heads, head_dim)
            bkv_stacked = jnp.stack(
                bkv_vec)  # Shape: (seq_batch_size, bkv_sz, head_dim)
            scale_val_stacked = jnp.stack(
                scale_val_vec)  # Shape: (seq_batch_size, 1, bkv_sz)
            bq_weights_stacked = jnp.stack(
                bq_weights_vec)  # Shape: (seq_batch_size, bq_sz, num_q_heads)
            bq_pos_compressed_stacked = jnp.stack(
                bq_pos_compressed_vec)  # Shape: (seq_batch_size, bq_sz)

            # Compute in Registers
            s = jnp.einsum(
                "bnd,bmd->bnm",
                bq_stacked,
                bkv_stacked,
                preferred_element_type=jnp.float32,
            )  # Shape: (seq_batch_size, bq_sz * num_q_heads, bkv_sz)
            s = s.reshape(seq_batch_size, bq_sz, num_q_heads, bkv_sz)
            s = jnp.maximum(s, 0.0)
            s = s * bq_weights_stacked.astype(jnp.float32)[:, :, :, None]
            s_summed = s.sum(axis=2)  # Shape: (seq_batch_size, bq_sz, bkv_sz)

            s_summed = (s_summed * scale_val_stacked
                        )  # Shape: (seq_batch_size, bq_sz, bkv_sz)

            k_span = bkv_idx * bkv_sz + jnp.arange(
                bkv_sz, dtype=jnp.int32)  # Shape: (bkv_sz,)

            kv_lens_stacked = jnp.stack(kv_lens)  # Shape: (seq_batch_size,)
            valid_mask = (k_span[None, None, :] < kv_lens_stacked[:, None,
                                                                  None]
                          )  # Shape: (seq_batch_size, 1, bkv_sz)
            causal_mask = (k_span[None, None, :]
                           <= bq_pos_compressed_stacked[:, :, None]
                           )  # Shape: (seq_batch_size, bq_sz, bkv_sz)

            mask = jnp.logical_and(valid_mask, causal_mask)
            s_summed = jnp.where(mask, s_summed, -jnp.inf)

            return s_summed  # Shape: (seq_batch_size, bq_sz, bkv_sz)

        def compute_with_bq(bq_idx, _):

            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(
                batch_start_seq_idx, bq_idx, bq_sem_idx)

            # Prefetch next bq
            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            bq_pos_compressed_vec = []
            for batch_idx in range(seq_batch_size):
                q_pos = (seq_lens[batch_idx] - q_lens[batch_idx] +
                         bq_idx * bq_sz + jnp.arange(bq_sz, dtype=jnp.int32))
                bq_pos_compressed_vec.append(q_pos // compression_ratio)

            # Wait for cur bq if not ready yet
            wait_fetch_bq(batch_start_seq_idx, bq_idx, bq_sem_idx)
            bq_vec = load_bq(bq_sem_idx)
            bq_weights_vec = load_bq_weights(bq_sem_idx)

            # If seq_batch_size > 1, static_q_len is always 1, therefore sz is always
            # 1 for all sequences within the batch.
            token_start = cu_q_lens_ref[batch_start_seq_idx] + bq_idx * bq_sz
            curr_q_end = cu_q_lens_ref[batch_start_seq_idx + 1]
            sz = jnp.maximum(0, jnp.minimum(bq_sz, curr_q_end - token_start))

            def compute_with_bkv(bkv_idx, _):
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    batch_start_seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                # Prefetch next bkv
                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx,
                                    next_bkv_sem_idx)

                # Wait for cur bkv
                wait_fetch_bkv(batch_start_seq_idx, bkv_idx, bkv_sem_idx)
                bkv_vec, scale_val_vec = load_bkv(bkv_sem_idx)

                s_summed = compute_scores(
                    bq_vec,
                    bkv_vec,
                    scale_val_vec,
                    bq_weights_vec,
                    bq_pos_compressed_vec,
                    bkv_idx,
                )  # Shape: (seq_batch_size, bq_sz, bkv_sz)

                bo_sem_idx = sem_ids_ref[2]
                wait_send_scores(bo_sem_idx)
                scores_block_x2_ref[bo_sem_idx, ...] = s_summed.astype(
                    scores_block_x2_ref.dtype).reshape(seq_batch_size * bq_sz,
                                                       num_sublanes_bkv, 128)
                start_send_scores(bo_sem_idx, sz, token_start, bkv_idx)
                sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq(start_seq_idx, 0, 0)
        start_fetch_bkv(start_seq_idx, 0, 0)

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        for i in range(2):
            wait_send_scores(i)

    ### ------- Kernel end ------- ###


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


def prepare_index_weights(
    index_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads],
    q_dtype,
):
    _, actual_num_q_heads = index_weights.shape
    index_weights = index_weights.astype(jnp.float32)
    num_q_heads = align_to(actual_num_q_heads, get_dtype_packing(q_dtype))
    index_weights = jnp.pad(
        index_weights,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
        ),
        constant_values=0,
    )
    return index_weights


def prepare_outputs(out):
    if out.ndim == 3:
        out = out.reshape(out.shape[0], -1)
    return out


def _effective_row_lengths(
    seq_lens: jax.Array,  # i32[max_num_seqs]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    num_tokens: int,
    num_positions: int,
    compression_ratio: int,
) -> jax.Array:  # i32[num_tokens]
    """Visible compressed KV positions per token row of the score matrix."""
    max_num_seqs = seq_lens.shape[0]
    num_seqs = distribution[2]
    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_mask = token_ids[:, None] >= cu_q_lens[None, 1:max_num_seqs + 1]
    seq_mask = jnp.where(
        jnp.arange(max_num_seqs)[None, :] < num_seqs, seq_mask, False)
    seq_ids = jnp.sum(seq_mask, axis=1)

    q_start = cu_q_lens[seq_ids]
    q_len = cu_q_lens[seq_ids + 1] - q_start
    seq_len = seq_lens[seq_ids]
    token_pos = seq_len - q_len + (token_ids - q_start)
    kv_len = seq_len // compression_ratio
    visible = jnp.minimum(kv_len, token_pos // compression_ratio + 1)
    valid = token_ids < cu_q_lens[num_seqs]
    return jnp.where(valid, jnp.minimum(visible, num_positions), 0)


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "decode_req_batch_size",
        "enable_early_exit",
        "emit_pairs",
        "payload_stage2",
    ),
)
def streamindex_topk(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    indexer_weights: jax.Array,  # [max_num_tokens, actual_num_q_heads]
    cache_kv: jax.
    Array,  # [total_num_pages, page_size_per_kv_packing, kv_packing, head_dim]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    decode_req_batch_size: int = 4,
    enable_early_exit: bool = False,
    emit_pairs: bool = False,
    payload_stage2: bool = False,
) -> jax.Array:
    """StreamIndex Top-K retrieval.

  Args:
    q: concatenated all sequences' queries.
    indexer_weights: concatenated all sequences' indexer weights.
    cache_kv: the current kv cache.
    seq_lens: the length of each sequence in the kv cache (uncompressed).
    page_indices: flattened page indices look-up table by (seq_id, page_id).
    cu_q_lens: the cumulative sum of the effective query lengths. Similar to
      kv_lens, only the first num_seqs+1 values are valid.
    distribution: (i, j, k) represents that sequences[0:i] are decode-only,
      sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
      k is also the total number of sequences.
    k: Number of top-K elements to retrieve.
    compression_ratio: KV cache compression ratio.
    num_kv_pages_per_block: number of kv pages to be processed in one block in
      the pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
    num_queries_per_block: number of queries to be processed in one block in the
      pallas kernel. This is a tuple of (decode, prefill, mixed) cases.
    vmem_limit_bytes: the vmem limit for the pallas kernel.
    enable_early_exit: whether to enable early exit using jax.lax.cond when k >=
      kv_len for all sequences in the batch. Defaults to False.
    decode_req_batch_size: maximum decode batch size per iteration.
    emit_pairs: forwarded to `sparsecore_topk`. Only affects the two-stage
      path, which needs `_pick_partition(...) > 1`; a no-op otherwise.
      Defaults to False.
    payload_stage2: forwarded to `sparsecore_topk`. Same reachability caveat
      as `emit_pairs`. Defaults to False.

  Returns:
    Top-K indices (in compressed space).
  """
    # Scale factors for the FP8 index cache format are packed directly inside
    # `cache_kv` along the width dimension, keeping HBM transactions fused.

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

    max_num_seqs = seq_lens.shape[0]

    original_dtype = q.dtype

    prepared_indexer_weights = prepare_index_weights(indexer_weights,
                                                     original_dtype)
    q = prepare_q_inputs(q)
    lkv_dim = cache_kv.shape[-1]
    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    for bkv_p in num_kv_pages_per_blocks:
        bkv_sz = page_size * bkv_p
        if bkv_sz % 128 != 0:
            raise ValueError(
                f"bkv_sz ({page_size} * {bkv_p} = {bkv_sz}) must be a multiple"
                " of 128.")

    num_sublanes_total = max(
        align_to(pages_per_seq, bkv_p) * page_size // 128
        for bkv_p in num_kv_pages_per_blocks)

    def run_scores_kernel(
        q,
        prepared_indexer_weights,
        cache_kv,
        scores_init,
        seq_lens,
        page_indices,
        cu_q_lens,
        start_seq_idx,
        end_seq_idx,
        static_q_len,
        num_kv_pages_per_block,
        num_queries_per_block,
        seq_batch_size,
        out_dtype,
        case=MlaCase.MIXED,
    ):
        _, num_q_heads, head_dim = q.shape
        # Only support batching for decode sequences.
        # TODO: support batching for decode sequences with speculative decoding
        # enabled, e.g. static_q_len = gamma + 1.
        if seq_batch_size > 1:
            assert static_q_len == 1

        bkv_p = num_kv_pages_per_block
        if static_q_len is not None:
            bq_sz = min(num_queries_per_block, static_q_len)
        else:
            bq_sz = num_queries_per_block
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing
        bkv_sz = bkv_p * page_size
        num_sublanes_bkv = bkv_sz // 128

        # If seq_batch_size > 1, caller already guaranteed that
        # end_seq_idx - start_seq_idx % seq_batch_size == 0.
        grid = ((end_seq_idx - start_seq_idx) // seq_batch_size, )

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # prepared_indexer_weights
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv
            pl.BlockSpec(
                memory_space=pltpu.HBM),  # scores_init (aliased to out)
        ]
        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # scores

        bkv_double_buf = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bkv_buf_sz_per_kv_packing,
                kv_packing,
                lkv_dim,
            ),
            cache_kv.dtype,
        )
        bq_double_bufq = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bq_sz,
                num_q_heads,
                head_dim,
            ),
            q.dtype,
        )
        bq_weights_double_buf = pltpu.VMEM(
            (
                2,
                seq_batch_size,
                bq_sz,
                num_q_heads,
            ),
            prepared_indexer_weights.dtype,
        )
        bo_scores_double_buf = pltpu.VMEM(
            (2, seq_batch_size * bq_sz, num_sublanes_bkv, 128), out_dtype)

        scratch_shapes = [
            bkv_double_buf,
            bq_double_bufq,
            bq_weights_double_buf,
            bo_scores_double_buf,
            pltpu.SemaphoreType.DMA((4, 2, seq_batch_size)),
        ]

        scalar_prefetches = (
            seq_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            jnp.zeros((3, ), jnp.int32),  # (bq, bkv, bo) sem indices
            jnp.full((2, ), -1, jnp.int32),  # in-flight out DMA row counts
        )

        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}"
        pallas_kernel = jax.named_scope(scope_name)(pl.pallas_call(
            functools.partial(
                kernel,
                compression_ratio=compression_ratio,
                static_q_len=static_q_len,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
                seq_batch_size=seq_batch_size,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary", ),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(
                shape=(q.shape[0], num_sublanes_total, 128),
                dtype=out_dtype,
            ),
            input_output_aliases={len(scalar_prefetches) + 3: 0},
            name=scope_name,
        ))
        return pallas_kernel(
            *scalar_prefetches,
            q,
            prepared_indexer_weights,
            cache_kv,
            scores_init,
        )

    def _common_path(_):
        out_dtype = jnp.float32

        # Pre-fill the output with -inf and alias it, so masked / unwritten
        # columns are already -inf.
        scores_init = jnp.full(
            (q.shape[0], num_sublanes_total, 128),
            -jnp.inf,
            dtype=out_dtype,
        )

        # TODO: we shall sort the sequences by length, so that multiple decode
        # sequences in one batch have similar lengths to reduce waste of compute.
        # With the same batch size, the longest sequence will determine number of
        # blocks to run computation for.
        decode_batch_end = (distribution[0] // decode_req_batch_size *
                            decode_req_batch_size)
        scores = run_scores_kernel(
            q,
            prepared_indexer_weights,
            cache_kv,
            scores_init,
            seq_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            start_seq_idx=jnp.array(0),
            end_seq_idx=decode_batch_end,
            static_q_len=1,
            seq_batch_size=decode_req_batch_size,
            out_dtype=out_dtype,
            case=MlaCase.DECODE,
        )
        # Handle num_decode_seqs % decode_req_batch_size != 0 case.
        scores = run_scores_kernel(
            q,
            prepared_indexer_weights,
            cache_kv,
            scores,
            seq_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[0],
            num_queries_per_block=num_queries_per_blocks[0],
            start_seq_idx=decode_batch_end,
            end_seq_idx=distribution[1],
            static_q_len=1,
            seq_batch_size=1,
            out_dtype=out_dtype,
            case=MlaCase.DECODE,
        )

        scores = run_scores_kernel(
            q,
            prepared_indexer_weights,
            cache_kv,
            scores,
            seq_lens,
            page_indices,
            cu_q_lens,
            num_kv_pages_per_block=num_kv_pages_per_blocks[2],
            num_queries_per_block=num_queries_per_blocks[2],
            start_seq_idx=distribution[1],
            end_seq_idx=distribution[2],
            static_q_len=None,
            seq_batch_size=1,
            out_dtype=out_dtype,
            case=MlaCase.MIXED,
        )

        scores = scores.reshape(q.shape[0], -1)
        if scores.shape[1] < k:
            scores = jnp.pad(
                scores,
                ((0, 0), (0, k - scores.shape[1])),
                constant_values=-jnp.inf,
            )

        scores_to_reduce = scores
        if scores_to_reduce.dtype != jnp.float32:
            scores_to_reduce = scores_to_reduce.astype(jnp.float32)

        eff_row_lengths = _effective_row_lengths(
            seq_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            num_tokens=q.shape[0],
            num_positions=scores_to_reduce.shape[1],
            compression_ratio=compression_ratio,
        )
        topk_idxs = sparsecore_topk(
            scores_to_reduce,
            k,
            row_lengths=eff_row_lengths,
            write_empty_rows=True,
            emit_pairs=emit_pairs,
            payload_stage2=payload_stage2,
        )

        return topk_idxs[:q.shape[0], :k]

    def _fast_path(_):
        token_idx = jnp.arange(q.shape[0])
        seq_idx = jnp.minimum(
            jnp.searchsorted(cu_q_lens[1:], token_idx, side="right"),
            seq_lens.shape[0] - 1,
        )
        seq_len = seq_lens[seq_idx]
        q_len = cu_q_lens[seq_idx + 1] - cu_q_lens[seq_idx]
        q_start = cu_q_lens[seq_idx]
        q_abs_pos = (seq_len - q_len) + (token_idx - q_start)
        max_valid_idx = jnp.minimum(
            seq_len // compression_ratio - 1,
            q_abs_pos // compression_ratio,
        )
        max_valid_idx = jnp.where(token_idx < cu_q_lens[-1], max_valid_idx, -1)
        s_idx = jnp.arange(k, dtype=jnp.int32)[None, :]
        return jnp.where(s_idx <= max_valid_idx[:, None], s_idx, -1)

    if not enable_early_exit:
        return _common_path(None)
    return jax.lax.cond(
        jnp.max(seq_lens) // compression_ratio <= k,
        _fast_path,
        _common_path,
        None,
    )
