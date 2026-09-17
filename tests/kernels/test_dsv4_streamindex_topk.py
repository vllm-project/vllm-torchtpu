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
"""Tests for Deepseek V4 StreamIndex Top-K kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from google3.experimental.users.hwanginho.deepseek_v4.streamindex_topk.streamindex_topk import \
        streamindex_topk
except ImportError:
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import \
        streamindex_topk


# =====================================================================
# Helper functions from the compressor implementation
# =====================================================================
def _to_byte_lane(x: jax.Array) -> jax.Array:
    """Reinterpret each element of ``x``'s trailing dim as raw bytes."""
    b = jax.lax.bitcast_convert_type(x, jnp.uint8)
    if b.ndim > x.ndim:
        b = b.reshape(*x.shape[:-1], -1)
    return b


def quantize_fp8_ue8m0(x: jax.Array, block_size: int):
    """Block fp8 quantization with UE8M0 (power-of-two) block scales."""
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    *lead, dim = x.shape
    blocked = x.reshape(*lead, dim // block_size, block_size)
    amax = jnp.clip(jnp.max(jnp.abs(blocked), axis=-1, keepdims=True), 1e-4,
                    None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    q = (blocked * (1.0 / scale)).astype(jnp.float8_e4m3fn).reshape(x.shape)
    scale = jnp.squeeze(scale, -1).astype(jnp.float8_e8m0fnu)
    return q, scale


# =====================================================================

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def streamindex_topk_ref(
    q,
    weights,
    kv,
    block_table,
    T_list,
    S_list,
    cu_q_lens,
    k,
    comp_ratio,
    H_I,
    H_KV,
):
    """Naive NumPy reference implementation for StreamIndex Top-K."""
    num_tokens = q.shape[0]
    expected_topk = np.full((num_tokens, k), -1, dtype=np.int32)
    B = len(T_list)

    for b in range(B):
        T_seq = T_list[b]
        S_total = S_list[b]
        q_start = cu_q_lens[b]

        if T_seq == 0:
            continue

        S_valid = S_total // comp_ratio
        seq_blocks = block_table[b]
        seq_kv = np.concatenate([kv[p] for p in seq_blocks], axis=0)

        naive_scores = np.full((T_seq, max(S_valid, k)),
                               -np.inf,
                               dtype=np.float32)

        for t_idx in range(T_seq):
            global_t = q_start + t_idx
            q_abs_pos = (S_total - T_seq) + t_idx

            for s_idx in range(S_valid):
                if s_idx * comp_ratio > q_abs_pos:
                    continue

                score = 0.0
                for h in range(H_I):
                    h_kv = h // (H_I // H_KV)
                    inner = np.dot(q[global_t, h], seq_kv[s_idx, h_kv])
                    score += max(0.0, inner) * weights[global_t, h]

                naive_scores[t_idx, s_idx] = score

        seq_expected = np.argsort(-naive_scores, axis=-1)[:, :k]

        for t_idx in range(T_seq):
            for i in range(k):
                idx = seq_expected[t_idx, i]
                if naive_scores[t_idx, idx] == -np.inf:
                    expected_topk[q_start + t_idx, i] = -1
                else:
                    expected_topk[q_start + t_idx, i] = idx

    return expected_topk


@pytest.mark.parametrize(
    "B, num_tokens, page_size, max_blocks, H_I, H_KV, D, k, compression_ratio,"
    " bq_sz, bkv_p, seq_lens_list, cu_q_lens_list",
    [
        # Small standard case
        (2, 6, 16, 8, 4, 1, 64, 512, 2, 32, 8, [4, 6], [0, 3, 6]),
        # # Single token batch (Decode Phase)
        (1, 1, 16, 4, 2, 1, 32, 512, 1, 16, 8, [10], [0, 1]),
        # Large batch, multiple tokens per sequence (Prefill Phase)
        (
            4,
            20,
            32,
            16,
            8,
            1,
            128,
            512,
            4,
            64,
            4,
            [10, 20, 30, 40],
            [0, 5, 10, 15, 20],
        ),
        # Odd chunk sizes
        (2, 5, 8, 6, 4, 1, 16, 512, 1, 16, 16, [8, 12], [0, 2, 5]),
        # Single head for KV but multiple heads for Queries (MQA/GQA pattern)
        (2, 10, 16, 8, 8, 1, 64, 512, 2, 32, 8, [16, 24], [0, 4, 10]),
        # # High D, High K
        (1, 8, 16, 4, 2, 1, 256, 512, 1, 32, 8, [60], [0, 8]),
        # Mixed batch: sequence 0 has 1 token (decode), sequence 1 has 5 tokens
        # (prefill)
        (2, 6, 16, 8, 4, 1, 64, 512, 2, 32, 8, [4, 6], [0, 1, 6]),
    ],
)
def test_streamindex_topk_shape(
    B,
    num_tokens,
    page_size,
    max_blocks,
    H_I,
    H_KV,
    D,
    k,
    compression_ratio,
    bq_sz,
    bkv_p,
    seq_lens_list,
    cu_q_lens_list,
):
    """Tests the shape and basic execution bounds of streamindex_topk."""
    _ = H_KV
    print(f"\n{'-'*60}")
    print(f"SHAPE TEST: B={B}, Tokens={num_tokens}, k={k}")
    print(f"{'-'*60}")

    query_projection = jnp.zeros((num_tokens, H_I, D), dtype=jnp.float32)
    indexer_weights = jnp.zeros((num_tokens, H_I), dtype=jnp.float32)

    num_pages = max_blocks * B
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128
    kv_cache = jnp.zeros((num_pages, page_size // 4, 4, width),
                         dtype=jnp.uint8)

    block_table = jnp.zeros((B, max_blocks), dtype=jnp.int32)
    page_indices = block_table.flatten()
    cu_q_lens = jnp.array(cu_q_lens_list, dtype=jnp.int32)

    print("Inputs:")
    print(f"  - query_projection: {query_projection.shape}")
    print(f"  - kv_cache:         {kv_cache.shape}")
    print(f"  - seq_lens:         {seq_lens_list}")
    print(f"  - cu_q_lens:        {cu_q_lens_list}")

    # Count number of decode sequences (T == 1) at the beginning of the batch
    num_decodes = 0
    while (num_decodes < B
           and (cu_q_lens_list[num_decodes + 1] - cu_q_lens_list[num_decodes])
           == 1):
        num_decodes += 1
    distribution = (num_decodes, num_decodes, B)
    expected_shape = (num_tokens, k)

    seq_lens = jnp.array(seq_lens_list, dtype=jnp.int32)

    out_shape_idxs = jax.eval_shape(
        streamindex_topk,
        query_projection,
        indexer_weights,
        kv_cache,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        k=k,
        compression_ratio=compression_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )

    print("\nOutputs:")
    print(f"  - Expected shape: {expected_shape}, dtype: int32")
    print(f"  - Actual shape:   {out_shape_idxs.shape}, dtype:"
          f" {out_shape_idxs.dtype}")

    assert out_shape_idxs.shape == expected_shape
    assert out_shape_idxs.dtype == jnp.int32
    print("Result: PASS")


@pytest.mark.parametrize(
    "B, T_list, S_list, page_size, H_I, H_KV, D, k, comp_ratio, bq_sz, bkv_p,"
    " block_table_list",
    [
        # 0. Single sequence, single page (page_indices shape (1,))
        (1, [1], [8], 8, 4, 1, 16, 1024, 1, 1, 16, [[0]]),
        # 1. Single sequence, highly fragmented block table
        (1, [4], [16], 8, 4, 1, 16, 1024, 2, 8, 16, [[2, 0]]),
        # 2. Batched Decode (T=1, B=2)
        (2, [1, 1], [12, 16], 8, 4, 1, 16, 1024, 1, 8, 16, [[1, 3], [0, 2]]),
        # 3. High GQA (8 Query Heads, 2 KV Heads)
        (1, [6], [20], 4, 8, 1, 16, 1024, 1, 8, 32, [[4, 1, 3, 0, 2]]),
        # 4. Multi-Batch Prefill with variable query/sequence lengths
        (2, [5, 3], [16, 16], 8, 4, 1, 16, 1024, 1, 8, 16, [[3, 1], [2, 4]]),
        # 5. Dummy sequences / Padding (B=3, but sequence 1 has 0 tokens)
        (
            3,
            [2, 0, 3],
            [16, 0, 8],
            8,
            4,
            1,
            16,
            1024,
            1,
            8,
            16,
            [[1, 2], [0, 0], [4, 3]],
        ),
        # 6. Mixed batch: sequence 0 has 1 token, sequence 1 has 5 tokens
        (2, [1, 5], [12, 16], 8, 4, 1, 16, 1024, 2, 8, 16, [[1, 3], [2, 0]]),
        (
            1,
            [4],
            [384],
            16,
            4,
            1,
            16,
            1024,
            1,
            8,
            8,
            [[
                13, 2, 21, 7, 0, 18, 5, 11, 23, 1, 9, 16, 3, 20, 6, 14, 22, 4,
                10, 17, 8, 15, 19, 12
            ]],
        ),
    ],
)
def test_streamindex_topk_numerical_correctness(
    B,
    T_list,
    S_list,
    page_size,
    H_I,
    H_KV,
    D,
    k,
    comp_ratio,
    bq_sz,
    bkv_p,
    block_table_list,
):
    """Executes randomized input data against a naive NumPy ground truth."""
    print(f"\n{'='*60}")
    print(f"BATCHED NUMERICAL TEST (B={B})")
    print(f"{'='*60}")

    np.random.seed(42)

    # 1. Setup Random Tensors
    num_tokens = sum(T_list)
    q = np.random.randn(num_tokens, H_I, D).astype(np.float32)
    weights = np.random.uniform(-1.5, 1.5,
                                size=(num_tokens, H_I)).astype(np.float32)

    # Create a unified physical KV Cache pool large enough for all block indices
    max_physical_page = np.max(block_table_list)
    float32_kv = np.random.randn(max_physical_page + 1, page_size, H_KV,
                                 D).astype(np.float32)

    # Pack cache using compressor's quantize_fp8_ue8m0 and _to_byte_lane
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128

    cache_kv = np.zeros((max_physical_page + 1, page_size // 4, 4, width),
                        dtype=np.uint8)
    dequantized_kv = np.zeros_like(float32_kv)

    for p in range(max_physical_page + 1):
        for s in range(page_size):
            for h in range(H_KV):
                row_kv = float32_kv[p, s, h]
                # Quantize using D as block size (each head query key has dimension D)
                q_jax, scale_jax = quantize_fp8_ue8m0(jnp.array(row_kv), D)
                q_bytes = np.array(_to_byte_lane(q_jax))
                scale_bytes = np.array(_to_byte_lane(scale_jax))
                dq = np.array(q_jax).astype(np.float32)
                dequantized_kv[p, s, h] = dq * float(scale_jax[0])
                record = np.concatenate([q_bytes, scale_bytes], axis=-1)
                record = np.pad(record, (0, width - record.shape[-1]))

                w_idx = s // 4
                lane_idx = s % 4
                cache_kv[p, w_idx, lane_idx] = record

    block_table = np.array(block_table_list, dtype=np.int32)
    page_indices = block_table.flatten()
    seq_lens = np.array(S_list, dtype=np.int32)
    cu_q_lens = np.concatenate([[0], np.cumsum(T_list)]).astype(np.int32)

    print("Configuration:")
    print(f"  - Tokens total: {num_tokens}, Sequences(B): {B}, K: {k}")
    print(f"  - GQA: {H_I} Query Heads -> {H_KV} KV Heads")
    print(f"  - Compression Ratio: {comp_ratio}")

    print("\nInputs Generated:")
    print(f"  - Query Tensor: {q.shape}")
    print(f"  - KV Cache:     {cache_kv.shape}")
    print(f"  - Block Table:  {block_table.tolist()}")

    # =====================================================================
    # 2. NAIVE COMPUTATION (The Ground Truth)
    # =====================================================================
    print("\n[1/3] Computing exact ground-truth using naive NumPy loops...")
    expected_topk = streamindex_topk_ref(
        q=q,
        weights=weights,
        kv=dequantized_kv,
        block_table=block_table,
        T_list=T_list,
        S_list=S_list,
        cu_q_lens=cu_q_lens,
        k=k,
        comp_ratio=comp_ratio,
        H_I=H_I,
        H_KV=H_KV,
    )

    print("\nGROUND TRUTH (Naive NumPy):")
    print(expected_topk)

    # =====================================================================
    # 3. Pallas KERNEL COMPUTATION
    # =====================================================================
    print(
        "\n[2/3] Executing optimized JAX Kernel (streamindex_pallas_topk)...")
    # Count number of decode sequences (T == 1) at the beginning of the batch
    num_decodes = 0
    while num_decodes < B and T_list[num_decodes] == 1:
        num_decodes += 1
    distribution = (num_decodes, num_decodes, B)

    actual_topk = streamindex_topk(
        q=jnp.array(q),
        indexer_weights=jnp.array(weights),
        cache_kv=jnp.array(cache_kv),
        seq_lens=jnp.array(seq_lens),
        page_indices=jnp.array(page_indices),
        cu_q_lens=jnp.array(cu_q_lens),
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )

    actual_topk_np = np.array(actual_topk)

    print("\nACTUAL OUTPUT (JAX/XLA Kernel):")
    print(actual_topk_np)

    # =====================================================================
    # 4. VERIFY
    # =====================================================================
    print("\n[3/3] Verifying absolute correctness...")
    np.testing.assert_array_equal(
        np.sort(actual_topk_np, axis=-1),
        np.sort(expected_topk, axis=-1),
        err_msg="JAX Pallas Kernel Top-K math did not match Naive Ground Truth",
    )
    print(
        "MATCH VERIFIED! JAX kernel handles batches, fragmentation, and dummy"
        " padding.\n")


def test_streamindex_topk_quantized():
    """Verifies correctness of streamindex_topk on FP8 packed cache."""
    np.random.seed(42)

    T_seq = 4
    S_seq = 2048
    page_size = 16
    H_I = 2
    H_KV = 1
    D = 128
    k = 512
    comp_ratio = 4
    bkv_p = 8  # page_size * bkv_p = 128 (TPU DMA contract of the scores kernel)
    bq_sz = 1

    q = np.random.randn(T_seq, H_I, D).astype(np.float32)
    weights = np.random.uniform(0.5, 1.5, size=(T_seq, H_I)).astype(np.float32)

    S_valid = S_seq // comp_ratio
    # We need enough pages to hold S_valid compressed tokens.
    num_pages = (S_valid + page_size - 1) // page_size
    float32_kv = np.random.randn(num_pages, page_size, H_KV,
                                 D).astype(np.float32)

    # Pack cache using compressor's quantize_fp8_ue8m0 and _to_byte_lane
    width = 256
    cache_kv = np.zeros((num_pages, page_size // 4, 4, width), dtype=np.uint8)

    # Dequantized KV for naive ground truth
    dequantized_kv = np.zeros_like(float32_kv)

    for p in range(num_pages):
        for s in range(page_size):
            for h_kv in range(H_KV):
                row_kv = float32_kv[p, s, h_kv]

                # Quantize using compressor helper
                # block_size = 128 since we want 1 scale factor for D=128
                q_jax, scale_jax = quantize_fp8_ue8m0(jnp.array(row_kv), 128)

                # Convert to bytes
                q_bytes = np.array(_to_byte_lane(q_jax))
                scale_bytes = np.array(_to_byte_lane(scale_jax))

                # Store dequantized version exactly as hardware sees it for accurate
                # validation
                dq = np.array(q_jax).astype(np.float32)
                dequantized_kv[p, s, h_kv] = dq * float(scale_jax[0])

                record = np.concatenate([q_bytes, scale_bytes], axis=-1)
                record = np.pad(record, (0, width - record.shape[-1]))

                w_idx = s // 4
                lane_idx = s % 4
                cache_kv[p, w_idx, lane_idx] = record

    # Compute expected top-k using exact dequantized keys
    seq_kv = np.concatenate([dequantized_kv[p] for p in range(num_pages)],
                            axis=0)

    naive_scores = np.full((T_seq, max(S_valid, k)), -np.inf, dtype=np.float32)

    for t_idx in range(T_seq):
        for s_idx in range(S_valid):
            score = 0.0
            for h in range(H_I):
                h_kv = h // (H_I // H_KV)
                inner = np.dot(q[t_idx, h], seq_kv[s_idx, h_kv])
                score += max(0.0, inner) * weights[t_idx, h]
            naive_scores[t_idx, s_idx] = score

    expected_topk = np.argsort(-naive_scores, axis=-1)[:, :k]
    for t_idx in range(T_seq):
        for i in range(k):
            idx = expected_topk[t_idx, i]
            if naive_scores[t_idx, idx] == -np.inf:
                expected_topk[t_idx, i] = -1

    # Pallas parameters
    actual_topk = streamindex_topk(
        q=jnp.array(q),
        indexer_weights=jnp.array(weights),
        cache_kv=jnp.array(cache_kv),
        seq_lens=jnp.array([S_seq], dtype=jnp.int32),
        page_indices=jnp.arange(num_pages, dtype=jnp.int32),
        cu_q_lens=jnp.array([0, T_seq], dtype=jnp.int32),
        distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
    )

    actual_topk_np = np.array(actual_topk)

    np.testing.assert_array_equal(
        np.sort(actual_topk_np, axis=-1),
        np.sort(expected_topk, axis=-1),
    )


# =====================================================================
# Early exit
# =====================================================================
# `enable_early_exit=True` must change cost, never the answer. Once a
# sequence's compressed length is <= k, top-k selects every visible position,
# so the scores decide nothing and the scoring kernel can be skipped outright.
# The guard is batch-wide, so the tests below pin both halves of that: the
# shortcut agrees with the scoring path when the whole batch is short, and is
# inert the moment any single sequence is not.

# `page_size * bkv_p` is 128, the TPU DMA contract of the scores kernel.
_EE_PAGE_SIZE = 128
_EE_PAGES_PER_SEQ = 32
_EE_BKV_P = 1
_EE_BQ_SZ = 4
_EE_H_I = 4
_EE_D = 128


def _ee_pack_cache(keys):
    """fp8-quantize ``[num_pages, page_size, D]`` keys into the uint8 cache.

    Same record layout the tests above build row by row -- ``[fp8 x D | one
    e8m0 scale byte | pad]`` rounded up to a 128-lane group, addressed as
    ``[page, slot // 4, slot % 4, width]`` -- just quantized in one shot.
    """
    num_pages, page_size, head_dim = keys.shape
    fp8, scale = quantize_fp8_ue8m0(jnp.asarray(keys), head_dim)
    record = jnp.concatenate(
        [_to_byte_lane(fp8), _to_byte_lane(scale)], axis=-1)
    width = -(-record.shape[-1] // 128) * 128
    record = jnp.pad(record, ((0, 0), (0, 0), (0, width - record.shape[-1])))
    return np.asarray(record).reshape(num_pages, page_size // 4, 4, width)


def _ee_inputs(q_lens, seq_lens, seed):
    """One kernel invocation at ``compression_ratio=1``.

    Every sequence gets a shuffled, disjoint set of physical pages so a bug in
    the page walk cannot hide behind sequential layout.
    """
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    num_tokens = int(sum(q_lens))
    num_pages = num_seqs * _EE_PAGES_PER_SEQ

    block_table = rng.permutation(num_pages).astype(np.int32).reshape(
        num_seqs, _EE_PAGES_PER_SEQ)
    keys = rng.standard_normal((num_pages, _EE_PAGE_SIZE, _EE_D),
                               dtype=np.float32)

    # `distribution` requires the decode-only sequences to lead the batch.
    num_decodes = 0
    while num_decodes < num_seqs and q_lens[num_decodes] == 1:
        num_decodes += 1

    return {
        "q":
        rng.standard_normal((num_tokens, _EE_H_I, _EE_D), dtype=np.float32),
        "indexer_weights":
        rng.uniform(0.25, 1.75, (num_tokens, _EE_H_I)).astype(np.float32),
        "cache_kv":
        _ee_pack_cache(keys),
        "seq_lens":
        np.asarray(seq_lens, np.int32),
        "page_indices":
        block_table.reshape(-1),
        "cu_q_lens":
        np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32),
        "distribution":
        np.array([num_decodes, num_decodes, num_seqs], np.int32),
    }


def _ee_both_paths(q_lens, seq_lens, k, seed=11):
    """Run the same batch with the flag off and on."""
    inputs = _ee_inputs(q_lens, seq_lens, seed)

    def run(enable_early_exit):
        return np.asarray(
            streamindex_topk(**{
                name: jnp.asarray(value)
                for name, value in inputs.items()
            },
                             k=k,
                             compression_ratio=1,
                             num_kv_pages_per_block=_EE_BKV_P,
                             num_queries_per_block=_EE_BQ_SZ,
                             enable_early_exit=enable_early_exit))

    return run(False), run(True)


def _ee_visible_counts(q_lens, seq_lens):
    """Causally visible compressed positions per query token.

    At ``compression_ratio=1`` the queries are the tail of their sequence, so
    query ``i`` of a sequence of length ``S`` sits at absolute position
    ``S - q_len + i`` and sees every position up to and including it.
    """
    counts = []
    for q_len, seq_len in zip(q_lens, seq_lens):
        counts.extend(seq_len - q_len + i + 1 for i in range(q_len))
    return counts


def _ee_assert_same_selection(early, baseline):
    """Both paths must select the same positions, in whatever order.

    The scoring path emits winners in ``approx_max_k`` order and the early-exit
    path in increasing KV position. Order is deliberately unspecified, so
    comparing the raw rows would assert something the kernel never promised --
    compare the selected sets.
    """
    assert early.shape == baseline.shape, (early.shape, baseline.shape)
    for token, (row_e, row_b) in enumerate(zip(early, baseline)):
        kept_e = np.sort(row_e[row_e >= 0])
        kept_b = np.sort(row_b[row_b >= 0])
        np.testing.assert_array_equal(
            kept_e, kept_b, f"token {token}: early exit selected a different "
            "set of positions than the scoring path")
        assert np.all(row_e[len(kept_e):] < 0), (
            f"token {token}: -1 padding is not a suffix: {row_e}")


def _ee_assert_every_visible_position(actual, q_lens, seq_lens, k):
    """With the whole batch under k the answer is exact, not a ranking.

    Every visible position wins, so the expected index set is closed-form and
    the check does not depend on score arithmetic at all.
    """
    for token, n_visible in enumerate(_ee_visible_counts(q_lens, seq_lens)):
        assert n_visible <= k, "test setup: sequence is not short"
        row = actual[token]
        np.testing.assert_array_equal(
            np.sort(row[row >= 0]), np.arange(n_visible),
            f"token {token}: not every visible position was reported")
        assert np.all(row[n_visible:] < 0), (
            f"token {token}: -1 padding is not a suffix: {row}")


def _ee_assert_full_rows(actual, q_lens, seq_lens, k):
    """Cheap sanity for the long-sequence cases.

    Correctness of the scoring path itself is what the tests above cover; here
    the claim under test is that the flag is inert, so this only rules out
    gross breakage.
    """
    counts = _ee_visible_counts(q_lens, seq_lens)
    limits = []
    for q_len, seq_len in zip(q_lens, seq_lens):
        limits.extend([seq_len] * q_len)
    for token, (n_visible, limit) in enumerate(zip(counts, limits)):
        assert n_visible > k, "test setup: sequence is not long"
        row = actual[token]
        assert np.all(row >= 0), f"token {token}: padded a full-length row"
        assert row.max() < limit, (
            f"token {token}: index {row.max()} past seq_len {limit}")


def test_streamindex_topk_early_exit_all_short():
    """Whole batch under k: the global fast path answers on its own."""
    q_lens, seq_lens, k = (1, 1, 1), (64, 96, 128), 512
    baseline, early = _ee_both_paths(q_lens, seq_lens, k)

    _ee_assert_every_visible_position(early, q_lens, seq_lens, k)
    _ee_assert_same_selection(early, baseline)


def test_streamindex_topk_early_exit_no_short_is_inert():
    """Every sequence over k: nothing is skipped, so nothing may change.

    Here the flag must be inert, which is a stronger claim than agreeing as
    sets -- no row took the shortcut, so the rows are bit-identical.
    """
    q_lens, seq_lens, k = (1, 1, 4), (2048, 3072, 4096), 128
    baseline, early = _ee_both_paths(q_lens, seq_lens, k)

    _ee_assert_full_rows(early, q_lens, seq_lens, k)
    np.testing.assert_array_equal(early, baseline)


def test_streamindex_topk_early_exit_is_batch_wide():
    """The guard is batch-wide, not per sequence.

    Two short decodes alongside one long prefill: `jnp.max(seq_lens)` is over
    k, so every token goes back through the scoring kernel including the short
    ones, and the flag must be bit-for-bit inert.
    """
    q_lens, seq_lens, k = (1, 1, 4), (64, 96, 4096), 512
    baseline, early = _ee_both_paths(q_lens, seq_lens, k)

    np.testing.assert_array_equal(early, baseline)


def test_streamindex_topk_early_exit_keeps_causal_mask():
    """The shortcut is still causal: a prefill token sees only its past.

    The whole sequence fits in k, so a bug that returned `[0, k)` instead of
    `[0, position]` would still look plausible -- this is what catches it.
    """
    q_lens, seq_lens, k = (4, ), (64, ), 256
    _, early = _ee_both_paths(q_lens, seq_lens, k)

    _ee_assert_every_visible_position(early, q_lens, seq_lens, k)
    # Four query tokens at the tail of a 64-long sequence sit at absolute
    # positions 60..63, so they keep 61..64 positions respectively.
    kept = [int(np.count_nonzero(row >= 0)) for row in early]
    assert kept == [61, 62, 63, 64], kept


def test_pallas_smem_oom_when_unclamped(monkeypatch):
    """Verifies that Pallas compilation of _scores_kernel fails with SMEM OOM"""
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import metadata

    orig_compute_metadata = metadata.compute_metadata

    def _mock_compute_metadata(*args, **kwargs):
        res = orig_compute_metadata(*args, **kwargs)
        # Return metadata arrays of length 100,000.
        # In Pallas, 100,000 steps require ~2.4 MB of SMEM.
        return metadata.MetadataRef.create(
            num_steps=res.num_steps,
            batch_tile_idx=jnp.zeros((100000, ), dtype=jnp.int32),
            bq_idx=jnp.zeros((100000, ), dtype=jnp.int32),
            bkv_idx=jnp.zeros((100000, ), dtype=jnp.int32),
        )

    monkeypatch.setattr(metadata, "compute_metadata", _mock_compute_metadata)

    num_seqs = 32
    seq_len = 128
    num_pages_per_seq = seq_len // 128
    total_pages = num_seqs * num_pages_per_seq
    head_dim = 128
    width = 256

    q = jnp.zeros((num_seqs * seq_len, 1, head_dim), dtype=jnp.float32)
    weights = jnp.ones((num_seqs * seq_len, 1), dtype=jnp.float32)
    cache_kv = jnp.zeros((total_pages, 32, 4, width), dtype=np.uint8)
    seq_lens = jnp.full((num_seqs, ), seq_len, dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    cu_q_lens = jnp.arange(0, (num_seqs + 1) * seq_len,
                           seq_len,
                           dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    with pytest.raises(Exception) as exc_info:
        streamindex_topk(
            q=q,
            indexer_weights=weights,
            cache_kv=cache_kv,
            seq_lens=seq_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            k=512,
            compression_ratio=1,
            num_kv_pages_per_block=1,
            num_queries_per_block=1,
        )
    err_str = str(exc_info.value).lower()
    assert "smem" in err_str, f"Expected SMEM OOM error, got: {exc_info.value}"


def test_streamindex_topk_buffer_count():
    """Verifies configurable buffer_count parameter behavior."""
    B = 2
    T = 4
    S = 16
    page_size = 128
    H_I = 4
    D = 16
    k = 16
    comp_ratio = 1
    bq_sz = 8
    bkv_p = 2

    total_tokens = B * T
    max_blocks = 2
    num_pages = max_blocks * B
    q_lkv_dim = ((D + 127) // 128) * 128
    record_width = q_lkv_dim + (q_lkv_dim // 128)
    width = ((record_width + 127) // 128) * 128

    rng = np.random.default_rng(42)
    q = jnp.array(rng.standard_normal((total_tokens, H_I, D)),
                  dtype=jnp.float32)
    weights = jnp.array(rng.standard_normal((total_tokens, H_I)),
                        dtype=jnp.float32)
    cache_kv = jnp.zeros((num_pages, page_size // 4, 4, width),
                         dtype=jnp.uint8)
    seq_lens = jnp.array([S] * B, dtype=jnp.int32)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, T, 2 * T], dtype=jnp.int32)
    distribution = jnp.array([0, 0, B], dtype=jnp.int32)

    # 1. Custom integer buffer_count
    topk_int = streamindex_topk(
        q=q,
        indexer_weights=weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
        buffer_count=2,
    )
    assert topk_int.shape == (total_tokens, k)

    # 2. Custom tuple buffer_count
    topk_tuple = streamindex_topk(
        q=q,
        indexer_weights=weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
        buffer_count=(2, 2, 2),
    )
    assert topk_tuple.shape == (total_tokens, k)
    np.testing.assert_array_equal(np.array(topk_int), np.array(topk_tuple))

    # 3. Invalid buffer_count length raises ValueError
    with pytest.raises(ValueError, match="buffer_count must be a 3-tuple"):
        streamindex_topk(
            q=q,
            indexer_weights=weights,
            cache_kv=cache_kv,
            seq_lens=seq_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            k=k,
            compression_ratio=comp_ratio,
            num_kv_pages_per_block=bkv_p,
            num_queries_per_block=bq_sz,
            buffer_count=(2, 2),
        )


def test_streamindex_topk_return_scores_and_validation():
    q_dtype = jnp.float8_e4m3fn
    kv_dtype = jnp.uint8
    b = 2
    t = 4
    s = 128
    total_tokens = b * t
    h_i = 4
    d_i = 128
    k = 16
    comp_ratio = 1
    page_size = 64
    bkv_p = 2
    bq_sz = 4
    width = 256

    num_pages = b * (s // page_size)
    q = jnp.ones((total_tokens, h_i, d_i), dtype=q_dtype)
    weights = jnp.ones((total_tokens, h_i), dtype=jnp.float32)
    cache_kv = jnp.zeros((num_pages, page_size // 4, 4, width), dtype=kv_dtype)
    seq_lens = jnp.array([s] * b, dtype=jnp.int32)
    page_indices = jnp.arange(num_pages, dtype=jnp.int32)
    cu_q_lens = jnp.array([0, t, 2 * t], dtype=jnp.int32)
    distribution = jnp.array([0, 0, b], dtype=jnp.int32)

    # 1. return_scores=True
    idxs, scores = streamindex_topk(
        q=q,
        indexer_weights=weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        k=k,
        compression_ratio=comp_ratio,
        num_kv_pages_per_block=bkv_p,
        num_queries_per_block=bq_sz,
        return_scores=True,
    )
    assert idxs.shape == (total_tokens, k)
    assert scores.shape == (total_tokens, k)

    # 2. cp_size < 1 raises ValueError
    with pytest.raises(ValueError, match="cp_size must be >= 1"):
        streamindex_topk(
            q=q,
            indexer_weights=weights,
            cache_kv=cache_kv,
            seq_lens=seq_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            k=k,
            compression_ratio=comp_ratio,
            num_kv_pages_per_block=bkv_p,
            num_queries_per_block=bq_sz,
            cp_size=0,
        )

    # 3. interleave_size not multiple of compression_ratio
    with pytest.raises(ValueError,
                       match="must be a multiple of compression_ratio"):
        streamindex_topk(
            q=q,
            indexer_weights=weights,
            cache_kv=cache_kv,
            seq_lens=seq_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            k=k,
            compression_ratio=2,
            num_kv_pages_per_block=bkv_p,
            num_queries_per_block=bq_sz,
            cp_size=2,
            interleave_size=3,
        )

    # 4. enable_early_exit with cp_size > 1
    with pytest.raises(
            NotImplementedError,
            match="enable_early_exit is not supported with cp_size > 1",
    ):
        streamindex_topk(
            q=q,
            indexer_weights=weights,
            cache_kv=cache_kv,
            seq_lens=seq_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            k=k,
            compression_ratio=comp_ratio,
            num_kv_pages_per_block=bkv_p,
            num_queries_per_block=bq_sz,
            enable_early_exit=True,
            cp_size=2,
        )

    # 5. return_scores with enable_early_exit
    with pytest.raises(
            NotImplementedError,
            match="return_scores is not supported with enable_early_exit",
    ):
        streamindex_topk(
            q=q,
            indexer_weights=weights,
            cache_kv=cache_kv,
            seq_lens=seq_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            k=k,
            compression_ratio=comp_ratio,
            num_kv_pages_per_block=bkv_p,
            num_queries_per_block=bq_sz,
            enable_early_exit=True,
            return_scores=True,
        )


def main(argv):
    del argv

    raise SystemExit(pytest.main([__file__, "-p", "no:cacheprovider"]))


if __name__ == "__main__":
    from absl import app

    app.run(main)
