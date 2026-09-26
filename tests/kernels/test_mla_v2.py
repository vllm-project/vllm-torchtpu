# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention


def test_query_heads_are_padded_only_in_vmem() -> None:
    """Four BF16 query heads require padding to the eight-head VMEM tile."""
    num_q_heads = 4
    num_tokens = 1
    lkv_dim = 128
    r_dim = 128
    page_size = 128
    packing = 2

    ql_nope = jnp.ones((num_q_heads, num_tokens, lkv_dim), jnp.bfloat16)
    q_pe = jnp.ones((num_tokens, num_q_heads, r_dim), jnp.bfloat16)
    new_kv_c = jnp.arange(lkv_dim, dtype=jnp.bfloat16)[None, :]
    new_k_pe = jnp.ones((num_tokens, r_dim), jnp.bfloat16)
    cache_kv = jnp.zeros(
        (1, page_size // packing, packing, lkv_dim + r_dim),
        jnp.bfloat16,
    )

    output, _ = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        jnp.asarray([1], jnp.int32),
        jnp.asarray([0], jnp.int32),
        jnp.asarray([0, 1], jnp.int32),
        jnp.asarray([1, 1, 1], jnp.int32),
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
    )
    output.block_until_ready()

    assert output.shape == (num_q_heads, num_tokens, lkv_dim)
    np.testing.assert_array_equal(
        np.asarray(output),
        np.broadcast_to(np.asarray(new_kv_c), output.shape),
    )


def test_identical_mixed_prefills_do_not_interfere() -> None:
    """Identical ragged prefills spanning distinct pages stay identical.

    This mirrors Kimi-K3 at TP=8: 12 local query heads, 512 latent dims,
    64 RoPE dims, eight concurrent 114-token prompts, and 16-token cache
    pages.  The existing single-token test cannot cover token offsets,
    multi-page table rows, or the mixed-prefill program grid.
    """
    num_q_heads = 12
    tokens_per_seq = 114
    num_seqs = 8
    lkv_dim = 512
    r_dim = 64
    padded_r_dim = 128
    page_size = 16
    packing = 2
    pages_per_seq = (tokens_per_seq + page_size - 1) // page_size

    ql_nope_one = (
        (jnp.arange(num_q_heads * tokens_per_seq * lkv_dim, dtype=jnp.int32) % 31)
        .astype(jnp.bfloat16)
        .reshape(num_q_heads, tokens_per_seq, lkv_dim)
    )
    q_pe_one = (
        (jnp.arange(tokens_per_seq * num_q_heads * r_dim, dtype=jnp.int32) % 29)
        .astype(jnp.bfloat16)
        .reshape(tokens_per_seq, num_q_heads, r_dim)
    )
    new_kv_c_one = (
        (jnp.arange(tokens_per_seq * lkv_dim, dtype=jnp.int32) % 23)
        .astype(jnp.bfloat16)
        .reshape(tokens_per_seq, lkv_dim)
    )
    new_k_pe_one = (
        (jnp.arange(tokens_per_seq * r_dim, dtype=jnp.int32) % 19)
        .astype(jnp.bfloat16)
        .reshape(tokens_per_seq, r_dim)
    )

    ql_nope = jnp.concatenate([ql_nope_one] * num_seqs, axis=1)
    q_pe = jnp.concatenate([q_pe_one] * num_seqs, axis=0)
    new_kv_c = jnp.concatenate([new_kv_c_one] * num_seqs, axis=0)
    new_k_pe = jnp.concatenate([new_k_pe_one] * num_seqs, axis=0)
    cache_kv = jnp.zeros(
        (
            num_seqs * pages_per_seq,
            page_size // packing,
            packing,
            lkv_dim + padded_r_dim,
        ),
        jnp.bfloat16,
    )

    output, updated_cache = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        jnp.full((num_seqs,), tokens_per_seq, jnp.int32),
        jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32),
        jnp.arange(num_seqs + 1, dtype=jnp.int32) * tokens_per_seq,
        jnp.asarray([0, 0, num_seqs], jnp.int32),
        num_kv_pages_per_block=1,
        num_queries_per_block=16,
    )
    output.block_until_ready()
    updated_cache.block_until_ready()

    expected_output = np.asarray(output[:, :tokens_per_seq])
    expected_cache = np.asarray(updated_cache[:pages_per_seq])
    for seq_idx in range(1, num_seqs):
        np.testing.assert_array_equal(
            expected_output,
            np.asarray(
                output[:, seq_idx * tokens_per_seq : (seq_idx + 1) * tokens_per_seq]
            ),
        )
        np.testing.assert_array_equal(
            expected_cache,
            np.asarray(
                updated_cache[seq_idx * pages_per_seq : (seq_idx + 1) * pages_per_seq]
            ),
        )


def test_batched_decode_matches_unbatched_decode() -> None:
    """Kimi-K3's four-way MLA decode must preserve request isolation."""
    num_q_heads = 12
    num_seqs = 8
    max_num_tokens = 16
    previous_tokens = 114
    lkv_dim = 512
    r_dim = 64
    padded_r_dim = 128
    page_size = 16
    packing = 2
    pages_per_seq = (previous_tokens + 1 + page_size - 1) // page_size

    ql_nope_token = (
        (jnp.arange(num_q_heads * lkv_dim, dtype=jnp.int32) % 31)
        .astype(jnp.bfloat16)
        .reshape(num_q_heads, 1, lkv_dim)
    )
    ql_nope = jnp.pad(
        jnp.repeat(ql_nope_token, num_seqs, axis=1),
        ((0, 0), (0, max_num_tokens - num_seqs), (0, 0)),
    )
    q_pe_token = (
        (jnp.arange(num_q_heads * r_dim, dtype=jnp.int32) % 29)
        .astype(jnp.bfloat16)
        .reshape(1, num_q_heads, r_dim)
    )
    q_pe = jnp.pad(
        jnp.repeat(q_pe_token, num_seqs, axis=0),
        ((0, max_num_tokens - num_seqs), (0, 0), (0, 0)),
    )
    new_kv_c_token = (jnp.arange(lkv_dim, dtype=jnp.int32) % 23).astype(jnp.bfloat16)[
        None, :
    ]
    new_kv_c = jnp.pad(
        jnp.repeat(new_kv_c_token, num_seqs, axis=0),
        ((0, max_num_tokens - num_seqs), (0, 0)),
    )
    new_k_pe_token = (jnp.arange(r_dim, dtype=jnp.int32) % 19).astype(jnp.bfloat16)[
        None, :
    ]
    new_k_pe = jnp.pad(
        jnp.repeat(new_k_pe_token, num_seqs, axis=0),
        ((0, max_num_tokens - num_seqs), (0, 0)),
    )

    old_kv_c = (
        (jnp.arange(previous_tokens * lkv_dim, dtype=jnp.int32) % 17)
        .astype(jnp.bfloat16)
        .reshape(previous_tokens, lkv_dim)
    )
    old_k_pe = (
        (jnp.arange(previous_tokens * r_dim, dtype=jnp.int32) % 13)
        .astype(jnp.bfloat16)
        .reshape(previous_tokens, r_dim)
    )
    old_k_pe = jnp.pad(old_k_pe, ((0, 0), (0, padded_r_dim - r_dim)))
    cache_row = jnp.concatenate([old_kv_c, old_k_pe], axis=1)
    cache_row = jnp.pad(
        cache_row,
        ((0, pages_per_seq * page_size - previous_tokens), (0, 0)),
    ).reshape(pages_per_seq, page_size // packing, packing, lkv_dim + padded_r_dim)
    cache = jnp.concatenate([cache_row] * num_seqs, axis=0)

    args = (
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        jnp.full((num_seqs,), previous_tokens + 1, jnp.int32),
        jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32),
        jnp.arange(num_seqs + 1, dtype=jnp.int32),
        jnp.asarray([num_seqs, num_seqs, num_seqs], jnp.int32),
    )

    unbatched_output, unbatched_cache = mla_ragged_paged_attention(
        args[0],
        args[1],
        args[2],
        args[3],
        cache.copy(),
        *args[4:],
        num_kv_pages_per_block=(3, 1, 1),
        num_queries_per_block=(1, 16, 16),
        decode_batch_size=1,
    )
    batched_output, batched_cache = mla_ragged_paged_attention(
        args[0],
        args[1],
        args[2],
        args[3],
        cache.copy(),
        *args[4:],
        num_kv_pages_per_block=(3, 1, 1),
        num_queries_per_block=(1, 16, 16),
        decode_batch_size=4,
    )
    unbatched_output.block_until_ready()
    batched_output.block_until_ready()
    unbatched_cache.block_until_ready()
    batched_cache.block_until_ready()

    np.testing.assert_array_equal(
        np.asarray(batched_output),
        np.asarray(unbatched_output),
    )
    np.testing.assert_array_equal(
        np.asarray(batched_cache),
        np.asarray(unbatched_cache),
    )


def test_mixed_q_split_matches_unsplit_mixed_mode() -> None:
    """Mixed prefill+decode with mixed_q_split > 1 must numerically match
    unsplit execution.
    """
    num_q_heads = 12
    num_decode = 2
    num_prefill = 2
    prefill_len = 64
    num_seqs = num_decode + num_prefill
    total_tokens = num_decode * 1 + num_prefill * prefill_len
    lkv_dim = 512
    r_dim = 64
    padded_r_dim = 128
    page_size = 16
    packing = 2
    pages_per_seq = (prefill_len + page_size - 1) // page_size

    ql_nope = (
        (jnp.arange(num_q_heads * total_tokens * lkv_dim, dtype=jnp.int32) % 31)
        .astype(jnp.bfloat16)
        .reshape(num_q_heads, total_tokens, lkv_dim)
    )
    q_pe = (
        (jnp.arange(total_tokens * num_q_heads * r_dim, dtype=jnp.int32) % 29)
        .astype(jnp.bfloat16)
        .reshape(total_tokens, num_q_heads, r_dim)
    )
    new_kv_c = (
        (jnp.arange(total_tokens * lkv_dim, dtype=jnp.int32) % 23)
        .astype(jnp.bfloat16)
        .reshape(total_tokens, lkv_dim)
    )
    new_k_pe = (
        (jnp.arange(total_tokens * r_dim, dtype=jnp.int32) % 19)
        .astype(jnp.bfloat16)
        .reshape(total_tokens, r_dim)
    )

    cache_kv = jnp.zeros(
        (
            num_seqs * pages_per_seq,
            page_size // packing,
            packing,
            lkv_dim + padded_r_dim,
        ),
        jnp.bfloat16,
    )
    seq_lens = jnp.array([16, 16, prefill_len, prefill_len], dtype=jnp.int32)
    block_tables = jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32)
    query_start_loc = jnp.array(
        [0, 1, 2, 2 + prefill_len, total_tokens], dtype=jnp.int32
    )
    request_distribution = jnp.array(
        [num_decode, num_decode, num_seqs], dtype=jnp.int32
    )

    unsplit_out, unsplit_cache = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv.copy(),
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        num_kv_pages_per_block=(1, 1, 1),
        num_queries_per_block=(1, 16, 16),
        mixed_q_split=1,
    )
    split_out, split_cache = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv.copy(),
        seq_lens,
        block_tables,
        query_start_loc,
        request_distribution,
        num_kv_pages_per_block=(1, 1, 1),
        num_queries_per_block=(1, 16, 32),
        mixed_q_split=2,
    )
    unsplit_out.block_until_ready()
    split_out.block_until_ready()
    unsplit_cache.block_until_ready()
    split_cache.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(split_out),
        np.asarray(unsplit_out),
        rtol=1e-2,
        atol=1e-2,
    )
    np.testing.assert_array_equal(
        np.asarray(split_cache),
        np.asarray(unsplit_cache),
    )
