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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention

pytestmark = pytest.mark.tpu_test


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


def _build_ragged_case(
    q_lens: list[int],
    previous_tokens: list[int],
    num_q_heads: int,
    seed: int,
):
    """Inputs for a ragged batch plus a numpy log-sum-exp reference.

    Sequence s has `previous_tokens[s]` keys in the cache and `q_lens[s]` new
    tokens this step; query i of it sits at position `previous_tokens[s] + i`
    and, under the causal mask, attends the keys at positions 0..that one.
    The reference rounds every input to bf16 first, as the kernel reads them,
    and does the rest in f32.
    """
    lkv_dim, r_dim, padded_r_dim = 512, 64, 128
    page_size, packing = 16, 2
    num_seqs = len(q_lens)
    max_num_tokens = sum(q_lens)
    kv_lens = [p + q for p, q in zip(previous_tokens, q_lens)]
    pages_per_seq = max((kv + page_size - 1) // page_size for kv in kv_lens)
    rng = np.random.default_rng(seed)

    def bf16(x):
        return jnp.asarray(x, jnp.bfloat16)

    ql_nope = bf16(rng.normal(size=(num_q_heads, max_num_tokens, lkv_dim)) * 0.05)
    q_pe = bf16(rng.normal(size=(max_num_tokens, num_q_heads, r_dim)) * 0.05)
    new_kv_c = bf16(rng.normal(size=(max_num_tokens, lkv_dim)))
    new_k_pe = bf16(rng.normal(size=(max_num_tokens, r_dim)))
    old_kv_c = bf16(rng.normal(size=(num_seqs, pages_per_seq * page_size, lkv_dim)))
    old_k_pe = bf16(rng.normal(size=(num_seqs, pages_per_seq * page_size, r_dim)))
    cache = jnp.concatenate(
        [old_kv_c, jnp.pad(old_k_pe, ((0, 0), (0, 0), (0, padded_r_dim - r_dim)))],
        axis=2,
    ).reshape(
        num_seqs * pages_per_seq, page_size // packing, packing, lkv_dim + padded_r_dim
    )

    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    reference = np.zeros((max_num_tokens, num_q_heads), np.float32)
    f32 = lambda x: np.asarray(x.astype(jnp.float32))  # noqa: E731
    for s in range(num_seqs):
        q_start = int(cu_q_lens[s])
        q = np.concatenate(
            [
                f32(ql_nope)[:, q_start : q_start + q_lens[s]],
                f32(q_pe)[q_start : q_start + q_lens[s]].transpose(1, 0, 2),
            ],
            axis=2,
        )  # [heads, q_len, 576]
        old = np.concatenate(
            [
                f32(old_kv_c)[s, : previous_tokens[s]],
                f32(old_k_pe)[s, : previous_tokens[s]],
            ],
            axis=1,
        )
        new = np.concatenate(
            [
                f32(new_kv_c)[q_start : q_start + q_lens[s]],
                f32(new_k_pe)[q_start : q_start + q_lens[s]],
            ],
            axis=1,
        )
        keys = np.concatenate([old, new], axis=0)  # [kv_len, 576]
        scores = np.einsum("hqd,kd->hqk", q, keys)
        for i in range(q_lens[s]):
            visible = scores[:, i, : previous_tokens[s] + i + 1]
            m = visible.max(axis=1)
            reference[q_start + i] = m + np.log(
                np.exp(visible - m[:, None]).sum(axis=1)
            )

    args = (
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        jnp.asarray(kv_lens, jnp.int32),
        jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32),
        jnp.asarray(cu_q_lens),
    )
    return args, cache, reference


@pytest.mark.parametrize(
    ("s_dtype", "atol"),
    [(jnp.float32, 1e-4), (jnp.bfloat16, 5e-2)],
    ids=["f32_scores", "bf16_scores"],
)
def test_return_lse_matches_reference_and_leaves_output_unchanged(
    s_dtype, atol
) -> None:
    """`return_lse` adds the log-sum-exp and changes nothing else.

    Covers the batched-decode, decode and mixed programs in one call, a
    head count that needs VMEM padding (12, as Kimi-K3 at TP=8), query
    blocks larger than one, the causal mask inside a ragged prefill, and
    the sequence tail beyond the last query block.
    """
    q_lens = [1, 1, 1, 1, 1, 5, 16, 33]
    previous_tokens = [114, 7, 0, 40, 200, 30, 9, 100]
    distribution = jnp.asarray([5, 5, len(q_lens)], jnp.int32)
    args, cache, reference = _build_ragged_case(q_lens, previous_tokens, 12, seed=7)
    kwargs = dict(
        num_kv_pages_per_block=(3, 1, 2),
        num_queries_per_block=(1, 16, 16),
        decode_batch_size=4,
        s_dtype=s_dtype,
    )
    out, updated = mla_ragged_paged_attention(
        *args[:4], cache.copy(), *args[4:], distribution, **kwargs
    )
    out_lse, updated_lse, lse = mla_ragged_paged_attention(
        *args[:4], cache.copy(), *args[4:], distribution, **kwargs, return_lse=True
    )
    jax.block_until_ready((out, updated, out_lse, updated_lse, lse))

    np.testing.assert_array_equal(np.asarray(out_lse), np.asarray(out))
    np.testing.assert_array_equal(np.asarray(updated_lse), np.asarray(updated))
    assert lse.shape == (sum(q_lens), 12)
    assert lse.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(lse), reference, atol=atol, rtol=0)


def test_return_lse_with_three_heads_and_padded_tokens() -> None:
    """Three heads (Kimi-K3 at TP=32) pad to four in the kernel; the padding
    heads and the tokens past the last sequence must not leak into the
    result, which reads 0 there."""
    q_lens = [1, 1, 2]
    previous_tokens = [50, 3, 20]
    args, cache, reference = _build_ragged_case(q_lens, previous_tokens, 3, seed=11)
    pad_tokens = 4
    ql_nope, q_pe, new_kv_c, new_k_pe, kv_lens, page_indices, cu_q_lens = args
    ql_nope = jnp.pad(ql_nope, ((0, 0), (0, pad_tokens), (0, 0)))
    q_pe = jnp.pad(q_pe, ((0, pad_tokens), (0, 0), (0, 0)))
    new_kv_c = jnp.pad(new_kv_c, ((0, pad_tokens), (0, 0)))
    new_k_pe = jnp.pad(new_k_pe, ((0, pad_tokens), (0, 0)))
    distribution = jnp.asarray([2, 2, 3], jnp.int32)
    _, _, lse = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        num_kv_pages_per_block=1,
        num_queries_per_block=8,
        s_dtype=jnp.float32,
        return_lse=True,
    )
    lse.block_until_ready()
    assert lse.shape == (sum(q_lens) + pad_tokens, 3)
    np.testing.assert_allclose(
        np.asarray(lse)[: sum(q_lens)], reference, atol=1e-4, rtol=0
    )
    np.testing.assert_array_equal(np.asarray(lse)[sum(q_lens) :], 0.0)


@pytest.mark.parametrize(
    ("kwargs", "distribution"),
    [
        ({"mixed_q_split": 2}, [2, 2, 5]),
        ({"two_step_flash_attention": False}, [2, 2, 5]),
        # The prefill program only runs with chunk_prefill_size; without it
        # a sequence in the [i, j) range is not attended by any program.
        ({"chunk_prefill_size": 16}, [2, 4, 5]),
    ],
    ids=["q_split_2", "one_step", "chunk_prefill_16"],
)
def test_return_lse_on_other_kernel_paths(kwargs, distribution) -> None:
    """The lse is read from the same scratch on every kernel variant; the
    split-query mixed path, the one-step attention and the fixed-length
    prefill program must agree with the reference as well."""
    q_lens = [1, 1, 16, 16, 7]
    previous_tokens = [40, 3, 20, 100, 9]
    distribution = jnp.asarray(distribution, jnp.int32)
    args, cache, reference = _build_ragged_case(q_lens, previous_tokens, 12, seed=3)
    out, updated = mla_ragged_paged_attention(
        *args[:4],
        cache.copy(),
        *args[4:],
        distribution,
        num_kv_pages_per_block=(3, 1, 2),
        num_queries_per_block=(1, 16, 16),
        s_dtype=jnp.float32,
        **kwargs,
    )
    out_lse, updated_lse, lse = mla_ragged_paged_attention(
        *args[:4],
        cache.copy(),
        *args[4:],
        distribution,
        num_kv_pages_per_block=(3, 1, 2),
        num_queries_per_block=(1, 16, 16),
        s_dtype=jnp.float32,
        return_lse=True,
        **kwargs,
    )
    jax.block_until_ready((out, updated, out_lse, updated_lse, lse))
    np.testing.assert_array_equal(np.asarray(out_lse), np.asarray(out))
    np.testing.assert_array_equal(np.asarray(updated_lse), np.asarray(updated))
    np.testing.assert_allclose(np.asarray(lse), reference, atol=1e-4, rtol=0)
