# SPDX-License-Identifier: Apache-2.0
"""Widened RPAd buckets must respect runtime lengths and preserve paged KV."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper


@pytest.mark.parametrize(
    ("bucket", "q_lens", "kv_lengths"),
    [
        (4, (1, 3, 3, 16), (1025, 1025, 4, 1030)),
        (8, (1, 5, 7, 5, 16), (4097, 4098, 4099, 6, 4102)),
    ],
)
@pytest.mark.parametrize("widened", [False, True], ids=["baseline", "widened"])
def test_ragged_verify_matches_reference_and_preserves_cache(
    bucket, q_lens, kv_lengths, widened
):
    """Catch padded-query leakage, cross-page writeback and mixed-stage damage.

    The last request stays in MIXED. Verification uses MIXED in the baseline
    and shares a wider DECODE bucket with q1 in the candidate. Unused
    requests and physical pages are sentinels. A short-history verification
    makes missing causal masks visible instead of diluted by 4K history.
    """
    rng = np.random.default_rng(994)
    num_q_heads, head_dim, page_size = 16, 256, 256
    max_seqs = 8
    pages_per_seq = (max(kv_lengths) + page_size - 1) // page_size + 1
    num_seqs = len(q_lens)
    total_tokens = sum(q_lens)
    layout = configs.KVLayout.SEQ_ALONG_LANE
    cache_shape = wrapper.get_kv_cache_shape(
        max_seqs * pages_per_seq + 2,
        page_size,
        1,
        head_dim,
        jnp.float8_e4m3fn,
        kv_layout=layout,
    )

    def random_array(shape, dtype):
        return (rng.standard_normal(shape) * 0.5).astype(dtype)

    q = random_array((total_tokens, num_q_heads, head_dim), jnp.bfloat16)
    k = random_array((total_tokens, 1, head_dim), jnp.float8_e4m3fn)
    v = random_array((total_tokens, 1, head_dim), jnp.float8_e4m3fn)
    before = random_array(cache_shape, jnp.float8_e4m3fn)
    expected_cache = before.copy()
    page_table = rng.permutation(cache_shape[0])[: max_seqs * pages_per_seq]
    page_table = page_table.reshape(max_seqs, pages_per_seq).astype(np.int32)
    kv_lens = np.zeros(max_seqs, dtype=np.int32)
    kv_lens[:num_seqs] = kv_lengths
    cu_q_lens = np.full(max_seqs + 1, total_tokens, dtype=np.int32)
    cu_q_lens[: num_seqs + 1] = np.cumsum((0, *q_lens))
    reference = np.empty(q.shape, dtype=np.float32)
    protected = np.ones((cache_shape[0], page_size), dtype=bool)

    for seq, (q_len, kv_len) in enumerate(zip(q_lens, kv_lengths)):
        start, end = cu_q_lens[seq : seq + 2]
        history_len = kv_len - q_len
        # HND writes whole pages. Beyond kv_len, the partially filled final
        # page is scratch, not live KV. Every other location is protected.
        tail = kv_len % page_size
        if tail:
            protected[page_table[seq, kv_len // page_size], tail:] = False
        for token, pos in enumerate(range(history_len, kv_len), start):
            page = page_table[seq, pos // page_size]
            lane = pos % page_size
            expected_cache[page, :, :, :, lane] = np.stack(
                (k[token, 0], v[token, 0])
            ).reshape(2, head_dim // 4, 4)

        # Independently gather logical K/V from the full expected cache.
        logical_kv = np.stack(
            [
                expected_cache[
                    page_table[seq, pos // page_size], :, :, :, pos % page_size
                ].reshape(2, head_dim)
                for pos in range(kv_len)
            ]
        ).astype(np.float32)
        scores = (
            np.einsum("qhd,kd->qhk", q[start:end].astype(np.float32), logical_kv[:, 0])
            * head_dim**-0.5
        )
        visible = np.arange(kv_len)[None, :] <= history_len + np.arange(q_len)[:, None]
        scores = np.where(visible[:, None, :], scores, -np.inf)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        reference[start:end] = np.einsum("qhk,kd->qhd", weights, logical_kv[:, 1])

    decode_end = num_seqs - 1 if widened else 1
    output, cache = jax.block_until_ready(
        wrapper.ragged_paged_attention(
            jnp.asarray(q),
            jnp.asarray(k),
            jnp.asarray(v),
            jnp.asarray(before),
            jnp.asarray(kv_lens),
            jnp.asarray(page_table.ravel()),
            jnp.asarray(cu_q_lens),
            jnp.asarray([decode_end, decode_end, num_seqs], jnp.int32),
            sm_scale=head_dim**-0.5,
            kv_layout=layout,
            decode_query_size=bucket if widened else 1,
        )
    )
    actual = np.asarray(output.astype(jnp.float32))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, atol=2e-3, rtol=2e-2)
    for seq in range(num_seqs):
        start, end = cu_q_lens[seq : seq + 2]
        relative_l2 = np.linalg.norm(
            actual[start:end] - reference[start:end]
        ) / np.linalg.norm(reference[start:end])
        assert relative_l2 < 0.02, f"seq {seq}: relative L2 = {relative_l2}"
    # Protect all history, new destinations (on both sides of a page),
    # inactive requests, untouched pages and free pages. Only tail scratch
    # in each active request's last page is excluded.
    np.testing.assert_array_equal(
        np.asarray(cache).transpose(0, 4, 1, 2, 3)[protected],
        expected_cache.transpose(0, 4, 1, 2, 3)[protected],
    )


def _lse_combine(out_a, lse_a, out_b, lse_b):
    """numpy twin of cp_attention.lse_weighted_combine (handles -inf lse)."""
    m = np.maximum(lse_a, lse_b)
    m_finite = np.isfinite(m)
    m_safe = np.where(m_finite, m, 0.0)
    ea = np.exp(lse_a - m_safe)
    eb = np.exp(lse_b - m_safe)
    norm = ea + eb
    out_a = np.nan_to_num(out_a, nan=0.0, posinf=0.0, neginf=0.0)
    out_b = np.nan_to_num(out_b, nan=0.0, posinf=0.0, neginf=0.0)
    norm_safe = np.where(norm > 0, norm, 1.0)
    out = (ea[..., None] * out_a + eb[..., None] * out_b) / norm_safe[..., None]
    lse = np.where(m_finite, m + np.log(norm_safe), m)
    return out, lse


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("widened", [False, True], ids=["baseline", "widened"])
def test_dcp_two_pass_widened_bucket_matches_reference(cp_size, widened):
    """The DCP forward (CACHE_ONLY per rank + NEW_TOKENS_ONLY, LSE-combined)
    must match full causal attention with verify windows in the wide DECODE
    bucket, and each rank must write back only the new-token pages it owns.

    Global page g of a request lives on rank g % cp_size at local slot
    g // cp_size (the interleave schedule_cp.CPMetadataComputer assumes).
    Requests: a q1 decode, two verify windows (one crossing a page boundary
    onto a different rank, one with no history so every CACHE_ONLY pass
    yields lse=-inf) and a 16-token MIXED prefill.
    """
    rng = np.random.default_rng(831)
    bucket = 4
    q_lens = (1, 3, 4, 16)
    kv_lengths = (1025, 1026, 4, 1030)
    num_q_heads, head_dim, page_size = 16, 256, 256
    max_seqs = 8
    num_seqs = len(q_lens)
    total_tokens = sum(q_lens)
    global_pages = (max(kv_lengths) + page_size - 1) // page_size + 1
    pages_per_seq = (global_pages + cp_size - 1) // cp_size + 1
    layout = configs.KVLayout.SEQ_ALONG_LANE
    cache_shape = wrapper.get_kv_cache_shape(
        max_seqs * pages_per_seq + 2,
        page_size,
        1,
        head_dim,
        jnp.float8_e4m3fn,
        kv_layout=layout,
    )

    def random_array(shape, dtype):
        return (rng.standard_normal(shape) * 0.5).astype(dtype)

    q = random_array((total_tokens, num_q_heads, head_dim), jnp.bfloat16)
    k = random_array((total_tokens, 1, head_dim), jnp.float8_e4m3fn)
    v = random_array((total_tokens, 1, head_dim), jnp.float8_e4m3fn)
    kv_lens = np.zeros(max_seqs, dtype=np.int32)
    kv_lens[:num_seqs] = kv_lengths
    cu_q_lens = np.full(max_seqs + 1, total_tokens, dtype=np.int32)
    cu_q_lens[: num_seqs + 1] = np.cumsum((0, *q_lens))

    # Per-rank shards: physical cache, local page table, expected cache after
    # the NEW_TOKENS_ONLY pass and the locations that must stay untouched.
    before, page_tables, expected, protected = [], [], [], []
    for _ in range(cp_size):
        before.append(random_array(cache_shape, jnp.float8_e4m3fn))
        table = rng.permutation(cache_shape[0])[: max_seqs * pages_per_seq]
        page_tables.append(table.reshape(max_seqs, pages_per_seq).astype(np.int32))
        expected.append(before[-1].copy())
        protected.append(np.ones((cache_shape[0], page_size), dtype=bool))

    def physical(seq, pos):
        g = pos // page_size
        rank, slot = g % cp_size, g // cp_size
        return rank, page_tables[rank][seq, slot], pos % page_size

    reference = np.empty(q.shape, dtype=np.float32)
    for seq, (q_len, kv_len) in enumerate(zip(q_lens, kv_lengths)):
        start, end = cu_q_lens[seq : seq + 2]
        history_len = kv_len - q_len
        tail = kv_len % page_size
        if tail:
            rank, page, _ = physical(seq, kv_len)
            protected[rank][page, tail:] = False
        for token, pos in enumerate(range(history_len, kv_len), start):
            rank, page, lane = physical(seq, pos)
            expected[rank][page, :, :, :, lane] = np.stack(
                (k[token, 0], v[token, 0])
            ).reshape(2, head_dim // 4, 4)
        logical_kv = np.stack(
            [
                expected[rank][page, :, :, :, lane].reshape(2, head_dim)
                for rank, page, lane in (physical(seq, pos) for pos in range(kv_len))
            ]
        ).astype(np.float32)
        scores = (
            np.einsum("qhd,kd->qhk", q[start:end].astype(np.float32), logical_kv[:, 0])
            * head_dim**-0.5
        )
        visible = np.arange(kv_len)[None, :] <= history_len + np.arange(q_len)[:, None]
        scores = np.where(visible[:, None, :], scores, -np.inf)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        reference[start:end] = np.einsum("qhk,kd->qhd", weights, logical_kv[:, 1])

    decode_end = num_seqs - 1 if widened else 1
    distribution = jnp.asarray([decode_end, decode_end, num_seqs], jnp.int32)

    def run(rank, cache, scope):
        return jax.block_until_ready(
            wrapper.ragged_paged_attention(
                jnp.asarray(q),
                jnp.asarray(k),
                jnp.asarray(v),
                cache,
                jnp.asarray(kv_lens),
                jnp.asarray(page_tables[rank].ravel()),
                jnp.asarray(cu_q_lens),
                distribution,
                sm_scale=head_dim**-0.5,
                kv_layout=layout,
                decode_query_size=bucket if widened else 1,
                cp_group_size=cp_size,
                cp_rank=jnp.asarray([rank], jnp.int32),
                attention_scope=scope,
                return_lse=True,
            )
        )

    caches = [jnp.asarray(shard) for shard in before]
    combined = None
    for rank in range(cp_size):
        out, caches[rank], lse = run(
            rank, caches[rank], configs.AttentionScope.CACHE_ONLY
        )
        part = (
            np.asarray(out.astype(jnp.float32)),
            np.asarray(lse.astype(jnp.float32)),
        )
        combined = part if combined is None else _lse_combine(*combined, *part)
    new_parts = []
    for rank in range(cp_size):
        out, caches[rank], lse = run(
            rank, caches[rank], configs.AttentionScope.NEW_TOKENS_ONLY
        )
        new_parts.append(
            (np.asarray(out.astype(jnp.float32)), np.asarray(lse.astype(jnp.float32)))
        )
    # NEW_TOKENS_ONLY is not KV-sharded: every rank computes the same thing.
    for out, lse in new_parts[1:]:
        np.testing.assert_allclose(out, new_parts[0][0], atol=1e-6)
        np.testing.assert_allclose(lse, new_parts[0][1], atol=1e-6)
    actual, _ = _lse_combine(*combined, *new_parts[0])

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, atol=2e-3, rtol=2e-2)
    for seq in range(num_seqs):
        start, end = cu_q_lens[seq : seq + 2]
        relative_l2 = np.linalg.norm(
            actual[start:end] - reference[start:end]
        ) / np.linalg.norm(reference[start:end])
        assert relative_l2 < 0.02, f"seq {seq}: relative L2 = {relative_l2}"
    for rank in range(cp_size):
        np.testing.assert_array_equal(
            np.asarray(caches[rank]).transpose(0, 4, 1, 2, 3)[protected[rank]],
            expected[rank].transpose(0, 4, 1, 2, 3)[protected[rank]],
            err_msg=f"rank {rank} cache damaged or missing writeback",
        )
