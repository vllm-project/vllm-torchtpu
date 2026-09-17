# SPDX-License-Identifier: Apache-2.0
"""The widened O(1) KV stitch must match the O(N) roll it replaces.

Buckets of 2..num_lanes moved from the prefill roll to the decode path, so new
tokens now land through a single lane rotation spanning at most two 128-lane
columns. Histories put that run before, on, and across a column boundary, at
widths from a single token up to the full bucket.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import (configs,
                                                                    wrapper)

_LAYOUT = configs.KVLayout.SEQ_ALONG_LANE
_PAGE_SIZE = 256
_HEAD_DIM = 128
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 1
_DTYPE = jnp.bfloat16
_MAX_SEQS = 8
# Cached-token counts mod 128: interior, last lane, aligned, first lane.
_HISTORIES = (895, 1020, 1023, 1024, 1025, 1151)


def _run(q, k, v, cache, kv_lens, page_table, cu_q_lens, distribution, bucket):
    out, new_cache = wrapper.ragged_paged_attention(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(cache),
        jnp.asarray(kv_lens),
        jnp.asarray(page_table.ravel()),
        jnp.asarray(cu_q_lens),
        jnp.asarray(distribution, jnp.int32),
        sm_scale=_HEAD_DIM**-0.5,
        kv_layout=_LAYOUT,
        decode_query_size=bucket)
    out, new_cache = jax.block_until_ready((out, new_cache))
    return (np.asarray(out.astype(jnp.float32)),
            np.asarray(new_cache.astype(jnp.float32)))


def _logical_kv(cache, page_table, seq, kv_len):
    """Gather a sequence's live KV out of its pages, in position order."""
    return np.stack([
        cache[page_table[seq, pos // _PAGE_SIZE], :, :, :, pos % _PAGE_SIZE]
        for pos in range(kv_len)
    ])


@pytest.mark.parametrize(("bucket", "num_kv_heads"), [(2, 1), (4, 1), (8, 1),
                                                      (16, 1), (128, 1),
                                                      (4, 2)])
def test_widened_bucket_matches_prefill_stitch(bucket, num_kv_heads):
    """Run identical KV through the widened decode stitch and the O(N) roll.

    MIXED keeps bq_sz at bkv_sz and takes the prefill formulation; DECODE with
    a bucket of 2..128 takes the widened path. Same history, same new tokens,
    so both the attention output and every live cache slot must agree. Every
    third request runs the bucket to its full width, which is the only way the
    second destination column fills.
    """
    rng = np.random.default_rng(7)
    num_seqs = len(_HISTORIES)
    q_lens = tuple(bucket if i % 3 == 2 else 1 + i % bucket
                   for i in range(num_seqs))
    kv_lengths = tuple(h + q for h, q in zip(_HISTORIES, q_lens))
    total_tokens = sum(q_lens)
    pages_per_seq = (max(kv_lengths) + _PAGE_SIZE - 1) // _PAGE_SIZE + 1
    cache_shape = wrapper.get_kv_cache_shape(_MAX_SEQS * pages_per_seq + 2,
                                             _PAGE_SIZE,
                                             num_kv_heads,
                                             _HEAD_DIM,
                                             _DTYPE,
                                             kv_layout=_LAYOUT)

    def random_array(shape, dtype):
        return (rng.standard_normal(shape) * 0.5).astype(dtype)

    q = random_array((total_tokens, _NUM_Q_HEADS, _HEAD_DIM), _DTYPE)
    k = random_array((total_tokens, num_kv_heads, _HEAD_DIM), _DTYPE)
    v = random_array((total_tokens, num_kv_heads, _HEAD_DIM), _DTYPE)
    before = random_array(cache_shape, _DTYPE)
    page_table = rng.permutation(cache_shape[0])[:_MAX_SEQS * pages_per_seq]
    page_table = page_table.reshape(_MAX_SEQS, pages_per_seq).astype(np.int32)
    kv_lens = np.zeros(_MAX_SEQS, dtype=np.int32)
    kv_lens[:num_seqs] = kv_lengths
    cu_q_lens = np.full(_MAX_SEQS + 1, total_tokens, dtype=np.int32)
    cu_q_lens[:num_seqs + 1] = np.cumsum((0, *q_lens))

    mixed_out, mixed_cache = _run(q, k, v, before, kv_lens, page_table,
                                  cu_q_lens, (0, 0, num_seqs), 1)
    decode_out, decode_cache = _run(q, k, v, before, kv_lens, page_table,
                                    cu_q_lens, (num_seqs, num_seqs, num_seqs),
                                    bucket)

    assert np.isfinite(decode_out).all()
    np.testing.assert_allclose(decode_out, mixed_out, atol=2e-2, rtol=0)

    packed_k = np.asarray(k.astype(jnp.float32)).reshape(
        total_tokens, num_kv_heads, cache_shape[2], cache_shape[3])
    packed_v = np.asarray(v.astype(jnp.float32)).reshape(
        total_tokens, num_kv_heads, cache_shape[2], cache_shape[3])
    for seq, (q_len, kv_len) in enumerate(zip(q_lens, kv_lengths)):
        live = _logical_kv(decode_cache, page_table, seq, kv_len)
        np.testing.assert_array_equal(
            live, _logical_kv(mixed_cache, page_table, seq, kv_len),
            f"seq {seq}: stitched cache diverges from the prefill roll")
        # The run itself must hold this step's tokens, not a rolled-over row.
        # K and V interleave per head: the wrapper concatenates them along
        # head_dim before splitting the axis, so row 2h is K and 2h+1 is V.
        start = cu_q_lens[seq]
        np.testing.assert_array_equal(
            live[kv_len - q_len:, 0::2], packed_k[start:start + q_len],
            f"seq {seq}: new K landed at the wrong lane")
        np.testing.assert_array_equal(
            live[kv_len - q_len:, 1::2], packed_v[start:start + q_len],
            f"seq {seq}: new V landed at the wrong lane")


@pytest.mark.parametrize("bucket", [4, 16, 128])
def test_short_run_does_not_pull_padding_into_the_page(bucket):
    """A request shorter than its bucket must zero-fill the rest of the run.

    The new-KV fetch is page-granular, so the tail of the last request's page
    is whatever HBM held past the token count -- NaN here. Sizing the run off
    the static bucket instead of the runtime length drags that padding into
    the destination page, where it poisons subsequent dot products.
    """
    rng = np.random.default_rng(11)
    num_seqs = 2
    q_lens = (bucket, 1)
    kv_lengths = (1024 + bucket, 1024 + 1)
    total_tokens = sum(q_lens)
    padded_tokens = total_tokens + _PAGE_SIZE
    pages_per_seq = (max(kv_lengths) + _PAGE_SIZE - 1) // _PAGE_SIZE + 1
    cache_shape = wrapper.get_kv_cache_shape(_MAX_SEQS * pages_per_seq + 2,
                                             _PAGE_SIZE,
                                             _NUM_KV_HEADS,
                                             _HEAD_DIM,
                                             _DTYPE,
                                             kv_layout=_LAYOUT)

    def padded(shape_tail, heads):
        out = np.full((padded_tokens, heads, *shape_tail), np.nan)
        out[:total_tokens] = rng.standard_normal(
            (total_tokens, heads, *shape_tail)) * 0.5
        return out.astype(_DTYPE)

    q = padded((_HEAD_DIM, ), _NUM_Q_HEADS)
    k = padded((_HEAD_DIM, ), _NUM_KV_HEADS)
    v = padded((_HEAD_DIM, ), _NUM_KV_HEADS)
    before = (rng.standard_normal(cache_shape) * 0.5).astype(_DTYPE)
    page_table = rng.permutation(cache_shape[0])[:_MAX_SEQS * pages_per_seq]
    page_table = page_table.reshape(_MAX_SEQS, pages_per_seq).astype(np.int32)
    kv_lens = np.zeros(_MAX_SEQS, dtype=np.int32)
    kv_lens[:num_seqs] = kv_lengths
    cu_q_lens = np.full(_MAX_SEQS + 1, total_tokens, dtype=np.int32)
    cu_q_lens[:num_seqs + 1] = np.cumsum((0, *q_lens))

    out, cache = _run(q, k, v, before, kv_lens, page_table, cu_q_lens,
                      (num_seqs, num_seqs, num_seqs), bucket)

    assert np.isfinite(out[:total_tokens]).all()
    for seq, kv_len in enumerate(kv_lengths):
        page = page_table[seq, (kv_len - 1) // _PAGE_SIZE]
        assert np.isfinite(cache[page]).all(), (
            f"seq {seq}: padding leaked into the stitched page")
