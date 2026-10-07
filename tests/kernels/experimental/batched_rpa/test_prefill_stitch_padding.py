# SPDX-License-Identifier: Apache-2.0
"""The O(N) prefill stitch must not carry step padding into attention or HBM.

Under SEQ_ALONG_LANE the new-KV fetch is page-granular, so the last page of the
last request also holds the step's padded rows. Lanes past a request's last new
token are causally masked, but 0 * NaN in p @ V is still NaN, and the
whole-page writeback lands those lanes in the HBM page tail.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_LAYOUT = configs.KVLayout.SEQ_ALONG_LANE
_HEAD_DIM = 128
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 1
_DTYPE = jnp.bfloat16
_MAX_SEQS = 4
_SM_SCALE = _HEAD_DIM**-0.5


def _reference(q, k, v, before, page_table, page_size, histories, q_lens):
    """Causal attention in float32 over each request's history and new tokens."""
    q, k, v, before = (np.asarray(x, np.float32) for x in (q, k, v, before))
    group = _NUM_Q_HEADS // _NUM_KV_HEADS
    out = []
    start = 0
    for seq, (hist, q_len) in enumerate(zip(histories, q_lens)):
        # K and V interleave per head along axis 1 of the cache.
        cached = np.stack(
            [
                before[page_table[seq, pos // page_size], ..., pos % page_size]
                for pos in range(hist)
            ]
        ).reshape(hist, 2 * _NUM_KV_HEADS, _HEAD_DIM)
        keys = np.concatenate([cached[:, 0::2], k[start : start + q_len]])
        values = np.concatenate([cached[:, 1::2], v[start : start + q_len]])
        keys = np.repeat(keys, group, axis=1)
        values = np.repeat(values, group, axis=1)
        scores = np.einsum("qhd,khd->hqk", q[start : start + q_len], keys)
        scores *= _SM_SCALE
        causal = np.arange(hist + q_len)[None, :] <= hist + np.arange(q_len)[:, None]
        scores = np.where(causal, scores, -np.inf)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        out.append(np.einsum("hqk,khd->qhd", probs, values))
        start += q_len
    return np.concatenate(out)


@pytest.mark.parametrize("page_size", [128, 256])
def test_prefill_padding_does_not_poison_attention(page_size):
    """NaN rows after the last request's final token must be zeroed, not kept.

    An all-MIXED distribution keeps bq_sz at bkv_sz (> num_lanes), so the
    stitch takes the O(N) roll. Both requests end mid-page in the new-KV array,
    and the first one's last fetched page also holds the second's tokens.
    """
    rng = np.random.default_rng(13)
    histories = (200, 130)
    q_lens = (300, 77)
    kv_lengths = tuple(h + q for h, q in zip(histories, q_lens))
    num_seqs = len(q_lens)
    total_tokens = sum(q_lens)
    padded_tokens = total_tokens + page_size
    pages_per_seq = (max(kv_lengths) + page_size - 1) // page_size + 1
    cache_shape = wrapper.get_kv_cache_shape(
        _MAX_SEQS * pages_per_seq + 2,
        page_size,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _DTYPE,
        kv_layout=_LAYOUT,
    )

    def padded(heads):
        out = np.full((padded_tokens, heads, _HEAD_DIM), np.nan)
        out[:total_tokens] = rng.standard_normal((total_tokens, heads, _HEAD_DIM)) * 0.5
        return out.astype(_DTYPE)

    q = padded(_NUM_Q_HEADS)
    k = padded(_NUM_KV_HEADS)
    v = padded(_NUM_KV_HEADS)
    before = (rng.standard_normal(cache_shape) * 0.5).astype(_DTYPE)
    page_table = rng.permutation(cache_shape[0])[: _MAX_SEQS * pages_per_seq]
    page_table = page_table.reshape(_MAX_SEQS, pages_per_seq).astype(np.int32)
    kv_lens = np.zeros(_MAX_SEQS, dtype=np.int32)
    kv_lens[:num_seqs] = kv_lengths
    cu_q_lens = np.full(_MAX_SEQS + 1, total_tokens, dtype=np.int32)
    cu_q_lens[: num_seqs + 1] = np.cumsum((0, *q_lens))

    out, cache = wrapper.ragged_paged_attention(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(before),
        jnp.asarray(kv_lens),
        jnp.asarray(page_table.ravel()),
        jnp.asarray(cu_q_lens),
        jnp.asarray((0, 0, num_seqs), jnp.int32),
        sm_scale=_SM_SCALE,
        kv_layout=_LAYOUT,
    )
    out, cache = jax.block_until_ready((out, cache))
    out = np.asarray(out.astype(jnp.float32))[:total_tokens]
    cache = np.asarray(cache.astype(jnp.float32))

    assert np.isfinite(out).all(), "padding leaked into p @ V"
    expected = _reference(
        q[:total_tokens],
        k[:total_tokens],
        v[:total_tokens],
        before,
        page_table,
        page_size,
        histories,
        q_lens,
    )
    np.testing.assert_allclose(out, expected, atol=3e-2, rtol=0)
    for seq, kv_len in enumerate(kv_lengths):
        page = page_table[seq, (kv_len - 1) // page_size]
        np.testing.assert_array_equal(
            cache[page, ..., (kv_len - 1) % page_size + 1 :],
            0,
            f"seq {seq}: the page tail past kv_len is not zero",
        )
