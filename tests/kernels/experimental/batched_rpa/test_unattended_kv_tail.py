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
"""Undefined bytes past a sequence's cached tokens must not reach the output.

Under SEQ_ALONG_LANE the kernel DMAs whole pages, so the tail of the last
cache page (and any page it never wrote) lands in VMEM. Those bytes are
undefined: with the unified hybrid block pool they are recycled Mamba state,
which often decodes as an fp8 NaN. Score masking alone cannot neutralize them
because 0 * NaN = NaN in the P @ V matmul, so the stitch must zero them.

The CACHE_ONLY pass used by decode context parallelism fetches no new tokens,
so every lane past the local cache is such a tail. With a prefill block
larger than one lane register (bq_sz > 128) the MIXED kernel takes the
`_stitch_prefill_lane` path, which used to leave those lanes untouched.

Each case fills every non-cached token slot with NaN and with 0, and checks
that both runs are finite, identical, and match a float32 reference.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_PAGE_SIZE = 128
_NUM_KV_HEADS = 2
_NUM_Q_HEADS = 8
_HEAD_DIM = 128
_MAX_NUM_SEQS = 8
# (q_len, kv_len). Two decode sequences followed by three MIXED sequences
# whose cache ends mid-page, exactly on a page boundary, and across several
# KV blocks. Every sequence has a non-empty cache so CACHE_ONLY is defined.
_SEQ_LENS = ((1, 300), (1, 129), (200, 457), (256, 512), (64, 1000))
_DISTRIBUTION = (2, 2, len(_SEQ_LENS))


def _run_case(scope, kv_dtype, prefill_bq_sz, fill_value):
    rng = np.random.default_rng(0)
    q_lens = np.array([q for q, _ in _SEQ_LENS], np.int32)
    kv_lens = np.array([kv for _, kv in _SEQ_LENS], np.int32)
    cache_lens = kv_lens - q_lens
    num_seqs = len(_SEQ_LENS)
    num_tokens = int(q_lens.sum())
    max_tokens = -(-num_tokens // 128) * 128
    pages_per_seq = max(-(-int(kv) // _PAGE_SIZE) for kv in kv_lens)

    # Scatter sequences over shuffled pages. Page 0 is a spare that pads the
    # page table; it holds only fill values.
    seq_pages = [-(-int(kv) // _PAGE_SIZE) for kv in kv_lens]
    total_pages = 1 + sum(seq_pages)
    perm = 1 + rng.permutation(total_pages - 1)
    page_indices = np.zeros((_MAX_NUM_SEQS, pages_per_seq), np.int32)
    start = 0
    for i, n in enumerate(seq_pages):
        page_indices[i, :n] = perm[start : start + n]
        start += n

    def rand(*shape):
        # Round through kv_dtype so the reference sees the stored values.
        x = rng.standard_normal(shape).astype(np.float32) * 0.5
        return np.asarray(jnp.asarray(x, kv_dtype).astype(jnp.float32))

    cache_shape = wrapper.get_kv_cache_shape(
        total_pages,
        _PAGE_SIZE,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        kv_dtype,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
    )
    _, kv_heads_x2, d_per_pack, packing, _ = cache_shape
    assert d_per_pack * packing == _HEAD_DIM
    cache = np.full(cache_shape, fill_value, np.float32)
    cache_k, cache_v = [], []
    for i in range(num_seqs):
        k = rand(int(cache_lens[i]), _NUM_KV_HEADS, _HEAD_DIM)
        v = rand(int(cache_lens[i]), _NUM_KV_HEADS, _HEAD_DIM)
        cache_k.append(k)
        cache_v.append(v)
        for pos in range(int(cache_lens[i])):
            page = page_indices[i, pos // _PAGE_SIZE]
            slot = pos % _PAGE_SIZE
            for h in range(_NUM_KV_HEADS):
                cache[page, 2 * h, :, :, slot] = k[pos, h].reshape(d_per_pack, packing)
                cache[page, 2 * h + 1, :, :, slot] = v[pos, h].reshape(
                    d_per_pack, packing
                )
    assert kv_heads_x2 == 2 * _NUM_KV_HEADS

    query = rng.standard_normal((max_tokens, _NUM_Q_HEADS, _HEAD_DIM)).astype(
        np.float32
    )
    query = np.asarray(jnp.asarray(query, jnp.bfloat16).astype(jnp.float32))
    new_k = rand(max_tokens, _NUM_KV_HEADS, _HEAD_DIM)
    new_v = rand(max_tokens, _NUM_KV_HEADS, _HEAD_DIM)

    cu_q_lens = np.zeros(_MAX_NUM_SEQS + 1, np.int32)
    cu_q_lens[1 : num_seqs + 1] = np.cumsum(q_lens)
    cu_q_lens[num_seqs + 1 :] = num_tokens
    padded_kv_lens = np.zeros(_MAX_NUM_SEQS, np.int32)
    padded_kv_lens[:num_seqs] = kv_lens

    output, _, lse = wrapper.ragged_paged_attention(
        jnp.asarray(query, jnp.bfloat16),
        jnp.asarray(new_k, kv_dtype),
        jnp.asarray(new_v, kv_dtype),
        jnp.asarray(cache, kv_dtype),
        jnp.asarray(padded_kv_lens),
        jnp.asarray(page_indices.reshape(-1)),
        jnp.asarray(cu_q_lens),
        jnp.asarray(_DISTRIBUTION, jnp.int32),
        sm_scale=_HEAD_DIM**-0.5,
        decode_block_sizes=configs.BlockSizes(
            bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=1, n_buffer=2
        ),
        prefill_block_sizes=configs.BlockSizes(
            bq_sz=prefill_bq_sz, bq_c_sz=32, bkv_sz=256, batch_size=1, n_buffer=2
        ),
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        attention_scope=scope,
        return_lse=True,
    )
    output = np.asarray(output[:num_tokens].astype(jnp.float32))
    lse = np.asarray(lse[:num_tokens].astype(jnp.float32))

    # float32 reference.
    group = _NUM_Q_HEADS // _NUM_KV_HEADS
    ref_out = np.zeros((num_tokens, _NUM_Q_HEADS, _HEAD_DIM), np.float32)
    ref_lse = np.zeros((num_tokens, _NUM_Q_HEADS), np.float32)
    for i in range(num_seqs):
        c = int(cache_lens[i])
        q0 = int(cu_q_lens[i])
        keys = np.concatenate([cache_k[i], new_k[q0 : q0 + q_lens[i]]])
        values = np.concatenate([cache_v[i], new_v[q0 : q0 + q_lens[i]]])
        keys = np.repeat(keys, group, axis=1)
        values = np.repeat(values, group, axis=1)
        for t in range(int(q_lens[i])):
            if scope == configs.AttentionScope.CACHE_ONLY:
                lo, hi = 0, c
            elif scope == configs.AttentionScope.NEW_TOKENS_ONLY:
                lo, hi = c, c + t + 1
            else:
                lo, hi = 0, c + t + 1
            s = np.einsum("hd,khd->hk", query[q0 + t], keys[lo:hi]) * _HEAD_DIM**-0.5
            m = s.max(axis=-1, keepdims=True)
            w = np.exp(s - m)
            denom = w.sum(axis=-1, keepdims=True)
            ref_out[q0 + t] = np.einsum("hk,khd->hd", w / denom, values[lo:hi])
            ref_lse[q0 + t] = (m + np.log(denom))[:, 0]
    return output, lse, ref_out, ref_lse


@pytest.mark.parametrize(
    "scope",
    [
        configs.AttentionScope.CACHE_ONLY,
        configs.AttentionScope.NEW_TOKENS_ONLY,
        configs.AttentionScope.FULL,
    ],
)
@pytest.mark.parametrize("kv_dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
# 128 keeps the MIXED kernel on the in-register decode stitch; 256 takes the
# full-roll prefill stitch.
@pytest.mark.parametrize("prefill_bq_sz", [128, 256])
def test_unattended_kv_tail_does_not_leak(scope, kv_dtype, prefill_bq_sz):
    if not jtu.is_device_tpu_at_least(version=5):
        pytest.skip("Expect TPUv5+")
    out_nan, lse_nan, ref_out, ref_lse = _run_case(
        scope, kv_dtype, prefill_bq_sz, np.nan
    )
    out_zero, lse_zero, _, _ = _run_case(scope, kv_dtype, prefill_bq_sz, 0.0)

    assert np.isfinite(out_zero).all()
    assert np.isfinite(out_nan).all(), (
        f"{np.isnan(out_nan).any(axis=(1, 2)).sum()} tokens have NaN outputs"
    )
    assert np.isfinite(lse_nan).all()
    np.testing.assert_array_equal(out_nan, out_zero)
    np.testing.assert_array_equal(lse_nan, lse_zero)
    np.testing.assert_allclose(out_zero, ref_out, atol=3e-2, rtol=1e-2)
    np.testing.assert_allclose(lse_zero, ref_lse, atol=4e-2, rtol=1e-2)
