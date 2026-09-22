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
"""Numeric parity for the SEQ_ALONG_LANE KV-cache layout.

Selected with `VLLM_KV_CACHE_LAYOUT=HND`, which drives both this fork and the
mainline batched-RPA kernel.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa_longctx import configs, wrapper

_DECODE_BLOCKS = configs.BlockSizes(
    bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8, n_buffer=3
)
_PREFILL_BLOCKS = configs.BlockSizes(
    bq_sz=1792, bq_c_sz=28, bkv_sz=256, batch_size=2, n_buffer=3
)


def _build_cache(
    kv_layout,
    target_k,
    target_v,
    kv_len,
    total_pages,
    page_size,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    page_indices,
    sm_scale,
):
    cache_shape = wrapper.get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, dtype, kv_layout=kv_layout
    )
    _, cache = wrapper.ragged_paged_attention(
        jnp.zeros((kv_len, num_q_heads, head_dim), dtype),
        jnp.asarray(target_k, dtype),
        jnp.asarray(target_v, dtype),
        jnp.zeros(cache_shape, dtype),
        jnp.array([kv_len], jnp.int32),
        page_indices,
        jnp.array([0, kv_len], jnp.int32),
        jnp.array([0, 0, 1], jnp.int32),
        sm_scale=sm_scale,
        decode_block_sizes=_DECODE_BLOCKS,
        prefill_block_sizes=_PREFILL_BLOCKS,
        kv_layout=kv_layout,
    )
    return cache


@pytest.mark.parametrize(
    "kv_len, q_len, distribution",
    [
        (32, 16, (0, 0, 1)),
        (33, 1, (1, 1, 1)),
        (4096, 1, (1, 1, 1)),
    ],
)
def test_seq_along_lane_matches_head_along_sublane(kv_len, q_len, distribution):
    rng = np.random.default_rng(0)
    dtype = jnp.bfloat16
    head_dim = 128
    num_kv_heads, num_q_heads = 2, 4
    page_size, total_pages = 128, max(8, -(-kv_len // 16) + 1)
    pages_per_seq = total_pages
    sm_scale = head_dim**-0.5
    page_indices = jnp.arange(pages_per_seq, dtype=jnp.int32)

    def r(*shape):
        return (rng.standard_normal(shape) * 0.5).astype(np.float32)

    target_k, target_v = (
        r(kv_len, num_kv_heads, head_dim),
        r(kv_len, num_kv_heads, head_dim),
    )
    query = r(q_len, num_q_heads, head_dim)
    new_k, new_v = target_k[kv_len - q_len : kv_len], target_v[kv_len - q_len : kv_len]

    kv_lens = jnp.array([kv_len], jnp.int32)
    cu_q_lens = jnp.array([0, q_len], jnp.int32)
    distribution = jnp.asarray(distribution, jnp.int32)

    def run(kv_layout):
        cache = _build_cache(
            kv_layout,
            target_k,
            target_v,
            kv_len,
            total_pages,
            page_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            dtype,
            page_indices,
            sm_scale,
        )
        out, _ = wrapper.ragged_paged_attention(
            jnp.asarray(query, dtype),
            jnp.asarray(new_k, dtype),
            jnp.asarray(new_v, dtype),
            cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            distribution,
            sm_scale=sm_scale,
            decode_block_sizes=_DECODE_BLOCKS,
            prefill_block_sizes=_PREFILL_BLOCKS,
            kv_layout=kv_layout,
        )
        return np.asarray(out.astype(jnp.float32))

    out_sublane = run(configs.KVLayout.HEAD_ALONG_SUBLANE)
    out_lane = run(configs.KVLayout.SEQ_ALONG_LANE)

    np.testing.assert_allclose(out_lane, out_sublane, atol=2e-2, rtol=0)
    assert np.isfinite(out_lane).all()


@pytest.mark.parametrize("kv_layout", list(configs.KVLayout))
@pytest.mark.parametrize("q_heads_per_kv", [3, 10])
def test_return_lse_preserves_gqa_heads(kv_layout, q_heads_per_kv):
    """LSE writeback preserves GQA head mapping and logical output shapes.

    Exercise multiple KV heads, small and non-power-of-two Q groups, and
    unequal request lengths through both decode and prefill scheduling.
    """
    rng = np.random.default_rng(42)
    q_lens = (1, 3, 2)
    starts = np.cumsum((0, *q_lens)).astype(np.int32)
    num_kv_heads, head_dim = 2, 128
    num_q_heads = num_kv_heads * q_heads_per_kv
    num_tokens = sum(q_lens)

    def random_array(shape, dtype):
        return jnp.asarray(rng.normal(0, 0.5, shape), dtype)

    query = random_array((num_tokens, num_q_heads, head_dim), jnp.bfloat16)
    key = random_array((num_tokens, num_kv_heads, head_dim), jnp.float8_e4m3fn)
    value = random_array((num_tokens, num_kv_heads, head_dim), jnp.float8_e4m3fn)
    # Reserve two pages per request for the default 256-token KV block.
    shape = wrapper.get_kv_cache_shape(
        6, 128, num_kv_heads, head_dim, key.dtype, kv_layout=kv_layout
    )

    def run(return_lse):
        # LONGCTX routes prefill through the MIXED segment, as in the
        # existing parity cases above; its standalone PREFILL pass is unused.
        return wrapper.ragged_paged_attention(
            query,
            key,
            value,
            jnp.zeros(shape, key.dtype),
            jnp.asarray(q_lens, jnp.int32),
            jnp.array([4, 0, 2, 5, 1, 3], jnp.int32),
            jnp.asarray(starts),
            jnp.array([1, 1, 3], jnp.int32),
            sm_scale=head_dim**-0.5,
            kv_layout=kv_layout,
            return_lse=return_lse,
        )

    out_without_lse, cache_without_lse = run(False)
    output, cache, lse = run(True)
    np.testing.assert_allclose(
        np.asarray(output, np.float32),
        np.asarray(out_without_lse, np.float32),
        atol=3e-3,
        rtol=1e-2,
    )
    np.testing.assert_array_equal(
        np.asarray(cache, np.float32), np.asarray(cache_without_lse, np.float32)
    )

    queries = np.asarray(query, np.float32)
    keys = np.repeat(np.asarray(key, np.float32), q_heads_per_kv, axis=1)
    values = np.repeat(np.asarray(value, np.float32), q_heads_per_kv, axis=1)
    expected_out, expected_lse = [], []
    for start, end in zip(starts[:-1], starts[1:]):
        for pos in range(start, end):
            scores = (
                np.einsum("hd,thd->ht", queries[pos], keys[start : pos + 1])
                * head_dim**-0.5
            )
            maximum = scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores - maximum)
            denominator = weights.sum(axis=-1, keepdims=True)
            expected_out.append(
                np.einsum("ht,thd->hd", weights / denominator, values[start : pos + 1])
            )
            expected_lse.append((maximum + np.log(denominator))[:, 0])
    assert output.shape == query.shape
    assert lse.shape == query.shape[:2]
    np.testing.assert_allclose(
        np.asarray(output, np.float32), expected_out, atol=3e-3, rtol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(lse, np.float32), expected_lse, atol=4e-2, rtol=1e-2
    )
