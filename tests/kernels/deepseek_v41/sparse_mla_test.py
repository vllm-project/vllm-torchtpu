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
"""DeepSeek-V4.1 sparse MLA on pre-encoded top-k must match DeepSeek-V4."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla as v4_mla
from vllm_torchtpu.kernels.deepseek_v41 import sparse_mla as v41_mla


def _random(rng, shape, dtype):
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32)).astype(dtype)


def _random_caches(rng, total_pages, page_size):
    """FP8 nope and bf16 rope caches in the int32 CSA host layout."""
    fp8 = _random(rng, (total_pages, page_size, 448), jnp.float8_e4m3fn)
    scales = jnp.asarray(
        2.0 ** rng.integers(-3, 3, (total_pages, page_size, 64)), jnp.float32
    ).astype(jnp.float8_e8m0fnu)
    nope_slabs = jnp.concatenate(
        [
            jax.lax.bitcast_convert_type(fp8, jnp.uint8),
            jax.lax.bitcast_convert_type(scales, jnp.uint8),
        ],
        axis=-1,
    ).reshape(total_pages, page_size, 4, 128)
    rope = _random(rng, (total_pages * page_size, 64), jnp.bfloat16)
    return (
        csa_cache_layout.slabs_to_words(nope_slabs),
        csa_cache_layout.encode_rope(rope).reshape(total_pages, page_size // 4, 128),
    )


class SparseMlaV41Test(absltest.TestCase):
    def test_preencoded_topk_matches_v4(self):
        rng = np.random.default_rng(0)
        topk, num_heads, head_dim, page_size = 512, 64, 512, 16
        batch_size, num_decode_seqs, pages_per_seq = 12, 6, 64

        q_lens = np.concatenate(
            [
                np.ones(num_decode_seqs, np.int32),
                rng.integers(8, 60, batch_size - num_decode_seqs),
            ]
        )
        # Some rows fill all topk slots, the rest are -1 padded.
        kv_lens = q_lens + rng.integers(30, 900, batch_size)
        cu_q_lens = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32)
        num_tokens = int(q_lens.sum())
        total_pages = batch_size * pages_per_seq
        page_indices = jnp.asarray(rng.permutation(total_pages), jnp.int32)
        rows = []
        for kv_len, q_len in zip(kv_lens, q_lens, strict=True):
            for _ in range(q_len):
                picked = rng.permutation(kv_len)[:topk]
                rows.append(np.pad(picked, (0, topk - len(picked)), constant_values=-1))
        topk_indices = jnp.asarray(np.stack(rows), jnp.int32)

        cache_kv_nope, cache_kv_rope = _random_caches(rng, total_pages, page_size)
        q = _random(rng, (num_tokens, num_heads, head_dim), jnp.bfloat16)
        rest = (
            cu_q_lens,
            jnp.array([num_decode_seqs, num_decode_seqs, batch_size], jnp.int32),
            _random(rng, (num_heads,), jnp.float32),  # attention_sinks
            _random(rng, (num_tokens, num_heads, head_dim), jnp.bfloat16),
            jnp.asarray(rng.random((num_tokens, num_heads), np.float32)),  # swa_l
            _random(rng, (num_tokens, num_heads), jnp.float32),  # swa_m
        )
        kwargs = dict(
            sm_scale=1.0,
            attention_kernel_batch_size=4,
            gather_and_attention_chunk_size=32,
        )

        encoded = v41_mla.encode_csa_gather_indices(
            topk_indices, page_indices, cu_q_lens, total_pages, page_size
        )
        out_v41 = v41_mla.sparse_ragged_paged_attention(
            q, cache_kv_nope, cache_kv_rope, encoded, *rest, **kwargs
        )
        out_v4 = v4_mla.sparse_ragged_paged_attention(
            q, cache_kv_nope, cache_kv_rope, topk_indices, page_indices, *rest, **kwargs
        )
        np.testing.assert_array_equal(
            np.asarray(out_v41, np.float32), np.asarray(out_v4, np.float32)
        )


if __name__ == "__main__":
    absltest.main()
