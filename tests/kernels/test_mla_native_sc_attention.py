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
"""Standalone sparse-attention tests for both physical cache layouts."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout

NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM


def _build_problem(rng, *, k_scale, query_tokens=128, topk=256):
    page_size = 256
    cache_tokens = 8192
    pages = cache_tokens // page_size
    num_heads = 8

    def quantized_bytes(dim):
        values = (rng.random(
            (pages, page_size, dim), dtype=np.float32) * 2 - 1)
        fp8 = jnp.asarray(values / k_scale).astype(jnp.float8_e4m3fn)
        return np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))

    nope_bytes = quantized_bytes(NOPE_DIM)
    rope_bytes = quantized_bytes(ROPE_DIM)
    tensorcore_nope = nope_bytes.reshape(pages, page_size, 4, 128)
    tensorcore_rope = np.pad(rope_bytes,
                             ((0, 0), (0, 0),
                              (0, ROPE_DIM))).reshape(pages, page_size // 4, 4,
                                                      128)

    page_table = rng.permutation(pages).astype(np.int32)
    inverse_page_table = np.argsort(page_table)
    physical = np.stack([
        rng.choice(cache_tokens, size=topk, replace=False)
        for _ in range(query_tokens)
    ]).astype(np.int32)
    logical = (inverse_page_table[physical // page_size] * page_size +
               physical % page_size).astype(np.int32)
    q = jnp.asarray(rng.normal(size=(query_tokens, num_heads,
                                     HEAD_DIM)).astype(np.float32),
                    dtype=jnp.bfloat16)
    return {
        "q":
        q,
        "tensorcore_nope":
        jnp.asarray(tensorcore_nope),
        "tensorcore_rope":
        jnp.asarray(tensorcore_rope),
        "sparsecore_nope":
        jnp.asarray(native_sc_cache_layout.pack_nope(tensorcore_nope)),
        "sparsecore_rope":
        jnp.asarray(native_sc_cache_layout.pack_rope_banded(tensorcore_rope)),
        "topk":
        jnp.asarray(logical),
        "physical_topk":
        jnp.asarray(physical),
        "page_table":
        jnp.asarray(page_table),
        "cu_q_lens":
        jnp.array([0, query_tokens], jnp.int32),
        "distribution":
        jnp.array([0, 1, 1], jnp.int32),
    }


class SparseMlaCacheLayoutsTest(parameterized.TestCase):

    @parameterized.product(k_scale=(0.75, 1.0), return_lse=(False, True))
    def test_sparsecore_layout_matches_tensorcore_layout(
            self, k_scale, return_lse):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore cache-layout attention requires a TPU")
        problem = _build_problem(np.random.default_rng(20260830),
                                 k_scale=k_scale)
        kwargs = {
            "return_lse": return_lse,
            "sm_scale": 0.125,
            "k_scale": k_scale,
            "gather_and_attention_chunk_size": 128,
            "attention_kernel_batch_size": 16,
        }

        tensorcore_out = sparse_mla.sparse_ragged_paged_attention(
            problem["q"],
            problem["tensorcore_nope"],
            problem["tensorcore_rope"],
            problem["topk"],
            problem["page_table"],
            problem["cu_q_lens"],
            problem["distribution"],
            **kwargs,
        )
        sparsecore_out = sparse_mla.sparse_ragged_paged_attention(
            problem["q"],
            problem["sparsecore_nope"],
            problem["sparsecore_rope"],
            problem["topk"],
            problem["page_table"],
            problem["cu_q_lens"],
            problem["distribution"],
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            sparsecore_atoms_per_batch=16,
            **kwargs,
        )
        tensorcore_out, sparsecore_out = jax.block_until_ready(
            (tensorcore_out, sparsecore_out))

        if return_lse:
            tensorcore_out, tensorcore_lse = tensorcore_out
            sparsecore_out, sparsecore_lse = sparsecore_out
            np.testing.assert_allclose(np.asarray(sparsecore_lse),
                                       np.asarray(tensorcore_lse),
                                       rtol=2e-2,
                                       atol=2e-2)

        np.testing.assert_allclose(np.asarray(sparsecore_out,
                                              dtype=np.float32),
                                   np.asarray(tensorcore_out,
                                              dtype=np.float32),
                                   rtol=2e-2,
                                   atol=2e-2)

    def test_cache_layouts_support_physical_indices_and_padding(self):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore cache-layout attention requires a TPU")
        problem = _build_problem(np.random.default_rng(20260831), k_scale=1.0)
        logical_topk = problem["topk"].at[:, 200:].set(-1)
        physical_topk = problem["physical_topk"].at[:, 200:].set(-1)
        kwargs = {
            "sm_scale": 0.125,
            "gather_and_attention_chunk_size": 128,
            "attention_kernel_batch_size": 16,
        }

        paged_out = sparse_mla.sparse_ragged_paged_attention(
            problem["q"], problem["tensorcore_nope"],
            problem["tensorcore_rope"], logical_topk, problem["page_table"],
            problem["cu_q_lens"], problem["distribution"], **kwargs)
        tensorcore_physical_out = sparse_mla.sparse_ragged_paged_attention(
            problem["q"],
            problem["tensorcore_nope"],
            problem["tensorcore_rope"],
            physical_topk,
            None,
            problem["cu_q_lens"],
            problem["distribution"],
            **kwargs,
        )
        sparsecore_physical_out = sparse_mla.sparse_ragged_paged_attention(
            problem["q"],
            problem["sparsecore_nope"],
            problem["sparsecore_rope"],
            physical_topk,
            None,
            problem["cu_q_lens"],
            problem["distribution"],
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            **kwargs,
        )
        paged_out, tensorcore_physical_out, sparsecore_physical_out = (
            jax.block_until_ready(
                (paged_out, tensorcore_physical_out, sparsecore_physical_out)))

        for physical_out in (tensorcore_physical_out, sparsecore_physical_out):
            np.testing.assert_allclose(np.asarray(physical_out,
                                                  dtype=np.float32),
                                       np.asarray(paged_out, dtype=np.float32),
                                       rtol=2e-2,
                                       atol=2e-2)

    def test_sparsecore_layout_supports_short_query_chunk(self):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore cache-layout attention requires a TPU")
        problem = _build_problem(np.random.default_rng(20260903),
                                 k_scale=1.0,
                                 query_tokens=16,
                                 topk=2048)
        kwargs = {
            "sm_scale": 0.125,
            "gather_and_attention_chunk_size": 64,
            "attention_kernel_batch_size": 16,
        }

        tensorcore_out = sparse_mla.sparse_ragged_paged_attention(
            problem["q"], problem["tensorcore_nope"],
            problem["tensorcore_rope"], problem["topk"], problem["page_table"],
            problem["cu_q_lens"], problem["distribution"], **kwargs)
        sparsecore_out = sparse_mla.sparse_ragged_paged_attention(
            problem["q"],
            problem["sparsecore_nope"],
            problem["sparsecore_rope"],
            problem["topk"],
            problem["page_table"],
            problem["cu_q_lens"],
            problem["distribution"],
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            **kwargs,
        )
        tensorcore_out, sparsecore_out = jax.block_until_ready(
            (tensorcore_out, sparsecore_out))

        np.testing.assert_allclose(np.asarray(sparsecore_out,
                                              dtype=np.float32),
                                   np.asarray(tensorcore_out,
                                              dtype=np.float32),
                                   rtol=2e-2,
                                   atol=2e-2)

    def test_sparsecore_layout_validation(self):
        q = jnp.zeros((128, 8, HEAD_DIM), jnp.bfloat16)
        nope = jnp.zeros((16, 256, 128), jnp.uint32)
        rope = jnp.zeros((16, 64, 128), jnp.uint32)
        indices = jnp.zeros((128, 256), jnp.int32)
        cu_q_lens = jnp.array([0, 128], jnp.int32)
        distribution = jnp.array([0, 1, 1], jnp.int32)

        with self.assertRaisesRegex(ValueError, "sparsecore NOPE"):
            sparse_mla.sparse_ragged_paged_attention(
                q,
                nope.astype(jnp.uint8),
                rope,
                indices,
                None,
                cu_q_lens,
                distribution,
                cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            )
        with self.assertRaisesRegex(ValueError, "sparsecore ROPE"):
            sparse_mla.sparse_ragged_paged_attention(
                q,
                nope,
                rope.astype(jnp.uint8),
                indices,
                None,
                cu_q_lens,
                distribution,
                cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            )
        with self.assertRaisesRegex(ValueError, "4 \\* ROPE cache size"):
            sparse_mla.sparse_ragged_paged_attention(
                q,
                nope[:8],
                rope,
                indices,
                None,
                cu_q_lens,
                distribution,
                cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            )
        with self.assertRaisesRegex(ValueError, "multiple of 4"):
            sparse_mla.sparse_ragged_paged_attention(
                q,
                nope,
                rope,
                indices[:, :-1],
                None,
                cu_q_lens,
                distribution,
                cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            )


if __name__ == "__main__":
    absltest.main()
