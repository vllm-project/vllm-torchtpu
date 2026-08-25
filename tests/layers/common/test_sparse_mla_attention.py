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
"""Functional tests for the GLM-5.2 sparse (DSA) MLA attention path.

The kernel is validated against a pure-numpy MLA reference: `topk_indices`
are synthesized (causal or random subsets), so the reference knows exactly
which KV entries each token attends to and computes softmax(q @ k^T) @ v in
float32 over the same fp8-rounded latents the kernel reads. With causal
arange indices and kv_len <= topk this is exactly dense MLA attention, which
makes the first cases a dense-parity check of the full path
(insert -> SparseCore gather -> attention kernel).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu.kernels.mla.sparse import dsa_gather
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.common import attention_interface, utils

LKV_DIM = 512
ROPE_DIM = 64
NUM_HEADS = 16
PAGE_SIZE = 32
PAGES_PER_SEQ = 4
TOTAL_PAGES = 16
TOKEN_PAD = 16  # attention kernel batch size; token count must be a multiple


def _empty_pair():
    """Zeroed uint8 (nope, rope) caches in dsa_gather's native tiled layouts."""
    nope_shape, rope_shape = (sparse_mla_kernel.get_kv_cache_shape(
        TOTAL_PAGES, PAGE_SIZE, dim, jnp.float8_e4m3fn)
                              for dim in (LKV_DIM, ROPE_DIM))
    blocks, ps_div, kv_packing, nope_dim = nope_shape
    lane = dsa_gather.TILE_LANE_BYTES
    return (jnp.zeros((blocks, ps_div * kv_packing, nope_dim // lane, lane),
                      jnp.uint8), jnp.zeros(rope_shape, jnp.uint8))


def _quantize_fp8(x: np.ndarray, k_scale: float) -> jax.Array:
    return jnp.asarray(x / k_scale).astype(jnp.float8_e4m3fn)


def _dequantize(x: jax.Array, k_scale: float) -> np.ndarray:
    return np.asarray(x.astype(jnp.float32)) * k_scale


def _causal_topk(positions: list[int], topk: int) -> np.ndarray:
    """topk_indices row per token: [0..pos] then -1 padding."""
    rows = np.full((len(positions), topk), -1, np.int32)
    for i, pos in enumerate(positions):
        assert pos + 1 <= topk
        rows[i, :pos + 1] = np.arange(pos + 1)
    return rows


class SparseMlaAttentionTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(1234)
        self.k_scale = 1.5
        self.sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)

    def _require_tpu(self):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA kernel requires a TPU (SparseCore).")

    def _make_mesh(self):
        return jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x", ))

    def _random_latents(self, n):
        kv_c = self.rng.standard_normal((n, LKV_DIM)).astype(np.float32)
        k_pe = self.rng.standard_normal((n, ROPE_DIM)).astype(np.float32)
        return kv_c, k_pe

    def _random_queries(self, num_tokens):
        ql_nope = jnp.asarray(self.rng.standard_normal(
            (num_tokens, NUM_HEADS, LKV_DIM)).astype(np.float32),
                              dtype=jnp.bfloat16)
        q_pe = jnp.asarray(self.rng.standard_normal(
            (num_tokens, NUM_HEADS, ROPE_DIM)).astype(np.float32),
                           dtype=jnp.bfloat16)
        return ql_nope, q_pe

    def _block_tables(self, num_seqs):
        """Distinct physical pages per sequence, permuted."""
        perm = self.rng.permutation(TOTAL_PAGES)[:num_seqs * PAGES_PER_SEQ]
        return jnp.asarray(perm.reshape(num_seqs, PAGES_PER_SEQ).reshape(-1),
                           dtype=jnp.int32)

    def _empty_cache(self):
        return _empty_pair()

    def _call(self, kv_cache, ql_nope, q_pe, kv_c_fp8, k_pe_fp8, topk_rows,
              seq_lens, query_start_loc, distribution, block_tables, mesh):
        nope_cache, rope_cache, output = \
            attention_interface.sparse_mla_attention(
            ql_nope,
            q_pe,
            kv_c_fp8,
            k_pe_fp8,
            kv_cache[0],
            kv_cache[1],
            jnp.asarray(topk_rows, dtype=jnp.int32),
            jnp.asarray(seq_lens, dtype=jnp.int32),
            block_tables,
            jnp.asarray(query_start_loc, dtype=jnp.int32),
            jnp.asarray(distribution, dtype=jnp.int32),
            mesh,
            sm_scale=self.sm_scale,
            k_scale=self.k_scale,
        )
        return (nope_cache, rope_cache), output

    def _reference(self, ql_nope, q_pe, topk_rows, q_lens, query_start_loc,
                   kv_c_deq, k_pe_deq):
        """f32 MLA attention over exactly the selected (dequantized) KVs.

        kv_c_deq/k_pe_deq: per-sequence arrays [seq_len, dim] of the values
        the cache actually holds after fp8 rounding.
        """
        num_tokens = ql_nope.shape[0]
        q = np.concatenate([
            np.asarray(ql_nope.astype(jnp.float32)),
            np.asarray(q_pe.astype(jnp.float32))
        ], -1)  # [T, N, 576]
        out = np.zeros((num_tokens, NUM_HEADS, LKV_DIM), np.float32)
        mask = np.zeros(num_tokens, bool)
        for s in range(len(q_lens)):
            for local in range(q_lens[s]):
                t = query_start_loc[s] + local
                sel = topk_rows[t][topk_rows[t] >= 0]
                keys = np.concatenate([kv_c_deq[s][sel], k_pe_deq[s][sel]],
                                      -1)  # [n, 576]
                scores = q[t] @ keys.T * self.sm_scale  # [N, n]
                scores -= scores.max(-1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(-1, keepdims=True)
                out[t] = probs @ kv_c_deq[s][sel]
                mask[t] = True
        return out, mask

    def _check(self, output, expected, valid_mask):
        got = np.asarray(output.astype(jnp.float32))[valid_mask]
        np.testing.assert_allclose(got,
                                   expected[valid_mask],
                                   rtol=2e-2,
                                   atol=2e-2)

    @parameterized.named_parameters(
        dict(testcase_name="single_prefill", q_lens=[48], topk=64),
        dict(testcase_name="mixed_prefill", q_lens=[40, 24], topk=64),
        dict(testcase_name="padded_tokens", q_lens=[20, 16], topk=64),
    )
    def test_causal_matches_dense_reference(self, q_lens, topk):
        """Causal-arange topk == dense MLA; validates the full path."""
        self._require_tpu()
        num_seqs = len(q_lens)
        total = sum(q_lens)
        num_tokens = math.ceil(total / TOKEN_PAD) * TOKEN_PAD
        query_start_loc = np.concatenate([[0], np.cumsum(q_lens)])

        kv_c_deq, k_pe_deq = [], []
        kv_c_rows, k_pe_rows, positions = [], [], []
        for s, q_len in enumerate(q_lens):
            kv_c, k_pe = self._random_latents(q_len)
            kv_c_fp8 = _quantize_fp8(kv_c, self.k_scale)
            k_pe_fp8 = _quantize_fp8(k_pe, self.k_scale)
            kv_c_rows.append(kv_c_fp8)
            k_pe_rows.append(k_pe_fp8)
            kv_c_deq.append(_dequantize(kv_c_fp8, self.k_scale))
            k_pe_deq.append(_dequantize(k_pe_fp8, self.k_scale))
            positions.extend(range(q_len))

        pad = num_tokens - total
        kv_c_fp8 = jnp.concatenate(
            kv_c_rows + [jnp.zeros((pad, LKV_DIM), jnp.float8_e4m3fn)])
        k_pe_fp8 = jnp.concatenate(
            k_pe_rows + [jnp.zeros((pad, ROPE_DIM), jnp.float8_e4m3fn)])
        # Padding tokens still need >= 1 valid topk entry (kernel contract).
        topk_rows = np.concatenate(
            [_causal_topk(positions, topk),
             _causal_topk([0] * pad, topk)])

        ql_nope, q_pe = self._random_queries(num_tokens)
        mesh = self._make_mesh()
        _, output = self._call(self._empty_cache(), ql_nope, q_pe, kv_c_fp8,
                               k_pe_fp8, topk_rows, q_lens, query_start_loc,
                               [0, num_seqs, num_seqs],
                               self._block_tables(num_seqs), mesh)

        expected, valid = self._reference(ql_nope, q_pe, topk_rows, q_lens,
                                          query_start_loc, kv_c_deq, k_pe_deq)
        self._check(output, expected, valid)

    def test_sparse_subset_selection(self):
        """Random (non-causal) subsets: topk selection is actually honored."""
        self._require_tpu()
        q_len, topk = 32, 16
        kv_c, k_pe = self._random_latents(q_len)
        kv_c_fp8 = _quantize_fp8(kv_c, self.k_scale)
        k_pe_fp8 = _quantize_fp8(k_pe, self.k_scale)

        topk_rows = np.full((q_len, topk), -1, np.int32)
        for pos in range(q_len):
            n_sel = min(pos + 1, 8)
            sel = self.rng.choice(pos + 1, size=n_sel, replace=False)
            sel[0] = pos  # always include self
            topk_rows[pos, :n_sel] = np.sort(sel)

        ql_nope, q_pe = self._random_queries(q_len)
        mesh = self._make_mesh()
        _, output = self._call(self._empty_cache(), ql_nope, q_pe, kv_c_fp8,
                               k_pe_fp8, topk_rows, [q_len], [0, q_len],
                               [0, 1, 1], self._block_tables(1), mesh)

        expected, valid = self._reference(
            ql_nope, q_pe, topk_rows, [q_len], [0, q_len],
            [_dequantize(kv_c_fp8, self.k_scale)],
            [_dequantize(k_pe_fp8, self.k_scale)])
        self._check(output, expected, valid)

    def test_prefill_then_decode(self):
        """Two steps: decode gathers KVs inserted by an earlier call."""
        self._require_tpu()
        prior_lens, topk = [24, 40], 128
        num_seqs = len(prior_lens)
        mesh = self._make_mesh()
        block_tables = self._block_tables(num_seqs)

        kv_c_deq, k_pe_deq = [], []
        kv_cache = self._empty_cache()

        # Step 1: prefill both sequences (output ignored).
        kv_c_rows, k_pe_rows, positions = [], [], []
        for q_len in prior_lens:
            kv_c, k_pe = self._random_latents(q_len)
            kv_c_rows.append(_quantize_fp8(kv_c, self.k_scale))
            k_pe_rows.append(_quantize_fp8(k_pe, self.k_scale))
            positions.extend(range(q_len))
        total = sum(prior_lens)
        query_start_loc = np.concatenate([[0], np.cumsum(prior_lens)])
        ql_nope, q_pe = self._random_queries(total)
        kv_cache, _ = self._call(kv_cache, ql_nope, q_pe,
                                 jnp.concatenate(kv_c_rows),
                                 jnp.concatenate(k_pe_rows),
                                 _causal_topk(positions, topk), prior_lens,
                                 query_start_loc, [0, num_seqs, num_seqs],
                                 block_tables, mesh)
        for s in range(num_seqs):
            kv_c_deq.append(_dequantize(kv_c_rows[s], self.k_scale))
            k_pe_deq.append(_dequantize(k_pe_rows[s], self.k_scale))

        # Step 2: one decode token per sequence, attending causally over the
        # prefilled context plus itself.
        num_tokens = TOKEN_PAD
        kv_c_new, k_pe_new = self._random_latents(num_seqs)
        kv_c_fp8 = _quantize_fp8(kv_c_new, self.k_scale)
        k_pe_fp8 = _quantize_fp8(k_pe_new, self.k_scale)
        pad = num_tokens - num_seqs
        topk_rows = np.concatenate([
            _causal_topk(prior_lens, topk),  # decode pos == prior_len
            _causal_topk([0] * pad, topk),
        ])
        ql_nope, q_pe = self._random_queries(num_tokens)
        seq_lens = [n + 1 for n in prior_lens]
        _, output = self._call(
            kv_cache, ql_nope, q_pe,
            jnp.concatenate(
                [kv_c_fp8,
                 jnp.zeros((pad, LKV_DIM), jnp.float8_e4m3fn)]),
            jnp.concatenate(
                [k_pe_fp8,
                 jnp.zeros((pad, ROPE_DIM),
                           jnp.float8_e4m3fn)]), topk_rows, seq_lens,
            list(range(num_seqs + 1)), [num_seqs, num_seqs, num_seqs],
            block_tables, mesh)

        for s in range(num_seqs):
            kv_c_deq[s] = np.concatenate(
                [kv_c_deq[s],
                 _dequantize(kv_c_fp8[s:s + 1], self.k_scale)])
            k_pe_deq[s] = np.concatenate(
                [k_pe_deq[s],
                 _dequantize(k_pe_fp8[s:s + 1], self.k_scale)])
        expected, valid = self._reference(ql_nope, q_pe, topk_rows,
                                          [1] * num_seqs,
                                          list(range(num_seqs + 1)), kv_c_deq,
                                          k_pe_deq)
        self._check(output, expected, valid)


class UpdateSparseMLAKvCacheTest(parameterized.TestCase):
    """Insert-layout unit test; pure jnp scatter, runs on CPU too."""

    def test_rows_land_at_page_slot(self):
        rng = np.random.default_rng(7)
        q_lens, seq_lens = [5, 3], [37, 3]  # seq 0 appends across a page edge
        total = sum(q_lens)
        kv_c = _quantize_fp8(
            rng.standard_normal((total, LKV_DIM)).astype(np.float32), 1.0)
        k_pe = _quantize_fp8(
            rng.standard_normal((total, ROPE_DIM)).astype(np.float32), 1.0)
        block_tables = np.full((2, PAGES_PER_SEQ), 7, np.int32)
        block_tables[0, :2] = [5, 2]
        block_tables[1, 0] = 9

        nope_cache, rope_cache = utils.update_sparse_mla_kv_cache(
            *_empty_pair(), kv_c, k_pe, jnp.asarray(seq_lens, jnp.int32),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray([0, 5, 8], jnp.int32))

        nope_rows = np.asarray(
            jax.lax.bitcast_convert_type(nope_cache, jnp.uint8)).reshape(
                TOTAL_PAGES, PAGE_SIZE, LKV_DIM)
        rope_rows = np.asarray(
            jax.lax.bitcast_convert_type(rope_cache, jnp.uint8)).reshape(
                TOTAL_PAGES, PAGE_SIZE, rope_cache.shape[-1])
        exp_nope = np.asarray(jax.lax.bitcast_convert_type(kv_c, jnp.uint8))
        exp_rope = np.asarray(jax.lax.bitcast_convert_type(k_pe, jnp.uint8))
        # seq 0: positions 32..36 -> page 2 slots 0..4; seq 1: page 9 slots 0..2.
        placements = [(2, i, i) for i in range(5)]
        placements += [(9, i, 5 + i) for i in range(3)]
        touched = np.zeros((TOTAL_PAGES, PAGE_SIZE), bool)
        for page, slot, token in placements:
            np.testing.assert_array_equal(nope_rows[page, slot],
                                          exp_nope[token])
            np.testing.assert_array_equal(rope_rows[page, slot, :ROPE_DIM],
                                          exp_rope[token])
            # Pad tail is never written (zero fill).
            self.assertEqual(
                int(np.count_nonzero(rope_rows[page, slot, ROPE_DIM:])), 0)
            touched[page, slot] = True
        self.assertEqual(int(np.count_nonzero(nope_rows[~touched])), 0)
        self.assertEqual(int(np.count_nonzero(rope_rows[~touched])), 0)

    def test_padded_batch_is_dropped(self):
        """Padded tokens (valid=False) must be dropped, never written."""
        rng = np.random.default_rng(42)
        total_tokens, valid_tokens = 16, 6
        kv_c = _quantize_fp8(
            rng.standard_normal((total_tokens, LKV_DIM)).astype(np.float32),
            1.0)
        k_pe = _quantize_fp8(
            rng.standard_normal((total_tokens, ROPE_DIM)).astype(np.float32),
            1.0)
        block_tables = np.full((2, PAGES_PER_SEQ), 7, np.int32)
        block_tables[0, 0] = 1
        block_tables[1, 0] = 2

        nope_cache, rope_cache = utils.update_sparse_mla_kv_cache(
            *_empty_pair(), kv_c, k_pe, jnp.asarray([4, 2], jnp.int32),
            jnp.asarray(block_tables.reshape(-1)),
            jnp.asarray([0, 4, valid_tokens], jnp.int32))

        nope_rows = np.asarray(
            jax.lax.bitcast_convert_type(nope_cache,
                                         jnp.uint8)).reshape(-1, LKV_DIM)
        rope_rows = np.asarray(
            jax.lax.bitcast_convert_type(rope_cache, jnp.uint8)).reshape(
                -1, rope_cache.shape[-1])
        # Exactly the valid tokens landed, nothing else.
        self.assertEqual(int(nope_rows.any(axis=1).sum()), valid_tokens)
        self.assertEqual(int(rope_rows.any(axis=1).sum()), valid_tokens)
