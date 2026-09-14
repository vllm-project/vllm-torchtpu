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
"""Correctness tests for the GLM-5.2 masked-dense (DSA) MLA kernel.

The kernel streams the paged KV cache and masks the scores with the indexer's
`topk_indices`, so the reference is a plain float32 numpy attention over
exactly the KV rows each token selected -- the same reference shape
`tests/layers/core/test_sparse_mla_attention.py` uses for the gather kernel.

The caches are built through `update_sparse_mla_kv_cache`, i.e. the same insert
path the model runs, so a layout mistake in either the insert or the kernel's
row addressing shows up here rather than only on a server.

`test_matches_sparse_kernel` is the strongest gate: both kernels read the
identical cache tensors after PR #545, so they can be driven on the same inputs
and diffed directly. They are not bit-identical -- the sparse kernel does a
single-block softmax over the gathered top-k while this one runs an online
FlashAttention-2 softmax across streamed blocks -- so the comparison carries a
tolerance.
"""

import math
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout, KVCacheType, SparseMLAKVCacheSpec,
    update_sparse_mla_kv_cache, update_sparse_mla_kv_cache_jax)
from vllm_torchtpu.kernels.mla.masked_dense import kernel as masked_dense
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel

LKV_DIM = 512
ROPE_DIM = 64
NUM_HEADS = 16
PAGE_SIZE = 32
PAGES_PER_SEQ = 8
MAX_KV_LEN = PAGE_SIZE * PAGES_PER_SEQ  # 256
TOTAL_PAGES = 64
# The sparse kernel batches its attention grid in this many tokens; both arms
# of the parity test get a token count that is a multiple of it.
TOKEN_PAD = 16

KV_PACKING = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
NOPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.NOPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, LKV_DIM, KV_PACKING)
ROPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.ROPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, ROPE_DIM, KV_PACKING)


def _empty_caches():
    """Zeroed uint8 (nope, rope) caches in the native split sparse layout."""
    return (jnp.zeros(NOPE_SPEC.shape, NOPE_SPEC.jax_dtype),
            jnp.zeros(ROPE_SPEC.shape, ROPE_SPEC.jax_dtype))


def _quantize_fp8(x: np.ndarray, k_scale: float) -> jax.Array:
    return jnp.asarray(x / k_scale).astype(jnp.float8_e4m3fn)


def _dequantize(x: jax.Array, k_scale: float) -> np.ndarray:
    return np.asarray(x.astype(jnp.float32)) * k_scale


class MaskedDenseTestBase(parameterized.TestCase):
    """Cache construction, reference attention, and the kernel call."""

    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(1234)
        self.sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
        if jax.devices()[0].platform != "tpu":
            self.skipTest("masked-dense MLA kernel requires a TPU "
                          "(SparseCore builds the mask).")

    # ---------------------------------------------------------------- setup

    def _block_tables(self, num_seqs):
        """Distinct, permuted physical pages per sequence."""
        perm = self.rng.permutation(TOTAL_PAGES)[:num_seqs * PAGES_PER_SEQ]
        return jnp.asarray(perm.reshape(-1), dtype=jnp.int32)

    def _fill_cache(self, seq_lens, block_tables, k_scale):
        """Write `seq_lens[s]` random tokens into each sequence's pages.

        Returns the caches plus the dequantized per-sequence latents, i.e.
        exactly the values the kernel will read back.
        """
        nope_cache, rope_cache = _empty_caches()
        kv_c_deq, k_pe_deq = [], []
        for s, seq_len in enumerate(seq_lens):
            kv_c = self.rng.standard_normal(
                (seq_len, LKV_DIM)).astype(np.float32)
            k_pe = self.rng.standard_normal(
                (seq_len, ROPE_DIM)).astype(np.float32)
            kv_c_fp8 = _quantize_fp8(kv_c, k_scale)
            k_pe_fp8 = _quantize_fp8(k_pe, k_scale)
            kv_c_deq.append(_dequantize(kv_c_fp8, k_scale))
            k_pe_deq.append(_dequantize(k_pe_fp8, k_scale))
            # Insert the whole sequence as one "prefill" of this sequence
            # alone; query_start_loc has one segment so the scatter lands at
            # positions 0..seq_len-1 of sequence `s`.
            per_seq_lens = np.zeros(len(seq_lens), np.int32)
            per_seq_lens[s] = seq_len
            starts = np.zeros(len(seq_lens) + 1, np.int32)
            starts[s + 1:] = seq_len
            nope_cache, rope_cache = update_sparse_mla_kv_cache(
                nope_cache,
                rope_cache,
                kv_c_fp8,
                k_pe_fp8,
                jnp.asarray(per_seq_lens),
                block_tables,
                jnp.asarray(starts, jnp.int32),
                nope_spec=NOPE_SPEC,
                rope_spec=ROPE_SPEC)
        return (nope_cache, rope_cache), kv_c_deq, k_pe_deq

    def _random_queries(self, num_tokens):
        q = self.rng.standard_normal(
            (num_tokens, NUM_HEADS, LKV_DIM + ROPE_DIM)).astype(np.float32)
        return jnp.asarray(q, dtype=jnp.bfloat16)

    def _topk_rows(self, num_tokens, seq_of_token, seq_lens, topk,
                   num_selected):
        """Random KV subsets per token, `-1` padded, always including self.

        `num_selected(seq_len)` says how many rows that sequence's tokens
        select. Rows beyond a sequence's `kv_len` are never selected: the
        kernel zeroes those cache rows, so selecting one would add a zero
        value vector to the softmax rather than being excluded.
        """
        rows = np.full((num_tokens, topk), -1, np.int32)
        for t in range(num_tokens):
            s = seq_of_token[t]
            if s < 0:
                continue
            kv_len = seq_lens[s]
            n = min(num_selected(kv_len), kv_len, topk)
            sel = self.rng.choice(kv_len, size=n, replace=False)
            rows[t, :n] = np.sort(sel)
        return rows

    # ------------------------------------------------------------ reference

    def _reference(self, q, topk_rows, seq_of_token, kv_c_deq, k_pe_deq,
                   k_scale):
        """float32 MLA attention over exactly the selected, dequantized KVs."""
        num_tokens = q.shape[0]
        q32 = np.asarray(q.astype(jnp.float32))
        out = np.zeros((num_tokens, NUM_HEADS, LKV_DIM + ROPE_DIM), np.float32)
        valid = np.zeros(num_tokens, bool)
        for t in range(num_tokens):
            s = seq_of_token[t]
            if s < 0:
                continue
            sel = topk_rows[t][topk_rows[t] >= 0]
            if sel.size == 0:
                continue
            keys = np.concatenate([kv_c_deq[s][sel], k_pe_deq[s][sel]], -1)
            values = np.concatenate([
                kv_c_deq[s][sel], k_pe_deq[s][sel],
                np.zeros(
                    (sel.size, out.shape[-1] - keys.shape[-1]), np.float32)
            ], -1)
            scores = q32[t] @ keys.T * self.sm_scale
            scores -= scores.max(-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(-1, keepdims=True)
            out[t] = probs @ values
            valid[t] = True
        return out, valid

    def _run(self,
             q,
             caches,
             kv_lens,
             topk_rows,
             block_tables,
             cu_q_lens,
             distribution,
             k_scale,
             bkv_p=1,
             bq_sz=8,
             max_kv_len=None):
        return masked_dense.masked_dense_ragged_paged_attention(
            q,
            caches[0],
            caches[1],
            jnp.asarray(kv_lens, jnp.int32),
            jnp.asarray(topk_rows, jnp.int32),
            block_tables,
            jnp.asarray(cu_q_lens, jnp.int32),
            jnp.asarray(distribution, jnp.int32),
            sm_scale=self.sm_scale,
            k_scale=k_scale,
            max_kv_len=max_kv_len,
            num_kv_pages_per_block=bkv_p,
            num_queries_per_block=bq_sz,
        )

    def _check(self, output, expected, valid):
        got = np.asarray(output.astype(jnp.float32))[valid]
        # bf16 queries and fp8 keys: ~2e-2 is the fp8 rounding floor, matching
        # the sparse kernel's own functional test.
        np.testing.assert_allclose(got, expected[valid], rtol=2e-2, atol=2e-2)


class MaskedDenseCacheSetupTest(parameterized.TestCase):
    """Validate test-cache setup without needing the TPU attention kernels."""

    def test_fill_cache_preserves_selected_pages_and_fp8_values(self):
        if jax.devices()[0].platform == "cpu":
            # Keep the production signature checked while using the CPU writer.
            self.enter_context(
                mock.patch(f"{__name__}.update_sparse_mla_kv_cache",
                           autospec=True,
                           side_effect=update_sparse_mla_kv_cache_jax))
        helper = MaskedDenseTestBase()
        helper.rng = np.random.default_rng(1234)
        seq_lens, k_scale = [3, 5], 0.5
        block_tables = helper._block_tables(len(seq_lens))
        (nope, rope), expected_nope, expected_rope = helper._fill_cache(
            seq_lens, block_tables, k_scale)
        self.assertEqual(nope.shape, (TOTAL_PAGES, PAGE_SIZE, 4, 128))
        self.assertEqual(rope.shape, (TOTAL_PAGES, PAGE_SIZE // 4, 4, 128))
        for cache, expected, dim in ((nope, expected_nope, LKV_DIM),
                                     (rope, expected_rope, ROPE_DIM)):
            rows = cache.reshape(TOTAL_PAGES, PAGE_SIZE, -1)
            values = jax.lax.bitcast_convert_type(
                rows, jnp.float8_e4m3fn).astype(jnp.float32) * k_scale
            for seq, length in enumerate(seq_lens):
                page = int(block_tables[seq * PAGES_PER_SEQ])
                np.testing.assert_array_equal(
                    np.asarray(values[page, :length, :dim]), expected[seq])
                np.testing.assert_array_equal(np.asarray(rows[page, length:]),
                                              0)


class MaskedDenseMlaTest(MaskedDenseTestBase):
    """Correctness of the kernel against a float32 numpy reference."""

    @parameterized.named_parameters(
        # topk = 64. `kv_len` below / at / above topk, and on / off a
        # bkv_sz (= bkv_p * PAGE_SIZE) boundary.
        dict(testcase_name="prefill_kv_below_topk",
             q_lens=[40],
             seq_lens=[40],
             num_decode=0,
             topk=64,
             k_scale=1.0),
        dict(testcase_name="prefill_kv_at_topk",
             q_lens=[64],
             seq_lens=[64],
             num_decode=0,
             topk=64,
             k_scale=1.0),
        dict(testcase_name="prefill_kv_above_topk",
             q_lens=[48],
             seq_lens=[200],
             num_decode=0,
             topk=64,
             k_scale=1.0),
        dict(testcase_name="kv_off_block_boundary",
             q_lens=[24],
             seq_lens=[97],
             num_decode=0,
             topk=64,
             k_scale=1.0),
        dict(testcase_name="kv_on_block_boundary",
             q_lens=[24],
             seq_lens=[96],
             num_decode=0,
             topk=64,
             k_scale=1.0),
        dict(testcase_name="decode_only",
             q_lens=[1, 1, 1],
             seq_lens=[33, 128, 201],
             num_decode=3,
             topk=64,
             k_scale=1.0),
        dict(testcase_name="mixed_prefill_and_decode",
             q_lens=[1, 1, 20, 33],
             seq_lens=[70, 130, 20, 190],
             num_decode=2,
             topk=64,
             k_scale=1.0),
        dict(testcase_name="k_scale_half",
             q_lens=[1, 24],
             seq_lens=[65, 150],
             num_decode=1,
             topk=64,
             k_scale=0.5),
        dict(testcase_name="wide_kv_block",
             q_lens=[20, 20],
             seq_lens=[150, 250],
             num_decode=0,
             topk=64,
             k_scale=1.0,
             bkv_p=4),
        dict(testcase_name="prefill_only_segment",
             q_lens=[24, 16],
             seq_lens=[100, 60],
             num_decode=0,
             topk=64,
             k_scale=1.0,
             prefill_only_distribution=True),
    )
    def test_matches_dense_reference(self,
                                     q_lens,
                                     seq_lens,
                                     num_decode,
                                     topk,
                                     k_scale,
                                     bkv_p=1,
                                     prefill_only_distribution=False):
        num_seqs = len(q_lens)
        total = sum(q_lens)
        num_tokens = math.ceil(total / TOKEN_PAD) * TOKEN_PAD
        cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)

        block_tables = self._block_tables(num_seqs)
        caches, kv_c_deq, k_pe_deq = self._fill_cache(seq_lens, block_tables,
                                                      k_scale)

        seq_of_token = np.full(num_tokens, -1, np.int32)
        for s, q_len in enumerate(q_lens):
            seq_of_token[cu_q_lens[s]:cu_q_lens[s] + q_len] = s

        topk_rows = self._topk_rows(num_tokens, seq_of_token, seq_lens, topk,
                                    lambda kv_len: max(1, kv_len // 2))
        q = self._random_queries(num_tokens)

        # `tpu_runner.py` always emits (d, d, num_seqs), so the prefill-only
        # segment is empty in practice; `prefill_only_distribution` covers the
        # (0, num_seqs, num_seqs) form a direct caller can still pass.
        distribution = ([0, num_seqs, num_seqs] if prefill_only_distribution
                        else [num_decode, num_decode, num_seqs])
        output = self._run(q,
                           caches,
                           seq_lens,
                           topk_rows,
                           block_tables,
                           cu_q_lens,
                           distribution,
                           k_scale,
                           bkv_p=bkv_p)

        expected, valid = self._reference(q, topk_rows, seq_of_token, kv_c_deq,
                                          k_pe_deq, k_scale)
        self.assertEqual(output.dtype, q.dtype)
        self._check(output, expected, valid)

    def test_fully_padded_token_is_finite(self):
        """An all-`-1` mask row must not produce NaN/inf.

        Such a token is padding whose output is discarded, but its softmax
        sees `mask_value` in every column of every block, so `p == 1`
        everywhere and the result is `sum(kv) / bkv_sz`. Garbage, but it must
        stay finite: a NaN here would spread through the W_UV projection.
        """
        q_lens, seq_lens = [8], [96]
        num_tokens = TOKEN_PAD
        topk = 64
        block_tables = self._block_tables(1)
        caches, _, _ = self._fill_cache(seq_lens, block_tables, 1.0)

        seq_of_token = np.full(num_tokens, -1, np.int32)
        seq_of_token[:q_lens[0]] = 0
        topk_rows = self._topk_rows(num_tokens, seq_of_token, seq_lens, topk,
                                    lambda kv_len: max(1, kv_len // 2))
        # Token 0 selects nothing at all.
        topk_rows[0, :] = -1

        q = self._random_queries(num_tokens)
        output = self._run(q, caches, seq_lens, topk_rows, block_tables,
                           [0, q_lens[0]], [0, 0, 1], 1.0)
        got = np.asarray(output[:q_lens[0]].astype(jnp.float32))
        self.assertTrue(np.all(np.isfinite(got)),
                        "masked-dense output contains NaN or inf")

    @parameterized.named_parameters(
        dict(testcase_name="prefill",
             q_lens=[32],
             seq_lens=[160],
             num_decode=0),
        dict(testcase_name="decode",
             q_lens=[1, 1],
             seq_lens=[97, 224],
             num_decode=2),
    )
    def test_matches_sparse_kernel(self, q_lens, seq_lens, num_decode):
        """Same inputs, same caches, both kernels: outputs must agree."""
        num_seqs = len(q_lens)
        topk, k_scale = 64, 1.0
        total = sum(q_lens)
        num_tokens = math.ceil(total / TOKEN_PAD) * TOKEN_PAD
        cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)

        block_tables = self._block_tables(num_seqs)
        caches, _, _ = self._fill_cache(seq_lens, block_tables, k_scale)

        seq_of_token = np.full(num_tokens, -1, np.int32)
        for s, q_len in enumerate(q_lens):
            seq_of_token[cu_q_lens[s]:cu_q_lens[s] + q_len] = s
        # The sparse kernel requires every token -- padding included -- to
        # carry at least one valid entry, so pad rows select KV row 0.
        seq_of_token_padded = np.where(seq_of_token < 0, 0, seq_of_token)
        topk_rows = self._topk_rows(num_tokens, seq_of_token_padded, seq_lens,
                                    topk, lambda kv_len: max(1, kv_len // 3))

        q = self._random_queries(num_tokens)
        dense_out = self._run(q, caches, seq_lens, topk_rows, block_tables,
                              cu_q_lens, [num_decode, num_decode, num_seqs],
                              k_scale)
        sparse_out = sparse_mla_kernel.sparse_ragged_paged_attention(
            q,
            caches[0],
            caches[1],
            jnp.asarray(topk_rows, jnp.int32),
            block_tables,
            jnp.asarray(cu_q_lens, jnp.int32),
            jnp.asarray([num_decode, num_decode, num_seqs], jnp.int32),
            sm_scale=self.sm_scale,
            k_scale=k_scale,
        )

        valid = seq_of_token >= 0
        dense = np.asarray(dense_out.astype(jnp.float32))[valid][..., :LKV_DIM]
        sparse = np.asarray(sparse_out.astype(
            jnp.float32))[valid][..., :LKV_DIM]
        # Different summation orders over fp32 accumulators; the two are the
        # same quantity, not the same bits.
        np.testing.assert_allclose(dense, sparse, rtol=2e-2, atol=2e-2)


class AnalyticCausalMaskTest(MaskedDenseTestBase):
    """`max_kv_len <= topk`: the CSA mask degenerates to the causal mask.

    `streamindex_topk` scores every non-causal or past-`kv_len` position as
    `-inf`, takes an exact top-k, and maps `-inf` back to `-1`. So a token
    with at most `topk` causal candidates has all of them selected, and the
    kernel can compute the mask from `lax.broadcasted_iota` instead of
    reading a materialized bitmap -- skipping `generate_mask_sc` entirely.

    These tests build genuinely causal `topk_indices`, which is what the
    indexer emits in that regime, and check the analytic kernel against both
    a numpy reference and the bitmap path on identical inputs.
    """

    def _causal_topk(self, num_tokens, seq_of_token, seq_lens, q_lens,
                     cu_q_lens, topk):
        """What the indexer emits when every token has <= topk candidates."""
        rows = np.full((num_tokens, topk), -1, np.int32)
        for s, q_len in enumerate(q_lens):
            base = seq_lens[s] - q_len
            for local in range(q_len):
                pos = base + local
                assert pos + 1 <= topk
                rows[cu_q_lens[s] + local, :pos + 1] = np.arange(pos + 1)
        return rows

    @parameterized.named_parameters(
        dict(testcase_name="prefill", q_lens=[48], seq_lens=[48],
             num_decode=0),
        dict(testcase_name="chunked_prefill",
             q_lens=[24],
             seq_lens=[100],
             num_decode=0),
        dict(testcase_name="decode",
             q_lens=[1, 1],
             seq_lens=[97, 128],
             num_decode=2),
        dict(testcase_name="mixed",
             q_lens=[1, 1, 30],
             seq_lens=[64, 90, 120],
             num_decode=2),
    )
    def test_analytic_matches_materialized_mask(self, q_lens, seq_lens,
                                                num_decode):
        num_seqs = len(q_lens)
        total = sum(q_lens)
        num_tokens = math.ceil(total / TOKEN_PAD) * TOKEN_PAD
        cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        topk = MAX_KV_LEN  # >= every sequence's kv_len, so analytic applies

        block_tables = self._block_tables(num_seqs)
        caches, kv_c_deq, k_pe_deq = self._fill_cache(seq_lens, block_tables,
                                                      1.0)
        seq_of_token = np.full(num_tokens, -1, np.int32)
        for s, q_len in enumerate(q_lens):
            seq_of_token[cu_q_lens[s]:cu_q_lens[s] + q_len] = s
        topk_rows = self._causal_topk(num_tokens, seq_of_token, seq_lens,
                                      q_lens, cu_q_lens, topk)

        q = self._random_queries(num_tokens)
        args = (q, caches, seq_lens, topk_rows, block_tables, cu_q_lens,
                [num_decode, num_decode, num_seqs], 1.0)
        # max_kv_len == MAX_KV_LEN <= topk -> analytic; the default
        # (page-table stride) is also MAX_KV_LEN here, so force the bitmap
        # path with a topk narrower than the stride instead.
        analytic = self._run(*args, max_kv_len=MAX_KV_LEN)

        expected, valid = self._reference(q, topk_rows, seq_of_token, kv_c_deq,
                                          k_pe_deq, 1.0)
        self._check(analytic, expected, valid)

        # Same batch through the materialized CSA mask: pad topk_indices to a
        # width above max_kv_len so `analytic_mask` is False.
        wide = np.full((num_tokens, MAX_KV_LEN // 2), -1, np.int32)
        wide[:, :topk_rows.shape[1]] = topk_rows[:, :MAX_KV_LEN // 2]
        bitmap = self._run(q, caches, seq_lens, wide, block_tables, cu_q_lens,
                           [num_decode, num_decode, num_seqs], 1.0)
        np.testing.assert_allclose(np.asarray(analytic.astype(jnp.float32)),
                                   np.asarray(bitmap.astype(jnp.float32)),
                                   rtol=1e-6,
                                   atol=1e-6)

    def test_max_kv_len_beyond_page_table_is_rejected(self):
        q_lens, seq_lens = [8], [32]
        block_tables = self._block_tables(1)
        caches, _, _ = self._fill_cache(seq_lens, block_tables, 1.0)
        topk_rows = np.zeros((TOKEN_PAD, 64), np.int32)
        q = self._random_queries(TOKEN_PAD)
        with self.assertRaises(AssertionError):
            self._run(q,
                      caches,
                      seq_lens,
                      topk_rows,
                      block_tables, [0, q_lens[0]], [0, 0, 1],
                      1.0,
                      max_kv_len=MAX_KV_LEN * 2)


if __name__ == "__main__":
    absltest.main()
