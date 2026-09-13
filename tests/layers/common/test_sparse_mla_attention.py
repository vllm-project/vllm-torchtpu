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

import contextlib
import math
import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.mla import dispatch as mla_dispatch
from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import (KVCacheLayout,
                                                      KVCacheType,
                                                      SparseMLAKVCacheSpec)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.common import attention_interface

LKV_DIM = 512
ROPE_DIM = 64
NUM_HEADS = 16
PAGE_SIZE = 32
PAGES_PER_SEQ = 4
TOTAL_PAGES = 16
TOKEN_PAD = 16  # attention kernel batch size; token count must be a multiple

# Layouts pinned explicitly rather than read from the environment: these
# tests check the exact geometry dsa_gather is written against.
KV_PACKING = sparse_mla_kernel.get_dtype_packing(jnp.float8_e4m3fn)
NOPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.NOPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, LKV_DIM, KV_PACKING)
ROPE_SPEC = SparseMLAKVCacheSpec.create(KVCacheType.ROPE,
                                        KVCacheLayout.TENSORCORE, TOTAL_PAGES,
                                        PAGE_SIZE, ROPE_DIM, KV_PACKING)

_LIMIT_NAMES = ("MASKED_DENSE_ANALYTIC_LIMIT", "MASKED_DENSE_LIMIT")

# Pinned to the gather kernel: every masked-dense tier disabled.
GATHER_ONLY = (0, 0)


@contextlib.contextmanager
def _masked_dense_limits(*limits: int):
    """Forces the tier the dispatch picks, for the duration of the block.

    `limits` is (analytic, bitmap). These module constants are resolved from
    the environment at import, so tests override them here rather than by
    setting `os.environ`.
    """
    assert len(limits) == len(_LIMIT_NAMES)
    module = mla_dispatch
    saved = [getattr(module, n) for n in _LIMIT_NAMES]
    for name, value in zip(_LIMIT_NAMES, limits):
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in zip(_LIMIT_NAMES, saved):
            setattr(module, name, value)


@contextlib.contextmanager
def _noop():
    yield


def _empty_pair():
    """Zeroed (nope, rope) caches in dsa_gather's native tiled layouts."""
    return (jnp.zeros(NOPE_SPEC.shape, NOPE_SPEC.jax_dtype),
            jnp.zeros(ROPE_SPEC.shape, ROPE_SPEC.jax_dtype))


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

    def _call(self,
              kv_cache,
              ql_nope,
              q_pe,
              kv_c_fp8,
              k_pe_fp8,
              topk_rows,
              seq_lens,
              query_start_loc,
              distribution,
              block_tables,
              mesh,
              force_sparse=False):
        with (_masked_dense_limits(*GATHER_ONLY) if force_sparse else _noop()):
            return self._call_inner(kv_cache, ql_nope, q_pe, kv_c_fp8,
                                    k_pe_fp8, topk_rows, seq_lens,
                                    query_start_loc, distribution,
                                    block_tables, mesh)

    def _call_inner(self, kv_cache, ql_nope, q_pe, kv_c_fp8, k_pe_fp8,
                    topk_rows, seq_lens, query_start_loc, distribution,
                    block_tables, mesh):
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
            NOPE_SPEC,
            ROPE_SPEC,
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
        dict(testcase_name="single_prefill_sparse",
             q_lens=[48],
             topk=64,
             force_sparse=True),
        dict(testcase_name="mixed_prefill_sparse",
             q_lens=[40, 24],
             topk=64,
             force_sparse=True),
        dict(testcase_name="padded_tokens_sparse",
             q_lens=[20, 16],
             topk=64,
             force_sparse=True),
    )
    def test_causal_matches_dense_reference(self,
                                            q_lens,
                                            topk,
                                            force_sparse=False):
        """Causal-arange topk == dense MLA; validates the full path.

        These sequences are far below the masked-dense limit, so the default
        run exercises the masked-dense kernel; `force_sparse` pins the same
        case to the gather kernel.
        """
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
        _, output = self._call(self._empty_cache(),
                               ql_nope,
                               q_pe,
                               kv_c_fp8,
                               k_pe_fp8,
                               topk_rows,
                               q_lens,
                               query_start_loc, [0, num_seqs, num_seqs],
                               self._block_tables(num_seqs),
                               mesh,
                               force_sparse=force_sparse)

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
            # Always include self, and never repeat an index: the indexer
            # emits distinct positions, and the two DSA kernels do not agree
            # on repeats -- a gathered row is attended twice, a mask bit set
            # twice is still one column.
            others = self.rng.choice(pos, size=n_sel -
                                     1, replace=False) if pos else np.empty(
                                         0, np.int64)
            topk_rows[pos, :n_sel] = np.sort(np.append(others, pos))

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


class MaskedDensePrefillTierTest(parameterized.TestCase):
    """Prefill cost tier: 0 analytic, 1 CSA bitmap, 2 sparse gather.

    Pure jnp on tiny arrays, so it runs anywhere and can cover cases the
    numeric tests below cannot reach cheaply.
    """

    LIMITS = (2048, 6144)

    def _tier(self, seq_lens, q_lens, distribution, limits=None):
        starts = np.zeros(len(seq_lens) + 1, np.int32)
        starts[1:len(q_lens) + 1] = np.cumsum(q_lens)
        starts[len(q_lens) + 1:] = sum(q_lens)
        return int(
            mla_dispatch.masked_dense_prefill_mla_tier(
                jnp.asarray(seq_lens, jnp.int32),
                jnp.asarray(distribution, jnp.int32),
                jnp.asarray(starts, jnp.int32), limits or self.LIMITS))

    def test_measured_prefill_crossovers(self):
        self.assertEqual(self._tier([1024], [1024], [0, 0, 1]), 0)
        self.assertEqual(self._tier([2048], [1024], [0, 0, 1]), 0)
        self.assertEqual(self._tier([3072], [1024], [0, 0, 1]), 1)
        self.assertEqual(self._tier([6144], [1024], [0, 0, 1]), 1)
        self.assertEqual(self._tier([7168], [1024], [0, 0, 1]), 2)

    @parameterized.parameters(([1], [1, 1, 1]), ([1, 1024], [1, 1, 2]))
    def test_any_decode_forces_sparse(self, q_lens, distribution):
        self.assertEqual(
            self._tier([1024] * len(q_lens), q_lens, distribution), 2)

    def test_cost_sums_multiple_prefill_requests(self):
        self.assertEqual(self._tier([1024, 2048], [512, 512], [0, 0, 2]), 0)

    def test_multi_token_verification_is_kernel_supported(self):
        # With no one-token decode prefix, a verification-shaped query is safe
        # to select just like any other mixed launch. This removes the need for
        # a speculative-decoding classification bit in attention metadata.
        self.assertEqual(self._tier([1024], [4], [0, 0, 1]), 0)

    def test_padded_entries_past_num_seqs_do_not_veto(self):
        self.assertEqual(self._tier([40, 10**6, 10**6], [40], [0, 0, 1]), 0)

    def test_zero_limit_disables_a_tier(self):
        self.assertEqual(self._tier([8], [8], [0, 0, 1], (0, 6144)), 1)
        self.assertEqual(self._tier([8], [8], [0, 0, 1], GATHER_ONLY), 2)


class MaskedDenseDispatchTest(parameterized.TestCase):
    """End to end: every tier of the dispatch computes the same attention.

    The branch taken is not observable from Python -- it is a `jax.lax.switch`
    inside the traced function -- so the check is indirect. Each case runs the
    same batch twice, once under limits that route it to a masked-dense tier
    and once pinned to the gather kernel, and the two outputs must agree to
    tolerance. They are not bit-identical in general: the gather kernel does a
    single-block softmax while masked-dense runs an online FlashAttention-2
    softmax over streamed blocks.
    """

    TOPK = 128

    def setUp(self):
        super().setUp()
        self.k_scale = 1.5
        self.sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA kernels require a TPU (SparseCore).")

    def _run(self, q_lens, seq_lens, num_decode, limits):
        """Runs one step under `limits` and returns the attention output.

        Any prior context is written by a preceding prefill step, so every
        causal position a token may attend to is actually in the cache. That
        matters for the analytic tier, which masks by position rather than by
        the caller's `topk_indices`: with an empty prefix it would average in
        zero rows the gather kernel never sees.

        `seq_lens` may be longer than `q_lens`: the extra entries stand in for
        the runner's padding past `num_seqs`.
        """
        num_seqs = len(q_lens)
        pad_seqs = len(seq_lens)
        rng = np.random.default_rng(99)
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x", ))
        block_tables = jnp.asarray(
            rng.permutation(TOTAL_PAGES)[:pad_seqs * PAGES_PER_SEQ].astype(
                np.int32))
        caches = _empty_pair()

        prior = [seq_lens[s] - q_lens[s] for s in range(num_seqs)]
        if any(prior):
            caches, _ = self._step(caches, prior, prior, 0, GATHER_ONLY, rng,
                                   block_tables, mesh, pad_seqs, num_seqs)
        _, out = self._step(caches, q_lens, seq_lens[:num_seqs], num_decode,
                            limits, rng, block_tables, mesh, pad_seqs,
                            num_seqs, seq_lens)
        assert np.any(out), "degenerate case: both kernels would return zeros"
        return out

    def _step(self,
              caches,
              q_lens,
              kv_lens,
              num_decode,
              limits,
              rng,
              block_tables,
              mesh,
              pad_seqs,
              num_seqs,
              seq_lens=None):
        """One `sparse_mla_attention` call; returns (caches, output)."""
        total = sum(q_lens)
        num_tokens = max(TOKEN_PAD, math.ceil(total / TOKEN_PAD) * TOKEN_PAD)
        starts = np.zeros(pad_seqs + 1, np.int32)
        starts[1:num_seqs + 1] = np.cumsum(q_lens)
        starts[num_seqs + 1:] = total
        if seq_lens is None:
            seq_lens = list(kv_lens) + [1] * (pad_seqs - len(kv_lens))

        kv_c = rng.standard_normal((num_tokens, LKV_DIM)).astype(np.float32)
        k_pe = rng.standard_normal((num_tokens, ROPE_DIM)).astype(np.float32)

        # Fully causal selection: with kv_len <= topk this is exactly what
        # `streamindex_topk` emits, since it scores every non-causal or
        # past-kv_len position -inf and takes an exact top-k.
        positions = []
        for s, q_len in enumerate(q_lens):
            base = kv_lens[s] - q_len
            positions.extend(range(base, base + q_len))
        positions.extend([0] * (num_tokens - total))
        topk_rows = _causal_topk(positions, self.TOPK)

        ql_nope = jnp.asarray(rng.standard_normal(
            (num_tokens, NUM_HEADS, LKV_DIM)).astype(np.float32),
                              dtype=jnp.bfloat16)
        q_pe = jnp.asarray(rng.standard_normal(
            (num_tokens, NUM_HEADS, ROPE_DIM)).astype(np.float32),
                           dtype=jnp.bfloat16)

        with (_masked_dense_limits(*limits),
              mock.patch.object(envs, "TPU_MLA_MASKED_DENSE_ENABLED", True),
              mock.patch.object(mla_dispatch, "MASKED_DENSE_MIN_TOKEN_BUCKET",
                                0),
              mock.patch.object(mla_dispatch,
                                "matches_glm52_tpu7x_profile",
                                return_value=True)):
            nope, rope, output = attention_interface.sparse_mla_attention(
                ql_nope,
                q_pe,
                _quantize_fp8(kv_c, self.k_scale),
                _quantize_fp8(k_pe, self.k_scale),
                caches[0],
                caches[1],
                jnp.asarray(topk_rows, jnp.int32),
                jnp.asarray(seq_lens, jnp.int32),
                block_tables,
                jnp.asarray(starts, jnp.int32),
                jnp.asarray([num_decode, num_decode, num_seqs], jnp.int32),
                mesh,
                NOPE_SPEC,
                ROPE_SPEC,
                sm_scale=self.sm_scale,
                k_scale=self.k_scale,
            )
        return (nope, rope), np.asarray(output.astype(jnp.float32))[:total]

    @parameterized.named_parameters(
        # Analytic tier. Prefill/mixed step: distribution[0] != [2].
        dict(testcase_name="prefill_analytic",
             q_lens=[40, 24],
             seq_lens=[40, 24],
             num_decode=0,
             limits=(128, 4096)),
        # Bitmap tier: over the analytic limit, under the bitmap one.
        dict(testcase_name="prefill_bitmap",
             q_lens=[40, 24],
             seq_lens=[40, 24],
             num_decode=0,
             limits=(0, 4096)),
        # All three tiers live, so the switch has three branches.
        dict(testcase_name="prefill_three_branches",
             q_lens=[40, 24],
             seq_lens=[40, 24],
             num_decode=0,
             limits=(64, 4096)),
        # `seq_lens` carries stale padding past num_seqs=1.
        dict(testcase_name="padded_seq_lens",
             q_lens=[40],
             seq_lens=[40, 100000],
             num_decode=0,
             limits=(128, 4096)),
    )
    def test_masked_dense_agrees_with_gather(self, q_lens, seq_lens,
                                             num_decode, limits):
        np.testing.assert_allclose(self._run(q_lens, seq_lens, num_decode,
                                             limits),
                                   self._run(q_lens, seq_lens, num_decode,
                                             GATHER_ONLY),
                                   rtol=2e-2,
                                   atol=2e-2)

    @parameterized.named_parameters(
        # One over-limit sequence pushes the whole step to the gather kernel,
        # short neighbour included -- the dispatch is per step.
        dict(testcase_name="one_long_sequence",
             q_lens=[40, 24],
             seq_lens=[40, 100],
             num_decode=0,
             limits=(32, 64)),
        # A one-token decode prefix keeps PR1 on the original sparse path.
        dict(testcase_name="decode_always_sparse",
             q_lens=[1, 1],
             seq_lens=[80, 96],
             num_decode=2,
             limits=(4096, 4096)),
        dict(testcase_name="mixed_always_sparse",
             q_lens=[1, 40],
             seq_lens=[80, 40],
             num_decode=1,
             limits=(4096, 4096)),
    )
    def test_routes_to_gather(self, q_lens, seq_lens, num_decode, limits):
        """Routed to the gather kernel, so bit-identical to the pinned run."""
        np.testing.assert_array_equal(
            self._run(q_lens, seq_lens, num_decode, limits),
            self._run(q_lens, seq_lens, num_decode, GATHER_ONLY))


class MaskedDenseDispatchSetupTest(parameterized.TestCase):
    """Exercise the dispatch test's cache setup without a TPU kernel."""

    def test_step_updates_typed_caches(self):
        if jax.devices()[0].platform == "cpu":
            # Keep the production signature checked while using the CPU writer.
            self.enter_context(
                mock.patch.object(
                    attention_interface,
                    "update_sparse_mla_kv_cache",
                    autospec=True,
                    side_effect=kv_cache_utils.update_sparse_mla_kv_cache_jax))
        case = MaskedDenseDispatchTest()
        case.k_scale = 1.5
        case.sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x", ))
        with mock.patch.object(mla_dispatch,
                               "ragged_paged_attention",
                               side_effect=lambda q, *args, **kwargs: q):
            caches, output = case._step(
                _empty_pair(), [8], [8], 0, GATHER_ONLY,
                np.random.default_rng(99),
                jnp.arange(PAGES_PER_SEQ, dtype=jnp.int32), mesh, 1, 1)

        self.assertEqual(output.shape, (8, NUM_HEADS, LKV_DIM))
        self.assertTrue(np.any(output))
        for cache, spec in zip(caches, (NOPE_SPEC, ROPE_SPEC)):
            self.assertEqual(cache.shape, spec.shape)
            self.assertEqual(cache.dtype, spec.jax_dtype)
            self.assertTrue(np.any(np.asarray(cache)))
            np.testing.assert_array_equal(np.asarray(cache)[1:], 0)


class MaskedDenseLimitResolutionTest(parameterized.TestCase):
    """`_resolve_masked_dense_limit` rejects rather than rounds."""

    ENV = "TPU_MLA_MASKED_DENSE_ANALYTIC_MAX_KV_LEN"

    def _resolve(self, default=6144):
        return mla_dispatch._resolve_masked_dense_limit(self.ENV, default)

    def test_masked_dense_routing_is_opt_in(self):
        getter = envs.environment_variables["TPU_MLA_MASKED_DENSE_ENABLED"]
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TPU_MLA_MASKED_DENSE_ENABLED", None)
            self.assertFalse(getter())
        with mock.patch.dict(os.environ,
                             {"TPU_MLA_MASKED_DENSE_ENABLED": "1"}):
            self.assertTrue(getter())

    def test_unset_uses_default(self):
        os.environ.pop(self.ENV, None)
        self.assertEqual(self._resolve(), 6144)

    @parameterized.parameters(0, 1024, 6144)
    def test_valid_override(self, value):
        os.environ[self.ENV] = str(value)
        try:
            self.assertEqual(self._resolve(), value)
        finally:
            del os.environ[self.ENV]

    @parameterized.parameters(-1024, 1000, 1)
    def test_misaligned_override_raises(self, value):
        os.environ[self.ENV] = str(value)
        try:
            with self.assertRaises(ValueError):
                self._resolve()
        finally:
            del os.environ[self.ENV]


class MaskedDenseProfileTest(parameterized.TestCase):

    @staticmethod
    def _operands(*, tokens=512, heads=64, topk=2048, page_size=1024):
        shape = jax.ShapeDtypeStruct
        return (
            shape((tokens, heads, 512 + 64), jnp.bfloat16),
            shape((8, page_size, 4, 128), jnp.uint8),
            shape((8, page_size // 4, 4, 128), jnp.uint8),
            shape((tokens, topk), jnp.int32),
        )

    def test_exact_glm52_profile_is_enabled(self):
        self.assertTrue(
            mla_dispatch.matches_glm52_tpu7x_profile(*self._operands()))

    @parameterized.parameters(dict(heads=16), dict(topk=1024),
                              dict(page_size=512))
    def test_other_sparse_mla_profiles_fall_back(self, **overrides):
        self.assertFalse(
            mla_dispatch.matches_glm52_tpu7x_profile(*self._operands(
                **overrides)))

    @parameterized.parameters(16, 32, 64, 128, 256)
    def test_decode_sized_bucket_bypasses_dispatch(self, tokens):
        q, nope, rope, topk = self._operands(tokens=tokens)
        sentinel = object()
        with (mock.patch.object(envs, "TPU_MLA_MASKED_DENSE_ENABLED", True),
              mock.patch.object(mla_dispatch,
                                "sparse_ragged_paged_attention",
                                return_value=sentinel) as sparse,
              mock.patch.object(mla_dispatch,
                                "_masked_dense_prefill_cost_tier",
                                side_effect=AssertionError(
                                    "small buckets must not build dispatch"))):
            result = mla_dispatch.ragged_paged_attention(
                q, nope, rope, topk, jax.ShapeDtypeStruct((64, ), jnp.int32),
                jax.ShapeDtypeStruct((64 * 9, ), jnp.int32),
                jax.ShapeDtypeStruct((65, ), jnp.int32),
                jax.ShapeDtypeStruct((3, ), jnp.int32))
        self.assertIs(result, sentinel)
        sparse.assert_called_once()
