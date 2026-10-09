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
import types
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
from vllm_torchtpu.kernels.mla import dispatch as mla_dispatch
from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.core import attention_interface

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
NOPE_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.NOPE,
    KVCacheLayout.TENSORCORE,
    TOTAL_PAGES,
    PAGE_SIZE,
    LKV_DIM,
    KV_PACKING,
)
ROPE_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.ROPE,
    KVCacheLayout.TENSORCORE,
    TOTAL_PAGES,
    PAGE_SIZE,
    ROPE_DIM,
    KV_PACKING,
)

NOPE_SC_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.NOPE,
    KVCacheLayout.SPARSECORE,
    TOTAL_PAGES,
    PAGE_SIZE,
    LKV_DIM,
    KV_PACKING,
)
ROPE_SC_SPEC = SparseMLAKVCacheSpec.create(
    KVCacheType.ROPE,
    KVCacheLayout.SPARSECORE,
    TOTAL_PAGES,
    PAGE_SIZE,
    ROPE_DIM,
    KV_PACKING,
)

# The real resolver, before any test replaces it.
_RESOLVE_LIMITS = mla_dispatch.masked_dense_limits

# Pinned to the gather kernel: every masked-dense tier disabled.
GATHER_ONLY = (0, 0)


@contextlib.contextmanager
def _masked_dense_limits(*limits: int):
    """Forces the tier the dispatch picks, for the duration of the block.

    `limits` is (analytic, bitmap). `masked_dense_limits` resolves them from
    the environment once per process, so tests override it here rather than
    by setting `os.environ`.
    """
    assert len(limits) == 2
    with mock.patch.object(
        mla_dispatch, "masked_dense_limits", return_value=tuple(limits)
    ):
        yield


@contextlib.contextmanager
def _noop():
    yield


def _empty_pair(nope_spec=NOPE_SPEC, rope_spec=ROPE_SPEC):
    """Zeroed (nope, rope) caches in dsa_gather's native tiled layouts."""
    return (
        jnp.zeros(nope_spec.shape, nope_spec.jax_dtype),
        jnp.zeros(rope_spec.shape, rope_spec.jax_dtype),
    )


def _quantize_fp8(x: np.ndarray, k_scale: float) -> jax.Array:
    return jnp.asarray(x / k_scale).astype(jnp.float8_e4m3fn)


def _dequantize(x: jax.Array, k_scale: float) -> np.ndarray:
    return np.asarray(x.astype(jnp.float32)) * k_scale


def _causal_topk(positions: list[int], topk: int) -> np.ndarray:
    """topk_indices row per token: [0..pos] then -1 padding."""
    rows = np.full((len(positions), topk), -1, np.int32)
    for i, pos in enumerate(positions):
        assert pos + 1 <= topk
        rows[i, : pos + 1] = np.arange(pos + 1)
    return rows


@pytest.mark.tpu_test
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
        return jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x",))

    def _random_latents(self, n):
        kv_c = self.rng.standard_normal((n, LKV_DIM)).astype(np.float32)
        k_pe = self.rng.standard_normal((n, ROPE_DIM)).astype(np.float32)
        return kv_c, k_pe

    def _random_queries(self, num_tokens):
        ql_nope = jnp.asarray(
            self.rng.standard_normal((num_tokens, NUM_HEADS, LKV_DIM)).astype(
                np.float32
            ),
            dtype=jnp.bfloat16,
        )
        q_pe = jnp.asarray(
            self.rng.standard_normal((num_tokens, NUM_HEADS, ROPE_DIM)).astype(
                np.float32
            ),
            dtype=jnp.bfloat16,
        )
        return ql_nope, q_pe

    def _block_tables(self, num_seqs):
        """Distinct physical pages per sequence, permuted."""
        perm = self.rng.permutation(TOTAL_PAGES)[: num_seqs * PAGES_PER_SEQ]
        return jnp.asarray(
            perm.reshape(num_seqs, PAGES_PER_SEQ).reshape(-1), dtype=jnp.int32
        )

    def _empty_cache(self):
        return _empty_pair()

    def _call(
        self,
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
        force_sparse=False,
        *,
        nope_spec=NOPE_SPEC,
        rope_spec=ROPE_SPEC,
    ):
        with _masked_dense_limits(*GATHER_ONLY) if force_sparse else _noop():
            return self._call_inner(
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
                nope_spec=nope_spec,
                rope_spec=rope_spec,
            )

    def _call_inner(
        self,
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
        *,
        nope_spec=NOPE_SPEC,
        rope_spec=ROPE_SPEC,
    ):
        nope_cache, rope_cache, output = attention_interface.sparse_mla_attention(
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
            nope_spec,
            rope_spec,
            sm_scale=self.sm_scale,
            k_scale=self.k_scale,
        )
        return (nope_cache, rope_cache), output

    def test_sparsecore_writer_to_attention(self):
        """The uint32 writer feeds the SparseCore-layout mode directly."""
        self._require_tpu()
        q_len = 64
        kv_c, k_pe = self._random_latents(q_len)
        kv_c_fp8 = _quantize_fp8(kv_c, self.k_scale)
        k_pe_fp8 = _quantize_fp8(k_pe, self.k_scale)
        ql_nope, q_pe = self._random_queries(q_len)
        topk_rows = _causal_topk(list(range(q_len)), q_len)

        cache, output = self._call(
            _empty_pair(NOPE_SC_SPEC, ROPE_SC_SPEC),
            ql_nope,
            q_pe,
            kv_c_fp8,
            k_pe_fp8,
            topk_rows,
            [q_len],
            [0, q_len],
            [0, 1, 1],
            self._block_tables(1),
            self._make_mesh(),
            nope_spec=NOPE_SC_SPEC,
            rope_spec=ROPE_SC_SPEC,
        )

        self.assertEqual(cache[0].dtype, jnp.uint32)
        self.assertEqual(cache[1].dtype, jnp.uint32)
        expected, valid = self._reference(
            ql_nope,
            q_pe,
            topk_rows,
            [q_len],
            [0, q_len],
            [_dequantize(kv_c_fp8, self.k_scale)],
            [_dequantize(k_pe_fp8, self.k_scale)],
        )
        self._check(output, expected, valid)

    def _reference(
        self, ql_nope, q_pe, topk_rows, q_lens, query_start_loc, kv_c_deq, k_pe_deq
    ):
        """f32 MLA attention over exactly the selected (dequantized) KVs.

        kv_c_deq/k_pe_deq: per-sequence arrays [seq_len, dim] of the values
        the cache actually holds after fp8 rounding.
        """
        num_tokens = ql_nope.shape[0]
        q = np.concatenate(
            [
                np.asarray(ql_nope.astype(jnp.float32)),
                np.asarray(q_pe.astype(jnp.float32)),
            ],
            -1,
        )  # [T, N, 576]
        out = np.zeros((num_tokens, NUM_HEADS, LKV_DIM), np.float32)
        mask = np.zeros(num_tokens, bool)
        for s in range(len(q_lens)):
            for local in range(q_lens[s]):
                t = query_start_loc[s] + local
                sel = topk_rows[t][topk_rows[t] >= 0]
                keys = np.concatenate(
                    [kv_c_deq[s][sel], k_pe_deq[s][sel]], -1
                )  # [n, 576]
                scores = q[t] @ keys.T * self.sm_scale  # [N, n]
                scores -= scores.max(-1, keepdims=True)
                probs = np.exp(scores)
                probs /= probs.sum(-1, keepdims=True)
                out[t] = probs @ kv_c_deq[s][sel]
                mask[t] = True
        return out, mask

    def _check(self, output, expected, valid_mask):
        got = np.asarray(output.astype(jnp.float32))[valid_mask]
        np.testing.assert_allclose(got, expected[valid_mask], rtol=2e-2, atol=2e-2)

    @parameterized.named_parameters(
        dict(testcase_name="single_prefill", q_lens=[48], topk=64),
        dict(testcase_name="mixed_prefill", q_lens=[40, 24], topk=64),
        dict(testcase_name="padded_tokens", q_lens=[20, 16], topk=64),
        dict(
            testcase_name="single_prefill_sparse",
            q_lens=[48],
            topk=64,
            force_sparse=True,
        ),
        dict(
            testcase_name="mixed_prefill_sparse",
            q_lens=[40, 24],
            topk=64,
            force_sparse=True,
        ),
        dict(
            testcase_name="padded_tokens_sparse",
            q_lens=[20, 16],
            topk=64,
            force_sparse=True,
        ),
    )
    def test_causal_matches_dense_reference(self, q_lens, topk, force_sparse=False):
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
            kv_c_rows + [jnp.zeros((pad, LKV_DIM), jnp.float8_e4m3fn)]
        )
        k_pe_fp8 = jnp.concatenate(
            k_pe_rows + [jnp.zeros((pad, ROPE_DIM), jnp.float8_e4m3fn)]
        )
        # Padding tokens still need >= 1 valid topk entry (kernel contract).
        topk_rows = np.concatenate(
            [_causal_topk(positions, topk), _causal_topk([0] * pad, topk)]
        )

        ql_nope, q_pe = self._random_queries(num_tokens)
        mesh = self._make_mesh()
        _, output = self._call(
            self._empty_cache(),
            ql_nope,
            q_pe,
            kv_c_fp8,
            k_pe_fp8,
            topk_rows,
            q_lens,
            query_start_loc,
            [0, num_seqs, num_seqs],
            self._block_tables(num_seqs),
            mesh,
            force_sparse=force_sparse,
        )

        expected, valid = self._reference(
            ql_nope, q_pe, topk_rows, q_lens, query_start_loc, kv_c_deq, k_pe_deq
        )
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
            others = (
                self.rng.choice(pos, size=n_sel - 1, replace=False)
                if pos
                else np.empty(0, np.int64)
            )
            topk_rows[pos, :n_sel] = np.sort(np.append(others, pos))

        ql_nope, q_pe = self._random_queries(q_len)
        mesh = self._make_mesh()
        _, output = self._call(
            self._empty_cache(),
            ql_nope,
            q_pe,
            kv_c_fp8,
            k_pe_fp8,
            topk_rows,
            [q_len],
            [0, q_len],
            [0, 1, 1],
            self._block_tables(1),
            mesh,
        )

        expected, valid = self._reference(
            ql_nope,
            q_pe,
            topk_rows,
            [q_len],
            [0, q_len],
            [_dequantize(kv_c_fp8, self.k_scale)],
            [_dequantize(k_pe_fp8, self.k_scale)],
        )
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
        kv_cache, _ = self._call(
            kv_cache,
            ql_nope,
            q_pe,
            jnp.concatenate(kv_c_rows),
            jnp.concatenate(k_pe_rows),
            _causal_topk(positions, topk),
            prior_lens,
            query_start_loc,
            [0, num_seqs, num_seqs],
            block_tables,
            mesh,
        )
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
        topk_rows = np.concatenate(
            [
                _causal_topk(prior_lens, topk),  # decode pos == prior_len
                _causal_topk([0] * pad, topk),
            ]
        )
        ql_nope, q_pe = self._random_queries(num_tokens)
        seq_lens = [n + 1 for n in prior_lens]
        _, output = self._call(
            kv_cache,
            ql_nope,
            q_pe,
            jnp.concatenate([kv_c_fp8, jnp.zeros((pad, LKV_DIM), jnp.float8_e4m3fn)]),
            jnp.concatenate([k_pe_fp8, jnp.zeros((pad, ROPE_DIM), jnp.float8_e4m3fn)]),
            topk_rows,
            seq_lens,
            list(range(num_seqs + 1)),
            [num_seqs, num_seqs, num_seqs],
            block_tables,
            mesh,
        )

        for s in range(num_seqs):
            kv_c_deq[s] = np.concatenate(
                [kv_c_deq[s], _dequantize(kv_c_fp8[s : s + 1], self.k_scale)]
            )
            k_pe_deq[s] = np.concatenate(
                [k_pe_deq[s], _dequantize(k_pe_fp8[s : s + 1], self.k_scale)]
            )
        expected, valid = self._reference(
            ql_nope,
            q_pe,
            topk_rows,
            [1] * num_seqs,
            list(range(num_seqs + 1)),
            kv_c_deq,
            k_pe_deq,
        )
        self._check(output, expected, valid)


@pytest.mark.tpu_test
class SparseMlaAttentionDcpValidationTest(parameterized.TestCase):
    """Argument checks in `sparse_mla_attention_dcp`, the decode-context-
    parallel (DCP) variant where the KV cache is split across several chips.

    The function checks that the caller's `dcp_size` agrees with the mesh
    and with the shape of the top-k indices before it does anything else.
    These checks run in plain Python at the top of the function, so each
    test can pass `None` for every argument the check under test does not
    read, and none of them need a full DCP-sized mesh or a kernel call.
    """

    def _mesh(self, axis_name, size):
        """A one-axis mesh over the first `size` local devices, or skip if
        the host does not have that many."""
        devices = jax.local_devices()
        if len(devices) < size:
            self.skipTest(f"needs {size} local devices, got {len(devices)}")
        return jax.sharding.Mesh(np.array(devices[:size]), (axis_name,))

    def test_dcp_size_must_exceed_one(self):
        """`dcp_size=1` means no sharding, which is what the plain
        `sparse_mla_attention` is for. The DCP variant refuses it, and does
        so before reading `mesh`, so `mesh=None` is fine here."""
        with self.assertRaisesRegex(ValueError, "dcp_size > 1"):
            attention_interface.sparse_mla_attention_dcp(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                mesh=None,
                nope_spec=None,
                rope_spec=None,
                dcp_size=1,
                interleave_size=1,
            )

    def test_mesh_missing_dcp_axis_raises(self):
        """The mesh must have an axis with the DCP axis name. A mesh whose
        only axis is called something else is rejected."""
        mesh = self._mesh("x", 1)
        with self.assertRaisesRegex(ValueError, "requires a mesh with"):
            attention_interface.sparse_mla_attention_dcp(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                mesh=mesh,
                nope_spec=None,
                rope_spec=None,
                dcp_size=2,
                interleave_size=1,
            )

    def test_mesh_axis_size_mismatch_raises(self):
        """The DCP axis must have exactly `dcp_size` devices on it. Here the
        axis has the right name but only one device while `dcp_size=2`."""
        mesh = self._mesh(streamindex_topk.DCP_AXIS_NAME, 1)
        with self.assertRaisesRegex(ValueError, "does not match mesh axis"):
            attention_interface.sparse_mla_attention_dcp(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                mesh=mesh,
                nope_spec=None,
                rope_spec=None,
                dcp_size=2,
                interleave_size=1,
            )

    def test_local_topk_indices_row_count_mismatch_raises(self):
        """`local_topk_indices` holds one row per (chip, token), so it must
        have `dcp_size * num_tokens` rows. This passes a valid mesh so the
        earlier checks succeed, then hands over a table one row short."""
        mesh = self._mesh(streamindex_topk.DCP_AXIS_NAME, 2)
        num_tokens = 4
        ql_nope = jnp.zeros((num_tokens, 2, 8))
        # One row short of the required dcp_size * num_tokens.
        bad_topk = jnp.zeros((2 * num_tokens - 1, 8), jnp.int32)

        with self.assertRaisesRegex(ValueError, "one row per"):
            attention_interface.sparse_mla_attention_dcp(
                ql_nope,
                None,
                None,
                None,
                None,
                None,
                bad_topk,
                None,
                None,
                None,
                None,
                mesh=mesh,
                nope_spec=None,
                rope_spec=None,
                dcp_size=2,
                interleave_size=1,
            )


class MaskedDenseTierTest(parameterized.TestCase):
    """Route selection: 0 analytic, 1 CSA bitmap, 2 sparse gather.

    Pure jnp on tiny arrays, so it runs anywhere and can cover cases the
    numeric tests below cannot reach cheaply.
    """

    LIMITS = (2048, 6144)

    def _tier(self, seq_lens, q_lens, distribution, limits=None):
        starts = np.zeros(len(seq_lens) + 1, np.int32)
        starts[1 : len(q_lens) + 1] = np.cumsum(q_lens)
        starts[len(q_lens) + 1 :] = sum(q_lens)
        return int(
            mla_dispatch.select_tier(
                jnp.asarray(seq_lens, jnp.int32),
                jnp.asarray(distribution, jnp.int32),
                jnp.asarray(starts, jnp.int32),
                limits or self.LIMITS,
            )
        )

    def test_measured_prefill_crossovers(self):
        self.assertEqual(self._tier([1024], [1024], [0, 0, 1]), 0)
        self.assertEqual(self._tier([2048], [1024], [0, 0, 1]), 0)
        self.assertEqual(self._tier([3072], [1024], [0, 0, 1]), 1)
        self.assertEqual(self._tier([6144], [1024], [0, 0, 1]), 1)
        self.assertEqual(self._tier([7168], [1024], [0, 0, 1]), 2)

    def test_decode_only_uses_sparse(self):
        self.assertEqual(self._tier([1024], [1], [1, 1, 1]), 2)

    def test_decode_prefix_and_short_suffix_can_split(self):
        self.assertEqual(self._tier([9000, 1024], [1, 1024], [1, 1, 2]), 0)

    def test_long_prefix_and_short_suffix_can_split(self):
        self.assertEqual(self._tier([8192, 1024], [512, 512], [0, 0, 2]), 0)

    def test_noncontiguous_prefix_falls_back_to_sparse(self):
        self.assertEqual(self._tier([1024, 8192, 1024], [512] * 3, [0, 0, 3]), 2)

    def test_small_suffix_uses_sparse(self):
        self.assertEqual(self._tier([8192, 9216, 8192, 1536], [256] * 4, [0, 0, 4]), 2)

    def test_small_all_short_prefill_still_uses_masked_dense(self):
        self.assertEqual(self._tier([128], [128], [0, 0, 1]), 0)

    def test_measured_mixed_crossover(self):
        self.assertEqual(
            self._tier([9216] * 63 + [5120], [1] * 63 + [961], [63, 63, 64]), 1
        )
        self.assertEqual(
            self._tier([9216] * 63 + [6144], [1] * 63 + [961], [63, 63, 64]), 2
        )

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

    def test_split_point(self):
        seq_lens = jnp.asarray([80, 100, 40, 100000], jnp.int32)
        split_seq, contiguous = mla_dispatch.split_point(
            seq_lens, jnp.asarray([1, 1, 3], jnp.int32), 64
        )
        self.assertEqual(int(split_seq), 2)
        self.assertTrue(bool(contiguous))
        split_seq, contiguous = mla_dispatch.split_point(
            jnp.asarray([40, 100, 40], jnp.int32), jnp.asarray([0, 0, 3], jnp.int32), 64
        )
        self.assertEqual(int(split_seq), 1)
        self.assertFalse(bool(contiguous))


# Flag off: the layer's original sparse call, the reference for every route.
ORIGINAL_SPARSE = None

# (name, q_lens, seq_lens, num_decode). `seq_lens` may be longer than `q_lens`
# to stand in for the runner's padding past `num_seqs`.
ROUTE_COMPOSITIONS = (
    ("whole", [40, 24], [40, 24], 0),
    ("long_prefix", [24, 40], [100, 40], 0),
    ("decode_prefix", [1, 40], [80, 40], 1),
    ("prefix_on_batch_boundary", [16, 48], [100, 48], 0),
    ("prefix_near_end", [56, 8], [100, 8], 0),
)


@pytest.mark.tpu_test
class MaskedDenseDispatchTest(parameterized.TestCase):
    """End to end: every route computes the same attention as sparse.

    The branch taken is not observable from Python -- it is a `jax.lax.switch`
    inside the traced function -- so the check is indirect. Each case runs the
    same batch under routing and through the layer's original sparse call, and
    the two outputs must agree to tolerance. Masked-dense results are not
    bit-identical in general: the gather kernel does a single-block softmax
    while masked-dense runs an online FlashAttention-2 softmax over streamed
    blocks.
    """

    TOPK = 128

    def setUp(self):
        super().setUp()
        self.k_scale = 1.5
        self.sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA kernels require a TPU (SparseCore).")

    def _run(self, q_lens, seq_lens, num_decode, limits, min_suffix_tokens=None):
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
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x",))
        block_tables = jnp.asarray(
            rng.permutation(TOTAL_PAGES)[: pad_seqs * PAGES_PER_SEQ].astype(np.int32)
        )
        caches = _empty_pair()

        prior = [seq_lens[s] - q_lens[s] for s in range(num_seqs)]
        if any(prior):
            caches, _ = self._step(
                caches,
                prior,
                prior,
                0,
                ORIGINAL_SPARSE,
                rng,
                block_tables,
                mesh,
                pad_seqs,
                num_seqs,
            )
        _, out = self._step(
            caches,
            q_lens,
            seq_lens[:num_seqs],
            num_decode,
            limits,
            rng,
            block_tables,
            mesh,
            pad_seqs,
            num_seqs,
            seq_lens,
            min_suffix_tokens=min_suffix_tokens,
        )
        assert np.any(out), "degenerate case: both kernels would return zeros"
        return out

    def _step(
        self,
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
        seq_lens=None,
        min_suffix_tokens=None,
    ):
        """One `sparse_mla_attention` call; returns (caches, output).

        `limits=ORIGINAL_SPARSE` disables routing, so the layer takes its
        original sparse call.
        """
        total = sum(q_lens)
        num_tokens = max(TOKEN_PAD, math.ceil(total / TOKEN_PAD) * TOKEN_PAD)
        starts = np.zeros(pad_seqs + 1, np.int32)
        starts[1 : num_seqs + 1] = np.cumsum(q_lens)
        starts[num_seqs + 1 :] = total
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

        ql_nope = jnp.asarray(
            rng.standard_normal((num_tokens, NUM_HEADS, LKV_DIM)).astype(np.float32),
            dtype=jnp.bfloat16,
        )
        q_pe = jnp.asarray(
            rng.standard_normal((num_tokens, NUM_HEADS, ROPE_DIM)).astype(np.float32),
            dtype=jnp.bfloat16,
        )

        routed = limits is not ORIGINAL_SPARSE
        with contextlib.ExitStack() as stack:
            if routed:
                stack.enter_context(_masked_dense_limits(*limits))
                stack.enter_context(
                    mock.patch.object(mla_dispatch, "MASKED_DENSE_MIN_TOKEN_BUCKET", 0)
                )
                stack.enter_context(
                    mock.patch.object(
                        mla_dispatch, "matches_glm52_tpu7x_profile", return_value=True
                    )
                )
            if min_suffix_tokens is not None:
                stack.enter_context(
                    mock.patch.object(
                        mla_dispatch,
                        "MASKED_DENSE_MIN_SUFFIX_TOKENS",
                        min_suffix_tokens,
                    )
                )
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
                allow_masked_dense=routed,
            )
        return (nope, rope), np.asarray(output.astype(jnp.float32))[:total]

    @parameterized.named_parameters(
        # Analytic tier. Prefill/mixed step: distribution[0] != [2].
        dict(
            testcase_name="prefill_analytic",
            q_lens=[40, 24],
            seq_lens=[40, 24],
            num_decode=0,
            limits=(128, 4096),
        ),
        # Bitmap tier: over the analytic limit, under the bitmap one.
        dict(
            testcase_name="prefill_bitmap",
            q_lens=[40, 24],
            seq_lens=[40, 24],
            num_decode=0,
            limits=(0, 4096),
        ),
        # All three tiers live, so the switch has three branches.
        dict(
            testcase_name="prefill_three_branches",
            q_lens=[40, 24],
            seq_lens=[40, 24],
            num_decode=0,
            limits=(64, 4096),
        ),
        # `seq_lens` carries stale padding past num_seqs=1.
        dict(
            testcase_name="padded_seq_lens",
            q_lens=[40],
            seq_lens=[40, 100000],
            num_decode=0,
            limits=(128, 4096),
        ),
    )
    def test_masked_dense_agrees_with_sparse(
        self, q_lens, seq_lens, num_decode, limits
    ):
        np.testing.assert_allclose(
            self._run(q_lens, seq_lens, num_decode, limits),
            self._run(q_lens, seq_lens, num_decode, ORIGINAL_SPARSE),
            rtol=2e-2,
            atol=2e-2,
        )

    @parameterized.named_parameters(
        # A noncontiguous long request keeps the whole step on sparse.
        dict(
            testcase_name="noncontiguous_long_sequence",
            q_lens=[40, 24],
            seq_lens=[40, 100],
            num_decode=0,
            limits=(32, 64),
        ),
        dict(
            testcase_name="decode_only",
            q_lens=[1, 1],
            seq_lens=[80, 96],
            num_decode=2,
            limits=(4096, 4096),
        ),
        # The suffix is under MASKED_DENSE_MIN_SUFFIX_TOKENS.
        dict(
            testcase_name="short_suffix",
            q_lens=[1, 40],
            seq_lens=[80, 40],
            num_decode=1,
            limits=(4096, 4096),
        ),
        # Every masked-dense tier disabled.
        dict(
            testcase_name="gather_only",
            q_lens=[40, 24],
            seq_lens=[40, 24],
            num_decode=0,
            limits=GATHER_ONLY,
        ),
    )
    def test_sparse_route_is_the_original_sparse_call(
        self, q_lens, seq_lens, num_decode, limits
    ):
        """The sparse route uses the same stages, so it is bit-identical."""
        np.testing.assert_array_equal(
            self._run(q_lens, seq_lens, num_decode, limits),
            self._run(q_lens, seq_lens, num_decode, ORIGINAL_SPARSE),
        )

    @parameterized.named_parameters(
        (f"{name}_tier{tier}", tier, q_lens, seq_lens, num_decode)
        for name, q_lens, seq_lens, num_decode in ROUTE_COMPOSITIONS
        for tier in (0, 1, 2)
    )
    def test_forced_route_matches_original_sparse(
        self, tier, q_lens, seq_lens, num_decode
    ):
        """Every route, including split steps, against the original call.

        The tier is forced through a device operand so all branches compile.
        The minimum suffix is lifted so these small steps can split.
        """
        offset = (tier - seq_lens[0]) % 3
        with mock.patch.object(
            mla_dispatch,
            "select_tier",
            side_effect=lambda lengths, *a, **k: (lengths[0] + offset) % 3,
        ):
            actual = self._run(
                q_lens, seq_lens, num_decode, (64, 4096), min_suffix_tokens=0
            )
        expected = self._run(q_lens, seq_lens, num_decode, ORIGINAL_SPARSE)
        if tier == 2:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


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
                    side_effect=kv_cache_utils.update_sparse_mla_kv_cache_jax,
                )
            )
        case = MaskedDenseDispatchTest()
        case.k_scale = 1.5
        case.sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x",))
        with mock.patch.object(
            mla_dispatch,
            "attention_from_parts",
            side_effect=lambda ql_nope, q_pe, *args, **kwargs: ql_nope,
        ):
            caches, output = case._step(
                _empty_pair(),
                [8],
                [8],
                0,
                GATHER_ONLY,
                np.random.default_rng(99),
                jnp.arange(PAGES_PER_SEQ, dtype=jnp.int32),
                mesh,
                1,
                1,
            )

        self.assertEqual(output.shape, (8, NUM_HEADS, LKV_DIM))
        self.assertTrue(np.any(output))
        for cache, spec in zip(caches, (NOPE_SPEC, ROPE_SPEC)):
            self.assertEqual(cache.shape, spec.shape)
            self.assertEqual(cache.dtype, spec.jax_dtype)
            self.assertTrue(np.any(np.asarray(cache)))
            np.testing.assert_array_equal(np.asarray(cache)[1:], 0)


class SparseMlaLayerBranchTest(parameterized.TestCase):
    """The layer calls the sparse kernel directly unless routing applies."""

    K_SCALE = 1.5
    SM_SCALE = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
    # (allowed, applies) for an unadmitted layer, an admitted layer whose call
    # does not apply and an admitted layer whose call does. The unadmitted
    # case mocks `applies` true to check the layer's own guard.
    CASES = ((False, True), (True, False), (True, True))

    def setUp(self):
        super().setUp()
        if jax.devices()[0].platform == "cpu":
            self.enter_context(
                mock.patch.object(
                    attention_interface,
                    "update_sparse_mla_kv_cache",
                    autospec=True,
                    side_effect=kv_cache_utils.update_sparse_mla_kv_cache_jax,
                )
            )

    def _run_layer(self, flag, applies):
        """One layer step with both kernels stubbed.

        The sparse stub returns ones in the latent part and sevens in the rope
        tail, so the layer's crop is visible; the routed stub returns threes.
        """
        case = MaskedDenseDispatchTest()
        case.k_scale = self.K_SCALE
        case.sm_scale = self.SM_SCALE
        mesh = jax.sharding.Mesh(np.array(jax.local_devices()[:1]), ("x",))

        def sparse_kernel(q, *args, **kwargs):
            del args, kwargs
            return jnp.concatenate(
                [jnp.ones_like(q[..., :LKV_DIM]), jnp.full_like(q[..., LKV_DIM:], 7)],
                axis=-1,
            )

        with (
            mock.patch.object(
                attention_interface,
                "sparse_mla_ragged_paged_attention",
                side_effect=sparse_kernel,
            ) as sparse,
            mock.patch.object(
                mla_dispatch,
                "attention_from_parts",
                side_effect=lambda ql_nope, *a, **k: jnp.full_like(ql_nope, 3),
            ) as routed,
            mock.patch.object(mla_dispatch, "applies", return_value=applies),
        ):
            caches, output = case._step(
                _empty_pair(),
                [8],
                [8],
                0,
                GATHER_ONLY if flag else ORIGINAL_SPARSE,
                np.random.default_rng(99),
                jnp.arange(PAGES_PER_SEQ, dtype=jnp.int32),
                mesh,
                1,
                1,
            )
        return caches, output, sparse, routed

    @parameterized.named_parameters(
        ("flag_off", *CASES[0]),
        ("flag_on_ineligible", *CASES[1]),
        ("flag_on_eligible", *CASES[2]),
    )
    def test_layer_branch(self, flag, applies):
        _, output, sparse, routed = self._run_layer(flag, applies)
        takes_route = flag and applies
        self.assertEqual(output.shape, (8, NUM_HEADS, LKV_DIM))
        self.assertEqual(routed.call_count, int(takes_route))
        self.assertEqual(sparse.call_count, int(not takes_route))
        if takes_route:
            args, kwargs = routed.call_args
            self.assertEqual(args[0].shape[-1], LKV_DIM)
            self.assertEqual(args[1].shape[-1], ROPE_DIM)
            self.assertEqual(kwargs, dict(sm_scale=self.SM_SCALE, k_scale=self.K_SCALE))
            np.testing.assert_array_equal(output, 3)
        else:
            args, kwargs = sparse.call_args
            # q, caches, topk, block tables, query starts, distribution.
            self.assertLen(args, 7)
            self.assertEqual(args[0].shape[-1], LKV_DIM + ROPE_DIM)
            self.assertEqual(
                kwargs,
                dict(
                    sm_scale=self.SM_SCALE,
                    k_scale=self.K_SCALE,
                    cache_layout=NOPE_SPEC.layout.value,
                ),
            )
            # The rope tail is cropped from the sparse output.
            np.testing.assert_array_equal(output, 1)

    def test_every_branch_returns_the_updated_caches(self):
        results = [self._run_layer(*case)[0] for case in self.CASES]
        first_nope, first_rope = map(np.asarray, results[0])
        self.assertTrue(np.any(first_nope))
        self.assertTrue(np.any(first_rope))
        for nope, rope in results[1:]:
            np.testing.assert_array_equal(np.asarray(nope), first_nope)
            np.testing.assert_array_equal(np.asarray(rope), first_rope)


def _cond_boundaries(jaxpr, depth=0):
    """(nesting depth, aval) of every cond operand and result."""
    for eqn in jaxpr.eqns:
        inner_depth = depth
        if eqn.primitive.name == "cond":
            inner_depth = depth + 1
            for var in list(eqn.invars[1:]) + list(eqn.outvars):
                yield inner_depth, var.aval
        for param in eqn.params.values():
            for sub in param if isinstance(param, (list, tuple)) else [param]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _cond_boundaries(inner, inner_depth)


class MaskedDenseRouteCompositionTest(parameterized.TestCase):
    """How `attention_from_parts` composes its routes, with kernels stubbed.

    The sparse chunk stub scales the rows it owns by 5 and, like the real
    attention grid, overwrites rows just past its token end. The masked-dense
    stub scales every row from `sequence_start` by 7, or all rows by 3 when it
    covers the whole step. The tier is forced through a device operand, so all
    branches compile and the selected one runs on CPU.
    """

    NUM_TOKENS = 64
    TOPK = 128
    LIMITS = (64, 4096)
    OVERWRITTEN = -1e4

    @classmethod
    def _fake_sparse_chunk(
        cls, q, *args, start_pos, end_pos, token_end, preserve_suffix=False, **kwargs
    ):
        del args, kwargs
        rows = jnp.arange(q.shape[0])[:, None, None]
        in_chunk = (rows >= start_pos) & (rows < end_pos)
        owned = in_chunk & (rows < token_end)
        overwritten = (
            in_chunk
            & (rows >= token_end)
            & (
                rows
                < token_end + sparse_mla_kernel.DEFAULT_ATTENTION_KERNEL_BATCH_SIZE - 1
            )
        )
        if preserve_suffix:
            overwritten = jnp.zeros_like(overwritten)
        return jnp.where(owned, q * 5, jnp.where(overwritten, cls.OVERWRITTEN, q))

    @staticmethod
    def _fake_masked_dense(
        q,
        nope,
        rope,
        seq_lens,
        topk,
        block_tables,
        starts,
        distribution,
        *,
        sequence_start,
        **kwargs,
    ):
        del nope, rope, seq_lens, topk, block_tables, distribution, kwargs
        if sequence_start is None:
            return q * 3
        rows = jnp.arange(q.shape[0])[:, None, None]
        return jnp.where(rows >= starts[sequence_start], q * 7, q)

    def _operands(self, q_lens, seq_lens, num_decode):
        rng = np.random.default_rng(7)
        num_seqs = len(q_lens)
        starts = np.zeros(num_seqs + 1, np.int32)
        starts[1:] = np.cumsum(q_lens)
        nope, rope = _empty_pair()
        return (
            jnp.asarray(
                rng.standard_normal((self.NUM_TOKENS, NUM_HEADS, LKV_DIM)), jnp.float32
            ),
            jnp.asarray(
                rng.standard_normal((self.NUM_TOKENS, NUM_HEADS, ROPE_DIM)), jnp.float32
            ),
            nope,
            rope,
            jnp.asarray(_causal_topk([0] * self.NUM_TOKENS, self.TOPK)),
            jnp.asarray(seq_lens, jnp.int32),
            jnp.arange(num_seqs * PAGES_PER_SEQ, dtype=jnp.int32),
            jnp.asarray(starts),
            jnp.asarray([num_decode, num_decode, num_seqs], jnp.int32),
        )

    def _route(self):
        def fn(*operands_and_tier):
            *operands, forced = operands_and_tier
            with (
                mock.patch.object(
                    sparse_mla_kernel,
                    "sparse_attention_chunk",
                    side_effect=self._fake_sparse_chunk,
                ),
                mock.patch.object(
                    sparse_mla_kernel,
                    "prepare_gather_indices",
                    side_effect=lambda topk, *a, **k: (
                        jnp.zeros(topk.shape[:1], jnp.int32),
                        topk,
                    ),
                ),
                mock.patch.object(
                    mla_dispatch,
                    "masked_dense_ragged_paged_attention",
                    side_effect=self._fake_masked_dense,
                ),
                mock.patch.object(mla_dispatch, "MASKED_DENSE_MIN_SUFFIX_TOKENS", 0),
                _masked_dense_limits(*self.LIMITS),
                mock.patch.object(
                    mla_dispatch, "select_tier", side_effect=lambda *a, **k: forced
                ),
            ):
                return mla_dispatch.attention_from_parts(
                    *operands, sm_scale=1.0, k_scale=1.0
                )

        return fn

    @parameterized.named_parameters(
        (f"{name}_tier{tier}", tier, q_lens, seq_lens, num_decode)
        for name, q_lens, seq_lens, num_decode in ROUTE_COMPOSITIONS
        for tier in (0, 1, 2)
    )
    def test_route_rows(self, tier, q_lens, seq_lens, num_decode):
        operands = self._operands(q_lens, seq_lens, num_decode)
        total = sum(q_lens)
        ql_nope = np.asarray(operands[0])[:total]
        actual = np.asarray(
            jax.jit(self._route())(*operands, jnp.asarray(tier, jnp.int32))
        )
        self.assertEqual(actual.shape, operands[0].shape)

        factors = np.full(total, 5.0)
        if tier < 2:
            split_seq = sum(
                1
                for s, length in enumerate(seq_lens[: len(q_lens)])
                if s < num_decode or length > self.LIMITS[tier]
            )
            split_token = sum(q_lens[:split_seq])
            if split_seq == 0:
                factors[:] = 3.0
            else:
                factors[split_token:] = 7.0
        np.testing.assert_allclose(
            actual[:total], ql_nope * factors[:, None, None], rtol=1e-6
        )

    @parameterized.named_parameters(
        (name, q_lens, seq_lens, num_decode)
        for name, q_lens, seq_lens, num_decode in ROUTE_COMPOSITIONS
    )
    def test_switch_and_split_do_not_carry_the_query(
        self, q_lens, seq_lens, num_decode
    ):
        """The switch and the split conditional carry only the query parts.

        The per-chunk conditionals of a sparse prefix (depth 3) do carry the
        padded query; whether that copy costs more than the gathers it skips
        is a measurement, not something this test decides.
        """
        operands = self._operands(q_lens, seq_lens, num_decode)
        jaxpr = jax.make_jaxpr(self._route())(*operands, jnp.asarray(1, jnp.int32))
        boundaries = list(_cond_boundaries(jaxpr.jaxpr))
        self.assertNotEmpty(boundaries)
        query_widths = (
            LKV_DIM + ROPE_DIM,
            sparse_mla_kernel.align_to(LKV_DIM + ROPE_DIM, 128),
        )
        for depth, aval in boundaries:
            if depth <= 2 and getattr(aval, "ndim", 0) == 3:
                self.assertNotIn(aval.shape[-1], query_widths)


class SparsePrefixTest(parameterized.TestCase):
    """Prefix routing skips chunks outside its owned rows."""

    NUM_ROWS = 128

    @parameterized.parameters(1, 15, 16, 17, 63, 64, 65, 100, 113, 127)
    def test_suffix_rows_are_preserved(self, split_token):
        q = jnp.asarray(
            np.random.default_rng(3).standard_normal((self.NUM_ROWS, 2, 8)), jnp.float32
        )

        def run_chunk(q, start_pos, token_end):
            return MaskedDenseRouteCompositionTest._fake_sparse_chunk(
                q,
                start_pos=start_pos,
                end_pos=min(start_pos + 64, self.NUM_ROWS),
                token_end=token_end,
                preserve_suffix=True,
            )

        with mock.patch.object(mla_dispatch, "MASKED_DENSE_MIN_SUFFIX_TOKENS", 0):
            out = np.asarray(
                jax.jit(
                    lambda q, split: mla_dispatch._sparse_prefix(q, split, run_chunk)
                )(q, jnp.asarray(split_token, jnp.int32))
            )
        expected = np.asarray(q).copy()
        expected[:split_token] *= 5
        np.testing.assert_array_equal(out, expected)

    def test_chunks_in_the_minimum_suffix_are_not_emitted(self):
        q = jnp.zeros((1024, 2, 8), jnp.float32)
        jaxpr = jax.make_jaxpr(
            lambda q, split: mla_dispatch._sparse_prefix(q, split, lambda q, **k: q)
        )(q, jnp.asarray(1, jnp.int32))
        conds = [e for e in jaxpr.jaxpr.eqns if e.primitive.name == "cond"]
        self.assertLen(conds, (1024 - 512) // 64)

    @parameterized.parameters(
        (0, 64, 17, 32),
        (64, 128, 17, 0),
        (0, 64, 100, 64),
    )
    def test_sparse_attention_chunk_valid_indices(
        self, start_pos, end_pos, token_end, expected_batch_rows
    ):
        topk = 8
        chunk_size = end_pos - start_pos
        q = jnp.zeros((self.NUM_ROWS, NUM_HEADS, LKV_DIM + ROPE_DIM), jnp.bfloat16)
        nope, rope = _empty_pair()
        kv_lens = jnp.ones((self.NUM_ROWS,), jnp.int32)
        physical_indices = jnp.zeros((self.NUM_ROWS, topk), jnp.int32)
        captured = {}

        def fake_gather(nope_c, rope_c, flat_indices, *, num_valid_indices):
            del nope_c, rope_c, flat_indices
            captured["num_valid_indices"] = int(num_valid_indices)
            return (
                jnp.zeros(
                    (
                        chunk_size * topk,
                        kv_cache_utils.WORD_BYTES,
                        kv_cache_utils.TILE_LANE_BYTES,
                    ),
                    jnp.uint8,
                ),
                jnp.zeros(
                    (chunk_size * topk, kv_cache_utils.TILE_LANE_BYTES), jnp.uint8
                ),
            )

        def fake_pallas_call(*args, **kwargs):
            del args, kwargs

            def kernel(_kv_lens, bounds, _sems, q_in, *_rest):
                captured["bounds"] = [int(x) for x in bounds]
                return q_in

            return kernel

        with (
            mock.patch.object(
                sparse_mla_kernel.dsa_gather, "dsa_gather", side_effect=fake_gather
            ),
            mock.patch.object(
                sparse_mla_kernel.pl, "pallas_call", side_effect=fake_pallas_call
            ),
        ):
            sparse_mla_kernel.sparse_attention_chunk(
                q,
                nope,
                rope,
                kv_lens,
                physical_indices,
                None,
                start_pos=start_pos,
                end_pos=end_pos,
                token_end=jnp.asarray(token_end, jnp.int32),
                sm_scale=1.0,
                k_scale=1.0,
                preserve_suffix=True,
                attention_kernel_batch_size=16,
            )
        self.assertEqual(captured["num_valid_indices"], expected_batch_rows * topk)
        self.assertEqual(
            captured["bounds"], [start_pos, start_pos + expected_batch_rows, token_end]
        )


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
        with mock.patch.dict(os.environ, {"TPU_MLA_MASKED_DENSE_ENABLED": "1"}):
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
        self.assertTrue(mla_dispatch.matches_glm52_tpu7x_profile(*self._operands()))

    @parameterized.parameters(dict(heads=16), dict(topk=1024), dict(page_size=512))
    def test_other_sparse_mla_profiles_fall_back(self, **overrides):
        self.assertFalse(
            mla_dispatch.matches_glm52_tpu7x_profile(*self._operands(**overrides))
        )

    def _applies(
        self,
        *,
        nope=None,
        rope=None,
        cache_layout="tensorcore",
        **overrides,
    ):
        q, default_nope, default_rope, topk = self._operands(**overrides)
        tokens, heads, _ = q.shape
        ql_nope = jax.ShapeDtypeStruct((tokens, heads, 512), q.dtype)
        q_pe = jax.ShapeDtypeStruct((tokens, heads, 64), q.dtype)
        return mla_dispatch.applies(
            ql_nope,
            q_pe,
            nope or default_nope,
            rope or default_rope,
            topk,
            cache_layout,
        )

    def test_exact_profile_applies(self):
        self.assertTrue(self._applies())

    @parameterized.parameters(16, 32, 64, 128, 256)
    def test_decode_sized_bucket_keeps_sparse(self, tokens):
        self.assertFalse(self._applies(tokens=tokens))

    @parameterized.parameters(dict(heads=16), dict(topk=1024), dict(page_size=512))
    def test_other_profiles_keep_sparse(self, **overrides):
        self.assertFalse(self._applies(**overrides))

    def test_native_cache_layout_keeps_sparse(self):
        self.assertFalse(
            self._applies(
                nope=jax.ShapeDtypeStruct((8, 1024, 128), jnp.uint32),
                rope=jax.ShapeDtypeStruct((8, 256, 128), jnp.uint32),
                cache_layout="sparsecore",
            )
        )


def _glm52_vllm_config(*, hf=(), parallel=(), cache=(), **top):
    """The admitted configuration, GLM-5.2 on DP16/EP, with overrides."""
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                **{
                    "model_type": "glm_moe_dsa",
                    "architectures": ["GlmMoeDsaForCausalLM"],
                    "num_attention_heads": 64,
                    "kv_lora_rank": 512,
                    "qk_rope_head_dim": 64,
                    "index_topk": 2048,
                    **dict(hf),
                }
            )
        ),
        parallel_config=types.SimpleNamespace(
            **{
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "decode_context_parallel_size": 1,
                "prefill_context_parallel_size": 1,
                "data_parallel_size": 16,
                "enable_expert_parallel": True,
                **dict(parallel),
            }
        ),
        cache_config=types.SimpleNamespace(
            **{
                "kv_offloading_size": None,
                "kv_offloading_backend": "native",
                **dict(cache),
            }
        ),
        **{"speculative_config": None, "kv_transfer_config": None, **top},
    )


def _glm52_layer(**overrides):
    """A GLM-5.2 sparse MLA layer with its own indexer, with overrides."""
    attrs = dict(
        num_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        indexer=types.SimpleNamespace(topk_tokens=2048),
        topk_indices_buffer=None,
        impl=types.SimpleNamespace(non_causal_multi_token_decode=False),
    )
    attrs.update(overrides)
    return types.SimpleNamespace(**attrs)


def _specs(layout=KVCacheLayout.TENSORCORE):
    return (mock.Mock(layout=layout), mock.Mock(layout=layout))


class MaskedDenseAdmissionTest(parameterized.TestCase):
    """`admit_layer` and its configuration and layer checks.

    Admission is positive: only the validated GLM-5.2 configuration passes,
    and anything missing or unrecognized is rejected.
    """

    def setUp(self):
        super().setUp()
        self.enter_context(
            mock.patch.object(envs, "TPU_MLA_MASKED_DENSE_ENABLED", True)
        )
        self.enter_context(_masked_dense_limits(2048, 6144))

    def test_glm52_dp_ep_configuration_is_admitted(self):
        self.assertIsNone(mla_dispatch.configuration_rejection(_glm52_vllm_config()))

    @parameterized.named_parameters(
        dict(testcase_name="model_type", hf={"model_type": "deepseek_v32"}),
        dict(testcase_name="architecture", hf={"architectures": ["Other"]}),
        dict(testcase_name="no_architecture", hf={"architectures": None}),
        dict(testcase_name="heads", hf={"num_attention_heads": 128}),
        dict(testcase_name="latent", hf={"kv_lora_rank": 448}),
        dict(testcase_name="rope", hf={"qk_rope_head_dim": 32}),
        dict(testcase_name="topk", hf={"index_topk": 1024}),
        dict(testcase_name="no_topk", hf={"index_topk": None}),
        dict(testcase_name="sliding_window", hf={"sliding_window": 4096}),
        dict(testcase_name="tensor_parallel", parallel={"tensor_parallel_size": 2}),
        dict(testcase_name="pipeline_parallel", parallel={"pipeline_parallel_size": 2}),
        dict(
            testcase_name="decode_context_parallel",
            parallel={"decode_context_parallel_size": 2},
        ),
        dict(
            testcase_name="prefill_context_parallel",
            parallel={"prefill_context_parallel_size": 2},
        ),
        dict(
            testcase_name="unknown_parallel_size",
            parallel={"prefill_context_parallel_size": None},
        ),
        dict(testcase_name="speculative", speculative_config=object()),
        dict(testcase_name="kv_transfer", kv_transfer_config=object()),
        dict(testcase_name="kv_offloading", cache={"kv_offloading_size": 8.0}),
    )
    def test_other_configurations_are_rejected(self, **overrides):
        self.assertIsNotNone(
            mla_dispatch.configuration_rejection(_glm52_vllm_config(**overrides))
        )

    def test_missing_configuration_is_rejected(self):
        self.assertIsNotNone(mla_dispatch.configuration_rejection(None))
        config = _glm52_vllm_config()
        del config.parallel_config
        self.assertIsNotNone(mla_dispatch.configuration_rejection(config))

    def test_glm52_layer_is_admitted(self):
        self.assertIsNone(mla_dispatch.layer_rejection(_glm52_layer(), _specs(), 1024))

    def test_shared_indexer_layer_needs_the_forwarded_buffer(self):
        def rejection(buffer):
            layer = _glm52_layer(indexer=None, topk_indices_buffer=buffer)
            return mla_dispatch.layer_rejection(layer, _specs(), 1024)

        self.assertIsNone(rejection(np.empty((16, 2048), np.int32)))
        self.assertIsNotNone(rejection(None))
        self.assertIsNotNone(rejection(np.empty((16, 1024), np.int32)))

    @parameterized.named_parameters(
        dict(testcase_name="heads", num_heads=16),
        dict(testcase_name="latent", kv_lora_rank=448),
        dict(testcase_name="rope", qk_rope_head_dim=32),
        dict(testcase_name="no_heads", num_heads=None),
        dict(testcase_name="topk", indexer=types.SimpleNamespace(topk_tokens=1024)),
        dict(testcase_name="unknown_topk", indexer=types.SimpleNamespace()),
        dict(
            testcase_name="non_causal_impl",
            impl=types.SimpleNamespace(non_causal_multi_token_decode=True),
        ),
        dict(testcase_name="non_causal_layer", non_causal_multi_token_decode=True),
    )
    def test_other_layers_are_rejected(self, **overrides):
        self.assertIsNotNone(
            mla_dispatch.layer_rejection(_glm52_layer(**overrides), _specs(), 1024)
        )

    def test_other_caches_are_rejected(self):
        layer = _glm52_layer()
        sparsecore = _specs(KVCacheLayout.SPARSECORE)
        self.assertIsNotNone(mla_dispatch.layer_rejection(layer, sparsecore, 1024))
        self.assertIsNotNone(mla_dispatch.layer_rejection(layer, (), 1024))
        self.assertIsNotNone(mla_dispatch.layer_rejection(layer, _specs(), 512))

    def _admit(self, runner, config=None, specs=None, page_size=1024, **layer):
        module = _glm52_layer(**layer)
        runner.vllm_config = config or _glm52_vllm_config()
        mla_dispatch.admit_layer(runner, module, specs or _specs(), page_size)
        return module

    @staticmethod
    def _runner():
        return types.SimpleNamespace(_masked_dense_groups_requests=None)

    def test_one_disabled_tier_is_admitted(self):
        with _masked_dense_limits(0, 6144):
            self.assertIs(self._admit(self._runner()).allow_masked_dense, True)

    def test_every_tier_disabled_is_rejected(self):
        """Both tiers off: no route preparation and no request ordering."""
        runner = self._runner()
        with _masked_dense_limits(0, 0):
            self.assertIs(self._admit(runner).allow_masked_dense, False)
        self.assertIs(runner._masked_dense_groups_requests, False)

    def test_flag_off_records_nothing(self):
        """An invalid tuning value cannot fail a process that never routes."""
        runner = self._runner()
        with (
            mock.patch.object(envs, "TPU_MLA_MASKED_DENSE_ENABLED", False),
            mock.patch.object(
                mla_dispatch, "masked_dense_limits", side_effect=ValueError
            ) as limits,
        ):
            layer = self._admit(runner)
        limits.assert_not_called()
        self.assertFalse(hasattr(layer, "allow_masked_dense"))
        self.assertIsNone(runner._masked_dense_groups_requests)

    @parameterized.named_parameters(
        dict(
            testcase_name="decode_context_parallel",
            config=_glm52_vllm_config(parallel={"decode_context_parallel_size": 2}),
        ),
        dict(
            testcase_name="prefill_context_parallel",
            config=_glm52_vllm_config(parallel={"prefill_context_parallel_size": 2}),
        ),
        dict(
            testcase_name="speculative",
            config=_glm52_vllm_config(speculative_config=object()),
        ),
        dict(
            testcase_name="other_model",
            config=_glm52_vllm_config(hf={"index_topk": 1024}),
        ),
        dict(testcase_name="sparsecore", specs=_specs(KVCacheLayout.SPARSECORE)),
        dict(testcase_name="page_size", page_size=512),
        dict(testcase_name="heads", num_heads=16),
        dict(testcase_name="unknown_topk", indexer=None),
        dict(testcase_name="non_causal", non_causal_multi_token_decode=True),
    )
    def test_ineligible_layers_do_not_resolve_limits(self, **case):
        """A malformed tuning value cannot fail a flag-on process whose
        configuration or layer is not admitted."""
        runner = self._runner()
        with mock.patch.object(
            mla_dispatch, "masked_dense_limits", side_effect=ValueError
        ) as limits:
            self.assertIs(self._admit(runner, **case).allow_masked_dense, False)
        limits.assert_not_called()
        self.assertIs(runner._masked_dense_groups_requests, False)

    def test_invalid_tuning_fails_an_admitted_layer(self):
        """Once everything else admits the layer, a bad limit is an error."""
        _RESOLVE_LIMITS.cache_clear()
        self.addCleanup(_RESOLVE_LIMITS.cache_clear)
        with (
            mock.patch.object(mla_dispatch, "masked_dense_limits", _RESOLVE_LIMITS),
            mock.patch.dict(os.environ, {"TPU_MLA_MASKED_DENSE_MAX_KV_LEN": "1000"}),
            self.assertRaises(ValueError),
        ):
            self._admit(self._runner())

    def test_admitted_layers_allow_routing_and_ordering(self):
        runner = self._runner()
        for _ in range(2):
            self.assertIs(self._admit(runner).allow_masked_dense, True)
        self.assertIs(runner._masked_dense_groups_requests, True)

    def test_admission_is_logged_once_per_outcome(self):
        """Serving checks every worker for the admitted message and for the
        absence of the rejection warning."""
        with mock.patch.object(mla_dispatch, "logger") as logger:
            runner = self._runner()
            self._admit(runner)
            self._admit(runner)
        self.assertEqual(logger.info_once.call_count, 2)
        messages = {c.args for c in logger.info_once.call_args_list}
        self.assertLen(messages, 1)
        self.assertIn("admitted", next(iter(messages))[0])
        # Every worker reports, not only each host's first rank.
        for call in logger.info_once.call_args_list:
            self.assertEqual(call.kwargs, {"scope": "process"})
        logger.warning_once.assert_not_called()

        with mock.patch.object(mla_dispatch, "logger") as logger:
            self._admit(self._runner(), num_heads=16)
        logger.info_once.assert_not_called()
        logger.warning_once.assert_called_once()
        self.assertEqual(logger.warning_once.call_args.kwargs, {"scope": "process"})

    def test_rejected_configuration_allows_neither(self):
        runner = self._runner()
        speculative = _glm52_vllm_config(speculative_config=object())
        self.assertIs(self._admit(runner, speculative).allow_masked_dense, False)
        self.assertIs(runner._masked_dense_groups_requests, False)

    def test_mixed_layers_keep_request_order(self):
        """A rejected layer keeps ordering off even if a later one passes."""
        runner = self._runner()
        self.assertTrue(self._admit(runner).allow_masked_dense)
        self.assertFalse(self._admit(runner, num_heads=16).allow_masked_dense)
        self.assertTrue(self._admit(runner).allow_masked_dense)
        self.assertIs(runner._masked_dense_groups_requests, False)
