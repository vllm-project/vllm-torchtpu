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
"""`fori_min_chunks` must change the sparse MLA program, not its result.

The lowering tests run on CPU: they lower the kernel for TPU and count the
attention kernels and while loops, so they show which path a call takes. The
parity test needs a TPU; it compares both paths bit for bit, which the
lowering tests keep from passing vacuously.
"""

import math
import re
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

try:  # jax 0.11 moved the TPU info out of Pallas; Pallas re-exports it.
    from jax._src import tpu_info
except ImportError:
    from jax._src.pallas.mosaic import tpu_info

from tests.kernels.test_masked_dense_mla import TOKEN_PAD, MaskedDenseTestBase
from vllm_torchtpu.kernels.mla import dispatch as mla_dispatch
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel

CHUNK = sparse_mla_kernel.DEFAULT_GATHER_AND_ATTENTION_CHUNK_SIZE


def _lowered_text(num_tokens, cache_layout, split_q=False, **kwargs):
    if cache_layout == sparse_mla_kernel.SPARSECORE_CACHE_LAYOUT:
        nope = jax.ShapeDtypeStruct((64, 32, 128), jnp.uint32)
        rope = jax.ShapeDtypeStruct((64, 8, 128), jnp.uint32)
    else:
        nope = jax.ShapeDtypeStruct((64, 32, 4, 128), jnp.uint8)
        rope = jax.ShapeDtypeStruct((64, 8, 4, 128), jnp.uint8)
    # Split q: q_nope [T, H, 512] plus q_pe [T, H, 64] as an extra operand.
    q_dim = 512 if split_q else 576
    args = (
        jax.ShapeDtypeStruct((num_tokens, 16, q_dim), jnp.bfloat16),
        nope,
        rope,
        jax.ShapeDtypeStruct((num_tokens, 2048), jnp.int32),
        jax.ShapeDtypeStruct((8,), jnp.int32),
        jax.ShapeDtypeStruct((2,), jnp.int32),
        jax.ShapeDtypeStruct((3,), jnp.int32),
    )

    if split_q:
        args += (jax.ShapeDtypeStruct((num_tokens, 16, 64), jnp.bfloat16),)

    def attend(*a):
        if split_q:
            *a, q_pe = a
            kwargs["q_pe"] = q_pe
        return sparse_mla_kernel.sparse_ragged_paged_attention(
            *a, sm_scale=0.1, cache_layout=cache_layout, **kwargs
        )

    # Pallas asks the default device what it lowers for; answer TPU7x.
    tpu_info.get_tpu_info.cache_clear()
    try:
        with (
            mock.patch.object(tpu_info, "get_device_kind", return_value="TPU7x"),
            mock.patch.object(tpu_info, "get_num_device_cores", return_value=1),
        ):
            lowered = jax.jit(attend).trace(*args).lower(lowering_platforms=("tpu",))
            return lowered.as_text()
    finally:
        tpu_info.get_tpu_info.cache_clear()


def _program_shape(text):
    return (
        text.count("stablehlo.while"),
        len(re.findall(r'kernel_name = "MLA-', text)),
    )


class SparseChunkForiLoweringTest(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(testcase_name="tensorcore", layout="tensorcore", kwargs={}),
        dict(testcase_name="sparsecore", layout="sparsecore", kwargs={}),
        dict(
            testcase_name="sparsecore_topk_cap",
            layout="sparsecore",
            kwargs=dict(topk_cap=512),
        ),
        dict(
            testcase_name="tensorcore_lse",
            layout="tensorcore",
            kwargs=dict(return_lse=True),
        ),
        # TPU_MLA_SPARSE_SPLIT_Q: q_pe joins the chunk operands, lse lane dense.
        dict(
            testcase_name="sparsecore_split_q_lse",
            layout="sparsecore",
            kwargs=dict(split_q=True, return_lse=True, lse_lane_dense=True),
        ),
    )
    def test_loop_holds_one_chunk(self, layout, kwargs):
        num_tokens = 4 * CHUNK
        unrolled = _lowered_text(num_tokens, layout, **kwargs)
        fori = _lowered_text(num_tokens, layout, fori_min_chunks=2, **kwargs)
        self.assertEqual(_program_shape(unrolled), (0, 4))
        self.assertEqual(_program_shape(fori), (1, 1))

    def test_partial_last_chunk_stays_unrolled(self):
        text = _lowered_text(2 * CHUNK + TOKEN_PAD, "tensorcore", fori_min_chunks=2)
        self.assertEqual(_program_shape(text), (1, 2))

    def test_unroll_below_the_chunk_count_keeps_the_loop(self):
        # The unrolled copies call one lowered body, so count loops, not kernels.
        text = _lowered_text(4 * CHUNK, "tensorcore", fori_min_chunks=2, fori_unroll=2)
        self.assertEqual(text.count("stablehlo.while"), 1)

    def test_unroll_at_the_chunk_count_leaves_no_loop(self):
        text = _lowered_text(4 * CHUNK, "tensorcore", fori_min_chunks=2, fori_unroll=4)
        self.assertEqual(text.count("stablehlo.while"), 0)

    @parameterized.named_parameters(
        dict(testcase_name="tensorcore", layout="tensorcore"),
        dict(testcase_name="sparsecore", layout="sparsecore"),
    )
    def test_loop_ties_the_caches_to_the_index(self, layout):
        # Without the barrier the looped program held a cache-sized buffer per
        # layer for the SparseCore RoPE cache's [rows, 32] view, and ran out of
        # HBM at the default KV cache size.
        unrolled = _lowered_text(4 * CHUNK, layout)
        fori = _lowered_text(4 * CHUNK, layout, fori_min_chunks=2)
        self.assertEqual(unrolled.count("stablehlo.optimization_barrier"), 0)
        self.assertEqual(fori.count("stablehlo.optimization_barrier"), 1)

    def test_below_min_chunks_stays_unrolled(self):
        text = _lowered_text(4 * CHUNK, "tensorcore", fori_min_chunks=5)
        self.assertEqual(_program_shape(text), (0, 4))


class SparsePrefixChunkBoundsTest(parameterized.TestCase):
    """The masked-dense route's sparse prefix never hands a chunk past the rows."""

    @parameterized.parameters(
        dict(num_rows=41, fori_min_chunks=None),
        dict(num_rows=3 * CHUNK + 5, fori_min_chunks=None),
        dict(num_rows=3 * CHUNK + 5, fori_min_chunks=2),
    )
    def test_chunks_stay_inside_the_rows(self, num_rows, fori_min_chunks):
        spans = []

        def run_chunk(q, start_pos, chunk_size, token_end, caches):
            spans.append((start_pos, chunk_size))
            return q

        # Under one chunk of suffix minimum, the rounded-up starts overrun.
        with mock.patch.object(mla_dispatch, "MASKED_DENSE_MIN_SUFFIX_TOKENS", 0):
            jax.make_jaxpr(
                lambda q, split: mla_dispatch._sparse_prefix(
                    q, split, run_chunk, (), fori_min_chunks=fori_min_chunks
                )
            )(jnp.zeros((num_rows, 1)), jnp.int32(num_rows))
        covered = 0
        for start, size in spans:
            if isinstance(start, int):
                self.assertLessEqual(start + size, num_rows)
                covered = max(covered, start + size)
            else:  # One traced fori body stands for every full chunk.
                covered = max(covered, (num_rows // CHUNK) * CHUNK)
        self.assertEqual(covered, num_rows)


@pytest.mark.tpu_test
class SparseChunkForiParityTest(MaskedDenseTestBase):
    def _inputs(self, q_lens, seq_lens, num_tokens):
        cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        block_tables = self._block_tables(len(q_lens))
        caches, _, _ = self._fill_cache(seq_lens, block_tables, 1.0)
        seq_of_token = np.full(num_tokens, -1, np.int32)
        for s, q_len in enumerate(q_lens):
            seq_of_token[cu_q_lens[s] : cu_q_lens[s] + q_len] = s
        topk_rows = self._topk_rows(
            num_tokens,
            np.where(seq_of_token < 0, 0, seq_of_token),
            seq_lens,
            64,
            lambda kv_len: max(1, kv_len // 3),
        )
        return (
            self._random_queries(num_tokens),
            *caches,
            jnp.asarray(topk_rows, jnp.int32),
            block_tables,
            jnp.asarray(cu_q_lens, jnp.int32),
        )

    @parameterized.named_parameters(
        dict(testcase_name="prefill", q_lens=[48], seq_lens=[160], num_decode=0),
        dict(
            testcase_name="decode",
            q_lens=[1, 1, 1],
            seq_lens=[97, 224, 40],
            num_decode=3,
        ),
        dict(
            testcase_name="mixed",
            q_lens=[1, 1, 30, 20],
            seq_lens=[97, 224, 130, 60],
            num_decode=2,
        ),
    )
    def test_fori_matches_unrolled(self, q_lens, seq_lens, num_decode):
        chunk = 16
        whole = math.ceil(sum(q_lens) / chunk) * chunk
        distribution = jnp.asarray([num_decode, num_decode, len(q_lens)], jnp.int32)
        # A whole number of chunks, and one with a partial last chunk.
        for num_tokens in (max(whole, 2 * chunk), max(whole, 2 * chunk) + 8):
            args = self._inputs(q_lens, seq_lens, num_tokens)
            for return_lse in (False, True):
                outs = [
                    sparse_mla_kernel.sparse_ragged_paged_attention(
                        *args,
                        distribution,
                        sm_scale=self.sm_scale,
                        gather_and_attention_chunk_size=chunk,
                        return_lse=return_lse,
                        fori_min_chunks=fori_min_chunks,
                    )
                    for fori_min_chunks in (None, 2)
                ]
                with self.subTest(num_tokens=num_tokens, return_lse=return_lse):
                    for a, b in zip(jax.tree.leaves(outs[0]), jax.tree.leaves(outs[1])):
                        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


if __name__ == "__main__":
    absltest.main()
