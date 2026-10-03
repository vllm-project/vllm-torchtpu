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
"""The `-1` tail of a top-k row must not reach the softmax.

Under DCP most of a token's top-k row is `-1` padding, gathered from scattered
pages and masked away twice: the keys are zeroed (the tail decodes to fp8 NaN,
and `0 * NaN = NaN` in the PV einsum) and the scores are masked (a zeroed key
still scores 0 and inflates the softmax denominator).

The existing padding coverage compares padded runs only against other padded
runs of the same kernel, so dropping either mask moves every side of that
comparison together and stays green. These tests compare a padded row against
a *narrower, unpadded* row instead -- a configuration that cannot share the
defect -- so the equality holds only if the tail really was excluded.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout

NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
TOPK = 256

LAYOUTS = (None, sparse_mla.SPARSECORE_CACHE_LAYOUT)


def _build_problem(rng, *, query_tokens=128, topk=TOPK):
    page_size = 256
    cache_tokens = 8192
    pages = cache_tokens // page_size
    num_heads = 8

    def quantized_bytes(dim):
        values = rng.random((pages, page_size, dim), dtype=np.float32) * 2 - 1
        fp8 = jnp.asarray(values).astype(jnp.float8_e4m3fn)
        return np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))

    tensorcore_nope = quantized_bytes(NOPE_DIM).reshape(pages, page_size, 4, 128)
    tensorcore_rope = np.pad(
        quantized_bytes(ROPE_DIM), ((0, 0), (0, 0), (0, ROPE_DIM))
    ).reshape(pages, page_size // 4, 4, 128)

    physical = np.stack(
        [
            rng.choice(cache_tokens, size=topk, replace=False)
            for _ in range(query_tokens)
        ]
    ).astype(np.int32)
    q = jnp.asarray(
        rng.normal(size=(query_tokens, num_heads, HEAD_DIM)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    return {
        "q": q,
        "tensorcore_nope": jnp.asarray(tensorcore_nope),
        "tensorcore_rope": jnp.asarray(tensorcore_rope),
        "sparsecore_nope": jnp.asarray(
            native_sc_cache_layout.pack_nope(tensorcore_nope)
        ),
        "sparsecore_rope": jnp.asarray(
            native_sc_cache_layout.pack_rope_banded(tensorcore_rope)
        ),
        "physical_topk": jnp.asarray(physical),
        "cu_q_lens": jnp.array([0, query_tokens], jnp.int32),
        "distribution": jnp.array([0, 1, 1], jnp.int32),
    }


def _build_poisoned_problem(rng, *, query_tokens=128, topk=TOPK, winner_pages=4):
    """A cache that is fp8 NaN everywhere except the pages the winners live in.

    The hot-spot scatter rewrites every `-1` to a real page id, so the padded
    tail gathers whatever those pages hold. Poisoning the pages no winner
    occupies is what makes the tail decode to NaN -- which is the condition the
    key zeroing exists for, and which a cache of ordinary random bytes never
    reproduces.
    """
    page_size = 256
    pages = 32
    num_heads = 8

    def poisoned_bytes(dim):
        raw = np.full((pages, page_size, dim), 0xFF, np.uint8)
        values = rng.random((winner_pages, page_size, dim), dtype=np.float32) * 2 - 1
        fp8 = jnp.asarray(values).astype(jnp.float8_e4m3fn)
        raw[:winner_pages] = np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))
        return raw

    tensorcore_nope = poisoned_bytes(NOPE_DIM).reshape(pages, page_size, 4, 128)
    tensorcore_rope = np.pad(
        poisoned_bytes(ROPE_DIM), ((0, 0), (0, 0), (0, ROPE_DIM))
    ).reshape(pages, page_size // 4, 4, 128)

    physical = np.stack(
        [
            rng.choice(winner_pages * page_size, size=topk, replace=False)
            for _ in range(query_tokens)
        ]
    ).astype(np.int32)
    q = jnp.asarray(
        rng.normal(size=(query_tokens, num_heads, HEAD_DIM)).astype(np.float32),
        dtype=jnp.bfloat16,
    )
    return {
        "q": q,
        "tensorcore_nope": jnp.asarray(tensorcore_nope),
        "tensorcore_rope": jnp.asarray(tensorcore_rope),
        "sparsecore_nope": jnp.asarray(
            native_sc_cache_layout.pack_nope(tensorcore_nope)
        ),
        "sparsecore_rope": jnp.asarray(
            native_sc_cache_layout.pack_rope_banded(tensorcore_rope)
        ),
        "physical_topk": jnp.asarray(physical),
        "cu_q_lens": jnp.array([0, query_tokens], jnp.int32),
        "distribution": jnp.array([0, 1, 1], jnp.int32),
    }


def _attend(problem, indices, layout, *, return_lse=False, is_dcp_mode=False):
    """Physical-index attention over `indices`, on either cache layout."""
    sparsecore = layout == sparse_mla.SPARSECORE_CACHE_LAYOUT
    out = sparse_mla.sparse_ragged_paged_attention(
        problem["q"],
        problem["sparsecore_nope"] if sparsecore else problem["tensorcore_nope"],
        problem["sparsecore_rope"] if sparsecore else problem["tensorcore_rope"],
        indices,
        None,
        problem["cu_q_lens"],
        problem["distribution"],
        return_lse=return_lse,
        is_dcp_mode=is_dcp_mode,
        sm_scale=0.125,
        gather_and_attention_chunk_size=128,
        attention_kernel_batch_size=16,
        **({"cache_layout": layout} if sparsecore else {}),
    )
    return jax.block_until_ready(out)


class PaddedTopkMatchesNarrowTopkTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA attention requires a TPU")
        self.problem = _build_problem(np.random.default_rng(20261001))

    def _pair(self, kv_len):
        """The same kv_len winners, once padded to TOPK and once not."""
        indices = self.problem["physical_topk"]
        return indices[:, :kv_len], indices.at[:, kv_len:].set(-1)

    @parameterized.product(kv_len=(64, 128, 192), layout=LAYOUTS)
    def test_padding_does_not_change_the_output(self, kv_len, layout):
        narrow, padded = self._pair(kv_len)
        np.testing.assert_allclose(
            np.asarray(_attend(self.problem, padded, layout), dtype=np.float32),
            np.asarray(_attend(self.problem, narrow, layout), dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

    @parameterized.product(kv_len=(64, 128, 192), layout=LAYOUTS)
    def test_padding_does_not_change_the_lse(self, kv_len, layout):
        # The LSE is what a DCP peer merges on, so an inflated softmax
        # denominator corrupts the cross-rank result even when `out` looks sane.
        narrow, padded = self._pair(kv_len)
        _, narrow_lse = _attend(self.problem, narrow, layout, return_lse=True)
        _, padded_lse = _attend(self.problem, padded, layout, return_lse=True)
        np.testing.assert_allclose(
            np.asarray(padded_lse, dtype=np.float32),
            np.asarray(narrow_lse, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

    @parameterized.parameters(*LAYOUTS)
    def test_heavy_padding_stays_finite(self, layout):
        _, padded = self._pair(TOPK // 4)
        out = np.asarray(_attend(self.problem, padded, layout), dtype=np.float32)
        self.assertTrue(np.isfinite(out).all(), "padded tail leaked into the output")


class NanPaddingTailIsZeroedTest(parameterized.TestCase):
    """Score masking alone is not enough: `0 * NaN = NaN` in the PV einsum."""

    def setUp(self):
        super().setUp()
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA attention requires a TPU")
        self.problem = _build_poisoned_problem(np.random.default_rng(20261001))

    @parameterized.parameters(*LAYOUTS)
    def test_a_nan_tail_does_not_reach_the_output(self, layout):
        indices = self.problem["physical_topk"]
        narrow = indices[:, : TOPK // 4]
        padded = indices.at[:, TOPK // 4 :].set(-1)

        # The winners are finite, so an unpadded run must be clean -- otherwise
        # this asserts nothing about the tail.
        baseline = np.asarray(_attend(self.problem, narrow, layout), np.float32)
        self.assertTrue(np.isfinite(baseline).all(), "the winners are poisoned")

        out = np.asarray(_attend(self.problem, padded, layout), np.float32)
        self.assertTrue(np.isfinite(out).all(), "NaN tail reached the PV einsum")
        np.testing.assert_allclose(out, baseline, rtol=2e-2, atol=2e-2)


class DcpModeMatchesDefaultTest(parameterized.TestCase):
    """`is_dcp_mode=True` is an optimization, so it must not move the answer.

    It skips the gather DMAs for all-padding windows and narrows both the
    HBM->VMEM slab copy and the QK/PV loop to a rung chosen from the batch's
    widest `kv_len`. Every rung must agree with the unoptimized path exactly.

    `DCP_TOPK` is 512 so all four rungs are reachable: the eighth rung is gated
    on `topk % 8 == 0` at the fetch and `topk // 8 % 64 == 0` at the compute,
    and at topk=256 the second gate fails, so the ladder silently starts at a
    quarter and the narrowest slice never gets exercised.
    """

    DCP_TOPK = 512

    def setUp(self):
        super().setUp()
        if jax.devices()[0].platform != "tpu":
            self.skipTest("sparse MLA attention requires a TPU")
        self.problem = _build_problem(
            np.random.default_rng(20261002), topk=self.DCP_TOPK
        )

    def _padded(self, kv_len):
        return self.problem["physical_topk"].at[:, kv_len:].set(-1)

    # 64 -> eighth, 128 -> quarter, 256 -> half, 512 -> full width.
    @parameterized.product(kv_len=(64, 128, 256, 512), layout=LAYOUTS)
    def test_each_ladder_rung_matches_the_default_path(self, kv_len, layout):
        padded = self._padded(kv_len)
        dcp, dcp_lse = _attend(
            self.problem, padded, layout, return_lse=True, is_dcp_mode=True
        )
        ref, ref_lse = _attend(
            self.problem, padded, layout, return_lse=True, is_dcp_mode=False
        )
        np.testing.assert_allclose(
            np.asarray(dcp, np.float32),
            np.asarray(ref, np.float32),
            rtol=2e-2,
            atol=2e-2,
        )
        np.testing.assert_allclose(
            np.asarray(dcp_lse, np.float32),
            np.asarray(ref_lse, np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

    @parameterized.product(kv_len=(64, 128), layout=LAYOUTS)
    def test_a_narrowed_rung_still_excludes_the_tail(self, kv_len, layout):
        # Against a *narrower unpadded* row, not against itself: a rung that
        # over-fetched and then forgot to mask would match the default path
        # (which has the same defect-free masks) only by accident.
        narrow = self.problem["physical_topk"][:, :kv_len]
        np.testing.assert_allclose(
            np.asarray(
                _attend(self.problem, self._padded(kv_len), layout, is_dcp_mode=True),
                np.float32,
            ),
            np.asarray(_attend(self.problem, narrow, layout), np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

    @parameterized.parameters(*LAYOUTS)
    def test_a_ragged_batch_uses_the_widest_rung_without_overreading(self, layout):
        # The rung is picked from the batch maximum but the masks are per row,
        # so rows narrower than the rung must still exclude their own tail.
        indices = np.asarray(self.problem["physical_topk"]).copy()
        kv_lens = 64 * (1 + np.arange(indices.shape[0]) % 8)
        for row, kv_len in enumerate(kv_lens):
            indices[row, kv_len:] = -1
        padded = jnp.asarray(indices)

        np.testing.assert_allclose(
            np.asarray(
                _attend(self.problem, padded, layout, is_dcp_mode=True), np.float32
            ),
            np.asarray(_attend(self.problem, padded, layout), np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

    @parameterized.parameters(*LAYOUTS)
    def test_a_nan_tail_does_not_reach_the_output(self, layout):
        # The skipped gather windows are never written, so whatever VMEM holds
        # is what the masks have to cover. Poison it with fp8 NaN.
        problem = _build_poisoned_problem(
            np.random.default_rng(20261002), topk=self.DCP_TOPK, winner_pages=8
        )
        kv_len = self.DCP_TOPK // 8
        narrow = problem["physical_topk"][:, :kv_len]
        padded = problem["physical_topk"].at[:, kv_len:].set(-1)

        baseline = np.asarray(_attend(problem, narrow, layout), np.float32)
        self.assertTrue(np.isfinite(baseline).all(), "the winners are poisoned")

        out = np.asarray(_attend(problem, padded, layout, is_dcp_mode=True), np.float32)
        self.assertTrue(np.isfinite(out).all(), "NaN tail reached the PV einsum")
        np.testing.assert_allclose(out, baseline, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    absltest.main()
