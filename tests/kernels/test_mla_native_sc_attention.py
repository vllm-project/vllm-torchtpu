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
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.kernels.mla.sparse import native_sc_cache_layout

pytestmark = pytest.mark.tpu_test

NOPE_DIM = 512
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM


def _build_problem(rng, *, k_scale, query_tokens=128, topk=256, num_heads=8):
    page_size = 256
    cache_tokens = 8192
    pages = cache_tokens // page_size

    def quantized_bytes(dim):
        values = rng.random((pages, page_size, dim), dtype=np.float32) * 2 - 1
        fp8 = jnp.asarray(values / k_scale).astype(jnp.float8_e4m3fn)
        return np.asarray(jax.lax.bitcast_convert_type(fp8, jnp.uint8))

    nope_bytes = quantized_bytes(NOPE_DIM)
    rope_bytes = quantized_bytes(ROPE_DIM)
    tensorcore_nope = nope_bytes.reshape(pages, page_size, 4, 128)
    tensorcore_rope = np.pad(rope_bytes, ((0, 0), (0, 0), (0, ROPE_DIM))).reshape(
        pages, page_size // 4, 4, 128
    )

    page_table = rng.permutation(pages).astype(np.int32)
    inverse_page_table = np.argsort(page_table)
    physical = np.stack(
        [
            rng.choice(cache_tokens, size=topk, replace=False)
            for _ in range(query_tokens)
        ]
    ).astype(np.int32)
    logical = (
        inverse_page_table[physical // page_size] * page_size + physical % page_size
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
        "topk": jnp.asarray(logical),
        "physical_topk": jnp.asarray(physical),
        "page_table": jnp.asarray(page_table),
        "cu_q_lens": jnp.array([0, query_tokens], jnp.int32),
        "distribution": jnp.array([0, 1, 1], jnp.int32),
    }


class SparseMlaCacheLayoutsTest(parameterized.TestCase):
    @parameterized.product(k_scale=(0.75, 1.0), return_lse=(False, True))
    def test_sparsecore_layout_matches_tensorcore_layout(self, k_scale, return_lse):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore cache-layout attention requires a TPU")
        problem = _build_problem(np.random.default_rng(20260830), k_scale=k_scale)
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
            (tensorcore_out, sparsecore_out)
        )

        if return_lse:
            tensorcore_out, tensorcore_lse = tensorcore_out
            sparsecore_out, sparsecore_lse = sparsecore_out
            np.testing.assert_allclose(
                np.asarray(sparsecore_lse),
                np.asarray(tensorcore_lse),
                rtol=2e-2,
                atol=2e-2,
            )

        np.testing.assert_allclose(
            np.asarray(sparsecore_out, dtype=np.float32),
            np.asarray(tensorcore_out, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

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
            problem["q"],
            problem["tensorcore_nope"],
            problem["tensorcore_rope"],
            logical_topk,
            problem["page_table"],
            problem["cu_q_lens"],
            problem["distribution"],
            **kwargs,
        )
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
                (paged_out, tensorcore_physical_out, sparsecore_physical_out)
            )
        )

        for physical_out in (tensorcore_physical_out, sparsecore_physical_out):
            np.testing.assert_allclose(
                np.asarray(physical_out, dtype=np.float32),
                np.asarray(paged_out, dtype=np.float32),
                rtol=2e-2,
                atol=2e-2,
            )

    def test_sparsecore_layout_supports_short_query_chunk(self):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore cache-layout attention requires a TPU")
        problem = _build_problem(
            np.random.default_rng(20260903), k_scale=1.0, query_tokens=16, topk=2048
        )
        kwargs = {
            "sm_scale": 0.125,
            "gather_and_attention_chunk_size": 64,
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
            **kwargs,
        )
        tensorcore_out, sparsecore_out = jax.block_until_ready(
            (tensorcore_out, sparsecore_out)
        )

        np.testing.assert_allclose(
            np.asarray(sparsecore_out, dtype=np.float32),
            np.asarray(tensorcore_out, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )

    @parameterized.product(real_tokens=(1, 5, 16), return_lse=(False, True))
    def test_decode_bucket_real_rows_match_reference(self, real_tokens, return_lse):
        """The decode bucket (T=16, topk=2048) with only `real_tokens` live.

        Chunks of <= `small_chunk_tokens` tokens default to a 1-token attention
        grid step and a SparseCore gather bounded to the live rows, so padding
        rows are neither gathered nor attended. The real rows must match both
        the TensorCore layout and the SparseCore path with that bound disabled.
        """
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore cache-layout attention requires a TPU")
        problem = _build_problem(
            np.random.default_rng(20260925), k_scale=0.75, query_tokens=16, topk=2048
        )
        cu_q_lens = jnp.array([0, real_tokens], jnp.int32)
        # One sequence: decode-only for a single token, prefill-only otherwise.
        distribution = (
            jnp.array([1, 1, 1], jnp.int32)
            if real_tokens == 1
            else jnp.array([0, 1, 1], jnp.int32)
        )
        kwargs = {
            "sm_scale": 0.125,
            "k_scale": 0.75,
            "return_lse": return_lse,
            "gather_and_attention_chunk_size": 64,
            "attention_kernel_batch_size": 16,
        }

        def run(nope, rope, **extra):
            return sparse_mla.sparse_ragged_paged_attention(
                problem["q"],
                nope,
                rope,
                problem["topk"],
                problem["page_table"],
                cu_q_lens,
                distribution,
                **kwargs,
                **extra,
            )

        tensorcore_out = run(problem["tensorcore_nope"], problem["tensorcore_rope"])
        sparsecore_nope = problem["sparsecore_nope"]
        sparsecore_rope = problem["sparsecore_rope"]
        bounded_out = run(
            sparsecore_nope,
            sparsecore_rope,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
        )
        unbounded_out = run(
            sparsecore_nope,
            sparsecore_rope,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            small_chunk_tokens=0,
        )
        tensorcore_out, bounded_out, unbounded_out = jax.block_until_ready(
            (tensorcore_out, bounded_out, unbounded_out)
        )
        if return_lse:
            (tensorcore_out, tensorcore_lse) = tensorcore_out
            (bounded_out, bounded_lse) = bounded_out
            (unbounded_out, unbounded_lse) = unbounded_out
            for lse in (bounded_lse, unbounded_lse):
                np.testing.assert_allclose(
                    np.asarray(lse)[:real_tokens],
                    np.asarray(tensorcore_lse)[:real_tokens],
                    rtol=2e-2,
                    atol=2e-2,
                )

        real = np.asarray(bounded_out, dtype=np.float32)[:real_tokens]
        for reference in (tensorcore_out, unbounded_out):
            np.testing.assert_allclose(
                real,
                np.asarray(reference, dtype=np.float32)[:real_tokens],
                rtol=2e-2,
                atol=2e-2,
            )
        # Positive control: the comparison must be able to fail.
        self.assertGreater(
            np.abs(real - np.asarray(problem["q"], np.float32)[:real_tokens]).max(),
            2e-2,
        )

    @parameterized.parameters(1, 100)
    def test_mixed_step_bucket_real_rows_match_reference(self, real_tokens):
        """A DP-padded mixed step: the T=1024 bucket with `real_tokens` live.

        Chunks larger than `small_chunk_tokens` also bound the SparseCore
        gather to the live rows by default (`live_gather_all_chunks`), so fully
        dead chunks issue no gather. The real rows must match both the
        TensorCore layout and the SparseCore path with that bound disabled.
        """
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore cache-layout attention requires a TPU")
        problem = _build_problem(
            np.random.default_rng(20260927),
            k_scale=0.75,
            query_tokens=1024,
            topk=2048,
        )
        cu_q_lens = jnp.array([0, real_tokens], jnp.int32)
        distribution = (
            jnp.array([1, 1, 1], jnp.int32)
            if real_tokens == 1
            else jnp.array([0, 1, 1], jnp.int32)
        )
        kwargs = {
            "sm_scale": 0.125,
            "k_scale": 0.75,
            "gather_and_attention_chunk_size": 64,
            "attention_kernel_batch_size": 16,
        }

        def run(nope, rope, **extra):
            return sparse_mla.sparse_ragged_paged_attention(
                problem["q"],
                nope,
                rope,
                problem["topk"],
                problem["page_table"],
                cu_q_lens,
                distribution,
                **kwargs,
                **extra,
            )

        tensorcore_out = run(problem["tensorcore_nope"], problem["tensorcore_rope"])
        sc = {"cache_layout": sparse_mla.SPARSECORE_CACHE_LAYOUT}
        bounded_out = run(problem["sparsecore_nope"], problem["sparsecore_rope"], **sc)
        unbounded_out = run(
            problem["sparsecore_nope"],
            problem["sparsecore_rope"],
            live_gather_all_chunks=False,
            **sc,
        )
        tensorcore_out, bounded_out, unbounded_out = jax.block_until_ready(
            (tensorcore_out, bounded_out, unbounded_out)
        )

        real = np.asarray(bounded_out, dtype=np.float32)[:real_tokens]
        for reference in (tensorcore_out, unbounded_out):
            np.testing.assert_allclose(
                real,
                np.asarray(reference, dtype=np.float32)[:real_tokens],
                rtol=2e-2,
                atol=2e-2,
            )
        # Positive control: the comparison must be able to fail.
        self.assertGreater(
            np.abs(real - np.asarray(problem["q"], np.float32)[:real_tokens]).max(),
            2e-2,
        )

    def test_decode_bucket_zero_real_tokens_then_decode(self):
        """An idle rank's dummy step: T=16, no real token, all top-k `-1`.

        The attention grid has zero steps and the bounded gather takes its
        zero-trip path. Outputs are don't-care, but the launch must complete,
        and a normal decode afterwards in the same process -- twice -- must
        still be right: an unwaited DMA would halt the core and fail it.
        """
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore cache-layout attention requires a TPU")
        problem = _build_problem(
            np.random.default_rng(20260926), k_scale=1.0, query_tokens=16, topk=2048
        )
        kwargs = {
            "sm_scale": 0.125,
            "gather_and_attention_chunk_size": 64,
            "attention_kernel_batch_size": 16,
        }
        sc = {
            "cache_layout": sparse_mla.SPARSECORE_CACHE_LAYOUT,
            **kwargs,
        }

        idle = sparse_mla.sparse_ragged_paged_attention(
            problem["q"],
            problem["sparsecore_nope"],
            problem["sparsecore_rope"],
            jnp.full_like(problem["topk"], -1),
            problem["page_table"],
            jnp.array([0, 0], jnp.int32),
            jnp.array([0, 0, 0], jnp.int32),
            **sc,
        )
        jax.block_until_ready(idle)

        cu_q_lens = jnp.array([0, 1], jnp.int32)
        distribution = jnp.array([1, 1, 1], jnp.int32)
        reference = sparse_mla.sparse_ragged_paged_attention(
            problem["q"],
            problem["tensorcore_nope"],
            problem["tensorcore_rope"],
            problem["topk"],
            problem["page_table"],
            cu_q_lens,
            distribution,
            **kwargs,
        )
        reference = np.asarray(jax.block_until_ready(reference), np.float32)[:1]
        for _ in range(2):
            out = sparse_mla.sparse_ragged_paged_attention(
                problem["q"],
                problem["sparsecore_nope"],
                problem["sparsecore_rope"],
                problem["topk"],
                problem["page_table"],
                cu_q_lens,
                distribution,
                **sc,
            )
            np.testing.assert_allclose(
                np.asarray(jax.block_until_ready(out), np.float32)[:1],
                reference,
                rtol=2e-2,
                atol=2e-2,
            )

    @parameterized.product(
        (
            {"query_tokens": 16, "real_tokens": 13, "chunk": 64, "late": False},
            {"query_tokens": 128, "real_tokens": 100, "chunk": 64, "late": False},
            {"query_tokens": 128, "real_tokens": 128, "chunk": 128, "late": False},
            # Only the last chunk overflows the cap.
            {"query_tokens": 128, "real_tokens": 128, "chunk": 64, "late": True},
        ),
        topk_cap=(256, 512),
    )
    def test_topk_cap_matches_uncapped(
        self, query_tokens, real_tokens, chunk, topk_cap, late
    ):
        """`topk_cap` against the uncapped SparseCore path, per real row.

        Counts span 1 to topk, so chunks hold rows within the cap, rows one
        past it, and rows reaching every window (2048), i.e. the in-kernel
        multi-window fix-up and the window-major live bound. `late` caps every
        row before the last chunk, so earlier chunks gather fewer windows than
        the once-per-call window prep (`capped_window_indices`) translates.
        """
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore cache-layout attention requires a TPU")
        topk = 2048
        rng = np.random.default_rng(20261003)
        problem = _build_problem(
            rng, k_scale=0.75, query_tokens=query_tokens, topk=topk
        )
        counts = np.resize(
            np.array([1, 63, 255, 256, 257, 511, 512, 513, 700, 1024, 1025, 2048, 130]),
            query_tokens,
        )
        rng.shuffle(counts)
        if late:
            counts[: query_tokens - chunk] = np.minimum(
                counts[: query_tokens - chunk], topk_cap
            )
        live = np.arange(topk)[None, :] < counts[:, None]
        topk_idx = jnp.where(jnp.asarray(live), problem["topk"], -1)
        cu_q_lens = jnp.array([0, real_tokens], jnp.int32)
        distribution = jnp.array([0, 1, 1], jnp.int32)

        def run(**extra):
            return sparse_mla.sparse_ragged_paged_attention(
                problem["q"],
                problem["sparsecore_nope"],
                problem["sparsecore_rope"],
                topk_idx,
                problem["page_table"],
                cu_q_lens,
                distribution,
                sm_scale=0.125,
                k_scale=0.75,
                return_lse=True,
                cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
                gather_and_attention_chunk_size=chunk,
                attention_kernel_batch_size=16,
                **extra,
            )

        # The capped run takes the window prep, not the full-width fallback.
        capped_jaxpr = str(jax.make_jaxpr(lambda: run(topk_cap=topk_cap))())
        self.assertIn("capped_window_indices", capped_jaxpr)
        (ref, ref_lse), (out, lse) = jax.block_until_ready(
            (run(), run(topk_cap=topk_cap))
        )
        real = np.asarray(out, np.float32)[:real_tokens]
        np.testing.assert_allclose(
            real, np.asarray(ref, np.float32)[:real_tokens], rtol=2e-2, atol=2e-2
        )
        np.testing.assert_allclose(
            np.asarray(lse)[:real_tokens],
            np.asarray(ref_lse)[:real_tokens],
            rtol=1e-4,
            atol=1e-4,
        )
        # Positive control: the comparison must be able to fail.
        self.assertGreater(
            np.abs(real - np.asarray(problem["q"], np.float32)[:real_tokens]).max(),
            2e-2,
        )

    @parameterized.product(
        (
            {"page_size": 256, "total_pages": 2085, "pages_per_seq": 256},
            {"page_size": 96, "total_pages": 1001, "pages_per_seq": 300},
        ),
        (
            {"num_tokens": 16, "chunk": 16, "topk": 2048, "cap": 512},
            {"num_tokens": 512, "chunk": 256, "topk": 2048, "cap": 512},
            {"num_tokens": 128, "chunk": 64, "topk": 512, "cap": 128},
        ),
        block_tokens=(None, 8),
    )
    def test_capped_window_indices_match_reference(
        self,
        page_size,
        total_pages,
        pages_per_seq,
        num_tokens,
        chunk,
        topk,
        cap,
        block_tokens,
    ):
        """`prepare_capped_window_indices` == window-major `prepare_gather_indices`.

        Every live window, bit for bit, incl. the `-1` scatter (non-power-of-2
        page counts and page sizes take the in-kernel f32 divmod).
        """
        if jax.devices()[0].platform != "tpu":
            self.skipTest("Mosaic kernel requires a TPU")
        rng = np.random.default_rng(num_tokens + topk + page_size)
        num_seqs = 3
        page_table = jnp.asarray(
            rng.integers(0, total_pages, size=num_seqs * pages_per_seq), jnp.int32
        )
        cu_q_lens = jnp.array(
            [0, num_tokens // 4, num_tokens // 2, num_tokens], jnp.int32
        )
        counts = rng.integers(0, topk + 1, size=num_tokens)
        logical = rng.integers(0, pages_per_seq * page_size, size=(num_tokens, topk))
        topk_idx = jnp.asarray(
            np.where(np.arange(topk)[None] < counts[:, None], logical, -1), jnp.int32
        )
        _, phys = sparse_mla.prepare_gather_indices(
            topk_idx,
            page_table,
            cu_q_lens,
            page_size=page_size,
            total_num_pages=total_pages,
        )
        windows = topk // cap
        want = (
            np.asarray(phys)
            .reshape(num_tokens // chunk, chunk, windows, cap)
            .transpose(0, 2, 1, 3)
        )

        def got(lw, total=total_pages):
            out = sparse_mla.prepare_capped_window_indices(
                topk_idx,
                page_table,
                cu_q_lens,
                jnp.int32(lw),
                chunk_size=chunk,
                topk_cap=cap,
                page_size=page_size,
                total_num_pages=total,
                block_tokens=block_tokens,
            )
            return np.asarray(out).reshape(num_tokens // chunk, windows, chunk, cap)

        for lw in range(1, windows + 1):
            np.testing.assert_array_equal(got(lw)[:, :lw], want[:, :lw])
        # Positive control: a different scatter modulus moves the padding rows.
        self.assertFalse(np.array_equal(got(windows, total_pages + 1), want))

    @parameterized.product(
        (
            # Decode buckets (MTP 3: q=4 per request) and a mixed step.
            {"query_tokens": 16, "seq_q": (4, 4)},
            {"query_tokens": 512, "seq_q": (4, 4, 492)},
        ),
        cache_layout=(
            sparse_mla.SPARSECORE_CACHE_LAYOUT,
            sparse_mla.TENSORCORE_CACHE_LAYOUT,
        ),
    )
    def test_split_q_lse_matches_concat(self, query_tokens, seq_q, cache_layout):
        """TPU_MLA_SPARSE_SPLIT_Q on (`q_pe=`, `lse_lane_dense=True`) == off.

        DCP-shaped: 64 heads, per-row live counts incl. empty rows and rows
        past the 512 cap. Out and lse must match bit for bit.
        """
        if jax.devices()[0].platform != "tpu":
            self.skipTest("Sparse MLA attention requires a TPU")
        rng = np.random.default_rng(20261006)
        topk = 2048
        problem = _build_problem(
            rng, k_scale=0.75, query_tokens=query_tokens, topk=topk, num_heads=64
        )
        counts = rng.integers(0, topk + 1, size=query_tokens)
        counts[0], counts[1] = 0, 1025  # an empty row and an overflow row
        live = np.arange(topk)[None, :] < counts[:, None]
        topk_idx = jnp.where(jnp.asarray(live), problem["topk"], -1)
        # As `sparse_mla_attention_dcp`: a row owning none of its top-k gets
        # key 0, and `owns` masks its lse afterwards.
        owns = jnp.any(topk_idx >= 0, axis=-1)
        topk_idx = topk_idx.at[:, 0].set(jnp.where(owns, topk_idx[:, 0], 0))
        num_seqs = len(seq_q)
        page_table = jnp.tile(problem["page_table"], num_seqs)
        cu_q_lens = jnp.asarray(np.concatenate([[0], np.cumsum(seq_q)]), jnp.int32)
        num_decode = sum(1 for n in seq_q if n == 4)
        distribution = jnp.array([num_decode, num_decode, num_seqs], jnp.int32)
        sparsecore = cache_layout == sparse_mla.SPARSECORE_CACHE_LAYOUT
        q_nope = problem["q"][..., :NOPE_DIM]
        q_pe = problem["q"][..., NOPE_DIM:]

        def run(split):
            out, lse = sparse_mla.sparse_ragged_paged_attention(
                q_nope if split else problem["q"],
                problem[f"{cache_layout}_nope"],
                problem[f"{cache_layout}_rope"],
                topk_idx,
                page_table,
                cu_q_lens,
                distribution,
                sm_scale=0.125,
                k_scale=0.75,
                return_lse=True,
                cache_layout=cache_layout,
                **({"topk_cap": 512} if sparsecore else {}),
                **({"q_pe": q_pe, "lse_lane_dense": True} if split else {}),
            )
            return out[..., :NOPE_DIM], jnp.where(owns[:, None], lse, -jnp.inf)

        (ref, ref_lse), (out, lse) = jax.block_until_ready((run(False), run(True)))
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(out, jnp.uint16)),
            np.asarray(jax.lax.bitcast_convert_type(ref, jnp.uint16)),
        )
        np.testing.assert_array_equal(
            np.asarray(lse).view(np.uint32), np.asarray(ref_lse).view(np.uint32)
        )

    def test_topk_cap_validation(self):
        q = jnp.zeros((16, 8, HEAD_DIM), jnp.bfloat16)
        nope = jnp.zeros((16, 256, 128), jnp.uint32)
        rope = jnp.zeros((16, 64, 128), jnp.uint32)
        topk_idx = jnp.zeros((16, 256), jnp.int32)
        for bad_cap in (96, 192):
            with self.assertRaisesRegex(ValueError, "topk_cap"):
                sparse_mla.sparse_ragged_paged_attention(
                    q,
                    nope,
                    rope,
                    topk_idx,
                    jnp.arange(16, dtype=jnp.int32),
                    jnp.array([0, 16], jnp.int32),
                    jnp.array([0, 1, 1], jnp.int32),
                    cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
                    topk_cap=bad_cap,
                )

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
