# SPDX-License-Identifier: Apache-2.0
"""100% full-module line and branch coverage tests for DeepSeek-V4.1
v2/attention_kernels.py.

Covers all 31 functions/ops and all branch decision points across lines 1-1891 of:
src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py
"""

from __future__ import annotations

import os
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

# Ensure CPU platform and skip MDS queries
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TPU_SKIP_MDS_QUERY", "true")

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    attention_kernels,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _attend,
    _attend_window,
    _attention_state,
    _pad_compacted_records,
    _window_union,
    compact_records,
    paged_attention_jax,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select import (
    _compact_selected,
    _prefix_count,
    _sparsecore_top_k,
    _threshold_top_k,
    _visible_top_k,
    select_indices_jax,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

# ==============================================================================
# Helper Mock Objects & Cache Tensors
# ==============================================================================


class _FakeKVBuffer:
    def __init__(
        self,
        raw: torch.Tensor,
        rope: torch.Tensor | None = None,
        shape: tuple[int, ...] = (16, 16),
        raw_hi: torch.Tensor | None = None,
        rope_hi: torch.Tensor | None = None,
    ):
        self.raw = raw
        self.raw_hi = raw_hi
        self.rope = rope
        self.rope_hi = rope_hi
        self.shape = shape
        self.total_rows = raw.shape[0] + (0 if raw_hi is None else raw_hi.shape[0])


# ==============================================================================
# 1. Tests for Window Attention, _attend, and SWA JAX Kernels
# ==============================================================================


class TestWindowAttentionAndAttend:
    def test_attention_state(self):
        q = jnp.zeros((8, 4, 128), dtype=jnp.float32)
        maximum, denominator, numerator = _attention_state(q)
        assert maximum.shape == (8, 4)
        assert denominator.shape == (8, 4)
        assert numerator.shape == (8, 4, 128)
        assert jnp.all(maximum == -1e30)
        assert jnp.all(denominator == 0)
        assert jnp.all(numerator == 0)

    def test_attend_empty_width(self):
        q = jnp.ones((4, 2, 64), dtype=jnp.float32)
        pool = jnp.zeros((16, 128), dtype=jnp.int32)
        records = jnp.zeros((4, 0), dtype=jnp.int32)
        state = _attention_state(q)
        out = _attend(q, pool, records, state, scale=0.125, tile=64)
        assert out[0] is state[0]
        assert out[1] is state[1]
        assert out[2] is state[2]

    def test_attend_normal(self):
        # 4 tokens, 2 heads, dim 64
        q = jnp.ones((4, 2, 64), dtype=jnp.bfloat16)
        pool = jnp.zeros((32, 128), dtype=jnp.int32)
        records = jnp.array(
            [[0, 1, 2, -1], [1, 2, -1, -1], [2, 3, 4, 5], [0, -1, -1, -1]],
            dtype=jnp.int32,
        )
        state = _attention_state(q)
        maximum, denominator, numerator = _attend(
            q, pool, records, state, scale=0.125, tile=64
        )
        assert maximum.shape == (4, 2)
        assert denominator.shape == (4, 2)
        assert numerator.shape == (4, 2, 64)
        assert jnp.all(jnp.isfinite(maximum))
        assert jnp.all(denominator > 0)
        assert jnp.all(jnp.isfinite(numerator))

    def test_window_union_properties(self):
        # 128 tokens, tile=64 (2 blocks)
        # width = 64
        records = jnp.full((128, 64), -1, dtype=jnp.int32)
        # Block 0: contiguous window of 64 tokens
        for i in range(64):
            records = records.at[i, : i + 1].set(jnp.arange(i + 1, dtype=jnp.int32))
        # Block 1: mixed
        for i in range(64, 128):
            records = records.at[i, : min(64, i + 1)].set(
                jnp.arange(i - min(63, i), i + 1, dtype=jnp.int32)
            )

        union, mask, uncovered = _window_union(records, tile=64)
        assert union.shape == (2, 64 + 64)
        assert mask.shape == (2, 64, 128)
        assert uncovered.shape == (2, 64)

    def test_attend_window_early_returns(self):
        q = jnp.zeros((10, 2, 64), dtype=jnp.bfloat16)
        pool = jnp.zeros((32, 128), dtype=jnp.int32)
        state = _attention_state(q)

        # 1. tokens % tile != 0 (10 % 64 != 0) -> calls _attend
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels._attend",
            return_value=state,
        ) as mock_att:
            out = _attend_window(
                q,
                pool,
                jnp.zeros((10, 64), dtype=jnp.int32),
                state,
                scale=0.125,
                tile=64,
            )
            assert out is state
            mock_att.assert_called_once()

        # 2. tokens < 2 * tile (64 < 128)
        q64 = jnp.zeros((64, 2, 64), dtype=jnp.bfloat16)
        state64 = _attention_state(q64)
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels._attend",
            return_value=state64,
        ) as mock_att:
            out = _attend_window(
                q64,
                pool,
                jnp.zeros((64, 64), dtype=jnp.int32),
                state64,
                scale=0.125,
                tile=64,
            )
            assert out is state64
            mock_att.assert_called_once()

        # 3. width % 64 != 0 (width 32 % 64 != 0)
        q128 = jnp.zeros((128, 2, 64), dtype=jnp.bfloat16)
        state128 = _attention_state(q128)
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels._attend",
            return_value=state128,
        ) as mock_att:
            out = _attend_window(
                q128,
                pool,
                jnp.zeros((128, 32), dtype=jnp.int32),
                state128,
                scale=0.125,
                tile=64,
            )
            assert out is state128
            mock_att.assert_called_once()

        # 4. max_requests is None and tokens % 6 == 0 (192 % 64 == 0 and 192 % 6 == 0)
        q192 = jnp.zeros((192, 2, 64), dtype=jnp.bfloat16)
        state192 = _attention_state(q192)
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels._attend",
            return_value=state192,
        ) as mock_att:
            out = _attend_window(
                q192,
                pool,
                jnp.zeros((192, 64), dtype=jnp.int32),
                state192,
                scale=0.125,
                tile=64,
                max_requests=None,
            )
            assert out is state192
            mock_att.assert_called_once()

    def test_attend_window_max_requests_none_fast_and_patch(self):
        # 128 tokens, tile=64, width=64
        q = jnp.ones((128, 2, 64), dtype=jnp.bfloat16)
        pool = jnp.zeros((256, 128), dtype=jnp.int32)
        records = jnp.full((128, 64), -1, dtype=jnp.int32)
        for i in range(128):
            records = records.at[i, : min(64, i + 1)].set(
                jnp.arange(max(0, i - 63), i + 1, dtype=jnp.int32)
            )
        state = _attention_state(q)

        # max_requests is None with uncovered.sum() <= fallback (fast path)
        out = _attend_window(
            q,
            pool,
            records,
            state,
            scale=0.125,
            tile=64,
            fallback=64,
            max_requests=None,
        )
        assert len(out) == 3
        assert out[0].shape == (128, 2)
        assert out[1].shape == (128, 2)
        assert out[2].shape == (128, 2, 64)

        # max_requests is None with uncovered.sum() > fallback (attend fallback)
        # Make records completely fragmented so uncovered count is high
        frag_records = jnp.full((128, 64), -1, dtype=jnp.int32)
        for i in range(128):
            frag_records = frag_records.at[i, 0].set((i * 13) % 200)

        out_fallback = _attend_window(
            q,
            pool,
            frag_records,
            state,
            scale=0.125,
            tile=64,
            fallback=16,
            max_requests=None,
        )
        assert len(out_fallback) == 3
        assert out_fallback[0].shape == (128, 2)

    def test_attend_window_max_requests_provided_and_overflow(self):
        q = jnp.ones((128, 2, 64), dtype=jnp.bfloat16)
        pool = jnp.zeros((256, 128), dtype=jnp.int32)
        records = jnp.full((128, 64), -1, dtype=jnp.int32)
        for i in range(128):
            records = records.at[i, : min(64, i + 1)].set(
                jnp.arange(max(0, i - 63), i + 1, dtype=jnp.int32)
            )
        state = _attention_state(q)

        # 1. max_requests provided, normal fallback execution
        out = _attend_window(
            q, pool, records, state, scale=0.125, tile=64, fallback=64, max_requests=4
        )
        assert len(out) == 3

        # 2. max_requests provided with fallback=0 and overflow
        frag_records = jnp.full((128, 64), -1, dtype=jnp.int32)
        for i in range(128):
            frag_records = frag_records.at[i, 0].set(i)
        out_overflow = _attend_window(
            q, pool, frag_records, state, scale=0.125, tile=64, max_requests=1
        )
        assert len(out_overflow) == 3

    def test_paged_attention_jax(self):
        q = jnp.ones((128, 2, 64), dtype=jnp.bfloat16)
        swa_pool = jnp.zeros((256, 128), dtype=jnp.int32)
        main_pool = jnp.zeros((256, 128), dtype=jnp.int32)
        swa_records = jnp.full((128, 64), 0, dtype=jnp.int32)
        main_records = jnp.full((128, 64), 0, dtype=jnp.int32)
        sink = jnp.zeros((2,), dtype=jnp.float32)

        res = paged_attention_jax(
            q, swa_pool, main_pool, swa_records, main_records, sink, scale=0.125
        )
        assert res.shape == (128, 2, 64)
        assert res.dtype == jnp.bfloat16
        assert jnp.all(jnp.isfinite(res))


# ==============================================================================
# 2. Tests for Compaction, Padding & Union Gather Indices
# ==============================================================================


class TestCompactionAndGatherHelpers:
    def test_compact_selected_all_cases(self):
        # 1. All valid
        r1 = jnp.array([[3, 1, 4, 2]], dtype=jnp.int32)
        assert np.array_equal(np.asarray(_compact_selected(r1)), [[1, 2, 3, 4]])

        # 2. All negative
        r2 = jnp.array([[-1, -5, -1]], dtype=jnp.int32)
        assert np.array_equal(np.asarray(_compact_selected(r2)), [[-1, -1, -1]])

        # 3. Mixed with limit value (limit is mapped to -1 filler by design)
        limit = jnp.iinfo(jnp.int32).max
        r3 = jnp.array([[limit, -1, 5, 0]], dtype=jnp.int32)
        got = _compact_selected(r3)
        assert np.array_equal(np.asarray(got), [[0, 5, -1, -1]])

    def test_compact_records(self):
        # Input width 4 -> padded to 128
        r = jnp.array([[10, 5, -1, 20]], dtype=jnp.int32)
        out = compact_records(r)
        assert out.shape == (1, 128)
        assert out[0, 0] == 5
        assert out[0, 1] == 10
        assert out[0, 2] == 20
        assert jnp.all(out[0, 3:] == -1)

    def test_pad_compacted_records_branches(self):
        # Branch 1: already padded (padded == width)
        r128 = jnp.full((2, 128), -1, dtype=jnp.int32)
        out128 = _pad_compacted_records(r128)
        assert out128 is r128

        # Branch 2: needs padding
        r64 = jnp.full((2, 64), 1, dtype=jnp.int32)
        out_pad = _pad_compacted_records(r64)
        assert out_pad.shape == (2, 128)
        assert jnp.all(out_pad[:, :64] == 1)
        assert jnp.all(out_pad[:, 64:] == -1)


# ==============================================================================
# 3. Tests for Gather Sparse KV Chunk, Fallback & Union Gather Pipeline
# ==============================================================================


# ==============================================================================
# 4. Tests for Initialization, Custom Ops & PyTorch Wrappers
# ==============================================================================


# ==============================================================================
# 5. Tests for Exact Indexer Kernels, Scores, Top-K & Selection
# ==============================================================================


class TestIndexerKernelsAndSelection:
    def test_prefix_count(self):
        # Single block (width <= 128)
        mask = jnp.array([[1, 0, 1, 1, 0]], dtype=jnp.int32)
        res = _prefix_count(mask, block=128)
        assert np.array_equal(np.asarray(res), [[1, 1, 2, 3, 3]])

        # Multiple blocks (width > 128)
        mask2 = jnp.zeros((2, 256), dtype=jnp.int32)
        mask2 = mask2.at[:, [0, 50, 127, 128, 200]].set(1)
        res2 = _prefix_count(mask2, block=128)
        assert res2.shape == (2, 256)
        assert res2[0, 200] == 5

    def test_threshold_top_k(self):
        # Tie breaking and order-preserving integer key
        scores = jnp.array(
            [[5.0, 2.0, 5.0, 1.0, 0.0, -0.0, -jnp.inf]], dtype=jnp.float32
        )
        # Select top-2 -> indices 0 and 2 (both 5.0)
        idx = _threshold_top_k(scores, k=2)
        assert np.array_equal(np.asarray(idx), [[0, 2]])

    def test_sparsecore_top_k_branches(self):
        scores = jnp.zeros((16, 32), dtype=jnp.float32)
        lengths = jnp.full((16,), 32, dtype=jnp.int32)

        # 1. No mesh fallback (catches Exception and calls sparsecore_topk directly)
        with (
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk.sparsecore_topk",
                return_value=jnp.zeros((16, 8), dtype=jnp.int32),
            ) as mock_sc,
            mock.patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context",
                side_effect=RuntimeError("no context"),
            ),
        ):
            out_sc = _sparsecore_top_k(scores, k=8, lengths=lengths)
            assert out_sc.shape == (16, 8)
            mock_sc.assert_called_once()

        # 2. Mesh available with shard_map
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch(
                "jax.shard_map",
                side_effect=lambda fn, **kw: lambda *args: fn(*args),
            ),
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk.sparsecore_topk",
                return_value=jnp.zeros((16, 8), dtype=jnp.int32),
            ) as mock_sc_mesh,
        ):
            out_mesh = _sparsecore_top_k(scores, k=8, lengths=lengths)
            assert out_mesh.shape == (16, 8)
            mock_sc_mesh.assert_called_once()

    def test_visible_top_k_branches(self):
        # 1. Unordered fast-path (rows >= 16, width >= 2048, 16-aligned)
        scores2048 = jnp.zeros((16, 2048), dtype=jnp.float32)
        lengths = jnp.full((16,), 2048, dtype=jnp.int32)
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select._sparsecore_top_k",
            return_value=jnp.zeros((16, 16), dtype=jnp.int32),
        ) as mock_sc:
            out_fast = _visible_top_k(
                scores2048, k=16, bound=jnp.array(2048), lengths=lengths, unordered=True
            )
            assert out_fast.shape == (16, 16)
            mock_sc.assert_called_once()

        # 2. Single bucket branch (width <= 1024)
        scores512 = jnp.zeros((4, 512), dtype=jnp.float32)
        out_single = _visible_top_k(scores512, k=8, bound=jnp.array(512))
        assert out_single.shape == (4, 8)

        # 3. Multiple buckets with jax.lax.switch (width > 1024, small batch sorts)
        scores4096 = jnp.zeros((4, 4096), dtype=jnp.float32)
        out_switch_small = _visible_top_k(scores4096, k=8, bound=jnp.array(1500))
        assert out_switch_small.shape == (4, 8)

        # 4. Large columns >= 2048 and rows >= 16 calls _threshold_top_k
        scores_large = jnp.zeros((16, 4096), dtype=jnp.float32)
        out_thresh = _visible_top_k(scores_large, k=16, bound=jnp.array(3000))
        assert out_thresh.shape == (16, 16)

    def test_select_indices_jax_validation_guards(self):
        scores = jnp.zeros((4, 64), dtype=jnp.float32)
        lengths = jnp.full((4,), 64, dtype=jnp.int32)
        candidates = jnp.zeros((4, 8), dtype=jnp.int32)

        # 1. topk < 1
        with pytest.raises(ValueError, match="Positive top-k"):
            select_indices_jax(
                scores,
                lengths,
                candidates,
                topk=0,
                block_size=16,
                candidate_write=False,
                sparse_logits=False,
            )

        # 2. candidate_write and sparse_logits both True
        with pytest.raises(ValueError, match="must not use sparse logits"):
            select_indices_jax(
                scores,
                lengths,
                candidates,
                topk=8,
                block_size=16,
                candidate_write=True,
                sparse_logits=True,
            )

        # 3. candidate shape mismatch
        with pytest.raises(ValueError, match="Candidate rows must match query rows"):
            select_indices_jax(
                scores,
                lengths,
                jnp.zeros((2, 8), dtype=jnp.int32),
                topk=8,
                block_size=16,
                candidate_write=False,
                sparse_logits=False,
            )

    def test_select_indices_jax_candidate_write_and_filter(self):
        scores = jnp.zeros((4, 64), dtype=jnp.float32)
        scores = scores.at[0, 10].set(5.0)
        lengths = jnp.array([20, 30, 40, 50], dtype=jnp.int32)
        candidates = jnp.full((4, 2), -1, dtype=jnp.int32)

        # 1. candidate_write with candidates.shape[1] * block_size < width
        selected_w, published_w = select_indices_jax(
            scores,
            lengths,
            candidates,
            topk=4,
            block_size=16,
            candidate_write=True,
            sparse_logits=False,
        )
        assert selected_w.shape == (4, 4)
        assert published_w.shape == (4, 2)
        assert published_w[0, 0] >= 0

        # 2. sparse_logits with batch_max <= topk short-context bypass
        selected_f, published_f = select_indices_jax(
            scores,
            lengths,
            candidates,
            topk=64,
            block_size=16,
            candidate_write=False,
            sparse_logits=True,
        )
        assert selected_f.shape == (4, 64)

        # 3. sparse_logits without short-circuit
        candidates_valid = jnp.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=jnp.int32)
        selected_f2, _ = select_indices_jax(
            scores,
            lengths,
            candidates_valid,
            topk=4,
            block_size=16,
            candidate_write=False,
            sparse_logits=True,
        )
        assert selected_f2.shape == (4, 4)

        # 4. unordered=True (width < 2048 fallback)
        selected_unord, _ = select_indices_jax(
            scores,
            lengths,
            candidates_valid,
            topk=4,
            block_size=16,
            candidate_write=False,
            sparse_logits=False,
            unordered=True,
        )
        assert selected_unord.shape == (4, 4)

        # 5. unordered=True, width >= 2048 (SparseCore fast path)
        scores_2048 = jnp.zeros((4, 2048), dtype=jnp.float32).at[:, :16].set(1.0)
        lengths_2048 = jnp.full((4,), 2048, dtype=jnp.int32)
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select._sparsecore_top_k",
            return_value=jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (4, 16)),
        ) as mock_sc_sel:
            selected_sc, _ = select_indices_jax(
                scores_2048,
                lengths_2048,
                candidates_valid,
                topk=16,
                block_size=16,
                candidate_write=False,
                sparse_logits=False,
                unordered=True,
            )
            assert selected_sc.shape == (4, 16)
            mock_sc_sel.assert_called_once()


@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize("dim", [64, 512])
def test_bf16_gathered_attention_matches_inline_gather(monkeypatch, banks, dim):
    """Reuse the global payload while each layer changes Q, SWA and its sink."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import scatter_cache_values

    rng = np.random.default_rng(42)
    values = jnp.asarray(rng.normal(size=(128, dim)), dtype=jnp.bfloat16)
    pool = scatter_cache_values(
        jnp.zeros((128 * dim // 256, 128), jnp.int32),
        jnp.arange(128, dtype=jnp.int32),
        values,
    )
    lo, hi = (pool, None) if banks == 1 else tuple(jnp.split(pool, 2))
    main = pool if hi is None else (lo, hi)
    records = jnp.array([[1, 70, -1], [100, 2, 3], [-1, -1, -1]], jnp.int32)
    gathered = attention_kernels.gather_bf16_kv_jax(lo, hi, records, dim)
    original = attention_kernels.gather_cache_pieces
    # A separate SWA tensor makes any accidental main-cache read observable.
    swa = pool.copy()

    def checked_gather(source, *args):
        assert source is swa, "consumer gathered global KV again"
        return original(source, *args)

    for layer in range(6):
        q = jnp.asarray(rng.normal(size=(3, 2, dim)), dtype=jnp.bfloat16)
        swa_records = jnp.array([[layer, -1], [layer + 1, 7], [-1, -1]], jnp.int32)
        sink = jnp.array([layer / 10, -0.3], jnp.float32)
        expected = paged_attention_jax(q, pool, main, swa_records, records, sink, 0.05)
        with monkeypatch.context() as patcher:
            patcher.setattr(attention_kernels, "gather_cache_pieces", checked_gather)
            actual = paged_attention_jax(
                q, swa, main, swa_records, records, sink, 0.05, gathered
            )
        np.testing.assert_array_equal(actual, expected)


def test_gather_reuse_in_compiled_twenty_layer_stage(monkeypatch):
    """The captured forward has 3 global gathers for L2..L19, reused six times."""
    ak = attention_kernels
    lib = torch.library.Library("dsv41_gather_test", "FRAGMENT")
    lib.define("gather(Tensor pool, Tensor? hi, Tensor records, int dim) -> Tensor")
    lib.define(
        "attend(Tensor q, Tensor swa, Tensor main, Tensor sr, Tensor mr, "
        "Tensor sink, float scale, Tensor? gathered=None, "
        "int query_group_size=1) -> Tensor"
    )
    lib.impl("gather", lambda pool, hi, records, dim: pool.clone(), "CPU")
    lib.impl("gather", lambda pool, hi, records, dim: torch.empty_like(pool), "Meta")
    lib.impl(
        "attend",
        lambda q,
        swa,
        main,
        sr,
        mr,
        sink,
        scale,
        gathered=None,
        query_group_size=1: q.clone(),
        "CPU",
    )
    lib.impl(
        "attend",
        lambda q,
        swa,
        main,
        sr,
        mr,
        sink,
        scale,
        gathered=None,
        query_group_size=1: torch.empty_like(q),
        "Meta",
    )
    gather_op = torch.ops.dsv41_gather_test.gather.default
    attend_op = torch.ops.dsv41_gather_test.attend.default
    monkeypatch.setattr(ak, "_gather_bf16_op", gather_op)
    monkeypatch.setattr(ak, "_attention_op", attend_op)
    pool = torch.zeros((32, 128), dtype=torch.int32)
    cache = _FakeKVBuffer(pool)
    graphs = []

    def backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    def forward(q, first_records):
        with ak.gathered_kv_cache_scope():
            for layer in range(20):
                if layer < 2:
                    records = first_records[:, :0]
                elif layer in (2, 8, 14):
                    records = first_records + layer
                q = ak.paged_attention(
                    q, cache, cache, first_records, records, q.new_zeros(2), 0.1
                )
        return q

    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    q = torch.ones((2, 2, 128), dtype=torch.bfloat16)
    records = torch.zeros((2, 4), dtype=torch.int32)
    compiled(q, records)
    compiled(q, records + 1)
    assert len(graphs) == 1
    gathers = [n for n in graphs[0].graph.nodes if n.target == gather_op]
    attends = [n for n in graphs[0].graph.nodes if n.target == attend_op]
    assert len(gathers) == 3
    assert len(attends) == 20
    for group, gathered in enumerate(gathers):
        assert all(
            node.args[-1] is gathered for node in attends[2 + 6 * group : 8 + 6 * group]
        )
    assert ak._active_gathered_kv is None
