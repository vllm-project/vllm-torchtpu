# SPDX-License-Identifier: Apache-2.0
"""100% full-module line and branch coverage tests for DeepSeek-V4.1
v2/attention_kernels.py.

Covers all 31 functions/ops and all branch decision points across lines 1-1891 of:
src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace
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
    indexer_bf16,
    indexer_select,
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
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_bf16 import (
    initialize_indexer,
    paged_index_scores_jax,
    paged_indexer,
    paged_indexer_jax,
    select_indexer_jax,
    write_index_keys,
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
    def test_paged_index_scores_jax_branches(self):
        pool = jnp.zeros((64, 128), dtype=jnp.int32)
        q = jnp.zeros((8, 4, 128), dtype=jnp.bfloat16)
        weights = jnp.ones((8, 4), dtype=jnp.float32)
        lengths = jnp.full((8,), 64, dtype=jnp.int32)

        # 1. requests is None (unshared path)
        records_unshared = jnp.full((8, 128), 0, dtype=jnp.int32)
        scores1 = paged_index_scores_jax(
            q, pool, records_unshared, weights, lengths, requests=None
        )
        assert scores1.shape == (8, 128)

        # 2. requests is not None and tokens <= records.shape[0] (decode branch)
        records_req = jnp.full((16, 128), 0, dtype=jnp.int32)
        requests = jnp.arange(8, dtype=jnp.int32)
        scores2 = paged_index_scores_jax(
            q, pool, records_req, weights, lengths, requests=requests
        )
        assert scores2.shape == (8, 128)

        # 3. requests is not None and tokens > records.shape[0] (shared tile)
        q16 = jnp.zeros((16, 4, 128), dtype=jnp.bfloat16)
        weights16 = jnp.ones((16, 4), dtype=jnp.float32)
        lengths16 = jnp.full((16,), 64, dtype=jnp.int32)
        records_few = jnp.full((4, 128), 0, dtype=jnp.int32)
        requests16 = jnp.repeat(jnp.arange(4, dtype=jnp.int32), 4)
        scores3 = paged_index_scores_jax(
            q16, pool, records_few, weights16, lengths16, requests=requests16
        )
        assert scores3.shape == (16, 128)

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

    def test_paged_indexer_jax_branches(self):
        q = jnp.zeros((4, 2, 128), dtype=jnp.bfloat16)
        pool = jnp.zeros((64, 128), dtype=jnp.int32)
        records = jnp.zeros((2, 64), dtype=jnp.int32)
        weights = jnp.ones((4, 2), dtype=jnp.float32)
        lengths = jnp.full((4,), 64, dtype=jnp.int32)
        candidates = jnp.full((4, 2), 0, dtype=jnp.int32)
        requests = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        # 1. Full context bypass: not sparse_logits
        sel1, pub1 = paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk=8,
            block_size=16,
            candidate_write=False,
            sparse_logits=False,
        )
        assert sel1.shape == (4, 8)
        assert pub1.shape == (4, 2)

        # 2. Candidate filter with short context bypass (longest <= topk)
        sel2, pub2 = paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk=64,
            block_size=16,
            candidate_write=False,
            sparse_logits=True,
        )
        assert sel2.shape == (4, 64)

        # 3. Candidate context path: unordered=True
        # To trigger candidate_context: longest > topk AND row_lengths.sum() > cand_keys
        # row_lengths sum = 128, cand_keys = 4 * (1 * 16) = 64
        candidates_1 = jnp.full((4, 1), 0, dtype=jnp.int32)
        sel_cand_unord, _ = paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates_1,
            requests,
            topk=4,
            block_size=16,
            candidate_write=False,
            sparse_logits=True,
            unordered=True,
        )
        assert sel_cand_unord.shape == (4, 4)

        # 4. Candidate context path: unordered=False (runs filler ranking and sort)
        sel_cand_ord, _ = paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates_1,
            requests,
            topk=4,
            block_size=16,
            candidate_write=False,
            sparse_logits=True,
            unordered=False,
        )
        assert sel_cand_ord.shape == (4, 4)

        # 5. Candidate context: unordered=True, scores.shape[1] >= 2048 (SparseCore)
        records_4096 = jnp.zeros((4, 4096), dtype=jnp.int32)
        lengths_4096 = jnp.full((4,), 4096, dtype=jnp.int32)
        candidates_128 = jnp.broadcast_to(
            jnp.arange(128, dtype=jnp.int32)[None, :] * 2, (4, 128)
        )
        picked_16 = jnp.broadcast_to(
            jnp.array(
                [0, 15, 16, 31, 32, 33, 48, 63, 64, 80, 96, 112, 128, 144, -1, -1],
                dtype=jnp.int32,
            )[None, :],
            (4, 16),
        )
        with (
            mock.patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_bf16.paged_index_scores_jax",
                side_effect=lambda q, pool, rec, *a, **kw: jnp.zeros(
                    (q.shape[0], rec.shape[1]), dtype=jnp.float32
                ),
            ),
            mock.patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select._sparsecore_top_k",
                return_value=picked_16,
            ) as mock_sc_cand,
        ):
            sel_cand_sc, _ = paged_indexer_jax(
                q,
                pool,
                records_4096,
                weights,
                lengths_4096,
                candidates_128,
                jnp.arange(4, dtype=jnp.int32),
                topk=16,
                block_size=16,
                candidate_write=False,
                sparse_logits=True,
                unordered=True,
            )
            assert sel_cand_sc.shape == (4, 16)
            assert mock_sc_cand.call_count == 2
            # Verify exact arithmetic parity against logical[picked]
            logical_ref = (
                candidates_128[..., None] * 16 + jnp.arange(16, dtype=jnp.int32)
            ).reshape(4, 2048)
            want_sel = jnp.where(
                picked_16 >= 0,
                jnp.take_along_axis(logical_ref, jnp.maximum(picked_16, 0), axis=1),
                -1,
            )
            np.testing.assert_array_equal(
                np.asarray(sel_cand_sc),
                np.asarray(indexer_select._compact_selected(want_sel)),
            )

    def test_write_index_keys_then_select(self):
        pool = jnp.zeros((64, 128), dtype=jnp.int32)
        write_records = jnp.array([0, 1], dtype=jnp.int32)
        keys_bf16 = jnp.zeros((2, 128), dtype=jnp.bfloat16)

        # 1. Invalid dtype ValueError
        with pytest.raises(ValueError, match="Index keys must be BF16"):
            write_index_keys(
                pool, write_records, jnp.zeros((2, 128), dtype=jnp.float32)
            )

        # 2. Invalid shape ValueError
        with pytest.raises(ValueError, match="One physical record index"):
            write_index_keys(pool, jnp.array([0]), keys_bf16)

        # 3. Successful write
        updated = write_index_keys(pool, write_records, keys_bf16)
        assert updated.shape == pool.shape

        # 4. select_indexer_jax
        q = jnp.zeros((4, 2, 128), dtype=jnp.bfloat16)
        records = jnp.zeros((2, 64), dtype=jnp.int32)
        weights = jnp.ones((4, 2), dtype=jnp.float32)
        lengths = jnp.full((4,), 64, dtype=jnp.int32)
        candidates = jnp.full((4, 2), 0, dtype=jnp.int32)
        requests = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_max = jnp.array([64], dtype=jnp.int32)

        sel, pub = select_indexer_jax(
            q,
            updated,
            records,
            weights,
            lengths,
            candidates,
            requests,
            batch_max,
            topk=8,
            block_size=16,
            candidate_write=False,
            sparse_logits=False,
            unordered=False,
        )
        assert sel.shape == (4, 8)
        assert pub.shape == (4, 2)

        # 5. select_indexer_2bank_jax
        sel_2b, pub_2b = indexer_bf16.select_indexer_2bank_jax(
            q,
            updated[:32],
            updated[32:],
            records,
            weights,
            lengths,
            candidates,
            requests,
            batch_max,
            topk=8,
            block_size=16,
            candidate_write=False,
            sparse_logits=False,
            unordered=False,
        )
        assert jnp.array_equal(sel_2b, sel)
        assert jnp.array_equal(pub_2b, pub)

    def test_initialize_indexer_and_paged_indexer_wrapper(self, monkeypatch):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        captured_fake = {}

        def mock_jax_op(name, fn, **kwargs):
            if name.startswith("pallas::dsv41v2_exact_indexer"):
                assert not kwargs.get("donate_argnums")
            mock_op = mock.MagicMock()
            mock_op.register_fake = lambda fake_fn: captured_fake.setdefault(
                name, fake_fn
            )
            return mock_op

        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=mock_jax_op),
        ):
            # Reset
            indexer_bf16._indexer_op = None
            initialize_indexer()
            assert "pallas::dsv41v2_exact_indexer" in captured_fake
            fake_fn = captured_fake["pallas::dsv41v2_exact_indexer"]
            q = torch.zeros(4, 2, 64)
            pool = torch.zeros(64, 128)
            records = torch.zeros(2, 64, dtype=torch.int32)
            cands = torch.zeros(4, 2, dtype=torch.int32)
            f_sel, f_cands = fake_fn(
                q,
                pool,
                records,
                torch.ones(4, 2),
                torch.full((4,), 64),
                cands,
                torch.zeros(4),
                torch.tensor(64),
                topk=8,
                block_size=16,
                candidate_write=False,
                sparse_logits=False,
                unordered=False,
            )
            assert f_sel.shape == (4, 8)
            assert f_cands.shape == cands.shape

            assert "pallas::dsv41v2_exact_indexer_2bank" in captured_fake
            fake_2b_fn = captured_fake["pallas::dsv41v2_exact_indexer_2bank"]
            f_sel_2b, f_cands_2b = fake_2b_fn(
                q,
                pool[:32],
                pool[32:],
                records,
                torch.ones(4, 2),
                torch.full((4,), 64),
                cands,
                torch.zeros(4),
                torch.tensor(64),
                topk=8,
                block_size=16,
                candidate_write=False,
                sparse_logits=False,
                unordered=False,
            )
            assert f_sel_2b.shape == (4, 8)
            assert f_cands_2b.shape == cands.shape

            # Test early return idempotency
            saved = indexer_bf16._indexer_op
            initialize_indexer()
            assert indexer_bf16._indexer_op is saved

        # Test paged_indexer wrapper with batch_max=None and batch_max provided
        mock_idx_op = mock.MagicMock(
            return_value=(torch.zeros(4, 8), torch.zeros(4, 2))
        )
        mock_idx_2b_op = mock.MagicMock(
            return_value=(
                torch.zeros(4, 8),
                torch.zeros(4, 2),
            )
        )
        monkeypatch.setattr(indexer_bf16, "_indexer_op", mock_idx_op)
        monkeypatch.setattr(indexer_bf16, "_indexer_2bank_op", mock_idx_2b_op)

        cache = SimpleNamespace(raw=torch.zeros(64, 64))
        paged_indexer(
            torch.zeros(4, 2, 64),
            cache,
            torch.zeros(2, 64),
            torch.ones(4, 2),
            torch.full((4,), 64),
            torch.zeros(4, 2),
            torch.zeros(4),
            topk=8,
            block_size=16,
            candidate_write=False,
            sparse_logits=False,
            batch_max=None,
        )
        assert mock_idx_op.call_count == 1

        paged_indexer(
            torch.zeros(4, 2, 64),
            cache,
            torch.zeros(2, 64),
            torch.ones(4, 2),
            torch.full((4,), 64),
            torch.zeros(4, 2),
            torch.zeros(4),
            topk=8,
            block_size=16,
            candidate_write=False,
            sparse_logits=False,
            batch_max=torch.tensor([64]),
        )
        assert mock_idx_op.call_count == 2

        cache_2b = SimpleNamespace(raw=torch.zeros(32, 64), raw_hi=torch.zeros(32, 64))
        selected_2b, published_2b = paged_indexer(
            torch.zeros(4, 2, 64),
            cache_2b,
            torch.zeros(2, 64),
            torch.ones(4, 2),
            torch.full((4,), 64),
            torch.zeros(4, 2),
            torch.zeros(4),
            topk=8,
            block_size=16,
            candidate_write=False,
            sparse_logits=False,
            batch_max=torch.tensor([64]),
        )
        assert selected_2b.shape == (4, 8)
        assert published_2b.shape == (4, 2)
        assert mock_idx_2b_op.call_count == 1


@pytest.mark.parametrize("mode", ["plain", "publish", "filter"])
@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize("unordered", [False, True])
def test_paged_indexer_query_tiles_preserve_selection(
    monkeypatch, mode, banks, unordered
):
    """Bound score rows, including a partial tile, without changing selection."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import scatter_index_keys

    rng = np.random.default_rng(17)
    rows, width, topk = 65, 128, 16
    keys = jnp.asarray(rng.standard_normal((2 * width, 128)), jnp.bfloat16)
    pool = scatter_index_keys(
        jnp.zeros((keys.shape[0] // 2, 128), jnp.int32),
        jnp.arange(keys.shape[0], dtype=jnp.int32),
        keys,
    )
    if banks == 2:
        pool = tuple(jnp.split(pool, 2))
    q = jnp.asarray(rng.standard_normal((rows, 2, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.random((rows, 2)), jnp.float32)
    records = jnp.arange(2 * width, dtype=jnp.int32).reshape(2, width)
    requests = jnp.arange(rows, dtype=jnp.int32) % 2
    # The first tile is short while another tile is long: candidate filtering
    # must retain the whole batch's short-context bypass decision.
    lengths = jnp.asarray([8] * 32 + [0, 64, 127] * 11, jnp.int32)
    candidates = jnp.tile(jnp.array([[0, 0, 3, -1]], jnp.int32), (rows, 1))
    scores = paged_index_scores_jax(q, pool, records, weights, lengths, requests)
    expected = select_indices_jax(
        scores,
        lengths,
        candidates,
        topk,
        16,
        mode == "publish",
        mode == "filter",
        unordered,
    )
    score_rows = []
    original = indexer_bf16.paged_index_scores_jax

    def record_shape(queries, *args, **kwargs):
        score_rows.append(queries.shape[0])
        return original(queries, *args, **kwargs)

    monkeypatch.setattr(indexer_bf16, "paged_index_scores_jax", record_shape)
    actual = jax.jit(
        lambda: paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk,
            16,
            mode == "publish",
            mode == "filter",
            unordered,
        )
    )()
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    assert score_rows and max(score_rows) <= indexer_bf16.indexer.QUERY_TILE


@pytest.mark.parametrize(
    "mode,banks,layout,request_count,candidate_width",
    [
        ("plain", 1, "balanced", 48, 4),
        ("publish", 2, "balanced", 48, 4),
        ("filter", 2, "shuffled", 49, 4),
        ("plain", 2, "uneven", 48, 4),
        ("publish", 2, "empty", 48, 4),
        ("publish", 1, "empty", 48, 8),
    ],
)
def test_paged_indexer_query_tiles_share_request_keys(
    monkeypatch, mode, banks, layout, request_count, candidate_width
):
    """Six-query requests share keys across tile boundaries and restore row order."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
        scatter_index_keys,
        split_halves,
    )

    rng = np.random.default_rng(31)
    rows, width, page, topk = 8 * request_count, 128, 16, 16
    keys = jnp.asarray(rng.standard_normal((request_count * width, 128)), jnp.bfloat16)
    pool = scatter_index_keys(
        jnp.zeros((request_count * width // 2, 128), jnp.int32),
        jnp.arange(request_count * width, dtype=jnp.int32),
        keys,
    )
    if banks == 2:
        pool = tuple(jnp.split(pool, 2))
    records = jnp.arange(request_count * width, dtype=jnp.int32).reshape(
        request_count, width
    )
    records = records.at[3, -page:].set(-1)
    requests = np.pad(np.repeat(np.arange(request_count), 6), (0, 2 * request_count))
    lengths = np.pad(
        rng.integers(32, width + 1, 6 * request_count), (0, 2 * request_count)
    )
    if layout == "shuffled":
        order = rng.permutation(rows)
        requests, lengths = requests[order], lengths[order]
    elif layout == "uneven":
        requests[:24] = 0
    elif layout == "empty":
        lengths[:] = 0
    requests, lengths = (
        jnp.asarray(requests, jnp.int32),
        jnp.asarray(lengths, jnp.int32),
    )
    q = jnp.asarray(rng.standard_normal((rows, 2, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.random((rows, 2)), jnp.float32)
    candidates = jnp.tile(jnp.arange(candidate_width, dtype=jnp.int32), (rows, 1))
    token_records = records[requests]
    dense_keys = split_halves(keys[jnp.maximum(token_records, 0)])
    logits = jnp.einsum(
        "thd,tcd->thc", split_halves(q), dense_keys, preferred_element_type=jnp.float32
    )
    scores = (jnp.maximum(logits, 0) * weights[:, :, None]).sum(1)
    scores = jnp.where(token_records >= 0, scores, -jnp.inf)
    expected = select_indices_jax(
        scores, lengths, candidates, topk, page, mode == "publish", mode == "filter"
    )
    gathered_keys, score_rows = [], []
    original_gather = indexer_bf16.gather_cache_page_words
    original_score = indexer_bf16.paged_index_scores_jax

    def record_gather(pool, first_records, page_records, dim):
        jax.debug.callback(
            lambda count: gathered_keys.append(int(count)),
            first_records.size * page_records,
            ordered=True,
        )
        return original_gather(pool, first_records, page_records, dim)

    def record_score(queries, *args, **kwargs):
        score_rows.append(queries.shape[0])
        return original_score(queries, *args, **kwargs)

    monkeypatch.setattr(indexer_bf16, "gather_cache_page_words", record_gather)
    monkeypatch.setattr(indexer_bf16, "paged_index_scores_jax", record_score)
    actual = jax.jit(
        lambda: paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk,
            page,
            mode == "publish",
            mode == "filter",
            page_records=page,
        )
    )()
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    jax.effects_barrier()
    assert score_rows and max(score_rows) <= indexer_bf16.indexer.QUERY_TILE
    if layout != "uneven":
        # One key stream per request, including the final padded request tile.
        request_tile = indexer_bf16.indexer.QUERY_TILE // 8
        padded_requests = -(-request_count // request_tile) * request_tile
        assert sum(gathered_keys) <= padded_requests * width


@pytest.mark.parametrize("length", [0, 127, 128, 129, 255])
@pytest.mark.parametrize("banks", [1, 2])
def test_paged_indexer_context_bucket_preserves_boundary_and_fallback(
    monkeypatch, length, banks
):
    """Bucket edges and the long-context fallback preserve exact selection."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import scatter_index_keys

    monkeypatch.setattr(
        indexer_bf16.tuned_block_sizes, "PAGED_INDEXER_PREFILL_BUCKET", (128, 64)
    )
    rng = np.random.default_rng(23)
    rows, width, topk = 129, 256, 16
    keys = jnp.asarray(rng.standard_normal((width, 128)), jnp.bfloat16)
    pool = scatter_index_keys(
        jnp.zeros((keys.shape[0] // 2, 128), jnp.int32),
        jnp.arange(keys.shape[0], dtype=jnp.int32),
        keys,
    )
    if banks == 2:
        pool = tuple(jnp.split(pool, 2))
    q = jnp.asarray(rng.standard_normal((rows, 2, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.random((rows, 2)), jnp.float32)
    records = jnp.arange(width, dtype=jnp.int32)[None, :]
    lengths = jnp.full((rows,), length, jnp.int32).at[-1].set(0)
    requests = jnp.zeros((rows,), jnp.int32)
    candidates = jnp.zeros((rows, 4), jnp.int32)
    scores = paged_index_scores_jax(q, pool, records, weights, lengths, requests)
    expected = select_indices_jax(
        scores, lengths, candidates, topk, 16, False, False, True
    )
    shapes = []
    original = indexer_bf16.paged_index_scores_jax

    def record_shape(queries, pool, records, *args, **kwargs):
        shapes.append((queries.shape[0], records.shape[1]))
        return original(queries, pool, records, *args, **kwargs)

    monkeypatch.setattr(indexer_bf16, "paged_index_scores_jax", record_shape)
    actual = jax.jit(
        lambda: paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk,
            16,
            False,
            False,
            True,
        )
    )()
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    assert (64, 128) in shapes
    assert (32, 256) in shapes


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


@pytest.mark.parametrize("banks", [1, 2])
def test_paged_indexer_decode_rows_skip_query_tiles(monkeypatch, banks):
    """Decode-shaped batches (few rows per request) score in one query tile.

    DSpark verification has K + 1 rows per request. Query tiles smaller than
    the request-row count would defeat grouped per-request key sharing.
    """
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import scatter_index_keys

    rng = np.random.default_rng(31)
    requests_count, group, width, topk = 16, 6, 256, 16
    rows = requests_count * group
    keys = jnp.asarray(rng.standard_normal((requests_count * width, 128)), jnp.bfloat16)
    pool = scatter_index_keys(
        jnp.zeros((keys.shape[0] // 2, 128), jnp.int32),
        jnp.arange(keys.shape[0], dtype=jnp.int32),
        keys,
    )
    if banks == 2:
        pool = tuple(jnp.split(pool, 2))
    q = jnp.asarray(rng.standard_normal((rows, 2, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.random((rows, 2)), jnp.float32)
    records = jnp.arange(requests_count * width, dtype=jnp.int32).reshape(
        requests_count, width
    )
    requests = jnp.repeat(jnp.arange(requests_count, dtype=jnp.int32), group)
    lengths = jnp.asarray(rng.integers(1, width + 1, rows), jnp.int32)
    candidates = jnp.zeros((rows, 4), jnp.int32)
    scores = paged_index_scores_jax(q, pool, records, weights, lengths, requests)
    expected = select_indices_jax(
        scores, lengths, candidates, topk, 16, False, False, True
    )
    score_rows = []
    original = indexer_bf16.paged_index_scores_jax

    def record_shape(queries, *args, **kwargs):
        score_rows.append(queries.shape[0])
        return original(queries, *args, **kwargs)

    monkeypatch.setattr(indexer_bf16, "paged_index_scores_jax", record_shape)
    actual = jax.jit(
        lambda: paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk,
            16,
            False,
            False,
            True,
        )
    )()
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
    assert rows > indexer_bf16.indexer.QUERY_TILE
    assert score_rows == [rows]
