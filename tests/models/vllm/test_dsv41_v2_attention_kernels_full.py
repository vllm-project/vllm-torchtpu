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

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _attend,
    _attend_window,
    _attention_state,
    _compact_selected,
    _pad_compacted_records,
    _prefix_count,
    _sparsecore_top_k,
    _threshold_top_k,
    _visible_top_k,
    _window_union,
    compact_records,
    compute_chunk_union_gather_indices,
    gather_sparse_kv_chunk_with_union_fallback,
    initialize_indexer,
    initialize_paged_attention,
    paged_attention,
    paged_attention_jax,
    paged_index_scores_jax,
    paged_indexer,
    paged_indexer_jax,
    remap_union_kv_to_chunk,
    repack_chunk_rope_words,
    select_indices_jax,
    slice_token_parallel_attention_inputs,
    sparse_fp8_attention,
    sparse_fp8_attention_jax,
    sparse_fp8_attention_qo,
    sparse_ragged_paged_attention_union_gather,
    unpack_union_rope_words,
    update_and_select_jax,
    write_index_keys,
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

    def test_compute_chunk_union_gather_indices(self):
        # Test both padded_tokens > num_tokens and normal cases
        # num_tokens = 100, chunk_size = 128 -> padded_tokens = 128 > 100
        records = jnp.full((100, 16), -1, dtype=jnp.int32)
        records = records.at[:, :8].set(jnp.arange(8, dtype=jnp.int32))
        page_indices = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 50, 100], dtype=jnp.int32)

        # CPU backend path
        union_phys, inv_idx, num_unique = compute_chunk_union_gather_indices(
            records,
            page_indices,
            cu_q_lens,
            page_size=16,
            total_num_pages=4,
            chunk_size=128,
            union_cap=64,
        )
        assert union_phys.shape == (1, 64)
        assert inv_idx.shape == (1, 128, 16)
        assert num_unique.shape == (1,)

        # TPU backend path mock
        with (
            mock.patch("jax.default_backend", return_value="tpu"),
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla.gather_page_ids",
                return_value=jnp.zeros((1, 64), dtype=jnp.int32),
            ) as mock_gather_page,
        ):
            union_tpu, _, _ = compute_chunk_union_gather_indices(
                records,
                page_indices,
                cu_q_lens,
                page_size=16,
                total_num_pages=4,
                chunk_size=128,
                union_cap=64,
            )
            mock_gather_page.assert_called_once()
            assert union_tpu.shape == (1, 64)

    def test_rope_words_unpack_and_repack_roundtrip(self):
        # union_cap = 64, topk = 16
        # u_rope input is (union_cap // 4, 128) = (16, 128)
        rng = np.random.default_rng(42)
        u_rope_np = rng.integers(-(2**31), 2**31 - 1, size=(16, 128), dtype=np.int32)
        u_rope = jnp.asarray(u_rope_np)

        unpacked = unpack_union_rope_words(u_rope, topk=16)
        assert unpacked.shape == (64, 32)

        # Repack expects (chunk_size, topk, 32)
        r32 = unpacked.reshape(4, 16, 32)
        repacked = repack_chunk_rope_words(r32)
        assert repacked.shape == (4, 4, 128)

    def test_remap_union_kv_to_chunk(self):
        # u_nope: (union_cap, 64)
        # u_rope: (union_cap // 4, 128)
        u_nope = jnp.ones((64, 64), dtype=jnp.uint8)
        u_rope = jnp.ones((16, 128), dtype=jnp.int32)
        inv_idx = jnp.zeros((8, 16), dtype=jnp.int32)
        g_nope, g_rope = remap_union_kv_to_chunk(u_nope, u_rope, inv_idx, topk=16)
        assert g_nope.shape == (8, 16, 64)
        assert g_rope.shape == (8, 4, 128)


# ==============================================================================
# 3. Tests for Gather Sparse KV Chunk, Fallback & Union Gather Pipeline
# ==============================================================================


class TestGatherSparseKVAndUnionGather:
    def test_gather_sparse_kv_chunk_union_and_direct_branches(self):
        cache_nope = jnp.ones((4, 16, 64), dtype=jnp.uint8)
        cache_rope = jnp.ones((4, 4, 128), dtype=jnp.int32)
        union_phys = jnp.zeros((32,), dtype=jnp.int32)
        inv_idx = jnp.zeros((8, 16), dtype=jnp.int32)
        chunk_rec = jnp.zeros((8, 16), dtype=jnp.int32)
        chunk_seq = jnp.zeros((8,), dtype=jnp.int32)
        page_idx = jnp.zeros((4,), dtype=jnp.int32)

        # 1. Union branch on CPU (num_unique <= union_cap)
        g_nope_u, g_rope_u = gather_sparse_kv_chunk_with_union_fallback(
            cache_nope,
            cache_rope,
            union_phys,
            inv_idx,
            num_unique_chunk=jnp.array(20, dtype=jnp.int32),
            chunk_records=chunk_rec,
            chunk_seq_ids=chunk_seq,
            page_indices=page_idx,
            max_num_seqs=1,
            start_pos=0,
            batch_end=jnp.array(8, dtype=jnp.int32),
            topk=16,
        )
        assert g_nope_u.shape == (8, 16, 64)

        def _mock_csa(c_nope, c_rope, indices, num_valid_indices, top_k):
            n = indices.shape[0]
            return (
                jnp.ones((n, 64), dtype=jnp.uint8),
                jnp.ones((n // 4, 128), dtype=jnp.int32),
            )

        # 2. Union branch on TPU mock
        with (
            mock.patch("jax.default_backend", return_value="tpu"),
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla.gather_page_ids",
                return_value=jnp.zeros((8, 16), dtype=jnp.int32),
            ),
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather.csa_gather",
                side_effect=_mock_csa,
            ) as mock_csa,
        ):
            gather_sparse_kv_chunk_with_union_fallback(
                cache_nope,
                cache_rope,
                union_phys,
                inv_idx,
                num_unique_chunk=jnp.array(20, dtype=jnp.int32),
                chunk_records=chunk_rec,
                chunk_seq_ids=chunk_seq,
                page_indices=page_idx,
                max_num_seqs=1,
                start_pos=0,
                batch_end=jnp.array(8, dtype=jnp.int32),
                topk=16,
            )
            mock_csa.assert_called()

        # 3. Direct branch on CPU (num_unique > union_cap)
        g_nope_d, g_rope_d = gather_sparse_kv_chunk_with_union_fallback(
            cache_nope,
            cache_rope,
            union_phys,
            inv_idx,
            num_unique_chunk=jnp.array(50, dtype=jnp.int32),  # > 32
            chunk_records=chunk_rec,
            chunk_seq_ids=chunk_seq,
            page_indices=page_idx,
            max_num_seqs=1,
            start_pos=0,
            batch_end=jnp.array(8, dtype=jnp.int32),
            topk=16,
        )
        assert g_nope_d.shape == (8, 16, 64)

        # 4. Direct branch on TPU mock
        with (
            mock.patch("jax.default_backend", return_value="tpu"),
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla.gather_page_ids",
                return_value=jnp.zeros((8, 16), dtype=jnp.int32),
            ),
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather.csa_gather",
                side_effect=_mock_csa,
            ) as mock_csa_direct,
        ):
            gather_sparse_kv_chunk_with_union_fallback(
                cache_nope,
                cache_rope,
                union_phys,
                inv_idx,
                num_unique_chunk=jnp.array(50, dtype=jnp.int32),
                chunk_records=chunk_rec,
                chunk_seq_ids=chunk_seq,
                page_indices=page_idx,
                max_num_seqs=1,
                start_pos=0,
                batch_end=jnp.array(8, dtype=jnp.int32),
                topk=16,
            )
            mock_csa_direct.assert_called()

    def test_sparse_ragged_paged_attention_union_gather(self):
        q = jnp.ones((128, 4, 64), dtype=jnp.bfloat16)
        cache_nope = jnp.ones((4, 16, 64), dtype=jnp.uint8)
        cache_rope = jnp.ones((4, 4, 128), dtype=jnp.int32)
        topk_indices = jnp.full((128, 16), 0, dtype=jnp.int32)
        page_indices = jnp.zeros((4,), dtype=jnp.int32)
        cu_q_lens = jnp.array([0, 128], dtype=jnp.int32)
        distribution = jnp.array([0, 0, 1], dtype=jnp.int32)
        attention_sinks = jnp.zeros((4,), dtype=jnp.float32)
        swa_accum = jnp.zeros((128, 4, 64), dtype=jnp.bfloat16)
        swa_l = jnp.ones((128, 4), dtype=jnp.float32)
        swa_m = jnp.zeros((128, 4), dtype=jnp.float32)

        # Mock pallas_call to execute and return output tensor cleanly on CPU
        def fake_pallas_call(fn, **kwargs):
            return lambda *args: q

        with mock.patch(
            "jax.experimental.pallas.pallas_call", side_effect=fake_pallas_call
        ):
            res = sparse_ragged_paged_attention_union_gather(
                q,
                cache_nope,
                cache_rope,
                topk_indices,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                swa_accum,
                swa_l,
                swa_m,
                gather_and_attention_chunk_size=128,
            )
            assert res.shape == (128, 4, 64)

            # 2-bank with union_gather=False
            res_2b = sparse_ragged_paged_attention_union_gather(
                q,
                (cache_nope[:2], cache_nope[2:]),
                (cache_rope[:2], cache_rope[2:]),
                topk_indices,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                swa_accum,
                swa_l,
                swa_m,
                gather_and_attention_chunk_size=128,
                union_gather=False,
            )
            assert res_2b.shape == (128, 4, 64)

        # Direct test of _csa_gather_pool with 2-bank tuple and num_valid_indices=None
        indices_2b = jnp.arange(32, dtype=jnp.int32)
        g_n_all, g_r_all = attention_kernels._csa_gather_pool(
            cache_nope, cache_rope, indices_2b, num_valid_indices=None, top_k=16
        )
        g_n_2b, g_r_2b = attention_kernels._csa_gather_pool(
            (cache_nope[:2], cache_nope[2:]),
            (cache_rope[:2], cache_rope[2:]),
            indices_2b,
            num_valid_indices=None,
            top_k=16,
        )
        assert jnp.array_equal(g_n_2b, g_n_all)
        assert jnp.array_equal(g_r_2b, g_r_all)

    def test_sparse_fp8_attention_jax_variants(self):
        q = jnp.ones((128, 4, 64), dtype=jnp.bfloat16)
        swa_pool = jnp.zeros((256, 128), dtype=jnp.int32)
        main_pool = jnp.ones((256, 128), dtype=jnp.int32)
        rope_pool = jnp.ones((64, 128), dtype=jnp.int32)
        swa_records = jnp.zeros((128, 64), dtype=jnp.int32)
        main_records = jnp.zeros((128, 128), dtype=jnp.int32)
        pages = jnp.zeros((1, 1), dtype=jnp.int32)
        query_start_loc = jnp.array([0, 128], dtype=jnp.int32)
        sink = jnp.zeros((4,), dtype=jnp.float32)

        # 1. Non-union gather, precompacted=False
        with mock.patch(
            "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla.sparse_ragged_paged_attention",
            return_value=q,
        ) as mock_std_attn:
            res1 = sparse_fp8_attention_jax(
                q,
                swa_pool,
                main_pool,
                rope_pool,
                swa_records,
                main_records,
                pages,
                query_start_loc,
                sink,
                scale=0.125,
                page_size=16,
                precompacted=False,
                union_gather=False,
            )
            assert res1 is q
            mock_std_attn.assert_called_once()

        # 2. Union gather, precompacted=True
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels.sparse_ragged_paged_attention_union_gather",
            return_value=q,
        ) as mock_union_attn:
            res2 = sparse_fp8_attention_jax(
                q,
                swa_pool,
                main_pool,
                rope_pool,
                swa_records,
                main_records,
                pages,
                query_start_loc,
                sink,
                scale=0.125,
                page_size=16,
                precompacted=True,
                union_gather=True,
            )
            assert res2 is q
            mock_union_attn.assert_called_once()

            # 3. 2-bank tuple with union_gather=False routes through union_gather helper
            res3 = sparse_fp8_attention_jax(
                q,
                (swa_pool[:128], swa_pool[128:]),
                (main_pool[:128], main_pool[128:]),
                (rope_pool[:32], rope_pool[32:]),
                swa_records,
                main_records,
                pages,
                query_start_loc,
                sink,
                scale=0.125,
                page_size=16,
                precompacted=True,
                union_gather=False,
            )
            assert res3 is q
            assert mock_union_attn.call_count == 2

    def test_slice_token_parallel_attention_inputs(self):
        swa_rec = jnp.arange(1024 * 64, dtype=jnp.int32).reshape(1024, 64)
        main_rec = jnp.arange(1024 * 128, dtype=jnp.int32).reshape(1024, 128)
        q_loc = jnp.array([0, 512, 1024], dtype=jnp.int32)
        tp_rank = jnp.array([1], dtype=jnp.int32)

        s_loc, m_loc, ql_loc = slice_token_parallel_attention_inputs(
            swa_rec, main_rec, q_loc, tp_rank, tp_size=2, tokens=1024
        )
        assert s_loc.shape == (512, 64)
        assert m_loc.shape == (512, 128)
        assert ql_loc.shape == (3,)
        assert ql_loc[0] == 0
        assert ql_loc[1] == 0
        assert ql_loc[2] == 512


# ==============================================================================
# 4. Tests for Initialization, Custom Ops & PyTorch Wrappers
# ==============================================================================


class TestInitializationAndCustomOps:
    def test_initialize_paged_attention_lifecycle_and_idempotency(self):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        with set_vllm_model_wrapper_context(mesh=mesh):
            # Reset global ops
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels._fp8_attention_qo_op = None

            # First initialization
            initialize_paged_attention()
            assert attention_kernels._attention_op is not None
            assert attention_kernels._fp8_attention_op is not None
            assert attention_kernels._fp8_attention_qo_op is not None

            # Second initialization (idempotent early return)
            saved_op = attention_kernels._attention_op
            initialize_paged_attention()
            assert attention_kernels._attention_op is saved_op

    def test_fake_tensor_registrations_for_paged_attention(self):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        captured_fakes = {}

        def mock_jax_op(name, fn, **kwargs):
            mock_op = mock.MagicMock()
            mock_op._fn = fn

            def register_fake(fake_fn):
                captured_fakes[name] = fake_fn

            mock_op.register_fake = register_fake
            return mock_op

        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=mock_jax_op),
        ):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels._fp8_attention_qo_op = None
            initialize_paged_attention()

            # 1. _attention_op fake: lambda q, *args: q.new_empty(q.shape)
            fake_attn = captured_fakes["pallas::dsv41v2_paged_bf16_attention"]
            q_t = torch.zeros(8, 4, 64)
            out_fake_attn = fake_attn(q_t)
            assert out_fake_attn.shape == (8, 4, 64)

            # 2. _fp8_attention_op fake: lambda q, *args: q.new_empty(q.shape)
            fake_fp8 = captured_fakes["pallas::dsv41v2_sparse_fp8_attention"]
            out_fake_fp8 = fake_fp8(q_t)
            assert out_fake_fp8.shape == (8, 4, 64)

            # 3. _fp8_attention_qo_op fake: positions at args[5], wo_a at args[17]
            # args[5] is positions, args[17] is wo_a
            fake_qo = captured_fakes["pallas::dsv41v2_sparse_fp8_attention_qo"]
            qr = torch.zeros(8, 64)
            dummy_args = [torch.zeros(1)] * 22
            dummy_args[5] = torch.zeros(16, dtype=torch.int64)  # positions [T=16]
            dummy_args[17] = torch.zeros(128, 32)  # wo_a [..., o_lora_rank=32]
            out_fake_qo = fake_qo(qr, *dummy_args)
            assert out_fake_qo.shape == (16, 32)

            # 4. 2-bank fake registrations
            fake_attn_2b = captured_fakes["pallas::dsv41v2_paged_bf16_attention_2bank"]
            assert fake_attn_2b(q_t).shape == (8, 4, 64)
            fake_fp8_2b = captured_fakes["pallas::dsv41v2_sparse_fp8_attention_2bank"]
            assert fake_fp8_2b(q_t).shape == (8, 4, 64)
            fake_qo_2b = captured_fakes["pallas::dsv41v2_sparse_fp8_attention_qo_2bank"]
            dummy_args_2b = [torch.zeros(1)] * 25
            dummy_args_2b[5] = torch.zeros(16, dtype=torch.int64)
            dummy_args_2b[20] = torch.zeros(128, 32)
            assert fake_qo_2b(qr, *dummy_args_2b).shape == (16, 32)

            # 5. Execute 2-bank inner wrappers (paged_attention_2bank)
            attn_2b_fn = attention_kernels._attention_2bank_op._fn
            res_2b_attn = attn_2b_fn(
                jnp.zeros((2, 2, 128), dtype=jnp.bfloat16),
                jnp.zeros((4, 128), dtype=jnp.int32),
                jnp.zeros((4, 128), dtype=jnp.int32),
                jnp.zeros((4, 128), dtype=jnp.int32),
                jnp.zeros((4, 128), dtype=jnp.int32),
                jnp.zeros((2, 64), dtype=jnp.int32),
                jnp.zeros((2, 64), dtype=jnp.int32),
                jnp.zeros((2,), dtype=jnp.float32),
                0.5,
            )
            assert res_2b_attn.shape == (2, 2, 128)

    def test_fp8_attention_inner_bodies_tp_and_non_tp(self):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        captured_ops = {}

        def mock_jax_op(name, fn, **kwargs):
            mock_op = mock.MagicMock()
            mock_op._fn = fn
            captured_ops[name] = fn
            return mock_op

        mock_tp_group = mock.MagicMock()
        mock_tp_group.size.return_value = 2

        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch(
                "vllm.distributed.get_tp_group",
                return_value=SimpleNamespace(device_group=mock_tp_group),
            ),
            mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=mock_jax_op),
        ):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels._fp8_attention_qo_op = None
            initialize_paged_attention()

            fp8_attn_fn = captured_ops["pallas::dsv41v2_sparse_fp8_attention"]

            with (
                mock.patch(
                    "jax.shard_map",
                    side_effect=lambda fn, **kw: lambda *args: fn(*args),
                ),
                mock.patch("jax.lax.all_to_all", side_effect=lambda x, **kw: x),
                mock.patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels.sparse_fp8_attention_jax",
                    side_effect=lambda q, *args, **kw: q,
                ),
            ):
                # 1. Token-parallel branch: tp_size=2, tokens=256 >= 128*2
                res_tp = fp8_attn_fn(
                    jnp.zeros((256, 4, 64)),
                    jnp.zeros((256, 64)),
                    jnp.zeros((256, 64)),
                    jnp.zeros((64, 128)),
                    jnp.zeros((256, 64), dtype=jnp.int32),
                    jnp.zeros((256, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((4,)),
                    jnp.zeros((4,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    True,  # token_parallel=True
                    False,
                    4096,
                )
                assert res_tp.shape == (256, 4, 64)

                # 2. Non-token-parallel branch: token_parallel=False
                res_non_tp = fp8_attn_fn(
                    jnp.zeros((64, 4, 64)),
                    jnp.zeros((64, 64)),
                    jnp.zeros((64, 64)),
                    jnp.zeros((16, 128)),
                    jnp.zeros((64, 64), dtype=jnp.int32),
                    jnp.zeros((64, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((4,)),
                    jnp.zeros((4,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    False,  # token_parallel=False
                    False,
                    4096,
                )
                assert res_non_tp.shape == (64, 4, 64)

                # 3. 2-bank wrapper: fp8_attention_2bank
                fp8_attn_2b_fn = captured_ops[
                    "pallas::dsv41v2_sparse_fp8_attention_2bank"
                ]
                res_2b = fp8_attn_2b_fn(
                    jnp.zeros((64, 4, 64)),
                    jnp.zeros((32, 64)),
                    jnp.zeros((32, 64)),
                    jnp.zeros((32, 64)),
                    jnp.zeros((32, 64)),
                    jnp.zeros((8, 128)),
                    jnp.zeros((8, 128)),
                    jnp.zeros((64, 64), dtype=jnp.int32),
                    jnp.zeros((64, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((4,)),
                    jnp.zeros((4,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    False,
                    False,
                    4096,
                )
                assert res_2b.shape == (64, 4, 64)

    def test_fp8_attention_qo_inner_bodies(self):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        captured_ops = {}

        def mock_jax_op(name, fn, **kwargs):
            mock_op = mock.MagicMock()
            mock_op._fn = fn
            captured_ops[name] = fn
            return mock_op

        mock_tp_group = mock.MagicMock()
        mock_tp_group.size.return_value = 2

        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch(
                "vllm.distributed.get_tp_group",
                return_value=SimpleNamespace(device_group=mock_tp_group),
            ),
            mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
            mock.patch("torch_tpu._internal.pallas.jax_op", side_effect=mock_jax_op),
        ):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels._fp8_attention_qo_op = None
            initialize_paged_attention()

            fp8_qo_fn = captured_ops["pallas::dsv41v2_sparse_fp8_attention_qo"]

            with (
                mock.patch(
                    "jax.shard_map",
                    side_effect=lambda fn, **kw: lambda *args: fn(*args),
                ),
                mock.patch("jax.lax.all_to_all", side_effect=lambda x, **kw: x),
                mock.patch(
                    "jax.lax.all_gather",
                    side_effect=lambda x, **kw: jnp.repeat(x, 2, axis=0),
                ),
                mock.patch(
                    "vllm_torchtpu.layers.adapter.linear_common._quantized_matmul_jax",
                    side_effect=lambda q, w, ws: jnp.zeros((q.shape[0], 256)),
                ),
                mock.patch(
                    "vllm_torchtpu.models.vllm.deepseek_v4.attention._rope_jax",
                    side_effect=lambda q, p, c: q,
                ),
                mock.patch(
                    "vllm_torchtpu.models.vllm.deepseek_v4.attention._o_proj_jax",
                    side_effect=lambda o, p, c, w, ws, head_dim: jnp.zeros(
                        (o.shape[0], 64)
                    ),
                ),
                mock.patch(
                    "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels.sparse_fp8_attention_jax",
                    side_effect=lambda q, *args, **kw: q,
                ),
            ):
                # 1. Local SP body: tokens=256, tokens_local=128, tp_size=2
                res_local = fp8_qo_fn(
                    jnp.zeros((256, 128)),  # qr
                    jnp.zeros((128, 128)),  # qr_local
                    jnp.zeros((128, 256)),  # wq_b
                    jnp.zeros((256,)),  # wq_b_scale
                    jnp.zeros((128, 256)),  # wq_b_full
                    jnp.zeros((256,)),  # wq_b_full_scale
                    jnp.zeros((256,), dtype=jnp.int32),  # positions
                    jnp.zeros((256, 64)),  # cos_sin_cache
                    jnp.zeros((256, 64)),  # swa_pool
                    jnp.zeros((256, 64)),  # main_pool
                    jnp.zeros((64, 128)),  # rope_pool
                    jnp.zeros((256, 64), dtype=jnp.int32),
                    jnp.zeros((256, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((4,)),
                    jnp.zeros((4,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.zeros((256, 64)),  # wo_a
                    jnp.zeros((64,)),
                    jnp.zeros((256, 64)),  # wo_a_full
                    jnp.zeros((64,)),
                    jnp.ones((256,), dtype=jnp.int32),  # valid
                    0.125,
                    16,
                    None,
                    False,
                    64,
                )
                assert res_local.shape == (256, 32) or res_local.shape == (256, 64)

                # 2. Sharded body with q_in.shape[0] < tokens (all_gather branch)
                res_sharded = fp8_qo_fn(
                    jnp.zeros((128, 128)),  # qr shape 128 < positions 256
                    jnp.zeros((64, 128)),  # qr_local (does not match tokens condition)
                    jnp.zeros((128, 256)),
                    jnp.zeros((256,)),
                    jnp.zeros((128, 256)),
                    jnp.zeros((256,)),
                    jnp.zeros((256,), dtype=jnp.int32),  # positions 256
                    jnp.zeros((256, 64)),
                    jnp.zeros((256, 64)),
                    jnp.zeros((256, 64)),
                    jnp.zeros((64, 128)),
                    jnp.zeros((256, 64), dtype=jnp.int32),
                    jnp.zeros((256, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((4,)),
                    jnp.zeros((4,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.zeros((256, 64)),
                    jnp.zeros((64,)),
                    jnp.zeros((256, 64)),
                    jnp.zeros((64,)),
                    jnp.ones((256,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    64,
                )
                assert res_sharded.shape[0] == 256

                # 3. Sharded body with q_in.shape[0] == tokens (bypasses all_gather)
                res_sharded_no_gather = fp8_qo_fn(
                    jnp.zeros((64, 128)),  # qr shape 64 == positions 64
                    jnp.zeros((32, 128)),  # qr_local
                    jnp.zeros((128, 256)),
                    jnp.zeros((256,)),
                    jnp.zeros((128, 256)),
                    jnp.zeros((256,)),
                    jnp.zeros(
                        (64,), dtype=jnp.int32
                    ),  # positions 64 (< 256, so tokens < 128 * tp_size)
                    jnp.zeros((64, 64)),
                    jnp.zeros((256, 128), dtype=jnp.int32),
                    jnp.zeros((256, 64), dtype=jnp.uint8),
                    jnp.zeros((64, 128), dtype=jnp.int32),
                    jnp.zeros((64, 64), dtype=jnp.int32),
                    jnp.zeros((64, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((4,)),
                    jnp.zeros((4,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.zeros((64, 64)),
                    jnp.zeros((64,)),
                    jnp.zeros((64, 64)),
                    jnp.zeros((64,)),
                    jnp.ones((64,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    64,
                )
                assert res_sharded_no_gather.shape[0] == 64

                # 4. 2-bank wrapper: fp8_attention_qo_2bank
                fp8_qo_2b_fn = captured_ops[
                    "pallas::dsv41v2_sparse_fp8_attention_qo_2bank"
                ]
                res_qo_2b = fp8_qo_2b_fn(
                    jnp.zeros((64, 128)),
                    jnp.zeros((32, 128)),
                    jnp.zeros((128, 256)),
                    jnp.zeros((256,)),
                    jnp.zeros((128, 256)),
                    jnp.zeros((256,)),
                    jnp.zeros((64,), dtype=jnp.int32),
                    jnp.zeros((64, 64)),
                    jnp.zeros((128, 128), dtype=jnp.int32),
                    jnp.zeros((128, 128), dtype=jnp.int32),
                    jnp.zeros((128, 64), dtype=jnp.uint8),
                    jnp.zeros((128, 64), dtype=jnp.uint8),
                    jnp.zeros((32, 128), dtype=jnp.int32),
                    jnp.zeros((32, 128), dtype=jnp.int32),
                    jnp.zeros((64, 64), dtype=jnp.int32),
                    jnp.zeros((64, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((4,)),
                    jnp.zeros((4,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.zeros((64, 64)),
                    jnp.zeros((64,)),
                    jnp.zeros((64, 64)),
                    jnp.zeros((64,)),
                    jnp.ones((64,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    64,
                )
                assert res_qo_2b.shape[0] == 64

    def test_pytorch_wrappers(self, monkeypatch):
        # 1. paged_attention (1-bank and 2-bank)
        mock_paged_op = mock.MagicMock(return_value=torch.zeros(4, 2, 64))
        mock_paged_2b_op = mock.MagicMock(return_value=torch.zeros(4, 2, 64))
        monkeypatch.setattr(attention_kernels, "_attention_op", mock_paged_op)
        monkeypatch.setattr(attention_kernels, "_attention_2bank_op", mock_paged_2b_op)

        q = torch.zeros(4, 2, 64)
        swa = _FakeKVBuffer(torch.zeros(32, 64))
        main = _FakeKVBuffer(torch.zeros(32, 64))
        out_paged = paged_attention(
            q, swa, main, torch.zeros(4, 64), torch.zeros(4, 64), torch.zeros(2), 0.125
        )
        assert out_paged.shape == (4, 2, 64)
        mock_paged_op.assert_called_once()

        swa_2b = _FakeKVBuffer(torch.zeros(16, 64), raw_hi=torch.zeros(16, 64))
        main_2b = _FakeKVBuffer(torch.zeros(16, 64), raw_hi=torch.zeros(16, 64))
        out_paged_2b = paged_attention(
            q,
            swa_2b,
            main_2b,
            torch.zeros(4, 64),
            torch.zeros(4, 64),
            torch.zeros(2),
            0.125,
        )
        assert out_paged_2b.shape == (4, 2, 64)
        mock_paged_2b_op.assert_called_once()

        # 2. sparse_fp8_attention_qo (pages is None vs pages provided, 1-bank & 2-bank)
        mock_qo_op = mock.MagicMock(return_value=torch.zeros(16, 32))
        mock_qo_2b_op = mock.MagicMock(return_value=torch.zeros(16, 32))
        monkeypatch.setattr(attention_kernels, "_fp8_attention_qo_op", mock_qo_op)
        monkeypatch.setattr(
            attention_kernels, "_fp8_attention_qo_2bank_op", mock_qo_2b_op
        )
        main_rope = _FakeKVBuffer(torch.zeros(32, 64), rope=torch.zeros(32, 32))
        main_rope_2b = _FakeKVBuffer(
            torch.zeros(16, 64),
            rope=torch.zeros(16, 32),
            raw_hi=torch.zeros(16, 64),
            rope_hi=torch.zeros(16, 32),
        )

        # pages is None branch (1-bank)
        sparse_fp8_attention_qo(
            torch.zeros(16, 64),
            torch.zeros(8, 64),
            torch.zeros(64, 128),
            torch.zeros(128),
            torch.zeros(64, 128),
            torch.zeros(128),
            torch.zeros(16, dtype=torch.int64),
            torch.zeros(16, 32),
            swa,
            main_rope,
            torch.zeros(16, 64),
            torch.zeros(16, 128),
            torch.zeros(4),
            torch.zeros(4),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(64, 32),
            torch.zeros(32),
            torch.zeros(64, 32),
            torch.zeros(32),
            torch.ones(16, dtype=torch.int32),
            0.125,
            pages=None,
            starts=None,
            max_requests=None,
            precompacted=False,
            head_dim=64,
        )
        assert mock_qo_op.call_count == 1

        # pages is None branch (2-bank)
        sparse_fp8_attention_qo(
            torch.zeros(16, 64),
            torch.zeros(8, 64),
            torch.zeros(64, 128),
            torch.zeros(128),
            torch.zeros(64, 128),
            torch.zeros(128),
            torch.zeros(16, dtype=torch.int64),
            torch.zeros(16, 32),
            swa_2b,
            main_rope_2b,
            torch.zeros(16, 64),
            torch.zeros(16, 128),
            torch.zeros(4),
            torch.zeros(4),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(64, 32),
            torch.zeros(32),
            torch.zeros(64, 32),
            torch.zeros(32),
            torch.ones(16, dtype=torch.int32),
            0.125,
            pages=None,
            starts=None,
            max_requests=None,
            precompacted=False,
            head_dim=64,
        )
        assert mock_qo_2b_op.call_count == 1

        # pages is not None branch (1221 -> 1225)
        sparse_fp8_attention_qo(
            torch.zeros(16, 64),
            torch.zeros(8, 64),
            torch.zeros(64, 128),
            torch.zeros(128),
            torch.zeros(64, 128),
            torch.zeros(128),
            torch.zeros(16, dtype=torch.int64),
            torch.zeros(16, 32),
            swa,
            main_rope,
            torch.zeros(16, 64),
            torch.zeros(16, 128),
            torch.zeros(4),
            torch.zeros(4),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(64, 32),
            torch.zeros(32),
            torch.zeros(64, 32),
            torch.zeros(32),
            torch.ones(16, dtype=torch.int32),
            0.125,
            pages=torch.zeros((1, 1), dtype=torch.int32),
            starts=torch.tensor([0, 16], dtype=torch.int32),
            max_requests=None,
            precompacted=False,
            head_dim=64,
        )
        assert mock_qo_op.call_count == 2

        # 3. sparse_fp8_attention wrapper (lines 1283-1296)
        mock_fp8_op = mock.MagicMock(return_value=torch.zeros(4, 2, 64))
        mock_fp8_2b_op = mock.MagicMock(return_value=torch.zeros(4, 2, 64))
        monkeypatch.setattr(attention_kernels, "_fp8_attention_op", mock_fp8_op)
        monkeypatch.setattr(
            attention_kernels, "_fp8_attention_2bank_op", mock_fp8_2b_op
        )

        # Call with defaults (pages=None, full_sink=None, tp_rank=None)
        out_fp8_def = sparse_fp8_attention(
            q,
            swa,
            main_rope,
            torch.zeros(4, 64, dtype=torch.int32),
            torch.zeros(4, 128, dtype=torch.int32),
            torch.zeros(2),
            0.125,
            pages=None,
            starts=None,
            full_sink=None,
            tp_rank=None,
        )
        assert out_fp8_def.shape == (4, 2, 64)
        assert mock_fp8_op.call_count == 1

        # Call with 2-bank
        out_fp8_2b = sparse_fp8_attention(
            q,
            swa_2b,
            main_rope_2b,
            torch.zeros(4, 64, dtype=torch.int32),
            torch.zeros(4, 128, dtype=torch.int32),
            torch.zeros(2),
            0.125,
            pages=None,
            starts=None,
            full_sink=None,
            tp_rank=None,
        )
        assert out_fp8_2b.shape == (4, 2, 64)
        assert mock_fp8_2b_op.call_count == 1

        # Call with explicit arguments (pages, full_sink, tp_rank provided)
        out_fp8_exp = sparse_fp8_attention(
            q,
            swa,
            main_rope,
            torch.zeros(4, 64, dtype=torch.int32),
            torch.zeros(4, 128, dtype=torch.int32),
            torch.zeros(2),
            0.125,
            pages=torch.zeros((1, 1), dtype=torch.int32),
            starts=torch.tensor([0, 4], dtype=torch.int32),
            full_sink=torch.zeros(2),
            tp_rank=torch.tensor([0], dtype=torch.int32),
        )
        assert out_fp8_exp.shape == (4, 2, 64)
        assert mock_fp8_op.call_count == 2


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
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels._sparsecore_top_k",
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
                candidate_filter=False,
            )

        # 2. candidate_write and candidate_filter both True
        with pytest.raises(ValueError, match="Candidate sources must not filter"):
            select_indices_jax(
                scores,
                lengths,
                candidates,
                topk=8,
                block_size=16,
                candidate_write=True,
                candidate_filter=True,
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
                candidate_filter=False,
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
            candidate_filter=False,
        )
        assert selected_w.shape == (4, 4)
        assert published_w.shape == (4, 2)
        assert published_w[0, 0] >= 0

        # 2. candidate_filter with batch_max <= topk short-context bypass
        selected_f, published_f = select_indices_jax(
            scores,
            lengths,
            candidates,
            topk=64,
            block_size=16,
            candidate_write=False,
            candidate_filter=True,
        )
        assert selected_f.shape == (4, 64)

        # 3. candidate_filter without short-circuit
        candidates_valid = jnp.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=jnp.int32)
        selected_f2, _ = select_indices_jax(
            scores,
            lengths,
            candidates_valid,
            topk=4,
            block_size=16,
            candidate_write=False,
            candidate_filter=True,
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
            candidate_filter=False,
            unordered=True,
        )
        assert selected_unord.shape == (4, 4)

        # 5. unordered=True, width >= 2048 (SparseCore fast path)
        scores_2048 = jnp.zeros((4, 2048), dtype=jnp.float32).at[:, :16].set(1.0)
        lengths_2048 = jnp.full((4,), 2048, dtype=jnp.int32)
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels._sparsecore_top_k",
            return_value=jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (4, 16)),
        ) as mock_sc_sel:
            selected_sc, _ = select_indices_jax(
                scores_2048,
                lengths_2048,
                candidates_valid,
                topk=16,
                block_size=16,
                candidate_write=False,
                candidate_filter=False,
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

        # 1. Full context bypass: not candidate_filter
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
            candidate_filter=False,
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
            candidate_filter=True,
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
            candidate_filter=True,
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
            candidate_filter=True,
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
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels.paged_index_scores_jax",
                side_effect=lambda q, pool, rec, *a, **kw: jnp.zeros(
                    (q.shape[0], rec.shape[1]), dtype=jnp.float32
                ),
            ),
            mock.patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels._sparsecore_top_k",
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
                candidate_filter=True,
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
                np.asarray(attention_kernels._compact_selected(want_sel)),
            )

    def test_write_index_keys_and_update_and_select(self):
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

        # 4. update_and_select_jax
        q = jnp.zeros((4, 2, 128), dtype=jnp.bfloat16)
        records = jnp.zeros((2, 64), dtype=jnp.int32)
        weights = jnp.ones((4, 2), dtype=jnp.float32)
        lengths = jnp.full((4,), 64, dtype=jnp.int32)
        candidates = jnp.full((4, 2), 0, dtype=jnp.int32)
        requests = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        batch_max = jnp.array([64], dtype=jnp.int32)

        up_pool, sel, pub = update_and_select_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            write_records,
            keys_bf16,
            requests,
            batch_max,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            unordered=False,
        )
        assert up_pool.shape == pool.shape
        assert sel.shape == (4, 8)
        assert pub.shape == (4, 2)

        # 5. update_and_select_2bank_jax
        up_lo, up_hi, sel_2b, pub_2b = attention_kernels.update_and_select_2bank_jax(
            q,
            pool[:32],
            pool[32:],
            records,
            weights,
            lengths,
            candidates,
            write_records,
            keys_bf16,
            requests,
            batch_max,
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            unordered=False,
        )
        assert jnp.array_equal(jnp.concatenate((up_lo, up_hi), axis=0), up_pool)
        assert jnp.array_equal(sel_2b, sel)
        assert jnp.array_equal(pub_2b, pub)

    def test_initialize_indexer_and_paged_indexer_wrapper(self, monkeypatch):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        captured_fake = {}

        def mock_jax_op(name, fn, **kwargs):
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
            attention_kernels._indexer_op = None
            initialize_indexer()
            assert "pallas::dsv41v2_exact_indexer" in captured_fake
            fake_fn = captured_fake["pallas::dsv41v2_exact_indexer"]
            q = torch.zeros(4, 2, 64)
            pool = torch.zeros(64, 128)
            records = torch.zeros(2, 64, dtype=torch.int32)
            cands = torch.zeros(4, 2, dtype=torch.int32)
            f_pool, f_sel, f_cands = fake_fn(
                q,
                pool,
                records,
                torch.ones(4, 2),
                torch.full((4,), 64),
                cands,
                torch.zeros(2),
                torch.zeros(2, 128),
                torch.zeros(4),
                torch.tensor(64),
                topk=8,
                block_size=16,
                candidate_write=False,
                candidate_filter=False,
                unordered=False,
            )
            assert f_pool.shape == pool.shape
            assert f_sel.shape == (4, 8)
            assert f_cands.shape == cands.shape

            assert "pallas::dsv41v2_exact_indexer_2bank" in captured_fake
            fake_2b_fn = captured_fake["pallas::dsv41v2_exact_indexer_2bank"]
            f_lo, f_hi, f_sel_2b, f_cands_2b = fake_2b_fn(
                q,
                pool[:32],
                pool[32:],
                records,
                torch.ones(4, 2),
                torch.full((4,), 64),
                cands,
                torch.zeros(2),
                torch.zeros(2, 128),
                torch.zeros(4),
                torch.tensor(64),
                topk=8,
                block_size=16,
                candidate_write=False,
                candidate_filter=False,
                unordered=False,
            )
            assert f_lo.shape == (32, 128)
            assert f_hi.shape == (32, 128)
            assert f_sel_2b.shape == (4, 8)
            assert f_cands_2b.shape == cands.shape

            # Test early return idempotency
            saved = attention_kernels._indexer_op
            initialize_indexer()
            assert attention_kernels._indexer_op is saved

        # Test paged_indexer wrapper with batch_max=None and batch_max provided
        mock_idx_op = mock.MagicMock(
            return_value=(torch.zeros(1), torch.zeros(4, 8), torch.zeros(4, 2))
        )
        mock_idx_2b_op = mock.MagicMock(
            return_value=(
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(4, 8),
                torch.zeros(4, 2),
            )
        )
        monkeypatch.setattr(attention_kernels, "_indexer_op", mock_idx_op)
        monkeypatch.setattr(attention_kernels, "_indexer_2bank_op", mock_idx_2b_op)

        cache = SimpleNamespace(raw=torch.zeros(64, 64))
        paged_indexer(
            torch.zeros(4, 2, 64),
            cache,
            torch.zeros(2, 64),
            torch.ones(4, 2),
            torch.full((4,), 64),
            torch.zeros(4, 2),
            torch.zeros(2),
            torch.zeros(2, 64),
            torch.zeros(4),
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
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
            torch.zeros(2),
            torch.zeros(2, 64),
            torch.zeros(4),
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            batch_max=torch.tensor([64]),
        )
        assert mock_idx_op.call_count == 2

        cache_2b = SimpleNamespace(raw=torch.zeros(32, 64), raw_hi=torch.zeros(32, 64))
        up_2b, _, _ = paged_indexer(
            torch.zeros(4, 2, 64),
            cache_2b,
            torch.zeros(2, 64),
            torch.ones(4, 2),
            torch.full((4,), 64),
            torch.zeros(4, 2),
            torch.zeros(2),
            torch.zeros(2, 64),
            torch.zeros(4),
            topk=8,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            batch_max=torch.tensor([64]),
        )
        assert isinstance(up_2b, tuple) and len(up_2b) == 2
        assert mock_idx_2b_op.call_count == 1


@pytest.mark.parametrize("mode", ["plain", "publish", "filter"])
@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize("unordered", [False, True])
def test_paged_indexer_query_tiles_preserve_selection(
    monkeypatch, mode, banks, unordered
):
    """Bound score rows, including a partial tile, without changing selection."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import _pack_bf16_words

    rng = np.random.default_rng(17)
    rows, width, topk = 65, 128, 16
    keys = jnp.asarray(rng.standard_normal((2 * width, 128)), jnp.bfloat16)
    pool = _pack_bf16_words(keys).reshape(-1, 128)
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
    original = attention_kernels.paged_index_scores_jax

    def record_shape(queries, *args, **kwargs):
        score_rows.append(queries.shape[0])
        return original(queries, *args, **kwargs)

    monkeypatch.setattr(attention_kernels, "paged_index_scores_jax", record_shape)
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
    assert score_rows and max(score_rows) <= attention_kernels.indexer.QUERY_TILE


@pytest.mark.parametrize("length", [0, 127, 128, 129, 255])
@pytest.mark.parametrize("banks", [1, 2])
def test_paged_indexer_context_bucket_preserves_boundary_and_fallback(
    monkeypatch, length, banks
):
    """Bucket edges and the long-context fallback preserve exact selection."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import _pack_bf16_words

    monkeypatch.setattr(
        attention_kernels.tuned_block_sizes, "PAGED_INDEXER_PREFILL_BUCKET", (128, 64)
    )
    rng = np.random.default_rng(23)
    rows, width, topk = 129, 256, 16
    keys = jnp.asarray(rng.standard_normal((width, 128)), jnp.bfloat16)
    pool = _pack_bf16_words(keys).reshape(-1, 128)
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
    original = attention_kernels.paged_index_scores_jax

    def record_shape(queries, pool, records, *args, **kwargs):
        shapes.append((queries.shape[0], records.shape[1]))
        return original(queries, pool, records, *args, **kwargs)

    monkeypatch.setattr(attention_kernels, "paged_index_scores_jax", record_shape)
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
def test_paged_indexer_decode_rows_skip_query_tiles(monkeypatch, banks):
    """Decode-shaped batches (few rows per request) score in one query tile.

    DSpark verification has K + 1 rows per request. Query tiles smaller than
    the request-row count would defeat grouped per-request key sharing.
    """
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import _pack_bf16_words

    rng = np.random.default_rng(31)
    requests_count, group, width, topk = 16, 6, 256, 16
    rows = requests_count * group
    keys = jnp.asarray(rng.standard_normal((requests_count * width, 128)), jnp.bfloat16)
    pool = _pack_bf16_words(keys).reshape(-1, 128)
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
    original = attention_kernels.paged_index_scores_jax

    def record_shape(queries, *args, **kwargs):
        score_rows.append(queries.shape[0])
        return original(queries, *args, **kwargs)

    monkeypatch.setattr(attention_kernels, "paged_index_scores_jax", record_shape)
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
    assert rows > attention_kernels.indexer.QUERY_TILE
    assert score_rows == [rows]
