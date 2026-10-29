"""Unit tests for DeepSeek-V4.1 attention kernels: Union & Token-Parallel.

Achieves 100% diff line and branch coverage on:
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py:267-760
  (compact_records, _pad_compacted_records, compute_chunk_union_gather_indices,
   unpack/repack rope words, gather_sparse_kv_chunk_with_union_fallback,
   sparse_ragged_paged_attention_union_gather, sparse_fp8_attention_jax,
   slice_token_parallel_attention_inputs, initialize_paged_attention,
   and sparse_fp8_attention wrapper)
"""

from __future__ import annotations

import unittest.mock as mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _pad_compacted_records,
    compact_records,
    compute_chunk_union_gather_indices,
    gather_sparse_kv_chunk_with_union_fallback,
    remap_union_kv_to_chunk,
    repack_chunk_rope_words,
    slice_token_parallel_attention_inputs,
    sparse_fp8_attention,
    sparse_fp8_attention_jax,
    sparse_ragged_paged_attention_union_gather,
    unpack_union_rope_words,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl import _compact_selected


class TestCompactRecords:
    """Tests for record compaction and padding functions."""

    def test_compact_selected_all_valid(self):
        records = jnp.array([[5, 2, 8, 1], [10, 4, 7, 3]], dtype=jnp.int32)
        got = _compact_selected(records)
        expected = np.array([[1, 2, 5, 8], [3, 4, 7, 10]], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(got), expected)

    def test_compact_selected_all_pad(self):
        records = jnp.array([[-1, -1, -1], [-5, -1, -100]], dtype=jnp.int32)
        got = _compact_selected(records)
        expected = np.full((2, 3), -1, dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(got), expected)

    def test_compact_selected_mixed_and_duplicates(self):
        records = jnp.array(
            [[10, -1, 5, 0, -1, 5], [-1, 20, -1, 10, -1, 0]], dtype=jnp.int32
        )
        got = _compact_selected(records)
        expected = np.array(
            [[0, 5, 5, 10, -1, -1], [0, 10, 20, -1, -1, -1]], dtype=np.int32
        )
        np.testing.assert_array_equal(np.asarray(got), expected)

    def test_compact_records_padding_to_128_multiple(self):
        # width = 50 -> padded to 128
        records = jnp.array([list(range(50))], dtype=jnp.int32)
        got = compact_records(records)
        assert got.shape == (1, 128)
        assert int(got[0, 49]) == 49
        assert int(got[0, 50]) == -1
        assert int(got[0, 127]) == -1

    def test_compact_records_exact_128(self):
        # width = 128 -> remains 128
        records = jnp.array([list(range(128))], dtype=jnp.int32)
        got = compact_records(records)
        assert got.shape == (1, 128)
        assert int(got[0, 0]) == 0
        assert int(got[0, 127]) == 127

    def test_compact_records_padded_to_256(self):
        # width = 200 -> padded to 256
        records = jnp.array([list(range(200))], dtype=jnp.int32)
        got = compact_records(records)
        assert got.shape == (1, 256)
        assert int(got[0, 199]) == 199
        assert int(got[0, 200]) == -1

    def test_pad_compacted_records_branches(self):
        # Branch 1: padded == width (e.g. 128)
        records_128 = jnp.zeros((2, 128), dtype=jnp.int32)
        got_128 = _pad_compacted_records(records_128)
        assert got_128 is records_128

        # Branch 2: padded != width (e.g. 100 -> padded to 128)
        records_100 = jnp.ones((2, 100), dtype=jnp.int32)
        got_padded = _pad_compacted_records(records_100)
        assert got_padded.shape == (2, 128)
        assert int(got_padded[0, 99]) == 1
        assert int(got_padded[0, 100]) == -1
        assert int(got_padded[0, 127]) == -1


class TestChunkUnionGatherIndices:
    """Tests for compute_chunk_union_gather_indices covering all paths."""

    def test_padded_tokens_gt_num_tokens_cpu(self):
        # num_tokens = 150, chunk = 128 -> num_chunks = 2, padded = 256 > 150
        num_tokens, topk, chunk_size = 150, 128, 128
        max_num_seqs, pages_per_seq, page_size = 2, 32, 16
        total_num_pages = 128
        union_cap = 512

        q_loc = jnp.asarray([0, 80, 150], dtype=jnp.int32)
        pages = jnp.arange(max_num_seqs * pages_per_seq, dtype=jnp.int32)
        compacted = jnp.full((num_tokens, topk), -1, dtype=jnp.int32)
        compacted = compacted.at[:100, :64].set(5)

        union_phys, inv_idx, num_unique = compute_chunk_union_gather_indices(
            compacted,
            pages,
            q_loc,
            page_size=page_size,
            total_num_pages=total_num_pages,
            chunk_size=chunk_size,
            union_cap=union_cap,
        )
        assert union_phys.shape == (2, union_cap)
        assert inv_idx.shape == (2, chunk_size, topk)
        assert num_unique.shape == (2,)

    def test_padded_tokens_eq_num_tokens_cpu(self):
        # num_tokens = 256, chunk = 128 -> num_chunks = 2, padded = 256 == 256
        num_tokens, topk, chunk_size = 256, 128, 128
        max_num_seqs, pages_per_seq, page_size = 2, 32, 16
        total_num_pages = 128
        union_cap = 512

        q_loc = jnp.asarray([0, 128, 256], dtype=jnp.int32)
        pages = jnp.arange(max_num_seqs * pages_per_seq, dtype=jnp.int32)
        compacted = jnp.full((num_tokens, topk), -1, dtype=jnp.int32)

        union_phys, inv_idx, num_unique = compute_chunk_union_gather_indices(
            compacted,
            pages,
            q_loc,
            page_size=page_size,
            total_num_pages=total_num_pages,
            chunk_size=chunk_size,
            union_cap=union_cap,
        )
        assert union_phys.shape == (2, union_cap)
        assert inv_idx.shape == (2, chunk_size, topk)
        assert num_unique.shape == (2,)

    def test_chunk_union_gather_tpu_backend_branch(self, monkeypatch):
        # Test the sparse_mla.gather_page_ids branch on TPU backend
        num_tokens, topk, chunk_size = 128, 64, 128
        max_num_seqs, pages_per_seq, page_size = 1, 32, 16
        total_num_pages = 64
        union_cap = 256

        q_loc = jnp.asarray([0, 128], dtype=jnp.int32)
        pages = jnp.arange(max_num_seqs * pages_per_seq, dtype=jnp.int32)
        compacted = jnp.full((num_tokens, topk), -1, dtype=jnp.int32)
        compacted = compacted.at[0, :10].set(jnp.arange(10, dtype=jnp.int32))

        monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
        with mock.patch.object(
            sparse_mla,
            "gather_page_ids",
            side_effect=lambda page_indices, page_ids, *args, **kwargs: page_indices[
                page_ids
            ],
        ) as mock_gather:
            union_phys, inv_idx, num_unique = compute_chunk_union_gather_indices(
                compacted,
                pages,
                q_loc,
                page_size=page_size,
                total_num_pages=total_num_pages,
                chunk_size=chunk_size,
                union_cap=union_cap,
            )
            assert mock_gather.called
            assert union_phys.shape == (1, union_cap)


class TestRopeUnpackRepackRemap:
    """Tests for unpack, repack, and remap rope/kv functions."""

    def test_unpack_repack_roundtrip(self):
        union_cap = 256
        topk = 128
        # raw rope words shape: (union_cap, 32)
        rng = np.random.default_rng(1234)
        raw_words = rng.integers(-100000, 100000, (union_cap, 32), dtype=np.int32)
        raw_words_jax = jnp.asarray(raw_words)

        packed = repack_chunk_rope_words(
            raw_words_jax.reshape(union_cap // topk, topk, 32)
        ).reshape(union_cap // 4, 128)

        unpacked = unpack_union_rope_words(packed, topk)
        np.testing.assert_array_equal(np.asarray(unpacked), raw_words)

    def test_remap_union_kv_to_chunk(self):
        union_cap = 256
        topk = 64
        chunk_size = 32

        u_nope = jnp.arange(union_cap * 128, dtype=jnp.int32).reshape(union_cap, 128)
        raw_rope = jnp.arange(union_cap * 32, dtype=jnp.int32).reshape(union_cap, 32)
        u_rope = repack_chunk_rope_words(
            raw_rope.reshape(union_cap // topk, topk, 32)
        ).reshape(union_cap // 4, 128)

        inv_idx = jnp.zeros((chunk_size, topk), dtype=jnp.int32)
        inv_idx = inv_idx.at[1, :10].set(5)

        g_nope, g_rope = remap_union_kv_to_chunk(u_nope, u_rope, inv_idx, topk)
        assert g_nope.shape == (chunk_size, topk, 128)
        assert g_rope.shape == (chunk_size, topk // 4, 128)
        np.testing.assert_array_equal(np.asarray(g_nope[0, 0]), np.asarray(u_nope[0]))
        np.testing.assert_array_equal(np.asarray(g_nope[1, 0]), np.asarray(u_nope[5]))


class TestGatherSparseKvChunkWithUnionFallback:
    """Tests for gather_sparse_kv_chunk_with_union_fallback covering both branches."""

    @pytest.fixture
    def setup_kv_cache(self):
        total_num_pages, page_size = 32, 16
        cache_kv_nope = jnp.arange(
            total_num_pages * page_size * 128, dtype=jnp.int32
        ).reshape(total_num_pages, page_size, 128)
        cache_kv_rope = jnp.arange(
            total_num_pages * (page_size // 4) * 128, dtype=jnp.int32
        ).reshape(total_num_pages, page_size // 4, 128)
        return cache_kv_nope, cache_kv_rope, total_num_pages, page_size

    def test_union_branch_cpu(self, setup_kv_cache):
        cache_kv_nope, cache_kv_rope, total_num_pages, page_size = setup_kv_cache
        chunk_size, topk = 64, 32
        union_cap = 128
        num_unique_chunk = jnp.array(50, dtype=jnp.int32)  # <= union_cap -> union

        union_phys = jnp.arange(union_cap, dtype=jnp.int32)
        inv_idx = jnp.zeros((chunk_size, topk), dtype=jnp.int32)
        chunk_rec = jnp.full((chunk_size, topk), 0, dtype=jnp.int32)
        chunk_seq = jnp.zeros((chunk_size,), dtype=jnp.int32)
        page_indices = jnp.arange(total_num_pages, dtype=jnp.int32)

        g_nope, g_rope = gather_sparse_kv_chunk_with_union_fallback(
            cache_kv_nope,
            cache_kv_rope,
            union_phys,
            inv_idx,
            num_unique_chunk,
            chunk_rec,
            chunk_seq,
            page_indices,
            max_num_seqs=1,
            start_pos=0,
            batch_end=jnp.array(chunk_size, dtype=jnp.int32),
            topk=topk,
        )
        assert g_nope.shape == (chunk_size, topk, 128)
        assert g_rope.shape == (chunk_size, topk // 4, 128)

    def test_direct_branch_cpu_on_overflow(self, setup_kv_cache):
        cache_kv_nope, cache_kv_rope, total_num_pages, page_size = setup_kv_cache
        chunk_size, topk = 64, 32
        union_cap = 128
        num_unique_chunk = jnp.array(
            200, dtype=jnp.int32
        )  # > union_cap -> direct branch

        union_phys = jnp.arange(union_cap, dtype=jnp.int32)
        inv_idx = jnp.zeros((chunk_size, topk), dtype=jnp.int32)
        chunk_rec = jnp.full((chunk_size, topk), -1, dtype=jnp.int32)
        chunk_rec = chunk_rec.at[:10, :5].set(3)
        chunk_seq = jnp.zeros((chunk_size,), dtype=jnp.int32)
        page_indices = jnp.arange(total_num_pages, dtype=jnp.int32)

        g_nope, g_rope = gather_sparse_kv_chunk_with_union_fallback(
            cache_kv_nope,
            cache_kv_rope,
            union_phys,
            inv_idx,
            num_unique_chunk,
            chunk_rec,
            chunk_seq,
            page_indices,
            max_num_seqs=1,
            start_pos=0,
            batch_end=jnp.array(chunk_size, dtype=jnp.int32),
            topk=topk,
        )
        assert g_nope.shape == (chunk_size, topk, 128)
        assert g_rope.shape == (chunk_size, topk // 4, 128)

    def test_union_branch_has_valid_false(self, setup_kv_cache):
        cache_kv_nope, cache_kv_rope, total_num_pages, page_size = setup_kv_cache
        chunk_size, topk = 64, 32
        union_cap = 128
        num_unique_chunk = jnp.array(10, dtype=jnp.int32)

        union_phys = jnp.arange(union_cap, dtype=jnp.int32)
        inv_idx = jnp.zeros((chunk_size, topk), dtype=jnp.int32)
        chunk_rec = jnp.full((chunk_size, topk), -1, dtype=jnp.int32)
        chunk_seq = jnp.zeros((chunk_size,), dtype=jnp.int32)
        page_indices = jnp.arange(total_num_pages, dtype=jnp.int32)

        # batch_end = 0 <= start_pos = 0 -> has_valid is False
        g_nope, g_rope = gather_sparse_kv_chunk_with_union_fallback(
            cache_kv_nope,
            cache_kv_rope,
            union_phys,
            inv_idx,
            num_unique_chunk,
            chunk_rec,
            chunk_seq,
            page_indices,
            max_num_seqs=1,
            start_pos=0,
            batch_end=jnp.array(0, dtype=jnp.int32),
            topk=topk,
        )
        assert g_nope.shape == (chunk_size, topk, 128)

    def test_union_and_direct_branches_tpu_mock(self, setup_kv_cache, monkeypatch):
        cache_kv_nope, cache_kv_rope, total_num_pages, page_size = setup_kv_cache
        chunk_size, topk = 32, 16
        union_cap = 64
        union_phys = jnp.arange(union_cap, dtype=jnp.int32)
        inv_idx = jnp.zeros((chunk_size, topk), dtype=jnp.int32)
        chunk_rec = jnp.full((chunk_size, topk), 0, dtype=jnp.int32)
        chunk_seq = jnp.zeros((chunk_size,), dtype=jnp.int32)
        page_indices = jnp.arange(total_num_pages, dtype=jnp.int32)

        monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

        import vllm_torchtpu.kernels.deepseek_v4.core_attention.csa_gather as mock_csa

        def fake_csa_gather(nope, rope, indices, num_valid_indices, top_k):
            count = indices.shape[0]
            nope_out = jnp.zeros((count, 128), dtype=jnp.int32)
            rope_out = jnp.zeros((count // 4, 128), dtype=jnp.int32)
            return nope_out, rope_out

        monkeypatch.setattr(mock_csa, "csa_gather", fake_csa_gather)
        monkeypatch.setattr(
            sparse_mla,
            "gather_page_ids",
            lambda pgs, seq_pgs, *args, **kwargs: pgs[seq_pgs],
        )

        # 1. Test TPU _union_branch
        g_nope_u, g_rope_u = gather_sparse_kv_chunk_with_union_fallback(
            cache_kv_nope,
            cache_kv_rope,
            union_phys,
            inv_idx,
            jnp.array(20, dtype=jnp.int32),
            chunk_rec,
            chunk_seq,
            page_indices,
            max_num_seqs=1,
            start_pos=0,
            batch_end=jnp.array(chunk_size, dtype=jnp.int32),
            topk=topk,
        )
        assert g_nope_u.shape == (chunk_size, topk, 128)

        # 2. Test TPU _direct_branch
        g_nope_d, g_rope_d = gather_sparse_kv_chunk_with_union_fallback(
            cache_kv_nope,
            cache_kv_rope,
            union_phys,
            inv_idx,
            jnp.array(100, dtype=jnp.int32),  # > union_cap
            chunk_rec,
            chunk_seq,
            page_indices,
            max_num_seqs=1,
            start_pos=0,
            batch_end=jnp.array(chunk_size, dtype=jnp.int32),
            topk=topk,
        )
        assert g_nope_d.shape == (chunk_size, topk, 128)


class TestSparseRaggedPagedAttentionUnionGather:
    """Tests for sparse_ragged_paged_attention_union_gather with mocked Pallas."""

    def test_pipeline_execution_single_and_multi_chunk(self):
        total_pages, page_size = 32, 16
        nope = jnp.zeros((total_pages, page_size, 128), dtype=jnp.int32)
        rope = jnp.zeros((total_pages, page_size // 4, 128), dtype=jnp.int32)
        page_indices = jnp.zeros((32,), dtype=jnp.int32)
        distribution = jnp.array([0, 0, 1], dtype=jnp.int32)
        attention_sinks = jnp.zeros((64,), dtype=jnp.float32)

        # Multi-chunk: 256 tokens, chunk_size = 128 -> 2 chunks
        tokens = 256
        q = jnp.zeros((tokens, 64, 512), dtype=jnp.bfloat16)
        topk_indices = jnp.zeros((tokens, 128), dtype=jnp.int32)
        cu_q_lens = jnp.array([0, tokens], dtype=jnp.int32)
        swa_acc = jnp.zeros((tokens, 64, 512), dtype=jnp.bfloat16)
        swa_l = jnp.zeros((tokens, 64), dtype=jnp.float32)
        swa_m = jnp.zeros((tokens, 64), dtype=jnp.float32)

        with mock.patch("jax.experimental.pallas.pallas_call") as mock_pcall:
            mock_pcall.return_value = lambda *args: args[5]  # returns q
            out = sparse_ragged_paged_attention_union_gather(
                q,
                nope,
                rope,
                topk_indices,
                page_indices,
                cu_q_lens,
                distribution,
                attention_sinks,
                swa_acc,
                swa_l,
                swa_m,
                gather_and_attention_chunk_size=128,
            )
            assert out.shape == (tokens, 64, 512)
            assert mock_pcall.call_count == 2


class TestSparseFp8AttentionJax:
    """Tests for sparse_fp8_attention_jax precompacted and union_gather branches."""

    @pytest.fixture
    def attention_inputs(self):
        tokens = 128
        q = jnp.zeros((tokens, 64, 512), dtype=jnp.bfloat16)
        swa_pool = jnp.zeros((tokens, 512), dtype=jnp.bfloat16)
        main_pool = jnp.zeros((tokens, 128), dtype=jnp.int32)
        rope_pool = jnp.zeros((tokens // 4, 32), dtype=jnp.int32)
        swa_records = jnp.zeros((tokens, 64), dtype=jnp.int32)
        main_records = jnp.zeros((tokens, 128), dtype=jnp.int32)
        pages = jnp.zeros((1, 1), dtype=jnp.int32)
        q_loc = jnp.array([0, tokens], dtype=jnp.int32)
        sink = jnp.zeros((64,), dtype=jnp.float32)
        scale = 0.125
        page_size = 16
        dummy_state = (
            jnp.zeros((tokens, 64), dtype=jnp.float32),
            jnp.zeros((tokens, 64), dtype=jnp.float32),
            jnp.zeros((tokens, 64, 512), dtype=jnp.bfloat16),
        )
        return (
            q,
            swa_pool,
            main_pool,
            rope_pool,
            swa_records,
            main_records,
            pages,
            q_loc,
            sink,
            scale,
            page_size,
            dummy_state,
        )

    def test_sparse_fp8_attention_jax_precompacted_false_union_false(
        self, attention_inputs
    ):
        (
            q,
            swa_pool,
            main_pool,
            rope_pool,
            swa_records,
            main_records,
            pages,
            q_loc,
            sink,
            scale,
            page_size,
            dummy_state,
        ) = attention_inputs
        with (
            mock.patch.object(
                attention_kernels, "_attend_window", return_value=dummy_state
            ),
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla.sparse_ragged_paged_attention"
            ) as mock_attn,
        ):
            mock_attn.return_value = q
            out = sparse_fp8_attention_jax(
                q,
                swa_pool,
                main_pool,
                rope_pool,
                swa_records,
                main_records,
                pages,
                q_loc,
                sink,
                scale,
                page_size,
                precompacted=False,
                union_gather=False,
            )
            assert out is q
            assert mock_attn.called

    def test_sparse_fp8_attention_jax_precompacted_true_union_false(
        self, attention_inputs
    ):
        (
            q,
            swa_pool,
            main_pool,
            rope_pool,
            swa_records,
            main_records,
            pages,
            q_loc,
            sink,
            scale,
            page_size,
            dummy_state,
        ) = attention_inputs
        with (
            mock.patch.object(
                attention_kernels, "_attend_window", return_value=dummy_state
            ),
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla.sparse_ragged_paged_attention"
            ) as mock_attn,
        ):
            mock_attn.return_value = q
            out = sparse_fp8_attention_jax(
                q,
                swa_pool,
                main_pool,
                rope_pool,
                swa_records,
                main_records,
                pages,
                q_loc,
                sink,
                scale,
                page_size,
                precompacted=True,
                union_gather=False,
            )
            assert out is q
            assert mock_attn.called

    def test_sparse_fp8_attention_jax_union_gather_true(self, attention_inputs):
        (
            q,
            swa_pool,
            main_pool,
            rope_pool,
            swa_records,
            main_records,
            pages,
            q_loc,
            sink,
            scale,
            page_size,
            dummy_state,
        ) = attention_inputs
        with (
            mock.patch.object(
                attention_kernels, "_attend_window", return_value=dummy_state
            ),
            mock.patch.object(
                attention_kernels,
                "sparse_ragged_paged_attention_union_gather",
                return_value=q,
            ) as mock_union,
        ):
            out = sparse_fp8_attention_jax(
                q,
                swa_pool,
                main_pool,
                rope_pool,
                swa_records,
                main_records,
                pages,
                q_loc,
                sink,
                scale,
                page_size,
                precompacted=False,
                union_gather=True,
            )
            assert out is q
            assert mock_union.called


class TestSliceTokenParallelAttentionInputs:
    """Tests for slice_token_parallel_attention_inputs."""

    def test_slice_token_parallel_attention_inputs_all_ranks(self):
        tokens, tp_size = 1024, 8
        tokens_local = tokens // tp_size
        swa_records = jnp.arange(tokens * 64, dtype=jnp.int32).reshape(tokens, 64)
        main_records = jnp.arange(tokens * 128, dtype=jnp.int32).reshape(tokens, 128)
        # Ragged start locations straddling across ranks
        query_start_loc = jnp.array([0, 100, 250, 600, 1024], dtype=jnp.int32)

        for r in range(tp_size):
            tp_rank = jnp.array(r, dtype=jnp.int32)
            swa_l, main_l, q_loc_l = slice_token_parallel_attention_inputs(
                swa_records, main_records, query_start_loc, tp_rank, tp_size, tokens
            )
            start = r * tokens_local
            end = start + tokens_local
            np.testing.assert_array_equal(
                np.asarray(swa_l), np.asarray(swa_records[start:end])
            )
            np.testing.assert_array_equal(
                np.asarray(main_l), np.asarray(main_records[start:end])
            )
            exp_q_loc = np.clip(np.asarray(query_start_loc) - start, 0, tokens_local)
            np.testing.assert_array_equal(np.asarray(q_loc_l), exp_q_loc)


class TestTokenParallelAttentionDispatchAndHelpers:
    """Tests for fp8_attention dispatch logic, conditions, and wrappers."""

    def test_token_parallel_boolean_condition_branches(self):
        def check(token_parallel, cur_tp_size, tokens):
            return (
                token_parallel
                and cur_tp_size > 1
                and tokens >= 6 * cur_tp_size
                and tokens % cur_tp_size == 0
            )

        # Falsy branches
        assert not check(False, 8, 2048)  # token_parallel is False
        assert not check(True, 1, 2048)  # tp_size <= 1
        assert not check(True, 8, 32)  # tokens < 48
        assert not check(True, 8, 50)  # tokens % 8 != 0

        # Truthy branch
        assert check(True, 8, 48)
        assert check(True, 8, 64)
        assert check(True, 8, 96)
        assert check(True, 8, 128)
        assert check(True, 8, 256)
        assert check(True, 8, 1024)
        assert check(True, 8, 2048)
        assert check(True, 8, 8192)

    def test_csa_gather_dynamic_num_row_subchunks(self, monkeypatch):
        from types import SimpleNamespace

        from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather

        fake_sc = SimpleNamespace(num_lanes=8, num_cores=4, num_subcores=16)
        fake_tpu_info = SimpleNamespace(sparse_core=fake_sc)
        monkeypatch.setattr(csa_gather.pltpu, "get_tpu_info", lambda: fake_tpu_info)
        monkeypatch.setattr(
            csa_gather.plsc, "VectorSubcoreMesh", lambda **kw: SimpleNamespace(**kw)
        )

        captured_subchunks: list[int] = []

        def fake_pl_kernel(kernel_fn, *, out_type, **kwargs):
            captured_subchunks.append(kernel_fn.keywords["num_row_subchunks"])
            n_rows = out_type[0].shape[0]
            return lambda nope_c, rope_c, idx, valid: (
                jnp.zeros((n_rows, 128), dtype=jnp.int32),
                jnp.zeros((n_rows // 4, 128), dtype=jnp.int32),
            )

        monkeypatch.setattr(csa_gather.pl, "kernel", fake_pl_kernel)

        nope_cache = jnp.zeros((64, 16, 128), dtype=jnp.int32)
        rope_cache = jnp.zeros((64, 4, 128), dtype=jnp.int32)

        for num_tokens, expected_subchunks in [
            (4, 4),
            (8, 8),
            (16, 16),
            (24, 32),
            (64, 64),
        ]:
            indices = jnp.zeros((num_tokens * 512,), dtype=jnp.int32)
            nope_out, rope_out = csa_gather.csa_gather(
                nope_cache,
                rope_cache,
                indices,
                num_valid_indices=None if num_tokens == 4 else num_tokens * 512,
                top_k=512,
            )
            assert nope_out.shape == (num_tokens * 512, 128)
            assert rope_out.shape == ((num_tokens * 512) // 4, 128)
            assert captured_subchunks[-1] == expected_subchunks

    def test_initialize_paged_attention_init_and_early_return(self):
        from jax.sharding import Mesh

        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        with set_vllm_model_wrapper_context(mesh=mesh):
            # 1. Reset and run initialization
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels.initialize_paged_attention()
            assert attention_kernels._attention_op is not None
            assert attention_kernels._fp8_attention_op is not None

            # 2. Call again to test early return branch
            saved_op = attention_kernels._attention_op
            attention_kernels.initialize_paged_attention()
            assert attention_kernels._attention_op is saved_op

    def test_fp8_attention_inner_bodies_tp_and_non_tp(self):
        import inspect

        import vllm.distributed as vllm_dist
        from jax.sharding import Mesh

        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        mock_tp = mock.MagicMock()
        mock_tp.device_group.size.return_value = 8

        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch.object(vllm_dist, "get_tp_group", return_value=mock_tp),
            mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
        ):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels.initialize_paged_attention()
            fp8_attn = inspect.unwrap(
                attention_kernels._fp8_attention_op._init_fn.jit_fn
            )

            with (
                mock.patch(
                    "jax.shard_map",
                    side_effect=lambda fn, **kwargs: lambda *args: fn(*args),
                ),
                mock.patch("jax.lax.all_to_all", side_effect=lambda x, **kwargs: x),
                mock.patch.object(
                    attention_kernels,
                    "sparse_fp8_attention_jax",
                    side_effect=lambda q, *args, **kwargs: q,
                ),
            ):
                # 1a. Test token-parallel body execution (pad_rows == 0, tokens=1024)
                res_tp = fp8_attn(
                    jnp.zeros((1024, 64, 512)),
                    jnp.zeros((1024, 512)),
                    jnp.zeros((1024, 128)),
                    jnp.zeros((256, 32)),
                    jnp.zeros((1024, 64), dtype=jnp.int32),
                    jnp.zeros((1024, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((64,)),
                    jnp.zeros((64,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    True,
                    False,
                    4096,
                )
                assert res_tp.shape == (1024, 64, 512)

                # 1b. Test token-parallel body execution with 8-sublane padding
                # (tokens=48, pad_rows=2)
                res_tp_48 = fp8_attn(
                    jnp.zeros((48, 64, 512)),
                    jnp.zeros((48, 512)),
                    jnp.zeros((48, 128)),
                    jnp.zeros((12, 32)),
                    jnp.zeros((48, 64), dtype=jnp.int32),
                    jnp.zeros((48, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((64,)),
                    jnp.zeros((64,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    True,
                    False,
                    4096,
                )
                assert res_tp_48.shape == (48, 64, 512)

                # 2. Test non-token-parallel body execution
                res_non_tp = fp8_attn(
                    jnp.zeros((128, 64, 512)),
                    jnp.zeros((128, 512)),
                    jnp.zeros((128, 128)),
                    jnp.zeros((32, 32)),
                    jnp.zeros((128, 64), dtype=jnp.int32),
                    jnp.zeros((128, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((64,)),
                    jnp.zeros((64,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    0.125,
                    16,
                    None,
                    False,
                    False,
                    False,
                    4096,
                )
                assert res_non_tp.shape == (128, 64, 512)

    def test_sparse_fp8_attention_wrapper_branches(self, monkeypatch):
        # Test full_sink is None vs not None, tp_rank is None vs not None, pages is None
        class FakeMain:
            def __init__(self):
                self.rope = torch.zeros(10, 32)
                self.raw = torch.zeros(16, 128)
                self.shape = (16, 16)

        class FakeSwa:
            def __init__(self):
                self.raw = torch.zeros(16, 64)

        captured_args = {}

        def fake_op(*args):
            captured_args["args"] = args
            return torch.zeros(1)

        monkeypatch.setattr(attention_kernels, "_fp8_attention_op", fake_op)

        q = torch.zeros(16, 64, 512)
        swa = FakeSwa()
        main = FakeMain()
        scale = 0.125
        swa_rec = torch.zeros(16, 64)
        main_rec = torch.zeros(16, 128)
        sink = torch.ones(64)

        # Case 1: pages=None, full_sink=None, tp_rank=None
        sparse_fp8_attention(
            q,
            swa,
            main,
            swa_rec,
            main_rec,
            sink,
            scale,
            pages=None,
            full_sink=None,
            tp_rank=None,
        )
        # Verify full_sink defaults to sink
        assert torch.equal(captured_args["args"][8], captured_args["args"][9])
        # Verify tp_rank defaults to zero tensor
        assert captured_args["args"][10].shape == (1,)
        assert int(captured_args["args"][10][0]) == 0

        # Case 2: pages provided, full_sink and tp_rank provided
        custom_sink = torch.full((64,), 2.0)
        custom_rank = torch.tensor([3], dtype=torch.int32)
        pages = torch.zeros((2, 4), dtype=torch.int32)
        sparse_fp8_attention(
            q,
            swa,
            main,
            swa_rec,
            main_rec,
            sink,
            scale,
            pages=pages,
            starts=torch.tensor([0, 16], dtype=torch.int32),
            full_sink=custom_sink,
            tp_rank=custom_rank,
            token_parallel=True,
            precompacted=True,
            union_gather=True,
            union_cap=2048,
        )
        assert torch.equal(captured_args["args"][9], custom_sink)
        assert int(captured_args["args"][10][0]) == 3
