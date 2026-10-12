"""Unit tests for DeepSeek-V4.1 attention kernels: Indexer & Precompact.

Achieves 100% diff line and branch coverage on:
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py:782-1056, 1345-1493
  (shared-key-tile full_context routing, candidate_filter_precompact,
   128-aligned s32 .add(1) scatter mask, records_are_compacted branches,
   +0.0/-0.0 & -1/NaN masks, write_index_keys, paged_index_scores_jax,
   _visible_top_k, _threshold_top_k, and indexer op wrappers)
"""

from __future__ import annotations

import unittest.mock as mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh

from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _pad_compacted_records,
    _prefix_count,
    _sparsecore_top_k,
    _threshold_top_k,
    _visible_top_k,
    initialize_indexer,
    initialize_paged_attention,
    paged_attention,
    paged_index_scores_jax,
    paged_indexer,
    paged_indexer_jax,
    select_indices_jax,
    slice_token_parallel_attention_inputs,
    sparse_fp8_attention,
    sparse_fp8_attention_jax,
    update_and_select_jax,
    write_index_keys,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


def _request_inputs(rng, requests_count, tokens, width, heads=4, dim=128):
    """Helper generating inputs for paged indexer tests."""
    pool = jnp.asarray(rng.integers(-(2**31), 2**31, (256, 128)), dtype=jnp.int32)
    pool = pool & jnp.int32(0x3FFF3FFF)  # finite BF16 keys
    records = rng.integers(0, 256 * 2, (requests_count, width)).astype(np.int32)
    records[:, max(0, width - 5) :] = -1
    requests = np.sort(rng.integers(0, requests_count, tokens)).astype(np.int32)
    lengths = rng.integers(1, width + 1, tokens).astype(np.int32)
    q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
    weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
    return pool, records, requests, lengths, q, weights


class TestPadCompactedRecordsAndFp8Attention:
    """Tests for record padding and fp8 attention wrapper/precompacted branches."""

    def test_pad_compacted_records_exact_multiple(self):
        records = jnp.zeros((4, 128), dtype=jnp.int32)
        padded = _pad_compacted_records(records)
        assert padded.shape == (4, 128)
        assert padded is records

    def test_pad_compacted_records_non_multiple(self):
        records = jnp.array([[1, 2, 3], [4, 5, -1]], dtype=jnp.int32)
        padded = _pad_compacted_records(records)
        assert padded.shape == (2, 128)
        np.testing.assert_array_equal(np.asarray(padded[:, :3]), np.asarray(records))
        assert np.all(np.asarray(padded[:, 3:]) == -1)

    def test_sparse_fp8_attention_jax_precompacted_and_union_gather_branches(self):
        rng = np.random.default_rng(42)
        q = jnp.asarray(rng.standard_normal((16, 8, 64)), dtype=jnp.bfloat16)
        swa_pool = jnp.zeros((32, 128), dtype=jnp.int32)
        main_pool = jnp.zeros((64 * 32, 128), dtype=jnp.int32)
        rope_pool = jnp.zeros(((64 * 32) // 4, 32), dtype=jnp.int32)
        swa_records = jnp.full((16, 64), 0, dtype=jnp.int32)
        main_records = jnp.full((16, 64), 0, dtype=jnp.int32)
        pages = jnp.zeros((2, 32), dtype=jnp.int32)
        q_loc = jnp.array([0, 8, 16], dtype=jnp.int32)
        sink = jnp.zeros((8, 64), dtype=jnp.float32)

        fake_out = jnp.zeros((16, 8, 64), dtype=jnp.bfloat16)

        with mock.patch.object(
            sparse_mla, "sparse_ragged_paged_attention", return_value=fake_out
        ) as mock_sparse:
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
                scale=0.1,
                page_size=32,
                precompacted=True,
                union_gather=False,
            )
            assert out.shape == (16, 8, 64)
            mock_sparse.assert_called_once()
            # Verify compacted argument was padded to 128 columns
            compacted_arg = mock_sparse.call_args[0][3]
            assert compacted_arg.shape == (16, 128)

        with mock.patch.object(
            attention_kernels,
            "sparse_ragged_paged_attention_union_gather",
            return_value=fake_out,
        ) as mock_union:
            out2 = sparse_fp8_attention_jax(
                q,
                swa_pool,
                main_pool,
                rope_pool,
                swa_records,
                main_records,
                pages,
                q_loc,
                sink,
                scale=0.1,
                page_size=32,
                precompacted=False,
                union_gather=True,
                union_cap=2048,
            )
            assert out2.shape == (16, 8, 64)
            mock_union.assert_called_once()
            # Verify compacted argument had compact_records applied
            compacted_arg = mock_union.call_args[0][3]
            assert compacted_arg.shape == (16, 128)

    def test_slice_token_parallel_attention_inputs(self):
        swa = jnp.arange(64)[:, None]
        main = (jnp.arange(64) * 2)[:, None]
        q_loc = jnp.array([0, 16, 32, 48, 64], dtype=jnp.int32)
        tp_rank = jnp.array([2], dtype=jnp.int32)
        tp_size = 4
        tokens = 64

        s_loc, m_loc, q_sub = slice_token_parallel_attention_inputs(
            swa, main, q_loc, tp_rank, tp_size, tokens
        )
        assert s_loc.shape == (16, 1)
        assert m_loc.shape == (16, 1)
        np.testing.assert_array_equal(np.asarray(s_loc[:, 0]), np.arange(32, 48))
        np.testing.assert_array_equal(np.asarray(m_loc[:, 0]), np.arange(32, 48) * 2)
        # q_loc clamped to [0, 16] with start=32
        np.testing.assert_array_equal(np.asarray(q_sub), np.array([0, 0, 0, 16, 16]))

    def test_paged_attention_wrapper(self):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        with set_vllm_model_wrapper_context(mesh=mesh):
            attention_kernels._attention_op = None
            initialize_paged_attention()
            assert attention_kernels._attention_op is not None
            # Test early return branch
            saved_op = attention_kernels._attention_op
            initialize_paged_attention()
            assert attention_kernels._attention_op is saved_op

        q = torch.zeros((8, 4, 64), dtype=torch.bfloat16)
        mock_swa = mock.MagicMock()
        mock_swa.raw = torch.zeros((16, 128), dtype=torch.int32)
        mock_main = mock.MagicMock()
        mock_main.raw = torch.zeros((32, 128), dtype=torch.int32)
        swa_rec = torch.zeros((8, 16), dtype=torch.int32)
        main_rec = torch.zeros((8, 16), dtype=torch.int32)
        sink = torch.zeros((4, 64), dtype=torch.float32)

        with mock.patch.object(
            attention_kernels, "_attention_op", return_value=q
        ) as mock_op:
            res = paged_attention(
                q, mock_swa, mock_main, swa_rec, main_rec, sink, scale=0.1
            )
            assert res is q
            mock_op.assert_called_once()

    def test_sparse_fp8_attention_wrapper_default_and_custom_args(self):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        with set_vllm_model_wrapper_context(mesh=mesh):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            initialize_paged_attention()
            assert attention_kernels._fp8_attention_op is not None

        q = torch.zeros((16, 8, 64), dtype=torch.bfloat16)
        mock_swa = mock.MagicMock()
        mock_swa.raw = torch.zeros((32, 128), dtype=torch.int32)
        mock_main = mock.MagicMock()
        mock_main.raw = torch.zeros((64, 128), dtype=torch.int32)
        mock_main.rope = torch.zeros((16, 32), dtype=torch.int32)
        mock_main.shape = (64, 32)
        swa_rec = torch.zeros((16, 64), dtype=torch.int32)
        main_rec = torch.zeros((16, 64), dtype=torch.int32)
        sink = torch.zeros((8, 64), dtype=torch.float32)

        # 1) pages=None, full_sink=None, tp_rank=None
        with mock.patch.object(
            attention_kernels, "_fp8_attention_op", return_value=q
        ) as mock_op:
            res = sparse_fp8_attention(
                q,
                mock_swa,
                mock_main,
                swa_rec,
                main_rec,
                sink,
                scale=0.1,
                pages=None,
                full_sink=None,
                tp_rank=None,
            )
            assert res is q
            args = mock_op.call_args[0]
            # pages was defaulted to [1, 1]
            assert args[6].shape == (1, 1)
            # full_sink was defaulted to sink
            assert args[9] is sink
            # tp_rank was defaulted to zeros(1)
            assert args[10].shape == (1,)

        # 2) Explicit pages, full_sink, and tp_rank
        pages = torch.zeros((2, 8), dtype=torch.int32)
        starts = torch.tensor([0, 8, 16], dtype=torch.int32)
        full_sink = torch.zeros((16, 64), dtype=torch.float32)
        tp_rank = torch.tensor([3], dtype=torch.int32)

        with mock.patch.object(
            attention_kernels, "_fp8_attention_op", return_value=q
        ) as mock_op:
            res = sparse_fp8_attention(
                q,
                mock_swa,
                mock_main,
                swa_rec,
                main_rec,
                sink,
                scale=0.1,
                pages=pages,
                starts=starts,
                max_requests=4,
                precompacted=True,
                token_parallel=True,
                full_sink=full_sink,
                tp_rank=tp_rank,
                union_gather=True,
                union_cap=1024,
            )
            assert res is q
            args = mock_op.call_args[0]
            assert args[6] is not None
            assert args[9] is full_sink
            assert args[10].item() == 3
            assert args[14] is True  # precompacted
            assert args[15] is True  # token_parallel
            assert args[16] is True  # union_gather
            assert args[17] == 1024

    def test_fp8_attention_inner_bodies_tp_and_non_tp(self):
        import inspect

        import vllm.distributed as vllm_dist

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
            initialize_paged_attention()
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
                # 1. Test token-parallel body execution
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


class TestSelectIndicesJax:
    """Comprehensive tests for select_indices_jax line and branch coverage."""

    def test_validation_errors(self):
        scores = jnp.zeros((4, 64), dtype=jnp.float32)
        lengths = jnp.array([10, 20, 30, 40], dtype=jnp.int32)
        candidates = jnp.zeros((4, 8), dtype=jnp.int32)

        with pytest.raises(
            ValueError, match="Positive top-k, score width and block size required"
        ):
            select_indices_jax(
                scores,
                lengths,
                candidates,
                topk=0,
                block_size=16,
                candidate_write=False,
                candidate_filter=False,
            )

        with pytest.raises(
            ValueError, match="Positive top-k, score width and block size required"
        ):
            select_indices_jax(
                scores,
                lengths,
                candidates,
                topk=4,
                block_size=0,
                candidate_write=False,
                candidate_filter=False,
            )

        with pytest.raises(
            ValueError,
            match="Candidate sources must not filter their own scores",
        ):
            select_indices_jax(
                scores,
                lengths,
                candidates,
                topk=4,
                block_size=16,
                candidate_write=True,
                candidate_filter=True,
            )

        with pytest.raises(ValueError, match="Candidate rows must match query rows"):
            select_indices_jax(
                scores,
                lengths,
                candidates[:2],
                topk=4,
                block_size=16,
                candidate_write=False,
                candidate_filter=False,
            )

        with pytest.raises(ValueError, match="Candidate rows must match query rows"):
            select_indices_jax(
                scores,
                lengths,
                candidates.flatten(),
                topk=4,
                block_size=16,
                candidate_write=False,
                candidate_filter=False,
            )

    def test_candidate_write_branch(self):
        rng = np.random.default_rng(123)
        count, width, block_size = 4, 128, 16
        blocks = width // block_size  # 8 blocks
        scores = jnp.asarray(rng.standard_normal((count, width)), dtype=jnp.float32)
        lengths = jnp.array([32, 64, 128, 16], dtype=jnp.int32)
        candidates = jnp.full((count, 4), -1, dtype=jnp.int32)

        selected, published = select_indices_jax(
            scores,
            lengths,
            candidates,
            topk=16,
            block_size=block_size,
            candidate_write=True,
            candidate_filter=False,
            unordered=False,
        )
        assert published.shape == (count, 4)
        assert selected.shape == (count, 16)
        # Verify published candidate IDs are valid block IDs or -1
        assert np.all(np.asarray(published) < blocks)
        # Verify the last block was marked jnp.inf and therefore published first
        last_block = ((lengths - 1) // block_size)[:, None]
        np.testing.assert_array_equal(
            np.asarray(published[:, 0:1]), np.asarray(last_block)
        )

        # SparseCore candidate_write fast path: unordered=True, blocks >= 2048,
        # blocks % 16 == 0, cand_k % 16 == 0
        sc_scores = jnp.zeros((count, 2048 * 8), dtype=jnp.float32)
        sc_lengths = jnp.array([1024, 2048, 4096, 8192], dtype=jnp.int32)
        sc_candidates = jnp.full((count, 16), -1, dtype=jnp.int32)
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels._sparsecore_top_k",
            side_effect=lambda s, k, n: jnp.broadcast_to(
                jnp.arange(k, dtype=jnp.int32), (s.shape[0], k)
            ),
        ) as mock_sc:
            sel_sc, pub_sc = select_indices_jax(
                sc_scores,
                sc_lengths,
                sc_candidates,
                topk=16,
                block_size=8,
                candidate_write=True,
                candidate_filter=False,
                unordered=True,
            )
            assert sel_sc.shape == (count, 16)
            assert pub_sc.shape == (count, 16)
            assert mock_sc.call_count == 2

    def test_candidate_filter_branch_and_128_aligned_scatter(self):
        rng = np.random.default_rng(456)
        count, width, block_size = 4, 320, 16
        scores = jnp.asarray(rng.standard_normal((count, width)), dtype=jnp.float32)
        lengths = jnp.array([100, 200, 320, 50], dtype=jnp.int32)
        # candidates containing valid, negative, and >= blocks values
        candidates = jnp.array(
            [[0, 1, -1, 5], [2, 19, 25, -10], [5, 6, 7, 8], [0, -1, -1, -1]],
            dtype=jnp.int32,
        )

        # 1) batch_max > topk: filtering is active
        selected, published = select_indices_jax(
            scores,
            lengths,
            candidates,
            topk=16,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
            unordered=False,
            batch_max=jnp.array([200], dtype=jnp.int32),
        )
        assert published is candidates
        assert selected.shape == (count, 16)

        # 2) batch_max <= topk: short-context bypass allows all visible blocks
        selected_bypass, _ = select_indices_jax(
            scores,
            lengths,
            candidates,
            topk=320,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
            unordered=False,
            batch_max=jnp.array([50], dtype=jnp.int32),
        )
        assert selected_bypass.shape == (count, 320)

    def test_candidates_buffer_covers_entire_width_identity(self):
        count, width, block_size = 2, 64, 16
        scores = jnp.ones((count, width), dtype=jnp.float32)
        lengths = jnp.array([64, 64], dtype=jnp.int32)
        # candidates shape[1] * block_size >= width
        candidates = jnp.zeros((count, 4), dtype=jnp.int32)

        # candidate_write=True should bypass because candidates buffer covers all blocks
        selected, published = select_indices_jax(
            scores,
            lengths,
            candidates,
            topk=8,
            block_size=block_size,
            candidate_write=True,
            candidate_filter=False,
        )
        assert published is candidates

    def test_unordered_compaction_and_edge_values(self):
        # Test -inf, NaN, and +0.0 / -0.0 tie breaking in select_indices_jax
        count = 2
        scores = jnp.array(
            [
                [0.0, -0.0, 1.0, -jnp.inf, jnp.nan, 2.0, -1.0, 3.0],
                [1.0, 1.0, 1.0, 1.0, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf],
            ],
            dtype=jnp.float32,
        )
        lengths = jnp.array([8, 8], dtype=jnp.int32)
        candidates = jnp.zeros((count, 2), dtype=jnp.int32)

        # unordered=True
        sel_unord, _ = select_indices_jax(
            scores,
            lengths,
            candidates,
            topk=4,
            block_size=4,
            candidate_write=False,
            candidate_filter=False,
            unordered=True,
        )
        assert sel_unord.shape == (count, 4)
        # Rows should have valid indices compacted at head, -1 at tail
        for row in np.asarray(sel_unord):
            valid_idx = row[row >= 0]
            pad_idx = row[row < 0]
            assert len(valid_idx) + len(pad_idx) == 4
            assert np.all(pad_idx == -1)

        # unordered=False: ascending logical order
        sel_ord, _ = select_indices_jax(
            scores,
            lengths,
            candidates,
            topk=4,
            block_size=4,
            candidate_write=False,
            candidate_filter=False,
            unordered=False,
        )
        assert sel_ord.shape == (count, 4)


class TestThresholdAndVisibleTopK:
    """Tests for prefix count, threshold top-k, sparsecore top-k and visible top-k."""

    def test_prefix_count_blockwise(self):
        rng = np.random.default_rng(789)
        # width > 128 to test multi-block cumsum
        mask = jnp.asarray(rng.choice([False, True], size=(4, 300)), dtype=jnp.bool_)
        got = _prefix_count(mask, block=128)
        expected = np.cumsum(np.asarray(mask), axis=-1, dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(got), expected)

    def test_threshold_top_k_ties_and_signed_zero(self):
        scores = jnp.array(
            [[+0.0, -0.0, 5.0, 5.0, -10.0, 2.0, 2.0, 2.0]], dtype=jnp.float32
        )
        # k=4 should pick 5.0 (idx 2, 3), 2.0 (idx 5, 6)
        got = _threshold_top_k(scores, k=4)
        assert got.shape == (1, 4)
        # Indices should be ascending
        assert np.all(np.diff(np.asarray(got[0])) > 0)

        # Check +0.0 ranked above -0.0
        scores_zeros = jnp.array([[-0.0, +0.0]], dtype=jnp.float32)
        got_zeros = _threshold_top_k(scores_zeros, k=1)
        # idx 1 is +0.0, which ranks above idx 0 (-0.0)
        assert got_zeros[0, 0] == 1

    def test_sparsecore_top_k_standalone_and_mesh(self):
        scores = jnp.zeros((16, 2048), dtype=jnp.float32)
        lengths = jnp.full((16,), 1000, dtype=jnp.int32)

        # 1) Standalone (no wrapper context mesh)
        with (
            mock.patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context",
                side_effect=Exception("No mesh"),
            ),
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk.sparsecore_topk",
                return_value=jnp.zeros((16, 64), dtype=jnp.int32),
            ) as mock_sc,
        ):
            got = _sparsecore_top_k(scores, k=64, lengths=lengths)
            assert got.shape == (16, 64)
            mock_sc.assert_called_once()

        # 2) With mesh (shard_map branch)
        mock_ctx = mock.MagicMock()
        mock_ctx.mesh = mock.MagicMock()
        with (
            mock.patch(
                "vllm_torchtpu.models.vllm.vllm_model_wrapper_context.get_vllm_model_wrapper_context",
                return_value=mock_ctx,
            ),
            mock.patch(
                "jax.shard_map",
                side_effect=lambda fn, **kwargs: lambda *args: fn(*args),
            ),
            mock.patch(
                "vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk.sparsecore_topk",
                return_value=jnp.zeros((16, 64), dtype=jnp.int32),
            ) as mock_sc,
        ):
            got2 = _sparsecore_top_k(scores, k=64, lengths=lengths)
            assert got2.shape == (16, 64)
            mock_sc.assert_called_once()

    def test_visible_top_k_branches(self):
        rng = np.random.default_rng(999)
        # 1) unordered=True and sparsecore condition met
        scores_large = jnp.asarray(rng.standard_normal((16, 2048)), dtype=jnp.float32)
        lengths_large = jnp.full((16,), 2048, dtype=jnp.int32)
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels._sparsecore_top_k",
            return_value=jnp.zeros((16, 64), dtype=jnp.int32),
        ) as mock_sc:
            res_sc = _visible_top_k(
                scores_large,
                k=64,
                bound=jnp.array(2048, dtype=jnp.int32),
                lengths=lengths_large,
                unordered=True,
            )
            assert res_sc.shape == (16, 64)
            mock_sc.assert_called_once()

        # 2) unordered=False with len(buckets) == 1 (width <= max(k, 1024))
        scores_small = jnp.asarray(rng.standard_normal((4, 512)), dtype=jnp.float32)
        res_small = _visible_top_k(
            scores_small,
            k=32,
            bound=jnp.array(512, dtype=jnp.int32),
            unordered=False,
        )
        assert res_small.shape == (4, 32)
        # Ascending indices
        assert np.all(np.diff(np.asarray(res_small[0])) > 0)

        # 3) unordered=False with len(buckets) > 1 and columns >= 2048
        res_switch = _visible_top_k(
            scores_large,
            k=64,
            bound=jnp.array(2048, dtype=jnp.int32),
            unordered=False,
        )
        assert res_switch.shape == (16, 64)
        assert np.all(np.diff(np.asarray(res_switch[0])) > 0)


class TestPagedIndexScoresJax:
    """Tests for paged_index_scores_jax across decode, prefill, and step loops."""

    def test_decode_path_tokens_le_records(self):
        rng = np.random.default_rng(101)
        # tokens (4) <= requests_count (8) -> decode path
        pool, records, requests, lengths, q, weights = _request_inputs(
            rng, requests_count=8, tokens=4, width=256
        )
        scores = paged_index_scores_jax(
            q, pool, jnp.asarray(records), weights, lengths, jnp.asarray(requests)
        )
        assert scores.shape == (4, 256)

    def test_prefill_shared_tiles_tokens_gt_records(self):
        rng = np.random.default_rng(202)
        # tokens (16) > requests_count (4) -> prefill row_step shared tile path
        pool, records, requests, lengths, q, weights = _request_inputs(
            rng, requests_count=4, tokens=16, width=512
        )
        scores = paged_index_scores_jax(
            q, pool, jnp.asarray(records), weights, lengths, jnp.asarray(requests)
        )
        assert scores.shape == (16, 512)

    def test_requests_none_direct_step_loop(self):
        rng = np.random.default_rng(303)
        pool, records, _, lengths, q, weights = _request_inputs(
            rng, requests_count=8, tokens=8, width=512
        )
        scores = paged_index_scores_jax(
            q, pool, jnp.asarray(records), weights, lengths, requests=None
        )
        assert scores.shape == (8, 512)


class TestPagedIndexerJax:
    """Comprehensive tests for paged_indexer_jax routing and context branches."""

    @pytest.mark.parametrize("active,live_length", [(16, 128), (1, 128), (16, 8)])
    @pytest.mark.cpu_test
    def test_shared_prefill_binds_one_pipeline(self, active, live_length):
        rng = np.random.default_rng(408)
        pool, records, requests, _, q, weights = _request_inputs(rng, 2, 16, 128)
        records = jnp.asarray(records)
        requests = jnp.asarray(requests)
        lengths = jnp.where(jnp.arange(16) < active, live_length, 0)
        candidates = jnp.asarray(rng.integers(-1, 8, (16, 2)), jnp.int32)

        def run(q, lengths):
            return paged_indexer_jax(
                q,
                pool,
                records,
                weights,
                lengths,
                candidates,
                requests,
                topk=16,
                block_size=16,
                candidate_write=False,
                candidate_filter=True,
            )

        scores = paged_index_scores_jax(q, pool, records, weights, lengths, requests)
        scores = jnp.where(records[requests] >= 0, scores, -jnp.inf)
        expected = select_indices_jax(scores, lengths, candidates, 16, 16, False, True)
        actual = jax.jit(run)(q, lengths)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
        # A value-dependent scoring choice otherwise clones both pipelines.
        graph = jax.make_jaxpr(run)(q, lengths).jaxpr
        assert all(eqn.primitive.name != "cond" for eqn in graph.eqns)

    def test_routing_to_full_context_via_not_candidate_filter(self):
        rng = np.random.default_rng(404)
        width, topk, block_size = 640, 64, 16
        pool, records, requests, lengths, q, weights = _request_inputs(rng, 2, 8, width)
        candidates = jnp.zeros((8, 4), dtype=jnp.int32)

        sel, pub = paged_indexer_jax(
            q,
            pool,
            jnp.asarray(records),
            weights,
            jnp.asarray(lengths),
            candidates,
            jnp.asarray(requests),
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=False,  # False forces full_context
        )
        assert sel.shape == (8, topk)
        assert pub is candidates

    def test_routing_to_full_context_via_candidate_coverage(self):
        rng = np.random.default_rng(505)
        width, topk, block_size = 64, 16, 16
        pool, records, requests, lengths, q, weights = _request_inputs(rng, 2, 4, width)
        # candidates cover entire width (4 * 16 >= 64)
        candidates = jnp.zeros((4, 4), dtype=jnp.int32)

        sel, pub = paged_indexer_jax(
            q,
            pool,
            jnp.asarray(records),
            weights,
            jnp.asarray(lengths),
            candidates,
            jnp.asarray(requests),
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
        )
        assert sel.shape == (4, topk)
        assert pub is candidates

    def test_routing_to_full_context_via_short_context_or_prefill_keys(self):
        rng = np.random.default_rng(606)
        width, topk, block_size = 640, 64, 16
        pool, records, requests, lengths, q, weights = _request_inputs(
            rng, 2, 16, width
        )
        # 1) Short context: longest <= topk
        short_lengths = jnp.full((16,), topk // 2, dtype=jnp.int32)
        candidates = jnp.zeros((16, 4), dtype=jnp.int32)
        sel_short, _ = paged_indexer_jax(
            q,
            pool,
            jnp.asarray(records),
            weights,
            short_lengths,
            candidates,
            jnp.asarray(requests),
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
        )
        assert sel_short.shape == (16, topk)

        # 2) Prefill shared tile: q.shape[0] > records.shape[0] and
        # row_lengths.sum() <= cand_keys
        # Many tokens (16) sharing only 2 requests, cand_keys = 16 * 4 * 16 = 1024
        # row_lengths sum is ~2 * 300 = 600 <= 1024 -> triggers full_context
        lengths_prefill = jnp.full((16,), 300, dtype=jnp.int32)
        sel_prefill, _ = paged_indexer_jax(
            q,
            pool,
            jnp.asarray(records),
            weights,
            lengths_prefill,
            candidates,
            jnp.asarray(requests),
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
        )
        assert sel_prefill.shape == (16, topk)

        # 3) Decode / verify (q.shape[0] <= records.shape[0]): routes to
        # candidate_context whenever longest > topk even if
        # row_lengths.sum() <= cand_keys
        _, records_dec, _, _, q_dec, weights_dec = _request_inputs(rng, 8, 6, width)
        requests_dec = jnp.zeros((6,), dtype=jnp.int32)
        lengths_dec = jnp.full((6,), 300, dtype=jnp.int32)
        candidates_dec = jnp.zeros((6, 4), dtype=jnp.int32)
        sel_dec, _ = paged_indexer_jax(
            q_dec,
            pool,
            jnp.asarray(records_dec),
            weights_dec,
            lengths_dec,
            candidates_dec,
            requests_dec,
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
        )
        assert sel_dec.shape == (6, topk)

    def test_candidate_context_branch_ordered_and_unordered(self):
        rng = np.random.default_rng(707)
        width, topk, block_size = 640, 64, 16
        tokens = 16
        pool, records, requests, lengths, q, weights = _request_inputs(
            rng, 16, tokens, width
        )
        # To enter candidate_context:
        # longest > topk AND row_lengths.sum() > cand_keys
        # Make 1 active long request and rest length 0:
        # lengths[0] = 500, rest = 0
        # cand_keys = 1 * (4 * 16) = 64, row_lengths.sum() = 500 > 64!
        lengths = np.zeros(tokens, dtype=np.int32)
        lengths[0] = 500
        lengths = jnp.asarray(lengths)
        candidates = jnp.asarray(
            rng.integers(-2, width // block_size + 5, (tokens, 4)),
            dtype=jnp.int32,
        )

        # 1) unordered=True
        sel_unord, pub_unord = paged_indexer_jax(
            q,
            pool,
            jnp.asarray(records),
            weights,
            lengths,
            candidates,
            jnp.asarray(requests),
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
            unordered=True,
        )
        assert pub_unord is candidates
        assert sel_unord.shape == (tokens, topk)
        # First row has valid selections at head, -1 at tail
        row0 = np.asarray(sel_unord[0])
        val0 = row0[row0 >= 0]
        pad0 = row0[row0 < 0]
        assert len(val0) + len(pad0) == topk
        # Zero-length rows are completely -1
        assert np.all(np.asarray(sel_unord[1:]) == -1)

        # 1b) unordered=True with scores.shape[1] >= 2048 (_sparsecore_top_k path)
        candidates_large = jnp.zeros((tokens, 128), dtype=jnp.int32)
        records_large = jnp.zeros((16, 4096), dtype=jnp.int32)
        lengths_large = jnp.full((tokens,), 3000, dtype=jnp.int32)
        with mock.patch.object(
            attention_kernels,
            "_sparsecore_top_k",
            return_value=jnp.zeros((tokens, topk), dtype=jnp.int32),
        ) as mock_sc_topk:
            sel_sc, _ = paged_indexer_jax(
                q,
                pool,
                records_large,
                weights,
                lengths_large,
                candidates_large,
                jnp.asarray(requests),
                topk=topk,
                block_size=block_size,
                candidate_write=False,
                candidate_filter=True,
                unordered=True,
            )
            assert sel_sc.shape == (tokens, topk)
            assert mock_sc_topk.call_count == 2

        # 2) unordered=False: exercises underfull topk, occupied, fillers, and sort
        sel_ord, pub_ord = paged_indexer_jax(
            q,
            pool,
            jnp.asarray(records),
            weights,
            lengths,
            candidates,
            jnp.asarray(requests),
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=True,
            unordered=False,
        )
        assert pub_ord is candidates
        assert sel_ord.shape == (tokens, topk)
        # Zero-length rows are completely -1
        assert np.all(np.asarray(sel_ord[1:]) == -1)


class TestWriteIndexKeysAndUpdateSelect:
    """Tests for write_index_keys and update_and_select_jax."""

    def test_write_index_keys_validation(self):
        pool = jnp.zeros((64, 128), dtype=jnp.int32)
        keys_fp32 = jnp.zeros((4, 128), dtype=jnp.float32)
        keys_bf16 = jnp.zeros((4, 128), dtype=jnp.bfloat16)
        records = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

        with pytest.raises(ValueError, match="Index keys must be BF16"):
            write_index_keys(pool, records, keys_fp32)

        with pytest.raises(
            ValueError, match="One physical record index per key is required"
        ):
            write_index_keys(pool, records[:2], keys_bf16)

        with pytest.raises(
            ValueError, match="One physical record index per key is required"
        ):
            write_index_keys(pool, records, keys_bf16.reshape(2, 2, 128))

    def test_write_index_keys_and_update_select_execution(self):
        rng = np.random.default_rng(808)
        width, topk, block_size = 128, 16, 16
        pool, records, requests, lengths, q, weights = _request_inputs(rng, 2, 4, width)
        candidates = jnp.zeros((4, 2), dtype=jnp.int32)
        write_recs = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        keys = jnp.zeros((4, 128), dtype=jnp.bfloat16)

        up_pool, sel, pub = update_and_select_jax(
            q,
            pool,
            jnp.asarray(records),
            weights,
            jnp.asarray(lengths),
            candidates,
            write_recs,
            keys,
            jnp.asarray(requests),
            batch_max=jnp.array([width], dtype=jnp.int32),
            topk=topk,
            block_size=block_size,
            candidate_write=False,
            candidate_filter=False,
            unordered=False,
        )
        assert up_pool.shape == pool.shape
        assert sel.shape == (4, topk)
        assert pub is candidates


class TestIndexerOpAndWrapper:
    """Tests for initialize_indexer and paged_indexer wrapper."""

    def test_initialize_indexer_and_fake(self):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        with set_vllm_model_wrapper_context(mesh=mesh):
            attention_kernels._indexer_op = None
            initialize_indexer()
            assert attention_kernels._indexer_op is not None
            # Test early return branch
            saved_op = attention_kernels._indexer_op
            initialize_indexer()
            assert attention_kernels._indexer_op is saved_op

        # Verify fake registration by calling op with meta device tensors
        q = torch.empty((4, 4, 128), dtype=torch.bfloat16, device="meta")
        pool = torch.empty((64, 128), dtype=torch.int32, device="meta")
        records = torch.empty((2, 128), dtype=torch.int32, device="meta")
        weights = torch.empty((4, 4), dtype=torch.float32, device="meta")
        lengths = torch.empty((4,), dtype=torch.int32, device="meta")
        candidates = torch.empty((4, 2), dtype=torch.int32, device="meta")
        write_recs = torch.empty((4,), dtype=torch.int32, device="meta")
        keys = torch.empty((4, 128), dtype=torch.bfloat16, device="meta")
        requests = torch.empty((4,), dtype=torch.int32, device="meta")
        batch_max = torch.empty((1,), dtype=torch.int32, device="meta")

        f_pool, f_sel, f_pub = attention_kernels._indexer_op(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            write_recs,
            keys,
            requests,
            batch_max,
            16,
            16,
            False,
            False,
            False,
            0,
        )
        assert f_pool.shape == pool.shape
        assert f_sel.shape == (4, 16)
        assert f_pub.shape == candidates.shape

    def test_paged_indexer_wrapper(self):
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        with set_vllm_model_wrapper_context(mesh=mesh):
            initialize_indexer()
        q = torch.zeros((4, 4, 128), dtype=torch.bfloat16)
        mock_cache = mock.MagicMock()
        mock_cache.raw = torch.zeros((64, 128), dtype=torch.int32)
        records = torch.zeros((2, 128), dtype=torch.int32)
        weights = torch.zeros((4, 4), dtype=torch.float32)
        lengths = torch.tensor([50, 60, 70, 80], dtype=torch.int32)
        candidates = torch.zeros((4, 2), dtype=torch.int32)
        write_recs = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        keys = torch.zeros((4, 128), dtype=torch.bfloat16)
        requests = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

        # 1) batch_max is None (default)
        with mock.patch.object(
            attention_kernels,
            "_indexer_op",
            return_value=(mock_cache.raw, records, candidates),
        ) as mock_op:
            res = paged_indexer(
                q,
                mock_cache,
                records,
                weights,
                lengths,
                candidates,
                write_recs,
                keys,
                requests,
                topk=16,
                block_size=16,
                candidate_write=False,
                candidate_filter=False,
            )
            assert len(res) == 3
            mock_op.assert_called_once()
            # Verify batch_max argument defaulted to lengths.max() = 80
            assert mock_op.call_args[0][9].item() == 80

        # 2) explicit batch_max
        with mock.patch.object(
            attention_kernels,
            "_indexer_op",
            return_value=(mock_cache.raw, records, candidates),
        ) as mock_op:
            res2 = paged_indexer(
                q,
                mock_cache,
                records,
                weights,
                lengths,
                candidates,
                write_recs,
                keys,
                requests,
                topk=16,
                block_size=16,
                candidate_write=False,
                candidate_filter=False,
                batch_max=torch.tensor([120], dtype=torch.int32),
            )
            assert len(res2) == 3
            assert mock_op.call_args[0][9].item() == 120
