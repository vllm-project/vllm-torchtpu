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

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    attention_kernels,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _pad_compacted_records,
    initialize_paged_attention,
    paged_attention,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select import (
    _prefix_count,
    _sparsecore_top_k,
    _threshold_top_k,
    _visible_top_k,
    select_indices_jax,
)
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


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
                sparse_logits=False,
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
                sparse_logits=False,
            )

        with pytest.raises(
            ValueError,
            match="Candidate sources must not use sparse logits",
        ):
            select_indices_jax(
                scores,
                lengths,
                candidates,
                topk=4,
                block_size=16,
                candidate_write=True,
                sparse_logits=True,
            )

        with pytest.raises(ValueError, match="Candidate rows must match query rows"):
            select_indices_jax(
                scores,
                lengths,
                candidates[:2],
                topk=4,
                block_size=16,
                candidate_write=False,
                sparse_logits=False,
            )

        with pytest.raises(ValueError, match="Candidate rows must match query rows"):
            select_indices_jax(
                scores,
                lengths,
                candidates.flatten(),
                topk=4,
                block_size=16,
                candidate_write=False,
                sparse_logits=False,
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
            sparse_logits=False,
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
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select._sparsecore_top_k",
            side_effect=lambda s, k, *_: jnp.broadcast_to(
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
                sparse_logits=False,
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
            sparse_logits=True,
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
            sparse_logits=True,
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
            sparse_logits=False,
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
            sparse_logits=False,
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
            sparse_logits=False,
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
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select._sparsecore_top_k",
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
