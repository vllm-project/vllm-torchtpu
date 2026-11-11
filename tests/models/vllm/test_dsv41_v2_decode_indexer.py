"""Decode indexer coverage: top-k branches and TP query splitting.

1. indexer_select._visible_top_k SparseCore and fallback branches.
2. DeepseekV41Attention._select_paged_indices over the packed FP8 cache:
   split and unsplit queries, padding rows, candidate publication and
   stage-cache reuse of the query split.
"""

from functools import partial
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
import torch

import vllm_torchtpu.models.vllm.deepseek_v41.v2.attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    indexer_fp8,
    indexer_select,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_cache import (
    PackedFP8IndexCacheRows,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select import _visible_top_k


class TestAttentionKernelsDecodeIndexer:
    """Coverage for v2/attention_kernels.py / changes."""

    def test_visible_top_k_sparsecore_and_fallback_branches(self, monkeypatch):
        """Covers _visible_top_k unpadded rows < 16, rows >= 16, and fallback."""
        sparsecore_calls = []

        def mock_sc_top_k(scores, k, lengths):
            sparsecore_calls.append((scores.shape, k))
            return jnp.zeros((scores.shape[0], k), dtype=jnp.int32)

        monkeypatch.setattr(indexer_select, "_sparsecore_top_k", mock_sc_top_k)

        bound = jnp.int32(2048)

        # 1. SparseCore branch with rows < 16 (e.g. rows = 3, unpadded)
        scores_3 = jnp.zeros((3, 2048), dtype=jnp.float32)
        lengths_3 = jnp.array([100, 200, 300], dtype=jnp.int32)
        out_3 = _visible_top_k(scores_3, 16, bound, lengths_3, unordered=True)
        assert out_3.shape == (3, 16)
        assert sparsecore_calls[-1] == ((3, 2048), 16)

        # 2. SparseCore branch with rows >= 16 (e.g. rows = 16)
        scores_16 = jnp.zeros((16, 2048), dtype=jnp.float32)
        lengths_16 = jnp.full((16,), 100, dtype=jnp.int32)
        out_16 = _visible_top_k(scores_16, 32, bound, lengths_16, unordered=True)
        assert out_16.shape == (16, 32)
        assert sparsecore_calls[-1] == ((16, 2048), 32)

        # 3. Fallback branch: unordered=False (calls select via switch)
        scores_fb = jnp.zeros((2, 2048), dtype=jnp.float32)
        lengths_fb = jnp.array([10, 20], dtype=jnp.int32)
        out_fb = _visible_top_k(scores_fb, 16, bound, lengths_fb, unordered=False)
        assert out_fb.shape == (2, 16)

        # 4. Fallback branch: width < 2048
        scores_narrow = jnp.zeros((2, 1024), dtype=jnp.float32)
        out_narrow = _visible_top_k(
            scores_narrow, 16, jnp.int32(1024), lengths_fb, unordered=True
        )
        assert out_narrow.shape == (2, 16)

        # 5. Fallback branch: width % 16 != 0
        scores_unaligned_w = jnp.zeros((2, 2050), dtype=jnp.float32)
        out_unaligned_w = _visible_top_k(
            scores_unaligned_w, 16, jnp.int32(2050), lengths_fb, unordered=True
        )
        assert out_unaligned_w.shape == (2, 16)

        # 6. Fallback branch: k % 16 != 0
        out_unaligned_k = _visible_top_k(
            scores_fb, 10, bound, lengths_fb, unordered=True
        )
        assert out_unaligned_k.shape == (2, 10)

        # 7. Fallback branch: lengths is None
        out_no_len = _visible_top_k(scores_fb, 16, bound, None, unordered=True)
        assert out_no_len.shape == (2, 16)

        # 8. Fallback branch: scores.shape[0] >= 16 and columns >= 2048
        # (triggers _threshold_top_k)
        scores_thresh = jnp.zeros((16, 2048), dtype=jnp.float32)
        out_thresh = _visible_top_k(
            scores_thresh, 16, bound, lengths_16, unordered=False
        )
        assert out_thresh.shape == (16, 16)


class TestAttentionSelectPagedIndicesDecode:
    """Coverage for v2/attention.py _select_paged_indices / changes."""

    def _setup_mock_attn(self, monkeypatch, ranks=8, tp_rank=0, candidate_write=False):
        """Constructs an attention module mock with target attributes."""
        attn: v2_attn.DeepseekV41Attention = object.__new__(
            v2_attn.DeepseekV41Attention
        )
        torch.nn.Module.__init__(attn)
        attn.compress_ratio = 2
        attn._tp_rank = tp_rank
        attn._fp8_kv = True
        attn._config = {"index_topk": 4}
        attn._max_model_len = 196608
        attn.candidate_source_layer = 20
        attn.candidate_block_size = 16
        attn.candidate_block_buffer = torch.full((128, 8), -1, dtype=torch.int32)
        attn.indexer = SimpleNamespace(
            head_dim=128,
            softmax_scale=128**-0.5,
            n_head=2,
            indexer_op=v2_attn.SparseMQAIndexer(
                SimpleNamespace(prefix="source-a"),
                4,
                torch.zeros(1),
                16,
                candidate_source_layer=20,
                candidate_write=candidate_write,
            ),
        )
        attn._project_index_q_unquantized = lambda q, p: torch.zeros(q.shape[0], 2, 128)

        monkeypatch.setattr(
            v2_attn, "get_tensor_model_parallel_world_size", lambda: ranks
        )
        monkeypatch.setattr(
            v2_attn,
            "_gather_query_rows",
            lambda x: torch.cat([x] * ranks, dim=0),
        )
        return attn

    def _make_selection_call(self, attn, count, *, cache_offset=0):
        cache = PackedFP8IndexCacheRows(
            torch.zeros((64, 128), dtype=torch.int32),
            ((4, 16, 132), torch.uint8, 32 * 256, cache_offset),
        )
        return partial(
            attn._select_paged_indices,
            qr=torch.zeros(count, 128),
            positions=torch.arange(count, dtype=torch.int32),
            index_cache=cache,
            records=torch.zeros(count, 256, dtype=torch.int32),
            requests=torch.arange(count, dtype=torch.int32),
            weights=torch.zeros(count, 2),
            lengths=torch.full((count, 1), 50, dtype=torch.int32),
            swa_valid=torch.ones(count, dtype=torch.bool),
        )

    @pytest.mark.parametrize("count", [1, 16, 17, 32])
    def test_select_paged_indices_split_part_and_padding_calculation(
        self, count, monkeypatch
    ):
        """Split queries keep their owners; padding rows have no request owner."""
        ranks = 8
        attn = self._setup_mock_attn(monkeypatch, ranks=ranks)

        captured_args = {}

        def mock_paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            requests_part,
            *args,
            **kwargs,
        ):
            captured_args["q_part_shape"] = q_part.shape
            captured_args["lengths_part_shape"] = lengths_part.shape
            expected_part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
            live = min(count, expected_part)
            torch.testing.assert_close(
                requests_part[:live], torch.arange(live, dtype=torch.int32)
            )
            assert torch.all(requests_part[live:] == -1)
            assert torch.all(lengths_part[live:] == 0)
            return (
                torch.zeros(expected_part, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(indexer_fp8, "paged_fp8_indexer", mock_paged_indexer)

        select = self._make_selection_call(attn, count, cache_offset=16 * 256)

        with v2_attn.stage_attention_cache_scope():
            selected = select()

        assert selected.shape == (count, 4)
        expected_part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
        assert captured_args["q_part_shape"][0] == expected_part
        assert captured_args["lengths_part_shape"][0] == expected_part

    def test_select_paged_indices_candidate_write_and_published_buffer(
        self, monkeypatch
    ):
        """candidate_write=True gathers published rows and updates buffer."""
        ranks = 2
        count = 6
        attn = self._setup_mock_attn(monkeypatch, ranks=ranks, candidate_write=True)

        def mock_paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            requests_part,
            *args,
            **kwargs,
        ):
            part = q_part.shape[0]
            # published part
            pub_part = torch.full((part, 8), 42, dtype=torch.int32)
            return (
                torch.zeros(part, 4, dtype=torch.int32),
                pub_part,
            )

        monkeypatch.setattr(indexer_fp8, "paged_fp8_indexer", mock_paged_indexer)

        select = self._make_selection_call(attn, count)

        selected = select()

        assert selected.shape == (count, 4)
        # candidate_block_buffer [:count] must have received 42
        assert (attn.candidate_block_buffer[:count] == 42).all()
        # and tail must be padded with -1
        assert (attn.candidate_block_buffer[count:] == -1).all()

    def test_select_paged_indices_split_false_and_no_candidates(self, monkeypatch):
        """split=False (ranks=1) bypasses query splitting and row gathers."""
        ranks = 1
        count = 5
        attn = self._setup_mock_attn(monkeypatch, ranks=ranks)
        attn.indexer.indexer_op.candidate_blocks = None  # has_candidates = False

        gather_called = []
        monkeypatch.setattr(
            v2_attn,
            "_gather_query_rows",
            lambda x: gather_called.append(True) or x,
        )

        def mock_paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            requests_part,
            *args,
            **kwargs,
        ):
            assert q_part.shape[0] == count
            assert candidates_part.shape == (count, 0)
            return (
                torch.zeros(count, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(indexer_fp8, "paged_fp8_indexer", mock_paged_indexer)

        select = self._make_selection_call(attn, count)

        selected = select()

        assert selected.shape == (count, 4)
        assert len(gather_called) == 0  # No gather called when split=False

    def test_stage_cache_hit_and_miss_branches(self, monkeypatch):
        """stage_cache miss computes split; subsequent call hits cached_query_split."""
        ranks = 4
        count = 12
        attn = self._setup_mock_attn(monkeypatch, ranks=ranks)

        call_count = 0

        def mock_paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            requests_part,
            *args,
            **kwargs,
        ):
            nonlocal call_count
            call_count += 1
            part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
            assert q_part.shape[0] == part
            return (
                torch.zeros(part, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(indexer_fp8, "paged_fp8_indexer", mock_paged_indexer)

        select = self._make_selection_call(attn, count)

        with v2_attn.stage_attention_cache_scope():
            # Call 1: cache miss
            sel1 = select()
            assert sel1.shape == (count, 4)
            assert call_count == 1

            # Call 2: cache hit
            sel2 = select()
            assert sel2.shape == (count, 4)
            assert call_count == 2
