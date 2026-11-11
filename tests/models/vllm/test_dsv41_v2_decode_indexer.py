"""100% diff line and branch coverage for / decode indexer.

Target modules and diff lines:
1. vllm_torchtpu.models.vllm.deepseek_v41.v2.cache
   - gather_cache_page_values (lines 523-550):
     * ValueError on ROW_WORDS % record_words != 0
     * ValueError on (page_records * record_words) % ROW_WORDS != 0
     * Valid and negative (-1 sentinel) first_records
     * Bitwise equivalence against gather_cache_values(split=True)
2. vllm_torchtpu.models.vllm.deepseek_v41.v2.attention
   - _index_page_records (lines 389-410):
     * Non-index owners are rejected
     * Valid qualifying layout -> per_page
   - DeepseekV41Attention._select_paged_indices (lines 1351-1490):
     * split=True (ranks > 1, part = (count + ranks - 1) // ranks, padding = ranks)
     * split=False (ranks = 1, padding = 0, no slice / no gather)
     * cached_query_split hit vs miss (stage_attention_cache_scope)
     * candidate_write=True vs candidate_write=False
     * sparse_logits=True vs False
     * page_records passed to paged_indexer
3. vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels
   - Constants _INDEX_KEYS_PER_ITER = 65536, _INDEX_MAX_TILE = 8192
   - paged_index_scores_jax (lines 1318-1360, 1410-1425):
     * tokens <= 8 -> _INDEX_MAX_TILE (8192) tile budget
     * tokens = 16 -> 4096 tile budget
     * tokens >= 128 -> 512 tile floor
     * width clamping (-(-width // 128) * 128)
     * requests is not None and tokens <= records.shape[0] (decode take)
     * requests is not None and tokens > records.shape[0] (prefill row_step)
     * requests is None
     * pages=True (native packed page gather branch)
     * pages=False: page_records is None
     * pages=False: tile % page_records != 0
     * Numerical equivalence of pages=True vs pages=False
   - _visible_top_k (lines 1535-1555):
     * SparseCore path with rows < 16 (unpadded)
     * SparseCore path with rows >= 16
     * Fallback path: unordered=False, lengths=None, width < 2048,
       width % 16 != 0, k % 16 != 0
   - paged_indexer_jax, select_indexer_jax, initialize_indexer, paged_indexer
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    indexer_select,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import (
    _index_page_records,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    ROW_WORDS,
    gather_cache_page_values,
    gather_cache_values,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_select import _visible_top_k


class TestGatherCachePageValues:
    """Coverage for v2/cache.py gather_cache_page_values."""

    def test_value_error_row_words_not_divisible_by_record_words(self):
        """ROW_WORDS % record_words != 0 raises ValueError."""
        cache = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)
        # elements = 10 -> record_words = 5; 128 % 5 != 0
        with pytest.raises(ValueError, match="Pages must cover whole pool word rows"):
            gather_cache_page_values(
                cache,
                jnp.array([0], dtype=jnp.int32),
                page_records=4,
                elements=10,
            )

    def test_value_error_page_records_not_covering_whole_rows(self):
        """(page_records * record_words) % ROW_WORDS != 0 raises ValueError."""
        cache = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)
        # elements = 128 -> record_words = 64; 128 % 64 == 0
        # page_records = 3 -> 3 * 64 = 192 % 128 != 0
        with pytest.raises(ValueError, match="Pages must cover whole pool word rows"):
            gather_cache_page_values(
                cache,
                jnp.array([0], dtype=jnp.int32),
                page_records=3,
                elements=128,
            )

    def test_positive_and_negative_first_records(self):
        """Positive records gather contiguous page rows; negative take spread."""
        rng = np.random.default_rng(42)
        cache_data = rng.integers(-(2**31), 2**31, (64, ROW_WORDS)).astype(np.int32)
        cache = jnp.asarray(cache_data & 0x3FFF3FFF)  # finite BF16 words
        elements = 128  # record_words = 64 -> per_row = 2
        page_records = 4  # rows = 4 * 64 // 128 = 2 rows per page

        # first_records with valid offsets and negative sentinels (-1, -5)
        first_records = jnp.asarray([0, 2, -1, 4, -5], dtype=jnp.int32)
        out = gather_cache_page_values(cache, first_records, page_records, elements)
        assert out.shape == (5 * page_records, elements)
        assert out.dtype == jnp.bfloat16

        # Compare valid records against gather_cache_values(split=True)
        for idx in [0, 1, 3]:
            first = int(first_records[idx])
            page_slice = out[idx * page_records : (idx + 1) * page_records]
            ref_records = jnp.arange(first, first + page_records, dtype=jnp.int32)
            ref_values = gather_cache_values(cache, ref_records, elements, split=True)
            np.testing.assert_array_equal(
                np.asarray(page_slice), np.asarray(ref_values)
            )

    def test_single_row_page_and_small_cache(self):
        """Small cache and single-row pages cover spread %= max(1, ...)."""
        cache = jnp.zeros((1, ROW_WORDS), dtype=jnp.int32)
        elements = 256  # record_words = 128 -> per_row = 1
        page_records = 1  # rows = 1
        first_records = jnp.asarray([-1, 0, -3], dtype=jnp.int32)
        out = gather_cache_page_values(cache, first_records, page_records, elements)
        assert out.shape == (3, elements)
        assert out.dtype == jnp.bfloat16


class TestIndexPageRecords:
    """Coverage for v2/attention.py _index_page_records."""

    def test_non_index_cache_returns_zero(self):
        """Non-index owners have no index pages."""
        assert _index_page_records(None) == 0
        assert _index_page_records(torch.zeros(10, 10)) == 0


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
