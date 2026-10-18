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
     * Not PagedCacheRows instance -> 0
     * dtype != torch.bfloat16 (e.g. fp8, fp32, None) -> 0
     * row_bytes % record_bytes != 0 -> 0
     * (per_page * record_bytes) % row_bytes != 0 -> 0
     * row_offset % per_row != 0 -> 0
     * page_stride % per_row != 0 -> 0
     * Valid qualifying layout -> per_page
   - DeepseekV41Attention._select_paged_indices (lines 1351-1490):
     * split=True (ranks > 1, part = (count + ranks - 1) // ranks, padding = ranks)
     * split=False (ranks = 1, padding = 0, no slice / no gather)
     * cached_query_split hit vs miss (stage_attention_cache_scope)
     * candidate_write=True vs candidate_write=False
     * candidate_filter=True vs False
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
     * pages=True (gather_cache_page_values branch)
     * pages=False: page_records is None
     * pages=False: tile % page_records != 0
     * Numerical equivalence of pages=True vs pages=False
   - _visible_top_k (lines 1535-1555):
     * SparseCore path with rows < 16 (unpadded)
     * SparseCore path with rows >= 16
     * Fallback path: unordered=False, lengths=None, width < 2048,
       width % 16 != 0, k % 16 != 0
   - paged_indexer_jax, update_and_select_jax, initialize_indexer, paged_indexer
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.models.vllm.deepseek_v41.v2.attention as v2_attn
import vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels as v2_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import (
    _index_page_records,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _INDEX_KEYS_PER_ITER,
    _INDEX_MAX_TILE,
    _visible_top_k,
    initialize_indexer,
    paged_index_scores_jax,
    paged_indexer,
    update_and_select_jax,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    ROW_WORDS,
    PagedCacheRows,
    gather_cache_page_values,
    gather_cache_values,
)


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

    def test_not_paged_cache_rows_returns_zero(self):
        """Non-PagedCacheRows object returns 0."""
        assert _index_page_records(None) == 0
        assert _index_page_records(torch.zeros(10, 10)) == 0

    def test_non_bf16_dtype_returns_zero(self):
        """Dtype other than torch.bfloat16 returns 0."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        layout_fp32 = ((4, 16, 64), torch.float32, 32 * 64 * 4, 0)
        cache_fp32 = PagedCacheRows(raw, layout_fp32)
        assert _index_page_records(cache_fp32) == 0

        # Mock PagedCacheRows with FP8 dtype
        mock_fp8 = MagicMock(spec=PagedCacheRows)
        mock_fp8.dtype = torch.float8_e4m3fn
        assert _index_page_records(mock_fp8) == 0

    def test_row_bytes_not_divisible_by_record_bytes_returns_zero(self):
        """row_bytes % record_bytes != 0 returns 0."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 512, BF16 -> record_bytes = 1024; ROW_BYTES = 512; 512 % 1024 != 0
        layout = ((4, 16, 512), torch.bfloat16, 32 * 1024, 0)
        cache = PagedCacheRows(raw, layout)
        assert _index_page_records(cache) == 0

    def test_page_record_bytes_not_divisible_by_row_bytes_returns_zero(self):
        """(per_page * record_bytes) % row_bytes != 0 returns 0."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 128 -> record_bytes = 256; ROW_BYTES = 512;
        # per_page = 3 -> 768 % 512 != 0
        layout = ((4, 3, 128), torch.bfloat16, 32 * 256, 0)
        cache = PagedCacheRows(raw, layout)
        assert _index_page_records(cache) == 0

    def test_row_offset_not_aligned_returns_zero(self):
        """index_cache.row_offset % per_row != 0 returns 0."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 128 -> record_bytes = 256 -> per_row = 2
        # offset = 1 record -> offset % 2 == 1 != 0
        layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 1 * 256)
        cache = PagedCacheRows(raw, layout)
        assert _index_page_records(cache) == 0

    def test_page_stride_not_aligned_returns_zero(self):
        """index_cache.page_stride % per_row != 0 returns 0."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 128 -> record_bytes = 256 -> per_row = 2
        # stride = 33 records -> stride % 2 == 1 != 0
        layout = ((4, 16, 128), torch.bfloat16, 33 * 256, 0)
        cache = PagedCacheRows(raw, layout)
        assert _index_page_records(cache) == 0

    def test_valid_qualifying_layout_returns_per_page(self):
        """Properly aligned BF16 layout returns per_page."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 128 -> record_bytes = 256 -> per_row = 2
        # per_page = 16, stride = 32, offset = 16 -> all % 2 == 0
        layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 16 * 256)
        cache = PagedCacheRows(raw, layout)
        assert _index_page_records(cache) == 16

    def test_2bank_lo_rows_alignment(self):
        """2-bank cache returns per_page if lo_rows*per_row % per_page == 0, else 0."""
        raw_lo_aligned = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
        raw_hi = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
        layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 0)
        cache_aligned = PagedCacheRows((raw_lo_aligned, raw_hi), layout)
        assert _index_page_records(cache_aligned) == 16

        raw_lo_unaligned = torch.zeros((5, ROW_WORDS), dtype=torch.int32)
        cache_unaligned = PagedCacheRows((raw_lo_unaligned, raw_hi), layout)
        assert _index_page_records(cache_unaligned) == 0


class TestAttentionKernelsDecodeIndexer:
    """Coverage for v2/attention_kernels.py / changes."""

    def test_constants_and_tile_budget_spectrum(self):
        """Test _INDEX_KEYS_PER_ITER = 262144 and _INDEX_MAX_TILE = 16384 budget."""
        assert _INDEX_KEYS_PER_ITER == 262144
        assert _INDEX_MAX_TILE == 16384

        # 1. tokens <= 16: budget capped at _INDEX_MAX_TILE (16384)
        pool = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)
        q = jnp.zeros((1, 2, 128), dtype=jnp.bfloat16)
        records = jnp.full((1, 16384), -1, dtype=jnp.int32)
        weights = jnp.zeros((1, 2), dtype=jnp.float32)
        lengths = jnp.array([10], dtype=jnp.int32)

        # tokens = 1 -> budget = min(max(512, min(16384, 262144 // 1)), 16384) = 16384
        scores = paged_index_scores_jax(q, pool, records, weights, lengths)
        assert scores.shape == (1, 16384)

        # 2. tokens = 16 -> min(max(512, min(16384, 262144 // 16)), 16384) = 16384
        q16 = jnp.zeros((16, 2, 128), dtype=jnp.bfloat16)
        weights16 = jnp.zeros((16, 2), dtype=jnp.float32)
        lengths16 = jnp.full((16,), 10, dtype=jnp.int32)
        scores16 = paged_index_scores_jax(q16, pool, records, weights16, lengths16)
        assert scores16.shape == (16, 16384)

        # 3. tokens = 512 -> budget hits 512 floor
        q512 = jnp.zeros((512, 2, 128), dtype=jnp.bfloat16)
        weights512 = jnp.zeros((512, 2), dtype=jnp.float32)
        lengths512 = jnp.full((512,), 10, dtype=jnp.int32)
        scores512 = paged_index_scores_jax(q512, pool, records, weights512, lengths512)
        assert scores512.shape == (512, 16384)

        # 4. Small width clamps budget via -(-width // 128) * 128
        records_short = jnp.full((1, 256), -1, dtype=jnp.int32)
        scores_short = paged_index_scores_jax(q, pool, records_short, weights, lengths)
        assert scores_short.shape == (1, 256)

    def test_paged_index_scores_pages_vs_unpaged_equivalence(self):
        """paged_index_scores_jax with pages=True vs pages=False match."""
        rng = np.random.default_rng(200)
        heads, dim = 2, 128
        tokens, width = 4, 1024
        page_records = 32

        pool = jnp.asarray(
            rng.integers(-(2**31), 2**31, (256, ROW_WORDS)), dtype=jnp.int32
        ) & jnp.int32(0x3FFF3FFF)
        records = rng.integers(0, 256 * 2, (tokens, width)).astype(np.int32)
        # Align records to contiguous page blocks of 32
        for r in range(tokens):
            for p in range(0, width, page_records):
                base = int(records[r, p] // 2 * 2)
                records[r, p : p + page_records] = base + np.arange(page_records)
        records[:, -10:] = -1  # invalid tail
        records = jnp.asarray(records)

        q = jnp.asarray(rng.standard_normal((tokens, heads, dim)), jnp.bfloat16)
        weights = jnp.asarray(rng.standard_normal((tokens, heads)), jnp.float32)
        lengths = jnp.asarray(rng.integers(50, width, tokens), dtype=jnp.int32)

        # 1. Unpaged (page_records = None)
        unpaged_scores = paged_index_scores_jax(
            q, pool, records, weights, lengths, page_records=None
        )
        # 2. Paged (page_records = 32)
        paged_scores = paged_index_scores_jax(
            q, pool, records, weights, lengths, page_records=page_records
        )
        # 3. Non-dividing page_records (e.g. 300, tile % 300 != 0 -> pages = False)
        nondiv_scores = paged_index_scores_jax(
            q, pool, records, weights, lengths, page_records=300
        )

        valid = np.asarray(records >= 0)
        np.testing.assert_allclose(
            np.asarray(paged_scores)[valid],
            np.asarray(unpaged_scores)[valid],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(nondiv_scores)[valid],
            np.asarray(unpaged_scores)[valid],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_paged_index_scores_request_branches(self):
        """Covers requests is not None (decode tokens <= records vs prefill)."""
        heads, dim = 2, 128
        pool = jnp.zeros((64, ROW_WORDS), dtype=jnp.int32)

        # Branch 1a: requests is not None, tokens <= records.shape[0],
        # tokens >= 6 and tokens % 6 == 0 (grouped branch with tokens=6 and tokens=18)
        q_dec = jnp.zeros((18, heads, dim), dtype=jnp.bfloat16)
        records_dec = jnp.full((24, 256), -1, dtype=jnp.int32)
        weights_dec = jnp.zeros((18, heads), dtype=jnp.float32)
        lengths_dec = jnp.full((18,), 10, dtype=jnp.int32)
        requests_dec = jnp.zeros((18,), dtype=jnp.int32)

        res_dec_18 = paged_index_scores_jax(
            q_dec,
            pool,
            records_dec,
            weights_dec,
            lengths_dec,
            requests=requests_dec,
            page_records=16,
        )
        assert res_dec_18.shape == (18, 256)

        res_dec = paged_index_scores_jax(
            q_dec[:6],
            pool,
            records_dec,
            weights_dec[:6],
            lengths_dec[:6],
            requests=requests_dec[:6],
            page_records=16,
        )
        assert res_dec.shape == (6, 256)

        # Branch 1b: requests is not None, tokens <= records.shape[0],
        # tokens not divisible by 6 (tokens = 2 and tokens = 7)
        res_dec_2 = paged_index_scores_jax(
            q_dec[:2],
            pool,
            records_dec,
            weights_dec[:2],
            lengths_dec[:2],
            requests=requests_dec[:2],
            page_records=16,
        )
        assert res_dec_2.shape == (2, 256)

        res_dec_7 = paged_index_scores_jax(
            q_dec[:7],
            pool,
            records_dec,
            weights_dec[:7],
            lengths_dec[:7],
            requests=requests_dec[:7],
            page_records=16,
        )
        assert res_dec_7.shape == (7, 256)

        # Branch 2: requests is not None and tokens > records.shape[0]
        # (prefill row_step loop)
        q_pref = jnp.zeros((4, heads, dim), dtype=jnp.bfloat16)
        records_pref = jnp.full((2, 256), -1, dtype=jnp.int32)
        weights_pref = jnp.zeros((4, heads), dtype=jnp.float32)
        lengths_pref = jnp.array([10, 10, 20, 20], dtype=jnp.int32)
        requests_pref = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

        res_pref = paged_index_scores_jax(
            q_pref,
            pool,
            records_pref,
            weights_pref,
            lengths_pref,
            requests=requests_pref,
            page_records=16,
        )
        assert res_pref.shape == (4, 256)

    def test_visible_top_k_sparsecore_and_fallback_branches(self, monkeypatch):
        """Covers _visible_top_k unpadded rows < 16, rows >= 16, and fallback."""
        sparsecore_calls = []

        def mock_sc_top_k(scores, k, lengths):
            sparsecore_calls.append((scores.shape, k))
            return jnp.zeros((scores.shape[0], k), dtype=jnp.int32)

        monkeypatch.setattr(v2_kernels, "_sparsecore_top_k", mock_sc_top_k)

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

    def test_paged_indexer_and_wrappers_dispatch(self, monkeypatch):
        """Covers paged_indexer, update_and_select_jax, and initialize_indexer."""
        # 1. paged_indexer wrapper with batch_max=None and page_records
        called_args = []

        def mock_op(*args):
            called_args.append(args)
            return (
                args[1],
                torch.zeros(args[0].shape[0], args[9], dtype=torch.int32),
                args[5],
            )

        monkeypatch.setattr(v2_kernels, "_indexer_op", mock_op)

        q = torch.zeros(4, 2, 128)
        pool = torch.zeros(16, ROW_WORDS, dtype=torch.int32)
        cache_obj = SimpleNamespace(raw=pool)
        records = torch.zeros(4, 256, dtype=torch.int32)
        weights = torch.zeros(4, 256)
        lengths = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        candidates = torch.zeros(4, 8, dtype=torch.int32)
        write_records = torch.zeros(0, dtype=torch.int32)
        write_keys = torch.zeros(0, 128)
        requests = torch.arange(4, dtype=torch.int32)

        # Without batch_max (defaults to lengths.max()) and without
        # page_records (defaults to 0)
        paged_indexer(
            q,
            cache_obj,
            records,
            weights,
            lengths,
            candidates,
            write_records,
            write_keys,
            requests,
            topk=16,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
        )
        assert len(called_args) == 1
        # page_records arg is index 15
        assert called_args[0][15] == 0
        # batch_max arg is index 9
        assert int(called_args[0][9].item()) == 40

        # With explicit batch_max and page_records
        paged_indexer(
            q,
            cache_obj,
            records,
            weights,
            lengths,
            candidates,
            write_records,
            write_keys,
            requests,
            topk=16,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            batch_max=torch.tensor(50, dtype=torch.int32),
            page_records=32,
        )
        assert len(called_args) == 2
        assert called_args[1][15] == 32
        assert int(called_args[1][9].item()) == 50

        # 2. update_and_select_jax with page_records=0 vs page_records=64
        q_j = jnp.zeros((2, 2, 128), dtype=jnp.bfloat16)
        pool_j = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)
        records_j = jnp.full((2, 256), -1, dtype=jnp.int32)
        weights_j = jnp.zeros((2, 2), dtype=jnp.float32)
        lengths_j = jnp.array([10, 20], dtype=jnp.int32)
        cand_j = jnp.full((2, 4), -1, dtype=jnp.int32)
        wrec_j = jnp.zeros((0,), dtype=jnp.int32)
        wkeys_j = jnp.zeros((0, 128), dtype=jnp.bfloat16)
        req_j = jnp.arange(2, dtype=jnp.int32)

        up_pool0, sel0, pub0 = update_and_select_jax(
            q_j,
            pool_j,
            records_j,
            weights_j,
            lengths_j,
            cand_j,
            wrec_j,
            wkeys_j,
            req_j,
            topk=16,
            block_size=16,
            batch_max=jnp.array(20, dtype=jnp.int32),
            candidate_write=False,
            candidate_filter=False,
            unordered=False,
            page_records=0,
        )
        assert sel0.shape == (2, 16)

        up_pool64, sel64, pub64 = update_and_select_jax(
            q_j,
            pool_j,
            records_j,
            weights_j,
            lengths_j,
            cand_j,
            wrec_j,
            wkeys_j,
            req_j,
            topk=16,
            block_size=16,
            batch_max=jnp.array(20, dtype=jnp.int32),
            candidate_write=False,
            candidate_filter=False,
            unordered=False,
            page_records=64,
        )
        assert sel64.shape == (2, 16)

    def test_initialize_indexer_and_fake_handler(self):
        """Covers initialize_indexer custom op and fake tensor registration."""
        from jax.sharding import Mesh

        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        with set_vllm_model_wrapper_context(mesh=mesh):
            v2_kernels._indexer_op = None
            initialize_indexer()
            assert v2_kernels._indexer_op is not None
            # Test early return branch
            saved_op = v2_kernels._indexer_op
            initialize_indexer()
            assert v2_kernels._indexer_op is saved_op

        # Verify fake registration by calling op with meta device tensors
        q = torch.empty((4, 2, 128), dtype=torch.bfloat16, device="meta")
        pool = torch.empty((16, ROW_WORDS), dtype=torch.int32, device="meta")
        records = torch.empty((4, 256), dtype=torch.int32, device="meta")
        weights = torch.empty((4, 256), dtype=torch.float32, device="meta")
        lengths = torch.empty((4,), dtype=torch.int32, device="meta")
        candidates = torch.empty((4, 8), dtype=torch.int32, device="meta")
        write_recs = torch.empty((0,), dtype=torch.int32, device="meta")
        keys = torch.empty((0, 128), dtype=torch.bfloat16, device="meta")
        requests = torch.empty((4,), dtype=torch.int32, device="meta")
        batch_max = torch.empty((1,), dtype=torch.int32, device="meta")

        f_pool, f_sel, f_pub = v2_kernels._indexer_op(
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
            32,
        )
        assert f_pool.shape == pool.shape
        assert f_sel.shape == (4, 16)
        assert f_pub.shape == candidates.shape


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
        attn.candidate_block_size = 16
        attn.candidate_block_buffer = torch.full((128, 8), -1, dtype=torch.int32)
        attn.indexer = SimpleNamespace(
            head_dim=128,
            n_head=2,
            indexer_op=SimpleNamespace(
                candidate_blocks=torch.zeros(1),
                candidate_write=candidate_write,
            ),
        )
        attn._project_index_q = lambda q, p: q

        monkeypatch.setattr(
            v2_attn, "get_tensor_model_parallel_world_size", lambda: ranks
        )
        monkeypatch.setattr(
            v2_attn,
            "_gather_query_rows",
            lambda x: torch.cat([x] * ranks, dim=0),
        )
        return attn

    @pytest.mark.parametrize("count", [1, 16, 17, 32])
    def test_select_paged_indices_split_part_and_padding_calculation(
        self, count, monkeypatch
    ):
        """Verifies part = (count + ranks - 1) // ranks and padding = ranks."""
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
            write_records,
            write_keys,
            requests_part,
            *args,
            **kwargs,
        ):
            captured_args["q_part_shape"] = q_part.shape
            captured_args["lengths_part_shape"] = lengths_part.shape
            captured_args["page_records"] = kwargs.get("page_records")
            expected_part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
            return (
                index_cache.raw,
                torch.zeros(expected_part, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(v2_attn, "paged_indexer", mock_paged_indexer)

        # Mock PagedCacheRows with qualifying layout (per_page = 16)
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 16 * 256)
        index_cache_mock = PagedCacheRows(raw, layout)

        qr = torch.zeros(count, 128)
        positions = torch.arange(count, dtype=torch.int32)
        records = torch.zeros(count, 256, dtype=torch.int32)
        requests = torch.arange(count, dtype=torch.int32)
        weights = torch.zeros(count, 2)
        lengths = torch.full((count, 1), 50, dtype=torch.int32)
        swa_slots = torch.zeros(count, 1, dtype=torch.int32)

        with v2_attn.stage_attention_cache_scope():
            selected = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                records,
                requests,
                weights,
                lengths,
                swa_slots,
                None,
            )

        assert selected.shape == (count, 4)
        expected_part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
        assert captured_args["q_part_shape"][0] == expected_part
        assert captured_args["lengths_part_shape"][0] == expected_part
        assert captured_args["page_records"] == 16

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
            write_records,
            write_keys,
            requests_part,
            *args,
            **kwargs,
        ):
            part = q_part.shape[0]
            # published part
            pub_part = torch.full((part, 8), 42, dtype=torch.int32)
            return (
                index_cache.raw,
                torch.zeros(part, 4, dtype=torch.int32),
                pub_part,
            )

        monkeypatch.setattr(v2_attn, "paged_indexer", mock_paged_indexer)

        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        index_cache_mock = SimpleNamespace(raw=raw)

        qr = torch.zeros(count, 128)
        positions = torch.arange(count, dtype=torch.int32)
        records = torch.zeros(count, 256, dtype=torch.int32)
        requests = torch.arange(count, dtype=torch.int32)
        weights = torch.zeros(count, 2)
        lengths = torch.full((count, 1), 50, dtype=torch.int32)
        swa_slots = torch.zeros(count, 1, dtype=torch.int32)

        # Also supply pending_index_write to test pending_index_write is not None
        pending_write = (
            torch.tensor([1, 2], dtype=torch.int32),
            torch.zeros(2, 128),
        )

        selected = attn._select_paged_indices(
            qr,
            positions,
            index_cache_mock,
            records,
            requests,
            weights,
            lengths,
            swa_slots,
            pending_write,
        )

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
            write_records,
            write_keys,
            requests_part,
            *args,
            **kwargs,
        ):
            assert q_part.shape[0] == count
            assert candidates_part.shape == (count, 0)
            return (
                index_cache.raw,
                torch.zeros(count, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(v2_attn, "paged_indexer", mock_paged_indexer)

        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        index_cache_mock = SimpleNamespace(raw=raw)

        qr = torch.zeros(count, 128)
        positions = torch.arange(count, dtype=torch.int32)
        records = torch.zeros(count, 256, dtype=torch.int32)
        requests = torch.arange(count, dtype=torch.int32)
        weights = torch.zeros(count, 2)
        lengths = torch.full((count, 1), 50, dtype=torch.int32)
        swa_slots = torch.zeros(count, 1, dtype=torch.int32)

        selected = attn._select_paged_indices(
            qr,
            positions,
            index_cache_mock,
            records,
            requests,
            weights,
            lengths,
            swa_slots,
            None,
        )

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
            write_records,
            write_keys,
            requests_part,
            *args,
            **kwargs,
        ):
            nonlocal call_count
            call_count += 1
            part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
            assert q_part.shape[0] == part
            return (
                index_cache.raw,
                torch.zeros(part, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(v2_attn, "paged_indexer", mock_paged_indexer)

        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        index_cache_mock = SimpleNamespace(raw=raw)

        qr = torch.zeros(count, 128)
        positions = torch.arange(count, dtype=torch.int32)
        records = torch.zeros(count, 256, dtype=torch.int32)
        requests = torch.arange(count, dtype=torch.int32)
        weights = torch.zeros(count, 2)
        lengths = torch.full((count, 1), 50, dtype=torch.int32)
        swa_slots = torch.zeros(count, 1, dtype=torch.int32)

        with v2_attn.stage_attention_cache_scope():
            # Call 1: cache miss
            sel1 = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                records,
                requests,
                weights,
                lengths,
                swa_slots,
                None,
            )
            assert sel1.shape == (count, 4)
            assert call_count == 1

            # Call 2: cache hit
            sel2 = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                records,
                requests,
                weights,
                lengths,
                swa_slots,
                None,
            )
            assert sel2.shape == (count, 4)
            assert call_count == 2

            # Call 3: 2-bank PagedCacheRows returning (lo, hi) tuple
            raw_lo = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
            raw_hi = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
            layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 0)
            index_cache_2b = PagedCacheRows((raw_lo, raw_hi), layout)
            monkeypatch.setattr(
                v2_attn,
                "paged_indexer",
                lambda q_part, idx_c, *a, **kw: (
                    (torch.ones_like(raw_lo), torch.full_like(raw_hi, 2)),
                    torch.zeros(q_part.shape[0], 4, dtype=torch.int32),
                    a[3],
                ),
            )
            sel3 = attn._select_paged_indices(
                qr,
                positions,
                index_cache_2b,
                records,
                requests,
                weights,
                lengths,
                swa_slots,
                None,
            )
            assert sel3.shape == (count, 4)
            assert torch.all(index_cache_2b.raw == 1)
            assert torch.all(index_cache_2b.raw_hi == 2)


@pytest.mark.cpu_test
def test_candidate_envelope_keeps_token_rows_dynamic():
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.candidate import (
        cache_capacity,
    )

    def route(q):
        capacity = cache_capacity(tuple(q.shape), (2, 98304), 2048, 8, 128)
        return q + (1 if capacity is not None else 0)

    compiled = torch.compile(route, backend="eager", fullgraph=True)
    for rows in (12, 13, 2048):
        q = torch.zeros((rows, 32, 128), dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(q, 0, min=2, max=4096)
        torch.testing.assert_close(compiled(q), q + 1)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "groups,width,layout,capacity,packed",
    [
        (2, 98304, "ordinary", 49152, False),
        (48, 98816, "halfplane", 49152, True),
        (48, 98816, "ordinary", 49152, False),
        (24, 98816, "halfplane", 49152, True),
        (48, 98304, "halfplane", 49152, True),
        (8, 65536, "halfplane", 32768, True),
        (2, 1048576, "halfplane", 49152, True),
    ],
)
def test_candidate_cache_reuses_only_within_stage_and_source_version(
    monkeypatch, groups, width, layout, capacity, packed
):
    """Eligible halfplane shapes share packed keys only within a source stage."""
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import tuned_block_sizes

    monkeypatch.setattr(tuned_block_sizes, "CANDIDATE_KEY_CAPACITY", capacity)
    count = groups * 6
    attn = TestAttentionSelectPagedIndicesDecode()._setup_mock_attn(
        monkeypatch, ranks=1
    )
    attn._config = {"index_topk": 512}
    attn.compress_ratio = 1
    attn.candidate_block_size = 8
    attn.candidate_source_layer = 20
    attn.candidate_block_buffer = torch.zeros((count, 2048), dtype=torch.int32)
    attn.indexer.k_cache = SimpleNamespace(prefix="source-a")
    cache = SimpleNamespace(
        raw=torch.zeros((128, 128), dtype=torch.int32), index_key_layout=layout
    )
    q = torch.zeros((count, 32, 128), dtype=torch.bfloat16)
    positions = torch.arange(count, dtype=torch.int32)
    records = torch.arange(groups * width, dtype=torch.int32).reshape(groups, -1)
    requests = torch.arange(groups, dtype=torch.int32).repeat_interleave(6)
    weights = torch.ones((count, 32))
    lengths = torch.full((count, 1), width, dtype=torch.int32)
    swa_valid = torch.ones(count, dtype=torch.bool)
    prepared = []
    used = []

    def prepare(cache, records, candidates, requests, lengths, **kwargs):
        assert kwargs["packed"] is packed
        assert kwargs["capacity"] == capacity
        result = object()
        prepared.append((result, requests.clone()))
        return result

    def select(
        q,
        cache,
        records,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        prepared,
        **kwargs,
    ):
        used.append(prepared)
        return torch.zeros((count, 512), dtype=torch.int32)

    def old(q, cache, records, weights, lengths, candidates, *args, **kwargs):
        return cache.raw, torch.zeros((count, 512), dtype=torch.int32), candidates + 1

    monkeypatch.setattr(v2_attn, "_index_page_records", lambda cache: 128)
    monkeypatch.setattr(v2_attn, "prepare_candidate_cache", prepare)
    monkeypatch.setattr(v2_attn, "select_candidate_cache", select)
    monkeypatch.setattr(v2_attn, "paged_indexer", old)

    def run(mapping):
        return attn._select_paged_indices(
            q, positions, cache, records, mapping, weights, lengths, swa_valid, None
        )

    with v2_attn.stage_attention_cache_scope():
        for _ in range(4):
            run(requests)
        assert len(prepared) == 1
        assert all(x is prepared[0][0] for x in used)
        attn.indexer.indexer_op.candidate_write = True
        run(requests)
        attn.indexer.indexer_op.candidate_write = False
        run(requests)
        assert len(prepared) == 2
        attn.indexer.k_cache.prefix = "source-b"
        run(requests)
        assert len(prepared) == 3
        v2_attn._active_stage_attn_cache[("candidate_index_version", "source-b")] = 1
        run(requests)
        assert len(prepared) == 4
    with v2_attn.stage_attention_cache_scope():
        reordered = requests.flip(0)
        run(reordered)
        assert len(prepared) == 5
        torch.testing.assert_close(prepared[-1][1], reordered)
