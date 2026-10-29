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

from dataclasses import astuple, replace
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import vllm_torchtpu.models.vllm.deepseek_v41.v2.attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import indexer_impl as indexer_ops
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import (
    _index_page_records,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    ROW_WORDS,
    IndexCacheRows,
    gather_cache_page_values,
    gather_cache_values,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.candidate_cache import CandidateCache
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_cache import (
    PackedFP8IndexCacheRows,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl import (
    _INDEX_KEYS_PER_ITER,
    _INDEX_MAX_TILE,
    _visible_top_k,
    initialize_indexer,
    paged_index_scores_jax,
    paged_indexer,
    select_indexer_jax,
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

    def test_non_index_cache_returns_zero(self):
        """Non-index owners have no index pages."""
        assert _index_page_records(None) == 0
        assert _index_page_records(torch.zeros(10, 10)) == 0

    def test_valid_qualifying_layout_returns_per_page(self):
        """Properly aligned BF16 layout returns per_page."""
        raw = torch.zeros((64, ROW_WORDS), dtype=torch.int32)
        # dim = 128 -> record_bytes = 256 -> per_row = 2
        # per_page = 16, stride = 32, offset = 16 -> all % 2 == 0
        layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 16 * 256)
        cache = IndexCacheRows(raw, layout)
        assert _index_page_records(cache) == 16

    def test_2bank_lo_rows_alignment(self):
        """2-bank cache returns per_page if lo_rows*per_row % per_page == 0, else 0."""
        raw_lo_aligned = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
        raw_hi = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
        layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 0)
        cache_aligned = IndexCacheRows((raw_lo_aligned, raw_hi), layout)
        assert _index_page_records(cache_aligned) == 16

        raw_lo_unaligned = torch.zeros((5, ROW_WORDS), dtype=torch.int32)
        cache_unaligned = IndexCacheRows((raw_lo_unaligned, raw_hi), layout)
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
        from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout

        rng = np.random.default_rng(200)
        heads, dim = 2, 128
        tokens, width = 4, 1024
        page_records = 32

        pool = jnp.asarray(
            rng.integers(-(2**31), 2**31, (256, ROW_WORDS)), dtype=jnp.int32
        ) & jnp.int32(0x3FFF3FFF)
        pool_records = pool.shape[0] * index_key_layout.SLOTS
        pages = rng.integers(
            0, pool_records // page_records, (tokens, width // page_records)
        )
        records = (
            (pages[..., None] * page_records + np.arange(page_records))
            .reshape(tokens, width)
            .astype(np.int32)
        )
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

        monkeypatch.setattr(indexer_ops, "_sparsecore_top_k", mock_sc_top_k)

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
        """Covers paged_indexer, select_indexer_jax, and initialize_indexer."""
        # 1. paged_indexer wrapper with batch_max=None and page_records
        called_args = []

        def mock_op(*args):
            called_args.append(args)
            return (
                torch.zeros(args[0].shape[0], args[8], dtype=torch.int32),
                args[5],
            )

        monkeypatch.setattr(indexer_ops, "_indexer_op", mock_op)

        q = torch.zeros(4, 2, 128)
        pool = torch.zeros(16, ROW_WORDS, dtype=torch.int32)
        cache_obj = SimpleNamespace(raw=pool)
        records = torch.zeros(4, 256, dtype=torch.int32)
        weights = torch.zeros(4, 256)
        lengths = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        candidates = torch.zeros(4, 8, dtype=torch.int32)
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
            requests,
            topk=16,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
        )
        assert len(called_args) == 1
        # page_records follows the eight tensor inputs and five selectors.
        assert called_args[0][13] == 0
        assert int(called_args[0][7].item()) == 40

        # With explicit batch_max and page_records
        paged_indexer(
            q,
            cache_obj,
            records,
            weights,
            lengths,
            candidates,
            requests,
            topk=16,
            block_size=16,
            candidate_write=False,
            candidate_filter=False,
            batch_max=torch.tensor(50, dtype=torch.int32),
            page_records=32,
        )
        assert len(called_args) == 2
        assert called_args[1][13] == 32
        assert int(called_args[1][7].item()) == 50

        # 2. select_indexer_jax with page_records=0 vs page_records=64
        q_j = jnp.zeros((2, 2, 128), dtype=jnp.bfloat16)
        pool_j = jnp.zeros((16, ROW_WORDS), dtype=jnp.int32)
        records_j = jnp.full((2, 256), -1, dtype=jnp.int32)
        weights_j = jnp.zeros((2, 2), dtype=jnp.float32)
        lengths_j = jnp.array([10, 20], dtype=jnp.int32)
        cand_j = jnp.full((2, 4), -1, dtype=jnp.int32)
        req_j = jnp.arange(2, dtype=jnp.int32)

        sel0, pub0 = select_indexer_jax(
            q_j,
            pool_j,
            records_j,
            weights_j,
            lengths_j,
            cand_j,
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

        sel64, pub64 = select_indexer_jax(
            q_j,
            pool_j,
            records_j,
            weights_j,
            lengths_j,
            cand_j,
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

    @pytest.mark.cpu_test
    @pytest.mark.parametrize("bank_count", [1, 2])
    def test_initialize_indexer_and_fake_handler(self, bank_count):
        """Covers initialize_indexer custom op and fake tensor registration."""
        from jax.sharding import Mesh

        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        with set_vllm_model_wrapper_context(mesh=mesh):
            indexer_ops._indexer_op = None
            initialize_indexer()
            assert indexer_ops._indexer_op is not None
            # Test early return branch
            saved_op = indexer_ops._indexer_op
            initialize_indexer()
            assert indexer_ops._indexer_op is saved_op

        # Verify fake registration by calling op with meta device tensors
        q = torch.empty((4, 2, 128), dtype=torch.bfloat16, device="meta")
        pool = torch.empty((16, ROW_WORDS), dtype=torch.int32, device="meta")
        records = torch.empty((4, 256), dtype=torch.int32, device="meta")
        weights = torch.empty((4, 256), dtype=torch.float32, device="meta")
        lengths = torch.empty((4,), dtype=torch.int32, device="meta")
        candidates = torch.empty((4, 8), dtype=torch.int32, device="meta")
        requests = torch.empty((4,), dtype=torch.int32, device="meta")
        batch_max = torch.empty((1,), dtype=torch.int32, device="meta")

        op = (
            indexer_ops._indexer_op
            if bank_count == 1
            else indexer_ops._indexer_2bank_op
        )
        f_sel, f_pub = op(
            q,
            *((pool,) * bank_count),
            records,
            weights,
            lengths,
            candidates,
            requests,
            batch_max,
            16,
            16,
            False,
            False,
            False,
            32,
        )
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
        attn._max_model_len = 196608
        attn.candidate_source_layer = 20
        attn.candidate_block_size = 16
        attn.candidate_block_buffer = torch.full((128, 8), -1, dtype=torch.int32)
        attn.indexer = SimpleNamespace(
            head_dim=128,
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
        attn._project_index_q = lambda q, p: q
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

    def _make_selection_call(self, attn, count, *, cache_offset=0, packed=False):
        cache_type = PackedFP8IndexCacheRows if packed else IndexCacheRows
        cache = cache_type(
            torch.zeros((64, 128 if packed else ROW_WORDS), dtype=torch.int32),
            (
                (4, 16, 132 if packed else 128),
                torch.uint8 if packed else torch.bfloat16,
                32 * 256,
                cache_offset,
            ),
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
    @pytest.mark.parametrize("packed", [False, True], ids=["bf16", "fp8"])
    def test_select_paged_indices_split_part_and_padding_calculation(
        self, count, packed, monkeypatch
    ):
        """Split queries keep their owners; FP8 padding has no request owner."""
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
            captured_args["page_records"] = kwargs.get("page_records")
            expected_part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
            live = min(count, expected_part)
            torch.testing.assert_close(
                requests_part[:live], torch.arange(live, dtype=torch.int32)
            )
            assert torch.all(requests_part[live:] == (-1 if packed else 0))
            assert torch.all(lengths_part[live:] == 0)
            return (
                torch.zeros(expected_part, 4, dtype=torch.int32),
                candidates_part,
            )

        monkeypatch.setattr(indexer_ops, "paged_indexer", mock_paged_indexer)
        monkeypatch.setattr(indexer_ops, "paged_fp8_indexer", mock_paged_indexer)

        select = self._make_selection_call(
            attn, count, cache_offset=16 * 256, packed=packed
        )

        with v2_attn.stage_attention_cache_scope():
            selected = select()

        assert selected.shape == (count, 4)
        expected_part = max(2, ((count + ranks - 1) // ranks + 1) & ~1)
        assert captured_args["q_part_shape"][0] == expected_part
        assert captured_args["lengths_part_shape"][0] == expected_part
        assert captured_args["page_records"] == (0 if packed else 16)

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

        monkeypatch.setattr(indexer_ops, "paged_indexer", mock_paged_indexer)

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

        monkeypatch.setattr(indexer_ops, "paged_indexer", mock_paged_indexer)

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

        monkeypatch.setattr(indexer_ops, "paged_indexer", mock_paged_indexer)

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

            # Selection must leave both banks of the Index-K owner unchanged.
            raw_lo = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
            raw_hi = torch.zeros((32, ROW_WORDS), dtype=torch.int32)
            layout = ((4, 16, 128), torch.bfloat16, 32 * 256, 0)
            index_cache_2b = IndexCacheRows((raw_lo, raw_hi), layout)
            monkeypatch.setattr(
                indexer_ops,
                "paged_indexer",
                lambda q_part, idx_c, *a, **kw: (
                    torch.zeros(q_part.shape[0], 4, dtype=torch.int32),
                    a[3],
                ),
            )
            sel3 = select(index_cache=index_cache_2b)
            assert sel3.shape == (count, 4)
            assert torch.all(index_cache_2b.raw == 0)
            assert torch.all(index_cache_2b.raw_hi == 0)


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


@pytest.fixture(params=[False, True], ids=["bf16", "fp8"])
def sparse_indexer_case(request):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import sparse_mqa_indexer as module
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_cache import (
        PackedFP8IndexCacheRows,
    )

    packed = request.param
    raw = torch.zeros((128, 128), dtype=torch.int32)
    cache_type = PackedFP8IndexCacheRows if packed else IndexCacheRows
    cache = cache_type(
        raw,
        (
            (1, 128, 132 if packed else 128),
            torch.uint8 if packed else torch.bfloat16,
            32768,
            0,
        ),
    )
    candidates = torch.zeros((2, 2048), dtype=torch.int32)
    op = module.SparseMQAIndexer(
        SimpleNamespace(prefix="source-a"),
        512,
        candidates,
        8,
        candidate_source_layer=20,
    )
    inputs = dict(
        q=torch.zeros(
            (2, 32, 128), dtype=torch.float8_e4m3fn if packed else torch.bfloat16
        ),
        weights=torch.ones((2, 32)),
        cache=cache,
        records=torch.zeros((1, 768 if packed else 98304), dtype=torch.int32),
        lengths=torch.full((2, 1), 98304, dtype=torch.int32),
        requests=torch.zeros(2, dtype=torch.int32),
        candidates=candidates,
        batch_max=torch.tensor(98304, dtype=torch.int32),
        page_records=128,
        unordered=True,
        score_width=98304,
        stage_cache={},
    )
    return module, op, inputs, packed


@pytest.mark.cpu_test
@pytest.mark.parametrize("mode", ["full", "publish", "reuse", "uncached"])
def test_sparse_mqa_indexer_routes_both_cache_formats(
    monkeypatch, sparse_indexer_case, mode
):
    """Full scans, publication and candidate reuse share one callable entry."""
    module, op, inputs, packed = sparse_indexer_case
    op.candidate_blocks = None if mode == "full" else inputs["candidates"]
    op.candidate_write = mode == "publish"
    if mode == "uncached":
        inputs["stage_cache"] = None

    def paged(
        q,
        cache,
        records,
        weights,
        lengths,
        candidates,
        requests,
        **kwargs,
    ):
        assert kwargs["candidate_write"] == (mode == "publish")
        assert kwargs["candidate_filter"] == (mode in ("reuse", "uncached"))
        assert kwargs["score_width" if packed else "page_records"] == (
            98304 if packed else 128
        )
        selected = q[:, 0, :1].float().int().expand(-1, 512)
        if mode == "reuse":
            return selected + 7, candidates
        return selected, candidates + 1

    def cached(q, *args, **kwargs):
        return q[:, 0, :1].float().int().expand(-1, 512) + 7

    monkeypatch.setattr(module.indexer_impl, "paged_fp8_indexer", paged)
    monkeypatch.setattr(module.indexer_impl, "paged_indexer", paged)

    def prepare(*args, **kwargs):
        return CandidateCache(*(inputs["requests"],) * 5)

    monkeypatch.setattr(module, "prepare_candidate_cache", prepare)
    monkeypatch.setattr(module, "select_candidate_cache", cached)
    call = torch.compile(op, backend="eager", fullgraph=True)
    original_cache = inputs["cache"].raw.clone()
    selected, published = call(**inputs)
    torch.testing.assert_close(
        selected,
        torch.full(
            (2, 512),
            7 if mode == "reuse" else 0,
            dtype=torch.int32,
        ),
    )
    if mode == "reuse":
        torch.testing.assert_close(published, inputs["candidates"])
    else:
        torch.testing.assert_close(published, inputs["candidates"] + 1)
    torch.testing.assert_close(inputs["cache"].raw, original_cache)


@pytest.mark.cpu_test
@pytest.mark.parametrize("two_banks", [False, True])
def test_sparse_mqa_indexer_reuses_only_matching_stage_and_source(
    monkeypatch, sparse_indexer_case, two_banks
):
    """Both formats gather once and invalidate on publication, writes or new stage."""
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import candidate_cache

    module, op, inputs, packed = sparse_indexer_case
    if two_banks:
        inputs["cache"].raw_hi = torch.zeros_like(inputs["cache"].raw)
    prepared = []

    def prepare(*args, **kwargs):
        token = torch.tensor(len(prepared), dtype=torch.int32)
        prepared.append(token)
        if packed:
            assert len(args[0]) == (2 if two_banks else 1)
            assert args[0][0] is inputs["cache"].raw
            return (token,) * 6
        return CandidateCache(*(token,) * 5)

    def select(q, *args):
        if packed:
            assert len(args[2]) == 6
        return torch.zeros((q.shape[0], 512), dtype=torch.int32)

    monkeypatch.setattr(module, "prepare_candidate_cache", prepare)
    monkeypatch.setattr(candidate_cache, "_select_op", select)
    monkeypatch.setattr(candidate_cache, "_select_2bank_op", select)
    monkeypatch.setattr(module.indexer_impl, "_prepare_op", prepare)
    monkeypatch.setattr(module.indexer_impl, "_prepared_select_op", select)
    for _ in range(4):
        op(**inputs)
    assert len(prepared) == 1
    inputs["stage_cache"][("candidate_publication", 20)] = 1
    op(**inputs)
    assert len(prepared) == 2
    op.k_cache.prefix = "source-b"
    op(**inputs)
    assert len(prepared) == 3
    inputs["stage_cache"][("candidate_index_version", "source-b")] = 1
    op(**inputs)
    assert len(prepared) == 4
    inputs["stage_cache"] = {}
    op(**inputs)
    assert len(prepared) == 5


@pytest.mark.cpu_test
@pytest.mark.parametrize("bank_count,page_size", [(1, 128), (2, 8)])
def test_fp8_indexer_bank_sequence_roundtrip(monkeypatch, bank_count, page_size):
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
        quantize_index_rows,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        indexer_impl as indexer_ops,
    )

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with jax.default_device(jax.devices("cpu")[0]):
        rows = (33 * page_size + 127) // 128
        banks = tuple(jnp.zeros((2 * rows, 128), jnp.int32) for _ in range(bank_count))
        layout = (2 * bank_count, rows, 0, page_size)
        records = jnp.array([0, 2 * bank_count * page_size - 1, -1], jnp.int32)
        keys = jnp.asarray(
            np.random.default_rng(19).normal(size=(3, 128)), jnp.bfloat16
        )
        updated = indexer_ops._write(banks, records, keys, jnp.empty((0,)), layout)
        actual = indexer_ops._read(updated, records, layout)
        quantized, scales, _ = quantize_index_rows(keys)
        expected = (quantized.astype(jnp.float32) * scales[:, None]).astype(
            jnp.bfloat16
        )
        expected = expected.at[-1].set(0)
        np.testing.assert_array_equal(actual, expected)
        untouched = indexer_ops._read(updated, jnp.array([1]), layout)
        np.testing.assert_array_equal(untouched, jnp.zeros((1, 128), jnp.bfloat16))


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens,width,budget_rows,key_tile,context_length",
    [
        (64, 16384, 24, None, 1024),
        (64, 16384, 16, 4096, 4096),
        (96, 32768, 24, 4096, 4096),
        (64, 16384, 16, 4096, 4097),
        (64, 16384, 16, 4096, 8193),
        (64, 131072, 16, 65536, 98304),
    ],
)
def test_fp8_indexer_multitile_compilation(
    monkeypatch, tokens, width, budget_rows, key_tile, context_length
):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
        PackedFP8Layout,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
        indexer_impl as indexer_ops,
    )

    monkeypatch.setattr(
        indexer_ops, "_INDEX_UNTILED_SCORE_ELEMENTS", budget_rows * width
    )
    if key_tile is not None:
        monkeypatch.setattr(
            indexer_ops,
            "PACKED_FP8_MQA_CONFIG",
            replace(indexer_ops.PACKED_FP8_MQA_CONFIG, key_tile=key_tile),
        )
    layout = PackedFP8Layout(2)
    page = jnp.concatenate(
        (
            jnp.full((32, 128), 0x38383838, jnp.int32),  # FP8 ones.
            jnp.ones((1, 128), jnp.float32).view(jnp.int32),
        )
    )
    lengths = jnp.arange(tokens, dtype=jnp.int32) % 5 * (context_length // 4)
    lengths = lengths.at[-1].set(context_length)

    def select(q, banks, pages, weights, lengths, requests):
        return indexer_ops.select_fp8_indexer_jax(
            q,
            banks,
            pages,
            weights,
            lengths,
            jnp.empty((tokens, 0), jnp.int32),
            requests,
            lengths.max(),
            astuple(layout),
            512,
            8,
            False,
            False,
            True,
            width,
        )[0]

    actual = jax.jit(select)(
        jnp.ones((tokens, 32, 128), jnp.float8_e4m3fn),
        (page, page),
        jnp.broadcast_to(jnp.arange(2)[:, None], (2, width // 128)),
        jnp.ones((tokens, 32), jnp.float32),
        lengths,
        jnp.repeat(jnp.array([0, 1], jnp.int32), tokens // 2)
        if key_tile is not None
        else jnp.tile(jnp.array([1, 0], jnp.int32), tokens // 2),
    )
    expected = jnp.where(jnp.arange(512) < lengths[:, None], jnp.arange(512), -1)
    np.testing.assert_array_equal(indexer_ops._compact_selected(actual), expected)
    np.testing.assert_array_equal(actual >= 0, expected >= 0)


@pytest.mark.parametrize("mode", ["dense", "publish", "candidate"])
def test_fp8_decode_schedule_matches_bounded(monkeypatch, mode):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    from tests.kernels.deepseek_v41.test_fp8_paged_mqa_logits import _pack_fp8_cache
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
        PackedFP8Layout,
    )

    rng = np.random.default_rng(902)
    requests, queries, width, chunk = 48, 6, 8192, 96
    rows = requests * queries
    count = 0 if mode == "dense" else 128
    layout = PackedFP8Layout(requests * width // 128)
    keys = torch.from_numpy(
        rng.normal(size=(layout.pages, 128, 1, 128)).astype(np.float32)
    ).to(torch.bfloat16)
    words = _pack_fp8_cache(keys, layout)
    split = layout.pages // 2 * layout.page_rows
    pages = jnp.asarray(rng.permutation(layout.pages).reshape(requests, -1), jnp.int32)
    q = jnp.asarray(rng.normal(size=(rows, 32, 128)), jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.1, 1, size=(rows, 32)), jnp.float32)
    lengths = jnp.tile(width - queries + 1 + jnp.arange(queries), requests)
    if mode != "publish":
        lengths = lengths.at[-queries:].set(jnp.array([0, 1, 127, 511, 512, 513]))
    candidates = jnp.broadcast_to(jnp.arange(count, dtype=jnp.int32), (rows, count))
    owners = jnp.repeat(jnp.arange(requests, dtype=jnp.int32), queries)
    monkeypatch.setitem(
        indexer_ops.PACKED_FP8_INDEXER_QUERY_ROWS,
        (rows, width, count, 512, 8, 2),
        chunk,
    )
    monkeypatch.setitem(
        indexer_ops.PACKED_FP8_PREPARED_QUERY_ROWS, (rows, count * 8, 512), chunk
    )

    def run(q, banks, pages, weights, lengths, candidates, owners):
        return indexer_ops.select_fp8_indexer_jax(
            q,
            banks,
            pages,
            weights,
            lengths,
            candidates,
            owners,
            lengths.max(),
            astuple(layout),
            512,
            8,
            mode == "publish",
            mode == "candidate",
            True,
            width,
        )

    args = (
        q,
        (words[:split], words[split:]),
        pages,
        weights,
        lengths,
        candidates,
        owners,
    )
    calls = []
    unrolled = indexer_ops.indexer.pipeline_score_topk_unrolled

    def traced_unrolled(*args):
        calls.append(args[0])
        return unrolled(*args)

    monkeypatch.setattr(
        indexer_ops.indexer, "pipeline_score_topk_unrolled", traced_unrolled
    )
    candidate = jax.jit(run).lower(*args).compile()
    actual = jax.tree.map(np.asarray, candidate(*args))
    assert calls and set(calls) == {rows // chunk}
    monkeypatch.setattr(
        indexer_ops.indexer,
        "pipeline_score_topk_unrolled",
        indexer_ops.indexer.pipeline_score_topk_bounded,
    )
    reference = jax.jit(lambda *values: run(*values)).lower(*args).compile()
    expected = jax.tree.map(np.asarray, reference(*args))
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.sort(got, axis=-1), np.sort(want, axis=-1))
    valid = actual[0] >= 0
    assert np.all(~valid[:, 1:] | valid[:, :-1])


@pytest.mark.parametrize("dtype", ["auto", "bf16", "fp8", "mxfp4"])
def test_indexer_cache_initializes_only_its_format(monkeypatch, dtype):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import indexer_cache

    initialized = []
    config = SimpleNamespace(
        cache_config=SimpleNamespace(),
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda default: default
            if dtype == "auto"
            else dtype
        ),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(index_head_dim=128)),
    )
    monkeypatch.setattr(indexer_cache, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(
        indexer_cache.upstream.DeepseekV4IndexerCache,
        "__init__",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        indexer_cache.indexer_ops,
        "initialize_fp8_indexer",
        lambda: initialized.append(1),
    )
    if dtype == "mxfp4":
        with pytest.raises(ValueError, match="requires bf16 or fp8"):
            indexer_cache.IndexerCache()
        assert not initialized
        return
    cache = indexer_cache.IndexerCache()
    assert cache.packed_fp8 == (dtype != "bf16")
    assert initialized == ([1] if cache.packed_fp8 else [])
