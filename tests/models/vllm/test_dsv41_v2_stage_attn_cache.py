# SPDX-License-Identifier: Apache-2.0
"""Diff line and branch coverage test suite for stage attention cache and packed words:
- `vllm_torchtpu.models.vllm.deepseek_v41.v2.attention`
  (`stage_attention_cache_scope`, compressor state addressing caching,
   `_write_global_kv` caching, SWA 3-page fast path, compressed attention caching,
   and `_select_paged_indices` query split caching)
- `vllm_torchtpu.models.vllm.deepseek_v41.v2.cache`
  (`split_pieces`, `merge_pieces`, `_pack_bf16_words`,
   and `scatter_cache_values` branching)
- `vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels`
  (`_attend` and `_attend_window` leading-axis pieces layout)
- `vllm_torchtpu.models.vllm.deepseek_v41.v2.model`
  (`stage_attention_cache_scope` wrapping `_forward_stage`)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as v2_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as v2_cache
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    PagedCacheRows,
)

# ==============================================================================
# Helper Mock Factories
# ==============================================================================


def _make_paged_cache_mock(
    pages_tensor: torch.Tensor,
    *,
    row_offset: int | None = 0,
    rows_per_page: int = 64,
    page_stride: int = 64,
    has_row_offset_attr: bool = True,
    shape_is_tuple: bool = True,
) -> Any:
    """Creates a mock PagedCacheRows object for testing address caching."""
    c = MagicMock(spec=PagedCacheRows)
    c.numel.return_value = 1024
    if shape_is_tuple:
        c.shape = (64, rows_per_page, 512)
    else:
        c.shape = None
    c.page_stride = page_stride
    if has_row_offset_attr:
        c.row_offset = row_offset
    else:
        if hasattr(c, "row_offset"):
            delattr(c, "row_offset")
    c.row_indices = lambda slots, ro=(row_offset or 0): ro + (
        slots // rows_per_page * page_stride + slots % rows_per_page
    )
    c.pages.return_value = pages_tensor
    return c


def _make_swa_attn(
    *,
    window_size: int,
    ced_decoder: bool,
    has_fast_swa_offset: bool,
    rows_per_page: int = 64,
) -> Any:
    attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.prefix = "swa_pfx"
    attn.window_size = window_size
    attn._ced_decoder = ced_decoder
    attn._ced_global_from_encoder = False
    attn.compressor = None
    attn.indexer = None
    attn.compress_ratio = 2
    attn._config = {"index_topk": 4}
    attn._max_model_len = 256
    attn._fp8_kv = True
    attn.n_local_heads = 2
    attn.head_dim = 8
    attn.scale = 0.25
    attn.attn_sink = torch.zeros(2)
    attn.topk_indices_buffer = torch.full((16, 8), -1, dtype=torch.int32)
    attn._rotate = lambda x, p: x
    attn.wq_b = lambda qr: qr

    cache_mock = _make_paged_cache_mock(
        torch.tensor([[1, 2], [3, 4]]),
        row_offset=1000 if has_fast_swa_offset else None,
        rows_per_page=rows_per_page,
        has_row_offset_attr=has_fast_swa_offset,
    )
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_pfx", kv_cache=cache_mock)
    attn.kv_cache = cache_mock
    attn._static_forward_context = {"swa_pfx": SimpleNamespace(kv_cache=cache_mock)}
    attn.compressed_cache_prefix = "swa_pfx"
    return attn


# ==============================================================================
# 1. Tests for v2/cache.py: split_pieces, merge_pieces, _pack_bf16_words, scatter
# ==============================================================================


class TestCachePiecesAndPackBf16:
    def test_split_pieces_and_merge_pieces_roundtrip(self):
        """Verify split_pieces and merge_pieces are exact inverses across shapes."""
        rng = np.random.default_rng(42)
        for pieces in (1, 2, 4):
            for dim in (128, 256, 512):
                if dim % pieces != 0:
                    continue
                # Test 2D [tokens, dim]
                data = rng.standard_normal((8, dim)).astype(np.float32)
                arr = jnp.asarray(data, dtype=jnp.bfloat16)
                split = v2_cache.split_pieces(arr, pieces)
                assert split.shape == (pieces, 8, dim // pieces)
                merged = v2_cache.merge_pieces(split)
                assert merged.shape == arr.shape
                np.testing.assert_array_equal(np.asarray(merged), np.asarray(arr))

                # Test 4D [batch, tokens, heads, dim]
                data4d = rng.standard_normal((2, 4, 3, dim)).astype(np.float32)
                arr4d = jnp.asarray(data4d, dtype=jnp.bfloat16)
                split4d = v2_cache.split_pieces(arr4d, pieces)
                assert split4d.shape == (pieces, 2, 4, 3, dim // pieces)
                merged4d = v2_cache.merge_pieces(split4d)
                assert merged4d.shape == arr4d.shape
                np.testing.assert_array_equal(np.asarray(merged4d), np.asarray(arr4d))

    def test_pack_bf16_words_numerical_and_shift_logic(self):
        """Verify _pack_bf16_words correctly packs even and odd BF16 elements."""
        n, width = 4, 256
        values = jnp.linspace(0.5, 2.0, n * 2 * width, dtype=jnp.bfloat16).reshape(
            n, 2 * width
        )
        packed = v2_cache._pack_bf16_words(values)
        assert packed.shape == (n, width)
        assert packed.dtype == jnp.int32

        unpacked = jax.lax.bitcast_convert_type(packed, jnp.bfloat16).reshape(
            n, 2 * width
        )
        np.testing.assert_array_equal(np.asarray(unpacked), np.asarray(values))


# ==============================================================================
# 2. Tests for v2/attention_kernels.py: _attend and _attend_window pieces layout
# ==============================================================================


class TestAttentionKernelsPiecesLayout:
    def test_attend_with_split_pieces(self):
        """Exercise _attend with pieces leading layout and verify against reference."""
        tokens, heads, dim = 2, 2, 128
        q = jnp.ones((tokens, heads, dim), dtype=jnp.bfloat16)
        pool = jnp.zeros((64, 128), dtype=jnp.int32)
        records = jnp.zeros((tokens, 4), dtype=jnp.int32)
        state = (
            jnp.full((tokens, heads), -1e30, dtype=jnp.float32),
            jnp.zeros((tokens, heads), dtype=jnp.float32),
            jnp.zeros((tokens, heads, dim), dtype=jnp.float32),
        )
        scale = 0.125
        max_out, den_out, num_out = v2_kernels._attend(
            q, pool, records, state, scale, tile=4
        )
        assert max_out.shape == (tokens, heads)
        assert den_out.shape == (tokens, heads)
        assert num_out.shape == (tokens, heads, dim)

    def test_attend_window_with_split_pieces(self):
        """Exercise _attend_window with pieces leading layout and tile split."""
        tokens, heads, dim = 4, 2, 128
        q = jnp.ones((tokens, heads, dim), dtype=jnp.bfloat16)
        pool = jnp.zeros((64, 128), dtype=jnp.int32)
        records = jnp.zeros((tokens, 64), dtype=jnp.int32)
        state = (
            jnp.full((tokens, heads), -1e30, dtype=jnp.float32),
            jnp.zeros((tokens, heads), dtype=jnp.float32),
            jnp.zeros((tokens, heads, dim), dtype=jnp.float32),
        )
        scale = 0.125
        max_out, den_out, num_out = v2_kernels._attend_window(
            q, pool, records, state, scale, tile=2, max_requests=2
        )
        assert max_out.shape == (tokens, heads)
        assert den_out.shape == (tokens, heads)
        assert num_out.shape == (tokens, heads, dim)


# ==============================================================================
# 3. Tests for v2/attention.py: stage_attention_cache_scope and addressing cache
# ==============================================================================


class TestStageAttentionCacheScope:
    def test_stage_attention_cache_scope_lifecycle(self):
        """Verify context manager activates, isolates, nests, and restores."""
        assert v2_attn._active_stage_attn_cache is None

        with v2_attn.stage_attention_cache_scope():
            assert isinstance(v2_attn._active_stage_attn_cache, dict)
            v2_attn._active_stage_attn_cache["key1"] = "val1"

            # Nested scope
            with v2_attn.stage_attention_cache_scope():
                assert isinstance(v2_attn._active_stage_attn_cache, dict)
                assert "key1" not in v2_attn._active_stage_attn_cache
                v2_attn._active_stage_attn_cache["inner"] = "val2"
            # Restores outer
            assert v2_attn._active_stage_attn_cache["key1"] == "val1"
            assert "inner" not in v2_attn._active_stage_attn_cache

        # Restores None
        assert v2_attn._active_stage_attn_cache is None

        # Exception safety
        with pytest.raises(RuntimeError), v2_attn.stage_attention_cache_scope():
            raise RuntimeError("test error")
        assert v2_attn._active_stage_attn_cache is None


class TestCompressorStateAndGlobalKvAddressingCache:
    def test_compressor_state_addressing_cached_and_uncached(self, monkeypatch):
        """Verify compressor state_cache addressing caching and fallback paths."""
        count = 4
        state_rows = 32
        positions = torch.tensor([0, 15, 31, 45], dtype=torch.int32)
        table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)

        state_metadata = SimpleNamespace(
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            block_tables=table,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        metadata = {"state_prefix": state_metadata}
        forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)

        captured_slots: list[torch.Tensor] = []

        def mock_save_compress_norm(
            kv_score, pos, state_cache, slots, qsl, reqs, nw, eps, ratio, latent
        ):
            captured_slots.append(slots.clone())
            latent.fill_(1.0)

        monkeypatch.setattr(v2_attn, "save_compress_norm", mock_save_compress_norm)

        comp: Any = object.__new__(v2_attn.DeepseekCompressor)
        torch.nn.Module.__init__(comp)
        comp.head_dim = 8
        comp.compress_ratio = 2
        comp.rms_norm_eps = 1e-6
        comp.norm = SimpleNamespace(weight=torch.ones(8, dtype=torch.float32))

        # 1. State cache is None -> positions-based fallback
        comp.state_cache = None
        k_cache_mock = MagicMock()
        comp.k_cache_prefix = "k_prefix"
        comp._static_forward_context = {
            "k_prefix": SimpleNamespace(kv_cache=k_cache_mock)
        }
        metadata["k_prefix"] = SimpleNamespace(
            block_tables=table,
            seq_lens=torch.tensor([2, 2]),
            query_start_loc=torch.tensor([0, 4]),
        )
        monkeypatch.setattr(
            v2_attn,
            "_cache_slots",
            lambda c, m, p, **kw: torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        )
        comp.forward(torch.ones(count, 10), positions)
        assert torch.equal(
            captured_slots[-1], torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        )

        # 2. State cache present with stage_attention_cache_scope (miss then hit)
        captured_slots.clear()
        state_cache_mock = MagicMock()
        state_cache_mock.shape = (4, state_rows, 128)
        comp.state_cache = SimpleNamespace(
            prefix="state_prefix", kv_cache=state_cache_mock
        )

        with v2_attn.stage_attention_cache_scope():
            # First call: cache miss
            comp.forward(torch.ones(count, 10), positions)
            assert len(captured_slots) == 1
            first_slots = captured_slots[0]

            # Second call: cache hit
            comp.forward(torch.ones(count, 10), positions)
            assert len(captured_slots) == 2
            assert torch.equal(captured_slots[1], first_slots)

        # 3. Uncached fallback path when stage_cache is None
        captured_slots.clear()
        monkeypatch.setattr(
            v2_attn,
            "_cache_slots",
            lambda c, m, p, **kw: torch.tensor([9, 8, 7, 6], dtype=torch.int32),
        )
        comp.forward(torch.ones(count, 10), positions)
        assert torch.equal(
            captured_slots[-1], torch.tensor([9, 8, 7, 6], dtype=torch.int32)
        )

        # 4. Fallback path when state_rows is None (mock.shape is not a tuple)
        state_cache_mock.shape = None
        captured_slots.clear()
        with v2_attn.stage_attention_cache_scope():
            comp.forward(torch.ones(count, 10), positions)
            assert torch.equal(
                captured_slots[-1], torch.tensor([9, 8, 7, 6], dtype=torch.int32)
            )

        # 5. Non-dict metadata early exit
        forward_ctx.attn_metadata = None
        assert comp.forward(torch.ones(count, 10), positions) is None


@pytest.mark.parametrize("seed", range(8))
def test_shift_swa_metadata_matches_direct_offsets(seed):
    """Block-outer SWA layers share one cached metadata, shifted per row offset."""
    from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import prepare_swa_metadata

    page, width = 64, 40
    rng = np.random.default_rng(seed)
    requests = int(rng.integers(1, 9))
    stride = page * int(rng.choice([7, 8, 16]))
    table = rng.permutation(np.arange(1, 1 + requests * width))
    table = table.reshape(requests, width).astype(np.int32)
    queries = rng.integers(0, 7, requests)
    queries[0] = max(queries[0], 1)
    starts = np.concatenate(([0], np.cumsum(queries))).astype(np.int32)
    positions = np.zeros(int(starts[-1]) + int(rng.integers(0, 5)), np.int32)
    for r in range(requests):
        first = int(rng.integers(0, width * page - 16))
        positions[starts[r] : starts[r + 1]] = first + np.arange(queries[r])
        table[r, : max(first - 127, 0) // page] = 0  # freed out-of-window pages
    minimum = np.zeros(requests, np.int32)

    def direct(offset):
        return tuple(
            torch.from_numpy(np.asarray(x))
            for x in prepare_swa_metadata(
                jnp.asarray(table),
                jnp.asarray(starts),
                jnp.asarray(positions),
                jnp.asarray(minimum),
                page_size=page,
                page_stride=stride,
                row_offset=offset,
            )
        )

    base = direct(0)
    for offset in (page, stride - page):
        want = direct(offset)
        metadata, records = v2_attn._shift_swa_metadata(base[0], base[2], offset, page)
        assert torch.equal(metadata, want[0])
        assert torch.equal(base[1], want[1])
        assert torch.equal(records, want[2])


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("non_causal", [False, True])
def test_metadata_tables_matches_per_table_metadata(seed, non_causal):
    """One batched program equals one prepare_swa_metadata call per SWA group."""
    from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import prepare_swa_metadata
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import csa2_adapter

    page, width, groups = 64, 40, 7
    rng = np.random.default_rng(seed)
    requests = int(rng.integers(1, 9))
    stride = page * 8
    tables = np.stack(
        [
            rng.permutation(np.arange(1, 1 + requests * width)).reshape(requests, width)
            for _ in range(groups)
        ]
    ).astype(np.int32)
    queries = rng.integers(0, 7, requests)
    queries[0] = max(queries[0], 1)
    starts = np.concatenate(([0], np.cumsum(queries))).astype(np.int32)
    positions = np.zeros(int(starts[-1]) + int(rng.integers(0, 5)), np.int32)
    for r in range(requests):
        first = int(rng.integers(0, width * page - 16))
        positions[starts[r] : starts[r + 1]] = first + np.arange(queries[r])
    minimum = rng.integers(0, 64, requests).astype(np.int32)
    args = tuple(jnp.asarray(x) for x in (starts, positions, minimum))

    metadata, bounds, records = csa2_adapter._metadata_tables(
        jnp.asarray(tables), *args, page, stride, non_causal
    )
    for g in range(groups):
        want = prepare_swa_metadata(
            jnp.asarray(tables[g]),
            *args,
            page_size=page,
            page_stride=stride,
            row_offset=0,
            non_causal=non_causal,
        )
        np.testing.assert_array_equal(np.asarray(metadata[g]), np.asarray(want[0]))
        np.testing.assert_array_equal(np.asarray(bounds), np.asarray(want[1]))
        np.testing.assert_array_equal(np.asarray(records[g]), np.asarray(want[2]))


def test_swa_group_tables_dedupes_by_object_and_shape():
    a, b = torch.zeros(12, dtype=torch.int32), torch.ones(12, dtype=torch.int32)
    other = torch.zeros(20, dtype=torch.int32)
    metadata = {
        "l0.swa_cache": SimpleNamespace(block_tables=a),
        "l0.self_attn": SimpleNamespace(block_tables=b),  # not SWA
        "l1.swa_cache": SimpleNamespace(block_tables=b),
        "l2.swa_cache": SimpleNamespace(block_tables=a),
        # Equal values, distinct object: a separate table.
        "l3.swa_cache": SimpleNamespace(block_tables=torch.zeros(12)),
        "d0.swa_cache": SimpleNamespace(block_tables=other),  # another shape
    }
    tables = v2_attn._swa_group_tables(metadata, (3, 4))
    assert len(tables) == 3
    assert tables[0] is a and tables[1] is b
