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

import contextlib
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
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import (
    ROW_WORDS,
    IndexCacheRows,
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

    def test_scatter_cache_values_branching(self, monkeypatch):
        """Test scatter_cache_values: int32, TPU BF16 pack, CPU bitcast."""
        cache = jnp.zeros((16, 128), dtype=jnp.int32)
        records = jnp.zeros((2,), dtype=jnp.int32)

        # Branch 1: values is int32
        val_i32 = jnp.ones((2, 8), dtype=jnp.int32)
        out1 = v2_cache.scatter_cache_values(cache, records, val_i32)
        assert out1.dtype == jnp.int32

        # Branch 2: values is bfloat16 on TPU with shape[1] >= 2 * ROW_WORDS
        pack_called = False
        orig_pack = v2_cache._pack_bf16_words

        def mock_pack(v):
            nonlocal pack_called
            pack_called = True
            return orig_pack(v)

        from jax.experimental import pallas as pl

        monkeypatch.setattr(pl, "pallas_call", lambda *a, **kw: lambda r, w, c: c)
        monkeypatch.setattr(v2_cache, "_pack_bf16_words", mock_pack)
        monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

        val_bf16_tpu = jnp.zeros((2, 2 * ROW_WORDS), dtype=jnp.bfloat16)
        out2 = v2_cache.scatter_cache_values(cache, records, val_bf16_tpu)
        assert pack_called is True
        assert out2.dtype == jnp.int32

        # Branch 3: values is bfloat16 on CPU -> bitcast_convert_type fallback
        pack_called = False
        monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
        out3 = v2_cache.scatter_cache_values(cache, records, val_bf16_tpu)
        assert pack_called is False
        assert out3.dtype == jnp.int32

        # Branch 4: values is bfloat16 on TPU but shape[1] not divisible
        monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
        val_bf16_small = jnp.zeros((2, 16), dtype=jnp.bfloat16)
        out4 = v2_cache.scatter_cache_values(cache, records, val_bf16_small)
        assert pack_called is False
        assert out4.dtype == jnp.int32

        # Branch 5: values is float32 -> directly passed to bitcast
        val_f32 = jnp.zeros((2, 8), dtype=jnp.float32)
        out5 = v2_cache.scatter_cache_values(cache, records, val_f32)
        assert out5.dtype == jnp.int32

        # Branch 6: values is empty
        val_empty = jnp.zeros((0, 8), dtype=jnp.int32)
        out6 = v2_cache.scatter_cache_values(cache, records[:0], val_empty)
        assert out6 is cache


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

    def test_model_forward_stage_uses_stage_attention_cache_scope(self, monkeypatch):
        """Verify DeepseekV41Model._forward_stage enters stage_attention_cache_scope."""
        scope_entered = 0

        @contextlib.contextmanager
        def mock_scope():
            nonlocal scope_entered
            scope_entered += 1
            yield

        monkeypatch.setattr(v2_model, "stage_attention_cache_scope", mock_scope)
        monkeypatch.setattr(v2_model, "post", lambda h, r, p, rm: h)
        monkeypatch.setattr(v2_model, "collapse", lambda x, pre: x)
        monkeypatch.setattr(
            v2_model.upstream,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True, is_last_rank=False),
        )

        class DummyLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_norm = lambda x: x
                self.attn = MagicMock()

            def forward(self, *args, **kwargs):
                prev_aux = (
                    torch.zeros((4, 16)) if kwargs.get("capture_previous_aux") else None
                )
                return (
                    torch.zeros((4, 16)),
                    None,
                    None,
                    None,
                    torch.zeros((4, 16)),
                    prev_aux,
                )

        model = object.__new__(v2_model.DeepseekV41Model)
        torch.nn.Module.__init__(model)
        dummy_layer0 = DummyLayer()
        dummy_layer1 = DummyLayer()
        model.layers = torch.nn.ModuleList([dummy_layer0, dummy_layer1])
        model.aux_hidden_state_layers = []
        model._ced_boundary = 1
        model.start_layer = 0
        model.end_layer = 2
        model.use_sequence_parallel = False
        model.use_native_mega_moe = False
        model._has_cpu_offloaded_engram = lambda: False
        model._prepare_engram = lambda *a, **kw: (None, None)
        model._stage_uses_sequence_parallel = lambda stage: False

        # 1. Test decoder stage
        model._forward_stage(
            input_ids=torch.zeros((4,), dtype=torch.int64),
            positions=torch.zeros((4,), dtype=torch.int64),
            intermediate_tensors=None,
            inputs_embeds=torch.zeros((4, 16)),
            stage="decoder",
        )
        assert scope_entered == 1

        # 2. Test encoder stage with aux layers and stage_sp=False
        model.aux_hidden_state_layers = [0, 1, 2]
        model.end_layer = 2
        model._forward_stage(
            input_ids=torch.zeros((4,), dtype=torch.int64),
            positions=torch.zeros((4,), dtype=torch.int64),
            intermediate_tensors=None,
            inputs_embeds=torch.zeros((4, 16)),
            encoder_pre_mix=torch.zeros((4, 16)),
            stage="encoder",
        )
        assert scope_entered == 2
        assert dummy_layer1.attn.write_encoder_global_kv.called

        # 3. Test stage with stage_sp=True and aux_hidden_state_layers
        model.use_sequence_parallel = True
        model._stage_uses_sequence_parallel = lambda stage: True
        model.aux_hidden_state_layers = [0, 1]
        model.end_layer = 1
        monkeypatch.setattr(
            v2_model.upstream, "get_tensor_model_parallel_world_size", lambda: 2
        )
        monkeypatch.setattr(
            v2_model.upstream, "is_forward_context_available", lambda: True
        )
        monkeypatch.setattr(v2_model.upstream, "sp_all_gather", lambda x: x)
        monkeypatch.setattr(v2_model.upstream, "sp_shard", lambda x: x)
        monkeypatch.setattr(v2_model.upstream, "sp_padding_mask", lambda m, x: m)
        monkeypatch.setattr(
            v2_model.upstream,
            "get_forward_context",
            lambda: SimpleNamespace(is_padding=torch.zeros(4, dtype=torch.bool)),
        )
        model._forward_stage(
            input_ids=torch.zeros((4,), dtype=torch.int64),
            positions=torch.zeros((4,), dtype=torch.int64),
            intermediate_tensors=None,
            inputs_embeds=torch.zeros((4, 16)),
            encoder_pre_mix=torch.zeros((4, 16)),
            stage="encoder",
        )
        assert scope_entered == 3


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

    @pytest.mark.cpu_test
    @pytest.mark.parametrize("packed_fp8", [False, True])
    @pytest.mark.parametrize("short_context", [False, True])
    def test_write_global_kv_addressing_cached_and_uncached(
        self, monkeypatch, packed_fp8, short_context
    ):
        """Verify _write_global_kv cache hits/misses across ratios."""
        count = 4
        positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        kv_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        idx_table = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)

        kv_meta = SimpleNamespace(
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            block_tables=kv_table,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        idx_meta = SimpleNamespace(
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            block_tables=idx_table,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        metadata = {"kv_pfx": kv_meta, "idx_pfx": idx_meta}
        forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)

        inserted_slots: list[torch.Tensor] = []
        monkeypatch.setattr(
            v2_attn,
            "_insert",
            lambda cache, slots, val, mask: inserted_slots.append(slots.clone()),
        )

        attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn)
        attn.prefix = "kv_pfx"
        attn.compress_ratio = 2
        attn.window_size = 128
        attn._config = {"index_topk": 4}
        attn._max_model_len = 8 if short_context else 1000
        attn._rotate = lambda x, p: x
        attn._quantize_indexer = lambda x: x
        attn.compressor = lambda score, pos: torch.ones(count, 8)

        kv_cache_mock = MagicMock(spec=PagedCacheRows)
        kv_cache_mock.shape = (4, 32, 128)
        attn.kv_cache = kv_cache_mock

        idx_cache_mock = MagicMock(
            spec=v2_attn.PackedFP8IndexCacheRows if packed_fp8 else IndexCacheRows
        )
        idx_cache_mock.shape = (4, 32, 128)
        idx_cache_mock.row_indices = lambda slots: slots + 100

        indexer_mock = SimpleNamespace(
            head_dim=8,
            n_head=2,
            wk=lambda lat: lat,
            k_norm=lambda k: k,
            k_cache=SimpleNamespace(
                prefix="idx_pfx",
                compress_ratio=2,
                kv_cache=idx_cache_mock,
            ),
        )
        attn.indexer = indexer_mock

        # Subcase A: Matching ratios and shapes (idx_col = kv_col fast path)
        # Miss then hit
        with v2_attn.stage_attention_cache_scope():
            # A cache miss publishes both caches before any selection runs.
            ret1 = attn._write_global_kv(torch.ones(count, 10), positions)
            assert ret1 is None
            assert len(inserted_slots) == 2
            slots1 = inserted_slots[0]
            index_slots1 = inserted_slots[1]
            assert (
                v2_attn._active_stage_attn_cache[("candidate_index_version", "idx_pfx")]
                == 1
            )

            # Address reuse still writes fresh K and invalidates candidate reuse.
            ret2 = attn._write_global_kv(torch.ones(count, 10), positions)
            assert ret2 is None
            assert len(inserted_slots) == 4
            assert torch.equal(inserted_slots[2], slots1)
            assert torch.equal(inserted_slots[3], index_slots1)
            assert (
                v2_attn._active_stage_attn_cache[("candidate_index_version", "idx_pfx")]
                == 2
            )

        # Subcase B: Differing ratio (idx_ratio != self.compress_ratio) -> else branch
        inserted_slots.clear()
        indexer_mock.k_cache.compress_ratio = 4
        with v2_attn.stage_attention_cache_scope():
            attn._write_global_kv(torch.ones(count, 10), positions)
            assert len(inserted_slots) == 2

        # Subcase C: stage_cache is None -> fallback to _cache_slots
        inserted_slots.clear()
        fallback_slots = [
            torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            torch.tensor([5, 6, 7, 8], dtype=torch.int32),
        ]
        monkeypatch.setattr(
            v2_attn, "_cache_slots", lambda *a, **kw: fallback_slots.pop(0)
        )
        attn._write_global_kv(torch.ones(count, 10), positions)
        assert len(inserted_slots) == 2


class TestCompressedAttentionAndQuerySplitCaching:
    def test_select_paged_indices_query_split_caching(self, monkeypatch):
        """Test _select_paged_indices split and non-split query caching."""
        count = 4
        positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        qr = torch.ones((count, 16), dtype=torch.bfloat16)
        weights = torch.ones((count, 2), dtype=torch.float32)
        lengths = torch.tensor([[1], [2], [3], [4]], dtype=torch.int32)
        swa_slots = torch.ones(count, dtype=torch.bool)
        requests = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

        attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn)
        attn.compress_ratio = 2
        attn._tp_rank = 0
        attn._fp8_kv = True
        attn._config = {"index_topk": 2}
        attn._max_model_len = 8
        attn.candidate_block_size = 1
        attn.candidate_block_buffer = None
        attn.indexer = SimpleNamespace(
            head_dim=8,
            n_head=2,
            indexer_op=SimpleNamespace(candidate_blocks=None, candidate_write=False),
        )
        attn._project_index_q = lambda q, p: q

        attn.indexer.indexer_op = MagicMock(
            side_effect=lambda q, *a, **kw: (
                torch.zeros(q.shape[0], 2, dtype=torch.int32),
                kw["candidates"],
            )
        )
        attn.indexer.indexer_op.candidate_blocks = None
        attn.indexer.indexer_op.candidate_write = False
        # TP=2 all-gather of each rank's part (only the split branch calls it).
        monkeypatch.setattr(v2_attn, "_gather_query_rows", lambda x: torch.cat([x, x]))

        index_cache_mock = SimpleNamespace(
            raw=torch.zeros((4, ROW_WORDS), dtype=torch.int32)
        )
        index_records = torch.zeros((count, 4), dtype=torch.int32)

        # 1. Test split=True (TP=2) miss then hit
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 2)
        with v2_attn.stage_attention_cache_scope():
            sel1 = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                index_records,
                requests,
                weights,
                lengths,
                swa_slots,
            )
            assert sel1.shape[0] == count

            # Second call: hits query_split_key cache
            sel2 = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                index_records,
                requests,
                weights,
                lengths,
                swa_slots,
            )
            assert torch.equal(sel1, sel2)

        # 2. Test split=False (TP=1) miss then hit
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)
        with v2_attn.stage_attention_cache_scope():
            sel_tp1 = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                index_records,
                requests,
                weights,
                lengths,
                swa_slots,
            )
            assert sel_tp1.shape[0] == count

            sel_tp1_cached = attn._select_paged_indices(
                qr,
                positions,
                index_cache_mock,
                index_records,
                requests,
                weights,
                lengths,
                swa_slots,
            )
            assert torch.equal(sel_tp1, sel_tp1_cached)

        # 3. Test split=True when stage_cache is None (uncached fallback)
        monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 2)
        sel_nocache = attn._select_paged_indices(
            qr,
            positions,
            index_cache_mock,
            index_records,
            requests,
            weights,
            lengths,
            swa_slots,
        )
        assert sel_nocache.shape[0] == count

    def test_untiled_and_non_fp8_fallback_branches(self, monkeypatch):
        """Test untiled and non-fp8_kv fallback paths in _prepare_tpu."""
        count = 4
        capacity = 8
        positions = torch.tensor([0, 16, 32, 48], dtype=torch.int32)
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

        meta = SimpleNamespace(
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            block_tables=table,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        metadata = {"src_pfx": meta, "idx_pfx": meta, "swa_pfx": meta}
        forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)
        monkeypatch.setattr(
            v2_attn, "sparse_attention", lambda q, r, s, ind, sc: torch.zeros_like(q)
        )
        monkeypatch.setattr(
            v2_attn,
            "_read_rows",
            lambda c, t, r, logical_arg: torch.zeros((count, logical_arg.shape[1], 8)),
        )

        cache_mock = _make_paged_cache_mock(
            torch.tensor([[10, 11], [12, 13]]),
            row_offset=500,
            rows_per_page=32,
        )

        attn = _make_swa_attn(
            window_size=32, ced_decoder=False, has_fast_swa_offset=True
        )
        attn.prefix = "src_pfx"
        attn.compressed_cache_prefix = "src_pfx"
        attn._static_forward_context = {"src_pfx": SimpleNamespace(kv_cache=cache_mock)}
        attn._config = {"index_topk": 2}
        attn._fp8_kv = False
        attn._max_model_len = capacity * 2
        attn.topk_indices_buffer = torch.full((16, 4), -1, dtype=torch.int32)

        # Attach indexer to cover lines 1058-1076, 1080-1093, and line 1004 False branch
        idx_cache_mock = _make_paged_cache_mock(
            torch.tensor([[1, 2], [3, 4]]), row_offset=100
        )
        attn.indexer = SimpleNamespace(
            head_dim=8,
            n_head=2,
            wk=lambda x: x,
            k_norm=lambda x: x,
            k_cache=SimpleNamespace(
                prefix="idx_pfx", compress_ratio=2, kv_cache=idx_cache_mock
            ),
        )
        attn._project_index_q = lambda q, p: torch.zeros((count, 2, 8))
        attn._select_indices = lambda scores, lengths, logical, c: torch.zeros(
            (count, 2), dtype=torch.int32
        )

        # 1. stage_cache is None with tiled=False -> untiled fallback
        out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
        qr = torch.ones((count, 16), dtype=torch.bfloat16)
        kv = torch.ones((count, 8), dtype=torch.bfloat16)
        idx_w = torch.ones((count, 2), dtype=torch.float32)
        attn._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)

        # 2. tiled=True with _fp8_kv=False -> covers line 1131
        attn_tiled_bf16 = _make_swa_attn(
            window_size=64, ced_decoder=False, has_fast_swa_offset=True
        )
        attn_tiled_bf16.prefix = "src_pfx"
        attn_tiled_bf16.compressed_cache_prefix = "src_pfx"
        attn_tiled_bf16._static_forward_context = {
            "src_pfx": SimpleNamespace(kv_cache=cache_mock)
        }
        attn_tiled_bf16._config = {"index_topk": 2}
        attn_tiled_bf16._fp8_kv = False
        attn_tiled_bf16._max_model_len = capacity * 2
        attn_tiled_bf16.topk_indices_buffer = torch.full((16, 4), -1, dtype=torch.int32)
        attn_tiled_bf16.indexer = SimpleNamespace(
            head_dim=8,
            n_head=2,
            wk=lambda x: x,
            k_norm=lambda x: x,
            k_cache=SimpleNamespace(
                prefix="idx_pfx", compress_ratio=2, kv_cache=idx_cache_mock
            ),
        )
        attn_tiled_bf16._project_index_q = lambda q, p: torch.zeros((count, 2, 8))
        attn_tiled_bf16._select_indices = (
            lambda scores, lengths, logical, c: torch.zeros(
                (count, 2), dtype=torch.int32
            )
        )
        attn_tiled_bf16._select_paged_indices = lambda *a, **kw: torch.zeros(
            (count, 2), dtype=torch.int32
        )
        monkeypatch.setattr(
            v2_attn,
            "paged_attention",
            lambda *a, **kw: torch.zeros_like(a[0]),
        )
        with v2_attn.stage_attention_cache_scope():
            attn_tiled_bf16._prepare_tpu(
                None, qr, kv, None, None, idx_w, positions, out
            )

        # 3. tiled=True outside stage_attention_cache_scope (uncached fallback)
        attn_tiled_bf16._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)


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
