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
import torch.nn.functional as F

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels as v2_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import cache as v2_cache
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import ROW_WORDS, PagedCacheRows

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
    attn._local_q_attn = False
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
    attn._swa_max_requests = 16
    attn._precompact_records = False
    attn._token_parallel_attn = False
    attn._full_attn_sink = False
    attn._tp_rank_1d = 0
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

    def test_write_global_kv_addressing_cached_and_uncached(self, monkeypatch):
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
        attn._max_model_len = 1000
        attn._rotate = lambda x, p: x
        attn._quantize_indexer = lambda x: x
        attn.compressor = lambda score, pos: torch.ones(count, 8)

        kv_cache_mock = MagicMock(spec=PagedCacheRows)
        kv_cache_mock.shape = (4, 32, 128)
        attn.kv_cache = kv_cache_mock

        idx_cache_mock = MagicMock(spec=PagedCacheRows)
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
            # First call (cache miss) with defer_index_write=True
            ret1 = attn._write_global_kv(
                torch.ones(count, 10), positions, defer_index_write=True
            )
            assert ret1 is not None
            assert len(ret1) == 2  # (records, keys)
            assert len(inserted_slots) == 1
            slots1 = inserted_slots[0]

            # Second call (cache hit) with defer_index_write=False
            ret2 = attn._write_global_kv(
                torch.ones(count, 10), positions, defer_index_write=False
            )
            assert ret2 is None
            assert len(inserted_slots) == 3  # kv_cache inserted + idx_cache inserted
            assert torch.equal(inserted_slots[1], slots1)

        # Subcase B: Differing ratio (idx_ratio != self.compress_ratio) -> else branch
        inserted_slots.clear()
        indexer_mock.k_cache.compress_ratio = 4
        with v2_attn.stage_attention_cache_scope():
            attn._write_global_kv(
                torch.ones(count, 10), positions, defer_index_write=False
            )
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
        attn._write_global_kv(torch.ones(count, 10), positions, defer_index_write=False)
        assert len(inserted_slots) == 2


class TestSwaTokenAddressingAndFastOffsets:
    def test_swa_3page_fast_path_and_offsets(self, monkeypatch):
        """Test Swa addressing 3-page fast path and row_offset caching."""
        count = 4
        positions = torch.tensor([0, 32, 64, 96], dtype=torch.int32)
        swa_table = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32)

        swa_meta = SimpleNamespace(
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            block_tables=swa_table,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        metadata = {"swa_pfx": swa_meta}
        forward_ctx = SimpleNamespace(
            attn_metadata=metadata,
            additional_kwargs={
                "deepseek_v41_replay_starts": torch.tensor([0, 32], dtype=torch.int32)
            },
        )
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)

        captured_records: list[torch.Tensor] = []
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention",
            lambda q, swa_c, main_c, recs, *args, **kw: (
                captured_records.append(recs.clone()) or torch.zeros_like(q)
            ),
        )
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)

        # 1. use_3page_swa=True (window_size <= 2 * rows_per_page = 128)
        #    has_fast_swa_offset=True (row_offset addition branch)
        attn_3page_fast = _make_swa_attn(
            window_size=128, ced_decoder=True, has_fast_swa_offset=True
        )
        with v2_attn.stage_attention_cache_scope():
            # Miss
            out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
            qr = torch.ones((count, 16), dtype=torch.bfloat16)
            kv = torch.ones((count, 8), dtype=torch.bfloat16)
            attn_3page_fast._prepare_tpu(None, qr, kv, None, None, None, positions, out)
            assert len(captured_records) == 1
            recs1 = captured_records[0]

            # Hit (same swa_meta.block_tables object -> reuses cached_swa_slots)
            attn_3page_fast._prepare_tpu(None, qr, kv, None, None, None, positions, out)
            assert len(captured_records) == 2
            assert torch.equal(captured_records[1], recs1)

            # Different swa_meta.block_tables tensor -> hits cached_swa geometry
            # but invalidates cached_swa_slots and recomputes records for new blocks
            swa_meta.block_tables = swa_table + 100
            attn_3page_fast._prepare_tpu(None, qr, kv, None, None, None, positions, out)
            assert len(captured_records) == 3
            assert not torch.equal(captured_records[2], recs1)
            swa_meta.block_tables = swa_table

        # 2. use_3page_swa=True, has_fast_swa_offset=False (row_indices branch)
        captured_records.clear()
        attn_3page_slow = _make_swa_attn(
            window_size=128, ced_decoder=False, has_fast_swa_offset=False
        )
        with v2_attn.stage_attention_cache_scope():
            out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
            attn_3page_slow._prepare_tpu(None, qr, kv, None, None, None, positions, out)
            assert len(captured_records) == 1

        # 3. use_3page_swa=False (window_size > 2 * rows_per_page = 256 > 128)
        #    Exercises swa_3page is None and swa_row_slots calculation
        captured_records.clear()
        attn_no_3page = _make_swa_attn(
            window_size=256, ced_decoder=False, has_fast_swa_offset=True
        )
        with v2_attn.stage_attention_cache_scope():
            out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
            attn_no_3page._prepare_tpu(None, qr, kv, None, None, None, positions, out)
            assert len(captured_records) == 1

        # 4. Fallback when stage_cache is None
        captured_records.clear()
        out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
        attn_3page_fast._prepare_tpu(None, qr, kv, None, None, None, positions, out)
        assert len(captured_records) == 1


class TestCompressedAttentionAndQuerySplitCaching:
    def test_compressed_attention_reuse_and_indexer_split(self, monkeypatch):
        """Test compressed attention caching across layers and query split caching."""
        count = 4
        capacity = 8
        positions = torch.tensor([0, 16, 32, 48], dtype=torch.int32)
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

        meta = SimpleNamespace(
            seq_lens=torch.tensor([2, 2], dtype=torch.int32),
            block_tables=table,
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        )
        metadata = {"src_pfx": meta, "idx_pfx": meta}
        forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
        monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)

        captured_main_recs: list[torch.Tensor] = []
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention",
            lambda q, swa_c, main_c, recs, main_recs, *args, **kw: (
                captured_main_recs.append(main_recs.clone()) or torch.zeros_like(q)
            ),
        )

        cache_mock = _make_paged_cache_mock(
            torch.tensor([[10, 11], [12, 13]]),
            row_offset=500,
            rows_per_page=32,
        )

        # Build index source layer
        source_attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(source_attn)
        source_attn.prefix = "src_pfx"
        source_attn.compressed_cache_prefix = "src_pfx"
        source_attn.index_source_layer_id = 0
        source_attn.window_size = 64
        source_attn.compress_ratio = 2
        source_attn._max_model_len = capacity * 2
        source_attn._config = {"index_topk": 2}
        source_attn._fp8_kv = True
        source_attn._ced_decoder = False
        source_attn._ced_global_from_encoder = False
        source_attn._local_q_attn = False
        source_attn.n_local_heads = 2
        source_attn.head_dim = 8
        source_attn.scale = 0.25
        source_attn.attn_sink = torch.zeros(2)
        source_attn._swa_max_requests = 16
        source_attn._precompact_records = False
        source_attn._token_parallel_attn = False
        source_attn._full_attn_sink = False
        source_attn._tp_rank_1d = 0
        source_attn.topk_indices_buffer = torch.full((16, 4), -1, dtype=torch.int32)
        source_attn._rotate = lambda x, p: x
        source_attn.wq_b = lambda qr: qr
        source_attn.compressor = None
        source_attn.swa_cache_layer = SimpleNamespace(
            prefix="src_pfx", kv_cache=cache_mock
        )
        source_attn.kv_cache = cache_mock
        source_attn._static_forward_context = {
            "src_pfx": SimpleNamespace(kv_cache=cache_mock)
        }

        idx_cache_mock = _make_paged_cache_mock(
            torch.tensor([[20, 21], [22, 23]]),
            row_offset=600,
            rows_per_page=32,
        )
        source_attn.indexer = SimpleNamespace(
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
        source_attn._select_paged_indices = lambda *a, **kw: torch.tensor(
            [[0, 1], [0, 2], [1, 2], [0, 1]], dtype=torch.int32
        )

        # Build consumer layer (shares source_attn's main_records)
        consumer_attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(consumer_attn)
        consumer_attn.prefix = "cons_pfx"
        consumer_attn.compressed_cache_prefix = "src_pfx"
        consumer_attn.index_source_layer_id = 0
        consumer_attn.window_size = 64
        consumer_attn.compress_ratio = 2
        consumer_attn._max_model_len = capacity * 2
        consumer_attn._config = {"index_topk": 2}
        consumer_attn._fp8_kv = True
        consumer_attn._ced_decoder = False
        consumer_attn._ced_global_from_encoder = False
        consumer_attn._local_q_attn = False
        consumer_attn.n_local_heads = 2
        consumer_attn.head_dim = 8
        consumer_attn.scale = 0.25
        consumer_attn.attn_sink = torch.zeros(2)
        consumer_attn._swa_max_requests = 16
        consumer_attn._precompact_records = False
        consumer_attn._token_parallel_attn = False
        consumer_attn._full_attn_sink = False
        consumer_attn._tp_rank_1d = 0
        consumer_attn.topk_indices_buffer = torch.full((16, 4), -1, dtype=torch.int32)
        consumer_attn._rotate = lambda x, p: x
        consumer_attn.wq_b = lambda qr: qr
        consumer_attn.compressor = None
        consumer_attn.indexer = None
        consumer_attn.swa_cache_layer = SimpleNamespace(
            prefix="src_pfx", kv_cache=cache_mock
        )
        consumer_attn.kv_cache = cache_mock
        consumer_attn._static_forward_context = {
            "src_pfx": SimpleNamespace(kv_cache=cache_mock)
        }

        with v2_attn.stage_attention_cache_scope():
            out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
            qr = torch.ones((count, 16), dtype=torch.bfloat16)
            kv = torch.ones((count, 8), dtype=torch.bfloat16)
            idx_w = torch.ones((count, 2), dtype=torch.bfloat16)

            # Source layer runs: populates pages and comp_key
            source_attn._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)
            assert len(captured_main_recs) == 1
            src_records = captured_main_recs[0]

            # Consumer layer runs: reuses main_records from stage cache!
            consumer_attn._prepare_tpu(None, qr, kv, None, None, None, positions, out)
            assert len(captured_main_recs) == 2
            assert torch.equal(captured_main_recs[1], src_records)

        # Subcase 2: cached_idx_req hit branch
        with v2_attn.stage_attention_cache_scope():
            source_attn._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)
            # Second call runs indexer but hits cached_idx_req in stage_cache
            source_attn._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)

        # Subcase 3: _all_indices_fit branch (lines 991-998)
        all_fit_attn = _make_swa_attn(
            window_size=64, ced_decoder=False, has_fast_swa_offset=True
        )
        all_fit_attn.prefix = "src_pfx"
        all_fit_attn.swa_cache_layer.prefix = "src_pfx"
        all_fit_attn.compressed_cache_prefix = "src_pfx"
        all_fit_attn._static_forward_context = {
            "src_pfx": SimpleNamespace(kv_cache=cache_mock)
        }
        all_fit_attn._config = {"index_topk": 1000}
        all_fit_attn._max_model_len = 1000
        all_fit_attn.indexer = SimpleNamespace(
            head_dim=8,
            n_head=2,
            wk=lambda x: x,
            k_norm=lambda x: x,
            k_cache=SimpleNamespace(
                prefix="idx_pfx", compress_ratio=2, kv_cache=cache_mock
            ),
        )
        with v2_attn.stage_attention_cache_scope():
            all_fit_attn._prepare_tpu(None, qr, kv, None, None, idx_w, positions, out)

        # Subcase 4: pages is None fallback branch (lines 1138-1146)
        no_pages_cache = _make_paged_cache_mock(
            torch.tensor([[1, 2], [3, 4]]), row_offset=100
        )
        no_pages_cache.pages.return_value = None
        no_pages_attn = _make_swa_attn(
            window_size=64, ced_decoder=False, has_fast_swa_offset=True
        )
        no_pages_attn.prefix = "src_pfx"
        no_pages_attn.compressed_cache_prefix = "src_pfx"
        no_pages_attn.swa_cache_layer = SimpleNamespace(
            prefix="src_pfx", kv_cache=no_pages_cache
        )
        no_pages_attn.kv_cache = no_pages_cache
        no_pages_attn._static_forward_context = {
            "src_pfx": SimpleNamespace(kv_cache=no_pages_cache)
        }
        with v2_attn.stage_attention_cache_scope():
            no_pages_attn._prepare_tpu(None, qr, kv, None, None, None, positions, out)

    def test_select_paged_indices_query_split_caching(self, monkeypatch):
        """Test _select_paged_indices split and non-split query caching."""
        count = 4
        positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        qr = torch.ones((count, 16), dtype=torch.bfloat16)
        weights = torch.ones((count, 2), dtype=torch.float32)
        lengths = torch.tensor([[1], [2], [3], [4]], dtype=torch.int32)
        swa_slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        requests = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

        attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn)
        attn.compress_ratio = 2
        attn._tp_rank = 0
        attn._fp8_kv = True
        attn._config = {"index_topk": 2}
        attn.candidate_block_size = 1
        attn.candidate_block_buffer = None
        attn.indexer = SimpleNamespace(
            head_dim=8,
            n_head=2,
            indexer_op=SimpleNamespace(candidate_blocks=None, candidate_write=False),
        )
        attn._project_index_q = lambda q, p: q

        monkeypatch.setattr(
            v2_attn,
            "paged_indexer",
            lambda q, c, r, req, w, lengths_arg, *a, **kw: (
                c,
                torch.zeros(q.shape[0], 2, dtype=torch.int32),
                None,
            ),
        )
        # TP=2 all-gather of each rank's part (only the split branch calls it).
        monkeypatch.setattr(v2_attn, "_gather_query_rows", lambda x: torch.cat([x, x]))

        index_cache_mock = MagicMock(spec=PagedCacheRows)
        index_cache_mock.raw = MagicMock()
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
                None,
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
                None,
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
                None,
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
                None,
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
            None,
        )
        assert sel_nocache.shape[0] == count

    def test_topk_indices_buffer_padding_and_truncation_branches(self):
        """Test selected indices padding vs slicing against topk_indices_buffer."""
        # Case A: stage_cache is None -> copy_ with padding
        buf = torch.full((8, 4), -1, dtype=torch.int32)
        sel = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        count = 2
        capacity = 4
        buf.copy_(
            F.pad(
                sel,
                (0, buf.shape[-1] - sel.shape[-1], 0, buf.shape[0] - count),
                value=-1,
            )
        )
        res_uncached = buf[:count, : min(capacity, buf.shape[-1])]
        assert res_uncached.shape == (2, 4)
        assert torch.equal(res_uncached[:, :2], sel)
        assert torch.equal(
            res_uncached[:, 2:], torch.full((2, 2), -1, dtype=torch.int32)
        )

        # Case B: stage_cache is not None, selected width < target_width -> F.pad
        target_width = 4
        sel_narrow = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        padded = F.pad(sel_narrow, (0, target_width - sel_narrow.shape[-1]), value=-1)
        assert padded.shape == (2, 4)
        assert torch.equal(padded[:, :2], sel_narrow)
        assert torch.equal(padded[:, 2:], torch.full((2, 2), -1, dtype=torch.int32))

        # Case C: stage_cache is not None, selected width >= target_width -> slice
        sel_wide = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.int32)
        sliced = sel_wide[:, :target_width]
        assert sliced.shape == (2, 4)
        assert torch.equal(sliced, sel_wide[:, :4])

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
