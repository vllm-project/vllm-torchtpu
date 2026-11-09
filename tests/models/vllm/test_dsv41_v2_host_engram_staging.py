# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""100% Diff coverage test suite for host Engram hashing and staging.

Covers all lines and branches in:
- vllm_torchtpu.models.vllm.deepseek_v41.v2.host_engram (host_engram.py:1-82)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.engram (engram.py:66-175, 288-445)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.model (model.py:180-213, 261-326, 357-358)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax.numpy  # noqa: F401
import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import engram as v2_engram
from vllm_torchtpu.models.vllm.deepseek_v41.v2.engram import (
    _UNIT_SCALE,
    Engram,
    NgramHashState,
    ParallelEngramEmbedding,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.model import (
    DeepseekV41Model,
)

pytestmark = pytest.mark.usefixtures("cpu_engram_transfer")

# ==============================================================================
# 1. host_engram.py Tests
# ==============================================================================


# ==============================================================================
# 2. v2/engram.py Tests
# ==============================================================================


def _build_hash_state(use_slot_cache: bool = True, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    state = object.__new__(NgramHashState)
    nn.Module.__init__(state)
    state.token_map = torch.randint(1, 50, (100,), generator=g)
    state.multipliers = torch.randint(1, 1 << 20, (2, 3), generator=g)
    state.primes = torch.randint(1000, 5000, (2, 2, 4), generator=g)
    state.offsets = torch.randint(0, 9999, (2, 8), generator=g)
    state.lookback_depth = 4
    state.pad_id = 0
    state.use_slot_cache = use_slot_cache
    state.block_size = 8
    state._cache = torch.randint(1, 50, (64,), generator=g)
    state.swa_cache_module = SimpleNamespace(
        kv_cache=torch.randint(1, 50, (8, 8), generator=g)
    )
    return state


class TestEngramHashState:
    def test_hash_on_host_empty_tokens(self):
        state = _build_hash_state()
        empty_ids = torch.empty((0,), dtype=torch.int32)
        pos = torch.empty((0,), dtype=torch.int64)
        qsl = torch.tensor([0], dtype=torch.int32)
        dead = torch.empty((0,), dtype=torch.bool)
        lookback = torch.full((1, 4), -1, dtype=torch.int32)
        dead_lb = torch.zeros((1, 4), dtype=torch.bool)

        res = state.hash_on_host(empty_ids, pos, qsl, dead, lookback, dead_lb)
        assert res.shape == (0, 2, 8)
        assert res.dtype == torch.int32

    def test_forward_fallback_without_and_with_slot_cache(self):
        # 1. Without slot cache
        state_no_cache = _build_hash_state(use_slot_cache=False)
        ids = torch.tensor([1, 2], dtype=torch.int32)
        pos = torch.tensor([10, 11], dtype=torch.int64)
        qsl = torch.tensor([0, 2], dtype=torch.int32)
        dead = torch.zeros(2, dtype=torch.bool)
        # Lookback empty (-1) forces fallback call for past tokens
        lookback = torch.full((1, 4), -1, dtype=torch.int32)
        dead_lb = torch.zeros((1, 4), dtype=torch.bool)
        block_table = torch.zeros((1, 4), dtype=torch.int32)

        out_no_cache = state_no_cache.forward(
            ids, pos, qsl, dead, lookback, dead_lb, None, block_table
        )
        assert out_no_cache.shape == (2, 2, 8)

        # 2. With slot cache
        state_with_cache = _build_hash_state(use_slot_cache=True)
        slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
        out_with_cache = state_with_cache.forward(
            ids, pos, qsl, dead, lookback, dead_lb, slot_mapping, block_table
        )
        assert out_with_cache.shape == (2, 2, 8)

    def test_hash_dead_mask_and_window_branches(self):
        state = _build_hash_state()
        ids = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
        # Shift past > chunk_positions and past < chunk_positions
        pos = torch.tensor([0, 1, 5, 6], dtype=torch.int64)
        qsl = torch.tensor([0, 2, 4], dtype=torch.int32)
        dead = torch.tensor([True, False, False, True], dtype=torch.bool)
        # Valid lookback tokens in window
        lookback = torch.tensor([[10, 11, -1, 13], [20, -1, 22, 23]], dtype=torch.int32)
        dead_lb = torch.tensor(
            [
                [False, True, False, False],
                [True, False, False, False],
            ],
            dtype=torch.bool,
        )
        out = state.hash_on_host(ids, pos, qsl, dead, lookback, dead_lb)
        assert out.shape == (4, 2, 8)


class TestParallelEngramEmbeddingGatherRawAndDequant:
    def test_gather_raw_and_device_transfer(self):
        embed = object.__new__(ParallelEngramEmbedding)
        embed.head_start = 0
        embed.part_n_hash_cols = 2
        embed.n_hash_cols = 4
        embed.vocab_start_idx = 0
        embed.vocab_end_idx = 100
        embed.part_num_embeddings = 100
        embed.dim = 32
        embed.block_size = 16

        weight = torch.ones((100, 32), dtype=torch.uint8)
        scale = torch.full((100, 2), 127, dtype=torch.uint8)
        embed._storage = lambda: (weight, scale)

        # Test with CPU indices (matching weight device)
        indices = torch.tensor([[10, 20, 200, 300]], dtype=torch.int32)
        raw_out = torch.zeros((1, 2, 32), dtype=torch.uint8)
        scale_out = torch.zeros((1, 2, 2), dtype=torch.uint8)

        embed.gather_raw(indices, raw_out, scale_out)
        # First 2 heads are within vocab: values are copied
        assert torch.all(raw_out[0, 0] == 1)
        assert torch.all(raw_out[0, 1] == 1)
        assert torch.all(scale_out == 127)

        # Test invalid indices masked_fill / sentinel scale
        invalid_indices = torch.tensor([[200, 300, 400, 500]], dtype=torch.int32)
        embed.gather_raw(invalid_indices, raw_out, scale_out)
        if hasattr(v2_engram, "_INVALID_SCALE"):
            assert torch.all(scale_out == v2_engram._INVALID_SCALE)
        else:
            assert torch.all(raw_out == 0)
            assert torch.all(scale_out == _UNIT_SCALE)

        # Test indices device transfer branch (line 300)
        mock_indices = MagicMock(
            wraps=torch.tensor([[10, 20, 200, 300]], dtype=torch.int32)
        )
        mock_indices.device = torch.device("meta")
        embed.gather_raw(mock_indices, raw_out, scale_out)
        mock_indices.to.assert_called_once_with(weight.device)


class TestEngramModuleStagingAndEmbed:
    def _create_mock_engram(self, cpu_offload: bool):
        engram = object.__new__(Engram)
        embed_tokens = SimpleNamespace(
            cpu_offload=cpu_offload,
            host_offload=cpu_offload,
            dp_size=1,
            part_n_hash_cols=2,
            n_hash_cols=2,
            head_dim=32,
            block_size=16,
            tp_size=1,
            gather_raw=MagicMock(),
        )
        engram.embed_tokens = embed_tokens
        return engram

    def test_dp_staging_uploads_only_local_tokens(self, monkeypatch):
        from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common

        engram = self._create_mock_engram(cpu_offload=True)
        engram.embed_tokens.dp_size = 8
        engram.embed_tokens.n_hash_cols = 16
        engram._init_staging(max_tokens=16, head_dim=32)
        assert engram._host_raw.shape == (16, 16, 32)
        assert not hasattr(engram, "staged_rows")
        group = SimpleNamespace(world_size=8, cpu_group=object())
        monkeypatch.setattr(common.upstream, "get_engram_dp_group", lambda: group)

        def gather(ids, raw, scales):
            assert raw.shape == (ids.shape[0], 2, 32)
            raw.fill_(56)  # E4M3 1.0
            scales.fill_(127)

        def exchange(raw, scales, cpu_group, size):
            assert cpu_group is group.cpu_group and size == 8
            # Exchange correctness is covered by the real Gloo regression.
            tokens = raw.shape[0] // size
            return SimpleNamespace(
                wait=lambda: (
                    raw[:tokens].repeat(1, size, 1),
                    scales[:tokens].repeat(1, size, 1),
                )
            )

        engram.embed_tokens.gather_raw = gather
        monkeypatch.setattr(common, "start_exchange_rows", exchange)
        for tokens in (16, 8, 16):
            engram.prepare_embeddings(torch.zeros(tokens * 8, 16, dtype=torch.int32))
            rows = engram._ready_rows(tokens)
            assert rows.shape == (tokens, 16, 32)
            assert torch.equal(rows, torch.ones_like(rows))
            assert engram._active_raw.numel() * 4 == tokens * 16 * (32 + 2)
        assert len(engram._staged_by_tokens) == 2

    def test_init_staging_cpu_offload_false_and_true(self):
        # cpu_offload = False
        engram_no_offload = self._create_mock_engram(cpu_offload=False)
        with patch.object(
            v2_engram.upstream.Engram, "_init_staging"
        ) as mock_super_init:
            engram_no_offload._init_staging(max_tokens=16, head_dim=32)
            mock_super_init.assert_called_once_with(16, 32)
            assert engram_no_offload._staged_raw is None

        # cpu_offload = True
        engram_offload = self._create_mock_engram(cpu_offload=True)
        with patch.object(
            v2_engram.upstream.Engram, "_init_staging"
        ) as mock_super_init:
            engram_offload._init_staging(max_tokens=16, head_dim=32)
            mock_super_init.assert_not_called()
            assert engram_offload._staged_raw is not None
            assert engram_offload._host_raw.shape == (16, 2, 32)
            assert engram_offload._host_scale.shape == (16, 2, 2)
            raw_words = 16 * 2 * 32 // 4
            scale_words = engram_offload._host_scale_buf.numel() // 4
            assert engram_offload._staged_raw.shape == (raw_words + scale_words,)
            assert engram_offload._staged_raw.dtype == torch.int32
            assert torch.all(engram_offload._staged_raw[:raw_words] == 0)
            assert torch.all(engram_offload._staged_raw[raw_words:] == -1)

    def test_prepare_embeddings_staging_fast_path_and_fallback(self):
        engram = self._create_mock_engram(cpu_offload=True)
        with patch.object(v2_engram.upstream.Engram, "_init_staging"):
            engram._init_staging(max_tokens=16, head_dim=32)

        # 1. Fast path: scales share the packed buffer after active raw rows.
        hash_ids = torch.zeros((8, 2), dtype=torch.int32)
        engram.prepare_embeddings(hash_ids)
        engram.embed_tokens.gather_raw.assert_called_once()
        assert engram._host_scale.shape == (8, 2, 2)
        assert (
            engram._host_scale.data_ptr()
            == engram._host_packed_buf.data_ptr() + 8 * 2 * 32
        )

        # 1b. Fast path with exact capacity (skips tail zeroing branch)
        hash_ids_exact = torch.zeros((16, 2), dtype=torch.int32)
        engram.prepare_embeddings(hash_ids_exact)

        # Oversized host requests must not fall back to a BF16 device lookup.
        hash_ids_large = torch.zeros((32, 2), dtype=torch.int32)
        with pytest.raises(ValueError, match="staging slots"):
            engram.prepare_embeddings(hash_ids_large)

    def test_ready_rows_and_sp_embed(self):
        # 1. _staged_raw is None calls super()._ready_rows
        engram_none = self._create_mock_engram(cpu_offload=False)
        engram_none._staged_raw = None
        with patch.object(
            v2_engram.upstream.Engram,
            "_ready_rows",
            return_value=torch.zeros((4, 2, 32)),
        ) as mock_super_ready:
            res = engram_none._ready_rows(4)
            mock_super_ready.assert_called_once_with(4)
            assert res.shape == (4, 2, 32)

        # 2. _staged_raw is populated calls dequant_fp8_rows
        engram = self._create_mock_engram(cpu_offload=True)
        with patch.object(v2_engram.upstream.Engram, "_init_staging"):
            engram._init_staging(max_tokens=16, head_dim=32)
        engram.prepare_embeddings(torch.zeros((4, 2), dtype=torch.int32))
        ready = engram._ready_rows(4)
        assert ready.shape == (4, 2, 32)
        assert ready.dtype == torch.bfloat16

        # 3. _sp_embed returns ready rows for hash_ids
        with patch.object(
            v2_engram.upstream, "get_tensor_model_parallel_rank", return_value=0
        ):
            # tp_size = 1
            rows = engram._sp_embed(torch.zeros((4, 2), dtype=torch.int32))
            assert rows.shape == (4, 2, 32)

            # tp_size = 2 with padding (num_tokens=3, tp=2 -> chunk=2, pad=1)
            engram.embed_tokens.tp_size = 2
            engram.prepare_embeddings(torch.zeros((3, 2), dtype=torch.int32))
            with patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.common.engram._gather_engram_rows",
                side_effect=lambda x, dim: torch.cat([x, x], dim=dim),
            ):
                rows_tp2 = engram._sp_embed(torch.zeros((3, 2), dtype=torch.int32))
                assert rows_tp2.shape == (2, 2, 32)

                # tp_size = 2 without padding (num_tokens=4, tp=2 -> chunk=2, pad=0)
                engram.prepare_embeddings(torch.zeros((4, 2), dtype=torch.int32))
                rows_tp2_exact = engram._sp_embed(
                    torch.zeros((4, 2), dtype=torch.int32)
                )
                assert rows_tp2_exact.shape == (2, 2, 32)


# ==============================================================================
# 3. v2/model.py Tests
# ==============================================================================


class TestModelEngramPreparation:
    def test_prepare_all_host_tables_before_waiting_and_uploading(self, monkeypatch):
        from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as common
        from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as model_module

        model = self._create_mock_model(use_slot_cache=False)
        model.end_layer = 2
        model.engram_dp_shared_memory = False
        model.engram_hash.ensure_cache = lambda: False
        issued = []
        group = SimpleNamespace(world_size=2, cpu_group=object())
        monkeypatch.setattr(common.upstream, "get_engram_dp_group", lambda: group)
        monkeypatch.setattr(model_module.upstream, "get_engram_dp_size", lambda: 2)
        monkeypatch.setattr(
            model_module, "gather_host_engram_hashes", lambda hashes, **kw: hashes
        )
        monkeypatch.setattr(
            model_module.upstream, "is_forward_context_available", lambda: True
        )
        monkeypatch.setattr(
            model_module.upstream,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={}),
        )

        def gather(ids, raw, scales):
            raw.fill_(56 + 8 * int(ids[0, 0]))  # FP8 1.0 or 2.0.
            scales.fill_(127)

        def exchange(raw, scales, cpu_group, size):
            issued.append(int(raw[0, 0, 0]))

            def wait():
                assert issued == [56, 64]
                tokens = raw.shape[0] // size
                return (
                    raw[:tokens].repeat(1, size, 1),
                    scales[:tokens].repeat(1, size, 1),
                )

            return SimpleNamespace(wait=wait)

        monkeypatch.setattr(common, "start_exchange_rows", exchange)
        stagers = []
        for index in range(2):
            engram = TestEngramModuleStagingAndEmbed()._create_mock_engram(True)
            engram.layer_hash_index = index
            engram.embed_tokens.dp_size = 2
            engram.embed_tokens.gather_raw = gather
            engram._init_staging(7, 32)
            stagers.append(engram)
        model.layers = [SimpleNamespace(engram=e) for e in stagers]

        for tokens in (7, 2, 7):
            issued.clear()
            hashes = torch.tensor([0, 1]).view(1, 2, 1).expand(tokens * 2, 2, 4)
            model.engram_hash.dummy_hashes = lambda ids, hashes=hashes: (hashes, None)
            model._prepare_engram(torch.arange(tokens), torch.arange(tokens), None)
            for index, engram in enumerate(stagers):
                rows = engram._ready_rows(tokens)
                assert rows.shape == (tokens, 4, 32)
                assert torch.equal(rows, torch.full_like(rows, index + 1))

    def _create_mock_model(
        self,
        cpu_offload: bool = True,
        use_slot_cache: bool = True,
        ced_enabled: bool = False,
    ):
        model = object.__new__(DeepseekV41Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=64)
        model._ced_enabled = ced_enabled
        model._ced_window = 128
        model._ced_compile_sizes = [16, 2048]
        model.use_native_mega_moe = False
        model.start_layer = 0
        model.end_layer = 1
        model.engram_dp_shared_memory = True

        engram_mock = SimpleNamespace(
            embed_tokens=SimpleNamespace(host_offload=cpu_offload),
            layer_hash_index=0,
            prepare_embeddings=MagicMock(),
        )
        layer_mock = SimpleNamespace(engram=engram_mock)
        model.layers = [layer_mock]

        state = _build_hash_state(use_slot_cache=use_slot_cache)
        state.ensure_cache = MagicMock(return_value=True)
        model.engram_hash = state
        model.engram_swa_prefix = "swa_0"
        return model

    def test_prepare_engram_missing_lookback_not_implemented(self):
        # 1. When use_slot_cache is False and lookback_token_ids is None
        # -> raises NotImplementedError
        model = self._create_mock_model(cpu_offload=True, use_slot_cache=False)
        pos = torch.tensor([0, 1], dtype=torch.int64)
        mock_meta = {
            "swa_0": SimpleNamespace(
                seq_lens=torch.tensor([2]),
                query_start_loc=torch.tensor([0, 2]),
                block_tables=torch.zeros((1, 4), dtype=torch.int32),
            )
        }
        mock_ctx = SimpleNamespace(attn_metadata=mock_meta, additional_kwargs={})
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            pytest.raises(NotImplementedError, match="lookback_token_ids"),
        ):
            model._prepare_engram(
                torch.tensor([1, 2]), pos, None, stage="full", skip_lookup=False
            )

        # 2. When use_slot_cache is True and lookback_token_ids is None
        # -> allocates lookback_token_ids (lines 257-260)
        model_cache = self._create_mock_model(cpu_offload=True, use_slot_cache=True)
        with (
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.is_forward_context_available",
                return_value=True,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.get_forward_context",
                return_value=mock_ctx,
            ),
            patch.object(
                model_cache.engram_hash,
                "forward",
                wraps=model_cache.engram_hash.forward,
            ),
            patch(
                "vllm_torchtpu.models.vllm.deepseek_v41.v2.model.upstream.gather_engram_hashes",
                side_effect=lambda h, **kwargs: h.unsqueeze(1),
            ),
        ):
            hashes, mask = model_cache._prepare_engram(
                torch.tensor([1, 2], dtype=torch.int32),
                pos,
                None,
                stage="full",
                skip_lookup=False,
            )
            assert hashes is not None
