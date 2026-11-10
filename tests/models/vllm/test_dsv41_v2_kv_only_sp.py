# SPDX-License-Identifier: Apache-2.0
"""CPU checks for DeepSeek-V4.1 attention under sequence-parallel input:
the SP all-gather of qr/kv/indexer weights and the untiled BF16 fallback."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows


# ==============================================================================
# Helper Mock Objects and Factories
# ==============================================================================
class _FakeForwardContext:
    def __init__(
        self,
        attn_metadata: dict | None = None,
        additional_kwargs: dict | None = None,
    ):
        self.attn_metadata = attn_metadata or {}
        self.additional_kwargs = additional_kwargs or {}

    def __getitem__(self, key: str):
        return self.attn_metadata[key]

    def get(self, key: str, default=None):
        return self.attn_metadata.get(key, default)


def _setup_mock_attn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sp_sharded_input: bool = True,
    ced_decoder: bool = False,
    fp8_kv: bool = True,
    compress_ratio: int = 4,
    window_size: int = 128,
    tp_size: int = 2,
    tiled: bool = True,
) -> v2_attn.DeepseekV41Attention:
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn._sp_sharded_input = sp_sharded_input
    attn._ced_decoder = ced_decoder
    attn._fp8_kv = fp8_kv
    attn.compress_ratio = compress_ratio
    attn.window_size = window_size
    attn.n_local_groups = 1
    attn.n_local_heads = 4
    attn.padded_heads = 4
    attn.head_dim = 128
    attn.q_lora_rank = 16
    attn.scale = 0.125
    attn._ced_global_from_encoder = False
    attn.compressor = None
    attn._tpu_rope_cache = torch.zeros(1)
    attn.attn_sink = torch.zeros(4)
    attn.compressed_cache_prefix = "comp"
    attn._max_model_len = 64
    attn.indexer = None
    attn.topk_indices_buffer = torch.zeros((4, 64), dtype=torch.int32)
    attn.max_position_embeddings = 4096

    wq = torch.zeros(16, 512, dtype=torch.float8_e4m3fn)
    wo = torch.zeros(512, 1024, dtype=torch.float8_e4m3fn)
    attn.wq_b = SimpleNamespace(weight=wq, weight_scale=torch.ones(512))
    attn.wo_a = SimpleNamespace(weight=wo, weight_scale=torch.ones(1024))

    monkeypatch.setattr(
        v2_attn, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(
        v2_attn,
        "tensor_model_parallel_all_gather",
        lambda t, dim=0: torch.cat([t] * tp_size, dim=dim),
    )

    if tiled:
        fake_cache = mock.MagicMock(spec=PagedCacheRows)
        fake_cache.numel.return_value = 100
        fake_cache.row_indices = lambda slots: slots.int()
        fake_cache.pages = lambda table: None
        attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=fake_cache)
        attn.main_cache_layer = SimpleNamespace(prefix="main", kv_cache=fake_cache)
        attn._static_forward_context = {"comp": SimpleNamespace(kv_cache=fake_cache)}
    else:
        raw_cache = torch.zeros(16, 64, 128)
        attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=raw_cache)
        attn.main_cache_layer = SimpleNamespace(prefix="main", kv_cache=raw_cache)
        attn._static_forward_context = {"comp": SimpleNamespace(kv_cache=raw_cache)}

    return attn


# ==============================================================================
# Part 2: attention.py SP All-Gather KV-Only Routing Branches (lines 536-554)
# ==============================================================================
class TestSpAllGatherRoutingKvOnly:
    """Verify lines 536-554: source layer, non-source with local_q, and fallback."""

    def test_source_layer_with_indexer_gathers_all(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = _setup_mock_attn(monkeypatch, sp_sharded_input=True)
        attn.swa_cache_layer = None
        attn.indexer = SimpleNamespace(n_head=8)
        tokens_local = 2
        tokens_global = 4
        qr = torch.zeros(tokens_local, 16)
        kv = torch.zeros(tokens_local, 128)
        indexer_weights = torch.zeros(tokens_local, 8)
        kv_score = torch.zeros(tokens_local, 1)

        monkeypatch.setattr(
            attn,
            "_run_parallel_input_projections",
            lambda h: (None, kv_score, indexer_weights),
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))

        gathered_tensors: list[torch.Tensor] = []

        def fake_sp_gather(t):
            gathered_tensors.append(t)
            return torch.cat([t, t], dim=0)

        monkeypatch.setattr(v2_attn, "sp_all_gather", fake_sp_gather)
        seen_args: dict[str, torch.Tensor] = {}

        def fake_prepare_and_attn(*args, qr_local=None):
            seen_args["qr"] = args[1]
            seen_args["kv"] = args[2]
            seen_args["kv_score"] = args[4]
            seen_args["indexer_weights"] = args[5]
            return None

        monkeypatch.setattr(
            attn, "_prepare_and_attn_fn", fake_prepare_and_attn, raising=False
        )
        monkeypatch.setattr(
            attn, "_o_proj", lambda o, p: torch.zeros(tokens_global, 4, 128)
        )

        v2_attn.DeepseekV41Attention.forward(
            attn,
            torch.zeros(tokens_global, dtype=torch.long),
            torch.zeros(tokens_local, 8),
        )

        assert len(gathered_tensors) == 2
        # Gathered concatenated (qr, kv, indexer_weights) = 16 + 128 + 8 = 152 cols
        assert gathered_tensors[0].shape == (tokens_local, 16 + 128 + 8)
        # Gathered kv_score
        assert gathered_tensors[1].shape == (tokens_local, 1)
        assert seen_args["qr"].shape == (tokens_global, 16)
        assert seen_args["kv"].shape == (tokens_global, 128)
        assert seen_args["kv_score"].shape == (tokens_global, 1)
        assert seen_args["indexer_weights"].shape == (tokens_global, 8)

    def test_non_source_layer_without_qr_local_gathers_qr_and_kv(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Fallback: non-source layer with local_q disabled gathers (qr, kv)."""
        attn = _setup_mock_attn(monkeypatch, sp_sharded_input=True)
        attn.swa_cache_layer = None
        attn.indexer = None
        tokens_local = 2
        tokens_global = 4
        qr = torch.zeros(tokens_local, 16)
        kv = torch.zeros(tokens_local, 128)

        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))

        gathered_tensors: list[torch.Tensor] = []

        def fake_sp_gather(t):
            gathered_tensors.append(t)
            return torch.cat([t, t], dim=0)

        monkeypatch.setattr(v2_attn, "sp_all_gather", fake_sp_gather)
        seen_args: dict[str, torch.Tensor] = {}

        def fake_prepare_and_attn(*args, qr_local=None):
            seen_args["qr"] = args[1]
            seen_args["kv"] = args[2]
            return None

        monkeypatch.setattr(
            attn, "_prepare_and_attn_fn", fake_prepare_and_attn, raising=False
        )
        monkeypatch.setattr(
            attn, "_o_proj", lambda o, p: torch.zeros(tokens_global, 4, 128)
        )

        v2_attn.DeepseekV41Attention.forward(
            attn,
            torch.zeros(tokens_global, dtype=torch.long),
            torch.zeros(tokens_local, 8),
        )

        assert len(gathered_tensors) == 1
        # Gathered concatenated (qr, kv) = 16 + 128 = 144 cols
        assert gathered_tensors[0].shape == (tokens_local, 16 + 128)
        assert seen_args["qr"].shape == (tokens_global, 16)
        assert seen_args["kv"].shape == (tokens_global, 128)
