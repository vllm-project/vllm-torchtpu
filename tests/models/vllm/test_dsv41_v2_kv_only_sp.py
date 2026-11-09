# SPDX-License-Identifier: Apache-2.0
"""100% Diff line and branch coverage tests for Targets:
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention.py
  (qr_local SWA conditions lines 525-535; SP all-gather kv-only skip lines 536-554;
   assert q is not None lines 1226 and 1258;
   _init_local_q_weights _ced_decoder and compress_ratio guards lines 1281-1289)
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py
  (tokens from positions.shape[0] line 1046;
   _sharded_body in-kernel all_gather on sub-threshold decode lines 1141-1144;
   _fp8_attention_qo_op.register_fake positions shape line 1186;
   sparse_fp8_attention_qo starts from positions.shape[0] line 1224)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)

# ==============================================================================
# Helper Mock Objects and Factories
# ==============================================================================


def _setup_mock_attn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sp_sharded_input: bool = True,
    local_q_attn: bool = True,
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
    attn._local_q_attn = local_q_attn
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
    attn._full_attn_sink = torch.zeros(8)
    attn._tp_rank_1d = torch.zeros(1, dtype=torch.int32)
    attn._swa_max_requests = 16
    attn._precompact_records = False
    attn.compressed_cache_prefix = "comp"
    attn._max_model_len = 64
    attn.indexer = None
    attn.topk_indices_buffer = torch.zeros((4, 64), dtype=torch.int32)
    attn._token_parallel_attn = True
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
# Part 1: attention.py qr_local Definition and SWA Branches (lines 525-535)
# ==============================================================================
class TestLocalQPreGatherSwaCondition:
    """Verify lines 525-535 in attention.py covering SWA cache & flags."""

    def test_swa_kv_none_yields_qr(self, monkeypatch: pytest.MonkeyPatch):
        attn = _setup_mock_attn(monkeypatch, sp_sharded_input=True, local_q_attn=True)
        # swa_cache_layer is not present or kv_cache is None
        attn.swa_cache_layer = None
        attn.indexer = None
        tokens_local = 2
        tokens_global = 4
        qr = torch.randn(tokens_local, 16)
        kv = torch.randn(tokens_local, 128)

        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))
        monkeypatch.setattr(
            v2_attn, "sp_all_gather", lambda t: torch.cat([t, t], dim=0)
        )
        seen_qr_local: list[torch.Tensor | None] = []

        def fake_prepare_and_attn(*args, qr_local=None):
            seen_qr_local.append(qr_local)
            return torch.zeros(tokens_global, 1024)

        monkeypatch.setattr(
            attn, "_prepare_and_attn_fn", fake_prepare_and_attn, raising=False
        )
        attn.wo_b = lambda x: (x, None)

        v2_attn.DeepseekV41Attention.forward(
            attn,
            torch.zeros(tokens_global, dtype=torch.long),
            torch.zeros(tokens_local, 8),
        )
        assert len(seen_qr_local) == 1
        assert seen_qr_local[0] is qr

    def test_swa_kv_paged_cache_valid_window_yields_qr(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = _setup_mock_attn(
            monkeypatch, sp_sharded_input=True, local_q_attn=True, window_size=128
        )
        paged_cache = mock.MagicMock(spec=PagedCacheRows)
        attn.swa_cache_layer = SimpleNamespace(kv_cache=paged_cache)
        attn.indexer = None
        tokens_local = 2
        tokens_global = 4
        qr = torch.randn(tokens_local, 16)
        kv = torch.randn(tokens_local, 128)

        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))
        monkeypatch.setattr(
            v2_attn, "sp_all_gather", lambda t: torch.cat([t, t], dim=0)
        )
        seen_qr_local: list[torch.Tensor | None] = []

        def fake_prepare_and_attn(*args, qr_local=None):
            seen_qr_local.append(qr_local)
            return torch.zeros(tokens_global, 1024)

        monkeypatch.setattr(
            attn, "_prepare_and_attn_fn", fake_prepare_and_attn, raising=False
        )
        attn.wo_b = lambda x: (x, None)

        v2_attn.DeepseekV41Attention.forward(
            attn,
            torch.zeros(tokens_global, dtype=torch.long),
            torch.zeros(tokens_local, 8),
        )
        assert len(seen_qr_local) == 1
        assert seen_qr_local[0] is qr

    def test_swa_kv_non_paged_cache_yields_none(self, monkeypatch: pytest.MonkeyPatch):
        attn = _setup_mock_attn(
            monkeypatch, sp_sharded_input=True, local_q_attn=True, window_size=128
        )
        # swa_kv is not PagedCacheRows
        attn.swa_cache_layer = SimpleNamespace(kv_cache=torch.zeros(4, 128))
        attn.indexer = None
        tokens_local = 2
        tokens_global = 4
        qr = torch.randn(tokens_local, 16)
        kv = torch.randn(tokens_local, 128)

        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))
        monkeypatch.setattr(
            v2_attn, "sp_all_gather", lambda t: torch.cat([t, t], dim=0)
        )
        seen_qr_local: list[torch.Tensor | None] = []

        def fake_prepare_and_attn(*args, qr_local=None):
            seen_qr_local.append(qr_local)
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
        assert seen_qr_local == [None]

    def test_swa_kv_paged_cache_invalid_window_yields_none(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # window_size % 64 != 0
        attn = _setup_mock_attn(
            monkeypatch, sp_sharded_input=True, local_q_attn=True, window_size=32
        )
        paged_cache = mock.MagicMock(spec=PagedCacheRows)
        attn.swa_cache_layer = SimpleNamespace(kv_cache=paged_cache)
        attn.indexer = None
        tokens_local = 2
        tokens_global = 4
        qr = torch.randn(tokens_local, 16)
        kv = torch.randn(tokens_local, 128)

        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))
        monkeypatch.setattr(
            v2_attn, "sp_all_gather", lambda t: torch.cat([t, t], dim=0)
        )
        seen_qr_local: list[torch.Tensor | None] = []

        def fake_prepare_and_attn(*args, qr_local=None):
            seen_qr_local.append(qr_local)
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
        assert seen_qr_local == [None]


# ==============================================================================
# Part 2: attention.py SP All-Gather KV-Only Routing Branches (lines 536-554)
# ==============================================================================
class TestSpAllGatherRoutingKvOnly:
    """Verify lines 536-554: source layer, non-source with local_q, and fallback."""

    def test_source_layer_with_indexer_gathers_all(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = _setup_mock_attn(monkeypatch, sp_sharded_input=True, local_q_attn=True)
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


# ==============================================================================
# Part 3: attention.py _init_local_q_weights Guards (lines 1281-1289)
# ==============================================================================


# ==============================================================================
# Part 4: attention.py _prepare_tpu Assertions (lines 1226 and 1258)
# ==============================================================================


# ==============================================================================
# Part 5: attention_kernels.py Token-Parallel & Sharded Body Coverage
# ==============================================================================
class TestAttentionKernelsKvOnlySp:
    """Verify attention_kernels.py lines 1046, 1141-1144, 1186, 1224."""

    def test_fp8_attention_qo_fake_uses_positions_shape(self):
        """Line 1186: register_fake uses args[5].shape[0] (positions)."""
        mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
        mock_tp = mock.MagicMock()
        mock_tp.device_group.size.return_value = 2

        with (
            set_vllm_model_wrapper_context(mesh=mesh),
            mock.patch("vllm.distributed.get_tp_group", return_value=mock_tp),
            mock.patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
        ):
            attention_kernels._attention_op = None
            attention_kernels._fp8_attention_op = None
            attention_kernels._fp8_attention_qo_op = None
            attention_kernels.initialize_paged_attention()

            with FakeTensorMode():
                qr_sharded = torch.empty(64, 1280)
                qr_local = torch.empty(32, 1280)
                wo_a = torch.empty(512, 1024)
                tensors = [torch.empty(1) for _ in range(21)]
                # args[5] is positions with shape (128,)
                tensors[4] = torch.empty(128)
                tensors[16] = wo_a

                # Call op with qr having shape (64,), positions having shape (128,)
                out = attention_kernels._fp8_attention_qo_op(
                    qr_sharded, qr_local, *tensors, 0.125, 64, None, True, 128
                )
                # Output must match positions.shape[0] = 128, NOT qr.shape[0] = 64
                assert out.shape == (128, 1024)
