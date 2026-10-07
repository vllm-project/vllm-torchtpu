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

import inspect
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


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

    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_LOCAL_Q", True)
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

    def test_sp_sharded_input_false_yields_none(self, monkeypatch: pytest.MonkeyPatch):
        attn = _setup_mock_attn(monkeypatch, sp_sharded_input=False, local_q_attn=True)
        tokens_local = 2
        tokens_global = 4
        qr = torch.randn(tokens_local, 16)
        kv = torch.randn(tokens_local, 128)

        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))
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

    def test_local_q_disabled_yields_none(self, monkeypatch: pytest.MonkeyPatch):
        attn = _setup_mock_attn(monkeypatch, sp_sharded_input=True, local_q_attn=False)
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

    def test_non_source_layer_with_qr_local_gathers_kv_only(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """core optimization: non-source layer skips qr in sp_all_gather."""
        attn = _setup_mock_attn(monkeypatch, sp_sharded_input=True, local_q_attn=True)
        attn.swa_cache_layer = None
        attn.indexer = None
        tokens_local = 2
        tokens_global = 4
        qr = torch.arange(tokens_local * 16, dtype=torch.float32).reshape(
            tokens_local, 16
        )
        kv = torch.arange(tokens_local * 128, dtype=torch.float32).reshape(
            tokens_local, 128
        )

        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))

        gathered_tensors: list[torch.Tensor] = []

        def fake_sp_gather(t):
            gathered_tensors.append(t)
            return torch.cat([t, t], dim=0)

        monkeypatch.setattr(v2_attn, "sp_all_gather", fake_sp_gather)
        seen_args: dict[str, torch.Tensor | None] = {}

        def fake_prepare_and_attn(*args, qr_local=None):
            seen_args["qr_local"] = qr_local
            seen_args["qr"] = args[1]
            seen_args["kv"] = args[2]
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

        assert len(gathered_tensors) == 1
        # CRITICAL: sp_all_gather gathered ONLY kv (128 cols), NOT qr (16 cols)!
        assert gathered_tensors[0].shape == (tokens_local, 128)
        # qr was not all-gathered on the host; passed directly
        assert seen_args["qr_local"] is qr
        assert seen_args["qr"] is qr
        assert seen_args["kv"].shape == (tokens_global, 128)

    def test_non_source_layer_without_qr_local_gathers_qr_and_kv(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Fallback: non-source layer with local_q disabled gathers (qr, kv)."""
        attn = _setup_mock_attn(monkeypatch, sp_sharded_input=True, local_q_attn=False)
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


# ==============================================================================
# Part 3: attention.py _init_local_q_weights Guards (lines 1281-1289)
# ==============================================================================
class TestInitLocalQWeightsCedDecoderAndCompressRatio:
    """Verify lines 1281-1289: _ced_decoder and (fp8_kv and compress_ratio)."""

    def test_ced_decoder_skips_allocation(self, monkeypatch: pytest.MonkeyPatch):
        """CED decoder layers L20..L39 skip allocating replicated full weights."""
        attn = _setup_mock_attn(
            monkeypatch, ced_decoder=True, fp8_kv=True, compress_ratio=4
        )
        v2_attn.DeepseekV41Attention._init_local_q_weights(attn)
        assert not hasattr(attn, "_wq_b_full")
        assert not hasattr(attn, "_wo_a_full")

    def test_non_fp8_kv_skips_allocation(self, monkeypatch: pytest.MonkeyPatch):
        attn = _setup_mock_attn(
            monkeypatch, ced_decoder=False, fp8_kv=False, compress_ratio=4
        )
        v2_attn.DeepseekV41Attention._init_local_q_weights(attn)
        assert not hasattr(attn, "_wq_b_full")
        assert not hasattr(attn, "_wo_a_full")

    def test_zero_compress_ratio_full_swa_skips_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Full-SWA layers L0, L1 (compress_ratio=0) skip allocating full weights."""
        attn = _setup_mock_attn(
            monkeypatch, ced_decoder=False, fp8_kv=True, compress_ratio=0
        )
        v2_attn.DeepseekV41Attention._init_local_q_weights(attn)
        assert not hasattr(attn, "_wq_b_full")
        assert not hasattr(attn, "_wo_a_full")

    def test_happy_path_allocates_full_weights(self, monkeypatch: pytest.MonkeyPatch):
        """Compressed encoder layers allocate full weights."""
        attn = _setup_mock_attn(
            monkeypatch, ced_decoder=False, fp8_kv=True, compress_ratio=4
        )
        v2_attn.DeepseekV41Attention._init_local_q_weights(attn)
        assert hasattr(attn, "_wq_b_full")
        assert hasattr(attn, "_wq_b_full_scale")
        assert hasattr(attn, "_wo_a_full")
        assert hasattr(attn, "_wo_a_full_scale")
        assert attn._local_q_attn is True

    def test_env_flag_disabled_skips_allocation(self, monkeypatch: pytest.MonkeyPatch):
        attn = _setup_mock_attn(
            monkeypatch, ced_decoder=False, fp8_kv=True, compress_ratio=4
        )
        monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_LOCAL_Q", False)
        v2_attn.DeepseekV41Attention._init_local_q_weights(attn)
        assert not hasattr(attn, "_wq_b_full")

    def test_tp_size_one_skips_allocation(self, monkeypatch: pytest.MonkeyPatch):
        attn = _setup_mock_attn(
            monkeypatch, ced_decoder=False, fp8_kv=True, compress_ratio=4, tp_size=1
        )
        v2_attn.DeepseekV41Attention._init_local_q_weights(attn)
        assert not hasattr(attn, "_wq_b_full")

    def test_n_local_groups_not_one_skips_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = _setup_mock_attn(
            monkeypatch, ced_decoder=False, fp8_kv=True, compress_ratio=4
        )
        attn.n_local_groups = 2
        v2_attn.DeepseekV41Attention._init_local_q_weights(attn)
        assert not hasattr(attn, "_wq_b_full")

    def test_missing_scale_or_vllm_layout_skips_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR

        attn = _setup_mock_attn(
            monkeypatch, ced_decoder=False, fp8_kv=True, compress_ratio=4
        )
        attn.wq_b.weight_scale = None
        v2_attn.DeepseekV41Attention._init_local_q_weights(attn)
        assert not hasattr(attn, "_wq_b_full")

        attn2 = _setup_mock_attn(
            monkeypatch, ced_decoder=False, fp8_kv=True, compress_ratio=4
        )
        setattr(attn2.wq_b, KEEP_VLLM_LAYOUT_ATTR, True)
        v2_attn.DeepseekV41Attention._init_local_q_weights(attn2)
        assert not hasattr(attn2, "_wq_b_full")


# ==============================================================================
# Part 4: attention.py _prepare_tpu Assertions (lines 1226 and 1258)
# ==============================================================================
class TestPrepareTpuAssertQNotNone:
    """Verify lines 1226/1258: assert q is not None (tiled/untiled non-fused)."""

    def test_assert_q_not_none_tiled_fused_qo_false(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = _setup_mock_attn(
            monkeypatch, local_q_attn=False, fp8_kv=True, compress_ratio=4, tiled=True
        )
        attn.wq_b = lambda x: (torch.zeros(4, 4 * 128), None)
        attn._sparse_fp8_attention_supported = lambda *args, **kwargs: True

        forward_ctx = _FakeForwardContext(
            {
                "swa": SimpleNamespace(
                    block_table=torch.zeros((1, 4), dtype=torch.int32),
                    slot_mapping=torch.zeros((4,), dtype=torch.int64),
                    query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
                    seq_lens=torch.empty(0),
                ),
                "comp": SimpleNamespace(
                    block_table=torch.zeros((1, 4), dtype=torch.int32),
                    query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
                ),
            }
        )
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
        monkeypatch.setattr(v2_attn, "_block_table", lambda swa: swa.block_table)
        monkeypatch.setattr(
            v2_attn, "_cache_slots", lambda cache, swa, pos: swa.slot_mapping
        )
        monkeypatch.setattr(
            v2_attn,
            "_token_requests",
            lambda meta, pos: torch.zeros(pos.shape[0], dtype=torch.long),
        )
        monkeypatch.setattr(
            v2_attn,
            "_row_slots",
            lambda cache, table, reqs, logical: torch.zeros_like(
                logical, dtype=torch.int32
            ),
        )
        monkeypatch.setattr(
            v2_attn,
            "_read_rows",
            lambda cache, table, reqs, logical: torch.zeros((4, 64, 128)),
        )
        monkeypatch.setattr(v2_attn, "_insert", lambda *args: None)
        attn._rotate = lambda q, pos: q
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention",
            lambda *args, **kwargs: torch.zeros(4, 4, 128),
        )

        qr = torch.randn(4, 16)
        kv = torch.randn(4, 128)
        positions = torch.zeros(4, dtype=torch.long)
        output = torch.zeros(4, 4, 128)

        res = v2_attn.DeepseekV41Attention._prepare_tpu(
            attn,
            hidden_states=None,
            qr=qr,
            kv=kv,
            qr_scale=None,
            kv_score=None,
            index_weights=None,
            positions=positions,
            output=output,
            qr_local=None,
        )
        assert res is None

    def test_assert_q_not_none_untiled_bf16_fused_qo_false(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = _setup_mock_attn(
            monkeypatch,
            local_q_attn=False,
            fp8_kv=False,
            compress_ratio=0,
            tiled=False,
            window_size=64,
        )
        attn.wq_b = lambda x: (torch.zeros(4, 4 * 128), None)

        forward_ctx = _FakeForwardContext(
            {
                "swa": SimpleNamespace(
                    block_table=torch.zeros((1, 4), dtype=torch.int32),
                    slot_mapping=torch.zeros((4,), dtype=torch.int64),
                    query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
                    seq_lens=torch.empty(0),
                ),
            }
        )
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
        monkeypatch.setattr(v2_attn, "_block_table", lambda swa: swa.block_table)
        monkeypatch.setattr(
            v2_attn, "_cache_slots", lambda cache, swa, pos: swa.slot_mapping
        )
        monkeypatch.setattr(
            v2_attn,
            "_token_requests",
            lambda meta, pos: torch.zeros(pos.shape[0], dtype=torch.long),
        )
        monkeypatch.setattr(
            v2_attn,
            "_row_slots",
            lambda cache, table, reqs, logical: torch.zeros_like(
                logical, dtype=torch.int32
            ),
        )
        monkeypatch.setattr(
            v2_attn,
            "_read_rows",
            lambda cache, table, reqs, logical: torch.zeros((4, 64, 128)),
        )
        monkeypatch.setattr(v2_attn, "_insert", lambda *args: None)
        attn._rotate = lambda q, pos: q
        monkeypatch.setattr(
            v2_attn,
            "sparse_attention",
            lambda *args, **kwargs: torch.zeros(4, 1, 4, 128),
        )

        qr = torch.randn(4, 16)
        kv = torch.randn(4, 128)
        positions = torch.zeros(4, dtype=torch.long)
        output = torch.zeros(4, 4, 128)

        res = v2_attn.DeepseekV41Attention._prepare_tpu(
            attn,
            hidden_states=None,
            qr=qr,
            kv=kv,
            qr_scale=None,
            kv_score=None,
            index_weights=None,
            positions=positions,
            output=output,
            qr_local=None,
        )
        assert res is None


# ==============================================================================
# Part 5: attention_kernels.py Token-Parallel & Sharded Body Coverage
# ==============================================================================
class TestAttentionKernelsKvOnlySp:
    """Verify attention_kernels.py lines 1046, 1141-1144, 1186, 1224."""

    def test_sparse_fp8_attention_qo_starts_from_positions_when_pages_none(self):
        """Line 1224: starts = [0, positions.shape[0]] when pages is None."""
        saved_op = attention_kernels._fp8_attention_qo_op
        mock_op = mock.MagicMock(return_value="op_result")
        attention_kernels._fp8_attention_qo_op = mock_op

        try:
            main_records = torch.ones(2, 4, dtype=torch.long)
            main = SimpleNamespace(
                rope=torch.zeros(1),
                raw=torch.zeros(8, 64, 128),
                shape=(16, 64, 128),
            )
            swa = SimpleNamespace(raw=torch.zeros(8, 64, 128))
            # Test positions.shape[0] != qr.shape[0] to verify line 1224
            qr = torch.zeros(64, 16)
            qr_local = torch.zeros(32, 16)
            positions = torch.zeros(128, dtype=torch.long)

            res = attention_kernels.sparse_fp8_attention_qo(
                qr,
                qr_local,
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                positions,
                torch.zeros(1),
                swa,
                main,
                torch.zeros(2, 4),
                main_records,
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(128),
                0.125,
                None,  # pages is None -> triggers line 1224 starts initialization
                None,  # starts is None
                16,
                False,
                128,
            )
            assert res == "op_result"
            args = mock_op.call_args[0]
            starts_arg = args[14]
            # Verifies starts contains [0, positions.shape[0]] = [0, 128]
            assert starts_arg.tolist() == [0, 128]

            # Second call: pages is not None, starts is provided (line 1221 False)
            pages_tensor = torch.zeros(1, 1, dtype=torch.int32)
            starts_tensor = torch.tensor([0, 128], dtype=torch.int32)
            res2 = attention_kernels.sparse_fp8_attention_qo(
                qr,
                qr_local,
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                positions,
                torch.zeros(1),
                swa,
                main,
                torch.zeros(2, 4),
                main_records,
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(128),
                0.125,
                pages_tensor,
                starts_tensor,
                16,
                False,
                128,
            )
            assert res2 == "op_result"
            args2 = mock_op.call_args[0]
            assert args2[13] is pages_tensor
            assert args2[14] is starts_tensor
        finally:
            attention_kernels._fp8_attention_qo_op = saved_op

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

    def test_sharded_body_in_kernel_all_gather_branches(self):
        """Lines 1141-1144: in-kernel all_gather vs bypass branches."""
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
            fn = inspect.unwrap(attention_kernels._fp8_attention_qo_op._init_fn.jit_fn)

            all_gather_calls: list[int] = []

            def fake_all_gather(x, **kwargs):
                all_gather_calls.append(x.shape[0])
                return jnp.concatenate([x, x], axis=0)

            with (
                mock.patch(
                    "jax.shard_map",
                    side_effect=lambda body, **kwargs: lambda *args: body(*args),
                ),
                mock.patch("jax.lax.all_to_all", side_effect=lambda x, **kwargs: x),
                mock.patch("jax.lax.all_gather", side_effect=fake_all_gather),
                mock.patch(
                    "vllm_torchtpu.layers.adapter.linear_common._quantized_matmul_jax",
                    side_effect=lambda q_in, w, ws: jnp.zeros(
                        (q_in.shape[0], 512), dtype=jnp.float32
                    ),
                ),
                mock.patch(
                    "vllm_torchtpu.models.vllm.deepseek_v4.attention._rope_jax",
                    side_effect=lambda q, pos, cs: q,
                ),
                mock.patch.object(
                    attention_kernels,
                    "sparse_fp8_attention_jax",
                    side_effect=lambda q, *args, **kwargs: q,
                ),
                mock.patch(
                    "vllm_torchtpu.models.vllm.deepseek_v4.attention._o_proj_jax",
                    side_effect=lambda out, pos, cs, w, ws, head_dim: jnp.zeros(
                        (out.shape[0], 1024), dtype=jnp.float32
                    ),
                ),
            ):
                # 1. Branch True: tokens = 128, q_in.shape[0] = 64 (< tokens)
                # Sub-threshold decode bucket where host sent local shard qr
                qr_small = jnp.zeros((128, 1280), dtype=jnp.bfloat16)
                qr_local = jnp.zeros((64, 1280), dtype=jnp.bfloat16)
                args_sub = (
                    qr_local,
                    qr_local,
                    jnp.zeros((1280, 256)),
                    jnp.zeros((256,)),
                    jnp.zeros((1280, 512)),
                    jnp.zeros((512,)),
                    jnp.zeros((128,)),  # positions is length 128 (tokens = 128)
                    jnp.zeros((128, 64)),
                    jnp.zeros((16, 64, 128)),
                    jnp.zeros((16, 64, 128)),
                    jnp.zeros((16, 64, 64)),
                    jnp.zeros((128, 64), dtype=jnp.int32),
                    jnp.zeros((128, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((64,)),
                    jnp.zeros((64,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.zeros((512, 1024)),
                    jnp.zeros((1024,)),
                    jnp.zeros((512, 1024)),
                    jnp.zeros((1024,)),
                    jnp.zeros((128,), dtype=jnp.int32),
                    0.125,
                    64,
                    None,
                    False,
                    128,
                )
                res_sub = fn(*args_sub)
                assert res_sub.shape == (128, 1024)
                assert len(all_gather_calls) == 1
                assert all_gather_calls[0] == 64

                # 2. Branch False: tokens = 128, q_in.shape[0] = 128 (>= tokens)
                # Source layer where host already gathered qr
                all_gather_calls.clear()
                res_full = fn(qr_small, *args_sub[1:])
                assert res_full.shape == (128, 1024)
                assert len(all_gather_calls) == 0  # In-kernel all_gather was skipped!
