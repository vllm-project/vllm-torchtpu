"""100% diff line and branch coverage tests for local_q attention op.

Targets:
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention.py (lines 255-268, 430,
  525-529, 565-571, 789, 796, 934-942, 1186-1215, 1243, 1258, 1260-1304, 1544)
- src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention_kernels.py (lines 20, 838,
  997-1184, 1187-1251)
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

from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention_kernels
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)


# ==============================================================================
# Part 2: v2/attention.py _init_local_q_weights & process_weights_after_loading
# ==============================================================================
def _fake_all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Fake all-gather: 2 ranks where rank 1 is inverted."""
    other = tensor ^ 0xFF if tensor.dtype == torch.uint8 else -tensor
    return torch.cat((tensor, other), dim=dim)


def _stub_attn(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    tp: int = 2,
    compress_ratio: int = 4,
    groups: int = 1,
    fp8_kv: bool = True,
):
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(v2_attn, "tensor_model_parallel_all_gather", _fake_all_gather)

    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn._fp8_kv = fp8_kv
    attn.compress_ratio = compress_ratio
    attn.n_local_groups = groups
    attn._local_q_attn = False
    attn._token_parallel_attn = enabled

    gen = torch.Generator().manual_seed(42)
    wq = torch.randn(16, 8, generator=gen).to(torch.float8_e4m3fn)
    wo = torch.randn(32, 4, generator=gen).to(torch.float8_e4m3fn)
    attn.wq_b = SimpleNamespace(weight=wq, weight_scale=torch.rand(8, generator=gen))
    attn.wo_a = SimpleNamespace(weight=wo, weight_scale=torch.rand(1, 4, generator=gen))
    return attn


class TestInitLocalQWeights:
    """Verify _init_local_q_weights early return guards and gather logic."""

    def test_gather_columns_and_register_buffers(self, monkeypatch: pytest.MonkeyPatch):
        attn = _stub_attn(monkeypatch)
        attn._init_local_q_weights()

        assert attn._local_q_attn is True
        assert hasattr(attn, "_wq_b_full")
        assert hasattr(attn, "_wq_b_full_scale")
        assert hasattr(attn, "_wo_a_full")
        assert hasattr(attn, "_wo_a_full_scale")

        # Verify shapes and types
        assert attn._wq_b_full.dtype == torch.float8_e4m3fn
        assert attn._wq_b_full.shape == (16, 16)
        assert attn._wq_b_full_scale.shape == (16,)
        assert attn._wo_a_full.dtype == torch.float8_e4m3fn
        assert attn._wo_a_full.shape == (32, 8)
        assert attn._wo_a_full_scale.shape == (8,)

        # Non-persistent buffers check
        assert "_wq_b_full" not in attn.state_dict()
        assert "_wo_a_full" not in attn.state_dict()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"enabled": False},
            {"tp": 1},
            {"fp8_kv": False},
            {"compress_ratio": 0},
            {"groups": 2},
        ],
    )
    def test_early_returns_config_guards(
        self, monkeypatch: pytest.MonkeyPatch, kwargs: dict
    ):
        attn = _stub_attn(monkeypatch, **kwargs)
        attn._init_local_q_weights()
        assert attn._local_q_attn is False
        assert not hasattr(attn, "_wq_b_full")

    def test_early_return_missing_or_invalid_scales(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # 1. wq_scale is None
        attn1 = _stub_attn(monkeypatch)
        attn1.wq_b = SimpleNamespace(weight=attn1.wq_b.weight)
        attn1._init_local_q_weights()
        assert attn1._local_q_attn is False

        # 2. wq_scale.ndim != 1 (e.g. 2D)
        attn2 = _stub_attn(monkeypatch)
        attn2.wq_b.weight_scale = torch.ones(2, 4)
        attn2._init_local_q_weights()
        assert attn2._local_q_attn is False

        # 3. wo_scale is None
        attn3 = _stub_attn(monkeypatch)
        attn3.wo_a = SimpleNamespace(weight=attn3.wo_a.weight)
        attn3._init_local_q_weights()
        assert attn3._local_q_attn is False

        # 4. KEEP_VLLM_LAYOUT_ATTR is True
        attn4 = _stub_attn(monkeypatch)
        setattr(attn4.wq_b, KEEP_VLLM_LAYOUT_ATTR, True)
        attn4._init_local_q_weights()
        assert attn4._local_q_attn is False

    def test_process_weights_after_loading_calls_init_local_q(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = _stub_attn(monkeypatch, tp=2)
        attn.attn_sink = torch.randn(4)
        attn._full_attn_sink = torch.zeros(8)
        attn.wo_a = SimpleNamespace(
            weight=torch.randn(8, 4, device="cpu"),
            weight_scale=torch.rand(1, 4),
        )

        init_called = []
        monkeypatch.setattr(
            attn, "_init_local_q_weights", lambda: init_called.append(True)
        )
        monkeypatch.setattr(attn, "_assert_o_proj_supported", lambda: None)
        with mock.patch(
            "vllm_torchtpu.models.vllm.deepseek_v4.attention.VllmDeepseekV4MLAAttention.process_weights_after_loading"
        ):
            # tp_size > 1
            attn.process_weights_after_loading(torch.bfloat16)
            assert init_called == [True]

            # tp_size == 1 else branch for full_attn_sink
            monkeypatch.setattr(
                v2_attn, "get_tensor_model_parallel_world_size", lambda: 1
            )
            attn._full_attn_sink = torch.zeros(4)
            attn.process_weights_after_loading(torch.bfloat16)
            assert init_called == [True, True]
            assert torch.equal(attn._full_attn_sink, attn.attn_sink.float())


# ==============================================================================
# Part 3: v2/attention.py forward() qr_local slicing and routing
# ==============================================================================
class TestAttentionForwardLocalQ:
    """Verify forward() qr_local slicing, projected output, and fallbacks."""

    @pytest.mark.parametrize(
        ("sp_sharded", "local_q", "expect_local"),
        [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ],
    )
    def test_qr_local_slicing_condition(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sp_sharded: bool,
        local_q: bool,
        expect_local: bool,
    ):
        attn = _stub_attn(monkeypatch)
        attn._sp_sharded_input = sp_sharded
        attn._local_q_attn = local_q
        attn.padded_heads = attn.n_local_heads = 2
        attn.head_dim = 4
        attn.q_lora_rank = 3
        attn.indexer = None

        tokens = 4
        qr = torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3)
        kv = torch.zeros(2, 4)
        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))
        monkeypatch.setattr(v2_attn, "sp_all_gather", lambda t: torch.cat((t, t)))

        seen = {}

        def prepare(*args, qr_local=None):
            seen["qr_local"] = qr_local
            return None

        attn._prepare_and_attn_fn = prepare
        monkeypatch.setattr(attn, "_o_proj", lambda o, positions: "fallback_o_proj")

        out = attn.forward(torch.zeros(tokens, dtype=torch.long), torch.zeros(2, 8))
        assert out == "fallback_o_proj"
        if expect_local:
            assert seen["qr_local"] is qr
        else:
            assert seen["qr_local"] is None

    def test_forward_projected_returns_wo_b(self, monkeypatch: pytest.MonkeyPatch):
        attn = _stub_attn(monkeypatch)
        attn._sp_sharded_input = True
        attn._local_q_attn = True
        attn.padded_heads = attn.n_local_heads = 2
        attn.head_dim = 4
        attn.q_lora_rank = 3
        attn.indexer = None

        tokens = 4
        qr = torch.ones(2, 3)
        kv = torch.zeros(2, 4)
        monkeypatch.setattr(
            attn, "_run_parallel_input_projections", lambda h: (None, None, None)
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))
        monkeypatch.setattr(v2_attn, "sp_all_gather", lambda t: torch.cat((t, t)))

        projected_tensor = torch.full((tokens, 5), 3.5)
        attn._prepare_and_attn_fn = lambda *args, qr_local=None: projected_tensor
        attn.wo_b = lambda x: (x * 2, None)

        out = attn.forward(torch.zeros(tokens, dtype=torch.long), torch.zeros(2, 8))
        assert torch.equal(out, projected_tensor * 2)

    def test_forward_sp_sharded_with_indexer_and_kv_score(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = _stub_attn(monkeypatch)
        attn._sp_sharded_input = True
        attn._local_q_attn = True
        attn.padded_heads = attn.n_local_heads = 2
        attn.head_dim = 4
        attn.q_lora_rank = 3
        attn.indexer = SimpleNamespace(n_head=2)

        tokens = 4
        qr = torch.ones(2, 3)
        kv = torch.ones(2, 4)
        idx_w = torch.ones(2, 2)
        kv_score = torch.ones(2, 1)

        monkeypatch.setattr(
            attn,
            "_run_parallel_input_projections",
            lambda h: (None, kv_score, idx_w),
        )
        monkeypatch.setattr(attn, "_split_qkv_and_norm", lambda x: (qr, None, kv))
        monkeypatch.setattr(v2_attn, "sp_all_gather", lambda t: torch.cat((t, t)))

        prepared_args = []

        def prepare(*args, qr_local=None):
            prepared_args.extend(args)
            return torch.zeros(tokens, 8)

        attn._prepare_and_attn_fn = prepare
        attn.wo_b = lambda x: (x, None)

        out = attn.forward(torch.zeros(tokens, dtype=torch.long), torch.zeros(2, 8))
        assert out.shape == (tokens, 8)
        assert len(prepared_args) == 8


# ==============================================================================
# Part 4: v2/attention.py _prepare_tpu() fused_qo dispatch & fallbacks
# ==============================================================================
class _FakeForwardContext:
    def __init__(self, attn_metadata, additional_kwargs=None):
        self.attn_metadata = attn_metadata
        self.additional_kwargs = additional_kwargs or {}


class TestPrepareTpuLocalQ:
    """Verify _prepare_tpu fused_qo condition, qo op call, and fallbacks."""

    def test_metadata_not_dict_early_exit(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            v2_attn, "get_forward_context", lambda: _FakeForwardContext(None)
        )
        attn = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=torch.ones(4, 8)),
        )
        output = torch.ones(2, 4)
        res = v2_attn.DeepseekV41Attention._prepare_tpu(
            attn, None, None, None, None, None, None, None, output=output
        )
        assert res is None
        assert torch.equal(output, torch.zeros(2, 4))

    def test_empty_cache_early_exit(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            v2_attn,
            "get_forward_context",
            lambda: _FakeForwardContext({"swa": SimpleNamespace()}),
        )
        attn = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(prefix="swa", kv_cache=torch.empty(0)),
        )
        output = torch.ones(2, 4)
        res = v2_attn.DeepseekV41Attention._prepare_tpu(
            attn, None, None, None, None, None, None, None, output=output
        )
        assert res is None
        assert torch.equal(output, torch.zeros(2, 4))

    def _setup_mock_attn(self, monkeypatch: pytest.MonkeyPatch, *, tiled=True):
        metadata = {
            "swa": SimpleNamespace(
                block_table=torch.zeros((1, 4), dtype=torch.int32),
                slot_mapping=torch.zeros((4,), dtype=torch.int64),
                query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            ),
            "comp": SimpleNamespace(
                block_table=torch.zeros((1, 4), dtype=torch.int32),
                query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
            ),
        }
        monkeypatch.setattr(
            v2_attn, "get_forward_context", lambda: _FakeForwardContext(metadata)
        )
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

        attn = _stub_attn(monkeypatch)
        attn.window_size = 64
        attn.head_dim = 128
        attn.scale = 0.125
        attn.n_local_heads = 4
        attn._ced_decoder = False
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

        if tiled:
            fake_cache = mock.MagicMock(spec=PagedCacheRows)
            fake_cache.numel.return_value = 100
            fake_cache.row_indices = lambda slots: slots.int()
            fake_cache.pages = lambda table: None
            attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=fake_cache)
            attn.main_cache_layer = SimpleNamespace(prefix="main", kv_cache=fake_cache)
            attn._static_forward_context = {
                "comp": SimpleNamespace(kv_cache=fake_cache)
            }
        else:
            raw_cache = torch.zeros(16, 64, 128)
            attn.swa_cache_layer = SimpleNamespace(prefix="swa", kv_cache=raw_cache)
            attn.main_cache_layer = SimpleNamespace(prefix="main", kv_cache=raw_cache)
            attn._static_forward_context = {"comp": SimpleNamespace(kv_cache=raw_cache)}

        attn._rotate = lambda x, pos: x
        return attn

    def test_fused_qo_dispatch_success(self, monkeypatch: pytest.MonkeyPatch):
        attn = self._setup_mock_attn(monkeypatch, tiled=True)
        attn._local_q_attn = True
        attn._init_local_q_weights()

        called_qo = {}

        def mock_qo(*args, **kwargs):
            called_qo["args"] = args
            return torch.full((4, 1024), 9.0)

        monkeypatch.setattr(v2_attn, "sparse_fp8_attention_qo", mock_qo)

        qr = torch.randn(4, 16)
        qr_local = torch.randn(2, 16)
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
            qr_local=qr_local,
        )

        assert res is not None
        assert torch.equal(res, torch.full((4, 1024), 9.0))
        assert "args" in called_qo
        # Arg 0 is qr, Arg 1 is qr_local
        assert called_qo["args"][0] is qr
        assert called_qo["args"][1] is qr_local

    def test_fused_qo_false_falls_back_to_unfused_attention(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        attn = self._setup_mock_attn(monkeypatch, tiled=True)
        attn._local_q_attn = True
        attn.wq_b = lambda x: (torch.zeros(4, 4 * 128), None)

        called_attn = {}

        def mock_sparse_fp8(*args, **kwargs):
            called_attn["called"] = True
            return torch.full((4, 4, 128), 5.0)

        monkeypatch.setattr(v2_attn, "sparse_fp8_attention", mock_sparse_fp8)

        qr = torch.randn(4, 16)
        # qr_local is None -> fused_qo is False
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
        assert called_attn.get("called") is True
        assert torch.equal(output, torch.full((4, 4, 128), 5.0))

    def test_untiled_fp8_raises_not_implemented(self, monkeypatch: pytest.MonkeyPatch):
        attn = self._setup_mock_attn(monkeypatch, tiled=False)
        attn._local_q_attn = False
        attn.wq_b = lambda x: (torch.zeros(4, 4 * 128), None)

        qr = torch.randn(4, 16)
        kv = torch.randn(4, 128)
        positions = torch.zeros(4, dtype=torch.long)
        output = torch.zeros(4, 4, 128)

        with pytest.raises(
            NotImplementedError, match="FP8 V4.1 KV cache requires the tiled path"
        ):
            v2_attn.DeepseekV41Attention._prepare_tpu(
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

    def test_prepare_tpu_ced_decoder_branch(self, monkeypatch: pytest.MonkeyPatch):
        attn = self._setup_mock_attn(monkeypatch, tiled=True)
        forward_ctx = _FakeForwardContext(
            {
                "swa": SimpleNamespace(
                    block_table=torch.zeros((1, 4), dtype=torch.int32),
                    slot_mapping=torch.zeros((4,), dtype=torch.int64),
                    query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
                ),
                "comp": SimpleNamespace(
                    block_table=torch.zeros((1, 4), dtype=torch.int32),
                    query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
                ),
            },
            additional_kwargs={
                "deepseek_v41_replay_starts": torch.zeros((4,), dtype=torch.long)
            },
        )
        monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
        attn._local_q_attn = True
        attn._init_local_q_weights()
        attn._ced_decoder = True
        monkeypatch.setattr(
            v2_attn,
            "sparse_fp8_attention_qo",
            lambda *args, **kwargs: torch.zeros(4, 1024),
        )

        qr = torch.randn(4, 16)
        qr_local = torch.randn(2, 16)
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
            qr_local=qr_local,
        )
        assert res is not None

    def test_prepare_tpu_untiled_bf16_fallback(self, monkeypatch: pytest.MonkeyPatch):
        attn = self._setup_mock_attn(monkeypatch, tiled=False)
        attn._fp8_kv = False
        attn.compress_ratio = 0
        attn._local_q_attn = False
        attn.wq_b = lambda x: (torch.zeros(4, 4 * 128), None)
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
# Part 5: v2/attention_kernels.py sparse_fp8_attention_qo & JAX bodies
# ==============================================================================
class TestAttentionKernelsLocalQ:
    """Verify sparse_fp8_attention_qo wrapper, fake reg, and JAX body branches."""

    def test_sparse_fp8_attention_qo_wrapper_assertions(self):
        saved_op = attention_kernels._fp8_attention_qo_op
        try:
            # 1. _fp8_attention_qo_op is None assertion
            attention_kernels._fp8_attention_qo_op = None
            main = SimpleNamespace(rope=torch.zeros(1), shape=(1, 64))
            with pytest.raises(AssertionError):
                attention_kernels.sparse_fp8_attention_qo(
                    *[None] * 8, main, main, *[None] * 16
                )

            # 2. main.rope is None assertion
            attention_kernels._fp8_attention_qo_op = mock.MagicMock()
            main_no_rope = SimpleNamespace(rope=None, shape=(1, 64))
            with pytest.raises(AssertionError):
                attention_kernels.sparse_fp8_attention_qo(
                    *[None] * 8, main_no_rope, main_no_rope, *[None] * 16
                )
        finally:
            attention_kernels._fp8_attention_qo_op = saved_op

    def test_sparse_fp8_attention_qo_pages_none_and_not_none(self):
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
            qr = torch.zeros(128, 16)
            qr_local = torch.zeros(64, 16)

            # Branch 1: pages is None
            res1 = attention_kernels.sparse_fp8_attention_qo(
                qr,
                qr_local,
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(128),
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
                None,  # pages is None
                None,  # starts is None
                16,
                False,
                128,
            )
            assert res1 == "op_result"
            args1 = mock_op.call_args[0]
            # page_size passed is main.raw.shape[0] = 8 (index 24)
            assert args1[24] == 8

            # Branch 2: pages is not None
            pages_tensor = torch.zeros((4, 2), dtype=torch.int32)
            starts_tensor = torch.zeros((4,), dtype=torch.int32)
            res2 = attention_kernels.sparse_fp8_attention_qo(
                qr,
                qr_local,
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(128),
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
            # page_size passed is main.shape[1] = 64 (index 24)
            assert args2[24] == 64
        finally:
            attention_kernels._fp8_attention_qo_op = saved_op

    def test_pallas_op_fake_registration(self):
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
                qr = torch.empty(256, 1280)
                qr_local = torch.empty(128, 1280)
                wo_a = torch.empty(512, 1024)
                tensors = [torch.empty(1) for _ in range(21)]
                tensors[4] = torch.empty(256)  # positions is arg 6 (args[5])
                tensors[16] = wo_a  # wo_a is arg 18 in full args list

                out = attention_kernels._fp8_attention_qo_op(
                    qr, qr_local, *tensors, 0.125, 64, None, True, 128
                )
                assert out.shape == (256, 1024)

    def test_fp8_attention_qo_inner_bodies_token_parallel_and_sharded(self):
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

            with (
                mock.patch(
                    "jax.shard_map",
                    side_effect=lambda body, **kwargs: lambda *args: body(*args),
                ),
                mock.patch("jax.lax.all_to_all", side_effect=lambda x, **kwargs: x),
                mock.patch(
                    "jax.lax.all_gather",
                    side_effect=lambda x, **kwargs: jnp.concatenate([x, x], axis=0),
                ),
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
                # 1. Condition True -> _local_body:
                # tp_size=2, tokens=256 (>= 128*tp, % 256 == 0), tokens_local=128
                qr = jnp.zeros((256, 1280), dtype=jnp.bfloat16)
                qr_local = jnp.zeros((128, 1280), dtype=jnp.bfloat16)
                args_28 = (
                    qr,
                    qr_local,
                    jnp.zeros((1280, 256)),
                    jnp.zeros((256,)),
                    jnp.zeros((1280, 512)),
                    jnp.zeros((512,)),
                    jnp.zeros((256,)),
                    jnp.zeros((256, 64)),
                    jnp.zeros((16, 64, 128)),
                    jnp.zeros((16, 64, 128)),
                    jnp.zeros((16, 64, 64)),
                    jnp.zeros((256, 64), dtype=jnp.int32),
                    jnp.zeros((256, 128), dtype=jnp.int32),
                    jnp.zeros((1, 1), dtype=jnp.int32),
                    jnp.zeros((2,), dtype=jnp.int32),
                    jnp.zeros((64,)),
                    jnp.zeros((64,)),
                    jnp.zeros((1,), dtype=jnp.int32),
                    jnp.zeros((512, 1024)),
                    jnp.zeros((1024,)),
                    jnp.zeros((512, 1024)),
                    jnp.zeros((1024,)),
                    jnp.zeros((256,), dtype=jnp.int32),
                    0.125,
                    64,
                    None,
                    False,
                    128,
                )
                res_local = fn(*args_28)
                assert res_local.shape == (256, 512)

                # 2. Condition False -> _sharded_body:
                # E.g. tokens = 128 (< 128 * tp_size=256), tokens_local = 64
                qr_small = jnp.zeros((128, 1280), dtype=jnp.bfloat16)
                qr_small_local = jnp.zeros((64, 1280), dtype=jnp.bfloat16)
                args_small = (
                    qr_small_local,
                    qr_small_local,
                    jnp.zeros((1280, 256)),
                    jnp.zeros((256,)),
                    jnp.zeros((1280, 512)),
                    jnp.zeros((512,)),
                    jnp.zeros((128,)),
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
                res_sharded = fn(*args_small)
                assert res_sharded.shape == (128, 1024)
                res_sharded_full = fn(qr_small, *args_small[1:])
                assert res_sharded_full.shape == (128, 1024)
