"""CPU checks for DeepSeek-V4.1 attention forward routing and `_prepare_tpu`
early exits."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows


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

    gen = torch.Generator().manual_seed(42)
    wq = torch.randn(16, 8, generator=gen).to(torch.float8_e4m3fn)
    wo = torch.randn(32, 4, generator=gen).to(torch.float8_e4m3fn)
    attn.wq_b = SimpleNamespace(weight=wq, weight_scale=torch.rand(8, generator=gen))
    attn.wo_a = SimpleNamespace(weight=wo, weight_scale=torch.rand(1, 4, generator=gen))
    return attn


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

    def test_untiled_fp8_raises_not_implemented(self, monkeypatch: pytest.MonkeyPatch):
        attn = self._setup_mock_attn(monkeypatch, tiled=False)
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

    def test_prepare_tpu_untiled_bf16_fallback(self, monkeypatch: pytest.MonkeyPatch):
        attn = self._setup_mock_attn(monkeypatch, tiled=False)
        attn._fp8_kv = False
        attn.compress_ratio = 0
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
