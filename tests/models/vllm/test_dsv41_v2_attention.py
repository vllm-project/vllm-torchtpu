# SPDX-License-Identifier: Apache-2.0
"""Coverage specification tests for DeepSeek V4.1 Attention module.

Covers all modified lines and branches in
`vllm_torchtpu.models.vllm.deepseek_v41.v2.attention`:
- CSA2 selection for FP8 KV at any TP size; fused preprocessing only at TP1
- Cache index selection (`_all_indices_fit`, tiled, untiled, non-source)
- BF16 `paged_attention` dispatch
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_cache import (
    PackedFP8IndexCacheRows,
)


def _make_dummy_vllm_config(max_num_seqs=4, max_model_len=4096):
    """Builds a minimal mock vllm_config for DeepseekV41Attention."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(to_dict=lambda: {"model_type": "deepseek_v41"}),
            max_model_len=max_model_len,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=None,
    )


@pytest.mark.parametrize(
    "tp,heads,fused", [(1, 64, True), (2, 32, False), (8, 8, False)]
)
def test_dsv41_attention_init_fp8_selects_csa2_at_every_tp(
    monkeypatch, tp, heads, fused
):
    """FP8 KV always uses CSA2 records; only TP1 64-head layers fuse preprocessing."""

    def fake_upstream_init(self, vllm_config, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.kv_cache_torch_dtype = torch.float8_e4m3fn
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.zeros(128, 64, dtype=torch.float32)
        )
        self.attn_sink = torch.zeros(heads, dtype=torch.float32)
        self.swa_cache_layer = SimpleNamespace()
        self.indexer = None
        self.compress_ratio = 0
        self.head_dim = 512
        self.window_size = 128
        self.n_local_heads = heads

    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention, "__init__", fake_upstream_init
    )
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_rank", lambda: tp - 1)
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: tp)

    attn = v2_attn.DeepseekV41Attention(_make_dummy_vllm_config(max_num_seqs=4))
    assert attn._fp8_kv is True
    assert attn._csa2 is True
    assert attn.swa_cache_layer._tpu_swa_words is True
    assert attn._fused_preprocess is fused
    assert attn._tp_rank.item() == tp - 1


def test_dsv41_attention_init_fp8_rejects_unsupported_heads(monkeypatch):
    def fake_upstream_init(self, vllm_config, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.kv_cache_torch_dtype = torch.float8_e4m3fn
        self.swa_cache_layer = SimpleNamespace()
        self.head_dim = 512
        self.window_size = 128
        self.n_local_heads = 4

    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention, "__init__", fake_upstream_init
    )
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 16)
    with pytest.raises(NotImplementedError, match="CSA2"):
        v2_attn.DeepseekV41Attention(_make_dummy_vllm_config())


@pytest.mark.parametrize("speculative_tokens", [None, 0, 3, 5])
def test_dsv41_attention_init_buffers_and_flags_bfloat16(
    monkeypatch, speculative_tokens
):
    """Test __init__ buffer registration and flag setting when BF16 KV cache is used."""

    def fake_upstream_init(self, vllm_config, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.kv_cache_torch_dtype = torch.bfloat16
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.zeros(64, 64, dtype=torch.float32)
        )
        self.attn_sink = torch.tensor([0.5, 1.5], dtype=torch.float32)
        self.swa_cache_layer = SimpleNamespace()
        self.indexer = None
        self.compress_ratio = 0

    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention, "__init__", fake_upstream_init
    )
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)

    cfg = _make_dummy_vllm_config(max_num_seqs=8)
    if speculative_tokens is not None:
        cfg.speculative_config = SimpleNamespace(
            num_speculative_tokens=speculative_tokens
        )
    attn = v2_attn.DeepseekV41Attention(cfg)
    assert attn._swa_query_group_size == (speculative_tokens or 0) + 1

    # BF16 KV keeps the plain paged path.
    assert attn._fp8_kv is False
    assert attn._csa2 is False
    assert attn._fused_preprocess is False


def test_dsv41_attention_init_unsupported_dtype(monkeypatch):
    """Test __init__ rejects unsupported KV cache data types."""

    def fake_upstream_init(self, vllm_config, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.kv_cache_torch_dtype = torch.float32

    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention, "__init__", fake_upstream_init
    )

    cfg = _make_dummy_vllm_config()
    with pytest.raises(
        NotImplementedError,
        match="TPU V4.1 supports --kv-cache-dtype auto, bfloat16 or fp8",
    ):
        v2_attn.DeepseekV41Attention(cfg)


@pytest.mark.parametrize("query_group_size", [1, 4, 6])
def test_prepare_tpu_tiled_paged_attention_dispatch_bfloat16(
    monkeypatch, query_group_size
):
    """Test BF16 tiled path dispatches to paged_attention."""
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)

    count = 2
    capacity = 4
    attn.compress_ratio = 1
    attn._max_model_len = capacity
    attn.indexer = None
    attn._fp8_kv = False  # BF16!
    attn._swa_query_group_size = query_group_size
    attn.window_size = 64
    attn.n_local_heads = 2
    attn.head_dim = 16
    attn.scale = 0.25
    attn._ced_decoder = False
    attn._ced_global_from_encoder = True
    attn.compressor = None
    attn.compressed_cache_prefix = "comp_kv"

    attn.topk_indices_buffer = torch.full((count, 4), 0, dtype=torch.int32)
    attn.attn_sink = torch.zeros(4, dtype=torch.float32)

    positions = torch.tensor([10, 20], dtype=torch.int32)
    swa_slots = torch.tensor([0, 1], dtype=torch.int32)

    fake_swa_cache = MagicMock(spec=PagedCacheRows)
    fake_swa_cache.numel.return_value = 100
    fake_swa_cache.row_indices = lambda slots: slots.int()
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_kv", kv_cache=fake_swa_cache)

    fake_comp_cache = MagicMock(spec=PagedCacheRows)
    fake_comp_cache.row_indices = lambda slots: slots.int()
    attn._static_forward_context = {
        "comp_kv": SimpleNamespace(kv_cache=fake_comp_cache)
    }

    metadata = {
        "swa_kv": SimpleNamespace(block_table=torch.zeros((2, 4), dtype=torch.int32)),
        "comp_kv": SimpleNamespace(block_table=torch.zeros((2, 4), dtype=torch.int32)),
    }

    forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
    monkeypatch.setattr(v2_attn, "_block_table", lambda meta: meta.block_table)
    monkeypatch.setattr(v2_attn, "_cache_slots", lambda cache, meta, pos: swa_slots)
    monkeypatch.setattr(
        v2_attn,
        "_token_requests",
        lambda meta, pos: torch.zeros(count, dtype=torch.int32),
    )
    monkeypatch.setattr(
        v2_attn,
        "_row_slots",
        lambda cache, table, reqs, log: torch.zeros_like(log, dtype=torch.int64),
    )
    monkeypatch.setattr(v2_attn, "_insert", lambda cache, slots, val, mask: None)
    attn.wq_b = lambda qr: qr
    attn._rotate = lambda x, pos: x

    paged_attn_called = []

    def fake_paged_attn(
        q, swa_cache, main_cache, records, main_records, sink, scale, group_size
    ):
        paged_attn_called.append(group_size)
        return torch.zeros(
            count, attn.n_local_heads, attn.head_dim, dtype=torch.bfloat16
        )

    monkeypatch.setattr(v2_attn, "paged_attention", fake_paged_attn)

    qr = torch.randn(count, attn.n_local_heads * attn.head_dim)
    kv = torch.randn(count, attn.n_local_heads * attn.head_dim)
    output = torch.zeros(count, attn.n_local_heads, attn.head_dim, dtype=torch.bfloat16)

    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn, None, qr, kv, None, None, None, positions, output
    )

    assert paged_attn_called == [query_group_size]


def test_prepare_tpu_untiled_fp8_kv_raises_not_implemented(monkeypatch):
    """Test untiled path with FP8 KV raises NotImplementedError."""
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)

    count = 1
    attn.compress_ratio = 2
    attn._max_model_len = 16
    attn.indexer = None
    attn._fp8_kv = True
    # window_size not a multiple of 64 makes tiled == False
    attn.window_size = 32
    attn.n_local_heads = 2
    attn.head_dim = 16
    attn.scale = 0.25
    attn._ced_decoder = False
    attn._ced_global_from_encoder = True
    attn.compressor = None
    attn.compressed_cache_prefix = "comp_kv"
    attn.topk_indices_buffer = torch.zeros((1, 8), dtype=torch.int32)
    attn.attn_sink = torch.zeros(4, dtype=torch.float32)

    positions = torch.tensor([2], dtype=torch.int32)
    swa_slots = torch.tensor([0], dtype=torch.int32)

    # swa_cache not PagedCacheRows also ensures tiled == False
    fake_swa_cache = torch.zeros((10, 32))
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_kv", kv_cache=fake_swa_cache)

    fake_comp_cache = torch.zeros((10, 32))
    attn._static_forward_context = {
        "comp_kv": SimpleNamespace(kv_cache=fake_comp_cache)
    }

    metadata = {
        "swa_kv": SimpleNamespace(block_table=torch.zeros((1, 4), dtype=torch.int32)),
        "comp_kv": SimpleNamespace(block_table=torch.zeros((1, 4), dtype=torch.int32)),
    }

    forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
    monkeypatch.setattr(v2_attn, "_block_table", lambda meta: meta.block_table)
    monkeypatch.setattr(v2_attn, "_cache_slots", lambda cache, meta, pos: swa_slots)
    monkeypatch.setattr(
        v2_attn,
        "_token_requests",
        lambda meta, pos: torch.zeros(count, dtype=torch.int32),
    )
    monkeypatch.setattr(
        v2_attn,
        "_read_rows",
        lambda cache, table, reqs, log: torch.zeros((count, log.shape[1], 16)),
    )
    monkeypatch.setattr(v2_attn, "_insert", lambda cache, slots, val, mask: None)
    attn.wq_b = lambda qr: qr
    attn._rotate = lambda x, pos: x

    qr = torch.randn(count, attn.n_local_heads * attn.head_dim)
    kv = torch.randn(count, attn.n_local_heads * attn.head_dim)
    output = torch.zeros(count, attn.n_local_heads, attn.head_dim, dtype=torch.bfloat16)

    with pytest.raises(
        NotImplementedError, match="FP8 V4.1 KV cache requires the tiled path"
    ):
        v2_attn.DeepseekV41Attention._prepare_tpu(
            attn, None, qr, kv, None, None, None, positions, output
        )


def test_prepare_tpu_indexer_not_all_indices_fit_both_branches(monkeypatch):
    """_prepare_tpu routes both tiled and untiled windows to the paged indexer."""
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)

    count = 2
    capacity = 4
    attn.compress_ratio = 1
    attn._max_model_len = capacity
    monkeypatch.setattr(
        v2_attn.DeepseekV41Attention,
        "_all_indices_fit",
        property(lambda self: False),
    )
    attn._fp8_kv = False  # BF16
    attn.n_local_heads = 2
    attn.head_dim = 16
    attn.scale = 0.25
    attn._ced_decoder = False
    attn._ced_global_from_encoder = True
    attn.compressor = None
    attn.compressed_cache_prefix = "comp_kv"
    attn.topk_indices_buffer = torch.zeros((count, 8), dtype=torch.int32)
    attn.attn_sink = torch.zeros(4, dtype=torch.float32)

    indexer_mock = SimpleNamespace(
        wq_b=lambda qr: qr,
        n_head=2,
        head_dim=16,
        softmax_scale=16**-0.5,
        k_cache=SimpleNamespace(
            prefix="idx_k", kv_cache=MagicMock(spec=PackedFP8IndexCacheRows)
        ),
    )
    indexer_mock.k_cache.kv_cache.row_indices = lambda slots: slots.int()
    attn.indexer = indexer_mock

    positions = torch.tensor([1, 2], dtype=torch.int32)
    swa_slots = torch.tensor([0, 1], dtype=torch.int32)

    # 1. Test tiled branch
    attn.window_size = 64
    fake_swa_cache = MagicMock(spec=PagedCacheRows)
    fake_swa_cache.numel.return_value = 100
    fake_swa_cache.row_indices = lambda slots: slots.int()
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_kv", kv_cache=fake_swa_cache)

    fake_comp_cache = MagicMock(spec=PagedCacheRows)
    fake_comp_cache.row_indices = lambda slots: slots.int()
    attn._static_forward_context = {
        "comp_kv": SimpleNamespace(kv_cache=fake_comp_cache)
    }

    metadata = {
        "swa_kv": SimpleNamespace(block_table=torch.zeros((2, 4), dtype=torch.int32)),
        "comp_kv": SimpleNamespace(block_table=torch.zeros((2, 4), dtype=torch.int32)),
        "idx_k": SimpleNamespace(
            block_table=torch.zeros((2, 4), dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        ),
    }

    forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
    monkeypatch.setattr(v2_attn, "_block_table", lambda meta: meta.block_table)
    monkeypatch.setattr(v2_attn, "_cache_slots", lambda cache, meta, pos: swa_slots)
    monkeypatch.setattr(
        v2_attn,
        "_token_requests",
        lambda meta, pos: torch.zeros(count, dtype=torch.int32),
    )
    monkeypatch.setattr(
        v2_attn,
        "_row_slots",
        lambda cache, table, reqs, log: torch.zeros_like(log, dtype=torch.int64),
    )
    monkeypatch.setattr(v2_attn, "_insert", lambda cache, slots, val, mask: None)
    attn.wq_b = lambda qr: qr
    attn._rotate = lambda x, pos: x

    select_paged_called = []

    def fake_select_paged(*args, **kwargs):
        select_paged_called.append(True)
        return torch.zeros((count, 2), dtype=torch.int32)

    monkeypatch.setattr(attn, "_select_paged_indices", fake_select_paged)
    monkeypatch.setattr(
        v2_attn,
        "paged_attention",
        lambda *args, **kwargs: torch.zeros(count, attn.n_local_heads, attn.head_dim),
    )

    qr = torch.randn(count, attn.n_local_heads * attn.head_dim)
    kv = torch.randn(count, attn.n_local_heads * attn.head_dim)
    idx_weights = torch.ones(count, indexer_mock.n_head)
    output = torch.zeros(count, attn.n_local_heads, attn.head_dim, dtype=torch.bfloat16)

    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn, None, qr, kv, None, None, idx_weights, positions, output
    )
    assert len(select_paged_called) == 1

    # 2. Test untiled branch (window_size not multiple of 64 and plain tensor cache)
    attn.window_size = 30
    fake_swa_untiled = torch.zeros((10, 30))
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_kv", kv_cache=fake_swa_untiled)
    attn.indexer.k_cache = SimpleNamespace(
        prefix="idx_k", kv_cache=torch.zeros((10, 30))
    )
    attn._static_forward_context = {
        "comp_kv": SimpleNamespace(kv_cache=torch.zeros((10, 30)))
    }

    monkeypatch.setattr(
        v2_attn,
        "_read_rows",
        lambda cache, table, reqs, log: torch.zeros(
            (count, log.shape[1], indexer_mock.head_dim)
        ),
    )
    monkeypatch.setattr(
        v2_attn,
        "sparse_attention",
        lambda *args, **kwargs: torch.zeros(
            count, 1, attn.n_local_heads, attn.head_dim
        ),
    )

    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn, None, qr, kv, None, None, idx_weights, positions, output
    )
    assert len(select_paged_called) == 2


def test_write_encoder_global_kv_empty_metadata(monkeypatch):
    """Test write_encoder_global_kv early return when kv_cache is empty."""
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    attn.kv_cache = torch.empty(0)
    monkeypatch.setattr(
        v2_attn, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )

    # With empty kv_cache or None metadata, does nothing
    attn.write_encoder_global_kv(torch.randn(2, 10), torch.tensor([0, 1]))
