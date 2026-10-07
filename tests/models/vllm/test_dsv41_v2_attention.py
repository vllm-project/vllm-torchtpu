# SPDX-License-Identifier: Apache-2.0
"""Coverage specification tests for DeepSeek V4.1 Attention module.

Covers all modified lines and branches in
`vllm_torchtpu.models.vllm.deepseek_v41.v2.attention`:
- Buffer registration: `_tp_rank_1d`, `_full_attn_sink`,
  `_precompact_records`, `_token_parallel_attn`
- `_swa_max_requests` configuration across scheduler capacities (<=4 vs >4)
- Post-loading weight synchronization under TP=1 vs TP>1
- Cache index selection (`_all_indices_fit`, tiled, untiled, non-source)
- Attention dispatch routing (`sparse_fp8_attention` vs `paged_attention`)
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows


def _make_dummy_vllm_config(max_num_seqs=4, max_model_len=4096):
    """Builds a minimal mock vllm_config for DeepseekV41Attention."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(to_dict=lambda: {"model_type": "deepseek_v41"}),
            max_model_len=max_model_len,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_dsv41_attention_init_buffers_and_flags_fp8(monkeypatch):
    """Test __init__ buffer registration and flags when FP8 KV is enabled."""

    def fake_upstream_init(self, vllm_config, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.kv_cache_torch_dtype = torch.float8_e4m3fn
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.zeros(128, 64, dtype=torch.float32)
        )
        self.attn_sink = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention, "__init__", fake_upstream_init
    )
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 8)

    cfg = _make_dummy_vllm_config(max_num_seqs=4)
    attn = v2_attn.DeepseekV41Attention(cfg)

    # Invariant: _tp_rank_1d is a 1D tensor of shape (1,) holding TP rank
    assert attn._tp_rank_1d.shape == (1,)
    assert attn._tp_rank_1d.item() == 3

    # Invariant: _full_attn_sink holds repeated sink heads for tp_size ranks
    assert attn._full_attn_sink.shape == (8 * 4,)
    torch.testing.assert_close(attn._full_attn_sink[:4], attn.attn_sink)

    # Invariant: FP8 KV enables precompact records and token-parallel attention
    assert attn._fp8_kv is True
    assert attn._precompact_records is True
    assert attn._token_parallel_attn is True
    assert attn._swa_max_requests == 4


def test_dsv41_attention_init_buffers_and_flags_bfloat16(monkeypatch):
    """Test __init__ buffer registration and flag setting when BF16 KV cache is used."""

    def fake_upstream_init(self, vllm_config, *args, **kwargs):
        torch.nn.Module.__init__(self)
        self.kv_cache_torch_dtype = torch.bfloat16
        self.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.zeros(64, 64, dtype=torch.float32)
        )
        self.attn_sink = torch.tensor([0.5, 1.5], dtype=torch.float32)

    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention, "__init__", fake_upstream_init
    )
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)

    cfg = _make_dummy_vllm_config(max_num_seqs=8)
    attn = v2_attn.DeepseekV41Attention(cfg)

    # Invariant: BF16 disables precompact_records and token_parallel_attn
    assert attn._fp8_kv is False
    assert attn._precompact_records is False
    assert attn._token_parallel_attn is False

    # Invariant: max_num_seqs > 4 disables SWA patch (_swa_max_requests is None)
    assert attn._swa_max_requests is None
    assert attn._full_attn_sink.shape == (2,)


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


def test_process_weights_after_loading_tp_size_1(monkeypatch):
    """Test process_weights_after_loading copies directly when tp_size == 1."""
    import sys
    from types import ModuleType

    mock_mod = ModuleType("vllm_torchtpu.models.vllm.deepseek_v4.attention")
    mock_base = type(
        "VllmDeepseekV4MLAAttention",
        (),
        {"process_weights_after_loading": staticmethod(lambda self, act_dtype: None)},
    )
    mock_mod.VllmDeepseekV4MLAAttention = mock_base
    monkeypatch.setitem(
        sys.modules, "vllm_torchtpu.models.vllm.deepseek_v4.attention", mock_mod
    )

    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)

    attn.attn_sink = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    full_sink = torch.zeros(3, dtype=torch.float32)
    attn.register_buffer("_full_attn_sink", full_sink, persistent=False)
    attn.wo_a = SimpleNamespace(weight=SimpleNamespace(device=torch.device("cpu")))

    monkeypatch.setattr(attn, "_assert_o_proj_supported", lambda: None)
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)

    attn.process_weights_after_loading(torch.bfloat16)
    torch.testing.assert_close(attn._full_attn_sink, attn.attn_sink.float())


def test_process_weights_after_loading_tp_size_multi(monkeypatch):
    """Test process_weights_after_loading uses all-gather when tp_size > 1."""
    import sys
    from types import ModuleType

    mock_mod = ModuleType("vllm_torchtpu.models.vllm.deepseek_v4.attention")
    mock_base = type(
        "VllmDeepseekV4MLAAttention",
        (),
        {"process_weights_after_loading": staticmethod(lambda self, act_dtype: None)},
    )
    mock_mod.VllmDeepseekV4MLAAttention = mock_base
    monkeypatch.setitem(
        sys.modules, "vllm_torchtpu.models.vllm.deepseek_v4.attention", mock_mod
    )

    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)

    attn.attn_sink = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    full_sink = torch.zeros(4, dtype=torch.float32)
    attn.register_buffer("_full_attn_sink", full_sink, persistent=False)

    # Mock device as tpu to trigger o_proj_op registration branch
    attn.wo_a = SimpleNamespace(
        weight=SimpleNamespace(device=SimpleNamespace(type="tpu"))
    )

    called_o_proj = []
    monkeypatch.setattr(
        v2_attn.DeepseekV41Attention,
        "o_proj_op",
        property(lambda self: called_o_proj.append(True)),
    )

    monkeypatch.setattr(attn, "_assert_o_proj_supported", lambda: None)
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 2)

    def fake_all_gather(tensor, dim=0):
        assert dim == 0
        return torch.cat([tensor, tensor * 10], dim=0)

    monkeypatch.setattr(v2_attn, "tensor_model_parallel_all_gather", fake_all_gather)

    attn.process_weights_after_loading(torch.bfloat16)

    # Invariant: gathered tensor of size 4 is copied to _full_attn_sink
    expected = torch.tensor([1.0, 2.0, 10.0, 20.0], dtype=torch.float32)
    torch.testing.assert_close(attn._full_attn_sink, expected)
    assert len(called_o_proj) == 1


def test_prepare_tpu_indexer_all_indices_fit_masking(monkeypatch):
    """Test _all_indices_fit masks (logical >= lengths) | (swa_slots < 0) to -1."""
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)

    count = 4
    capacity = 8
    attn.compress_ratio = 2
    attn._max_model_len = capacity * attn.compress_ratio
    attn.indexer = SimpleNamespace()
    monkeypatch.setattr(
        v2_attn.DeepseekV41Attention,
        "_all_indices_fit",
        property(lambda self: True),
    )
    attn._fp8_kv = True
    attn.window_size = 64
    attn.n_local_heads = 2
    attn.head_dim = 16
    attn.scale = 0.25
    attn._ced_decoder = False
    attn._ced_global_from_encoder = True
    attn.compressor = None
    attn.compressed_cache_prefix = "comp_kv"

    attn.topk_indices_buffer = torch.full((count, 16), -1, dtype=torch.int32)
    attn.attn_sink = torch.zeros(4, dtype=torch.float32)
    attn.register_buffer(
        "_full_attn_sink", torch.zeros(8, dtype=torch.float32), persistent=False
    )
    attn.register_buffer(
        "_tp_rank_1d", torch.tensor([0], dtype=torch.int32), persistent=False
    )
    attn._precompact_records = True
    attn._token_parallel_attn = True
    attn._swa_max_requests = 4

    positions = torch.tensor([1, 5, 9, 3], dtype=torch.int32)
    swa_slots = torch.tensor(
        [0, 1, -1, 2], dtype=torch.int32
    )  # token 2 is padding (-1)

    # Cache setup
    fake_swa_cache = MagicMock(spec=PagedCacheRows)
    fake_swa_cache.numel.return_value = 100
    fake_swa_cache.row_indices = lambda slots: slots.int()
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_kv", kv_cache=fake_swa_cache)

    fake_comp_cache = MagicMock(spec=PagedCacheRows)
    fake_comp_cache.pages.return_value = torch.zeros((4, 2), dtype=torch.int32)
    attn._static_forward_context = {
        "comp_kv": SimpleNamespace(kv_cache=fake_comp_cache)
    }

    metadata = {
        "swa_kv": SimpleNamespace(
            block_table=torch.zeros((2, 4), dtype=torch.int32), query_start_loc=None
        ),
        "comp_kv": SimpleNamespace(
            block_table=torch.zeros((2, 4), dtype=torch.int32),
            query_start_loc=torch.tensor([0, 4]),
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

    sparse_fp8_called = []

    def fake_sparse_fp8_attn(
        q,
        swa_cache,
        main_cache,
        records,
        main_records,
        sink,
        scale,
        pages,
        starts,
        max_reqs,
        precompacted,
        token_parallel,
        full_sink,
        tp_rank,
    ):
        sparse_fp8_called.append(
            {
                "precompacted": precompacted,
                "token_parallel": token_parallel,
                "main_records": main_records.clone(),
                "full_sink": full_sink,
                "tp_rank": tp_rank,
            }
        )
        return torch.ones(
            count, attn.n_local_heads, attn.head_dim, dtype=torch.bfloat16
        )

    monkeypatch.setattr(v2_attn, "sparse_fp8_attention", fake_sparse_fp8_attn)

    qr = torch.randn(count, attn.n_local_heads * attn.head_dim)
    kv = torch.randn(count, attn.n_local_heads * attn.head_dim)
    output = torch.zeros(count, attn.n_local_heads, attn.head_dim, dtype=torch.bfloat16)

    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn, None, qr, kv, None, None, None, positions, output
    )

    assert len(sparse_fp8_called) == 1
    call = sparse_fp8_called[0]
    assert call["precompacted"] is True
    assert call["token_parallel"] is True
    assert call["tp_rank"].item() == 0

    # Verify _all_indices_fit mask logic:
    # Token 2 (index 2) had swa_slots == -1, so all its main_records must be -1
    assert (call["main_records"][2] == -1).all()

    # Token 0 had length=1, so only index 0 can be valid (>= length must be -1)
    assert call["main_records"][0, 0] == 0
    assert (call["main_records"][0, 1:capacity] == -1).all()

    # Output masked: swa_slots < 0 row (token 2) must be zeroed in output
    assert (output[2] == 0).all()
    assert (output[0] == 1).all()


def test_prepare_tpu_non_source_layer_buffer_reuse(monkeypatch):
    """Test non-source layers (indexer is None) reuse topk_indices_buffer directly."""
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)

    count = 2
    capacity = 4
    attn.compress_ratio = 1
    attn._max_model_len = capacity
    attn.indexer = None  # Non-source layer!
    attn._fp8_kv = True
    attn.window_size = 64
    attn.n_local_heads = 2
    attn.head_dim = 16
    attn.scale = 0.25
    attn._ced_decoder = False
    attn._ced_global_from_encoder = True
    attn.compressor = None
    attn.compressed_cache_prefix = "comp_kv"

    # Pre-populate topk_indices_buffer as written by a previous source layer
    buf = torch.full((count, 8), -1, dtype=torch.int32)
    buf[0, :3] = torch.tensor([10, 20, 30])
    buf[1, :2] = torch.tensor([5, 15])
    attn.topk_indices_buffer = buf

    attn.attn_sink = torch.zeros(4, dtype=torch.float32)
    attn.register_buffer(
        "_full_attn_sink", torch.zeros(4, dtype=torch.float32), persistent=False
    )
    attn.register_buffer(
        "_tp_rank_1d", torch.tensor([0], dtype=torch.int32), persistent=False
    )
    attn._precompact_records = True
    attn._token_parallel_attn = True
    attn._swa_max_requests = None

    positions = torch.tensor([50, 60], dtype=torch.int32)
    swa_slots = torch.tensor([0, 1], dtype=torch.int32)

    fake_swa_cache = MagicMock(spec=PagedCacheRows)
    fake_swa_cache.numel.return_value = 100
    fake_swa_cache.row_indices = lambda slots: slots.int()
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_kv", kv_cache=fake_swa_cache)

    fake_comp_cache = MagicMock(spec=PagedCacheRows)
    fake_comp_cache.pages.return_value = torch.zeros((2, 2), dtype=torch.int32)
    attn._static_forward_context = {
        "comp_kv": SimpleNamespace(kv_cache=fake_comp_cache)
    }

    metadata = {
        "swa_kv": SimpleNamespace(
            block_table=torch.zeros((2, 4), dtype=torch.int32), query_start_loc=None
        ),
        "comp_kv": SimpleNamespace(
            block_table=torch.zeros((2, 4), dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2]),
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

    captured_records = []

    def fake_sparse_fp8(
        q,
        swa_cache,
        main_cache,
        records,
        main_records,
        sink,
        scale,
        pages,
        starts,
        max_reqs,
        precompacted,
        token_parallel,
        full_sink,
        tp_rank,
    ):
        captured_records.append(main_records)
        return torch.zeros(
            count, attn.n_local_heads, attn.head_dim, dtype=torch.bfloat16
        )

    monkeypatch.setattr(v2_attn, "sparse_fp8_attention", fake_sparse_fp8)

    qr = torch.randn(count, attn.n_local_heads * attn.head_dim)
    kv = torch.randn(count, attn.n_local_heads * attn.head_dim)
    output = torch.zeros(count, attn.n_local_heads, attn.head_dim, dtype=torch.bfloat16)

    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn, None, qr, kv, None, None, None, positions, output
    )

    # Invariant: main_records matches topk_indices_buffer sliced to capacity (4)
    assert len(captured_records) == 1
    recs = captured_records[0]
    torch.testing.assert_close(
        recs[0, :3], torch.tensor([10, 20, 30], dtype=torch.int32)
    )
    assert recs[0, 3] == -1
    torch.testing.assert_close(recs[1, :2], torch.tensor([5, 15], dtype=torch.int32))


def test_prepare_tpu_tiled_paged_attention_dispatch_bfloat16(monkeypatch):
    """Test BF16 tiled path dispatches to paged_attention."""
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)

    count = 2
    capacity = 4
    attn.compress_ratio = 1
    attn._max_model_len = capacity
    attn.indexer = None
    attn._fp8_kv = False  # BF16!
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
    attn.register_buffer(
        "_full_attn_sink", torch.zeros(4, dtype=torch.float32), persistent=False
    )
    attn.register_buffer(
        "_tp_rank_1d", torch.tensor([0], dtype=torch.int32), persistent=False
    )
    attn._precompact_records = False
    attn._token_parallel_attn = False

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

    def fake_paged_attn(q, swa_cache, main_cache, records, main_records, sink, scale):
        paged_attn_called.append(True)
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

    assert len(paged_attn_called) == 1


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
    """Test _prepare_tpu indexer when _all_indices_fit is False (tiled & untiled)."""
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
    attn.register_buffer(
        "_full_attn_sink", torch.zeros(4, dtype=torch.float32), persistent=False
    )
    attn.register_buffer(
        "_tp_rank_1d", torch.tensor([0], dtype=torch.int32), persistent=False
    )
    attn._precompact_records = False
    attn._token_parallel_attn = False

    indexer_mock = SimpleNamespace(
        wq_b=lambda qr: qr,
        n_head=2,
        head_dim=16,
        k_cache=SimpleNamespace(
            prefix="idx_k", kv_cache=MagicMock(spec=PagedCacheRows)
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
        "idx_k": SimpleNamespace(block_table=torch.zeros((2, 4), dtype=torch.int32)),
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
        "_request_row_slots",
        lambda cache, table, cap: torch.zeros((2, cap), dtype=torch.int64),
    )
    monkeypatch.setattr(
        v2_attn,
        "_row_slots",
        lambda cache, table, reqs, log: torch.zeros_like(log, dtype=torch.int64),
    )
    monkeypatch.setattr(v2_attn, "_insert", lambda cache, slots, val, mask: None)
    attn.wq_b = lambda qr: qr
    attn._rotate = lambda x, pos: x
    attn._quantize_indexer = lambda x: x

    select_paged_called = []

    def fake_select_paged(
        qr, pos, cache, records, reqs, weights, lengths, slots, pending
    ):
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
    select_indices_called = []

    def fake_select_indices(scores, lengths, logical, cnt):
        select_indices_called.append(True)
        return torch.zeros((cnt, 2), dtype=torch.int32)

    monkeypatch.setattr(attn, "_select_indices", fake_select_indices)
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
    assert len(select_indices_called) == 1


def test_write_encoder_global_kv_empty_metadata(monkeypatch):
    """Test write_encoder_global_kv early return when kv_cache is empty."""
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    attn.kv_cache = torch.empty(0)
    monkeypatch.setattr(
        v2_attn, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )

    # With empty kv_cache or None metadata, does nothing
    attn.write_encoder_global_kv(torch.randn(2, 10), torch.tensor([0, 1]))
