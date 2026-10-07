# SPDX-License-Identifier: Apache-2.0
"""100% Diff coverage test suite for SP attention and projections.

Covers all lines and branches in:
- vllm_torchtpu.models.vllm.deepseek_v41.v2.attention
  (attention.py:424-521, 589-600, 629-705, 813-882)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.model
  (model.py:38-74, 524-530, 616-625)
"""

from __future__ import annotations

import dataclasses
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu import envs
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows

# ==============================================================================
# 1. Tests for v2/model.py: _ced_config, DeepseekV41DecoderLayer, DeepseekV41Model
# ==============================================================================


def test_ced_config_combinations(monkeypatch, caplog):
    """Test _ced_config CED on/off, prefix caching, warnings, and fallback."""
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_CED", True)

    # 1. CED=1 and prefix caching disabled -> ced_enabled=True
    cfg1 = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=40),
        ),
    )
    enabled, boundary = v2_model._ced_config(cfg1)
    assert enabled is True
    assert boundary == 20

    # 2. CED=1 and prefix caching enabled with warn_prefix_caching=True
    cfg2 = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=30),
        ),
    )
    with caplog.at_level(logging.WARNING):
        enabled2, boundary2 = v2_model._ced_config(cfg2, warn_prefix_caching=True)
    assert enabled2 is False
    assert boundary2 == 15
    assert "CED bounded replay requires prefix caching disabled" in caplog.text

    # 3. CED=1 and prefix caching enabled with warn_prefix_caching=False
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        enabled3, boundary3 = v2_model._ced_config(cfg2, warn_prefix_caching=False)
    assert enabled3 is False
    assert boundary3 == 15
    assert caplog.text == ""

    # 4. CED=0 -> ced_enabled=False
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_CED", False)
    enabled4, boundary4 = v2_model._ced_config(cfg1)
    assert enabled4 is False
    assert boundary4 == 20

    # 5. model_config is None, fallback_config provided
    cfg5 = SimpleNamespace(
        cache_config=None,
        model_config=None,
    )
    fallback = SimpleNamespace(num_hidden_layers=24)
    enabled5, boundary5 = v2_model._ced_config(cfg5, fallback_config=fallback)
    assert enabled5 is False
    assert boundary5 == 12


def test_decoder_layer_init_and_sp_binding(monkeypatch):
    """Test DeepseekV41DecoderLayer __init__ and forward device_ops binding."""
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_CED", True)

    def fake_upstream_init(self, vllm_config, prefix="", **kwargs):
        nn.Module.__init__(self)
        self.prefix = prefix
        self.use_sequence_parallel = True
        self.attn = nn.Module()

    monkeypatch.setattr(
        v2_model.upstream.DeepseekV4DecoderLayer, "__init__", fake_upstream_init
    )

    # 1. layer_idx >= ced_boundary and ced_enabled=True -> uses device_hooks
    monkeypatch.setattr(v2_model.upstream, "extract_layer_index", lambda p: 25)
    vllm_cfg = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=40),
        ),
    )
    layer = v2_model.DeepseekV41DecoderLayer(vllm_cfg, prefix="model.layers.25")
    assert getattr(layer.attn, "_sp_sharded_input", None) is True

    # 2. layer_idx < ced_boundary -> normal init
    monkeypatch.setattr(v2_model.upstream, "extract_layer_index", lambda p: 5)
    layer2 = v2_model.DeepseekV41DecoderLayer(vllm_cfg, prefix="model.layers.5")
    assert getattr(layer2.attn, "_sp_sharded_input", None) is True

    # 3. Layer without attn attribute
    def fake_no_attn_init(self, vllm_config, prefix="", **kwargs):
        nn.Module.__init__(self)
        self.prefix = prefix
        self.use_sequence_parallel = False

    monkeypatch.setattr(
        v2_model.upstream.DeepseekV4DecoderLayer, "__init__", fake_no_attn_init
    )
    layer3 = v2_model.DeepseekV41DecoderLayer(vllm_cfg, prefix="model.layers.5")
    assert not hasattr(layer3, "attn")

    # 4. Verify forward device_ops includes sp_all_gather identity
    bound_sp_all_gather = v2_model.DeepseekV41DecoderLayer.forward.__globals__.get(
        "sp_all_gather"
    )
    assert bound_sp_all_gather is not None
    assert bound_sp_all_gather("identity_token") == "identity_token"


def test_model_forward_stage_write_encoder_global_kv(monkeypatch):
    """Test _forward_stage passes sp_sharded=stage_sp to global KV write."""
    written_calls: list[dict] = []

    class FakeAttn:
        def write_encoder_global_kv(self, norm, pos, *, sp_sharded=False):
            written_calls.append({"norm": norm, "pos": pos, "sp_sharded": sp_sharded})

    class FakeDecoderLayer:
        attn = FakeAttn()

        def attn_norm(self, x):
            return x

    class FakeEncoderLayer:
        def __call__(self, *args, **kwargs):
            hidden = args[0]
            pre = torch.ones((hidden.shape[0], 4), dtype=torch.float32)
            return hidden, None, None, None, pre, None

    model = MagicMock()
    model.start_layer = 0
    model.end_layer = 1
    model._ced_boundary = 1
    model.layers = [FakeEncoderLayer(), FakeDecoderLayer()]
    model._stage_uses_sequence_parallel = lambda stage: True
    model._prepare_engram = MagicMock(return_value=(None, None))
    model.use_mega_moe = False
    model.aux_hidden_state_layers = []
    model.use_sparse_mhc = False
    model.use_spec_layer = False
    model.has_engram = False

    monkeypatch.setattr(v2_model, "collapse", lambda h, pm: h)
    monkeypatch.setattr(v2_model, "post", lambda h, *a: h)
    monkeypatch.setattr(v2_model.upstream, "sp_shard", lambda x: x)
    monkeypatch.setattr(v2_model.upstream, "sp_all_gather", lambda x: x)
    monkeypatch.setattr(
        v2_model.upstream, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(
        v2_model.upstream,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        v2_model.upstream, "is_forward_context_available", lambda: False
    )

    ids = torch.tensor([1, 2], dtype=torch.int32)
    pos = torch.tensor([0, 1], dtype=torch.int32)
    embeds = torch.randn(2, 64)

    # 1. stage_sp = True
    v2_model.DeepseekV41Model._forward_stage(
        model, ids, pos, None, inputs_embeds=embeds, stage="encoder"
    )
    assert len(written_calls) == 1
    assert written_calls[0]["sp_sharded"] is True

    # 2. stage_sp = False
    written_calls.clear()
    model._stage_uses_sequence_parallel = lambda stage: False
    v2_model.DeepseekV41Model._forward_stage(
        model, ids, pos, None, inputs_embeds=embeds, stage="encoder"
    )
    assert len(written_calls) == 1
    assert written_calls[0]["sp_sharded"] is False


def test_model_init_ced_compile_sizes_and_parallel_config(monkeypatch):
    """Test DeepseekV41Model.__init__ CED compile sizes and CP validation."""
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_CED", True)
    monkeypatch.setattr(v2_model, "initialize_cache_ops", lambda: None)
    monkeypatch.setattr(v2_model, "initialize_delayed_mhc", lambda: None)
    monkeypatch.setattr(v2_model, "initialize_mhc_post", lambda: None)
    monkeypatch.setattr(v2_model, "initialize_paged_attention", lambda: None)
    monkeypatch.setattr(v2_model, "initialize_indexer", lambda: None)
    monkeypatch.setattr(v2_model, "compact_rope_caches", lambda *a, **k: None)
    monkeypatch.setattr(v2_model, "DeepseekV41EncoderStage", MagicMock())
    monkeypatch.setattr(v2_model, "DeepseekV41DecoderStage", MagicMock())
    monkeypatch.setattr(v2_model, "DeepseekV41FullStage", MagicMock())

    def fake_upstream_model_init(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        self.use_sequence_parallel = False
        self.config = SimpleNamespace(num_hidden_layers=10)
        self.end_layer = 10
        self.layers = [
            SimpleNamespace(
                engram=None,
                attn=SimpleNamespace(
                    window_size=64,
                    compressor=SimpleNamespace() if i == 5 else None,
                ),
            )
            for i in range(10)
        ]

    monkeypatch.setattr(
        v2_model.upstream.DeepseekV4Model, "__init__", fake_upstream_model_init
    )

    # 1. compile_sizes filters ints, CP size != 1 raises NotImplementedError
    vllm_cfg_invalid_cp = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=10),
            max_model_len=4096,
        ),
        compilation_config=SimpleNamespace(compile_sizes=[64, "not_an_int", 128]),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
    )
    with pytest.raises(
        NotImplementedError, match="CED bounded replay requires unsharded token rows"
    ):
        v2_model.DeepseekV41Model(vllm_config=vllm_cfg_invalid_cp)

    # 2. compile_sizes None, prefill_context_parallel_size == 1 succeeds
    vllm_cfg_valid = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=10),
            max_model_len=4096,
        ),
        compilation_config=SimpleNamespace(compile_sizes=None),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
    )
    model = v2_model.DeepseekV41Model(vllm_config=vllm_cfg_valid)
    assert model._ced_compile_sizes == ()
    assert model._ced_enabled is True
    assert model._ced_window == 64

    # 3. CED disabled
    monkeypatch.setattr(envs, "TPU_DEEPSEEK_V41_CED", False)
    model_no_ced = v2_model.DeepseekV41Model(vllm_config=vllm_cfg_invalid_cp)
    assert model_no_ced._ced_compile_sizes == (64, 128)
    assert model_no_ced._ced_enabled is False


# ==============================================================================
# 2. Tests for v2/attention.py: forward & _sp_sharded_input branches
# ==============================================================================


def _create_mock_attention():
    """Builds a bare DeepseekV41Attention instance with mock components."""
    attn = object.__new__(v2_attn.DeepseekV41Attention)
    nn.Module.__init__(attn)
    attn.padded_heads = 16
    attn.n_local_heads = 16
    attn.head_dim = 128
    attn.q_lora_rank = 512
    attn.compress_ratio = 2
    attn.candidate_block_size = 16
    attn._max_model_len = 4096
    attn._fp8_kv = False
    attn._config = {"index_topk": 2048, "candidate_topk_blocks": 64}
    attn._sp_sharded_input = False
    attn._ced_global_from_encoder = False
    attn._prepare_and_attn_fn = MagicMock()
    attn._o_proj = MagicMock(side_effect=lambda o, pos: o)
    attn.compressor = SimpleNamespace(project=lambda x: x[..., :64])
    attn.indexer = SimpleNamespace(
        n_head=64,
        head_dim=128,
        weights_proj=lambda x: x[..., :64],
        wq_b=lambda x: x,
        indexer_op=SimpleNamespace(
            candidate_blocks=torch.zeros(4, 4), candidate_write=False
        ),
    )
    attn._fused_wqa_wkv_gemm = MagicMock(
        side_effect=lambda x: torch.cat((x[..., :512], x[..., :128]), dim=-1)
    )
    attn.q_norm = MagicMock(side_effect=lambda x: x)
    attn.kv_norm = MagicMock(side_effect=lambda x: x)
    attn.kv_cache = torch.zeros(10, 10)
    attn.candidate_block_buffer = torch.zeros(100, 16, dtype=torch.int32)
    attn.topk_indices_buffer = torch.zeros(100, 128, dtype=torch.int32)
    return attn


def test_attention_init_executes_prepare_fn(monkeypatch):
    """Test DeepseekV41Attention.__init__ binds _prepare_and_attn_fn (line 424)."""

    def fake_upstream_attn_init(self, vllm_config, *args, **kwargs):
        nn.Module.__init__(self)
        self.kv_cache_torch_dtype = torch.bfloat16
        self.rotary_emb = SimpleNamespace(cos_sin_cache=torch.zeros(10, 10))
        self.attn_sink = torch.zeros(10)
        self._swa_max_requests = 16

    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention, "__init__", fake_upstream_attn_init
    )
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)

    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(to_dict=lambda: {"index_topk": 2048}),
            max_model_len=4096,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    attn = v2_attn.DeepseekV41Attention(vllm_cfg)
    assert attn._prepare_and_attn_fn == attn._prepare_tpu


def test_attention_get_kv_cache_spec(monkeypatch):
    """Test get_kv_cache_spec for None, non-FP8, and FP8 replacement."""
    attn = _create_mock_attention()

    # 1. Spec is None -> returns None
    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention,
        "get_kv_cache_spec",
        lambda self, cfg: None,
    )
    assert attn.get_kv_cache_spec(None) is None

    # 2. Spec is present but _fp8_kv is False -> returns unmodified spec
    fake_spec = SimpleNamespace(dtype=torch.bfloat16, page_size_padded=128)
    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention,
        "get_kv_cache_spec",
        lambda self, cfg: fake_spec,
    )
    attn._fp8_kv = False
    assert attn.get_kv_cache_spec(None) is fake_spec

    # 3. Spec is present and _fp8_kv is True -> returns replaced spec
    @dataclasses.dataclass
    class FakeSpecCls:
        dtype: torch.dtype
        page_size_padded: int | None

    real_fake_spec = FakeSpecCls(dtype=torch.bfloat16, page_size_padded=128)
    monkeypatch.setattr(
        v2_attn.upstream.DeepseekV4Attention,
        "get_kv_cache_spec",
        lambda self, cfg: real_fake_spec,
    )
    attn._fp8_kv = True
    replaced = attn.get_kv_cache_spec(None)
    assert replaced.dtype == torch.uint8
    assert replaced.page_size_padded is None


def test_attention_forward_unsharded():
    """Test DeepseekV41Attention.forward when _sp_sharded_input is False."""
    attn = _create_mock_attention()
    attn._sp_sharded_input = False
    attn.compress_ratio = 0  # _all_indices_fit = False

    hidden_states = torch.randn(8, 1024)
    positions = torch.arange(8)

    out = v2_attn.DeepseekV41Attention.forward(attn, positions, hidden_states)
    assert out.shape == (8, 16, 128)
    attn._prepare_and_attn_fn.assert_called_once()
    call_args = attn._prepare_and_attn_fn.call_args[0]
    qr = call_args[1]
    kv = call_args[2]
    assert qr.shape == (8, 512)
    assert kv.shape == (8, 128)


def test_attention_forward_sp_sharded_with_indexer_weights(monkeypatch):
    """Test forward when _sp_sharded_input is True with indexer weights."""
    attn = _create_mock_attention()
    attn._sp_sharded_input = True
    attn.compress_ratio = 0  # _all_indices_fit = False

    gathered = []

    def fake_sp_all_gather(tensor):
        gathered.append(tensor)
        return torch.cat((tensor, tensor), dim=0)

    monkeypatch.setattr(v2_attn, "sp_all_gather", fake_sp_all_gather)

    hidden_states = torch.randn(4, 1024)
    positions = torch.arange(4)

    out = v2_attn.DeepseekV41Attention.forward(attn, positions, hidden_states)
    assert out.shape == (4, 16, 128)
    # 2 gathers: packed (qr, kv, indexer_weights) and kv_score
    assert len(gathered) == 2
    attn._prepare_and_attn_fn.assert_called_once()


def test_attention_forward_sp_sharded_without_indexer_weights_or_kv_score(
    monkeypatch,
):
    """Test forward when _sp_sharded_input is True without extra weights."""
    attn = _create_mock_attention()
    attn._sp_sharded_input = True
    attn.compressor = None
    attn.indexer = None

    gathered = []

    def fake_sp_all_gather(tensor):
        gathered.append(tensor)
        return torch.cat((tensor, tensor), dim=0)

    monkeypatch.setattr(v2_attn, "sp_all_gather", fake_sp_all_gather)

    hidden_states = torch.randn(4, 1024)
    positions = torch.arange(4)

    out = v2_attn.DeepseekV41Attention.forward(attn, positions, hidden_states)
    assert out.shape == (4, 16, 128)
    # Only 1 gather: packed (qr, kv)
    assert len(gathered) == 1
    attn._prepare_and_attn_fn.assert_called_once()


def test_run_parallel_input_projections_and_all_indices_fit():
    """Test _run_parallel_input_projections and _all_indices_fit branches."""
    attn = _create_mock_attention()

    # 1. compressor is not None and not _ced_global_from_encoder
    #    indexer is not None and not _all_indices_fit
    attn.compress_ratio = 2
    attn._max_model_len = 8192
    attn._config = {"index_topk": 2048}  # 8192 // 2 = 4096 > 2048 -> fit=False
    attn._ced_global_from_encoder = False

    x = torch.randn(2, 64)
    qr_kv, kv_score, idx_w = attn._run_parallel_input_projections(x)
    assert kv_score is not None
    assert idx_w is not None

    # 2. _ced_global_from_encoder is True -> kv_score is None
    attn._ced_global_from_encoder = True
    _, kv_score2, _ = attn._run_parallel_input_projections(x)
    assert kv_score2 is None

    # 3. _all_indices_fit is True -> idx_w is None
    attn._ced_global_from_encoder = False
    attn._config = {"index_topk": 5000}  # 4096 <= 5000 -> fit=True
    _, _, idx_w2 = attn._run_parallel_input_projections(x)
    assert idx_w2 is None

    # 4. compressor is None and indexer is None
    attn.compressor = None
    attn.indexer = None
    _, kv_score3, idx_w3 = attn._run_parallel_input_projections(x)
    assert kv_score3 is None
    assert idx_w3 is None

    # 5. compress_ratio is 0 -> _all_indices_fit is False
    attn.compress_ratio = 0
    assert attn._all_indices_fit is False


def test_attention_quantize_and_project_index_q():
    """Test _quantize_indexer and _project_index_q."""
    attn = _create_mock_attention()
    attn.indexer.head_dim = 16
    attn.indexer.n_head = 2
    attn._rotate = lambda q, pos: q

    qr = torch.randn(4, 32)
    positions = torch.arange(4)
    projected = attn._project_index_q(qr, positions)
    assert projected.shape == (4, 2, 16)


def test_write_encoder_global_kv_branches(monkeypatch):
    """Test write_encoder_global_kv early returns and sp_sharded gather."""
    attn = _create_mock_attention()
    written_calls: list[dict] = []
    attn._write_global_kv = lambda kv, pos, defer_index_write=False: (
        written_calls.append({"kv": kv, "pos": pos, "defer": defer_index_write})
    )

    # 1. Metadata not dict -> early return
    forward_ctx = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)
    attn.write_encoder_global_kv(torch.randn(2, 64), torch.arange(2))
    assert len(written_calls) == 0

    # 2. kv_cache has 0 numel -> early return
    forward_ctx.attn_metadata = {}
    attn.kv_cache = torch.zeros(0)
    attn.write_encoder_global_kv(torch.randn(2, 64), torch.arange(2))
    assert len(written_calls) == 0

    # 3. sp_sharded = False -> normal write
    attn.kv_cache = torch.zeros(10, 10)
    pos = torch.arange(2)
    hidden = torch.randn(2, 64)
    attn.write_encoder_global_kv(hidden, pos, sp_sharded=False)
    assert len(written_calls) == 1

    # 4. sp_sharded = True -> gathers kv_score
    written_calls.clear()
    gathered = []
    monkeypatch.setattr(
        v2_attn, "sp_all_gather", lambda t: (gathered.append(t), t.repeat(2, 1))[1]
    )
    attn.write_encoder_global_kv(hidden, pos, sp_sharded=True)
    assert len(gathered) == 1
    assert len(written_calls) == 1


# ==============================================================================
# 3. Tests for v2/attention.py: _prepare_tpu & _select_paged_indices
# ==============================================================================


def test_prepare_tpu_branches(monkeypatch):
    """Test _prepare_tpu indexer routing, tiled/untiled, fp8/bf16, and CED."""
    attn = _create_mock_attention()
    count = 2
    capacity = 4
    attn.compress_ratio = 1
    attn._max_model_len = capacity
    attn._fp8_kv = True
    attn.n_local_heads = 2
    attn.head_dim = 16
    attn.scale = 0.25
    attn._ced_decoder = False
    attn._ced_global_from_encoder = False
    attn.compressed_cache_prefix = "comp_kv"
    attn.topk_indices_buffer = torch.zeros((count, 8), dtype=torch.int32)
    attn.attn_sink = torch.zeros(4, dtype=torch.float32)
    attn._precompact_records = False
    attn._token_parallel_attn = False
    attn._full_attn_sink = torch.zeros(4)
    attn._tp_rank_1d = torch.zeros(1, dtype=torch.int32)
    attn._swa_max_requests = 16
    attn.candidate_block_buffer = torch.zeros((count, 4), dtype=torch.int32)

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

    # 1. Tiled branch (window_size % 64 == 0) with FP8 -> sparse_fp8_attention
    attn.window_size = 64
    fake_swa_cache = MagicMock(spec=PagedCacheRows)
    fake_swa_cache.numel.return_value = 100
    fake_swa_cache.row_indices = lambda slots: slots.int()
    attn.swa_cache_layer = SimpleNamespace(prefix="swa_kv", kv_cache=fake_swa_cache)

    fake_comp_cache = MagicMock(spec=PagedCacheRows)
    fake_comp_cache.row_indices = lambda slots: slots.int()
    fake_comp_cache.pages = lambda table: torch.zeros((2, 4), dtype=torch.int32)
    attn._static_forward_context = {
        "comp_kv": SimpleNamespace(kv_cache=fake_comp_cache)
    }

    metadata = {
        "swa_kv": SimpleNamespace(block_table=torch.zeros((2, 4), dtype=torch.int32)),
        "comp_kv": SimpleNamespace(
            block_table=torch.zeros((2, 4), dtype=torch.int32),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        ),
        "idx_k": SimpleNamespace(block_table=torch.zeros((2, 4), dtype=torch.int32)),
    }

    forward_ctx = SimpleNamespace(
        attn_metadata=metadata,
        additional_kwargs={
            "deepseek_v41_replay_starts": torch.tensor([0, 0], dtype=torch.int32)
        },
    )
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

    sparse_fp8_called = []
    monkeypatch.setattr(
        v2_attn,
        "sparse_fp8_attention",
        lambda *args, **kwargs: (
            sparse_fp8_called.append(True),
            torch.zeros(count, 2, 16),
        )[1],
    )
    paged_attn_called = []
    monkeypatch.setattr(
        v2_attn,
        "paged_attention",
        lambda *args, **kwargs: (
            paged_attn_called.append(True),
            torch.zeros(count, 2, 16),
        )[1],
    )

    attn.wq_b = lambda qr: qr
    attn._rotate = lambda x, pos: x
    attn._quantize_indexer = lambda x: x
    attn._write_global_kv = MagicMock(return_value=None)

    def fake_select_paged(
        qr, pos, cache, records, reqs, weights, lengths, slots, pending
    ):
        return torch.zeros((count, 2), dtype=torch.int32)

    monkeypatch.setattr(attn, "_select_paged_indices", fake_select_paged)
    monkeypatch.setattr(
        v2_attn.DeepseekV41Attention,
        "_all_indices_fit",
        property(lambda self: False),
    )

    qr = torch.randn(count, 32)
    kv = torch.randn(count, 16)
    output = torch.zeros(count, 2, 16)

    # 1a. Tiled + FP8 + ced_decoder=True
    attn._ced_decoder = True
    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn,
        None,
        qr,
        kv,
        None,
        torch.zeros(count, 64),
        torch.ones(count, 2),
        positions,
        output,
    )
    assert len(sparse_fp8_called) == 1

    # 1b. Tiled + BF16 (pages is None) -> calls paged_attention
    attn._fp8_kv = False
    attn._ced_decoder = False
    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn,
        None,
        qr,
        kv,
        None,
        torch.zeros(count, 64),
        torch.ones(count, 2),
        positions,
        output,
    )
    assert len(paged_attn_called) == 1

    # 2. Untiled branch (window_size = 10, not % 64), BF16 -> sparse_attention
    attn.window_size = 10
    attn._fp8_kv = False
    project_index_q_called = []

    def fake_project_index_q(qr, pos):
        project_index_q_called.append(True)
        return torch.zeros(count, 2, 16)

    monkeypatch.setattr(attn, "_project_index_q", fake_project_index_q)
    monkeypatch.setattr(
        v2_attn,
        "_read_rows",
        lambda cache, table, reqs, log: torch.zeros(count, log.shape[-1], 16),
    )
    monkeypatch.setattr(
        attn,
        "_select_indices",
        lambda scores, lengths, logical, cnt: torch.zeros((cnt, 2), dtype=torch.int32),
    )
    sparse_attn_called = []
    monkeypatch.setattr(
        v2_attn,
        "sparse_attention",
        lambda *args, **kwargs: (
            sparse_attn_called.append(True),
            torch.zeros(count, 1, 2, 16),
        )[1],
    )

    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn,
        None,
        qr,
        kv,
        None,
        torch.zeros(count, 64),
        torch.ones(count, 2),
        positions,
        output,
    )
    assert len(project_index_q_called) == 1
    assert len(sparse_attn_called) == 1

    # 3. Untiled branch with FP8 and compress_ratio > 0 -> NotImplementedError
    attn._fp8_kv = True
    with pytest.raises(
        NotImplementedError, match="FP8 V4.1 KV cache requires the tiled path"
    ):
        v2_attn.DeepseekV41Attention._prepare_tpu(
            attn,
            None,
            qr,
            kv,
            None,
            torch.zeros(count, 64),
            torch.ones(count, 2),
            positions,
            output,
        )

    # 4. _all_indices_fit = True (tiled path)
    attn.window_size = 64
    monkeypatch.setattr(
        v2_attn.DeepseekV41Attention,
        "_all_indices_fit",
        property(lambda self: True),
    )
    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn,
        None,
        qr,
        kv,
        None,
        torch.zeros(count, 64),
        torch.ones(count, 2),
        positions,
        output,
    )

    # 5. compressor is None, compress_ratio = 0 -> covers arc 631->633, arc 660->767
    attn.compressor = None
    attn.compress_ratio = 0
    attn.window_size = 64
    attn._fp8_kv = False
    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn,
        None,
        qr,
        kv,
        None,
        torch.zeros(count, 64),
        None,
        positions,
        output,
    )

    # 6. indexer is None, compress_ratio = 1 -> covers arc 665->733
    attn.compress_ratio = 1
    attn.indexer = None
    v2_attn.DeepseekV41Attention._prepare_tpu(
        attn,
        None,
        qr,
        kv,
        None,
        torch.zeros(count, 64),
        None,
        positions,
        output,
    )


def test_select_paged_indices_split_branches(monkeypatch):
    """Test _select_paged_indices under TP > 1 (split=True) and TP = 1 (split=False)."""
    attn = _create_mock_attention()
    attn._tp_rank = torch.tensor(0, dtype=torch.int32)
    attn.candidate_block_size = 16
    attn._fp8_kv = True
    attn.indexer = SimpleNamespace(
        head_dim=128,
        n_head=2,
        indexer_op=SimpleNamespace(
            candidate_blocks=torch.zeros(4, 4), candidate_write=True
        ),
    )
    attn._project_index_q = MagicMock(return_value=torch.randn(32, 2, 128))
    attn.candidate_block_buffer = torch.zeros(32, 16, dtype=torch.int32)

    fake_paged_indexer = MagicMock(
        return_value=(
            torch.zeros(10, 128),
            torch.zeros(32, 32),
            torch.zeros(32, 16, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(v2_attn, "paged_indexer", fake_paged_indexer)
    monkeypatch.setattr(v2_attn, "tensor_model_parallel_all_gather", lambda x, dim=0: x)

    qr = torch.randn(4, 512)
    positions = torch.arange(4)
    index_cache = SimpleNamespace(raw=torch.zeros(10, 128))
    records = torch.zeros(4, 128)
    requests = torch.arange(4)
    weights = torch.ones(4, 2)
    lengths = torch.full((4, 1), 10)
    swa_slots = torch.arange(4)

    # 1. TP = 2 (split = True), candidate_write = True, pending_write is not None
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 2)
    pending_write = (torch.tensor([0]), torch.randn(1, 128))
    selected = v2_attn.DeepseekV41Attention._select_paged_indices(
        attn,
        qr,
        positions,
        index_cache,
        records,
        requests,
        weights,
        lengths,
        swa_slots,
        pending_write,
    )
    assert selected.shape == (4, 32)
    attn._project_index_q.assert_called()

    # 2. TP = 2 (split = True), candidate_write = False, pending_write is None
    attn.indexer.indexer_op.candidate_write = False
    attn._fp8_kv = False
    selected2 = v2_attn.DeepseekV41Attention._select_paged_indices(
        attn,
        qr,
        positions,
        index_cache,
        records,
        requests,
        weights,
        lengths,
        swa_slots,
        None,
    )
    assert selected2.shape == (4, 32)

    # 3. TP = 1 (split = False), candidate_blocks = None (no candidates)
    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: 1)
    attn.indexer.indexer_op.candidate_blocks = None
    attn.indexer.indexer_op.candidate_write = False
    attn._project_index_q = MagicMock(return_value=torch.randn(4, 2, 128))
    fake_paged_indexer.return_value = (
        torch.zeros(10, 128),
        torch.zeros(4, 32),
        torch.zeros(4, 0, dtype=torch.int32),
    )
    selected_single = v2_attn.DeepseekV41Attention._select_paged_indices(
        attn,
        qr,
        positions,
        index_cache,
        records,
        requests,
        weights,
        lengths,
        swa_slots,
        None,
    )
    assert selected_single.shape == (4, 32)
