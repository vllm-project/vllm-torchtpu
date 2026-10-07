# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DeepSeek-V4.1 v2 DSpark speculative decoding."""

import unittest.mock
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    attention as v2_attn,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    attention_kernels as v2_kernels,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    dspark as v2_dspark,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import (
    model as v2_model,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows


def test_dspark_model_registration():
    with unittest.mock.patch(
        "vllm.model_executor.models.ModelRegistry.register_model"
    ) as mock_reg:
        register_models()
        calls = {call.args[0]: call.args[1] for call in mock_reg.call_args_list}
        assert "DSparkV41DraftModel" in calls
        assert calls["DSparkV41DraftModel"] == (
            "vllm_torchtpu.models.vllm.deepseek_v41.v2.dspark:"
            "DSparkDeepseekV41ForCausalLM"
        )


def test_get_dspark_swa_index_width():
    assert v2_dspark.get_dspark_swa_index_width(512, 16) == 576
    assert v2_dspark.get_dspark_swa_index_width(512, 64) == 576
    assert v2_dspark.get_dspark_swa_index_width(512, 65) == 640


def test_remap_dspark_name():
    assert v2_dspark._remap_dspark_name("model.layers.0.attn.wo.weight") is None
    assert (
        v2_dspark._remap_dspark_name(
            "mtp.0.confidence_head.w1.weight", enable_confidence_head=False
        )
        is None
    )
    assert (
        v2_dspark._remap_dspark_name(
            "mtp.0.confidence_head.w1.weight", enable_confidence_head=True
        )
        == "model.confidence_head.w1.weight"
    )
    assert (
        v2_dspark._remap_dspark_name("mtp.0.markov_head.embed.weight")
        == "model.markov_head.markov_w1.weight"
    )
    assert (
        v2_dspark._remap_dspark_name("mtp.0.markov_head.head.weight")
        == "model.markov_head.markov_w2.weight"
    )
    assert (
        v2_dspark._remap_dspark_name("mtp.0.markov_head.other.weight")
        == "model.markov_head.other.weight"
    )
    assert (
        v2_dspark._remap_dspark_name("mtp.0.main_proj.weight")
        == "model.main_proj.weight"
    )
    assert (
        v2_dspark._remap_dspark_name("mtp.0.main_norm.weight")
        == "model.main_norm.weight"
    )
    assert v2_dspark._remap_dspark_name("mtp.2.norm.weight") == "model.norm.weight"
    assert (
        v2_dspark._remap_dspark_name("mtp.1.ffn.gate.bias_vl")
        == "model.layers.1.ffn.gate.bias"
    )
    assert (
        v2_dspark._remap_dspark_name("mtp.2.attn.wo.weight")
        == "model.layers.2.attn.wo.weight"
    )


def test_insert_context_kv(monkeypatch: pytest.MonkeyPatch):
    inserted: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def fake_insert(cache, slots, kv, valid):
        inserted.append((slots.clone(), kv.clone(), valid.clone()))

    monkeypatch.setattr(v2_dspark, "_insert", fake_insert)

    attn = SimpleNamespace(
        window_size=4,
        swa_cache_layer=SimpleNamespace(kv_cache=torch.empty(0)),
        _rotate=lambda kv, pos: kv + pos.unsqueeze(-1),
    )
    kv = torch.ones((3, 8))
    positions = torch.tensor([0, 1, 2], dtype=torch.long)

    # 1. Empty cache or metadata=None -> early return
    v2_dspark._insert_context_kv(attn, kv, positions, None)
    assert len(inserted) == 0

    attn.swa_cache_layer.kv_cache = torch.zeros((2, 4, 1, 8))
    v2_dspark._insert_context_kv(attn, kv, positions, None)
    assert len(inserted) == 0

    # 2. Tensor metadata (direct slot mapping)
    slot_tensor = torch.tensor([2, -1, 5], dtype=torch.long)
    v2_dspark._insert_context_kv(attn, kv, positions, slot_tensor)
    assert len(inserted) == 1
    assert torch.equal(inserted[0][0], slot_tensor)
    assert torch.equal(inserted[0][2], torch.tensor([True, False, True]))

    # 3. Object metadata with non-empty seq_lens
    inserted.clear()
    monkeypatch.setattr(
        v2_dspark, "_cache_slots", lambda cache, md, pos: torch.tensor([1, 2, 3])
    )
    monkeypatch.setattr(
        v2_dspark,
        "_token_requests",
        lambda md, pos: torch.tensor([0, 0, 0], dtype=torch.long),
    )
    md_nonempty = SimpleNamespace(seq_lens=torch.tensor([6], dtype=torch.int32))
    pos2 = torch.tensor([1, 2, 5], dtype=torch.long)
    v2_dspark._insert_context_kv(attn, kv, pos2, md_nonempty)
    assert len(inserted) == 1
    assert torch.equal(inserted[0][2], torch.tensor([False, True, True]))

    # 4. Object metadata with empty seq_lens
    inserted.clear()
    md_empty = SimpleNamespace(seq_lens=torch.empty(0, dtype=torch.int32))
    v2_dspark._insert_context_kv(attn, kv, pos2, md_empty)
    assert len(inserted) == 1
    assert torch.equal(inserted[0][2], torch.tensor([True, True, True]))


def test_attend_window_width_over_512(monkeypatch: pytest.MonkeyPatch):
    attend_called = False

    def fake_attend(q, pool, records, state, scale):
        nonlocal attend_called
        attend_called = True
        return state

    monkeypatch.setattr(v2_kernels, "_attend", fake_attend)
    q = torch.zeros((128, 4, 16))
    pool = torch.zeros((4, 64, 1, 16))
    records = torch.zeros((128, 576), dtype=torch.int32)
    state = (
        torch.zeros((128, 4)),
        torch.zeros((128, 4)),
        torch.zeros((1, 128, 4, 16)),
    )
    out = v2_kernels._attend_window(q, pool, records, state, 1.0)
    assert attend_called is True
    assert out is state


def test_prepare_tpu_non_causal_dspark_swa(monkeypatch: pytest.MonkeyPatch):
    attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
    torch.nn.Module.__init__(attn)
    attn.prefix = "swa_pfx"
    attn.window_size = 512
    attn._ced_decoder = False
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
    attn.non_causal_multi_token_decode = True
    attn.dspark_block_size = 16

    cache_mock = MagicMock(spec=PagedCacheRows)
    cache_mock.numel.return_value = 1024
    cache_mock.shape = (64, 64, 512)
    cache_mock.page_stride = 64
    cache_mock.row_offset = 100
    cache_mock.row_indices = lambda slots: 100 + slots
    cache_mock.pages.return_value = torch.arange(16, dtype=torch.int32).unsqueeze(0)

    attn.swa_cache_layer = SimpleNamespace(prefix="swa_pfx", kv_cache=cache_mock)
    attn.kv_cache = cache_mock
    attn._static_forward_context = {"swa_pfx": SimpleNamespace(kv_cache=cache_mock)}
    attn.compressed_cache_prefix = "swa_pfx"

    # 4 tokens in a draft block at positions [20, 21, 22, 23] with seq_len=24
    positions = torch.tensor([20, 21, 22, 23], dtype=torch.long)
    swa_md = SimpleNamespace(
        seq_lens=torch.tensor([24], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        block_tables=torch.arange(16, dtype=torch.int32).unsqueeze(0),
        slot_mapping=torch.tensor([20, 21, 22, 23], dtype=torch.long),
    )
    ctx = SimpleNamespace(attn_metadata={"swa_pfx": swa_md}, additional_kwargs={})
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(v2_attn, "_insert", lambda *a, **kw: None)

    captured_records: list[torch.Tensor] = []

    def fake_sparse_fp8_attn(q, swa_c, main_c, records, main_rec, *args, **kwargs):
        captured_records.append(records.clone())
        return torch.zeros_like(q)

    monkeypatch.setattr(v2_attn, "sparse_fp8_attention", fake_sparse_fp8_attn)

    hidden = torch.zeros((4, 16))
    qr = torch.zeros((4, 16))
    kv = torch.zeros((4, 16))
    out = torch.zeros((4, 2, 8))

    # Path A: uncached (no stage_attention_cache_scope)
    attn._prepare_tpu(hidden, qr, kv, None, None, None, positions, out)
    assert len(captured_records) == 1
    rec_uncached = captured_records[0]
    assert rec_uncached.shape == (4, 576)
    # Token 0 (pos=20) attends up to pos=23 (end_pos=24) in non-causal mode!
    assert int(rec_uncached[0, 23].item()) >= 0
    assert int(rec_uncached[0, 24].item()) == -1

    # Path B: with stage_attention_cache_scope
    captured_records.clear()
    with v2_attn.stage_attention_cache_scope():
        attn._prepare_tpu(hidden, qr, kv, None, None, None, positions, out)
    assert len(captured_records) == 1
    rec_cached = captured_records[0]
    assert torch.equal(rec_cached, rec_uncached)


def test_ced_aux_hidden_states_propagation(monkeypatch: pytest.MonkeyPatch):
    # Verify _scatter_decoder_rows (with padded extra index at total_rows=4)
    src = torch.tensor([[10.0, 20.0], [30.0, 40.0], [99.0, 99.0]])
    idx_tensor = torch.tensor([1, 3, 4], dtype=torch.int64)
    scattered = v2_model._scatter_decoder_rows(src, idx_tensor, 4)
    expected = torch.tensor([[0.0, 0.0], [10.0, 20.0], [0.0, 0.0], [30.0, 40.0]])
    assert torch.equal(scattered, expected)

    # Verify _forward_stage with encoder (layer 1) + decoder (start_layer=2, layer 3)
    class _FakeStageLayer:
        def __init__(self, val: float):
            self.val = val
            self.attn = SimpleNamespace(
                write_encoder_global_kv=lambda norm, pos, sp_sharded=False: None
            )

        def attn_norm(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def __call__(
            self,
            hidden: torch.Tensor,
            pos: torch.Tensor,
            ids: torch.Tensor,
            pre: torch.Tensor | None,
            post_m: Any,
            res_m: Any,
            res: Any,
            eh: Any,
            em: Any,
            capture_previous_aux: bool = False,
        ):
            out_pre = (
                pre
                if pre is not None
                else torch.ones((hidden.shape[0], 2), dtype=torch.float32)
            )
            prev_aux = (
                torch.full((hidden.shape[0], hidden.shape[-1]), self.val)
                if capture_previous_aux
                else None
            )
            return hidden, hidden, None, None, out_pre, prev_aux

    monkeypatch.setattr(
        v2_model.upstream,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(v2_model, "post", lambda h, *a: h)
    monkeypatch.setattr(v2_model, "collapse", lambda h, pm: h.mean(dim=1))

    m = SimpleNamespace(
        _ced_boundary=2,
        start_layer=0,
        end_layer=4,
        _stage_uses_sequence_parallel=lambda stage: False,
        use_mega_moe=False,
        _has_cpu_offloaded_engram=lambda: False,
        _prepare_engram=lambda *a, **kw: (None, None),
        aux_hidden_state_layers=(1, 2, 3),
        layers=[_FakeStageLayer(float(i + 10)) for i in range(4)],
        norm=lambda x: x,
        _mtp_hidden_buffer=None,
        config=SimpleNamespace(hidden_size=4),
        _ced_enabled=True,
        _ced_window=512,
        _ced_compile_sizes=(2, 4),
    )

    ids = torch.arange(4, dtype=torch.int64)
    pos = torch.arange(4, dtype=torch.int64)
    embeds = torch.full((4, 2, 4), 20.0)

    enc_out = v2_model.DeepseekV41Model._forward_stage(
        m, ids, pos, None, inputs_embeds=embeds, stage="encoder"
    )
    assert len(enc_out) == 3
    h_enc, pm_enc, enc_aux = enc_out
    assert len(enc_aux) == 1
    assert torch.all(enc_aux[0] == 11.0)  # captured at idx=1

    dec_out = v2_model.DeepseekV41Model._forward_stage(
        m,
        ids,
        pos,
        None,
        inputs_embeds=h_enc,
        encoder_pre_mix=pm_enc,
        stage="decoder",
    )
    assert isinstance(dec_out, tuple) and len(dec_out) == 2
    h_dec, dec_aux = dec_out
    assert len(dec_aux) == 2
    assert torch.all(dec_aux[0] == 20.0)  # captured at start_layer=2
    assert torch.all(dec_aux[1] == 13.0)  # captured at idx=3

    # Verify 1-based aux_hidden_state_layers=(2, 3, 4) where 4 == end_layer:
    # encoder_stage must NOT capture end_layer=4; decoder_stage captures (2, 3, 4).
    m.aux_hidden_state_layers = (2, 3, 4)
    enc_out_1based = v2_model.DeepseekV41Model._forward_stage(
        m, ids, pos, None, inputs_embeds=embeds, stage="encoder"
    )
    assert len(enc_out_1based) == 2
    dec_out_1based = v2_model.DeepseekV41Model._forward_stage(
        m,
        ids,
        pos,
        None,
        inputs_embeds=enc_out_1based[0],
        encoder_pre_mix=enc_out_1based[1],
        stage="decoder",
    )
    assert isinstance(dec_out_1based, tuple) and len(dec_out_1based) == 2
    assert len(dec_out_1based[1]) == 3
    assert torch.allclose(
        v2_model._mean_hc_streams(embeds.to(torch.bfloat16)),
        embeds.to(torch.bfloat16).mean(dim=1),
    )
    m.aux_hidden_state_layers = (1, 2, 3)

    # Test DeepseekV41Model.forward CED=1 paths with aux_hidden_states
    @dataclass
    class _FakeMeta:
        seq_lens: torch.Tensor = field(
            default_factory=lambda: torch.tensor([4], dtype=torch.int32)
        )
        query_start_loc: torch.Tensor = field(
            default_factory=lambda: torch.tensor([0, 4], dtype=torch.int32)
        )
        input_positions: torch.Tensor = field(
            default_factory=lambda: torch.arange(4, dtype=torch.int64)
        )

    @dataclass
    class _FakeCtx:
        attn_metadata: Any = field(default_factory=lambda: {"swa": _FakeMeta()})
        is_padding: Any = None
        additional_kwargs: dict[str, Any] = field(default_factory=dict)

    monkeypatch.setattr(v2_model.upstream, "get_forward_context", lambda: _FakeCtx())
    monkeypatch.setattr(
        v2_model,
        "override_forward_context",
        lambda ctx: unittest.mock.MagicMock(
            __enter__=lambda s: None, __exit__=lambda s, *a: None
        ),
    )

    m.encoder_stage = lambda *a, **kw: enc_out
    m.decoder_stage = lambda *a, **kw: dec_out

    # 1. Decode (plan is None)
    monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *a: None)
    out_dec_full, aux_dec_full = v2_model.DeepseekV41Model.forward(m, ids, pos)
    assert out_dec_full.shape == (4, 4)
    assert len(aux_dec_full) == 3

    # 2. Bounded replay prefill (plan.indices = (1, 3), padded_num_tokens = 2)
    plan_bounded = SimpleNamespace(
        indices=(1, 3),
        padded_num_tokens=2,
        query_start_loc=(0, 2),
        replay_starts=(0,),
    )
    m.decoder_stage = lambda *a, **kw: (
        torch.full((2, 4), 9.0),
        [torch.full((2, 4), 20.0), torch.full((2, 4), 30.0)],
    )
    monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *a: plan_bounded)
    out_bounded, aux_bounded = v2_model.DeepseekV41Model.forward(m, ids, pos)
    assert out_bounded.shape == (4, 4)
    assert out_bounded[1, 0].item() == 9.0
    assert out_bounded[0, 0].item() == 0.0
    assert len(aux_bounded) == 3
    assert aux_bounded[0].shape == (4, 4)
    assert aux_bounded[1][3, 0].item() == 20.0
    assert aux_bounded[1][2, 0].item() == 0.0

    # 3. Incomplete chunked prefill (not plan.indices)
    plan_empty = SimpleNamespace(
        indices=(),
        padded_num_tokens=0,
        query_start_loc=(0, 0),
        replay_starts=(0,),
    )
    monkeypatch.setattr(v2_model, "plan_decoder_replay", lambda *a: plan_empty)
    out_empty, aux_empty = v2_model.DeepseekV41Model.forward(m, ids, pos)
    assert out_empty.shape == (4, 4)
    assert torch.all(out_empty == 0)
    assert len(aux_empty) == 3
    for a in aux_empty:
        assert a.shape == (4, 4)
        assert torch.all(a == 0)


class _FakeDraftAttn(nn.Module):
    def __init__(self, hidden_size: int, window_size: int = 512, prefix: str = ""):
        super().__init__()
        self.window_size = window_size
        self.non_causal_multi_token_decode = False
        self.dspark_block_size = 0
        self.swa_cache_layer = SimpleNamespace(
            prefix=f"{prefix}.swa_cache",
            kv_cache=torch.zeros((4, 16, 1, hidden_size)),
        )
        self.fused_wqa_wkv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_sink = nn.Parameter(torch.zeros(4))

    def _fused_wqa_wkv_gemm(self, x: torch.Tensor) -> torch.Tensor:
        return self.fused_wqa_wkv(x)

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return qr_kv, qr_kv, qr_kv

    def _rotate(self, kv: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return kv


class _FakeDraftFFN(nn.Module):
    def __init__(self, hidden_size: int, use_mega_moe: bool = False):
        super().__init__()
        self.use_mega_moe = use_mega_moe
        self.gate = nn.Module()
        self.gate.e_score_correction_bias = nn.Parameter(torch.zeros(4))
        self.shared_experts = nn.Module()
        self.shared_experts.gate_up_proj = nn.Parameter(
            torch.zeros(hidden_size, hidden_size)
        )
        self.shared_experts.down_proj = nn.Parameter(
            torch.zeros(hidden_size, hidden_size)
        )
        self.experts = nn.Module()
        self.experts.w13_weight = nn.Parameter(torch.zeros(2, hidden_size, hidden_size))
        self.experts.w2_weight = nn.Parameter(torch.zeros(2, hidden_size, hidden_size))
        self.experts.w13_weight_scale = nn.Parameter(
            torch.zeros(2, 4, dtype=torch.uint8), requires_grad=False
        )
        self.experts.w13_weight_scale_inv = nn.Parameter(
            torch.zeros(2, 4, dtype=torch.float32), requires_grad=False
        )
        self.finalized = False

    def finalize_mega_moe_weights(self) -> None:
        self.finalized = True


class _FakeDraftDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: Any,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
    ):
        super().__init__()
        hidden_size = 16
        self.attn = _FakeDraftAttn(
            hidden_size, window_size=512, prefix=f"{prefix}.attn"
        )
        self.ffn = _FakeDraftFFN(hidden_size)
        self.hc_attn_fn = nn.Parameter(
            torch.zeros((2, hidden_size), dtype=torch.float32)
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.zeros((2, hidden_size), dtype=torch.float32)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        pre_mix: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
    ):
        n, k, d = residual.shape if residual is not None else hidden_states.shape
        x = (
            hidden_states + 1.0
            if hidden_states.ndim == 2
            else hidden_states[:, 0] + 1.0
        )
        return (
            x,
            torch.zeros((n, k, d)),
            torch.ones((n, k, 1)),
            torch.eye(k).unsqueeze(0).expand(n, -1, -1),
            torch.full((n, k), 1.0 / k),
            None,
        )


def _make_dspark_vllm_config(
    *,
    num_speculative_tokens: int = 15,
    enable_confidence_head: bool = True,
    expert_dtype: str = "fp4",
    weight_block_size: list[int] | None = None,
) -> Any:
    hf_config = SimpleNamespace(
        hidden_size=16,
        hc_mult=2,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        num_hidden_layers=4,
        dspark_target_layer_ids=[1, 2, 3],
        n_mtp_layers=2,
        index_topk=4,
        dspark_block_size=num_speculative_tokens + 1,
        vocab_size=32,
        draft_vocab_size=32,
        dspark_markov_rank=8,
        enable_confidence_head=enable_confidence_head,
        dspark_n_routed_experts=2,
        n_routed_experts=2,
        num_attention_heads=4,
        expert_dtype=expert_dtype,
    )
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    quant_config = (
        SimpleNamespace(
            weight_block_size=weight_block_size,
            online_quantization_config=None,
            get_quant_method=lambda layer, prefix: UnquantizedLinearMethod(),
        )
        if weight_block_size is not None
        else None
    )
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=num_speculative_tokens,
            draft_model_config=SimpleNamespace(hf_config=hf_config),
        ),
        quant_config=quant_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        model_config=SimpleNamespace(max_model_len=1024),
    )


def _patch_dspark_env(monkeypatch: pytest.MonkeyPatch):
    fake_cur_cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            max_logprobs=20, logits_processor_pattern=None, head_dtype=None
        ),
        kernel_config=SimpleNamespace(linear_backend="auto"),
        compilation_config=SimpleNamespace(
            custom_ops=["none"],
            enabled_custom_ops=set(),
            disabled_custom_ops=set(),
        ),
    )
    monkeypatch.setattr("vllm.config.vllm._current_vllm_config", fake_cur_cfg)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        SimpleNamespace(
            world_size=1,
            rank_in_group=0,
            is_first_rank=True,
            is_last_rank=True,
            all_reduce=lambda x: x,
            gather=lambda x, dst=0, dim=-1: x,
        ),
    )
    monkeypatch.setattr(
        "vllm.model_executor.custom_op.get_cached_compilation_config",
        lambda: fake_cur_cfg.compilation_config,
    )
    monkeypatch.setattr(v2_dspark, "initialize_cache_ops", lambda: None)
    monkeypatch.setattr(v2_dspark, "initialize_delayed_mhc", lambda: None)
    monkeypatch.setattr(v2_dspark, "initialize_mhc_post", lambda: None)
    monkeypatch.setattr(v2_dspark, "initialize_paged_attention", lambda: None)
    monkeypatch.setattr(v2_dspark, "initialize_indexer", lambda: None)
    monkeypatch.setattr(v2_dspark, "compact_rope_caches", lambda *a: None)
    monkeypatch.setattr(v2_dspark, "get_current_vllm_config", lambda: SimpleNamespace())
    monkeypatch.setattr(v2_dspark, "DeepseekV41DecoderLayer", _FakeDraftDecoderLayer)
    monkeypatch.setattr(
        v2_dspark,
        "pack_mix_weights",
        lambda layers: [
            setattr(
                layer,
                "hc_attn_fn",
                nn.Parameter(layer.hc_attn_fn.to(torch.bfloat16), requires_grad=False),
            )
            for layer in layers
        ],
    )
    monkeypatch.setattr(v2_dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(v2_dspark, "get_tensor_model_parallel_rank", lambda: 0)


def test_dspark_deepseek_v41_for_causal_lm_forward_and_hooks(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_dspark_env(monkeypatch)
    vllm_cfg = _make_dspark_vllm_config()
    draft_lm = v2_dspark.DSparkDeepseekV41ForCausalLM(vllm_config=vllm_cfg)

    # Share embeddings & lm_head and initialize uninitialized linear weights
    draft_lm.model.embed_tokens = nn.Embedding(32, 16)
    draft_lm.lm_head = v2_dspark.ReplicatedLinear(16, 32, bias=False, return_bias=False)
    for p in draft_lm.parameters():
        if p.is_floating_point():
            p.data.normal_(0.0, 0.02)

    assert draft_lm.get_draft_attn_causal() == [False, False]
    assert len(draft_lm.get_draft_kv_cache_layer_names()) == 2
    assert len(draft_lm.model.layers) == 2
    for layer in draft_lm.model.layers:
        assert layer.attn.non_causal_multi_token_decode is True
        assert layer.attn.dspark_block_size == 16

    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    embeds = draft_lm.embed_input_ids(input_ids)
    assert embeds.shape == (4, 16)

    # Combine 3 target aux hidden states (4, 48) -> (4, 16)
    raw_aux = torch.randn(4, 48)
    combined = draft_lm.combine_hidden_states(raw_aux)
    assert combined.shape == (4, 16)

    # precompute_and_store_context_kv with raw (4, 48) and already-combined (4, 16)
    inserted_calls = []
    monkeypatch.setattr(
        v2_dspark,
        "_insert_context_kv",
        lambda attn, kv, pos, md: inserted_calls.append((kv.shape, md)),
    )
    main_x1 = draft_lm.precompute_and_store_context_kv(
        raw_aux, positions, metadata=None
    )
    assert main_x1.shape == (4, 16)
    assert len(inserted_calls) == 2
    assert inserted_calls[0][1] is None

    inserted_calls.clear()
    main_x2 = draft_lm.precompute_and_store_context_kv(
        combined, positions, metadata=["md0", "md1"]
    )
    assert torch.equal(main_x2, combined)
    assert [c[1] for c in inserted_calls] == ["md0", "md1"]

    # Forward both with inputs_embeds=None and inputs_embeds provided
    out_from_ids = draft_lm(input_ids, positions)
    out_from_embeds = draft_lm(input_ids, positions, inputs_embeds=embeds)
    assert out_from_ids.shape == (4, 16)
    assert torch.allclose(out_from_ids, out_from_embeds)

    # Logits, Markov head, and Confidence head
    logits = draft_lm.compute_draft_logits(out_from_ids)
    assert logits.shape == (4, 32)
    assert torch.equal(draft_lm.map_draft_to_target(input_ids), input_ids)

    m_embed = draft_lm.markov_embed(input_ids)
    assert m_embed.shape == (4, 8)
    m_bias = draft_lm.markov_bias(m_embed)
    assert m_bias.shape == (4, 32)
    conf = draft_lm.compute_confidence(out_from_ids, m_embed)
    assert conf.shape == (4,)
    assert torch.all((conf >= 0.0) & (conf <= 1.0))

    # process_weights_after_loading idempotency
    draft_lm.process_weights_after_loading()
    assert draft_lm.model.layers[0].hc_attn_fn.dtype == torch.bfloat16
    draft_lm.process_weights_after_loading()
    assert draft_lm.model.layers[0].hc_attn_fn.dtype == torch.bfloat16

    # Window size validation error when num_speculative_tokens + 1 > window_size
    bad_cfg = _make_dspark_vllm_config(num_speculative_tokens=600)
    with pytest.raises(ValueError, match="exceeds the DSV4.1 sliding window"):
        v2_dspark.DSparkDeepseekV41ForCausalLM(vllm_config=bad_cfg)


def test_dspark_load_weights(monkeypatch: pytest.MonkeyPatch):
    _patch_dspark_env(monkeypatch)
    vllm_cfg = _make_dspark_vllm_config(
        enable_confidence_head=True,
        expert_dtype="fp4",
        weight_block_size=[128, 128],
    )
    draft_lm = v2_dspark.DSparkDeepseekV41ForCausalLM(vllm_config=vllm_cfg)

    stacked_calls: list[tuple[str, int]] = []
    expert_calls: list[tuple[str, int, str]] = []

    def fake_stacked_loader(param, weight, shard_id):
        stacked_calls.append((str(shard_id), weight.numel()))

    def fake_expert_loader(
        param, weight, name_mapped, *, shard_id, expert_id, return_success=True
    ):
        expert_calls.append((name_mapped, expert_id, shard_id))
        return True

    layer0 = draft_lm.model.layers[0]
    layer0.attn.fused_wqa_wkv.weight.weight_loader = fake_stacked_loader  # type: ignore[attr-defined]
    layer0.ffn.shared_experts.gate_up_proj.weight_loader = fake_stacked_loader  # type: ignore[attr-defined]
    layer0.ffn.experts.w13_weight.weight_loader = fake_expert_loader  # type: ignore[attr-defined]
    layer0.ffn.experts.w13_weight_scale.weight_loader = fake_expert_loader  # type: ignore[attr-defined]
    draft_lm.model.main_proj.weight_scale_inv = nn.Parameter(
        torch.zeros(1), requires_grad=False
    )

    monkeypatch.setattr(
        v2_dspark.upstream.DeepseekV4Model,
        "_pad_shared_expert_weight",
        staticmethod(lambda qcfg, name, w: w),
    )
    monkeypatch.setattr(
        v2_dspark,
        "fused_moe_make_expert_params_mapping",
        lambda *a, **kw: [
            ("experts.w13_weight", "experts.0.w1.weight", 0, "w1"),
            ("experts.w13_weight_scale", "experts.0.w1.weight_scale", 0, "w1"),
        ],
    )

    weights = [
        ("model.layers.0.attn.wo.weight", torch.zeros(16, 16)),  # non-mtp skipped
        ("mtp.0.main_proj.scale", torch.ones(1)),  # linear .scale -> .weight_scale_inv
        ("mtp.0.main_norm.weight", torch.ones(16)),
        ("mtp.0.ffn.shared_experts.w2", torch.zeros(16, 16)),
        ("mtp.0.ffn.shared_experts.w1", torch.zeros(16, 16)),
        ("mtp.0.attn.wq_a.weight", torch.zeros(16, 16)),
        ("mtp.0.attn.attn_sink", torch.ones(4)),
        ("mtp.0.ffn.gate.bias_vl", torch.ones(4)),
        ("mtp.0.ffn.experts.0.w1.weight", torch.zeros(16, 16)),
        (
            "mtp.0.ffn.experts.0.w1.scale",
            torch.ones((2, 4), dtype=torch.float8_e8m0fnu),
        ),
    ]
    loaded = draft_lm.load_weights(weights)
    assert "model.main_proj.weight_scale_inv" in loaded
    assert "model.main_norm.weight" in loaded
    assert "model.layers.0.ffn.shared_experts.down_proj" in loaded
    assert "model.layers.0.ffn.shared_experts.gate_up_proj" in loaded
    assert "model.layers.0.attn.fused_wqa_wkv.weight" in loaded
    assert "model.layers.0.attn.attn_sink" in loaded
    assert "model.layers.0.ffn.gate.e_score_correction_bias" in loaded
    assert "model.layers.0.ffn.experts.w13_weight" in loaded
    assert "model.layers.0.ffn.experts.w13_weight_scale" in loaded
    assert draft_lm.model.confidence_head is None

    # Test use_mega_moe=True and confidence_head loaded
    vllm_cfg2 = _make_dspark_vllm_config(
        enable_confidence_head=True, expert_dtype="fp8"
    )
    draft_lm2 = v2_dspark.DSparkDeepseekV41ForCausalLM(vllm_config=vllm_cfg2)
    draft_lm2.model.layers[0].ffn.use_mega_moe = True
    monkeypatch.setattr(
        v2_dspark.upstream,
        "make_deepseek_v4_expert_params_mapping",
        lambda n: [
            ("experts.w13_weight_scale_inv", "experts.0.w1.weight_scale_inv", 0, "w1")
        ],
    )
    draft_lm2.model.layers[0].ffn.experts.w13_weight_scale_inv.weight_loader = (  # type: ignore[attr-defined]
        fake_expert_loader
    )
    loaded2 = draft_lm2.load_weights(
        [
            ("mtp.0.confidence_head.proj.weight", torch.zeros(1, 24)),
            ("mtp.0.ffn.experts.0.w1.scale", torch.ones(2, 4)),
        ]
    )
    assert "model.confidence_head.proj.weight" in loaded2
    assert "model.layers.0.ffn.experts.w13_weight_scale_inv" in loaded2
    assert draft_lm2.model.confidence_head is not None
