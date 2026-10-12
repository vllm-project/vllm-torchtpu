# SPDX-License-Identifier: Apache-2.0
"""CPU checks for TPU V4.1 Engram sequence-parallel row selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as v2_engram


@pytest.mark.cpu_test
@pytest.mark.parametrize("tp", [1, 2, 8])
def test_fused_offload_preserves_packed_shards_and_sp_mask(monkeypatch, tp):
    from vllm_torchtpu.kernels.engram import vllm_adapter

    local_tokens, cols, head_dim = 4, 3, 256
    tokens = local_tokens * tp
    raw = torch.randint(0, 256, (tp, tokens, cols, head_dim), dtype=torch.uint8)
    scale = torch.randint(
        120, 128, (tp, tokens, cols, head_dim // 32), dtype=torch.uint8
    )
    packed = torch.stack(
        [
            torch.cat([r.flatten(), s.flatten()]).view(torch.int32).view(tokens, -1)
            for r, s in zip(raw, scale)
        ]
    )
    rank = tp - 1
    monkeypatch.setattr(
        v2_engram.upstream, "get_tensor_model_parallel_rank", lambda: rank
    )

    def gather(value, dim):
        assert dim == 0
        torch.testing.assert_close(value, packed[rank])
        return packed.flatten(0, 1)

    monkeypatch.setattr(v2_engram.upstream, "tensor_model_parallel_all_gather", gather)
    module = object.__new__(v2_engram.Engram)
    torch.nn.Module.__init__(module)
    module._fused_offload = True
    module.use_sequence_parallel = True
    module.embed_tokens = SimpleNamespace(tp_size=tp)
    module._active_raw = packed[rank]
    module._host_raw = raw[rank]
    module._host_scale = scale[rank]
    module.wkv = SimpleNamespace(weight=torch.empty(1), weight_scale=torch.empty(1))
    module.q_weight = torch.empty(1)
    module.k_weight = torch.empty(1)
    module.eps, module.clamp_value = 1e-20, 1e-6
    hidden = torch.randn(local_tokens, 4, 8).bfloat16()
    mask = torch.arange(tokens) % 3 != 0
    calls = []

    def fused(raw_words, scale_words, h, w, ws, q, k, local_mask, size, r, eps, clamp):
        torch.testing.assert_close(
            raw_words, raw.view(torch.int32).reshape(tp * tokens, -1)
        )
        torch.testing.assert_close(
            scale_words, scale.view(torch.int32).reshape(tp * tokens, -1)
        )
        torch.testing.assert_close(local_mask, mask[rank * local_tokens :])
        assert size == tp and r == rank
        calls.append(True)
        return h + 1

    monkeypatch.setattr(vllm_adapter, "fused_engram_torch", fused)
    actual = module(hidden, torch.zeros(tokens, cols * tp, dtype=torch.int32), mask)
    torch.testing.assert_close(actual, hidden + 1)
    assert calls == [True]


@pytest.mark.parametrize("num_tokens", [16, 13, 3])
@pytest.mark.parametrize("tp", [1, 2, 8])
def test_sp_embed_matches_head_gather(monkeypatch, num_tokens, tp):
    local_heads, dim = 3, 4
    cols = tp * local_heads - 1  # trailing padded head is dropped
    table = torch.randn(num_tokens, tp * local_heads, dim)
    expected_full = table[:, :cols]  # non-SP: dim=1 head gather
    chunk = -(-num_tokens // tp)

    def rank_rows(r):
        return table[:, r * local_heads : (r + 1) * local_heads]

    def padded(r):
        rows = rank_rows(r)
        return torch.cat(
            [rows, rows.new_zeros(chunk * tp - num_tokens, *rows.shape[1:])]
        )

    for rank in range(tp):
        monkeypatch.setattr(
            v2_engram.upstream, "get_tensor_model_parallel_rank", lambda r=rank: r
        )

        def fake_gather(rows, dim, rank=rank):
            assert dim == 0
            torch.testing.assert_close(rows, padded(rank), rtol=0, atol=0)
            return torch.cat([padded(r) for r in range(tp)])

        monkeypatch.setattr(v2_engram, "_gather_engram_rows", fake_gather)
        fake = SimpleNamespace(
            _ready_rows=lambda n, rank=rank: rank_rows(rank)[:n],
            embed_tokens=SimpleNamespace(n_hash_cols=cols, tp_size=tp),
        )
        hash_ids = torch.zeros(num_tokens, cols, dtype=torch.int32)
        out = v2_engram.Engram._sp_embed(fake, hash_ids)
        start = rank * chunk
        want = expected_full[start : start + chunk]
        if tp == 1:
            torch.testing.assert_close(out, want, rtol=0, atol=0)
            continue
        assert out.shape == (chunk, cols, dim)
        valid = max(0, min(chunk, num_tokens - start))
        torch.testing.assert_close(out[:valid], want[:valid], rtol=0, atol=0)
        assert torch.count_nonzero(out[valid:]) == 0


def test_ced_decoder_layer_disables_sequence_parallel_past_boundary(monkeypatch):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model

    observed_sp: list[tuple[int, bool]] = []

    def fake_super_init(self, vllm_config, prefix="", **kwargs):
        del kwargs
        torch.nn.Module.__init__(self)
        observed_sp.append(
            (
                v2_model.upstream.extract_layer_index(prefix),
                bool(v2_model.upstream._use_sequence_parallel(vllm_config)),
            )
        )

    monkeypatch.setattr(
        v2_model.upstream.DeepseekV4DecoderLayer, "__init__", fake_super_init
    )
    monkeypatch.setattr(v2_model.upstream, "_use_sequence_parallel", lambda _: True)
    monkeypatch.setattr(v2_model.envs, "TPU_DEEPSEEK_V41_CED", True)

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(num_hidden_layers=4)),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    for idx in range(4):
        v2_model.DeepseekV41DecoderLayer(vllm_config, prefix=f"model.layers.{idx}")
    assert observed_sp == [(0, True), (1, True), (2, False), (3, False)]

    observed_sp.clear()
    vllm_config.cache_config.enable_prefix_caching = True
    for idx in range(4):
        v2_model.DeepseekV41DecoderLayer(vllm_config, prefix=f"model.layers.{idx}")
    assert observed_sp == [(0, True), (1, True), (2, True), (3, True)]


def test_ced_forward_skips_sp_all_gather_on_empty_replay_plan(monkeypatch):
    from dataclasses import dataclass, field
    from typing import Any

    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import model as v2_model

    @dataclass
    class _FakeMeta:
        seq_lens: torch.Tensor
        input_positions: torch.Tensor | None = None
        query_start_loc: torch.Tensor | None = None

    @dataclass
    class _FakeCtx:
        attn_metadata: dict[str, _FakeMeta]
        is_padding: torch.Tensor | None = None
        additional_kwargs: dict[str, object] = field(default_factory=dict)

    gather_calls: list[torch.Size] = []

    def fake_sp_all_gather(x):
        gather_calls.append(x.shape)
        return x.repeat(2, *([1] * (x.ndim - 1)))

    monkeypatch.setattr(v2_model.upstream, "sp_all_gather", fake_sp_all_gather)
    monkeypatch.setattr(
        v2_model.upstream,
        "get_forward_context",
        lambda: _FakeCtx(attn_metadata={"l0": _FakeMeta(seq_lens=torch.tensor([8]))}),
    )

    hidden = torch.ones((4, 4, 16), dtype=torch.bfloat16)
    pre_mix = torch.ones((4, 4), dtype=torch.float32)
    decoder_inputs: list[tuple[torch.Size, torch.Size]] = []

    dummy: Any = SimpleNamespace(
        _has_cpu_offloaded_engram=lambda: False,
        _ced_enabled=True,
        _ced_window=4,
        _ced_compile_sizes=(4,),
        use_sequence_parallel=True,
        config=SimpleNamespace(hidden_size=16),
        encoder_stage=lambda *args, **kwargs: (hidden, pre_mix),
        decoder_stage=lambda ids, pos, inputs_embeds, encoder_pre_mix: (
            decoder_inputs.append((inputs_embeds.shape, encoder_pre_mix.shape))
            or torch.ones((pos.shape[0], 16), dtype=torch.bfloat16)
        ),
    )
    dummy._stage_uses_sequence_parallel = (
        v2_model.DeepseekV41Model._stage_uses_sequence_parallel.__get__(dummy)
    )
    assert dummy._stage_uses_sequence_parallel("encoder") is True
    assert dummy._stage_uses_sequence_parallel("full") is True
    assert dummy._stage_uses_sequence_parallel("decoder") is False

    # Incomplete prefill chunk: empty plan.indices must skip sp_all_gather and decoder.
    monkeypatch.setattr(
        v2_model,
        "plan_decoder_replay",
        lambda *args: SimpleNamespace(
            indices=(),
            padded_num_tokens=0,
            query_start_loc=(0, 0),
            replay_starts=(0,),
        ),
    )
    positions = torch.arange(8, dtype=torch.int32)
    input_ids = torch.arange(8, dtype=torch.int32)
    out_empty = v2_model.DeepseekV41Model.forward(dummy, input_ids, positions)
    assert out_empty.shape == (8, 16)
    assert torch.count_nonzero(out_empty) == 0
    assert gather_calls == []
    assert decoder_inputs == []

    # Final chunk / replay: sp_all_gather runs once for hidden and once for pre_mix.
    monkeypatch.setattr(
        v2_model,
        "plan_decoder_replay",
        lambda *args: SimpleNamespace(
            indices=(6, 7),
            padded_num_tokens=4,
            query_start_loc=(0, 2),
            replay_starts=(6,),
        ),
    )
    out_replay = v2_model.DeepseekV41Model.forward(dummy, input_ids, positions)
    assert out_replay.shape == (8, 16)
    assert len(gather_calls) == 2
    assert decoder_inputs == [(torch.Size([4, 4, 16]), torch.Size([4, 4]))]


def test_mhc_pre_tpu_default_pre_mix_avoids_new_zeros_symint(monkeypatch):
    from vllm_torchtpu.models.vllm.deepseek_v41.common import mhc as v2_mhc

    captured_pre_mix: list[torch.Tensor] = []

    def fake_delayed_pre(
        residual,
        x,
        fn,
        hc_scale,
        hc_base,
        pre_mix,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    ):
        del x, fn, hc_scale, hc_base, rms_eps
        del hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat
        captured_pre_mix.append(pre_mix)
        return (
            torch.zeros_like(pre_mix),
            torch.zeros((residual.shape[0], residual.shape[1], residual.shape[1])),
            residual[:, 0],
            pre_mix,
        )

    class _FakeTpuTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="tpu")

        def new_zeros(self, *args, **kwargs):
            raise AssertionError("new_zeros must not be called for pre_mix=None")

    monkeypatch.setattr(v2_mhc, "delayed_pre", fake_delayed_pre)
    monkeypatch.setattr(
        torch,
        "arange",
        lambda end, device=None, **kw: torch.tensor(list(range(int(end))), **kw),
    )

    raw = torch.randn(5, 4, 8, dtype=torch.bfloat16)
    residual = raw.as_subclass(_FakeTpuTensor)
    hc_scale = torch.ones(3, dtype=torch.float32)
    v2_mhc.pre(
        residual,
        torch.ones((24, 32), dtype=torch.float32),
        hc_scale,
        torch.zeros(24, dtype=torch.float32),
        1e-6,
        1e-5,
        1e-5,
        2.0,
        4,
        pre_mix=None,
    )
    assert len(captured_pre_mix) == 1
    expected = torch.zeros((5, 4), dtype=torch.float32)
    expected[:, 0] = 1.0
    torch.testing.assert_close(
        captured_pre_mix[0].as_subclass(torch.Tensor), expected, rtol=0, atol=0
    )


@pytest.mark.parametrize("has_source", [False, True])
def test_sp_sharded_attention_input_projections_match_tp(monkeypatch, has_source):
    from typing import Any

    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn

    tp = 4
    num_tokens = 12
    chunk = num_tokens // tp
    d_in, q_lora, head_dim, n_idx_heads = 32, 16, 8, 4
    w_qkv = torch.randint(-8, 9, (d_in, q_lora + head_dim), dtype=torch.bfloat16) / 8.0
    w_comp = torch.randint(-8, 9, (d_in, head_dim + 1), dtype=torch.float32) / 8.0
    w_idx = torch.randint(-8, 9, (d_in, n_idx_heads), dtype=torch.bfloat16) / 8.0
    hidden_full = torch.randint(-8, 9, (num_tokens, d_in), dtype=torch.bfloat16) / 8.0
    positions = torch.arange(num_tokens, dtype=torch.int64)

    def _make_attn(sp_sharded: bool) -> Any:
        captured: dict[str, Any] = {}

        def fake_prepare(
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_weights,
            pos,
            o_padded,
        ):
            del hidden_states, qr_scale, pos
            captured["qr"] = qr
            captured["kv"] = kv
            captured["kv_score"] = kv_score
            captured["indexer_weights"] = indexer_weights
            o_padded.fill_(1.0)

        attn: Any = SimpleNamespace(
            padded_heads=2,
            n_local_heads=2,
            head_dim=head_dim,
            q_lora_rank=q_lora,
            _sp_sharded_input=sp_sharded,
            _ced_global_from_encoder=False,
            _all_indices_fit=False,
            compressor=(
                SimpleNamespace(project=lambda h: h.float() @ w_comp)
                if has_source
                else None
            ),
            indexer=(
                SimpleNamespace(
                    n_head=n_idx_heads,
                    weights_proj=lambda h: (h @ w_idx, None),
                )
                if has_source
                else None
            ),
            _fused_wqa_wkv_gemm=lambda h: h @ w_qkv,
            q_norm=lambda x: x * 2.0,
            kv_norm=lambda x: x * 3.0,
            _prepare_and_attn_fn=fake_prepare,
            _o_proj=lambda o, pos: o.reshape(pos.shape[0], -1),
            captured=captured,
        )
        attn._run_parallel_input_projections = (
            v2_attn.DeepseekV41Attention._run_parallel_input_projections.__get__(attn)
        )
        attn._split_qkv_and_norm = (
            v2_attn.DeepseekV41Attention._split_qkv_and_norm.__get__(attn)
        )
        return attn

    tp_attn = _make_attn(sp_sharded=False)
    out_tp = v2_attn.DeepseekV41Attention.forward(tp_attn, positions, hidden_full)

    # Simulate TP ranks running SP-local projections on their chunk, then all-gathering.
    rank_Valid_shards = [hidden_full[r * chunk : (r + 1) * chunk] for r in range(tp)]
    gather_widths: list[int] = []

    def fake_sp_all_gather(local_tensor: torch.Tensor) -> torch.Tensor:
        gather_widths.append(local_tensor.shape[-1])
        # Replay the same local projection across all ranks to form the gathered tensor.
        if local_tensor.dtype == torch.float32:
            return torch.cat([s.float() @ w_comp for s in rank_Valid_shards], dim=0)
        pieces = []
        for s in rank_Valid_shards:
            qr_kv_r = s @ w_qkv
            qr_r = qr_kv_r[:, :q_lora] * 2.0
            kv_r = qr_kv_r[:, q_lora:] * 3.0
            if has_source:
                idx_r = s @ w_idx
                pieces.append(torch.cat((qr_r, kv_r, idx_r), dim=-1))
            else:
                pieces.append(torch.cat((qr_r, kv_r), dim=-1))
        return torch.cat(pieces, dim=0)

    monkeypatch.setattr(v2_attn, "sp_all_gather", fake_sp_all_gather)
    sp_attn = _make_attn(sp_sharded=True)
    out_sp = v2_attn.DeepseekV41Attention.forward(
        sp_attn, positions, rank_Valid_shards[0]
    )

    torch.testing.assert_close(out_sp, out_tp, rtol=0, atol=0)
    torch.testing.assert_close(
        sp_attn.captured["qr"], tp_attn.captured["qr"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        sp_attn.captured["kv"], tp_attn.captured["kv"], rtol=0, atol=0
    )
    if has_source:
        assert gather_widths == [q_lora + head_dim + n_idx_heads, head_dim + 1]
        torch.testing.assert_close(
            sp_attn.captured["indexer_weights"],
            tp_attn.captured["indexer_weights"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            sp_attn.captured["kv_score"],
            tp_attn.captured["kv_score"],
            rtol=0,
            atol=0,
        )
    else:
        assert gather_widths == [q_lora + head_dim]
        assert sp_attn.captured["indexer_weights"] is None
        assert sp_attn.captured["kv_score"] is None


def test_write_encoder_global_kv_sp_sharded_matches_tp(monkeypatch):
    from typing import Any

    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn

    tp = 4
    num_tokens = 12
    chunk = num_tokens // tp
    d_in, head_dim = 32, 8
    w_comp = torch.randint(-8, 9, (d_in, head_dim), dtype=torch.float32) / 8.0
    hidden_full = torch.randint(-8, 9, (num_tokens, d_in), dtype=torch.bfloat16) / 8.0
    positions = torch.arange(num_tokens, dtype=torch.int64)
    recorded_kv_scores: list[torch.Tensor] = []
    projected_rows: list[int] = []

    monkeypatch.setattr(
        v2_attn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"l0": object()}),
    )
    monkeypatch.setattr(
        v2_attn,
        "sp_all_gather",
        lambda local_score: torch.cat(
            [
                hidden_full[r * chunk : (r + 1) * chunk].float() @ w_comp
                for r in range(tp)
            ],
            dim=0,
        ),
    )

    def fake_project(h: torch.Tensor) -> torch.Tensor:
        projected_rows.append(h.shape[0])
        return h.float() @ w_comp

    attn: Any = SimpleNamespace(
        kv_cache=torch.ones(1),
        compressor=SimpleNamespace(project=fake_project),
        _write_global_kv=lambda kv_score, pos, defer_index_write=True: (
            recorded_kv_scores.append((kv_score, defer_index_write))
        ),
    )
    v2_attn.DeepseekV41Attention.write_encoder_global_kv(
        attn, hidden_full, positions, sp_sharded=False
    )
    v2_attn.DeepseekV41Attention.write_encoder_global_kv(
        attn, hidden_full[:chunk], positions, sp_sharded=True
    )
    assert projected_rows == [num_tokens, chunk]
    assert len(recorded_kv_scores) == 2
    assert recorded_kv_scores[0][1] is False and recorded_kv_scores[1][1] is False
    torch.testing.assert_close(
        recorded_kv_scores[1][0], recorded_kv_scores[0][0], rtol=0, atol=0
    )


def test_select_paged_indices_projects_index_q_on_local_query_shard(monkeypatch):
    from typing import Any

    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn

    tp = 2
    count = 20
    q_lora, n_head, head_dim, topk = 16, 4, 8, 4
    qr = torch.randn(count, q_lora, dtype=torch.bfloat16)
    positions = torch.arange(count, dtype=torch.int64)
    weights = torch.randn(count, n_head, dtype=torch.float32)
    lengths = torch.full((count, 1), 8, dtype=torch.int64)
    swa_slots = torch.arange(count, dtype=torch.int64)
    swa_slots[-2:] = -1
    records = torch.zeros((1, 8), dtype=torch.int32)
    requests = torch.zeros(count, dtype=torch.int64)
    w_idx_q = torch.randn(q_lora, n_head * head_dim, dtype=torch.bfloat16)

    def _project_fn(q_in: torch.Tensor, pos_in: torch.Tensor) -> torch.Tensor:
        proj = (q_in @ w_idx_q).reshape(-1, n_head, head_dim)
        return proj + pos_in[:, None, None].to(proj.dtype) * 0.01

    full_index_q = _project_fn(qr, positions)
    part, _ = v2_attn._query_split(count, tp)
    projected_shapes: list[torch.Size] = []
    captured_q_parts: list[torch.Tensor] = []

    monkeypatch.setattr(v2_attn, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(
        v2_attn,
        "_gather_query_rows",
        lambda x: x.repeat(tp, 1),
    )

    def fake_paged_indexer(
        q_part,
        index_cache,
        recs,
        weights_part,
        lengths_part,
        candidates_part,
        write_records,
        write_keys,
        requests_part,
        **kwargs,
    ):
        del recs, weights_part, lengths_part, candidates_part
        del write_records, write_keys, requests_part, kwargs
        captured_q_parts.append(q_part)
        sel = torch.zeros((q_part.shape[0], topk), dtype=torch.int32)
        pub = torch.zeros((q_part.shape[0], 0), dtype=torch.int32)
        return index_cache.raw, sel, pub

    monkeypatch.setattr(v2_attn, "paged_indexer", fake_paged_indexer)

    for rank in range(tp):
        attn: Any = SimpleNamespace(
            indexer=SimpleNamespace(
                n_head=n_head,
                head_dim=head_dim,
                indexer_op=SimpleNamespace(
                    candidate_blocks=None, candidate_write=False
                ),
            ),
            candidate_block_buffer=torch.zeros((count, 0), dtype=torch.int32),
            _tp_rank=torch.tensor(rank, dtype=torch.int32),
            _config={"index_topk": topk},
            candidate_block_size=16,
            _fp8_kv=True,
            _project_index_q=lambda q_in, pos_in: (
                projected_shapes.append(q_in.shape) or _project_fn(q_in, pos_in)
            ),
        )
        index_cache = SimpleNamespace(raw=torch.zeros((4, 8), dtype=torch.int32))
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
            None,
        )
        assert selected.shape == (count, topk)

    # _project_index_q must run ONLY on the local [part, q_lora] shard.
    assert projected_shapes == [torch.Size([part, q_lora])] * tp
    valid_rank0 = min(part, count)
    torch.testing.assert_close(
        captured_q_parts[0][:valid_rank0],
        full_index_q[:valid_rank0],
        rtol=0,
        atol=0,
    )
    valid_rank1 = max(0, count - part)
    torch.testing.assert_close(
        captured_q_parts[1][:valid_rank1],
        full_index_q[part : part + valid_rank1],
        rtol=0,
        atol=0,
    )


def test_partitioned_moe_routing_bitwise_equal(monkeypatch):
    """SP-sharded router_logits routes locally on [T/8, 384] and packs
    (topk_weights, topk_ids) into a single int32[T/8, 9] all-gather, producing
    bitwise-identical outputs to unsharded routing.
    """
    from vllm_torchtpu.layers.adapter import moe_routing
    from vllm_torchtpu.layers.adapter.custom_ops import moe_runner as oot_moe_runner

    torch.manual_seed(2026)
    world_size = 8
    total_tokens = 64
    local_tokens = total_tokens // world_size
    num_experts = 384
    topk = 6
    hidden_size = 128

    full_hidden = torch.randn((total_tokens, hidden_size), dtype=torch.bfloat16)
    full_router_logits = torch.randn((total_tokens, num_experts), dtype=torch.float32)
    bias = torch.randn((num_experts,), dtype=torch.float32) * 0.1

    layer = SimpleNamespace(
        moe_config=SimpleNamespace(
            experts_per_token=topk,
            is_sequence_parallel=True,
            dp_size=1,
            pcp_size=1,
        ),
        custom_routing_function=None,
        renormalize=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.5,
        e_score_correction_bias=bias,
        hash_indices_table=None,
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
    )

    ref_weights, ref_ids = moe_routing.route(layer, full_hidden, full_router_logits)

    # 1. Verify VllmMoERunner._maybe_dispatch only all-gathers hidden_states
    #    and leaves router_logits at [T/8, 384].
    all_gather_shapes: list[torch.Size] = []

    class _FakeEpGroup:
        def __init__(self, rank: int, all_shards: list[torch.Tensor] | None = None):
            self.world_size = world_size
            self.rank_in_group = rank
            self._all_shards = all_shards

        def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
            all_gather_shapes.append(tensor.shape)
            assert dim == 0
            assert self._all_shards is not None
            return torch.cat(self._all_shards, dim=0)

    hidden_shards = list(full_hidden.chunk(world_size, dim=0))
    router_shards = list(full_router_logits.chunk(world_size, dim=0))

    monkeypatch.setattr(
        oot_moe_runner,
        "get_ep_group",
        lambda: _FakeEpGroup(0, hidden_shards),
    )
    runner = SimpleNamespace(
        _fused_ep_owns_pcp_collectives=lambda: False,
        do_naive_dispatch_combine=True,
        moe_config=layer.moe_config,
        routed_experts=SimpleNamespace(
            quant_method=SimpleNamespace(is_monolithic=True),
            custom_routing_function=None,
        ),
    )
    dispatched_hidden, dispatched_router = oot_moe_runner.VllmMoERunner._maybe_dispatch(
        runner, hidden_shards[0], router_shards[0]
    )
    assert all_gather_shapes == [torch.Size([local_tokens, hidden_size])]
    assert torch.equal(dispatched_hidden, full_hidden)
    assert dispatched_router is router_shards[0]

    # 2. Simulate partitioned moe_routing.route across all 8 EP ranks and verify
    #    single packed int32[T/8, 9] all-gather and bitwise-exact outputs.
    packed_shards: list[torch.Tensor] = []
    for r in range(world_size):
        w_r, id_r = moe_routing.select_experts(
            hidden_states=full_hidden[:local_tokens],
            router_logits=router_shards[r],
            topk=topk,
            renormalize=layer.renormalize,
            scoring_fn=layer.scoring_func,
            layer=layer,
        )
        packed_shards.append(
            torch.cat([w_r.contiguous().view(torch.int32), id_r.contiguous()], dim=-1)
        )

    route_gather_shapes: list[torch.Size] = []

    def _fake_route_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        route_gather_shapes.append(tensor.shape)
        assert dim == 0
        assert torch.equal(tensor, packed_shards[0])
        return torch.cat(packed_shards, dim=0)

    fake_route_group = SimpleNamespace(
        world_size=world_size,
        rank_in_group=0,
        all_gather=_fake_route_all_gather,
    )

    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: fake_route_group)

    part_weights, part_ids = moe_routing.route(layer, full_hidden, router_shards[0])
    assert route_gather_shapes == [torch.Size([local_tokens, topk // 2 + topk])]
    assert torch.equal(part_weights, ref_weights)
    assert torch.equal(part_ids, ref_ids)


def test_stage_attention_cache_scope_shares_addressing(monkeypatch):
    """stage_attention_cache_scope shares stage-invariant token/window
    addressing and shared-owner index/compressed records across layers while
    indexing each layer's own distinct block_tables (row_offset=0).
    """
    from typing import Any
    from unittest.mock import MagicMock

    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import PagedCacheRows

    count = 4
    capacity = 8
    positions = torch.tensor([0, 63, 65, 130], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 3], dtype=torch.int32)

    def _make_paged_cache(pages_tensor: torch.Tensor, row_offset: int = 0) -> Any:
        c = MagicMock(spec=PagedCacheRows)
        c.numel.return_value = 1024
        c.shape = (64, 64, 512)
        c.page_stride = 64
        c.row_offset = row_offset
        c.row_indices = lambda slots, ro=row_offset: ro + slots // 64 * 64 + slots % 64
        c.pages.return_value = pages_tensor
        return c

    comp_cache_1 = _make_paged_cache(
        torch.tensor([[10, 11, 12], [13, 14, 15]], dtype=torch.int32)
    )
    comp_cache_2 = _make_paged_cache(
        torch.tensor([[20, 21, 22], [23, 24, 25]], dtype=torch.int32)
    )
    idx_cache_1 = _make_paged_cache(
        torch.tensor([[30, 31, 32], [33, 34, 35]], dtype=torch.int32)
    )
    idx_cache_2 = _make_paged_cache(
        torch.tensor([[40, 41, 42], [43, 44, 45]], dtype=torch.int32)
    )
    shared_topk_buf = torch.full((16, 8), -1, dtype=torch.int32)
    static_ctx = {
        "attn_0": SimpleNamespace(kv_cache=comp_cache_1),
        "attn_2": SimpleNamespace(kv_cache=comp_cache_2),
    }

    def _make_layer(
        layer_id: int,
        *,
        is_kv_source: bool,
        is_index_source: bool,
        kv_source_id: int,
        index_source_id: int,
        selected_rows: list[list[int]] | None = None,
    ) -> Any:
        attn: Any = object.__new__(v2_attn.DeepseekV41Attention)
        torch.nn.Module.__init__(attn)
        attn.prefix = f"attn_{layer_id}"
        attn.compress_ratio = 2
        attn._max_model_len = capacity * 2
        attn._config = {"index_topk": 4}
        attn._fp8_kv = True
        attn.window_size = 128
        attn.n_local_heads = 2
        attn.head_dim = 8
        attn.scale = 0.25
        attn._ced_decoder = False
        attn._ced_global_from_encoder = False
        attn.compressed_cache_prefix = f"attn_{kv_source_id}"
        attn.kv_source_layer_id = kv_source_id
        attn.index_source_layer_id = index_source_id
        attn.topk_indices_buffer = shared_topk_buf
        attn.attn_sink = torch.zeros(2, dtype=torch.float32)
        attn.register_buffer(
            "_full_attn_sink", torch.zeros(2, dtype=torch.float32), persistent=False
        )
        attn.register_buffer(
            "_tp_rank_1d", torch.tensor([0], dtype=torch.int32), persistent=False
        )
        attn._precompact_records = True
        attn._token_parallel_attn = True
        attn._swa_max_requests = 4
        attn.swa_cache_layer = SimpleNamespace(
            prefix=f"swa_{layer_id}",
            kv_cache=_make_paged_cache(
                torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
            ),
        )
        attn.kv_cache = comp_cache_1 if kv_source_id == 0 else comp_cache_2
        attn._static_forward_context = static_ctx
        attn.wq_b = lambda qr: qr
        attn._rotate = lambda x, pos: x
        attn._quantize_indexer = lambda x: x
        if is_kv_source:
            comp: Any = object.__new__(v2_attn.DeepseekCompressor)
            torch.nn.Module.__init__(comp)
            comp.head_dim = 8
            comp.compress_ratio = 2
            comp.rms_norm_eps = 1e-6
            comp.norm = SimpleNamespace(weight=torch.ones(8, dtype=torch.float32))
            comp.state_cache = SimpleNamespace(
                prefix=f"state_{layer_id}",
                kv_cache=_make_paged_cache(
                    torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.int32)
                ),
            )
            attn.compressor = comp
        else:
            attn.compressor = None
        if is_index_source:
            idx_cache = idx_cache_1 if kv_source_id == 0 else idx_cache_2
            attn.indexer = SimpleNamespace(
                head_dim=8,
                n_head=2,
                wk=lambda lat: lat,
                k_norm=lambda k: k,
                k_cache=SimpleNamespace(
                    prefix=f"idx_{kv_source_id}",
                    compress_ratio=2,
                    kv_cache=idx_cache,
                ),
            )
            assert selected_rows is not None
            sel_tensor = torch.tensor(selected_rows, dtype=torch.int32)
            attn._select_paged_indices = (
                lambda qr, pos, ic, irec, reqs, w, lens, slots, pend, st=sel_tensor: (
                    captured_idx_recs.append(irec.clone()) or st.clone()
                )
            )
        else:
            attn.indexer = None
        return attn

    # Give EVERY cache prefix its own distinct block_table (row_offset=0),
    # matching 40-layer vLLM where all 51 kv_cache_groups alias from byte 0
    # and differ exclusively via metadata[prefix].block_tables.
    prefix_names = (
        "swa_0",
        "swa_1",
        "swa_2",
        "swa_3",
        "swa_4",
        "state_0",
        "state_2",
        "attn_0",
        "attn_2",
        "idx_0",
        "idx_2",
    )
    metadata = {
        k: SimpleNamespace(
            seq_lens=torch.tensor([3, 0], dtype=torch.int32),
            block_tables=torch.tensor(
                [[i * 6 + 1, i * 6 + 2, i * 6 + 3], [i * 6 + 4, i * 6 + 5, i * 6 + 6]],
                dtype=torch.int32,
            ),
            query_start_loc=query_start_loc,
        )
        for i, k in enumerate(prefix_names)
    }
    forward_ctx = SimpleNamespace(attn_metadata=metadata, additional_kwargs={})
    monkeypatch.setattr(v2_attn, "get_forward_context", lambda: forward_ctx)

    captured_inserts: list[torch.Tensor] = []
    captured_states: list[torch.Tensor] = []
    captured_idx_recs: list[torch.Tensor] = []

    monkeypatch.setattr(
        v2_attn,
        "_insert",
        lambda cache, slots, val, mask: captured_inserts.append(slots.clone()),
    )

    def _fake_save_compress_norm(
        kv_score,
        pos,
        state_cache,
        slot_mapping,
        qsl,
        reqs,
        nw,
        eps,
        ratio,
        latent,
    ):
        del kv_score, pos, state_cache, qsl, reqs, nw, eps, ratio
        captured_states.append(slot_mapping.clone())
        latent.fill_(1.0)

    monkeypatch.setattr(v2_attn, "save_compress_norm", _fake_save_compress_norm)

    row_slots_calls = 0
    orig_row_slots = v2_attn._row_slots

    def _counting_row_slots(cache, table, reqs, log):
        nonlocal row_slots_calls
        row_slots_calls += 1
        return orig_row_slots(cache, table, reqs, log)

    req_row_slots_calls = 0
    orig_req_row_slots = v2_attn._request_row_slots

    def _counting_req_row_slots(cache, table, width):
        nonlocal req_row_slots_calls
        req_row_slots_calls += 1
        return orig_req_row_slots(cache, table, width)

    monkeypatch.setattr(v2_attn, "_row_slots", _counting_row_slots)
    monkeypatch.setattr(v2_attn, "_request_row_slots", _counting_req_row_slots)

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _fake_sparse_fp8_attn(
        q, swa_cache, main_cache, records, main_records, *args, **kwargs
    ):
        del swa_cache, main_cache, args, kwargs
        captured.append((records.clone(), main_records.clone()))
        return q.clone()

    monkeypatch.setattr(v2_attn, "sparse_fp8_attention", _fake_sparse_fp8_attn)

    qr = torch.ones((count, 16), dtype=torch.bfloat16)
    kv = torch.ones((count, 8), dtype=torch.bfloat16)
    kv_score = torch.ones((count, 10), dtype=torch.float32)
    idx_w = torch.ones((count, 2), dtype=torch.bfloat16)

    l0 = _make_layer(
        0,
        is_kv_source=True,
        is_index_source=True,
        kv_source_id=0,
        index_source_id=0,
        selected_rows=[[0, -1, -1, -1], [0, 1, -1, -1], [0, 1, 2, -1], [0, -1, -1, -1]],
    )
    l1 = _make_layer(
        1,
        is_kv_source=False,
        is_index_source=False,
        kv_source_id=0,
        index_source_id=0,
    )
    l2 = _make_layer(
        2,
        is_kv_source=True,
        is_index_source=True,
        kv_source_id=2,
        index_source_id=2,
        selected_rows=[[0, -1, -1, -1], [1, 2, -1, -1], [0, 2, 3, -1], [1, -1, -1, -1]],
    )
    l3 = _make_layer(
        3,
        is_kv_source=False,
        is_index_source=True,
        kv_source_id=2,
        index_source_id=3,
        selected_rows=[[0, -1, -1, -1], [0, 3, -1, -1], [1, 2, 3, -1], [2, -1, -1, -1]],
    )
    l4 = _make_layer(
        4,
        is_kv_source=False,
        is_index_source=False,
        kv_source_id=2,
        index_source_id=3,
    )
    layers = (l0, l1, l2, l3, l4)

    # 1. Baseline without stage_attention_cache_scope
    for lyr in layers:
        out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
        lyr._prepare_tpu(None, qr, kv, None, kv_score, idx_w, positions, out)
    ref_captured = list(captured)
    ref_inserts = list(captured_inserts)
    ref_states = list(captured_states)
    ref_idx_recs = list(captured_idx_recs)
    assert row_slots_calls == 5
    assert req_row_slots_calls == 3

    # 2. With stage_attention_cache_scope: 3-page SWA fast path eliminates
    #    _row_slots completely (0 calls), _request_row_slots runs once per
    #    distinct indexer.k_cache.prefix (2 calls for idx_0 and idx_2, reused by l3),
    #    and all per-layer slots/records match the baseline bitwise.
    captured.clear()
    captured_inserts.clear()
    captured_states.clear()
    captured_idx_recs.clear()
    row_slots_calls = 0
    req_row_slots_calls = 0
    with v2_attn.stage_attention_cache_scope():
        for lyr in layers:
            out = torch.zeros((count, 2, 8), dtype=torch.bfloat16)
            lyr._prepare_tpu(None, qr, kv, None, kv_score, idx_w, positions, out)
    assert row_slots_calls == 0
    assert req_row_slots_calls == 2
    for got, ref in zip(captured_states, ref_states, strict=True):
        assert torch.equal(got, ref)
    for got, ref in zip(captured_inserts, ref_inserts, strict=True):
        assert torch.equal(got, ref)
    for got, ref in zip(captured_idx_recs, ref_idx_recs, strict=True):
        assert torch.equal(got, ref)
    for (rec_got, main_got), (rec_ref, main_ref) in zip(
        captured, ref_captured, strict=True
    ):
        assert torch.equal(rec_got, rec_ref)
        assert torch.equal(main_got, main_ref)
