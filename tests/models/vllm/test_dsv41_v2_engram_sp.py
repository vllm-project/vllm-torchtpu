# SPDX-License-Identifier: Apache-2.0
"""CPU checks for TPU V4.1 Engram sequence-parallel row selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.models.vllm.deepseek_v41.common import engram as v2_engram


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
        engram_host_lookup=None,
        engram_split_a=None,
        engram_split_b=None,
        _engram_hash_ids=lambda *args: (None, None),
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
    dummy._run_front = v2_model.DeepseekV41Model._run_front.__get__(dummy)
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
    rows = 16 * tp
    padding = (count // rows + 1) * rows - count
    part = (count + padding) // tp
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
