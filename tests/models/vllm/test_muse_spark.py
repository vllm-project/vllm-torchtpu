# SPDX-License-Identifier: Apache-2.0
"""Numerical contracts that can silently corrupt Muse text generation."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.vllm.muse_spark.config import MuseSparkTextConfig
from vllm_torchtpu.models.vllm.muse_spark.layers import (
    MuseResidual,
    MuseRMSNorm,
    MuseRotaryEmbedding,
    attention_layout,
    route_experts,
    selected_expert_outputs,
)
from vllm_torchtpu.models.vllm.muse_spark.weights import (
    decode_expert_weight,
    dequantize_nvfp4,
    expert_tp_slice,
)


def test_packed_expert_execution_matches_bf16_decoder_with_repeated_routes():
    from vllm_torchtpu.models.vllm.muse_spark.weights import (
        selected_packed_expert_outputs,
    )

    parts = {
        name: expert_tp_slice(value, name, 1, 2)
        for name, value in _packed_experts().items()
    }
    x = torch.randn(4, 32, generator=torch.Generator().manual_seed(9)).bfloat16()
    x[-1].zero_()
    ids = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 1]])
    expected = selected_expert_outputs(
        x,
        ids,
        decode_expert_weight(parts, "gate_up_proj"),
        decode_expert_weight(parts, "down_proj"),
    )
    actual = selected_packed_expert_outputs(
        x,
        ids,
        parts["gate_up_proj"],
        parts["down_proj"],
        parts["gate_up_proj_weight_scale"].view(torch.uint8),
        parts["down_proj_weight_scale"].view(torch.uint8),
        parts["gate_up_proj_weight_scale_2"],
        parts["down_proj_weight_scale_2"],
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[-1].count_nonzero() == 0


def test_packed_loader_keeps_e4m3_bytes_and_separate_global_scales():
    model = _loader_target()
    expert = model.model.language_model.layers[0].mlp.experts
    parts = _packed_experts()
    local = {name: expert_tp_slice(value, name, 1, 2) for name, value in parts.items()}
    for name in ("gate_up_proj", "down_proj"):
        setattr(
            expert,
            name,
            nn.Parameter(
                torch.empty_like(local[name].transpose(1, 2)), requires_grad=False
            ),
        )
        setattr(
            expert,
            name + "_weight_scale",
            nn.Parameter(
                torch.empty_like(
                    local[name + "_weight_scale"].transpose(1, 2), dtype=torch.uint8
                ),
                requires_grad=False,
            ),
        )
        setattr(
            expert,
            name + "_weight_scale_2",
            nn.Parameter(
                torch.empty_like(local[name + "_weight_scale_2"]), requires_grad=False
            ),
        )
    prefix = "model.language_model.layers.0.mlp.experts."
    loaded = model.load_weights(
        [(prefix + name, value) for name, value in reversed(list(parts.items()))]
    )
    assert len(loaded) == 6
    for name in ("gate_up_proj", "down_proj"):
        torch.testing.assert_close(
            getattr(expert, name), local[name].transpose(1, 2), rtol=0, atol=0
        )
        torch.testing.assert_close(
            getattr(expert, name + "_weight_scale"),
            local[name + "_weight_scale"].view(torch.uint8).transpose(1, 2),
            rtol=0,
            atol=0,
        )

        torch.testing.assert_close(
            getattr(expert, name + "_weight_scale_2"),
            local[name + "_weight_scale_2"],
            rtol=0,
            atol=0,
        )


def test_config_preserves_text_numerics_and_roundtrips():
    config = MuseSparkTextConfig(
        hidden_size=8192,
        num_hidden_layers=2,
        num_local_experts=256,
        num_experts_per_tok=8,
        moe_hidden_size=4096,
        moe_intermediate_size=4096,
        hidden_act="silu",
        attention_bias=False,
        num_shared_experts=0,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        layer_rope_theta=[500000.0, 0.0],
        layer_types=["sliding_attention", "full_attention"],
        mlp_layer_types=["sparse", "sparse"],
        qk_scale_factor=5.4730064863875,
        output_multiplier=0.17677669529663687,
        residual_gate_temperature=0.3,
        post_norm_eps=1e-8,
    )
    copy = MuseSparkTextConfig(**config.to_dict())
    assert copy.to_dict() == config.to_dict()
    assert copy.model_type == "muse_spark_text"


def test_gain_center_and_weight_placement():
    weight_first, center = True, 1.0
    x = torch.tensor([[0.25, -1.5, 4.0, 2.0]], dtype=torch.bfloat16)
    norm = MuseRMSNorm(
        4, 1e-5, zero_centered=True, center=center, weight_first=weight_first
    )
    norm.weight.data.copy_(torch.tensor([0.5, 2.0, -0.25, 1.25]))
    w = norm.weight.float() + center
    values = x.float() * w if weight_first else x.float()
    denominator = torch.sqrt(sum(values[0, i] ** 2 for i in range(4)) / 4 + 1e-5)
    expected = values / denominator
    if not weight_first:
        expected *= w
    torch.testing.assert_close(norm(x, out_dtype=torch.float32), expected)
    assert norm(x).dtype == torch.bfloat16


def test_residual_gate_preserves_fp32_stream():
    residual = MuseResidual(4, 0.3)
    residual.gate.data.copy_(torch.tensor([-1.0, 0.0, 0.3, 1.0]))
    stream = torch.tensor([[1.001, 2.003, -3.001, 4.002]])
    branch = torch.tensor([[0.5, 1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    beta = torch.sigmoid(residual.gate.float() / 0.3)
    alpha = torch.sqrt((1 - beta.square()).clamp_min(0.001))
    out = residual(stream, branch)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, alpha * stream + beta * branch.float())


def test_router_bias_affects_selection_but_not_contribution():
    logits = torch.tensor([[2.0, -2.0, 1.0, 0.0]])
    bias = torch.tensor([-1.0, 1.0, 0.0, 0.0])
    weights, ids = route_experts(
        logits,
        bias,
        top_k=2,
        multiplier=0.5,
        scoring="sigmoid",
        eps=0.1,
        route_scale=1.7,
    )
    assert ids.tolist() == [[1, 2]]
    scores = [1 / (1 + math.exp(1)), 1 / (1 + math.exp(-0.5))]
    expected = torch.tensor([[v / (sum(scores) + 0.1) * 1.7 for v in scores]])
    torch.testing.assert_close(weights, expected)


def test_nvfp4_all_bytes_use_low_nibble_first():
    packed = torch.arange(256).to(torch.uint8).reshape(1, 256)
    scales = torch.ones(1, 32, dtype=torch.float8_e4m3fn)
    decoded = dequantize_nvfp4(packed, scales, torch.tensor(2.0))
    positive = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    values = positive + [-x for x in positive]
    expected = torch.tensor(
        [[2 * values[nibble] for b in range(256) for nibble in (b % 16, b // 16)]]
    )
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


def _packed_experts():
    generator = torch.Generator().manual_seed(17)
    e, h, i = 2, 32, 64
    return {
        "gate_up_proj": torch.randint(
            0, 256, (e, 2 * i, h // 2), generator=generator, dtype=torch.uint8
        ),
        "gate_up_proj_weight_scale": torch.randint(
            1, 8, (e, 2 * i, h // 16), generator=generator
        ).to(torch.float8_e4m3fn),
        "gate_up_proj_weight_scale_2": torch.tensor([[0.01, 0.02], [0.03, 0.005]]),
        "down_proj": torch.randint(
            0, 256, (e, h, i // 2), generator=generator, dtype=torch.uint8
        ),
        "down_proj_weight_scale": torch.randint(
            1, 8, (e, h, i // 16), generator=generator
        ).to(torch.float8_e4m3fn),
        "down_proj_weight_scale_2": torch.tensor([0.02, 0.04]),
    }


def test_nvfp4_shard_before_decode_matches_full_decode():
    tp = 2
    parts = _packed_experts()
    full13 = decode_expert_weight(parts, "gate_up_proj")
    full2 = decode_expert_weight(parts, "down_proj")
    shards = [
        {k: expert_tp_slice(v, k, rank, tp) for k, v in parts.items()}
        for rank in range(tp)
    ]
    local13 = [decode_expert_weight(s, "gate_up_proj") for s in shards]
    local2 = [decode_expert_weight(s, "down_proj") for s in shards]
    gate = torch.cat([s.chunk(2, -1)[0] for s in local13], dim=-1)
    up = torch.cat([s.chunk(2, -1)[1] for s in local13], dim=-1)
    torch.testing.assert_close(torch.cat((gate, up), dim=-1), full13, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(local2, dim=1), full2, rtol=0, atol=0)


def test_native_nope_and_window_alignment():
    config = SimpleNamespace(
        layer_rope_theta=[0 if i % 4 == 1 else 500000 for i in range(62)],
        layer_types=[
            "full_attention" if i % 4 == 1 else "sliding_attention" for i in range(62)
        ],
        sliding_window=2048,
    )
    for i in range(62):
        rope, window = attention_layout(config, i)
        assert (rope, window) == ((False, None) if i % 4 == 1 else (True, 2048))


def test_rope_matches_complex_rotation_across_window_boundary():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(3, 2, 64, generator=generator, dtype=torch.bfloat16)
    positions = torch.tensor([2047, 2048, 2049])
    rope = MuseRotaryEmbedding(64, 2050, 500000.0)
    freq = 1 / (500000.0 ** (torch.arange(0, 64, 2).float() / 64))
    phase = torch.outer(positions.float(), freq)
    rotation = torch.polar(torch.ones_like(phase), phase)
    z = torch.complex(*x.float().chunk(2, dim=-1))
    rotated = z * rotation[:, None, :]
    expected = torch.cat((rotated.real, rotated.imag), dim=-1).bfloat16()
    torch.testing.assert_close(rope(x, positions), expected, rtol=0, atol=0)


def _loader_target():
    from vllm_torchtpu.models.vllm.muse_spark.model import MuseSparkForCausalLM

    model = MuseSparkForCausalLM.__new__(MuseSparkForCausalLM)
    nn.Module.__init__(model)
    model.tp_rank, model.tp_size = 1, 2
    model.ep_rank, model.ep_size = 0, 1
    model.use_ep = False
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.Module()
    layer.mlp.experts.gate_up_proj = nn.Parameter(
        torch.empty(2, 32, 64, dtype=torch.bfloat16)
    )
    layer.mlp.experts.down_proj = nn.Parameter(
        torch.empty(2, 32, 32, dtype=torch.bfloat16)
    )
    model.model.language_model.layers = nn.ModuleList([layer])
    return model


def test_loader_accepts_reordered_quant_components_and_ignores_activation_scales():
    model = _loader_target()
    prefix = "model.language_model.layers.0.mlp.experts."
    parts = _packed_experts()
    weights = [(prefix + k, parts[k]) for k in reversed(list(parts))]
    weights += [
        (prefix + "gate_up_proj_input_scale", torch.ones(2, 2)),
        ("model.vision_tower.unused.weight", torch.ones(1)),
    ]
    loaded = model.load_weights(weights)
    assert loaded == {prefix + "gate_up_proj", prefix + "down_proj"}
    local = {k: expert_tp_slice(v, k, 1, 2) for k, v in parts.items()}
    torch.testing.assert_close(
        model.model.language_model.layers[0].mlp.experts.gate_up_proj,
        decode_expert_weight(local, "gate_up_proj"),
        rtol=0,
        atol=0,
    )


def test_bf16_loader_shards_gate_up_and_down_without_quantization():
    tp, rank = 32, 17
    model = _loader_target()
    model.tp_size, model.tp_rank = tp, rank
    expert = model.model.language_model.layers[0].mlp.experts
    width = 64 // tp
    expert.gate_up_proj = nn.Parameter(
        torch.empty(2, 32, 2 * width, dtype=torch.bfloat16)
    )
    expert.down_proj = nn.Parameter(torch.empty(2, width, 32, dtype=torch.bfloat16))
    prefix = "model.language_model.layers.0.mlp.experts."
    w13 = torch.arange(2 * 128 * 32).reshape(2, 128, 32).bfloat16()
    w2 = -torch.arange(2 * 32 * 64).reshape(2, 32, 64).bfloat16()
    loaded = model.load_weights(
        [(prefix + "down_proj", w2), (prefix + "gate_up_proj", w13)]
    )
    assert loaded == {prefix + "gate_up_proj", prefix + "down_proj"}
    start, end = rank * width, (rank + 1) * width
    expected13 = torch.cat(
        (w13[:, start:end], w13[:, 64 + start : 64 + end]), 1
    ).transpose(1, 2)
    torch.testing.assert_close(expert.gate_up_proj, expected13, rtol=0, atol=0)
    torch.testing.assert_close(
        expert.down_proj, w2[..., start:end].transpose(1, 2), rtol=0, atol=0
    )


def test_logits_apply_fp32_multiplier_and_soft_cap():
    model = _loader_target()
    model.lm_head = nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
    model.lm_head.weight.data.copy_(
        torch.tensor(
            [
                [1.001, -2.003, 3.005],
                [100.0, 200.0, -300.0],
                [-100.0, -200.0, 300.0],
                [0.01, 0.02, -0.03],
            ]
        )
    )
    model.logits_processor = lambda head, hidden: head(hidden)
    model.multiplier = 1 / math.sqrt(32)
    model.soft_cap = 20.0
    hidden = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.bfloat16)
    actual = model.compute_logits(hidden)
    expected = 20 * torch.tanh(
        (hidden @ model.lm_head.weight.t()).float() / (20 * math.sqrt(32))
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float(), rtol=2e-6, atol=1e-7)


@torch.no_grad()
def test_latent_projections_shard_and_preserve_full_vector(monkeypatch):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    from vllm_torchtpu.models.vllm.muse_spark import model

    rank, tp = 0, 8
    for module in (linear, parameter, model):
        if hasattr(module, "get_tensor_model_parallel_rank"):
            monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
        if hasattr(module, "get_tensor_model_parallel_world_size"):
            monkeypatch.setattr(
                module, "get_tensor_model_parallel_world_size", lambda: tp
            )
    config = SimpleNamespace(
        hidden_size=64,
        moe_hidden_size=32,
        moe_intermediate_size=128,
        num_local_experts=2,
        num_experts_per_tok=1,
        output_multiplier=1.0,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
    )
    rng = torch.Generator().manual_seed(87)
    pool = torch.randn(32, 64, generator=rng)
    unpool = torch.randn(64, 32, generator=rng)
    x = torch.randn(17, 64, generator=rng)
    pooled = x @ pool.T
    width = 32 // tp
    loaded_pool, loaded_unpool = [], []
    for rank in range(tp):
        moe = model.MuseMoE(config, "test.mlp")
        for layer, full in (
            (moe.pre_expert_proj, pool),
            (moe.post_expert_proj, unpool),
        ):
            assert layer.weight.numel() == full.numel() // tp
            layer.weight.weight_loader(layer.weight, full)
            layer.quant_method.process_weights_after_loading(layer)
        loaded_pool.append(moe.pre_expert_proj.weight.clone())
        loaded_unpool.append(moe.post_expert_proj.weight.clone())
        start, end = rank * width, (rank + 1) * width

        def gather(partial, start=start, end=end):
            torch.testing.assert_close(partial, pooled[:, start:end])
            return pooled

        monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", gather)
        actual, _ = moe.pre_expert_proj(x)
        torch.testing.assert_close(actual, pooled)
        moe.pre_expert_norm.weight.fill_(0.5)
        expected_norm = moe.pre_expert_norm(pooled)
        torch.testing.assert_close(moe.pre_expert_norm(actual), expected_norm)
        expected = expected_norm @ unpool.T

        def reduce(
            partial,
            start=start,
            end=end,
            expected_norm=expected_norm,
            expected=expected,
        ):
            torch.testing.assert_close(
                partial, expected_norm[:, start:end] @ unpool[:, start:end].T
            )
            return expected

        monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", reduce)
        output, _ = moe.post_expert_proj(expected_norm)
        torch.testing.assert_close(output, expected)
    torch.testing.assert_close(torch.cat(loaded_pool, dim=1), pool.T, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat(loaded_unpool, dim=0), unpool.T, rtol=0, atol=0
    )


def _ep_packed_weights():
    rng = torch.Generator().manual_seed(101)
    return dict(
        gate_up_proj=torch.randint(
            256, (32, 128, 16), generator=rng, dtype=torch.uint8
        ),
        down_proj=torch.randint(256, (32, 32, 32), generator=rng, dtype=torch.uint8),
        gate_up_proj_weight_scale=(torch.rand(32, 128, 2, generator=rng) * 4).to(
            torch.float8_e4m3fn
        ),
        down_proj_weight_scale=(torch.rand(32, 32, 4, generator=rng) * 4).to(
            torch.float8_e4m3fn
        ),
        gate_up_proj_weight_scale_2=torch.rand(32, 2, generator=rng) * 0.01,
        down_proj_weight_scale_2=torch.rand(32, generator=rng) * 0.01,
    )


@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_ep_loader_and_local_expert_outputs(monkeypatch, packed):
    from vllm_torchtpu.models.vllm.muse_spark import model

    tp, dp = 4, 2
    ep = tp * dp
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: tp)
    parts = _ep_packed_weights()
    full13 = decode_expert_weight(parts, "gate_up_proj")
    full2 = decode_expert_weight(parts, "down_proj")
    if not packed:
        parts = dict(
            gate_up_proj=full13.transpose(1, 2), down_proj=full2.transpose(1, 2)
        )
    x = torch.randn(4, 32, generator=torch.Generator().manual_seed(102)).bfloat16()
    x[-1] = 0
    ids = torch.tensor(
        [[0, 31, 17, 17], [31, 0, 0, 8], [16, 17, 30, 31], [0, 8, 16, 31]]
    )
    expected = selected_expert_outputs(x, ids, full13, full2)
    for rank in (0, ep // 2, ep - 1):
        monkeypatch.setattr(
            model, "get_tensor_model_parallel_rank", lambda rank=rank: rank % tp
        )
        monkeypatch.setattr(
            model,
            "get_ep_group",
            lambda rank=rank: SimpleNamespace(world_size=ep, rank_in_group=rank),
        )
        target = model.MuseSparkForCausalLM.__new__(model.MuseSparkForCausalLM)
        nn.Module.__init__(target)
        target.tp_rank, target.tp_size, target.use_ep = rank % tp, tp, True
        target.ep_rank, target.ep_size = rank, ep
        target.model = nn.Module()
        target.model.language_model = nn.Module()
        layer = nn.Module()
        layer.mlp = nn.Module()
        experts = model.MuseExperts(32, 64, 32, packed=packed, use_ep=True)
        if not packed:
            experts.to(torch.bfloat16)
        layer.mlp.experts = experts
        target.model.language_model.layers = nn.ModuleList([layer])
        prefix = "model.language_model.layers.0.mlp.experts."
        loaded = target.load_weights(
            (prefix + k, v) for k, v in reversed(list(parts.items()))
        )
        assert len(loaded) == (6 if packed else 2)
        start, end = rank * (32 // ep), (rank + 1) * (32 // ep)
        for name, value in parts.items():
            want = value[start:end]
            if name.endswith("_weight_scale"):
                want = want.view(torch.uint8)
            if not name.endswith("_weight_scale_2"):
                want = want.transpose(1, 2)
            torch.testing.assert_close(getattr(experts, name), want, rtol=0, atol=0)
        actual = experts(x, ids)
        owned = (ids >= start) & (ids < end)
        torch.testing.assert_close(
            actual, expected.masked_fill(~owned[..., None], 0), rtol=0, atol=0
        )
        assert actual[-1].count_nonzero() == 0


@torch.inference_mode()
def test_ep_normalizes_each_expert_before_weighted_reduction(monkeypatch):
    from vllm_torchtpu.models.vllm.muse_spark import model

    ep = 8
    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        SimpleNamespace(world_size=ep, rank_in_group=0),
    )
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: ep)
    monkeypatch.setattr(model, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    config = SimpleNamespace(
        hidden_size=32,
        moe_hidden_size=32,
        moe_intermediate_size=64,
        num_local_experts=32,
        num_experts_per_tok=4,
        output_multiplier=1.0,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
    )
    rng = torch.Generator().manual_seed(103)
    x = torch.randn(5, 32, generator=rng).bfloat16()
    x[-1] = 0
    w13 = (torch.randn(32, 32, 128, generator=rng) * 0.1).bfloat16()
    w2 = (torch.randn(32, 64, 32, generator=rng) * 0.1).bfloat16()
    # All routes belong to the first few ranks: others must reduce exact zero.
    bias = torch.arange(32, 0, -1).float()
    for rank in range(ep):
        monkeypatch.setattr(
            model, "get_tensor_model_parallel_rank", lambda rank=rank: rank
        )
        monkeypatch.setattr(
            model,
            "get_ep_group",
            lambda rank=rank: SimpleNamespace(world_size=ep, rank_in_group=rank),
        )
        moe = model.MuseMoE(config, "test", use_ep=True)
        moe.pre_expert_proj.forward = lambda x: (x, None)
        moe.post_expert_proj.forward = lambda x: (x, None)
        moe.pre_expert_norm = nn.Identity()
        moe.experts.to(torch.bfloat16)
        moe.gate.weight.zero_()
        moe.gate.quant_method.process_weights_after_loading(moe.gate)
        moe.gate.e_score_correction_bias.copy_(bias)
        moe.experts.post_expert_norm.weight.fill_(0.7)
        count = 32 // ep
        moe.experts.gate_up_proj.copy_(w13[rank * count : (rank + 1) * count])
        moe.experts.down_proj.copy_(w2[rank * count : (rank + 1) * count])
        weights, ids = route_experts(
            moe.gate(x.float())[0],
            bias,
            top_k=4,
            multiplier=moe.multiplier,
            scoring=moe.scoring,
            eps=moe.division_eps,
            route_scale=moe.route_scale,
        )
        rows = selected_expert_outputs(x, ids, w13, w2)
        normalized = moe.experts.post_expert_norm(rows, out_dtype=torch.float32)
        expected = (normalized * weights[..., None]).sum(1)
        owned = (ids >= rank * count) & (ids < (rank + 1) * count)
        local = (normalized.masked_fill(~owned[..., None], 0) * weights[..., None]).sum(
            1
        )
        calls = []

        def reduce(partial, calls=calls, local=local, expected=expected):
            calls.append(partial)
            assert partial.dtype == torch.float32
            torch.testing.assert_close(partial, local, rtol=0, atol=0)
            return expected

        monkeypatch.setattr(model, "tensor_model_parallel_all_reduce", reduce)
        torch.testing.assert_close(moe(x), expected.bfloat16(), rtol=0, atol=0)
        assert len(calls) == 1
        wrong = moe.experts.post_expert_norm(
            (rows.float() * weights[..., None]).sum(1), out_dtype=torch.float32
        )
        assert not torch.allclose(wrong, expected, rtol=0.01, atol=0.01)


@torch.inference_mode()
def test_dp_attention_returns_only_local_tokens(monkeypatch):
    from vllm_torchtpu.models.vllm.muse_spark import model

    """Distinct DP inputs, an idle engine, and owners outside the local TP group."""
    tp, dp, packed = 2, 4, True
    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        SimpleNamespace(world_size=tp, rank_in_group=0),
    )
    ep, tokens, hidden = tp * dp, 5, 32
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: tp)
    config = SimpleNamespace(
        hidden_size=hidden,
        moe_hidden_size=32,
        moe_intermediate_size=64,
        num_local_experts=32,
        num_experts_per_tok=4,
        output_multiplier=1.0,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
    )
    rng = torch.Generator().manual_seed(104)
    inputs = torch.randn(dp, tokens, hidden, generator=rng).bfloat16()
    # The runner supplies equal bucket shapes, including padding/idle engines.
    for rank in range(dp):
        inputs[rank, tokens - rank % tokens :] = 0
    inputs[-1] = 0
    x = inputs.flatten(0, 1)
    router = torch.randn(32, hidden, generator=rng)
    parts = _ep_packed_weights()
    w13 = decode_expert_weight(parts, "gate_up_proj")
    w2 = decode_expert_weight(parts, "down_proj")
    count = 32 // ep
    for rank in range(ep):
        dp_rank, tp_rank = divmod(rank, tp)
        monkeypatch.setattr(
            model,
            "get_ep_group",
            lambda rank=rank: SimpleNamespace(world_size=ep, rank_in_group=rank),
        )
        group = SimpleNamespace(world_size=dp, rank_in_group=dp_rank)
        monkeypatch.setattr(model, "get_dp_group", lambda group=group: group)
        moe = model.MuseMoE(config, "test", packed=packed, use_ep=True)
        moe.pre_expert_proj.forward = lambda x: (x, None)
        moe.post_expert_proj.forward = lambda x: (x, None)
        moe.pre_expert_norm = nn.Identity()
        moe.gate.weight.copy_(router)
        moe.gate.quant_method.process_weights_after_loading(moe.gate)
        moe.experts.post_expert_norm.weight.copy_(torch.linspace(0.2, 1.2, hidden))
        if packed:
            for name, value in parts.items():
                value = value[rank * count : (rank + 1) * count]
                if name.endswith("_weight_scale"):
                    value = value.view(torch.uint8)
                if not name.endswith("_weight_scale_2"):
                    value = value.transpose(1, 2)
                getattr(moe.experts, name).copy_(value)
        else:
            moe.experts.to(torch.bfloat16)
            moe.experts.gate_up_proj.copy_(w13[rank * count : (rank + 1) * count])
            moe.experts.down_proj.copy_(w2[rank * count : (rank + 1) * count])
        weights, ids = route_experts(
            x.float() @ router.T,
            moe.gate.e_score_correction_bias,
            top_k=4,
            multiplier=moe.multiplier,
            scoring=moe.scoring,
            eps=moe.division_eps,
            route_scale=moe.route_scale,
        )
        rows = moe.experts.post_expert_norm(
            selected_expert_outputs(x, ids, w13, w2), out_dtype=torch.float32
        )
        contributions = rows * weights[..., None]
        expected = contributions.sum(1).reshape(dp, tokens, hidden)[dp_rank]
        owned = ids // count == rank
        local = contributions.masked_fill(~owned[..., None], 0).sum(1)
        same_tp_lane = ids // count % tp == tp_rank
        lane_sum = contributions.masked_fill(~same_tp_lane[..., None], 0).sum(1)
        scattered = lane_sum.reshape(dp, tokens, hidden)[dp_rank]
        gathers = iter((x, weights, ids))
        calls = []

        def gather(value, dim, gathers=gathers, dp_rank=dp_rank, calls=calls):
            full = next(gathers)
            torch.testing.assert_close(value, full.reshape(dp, tokens, -1)[dp_rank])
            assert dim == 0
            calls.append("gather")
            return full

        def scatter(value, dim, local=local, calls=calls, scattered=scattered):
            assert value.dtype == torch.float32 and dim == 0
            torch.testing.assert_close(value, local, rtol=0, atol=0)
            calls.append("scatter")
            return scattered

        def reduce(value, scattered=scattered, calls=calls, expected=expected):
            assert value.dtype == torch.float32
            torch.testing.assert_close(value, scattered, rtol=0, atol=0)
            calls.append("reduce")
            return expected

        group.all_gather, group.reduce_scatter = gather, scatter
        monkeypatch.setattr(model, "tensor_model_parallel_all_reduce", reduce)
        actual = moe(inputs[dp_rank])
        torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
        assert calls == ["gather"] * 3 + ["scatter"] + (["reduce"] if tp > 1 else [])
        if dp_rank == dp - 1:
            assert actual.count_nonzero() == 0
