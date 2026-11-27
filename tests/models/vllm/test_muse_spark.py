# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numerical contracts that can silently corrupt Muse text generation."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.models.muse_spark.common.layers import MulTanhSoftCap

from vllm_torchtpu.models.vllm.muse_spark.layers import (
    MuseRMSNormWithInputScale,
    attention_layout,
    selected_expert_outputs,
)
from vllm_torchtpu.models.vllm.muse_spark.weights import dequantize_nvfp4


@pytest.fixture(scope="module", autouse=True)
def _cpu_platform_without_tpu_device():
    """CPU CI registers no TPU device type with PyTorch, so there the vLLM
    configs and the Gloo groups from `dist_ctx` must use the CPU device.

    Module-scoped and autouse so it is in place before `dist_ctx`; TPU hosts
    keep the real platform for the TPU-marked tests.
    """
    with pytest.MonkeyPatch.context() as patch:
        try:
            torch.device("tpu")
        except RuntimeError:
            from vllm.platforms import current_platform

            patch.setattr(current_platform, "device_name", "cpu")
            patch.setattr(current_platform, "device_type", "cpu")
        yield


def decode_expert_weight(parts: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return BF16 [E, K, N], decoding one expert at a time to bound CPU memory."""
    packed = parts[name]
    out_shape = (packed.shape[0], packed.shape[-1] * 2, packed.shape[1])
    result = torch.empty(out_shape, dtype=torch.bfloat16, device="cpu")
    scales = parts[name + "_weight_scale"]
    globals_ = parts[name + "_weight_scale_2"]
    for e in range(packed.shape[0]):
        if name == "gate_up_proj":
            if globals_.shape != (packed.shape[0], 2):
                raise ValueError(
                    "Muse gate/up require separate per-expert global scales"
                )
            gate, up = packed[e].chunk(2, dim=0)
            sg, su = scales[e].chunk(2, dim=0)
            decoded = torch.cat(
                (
                    dequantize_nvfp4(gate, sg, globals_[e, 0]),
                    dequantize_nvfp4(up, su, globals_[e, 1]),
                ),
                dim=0,
            )
        else:
            if globals_.shape != (packed.shape[0],):
                raise ValueError(
                    "Muse down projection requires one global scale per expert"
                )
            decoded = dequantize_nvfp4(packed[e], scales[e], globals_[e])
        result[e].copy_(decoded.t())
    return result


@pytest.mark.parametrize(
    "arch", ["MuseSparkForCausalLM", "MuseSparkForConditionalGeneration"]
)
def test_architectures_register_the_text_model(arch):
    """The released checkpoints declare MuseSparkForConditionalGeneration;
    both names load the TPU text model instead of upstream's classes."""
    from vllm.model_executor.models import ModelRegistry

    from vllm_torchtpu.models.vllm import register_models
    from vllm_torchtpu.models.vllm.muse_spark.model import MuseSparkForCausalLM

    previous = ModelRegistry.models.get(arch)
    try:
        register_models()
        assert ModelRegistry.models[arch].load_model_cls() is MuseSparkForCausalLM
    finally:
        if previous is None:
            ModelRegistry.models.pop(arch, None)
        else:
            ModelRegistry.models[arch] = previous


def _packed_weights(num_experts=2, hidden=32, intermediate=64, seed=17):
    """Random NVFP4 checkpoint experts: packed E2M1 bytes, E4M3 block scales
    and per-expert global scales, with separate gate and up global scales."""
    rng = torch.Generator().manual_seed(seed)

    def block_scales(rows, columns):
        values = torch.rand(num_experts, rows, columns // 16, generator=rng) * 4 + 0.25
        return values.to(torch.float8_e4m3fn)

    return dict(
        gate_up_proj=torch.randint(
            256,
            (num_experts, 2 * intermediate, hidden // 2),
            generator=rng,
            dtype=torch.uint8,
        ),
        gate_up_proj_weight_scale=block_scales(2 * intermediate, hidden),
        gate_up_proj_weight_scale_2=torch.rand(num_experts, 2, generator=rng) * 0.01
        + 1e-3,
        down_proj=torch.randint(
            256,
            (num_experts, hidden, intermediate // 2),
            generator=rng,
            dtype=torch.uint8,
        ),
        down_proj_weight_scale=block_scales(hidden, intermediate),
        down_proj_weight_scale_2=torch.rand(num_experts, generator=rng) * 0.01 + 1e-3,
    )


def test_packed_expert_execution_matches_bf16_decoder_with_repeated_routes():
    from vllm_torchtpu.models.vllm.muse_spark.weights import (
        selected_packed_expert_outputs,
    )

    parts = _tp_shard(_packed_weights(), 1, 2)
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


def _tp_shard(parts, rank, tp):
    """One TP rank's slice of stacked experts, as the reference for the loader:
    this rank's gate and up rows, and this rank's down-projection columns."""
    shard = {}
    for name, value in parts.items():
        if name.endswith(("_scale_2", "_input_scale")):
            shard[name] = value
        elif name.startswith("gate_up_proj"):
            gate, up = value.chunk(2, dim=1)
            width = gate.shape[1] // tp
            rows = slice(rank * width, (rank + 1) * width)
            shard[name] = torch.cat((gate[:, rows], up[:, rows]), dim=1)
        else:
            width = value.shape[-1] // tp
            shard[name] = value[..., rank * width : (rank + 1) * width]
    return shard


def _make_experts(
    num_experts, hidden, intermediate, *, packed, tp=1, tp_rank=0, ep=1, ep_rank=0
):
    """CPU MuseRoutedExperts holding one TP or EP rank's experts."""
    with torch.inference_mode(False):
        return _build_experts(
            num_experts, hidden, intermediate, packed, tp, tp_rank, ep, ep_rank
        )


def _build_experts(num_experts, hidden, intermediate, packed, tp, tp_rank, ep, ep_rank):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        MoEActivation,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        ExpertMapManager,
    )

    from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4Config
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedConfig,
    )
    from vllm_torchtpu.models.vllm.muse_spark.model import MuseRoutedExperts

    parallel = FusedMoEParallelConfig(
        tp_size=tp,
        pcp_size=1,
        dp_size=1,
        ep_size=ep,
        tp_rank=tp_rank,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=ep_rank,
        sp_size=1,
        use_ep=ep > 1,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=2,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=num_experts // ep,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    expert_map = ExpertMapManager(
        max_num_batched_tokens=16,
        top_k=2,
        global_num_experts=num_experts,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy="linear",
        enable_eplb=False,
    )
    quant = (
        VllmNvfp4Config.from_config(
            {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16}
        )
        if packed
        else VllmUnquantizedConfig()
    )
    quant.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=ep > 1),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    with set_current_vllm_config(VllmConfig()):
        experts = MuseRoutedExperts(
            layer_name="model.language_model.layers.0.mlp.experts",
            params_dtype=torch.bfloat16,
            moe_config=config,
            quant_config=quant,
            expert_map_manager=expert_map,
        )
    return experts


def _prepare(experts, parts):
    """Load stacked checkpoint tensors. BF16 experts are laid out for the
    kernel here; the NVFP4 layout needs a TPU (see ``_lay_out_on_tpu``), so on
    CPU NVFP4 experts keep the checkpoint layout the reference path reads."""
    from vllm_torchtpu.layers.adapter import moe_routing

    loaded = experts.load_weights(list(parts.items()))
    if experts.w13_weight.dtype != torch.uint8:
        experts.quant_method.process_weights_after_loading(experts)
    else:
        # The layout step also records this rank's first expert id.
        moe_routing.register_experts_start_buffer(
            experts, device=experts.w13_weight.device
        )
    return loaded


def _lay_out_on_tpu(experts):
    """Move loaded experts to TPU, where the NVFP4 MoE method lays them out."""
    experts.to("tpu")
    if experts.w13_weight.dtype == torch.uint8:
        experts.quant_method.process_weights_after_loading(experts)


def test_shared_loader_keeps_each_tp_ranks_checkpoint_bytes(dist_ctx):
    parts = _packed_weights()
    local = _tp_shard(parts, 1, 2)
    experts = _make_experts(2, 32, 64, packed=True, tp=2, tp_rank=1)
    loaded = _prepare(experts, dict(reversed(list(parts.items()))))
    assert len(loaded) == 6
    for name, prefix in (("gate_up_proj", "w13"), ("down_proj", "w2")):
        torch.testing.assert_close(
            getattr(experts, f"{prefix}_weight"), local[name], rtol=0, atol=0
        )
        torch.testing.assert_close(
            getattr(experts, f"{prefix}_weight_scale").view(torch.uint8),
            local[name + "_weight_scale"].view(torch.uint8),
            rtol=0,
            atol=0,
        )
    # Gate and up keep their own per-expert global scales.
    torch.testing.assert_close(
        experts.w13_weight_scale_2,
        local["gate_up_proj_weight_scale_2"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        experts.w2_weight_scale_2, local["down_proj_weight_scale_2"], rtol=0, atol=0
    )


def test_router_bias_affects_selection_but_not_contribution():
    from vllm_torchtpu.models.vllm.muse_spark.model import MuseMoE

    class Gate(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "e_score_correction_bias", torch.tensor([-1.0, 1.0, 0.0, 0.0])
            )
            self.routed_scaling_factor = 1.7
            self.use_grouped_topk = False
            # [in, out], as the TPU dense loader stores it; row 0 holds the logits.
            self.weight = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
            self.weight.data[0] = torch.tensor([2.0, -2.0, 1.0, 0.0])

    moe = SimpleNamespace(gate=Gate(), multiplier=0.5, top_k=2)
    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)
    weights, ids = MuseMoE.route(moe, x)
    assert ids.tolist() == [[1, 2]]
    assert weights.dtype == torch.float32
    scores = [1 / (1 + math.exp(1)), 1 / (1 + math.exp(-0.5))]
    expected = torch.tensor([[v / sum(scores) * 1.7 for v in scores]])
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


def _causal_lm():
    from vllm_torchtpu.models.vllm.muse_spark.model import MuseSparkForCausalLM

    model = MuseSparkForCausalLM.__new__(MuseSparkForCausalLM)
    nn.Module.__init__(model)
    return model


def test_model_loader_routes_experts_and_ignores_modality_tensors(dist_ctx):
    model = _causal_lm()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = _make_experts(2, 32, 64, packed=True, tp=2, tp_rank=1)
    model.model.language_model.layers = nn.ModuleList([layer])
    prefix = "model.language_model.layers.0.mlp.experts."
    parts = _packed_weights()
    weights = [(prefix + k, parts[k]) for k in reversed(list(parts))]
    weights += [
        (prefix + "gate_up_proj_input_scale", torch.ones(2, 2)),
        ("model.vision_tower.unused.weight", torch.ones(1)),
        # Skipped before the BF16 check, which would reject these names.
        ("model.vision_tower.blocks.0.mlp.fc1.weight", torch.ones(2, 4).byte()),
        ("model.audio_tower.proj.weight_scale", torch.ones(1)),
    ]
    loaded = model.load_weights(weights)
    names = ("weight", "weight_scale", "weight_scale_2")
    assert loaded == {prefix + f"{p}_{n}" for p in ("w13", "w2") for n in names} | {
        prefix + "w13_input_scale"
    }
    torch.testing.assert_close(
        layer.mlp.experts.w13_weight,
        _tp_shard(parts, 1, 2)["gate_up_proj"],
        rtol=0,
        atol=0,
    )


def test_bf16_loader_shards_gate_up_and_down_without_quantization(dist_ctx):
    tp, rank = 32, 17
    experts = _make_experts(2, 32, 64, packed=False, tp=tp, tp_rank=rank)
    w13 = torch.arange(2 * 128 * 32).reshape(2, 128, 32).bfloat16()
    w2 = -torch.arange(2 * 32 * 64).reshape(2, 32, 64).bfloat16()
    loaded = experts.load_weights([("down_proj", w2), ("gate_up_proj", w13)])
    assert loaded == {"w13_weight", "w2_weight"}
    local = _tp_shard({"gate_up_proj": w13, "down_proj": w2}, rank, tp)
    torch.testing.assert_close(
        experts.w13_weight, local["gate_up_proj"], rtol=0, atol=0
    )
    torch.testing.assert_close(experts.w2_weight, local["down_proj"], rtol=0, atol=0)


def test_logits_apply_fp32_multiplier_and_soft_cap():
    model = _causal_lm()
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
    model.soft_cap = MulTanhSoftCap(model.multiplier, 20.0)
    hidden = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.bfloat16)
    actual = model.compute_logits(hidden)
    expected = 20 * torch.tanh(
        (hidden @ model.lm_head.weight.t()).float() / (20 * math.sqrt(32))
    )
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float(), rtol=2e-6, atol=1e-7)


@torch.no_grad()
def test_latent_projections_shard_and_preserve_full_vector(
    vllm_config_context, monkeypatch
):
    import vllm.model_executor.layers.linear as linear
    import vllm.model_executor.parameter as parameter

    from vllm_torchtpu.models.vllm.muse_spark import model

    # The routed experts are not under test here.
    monkeypatch.setattr(
        model,
        "FusedMoEFactory",
        lambda **_: SimpleNamespace(routed_experts=nn.Module()),
    )
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
        out_start, out_end = rank * 64 // tp, (rank + 1) * 64 // tp

        def gather_unpooled(
            partial,
            out_start=out_start,
            out_end=out_end,
            expected_norm=expected_norm,
            expected=expected,
        ):
            torch.testing.assert_close(
                partial, expected_norm @ unpool[out_start:out_end].T
            )
            return expected

        monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", gather_unpooled)
        output, _ = moe.post_expert_proj(expected_norm)
        torch.testing.assert_close(output, expected)
    torch.testing.assert_close(torch.cat(loaded_pool, dim=1), pool.T, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat(loaded_unpool, dim=1), unpool.T, rtol=0, atol=0
    )


def _laid_out_scales(parts):
    """Block x global scales as the TPU reads them: the NVFP4 MoE method fuses
    them into FP32 block scales, and gmm_v2 rounds those to BF16 in each tile."""
    gate, up = parts["gate_up_proj_weight_scale"].float().chunk(2, dim=1)
    g13 = parts["gate_up_proj_weight_scale_2"]
    w13 = torch.cat((gate * g13[:, :1, None], up * g13[:, 1:, None]), dim=1)
    w2 = parts["down_proj_weight_scale"].float() * parts[
        "down_proj_weight_scale_2"
    ].reshape(-1, 1, 1)
    return w13.bfloat16().float(), w2.bfloat16().float()


def _routed_tokens(tokens, hidden, num_experts, top_k, seed):
    """BF16 tokens with distinct random experts and positive weights."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, hidden, generator=rng).bfloat16()
    ids = torch.rand(tokens, num_experts, generator=rng).argsort(-1)[:, :top_k].int()
    weights = torch.rand(tokens, top_k, generator=rng) + 0.1
    return x, ids, weights


@pytest.mark.tpu_test
@pytest.mark.parametrize("use_ep", [False, True], ids=["tp", "ep"])
@pytest.mark.parametrize("packed", [True, False], ids=["nvfp4", "bf16"])
@torch.inference_mode()
def test_tpu_grouped_experts_match_reference_decoder(dist_ctx, packed, use_ep):
    """Serving path (fused_moe_func rows -> gmm_v2) against the reference
    decoder, for NVFP4 and BF16 checkpoint experts."""
    from vllm_torchtpu.models.vllm.muse_spark import model
    from vllm_torchtpu.models.vllm.muse_spark.weights import (
        selected_packed_expert_outputs,
    )

    # Per-rank expert shapes of the NVFP4 checkpoint at TP8, with fewer experts.
    hidden, intermediate, num_experts, top_k = 4096, 512, 8, 4
    ep_size, ep_rank = (2, 1) if use_ep else (1, 0)
    experts = _make_experts(
        num_experts, hidden, intermediate, packed=packed, ep=ep_size, ep_rank=ep_rank
    )
    parts = _packed_weights(num_experts, hidden, intermediate, seed=105)
    x, ids, weights = _routed_tokens(37, hidden, num_experts, top_k, seed=106)
    x[-1] = 0
    if packed:
        assert len(_prepare(experts, parts)) == 6
        # The reference rounds the scales as the kernel does; only FP32
        # summation order changes BF16 rows.
        expected = selected_packed_expert_outputs(
            x, ids, parts["gate_up_proj"], parts["down_proj"], *_laid_out_scales(parts)
        )
    else:
        w13 = decode_expert_weight(parts, "gate_up_proj")
        w2 = decode_expert_weight(parts, "down_proj")
        # The BF16 checkpoint stores each projection as [experts, out, in].
        checkpoint = {
            "gate_up_proj": w13.transpose(1, 2).contiguous(),
            "down_proj": w2.transpose(1, 2).contiguous(),
        }
        assert _prepare(experts, checkpoint) == {"w13_weight", "w2_weight"}
        expected = selected_expert_outputs(x, ids, w13, w2)
    if use_ep:
        owned = ids // (num_experts // ep_size) == ep_rank
        expected = expected.masked_fill(~owned[..., None], 0)

    _lay_out_on_tpu(experts)
    actual = model._expert_rows(
        experts, x.to("tpu"), weights.to("tpu"), ids.to("tpu")
    ).cpu()

    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=2**-5)
    assert actual[-1].count_nonzero() == 0


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "settings",
    [
        {"USE_MOE_FUSED_EP_KERNEL": "1", "MOE_FUSED_EP_ENABLE_W4A8": "1"},
        {"MOE_FUSED_EP_KERNEL_IMPL": "adaptive"},
    ],
    ids=["w4a8", "adaptive"],
)
@torch.inference_mode()
def test_tpu_experts_keep_block16_scales_under_fused_ep_settings(
    dist_ctx, monkeypatch, settings
):
    """An admitted fused EP op would requantize the experts to its block
    (64 by default) and never run: _expert_rows calls only gmm_v2. Admission
    itself must be skipped, whatever the op's shape checks would decide."""
    from vllm_torchtpu.layers.adapter.fused_moe_ep import FUSED_MOE_EP_OP_ATTR
    from vllm_torchtpu.layers.adapter.quantization import nvfp4

    def admission(*args, **kwargs):
        raise AssertionError("fused EP admission ran for Muse's experts")

    monkeypatch.setattr(nvfp4, "_prebuild_w4a8", admission)
    monkeypatch.setattr(nvfp4.adaptive_moe, "prebuild_adaptive_nvfp4", admission)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    hidden, intermediate, num_experts = 4096, 512, 8
    experts = _make_experts(
        num_experts, hidden, intermediate, packed=True, ep=2, ep_rank=1
    )
    _prepare(experts, _packed_weights(num_experts, hidden, intermediate, seed=105))
    _lay_out_on_tpu(experts)

    method = experts.quant_method
    assert getattr(method, FUSED_MOE_EP_OP_ATTR) is None
    assert method._adaptive_fused_moe_op is None
    assert experts.w13_weight_scale.shape[1] == hidden // 16
    assert experts.w2_weight_scale.shape[1] == intermediate // 16


@pytest.mark.tpu_test
@torch.inference_mode()
def test_tpu_one_hot_ep_permute_keeps_non_finite_row_local(dist_ctx, monkeypatch):
    """A NaN row must not reach other tokens through the one-hot EP permute."""
    from vllm_torchtpu.layers.adapter import fused_moe
    from vllm_torchtpu.models.vllm.muse_spark import model
    from vllm_torchtpu.models.vllm.muse_spark.weights import (
        selected_packed_expert_outputs,
    )

    monkeypatch.setenv("USE_MOE_SPARSE_CORE", "1")
    monkeypatch.setenv("ONEHOT_MOE_PERMUTE_THRESHOLD", "4096")
    # The shared op binds the permute settings when it is first built.
    monkeypatch.setattr(fused_moe, "_fused_moe_kernel_cache", {})
    hidden, intermediate, num_experts, top_k = 4096, 512, 8, 4
    ep_size, ep_rank = 2, 1
    experts = _make_experts(
        num_experts, hidden, intermediate, packed=True, ep=ep_size, ep_rank=ep_rank
    )
    parts = _packed_weights(num_experts, hidden, intermediate, seed=105)
    assert len(_prepare(experts, parts)) == 6

    x, ids, weights = _routed_tokens(16, hidden, num_experts, top_k, seed=107)
    x[-1] = float("nan")
    clean = x.clone()
    clean[-1] = 0
    expected = selected_packed_expert_outputs(
        clean,
        ids,
        parts["gate_up_proj"],
        parts["down_proj"],
        *_laid_out_scales(parts),
    )
    owned = ids // (num_experts // ep_size) == ep_rank
    expected = expected.masked_fill(~owned[..., None], 0)

    _lay_out_on_tpu(experts)
    actual = model._expert_rows(
        experts, x.to("tpu"), weights.to("tpu"), ids.to("tpu")
    ).cpu()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=2**-5)
    assert actual[-1].count_nonzero() == 0


@pytest.mark.parametrize("packed", [False, True])
@torch.inference_mode()
def test_ep_loader_and_local_expert_outputs(dist_ctx, packed):
    from vllm_torchtpu.models.vllm.muse_spark import model

    ep = 8
    parts = _packed_weights(32, 32, 64, seed=101)
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
        experts = _make_experts(32, 32, 64, packed=packed, ep=ep, ep_rank=rank)
        loaded = _prepare(experts, dict(reversed(list(parts.items()))))
        assert len(loaded) == (6 if packed else 2)
        start, end = rank * (32 // ep), (rank + 1) * (32 // ep)
        actual = model._expert_rows(experts, x, torch.ones(ids.shape), ids)
        owned = (ids >= start) & (ids < end)
        torch.testing.assert_close(
            actual, expected.masked_fill(~owned[..., None], 0), rtol=0, atol=0
        )
        assert actual[-1].count_nonzero() == 0


def _rank_moe(model, monkeypatch, experts, router):
    """MuseMoE around one rank's experts with identity latent projections and
    pre-expert norm, so only routing, experts, post-norm and collectives run.
    ``router`` is the [experts, hidden] router weight."""
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
    monkeypatch.setattr(
        model, "FusedMoEFactory", lambda **_: SimpleNamespace(routed_experts=experts)
    )
    moe = model.MuseMoE(config, "test", use_ep=True)
    moe.pre_expert_proj.forward = lambda x: (x, None)
    moe.post_expert_proj.forward = lambda x: (x, None)
    moe.pre_expert_norm = nn.Identity()
    moe.gate.weight.copy_(router)
    moe.gate.quant_method.process_weights_after_loading(moe.gate)
    return moe


@torch.inference_mode()
def test_ep_normalizes_each_expert_before_weighted_reduction(dist_ctx, monkeypatch):
    from vllm_torchtpu.models.vllm.muse_spark import model

    ep = 8
    rng = torch.Generator().manual_seed(103)
    x = torch.randn(5, 32, generator=rng).bfloat16()
    x[-1] = 0
    w13 = (torch.randn(32, 32, 128, generator=rng) * 0.1).bfloat16()
    w2 = (torch.randn(32, 64, 32, generator=rng) * 0.1).bfloat16()
    stacked = {"gate_up_proj": w13.transpose(1, 2), "down_proj": w2.transpose(1, 2)}
    rank_experts = []
    for rank in range(ep):
        experts = _make_experts(32, 32, 64, packed=False, ep=ep, ep_rank=rank)
        _prepare(experts, stacked)
        rank_experts.append(experts)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        SimpleNamespace(world_size=ep, rank_in_group=0),
    )
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: ep)
    monkeypatch.setattr(model, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    # All routes belong to the first few ranks: others must reduce exact zero.
    bias = torch.arange(32, 0, -1).float()
    for rank, experts in enumerate(rank_experts):
        moe = _rank_moe(model, monkeypatch, experts, torch.zeros(32, 32))
        moe.gate.e_score_correction_bias.copy_(bias)
        moe.post_expert_norm.weight.fill_(0.7)
        count = 32 // ep
        weights, ids = moe.route(x)
        rows = selected_expert_outputs(x, ids, w13, w2)
        normalized = moe.post_expert_norm(rows, out_dtype=torch.float32)
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
        wrong = moe.post_expert_norm(
            (rows.float() * weights[..., None]).sum(1), out_dtype=torch.float32
        )
        assert not torch.allclose(wrong, expected, rtol=0.01, atol=0.01)


@torch.inference_mode()
def test_dp_attention_returns_only_local_tokens(dist_ctx, monkeypatch):
    from vllm_torchtpu.models.vllm.muse_spark import model

    """Distinct DP inputs, an idle engine, and owners outside the local TP group."""
    tp, dp = 2, 4
    rank_experts = [
        _make_experts(32, 32, 64, packed=True, ep=tp * dp, ep_rank=rank)
        for rank in range(tp * dp)
    ]
    for experts in rank_experts:
        _prepare(experts, _packed_weights(32, 32, 64, seed=101))
    monkeypatch.setattr(
        "vllm.distributed.parallel_state._TP",
        SimpleNamespace(world_size=tp, rank_in_group=0),
    )
    ep, tokens, hidden = tp * dp, 5, 32
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: tp)
    rng = torch.Generator().manual_seed(104)
    inputs = torch.randn(dp, tokens, hidden, generator=rng).bfloat16()
    # The runner supplies equal bucket shapes, including padding/idle engines.
    for rank in range(dp):
        inputs[rank, tokens - rank % tokens :] = 0
    inputs[-1] = 0
    x = inputs.flatten(0, 1)
    router = torch.randn(32, hidden, generator=rng)
    parts = _packed_weights(32, 32, 64, seed=101)
    w13 = decode_expert_weight(parts, "gate_up_proj")
    w2 = decode_expert_weight(parts, "down_proj")
    count = 32 // ep
    for rank in range(ep):
        dp_rank, tp_rank = divmod(rank, tp)
        group = SimpleNamespace(world_size=dp, rank_in_group=dp_rank)
        monkeypatch.setattr(model, "get_dp_group", lambda group=group: group)
        moe = _rank_moe(model, monkeypatch, rank_experts[rank], router)
        moe.post_expert_norm.weight.copy_(torch.linspace(0.2, 1.2, hidden))
        weights, ids = moe.route(x)
        rows = moe.post_expert_norm(
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


class _ReferenceAttention(nn.Module):
    """Causal single-sequence GQA attention standing in for the paged backend."""

    def __init__(self, num_heads, head_size, scale, num_kv_heads, **kwargs):
        super().__init__()
        self.num_heads, self.head_size, self.scale = num_heads, head_size, scale
        self.num_kv_heads = num_kv_heads
        self.window = kwargs.get("per_layer_sliding_window")
        self.widths = []

    def forward(self, q, k, v):
        self.widths.append((q.shape[-1], k.shape[-1], v.shape[-1]))
        tokens = q.shape[0]
        q = q.reshape(tokens, self.num_heads, self.head_size).float()
        k = k.reshape(tokens, self.num_kv_heads, self.head_size).float()
        v = v.reshape(tokens, self.num_kv_heads, self.head_size).float()
        groups = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * self.scale
        position = torch.arange(tokens)
        mask = position[None, :] <= position[:, None]
        if self.window is not None:
            mask &= position[:, None] - position[None, :] < self.window
        scores = scores.masked_fill(~mask, float("-inf"))
        out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        return out.reshape(tokens, -1).to(torch.bfloat16)


def _text_config(**overrides):
    values = dict(
        vocab_size=48,
        hidden_size=64,
        head_dim=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        layer_rope_theta=[500000.0, 0.0],
        layer_types=["sliding_attention", "full_attention"],
        mlp_layer_types=["sparse", "sparse"],
        sliding_window=4,
        qk_scale_factor=1.5,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        num_experts_per_tok=2,
        num_local_experts=4,
        moe_hidden_size=32,
        moe_intermediate_size=32,
        output_multiplier=0.25,
        routed_scaling_factor=1.0,
        residual_gate_temperature=0.3,
        tie_word_embeddings=False,
        final_logit_softcapping=30.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _text_vllm_config(config, parallel=None, **model):
    from vllm.config.compilation import CompilationMode

    parallel_config = dict(
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_eplb=False,
        enable_expert_parallel=False,
        data_parallel_size=1,
        expert_placement_strategy="linear",
    )
    parallel_config.update(parallel or {})
    model_config = dict(
        hf_text_config=config,
        dtype=torch.bfloat16,
        max_model_len=64,
        quantization=None,
    )
    model_config.update(model)
    return SimpleNamespace(
        model_config=SimpleNamespace(**model_config),
        parallel_config=SimpleNamespace(**parallel_config),
        load_config=SimpleNamespace(load_format="safetensors"),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        cache_config=None,
        quant_config=None,
    )


@pytest.fixture
def text_stack(dist_ctx, monkeypatch):
    """The Muse text modules on CPU, with reference attention and BF16 experts.

    Modules build under the BF16 default dtype, as the model loader builds
    them; tests create FP32 inputs explicitly.
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_torchtpu.models.vllm.muse_spark import model as muse

    monkeypatch.setattr(muse, "Attention", _ReferenceAttention)
    monkeypatch.setattr(
        muse,
        "FusedMoEFactory",
        lambda **kw: SimpleNamespace(
            routed_experts=_make_experts(
                kw["num_experts"],
                kw["hidden_size"],
                kw["intermediate_size"],
                packed=False,
            )
        ),
    )
    with (
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))),
        set_default_torch_dtype(torch.bfloat16),
    ):
        yield muse


def _initialize_text_weights(module, seed):
    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            values = torch.randn(param.shape, generator=rng, dtype=torch.float32)
            param.copy_((values * 0.05).to(param.dtype))
    for child in module.modules():
        quant_method = getattr(child, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(child)


@pytest.mark.parametrize("layer_idx", [0, 1], ids=["sliding_rope", "full_nope"])
@torch.inference_mode()
def test_attention_uses_native_head_size_64(text_stack, layer_idx):
    config = _text_config()
    attention = text_stack.MuseAttention(
        config, _text_vllm_config(config), layer_idx, "layers.0.self_attn"
    )
    _initialize_text_weights(attention, 11)
    assert attention.attn.head_size == 64
    assert (attention.rotary_emb is None) == (layer_idx == 1)
    assert attention.attn.window == (4 if layer_idx == 0 else None)
    # gate_proj keeps vLLM's [out, in] layout; the other linears are flipped.
    assert attention.gate_proj.weight.shape == (4 * 64, 64)
    assert attention.qkv_proj.weight.shape == (64, (4 + 2 * 2) * 64)
    rng = torch.Generator().manual_seed(12)
    x = torch.randn(7, 64, generator=rng, dtype=torch.float32).bfloat16()

    out = attention(torch.arange(7), x)

    assert out.shape == (7, 64) and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    assert attention.attn.widths == [(4 * 64, 2 * 64, 2 * 64)]


@torch.inference_mode()
def test_attention_rope_matches_complex_rotation_across_window_boundary(text_stack):
    """The sliding layer's rotary embedding as MuseAttention builds it: an FP32
    table, NeoX-style on the checkpoint's [even | odd] q/k rows."""
    config = _text_config()
    vllm_config = _text_vllm_config(config, max_model_len=2050)
    attention = text_stack.MuseAttention(config, vllm_config, 0, "layers.0.self_attn")
    rope = attention.rotary_emb
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(3, 2, 64, generator=generator, dtype=torch.float32).bfloat16()
    positions = torch.tensor([2047, 2048, 2049])
    freq = 1 / (500000.0 ** (torch.arange(0, 64, 2).float() / 64))
    phase = torch.outer(positions.float(), freq)
    rotation = torch.polar(torch.ones_like(phase), phase)
    z = torch.complex(*x.float().chunk(2, dim=-1))
    rotated = z * rotation[:, None, :]
    expected = torch.cat((rotated.real, rotated.imag), dim=-1).bfloat16()
    assert rope.cos_sin_cache.dtype == torch.float32
    q, k = rope(positions, x.float(), x.float())
    torch.testing.assert_close(q.bfloat16(), expected, rtol=0, atol=0)
    torch.testing.assert_close(k.bfloat16(), expected, rtol=0, atol=0)


@torch.inference_mode()
def test_model_embeds_runs_every_layer_and_normalizes(text_stack, monkeypatch):
    config = _text_config()
    model = text_stack.MuseModel(
        vllm_config=_text_vllm_config(config), prefix="model.language_model"
    )
    assert len(model.layers) == 2 and model.prefix == "model.language_model"
    _initialize_text_weights(model, 15)
    calls = []
    for index, layer in enumerate(model.layers):
        forward = layer.forward

        def record(positions, stream, index=index, forward=forward):
            assert stream.dtype == torch.float32
            calls.append(index)
            return forward(positions, stream)

        monkeypatch.setattr(layer, "forward", record)
    ids = torch.tensor([3, 1, 4, 1, 5])
    positions = torch.arange(5)

    from_ids = model(ids, positions)
    from_embeds = model(None, positions, inputs_embeds=model.embed_input_ids(ids))

    assert calls == [0, 1, 0, 1]
    assert from_ids.shape == (5, 64) and from_ids.dtype == torch.bfloat16
    torch.testing.assert_close(from_ids, from_embeds, rtol=0, atol=0)


@torch.inference_mode()
def test_causal_lm_builds_text_stack_and_returns_capped_logits(text_stack):
    config = _text_config()
    lm = text_stack.MuseSparkForCausalLM(vllm_config=_text_vllm_config(config))
    assert isinstance(lm.model.language_model, text_stack.MuseModel)
    assert lm.lm_head.org_vocab_size == 48 and lm.lm_head.weight.shape[1] == 64
    assert lm.soft_cap is not None and not lm._dummy_load
    _initialize_text_weights(lm, 16)

    hidden = lm(torch.tensor([7, 8, 9]), torch.arange(3))
    logits = lm.compute_logits(hidden)

    assert logits.shape == (3, 48) and logits.dtype == torch.float32
    assert logits.abs().max() <= 30.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parallel": {"pipeline_parallel_size": 2}}, "PP, context parallelism"),
        ({"parallel": {"decode_context_parallel_size": 2}}, "context parallelism"),
        ({"parallel": {"enable_eplb": True}}, "EPLB are not supported"),
        ({"parallel": {"data_parallel_size": 2}}, "requires expert parallelism"),
        (
            {
                "parallel": {
                    "enable_expert_parallel": True,
                    "expert_placement_strategy": "round_robin",
                }
            },
            "linear expert placement",
        ),
        ({"config": {"mlp_layer_types": ["dense", "sparse"]}}, "routed MoE layers"),
        ({"config": {"tie_word_embeddings": True}}, "untied output projection"),
        ({"model": {"dtype": torch.float32}}, "--dtype bfloat16"),
        ({"model": {"quantization": "fp8"}}, "BF16 and NVFP4 checkpoints"),
        (
            {
                "model": {"quantization": "modelopt_fp4"},
                "config": {"moe_intermediate_size": 40},
            },
            "align to 16-value blocks",
        ),
    ],
    ids=[
        "pipeline",
        "context",
        "eplb",
        "dp_without_ep",
        "placement",
        "dense_mlp",
        "tied",
        "fp32",
        "fp8",
        "nvfp4_blocks",
    ],
)
def test_causal_lm_rejects_unsupported_setups(text_stack, overrides, message):
    config = _text_config(**overrides.get("config", {}))
    vllm_config = _text_vllm_config(
        config, overrides.get("parallel"), **overrides.get("model", {})
    )
    with pytest.raises(ValueError, match=message):
        text_stack.MuseSparkForCausalLM(vllm_config=vllm_config)


def test_causal_lm_rejects_silu_bf16_reference_parity(text_stack, monkeypatch):
    """The expert rows run SiLU outside gmm_v2, so they cannot round it the
    way additional_config["tpu_moe_silu_bf16"] asks."""
    from vllm_torchtpu.moe_reference_parity import MoEReferenceParity

    monkeypatch.setattr(
        text_stack,
        "get_moe_reference_parity",
        lambda _: MoEReferenceParity(silu_bf16=True),
    )
    vllm_config = _text_vllm_config(_text_config())
    with pytest.raises(ValueError, match="tpu_moe_silu_bf16"):
        text_stack.MuseSparkForCausalLM(vllm_config=vllm_config)


def test_causal_lm_rejects_expert_count_not_divisible_by_ep(text_stack, monkeypatch):
    monkeypatch.setattr(
        text_stack,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=3, rank_in_group=0),
    )
    config = _text_config()
    vllm_config = _text_vllm_config(config, {"enable_expert_parallel": True})
    with pytest.raises(ValueError, match="divisible by EP"):
        text_stack.MuseSparkForCausalLM(vllm_config=vllm_config)


@pytest.mark.parametrize(
    ("prefixes", "message"),
    [("mlp.gate", "router weight in FP32"), ("mlp.experts", "BF16 or NVFP4 weights")],
    ids=["router", "experts"],
)
def test_causal_lm_rejects_quantize_on_load_of_router_and_experts(
    text_stack, monkeypatch, prefixes, message
):
    """Routing reads the router weight and the expert rows read the experts
    directly, so an FP8 quantize-on-load method would bypass its scales."""
    monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", prefixes)
    vllm_config = _text_vllm_config(_text_config())
    with pytest.raises(ValueError, match=message):
        text_stack.MuseSparkForCausalLM(vllm_config=vllm_config)


class _StubAttention(nn.Module):
    """The same linear map for both decoder layers; carries upstream's norm name."""

    def __init__(self, weight, pre_qkv_proj_norm):
        super().__init__()
        self.weight = weight
        self.pre_qkv_proj_norm = pre_qkv_proj_norm

    def forward(self, positions, hidden_states):
        return (hidden_states.float() @ self.weight).to(hidden_states.dtype)


class _StubExperts(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return (x.float() @ self.weight).to(x.dtype)


@torch.no_grad()
def test_decoder_layer_matches_upstream(text_stack):
    """Norm order, dtypes and gated residual blends match upstream
    ``MuseSpark1xDecoderLayer.forward`` on the same norms and gates, with
    attention and experts replaced by the same stand-ins, as in
    ``test_qwen3_5_patch.py::test_forward_matches_upstream``.

    Upstream's CPU fallback rounds each post-norm branch to BF16 before the
    blend, where its fused CUDA kernels, like this model, keep it in FP32, so
    the comparison allows that rounding.
    """
    from vllm.models.muse_spark.common.layers import norm_then_gated_residual
    from vllm.models.muse_spark.nvidia.model import MuseSpark1xDecoderLayer

    config = _text_config()
    layer = text_stack.MuseDecoderLayer(
        config, _text_vllm_config(config), 0, "layers.0"
    )
    _initialize_text_weights(layer, 21)
    rng = torch.Generator().manual_seed(22)
    layer.self_attn = _StubAttention(
        torch.randn(64, 64, generator=rng, dtype=torch.float32) / 8,
        layer.input_layernorm,
    )
    layer.mlp = _StubExperts(
        torch.randn(64, 64, generator=rng, dtype=torch.float32) / 8
    )

    upstream = MuseSpark1xDecoderLayer.__new__(MuseSpark1xDecoderLayer)
    nn.Module.__init__(upstream)
    upstream.self_attn = layer.self_attn
    upstream.post_attn_norm = layer.post_attn_norm
    upstream.add_attn = layer.post_attention_residual_gate
    upstream.pre_ffn_norm = layer.pre_feedforward_layernorm
    upstream.feed_forward = layer.mlp
    upstream.post_ffn_norm = layer.post_feedforward_layernorm
    upstream.add_ffn = layer.post_feedforward_residual_gate
    upstream.residual_dtype = torch.float32
    upstream.default_dtype = torch.bfloat16
    upstream.is_sequence_parallel = upstream.is_sequence_parallel_input = False
    upstream.all_gather_dual_norm = None

    stream = torch.randn(6, 64, generator=rng, dtype=torch.float32)
    positions = torch.arange(6)
    expected = norm_then_gated_residual(upstream(positions, stream))
    actual = layer(positions, stream)

    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-2)


@pytest.mark.parametrize("zero_centered", [False, True])
def test_weight_first_norm_matches_upstream(zero_centered):
    from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNormWithInputScale

    norm = MuseRMSNormWithInputScale(64, 1e-8, zero_centered=zero_centered)
    rng = torch.Generator().manual_seed(18)
    norm.weight.data.copy_(torch.randn(64, generator=rng, dtype=torch.float32))
    x = torch.randn(5, 64, generator=rng, dtype=torch.float32).bfloat16()

    actual = norm(x, out_dtype=torch.float32)
    expected = MuseSpark1xRMSNormWithInputScale.forward_native(
        norm, x, out_dtype=torch.float32
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
