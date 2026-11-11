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
"""CPU tests for the Muse Spark Open text model on TorchTPU."""

from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops import muse_moe_tpu
from vllm_torchtpu.models.vllm.muse_spark import text_model as tm
from vllm_torchtpu.models.vllm.muse_spark.configuration import MuseSparkTextConfig

OUTPUT_MULTIPLIER = 0.17677669529663687


@pytest.fixture(autouse=True)
def _cpu_model_tests(monkeypatch):
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return
    monkeypatch.setattr(tpu, "is_available", lambda: False)
    monkeypatch.setattr(tpu, "current_device", lambda: 0)
    monkeypatch.setattr(tpu, "device_count", lambda: 0)


def test_architectures_are_registered() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_torchtpu.models.vllm import register_models
    from vllm_torchtpu.models.vllm.muse_spark import (
        MuseSparkForCausalLM,
        MuseSparkForConditionalGeneration,
    )

    register_models()
    models = ModelRegistry.models
    assert (
        models["MuseSparkForConditionalGeneration"].load_model_cls()
        is MuseSparkForConditionalGeneration
    )
    assert models["MuseSparkForCausalLM"].load_model_cls() is MuseSparkForCausalLM
    for arch in ("MuseSparkForConditionalGeneration", "MuseSparkForCausalLM"):
        assert MODELS_CONFIG_MAP[arch] is tm.TPUMuseSparkConfig
    assert MuseSparkForConditionalGeneration.supports_multimodal
    assert MuseSparkForConditionalGeneration.get_placeholder_str("image", 0) == (
        "<|patch|>"
    )
    assert MuseSparkForConditionalGeneration.get_placeholder_str("audio", 0) == (
        "<|audio|>"
    )


def test_layer_schedule_matches_released_model() -> None:
    cfg = MuseSparkTextConfig()
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    # Every fourth layer counting back from the last is global NoPE attention.
    assert full == list(range(1, 62, 4))
    for i in range(cfg.num_hidden_layers):
        is_full = cfg.layer_types[i] == "full_attention"
        assert (cfg.layer_rope_theta[i] == 0) == is_full
    assert cfg.layer_types[61] == "full_attention"
    assert cfg.layer_types[60] == "sliding_attention"


def test_residual_gate_formula() -> None:
    gate = torch.tensor([-3.0, -0.2, 0.0, 0.4, 5.0])
    alpha, beta = tm.compute_residual_gate_alpha_beta(gate, 0.3, 1e-3)
    w = gate.double() / 0.3
    sig = 1.0 / (1.0 + torch.exp(-w))
    expect_alpha = torch.sqrt(torch.clamp((1.0 - sig) * (1.0 + sig), min=1e-3))
    torch.testing.assert_close(beta.double(), sig, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(alpha.double(), expect_alpha, rtol=1e-6, atol=1e-7)
    # Variance preserving: alpha^2 + beta^2 == 1 wherever the clamp is inactive.
    torch.testing.assert_close(
        (alpha.double() ** 2 + beta.double() ** 2)[:4],
        torch.ones(4, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )

    module = tm.MuseSparkResidualGate(
        SimpleNamespace(
            hidden_size=5, residual_gate_temperature=0.3, residual_dtype="fp32"
        )
    )
    module.gate.data.copy_(gate)
    module.process_weights_after_loading()
    residual = torch.randn(3, 5)
    hidden = torch.randn(3, 5, dtype=torch.bfloat16)
    out = module(residual, hidden)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, alpha * residual + beta * hidden.float())


def test_norms_match_reference_math() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    w = torch.randn(16) * 0.1

    centered = tm.MuseSparkCenteredRMSNorm(16, eps=1e-5)
    centered.weight.data.copy_(w)
    xf = x.float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * (1 + w)
    torch.testing.assert_close(centered(x), ref.to(torch.bfloat16))

    scaled = tm.MuseSparkInputScaledRMSNorm(16, eps=1e-8)
    scaled.weight.data.copy_(w)
    s = (xf * (1 + w)).to(torch.bfloat16).float()
    ref = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-8)
    torch.testing.assert_close(scaled(x), ref.to(torch.bfloat16))

    jax_scaled = muse_moe_tpu.muse_input_scaled_rms_norm_jax(
        _to_jax(x), _to_jax(w), 1e-8
    )
    torch.testing.assert_close(
        _from_jax(jax_scaled).float(), ref.to(torch.bfloat16).float()
    )


def test_checkpoint_name_mapping() -> None:
    from vllm_torchtpu.models.vllm.muse_spark import MuseSparkForCausalLM

    mapper = MuseSparkForCausalLM.hf_to_vllm_mapper
    names = [
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.mlp.experts.gate_up_proj",
        "model.language_model.layers.3.mlp.experts.gate_up_proj_weight_scale_2",
        "model.language_model.layers.3.mlp.experts.post_expert_norm.weight",
        "model.language_model.layers.3.mlp.gate.e_score_correction_bias",
        "model.language_model.layers.3.post_attention_residual_gate.gate",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.vision_tower.blocks.0.attn.q_proj.weight",
        "model.audio_tower.layers.0.conv_blocks.0.norm.weight",
        "lm_head.weight",
    ]
    assert mapper.apply_list(names) == [
        "model.layers.3.self_attn.qkv_proj.weight",
        "model.layers.3.mlp.experts.gate_up_proj",
        "model.layers.3.mlp.experts.gate_up_proj_weight_scale_2",
        "model.layers.3.mlp.post_expert_norm.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
        "model.layers.3.post_attention_residual_gate.gate",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]


def test_modelopt_excludes_are_normalized_into_ignore() -> None:
    hf = SimpleNamespace(
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "exclude_modules": ["layers.0.", "layers.61.", "self_attn", "lm_head"],
            "modules_to_not_convert": ["lm_head", "model.vision_tower.blocks.0.mlp"],
        }
    )
    tm.normalize_muse_spark_hf_quant_config(hf)
    assert hf.quantization_config["ignore"] == [
        "layers.0.",
        "layers.61.",
        "self_attn",
        "lm_head",
        "model.vision_tower.blocks.0.mlp",
    ]
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf),
        cache_config=SimpleNamespace(block_size=16, user_specified_block_size=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        speculative_config=None,
    )
    tm.TPUMuseSparkConfig.verify_and_update_config(vllm_cfg)
    assert vllm_cfg.cache_config.block_size == 128
    assert vllm_cfg.cache_config.user_specified_block_size is True


class _FakeRouted:
    """Records RoutedExperts.weight_loader calls for two of four local experts."""

    global_num_experts = 4

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, tuple[int, ...], float]] = []

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        return {2: 0, 3: 1}.get(expert_id, -1)

    def make_param(self, shape):
        param = torch.nn.Parameter(torch.zeros(shape), requires_grad=False)

        def loader(p, w, name, *, shard_id, expert_id):
            self.calls.append(
                (name, shard_id, expert_id, tuple(w.shape), float(w.float().sum()))
            )

        param.weight_loader = loader
        return param


@pytest.mark.parametrize(
    "suffix,attr",
    [
        ("", "w13_weight"),
        ("_weight_scale", "w13_weight_scale"),
        ("_weight_scale_2", "w13_weight_scale_2"),
    ],
)
def test_fused_gate_up_slices_local_experts(suffix: str, attr: str) -> None:
    routed = _FakeRouted()
    prefix = "model.layers.3.mlp.experts"
    params = {f"{prefix}.routed_experts.{attr}": routed.make_param((2, 1))}
    if suffix == "_weight_scale_2":
        tensor = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    else:
        tensor = torch.arange(4 * 6 * 3, dtype=torch.float32).reshape(4, 6, 3)
    loaded: set[str] = set()
    handled = tm._load_fused_3d_expert_tensor(
        f"{prefix}.gate_up_proj{suffix}", tensor, params, {prefix: routed}, loaded
    )
    assert handled
    assert loaded == {f"{prefix}.routed_experts.{attr}"}
    assert [(c[1], c[2]) for c in routed.calls] == [
        ("w1", 2),
        ("w3", 2),
        ("w1", 3),
        ("w3", 3),
    ]
    if suffix == "_weight_scale_2":
        assert [c[4] for c in routed.calls] == [4.0, 5.0, 6.0, 7.0]
    else:
        # Gate is the first half of the 2I rows, up the second.
        assert routed.calls[0][4] == float(tensor[2, :3].sum())
        assert routed.calls[1][4] == float(tensor[2, 3:].sum())
        assert routed.calls[0][3] == (3, 3)


def test_fused_down_proj_slices_local_experts() -> None:
    routed = _FakeRouted()
    prefix = "model.layers.0.mlp.experts"
    params = {f"{prefix}.routed_experts.w2_weight": routed.make_param((2, 1))}
    tensor = torch.randn(4, 5, 3)
    assert tm._load_fused_3d_expert_tensor(
        f"{prefix}.down_proj", tensor, params, {prefix: routed}, set()
    )
    assert [(c[1], c[2], c[3]) for c in routed.calls] == [
        ("w2", 2, (5, 3)),
        ("w2", 3, (5, 3)),
    ]


# Fused MoE op math (pure JAX on CPU) against a numpy reference.


def _to_jax(t: torch.Tensor):
    import jax.numpy as jnp

    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.float().numpy()).astype(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _from_jax(a) -> torch.Tensor:
    import jax.numpy as jnp

    if a.dtype == jnp.bfloat16:
        return torch.from_numpy(np.asarray(a.astype(jnp.float32)).copy()).to(
            torch.bfloat16
        )
    return torch.from_numpy(np.asarray(a).copy())


def _bf16(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(x.astype(np.float32)).to(torch.bfloat16).float().numpy()


def _numpy_muse_moe(
    h, router_w, bias, pre_w, pre_nw, w13, w2, post_nw, topk, start, local
):
    """HF MuseSparkMoE math; expert weights in HF [E, out, in] orientation."""
    h32 = h.astype(np.float64)
    logits = h32 @ router_w.astype(np.float64).T * OUTPUT_MULTIPLIER
    scores = 1.0 / (1.0 + np.exp(-logits))
    choice = scores + bias[None, :]
    ids = np.argsort(-choice, axis=-1, kind="stable")[:, :topk]
    wts = np.take_along_axis(scores, ids, axis=-1)
    wts = wts / (wts.sum(-1, keepdims=True) + 1e-15)

    z = _bf16(h32 @ pre_w.astype(np.float64).T)
    z = _bf16(z * (1.0 / np.sqrt((z**2).mean(-1, keepdims=True) + 1e-5)) * (1 + pre_nw))
    out = np.zeros((h.shape[0], pre_w.shape[0]))
    inter = w2.shape[-1]
    for t in range(h.shape[0]):
        for k in range(topk):
            e = int(ids[t, k])
            if not (start <= e < start + local):
                continue
            gu = _bf16(z[t] @ w13[e].astype(np.float64).T)
            gate, up = gu[:inter], gu[inter:]
            act = _bf16(gate / (1.0 + np.exp(-gate)) * up)
            down = _bf16(act @ w2[e].astype(np.float64).T)
            s = _bf16(down * (1 + post_nw))
            y = _bf16(s / np.sqrt((s**2).mean() + 1e-8))
            out[t] += wts[t, k] * y
    return out


@pytest.mark.parametrize("start,local", [(0, 8), (4, 4)])
def test_fused_moe_jax_matches_numpy(start: int, local: int) -> None:
    import jax.numpy as jnp

    rng = np.random.default_rng(start)
    tokens, hidden, moe_hidden, inter, experts, topk = 6, 32, 16, 8, 8, 2
    h = _bf16(rng.standard_normal((tokens, hidden)))
    router_w = rng.standard_normal((experts, hidden)).astype(np.float32)
    bias = (rng.standard_normal(experts) * 0.05).astype(np.float32)
    pre_w = _bf16(rng.standard_normal((moe_hidden, hidden)) / math.sqrt(hidden))
    pre_nw = _bf16(rng.standard_normal(moe_hidden) * 0.1)
    w13 = _bf16(rng.standard_normal((experts, 2 * inter, moe_hidden)) / 4)
    w2 = _bf16(rng.standard_normal((experts, moe_hidden, inter)) / 3)
    post_nw = _bf16(rng.standard_normal(moe_hidden) * 0.1)

    ref = _numpy_muse_moe(
        h, router_w, bias, pre_w, pre_nw, w13, w2, post_nw, topk, start, local
    )

    bf = jnp.bfloat16
    out = muse_moe_tpu._muse_moe_jax(
        jnp.asarray(h).astype(bf),
        jnp.asarray(router_w.T),  # K major, as router_w_kmajor
        jnp.asarray(bias),
        jnp.asarray(pre_w.T).astype(bf),  # K major, as after the linear transpose
        jnp.asarray(pre_nw).astype(bf),
        # TPU kernel layout: w13 [E, K, 2I], w2 [E, I, H].
        jnp.asarray(np.transpose(w13[start : start + local], (0, 2, 1))).astype(bf),
        None,
        jnp.asarray(np.transpose(w2[start : start + local], (0, 2, 1))).astype(bf),
        None,
        jnp.asarray(post_nw).astype(bf),
        jnp.asarray(start, dtype=jnp.int32),
        topk=topk,
        use_ep=local != experts,
        use_sparse_core=False,
    )
    assert out.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(out), ref, rtol=3e-2, atol=3e-2)


def test_router_bias_only_affects_selection() -> None:
    import jax.numpy as jnp

    h = jnp.ones((1, 4), jnp.float32)
    router_w = jnp.asarray(np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.float32))
    bias = jnp.asarray([10.0, 0.0, 0.0, 0.0], jnp.float32)
    weights, ids = muse_moe_tpu.muse_route_jax(h, router_w, bias, topk=2)
    assert sorted(np.asarray(ids)[0].tolist()) == [0, 3]
    s = 1.0 / (1.0 + np.exp(-np.array([1.0, 4.0]) * OUTPUT_MULTIPLIER))
    expected = dict(zip([0, 3], s / s.sum()))
    got = dict(zip(np.asarray(ids)[0].tolist(), np.asarray(weights)[0].tolist()))
    for k, v in expected.items():
        assert got[k] == pytest.approx(v, rel=1e-6)


def test_logits_multiplier_precedes_softcap() -> None:
    from vllm_torchtpu.models.vllm.muse_spark import MuseSparkForCausalLM

    model = MuseSparkForCausalLM.__new__(MuseSparkForCausalLM)
    torch.nn.Module.__init__(model)
    model.output_multiplier = OUTPUT_MULTIPLIER
    model.final_logit_softcapping = 20.0
    raw = torch.tensor([[-500.0, -10.0, 0.0, 50.0, 500.0]])
    model.logits_processor = lambda head, hs: hs
    model.lm_head = None
    expected = 20.0 * torch.tanh(raw * OUTPUT_MULTIPLIER / 20.0)
    torch.testing.assert_close(model.compute_logits(raw), expected)


def test_dspark_aux_layers_default() -> None:
    from vllm_torchtpu.models.vllm.muse_spark import MuseSparkForCausalLM

    model = MuseSparkForCausalLM.__new__(MuseSparkForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace(layers=[None] * 62)
    assert model.get_eagle3_default_aux_hidden_state_layers() == (14, 30, 46, 58, 62)
    model.set_aux_hidden_state_layers((14, 30, 46, 58, 62))
    assert model.model.aux_hidden_state_layers == (14, 30, 46, 58, 62)


def test_set_aux_hidden_state_layers_stores_ids_as_given() -> None:
    from vllm_torchtpu.models.vllm.muse_spark import MuseSparkForCausalLM
    from vllm_torchtpu.models.vllm.muse_spark.configuration import (
        MuseSparkDSparkConfig,
    )

    model = MuseSparkForCausalLM.__new__(MuseSparkForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace(layers=[None] * 62)
    # dflash.load_model passes target_layer_id + 1. The released draft's ids
    # tag the same layers as before (outputs of layers 13, 29, 45, 57, 61).
    released = MuseSparkDSparkConfig().target_layer_ids
    assert released == [13, 29, 45, 57, 61]
    model.set_aux_hidden_state_layers(tuple(i + 1 for i in released))
    assert model.model.aux_hidden_state_layers == (14, 30, 46, 58, 62)
    # A draft that does not tap the last layer is not shifted a second time.
    model.set_aux_hidden_state_layers((2, 31, 59))
    assert model.model.aux_hidden_state_layers == (2, 31, 59)
    model.set_aux_hidden_state_layers((0, 62))
    assert model.model.aux_hidden_state_layers == (0, 62)
    for bad in ((63,), (-1, 5)):
        with pytest.raises(ValueError, match="Invalid Muse Spark aux"):
            model.set_aux_hidden_state_layers(bad)


class _FakeQuantMethod:
    pass


def test_eager_moe_rejects_quantized_routed_experts() -> None:
    from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod

    def fake_moe(method):
        return SimpleNamespace(
            prefix="model.layers.3.mlp",
            experts=SimpleNamespace(
                routed_experts=SimpleNamespace(quant_method=method)
            ),
        )

    quantized = fake_moe(_FakeQuantMethod())
    with pytest.raises(NotImplementedError, match="_FakeQuantMethod"):
        tm.MuseSparkMoE._check_eager_supported(quantized)
    # The eager forward refuses before touching the packed weights.
    quantized._check_eager_supported = lambda: tm.MuseSparkMoE._check_eager_supported(
        quantized
    )
    with pytest.raises(NotImplementedError, match="VLLM_TPU_MUSE_MOE_IMPL=fused"):
        tm.MuseSparkMoE._forward_eager(quantized, torch.zeros(2, 4))

    unquantized = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)
    tm.MuseSparkMoE._check_eager_supported(fake_moe(unquantized))


def test_normalize_quant_config_targets_concrete_exclusion_list() -> None:
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    from vllm_torchtpu.models.vllm.muse_spark import MuseSparkForCausalLM

    hf = SimpleNamespace(
        quantization_config={"ignore": ["model.language_model.layers.0.mlp"]}
    )
    mapper = MuseSparkForCausalLM.hf_to_vllm_mapper
    packed = MuseSparkForCausalLM.packed_modules_mapping

    modelopt = ModelOptNvFp4Config.__new__(ModelOptNvFp4Config)
    modelopt.packed_modules_mapping = {}
    modelopt.exclude_modules = ["lm_head"]
    tm.normalize_muse_spark_quant_config(modelopt, hf, mapper, packed)
    assert modelopt.packed_modules_mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
    assert modelopt.exclude_modules == [
        "lm_head",
        "model.language_model.layers.0.mlp",
        "model.layers.0.mlp",
    ]

    fp8 = Fp8Config.__new__(Fp8Config)
    fp8.packed_modules_mapping = {}
    fp8.ignored_layers = []
    tm.normalize_muse_spark_quant_config(fp8, hf, mapper, packed)
    assert fp8.ignored_layers == [
        "model.language_model.layers.0.mlp",
        "model.layers.0.mlp",
    ]


# Tiny model parity against an optional standalone JAX reference (muse_ref.py).

_MUSE_REF = os.environ.get("MUSE_REF_PATH", "")

_CKPT_NAMES = {
    "input_ln": "input_layernorm.weight",
    "pre_ff_ln": "pre_feedforward_layernorm.weight",
    "post_ff_ln": "post_feedforward_layernorm.weight",
    "q": "self_attn.q_proj.weight",
    "k": "self_attn.k_proj.weight",
    "v": "self_attn.v_proj.weight",
    "gate": "self_attn.gate_proj.weight",
    "o": "self_attn.o_proj.weight",
    "attn_gate": "post_attention_residual_gate.gate",
    "ffn_gate": "post_feedforward_residual_gate.gate",
    "router": "mlp.gate.weight",
    "router_bias": "mlp.gate.e_score_correction_bias",
    "pre_expert": "mlp.pre_expert_proj.weight",
    "pre_expert_norm": "mlp.pre_expert_norm.weight",
    "post_expert": "mlp.post_expert_proj.weight",
    "post_expert_norm": "mlp.experts.post_expert_norm.weight",
}


class _DenseAttention(torch.nn.Module):
    """Causal (optionally sliding) attention over one sequence, no KV cache."""

    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads=None,
        per_layer_sliding_window=None,
        **kwargs,
    ):
        super().__init__()
        self.h, self.d, self.kv = num_heads, head_size, num_kv_heads
        self.scale, self.window = scale, per_layer_sliding_window

    def forward(self, q, k, v):
        t, dtype = q.shape[0], q.dtype
        q = q.view(t, self.h, self.d).float()
        k = k.view(t, self.kv, self.d).float().repeat_interleave(self.h // self.kv, 1)
        v = v.view(t, self.kv, self.d).float().repeat_interleave(self.h // self.kv, 1)
        s = torch.einsum("qhd,khd->hqk", q, k) * self.scale
        i = torch.arange(t)
        keep = i[None, :] <= i[:, None]
        if self.window:
            keep &= (i[:, None] - i[None, :]) < self.window
        p = s.masked_fill(~keep, -float("inf")).softmax(-1)
        return torch.einsum("hqk,khd->qhd", p, v).reshape(t, -1).to(dtype)


@pytest.fixture
def _single_rank_dist(monkeypatch):
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(current_platform, "device_type", "cpu", raising=False)
    if not torch.distributed.is_initialized():
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="gloo",
        )
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)


@pytest.mark.skipif(
    not _MUSE_REF or not os.path.exists(_MUSE_REF),
    reason="MUSE_REF_PATH not configured",
)
def test_tiny_model_matches_jax_reference(_single_rank_dist, monkeypatch) -> None:
    import importlib.util

    from vllm.config import CompilationMode, VllmConfig, set_current_vllm_config

    from vllm_torchtpu.models.vllm.muse_spark import MuseSparkForCausalLM

    spec = importlib.util.spec_from_file_location("muse_ref", _MUSE_REF)
    ref_mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "muse_ref", ref_mod)
    spec.loader.exec_module(ref_mod)

    n, seq = 4, 10
    cfg = MuseSparkTextConfig(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=n,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_hidden_size=32,
        moe_intermediate_size=16,
        num_local_experts=8,
        num_experts_per_tok=2,
        sliding_window=4,
        max_position_embeddings=256,
    )
    rcfg = ref_mod.Config(
        hidden_size=64,
        num_layers=n,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        vocab_size=64,
        moe_hidden_size=32,
        moe_intermediate_size=16,
        num_experts=8,
        top_k=2,
        sliding_window=4,
        layer_types=tuple(cfg.layer_types),
        layer_rope_theta=tuple(float(x) for x in cfg.layer_rope_theta),
        bos_token_id=0,
        pad_token_id=63,
    )
    glob, layers = ref_mod.synthetic_weights(rcfg, seed=1)

    def t(a, keep_fp32=False):
        x = torch.from_numpy(np.asarray(a, np.float32))
        return x if keep_fp32 else x.bfloat16()

    pre = "model.language_model."
    weights = [
        (pre + "embed_tokens.weight", t(glob["embed"])),
        (pre + "norm.weight", t(glob["norm"])),
        ("lm_head.weight", t(glob["lm_head"])),
    ]
    for i, layer in enumerate(layers):
        for key, name in _CKPT_NAMES.items():
            fp32 = np.asarray(layer[key]).dtype == np.float32
            weights.append((f"{pre}layers.{i}.{name}", t(layer[key], fp32)))
        experts = layer["experts"]
        weights.append(
            (f"{pre}layers.{i}.mlp.experts.gate_up_proj", t(experts["gate_up"]))
        )
        weights.append((f"{pre}layers.{i}.mlp.experts.down_proj", t(experts["down"])))

    vc = VllmConfig()
    vc.compilation_config.mode = CompilationMode.NONE
    vc.model_config = SimpleNamespace(
        hf_config=cfg,
        hf_text_config=cfg,
        dtype=torch.bfloat16,
        head_dtype=None,
        multimodal_config=None,
        get_multimodal_config=lambda: None,
    )
    monkeypatch.setattr(tm, "Attention", _DenseAttention)
    with set_current_vllm_config(vc):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = MuseSparkForCausalLM(vllm_config=vc, prefix="")
        finally:
            torch.set_default_dtype(default_dtype)
        loaded = model.load_weights(iter(weights))
        assert {k for k, _ in model.named_parameters()} <= loaded

        ids = torch.tensor([3, 17, 5, 60, 2, 9, 41, 33, 7, 1])
        ref = ref_mod.forward_prefill(
            glob,
            list(enumerate(layers)),
            [ids.tolist()],
            rcfg,
            mode="bf16",
            residual_fp32=True,
            chunked=False,
            tap_layers=list(range(n)),
        )
        taps = [np.asarray(ref["taps"][i][0]) for i in range(n)]
        with torch.no_grad():
            emb = model.model.embed_input_ids(ids).float().numpy()
            np.testing.assert_allclose(
                emb,
                np.asarray(ref_mod.embed(glob, ids.tolist(), rcfg, "bf16", True)),
                atol=1e-5,
            )
            # Layer by layer on the reference input: the stack amplifies bf16
            # noise chaotically with random weights, so compare each layer alone.
            x = torch.from_numpy(emb)
            for i in range(n):
                out = model.model.layers[i](torch.arange(seq), x).float().numpy()
                np.testing.assert_allclose(out, taps[i], atol=6e-2, rtol=0)
                x = torch.from_numpy(taps[i].copy())
            final = model.model.norm(x).bfloat16()
            logits = model.compute_logits(final).float().numpy()
        np.testing.assert_allclose(
            logits, np.asarray(ref["logits"][0], np.float32), atol=6e-2, rtol=0
        )


def test_hd64_wrapper_slices_padded_qkv_only() -> None:
    from vllm_torchtpu.models.vllm.muse_spark.attention_compat import (
        wrap_hd64_rpa_kernel,
    )

    calls = []

    def inner(kv_cache, q, k, v, *rest, **kwargs):
        calls.append((kv_cache, q, k, v, rest, kwargs))
        return q

    kv = torch.zeros(3, 4, 1, 2, 128)
    q = torch.cat([torch.randn(5, 4, 64), torch.zeros(5, 4, 64)], dim=-1)
    k = torch.cat([torch.randn(5, 2, 64), torch.zeros(5, 2, 64)], dim=-1)
    v = torch.cat([torch.randn(5, 2, 64), torch.zeros(5, 2, 64)], dim=-1)
    rest = (torch.arange(2), torch.arange(3), None)
    out = wrap_hd64_rpa_kernel(inner)(kv, q, k, v, *rest, flag=1)

    got_kv, got_q, got_k, got_v, got_rest, got_kwargs = calls[0]
    assert got_kv is kv
    for got, ref in ((got_q, q), (got_k, k), (got_v, v)):
        assert got.shape[-1] == 64
        torch.testing.assert_close(got, ref[..., :64])
    assert got_rest == rest and got_kwargs == {"flag": 1}
    assert out.shape == (5, 4, 64)
    # Already 64 wide inputs pass through untouched.
    wrap_hd64_rpa_kernel(inner)(kv, q[..., :64], k[..., :64], v[..., :64])
    assert calls[1][1].data_ptr() == q.data_ptr()


def test_hd64_install_swaps_only_plain_pallas_impl(monkeypatch) -> None:
    from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
    from vllm_torchtpu.models.vllm.muse_spark import attention_compat as ac

    def fake_attn(cls, head_size):
        impl = cls.__new__(cls)
        impl.head_size = head_size
        return SimpleNamespace(impl=impl)

    hd64 = fake_attn(PallasAttentionBackendImpl, 64)
    assert ac.install_muse_hd64_attention(hd64)
    assert type(hd64.impl) is ac.MuseHd64PallasAttentionImpl
    with pytest.raises(NotImplementedError, match="bundled"):
        hd64.impl.setup_bundled(0, torch.device("cpu"), None)

    hd128 = fake_attn(PallasAttentionBackendImpl, 128)
    assert not ac.install_muse_hd64_attention(hd128)
    assert type(hd128.impl) is PallasAttentionBackendImpl

    class OtherImpl(PallasAttentionBackendImpl):
        pass

    other = fake_attn(OtherImpl, 64)
    assert not ac.install_muse_hd64_attention(other)
    assert not ac.install_muse_hd64_attention(SimpleNamespace())

    # The overridden builder wraps (and caches) the base kernel for hd64.
    def inner_fn(kv, q, k, v, *rest):
        return (q.shape[-1], k.shape[-1], v.shape[-1])

    monkeypatch.setattr(
        PallasAttentionBackendImpl,
        "_build_rpa_kernel",
        lambda self, *a, **kw: inner_fn,
    )
    k1 = hd64.impl._build_rpa_kernel(None, None, None)
    assert k1 is hd64.impl._build_rpa_kernel(None, None, None)
    pad = torch.zeros(2, 1, 128)
    assert k1(None, pad, pad, pad) == (64, 64, 64)

    import inspect

    base_params = list(
        inspect.signature(PallasAttentionBackendImpl._kernel_entry).parameters
    )
    hd64.impl._build_rpa_kernel(None, None, None)
    assert hd64.impl._kernel_entry is ac._muse_rpa_kernel_entry
    assert list(inspect.signature(hd64.impl._kernel_entry).parameters) == base_params


# Tiny model on CPU against a hand-written torch reference (no external files).

_TINY = dict(
    vocab_size=64,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    moe_hidden_size=32,
    moe_intermediate_size=16,
    num_local_experts=8,
    num_experts_per_tok=2,
    sliding_window=4,
    max_position_embeddings=256,
    routed_scaling_factor=1.5,
)


def _tiny_weights(cfg: MuseSparkTextConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    """Synthetic weights keyed like _CKPT_NAMES (plus embed/norm/lm_head)."""
    g = torch.Generator().manual_seed(seed)
    h, hd = cfg.hidden_size, cfg.head_dim
    nq, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hm, im, e = cfg.moe_hidden_size, cfg.moe_intermediate_size, cfg.num_local_experts

    def bf(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).bfloat16()

    def f32(*shape, scale=1.0):
        return torch.randn(*shape, generator=g) * scale

    w = {
        "embed": bf(cfg.vocab_size, h),
        "norm": bf(h, scale=0.1),
        "lm_head": bf(cfg.vocab_size, h, scale=h**-0.5),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        w |= {
            p + "input_ln": bf(h, scale=0.1),
            p + "pre_ff_ln": bf(h, scale=0.1),
            p + "post_ff_ln": bf(h, scale=0.1),
            p + "q": bf(nq * hd, h, scale=h**-0.5),
            p + "k": bf(nkv * hd, h, scale=h**-0.5),
            p + "v": bf(nkv * hd, h, scale=h**-0.5),
            p + "gate": bf(nq * hd, h, scale=h**-0.5),
            p + "o": bf(h, nq * hd, scale=(nq * hd) ** -0.5),
            p + "attn_gate": f32(h),
            p + "ffn_gate": f32(h),
            p + "router": f32(e, h, scale=h**-0.5),
            p + "router_bias": f32(e, scale=0.05),
            p + "pre_expert": bf(hm, h, scale=h**-0.5),
            p + "pre_expert_norm": bf(hm, scale=0.1),
            p + "post_expert": bf(h, hm, scale=hm**-0.5),
            p + "post_expert_norm": bf(hm, scale=0.1),
            p + "gate_up": bf(e, 2 * im, hm, scale=hm**-0.5),
            p + "down": bf(e, hm, im, scale=im**-0.5),
        }
    return w


def _tiny_checkpoint(cfg, w):
    pre = "model.language_model."
    out = [
        (pre + "embed_tokens.weight", w["embed"]),
        (pre + "norm.weight", w["norm"]),
        ("lm_head.weight", w["lm_head"]),
    ]
    for i in range(cfg.num_hidden_layers):
        p = f"layers.{i}."
        for key, name in _CKPT_NAMES.items():
            out.append((f"{pre}{p}{name}", w[p + key]))
        out.append((f"{pre}{p}mlp.experts.gate_up_proj", w[p + "gate_up"]))
        out.append((f"{pre}{p}mlp.experts.down_proj", w[p + "down"]))
    return out


def _build_tiny_model(monkeypatch, cfg):
    from vllm.config import CompilationMode, VllmConfig, set_current_vllm_config

    from vllm_torchtpu.models.vllm.muse_spark import MuseSparkForCausalLM

    vc = VllmConfig()
    vc.compilation_config.mode = CompilationMode.NONE
    vc.model_config = SimpleNamespace(
        hf_config=cfg,
        hf_text_config=cfg,
        dtype=torch.bfloat16,
        head_dtype=None,
        multimodal_config=None,
        get_multimodal_config=lambda: None,
    )
    monkeypatch.setattr(tm, "Attention", _DenseAttention)
    w = _tiny_weights(cfg)
    with set_current_vllm_config(vc):
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = MuseSparkForCausalLM(vllm_config=vc, prefix="")
        finally:
            torch.set_default_dtype(default_dtype)
        expected = {k for k, _ in model.named_parameters()}
        loaded = model.load_weights(iter(_tiny_checkpoint(cfg, w)))
    assert expected <= loaded
    return model, w


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _lin(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.float() @ weight.float().t()).bfloat16()


def _ref_rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    ang = pos.float()[:, None] * inv[None, :]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _ref_moe(cfg, w, p, x: torch.Tensor) -> torch.Tensor:
    """HF MuseSparkMoE: sigmoid router with selection-only bias, compressed
    bottleneck, per expert InputScaledRMSNorm, post projection."""
    logits = x.float() @ w[p + "router"].float().t() * cfg.output_multiplier
    scores = torch.sigmoid(logits)
    _, ids = torch.topk(scores + w[p + "router_bias"], cfg.num_experts_per_tok)
    wts = scores.gather(1, ids)
    wts = wts / (wts.sum(-1, keepdim=True) + 1e-15) * cfg.routed_scaling_factor
    z = _lin(x, w[p + "pre_expert"])
    z = (_rms(z, cfg.rms_norm_eps) * (1 + w[p + "pre_expert_norm"].float())).bfloat16()
    mixed = torch.zeros(x.shape[0], cfg.moe_hidden_size)
    for t in range(x.shape[0]):
        for k in range(cfg.num_experts_per_tok):
            e = int(ids[t, k])
            gate, up = _lin(z[t : t + 1], w[p + "gate_up"][e]).chunk(2, dim=-1)
            act = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
            down = _lin(act, w[p + "down"][e])
            s = (down.float() * (1 + w[p + "post_expert_norm"].float())).bfloat16()
            y = _rms(s, cfg.post_norm_eps).bfloat16()
            mixed[t] += wts[t, k] * y[0].float()
    return _lin(mixed.bfloat16(), w[p + "post_expert"])


def _ref_layer(cfg, w, i: int, x: torch.Tensor) -> torch.Tensor:
    """One decoder layer on the fp32 residual stream x [T, H]."""
    p = f"layers.{i}."
    t, hd = x.shape[0], cfg.head_dim
    nq, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    pos = torch.arange(t)

    def gate_ab(g):
        beta = torch.sigmoid(g / cfg.residual_gate_temperature)
        alpha = torch.sqrt(torch.clamp((1 - beta) * (1 + beta), min=1e-3))
        return alpha, beta

    h = (_rms(x, cfg.rms_norm_eps) * (1 + w[p + "input_ln"].float())).bfloat16()
    q = _rms(_lin(h, w[p + "q"]).view(t, nq, hd), cfg.rms_norm_eps).bfloat16()
    q = (q * cfg.qk_scale_factor).bfloat16()
    k = _rms(_lin(h, w[p + "k"]).view(t, nkv, hd), cfg.rms_norm_eps).bfloat16()
    v = _lin(h, w[p + "v"]).view(t, nkv, hd)
    theta = cfg.layer_rope_theta[i]
    if theta:
        q = _ref_rope(q.float(), pos, theta).bfloat16()
        k = _ref_rope(k.float(), pos, theta).bfloat16()
    window = cfg.sliding_window if cfg.layer_types[i] == "sliding_attention" else None
    attn = _DenseAttention(
        nq, hd, hd**-0.5, num_kv_heads=nkv, per_layer_sliding_window=window
    )
    a = attn(q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1))
    a = _rms(a.view(t, nq, hd), cfg.rms_norm_eps).bfloat16().reshape(t, -1)
    a = (a * torch.sigmoid(_lin(h, w[p + "gate"]))).bfloat16()
    a = _rms(_lin(a, w[p + "o"]), cfg.post_norm_eps).bfloat16()
    alpha, beta = gate_ab(w[p + "attn_gate"])
    x = alpha * x + beta * a.float()

    h = (_rms(x, cfg.rms_norm_eps) * (1 + w[p + "pre_ff_ln"].float())).bfloat16()
    ff = _ref_moe(cfg, w, p, h)
    s = (ff.float() * (1 + w[p + "post_ff_ln"].float())).bfloat16()
    ff = _rms(s, cfg.post_norm_eps).bfloat16()
    alpha, beta = gate_ab(w[p + "ffn_gate"])
    return alpha * x + beta * ff.float()


@pytest.mark.parametrize("moe_impl", ["fused", "eager"])
def test_tiny_decoder_layers_match_torch_reference(
    _single_rank_dist, monkeypatch, moe_impl: str
) -> None:
    # On CPU the fused op is never resolved, so both settings run the eager
    # MoE; "eager" also runs the construction-time quantization check.
    monkeypatch.setattr(muse_moe_tpu, "muse_moe_impl", lambda: moe_impl)
    cfg = MuseSparkTextConfig(**_TINY)
    model, w = _build_tiny_model(monkeypatch, cfg)
    ids = torch.tensor([3, 17, 5, 60, 2, 9, 41, 33, 7, 1])
    seq = ids.shape[0]
    # Layer 0 is sliding window RoPE attention, layer 3 global NoPE.
    assert cfg.layer_types[0] == "sliding_attention" and cfg.layer_rope_theta[0]
    assert cfg.layer_types[3] == "full_attention" and not cfg.layer_rope_theta[3]
    with torch.no_grad():
        emb = model.model.embed_input_ids(ids)
        assert emb.dtype == torch.float32
        torch.testing.assert_close(emb, _rms(w["embed"][ids], cfg.rms_norm_eps))
        x = emb
        for i in range(cfg.num_hidden_layers):
            out = model.model.layers[i](torch.arange(seq), x)
            ref = _ref_layer(cfg, w, i, x)
            assert out.dtype == torch.float32
            torch.testing.assert_close(out, ref, atol=3e-2, rtol=0)
            x = ref

        moe = model.model.layers[1].mlp
        assert isinstance(moe, tm.MuseSparkMoE) and moe._fused_moe() is None
        h = torch.randn(seq, cfg.hidden_size).bfloat16()
        torch.testing.assert_close(
            moe(h).float(),
            _ref_moe(cfg, w, "layers.1.", h).float(),
            atol=3e-2,
            rtol=0,
        )


def test_moe_process_weights_drops_cached_fused_op(
    _single_rank_dist, monkeypatch
) -> None:
    cfg = MuseSparkTextConfig(**_TINY)
    model, _ = _build_tiny_model(monkeypatch, cfg)
    moe = model.model.layers[0].mlp
    stale = object()
    moe.__dict__["_fused_moe_args"] = stale
    # The model-level hook vLLM runs after quant finalize re-resolves the op.
    model.process_weights_after_loading()
    assert moe.__dict__["_fused_moe_args"] is not stale


def test_forward_sharded_last_layer_without_fused_moe(
    _single_rank_dist, monkeypatch
) -> None:
    """With fused MoE off the last layer takes the dense branch, which must
    still return the gathered residual instead of asserting a next layer."""
    cfg = MuseSparkTextConfig(**_TINY)
    model, _ = _build_tiny_model(monkeypatch, cfg)
    layers = model.model.layers
    for layer in layers:
        layer._tp_shard_idx = 0
        for name, src in (
            ("_in_norm_w_s", layer.input_layernorm.weight),
            ("_pre_ffn_w_s", layer.pre_feedforward_layernorm.weight),
            ("_post_ffn_w_s", layer.post_feedforward_layernorm.weight),
            ("_attn_alpha_s", layer.post_attention_residual_gate.alpha),
            ("_attn_beta_s", layer.post_attention_residual_gate.beta),
            ("_ffn_alpha_s", layer.post_feedforward_residual_gate.alpha),
            ("_ffn_beta_s", layer.post_feedforward_residual_gate.beta),
        ):
            setattr(layer, name, src.detach().reshape(1, -1))

    def fake_attn_post(attn_out, h_cur, gate_w, o_w, res_s, *args, **kwargs):
        assert "pre_exp_w" not in kwargs  # dense branch
        return res_s + 1.0, h_cur

    calls = []

    def fake_dense_post(ff_s, res_s, pnw, a, b, next_w, **kwargs):
        calls.append((next_w, kwargs))
        hid = res_s * 2.0
        h_next = ff_s.bfloat16()
        return (hid, h_next, hid * 3.0) if kwargs["gather_hid"] else (hid, h_next)

    monkeypatch.setattr(tm, "det_attn_post_sandwich", fake_attn_post)
    monkeypatch.setattr(tm, "det_dense_post_sandwich", fake_dense_post)

    seq = 5
    pos = torch.arange(seq)
    h_cur = torch.randn(seq, cfg.hidden_size).bfloat16()
    res_s = torch.randn(seq, cfg.hidden_size)
    last, prev = layers[-1], layers[-2]
    assert last.mlp._fused_moe() is None
    with torch.no_grad():
        _, out_res, hid_full = last.forward_sharded(pos, h_cur, res_s, None)
    next_w, kwargs = calls[-1]
    assert next_w is last._in_norm_w_s
    assert kwargs["gather_hid"] is True
    assert kwargs["next_in_eps"] == last.input_layernorm.eps
    torch.testing.assert_close(out_res, (res_s + 1.0) * 2.0)
    torch.testing.assert_close(hid_full, (res_s + 1.0) * 6.0)

    # Middle layers keep using the next layer's norm and the caller's gather.
    with torch.no_grad():
        _, _, hid = prev.forward_sharded(pos, h_cur, res_s, last)
    next_w, kwargs = calls[-1]
    assert next_w is last._in_norm_w_s and kwargs["gather_hid"] is False
    assert hid is None
