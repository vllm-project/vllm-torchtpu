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
"""Unit tests for the OLMo 3.5 model: registration, config, weight loading
and the model-specific forward math, on CPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.utils import set_weight_attrs

import vllm_torchtpu.models.vllm.olmo35.config as config_mod
import vllm_torchtpu.models.vllm.olmo35.model as model_mod
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.olmo35.config import Olmo35Config


def test_olmo35_is_registered():
    """The arch resolves from the vLLM registry and `model_type: olmo35`
    resolves from AutoConfig, since transformers has no class for it."""
    register_models()
    entry = ModelRegistry.models.get("Olmo35ForCausalLM")
    assert entry is not None, "Olmo35ForCausalLM is not registered"
    assert entry.load_model_cls() is model_mod.Olmo35ForCausalLM
    assert isinstance(AutoConfig.for_model("olmo35"), Olmo35Config)
    # Registering twice in one process is a no-op, not an error.
    register_models()


def test_default_layer_types_follow_the_full_attention_interval():
    """Full attention sits where (layer + 1) % interval == 0, as in MaxText;
    the remaining layers are KDA."""
    config = Olmo35Config()
    full = [i for i, t in enumerate(config.layer_types) if t == "full_attention"]
    assert full == [7, 15, 23, 31, 39]
    assert config.layer_types.count("linear_attention") == 35

    config = Olmo35Config(num_hidden_layers=8, full_attention_interval=4)
    assert config.layer_types == (["linear_attention"] * 3 + ["full_attention"]) * 2


def test_routed_scaling_factor_defaults_to_top_k():
    """OLMo-core rescales the renormalized top-k weights to sum to top_k."""
    assert Olmo35Config(num_experts_per_tok=8).routed_scaling_factor == 8.0
    assert Olmo35Config(routed_scaling_factor=1.0).routed_scaling_factor == 1.0


def test_layer_types_must_cover_every_layer():
    with pytest.raises(ValueError, match="one entry per layer"):
        Olmo35Config(num_hidden_layers=4, layer_types=["linear_attention"] * 3)


@pytest.mark.parametrize(
    ("total_heads", "tp_size", "expected_starts"),
    [
        # TP above the head count: each head is replicated on tp/heads ranks,
        # matching QKVParallelLinear's KV-head replication.
        (4, 8, [0, 0, 1, 1, 2, 2, 3, 3]),
        # TP at or below the head count: contiguous blocks of heads.
        (8, 4, [0, 2, 4, 6]),
        (8, 8, list(range(8))),
    ],
)
def test_head_sharded_loader_selects_this_ranks_heads(
    monkeypatch, total_heads, tp_size, expected_starts
):
    head_dim = 3
    local = max(1, total_heads // tp_size)
    full = torch.arange(total_heads * head_dim, dtype=torch.float32)
    loader = model_mod._head_sharded_loader(total_heads)
    monkeypatch.setattr(
        model_mod, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    for rank, start in enumerate(expected_starts):
        monkeypatch.setattr(
            model_mod, "get_tensor_model_parallel_rank", lambda r=rank: r
        )
        parameter = torch.empty(local, head_dim)
        loader(parameter, full)
        expected = full.reshape(total_heads, head_dim)[start : start + local]
        torch.testing.assert_close(parameter, expected)


def test_register_rejects_a_different_class_under_the_same_name(monkeypatch):
    """A repeat registration is a no-op only for the same class; a clash with
    another config class is an error, not a silent fallback."""
    register_models()
    config_mod.register_olmo35_config()  # Same class: no-op.

    class OtherConfig(Olmo35Config):
        pass

    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "olmo35", OtherConfig)
    with pytest.raises(ValueError, match="already registered to"):
        config_mod.register_olmo35_config()


def test_tied_word_embeddings_are_rejected():
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        Olmo35Config(tie_word_embeddings=True)


def test_unified_pool_is_rejected_at_construction(monkeypatch):
    """The unsupported unified KV pool fails when the layer is built, before
    weights load, not at the first forward."""
    monkeypatch.setattr(
        model_mod, "unified_kv_layout_enabled_for_architecture", lambda arch: True
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architecture="Olmo35ForCausalLM"),
        speculative_config=None,
    )
    with pytest.raises(NotImplementedError, match="unified-pool"):
        model_mod.Olmo35KimiDeltaAttention(vllm_config, "model.layers.0.mixer")


def _recording_parameter(calls, *shape):
    parameter = nn.Parameter(torch.zeros(*shape), requires_grad=False)

    def weight_loader(param, loaded_weight, *args, **kwargs):
        calls.append((param, loaded_weight.clone(), args, kwargs))

    set_weight_attrs(parameter, {"weight_loader": weight_loader})
    return parameter


def _tiny_model_for_loading(routed_scale):
    """Skeleton with one KDA layer (0) and one attention + MoE layer (1),
    holding only the parameters `load_weights` routes to."""
    model = model_mod.Olmo35ForCausalLM.__new__(model_mod.Olmo35ForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_experts=2, routed_scaling_factor=routed_scale, tie_word_embeddings=False
    )
    calls = {"qkvb": [], "fg_a": [], "qkv": [], "w13": [], "w2": []}
    kda = nn.Module()
    kda.mixer = nn.Module()
    kda.mixer.qkvb_proj = nn.Module()
    kda.mixer.qkvb_proj.weight = _recording_parameter(calls["qkvb"], 1)
    kda.mixer.fg_a_proj = nn.Module()
    kda.mixer.fg_a_proj.weight = _recording_parameter(calls["fg_a"], 1)
    attn = nn.Module()
    attn.mixer = nn.Module()
    attn.mixer.qkv_proj = nn.Module()
    attn.mixer.qkv_proj.weight = _recording_parameter(calls["qkv"], 1)
    attn.moe = nn.Module()
    attn.moe.latent_up = nn.Linear(2, 2, bias=False)
    attn.moe.experts = nn.Module()
    attn.moe.experts.routed_experts = nn.Module()
    routed = attn.moe.experts.routed_experts
    routed.w13_weight = _recording_parameter(calls["w13"], 1)
    routed.w2_weight = _recording_parameter(calls["w2"], 1)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([kda, attn])
    return model, calls


@pytest.mark.parametrize("routed_scale", [8.0, 1.0])
def test_load_weights_routes_shared_names_by_layer_type(routed_scale):
    """`mixer.q_proj` lands in `qkvb_proj` on a KDA layer and in `qkv_proj` on
    an attention layer; `latent_up` is pre-multiplied by the routed scale; and
    expert weights go to the fused-MoE parameters with their expert id."""
    model, calls = _tiny_model_for_loading(routed_scale)
    w = {i: torch.full((1,), float(i)) for i in range(8)}
    latent_up = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    loaded = model.load_weights(
        [
            ("model.layers.0.mixer.q_proj.weight", w[0]),
            ("model.layers.0.mixer.b_proj.weight", w[1]),
            ("model.layers.0.mixer.g_a_proj.weight", w[2]),
            ("model.layers.1.mixer.q_proj.weight", w[3]),
            ("model.layers.1.mixer.v_proj.weight", w[4]),
            ("model.layers.1.moe.latent_up.weight", latent_up),
            ("model.layers.1.moe.experts.1.gate_proj.weight", w[5]),
            ("model.layers.1.moe.experts.0.down_proj.weight", w[6]),
        ]
    )

    def summary(name):
        return [(v.item(), args, kwargs) for _, v, args, kwargs in calls[name]]

    assert summary("qkvb") == [(0.0, (0,), {}), (1.0, (3,), {})]
    assert summary("fg_a") == [(2.0, (1,), {})]
    assert summary("qkv") == [(3.0, ("q",), {}), (4.0, ("v",), {})]
    w13 = "model.layers.1.moe.experts.routed_experts.w13_weight"
    w2 = "model.layers.1.moe.experts.routed_experts.w2_weight"
    assert summary("w13") == [(5.0, (w13,), {"expert_id": 1, "shard_id": "w1"})]
    assert summary("w2") == [(6.0, (w2,), {"expert_id": 0, "shard_id": "w2"})]
    torch.testing.assert_close(
        model.model.layers[1].moe.latent_up.weight.detach(), latent_up * routed_scale
    )
    assert {w13, w2, "model.layers.1.moe.latent_up.weight"} <= loaded
    assert "model.layers.0.mixer.qkvb_proj.weight" in loaded
    assert "model.layers.1.mixer.qkv_proj.weight" in loaded


def test_load_weights_rejects_unknown_expert_weights():
    model, _ = _tiny_model_for_loading(8.0)
    with pytest.raises(ValueError, match="no module or parameter named"):
        model.load_weights(
            [("model.layers.1.moe.experts.0.extra_proj.weight", torch.ones(1))]
        )


def test_conv_weights_concatenate_into_the_kernel_layout():
    """q, k and v convs (v twice as wide) concatenate channel-wise into the
    fused [kernel_size, q | k | v] buffer the KDA op reads."""
    layer = model_mod.Olmo35KimiDeltaAttention.__new__(
        model_mod.Olmo35KimiDeltaAttention
    )
    nn.Module.__init__(layer)
    kernel, key_width, value_width = 4, 3, 6
    weights = {}
    for name, width, offset in (
        ("q_conv1d", key_width, 0),
        ("k_conv1d", key_width, 100),
        ("v_conv1d", value_width, 200),
    ):
        weight = offset + torch.arange(width * kernel, dtype=torch.float32)
        weights[name] = weight.reshape(width, 1, kernel)
        setattr(layer, name, SimpleNamespace(weight=weights[name]))

    layer.process_weights_after_loading(torch.bfloat16)

    fused = layer.conv_weight_fused
    assert fused.shape == (kernel, 2 * key_width + value_width)
    assert fused.is_contiguous()
    expected = torch.cat(
        [weights[n][:, 0, :].t() for n in ("q_conv1d", "k_conv1d", "v_conv1d")],
        dim=1,
    )
    torch.testing.assert_close(fused, expected)
    # Column j of the v block is v channel j.
    torch.testing.assert_close(
        fused[:, 2 * key_width + 5], weights["v_conv1d"][5, 0, :]
    )


@pytest.mark.parametrize("scalable_softmax", [True, False])
def test_attention_folds_the_scalable_softmax_factor_into_q(scalable_softmax):
    """q reaches the attention op as rms_norm(q) * gain_norm * head_dim^-0.5,
    times log(position + 1) * ssmax_scale[head] with scalable softmax on; the
    op itself runs with scale 1."""
    heads, kv_heads, head_dim, eps = 2, 1, 4, 1e-6
    layer = model_mod.Olmo35Attention.__new__(model_mod.Olmo35Attention)
    nn.Module.__init__(layer)
    layer.num_heads, layer.num_kv_heads, layer.head_dim = heads, kv_heads, head_dim
    layer.q_size, layer.kv_size = heads * head_dim, kv_heads * head_dim
    layer.eps, layer.scaling = eps, head_dim**-0.5
    layer.use_scalable_softmax = scalable_softmax
    layer.q_norm = nn.Parameter(torch.tensor([[1.0, 2.0, 3.0, 4.0]] * heads))
    layer.k_norm = nn.Parameter(torch.ones(kv_heads, head_dim))
    layer.ssmax_scale = nn.Parameter(torch.tensor([0.5, 3.0]))
    captured = {}

    def fake_attn(q, k, v):
        captured["q"] = q
        return torch.zeros_like(q)

    layer.qkv_proj = lambda x: (x, None)
    layer.g_proj = lambda x: (torch.zeros(x.shape[0], layer.q_size), None)
    layer.attn = fake_attn
    layer.o_proj = lambda x: (x, None)

    torch.manual_seed(0)
    positions = torch.tensor([0, 5, 1000])
    hidden = torch.randn(3, layer.q_size + 2 * layer.kv_size)
    layer.forward(positions, hidden)

    q = hidden[:, : layer.q_size].reshape(3, heads, head_dim)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps) * layer.q_norm
    q = q * head_dim**-0.5
    if scalable_softmax:
        factor = torch.log(positions.float() + 1.0)[:, None] * layer.ssmax_scale
        q = q * factor[..., None]
    torch.testing.assert_close(captured["q"], q.reshape(3, -1).detach())
