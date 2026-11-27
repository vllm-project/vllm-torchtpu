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
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.utils import set_weight_attrs

import vllm_torchtpu.gdn_pool_layout as gdn_pool_layout
import vllm_torchtpu.layers.adapter.kda_attention as kda_attention
import vllm_torchtpu.models.olmo35.config as config_mod
import vllm_torchtpu.models.olmo35.model as model_mod
from vllm_torchtpu.models.olmo35.config import Olmo35Config
from vllm_torchtpu.models.vllm import register_models


def test_olmo35_is_registered():
    """The arch resolves from the vLLM registry and `model_type: olmo35`
    resolves from AutoConfig, since transformers has no class for it."""
    register_models()
    entry = ModelRegistry.models.get("Olmo35ForCausalLM")
    assert entry is not None, "Olmo35ForCausalLM is not registered"
    assert entry.load_model_cls() is model_mod.Olmo35ForCausalLM
    assert isinstance(AutoConfig.for_model("olmo35"), Olmo35Config)
    # Without this entry the default flags fail in the platform check again.
    assert MODELS_CONFIG_MAP["Olmo35ForCausalLM"] is config_mod.TPUOlmo35Config
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


def test_unknown_layer_types_are_rejected():
    """A misspelled type would otherwise silently build a KDA layer."""
    with pytest.raises(ValueError, match="Unknown layer_types"):
        Olmo35Config(
            num_hidden_layers=2, layer_types=["linear_attention", "full_atention"]
        )


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


def test_register_skips_a_clash_and_the_model_refuses_other_configs(monkeypatch):
    """A repeat registration is a no-op for the same class. A clash (e.g. a
    transformers release with its own `olmo35`) logs and skips, so other
    models still start, and the OLMo model refuses the foreign config."""
    register_models()
    assert config_mod.register_olmo35_config()  # Same class: no-op.

    class OtherConfig(Olmo35Config):
        pass

    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "olmo35", OtherConfig)
    assert not config_mod.register_olmo35_config()
    foreign = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=object()))
    with pytest.raises(TypeError, match="Olmo35Config"):
        model_mod._config(foreign)


def _tpu_config_vllm_config(**overrides):
    values = dict(
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(architecture="Olmo35ForCausalLM"),
        speculative_config=None,
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_tpu_config_turns_prefix_caching_off(monkeypatch):
    """vLLM enables prefix caching by default, which would make the platform
    demand the unified pool; the hook turns it off instead of failing."""
    monkeypatch.setattr(
        gdn_pool_layout, "unified_kv_layout_enabled_for_architecture", lambda a: False
    )
    vllm_config = _tpu_config_vllm_config(
        cache_config=SimpleNamespace(enable_prefix_caching=True)
    )
    config_mod.TPUOlmo35Config.verify_and_update_config(vllm_config)
    assert vllm_config.cache_config.enable_prefix_caching is False


@pytest.mark.parametrize(
    ("overrides", "unified", "match"),
    [
        ({}, True, "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0"),
        ({"speculative_config": SimpleNamespace()}, False, "--speculative-config"),
        (
            {"kv_transfer_config": SimpleNamespace(kv_connector="TpuRaidenConnector")},
            False,
            "--kv-transfer-config",
        ),
        (
            {
                "parallel_config": SimpleNamespace(
                    prefill_context_parallel_size=1, decode_context_parallel_size=2
                )
            },
            False,
            "context parallel",
        ),
    ],
    ids=["unified_pool", "speculative", "kv_connector", "context_parallel"],
)
def test_tpu_config_rejects_unsupported_features(
    monkeypatch, overrides, unified, match
):
    monkeypatch.setattr(
        gdn_pool_layout, "unified_kv_layout_enabled_for_architecture", lambda a: unified
    )
    with pytest.raises(NotImplementedError, match=match):
        config_mod.TPUOlmo35Config.verify_and_update_config(
            _tpu_config_vllm_config(**overrides)
        )


def test_tpu_config_allows_the_decode_bench_connector(monkeypatch):
    monkeypatch.setattr(
        gdn_pool_layout, "unified_kv_layout_enabled_for_architecture", lambda a: False
    )
    config_mod.TPUOlmo35Config.verify_and_update_config(
        _tpu_config_vllm_config(
            kv_transfer_config=SimpleNamespace(kv_connector="DecodeBenchConnector")
        )
    )


def test_tp_must_divide_the_kda_heads():
    with pytest.raises(ValueError, match=r"TP \(3\) must divide"):
        model_mod._local_kda_heads(Olmo35Config(), 3)


class _FakeLinear(nn.Module):
    """Stands in for vLLM's TP linear layers, which need a distributed group."""

    def __init__(self, input_size, output_size, *args, bias=False, **kwargs):
        super().__init__()
        del args, kwargs
        out = sum(output_size) if isinstance(output_size, list) else output_size
        self.output_sizes = output_size
        self.weight = nn.Parameter(torch.empty(out, input_size))
        self.bias = nn.Parameter(torch.empty(out)) if bias else None


def _small_kda_config():
    """Two heads, key width 4, value width 8 (d_v = 2 d_k, as in OLMo 3.5)."""
    return Olmo35Config(
        hidden_size=32,
        num_hidden_layers=2,
        full_attention_interval=2,
        linear_num_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )


def _build_kda(monkeypatch):
    built = {}

    def build_op(prefix, **kwargs):
        built.update(kwargs, prefix=prefix)
        return lambda *args: None

    def build_pooled_op(*args, **kwargs):
        raise AssertionError("the pooled KDA op does not support OLMo 3.5")

    for name in (
        "ColumnParallelLinear",
        "MergedColumnParallelLinear",
        "RowParallelLinear",
    ):
        monkeypatch.setattr(kda_attention, name, _FakeLinear)
    monkeypatch.setattr(
        kda_attention, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(kda_attention, "build_kimi_dispatched_kda_op", build_op)
    monkeypatch.setattr(kda_attention, "build_kimi_pooled_kda_op", build_pooled_op)
    # TPUOlmo35Config rejects the unified pool before any layer is built.
    monkeypatch.setattr(
        kda_attention, "unified_kv_layout_enabled_for_architecture", lambda a: False
    )
    config = _small_kda_config()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Olmo35ForCausalLM", hf_text_config=config
        ),
        speculative_config=None,
        quant_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    # Pinned to CPU: the default device detection can fail on CPU CI.
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        layer = model_mod.olmo35_kda(vllm_config, "model.layers.0.mixer")
    return layer, vllm_config, built


def test_kda_construction_wires_wide_value_heads_and_state(monkeypatch):
    """The shared KDA layer built from the OLMo config: projections sized for
    d_v != d_k, a biased output gate, the op built with the config's beta
    scale and no pooled op, the layer registered under its prefix, and the
    per-layer state shape equal to the one the KV cache is sized with."""
    layer, vllm_config, built = _build_kda(monkeypatch)

    assert (layer.num_heads, layer.head_k_dim, layer.head_v_dim) == (2, 4, 8)
    assert layer.fused_qkvb_proj.output_sizes == [8, 8, 16, 2]
    assert layer.fused_fa_ga_proj.output_sizes == [8, 8]
    assert layer.f_b_proj.weight.shape == (8, 8)
    assert layer.g_b_proj.weight.shape == (16, 8) and layer.g_b_proj.bias is not None
    assert built["beta_scale"] == 2.0 and built["prefix"] == "model.layers.0.mixer"
    assert built["lower_bound"] is None and layer.pooled_kda_op is None
    context = vllm_config.compilation_config.static_forward_context
    assert context["model.layers.0.mixer"] is layer
    assert layer.get_state_shape() == ((3, 1, 32), (2, 4, 8))
    assert (
        layer.get_state_shape()
        == model_mod.Olmo35ForCausalLM.get_mamba_state_shape_from_config(vllm_config)
    )


class _Fn(nn.Module):
    """A projection stand-in returning `(fn(x), None)` like vLLM's linears."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x), None


def test_kda_forward_feeds_the_op_wide_value_shapes(monkeypatch):
    """forward splits the projections into the op's inputs at d_v != d_k
    widths, hands it both caches, and reshapes its [T, heads, d_v] output."""
    layer, _, _ = _build_kda(monkeypatch)
    tokens, hidden = 3, 32
    layer.fused_qkvb_proj = _Fn(lambda x: torch.arange(34.0).expand(x.shape[0], 34))
    layer.fused_fa_ga_proj = _Fn(lambda x: torch.ones(x.shape[0], 16))
    layer.f_b_proj = _Fn(lambda x: torch.full((x.shape[0], 8), 2.0))
    layer.g_b_proj = _Fn(lambda x: torch.full((x.shape[0], 16), 3.0))
    layer.o_proj = _Fn(lambda x: x.repeat(1, 2))
    conv_cache, recurrent_cache = torch.zeros(4, 3, 1, 32), torch.zeros(4, 2, 4, 8)
    layer.kv_cache = (conv_cache, recurrent_cache)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        mamba_state_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([5, 5, 5], dtype=torch.int32),
        request_distribution=torch.tensor([3, 3, 3], dtype=torch.int32),
        mamba_request_distribution=None,
        mamba_slot_read_offsets=None,
    )
    layer._metadata = lambda: metadata
    seen = {}

    def kda_op(mixed_qkv, raw_gate, beta, output_gate, conv, rec, *rest):
        seen.update(
            mixed=mixed_qkv,
            gate=raw_gate,
            beta=beta,
            out_gate=output_gate,
            conv=conv,
            rec=rec,
        )
        return torch.ones(mixed_qkv.shape[0], 2, 8)

    layer.dispatched_kda_op = kda_op
    monkeypatch.setattr(
        kda_attention, "get_tp_group", lambda: SimpleNamespace(all_reduce=lambda x: x)
    )
    out = layer(torch.arange(tokens), torch.zeros(tokens, hidden))

    assert seen["mixed"].shape == (tokens, 32)
    torch.testing.assert_close(seen["mixed"][0], torch.arange(32.0))
    torch.testing.assert_close(seen["beta"][0], torch.tensor([32.0, 33.0]))
    assert seen["gate"].shape == (tokens, 8) and seen["out_gate"].shape == (tokens, 16)
    assert seen["conv"] is conv_cache and seen["rec"] is recurrent_cache
    torch.testing.assert_close(out, torch.ones(tokens, hidden))


def test_tied_word_embeddings_are_rejected():
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        Olmo35Config(tie_word_embeddings=True)


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
    kda.mixer.fused_qkvb_proj = nn.Module()
    kda.mixer.fused_qkvb_proj.weight = _recording_parameter(calls["qkvb"], 1)
    kda.mixer.fused_fa_ga_proj = nn.Module()
    kda.mixer.fused_fa_ga_proj.weight = _recording_parameter(calls["fg_a"], 1)
    attn = nn.Module()
    attn.mixer = nn.Module()
    attn.mixer.qkv_proj = nn.Module()
    attn.mixer.qkv_proj.weight = _recording_parameter(calls["qkv"], 1)
    attn.moe = nn.Module()
    attn.moe.routed_expert_down_proj = nn.Linear(2, 2, bias=False)
    attn.moe.routed_expert_up_proj = nn.Linear(2, 2, bias=False)
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
    """`mixer.q_proj` lands in `fused_qkvb_proj` on a KDA layer and in
    `qkv_proj` on an attention layer; `latent_down` / `latent_up` land in the
    shared MoE's `routed_expert_down_proj` / `routed_expert_up_proj`, the
    latter pre-multiplied by the routed scale; and expert weights go to the
    fused-MoE parameters with their expert id."""
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
            ("model.layers.1.moe.latent_down.weight", latent_up),
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
    moe = model.model.layers[1].moe
    torch.testing.assert_close(moe.routed_expert_down_proj.weight.detach(), latent_up)
    torch.testing.assert_close(
        moe.routed_expert_up_proj.weight.detach(), latent_up * routed_scale
    )
    down = "model.layers.1.moe.routed_expert_down_proj.weight"
    up = "model.layers.1.moe.routed_expert_up_proj.weight"
    assert {w13, w2, down, up} <= loaded
    assert "model.layers.0.mixer.fused_qkvb_proj.weight" in loaded
    assert "model.layers.1.mixer.qkv_proj.weight" in loaded


def test_load_weights_rejects_unknown_expert_weights():
    model, _ = _tiny_model_for_loading(8.0)
    with pytest.raises(ValueError, match="no module or parameter named"):
        model.load_weights(
            [("model.layers.1.moe.experts.0.extra_proj.weight", torch.ones(1))]
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
    layer.k_norm = nn.Parameter(torch.tensor([[0.5, 1.0, 1.5, 2.0]] * kv_heads))
    layer.ssmax_scale = nn.Parameter(torch.tensor([0.5, 3.0]))
    captured = {}

    def fake_attn(q, k, v):
        captured["q"], captured["k"] = q, k
        return torch.full_like(q, 2.0)

    gate = torch.linspace(-3.0, 3.0, 3 * layer.q_size).reshape(3, layer.q_size)
    layer.qkv_proj = lambda x: (x, None)
    layer.g_proj = lambda x: (gate, None)
    layer.attn = fake_attn
    layer.o_proj = lambda x: (x, None)

    torch.manual_seed(0)
    positions = torch.tensor([0, 5, 1000])
    hidden = torch.randn(3, layer.q_size + 2 * layer.kv_size)
    out = layer.forward(positions, hidden)

    # The attention output is gated by sigmoid(g_proj(x)).
    torch.testing.assert_close(out, 2.0 * torch.sigmoid(gate))
    k = hidden[:, layer.q_size : layer.q_size + layer.kv_size].reshape(3, kv_heads, -1)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * layer.k_norm
    torch.testing.assert_close(captured["k"], k.reshape(3, -1).detach())
    q = hidden[:, : layer.q_size].reshape(3, heads, head_dim)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps) * layer.q_norm
    q = q * head_dim**-0.5
    if scalable_softmax:
        factor = torch.log(positions.float() + 1.0)[:, None] * layer.ssmax_scale
        q = q * factor[..., None]
    torch.testing.assert_close(captured["q"], q.reshape(3, -1).detach())
