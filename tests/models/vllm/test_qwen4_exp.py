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
"""Tests for the TPU Qwen4Exp (Qwen3.8 Flash Next) text model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import CompilationMode
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig

import vllm_torchtpu.models.vllm.qwen4_exp.model as qwen4_model
from vllm_torchtpu.gdn_pool_layout import QWEN_GDN_ARCHITECTURES
from vllm_torchtpu.layers import register_layers
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.models.vllm.qwen4_exp.config import TPUQwen4ExpConfig
from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpForCausalLM

HIDDEN = 16
HC = 4
VOCAB = 32
LAYER_TYPES = ["linear_attention", "full_attention"]


@pytest.fixture(autouse=True)
def _cpu_model_tests(monkeypatch):
    # Same guard as the Kimi model tests: keep Dynamo from querying a TPU
    # device while these CPU-only tests run.
    tpu = getattr(torch, "tpu", None)
    if tpu is not None:
        monkeypatch.setattr(tpu, "is_available", lambda: False)
        monkeypatch.setattr(tpu, "current_device", lambda: 0)
        monkeypatch.setattr(tpu, "device_count", lambda: 0)
        monkeypatch.setattr(tpu, "manual_seed_all", lambda seed: None)
    if hasattr(torch, "accelerator"):
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
        monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
        monkeypatch.setattr(torch.accelerator, "device_count", lambda: 0)


def _text_config(**overrides) -> Qwen4ExpTextConfig:
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        hc_count=HC,
        hc_lowrank=8,
        num_hidden_layers=len(LAYER_TYPES),
        layer_types=LAYER_TYPES,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_num_key_heads=1,
        linear_num_value_heads=3,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    kwargs.update(overrides)
    return Qwen4ExpTextConfig(**kwargs)


# ---------------------------------------------------------------------------
# Registration and config checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "arch", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_qwen4_exp_is_registered(arch: str) -> None:
    previous_model = ModelRegistry.models.get(arch)
    previous_config = MODELS_CONFIG_MAP.get(arch)
    try:
        register_layers()
        assert ModelRegistry.models[arch].load_model_cls() is Qwen4ExpForCausalLM
        assert MODELS_CONFIG_MAP[arch] is TPUQwen4ExpConfig
        assert arch in QWEN_GDN_ARCHITECTURES
    finally:
        if previous_model is None:
            ModelRegistry.models.pop(arch, None)
        else:
            ModelRegistry.models[arch] = previous_model
        if previous_config is None:
            MODELS_CONFIG_MAP.pop(arch, None)
        else:
            MODELS_CONFIG_MAP[arch] = previous_config


def _vllm_config_for_checks(
    text_config: Qwen4ExpTextConfig,
    *,
    max_model_len: int = 2048,
    speculative_config: object | None = None,
    pipeline_parallel_size: int = 1,
    use_sequence_parallel_moe: bool = False,
    prefill_context_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=text_config,
            hf_text_config=text_config,
            multimodal_config=None,
            max_model_len=max_model_len,
        ),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            ubatch_size=1,
            pipeline_parallel_size=pipeline_parallel_size,
            use_sequence_parallel_moe=use_sequence_parallel_moe,
            prefill_context_parallel_size=prefill_context_parallel_size,
            decode_context_parallel_size=decode_context_parallel_size,
        ),
        speculative_config=speculative_config,
    )


def test_config_accepts_supported_text_model_and_strips_mrope() -> None:
    text_config = _text_config(
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        }
    )
    TPUQwen4ExpConfig.verify_and_update_config(_vllm_config_for_checks(text_config))
    assert "mrope_section" not in text_config.rope_parameters
    assert "mrope_interleaved" not in text_config.rope_parameters


_QSA = dict(
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=8,
    indexer_budget=2048,
    indexer_compress_ratio=4,
)


def test_config_accepts_qsa_within_indexer_budget() -> None:
    vllm_config = _vllm_config_for_checks(_text_config(**_QSA), max_model_len=2048)
    TPUQwen4ExpConfig.verify_and_update_config(vllm_config)


def test_config_rejects_qsa_layer_type() -> None:
    text_config = _text_config(**_QSA)
    # Set after construction: the pinned transformers does not know this
    # layer type, but newer versions accept it.
    text_config.layer_types = ["linear_attention", "qwen_sparse_attention"]
    with pytest.raises(NotImplementedError, match="qwen_sparse_attention"):
        TPUQwen4ExpConfig.verify_and_update_config(_vllm_config_for_checks(text_config))


@pytest.mark.parametrize(
    "text_overrides,config_overrides,match",
    [
        ({"ple_layer_ids": [2]}, {}, "PLE"),
        (_QSA, {"max_model_len": 4096}, "indexer_budget=2048"),
        (
            {},
            {"speculative_config": SimpleNamespace(method="mtp")},
            "speculative decoding",
        ),
        ({}, {"pipeline_parallel_size": 2}, "pipeline_parallel_size=1"),
        ({}, {"use_sequence_parallel_moe": True}, "sequence-parallel MoE"),
        ({}, {"prefill_context_parallel_size": 2}, "context parallelism"),
        ({}, {"decode_context_parallel_size": 2}, "context parallelism"),
    ],
    ids=["ple", "long_context_qsa", "mtp", "pp", "sp_moe", "pcp", "dcp"],
)
def test_config_rejects_unsupported_features(
    text_overrides, config_overrides, match
) -> None:
    vllm_config = _vllm_config_for_checks(
        _text_config(**text_overrides), **config_overrides
    )
    with pytest.raises(NotImplementedError, match=match):
        TPUQwen4ExpConfig.verify_and_update_config(vllm_config)


# ---------------------------------------------------------------------------
# Decoder wiring with stand-in sublayers
# ---------------------------------------------------------------------------
class _FakeSublayer(nn.Module):
    """A deterministic hidden_size -> hidden_size stand-in that records calls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(self, *args, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        hidden_states = (
            kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        )
        return torch.tanh(self.proj(hidden_states))


class _FakeEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, prefix: str = ""):
        super().__init__(num_embeddings, embedding_dim)


class _FakeLMHead(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, prefix: str = ""):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))


@pytest.fixture
def fake_sublayers(monkeypatch):
    for name in (
        "QwenGatedDeltaNetAttention",
        "Qwen3NextAttention",
        "Qwen3NextSparseMoeBlock",
    ):
        monkeypatch.setattr(qwen4_model, name, _FakeSublayer)
    monkeypatch.setattr(qwen4_model, "VocabParallelEmbedding", _FakeEmbedding)
    monkeypatch.setattr(qwen4_model, "ParallelLMHead", _FakeLMHead)
    monkeypatch.setattr(qwen4_model, "LogitsProcessor", lambda vocab_size: None)


def _model_vllm_config(text_config: Qwen4ExpTextConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text_config),
        cache_config=None,
        quant_config=None,
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


def _randomize_hc_norms(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.endswith("hc_norm.weight"):
            with torch.no_grad():
                param.normal_(0.0, 0.1)


def _ref_mix(hc: nn.Module, hyper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference GatedResidual.mix with per-stream norm (hc_per_branch_norm)."""
    streams = hyper.unflatten(-1, (HC, HIDDEN))
    weight = hc.hc_norm.weight.view(HC, HIDDEN)
    eps = hc.hc_norm.variance_epsilon
    normed = streams * torch.rsqrt(streams.square().mean(-1, keepdim=True) + eps)
    normed = (normed * (1.0 + weight)).flatten(-2)
    gate = F.silu(normed @ hc.input_mix_weight_down.weight.T / HC)
    gate = torch.sigmoid(gate @ hc.input_mix_weight_up.weight.T)
    mixed = (gate * normed).unflatten(-1, (HC, HIDDEN)).mean(-2)
    return mixed, normed


def _ref_combine(
    hc: nn.Module, hyper: torch.Tensor, normed: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    inject = 2.0 * torch.sigmoid(normed @ hc.block_inject_weight.weight.T / HC)
    streams = hyper.unflatten(-1, (HC, HIDDEN))
    return (streams + out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)


def test_model_matches_hyper_connection_reference(fake_sublayers) -> None:
    torch.manual_seed(0)
    model = qwen4_model.Qwen4ExpModel(
        vllm_config=_model_vllm_config(_text_config()), prefix="model"
    )
    _randomize_hc_norms(model)
    input_ids = torch.tensor([1, 5, 7, 30, 2])
    positions = torch.arange(input_ids.numel())

    with torch.no_grad():
        output = model(input_ids, positions)

        hyper = model.embed_tokens(input_ids).repeat(1, HC)
        for layer in model.layers:
            sublayer = (
                layer.linear_attn
                if layer.layer_type == "linear_attention"
                else layer.self_attn
            )
            for hc, block in (
                (layer.attn_hyper_connection, sublayer),
                (layer.mlp_hyper_connection, layer.mlp),
            ):
                mixed, normed = _ref_mix(hc, hyper)
                block_out = torch.tanh(block.proj(mixed))
                hyper = _ref_combine(hc, hyper, normed, block_out)
        expected, _ = _ref_mix(model.hyper_connection_mixer, hyper)

    assert output.shape == (input_ids.numel(), HIDDEN)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_layer_types_select_gdn_or_dense_gated_attention(fake_sublayers) -> None:
    text_config = _text_config()
    model = qwen4_model.Qwen4ExpModel(
        vllm_config=_model_vllm_config(text_config), prefix="model"
    )
    gdn_layer, attn_layer = model.layers
    assert not hasattr(gdn_layer, "self_attn")
    assert not hasattr(attn_layer, "linear_attn")
    # Upstream Qwen4Exp, like Qwen3.5, uses the non-interleaved GDN layout.
    assert gdn_layer.linear_attn.kwargs["gqa_interleaved_layout"] is False
    assert gdn_layer.linear_attn.args[0] is text_config
    assert attn_layer.self_attn.kwargs["prefix"] == "model.layers.1.self_attn"
    # The final mixer only mixes; the checkpoint has no injection for it.
    assert not hasattr(model.hyper_connection_mixer, "block_inject_weight")

    positions = torch.arange(3)
    with torch.no_grad():
        model(torch.tensor([1, 2, 3]), positions)
    assert attn_layer.self_attn.calls[0]["positions"] is positions
    assert "positions" not in gdn_layer.linear_attn.calls[0]


def test_unknown_layer_type_is_rejected(fake_sublayers) -> None:
    text_config = _text_config()
    text_config.layer_types = ["linear_attention", "sliding_attention"]
    with pytest.raises(ValueError, match="sliding_attention"):
        qwen4_model.Qwen4ExpModel(
            vllm_config=_model_vllm_config(text_config), prefix="model"
        )


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def test_load_weights_maps_hyper_connection_names_and_skips_mtp_and_indexer(
    fake_sublayers,
) -> None:
    model = Qwen4ExpForCausalLM(vllm_config=_model_vllm_config(_text_config()))
    prefix = "model.language_model"
    skipped = (
        # MTP weights belong to the draft model.
        "mtp.fc_hidden.weight",
        # Dense attention does not use the QSA indexer.
        f"{prefix}.layers.1.self_attn.indexer.index_qk_proj.weight",
    )
    weights = {
        f"{prefix}.layers.0.attn_hyper_connection.hc_norm.weight": torch.randn(
            HC * HIDDEN
        ),
        f"{prefix}.layers.1.mlp_hyper_connection.input_mix_weight_down.weight": (
            torch.randn(8, HC * HIDDEN)
        ),
        f"{prefix}.layers.1.mlp_hyper_connection.block_inject_weight.weight": (
            torch.randn(HC, HC * HIDDEN)
        ),
        f"{prefix}.hyper_connection_mixer.input_mix_weight_up.weight": torch.randn(
            HC * HIDDEN, 8
        ),
        f"{prefix}.embed_tokens.weight": torch.randn(VOCAB, HIDDEN),
        "lm_head.weight": torch.randn(VOCAB, HIDDEN),
        **{name: torch.randn(HIDDEN, HIDDEN) for name in skipped},
    }

    loaded = model.load_weights(iter(weights.items()))

    params = dict(model.named_parameters())
    for name, tensor in weights.items():
        if name in skipped:
            continue
        vllm_name = name.replace(f"{prefix}.", "model.")
        assert vllm_name in loaded
        param = params[vllm_name]
        torch.testing.assert_close(param, tensor.to(param.dtype))
    assert not any("mtp." in name or ".indexer." in name for name in loaded)


_LM = "model.language_model"


@pytest.mark.parametrize(
    "checkpoint_name,expected",
    [
        (f"{_LM}.embed_tokens.weight", ("model.embed_tokens.weight", None)),
        ("lm_head.weight", ("lm_head.weight", None)),
        (
            f"{_LM}.hyper_connection_mixer.hc_norm.weight",
            ("model.hyper_connection_mixer.hc_norm.weight", None),
        ),
        (
            f"{_LM}.layers.0.linear_attn.in_proj_qkv.weight",
            ("model.layers.0.linear_attn.in_proj_qkvz.weight", (0, 1, 2)),
        ),
        (
            f"{_LM}.layers.0.linear_attn.in_proj_z.weight",
            ("model.layers.0.linear_attn.in_proj_qkvz.weight", 3),
        ),
        (
            f"{_LM}.layers.0.linear_attn.in_proj_b.weight",
            ("model.layers.0.linear_attn.in_proj_ba.weight", 0),
        ),
        (
            f"{_LM}.layers.0.linear_attn.in_proj_a.weight",
            ("model.layers.0.linear_attn.in_proj_ba.weight", 1),
        ),
        (
            f"{_LM}.layers.3.self_attn.q_proj.weight",
            ("model.layers.3.self_attn.qkv_proj.weight", "q"),
        ),
        (
            f"{_LM}.layers.0.mlp.shared_expert.up_proj.weight",
            ("model.layers.0.mlp.shared_expert.gate_up_proj.weight", 1),
        ),
        (
            f"{_LM}.layers.0.mlp.experts.7.down_proj.weight_scale_inv",
            ("model.layers.0.mlp.experts.7.down_proj.weight_scale_inv", None),
        ),
        (f"{_LM}.layers.3.self_attn.indexer.index_qk_proj.weight", None),
        (f"{_LM}.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", None),
        (f"{_LM}.layers.1.ple.key_proj.weight", None),
        ("model.visual.blocks.0.attn.qkv.weight", None),
        ("mtp.layers.0.mlp.experts.0.gate_proj.weight", None),
    ],
)
def test_checkpoint_names_map_to_params_or_are_dropped(
    checkpoint_name: str, expected: tuple[str, object] | None
) -> None:
    mapped = list(
        Qwen4ExpForCausalLM.hf_to_vllm_mapper.apply([(checkpoint_name, torch.empty(0))])
    )
    if expected is None:
        assert mapped == []
        return
    ((name, tensor),) = mapped
    assert (name, getattr(tensor, "shard_id", None)) == expected


def test_fp8_checkpoint_quantizes_only_routed_experts() -> None:
    not_converted = [
        "lm_head",
        f"{_LM}.embed_tokens",
        *(
            f"{_LM}.layers.0.linear_attn.{name}"
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
        ),
        f"{_LM}.layers.0.linear_attn.out_proj",
        *(
            f"{_LM}.layers.1.self_attn.{name}"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ),
        *(
            f"{_LM}.layers.{idx}.mlp.shared_expert.{name}"
            for idx in (0, 1)
            for name in ("gate_proj", "up_proj", "down_proj")
        ),
    ]
    quant_config = VllmFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": not_converted,
        }
    )
    configure_quant_config(quant_config, Qwen4ExpForCausalLM)

    def skipped(prefix: str) -> bool:
        return is_layer_skipped(
            prefix,
            quant_config.ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
        )

    for prefix in (
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.0.linear_attn.in_proj_ba",
        "model.layers.0.linear_attn.out_proj",
        "model.layers.1.self_attn.qkv_proj",
        "model.layers.1.self_attn.o_proj",
        "model.layers.0.mlp.shared_expert.gate_up_proj",
        "model.layers.1.mlp.shared_expert.down_proj",
    ):
        assert skipped(prefix), prefix
    assert not skipped("model.layers.0.mlp.experts")
