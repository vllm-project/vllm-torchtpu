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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP

from tests.models.vllm.test_qwen4_exp_parallel import config
from vllm_torchtpu.gdn_pool_layout import POOLED_GDN_ARCHITECTURES
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.qwen4_exp import model as qwen4_model
from vllm_torchtpu.models.vllm.qwen4_exp.config import (
    TPUQwen4ExpConfig,
    validate_config,
)
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import HyperConnectionConfig

HIDDEN = 16
HC = 4
VOCAB = 32


@pytest.fixture(autouse=True)
def cpu_tensors():
    with torch.device("cpu"):
        yield


@pytest.mark.parametrize(
    "arch", ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"]
)
def test_qwen4_exp_is_registered(arch, monkeypatch):
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    monkeypatch.setitem(MODELS_CONFIG_MAP, arch, MODELS_CONFIG_MAP.get(arch))
    register_models()
    assert (
        ModelRegistry.models[arch].load_model_cls() is qwen4_model.Qwen4ExpForCausalLM
    )
    assert MODELS_CONFIG_MAP[arch] is TPUQwen4ExpConfig
    assert arch in POOLED_GDN_ARCHITECTURES


def test_native_qsa_and_ple_lift_initial_dense_model_limits():
    cfg = config()
    assert cfg.model_config.hf_text_config.ple_layer_ids
    assert (
        cfg.model_config.max_model_len > cfg.model_config.hf_text_config.indexer_budget
    )
    validate_config(cfg)


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        (None, "speculative_config", SimpleNamespace(method="mtp"), "MTP"),
        ("parallel_config", "pipeline_parallel_size", 2, "PP/CP1"),
        ("parallel_config", "prefill_context_parallel_size", 2, "PP/CP1"),
        ("parallel_config", "decode_context_parallel_size", 2, "PP/CP1"),
    ],
)
def test_config_keeps_unsupported_execution_modes_closed(
    section, field, value, message
):
    cfg = config()
    setattr(cfg if section is None else getattr(cfg, section), field, value)
    with pytest.raises(ValueError, match=message):
        validate_config(cfg)


def test_unknown_layer_type_is_rejected():
    cfg = config()
    cfg.model_config.hf_text_config.layer_types[0] = "sliding_attention"
    with pytest.raises(ValueError, match="layer geometry"):
        validate_config(cfg)


class _FakeSublayer(nn.Module):
    """Deterministic stand-in; the backbone and decoder forward methods are real."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False, device="cpu")
        self.calls = []

    def forward(self, hidden, *args):
        self.calls.append(args)
        return torch.tanh(self.proj(hidden))


def _hc(*, use_combine=True):
    return qwen4_model.GatedResidual(
        HyperConnectionConfig(
            hc_count=HC,
            hidden_size=HIDDEN,
            hc_lowrank=8,
            params_dtype=torch.float32,
            hc_per_branch_norm=True,
        ),
        use_combine=use_combine,
    )


@pytest.fixture
def backbone():
    # Exercise composition without distributed construction, weights or TPU ops.
    with torch.device("cpu"):
        result = qwen4_model.Qwen4ExpBackbone.__new__(qwen4_model.Qwen4ExpBackbone)
        nn.Module.__init__(result)
        result.hc_count = HC
        result.host_ple_index = None
        result.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        result.layers = nn.ModuleList()
        for kind in ("linear_attention", "full_attention"):
            layer = qwen4_model.Qwen4ExpDecoderLayer.__new__(
                qwen4_model.Qwen4ExpDecoderLayer
            )
            nn.Module.__init__(layer)
            layer.ple = None
            layer.layer_type = kind
            setattr(
                layer,
                "linear_attn" if kind == "linear_attention" else "self_attn",
                _FakeSublayer(),
            )
            layer.mlp = _FakeSublayer()
            layer.attn_hyper_connection = _hc()
            layer.mlp_hyper_connection = _hc()
            result.layers.append(layer)
        result.hyper_connection_mixer = _hc(use_combine=False)
        generator = torch.Generator(device="cpu").manual_seed(0)
        with torch.no_grad():
            for param in result.parameters():
                param.copy_(torch.randn(param.shape, generator=generator) * 0.1)
        yield result


def _metadata(tokens):
    return dict(
        query_start_loc=torch.tensor([0, tokens]),
        ngram_context=torch.zeros((1, 2), dtype=torch.int64),
        state_slots=torch.tensor([1]),
        context_lens=torch.tensor([0]),
        valid_tokens=torch.ones(tokens, dtype=torch.bool),
        main_block_table=torch.tensor([[1]]),
        compressed_block_table=torch.tensor([[2]]),
    )


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


def test_model_matches_hyper_connection_reference(backbone):
    input_ids = torch.tensor([1, 5, 7, 30, 2])
    positions = torch.arange(input_ids.numel())
    with torch.no_grad():
        output = backbone.forward(input_ids, positions, **_metadata(input_ids.numel()))
        hyper = backbone.embed_tokens(input_ids).repeat(1, HC)
        for layer in backbone.layers:
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
                hyper = _ref_combine(hc, hyper, normed, torch.tanh(block.proj(mixed)))
        expected, _ = _ref_mix(backbone.hyper_connection_mixer, hyper)
    assert output.shape == (input_ids.numel(), HIDDEN)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_layer_types_route_positions_and_native_qsa_metadata(backbone):
    positions = torch.arange(3)
    metadata = _metadata(3)
    with torch.no_grad():
        backbone.forward(torch.tensor([1, 2, 3]), positions, **metadata)
    gdn, qsa = backbone.layers
    assert gdn.linear_attn.calls == [()]
    actual_positions, actual_metadata = qsa.self_attn.calls[0]
    assert actual_positions is positions
    assert actual_metadata.query_start_loc is metadata["query_start_loc"]
    assert not hasattr(backbone.hyper_connection_mixer, "block_inject_weight")


def test_ordinary_loader_maps_hc_names_and_skips_mtp():
    # Complete ordinary parameters; expert/QSA completeness has separate tests.
    model = qwen4_model.Qwen4ExpForCausalLM.__new__(qwen4_model.Qwen4ExpForCausalLM)
    nn.Module.__init__(model)
    model._pd_weight_identity_enabled = False
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    model.model.hyper_connection_mixer = _hc(use_combine=False)
    model.model.embed_tokens = nn.Embedding(VOCAB, HIDDEN, device="cpu")
    model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False, device="cpu")
    weights = {
        name.replace("model.", "model.language_model.", 1): torch.full_like(param, 0.25)
        for name, param in model.named_parameters()
    }
    weights["mtp.fc_hidden.weight"] = torch.zeros((HIDDEN, HIDDEN))
    loaded = model.load_weights(iter(weights.items()))
    assert loaded == set(dict(model.named_parameters()))
    for name, param in model.named_parameters():
        torch.testing.assert_close(
            param, weights[name.replace("model.", "model.language_model.", 1)]
        )
