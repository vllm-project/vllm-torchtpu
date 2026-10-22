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
"""Model-owned immutable buffers must not alias mutable attention state."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config


@pytest.fixture
def model_cpu_config_context():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    resolve_kv_cache_layout(config, [["LBNHC", "LBHNC"]])
    with set_current_vllm_config(config):
        yield


@pytest.mark.usefixtures("model_cpu_config_context")
@pytest.mark.parametrize(
    "layer_types",
    [
        ["linear_attention", "indexed_attention", "indexed_attention"],
        ["linear_attention"],
    ],
)
def test_rope_sharing_is_model_local_and_excludes_kv_state(monkeypatch, layer_types):
    from vllm_torchtpu.models.vllm.qwen4_exp import model
    from vllm_torchtpu.models.vllm.qwen4_exp.qsa import Qwen4ExpQSA

    text = q4exp_text_config(
        hidden_size=16,
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        indexer_n_heads=1,
        indexer_head_dim=16,
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
    )

    # Keep real QSA buffers; replace unrelated GDN/MoE/PLE construction so this
    # ownership test requires neither a distributed group nor model weights.
    class AttentionOnlyLayer(nn.Module):
        def __init__(self, cfg, index, prefix):
            super().__init__()
            self.ple = None
            self.layer_type = text.layer_types[index]
            if self.layer_type == "indexed_attention":
                self.self_attn = Qwen4ExpQSA(
                    config=text,
                    tp_rank=0,
                    tp_size=1,
                    max_model_len=cfg.model_config.max_model_len,
                    device="cpu",
                    prefix=prefix + ".self_attn",
                )
                self.self_attn.cache_owners = tuple(torch.zeros(1) for _ in range(4))

    monkeypatch.setattr(model, "Qwen4ExpDecoderLayer", AttentionOnlyLayer)
    monkeypatch.setattr(model, "VocabParallelEmbedding", nn.Embedding)
    monkeypatch.setattr(model, "GatedResidual", lambda *a, **kw: nn.Identity())
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1
    )
    with torch.device("cpu"):
        first, second = [
            model.Qwen4ExpBackbone(
                NS(
                    model_config=NS(hf_text_config=text, max_model_len=length),
                    additional_config={},
                )
            )
            for length in (128, 256)
        ]
    attentions = [
        layer.self_attn
        for layer in first.layers
        if layer.layer_type == "indexed_attention"
    ]
    if not attentions:
        assert all(layer.layer_type == "linear_attention" for layer in first.layers)
        return
    assert attentions[0].rope_cache.shape[0] == 128
    assert all(a.rope_cache is attentions[0].rope_cache for a in attentions)
    assert len({id(state) for a in attentions for state in a.cache_owners}) == (
        4 * len(attentions)
    )
    other = next(
        layer.self_attn
        for layer in second.layers
        if layer.layer_type == "indexed_attention"
    )
    assert other.rope_cache.shape[0] == 256
    assert other.rope_cache is not attentions[0].rope_cache
    assert not any(name.endswith("rope_cache") for name in first.state_dict())


def test_qwen_decoder_opts_into_compact_gdn_inputs(monkeypatch):
    from unittest.mock import Mock

    from vllm_torchtpu.models.vllm.qwen4_exp import model

    text = q4exp_text_config(ple_layer_ids=[], layer_types=["linear_attention"])
    cfg = NS(model_config=NS(hf_text_config=text), additional_config={})
    attention = Mock(return_value=nn.Identity())
    monkeypatch.setattr(
        model, "get_tp_group", lambda: NS(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(model, "Qwen4ExpGatedDeltaNetAttention", attention)
    monkeypatch.setattr(model, "Qwen3NextSparseMoeBlock", lambda **kw: nn.Identity())
    monkeypatch.setattr(model, "GatedResidual", lambda *a, **kw: nn.Identity())

    layer = model.Qwen4ExpDecoderLayer(cfg, 0, "model.layers.0")

    assert layer.linear_attn is attention.return_value
    assert "use_compact_qkv" not in attention.call_args.kwargs
    assert "pooled_conv_state_dtype" not in attention.call_args.kwargs


@pytest.mark.usefixtures("model_cpu_config_context")
@pytest.mark.parametrize("full_tile", [0, 128])
def test_model_constructs_real_hc_with_only_full_mix_option(monkeypatch, full_tile):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter
    from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware
    from vllm_torchtpu.models.vllm.qwen4_exp import model
    from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import GatedResidual

    # CPU construction fixture: qualified layout facts, not actual TPU access.
    monkeypatch.setattr(
        vllm_adapter,
        "current_hc_hardware",
        lambda: HcHardware("7x", 64 << 20, 128, 16, 8, 256),
    )
    text = q4exp_text_config(
        hidden_size=128,
        hc_lowrank=32,
        vocab_size=32,
        num_hidden_layers=1,
        ple_layer_ids=[],
        layer_types=["linear_attention"],
    )
    cfg = NS(
        model_config=NS(hf_text_config=text),
        additional_config={"qwen4_exp": {"hc_prefill_full_tile": full_tile}},
    )
    # Construct the actual decoder and final mixer. Only unrelated distributed
    # attention/MoE and vocabulary embedding are replaced with CPU fixtures.
    monkeypatch.setattr(
        model, "get_tp_group", lambda: NS(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        model, "Qwen4ExpGatedDeltaNetAttention", lambda *a, **kw: nn.Identity()
    )
    monkeypatch.setattr(model, "Qwen3NextSparseMoeBlock", lambda **kw: nn.Identity())
    monkeypatch.setattr(model, "VocabParallelEmbedding", nn.Embedding)
    with torch.device("cpu"):
        backbone = model.Qwen4ExpBackbone(cfg)
    decoder = backbone.layers[0]
    connections = [
        decoder.attn_hyper_connection,
        decoder.mlp_hyper_connection,
        backbone.hyper_connection_mixer,
    ]
    for connection in connections:
        assert isinstance(connection, GatedResidual)
        assert connection.prefill_full_tile == full_tile
        assert (connection._prefill_full_op is not None) == bool(full_tile)
    for connection in connections[:2]:
        assert (connection._prefill_combine_op is not None) == bool(full_tile)
    assert backbone.hyper_connection_mixer._prefill_combine_op is None


@pytest.fixture
def hc_model_boundaries(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter
    from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware
    from vllm_torchtpu.models.vllm.qwen4_exp import model

    monkeypatch.setattr(
        model, "get_tp_group", lambda: NS(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(
        model, "Qwen4ExpGatedDeltaNetAttention", lambda *a, **kw: nn.Identity()
    )
    monkeypatch.setattr(model, "Qwen3NextSparseMoeBlock", lambda **kw: nn.Identity())
    monkeypatch.setattr(model, "VocabParallelEmbedding", nn.Embedding)
    monkeypatch.setattr(
        vllm_adapter,
        "current_hc_hardware",
        lambda: HcHardware("7x", 64 << 20, 128, 16, 8, 256),
    )

    def forbidden(*args):
        pytest.fail("metadata-only model fixture executed a native op")

    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", lambda *a: forbidden)
    monkeypatch.setattr(vllm_adapter, "build_gated_residual_op", lambda *a: forbidden)
    return model


@pytest.mark.usefixtures("model_cpu_config_context")
@pytest.mark.parametrize(
    "options,full,combine",
    [
        ({}, "auto", "auto"),
        ({"hc_prefill_full_tile": 0}, 0, 0),
        ({"hc_prefill_full_tile": 128}, 128, 128),
        ({"hc_prefill_full_tile": 0, "hc_prefill_combine_tile": "auto"}, 0, "auto"),
        ({"hc_prefill_full_tile": "auto", "hc_prefill_combine_tile": 0}, "auto", 0),
        ({"hc_prefill_full_tile": 128, "hc_prefill_combine_tile": 32}, 128, 32),
    ],
)
def test_model_hc_independent_modes_reach_all_constructors(
    options, full, combine, hc_model_boundaries
):
    from tests.models.vllm.test_qwen4_exp_hc_auto import (
        combine_selected,
        mix_selected,
        route_inputs,
    )

    model = hc_model_boundaries
    text = q4exp_text_config(
        hidden_size=2560,
        hc_lowrank=320,
        hc_per_branch_norm=True,
        vocab_size=32,
        num_hidden_layers=1,
        ple_layer_ids=[],
        layer_types=["linear_attention"],
    )
    cfg = NS(
        model_config=NS(hf_text_config=text), additional_config={"qwen4_exp": options}
    )
    with torch.device("meta"):
        backbone = model.Qwen4ExpBackbone(cfg)
    x, block = route_inputs(1024)
    decoder = backbone.layers[0]
    connections = [
        decoder.attn_hyper_connection,
        decoder.mlp_hyper_connection,
        backbone.hyper_connection_mixer,
    ]
    for connection in connections:
        assert connection.prefill_full_tile == full
        assert connection.prefill_combine_tile == combine
        assert mix_selected(connection, x) is bool(full)
    for connection in connections[:2]:
        assert combine_selected(connection, block, x, x) is bool(combine)
    assert connections[2]._prefill_combine_op is None


@pytest.mark.usefixtures("model_cpu_config_context")
@pytest.mark.parametrize(
    "key,value",
    [
        ("hc_prefill_full_tile", "128"),
        ("hc_prefill_full_tile", True),
        ("hc_prefill_full_tile", -128),
        ("hc_prefill_combine_tile", False),
        ("hc_prefill_combine_tile", "invalid"),
    ],
)
def test_model_hc_rejects_malformed_modes(key, value, hc_model_boundaries):
    text = q4exp_text_config(
        hidden_size=128,
        vocab_size=32,
        num_hidden_layers=1,
        ple_layer_ids=[],
        layer_types=["linear_attention"],
    )
    cfg = NS(
        model_config=NS(hf_text_config=text),
        additional_config={"qwen4_exp": {key: value}},
    )
    with torch.device("meta"), pytest.raises(ValueError, match="HC prefill mode"):
        hc_model_boundaries.Qwen4ExpBackbone(cfg)
