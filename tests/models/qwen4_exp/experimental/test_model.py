# SPDX-License-Identifier: Apache-2.0
"""Model-owned immutable buffers must not alias mutable attention state."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from tests.models.qwen4_exp.experimental.qwen38_test_utils import q4exp_text_config


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
        ["linear_attention", "full_attention", "full_attention"],
        ["linear_attention"],
    ],
)
def test_rope_sharing_is_model_local_and_excludes_kv_state(monkeypatch, layer_types):
    from vllm_torchtpu.models.qwen4_exp.experimental import model
    from vllm_torchtpu.models.qwen4_exp.experimental.attention import Qwen4ExpQSA

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
            if self.layer_type == "full_attention":
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
        if layer.layer_type == "full_attention"
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
        if layer.layer_type == "full_attention"
    )
    assert other.rope_cache.shape[0] == 256
    assert other.rope_cache is not attentions[0].rope_cache
    assert not any(name.endswith("rope_cache") for name in first.state_dict())


def test_qwen_decoder_opts_into_compact_gdn_inputs(monkeypatch):
    from unittest.mock import Mock

    from vllm_torchtpu.models.qwen4_exp.experimental import model

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
    assert attention.call_args.kwargs["pooled_conv_state_dtype"] == torch.float32


@pytest.mark.parametrize(
    "name, local_ids, skipped",
    [
        # Another EP rank's routed-expert block scale: skipped before read.
        ("model.layers.3.mlp.experts.17.down_proj.weight_scale_inv", {0, 1}, True),
        ("mtp.layers.0.mlp.experts.5.gate_proj.weight_scale_inv", {0, 1}, True),
        # This rank's expert, or no EP: loaded.
        ("model.layers.3.mlp.experts.1.down_proj.weight_scale_inv", {0, 1}, False),
        ("model.layers.3.mlp.experts.17.down_proj.weight_scale_inv", None, False),
        # Not a routed expert's block scale: left to the other filters.
        ("model.layers.3.mlp.shared_expert.down_proj.weight_scale_inv", {0}, False),
        ("model.layers.3.mlp.experts.17.down_proj.weight", {0, 1}, False),
        ("model.layers.3.self_attn.o_proj.weight_scale_inv", {0, 1}, False),
    ],
)
def test_remote_expert_block_scales_are_unused_checkpoint_weights(
    name, local_ids, skipped
):
    """The same tensors the shared EP weight filter skipped for the
    ``.weight_scale_inv`` suffix, now named by the model."""
    from vllm_torchtpu.models.qwen4_exp.experimental.model import (
        Qwen4ExpForCausalLM,
        is_remote_expert_block_scale,
    )

    assert is_remote_expert_block_scale(name, local_ids) is skipped
    model = NS(_unused_check_local_ids=local_ids)
    assert Qwen4ExpForCausalLM.is_unused_checkpoint_weight(model, name) is skipped
