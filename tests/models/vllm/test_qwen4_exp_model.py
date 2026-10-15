# SPDX-License-Identifier: Apache-2.0
"""Model-owned immutable buffers must not alias mutable attention state."""

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config


@pytest.mark.usefixtures("vllm_config_context")
@pytest.mark.parametrize(
    "layer_types",
    [
        ["linear_attention", "full_attention", "full_attention"],
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
