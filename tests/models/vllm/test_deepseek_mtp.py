# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models import deepseek_mtp as upstream
from vllm.model_executor.models.interfaces import supports_pp

from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.deepseek_mtp import DeepSeekMTP


def test_dsa_mtp_draft_declares_pipeline_support():
    # Upstream's draft fails vLLM's pipeline check; drop the subclass once it
    # passes.
    assert not supports_pp(upstream.DeepSeekMTP)
    assert supports_pp(DeepSeekMTP)


def test_dsa_mtp_draft_is_registered_over_upstream():
    register_models()
    assert ModelRegistry.models["DeepseekV32MTPModel"].load_model_cls() is DeepSeekMTP


def test_dsa_mtp_draft_never_makes_intermediate_tensors():
    draft = DeepSeekMTP.__new__(DeepSeekMTP)
    with pytest.raises(NotImplementedError, match="last pipeline stage"):
        draft.make_empty_intermediate_tensors(1, torch.bfloat16, torch.device("cpu"))


def test_dsa_mtp_draft_offers_the_top_level_embedding_as_its_first_layers(
    monkeypatch,
):
    # GLM-5.2's MTP layer has no embedding of its own, and upstream loads only
    # names inside an MTP layer.
    seen = []

    def upstream_load(self, weights):
        names = [name for name, _ in weights]
        seen.extend(names)
        return set(names)

    monkeypatch.setattr(upstream.DeepSeekMTP, "load_weights", upstream_load)
    draft = DeepSeekMTP.__new__(DeepSeekMTP)
    object.__setattr__(draft, "model", SimpleNamespace(mtp_start_layer_idx=78))
    weight = torch.zeros(1)
    draft.load_weights(
        [
            ("model.embed_tokens.weight", weight),
            ("model.layers.78.eh_proj.weight", weight),
            ("model.layers.3.self_attn.kv_a_layernorm.weight", weight),
            ("lm_head.weight", weight),
        ]
    )
    assert seen == [
        "model.layers.78.embed_tokens.weight",
        "model.layers.78.eh_proj.weight",
        "model.layers.3.self_attn.kv_a_layernorm.weight",
        "lm_head.weight",
    ]


def test_upstream_loads_the_first_layers_embedding_into_the_draft_table():
    # The contract the rename relies on: upstream counts the name as the first
    # MTP layer's and rewrites it to the draft's model.embed_tokens.
    name = "model.layers.78.embed_tokens.weight"
    config = SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=1)
    assert upstream.get_spec_layer_idx_from_weight_name(config, name) == 78
    assert (
        upstream.get_spec_layer_idx_from_weight_name(
            config, "model.embed_tokens.weight"
        )
        is None
    )
    draft = DeepSeekMTP.__new__(DeepSeekMTP)
    assert draft._rewrite_spec_layer_name(78, name) == "model.embed_tokens.weight"
