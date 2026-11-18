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
"""Unit tests for the OLMo 3.5 model: registration, config and per-head
weight sharding.
"""

import pytest
import torch
from transformers import AutoConfig
from vllm.model_executor.models import ModelRegistry

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
    every other layer is KDA."""
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
