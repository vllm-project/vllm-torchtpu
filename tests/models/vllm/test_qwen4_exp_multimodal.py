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
"""Conditional-generation registration, raw token and native pool contracts."""

from types import SimpleNamespace

import torch
from torch import nn
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.interfaces import supports_multimodal

from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpForCausalLM
from vllm_torchtpu.models.vllm.qwen4_exp.multimodal import (
    Qwen4ExpForConditionalGeneration,
)
from vllm_torchtpu.models.vllm.qwen4_exp.pooled_cache import Qwen4ExpPooledCache
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class NativeText(nn.Module):
    requires_raw_input_tokens = True

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(248320, 8)
        self.model.layers = nn.ModuleList([nn.Linear(8, 8, bias=False)])
        self.seen_weights = []

    def embed_input_ids(self, ids):
        return self.model.embed_tokens(ids)

    def load_weights(self, weights):
        self.seen_weights = [name for name, _ in weights]
        return {"model.layers.0.weight"}


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(8))
        self.seen_weights = []

    def load_weights(self, weights):
        self.seen_weights.extend(name for name, _ in weights)
        return {"weight"}


def wrapper():
    result = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    nn.Module.__init__(result)
    result.language_model = NativeText()
    result.visual = Vision()
    result.use_deepstack = False
    result.language_model_only = False
    return result


def test_conditional_registry_truthfully_supports_multimodal(monkeypatch):
    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    register_models()
    cls = ModelRegistry.models["Qwen4ExpForConditionalGeneration"].load_model_cls()
    assert cls is Qwen4ExpForConditionalGeneration
    assert supports_multimodal(cls)
    assert not supports_multimodal(Qwen4ExpForCausalLM)


def test_forward_preserves_native_embeddings_and_intermediate_contract(monkeypatch):
    seen = []

    def forward(self, ids, positions, inputs_embeds=None, **kwargs):
        seen.append((ids, positions, inputs_embeds, kwargs))
        return inputs_embeds

    monkeypatch.setattr(NativeText, "forward", forward)
    model = wrapper()
    ids, positions = torch.arange(3), torch.arange(3).expand(3, -1)
    embeds = torch.randn(3, 8)
    assert model(ids, positions, inputs_embeds=embeds) is embeds
    assert seen[-1] == (ids, positions, embeds, {})
    intermediate = object()
    assert model(ids, positions, intermediate, embeds) is embeds
    assert seen[-1][0] is ids and seen[-1][1] is positions
    assert seen[-1][2] is embeds
    assert seen[-1][3] == {"intermediate_tensors": intermediate}


def test_runner_retains_raw_ids_while_merging_visual_embeddings():
    model = wrapper()
    runner = SimpleNamespace(supports_mm_inputs=True, model=model)
    ids = torch.tensor([3, 248056, 4, 0])
    mask = torch.tensor([False, True, False])
    vision = torch.full((1, 8), 99.0)
    raw, embeds = TPUModelRunner._get_model_inputs(runner, ids, ([vision], mask))
    assert raw is ids
    torch.testing.assert_close(embeds[1], vision[0], rtol=0, atol=0)
    torch.testing.assert_close(embeds[[0, 2, 3]], model.embed_input_ids(ids[[0, 2, 3]]))
    assert raw.tolist() == [3, 248056, 4, 0]


def test_composition_has_one_native_decoder_and_preserves_pool_owner():
    model = wrapper()
    pool = Qwen4ExpPooledCache(
        SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None)),
        model,
    )
    assert pool.model is model.language_model
    assert model.get_language_model() is model.language_model
    names = list(model.state_dict())
    assert len(names) == len(set(names))
    assert all(name.startswith(("language_model.", "visual.")) for name in names)
    tensors = list(model.state_dict().values())
    assert len({x.untyped_storage().data_ptr() for x in tensors}) == len(tensors)
    assert pool.model.model.layers is model.language_model.model.layers


def test_visual_weights_are_consumed_and_decoder_loader_keeps_native_names():
    model = wrapper()
    loaded = model.load_weights(
        iter(
            [
                ("model.visual.weight", torch.ones(8)),
                ("model.language_model.layers.0.weight", torch.ones(8, 8)),
            ]
        )
    )
    assert model.visual.seen_weights == ["weight"]
    assert model.language_model.seen_weights == ["model.language_model.layers.0.weight"]
    assert loaded == {"visual.weight", "language_model.model.layers.0.weight"}
