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
"""Checks checkpoint completion before releasing BF16 layer weights."""

import sys
from types import ModuleType

import pytest
import torch

from vllm_torchtpu.model_loader_patches import _OnlineFp8LayerFinisher


class CpuOnlineFp8:
    online_fp8 = True

    def __init__(self):
        self.inputs = []

    def process_weights_after_loading(self, layer):
        self.inputs.append(layer.weight.detach().clone())
        layer.weight = torch.nn.Parameter(
            layer.weight.to(torch.float8_e4m3fn), requires_grad=False
        )


@pytest.fixture
def model(monkeypatch):
    fp8 = ModuleType("vllm_torchtpu.layers.adapter.quantization.fp8")
    fp8.VllmFp8LinearMethodTPU = CpuOnlineFp8
    monkeypatch.setitem(sys.modules, fp8.__name__, fp8)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleDict()
    for index in ("0", "10"):
        layer = torch.nn.Module()
        layer.qkv = torch.nn.Module()
        layer.qkv.quant_method = CpuOnlineFp8()
        layer.qkv.weight = torch.nn.Parameter(
            torch.zeros(3, 2, dtype=torch.bfloat16), requires_grad=False
        )
        layer.norm = torch.nn.LayerNorm(2)
        model.model.layers[index] = layer
    return model


def test_waits_for_all_checkpoint_shards_before_quantizing(model):
    names = [f"model.layers.0.{part}.weight" for part in ("q", "k", "v")]
    names.append("model.layers.0.mlp.experts.0.gate_proj.weight")
    finisher = _OnlineFp8LayerFinisher(model, names)
    layer = model.model.layers["0"].qkv
    for row, name in enumerate(names[:3]):
        layer.weight.data[row].fill_(row + 1)
        finisher.consumed(name)
        assert layer.weight.dtype == torch.bfloat16
        assert not layer.quant_method.inputs
    finisher.consumed(names[-1])
    assert layer.weight.dtype == torch.float8_e4m3fn
    expected = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.bfloat16)
    torch.testing.assert_close(layer.quant_method.inputs[0], expected)
    torch.testing.assert_close(layer.weight.float(), expected.float())
    finisher.consumed(names[-1])
    assert len(layer.quant_method.inputs) == 1


def test_interleaved_layers_complete_independently(model):
    names = [
        f"model.layers.{index}.{part}.weight"
        for index in ("0", "10")
        for part in ("q", "k")
    ]
    finisher = _OnlineFp8LayerFinisher(model, names)
    finisher.consumed(names[0])
    finisher.consumed(names[2])
    finisher.consumed(names[3])
    assert model.model.layers["0"].qkv.weight.dtype == torch.bfloat16
    assert model.model.layers["10"].qkv.weight.dtype == torch.float8_e4m3fn
    finisher.consumed(names[1])
    assert model.model.layers["0"].qkv.weight.dtype == torch.float8_e4m3fn


def test_incomplete_or_absent_layer_is_left_for_final_processing(model):
    names = ["model.layers.0.q.weight", "model.layers.0.v.weight"]
    finisher = _OnlineFp8LayerFinisher(model, names)
    finisher.consumed(names[0])
    finisher.consumed("mtp.layers.0.q.weight")
    for layer in model.model.layers.values():
        assert layer.qkv.weight.dtype == torch.bfloat16
        assert not layer.qkv.quant_method.inputs


def test_checkpoint_names_are_mapped_before_grouping(model):
    from vllm.model_executor.models.utils import WeightsMapper

    model.hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model.", "mtp.": None}
    )
    names = [
        "model.language_model.layers.0.q.weight",
        "model.language_model.layers.0.k.weight",
        "mtp.layers.0.q.weight",
    ]
    finisher = _OnlineFp8LayerFinisher(model, names)
    layer = model.model.layers["0"].qkv
    finisher.consumed(names[2])
    finisher.consumed(names[0])
    assert layer.weight.dtype == torch.bfloat16
    finisher.consumed(names[1])
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert model.model.layers["10"].qkv.weight.dtype == torch.bfloat16


def test_stacked_mapper_groups_shards_under_their_layer(model):
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    model.hf_to_vllm_mapper = Qwen3MoeForCausalLM.hf_to_vllm_mapper
    names = [f"model.layers.0.{proj}.weight" for proj in ("q_proj", "k_proj", "v_proj")]
    finisher = _OnlineFp8LayerFinisher(model, names)
    layer = model.model.layers["0"].qkv
    finisher.consumed(names[0])
    finisher.consumed(names[1])
    assert layer.weight.dtype == torch.bfloat16
    finisher.consumed(names[2])
    assert layer.weight.dtype == torch.float8_e4m3fn


def test_stacked_mapper_without_online_fp8_layers(model):
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    plain = torch.nn.Module()
    plain.model = torch.nn.Module()
    plain.model.layers = torch.nn.ModuleDict({"0": torch.nn.Linear(2, 2)})
    plain.hf_to_vllm_mapper = Qwen3MoeForCausalLM.hf_to_vllm_mapper
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ]
    finisher = _OnlineFp8LayerFinisher(plain, names)
    for name in names:
        finisher.consumed(name)
