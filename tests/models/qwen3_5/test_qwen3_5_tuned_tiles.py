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
"""The Qwen3.5 rule for the batched RPA tuned tiles and decode lanes: on for a
Qwen3.5 target, off for every other model, including one with the same layer
shape."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.layers.adapter import attention

QWEN3_5 = "Qwen3_5MoeForConditionalGeneration"
OTHERS = [
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen2ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "LlamaForCausalLM",
]


def _engine(arch, max_model_len=262144):
    return SimpleNamespace(
        speculative_config=None,
        model_config=SimpleNamespace(architectures=[arch], max_model_len=max_model_len),
        additional_config={},
    )


def test_the_rule_is_qwen3_5_only():
    assert attention.qwen3_5_tuned_tiles(_engine(QWEN3_5)) is True
    for arch in OTHERS:
        assert attention.qwen3_5_tuned_tiles(_engine(arch)) is False, arch
    assert attention.qwen3_5_tuned_tiles(None) is False
    assert attention.qwen3_5_tuned_tiles(SimpleNamespace(model_config=None)) is False


def _batched_layer(monkeypatch, engine):
    """A batched RPA layer with one fp8 KV head of 256 per device (a DP4 x TP2
    decode device of Qwen3.5-397B) built under `engine`."""

    def base_init(self, *args, **kwargs):
        self.head_size = 256
        self.num_kv_heads = 1
        self.kv_cache_quantized_dtype = torch.float8_e4m3fn

    monkeypatch.setattr(attention.PallasAttentionBackendImpl, "__init__", base_init)
    monkeypatch.setattr(attention, "get_current_vllm_config_or_none", lambda: engine)
    return attention.PallasBatchedRPAAttentionBackendImpl()


def test_qwen3_5_layers_take_the_tuned_tiles_and_lanes(monkeypatch):
    layer = _batched_layer(monkeypatch, _engine(QWEN3_5))
    assert layer.tuned_tiles is True
    assert layer.decode_lanes == 4


@pytest.mark.parametrize("arch", OTHERS)
def test_other_models_keep_the_default_tiles_and_lanes(monkeypatch, arch):
    layer = _batched_layer(monkeypatch, _engine(arch))
    assert layer.tuned_tiles is False
    assert layer.decode_lanes is None
