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
"""Tests for TPU_ROPE_CACHE_TRUNCATE default behavior."""

import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from vllm_torchtpu import envs
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_rope_cache_truncate_enabled_by_default(monkeypatch):
    monkeypatch.delenv("TPU_ROPE_CACHE_TRUNCATE", raising=False)
    assert envs.TPU_ROPE_CACHE_TRUNCATE is True

    monkeypatch.setenv("TPU_ROPE_CACHE_TRUNCATE", "0")
    assert envs.TPU_ROPE_CACHE_TRUNCATE is False


def test_truncate_rope_caches_slices_text_only_model():
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        rope = RotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=32768,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )
    model = torch.nn.Module()
    model.rope = rope

    fake_runner = type("FakeRunner", (), {})()
    fake_runner.is_multimodal_raw_input_only_supported = False
    fake_runner.model_config = type(
        "FakeModelConfig",
        (),
        {
            "is_multimodal_model": False,
            "multimodal_config": None,
            "max_model_len": 4096,
        },
    )()
    fake_runner.model = model

    TPUModelRunner._truncate_rope_caches(fake_runner)
    assert rope.cos_sin_cache.shape[0] == 4096
