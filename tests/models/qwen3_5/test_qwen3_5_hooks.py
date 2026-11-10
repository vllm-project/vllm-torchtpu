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
"""Tests for the Qwen3.5 config hook (models/qwen3_5/hooks.py)."""

from __future__ import annotations

import pytest
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    Qwen3_5ForConditionalGenerationConfig,
)

from vllm_torchtpu.models.qwen3_5 import settings
from vllm_torchtpu.models.qwen3_5.hooks import TPUQwen35MoeConfig
from vllm_torchtpu.models.vllm import register_models

QWEN35 = "Qwen3_5MoeForConditionalGeneration"


@pytest.fixture
def restore_config_map():
    saved = dict(MODELS_CONFIG_MAP)
    yield
    MODELS_CONFIG_MAP.clear()
    MODELS_CONFIG_MAP.update(saved)


def test_hook_is_registered_for_qwen3_5_only(restore_config_map):
    register_models()
    assert frozenset({QWEN35}) == settings.ARCHITECTURES
    assert MODELS_CONFIG_MAP[QWEN35] is TPUQwen35MoeConfig
    # vLLM's own Qwen3.5 hook stays in charge of the sibling architectures.
    for sibling in (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
    ):
        assert not issubclass(MODELS_CONFIG_MAP[sibling], TPUQwen35MoeConfig)


def test_hook_extends_vllms_qwen3_5_hook():
    assert issubclass(TPUQwen35MoeConfig, Qwen3_5ForConditionalGenerationConfig)
