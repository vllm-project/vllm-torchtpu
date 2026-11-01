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
"""When the sequence-parallel MoE forwards patch installs."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.models.vllm import deepseek_v2_patch as dp


def _cfg(*archs):
    return SimpleNamespace(architectures=list(archs))


def test_flag_off_installs_nothing():
    with patch.object(envs, "TPU_MOE_PREFILL_SEQUENCE_PARALLEL", False):
        assert dp.maybe_patch_deepseek_v2(_cfg("GlmMoeDsaForCausalLM")) is None


def test_without_a_model_config_it_retries_at_model_load():
    with patch.object(envs, "TPU_MOE_PREFILL_SEQUENCE_PARALLEL", True):
        assert dp.maybe_patch_deepseek_v2(None) is False


def test_other_models_are_left_alone():
    with patch.object(envs, "TPU_MOE_PREFILL_SEQUENCE_PARALLEL", True):
        assert dp.maybe_patch_deepseek_v2(_cfg("Qwen3MoeForCausalLM")) is None


def test_deepseek_v2_backbones_install():
    calls = []
    fake = SimpleNamespace(install_forwards=lambda: calls.append(1))
    with (
        patch.object(envs, "TPU_MOE_PREFILL_SEQUENCE_PARALLEL", True),
        patch.dict(sys.modules, {"vllm_torchtpu.layers.adapter.moe_prefill_sp": fake}),
        patch("vllm_torchtpu.layers.adapter.moe_prefill_sp", fake, create=True),
    ):
        assert dp.maybe_patch_deepseek_v2(_cfg("GlmMoeDsaForCausalLM")) is None
    assert calls == [1]
