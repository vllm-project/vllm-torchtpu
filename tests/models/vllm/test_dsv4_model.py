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
"""Unit tests for DeepSeek-V4 model registration."""

from vllm.model_executor.models import ModelRegistry

from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.models.vllm.deepseek_v4 import DeepseekV4ForCausalLM


def test_deepseek_v4_model_registration():
    """Verify DeepseekV4ForCausalLM resolves from the vLLM ModelRegistry."""
    register_models()
    entry = ModelRegistry.models.get("DeepseekV4ForCausalLM")
    assert entry is not None, "DeepseekV4ForCausalLM is not registered"
    model_cls = entry.load_model_cls()
    assert model_cls is DeepseekV4ForCausalLM
