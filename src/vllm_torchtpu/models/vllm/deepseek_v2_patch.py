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
"""Per-bucket sequence-parallel forwards for vLLM's DeepSeek-V2 family
(TPU_MOE_PREFILL_SEQUENCE_PARALLEL; `layers/adapter/moe_prefill_sp.py`)."""

from typing import TYPE_CHECKING, Optional

import vllm_torchtpu.envs as envs

if TYPE_CHECKING:
    from vllm.config import ModelConfig

# The architectures whose backbone is vLLM's DeepseekV2Model.
DEEPSEEK_V2_BACKBONES = frozenset(
    {"DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM", "GlmMoeDsaForCausalLM"}
)


def maybe_patch_deepseek_v2(
    model_config: Optional["ModelConfig"] = None,
) -> bool | None:
    """Install the forwards when the flag is set and the model is one of
    `DEEPSEEK_V2_BACKBONES`; import nothing otherwise. Without a model config
    yet, decide at model load (False asks the registry to retry). A model the
    split does not serve is left alone here and refused by `moe_prefill_sp.arm`.
    """
    if not envs.TPU_MOE_PREFILL_SEQUENCE_PARALLEL:
        return None
    if model_config is None:
        return False
    if not DEEPSEEK_V2_BACKBONES & set(model_config.architectures or ()):
        return None
    from vllm_torchtpu.layers.adapter import moe_prefill_sp

    moe_prefill_sp.install_forwards()
    return None
