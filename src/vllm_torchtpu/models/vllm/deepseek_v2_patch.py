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

import vllm_torchtpu.envs as envs


def maybe_patch_deepseek_v2(*args: object, **kwargs: object) -> None:
    """Install the forwards when the flag is set; import nothing otherwise."""
    if not envs.TPU_MOE_PREFILL_SEQUENCE_PARALLEL:
        return None
    from vllm_torchtpu.layers.adapter import moe_prefill_sp

    moe_prefill_sp.install_forwards()
    return None
