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
"""Registration of the models in their own directories under ``models/``.

The models under ``models/vllm/`` register in ``models.vllm.register_models``,
which calls ``register_top_level_models`` for these.
"""

_BEAM_MODEL = "vllm_torchtpu.models.beam.model:BeamForCausalLM"


def register_top_level_models() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model("BeamForCausalLM", _BEAM_MODEL)
