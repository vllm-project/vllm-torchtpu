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
"""MiniMax M3 model directory (see README.md).

Import-free: `models.vllm.register_model_directories` calls `register` at
plugin load in every process, before vLLM selects a model.
"""


def register() -> None:
    from vllm_torchtpu.models.minimax_m3.experimental import register as _register

    _register()
