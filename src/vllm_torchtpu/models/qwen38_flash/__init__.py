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
"""Qwen3.8-Flash-Next model directory (see README.md).

Import-free: the shared model registration calls `register()` of every model
directory at plugin load, before vLLM selects a model.
"""


def register() -> None:
    from vllm_torchtpu.models.qwen38_flash.experimental import hooks

    hooks.register()
