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
"""Configures upstream EAGLE prefix-cache block retention."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def apply_eagle_block_drop_override(vllm_config: VllmConfig) -> None:
    """Propagates VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP into SpeculativeConfig."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None and envs.VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP:
        speculative_config.disable_eagle_block_drop = True
