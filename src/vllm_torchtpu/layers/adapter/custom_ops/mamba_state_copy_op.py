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
"""Donating slot-to-slot copy for Mamba recurrent state arrays.

Re-exports core primitives from `llm_tpu_core.layers.mamba_state_copy_op`.
"""

from vllm_torchtpu.llm_tpu_core.layers.mamba_state_copy_op import (
    copy_mamba_state_blocks,
    copy_mamba_state_rows,
)

__all__ = [
    "copy_mamba_state_blocks",
    "copy_mamba_state_rows",
]
