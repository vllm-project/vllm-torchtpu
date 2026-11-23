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
"""Public while-loop helpers for multi-step TPU model execution."""

from vllm_torchtpu.compilation.while_loop.ops import (
    masked_update_step_slice,
    tpu_while_loop,
)

__all__ = [
    "masked_update_step_slice",
    "tpu_while_loop",
]
