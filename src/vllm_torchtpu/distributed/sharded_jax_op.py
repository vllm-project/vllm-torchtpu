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
"""Register a torch custom op backed by a multi-device JAX/Pallas function.

Re-exports core primitives from `llm_tpu_core.distributed.sharded_jax_op`.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.distributed.sharded_jax_op import (
    sharded_jax_op,
)

__all__ = [
    "sharded_jax_op",
]
