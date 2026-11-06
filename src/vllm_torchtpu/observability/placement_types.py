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
"""Typed records behind the TPU parallelism report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ParallelSummary:
    """Engine-wide parallel configuration."""

    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    enable_expert_parallel: bool
    prefill_context_parallel_size: int
    decode_context_parallel_size: int
    world_size: int
    nnodes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
