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
"""Group layout and JAX mesh construction shared by the context-parallel axes.

Re-exports core primitives from `llm_tpu_core.distributed.mesh_utils`.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.distributed.mesh_utils import (
    _LAYOUT_CACHE,
    _MESH_CACHE,
    CpGroupLayout,
    _collect_rank_to_device_id,
    _get_current_global_rank,
    _get_tpu_global_device_id,
    get_cp_group_layout,
    get_or_create_cp_mesh,
    logger,
)

__all__ = [
    "CpGroupLayout",
    "_LAYOUT_CACHE",
    "_MESH_CACHE",
    "_collect_rank_to_device_id",
    "_get_current_global_rank",
    "_get_tpu_global_device_id",
    "get_cp_group_layout",
    "get_or_create_cp_mesh",
    "logger",
]
