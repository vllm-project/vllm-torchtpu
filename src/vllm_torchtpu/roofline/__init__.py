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
"""TPU chip specifications, analytical torus collectives, and model geometry."""

from __future__ import annotations

from .chips import CHIPS, TOPOLOGIES, ChipSpec, SliceTopology
from .collectives import (
    Collective,
    collective_ici_hops,
    submesh_for_group,
    torus_collective_s,
)
from .geometry import LAYER_FAMILIES, ModelGeometry, from_hf_config

__all__ = [
    "CHIPS",
    "LAYER_FAMILIES",
    "TOPOLOGIES",
    "ChipSpec",
    "Collective",
    "ModelGeometry",
    "collective_ici_hops",
    "SliceTopology",
    "from_hf_config",
    "submesh_for_group",
    "torus_collective_s",
]
