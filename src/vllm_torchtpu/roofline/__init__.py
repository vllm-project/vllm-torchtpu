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
"""TPU roofline serving, memory, collective, and execution modeling."""

from __future__ import annotations

from .chips import (
    CHIP_TOPOLOGIES,
    CHIPS,
    SUPPORTED_CHIPS,
    TOPOLOGIES,
    ChipOverrides,
    ChipSpec,
    SliceTopology,
    load_chip_spec,
    override_chip_ici_bw,
    register_chip,
    topologies_for_chip,
)
from .collectives import (
    Collective,
    CollectiveConfig,
    CutPlacementOptions,
    LinkCutLoad,
    collective_axis_cut_bytes,
    collective_cut_bound_s,
    eval_link_subset_bound_s,
    min_collective_cut_bound_s,
    outer_submesh_for_group,
    torus_collective_s,
)
from .constants import (
    BENIGN_HF_METADATA_KEYS,
    HF_REPO_TO_CANONICAL_ID,
    SUPPORTED_MODEL_TYPES,
    SUPPORTED_MODELS,
    ModelSpec,
)
from .geometry import (
    KeyTracker,
    LayerCounts,
    ModelGeometry,
    ModelIdentity,
    count_layer_types,
    from_hf_config,
    resolve_model_identity,
)

__all__ = [
    "BENIGN_HF_METADATA_KEYS",
    "CHIPS",
    "CHIP_TOPOLOGIES",
    "HF_REPO_TO_CANONICAL_ID",
    "SUPPORTED_CHIPS",
    "SUPPORTED_MODELS",
    "SUPPORTED_MODEL_TYPES",
    "TOPOLOGIES",
    "ChipOverrides",
    "ChipSpec",
    "Collective",
    "CollectiveConfig",
    "CutPlacementOptions",
    "KeyTracker",
    "LayerCounts",
    "LinkCutLoad",
    "ModelGeometry",
    "ModelIdentity",
    "ModelSpec",
    "SliceTopology",
    "collective_axis_cut_bytes",
    "collective_cut_bound_s",
    "count_layer_types",
    "eval_link_subset_bound_s",
    "from_hf_config",
    "load_chip_spec",
    "min_collective_cut_bound_s",
    "outer_submesh_for_group",
    "override_chip_ici_bw",
    "register_chip",
    "resolve_model_identity",
    "topologies_for_chip",
    "torus_collective_s",
]
