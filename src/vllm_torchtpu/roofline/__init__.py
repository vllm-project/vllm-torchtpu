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
    CHIPS,
    SUPPORTED_CHIPS,
    TOPOLOGIES,
    ChipOverrides,
    ChipSpec,
    SliceTopology,
)
from .collectives import (
    Collective,
    CollectiveConfig,
    torus_collective_s,
)
from .constants import (
    ACTIVATION_HULL_SCHEMA,
    SUPPORTED_MODEL_TYPES,
)
from .geometry import (
    LayerCounts,
    ModelGeometry,
    count_layer_types,
    from_hf_config,
)
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    pass_costs,
)
from .occupancy import (
    ActivationRegime,
    ExpertActivationModel,
    ExpertDispatch,
    active_experts_envelope,
    bernoulli_expected_experts,
    load_activation_hull,
    min_experts_per_ep_shard,
    parse_activation_spec,
)
from .serving import ConcurrencyRow, ServingRequest, plan
from .step import (
    FloorOptions,
    SpecShape,
    StepFloor,
    spec_cycle_floor,
    step_floor,
)

__all__ = [
    "ACTIVATION_HULL_SCHEMA",
    "CHIPS",
    "SUPPORTED_CHIPS",
    "SUPPORTED_MODEL_TYPES",
    "TOPOLOGIES",
    "ActivationRegime",
    "ChipOverrides",
    "ChipSpec",
    "Collective",
    "CollectiveConfig",
    "ConcurrencyRow",
    "Dtypes",
    "ExpertActivationModel",
    "ExpertDispatch",
    "FloorOptions",
    "LayerCounts",
    "ModelGeometry",
    "ModuleCost",
    "Parallel",
    "ServingRequest",
    "SliceTopology",
    "SpecShape",
    "StepFloor",
    "StepShape",
    "active_experts_envelope",
    "bernoulli_expected_experts",
    "count_layer_types",
    "dtypes_from_hf_config",
    "from_hf_config",
    "load_activation_hull",
    "load_hf_config",
    "min_experts_per_ep_shard",
    "parse_activation_spec",
    "pass_costs",
    "plan",
    "spec_cycle_floor",
    "step_floor",
    "torus_collective_s",
]
