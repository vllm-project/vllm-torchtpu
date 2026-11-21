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
"""Analytical TPU serving roofline, step floor, and mesh solver."""

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
    register_chip,
    topologies_for_chip,
)
from .collectives import (
    Collective,
    CollectiveConfig,
    torus_collective_s,
)
from .constants import (
    ACTIVATION_HULL_SCHEMA,
    BENIGN_HF_METADATA_KEYS,
    INSPECT_SCHEMA,
    NON_GATED_MLP_MODEL_TYPES,
    SUPPORTED_MODEL_TYPES,
)
from .geometry import (
    KeyTracker,
    LayerCounts,
    ModelGeometry,
    count_layer_types,
    from_hf_config,
)
from .hf_loader import (
    ConfigAudit,
    audit_hf_config,
    dtypes_from_hf_config,
    load_hf_config,
)
from .memory import (
    HbmBreakdown,
    KvFitInput,
    ParamBreakdown,
    SeqKvLayout,
    hbm_footprint_per_core_bytes,
    param_counts,
)
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
from .solver import (
    ParetoPoint,
    PoolMeshPlan,
    ScenarioOverrides,
    ServingPlan,
    SolveOptions,
    TrafficScenario,
    build_example_scenario,
    solve_serving_plan,
)
from .step import (
    FloorOptions,
    SpecShape,
    StepFloor,
    spec_cycle_floor,
    step_floor,
)

__all__ = [
    "ACTIVATION_HULL_SCHEMA",
    "BENIGN_HF_METADATA_KEYS",
    "CHIPS",
    "CHIP_TOPOLOGIES",
    "INSPECT_SCHEMA",
    "NON_GATED_MLP_MODEL_TYPES",
    "SUPPORTED_CHIPS",
    "SUPPORTED_MODEL_TYPES",
    "TOPOLOGIES",
    "ActivationRegime",
    "ChipOverrides",
    "ChipSpec",
    "Collective",
    "CollectiveConfig",
    "ConcurrencyRow",
    "ConfigAudit",
    "Dtypes",
    "ExpertActivationModel",
    "ExpertDispatch",
    "FloorOptions",
    "HbmBreakdown",
    "KeyTracker",
    "KvFitInput",
    "LayerCounts",
    "ModelGeometry",
    "ModuleCost",
    "Parallel",
    "ParamBreakdown",
    "ParetoPoint",
    "PoolMeshPlan",
    "ScenarioOverrides",
    "SeqKvLayout",
    "ServingPlan",
    "ServingRequest",
    "SliceTopology",
    "SolveOptions",
    "SpecShape",
    "StepFloor",
    "StepShape",
    "TrafficScenario",
    "active_experts_envelope",
    "audit_hf_config",
    "bernoulli_expected_experts",
    "build_example_scenario",
    "count_layer_types",
    "dtypes_from_hf_config",
    "from_hf_config",
    "hbm_footprint_per_core_bytes",
    "load_activation_hull",
    "load_chip_spec",
    "load_hf_config",
    "min_experts_per_ep_shard",
    "param_counts",
    "parse_activation_spec",
    "pass_costs",
    "plan",
    "register_chip",
    "solve_serving_plan",
    "spec_cycle_floor",
    "step_floor",
    "topologies_for_chip",
    "torus_collective_s",
]
