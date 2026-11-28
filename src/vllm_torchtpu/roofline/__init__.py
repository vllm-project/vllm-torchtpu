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
"""Analytical TPU serving roofline, step floor, and physics mesh solver."""

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
    ACTIVATION_HULL_SCHEMA,
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
from .hf_loader import (
    ConfigAudit,
    apply_hf_overrides,
    audit_hf_config,
    dtypes_from_hf_config,
    load_hf_config,
    load_hf_config_with_identity,
)
from .memory import (
    HbmBreakdown,
    KvFitInput,
    ParamBreakdown,
    SeqKvLayout,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    param_counts,
    pp_stage_layer_scale,
)
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    dsa_indexer_layer_weight_elems,
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
    EXAMPLE_SCENARIOS,
    ParetoPoint,
    PoolFloorRecord,
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
    "EXAMPLE_SCENARIOS",
    "HF_REPO_TO_CANONICAL_ID",
    "SUPPORTED_CHIPS",
    "SUPPORTED_MODELS",
    "SUPPORTED_MODEL_TYPES",
    "TOPOLOGIES",
    "ActivationRegime",
    "ChipOverrides",
    "ChipSpec",
    "Collective",
    "CollectiveConfig",
    "ConcurrencyRow",
    "ConfigAudit",
    "CutPlacementOptions",
    "Dtypes",
    "ExpertActivationModel",
    "ExpertDispatch",
    "FloorOptions",
    "HbmBreakdown",
    "KeyTracker",
    "KvFitInput",
    "LayerCounts",
    "LinkCutLoad",
    "ModelGeometry",
    "ModelIdentity",
    "ModelSpec",
    "ModuleCost",
    "Parallel",
    "ParamBreakdown",
    "ParetoPoint",
    "PoolFloorRecord",
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
    "apply_hf_overrides",
    "audit_hf_config",
    "bernoulli_expected_experts",
    "build_example_scenario",
    "collective_axis_cut_bytes",
    "collective_cut_bound_s",
    "count_layer_types",
    "dsa_indexer_layer_weight_elems",
    "dtypes_from_hf_config",
    "eval_link_subset_bound_s",
    "from_hf_config",
    "hbm_breakdown_per_core_bytes",
    "hbm_footprint_per_core_bytes",
    "load_activation_hull",
    "load_chip_spec",
    "load_hf_config",
    "load_hf_config_with_identity",
    "min_collective_cut_bound_s",
    "min_experts_per_ep_shard",
    "outer_submesh_for_group",
    "override_chip_ici_bw",
    "param_counts",
    "parse_activation_spec",
    "pass_costs",
    "plan",
    "pp_stage_layer_scale",
    "register_chip",
    "resolve_model_identity",
    "solve_serving_plan",
    "spec_cycle_floor",
    "step_floor",
    "topologies_for_chip",
    "torus_collective_s",
]
