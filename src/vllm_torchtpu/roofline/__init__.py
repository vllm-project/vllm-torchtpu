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
"""Analytical TPU serving roofline, step floor, mesh solver, and CLI emitter."""

from __future__ import annotations

from .chips import CHIPS, TOPOLOGIES, ChipOverrides, ChipSpec, SliceTopology
from .collectives import Collective, CollectiveConfig, torus_collective_s
from .emit import (
    CeilingRow,
    LaunchPayload,
    emit_ceiling_rows,
    emit_launch_args,
    format_parallelism,
    pool_cli_args,
)
from .geometry import ModelGeometry, from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .memory import HbmBreakdown, KvFitInput, SeqKvLayout, hbm_footprint_per_core_bytes
from .modules import Dtypes, ModuleCost, Parallel, StepShape, pass_costs
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
from .step import FloorOptions, SpecShape, StepFloor, spec_cycle_floor, step_floor

__all__ = [
    "CHIPS",
    "TOPOLOGIES",
    "CeilingRow",
    "ChipOverrides",
    "ChipSpec",
    "Collective",
    "CollectiveConfig",
    "ConcurrencyRow",
    "Dtypes",
    "FloorOptions",
    "HbmBreakdown",
    "KvFitInput",
    "LaunchPayload",
    "ModelGeometry",
    "ModuleCost",
    "Parallel",
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
    "build_example_scenario",
    "dtypes_from_hf_config",
    "emit_ceiling_rows",
    "emit_launch_args",
    "format_parallelism",
    "from_hf_config",
    "hbm_footprint_per_core_bytes",
    "load_hf_config",
    "pass_costs",
    "plan",
    "pool_cli_args",
    "solve_serving_plan",
    "spec_cycle_floor",
    "step_floor",
    "torus_collective_s",
]
