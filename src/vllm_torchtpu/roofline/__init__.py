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

from .chips import (
    CHIPS,
    SUPPORTED_CHIPS,
    TOPOLOGIES,
    ChipOverrides,
    ChipSpec,
    ConstantSpec,
    SliceTopology,
    chip_catalog,
)
from .collectives import (
    Collective,
    CollectiveConfig,
    collective_cut_bound_s,
    torus_collective_s,
)
from .constants import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_TOP_K,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SLO_TIERS,
    MAX_EMBEDDED_CANDIDATES,
    SOLVE_SCHEMA,
    SUPPORTED_MODEL_TYPES,
)
from .crossover import max_ep_load_ratio, sweep_prefix_hit_crossovers
from .emit import (
    CeilingRow,
    LaunchOptions,
    LaunchPayload,
    PointLaunchPayload,
    ProvenancePayload,
    SolvePayload,
    TopKEntry,
    build_solve_payload,
    emit_ceiling_rows,
    emit_launch_args,
    format_parallelism,
    parse_layout,
    point_launch_args,
    pool_cli_args,
    pool_label,
    reference_layout_report,
    top_layouts,
)
from .geometry import (
    LayerCounts,
    ModelGeometry,
    UnsupportedArchitectureError,
    count_layer_types,
    from_hf_config,
    layer_ffn_kinds,
    supports,
)
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .memory import (
    HbmBreakdown,
    KvFitInput,
    SeqKvLayout,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
)
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    kv_bytes_per_token_per_layer,
    pass_costs,
)
from .partition import partition_pp_layers
from .report import render_html, write_html
from .serving import ConcurrencyRow, ServingRequest, plan
from .solver import (
    EXAMPLE_SCENARIOS,
    FloorTimes,
    ParetoPoint,
    PoolMeshPlan,
    ScenarioOverrides,
    SearchStats,
    ServingPlan,
    SolveOptions,
    TrafficScenario,
    build_example_scenario,
    explain_layout_rejection,
    solve_serving_plan,
)
from .step import FloorOptions, SpecShape, StepFloor, spec_cycle_floor, step_floor

__all__ = [
    "CHIPS",
    "DEFAULT_GPU_MEMORY_UTILIZATION",
    "DEFAULT_TOP_K",
    "EXAMPLE_CONCURRENCIES",
    "EXAMPLE_SCENARIOS",
    "EXAMPLE_SLO_TIERS",
    "MAX_EMBEDDED_CANDIDATES",
    "SOLVE_SCHEMA",
    "SUPPORTED_CHIPS",
    "SUPPORTED_MODEL_TYPES",
    "TOPOLOGIES",
    "CeilingRow",
    "ChipOverrides",
    "ChipSpec",
    "Collective",
    "CollectiveConfig",
    "ConcurrencyRow",
    "ConstantSpec",
    "Dtypes",
    "FloorOptions",
    "FloorTimes",
    "HbmBreakdown",
    "KvFitInput",
    "LaunchOptions",
    "LaunchPayload",
    "LayerCounts",
    "ModelGeometry",
    "ModuleCost",
    "Parallel",
    "ParetoPoint",
    "PointLaunchPayload",
    "PoolMeshPlan",
    "ProvenancePayload",
    "ScenarioOverrides",
    "SearchStats",
    "SeqKvLayout",
    "ServingPlan",
    "ServingRequest",
    "SliceTopology",
    "SolveOptions",
    "SolvePayload",
    "SpecShape",
    "StepFloor",
    "StepShape",
    "TopKEntry",
    "TrafficScenario",
    "UnsupportedArchitectureError",
    "build_example_scenario",
    "build_solve_payload",
    "chip_catalog",
    "collective_cut_bound_s",
    "count_layer_types",
    "dtypes_from_hf_config",
    "emit_ceiling_rows",
    "emit_launch_args",
    "explain_layout_rejection",
    "format_parallelism",
    "from_hf_config",
    "hbm_breakdown_per_core_bytes",
    "hbm_footprint_per_core_bytes",
    "kv_bytes_per_token_per_layer",
    "layer_ffn_kinds",
    "load_hf_config",
    "max_ep_load_ratio",
    "parse_layout",
    "partition_pp_layers",
    "pass_costs",
    "plan",
    "point_launch_args",
    "pool_cli_args",
    "pool_label",
    "reference_layout_report",
    "render_html",
    "solve_serving_plan",
    "spec_cycle_floor",
    "step_floor",
    "supports",
    "sweep_prefix_hit_crossovers",
    "top_layouts",
    "torus_collective_s",
    "write_html",
]
