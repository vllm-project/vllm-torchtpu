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
"""Analytical TPU serving roofline, step floor, and measured-vs-floor gap."""

from __future__ import annotations

from . import recipe
from .chips import CHIPS, TOPOLOGIES, ChipSpec, SliceTopology
from .collectives import (
    Collective,
    CollectiveTable,
    collective_ici_hops,
    submesh_for_group,
    torus_collective_s,
)
from .gap import (
    ModuleGap,
    StepGapReport,
    compare_measured,
    normalize_measured,
    step_gap,
)
from .geometry import LAYER_FAMILIES, ModelGeometry, from_hf_config
from .hf_loader import dtypes_from_hf_config, load_hf_config
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    ep_busiest_experts,
    experts_touched,
    pass_costs,
)
from .recipe import (
    memory_utilization_from_serve_args,
    plan_from_serve_args,
    spec_window_from_serve_args,
)
from .serving import ConcurrencyRow, plan
from .step import StepFloor, spec_cycle_floor, step_floor

__all__ = [
    "CHIPS",
    "LAYER_FAMILIES",
    "TOPOLOGIES",
    "ChipSpec",
    "Collective",
    "CollectiveTable",
    "ConcurrencyRow",
    "Dtypes",
    "ModelGeometry",
    "ModuleCost",
    "ModuleGap",
    "Parallel",
    "SliceTopology",
    "StepFloor",
    "StepGapReport",
    "StepShape",
    "compare_measured",
    "dtypes_from_hf_config",
    "ep_busiest_experts",
    "experts_touched",
    "memory_utilization_from_serve_args",
    "from_hf_config",
    "load_hf_config",
    "normalize_measured",
    "pass_costs",
    "plan",
    "plan_from_serve_args",
    "recipe",
    "spec_cycle_floor",
    "spec_window_from_serve_args",
    "step_floor",
    "step_gap",
    "collective_ici_hops",
    "submesh_for_group",
    "torus_collective_s",
]
