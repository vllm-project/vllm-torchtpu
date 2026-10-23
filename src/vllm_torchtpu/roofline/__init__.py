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
"""TPU v7x decode roofline model and serving ceiling calculator."""

from __future__ import annotations

from .adapter import (
    PlanReconciliation,
    floor_for_profile,
    floor_from_plan,
    reconcile_plan_vs_standalone,
)
from .chips import CHIPS, ChipSpec
from .collectives import Collective
from .gap import ModuleGap, StepGapReport, compare_measured, normalize_measured
from .geometry import ModelGeometry, from_hf_config
from .modules import (
    Dtypes,
    ModuleCost,
    Parallel,
    StepShape,
    ep_busiest_experts,
    experts_touched,
    pass_costs,
)
from .recipe import plan_from_serve_args
from .serving import ConcurrencyRow, plan
from .step import StepFloor, spec_cycle_floor, step_floor

__all__ = [
    "CHIPS",
    "ChipSpec",
    "Collective",
    "Dtypes",
    "ModelGeometry",
    "ModuleCost",
    "ModuleGap",
    "Parallel",
    "PlanReconciliation",
    "ConcurrencyRow",
    "StepFloor",
    "StepGapReport",
    "StepShape",
    "compare_measured",
    "ep_busiest_experts",
    "experts_touched",
    "floor_for_profile",
    "floor_from_plan",
    "from_hf_config",
    "normalize_measured",
    "pass_costs",
    "plan",
    "plan_from_serve_args",
    "reconcile_plan_vs_standalone",
    "spec_cycle_floor",
    "step_floor",
]
