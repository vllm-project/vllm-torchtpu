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
"""TPU v7x megakernel performance advisor public exports."""

from __future__ import annotations

from . import chips, constants, models, roofline, spec_bridge, specs, vmem
from .chips import TPU7X_TOPOLOGIES, ChipSpec, resolve_chip_spec
from .constants import CORE_LIMITS, CoreLimits
from .models import (
    DEPLOYMENT_PRESETS,
    MODEL_PRESETS,
    ModelProfile,
    ParallelLayout,
    UnsupportedArchitectureError,
    model_profile_from_hf_config,
    resolve_model_profile,
)
from .roofline import evaluate_scopes, experts_hit_busiest
from .spec_bridge import verify_core_limits_sync
from .specs import (
    EfficiencySpec,
    FusionScope,
    MoEMode,
    ObjectiveMode,
    Phase,
    RingPlan,
    ScopeEstimate,
    WorkloadSpec,
    load_measurements,
)
from .vmem import (
    compute_all_reduce_transfer_cost,
    compute_per_token_vmem_bytes,
    evaluate_core_legality,
    exceeds_default_buckets,
    fit_vmem_budget_bytes,
    validate_advise_inputs,
)

__all__ = [
    "CORE_LIMITS",
    "DEPLOYMENT_PRESETS",
    "MODEL_PRESETS",
    "TPU7X_TOPOLOGIES",
    "ChipSpec",
    "CoreLimits",
    "EfficiencySpec",
    "FusionScope",
    "MoEMode",
    "ModelProfile",
    "ObjectiveMode",
    "ParallelLayout",
    "Phase",
    "RingPlan",
    "ScopeEstimate",
    "UnsupportedArchitectureError",
    "WorkloadSpec",
    "chips",
    "compute_all_reduce_transfer_cost",
    "compute_per_token_vmem_bytes",
    "constants",
    "evaluate_core_legality",
    "evaluate_scopes",
    "exceeds_default_buckets",
    "experts_hit_busiest",
    "fit_vmem_budget_bytes",
    "load_measurements",
    "model_profile_from_hf_config",
    "models",
    "resolve_chip_spec",
    "resolve_model_profile",
    "roofline",
    "spec_bridge",
    "specs",
    "validate_advise_inputs",
    "verify_core_limits_sync",
    "vmem",
]
