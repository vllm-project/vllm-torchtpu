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
"""First-principles fusion-scope advisor for TPU megakernel evaluation."""

from __future__ import annotations

from . import chips, constants, models, specs
from .chips import TPU7X_TOPOLOGIES, ChipSpec, resolve_chip_spec
from .models import (
    DEPLOYMENT_PRESETS,
    MODEL_PRESETS,
    ModelProfile,
    ParallelLayout,
    UnsupportedArchitectureError,
    model_profile_from_hf_config,
    resolve_model_profile,
)
from .specs import (
    EfficiencySpec,
    FusionScope,
    MoEMode,
    ObjectiveMode,
    Phase,
    RingPlan,
    WorkloadSpec,
    load_measurements,
)

__all__ = [
    "DEPLOYMENT_PRESETS",
    "MODEL_PRESETS",
    "TPU7X_TOPOLOGIES",
    "ChipSpec",
    "EfficiencySpec",
    "FusionScope",
    "MoEMode",
    "ModelProfile",
    "ObjectiveMode",
    "ParallelLayout",
    "Phase",
    "RingPlan",
    "UnsupportedArchitectureError",
    "WorkloadSpec",
    "chips",
    "constants",
    "load_measurements",
    "model_profile_from_hf_config",
    "models",
    "resolve_chip_spec",
    "resolve_model_profile",
    "specs",
]
