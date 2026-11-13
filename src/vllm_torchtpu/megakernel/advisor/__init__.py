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

from . import (
    chips,
    constants,
    jax_bridge,
    models,
    roofline,
    spec_bridge,
    specs,
    support,
    verdict,
    vmem,
)
from .chips import (
    BUILTIN_CHIP_SPECS,
    REQUIRED_CHIP_SPEC_KEYS,
    SUPPORTED_CHIPS,
    TPU6E_TOPOLOGIES,
    TPU7X_TOPOLOGIES,
    ChipSpec,
    FieldSource,
    UnsupportedChipError,
    detect_tpu_from_sysfs,
    load_chip_spec_file,
    load_public_calibration,
    public_calibration_path,
    resolve_chip_spec,
    verify_chip_spec_against_jax,
)
from .models import (
    DEPLOYMENT_PRESET_SOURCES,
    DEPLOYMENT_PRESETS,
    MODEL_PRESET_SOURCES,
    MODEL_PRESETS,
    RECOGNIZED_MODEL_TYPES,
    REGISTERED_MODEL_ALIASES,
    SUPPORTED_MODEL_TYPES,
    DeploymentPreset,
    ModelProfile,
    ParallelLayout,
    UnsupportedArchitectureError,
    model_profile_from_hf_config,
    resolve_hf_snapshot_config,
    resolve_model_profile,
    supports_hf_config,
)
from .roofline import validate_workload
from .spec_bridge import advise_from_spec
from .specs import (
    SUPPORTED_ONCHIP_FAMILIES,
    AdvisorReport,
    DispatchRegime,
    EfficiencySpec,
    FusionScope,
    LayerGroupCandidate,
    LayerGroupSearchResult,
    MeasurementsData,
    MoEMode,
    ObjectiveMode,
    Phase,
    ProvenanceMetric,
    ProvenanceTag,
    RingPlan,
    ScopeEstimate,
    SupportStatus,
    WorkloadSpec,
    default_double_buffer_ring_plan,
    load_measurements,
)
from .support import CORE_LIMITS, CoreLimits, evaluate_support
from .verdict import (
    advise,
    compare_parallelism_candidates,
    compute_scope_raw_step_us,
    search_layer_groups,
)
from .vmem import (
    compute_all_reduce_transfer_cost,
    compute_per_token_vmem_bytes,
    fit_vmem_budget_bytes,
    validate_advise_inputs,
)

__all__ = [
    "BUILTIN_CHIP_SPECS",
    "CORE_LIMITS",
    "DEPLOYMENT_PRESETS",
    "DEPLOYMENT_PRESET_SOURCES",
    "MODEL_PRESETS",
    "MODEL_PRESET_SOURCES",
    "RECOGNIZED_MODEL_TYPES",
    "REGISTERED_MODEL_ALIASES",
    "REQUIRED_CHIP_SPEC_KEYS",
    "SUPPORTED_CHIPS",
    "SUPPORTED_MODEL_TYPES",
    "SUPPORTED_ONCHIP_FAMILIES",
    "TPU6E_TOPOLOGIES",
    "TPU7X_TOPOLOGIES",
    "AdvisorReport",
    "ChipSpec",
    "CoreLimits",
    "DeploymentPreset",
    "DispatchRegime",
    "EfficiencySpec",
    "FieldSource",
    "FusionScope",
    "LayerGroupCandidate",
    "LayerGroupSearchResult",
    "MeasurementsData",
    "MoEMode",
    "ModelProfile",
    "ObjectiveMode",
    "ParallelLayout",
    "Phase",
    "ProvenanceMetric",
    "ProvenanceTag",
    "RingPlan",
    "ScopeEstimate",
    "SupportStatus",
    "UnsupportedArchitectureError",
    "UnsupportedChipError",
    "WorkloadSpec",
    "advise",
    "advise_from_spec",
    "chips",
    "compare_parallelism_candidates",
    "compute_all_reduce_transfer_cost",
    "compute_per_token_vmem_bytes",
    "compute_scope_raw_step_us",
    "constants",
    "default_double_buffer_ring_plan",
    "detect_tpu_from_sysfs",
    "evaluate_support",
    "fit_vmem_budget_bytes",
    "jax_bridge",
    "load_chip_spec_file",
    "load_measurements",
    "load_public_calibration",
    "model_profile_from_hf_config",
    "models",
    "public_calibration_path",
    "resolve_chip_spec",
    "resolve_hf_snapshot_config",
    "resolve_model_profile",
    "roofline",
    "search_layer_groups",
    "spec_bridge",
    "specs",
    "support",
    "supports_hf_config",
    "validate_advise_inputs",
    "validate_workload",
    "verdict",
    "verify_chip_spec_against_jax",
    "vmem",
]
