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

from . import (
    assets,
    calibrate,
    chips,
    cli,
    constants,
    models,
    regimes,
    report,
    roofline,
    serve,
    spec_bridge,
    specs,
    text_format,
    verdict,
    vmem,
    workbench,
)
from .calibrate import (
    CalibrateOptions,
    CalibrationPayload,
    CalibrationSourceMeta,
    CollectiveSummary,
    QuantileSummary,
    StepCoverageSummary,
    XPlaneToolUnavailableError,
    calibrate_trace,
)
from .chips import TPU7X_TOPOLOGIES, ChipSpec, resolve_chip_spec
from .cli import (
    build_calibrate_arg_parser,
    build_parser,
    build_serve_arg_parser,
    derive_decoding_sequences,
    format_model_listing,
    main,
)
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
from .regimes import (
    ParallelismSearchOptions,
    RegimeSweepOptions,
    advise_regimes,
    recommend_parallelism,
)
from .report import (
    build_bucket_schedule,
    render_advisor_html,
    write_advisor_html,
    write_bucket_schedule,
)
from .roofline import evaluate_scopes, experts_hit_busiest
from .serve import (
    WorkbenchServerOptions,
    create_workbench_server,
    run_workbench_server,
)
from .spec_bridge import verify_core_limits_sync
from .specs import (
    BUCKET_CONSUMER_STATUS,
    BUCKET_SCHEDULE_SCHEMA,
    AdvisorReport,
    AdvisorSweepReport,
    BucketScheduleEntry,
    BucketScheduleReport,
    DisaggHandoffSpec,
    EfficiencySpec,
    ExecutionMode,
    FusionScope,
    HostTableBreakSpec,
    HyperConnectionSpec,
    LayerGroupCandidate,
    LayerGroupSearchReport,
    MoEMode,
    ObjectiveMode,
    ParallelismComparisonRow,
    ParallelismRecommendation,
    Phase,
    RegimeEntry,
    RingPlan,
    ScopeEstimate,
    SparseAttnSpec,
    SparseCacheOwner,
    WorkloadSpec,
    load_measurements,
)
from .text_format import (
    format_advisor_summary,
    format_layer_group_search,
    format_regime_table,
    vector_register_load_cap_line,
)
from .verdict import advise, compare_parallelism_candidates, search_layer_groups
from .vmem import (
    compute_all_reduce_transfer_cost,
    compute_per_token_vmem_bytes,
    evaluate_core_legality,
    exceeds_default_buckets,
    fit_vmem_budget_bytes,
    validate_advise_inputs,
)
from .workbench import (
    PhaseMapOptions,
    render_workbench_html,
    write_workbench_html,
)

__all__ = [
    "BUCKET_CONSUMER_STATUS",
    "BUCKET_SCHEDULE_SCHEMA",
    "CORE_LIMITS",
    "DEPLOYMENT_PRESETS",
    "MODEL_PRESETS",
    "TPU7X_TOPOLOGIES",
    "AdvisorReport",
    "AdvisorSweepReport",
    "BucketScheduleEntry",
    "BucketScheduleReport",
    "CalibrateOptions",
    "CalibrationPayload",
    "CalibrationSourceMeta",
    "ChipSpec",
    "CollectiveSummary",
    "CoreLimits",
    "DisaggHandoffSpec",
    "EfficiencySpec",
    "ExecutionMode",
    "FusionScope",
    "HostTableBreakSpec",
    "HyperConnectionSpec",
    "LayerGroupCandidate",
    "LayerGroupSearchReport",
    "MoEMode",
    "ModelProfile",
    "ObjectiveMode",
    "ParallelLayout",
    "ParallelismComparisonRow",
    "ParallelismRecommendation",
    "ParallelismSearchOptions",
    "Phase",
    "PhaseMapOptions",
    "QuantileSummary",
    "RegimeEntry",
    "RegimeSweepOptions",
    "RingPlan",
    "ScopeEstimate",
    "SparseAttnSpec",
    "SparseCacheOwner",
    "StepCoverageSummary",
    "UnsupportedArchitectureError",
    "WorkbenchServerOptions",
    "WorkloadSpec",
    "XPlaneToolUnavailableError",
    "advise",
    "advise_regimes",
    "assets",
    "build_bucket_schedule",
    "build_calibrate_arg_parser",
    "build_parser",
    "build_serve_arg_parser",
    "calibrate",
    "calibrate_trace",
    "chips",
    "cli",
    "compare_parallelism_candidates",
    "compute_all_reduce_transfer_cost",
    "compute_per_token_vmem_bytes",
    "constants",
    "create_workbench_server",
    "derive_decoding_sequences",
    "evaluate_core_legality",
    "evaluate_scopes",
    "exceeds_default_buckets",
    "experts_hit_busiest",
    "fit_vmem_budget_bytes",
    "format_advisor_summary",
    "format_layer_group_search",
    "format_model_listing",
    "format_regime_table",
    "load_measurements",
    "main",
    "model_profile_from_hf_config",
    "models",
    "recommend_parallelism",
    "regimes",
    "render_advisor_html",
    "render_workbench_html",
    "report",
    "resolve_chip_spec",
    "resolve_model_profile",
    "roofline",
    "run_workbench_server",
    "search_layer_groups",
    "serve",
    "spec_bridge",
    "specs",
    "text_format",
    "validate_advise_inputs",
    "vector_register_load_cap_line",
    "verdict",
    "verify_core_limits_sync",
    "vmem",
    "workbench",
    "write_advisor_html",
    "write_bucket_schedule",
    "write_workbench_html",
]
