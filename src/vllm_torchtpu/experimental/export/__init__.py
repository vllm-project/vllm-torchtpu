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
"""AOT StableHLO model export package for vLLM TorchTPU."""

from __future__ import annotations

from vllm_torchtpu.experimental.export.api import (
    ExportableModel,
    ExportableTextOnlyLLM,
    ModelExportConfig,
    ShapeDtypeStruct,
    TextOnlyLLMExportConfig,
)
from vllm_torchtpu.experimental.export.exporter import export
from vllm_torchtpu.experimental.export.rope_patch import (
    compute_mrope_dim_modalities,
    is_in_graph_rope_eligible,
    patch_in_graph_rotary_embedding,
)
from vllm_torchtpu.experimental.export.universal_adapter import (
    UniversalVllmExportableLLM,
    build_export_config_from_vllm,
    make_attention_metadata,
)
from vllm_torchtpu.experimental.export.validator import (
    ALLOWED_CUSTOM_CALLS,
    canonical_dtype,
    parse_mlir_main_signature,
    validate_exported_model_dir,
    validate_mlir_custom_calls,
)

__all__ = [
    "ALLOWED_CUSTOM_CALLS",
    "ExportableModel",
    "ExportableTextOnlyLLM",
    "ModelExportConfig",
    "ShapeDtypeStruct",
    "TextOnlyLLMExportConfig",
    "UniversalVllmExportableLLM",
    "build_export_config_from_vllm",
    "canonical_dtype",
    "compute_mrope_dim_modalities",
    "export",
    "is_in_graph_rope_eligible",
    "make_attention_metadata",
    "parse_mlir_main_signature",
    "patch_in_graph_rotary_embedding",
    "validate_exported_model_dir",
    "validate_mlir_custom_calls",
]
