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
"""PyTorch Kineto and TorchTPU profiler tracing module."""

from vllm_torchtpu.tracing.annotation import (
    TraceAnnotation,
    is_trace_annotation_enabled,
)
from vllm_torchtpu.tracing.utils import (
    extract_kv_lens_for_tracing,
    extract_request_ids_for_tracing,
    trim_request_id_suffix,
)

__all__ = [
    "TraceAnnotation",
    "is_trace_annotation_enabled",
    "extract_kv_lens_for_tracing",
    "extract_request_ids_for_tracing",
    "trim_request_id_suffix",
]
