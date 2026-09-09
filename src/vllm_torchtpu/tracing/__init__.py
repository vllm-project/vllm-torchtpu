# Copyright 2026 Google LLC
"""PyTorch Kineto and TorchTPU profiler tracing module."""
from vllm_torchtpu.tracing.annotation import (TraceAnnotation,
                                              is_trace_annotation_enabled)
from vllm_torchtpu.tracing.utils import (extract_kv_lens_for_tracing,
                                         extract_request_ids_for_tracing,
                                         trim_request_id_suffix)

__all__ = [
    "TraceAnnotation",
    "is_trace_annotation_enabled",
    "extract_kv_lens_for_tracing",
    "extract_request_ids_for_tracing",
    "trim_request_id_suffix",
]
