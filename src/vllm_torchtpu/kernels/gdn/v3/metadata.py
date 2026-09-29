# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.metadata`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.metadata import (
    compute_batched_seq_metadata,
    compute_per_seq_metadata,
)

__all__ = [
    "compute_batched_seq_metadata",
    "compute_per_seq_metadata",
]
