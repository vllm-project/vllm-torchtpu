# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.pcp_metadata`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.pcp_metadata import (
    PcpStageMetadata,
    compute_pcp_stage_metadata,
)

__all__ = [
    "PcpStageMetadata",
    "compute_pcp_stage_metadata",
]
