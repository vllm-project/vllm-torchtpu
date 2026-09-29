# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.pcp_wrapper`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.pcp_wrapper import (
    fused_qkvz_projection_pcp_gdn,
)

__all__ = [
    "fused_qkvz_projection_pcp_gdn",
]
