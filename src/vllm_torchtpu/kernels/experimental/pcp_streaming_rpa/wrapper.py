# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""PCP streaming RPA wrapper.

Re-exports core primitives from
`llm_tpu_core.kernels.experimental.pcp_streaming_rpa.wrapper`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.experimental.pcp_streaming_rpa.wrapper import (
    PCP_AXIS_NAME,
    compute_pcp_local_mapping,
    sharded_pcp_ragged_paged_attention,
)

__all__ = [
    "PCP_AXIS_NAME",
    "compute_pcp_local_mapping",
    "sharded_pcp_ragged_paged_attention",
]
