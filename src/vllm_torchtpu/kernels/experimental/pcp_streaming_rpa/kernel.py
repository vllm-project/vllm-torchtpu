# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""PCP streaming prefill RPA kernels.

Re-exports core primitives from
`llm_tpu_core.kernels.experimental.pcp_streaming_rpa.kernel`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.experimental.pcp_streaming_rpa.kernel import (
    AXIS,
    ONLINE_STATE_INIT_VALUE,
    ONLINE_STATE_PADDED_LANES,
    PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
    TPU_HBM_ROW_TILE,
    TPU_KV_DMA_ALIGNMENT,
    pcp_streaming_attention_page_groups_packed_local_from_metadata,
)

__all__ = [
    "AXIS",
    "ONLINE_STATE_INIT_VALUE",
    "PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE",
    "TPU_HBM_ROW_TILE",
    "TPU_KV_DMA_ALIGNMENT",
    "ONLINE_STATE_PADDED_LANES",
    "pcp_streaming_attention_page_groups_packed_local_from_metadata",
]
