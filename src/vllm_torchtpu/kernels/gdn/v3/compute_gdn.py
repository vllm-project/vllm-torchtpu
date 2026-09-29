# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.compute_gdn`.
"""

from vllm_torchtpu.llm_tpu_core.kernels.gdn.v3.compute_gdn import (
    chunked_gdn,
    chunked_gdn_per_seq,
    fused_transpose_broadcast,
    get_mask_dtype,
    invert_triangular_matrix,
    l2_norm,
    recurrent_gdn,
    recurrent_gdn_per_seq,
)

__all__ = [
    "chunked_gdn",
    "chunked_gdn_per_seq",
    "fused_transpose_broadcast",
    "get_mask_dtype",
    "invert_triangular_matrix",
    "l2_norm",
    "recurrent_gdn",
    "recurrent_gdn_per_seq",
]
