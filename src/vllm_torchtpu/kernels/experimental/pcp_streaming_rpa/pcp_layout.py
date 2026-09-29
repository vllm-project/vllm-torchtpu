# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""PCP rank-major token layout helpers.

Re-exports core primitives from .
"""

from vllm_torchtpu.llm_tpu_core.distributed.pcp_layout import (
    apply_pcp_rank_major_token_order,
    build_pcp_logits_indices,
    build_pcp_rank_major_token_order,
    pcp_local_token_counts,
    pcp_query_chunk_ranges,
    pcp_token_owner_chunk_ranges,
    pcp_token_owner_start_offsets,
)

__all__ = [
    "apply_pcp_rank_major_token_order",
    "build_pcp_logits_indices",
    "build_pcp_rank_major_token_order",
    "pcp_local_token_counts",
    "pcp_query_chunk_ranges",
    "pcp_token_owner_chunk_ranges",
    "pcp_token_owner_start_offsets",
]
