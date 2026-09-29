# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.distributed.pcp`.
"""

from vllm_torchtpu.llm_tpu_core.distributed.pcp import (
    PcpGroupLayout,
    all_gather_equal_tokens,
    all_reduce_sum,
    get_or_create_pcp_mesh,
    get_pcp_group,
    get_pcp_group_layout,
    get_pcp_rank,
    get_pcp_world_size,
)

__all__ = [
    "PcpGroupLayout",
    "all_gather_equal_tokens",
    "all_reduce_sum",
    "get_or_create_pcp_mesh",
    "get_pcp_group",
    "get_pcp_group_layout",
    "get_pcp_rank",
    "get_pcp_world_size",
]
