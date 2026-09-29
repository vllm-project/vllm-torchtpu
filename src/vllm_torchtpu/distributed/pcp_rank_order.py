# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.distributed.pcp_rank_order`.
"""

from vllm_torchtpu.llm_tpu_core.distributed.pcp_rank_order import (
    build_topology_mesh_groups,
    validate_mesh_permutation,
)

__all__ = [
    "build_topology_mesh_groups",
    "validate_mesh_permutation",
]
