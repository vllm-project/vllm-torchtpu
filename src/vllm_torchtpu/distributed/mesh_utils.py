# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.distributed.mesh_utils`.
"""

from vllm_torchtpu.llm_tpu_core.distributed.mesh_utils import (
    CpGroupLayout,
    get_cp_group_layout,
    get_mesh_shape_product,
    get_or_create_cp_mesh,
    logger,
)

__all__ = [
    "CpGroupLayout",
    "get_cp_group_layout",
    "get_mesh_shape_product",
    "get_or_create_cp_mesh",
    "logger",
]
