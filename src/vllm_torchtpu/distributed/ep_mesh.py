# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compatibility wrapper.

Re-exports core primitives from
`vllm_torchtpu.llm_tpu_core.distributed.ep_mesh`.
"""

from vllm_torchtpu.llm_tpu_core.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_device_ids,
    ep_mesh_index,
    ep_rank_order,
    ep_token_replica_groups,
    get_ep_group,
    logger,
)

__all__ = [
    "EP_AXIS_NAME",
    "build_ep_mesh",
    "ep_device_ids",
    "ep_mesh_index",
    "ep_rank_order",
    "ep_token_replica_groups",
    "get_ep_group",
    "logger",
]
