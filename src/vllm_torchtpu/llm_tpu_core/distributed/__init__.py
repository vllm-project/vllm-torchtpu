# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Primitive distributed collective helpers and sharded JAX ops.

Strictly zero dependencies on vLLM.
"""

from __future__ import annotations

from vllm_torchtpu.llm_tpu_core.distributed.chip_topology import (
    ChipTopology,
    get_chip_topology,
)
from vllm_torchtpu.llm_tpu_core.distributed.collectives import (
    all_gather_tensor,
    all_reduce_sum,
)
from vllm_torchtpu.llm_tpu_core.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_device_ids,
    ep_mesh_index,
    ep_rank_order,
    ep_token_replica_groups,
    get_ep_group,
)
from vllm_torchtpu.llm_tpu_core.distributed.group_provider import (
    DefaultGroupProvider,
    ProcessGroupProvider,
    get_dcp_group,
    get_group_provider,
    get_pcp_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
    set_group_provider,
)
from vllm_torchtpu.llm_tpu_core.distributed.mesh_utils import (
    CpGroupLayout,
    get_cp_group_layout,
    get_mesh_shape_product,
    get_or_create_cp_mesh,
)
from vllm_torchtpu.llm_tpu_core.distributed.pallas_shapes import (
    get_global_shape,
    get_local_shape,
    install_pallas_shape_conversion,
)
from vllm_torchtpu.llm_tpu_core.distributed.pcp_rank_order import (
    build_topology_mesh_groups,
    validate_mesh_permutation,
)
from vllm_torchtpu.llm_tpu_core.distributed.pp_shift import (
    AXIS as PP_AXIS_NAME,
)
from vllm_torchtpu.llm_tpu_core.distributed.pp_shift import (
    get_or_create_pp_mesh,
    pp_permute_op,
    pp_positions,
)
from vllm_torchtpu.llm_tpu_core.distributed.sharded_jax_op import (
    sharded_jax_op,
)

__all__ = [
    "ChipTopology",
    "CpGroupLayout",
    "DefaultGroupProvider",
    "EP_AXIS_NAME",
    "ProcessGroupProvider",
    "PP_AXIS_NAME",
    "get_or_create_pp_mesh",
    "pp_permute_op",
    "pp_positions",
    "all_gather_tensor",
    "all_reduce_sum",
    "build_ep_mesh",
    "build_topology_mesh_groups",
    "get_cp_group_layout",
    "get_mesh_shape_product",
    "get_or_create_cp_mesh",
    "ep_device_ids",
    "ep_mesh_index",
    "ep_rank_order",
    "ep_token_replica_groups",
    "get_chip_topology",
    "get_dcp_group",
    "get_ep_group",
    "get_global_shape",
    "get_group_provider",
    "get_local_shape",
    "get_pcp_group",
    "get_pp_group",
    "get_tp_group",
    "get_world_group",
    "install_pallas_shape_conversion",
    "set_group_provider",
    "sharded_jax_op",
    "validate_mesh_permutation",
]
