# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DCP (Decode Context Parallelism) process-group accessors.

Mirrors the role of distributed/pcp.py for PCP: a narrow, safe accessor
layer every call site shares, rather than each re-deriving group state
independently. The layout and mesh construction the two axes have in common
lives in mesh_utils.py.

"""

from __future__ import annotations

from typing import Any

from vllm_torchtpu.distributed.mesh_utils import (CpGroupLayout,
                                                  get_cp_group_layout,
                                                  get_or_create_cp_mesh)

# The JAX mesh axis the DCP kernels shard the KV cache over.
DCP_AXIS_NAME = "dcp"


def get_dcp_group() -> Any | None:
    """Return vLLM's DCP GroupCoordinator, or None if not initialized."""
    try:
        from vllm.distributed.parallel_state import get_dcp_group as _get_group
        return _get_group()
    except (AssertionError, ImportError):
        return None


def get_dcp_rank() -> int:
    group = get_dcp_group()
    return 0 if group is None else int(group.rank_in_group)


def get_dcp_world_size() -> int:
    group = get_dcp_group()
    return 1 if group is None else int(group.world_size)


def get_dcp_group_layout() -> CpGroupLayout:
    """The DCP group's rank order and the TPU device each rank sits on."""
    return get_cp_group_layout(get_dcp_group())


def get_or_create_dcp_mesh(axis_name: str | None = None) -> Any:
    """Build the op-local JAX mesh for the DCP axis.

    Under `tp=N, dcp=N` the DCP group covers the same devices as the TP group,
    and the mesh cache is keyed on the axis name as well as the device ids, so
    a run with both CP axes live keeps two distinct meshes rather than
    aliasing them.

    The axis name defaults to `DCP_AXIS_NAME`; the kernels take the same
    constant from here, so a caller that overrides it has to override it in
    both places or the mesh and the partition specs will not agree.
    """
    if axis_name is None:
        axis_name = DCP_AXIS_NAME
    return get_or_create_cp_mesh(axis_name=axis_name, group=get_dcp_group())
