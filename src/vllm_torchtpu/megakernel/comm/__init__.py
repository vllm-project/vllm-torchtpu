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
"""Framework-free in-kernel ICI remote-DMA collectives and SPMD helpers."""

from .async_ring import (
    AllReduceHandle,
    advance_allreduce,
    drain_async_sends,
    slot_for_step,
    start_allreduce,
    wait_allreduce,
)
from .collectives import (
    CollectiveHandle,
    all_gather,
    all_gather_scratch,
    reduce_scatter,
    reduce_scatter_scratch,
    start_all_gather,
    start_reduce_scatter,
    wait_all_gather,
    wait_reduce_scatter,
)
from .params import CommParams, comm_params, ring_scratch, stage_scratch
from .ring import AllReduceRing
from .spmd import mesh_axis_index, spmd_safe_axis_index

__all__ = [
    "AllReduceHandle",
    "AllReduceRing",
    "CollectiveHandle",
    "CommParams",
    "advance_allreduce",
    "all_gather",
    "all_gather_scratch",
    "comm_params",
    "drain_async_sends",
    "mesh_axis_index",
    "reduce_scatter",
    "reduce_scatter_scratch",
    "ring_scratch",
    "slot_for_step",
    "spmd_safe_axis_index",
    "stage_scratch",
    "start_all_gather",
    "start_allreduce",
    "start_reduce_scatter",
    "wait_all_gather",
    "wait_allreduce",
    "wait_reduce_scatter",
]
