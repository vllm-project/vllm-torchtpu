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
"""In-kernel inter-rank collectives and their build-time parameters, plus the
collective topology, remote DMA batching and SPMD axis helpers."""

from . import params, remote_copy, ring, spmd, topology
from .params import CommParams, comm_params, ring_scratch, stage_scratch
from .remote_copy import (
    RemoteCopyBatch,
    make_remote_copy_batch,
)
from .ring import AllReduceRing
from .spmd import (
    kernel_axis_index,
    mesh_axis_index,
)
from .topology import PeerTopology

__all__ = [
    "AllReduceRing",
    "CommParams",
    "PeerTopology",
    "RemoteCopyBatch",
    "comm_params",
    "kernel_axis_index",
    "make_remote_copy_batch",
    "mesh_axis_index",
    "params",
    "remote_copy",
    "ring",
    "ring_scratch",
    "spmd",
    "stage_scratch",
    "topology",
]
