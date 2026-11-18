# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Shared byte-unit, topology, and layout constants."""

from __future__ import annotations

MIB_BYTES: int = 1 << 20
GIB_BYTES: int = 1 << 30
SUBLANE_ROWS: int = 8
VREG_LANE_COUNT: int = 128

AXIS_TP: str = "tp"
TP_SIZE: int = 32
TP_PEERS: int = TP_SIZE - 1
DP2_GROUPS: int = 2

NUM_LAYERS: int = 93
HIDDEN_SIZE: int = 7168
HEAD_DIM: int = 128
KDA_HEADS: int = 3
CONV_TAPS: int = 3
CONV_HEAD_ROWS: int = 4
ROUTED_EXPERTS: int = 384
ROUTED_EXPERTS_PER_TOKEN: int = 8

OVERFLOW_WINDOW_TOKENS: int = 8192
STORE_TILE: int = 128
PREFILL_WINDOWS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)

MLP_PADDED: int = 512
CHUNK: int = HIDDEN_SIZE // 8
FC_IN: int = 35840 // 4
ROW_PACKED: int = HIDDEN_SIZE // 256
KERNEL_SHARD_FORMAT: int = 1
EXPERT_FAMILIES: tuple[str, ...] = (
    "expert_gu",
    "expert_gus",
    "expert_down",
    "expert_ds",
)

__all__ = [
    "AXIS_TP",
    "CHUNK",
    "CONV_HEAD_ROWS",
    "CONV_TAPS",
    "DP2_GROUPS",
    "EXPERT_FAMILIES",
    "FC_IN",
    "GIB_BYTES",
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "KDA_HEADS",
    "KERNEL_SHARD_FORMAT",
    "MIB_BYTES",
    "MLP_PADDED",
    "NUM_LAYERS",
    "OVERFLOW_WINDOW_TOKENS",
    "PREFILL_WINDOWS",
    "ROUTED_EXPERTS",
    "ROUTED_EXPERTS_PER_TOKEN",
    "ROW_PACKED",
    "STORE_TILE",
    "SUBLANE_ROWS",
    "TP_PEERS",
    "TP_SIZE",
    "VREG_LANE_COUNT",
]
