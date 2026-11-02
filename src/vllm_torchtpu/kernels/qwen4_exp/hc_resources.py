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
"""Static HC admission estimates, separate from compiler memory limits.

Sizes include vector-layout padding and explicit working planes, not measured
peak VMEM. The compiler still validates the admitted lowering. Hardware facts
are immutable and belong in any factory/JIT cache key that captures them.
"""

import dataclasses
import math

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

RESIDUAL_PIPELINE = pl.Buffered(2)


@dataclasses.dataclass(frozen=True)
class HcHardware:
    """Per-TensorCore capacity and pinned-lowering layout facts."""

    chip: str
    vmem_bytes: int
    lanes: int
    bf16_rows: int
    fp32_rows: int
    mxu_columns: int

    @classmethod
    def from_tpu_info(cls, info: pltpu.TpuInfo) -> "HcHardware":
        return cls(
            str(info.chip_version),
            info.vmem_capacity_bytes,
            info.num_lanes,
            info.get_sublane_tiling(jnp.bfloat16),
            info.get_sublane_tiling(jnp.float32),
            info.mxu_column_size,
        )

    @property
    def row_alignment(self) -> int:
        return math.lcm(self.bf16_rows, self.fp32_rows)

    @property
    def supported(self) -> bool:
        # HC branch slices retain the existing128-lane width contract.
        return (
            bool(self.chip)
            and 0 < self.vmem_bytes < 2**63
            and self.lanes == 128
            and self.bf16_rows > 0
            and self.fp32_rows > 0
        )


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _plane(rows: int, columns: int, row_tile: int, lanes: int, item: int) -> int:
    return _align(rows, row_tile) * _align(columns, lanes) * item


@dataclasses.dataclass(frozen=True)
class FullMixMemory:
    resident_bytes: int
    buffer_bytes: int
    working_bytes: int
    headroom_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.resident_bytes
            + self.buffer_bytes
            + self.working_bytes
            + self.headroom_bytes
        )


def full_mix_memory(
    branches: int, hidden: int, lowrank: int, tile: int, hardware: HcHardware
) -> FullMixMemory:
    """Size padded weights/rings and a conservative explicit temporary policy.

    Two full-width FP32 planes cover logits and gated normalized values;
    two hidden planes allow branch-reduction ping-pong. Lowrank BF16/FP32
    planes cover gate/projection boundaries. Reserve the entire normalization
    working set as additional compiler scheduling/layout headroom, rather than
    asserting that every logical expression materializes simultaneously.
    """
    if min(branches, hidden, lowrank, tile) <= 0 or not hardware.supported:
        raise ValueError("HC sizing requires positive geometry and known layout")
    width = branches * hidden

    def bf(rows, cols):
        return _plane(rows, cols, hardware.bf16_rows, hardware.lanes, 2)

    def fp(rows, cols):
        return _plane(rows, cols, hardware.fp32_rows, hardware.lanes, 4)

    resident = (
        _align(width, hardware.lanes) * 2 + bf(lowrank, width) + bf(width, lowrank)
    )
    buffers = 4 * bf(tile, width) + 2 * bf(tile, hidden)
    normalization = (
        6 * fp(32, hidden) + 2 * fp(32, 1) + _align(width, hardware.lanes) * 4
    )
    lowrank_accumulator = _plane(
        tile, lowrank, hardware.fp32_rows, hardware.mxu_columns, 4
    )
    working = (
        2 * fp(tile, width)
        + 2 * fp(tile, hidden)
        + bf(tile, lowrank)
        + lowrank_accumulator
    )
    return FullMixMemory(resident, buffers, working, normalization)


def supports_full_mix(
    branches: int,
    hidden: int,
    lowrank: int,
    tile: int,
    hardware: HcHardware | None,
) -> bool:
    if (
        hardware is None
        or not hardware.supported
        or hardware.mxu_columns <= 0
        or hardware.mxu_columns % hardware.lanes
        or 32 % hardware.bf16_rows
        or 32 % hardware.fp32_rows
        or min(branches, hidden, lowrank, tile) <= 0
        or hidden % hardware.lanes
        or tile % 128
        or max(branches * hidden, lowrank, tile) > 2**31 - 1
    ):
        return False
    return full_mix_memory(branches, hidden, lowrank, tile, hardware).total_bytes <= (
        hardware.vmem_bytes
    )


def supports_residual(
    branches: int, hidden: int, tile: int, hardware: HcHardware | None
) -> bool:
    if (
        hardware is None
        or not hardware.supported
        or min(branches, hidden, tile) <= 0
        or hidden % hardware.lanes
        or tile % hardware.row_alignment
        or max(branches * hidden, tile) > 2**31 - 1
    ):
        return False
    width = branches * hidden

    def bf(cols):
        return _plane(tile, cols, hardware.bf16_rows, hardware.lanes, 2)

    def fp(cols):
        return _plane(tile, cols, hardware.fp32_rows, hardware.lanes, 4)

    # Match the explicit BlockSpec strategy for residual input/output, block
    # input, and padded FP32 gates, including each plane's layout padding.
    # Keep that conservative budget for single-tile calls too. The three live
    # FP32 hidden working planes (block, residual, multiply/add) are not rings.
    buffers = RESIDUAL_PIPELINE.buffer_count * (
        2 * bf(width) + bf(hidden) + fp(branches)
    )
    return buffers + 3 * fp(hidden) <= hardware.vmem_bytes
