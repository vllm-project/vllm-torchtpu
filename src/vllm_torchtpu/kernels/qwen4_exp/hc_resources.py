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
class _MixMemory:
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


@dataclasses.dataclass(frozen=True)
class CombineMixMemory(_MixMemory):
    """Sizing for the boundary kernel fusing one combine with the next mix.

    The staged residual tile becomes the normalization target in place, the
    combined value lives in one single-buffered full-width ring whose output
    copy drains during each tile's projections, and the combine pre-stage adds
    a block ring, an incoming-logit ring, and their FP32 conversion planes.
    """

    pass


@dataclasses.dataclass(frozen=True)
class FullMixMemory(_MixMemory):
    pass


@dataclasses.dataclass(frozen=True)
class InjectMixMemory(_MixMemory):
    """Sizing for the mix variant that also projects injection gates.

    The normalized hyper input is consumed in vector memory in place of the
    staged input tile, so the kernel drops the full-width normalized output
    ring and adds the resident injection weight plus the per-token logit
    planes instead.
    """

    pass


@dataclasses.dataclass(frozen=True)
class TerminalMixMemory(_MixMemory):
    """Sizing for the boundary variant fusing the final combine with the
    terminal mix.

    The terminal consumer has no combine and no inject projection of its own,
    so the combined value needs no HBM stream and no combined ring: it is
    formed in place over the staged residual tile, which then carries the
    normalized value. The mix reduction is strip-scheduled like the boundary
    kernel, so no [tile, width] FP32 plane exists.
    """

    pass


def packed_projection_rows(
    lowrank: int, branches: int, hardware: HcHardware
) -> tuple[int, int]:
    """Row boundary and total rows of the packed down+inject resident plane.

    The down rows occupy ``[0, lowrank)``; the inject rows start at the first
    hardware row boundary at or after ``lowrank`` so their DMA destination
    begins on a physical sublane-tile row of the shared plane. The boundary
    rows in between belong to no DMA range.
    """
    inject_row = _align(lowrank, hardware.row_alignment)
    return inject_row, inject_row + branches


def _supports_mix_geometry(
    branches: int,
    hidden: int,
    lowrank: int,
    tile: int,
    hardware: HcHardware | None,
) -> bool:
    return not (
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
    )


def _normalization_headroom(width: int, hidden: int, hardware: HcHardware) -> int:
    return (
        6 * _plane(32, hidden, hardware.fp32_rows, hardware.lanes, 4)
        + 2 * _plane(32, 1, hardware.fp32_rows, hardware.lanes, 4)
        + _align(width, hardware.lanes) * 4
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
    normalization = _normalization_headroom(width, hidden, hardware)
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


def inject_mix_memory(
    branches: int, hidden: int, lowrank: int, tile: int, hardware: HcHardware
) -> InjectMixMemory:
    """Size the inject-projecting mix kernel with the same conservative policy.

    Relative to ``full_mix_memory`` the normalized-value output ring is
    replaced by in-place reuse of the staged input tile, the injection weight
    becomes resident, and the logits add one small BF16 ring plus one FP32
    working plane. The down and inject weights share one resident plane: the
    down rows run to the row-aligned boundary and the inject rows follow, so
    the two DMA ranges tile the plane without overlap and the two projections
    contract in one fullK dot. The remaining normalization/dot working-set
    headroom is unchanged.
    """
    if min(branches, hidden, lowrank, tile) <= 0 or not hardware.supported:
        raise ValueError("HC sizing requires positive geometry and known layout")
    width = branches * hidden
    inject_row, packed_rows = packed_projection_rows(lowrank, branches, hardware)

    def bf(rows, cols):
        return _plane(rows, cols, hardware.bf16_rows, hardware.lanes, 2)

    def fp(rows, cols):
        return _plane(rows, cols, hardware.fp32_rows, hardware.lanes, 4)

    resident = (
        _align(width, hardware.lanes) * 2 + bf(packed_rows, width) + bf(width, lowrank)
    )
    # One staged input ring (normalized in place) instead of separate input
    # and normalized-value rings, plus the mixed output and logit rings.
    buffers = 2 * bf(tile, width) + 2 * bf(tile, hidden) + 2 * bf(tile, branches)
    normalization = _normalization_headroom(width, hidden, hardware)
    packed_accumulator = _plane(
        tile, packed_rows, hardware.fp32_rows, hardware.mxu_columns, 4
    )
    working = (
        2 * fp(tile, width)
        + 2 * fp(tile, hidden)
        + bf(tile, packed_rows)
        + packed_accumulator
        + fp(tile, branches)
    )
    return InjectMixMemory(resident, buffers, working, normalization)


def supports_inject_mix(
    branches: int,
    hidden: int,
    lowrank: int,
    tile: int,
    hardware: HcHardware | None,
) -> bool:
    if not _supports_mix_geometry(branches, hidden, lowrank, tile, hardware):
        return False
    assert hardware is not None
    return inject_mix_memory(branches, hidden, lowrank, tile, hardware).total_bytes <= (
        hardware.vmem_bytes
    )


def inject_broadcast_mix_memory(
    branches: int, hidden: int, lowrank: int, tile: int, hardware: HcHardware
) -> InjectMixMemory:
    """Size the broadcast-input variant of the inject-projecting mix.

    The caller hands this kernel the un-expanded ``[tokens, hidden]`` residual
    whose expanded plane would repeat every branch slice, so the input ring
    stages ``hidden`` columns instead of the full width. The normalized dot
    operand stays full-width — the packed down+inject projection contracts the
    whole width — but it is no longer the DMA target: it is written by the
    in-kernel broadcast normalization and read only within the same token tile,
    so a single plane replaces the double-buffered input ring.
    """
    if min(branches, hidden, lowrank, tile) <= 0 or not hardware.supported:
        raise ValueError("HC sizing requires positive geometry and known layout")
    width = branches * hidden
    inject_row, packed_rows = packed_projection_rows(lowrank, branches, hardware)

    def bf(rows, cols):
        return _plane(rows, cols, hardware.bf16_rows, hardware.lanes, 2)

    def fp(rows, cols):
        return _plane(rows, cols, hardware.fp32_rows, hardware.lanes, 4)

    resident = (
        _align(width, hardware.lanes) * 2 + bf(packed_rows, width) + bf(width, lowrank)
    )
    buffers = (
        2 * bf(tile, hidden)
        + bf(tile, width)
        + 2 * bf(tile, hidden)
        + 2 * bf(tile, branches)
    )
    normalization = _normalization_headroom(width, hidden, hardware)
    packed_accumulator = _plane(
        tile, packed_rows, hardware.fp32_rows, hardware.mxu_columns, 4
    )
    working = (
        2 * fp(tile, width)
        + 2 * fp(tile, hidden)
        + bf(tile, packed_rows)
        + packed_accumulator
        + fp(tile, branches)
    )
    return InjectMixMemory(resident, buffers, working, normalization)


def supports_inject_broadcast_mix(
    branches: int,
    hidden: int,
    lowrank: int,
    tile: int,
    hardware: HcHardware | None,
) -> bool:
    if not _supports_mix_geometry(branches, hidden, lowrank, tile, hardware):
        return False
    assert hardware is not None
    memory = inject_broadcast_mix_memory(branches, hidden, lowrank, tile, hardware)
    return memory.total_bytes <= hardware.vmem_bytes


def combine_broadcast_mix_memory(
    branches: int, hidden: int, lowrank: int, tile: int, hardware: HcHardware
) -> CombineMixMemory:
    """Size the broadcast-residual variant of the boundary kernel.

    The prior owner's residual arrives un-expanded — every branch slice of the
    expanded plane carries the same ``[tokens, hidden]`` value — so the residual
    ring stages ``hidden`` columns instead of the full width. The combined
    value is formed from that ring per branch as before, and the normalized dot
    operand moves from the residual ring to a single full-width plane written
    by the strip normalization and read only within the same token tile.
    """
    if min(branches, hidden, lowrank, tile) <= 0 or not hardware.supported:
        raise ValueError("HC sizing requires positive geometry and known layout")
    width = branches * hidden
    inject_row, packed_rows = packed_projection_rows(lowrank, branches, hardware)

    def bf(rows, cols):
        return _plane(rows, cols, hardware.bf16_rows, hardware.lanes, 2)

    def fp(rows, cols):
        return _plane(rows, cols, hardware.fp32_rows, hardware.lanes, 4)

    resident = (
        _align(width, hardware.lanes) * 2 + bf(packed_rows, width) + bf(width, lowrank)
    )
    buffers = (
        2 * bf(tile, hidden)
        + bf(tile, width)
        + 2 * bf(tile, hidden)
        + 2 * bf(tile, branches)
        + bf(tile, width)
        + 2 * bf(tile, hidden)
        + 2 * bf(tile, branches)
    )
    normalization = _normalization_headroom(width, hidden, hardware)
    packed_accumulator = _plane(
        tile, packed_rows, hardware.fp32_rows, hardware.mxu_columns, 4
    )
    working = (
        4 * fp(tile, hidden)
        + 2 * fp(tile, branches)
        + bf(tile, packed_rows)
        + packed_accumulator
    )
    return CombineMixMemory(resident, buffers, working, normalization)


def supports_combine_broadcast_mix(
    branches: int,
    hidden: int,
    lowrank: int,
    tile: int,
    hardware: HcHardware | None,
) -> bool:
    if not _supports_mix_geometry(branches, hidden, lowrank, tile, hardware):
        return False
    assert hardware is not None
    memory = combine_broadcast_mix_memory(branches, hidden, lowrank, tile, hardware)
    return memory.total_bytes <= hardware.vmem_bytes


def terminal_mix_memory(
    branches: int, hidden: int, lowrank: int, tile: int, hardware: HcHardware
) -> TerminalMixMemory:
    """Size the terminal boundary kernel with the same conservative policy.

    Relative to ``combine_mix_memory`` the combined value never streams to
    HBM (no following combine exists) so its single-buffered ring and the
    outgoing logit ring disappear, and the resident weight plane holds only
    the down rows. The remaining combine pre-stage, normalization, and
    strip-scheduled reduction working sets are unchanged.
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
    buffers = (
        2 * bf(tile, width)
        + 2 * bf(tile, hidden)
        + 2 * bf(tile, branches)
        + 2 * bf(tile, hidden)
    )
    normalization = _normalization_headroom(width, hidden, hardware)
    lowrank_accumulator = _plane(
        tile, lowrank, hardware.fp32_rows, hardware.mxu_columns, 4
    )
    # Combine pre-stage plus strip-scheduled reduction: the peak concurrent
    # set matches the boundary kernel minus the inject logit planes.
    working = (
        4 * fp(tile, hidden)
        + 2 * fp(tile, branches)
        + bf(tile, lowrank)
        + lowrank_accumulator
    )
    return TerminalMixMemory(resident, buffers, working, normalization)


def supports_terminal_mix(
    branches: int,
    hidden: int,
    lowrank: int,
    tile: int,
    hardware: HcHardware | None,
) -> bool:
    if not _supports_mix_geometry(branches, hidden, lowrank, tile, hardware):
        return False
    assert hardware is not None
    return terminal_mix_memory(
        branches, hidden, lowrank, tile, hardware
    ).total_bytes <= (hardware.vmem_bytes)


def combine_mix_memory(
    branches: int, hidden: int, lowrank: int, tile: int, hardware: HcHardware
) -> CombineMixMemory:
    """Size the boundary kernel with the same conservative policy.

    Relative to ``inject_mix_memory`` the combine pre-stage adds the
    single-buffered combined ring, a double-buffered block ring and incoming
    logit ring, and the FP32 planes for the block operand and the injection
    gates. The down and inject weights share one resident plane: the down rows
    run to the row-aligned boundary and the inject rows follow, so the two DMA
    ranges tile the plane without overlap and the two projections contract in
    one fullK dot. The mix reduction runs one branch strip at a time, so its
    dot outputs, sigmoid-weighted products, and the branch accumulator hold
    ``[tile, hidden]`` planes reused strip-to-strip instead of the full-width
    FP32 gate planes of the wide reduction; no ``[tile, width]`` FP32 plane
    exists in this kernel.
    """
    if min(branches, hidden, lowrank, tile) <= 0 or not hardware.supported:
        raise ValueError("HC sizing requires positive geometry and known layout")
    width = branches * hidden
    inject_row, packed_rows = packed_projection_rows(lowrank, branches, hardware)

    def bf(rows, cols):
        return _plane(rows, cols, hardware.bf16_rows, hardware.lanes, 2)

    def fp(rows, cols):
        return _plane(rows, cols, hardware.fp32_rows, hardware.lanes, 4)

    resident = (
        _align(width, hardware.lanes) * 2 + bf(packed_rows, width) + bf(width, lowrank)
    )
    buffers = (
        2 * bf(tile, width)
        + 2 * bf(tile, hidden)
        + 2 * bf(tile, branches)
        + bf(tile, width)
        + 2 * bf(tile, hidden)
        + 2 * bf(tile, branches)
    )
    normalization = _normalization_headroom(width, hidden, hardware)
    packed_accumulator = _plane(
        tile, packed_rows, hardware.fp32_rows, hardware.mxu_columns, 4
    )
    # Strip-scheduled mix reduction plus the combine stage's residual/block
    # strips: the peak concurrent set is four [tile, hidden] FP32 planes (dot
    # output, sigmoid product, accumulator, block conversion), the two small
    # gate/logit planes, and the packed projection's lowrank+branch planes.
    working = (
        4 * fp(tile, hidden)
        + 2 * fp(tile, branches)
        + bf(tile, packed_rows)
        + packed_accumulator
    )
    return CombineMixMemory(resident, buffers, working, normalization)


def supports_combine_mix(
    branches: int,
    hidden: int,
    lowrank: int,
    tile: int,
    hardware: HcHardware | None,
) -> bool:
    if not _supports_mix_geometry(branches, hidden, lowrank, tile, hardware):
        return False
    assert hardware is not None
    return combine_mix_memory(
        branches, hidden, lowrank, tile, hardware
    ).total_bytes <= (hardware.vmem_bytes)


def supports_full_mix(
    branches: int,
    hidden: int,
    lowrank: int,
    tile: int,
    hardware: HcHardware | None,
) -> bool:
    if not _supports_mix_geometry(branches, hidden, lowrank, tile, hardware):
        return False
    assert hardware is not None
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
