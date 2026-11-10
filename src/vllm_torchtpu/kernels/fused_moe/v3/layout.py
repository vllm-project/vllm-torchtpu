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
"""Weight tensor relayout and lane-aligned packing utilities for Fused EP MoE v3."""

# Lane count of the vector unit: a token row is a whole number of LANES-wide
# blocks, and every (SUBLANES, LANES) tile of 32-bit words the tables, the
# routing message and the arrival scale rows move is this wide.
LANES = 128
# Sublane count of a 32-bit tile. A packed four-bit weight block has to be a
# whole number of these deep. device.check_u32_sublane_tile checks the
# literal against the chip at build time.
SUBLANES = 8
# Every transport moves whole ROW_BLOCK-row blocks. Dynamic DMA offsets are
# block-aligned.
ROW_BLOCK = 8
# Eight four-bit values pack into one 32-bit word.
FP4_PER_WORD = 8
# Bytes in a 32-bit word, the unit the sparse-core gathers and the word
# views move.
WORD_BYTES = 4
# Bits and mask of one byte, for tables packed byte by byte.
BYTE_BITS = 8
BYTE_MASK = (1 << BYTE_BITS) - 1
# The first value a signed 32-bit word cannot hold.
INT32_LIMIT = 2**31
# Bytes in a mebibyte, for the VMEM figures in messages.
MIB = 2**20
# The single-axis mesh name the layer's shard_map and every collective
# kernel use.
MESH_AXIS = "d"

# Whether the programs that issue remote DMAs declare themselves side-effecting
# to XLA. True for the layer's own callers. Under torch_tpu's export bridge the
# SPMD partitioner refuses a side-effecting custom call that carries no sharding
# ("Side-effect HLO must have sharding"), and every such program's
# outputs are consumed downstream (the finish program's `done` word gates the
# combine's index), so the flag is not what keeps them alive; the bridge's
# caller sets it False.
SIDE_EFFECTS = True

# Whether the shard's own index reaches the transport and shard-tables
# programs as an operand (an int32 [1] in SMEM, from a [num_shards, 1] array
# the layer shards over the mesh axis) instead of `lax.axis_index`. False
# for the layer's own callers. Under torch_tpu's export bridge `axis_index`
# lowers to a partition-id that XLA's SPMD partitioner refuses ("partitionId
# instruction is not supported for SPMD partitioning"), so the
# bridge's caller sets it True and supplies the index as data.
SHARD_INDEX_OPERAND = False


def device_coords(peer, mesh_axes=(MESH_AXIS,)):
    """The mesh coordinates of shard `peer` of the expert-parallel axis, as
    a remote copy or signal takes them (pl.DeviceIdType.MESH): `peer` on
    MESH_AXIS and this core's own index on every other axis, so the cores
    of one replica only ever address each other."""
    from jax import lax

    if SHARD_INDEX_OPERAND:
        # The dict form, keyed by axis name: it names the mesh axis to Pallas
        # (a NamedAxisEffect) without an axis lookup, so the program's mesh
        # context is built even when nothing else references the axis.
        # With the shard index supplied as data, the tuple form can leave
        # the mesh without axes and fail device-coordinate validation.
        # Axes other than MESH_AXIS keep this core's own index.
        return {
            axis: (peer if axis == MESH_AXIS else lax.axis_index(axis))
            for axis in mesh_axes
        }
    return tuple(
        peer if axis == MESH_AXIS else lax.axis_index(axis) for axis in mesh_axes
    )


# A barrier semaphore is shared by every kernel built with the same
# collective id, so the two collective kernels of one layer call take
# different ones.
COLLECTIVE_ID_FFN = 0
COLLECTIVE_ID_TRANSPORT = 3


def align_up(value, multiple):
    """Round value up to a multiple of multiple."""
    return -(-value // multiple) * multiple


def row_lane_blocks(hidden):
    """LANES-wide blocks one token row of `hidden` values spans."""
    return hidden // LANES


def result_lane_blocks(hidden):
    """Blocks a result row is staged, pushed and gathered as: the row's
    blocks padded to a whole number of SUBLANES. The combine's sum stores
    its staging tile eight sublanes at a time and the sparse-core gather
    moves rows as 32-bit words (four fp8 or two bf16 blocks to a word), so
    a width that is not a multiple of SUBLANES * LANES carries zero blocks
    to the next multiple; the sum writes only the true width."""
    return align_up(row_lane_blocks(hidden), SUBLANES)


def scale_table_rows(routed_rows):
    """Sublanes the activation row-scale table is handed to the kernel as.

    The table holds one 32-bit word per routed row, laid out as the dense
    [rows / LANES, LANES] view of the flat array (the same bytes, so the
    layer builds it at no cost), plus one row: a tile whose first row is
    not on a LANES boundary reads into the next sublane, which at the last
    tile is past the routed rows. Nothing selects from that row.
    """
    return align_up(routed_rows, LANES) // LANES + 1


def scale_window_rows(tile_rows):
    """Sublanes of the scale table one tile of `tile_rows` rows can touch."""
    return align_up(tile_rows, LANES) // LANES + 1


def power_of_two_shift(value, name):
    """The shift a division by `value` is. Raises ValueError for a value
    that is not a power of two."""
    if value <= 0 or value & (value - 1):
        raise ValueError(
            f"{name} is {value}, not a power of two. The routing "
            "tables write divisions by it as shifts"
        )
    return value.bit_length() - 1


ROW_BLOCK_SHIFT = power_of_two_shift(ROW_BLOCK, "ROW_BLOCK")
