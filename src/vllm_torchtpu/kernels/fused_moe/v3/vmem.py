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
"""The FFN kernel's VMEM residents, and whether a build fits.

ffn_kernel.py declares its scratch from scratch_arrays and the estimate
sums the same list, so a buffer cannot be resized on one side only.
"""

from typing import Any

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3 import device
from vllm_torchtpu.kernels.fused_moe.v3.config import (
    RESULT_SLOTS_DEFAULT,
    WEIGHT_SLOTS_DEFAULT,
)
from vllm_torchtpu.kernels.fused_moe.v3.formats import (
    WEIGHT_BLOCK_DEFAULT,
    WeightFormat,
    weight_form,
)
from vllm_torchtpu.kernels.fused_moe.v3.layout import (
    FP4_PER_WORD,
    LANES,
    ROW_BLOCK,
    result_lane_blocks,
    row_lane_blocks,
    scale_window_rows,
)
from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8

# The result scales travel one f32 word per row, packed ROW_BLOCK words to
# a [1, ROW_BLOCK] tile row per block of rows, so a block's scales move as
# one unit with the block's rows (the same block offsets), and the
# sparse-core gather reads a word per row.
SCALE_WORDS = ROW_BLOCK
# A four-bit weight block is upcast to fp8 in VMEM, double-buffered.
WIDENED_BLOCK_BUFFERS = 2


# (name, shape, dtype) of one VMEM array.
ArraySpec = tuple[str, tuple[int, ...], Any]


def scratch_arrays(
    experts_per_shard,
    tile_rows,
    hidden,
    inter,
    *,
    intermediate="fp8",
    weight_slots=WEIGHT_SLOTS_DEFAULT,
    result_slots=RESULT_SLOTS_DEFAULT,
    weight_format=WeightFormat.FP8,
    weight_block=WEIGHT_BLOCK_DEFAULT,
    has_w1_bias=False,
    has_w2_bias=False,
    no_gate=False,
    activation_block=0,
    result_dtype=FP8,
    stream_block=0,
    stream_rows=0,
    token_block=None,
    token_dtype=FP8,
):
    """The kernel's VMEM scratch, as (name, shape, dtype), in the order the
    kernel body unpacks it. activation_block: the row scales come in
    hidden // activation_block planes (one at 0), a window of each per
    tile. result_dtype: the result rows' element type (fp8 or bf16).
    stream_block: a streamed build's weight slots hold column blocks of
    that many intermediate columns; its row buffers hold a group of
    stream_rows rows, double-buffered, with the group's float32 running
    sum last in the list.

    Token rows and result rows are staged as [rows, lane blocks, LANES]: an
    eight-bit array's rows tile to 32 sublanes, so a flat [rows, hidden]
    buffer could only be sliced at multiples of 32 rows. Splitting a row
    across the two minor dimensions leaves the row axis untiled, which an
    8-row transport offset needs.
    """
    form = weight_form(weight_format)
    slot_inter = stream_block or inter
    w1_cols = slot_inter if no_gate else 2 * slot_inter
    all_cols = inter if no_gate else 2 * inter
    lane_blocks = row_lane_blocks(hidden)
    result_blocks = result_lane_blocks(hidden)
    tile_blocks = tile_rows // ROW_BLOCK
    # The token rows' scale planes: one per token block (the activation
    # block when the caller names none).
    planes_block = activation_block if token_block is None else token_block
    scale_planes = hidden // planes_block if planes_block else 1
    # A streamed build's up-projection scales and bias travel with each
    # weight block into its slot; the down-projection scales and bias are
    # per expert over hidden and stay resident either way.
    if form.packed:
        # Four-bit weights stream as packed 32-bit words, half the bytes
        # of the fp8 form.
        weights: list[ArraySpec] = [
            ("w1_vmem", (weight_slots, hidden // FP4_PER_WORD, w1_cols), jnp.uint32),
            ("w2_vmem", (weight_slots, slot_inter // FP4_PER_WORD, hidden), jnp.uint32),
        ]
        if not stream_block:
            # The CURRENT expert's weights widened to fp8 once, when its
            # slot becomes current, so a tile's matmul reads fp8 like the
            # fp8 build's (widening per tile repeated the expert's whole
            # slab for every tile: 80 times a call at 8192 tokens).
            weights += [
                ("w1_fp8_vmem", (hidden, w1_cols), FP8),
                ("w2_fp8_vmem", (slot_inter, hidden), FP8),
            ]
    else:
        weights: list[ArraySpec] = [
            ("w1_vmem", (weight_slots, hidden, w1_cols), form.weight_dtype),
            ("w2_vmem", (weight_slots, slot_inter, hidden), form.weight_dtype),
        ]
        if intermediate == "bf16":
            # The down matmul's bf16 copy of the CURRENT expert's w2: one
            # buffer, filled after the expert's w2 lands and consumed by its
            # tiles (a copy per weight slot would not fit the 64 MiB VMEM
            # beside three fp8 slots).
            weights.append(("w2_bf16_vmem", (slot_inter, hidden), jnp.bfloat16))
        if (
            jnp.dtype(token_dtype) == jnp.dtype(jnp.bfloat16)
            and form.has_scales
            and not form.block_scaled
        ):
            # bf16 token rows against fp8 weights: the up matmul runs in
            # bf16 on the CURRENT expert's copy of w1 (whole-expert builds).
            weights.append(("w1_bf16_vmem", (hidden, w1_cols), jnp.bfloat16))
    if form.block_scaled:
        # The block scales are resident per local expert, or per slot when
        # streamed.
        if stream_block:
            weights += [
                (
                    "w1_scales_vmem",
                    (weight_slots, hidden // weight_block, w1_cols),
                    jnp.float32,
                ),
                (
                    "w2_scales_vmem",
                    (weight_slots, slot_inter // weight_block, hidden),
                    jnp.float32,
                ),
            ]
        else:
            weights += [
                (
                    "w1_scales_vmem",
                    (experts_per_shard, hidden // weight_block, all_cols),
                    jnp.float32,
                ),
                (
                    "w2_scales_vmem",
                    (experts_per_shard, inter // weight_block, hidden),
                    jnp.float32,
                ),
            ]
    elif form.has_scales:
        weights += [
            (
                "w1_scales_vmem",
                (weight_slots, 1, w1_cols)
                if stream_block
                else (experts_per_shard, all_cols),
                jnp.float32,
            ),
            ("w2_scales_vmem", (experts_per_shard, hidden), jnp.float32),
        ]
    biases = []
    if has_w1_bias:
        biases.append(
            (
                "w1_bias_vmem",
                (weight_slots, 1, w1_cols)
                if stream_block
                else (experts_per_shard, all_cols),
                jnp.float32,
            )
        )
    if has_w2_bias:
        biases.append(("w2_bias_vmem", (experts_per_shard, hidden), jnp.float32))
    if stream_block:
        tiles_per_group = stream_rows // tile_rows
        rows = [("rows_vmem", (2, stream_rows, lane_blocks, LANES), token_dtype)]
        row_scales = [
            (
                "row_scales_vmem",
                (2, tiles_per_group, scale_planes, scale_window_rows(tile_rows), LANES),
                jnp.int32,
            )
        ]
        running_sum = [("acc_vmem", (stream_rows, hidden), jnp.float32)]
    else:
        rows = [
            ("rows_vmem", (result_slots, tile_rows, lane_blocks, LANES), token_dtype)
        ]
        row_scales = [
            (
                "row_scales_vmem",
                (result_slots, scale_planes, scale_window_rows(tile_rows), LANES),
                jnp.int32,
            )
        ]
        running_sum = []
    return [
        *rows,
        *weights,
        *biases,
        *row_scales,
        (
            "result_rows_vmem",
            (result_slots, tile_rows, result_blocks, LANES),
            result_dtype,
        ),
        (
            "result_scales_vmem",
            (result_slots, tile_blocks, 1, SCALE_WORDS),
            jnp.float32,
        ),
        *running_sum,
    ]


def tile_body_arrays(
    tile_rows,
    hidden,
    inter,
    *,
    weight_format=WeightFormat.FP8,
    weight_block=WEIGHT_BLOCK_DEFAULT,
    no_gate=False,
    stream_block=0,
):
    """One tile body's live values besides the scratch, as (name, shape,
    dtype): the first accumulator, the bf16 intermediate column slices,
    their fp8
    requantization, the second accumulator, its bf16 rows and their fp8
    fp8 form, and for four-bit weights the upcast block, double-buffered.
    Counted as though all were live at once, so the estimate is an upper
    bound."""
    form = weight_form(weight_format)
    inter = stream_block or inter  # a streamed body holds one block's worth
    w1_cols = inter if no_gate else 2 * inter
    arrays: list[ArraySpec] = [
        ("acc1", (tile_rows, w1_cols), jnp.float32),
        ("mid_column_slices", (tile_rows, inter), jnp.bfloat16),
        ("mid_fp8", (tile_rows, inter), FP8),
        ("acc2", (tile_rows, hidden), jnp.float32),
        ("down_bf16", (tile_rows, hidden), jnp.bfloat16),
        ("result_rows", (tile_rows, hidden), FP8),
    ]
    if form.packed:
        arrays.append(
            (
                "upcast_block",
                (WIDENED_BLOCK_BUFFERS, weight_block, max(w1_cols, hidden)),
                FP8,
            )
        )
    if not form.has_scales:
        # bf16 weights contract the rows and the intermediate upcast to bf16.
        arrays += [
            ("rows_bf16", (tile_rows, hidden), jnp.bfloat16),
            ("mid_bf16", (tile_rows, inter), jnp.bfloat16),
        ]
    return arrays


def estimate_bytes(
    experts_per_shard,
    tile_rows,
    hidden,
    inter,
    *,
    intermediate="fp8",
    weight_slots=WEIGHT_SLOTS_DEFAULT,
    result_slots=RESULT_SLOTS_DEFAULT,
    weight_format=WeightFormat.FP8,
    weight_block=WEIGHT_BLOCK_DEFAULT,
    has_w1_bias=False,
    has_w2_bias=False,
    no_gate=False,
    activation_block=0,
    result_dtype=FP8,
    stream_block=0,
    stream_rows=0,
    info=None,
    token_block=None,
    token_dtype=FP8,
):
    """VMEM one built kernel occupies, an upper bound from the arrays it
    declares and the values its tile body holds. Reads the chip's lane
    count and sublane tiling off the device record."""
    if info is None:
        info = pltpu.get_tpu_info()
    arrays = scratch_arrays(
        experts_per_shard,
        tile_rows,
        hidden,
        inter,
        intermediate=intermediate,
        weight_slots=weight_slots,
        result_slots=result_slots,
        weight_format=weight_format,
        weight_block=weight_block,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        no_gate=no_gate,
        activation_block=activation_block,
        result_dtype=result_dtype,
        stream_block=stream_block,
        stream_rows=stream_rows,
        token_block=token_block,
        token_dtype=token_dtype,
    )
    arrays += tile_body_arrays(
        tile_rows,
        hidden,
        inter,
        weight_format=weight_format,
        weight_block=weight_block,
        no_gate=no_gate,
        stream_block=stream_block,
    )
    return sum(
        device.array_vmem_bytes(shape, dtype, info) for _, shape, dtype in arrays
    )
