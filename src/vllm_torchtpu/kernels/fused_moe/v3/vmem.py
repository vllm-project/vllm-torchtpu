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
"""VMEM scratchpad allocation and alignment helpers for Fused EP MoE v3 kernels."""

from typing import Any

import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from . import device
from .config import RESULT_SLOTS_DEFAULT, WEIGHT_SLOTS_DEFAULT
from .formats import WeightFormat, weight_form
from .layout import (
    LANES,
    ROW_BLOCK,
    result_lane_blocks,
    row_lane_blocks,
    scale_window_rows,
)
from .rowquant import FP8

# The result scales travel one f32 word per row, packed ROW_BLOCK words to
# a [1, ROW_BLOCK] tile row per block of rows, so a block's scales move as
# one unit with the block's rows (the same block offsets), and the
# sparse-core gather reads a word per row.
SCALE_WORDS = ROW_BLOCK


def scratch_arrays(
    experts_per_shard,
    tile_rows,
    hidden,
    inter,
    *,
    weight_slots=WEIGHT_SLOTS_DEFAULT,
    result_slots=RESULT_SLOTS_DEFAULT,
    weight_format=WeightFormat.FP8,
    has_w1_bias=False,
    has_w2_bias=False,
    no_gate=False,
    activation_block=0,
    result_dtype=FP8,
    token_block=None,
):
    """The kernel's VMEM scratch, as (name, shape, dtype), in the order the
    kernel body unpacks it. activation_block: the row scales come in
    hidden // activation_block planes (one at 0), a window of each per
    tile. result_dtype: the result rows' element type (fp8 or bf16).

    Token rows and result rows are staged as [rows, lane blocks, LANES]: an
    eight-bit array's rows tile to 32 sublanes, so a flat [rows, hidden]
    buffer could only be sliced at multiples of 32 rows. Splitting a row
    across the two minor dimensions leaves the row axis untiled, which an
    8-row transport offset needs.
    """
    form = weight_form(weight_format)
    w1_cols = inter if no_gate else 2 * inter
    lane_blocks = row_lane_blocks(hidden)
    result_blocks = result_lane_blocks(hidden)
    tile_blocks = tile_rows // ROW_BLOCK
    # The token rows' scale planes: one per token block (the activation
    # block when the caller names none).
    planes_block = activation_block if token_block is None else token_block
    scale_planes = hidden // planes_block if planes_block else 1
    weights: list[tuple[str, tuple[int, ...], Any]] = [
        ("w1_vmem", (weight_slots, hidden, w1_cols), form.weight_dtype),
        ("w2_vmem", (weight_slots, inter, hidden), form.weight_dtype),
    ]
    if form.has_scales:
        weights += [
            ("w1_scales_vmem", (experts_per_shard, w1_cols), jnp.float32),
            ("w2_scales_vmem", (experts_per_shard, hidden), jnp.float32),
        ]
    biases = []
    if has_w1_bias:
        biases.append(("w1_bias_vmem", (experts_per_shard, w1_cols), jnp.float32))
    if has_w2_bias:
        biases.append(("w2_bias_vmem", (experts_per_shard, hidden), jnp.float32))
    return [
        ("rows_vmem", (result_slots, tile_rows, lane_blocks, LANES), FP8),
        *weights,
        *biases,
        (
            "row_scales_vmem",
            (result_slots, scale_planes, scale_window_rows(tile_rows), LANES),
            jnp.int32,
        ),
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
    ]


def tile_body_arrays(
    tile_rows,
    hidden,
    inter,
    *,
    weight_format=WeightFormat.FP8,
    no_gate=False,
):
    """One tile body's live values besides the scratch, as (name, shape,
    dtype): the first accumulator, the bf16 intermediate column slices,
    their fp8 requantization, the second accumulator, its bf16 rows and
    their fp8 form. Counted as though all were live at once, so the
    estimate is an upper bound."""
    form = weight_form(weight_format)
    w1_cols = inter if no_gate else 2 * inter
    arrays: list[tuple[str, tuple[int, ...], Any]] = [
        ("acc1", (tile_rows, w1_cols), jnp.float32),
        ("mid_column_slices", (tile_rows, inter), jnp.bfloat16),
        ("mid_fp8", (tile_rows, inter), FP8),
        ("acc2", (tile_rows, hidden), jnp.float32),
        ("down_bf16", (tile_rows, hidden), jnp.bfloat16),
        ("result_rows", (tile_rows, hidden), FP8),
    ]
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
    weight_slots=WEIGHT_SLOTS_DEFAULT,
    result_slots=RESULT_SLOTS_DEFAULT,
    weight_format=WeightFormat.FP8,
    has_w1_bias=False,
    has_w2_bias=False,
    no_gate=False,
    activation_block=0,
    result_dtype=FP8,
    info=None,
    token_block=None,
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
        weight_slots=weight_slots,
        result_slots=result_slots,
        weight_format=weight_format,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
        no_gate=no_gate,
        activation_block=activation_block,
        result_dtype=result_dtype,
        token_block=token_block,
    )
    arrays += tile_body_arrays(
        tile_rows, hidden, inter, weight_format=weight_format, no_gate=no_gate
    )
    return sum(
        device.array_vmem_bytes(shape, dtype, info) for _, shape, dtype in arrays
    )
