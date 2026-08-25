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
"""Utility functions for Ragged Paged Attention."""

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


def align_to(a, b):
    """Returns 'a' aligned to 'b'."""
    return pl.cdiv(a, b) * b


def broadcast_minor(src, shape):
    """Broadcasts 'src' to 'shape' in the minor dimension."""
    if src.shape == shape:
        return src
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert src.shape[:-1] == shape[:-1]
    assert src.shape[-1] % num_lanes == 0
    target_minor = align_to(shape[-1], src.shape[-1])
    # no-op concatenation.
    broadcasted = jnp.tile(src, (target_minor // src.shape[-1], ))
    return broadcasted[..., :shape[-1]]


def get_dtype_packing(dtype):
    return 32 // jax.dtypes.itemsize_bits(dtype)


def kv_head_dim_fold(num_kv_heads: int, head_dim: int, kv_packing: int,
                     num_lanes: int) -> int:
    """How many packing slots one (K|V, head) vector occupies in the cache.

    HEAD_ALONG_SUBLANE stores one (K|V, head) vector per packing slot and pads
    `num_kv_heads * 2` up to a whole 32-bit word, so a rank holding a single KV
    head of an fp8 cache spends half of every token on structural zeros. The
    fold spends those slots on head_dim instead: the vector is cut into
    `fold` lane-wide segments that occupy `fold` consecutive slots.

    Returns 1 (no fold) unless every condition below holds, so the caller never
    has to know which parallelism it is running under:
      - there is padding to reclaim: `num_kv_heads * 2` is not already a whole
        number of words (under DP the rank keeps both KV heads and the fold
        would only move bytes around);
      - head_dim spans more than one lane register, so there is something to
        cut;
      - a packed word holds a whole number of (K|V, head) vectors, and a load
        of `packing // (2 * fold)` heads is a whole number of heads. The kernel
        de-interleaves one word at a time, so a vector split across two words
        -- or a word holding K without its V -- has no single-load form.
    """
    heads_x2 = num_kv_heads * 2
    if heads_x2 % kv_packing == 0:
        return 1
    fold = align_to(head_dim, num_lanes) // num_lanes
    if fold <= 1 or kv_packing % (2 * fold) != 0:
        return 1
    if num_kv_heads % (kv_packing // (2 * fold)) != 0:
        return 1
    return fold


def strided_load(ref, start_row, num_rows, step, *, dtype=None):
    """Loads data from HBM with strided access, handling 128-lane alignment."""
    _, row_width = ref.shape
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sub_lanes = row_width // num_lanes
    ref_flat = ref.reshape(-1, num_lanes)

    # scale indices to match flattened arraw.
    v_start = start_row * num_sub_lanes
    v_num = num_rows * num_sub_lanes
    v_step = step * num_sub_lanes

    # Gather the chunks into the original head dimension.
    chunks = [
        ref_flat[pl.ds(v_start + i, v_num // v_step, v_step)]
        for i in range(num_sub_lanes)
    ]
    vec = jnp.concat(chunks, axis=1)

    return pltpu.bitcast(vec, dtype) if dtype is not None else vec


def strided_store(ref, start, sz, step, val):
    """Stores data to HBM with strided access, handling 128-lane alignment."""
    assert get_dtype_packing(ref.dtype) == 1
    assert ref.dtype == val.dtype
    assert ref.shape == val.shape
    assert ref.ndim == 2
    rows, cols = ref.shape
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert cols % num_lanes == 0
    folds = cols // num_lanes
    ref = ref.reshape(rows * folds, num_lanes)
    start *= folds
    sz *= folds
    step *= folds
    assert sz % step == 0
    for i in range(folds):
        val_slice = val[:, i * num_lanes:(i + 1) * num_lanes]
        ref[pl.ds(start + i, sz // step, step)] = val_slice


def convert_to_target_bitwidth(val, target_bitwidth: int, kv_dtype: jnp.dtype):
    """Converts a value to a target bitwidth."""
    # If we want to convert 32-bits into 32//N number of N-bits value, naive
    # approach would be to perform 32//N number of 32-bits to N-bits conversion.
    # However, we can reduce number of instructions by utilizing binary tree.
    # 0: [32]
    # 1: [16, 16]
    # ...
    # log2(32//N): [N, N, ... N]

    curr_dtype = val.dtype
    curr_bitwidth = jax.dtypes.itemsize_bits(curr_dtype)
    assert target_bitwidth != curr_bitwidth, "No conversion is needed."

    # We split val into two vals (left and right) where each have half of the
    # original bitwidth.
    next_bitwidth = curr_bitwidth // 2
    next_dtype = jnp.dtype(f"uint{next_bitwidth}")

    left = val.astype(next_dtype)

    # Bitwise shift is only supported in uint32.
    val_u32 = pltpu.bitcast(val, jnp.uint32)
    val_u32_shifted = val_u32 >> next_bitwidth
    # Convert back to original dtype.
    val_shifted = pltpu.bitcast(val_u32_shifted, curr_dtype)
    right = val_shifted.astype(next_dtype)

    if next_bitwidth == target_bitwidth:
        k = pltpu.bitcast(left, kv_dtype)
        v = pltpu.bitcast(right, kv_dtype)
        return [(k, v)]
    else:
        left_out = convert_to_target_bitwidth(left,
                                              target_bitwidth=target_bitwidth,
                                              kv_dtype=kv_dtype)
        right_out = convert_to_target_bitwidth(right,
                                               target_bitwidth=target_bitwidth,
                                               kv_dtype=kv_dtype)
        return left_out + right_out


def has_bank_conflicts(stride: int, distance=24, num_banks=32) -> bool:
    banks = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False
