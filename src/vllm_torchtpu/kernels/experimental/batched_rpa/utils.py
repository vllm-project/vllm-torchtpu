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
"""Utility functions for Batched Ragged Paged Attention."""

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import (
    align_to,
    broadcast_minor,
    get_dtype_bitwidth,
    get_dtype_packing,
    has_bank_conflicts,
    strided_load,
    strided_store,
)

get_dtype_bits = get_dtype_bitwidth

__all__ = [
    "align_span",
    "align_to",
    "broadcast_minor",
    "convert_to_target_bitwidth",
    "cp_local_cache_len",
    "get_dtype_bits",
    "get_dtype_packing",
    "has_bank_conflicts",
    "strided_load",
    "strided_store",
]


def align_span(start, size, align):
    """Returns (aligned_start, lead, aligned_size) for [start, start + size)."""
    # Grows the span out to 'align'; an 'align' of 1 is the identity.
    lead = start % align
    return start - lead, lead, align_to(lead + size, align)


def cp_local_cache_len(global_kv_cache_len, cp_group_size, cp_rank, page_size):
    """Number of cache tokens owned by this CP rank.

    Currently only support cp_kv_cache_interleave_size = page_size
    """
    super_page = cp_group_size * page_size
    full_tokens = (global_kv_cache_len // super_page) * page_size
    rem_tokens = jnp.maximum(
        0,
        jnp.minimum(
            page_size, (global_kv_cache_len % super_page) - cp_rank * page_size
        ),
    )
    return full_tokens + rem_tokens


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
    left_out = convert_to_target_bitwidth(
        left, target_bitwidth=target_bitwidth, kv_dtype=kv_dtype
    )
    right_out = convert_to_target_bitwidth(
        right, target_bitwidth=target_bitwidth, kv_dtype=kv_dtype
    )
    return left_out + right_out
