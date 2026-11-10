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

import os

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp


def bench_skip(stage: str) -> bool:
    """Whether TPU_RPA_BENCH_SKIP names `stage` ("matmul", "softmax", "kv_dma",
    "mask").

    The decode microbench (scripts/vllm/benchmark/rpa_decode_bench.py) sets it
    to leave a stage out and attribute the kernel's time; the output is wrong
    with any stage listed. Read at trace time. Empty: every stage runs.
    """
    return stage in os.environ.get("TPU_RPA_BENCH_SKIP", "").split(",")


def align_to(a, b):
    """Returns 'a' aligned to 'b'."""
    return pl.cdiv(a, b) * b


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


def broadcast_minor(src, shape):
    """Broadcasts 'src' to 'shape' in the minor dimension."""
    if src.shape == shape:
        return src
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert src.shape[:-1] == shape[:-1]
    assert src.shape[-1] % num_lanes == 0
    target_minor = align_to(shape[-1], src.shape[-1])
    # no-op concatenation.
    broadcasted = jnp.tile(src, (target_minor // src.shape[-1],))
    return broadcasted[..., : shape[-1]]


def get_dtype_packing(dtype):
    return 32 // jax.dtypes.itemsize_bits(dtype)


def get_dtype_bits(dtype):
    return jax.dtypes.itemsize_bits(dtype)


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
    assert ref.ndim == 2 and val.ndim == 2
    rows, cols = ref.shape
    assert val.shape == (sz // step, cols)
    assert start + sz <= rows
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert cols % num_lanes == 0
    folds = cols // num_lanes
    ref = ref.reshape(rows * folds, num_lanes)
    start *= folds
    sz *= folds
    step *= folds
    assert sz % step == 0
    for i in range(folds):
        val_slice = val[:, i * num_lanes : (i + 1) * num_lanes]
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
        left_out = convert_to_target_bitwidth(
            left, target_bitwidth=target_bitwidth, kv_dtype=kv_dtype
        )
        right_out = convert_to_target_bitwidth(
            right, target_bitwidth=target_bitwidth, kv_dtype=kv_dtype
        )
        return left_out + right_out


def has_bank_conflicts(stride: int, distance=24, num_banks=32) -> bool:
    banks = set()
    for i in range(distance):
        bank = (i * stride) % num_banks
        if bank in banks:
            return True
        banks.add(bank)
    return False


# ---------------------------------------------------------------------------
# Compact (lane-dense) LSE (TPU_PCP_LSE_COMPACT; ServingConfigs.lse_compact).
# The LSE of row (t, g) lives at flat index t * Gp + g of a [rows, 128] f32
# buffer, 128 values per row, instead of one lane-replicated 128-lane row per
# (t, g).


def lane_dense_rows(x: jax.Array) -> jax.Array:
    """(R, L) rows whose lanes all hold the row's value -> (ceil(R / L), L) lane-dense.

    Lane j of output row k is x[k * L + j, j], the diagonal of the k-th L x L
    block; with lane-replicated rows that is row k * L + j's value. R % L == 0,
    or R < L (padded with -inf).
    -inf stays -inf (the off-diagonal is replaced by 0 before the sum).
    """
    rows, lanes = x.shape
    if rows % lanes:
        if rows > lanes:
            raise ValueError(f"lane_dense_rows needs rows % {lanes} == 0 or rows < {lanes}, got {rows}")
        # A short block (e.g. the DECODE kernel's bq = 1): pad with -inf rows;
        # their lanes lie past the block's valid values.
        x = jnp.concatenate([x, jnp.full((lanes - rows, lanes), -jnp.inf, x.dtype)], axis=0)
        rows = lanes
    eye = jax.lax.broadcasted_iota(jnp.int32, (lanes, lanes), 0) == jax.lax.broadcasted_iota(
        jnp.int32, (lanes, lanes), 1
    )
    out = [
        jnp.sum(jnp.where(eye, x[k * lanes : (k + 1) * lanes, :], 0.0), axis=0, keepdims=True)
        for k in range(rows // lanes)
    ]
    return out[0] if len(out) == 1 else jnp.concatenate(out, axis=0)


def lane_dense_window_rows(k: int) -> int:
    """Rows of the aligned read-modify-write window ``place_lane_dense`` uses
    for a K-row block (the destination needs this many rows of headroom past
    the block's first row rounded down to 8)."""
    return align_to(k + 8, 8)


def place_lane_dense(dst_ref, src: jax.Array, flat_start, n_valid) -> None:
    """Write a lane-dense block into a lane-dense buffer at any flat offset.

    src (K, L): block-relative flat index j = r * L + l, values j < n_valid are
    written. dst_ref (rows, L) VMEM: they land at absolute flat index
    flat_start + j; every other value of dst_ref is preserved. The access is an
    8-row-aligned window of ``lane_dense_window_rows(K)`` rows at
    floor(flat_start / L) rounded down to 8, so dst_ref needs that headroom.
    """
    k, lanes = src.shape
    w = lane_dense_window_rows(k)
    flat_start = jnp.asarray(flat_start, jnp.int32)
    off = jax.lax.rem(flat_start, lanes)
    row0 = flat_start // lanes
    row0a = pl.multiple_of((row0 // 8) * 8, 8)
    sh = row0 - row0a
    # dr[r, l] = src[r, (l - off) mod L]
    dr = pltpu.roll(src, off, 1)
    fill = jnp.zeros((1, lanes), src.dtype)
    cur = jnp.concatenate([dr, fill], axis=0)
    prev = jnp.concatenate([fill, dr], axis=0)
    lane_k = jax.lax.broadcasted_iota(jnp.int32, (k + 1, lanes), 1)
    # row r' covers absolute row row0 + r': lanes >= off come from block row r',
    # lanes < off from block row r' - 1 (the carry).
    out = jnp.where(lane_k >= off, cur, prev)
    out = jnp.concatenate([out, jnp.zeros((w - k - 1, lanes), src.dtype)], axis=0)
    out = pltpu.roll(out, sh, 0)  # window row r_w holds out row r_w - sh
    r_w = jax.lax.broadcasted_iota(jnp.int32, (w, lanes), 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (w, lanes), 1)
    rel = r_w - sh
    j = rel * lanes + lane - off
    valid = (rel >= 0) & (j >= 0) & (j < n_valid)
    old = dst_ref[pl.ds(row0a, w), :]
    dst_ref[pl.ds(row0a, w), :] = jnp.where(valid, out, old)
