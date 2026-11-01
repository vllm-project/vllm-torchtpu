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

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout


def interleaved_rope_vector(x, cos_val_32, sin_val_32):
    """Applies interleaved Rotary Position Embedding (RoPE) to a vector."""
    # x: (tile_n, 1, 128)
    # cos_val_32: (tile_n, 1, 32)
    # sin_val_32: (tile_n, 1, 32)
    tile_n = x.shape[0]

    # We work with 2D tensors for gather to keep dimensions simple.
    x_2d = jnp.squeeze(x, axis=1)  # (tile_n, 128)

    # swap adjacent pairs: [1, 0, 3, 2, ...]
    iota = jnp.arange(128)
    swap_indices = jnp.bitwise_xor(iota, 1)  # (128,)
    swap_coords = jnp.broadcast_to(swap_indices, (tile_n, 128))[
        :, :, None
    ]  # (tile_n, 128, 1)

    gather_dn = jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(1,),
        start_index_map=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
    )

    x_swapped_2d = jax.lax.gather(
        x_2d,
        swap_coords,
        dimension_numbers=gather_dn,
        slice_sizes=(1, 1),
        unique_indices=True,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )

    # cos_val_32/sin_val_32 are (tile_n, 1, 32). Squeeze to (tile_n, 32)
    cos_32 = jnp.squeeze(cos_val_32, axis=1).astype(x.dtype)
    sin_32 = jnp.squeeze(sin_val_32, axis=1).astype(x.dtype)

    ones_32 = jnp.ones((tile_n, 32), dtype=x.dtype)
    zeros_32 = jnp.zeros((tile_n, 32), dtype=x.dtype)

    # pad to pairs: (tile_n, 64)
    cos_pairs = jnp.concatenate([ones_32, cos_32], axis=-1)
    sin_pairs = jnp.concatenate([zeros_32, sin_32], axis=-1)

    # duplicate indices: [0, 0, 1, 1, 2, 2, ..., 63, 63]
    dup_indices = iota // 2
    dup_coords = jnp.broadcast_to(dup_indices, (tile_n, 128))[:, :, None]

    cos_dup_2d = jax.lax.gather(
        cos_pairs,
        dup_coords,
        dimension_numbers=gather_dn,
        slice_sizes=(1, 1),
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )

    sin_dup_2d = jax.lax.gather(
        sin_pairs,
        dup_coords,
        dimension_numbers=gather_dn,
        slice_sizes=(1, 1),
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )

    # alternate sin: [-1, 1, -1, 1, ...]
    alt_mask = ((iota % 2) * 2 - 1).astype(x.dtype)[None, :]  # (1, 128)
    sin_alt_2d = sin_dup_2d * alt_mask

    out_2d = x_2d * cos_dup_2d + x_swapped_2d * sin_alt_2d

    return out_2d[:, None, :]


def quantize_fp8_tiled(x, block_size):
    # x: (tile_n, S, 128) f32
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    tile_n, slots, width = x.shape  # (tile_n, S, 128) f32
    num_blocks = width // block_size

    qs = []
    scales = []
    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        x_block = x[:, :, start:end]  # (tile_n, S, block_size) f32

        amax = jnp.clip(
            jnp.max(jnp.abs(x_block), axis=-1, keepdims=True), 1e-4, None
        )  # (tile_n, S, 1) f32

        log2_val = jnp.log2(amax / fp8_max)  # f32
        scale = jnp.exp2(jnp.ceil(log2_val))  # (tile_n, S, 1) f32

        q_block = (x_block * (1.0 / scale)).astype(
            jnp.float8_e4m3fn
        )  # (tile_n, S, block_size) fp8

        qs.append(q_block)
        scales.append(scale)

    q = jnp.concatenate(qs, axis=-1)  # (tile_n, S, 128) fp8
    scale = jnp.concatenate(scales, axis=-1)  # (tile_n, S, num_blocks) f32

    # Bitcast workaround directly on f32 to extract exponent
    # f32 exponent is at bits 23-30. Shift right by 23 to align it.
    scale_u32 = pltpu.bitcast(scale, jnp.uint32)
    scale_exp = scale_u32 >> 23
    scale_u8 = scale_exp.astype(jnp.uint8)
    scale_f8 = pltpu.bitcast(scale_u8, jnp.float8_e8m0fnu)

    return q, scale_f8


def quantize_fp8_lane_periodic(x, period, nope_dim):
    """Quantize with one e8m0 scale per lane of a `period`-lane cycle.

    Group j is ``{d < nope_dim : d % period == j}``, so the scale a stored
    value needs is a function of its lane alone. The attention kernel then
    reads the scales as a tile it can rotate into place instead of expanding
    them with a one-hot matmul (see sparse_mla._dequant_dsv4_fp8).

    x: (tile_n, S, 128) f32, of which the first `nope_dim` flattened dims are
    the ones the record keeps.
    Returns (tile_n, S, 128) fp8 values and (tile_n, 1, period) e8m0 scales.
    """
    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    _, _, width = x.shape
    assert width == 2 * period, (width, period)

    dim = jax.lax.broadcasted_iota(
        jnp.int32, x.shape, 1
    ) * width + jax.lax.broadcasted_iota(jnp.int32, x.shape, 2)
    # Dims past the record's end are dropped by the packer; keeping them out
    # of the amax stops them widening a group they are not stored in.
    a = jnp.where(dim < nope_dim, jnp.abs(x), 0.0)
    folded = jnp.maximum(a[:, :, :period], a[:, :, period:])
    amax = jnp.clip(jnp.max(folded, axis=1, keepdims=True), 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))  # (tile_n, 1, period)

    q = (x * (1.0 / jnp.concatenate([scale, scale], axis=-1))).astype(jnp.float8_e4m3fn)
    scale_u8 = (pltpu.bitcast(scale, jnp.uint32) >> 23).astype(jnp.uint8)
    return q, pltpu.bitcast(scale_u8, jnp.float8_e8m0fnu)


def pack_nope_lane_periodic(q, scale, nope_dim, nope_width_bytes, last_dim_size):
    """Pack `nope_dim` fp8 values followed by the e8m0 scales."""
    tile_n = q.shape[0]
    q_nope = pltpu.bitcast(q, jnp.uint8).reshape(tile_n, -1)[:, :nope_dim]
    scale_bytes = pltpu.bitcast(scale, jnp.uint8).reshape(tile_n, -1)
    record = jnp.concatenate([q_nope, scale_bytes], axis=1)
    assert record.shape[1] == nope_width_bytes, record.shape
    return record.reshape(tile_n, -1, last_dim_size)


def pack_nope_tiled(
    q, scale, nope_dim, block_size, nope_width_bytes=512, last_dim_size=128
):
    # q: (tile_n, S, 128) fp8
    # scale: (tile_n, S, num_blocks) e8m0 (uint8 bitcasted)
    tile_n, _, _ = q.shape

    # Bitcast to uint8
    q_bytes = pltpu.bitcast(q, jnp.uint8)
    scale_bytes = pltpu.bitcast(scale, jnp.uint8)

    # Flat representations
    q_flat = q_bytes.reshape(tile_n, -1)  # (tile_n, S * 128)
    scale_flat = scale_bytes.reshape(tile_n, -1)  # (tile_n, S * num_blocks)

    # Select NOPE parts
    q_nope = q_flat[:, :nope_dim] if nope_dim < q_flat.shape[1] else q_flat

    nope_blocks = (nope_dim + block_size - 1) // block_size
    if nope_blocks < scale_flat.shape[1]:
        scale_nope = scale_flat[:, :nope_blocks]
    else:
        scale_nope = scale_flat

    # Pad with zeros
    pad_size = nope_width_bytes - (nope_dim + nope_blocks)
    zeros = jnp.zeros((tile_n, pad_size), dtype=jnp.uint8)

    nope_record_padded = jnp.concatenate(
        [q_nope, scale_nope, zeros], axis=1
    )  # (tile_n, 512)
    return nope_record_padded.reshape(tile_n, -1, last_dim_size)


def rope_words_tiled(rope_slot_ropped, rope_head_dim):
    """RoPE channels as cache words: low half bf16 ``k``, high half ``k + 32``.

    rope_slot_ropped: (tile_n, 1, 128) f32 holding the channels in its last
    ``rope_head_dim`` lanes. Returns (tile_n, 1, 128) int32 with word ``k`` at
    lane ``128 - rope_head_dim + k``; the lanes around it are junk.
    """
    bits = pltpu.bitcast(rope_slot_ropped.astype(jnp.bfloat16), jnp.uint16)
    bits = bits.astype(jnp.int32)
    half = rope_head_dim // 2
    # Lane c picks up channel c + half as its high half.
    return bits | (pltpu.roll(bits, bits.shape[-1] - half, 2) << 16)


def merge_rope_rows(
    rows,  # (tile_n, 1, 128) int32, each token's RoPE row as read from HBM
    kv_slots,  # tile_n scalar kv slots, one per token of the tile
    words,  # rope_words_tiled(...)
    rope_lane,  # int, lane of RoPE word 0 in `words`
):
    """Writes every tile token's RoPE words into each tile row that holds it.

    Token ``j`` of a row owns words ``[32 j, 32 j + 32)`` (see
    `csa_cache_layout`), so one rotation puts a token's words in its window;
    a later token of the tile wins. Rows of invalid slots come back merged
    with nothing.
    """
    per_row = csa_cache_layout.ROPE_TOKENS_PER_ROW
    width = csa_cache_layout.ROPE_WORDS
    lanes = rows.shape[-1]
    window = jax.lax.broadcasted_iota(jnp.int32, rows.shape[1:], 1) // width
    row_ids = [jnp.where(slot >= 0, slot // per_row, -1) for slot in kv_slots]

    placed, in_window = [], []
    for i, slot in enumerate(kv_slots):
        pos = slot % per_row
        placed.append(pltpu.roll(words[i], (pos * width - rope_lane) % lanes, 1))
        in_window.append(window == pos)

    merged = []
    for n in range(len(kv_slots)):
        row = rows[n]
        for i, slot in enumerate(kv_slots):
            hit = (slot >= 0) & (row_ids[i] == row_ids[n])
            row = jnp.where(hit & in_window[i], placed[i], row)
        merged.append(row)
    return jnp.stack(merged)


def merge_slot_rows(
    rows,  # (tile_n, 1, lanes) int32, each token's cache row as read from HBM
    kv_slots,  # tile_n scalar kv slots, one per token of the tile
    records,  # (tile_n, 1, lanes) int32, each token's record bytes, zero-extended
):
    """Writes every tile token's record into each tile row that holds it.

    A row packs 4 slots, slot ``s`` in byte ``s`` of every word, so a record
    replaces one byte of the row's words; a later token of the tile wins. Rows
    of invalid slots come back merged with nothing.
    """
    bits = jnp.iinfo(jnp.uint8).bits
    per_row = jnp.iinfo(jnp.int32).bits // bits
    row_ids = [jnp.where(slot >= 0, slot // per_row, -1) for slot in kv_slots]
    shifts = [(slot % per_row) * bits for slot in kv_slots]

    merged = []
    for n in range(len(kv_slots)):
        row = rows[n]
        for i, slot in enumerate(kv_slots):
            hit = (slot >= 0) & (row_ids[i] == row_ids[n])
            byte = jnp.left_shift(jnp.int32(0xFF), shifts[i])
            placed = (row & ~byte) | jnp.left_shift(records[i], shifts[i])
            row = jnp.where(hit, placed, row)
        merged.append(row)
    return jnp.stack(merged)
