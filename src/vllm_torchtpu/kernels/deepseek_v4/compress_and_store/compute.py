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
from jax.experimental import pallas as pl
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


def place_rope_words(kv_slots, words, rope_lane):
    """Every tile token's RoPE words, rotated to their window of its row.

    kv_slots: sequence of tile_n scalar ints. Returns one (row key, window,
    rotated words) per token; the row key is -1 for a token with no slot.
    Token ``j`` of a row owns words ``[32 j, 32 j + 32)`` (see
    `csa_cache_layout`), and word 0 of `words` (`rope_words_tiled`) is at lane
    `rope_lane`. The rotations do not depend on which row is being written,
    so they are done once per tile, ahead of `merge_rope_rows`.
    """
    per_row = csa_cache_layout.ROPE_TOKENS_PER_ROW
    width = csa_cache_layout.ROPE_WORDS
    lanes = words.shape[-1]
    # Powers of two: shifts and masks are floor `//` and `%`, with no sign fix-up.
    assert per_row & (per_row - 1) == 0 and lanes & (lanes - 1) == 0
    placed = []
    for i, slot in enumerate(kv_slots):
        key = jnp.where(slot >= 0, slot >> (per_row.bit_length() - 1), -1)
        pos = slot & (per_row - 1)
        rotation = (pos * width - rope_lane) & (lanes - 1)
        placed.append((key, pos, pltpu.roll(words[i], rotation, 1)))
    return placed


def merge_rope_rows(
    rows,  # (tile_n, 128) int32, each token's RoPE row as read from HBM
    placed,  # place_rope_words(...)
):
    """Writes every tile token's RoPE words into each row that holds it.

    Row n gets the words of every token whose row key equals token n's, in
    token order. All rows are merged at once, so only the rows of row-writer
    tokens are meaningful; the others are never written back.
    """
    width = csa_cache_layout.ROPE_WORDS
    window = jax.lax.broadcasted_iota(jnp.int32, (1, rows.shape[-1]), 1) >> (
        width.bit_length() - 1
    )
    row_keys = jnp.stack([key for key, _, _ in placed])[:, None]  # (tile_n, 1)
    for key, pos, words in placed:
        hit = (row_keys == key) & (key >= 0) & (window == pos)
        rows = jnp.where(hit, words, rows)
    return rows


def merge_slot_updates(
    slots_val,  # (pack_factor, physical_slot_size) uint8 (the row)
    kv_slots,  # (tile_n,) int (all slots in tile)
    val_padded,  # (tile_n, record_subslots, 128) uint8 (all values in tile)
    n,  # int (current token index)
):
    """multiple KV slots are packed into a single physical 512-byte row in HBM.

    Thus, different slots may map to the same tile.

    We only send the DMA from the first tile that updates a physical row,
    defined by `is_first_mask`.

    In this function, we:
    1. Read the current value of `slot_val`
    2. Scan all other tokens in the current tile
    3. Consolidate all other updates into the single tile.
    """
    tile_n = kv_slots.shape[0]
    curr_slot = kv_slots[n]
    pack_factor = slots_val.shape[0]

    curr_row = curr_slot // pack_factor

    slots_list = [slots_val[i] for i in range(pack_factor)]

    for i in range(tile_n):
        slot_i = kv_slots[i]
        valid_i = slot_i >= 0
        row_i = slot_i // pack_factor
        sub_idx = slot_i % pack_factor

        update_cond = valid_i & (row_i == curr_row) & (curr_slot >= 0)

        val = val_padded[i].flatten()

        for target_s in range(pack_factor):
            slots_list[target_s] = jax.lax.select(
                update_cond & (sub_idx == target_s),
                val,
                slots_list[target_s],
            )

    return jnp.stack(slots_list)


def stage_sal_columns(
    records_t_ref,  # (groups, 4, padded_tokens) uint8 VMEM, SEQ_ALONG_LANE
    records,  # (n, record_bytes) uint8, HEAD_ALONG_SUBLANE records
    first_lane,  # traced int >= 0
    lane_offsets,  # n traced ints: record i goes to lane first_lane + offset
    num_staged,  # traced int: the offsets >= 0 are exactly 0..num_staged-1
):
    """Copies record `i` into lane `first_lane + lane_offsets[i]`, if >= 0.

    The staged lanes `[first_lane, first_lane + num_staged)` may run into the
    next 128-lane tile.

    In the SEQ_ALONG_LANE layout (`streamindex_topk.KVLayout.SEQ_ALONG_LANE`)
    each token is one lane: sublane group `g` of lane `t` holds bytes
    `4g..4g+3` of token `t`'s record. Getting there from one-row-per-token
    records is a byte transpose, which is slow on TPU, so it is done with two
    matmuls against 0/1 matrices instead:

    1. `pick @ records^T` selects bytes: row `k * groups_pad + g` gets byte
       `4g + k` of every record.
    2. `@ place` moves each record's column to its target lane.

    The result is exact. Every input is either 0/1 or a byte value (0-255),
    which bf16 represents exactly, and each output element is a single
    nonzero product, so nothing is rounded.
    """
    n, width = records.shape
    groups = records_t_ref.shape[0]
    lanes = 128
    # TPU register tiles have 8 sublanes; pad both matmul dimensions to a
    # multiple of `sublane_tile`.
    sublane_tile = 8
    rows_pad = sublane_tile
    groups_pad = pl.cdiv(groups, sublane_tile) * sublane_tile
    assert n <= rows_pad and len(lane_offsets) == n, (n, rows_pad)
    padded_records = records.astype(jnp.int32)
    if n < rows_pad:
        padded_records = jnp.concatenate(
            [padded_records, jnp.zeros((rows_pad - n, width), jnp.int32)], axis=0
        )
    padded_records = padded_records.astype(jnp.bfloat16)  # (8, width)

    lane_offset = first_lane & (lanes - 1)
    token_idx = jax.lax.broadcasted_iota(jnp.int32, (rows_pad, lanes), 0)
    lane_idx = jax.lax.broadcasted_iota(jnp.int32, (rows_pad, lanes), 1)
    target_lane = jnp.full((rows_pad, lanes), -1, jnp.int32)
    for i, offset in enumerate(lane_offsets):
        lane_i = jnp.where(offset >= 0, (lane_offset + offset) & (lanes - 1), -1)
        target_lane = jnp.where(token_idx == i, lane_i, target_lane)
    place = (lane_idx == target_lane).astype(jnp.bfloat16)  # (8, 128)

    # Row `out_row = byte_in_word * groups_pad + group_idx` picks byte
    # `4 * group_idx + byte_in_word` (the 4 byte planes stacked).
    pick_shape = (4 * groups_pad, width)
    out_row = jax.lax.broadcasted_iota(jnp.int32, pick_shape, 0)
    byte_idx = jax.lax.broadcasted_iota(jnp.int32, pick_shape, 1)
    byte_in_word, group_idx = out_row // groups_pad, out_row % groups_pad
    pick = ((byte_idx == 4 * group_idx + byte_in_word) & (group_idx < groups)).astype(
        jnp.bfloat16
    )
    planes = jax.lax.dot_general(
        pick,
        padded_records,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )  # (4 * groups_pad, 8)
    placed = jnp.dot(
        planes.astype(jnp.bfloat16), place, preferred_element_type=jnp.float32
    ).astype(jnp.int32)  # (4 * groups_pad, 128)
    words = placed[:groups_pad]
    for byte_pos in range(1, 4):
        words = words | (
            placed[byte_pos * groups_pad : (byte_pos + 1) * groups_pad]
            << (8 * byte_pos)
        )
    words = pltpu.bitcast(words[:groups], jnp.uint32)  # (groups, 128)

    tile_lanes = jax.lax.broadcasted_iota(jnp.int32, (groups, lanes), 1)

    def update_tile(start_lane, mask):
        tile = records_t_ref.at[:, :, pl.ds(pl.multiple_of(start_lane, lanes), lanes)]
        current_words = pltpu.bitcast(tile[...].reshape(groups * 4, lanes), jnp.uint32)
        tile[...] = pltpu.bitcast(
            jnp.where(mask, words, current_words), jnp.uint8
        ).reshape(groups, 4, lanes)

    end = lane_offset + num_staged
    tile_start_lane = first_lane - lane_offset
    update_tile(tile_start_lane, (tile_lanes >= lane_offset) & (tile_lanes < end))

    @pl.when(end > lanes)
    def _next_tile():
        update_tile(tile_start_lane + lanes, tile_lanes < end - lanes)
