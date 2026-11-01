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

import functools
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    buffered_ref,
    compute,
    config,
)

# Boundary tokens per grid step. An HCA window is 512 KiB of state rows, a
# CSA one 64 KiB and an indexer one 16 KiB.
TILE_N = 16
HCA_TILE_N = 4


def _rows(ref, rows):
    return None if ref is None else ref.at[rows]


def fill_window(groups_vmem, group_slots, window_vmem, *, cfgs: config.Configs):
    """Copies a row tile's window states into the f32 window scratch.

    groups_vmem: the window buffer's groups of state rows, flat (see
    `buffered_ref.flat_groups`), as `buffered_ref.WindowGroupRef` fetched
    them: int32 words, or uint8 slabs whose (4, lanes) row holds byte s of
    f32 lane l at [s, l]; token n's group g is group_slots[n][g]. Every slice
    is static. window_vmem row n * head_tiles + h holds token n's head tile h.
    """
    group_positions = cfgs.group_positions
    groups = cfgs.window_groups
    field_rows = cfgs.field_rows
    head_tiles = cfgs.head_tiles
    is_words = csa_cache_layout.is_word_array(groups_vmem.dtype)

    def as_f32(rows):
        if is_words:
            return jax.lax.bitcast_convert_type(rows, jnp.float32)
        return pltpu.bitcast(rows, jnp.float32)

    for g in range(groups):
        # The first half of an overlapping window reads the prev half of the
        # state.
        is_prev = not cfgs.dims.overlap or g < groups // 2
        window_rows = pl.ds(g * group_positions, group_positions)
        for n in range(config.ROW_TILE):
            state_vmem = groups_vmem.at[group_slots[n][g]]
            if cfgs.dims.mode == config.Mode.CSA_INDEXER:
                # uint8 (4, 256) rows, kv then score; the halves are lanes.
                lanes = pl.ds(0 if is_prev else config.LANE, config.LANE)
                for field in range(config.N_FIELDS):
                    window_vmem[field, window_rows, pl.ds(n, 1)] = as_f32(
                        state_vmem[:, field, :, lanes]
                    )
                continue
            if cfgs.dims.overlap:
                kv_row = 0 if is_prev else field_rows
                score_row = (2 if is_prev else 3) * field_rows
            else:
                kv_row, score_row = 0, field_rows
            first = n * head_tiles
            for field, row in ((0, kv_row), (1, score_row)):
                if is_words:
                    rows = state_vmem[:, pl.ds(row, head_tiles)]
                    window_vmem[field, window_rows, pl.ds(first, head_tiles)] = as_f32(
                        rows
                    )
                    continue
                # A uint8 slab row is one f32 row.
                for h in range(head_tiles):
                    rows = state_vmem[:, row + h]
                    window_vmem[field, window_rows, pl.ds(first + h, 1)] = as_f32(rows)


def inner_kernel(
    cos_sin_vmem,
    out_vmem,
    state_vmem,
    rope_out_vmem,
    positions_ref,
    rms_weight_vmem_ref,
    window_vmem,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    whole_counts_ref,
    *,
    cfgs: config.Configs,
):
    tile_n = cfgs.tile_sizes.tile_n
    global_idx = pl.program_id(0) * tile_n
    compress = functools.partial(
        compress_tile,
        positions_ref=positions_ref,
        rms_weight_vmem_ref=rms_weight_vmem_ref,
        window_vmem=window_vmem,
        kv_slot_mapping_ref=kv_slot_mapping_ref,
        is_first_mask_ref=is_first_mask_ref,
        cfgs=cfgs,
    )

    row_tile = config.ROW_TILE

    groups_vmem = buffered_ref.flat_groups(state_vmem)

    def run(whole_rows):
        for r in range(tile_n // row_tile):
            rows = pl.ds(r * row_tile, row_tile)
            group_slots = buffered_ref.window_group_slots(cfgs, r, runs=whole_rows)
            fill_window(groups_vmem, group_slots, window_vmem, cfgs=cfgs)
            compress(
                _rows(cos_sin_vmem, rows),
                out_vmem.at[rows],
                _rows(rope_out_vmem, rows),
                global_idx=global_idx + r * row_tile,
                whole_rows=whole_rows,
            )

    buffered_ref.by_rows(
        cfgs,
        whole_counts_ref[pl.program_id(0)],
        functools.partial(run, whole_rows=True),
        functools.partial(run, whole_rows=False),
    )


def compress_tile(
    cos_sin_vmem,
    out_vmem,
    rope_out_vmem,
    *,
    positions_ref,
    rms_weight_vmem_ref,
    window_vmem,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    global_idx,
    whole_rows: bool,
    cfgs: config.Configs,
):
    """Compresses, normalizes, ropes and packs the tokens whose windows
    window_vmem holds; the other refs are views of those tokens' rows.

    whole_rows: the tokens fill one packed row in slot order, so the rows
    they merge into are a fixed placement of their records and were not
    read back.
    """
    tile_n = config.ROW_TILE
    window = cfgs.window
    head_tiles = cfgs.head_tiles

    # Per-token window start = position - window + 1
    start_list = []
    for i in range(tile_n):
        idx = global_idx + i
        safe_idx = jax.lax.select(idx < cfgs.dims.size_n, idx, 0)
        start_list.append(positions_ref[safe_idx] - window + 1)

    rms_weight_tiled = rms_weight_vmem_ref[...].astype(jnp.float32)  # (head_tiles, 128)

    # --- windowed softmax ---
    # (window, tile_n * head_tiles, 128): full vregs, one row per head tile.
    kv_val = window_vmem[0]
    scores_val = window_vmem[1]
    row_shape = (window, tile_n * head_tiles, 1)
    row_token = jax.lax.broadcasted_iota(jnp.int32, row_shape, 1) // head_tiles
    row_start = start_list[-1]
    for i in reversed(range(tile_n - 1)):
        row_start = jnp.where(row_token == i, start_list[i], row_start)
    curr_pos = row_start + jax.lax.broadcasted_iota(jnp.int32, row_shape, 0)
    mask = curr_pos >= 0
    mask_float = mask.astype(scores_val.dtype)
    neg_inf = jnp.array(-jnp.inf, dtype=scores_val.dtype)
    masked_scores = jnp.where(mask_float > 0.5, scores_val, neg_inf)
    weights = jax.nn.softmax(masked_scores, axis=0)
    kv_val = jnp.where(mask_float > 0.5, kv_val, 0.0)
    compressed = jnp.sum(weights * kv_val, axis=0)  # (tile_n * head_tiles, 128)
    compressed = compressed.reshape(tile_n, head_tiles, config.LANE)

    # --- rms norm ---
    variance = jnp.mean(jnp.square(compressed), axis=(1, 2), keepdims=True)
    normed = (
        compressed
        * jax.lax.rsqrt(variance + cfgs.dims.rms_eps)
        * rms_weight_tiled[None, :, :]
    )  # (tile_n, head_tiles, 128)

    # --- rope ---
    rope_ropped = None
    if cfgs.dims.has_rope:
        rope_slot = cfgs.rope_slot
        rope_val = normed[:, rope_slot : rope_slot + 1]

        cos_sin = cos_sin_vmem[...][:, None, :]
        cos_val = cos_sin[:, :, : cfgs.half_rope]
        sin_val = cos_sin[:, :, cfgs.half_rope :]

        rope_ropped = compute.interleaved_rope_vector(rope_val, cos_val, sin_val)
        if head_tiles > 1:
            normed = jnp.concatenate([normed[:, :rope_slot], rope_ropped], axis=1)
        else:
            normed = rope_ropped

    # --- pack + store ---
    if cfgs.dims.is_quantized:
        if cfgs.dims.mode == config.Mode.CSA:
            # CSA's reader wants the scales per lane, not per block.
            q, scale = compute.quantize_fp8_lane_periodic(
                normed, cfgs.dims.quant_block, cfgs.nope_store_dim
            )
            nope_val_padded = compute.pack_nope_lane_periodic(
                q,
                scale,
                cfgs.nope_store_dim,
                cfgs.record_bytes,
                cfgs.last_dim_size,
            )
        else:
            q, scale = compute.quantize_fp8_tiled(normed, cfgs.dims.quant_block)
            nope_val_padded = compute.pack_nope_tiled(
                q,
                scale,
                cfgs.nope_store_dim,
                cfgs.dims.quant_block,
                nope_width_bytes=cfgs.record_bytes,
                last_dim_size=cfgs.last_dim_size,
            )
        if cfgs.dims.mode == config.Mode.CSA_INDEXER:
            # out_vmem: (tile_n, 1, 4, 256) uint8, each token's whole row; a
            # row's sub-slot s is byte s of its words. Only the rows of
            # is_first_mask tokens are written back.
            records = nope_val_padded.astype(jnp.int32)  # (tile_n, 1, 256)
            if whole_rows:
                merged = compute.place_slot_row(records)
            else:
                kv_slots = [kv_slot_mapping_ref[global_idx + i] for i in range(tile_n)]
                rows = pltpu.bitcast(out_vmem[:, 0], jnp.int32)  # (tile_n, 1, 256)
                merged = compute.merge_slot_rows(rows, kv_slots, records)
            out_vmem[:, 0] = pltpu.bitcast(merged, jnp.uint8)

        else:
            out_vmem[:, 0] = nope_val_padded
        if cfgs.dims.has_rope_cache:
            # RoPE row of 4 tokens, token j in words [32 j, 32 j + 32) (see
            # `csa_cache_layout`); rope_out_vmem holds each row-writer's row.
            words = compute.rope_words_tiled(rope_ropped, cfgs.dims.rope_head_dim)
            rope_lane = config.LANE - cfgs.dims.rope_head_dim
            if whole_rows:
                merged = compute.place_rope_row(words, rope_lane)
            else:
                kv_slots = [kv_slot_mapping_ref[global_idx + i] for i in range(tile_n)]
                # Only the rows of is_first_mask_rope tokens are written back.
                rows = pltpu.bitcast(rope_out_vmem[...], jnp.int32)  # (tile_n, 1, 128)
                merged = compute.merge_rope_rows(rows, kv_slots, words, rope_lane)
            rope_out_vmem[...] = pltpu.bitcast(merged, jnp.uint8)

    else:
        # hca: bitcast bf16 -> uint8 and match the output block shape.
        out_vmem[...] = pltpu.bitcast(
            normed.astype(cfgs.dims.nope_dtype), jnp.uint8
        ).reshape(tile_n, cfgs.record_rows, cfgs.hbm_pack, 128)


def kernel_fn(
    # prefetched inputs (scalar)
    token_pages_ref,
    positions_ref,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    is_first_mask_rope_ref,
    valid_counts_ref,
    first_counts_ref,
    first_rope_counts_ref,
    whole_counts_ref,
    grid_size_ref,
    rms_weight_ref,
    # HBM inputs (dynamic access)
    cos_sin_cache_ref,
    cache_ref,
    rope_cache_ref,
    state_cache_ref,
    # outputs (aliased)
    _out_cache_ref,
    _out_rope_cache_ref,
    window_scratch_ref,
    *,
    cfgs: config.Configs,
):
    """Pallas kernel entry point."""
    grid_size = grid_size_ref[...]
    allocs, in_specs, pipeline_args = buffered_ref.create_allocs_and_specs(
        cfgs=cfgs,
        cache_ref=cache_ref,
        rope_cache_ref=rope_cache_ref,
        state_cache_ref=state_cache_ref,
        cos_sin_cache_ref=cos_sin_cache_ref,
        positions_ref=positions_ref,
        token_pages_ref=token_pages_ref,
        kv_slot_mapping_ref=kv_slot_mapping_ref,
        is_first_mask_ref=is_first_mask_ref,
        is_first_mask_rope_ref=is_first_mask_rope_ref,
        copy_counts_refs=(
            valid_counts_ref,
            first_counts_ref,
            first_rope_counts_ref,
            whole_counts_ref,
        ),
    )

    pipeline_func = pltpu.emit_pipeline(
        body=functools.partial(inner_kernel, cfgs=cfgs),
        grid=(grid_size,),
        in_specs=in_specs,
        out_specs=[],
    )

    @pl.with_scoped(allocations=tuple(allocs))
    def _run(allocations):
        pipeline_func(
            *pipeline_args,
            scratches=(
                positions_ref,
                rms_weight_ref,
                window_scratch_ref,
                kv_slot_mapping_ref,
                is_first_mask_ref,
                whole_counts_ref,
            ),
            allocations=allocations,
        )

    _run()


def _select_mode(head_dim: int, overlap: bool) -> config.Mode:
    return config.select_mode(head_dim, overlap)


def derive_aliases(
    has_rope: bool, has_rope_cache: bool, num_scalar_prefetch: int
) -> dict[int, int]:
    cache_index = num_scalar_prefetch + 1 + int(has_rope)
    aliases = {cache_index: 0}
    if has_rope_cache:
        aliases[cache_index + 1] = 1
    return aliases


def compute_is_first_mask(kv_slot_mapping, tile_n, pack_factor=4):
    """Determines, for every token in a sequence, whether it is the first token within
    its execution tile to map to a particular physical HBM row.

    If multiple tokens in the same tile map to the same row, only the first one is
    responsible for writing the merged VMEM row buffer back to HBM. The
    subsequent tokens in the same tile that conflict will skip the HBM write to
    prevent overwriting each other's data and reduce memory traffic.
    """
    num_tokens = kv_slot_mapping.shape[0]
    pad_len = (tile_n - (num_tokens % tile_n)) % tile_n
    if pad_len > 0:
        kv_slots_padded = jnp.pad(kv_slot_mapping, (0, pad_len), constant_values=-1)
    else:
        kv_slots_padded = kv_slot_mapping

    total_tokens = kv_slots_padded.shape[0]
    num_tiles = total_tokens // tile_n
    kv_slots_tiled = kv_slots_padded.reshape(num_tiles, tile_n)

    row_idxs = kv_slots_tiled // pack_factor
    valid = kv_slots_tiled >= 0

    eq = row_idxs[:, None, :] == row_idxs[:, :, None]
    tril = jnp.tril(jnp.ones((tile_n, tile_n), dtype=bool), k=-1)
    conflict = eq & tril[None, :, :] & valid[:, None, :]
    has_conflict = jnp.any(conflict, axis=-1)
    is_first = valid & ~has_conflict

    return is_first.flatten()[:num_tokens]


def compute_whole_row_mask(kv_slot_mapping, pack_factor=4):
    """Whether each token is token j of an aligned run of pack_factor tokens
    that fills one packed HBM row in slot order, as a prefill row tile does:
    sub-slot j, right after token j - 1."""
    j = jnp.arange(kv_slot_mapping.shape[0]) % pack_factor
    after_prev = kv_slot_mapping == jnp.roll(kv_slot_mapping, 1) + 1
    return (
        (kv_slot_mapping >= 0)
        & (kv_slot_mapping % pack_factor == j)
        & ((j == 0) | after_prev)
    )


def compute_window_run_mask(positions, token_to_req_indices, run_tokens, *, cfgs):
    """Whether each token is token j of an aligned run of run_tokens tokens
    whose `window_group_index` groups slide over one run of groups, the last
    run_tokens of them consecutive, as a prefill row tile's overlapping
    windows do: token j's groups are token j - 1's shifted by one group.

    Decided on the terms window_group_index derives groups from, not on the
    groups, so that it stays one elementwise pass over the tokens: token j's
    window ends one group after token j - 1's in the same request, neither
    is clamped at the request's start, and both newest groups are in one
    state block (one page, consecutive groups).
    """
    group_positions = cfgs.group_positions
    if group_positions != cfgs.dims.compress_ratio:
        return jnp.zeros(positions.shape, jnp.bool_)
    window_end = (positions // cfgs.dims.compress_ratio + 1) * cfgs.dims.compress_ratio
    prev_end = jnp.roll(window_end, 1)
    block_size = cfgs.state_block_size
    slides = (
        (token_to_req_indices == jnp.roll(token_to_req_indices, 1))
        & (window_end == prev_end + group_positions)
        & (prev_end >= cfgs.window)
        & (
            (window_end - group_positions) // block_size
            == (prev_end - group_positions) // block_size
        )
    )
    j = jnp.arange(positions.shape[0]) % run_tokens
    return (j == 0) | slides


def window_group_index(positions, rows, block_table, *, cfgs: config.Configs):
    """Each token's window groups, oldest first, as indices into the state rows
    viewed as (num_pages * groups_per_page) groups.

    Valid tokens are compression boundaries, so a window ends at a multiple of
    compress_ratio and its groups are aligned. Positions before the request's
    start are clamped to its first group and masked by the kernel.
    """
    ratio = cfgs.dims.compress_ratio
    group_positions = cfgs.group_positions
    window_end = (positions // ratio + 1) * ratio
    offsets = group_positions * jnp.arange(cfgs.window_groups, dtype=positions.dtype)
    starts = jnp.maximum(window_end[:, None] - cfgs.window + offsets[None, :], 0)
    block_size = cfgs.state_block_size
    pages = block_table[rows + starts // block_size]
    groups = pages * cfgs.groups_per_page + (starts % block_size) // group_positions
    return groups.reshape(-1)


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_table_stride",
        "state_block_size",
        "compress_ratio",
        "overlap",
        "quant_block",
        "rms_eps",
        "interpret",
        "name",
    ),
    donate_argnames=("cache", "rope_cache"),
)
def compress_norm_rope_store(
    cache: jax.Array,
    positions: jax.Array,
    block_table: jax.Array,  # [num_reqs * block_table_stride] int
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    rms_weight: jax.Array,
    *,
    block_table_stride: int,
    state_block_size: int,
    state_cache: jax.Array | None = None,
    rope_cache: jax.Array | None = None,
    cos_sin_cache: jax.Array | None = None,
    compress_ratio: int,
    overlap: bool,
    quant_block: int = 64,
    rms_eps: float = 1e-6,
    interpret: bool = False,
    name: str = "compress_norm_rope_store",
) -> tuple[jax.Array, jax.Array | None]:
    """Compresses, normalizes, applies RoPE and stores to cache."""
    assert block_table.ndim == 1
    assert block_table.shape[0] % block_table_stride == 0

    num_tokens = positions.shape[0]
    head_dim = rms_weight.shape[0]
    rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0

    state_operand = None if state_cache is None or state_cache is cache else state_cache
    state_source = cache if state_operand is None else state_operand

    mode = _select_mode(head_dim, overlap)
    cfgs = config.Configs.make(
        mode,
        size_n=num_tokens,
        physical_page_size=cache.shape[1],
        state_physical_page_size=state_source.shape[1],
        state_block_size=state_block_size,
        rms_eps=rms_eps,
        tile_n=HCA_TILE_N if mode is config.Mode.HCA else TILE_N,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
    )

    if cfgs.dims.mode in (config.Mode.CSA, config.Mode.CSA_INDEXER):
        assert cfgs.tile_sizes.tile_n % 4 == 0, (
            f"tile_n must be a multiple of 4 for {cfgs.dims.mode.value}, "
            f"got {cfgs.tile_sizes.tile_n}"
        )

    if cfgs.dims.has_rope_cache and rope_cache is None:
        raise ValueError("rope_cache must be provided when has_rope_cache is True")

    rms_weight_reshaped = rms_weight.reshape(cfgs.head_tiles, 128)

    # Compute grid size dynamically based on the maximum index in kv_slot_mapping.
    valid = kv_slot_mapping >= 0
    indices = jnp.arange(kv_slot_mapping.shape[0])
    max_idx = jnp.max(jnp.where(valid, indices, -1))
    grid_size = jnp.where(max_idx >= 0, pl.cdiv(max_idx + 1, cfgs.tile_sizes.tile_n), 0)

    is_first_mask = compute_is_first_mask(
        kv_slot_mapping,
        config.ROW_TILE,
        pack_factor=cfgs.tokens_in_second_minor,
    )
    is_first_mask_rope = compute_is_first_mask(
        kv_slot_mapping,
        config.ROW_TILE,
        # TODO: this is confusing, should probably find a better name for it
        pack_factor=cfgs.hbm_pack,
    )
    # The last grid step reads whole tiles of the per-token arrays.
    pad = (-num_tokens) % cfgs.tile_sizes.tile_n
    if pad:
        positions = jnp.pad(positions, (0, pad))
        token_to_req_indices = jnp.pad(token_to_req_indices, (0, pad))
        kv_slot_mapping = jnp.pad(kv_slot_mapping, (0, pad), constant_values=-1)
        is_first_mask = jnp.pad(is_first_mask, (0, pad))
        is_first_mask_rope = jnp.pad(is_first_mask_rope, (0, pad))
    rows = token_to_req_indices[:, None] * block_table_stride
    # Look up each token's window groups here so the kernel keeps only those
    # entries in SMEM; the full block table does not fit at long context lengths.
    token_pages = window_group_index(positions, rows, block_table, cfgs=cfgs)
    # A step whose tokens all fill whole rows merges them by a fixed
    # placement (see `compress_tile`) and fetches their windows as runs.
    whole_row_mask = jnp.zeros_like(is_first_mask)
    if cfgs.merges_rows:
        whole_row_mask = compute_whole_row_mask(
            kv_slot_mapping, config.ROW_TILE
        ) & compute_window_run_mask(
            positions, token_to_req_indices, config.ROW_TILE, cfgs=cfgs
        )
    # Each tile's record copies, by `buffered_ref.CopyCount`.
    copy_counts = (
        jnp.stack(
            [kv_slot_mapping >= 0, is_first_mask, is_first_mask_rope, whole_row_mask]
        )
        .reshape(len(buffered_ref.CopyCount), -1, cfgs.tile_sizes.tile_n)
        .sum(axis=-1, dtype=jnp.int32)
    )
    # Outer pallas_call operands, in call order. Optional operands are passed as
    # None to keep kernel_fn's argument positions fixed; pallas drops the Nones
    # when indexing, so alias indices count only the operands actually present.
    scalar_prefetch = (
        token_pages,
        positions,
        kv_slot_mapping,
        is_first_mask,
        is_first_mask_rope,
        *copy_counts,
        grid_size,
    )
    cos_sin_operand = cos_sin_cache if cfgs.dims.has_rope else None
    rope_operand = rope_cache if cfgs.dims.has_rope_cache else None

    in_specs = (
        pl.BlockSpec(memory_space=pltpu.VMEM),  # rms_weight
        (
            pl.BlockSpec(memory_space=pltpu.HBM) if cfgs.dims.has_rope else None
        ),  # cos_sin
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache
        (
            pl.BlockSpec(memory_space=pltpu.HBM) if cfgs.dims.has_rope_cache else None
        ),  # rope_cache
        (
            pl.BlockSpec(memory_space=pltpu.HBM) if state_operand is not None else None
        ),  # state_cache
    )
    out_specs = (
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache
        (
            pl.BlockSpec(memory_space=pltpu.HBM) if cfgs.dims.has_rope_cache else None
        ),  # rope_cache
    )
    out_shapes = (
        jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        jax.ShapeDtypeStruct(
            cast(jax.Array, rope_cache).shape, cast(jax.Array, rope_cache).dtype
        )
        if cfgs.dims.has_rope_cache
        else None,
    )

    aliases = derive_aliases(
        cfgs.dims.has_rope, cfgs.dims.has_rope_cache, len(scalar_prefetch)
    )

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=len(scalar_prefetch),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[pltpu.VMEM(cfgs.window_shape(), jnp.float32)],
    )

    out_cache, out_rope_cache = pl.pallas_call(
        functools.partial(kernel_fn, cfgs=cfgs),
        out_shape=out_shapes,
        grid_spec=grid_spec,
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=interpret,
        name=name,
    )(
        *scalar_prefetch,
        rms_weight_reshaped,
        cos_sin_operand,
        cache,
        rope_operand,
        state_operand,
    )

    return out_cache, out_rope_cache
