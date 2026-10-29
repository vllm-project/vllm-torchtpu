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
import math
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    buffered_ref,
    compute,
    config,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import kv_cache_append

# SEQ_ALONG_LANE store: `_write_runs` processes destination cache tiles in
# batches ("waves") of up to `_SAL_MAX_WAVE` tiles at a time -- issuing async
# HBM->VMEM reads into a `_SAL_MAX_WAVE`-slot VMEM buffer (`rmw_ref`), patching
# the staged lanes in VMEM, and DMAing the batch back to HBM.
_SAL_MAX_WAVE = 128

# Boundary tokens per grid step. A step's per-token work is short dependency
# chains, which more tokens fill better. CSA and the indexer keep it a multiple
# of 4: `prepare_boundary_batch` aligns each prefill 4-token cache row to 4, so
# a row is never split across steps. The indexer's SEQ_ALONG_LANE staging
# takes at most 8 (`compute.stage_sal_columns`). HCA has one boundary token per
# 128 and a 512 KiB window per token, so it keeps 4 and two steps to overlap
# the window fetch with compute.
CSA_TILE_N = 16
INDEXER_TILE_N = 8
HCA_TILE_N = 4
# Per-token inputs whose length is a multiple of this need no padding.
TOKEN_ALIGN = math.lcm(CSA_TILE_N, INDEXER_TILE_N, HCA_TILE_N)


def inner_kernel(
    cos_sin_vmem,
    out_vmem,
    window_vmem,
    rope_out_vmem,
    positions_ref,
    rms_weight_vmem_ref,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    is_first_mask_rope_ref,
    records_t_ref=None,
    staged_slots_ref=None,
    num_staged_ref=None,
    *,
    cfgs: config.Configs,
):
    tile_n = cfgs.tile_sizes.tile_n
    window = cfgs.window
    head_tiles = cfgs.head_tiles

    pid = pl.program_id(0)
    global_idx = pid * tile_n

    # Per-token window start = position - window + 1
    start_list = []
    for i in range(tile_n):
        idx = global_idx + i
        safe_idx = jax.lax.select(idx < cfgs.dims.size_n, idx, 0)
        start_list.append(positions_ref[safe_idx] - window + 1)
    rms_weight_tiled = rms_weight_vmem_ref[...].astype(jnp.float32)  # (head_tiles, 128)

    # --- windowed softmax ---
    # Window positions before 0 hold no state: give them a -inf score and a
    # zero kv. Only a token with a kv slot is ever stored, so the window is
    # masked only when one of those has such positions (a sequence's first
    # compressed tokens), and left as fetched otherwise. A slotless token's
    # window may then hold any bytes (NaN, Inf); this is safe only while every
    # step after the softmax stays per token.
    needs_mask = functools.reduce(
        jnp.logical_or,
        [
            (start < 0) & (kv_slot_mapping_ref[global_idx + i] >= 0)
            for i, start in enumerate(start_list)
        ],
    )

    @pl.when(needs_mask)
    def _mask_window():
        # The window packs window_tokens_per_vreg tokens per vreg (see
        # `config.Configs.window_shape`); give each row its token's start.
        group_rows = cfgs.window_tokens_per_vreg * head_tiles
        row_start = jnp.stack(
            [start_list[r // head_tiles] for r in range(tile_n * head_tiles)]
        ).reshape(-1, 1, group_rows, 1)
        curr_pos = row_start + jax.lax.broadcasted_iota(
            jnp.int32, (1, window, 1, 1), 1
        )
        mask = curr_pos >= 0  # (groups, window, group_rows, 1)
        neg_inf = jnp.array(-jnp.inf, dtype=window_vmem.dtype)
        window_vmem[:, :, 1] = jnp.where(mask, window_vmem[:, :, 1], neg_inf)
        window_vmem[:, :, 0] = jnp.where(mask, window_vmem[:, :, 0], 0.0)

    kv_val = window_vmem[:, :, 0]  # (groups, window, group_rows, 128)
    scores_val = window_vmem[:, :, 1]  # (groups, window, group_rows, 128)
    weights = jax.nn.softmax(scores_val, axis=1)
    compressed = jnp.sum(weights * kv_val, axis=1)  # (groups, group_rows, 128)
    compressed = compressed.reshape(tile_n, head_tiles, 128)

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
        if cfgs.dims.seq_along_lane:
            # Stage the records of the tokens with a kv slot as consecutive
            # lanes of a cache-shaped VMEM array, and their slots alongside;
            # the cache is written from them once the pipeline is done
            # (kernel_fn).
            first_lane = num_staged_ref[0]
            num_staged = first_lane
            lane_offsets = []
            for i in range(tile_n):
                slot = kv_slot_mapping_ref[global_idx + i]
                staged = slot >= 0
                # Branch-free: an unstaged slot is overwritten or left past
                # the end.
                staged_slots_ref[num_staged] = slot
                lane_offsets.append(jnp.where(staged, num_staged - first_lane, -1))
                num_staged = num_staged + staged.astype(jnp.int32)
            num_staged_ref[0] = num_staged
            compute.stage_sal_columns(
                records_t_ref,
                jnp.stack([nope_val_padded[n].flatten() for n in range(tile_n)]),
                first_lane,
                lane_offsets,
                num_staged - first_lane,
            )
        elif cfgs.dims.mode == config.Mode.CSA_INDEXER:
            kv_slots = []
            for i in range(tile_n):
                kv_slots.append(kv_slot_mapping_ref[global_idx + i])
            kv_slots = jnp.stack(kv_slots)

            for n in range(tile_n):
                is_first = is_first_mask_ref[global_idx + n]

                @pl.when(is_first)
                def _merge_nope():
                    # only applicable to csa-indexer
                    # out_vmem shape: (tile_n, 1, 4, 256)
                    slots_val = out_vmem[n, 0]
                    out_vmem[n, 0] = compute.merge_slot_updates(
                        slots_val,
                        kv_slots,
                        nope_val_padded,
                        n,
                    )

        else:
            out_vmem[:, 0] = nope_val_padded
        if cfgs.dims.has_rope_cache:
            # RoPE row of 4 tokens, token j in words [32 j, 32 j + 32) (see
            # `csa_cache_layout`); rope_out_vmem holds each row-writer's row.
            words = compute.rope_words_tiled(rope_ropped, cfgs.dims.rope_head_dim)
            rope_lane = config.LANE - cfgs.dims.rope_head_dim
            placed = compute.place_rope_words(
                [kv_slot_mapping_ref[global_idx + i] for i in range(tile_n)],
                words,
                rope_lane,
            )
            # Every token's row is merged; only the row writers' rows (those
            # `is_first_mask_rope` marks) were read and are written back.
            rows = pltpu.bitcast(rope_out_vmem[...], jnp.int32)  # (tile_n, 1, 128)
            rows = compute.merge_rope_rows(rows.reshape(tile_n, -1), placed)
            rope_out_vmem[...] = pltpu.bitcast(rows[:, None, :], jnp.uint8)

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
    *sal_scratch,
    cfgs: config.Configs,
):
    """Pallas kernel entry point.

    `sal_scratch` is only used when the indexer cache is SEQ_ALONG_LANE. It is
    then `(records_t, staged_slots, num_staged, run_src, run_dst, rmw, sems)`,
    and the cache is written in two phases:

    1. During the pipeline, each grid step copies the records of its tokens
       that have a kv slot into the next lanes of `records_t`, a VMEM buffer
       laid out like the cache, and their slots into `staged_slots`.
    2. After the pipeline, one pass copies `records_t` into the cache, one
       128-lane tile at a time. With no slot-less tokens between staged
       records, it moves none of them.

    The write can't happen inside the pipeline. One 128-lane tile holds tokens
    from many grid steps, so each step would have to read the tile, update it
    and write it back, while the pipeline may already be prefetching that same
    tile for a later step. The two would race.
    """
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
    )
    sal_staging = ()
    if cfgs.dims.seq_along_lane:
        records_t_ref, staged_slots_ref, num_staged_ref = sal_scratch[:3]
        num_staged_ref[0] = 0
        sal_staging = (records_t_ref, staged_slots_ref, num_staged_ref)

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
                kv_slot_mapping_ref,
                is_first_mask_ref,
                is_first_mask_rope_ref,
            )
            + sal_staging,
            allocations=allocations,
        )

    _run()

    if cfgs.dims.seq_along_lane:
        *_, run_src_ref, run_dst_ref, rmw_ref, sems = sal_scratch
        ops = kv_cache_append._SalTiles(
            records_t_ref, _out_cache_ref, rmw_ref, config.SAL_PAGE_SIZE
        )
        num_runs = kv_cache_append._find_runs(
            ops,
            staged_slots_ref,
            run_src_ref,
            run_dst_ref,
            num_rows=_out_cache_ref.shape[0] * config.SAL_PAGE_SIZE,
            num_tokens=num_staged_ref[0],
        )
        kv_cache_append._write_runs(
            ops,
            num_runs,
            run_src_ref,
            run_dst_ref,
            sems.at[0],
            sems.at[1],
            wave=rmw_ref.shape[0],
        )


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
    row_idxs = kv_slot_mapping // pack_factor
    valid = kv_slot_mapping >= 0
    pos_in_tile = jnp.arange(num_tokens) % tile_n

    # Compare each token with the up to tile_n - 1 tokens before it in its tile,
    # one static shift at a time, so it stays a single elementwise fusion.
    has_conflict = jnp.zeros((num_tokens,), jnp.bool_)
    for shift in range(1, min(tile_n, num_tokens)):
        earlier_row = jnp.pad(row_idxs[:-shift], (shift, 0))
        earlier_valid = jnp.pad(valid[:-shift], (shift, 0))
        has_conflict |= (
            (pos_in_tile >= shift) & earlier_valid & (earlier_row == row_idxs)
        )
    return valid & ~has_conflict


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
        "seq_along_lane",
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
    seq_along_lane: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Compresses, normalizes, applies RoPE and stores to cache.

    With `seq_along_lane` (CSA indexer only), `cache` is the SEQ_ALONG_LANE
    `[num_pages, SAL_SUBLANE_GROUPS, SLOT_PACK, SAL_PAGE_SIZE]` =
    `[num_pages, 33, 4, 256]` array (32 fp8 sublane groups + 1 scale group,
    4 bytes per group, 256 compressed tokens per page). Its f32 state is read
    through the same `(rows, 4, 256)` view as always; the records are staged in
    VMEM as cache lanes and stored after the pipeline (see `kernel_fn`).
    """
    assert block_table.ndim == 1
    assert block_table.shape[0] % block_table_stride == 0

    num_tokens = positions.shape[0]
    head_dim = rms_weight.shape[0]
    rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0

    state_operand = None if state_cache is None or state_cache is cache else state_cache
    state_source = cache if state_operand is None else state_operand

    mode = _select_mode(head_dim, overlap)
    tile_n = {
        config.Mode.CSA: CSA_TILE_N,
        config.Mode.CSA_INDEXER: INDEXER_TILE_N,
        config.Mode.HCA: HCA_TILE_N,
    }[mode]
    if pad := -num_tokens % tile_n:
        # Every grid step reads tile_n entries of each per-token input.
        positions = jnp.pad(positions, (0, pad))
        token_to_req_indices = jnp.pad(token_to_req_indices, (0, pad))
        kv_slot_mapping = jnp.pad(kv_slot_mapping, (0, pad), constant_values=-1)
        num_tokens += pad
    cfgs = config.Configs.make(
        mode,
        size_n=num_tokens,
        physical_page_size=cache.shape[1],
        state_physical_page_size=state_source.shape[1],
        state_block_size=state_block_size,
        rms_eps=rms_eps,
        tile_n=tile_n,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
        seq_along_lane=seq_along_lane,
    )
    if seq_along_lane and cache.shape[-1] != config.SAL_PAGE_SIZE:
        raise ValueError(
            f"SEQ_ALONG_LANE indexer cache {cache.shape} must have "
            f"{config.SAL_PAGE_SIZE} token lanes per page."
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

    # Only the row-merging writers read a mask (see
    # `buffered_ref.create_allocs_and_specs`): the HEAD_ALONG_SUBLANE indexer's
    # NoPE rows and CSA's RoPE rows. Elsewhere it is a placeholder, which also
    # keeps SMEM for SAL's run tables.
    is_first_mask = is_first_mask_rope = jnp.zeros((1,), jnp.bool_)
    if cfgs.dims.mode == config.Mode.CSA_INDEXER and not seq_along_lane:
        is_first_mask = compute_is_first_mask(
            kv_slot_mapping,
            cfgs.tile_sizes.tile_n,
            pack_factor=cfgs.tokens_in_second_minor,
        )
    if cfgs.dims.has_rope_cache:
        is_first_mask_rope = compute_is_first_mask(
            kv_slot_mapping,
            cfgs.tile_sizes.tile_n,
            # TODO: this is confusing, should probably find a better name for it
            pack_factor=cfgs.hbm_pack,
        )
    # Look up each token's state pages here so the kernel keeps only those
    # entries in SMEM; the full block table does not fit at long context lengths.
    pages_per_token = cfgs.pages_to_buffer_per_token
    offsets = jnp.arange(1 - pages_per_token, 1, dtype=positions.dtype)
    window_blocks = jnp.maximum(
        positions[:, None] // state_block_size + offsets[None, :], 0
    )
    rows = token_to_req_indices[:, None] * block_table_stride
    token_pages = block_table[rows + window_blocks].reshape(-1)
    # Outer pallas_call operands, in call order. Optional operands are passed as
    # None to keep kernel_fn's argument positions fixed; pallas drops the Nones
    # when indexing, so alias indices count only the operands actually present.
    scalar_prefetch = (
        token_pages,
        positions,
        kv_slot_mapping,
        is_first_mask,
        is_first_mask_rope,
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

    scratch_shapes = []
    if seq_along_lane:
        # See kernel_fn: staged records, their slots (one per token + one an
        # unstaged token may write) and count, the run tables of the final
        # store (one slot per token + a sentinel), its read-modify-write tiles.
        wave = max(1, min(num_tokens, _SAL_MAX_WAVE))
        scratch_shapes += [
            pltpu.VMEM(cfgs.records_t_shape(), jnp.uint8),
            pltpu.SMEM((num_tokens + 1,), jnp.int32),
            pltpu.SMEM((1,), jnp.int32),
            pltpu.SMEM((num_tokens + 1,), jnp.int32),
            pltpu.SMEM((num_tokens + 1,), jnp.int32),
            pltpu.VMEM(
                (wave, config.SAL_SUBLANE_GROUPS, config.SLOT_PACK, config.LANE),
                jnp.uint8,
            ),
            pltpu.SemaphoreType.DMA((2,)),
        ]

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=len(scalar_prefetch),
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch_shapes,
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
