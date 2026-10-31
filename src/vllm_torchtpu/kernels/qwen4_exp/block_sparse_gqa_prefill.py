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
"""QSA block-sparse prefill with query-block reuse across paged KV tiles."""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import utils
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import (
    TC_METADATA_BITSET_TARGET_BYTES,
    TC_METADATA_MAX_BATCH_BLOCKS,
    TC_PACKING_TARGET_BYTES,
    TC_SELECTION_TARGET_BYTES,
    _paired_working_set,
    prefill_vmem_limit,
    supports_paired_prefill,
)
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16


def _kernel(
    table_ref,
    active_tiles_ref,
    active_counts_ref,
    last_page_ref,
    query_ref,
    bitset_ref,
    lengths_ref,
    cache_ref,
    output_ref,
    kv0_ref,
    kv1_ref,
    sem0_ref,
    sem1_ref,
    max_ref,
    mass_ref,
    accumulator_ref,
    *,
    ratio: int,
    kv_tile: int,
    physical_page_size: int,
    pages_per_query: int,
    num_kv_tiles: int,
    head_dim: int,
    sm_scale: float,
    packing: int,
    kv_heads: int,
    q_heads_per_kv: int,
    query_block: int,
    k_scale: float,
    v_scale: float,
):
    block = pl.program_id(0)
    # The model produces BF16 queries.  A BF16 call boundary can avoid a large
    # FP32 input window while preserving the existing FP32 QK accumulation.
    q = (
        query_ref[...]
        .astype(jnp.float32)
        .reshape(query_block, kv_heads, q_heads_per_kv, head_dim)
        .transpose(1, 2, 0, 3)
    )
    positions = lengths_ref[:, 0] - 1
    row = jnp.arange(kv_tile, dtype=jnp.int32)
    count = active_counts_ref[block]

    def active_tile(slot):
        slot = jnp.minimum(slot, num_kv_tiles - 1)
        return active_tiles_ref[block * num_kv_tiles + slot]

    def copy_tiles(tile, kv_ref, sem_ref):
        copies = []
        copy_rows = math.gcd(kv_tile, physical_page_size, ratio * 32)
        for offset in range(0, kv_tile, copy_rows):
            logical = tile * kv_tile + offset
            page = logical // physical_page_size
            in_page = logical % physical_page_size
            # Masked segments repeat an owned page's DMA address.
            address = block * pages_per_query + jnp.minimum(page, last_page_ref[block])
            physical = table_ref[address]
            destination = (
                kv_ref if copy_rows == kv_tile else kv_ref.at[pl.ds(offset, copy_rows)]
            )
            copies.append(
                pltpu.make_async_copy(
                    cache_ref.at[
                        pl.ds(physical * physical_page_size + in_page, copy_rows)
                    ],
                    destination,
                    sem_ref,
                )
            )
        return copies

    def start_tile(tile, kv_ref, sem_ref):
        for copy in copy_tiles(tile, kv_ref, sem_ref):
            copy.start()

    def wait_tile(tile, kv_ref, sem_ref):
        for copy in copy_tiles(tile, kv_ref, sem_ref):
            copy.wait()

    def split_kv(kv_ref):
        cache_words = kv_ref.bitcast(jnp.uint32).reshape(-1, head_dim)
        pairs = []
        head_axes = kv_ref.shape[1]
        for head_axis in range(head_axes):
            words = utils.strided_load(
                cache_words,
                head_axis,
                kv_tile * head_axes,
                head_axes,
            )
            pairs.extend(
                utils.convert_to_target_bitwidth(
                    words,
                    target_bitwidth=32 // packing,
                    kv_dtype=kv_ref.dtype,
                )
            )
        return pairs

    query_tile = min(query_block, 256)
    num_query_tiles = query_block // query_tile
    # Preserve -inf initialization and all-masked-row guards exactly.
    max_ref[...] = jnp.full(max_ref.shape, -jnp.inf, jnp.float32)
    mass_ref[...] = jnp.zeros(mass_ref.shape, jnp.float32)
    accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)

    def compute_tile(tile, kv_ref):
        logical = tile * kv_tile
        # Dynamic scalar indexing must address a native 128-wide row.  Keep
        # query bits on that row and slice the statically known live prefix.
        if kv_tile // ratio == 64:
            low = bitset_ref[0, 2 * tile, :query_block]
            high = bitset_ref[0, 2 * tile + 1, :query_block]
            # Select integer words before extracting bits. A select between
            # two bool matrices forces an unsupported Mosaic i1->i8 layout.
            word = jnp.where(row[None, :] < ratio * 32, low[:, None], high[:, None])
            visible = ((word >> ((row // ratio) % 32)[None, :]) & 1) != 0
        else:
            word = bitset_ref[0, tile, :query_block]
            group_in_tile = row // ratio
            visible = ((word[:, None] >> group_in_tile[None, :]) & 1) != 0
        visible &= logical + row[None, :] <= positions[:, None]
        written = logical + lax.broadcasted_iota(
            jnp.int32, (kv_tile, head_dim), 0
        ) < jnp.max(lengths_ref[:, 0])
        key_values = split_kv(kv_ref)[:kv_heads]
        for query_tile_index in range(num_query_tiles):
            start = query_tile_index * query_tile
            tile_visible = visible[start : start + query_tile]
            for head, (key, value) in enumerate(key_values):
                if kv_ref.dtype == jnp.float8_e4m3fn:
                    key = key.astype(jnp.float32)
                    value = value.astype(jnp.float32)
                    key = jnp.where(jnp.isfinite(key) & written, key, 0)
                    value = jnp.where(jnp.isfinite(value) & written, value, 0)
                else:
                    key = jnp.where(written, key, 0)
                    value = jnp.where(written, value, 0)
                m = max_ref[query_tile_index, head, :]
                mass = mass_ref[query_tile_index, head, :]
                acc = accumulator_ref[query_tile_index, head, :, :]
                rows = query_tile * q_heads_per_kv
                query = q[head, :, start : start + query_tile].reshape(rows, head_dim)
                score = lax.dot(
                    query,
                    key,
                    dimension_numbers=(([1], [1]), ([], [])),
                    preferred_element_type=jnp.float32,
                )
                row_visible = jnp.broadcast_to(
                    tile_visible[None, :, :],
                    (q_heads_per_kv, query_tile, kv_tile),
                ).reshape(rows, kv_tile)
                score = jnp.where(
                    row_visible,
                    score * (sm_scale * k_scale),
                    -jnp.inf,
                )
                next_m = jnp.maximum(m, jnp.max(score, axis=1))
                finite_m = jnp.where(jnp.isneginf(next_m), 0.0, next_m)
                prob = jnp.exp(score - finite_m[:, None])
                alpha = jnp.where(jnp.isneginf(next_m), 1.0, jnp.exp(m - next_m))
                weighted = lax.dot(
                    prob,
                    value,
                    dimension_numbers=(([1], [0]), ([], [])),
                    preferred_element_type=jnp.float32,
                )
                max_ref[query_tile_index, head, :] = next_m
                mass_ref[query_tile_index, head, :] = mass * alpha + jnp.sum(
                    prob, axis=1
                )
                accumulator_ref[query_tile_index, head, :, :] = (
                    acc * alpha[:, None] + weighted * v_scale
                )

    @pl.when(count > 0)
    def prefetch_first():
        start_tile(active_tile(0), kv0_ref, sem0_ref)

    @pl.when(count > 1)
    def prefetch_second():
        start_tile(active_tile(1), kv1_ref, sem1_ref)

    def pair_step(pair, _):
        slot = pair * 2
        tile = active_tile(slot)
        wait_tile(tile, kv0_ref, sem0_ref)
        compute_tile(tile, kv0_ref)

        @pl.when(slot + 2 < count)
        def prefetch_next_even():
            start_tile(active_tile(slot + 2), kv0_ref, sem0_ref)

        tile = active_tile(slot + 1)
        wait_tile(tile, kv1_ref, sem1_ref)
        compute_tile(tile, kv1_ref)

        @pl.when(slot + 3 < count)
        def prefetch_next_odd():
            start_tile(active_tile(slot + 3), kv1_ref, sem1_ref)

        return None

    lax.fori_loop(0, count // 2, pair_step, None)

    @pl.when(count % 2 != 0)
    def consume_odd():
        tile = active_tile(count - 1)
        wait_tile(tile, kv0_ref, sem0_ref)
        compute_tile(tile, kv0_ref)

    output_tiles = []
    for query_tile_index in range(num_query_tiles):
        heads = []
        for head in range(kv_heads):
            mass = mass_ref[query_tile_index, head, :]
            acc = accumulator_ref[query_tile_index, head, :, :]
            heads.append(
                (acc / jnp.where(mass == 0, 1, mass)[:, None])
                .reshape(q_heads_per_kv, query_tile, head_dim)
                .transpose(1, 0, 2)
            )
        output_tiles.append(jnp.concatenate(heads, axis=1))
    output_ref[...] = jnp.concatenate(output_tiles, axis=0).astype(output_ref.dtype)


def _selection_batch_rows(budget, groups, words):
    """Plan logical arrays; fusion, sort scratch and layouts affect allocations."""
    index_bytes = jnp.dtype(jnp.int32).itemsize
    mask_bytes = jnp.dtype(jnp.bool_).itemsize
    slots = budget + 1
    # Masked/sorted complete IDs, selected IDs with tail, and word indices.
    id_arrays_bytes = (budget + budget + slots + slots) * index_bytes
    scatter_indices_bytes = slots * 2 * index_bytes
    # Prefix, valid complete IDs, unique complete IDs, and selected IDs with tail.
    mask_arrays_bytes = (budget + budget + budget + slots) * mask_bytes
    mask_arrays_bytes += slots * jnp.dtype(jnp.uint32).itemsize
    packed_bitset_bytes = words * index_bytes
    # Complete count, tail ID, and row iota.
    row_scalars_bytes = 3 * index_bytes
    row_bytes = (
        id_arrays_bytes
        + scatter_indices_bytes
        + mask_arrays_bytes
        + packed_bitset_bytes
        + row_scalars_bytes
    )
    return max(1, TC_SELECTION_TARGET_BYTES // row_bytes)


def _selection_bitset(blocks, lengths, *, ratio, groups, words):
    rows, budget = blocks.shape
    if not rows or not groups or not words:
        return _selection_bitset_chunk(
            blocks, lengths, ratio=ratio, groups=groups, words=words
        )
    chunk = min(rows, _selection_batch_rows(budget, groups, words))
    if rows == chunk:
        return _selection_bitset_chunk(
            blocks, lengths, ratio=ratio, groups=groups, words=words
        )
    padding = (-rows) % chunk
    chunks = (rows + padding) // chunk
    bs = jnp.pad(blocks, ((0, padding), (0, 0)), constant_values=-1).reshape(
        chunks, chunk, budget
    )
    ls = jnp.pad(lengths, (0, padding)).reshape(chunks, chunk)
    bits = lax.map(
        lambda data: _selection_bitset_chunk(
            *data, ratio=ratio, groups=groups, words=words
        ),
        (bs, ls),
    )
    return bits.reshape(rows + padding, words)[:rows]


def _selection_bitset_chunk(blocks, lengths, *, ratio, groups, words):
    """Deduplicate complete groups before signed bitmask accumulation."""
    rows, slots = blocks.shape
    if not words or not groups or not rows:
        return jnp.zeros((rows, words), jnp.int32)
    complete = jnp.minimum(lengths // ratio, groups)
    tail = lengths // ratio
    prefix = jnp.arange(slots)[None, :] < jnp.minimum(complete[:, None], slots)
    valid = (
        prefix & (lengths[:, None] > 0) & (blocks >= 0) & (blocks < complete[:, None])
    )
    sentinel = jnp.iinfo(jnp.int32).max
    masked_blocks = jnp.where(valid, blocks, sentinel)
    ordered = lax.sort(masked_blocks, dimension=1, is_stable=False)
    unique = jnp.concatenate(
        (jnp.ones((rows, 1), dtype=jnp.bool_), ordered[:, 1:] != ordered[:, :-1]),
        axis=1,
    )[:, :slots]
    unique &= ordered != sentinel
    # Complete groups precede the implicit tail, so the tail is already unique.
    selected = jnp.concatenate((ordered, tail[:, None]), axis=1)
    selected_unique = jnp.concatenate(
        (
            unique,
            ((lengths > 0) & (lengths % ratio != 0) & (tail < groups))[:, None],
        ),
        axis=1,
    )
    word = jnp.where(selected_unique, selected // 32, words)
    row = jnp.broadcast_to(jnp.arange(rows)[:, None], selected.shape)
    indices = jnp.stack((row, word), axis=-1)
    masks = jnp.where(
        selected_unique,
        jnp.uint32(1) << (selected % 32).astype(jnp.uint32),
        jnp.uint32(0),
    )
    masks = lax.bitcast_convert_type(masks, jnp.int32)
    # Distinct bit positions sum to their OR; bit31 is one signed contribution.
    packed = lax.scatter_add(
        jnp.zeros((rows, words), jnp.int32),
        indices,
        masks,
        lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1),
        ),
        indices_are_sorted=False,
        unique_indices=False,
        mode="drop",
    )
    return packed


def _metadata_prefix_pages(
    tokens: int,
    selected_groups: int,
    ratio: int,
    page_size: int,
    full_pages: int,
    kv_tile: int,
) -> int:
    if min(tokens, selected_groups, ratio, page_size, full_pages, kv_tile) <= 0:
        return full_pages
    if kv_tile % (ratio * 32):
        return full_pages
    pages = min(
        full_pages, (max(tokens, selected_groups * ratio) + page_size - 1) // page_size
    )
    safe_tokens = pages * page_size // kv_tile * kv_tile
    if safe_tokens < ratio or safe_tokens > jnp.iinfo(jnp.int32).max:
        return full_pages
    return pages


def _metadata_prefix_fits(
    blocks: jax.Array,
    lengths: jax.Array,
    *,
    ratio: int,
    full_groups: int,
    prefix_groups: int,
) -> jax.Array:
    return jnp.all(
        (blocks < 0) | (blocks >= full_groups) | (blocks < prefix_groups)
    ) & jnp.all((lengths >= 0) & (lengths <= prefix_groups * ratio))


@functools.partial(
    jax.jit,
    static_argnames=(
        "ratio",
        "kv_heads",
        "query_block",
        "kv_tile",
        "k_scale",
        "v_scale",
        "bf16_io",
    ),
)
def block_sparse_gqa_prefill(
    query,
    blocks,
    lengths,
    page_table,
    cache,
    *,
    ratio: int,
    kv_heads: int,
    query_block: int = 32,
    kv_tile: int = 128,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    bf16_io: bool = False,
):
    def run(table):
        return _prefill(
            query,
            blocks,
            lengths,
            table,
            cache,
            None,
            ratio=ratio,
            kv_heads=kv_heads,
            query_block=query_block,
            kv_tile=kv_tile,
            k_scale=k_scale,
            v_scale=v_scale,
            bf16_io=bf16_io,
        )

    full_pages = page_table.shape[1]
    page_size = cache.shape[1]
    prefix_pages = _metadata_prefix_pages(
        len(query), blocks.shape[1], ratio, page_size, full_pages, kv_tile
    )
    if prefix_pages >= full_pages:
        return run(page_table)
    # A visited KV tile is DMAed in full, including its masked token rows.
    prefix_groups = prefix_pages * page_size // kv_tile * kv_tile // ratio
    return lax.cond(
        _metadata_prefix_fits(
            blocks,
            lengths,
            ratio=ratio,
            full_groups=full_pages * page_size // ratio,
            prefix_groups=prefix_groups,
        ),
        lambda _: run(page_table[:, :prefix_pages]),
        lambda _: run(page_table),
        None,
    )


def _prefill(
    query,
    blocks,
    lengths,
    page_table,
    cache,
    bitset,
    *,
    ratio,
    kv_heads,
    query_block,
    kv_tile,
    k_scale,
    v_scale,
    bf16_io,
):
    """Run exact selected-group attention with query-block KV reuse."""
    tokens, query_heads, head_dim = query.shape
    pages, physical_page_size, head_axis, packing, cache_dim = cache.shape
    if tokens % query_block or query_block % 8:
        raise ValueError("Token rows must form aligned query blocks")
    paired_pages = kv_tile == 256 and physical_page_size % 128 == 0 and ratio == 4
    if paired_pages:
        if not bf16_io or not supports_paired_prefill(
            query, cache, kv_heads=kv_heads, ratio=ratio, query_block=query_block
        ):
            raise ValueError(
                "Paired pages require BF16 queries, native BF16/FP8 KV, "
                "128-aligned pages/dimensions and sufficient VMEM"
            )
    elif (
        ratio <= 0
        or kv_tile != ratio * 32
        or math.gcd(physical_page_size, kv_tile) % 128
    ):
        raise ValueError(
            "Each KV tile must use one selection word or two mask segments"
        )
    if query_heads % kv_heads or cache_dim < head_dim:
        raise ValueError("QSA query and KV geometry is inconsistent")
    original_head_dim = head_dim
    query = jnp.pad(query, ((0, 0), (0, 0), (0, cache_dim - head_dim)))
    head_dim = cache_dim
    if packing != (2 if cache.dtype == jnp.bfloat16 else 4):
        raise ValueError("Cache packing does not match its dtype")
    vmem_limit = prefill_vmem_limit()
    if (
        _paired_working_set(
            query_heads,
            kv_heads,
            head_dim,
            packing,
            jax.dtypes.itemsize_bits(cache.dtype) // 8,
            query_block,
            kv_tile,
        )
        > vmem_limit
    ):
        raise ValueError("Sparse prefill tile exceeds available VMEM")

    groups = page_table.shape[1] * physical_page_size // ratio
    num_kv_tiles = (groups * ratio + kv_tile - 1) // kv_tile
    words_per_tile = kv_tile // (ratio * 32)
    words = ((num_kv_tiles * words_per_tile + 127) // 128) * 128
    nblocks = tokens // query_block
    bitset_row_bytes = words * jnp.dtype(jnp.int32).itemsize
    native_query_width = ((query_block + 127) // 128) * 128
    live = jnp.clip(
        lengths.reshape(nblocks, query_block).max(axis=1),
        0,
        page_table.shape[1] * physical_page_size,
    )
    last_page = jnp.maximum((live - 1) // physical_page_size, 0)
    if bitset is None and (
        (paired_pages or cache.dtype == jnp.bfloat16)
        and query_block == 256
        and tokens * bitset_row_bytes >= TC_METADATA_BITSET_TARGET_BYTES
    ):
        batch = TC_METADATA_MAX_BATCH_BLOCKS
        while (
            batch > 1
            and batch * query_block * bitset_row_bytes
            >= TC_METADATA_BITSET_TARGET_BYTES
        ):
            batch //= 2
        table, active_tiles, active_counts, block_bits_native = (
            _prepare_metadata_bounded(
                blocks,
                lengths,
                page_table,
                ratio=ratio,
                query_block=query_block,
                kv_tile=kv_tile,
                physical_page_size=physical_page_size,
                batch_size=batch,
            )
        )
    else:
        if bitset is None:
            bitset = _selection_bitset(
                blocks, lengths, ratio=ratio, groups=groups, words=words
            )
        table, active_tiles, active_counts, block_bits_native = _metadata_from_bits(
            bitset,
            page_table,
            query_block=query_block,
            kv_tile=kv_tile,
            ratio=ratio,
            physical_page_size=physical_page_size,
        )

    query_tile = min(query_block, 256)
    state_shape = (
        query_block // query_tile,
        kv_heads,
        query_tile * (query_heads // kv_heads),
    )
    score_elements = tokens * query_heads * kv_tile * num_kv_tiles
    alpha_elements = tokens * query_heads * num_kv_tiles
    output = pl.pallas_call(
        functools.partial(
            _kernel,
            ratio=ratio,
            kv_tile=kv_tile,
            physical_page_size=physical_page_size,
            pages_per_query=page_table.shape[1],
            num_kv_tiles=num_kv_tiles,
            head_dim=head_dim,
            sm_scale=original_head_dim**-0.5,
            packing=packing,
            kv_heads=kv_heads,
            q_heads_per_kv=query_heads // kv_heads,
            query_block=query_block,
            k_scale=k_scale,
            v_scale=v_scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(nblocks,),
            in_specs=[
                pl.BlockSpec(
                    (query_block, query_heads, head_dim),
                    lambda b, *_: (b, 0, 0),
                ),
                pl.BlockSpec(
                    (1, words, native_query_width),
                    lambda b, *_: (b, 0, 0),
                ),
                pl.BlockSpec((query_block, 1), lambda b, *_: (b, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (query_block, query_heads, head_dim),
                lambda b, *_: (b, 0, 0),
            ),
            scratch_shapes=[
                pltpu.VMEM((kv_tile, head_axis, packing, head_dim), cache.dtype),
                pltpu.VMEM((kv_tile, head_axis, packing, head_dim), cache.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM((*state_shape, head_dim), jnp.float32),
            ],
        ),
        cost_estimate=pl.CostEstimate(
            flops=4 * score_elements * head_dim,
            transcendentals=score_elements + alpha_elements,
            bytes_accessed=(
                2
                * query.size
                * jnp.dtype(jnp.bfloat16 if bf16_io else jnp.float32).itemsize
                + block_bits_native.nbytes
                + lengths.nbytes
                + table.nbytes
                + active_tiles.nbytes
                + active_counts.nbytes
                + last_page.nbytes
                + nblocks
                * num_kv_tiles
                * kv_tile
                * math.prod(cache.shape[2:])
                * jnp.dtype(cache.dtype).itemsize
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
        ),
        out_shape=jax.ShapeDtypeStruct(
            query.shape, jnp.bfloat16 if bf16_io else jnp.float32
        ),
        name=(
            "general_qsub256_state_ref_sparse_gqa_prefill_paired_pages"
            if paired_pages
            else "general_qsub256_state_ref_sparse_gqa_prefill"
        ),
    )(
        table.reshape(-1),
        active_tiles.reshape(-1),
        active_counts,
        last_page,
        query.astype(jnp.bfloat16 if bf16_io else jnp.float32),
        block_bits_native,
        lengths[:, None],
        cache.reshape(pages * physical_page_size, head_axis, packing, head_dim),
    )
    return output[:, :, :original_head_dim]


def _metadata_from_bits(
    bitset, page_table, *, query_block, kv_tile, ratio, physical_page_size
):
    tokens, words = bitset.shape
    nblocks = tokens // query_block
    num_kv_tiles = (page_table.shape[1] * physical_page_size + kv_tile - 1) // kv_tile
    block_bits = bitset.reshape(nblocks, query_block, words)
    if kv_tile == 256 and physical_page_size % 128 == 0 and ratio == 4:
        active = jnp.any(
            block_bits[:, :, : 2 * num_kv_tiles].reshape(
                nblocks, query_block, num_kv_tiles, 2
            )
            != 0,
            axis=(1, 3),
        )
    else:
        active = jnp.any(block_bits[:, :, :num_kv_tiles] != 0, axis=1)
    native_width = ((query_block + 127) // 128) * 128
    native_bits = jnp.pad(
        block_bits.transpose(0, 2, 1),
        ((0, 0), (0, 0), (0, native_width - query_block)),
    )
    tiles = jax.vmap(lambda row: jnp.nonzero(row, size=num_kv_tiles, fill_value=0)[0])(
        active
    ).astype(jnp.int32)
    counts = jnp.sum(active, axis=1, dtype=jnp.int32)
    table = page_table.reshape(nblocks, query_block, -1)[:, 0]
    return table, tiles, counts, native_bits


def _gate_output(output, gate, lengths):
    output = bf16(
        bf16(output).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
    )
    return jnp.where((lengths > 0)[:, None, None], output, 0)


def _request_tile_capacity(size: int, query_block: int, request_capacity: int) -> int:
    """Bound tiles in a partition with at most request_capacity nonempty pieces."""
    return min(
        size, (size + query_block - 1) // query_block + min(request_capacity, size) - 1
    )


def _ragged_plan(
    size,
    query_heads,
    head_dim,
    words,
    groups,
    columns,
    budget,
    query_block,
    *,
    request_capacity: int,
    query_dtype=jnp.bfloat16,
):
    """Plan packed arrays and compaction; gate temporaries are excluded."""
    if request_capacity <= 0:
        raise ValueError("Request capacity must be positive")
    index_bytes = jnp.dtype(jnp.int32).itemsize
    mask_bytes = jnp.dtype(jnp.bool_).itemsize
    query_bytes = query_heads * head_dim * jnp.dtype(query_dtype).itemsize
    result_bytes = query_heads * head_dim * jnp.dtype(jnp.bfloat16).itemsize
    result_compaction_bytes = result_bytes + result_bytes  # result, compact
    output_window_bytes = query_bytes + query_bytes  # old, window
    compaction_bytes = result_compaction_bytes + output_window_bytes
    # Source index/length and tile ID/first/end.
    source_metadata_bytes = 5 * index_bytes
    # Packed index, destination, bit-row index, and packed length.
    packed_metadata_bytes = 4 * index_bytes
    compaction_metadata_bytes = 2 * index_bytes  # result tile, result row
    row_masks_bytes = 3 * mask_bytes  # live, valid, current
    row_metadata_bytes = (
        source_metadata_bytes
        + packed_metadata_bytes
        + compaction_metadata_bytes
        + row_masks_bytes
    )
    pack_row_bytes = (
        max(query_bytes + result_bytes, compaction_bytes)
        + words * index_bytes  # source bits
        + words * index_bytes  # packed bits
        + columns * index_bytes
        + budget * index_bytes
        + row_metadata_bytes
    )
    # Each tile also has tile_slots, tile_first and tile_end descriptors.
    pack_tile_bytes = query_block * pack_row_bytes + 3 * index_bytes
    # Splitting a partition into n requests adds at most n-1 partial tiles.
    tile_capacity = _request_tile_capacity(size, query_block, request_capacity)
    pack_tiles = min(tile_capacity, max(1, TC_PACKING_TARGET_BYTES // pack_tile_bytes))
    # Selection scratch is row-chunked separately from its output bitsets.
    source_rows = min(size, pack_tiles * query_block)
    selection_rows = min(source_rows, _selection_batch_rows(budget, groups, words))
    return pack_tiles, source_rows, selection_rows


def _take_with_dynamic_fill(values, indices, fill_value):
    """Gather metadata with a dynamic out-of-bounds value."""
    selected = jnp.take(values, indices, mode="fill", fill_value=0)
    valid = (indices >= -len(values)) & (indices < len(values))
    return jnp.where(valid, selected, jnp.asarray(fill_value, values.dtype))


def _request_tiles(starts, offset, size, query_block):
    local = jnp.clip(starts, offset, offset + size)
    counts = (jnp.diff(local) + query_block - 1) // query_block
    cumulative = jnp.cumsum(counts)
    total = cumulative[-1]
    # Every nonempty tile contains at least one true query row.
    slots = jnp.arange(size, dtype=jnp.int32)
    requests = jnp.searchsorted(cumulative, slots, side="right")
    previous = jnp.concatenate((jnp.zeros(1, jnp.int32), cumulative))
    first = (
        _take_with_dynamic_fill(local, requests, offset + size)
        + (slots - jnp.take(previous, requests, mode="fill", fill_value=0))
        * query_block
    )
    ends = jnp.minimum(
        first + query_block,
        _take_with_dynamic_fill(local, requests + 1, offset + size),
    )
    return (
        jnp.where(slots < total, first, offset + size),
        jnp.where(slots < total, ends, offset + size),
        total,
    )


def _ragged_call_count(first, ends, total, source_rows, pack_tiles, limit):
    def group(state):
        cursor, calls = state
        stop = jnp.minimum(
            jnp.searchsorted(
                ends, jnp.minimum(first[cursor] + source_rows, limit), side="right"
            ),
            total,
        )
        return stop, calls + (stop - cursor + pack_tiles - 1) // pack_tiles

    return lax.while_loop(
        lambda state: state[0] < total, group, (jnp.int32(0), jnp.int32(0))
    )[1]


def _ragged_prefill(
    query,
    gate,
    blocks,
    lengths,
    page_table,
    cache,
    starts,
    output,
    offset,
    size,
    *,
    ratio,
    kv_heads,
    query_block,
    kv_tile,
    k_scale,
    v_scale,
    output_base=0,
):
    """Read global source rows and write output relative to output_base."""
    if size == 0 or len(output) == 0:
        return output
    first, ends, total = _request_tiles(starts, offset, size, query_block)
    groups = page_table.shape[1] * cache.shape[1] // ratio
    num_tiles = (page_table.shape[1] * cache.shape[1] + kv_tile - 1) // kv_tile
    words = ((num_tiles * (kv_tile // (ratio * 32)) + 127) // 128) * 128
    pack_tiles, source_rows, _ = _ragged_plan(
        size,
        query.shape[1],
        cache.shape[4],
        words,
        groups,
        page_table.shape[1],
        blocks.shape[1],
        query_block,
        request_capacity=len(starts) - 1,
        query_dtype=query.dtype,
    )

    def batch(state):
        cursor, output = state
        source_first = first[cursor]
        limit = jnp.minimum(source_first + source_rows, offset + size)
        stop = jnp.minimum(jnp.searchsorted(ends, limit, side="right"), total)
        source_end = ends[stop - 1]
        source_indices = source_first + jnp.arange(source_rows, dtype=jnp.int32)
        output_indices = source_indices - output_base
        read_indices = jnp.where(output_indices >= 0, output_indices, len(output))
        live = source_indices < source_end
        source_lengths = jnp.where(
            live, jnp.take(lengths, source_indices, mode="fill", fill_value=0), 0
        )
        source_blocks = jnp.where(
            live[:, None],
            jnp.take(blocks, source_indices, axis=0, mode="fill", fill_value=-1),
            -1,
        )
        bits = _selection_bitset(
            source_blocks, source_lengths, ratio=ratio, groups=groups, words=words
        )
        source_tile = jnp.searchsorted(ends, source_indices, side="right")
        source_tile_first = jnp.take(
            first, source_tile, mode="fill", fill_value=len(query)
        )
        source_tile_end = jnp.take(
            ends, source_tile, mode="fill", fill_value=len(query)
        )

        def pack(i, output):
            tile_slots = cursor + i * pack_tiles + jnp.arange(pack_tiles)
            tile_first = _take_with_dynamic_fill(first, tile_slots, offset + size)
            tile_end = _take_with_dynamic_fill(ends, tile_slots, offset + size)
            indices = tile_first[:, None] + jnp.arange(query_block)[None, :]
            valid = (tile_slots[:, None] < stop) & (indices < tile_end[:, None])
            # Positive out-of-bounds indices are dropped, including unused tail tiles.
            destination = jnp.where(valid, indices, len(query)).reshape(-1)
            bit_rows = jnp.where(valid, indices - source_first, source_rows).reshape(-1)
            packed_query = jnp.take(
                query, destination, axis=0, mode="fill", fill_value=0
            )
            packed_bits = jnp.take(bits, bit_rows, axis=0, mode="fill", fill_value=0)
            packed_lengths = jnp.take(
                source_lengths, bit_rows, mode="fill", fill_value=0
            )
            owners = jnp.take(page_table, tile_first, axis=0, mode="fill", fill_value=0)
            packed_table = jnp.broadcast_to(
                owners[:, None], (pack_tiles, query_block, page_table.shape[1])
            ).reshape(pack_tiles * query_block, -1)
            result = _prefill(
                packed_query,
                None,
                packed_lengths,
                packed_table,
                cache,
                packed_bits,
                ratio=ratio,
                kv_heads=kv_heads,
                query_block=query_block,
                kv_tile=kv_tile,
                k_scale=k_scale,
                v_scale=v_scale,
                bf16_io=True,
            )
            result_tile = source_tile - (cursor + i * pack_tiles)
            current = (
                live
                & (result_tile >= 0)
                & (result_tile < pack_tiles)
                & (source_tile < stop)
                & (source_indices >= source_tile_first)
                & (source_indices < source_tile_end)
            )
            result_row = jnp.where(
                current,
                result_tile * query_block + source_indices - source_tile_first,
                len(result),
            )
            compact = jnp.take(result, result_row, axis=0, mode="fill", fill_value=0)
            old = jnp.take(output, read_indices, axis=0, mode="fill", fill_value=0)
            window = jnp.where(current[:, None, None], compact, old)
            # Arange row indices and their full H/D windows are disjoint,
            # including distinct positive out-of-bounds rows.
            return output.at[output_indices].set(
                window,
                mode="drop",
                indices_are_sorted=True,
                unique_indices=True,
                wrap_negative_indices=False,
            )

        output = lax.fori_loop(
            0, (stop - cursor + pack_tiles - 1) // pack_tiles, pack, output
        )
        raw = jnp.take(output, read_indices, axis=0, mode="fill", fill_value=0)
        source_gate = jnp.take(gate, source_indices, axis=0, mode="fill", fill_value=0)
        gated = _gate_output(raw, source_gate, source_lengths)
        window = jnp.where(live[:, None, None], gated, raw)
        output = output.at[output_indices].set(
            window,
            mode="drop",
            indices_are_sorted=True,
            unique_indices=True,
            wrap_negative_indices=False,
        )
        return stop, output

    return lax.while_loop(
        lambda state: state[0] < total, batch, (jnp.int32(0), output)
    )[1]


def block_sparse_gqa_attention(
    query,
    gate,
    blocks,
    lengths,
    page_table,
    cache,
    starts,
    *,
    kv_heads: int,
    ratio: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    query_block: int = 32,
    kv_tile: int = 128,
):
    """Align requests, share KV tiles, then restore the original token order."""
    real = starts[1:] - starts[:-1]

    def run_aligned(_):
        raw = block_sparse_gqa_prefill(
            query,
            blocks,
            lengths,
            page_table,
            cache,
            ratio=ratio,
            kv_heads=kv_heads,
            query_block=query_block,
            kv_tile=kv_tile,
            k_scale=k_scale,
            v_scale=v_scale,
            bf16_io=True,
        )
        return _gate_output(raw, gate, lengths)

    def run_with_padding(_):
        return _ragged_prefill(
            query,
            gate,
            blocks,
            lengths,
            page_table,
            cache,
            starts,
            jnp.zeros_like(query),
            0,
            len(query),
            ratio=ratio,
            kv_heads=kv_heads,
            query_block=query_block,
            kv_tile=kv_tile,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # Both lax.cond branches are traced. A statically unaligned total cannot
    # enter run_aligned, whose core rejects its shape before execution.
    if len(query) % query_block:
        output = run_with_padding(None)
    else:
        output = lax.cond(
            # Only starts of nonempty requests must align. The final request
            # may end inside a tile whose remaining token rows are inactive.
            jnp.all((starts[:-1] % query_block == 0) | (real == 0)),
            run_aligned,
            run_with_padding,
            None,
        )
    return output, cache


@functools.partial(
    jax.jit,
    static_argnames=(
        "ratio",
        "query_block",
        "kv_tile",
        "physical_page_size",
        "batch_size",
    ),
)
def _prepare_metadata_bounded(
    blocks,
    lengths,
    page_table,
    *,
    ratio=4,
    query_block=256,
    kv_tile=256,
    physical_page_size: int,
    batch_size=1,
):
    tokens = blocks.shape[0]
    assert tokens % query_block == 0
    groups = page_table.shape[1] * physical_page_size // ratio
    num_kv_tiles = (groups * ratio + kv_tile - 1) // kv_tile
    words_per_tile = kv_tile // (ratio * 32)
    words = ((num_kv_tiles * words_per_tile + 127) // 128) * 128
    paired_pages = kv_tile == 256 and physical_page_size % 128 == 0 and ratio == 4
    native_query_width = ((query_block + 127) // 128) * 128
    nblocks = tokens // query_block

    batch_size = min(
        batch_size,
        max(1, _selection_batch_rows(blocks.shape[1], groups, words) // query_block),
    )

    def one(inputs):
        selected_blocks, live_lengths = inputs
        bits = _selection_bitset(
            selected_blocks, live_lengths, ratio=ratio, groups=groups, words=words
        )
        if paired_pages:
            active = jnp.any(
                bits[:, : 2 * num_kv_tiles].reshape(query_block, num_kv_tiles, 2) != 0,
                axis=(0, 2),
            )
        else:
            active = jnp.any(bits[:, :num_kv_tiles] != 0, axis=0)
        native_bits = jnp.pad(bits.T, ((0, 0), (0, native_query_width - query_block)))
        tiles = jnp.nonzero(active, size=num_kv_tiles, fill_value=0)[0].astype(
            jnp.int32
        )
        count = jnp.sum(active, dtype=jnp.int32)
        return tiles, count, native_bits

    active_tiles, active_counts, native_bits = lax.map(
        one,
        (
            blocks.reshape(nblocks, query_block, -1),
            lengths.reshape(nblocks, query_block),
        ),
        batch_size=batch_size,
    )
    table = page_table.reshape(nblocks, query_block, -1)[:, 0]
    return table, active_tiles, active_counts, native_bits
