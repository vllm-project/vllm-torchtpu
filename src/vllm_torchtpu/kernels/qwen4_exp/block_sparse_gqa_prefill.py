# SPDX-License-Identifier: Apache-2.0
"""QSA block-sparse prefill with query-block reuse across paged KV tiles."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import utils
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import PAIRED_PAGE_VMEM_MIB
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16


def _kernel(
    table_ref,
    active_tiles_ref,
    active_counts_ref,
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
        # One mask word covers ratio * 32 tokens. Split there because a
        # shared pool page may contain an odd number of these segments.
        copy_rows = min(kv_tile, physical_page_size, ratio * 32)
        for offset in range(0, kv_tile, copy_rows):
            logical = tile * kv_tile + offset
            page = logical // physical_page_size
            in_page = logical % physical_page_size
            # An odd final logical page has no second half. Its selection
            # word is zero; repeat a valid DMA address without exposing it.
            address = block * pages_per_query + jnp.minimum(page, pages_per_query - 1)
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

    # B64 is the largest production query tile that fits the full-model v7
    # scoped-VMEM limit while retaining FP32 online-softmax state. Larger
    # query blocks reuse one resident KV tile across multiple B64 subtiles.
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
    """Run exact selected-group attention with query-block KV reuse."""
    tokens, query_heads, head_dim = query.shape
    pages, physical_page_size, head_axis, packing, cache_dim = cache.shape
    if tokens % query_block or query_block % 8:
        raise ValueError("Token rows must form aligned query blocks")
    paired_pages = kv_tile == 256 and physical_page_size % 128 == 0 and ratio == 4
    if paired_pages:
        if (
            query_heads != 24
            or kv_heads != 2
            or head_dim != 256
            or cache.dtype != jnp.float8_e4m3fn
            or query_block not in PAIRED_PAGE_VMEM_MIB
            or not bf16_io
        ):
            raise ValueError("Paired pages require BF16 Q24/KV2/D256 with FP8 cache")
    elif kv_tile // ratio != 32 or max(physical_page_size, kv_tile) % min(
        physical_page_size, kv_tile
    ):
        raise ValueError(
            "Each KV tile must use one selection word or two FP8 mask segments"
        )
    if query_heads % kv_heads or cache_dim < head_dim:
        raise ValueError("QSA query and KV geometry is inconsistent")
    original_head_dim = head_dim
    query = jnp.pad(query, ((0, 0), (0, 0), (0, cache_dim - head_dim)))
    head_dim = cache_dim
    if packing != (2 if cache.dtype == jnp.bfloat16 else 4):
        raise ValueError("Cache packing does not match its dtype")

    groups = page_table.shape[1] * physical_page_size // ratio
    num_kv_tiles = (groups * ratio + kv_tile - 1) // kv_tile
    words_per_tile = kv_tile // (ratio * 32)
    words = ((num_kv_tiles * words_per_tile + 127) // 128) * 128
    nblocks = tokens // query_block
    native_query_width = ((query_block + 127) // 128) * 128
    if paired_pages and query_block == 256 and tokens * words * 4 >= 16 * 1024**2:
        batch = 8
        while batch > 1 and batch * query_block * words * 4 >= 16 * 1024**2:
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
        tail = lengths // ratio
        selected = jnp.concatenate((blocks, tail[:, None]), axis=1)
        valid = jnp.concatenate(
            (
                (blocks >= 0) & (blocks < groups),
                ((lengths > 0) & (lengths % ratio != 0) & (tail < groups))[:, None],
            ),
            axis=1,
        )
        invalid = words + jnp.arange(blocks.shape[1] + 1, dtype=jnp.int32)
        indices = jnp.where(valid, selected // 32, invalid[None, :])
        weights = jnp.where(valid, jnp.int32(1) << (selected % 32), 0)
        bitset = jax.vmap(
            lambda row, weight: (
                jnp.zeros((words,), jnp.int32).at[row].add(weight, mode="drop")
            )
        )(indices, weights)

        nblocks = tokens // query_block
        block_bits = bitset.reshape(nblocks, query_block, words)
        if paired_pages:
            active = jnp.any(
                block_bits[:, :, : 2 * num_kv_tiles].reshape(
                    nblocks, query_block, num_kv_tiles, 2
                )
                != 0,
                axis=(1, 3),
            )
        else:
            active = jnp.any(block_bits[:, :, :num_kv_tiles] != 0, axis=1)
        native_query_width = ((query_block + 127) // 128) * 128
        block_bits_native = jnp.pad(
            block_bits.transpose(0, 2, 1),
            ((0, 0), (0, 0), (0, native_query_width - query_block)),
        )
        active_tiles = jax.vmap(
            lambda row: jnp.nonzero(row, size=num_kv_tiles, fill_value=0)[0]
        )(active).astype(jnp.int32)
        active_counts = jnp.sum(active, axis=1, dtype=jnp.int32)
        table = page_table.reshape(nblocks, query_block, -1)[:, 0]

    query_tile = min(query_block, 256)
    state_shape = (
        query_block // query_tile,
        kv_heads,
        query_tile * (query_heads // kv_heads),
    )
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
            num_scalar_prefetch=3,
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
        # B128 retains two B64 FP32 online-softmax states so one resident KV
        # tile can serve twice as many queries.  The compiled production
        # kernel needs about 40.44 MB at the full T8192 production shape, just
        # above the effective limit produced by a 36 MiB setting.
        # Paired pages need 61.13 MiB at Q256 on v7; the tuned allowance
        # remains below its 64 MiB per core. Preserve the KV128 budgets.
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=(
                PAIRED_PAGE_VMEM_MIB[query_block]
                if paired_pages
                else 61
                if query_block > 128
                else 38
                if query_block > 64
                else 32
            )
            * 1024**2,
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
        query.astype(jnp.bfloat16 if bf16_io else jnp.float32),
        block_bits_native,
        lengths[:, None],
        cache.reshape(pages * physical_page_size, head_axis, packing, head_dim),
    )
    return output[:, :, :original_head_dim]


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
        return block_sparse_gqa_prefill(
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

    def run_with_padding(_):
        padded = ((real + query_block - 1) // query_block) * query_block
        offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(padded)))
        max_tokens = (
            (len(query) + (query_block - 1) * (len(starts) - 1) + query_block - 1)
            // query_block
            * query_block
        )
        req = jnp.searchsorted(starts[1:], jnp.arange(len(query)), side="right")
        req = jnp.minimum(req, len(real) - 1)
        positions = jnp.arange(len(query)) - starts[req]
        valid = jnp.arange(len(query)) < starts[-1]
        slots = jnp.where(valid, offsets[req] + positions, max_tokens)

        def align(array, fill):
            shape = (max_tokens, *array.shape[1:])
            return jnp.full(shape, fill, array.dtype).at[slots].set(array, mode="drop")

        output = block_sparse_gqa_prefill(
            align(query, 0),
            align(blocks, -1),
            align(lengths, 0),
            align(page_table, 0),
            cache,
            ratio=ratio,
            kv_heads=kv_heads,
            query_block=query_block,
            kv_tile=kv_tile,
            k_scale=k_scale,
            v_scale=v_scale,
            bf16_io=True,
        )
        return output[jnp.minimum(slots, max_tokens - 1)]

    # Both lax.cond branches are traced. A statically unaligned total cannot
    # enter run_aligned, whose core rejects its shape before execution.
    if len(query) % query_block:
        output = run_with_padding(None)
    else:
        output = lax.cond(
            jnp.all(real % query_block == 0),
            run_aligned,
            run_with_padding,
            None,
        )
    output = bf16(
        bf16(output).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
    )
    return jnp.where((lengths > 0)[:, None, None], output, 0), cache


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
    physical_page_size=1664,
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

    def one(inputs):
        selected_blocks, live_lengths = inputs
        tail = live_lengths // ratio
        selected = jnp.concatenate((selected_blocks, tail[:, None]), axis=1)
        valid = jnp.concatenate(
            (
                (selected_blocks >= 0) & (selected_blocks < groups),
                ((live_lengths > 0) & (live_lengths % ratio != 0) & (tail < groups))[
                    :, None
                ],
            ),
            axis=1,
        )
        invalid = words + jnp.arange(blocks.shape[1] + 1, dtype=jnp.int32)
        indices = jnp.where(valid, selected // 32, invalid[None, :])
        weights = jnp.where(valid, jnp.int32(1) << (selected % 32), 0)
        bits = jax.vmap(
            lambda row, weight: jnp.zeros((words,), jnp.int32)
            .at[row]
            .add(weight, mode="drop")
        )(indices, weights)
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
