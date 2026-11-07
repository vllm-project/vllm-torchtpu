# SPDX-License-Identifier: Apache-2.0
"""D2 union-dense QSA prefill with the union gather fused into the kernel (P-ATTN).

Served only through ``union_walk_prefill`` behind VLLM_TPU_QSA_PREFILL_UNION_WALK
(default off). The D2 prototype (``qsa_union_prefill``) writes
each query group's union of selected 4-token groups to a contiguous HBM copy
and runs the served block-sparse kernel on it. This kernel reads the same
union straight from the paged pool. Per KV tile it issues one 4-token DMA per
union group into the VMEM tile buffer, double-buffered as the served kernel
does with page segments. So the union copy is never written to HBM or read
back. The group's union list (physical token of each group) moves to SMEM at
the start of the block, as the fused MoE kernel stages its token-gather
windows.

Structure: DeepSeek-V4's union-dense CSA kernel (contiguous union tiles, one
row mask over union ranks per query). The gather runs on the TensorCore's DMA
engines, not on SparseCore.

Numerics: every tile of a group's union holds a group some row selected, so
every tile is active, in rank order. The per-tile math is the served
kernel's paired-page f32 path. The output is therefore bitwise equal to the
prototype's (``tests/kernels/test_qwen4_exp_union_dense.py``).
"""

import contextlib
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import utils
from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_prefill as up
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import (
    PAIRED_PAGE_VMEM_MIB,
    get_sparse_prefill_tiles,
)
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16

# HBM int32 rows are tiled in 128-element units: a block's union list starts
# on a tile so its HBM-to-SMEM copy is aligned.
_LIST_ALIGN = 128
_ISSUE_UNROLL = 8


def _cdiv(a, b):
    return (a + b - 1) // b


def _kernel(
    tiles_ref,
    query_ref,
    bitset_ref,
    lengths_ref,
    phys_hbm_ref,
    cache_ref,
    output_ref,
    phys_ref,
    list_sem,
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
    head_dim: int,
    sm_scale: float,
    packing: int,
    kv_heads: int,
    q_heads_per_kv: int,
    query_block: int,
    k_scale: float,
    v_scale: float,
    list_width: int,
    single_wait: bool,
    window: int = 0,
    coalesce: int = 0,
    lean: bool = False,
    piece_width: int = 0,
    first_width: int = 0,
):
    # window (i1b hedge): at most two windows of ``window`` group copies in
    # flight per tile buffer; issuing window k waits for one window's bytes.
    # coalesce (i1b): copies of 1..coalesce physically contiguous union groups,
    # from a per-block piece list: rows [0, piece_width) hold each piece's
    # source pool row, then ``first_width`` rows of per-tile first-piece
    # indices, then each piece's (destination group | size class << 16).
    # lean: no finite/written guard on keys (as the served LEAN option; a key
    # only reaches a visible score through the mask select).
    block = pl.program_id(0)
    count = tiles_ref[block]
    groups_per_tile = kv_tile // ratio

    # This block's union list (physical token of each 4-token group) to SMEM.
    stage = pltpu.make_async_copy(
        phys_hbm_ref.at[pl.ds(pl.multiple_of(block * list_width, _LIST_ALIGN), list_width)],
        phys_ref,
        list_sem,
    )
    stage.start()

    # From here to the online softmax: the served kernel's paired-page f32 path
    # (block_sparse_gqa_prefill._kernel), with one query subtile of
    # ``query_block`` rows and the union tile in place of the page tile.
    q = (
        query_ref[...]
        .astype(jnp.float32)
        .reshape(query_block, kv_heads, q_heads_per_kv, head_dim)
        .transpose(1, 2, 0, 3)
    )
    score_scale = sm_scale * k_scale
    q = q * score_scale
    positions = lengths_ref[:, 0] - 1
    row = jnp.arange(kv_tile, dtype=jnp.int32)
    max_ref[...] = jnp.full(max_ref.shape, -jnp.inf, jnp.float32)
    mass_ref[...] = jnp.zeros(mass_ref.shape, jnp.float32)
    accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)
    stage.wait()

    def wait_rows(rows, kv_ref, sem_ref):
        # DMA semaphores count bytes; the source address of a wait descriptor
        # does not matter.
        pltpu.make_async_copy(
            cache_ref.at[pl.ds(0, rows)], kv_ref.at[pl.ds(0, rows)], sem_ref
        ).wait()

    nwin = groups_per_tile // window if window else 0

    def start_tile(tile, kv_ref, sem_ref):
        base = tile * groups_per_tile
        if coalesce:
            first = piece_width + tile
            meta_base = piece_width + first_width

            def issue_piece(piece, carry):
                source = phys_ref[piece]
                meta = phys_ref[meta_base + piece]
                destination = (meta & 0xFFFF) * ratio
                size_class = meta >> 16
                for k in range(coalesce.bit_length()):
                    rows = (1 << k) * ratio

                    @pl.when(size_class == k)
                    def _(rows=rows):
                        pltpu.make_async_copy(
                            cache_ref.at[pl.ds(source, rows)],
                            kv_ref.at[pl.ds(destination, rows)],
                            sem_ref,
                        ).start()

                return carry

            lax.fori_loop(phys_ref[first], phys_ref[first + 1], issue_piece, None)
            return

        def issue(step, carry):
            # Eight group DMAs per step, as the fused MoE kernel issues its
            # token-row gathers.
            for offset in range(_ISSUE_UNROLL):
                index = step * _ISSUE_UNROLL + offset
                pltpu.make_async_copy(
                    cache_ref.at[pl.ds(phys_ref[base + index], ratio)],
                    kv_ref.at[pl.ds(index * ratio, ratio)],
                    sem_ref,
                ).start()
            return carry

        if not window:
            lax.fori_loop(0, groups_per_tile // _ISSUE_UNROLL, issue, None)
            return
        steps = window // _ISSUE_UNROLL
        for w in range(nwin):
            if w >= 2:
                wait_rows(window * ratio, kv_ref, sem_ref)
            lax.fori_loop(w * steps, (w + 1) * steps, issue, None)

    def wait_tile(kv_ref, sem_ref):
        # One whole-tile wait covers every copy whatever their landing order
        # (as in the fused MoE kernel); a windowed tile has already consumed
        # all but its last two windows.
        if window:
            wait_rows(min(nwin, 2) * window * ratio, kv_ref, sem_ref)
            return
        if single_wait or coalesce:
            pltpu.make_async_copy(cache_ref.at[pl.ds(0, kv_tile)], kv_ref, sem_ref).wait()
            return
        for index in range(groups_per_tile):
            pltpu.make_async_copy(
                cache_ref.at[pl.ds(0, ratio)],
                kv_ref.at[pl.ds(index * ratio, ratio)],
                sem_ref,
            ).wait()

    def split_kv(kv_ref):
        cache_words = kv_ref.bitcast(jnp.uint32).reshape(-1, head_dim)
        pairs = []
        head_axes = kv_ref.shape[1]
        for head_axis in range(head_axes):
            words = utils.strided_load(cache_words, head_axis, kv_tile * head_axes, head_axes)
            pairs.extend(
                utils.convert_to_target_bitwidth(
                    words, target_bitwidth=32 // packing, kv_dtype=kv_ref.dtype
                )
            )
        return pairs

    def compute_tile(tile, kv_ref):
        logical = tile * kv_tile
        tile_words = kv_tile // (ratio * 32)
        word_base = tile_words * tile
        word = bitset_ref[0, word_base, :query_block][:, None]
        for index in range(1, tile_words):
            word = jnp.where(
                row[None, :] < index * ratio * 32,
                word,
                bitset_ref[0, word_base + index, :query_block][:, None],
            )
        visible = ((word >> ((row // ratio) % 32)[None, :]) & 1) != 0
        visible &= logical + row[None, :] <= positions[:, None]
        written = logical + lax.broadcasted_iota(jnp.int32, (kv_tile, head_dim), 0) < jnp.max(
            lengths_ref[:, 0]
        )
        key_values = []
        for key, value in split_kv(kv_ref)[:kv_heads]:
            key = key.astype(jnp.float32)
            value = value.astype(jnp.float32)
            if not lean:
                key = jnp.where(jnp.isfinite(key) & written, key, 0)
            value = jnp.where(jnp.isfinite(value) & written, value, 0)
            key_values.append((key, value))
        rows = query_block * q_heads_per_kv
        row_visible = jnp.broadcast_to(
            visible[None, :, :], (q_heads_per_kv, query_block, kv_tile)
        ).reshape(rows, kv_tile)
        for head, (key, value) in enumerate(key_values):
            m = max_ref[0, head, :]
            mass = mass_ref[0, head, :]
            acc = accumulator_ref[0, head, :, :]
            query = q[head].reshape(rows, head_dim)
            score = lax.dot(
                query,
                key,
                dimension_numbers=(([1], [1]), ([], [])),
                preferred_element_type=jnp.float32,
            )
            score = jnp.where(row_visible, score, -jnp.inf)
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
            max_ref[0, head, :] = next_m
            mass_ref[0, head, :] = mass * alpha + jnp.sum(prob, axis=1)
            accumulator_ref[0, head, :, :] = acc * alpha[:, None] + weighted

    @pl.when(count > 0)
    def prefetch_first():
        start_tile(0, kv0_ref, sem0_ref)

    @pl.when(count > 1)
    def prefetch_second():
        start_tile(1, kv1_ref, sem1_ref)

    def pair_step(pair, _):
        tile = pair * 2
        wait_tile(kv0_ref, sem0_ref)
        compute_tile(tile, kv0_ref)

        @pl.when(tile + 2 < count)
        def prefetch_next_even():
            start_tile(tile + 2, kv0_ref, sem0_ref)

        wait_tile(kv1_ref, sem1_ref)
        compute_tile(tile + 1, kv1_ref)

        @pl.when(tile + 3 < count)
        def prefetch_next_odd():
            start_tile(tile + 3, kv1_ref, sem1_ref)

        return None

    lax.fori_loop(0, count // 2, pair_step, None)

    @pl.when(count % 2 != 0)
    def consume_odd():
        wait_tile(kv0_ref, sem0_ref)
        compute_tile(count - 1, kv0_ref)

    heads = []
    for head in range(kv_heads):
        mass = mass_ref[0, head, :]
        acc = accumulator_ref[0, head, :, :]
        heads.append(
            (acc * (v_scale / jnp.where(mass == 0, 1, mass))[:, None])
            .reshape(q_heads_per_kv, query_block, head_dim)
            .transpose(1, 0, 2)
        )
    output_ref[...] = jnp.concatenate(heads, axis=1).astype(output_ref.dtype)


def union_physical_tokens(union, page_table_row, *, page_size, ratio, width):
    """int32[groups, width]: pool row of each union group's first token.

    Padding (rank >= the union size, never visible) repeats the group's first
    union entry, so every DMA address is valid.
    """
    first = jnp.maximum(union[:, :1], 0)
    groups = jnp.where(union >= 0, union, first)
    tokens = groups * ratio
    physical = page_table_row[tokens // page_size] * page_size + tokens % page_size
    pad = width - physical.shape[1]
    if pad:
        physical = jnp.concatenate(
            [physical, jnp.broadcast_to(physical[:, :1], (physical.shape[0], pad))], axis=1
        )
    return physical.astype(jnp.int32)


def coalesce_pieces(physical, usize, *, groups_per_tile, cap_tiles, max_piece, ratio):
    """Piece list for coalesced copies: (sources, first-piece per tile, metas).

    A run is a stretch of union ranks whose pool rows are consecutive groups,
    inside one tile and below the union size (padding ranks are single
    pieces). Each run splits into ``max_piece``-group pieces plus the binary
    pieces of its remainder, largest first. Pieces cover every rank of every
    tile exactly once, so a tile's copies add up to the whole tile.
    Returns int32 [groups, width] sources (pool row of each piece, rank
    order), [groups, cap_tiles + 1] first-piece index per tile, and [groups,
    width] metas (destination group in the tile | log2(size) << 16).
    """
    ngroups, width = physical.shape
    span = cap_tiles * groups_per_tile
    phys = physical[:, :span]
    rank = jnp.arange(span, dtype=jnp.int32)[None, :]
    live = rank < usize[:, None]
    previous = jnp.concatenate([jnp.full((ngroups, 1), -(2**30), jnp.int32), phys[:, :-1]], axis=1)
    joined = live & (rank % groups_per_tile != 0) & (phys == previous + ratio)
    start = ~joined
    run_start = lax.cummax(jnp.where(start, rank, 0), axis=1)
    ends_here = ~jnp.concatenate([joined[:, 1:], jnp.zeros((ngroups, 1), bool)], axis=1)
    run_end = lax.cummin(jnp.where(ends_here, rank, span), axis=1, reverse=True)
    length = run_end - run_start + 1
    offset = rank - run_start
    full = (length // max_piece) * max_piece
    rem = length - full
    in_full = offset < full
    tail_off = offset - full
    lowbit = tail_off & -tail_off
    is_piece = jnp.where(
        in_full,
        offset % max_piece == 0,
        (tail_off == 0) | (tail_off == (rem & ~(lowbit - 1))),
    )
    size = jnp.where(in_full, max_piece, 1 << (31 - lax.clz(jnp.maximum(rem - tail_off, 1))))
    size_class = 31 - lax.clz(size)
    slot = jnp.cumsum(is_piece, axis=1, dtype=jnp.int32) - 1
    slot = jnp.where(is_piece, slot, width + rank)
    rows = jnp.arange(ngroups, dtype=jnp.int32)[:, None]
    meta = (rank % groups_per_tile) | (size_class << 16)
    sources = jnp.zeros((ngroups, width + span), jnp.int32).at[rows, slot].set(
        phys, unique_indices=True
    )[:, :width]
    metas = jnp.zeros((ngroups, width + span), jnp.int32).at[rows, slot].set(
        meta, unique_indices=True
    )[:, :width]
    per_tile = jnp.sum(is_piece.reshape(ngroups, cap_tiles, groups_per_tile), axis=2, dtype=jnp.int32)
    first = jnp.concatenate(
        [jnp.zeros((ngroups, 1), jnp.int32), jnp.cumsum(per_tile, axis=1, dtype=jnp.int32)], axis=1
    )
    return sources, first, metas


@functools.partial(
    jax.jit,
    static_argnames=("group", "union_cap", "kv_tile", "ratio", "k_scale", "v_scale", "shift",
                     "single_wait", "window", "coalesce", "lean", "packed_bitset", "plan"),
)
def union_dense_attention(
    query,
    gate,
    blocks,
    lengths,
    pool,
    page_table_row,
    *,
    group: int,
    union_cap: int,
    kv_tile: int,
    ratio: int = 4,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    shift: int = 0,
    single_wait: bool = True,
    window: int = 0,
    coalesce: int = 0,
    lean: bool = False,
    packed_bitset: bool = False,
    plan: str = "sort",
):
    """D2 with the gather fused into the kernel, for one request's prefill rows.

    Same contract as ``qsa_union_prefill.union_attention``: returns (output
    [T, heads, D] bf16 gated as served, usize). ``union_cap`` is a multiple of
    32 and at least every group's union size; ``T % group == 0``.
    ``single_wait`` (default) waits each tile's DMAs with one whole-tile
    descriptor; False waits one descriptor per group (same bytes).
    ``window``: at most two windows of this many group copies in flight per
    tile buffer. ``coalesce``: one copy per run of up to this many physically
    contiguous union groups (a power of two). ``lean``: no guard on keys.
    ``packed_bitset``: build the row masks for 128 rows per bitset-kernel
    block (128 // group query groups side by side) instead of one group per
    block; the same bits, 128 // group times fewer kernel steps.
    ``plan``: "sort" (``qsa_union_prefill.union_plan``), "nosort_rk" (the
    sort-free plan ranking with its Pallas compare pass) or "nosort"
    (``qsa_union_plan.union_plan_nosort``: the same arrays from bitsets and
    prefix counts; group ids must lie below the page table's reach and be
    distinct within a row, as the indexer's top-k gives).
    All of these move the same bytes into the same tile rows, so the output
    is unchanged (``lean`` too while visible keys are finite).
    """
    tokens, query_heads, head_dim = query.shape
    pages, page_size, head_axis, packing, cache_dim = pool.shape
    kv_heads = 2 if query_heads == 24 else 1
    if kv_tile not in bsp.PAIRED_KV_TILES or ratio != 4 or page_size % ratio:
        raise ValueError("union-dense needs paired KV tiles (256-2048) and 4-token groups")
    if tokens % group or group % 8 or group > 256:
        raise ValueError("query groups must be 8-aligned, at most 256 rows, and split T")
    if (
        head_dim != 256
        or cache_dim != head_dim
        or pool.dtype != jnp.float8_e4m3fn
        or packing != 4
        or query_heads % kv_heads
    ):
        raise ValueError("union-dense needs BF16 D256 queries and an FP8 D256 pool")
    if window and (coalesce or window % _ISSUE_UNROLL or (kv_tile // ratio) % window):
        raise ValueError("a copy window is a multiple of 8 groups that splits the tile, without coalescing")
    if coalesce and (coalesce & (coalesce - 1) or not 2 <= coalesce <= kv_tile // ratio):
        raise ValueError("coalesced copies take a power-of-two maximum of 2 or more groups")
    if plan in ("nosort", "nosort_rk"):
        from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import union_plan_nosort

        union, vblocks, vlengths, usize = union_plan_nosort(
            blocks, lengths, group=group, union_cap=union_cap,
            groups_cap=page_table_row.shape[0] * page_size // ratio, ratio=ratio, shift=shift,
            rank_kernel=plan == "nosort_rk",
        )
    elif plan == "sort":
        union, vblocks, vlengths, usize = up.union_plan(
            blocks, lengths, group=group, union_cap=union_cap, ratio=ratio, shift=shift
        )
    else:
        raise ValueError(f"unknown union plan {plan!r}")
    groups_per_tile = kv_tile // ratio
    cap_tiles = _cdiv(union_cap * ratio, kv_tile)
    list_width = _cdiv(cap_tiles * groups_per_tile, _LIST_ALIGN) * _LIST_ALIGN
    physical = union_physical_tokens(
        union, page_table_row, page_size=page_size, ratio=ratio, width=list_width
    )
    piece_width = first_width = 0
    if coalesce:
        sources, first, metas = coalesce_pieces(
            physical, usize, groups_per_tile=groups_per_tile, cap_tiles=cap_tiles,
            max_piece=coalesce, ratio=ratio,
        )
        piece_width = list_width
        first_width = _cdiv(cap_tiles + 1, _LIST_ALIGN) * _LIST_ALIGN
        first = jnp.pad(first, ((0, 0), (0, first_width - first.shape[1])))
        physical = jnp.concatenate([sources, first, metas], axis=1)
        list_width = physical.shape[1]
    # Every union tile holds a selected group, so all of them are active.
    tiles = jnp.minimum(_cdiv(usize * ratio, kv_tile), cap_tiles).astype(jnp.int32)
    words_per_tile = kv_tile // (ratio * 32)
    words = _cdiv(cap_tiles * words_per_tile, 128) * 128
    width = _cdiv(group, 128) * 128
    if packed_bitset and group < 128 and 128 % group == 0 and tokens % 128 == 0:
        # Lane j of a 128-row block is row j, whatever its query group: the
        # per-row bits do not depend on the block. Regroup the lanes after.
        packed = bsp._selection_bitset(
            vblocks, vlengths, ratio=ratio, groups=union_cap, words=words, query_block=128,
            width=128,
        )
        segments = 128 // group
        bits = (
            packed.reshape(tokens // 128, words, segments, group)
            .transpose(0, 2, 1, 3)
            .reshape(tokens // group, words, group)
        )
        bits = jnp.pad(bits, ((0, 0), (0, 0), (0, width - group)))
    else:
        bits = bsp._selection_bitset(
            vblocks, vlengths, ratio=ratio, groups=union_cap, words=words, query_block=group,
            width=width,
        )
    nblocks = tokens // group
    q_heads_per_kv = query_heads // kv_heads
    state_shape = (1, kv_heads, group * q_heads_per_kv)
    output = pl.pallas_call(
        functools.partial(
            _kernel,
            ratio=ratio,
            kv_tile=kv_tile,
            head_dim=head_dim,
            sm_scale=head_dim**-0.5,
            packing=packing,
            kv_heads=kv_heads,
            q_heads_per_kv=q_heads_per_kv,
            query_block=group,
            k_scale=k_scale,
            v_scale=v_scale,
            list_width=list_width,
            single_wait=single_wait,
            window=window,
            coalesce=coalesce,
            lean=lean,
            piece_width=piece_width,
            first_width=first_width,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(nblocks,),
            in_specs=[
                pl.BlockSpec((group, query_heads, head_dim), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((1, words, width), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((group, 1), lambda b, *_: (b, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((group, query_heads, head_dim), lambda b, *_: (b, 0, 0)),
            scratch_shapes=[
                pltpu.SMEM((list_width,), jnp.int32),
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((kv_tile, head_axis, packing, head_dim), pool.dtype),
                pltpu.VMEM((kv_tile, head_axis, packing, head_dim), pool.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM((*state_shape, head_dim), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=PAIRED_PAGE_VMEM_MIB.get(group, max(PAIRED_PAGE_VMEM_MIB.values()))
            * 1024**2,
            disable_bounds_checks=True,
        ),
        out_shape=jax.ShapeDtypeStruct(query.shape, jnp.bfloat16),
        name=f"qsa_union_dense_g{group}_kv{kv_tile}"
        + (f"_w{window}" if window else "")
        + (f"_co{coalesce}" if coalesce else "")
        + ("_lean" if lean else "")
        + {"sort": "", "nosort": "_np", "nosort_rk": "_rk"}.get(plan, ""),
    )(
        tiles,
        query.astype(jnp.bfloat16),
        bits,
        vlengths[:, None],
        physical.reshape(-1),
        pool.reshape(pages * page_size, head_axis, packing, head_dim),
    )
    output = bf16(
        bf16(output).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
    )
    return jnp.where((lengths > 0)[:, None, None], output, 0), usize


# ---------------------------------------------------------------------------
# i1c: the union walked inside the kernel. No union list, ranks or virtual
# bitset are built in XLA (i1b: XLA gathers/scatters cost ~10-15 ns per element
# and the union-list step was the plan's main term). The kernel stages the
# group's union words in SMEM and, per tile, takes the next ``kv_tile / ratio``
# set bits with the scalar core: each is a union group (in rank order, so the
# tile holds the same rows as the prototype's), whose 4-token DMA address comes
# from the page table and whose row-mask line is read from the rows' real
# selection bitset. Same tiles, masks and per-tile math as the prototype, so
# the output is bitwise equal to it.

# De Bruijn table: lowest set bit of a 32-bit word without a count-zeros op.
_DEBRUIJN = np.asarray([0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
                        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9], np.int32)


def _walk_kernel(
    tiles_ref,
    usize_ref,
    table_ref,
    debruijn_ref,
    physw_ref,
    query_ref,
    bits_ref,
    lengths_ref,
    ubits_hbm_ref,
    cache_ref,
    output_ref,
    ubits_ref,
    list_sem,
    state_ref,
    nz_ref,
    line0_ref,
    line1_ref,
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
    head_dim: int,
    sm_scale: float,
    packing: int,
    kv_heads: int,
    q_heads_per_kv: int,
    query_block: int,
    k_scale: float,
    v_scale: float,
    words: int,
    words_padded: int,
    page_size: int,
    lean: bool,
    scan_list: bool = False,
    line_matmul: bool = False,
    fast_walk: bool = False,
    physw_hbm_ref=None,
    physw_sem=None,
    scopes: int = 0,
    walk_runs: int = 0,
    qk16: bool = False,
    gathered_refs=None,
    gathered_cap_tiles: int = 0,
    listed_refs=None,
    listed_lines_ref=None,
    runs_refs=None,
    run_sizes=(),
):
    # runs_refs (step 4, with listed_lines_ref): (counts in HBM, entries in HBM,
    #   their SMEM copies, DMA semaphore); per tile, the union's pool-contiguous
    #   runs cut into pieces of ``run_sizes`` groups (largest first), one list per
    #   size: entry = (pool group << 9) | tile slot. Each size's pieces are one
    #   loop of static-size copies, so no branch per copy (i1j: branchy size
    #   dispatch lost to the per-group walk). The pieces fill each tile exactly.
    # listed_lines_ref (with listed_refs): the tiles' mask lines precomputed in
    #   HBM (``union_gather_reference``'s lines layout); each tile's lines are
    #   one contiguous copy and the per-group line work goes (step 4 probe).
    # listed_refs (SparseCore list only): (ids in HBM, their SMEM copy, DMA
    #   semaphore); each query group's union group ids in rank order (-1 for
    #   padding), ``union_gather_list_reference``'s ``ids``. The kernel issues
    #   the walk's per-group copies and lines from the list: no bit scan, no
    #   word advance, no branching. Same tiles, lines and compute: bitwise the walk.
    # gathered_refs (SparseCore union gather, step 1): (gathered KV, gathered
    #   lines) in HBM, the union already compacted in rank order per query
    #   group (``union_gather_reference``). Each tile is two contiguous copies;
    #   no walk, no per-group work. Same tiles, lines and compute: bitwise the
    #   walk.
    # qk16 (opt-in): QK as one BF16 MXU pass with FP32 accumulation instead of
    #   the FP32 dot (a multi-pass FP32 matmul in Mosaic). Exact, not a
    #   precision cut: the queries arrive as BF16, the score scale is a power
    #   of two (checked by ``union_walk_kernel``), so the pre-scaled queries
    #   are still BF16 values; FP8 E4M3 keys are BF16 values. Both dots get the
    #   same operand values. PV is unchanged.
    # walk_runs (fast walk only; 0 = off): one DMA per run of consecutive set
    #   bits of the current word instead of one per union group (see
    #   ``start_tile_runs``). Same tile rows, lines and order.
    # Lowering fallbacks (i1d, if i1c's Mosaic lowering fails), same output:
    # scan_list: no scalar while loop; a branch-free fori_loop lists the
    #   group's non-zero words at block start, and the walk steps through it.
    # line_matmul: one mask line per union group (integer-indexed row store)
    #   instead of four-row stores at dynamic offsets; the tile's [rows, keys]
    #   mask is the lines times a 0/1 group-to-key expansion matrix (exact).
    # physw_hbm_ref (per-group page rows, ``_walk_kernel_per_group``): each
    #   group's per-word pool bases sit in HBM and are staged into the SMEM
    #   ``physw_ref`` at block start, as the union words are.
    block = pl.program_id(0)
    count = tiles_ref[block]
    usize = usize_ref[block]
    groups_per_tile = kv_tile // ratio
    width = bits_ref.shape[2]

    stage = pltpu.make_async_copy(
        ubits_hbm_ref.at[pl.ds(pl.multiple_of(block * words_padded, _LIST_ALIGN), words_padded)],
        ubits_ref,
        list_sem,
    )
    stage.start()
    if physw_hbm_ref is not None:
        physw_stage = pltpu.make_async_copy(
            physw_hbm_ref.at[pl.ds(pl.multiple_of(block * words_padded, _LIST_ALIGN), words_padded)],
            physw_ref,
            physw_sem,
        )
        physw_stage.start()
    q = (
        query_ref[...]
        .astype(jnp.float32)
        .reshape(query_block, kv_heads, q_heads_per_kv, head_dim)
        .transpose(1, 2, 0, 3)
    )
    score_scale = sm_scale * k_scale
    q = q * score_scale
    if qk16:
        q = q.astype(jnp.bfloat16)  # lossless: BF16 values times a power of two
    positions = lengths_ref[:, 0] - 1
    row = jnp.arange(kv_tile, dtype=jnp.int32)
    max_ref[...] = jnp.full(max_ref.shape, -jnp.inf, jnp.float32)
    mass_ref[...] = jnp.zeros(mass_ref.shape, jnp.float32)
    accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)
    # Walk state: current word, its bits not yet taken, groups issued so far.
    state_ref[0] = -1
    state_ref[1] = 0
    state_ref[2] = 0
    state_ref[3] = 0
    stage.wait()
    if physw_hbm_ref is not None:
        physw_stage.wait()
    if listed_refs is not None:
        ids_hbm_ref, ids_ref, ids_sem = listed_refs
        ids_width = ids_ref.shape[0]
        ids_stage = pltpu.make_async_copy(
            ids_hbm_ref.at[pl.ds(pl.multiple_of(block * ids_width, _LIST_ALIGN), ids_width)], ids_ref, ids_sem
        )
        ids_stage.start()
        ids_stage.wait()
    if runs_refs is not None:
        counts_hbm_ref, entries_hbm_ref, counts_ref, entries_ref, runs_sem = runs_refs
        counts_width, entries_width = counts_ref.shape[0], entries_ref.shape[0]
        counts_stage = pltpu.make_async_copy(
            counts_hbm_ref.at[pl.ds(pl.multiple_of(block * counts_width, _LIST_ALIGN), counts_width)], counts_ref,
            runs_sem,
        )
        counts_stage.start()
        counts_stage.wait()
        entries_stage = pltpu.make_async_copy(
            entries_hbm_ref.at[pl.ds(pl.multiple_of(block * entries_width, _LIST_ALIGN), entries_width)],
            entries_ref, runs_sem,
        )
        entries_stage.start()
        entries_stage.wait()
    if scan_list:
        def list_word(word, found):
            nz_ref[found] = word
            return found + (ubits_ref[word] != 0).astype(jnp.int32)

        lax.fori_loop(0, words, list_word, jnp.int32(0))

    def next_set_bit_listed():
        word, bits, step = state_ref[0], state_ref[1], state_ref[3]
        need = bits == 0
        word = jnp.where(need, nz_ref[jnp.minimum(step, words - 1)], word)
        word = jnp.maximum(word, 0)
        bits = jnp.where(need, ubits_ref[word], bits)
        low = bits & (-bits)
        bit = debruijn_ref[lax.shift_right_logical(low * 0x077CB531, 27)]
        state_ref[0] = word
        state_ref[1] = bits & (bits - 1)
        state_ref[3] = step + need.astype(jnp.int32)
        return word, bit

    def next_set_bit():
        if scan_list:
            return next_set_bit_listed()

        def more(carry):
            word, bits = carry
            return jnp.logical_and(bits == 0, word < words - 1)

        def advance(carry):
            word = carry[0] + 1
            return word, ubits_ref[word]

        word, bits = lax.while_loop(more, advance, (state_ref[0], state_ref[1]))
        low = bits & (-bits)
        bit = debruijn_ref[lax.shift_right_logical(low * 0x077CB531, 27)]
        state_ref[0] = word
        state_ref[1] = bits & (bits - 1)
        return jnp.maximum(word, 0), bit

    def start_tile_fast(kv_ref, line_ref, sem_ref):
        # i1e: the walk state rides in loop carries (no SMEM round trip per
        # group), and a word's pool row comes from the per-word base table
        # (a word is 32 groups = 128 tokens, inside one page), so a group's
        # address is one add instead of a division and a table load.
        def more(carry):
            word, bits = carry
            return jnp.logical_and(bits == 0, word < words - 1)

        def advance(carry):
            word = carry[0] + 1
            return word, ubits_ref[word]

        def issue(step, carry):
            word, bits, rank = carry
            for offset in range(_ISSUE_UNROLL):
                index = step * _ISSUE_UNROLL + offset
                live = rank < usize
                word, bits = lax.while_loop(more, advance, (word, bits))
                low = bits & (-bits)
                bit = debruijn_ref[lax.shift_right_logical(low * 0x077CB531, 27)]
                bits = bits & (bits - 1)
                at = jnp.maximum(word, 0)
                physical = jnp.where(live, physw_ref[at] + bit * ratio, physw_ref[0])
                pltpu.make_async_copy(
                    cache_ref.at[pl.ds(physical, ratio)],
                    kv_ref.at[pl.ds(index * ratio, ratio)],
                    sem_ref,
                ).start()
                if line_matmul:
                    line = lax.shift_right_logical(bits_ref[0, at, :], bit) & 1
                    line_ref[index, :] = jnp.where(live, line, 0)
                else:
                    line = lax.shift_right_logical(bits_ref[0, pl.ds(at, 1), :], bit) & 1
                    line = jnp.where(live, line, 0)
                    line_ref[pl.ds(index * ratio, ratio), :] = jnp.broadcast_to(line, (ratio, width))
                rank = rank + 1
            return word, bits, rank

        word, bits, rank = lax.fori_loop(
            0, groups_per_tile // _ISSUE_UNROLL, issue, (state_ref[0], state_ref[1], state_ref[2])
        )
        state_ref[0] = word
        state_ref[1] = bits
        state_ref[2] = rank

    def start_tile_runs(kv_ref, line_ref, sem_ref):
        # walk_runs: the fast walk with one DMA per run of consecutive set bits
        # of the current word. A run of union groups bit..bit+n-1 of one word
        # is 4n consecutive pool rows (a word sits inside one page) and n
        # consecutive union ranks, so one copy fills n consecutive tile slots.
        # Each piece takes the largest power of two at most min(run length,
        # walk_runs, slots left in the tile, bits left in the word); padding
        # ranks (past usize) take the largest power of two at most min(slots
        # left, walk_runs), copied from the first rows of word 0 (valid rows)
        # with zero lines. Pieces fill the tile exactly, in rank order, so the
        # whole-tile byte-count wait still matches. The size is dispatched to
        # static copy shapes with pl.when, as ``_kernel``'s coalesced pieces.
        # The run never passes usize: usize is the union's popcount, so every
        # set bit is a live rank.
        sizes = [1 << k for k in range(walk_runs.bit_length())]  # 1, 2, ..., walk_runs

        def more(carry):
            word, bits = carry
            return jnp.logical_and(bits == 0, word < words - 1)

        def advance(carry):
            word = carry[0] + 1
            return word, ubits_ref[word]

        def unfilled(carry):
            return carry[3] < groups_per_tile

        def piece(carry):
            word, bits, rank, index = carry
            live = rank < usize
            word, bits = lax.while_loop(more, advance, (word, bits))
            low = bits & (-bits)
            bit = debruijn_ref[lax.shift_right_logical(low * 0x077CB531, 27)]
            at = jnp.maximum(word, 0)
            # The word's bits from ``bit`` up; the logical shift clears the bits
            # past the word's end, so a run never crosses into the next word.
            ahead = lax.shift_right_logical(bits, bit)
            left = groups_per_tile - index
            size = jnp.int32(1)
            for s in sizes[1:]:
                run = (ahead & ((1 << s) - 1)) == (1 << s) - 1
                fits = jnp.logical_and(s <= left, jnp.logical_or(jnp.logical_not(live), run))
                size = jnp.where(fits, s, size)
            physical = jnp.where(live, physw_ref[at] + bit * ratio, physw_ref[0])
            destination = index * ratio
            for s in sizes:

                @pl.when(size == s)
                def _(s=s):
                    pltpu.make_async_copy(
                        cache_ref.at[pl.ds(physical, s * ratio)],
                        kv_ref.at[pl.ds(destination, s * ratio)],
                        sem_ref,
                    ).start()
                    # Group j of the piece is bit + j of the word: its line,
                    # as the per-group walk writes it.
                    lines = s if line_matmul else s * ratio
                    shift = bit + lax.broadcasted_iota(jnp.int32, (lines, width), 0) // (
                        1 if line_matmul else ratio
                    )
                    line = lax.shift_right_logical(
                        jnp.broadcast_to(bits_ref[0, pl.ds(at, 1), :], (lines, width)), shift
                    ) & 1
                    line_ref[pl.ds(index if line_matmul else destination, lines), :] = jnp.where(
                        live, line, 0
                    )

            # Clear the piece's bits (bit .. bit + size - 1; wraps cleanly at bit 31).
            taken = lax.shift_left(low, size) - low
            bits = jnp.where(live, bits & ~taken, bits)
            return word, bits, rank + size, index + size

        word, bits, rank, _ = lax.while_loop(
            unfilled, piece, (state_ref[0], state_ref[1], state_ref[2], jnp.int32(0))
        )
        state_ref[0] = word
        state_ref[1] = bits
        state_ref[2] = rank

    def start_tile_listed(kv_ref, line_ref, sem_ref):
        ids_ref = listed_refs[1]
        first = state_ref[2] * groups_per_tile
        if listed_lines_ref is not None:
            rows = line_ref.shape[0]
            at_tile = block * gathered_cap_tiles + state_ref[2]
            pltpu.make_async_copy(listed_lines_ref.at[pl.ds(at_tile * rows, rows)], line_ref, sem_ref).start()

        def issue(step, carry):
            for offset in range(_ISSUE_UNROLL):
                index = step * _ISSUE_UNROLL + offset
                group_id = ids_ref[first + index]
                live = group_id >= 0
                safe = jnp.maximum(group_id, 0)
                at = lax.shift_right_logical(safe, 5)
                bit = safe & 31
                physical = jnp.where(live, physw_ref[at] + bit * ratio, physw_ref[0])
                pltpu.make_async_copy(
                    cache_ref.at[pl.ds(physical, ratio)],
                    kv_ref.at[pl.ds(index * ratio, ratio)],
                    sem_ref,
                ).start()
                if listed_lines_ref is not None:
                    continue  # the tile's lines arrive as one copy
                if line_matmul:
                    line = lax.shift_right_logical(bits_ref[0, at, :], bit) & 1
                    line_ref[index, :] = jnp.where(live, line, 0)
                else:
                    line = lax.shift_right_logical(bits_ref[0, pl.ds(at, 1), :], bit) & 1
                    line = jnp.where(live, line, 0)
                    line_ref[pl.ds(index * ratio, ratio), :] = jnp.broadcast_to(line, (ratio, width))
            return carry

        lax.fori_loop(0, groups_per_tile // _ISSUE_UNROLL, issue, None)
        state_ref[2] = state_ref[2] + 1

    def start_tile_run_tables(kv_ref, line_ref, sem_ref):
        counts_ref, entries_ref = runs_refs[2], runs_refs[3]
        tile = state_ref[2]
        rows = line_ref.shape[0]
        at_tile = block * gathered_cap_tiles + tile
        pltpu.make_async_copy(listed_lines_ref.at[pl.ds(at_tile * rows, rows)], line_ref, sem_ref).start()
        first = tile * groups_per_tile
        offset = jnp.int32(0)
        for c, size in enumerate(run_sizes):  # one static copy size per loop: no branch per copy
            count = counts_ref[tile * len(run_sizes) + c]

            def piece(i, carry, size=size, start=offset):
                entry = entries_ref[first + start + i]
                source = lax.shift_right_logical(entry, 9) * ratio
                slot = (entry & 511) * ratio
                pltpu.make_async_copy(
                    cache_ref.at[pl.ds(source, size * ratio)],
                    kv_ref.at[pl.ds(slot, size * ratio)],
                    sem_ref,
                ).start()
                return carry

            lax.fori_loop(0, count, piece, None)
            offset = offset + count
        state_ref[2] = tile + 1

    def start_tile_gathered(kv_ref, line_ref, sem_ref):
        gathered_kv, gathered_lines = gathered_refs
        tile = state_ref[2]
        at = block * gathered_cap_tiles + tile
        rows = line_ref.shape[0]
        pltpu.make_async_copy(gathered_kv.at[pl.ds(at * kv_tile, kv_tile)], kv_ref, sem_ref).start()
        pltpu.make_async_copy(gathered_lines.at[pl.ds(at * rows, rows)], line_ref, sem_ref).start()
        state_ref[2] = tile + 1

    def start_tile(kv_ref, line_ref, sem_ref):
        if gathered_refs is not None:
            with sc("walk_issue"):
                return start_tile_gathered(kv_ref, line_ref, sem_ref)
        if runs_refs is not None:
            with sc("walk_issue"):
                return start_tile_run_tables(kv_ref, line_ref, sem_ref)
        if listed_refs is not None:
            with sc("walk_issue"):
                return start_tile_listed(kv_ref, line_ref, sem_ref)
        if fast_walk:
            with sc("walk_issue"):
                if walk_runs:
                    return start_tile_runs(kv_ref, line_ref, sem_ref)
                return start_tile_fast(kv_ref, line_ref, sem_ref)

        def issue(step, carry):
            for offset in range(_ISSUE_UNROLL):
                index = step * _ISSUE_UNROLL + offset
                rank = state_ref[2]
                live = rank < usize
                word, bit = next_set_bit()
                # Ranks past the union (the last tile's padding) copy group 0,
                # a valid address that every mask hides.
                group_id = jnp.where(live, word * 32 + bit, 0)
                token = group_id * ratio
                physical = table_ref[token // page_size] * page_size + token % page_size
                pltpu.make_async_copy(
                    cache_ref.at[pl.ds(physical, ratio)],
                    kv_ref.at[pl.ds(index * ratio, ratio)],
                    sem_ref,
                ).start()
                if line_matmul:
                    line = lax.shift_right_logical(bits_ref[0, word, :], bit) & 1
                    line_ref[index, :] = jnp.where(live, line, 0)
                else:
                    line = lax.shift_right_logical(bits_ref[0, pl.ds(word, 1), :], bit) & 1
                    line = jnp.where(live, line, 0)
                    line_ref[pl.ds(index * ratio, ratio), :] = jnp.broadcast_to(line, (ratio, width))
                state_ref[2] = rank + 1
            return carry

        lax.fori_loop(0, groups_per_tile // _ISSUE_UNROLL, issue, None)

    def sc(name):
        # in-kernel phase markers (P-ATTN profile arm): Mosaic trace scopes, visible on the "XLA TraceMe" line only
        # with LIBTPU_INIT_ARGS=--xla_enable_custom_call_region_trace=true. Off (0) = the same program.
        return jax.named_scope(name) if scopes else contextlib.nullcontext()

    def wait_tile(kv_ref, sem_ref, line_ref=None):
        with sc("dma_wait"):
            pltpu.make_async_copy(cache_ref.at[pl.ds(0, kv_tile)], kv_ref, sem_ref).wait()
            lines_hbm = gathered_refs[1] if gathered_refs is not None else listed_lines_ref
            if lines_hbm is not None:
                rows = line_ref.shape[0]
                pltpu.make_async_copy(lines_hbm.at[pl.ds(0, rows)], line_ref, sem_ref).wait()

    def split_kv(kv_ref):
        cache_words = kv_ref.bitcast(jnp.uint32).reshape(-1, head_dim)
        pairs = []
        head_axes = kv_ref.shape[1]
        for head_axis in range(head_axes):
            words_ = utils.strided_load(cache_words, head_axis, kv_tile * head_axes, head_axes)
            pairs.extend(
                utils.convert_to_target_bitwidth(
                    words_, target_bitwidth=32 // packing, kv_dtype=kv_ref.dtype
                )
            )
        return pairs

    def compute_tile(tile, kv_ref, line_ref):
        logical = tile * kv_tile
        mask_scope = sc("mask")
        mask_scope.__enter__()
        if line_matmul:
            # line_ref: [groups, width rows]. Expand groups to their 4 keys exactly.
            lines = line_ref[...].T[:query_block, :].astype(jnp.float32)
            expand = (
                lax.broadcasted_iota(jnp.int32, (groups_per_tile, kv_tile), 1) // ratio
                == lax.broadcasted_iota(jnp.int32, (groups_per_tile, kv_tile), 0)
            ).astype(jnp.float32)
            visible = lax.dot(lines, expand, preferred_element_type=jnp.float32) != 0
        else:
            # line_ref: [kv_tile keys, width rows]; the rows' mask over the tile's keys.
            visible = line_ref[...].T[:query_block, :] != 0
        visible &= logical + row[None, :] <= positions[:, None]
        written = logical + lax.broadcasted_iota(jnp.int32, (kv_tile, head_dim), 0) < jnp.max(
            lengths_ref[:, 0]
        )
        mask_scope.__exit__(None, None, None)
        decode_scope = sc("decode")
        decode_scope.__enter__()
        key_values = []
        for key, value in split_kv(kv_ref)[:kv_heads]:
            key = key.astype(jnp.bfloat16 if qk16 else jnp.float32)  # FP8 E4M3 is exact in BF16
            value = value.astype(jnp.float32)
            if not lean:
                key = jnp.where(jnp.isfinite(key) & written, key, 0)
            value = jnp.where(jnp.isfinite(value) & written, value, 0)
            key_values.append((key, value))
        decode_scope.__exit__(None, None, None)
        rows = query_block * q_heads_per_kv
        with sc("mask_bcast"):
            row_visible = jnp.broadcast_to(
                visible[None, :, :], (q_heads_per_kv, query_block, kv_tile)
            ).reshape(rows, kv_tile)
        for head, (key, value) in enumerate(key_values):
            m = max_ref[0, head, :]
            mass = mass_ref[0, head, :]
            acc = accumulator_ref[0, head, :, :]
            query = q[head].reshape(rows, head_dim)
            with sc("qk"):
                score = lax.dot(
                    query, key, dimension_numbers=(([1], [1]), ([], [])),
                    preferred_element_type=jnp.float32,
                )
            with sc("softmax"):
                score = jnp.where(row_visible, score, -jnp.inf)
                next_m = jnp.maximum(m, jnp.max(score, axis=1))
                finite_m = jnp.where(jnp.isneginf(next_m), 0.0, next_m)
                prob = jnp.exp(score - finite_m[:, None])
                alpha = jnp.where(jnp.isneginf(next_m), 1.0, jnp.exp(m - next_m))
            with sc("pv"):
                weighted = lax.dot(
                    prob, value, dimension_numbers=(([1], [0]), ([], [])),
                    preferred_element_type=jnp.float32,
                )
                max_ref[0, head, :] = next_m
                mass_ref[0, head, :] = mass * alpha + jnp.sum(prob, axis=1)
                accumulator_ref[0, head, :, :] = acc * alpha[:, None] + weighted

    @pl.when(count > 0)
    def prefetch_first():
        start_tile(kv0_ref, line0_ref, sem0_ref)

    @pl.when(count > 1)
    def prefetch_second():
        start_tile(kv1_ref, line1_ref, sem1_ref)

    def pair_step(pair, _):
        tile = pair * 2
        wait_tile(kv0_ref, sem0_ref, line0_ref)
        compute_tile(tile, kv0_ref, line0_ref)

        @pl.when(tile + 2 < count)
        def prefetch_next_even():
            start_tile(kv0_ref, line0_ref, sem0_ref)

        wait_tile(kv1_ref, sem1_ref, line1_ref)
        compute_tile(tile + 1, kv1_ref, line1_ref)

        @pl.when(tile + 3 < count)
        def prefetch_next_odd():
            start_tile(kv1_ref, line1_ref, sem1_ref)

        return None

    lax.fori_loop(0, count // 2, pair_step, None)

    @pl.when(count % 2 != 0)
    def consume_odd():
        wait_tile(kv0_ref, sem0_ref, line0_ref)
        compute_tile(count - 1, kv0_ref, line0_ref)

    heads = []
    for head in range(kv_heads):
        mass = mass_ref[0, head, :]
        acc = accumulator_ref[0, head, :, :]
        heads.append(
            (acc * (v_scale / jnp.where(mass == 0, 1, mass))[:, None])
            .reshape(q_heads_per_kv, query_block, head_dim)
            .transpose(1, 0, 2)
        )
    output_ref[...] = jnp.concatenate(heads, axis=1).astype(output_ref.dtype)


def _walk_kernel_per_group(
    tiles_ref,
    usize_ref,
    debruijn_ref,
    query_ref,
    bits_ref,
    lengths_ref,
    ubits_hbm_ref,
    physw_hbm_ref,
    cache_ref,
    output_ref,
    ubits_ref,
    list_sem,
    physw_ref,
    physw_sem,
    *scratch,
    **options,
):
    """``_walk_kernel`` (fast walk) with one page row per query group.

    A packed step holds several requests, so the per-word pool bases differ
    between query groups: they come from HBM, one ``words_padded`` row per
    group, staged into SMEM at block start instead of one scalar-prefetched
    table for the whole grid.
    """
    _walk_kernel(
        tiles_ref, usize_ref, None, debruijn_ref, physw_ref, query_ref, bits_ref, lengths_ref,
        ubits_hbm_ref, cache_ref, output_ref, ubits_ref, list_sem, *scratch,
        physw_hbm_ref=physw_hbm_ref, physw_sem=physw_sem, **options,
    )


def _walk_kernel_gathered(
    tiles_ref,
    usize_ref,
    table_ref,
    debruijn_ref,
    physw_ref,
    query_ref,
    bits_ref,
    lengths_ref,
    ubits_hbm_ref,
    cache_ref,
    gathered_kv_ref,
    gathered_lines_ref,
    output_ref,
    *scratch,
    **options,
):
    """``_walk_kernel`` reading pre-gathered union tiles (``union_walk_kernel(gathered=...)``)."""
    _walk_kernel(
        tiles_ref, usize_ref, table_ref, debruijn_ref, physw_ref, query_ref, bits_ref, lengths_ref,
        ubits_hbm_ref, cache_ref, output_ref, *scratch,
        gathered_refs=(gathered_kv_ref, gathered_lines_ref), **options,
    )


def _walk_kernel_listed(
    tiles_ref,
    usize_ref,
    table_ref,
    debruijn_ref,
    physw_ref,
    query_ref,
    bits_ref,
    lengths_ref,
    ubits_hbm_ref,
    cache_ref,
    ids_hbm_ref,
    output_ref,
    *scratch,
    **options,
):
    """``_walk_kernel`` issuing from a precomputed union id list (``union_walk_kernel(listed=...)``)."""
    *scratch, ids_ref, ids_sem = scratch
    _walk_kernel(
        tiles_ref, usize_ref, table_ref, debruijn_ref, physw_ref, query_ref, bits_ref, lengths_ref,
        ubits_hbm_ref, cache_ref, output_ref, *scratch,
        listed_refs=(ids_hbm_ref, ids_ref, ids_sem), **options,
    )


def _walk_kernel_listed_lines(
    tiles_ref,
    usize_ref,
    table_ref,
    debruijn_ref,
    physw_ref,
    query_ref,
    bits_ref,
    lengths_ref,
    ubits_hbm_ref,
    cache_ref,
    ids_hbm_ref,
    lines_hbm_ref,
    output_ref,
    *scratch,
    **options,
):
    """``_walk_kernel_listed`` with each tile's mask lines read as one copy (``listed_lines``)."""
    *scratch, ids_ref, ids_sem = scratch
    _walk_kernel(
        tiles_ref, usize_ref, table_ref, debruijn_ref, physw_ref, query_ref, bits_ref, lengths_ref,
        ubits_hbm_ref, cache_ref, output_ref, *scratch,
        listed_refs=(ids_hbm_ref, ids_ref, ids_sem), listed_lines_ref=lines_hbm_ref, **options,
    )


def _walk_kernel_run_tables(
    tiles_ref,
    usize_ref,
    table_ref,
    debruijn_ref,
    physw_ref,
    query_ref,
    bits_ref,
    lengths_ref,
    ubits_hbm_ref,
    cache_ref,
    counts_hbm_ref,
    entries_hbm_ref,
    lines_hbm_ref,
    output_ref,
    *scratch,
    **options,
):
    """``_walk_kernel`` issuing each tile from run tables (``union_walk_kernel(run_tables=...)``)."""
    *scratch, counts_ref, entries_ref, runs_sem = scratch
    _walk_kernel(
        tiles_ref, usize_ref, table_ref, debruijn_ref, physw_ref, query_ref, bits_ref, lengths_ref,
        ubits_hbm_ref, cache_ref, output_ref, *scratch,
        runs_refs=(counts_hbm_ref, entries_hbm_ref, counts_ref, entries_ref, runs_sem),
        listed_lines_ref=lines_hbm_ref, **options,
    )


def _walk_kernel_per_group_listed(
    tiles_ref,
    usize_ref,
    debruijn_ref,
    query_ref,
    bits_ref,
    lengths_ref,
    ubits_hbm_ref,
    physw_hbm_ref,
    cache_ref,
    ids_hbm_ref,
    output_ref,
    ubits_ref,
    list_sem,
    physw_ref,
    physw_sem,
    *scratch,
    **options,
):
    """``_walk_kernel_per_group`` issuing from a precomputed union id list."""
    *scratch, ids_ref, ids_sem = scratch
    _walk_kernel(
        tiles_ref, usize_ref, None, debruijn_ref, physw_ref, query_ref, bits_ref, lengths_ref,
        ubits_hbm_ref, cache_ref, output_ref, ubits_ref, list_sem, *scratch,
        physw_hbm_ref=physw_hbm_ref, physw_sem=physw_sem, listed_refs=(ids_hbm_ref, ids_ref, ids_sem), **options,
    )


def union_gather_list_reference(ubits, usize, page_rows, *, ratio, page_size, union_cap, kv_tile):
    """XLA reference of the SparseCore list step (tests and microbench staging; slow XLA ops).

    Per query group, its union's groups in rank order (ascending group id, as
    the walk visits them), padded to whole KV tiles: ``rows`` int32[ngroups,
    cap_ranks], each rank's first pool token row; ``ids`` int32[ngroups,
    cap_ranks], its group id or -1. Padding ranks take word 0's first row, as
    the walk's padding copies do. ``page_rows``: one page row for every group,
    or one per query group.
    """
    ngroups, words = ubits.shape
    groups_per_tile = kv_tile // ratio
    cap_ranks = _cdiv(union_cap * ratio, kv_tile) * groups_per_tile
    physw = _word_bases(page_rows, words=words, page_size=page_size, word_tokens=32 * ratio)
    physw = jnp.broadcast_to(physw, (ngroups, words))
    unsigned = lax.bitcast_convert_type(ubits.astype(jnp.int32), jnp.uint32)
    set_bits = lax.shift_right_logical(unsigned[..., None], jnp.arange(32, dtype=jnp.uint32)[None, None, :]) & 1
    ids = jax.vmap(lambda f: jnp.nonzero(f, size=cap_ranks, fill_value=-1)[0])(
        set_bits.reshape(ngroups, words * 32)
    ).astype(jnp.int32)
    live = (ids >= 0) & (jnp.arange(cap_ranks, dtype=jnp.int32)[None, :] < usize[:, None])
    safe = jnp.where(live, ids, 0)
    rows = jnp.where(
        live, jnp.take_along_axis(physw, safe // 32, axis=1) + (safe % 32) * ratio, physw[:, :1]
    )
    return rows.astype(jnp.int32), jnp.where(live, ids, -1)


def union_gather_reference(pool, bits, rows, ids, *, ratio, line_matmul):
    """XLA reference of the SparseCore gather step: ``(gathered KV, gathered lines)``.

    Gathered KV: each rank's ``ratio`` pool tokens, flat [ngroups * cap_ranks *
    ratio, head_axis, packing, dim]. Lines: each rank's per-row selection bits
    from the plan's bitset (0 for padding), one line per rank (``line_matmul``)
    or per token, flat [rows, width] int32: the walk's mask lines.
    """
    pages, page_size, head_axis, packing, dim = pool.shape
    flat = pool.reshape(pages * page_size, head_axis, packing, dim)
    tokens = (rows[..., None] + jnp.arange(ratio, dtype=jnp.int32)).reshape(-1)
    gathered_kv = flat[tokens]
    return gathered_kv, union_lines_reference(bits, ids, ratio=ratio, line_matmul=line_matmul)


def union_lines_reference(bits, ids, *, ratio, line_matmul):
    """XLA reference of the SparseCore lines-only step: each rank's mask line, flat [rows, width] int32.

    Bits ``(bits[g, id // 32, lane] >> id % 32) & 1`` of the plan's per-row bitset, 0 for padding (id -1), one line
    per rank (``line_matmul``) or per token: the walk's lines, and ``union_gather_reference``'s.
    """
    live = ids >= 0
    safe = jnp.where(live, ids, 0)
    word_rows = jnp.take_along_axis(bits, (safe // 32)[..., None], axis=1)
    lines = jnp.where(live[..., None], lax.shift_right_logical(word_rows, (safe % 32)[..., None]) & 1, 0)
    if not line_matmul:
        lines = jnp.repeat(lines, ratio, axis=1)
    return lines.reshape(-1, bits.shape[-1]).astype(jnp.int32)


def union_run_tables_xla(rows, ids, *, ratio, kv_tile, max_run=8):
    """Traceable (XLA) run tables, bit for bit ``union_run_tables_reference``; the SparseCore builder's spec.

    Per tile, vector-wide: a rank joins the previous one when both are live and pool-contiguous (or both padding);
    run starts and run ends come from cumulative max / min scans; each rank's offset ``o`` in its run of length
    ``L`` places the pieces: size ``max_run`` at ``o % max_run == 0`` below ``max_run * (L // max_run)``, then the
    remainder ``rem = L % max_run`` in descending powers of two, the piece of size ``s`` (a bit of ``rem``) at offset
    ``rem & ~(2s - 1)`` past the full pieces. Each size's piece starts are compacted in rank order behind the larger
    sizes' (a cumulative count per size), entry = (pool group << 9) | slot.
    """
    if max_run & (max_run - 1) or not 1 <= max_run <= 32:
        raise ValueError("max_run must be a power of two from 1 to 32")
    ngroups, cap_ranks = rows.shape
    gpt = kv_tile // ratio
    if cap_ranks % gpt or gpt > 512:
        raise ValueError("the list must hold whole tiles of at most 512 groups")
    tiles = cap_ranks // gpt
    sizes = [1 << k for k in reversed(range(max_run.bit_length()))]
    rows_t = rows.astype(jnp.int32).reshape(ngroups, tiles, gpt)
    live = (ids >= 0).reshape(ngroups, tiles, gpt)
    j = jnp.arange(gpt, dtype=jnp.int32)
    contiguous = rows_t[..., 1:] == rows_t[..., :-1] + ratio
    join = jnp.concatenate([jnp.zeros_like(live[..., :1]),
                            (live[..., 1:] == live[..., :-1]) & (~live[..., 1:] | contiguous)], axis=-1)
    start = ~join
    run_start = lax.cummax(jnp.where(start, j, -1), axis=2)
    after = jnp.concatenate([jnp.where(start[..., 1:], j[1:], gpt), jnp.full_like(rows_t[..., :1], gpt)], axis=-1)
    run_end = lax.cummin(after, axis=2, reverse=True)
    length = run_end - run_start
    offset = j - run_start
    full = (length // max_run) * max_run
    rem, past = length - full, offset - full
    piece = [(offset < full) & (offset % max_run == 0)]
    for size in sizes[1:]:
        piece.append(((rem & size) != 0) & (past == (rem & ~((size << 1) - 1))))
    entry = (lax.shift_right_logical(rows_t, int(np.log2(ratio))) << 9) | j
    counts = jnp.stack([jnp.sum(p, axis=-1, dtype=jnp.int32) for p in piece], axis=-1)
    before = jnp.cumsum(counts, axis=-1) - counts
    slot = jnp.full(rows_t.shape, gpt, jnp.int32)
    for c, p in enumerate(piece):
        slot = jnp.where(p, before[..., c:c + 1] + jnp.cumsum(p, axis=-1, dtype=jnp.int32) - 1, slot)
    flat_slot = (jnp.arange(ngroups * tiles, dtype=jnp.int32)[:, None] * (gpt + 1)
                 + slot.reshape(ngroups * tiles, gpt)).reshape(-1)
    entries = jnp.zeros((ngroups * tiles * (gpt + 1),), jnp.int32).at[flat_slot].set(entry.reshape(-1))
    entries = entries.reshape(ngroups * tiles, gpt + 1)[:, :gpt].reshape(ngroups, cap_ranks)
    return counts, entries


def union_run_tables_reference(rows, ids, *, ratio, kv_tile, max_run=8):
    """Host (numpy) reference of the SparseCore run-table step: ``(counts, entries)`` for ``run_tables=``.

    ``rows``/``ids``: ``union_gather_list_reference``'s. Per query group and KV tile: maximal runs of consecutive
    ranks whose pool rows are consecutive (union groups that are one contiguous stretch of the pool), cut into
    power-of-two pieces of at most ``max_run`` groups, largest first. Padding ranks (id -1) form their own runs,
    each piece sourced from word 0's first rows (the rows the walk's padding copies read; their lines are 0).
    ``counts`` int32[ngroups, cap_tiles, log2(max_run) + 1]: pieces per size, sizes descending. ``entries``
    int32[ngroups, cap_tiles * groups_per_tile]: per tile, each size's pieces in rank order, sizes descending,
    entry = (pool group << 9) | first tile slot.
    """
    rows, ids = np.asarray(rows), np.asarray(ids)
    if max_run & (max_run - 1) or not 1 <= max_run <= 32:
        raise ValueError("max_run must be a power of two from 1 to 32")
    ngroups, cap_ranks = rows.shape
    gpt = kv_tile // ratio
    if cap_ranks % gpt or gpt > 512:
        raise ValueError("the list must hold whole tiles of at most 512 groups")
    if int(rows.max(initial=0)) // ratio >= 1 << 22:
        raise ValueError("pool group index past 22 bits")
    sizes = [1 << k for k in reversed(range(max_run.bit_length()))]
    tiles = cap_ranks // gpt
    counts = np.zeros((ngroups, tiles, len(sizes)), np.int32)
    entries = np.zeros((ngroups, cap_ranks), np.int32)
    for g in range(ngroups):
        for t in range(tiles):
            r0 = t * gpt
            rr, ii = rows[g, r0:r0 + gpt], ids[g, r0:r0 + gpt]
            pieces = {s: [] for s in sizes}
            j = 0
            while j < gpt:
                live = bool(ii[j] >= 0)
                n = 1
                while (j + n < gpt and bool(ii[j + n] >= 0) == live
                       and (not live or rr[j + n] == rr[j] + n * ratio)):
                    n += 1
                k = 0
                for s in sizes:
                    while n - k >= s:
                        src = int(rr[j]) + (k * ratio if live else 0)
                        pieces[s].append(((src // ratio) << 9) | (j + k))
                        k += s
                j += n
            at = 0
            for c, s in enumerate(sizes):
                counts[g, t, c] = len(pieces[s])
                entries[g, r0 + at:r0 + at + len(pieces[s])] = pieces[s]
                at += len(pieces[s])
    return counts, entries


def _word_bases(page_rows, *, words, page_size, word_tokens):
    """int32[..., words]: pool row of each 32-group word's first token, per page row."""
    word_start = jnp.arange(words, dtype=jnp.int32) * word_tokens
    pages_per_row = page_rows.shape[-1]
    return (
        page_rows.astype(jnp.int32)[..., jnp.minimum(word_start // page_size, pages_per_row - 1)]
        * page_size
        + word_start % page_size
    )


def union_walk_kernel(query, bits, ubits, usize, vlengths, pool, page_table_row, *, group,
                      union_cap, kv_tile, ratio=4, k_scale=1.0, v_scale=1.0, lean=False,
                      scan_list=False, line_matmul=False, fast_walk=False, scopes=0, walk_runs=0,
                      qk16=False, gathered=None, listed=None, listed_lines=None, run_tables=None):
    """The walking kernel alone, on a precomputed ``union_bits_plan`` (ungated bf16 output).

    ``gathered`` (SparseCore union gather, step 1): ``(gathered KV, gathered
    lines)`` from ``union_gather_reference`` (or the SparseCore kernels, same
    layout). The kernel then reads each tile as two contiguous copies instead of
    walking the union; ``page_table_row`` only fed the gather. Bitwise the walk.
    ``listed`` (SparseCore list only): int32[ngroups, cap_ranks] union group ids
    in rank order, -1 padding (``union_gather_list_reference``'s ``ids``, or
    ``union_sc_list``). The kernel issues the walk's per-group copies and lines
    from the list (no bit scan, no word advance), with no extra HBM traffic.
    Bitwise the walk.

    ``page_table_row``: int32[pages_per_seq], one request's page row for every
    query group; or int32[T // group, pages_per_seq], each query group's own
    page row (a packed multi-request step whose requests start on group
    boundaries; needs ``fast_walk``).
    ``walk_runs`` (needs ``fast_walk``; 0 = off): one DMA per run of up to
    this many consecutive union groups of one bitset word (a power of two,
    2-16) instead of one per group; the same tiles and lines.
    ``qk16`` (needs ``fast_walk``): QK as one BF16 MXU pass (see ``_walk_kernel``);
    only for a power-of-two score scale ``head_dim ** -0.5 * k_scale``, where the
    pre-scaled BF16 queries stay BF16 values (magnitudes below 2**-126 after
    scaling excepted, where FP32 itself turns subnormal).
    """
    if qk16 and (not fast_walk or math.frexp(query.shape[-1] ** -0.5 * k_scale)[0] != 0.5):
        raise ValueError("qk16 needs the fast walk and a power-of-two score scale (exactness)")
    if walk_runs and (not fast_walk or walk_runs & (walk_runs - 1) or not 2 <= walk_runs <= 16):
        raise ValueError("walk_runs needs the fast walk and a power of two from 2 to 16")
    if gathered is not None and (walk_runs or scan_list or not fast_walk):
        raise ValueError("gathered tiles replace the fast walk (no walk_runs, no scan_list)")
    if listed is not None and (walk_runs or scan_list or not fast_walk or gathered is not None):
        raise ValueError("a listed walk replaces the fast walk's bit scan (no walk_runs, scan_list or gathered)")
    if run_tables is not None and page_table_row.ndim == 2:
        page_table_row = page_table_row[0]  # unused by the kernel: the run tables carry absolute pool rows
    if listed_lines is not None and ((listed is None and run_tables is None) or page_table_row.ndim == 2):
        raise ValueError("listed_lines needs a listed walk or run tables on one page row")
    if run_tables is not None and (listed_lines is None or listed is not None or gathered is not None or walk_runs
                                   or scan_list or not fast_walk):
        raise ValueError("run tables need listed_lines and replace the listed walk, the gather and walk_runs")
    tokens, query_heads, head_dim = query.shape
    pages, page_size, head_axis, packing, cache_dim = pool.shape
    kv_heads = 2 if query_heads == 24 else 1
    ngroups, words = ubits.shape
    words_padded = _cdiv(words, _LIST_ALIGN) * _LIST_ALIGN
    cap_tiles = _cdiv(union_cap * ratio, kv_tile)
    tiles = jnp.minimum(_cdiv(usize * ratio, kv_tile), cap_tiles).astype(jnp.int32)
    q_heads_per_kv = query_heads // kv_heads
    state_shape = (1, kv_heads, group * q_heads_per_kv)
    width = bits.shape[2]
    word_tokens = 32 * ratio
    if fast_walk and page_size % word_tokens:
        raise ValueError("the fast walk needs pages that hold whole 32-group words")
    if gathered is not None and page_table_row.ndim == 2:
        page_table_row = page_table_row[0]  # unused by the kernel: the gather already resolved pages
    if page_table_row.ndim == 2:
        if not fast_walk or scan_list or page_table_row.shape[0] != ngroups:
            raise ValueError("per-group page rows need the fast walk and one row per query group")
        return _union_walk_kernel_per_group(
            query, bits, ubits, tiles, usize, vlengths, pool, page_table_row, group=group,
            kv_tile=kv_tile, ratio=ratio, k_scale=k_scale, v_scale=v_scale, lean=lean,
            line_matmul=line_matmul, words_padded=words_padded, walk_runs=walk_runs, qk16=qk16,
            listed=listed, cap_tiles=cap_tiles,
        )
    # Pool row of each word's first token (a word never straddles a page here).
    word_start = jnp.arange(words, dtype=jnp.int32) * word_tokens
    physw = (
        page_table_row.astype(jnp.int32)[
            jnp.minimum(word_start // page_size, page_table_row.shape[0] - 1)
        ]
        * page_size
        + word_start % page_size
    )
    ids_extra, ids_scratch, ids_args = _listed_operands(listed, ngroups, cap_tiles * (kv_tile // ratio))
    run_sizes = ()
    if run_tables is not None:
        counts, entries = run_tables
        run_sizes = tuple(1 << k for k in reversed(range(counts.shape[-1])))
        groups_tile = kv_tile // ratio
        if tuple(counts.shape) != (ngroups, cap_tiles, len(run_sizes)) or tuple(entries.shape) != (
                ngroups, cap_tiles * groups_tile):
            raise ValueError(f"run tables {counts.shape}/{entries.shape} do not match {ngroups} groups x "
                             f"{cap_tiles} tiles (union_run_tables_reference)")
        counts_w = _cdiv(cap_tiles * len(run_sizes), _LIST_ALIGN) * _LIST_ALIGN
        entries_w = _cdiv(cap_tiles * groups_tile, _LIST_ALIGN) * _LIST_ALIGN
        counts_flat = jnp.pad(jnp.asarray(counts).reshape(ngroups, -1).astype(jnp.int32),
                              ((0, 0), (0, counts_w - cap_tiles * len(run_sizes)))).reshape(-1)
        entries_flat = jnp.pad(jnp.asarray(entries).astype(jnp.int32),
                               ((0, 0), (0, entries_w - entries.shape[1]))).reshape(-1)
        ids_extra = [pl.BlockSpec(memory_space=pltpu.HBM)] * 2
        ids_scratch = [pltpu.SMEM((counts_w,), jnp.int32), pltpu.SMEM((entries_w,), jnp.int32),
                       pltpu.SemaphoreType.DMA]
        ids_args = [counts_flat, entries_flat]
    return pl.pallas_call(
        functools.partial(
            _walk_kernel, ratio=ratio, kv_tile=kv_tile, head_dim=head_dim, sm_scale=head_dim**-0.5,
            packing=packing, kv_heads=kv_heads, q_heads_per_kv=q_heads_per_kv, query_block=group,
            k_scale=k_scale, v_scale=v_scale, words=words, words_padded=words_padded,
            page_size=page_size, lean=lean, scan_list=scan_list, line_matmul=line_matmul,
            fast_walk=fast_walk, scopes=scopes, walk_runs=walk_runs, qk16=qk16,
        ) if gathered is None and listed is None and run_tables is None else functools.partial(
            _walk_kernel_run_tables, ratio=ratio, kv_tile=kv_tile, head_dim=head_dim, sm_scale=head_dim**-0.5,
            packing=packing, kv_heads=kv_heads, q_heads_per_kv=q_heads_per_kv, query_block=group,
            k_scale=k_scale, v_scale=v_scale, words=words, words_padded=words_padded,
            page_size=page_size, lean=lean, line_matmul=line_matmul, fast_walk=True, scopes=scopes, qk16=qk16,
            gathered_cap_tiles=cap_tiles, run_sizes=run_sizes,
        ) if run_tables is not None else functools.partial(
            _walk_kernel_listed if listed_lines is None else _walk_kernel_listed_lines, ratio=ratio, kv_tile=kv_tile, head_dim=head_dim, sm_scale=head_dim**-0.5,
            packing=packing, kv_heads=kv_heads, q_heads_per_kv=q_heads_per_kv, query_block=group,
            k_scale=k_scale, v_scale=v_scale, words=words, words_padded=words_padded,
            page_size=page_size, lean=lean, line_matmul=line_matmul, fast_walk=True, scopes=scopes, qk16=qk16,
            **({"gathered_cap_tiles": cap_tiles} if listed_lines is not None else {}),
        ) if gathered is None else functools.partial(
            _walk_kernel_gathered, ratio=ratio, kv_tile=kv_tile, head_dim=head_dim,
            sm_scale=head_dim**-0.5, packing=packing, kv_heads=kv_heads,
            q_heads_per_kv=q_heads_per_kv, query_block=group, k_scale=k_scale, v_scale=v_scale,
            words=words, words_padded=words_padded, page_size=page_size, lean=lean,
            line_matmul=line_matmul, fast_walk=True, scopes=scopes, qk16=qk16,
            gathered_cap_tiles=cap_tiles,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(ngroups,),
            in_specs=[
                pl.BlockSpec((group, query_heads, head_dim), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((1, words, width), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((group, 1), lambda b, *_: (b, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ] + ([pl.BlockSpec(memory_space=pltpu.HBM)] * 2 if gathered is not None else []) + ids_extra
            + ([pl.BlockSpec(memory_space=pltpu.HBM)] if listed_lines is not None else []),
            out_specs=pl.BlockSpec((group, query_heads, head_dim), lambda b, *_: (b, 0, 0)),
            scratch_shapes=[
                pltpu.SMEM((words_padded,), jnp.int32),
                pltpu.SemaphoreType.DMA,
                pltpu.SMEM((4,), jnp.int32),
                pltpu.SMEM((words_padded if scan_list else 8,), jnp.int32),
                pltpu.VMEM((kv_tile // ratio if line_matmul else kv_tile, width), jnp.int32),
                pltpu.VMEM((kv_tile // ratio if line_matmul else kv_tile, width), jnp.int32),
                pltpu.VMEM((kv_tile, head_axis, packing, head_dim), pool.dtype),
                pltpu.VMEM((kv_tile, head_axis, packing, head_dim), pool.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM((*state_shape, head_dim), jnp.float32),
            ] + ids_scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=PAIRED_PAGE_VMEM_MIB.get(group, max(PAIRED_PAGE_VMEM_MIB.values()))
            * 1024**2,
            disable_bounds_checks=True,
        ),
        out_shape=jax.ShapeDtypeStruct(query.shape, jnp.bfloat16),
        name=f"qsa_union_walk_g{group}_kv{kv_tile}" + ("_lean" if lean else "")
        + ("_list" if scan_list else "") + ("_mm" if line_matmul else "") + ("_fast" if fast_walk else "")
        + ("_sc" if scopes else "") + (f"_runs{walk_runs}" if walk_runs else "")
        + ("_gathered" if gathered is not None else "") + ("_listed" if listed is not None else "")
        + ("_lines" if listed_lines is not None else "") + ("_runs" if run_tables is not None else ""),
    )(
        tiles,
        usize.astype(jnp.int32),
        page_table_row.astype(jnp.int32),
        jnp.asarray(_DEBRUIJN),
        physw,
        query.astype(jnp.bfloat16),
        bits,
        vlengths[:, None],
        jnp.pad(ubits, ((0, 0), (0, words_padded - words))).reshape(-1),
        pool.reshape(pages * page_size, head_axis, packing, head_dim),
        *(() if gathered is None else gathered),
        *ids_args,
        *(() if listed_lines is None else (listed_lines,)),
    )


def _union_walk_kernel_per_group(query, bits, ubits, tiles, usize, vlengths, pool, page_rows, *,
                                 group, kv_tile, ratio, k_scale, v_scale, lean, line_matmul,
                                 words_padded, walk_runs=0, qk16=False, listed=None, cap_tiles=0):
    """``union_walk_kernel``'s fast walk with ``page_rows[g]`` as query group g's page row."""
    tokens, query_heads, head_dim = query.shape
    pages, page_size, head_axis, packing, cache_dim = pool.shape
    kv_heads = 2 if query_heads == 24 else 1
    ngroups, words = ubits.shape
    q_heads_per_kv = query_heads // kv_heads
    state_shape = (1, kv_heads, group * q_heads_per_kv)
    width = bits.shape[2]
    # [ngroups, words_padded] in HBM; block g stages row g (a word never
    # straddles a page: pages hold whole 32-group words).
    physw = _word_bases(page_rows, words=words, page_size=page_size, word_tokens=32 * ratio)
    physw = jnp.pad(physw, ((0, 0), (0, words_padded - words))).reshape(-1)
    ids_extra, ids_scratch, ids_args = _listed_operands(listed, ngroups, cap_tiles * (kv_tile // ratio))
    return pl.pallas_call(
        functools.partial(
            _walk_kernel_per_group if listed is None else _walk_kernel_per_group_listed, ratio=ratio, kv_tile=kv_tile, head_dim=head_dim,
            sm_scale=head_dim**-0.5, packing=packing, kv_heads=kv_heads,
            q_heads_per_kv=q_heads_per_kv, query_block=group, k_scale=k_scale, v_scale=v_scale,
            words=words, words_padded=words_padded, page_size=page_size, lean=lean,
            line_matmul=line_matmul, fast_walk=True, walk_runs=walk_runs, qk16=qk16,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(ngroups,),
            in_specs=[
                pl.BlockSpec((group, query_heads, head_dim), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((1, words, width), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((group, 1), lambda b, *_: (b, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ] + ids_extra,
            out_specs=pl.BlockSpec((group, query_heads, head_dim), lambda b, *_: (b, 0, 0)),
            scratch_shapes=[
                pltpu.SMEM((words_padded,), jnp.int32),
                pltpu.SemaphoreType.DMA,
                pltpu.SMEM((words_padded,), jnp.int32),
                pltpu.SemaphoreType.DMA,
                pltpu.SMEM((4,), jnp.int32),
                pltpu.SMEM((8,), jnp.int32),
                pltpu.VMEM((kv_tile // ratio if line_matmul else kv_tile, width), jnp.int32),
                pltpu.VMEM((kv_tile // ratio if line_matmul else kv_tile, width), jnp.int32),
                pltpu.VMEM((kv_tile, head_axis, packing, head_dim), pool.dtype),
                pltpu.VMEM((kv_tile, head_axis, packing, head_dim), pool.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM((*state_shape, head_dim), jnp.float32),
            ] + ids_scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=PAIRED_PAGE_VMEM_MIB.get(group, max(PAIRED_PAGE_VMEM_MIB.values()))
            * 1024**2,
            disable_bounds_checks=True,
        ),
        out_shape=jax.ShapeDtypeStruct(query.shape, jnp.bfloat16),
        name=f"qsa_union_walk_g{group}_kv{kv_tile}" + ("_lean" if lean else "")
        + ("_mm" if line_matmul else "") + "_fast_pg" + (f"_runs{walk_runs}" if walk_runs else "")
        + ("_listed" if listed is not None else ""),
    )(
        tiles,
        usize.astype(jnp.int32),
        jnp.asarray(_DEBRUIJN),
        query.astype(jnp.bfloat16),
        bits,
        vlengths[:, None],
        jnp.pad(ubits, ((0, 0), (0, words_padded - words))).reshape(-1),
        physw,
        pool.reshape(pages * page_size, head_axis, packing, head_dim),
        *ids_args,
    )


def _listed_operands(listed, ngroups, cap_ranks):
    """(extra in_specs, extra scratch, extra args) for ``union_walk_kernel(listed=...)``: the id list in HBM,
    each query group's row padded with -1 to a whole number of SMEM list blocks, plus its SMEM copy."""
    if listed is None:
        return [], [], []
    if tuple(listed.shape) != (ngroups, cap_ranks):
        raise ValueError(f"listed ids {listed.shape} != {(ngroups, cap_ranks)} (union_gather_list_reference)")
    width = _cdiv(cap_ranks, _LIST_ALIGN) * _LIST_ALIGN
    ids = jnp.pad(listed.astype(jnp.int32), ((0, 0), (0, width - cap_ranks)), constant_values=-1).reshape(-1)
    return [pl.BlockSpec(memory_space=pltpu.HBM)], [pltpu.SMEM((width,), jnp.int32), pltpu.SemaphoreType.DMA], [ids]


@functools.partial(
    jax.jit, static_argnames=("group", "union_cap", "kv_tile", "ratio", "k_scale", "v_scale", "lean",
                              "scan_list", "line_matmul", "fast_walk", "walk_runs", "qk16")
)
def union_walk_attention(query, gate, blocks, lengths, pool, page_table_row, *, group, union_cap,
                         kv_tile, ratio=4, k_scale=1.0, v_scale=1.0, lean=False, scan_list=False,
                         line_matmul=False, fast_walk=False, walk_runs=0, qk16=False):
    """D2 with the union walked inside the kernel; same contract as ``union_dense_attention``.

    ``group`` divides 128; the selection ids follow ``union_plan_nosort``'s
    contract (below the page table's reach, distinct within a row).
    ``walk_runs``: see ``union_walk_kernel`` (needs ``fast_walk``).
    """
    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import union_bits_plan

    tokens, query_heads, head_dim = query.shape
    pages, page_size, head_axis, packing, cache_dim = pool.shape
    if kv_tile not in bsp.PAIRED_KV_TILES or ratio != 4 or page_size % ratio:
        raise ValueError("the union walk needs paired KV tiles (256-2048) and 4-token groups")
    if tokens % group or 128 % group or group < 8:
        raise ValueError("query groups must divide 128 and split T")
    if head_dim != 256 or cache_dim != head_dim or pool.dtype != jnp.float8_e4m3fn or packing != 4:
        raise ValueError("the union walk needs BF16 D256 queries and an FP8 D256 pool")
    bits, ubits, usize, vlengths = union_bits_plan(
        blocks, lengths, group=group, union_cap=union_cap,
        groups_cap=page_table_row.shape[0] * page_size // ratio, ratio=ratio,
    )
    output = union_walk_kernel(
        query, bits, ubits, usize, vlengths, pool, page_table_row, group=group, union_cap=union_cap,
        kv_tile=kv_tile, ratio=ratio, k_scale=k_scale, v_scale=v_scale, lean=lean,
        scan_list=scan_list, line_matmul=line_matmul, fast_walk=fast_walk, walk_runs=walk_runs,
        qk16=qk16,
    )
    output = bf16(
        bf16(output).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
    )
    return jnp.where((lengths > 0)[:, None, None], output, 0), usize


# ---------------------------------------------------------------------------
# Served-path integration (VLLM_TPU_QSA_PREFILL_UNION_WALK, default off): the
# fast union walk on a packed prefill step, with the served block-sparse
# kernel as the fallback.


def _served_prefill(query, gate, blocks, lengths, table, cache, starts, *, kv_heads, ratio,
                    k_scale, v_scale):
    """The call ``qsa_rpa._prefill_attention`` makes without DCP, KV head offset or LSE."""
    query_block, kv_tile, query_tile = get_sparse_prefill_tiles(
        query, cache, kv_heads=kv_heads, ratio=ratio, with_query_tile=True
    )
    return bsp.block_sparse_gqa_attention(
        query, gate, blocks, lengths, table, cache, starts, kv_heads=kv_heads, ratio=ratio,
        query_block=query_block, kv_tile=kv_tile, k_scale=k_scale, v_scale=v_scale,
        **({"query_tile": query_tile} if query_tile else {}),
    )


def union_walk_prefill(query, gate, blocks, lengths, table, cache, starts, *, kv_heads, ratio,
                       k_scale=1.0, v_scale=1.0, group=128, kv_tile=1024, union_cap, min_context,
                       lean, walk_runs=0, union_alpha=0.0, share_bits=False, qk16=False, listed=None,
                       run_tables=False, max_run=8, lazy_share=False, gathered=None, gather_chunks=4):
    """D2 union-walk attention for a packed prefill step; ``(gated bf16 output, cache)``.

    Same contract as ``block_sparse_gqa_attention`` without ``return_lse``.
    ``table`` holds one page row per query row and ``starts`` the request
    boundaries. As in its ``run_with_padding``, each request starts on a
    ``group``-row boundary of an aligned layout (padding rows have length 0),
    so no query group spans two requests and each group walks its own
    request's page row; outputs return to the original row order.

    Falls back (``lax.cond``) to the served kernel call of
    ``qsa_rpa._prefill_attention`` when the step's longest row is shorter
    than ``min_context`` (the plan is then skipped) or when any group's union
    exceeds ``union_cap`` groups (counted from the plan; the walk kernel is
    then skipped). The cache passes through untouched and is never returned
    from a conditional.

    ``walk_runs`` (0 = off): ``union_walk_kernel``'s run coalescing, one DMA
    per run of up to this many consecutive union groups; same output.

    ``union_alpha`` (0 = off): also fall back when any group's union exceeds
    this fraction of the group's visible groups (its longest row's length /
    ratio, rounded up): near-dense unions do almost the served kernel's work
    plus the walk's per-group cost, so the served kernel is faster there.

    ``run_tables`` (with ``listed``): step 4, each tile issued from run tables (pool-contiguous runs as per-size
    piece lists, pieces of at most ``max_run`` groups) with precomputed lines, built by the XLA builders
    (``listed="reference"``) or the SparseCore kernels (``"sparsecore"``). Bitwise the walk.

    ``listed`` (None, "reference" or "sparsecore"): the listed walk
    (``union_walk_kernel(listed=...)``) on the union id list built by the XLA
    reference or the SparseCore list kernel (``union_sc_list``; TPU only).
    Bitwise the walk.

    ``share_bits`` (any group dividing the served query block): align requests to the served kernel's query
    block instead of ``group``, so the walk's aligned layout is exactly the
    served call's, and let a cap/alpha fallback run the served prefill kernel
    on that layout with the plan's selection bitset (regrouped to the served
    blocks) instead of building it again. The fallback output is bitwise the
    served call's; the walk's per-request output is unchanged. Precondition:
    no row selects its own partial tail group (the indexer selects complete
    groups; none in the Job B dumps). For such a row the plan drops the
    duplicate tail entry while the served bitset adds that bit twice.
    A step whose served tiles are not paired 128/256-row tiles the group
    divides (e.g. Q256/KV128) runs as with share_bits off, with a one-time
    warning per (rows, group, tiles).

    ``lazy_share`` (with ``share_bits``): the walk aligns requests to ``group``, not the served query block, so a
    step whose requests are already group-aligned keeps the identity layout (i1s, M: share_bits' 256-row alignment
    cost 0.83-1.20 ms on 128-aligned steps). Only a cap/alpha fallback builds the served-aligned layout, with the
    plan's bitset moved to it and regrouped (``_lazy_served``); the served call still skips its own bitset build.
    The fallback output is bitwise the served call's; the walk is bitwise the unshared walk.

    ``gathered`` (None, "reference" or "sparsecore"; with ``listed``): the walk-free kernel on gathered union
    tiles, in ``gather_chunks`` chunks of whole query groups. Each chunk's tiles are gathered (the SparseCore
    gather with the dead-tile skip, or the XLA reference gather) from the list the listed walk would use, then
    attended; a chunk's gather does not depend on the previous chunk's attention, so the SparseCore gather overlaps
    TensorCore attention (i1s, M: 74% of the gather hidden). i1t (M): 0.755-0.803x the int-2 stack on 8 packed
    steps, bitwise. Bitwise the walk.

    ``qk16``: ``union_walk_kernel``'s single-pass BF16 QK (exact operands;
    power-of-two score scales only). Bitwise identity with the FP32 dot
    depends on the MXU's accumulation order and is a TPU measurement.
    """
    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import union_bits_plan

    tokens, query_heads, head_dim = query.shape
    pages, page_size, head_axis, packing, cache_dim = cache.shape
    if kv_heads != 2 or query_heads != 24 or head_dim != 256 or cache_dim != head_dim:
        raise ValueError("the union walk prefill needs 24 query heads, 2 KV heads and D256")
    if cache.dtype != jnp.float8_e4m3fn or packing != 4 or page_size % 128:
        raise ValueError("the union walk prefill needs an FP8 pool of 128-aligned pages")
    if kv_tile not in bsp.PAIRED_KV_TILES or ratio != 4:
        raise ValueError("the union walk needs paired KV tiles (256-2048) and 4-token groups")
    if 128 % group or group < 8:
        raise ValueError("query groups must divide 128")
    if listed not in (None, "reference", "sparsecore"):
        raise ValueError("listed: None, 'reference' (XLA list) or 'sparsecore' (union_sc_list)")
    if run_tables and listed is None:
        raise ValueError("run_tables needs listed='reference' or 'sparsecore' (the builders' source)")
    if gathered not in (None, "reference", "sparsecore"):
        raise ValueError("gathered: None, 'reference' (XLA gather) or 'sparsecore' (union_sc_gather)")
    if gathered and (listed is None or run_tables or int(gather_chunks) < 1):
        raise ValueError("gathered needs listed='reference' or 'sparsecore' (the list), no run_tables, chunks >= 1")
    groups_cap = table.shape[1] * page_size // ratio
    served_options = dict(kv_heads=kv_heads, ratio=ratio, k_scale=k_scale, v_scale=v_scale)
    # Trace-time record for the host-side D2 engagement count (VLLM_TPU_QSA_PREFILL_UNION_COUNT): the walk is
    # compiled into programs for steps of this many rows, with this minimum context. Host state, no graph change.
    _WALK_TRACED[int(tokens)] = int(min_context)
    if share_bits:
        served_qb, served_kt, served_qt = get_sparse_prefill_tiles(
            query, cache, kv_heads=kv_heads, ratio=ratio, with_query_tile=True
        )
        if served_qb % 128 or served_qb > 256 or served_qb % group or served_kt not in bsp.PAIRED_KV_TILES:
            # The shared fallback needs paired served tiles with a 128/256-row query block that the group divides.
            # A bucket without them (e.g. 3,200 rows on 3,200-row pages without VLLM_TPU_QSA_PREFILL_Q128_FALLBACK:
            # served Q256/KV128, unpaired) walks unshared: group-aligned rows, and a cap/alpha fallback takes the
            # outer served call. Same outputs as share_bits off; never an engine-init failure.
            key = (int(tokens), int(group), int(served_qb), int(served_kt))
            if key not in _SHARE_BITS_SKIPPED:
                _SHARE_BITS_SKIPPED.add(key)
                _share_bits_skip_log(
                    "QSA union walk: share_bits skipped for %d-row steps (group %d): served tiles Q%d/KV%d are not "
                    "paired 128/256-row tiles the group divides; the walk runs unshared and falls back to the "
                    "served call", *key)
            share_bits = False
    lazy = bool(share_bits and lazy_share)
    align = served_qb if share_bits and not lazy else group
    _WALK_ALIGN[int(tokens)] = (int(align), lazy)  # trace-time record (tests): the walk layout's alignment

    # The aligned layouts of block_sparse_gqa_attention, with align-row blocks.
    real = starts[1:] - starts[:-1]
    padded = (real + align - 1) // align * align
    offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(padded)))
    worst_tokens = (tokens + (align - 1) * (len(starts) - 1) + align - 1) // align * align
    bucket_tokens = (tokens + align - 1) // align * align
    sizes = sorted(
        {
            min(bucket_tokens + extra * align, worst_tokens)
            for extra in (0, bsp.SPARE_QUERY_BLOCKS, worst_tokens)
        }
    )

    def walk(max_tokens):
        """(ungated bf16 output in the original row order, whether every union fits)."""
        if max_tokens is None:
            # Every request already starts on a group boundary.
            layout = query, blocks, lengths, table
            back = None
        else:
            req = jnp.searchsorted(starts[1:], jnp.arange(tokens), side="right")
            req = jnp.minimum(req, len(real) - 1)
            positions = jnp.arange(tokens) - starts[req]
            valid = jnp.arange(tokens) < starts[-1]
            slots = jnp.where(valid, offsets[req] + positions, max_tokens)

            def align(array, fill):
                shape = (max_tokens, *array.shape[1:])
                return jnp.full(shape, fill, array.dtype).at[slots].set(array, mode="drop")

            layout = align(query, 0), align(blocks, -1), align(lengths, 0), align(table, 0)
            back = jnp.minimum(slots, max_tokens - 1)
        aligned_query, aligned_blocks, aligned_lengths, aligned_table = layout
        bits, ubits, usize, vlengths = union_bits_plan(
            aligned_blocks, aligned_lengths, group=group, union_cap=union_cap,
            groups_cap=groups_cap, ratio=ratio,
        )
        fits = jnp.max(usize) <= union_cap
        if union_alpha:
            visible = jnp.max(((aligned_lengths + ratio - 1) // ratio).reshape(-1, group), axis=1)
            fits &= jnp.all(usize.astype(jnp.float32) <= union_alpha * visible.astype(jnp.float32))
        # A group's rows belong to one request: its first row's page row.
        page_rows = aligned_table.reshape(-1, group, aligned_table.shape[1])[:, 0]

        def kernel():
            ids = None
            if listed == "reference":
                rows, ids = union_gather_list_reference(ubits, usize, page_rows, ratio=ratio, page_size=page_size,
                                                        union_cap=union_cap, kv_tile=kv_tile)
            elif listed == "sparsecore":
                from vllm_torchtpu.kernels.qwen4_exp.qsa_union_sc_gather import union_sc_list_pages

                rows, ids = union_sc_list_pages(ubits, usize, page_rows, ratio=ratio, page_size=page_size,
                                                union_cap=union_cap, kv_tile=kv_tile)
            if gathered:
                return _gathered_chunks(
                    aligned_query, bits, ubits, usize, vlengths, cache, page_rows, rows, ids, gathered=gathered,
                    chunks=int(gather_chunks), group=group, union_cap=union_cap, kv_tile=kv_tile, ratio=ratio,
                    k_scale=k_scale, v_scale=v_scale, lean=lean, qk16=qk16,
                )
            if run_tables:
                # Step 4: run tables + precomputed lines (XLA builders, or the SparseCore kernels).
                if listed == "reference":
                    tables = union_run_tables_xla(rows, ids, ratio=ratio, kv_tile=kv_tile, max_run=max_run)
                    lines = union_lines_reference(bits, ids, ratio=ratio, line_matmul=True)
                else:
                    from vllm_torchtpu.kernels.qwen4_exp.qsa_union_sc_gather import (
                        union_sc_lines,
                        union_sc_run_tables,
                    )

                    tables = union_sc_run_tables(rows, ids, ratio=ratio, kv_tile=kv_tile, max_run=max_run)
                    # one line per rank (the kernel expands lines to keys by a matmul): a quarter of the
                    # per-token lines' bytes for the SparseCore to write; tiles past the union's live tiles
                    # (which the kernel never reads) are skipped
                    lines = union_sc_lines(bits, ids, ratio=ratio, line_matmul=True, usize=usize, kv_tile=kv_tile)
                return union_walk_kernel(
                    aligned_query, bits, ubits, usize, vlengths, cache, page_rows, group=group,
                    union_cap=union_cap, kv_tile=kv_tile, ratio=ratio, k_scale=k_scale,
                    v_scale=v_scale, lean=lean, fast_walk=True, qk16=qk16, run_tables=tables, listed_lines=lines,
                    line_matmul=True,
                )
            return union_walk_kernel(
                aligned_query, bits, ubits, usize, vlengths, cache, page_rows, group=group,
                union_cap=union_cap, kv_tile=kv_tile, ratio=ratio, k_scale=k_scale,
                v_scale=v_scale, lean=lean, fast_walk=True, walk_runs=walk_runs, qk16=qk16, listed=ids,
            )

        if lazy:
            def served_lazy():
                return _lazy_served(
                    query, blocks, lengths, table, starts, offsets, real, bits, aligned_query.shape[0], cache,
                    group=group, ratio=ratio, kv_heads=kv_heads, k_scale=k_scale, v_scale=v_scale,
                    served_qb=served_qb, served_kt=served_kt, served_qt=served_qt, groups_cap=groups_cap,
                    identity=back is None,
                )

            raw = lax.cond(fits, kernel, served_lazy)
            return (raw if back is None else raw[back]), jnp.ones((), bool)
        if share_bits:
            def served_aligned():
                # The served call on this very layout (block_sparse_gqa_attention's aligned or padded branch),
                # with the plan's bitset regrouped from 128-row lanes to the served query blocks.
                shared = _served_block_bits(bits, groups_cap=groups_cap, ratio=ratio, served_kt=served_kt,
                                            served_qb=served_qb, group=group)
                return bsp.block_sparse_gqa_prefill(
                    aligned_query, aligned_blocks, aligned_lengths, aligned_table, cache, ratio=ratio,
                    kv_heads=kv_heads, query_block=served_qb, kv_tile=served_kt, k_scale=k_scale,
                    v_scale=v_scale, bf16_io=True, **({"query_tile": served_qt} if served_qt else {}),
                    block_bits=shared,
                )

            raw = lax.cond(fits, kernel, served_aligned)
            # Either branch produced this step's ungated output: no outer fallback needed.
            return (raw if back is None else raw[back]), jnp.ones((), bool)
        raw = lax.cond(fits, kernel, lambda: jnp.zeros(aligned_query.shape, jnp.bfloat16))
        return (raw if back is None else raw[back]), fits

    def skip():
        return jnp.zeros(query.shape, jnp.bfloat16), jnp.zeros((), bool)

    branches = [functools.partial(walk, n) for n in sizes]
    route = jnp.sum(offsets[-1] > jnp.asarray(sizes[:-1], jnp.int32))
    if tokens % align == 0:
        branches.insert(0, functools.partial(walk, None))
        route = jnp.where(jnp.all(real % align == 0), 0, route + 1)
    branches.append(skip)
    route = jnp.where(jnp.max(lengths) >= min_context, route, len(branches) - 1)
    raw, fits = lax.switch(route, branches)

    def walked():
        return bf16(
            bf16(raw).astype(jnp.float32)
            * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32)
        )

    def served():
        output, _ = _served_prefill(query, gate, blocks, lengths, table, cache, starts,
                                    **served_options)
        return output

    output = lax.cond(fits, walked, served)
    return jnp.where((lengths > 0)[:, None, None], output, 0), cache


# ---------------------------------------------------------------------------
# D2 engagement count (VLLM_TPU_QSA_PREFILL_UNION_COUNT, debug, log-only, outside the compile key).
# The union walk runs on a step when (a) union_walk_prefill was traced into the program for the step's row count
# (the static dispatch conditions held at trace time) and (b) the step's longest row reaches min_context, both known
# on the host, and (c) every query group's union fits union_cap, known only on the device. (c) is not observed here,
# so the count is "D2 eligible steps": an upper bound on walk steps, exact when no union can exceed the cap (e.g.
# short-context steps such as GSM8K's, whose unions are far below 12,288 groups).
_WALK_TRACED = {}  # step rows -> min_context, recorded when union_walk_prefill is traced
_ENGAGEMENT = {"steps": 0, "eligible": 0}


def _gathered_chunks(query, bits, ubits, usize, vlengths, cache, page_rows, rows, ids, *, gathered, chunks, group,
                     union_cap, kv_tile, ratio, k_scale, v_scale, lean, qk16):
    """``union_walk_prefill(gathered=...)``: per chunk of whole query groups, gather the chunk's union tiles from
    the list (``rows``/``ids``), then run the walk-free kernel on them; outputs concatenated in group order."""
    ngroups = ubits.shape[0]
    per = -(-ngroups // chunks)
    outs = []
    for k in range(chunks):
        g0, g1 = k * per, min(ngroups, (k + 1) * per)
        if g0 >= g1:
            break
        sl, rs = slice(g0, g1), slice(g0 * group, g1 * group)
        if gathered == "sparsecore":
            from vllm_torchtpu.kernels.qwen4_exp.qsa_union_sc_gather import union_sc_gather

            tiles = union_sc_gather(cache, bits[sl], rows[sl], ids[sl], ratio=ratio, line_matmul=True,
                                    usize=usize[sl], kv_tile=kv_tile)
        else:
            tiles = union_gather_reference(cache, bits[sl], rows[sl], ids[sl], ratio=ratio, line_matmul=True)
        outs.append(union_walk_kernel(
            query[rs], bits[sl], ubits[sl], usize[sl], vlengths[rs], cache, page_rows[sl], group=group,
            union_cap=union_cap, kv_tile=kv_tile, ratio=ratio, k_scale=k_scale, v_scale=v_scale, lean=lean,
            fast_walk=True, qk16=qk16, line_matmul=True, gathered=tiles,
        ))
    return jnp.concatenate(outs, axis=0)


_WALK_ALIGN = {}  # step rows -> (walk layout alignment, lazy_share), recorded at trace time (tests)


def _served_block_bits(bits, *, groups_cap, ratio, served_kt, served_qb, group):
    """The plan's bitset ([groups, words, 128], a group's rows on lanes 0..group-1) as the served kernel's
    ``block_bits``: its word count, with served_qb // group consecutive groups side by side per served block."""
    words_s = -(-(-(-groups_cap * ratio // served_kt)) * (served_kt // (ratio * 32)) // 128) * 128
    shared = bits[:, : min(words_s, bits.shape[1])]
    if words_s > shared.shape[1]:
        shared = jnp.pad(shared, ((0, 0), (0, words_s - shared.shape[1]), (0, 0)))
    per = served_qb // group
    return (
        shared[:, :, :group].reshape(-1, per, words_s, group).transpose(0, 2, 1, 3)
        .reshape(-1, words_s, served_qb)
    )


def _lazy_served(query, blocks, lengths, table, starts, offsets, real, bits, layout_rows, cache, *, group, ratio,
                 kv_heads, k_scale, v_scale, served_qb, served_kt, served_qt, groups_cap, identity=False):
    """``union_walk_prefill(lazy_share=True)``'s fallback: the served call on a served-aligned layout built here,
    with the walk plan's bitset (one block per group of the group-aligned layout) moved to that layout's groups
    and regrouped to served blocks; returns the ungated output in the group-aligned layout's row order.

    Both layouts start each request on a group boundary with its rows in order, so a group's rows sit on the same
    lanes in both; the served layout's extra padding groups get no bits (padding rows select nothing).

    ``identity`` (the walk ran on the original rows): when every request is also served-aligned, the original rows
    are the served layout, so the served call runs on them directly with the plan's bitset regrouped, as share_bits
    does (i1u, M: building the padded layout there cost +1.28 ms on an aligned step).
    """
    tokens = query.shape[0]
    nreq = starts.shape[0] - 1
    s_padded = (real + served_qb - 1) // served_qb * served_qb
    s_off = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(s_padded).astype(jnp.int32)))
    s_bucket = (tokens + served_qb - 1) // served_qb * served_qb
    s_worst = (tokens + (served_qb - 1) * nreq + served_qb - 1) // served_qb * served_qb
    s_spare = min(s_bucket + bsp.SPARE_QUERY_BLOCKS * served_qb, s_worst)
    ngroups_g = layout_rows // group
    row0 = jnp.arange(ngroups_g, dtype=jnp.int32) * group
    g_req = jnp.minimum(jnp.searchsorted(offsets[1:], row0, side="right"), nreq - 1)
    g_local = (row0 - offsets[g_req]) // group
    g_live = row0 < offsets[-1]
    t = jnp.arange(layout_rows, dtype=jnp.int32)
    t_req = jnp.minimum(jnp.searchsorted(offsets[1:], t, side="right"), nreq - 1)
    t_pos = t - offsets[t_req]
    t_live = (t < offsets[-1]) & (t_pos < real[t_req])
    o = jnp.arange(tokens)
    o_req = jnp.minimum(jnp.searchsorted(starts[1:], o, side="right"), nreq - 1)
    o_live = o < starts[-1]

    def run(size):
        slots = jnp.where(o_live, s_off[o_req] + o - starts[o_req], size)

        def place(array, fill):
            return jnp.full((size, *array.shape[1:]), fill, array.dtype).at[slots].set(array, mode="drop")

        target = jnp.where(g_live, s_off[g_req] // group + g_local, size // group)
        sbits = jnp.zeros((size // group, *bits.shape[1:]), bits.dtype).at[target].set(bits, mode="drop")
        out = bsp.block_sparse_gqa_prefill(
            place(query, 0), place(blocks, -1), place(lengths, 0), place(table, 0), cache, ratio=ratio,
            kv_heads=kv_heads, query_block=served_qb, kv_tile=served_kt, k_scale=k_scale, v_scale=v_scale,
            bf16_io=True, **({"query_tile": served_qt} if served_qt else {}),
            block_bits=_served_block_bits(sbits, groups_cap=groups_cap, ratio=ratio, served_kt=served_kt,
                                          served_qb=served_qb, group=group),
        )
        back = jnp.where(t_live, s_off[t_req] + t_pos, 0)
        return jnp.where(t_live[:, None, None], out[back], 0).astype(jnp.bfloat16)

    def padded():
        if s_spare == s_worst:
            return run(s_worst)
        return lax.cond(s_off[-1] <= s_spare, lambda: run(s_spare), lambda: run(s_worst))

    if not (identity and layout_rows == tokens and tokens % served_qb == 0):
        return padded()

    def direct():
        out = bsp.block_sparse_gqa_prefill(
            query, blocks, lengths, table, cache, ratio=ratio, kv_heads=kv_heads, query_block=served_qb,
            kv_tile=served_kt, k_scale=k_scale, v_scale=v_scale, bf16_io=True,
            **({"query_tile": served_qt} if served_qt else {}),
            block_bits=_served_block_bits(bits, groups_cap=groups_cap, ratio=ratio, served_kt=served_kt,
                                          served_qb=served_qb, group=group),
        )
        return out.astype(jnp.bfloat16)

    return lax.cond(jnp.all(real % served_qb == 0), direct, padded)


_SHARE_BITS_SKIPPED = set()  # (step rows, group, served query block, served KV tile): share_bits skipped, warned once


def _share_bits_skip_log(message, *args):
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).warning(message, *args)


def _engagement_log(message, *args):
    from vllm_torchtpu.logger import init_logger

    init_logger(__name__).info(message, *args)


def count_union_walk_step(tokens: int, longest: int, every: int) -> bool:
    """Count one prefill step on the host; log every ``every`` steps. Returns whether the step is D2 eligible."""
    _ENGAGEMENT["steps"] += 1
    min_context = _WALK_TRACED.get(int(tokens))
    eligible = min_context is not None and int(longest) >= min_context
    if eligible:
        _ENGAGEMENT["eligible"] += 1
    if every and _ENGAGEMENT["steps"] % every == 0:
        _engagement_log(
            "QSA union walk (D2) eligible steps: %d of %d prefill steps on this stage process "
            "(eligible = walk compiled for the step's row count and longest row >= min context; the device-side "
            "union-cap fallback is not observed; walk calls = eligible steps x QSA FA layers on the stage)",
            _ENGAGEMENT["eligible"],
            _ENGAGEMENT["steps"],
        )
    return eligible
