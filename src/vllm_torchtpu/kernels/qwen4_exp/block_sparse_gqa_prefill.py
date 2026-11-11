# SPDX-License-Identifier: Apache-2.0
"""QSA block-sparse prefill with query-block reuse across paged KV tiles."""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.experimental.batched_rpa import utils
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import PAIRED_PAGE_VMEM_MIB
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Extra query blocks of the intermediate block-aligned prefill layout, for
# steps whose aligned requests overflow the token bucket by a few blocks
# (several prefill chunks). Larger overflows take the worst-case layout.
SPARE_QUERY_BLOCKS = 4


# A Q24 Q256 block keeps 256 x 24 query-head rows of state resident; fewer
# query heads per rank (tensor-parallel shards) afford proportionally longer
# query blocks for the same online-softmax state.
PAIRED_PAGE_MAX_QUERY_ROWS = 256 * 24
# KV tiles spanning several 128-token FP8 mask segments (selection words).
PAIRED_KV_TILES = (256, 512, 1024)

# VLLM_TPU_QSA_PREFILL_VPU features (see envs.py) and the set "1" selects.
VPU_FEATURES = ("mask", "qk16", "pv16", "qkfirst")
VPU_DEFAULT = ("mask", "qk16", "pv16")
# Comma-free spellings (recipes pass server env as a comma-separated list).
VPU_PRESETS = {"1": VPU_DEFAULT, "2": VPU_DEFAULT + ("qkfirst",)}


def vpu_features():
    """Features of the reduced-VPU inner loop requested by the environment."""
    raw = envs.VLLM_TPU_QSA_PREFILL_VPU
    if raw in ("", "0"):
        return ()
    if raw in VPU_PRESETS:
        names = VPU_PRESETS[raw]
    else:
        names = tuple(name.strip() for name in raw.split(",") if name.strip())
    unknown = sorted(set(names) - set(VPU_FEATURES))
    if unknown:
        raise ValueError(f"Unknown VLLM_TPU_QSA_PREFILL_VPU features: {unknown}")
    # The per-segment bits and diagonal-only guards are the path's base.
    names = set(names) | {"mask"}
    return tuple(name for name in VPU_FEATURES if name in names)


LAZY_SOFTMAX_TOKENS = ("nobranch", "noguard")


def lazy_softmax_features():
    """VLLM_TPU_QSA_PREFILL_LAZY_SOFTMAX as a feature tuple; () when off."""
    raw = envs.VLLM_TPU_QSA_PREFILL_LAZY_SOFTMAX
    if raw in ("", "0"):
        return ()
    names = {name.strip() for name in raw.split(",") if name.strip()} - {"1"}
    unknown = sorted(names - set(LAZY_SOFTMAX_TOKENS))
    if unknown:
        raise ValueError(f"Unknown VLLM_TPU_QSA_PREFILL_LAZY_SOFTMAX tokens: {unknown}")
    return ("on",) + tuple(name for name in LAZY_SOFTMAX_TOKENS if name in names)


def default_query_tile(query_block):
    """Query subtile used when the tile table does not choose one."""
    return min(query_block, 256)


def _compact_active(active):
    """Indices of true entries along the last axis, zero filled.

    Equivalent to ``jnp.nonzero(row, size=row.shape[-1], fill_value=0)[0]``
    per row. ``jnp.nonzero`` lowers to a scatter-add with heavily repeated
    indices (a bincount of the running count), which XLA:TPU expands into a
    TensorCore sort over every mask element. A prefix sum gives each true
    entry its output slot directly; false entries target distinct
    out-of-range slots, so the scatter has unique indices and needs no sort.
    """
    size = active.shape[-1]
    lanes = jnp.arange(size, dtype=jnp.int32)
    slots = jnp.cumsum(active, axis=-1, dtype=jnp.int32) - 1
    slots = jnp.where(active, slots, size + lanes)
    compact = jnp.zeros((*active.shape[:-1], size + size), jnp.int32)
    if active.ndim == 1:
        compact = compact.at[slots].set(lanes, unique_indices=True)
    else:
        rows = jnp.arange(active.shape[0], dtype=jnp.int32)[:, None]
        compact = compact.at[rows, slots].set(
            jnp.broadcast_to(lanes, active.shape), unique_indices=True
        )
    return compact[..., :size]


def _selection_words(blocks, lengths, *, ratio, groups):
    """Per-selection mask word index and bit; invalid entries get -1 and 0.

    The selected groups are the indexer's blocks plus the partial tail group
    of a query whose length is not a multiple of ``ratio``.
    """
    tail = lengths // ratio
    selected = jnp.concatenate((blocks, tail[:, None]), axis=1)
    valid = jnp.concatenate(
        (
            (blocks >= 0) & (blocks < groups),
            ((lengths > 0) & (lengths % ratio != 0) & (tail < groups))[:, None],
        ),
        axis=1,
    )
    word = jnp.where(valid, selected // 32, -1)
    bit = jnp.where(valid, jnp.int32(1) << (selected % 32), 0)
    return word, bit


def _scatter_bitset(blocks, lengths, *, ratio, groups, words):
    """int32[tokens, words] selection bitset by a per-row scatter-add.

    XLA:TPU sorts every (row, word) index of this scatter because the
    indices repeat; this is the reference for ``_selection_bitset``.
    """
    word, bit = _selection_words(blocks, lengths, ratio=ratio, groups=groups)
    invalid = words + jnp.arange(word.shape[1], dtype=jnp.int32)
    indices = jnp.where(word >= 0, word, invalid[None, :])
    return jax.vmap(
        lambda row, weight: (
            jnp.zeros((words,), jnp.int32).at[row].add(weight, mode="drop")
        )
    )(indices, bit)


def _bitset_kernel(first_ref, end_ref, word_ref, bit_ref, out_ref):
    """Fill one query block's ``[words, width]`` bitset, eight words a step.

    Each step compares every selection of the block against eight word
    indices. Selections sit on sublanes and queries on lanes, so the
    selection reduction is elementwise until one final sublane all-reduce.
    Adding the bits reproduces the scatter-add exactly, duplicates included.
    """
    block = pl.program_id(0)
    rows, width = word_ref.shape[1:]
    out_ref[...] = jnp.zeros(out_ref.shape, jnp.int32)
    sublane = lax.broadcasted_iota(jnp.int32, (8, width), 0)

    def group(index, _):
        base = index * 8

        def accumulate(chunk, accs):
            start = pl.multiple_of(chunk * 8, 8)
            word = word_ref[0, pl.ds(start, 8), :]
            bit = bit_ref[0, pl.ds(start, 8), :]
            return tuple(
                acc + jnp.where(word == base + k, bit, 0) for k, acc in enumerate(accs)
            )

        accs = lax.fori_loop(
            0,
            rows // 8,
            accumulate,
            tuple(jnp.zeros((8, width), jnp.int32) for _ in range(8)),
            unroll=5 if (rows // 8) % 5 == 0 else 1,
        )
        slab = jnp.zeros((8, width), jnp.int32)
        for k, acc in enumerate(accs):
            for shift in (4, 2, 1):
                acc = acc + pltpu.roll(acc, shift, 0)
            slab = jnp.where(sublane == k, acc, slab)
        out_ref[0, pl.ds(pl.multiple_of(base, 8), 8), :] = slab
        return None

    lax.fori_loop(first_ref[block], end_ref[block], group, None)


def _ranged_bitset_kernel(
    first_ref, end_ref, lo_ref, hi_ref, word_ref, bit_ref, out_ref, *, groups
):
    """``_bitset_kernel`` visiting, per eight-word group, only the selection
    chunks ``[lo, hi)`` that can hold one of its words (``_chunk_ranges``).

    A chunk is eight selection rows of the block. Every chunk outside the
    range holds no word of the group, so it would only add zeros: the sums
    are the same integers, added in the same per-chunk order.
    """
    block = pl.program_id(0)
    width = word_ref.shape[2]
    out_ref[...] = jnp.zeros(out_ref.shape, jnp.int32)
    sublane = lax.broadcasted_iota(jnp.int32, (8, width), 0)

    def group(index, _):
        base = index * 8
        entry = block * groups + index

        def accumulate(chunk, accs):
            start = pl.multiple_of(chunk * 8, 8)
            word = word_ref[0, pl.ds(start, 8), :]
            bit = bit_ref[0, pl.ds(start, 8), :]
            return tuple(
                acc + jnp.where(word == base + k, bit, 0) for k, acc in enumerate(accs)
            )

        accs = lax.fori_loop(
            lo_ref[entry],
            hi_ref[entry],
            accumulate,
            tuple(jnp.zeros((8, width), jnp.int32) for _ in range(8)),
        )
        slab = jnp.zeros((8, width), jnp.int32)
        for k, acc in enumerate(accs):
            for shift in (4, 2, 1):
                acc = acc + pltpu.roll(acc, shift, 0)
            slab = jnp.where(sublane == k, acc, slab)
        out_ref[0, pl.ds(pl.multiple_of(base, 8), 8), :] = slab
        return None

    lax.fori_loop(first_ref[block], end_ref[block], group, None)


def _chunk_ranges(word, groups):
    """Per query block and eight-word group, the selection chunks to visit.

    ``word``: int32[blocks, rows, width] laid out for the bitset kernel (-1
    invalid), ``rows`` a multiple of 8. Chunk ``c`` is rows ``[8c, 8c + 8)``.
    Returns flat int32 ``(lo, hi)`` of ``[blocks * groups]``: group ``g``
    needs chunks ``[lo, hi)`` only.

    With ``M`` the running max of each chunk's largest valid word and ``m``
    the running min (from the end) of each chunk's smallest, both are
    nondecreasing, and a chunk holding a word of group ``g`` (``8g <= w <
    8g + 8``) has ``M[c] >= 8g`` and ``m[c] <= 8g + 7``: ``lo`` counts the
    chunks with ``M < 8g``, ``hi`` those with ``m <= 8g + 7``. This holds in
    any selection order; rows sorted by group id (the SparseCore top-k emits
    ascending ids) make the ranges tight.
    """
    nblocks, rows, width = word.shape
    chunks = word.reshape(nblocks, rows // 8, 8 * width)
    largest = jnp.max(chunks, axis=2)
    smallest = jnp.min(
        jnp.where(chunks >= 0, chunks, jnp.iinfo(jnp.int32).max), axis=2
    )
    running_max = lax.cummax(largest, axis=1)
    running_min = lax.cummin(smallest, axis=1, reverse=True)
    starts = (jnp.arange(groups, dtype=jnp.int32) * 8)[None, :, None]
    lo = jnp.sum(running_max[:, None, :] < starts, axis=2, dtype=jnp.int32)
    hi = jnp.sum(running_min[:, None, :] <= starts + 7, axis=2, dtype=jnp.int32)
    return lo.reshape(-1), jnp.maximum(hi, lo).reshape(-1)


def _power_of_two(value):
    return value > 0 and value & (value - 1) == 0


# Scalar-prefetched (lo, hi) chunk-range tables of the ranged bitset kernel;
# larger steps keep the full compare loop.
RANGED_BITSET_SMEM_BYTES = 256 * 1024


def _selection_bitset(
    blocks,
    lengths,
    *,
    ratio,
    groups,
    words,
    query_block,
    width,
    interpret=False,
    owned=None,
    ranged=None,
    shift_owned=False,
):
    """int32[blocks, words, width] selection bitset without a scatter.

    ``ranged`` (default VLLM_TPU_QSA_PREFILL_BITSET_RANGED): each eight-word
    group compares only the selection chunks that can hold its words
    (``_chunk_ranges``), O(selections) per block on sorted rows instead of
    O(words x selections). Bit-identical.

    ``owned=(rank, dcp_world, words_per_tile, live_words)``: keep only this
    DCP rank's words, numbered locally (``own_words`` of the global bitset).
    ``shift_owned``: number them with shifts and masks when ``words_per_tile``
    and ``dcp_world`` are powers of two (the same integers for valid words).

    Equals ``_scatter_bitset`` reshaped to query blocks, transposed to
    word-major rows and zero padded to ``width`` query lanes. Each block only
    visits the eight-word groups between its smallest and largest selected
    word, so the cost follows the context length instead of a global sort
    over every selection.
    """
    tokens = blocks.shape[0]
    nblocks = tokens // query_block
    word, bit = _selection_words(blocks, lengths, ratio=ratio, groups=groups)
    if owned is not None:
        rank, world, per_tile, live_words = owned
        if shift_owned and _power_of_two(per_tile) and _power_of_two(world):
            # Valid words are >= 0, where // and % by powers of two are shifts
            # and masks; invalid words (-1) are dropped by ``word >= 0``.
            tile_bits = per_tile.bit_length() - 1
            pair_bits = (per_tile * world).bit_length() - 1
            local = ((word >> pair_bits) << tile_bits) | (word & (per_tile - 1))
            owner = (word >> tile_bits) & (world - 1)
        else:
            local = (word // (per_tile * world)) * per_tile + word % per_tile
            owner = (word // per_tile) % world
        keep = (word >= 0) & (owner == rank) & (local < live_words)
        word = jnp.where(keep, local, -1)
        bit = jnp.where(keep, bit, 0)
    rows = (word.shape[1] + 7) // 8 * 8

    def layout(array, fill):
        array = array.reshape(nblocks, query_block, -1).transpose(0, 2, 1)
        return jnp.pad(
            array,
            ((0, 0), (0, rows - array.shape[1]), (0, width - query_block)),
            constant_values=fill,
        )

    per_block = word.reshape(nblocks, -1)
    lowest = jnp.min(jnp.where(per_block >= 0, per_block, words), axis=1)
    highest = jnp.max(per_block, axis=1)
    first = lowest // 8
    end = jnp.where(highest >= 0, highest // 8 + 1, 0)
    if ranged is None:
        ranged = envs.VLLM_TPU_QSA_PREFILL_BITSET_RANGED
    groups_per_block = words // 8
    if ranged and words % 8 == 0 and (
        2 * nblocks * groups_per_block * 4 <= RANGED_BITSET_SMEM_BYTES
    ):
        logger.info_once(
            "QSA prefill selection bitset: ranged kernel on "
            "(VLLM_TPU_QSA_PREFILL_BITSET_RANGED, %d words, %d selection rows)",
            words,
            rows,
        )
        word_layout, bit_layout = layout(word, -1), layout(bit, 0)
        lo, hi = _chunk_ranges(word_layout, groups_per_block)
        return pl.pallas_call(
            functools.partial(_ranged_bitset_kernel, groups=groups_per_block),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                grid=(nblocks,),
                in_specs=[
                    pl.BlockSpec((1, rows, width), lambda b, *_: (b, 0, 0)),
                    pl.BlockSpec((1, rows, width), lambda b, *_: (b, 0, 0)),
                ],
                out_specs=pl.BlockSpec((1, words, width), lambda b, *_: (b, 0, 0)),
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            out_shape=jax.ShapeDtypeStruct((nblocks, words, width), jnp.int32),
            interpret=interpret,
            name="qsa_prefill_selection_bitset_ranged",
        )(first, end, lo, hi, word_layout, bit_layout)
    return pl.pallas_call(
        _bitset_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(nblocks,),
            in_specs=[
                pl.BlockSpec((1, rows, width), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((1, rows, width), lambda b, *_: (b, 0, 0)),
            ],
            out_specs=pl.BlockSpec((1, words, width), lambda b, *_: (b, 0, 0)),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        out_shape=jax.ShapeDtypeStruct((nblocks, words, width), jnp.int32),
        interpret=interpret,
        name="qsa_prefill_selection_bitset",
    )(first, end, layout(word, -1), layout(bit, 0))


def _kernel(
    table_ref,
    active_tiles_ref,
    active_counts_ref,
    *refs,
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
    kv_head_offset: int = 0,
    bf16_mxu: bool = False,
    query_tile: int = 256,
    lean: bool = False,
    return_lse: bool = False,
    dcp_world: int = 1,
    vpu: tuple = (),
    page_copies: bool = False,
    lazy: tuple = (),
):
    # DCP: a fourth scalar operand holds this rank; its KV tiles are its owned
    # page-sized blocks only (local tile t is global block t * dcp_world + rank).
    rank_ref = refs[0] if dcp_world > 1 else None
    refs = refs[1:] if dcp_world > 1 else refs
    # VPU path: the next scalar operand holds each block's shortest live
    # length; KV tiles ending at or before it need no causal/written guard.
    min_length_ref = refs[0] if vpu else None
    refs = refs[1:] if vpu else refs
    query_ref, bitset_ref, lengths_ref, cache_ref, output_ref = refs[:5]
    refs = refs[5:]
    lse_ref = refs[0] if return_lse else None
    refs = refs[1 if return_lse else 0 :]
    kv0_ref, kv1_ref, sem0_ref, sem1_ref, max_ref, mass_ref, accumulator_ref = refs[:7]
    # VLLM_TPU_QSA_PREFILL_LAZY_SOFTMAX: one stabilizer per (query subtile,
    # KV head), lane-replicated; lane-partial masses; per-query "has a
    # visible key" bits for the underflow guard.
    stab_ref, partial_ref, has_key_ref = refs[7:10] if lazy else (None,) * 3
    block = pl.program_id(0)
    # The model produces BF16 queries.  A BF16 call boundary can avoid a large
    # FP32 input window while preserving the existing FP32 QK accumulation.
    q = (
        query_ref[...]
        .astype(jnp.float32)
        .reshape(query_block, kv_heads, q_heads_per_kv, head_dim)
        .transpose(1, 2, 0, 3)
    )
    # BF16 queries and BF16/FP8 KV values are exact in BF16, so one BF16 MXU
    # pass with FP32 accumulation forms the same QK products as an FP32 dot,
    # which Mosaic runs as a multi-pass FP32 matmul.
    q16 = q.astype(jnp.bfloat16) if bf16_mxu else None
    # The FP32 path folds the softmax and key scales into the queries once per
    # block instead of scaling every score of every visited KV tile.
    score_scale = sm_scale * k_scale
    # VPU path, qk16: BF16 queries are exact in BF16; a power-of-two score
    # scale keeps them exact after folding, otherwise the scores are scaled.
    scale_folded = math.frexp(score_scale)[0] == 0.5
    q16_vpu = (
        query_ref[...]
        .astype(jnp.float32)
        .reshape(query_block, kv_heads, q_heads_per_kv, head_dim)
        .transpose(1, 2, 0, 3)
        * (score_scale if scale_folded else 1.0)
    ).astype(jnp.bfloat16) if "qk16" in vpu else None
    if not bf16_mxu:
        q = q * score_scale
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
        if page_copies:
            # VLLM_TPU_QSA_PREFILL_PAGE_COPIES: tiles and pages nest, so one
            # copy per page-sized run moves the same rows in fewer DMAs.
            copy_rows = min(kv_tile, physical_page_size)
        for offset in range(0, kv_tile, copy_rows):
            # Local rows: under DCP the tiles walk this rank's own pages.
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

    # Larger query blocks reuse one resident KV tile across query subtiles.
    num_query_tiles = query_block // query_tile

    def init_state():
        # Preserve -inf initialization and all-masked-row guards exactly.
        max_ref[...] = jnp.full(max_ref.shape, -jnp.inf, jnp.float32)
        mass_ref[...] = jnp.zeros(mass_ref.shape, jnp.float32)
        accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)

    if lazy:
        stab_ref[...] = jnp.full(stab_ref.shape, -jnp.inf, jnp.float32)
        partial_ref[...] = jnp.zeros(partial_ref.shape, jnp.float32)
        has_key_ref[...] = jnp.zeros(has_key_ref.shape, jnp.int32)
        accumulator_ref[...] = jnp.zeros(accumulator_ref.shape, jnp.float32)
    else:
        init_state()

    def lazy_update(query_tile_index, head, score, value, tile_visible):
        """Online softmax step with a block-scalar stabilizer.

        ``stab`` only grows, by whole-tile maxima, and only when a tile
        exceeds it by more than LAZY_SOFTMAX_SLACK, so probabilities stay
        below e^slack and the accumulator is rescaled rarely. Masses are
        kept as 128-lane partial sums and reduced once in the epilogue.
        """
        del tile_visible
        tile_max = jnp.max(score, axis=(0, 1), keepdims=True)
        old = stab_ref[query_tile_index, head][:, :1]
        grow = tile_max > old + LAZY_SOFTMAX_SLACK
        if "nobranch" in lazy:
            new = jnp.where(grow, tile_max, old)
            alpha = jnp.where(grow, jnp.exp(old - tile_max), 1.0)
            accumulator_ref[query_tile_index, head] = (
                accumulator_ref[query_tile_index, head] * alpha
            )
            partial_ref[query_tile_index, head] = (
                partial_ref[query_tile_index, head] * alpha
            )
            stab_ref[query_tile_index, head] = jnp.broadcast_to(new, (1, 128))
        else:

            @pl.when(jnp.max(grow.astype(jnp.int32)) > 0)
            def rescale():
                # exp(-inf) = 0 clears nothing but zeros on the first raise.
                alpha = jnp.exp(old - tile_max)
                accumulator_ref[query_tile_index, head] = (
                    accumulator_ref[query_tile_index, head] * alpha
                )
                partial_ref[query_tile_index, head] = (
                    partial_ref[query_tile_index, head] * alpha
                )
                stab_ref[query_tile_index, head] = jnp.broadcast_to(
                    tile_max, (1, 128)
                )

            new = stab_ref[query_tile_index, head][:, :1]
        finite = jnp.where(jnp.isneginf(new), 0.0, new)
        prob = jnp.exp(score - finite)
        weighted = lax.dot(
            prob,
            value,
            dimension_numbers=(([1], [0]), ([], [])),
            preferred_element_type=jnp.float32,
        )
        accumulator_ref[query_tile_index, head] = (
            accumulator_ref[query_tile_index, head] + weighted
        )
        lane_sum = prob[:, :128]
        for offset in range(128, kv_tile, 128):
            lane_sum = lane_sum + prob[:, offset : offset + 128]
        partial_ref[query_tile_index, head] = (
            partial_ref[query_tile_index, head] + lane_sum
        )

    def compute_tile(tile, kv_ref, lazy_pass=False):
        logical = tile * kv_tile
        if dcp_world > 1:
            # Global token position of this owned block (masks, causality).
            logical = (tile * dcp_world + rank_ref[0]) * kv_tile
        # Dynamic scalar indexing must address a native 128-wide row.  Keep
        # query bits on that row and slice the statically known live prefix.
        # Query blocks longer than 256 rows store one bitset per 256 rows.
        parts = bitset_ref.shape[0]
        live = query_block // parts
        pieces = []
        for part in range(parts):
            tile_words = kv_tile // (ratio * 32)
            if tile_words > 1:
                # Select integer words before extracting bits. A select between
                # two bool matrices forces an unsupported Mosaic i1->i8 layout.
                word = bitset_ref[part, tile_words * tile, :live][:, None]
                for index in range(1, tile_words):
                    word = jnp.where(
                        row[None, :] < index * ratio * 32,
                        word,
                        bitset_ref[part, tile_words * tile + index, :live][:, None],
                    )
                pieces.append(((word >> ((row // ratio) % 32)[None, :]) & 1) != 0)
            else:
                word = bitset_ref[part, tile, :live]
                group_in_tile = row // ratio
                pieces.append(((word[:, None] >> group_in_tile[None, :]) & 1) != 0)
        visible = pieces[0] if parts == 1 else jnp.concatenate(pieces, axis=0)
        visible &= logical + row[None, :] <= positions[:, None]
        if lazy_pass and "noguard" not in lazy:
            seen = visible[:, :128]
            for offset in range(128, kv_tile, 128):
                seen = seen | visible[:, offset : offset + 128]
            has_key_ref[...] = has_key_ref[...] | seen.astype(jnp.int32)
        written = logical + lax.broadcasted_iota(
            jnp.int32, (kv_tile, head_dim), 0
        ) < jnp.max(lengths_ref[:, 0])
        # Decode each KV head once per tile; every query subtile reuses it.
        key_values = []
        for key, value in split_kv(kv_ref)[kv_head_offset : kv_head_offset + kv_heads]:
            if kv_ref.dtype == jnp.float8_e4m3fn:
                key = key.astype(jnp.float32)
                value = value.astype(jnp.float32)
                # A key only reaches visible (written, finite) scores through
                # the mask select below, so lean skips its guard. Values keep
                # theirs: a masked probability is 0, and 0 x NaN is NaN.
                if not lean:
                    key = jnp.where(jnp.isfinite(key) & written, key, 0)
                value = jnp.where(jnp.isfinite(value) & written, value, 0)
            else:
                key = jnp.where(written, key, 0)
                value = jnp.where(written, value, 0)
            if bf16_mxu:
                key = key.astype(jnp.bfloat16)
                value = value.astype(jnp.bfloat16)
            key_values.append((key, value))
        for query_tile_index in range(num_query_tiles):
            start = query_tile_index * query_tile
            tile_visible = visible[start : start + query_tile]
            for head, (key, value) in enumerate(key_values):
                if not lazy_pass:
                    m = max_ref[query_tile_index, head, :]
                    mass = mass_ref[query_tile_index, head, :]
                    acc = accumulator_ref[query_tile_index, head, :, :]
                rows = query_tile * q_heads_per_kv
                if bf16_mxu:
                    assert q16 is not None
                    query = q16[head, :, start : start + query_tile].reshape(
                        rows, head_dim
                    )
                else:
                    query = q[head, :, start : start + query_tile].reshape(
                        rows, head_dim
                    )
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
                    score * score_scale if bf16_mxu else score,
                    -jnp.inf,
                )
                if lazy_pass:
                    lazy_update(query_tile_index, head, score, value, tile_visible)
                    continue
                next_m = jnp.maximum(m, jnp.max(score, axis=1))
                finite_m = jnp.where(jnp.isneginf(next_m), 0.0, next_m)
                prob = jnp.exp(score - finite_m[:, None])
                alpha = jnp.where(jnp.isneginf(next_m), 1.0, jnp.exp(m - next_m))
                if bf16_mxu:
                    # Values are exact in BF16; split the FP32 probabilities
                    # into two BF16 terms (about 16 mantissa bits), i.e. the
                    # BF16x3 product without its zero value-residual pass.
                    value16 = value
                    high = prob.astype(jnp.bfloat16)
                    low = (prob - high.astype(jnp.float32)).astype(jnp.bfloat16)
                    weighted = lax.dot(
                        high,
                        value16,
                        dimension_numbers=(([1], [0]), ([], [])),
                        preferred_element_type=jnp.float32,
                    ) + lax.dot(
                        low,
                        value16,
                        dimension_numbers=(([1], [0]), ([], [])),
                        preferred_element_type=jnp.float32,
                    )
                else:
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
                    acc * alpha[:, None] + weighted
                )

    def compute_tile_vpu(tile, kv_ref, boundary):
        """compute_tile with less vector work per score (VLLM_TPU_QSA_PREFILL_VPU).

        Each 128-key segment of a KV tile is covered by one selection word, so
        the bits come from one lane-broadcast shift per segment instead of a
        select chain over the whole tile. Only a ``boundary`` tile (one that
        reaches the block's shortest live length) needs the causal and
        written guards.
        """
        logical = tile * kv_tile
        if dcp_world > 1:
            logical = (tile * dcp_world + rank_ref[0]) * kv_tile
        parts = bitset_ref.shape[0]
        live = query_block // parts
        tile_words = kv_tile // (ratio * 32)
        # Move bit (lane // ratio) of the segment's word to the sign bit.
        shift = 31 - lax.broadcasted_iota(jnp.int32, (1, ratio * 32), 1) // ratio
        part_bits = []
        for part in range(parts):
            segments = [
                bitset_ref[part, tile_words * tile + index, :live][:, None] << shift
                for index in range(tile_words)
            ]
            part_bits.append(
                segments[0] if tile_words == 1 else jnp.concatenate(segments, axis=1)
            )
        bits = part_bits[0] if parts == 1 else jnp.concatenate(part_bits, axis=0)
        visible = bits < 0
        if boundary:
            visible &= logical + row[None, :] <= positions[:, None]
            written = logical + lax.broadcasted_iota(
                jnp.int32, (kv_tile, head_dim), 0
            ) < jnp.max(lengths_ref[:, 0])
        pv16 = "pv16" in vpu
        key_values = []
        for key, value in split_kv(kv_ref)[kv_head_offset : kv_head_offset + kv_heads]:
            if kv_ref.dtype == jnp.float8_e4m3fn:
                key = key.astype(jnp.float32)
                value = value.astype(jnp.float32)
                if not lean:
                    guard = jnp.isfinite(key)
                    key = jnp.where(guard & written if boundary else guard, key, 0)
                guard = jnp.isfinite(value)
                value = jnp.where(guard & written if boundary else guard, value, 0)
            elif boundary:
                key = jnp.where(written, key, 0)
                value = jnp.where(written, value, 0)
            if "qk16" in vpu:
                key = key.astype(jnp.bfloat16)
            if pv16:
                value = value.astype(jnp.bfloat16)
            key_values.append((key, value))
        rows = query_tile * q_heads_per_kv
        source = q16_vpu if "qk16" in vpu else q

        def qk(query_tile_index, head, key):
            start = query_tile_index * query_tile
            query = source[head, :, start : start + query_tile].reshape(
                rows, head_dim
            )
            return lax.dot(
                query,
                key,
                dimension_numbers=(([1], [1]), ([], [])),
                preferred_element_type=jnp.float32,
            )

        # qkfirst: issue the next (subtile, head)'s QK before this one's
        # softmax and PV, so the QK matmuls do not queue behind them. One
        # unit of lookahead keeps at most two score tiles live (VMEM).
        units = [
            (query_tile_index, head)
            for query_tile_index in range(num_query_tiles)
            for head in range(len(key_values))
        ]
        early = {}

        def issue(index):
            if index < len(units) and units[index] not in early:
                query_tile_index, head = units[index]
                early[units[index]] = qk(query_tile_index, head, key_values[head][0])

        for index, (query_tile_index, head) in enumerate(units):
            key, value = key_values[head]
            start = query_tile_index * query_tile
            tile_visible = visible[start : start + query_tile]
            m = max_ref[query_tile_index, head, :]
            mass = mass_ref[query_tile_index, head, :]
            acc = accumulator_ref[query_tile_index, head, :, :]
            if "qkfirst" in vpu:
                issue(index)
                issue(index + 1)
                score = early.pop(units[index])
            else:
                score = qk(query_tile_index, head, key)
            if "qk16" in vpu and not scale_folded:
                score = score * score_scale
            row_visible = jnp.broadcast_to(
                tile_visible[None, :, :],
                (q_heads_per_kv, query_tile, kv_tile),
            ).reshape(rows, kv_tile)
            score = jnp.where(row_visible, score, -jnp.inf)
            next_m = jnp.maximum(m, jnp.max(score, axis=1))
            finite_m = jnp.where(jnp.isneginf(next_m), 0.0, next_m)
            alpha = jnp.where(jnp.isneginf(next_m), 1.0, jnp.exp(m - next_m))
            prob = jnp.exp(score - finite_m[:, None])
            prob_sum = jnp.sum(prob, axis=1)
            if pv16:
                # Mosaic already feeds an FP32 dot to the MXU as one BF16
                # pass (bit-identical on v7x); casting here only skips the
                # FP32 operand relayout.
                prob = prob.astype(jnp.bfloat16)
            weighted = lax.dot(
                prob,
                value,
                dimension_numbers=(([1], [0]), ([], [])),
                preferred_element_type=jnp.float32,
            )
            max_ref[query_tile_index, head, :] = next_m
            mass_ref[query_tile_index, head, :] = mass * alpha + prob_sum
            accumulator_ref[query_tile_index, head, :, :] = (
                acc * alpha[:, None] + weighted
            )

    if vpu:
        min_length = min_length_ref[block]

        def compute_tile(tile, kv_ref):  # noqa: F811 (the VPU path's tile)
            end = tile * kv_tile + kv_tile
            if dcp_world > 1:
                end = (tile * dcp_world + rank_ref[0]) * kv_tile + kv_tile

            @pl.when(end <= min_length)
            def interior():
                compute_tile_vpu(tile, kv_ref, False)

            @pl.when(end > min_length)
            def boundary():
                compute_tile_vpu(tile, kv_ref, True)

    def run_tiles(tile_fn):
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
            tile_fn(tile, kv0_ref)

            @pl.when(slot + 2 < count)
            def prefetch_next_even():
                start_tile(active_tile(slot + 2), kv0_ref, sem0_ref)

            tile = active_tile(slot + 1)
            wait_tile(tile, kv1_ref, sem1_ref)
            tile_fn(tile, kv1_ref)

            @pl.when(slot + 3 < count)
            def prefetch_next_odd():
                start_tile(active_tile(slot + 3), kv1_ref, sem1_ref)

            return None

        lax.fori_loop(0, count // 2, pair_step, None)

        @pl.when(count % 2 != 0)
        def consume_odd():
            tile = active_tile(count - 1)
            wait_tile(tile, kv0_ref, sem0_ref)
            tile_fn(tile, kv0_ref)

    def finalize(state):
        """Write output (and lse) from ``state(q_tile, head, field)``."""
        output_tiles = []
        for query_tile_index in range(num_query_tiles):
            heads = []
            for head in range(kv_heads):
                mass = state(query_tile_index, head, "mass")
                acc = accumulator_ref[query_tile_index, head, :, :]
                heads.append(
                    # The value scale is applied once here, not per KV tile.
                    (acc * (v_scale / jnp.where(mass == 0, 1, mass))[:, None])
                    .reshape(q_heads_per_kv, query_tile, head_dim)
                    .transpose(1, 0, 2)
                )
            output_tiles.append(jnp.concatenate(heads, axis=1))
        output_ref[...] = jnp.concatenate(output_tiles, axis=0).astype(
            output_ref.dtype
        )
        if return_lse:
            # Natural-log sum of the scaled scores; -inf for a row with no key.
            lse_tiles = []
            for query_tile_index in range(num_query_tiles):
                heads = []
                for head in range(kv_heads):
                    m = state(query_tile_index, head, "max")
                    mass = state(query_tile_index, head, "mass")
                    lse = jnp.where(
                        mass > 0, m + jnp.log(jnp.where(mass > 0, mass, 1)), -jnp.inf
                    )
                    heads.append(lse.reshape(q_heads_per_kv, query_tile).T)
                lse_tiles.append(jnp.concatenate(heads, axis=1))
            lse_ref[...] = jnp.concatenate(lse_tiles, axis=0)

    def row_state(query_tile_index, head, field):
        ref = max_ref if field == "max" else mass_ref
        return ref[query_tile_index, head, :]

    if not lazy:
        run_tiles(compute_tile)
        finalize(row_state)
        return

    run_tiles(functools.partial(compute_tile, lazy_pass=True))

    def lazy_state(query_tile_index, head, field):
        if field == "mass":
            return jnp.sum(partial_ref[query_tile_index, head], axis=1)
        stab = jnp.max(stab_ref[query_tile_index, head])
        return jnp.full((query_tile * q_heads_per_kv,), stab, jnp.float32)

    # Underflow guard: a row with a visible key whose largest lane partial is
    # under e^-40 (so its mass is under 128 e^-40) may have lost
    # probabilities to underflow below the shared stabilizer; recompute the
    # whole block with the per-row softmax. Conservative (never misses).
    if "noguard" in lazy:
        fallback = jnp.int32(0)
    else:
        fallback = jnp.int32(0)
        for query_tile_index in range(num_query_tiles):
            start = query_tile_index * query_tile
            has_key = (
                jnp.max(has_key_ref[start : start + query_tile], axis=1, keepdims=True)
                > 0
            )
            for head in range(kv_heads):
                lanes = partial_ref[query_tile_index, head].reshape(
                    q_heads_per_kv, query_tile, 128
                )
                for q_head in range(q_heads_per_kv):
                    largest = jnp.max(lanes[q_head], axis=1, keepdims=True)
                    low = has_key & (largest < LAZY_SOFTMAX_UNDERFLOW)
                    fallback = jnp.maximum(
                        fallback, jnp.max(low.astype(jnp.int32))
                    )

    @pl.when(fallback == 0)
    def lazy_epilogue():
        finalize(lazy_state)

    @pl.when(fallback != 0)
    def exact_fallback():
        init_state()
        run_tiles(compute_tile)
        finalize(row_state)


# VLLM_TPU_QSA_PREFILL_LAZY_SOFTMAX: a tile raises the stabilizer only when
# its maximum exceeds it by more than this (probabilities stay <= e^8), and a
# row whose largest lane-partial mass is under e^-40 triggers the fallback
# (then every dropped probability is under e^-36 of the row's largest: at
# most 37888 owned keys x e^-36.8 ~ 4e-12 of its mass).
LAZY_SOFTMAX_SLACK = 8.0
LAZY_SOFTMAX_UNDERFLOW = math.exp(-40.0)


# Soft cap on one step's selection bitset, for the resident prefix tier.
METADATA_BITSET_TARGET_BYTES = 16 * 1024**2
_WORD_BITS = 32
_NATIVE_ALIGN = 128


def _metadata_prefix_pages(
    tokens: int,
    selected_groups: int,
    ratio: int,
    page_size: int,
    full_pages: int,
    kv_tile: int,
) -> int:
    """Smallest page prefix covering the step's queries and its selections."""
    if min(tokens, selected_groups, ratio, page_size, full_pages, kv_tile) <= 0:
        return full_pages
    if kv_tile % (ratio * _WORD_BITS):
        return full_pages
    pages = min(
        full_pages, (max(tokens, selected_groups * ratio) + page_size - 1) // page_size
    )
    safe_tokens = pages * page_size // kv_tile * kv_tile
    if safe_tokens < ratio or safe_tokens > jnp.iinfo(jnp.int32).max:
        return full_pages
    return pages


def _metadata_resident_pages(
    tokens: int,
    query_block: int,
    ratio: int,
    page_size: int,
    full_pages: int,
    kv_tile: int,
    target_bytes: int = METADATA_BITSET_TARGET_BYTES,
) -> int:
    """Largest page prefix whose padded selection bitset fits target_bytes."""
    if min(tokens, query_block, ratio, page_size, full_pages, kv_tile) <= 0:
        return full_pages
    if kv_tile % (ratio * _WORD_BITS) or target_bytes <= 0:
        return full_pages
    nblocks = (tokens + query_block - 1) // query_block
    native_width = (query_block + _NATIVE_ALIGN - 1) // _NATIVE_ALIGN * _NATIVE_ALIGN
    word_budget = target_bytes // (nblocks * native_width * 4)
    word_budget = word_budget // _NATIVE_ALIGN * _NATIVE_ALIGN
    tiles = word_budget // (kv_tile // (ratio * _WORD_BITS))
    pages = min(full_pages, tiles * kv_tile // page_size)
    safe_tokens = pages * page_size // kv_tile * kv_tile
    if safe_tokens < ratio or safe_tokens > jnp.iinfo(jnp.int32).max:
        return full_pages
    return pages


def _metadata_prefix_fits(blocks, lengths, *, ratio, full_groups, prefix_groups):
    """Every valid selection and causal length lies inside the prefix."""
    return jnp.all(
        (blocks < 0) | (blocks >= full_groups) | (blocks < prefix_groups)
    ) & jnp.all((lengths >= 0) & (lengths <= prefix_groups * ratio))


def _metadata_prefix_dispatch(
    run, blocks, lengths, page_table, *, tokens, query_block, ratio, page_size, kv_tile
):
    """``run(table)`` on the smallest certified page prefix of ``page_table``.

    Two tiers: the step's own extent (queries and selection budget), then the
    largest prefix whose bitset stays under METADATA_BITSET_TARGET_BYTES; the
    full table otherwise. Outputs are identical across tiers.
    """
    full_pages = page_table.shape[1]
    small_pages = _metadata_prefix_pages(
        tokens, blocks.shape[1], ratio, page_size, full_pages, kv_tile
    )
    resident_pages = _metadata_resident_pages(
        tokens, query_block, ratio, page_size, full_pages, kv_tile
    )

    def fits(pages):
        # A visited KV tile is DMAed in full, including its masked token rows.
        return _metadata_prefix_fits(
            blocks,
            lengths,
            ratio=ratio,
            full_groups=full_pages * page_size // ratio,
            prefix_groups=pages * page_size // kv_tile * kv_tile // ratio,
        )

    def fallback(_):
        if not small_pages < resident_pages < full_pages:
            return run(page_table)
        return lax.cond(
            fits(resident_pages),
            lambda _: run(page_table[:, :resident_pages]),
            lambda _: run(page_table),
            None,
        )

    if small_pages >= full_pages:
        return fallback(None)
    return lax.cond(
        fits(small_pages),
        lambda _: run(page_table[:, :small_pages]),
        fallback,
        None,
    )


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
        "kv_head_offset",
        "query_tile",
        "return_lse",
        "dcp_world",
        "ascending",
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
    kv_head_offset: int = 0,
    query_tile: int | None = None,
    return_lse: bool = False,
    dcp_rank=None,
    dcp_world: int = 1,
    block_bits=None,
    ascending: bool = False,
):
    """``_block_sparse_gqa_prefill``, optionally on a bounded page-table prefix.

    VLLM_TPU_QSA_PREFILL_PREFIX_PAGES: when every selected group and causal
    length of the step fits a page prefix, build the selection metadata over
    that prefix instead of the whole max_model_len table (bit-identical: the
    same KV tiles are visited; rows outside the prefix were empty anyway).
    """

    def run(table):
        return _block_sparse_gqa_prefill(
            query,
            blocks,
            lengths,
            table,
            cache,
            ratio=ratio,
            kv_heads=kv_heads,
            query_block=query_block,
            kv_tile=kv_tile,
            k_scale=k_scale,
            v_scale=v_scale,
            bf16_io=bf16_io,
            kv_head_offset=kv_head_offset,
            query_tile=query_tile,
            return_lse=return_lse,
            dcp_rank=dcp_rank,
            dcp_world=dcp_world,
            **({"block_bits": block_bits} if block_bits is not None else {}),
            **({"ascending": True} if ascending else {}),
        )

    # A precomputed bitset (the union walk's shared fallback) is laid out over
    # the whole table: no prefix dispatch.
    if (
        not envs.VLLM_TPU_QSA_PREFILL_PREFIX_PAGES
        or dcp_rank is not None
        or block_bits is not None
    ):
        return run(page_table)
    return _metadata_prefix_dispatch(
        run,
        blocks,
        lengths,
        page_table,
        tokens=len(query),
        query_block=query_block,
        ratio=ratio,
        page_size=cache.shape[1],
        kv_tile=kv_tile,
    )


def _block_sparse_gqa_prefill(
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
    kv_head_offset: int = 0,
    query_tile: int | None = None,
    return_lse: bool = False,
    dcp_rank=None,
    dcp_world: int = 1,
    block_bits=None,
    ascending: bool = False,
):
    """Run exact selected-group attention with query-block KV reuse.

    ``block_bits``: this step's selection bitset, already built by the caller
    in ``_selection_bitset``'s layout ([blocks, words, native width]; one
    256-row part, no DCP, the bitset kernel on). The union walk's shared
    fallback passes its plan's bits.

    ``return_lse``: also return lse [tokens, heads] (natural log of the scaled
    scores' sum, -inf for a row with no key); the output is then f32.
    ``dcp_rank`` (int32[1]) / ``dcp_world``: main-KV decode context parallel.
    A table entry covers ``dcp_world * page`` tokens; this rank attends the
    tokens with ``(t // page) % dcp_world == dcp_rank``, held in the entry's
    page at row ``t % page``. The KV tiles walk only those owned pages (one
    page per tile: ``kv_tile`` equal to the page), so no other rank's rows are
    fetched and the tile size does not grow with ``dcp_world``.

    ``ascending``: each row's valid selections are in ascending order (only
    a speed hint: VLLM_TPU_QSA_PREFILL_LEAN_WRAPPER takes the ranged bitset
    loop on it; the bits are the same in any order).

    ``kv_head_offset`` skips that many KV heads of each cache row: a paired
    main KV page stores a second layer's head after the first layer's.
    ``query_tile`` is the query subtile that shares one decoded KV tile; it
    defaults to ``default_query_tile``.
    """
    tokens, query_heads, head_dim = query.shape
    pages, physical_page_size, head_axis, packing, cache_dim = cache.shape
    if tokens % query_block or query_block % 8:
        raise ValueError("Token rows must form aligned query blocks")
    paired_pages = (
        kv_tile in PAIRED_KV_TILES and physical_page_size % 128 == 0 and ratio == 4
    )
    if paired_pages:
        if (
            head_dim != 256
            or cache.dtype != jnp.float8_e4m3fn
            or query_block % 128
            or query_block * query_heads > PAIRED_PAGE_MAX_QUERY_ROWS
            or not bf16_io
        ):
            raise ValueError(
                "Paired pages require BF16 D256 queries, an FP8 cache and a "
                "128-aligned query block within the measured state footprint"
            )
    elif kv_tile // ratio != 32 or max(physical_page_size, kv_tile) % min(
        physical_page_size, kv_tile
    ):
        raise ValueError(
            "Each KV tile must use one selection word or two FP8 mask segments"
        )
    if query_heads % kv_heads or cache_dim < head_dim:
        raise ValueError("QSA query and KV geometry is inconsistent")
    if kv_head_offset and (
        kv_head_offset < 0 or (kv_head_offset + kv_heads) * 2 > head_axis * packing
    ):
        raise ValueError("KV head offset exceeds the cache row's head slots")
    original_head_dim = head_dim
    query = jnp.pad(query, ((0, 0), (0, 0), (0, cache_dim - head_dim)))
    head_dim = cache_dim
    if packing != (2 if cache.dtype == jnp.bfloat16 else 4):
        raise ValueError("Cache packing does not match its dtype")

    dcp = dcp_rank is not None
    if dcp and not (paired_pages and kv_tile == physical_page_size):
        raise ValueError("DCP prefill needs paired pages, one owned page per KV tile")
    # Selection groups / words over the request's global token range; under
    # DCP the kernel walks only this rank's pages (one per table entry), so
    # its tiles and bitset words are the owned ones (`owned_word_ids`).
    groups = page_table.shape[1] * physical_page_size * dcp_world // ratio
    words_per_tile = kv_tile // (ratio * 32)
    global_words = (
        (((groups * ratio + kv_tile - 1) // kv_tile) * words_per_tile + 127) // 128
    ) * 128
    num_kv_tiles = (
        page_table.shape[1]
        if dcp
        else (groups * ratio + kv_tile - 1) // kv_tile
    )
    words = ((num_kv_tiles * words_per_tile + 127) // 128) * 128
    nblocks = tokens // query_block
    native_query_width = ((query_block + 127) // 128) * 128
    # Blocks longer than 256 rows keep one [words, 256] bitset per 256 rows,
    # bounding the bitset kernel's lane width and VMEM windows.
    if query_block > 256 and query_block % 256:
        raise ValueError("Query blocks above 256 rows must be 256-aligned")
    parts = max(1, query_block // 256)
    part_width = native_query_width // parts
    if dcp:
        # Local word w (tile w // words_per_tile of this rank's pages) is
        # global word (tile * dcp_world + rank) * words_per_tile + w % ...
        rank = jnp.asarray(dcp_rank, jnp.int32).reshape(-1)[0]
        local = jnp.arange(words, dtype=jnp.int32)
        owned_word_ids = (
            (local // words_per_tile) * dcp_world + rank
        ) * words_per_tile + local % words_per_tile
        owned_word_live = (local < num_kv_tiles * words_per_tile) & (
            owned_word_ids < global_words
        )
        owned_word_ids = jnp.minimum(owned_word_ids, global_words - 1)

        def own_words(bits, axis):
            taken = jnp.take(bits, owned_word_ids, axis=axis)
            shape = [1] * bits.ndim
            shape[axis] = words
            return jnp.where(owned_word_live.reshape(shape), taken, 0)

    # VLLM_TPU_QSA_PREFILL_BITSET_OWNED: the bitset kernel numbers this rank's
    # words itself, so the global bitset, its relayout copy and the owned-word
    # gather drop out (bit-identical).
    #
    # VLLM_TPU_QSA_PREFILL_LEAN_WRAPPER: under DCP the owned bitset kernel,
    # numbered with shifts, and, on ascending rows, the ranged compare loop,
    # which visits only the chunks holding each word group
    # (bit-identical; the metadata outside the KV loop drops from about 1.5
    # to 0.6 ms per 8K-row call at 76K context).
    lean_wrapper = dcp and envs.VLLM_TPU_QSA_PREFILL_LEAN_WRAPPER
    owned_kernel = dcp and (envs.VLLM_TPU_QSA_PREFILL_BITSET_OWNED or lean_wrapper)
    if block_bits is not None:
        if dcp or parts != 1 or not envs.VLLM_TPU_QSA_PREFILL_BITSET_KERNEL:
            raise ValueError(
                "a precomputed bitset needs one 256-row part, no DCP and the bitset kernel"
            )
        if tuple(block_bits.shape) != (nblocks, words, part_width):
            raise ValueError(
                f"block_bits {block_bits.shape} != {(nblocks, words, part_width)}"
            )
    if envs.VLLM_TPU_QSA_PREFILL_BITSET_KERNEL and block_bits is not None:
        block_bits_native = block_bits
    elif envs.VLLM_TPU_QSA_PREFILL_BITSET_KERNEL:
        block_bits_native = _selection_bitset(
            blocks,
            lengths,
            ratio=ratio,
            groups=groups,
            words=global_words if dcp and not owned_kernel else words,
            query_block=query_block // parts,
            width=part_width,
            **(
                {"owned": (rank, dcp_world, words_per_tile, num_kv_tiles * words_per_tile)}
                if owned_kernel
                else {}
            ),
            **({"shift_owned": True} if lean_wrapper else {}),
            # Unordered rows make every chunk range span the block, which is
            # slower than the full compare loop: ranged on ascending rows only.
            **({"ranged": True} if lean_wrapper and ascending else {}),
        )
        if dcp and not owned_kernel:
            block_bits_native = own_words(block_bits_native, 1)
        part_bits = block_bits_native.reshape(nblocks, parts, words, part_width)
        if paired_pages:
            active = jnp.any(
                part_bits[:, :, : words_per_tile * num_kv_tiles].reshape(
                    nblocks, parts, num_kv_tiles, words_per_tile * part_width
                )
                != 0,
                axis=(1, 3),
            )
        else:
            active = jnp.any(part_bits[:, :, :num_kv_tiles] != 0, axis=(1, 3))
        active_tiles = _compact_active(active)
        active_counts = jnp.sum(active, axis=1, dtype=jnp.int32)
        table = page_table.reshape(nblocks, query_block, -1)[:, 0]
    elif (
        not dcp
        and paired_pages
        and query_block == 256
        and tokens * words * 4 >= 16 * 1024**2
    ):
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
        bitset = _scatter_bitset(
            blocks, lengths, ratio=ratio, groups=groups,
            words=global_words if dcp else words,
        )
        if dcp:
            bitset = own_words(bitset, 1)

        nblocks = tokens // query_block
        block_bits = bitset.reshape(nblocks, query_block, words)
        if paired_pages:
            active = jnp.any(
                block_bits[:, :, : words_per_tile * num_kv_tiles].reshape(
                    nblocks, query_block, num_kv_tiles, words_per_tile
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
        active_tiles = _compact_active(active)
        active_counts = jnp.sum(active, axis=1, dtype=jnp.int32)
        table = page_table.reshape(nblocks, query_block, -1)[:, 0]
    if parts > 1 and not envs.VLLM_TPU_QSA_PREFILL_BITSET_KERNEL:
        block_bits_native = (
            block_bits_native.reshape(nblocks, words, parts, part_width)
            .transpose(0, 2, 1, 3)
            .reshape(nblocks * parts, words, part_width)
        )

    if query_tile is None:
        query_tile = default_query_tile(query_block)
    if query_block % query_tile or query_tile % 8:
        raise ValueError("Query subtiles must evenly split the query block")
    # VLLM_TPU_QSA_PREFILL_VPU: the per-segment bits need one 128-key
    # selection word per lane row (ratio 4); BF16 operands need BF16 queries.
    vpu = vpu_features() if ratio * 32 == 128 and bf16_io else ()
    if query_tile % 16:
        vpu = tuple(name for name in vpu if name == "mask")
    if vpu:
        logger.info_once(
            "QSA sparse prefill VPU path on: features=%s "
            "(VLLM_TPU_QSA_PREFILL_VPU=%s)",
            ",".join(vpu),
            envs.VLLM_TPU_QSA_PREFILL_VPU,
        )
        live_lengths = jnp.where(lengths > 0, lengths, jnp.iinfo(jnp.int32).max)
        min_lengths = jnp.min(
            live_lengths.reshape(nblocks, query_block), axis=1
        ).astype(jnp.int32)
    # VLLM_TPU_QSA_PREFILL_PAGE_COPIES needs KV tiles and pages that nest.
    page_copies = (
        envs.VLLM_TPU_QSA_PREFILL_PAGE_COPIES
        and physical_page_size % (ratio * 32) == 0
        and (kv_tile % physical_page_size == 0 or physical_page_size % kv_tile == 0)
    )
    if page_copies:
        logger.info_once(
            "QSA sparse prefill page-sized KV copies on "
            "(VLLM_TPU_QSA_PREFILL_PAGE_COPIES, %d-row pages, %d-row tiles)",
            physical_page_size,
            kv_tile,
        )
    bf16_mxu = (
        not vpu
        and envs.VLLM_TPU_QSA_PREFILL_BF16_MXU
        and bf16_io
        and query_tile % 16 == 0
        and cache.dtype in (jnp.float8_e4m3fn, jnp.bfloat16)
    )
    state_shape = (
        query_block // query_tile,
        kv_heads,
        query_tile * (query_heads // kv_heads),
    )
    lazy = lazy_softmax_features() if dcp and not vpu and not bf16_mxu else ()
    if lazy and kv_tile % 128:
        lazy = ()
    if lazy:
        logger.info_once(
            "QSA sparse DCP prefill lazy softmax on: %s "
            "(VLLM_TPU_QSA_PREFILL_LAZY_SOFTMAX=%s)",
            ",".join(lazy),
            envs.VLLM_TPU_QSA_PREFILL_LAZY_SOFTMAX,
        )
    out_spec = pl.BlockSpec(
        (query_block, query_heads, head_dim),
        lambda b, *_: (b, 0, 0),
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
            **({"kv_head_offset": kv_head_offset} if kv_head_offset else {}),
            **({"bf16_mxu": True} if bf16_mxu else {}),
            query_tile=query_tile,
            **({"lean": True} if envs.VLLM_TPU_QSA_PREFILL_LEAN else {}),
            **({"return_lse": True} if return_lse else {}),
            **({"dcp_world": dcp_world} if dcp else {}),
            **({"vpu": vpu} if vpu else {}),
            **({"page_copies": True} if page_copies else {}),
            **({"lazy": lazy} if lazy else {}),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3 + int(dcp) + int(bool(vpu)),
            grid=(nblocks,),
            in_specs=[
                pl.BlockSpec(
                    (query_block, query_heads, head_dim),
                    lambda b, *_: (b, 0, 0),
                ),
                pl.BlockSpec(
                    (parts, words, part_width),
                    lambda b, *_: (b, 0, 0),
                ),
                pl.BlockSpec((query_block, 1), lambda b, *_: (b, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=(
                [out_spec, pl.BlockSpec((query_block, query_heads), lambda b, *_: (b, 0))]
                if return_lse
                else out_spec
            ),
            scratch_shapes=[
                pltpu.VMEM((kv_tile, head_axis, packing, head_dim), cache.dtype),
                pltpu.VMEM((kv_tile, head_axis, packing, head_dim), cache.dtype),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM((*state_shape, head_dim), jnp.float32),
            ]
            + (
                [
                    pltpu.VMEM((*state_shape[:2], 1, 128), jnp.float32),
                    pltpu.VMEM((*state_shape, 128), jnp.float32),
                    pltpu.VMEM((query_block, 128), jnp.int32),
                ]
                if lazy
                else []
            ),
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
                PAIRED_PAGE_VMEM_MIB.get(
                    query_block, max(PAIRED_PAGE_VMEM_MIB.values())
                )
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
        out_shape=(
            [
                jax.ShapeDtypeStruct(query.shape, jnp.float32),
                jax.ShapeDtypeStruct((tokens, query_heads), jnp.float32),
            ]
            if return_lse
            else jax.ShapeDtypeStruct(
                query.shape, jnp.bfloat16 if bf16_io else jnp.float32
            )
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
        *([jnp.asarray(dcp_rank, jnp.int32).reshape(1)] if dcp else []),
        *([min_lengths] if vpu else []),
        query.astype(jnp.bfloat16 if bf16_io else jnp.float32),
        block_bits_native,
        lengths[:, None],
        cache.reshape(pages * physical_page_size, head_axis, packing, head_dim),
    )
    if return_lse:
        return output[0][:, :, :original_head_dim], output[1]
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
    kv_head_offset: int = 0,
    query_tile: int | None = None,
    dcp=None,
    return_lse: bool = False,
    ascending: bool = False,
):
    """Align requests, share KV tiles, then restore the original token order.

    With ``return_lse`` returns ``(out, lse, cache)``: ``out`` f32, normalized
    and not gated, ``lse`` [tokens, heads] (-inf for an empty row). ``dcp =
    (rank int32[1], D)``: attend this rank's owned main-KV rows (see
    ``block_sparse_gqa_prefill``)."""
    real = starts[1:] - starts[:-1]
    offset = {"kv_head_offset": kv_head_offset} if kv_head_offset else {}
    if query_tile is not None:
        offset["query_tile"] = query_tile
    if return_lse:
        offset["return_lse"] = True
    if dcp is not None:
        offset["dcp_rank"], offset["dcp_world"] = dcp
    if ascending:
        offset["ascending"] = True

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
            **offset,
        )

    padded = ((real + query_block - 1) // query_block) * query_block
    offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(padded)))
    # Each request starts on a query-block boundary, so a block never spans
    # two requests (the kernel shares one page-table row and one KV tile list
    # per block). In the worst case every request wastes query_block - 1 rows.
    worst_tokens = (
        (len(query) + (query_block - 1) * (len(starts) - 1) + query_block - 1)
        // query_block
        * query_block
    )
    # Usually the block-aligned layout still fits the step's own token
    # bucket: a single prefill chunk wastes < query_block rows, and empty
    # rows of the bucket absorb that. A few more blocks cover a few chunks.
    # Padding to the worst case would double an 8192-token step.
    bucket_tokens = (len(query) + query_block - 1) // query_block * query_block
    sizes = sorted(
        {
            min(bucket_tokens + extra * query_block, worst_tokens)
            for extra in (0, SPARE_QUERY_BLOCKS, worst_tokens)
        }
    )

    def run_with_padding(max_tokens):
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
            **offset,
        )
        back = jnp.minimum(slots, max_tokens - 1)
        if return_lse:
            out, lse = output[0][back], output[1][back]
            if mask_in_padding:
                # Tokens past the step's rows read the last aligned row, which
                # can be a real one: mask them here, fused with the gather.
                out = jnp.where(live[:, None, None], out, 0)
                lse = jnp.where(live[:, None], lse, -jnp.inf)
            return out, lse
        return output[back]

    live = lengths > 0
    # VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED: only the padded layouts mask
    # dead rows. In the aligned layout a row with length 0 sees no key, so the
    # kernel already gives it out 0 and lse -inf (bit-identical, one f32
    # [T, heads, D] select less per call).
    mask_in_padding = return_lse and envs.VLLM_TPU_QSA_PREFILL_LIVE_MASK_PADDED

    # All lax.switch branches are traced. A statically unaligned total cannot
    # enter run_aligned, whose core rejects its shape before execution.
    branches = [functools.partial(lambda n, _: run_with_padding(n), n) for n in sizes]
    route = jnp.sum(offsets[-1] > jnp.asarray(sizes[:-1], jnp.int32))
    if len(query) % query_block == 0:
        branches.insert(0, run_aligned)
        route = jnp.where(jnp.all(real % query_block == 0), 0, route + 1)
    output = (
        branches[0](None) if len(branches) == 1 else lax.switch(route, branches, None)
    )
    if mask_in_padding:
        out, lse = output
        return out, lse, cache
    if return_lse:
        out, lse = output
        return (
            jnp.where(live[:, None, None], out, 0),
            jnp.where(live[:, None], lse, -jnp.inf),
            cache,
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
    paired_pages = (
        kv_tile in PAIRED_KV_TILES and physical_page_size % 128 == 0 and ratio == 4
    )
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
                bits[:, : words_per_tile * num_kv_tiles].reshape(
                    query_block, num_kv_tiles, words_per_tile
                )
                != 0,
                axis=(0, 2),
            )
        else:
            active = jnp.any(bits[:, :num_kv_tiles] != 0, axis=0)
        native_bits = jnp.pad(bits.T, ((0, 0), (0, native_query_width - query_block)))
        tiles = _compact_active(active)
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
