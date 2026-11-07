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
"""Pallas lightning-indexer block scores for MiniMax Sparse Attention.

For every query token and every KV page of its request, the score of the
page is the max over its tokens of ``index_q . index_k`` (no scale: only the
order matters). Each request's tokens are cut into query tiles of
`block_q` rows; a grid step scores one tile against `pages_per_step` pages
read straight from the paged index cache, so a request's index keys are read
once per tile. The cache packs consecutive tokens into 32-bit words
(`packed_page`); each packing slot is scored separately, which a max does not
mind.

Scores are produced transposed, ``[pages, tiles * block_q]``, so each output
block is lane-aligned, and gathered back to ``[tokens, pages]``; a decode
batch's 128-page groups are lane-aligned either way and come out token-major.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.minimax_m3.kernels.packed_page import (
    check_page_dtype,
    unpack_page,
)

DEFAULT_BLOCK_Q = 128
DEFAULT_LARGE_BLOCK_Q = 512
DEFAULT_PAGES_PER_STEP = 32
# The block_q class (decode, MTP verify) runs the v3 DMA ring; 128 pages per
# step keep more bytes in flight (j-dba77583: 32 req x 8 tokens at 128K,
# 1.95 -> 1.66 ms). The large prefill class keeps 32.
DEFAULT_SMALL_PAGES_PER_STEP = 128
DEFAULT_KERNEL = "v3"
# Large prefill tiles: v3's DMA ring is slower there on fp8 (1.79 vs 1.51 ms).
DEFAULT_LARGE_KERNEL = "v2"
_TILE_SHIFT = 16
# tile_info = query row of the tile << _ROW_SHIFT | request of the tile
_ROW_SHIFT = 10
_SEQ_MASK = (1 << _ROW_SHIFT) - 1
# f32 query rows are read at element offsets that are multiples of the sublane tile.
_ROW_ALIGN = 8
_GROUP_MASK = (1 << _TILE_SHIFT) - 1


def _kernel_v1(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    # In
    q_ref,  # [Bq, D]
    *refs,  # G x [1, S / p, p, D] index-cache pages, then the output
):
    pages_per_step = len(refs) - 1
    page_refs, o_ref = refs[:pages_per_step], refs[pages_per_step]
    q = q_ref[...].astype(jnp.bfloat16)
    maxima = []
    for page_ref in page_refs:
        page_max = None
        for keys in unpack_page(page_ref):
            s = jax.lax.dot_general(
                keys, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            m = jnp.max(s, axis=0, keepdims=True)
            page_max = m if page_max is None else jnp.maximum(page_max, m)
        maxima.append(page_max)
    o_ref[...] = jnp.concatenate(maxima, axis=0)


# Scores per dot in `_kernel_v2`: bounds the f32 ``[pages * S, block_q]``
# product kept in VMEM (2 MiB).
_MAX_SCORE_ELEMS = 512 * 1024


def _page_keys(page_ref) -> jax.Array:
    """``[1, S / p, p, D]`` page -> bf16 ``[S, D]`` keys in packing order.

    One width-changing bitcast of the packed 32-bit words: the token order
    inside the page changes, which a max over the page does not see.
    """
    rows, packing, head_dim = page_ref.shape[1:]
    words = page_ref.bitcast(jnp.uint32)[0, :, 0, :]
    keys = pltpu.bitcast(words, page_ref.dtype)
    assert keys.shape == (rows * packing, head_dim), keys.shape
    return keys.astype(jnp.bfloat16)


def _kernel_v2(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    # In
    q_ref,  # [Bq, D]
    *refs,  # G x [1, S / p, p, D] index-cache pages, then the output
):
    """`_kernel_v1` with the pages' keys unpacked by one bitcast each and
    stacked, so each dot keeps the query tile latched across many pages."""
    pages_per_step = len(refs) - 1
    page_refs, o_ref = refs[:pages_per_step], refs[pages_per_step]
    q = q_ref[...].astype(jnp.bfloat16)
    block_q = q.shape[0]
    page_size = page_refs[0].shape[1] * page_refs[0].shape[2]
    pages_per_dot = max(
        1, min(pages_per_step, _MAX_SCORE_ELEMS // (page_size * block_q))
    )
    maxima = []
    for first in range(0, pages_per_step, pages_per_dot):
        group = page_refs[first : first + pages_per_dot]
        keys = [_page_keys(r) for r in group]
        keys = keys[0] if len(keys) == 1 else jnp.concatenate(keys, axis=0)
        s = jax.lax.dot_general(
            keys, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        maxima.append(jnp.max(s.reshape(len(group), page_size, block_q), axis=1))
    o_ref[...] = maxima[0] if len(maxima) == 1 else jnp.concatenate(maxima, axis=0)


def _scores_v3(q, page_refs, o_ref, transposed):
    """`_kernel_v2`'s scoring over pages already in VMEM; a `transposed`
    output block is ``[Bq, G]``."""
    block_q = q.shape[0]
    pages_per_step = len(page_refs)
    rows, packing, _ = page_refs[0].shape
    page_size = rows * packing
    pages_per_dot = max(
        1, min(pages_per_step, _MAX_SCORE_ELEMS // (page_size * block_q))
    )
    maxima = []
    for first in range(0, pages_per_step, pages_per_dot):
        group = page_refs[first : first + pages_per_dot]
        keys = []
        for r in group:
            words = r.bitcast(jnp.uint32)[:, 0, :]
            keys.append(pltpu.bitcast(words, r.dtype).astype(jnp.bfloat16))
        keys = keys[0] if len(keys) == 1 else jnp.concatenate(keys, axis=0)
        s = jax.lax.dot_general(
            keys, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        maxima.append(jnp.max(s.reshape(len(group), page_size, block_q), axis=1))
    out = maxima[0] if len(maxima) == 1 else jnp.concatenate(maxima, axis=0)
    o_ref[...] = out.T if transposed else out


def _kernel_v3(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    seq_pages_ref,  # [max_num_seqs] pages of each request
    # In
    q_ref,  # [Bq, D]
    cache_ref,  # [pages, S / p, p, D] index cache, left in HBM
    # Out
    o_ref,  # [G, Bq]
    # Scratch
    buf0,  # [G, S / p, p, D] pages of even steps
    buf1,  # [G, S / p, p, D] pages of odd steps
    sems,  # DMA semaphores [2, G]
    *,
    transposed=False,
):
    """`_kernel_v2` with the pages fetched by a hand-written double-buffered
    DMA ring: one scalar address per page and no copy for pages past the
    request's end (their scores are never read)."""
    pages_per_step = buf0.shape[0]
    pages_per_seq = block_tables_ref.shape[0] // seq_pages_ref.shape[0]
    step, num_steps = pl.program_id(0), pl.num_programs(0)

    def copies(w, buf, slot):
        entry = work_ref[w]
        seq = tile_seq_ref[entry >> _TILE_SHIFT] & _SEQ_MASK
        first = (entry & _GROUP_MASK) * pages_per_step
        live = jnp.maximum(seq_pages_ref[seq] - first, 1)
        base = seq * pages_per_seq + first
        out = []
        for g in range(pages_per_step):
            col = jnp.minimum(first + g, pages_per_seq - 1) - first
            page = block_tables_ref[base + col]
            out.append(
                (
                    g < live,
                    pltpu.make_async_copy(
                        cache_ref.at[page], buf.at[g], sems.at[slot, g]
                    ),
                )
            )
        return out

    def start(w, buf, slot):
        for live, copy in copies(w, buf, slot):
            pl.when(live)(copy.start)

    def wait(w, buf, slot):
        for live, copy in copies(w, buf, slot):
            pl.when(live)(copy.wait)

    @pl.when(step == 0)
    def _():
        start(step, buf0, 0)

    even = step % 2 == 0

    @pl.when(step + 1 < num_steps)
    def _():
        @pl.when(even)
        def _():
            start(step + 1, buf1, 1)

        @pl.when(jnp.logical_not(even))
        def _():
            start(step + 1, buf0, 0)

    q = q_ref[...].astype(jnp.bfloat16)

    @pl.when(even)
    def _():
        wait(step, buf0, 0)
        _scores_v3(q, [buf0.at[g] for g in range(pages_per_step)], o_ref, transposed)

    @pl.when(jnp.logical_not(even))
    def _():
        wait(step, buf1, 1)
        _scores_v3(q, [buf1.at[g] for g in range(pages_per_step)], o_ref, transposed)


_KERNELS = {"v1": _kernel_v1, "v2": _kernel_v2}


def _tiles(cu_q_lens: jax.Array, seq_in_class: jax.Array, block_q: int, max_tiles: int):
    """Query tiles of one tile class.

    A request's tiles start at its first token rounded down to a multiple of
    `_ROW_ALIGN` (the kernel reads query rows at aligned element offsets), so
    a tile may begin with up to `_ROW_ALIGN - 1` rows of the previous request
    and end with rows of the next; those rows' scores are never read.
    """
    max_num_seqs = cu_q_lens.shape[0] - 1
    start = (cu_q_lens[:-1] // _ROW_ALIGN) * _ROW_ALIGN
    q_lens = jnp.where(seq_in_class, cu_q_lens[1:] - start, 0)
    tiles_per_seq = -(-q_lens // block_q)
    tile_ends = jnp.cumsum(tiles_per_seq)
    tile = jnp.arange(max_tiles, dtype=jnp.int32)
    tile_seq = jnp.minimum(
        jnp.searchsorted(tile_ends, tile, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_num_seqs - 1,
    )
    first_tile = tile_ends - tiles_per_seq
    tile_row0 = start[tile_seq] + (tile - first_tile[tile_seq]) * block_q
    return tile_seq, tile_row0, first_tile, start, tile_ends[-1]


def _class_scores(
    index_q,
    index_cache,
    seq_lens,
    block_tables,
    cu_q_lens,
    seq_in_class,
    *,
    block_q,
    max_tiles,
    pages_per_step,
    kernel,
    interpret,
    token_major=False,
):
    """Transposed scores ``[P, tiles * block_q]`` of one tile class, and the
    column of each token; `token_major` (kernel v3) returns them as
    ``[tiles * block_q, P]`` instead, so a token's scores are one row."""
    num_tokens, _, head_dim = index_q.shape
    _, rows_per_page, packing, _ = index_cache.shape
    page_size = rows_per_page * packing
    max_num_seqs = seq_lens.shape[0]
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    num_groups_per_seq = -(-pages_per_seq // pages_per_step)
    padded_pages = num_groups_per_seq * pages_per_step

    tile_seq, tile_row0, first_tile, tile_start, num_tiles = _tiles(
        cu_q_lens, seq_in_class, block_q, max_tiles
    )
    # The kernel reads each tile's query rows straight from the token-ordered
    # queries at an element offset (a gather of [tiles * block_q, D] cost
    # ~0.5 ms per layer on an 8K prefill, twin j-52e2df18). Rows past a
    # request's end read the next request's queries; their columns are never
    # gathered back. f32 so that any row offset is a whole 32-bit row.
    if max_num_seqs > _SEQ_MASK + 1:
        raise NotImplementedError(
            f"at most {_SEQ_MASK + 1} requests, got {max_num_seqs}"
        )
    q_rows = jnp.pad(index_q[:, 0].astype(jnp.float32), ((0, block_q), (0, 0)))
    tile_info = (
        jnp.clip(tile_row0, 0, num_tokens // _ROW_ALIGN * _ROW_ALIGN) << _ROW_SHIFT
    ) | tile_seq

    # Work list: every page group of every tile's request.
    seq_pages = -(-seq_lens // page_size)
    groups_per_tile = jnp.where(
        jnp.arange(max_tiles) < num_tiles,
        jnp.maximum(-(-seq_pages[tile_seq] // pages_per_step), 1),
        0,
    )
    group_ends = jnp.cumsum(groups_per_tile)
    w = jnp.arange(max_tiles * num_groups_per_seq, dtype=jnp.int32)
    work_tile = jnp.minimum(
        jnp.searchsorted(group_ends, w, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_tiles - 1,
    )
    work_group = w - (group_ends[work_tile] - groups_per_tile[work_tile])
    work = (work_tile << _TILE_SHIFT) | jnp.clip(work_group, 0, _GROUP_MASK)

    def page_block(g):
        def index_map(w, work, tile_info, block_tables, *_):
            entry = work[w]
            seq = tile_info[entry >> _TILE_SHIFT] & _SEQ_MASK
            col = jnp.minimum(
                (entry & _GROUP_MASK) * pages_per_step + g, pages_per_seq - 1
            )
            return (block_tables[seq * pages_per_seq + col], 0, 0, 0)

        return pl.BlockSpec((1, rows_per_page, packing, head_dim), index_map)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        # One step even with no tile in the class; its scores are never read.
        grid=(jnp.maximum(group_ends[-1], 1),),
        in_specs=[
            pl.BlockSpec(
                (pl.Element(block_q), pl.Element(head_dim)),
                lambda w, work, tile_info, *_: (
                    pl.multiple_of(
                        tile_info[work[w] >> _TILE_SHIFT] >> _ROW_SHIFT, _ROW_ALIGN
                    ),
                    0,
                ),
            ),
            *(page_block(g) for g in range(pages_per_step)),
        ],
        out_specs=pl.BlockSpec(
            (pages_per_step, block_q),
            lambda w, work, *_: (work[w] & _GROUP_MASK, work[w] >> _TILE_SHIFT),
        ),
    )
    out_shape = jax.ShapeDtypeStruct((padded_pages, max_tiles * block_q), jnp.float32)
    out_specs = grid_spec.out_specs
    if token_major:
        if kernel != "v3":
            raise NotImplementedError(
                f"token-major scores need kernel v3, got {kernel}"
            )
        out_shape = jax.ShapeDtypeStruct(out_shape.shape[::-1], jnp.float32)
        out_specs = pl.BlockSpec(
            (block_q, pages_per_step),
            lambda w, work, *_: (work[w] >> _TILE_SHIFT, work[w] & _GROUP_MASK),
        )
    params = pltpu.CompilerParams(dimension_semantics=("arbitrary",))
    if kernel == "v3":
        scores_t = pl.pallas_call(
            functools.partial(_kernel_v3, transposed=token_major),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                grid=grid_spec.grid,
                in_specs=[grid_spec.in_specs[0], pl.BlockSpec(memory_space=pl.ANY)],
                out_specs=out_specs,
                scratch_shapes=[
                    pltpu.VMEM(
                        (pages_per_step, rows_per_page, packing, head_dim),
                        index_cache.dtype,
                    ),
                    pltpu.VMEM(
                        (pages_per_step, rows_per_page, packing, head_dim),
                        index_cache.dtype,
                    ),
                    pltpu.SemaphoreType.DMA((2, pages_per_step)),
                ],
            ),
            out_shape=out_shape,
            compiler_params=params,
            interpret=interpret,
            name="minimax_m3_msa_block_scores",
        )(work, tile_info, block_tables, seq_pages, q_rows, index_cache)
    else:
        scores_t = pl.pallas_call(
            _KERNELS[kernel],
            grid_spec=grid_spec,
            out_shape=out_shape,
            compiler_params=params,
            interpret=interpret,
            name="minimax_m3_msa_block_scores",
        )(work, tile_info, block_tables, q_rows, *([index_cache] * pages_per_step))

    # Back to token order: token t of request r sits in r's tiles at its offset.
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.minimum(
        jnp.searchsorted(cu_q_lens[1:], tok, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_num_seqs - 1,
    )
    offset = tok - tile_start[seq]
    col = (first_tile[seq] + offset // block_q) * block_q + offset % block_q
    col = jnp.clip(col, 0, max_tiles * block_q - 1)
    return scores_t, col


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_q",
        "large_block_q",
        "pages_per_step",
        "small_pages_per_step",
        "kernel",
        "large_kernel",
        "interpret",
    ),
)
def block_scores(
    index_q: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    block_q: int = DEFAULT_BLOCK_Q,
    large_block_q: int = DEFAULT_LARGE_BLOCK_Q,
    pages_per_step: int = DEFAULT_PAGES_PER_STEP,
    small_pages_per_step: int = DEFAULT_SMALL_PAGES_PER_STEP,
    kernel: str = DEFAULT_KERNEL,
    large_kernel: str = DEFAULT_LARGE_KERNEL,
    interpret: bool = False,
) -> jax.Array:
    """Block-max index scores ``[T, 1, P]`` (``P`` padded to the page group).

    Requests of at most `block_q` query tokens (decode, short chunks) are
    scored in `block_q`-row tiles; longer prefill chunks in `large_block_q`-row
    tiles, so a request's index keys are read once per large tile rather than
    once per `block_q` rows.

    Args:
      index_q: ``[T, 1, D]`` index queries (one index head per rank).
      index_cache: ``[pages, S / p, p, D]`` key-only cache (bf16 or fp8).
      seq_lens, block_tables, cu_q_lens, num_seqs: ragged batch metadata.
      kernel: ``"v2"`` (bitcast unpack, query latched across pages), ``"v3"``
        (v2 with a hand-written DMA ring for the pages) or the original
        ``"v1"``, kept as the reference arm for benchmarks.
      large_kernel: the same choice for the `large_block_q` tiles.
      pages_per_step / small_pages_per_step: pages per grid step of the large
        and the block_q tile classes.
    Pages past a token's own page hold scores of keys it must not see, and
    pages past its request's last page are not written; callers mask both.
    """
    num_tokens, num_heads, _ = index_q.shape
    if num_heads != 1:
        raise NotImplementedError(
            f"MSA indexer scores one index head per rank, got {num_heads}."
        )
    check_page_dtype(index_cache.shape[2], index_cache.dtype)
    max_num_seqs = seq_lens.shape[0]
    seq_ids = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    active = seq_ids < num_seqs
    is_large = active & (q_lens > block_q)
    score = functools.partial(
        _class_scores,
        index_q,
        index_cache,
        seq_lens,
        block_tables,
        cu_q_lens,
        interpret=interpret,
    )
    # Large requests have more than block_q tokens each.
    max_large = min(max_num_seqs, num_tokens // (block_q + 1))
    # With one tile class (decode), token-major scores make the gather back to
    # token order a row gather; a column gather first relayouts the whole
    # padded [P, tiles * block_q] f32 block (4 us per layer in the b1 decode
    # twin, j-b312f26a). Needs lane-aligned [block_q, pages_per_step] blocks.
    token_major = max_large == 0 and kernel == "v3" and small_pages_per_step % 128 == 0
    small, small_col = score(
        active & ~is_large,
        block_q=block_q,
        # Aligned tile starts can push a request of <= block_q tokens into a
        # second tile.
        max_tiles=min(2 * max_num_seqs, num_tokens + max_num_seqs),
        pages_per_step=small_pages_per_step,
        kernel=kernel,
        token_major=token_major,
    )
    if token_major:
        return small[small_col][:, None, :]
    if max_large == 0:
        return small[:, small_col].T[:, None, :]
    large, large_col = score(
        is_large,
        block_q=large_block_q,
        # +1 per request for the partial last tile, plus the up-to-7 rows each
        # aligned start adds.
        max_tiles=num_tokens // large_block_q
        + max_large
        + -(-(_ROW_ALIGN - 1) * max_large // large_block_q)
        + 1,
        pages_per_step=pages_per_step,
        kernel=large_kernel,
    )
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.minimum(
        jnp.searchsorted(cu_q_lens[1:], tok, side="right", method="compare_all").astype(
            jnp.int32
        ),
        max_num_seqs - 1,
    )
    # One gather over both classes' columns (XLA fuses the concatenate into
    # it) instead of a transposing gather per class and a select.
    col = jnp.where(is_large[seq], small.shape[1] + large_col, small_col)
    # The classes may pad pages to different group multiples; pad to the wider.
    pages = max(small.shape[0], large.shape[0])
    small, large = (
        jnp.pad(x, ((0, pages - x.shape[0]), (0, 0))) for x in (small, large)
    )
    return jnp.concatenate([small, large], axis=1)[:, col].T[:, None, :]
