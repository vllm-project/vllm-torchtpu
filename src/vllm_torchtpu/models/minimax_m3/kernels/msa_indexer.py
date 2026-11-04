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
once per tile. The bf16 cache packs two consecutive tokens per 32-bit word;
even and odd tokens are scored separately, which a max does not mind.

Scores are produced transposed, ``[pages, tiles * block_q]``, so each output
block is lane-aligned, and gathered back to ``[tokens, pages]``.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_BLOCK_Q = 128
DEFAULT_PAGES_PER_STEP = 32
_TILE_SHIFT = 16
_GROUP_MASK = (1 << _TILE_SHIFT) - 1


def _kernel(
    # Scalar prefetch
    work_ref,  # [groups] tile << _TILE_SHIFT | page group within the request
    tile_seq_ref,  # [tiles] request of each tile
    block_tables_ref,  # [max_num_seqs * pages_per_seq]
    # In
    q_ref,  # [Bq, D]
    *refs,  # G x [1, S/2, 2, D] bf16 index-cache pages, then the output
):
    pages_per_step = len(refs) - 1
    page_refs, o_ref = refs[:pages_per_step], refs[pages_per_step]
    q = q_ref[...]
    maxima = []
    for page_ref in page_refs:
        # Same rank in and out: Mosaic bitcasts the packed axis 2 -> 1.
        packed = page_ref.bitcast(jnp.uint32)[0, :, 0, :]
        even = pltpu.bitcast((packed & 0xFFFF).astype(jnp.uint16), jnp.bfloat16)
        odd = pltpu.bitcast((packed >> 16).astype(jnp.uint16), jnp.bfloat16)
        page_max = None
        for keys in (even, odd):
            s = jax.lax.dot_general(
                keys, q, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            m = jnp.max(s, axis=0, keepdims=True)
            page_max = m if page_max is None else jnp.maximum(page_max, m)
        maxima.append(page_max)
    o_ref[...] = jnp.concatenate(maxima, axis=0)


def _tiles(cu_q_lens: jax.Array, num_seqs: jax.Array, num_tokens: int, block_q: int):
    """Query tiles that never span requests: (request, first row) per tile."""
    max_num_seqs = cu_q_lens.shape[0] - 1
    max_tiles = num_tokens // block_q + max_num_seqs
    seq_ids = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_lens = jnp.where(seq_ids < num_seqs, cu_q_lens[1:] - cu_q_lens[:-1], 0)
    tiles_per_seq = -(-q_lens // block_q)
    tile_ends = jnp.cumsum(tiles_per_seq)
    tile = jnp.arange(max_tiles, dtype=jnp.int32)
    tile_seq = jnp.minimum(
        jnp.searchsorted(tile_ends, tile, side="right").astype(jnp.int32),
        max_num_seqs - 1,
    )
    first_tile = tile_ends[tile_seq] - tiles_per_seq[tile_seq]
    tile_row0 = cu_q_lens[tile_seq] + (tile - first_tile) * block_q
    num_tiles = tile_ends[-1]
    return tile_seq, tile_row0, tiles_per_seq, tile_ends, num_tiles


@functools.partial(jax.jit, static_argnames=("block_q", "pages_per_step", "interpret"))
def block_scores(
    index_q: jax.Array,
    index_cache: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    num_seqs: jax.Array,
    *,
    block_q: int = DEFAULT_BLOCK_Q,
    pages_per_step: int = DEFAULT_PAGES_PER_STEP,
    interpret: bool = False,
) -> jax.Array:
    """Block-max index scores ``[T, 1, P]`` (``P`` padded to the page group).

    Args:
      index_q: ``[T, 1, D]`` index queries (one index head per rank).
      index_cache: ``[pages, S/2, 2, D]`` bf16 key-only cache.
      seq_lens, block_tables, cu_q_lens, num_seqs: ragged batch metadata.
    Pages past a token's own page hold scores of keys it must not see, and
    pages past its request's last page are not written; callers mask both.
    """
    num_tokens, num_heads, head_dim = index_q.shape
    if num_heads != 1:
        raise NotImplementedError(
            f"MSA indexer scores one index head per rank, got {num_heads}."
        )
    _, rows_per_page, packing, _ = index_cache.shape
    if (packing, index_cache.dtype) != (2, jnp.bfloat16):
        raise NotImplementedError(
            f"MSA indexer supports bf16 index keys, got {index_cache.dtype}."
        )
    page_size = rows_per_page * packing
    max_num_seqs = seq_lens.shape[0]
    pages_per_seq = block_tables.shape[0] // max_num_seqs
    num_groups_per_seq = -(-pages_per_seq // pages_per_step)
    padded_pages = num_groups_per_seq * pages_per_step

    tile_seq, tile_row0, _, _, num_tiles = _tiles(
        cu_q_lens, num_seqs, num_tokens, block_q
    )
    max_tiles = tile_seq.shape[0]
    rows = tile_row0[:, None] + jnp.arange(block_q, dtype=jnp.int32)
    q_tiles = index_q[jnp.clip(rows, 0, num_tokens - 1), 0].reshape(-1, head_dim)

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
        jnp.searchsorted(group_ends, w, side="right").astype(jnp.int32),
        max_tiles - 1,
    )
    work_group = w - (group_ends[work_tile] - groups_per_tile[work_tile])
    work = (work_tile << _TILE_SHIFT) | jnp.clip(work_group, 0, _GROUP_MASK)

    def page_block(g):
        def index_map(w, work, tile_seq, block_tables):
            entry = work[w]
            seq = tile_seq[entry >> _TILE_SHIFT]
            col = jnp.minimum(
                (entry & _GROUP_MASK) * pages_per_step + g, pages_per_seq - 1
            )
            return (block_tables[seq * pages_per_seq + col], 0, 0, 0)

        return pl.BlockSpec((1, rows_per_page, 2, head_dim), index_map)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        # One step even for an empty batch; its scores are never read.
        grid=(jnp.maximum(group_ends[-1], 1),),
        in_specs=[
            pl.BlockSpec(
                (block_q, head_dim),
                lambda w, work, *_: (work[w] >> _TILE_SHIFT, 0),
            ),
            *(page_block(g) for g in range(pages_per_step)),
        ],
        out_specs=pl.BlockSpec(
            (pages_per_step, block_q),
            lambda w, work, *_: (work[w] & _GROUP_MASK, work[w] >> _TILE_SHIFT),
        ),
    )
    scores_t = pl.pallas_call(
        _kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct(
            (padded_pages, max_tiles * block_q), jnp.float32
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="minimax_m3_msa_block_scores",
    )(work, tile_seq, block_tables, q_tiles, *([index_cache] * pages_per_step))

    # Back to token order: token t of request r sits in r's tiles at its offset.
    tok = jnp.arange(num_tokens, dtype=jnp.int32)
    seq = jnp.minimum(
        jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(jnp.int32),
        max_num_seqs - 1,
    )
    _, _, tiles_per_seq, tile_ends, _ = _tiles(cu_q_lens, num_seqs, num_tokens, block_q)
    offset = tok - cu_q_lens[seq]
    first_tile = tile_ends[seq] - tiles_per_seq[seq]
    col = (first_tile + offset // block_q) * block_q + offset % block_q
    col = jnp.clip(col, 0, max_tiles * block_q - 1)
    return scores_t[:, col].T[:, None, :]
