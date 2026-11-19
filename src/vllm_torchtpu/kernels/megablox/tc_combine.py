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
"""TensorCore MoE combine for expert-sorted GMM output (EP, large batches).

out[t] = sum_j w[t, j] * x[pos[t, j]] over the token's valid (local) routes,
where x holds the routed rows grouped by local expert and, within an expert,
in ascending token order (the stable sorts in fused_moe_gmm guarantee it). So
for a block of TB tokens, expert e's rows are one contiguous range of x. Each
block DMAs the 16-row tiles covering its ranges (aligned: bf16 rows are packed
in pairs, so a single row is not a DMA unit) into VMEM, and an MXU product
with the block's [TB, rows] weight matrix (at most `topk` nonzeros per token
row) sums them. Block b+1's tiles load while block b computes.

Against the SparseCore gather-reduce this moves ~2-3x the routed bytes (tile
rounding), at TensorCore HBM bandwidth, and frees the SparseCore for the
residual all-reduces.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Rows per DMA unit: one sublane tile of packed bf16.
TILE_ROWS = 16
# Rows per MXU contraction chunk.
_CHUNK_ROWS = 256
_TILES_PER_CHUNK = _CHUNK_ROWS // TILE_ROWS
TOKEN_BLOCK = 256
_VMEM_LIMIT_BYTES = 60 * 1024 * 1024


def max_tiles(token_block: int, topk: int, num_experts: int) -> int:
    """Tiles one block can need: its routed rows plus up to two partial tiles
    per expert range, rounded up to whole contraction chunks."""
    tiles = token_block * topk // TILE_ROWS + 2 * num_experts
    return -(-tiles // _TILES_PER_CHUNK) * _TILES_PER_CHUNK


def supported(num_tokens: int, hidden: int, topk: int, num_experts: int, dtype) -> bool:
    if dtype != jnp.bfloat16 or hidden % 128 or num_tokens % TOKEN_BLOCK:
        return False
    tiles = max_tiles(TOKEN_BLOCK, topk, num_experts)
    vmem = (
        2 * tiles * TILE_ROWS * hidden * 2  # double-buffered tiles
        + TOKEN_BLOCK * hidden * 4  # f32 accumulator
        + 2 * TOKEN_BLOCK * hidden * 2  # output blocks
    )
    return vmem <= _VMEM_LIMIT_BYTES - 8 * 1024 * 1024


def _plan(pos, valid, group_sizes, num_tokens: int, topk: int, token_block: int):
    """Per block: the x tiles to load (in buffer order) and their count, and
    each route's row in the block's buffer (-1: not a local route)."""
    num_experts = group_sizes.shape[0]
    nb = num_tokens // token_block
    ends = jnp.cumsum(group_sizes)
    expert = jnp.sum(pos[:, None] >= ends[None, :], axis=1, dtype=jnp.int32)
    valid = valid & (expert < num_experts)
    big = jnp.int32(2**30)
    eid = jnp.arange(num_experts, dtype=jnp.int32)
    hit = valid[:, None] & (expert[:, None] == eid[None, :])  # [T*k, E]
    hit = hit.reshape(nb, token_block * topk, num_experts)
    pos_b = pos.reshape(nb, token_block * topk, 1)
    lo = jnp.min(jnp.where(hit, pos_b, big), axis=1)  # [nb, E]
    hi = jnp.max(jnp.where(hit, pos_b + 1, 0), axis=1)
    present = hi > 0
    first = jnp.where(present, lo // TILE_ROWS, 0)
    ntiles = jnp.where(present, -(-hi // TILE_ROWS) - first, 0)
    tile_end = jnp.cumsum(ntiles, axis=1)  # [nb, E]
    tile_off = tile_end - ntiles
    total = tile_end[:, -1]
    maxt = max_tiles(token_block, topk, num_experts)
    slot = jnp.arange(maxt, dtype=jnp.int32)
    slot_expert = jnp.sum(
        slot[None, :, None] >= tile_end[:, None, :], axis=2, dtype=jnp.int32
    )
    slot_expert = jnp.minimum(slot_expert, num_experts - 1)
    # Gathers along these small axes lower to slow XLA gathers; one-hot
    # contractions over the local experts do the same selection.
    slot_hot = slot_expert[..., None] == eid  # [nb, maxt, E]
    tiles = (
        jnp.sum(jnp.where(slot_hot, (first - tile_off)[:, None, :], 0), axis=2)
        + slot[None, :]
    )
    tiles = jnp.where(slot[None, :] < total[:, None], tiles, 0)
    # A route's buffer row: its expert's buffer offset plus its offset from
    # the first loaded tile of that expert.
    shift = (tile_off - first) * TILE_ROWS  # [nb, E]
    route_shift = jnp.sum(jnp.where(hit, shift[:, None, :], 0), axis=2)
    row = route_shift + pos.reshape(nb, -1)
    row = jnp.where(valid.reshape(nb, -1), row, -1)
    return (
        tiles.reshape(-1).astype(jnp.int32),
        total.astype(jnp.int32),
        row.reshape(num_tokens, topk).astype(jnp.int32),
    )


def _kernel(
    tiles_ref,  # SMEM [nb * maxt]
    count_ref,  # SMEM [nb]
    valid_rows_ref,  # SMEM [1]: rows of x the GMM wrote
    row_ref,  # VMEM [TB, k]
    w_ref,  # VMEM [TB, k]
    x_hbm,  # ANY [Mx, H]
    out_ref,  # VMEM [TB, H]
    buf,  # VMEM [2, maxt * 16, H]
    acc,  # VMEM [TB, H] f32
    sems,  # DMA [2]
    *,
    maxt: int,
    topk: int,
):
    b = pl.program_id(0)
    nb = pl.num_programs(0)

    def tile_copy(block, i, parity):
        t = tiles_ref[block * maxt + i]
        return pltpu.make_async_copy(
            x_hbm.at[pl.ds(pl.multiple_of(t * TILE_ROWS, TILE_ROWS), TILE_ROWS)],
            buf.at[parity, pl.ds(pl.multiple_of(i * TILE_ROWS, TILE_ROWS), TILE_ROWS)],
            sems.at[parity],
        )

    def issue(block):
        parity = block % 2

        def body(i, c):
            tile_copy(block, i, parity).start()
            return c

        lax.fori_loop(0, count_ref[block], body, 0)

    @pl.when(b == 0)
    def _():
        issue(0)

    @pl.when(b + 1 < nb)
    def _():
        issue(b + 1)

    parity = b % 2
    n = count_ref[b]

    def wait(i, c):
        tile_copy(b, 0, parity).wait()
        return c

    lax.fori_loop(0, n, wait, 0)

    n_chunks = -(-n // _TILES_PER_CHUNK)
    hidden = buf.shape[-1]

    # Stale buffer rows may hold NaN, which a zero weight does not cancel:
    # zero the tiles past this block's own up to the chunk end, and the rows
    # of the one tile that straddles the end of the GMM's valid prefix.
    def zero_tile(i, c):
        start = pl.multiple_of(i * TILE_ROWS, TILE_ROWS)
        buf[parity, pl.ds(start, TILE_ROWS), :] = jnp.zeros(
            (TILE_ROWS, hidden), buf.dtype
        )
        return c

    lax.fori_loop(n, n_chunks * _TILES_PER_CHUNK, zero_tile, 0)
    valid_rows = valid_rows_ref[0]
    edge = valid_rows // TILE_ROWS

    def mask_edge(i, c):
        @pl.when((tiles_ref[b * maxt + i] == edge) & (valid_rows % TILE_ROWS != 0))
        def _():
            start = pl.multiple_of(i * TILE_ROWS, TILE_ROWS)
            tile = buf[parity, pl.ds(start, TILE_ROWS), :]
            row = lax.broadcasted_iota(jnp.int32, (TILE_ROWS, 1), 0)
            buf[parity, pl.ds(start, TILE_ROWS), :] = jnp.where(
                row < valid_rows % TILE_ROWS, tile, jnp.zeros_like(tile)
            )

        return c

    lax.fori_loop(0, n, mask_edge, 0)

    # One MXU product per chunk of buffer rows against the block's weight
    # matrix for that chunk (at most `topk` nonzeros per token row).
    rows = row_ref[...]
    w = w_ref[...].astype(jnp.float32)
    p_dtype = jnp.bfloat16 if w_ref.dtype == jnp.bfloat16 else jnp.float32
    lane = lax.broadcasted_iota(jnp.int32, (1, _CHUNK_ROWS), 1)

    def chunk(c, carry):
        local = rows - c * _CHUNK_ROWS
        p = jnp.zeros((rows.shape[0], _CHUNK_ROWS), jnp.float32)
        for j in range(topk):
            p = jnp.where(local[:, j : j + 1] == lane, w[:, j : j + 1], p)
        start = pl.multiple_of(c * _CHUNK_ROWS, _CHUNK_ROWS)
        xs = buf[parity, pl.ds(start, _CHUNK_ROWS), :]
        p = p.astype(p_dtype)
        xs = xs.astype(p_dtype)

        @pl.when(c == 0)
        def _():
            acc[...] = jnp.dot(p, xs, preferred_element_type=jnp.float32)

        @pl.when(c > 0)
        def _():
            acc[...] += jnp.dot(p, xs, preferred_element_type=jnp.float32)

        return carry

    lax.fori_loop(0, n_chunks, chunk, 0)

    @pl.when(n_chunks == 0)
    def _():
        out_ref[...] = jnp.zeros_like(out_ref)

    @pl.when(n_chunks > 0)
    def _():
        out_ref[...] = acc[...].astype(out_ref.dtype)


@functools.partial(jax.jit, static_argnames=("topk", "interpret"))
def tc_combine(
    x: jax.Array,
    pos: jax.Array,
    weights: jax.Array,
    valid: jax.Array,
    group_sizes: jax.Array,
    *,
    topk: int,
    interpret=False,
) -> jax.Array:
    """out[t] = sum_j where(valid, weights * x[pos], 0) over j < topk, f32
    accumulation, bf16 out. `x` rows [0, sum(group_sizes)) are the GMM output
    grouped by local expert (sizes `group_sizes`), tokens ascending within a
    group; `pos`, `weights`, `valid` are flat [T * topk] in token-major order."""
    num_tokens = pos.shape[0] // topk
    hidden = x.shape[1]
    num_experts = group_sizes.shape[0]
    token_block = TOKEN_BLOCK
    nb = num_tokens // token_block
    maxt = max_tiles(token_block, topk, num_experts)
    tiles, counts, rows = _plan(
        pos.astype(jnp.int32), valid, group_sizes, num_tokens, topk, token_block
    )
    if x.shape[0] % TILE_ROWS:
        x = jnp.pad(x, ((0, TILE_ROWS - x.shape[0] % TILE_ROWS), (0, 0)))
    valid_rows = jnp.sum(group_sizes, dtype=jnp.int32)[None]
    w = weights.reshape(num_tokens, topk)
    if w.dtype != jnp.bfloat16:
        w = w.astype(jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        grid=(nb,),
        in_specs=[
            pl.BlockSpec((token_block, topk), lambda b, *_: (b, 0)),
            pl.BlockSpec((token_block, topk), lambda b, *_: (b, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec((token_block, hidden), lambda b, *_: (b, 0)),
        scratch_shapes=[
            pltpu.VMEM((2, maxt * TILE_ROWS, hidden), x.dtype),
            pltpu.VMEM((token_block, hidden), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )
    combine = pl.pallas_call(
        functools.partial(_kernel, maxt=maxt, topk=topk),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="tc_moe_combine",
    )
    # A block needs more tiles than the buffer holds only if some expert's
    # rows are not in token order; then take the plain gather.
    return lax.cond(
        jnp.any(counts > maxt),
        lambda: _gather_combine(x, pos, weights, valid, topk),
        lambda: combine(tiles, counts, valid_rows, rows, w, x),
    )


def _gather_combine(x, pos, weights, valid, topk):
    rows = x[pos].astype(jnp.float32) * weights.astype(jnp.float32)[:, None]
    rows = jnp.where(valid[:, None], rows, 0.0)
    return rows.reshape(-1, topk, x.shape[1]).sum(axis=1).astype(jnp.bfloat16)


# ---------------------------------------------------------------------------
# v2 (TPU_MOE_TC_COMBINE_V2): the same buffers and the same products as
# `tc_combine`, so the same bits, with two costs removed.
#
# 1. The plan runs as one small Pallas kernel over [k, T] (tokens on lanes)
#    instead of ~14 XLA ops on [T * k, E] one-hots and [T, k] arrays whose
#    4-wide minor dimension is padded to 128 lanes in HBM.
# 2. Within a block an expert's routed rows are one contiguous range of x, and
#    block b + 1's range of an expert starts where block b's ended; so when
#    that boundary falls inside a 16-row tile, both blocks load the same tile.
#    With ~8 routed rows per expert per block that shared tile is about half
#    of all the tile traffic. `reuse` copies it from block b's buffer in VMEM
#    (into the same buffer slot the HBM copy would have filled) instead of
#    reading it from HBM again. Measured at served routing (8192 tokens) this
#    is slower (local DMA +46%, vector copy +7%): the kernel does not wait on
#    its tile reads (dropping every HBM copy leaves its time unchanged), so
#    reuse is off by default.
# ---------------------------------------------------------------------------

_PLAN_LANES = 128  # tile slots per block in the plan (>= max_tiles)
_KPAD = 8  # top-k rows of the [k, T] routing arrays, padded to a sublane tile


def _colsum_to_row(col, n):
    """[n, 1] -> [1, n] (the same values on lanes)."""
    r = lax.broadcasted_iota(jnp.int32, (n, n), 0)
    c = lax.broadcasted_iota(jnp.int32, (n, n), 1)
    return jnp.sum(jnp.where(r == c, col, 0), axis=0, keepdims=True)


def _plan_kernel(
    pos_ref,  # VMEM [KPAD, TB] int32: route positions in x (token-major)
    valid_ref,  # VMEM [KPAD, TB] int32: 1 for a local route
    ends_ref,  # VMEM [E, 1] int32: cumsum(group_sizes)
    tiles_ref,  # out [1, 1, LANES]: x tiles of this block, in buffer order
    src_ref,  # out [1, 1, LANES]: previous block's slot holding the tile, -1
    count_ref,  # out [1, 1, LANES]: lane 0 tiles of the block, lane 1 from HBM
    row_ref,  # out [KPAD, TB] int32: each route's buffer row, -1
    prev_last,  # scratch [E, 1] int32: last tile of each expert in block b-1
    prev_slot,  # scratch [E, 1] int32: its buffer slot
    *,
    topk: int,
    num_experts: int,
):
    b = pl.program_id(0)
    e_n = num_experts
    big = jnp.int32(2**30)

    @pl.when(b == 0)
    def _():
        prev_last[...] = jnp.full((e_n, 1), -1, jnp.int32)
        prev_slot[...] = jnp.full((e_n, 1), -1, jnp.int32)

    ends = ends_ref[...]
    eid = lax.broadcasted_iota(jnp.int32, (e_n, 1), 0)
    lo = jnp.full((e_n, 1), big, jnp.int32)
    hi = jnp.zeros((e_n, 1), jnp.int32)
    routes = []
    for j in range(topk):
        p = pos_ref[j : j + 1, :]
        ex = jnp.sum((p >= ends).astype(jnp.int32), axis=0, keepdims=True)
        v = (valid_ref[j : j + 1, :] != 0) & (ex < e_n)
        hit = v & (ex == eid)  # [E, TB]
        lo = jnp.minimum(lo, jnp.min(jnp.where(hit, p, big), axis=1, keepdims=True))
        hi = jnp.maximum(hi, jnp.max(jnp.where(hit, p + 1, 0), axis=1, keepdims=True))
        routes.append((p, v, hit))
    present = hi > 0
    first = jnp.where(present, lax.div(jnp.where(present, lo, 0), TILE_ROWS), 0)
    ntiles = jnp.where(present, lax.div(hi + (TILE_ROWS - 1), TILE_ROWS) - first, 0)
    nt_row = _colsum_to_row(ntiles, e_n)  # [1, E]
    r = lax.broadcasted_iota(jnp.int32, (e_n, e_n), 0)
    c = lax.broadcasted_iota(jnp.int32, (e_n, e_n), 1)
    tile_end = jnp.sum(jnp.where(c <= r, nt_row, 0), axis=1, keepdims=True)  # [E, 1]
    tile_off = tile_end - ntiles
    total = jnp.sum(ntiles, axis=0, keepdims=True)  # [1, 1]

    slot = lax.broadcasted_iota(jnp.int32, (1, _PLAN_LANES), 1)
    slot_expert = jnp.sum((slot >= tile_end).astype(jnp.int32), axis=0, keepdims=True)
    slot_expert = jnp.minimum(slot_expert, e_n - 1)
    slot_hot = slot_expert == eid  # [E, LANES]
    tiles = jnp.sum(jnp.where(slot_hot, first - tile_off, 0), axis=0, keepdims=True)
    tiles = jnp.where(slot < total, tiles + slot, 0)
    tiles_ref[...] = tiles

    # The first tile of an expert's range is the last one of block b-1's range
    # of that expert when the block boundary falls inside it.
    reuse = present & (first == prev_last[...])
    src = jnp.sum(
        jnp.where((slot == tile_off) & reuse, prev_slot[...] + 1, 0),
        axis=0,
        keepdims=True,
    )
    src_ref[...] = jnp.where(slot < total, src, 0) - 1
    # lane 0: tiles of the block; lane 1: those read from HBM (not reused).
    nreuse = jnp.sum(reuse.astype(jnp.int32), axis=0, keepdims=True)
    lane = lax.broadcasted_iota(jnp.int32, (1, _PLAN_LANES), 1)
    count_ref[...] = jnp.where(lane == 0, total, total - nreuse)
    prev_last[...] = jnp.where(present, first + ntiles - 1, -1)
    prev_slot[...] = tile_end - 1

    shift = (tile_off - first) * TILE_ROWS  # [E, 1]
    for j, (p, v, hit) in enumerate(routes):
        rs = jnp.sum(jnp.where(hit, shift, 0), axis=0, keepdims=True)
        row_ref[j : j + 1, :] = jnp.where(v, rs + p, -1)
    for j in range(topk, _KPAD):
        row_ref[j : j + 1, :] = jnp.full((1, row_ref.shape[1]), -1, jnp.int32)


def _plan_v2(pos_t, valid_t, ends, num_tokens, topk, token_block, interpret):
    nb = num_tokens // token_block
    num_experts = ends.shape[0]
    lanes3 = pl.BlockSpec((None, 1, _PLAN_LANES), lambda b: (b, 0, 0))
    kt = pl.BlockSpec((_KPAD, token_block), lambda b: (0, b))
    return pl.pallas_call(
        functools.partial(_plan_kernel, topk=topk, num_experts=num_experts),
        grid=(nb,),
        in_specs=[kt, kt, pl.BlockSpec((num_experts, 1), lambda b: (0, 0))],
        out_specs=[lanes3, lanes3, lanes3, kt],
        out_shape=[
            jax.ShapeDtypeStruct((nb, 1, _PLAN_LANES), jnp.int32),
            jax.ShapeDtypeStruct((nb, 1, _PLAN_LANES), jnp.int32),
            jax.ShapeDtypeStruct((nb, 1, _PLAN_LANES), jnp.int32),
            jax.ShapeDtypeStruct((_KPAD, num_tokens), jnp.int32),
        ],
        scratch_shapes=[
            pltpu.VMEM((num_experts, 1), jnp.int32),
            pltpu.VMEM((num_experts, 1), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
        name="tc_moe_combine_plan",
    )(pos_t, valid_t, ends)


def _kernel_v2(
    tiles_ref,  # SMEM [nb * LANES]
    src_ref,  # SMEM [nb * LANES]
    count_ref,  # SMEM [nb]
    hbm_count_ref,  # SMEM [nb]: tiles of the block read from HBM
    valid_rows_ref,  # SMEM [1]
    row_ref,  # VMEM [KPAD, TB] int32
    w_ref,  # VMEM [KPAD, TB] f32
    x_hbm,  # ANY [Mx, H]
    out_ref,  # VMEM [TB, H]
    buf,  # VMEM [2, maxt * 16, H]
    acc,  # VMEM [TB, H] f32
    sems,  # DMA [2]
    *,
    topk: int,
    p_dtype,
    reuse: str,
    diag: str = "",
):
    b = pl.program_id(0)
    nb = pl.num_programs(0)
    lanes = _PLAN_LANES

    def tile_dst(i, parity):
        return buf.at[parity, pl.ds(pl.multiple_of(i * TILE_ROWS, TILE_ROWS), TILE_ROWS)]

    def hbm_copy(block, i, parity):
        t = tiles_ref[block * lanes + i]
        return pltpu.make_async_copy(
            x_hbm.at[pl.ds(pl.multiple_of(t * TILE_ROWS, TILE_ROWS), TILE_ROWS)],
            tile_dst(i, parity),
            sems.at[parity],
        )

    def issue(block):
        parity = block % 2

        def body(i, c):
            if diag == "nodma":
                pass
            elif reuse:

                @pl.when(src_ref[block * lanes + i] < 0)
                def _():
                    hbm_copy(block, i, parity).start()

            else:
                hbm_copy(block, i, parity).start()
            return c

        lax.fori_loop(0, count_ref[block], body, 0)

    @pl.when(b == 0)
    def _():
        issue(0)

    @pl.when(b + 1 < nb)
    def _():
        issue(b + 1)

    parity = b % 2
    n = count_ref[b]

    def wait(i, c):
        hbm_copy(b, 0, parity).wait()
        return c

    if diag != "nodma":
        lax.fori_loop(0, hbm_count_ref[b] if reuse == "vec" else n, wait, 0)

    n_chunks = -(-n // _TILES_PER_CHUNK)
    hidden = buf.shape[-1]

    def zero_tile(i, c):
        start = pl.multiple_of(i * TILE_ROWS, TILE_ROWS)
        buf[parity, pl.ds(start, TILE_ROWS), :] = jnp.zeros((TILE_ROWS, hidden), buf.dtype)
        return c

    lax.fori_loop(n, n_chunks * _TILES_PER_CHUNK, zero_tile, 0)
    valid_rows = valid_rows_ref[0]
    edge = valid_rows // TILE_ROWS

    def mask_edge(i, c):
        @pl.when((tiles_ref[b * lanes + i] == edge) & (valid_rows % TILE_ROWS != 0))
        def _():
            start = pl.multiple_of(i * TILE_ROWS, TILE_ROWS)
            tile = buf[parity, pl.ds(start, TILE_ROWS), :]
            row = lax.broadcasted_iota(jnp.int32, (TILE_ROWS, 1), 0)
            buf[parity, pl.ds(start, TILE_ROWS), :] = jnp.where(
                row < valid_rows % TILE_ROWS, tile, jnp.zeros_like(tile)
            )

        return c

    lax.fori_loop(0, n, mask_edge, 0)

    if reuse == "vec":
        # Block b + 1's shared tiles, copied by vector loads and stores from
        # this block's finished buffer (after the edge mask, so the copy
        # carries it) into the slots its HBM copies would have filled.
        @pl.when(b + 1 < nb)
        def _():
            def local(i, c):
                s = src_ref[(b + 1) * lanes + i]

                @pl.when(s >= 0)
                def _():
                    src_start = pl.multiple_of(s * TILE_ROWS, TILE_ROWS)
                    dst_start = pl.multiple_of(i * TILE_ROWS, TILE_ROWS)
                    buf[1 - parity, pl.ds(dst_start, TILE_ROWS), :] = buf[
                        parity, pl.ds(src_start, TILE_ROWS), :
                    ]

                return c

            lax.fori_loop(0, count_ref[b + 1], local, 0)

    if reuse == "dma":
        # Block b + 1's shared tiles, from this block's finished buffer (after
        # the edge mask, so the copy carries it) into the slots its HBM copies
        # would have filled, on the semaphore block b + 1 waits on.
        @pl.when(b + 1 < nb)
        def _():
            def local(i, c):
                s = src_ref[(b + 1) * lanes + i]

                @pl.when(s >= 0)
                def _():
                    pltpu.make_async_copy(
                        tile_dst(s, parity), tile_dst(i, 1 - parity), sems.at[1 - parity]
                    ).start()

                return c

            lax.fori_loop(0, count_ref[b + 1], local, 0)

    rows = jnp.transpose(row_ref[...])  # [TB, KPAD]
    w = jnp.transpose(w_ref[...])
    lane = lax.broadcasted_iota(jnp.int32, (1, _CHUNK_ROWS), 1)

    def chunk(c, carry):
        local = rows - c * _CHUNK_ROWS
        p = jnp.zeros((rows.shape[0], _CHUNK_ROWS), jnp.float32)
        for j in range(topk):
            p = jnp.where(local[:, j : j + 1] == lane, w[:, j : j + 1], p)
        start = pl.multiple_of(c * _CHUNK_ROWS, _CHUNK_ROWS)
        xs = buf[parity, pl.ds(start, _CHUNK_ROWS), :]
        p = p.astype(p_dtype)
        xs = xs.astype(p_dtype)

        if diag == "nomm":
            acc[0:8, 0:128] = p[0:8, 0:128].astype(jnp.float32) + xs[0:8, 0:128].astype(jnp.float32)
            return carry

        @pl.when(c == 0)
        def _():
            acc[...] = jnp.dot(p, xs, preferred_element_type=jnp.float32)

        @pl.when(c > 0)
        def _():
            acc[...] += jnp.dot(p, xs, preferred_element_type=jnp.float32)

        return carry

    lax.fori_loop(0, n_chunks, chunk, 0)

    @pl.when(n_chunks == 0)
    def _():
        out_ref[...] = jnp.zeros_like(out_ref)

    @pl.when(n_chunks > 0)
    def _():
        out_ref[...] = acc[...].astype(out_ref.dtype)


_logged_v2 = False


@functools.partial(jax.jit, static_argnames=("topk", "reuse", "diag", "interpret"))
def tc_combine_v2(
    x: jax.Array,
    pos: jax.Array,
    weights: jax.Array,
    valid: jax.Array,
    group_sizes: jax.Array,
    *,
    topk: int,
    reuse: str = "",
    diag: str = "",
    interpret=False,
) -> jax.Array:
    """`tc_combine` with the plan in one Pallas kernel and, with `reuse`, each
    tile shared by consecutive blocks read from HBM once ("vec": vector copy
    in VMEM, "dma": local DMA, "": no reuse). Same result bits. `diag`
    ("nomm" / "nodma") drops the matmuls or the HBM copies, for timing only."""
    global _logged_v2
    if not _logged_v2:
        _logged_v2 = True
        print(
            f"TPU_MOE_TC_COMBINE_V2=1: tc_moe_combine_v2 traced ({pos.shape[0]} "
            f"routes, reuse={reuse})",
            flush=True,
        )
    num_tokens = pos.shape[0] // topk
    hidden = x.shape[1]
    num_experts = group_sizes.shape[0]
    token_block = TOKEN_BLOCK
    nb = num_tokens // token_block
    maxt = max_tiles(token_block, topk, num_experts)
    assert maxt <= _PLAN_LANES and topk <= _KPAD

    def kt(a, fill):
        a = a.reshape(num_tokens, topk).T
        return jnp.pad(a, ((0, _KPAD - topk), (0, 0)), constant_values=fill)

    pos_t = kt(pos.astype(jnp.int32), 0)
    valid_t = kt(valid.astype(jnp.int32), 0)
    ends = jnp.cumsum(group_sizes.astype(jnp.int32)).reshape(num_experts, 1)
    tiles, src, counts, rows_t = _plan_v2(
        pos_t, valid_t, ends, num_tokens, topk, token_block, interpret
    )
    hbm_counts = counts[:, 0, 1]
    counts = counts[:, 0, 0]
    if x.shape[0] % TILE_ROWS:
        x = jnp.pad(x, ((0, TILE_ROWS - x.shape[0] % TILE_ROWS), (0, 0)))
    valid_rows = jnp.sum(group_sizes, dtype=jnp.int32)[None]
    p_dtype = jnp.bfloat16 if weights.dtype == jnp.bfloat16 else jnp.float32
    w_t = kt(weights.astype(jnp.float32), 0.0)
    kspec = pl.BlockSpec((_KPAD, token_block), lambda b, *_: (0, b))
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=5,
        grid=(nb,),
        in_specs=[kspec, kspec, pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=pl.BlockSpec((token_block, hidden), lambda b, *_: (b, 0)),
        scratch_shapes=[
            pltpu.VMEM((2, maxt * TILE_ROWS, hidden), x.dtype),
            pltpu.VMEM((token_block, hidden), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )
    combine = pl.pallas_call(
        functools.partial(
            _kernel_v2, topk=topk, p_dtype=p_dtype, reuse=reuse, diag=diag
        ),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="tc_moe_combine_v2",
    )
    return lax.cond(
        jnp.any(counts > maxt),
        lambda: _gather_combine(x, pos, weights, valid, topk),
        lambda: combine(
            tiles.reshape(-1),
            src.reshape(-1),
            counts,
            hbm_counts,
            valid_rows,
            rows_t,
            w_t,
            x,
        ),
    )
