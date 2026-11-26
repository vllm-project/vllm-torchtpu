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

out[t] = sum_j w[t, j] * x[pos[t, j]] over the token's local routes, where x
holds the routed rows grouped by local expert and, within an expert, in
ascending token order (the stable sorts in fused_moe_gmm guarantee it). So
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
# Tokens per grid step, widest first: a block's buffer holds its routed rows
# plus up to two partial tiles per local expert, so many local experts (a
# hierarchical EP split) fit only a narrower block.
TOKEN_BLOCKS = (256, 128)
_VMEM_LIMIT_BYTES = 60 * 1024 * 1024


def max_tiles(token_block: int, topk: int, num_experts: int) -> int:
    """Tiles one block can need: its routed rows plus up to two partial tiles
    per expert range, rounded up to whole contraction chunks."""
    tiles = token_block * topk // TILE_ROWS + 2 * num_experts
    return -(-tiles // _TILES_PER_CHUNK) * _TILES_PER_CHUNK


def pick_token_block(num_tokens: int, hidden: int, topk: int, num_experts: int) -> int:
    """The widest of TOKEN_BLOCKS that divides the tokens and whose buffers
    fit in VMEM, or 0."""
    for block in TOKEN_BLOCKS:
        if num_tokens % block:
            continue
        tiles = max_tiles(block, topk, num_experts)
        vmem = (
            2 * tiles * TILE_ROWS * hidden * 2  # double-buffered tiles
            + block * hidden * 4  # f32 accumulator
            + 2 * block * hidden * 2  # output blocks
        )
        if vmem <= _VMEM_LIMIT_BYTES - 8 * 1024 * 1024:
            return block
    return 0


def supported(num_tokens: int, hidden: int, topk: int, num_experts: int, dtype) -> bool:
    if dtype != jnp.bfloat16 or hidden % 128:
        return False
    return pick_token_block(num_tokens, hidden, topk, num_experts) > 0


def plan_stride(maxt: int) -> int:
    """Entries per block in the plan table: `maxt` tile slots, then the
    block's tile count in the last entry, padded to whole lane rows."""
    return -(-(maxt + 1) // 128) * 128


def _plan_kernel(sizes_ref, pos_ref, plan_ref, row_ref):
    """A block per row, its routes along the lanes, 8 blocks per step; the
    local experts along the lanes of the per-block tables.

    Per local expert e: the block's routes in group e's rows (pos in
    [start_e, end_e)) span one contiguous range [lo, hi) of x, since a group
    holds its tokens in ascending order; routes past the last group are not
    local. The range's 16-row tiles go into the buffer after those of the
    experts before e, and a route's buffer row is its pos shifted by where
    its expert's tiles landed.
    """
    num_experts = sizes_ref.shape[0]
    nb, stride = plan_ref.shape
    step = 8 if nb % 8 == 0 else nb
    big = jnp.int32(2**30)
    starts = [jnp.int32(0)]
    for e in range(num_experts):
        starts.append(starts[-1] + sizes_ref[e])
    lane = lax.broadcasted_iota(jnp.int32, (step, 128), 1)
    slot = lax.broadcasted_iota(jnp.int32, (step, stride), 1)
    # Inclusive prefix over the expert lanes as a triangular matmul: exact,
    # the counts are small integers.
    upper = (
        lax.broadcasted_iota(jnp.int32, (128, 128), 0)
        <= lax.broadcasted_iota(jnp.int32, (128, 128), 1)
    ).astype(jnp.bfloat16)

    def group(g, carry):
        r0 = pl.multiple_of(g * step, step)
        pos = pos_ref[pl.ds(r0, step), :]
        expert = jnp.full(pos.shape, -1, jnp.int32)
        lo = jnp.full((step, 128), big, jnp.int32)
        hi = jnp.zeros((step, 128), jnp.int32)
        for e in range(num_experts):
            hit = (pos >= starts[e]) & (pos < starts[e + 1])
            expert = jnp.where(hit, e, expert)
            lo_e = jnp.min(jnp.where(hit, pos, big), axis=1, keepdims=True)
            hi_e = jnp.max(jnp.where(hit, pos + 1, 0), axis=1, keepdims=True)
            lo = jnp.where(lane == e, lo_e, lo)
            hi = jnp.where(lane == e, hi_e, hi)
        present = hi > 0
        first = jnp.where(present, lo // TILE_ROWS, 0)
        ntiles = jnp.where(present, -(-hi // TILE_ROWS) - first, 0)
        tile_end = jnp.dot(
            ntiles.astype(jnp.bfloat16), upper, preferred_element_type=jnp.float32
        ).astype(jnp.int32)
        tile_off = tile_end - ntiles
        total = jnp.max(tile_end, axis=1, keepdims=True)
        shift = first - tile_off

        # Slot s holds tile s + first[e] - tile_off[e] of the expert e whose
        # slots [tile_off[e], tile_end[e]) contain it.
        tiles = slot
        row = pos
        for e in range(num_experts):
            own = (slot >= tile_off[:, e : e + 1]) & (slot < tile_end[:, e : e + 1])
            tiles = tiles + jnp.where(own, shift[:, e : e + 1], 0)
            row = row - jnp.where(expert == e, shift[:, e : e + 1] * TILE_ROWS, 0)
        tiles = jnp.where(slot < total, tiles, 0)
        plan_ref[pl.ds(r0, step), :] = jnp.where(slot == stride - 1, total, tiles)
        row_ref[pl.ds(r0, step), :] = jnp.where(expert >= 0, row, -1)
        return carry

    lax.fori_loop(0, nb // step, group, 0)


def _plan(pos, group_sizes, num_tokens: int, topk: int, token_block: int, interpret):
    """Per block: the x tiles to load (in buffer order) and their count, in
    one flat table of `plan_stride` entries per block, and each route's row
    in the block's buffer (-1: not a local route)."""
    if group_sizes.shape[0] > 128:
        raise ValueError("the plan holds one local expert per lane")
    nb = num_tokens // token_block
    maxt = max_tiles(token_block, topk, group_sizes.shape[0])
    stride = plan_stride(maxt)
    routes = token_block * topk
    plan, rows = pl.pallas_call(
        _plan_kernel,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((nb, stride), jnp.int32),
            jax.ShapeDtypeStruct((nb, routes), jnp.int32),
        ],
        name="tc_moe_combine_plan",
        interpret=interpret,
    )(
        group_sizes.astype(jnp.int32),
        pos.reshape(nb, routes),
    )
    return plan.reshape(-1), rows.reshape(num_tokens, topk)


def _kernel(
    plan_ref,  # SMEM [nb * stride]: per block, tile slots then the count
    valid_rows_ref,  # SMEM [1]: rows of x the GMM wrote
    row_ref,  # VMEM [TB, k]
    w_ref,  # VMEM [TB, k]
    x_hbm,  # ANY [Mx, H]
    out_ref,  # VMEM [TB, H]
    buf,  # VMEM [2, maxt * 16, H]
    acc,  # VMEM [TB, H] f32
    sems,  # DMA [2]
    *,
    stride: int,
    topk: int,
):
    b = pl.program_id(0)
    nb = pl.num_programs(0)

    def count(block):
        return plan_ref[block * stride + stride - 1]

    def tile_copy(block, i, parity):
        t = plan_ref[block * stride + i]
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

        lax.fori_loop(0, count(block), body, 0)

    @pl.when(b == 0)
    def _():
        issue(0)

    @pl.when(b + 1 < nb)
    def _():
        issue(b + 1)

    parity = b % 2
    n = count(b)

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
        @pl.when((plan_ref[b * stride + i] == edge) & (valid_rows % TILE_ROWS != 0))
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
    group_sizes: jax.Array,
    *,
    topk: int,
    interpret=False,
) -> jax.Array:
    """out[t] = sum_j weights * x[pos] over the local routes j < topk, f32
    accumulation, bf16 out. `x` rows [0, sum(group_sizes)) are the GMM output
    grouped by local expert (sizes `group_sizes`), tokens ascending within a
    group (as the stable sorts in fused_moe_gmm leave them: a block's routes
    then never need more than `max_tiles` tiles); a route whose `pos` is past
    them is not local. `pos` and `weights` are flat [T * topk] in token-major
    order."""
    num_tokens = pos.shape[0] // topk
    hidden = x.shape[1]
    num_experts = group_sizes.shape[0]
    block = pick_token_block(num_tokens, hidden, topk, num_experts)
    nb = num_tokens // block
    maxt = max_tiles(block, topk, num_experts)
    plan, rows = _plan(
        pos.astype(jnp.int32), group_sizes, num_tokens, topk, block, interpret
    )
    if x.shape[0] % TILE_ROWS:
        x = jnp.pad(x, ((0, TILE_ROWS - x.shape[0] % TILE_ROWS), (0, 0)))
    valid_rows = jnp.sum(group_sizes, dtype=jnp.int32)[None]
    w = weights.reshape(num_tokens, topk)
    if w.dtype != jnp.bfloat16:
        w = w.astype(jnp.float32)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        grid=(nb,),
        in_specs=[
            pl.BlockSpec((block, topk), lambda b, *_: (b, 0)),
            pl.BlockSpec((block, topk), lambda b, *_: (b, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
        ],
        out_specs=pl.BlockSpec((block, hidden), lambda b, *_: (b, 0)),
        scratch_shapes=[
            pltpu.VMEM((2, maxt * TILE_ROWS, hidden), x.dtype),
            pltpu.VMEM((block, hidden), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )
    return pl.pallas_call(
        functools.partial(_kernel, stride=plan_stride(maxt), topk=topk),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="tc_moe_combine",
    )(plan, valid_rows, rows, w, x)
