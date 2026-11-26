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

With ``shared_down`` the block also adds a shared expert's down projection
(its [TB, i_s] activation against e4m3 weights held in VMEM), rounded as
fused_moe_gmm.shared_expert_mlp rounds it, so the routed sum is not written to
HBM and read back by a separate add.
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
# Chunks one MXU product contracts at most (bounds its weight matrix).
_PRODUCT_CHUNKS = 3
TOKEN_BLOCK = 256
_VMEM_LIMIT_BYTES = 60 * 1024 * 1024
# Output columns per shared-expert product (bounds its f32 temporary).
_SHARED_COLS = 512
# Sublanes of a 32-bit vreg tile: a slot-major route block's row padding.
_SLOT_ROWS = 8


def max_tiles(token_block: int, topk: int, num_experts: int) -> int:
    """Tiles one block can need: its routed rows plus up to two partial tiles
    per expert range, rounded up to whole contraction chunks."""
    tiles = token_block * topk // TILE_ROWS + 2 * num_experts
    return -(-tiles // _TILES_PER_CHUNK) * _TILES_PER_CHUNK


def supported(
    num_tokens: int,
    hidden: int,
    topk: int,
    num_experts: int,
    dtype,
    shared_inter: int = 0,
) -> bool:
    """``shared_inter``: the fused shared expert's intermediate size, 0 for
    none."""
    if dtype != jnp.bfloat16 or hidden % 128 or num_tokens % TOKEN_BLOCK:
        return False
    if shared_inter and (shared_inter % 128 or hidden % _SHARED_COLS):
        return False
    tiles = max_tiles(TOKEN_BLOCK, topk, num_experts)
    vmem = (
        2 * tiles * TILE_ROWS * hidden * 2  # double-buffered tiles
        + TOKEN_BLOCK * hidden * 4  # f32 accumulator
        + 2 * TOKEN_BLOCK * hidden * 2  # output blocks
        + shared_inter * hidden  # e4m3 shared weights
        + 2 * TOKEN_BLOCK * shared_inter * 2  # activation blocks
        + TOKEN_BLOCK * _PRODUCT_CHUNKS * _CHUNK_ROWS * (4 + 2)  # product weights
    )
    return vmem <= _VMEM_LIMIT_BYTES - 8 * 1024 * 1024


def _plan(pos, valid, group_sizes, num_tokens: int, topk: int, token_block: int):
    """Per block: the x tiles to load (in buffer order) and their count, and
    each route's row in the block's buffer (-1: not a local route).

    ``pos``/``valid`` are flat token-major ``[T * topk]`` or slot-major
    ``[topk, T]``; the rows come back as ``[T, topk]`` or ``[topk, T]``."""
    num_experts = group_sizes.shape[0]
    nb = num_tokens // token_block
    slot_major = pos.ndim == 2

    def by_block(a):
        # [nb, topk * token_block]: a block's routes (their order is free).
        if slot_major:
            a = a.reshape(topk, nb, token_block).transpose(1, 0, 2)
        return a.reshape(nb, -1)

    pos, valid = by_block(pos), by_block(valid)
    ends = jnp.cumsum(group_sizes)
    expert = jnp.sum(pos[..., None] >= ends, axis=2, dtype=jnp.int32)
    valid = valid & (expert < num_experts)
    big = jnp.int32(2**30)
    eid = jnp.arange(num_experts, dtype=jnp.int32)
    hit = valid[..., None] & (expert[..., None] == eid)  # [nb, k*TB, E]
    pos_b = pos[..., None]
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
    row = jnp.where(valid, route_shift + pos, -1)
    if slot_major:
        row = row.reshape(nb, topk, token_block).transpose(1, 0, 2)
        row = row.reshape(topk, num_tokens)
    else:
        row = row.reshape(num_tokens, topk)
    return (
        tiles.reshape(-1).astype(jnp.int32),
        total.astype(jnp.int32),
        row.astype(jnp.int32),
    )


def _kernel(
    tiles_ref,  # SMEM [nb * maxt]
    count_ref,  # SMEM [nb]
    valid_rows_ref,  # SMEM [1]: rows of x the GMM wrote
    row_ref,  # VMEM [TB, k], or slot-major [k padded to 8s, TB]
    w_ref,  # VMEM like row_ref
    x_hbm,  # ANY [Mx, H]
    *refs,
    maxt: int,
    topk: int,
    shared: bool,
    slot_major: bool,
    p_dtype,
):
    if shared:
        act_ref, w2s_ref, s2s_ref, *refs = refs  # [TB, i_s] bf16, [i_s, H], [1, H]
    (
        out_ref,  # VMEM [TB, H]
        buf,  # VMEM [2, maxt * 16, H]
        acc,  # VMEM [TB, H] f32
        sems,  # DMA [2]
    ) = refs
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

    # MXU products of the loaded rows against the block's weight matrix (at
    # most `topk` nonzeros per token row), each contracted over up to
    # _PRODUCT_CHUNKS chunks: the MXU sums those chunks in its accumulator,
    # where a product per chunk popped every chunk's [TB, H] f32 result and
    # summed them through VMEM.
    rows = row_ref[...]
    w = w_ref[...].astype(jnp.float32)
    if slot_major:
        # The block's lane-dense slot rows to tokens on sublanes (XLU).
        rows, w = rows.T[:, :topk], w.T[:, :topk]

    def product(first, chunks):
        size = chunks * _CHUNK_ROWS
        lane = lax.broadcasted_iota(jnp.int32, (1, size), 1)
        local = rows - first * _CHUNK_ROWS
        p = jnp.zeros((rows.shape[0], size), jnp.float32)
        for j in range(topk):
            p = jnp.where(local[:, j : j + 1] == lane, w[:, j : j + 1], p)
        start = pl.multiple_of(first * _CHUNK_ROWS, _CHUNK_ROWS)
        xs = buf[parity, pl.ds(start, size), :]
        return jnp.dot(
            p.astype(p_dtype), xs.astype(p_dtype), preferred_element_type=jnp.float32
        )

    def accumulate(first, chunks):
        @pl.when(first == 0)
        def _():
            acc[...] = product(first, chunks)

        @pl.when(first > 0)
        def _():
            acc[...] += product(first, chunks)

    # No product reads past the buffer (maxt tiles, whole chunks).
    group = min(_PRODUCT_CHUNKS, maxt // _TILES_PER_CHUNK)
    full = n_chunks // group

    def full_product(g, carry):
        accumulate(g * group, group)
        return carry

    lax.fori_loop(0, full, full_product, 0)
    rest = n_chunks - full * group
    first = full * group
    for chunks in range(1, group):

        @pl.when(rest == chunks)
        def _(chunks=chunks):
            accumulate(first, chunks)

    def finish(total):
        if not shared:
            out_ref[...] = total.astype(out_ref.dtype)
            return
        act = act_ref[...]
        for c0 in range(0, hidden, _SHARED_COLS):
            cols = slice(c0, c0 + _SHARED_COLS)
            y = jnp.dot(act, w2s_ref[:, cols], preferred_element_type=jnp.float32)
            y = (y.astype(jnp.bfloat16).astype(jnp.float32) * s2s_ref[:, cols]).astype(
                jnp.bfloat16
            )
            routed = total[:, cols].astype(out_ref.dtype).astype(jnp.float32)
            out_ref[:, cols] = (routed + y.astype(jnp.float32)).astype(out_ref.dtype)

    @pl.when(n_chunks == 0)
    def _():
        finish(jnp.zeros(acc.shape, jnp.float32))

    @pl.when(n_chunks > 0)
    def _():
        finish(acc[...])


@functools.partial(jax.jit, static_argnames=("topk", "fallback", "interpret"))
def tc_combine(
    x: jax.Array,
    pos: jax.Array,
    weights: jax.Array,
    valid: jax.Array,
    group_sizes: jax.Array,
    *,
    topk: int,
    shared_down: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    fallback: bool = True,
    interpret=False,
) -> jax.Array:
    """out[t] = sum_j where(valid, weights * x[pos], 0) over j < topk, f32
    accumulation, bf16 out. `x` rows [0, sum(group_sizes)) are the GMM output
    grouped by local expert (sizes `group_sizes`), tokens ascending within a
    group; `pos`, `weights`, `valid` are flat [T * topk] in token-major order,
    or [topk, T] slot-major (how the routing prep lays them out: a [T, topk]
    layout pads topk to the 128 lanes and costs relayouts per array).
    ``shared_down`` (activation [T, i_s] bf16, w2 [i_s, H] e4m3, s2 [1, H]
    f32) adds bf16(bf16(act @ w2) * s2) to the bf16 sum, in bf16.
    ``fallback=False`` drops the gather fallback for rows out of token order
    (a caller whose routes come from a stable sort by expert never takes it):
    the plain kernel call lets XLA overlap an async collective with it, which
    it does not across the conditional."""
    slot_major = pos.ndim == 2
    num_tokens = pos.shape[1] if slot_major else pos.shape[0] // topk
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
    p_dtype = jnp.bfloat16 if weights.dtype == jnp.bfloat16 else jnp.float32
    if slot_major:
        # Sublane-padded so the kernel transposes whole vregs; f32 for the
        # 32-bit transpose (exact: the products round to p_dtype as before).
        slots = -(-topk // _SLOT_ROWS) * _SLOT_ROWS
        pad = ((0, slots - topk), (0, 0))
        rows = jnp.pad(rows, pad, constant_values=-1)
        w = jnp.pad(weights.astype(jnp.float32), pad)
        route_spec = pl.BlockSpec((slots, token_block), lambda b, *_: (0, b))
    else:
        w = weights.reshape(num_tokens, topk)
        if w.dtype != jnp.bfloat16:
            w = w.astype(jnp.float32)
        route_spec = pl.BlockSpec((token_block, topk), lambda b, *_: (b, 0))
    shared_operands, shared_specs = (), []
    if shared_down is not None:
        act, w2s, s2s = shared_down
        shared_operands = (act, w2s, s2s.astype(jnp.float32).reshape(1, hidden))
        shared_specs = [
            pl.BlockSpec((token_block, act.shape[1]), lambda b, *_: (b, 0)),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        grid=(nb,),
        in_specs=[
            route_spec,
            route_spec,
            pl.BlockSpec(memory_space=pl.ANY),
            *shared_specs,
        ],
        out_specs=pl.BlockSpec((token_block, hidden), lambda b, *_: (b, 0)),
        scratch_shapes=[
            pltpu.VMEM((2, maxt * TILE_ROWS, hidden), x.dtype),
            pltpu.VMEM((token_block, hidden), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
        ],
    )
    combine = pl.pallas_call(
        functools.partial(
            _kernel,
            maxt=maxt,
            topk=topk,
            shared=shared_down is not None,
            slot_major=slot_major,
            p_dtype=p_dtype,
        ),
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
    if not fallback:
        return combine(tiles, counts, valid_rows, rows, w, x, *shared_operands)
    if slot_major:
        pos, weights, valid = (a.T.reshape(-1) for a in (pos, weights, valid))
    return lax.cond(
        jnp.any(counts > maxt),
        lambda: _gather_combine(x, pos, weights, valid, topk, shared_down),
        lambda: combine(tiles, counts, valid_rows, rows, w, x, *shared_operands),
    )


def _gather_combine(x, pos, weights, valid, topk, shared_down=None):
    rows = x[pos].astype(jnp.float32) * weights.astype(jnp.float32)[:, None]
    rows = jnp.where(valid[:, None], rows, 0.0)
    out = rows.reshape(-1, topk, x.shape[1]).sum(axis=1).astype(jnp.bfloat16)
    if shared_down is None:
        return out
    act, w2s, s2s = shared_down
    y = jnp.matmul(act, w2s.astype(jnp.bfloat16)).astype(jnp.float32) * s2s
    return y.astype(jnp.bfloat16) + out
