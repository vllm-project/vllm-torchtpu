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
import os

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
# Shared-expert epilogue: output columns per MXU product / store.
_SHARED_N_CHUNK = 512
# Shared expert's product computed before the block's tile wait (overlaps the
# tile DMAs; one more bf16 [TB, H] scratch). TPU_MOE_TC_COMBINE_SHARED_EARLY.
_SHARED_EARLY = os.environ.get("TPU_MOE_TC_COMBINE_SHARED_EARLY", "0") == "1"
# VMEM limit when the shared expert's down weight is resident.
_SHARED_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


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


def supported_shared(
    num_tokens: int, hidden: int, topk: int, num_experts: int, dtype, shared_k: int
) -> bool:
    """`supported` plus the shared expert's resident bf16 down weight
    [shared_k, hidden] and double-buffered activation blocks."""
    if not supported(num_tokens, hidden, topk, num_experts, dtype):
        return False
    if shared_k % 128 or hidden % _SHARED_N_CHUNK:
        return False
    tiles = max_tiles(TOKEN_BLOCK, topk, num_experts)
    vmem = (
        2 * tiles * TILE_ROWS * hidden * 2
        + TOKEN_BLOCK * hidden * 4
        + 2 * TOKEN_BLOCK * hidden * 2
        + shared_k * hidden * 2  # resident down weight
        + 2 * TOKEN_BLOCK * shared_k * 2  # activation blocks
        + 2 * hidden * 2  # scale
        + (TOKEN_BLOCK * hidden * 2 if _SHARED_EARLY else 0)  # early product
    )
    return vmem <= _SHARED_VMEM_LIMIT_BYTES - 6 * 1024 * 1024


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
    *rest,
    maxt: int,
    topk: int,
    shared: bool = False,
    routed_scale: float = 1.0,
    early: bool = False,
):
    sbuf = None
    if shared and early:
        (act_ref, ws_hbm, wscale_ref, out_ref, buf, acc, sems, ws_vmem, ws_sem,
         sbuf) = rest
    elif shared:
        # act_ref VMEM [TB, Ks]; ws_hbm ANY [Ks, H] bf16; wscale_ref VMEM [1, H]
        act_ref, ws_hbm, wscale_ref, out_ref, buf, acc, sems, ws_vmem, ws_sem = rest
    else:
        out_ref, buf, acc, sems = rest
    b = pl.program_id(0)
    nb = pl.num_programs(0)

    if shared:
        # The shared expert's down weight stays in VMEM for the whole grid
        # (sequential, one core): one DMA at the first block.
        # Started here, waited for just before the first epilogue, so it
        # overlaps the first block's tile loads and combine products.
        @pl.when(b == 0)
        def _():
            pltpu.make_async_copy(ws_hbm, ws_vmem, ws_sem.at[0]).start()

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

    if early:
        # The shared down projection does not need this block's tiles: run
        # it while their DMAs (issued during the previous block) land.
        @pl.when(b == 0)
        def _():
            pltpu.make_async_copy(ws_hbm, ws_vmem, ws_sem.at[0]).wait()

        act0 = act_ref[...]
        for n0 in range(0, buf.shape[-1], _SHARED_N_CHUNK):
            sl0 = slice(n0, n0 + _SHARED_N_CHUNK)
            s0 = jnp.dot(act0, ws_vmem[:, sl0], preferred_element_type=jnp.float32)
            sbuf[:, sl0] = s0.astype(jnp.bfloat16) * wscale_ref[:, sl0]

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

    if not shared:

        @pl.when(n_chunks == 0)
        def _():
            out_ref[...] = jnp.zeros_like(out_ref)

        @pl.when(n_chunks > 0)
        def _():
            out_ref[...] = acc[...].astype(out_ref.dtype)

        return

    # Shared expert down projection fused into the output pass, rounded as
    # the unfused graph rounds: routed = bf16(acc) * routed_scale (bf16),
    # shared = bf16(act @ W) * w_scale (bf16), out = bf16(routed + shared).
    @pl.when(n_chunks == 0)
    def _():
        acc[...] = jnp.zeros_like(acc)

    if not early:

        @pl.when(b == 0)
        def _():
            pltpu.make_async_copy(ws_hbm, ws_vmem, ws_sem.at[0]).wait()

        act = act_ref[...]
    for n0 in range(0, hidden, _SHARED_N_CHUNK):
        sl = slice(n0, n0 + _SHARED_N_CHUNK)
        if early:
            s = sbuf[:, sl]
        else:
            s = jnp.dot(act, ws_vmem[:, sl], preferred_element_type=jnp.float32)
            s = s.astype(jnp.bfloat16) * wscale_ref[:, sl]
        r = acc[:, sl].astype(jnp.bfloat16)
        if routed_scale != 1.0:
            r = r * jnp.asarray(routed_scale, jnp.bfloat16)
        out_ref[:, sl] = (r.astype(jnp.float32) + s.astype(jnp.float32)).astype(
            out_ref.dtype
        )


@functools.partial(jax.jit, static_argnames=("topk", "fallback", "interpret"))
def tc_combine(
    x: jax.Array,
    pos: jax.Array,
    weights: jax.Array,
    valid: jax.Array,
    group_sizes: jax.Array,
    *,
    topk: int,
    fallback: bool = True,
    interpret=False,
) -> jax.Array:
    """out[t] = sum_j where(valid, weights * x[pos], 0) over j < topk, f32
    accumulation, bf16 out. `x` rows [0, sum(group_sizes)) are the GMM output
    grouped by local expert (sizes `group_sizes`), tokens ascending within a
    group; `pos`, `weights`, `valid` are flat [T * topk] in token-major order.
    The tokens may be any whole-block run of a step's tokens (a row piece:
    `pos` still indexes the step's `x`).

    ``fallback=False`` drops the gather fallback for rows out of token order
    (a caller whose routes come from a stable sort by expert never takes
    it): the plain kernel call lets XLA overlap an async collective with it,
    which it does not across the conditional."""
    return _tc_combine(
        x, pos, weights, valid, group_sizes, None, topk, interpret, fallback
    )


@functools.partial(
    jax.jit,
    static_argnames=("topk", "routed_scale", "early", "fallback", "interpret"),
)
def tc_combine_shared(
    x: jax.Array,
    pos: jax.Array,
    weights: jax.Array,
    valid: jax.Array,
    group_sizes: jax.Array,
    shared_act: jax.Array,
    shared_w: jax.Array,
    shared_scale: jax.Array,
    *,
    topk: int,
    routed_scale: float = 1.0,
    early: bool | None = None,
    fallback: bool = True,
    interpret=False,
) -> jax.Array:
    """`tc_combine` with the shared expert's down projection in its output
    pass: bf16(bf16(combine) * routed_scale + bf16(shared_act @ shared_w) *
    shared_scale), the roundings of the unfused graph (combine, routed scale,
    shared down_proj with its per-channel power-of-two scale, bf16 add).

    `shared_act` [T, Ks] bf16, `shared_w` [Ks, H] (cast to bf16), and
    `shared_scale` [H]. The shared [T, H] partial never round-trips HBM."""
    shared = (
        shared_act.astype(jnp.bfloat16),
        shared_w.astype(jnp.bfloat16),
        shared_scale.astype(jnp.bfloat16).reshape(1, -1),
        float(routed_scale),
        _SHARED_EARLY if early is None else bool(early),
    )
    return _tc_combine(
        x, pos, weights, valid, group_sizes, shared, topk, interpret, fallback
    )


def _shared_epilogue(routed, shared):
    act, w, scale, routed_scale = shared[:4]
    s = jnp.dot(act, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    s = s * scale
    r = routed.astype(jnp.bfloat16)
    if routed_scale != 1.0:
        r = r * jnp.asarray(routed_scale, jnp.bfloat16)
    return (r.astype(jnp.float32) + s.astype(jnp.float32)).astype(jnp.bfloat16)


def _tc_combine(
    x, pos, weights, valid, group_sizes, shared, topk, interpret, fallback=True
):
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
    in_specs = [
        pl.BlockSpec((token_block, topk), lambda b, *_: (b, 0)),
        pl.BlockSpec((token_block, topk), lambda b, *_: (b, 0)),
        pl.BlockSpec(memory_space=pl.ANY),
    ]
    scratch = [
        pltpu.VMEM((2, maxt * TILE_ROWS, hidden), x.dtype),
        pltpu.VMEM((token_block, hidden), jnp.float32),
        pltpu.SemaphoreType.DMA((2,)),
    ]
    operands = [tiles, counts, valid_rows, rows, w, x]
    vmem_limit = _VMEM_LIMIT_BYTES
    kernel_kw = {"maxt": maxt, "topk": topk}
    if shared is not None:
        act, ws, wscale, routed_scale, early = shared
        shared_k = act.shape[1]
        in_specs += [
            pl.BlockSpec((token_block, shared_k), lambda b, *_: (b, 0)),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec((1, hidden), lambda b, *_: (0, 0)),
        ]
        scratch += [
            pltpu.VMEM((shared_k, hidden), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((1,)),
        ]
        if early:
            scratch.append(pltpu.VMEM((token_block, hidden), jnp.bfloat16))
        operands += [act, ws, wscale]
        vmem_limit = _SHARED_VMEM_LIMIT_BYTES
        kernel_kw.update(shared=True, routed_scale=routed_scale, early=early)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=3,
        grid=(nb,),
        in_specs=in_specs,
        out_specs=pl.BlockSpec((token_block, hidden), lambda b, *_: (b, 0)),
        scratch_shapes=scratch,
    )
    combine = pl.pallas_call(
        functools.partial(_kernel, **kernel_kw),
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((num_tokens, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit,
        ),
        interpret=interpret,
        name="tc_moe_combine" if shared is None else "tc_moe_combine_shared",
    )

    if not fallback:
        return combine(*operands)

    def gather():
        out = _gather_combine(x, pos, weights, valid, topk)
        return out if shared is None else _shared_epilogue(out, shared)

    # A block needs more tiles than the buffer holds only if some expert's
    # rows are not in token order; then take the plain gather.
    return lax.cond(
        jnp.any(counts > maxt),
        gather,
        lambda: combine(*operands),
    )


def _gather_combine(x, pos, weights, valid, topk):
    rows = x[pos].astype(jnp.float32) * weights.astype(jnp.float32)[:, None]
    rows = jnp.where(valid[:, None], rows, 0.0)
    return rows.reshape(-1, topk, x.shape[1]).sum(axis=1).astype(jnp.bfloat16)
