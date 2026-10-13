"""SparseCore MoE combine: destination-tiled gather-weight-reduce with a bf16 output.

Same contract as ``ragged_gather_reduce_v2`` (EP combine in ``moe_gmm``)::

    out[t] = bf16( sum_k  x[indices[t*K + k]] * w[t*K + k]   over valid k )

v2 keeps a running sum per source row and scatters every source row's partial
value to HBM as a separate float32 row DMA (non-final rows go to a garbage
row), waits for those 16 DMAs at the end of every column step, and leaves the
float32 -> bfloat16 cast and the empty-token mask to a TensorCore epilogue.

Here the work is tiled by *destination*: a tile of ``tile_tokens`` tokens is
owned by one vector subcore. Its routes (compacted on the TensorCore by one
stable sort, token-major / k-ascending; a token with no valid route keeps one
route masked to +0.0) are gathered 16 rows at a time by an indirect DMA, and a
segmented running sum over the rows stores each token's value into a float32
VMEM row. The finished tile is rounded to bfloat16 (``plsc.pack``,
round-to-nearest-even), packed in row pairs and written as one contiguous block
by the pipeline (asynchronously, overlapped with the next step). Per token the
float32 products and their summation order are v2's, so the output is
bit-identical to v2 followed by its cast.

The vector subcore is bound by its vector load/store issue (one gathered-row
load and one accumulator store per row and 16 lanes), not by the DMAs.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper

# aux word layout: local dst row in the low 8 bits, then flags.
_DST_MASK = 0xFF
_SAME_BIT = 8
_EVEN_BIT = 9
_ZERO_BIT = 10


def supported(
    x: jax.Array, indices: jax.Array, reduce_group_size: int, tile_tokens: int
) -> bool:
    """Whether v4 can run this call (else the caller should use v2)."""
    try:
        tpu_info = pltpu.get_tpu_info()
    except ValueError:
        return False
    sc = tpu_info.sparse_core
    if sc is None or sc.num_lanes != 16:
        return False
    if x.dtype != jnp.bfloat16 or x.ndim != 2 or x.shape[0] % 2:
        return False
    if indices.size % reduce_group_size:
        return False
    num_tokens = indices.size // reduce_group_size
    if num_tokens % tile_tokens or tile_tokens % 16 or tile_tokens > 255:
        return False
    return x.shape[-1] % tpu_info.num_lanes == 0


def pick_tile_tokens(num_tokens: int, num_workers: int = 32) -> int:
    """32-token tiles while every vector subcore still gets one, else 16."""
    if num_tokens % 32 == 0 and num_tokens // 32 >= num_workers:
        return 32
    return 16


def _col_chunk(hidden: int, want: int) -> int:
    for c in range(min(want, hidden), 15, -128):
        if hidden % c == 0 and c % 128 == 0:
            return c
    return hidden


def _prepare(indices, topk_weights, valid_rows_mask, k, tile_tokens, lanes):
    """Per-tile compacted route lists (TensorCore, routing-only inputs).

    Every token keeps its valid routes in k order; a token with none keeps one
    zero route (its k=0 slot, masked to +0.0 in the kernel), so every output row
    is written by the running sum and no separate zero fill is needed. One
    stable sort per tile carries the payloads, so no gather is emitted.
    """
    num_tokens = indices.size // k
    num_tiles = num_tokens // tile_tokens
    e = tile_tokens * k
    valid = valid_rows_mask.reshape(num_tiles, tile_tokens, k)
    empty = ~jnp.any(valid, axis=-1, keepdims=True)
    k0 = jnp.arange(k)[None, None, :] == 0
    zero_route = (empty & k0).reshape(num_tiles, e)
    keep = valid.reshape(num_tiles, e) | zero_route
    local = jnp.broadcast_to(jnp.arange(e, dtype=jnp.int32)[None, :], (num_tiles, e))
    # Stable partition: kept routes first, in token-major / k-ascending order.
    _, src, w, local_slot, zero = jax.lax.sort(
        (
            (~keep).astype(jnp.int32),
            indices.reshape(num_tiles, e).astype(jnp.int32),
            topk_weights.reshape(num_tiles, e).astype(jnp.float32),
            local,
            zero_route.astype(jnp.int32),
        ),
        dimension=1,
        is_stable=True,
        num_keys=1,
    )
    cnt = jnp.sum(keep, axis=1, dtype=jnp.int32)
    is_real = local < cnt[:, None]
    # Pad entries (past cnt) gather the tile's unkept rows (distinct rows, never
    # a repeated constant) and are masked to zero into the spare accumulator row.
    zero = (zero != 0) | ~is_real
    dst = local_slot // k
    prev_dst = jnp.concatenate(
        [jnp.full((num_tiles, 1), -1, jnp.int32), dst[:, :-1]], 1
    )
    same = is_real & (dst == prev_dst)
    even = (src & 1) == 0
    dst = jnp.where(is_real, dst, tile_tokens)
    w = jnp.where(zero, 0.0, w)
    aux = (
        dst
        + same.astype(jnp.int32) * (1 << _SAME_BIT)
        + even.astype(jnp.int32) * (1 << _EVEN_BIT)
        + zero.astype(jnp.int32) * (1 << _ZERO_BIT)
    )
    gidx = jnp.right_shift(src, 1)
    cnt_rep = jnp.repeat(cnt, lanes)
    return gidx.reshape(-1), aux.reshape(-1), w.reshape(-1), cnt_rep


def _rne_hi(u):
    """float32 bits -> bfloat16 bits (round-nearest-even) in the high 16 bits."""
    lsb = jnp.bitwise_and(jnp.right_shift(u, 16), jnp.uint32(1))
    r = u + jnp.uint32(0x7FFF) + lsb
    return jnp.bitwise_and(r, jnp.uint32(0xFFFF0000))


def _rne_lo(u):
    """float32 bits -> bfloat16 bits (round-nearest-even) in the low 16 bits."""
    lsb = jnp.bitwise_and(jnp.right_shift(u, 16), jnp.uint32(1))
    return jnp.right_shift(u + jnp.uint32(0x7FFF) + lsb, 16)


def _kernel(
    gidx_hbm,
    aux_hbm,
    w_hbm,
    cnt_hbm,
    x_hbm,
    out_hbm,
    gidx_vmem,
    aux_vmem,
    w_vmem,
    cnt_vmem,
    acc_vmem,
    sem,
    *,
    k: int,
    tile_tokens: int,
    col_chunk: int,
    num_tiles: int,
    unroll: int,
    pack_mode: str,
    skip_store: bool,
):
    lanes = 16
    e = tile_tokens * k
    hidden = x_hbm.shape[-1]
    num_chunks = hidden // col_chunk
    core_id = jax.lax.axis_index(("core", "subcore"))
    num_workers = jax.lax.axis_size(("core", "subcore"))
    # bf16 HBM refs viewed as 32-bit words pack two consecutive rows per word.
    x_u32 = x_hbm.bitcast(jnp.uint32)
    out_u32 = out_hbm.bitcast(jnp.uint32)
    half = tile_tokens // 2

    def acc_at(row, j):
        # Flat accumulator: row-major (tile_tokens + 1) x col_chunk words.
        return pl.ds(pl.multiple_of(row * col_chunk + j, lanes), lanes)

    def tile_body(i):
        tile = core_id + i * num_workers

        @pl.when(tile < num_tiles)
        def _():
            copies = [
                pltpu.make_async_copy(
                    gidx_hbm.at[pl.ds(tile * e, e)], gidx_vmem, sem.at[0]
                ),
                pltpu.make_async_copy(
                    aux_hbm.at[pl.ds(tile * e, e)], aux_vmem, sem.at[0]
                ),
                pltpu.make_async_copy(w_hbm.at[pl.ds(tile * e, e)], w_vmem, sem.at[0]),
                pltpu.make_async_copy(
                    cnt_hbm.at[pl.ds(tile * lanes, lanes)], cnt_vmem, sem.at[0]
                ),
            ]
            for c in copies:
                c.start()
            for c in copies:
                c.wait()
            cnt = cnt_vmem[...][0]
            num_groups = pl.cdiv(cnt, lanes)

            def gather_map(c, g):
                return (gidx_vmem[pl.ds(g * lanes, lanes)], c)

            def step(gather_ref, out_ref):
                g = pl.program_id(1)
                aux_vec = aux_vmem[pl.ds(g * lanes, lanes)]
                w_vec = w_vmem[pl.ds(g * lanes, lanes)]
                # Per-row scalars, hoisted out of the column loop.
                dsts, sames, shifts, masks, ws = [], [], [], [], []
                for r in range(lanes):
                    aux = aux_vec[r]
                    dsts.append(jnp.bitwise_and(aux, _DST_MASK))
                    sames.append(
                        jnp.bitwise_and(jnp.right_shift(aux, _SAME_BIT), 1) == 1
                    )
                    sh = jnp.bitwise_and(jnp.right_shift(aux, _EVEN_BIT), 1) * 16
                    shifts.append(jnp.broadcast_to(sh.astype(jnp.uint32), (lanes,)))
                    zero = jnp.bitwise_and(jnp.right_shift(aux, _ZERO_BIT), 1) == 1
                    m = jnp.where(zero, jnp.uint32(0), jnp.uint32(0xFFFF0000))
                    masks.append(jnp.broadcast_to(m, (lanes,)))
                    ws.append(w_vec[r])

                # Segmented running sum over the token-sorted rows (as v2): a row
                # either continues its token's sum or restarts it, and stores its
                # running value to its token's accumulator row; the last store per
                # token is the full sum. Row 0 continues from the accumulator when
                # its token began in an earlier group.
                @plsc.parallel_loop(0, col_chunk, step=lanes, unroll=unroll)
                def _(j):
                    run = acc_vmem[acc_at(dsts[0], j)]
                    for r in range(lanes):
                        v = gather_ref[r, pl.ds(j, lanes)]
                        v = jnp.bitwise_and(jnp.left_shift(v, shifts[r]), masks[r])
                        v = plsc.bitcast(v, jnp.float32) * ws[r]
                        run = jnp.where(sames[r], run + v, v)
                        if skip_store and r < lanes - 1:

                            @pl.when(jnp.logical_not(sames[r + 1]))
                            def _():
                                acc_vmem[acc_at(dsts[r], j)] = run

                        else:
                            acc_vmem[acc_at(dsts[r], j)] = run

                @pl.when(g == num_groups - 1)
                def _():
                    @plsc.parallel_loop(0, col_chunk, step=lanes, unroll=unroll)
                    def _(j):
                        for p in range(half):
                            a = acc_vmem[pl.ds(2 * p * col_chunk + j, lanes)]
                            b = acc_vmem[pl.ds((2 * p + 1) * col_chunk + j, lanes)]
                            if pack_mode == "pack":
                                packed = plsc.pack(
                                    a, b, format=plsc.PackFormat.INTERLEAVED
                                )
                                word = plsc.bitcast(packed, jnp.uint32)
                            else:
                                word = jnp.bitwise_or(
                                    _rne_lo(plsc.bitcast(a, jnp.uint32)),
                                    _rne_hi(plsc.bitcast(b, jnp.uint32)),
                                )
                            out_ref[p, pl.ds(j, lanes)] = word

            pltpu.emit_pipeline(
                step,
                grid=(num_chunks, num_groups),
                in_specs=pl.BlockSpec((pl.Indirect(lanes), col_chunk), gather_map),
                out_specs=pl.BlockSpec((half, col_chunk), lambda c, g: (tile, c)),
            )(x_u32, out_u32)

    pl.loop(0, pl.cdiv(num_tiles, num_workers))(tile_body)


@functools.partial(
    jax.jit,
    static_argnames=(
        "reduce_group_size",
        "tile_tokens",
        "col_chunk",
        "unroll",
        "pack_mode",
        "skip_store",
    ),
)
def ragged_gather_reduce_v4(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    tile_tokens: int | None = None,
    col_chunk: int = 1024,
    unroll: int = 1,
    pack_mode: str = "pack",
    skip_store: bool = False,
) -> jax.Array:
    """bf16 ``[num_tokens, hidden]`` combine; see the module docstring."""
    k = reduce_group_size
    tpu_info = pltpu.get_tpu_info()
    sc = tpu_info.sparse_core
    lanes = sc.num_lanes
    num_tokens = indices.size // k
    if tile_tokens is None:
        tile_tokens = pick_tile_tokens(num_tokens, sc.num_cores * sc.num_subcores)
    hidden = x.shape[-1]
    col_chunk = _col_chunk(hidden, col_chunk)
    num_tiles = num_tokens // tile_tokens
    e = tile_tokens * k
    meta = _prepare(indices, topk_weights, valid_rows_mask, k, tile_tokens, lanes)
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    scratch = (
        pltpu.VMEM((e,), jnp.int32),
        pltpu.VMEM((e,), jnp.int32),
        pltpu.VMEM((e,), jnp.float32),
        pltpu.VMEM((lanes,), jnp.int32),
        pltpu.VMEM(((tile_tokens + 1) * col_chunk,), jnp.float32),
        pltpu.SemaphoreType.DMA((1,)),
    )
    return core_map_helper.kernel(
        functools.partial(
            _kernel,
            k=k,
            tile_tokens=tile_tokens,
            col_chunk=col_chunk,
            num_tiles=num_tiles,
            unroll=unroll,
            pack_mode=pack_mode,
            skip_store=skip_store,
        ),
        out_type=jax.ShapeDtypeStruct((num_tokens, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=scratch,
        mesh=mesh,
        name="sc_ragged_gather_reduce_v4",
    )(*meta, x)
