"""SparseCore MoE combine, v6: v5 without the pipeline grid.

Same contract, inputs, tiling, groups and output bits as
``ragged_gather_reduce_v5`` (same ``_prepare``, same per-group arithmetic, same
k-ascending summation order and the same ``plsc.pack`` rounding), so the output
is bit-identical to v5 (and to v4 / v2 followed by its cast).

v5 drives each tile through ``pltpu.emit_pipeline`` over a dynamic
``(col_chunks, groups)`` grid with one gather in flight. Measured on the
SparseCore (2x2x1, GLM-5.2 shapes), each step costs ~0.30 us beyond its bytes,
of which the pipeline's scheduling is only ~0.04 us; the rest is the latency of
the step's 16-row indirect gather, exposed whenever compute does not cover it.

v6 runs the same steps as two plain loops per tile -- ``fori_loop`` over column
chunks, ``fori_loop`` over the tile's groups -- with a manually multi-buffered
indirect gather (``make_async_copy(x.at[idx, cols], ...)``, ``nslots`` slots,
one DMA semaphore each, ``nslots - 1`` gathers in flight) and a double-buffered
output block written once per column chunk. Prefetches past the last step are
clamped to the last step (issued and drained, never used) instead of
predicated, so the inner loop has no branch besides v5's own class dispatch.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v4 import (
    _DST_MASK,
    _EVEN_BIT,
    _col_chunk,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v5 import (
    _CLS_SHIFT,
    _CNT_SHIFT,
    _LANES,
    _START_MASK,
    _max_groups,
    _prepare,
    pick_tile_tokens,
    supported,
)

# Gather ring depth: 3 slots (2 gathers in flight while a group computes) beat
# 2 slots by ~14 us per c8k call; 4 slots no longer fit 1024 columns.
_NSLOTS = 3
# Per-subcore VMEM for the output block (2 slots) and the gather ring, in 32-bit
# words (the subcore has 128 Ki words; the route lists take ~1 Ki).
_VMEM_WORDS = 120 * 1024


def pick_col_chunk(hidden: int, tile_tokens: int, nslots: int = _NSLOTS) -> int:
    # out 2*(tt/2)*C + gather nslots*16*C words.
    want = _VMEM_WORDS // (tile_tokens + nslots * _LANES)
    want = max(128, min(2048, want // 128 * 128))
    return _col_chunk(hidden, want)


__all__ = ["ragged_gather_reduce_v6", "supported", "pick_tile_tokens"]


def _kernel(
    gidx_hbm,
    aux_hbm,
    w_hbm,
    grp_hbm,
    ng_hbm,
    x_hbm,
    out_hbm,
    gidx_vmem,
    aux_vmem,
    w_vmem,
    grp_vmem,
    ng_vmem,
    gbuf,
    obuf,
    sem,
    gsem,
    osem,
    *,
    k: int,
    tile_tokens: int,
    col_chunk: int,
    num_tiles: int,
    unroll: int,
    nslots: int,
):
    lanes = _LANES
    e2 = tile_tokens * k + lanes
    gpad = grp_vmem.shape[0]
    hidden = x_hbm.shape[-1]
    num_chunks = hidden // col_chunk
    core_id = jax.lax.axis_index(("core", "subcore"))
    num_workers = jax.lax.axis_size(("core", "subcore"))
    # bf16 HBM refs viewed as 32-bit words pack two consecutive rows per word
    # (the even row in the low half).
    x_u32 = x_hbm.bitcast(jnp.uint32)
    out_u32 = out_hbm.bitcast(jnp.uint32)
    half = tile_tokens // 2

    def tile_body(i):
        tile = core_id + i * num_workers

        @pl.when(tile < num_tiles)
        def _():
            copies = [
                pltpu.make_async_copy(
                    gidx_hbm.at[pl.ds(tile * e2, e2)], gidx_vmem, sem.at[0]
                ),
                pltpu.make_async_copy(
                    aux_hbm.at[pl.ds(tile * e2, e2)], aux_vmem, sem.at[0]
                ),
                pltpu.make_async_copy(
                    w_hbm.at[pl.ds(tile * e2, e2)], w_vmem, sem.at[0]
                ),
                pltpu.make_async_copy(
                    grp_hbm.at[pl.ds(tile * gpad, gpad)], grp_vmem, sem.at[0]
                ),
                pltpu.make_async_copy(
                    ng_hbm.at[pl.ds(tile * lanes, lanes)], ng_vmem, sem.at[0]
                ),
            ]
            for c in copies:
                c.start()
            for c in copies:
                c.wait()
            num_groups = ng_vmem[...][0]

            def group_word(g):
                return grp_vmem[pl.ds(g, lanes)][0]

            def gslot(slot):
                return gbuf.at[pl.ds(slot * lanes, lanes)]

            def oslot(slot):
                return obuf.at[pl.ds(slot * half, half)]

            def gather(c, word, slot):
                start = jnp.bitwise_and(word, _START_MASK)
                return pltpu.make_async_copy(
                    x_u32.at[
                        gidx_vmem[pl.ds(start, lanes)], pl.ds(c * col_chunk, col_chunk)
                    ],
                    gslot(slot),
                    gsem.at[slot],
                )

            def gather_wait(slot):
                # Only the destination shape and the semaphore matter for a wait.
                pltpu.make_async_copy(
                    x_u32.at[pl.ds(0, lanes), pl.ds(0, col_chunk)],
                    gslot(slot),
                    gsem.at[slot],
                ).wait()

            def out_copy(c, slot):
                return pltpu.make_async_copy(
                    oslot(slot),
                    out_u32.at[
                        pl.ds(tile * half, half), pl.ds(c * col_chunk, col_chunk)
                    ],
                    osem.at[slot],
                )

            def step(gather_ref, out_ref, word):
                start = jnp.bitwise_and(word, _START_MASK)
                cls_of_group = jnp.right_shift(word, _CLS_SHIFT)
                cnt = jnp.bitwise_and(jnp.right_shift(word, _CNT_SHIFT), 0xFF)
                aux_vec = aux_vmem[pl.ds(start, lanes)]
                w_vec = w_vmem[pl.ds(start, lanes)]
                shifts, ws, dsts = [], [], []
                for r in range(lanes):
                    aux = aux_vec[r]
                    dsts.append(jnp.bitwise_and(aux, _DST_MASK))
                    sh = jnp.bitwise_and(jnp.right_shift(aux, _EVEN_BIT), 1) * 16
                    shifts.append(jnp.broadcast_to(sh.astype(jnp.uint32), (lanes,)))
                    ws.append(w_vec[r])

                def row(r, j):
                    v = gather_ref[r, pl.ds(j, lanes)]
                    v = jnp.bitwise_and(
                        jnp.left_shift(v, shifts[r]), jnp.uint32(0xFFFF0000)
                    )
                    return plsc.bitcast(v, jnp.float32) * ws[r]

                for cls in range(1, k + 1):
                    per = lanes // cls

                    @pl.when(cls_of_group == cls)
                    def _(cls=cls, per=per):
                        pidx, hmask = [], []
                        for t in range(per):
                            d = dsts[t * cls]
                            pidx.append(jnp.right_shift(d, 1))
                            odd = jnp.bitwise_and(d, 1) == 1
                            hm = jnp.where(
                                odd, jnp.uint32(0xFFFF0000), jnp.uint32(0x0000FFFF)
                            )
                            hm = jnp.where(t < cnt, hm, jnp.uint32(0))
                            hmask.append(jnp.broadcast_to(hm, (lanes,)))

                        @plsc.parallel_loop(0, col_chunk, step=lanes, unroll=unroll)
                        def _(j):
                            for t in range(per):
                                r0 = t * cls
                                run = row(r0, j)
                                for q in range(1, cls):
                                    run = run + row(r0 + q, j)
                                packed = plsc.pack(
                                    run, run, format=plsc.PackFormat.INTERLEAVED
                                )
                                bits = jnp.bitwise_and(
                                    plsc.bitcast(packed, jnp.uint32), hmask[t]
                                )
                                plsc.addupdate(
                                    out_ref.at[pidx[t], pl.ds(j, lanes)], bits
                                )

            # Step s = c * num_groups + g gathers into slot s % nslots; the gather
            # of step s + nslots - 1 is issued before step s waits. A prefetch
            # cursor past the last step stays on the last step (issued and
            # drained, never used), so the loops carry no predicate.
            last_c = num_chunks - 1
            last_g = num_groups - 1

            def advance(pc, pg):
                wrap = pg == last_g
                done = jnp.logical_and(wrap, pc == last_c)
                pc2 = jnp.where(
                    jnp.logical_and(wrap, jnp.logical_not(done)), pc + 1, pc
                )
                pg2 = jnp.where(done, pg, jnp.where(wrap, 0, pg + 1))
                return pc2, pg2

            pc, pg = jnp.int32(0), jnp.int32(0)
            for d in range(nslots - 1):
                gather(pc, group_word(pg), d).start()
                pc, pg = advance(pc, pg)

            def chunk_body(c, carry):
                os_ = c % 2

                # The slot's previous output block (chunk c - 2) must be out.
                @pl.when(c >= 2)
                def _():
                    out_copy(c - 2, os_).wait()

                out_ref = oslot(os_)

                # The output block persists across this tile-chunk's groups.
                @plsc.parallel_loop(0, col_chunk, step=lanes, unroll=unroll)
                def _(j):
                    for p in range(half):
                        out_ref[p, pl.ds(j, lanes)] = jnp.zeros((lanes,), jnp.uint32)

                def group_body(g, carry):
                    pc, pg = carry
                    s = c * num_groups + g
                    gather(pc, group_word(pg), (s + nslots - 1) % nslots).start()
                    gather_wait(s % nslots)
                    step(gslot(s % nslots), out_ref, group_word(g))
                    return advance(pc, pg)

                carry = jax.lax.fori_loop(0, num_groups, group_body, carry)
                out_copy(c, os_).start()
                return carry

            jax.lax.fori_loop(0, num_chunks, chunk_body, (pc, pg))
            total = num_chunks * num_groups
            for d in range(nslots - 1):
                gather_wait((total + d) % nslots)
            for c in range(max(0, num_chunks - 2), num_chunks):
                out_copy(c, c % 2).wait()

    pl.loop(0, pl.cdiv(num_tiles, num_workers))(tile_body)


@functools.partial(
    jax.jit,
    static_argnames=(
        "reduce_group_size",
        "tile_tokens",
        "col_chunk",
        "unroll",
        "nslots",
    ),
)
def ragged_gather_reduce_v6(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    tile_tokens: int | None = None,
    col_chunk: int | None = None,
    unroll: int = 2,
    nslots: int = _NSLOTS,
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
    if nslots < 2:
        raise ValueError(f"nslots must be >= 2, got {nslots}")
    if col_chunk is None:
        col_chunk = pick_col_chunk(hidden, tile_tokens, nslots)
    else:
        # A requested chunk that does not fit the deeper ring is narrowed
        # (column chunking never changes the arithmetic).
        col_chunk = _col_chunk(
            hidden, min(col_chunk, pick_col_chunk(hidden, tile_tokens, nslots))
        )
    num_tiles = num_tokens // tile_tokens
    e2 = tile_tokens * k + lanes
    gpad = _max_groups(tile_tokens, k) + lanes
    meta = _prepare(indices, topk_weights, valid_rows_mask, k, tile_tokens)
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    scratch = (
        pltpu.VMEM((e2,), jnp.int32),
        pltpu.VMEM((e2,), jnp.int32),
        pltpu.VMEM((e2,), jnp.float32),
        pltpu.VMEM((gpad,), jnp.int32),
        pltpu.VMEM((lanes,), jnp.int32),
        pltpu.VMEM((nslots * lanes, col_chunk), jnp.uint32),
        pltpu.VMEM((2 * (tile_tokens // 2), col_chunk), jnp.uint32),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((nslots,)),
        pltpu.SemaphoreType.DMA((2,)),
    )
    return core_map_helper.kernel(
        functools.partial(
            _kernel,
            k=k,
            tile_tokens=tile_tokens,
            col_chunk=col_chunk,
            num_tiles=num_tiles,
            unroll=unroll,
            nslots=nslots,
        ),
        out_type=jax.ShapeDtypeStruct((num_tokens, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=scratch,
        mesh=mesh,
        name="sc_ragged_gather_reduce_v6",
    )(*meta, x)
