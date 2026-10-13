"""SparseCore MoE combine, v5: v4 with tokens grouped by route count.

Same contract and output bits as ``ragged_gather_reduce_v4``::

    out[t] = bf16( sum_k  x[indices[t*K + k]] * w[t*K + k]   over valid k )

v4 runs a segmented running sum over each 16-row gathered group and stores the
running value of *every* row to a float32 accumulator row (a token's last row
can sit anywhere in the group), then packs the accumulator to bf16 in a second
pass. On the vector subcore those per-row stores, the select that restarts the
sum and the accumulator round trip are what bind v4.

v5 sorts each tile's tokens by route count ``c`` (TensorCore, one stable sort,
routing-only inputs). A group holds ``16 // c`` whole tokens of one class, so
each token's rows sit at positions known at trace time::

    run = f(row_0); run = run + f(row_i) ...      (k ascending, as v4)

and the token's value is rounded once (``plsc.pack``, round-to-nearest-even, as
v4) into its half of the packed bf16 output word and added into the output
block, which is zeroed at the tile-chunk's first group. No float32 accumulator,
no per-row store, no select. A class's last group may hold fewer tokens; its
count travels in the group word and the surplus tokens add zero. Empty tokens
have no routes and stay +0.0 (v4 writes +0.0 too). Per token the float32
products and their summation order are v4's, so the output is bit-identical to
v4 (and to v2 followed by its cast).

A group gathers the 16 sorted routes starting at its first route (a dynamic,
unaligned offset into the tile's route list); rows of that window past the
group's tokens belong to the next group and are not used.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v4 import (
    _DST_MASK,
    _EVEN_BIT,
    _col_chunk,
)

_LANES = 16
# Group word: first route (bits 0-15) | token count (16-23) | class (24-31).
# The class is read with a plain shift: on this toolchain a pl.when predicate
# built from `(word >> s) & mask == c` never fired (j-bc52ff52), `word >> 24 == c`
# does.
_START_MASK = 0xFFFF
_CNT_SHIFT = 16
_CLS_SHIFT = 24
# Per-subcore VMEM budget for the output block and the gather buffer (both
# double buffered by the pipeline), in 32-bit words.
_VMEM_WORDS = 100 * 1024


def _max_groups(tile_tokens: int, k: int) -> int:
    m = [_LANES // c for c in range(1, k + 1)]
    g = min(sum(-(-tile_tokens // mc) for mc in m), tile_tokens // min(m) + k)
    return -(-max(g, 1) // _LANES) * _LANES


def supported(
    x: jax.Array, indices: jax.Array, reduce_group_size: int, tile_tokens: int
) -> bool:
    """Whether v5 can run this call (else the caller should use v4/v2)."""
    try:
        tpu_info = pltpu.get_tpu_info()
    except ValueError:
        return False
    sc = tpu_info.sparse_core
    if sc is None or sc.num_lanes != _LANES:
        return False
    if x.dtype != jnp.bfloat16 or x.ndim != 2 or x.shape[0] % 2:
        return False
    k = reduce_group_size
    if k < 1 or k > _LANES or indices.size % k:
        return False
    num_tokens = indices.size // k
    if num_tokens % tile_tokens or tile_tokens % 16 or tile_tokens > 255:
        return False
    if tile_tokens * k + _LANES > _START_MASK:
        return False
    return x.shape[-1] % tpu_info.num_lanes == 0


def pick_tile_tokens(num_tokens: int, num_workers: int = 32) -> int:
    """The largest tile (<= 64 tokens) that still gives every subcore one.

    Bigger tiles leave fewer partial groups (fewer gathered rows per token).
    """
    for tt in (64, 32):
        if num_tokens % tt == 0 and num_tokens // tt >= num_workers:
            return tt
    return 16


def pick_col_chunk(hidden: int, tile_tokens: int) -> int:
    # out 2*(tt/2)*C + gather 2*16*C words.
    want = _VMEM_WORDS // (tile_tokens + 2 * _LANES)
    want = max(128, min(2048, want // 128 * 128))
    return _col_chunk(hidden, want)


def _prepare(indices, topk_weights, valid_rows_mask, k, tile_tokens):
    """Per-tile class-grouped route lists and group tables (TensorCore)."""
    lanes = _LANES
    num_tokens = indices.size // k
    num_tiles = num_tokens // tile_tokens
    tt = tile_tokens
    e = tt * k
    gmax = _max_groups(tt, k)
    m = jnp.asarray(np.array([lanes // c for c in range(1, k + 1)], np.int32))
    cls_ids = jnp.arange(1, k + 1, dtype=jnp.int32)

    valid = valid_rows_mask.reshape(num_tiles, tt, k)
    n = jnp.sum(valid, axis=-1, dtype=jnp.int32)  # class; 0 = empty, no routes
    key = jnp.where(valid, n[..., None], k + 1).reshape(num_tiles, e)
    local = jnp.broadcast_to(jnp.arange(e, dtype=jnp.int32)[None, :], (num_tiles, e))
    # Stable: class-major, then token, then k ascending (v4's per-token order).
    _, src, w, local_slot = jax.lax.sort(
        (
            key,
            indices.reshape(num_tiles, e).astype(jnp.int32),
            topk_weights.reshape(num_tiles, e).astype(jnp.float32),
            local,
        ),
        dimension=1,
        is_stable=True,
        num_keys=1,
    )
    dst = local_slot // k
    even = (src & 1) == 0
    aux = dst + even.astype(jnp.int32) * (1 << _EVEN_BIT)
    gidx = jnp.right_shift(src, 1)
    # Slack past the route list for the last group's 16-row window: distinct
    # real rows (never a repeated constant index), never used.
    gidx = jnp.concatenate([gidx, gidx[:, :lanes]], axis=1)
    aux = jnp.concatenate([aux, jnp.zeros((num_tiles, lanes), jnp.int32)], axis=1)
    w = jnp.concatenate([w, jnp.zeros((num_tiles, lanes), jnp.float32)], axis=1)

    # Group table: class by class, ceil(n_c / m_c) groups each.
    n_cls = jnp.sum((n[..., None] == cls_ids).astype(jnp.int32), axis=1)  # (nt, k)
    g_cls = (n_cls + m - 1) // m
    rows_cls = n_cls * cls_ids
    start_cls = jnp.cumsum(rows_cls, axis=1) - rows_cls
    gcum = jnp.cumsum(g_cls, axis=1)
    gexc = gcum - g_cls
    total = gcum[:, -1]
    num_groups = jnp.maximum(total, 1)
    g = jnp.arange(gmax, dtype=jnp.int32)[None, :, None]
    in_cls = (g >= gexc[:, None, :]) & (g < gcum[:, None, :])  # (nt, gmax, k)
    gi = g - gexc[:, None, :]
    start = start_cls[:, None, :] + gi * (m * cls_ids)[None, None, :]
    cnt = jnp.minimum(m[None, None, :], n_cls[:, None, :] - gi * m[None, None, :])
    word = start + cnt * (1 << _CNT_SHIFT) + cls_ids[None, None, :] * (1 << _CLS_SHIFT)
    grp = jnp.sum(jnp.where(in_cls, word, 0), axis=-1)
    # A tile with no routes keeps one group (class 1, count 0): the zero fill.
    grp = jnp.where((g[..., 0] == 0) & (total[:, None] == 0), 1 << _CLS_SHIFT, grp)
    grp = jnp.concatenate([grp, jnp.zeros((num_tiles, lanes), jnp.int32)], axis=1)
    ng_rep = jnp.repeat(num_groups, lanes)
    return gidx.reshape(-1), aux.reshape(-1), w.reshape(-1), grp.reshape(-1), ng_rep


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
    sem,
    *,
    k: int,
    tile_tokens: int,
    col_chunk: int,
    num_tiles: int,
    unroll: int,
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

            def gather_map(c, g):
                start = jnp.bitwise_and(group_word(g), _START_MASK)
                return (gidx_vmem[pl.ds(start, lanes)], c)

            def step(gather_ref, out_ref):
                g = pl.program_id(1)
                word = group_word(g)
                start = jnp.bitwise_and(word, _START_MASK)
                cls_of_group = jnp.right_shift(word, _CLS_SHIFT)
                cnt = jnp.bitwise_and(jnp.right_shift(word, _CNT_SHIFT), 0xFF)

                # The output block persists across this tile-chunk's groups.
                @pl.when(g == 0)
                def _():
                    @plsc.parallel_loop(0, col_chunk, step=lanes, unroll=unroll)
                    def _(j):
                        for p in range(half):
                            out_ref[p, pl.ds(j, lanes)] = jnp.zeros(
                                (lanes,), jnp.uint32
                            )

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

            pltpu.emit_pipeline(
                step,
                grid=(num_chunks, num_groups),
                in_specs=pl.BlockSpec((pl.Indirect(lanes), col_chunk), gather_map),
                out_specs=pl.BlockSpec((half, col_chunk), lambda c, g: (tile, c)),
            )(x_u32, out_u32)

    pl.loop(0, pl.cdiv(num_tiles, num_workers))(tile_body)


@functools.partial(
    jax.jit,
    static_argnames=("reduce_group_size", "tile_tokens", "col_chunk", "unroll"),
)
def ragged_gather_reduce_v5(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    tile_tokens: int | None = None,
    col_chunk: int | None = None,
    unroll: int = 2,
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
    if col_chunk is None:
        col_chunk = pick_col_chunk(hidden, tile_tokens)
    else:
        col_chunk = _col_chunk(hidden, col_chunk)
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
        ),
        out_type=jax.ShapeDtypeStruct((num_tokens, hidden), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        scratch_types=scratch,
        mesh=mesh,
        name="sc_ragged_gather_reduce_v5",
    )(*meta, x)
