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
"""SparseCore destination-tiled MoE combine (segmented and route-grouped modes)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper

_LANES = 16
_DST_MASK = 0xFF
_SAME_BIT = 8
_EVEN_BIT = 9
_ZERO_BIT = 10
_START_MASK = 0xFFFF
_CNT_SHIFT = 16
_CLS_SHIFT = 24
_SC_VMEM_BUDGET_FRACTION = 0.8


def _max_groups(tile_tokens: int, k: int) -> int:
    m = [_LANES // c for c in range(1, k + 1)]
    g = min(sum(-(-tile_tokens // mc) for mc in m), tile_tokens // min(m) + k)
    return -(-max(g, 1) // _LANES) * _LANES


def _use_grouped(tile_tokens: int, k: int, grouped: bool | None) -> bool:
    if grouped is not None:
        return grouped
    return (
        tile_tokens >= 64
        and 1 <= k <= _LANES
        and tile_tokens * k + _LANES <= _START_MASK
    )


def _col_chunk(hidden: int, want: int) -> int:
    for c in range(min(want, hidden) // 128 * 128, 127, -128):
        if hidden % c == 0:
            return c
    return hidden


def _sc_budget_words(sc) -> int:
    return int(sc.vmem_capacity_bytes * _SC_VMEM_BUDGET_FRACTION) // 4


def _mode_vmem_words(
    tile_tokens: int, k: int, col_chunk: int, use_grouped: bool
) -> int:
    if use_grouped:
        reserve = (
            3 * (tile_tokens * k + _LANES) + _max_groups(tile_tokens, k) + 2 * _LANES
        )
        return reserve + (tile_tokens + 2 * _LANES) * col_chunk
    reserve = 3 * tile_tokens * k + _LANES
    return reserve + (2 * tile_tokens + 1 + 2 * _LANES) * col_chunk


def _pick_col_chunk(
    hidden: int, tile_tokens: int, k: int, sc, use_grouped: bool
) -> int:
    budget = _sc_budget_words(sc)
    reserve = _mode_vmem_words(tile_tokens, k, 0, use_grouped)
    per_col = _mode_vmem_words(tile_tokens, k, 1, use_grouped) - reserve
    cap = 2048 if use_grouped else 1024
    want = max(128, min(cap, max(budget - reserve, 0) // per_col // 128 * 128))
    return _col_chunk(hidden, want)


def supported(
    x: jax.Array,
    indices: jax.Array,
    reduce_group_size: int,
    tile_tokens: int,
    *,
    grouped: bool | None = None,
) -> bool:
    try:
        info = pltpu.get_tpu_info()
    except ValueError:
        return False
    sc = info.sparse_core
    if sc is None or sc.num_lanes != _LANES:
        return False
    if x.dtype != jnp.bfloat16 or x.ndim != 2 or x.shape[0] < 2 or x.shape[0] % 2:
        return False
    k = reduce_group_size
    if k < 1 or indices.size % k:
        return False
    num_tokens = indices.size // k
    if (
        num_tokens % tile_tokens
        or tile_tokens <= 0
        or tile_tokens % 16
        or tile_tokens > 255
    ):
        return False
    if x.shape[-1] % info.num_lanes != 0:
        return False
    use_grp = _use_grouped(tile_tokens, k, grouped)
    if use_grp and (k > _LANES or tile_tokens * k + _LANES > _START_MASK):
        return False
    col_chunk = _pick_col_chunk(x.shape[-1], tile_tokens, k, sc, use_grp)
    return _mode_vmem_words(tile_tokens, k, col_chunk, use_grp) <= _sc_budget_words(sc)


def pick_tile_tokens(
    num_tokens: int, num_workers: int = 32, *, reduce_group_size: int | None = None
) -> int:
    if (
        (reduce_group_size is None or reduce_group_size <= _LANES)
        and num_tokens % 64 == 0
        and num_tokens // 64 >= num_workers
    ):
        return 64
    if num_tokens % 32 == 0 and num_tokens // 32 >= num_workers:
        return 32
    return 16


def _sanitize_indices(
    indices: jax.Array, valid_rows_mask: jax.Array, x_rows: int
) -> jax.Array:
    clamped = jnp.clip(indices.astype(jnp.int32), 0, x_rows - 1)
    return jnp.where(valid_rows_mask.reshape(indices.shape), clamped, 0)


def _prepare_segmented(indices, topk_weights, valid_rows_mask, k, tile_tokens, x_rows):
    num_tokens = indices.size // k
    num_tiles = num_tokens // tile_tokens
    e = tile_tokens * k
    safe_idx = _sanitize_indices(indices, valid_rows_mask, x_rows)
    valid = valid_rows_mask.reshape(num_tiles, tile_tokens, k)
    empty = ~jnp.any(valid, axis=-1, keepdims=True)
    zero_route = (empty & (jnp.arange(k)[None, None, :] == 0)).reshape(num_tiles, e)
    keep = valid.reshape(num_tiles, e) | zero_route
    local = jnp.broadcast_to(jnp.arange(e, dtype=jnp.int32)[None, :], (num_tiles, e))
    _, src, w, local_slot, zero = jax.lax.sort(
        (
            (~keep).astype(jnp.int32),
            safe_idx.reshape(num_tiles, e),
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
    return gidx.reshape(-1), aux.reshape(-1), w.reshape(-1), jnp.repeat(cnt, _LANES)


def _prepare_grouped(indices, topk_weights, valid_rows_mask, k, tile_tokens, x_rows):
    num_tokens = indices.size // k
    num_tiles = num_tokens // tile_tokens
    tt = tile_tokens
    e = tt * k
    gmax = _max_groups(tt, k)
    safe_idx = _sanitize_indices(indices, valid_rows_mask, x_rows)
    m = jnp.asarray(np.array([_LANES // c for c in range(1, k + 1)], np.int32))
    cls_ids = jnp.arange(1, k + 1, dtype=jnp.int32)

    valid = valid_rows_mask.reshape(num_tiles, tt, k)
    n = jnp.sum(valid, axis=-1, dtype=jnp.int32)
    key = jnp.where(valid, n[..., None], k + 1).reshape(num_tiles, e)
    local = jnp.broadcast_to(jnp.arange(e, dtype=jnp.int32)[None, :], (num_tiles, e))
    _, src, w, local_slot = jax.lax.sort(
        (
            key,
            safe_idx.reshape(num_tiles, e),
            topk_weights.reshape(num_tiles, e).astype(jnp.float32),
            local,
        ),
        dimension=1,
        is_stable=True,
        num_keys=1,
    )
    valid_sorted = key.sort(axis=1) <= k
    w = jnp.where(valid_sorted, w, 0.0)
    dst = local_slot // k
    even = (src & 1) == 0
    aux = dst + even.astype(jnp.int32) * (1 << _EVEN_BIT)
    gidx = jnp.right_shift(src, 1)
    gidx = jnp.concatenate([gidx, jnp.zeros((num_tiles, _LANES), jnp.int32)], axis=1)
    aux = jnp.concatenate([aux, jnp.zeros((num_tiles, _LANES), jnp.int32)], axis=1)
    w = jnp.concatenate([w, jnp.zeros((num_tiles, _LANES), jnp.float32)], axis=1)

    n_cls = jnp.sum((n[..., None] == cls_ids).astype(jnp.int32), axis=1)
    g_cls = (n_cls + m - 1) // m
    rows_cls = n_cls * cls_ids
    start_cls = jnp.cumsum(rows_cls, axis=1) - rows_cls
    gcum = jnp.cumsum(g_cls, axis=1)
    gexc = gcum - g_cls
    total = gcum[:, -1]
    num_groups = jnp.maximum(total, 1)
    g = jnp.arange(gmax, dtype=jnp.int32)[None, :, None]
    in_cls = (g >= gexc[:, None, :]) & (g < gcum[:, None, :])
    gi = g - gexc[:, None, :]
    start = start_cls[:, None, :] + gi * (m * cls_ids)[None, None, :]
    cnt = jnp.minimum(m[None, None, :], n_cls[:, None, :] - gi * m[None, None, :])
    word = start + cnt * (1 << _CNT_SHIFT) + cls_ids[None, None, :] * (1 << _CLS_SHIFT)
    grp = jnp.sum(jnp.where(in_cls, word, 0), axis=-1)
    grp = jnp.where((g[..., 0] == 0) & (total[:, None] == 0), 1 << _CLS_SHIFT, grp)
    grp = jnp.concatenate([grp, jnp.zeros((num_tiles, _LANES), jnp.int32)], axis=1)
    return (
        gidx.reshape(-1),
        aux.reshape(-1),
        w.reshape(-1),
        grp.reshape(-1),
        jnp.repeat(num_groups, _LANES),
    )


def _segmented_kernel(
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
):
    lanes = _LANES
    e = tile_tokens * k
    num_chunks = x_hbm.shape[-1] // col_chunk
    core_id = jax.lax.axis_index(("core", "subcore"))
    num_workers = jax.lax.axis_size(("core", "subcore"))
    x_u32 = x_hbm.bitcast(jnp.uint32)
    out_u32 = out_hbm.bitcast(jnp.uint32)
    half = tile_tokens // 2

    def acc_at(row, j):
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
            num_groups = pl.cdiv(cnt_vmem[...][0], lanes)

            def gather_map(c, g):
                return (gidx_vmem[pl.ds(g * lanes, lanes)], c)

            def step(gather_ref, out_ref):
                g = pl.program_id(1)
                aux_vec = aux_vmem[pl.ds(g * lanes, lanes)]
                w_vec = w_vmem[pl.ds(g * lanes, lanes)]
                dsts, sames, shifts, masks, ws = [], [], [], [], []
                for r in range(lanes):
                    aux = aux_vec[r]
                    dsts.append(jnp.bitwise_and(aux, _DST_MASK))
                    sames.append(
                        jnp.bitwise_and(jnp.right_shift(aux, _SAME_BIT), 1) == 1
                    )
                    sh = jnp.bitwise_and(jnp.right_shift(aux, _EVEN_BIT), 1) * 16
                    shifts.append(jnp.broadcast_to(sh.astype(jnp.uint32), (lanes,)))
                    is_zero = jnp.bitwise_and(jnp.right_shift(aux, _ZERO_BIT), 1) == 1
                    m = jnp.where(is_zero, jnp.uint32(0), jnp.uint32(0xFFFF0000))
                    masks.append(jnp.broadcast_to(m, (lanes,)))
                    ws.append(w_vec[r])

                @plsc.parallel_loop(0, col_chunk, step=lanes, unroll=unroll)
                def _(j):
                    run = acc_vmem[acc_at(dsts[0], j)]
                    for r in range(lanes):
                        v = gather_ref[r, pl.ds(j, lanes)]
                        v = jnp.bitwise_and(jnp.left_shift(v, shifts[r]), masks[r])
                        v = plsc.bitcast(v, jnp.float32) * ws[r]
                        run = jnp.where(sames[r], run + v, v)
                        acc_vmem[acc_at(dsts[r], j)] = run

                @pl.when(g == num_groups - 1)
                def _():
                    @plsc.parallel_loop(0, col_chunk, step=lanes, unroll=unroll)
                    def _(j):
                        for p in range(half):
                            a = acc_vmem[pl.ds(2 * p * col_chunk + j, lanes)]
                            b = acc_vmem[pl.ds((2 * p + 1) * col_chunk + j, lanes)]
                            packed = plsc.pack(a, b, format=plsc.PackFormat.INTERLEAVED)
                            out_ref[p, pl.ds(j, lanes)] = plsc.bitcast(
                                packed, jnp.uint32
                            )

            pltpu.emit_pipeline(
                step,
                grid=(num_chunks, num_groups),
                in_specs=pl.BlockSpec((pl.Indirect(lanes), col_chunk), gather_map),
                out_specs=pl.BlockSpec((half, col_chunk), lambda c, g: (tile, c)),
            )(x_u32, out_u32)

    pl.loop(0, pl.cdiv(num_tiles, num_workers))(tile_body)


def _grouped_kernel(
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
    num_chunks = x_hbm.shape[-1] // col_chunk
    core_id = jax.lax.axis_index(("core", "subcore"))
    num_workers = jax.lax.axis_size(("core", "subcore"))
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
    static_argnames=(
        "reduce_group_size",
        "tile_tokens",
        "col_chunk",
        "unroll",
        "grouped",
    ),
)
def ragged_gather_reduce_v4(
    x: jax.Array,
    indices: jax.Array,
    topk_weights: jax.Array,
    valid_rows_mask: jax.Array,
    reduce_group_size: int,
    tile_tokens: int | None = None,
    col_chunk: int | None = None,
    unroll: int | None = None,
    grouped: bool | None = None,
) -> jax.Array:
    k = reduce_group_size
    info = pltpu.get_tpu_info()
    sc = info.sparse_core
    assert sc is not None
    num_tokens = indices.size // k
    if tile_tokens is None:
        tile_tokens = pick_tile_tokens(
            num_tokens, sc.num_cores * sc.num_subcores, reduce_group_size=k
        )
    use_grp = _use_grouped(tile_tokens, k, grouped)
    hidden = x.shape[-1]
    col_chunk = (
        _pick_col_chunk(hidden, tile_tokens, k, sc, use_grp)
        if col_chunk is None
        else _col_chunk(hidden, col_chunk)
    )
    if unroll is None:
        unroll = 2 if use_grp else 1
    num_tiles = num_tokens // tile_tokens
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    if use_grp:
        e2 = tile_tokens * k + _LANES
        gpad = _max_groups(tile_tokens, k) + _LANES
        meta = _prepare_grouped(
            indices, topk_weights, valid_rows_mask, k, tile_tokens, x.shape[0]
        )
        scratch = (
            pltpu.VMEM((e2,), jnp.int32),
            pltpu.VMEM((e2,), jnp.int32),
            pltpu.VMEM((e2,), jnp.float32),
            pltpu.VMEM((gpad,), jnp.int32),
            pltpu.VMEM((_LANES,), jnp.int32),
            pltpu.SemaphoreType.DMA((1,)),
        )
        kernel_fn = _grouped_kernel
    else:
        e = tile_tokens * k
        meta = _prepare_segmented(
            indices, topk_weights, valid_rows_mask, k, tile_tokens, x.shape[0]
        )
        scratch = (
            pltpu.VMEM((e,), jnp.int32),
            pltpu.VMEM((e,), jnp.int32),
            pltpu.VMEM((e,), jnp.float32),
            pltpu.VMEM((_LANES,), jnp.int32),
            pltpu.VMEM(((tile_tokens + 1) * col_chunk,), jnp.float32),
            pltpu.SemaphoreType.DMA((1,)),
        )
        kernel_fn = _segmented_kernel
    return core_map_helper.kernel(
        functools.partial(
            kernel_fn,
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
        name="sc_ragged_gather_reduce_v4",
    )(*meta, x)
