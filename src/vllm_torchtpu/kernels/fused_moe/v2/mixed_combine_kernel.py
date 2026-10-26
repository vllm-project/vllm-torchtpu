# SPDX-License-Identifier: Apache-2.0
"""The MIXED-step combine as one Pallas call, scale looked up in-kernel.

out[t] = route[t] ? sum_k f32(arrivals[min(pos[t,k], R-1)]) * coef[t,k] : 0,
coef[t,k] = mirror_flat[min(mirror_pos[t,k], M*128-1)] * w[t,k]  (f32),

with the same clamps, the same f32 coefficient and the same k order as
layer._combine_arrivals + its row_routes select, so the result is the
incumbent's bit for bit. The incumbent runs this as a SparseCore gather of
the scales, an XLA coefficient fusion, a row gather (an XLA gather fusion at
t_local 128, or moe_v2_combine's per-row DMA ring at 256) and a select; here
the scale mirror (100-400 KB) is DMA'd whole into VMEM and each slot's scale
is picked there, so the SparseCore leaves the combine.

Two gather forms:

* ``bulk`` -- the whole arrival buffer is copied into VMEM by a few large
  DMAs (one descriptor per chunk instead of one per row), and rows are then
  read by a dynamic major-index load. The scale picks run while the copy is
  in flight. Reads every arrival row, not only the routed ones.
* ``ring`` -- moe_v2_combine's per-row DMA ring (parities, lead, two
  priorities), with the coefficient computed in-kernel instead of in XLA.

The three per-slot tables and the per-token route flag travel as one i32
HBM array the kernel fetches into SMEM itself (as build_decode_combine does).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2.host import pow2_shift

# Tokens one sum-loop body handles: the scheduler's region, hence the
# register budget (as kernel.COMBINE_CHUNK).
GROUP = 16
# Static row chunks the bulk copy is split into: all in flight at once.
BULK_CHUNKS = 8
# Ring geometry (kernel.COMBINE_PARITIES / COMBINE_LEAD).
RING_PARITIES = 4
RING_LEAD = 2

# Timing-only removal arms and positive-control faults (tests only).
SKIPS = ("nodma", "nocompute", "nopick", "noout", "rowconst")
FAULTS = ("k_order", "pick_lane", "route_mul", "row_off")


def mixed_combine_supported(t_local, topk):
    return t_local > 0 and t_local % (2 * GROUP) == 0 and topk > 0


def _ring_step_tokens(t_local):
    tb = 64
    while tb >= GROUP:
        steps = t_local // tb
        if not t_local % tb and steps >= RING_PARITIES and not steps % RING_PARITIES:
            return tb
        tb //= 2
    return 0


def _pick(tab, scl_vm, p, P, mirror_elems, tile_m, shift, lane, opts):
    """coef of slot p as a [1, 1] f32: mirror scale (lane pick) * weight."""
    w = lax.bitcast_convert_type(tab[2 * P + p], jnp.float32)
    if "nopick" in opts:
        return jnp.full((1, 1), w, jnp.float32)
    mi = jnp.minimum(tab[P + p], mirror_elems - 1)
    sub = jnp.right_shift(mi, shift)
    ln = jnp.bitwise_and(mi, tile_m - 1)
    if "pick_lane" in opts:
        ln = ln + 1
    s = jnp.sum(
        jnp.where(lane == ln, scl_vm[pl.ds(sub, 1), :], 0.0), axis=1, keepdims=True
    )
    return s * w


def _mxu_coef(mir_ref, w_ref, scl_vm, coef_vm, *, M, opts):
    """coef_vm[k] = scale(mirror[:, k]) * w[:, k], lane-broadcast [t, 128], all on
    the vector unit and the MXU: the mirror row is selected by a one-hot dot over
    the scale BITS in byte planes (every product an integer below 2**8 times 0/1,
    so exact), the lane by a select and a dot with ones, and the four bytes are
    recombined into the f32 bit pattern."""
    t_local, topk = w_ref.shape
    Mp, L = scl_vm.shape
    f32, bf16, i32 = jnp.float32, jnp.bfloat16, jnp.int32
    bits = lax.bitcast_convert_type(scl_vm[...], i32)
    valid = lax.broadcasted_iota(i32, (Mp, L), 0) < M  # rows past the mirror: never read
    planes = [
        jnp.where(valid, lax.shift_right_logical(bits, i32(8 * j)) & 255, 0).astype(f32).astype(bf16)
        for j in range(4)
    ]
    rows_iota = lax.broadcasted_iota(i32, (t_local, Mp), 1)
    lane = lax.broadcasted_iota(i32, (t_local, L), 1)
    ones = jnp.ones((L, L), bf16)
    for k in range(topk):
        mi = jnp.minimum(mir_ref[:, k:k + 1], M * L - 1)  # [t, 1]
        ln = mi & (L - 1)
        if "pick_lane" in opts:
            ln = ln + 1
        oh = (lax.shift_right_logical(mi, i32(7)) == rows_iota).astype(f32).astype(bf16)
        b = None
        for j in range(4):
            g = jnp.dot(oh, planes[j], preferred_element_type=f32)  # [t, L]: mirror row bytes
            sel = jnp.where(lane == ln, g, 0.0).astype(bf16)
            byte = jnp.dot(sel, ones, preferred_element_type=f32).astype(i32)  # lane-broadcast
            part = lax.shift_left(byte, i32(8 * j))
            b = part if b is None else b | part
        scale = lax.bitcast_convert_type(b, f32)
        if "nopick" in opts:
            scale = jnp.ones_like(scale)
        coef_vm[k] = scale * w_ref[:, k:k + 1]


def _row(pos, R, opts):
    """Arrival row of a slot, clamped into the buffer as the XLA gather clamps."""
    if "rowconst" in opts:  # timing only: every slot reads row 0
        return jnp.int32(0)
    row = pos  # clamped to R - 1 in combine_tables (the XLA fusion that builds the table)
    if "row_off" in opts:  # fault: off by one row
        row = jnp.maximum(row - 1, 0)
    return row


def _route(acc, tab, P, i, opts):
    r = tab[3 * P + i]
    if "route_mul" in opts:
        return acc * r.astype(jnp.float32)
    return jnp.where(jnp.full(acc.shape, r, jnp.int32) != 0, acc, 0.0)


def _bulk_kernel(*refs, t_local, topk, out_dtype, pick, opts, strided=False):
    if pick == "mxu":
        (tables_hbm, arr_hbm, scale_hbm, mir_ref, w_ref, out_hbm,
         tab, arr_vm, scl_vm, coef_vm, tile_vm, out_vm, sems) = refs
    else:
        (tables_hbm, arr_hbm, scale_hbm, out_hbm,
         tab, arr_vm, scl_vm, coef_vm, tile_vm, out_vm, sems) = refs
    P = t_local * topk
    R, LB, L = arr_hbm.shape
    M, tile_m = scale_hbm.shape
    hidden = LB * L
    shift = pow2_shift(tile_m, "tile_m")
    fetch = pltpu.make_async_copy(tables_hbm, tab, sems.at[0])
    fetch.start()
    chunk = -(-R // BULK_CHUNKS)
    bulk = []
    if "nodma" not in opts:
        for c0 in range(0, R, chunk):
            n = min(chunk, R - c0)
            bulk.append(
                pltpu.make_async_copy(
                    arr_hbm.at[pl.ds(c0, n)], arr_vm.at[pl.ds(c0, n)], sems.at[1]
                )
            )
    for cp in bulk:
        cp.start()
    scl = pltpu.make_async_copy(scale_hbm, scl_vm.at[pl.ds(0, M)], sems.at[2])
    scl.start()
    fetch.wait()
    scl.wait()
    lane = lax.broadcasted_iota(jnp.int32, (1, tile_m), 1)
    pick_args = (P, M * tile_m, tile_m, shift, lane, opts)

    if pick == "mxu":
        _mxu_coef(mir_ref, w_ref, scl_vm, coef_vm, M=M, opts=opts)
    if pick == "pre":
        # Every slot's coefficient while the arrival copy is in flight.
        def pre(i, carry):
            for k in range(topk):
                p = i * topk + k
                c = _pick(tab, scl_vm, p, *pick_args)
                coef_vm[p] = jnp.broadcast_to(c, (1, L))
            return carry

        lax.fori_loop(0, t_local, pre, jnp.int32(0), unroll=8)
    for cp in bulk:
        cp.wait()

    def coef(p, i, k):
        if pick == "pre":
            return coef_vm[p]  # [1, L], lanes equal
        if pick == "mxu":
            return coef_vm[k, pl.ds(i, 1), :]  # [1, L], lanes equal
        return _pick(tab, scl_vm, p, *pick_args)

    def group(g, par):
        @pl.when(g >= 2)
        def _():
            pltpu.make_async_copy(out_vm.at[par], out_vm.at[par], sems.at[3 + par]).wait()

        for u in range(GROUP):
            i = g * GROUP + u
            ks = range(topk - 1, -1, -1) if "k_order" in opts else range(topk)
            acc = None
            for k in ks:
                p = i * topk + k
                if "nocompute" in opts:
                    continue
                row = _row(tab[p], R, opts)
                term = arr_vm[row].astype(jnp.float32) * coef(p, i, k)
                acc = term if acc is None else acc + term
            if acc is None:
                acc = jnp.zeros((LB, L), jnp.float32)
            _store_token(tile_vm, u, _route(acc, tab, P, i, opts), LB, out_dtype, strided)
        if "noout" not in opts:
            _relayout(tile_vm, out_vm, par, 0, 0, GROUP, LB, L, out_dtype, strided)
        pltpu.make_async_copy(
            out_vm.at[par],
            out_hbm.at[pl.ds(pl.multiple_of(g * GROUP, GROUP), GROUP)],
            sems.at[3 + par],
        ).start()

    def pair(j, carry):
        group(2 * j, 0)
        group(2 * j + 1, 1)
        return carry

    lax.fori_loop(0, t_local // (2 * GROUP), pair, jnp.int32(0))
    for par in range(2):
        pltpu.make_async_copy(out_vm.at[par], out_vm.at[par], sems.at[3 + par]).wait()


def _ring_kernel(
    tables_hbm, arr_hbm, scale_hbm, out_hbm,
    tab, rows_vm, scl_vm, tile_vm, out_vm, sems,
    *, t_local, topk, out_dtype, opts, strided=False,
):
    """kernel._build_combine_kernel's ring, coefficient picked in-kernel."""
    P = t_local * topk
    R, LB, L = arr_hbm.shape
    M, tile_m = scale_hbm.shape
    hidden = LB * L
    shift = pow2_shift(tile_m, "tile_m")
    tb = _ring_step_tokens(t_local)
    steps = t_local // tb
    groups = tb // GROUP
    # sems: 0 tables/scales, 1..PAR rows, PAR+1..2PAR out
    rsem = lambda par: sems.at[1 + par]  # noqa: E731
    osem = lambda par: sems.at[1 + RING_PARITIES + par]  # noqa: E731
    fetch = pltpu.make_async_copy(tables_hbm, tab, sems.at[0])
    fetch.start()
    fetch.wait()
    scl = pltpu.make_async_copy(scale_hbm, scl_vm, sems.at[0])
    scl.start()
    lane = lax.broadcasted_iota(jnp.int32, (1, tile_m), 1)
    pick_args = (P, M * tile_m, tile_m, shift, lane, opts)

    def start_row(base, par, i, k):
        if "nodma" in opts:
            return
        row = _row(tab[(base + i) * topk + k], R, opts)
        pltpu.make_async_copy(
            arr_hbm.at[row], rows_vm.at[par, k * tb + i], rsem(par)
        ).start(priority=k % 2)

    def one_step(step, par):
        nxt = jnp.minimum(step + RING_LEAD, steps - 1) * tb
        base = step * tb
        npar = (par + RING_LEAD) % RING_PARITIES
        if "nodma" not in opts:
            pltpu.make_async_copy(rows_vm.at[par], rows_vm.at[par], rsem(par)).wait()

        @pl.when(step >= RING_PARITIES)
        def _():
            pltpu.make_async_copy(out_vm.at[par], out_vm.at[par], osem(par)).wait()

        def chunk(c, carry):
            for u in range(GROUP):
                i = c * GROUP + u
                for k in range(topk):
                    start_row(nxt, npar, i, k)
                acc = None
                ks = range(topk - 1, -1, -1) if "k_order" in opts else range(topk)
                for k in ks:
                    if "nocompute" in opts:
                        continue
                    p = (base + i) * topk + k
                    term = rows_vm[par, k * tb + i].astype(jnp.float32) * _pick(
                        tab, scl_vm, p, *pick_args
                    )
                    acc = term if acc is None else acc + term
                if acc is None:
                    acc = jnp.zeros((LB, L), jnp.float32)
                _store_token(tile_vm, i, _route(acc, tab, P, base + i, opts), LB, out_dtype,
                             strided)
            return carry

        lax.fori_loop(0, groups, chunk, jnp.int32(0))
        for g in range(groups if "noout" not in opts else 0):
            _relayout(tile_vm, out_vm, par, g * GROUP * (LB if strided else 1), g * GROUP,
                      GROUP, LB, L, out_dtype, strided)
        pltpu.make_async_copy(
            out_vm.at[par], out_hbm.at[pl.ds(step * tb, tb)], osem(par)
        ).start()

    def step_group(j, carry):
        for par in range(RING_PARITIES):
            one_step(j * RING_PARITIES + par, par)
        return carry

    for p in range(RING_LEAD):
        for i in range(tb):
            for k in range(topk):
                start_row(p * tb, p, i, k)
    scl.wait()
    lax.fori_loop(0, steps // RING_PARITIES, step_group, jnp.int32(0))
    if "nodma" not in opts:
        for p in range(RING_LEAD):
            pltpu.make_async_copy(rows_vm.at[p], rows_vm.at[p], rsem(p)).wait()
    for p in range(RING_PARITIES):
        pltpu.make_async_copy(out_vm.at[p], out_vm.at[p], osem(p)).wait()


def _relayout(tile_vm, out_vm, par, row0, out0, n_tok, LB, L, out_dtype, strided):
    """Tokens [n_tok] of the row-layout tile -> lane-dense out_vm[par, out0:out0+n_tok].

    strided: tile_vm is f32 [tokens * LB, L] (token u's row block j at row u*LB+j);
    lane block j of n_tok tokens is ONE sublane-strided load (stride LB), cast
    and stored as an aligned [n_tok, L] tile. Otherwise tile_vm is out_dtype
    [tokens, LB, L] and the block is relaid out by a value reshape.
    """
    if strided:
        for j in range(LB):
            v = tile_vm[pl.ds(row0 + j, n_tok, stride=LB), :]
            out_vm[par, pl.ds(out0, n_tok), pl.ds(j * L, L)] = v.astype(out_dtype)
    else:
        out_vm[par, pl.ds(out0, n_tok)] = tile_vm[pl.ds(row0, n_tok)].reshape(n_tok, LB * L)


def _store_token(tile_vm, u, val, LB, out_dtype, strided):
    if strided:
        start = u * LB if isinstance(u, int) else pl.multiple_of(u * LB, 8)
        tile_vm[pl.ds(start, LB), :] = val
    else:
        tile_vm[u] = val.astype(out_dtype)


def combine_tables(pos, mirror_pos, weights, row_routes, recv_rows):
    """[3P + t] i32: min(pos, R-1) | mirror_pos | bits(weights) | route, token-major."""
    t_local, topk = weights.shape
    P = t_local * topk
    return jnp.concatenate(
        [
            jnp.minimum(pos.astype(jnp.int32).reshape(P), recv_rows - 1),
            mirror_pos.astype(jnp.int32).reshape(P),
            lax.bitcast_convert_type(weights.astype(jnp.float32), jnp.int32).reshape(P),
            row_routes.reshape(t_local).astype(jnp.int32),
        ]
    )


def build_mixed_combine(
    arrivals,
    arrival_scales,
    pos,
    mirror_pos,
    weights,
    row_routes,
    out_dtype,
    *,
    gather="bulk",
    pick="inline",
    out="strided",
    opts=(),
    interpret=False,
):
    """Combine arrivals f8[R, LB, 128] into [t_local, LB * 128] (see module doc).

    pos / mirror_pos: i32 [t_local, topk] or flat token-major ([P] or [1, P]).
    weights: f32 [t_local, topk]; row_routes: bool [t_local, 1].
    gather: "bulk" | "ring"; pick: "pre" | "inline" | "mxu" (bulk only; ring is inline).
    out: "strided" (sublane-strided relayout from an f32 tile; needs LB % 8 == 0,
    else falls back) | "reshape" (value reshape of an out_dtype tile).
    opts: timing-only removal arms (SKIPS) or positive-control faults (FAULTS).
    """
    t_local, topk = weights.shape
    R, LB, L = arrivals.shape
    M, tile_m = arrival_scales.shape
    hidden = LB * L
    P = t_local * topk
    opts = tuple(opts)
    for o in opts:
        if o not in SKIPS + FAULTS:
            raise ValueError(f"unknown option {o!r}")
    if not mixed_combine_supported(t_local, topk):
        raise ValueError(f"mixed combine needs t_local % {2 * GROUP} == 0, got {t_local}")
    tables = combine_tables(pos, mirror_pos, weights, row_routes, R)
    strided = out == "strided" and LB % 8 == 0
    tile_rows = GROUP if gather == "bulk" else _ring_step_tokens(t_local)
    tile = ((tile_rows * LB, L), jnp.float32) if strided else ((tile_rows, LB, L), out_dtype)
    vmem = pltpu.VMEM
    if gather == "bulk":
        if pick not in ("pre", "inline", "mxu"):
            raise ValueError(pick)
        kernel = functools.partial(
            _bulk_kernel, t_local=t_local, topk=topk, out_dtype=out_dtype, pick=pick, opts=opts,
            strided=strided,
        )
        scratch = (
            pltpu.SMEM((3 * P + t_local,), jnp.int32),
            vmem((R, LB, L), arrivals.dtype),
            vmem((-(-M // L) * L if pick == "mxu" else M, tile_m), jnp.float32),
            vmem((topk, t_local, L) if pick == "mxu" else (P if pick == "pre" else 1, 1, L),
                 jnp.float32),
            vmem(*tile),
            vmem((2, GROUP, hidden), out_dtype),
            pltpu.SemaphoreType.DMA((5,)),
        )
        name = f"moe_v2_mixed_combine_bulk_{pick}_t{t_local}_k{topk}"
    elif gather == "ring":
        tb = _ring_step_tokens(t_local)
        if not tb:
            raise ValueError(f"ring combine does not cover t_local={t_local}")
        kernel = functools.partial(
            _ring_kernel, t_local=t_local, topk=topk, out_dtype=out_dtype, opts=opts,
            strided=strided,
        )
        scratch = (
            pltpu.SMEM((3 * P + t_local,), jnp.int32),
            vmem((RING_PARITIES, topk * tb, LB, L), arrivals.dtype),
            vmem((M, tile_m), jnp.float32),
            vmem(*tile),
            vmem((RING_PARITIES, tb, hidden), out_dtype),
            pltpu.SemaphoreType.DMA((1 + 2 * RING_PARITIES,)),
        )
        name = f"moe_v2_mixed_combine_ring_t{t_local}_k{topk}_b{tb}"
    else:
        raise ValueError(gather)
    if strided:
        name += "_s"
    if opts:
        name += "_" + "_".join(opts)
    hbm = pltpu.MemorySpace.HBM
    space = pl.ANY if interpret else hbm
    params = dict(vmem_limit_bytes=64 * 1024 * 1024 * 98 // 100)
    if not interpret:
        params["disable_bounds_checks"] = True
    extra = []
    in_specs = [pl.BlockSpec(memory_space=space)] * 3
    if gather == "bulk" and pick == "mxu":
        extra = [mirror_pos.astype(jnp.int32).reshape(t_local, topk),
                 weights.astype(jnp.float32)]
        in_specs = in_specs + [pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)] * 2
    call = pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=pl.BlockSpec(memory_space=space),
        scratch_shapes=scratch,
        out_shape=jax.ShapeDtypeStruct((t_local, hidden), out_dtype),
        compiler_params=pltpu.CompilerParams(**params),
        interpret=interpret,
        name=name,
    )
    if interpret:
        return call(tables, arrivals, arrival_scales, *extra)
    return call(
        tables,
        # Pinned to HBM: the kernel copies them itself; unconstrained, XLA
        # would prefetch them into VMEM first (plan_kernel.build_decode_combine).
        _pin_hbm(arrivals),
        _pin_hbm(arrival_scales),
        *extra,
    )


def _pin_hbm(x):
    """with_memory_space_constraint(x, HBM), unless x is already typed HBM (an
    operand produced by another Pallas call's HBM output inside the layer)."""
    ms = getattr(getattr(x, "aval", None), "memory_space", None)
    if ms is not None and "hbm" in str(ms).lower():
        return x
    try:
        return pltpu.with_memory_space_constraint(x, pltpu.MemorySpace.HBM)
    except TypeError:
        # Already carries a Pallas HBM memory space this primitive refuses to
        # re-constrain (seen in the layer: the fused EP kernel's arrivals).
        return x


def descriptors(t_local, topk, recv_rows, gather):
    """M6 driver: DMA descriptors issued per call, bytes read (tile-padded rows)."""
    groups = t_local // GROUP
    row_bytes = 8192
    if gather == "bulk":
        n = min(BULK_CHUNKS, recv_rows) + 2 + groups
        return dict(descriptors=n, bytes_read=recv_rows * row_bytes, terms=t_local * topk)
    tb = _ring_step_tokens(t_local)
    steps = t_local // tb
    n = (steps + RING_LEAD) * tb * topk + 2 + steps
    return dict(descriptors=n, bytes_read=(steps + RING_LEAD) * tb * topk * row_bytes,
                terms=t_local * topk)
