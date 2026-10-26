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
"""The small TP all-reduce of a sublayer's partial output fused with the mHC
seam that consumes it, as one Pallas kernel (decode buckets).

At decode the seam (``fused_post_pre_kernel``) is bound by its per-call
intercept and its exposed weight DMA, and the all-reduce in front of it by
ICI latency with HBM idle. Fusing them deletes the seam's launch and
prologue, XLA's VMEM->HBM eviction of the all-reduce output and the seam's
reload of it: every seam operand (fn, residual, gates, norm weight) is DMA'd
from HBM at kernel entry, under the barrier and the two ICI rounds, and the
seam body runs on the summed block while it is still in VMEM.

The all-reduce is the in-repo kernel's math (``small_collectives.all_reduce``:
one-shot / two-shot / pairs, same f32 sums, same roundings, same group
order) and the body is ``fused_post_pre_kernel._fused_kernel``'s, so every
output is bit-identical to the unfused pair.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar
from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    barrier,
    push,
    scalar_operands,
    split_refs,
    sum_slots,
    wait_bytes_of,
)

# Scoped VMEM: the working set is <= ~16 MiB at T=32 with 72 fn rows. The
# mHC default (100 MiB, clamped to 64 MiB - 64 KiB) would make XLA evict its
# neighbours' VMEM buffers around the call.
VMEM_LIMIT_BYTES = 32 * 1024 * 1024

# Input DMA slots and output DMA slots.
_IN = ("res", "post", "comb", "fn", "sc", "hb", "nw")
_OUT = ("new_res", "post_mix", "comb_mix", "layer")

# Correctness fault controls (timing-invalid; tests only).
# A DMA or remote copy left un-waited at kernel exit halts the core (E0200),
# so the "missing wait" faults move the wait late instead of dropping it.
FAULTS = (
    "no_res_wait",  # residual DMA waited only after the body (stale VMEM)
    "fn_short",  # DMA only half the fn rows
    "x_slot",  # assemble x with owner slots 0 and 1 swapped
    "drop_round2_wait",  # last all-reduce hop waited only after the body (race)
    "no_newres_wait",  # new_res out-DMA started before the recombine stores it
)


def _form(rows, hidden, num_cores, form):
    return form or ar.form_for(rows, hidden, 2)


def _pad_rows(v, rows):
    if v.shape[0] == rows:
        return v
    return jnp.concatenate([v, jnp.zeros((rows - v.shape[0],) + v.shape[1:], v.dtype)], 0)


def _body(x_bf, res_buf, post_buf, comb_buf, fn_buf, sc_buf, hb_buf, nw_buf,
          newres_ref, ga_ref, gb_ref, layer_ref, start_out, *, hc_mult, hidden_size,
          body_rows, early_out, rms_eps, hc_pre_eps, gates, norm_eps, gemm_precision,
          s_buf=None):
    """``_fused_kernel``'s math on one whole block, op for op. ``body_rows``:
    the row count the unfused seam's block has at this T (it pads a small
    batch to a 16-row block), so the f32 reductions see the same layout and
    the bits match; the outputs keep the first T rows."""
    T = x_bf.shape[0]
    pad = functools.partial(_pad_rows, rows=body_rows)
    post = pad(post_buf[...])
    comb = pad(comb_buf[...])
    x_bf = pad(x_bf)
    old_bf = [pad(res_buf[:, i * hidden_size:(i + 1) * hidden_size]) for i in range(hc_mult)]
    streams_bf = []
    for j in range(hc_mult):
        if s_buf is not None:  # sum_i comb_ij res_i precomputed under the all-reduce
            acc = pad(s_buf[:, j * hidden_size:(j + 1) * hidden_size]) + post[:, j:j + 1] * x_bf.astype(
                jnp.float32)
        else:
            acc = post[:, j:j + 1] * x_bf.astype(jnp.float32)
            for i in range(hc_mult):
                k = i * hc_mult + j
                acc = acc + comb[:, k:k + 1] * old_bf[i].astype(jnp.float32)
        new_bf = acc.astype(jnp.bfloat16)
        newres_ref[:, j * hidden_size:(j + 1) * hidden_size] = new_bf[:T]
        streams_bf.append(new_bf)
    if early_out:
        start_out("new_res")

    g_bf = jnp.concatenate(streams_bf, axis=1)
    acc = utils.mix_gemm(g_bf, (fn_buf,), hc_mult * (hc_mult + 2), gemm_precision)
    gf = g_bf.astype(jnp.float32)
    sqr = jnp.sum(gf * gf, axis=-1, keepdims=True)
    m, h = hc_mult, hidden_size
    pre_mix, ga, gb = utils.gates_in_block(
        acc, sqr, sc_buf, hb_buf, hc_mult=m, hidden_size=h, rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps, gates=gates,
    )
    ga_ref[...] = ga[:T]
    gb_ref[...] = gb[:T]
    if early_out:
        start_out("post_mix")
        start_out("comb_mix")
    lay = pre_mix[:, 0:1] * gf[:, :h]
    for i in range(1, m):
        lay = lay + pre_mix[:, i:i + 1] * gf[:, i * h:(i + 1) * h]
    layer_ref[...] = utils.finish_layer_input(lay, nw_buf, norm_eps)[:T]


def _kernel(*refs, axis, n, form, chunks, rows, hidden, hc_mult, fn_rows, emit_x, body,
            fn_dma, fault, ar_scratch, seam_kw, in_dma, out_dma, early_out, body_rows, fn_mode,
            pre_sum):
    """in_dma: the seam operands other than fn that arrive by entry DMA from
    pl.ANY (else whole VMEM blocks XLA places). out_dma: outputs leave by
    async DMA from VMEM scratch to pl.ANY (else whole VMEM output blocks
    the body writes). fn always arrives by entry DMA."""
    me, group, refs = split_refs(refs, n, axis, True, True)
    refs = list(refs)
    x_ref = refs.pop(0)
    ins = {k: refs.pop(0) for k in _IN}
    outs = {k: refs.pop(0) for k in _OUT}
    x_out = refs.pop(0) if emit_x else None
    scr = [refs.pop(0) for _ in range(ar_scratch)]
    dma_in = [k for k in _IN if (k == "fn" and fn_mode != "vmem") or (k != "fn" and in_dma)]
    bufs = {k: (refs.pop(0) if k in dma_in else ins[k]) for k in _IN}
    obufs = {k: (refs.pop(0) if out_dma else outs[k]) for k in _OUT}
    x_buf = refs.pop(0)
    s_buf = refs.pop(0) if pre_sum else None
    in_sem, out_sem = refs.pop(0), refs.pop(0)
    ar_sems = refs

    # Fault controls poison the buffer a missing wait would expose (the
    # previous call's identical data would otherwise hide it, STRAT-MOE-16).
    def poison(ref):
        ref[...] = jnp.full(ref.shape, jnp.nan, ref.dtype)

    if fault == "no_res_wait":
        poison(bufs["res"])
    if fault == "no_newres_wait":
        poison(obufs["new_res"])
    if fault == "drop_round2_wait":
        poison(scr[-1] if form != "pairs" else scr[5])
    if fault == "fn_short":
        assert fn_mode != "vmem", "fn_short needs the fn entry DMA"
        poison(bufs["fn"])

    # 1. The seam operands HBM -> VMEM at entry: they land under the barrier
    #    and the ICI rounds (HBM is idle while the all-reduce waits).
    copies = {}
    for k in dma_in:
        if k == "fn" and not fn_dma:
            continue
        if k == "res" and fault == "no_res_wait":
            continue  # started after the all-reduce, below
        src, dst = ins[k], bufs[k]
        if k == "fn":
            nrows = hc_mult * (hc_mult + 2) if fault == "fn_short" else fn_rows
            src = src.at[pl.ds(0, nrows)]
            dst = dst.at[pl.ds(0, nrows)]
        copies[k] = pltpu.make_async_copy(src, dst, in_sem.at[_IN.index(k)])
        copies[k].start()

    overlap = None
    if pre_sum and body:
        # sum_i comb_ij * res_i needs no x: compute it while the all-reduce
        # waits on ICI. (post_j * x is then added last instead of first: the
        # f32 sum is re-associated, so new_res may differ by a bf16 ulp.)
        def overlap():
            if "res" in copies:
                copies.pop("res").wait()
            if "comb" in copies:
                copies.pop("comb").wait()
            comb = bufs["comb"][...]
            for j in range(hc_mult):
                acc = None
                for i in range(hc_mult):
                    k = i * hc_mult + j
                    t = comb[:, k:k + 1] * bufs["res"][:, i * hidden:(i + 1) * hidden].astype(jnp.float32)
                    acc = t if acc is None else acc + t
                s_buf[:, j * hidden:(j + 1) * hidden] = acc

    # 2. The all-reduce, x into x_buf (VMEM).
    if form == "oneshot":
        late = _oneshot(me, group, x_ref, x_buf, scr, ar_sems, n, overlap)
    elif form == "twoshot":
        late = _twoshot(me, group, x_ref, x_buf, scr, ar_sems, n, hidden, fault, overlap)
    else:
        late = _pairs(me, group, x_ref, x_buf, scr, ar_sems, hidden, chunks, fault, overlap)
    if emit_x:
        x_out[...] = x_buf[...]
    if fault == "no_res_wait":
        copies["res"] = pltpu.make_async_copy(ins["res"], bufs["res"], in_sem.at[0])
        copies["res"].start()

    # 3. The seam body on x_buf.
    for k, c in copies.items():
        if k == "res" and fault == "no_res_wait":
            continue
        c.wait()
    out_copies = {}

    def start_out(k):
        if not out_dma or k in out_copies:
            return
        out_copies[k] = pltpu.make_async_copy(obufs[k], outs[k], out_sem.at[_OUT.index(k)])
        out_copies[k].start()

    if fault == "no_newres_wait":
        start_out("new_res")
    fn_ref = bufs["fn"]
    if fn_mode == "vmem" and fn_ref.shape[0] != fn_rows:
        # A grid-free call takes whole blocks only: read the first fn_rows
        # rows of the whole stacked fn in place.
        fn_ref = fn_ref.at[pl.ds(0, fn_rows)]
    if body:
        _body(x_buf[...], bufs["res"], bufs["post"], bufs["comb"], fn_ref, bufs["sc"],
              bufs["hb"], bufs["nw"], obufs["new_res"], obufs["post_mix"], obufs["comb_mix"],
              obufs["layer"], start_out, hc_mult=hc_mult, hidden_size=hidden,
              body_rows=body_rows, early_out=early_out, s_buf=s_buf, **seam_kw)
    else:  # timing-only skeleton; keep the fn buffer live so its copy is not DCE'd
        obufs["post_mix"][...] = jnp.zeros(obufs["post_mix"].shape, jnp.float32) + bufs["fn"][
            0:1, 0:hc_mult].astype(jnp.float32)
    for k in _OUT:  # the rest (all of them unless early_out; all for the skeleton)
        start_out(k)

    # 4. Drain: the all-reduce's sends, then the output DMAs.
    if fault == "no_res_wait":
        copies["res"].wait()
    if late is not None:
        late()
    _drain(scr, ar_sems, form, n, chunks)
    for c in out_copies.values():
        c.wait()


def _oneshot(me, group, x_ref, x_buf, scr, sems, n, overlap=None):
    (arrivals,) = scr
    send_sem, recv_sem = sems
    barrier(group)
    for j in range(n):

        @pl.when(jnp.int32(j) != me)
        def _(j=j):
            push(x_ref, arrivals.at[me], send_sem, recv_sem, j, group)

    arrivals[pl.ds(me, 1)] = x_ref[...][None]
    if overlap is not None:
        overlap()
    wait_bytes_of(arrivals.at[pl.ds(0, n - 1)], recv_sem)
    x_buf[...] = sum_slots(arrivals, n).astype(x_buf.dtype)
    return None


def _twoshot(me, group, x_ref, x_buf, scr, sems, n, hidden, fault, overlap=None):
    own = hidden // n
    mine, slices, gathered = scr
    send1, recv1, send2, recv2 = sems
    barrier(group)
    for j in range(n):
        mine[j] = x_ref[:, pl.ds(j * own, own)]
    for j in range(n):

        @pl.when(jnp.int32(j) != me)
        def _(j=j):
            push(mine.at[j], slices.at[me], send1, recv1, j, group)

    slices[pl.ds(me, 1)] = mine[pl.ds(me, 1)]
    if overlap is not None:  # x-independent seam work, under the ICI rounds
        overlap()
    wait_bytes_of(slices.at[pl.ds(0, n - 1)], recv1)
    gathered[pl.ds(me, 1)] = sum_slots(slices, n).astype(gathered.dtype)[None]
    for j in range(n):

        @pl.when(jnp.int32(j) != me)
        def _(j=j):
            push(gathered.at[me], gathered.at[me], send2, recv2, j, group)

    def wait2():
        wait_bytes_of(gathered.at[pl.ds(0, n - 1)], recv2)

    if fault != "drop_round2_wait":
        wait2()
    for j in range(n):
        src = j
        if fault == "x_slot" and j < 2:
            src = 1 - j
        x_buf[:, pl.ds(j * own, own)] = gathered[src]
    return wait2 if fault == "drop_round2_wait" else None


def _pairs(me, group, x_ref, x_buf, scr, sems, hidden, chunks, fault, overlap=None):
    half = hidden // 2
    w = half // chunks
    stage, recv1, pair, recv2, final, recv3 = scr
    s1, r1, s2, r2, s3, r3 = sems
    b = lax.rem(me, 2)
    partner = me + 1 - 2 * b
    peer = lax.rem(me + 2, 4)

    def cols(side, c):
        if fault == "x_slot" and chunks > 1:
            c = (c + 1) % chunks if side == 0 else c
        return pl.ds(side * half + c * w, w)

    def src_cols(side, c):
        return pl.ds(side * half + c * w, w)

    barrier(group)

    def on_side(fn):
        @pl.when(b == 0)
        def _():
            fn(0, 1)

        @pl.when(b == 1)
        def _():
            fn(1, 0)

    def stage_other(mine, other):
        for c in range(chunks):
            stage[c] = x_ref[:, src_cols(other, c)]

    on_side(stage_other)
    for c in range(chunks):
        push(stage.at[c], recv1.at[c], s1.at[c], r1.at[c], partner, group)
    if overlap is not None:  # x-independent seam work, under the ICI hops
        overlap()
    for c in range(chunks):
        wait_bytes_of(recv1.at[c], r1.at[c])

        def pair_sum(mine, other, c=c):
            pair[c] = (
                x_ref[:, src_cols(mine, c)].astype(jnp.float32) + recv1[c].astype(jnp.float32)
            ).astype(pair.dtype)

        on_side(pair_sum)
        push(pair.at[c], recv2.at[c], s2.at[c], r2.at[c], peer, group)
    for c in range(chunks):
        wait_bytes_of(recv2.at[c], r2.at[c])
        final[c] = (pair[c].astype(jnp.float32) + recv2[c].astype(jnp.float32)).astype(final.dtype)
        push(final.at[c], recv3.at[c], s3.at[c], r3.at[c], partner, group)

        def write_mine(mine, other, c=c):
            x_buf[:, cols(mine, c)] = final[c]

        on_side(write_mine)
    def wait3():
        for c in range(chunks):
            wait_bytes_of(recv3.at[c], r3.at[c])

    for c in range(chunks):
        if fault != "drop_round2_wait":
            wait_bytes_of(recv3.at[c], r3.at[c])

        def write_other(mine, other, c=c):
            x_buf[:, cols(other, c)] = recv3[c]

        on_side(write_other)
    return wait3 if fault == "drop_round2_wait" else None


def _drain(scr, sems, form, n, chunks):
    if form == "oneshot":
        wait_bytes_of(scr[0].at[pl.ds(0, n - 1)], sems[0])
    elif form == "twoshot":
        wait_bytes_of(scr[1].at[pl.ds(0, n - 1)], sems[0])
        wait_bytes_of(scr[2].at[pl.ds(0, n - 1)], sems[2])
    else:
        stage, _, pair, _, final, _ = scr
        s1, _, s2, _, s3, _ = sems
        for c in range(chunks):
            wait_bytes_of(stage.at[c], s1.at[c])
            wait_bytes_of(pair.at[c], s2.at[c])
            wait_bytes_of(final.at[c], s3.at[c])


def _ar_scratch(form, n, rows, hidden, chunks, dtype):
    if form == "oneshot":
        return [pltpu.VMEM((n, rows, hidden), dtype)], [pltpu.SemaphoreType.DMA] * 2
    if form == "twoshot":
        own = hidden // n
        return [pltpu.VMEM((n, rows, own), dtype)] * 3, [pltpu.SemaphoreType.DMA] * 4
    w = hidden // 2 // chunks
    return ([pltpu.VMEM((chunks, rows, w), dtype)] * 6,
            [pltpu.SemaphoreType.DMA((chunks,))] * 6)


@functools.partial(
    jax.jit,
    static_argnames=(
        "num_cores", "form", "chunks", "rms_eps", "hc_pre_eps", "gates", "norm_eps",
        "fn_chunks", "skip_device_barrier", "gemm_precision", "vmem_limit_bytes",
        "emit_x", "body", "fn_dma", "fault", "collective_id", "axis", "in_mode", "out_mode",
        "early_out", "small_block", "fn_mode", "pre_sum",
    ),
)
def ar_seam(
    x_p, res2d, post2d, comb2d, fn_cat, hc_scale, hc_base, norm_weight, rank, peers, *,
    num_cores, axis, form, chunks, rms_eps, hc_pre_eps, gates, norm_eps, fn_chunks=3,
    skip_device_barrier=False, gemm_precision="highest",
    vmem_limit_bytes=VMEM_LIMIT_BYTES, emit_x=False, body=True, fn_dma=True, fault=None,
    collective_id=ar.COLLECTIVE_ID, in_mode="vmem", out_mode="vmem", early_out=False,
    small_block=utils.SUBLANE, fn_mode="vmem", pre_sum=False,
):
    """Sum x_p over the core's TP group, then run the seam on the sum.

    Args:
      x_p: (T, H) bf16, this core's partial (the sublayer output before its
        TP all-reduce).
      res2d: (T, m * H) bf16; post2d (T, m) f32; comb2d (T, m * m) f32.
      fn_cat: (3 * M3, m * H) bf16, ``utils.CatFn.cat``; the first
        ``fn_chunks * M3`` rows are read.
      hc_scale (3,), hc_base (M3,) f32; norm_weight (1, H) bf16.
      rank: int32 [1]; peers: int32 [num_cores] (``small_collectives`` form).
      form: "oneshot" / "twoshot" / "pairs" or None (by size, as
        ``small_all_reduce``); chunks: the pairs form's column chunks.
      gates: ``utils.GateConsts`` (in-kernel gates; required).
      small_block: the unfused seam's VLLM_TPU_DSV4_MHC_SMALL_BLOCK (its
        block shape at T < 16, which the body reproduces for bit parity).
      in_mode: how res/post/comb/scales/norm weight arrive -- "vmem" (whole
        VMEM blocks; XLA places them), "any" (entry DMA from wherever XLA
        put them), "pin" (entry DMA from HBM-pinned operands). fn always
        arrives by entry DMA from pl.ANY.
      out_mode: "vmem" (whole VMEM output blocks; XLA places them) or "dma"
        (async DMA from VMEM scratch to pl.ANY); early_out starts each
        output DMA as soon as it is final.
      fn_mode: "vmem" (the whole stacked fn as a VMEM block, its first
        fn_rows rows read in place -- as the unfused seam reads it), "pin"
        (entry DMA from an HBM-pinned fn), "any" (entry DMA from wherever
        XLA put it: a VMEM->VMEM copy lowers to TensorCore vld/vst, ~0.5 us
        at entry -- probe 06).
      emit_x, body, fn_dma, fault: test / timing controls (see FAULTS).
    Returns: (new_res (T, m*H) bf16, post_mix (T, m) f32, comb_mix (T, m*m)
      f32, layer (T, H) bf16[, x (T, H) bf16 with ``emit_x``]).
    """
    assert gates is not None and norm_eps is not None
    assert fault is None or fault in FAULTS, fault
    rows, hidden = x_p.shape
    hc_hidden = res2d.shape[1]
    hc_mult = hc_hidden // hidden
    m3 = hc_mult * (hc_mult + 2)
    n = num_cores
    if not ar.supports(rows, hidden, n):
        raise ValueError(f"hidden {hidden} does not split over {n} cores")
    form = _form(rows, hidden, n, form)
    if form == "pairs" and (n != 4 or (hidden // 2) % (chunks * LANES)):
        raise ValueError(f"pairs form: n={n} chunks={chunks}")
    if form != "pairs":
        chunks = 1
    fn_rows = fn_chunks * m3
    hbm = pltpu.HBM
    ins = {
        "res": res2d,
        "post": post2d.astype(jnp.float32),
        "comb": comb2d.astype(jnp.float32),
        "fn": fn_cat,
        "sc": hc_scale.reshape(1, 3).astype(jnp.float32),
        "hb": hc_base.reshape(1, m3).astype(jnp.float32),
        "nw": norm_weight.reshape(1, hidden),
    }
    if in_mode == "pin":
        ins = {k: pltpu.with_memory_space_constraint(v, hbm) for k, v in ins.items()}
    in_dma = in_mode != "vmem"
    out_dma = out_mode == "dma"
    dma_in = [k for k in _IN if (k == "fn" and fn_mode != "vmem") or (k != "fn" and in_dma)]
    if fn_mode == "pin":
        ins["fn"] = pltpu.with_memory_space_constraint(ins["fn"], hbm)
    if rows < utils.SUBLANE and not rows <= small_block < utils.SUBLANE:
        body_rows = utils.SUBLANE
    else:
        body_rows = utils.round_up(rows, utils.SUBLANE) if rows > small_block else rows
    buf_shapes = {
        "res": (rows, hc_hidden), "post": (rows, hc_mult), "comb": (rows, hc_mult * hc_mult),
        "fn": (fn_rows, hc_hidden), "sc": (1, 3), "hb": (1, m3), "nw": (1, hidden),
    }
    out_shapes = {
        "new_res": jax.ShapeDtypeStruct((rows, hc_hidden), jnp.bfloat16),
        "post_mix": jax.ShapeDtypeStruct((rows, hc_mult), jnp.float32),
        "comb_mix": jax.ShapeDtypeStruct((rows, hc_mult * hc_mult), jnp.float32),
        "layer": jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
    }
    ar_scr, ar_sems = _ar_scratch(form, n, rows, hidden, chunks, x_p.dtype)
    scratch = [
        *ar_scr,
        *(pltpu.VMEM(buf_shapes[k], ins[k].dtype) for k in dma_in),
        *(pltpu.VMEM(out_shapes[k].shape, out_shapes[k].dtype) for k in _OUT if out_dma),
        pltpu.VMEM((rows, hidden), x_p.dtype),
        *([pltpu.VMEM((rows, hc_hidden), jnp.float32)] if pre_sum else []),
        pltpu.SemaphoreType.DMA((len(_IN),)),
        pltpu.SemaphoreType.DMA((len(_OUT),)),
        *ar_sems,
    ]
    args, specs = scalar_operands(rank, peers)
    out_shape = [out_shapes[k] for k in _OUT]
    out_specs = [pl.BlockSpec(memory_space=pl.ANY) if out_dma else pl.BlockSpec()] * len(_OUT)
    if emit_x:
        out_shape.append(jax.ShapeDtypeStruct((rows, hidden), x_p.dtype))
        out_specs.append(pl.BlockSpec())
    seam_kw = dict(rms_eps=rms_eps, hc_pre_eps=hc_pre_eps, gates=gates, norm_eps=norm_eps,
                   gemm_precision=gemm_precision)
    params = dict(collective_id=collective_id, vmem_limit_bytes=vmem_limit_bytes)
    if skip_device_barrier:
        params["skip_device_barrier"] = True
    tag = f"{form}_n{n}_r{rows}" + (f"_c{chunks}" if form == "pairs" else "")
    tag += "" if body else "_skel"
    tag += "" if fn_dma else "_nofn"
    tag += f"_F{fault}" if fault else ""
    tag += f"_i{in_mode}_o{out_mode}" + ("e" if early_out else "") + f"_f{fn_mode}"
    tag += "_ps" if pre_sum else ""
    outs = pl.pallas_call(
        functools.partial(
            _kernel, axis=axis, n=n, form=form, chunks=chunks, rows=rows, hidden=hidden, hc_mult=hc_mult,
            fn_rows=fn_rows, emit_x=emit_x, body=body, fn_dma=fn_dma, fault=fault,
            ar_scratch=len(ar_scr), seam_kw=seam_kw, in_dma=in_dma, out_dma=out_dma,
            early_out=early_out, body_rows=body_rows, fn_mode=fn_mode, pre_sum=pre_sum,
        ),
        out_shape=tuple(out_shape),
        in_specs=[*specs, pl.BlockSpec(),
                  *(pl.BlockSpec(memory_space=pl.ANY) if k in dma_in
                    else pl.BlockSpec() for k in _IN)],
        out_specs=tuple(out_specs),
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(**params),
        name=f"kseam_ar_seam_{tag}",
    )(*args, x_p, *(ins[k] for k in _IN))
    return tuple(outs)


def mhc_ar_seam(
    x_p, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
    hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, *, norm_weight, norm_eps,
    num_cores, axis, rank, peers, form=None, chunks=1, fn_chunks=3, skip_device_barrier=False,
    **kw,
):
    """``mhc_fused_post_pre`` (in-kernel gates, fused norm, stacked fn) of
    the TP-group sum of ``x_p``, in the residual's shapes.

    Returns (residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur[, x]).
    """
    outer = residual.shape[:-2]
    hc_mult, hidden = residual.shape[-2:]
    x2d = x_p.reshape(-1, hidden).astype(jnp.bfloat16)
    gates = utils.GateConsts(hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)
    fn_cat = utils.cat_fn3(fn).cat
    outs = ar_seam(
        x2d, residual.reshape(-1, hc_mult * hidden), post_layer_mix.reshape(-1, hc_mult),
        comb_res_mix.reshape(-1, hc_mult * hc_mult), fn_cat, hc_scale, hc_base,
        norm_weight.reshape(1, hidden), rank, peers, num_cores=num_cores, axis=axis, form=form,
        chunks=chunks, rms_eps=rms_eps, hc_pre_eps=hc_pre_eps, gates=gates, norm_eps=norm_eps,
        fn_chunks=fn_chunks, skip_device_barrier=skip_device_barrier, **kw,
    )
    new_res, post_mix, comb_mix, layer = outs[:4]
    res = (
        new_res.reshape(*outer, hc_mult, hidden),
        post_mix.reshape(*outer, hc_mult, 1),
        comb_mix.reshape(*outer, hc_mult, hc_mult),
        layer.reshape(*outer, hidden),
    )
    return res + tuple(outs[4:])
