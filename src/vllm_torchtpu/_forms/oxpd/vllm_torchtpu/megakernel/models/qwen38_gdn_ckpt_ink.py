# SPDX-License-Identifier: Apache-2.0
"""q38 X-GLUE-1-D14 (VLLM_TPU_MK_GDN_CKPT_INK=1): the pool GDN block writes its verify checkpoints into the pool itself.

Without the flag the GDN block returns every verify row's recurrent state as `rec_ckpt` (bf16 `[NGDN, B * vh, KD, VD]`)
and its conv buffer as `conv_ckpt`, and the call wrapper then builds the conv windows in XLA and runs
`qwen38_scatter.scatter_gdn_ckpts_fast`, which reloads each checkpoint, widens the state to f32 and DMAs the 384 state
rows and the 16 conv rows of every non-null write id into its manager block.

With the flag `Qwen38PoolGDNInk` does that write from inside the kernel, row by row as the recurrence finishes them:
the row's bf16 state is widened to f32 and stored, with the scatter's typed view, into an FP8 staging buffer, its conv
window (history rows t+1 .. t+3, 256-lane rows, zero-padded to 16) into a second one, and the four DMAs to
`pool_rows(pool, m, ...)` start when the write id `m = gdn_wb[o, s * T + t]` is not 0 (null ids are skipped, as the
scatter skips them). The bytes are the scatter's, so the pool is bit-identical. The staging ring is the pool gather's
own two FP8 buffers (`pg_raw{b}{j}`, `pg_craw{b}`): the gather has finished with them by the time the recurrence ends,
and each GDN layer drains the ring before its gather reuses them, so the last two checkpoints' waits fall in the next
GDN layer (or the epilogue) and the flag adds no VMEM. A slot's previous DMAs are waited (`ck_pend`, SMEM) before the
slot is refilled. A row whose write id is null (a pad row) skips the f32 widen and the staging stores too (GDN-CK16 stage 0;
its DMAs were already skipped). The block drops the `rec_ckpt` / `conv_ckpt` outputs and their write-back DMAs, and the wrappers drop
the conv-window prep and the scatter call (`ink_outs` / `ink_xs` / `ink_extra_outs`).

Pool contract (FUSED's raw-ref contract, experimental/fused_pool_io.py): on TPU the DMAs write the `gdn_pool` input ref
(declared read-only to XLA, no pallas aliasing), and the outer jit's donated pool carries the bytes; correctness needs
the donation honoured with no full-pool copy (checked by the deviceless AOT). CPU interpret mode does not write inputs
back, so there the block copies the pool into an extra output `gdn_pool_out` at kernel start and writes that one; the
wrappers then use it as the pool. Reads stay on `gdn_pool`: a GDN layer reads its state blocks before it writes, no two
live requests share a block, and null ids are never written, so no read in the step sees a checkpoint write.

Limits (refused at build): verify only (T > 1), bf16 recurrent state (VLLM_TPU_MK_GDN_REC_F32 unset), no FUSED / GPR.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.gdn.recurrence import gdn_rec, gdn_rec_chunked
from ..blocks.gdn.states import rec_chunk, rec_in
from ..exec_spec import ExecSpec
from ..ops import BF16, F32, FP8, I32, REC, rnd, rup
from ..pool.config import cache_exec
from ..spec import ExtraInput, StateDecl
from .qwen38 import Qwen38GDN
from .qwen38_gdn_pool import CONV_ROWS, POOL, SSM_BLOCKS, SSM_ROWS, Qwen38PoolGDN, pool_rows

ENV = "VLLM_TPU_MK_GDN_CKPT_INK"
WRITE_BLOCKS = "gdn_wb"  # SMEM int32 [NGDN, B]: write block of (layer, request s, token t) at column s * T + t
POOL_OUT = "gdn_pool_out"  # CPU interpret only: the pool with the checkpoints written
DEPTH = 2  # staging slots: the gather's two FP8 staging buffers
ENGAGED = "q38 X-GLUE-1 GDN checkpoints written in-kernel"
_LOGGED = False


def ckpt_ink_on() -> bool:
    """VLLM_TPU_MK_GDN_CKPT_INK (registered in envs, hashed: it changes the traced program)."""
    from vllm_torchtpu import envs

    return bool(envs.VLLM_TPU_MK_GDN_CKPT_INK)


def pool_gdn_cls() -> type:
    """The pool GDN block class of `segment_spec` / `wholestep_spec`: `Qwen38PoolGDN` unless the flag is on. With the
    flag, one engagement line per process (STEP cars grep it; vLLM's handlers via the "vllm." prefix)."""
    global _LOGGED
    if not ckpt_ink_on():
        return Qwen38PoolGDN
    if not _LOGGED:
        _LOGGED = True
        logging.getLogger("vllm." + __name__).info(
            "%s: %s=1 (rec_ckpt / conv_ckpt outputs, conv-window prep and scatter_gdn_ckpts_fast dropped; staging "
            "depth %d on the gather's buffers)", ENGAGED, ENV, DEPTH)
    return Qwen38PoolGDNInk


def _store_typed(ref: Any, vals: Any, interpret: bool) -> None:
    """f32 `vals` (row-major) -> a whole FP8 byte ref `[rows, 1, 4, 256]` (qwen38_scatter._store_typed)."""
    if interpret:
        words = vals.reshape(-1, ref.shape[-1])
        ref[...] = pltpu.bitcast(words, ref.dtype).reshape(ref.shape)
        return
    view = ref.bitcast(jnp.float32)
    view.reshape(-1, vals.shape[-1])[...] = vals


def _dst(ctx: Any) -> Any:
    return ctx.O[POOL_OUT] if bool(ctx.ex.interpret) else ctx.extra[POOL]


def _copies(ctx: Any, d: int, m: Any) -> list[Any]:
    """Staging slot `d` -> manager block `m`: the 3 state chunks (rows 0 .. 383) and the 16 conv rows (384 .. 399)."""
    scr, dst = ctx.scr, _dst(ctx)
    cps = [
        pltpu.make_async_copy(
            scr[f"pg_raw{d}{j}"], pool_rows(dst, m, j * SSM_ROWS, SSM_ROWS), scr["ck_sem"].at[d, j]
        )
        for j in range(SSM_BLOCKS)
    ]
    cps.append(
        pltpu.make_async_copy(
            scr[f"pg_craw{d}"], pool_rows(dst, m, SSM_BLOCKS * SSM_ROWS, CONV_ROWS), scr["ck_sem"].at[d, SSM_BLOCKS]
        )
    )
    return cps


def _wait_slot(ctx: Any, d: int) -> None:
    @pl.when(ctx.scr["ck_pend"][d] != 0)
    def _():
        for c in _copies(ctx, d, 0):
            c.wait()


def drain(ctx: Any) -> None:
    """Wait every staging slot's in-flight checkpoint DMAs and mark the slots free."""
    for d in range(DEPTH):
        _wait_slot(ctx, d)
        ctx.scr["ck_pend"][d] = jnp.int32(0)


def write_ckpt(ctx: Any, gp: Any, o: Any, r: Any, d: int, state: Any, s: Any, t: int) -> None:
    """Checkpoint of GDN layer `o`, row `r = s * T + t`: `state` (the row's `[vh, KD, VD]` recb ref slice), widened to
    f32, and its conv window -> staging slot `d` -> the pool block `gdn_wb[o, r]` (skipped when 0)."""
    scr, interp = ctx.scr, bool(ctx.ex.interpret)
    vh, KD, VD, QKV = gp.vh, gp.KD, gp.VD, gp.QKV
    _wait_slot(ctx, d)  # unguarded: ck_pend[d] is overwritten below for null ids too, so the slot's old DMAs must be waited
    m = ctx.extra[WRITE_BLOCKS][o, r]

    @pl.when(m != 0)  # q38 GDN-CK16 stage 0: a null write id (pad row) skips the f32 widen and the staging stores
    def _():
        vals = state[...].astype(F32).reshape(vh * KD, VD)
        rows = vals.shape[0] // SSM_BLOCKS
        for j in range(SSM_BLOCKS):
            _store_typed(scr[f"pg_raw{d}{j}"], vals[j * rows : (j + 1) * rows], interp)
        cv = scr["convb"][s]  # [HR, QKV] f32: rows 0..2 history, 3.. this step's tokens
        per = QKV // 256
        crow = jnp.concatenate(
            [cv[t + 1 + k : t + 2 + k, c * 256 : (c + 1) * 256] for k in range(3) for c in range(per)]
            + [jnp.zeros((CONV_ROWS - 3 * per, 256), F32)],
            axis=0,
        )
        _store_typed(scr[f"pg_craw{d}"], crow, interp)
        for c in _copies(ctx, d, m):
            c.start()

    scr["ck_pend"][d] = m


def gdn_in_ink(ctx: Any, gp: Any, o: Any, slot: Any) -> Any:
    """`blocks/gdn/conv.gdn_in` (verify, no GPR) without the waits for the previous layer's `conv_ckpt` / `rec_ckpt`
    write-backs: with the flag there are none (the previous layer staged its checkpoints synchronously)."""
    scr, ST, skip, sc, stream = ctx.scr, ctx.ST, ctx.skip, ctx.sc, ctx.stream
    NLI, GCH, NRB = gp.NLI, gp.GCH, gp.NRB
    convb, sem = scr["convb"], scr["st_sem"]
    rec_in_recb = bool(ST.get("rec_in_recb", False))
    with sc("gdn_in"):
        if "gdn_state" not in skip:
            pltpu.make_async_copy(ST["conv"].at[o], convb, sem.at[0]).start()
            if GCH:
                for s_ in range(NRB):
                    rec_chunk(ctx, gp, o, s_, s_, False).start()
            elif not rec_in_recb:
                for d in rec_in(ctx, gp, o):
                    d.start()
        y = stream.dmatmul("lin_in", scr["xb"], NLI)
        ybuf = scr["ybuf"]
        ybuf[:, :NLI] = rnd(y * ctx.sm(slot, "in_s", NLI))
    return ybuf


def conv_step_ink(ctx: Any, gp: Any, o: Any, slot: Any, ybuf: Any) -> Any:
    """`blocks/gdn/conv.conv_step` (verify, no GPR; the same conv arithmetic) without the `conv_ckpt` write-back: the
    conv windows are staged with each row's checkpoint (`write_ckpt`)."""
    ex, scr, ST = ctx.ex, ctx.scr, ctx.ST
    skip, sc, stream = ctx.skip, ctx.sc, ctx.stream
    B, T, NS = ex.B, ex.T, ex.NS
    QKV, GCH = gp.QKV, gp.GCH
    convb, sem = scr["convb"], scr["st_sem"]
    with sc("gdn_conv"):
        if "gdn_state" not in skip:
            pltpu.make_async_copy(ST["conv"].at[o], convb, sem.at[0]).wait()
        ow, _ = stream.sp.lay["conv"]
        taps = [scr["sm"][slot, :, ow + t * QKV : ow + (t + 1) * QKV] for t in range(4)]
        u = None
        us = []
        if gp.vec_rec and GCH and NS >= 2 and NS % 2 == 0 and (2 * T) % 8 == 0:
            sc_chunk = 4 if NS % 4 == 0 else 2
            t0, t1, t2, t3 = [t[None, :, :] for t in taps]
            for s0 in range(0, NS, sc_chunk):
                x_seq = ybuf[s0 * T : (s0 + sc_chunk) * T, 0:QKV].reshape(
                    sc_chunk, T, QKV
                )
                cb = convb[s0 : s0 + sc_chunk, :, :]
                seq_buf = jnp.concatenate([cb[:, 0:3, :], x_seq], axis=1)
                if gp.HR > 3 + T:
                    cb_new = jnp.concatenate(
                        [seq_buf, cb[:, 3 + T :, :]], axis=1
                    )
                else:
                    cb_new = seq_buf
                convb[s0 : s0 + sc_chunk, :, :] = cb_new
                u_chunk = rnd(
                    jax.nn.silu(
                        seq_buf[:, 0:T, :] * t0
                        + seq_buf[:, 1 : 1 + T, :] * t1
                        + seq_buf[:, 2 : 2 + T, :] * t2
                        + x_seq * t3
                    )
                ).reshape(sc_chunk * T, QKV)
                us.append(u_chunk)
        else:
            for s in range(NS):
                convb[s, 3 : 3 + T, :] = ybuf[s * T : (s + 1) * T, 0:QKV]
                w = [convb[s, k : k + T, :] for k in range(4)]
                u_ = rnd(
                    jax.nn.silu(
                        w[0] * taps[0]
                        + w[1] * taps[1]
                        + w[2] * taps[2]
                        + w[3] * taps[3]
                    )
                )
                if GCH:
                    us.append(u_)
                else:
                    scr["ub"][s, 0:T, :] = u_
        if GCH:
            u = jnp.concatenate(us, axis=0)
            b_pad = rup(B, 8)
            u_pad = jnp.pad(u, ((0, b_pad - B), (0, 0))) if b_pad > B else u
            ybuf[0:b_pad, 0:QKV] = u_pad
    return u


class _ChunkOut:
    """`rec_chunk(..., out=True)` of the chunked (`GCH`) path: `start` writes row chunk `j`'s checkpoints from ring slot
    `s` (each row's state is in the slot's row group `q * vh`), `wait` has nothing to wait for (the slot is free once
    the rows are staged)."""

    def __init__(self, ctx: Any, gp: Any, o: Any, j: Any, s: Any) -> None:
        self.ctx, self.gp, self.o, self.j, self.s = ctx, gp, o, j, s
        tail = bool(gp.TR) and isinstance(j, int) and j == gp.NF
        self.n = gp.TR if tail else gp.CB

    def start(self) -> None:
        ctx, gp, j = self.ctx, self.gp, self.j
        T, vh = ctx.ex.T, gp.vh
        recb = ctx.scr["recb"]
        with ctx.sc("gdn_ckpt_write"):
            for q in range(self.n):
                r = j * gp.CB + q
                s = j * (gp.CB // T) + q // T
                write_ckpt(ctx, gp, self.o, r, q % DEPTH, recb.at[self.s, pl.ds(q * vh, vh)], s, q % T)

    def wait(self) -> None:
        return None


class _InkBlockEmit(Qwen38GDN):
    """`GDNBlock.emit` with the checkpoints written in-kernel (sits between `Qwen38PoolGDN` and `Qwen38GDN` in the MRO,
    so `Qwen38PoolGDN.emit`'s gather and X-K-LIVE dispatch run unchanged before it)."""

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex = ctx.ex
        gp = ctx.block_params(self.name)
        o = ctx.stream.sp.block_ord(l, self.name)
        ctx = _replace_st(ctx, rec_chunk_out=_ChunkOut)
        scr, skip, sc, stream = ctx.scr, ctx.skip, ctx.sc, ctx.stream
        Z, H, GCH = gp.Z, gp.H, gp.GCH
        with stream.segment(lazy=False):
            ybuf = gdn_in_ink(ctx, gp, o, slot)
            u = conv_step_ink(ctx, gp, o, slot, ybuf)
            if GCH:
                with sc("gdn_rec"):
                    gdn_rec_chunked(ctx, gp, o, slot)
            else:
                if "gdn_state" not in skip and not ctx.ST.get("rec_in_recb", False):
                    with sc("gdn_rec_wait"):
                        for d in rec_in(ctx, gp, o):
                            d.wait()
                if "gdn" not in skip:
                    with sc("gdn_rec"):
                        gdn_rec(ctx, gp, u, ybuf, slot)
            with sc("gdn_out"):
                if not GCH and "gdn_state" not in skip:
                    with sc("gdn_ckpt_write"):
                        vh = gp.vh
                        for r in range(ex.B):
                            write_ckpt(ctx, gp, o, r, r % DEPTH, scr["recb"].at[pl.ds(r * vh, vh)], r // ex.T,
                                       r % ex.T)
                scr["xb2"][:, :Z] = scr["mix"][:, :Z].astype(BF16)
                yo = stream.dmatmul("lin_out", scr["xb2"], H)
                return yo * ctx.sm(slot, "out_s")


def _replace_st(ctx: Any, **kw: Any) -> Any:
    import dataclasses

    return dataclasses.replace(ctx, ST={**ctx.ST, **kw})


class Qwen38PoolGDNInk(Qwen38PoolGDN, _InkBlockEmit):
    """`Qwen38PoolGDN` that writes its verify checkpoints into the pool in-kernel (module docstring)."""

    # Per-rank pool shape of the CPU interpret output (`ink_extra_outs` sets it from the call's layout).
    interpret_pool_shape: tuple[int, ...] | None = None

    def _check(self, ex: ExecSpec) -> None:
        if not ex.VER:
            raise NotImplementedError(f"{ENV}: verify steps only (T > 1), got T={ex.T}")
        if REC != BF16:
            raise NotImplementedError(f"{ENV}: needs the bf16 recurrent state (VLLM_TPU_MK_GDN_REC_F32 unset)")
        if cache_exec(ex).FUSED or cache_exec(ex).GPR:
            raise NotImplementedError(f"{ENV}: not with the FUSED / GPR cache paths")
        assert ex.B % DEPTH == 0, (ex.B, DEPTH)

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        self._check(ex)
        return super().extra_inputs(g, ex) + [ExtraInput(WRITE_BLOCKS, "smem", (g.NGDN, ex.B), I32)]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        self._check(ex)
        if not ex.interpret:
            return []
        shape = self.interpret_pool_shape
        assert shape is not None, f"{ENV} in CPU interpret mode: set the block's interpret_pool_shape (ink_extra_outs)"
        return [StateDecl(POOL_OUT, shape=tuple(shape), dtype=FP8, access="w", per_layer=False)]

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        scr = dict(super().scratch(g, ex, plan))
        gp = plan.block_params[self.name]
        assert not gp.GCH or gp.CB % DEPTH == 0, (gp.CB, DEPTH)
        scr.update(
            ck_sem=pltpu.SemaphoreType.DMA((DEPTH, SSM_BLOCKS + 1)),
            ck_pend=pltpu.SMEM((DEPTH,), I32),
        )
        return scr

    def init_buffers(self, ctx: Any) -> None:
        super().init_buffers(ctx)
        for d in range(DEPTH):
            ctx.scr["ck_pend"][d] = jnp.int32(0)
        if bool(ctx.ex.interpret):
            cp = pltpu.make_async_copy(ctx.extra[POOL], ctx.O[POOL_OUT], ctx.scr["ck_sem"].at[0, 0])
            cp.start()
            cp.wait()

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        if "gdn_state" not in ctx.skip:
            with ctx.sc("gdn_ckpt_drain"):
                drain(ctx)  # the gather below reuses the staging buffers
        return super().emit(ctx, l, slot)

    def epilogue(self, ctx: Any) -> None:
        gp = ctx.block_params(self.name)
        if gp.NGDN > 0 and "gdn_state" not in ctx.skip:
            drain(ctx)

    def host_update_states(self, g, ex, pos, states, raw_outputs):
        return {POOL_OUT: raw_outputs[POOL_OUT]} if ex.interpret else {}


# ---------------------------------------------------------------- call wrappers (qwen38_wholestep_call / segment_call)


def ink_blocks(spec: Any) -> list[Qwen38PoolGDNInk]:
    """The spec's in-kernel-checkpoint GDN blocks (inside their HC wrappers)."""
    if spec is None:
        return []
    out = []
    for b in spec.all_blocks():
        inner = getattr(b, "inner", b)
        if isinstance(inner, Qwen38PoolGDNInk):
            out.append(inner)
    return out


def ink_extra_outs(spec: Any, interpret: Any, layout: Any) -> tuple:
    """Extra kernel outputs the wrapper asks for: `(POOL_OUT,)` in CPU interpret mode when the spec writes checkpoints
    in-kernel (the blocks get the per-rank pool shape of `layout`), else ()."""
    blks = ink_blocks(spec)
    if not blks or not interpret:
        return ()
    for b in blks:
        b.interpret_pool_shape = tuple(int(x) for x in layout.pool_shape)
    return (POOL_OUT,)


def ink_xs(fn: Any, wb: Any) -> dict:
    """The kernel's write-block input from the call's `wb [NL, NS, T]` when it takes one, else {}."""
    if WRITE_BLOCKS not in getattr(fn, "extra_names", ()):
        return {}
    NL, NS, T = wb.shape
    return {WRITE_BLOCKS: wb.reshape(NL, NS * T).astype(jnp.int32)}


def ink_outs(fn: Any, outs: Any, pool: Any, interpret: Any) -> Any:
    """`outs` itself unless the kernel writes checkpoints in-kernel; then `(res, P, P, *rest)` in the checkpoint slots'
    places, `P` = the pool to write on (the kernel's `POOL_OUT`, last, in CPU interpret mode; else the pool itself),
    which the wrappers' scatter lines recognise by its pool shape and dtype and pass through."""
    if WRITE_BLOCKS not in getattr(fn, "extra_names", ()):
        return outs
    outs = tuple(outs)
    if interpret:
        src, outs = outs[-1], outs[:-1]
    else:
        src = pool
    return (outs[0], src, src) + outs[1:]


__all__ = [
    "DEPTH", "ENGAGED", "ENV", "POOL_OUT", "WRITE_BLOCKS", "Qwen38PoolGDNInk", "ckpt_ink_on", "drain", "ink_blocks",
    "ink_extra_outs", "ink_outs", "ink_xs", "pool_gdn_cls", "write_ckpt",
]
