# SPDX-License-Identifier: Apache-2.0
"""q38 HIX700 GDN-SIDECAR (VLLM_TPU_MK_GDN_SIDECAR=1): keep the whole step's GDN verify checkpoints in a slot-indexed sidecar between
steps and read the next step's state from it, instead of round-tripping every checkpoint through the unified pool.

Today (flag off) the pool GDN block gathers each request's state from pool block `rb = ck[off[st]]` (3 x 384 f32 FP8-typed rows + 16 conv rows,
cast to bf16 `REC`), writes every verify row's state as `rec_ckpt` (bf16) and its conv buffer as `conv_ckpt` (f32), and the call wrapper then
runs `scatter_gdn_ckpts_fast`, which reloads every checkpoint, widens it to f32 and DMAs it into its pool block: 130.5 us + 15 us of prep per
step at 16 rows (T90o / T62z, M), for a state the next step reads exactly one of.

With the flag:
  * the block takes two HBM inputs, `gdn_side_rec` (bf16 `[NGDN, B * vh, KD, VD]`, the rec_ckpt layout) and `gdn_side_conv` (f32
    `[NGDN, NS, rup(3 + T, 8), QKV]`, the conv_ckpt layout), and two SMEM inputs, `gdn_side_on` (int32 `[1]`) and `gdn_side_off` (int32
    `[NGDN, NS]`: the accepted checkpoint index `read_offsets[state id]` of each layer and request);
  * read: with `gdn_side_on[0] != 0` each request's state comes from `gdn_side_rec[o, (s * T + off) * vh : + vh]` by one DMA straight into
    `pg_rec` (bf16 = REC: no conversion), and its conv history from `gdn_side_conv[o, s]` rows off + 1 .. off + 3 (the scatter's conv window
    rule, `_conv_windows`); with 0 the pool gather runs as today (admission, after a flush);
  * write: the rec_ckpt / conv_ckpt write-backs go to the two side inputs (FUSED's raw-ref contract on TPU: the DMAs write the input refs,
    declared read-only to XLA with no pallas aliasing; the outer jit donates them and returns them, as it does the pool), so the kernel
    writes exactly today's bytes to a different place; in CPU interpret mode (which does not write inputs back) the block copies the two
    side inputs into extra outputs `gdn_side_rec_out` / `gdn_side_conv_out` at kernel start and reads and writes those;
  * the wrapper skips the checkpoint scatter on every call (the pool passes through); the runtime flushes the sidecar into the pool (the
    flag-off scatter bytes) whenever anything else could read the pool's checkpoint blocks (`megakernel_runtime/qwen38_gdn_sidecar_rt.py`).
A layer reads its own sidecar plane before it writes it (the gather is waited before the recurrence) and no other layer touches that plane,
so the in-place write never overwrites a state before it is read. Values: the pool path casts the f32-widened bf16 state back to bf16, which
is exact, so the side path's bf16 read is the same value (bitwise); conv rows are the same f32 bytes.
Limits (refused at build): verify only (T > 1), bf16 recurrent state, no FUSED / GPR, no chunk gather, no X-GLUE-1 / G1, no X-K-LIVE
mode 2 (v2 supports mode 1: a pad request's side copies are skipped under the pool path's live predicate, and its rows zeroed).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..exec_spec import ExecSpec
from ..ops import BF16, F32, I32, REC, rup
from ..pool.config import cache_exec
from ..spec import ExtraInput, StateDecl
from ..live_rows import req_live
from .qwen38_gdn_pool import HAS_STATE, Qwen38PoolGDN, _chunk_gather

ENV = "VLLM_TPU_MK_GDN_SIDECAR"
SIDE_REC = "gdn_side_rec"
SIDE_CONV = "gdn_side_conv"
SIDE_ON = "gdn_side_on"
SIDE_OFF = "gdn_side_off"
SIDE_REC_OUT = "gdn_side_rec_out"    # CPU interpret only
SIDE_CONV_OUT = "gdn_side_conv_out"  # CPU interpret only
ENGAGED = "q38 GDN-SIDECAR whole step"
_LOGGED = False


def sidecar_on() -> bool:
    """VLLM_TPU_MK_GDN_SIDECAR (registered in envs, hashed: it changes the traced program)."""
    from vllm_torchtpu import envs

    return bool(getattr(envs, "VLLM_TPU_MK_GDN_SIDECAR", False))


SIDE_ROWS = 16  # v1: the sidecar serves the 16-row bucket only (every HIX decode step at 1-4 live requests, M: T62z whole_b16)
_CTX_ROWS = [None]  # the bucket rows of the whole-step spec being built (set by `building_rows` around wholestep_spec)


def building_rows(B: int, fn: Any) -> Any:
    """`fn` (wholestep_spec) called with the bucket rows B in context, so `sidecar_gdn_cls` can gate on the bucket (the spec itself takes
    the rank count, not the rows)."""
    def call(*a: Any, **kw: Any) -> Any:
        prev = _CTX_ROWS[0]
        _CTX_ROWS[0] = int(B)
        try:
            return fn(*a, **kw)
        finally:
            _CTX_ROWS[0] = prev

    return call


def sidecar_gdn_cls(base: type, _ranks: int = 0) -> type:
    """The whole step's pool GDN block class: `base` (pool_gdn_cls()) unless the flag is on and the spec is built for a SIDE_ROWS bucket
    (`building_rows`); then `Qwen38SidecarGDN` (refused with the in-kernel checkpoint write, X-GLUE-1, whose block `base` then is). Segment
    specs and other buckets keep `base`."""
    global _LOGGED
    if not sidecar_on() or _CTX_ROWS[0] != SIDE_ROWS:
        return base
    if base is not Qwen38PoolGDN:
        raise ValueError(f"{ENV}=1 is not composed with VLLM_TPU_MK_GDN_CKPT_INK (both answer the checkpoint bytes); unset one")
    if not _LOGGED:
        _LOGGED = True
        logging.getLogger("vllm." + __name__).info(
            "%s: %s=1 (checkpoints kept in the slot-indexed sidecar; the next step reads sidecar[slot, off]; the scatter runs only "
            "when the runtime flushes)", ENGAGED, ENV)
    return Qwen38SidecarGDN


def _redirect(ctx: Any) -> Any:
    """`ctx` with the block's checkpoint outputs pointed at the sidecar (the interpret outputs in CPU interpret mode)."""
    if bool(ctx.ex.interpret):
        rec, conv = ctx.O[SIDE_REC_OUT], ctx.O[SIDE_CONV_OUT]
    else:
        rec, conv = ctx.extra[SIDE_REC], ctx.extra[SIDE_CONV]
    return dataclasses.replace(ctx, O={**ctx.O, "rec_ckpt": rec, "conv_ckpt": conv})


def _src(ctx: Any) -> tuple[Any, Any]:
    """The refs the side gather reads (the inputs on TPU, written in place; the interpret outputs on CPU)."""
    if bool(ctx.ex.interpret):
        return ctx.O[SIDE_REC_OUT], ctx.O[SIDE_CONV_OUT]
    return ctx.extra[SIDE_REC], ctx.extra[SIDE_CONV]


class Qwen38SidecarGDN(Qwen38PoolGDN):
    """`Qwen38PoolGDN` that keeps its verify checkpoints in the slot-indexed sidecar (module docstring)."""

    # Per-rank shapes of the CPU interpret outputs (`side_extra_outs` sets them from the call's geometry).
    interpret_shapes: tuple | None = None

    def _check(self, g: Any, ex: ExecSpec) -> None:
        if not ex.VER:
            raise NotImplementedError(f"{ENV}: verify steps only (T > 1), got T={ex.T}")
        if REC != BF16:
            raise NotImplementedError(f"{ENV}: needs the bf16 recurrent state (VLLM_TPU_MK_GDN_REC_F32 unset)")
        if cache_exec(ex).FUSED or cache_exec(ex).GPR:
            raise NotImplementedError(f"{ENV}: not with the FUSED / GPR cache paths")
        if int(getattr(ex, "live_rows", 0)) >= 2:  # v2: X-K-LIVE mode 1 is supported; mode 2 never builds at 16 rows (_live_rows_on)
            raise NotImplementedError(f"{ENV}: not with X-K-LIVE mode 2 (VLLM_TPU_MK_LIVE_GDN_COMPUTE)")

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        self._check(g, ex)
        return super().extra_inputs(g, ex) + [
            ExtraInput(SIDE_REC, "hbm"),
            ExtraInput(SIDE_CONV, "hbm"),
            ExtraInput(SIDE_ON, "smem", (1,), I32),
            ExtraInput(SIDE_OFF, "smem", (g.NGDN, ex.NS), I32),
        ]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        self._check(g, ex)
        if not ex.interpret:
            return []
        assert self.interpret_shapes is not None, f"{ENV} in CPU interpret mode: set interpret_shapes (side_extra_outs)"
        rec_shape, conv_shape = self.interpret_shapes
        return [
            StateDecl(SIDE_REC_OUT, shape=tuple(rec_shape), dtype=REC, access="w", per_layer=False),
            StateDecl(SIDE_CONV_OUT, shape=tuple(conv_shape), dtype=F32, access="w", per_layer=False),
        ]

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        scr = dict(super().scratch(g, ex, plan))
        gp = plan.block_params[self.name]
        if _chunk_gather(gp, ex) or "pg_rec" not in scr:
            raise NotImplementedError(f"{ENV}: needs the pg_rec staging path (no chunk gather, no direct recb gather)")
        lay = self.lin_in_layout(g, ex)
        scr.update(
            sd_conv=pltpu.VMEM((ex.NS, rup(3 + ex.T, 8), lay.QKV), F32),
            sd_sem=pltpu.SemaphoreType.DMA((2, ex.NS)),
        )
        return scr

    def init_buffers(self, ctx: Any) -> None:
        super().init_buffers(ctx)
        if bool(ctx.ex.interpret):
            for src, dst in ((SIDE_REC, SIDE_REC_OUT), (SIDE_CONV, SIDE_CONV_OUT)):
                cp = pltpu.make_async_copy(ctx.extra[src], ctx.O[dst], ctx.scr["sd_sem"].at[0, 0])
                cp.start()
                cp.wait()

    def _side_copies(self, ctx: Any, o: Any) -> list[Any]:
        ex, scr = ctx.ex, ctx.scr
        lay = self.lin_in_layout(ctx.g, ex)
        vh, T, NS = lay.vh, ex.T, ex.NS
        rec, conv = _src(ctx)
        cps = []
        for s in range(NS):
            off = ctx.extra[SIDE_OFF][o, s]
            r0 = (s * T + off) * vh
            cps.append((pltpu.make_async_copy(rec.at[o, pl.ds(r0, vh)], scr["pg_rec"].at[pl.ds(s * vh, vh)], scr["sd_sem"].at[0, s]),
                        pltpu.make_async_copy(conv.at[o, s], scr["sd_conv"].at[s], scr["sd_sem"].at[1, s])))
        return cps

    @staticmethod
    def _live_gate(ctx: Any, s: int, fn: Any) -> None:
        """`fn()` for request slot `s`; with X-K-LIVE (mode 1) only when the request is live (the pool path's `_gather_live` rule: a pad
        request's copies are neither started nor waited)."""
        if not ctx.ex.live_rows:
            fn()
            return
        pl.when(req_live(ctx, s))(fn)

    def _side_gather(self, ctx: Any, o: Any) -> None:
        """Layer `o`'s states from the sidecar: the accepted checkpoint's rows into `pg_rec` and its conv window into `pg_conv` rows 0..2."""
        ex, scr = ctx.ex, ctx.scr
        T, NS = ex.T, ex.NS
        vh = self.lin_in_layout(ctx.g, ex).vh
        cps = self._side_copies(ctx, o)

        def start(s: int) -> None:
            for c in cps[s]:
                c.start()

        def wait(s: int) -> None:
            for c in cps[s]:
                c.wait()

        if _kern_prefetch():  # q38 HIX-KERN K1: only the call's first GDN ordinal starts its own copies; ordinal o was started by o - 1
            @pl.when(o == 0)
            def _():
                for s in range(NS):
                    self._live_gate(ctx, s, lambda s=s: start(s))
        else:
            for s in range(NS):
                self._live_gate(ctx, s, lambda s=s: start(s))
        for s in range(NS):
            self._live_gate(ctx, s, lambda s=s: wait(s))
        scr["pg_conv"][...] = jnp.zeros(scr["pg_conv"].shape, F32)
        for s in range(NS):
            off = ctx.extra[SIDE_OFF][o, s]
            # The scatter's conv window of checkpoint t is the conv buffer's rows t + 1 .. t + 3 (`_conv_windows`); off < T, static select.
            win = scr["sd_conv"][s, 1:4, :]
            for t in range(1, T):
                win = jnp.where(off == t, scr["sd_conv"][s, t + 1 : t + 4, :], win)
            # The pool path's mask (`keep = HAS_STATE != 0`): a request without a state (the pad request at 3 live) gets zero state and
            # conv rows, not the stale plane its slot left in the sidecar. Written only where needed (no read of the kept rows).
            keep = ctx.extra[HAS_STATE][o, s] != 0
            if ex.live_rows:  # X-K-LIVE: a pad request's slot was not copied (its sd_conv / pg_rec rows are stale): zero, as `_gather_live`
                keep = jnp.logical_and(keep, req_live(ctx, s))

            @pl.when(keep)
            def _(s=s, win=win):
                scr["pg_conv"][s, 0:3, :] = win

            @pl.when(jnp.logical_not(keep))
            def _(s=s):
                scr["pg_rec"][s * vh : (s + 1) * vh] = jnp.zeros((vh,) + tuple(scr["pg_rec"].shape[1:]), scr["pg_rec"].dtype)

    def _gather(self, ctx: Any, o: Any, direct: bool = False, ssm: bool = True) -> None:
        assert not direct and ssm, (direct, ssm)
        on = ctx.extra[SIDE_ON][0] != 0

        @pl.when(on)
        def _():
            self._side_gather(ctx, o)

        @pl.when(jnp.logical_not(on))
        def _():
            Qwen38PoolGDN._gather(self, ctx, o, direct, ssm)

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        out = super().emit(_redirect(ctx), l, slot)
        if _kern_prefetch():
            self._kern_prefetch_next(ctx, l)
        return out

    def epilogue(self, ctx: Any) -> None:
        super().epilogue(_redirect(ctx))

    def host_update_states(self, g, ex, pos, states, raw_outputs):
        if ex.interpret:
            return {SIDE_REC_OUT: raw_outputs[SIDE_REC_OUT], SIDE_CONV_OUT: raw_outputs[SIDE_CONV_OUT]}
        return {}


def side_shapes(NGDN: int, B: int, NS: int, T: int, vh: int, KD: int, VD: int, QKV: int) -> tuple:
    """Per-rank shapes of the two sidecar buffers (= the block's rec_ckpt / conv_ckpt stacked over layers)."""
    return (NGDN, B * vh, KD, VD), (NGDN, NS, rup(3 + T, 8), QKV)


__all__ = ["ENV", "ENGAGED", "building_rows", "SIDE_ROWS", "SIDE_CONV", "SIDE_CONV_OUT", "SIDE_OFF", "SIDE_ON", "SIDE_REC", "SIDE_REC_OUT", "Qwen38SidecarGDN",
           "side_shapes", "sidecar_gdn_cls", "sidecar_on"]


# q38 HIX700 GDN-SIDECAR engine-start refusals (lead ruling 2026-11-24: hard-fail, outside Qwen38SegmentDriver._attach_whole's catch-all)
class SidecarConfigError(RuntimeError):
    """A GDN-SIDECAR composition the engine refuses at start. Raised before _attach_whole's `except Exception` (which would log 'setup
    failed; segment program only' and fall back silently), so engine setup ends in this error."""


def refuse_unsupported() -> None:
    """Engine start (Qwen38SegmentDriver._attach_whole, before its try): with the flag on, refuse VLLM_TPU_MK_GDN_COMMIT1. G1's stash
    (qwen38_gdn_commit.stash) pops the whole-step call's LAST 4 outputs, and the sidecar appends its 3 outputs last."""
    if not sidecar_on():
        return
    from vllm_torchtpu import envs

    if bool(getattr(envs, "VLLM_TPU_MK_GDN_COMMIT1", False)):
        raise SidecarConfigError(f"{ENV}=1 is not composed with VLLM_TPU_MK_GDN_COMMIT1 (G1's stash pops the whole-step call's last 4 "
                                 "outputs; the sidecar appends its 3 last); unset one")


# q38 HIX-KERN K1 begin (VLLM_TPU_MK_KERN_SIDE_PREFETCH, default off; appended, so no line above moves except the two marked edits).
# The side gather (`_side_gather`) starts and waits all of layer o's 2 x NS sidecar copies at the start of layer o, so one DMA latency plus
# the transfer is exposed per GDN layer. With the key, GDN ordinal o starts ordinal o+1's copies at the END of its own GDN body (after
# `GDNBlock.emit` returns: its rec_in copies out of `pg_rec` were waited before the recurrence and `sd_conv` was consumed by the window
# select in the gather), so they fly under layer o's MoE, collectives and any QSA layer in between; ordinal o+1's gather then only waits.
# Same descriptors (`_side_copies(ctx, o + 1)`: same source plane, rows, destination and semaphores), same live predicate (`_live_gate`,
# constant within a step), same `gdn_side_on` predicate (constant within a call): every start has exactly one wait and the values read
# are the same bytes (plane o+1 is written only by ordinal o+1's own write-back, after its gather), so outputs are bitwise the flag-off
# block's. Ordinal 0 starts its own copies; the last ordinal (NGDN - 1) issues none, so no copy is in flight at kernel end. The pool
# branch (`gdn_side_on` = 0: admission / after a flush) is untouched: nothing is prefetched there ("side path only").
def _kern_prefetch() -> bool:
    from vllm_torchtpu.megakernel_runtime.qwen38_kern import side_prefetch_on

    return side_prefetch_on()


KERN_PF_TRACES: list = []  # (NGDN, NS) per traced prefetch site (tests / engagement)
_KERN_PF_AHEAD = 1  # the ordinal distance prefetched (tests only: the wrong-plane mutant sets 0)


def _kern_prefetch_next(self: Any, ctx: Any, l: Any) -> None:  # noqa: E741
    o = ctx.stream.sp.block_ord(l, self.name)
    ngdn = ctx.block_params(self.name).NGDN
    on = ctx.extra[SIDE_ON][0] != 0
    KERN_PF_TRACES.append((int(ngdn), int(ctx.ex.NS)))

    @pl.when(jnp.logical_and(on, o + 1 < ngdn))
    def _():
        cps = self._side_copies(ctx, o + _KERN_PF_AHEAD)
        for s in range(ctx.ex.NS):
            def start(s: int = s) -> None:
                for c in cps[s]:
                    c.start()

            self._live_gate(ctx, s, start)


Qwen38SidecarGDN._kern_prefetch_next = _kern_prefetch_next
# q38 HIX-KERN K1 end
