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
"""Qwen3.8 GDN block that reads its states from the qwen4_exp unified pool.

The base `GDNBlock` takes per-layer `conv` / `rec` state arrays. Here the
kernel reads them itself from the model's pool (extra input `gdn_pool`, one
per-rank pool `[blocks * split, kb, 1, 4, 256]` FP8 bytes, `kb * split = 512`):
manager block `m` holds the f32 recurrent state `[vh, KD, VD]` in its rows
`0 .. 383` and the f32 conv history `[3, QKV]` in rows `384 .. 399`, whatever
the kernel block size `kb` (128 or 512 in served configs).
`gdn_rb [NGDN, NS]` (SMEM) names each request's read block and `gdn_hi`
whether it has a prior state (0 = start from zero). Bytes are reinterpreted
with the same typed view as the model's own GDN kernel (`ref.bitcast` to f32,
row-major). Checkpoints are still returned as `conv_ckpt` / `rec_ckpt`.

With an f32 recurrent state (`VLLM_TPU_MK_GDN_REC_F32=1`) on the whole-state path
the gather writes each request's state straight into the block's `recb` instead
of through `pg_rec`: an f32 `pg_rec` (NS * 384 KiB) does not fit beside the f32
`recb` at b32.

Chunk gather (`VLLM_TPU_MK_GDN_CHUNK_GATHER`, auto above 32 rows): on the chunked
(`GCH`) verify path each row chunk's states are read from the pool when the chunk's
`recb` slot is filled, through per-slot FP8 staging, instead of from a `pg_rec`
holding every request's state (NS * 384 KiB f32: 12 MiB at 128 rows). Same values
(typed f32 view, zero without a prior state, cast to `REC`); scheduling only.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .. import pool_alias
from ..exec_spec import ExecSpec
from ..blocks.gdn.params import GDNParams
from ..ops import BF16, F32, I32, REC, rup
from ..pool.config import cache_exec
from ..spec import ExtraInput, ScratchReq, StateDecl
from .qwen38 import Qwen38GDN

POOL = "gdn_pool"
READ_BLOCKS = "gdn_rb"
HAS_STATE = "gdn_hi"
MANAGER_ROWS = 512  # default TP=8 pool rows (tokens) per manager block
SSM_ROWS = 128  # pool rows per staged recurrent-state chunk
SSM_BLOCKS = 3  # default TP=8 chunks holding the recurrent state (rows 0 .. 383)
CONV_ROWS = 16  # default TP=8 pool rows holding the conv history (rounded to a power of two)


def gdn_slot_geometry(
    vh: int, KD: int, VD: int, QKV: int, manager_rows: int = 0
) -> tuple[int, int, int, int]:
    """`(ssm_blocks, conv_rows, log_row0, manager_rows)` for per-rank `(vh, KD, VD, QKV)`.

    Matches `gdn_pool_layout.derive_pooled_gdn_state_layout` + `models/vllm/qwen4_exp/pool_layout.py`
    across both TP=8 (`(3, 16, 400, 512)`) and TP=4 (`(6, 32, 800, 896)`).
    """
    ssm_tokens = -(-(vh * KD * VD * 4) // 1024)
    ssm_blocks = max(1, -(-ssm_tokens // SSM_ROWS))
    conv_rows = 1
    while conv_rows * 1024 < 3 * QKV * 4:
        conv_rows *= 2
    log_row0 = ssm_tokens + conv_rows
    mgr_rows = manager_rows if manager_rows > 0 else rup(log_row0, SSM_ROWS)
    return ssm_blocks, conv_rows, log_row0, mgr_rows


# "auto": chunk gather on chunked verify buckets above 32 rows; "1": every chunked
# verify bucket; "0": off (states staged in `pg_rec`).
CHUNK_GATHER = os.environ.get("VLLM_TPU_MK_GDN_CHUNK_GATHER", "auto")
CHUNK_GATHER_AUTO_ROWS = 32


def _chunk_gather(gp: GDNParams, ex: ExecSpec) -> bool:
    """Read each `GCH` row chunk's states from the pool into its `recb` slot."""
    if CHUNK_GATHER == "0" or not (gp.GCH and ex.VER) or gp.GPR or cache_exec(ex).FUSED:
        return False
    return CHUNK_GATHER == "1" or ex.B > CHUNK_GATHER_AUTO_ROWS


def _gather_into_recb(gp: GDNParams, ex: ExecSpec) -> bool:
    """Gather pool states straight into `recb` (f32 state, whole-state path)."""
    return REC == F32 and not gp.GCH and not cache_exec(ex).FUSED


def pool_rows(
    pool: Any, m: Any, row: int, n: int, manager_rows: int = MANAGER_ROWS
) -> Any:
    """Rows `[row, row + n)` of manager block `m` of a `[blocks, kb, ...]` pool ref."""
    kb = pool.shape[1]
    if kb > manager_rows:
        manager_rows = kb
    assert manager_rows % kb == 0 and row % kb + n <= kb, (manager_rows, kb, row, n)
    return pool.at[m * (manager_rows // kb) + row // kb, pl.ds(row % kb, n)]


class _LayerView:
    """Stands in for a per-layer state array: `.at[o, *rest]` -> `ref.at[*rest]`."""

    def __init__(self, ref: Any) -> None:
        self.ref = ref
        self.at = self

    def __getitem__(self, idx: Any) -> Any:
        rest = idx[1:] if isinstance(idx, tuple) else ()
        return self.ref.at[rest] if rest else self.ref


def _typed_f32(block_ref: Any, interpret: bool, lanes: int) -> Any:
    """Whole `[rows, 1, 4, 256]` FP8 scratch ref -> f32 values `(-1, lanes)`, row-major.

    4 consecutive pack rows form one 32-bit word per lane. On TPU this is the
    ref bitcast + lane split the model's GDN kernel uses
    (`kernels/typed_ldst.py`); the Pallas TPU interpreter has no ref bitcast,
    so interpret mode bitcasts the loaded bytes, which groups the same rows.
    """
    if interpret:
        raw = block_ref[...]
        words = pltpu.bitcast(raw.reshape(-1, raw.shape[-1]), jnp.float32)
        return words.reshape(-1, lanes)
    if jnp.dtype(block_ref.dtype) == jnp.float32:
        return block_ref[...].reshape(-1, lanes)
    view = block_ref.bitcast(jnp.float32)
    return view.reshape(-1, lanes)[...]


def _pool_state(raws: list[Any], interp: bool, vh: int, KD: int, VD: int) -> Any:
    """A request's f32 recurrent state `[vh, KD, VD]` from its staged FP8 pool rows."""
    # 128-lane rows: the state's row-major order, so only the leading
    # dimension is split (lane-changing reshapes do not lower).
    return jnp.concatenate([_typed_f32(r, interp, VD) for r in raws], axis=0)[
        : vh * KD
    ].reshape(vh, KD, VD)


class _ChunkIn:
    """`rec_chunk` read handle of the chunk gather: row chunk `j`'s requests' pool
    states -> the tail of `recb` slot `s` (where the stock chunk read puts them).

    `start` issues the requests' FP8 pool-row copies into slot `s`'s staging;
    `wait` waits for them and writes the f32 states (zero without a prior state)
    as `REC`. The caller orders both after slot `s`'s previous write-back."""

    def __init__(self, ctx: Any, gp: GDNParams, o: Any, j: Any, s: Any, n: int) -> None:
        T = ctx.ex.T
        self.ctx, self.gp, self.o, self.s, self.n = ctx, gp, o, s, n
        self.nseq = n // (T * gp.vh)
        per = gp.CB // T
        r0 = j * per if isinstance(j, int) else pl.multiple_of(j * per, per)
        self.seqs = [r0 + q for q in range(self.nseq)]

    def _slots(self, body: Any) -> None:
        """`body(sv)` for the static slot `sv == s` (staging refs are whole refs)."""
        if isinstance(self.s, int):
            body(self.s)
            return
        for sv in range(self.gp.NRB):

            @pl.when(self.s == sv)
            def _(sv=sv):
                body(sv)

    def _copies(self, sv: int) -> list[Any]:
        ctx, gp = self.ctx, self.gp
        ssm_blocks, _, _, mgr_rows = gdn_slot_geometry(gp.vh, gp.KD, gp.VD, gp.QKV)
        pool, scr = ctx.extra[POOL], ctx.scr
        cps = []
        for q, sq in enumerate(self.seqs):
            m = ctx.extra[READ_BLOCKS][self.o, sq]
            for j in range(ssm_blocks):
                cps.append(
                    pltpu.make_async_copy(
                        pool_rows(pool, m, j * SSM_ROWS, SSM_ROWS, mgr_rows),
                        scr[f"pg_cs{sv}{q}{j}"],
                        scr["pg_csem"].at[sv, q, j],
                    )
                )
        return cps

    def start(self) -> None:
        def body(sv: int) -> None:
            for c in self._copies(sv):
                c.start()

        self._slots(body)

    def wait(self) -> None:
        ctx, gp = self.ctx, self.gp
        scr, vh = ctx.scr, gp.vh
        interp = bool(ctx.ex.interpret)
        ni = self.n // ctx.ex.T

        def body(sv: int) -> None:
            ssm_blocks, _, _, _ = gdn_slot_geometry(gp.vh, gp.KD, gp.VD, gp.QKV)
            for c in self._copies(sv):
                c.wait()
            for q, sq in enumerate(self.seqs):
                keep = ctx.extra[HAS_STATE][self.o, sq] != 0
                state = _pool_state(
                    [scr[f"pg_cs{sv}{q}{j}"] for j in range(ssm_blocks)], interp, vh, gp.KD, gp.VD
                )
                r = self.n - ni + q * vh
                scr["recb"][sv, r : r + vh] = jnp.where(keep, state, 0.0).astype(REC)

        self._slots(body)


class Qwen38PoolGDN(Qwen38GDN):
    """`Qwen38GDN` whose input states come straight from the unified pool."""

    def states(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return []

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        n = (g.NGDN, ex.NS)
        return [
            ExtraInput(POOL, "hbm"),
            ExtraInput(READ_BLOCKS, "smem", n, I32),
            ExtraInput(HAS_STATE, "smem", n, I32),
        ]

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        scr = dict(super().scratch(g, ex, plan))
        lay = self.lin_in_layout(g, ex)
        f8 = jnp.float8_e4m3fn
        ssm_blocks, conv_rows, _, _ = gdn_slot_geometry(lay.vh, lay.KD, lay.VD, lay.QKV)
        # One whole ref per staged block: a typed view of a sliced ref is not
        # supported everywhere, a view of a whole scratch ref is.
        for b in range(2):
            for j in range(ssm_blocks):
                scr[f"pg_raw{b}{j}"] = pltpu.VMEM((SSM_ROWS, 1, 4, 256), f8)
            scr[f"pg_craw{b}"] = pltpu.VMEM((conv_rows, 1, 4, 256), f8)
        scr.update(
            pg_sem=pltpu.SemaphoreType.DMA((2, ssm_blocks + 1)),
            pg_conv=pltpu.VMEM((ex.NS, rup(3 + ex.T, 8), lay.QKV), F32),
        )
        gp = plan.block_params[self.name]
        if _chunk_gather(gp, ex):
            # Per ring slot and request of a chunk: the state's FP8 pool rows.
            for sv in range(gp.NRB):
                for q in range(gp.CB // ex.T):
                    for j in range(ssm_blocks):
                        scr[f"pg_cs{sv}{q}{j}"] = pltpu.VMEM((SSM_ROWS, 1, 4, 256), f8)
            scr["pg_csem"] = pltpu.SemaphoreType.DMA((gp.NRB, gp.CB // ex.T, ssm_blocks))
        elif not _gather_into_recb(gp, ex):
            scr["pg_rec"] = pltpu.VMEM((ex.NS * lay.vh, lay.KD, lay.VD), REC)
        if bool(ex.resolved_tiles.xlayer_prefetch) and not ex.interpret and ex.NS > 2:
            tiles = ex.resolved_tiles
            ebrows = tiles.ebank // 128
            conv_ep = conv_rows * 8  # (conv_rows * 256 * 4 bytes) // 128 bytes/row
            rows_per_seq = (
                (ssm_blocks * 1024 + conv_ep) if not _chunk_gather(gp, ex) else conv_ep
            )
            k_ep = min(ex.NS - 2, (tiles.nbe * ebrows) // rows_per_seq)
            if k_ep > 0:
                scr["pg_esem"] = pltpu.SemaphoreType.DMA((k_ep, ssm_blocks + 1))
        return scr

    def _ep_slots(
        self,
        ctx: Any,
        ssm: bool,
        ssm_blocks: int = SSM_BLOCKS,
        conv_rows: int = CONV_ROWS,
    ) -> int:
        scr = ctx.scr
        if (
            "epool" not in scr
            or "pg_esem" not in scr
            or scr["epool"].ndim != 2
            or scr["epool"].shape[1] != 128
        ):
            return 0
        conv_ep = conv_rows * 8
        rows_per_seq = (ssm_blocks * 1024 + conv_ep) if ssm else conv_ep
        return min(
            ctx.ex.NS - 2,
            scr["pg_esem"].shape[0],
            scr["epool"].shape[0] // rows_per_seq,
        )

    def _ep_views(
        self,
        ctx: Any,
        k: int,
        ssm: bool,
        ssm_blocks: int = SSM_BLOCKS,
        conv_rows: int = CONV_ROWS,
    ) -> tuple[list[Any], Any]:
        epool = ctx.scr["epool"]
        conv_ep = conv_rows * 8
        stride = (ssm_blocks * 1024 + conv_ep) if ssm else conv_ep
        base = k * stride
        raws = [
            pool_alias.view(
                epool.at[pl.ds(base + j * 1024, 1024)],
                (SSM_ROWS, 256),
                F32,
            )
            for j in range(ssm_blocks if ssm else 0)
        ]
        conv_off = base + (ssm_blocks * 1024 if ssm else 0)
        craw = pool_alias.view(
            epool.at[pl.ds(conv_off, conv_ep)],
            (conv_rows, 256),
            F32,
        )
        return raws, craw

    def _ep_copies(
        self,
        ctx: Any,
        o: Any,
        s: int,
        k: int,
        ssm: bool = True,
        ssm_blocks: int = SSM_BLOCKS,
        conv_rows: int = CONV_ROWS,
        mgr_rows: int = MANAGER_ROWS,
    ) -> list[Any]:
        pool, scr = ctx.extra[POOL], ctx.scr
        m = ctx.extra[READ_BLOCKS][o, s]
        raws, craw = self._ep_views(ctx, k, ssm, ssm_blocks, conv_rows)
        cps = [
            pltpu.make_async_copy(
                pool_rows(pool, m, j * SSM_ROWS, SSM_ROWS, mgr_rows)
                .bitcast(F32)
                .reshape(SSM_ROWS, 256),
                raws[j],
                scr["pg_esem"].at[k, j],
            )
            for j in range(ssm_blocks if ssm else 0)
        ]
        cps.append(
            pltpu.make_async_copy(
                pool_rows(pool, m, ssm_blocks * SSM_ROWS, conv_rows, mgr_rows)
                .bitcast(F32)
                .reshape(conv_rows, 256),
                craw,
                scr["pg_esem"].at[k, ssm_blocks],
            )
        )
        return cps

    def _copies(self, ctx: Any, o: Any, s: int, buf: int, ssm: bool = True) -> list[Any]:
        lay = self.lin_in_layout(ctx.g, ctx.ex)
        ssm_blocks, conv_rows, _, mgr_rows = gdn_slot_geometry(
            lay.vh, lay.KD, lay.VD, lay.QKV
        )
        pool, scr = ctx.extra[POOL], ctx.scr
        m = ctx.extra[READ_BLOCKS][o, s]
        cps = [
            pltpu.make_async_copy(
                pool_rows(pool, m, j * SSM_ROWS, SSM_ROWS, mgr_rows),
                scr[f"pg_raw{buf}{j}"],
                scr["pg_sem"].at[buf, j],
            )
            for j in range(ssm_blocks if ssm else 0)
        ]
        cps.append(
            pltpu.make_async_copy(
                pool_rows(pool, m, ssm_blocks * SSM_ROWS, conv_rows, mgr_rows),
                scr[f"pg_craw{buf}"],
                scr["pg_sem"].at[buf, ssm_blocks],
            )
        )
        return cps

    def prefetch_next(self, ctx: Any, l: Any) -> None:  # noqa: E741
        o = ctx.stream.sp.block_ord(l, self.name)
        gp = ctx.block_params(self.name)
        ssm = not _chunk_gather(gp, ctx.ex)
        lay = self.lin_in_layout(ctx.g, ctx.ex)
        ssm_blocks, conv_rows, _, mgr_rows = gdn_slot_geometry(
            lay.vh, lay.KD, lay.VD, lay.QKV
        )
        NS = ctx.ex.NS
        k_ep = self._ep_slots(ctx, ssm, ssm_blocks, conv_rows)
        n_pre = min(NS, 2 + k_ep)
        if ctx.ex.live_rows:
            from ..live_rows import req_live

            for s in range(min(NS, 2)):

                @pl.when(req_live(ctx, s))
                def _(s=s):
                    for c in self._copies(ctx, o, s, s, ssm):
                        c.start()

            for s in range(2, n_pre):

                @pl.when(req_live(ctx, s))
                def _(s=s):
                    for c in self._ep_copies(
                        ctx, o, s, s - 2, ssm, ssm_blocks, conv_rows, mgr_rows
                    ):
                        c.start()

            return
        for s in range(min(NS, 2)):
            for c in self._copies(ctx, o, s, s, ssm):
                c.start()
        for s in range(2, n_pre):
            for c in self._ep_copies(
                ctx, o, s, s - 2, ssm, ssm_blocks, conv_rows, mgr_rows
            ):
                c.start()

    def _unpack_seq(
        self,
        ctx: Any,
        o: Any,
        s: int,
        raws: list[Any],
        craw: Any,
        ssm: bool,
        rec_dst: Any,
        rec_stride: int,
        *,
        live_bit: Any = None,
        skip_pad_store: bool = False,
    ) -> None:
        ex, scr = ctx.ex, ctx.scr
        lay = self.lin_in_layout(ctx.g, ex)
        vh, KD, VD, QKV = lay.vh, lay.KD, lay.VD, lay.QKV
        interp = bool(ex.interpret)
        keep = ctx.extra[HAS_STATE][o, s] != 0
        if live_bit is not None:
            keep = jnp.logical_and(keep, live_bit)
        if ssm:
            state = _pool_state(raws, interp, vh, KD, VD)
            val = jnp.where(keep, state, 0.0).astype(REC)
            if skip_pad_store and live_bit is not None:
                import functools

                pl.when(live_bit)(
                    functools.partial(_store_rows, rec_dst, s * rec_stride, vh, val)
                )
            else:
                rec_dst[s * rec_stride : s * rec_stride + vh] = val
        crow = _typed_f32(craw, interp, 256)
        per = QKV // 256
        for r in range(3):
            for c in range(per):
                scr["pg_conv"][s, r : r + 1, c * 256 : (c + 1) * 256] = jnp.where(
                    keep, crow[r * per + c : r * per + c + 1], 0.0
                )

    def _gather(
        self,
        ctx: Any,
        o: Any,
        direct: bool = False,
        ssm: bool = True,
        l: Any = 0,  # noqa: E741
    ) -> None:
        """Layer `o`'s pool states -> `pg_rec` (`REC`) / `pg_conv` (f32 history rows). `direct`: the states go
        to each request's first `recb` row group instead (the slot `rec_in` would copy them to), once the previous
        GDN layer's `recb` write-back has drained; `gdn_in` then skips both. `ssm=False` (chunk gather): only the
        conv history; the chunks read their states (`_ChunkIn`). q38 X-K-LIVE: with `ctx.ex.live_rows` the next
        line hands over to `_gather_live` (pad requests skipped); it replaces a docstring line (no line moves)."""
        if ctx.ex.live_rows: return self._gather_live(ctx, o, direct, ssm, l=l)  # noqa: E701
        ex, scr = ctx.ex, ctx.scr
        lay = self.lin_in_layout(ctx.g, ex)
        vh, KD, VD, QKV = lay.vh, lay.KD, lay.VD, lay.QKV
        NS = ex.NS
        rec_dst, rec_stride = None, 0
        if direct:
            rec_dst, rec_stride = scr["recb"], (ex.T if ex.VER else 1) * vh
            if "gdn_state" not in ctx.skip:

                @pl.when(o > 0)
                def _():
                    pltpu.make_async_copy(
                        rec_dst,
                        ctx.O["rec_ckpt" if ex.VER else "rec"].at[0],
                        scr["st_sem"].at[3],
                    ).wait()

        elif ssm:
            rec_dst, rec_stride = scr["pg_rec"], vh
        scr["pg_conv"][...] = jnp.zeros(scr["pg_conv"].shape, F32)
        ssm_blocks, conv_rows, _, mgr_rows = gdn_slot_geometry(vh, KD, VD, QKV)
        if ctx.xlayer_prefetch is not None:
            k_ep = self._ep_slots(ctx, ssm, ssm_blocks, conv_rows)
            n_pre = min(NS, 2 + k_ep)

            @pl.when(l == 0)
            def _():
                for s0 in range(min(NS, 2)):
                    for c in self._copies(ctx, o, s0, s0, ssm):
                        c.start()
                for s0 in range(2, n_pre):
                    for c in self._ep_copies(
                        ctx, o, s0, s0 - 2, ssm, ssm_blocks, conv_rows, mgr_rows
                    ):
                        c.start()

            for s in range(min(NS, 2)):
                for c in self._copies(ctx, o, s, s, ssm):
                    c.wait()
                raws = [scr[f"pg_raw{s}{j}"] for j in range(ssm_blocks if ssm else 0)]
                self._unpack_seq(ctx, o, s, raws, scr[f"pg_craw{s}"], ssm, rec_dst, rec_stride)
                if n_pre + s < NS:
                    for c in self._copies(ctx, o, n_pre + s, s, ssm):
                        c.start()
            for s in range(2, n_pre):
                for c in self._ep_copies(
                    ctx, o, s, s - 2, ssm, ssm_blocks, conv_rows, mgr_rows
                ):
                    c.wait()
                raws, craw = self._ep_views(ctx, s - 2, ssm, ssm_blocks, conv_rows)
                self._unpack_seq(ctx, o, s, raws, craw, ssm, rec_dst, rec_stride)
            for s in range(n_pre, NS):
                buf = (s - n_pre) % 2
                for c in self._copies(ctx, o, s, buf, ssm):
                    c.wait()
                raws = [scr[f"pg_raw{buf}{j}"] for j in range(ssm_blocks if ssm else 0)]
                self._unpack_seq(ctx, o, s, raws, scr[f"pg_craw{buf}"], ssm, rec_dst, rec_stride)
                if s + 2 < NS:
                    for c in self._copies(ctx, o, s + 2, buf, ssm):
                        c.start()
            return

        pending = self._copies(ctx, o, 0, 0, ssm)
        for c in pending:
            c.start()
        nxt = pending
        for s in range(NS):
            buf = s % 2
            if s + 1 < NS:
                nxt = self._copies(ctx, o, s + 1, 1 - buf, ssm)
                for c in nxt:
                    c.start()
            for c in pending:
                c.wait()
            raws = [scr[f"pg_raw{buf}{j}"] for j in range(ssm_blocks if ssm else 0)]
            self._unpack_seq(ctx, o, s, raws, scr[f"pg_craw{buf}"], ssm, rec_dst, rec_stride)
            if s + 1 < NS:
                pending = nxt

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        o = ctx.stream.sp.block_ord(l, self.name)
        gp = ctx.block_params(self.name)
        chunked = _chunk_gather(gp, ctx.ex)
        direct = not chunked and _gather_into_recb(gp, ctx.ex)
        with ctx.sc("gdn_pool_gather"):
            self._gather(ctx, o, direct, ssm=not chunked, l=l)
        st = {**ctx.ST, "conv": _LayerView(ctx.scr["pg_conv"])}
        if chunked:
            st["rec_chunk_in"] = _ChunkInLive if ctx.ex.live_rows else _ChunkIn  # q38 X-K-LIVE
        elif direct:
            st["rec_in_recb"] = True
        else:
            st["rec"] = _LayerView(ctx.scr["pg_rec"])
        return super().emit(dataclasses.replace(ctx, ST=st), l, slot)

    def host_update_states(self, g, ex, pos, states, raw_outputs):
        return {
            "conv_ckpt": raw_outputs["conv_ckpt"],
            "rec_ckpt": raw_outputs["rec_ckpt"],
        }

    # q38 X-K-LIVE begin (L1 GDN half): appended to the class (no existing line moves).
    def _gather_live(
        self,
        ctx: Any,
        o: Any,
        direct: bool = False,
        ssm: bool = True,
        l: Any = 0,  # noqa: E741
    ) -> None:
        """`_gather` with each pad request's pool-row copies skipped (`ctx.extra["live_rows"]`): neither started nor
        waited. The unpacking after the wait stays straight-line, as in `_gather`, with the request's live bit folded
        into `keep`, so a pad request's `pg_rec` / `pg_conv` rows get zeros (what `_gather` writes for a request without
        a prior state, as every served pad request is) from its unfilled staging buffer; on the direct path (states
        straight into `recb`) in mode 2 a pad request's store is skipped, so no `recb` state slot of a pad request is
        written (mode 1 keeps it: the recurrence then runs on its zero state, as with the flag off).
        The live requests' copies, buffers and stores are `_gather`'s, in the same order. (9004e0aa9 guarded the
        unpacking too; l1-live-aot-i2 j-f7cac4a2 found the flag-on 16-row bucket 222 KiB over scoped VMEM.)"""
        from ..live_rows import req_live

        ex, scr = ctx.ex, ctx.scr
        lay = self.lin_in_layout(ctx.g, ex)
        vh, KD, VD, QKV = lay.vh, lay.KD, lay.VD, lay.QKV
        NS = ex.NS
        rec_dst, rec_stride = None, 0
        if direct:
            rec_dst, rec_stride = scr["recb"], (ex.T if ex.VER else 1) * vh
            if "gdn_state" not in ctx.skip:

                @pl.when(o > 0)
                def _():
                    pltpu.make_async_copy(
                        rec_dst,
                        ctx.O["rec_ckpt" if ex.VER else "rec"].at[0],
                        scr["st_sem"].at[3],
                    ).wait()

        elif ssm:
            rec_dst, rec_stride = scr["pg_rec"], vh
        scr["pg_conv"][...] = jnp.zeros(scr["pg_conv"].shape, F32)
        live = [req_live(ctx, s) for s in range(NS)]
        skip_pad = direct and int(ex.live_rows) >= 2
        ssm_blocks, conv_rows, _, mgr_rows = gdn_slot_geometry(vh, KD, VD, QKV)

        def start(s: int, buf: int) -> None:
            @pl.when(live[s])
            def _():
                for c in self._copies(ctx, o, s, buf, ssm):
                    c.start()

        if ctx.xlayer_prefetch is not None:
            k_ep = self._ep_slots(ctx, ssm, ssm_blocks, conv_rows)
            n_pre = min(NS, 2 + k_ep)

            @pl.when(l == 0)
            def _():
                for s0 in range(min(NS, 2)):
                    start(s0, s0)
                for s0 in range(2, n_pre):

                    @pl.when(live[s0])
                    def _(s0=s0):
                        for c in self._ep_copies(
                            ctx, o, s0, s0 - 2, ssm, ssm_blocks, conv_rows, mgr_rows
                        ):
                            c.start()

            for s in range(min(NS, 2)):

                @pl.when(live[s])
                def _(s=s):
                    for c in self._copies(ctx, o, s, s, ssm):
                        c.wait()

                raws = [scr[f"pg_raw{s}{j}"] for j in range(ssm_blocks if ssm else 0)]
                self._unpack_seq(
                    ctx, o, s, raws, scr[f"pg_craw{s}"], ssm, rec_dst, rec_stride,
                    live_bit=live[s], skip_pad_store=skip_pad,
                )
                if n_pre + s < NS:
                    start(n_pre + s, s)
            for s in range(2, n_pre):

                @pl.when(live[s])
                def _(s=s):
                    for c in self._ep_copies(
                        ctx, o, s, s - 2, ssm, ssm_blocks, conv_rows, mgr_rows
                    ):
                        c.wait()

                raws, craw = self._ep_views(ctx, s - 2, ssm, ssm_blocks, conv_rows)
                self._unpack_seq(
                    ctx, o, s, raws, craw, ssm, rec_dst, rec_stride,
                    live_bit=live[s], skip_pad_store=skip_pad,
                )
            for s in range(n_pre, NS):
                buf = (s - n_pre) % 2

                @pl.when(live[s])
                def _(s=s, buf=buf):
                    for c in self._copies(ctx, o, s, buf, ssm):
                        c.wait()

                raws = [scr[f"pg_raw{buf}{j}"] for j in range(ssm_blocks if ssm else 0)]
                self._unpack_seq(
                    ctx, o, s, raws, scr[f"pg_craw{buf}"], ssm, rec_dst, rec_stride,
                    live_bit=live[s], skip_pad_store=skip_pad,
                )
                if s + 2 < NS:
                    start(s + 2, buf)
            return

        start(0, 0)
        for s in range(NS):
            buf = s % 2
            if s + 1 < NS:
                start(s + 1, 1 - buf)

            @pl.when(live[s])
            def _(s=s, buf=buf):
                for c in self._copies(ctx, o, s, buf, ssm):
                    c.wait()

            raws = [scr[f"pg_raw{buf}{j}"] for j in range(ssm_blocks if ssm else 0)]
            self._unpack_seq(
                ctx, o, s, raws, scr[f"pg_craw{buf}"], ssm, rec_dst, rec_stride,
                live_bit=live[s], skip_pad_store=skip_pad,
            )


def _store_rows(ref: Any, r0: int, n: int, val: Any) -> None:
    ref[r0 : r0 + n] = val


class _ChunkInLive(_ChunkIn):
    """q38 X-K-LIVE: `_ChunkIn` with each pad request's pool-row copies skipped (start and wait). The unpacking stays
    unguarded with the live bit folded into `keep` (a pad request's `recb` input rows get zeros from its unfilled
    staging, what `_ChunkIn` writes for a request without a prior state); a live request's copies and store are
    `_ChunkIn`'s."""

    def start(self) -> None:
        from ..live_rows import req_live

        gp = self.gp

        def body(sv: int) -> None:
            ssm_blocks, _, _, _ = gdn_slot_geometry(gp.vh, gp.KD, gp.VD, gp.QKV)
            cps = self._copies(sv)
            for q, sq in enumerate(self.seqs):

                @pl.when(req_live(self.ctx, sq))
                def _(q=q):
                    for c in cps[q * ssm_blocks : (q + 1) * ssm_blocks]:
                        c.start()

        self._slots(body)

    def wait(self) -> None:
        from ..live_rows import req_live

        ctx, gp = self.ctx, self.gp
        scr, vh = ctx.scr, gp.vh
        interp = bool(ctx.ex.interpret)
        ni = self.n // ctx.ex.T

        def body(sv: int) -> None:
            ssm_blocks, _, _, _ = gdn_slot_geometry(gp.vh, gp.KD, gp.VD, gp.QKV)
            cps = self._copies(sv)
            for q, sq in enumerate(self.seqs):
                live = req_live(ctx, sq)

                @pl.when(live)
                def _(q=q):
                    for c in cps[q * ssm_blocks : (q + 1) * ssm_blocks]:
                        c.wait()

            for q, sq in enumerate(self.seqs):
                keep = jnp.logical_and(ctx.extra[HAS_STATE][self.o, sq] != 0, req_live(ctx, sq))
                state = _pool_state(
                    [scr[f"pg_cs{sv}{q}{j}"] for j in range(ssm_blocks)], interp, vh, gp.KD, gp.VD
                )
                r = self.n - ni + q * vh
                scr["recb"][sv, r : r + vh] = jnp.where(keep, state, 0.0).astype(REC)

        self._slots(body)
# q38 X-K-LIVE end
