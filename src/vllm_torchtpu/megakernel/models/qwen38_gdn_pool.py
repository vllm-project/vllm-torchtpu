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

from ..exec_spec import ExecSpec
from ..blocks.gdn.params import GDNParams
from ..ops import BF16, F32, I32, REC, rup
from ..pool.config import cache_exec
from ..spec import ExtraInput, ScratchReq, StateDecl
from .qwen38 import Qwen38GDN

POOL = "gdn_pool"
READ_BLOCKS = "gdn_rb"
HAS_STATE = "gdn_hi"
MANAGER_ROWS = 512  # pool rows (tokens) per manager block
SSM_ROWS = 128  # pool rows per staged recurrent-state chunk
SSM_BLOCKS = 3  # chunks holding the recurrent state (rows 0 .. 383)
CONV_ROWS = 16  # pool rows holding the conv history (rounded to a power of two)


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


def pool_rows(pool: Any, m: Any, row: int, n: int) -> Any:
    """Rows `[row, row + n)` of manager block `m` of a `[blocks, kb, ...]` pool ref."""
    kb = pool.shape[1]
    assert MANAGER_ROWS % kb == 0 and row % kb + n <= kb, (kb, row, n)
    return pool.at[m * (MANAGER_ROWS // kb) + row // kb, pl.ds(row % kb, n)]


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
        ctx = self.ctx
        pool, scr = ctx.extra[POOL], ctx.scr
        cps = []
        for q, sq in enumerate(self.seqs):
            m = ctx.extra[READ_BLOCKS][self.o, sq]
            for j in range(SSM_BLOCKS):
                cps.append(
                    pltpu.make_async_copy(
                        pool_rows(pool, m, j * SSM_ROWS, SSM_ROWS),
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
            for c in self._copies(sv):
                c.wait()
            for q, sq in enumerate(self.seqs):
                keep = ctx.extra[HAS_STATE][self.o, sq] != 0
                state = _pool_state(
                    [scr[f"pg_cs{sv}{q}{j}"] for j in range(SSM_BLOCKS)], interp, vh, gp.KD, gp.VD
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
        # One whole ref per staged block: a typed view of a sliced ref is not
        # supported everywhere, a view of a whole scratch ref is.
        for b in range(2):
            for j in range(SSM_BLOCKS):
                scr[f"pg_raw{b}{j}"] = pltpu.VMEM((SSM_ROWS, 1, 4, 256), f8)
            scr[f"pg_craw{b}"] = pltpu.VMEM((CONV_ROWS, 1, 4, 256), f8)
        scr.update(
            pg_sem=pltpu.SemaphoreType.DMA((2, SSM_BLOCKS + 1)),
            pg_conv=pltpu.VMEM((ex.NS, rup(3 + ex.T, 8), lay.QKV), F32),
        )
        gp = plan.block_params[self.name]
        if _chunk_gather(gp, ex):
            # Per ring slot and request of a chunk: the state's FP8 pool rows.
            for sv in range(gp.NRB):
                for q in range(gp.CB // ex.T):
                    for j in range(SSM_BLOCKS):
                        scr[f"pg_cs{sv}{q}{j}"] = pltpu.VMEM((SSM_ROWS, 1, 4, 256), f8)
            scr["pg_csem"] = pltpu.SemaphoreType.DMA((gp.NRB, gp.CB // ex.T, SSM_BLOCKS))
        elif not _gather_into_recb(gp, ex):
            scr["pg_rec"] = pltpu.VMEM((ex.NS * lay.vh, lay.KD, lay.VD), REC)
        return scr

    def _copies(self, ctx: Any, o: Any, s: int, buf: int, ssm: bool = True) -> list[Any]:
        pool, scr = ctx.extra[POOL], ctx.scr
        m = ctx.extra[READ_BLOCKS][o, s]
        cps = [
            pltpu.make_async_copy(
                pool_rows(pool, m, j * SSM_ROWS, SSM_ROWS),
                scr[f"pg_raw{buf}{j}"],
                scr["pg_sem"].at[buf, j],
            )
            for j in range(SSM_BLOCKS if ssm else 0)
        ]
        cps.append(
            pltpu.make_async_copy(
                pool_rows(pool, m, SSM_BLOCKS * SSM_ROWS, CONV_ROWS),
                scr[f"pg_craw{buf}"],
                scr["pg_sem"].at[buf, SSM_BLOCKS],
            )
        )
        return cps

    def _gather(self, ctx: Any, o: Any, direct: bool = False, ssm: bool = True) -> None:
        """Layer `o`'s pool states -> `pg_rec` (`REC`) / `pg_conv` (f32 history rows).

        `direct`: the states go to each request's first `recb` row group instead
        (the slot `rec_in` would copy them to), once the previous GDN layer's
        `recb` write-back has drained; `gdn_in` then skips both. `ssm=False` (chunk
        gather): only the conv history; the chunks read their states (`_ChunkIn`)."""
        ex, scr = ctx.ex, ctx.scr
        lay = self.lin_in_layout(ctx.g, ex)
        vh, KD, VD, QKV = lay.vh, lay.KD, lay.VD, lay.QKV
        NS = ex.NS
        interp = bool(ex.interpret)
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
        pending = self._copies(ctx, o, 0, 0, ssm)
        for c in pending:
            c.start()
        for s in range(NS):
            buf = s % 2
            if s + 1 < NS:
                nxt = self._copies(ctx, o, s + 1, 1 - buf, ssm)
                for c in nxt:
                    c.start()
            for c in pending:
                c.wait()
            keep = ctx.extra[HAS_STATE][o, s] != 0
            if ssm:
                state = _pool_state(
                    [scr[f"pg_raw{buf}{j}"] for j in range(SSM_BLOCKS)], interp, vh, KD, VD
                )
                rec_dst[s * rec_stride : s * rec_stride + vh] = jnp.where(
                    keep, state, 0.0
                ).astype(REC)
            crow = _typed_f32(scr[f"pg_craw{buf}"], interp, 256)  # [16, 256]
            per = QKV // 256
            for r in range(3):
                for c in range(per):
                    scr["pg_conv"][s, r : r + 1, c * 256 : (c + 1) * 256] = jnp.where(
                        keep, crow[r * per + c : r * per + c + 1], 0.0
                    )
            if s + 1 < NS:
                pending = nxt

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        o = ctx.stream.sp.block_ord(l, self.name)
        gp = ctx.block_params(self.name)
        chunked = _chunk_gather(gp, ctx.ex)
        direct = not chunked and _gather_into_recb(gp, ctx.ex)
        with ctx.sc("gdn_pool_gather"):
            self._gather(ctx, o, direct, ssm=not chunked)
        st = {**ctx.ST, "conv": _LayerView(ctx.scr["pg_conv"])}
        if chunked:
            st["rec_chunk_in"] = _ChunkIn
        elif direct:
            st["rec_in_recb"] = True
        else:
            st["rec"] = _LayerView(ctx.scr["pg_rec"])
        return super().emit(dataclasses.replace(ctx, ST=st), l, slot)

    def host_update_states(self, g, ex, pos, states, raw_outputs):
        return {"conv_ckpt": raw_outputs["conv_ckpt"], "rec_ckpt": raw_outputs["rec_ckpt"]}
