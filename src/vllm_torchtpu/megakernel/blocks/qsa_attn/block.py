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
"""`QSASparseAttnBlock`: QSA sparse decode attention + gate, then `o_proj`, in-kernel.

Replaces the external attention input of `QSAOutBlock`. Per verify row it reads
the row's selected tokens straight from the model's paged FP8 main-KV pool
(`qsa_pool`, per rank `[pages, 128, 1, 4, 256]`: each 1 KiB token row packs
`[K0, V0, K1, V1]` of a layer pair), one 4-token micro-page per DMA, through a
2-slot ring of `TILE`-token buffers, and runs an online softmax over them.

`qsa_pages [M, budget + 1]` holds each row's micro-page ids and `qsa_kvlen [M]`
its visible token count, both computed outside the kernel exactly as the
fused sparse decode path does (`virtual_page_metadata`); rows with zero length
output zero. The gated output follows the stock contract:
`bf16(bf16(attn) * bf16(sigmoid(gate)))`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...exec_spec import ExecSpec
from ...ops import BF16, F32, I32
from ...spec import ExtraInput, ScratchReq
from ..qsa_out.block import QSAOutBlock

POOL = "qsa_pool"
QUERY = "qsa_q"
GATE = "qsa_gate"
PAGES = "qsa_pages"
KV_LEN = "qsa_kvlen"
MICRO = 4  # tokens per micro-page (the compression ratio)
TILE = 256  # tokens per ring slot
SLOTS = 8  # ring slots: SLOTS - 1 tiles in flight ahead of the one being read


def _unpack_kv(slot_ref: Any, off: int, interpret: bool) -> tuple[jax.Array, jax.Array]:
    """FP8 K / V `[TILE, 256]` of head pair `off` from a `[TILE, 1, 4, 256]` slot.

    Four pack rows form one 32-bit word per lane, row `c` in byte `c`; K and V of
    pair `off` are bytes `2 * off` and `2 * off + 1`.
    """
    if interpret:
        raw = slot_ref[...]
        words = pltpu.bitcast(raw.reshape(-1, raw.shape[-1]), jnp.uint32)
    else:
        words = slot_ref.bitcast(jnp.uint32).reshape(-1, slot_ref.shape[-1])[...]

    def byte(c: int) -> jax.Array:
        b = ((words >> (8 * c)) & 0xFF).astype(jnp.uint16).astype(jnp.uint8)
        return pltpu.bitcast(b, jnp.float8_e4m3fn)

    return byte(2 * off), byte(2 * off + 1)


def attend_row(
    r: Any,
    g0: Any = 0,
    pre: Any = 0,
    *,
    nrows: int | None = None,
    pool: Any,
    ring: Any,
    sems: Any,
    pages: Any,
    kv_len: Any,
    q: jax.Array,
    gate: jax.Array,
    off: int,
    scale_k: float,
    scale_v: float,
    sm_scale: float,
    interpret: bool,
    fold: bool = False,
    ablate: str = "",
) -> tuple[jax.Array, Any, Any]:
    """Gated sparse attention of row `r`: `([nh, 256]` F32, next `g0`, next `pre`).

    Ring slots follow a tile counter carried across rows (`g0`: this row's
    first tile), so the ring never drains at a row boundary: once a row has
    issued its own tiles, its last iterations issue row `r + 1`'s first tiles
    into the slots that row will read. `pre` is how many of this row's tiles a
    previous row already issued. Rows run in order `0 .. nrows - 1`.

    `fold`: apply `scale_k * sm_scale` to q and `scale_v` to the result instead
    of to every K / V tile. `ablate` (timing only, wrong results): "dma" skips
    the copies, "unpack" uses zero K / V, "matmul" skips QK and PV.

    `q`, `gate`: `[nh, 256]`. `ring`: `[2, TILE, 1, 4, 256]` FP8 VMEM; `sems`: DMA
    semaphores `[2]`. `pages` / `kv_len`: SMEM refs.
    """
    per_page = pool.shape[1] // MICRO
    nrows = pages.shape[0] if nrows is None else nrows
    nslots = ring.shape[0]

    def n_tiles(length: Any) -> Any:
        return (length + TILE - 1) // TILE

    length = kv_len[r]
    ntiles = n_tiles(length)
    rn = jnp.minimum(r + 1, nrows - 1)
    ntiles_next = jnp.where(r + 1 < nrows, n_tiles(kv_len[rn]), 0)
    # Cross-row prefetch only when this row covers the ring, so the next row's
    # pre-issued tiles are always its first ones, contiguous from 0.
    cross = ntiles >= nslots - 1
    nh, dim = q.shape

    ncols = pages.shape[1]
    assert per_page & (per_page - 1) == 0, per_page
    shift = per_page.bit_length() - 1

    def copies(row: Any, t: Any, s: Any) -> list[Any]:
        """Tile `t` of `row`: `TILE // MICRO` micro-page copies, issued unconditionally.

        No per-copy branch: a straight-line issue loop costs a few ns per DMA,
        a `pl.when` per copy several times that. Columns past `budget + 1` clamp
        to the last one; tokens past the row's length are masked below.
        """
        out = []
        for j in range(TILE // MICRO):
            v = pages[row, jnp.minimum(t * (TILE // MICRO) + j, ncols - 1)]
            out.append(
                pltpu.make_async_copy(
                    pool.at[v >> shift, pl.ds((v & (per_page - 1)) * MICRO, MICRO)],
                    ring.at[s, pl.ds(j * MICRO, MICRO)],
                    sems.at[s],
                )
            )
        return out

    def start(row: Any, t: Any, s: Any) -> None:
        if ablate == "dma":
            return
        for cp in copies(row, t, s):
            cp.start()

    def wait(t: Any, s: Any) -> None:
        if ablate == "dma":
            return
        if interpret:
            for cp in copies(r, t, s):
                cp.wait()
        else:
            # One wait for the whole slot: the copies above add up to its bytes.
            pltpu.make_async_copy(ring.at[s], ring.at[s], sems.at[s]).wait()

    def slot(t: Any) -> Any:
        return lax_rem(g0 + t, nslots)

    for k in range(nslots - 1):  # this row's first S-1 tiles not yet in flight

        @pl.when((k >= pre) & (k < ntiles))
        def _(k=k):
            start(r, k, slot(k))

    qf = q.astype(F32) * (scale_k * sm_scale if fold else 1.0)
    k_mul, v_mul, s_mul = (1.0, 1.0, 1.0) if fold else (scale_k, scale_v, sm_scale)

    def tile(t: Any, carry: Any) -> Any:
        m, l, acc = carry
        s = slot(t)
        wait(t, s)
        ahead = t + nslots - 1  # into the slot tile t - 1 has freed

        @pl.when(ahead < ntiles)
        def _():
            start(r, ahead, slot(ahead))

        k_next = ahead - ntiles  # the next row's tile k lands in slot(ahead)

        @pl.when(cross & (k_next >= 0) & (k_next < ntiles_next))
        def _():
            start(rn, k_next, slot(ahead))

        if ablate == "unpack":
            k8 = v8 = jnp.zeros((TILE, dim), BF16)
        else:
            k8, v8 = _unpack_kv(ring.at[s], off, interpret)
        tok = t * TILE + jax.lax.broadcasted_iota(I32, (1, TILE), 1)
        valid = tok < length
        key = k8.astype(F32) * k_mul if k_mul != 1.0 else k8.astype(F32)
        # A column iota, not `valid.reshape`: Mosaic has no i1 shape cast.
        tok_col = t * TILE + jax.lax.broadcasted_iota(I32, (TILE, 1), 0)
        vf = v8.astype(F32) * v_mul if v_mul != 1.0 else v8.astype(F32)
        value = jnp.where(tok_col < length, vf, 0.0)
        if ablate == "matmul":
            score = jnp.zeros((nh, TILE), F32)
        else:
            score = jax.lax.dot_general(
                qf, key, (((1,), (1,)), ((), ())), preferred_element_type=F32
            )
            if s_mul != 1.0:
                score = score * s_mul
        score = jnp.where(valid, score, -jnp.inf)
        m_new = jnp.maximum(m, jnp.max(score, axis=-1, keepdims=True))
        p = jnp.where(valid, jnp.exp(score - m_new), 0.0)
        alpha = jnp.exp(m - m_new)
        l_new = l * alpha + jnp.sum(p, axis=-1, keepdims=True)
        pv = (
            jnp.zeros((nh, dim), F32)
            if ablate == "matmul"
            else jnp.dot(p, value, preferred_element_type=F32)
        )
        acc_new = acc * alpha + pv
        return m_new, l_new, acc_new

    init = (
        jnp.full((nh, 1), -jnp.inf, F32),
        jnp.zeros((nh, 1), F32),
        jnp.zeros((nh, dim), F32),
    )
    _, l, acc = jax.lax.fori_loop(0, ntiles, tile, init)
    attn = acc / jnp.where(l > 0, l, 1.0) * (scale_v if fold else 1.0)
    gated = (
        attn.astype(BF16).astype(F32) * jax.nn.sigmoid(gate.astype(F32)).astype(BF16).astype(F32)
    ).astype(BF16).astype(F32)
    pre_next = jnp.where(cross, jnp.minimum(nslots - 1, ntiles_next), 0)
    return jnp.where(length > 0, gated, 0.0), g0 + ntiles, pre_next


def lax_rem(a: Any, b: int) -> Any:
    return jax.lax.rem(a, b)


class QSASparseAttnBlock(QSAOutBlock):
    """In-kernel QSA sparse attention + gate for the verify rows, then `o_proj`.

    Keeps `QSAOutBlock`'s name: it owns the same weights (`o_proj` and the
    wrapping hyper-connection's `qsa_out_*`), so packers need no change.
    """

    def __init__(
        self,
        *,
        budget: int = 512,
        kv_head_offset: int = 0,
        scale_k: float = 1.0,
        scale_v: float = 1.0,
        fold: bool = False,
        ablate: str = "",
        slots: int = SLOTS,
        prefix: str = "self_attn",
    ) -> None:
        super().__init__(prefix=prefix)
        self.fold, self.ablate, self.slots = fold, ablate, slots
        self.budget = budget
        self.off = kv_head_offset
        self.scale_k, self.scale_v = scale_k, scale_v

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        nh = g.NH // ex.R
        return [
            ExtraInput(POOL, "hbm"),
            ExtraInput(QUERY, "hbm", (ex.M, nh, g.HD), BF16),
            ExtraInput(GATE, "hbm", (ex.M, nh, g.HD), BF16),
            ExtraInput(PAGES, "smem", (ex.M, self.budget + 1), I32),
            ExtraInput(KV_LEN, "smem", (ex.M,), I32),
        ]

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        nh = g.NH // ex.R
        f8 = jnp.float8_e4m3fn
        return [
            ScratchReq("qa_ring", pltpu.VMEM((self.slots, TILE, 1, 4, 256), f8)),
            ScratchReq("qa_ring_sem", pltpu.SemaphoreType.DMA((self.slots,))),
            ScratchReq("qa_q", pltpu.VMEM((ex.MB, nh, g.HD), BF16)),
            ScratchReq("qa_g", pltpu.VMEM((ex.MB, nh, g.HD), BF16)),
            ScratchReq("qa_qg_sem", pltpu.SemaphoreType.DMA((2,))),
        ]

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        nh, K = g.NH // ex.R, self.k_dim(g, ex)
        qg = [
            pltpu.make_async_copy(
                ctx.extra[name].at[pl.ds(0, ex.MB)], scr[dst], scr["qa_qg_sem"].at[i]
            )
            for i, (name, dst) in enumerate(((QUERY, "qa_q"), (GATE, "qa_g")))
        ]
        for c in qg:
            c.start()
        for c in qg:
            c.wait()

        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)

        def row(r: Any, carry: Any) -> Any:
            *outs, g0, pre = carry
            out, g0, pre = attend_row(
                r,
                g0,
                pre,
                nrows=ex.MB,
                pool=ctx.extra[POOL],
                ring=scr["qa_ring"],
                sems=scr["qa_ring_sem"],
                pages=ctx.extra[PAGES],
                kv_len=ctx.extra[KV_LEN],
                q=scr["qa_q"][r],
                gate=scr["qa_g"][r],
                off=self.off,
                scale_k=self.scale_k,
                scale_v=self.scale_v,
                sm_scale=g.HD**-0.5,
                interpret=ex.interpret,
                fold=self.fold,
                ablate=self.ablate,
            )
            # Row r of each head's [MB, HD] output by a mask, not a store at a
            # dynamic row (a bf16 row store must be 16-row aligned in Mosaic).
            return (
                *(jnp.where(rows == r, out[h : h + 1], o) for h, o in enumerate(outs)),
                g0,
                pre,
            )

        zeros = tuple(jnp.zeros((ex.MB, g.HD), F32) for _ in range(nh))
        *outs, _, _ = jax.lax.fori_loop(0, ex.MB, row, (*zeros, jnp.int32(0), jnp.int32(0)))
        for h, o in enumerate(outs):
            scr["xb2"][:, pl.ds(h * g.HD, g.HD)] = o.astype(BF16)
        with ctx.stream.segment(lazy=False):
            return ctx.stream.dmatmul("o_proj", scr["xb2"], g.H)


def reference_rows(
    q: jax.Array,
    gate: jax.Array,
    pool: jax.Array,
    pages: jax.Array,
    kv_len: jax.Array,
    *,
    off: int,
    scale_k: float,
    scale_v: float,
    sm_scale: float,
) -> jax.Array:
    """Pure-JAX oracle, the stock `fused_kernel.decode_kernel` semantics per row.

    `q`, `gate`: `[N, nh, 256]`; `pool`: `[P, 128, 1, 4, 256]` FP8; returns `[N, nh, 256]`.
    """
    micro = pool.reshape(-1, MICRO, 4, pool.shape[-1])  # [P * 32, 4, 4, 256]
    outs = []
    for r in range(q.shape[0]):
        rows = micro[pages[r]].reshape(-1, 4, pool.shape[-1])
        key = rows[:, 2 * off].astype(F32) * scale_k
        value = rows[:, 2 * off + 1].astype(F32) * scale_v
        visible = jnp.arange(len(rows)) < kv_len[r]
        value = jnp.where(visible[:, None], value, 0.0)
        score = jnp.einsum("hd,td->ht", q[r].astype(F32), key) * sm_scale
        score = jnp.where(visible[None], score, -jnp.inf)
        prob = jax.nn.softmax(score, axis=-1)
        prob = jnp.where(jnp.isnan(prob), 0.0, prob)
        attn = prob @ value
        g = jax.nn.sigmoid(gate[r].astype(F32)).astype(BF16).astype(F32)
        gated = (attn.astype(BF16).astype(F32) * g).astype(BF16).astype(F32)
        outs.append(jnp.where(kv_len[r] > 0, gated, 0.0))
    return jnp.stack(outs)
