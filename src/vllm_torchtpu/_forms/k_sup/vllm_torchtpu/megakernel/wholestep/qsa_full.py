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
"""`QSAFullBlock`: a whole Qwen3.8 QSA decode layer inside the megakernel.

Wrapped by `HCSublayer(mix=True)`, so the attention sublayer's HC mix runs
in-kernel too and the block reads the mixed input from `scr['xb']`. Stage A
(this file) computes, per verify row:

  qkvg = mixed @ W_qkvg              one streamed projection, BF16 out
  q_h  = rope(gemma_norm(q_h))       per local head, partial neox RoPE (64)
  attn = sparse attention            `attend_row` over SMEM micro-page ids
  out  = bf16(bf16(attn) * bf16(sigmoid(gate_h)))  then `o_proj` (row-parallel)

`W_qkvg` columns are `[q_0 | gate_0 | q_1 | gate_1 | ... | k | v]` (each 256),
the stock `q_proj` per-head `[q | gate]` view followed by the rank's KV head.
K and V of the step's own tokens are computed, normed and roped here and
returned as the per-layer output `qsa_kv_new [M, 512]` (BF16 `[k | v]`) for
the post-kernel cache write. Stage A still takes the selection
(`qsa_pages`, `qsa_kvlen`) as SMEM inputs and reads the step's own tokens from
the pool; the in-kernel indexer replaces both (stage B, `indexer.py`).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.qsa_attn.block import SLOTS, TILE, attend_row
from ..exec_spec import ExecSpec
from ..ops import BF16, F32, I32, rnd, rmsnorm
from ..spec import Block, ExtraInput, ScratchReq, SmallField, StateDecl, WeightDecl

POOL = "qsa_pool"
PAGES = "qsa_pages"
KV_LEN = "qsa_kvlen"
OFFSET = "qsa_off"
SCALES = "qsa_scales"
ROPE = "qsa_rope"  # [M, 128] f32 per row: cos[0:32] | sin[32:64] | 0
KV_NEW = "qsa_kv_new"
QKVG = "qkvg"


def qkvg_width(nh: int, hd: int) -> int:
    return nh * 2 * hd + 2 * hd


def main_rope(x: jax.Array, cos: jax.Array, sin: jax.Array, rot: int) -> jax.Array:
    """Stock `main_rope`: neox on `[0, rot)`, one BF16 rounding per output."""
    half = rot // 2
    a, b = x[:, :half], x[:, half:rot]
    y1 = rnd(a * cos - b * sin)
    y2 = rnd(b * cos + a * sin)
    return jnp.concatenate((y1, y2, x[:, rot:]), axis=-1)


class QSAFullBlock(Block):
    """One QSA decode layer (projections, norms, RoPE, sparse attention, o_proj)."""

    name = "qsa_full"
    reduction = "allreduce"
    input_norm = None

    def __init__(self, *, budget: int = 512, slots: int = SLOTS, rot: int = 64) -> None:
        self.budget, self.slots, self.rot = budget, slots, rot

    @staticmethod
    def nh(g: Any, ex: ExecSpec) -> int:
        return g.NH // ex.R

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        nh = self.nh(g, ex)
        return [
            WeightDecl(QKVG, K=g.H, N=qkvg_width(nh, g.HD), dtype=BF16),
            WeightDecl("o_proj", K=nh * g.HD, N=g.H, dtype=BF16),
        ]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        return [SmallField("qsa_qn", g.HD), SmallField("qsa_kn", g.HD)]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return [StateDecl(KV_NEW, (ex.M, 2 * g.HD), BF16, "w")]

    def host_update_states(
        self, g: Any, ex: ExecSpec, pos: Any, states: dict[str, Any], raw_outputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Pass the step's K / V rows through for the post-kernel cache write."""
        return {KV_NEW: raw_outputs[KV_NEW]}

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        return [
            ExtraInput(POOL, "hbm"),
            ExtraInput(PAGES, "smem", (ex.M, self.budget + 1), I32),
            ExtraInput(KV_LEN, "smem", (ex.M,), I32),
            ExtraInput(OFFSET, "smem", (1,), I32),
            ExtraInput(SCALES, "smem", (2,), F32),
            ExtraInput(ROPE, "hbm", (ex.M, 128), F32),
        ]

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        # `acc` holds every dmatmul output: the fused projection is the widest here (5 * 128
        # for the indexer projection); other blocks (GDN) may need more, the planner takes the max.
        nh = self.nh(g, ex)
        return {"xb2": nh * g.HD, "acc": max(qkvg_width(nh, g.HD), 5 * 128)}

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        nh, f8 = self.nh(g, ex), jnp.float8_e4m3fn
        return [
            ScratchReq("qf_ring", pltpu.VMEM((self.slots, TILE, 1, 4, 256), f8)),
            ScratchReq("qf_ring_sem", pltpu.SemaphoreType.DMA((self.slots,))),
            # F32 (BF16-valued): rows are written by static single-row stores, which a packed
            # BF16 tile or a strided per-head DMA does not lower to (Mosaic tile alignment).
            ScratchReq("qf_q", pltpu.VMEM((ex.MB, nh, g.HD), F32)),
            ScratchReq("qf_g", pltpu.VMEM((ex.MB, nh, g.HD), F32)),
            ScratchReq("qf_kv", pltpu.VMEM((ex.MB, 2 * g.HD), BF16)),
            ScratchReq("qf_rope", pltpu.VMEM((ex.MB, 128), F32)),
            ScratchReq("qf_sem", pltpu.SemaphoreType.DMA((2,))),
        ]

    # ------------------------------------------------------------ in kernel
    def _project(self, ctx: Any, l: Any, slot: Any) -> Any:
        """Projection, norms, RoPE: q / gate into `qf_q` / `qf_g`; returns the in-flight
        copy of `[k | v]` to this layer's `qsa_kv_new` row (waited after attention)."""
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        nh, hd, rot = self.nh(g, ex), g.HD, self.rot
        rc = pltpu.make_async_copy(
            ctx.extra[ROPE].at[pl.ds(0, ex.MB)], scr["qf_rope"], scr["qf_sem"].at[0]
        )
        rc.start()
        with ctx.stream.segment(lazy=False):
            y = rnd(ctx.stream.dmatmul(QKVG, scr["xb"], qkvg_width(nh, hd)))
        rc.wait()
        cos = scr["qf_rope"][:, : rot // 2]
        sin = scr["qf_rope"][:, rot // 2 : rot]
        qn, kn = ctx.sm(slot, "qsa_qn"), ctx.sm(slot, "qsa_kn")
        for h in range(nh):
            q = rmsnorm(y[:, 2 * h * hd : (2 * h + 1) * hd], qn, g.eps, zero_centered=True)
            q = main_rope(q, cos, sin, rot)
            gt = y[:, (2 * h + 1) * hd : (2 * h + 2) * hd]
            for r in range(ex.MB):  # head-major -> [MB, nh, HD] by static single-row stores
                scr["qf_q"][r, h : h + 1, :] = q[r : r + 1]
                scr["qf_g"][r, h : h + 1, :] = gt[r : r + 1]
        k0 = 2 * nh * hd
        k = rmsnorm(y[:, k0 : k0 + hd], kn, g.eps, zero_centered=True)
        scr["qf_kv"][:, :hd] = main_rope(k, cos, sin, rot).astype(BF16)
        scr["qf_kv"][:, hd:] = y[:, k0 + hd : k0 + 2 * hd].astype(BF16)
        if "qf_kr" in scr:  # the un-normed key projection, for the stock cache writer
            scr["qf_kr"][...] = y[:, k0 : k0 + hd].astype(BF16)
        o = ctx.stream.sp.block_ord(l, self.name)
        kv = pltpu.make_async_copy(
            scr["qf_kv"], ctx.O[KV_NEW].at[o, pl.ds(0, ex.MB)], scr["qf_sem"].at[1]
        )
        kv.start()
        return kv

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex, scr, g = ctx.ex, ctx.scr, ctx.g
        nh = self.nh(g, ex)
        kv = self._project(ctx, l, slot)
        rows = jax.lax.broadcasted_iota(I32, (ex.MB, 1), 0)

        def row(r: Any, carry: Any) -> Any:
            *outs, g0, pre = carry
            out, g0, pre = attend_row(
                r,
                g0,
                pre,
                nrows=ex.MB,
                pool=ctx.extra[POOL],
                ring=scr["qf_ring"],
                sems=scr["qf_ring_sem"],
                pages=ctx.extra[PAGES],
                kv_len=ctx.extra[KV_LEN],
                q=scr["qf_q"][r],
                gate=scr["qf_g"][r],
                off=ctx.extra[OFFSET][0],
                scale_k=ctx.extra[SCALES][0],
                scale_v=ctx.extra[SCALES][1],
                sm_scale=g.HD**-0.5,
                interpret=ex.interpret,
            )
            return (
                *(jnp.where(rows == r, out[h : h + 1], o) for h, o in enumerate(outs)),
                g0,
                pre,
            )

        zeros = tuple(jnp.zeros((ex.MB, g.HD), F32) for _ in range(nh))
        *outs, _, _ = jax.lax.fori_loop(0, ex.MB, row, (*zeros, jnp.int32(0), jnp.int32(0)))
        for h, o in enumerate(outs):
            scr["xb2"][:, pl.ds(h * g.HD, g.HD)] = o.astype(BF16)
        kv.wait()
        with ctx.stream.segment(lazy=False):
            return ctx.stream.dmatmul("o_proj", scr["xb2"], g.H)


# ---------------------------------------------------------------- reference
def project_reference(
    mixed: jax.Array, w_qkvg: jax.Array, qn: jax.Array, kn: jax.Array, rope: jax.Array,
    *, nh: int, hd: int, rot: int, eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-rank `(q [T, nh, hd], gate [T, nh, hd], k [T, hd], v [T, hd])`, BF16."""
    y = rnd(jnp.dot(mixed.astype(BF16), w_qkvg.astype(BF16), preferred_element_type=F32))
    cos, sin = rope[:, : rot // 2], rope[:, rot // 2 : rot]
    qs, gs = [], []
    for h in range(nh):
        q = rmsnorm(y[:, 2 * h * hd : (2 * h + 1) * hd], qn, eps, zero_centered=True)
        qs.append(main_rope(q, cos, sin, rot))
        gs.append(y[:, (2 * h + 1) * hd : (2 * h + 2) * hd])
    k0 = 2 * nh * hd
    k = main_rope(rmsnorm(y[:, k0 : k0 + hd], kn, eps, zero_centered=True), cos, sin, rot)
    v = y[:, k0 + hd : k0 + 2 * hd]
    st = lambda xs: jnp.stack(xs, axis=1).astype(BF16)  # noqa: E731
    return st(qs), st(gs), k.astype(BF16), v.astype(BF16)
