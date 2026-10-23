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
per-rank pool `[blocks * 4, 128, 1, 4, 256]` FP8 bytes): manager block `m`
holds the f32 recurrent state `[vh, KD, VD]` in kernel blocks `4m .. 4m+2`
and the f32 conv history `[3, QKV]` in the first rows of kernel block `4m+3`.
`gdn_rb [NGDN, NS]` (SMEM) names each request's read block and `gdn_hi`
whether it has a prior state (0 = start from zero). Bytes are reinterpreted
with the same typed view as the model's own GDN kernel (`ref.bitcast` to f32,
row-major). Checkpoints are still returned as `conv_ckpt` / `rec_ckpt`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..exec_spec import ExecSpec
from ..ops import BF16, F32, I32, rup
from ..spec import ExtraInput, ScratchReq, StateDecl
from .qwen38 import Qwen38GDN

POOL = "gdn_pool"
READ_BLOCKS = "gdn_rb"
HAS_STATE = "gdn_hi"
SPLIT = 4  # kernel blocks per manager block
SSM_BLOCKS = 3  # kernel blocks holding the recurrent state
CONV_ROWS = 16  # pool rows holding the conv history (rounded to a power of two)


class _LayerView:
    """Stands in for a per-layer state array: `.at[o, *rest]` -> `ref.at[*rest]`."""

    def __init__(self, ref: Any) -> None:
        self.ref = ref
        self.at = self

    def __getitem__(self, idx: Any) -> Any:
        rest = idx[1:] if isinstance(idx, tuple) else ()
        return self.ref.at[rest] if rest else self.ref


def _typed_f32(block_ref: Any) -> Any:
    """`[rows, 1, 4, lanes]` FP8 bytes -> `(rows, lanes)` f32 (4 pack rows per word).

    The array bitcast groups 4 consecutive second-minor rows into one 32-bit
    word per lane, as the model's typed ref view does.
    """
    raw = block_ref[...]
    raw = raw.reshape(-1, raw.shape[-1])
    return pltpu.bitcast(raw, jnp.float32)


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
        scr.update(
            pg_raw=pltpu.VMEM((2, SSM_BLOCKS, 128, 1, 4, 256), f8),
            pg_craw=pltpu.VMEM((2, CONV_ROWS, 1, 4, 256), f8),
            pg_sem=pltpu.SemaphoreType.DMA((2, 2)),
            pg_rec=pltpu.VMEM((ex.NS * lay.vh, lay.KD, lay.VD), BF16),
            pg_conv=pltpu.VMEM((ex.NS, rup(3 + ex.T, 8), lay.QKV), F32),
        )
        return scr

    def _copies(self, ctx: Any, o: Any, s: int, buf: int) -> tuple[Any, Any]:
        pool, scr = ctx.extra[POOL], ctx.scr
        m = ctx.extra[READ_BLOCKS][o, s]
        ssm = pltpu.make_async_copy(
            pool.at[pl.ds(m * SPLIT, SSM_BLOCKS)], scr["pg_raw"].at[buf], scr["pg_sem"].at[buf, 0]
        )
        conv = pltpu.make_async_copy(
            pool.at[m * SPLIT + SSM_BLOCKS, pl.ds(0, CONV_ROWS)],
            scr["pg_craw"].at[buf],
            scr["pg_sem"].at[buf, 1],
        )
        return ssm, conv

    def _gather(self, ctx: Any, o: Any) -> None:
        """Layer `o`'s pool states -> `pg_rec` (bf16) / `pg_conv` (f32 history rows)."""
        ex, scr = ctx.ex, ctx.scr
        lay = self.lin_in_layout(ctx.g, ex)
        vh, KD, VD, QKV = lay.vh, lay.KD, lay.VD, lay.QKV
        NS = ex.NS
        scr["pg_conv"][...] = jnp.zeros(scr["pg_conv"].shape, F32)
        pending = self._copies(ctx, o, 0, 0)
        for c in pending:
            c.start()
        for s in range(NS):
            buf = s % 2
            if s + 1 < NS:
                nxt = self._copies(ctx, o, s + 1, 1 - buf)
                for c in nxt:
                    c.start()
            for c in pending:
                c.wait()
            keep = ctx.extra[HAS_STATE][o, s] != 0
            ssm = jnp.concatenate(
                [_typed_f32(scr["pg_raw"].at[buf, j]) for j in range(SSM_BLOCKS)], axis=0
            ).reshape(vh, KD, VD)
            scr["pg_rec"][s * vh : (s + 1) * vh] = jnp.where(keep, ssm, 0.0).astype(BF16)
            conv = _typed_f32(scr["pg_craw"].at[buf]).reshape(-1)[: 3 * QKV].reshape(3, QKV)
            scr["pg_conv"][s, 0:3, :] = jnp.where(keep, conv, 0.0)
            if s + 1 < NS:
                pending = nxt

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        o = ctx.stream.sp.block_ord(l, self.name)
        with ctx.sc("gdn_pool_gather"):
            self._gather(ctx, o)
        st = {**ctx.ST, "conv": _LayerView(ctx.scr["pg_conv"]), "rec": _LayerView(ctx.scr["pg_rec"])}
        return super().emit(dataclasses.replace(ctx, ST=st), l, slot)

    def host_update_states(self, g, ex, pos, states, raw_outputs):
        return {"conv_ckpt": raw_outputs["conv_ckpt"], "rec_ckpt": raw_outputs["rec_ckpt"]}
