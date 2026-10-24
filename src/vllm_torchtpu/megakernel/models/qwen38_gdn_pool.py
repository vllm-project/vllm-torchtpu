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

The base `GDNModule` takes per-layer `conv` / `rec` state arrays. Here the
kernel reads them itself from the model's pool (extra input `gdn_pool`, one
per-rank pool `[blocks * split, kb, 1, 4, 256]` FP8 bytes, `kb * split = 512`):
manager block `m` holds the f32 recurrent state `[vh, KD, VD]` in its rows
`0 .. 383` and the f32 conv history `[3, QKV]` in rows `384 .. 399`, whatever
the kernel block size `kb` (128 or 512 in served configs).
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
from ..spec import ExtraInput, StateDecl
from .qwen38 import Qwen38GDN

POOL = "gdn_pool"
READ_BLOCKS = "gdn_rb"
HAS_STATE = "gdn_hi"
MANAGER_ROWS = 512  # pool rows (tokens) per manager block
SSM_ROWS = 128  # pool rows per staged recurrent-state chunk
SSM_BLOCKS = 3  # chunks holding the recurrent state (rows 0 .. 383)
CONV_ROWS = 16  # pool rows holding the conv history (rounded to a power of two)


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
            pg_rec=pltpu.VMEM((ex.NS * lay.vh, lay.KD, lay.VD), BF16),
            pg_conv=pltpu.VMEM((ex.NS, rup(3 + ex.T, 8), lay.QKV), F32),
        )
        return scr

    def _copies(self, ctx: Any, o: Any, s: int, buf: int) -> list[Any]:
        pool, scr = ctx.extra[POOL], ctx.scr
        m = ctx.extra[READ_BLOCKS][o, s]
        cps = [
            pltpu.make_async_copy(
                pool_rows(pool, m, j * SSM_ROWS, SSM_ROWS),
                scr[f"pg_raw{buf}{j}"],
                scr["pg_sem"].at[buf, j],
            )
            for j in range(SSM_BLOCKS)
        ]
        cps.append(
            pltpu.make_async_copy(
                pool_rows(pool, m, SSM_BLOCKS * SSM_ROWS, CONV_ROWS),
                scr[f"pg_craw{buf}"],
                scr["pg_sem"].at[buf, SSM_BLOCKS],
            )
        )
        return cps

    def _gather(self, ctx: Any, o: Any) -> None:
        """Layer `o`'s pool states -> `pg_rec` (bf16) / `pg_conv` (f32 history rows)."""
        ex, scr = ctx.ex, ctx.scr
        lay = self.lin_in_layout(ctx.g, ex)
        vh, KD, VD, QKV = lay.vh, lay.KD, lay.VD, lay.QKV
        NS = ex.NS
        interp = bool(ex.interpret)
        scr["pg_conv"][...] = jnp.zeros(scr["pg_conv"].shape, F32)
        pending = self._copies(ctx, o, 0, 0)
        for c in pending:
            c.start()
        for s in range(NS):
            buf = s % 2
            nxt = self._copies(ctx, o, s + 1, 1 - buf) if s + 1 < NS else pending
            if s + 1 < NS:
                for c in nxt:
                    c.start()
            for c in pending:
                c.wait()
            keep = ctx.extra[HAS_STATE][o, s] != 0
            # 128-lane rows: the state's row-major order, so only the leading
            # dimension is split (lane-changing reshapes do not lower).
            ssm = jnp.concatenate(
                [
                    _typed_f32(scr[f"pg_raw{buf}{j}"], interp, VD)
                    for j in range(SSM_BLOCKS)
                ],
                axis=0,
            )[: vh * KD].reshape(vh, KD, VD)
            scr["pg_rec"][s * vh : (s + 1) * vh] = jnp.where(keep, ssm, 0.0).astype(
                BF16
            )
            crow = _typed_f32(scr[f"pg_craw{buf}"], interp, 256)  # [16, 256]
            per = QKV // 256
            for r in range(3):
                for c in range(per):
                    scr["pg_conv"][s, r : r + 1, c * 256 : (c + 1) * 256] = jnp.where(
                        keep, crow[r * per + c : r * per + c + 1], 0.0
                    )
            pending = nxt

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        o = ctx.stream.sp.module_ord(l, self.name)
        with ctx.sc("gdn_pool_gather"):
            self._gather(ctx, o)
        st = {
            **ctx.ST,
            "conv": _LayerView(ctx.scr["pg_conv"]),
            "rec": _LayerView(ctx.scr["pg_rec"]),
        }
        return super().emit(dataclasses.replace(ctx, ST=st), l, slot)

    def host_update_states(self, g, ex, pos, states, raw_outputs):
        return {
            "conv_ckpt": raw_outputs["conv_ckpt"],
            "rec_ckpt": raw_outputs["rec_ckpt"],
        }
