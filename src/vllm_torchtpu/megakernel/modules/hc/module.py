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
"""`HCSublayer`: a hyper-connection around an inner sublayer block.

The model's residual is `NHC` streams of width `h`; the core residual buffer
`scr['res']` holds all of them (`g.H == NHC * h`). Per layer slot:

  mix      xn = per-stream RMSNorm(res); g = silu(xn @ down / NHC);
           mixed = sum_b sigmoid(g @ up)_b * xn_b / NHC              -> [MB, h]
  inner    partial = inner.emit(mixed)                               -> [MB, h]
  reduce   block = all-reduce(partial) over the `[MB, h]` hc ring, on its
           static double-buffer `ar_slot` (one per layer slot, as in the core)
  combine  res_b = rnd(res_b + rnd(block) * 2 sigmoid(xn @ inj / NHC)_b)

`down` and `inj` share one streamed weight (`<name>_hdn`, columns
`[down | inj | pad]`); `up` is one weight per stream (`<name>_hup<b>`) so a
128-row tile fits the weight bank. With `mix=False` the sublayer input comes
from the inner block alone and `xn` is recomputed for the injection
(`<name>_hinj`).

The inner block sees a context whose geometry is `inner_geometry` (width `h`)
and whose `xb` / `part` buffers are `h` wide; everything else (stream, smalls,
states) is the kernel's. The wrapper takes the inner block's name, so the
inner block's plan params, layer ordinals and tunables resolve unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...exec_spec import ExecSpec
from ...ops import BF16, F32, rmsnorm, rnd, rup
from ...spec import Module, ScratchReq, SmallField, StateDecl, WeightDecl
from .comm import hc_comm_scratch, hc_drain, hc_ring

RESIDUAL_OUT = "hc_res"


def _as_reqs(scr: Any) -> list[ScratchReq]:
    if isinstance(scr, dict):
        return [ScratchReq(k, v) for k, v in scr.items()]
    return list(scr or [])


class HCSublayer(Module):
    """Hyper-connection wrapper: mix -> inner sublayer -> all-reduce -> combine."""

    name = "hc_sublayer"
    reduction = "custom"
    # The inner block's `part` is remapped to `hc_part` (see `emit`); the core
    # `part` buffer is never read through this block.
    owns_part = True
    input_norm = None

    def __init__(
        self,
        inner: Module,
        *,
        inner_geometry: Any,
        nhc: int,
        lowrank: int,
        mix: bool = True,
        allreduce: bool = True,
        emit_residual: bool = False,
        ar_slot: int = 0,
    ) -> None:
        self.inner = inner
        self.gi = inner_geometry
        self.name = inner.name
        self.nhc = int(nhc)
        self.lowrank = int(lowrank)
        self.mix = bool(mix)
        self.allreduce = bool(allreduce)
        self.emit_residual = bool(emit_residual)
        self.ar_slot = int(ar_slot)
        self.prefix = getattr(inner, "prefix", "")

    # ------------------------------------------------------------ geometry
    @property
    def h(self) -> int:
        return int(self.gi.H)

    @property
    def width(self) -> int:
        return self.nhc * self.h

    @property
    def n_down(self) -> int:
        """Columns of `<name>_hdn`: low-rank gate, then the injection, padded."""
        return rup(self.lowrank + self.nhc, 128)

    @property
    def k_up(self) -> int:
        return rup(self.lowrank, 128)

    def _check(self, g: Any) -> None:
        assert self.width == g.H, (g.H, self.nhc, self.h)

    # ------------------------------------------------------- declarations
    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        self._check(g)
        n = self.name
        if self.mix:
            own = [
                WeightDecl(f"{n}_hdn", K=self.width, N=self.n_down, dtype=BF16),
            ] + [
                WeightDecl(f"{n}_hup{b}", K=self.k_up, N=self.h, dtype=BF16)
                for b in range(self.nhc)
            ]
        else:
            own = [WeightDecl(f"{n}_hinj", K=self.width, N=128, dtype=BF16)]
        return own + list(self.inner.weights(self.gi, ex))

    def states(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return list(self.inner.states(self.gi, ex))

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        out = list(self.inner.outputs(self.gi, ex))
        if self.emit_residual:
            out.append(
                StateDecl(RESIDUAL_OUT, (ex.M, self.width), BF16, "w", per_layer=False)
            )
        return out

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[Any]:
        return list(self.inner.extra_inputs(self.gi, ex))

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        return [SmallField(f"{self.name}_hcn", self.width)] + list(
            self.inner.smalls(self.gi, ex)
        )

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        sw = dict(self.inner.shared_widths(self.gi, ex))
        sw["acc"] = max(sw.get("acc", 0), self.h, self.n_down)
        return sw

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> Any:
        return self.inner.plan_params(self.gi, ex, plan)

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        reqs = _as_reqs(self.inner.scratch(self.gi, ex, plan))
        reqs += [
            ScratchReq("hc_xb", pltpu.VMEM((ex.MB, self.h), BF16)),
            ScratchReq("hc_part", pltpu.VMEM((ex.MB, self.h), F32)),
            ScratchReq("hc_g", pltpu.VMEM((ex.MB, self.k_up), BF16)),
            ScratchReq("hc_inj", pltpu.VMEM((ex.MB, 128), F32)),
        ]
        if self.emit_residual:
            reqs.append(ScratchReq("hc_res_sem", pltpu.SemaphoreType.DMA(())))
        if self.allreduce:
            reqs += hc_comm_scratch(ex, self.h)
        return reqs

    # ---------------------------------------------------------- in kernel
    def _inner_ctx(self, ctx: Any, norm_x: Any = None) -> Any:
        scr = dict(ctx.scr)
        scr["xb"], scr["part"] = ctx.scr["hc_xb"], ctx.scr["hc_part"]
        return dataclasses.replace(ctx, g=self.gi, scr=scr, norm_x=norm_x)

    def prologue(self, ctx: Any) -> None:
        self.inner.prologue(self._inner_ctx(ctx))

    def init_buffers(self, ctx: Any) -> None:
        ctx.scr["hc_g"][...] = jnp.zeros(ctx.scr["hc_g"].shape, BF16)
        self.inner.init_buffers(self._inner_ctx(ctx))

    def on_layer_start(self, ctx: Any, lyr: Any, slot: Any) -> None:
        self.inner.on_layer_start(self._inner_ctx(ctx), lyr, slot)

    def _normalize(self, ctx: Any, slot: Any) -> None:
        """`scr['xb'] = per-stream RMSNorm(res)` (BF16)."""
        h, eps = self.h, ctx.g.eps
        res = ctx.scr["res"]
        w = ctx.sm(slot, f"{self.name}_hcn")
        for b in range(self.nhc):
            sl = slice(b * h, (b + 1) * h)
            ctx.scr["xb"][:, sl] = rmsnorm(
                res[:, sl], w[:, sl], eps, zero_centered=True
            ).astype(BF16)

    def _set_injection(self, ctx: Any, y: Any) -> None:
        inj = 2.0 * jax.nn.sigmoid(rnd(y[:, : self.nhc]) / self.nhc)
        ctx.scr["hc_inj"][:, : self.nhc] = inj

    def _mix(self, ctx: Any, slot: Any) -> Any:
        """HC mix into `scr['hc_xb']`; returns the mixed input as f32."""
        n, nhc, lr, h = self.name, self.nhc, self.lowrank, self.h
        s, scr = ctx.stream, ctx.scr
        with ctx.sc("hc_mix"):
            self._normalize(ctx, slot)
            y = s.dmatmul(f"{n}_hdn", scr["xb"], self.n_down)
            gate = rnd(jax.nn.silu(rnd(rnd(y[:, :lr]) / nhc)))
            scr["hc_g"][:, :lr] = gate.astype(BF16)
            self._set_injection(ctx, y[:, lr : lr + nhc])
            mixed: Any = None
            for b in range(nhc):
                u = s.dmatmul(f"{n}_hup{b}", scr["hc_g"], h)
                xn_b = scr["xb"][:, b * h : (b + 1) * h].astype(F32)
                term = jax.nn.sigmoid(rnd(u)) * xn_b
                mixed = term if mixed is None else mixed + term
            mixed = rnd(mixed / nhc)
            scr["hc_xb"][...] = mixed.astype(BF16)
        return mixed

    def _combine(self, ctx: Any, block: Any) -> None:
        h, B, res, inj = self.h, ctx.ex.B, ctx.scr["res"], ctx.scr["hc_inj"]
        with ctx.sc("hc_combine"):
            blk = rnd(block[0:B])
            for b in range(self.nhc):
                sl = slice(b * h, (b + 1) * h)
                res[0:B, sl] = rnd(res[0:B, sl] + blk * inj[0:B, b : b + 1])

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        s = ctx.stream
        with s.segment(lazy=False):
            if self.mix:
                mixed = self._mix(ctx, slot)
            else:
                with ctx.sc("hc_inj"):
                    self._normalize(ctx, slot)
                    y = s.dmatmul(f"{self.name}_hinj", ctx.scr["xb"], 128)
                    self._set_injection(ctx, y)
                mixed = None
        part = self.inner.emit(self._inner_ctx(ctx, mixed), l, slot)
        if part is None:
            return None
        if self.allreduce:
            block = hc_ring(ctx, self.h).allreduce(self.ar_slot, part, l == 0)
        else:
            block = part
        self._combine(ctx, block)
        return None

    def epilogue(self, ctx: Any) -> None:
        self.inner.epilogue(self._inner_ctx(ctx))
        hc_drain(ctx)
        if self.emit_residual:
            MB = ctx.ex.MB
            cp = pltpu.make_async_copy(
                _res_bf16(ctx),
                ctx.O[RESIDUAL_OUT].at[pl.ds(0, MB)],
                ctx.scr["hc_res_sem"],
            )
            cp.start()
            cp.wait()

    def host_update_states(
        self,
        g: Any,
        ex: ExecSpec,
        pos: Any,
        states: dict[str, Any],
        raw_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        out = dict(self.inner.host_update_states(self.gi, ex, pos, states, raw_outputs))
        if self.emit_residual:
            out[RESIDUAL_OUT] = raw_outputs[RESIDUAL_OUT]
        return out


def _res_bf16(ctx: Any) -> Any:
    """Stage the residual as BF16 in `scr['xb']` (free after the last layer)."""
    ctx.scr["xb"][...] = ctx.scr["res"][...].astype(BF16)
    return ctx.scr["xb"]
