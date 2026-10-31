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
"""MoE block (`MoEBlock`): router, EP/TP dedup, shared and piped routed experts."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...decls import (
    BlockBinding,
    FrozenDict,
    HFParamDecl,
    ResolvedBlockGeometry,
)
from ...errors import MegakernelConfigError
from ...exec_spec import ExecSpec
from ...ops import BF16, F32, FP8, I32, rnd, rup
from ...spec import Block, ExtraInput, SmallField, WeightDecl
from .context import MoEEmit
from .decls import (
    MOE_FIELD_ALIASES,
    MOE_GEOMETRY_FIELDS,
    MOE_GEOMETRY_METHODS,
    MOE_OPTIONS,
    MOE_PARAM_NAMES,
    moe_hf_params,
    resolve_moe_binding,
    validate_moe_options,
)
from .experts import expert, experts_piped, experts_piped_ep
from .params import MoEDebug, MoEParams, MoETunables, derive_moe_params
from .reference import moe_reference
from .routing import route


class MoEBlock(Block):
    """Sparse + shared MoE block with dynamic `epool` streaming and `PIPE`."""

    name: str = "moe"
    reduction = "allreduce"
    input_norm: str | None = "post_norm"
    Tunables = MoETunables
    Debug = MoEDebug
    PARAM_NAMES = MOE_PARAM_NAMES
    GEOMETRY_FIELDS = MOE_GEOMETRY_FIELDS
    FIELD_ALIASES = MOE_FIELD_ALIASES
    OPTIONS = MOE_OPTIONS
    GEOMETRY_METHODS = MOE_GEOMETRY_METHODS

    def __init__(
        self,
        *,
        geometry: Any = None,
        prefix: str = "mlp",
        shared_gate: bool = True,
        moe_tp: bool | None = None,
        e_slots: int | None = None,
    ) -> None:
        self.geometry = geometry
        self.prefix = prefix
        self.shared_gate = bool(shared_gate)
        self._explicit_moe_tp = None if moe_tp is None else bool(moe_tp)
        self._explicit_e_slots = None if e_slots is None else int(e_slots)
        self.moe_tp = (
            self._explicit_moe_tp
            if self._explicit_moe_tp is not None
            else bool(getattr(geometry, "moe_tp", True))
        )
        self.e_slots = (
            self._explicit_e_slots
            if self._explicit_e_slots is not None
            else int(getattr(geometry, "e_slots", 0))
        )
        self._validate_options({"shared_gate": self.shared_gate})
        if geometry is not None:
            self._check_option_consistency(geometry)
        self.options = FrozenDict(
            {
                "shared_gate": self.shared_gate,
                "moe_tp": self.moe_tp,
                "e_slots": self.e_slots,
            }
        )

    def _check_option_consistency(self, g: Any) -> tuple[bool, int]:
        g_tp = getattr(g, "moe_tp", None)
        if (
            self._explicit_moe_tp is not None
            and g_tp is not None
            and bool(g_tp) != self._explicit_moe_tp
        ):
            raise MegakernelConfigError(
                f"MoEBlock(moe_tp={self._explicit_moe_tp}) disagrees with "
                f"geometry moe_tp={g_tp}"
            )
        g_slots = getattr(g, "e_slots", None)
        if (
            self._explicit_e_slots is not None
            and g_slots is not None
            and int(g_slots) != self._explicit_e_slots
        ):
            raise MegakernelConfigError(
                f"MoEBlock(e_slots={self._explicit_e_slots}) disagrees with "
                f"geometry e_slots={g_slots}"
            )
        moe_tp = (
            self._explicit_moe_tp
            if self._explicit_moe_tp is not None
            else (bool(g_tp) if g_tp is not None else self.moe_tp)
        )
        e_slots = (
            self._explicit_e_slots
            if self._explicit_e_slots is not None
            else (int(g_slots) if g_slots is not None else self.e_slots)
        )
        return moe_tp, e_slots

    @classmethod
    def _validate_options(cls, options: Mapping[str, Any]) -> None:
        validate_moe_options(options)

    @classmethod
    def _resolve_binding(
        cls, binding: BlockBinding, hf_config: Any
    ) -> ResolvedBlockGeometry:
        return resolve_moe_binding(
            cls, binding, super()._resolve_binding(binding, hf_config)
        )

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        """Debug routing inputs: `rbias` (HBM, `forced_routing`) and `moe_rows`
        (SMEM `[B]` int32 row mask, `row_mask`)."""
        dbg: MoEDebug = self.debug_flags(ex)
        out = []
        if dbg.forced_routing:
            out.append(ExtraInput("rbias", "hbm"))
        if dbg.row_mask:
            out.append(ExtraInput("moe_rows", "smem", (ex.B,), I32))
        return out

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        moe_tp, e_slots = self._check_option_consistency(g)
        er = g.E // ex.R
        isr = g.IS // ex.R
        ir = g.I // ex.R if moe_tp else g.I
        ew = e_slots if e_slots > 0 else (g.E if moe_tp else er)
        layout_e = "stacked" if ex.stack_e else "per_layer"
        return [
            WeightDecl("router", K=g.H, N=g.E, dtype=BF16, stream_ring="static"),
            WeightDecl("sh13", K=g.H, N=2 * isr, dtype=FP8, stream_ring="static"),
            WeightDecl("sh2", K=isr, N=g.H, dtype=FP8, stream_ring="static"),
            WeightDecl(
                "e13",
                K=g.H,
                N=2 * ir,
                dtype=FP8,
                stream_ring="dynamic",
                layout=layout_e,
                lead_dims=(ew,),
            ),
            WeightDecl(
                "e2",
                K=ir,
                N=g.H,
                dtype=FP8,
                stream_ring="dynamic",
                layout=layout_e,
                lead_dims=(ew,),
            ),
            WeightDecl(
                "es",
                K=1,
                N=2 * ir + g.H,
                dtype=F32,
                stream_ring="hbm",
                lead_dims=(ew,),
            ),
        ]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        moe_tp, e_slots = self._check_option_consistency(g)
        er = g.E // ex.R
        isr = g.IS // ex.R
        fields = [
            SmallField("post_norm", g.H),
            SmallField("sh13_s", 2 * isr),
            SmallField("sh2_s", g.H),
            SmallField("sh_gate", g.H),
        ]
        if not moe_tp and e_slots > er:
            fields.append(SmallField("ecopy", g.E))
        return fields

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        moe_tp, _ = self._check_option_consistency(g)
        isr = g.IS // ex.R
        ir = g.I // ex.R if moe_tp else g.I
        return {"acc": max(g.H, g.E, 2 * isr, 2 * ir), "xb2": max(isr, ir)}

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> MoEParams:
        moe_tp, e_slots = self._check_option_consistency(g)
        return derive_moe_params(
            g,
            ex,
            plan,
            self.tunables(ex),
            self.debug_flags(ex),
            moe_tp=moe_tp,
            e_slots=e_slots,
        )

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        mp: MoEParams = plan.block_params[self.name]
        nb_picks = rup(ex.B * mp.TOPK, 128)
        scr: dict[str, Any] = {
            "wn": pltpu.VMEM((ex.MR, mp.E), F32),
            "sg": pltpu.VMEM((ex.MB, 128), F32),
            "hitv": pltpu.VMEM((ex.MR, 128), I32),
            "hits": pltpu.SMEM((ex.MR, 128), I32),
            "elist": pltpu.SMEM((nb_picks,), I32),
            "seen": pltpu.SMEM((8, 128), I32),
            "misc_sem": pltpu.SemaphoreType.DMA(()),
            "es": pltpu.VMEM((mp.NSS, 1, 2 * mp.IR + mp.H), F32),
            "es_sem": pltpu.SemaphoreType.DMA((mp.NSS,)),
            "st": pltpu.SMEM((1,), I32),
        }
        if mp.PIPE_EP:
            scr["acc13"] = pltpu.VMEM((ex.MB, 2 * mp.IR), F32)
        if mp.ECOPY:
            scr.update(
                {
                    "ecopy": pltpu.SMEM((2, 1, mp.E), F32),
                    "ec_sem": pltpu.SemaphoreType.DMA((2,)),
                    "wlist": pltpu.SMEM((nb_picks,), I32),
                    "cplist": pltpu.SMEM((nb_picks,), I32),
                    "cpchoice": pltpu.SMEM((nb_picks,), I32),
                    "rload": pltpu.SMEM((ex.R,), I32),
                }
            )
            if mp.ecopy_opt == 1:
                scr["ulist"] = pltpu.SMEM((nb_picks,), I32)
        if self.debug_flags(ex).forced_routing:
            scr["rb"] = pltpu.VMEM((2, ex.M, mp.E), F32)
            scr["rb_sem"] = pltpu.SemaphoreType.DMA((2,))
        return scr

    def prologue(self, ctx: Any) -> None:
        """Start async zeroing of `seen` and prefetch layer-0 `ecopy`/`rb` if active."""
        ex = ctx.ex
        mp: MoEParams = ctx.block_params(self.name)
        scr, W = ctx.scr, ctx.W
        scr["hitv"][...] = jnp.zeros((ex.MR, 128), I32)
        pltpu.make_async_copy(
            scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"]
        ).start()
        if mp.ECOPY:
            o_ec = ctx.stream.sp.lay["ecopy"][0]
            pltpu.make_async_copy(
                W["smalls"].at[0, :, pl.ds(o_ec, mp.E)],
                scr["ecopy"].at[0],
                scr["ec_sem"].at[0],
            ).start()
        if self.debug_flags(ex).forced_routing:
            pltpu.make_async_copy(
                ctx.extra["rbias"].at[0], scr["rb"].at[0], scr["rb_sem"].at[0]
            ).start()

    def init_buffers(self, ctx: Any) -> None:
        """Wait for initial `seen` zeroing DMA to complete."""
        scr = ctx.scr
        pltpu.make_async_copy(
            scr["hitv"].at[pl.ds(0, 8)], scr["seen"], scr["misc_sem"]
        ).wait()

    def on_layer_start(self, ctx: Any, lyr: Any, slot: Any) -> None:
        """Double-buffer `ecopy` SMEM table alongside `smalls`."""
        mp: MoEParams = ctx.block_params(self.name)
        if not mp.ECOPY:
            return
        scr, W = ctx.scr, ctx.W
        o_ec = ctx.stream.sp.lay["ecopy"][0]
        pltpu.make_async_copy(
            W["smalls"].at[0, :, pl.ds(o_ec, mp.E)],
            scr["ecopy"].at[slot],
            scr["ec_sem"].at[slot],
        ).wait()

        @pl.when(lyr + 1 < ctx.stream.sp.L)
        def _():
            pltpu.make_async_copy(
                W["smalls"].at[lyr + 1, :, pl.ds(o_ec, mp.E)],
                scr["ecopy"].at[1 - slot],
                scr["ec_sem"].at[1 - slot],
            ).start()

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex = ctx.ex
        mp: MoEParams = ctx.block_params(self.name)
        m = MoEEmit.build(ctx, mp, self.debug_flags(ex))
        stream, scr, sc = m.stream, m.scr, m.sc
        MR, MB = ex.MR, ex.MB
        H, E, ISR = mp.H, mp.E, mp.ISR
        PIPE, PIPE_EP, LAZY_MOE = mp.PIPE, mp.PIPE_EP, mp.LAZY_MOE
        with stream.expert_stream(), stream.segment(lazy=LAZY_MOE):
            with sc("moe_router"):
                assert ctx.norm_x is not None, (
                    "MoEBlock requires input_norm='post_norm' to populate ctx.norm_x"
                )
                x = ctx.norm_x
                logits = stream.dmatmul("router", scr["xb"], E)[0:MR]
                glog = jnp.sum(x * ctx.sm(slot, "sh_gate"), axis=-1, keepdims=True)
                scr["sg"][...] = jnp.broadcast_to(jax.nn.sigmoid(glog), (MB, 128))
                mx = jnp.max(logits, axis=-1, keepdims=True)
                pexp = jnp.exp(logits - mx)
            with sc("moe_route"):
                n_hit = route(m, l, pexp)
            with sc("moe_shared"):
                g13 = stream.dmatmul("sh13", scr["xb"], 2 * ISR)
                g13 = g13 * ctx.sm(slot, "sh13_s")
                mid = rnd(rnd(jax.nn.silu(rnd(g13[:, :ISR]))) * rnd(g13[:, ISR:]))
                scr["xb2"][:, :ISR] = mid.astype(BF16)
                ysh = stream.dmatmul("sh2", scr["xb2"], H)
                scr["part"][...] = ysh * ctx.sm(slot, "sh2_s") * scr["sg"][:, 0:1]
            with sc("moe_experts"):
                if PIPE:
                    experts_piped(m, l, n_hit)
                elif PIPE_EP:
                    experts_piped_ep(m, l, n_hit)
                else:
                    lax.fori_loop(0, n_hit, functools.partial(expert, m, l), 0)
        return scr["part"][...]

    def hf_params(self, prefix: str | None = None, g: Any = None) -> list[HFParamDecl]:
        geom = self._require_geometry(g)
        moe_tp, _ = self._check_option_consistency(geom)
        return moe_hf_params(
            self.prefix if prefix is None else prefix,
            geom,
            shared_gate=self.shared_gate,
            moe_tp=moe_tp,
        )

    def cache_decl(self, g: Any = None) -> None:
        return None

    def reference(
        self,
        params: Any,
        x: Any,
        state: Any = None,
        *,
        g: Any = None,
        ex: ExecSpec,
        layer_idx: int = 0,
    ) -> tuple[jax.Array, Any]:
        geom = self._require_geometry(g)
        self._check_option_consistency(geom)
        return moe_reference(
            params,
            x,
            state,
            g=geom,
            ex=ex,
            layer_idx=layer_idx,
        )
