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
"""Composable GatedDeltaNet (`GDNModule`) megakernel module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from ...decls import (
    FrozenDict,
    HFParamDecl,
    ModuleBinding,
    RecurrentCacheDecl,
    ResolvedModuleGeometry,
)
from ...exec_spec import ExecSpec
from ...fused_io_hook import fused_io
from ...module import Module
from ...ops import BF16, F32, FP8, I32, rup
from ...pool.config import cache_exec
from ...spec import SmallField, StateDecl, WeightDecl
from .conv import conv_step, gdn_in
from .decls import (
    GDN_FIELD_ALIASES,
    GDN_GEOMETRY_FIELDS,
    GDN_GEOMETRY_METHODS,
    GDN_OPTIONS,
    GDN_PARAM_NAMES,
    gdn_cache_decl,
    gdn_hf_params,
    resolve_gdn_binding,
)
from .params import (
    GDNLinInLayout,
    GDNParams,
    GDNTunables,
    check_fused_gdn_options,
    derive_params,
    gdn_lin_in_layout,
    resolve_gdn_options,
    validate_gdn_options,
)
from .recurrence import gdn_rec, gdn_rec_chunked
from .reference import gdn_reference
from .states import conv_out_src, rec_drain, rec_in


class GDNModule(Module):
    """GatedDeltaNet module supporting decode (`T=1`), verify (`T>1`), and `GCH`."""

    name = "gdn"
    reduction = "allreduce"
    input_norm: str | None = "in_norm"
    Tunables = GDNTunables
    PARAM_NAMES = GDN_PARAM_NAMES
    GEOMETRY_FIELDS = GDN_GEOMETRY_FIELDS
    FIELD_ALIASES = GDN_FIELD_ALIASES
    OPTIONS: ClassVar[dict[str, Any]] = GDN_OPTIONS
    GEOMETRY_METHODS = GDN_GEOMETRY_METHODS

    def __init__(
        self,
        *,
        geometry: Any = None,
        prefix: str = "linear_attn",
        decay_type: str | None = None,
        decay_law: str | None = None,
        gate_type: str | None = None,
        decay_bound: float | None = None,
    ) -> None:
        self.geometry = geometry
        self.prefix = prefix
        requested = {
            "decay_type": decay_type,
            "decay_law": decay_law,
            "gate_type": gate_type,
            "decay_bound": decay_bound,
        }
        self._validate_options(requested)
        self.options = FrozenDict(requested)

    def resolved_options(self, g: Any) -> dict[str, Any]:
        """Options in effect for geometry `g`."""
        return resolve_gdn_options(self.options, g)

    @classmethod
    def _validate_options(cls, options: Mapping[str, Any]) -> None:
        validate_gdn_options(options)

    @classmethod
    def _resolve_binding(
        cls, binding: ModuleBinding, hf_config: Any
    ) -> ResolvedModuleGeometry:
        return resolve_gdn_binding(
            cls, binding, super()._resolve_binding(binding, hf_config)
        )

    @classmethod
    def chunk_rows(cls, ex: ExecSpec) -> tuple[int, int]:
        """`(gdn_rows, gdn_bufs)` of the chunked state path for `ex`."""
        tt = ex.resolved_tiles.module(cls.name, GDNTunables)
        return tt.gdn_rows, tt.gdn_bufs

    @staticmethod
    def lin_in_layout(
        g: Any, ex_or_r: ExecSpec | int, *, decay_type: str | None = None
    ) -> GDNLinInLayout:
        """Return shared column offsets for `lin_in` (`[qkv | z | b | a]`)."""
        return gdn_lin_in_layout(g, ex_or_r, decay_type=decay_type)

    @staticmethod
    def accept(
        states: dict[str, Any],
        ckpts: dict[str, Any],
        acc: Any,
        *,
        T: int,
        vh: int,
    ) -> dict[str, Any]:
        """Jittable on-device selection of post-verify GDN states from checkpoints."""
        c_ckpt = ckpts["conv_ckpt"] if "conv_ckpt" in ckpts else ckpts.get("conv")
        r_ckpt = ckpts["rec_ckpt"] if "rec_ckpt" in ckpts else ckpts.get("rec")
        out = {k: v for k, v in states.items() if k not in ("conv_ckpt", "rec_ckpt")}
        if c_ckpt is None or r_ckpt is None:
            return out

        NS = c_ckpt.shape[-3]
        acc_arr = jnp.asarray(acc, dtype=I32)
        assert acc_arr.shape == (NS,), (acc_arr.shape, NS)

        conv_next = jax.vmap(
            lambda c_seq, a_s: c_seq.at[..., 0:3, :].set(
                lax.dynamic_slice_in_dim(c_seq, a_s, 3, axis=-2)
            ),
            in_axes=(-3, 0),
            out_axes=-3,
        )(c_ckpt, acc_arr)

        lead = r_ckpt.shape[:-3]
        KD, VD = r_ckpt.shape[-2], r_ckpt.shape[-1]
        r_unflat = r_ckpt.reshape((*lead, NS, T, vh, KD, VD))
        r_sel = jax.vmap(
            lambda r_seq, a_s: lax.dynamic_index_in_dim(
                r_seq, a_s - 1, axis=-4, keepdims=False
            ),
            in_axes=(-5, 0),
            out_axes=-4,
        )(r_unflat, acc_arr)
        out["conv"] = conv_next
        out["rec"] = r_sel.reshape((*lead, NS * vh, KD, VD))
        return out

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        decay_type = self.resolved_options(g)["decay_type"]
        lay = self.lin_in_layout(g, ex, decay_type=decay_type)
        return [
            WeightDecl("lin_in", K=g.H, N=lay.NLI, dtype=FP8, stream_ring="static"),
            WeightDecl("lin_out", K=lay.Z, N=g.H, dtype=FP8, stream_ring="static"),
        ]

    def states(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        if cache_exec(ex).FUSED or cache_exec(ex).GPR:
            g_per_att = g.att_every - 1
            k_ckpt = cache_exec(ex).ckpt_k if cache_exec(ex).ckpt_k > 0 else ex.T
            ckpt_sh = (
                (ex.NS, k_ckpt)
                if cache_exec(ex).ckpt_2d
                else (g_per_att, ex.NS, k_ckpt)
            )
            roff_sh = (ex.NS,) if cache_exec(ex).roff_1d else (g_per_att, ex.NS)
            st = [
                StateDecl(
                    "ckpt_indices",
                    shape=ckpt_sh,
                    dtype=I32,
                    access="r",
                    per_layer=False,
                    memory_space="smem",
                ),
                StateDecl(
                    "read_offsets",
                    shape=roff_sh,
                    dtype=I32,
                    access="r",
                    per_layer=False,
                    memory_space="smem",
                ),
            ]
            if cache_exec(ex).act_mode == "ref":
                st.append(
                    StateDecl(
                        "active_seqs",
                        shape=(ex.NS,),
                        dtype=I32,
                        access="r",
                        per_layer=False,
                        memory_space="smem",
                    )
                )
            return st
        decay_type = self.resolved_options(g)["decay_type"]
        lay = self.lin_in_layout(g, ex, decay_type=decay_type)
        conv_shape = (
            (rup(3 * ex.B, 8), lay.QKV)
            if ex.T == 1
            else (ex.NS, rup(3 + ex.T, 8), lay.QKV)
        )
        return [
            StateDecl("conv", shape=conv_shape, dtype=F32, access="rw"),
            StateDecl(
                "rec", shape=(ex.NS * lay.vh, lay.KD, lay.VD), dtype=BF16, access="rw"
            ),
        ]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        opts = self.resolved_options(g)
        if cache_exec(ex).FUSED:
            check_fused_gdn_options(opts)
            return []
        decay_type = opts["decay_type"]
        lay = self.lin_in_layout(g, ex, decay_type=decay_type)
        if ex.VER:
            return [
                StateDecl(
                    "conv_ckpt",
                    shape=(ex.NS, rup(3 + ex.T, 8), lay.QKV),
                    dtype=BF16 if cache_exec(ex).GPR else F32,
                    access="w",
                ),
                StateDecl(
                    "rec_ckpt",
                    shape=(ex.B * lay.vh, lay.KD, lay.VD),
                    dtype=BF16,
                    access="w",
                ),
            ]
        return [
            StateDecl("conv", shape=(rup(3 * ex.B, 8), lay.QKV), dtype=F32, access="w"),
            StateDecl(
                "rec", shape=(ex.B * lay.vh, lay.KD, lay.VD), dtype=BF16, access="w"
            ),
        ]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        opts = self.resolved_options(g)
        if cache_exec(ex).FUSED:
            check_fused_gdn_options(opts)
        dt = opts["decay_type"]
        lay = self.lin_in_layout(g, ex, decay_type=dt)
        a_cols = lay.vh * lay.KD if dt == "key" else 128
        return [
            SmallField("in_norm", g.H),
            SmallField("in_s", lay.NLI),
            SmallField("out_s", g.H),
            SmallField("conv", 4 * lay.QKV),
            SmallField("alog", 128),
            SmallField("dtb", a_cols),
            SmallField("gnorm", lay.VD),
        ]

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        dt = self.resolved_options(g)["decay_type"]
        lay = self.lin_in_layout(g, ex, decay_type=dt)
        return {"acc": lay.NLI, "ybuf": lay.NLI, "xb2": lay.Z, "mix": lay.Z}

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> GDNParams:
        return derive_params(
            g,
            ex,
            plan,
            self.tunables(ex),
            self.name,
            options=self.resolved_options(g),
        )

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        gp: GDNParams = plan.module_params[self.name]
        if cache_exec(ex).FUSED:
            return fused_io().gdn_fused_scratch(ex, gp, plan)
        scr: dict[str, Any] = {
            "recb": (
                pltpu.VMEM((gp.NRB, gp.CBV, gp.KD, gp.VD), BF16)
                if gp.GCH
                else pltpu.VMEM((ex.B * gp.vh, gp.KD, gp.VD), BF16)
            ),
            "sbuf": pltpu.VMEM((gp.vh, gp.KD, gp.VD), F32),
            "convb": pltpu.VMEM(
                (ex.NS, gp.HR, gp.QKV) if ex.VER else (gp.CR, gp.QKV), F32
            ),
            "st_sem": pltpu.SemaphoreType.DMA((4,)),
        }
        if ex.VER and not gp.GCH:
            scr.update(
                {
                    "ub": pltpu.VMEM((ex.NS, rup(ex.T, 8), gp.QKV), F32),
                    "rin_sem": pltpu.SemaphoreType.DMA((ex.NS,)),
                }
            )
        if gp.GCH:
            scr["rec_sem"] = pltpu.SemaphoreType.DMA((2, gp.NRB))
            if gp.GPR:
                scr["gpr_rec"] = pltpu.VMEM(
                    (
                        gp.NRB,
                        gp.CB // ex.T,
                        gp.REC_PAGES,
                        *gp.pool_layout.kv_page_shape,
                    ),
                    FP8,
                )
                scr["gpr_conv"] = pltpu.VMEM(
                    (ex.NS, gp.GCONV_ROWS, *gp.pool_layout.kv_page_shape[2:]), FP8
                )
                scr["convb16"] = pltpu.VMEM((ex.NS, gp.HR, gp.QKV), BF16)
        return scr

    def init_buffers(self, ctx: Any) -> None:
        if not cache_exec(ctx.ex).FUSED:
            return
        fused_io().gdn_fused_init_buffers(ctx, ctx.module_params(self.name))

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex = ctx.ex
        gp: GDNParams = ctx.module_params(self.name)
        o = ctx.stream.sp.module_ord(l, self.name)
        if cache_exec(ex).FUSED:
            return fused_io().gdn_fused_emit(ctx, gp, o, slot)
        scr, O, skip, sc, stream = ctx.scr, ctx.O, ctx.skip, ctx.sc, ctx.stream  # noqa: E741
        Z, H, GCH = gp.Z, gp.H, gp.GCH
        rec_out_key = "rec_ckpt" if ex.VER else "rec"
        recb, sem = scr["recb"], scr["st_sem"]
        with stream.segment(lazy=False):
            ybuf = gdn_in(ctx, gp, o, slot)
            u = conv_step(ctx, gp, o, slot, ybuf)
            if GCH:
                with sc("gdn_rec"):
                    gdn_rec_chunked(ctx, gp, o, slot)
            else:
                if "gdn_state" not in skip:
                    with sc("gdn_rec_wait"):
                        for d in rec_in(ctx, gp, o):
                            d.wait()
                if "gdn" not in skip:
                    with sc("gdn_rec"):
                        gdn_rec(ctx, gp, u, ybuf, slot)
            with sc("gdn_out"):
                if not GCH and "gdn_state" not in skip:
                    pltpu.make_async_copy(recb, O[rec_out_key].at[o], sem.at[3]).start()
                scr["xb2"][:, :Z] = scr["mix"][:, :Z].astype(BF16)
                yo = stream.dmatmul("lin_out", scr["xb2"], H)
                return yo * ctx.sm(slot, "out_s")

    def epilogue(self, ctx: Any) -> None:
        gp: GDNParams = ctx.module_params(self.name)
        if cache_exec(ctx.ex).FUSED:
            fused_io().gdn_fused_epilogue(ctx, gp)
            return
        if gp.NGDN > 0 and "gdn_state" not in ctx.skip:
            conv_out_key = "conv_ckpt" if ctx.ex.VER else "conv"
            rec_out_key = "rec_ckpt" if ctx.ex.VER else "rec"
            o_last = gp.NGDN - 1
            pltpu.make_async_copy(
                conv_out_src(ctx, gp),
                ctx.O[conv_out_key].at[o_last],
                ctx.scr["st_sem"].at[2],
            ).wait()
            if gp.GCH:
                rec_drain(ctx, gp, o_last)
            else:
                pltpu.make_async_copy(
                    ctx.scr["recb"],
                    ctx.O[rec_out_key].at[o_last],
                    ctx.scr["st_sem"].at[3],
                ).wait()

    def host_update_states(
        self,
        g: Any,
        ex: ExecSpec,
        pos: Any,
        states: dict[str, Any],
        raw_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        if cache_exec(ex).FUSED:
            return {}
        if ex.VER:
            return {
                "conv_ckpt": raw_outputs["conv_ckpt"],
                "rec_ckpt": raw_outputs["rec_ckpt"],
            }
        return {"conv": raw_outputs["conv"], "rec": raw_outputs["rec"]}

    def hf_params(self, prefix: str | None = None, g: Any = None) -> list[HFParamDecl]:
        geom = self._require_geometry(g)
        return gdn_hf_params(
            self.prefix if prefix is None else prefix, geom, options=self.options
        )

    def cache_decl(self, g: Any = None) -> RecurrentCacheDecl:
        return gdn_cache_decl(self._require_geometry(g))

    def reference(
        self, params: Any, x: Any, state: Any = None, *, g: Any = None, **kw: Any
    ) -> tuple[jax.Array, dict[str, Any]]:
        assert state is not None, (
            "GDNModule.reference requires `state` with 'conv'/'rec'"
        )
        geom = self._require_geometry(g)
        kw.setdefault("options", self.resolved_options(geom))
        return gdn_reference(params, x, state, g=geom, **kw)
