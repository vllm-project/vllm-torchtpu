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
"""Grouped-Query Attention (`GQAModule`) with sequence-split KV ring and MTP."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
from jax import lax

from ...decls import (
    FrozenDict,
    HFParamDecl,
    KVCacheDecl,
    ModuleBinding,
    ResolvedModuleGeometry,
    normalize_weight_dtypes,
)
from ...errors import MegakernelConfigError
from ...exec_spec import ExecSpec
from ...fused_io_hook import fused_io
from ...module import Module
from ...ops import BF16, F32, FP8, I32, rnd
from ...pool.config import POOL_STATE_NAME, cache_exec, resolved_max_pages_per_seq
from ...spec import SmallField, StateDecl, WeightDecl
from .attention import att_core
from .context import GQAEmit
from .decls import (
    GQA_FIELD_ALIASES,
    GQA_GEOMETRY_FIELDS,
    GQA_GEOMETRY_METHODS,
    GQA_OPTIONS,
    GQA_PARAM_NAMES,
    GQA_WEIGHT_NAMES,
    gqa_cache_decl,
    gqa_hf_params,
    resolve_gqa_binding,
    validate_gqa_options,
)
from .kv import build_worklist, make_kv_ring
from .merge import split_merge
from .new_token import att_new
from .params import GQAParams, GQATunables, derive_params, gqa_scratch
from .reference import gqa_reference
from .staging import att_q, kv_new, q_exchange_start, q_gather_finish


class GQAModule(Module):
    """Full GQA module with QK RMSNorm, partial RoPE, Q/softmax exchange, and gate."""

    name: str = "gqa"
    reduction = "allreduce"
    input_norm: str | None = "in_norm"
    Tunables = GQATunables
    PARAM_NAMES = GQA_PARAM_NAMES
    GEOMETRY_FIELDS = GQA_GEOMETRY_FIELDS
    FIELD_ALIASES = GQA_FIELD_ALIASES
    OPTIONS = GQA_OPTIONS
    GEOMETRY_METHODS = GQA_GEOMETRY_METHODS

    def __init__(
        self,
        pool_layout: Any = None,
        *,
        geometry: Any = None,
        prefix: str = "self_attn",
        qk_norm: bool = True,
        output_gate: bool = True,
        weight_dtypes: Mapping[str, str] | None = None,
    ) -> None:
        # `PoolLayout` of the model's paged pools (None: the default page
        # packing `(2, HD // 4, 4, page_size)`, the only one the paged reads use).
        self.pool_layout = pool_layout
        self.geometry = geometry
        self.prefix = prefix
        self.qk_norm = bool(qk_norm)
        self.output_gate = bool(output_gate)
        self.weight_dtypes = normalize_weight_dtypes(
            weight_dtypes, GQA_WEIGHT_NAMES, "GQAModule", MegakernelConfigError
        )
        opts: dict[str, Any] = {
            "qk_norm": self.qk_norm,
            "output_gate": self.output_gate,
        }
        if weight_dtypes is not None:
            opts["weight_dtypes"] = self.weight_dtypes
        self._validate_options(opts)
        self.options = FrozenDict(opts)

    @classmethod
    def _validate_options(cls, options: Mapping[str, Any]) -> None:
        validate_gqa_options(options)

    @classmethod
    def _resolve_binding(
        cls, binding: ModuleBinding, hf_config: Any
    ) -> ResolvedModuleGeometry:
        return resolve_gqa_binding(
            cls, binding, super()._resolve_binding(binding, hf_config)
        )

    def kv_page_shape(self, g: Any, ex: ExecSpec) -> tuple[int, ...]:
        """Shape of one pool page, from `pool_layout` when the model sets one.

        Raises `MegakernelConfigError` if the layout's page packing differs from
        the `(2, HD // 4, 4, page_size)` packing the paged KV reads assume.
        """
        shape = (2, g.HD // 4, 4, cache_exec(ex).page_size)
        if self.pool_layout is None:
            return shape
        lay_shape = tuple(self.pool_layout.kv_page_shape)
        if lay_shape != shape:
            raise MegakernelConfigError(
                f"GQAModule paged KV reads need pool pages {shape}, but the pool "
                f"layout has {lay_shape}"
            )
        return lay_shape

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        in_dt = FP8 if self.weight_dtypes["att_in"] == "fp8" else BF16
        out_dt = FP8 if self.weight_dtypes["att_out"] == "fp8" else BF16
        return [
            WeightDecl("att_in", K=g.H, N=nai, dtype=in_dt, stream_ring="static"),
            WeightDecl(
                "att_out", K=nh * g.HD, N=g.H, dtype=out_dt, stream_ring="static"
            ),
        ]

    def smalls(self, g: Any, ex: ExecSpec) -> list[SmallField]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        fields = [SmallField("in_norm", g.H)]
        if self.weight_dtypes["att_in"] == "fp8":
            fields.append(SmallField("in_s", nai))
        if self.weight_dtypes["att_out"] == "fp8":
            fields.append(SmallField("out_s", g.H))
        fields.extend([SmallField("qn", g.HD), SmallField("kn", g.HD)])
        return fields

    def states(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        if cache_exec(ex).paged_kv:
            st = [
                StateDecl(
                    POOL_STATE_NAME,
                    (cache_exec(ex).num_pages, *self.kv_page_shape(g, ex)),
                    FP8,
                    access="r",
                    per_layer=True,
                    memory_space="hbm",
                    tupled=True,
                ),
                StateDecl(
                    "page_table",
                    (ex.NS, resolved_max_pages_per_seq(ex)),
                    I32,
                    access="r",
                    per_layer=False,
                    memory_space="smem",
                ),
            ]
            if cache_exec(ex).FUSED:
                st.extend(fused_io().gqa_fused_states(ex))
            return st
        return [
            StateDecl("kc", (ex.NS, ex.ctx, g.HD), BF16, access="r"),
            StateDecl("vc", (ex.NS, ex.ctx, g.HD), BF16, access="r"),
        ]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        if cache_exec(ex).FUSED:
            return fused_io().gqa_fused_outputs(g, ex)
        kv_dt = FP8 if cache_exec(ex).paged_kv else F32
        return [
            StateDecl("knew", (ex.MR, g.HD), kv_dt, access="w"),
            StateDecl("vnew", (ex.MR, g.HD), kv_dt, access="w"),
        ]

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return {"acc": max(nai, g.H), "ybuf": nai, "xb2": nh * g.HD, "mix": nh * g.HD}

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> GQAParams:
        if cache_exec(ex).paged_kv:
            self._check_pool_layout(plan)
        return derive_params(
            g,
            ex,
            plan,
            self.tunables(ex),
            in_scaled=(self.weight_dtypes["att_in"] == "fp8"),
            out_scaled=(self.weight_dtypes["att_out"] == "fp8"),
        )

    def _check_pool_layout(self, plan: Any) -> None:
        """`pool_layout` must give the same pool pages as the model's pool
        backend layout, which the cache pipes allocate and fill."""
        pb = plan.model_spec.cache_backend
        lay = None if pb is None else pb.layout
        if self.pool_layout is None or lay is None:
            return
        mine, theirs = tuple(self.pool_layout.kv_page_shape), tuple(lay.kv_page_shape)
        if mine != theirs:
            raise MegakernelConfigError(
                f"GQAModule pool_layout pages {mine} differ from the pool backend "
                f"layout's {theirs} on ModelSpec {plan.model_spec.name!r}"
            )

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        return gqa_scratch(ex, plan.module_params[self.name])

    def prologue(self, ctx: Any) -> None:
        """Build this rank's KV worklist (`wl_b`, `wl_t`, `wl_kp`, `wl_cnt[0]`)."""
        build_worklist(ctx, ctx.module_params(self.name))

    def init_buffers(self, ctx: Any) -> None:
        """Initialize attention-specific VMEM scratch buffers after barrier."""
        ex = ctx.ex
        ap: GQAParams = ctx.module_params(self.name)
        scr, pos_ref = ctx.scr, ctx.pos_ref
        scr["knew"][...] = jnp.zeros((ex.MR, ap.HD), F32)
        scr["vnew"][...] = jnp.zeros((ex.MR, ap.HD), F32)
        if cache_exec(ex).paged_kv:
            scr["knew_o"][...] = jnp.zeros((ex.MR, ap.HD), FP8)
            scr["vnew_o"][...] = jnp.zeros((ex.MR, ap.HD), FP8)
        if cache_exec(ex).FUSED:
            fused_io().gqa_fused_init_buffers(ctx)
        if ap.SPLIT and not ap.GQ:
            scr["pso"][...] = jnp.zeros(scr["pso"].shape, F32)
        if ap.PACK or ap.QV:
            scr["qg"][...] = jnp.zeros(scr["qg"].shape, F32)
        if ap.QV:
            half_ = ap.ROT // 2
            inv_ = 1.0 / (
                ap.theta
                ** (lax.broadcasted_iota(I32, (1, half_), 1).astype(F32) * 2.0 / ap.ROT)
            )
            rows_ = lax.broadcasted_iota(I32, (ex.MR, half_), 0)
            pf = jnp.zeros((ex.MR, half_), F32)
            for b in range(ex.B):
                pf = jnp.where(rows_ == b, pos_ref[b].astype(F32), pf)
            ang_ = pf * inv_
            scr["rope"][0, :, :half_] = rnd(jnp.cos(ang_))
            scr["rope"][1, :, :half_] = rnd(jnp.sin(ang_))

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ap: GQAParams = ctx.module_params(self.name)
        cx = GQAEmit.build(ctx, ap, l, self.name)
        ex, scr, sc, stream, a = cx.ex, cx.scr, cx.sc, cx.stream, cx.a
        nh, HD, H, SPLIT, LAZY_ATT = ap.nh, ap.HD, ap.H, ap.SPLIT, ap.LAZY_ATT
        kv = make_kv_ring(cx)
        with stream.segment(lazy=LAZY_ATT):
            ybuf, kn, half, koff, rcs = att_q(cx, slot)
            qx = q_exchange_start(cx)
            if cache_exec(ex).FUSED:
                with sc("att_kvrd"):
                    fused_io().gqa_fused_kvrd_issue(ctx, a)
            kv_out = kv_new(cx, ybuf, kn, half, koff, rcs)
            if SPLIT or ex.VER:
                q_gather_finish(cx, qx)
            att_core(cx, kv)
            if SPLIT:
                split_merge(cx)
            att_new(cx)
            with sc("att_out"):
                if cache_exec(ex).FUSED and not SPLIT:
                    fused_io().gqa_fused_kv_rmw(ctx, ap, a)
                elif not cache_exec(ex).FUSED:
                    for d in kv_out:
                        d.wait()
                NO = nh * HD
                scr["xb2"][:, :NO] = scr["mix"][:, :NO].astype(BF16)
                yo = stream.dmatmul("att_out", scr["xb2"], H)
        return yo * ctx.sm(slot, "out_s") if ap.out_scaled else yo

    def epilogue(self, ctx: Any) -> None:
        if cache_exec(ctx.ex).FUSED:
            fused_io().gqa_fused_epilogue(ctx)

    def host_update_states(
        self,
        g: Any,
        ex: ExecSpec,
        pos: Any,
        states: dict[str, Any],
        raw_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Scatter `knew` and `vnew` into `kc` and `vc` inside `shard_map`."""
        if cache_exec(ex).FUSED:
            if POOL_STATE_NAME in raw_outputs:
                return {POOL_STATE_NAME: raw_outputs[POOL_STATE_NAME]}
            return {}
        knew, vnew = raw_outputs["knew"], raw_outputs["vnew"]
        if cache_exec(ex).paged_kv or "kc" not in states:
            return {"knew": knew, "vnew": vnew}
        kc, vc = states["kc"], states["vc"]
        if self.tunables(ex).kv_scatter_vec_ns < ex.NS:
            bi = jnp.arange(ex.B) if ex.T == 1 else jnp.arange(ex.B) // ex.T
            kw = dict(
                indices_are_sorted=True, unique_indices=True, mode="promise_in_bounds"
            )
            kc = kc.at[:, bi, pos].set(knew[:, : ex.B].astype(BF16), **kw)
            vc = vc.at[:, bi, pos].set(vnew[:, : ex.B].astype(BF16), **kw)
            return {"kc": kc, "vc": vc}
        for s_idx in range(ex.NS):
            b0 = s_idx * ex.T
            upd = (0, s_idx, pos[b0], 0)
            kc = lax.dynamic_update_slice(
                kc, knew[:, None, b0 : b0 + ex.T].astype(BF16), upd
            )
            vc = lax.dynamic_update_slice(
                vc, vnew[:, None, b0 : b0 + ex.T].astype(BF16), upd
            )
        return {"kc": kc, "vc": vc}

    def hf_params(self, prefix: str | None = None, g: Any = None) -> list[HFParamDecl]:
        geom = self._require_geometry(g)
        return gqa_hf_params(
            self.prefix if prefix is None else prefix,
            geom,
            qk_norm=self.qk_norm,
            output_gate=self.output_gate,
            weight_dtypes=self.weight_dtypes,
        )

    def cache_decl(self, g: Any = None) -> KVCacheDecl:
        geom = g if g is not None else self.geometry
        if geom is None and self.pool_layout is None:
            geom = self._require_geometry(g)
        return gqa_cache_decl(geom, pool_layout=self.pool_layout)

    def init_logical_states(
        self, key: Any, g: Any, ex: ExecSpec, n_layers: int
    ) -> dict[str, Any]:
        import jax

        geom = self._require_geometry(g)
        cdecl = self.cache_decl(geom)
        nkv = int(geom.NKV)
        g_rep = ex.R // nkv
        k_kc, k_vc = jax.random.split(key)
        sh = (nkv, n_layers, ex.NS, ex.ctx, cdecl.hd)
        return {
            "kc": jnp.repeat(jax.random.normal(k_kc, sh, dtype=BF16) * 0.1, g_rep, 0),
            "vc": jnp.repeat(jax.random.normal(k_vc, sh, dtype=BF16) * 0.1, g_rep, 0),
        }

    def pack_logical_states(
        self, logical_states: Mapping[str, Any], g: Any, ex: ExecSpec
    ) -> dict[str, Any]:
        del g
        if (
            cache_exec(ex).paged_kv
            or "kc" not in logical_states
            or "vc" not in logical_states
        ):
            return {}
        kc, vc = logical_states["kc"], logical_states["vc"]
        return {
            "kc": kc.reshape((kc.shape[0] * kc.shape[1], *kc.shape[2:])),
            "vc": vc.reshape((vc.shape[0] * vc.shape[1], *vc.shape[2:])),
        }

    def unpack_kernel_states(
        self, raw_states: Mapping[str, Any], g: Any, ex: ExecSpec
    ) -> dict[str, Any]:
        del g
        return {
            k: raw_states[k].reshape(
                (ex.R, raw_states[k].shape[0] // ex.R, *raw_states[k].shape[1:])
            )
            for k in ("kc", "vc")
            if k in raw_states
        }

    def next_logical_states(
        self, out_states: Mapping[str, Any], g: Any, ex: ExecSpec
    ) -> dict[str, Any]:
        del g, ex
        return {k: out_states[k] for k in ("kc", "vc") if k in out_states}

    def reference(
        self, params: Any, x: Any, state: Any = None, *, g: Any = None, **kw: Any
    ) -> tuple[Any, dict[str, Any]]:
        assert state is not None, "GQAModule.reference requires `state` with 'kc'/'vc'"
        return gqa_reference(
            params, x, state, g=self._require_geometry(g), module_name=self.name, **kw
        )
