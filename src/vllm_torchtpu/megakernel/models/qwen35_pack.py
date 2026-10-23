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
"""Qwen3.5 logical input generation, megakernel packing, and verify accept."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .. import kernel, planner
from ..blocks.gdn import GDNBlock
from ..blocks.moe.reference import random_nvfp4_weight
from ..exec_spec import ExecSpec
from ..ops import BF16, F32, FP8, I32, rup
from ..pool.config import POOL_STATE_NAME, cache_exec, resolved_max_pages_per_seq
from . import qwen35_pool
from .qwen35_weights import default_eplb_cores, encode_eplb_ecopy


def accept_states(g: Any, ex: ExecSpec, st: dict[str, Any], acc: Any) -> dict[str, Any]:
    """Delegate verify-mode GDN state selection to `GDNBlock.accept`."""
    return GDNBlock.accept(st, st, acc, T=ex.T, vh=g.vh(ex.R))


def random_logical_inputs(
    key: jax.Array, g: Any, ex: ExecSpec
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate small random logical inputs for `reference_step` and `pack_inputs`."""
    g.check(ex)
    R, B, T, NS, ctx = ex.R, ex.B, ex.T, ex.NS, ex.ctx
    L, H, E, HD = g.L, g.H, g.E, g.HD
    nh, isr, ir, er, ew = g.nh(R), g.ISR(R), g.IR(R), g.ER(R), g.EW(R)
    lay_gdn = GDNBlock.lin_in_layout(g, ex)
    vh, KD, VD = lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
    qkv, z, nli = lay_gdn.QKV, lay_gdn.Z, lay_gdn.NLI
    natt, ngdn = g.NATT, g.NGDN
    it = iter(jax.random.split(key, 50))
    base_pos = jnp.arange(1, NS + 1, dtype=I32) * min(16, ctx // 4)
    pos = (
        base_pos
        if T == 1
        else (base_pos[:, None] + jnp.arange(T, dtype=I32)[None, :]).reshape(B)
    )
    hidden = jax.random.normal(next(it), (B, H), dtype=BF16) * 0.1

    def rfp8(shape):
        return (jax.random.normal(next(it), shape, dtype=F32) * 0.25).astype(FP8)

    def rscale(shape):
        return jnp.abs(jax.random.normal(next(it), shape, dtype=F32)) * 0.02 + 0.01

    G, nkv, koff = g.G(R), g.NKV, nh * 2 * HD
    att_in = jnp.concatenate(
        [rfp8((R, natt, H, koff)), jnp.repeat(rfp8((nkv, natt, H, 2 * HD)), G, axis=0)],
        axis=-1,
    )
    att_in_s = jnp.concatenate(
        [rscale((R, natt, koff)), jnp.repeat(rscale((nkv, natt, 2 * HD)), G, axis=0)],
        axis=-1,
    )

    kn = jnp.repeat(
        jax.random.normal(next(it), (nkv, natt, HD), dtype=F32) * 0.05, G, axis=0
    )
    sh_gate = jnp.broadcast_to(
        (jax.random.normal(next(it), (L, H), dtype=F32) * 0.1)[None, :, :], (R, L, H)
    )

    def rexp(k_dim: int, n_dim: int) -> jax.Array:
        if g.expert_quant == "nvfp4":
            return random_nvfp4_weight(next(it), (R, L, ew), k_dim, n_dim)
        return rfp8((R, L, ew, k_dim, n_dim))

    e13, e13_s = rexp(H, 2 * ir), rscale((R, L, ew, 2 * ir))
    e2, e2_s = rexp(ir, H), rscale((R, L, ew, H))

    logical_weights: dict[str, Any] = {
        "in_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "post_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "final_norm": jax.random.normal(next(it), (H,), dtype=F32) * 0.05,
        "att_in": att_in,
        "att_in_s": att_in_s,
        "att_out": rfp8((R, natt, nh * HD, H)),
        "att_out_s": rscale((R, natt, H)),
        "qn": jax.random.normal(next(it), (R, natt, HD), dtype=F32) * 0.05,
        "kn": kn,
        "router": jax.random.normal(next(it), (L, H, E), dtype=BF16) * 0.1,
        "sh_gate": sh_gate,
        "sh13": rfp8((R, L, H, 2 * isr)),
        "sh13_s": rscale((R, L, 2 * isr)),
        "sh2": rfp8((R, L, isr, H)),
        "sh2_s": rscale((R, L, H)),
        "e13": e13,
        "e13_s": e13_s,
        "e2": e2,
        "e2_s": e2_s,
    }
    sh_rep = g.sh_rep(R)
    if sh_rep > 1:
        for k in ("sh13", "sh13_s", "sh2"):
            w_k = logical_weights[k]
            logical_weights[k] = jnp.repeat(w_k[::sh_rep], sh_rep, axis=0)
    if not g.moe_tp and ew > er and R > 1:
        e_cores = default_eplb_cores(g, R)
        for lyr in range(L):
            for r1 in range(R):
                for k_idx, g_id in enumerate(e_cores[lyr][r1][er:ew]):
                    r0, s0 = divmod(int(g_id), er)
                    slot = er + k_idx
                    e13 = e13.at[r1, lyr, slot].set(e13[r0, lyr, s0])
                    e13_s = e13_s.at[r1, lyr, slot].set(e13_s[r0, lyr, s0])
                    e2 = e2.at[r1, lyr, slot].set(e2[r0, lyr, s0])
                    e2_s = e2_s.at[r1, lyr, slot].set(e2_s[r0, lyr, s0])
        ecopy = encode_eplb_ecopy(g, R, e_cores)
        logical_weights.update(
            e13=e13, e13_s=e13_s, e2=e2, e2_s=e2_s, e_cores=e_cores, ecopy=ecopy
        )

    if ngdn > 0:
        r_vh = jax.random.normal(next(it), (R, ngdn, vh), dtype=F32) * 0.1
        r_vd = jax.random.normal(next(it), (R, ngdn, VD), dtype=F32) * 0.05
        logical_weights.update(
            lin_in=rfp8((R, ngdn, H, nli)),
            lin_in_s=rscale((R, ngdn, nli)),
            lin_out=rfp8((R, ngdn, z, H)),
            lin_out_s=rscale((R, ngdn, H)),
            conv=jax.random.normal(next(it), (R, ngdn, 4 * qkv), dtype=F32) * 0.05,
            alog=r_vh - 1.0,
            dtb=jax.random.normal(next(it), (R, ngdn, vh), dtype=F32) * 0.1,
            gnorm=jnp.ones((R, ngdn, VD), dtype=F32) + r_vd,
        )

    kc_nkv = jax.random.normal(next(it), (nkv, natt, NS, ctx, HD), dtype=BF16) * 0.1
    vc_nkv = jax.random.normal(next(it), (nkv, natt, NS, ctx, HD), dtype=BF16) * 0.1
    logical_states: dict[str, Any] = {
        "kc": jnp.repeat(kc_nkv, G, axis=0),
        "vc": jnp.repeat(vc_nkv, G, axis=0),
    }
    if ngdn > 0:
        logical_states["conv"] = (
            jax.random.normal(next(it), (R, ngdn, NS, 3, qkv), dtype=F32) * 0.05
        )
        logical_states["rec"] = (
            jax.random.normal(next(it), (R, ngdn, NS, vh, KD, VD), dtype=BF16) * 0.05
        )
        kh_rep = g.kh_rep(R)
        if kh_rep > 1:
            qk_cols = 2 * lay_gdn.kh * KD
            for k in ("lin_in", "lin_in_s"):
                arr = logical_weights[k]
                rep_qk = jnp.repeat(arr[::kh_rep, ..., :qk_cols], kh_rep, axis=0)
                logical_weights[k] = arr.at[..., :qk_cols].set(rep_qk)
            cw = logical_weights["conv"].reshape(R, ngdn, 4, qkv)
            cw_qk = jnp.repeat(cw[::kh_rep, ..., :qk_cols], kh_rep, axis=0)
            logical_weights["conv"] = (
                cw.at[..., :qk_cols].set(cw_qk).reshape(R, ngdn, 4 * qkv)
            )
            cs = logical_states["conv"]
            logical_states["conv"] = cs.at[..., :qk_cols].set(
                jnp.repeat(cs[::kh_rep, ..., :qk_cols], kh_rep, axis=0)
            )
    return pos, hidden, logical_weights, logical_states


def pack_inputs(
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    g: Any,
    ex: ExecSpec,
    *,
    mesh: Any = None,
    build_default_pool: bool = False,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Pack logical model weights and states into the tiled/folded megakernel layout."""
    ms = g.to_model_spec()
    p = planner.plan(ms, ex)
    R, B, T, NS, M = ex.R, ex.B, ex.T, ex.NS, ex.M
    L, ngdn = g.L, g.NGDN

    hid_pad = jnp.pad(hidden[:B], ((0, M - B), (0, 0)))
    hid_mk = jnp.tile(hid_pad[None, :, :], (R, 1, 1)).reshape(R * M, g.H)

    def tile_w(arr: jax.Array, kind: str) -> jax.Array:
        *lead, k_in, n_in = arr.shape
        nt_, tk_ = p.nt[kind], p.tk[kind]
        exp_n = p.kinds[kind][1]
        assert nt_ * tk_ == k_in, (kind, k_in, nt_, tk_)
        assert n_in == exp_n, (kind, n_in, exp_n)
        return arr.reshape((*lead, nt_, tk_, n_in))

    lay = p.smalls_layout
    S = p.smalls_width
    sm = jnp.zeros((R, L, 1, S), dtype=F32)

    def set_sm_all(name: str, val: jax.Array):
        nonlocal sm
        if name not in lay:
            return
        o, w = lay[name]
        if val.ndim == 2:
            val = jnp.broadcast_to(val[None, :, :], (R, L, val.shape[-1]))
        assert val.shape[-1] <= w, (name, val.shape[-1], w)
        vw = min(w, val.shape[-1])
        sm = sm.at[:, :, 0, o : o + vw].set(val[:, :, :vw].astype(F32))

    def set_sm_layer(lyr: int, name: str, val_l: jax.Array):
        nonlocal sm
        if name not in lay:
            return
        o, w = lay[name]
        if val_l.ndim == 1:
            val_l = jnp.broadcast_to(val_l[None, :], (R, val_l.shape[-1]))
        assert val_l.shape[-1] <= w, (name, val_l.shape[-1], w)
        vw = min(w, val_l.shape[-1])
        sm = sm.at[:, lyr, 0, o : o + vw].set(val_l[:, :vw].astype(F32))

    def branch_slice(arr: jax.Array, lyr: int, ord_idx: int) -> jax.Array:
        return arr[:, lyr] if arr.shape[1] == L else arr[:, ord_idx]

    lw = logical_weights
    for k in ("in_norm", "post_norm", "sh_gate", "sh13_s", "sh2_s"):
        set_sm_all(k, lw[k])
    if "ecopy" in lw:
        set_sm_all("ecopy", lw["ecopy"])
    elif "e_cores" in lw:
        set_sm_all("ecopy", encode_eplb_ecopy(g, R, lw["e_cores"]))

    for lyr in range(L):
        if (lyr + 1) % g.att_every == 0:
            a = lyr // g.att_every
            set_sm_layer(lyr, "in_s", branch_slice(lw["att_in_s"], lyr, a))
            set_sm_layer(lyr, "out_s", branch_slice(lw["att_out_s"], lyr, a))
            for k in ("qn", "kn"):
                set_sm_layer(lyr, k, branch_slice(lw[k], lyr, a))
        else:
            o = lyr - (lyr + 1) // g.att_every
            set_sm_layer(lyr, "in_s", branch_slice(lw["lin_in_s"], lyr, o))
            set_sm_layer(lyr, "out_s", branch_slice(lw["lin_out_s"], lyr, o))
            for k in ("conv", "alog", "dtb", "gnorm"):
                set_sm_layer(lyr, k, branch_slice(lw[k], lyr, o))

    router_rep = jnp.broadcast_to(lw["router"][None, ...], (R, L, g.H, g.E))

    def _pack_expert(raw_e: Any, kind: str) -> Any:
        if ex.stack_e:
            if isinstance(raw_e, (tuple, list)):
                raw_e = jnp.stack(raw_e, axis=1)
            return tile_w(raw_e, kind)
        if isinstance(raw_e, (tuple, list)):
            return tuple(tile_w(raw_e[lyr], kind) for lyr in range(L))
        return tuple(tile_w(raw_e[:, lyr], kind) for lyr in range(L))

    e13_mk = _pack_expert(logical_weights["e13"], "e13")
    e2_mk = _pack_expert(logical_weights["e2"], "e2")

    es_mk = jnp.concatenate(
        [logical_weights["e13_s"], logical_weights["e2_s"]], axis=-1
    )[:, :, :, None, :].astype(F32)
    fn_mk = jnp.broadcast_to(
        logical_weights["final_norm"][None, None, :], (R, 1, g.H)
    ).astype(F32)

    weights_mk: dict[str, Any] = {
        "att_in": tile_w(logical_weights["att_in"], "att_in"),
        "att_out": tile_w(logical_weights["att_out"], "att_out"),
        "router": tile_w(router_rep, "router"),
        "sh13": tile_w(logical_weights["sh13"], "sh13"),
        "sh2": tile_w(logical_weights["sh2"], "sh2"),
        "e13": e13_mk,
        "e2": e2_mk,
        "es": es_mk,
        "smalls": sm,
        "final_norm": fn_mk,
    }
    if ngdn > 0:
        weights_mk["lin_in"] = tile_w(logical_weights["lin_in"], "lin_in")
        weights_mk["lin_out"] = tile_w(logical_weights["lin_out"], "lin_out")

    states_mk: dict[str, Any] = {}
    if ngdn > 0 and "conv" in logical_states and "rec" in logical_states:
        conv_l, rec_l = logical_states["conv"], logical_states["rec"]
        qkv, vh = g.QKV(R), g.vh(R)
        if conv_l.ndim == 5 and conv_l.shape[3] == 3:
            if T == 1:
                cr = rup(3 * B, 8)
                conv_3b = jnp.swapaxes(conv_l, 2, 3).reshape(R, ngdn, 3 * B, qkv)
                conv_p = jnp.pad(conv_3b, ((0, 0), (0, 0), (0, cr - 3 * B), (0, 0)))
            else:
                hr = rup(3 + T, 8)
                conv_p = jnp.pad(conv_l, ((0, 0), (0, 0), (0, 0), (0, hr - 3), (0, 0)))
        else:
            conv_p = conv_l
        rec_p = (
            rec_l.reshape(R, ngdn, NS * vh, g.KD, g.VD) if rec_l.ndim == 6 else rec_l
        )
        states_mk["conv"] = kernel.fold(conv_p.astype(F32))
        states_mk["rec"] = kernel.fold(rec_p.astype(BF16))

    if cache_exec(ex).paged_kv:
        if POOL_STATE_NAME in logical_states and "page_table" in logical_states:
            states_mk[POOL_STATE_NAME] = tuple(
                kernel.fold(r) if r.ndim == 6 else r
                for r in logical_states[POOL_STATE_NAME]
            )
            pt_in = logical_states["page_table"]
            max_pt = resolved_max_pages_per_seq(ex)
            if pt_in.ndim == 2:
                if pt_in.shape[0] not in (NS, R * NS) or pt_in.shape[1] < max_pt:
                    raise ValueError(
                        f"page_table has shape {pt_in.shape}, expected "
                        f"({NS}, >={max_pt}) or ({R * NS}, >={max_pt})"
                    )
                pt_in = pt_in[:, :max_pt]
                states_mk["page_table"] = (
                    pt_in if pt_in.shape[0] == R * NS else jnp.tile(pt_in, (R, 1))
                )
            elif pt_in.ndim == 3:
                if pt_in.shape[:2] != (R, NS) or pt_in.shape[2] < max_pt:
                    raise ValueError(
                        f"page_table has shape {pt_in.shape}, expected "
                        f"({R}, {NS}, >={max_pt})"
                    )
                states_mk["page_table"] = kernel.fold(pt_in[:, :, :max_pt])
            else:
                raise ValueError(
                    f"page_table must be 2D or 3D, got shape {pt_in.shape}"
                )
        elif build_default_pool:
            pt, ckpt, roff, _ = qwen35_pool.default_pool_layout(g, ex)
            states_mk[POOL_STATE_NAME] = qwen35_pool.states_to_pool(
                g, ex, logical_states, pt, ckpt, roff
            )
            states_mk["page_table"] = jnp.tile(
                pt[:, : resolved_max_pages_per_seq(ex)], (R, 1)
            )
        else:
            raise ValueError(
                f"pack_inputs with paged_kv=True requires {POOL_STATE_NAME!r} "
                "and 'page_table' in logical_states (or pass build_default_pool=True)."
            )
    elif "kc" in logical_states and "vc" in logical_states:
        states_mk["kc"] = kernel.fold(logical_states["kc"])
        states_mk["vc"] = kernel.fold(logical_states["vc"])

    if mesh is not None:
        rep, tp = NamedSharding(mesh, P()), NamedSharding(mesh, P("tp"))
        pos = jax.device_put(pos, rep)
        hid_mk = jax.device_put(hid_mk, tp)

        def _put_tp(v: Any) -> Any:
            if isinstance(v, tuple):
                return tuple(jax.device_put(x, tp) for x in v)
            return jax.device_put(v, tp)

        weights_mk = {k: _put_tp(v) for k, v in weights_mk.items()}
        states_mk = {k: _put_tp(v) for k, v in states_mk.items()}

    return pos, hid_mk, weights_mk, states_mk


def random_inputs(
    key: jax.Array, g: Any, ex: ExecSpec, *, mesh: Any = None
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate packed megakernel inputs `(pos, hidden, weights, states)`."""
    pos, hid_l, w_l, st_l = random_logical_inputs(key, g, ex)
    return pack_inputs(
        pos,
        hid_l,
        w_l,
        st_l,
        g,
        ex,
        mesh=mesh,
        build_default_pool=cache_exec(ex).paged_kv,
    )


__all__ = ["accept_states", "pack_inputs", "random_inputs", "random_logical_inputs"]
