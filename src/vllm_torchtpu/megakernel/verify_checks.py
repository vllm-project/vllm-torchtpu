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
"""Internal check runners and error metrics for `verify_model`."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from . import cache, kernel, pool
from . import weights as w_mod
from .dma import vmem
from .exec_spec import ExecSpec
from .geometry import ModelGeometry
from .model_def import ModelDef
from .module_options import registered_tunables
from .ops import I32
from .pool.config import POOL_STATE_NAME, cache_exec, with_cache
from .profile import Profile
from .profiles.default import chip_budget
from .verify_metrics import (
    _allclose,
    _compare_state_dict,
    _spec_state_decls,
    err_metrics,
)
from .weights.logical import resolve_spec_and_norms

if TYPE_CHECKING:
    from .verify import CheckResult, ToleranceSpec


@dataclass(frozen=True)
class VerifyCtx:
    """Shared context for `verify_model` sub-checks."""

    m_def: type[ModelDef]
    g: ModelGeometry
    R: int
    B_per_T: int
    t_vals: tuple[int, ...]
    ctx: int
    seed: int
    tol: ToleranceSpec
    inputs_by_t: Mapping[int, tuple[Any, Any, Any, Any]] = dataclasses.field(
        default_factory=dict
    )


def prof_exec(
    prof: Profile, g: ModelGeometry, *, R: int, B: int, T: int, ctx: int, interp: bool
) -> ExecSpec:
    ex = prof.exec_for(
        ExecSpec(R=R, B=B, T=T, ctx=ctx, interpret=interp), B=B, T=T, g=g
    )
    t = ex.resolve_tiles(g)
    if interp:
        nbd, nbe = min(t.nbd, 10), max(6, min(t.nbe, 12))
        b_sz, eb_sz = min(t.bank, 512 << 10), min(t.ebank, 256 << 10)
        t = dataclasses.replace(t, bank=b_sz, nbd=nbd, nbe=nbe, ebank=eb_sz)
    for m_name in registered_tunables():
        m_tun = t.module(m_name)
        kv_blk, kv_sub = getattr(m_tun, "kv_blk", None), getattr(m_tun, "kv_sub", None)
        if kv_blk is not None and (kv_blk > ctx or (kv_sub and kv_sub > ctx // 2)):
            upd = {"kv_blk": min(kv_blk, ctx)}
            if kv_sub is not None:
                upd["kv_sub"] = min(kv_sub, ctx // 2) if kv_sub else 0
            t = t.with_modules({m_name: upd})
    return dataclasses.replace(ex, tiles=t)


def check_derived_artifacts(
    vc: VerifyCtx, prof: Profile, res_cls: type[CheckResult]
) -> CheckResult:
    spec = vc.m_def.to_model_spec(vc.g, R=vc.R)
    has_cache = any(m.cache_decl(vc.g) is not None for m in spec.all_modules())
    play = vc.m_def.pool_layout(vc.g, R=vc.R) if has_cache else None
    chip = chip_budget()
    vmem_lim, smem_lim = chip.vmem_mib * 1024 * 1024, chip.smem_mib * 1024 * 1024
    max_vmem, max_smem = 0, 0
    for T in vc.t_vals:
        B = max(1, vc.B_per_T * int(T))
        ex = prof_exec(prof, vc.g, R=vc.R, B=B, T=int(T), ctx=vc.ctx, interp=False)
        vc.g.check(ex)
        cp, _ = vmem.build_scratch(spec, ex)
        max_vmem, max_smem = max(max_vmem, cp.vmem_bytes), max(max_smem, cp.smem_bytes)
        if play is not None:
            play.validate(ex)
            pt, ck, roff, min_p = pool.allocate_test_layout(play, ex)
            ex_p = with_cache(ex, num_pages=max(cache_exec(ex).num_pages, min_p))
            ck_kw = dict(ckpt_indices=ck, read_offsets=roff, check_bounds=True)
            pool.check_step_pool_metadata(
                play, ex_p, pt, jnp.arange(ex.B, dtype=I32), **ck_kw
            )
    ok = max_vmem <= vmem_lim and max_smem <= smem_lim
    st: Literal["passed", "failed"] = "passed" if ok else "failed"
    rsn = "" if ok else f"VMEM {max_vmem}/{vmem_lim} or SMEM {max_smem}/{smem_lim}"
    tols = {"vmem_limit_bytes": vmem_lim, "smem_limit_bytes": smem_lim}
    np_v, sp_v = (0, 0) if play is None else (play.num_pools, play.slots_per_pool)
    det = {
        **tols,
        "target_chip": chip.name,
        "max_vmem_bytes": max_vmem,
        "max_smem_bytes": max_smem,
        "num_pools": np_v,
        "slots_per_pool": sp_v,
    }
    return res_cls("derived_artifacts", st, tolerance=tols, reason=rsn, details=det)


def check_loader_parity(
    vc: VerifyCtx,
    ex: ExecSpec,
    w_hf_q: Mapping[str, Any],
    w_log: Mapping[str, Any],
    res_cls: type[CheckResult],
) -> CheckResult:
    """Verify exact declaration self-consistency across `load_hf`,
    `MappingWeightSource`, and `SyntheticVLLMWeightSource` (`moe_tp=True` & `False`).
    """
    modes = (True, False) if int(getattr(vc.g, "E", 0)) > 0 else (vc.g.moe_tp,)
    stats, total_tensors, lw = [0.0, 0.0, 0, 1.0], 0, w_mod.load_weights
    for mtp in modes:
        g_m = dataclasses.replace(vc.g, moe_tp=mtp)
        w_hf_mk = w_mod.load_hf(vc.m_def, ex, w_hf_q, g=g_m)
        w_log_m = (
            w_log
            if mtp == vc.g.moe_tp
            else w_mod.load_logical_weights(vc.m_def, ex, w_hf_q, g=g_m)
        )
        src_m = w_mod.MappingWeightSource(w_log_m)
        src_v = w_mod.SyntheticVLLMWeightSource(vc.m_def, ex, w_hf_q, g=g_m)
        map_d = [lw(vc.m_def, ex, src_m, g=g_m, tp_rank=r) for r in range(ex.R)]
        vllm_r = [
            lw(vc.m_def, ex, src_v, g=g_m, tp_rank=r)["w_flat"] for r in range(ex.R)
        ]
        map_r = [d["w_flat"] for d in map_d]
        flat_hf: list[Any] = []
        for k in map_d[0]["plan"].w_names:
            v = w_hf_mk[k]
            flat_hf.extend(v if isinstance(v, (tuple, list)) else (v,))
        total_tensors = max(total_tensors, len(flat_hf))
        for ranks in (map_r, vllm_r):
            for idx, a in enumerate(flat_hf):
                b = jnp.stack([rk[idx] for rk in ranks], axis=0)
                a_u, b_u = np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8)
                if a.shape != b.shape or not np.array_equal(a_u, b_u):
                    ab, rl, ul, cs = err_metrics(b, a)
                    stats[0], stats[1] = max(stats[0], max(ab, 1e-7)), max(stats[1], rl)
                    stats[2], stats[3] = max(stats[2], max(1, ul)), min(stats[3], cs)
    ok = stats[2] == 0 and stats[0] == 0.0
    st: Literal["passed", "failed"] = "passed" if ok else "failed"
    rsn = "" if ok else f"loader mismatch (ulp={stats[2]}, abs={stats[0]:.4g})"
    det = {"num_flat_tensors": total_tensors, "moe_tp_modes": modes}
    return res_cls("loader_parity", st, *stats, {"max_ulp": 0}, rsn, det)


def run_kernel_vs_ref(
    vc: VerifyCtx,
    prof: Profile,
    check_name: str,
    res_cls: type[CheckResult],
    *,
    include_direct: bool = True,
) -> CheckResult:
    m_def, g, R, tol = vc.m_def, vc.g, vc.R, vc.tol
    is_approx = prof.numerics == "approx"
    h_atol = tol.approx_hidden_atol if is_approx else tol.hidden_atol
    h_rtol = tol.approx_hidden_rtol if is_approx else tol.hidden_rtol
    h_cos = tol.approx_hidden_min_cos if is_approx else tol.hidden_min_cos
    devs = jax.devices()
    if len(devs) < R:
        raise RuntimeError(
            f"verify_model requires >= {R} JAX devices, found {len(devs)}"
        )
    spec = m_def.to_model_spec(g, R=R)
    has_cache = any(m.cache_decl(g) is not None for m in spec.all_modules())
    play = m_def.pool_layout(g, R=R) if has_cache else None
    all_have_hf = all(m.has_hf_params for m in spec.all_modules())
    _, _, pre_n, post_n, final_n = resolve_spec_and_norms(m_def, None, g=g)
    mesh = Mesh(mesh_utils.create_device_mesh((R,), devices=devs[:R]), ("tp",))
    stats, per_mode_t, failed = [0.0, 0.0, 0, 1.0], {}, []

    def _rec_h(out_h: Any, ref_h: Any, B: int, tag: str) -> dict[str, Any]:
        ref_b = jnp.broadcast_to(ref_h[:B][None, ...], (R, B, int(g.H)))
        u_h = kernel.unfold(out_h, R)[:, :B]
        ab, rl, ul, cs = err_metrics(u_h, ref_b)
        stats[0], stats[1] = max(stats[0], ab), max(stats[1], rl)
        stats[2], stats[3] = max(stats[2], ul), min(stats[3], cs)
        if not _allclose(u_h, ref_b, atol=h_atol, rtol=h_rtol):
            failed.append(f"{tag} hidden abs={ab:.4g} cos={cs:.6f}")
        elif cs < h_cos:
            failed.append(f"{tag} hidden cos={cs:.6f} < {h_cos}")
        res = {"hidden_max_abs": ab, "hidden_max_ulp": ul, "hidden_min_cos": cs}
        per_mode_t[tag] = res
        return res

    pk_i = w_mod.pack_inputs
    for T in vc.t_vals:
        B = max(1, vc.B_per_T * int(T))
        ex_b = prof_exec(prof, g, R=R, B=B, T=int(T), ctx=vc.ctx, interp=True)
        pos, hid, w_l, st_l = vc.inputs_by_t[int(T)]
        if include_direct:
            ex_d = with_cache(
                ex_b, paged_kv=False, fused_pool_io=False, pool_gather=False
            )
            ref_h, ref_st = m_def.reference_step(g, ex_d, pos, hid, w_l, st_l)
            if all_have_hf:
                pos_m, hid_m, w_m, st_m = pk_i(
                    pos, hid, w_l, st_l, g, ex_d, mesh=mesh, model_def=m_def
                )
            else:
                spec_kw = dict(pre_norm=pre_n, post_norm=post_n, final_norm=final_n)
                pos_m, hid_m, w_m, st_m = w_mod.pack_spec_inputs(
                    pos, hid, w_l, st_l, spec, ex_d, mesh=mesh, **spec_kw
                )
            step_d, _ = kernel.make_step(mesh, spec, ex_d)
            out_h, out_st = step_d(pos_m, hid_m, w_m, st_m)
            m_dir = _rec_h(out_h, ref_h, B, f"DirectHBM_T{T}")
            u_st: dict[str, Any] = {}
            for mod in spec.all_modules():
                u_st.update(mod.unpack_kernel_states(out_st, g, ex_d))
            for k, v in out_st.items():
                if k not in u_st:
                    u_st[k] = kernel.unfold(v, R)
            if int(T) > 1:
                for mod in spec.all_modules():
                    g_nxt = mod.next_logical_states(u_st, g, ex_d)
                    r_nxt = mod.next_logical_states(ref_st, g, ex_d)
                    for k in ("conv", "rec"):
                        if k in r_nxt:
                            ref_st = {**ref_st, f"{k}_final": r_nxt[k]}
                        if k in g_nxt:
                            u_st = {**u_st, f"{k}_final": g_nxt[k]}
            decls_d = _spec_state_decls(spec, g, ex_d)
            _compare_state_dict(
                u_st,
                ref_st,
                tol=tol,
                is_approx=is_approx,
                tag=f"DirectHBM_T{T}",
                m_out=m_dir,
                failed=failed,
                state_decls=decls_d,
            )

        if play is not None:
            pt, ck, roff, min_p = pool.allocate_test_layout(play, ex_b)
            ex_pw = with_cache(
                ex_b, paged_kv=True, num_pages=max(cache_exec(ex_b).num_pages, min_p)
            )
            ref_hp, ref_stp = m_def.reference_step(
                g, ex_pw, pos, hid, w_l, st_l, paged_kv=True, pool_io=True
            )
            pk_kw = dict(mesh=mesh, build_default_pool=True, model_def=m_def)
            pos_p, hid_p, w_p, st_p = pk_i(pos, hid, w_l, st_l, g, ex_pw, **pk_kw)
            ck_kw = dict(ckpt_indices=ck, read_offsets=roff, check_bounds=True)
            pool.check_step_pool_metadata(play, ex_pw, pt, pos, **ck_kw)
            p_args = (ck, roff) if play.has_recurrent else ()
            p_fn = cache.make_pooled_step(mesh, spec, ex_pw)
            out_hp, raw_out = p_fn(
                pos_p, hid_p, w_p, st_p[POOL_STATE_NAME], pt, *p_args
            )
            m_pw = _rec_h(out_hp, ref_hp, B, f"PallasWriter_T{T}")
            pool_got = pool.pool_to_states(
                play, ex_pw, raw_out, pt, ck if play.has_recurrent else None
            )
            pool_ref: dict[str, Any] = {"kc": ref_stp["kc"], "vc": ref_stp["vc"]}
            if play.has_recurrent:
                taps, qkv = play.conv_taps, play.qkv_per_rank(R)
                ngdn, vh = play.num_recurrent_layers, play.vh_per_rank(R)
                pool_ref["rec"] = ref_stp["rec"] if int(T) == 1 else ref_stp["rec_ckpt"]
                if int(T) == 1:
                    c_tb = ref_stp["conv"][:, :, : taps * B, :].reshape(
                        R, ngdn, taps, B, qkv
                    )
                    pool_ref["conv"] = jnp.swapaxes(c_tb, 2, 3)[:, :, :, None, :, :]
                else:
                    c_sl = [
                        ref_stp["conv_ckpt"][:, :, :, 1 + t : 1 + taps + t, :]
                        for t in range(int(T))
                    ]
                    pool_ref["conv"] = jnp.stack(c_sl, axis=3)
                    pool_ref["conv_final"] = pool_ref["conv"][:, :, :, -1]
                    pool_got["conv_final"] = pool_got["conv"][:, :, :, -1]
                    r_sh = (R, ngdn, ex_pw.NS, int(T), vh, play.KD, play.VD)
                    pool_ref["rec_final"] = pool_ref["rec"].reshape(r_sh)[:, :, :, -1]
                    pool_got["rec_final"] = pool_got["rec"].reshape(r_sh)[:, :, :, -1]
            _compare_state_dict(
                pool_got,
                pool_ref,
                tol=tol,
                is_approx=is_approx,
                paged_kv=True,
                tag=f"PallasWriter_T{T}",
                m_out=m_pw,
                failed=failed,
            )

    st: Literal["passed", "failed"] = "passed" if not failed else "failed"
    tols = {"hidden_atol": h_atol, "hidden_min_cos": h_cos}
    det, rsn = {"profile": prof.name, "metrics": per_mode_t}, "; ".join(failed)
    return res_cls(check_name, st, *stats, tols, rsn, det)
