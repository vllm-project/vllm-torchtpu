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
"""`verify_module` and `run_kernel_vs_oracle` verification harnesses."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh

from . import kernel
from . import weights as w_mod
from .decls import ModuleBinding
from .dma import vmem
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .module import Module
from .module_options import registered_tunables
from .modules.gdn import GDNModule
from .modules.gqa import GQAModule
from .ops import BF16, F32, rmsnorm, rnd
from .pool.config import with_cache
from .profiles.default import chip_budget
from .reference import finalize_reference_step_state, prepare_reference_step_state
from .spec import LayerType, ModelSpec
from .verify_checks import (
    _allclose,
    _compare_state_dict,
    _spec_state_decls,
    err_metrics,
)

if TYPE_CHECKING:
    from .verify import CheckResult, ToleranceSpec, VerificationReport

ModuleOracle = Callable[..., tuple[jax.Array, Mapping[str, Any]]]
"""Oracle `(params, x, state, *, g, ex, layer_idx=0, pos=pos, **kw) -> (out, states)`:
- `params`: full 1-layer logical weight dict (`dict[str, Any]`, including
  `final_norm` and `mod.input_norm` when declared), called with `layer_idx=0`.
  Generic `SmallField` entries have per-rank shape `(R, 1, width)`; only `*norm`
  fields are broadcast identically across ranks.
- `x`: input-normed `xb` in `bf16` (`(B, H)`) when `mod.input_norm is not None`,
  or un-normed `res0` in `f32` (`(B, H)`) when `mod.input_norm is None` (the
  kernel skips writing `xb` when `input_norm is None`).
- Return `(out, states)` where `out` is the rank-summed sublayer delta
  (`mod.reduction == "allreduce"`, to which the harness adds `rnd(res0 + delta)`)
  or the updated residual (`mod.reduction in ("none", "custom")`), followed by
  `final_norm` in the harness. Note that in-kernel `allreduce` rounds each
  rank's partial contribution to `bf16` before summing (with extra `bf16`
  roundings in hierarchical / RS+AG modes).
"""


def _module_exec(
    ex_base: ExecSpec | None, g1: Any, *, R: int, B: int, T: int, ctx: int, interp: Any
) -> ExecSpec:
    ex = (
        ExecSpec(R=R, B=B, T=T, ctx=ctx, interpret=interp)
        if ex_base is None
        else dataclasses.replace(ex_base, R=R, B=B, T=T, ctx=ctx, interpret=interp)
    )
    t = ex.resolve_tiles(g1)
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
    return with_cache(
        dataclasses.replace(ex, tiles=t),
        paged_kv=False,
        fused_pool_io=False,
        pool_gather=False,
    )


def check_module_derived_artifacts(
    spec: ModelSpec,
    g1: Any,
    ex_base: ExecSpec | None,
    *,
    R: int,
    B_per_T: int,
    t_vals: tuple[int, ...],
    ctx: int,
    res_cls: type[CheckResult],
) -> CheckResult:
    chip = chip_budget()
    vmem_lim, smem_lim = chip.vmem_mib * 1024 * 1024, chip.smem_mib * 1024 * 1024
    max_vmem, max_smem = 0, 0
    for T in t_vals:
        B = max(1, B_per_T * int(T))
        ex = _module_exec(ex_base, g1, R=R, B=B, T=int(T), ctx=ctx, interp=False)
        g1.check(ex)
        cp, _ = vmem.build_scratch(spec, ex)
        max_vmem, max_smem = max(max_vmem, cp.vmem_bytes), max(max_smem, cp.smem_bytes)
    ok = max_vmem <= vmem_lim and max_smem <= smem_lim
    st: Literal["passed", "failed"] = "passed" if ok else "failed"
    rsn = "" if ok else f"VMEM {max_vmem}/{vmem_lim} or SMEM {max_smem}/{smem_lim}"
    tols = {"vmem_limit_bytes": vmem_lim, "smem_limit_bytes": smem_lim}
    det = dict(
        tols,
        target_chip=chip.name,
        max_vmem_bytes=max_vmem,
        max_smem_bytes=max_smem,
    )
    return res_cls("derived_artifacts", st, tolerance=tols, reason=rsn, details=det)


def run_module_verify(
    mod: Module,
    spec: ModelSpec,
    g1: Any,
    ex_base: ExecSpec | None,
    *,
    R: int,
    B_per_T: int,
    t_vals: tuple[int, ...],
    ctx: int,
    in_by_t: Mapping[int, tuple[Any, Any, Any, Any]],
    tol: ToleranceSpec,
    module_oracle: Callable[..., tuple[Any, Any]] | None,
    check_name: str,
    res_cls: type[CheckResult],
) -> CheckResult:
    devs = jax.devices()
    if len(devs) < R:
        raise RuntimeError(
            f"verify_module requires >= {R} JAX devices, found {len(devs)}"
        )
    mesh = Mesh(mesh_utils.create_device_mesh((R,), devices=devs[:R]), ("tp",))
    fn_ref = module_oracle if module_oracle is not None else mod.reference
    stats, per_mode_t, failed = [0.0, 0.0, 0, 1.0], {}, []
    h_atol, h_rtol, h_cos = tol.hidden_atol, tol.hidden_rtol, tol.hidden_min_cos

    for i, T in enumerate(t_vals):
        B = max(1, B_per_T * int(T))
        interp_val: Any = pltpu.InterpretParams(detect_races=True) if i == 0 else True
        ex_d = _module_exec(ex_base, g1, R=R, B=B, T=int(T), ctx=ctx, interp=interp_val)
        pos, hid, w_l, st_l = in_by_t[int(T)]
        step_state, conv_in, rec_in, ngdn, vh, KD, VD = prepare_reference_step_state(
            st_l, g1, ex_d, pool_io=False, include_gdn_defaults=False
        )
        res0 = hid[:B].astype(F32)
        if mod.input_norm is not None:
            in_w = w_l[mod.input_norm]
            in_w_0 = (
                in_w[0, 0] if in_w.ndim == 3 else (in_w[0] if in_w.ndim == 2 else in_w)
            )
            x_in: Any = rmsnorm(
                res0,
                in_w_0,
                eps=float(g1.eps),
                zero_centered=bool(g1.zero_centered_norm),
            ).astype(BF16)
        else:
            x_in = res0

        if isinstance(mod, GQAModule) and module_oracle is None:
            ref_kw: dict[str, Any] = {
                "pos": pos,
                "att_idx": 0,
                "layer_idx": 0,
                "paged_kv": False,
            }
        elif isinstance(mod, GDNModule) and module_oracle is None:
            ref_kw = {"gdn_idx": 0, "layer_idx": 0, "multi_layer": True}
        else:
            ref_kw = {"layer_idx": 0, "pos": pos}

        sub_out, next_step_state = fn_ref(
            w_l, x_in, step_state, g=g1, ex=ex_d, **ref_kw
        )
        sub_arr = jnp.asarray(sub_out).astype(F32)
        res1 = rnd(res0 + sub_arr) if mod.reduction == "allreduce" else sub_arr
        fn_w = w_l["final_norm"]
        fn_w_0 = fn_w[0] if fn_w.ndim >= 2 else fn_w
        ref_h = rmsnorm(
            res1,
            fn_w_0,
            eps=float(g1.eps),
            zero_centered=bool(g1.zero_centered_norm),
        ).astype(BF16)
        ref_st_all = finalize_reference_step_state(
            next_step_state if isinstance(next_step_state, Mapping) else {},
            conv_in,
            rec_in,
            ngdn,
            vh,
            KD,
            VD,
            ex_d,
            pool_io=False,
        )

        pos_m, hid_m, w_m, st_m = w_mod.pack_spec_inputs(
            pos, hid, w_l, st_l, spec, ex_d, mesh=mesh
        )
        step_d, _ = kernel.make_step(mesh, spec, ex_d)
        out_h, out_st = jax.block_until_ready(step_d(pos_m, hid_m, w_m, st_m))
        tag = f"DirectHBM_T{T}"
        ref_b = (
            ref_h[:, :B]
            if ref_h.ndim == 3
            else jnp.broadcast_to(ref_h[:B][None, ...], (R, B, int(g1.H)))
        )
        u_h = kernel.unfold(out_h, R)[:, :B]
        ab, rl, ul, cs = err_metrics(u_h, ref_b)
        stats[0], stats[1] = max(stats[0], ab), max(stats[1], rl)
        stats[2], stats[3] = max(stats[2], ul), min(stats[3], cs)
        if not _allclose(u_h, ref_b, atol=h_atol, rtol=h_rtol):
            failed.append(f"{tag} hidden abs={ab:.4g} cos={cs:.6f}")
        elif cs < h_cos:
            failed.append(f"{tag} hidden cos={cs:.6f} < {h_cos}")
        m_dir: dict[str, Any] = {
            "hidden_max_abs": ab,
            "hidden_max_ulp": ul,
            "hidden_min_cos": cs,
        }
        per_mode_t[tag] = m_dir

        u_st = dict(mod.unpack_kernel_states(out_st, g1, ex_d))
        for k, v in out_st.items():
            if k not in u_st:
                u_st[k] = kernel.unfold(v, R)
        decls = _spec_state_decls(spec, g1, ex_d)
        ref_st = {k: v for k, v in ref_st_all.items() if k in decls or k in u_st}
        for sd in mod.outputs(g1, ex_d):
            if sd.name in u_st and sd.compare != "none" and sd.name not in ref_st:
                failed.append(f"{tag} reference/oracle missing output {sd.name!r}")
        if int(T) > 1:
            g_nxt = mod.next_logical_states(u_st, g1, ex_d)
            r_nxt = mod.next_logical_states(ref_st, g1, ex_d)
            for k in ("conv", "rec"):
                if k in r_nxt:
                    ref_st[f"{k}_final"] = r_nxt[k]
                if k in g_nxt:
                    u_st[f"{k}_final"] = g_nxt[k]
        _compare_state_dict(
            u_st,
            ref_st,
            tol=tol,
            is_approx=False,
            tag=tag,
            m_out=m_dir,
            failed=failed,
            state_decls=decls,
        )

    st: Literal["passed", "failed"] = "passed" if not failed else "failed"
    tols = {"hidden_atol": h_atol, "hidden_min_cos": h_cos}
    det, rsn = {"module": mod.name, "metrics": per_mode_t}, "; ".join(failed)
    return res_cls(check_name, st, *stats, tols, rsn, det)


def _resolve_verify_module(module: Module | ModuleBinding, g1: Any) -> Module:
    if isinstance(module, Module):
        return module
    if isinstance(module, ModuleBinding):
        opts = {k: v for k, v in module.options.items() if v is not None}
        return module.module_cls(geometry=g1, prefix=module.prefix, **opts)
    raise MegakernelConfigError(f"Expected Module or ModuleBinding, got {module!r}")


def verify_module(
    module: Module | ModuleBinding,
    g: Any,
    ex: ExecSpec | None = None,
    *,
    module_oracle: ModuleOracle | None = None,
    R: int = 2,
    B_per_T: int = 2,
    T_values: Sequence[int] = (1, 2),
    ctx: int = 256,
    seed: int = 0,
    tolerances: ToleranceSpec | None = None,
) -> VerificationReport:
    """Verify a single `Module` in isolation on `g` against `module.reference`
    or `module_oracle(params, x, state, *, g=g1, ex=ex_d, layer_idx=0, pos=pos)`.
    See `ModuleOracle` for the input/output composition and rounding contract.
    """
    from .verify import DEFAULT_TOLERANCES, CheckResult, VerificationReport, _safe_run

    tol = DEFAULT_TOLERANCES if tolerances is None else tolerances
    m_name = str(getattr(module, "name", getattr(module, "prefix", "module")))
    mod: Any = None
    g1: Any = None
    spec: Any = None
    t_v = tuple(int(t) for t in T_values)
    in_by_t: dict[int, tuple[Any, Any, Any, Any]] = {}

    def _run_setup() -> CheckResult:
        nonlocal m_name, mod, g1, spec
        # `ModelGeometry.check` requires `L % att_every == 0`, so reducing `L`
        # to `1` for single-layer module verification also sets `att_every=1`.
        g1_kw: dict[str, Any] = {"L": 1}
        if hasattr(g, "att_every"):
            g1_kw["att_every"] = 1
        g1 = dataclasses.replace(g, **g1_kw)
        mod = _resolve_verify_module(module, g1)
        m_name = mod.name
        spec = ModelSpec(
            name=mod.name,
            config=g1,
            layer_type=LayerType(name=mod.name, slots=(mod,)),
        )
        ex_t1 = _module_exec(ex, g1, R=R, B=max(1, B_per_T), T=1, ctx=ctx, interp=True)
        w_log = w_mod.random_spec_logical_weights(jax.random.key(seed), spec, ex_t1)
        for i, T in enumerate(t_v):
            ex_i = _module_exec(
                ex, g1, R=R, B=max(1, B_per_T * int(T)), T=int(T), ctx=ctx, interp=True
            )
            in_by_t[int(T)] = w_mod.random_spec_inputs(
                jax.random.key(seed + 100 + i), spec, ex_i, logical_weights=w_log
            )
        return CheckResult("setup", "passed", details={"T_values": t_v, "R": R})

    setup_res = _safe_run("setup", _run_setup)
    if setup_res.status != "passed":
        return VerificationReport(model_name=m_name, geometry=g1, checks=(setup_res,))

    checks = [
        setup_res,
        _safe_run(
            "derived_artifacts",
            lambda: check_module_derived_artifacts(
                spec,
                g1,
                ex,
                R=R,
                B_per_T=B_per_T,
                t_vals=t_v,
                ctx=ctx,
                res_cls=CheckResult,
            ),
        ),
    ]
    if module_oracle is None and not mod.has_reference:
        rsn_f = (
            f"Module {mod.name!r} implements no reference() "
            "and no module_oracle was provided"
        )
        checks.append(CheckResult("kernel_vs_reference", "failed", reason=rsn_f))
    else:
        chk_name = (
            "kernel_vs_oracle"
            if (module_oracle is not None and not mod.has_reference)
            else "kernel_vs_reference"
        )
        if module_oracle is not None and not mod.has_reference:
            rsn_s = (
                f"Module {mod.name!r} has no reference; verified via kernel_vs_oracle"
            )
            checks.append(CheckResult("kernel_vs_reference", "skipped", reason=rsn_s))
        mod_kw = dict(
            R=R,
            B_per_T=B_per_T,
            t_vals=t_v,
            ctx=ctx,
            in_by_t=in_by_t,
            tol=tol,
            module_oracle=module_oracle,
            check_name=chk_name,
            res_cls=CheckResult,
        )
        checks.append(
            _safe_run(chk_name, lambda: run_module_verify(mod, spec, g1, ex, **mod_kw))
        )
    return VerificationReport(model_name=m_name, geometry=g1, checks=tuple(checks))


__all__ = [
    "ModuleOracle",
    "check_module_derived_artifacts",
    "run_module_verify",
    "verify_module",
]
