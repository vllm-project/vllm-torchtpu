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
"""`reference_vs_hf` check: `ModelDef.reference_step` vs an injectable `oracle`.

The `oracle(weights, inputs) -> (hidden, states)` callable may lazily import
optional external frameworks (such as `torch` and `transformers` in a
model-specific HF oracle under `models/`). Only `ImportError` raised by the
oracle yields `status="skipped"`; all other errors fail closed with
`status="failed"`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from . import kernel
from . import weights as w_mod
from .exec_spec import ExecSpec
from .geometry import ModelGeometry
from .model_def import ModelDef
from .ops import BF16, I32
from .pool.config import with_cache
from .profile import Profile
from .spec import SwitchSlot
from .verify_checks import (
    VerifyCtx,
    _allclose,
    _compare_state_dict,
    _spec_state_decls,
    err_metrics,
    prof_exec,
)
from .verify_metrics import CheckResult, ToleranceSpec
from .weights.logical import (
    resolve_spec_and_norms,
    switch_branch_for_layer,
)


@dataclass(frozen=True)
class OracleInputs(Mapping[str, Any]):
    """Step inputs passed to an injectable `oracle(weights, inputs)` callable."""

    pos: jax.Array
    hidden: jax.Array
    states: Mapping[str, Any]
    g: ModelGeometry
    ex: ExecSpec
    w_logical: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)


OracleCallable = Callable[[Mapping[str, Any], OracleInputs], tuple[Any, Any]]


def _zero_logical_states(
    m_def: type[ModelDef], g: ModelGeometry, ex: ExecSpec
) -> tuple[list[tuple[Any, int]], dict[str, jax.Array]]:
    ms, geom, _, _, _ = resolve_spec_and_norms(m_def, ex, g=g)
    counts: dict[str, int] = {m.name: 0 for m in ms.all_modules()}
    for slot in ms.layer_type.slots:
        if isinstance(slot, SwitchSlot):
            for lyr in range(int(geom.L)):
                b_mod, _ = switch_branch_for_layer(slot, lyr)
                counts[b_mod.name] = counts.get(b_mod.name, 0) + 1
        else:
            counts[slot.name] = counts.get(slot.name, 0) + int(geom.L)
    active_mods = [
        (m, counts[m.name]) for m in ms.all_modules() if counts.get(m.name, 0) > 0
    ]
    st: dict[str, jax.Array] = {}
    k0 = jax.random.key(0)
    for mod, n_l in active_mods:
        for k, v in mod.init_logical_states(k0, geom, ex, n_l).items():
            st[k] = jnp.zeros_like(v, dtype=BF16)
    return active_mods, st


def _check_oracle_states(
    ref_st: Mapping[str, Any],
    ora_st: Any,
    label_prefix: str,
    per_step: dict[str, Any],
) -> tuple[float, float, int, float]:
    if not isinstance(ora_st, Mapping):
        raise TypeError(
            f"Oracle states for {label_prefix} must be a Mapping, "
            f"got {type(ora_st).__name__}"
        )
    max_abs, max_rel, max_ulp, min_cos = 0.0, 0.0, 0, 1.0
    for k, ora_v in ora_st.items():
        if str(k).startswith("_"):
            continue
        if k not in ref_st:
            raise KeyError(f"Oracle returned unknown state key {k!r}")
        ref_arr = np.asarray(ref_st[k])
        ora_arr = np.asarray(ora_v)
        if ref_arr.shape != ora_arr.shape:
            raise ValueError(
                f"Oracle state {k!r} shape {ora_arr.shape} != {ref_arr.shape}"
            )
        ab, rl, ul, cs = err_metrics(ref_st[k], ora_v)
        max_abs, max_rel = max(max_abs, ab), max(max_rel, rl)
        max_ulp, min_cos = max(max_ulp, ul), min(min_cos, cs)
        per_step[f"{label_prefix}_state_{k}"] = dict(max_abs=ab, max_ulp=ul, min_cos=cs)
    return max_abs, max_rel, max_ulp, min_cos


def verify_reference_vs_hf(
    m_def: type[ModelDef],
    g: ModelGeometry,
    ex: ExecSpec,
    w_hf_q: Mapping[str, Any],
    w_log: Mapping[str, Any],
    *,
    oracle: OracleCallable | None = None,
    tolerances: ToleranceSpec,
    seed: int = 0,
    num_steps: int = 3,
) -> CheckResult:
    """Compare `ModelDef.reference_step` vs `oracle(weights, inputs)` from empty cache.

    Only `ImportError` raised by `oracle` yields `status="skipped"`. All other
    errors (including a missing `oracle`, unpopulated parameters, or shape
    mismatches) fail closed with `status="failed"`.
    """
    eff_oracle = oracle if oracle is not None else getattr(m_def, "hf_oracle", None)
    if eff_oracle is None or not callable(eff_oracle):
        return CheckResult(
            name="reference_vs_hf",
            status="failed",
            reason=f"ModelDef {m_def.name!r} defines no callable hf_oracle",
        )

    try:
        N = int(num_steps)
        xs_j = (
            jax.random.normal(jax.random.key(seed + 99), (N, int(g.H)), dtype=BF16)
            * 0.5
        )
        ex_1 = dataclasses.replace(ex, B=1, T=1) if (ex.B != 1 or ex.T != 1) else ex
        ex_n = dataclasses.replace(ex_1, B=N, T=N)
        pos_n = jnp.arange(N, dtype=I32)
        _, st_n = _zero_logical_states(m_def, g, ex_n)
        inputs_n = OracleInputs(
            pos=pos_n, hidden=xs_j, states=st_n, g=g, ex=ex_n, w_logical=w_log
        )
        ora_seq_raw, ora_st_n = eff_oracle(w_hf_q, inputs_n)
    except ImportError as exc:
        return CheckResult(
            name="reference_vs_hf",
            status="skipped",
            reason=f"torch/transformers unavailable: {exc}",
        )
    except Exception as exc:
        return CheckResult(
            "reference_vs_hf", "failed", reason=f"{type(exc).__name__}: {exc}"
        )

    try:
        ora_seq = np.asarray(ora_seq_raw)
        if ora_seq.shape != (N, int(g.H)):
            raise ValueError(
                f"Oracle multi-token hidden shape {ora_seq.shape} != {(N, int(g.H))}"
            )
        carry: dict[str, Any] = (
            {k: v for k, v in ora_st_n.items() if str(k).startswith("_")}
            if isinstance(ora_st_n, Mapping)
            else {}
        )
        active_mods, st = _zero_logical_states(m_def, g, ex_1)
        max_abs, max_rel, max_ulp, min_cos = 0.0, 0.0, 0, 1.0
        per_step: dict[str, Any] = {}

        out_hn, ref_st_n = m_def.reference_step(
            g, ex_n, pos_n, xs_j, w_log, st_n, paged_kv=False
        )
        ab_n, rl_n, ul_n, cs_n = err_metrics(out_hn[:N], ora_seq[:N])
        max_abs, max_rel = max(max_abs, ab_n), max(max_rel, rl_n)
        max_ulp, min_cos = max(max_ulp, ul_n), min(min_cos, cs_n)
        per_step[f"multi_token_T{N}"] = dict(max_abs=ab_n, max_ulp=ul_n, min_cos=cs_n)
        s_ab, s_rl, s_ul, s_cs = _check_oracle_states(
            ref_st_n, ora_st_n, f"multi_token_T{N}", per_step
        )
        max_abs, max_rel = max(max_abs, s_ab), max(max_rel, s_rl)
        max_ulp, min_cos = max(max_ulp, s_ul), min(min_cos, s_cs)

        for t in range(N):
            pos_t = jnp.array([t], dtype=I32)
            x_t = xs_j[t : t + 1]
            out_h, out_st = m_def.reference_step(
                g, ex_1, pos_t, x_t, w_log, st, paged_kv=False
            )
            inputs_t = OracleInputs(
                pos=pos_t,
                hidden=x_t,
                states={**st, **carry},
                g=g,
                ex=ex_1,
                w_logical=w_log,
            )
            ora_h_t, ora_st_t = eff_oracle(w_hf_q, inputs_t)
            ora_h_arr = np.asarray(ora_h_t)
            if ora_h_arr.shape == (1, int(g.H)):
                ora_h_arr = ora_h_arr[0]
            if ora_h_arr.shape != (int(g.H),):
                raise ValueError(
                    f"Oracle step {t} hidden shape {np.asarray(ora_h_t).shape} "
                    f"!= {(1, int(g.H))} or {(int(g.H),)}"
                )
            for label, hf_ref in (("seq", ora_seq[t]), ("cached", ora_h_arr)):
                ab, rl, ul, cs = err_metrics(out_h[0], hf_ref)
                max_abs, max_rel = max(max_abs, ab), max(max_rel, rl)
                max_ulp, min_cos = max(max_ulp, ul), min(min_cos, cs)
                per_step[f"step{t}_{label}"] = dict(max_abs=ab, max_ulp=ul, min_cos=cs)
            s_ab, s_rl, s_ul, s_cs = _check_oracle_states(
                out_st, ora_st_t, f"step{t}", per_step
            )
            max_abs, max_rel = max(max_abs, s_ab), max(max_rel, s_rl)
            max_ulp, min_cos = max(max_ulp, s_ul), min(min_cos, s_cs)
            if isinstance(ora_st_t, Mapping):
                carry = {k: v for k, v in ora_st_t.items() if str(k).startswith("_")}
            st_next: dict[str, Any] = {}
            for mod, _ in active_mods:
                for k, v in mod.next_logical_states(out_st, g, ex_1).items():
                    st_next[k] = v.astype(BF16)
            st = st_next
    except Exception as exc:
        return CheckResult(
            "reference_vs_hf", "failed", reason=f"{type(exc).__name__}: {exc}"
        )

    ok = (
        max_abs <= tolerances.hf_hidden_atol or max_rel <= tolerances.hf_hidden_rtol
    ) and min_cos >= tolerances.hf_hidden_min_cos
    return CheckResult(
        name="reference_vs_hf",
        status="passed" if ok else "failed",
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        max_ulp_diff=max_ulp,
        min_cos=min_cos,
        tolerance={
            "hf_hidden_atol": tolerances.hf_hidden_atol,
            "hf_hidden_rtol": tolerances.hf_hidden_rtol,
            "hf_hidden_min_cos": tolerances.hf_hidden_min_cos,
        },
        reason=""
        if ok
        else f"reference_vs_hf max_abs={max_abs:.4g}, min_cos={min_cos:.6f}",
        details=per_step,
    )


def run_kernel_vs_oracle(
    vc: VerifyCtx,
    prof: Profile,
    w_hf_q: Mapping[str, Any] | None,
    oracle: Callable[..., tuple[Any, Any]],
    res_cls: type[CheckResult],
) -> CheckResult:
    """Compare `make_step` against `oracle(weights, inputs)` across `T_values`
    using decode-step inputs (`pos`, `hidden`, and random non-zero `states`).
    """
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
    all_have_hf = all(m.has_hf_params for m in spec.all_modules())
    _, _, pre_n, post_n, final_n = resolve_spec_and_norms(m_def, None, g=g)
    mesh = Mesh(mesh_utils.create_device_mesh((R,), devices=devs[:R]), ("tp",))
    stats, per_mode_t, failed = [0.0, 0.0, 0, 1.0], {}, []
    ora_weights: Mapping[str, Any] = (
        w_hf_q if (all_have_hf and w_hf_q is not None) else {}
    )

    for T in vc.t_vals:
        B = max(1, vc.B_per_T * int(T))
        ex_b = prof_exec(prof, g, R=R, B=B, T=int(T), ctx=vc.ctx, interp=True)
        ex_d = with_cache(ex_b, paged_kv=False, fused_pool_io=False, pool_gather=False)
        pos, hid, w_l, st_l = vc.inputs_by_t[int(T)]
        ora_in = OracleInputs(
            pos=pos, hidden=hid, states=st_l, g=g, ex=ex_d, w_logical=w_l
        )
        try:
            ref_h, ref_st_raw = oracle(ora_weights, ora_in)
        except ImportError as exc:
            return res_cls(
                "kernel_vs_oracle",
                "skipped",
                reason=f"oracle dependency unavailable: {exc}",
            )
        ref_st = dict(ref_st_raw) if isinstance(ref_st_raw, Mapping) else {}
        ref_st = {k: v for k, v in ref_st.items() if not str(k).startswith("_")}
        if all_have_hf:
            pos_m, hid_m, w_m, st_m = w_mod.pack_inputs(
                pos, hid, w_l, st_l, g, ex_d, mesh=mesh, model_def=m_def
            )
        else:
            pos_m, hid_m, w_m, st_m = w_mod.pack_spec_inputs(
                pos,
                hid,
                w_l,
                st_l,
                spec,
                ex_d,
                mesh=mesh,
                pre_norm=pre_n,
                post_norm=post_n,
                final_norm=final_n,
            )
        step_d, _ = kernel.make_step(mesh, spec, ex_d)
        out_h, out_st = step_d(pos_m, hid_m, w_m, st_m)
        tag = f"DirectHBM_T{T}"
        ref_h_arr = jnp.asarray(ref_h)
        ref_b = (
            ref_h_arr[:, :B]
            if ref_h_arr.ndim == 3
            else jnp.broadcast_to(ref_h_arr[:B][None, ...], (R, B, int(g.H)))
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
        u_st: dict[str, Any] = {}
        for mod in spec.all_modules():
            u_st.update(mod.unpack_kernel_states(out_st, g, ex_d))
        for k, v in out_st.items():
            if k not in u_st:
                u_st[k] = kernel.unfold(v, R)
        _compare_state_dict(
            u_st,
            ref_st,
            tol=tol,
            is_approx=is_approx,
            tag=tag,
            m_out=m_dir,
            failed=failed,
            state_decls=_spec_state_decls(spec, g, ex_d),
        )

    st: Literal["passed", "failed"] = "passed" if not failed else "failed"
    tols = {"hidden_atol": h_atol, "hidden_min_cos": h_cos}
    det, rsn = {"profile": prof.name, "metrics": per_mode_t}, "; ".join(failed)
    return res_cls("kernel_vs_oracle", st, *stats, tols, rsn, det)


__all__ = [
    "OracleCallable",
    "OracleInputs",
    "run_kernel_vs_oracle",
    "verify_reference_vs_hf",
]
