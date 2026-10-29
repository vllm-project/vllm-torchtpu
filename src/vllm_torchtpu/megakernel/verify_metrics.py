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
"""Numerical error metrics and state-comparison helpers for `verify_model`."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

from .exec_spec import ExecSpec
from .ops import F32
from .spec import ModelSpec, StateDecl


@dataclass(frozen=True)
class ToleranceSpec:
    """Numerical tolerance specification for `verify_model`.

    Per-field comparison vs `tests/megakernel/test_model_def.py:979-998` and
    `tests/megakernel/test_m3_cache_eplb.py`:
    - `hidden`: `atol=4e-2, rtol=2e-2, min_cos=0.9999` (`atol=4e-2` tightens
      `test_model_def.py`'s `5e-2`; `rtol=2e-2` is looser than `1e-2` for
      near-zero elements; seed-0 measured `max_abs`: primary hybrid model
      `0.039` at `T=4` / `0.019` at `T=1`, `SynthHybrid` `0.0156`).
    - `kv`: `atol=3.5e-2, rtol=1.5e-2` (looser than `test_model_def.py`'s
      `2e-2/1e-2` at `T=1` because `DirectHBM` at `T=4` has `kc` `max_abs =
      0.03125 = 1` bf16 ULP).
    - `paged_kv`: `atol=2.5e-1, rtol=1.5e-1` (matches `test_m3_cache_eplb.py`
      for FP8 `PallasWriter`, 1 FP8 ULP).
    - `conv`: `atol=1e-2, rtol=1e-2` (matches `test_model_def.py`).
    - `rec`: `atol=5e-2, rtol=2e-2` (matches `test_model_def.py`).
    - `approx_*`: governs `approx_profiles`; `hf_hidden_*`: `atol=4e-2,
      rtol=2e-2, min_cos=0.9999` (measured `max_abs ~0.0156`, `min_cos
      ~0.999998`); `max_ulp_diff` is in relative bf16 steps
      (`ldexp(1.0, floor(log2(max(|got|, |ref|, 1.0))) - 7)`).
    """

    hidden_atol: float = 4e-2
    hidden_rtol: float = 2e-2
    hidden_min_cos: float = 0.9999
    kv_atol: float = 3.5e-2
    kv_rtol: float = 1.5e-2
    paged_kv_atol: float = 2.5e-1
    paged_kv_rtol: float = 1.5e-1
    conv_atol: float = 1e-2
    conv_rtol: float = 1e-2
    rec_atol: float = 5e-2
    rec_rtol: float = 2e-2
    approx_hidden_atol: float = 4e-2
    approx_hidden_rtol: float = 2e-2
    approx_hidden_min_cos: float = 0.9999
    approx_kv_atol: float = 3.5e-2
    approx_kv_rtol: float = 1.5e-2
    approx_paged_kv_atol: float = 2.5e-1
    approx_paged_kv_rtol: float = 1.5e-1
    approx_conv_atol: float = 1e-2
    approx_conv_rtol: float = 1e-2
    approx_rec_atol: float = 5e-2
    approx_rec_rtol: float = 2e-2
    hf_hidden_atol: float = 4e-2
    hf_hidden_rtol: float = 2e-2
    hf_hidden_min_cos: float = 0.9999


DEFAULT_TOLERANCES = ToleranceSpec()


@dataclass(frozen=True)
class CheckResult:
    """Outcome and measured error metrics for a single verification check."""

    name: str
    status: Literal["passed", "failed", "skipped"]
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    max_ulp_diff: int = 0
    min_cos: float = 1.0
    tolerance: Mapping[str, float | int] = dataclasses.field(default_factory=dict)
    reason: str = ""
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class VerificationReport:
    """Structured report returned by `verify_model` and `verify_module`."""

    model_name: str
    geometry: Any
    checks: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(c.status in ("passed", "skipped") for c in self.checks)

    def assert_passed(self, *, allow_skipped: bool = True) -> None:
        ok_st = ("passed", "skipped") if allow_skipped else ("passed",)
        bad = [c for c in self.checks if c.status not in ok_st]
        if bad:
            pts = [f"{c.name}({c.status}: {c.reason!r})" for c in bad]
            raise AssertionError(f"verify_model({self.model_name!r}): {pts}")


def err_metrics(got: Any, ref: Any) -> tuple[float, float, int, float]:
    g_f = np.asarray(jnp.asarray(got).astype(F32)).ravel()
    r_f = np.asarray(jnp.asarray(ref).astype(F32)).ravel()
    if g_f.size == 0:
        return 0.0, 0.0, 0, 1.0
    diff = np.abs(g_f - r_f)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff / np.maximum(np.abs(r_f), 1.0)))
    scale = np.maximum(np.maximum(np.abs(g_f), np.abs(r_f)), 1.0)
    ulp_step = np.ldexp(1.0, np.floor(np.log2(scale)).astype(np.int32) - 7)
    ulp = int(np.max(np.rint(diff / ulp_step))) if max_abs > 0.0 else 0
    gn, rn = float(np.linalg.norm(g_f)), float(np.linalg.norm(r_f))
    dot = float(np.dot(g_f, r_f))
    cos = 1.0 if gn == 0.0 and rn == 0.0 else dot / max(gn * rn, 1e-12)
    return max_abs, max_rel, ulp, cos


def _allclose(got: Any, ref: Any, *, atol: float, rtol: float) -> bool:
    g_f, r_f = jnp.asarray(got).astype(F32), jnp.asarray(ref).astype(F32)
    return g_f.shape == r_f.shape and bool(np.allclose(g_f, r_f, atol=atol, rtol=rtol))


def _spec_state_decls(spec: ModelSpec, g: Any, ex: ExecSpec) -> dict[str, StateDecl]:
    return {
        sd.name: sd
        for mod in spec.all_modules()
        for sd in (*mod.states(g, ex), *mod.outputs(g, ex))
    }


def _check_finite(val: Any) -> bool:
    arr = jnp.asarray(val)
    return not jnp.issubdtype(arr.dtype, jnp.floating) or bool(
        np.all(np.isfinite(np.asarray(arr, dtype=np.float32)))
    )


def _compare_state_dict(
    got_st: Mapping[str, Any],
    ref_st: Mapping[str, Any],
    *,
    tol: ToleranceSpec,
    is_approx: bool,
    paged_kv: bool = False,
    tag: str,
    m_out: dict[str, Any],
    failed: list[str],
    state_decls: Mapping[str, StateDecl] | None = None,
) -> None:
    pfx = "approx_" if is_approx else ""
    for sk, sd in (state_decls or {}).items():
        if sk in got_st and sk not in ref_st and sd.compare == "none":
            if not _check_finite(got_st[sk]):
                failed.append(f"{tag} {sk} non-finite")
            m_out[f"{sk}_max_abs"] = 0.0
    for sk, r_v in ref_st.items():
        if sk not in got_st:
            failed.append(f"{tag} missing state key {sk!r}")
            continue
        g_v = got_st[sk]
        if tuple(g_v.shape) != tuple(r_v.shape):
            failed.append(f"{tag} {sk} shape {tuple(g_v.shape)} != {tuple(r_v.shape)}")
            continue
        sd = None if state_decls is None else state_decls.get(sk)
        cmp_mode = "close" if sd is None else sd.compare
        if cmp_mode == "none":
            if not _check_finite(g_v):
                failed.append(f"{tag} {sk} non-finite")
            m_out[f"{sk}_max_abs"] = 0.0
        elif cmp_mode == "exact":
            eq = bool(np.array_equal(np.asarray(g_v), np.asarray(r_v)))
            g_64 = np.asarray(g_v, dtype=np.float64)
            r_64 = np.asarray(r_v, dtype=np.float64)
            s_ab = 0.0 if eq else float(np.max(np.abs(g_64 - r_64)))
            m_out[f"{sk}_max_abs"] = s_ab
            if not eq:
                failed.append(f"{tag} {sk} exact mismatch (max_diff={s_ab})")
        else:
            s_ab, _, _, s_cs = err_metrics(g_v, r_v)
            m_out[f"{sk}_max_abs"] = s_ab
            def_h_cos = getattr(tol, f"{pfx}hidden_min_cos")
            if cmp_mode == "cos":
                min_c = def_h_cos if (sd is None or sd.min_cos is None) else sd.min_cos
                m_out[f"{sk}_min_cos"] = s_cs
                if s_cs < min_c:
                    failed.append(f"{tag} {sk} cos={s_cs:.6f} < {min_c}")
                continue
            is_tap = sd is not None and sd.role in ("tap", "residual")
            kv_k = "paged_kv" if paged_kv else "kv"
            k_nm = (
                "hidden"
                if is_tap
                else (
                    "conv"
                    if sk.startswith("conv")
                    else ("rec" if sk.startswith("rec") else kv_k)
                )
            )
            def_cos = def_h_cos if is_tap else None
            satol = (
                getattr(tol, f"{pfx}{k_nm}_atol")
                if (sd is None or sd.atol is None)
                else sd.atol
            )
            srtol = (
                getattr(tol, f"{pfx}{k_nm}_rtol")
                if (sd is None or sd.rtol is None)
                else sd.rtol
            )
            s_min_cos = def_cos if (sd is None or sd.min_cos is None) else sd.min_cos
            if not _allclose(g_v, r_v, atol=satol, rtol=srtol):
                failed.append(f"{tag} {sk} abs={s_ab:.4g} > {satol}")
            elif s_min_cos is not None and s_cs < s_min_cos:
                failed.append(f"{tag} {sk} cos={s_cs:.6f} < {s_min_cos}")
