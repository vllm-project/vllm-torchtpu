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
"""TPU_MINIMAX_M3_MSA_PREFILL_PRECISION: 4-bit MSA prefill attention with
fp8 levers against a float64 oracle (interpret).

Not bit-identical by design. Checks: the same -inf lse pattern, finite
outputs, and output / lse error versus the oracle within each lever's bound;
an empty spec is the incumbent bit for bit; decode calls (no page cache)
ignore the levers."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse, msa_fp4

INTERPRET = os.environ.get("INTERPRET", "1") == "1"
S, D, K, HG = 128, 128, 16, 16
ENV = "TPU_MINIMAX_M3_MSA_PREFILL_PRECISION"
# Relative rms output error bound vs the oracle (incumbent: bf16 rounding only).
BOUNDS = {"": 1e-2, "pv8": 4e-2, "qk8": 6e-2, "qk8,pv8": 8e-2,
          "qk8,msub": 6e-2, "qk8,unpk": 6e-2, "qk8,msub,unpk": 6e-2, "unpk": 1e-2,
          "qk8,e16": 6e-2, "qk8,msub,e16": 6e-2}
LSE_BOUNDS = {"": 1e-5, "pv8": 1e-5, "qk8": 0.3, "qk8,pv8": 0.3,
              "qk8,msub": 0.3, "qk8,unpk": 0.3, "qk8,msub,unpk": 0.3, "unpk": 1e-5,
              "qk8,e16": 0.3, "qk8,msub,e16": 0.3}
# Levers that must give their partner's bits ("unpk" unpacks the same values).
SAME_BITS = {"qk8,unpk": "qk8", "unpk": ""}


def make(T, seed, G, wide=False, qscale=1.5):
    rng = np.random.default_rng(seed)
    pool, nkeys = 64, 48
    sel = np.full((T, G, K), -1, np.int32)
    for t0 in range(0, T, 16):
        n = (t0 // 16) % 16 + 1
        if wide:
            n = 12 + 8 * ((t0 // 16) % 3)
        union = rng.choice(nkeys, size=n, replace=False)
        for t in range(t0, t0 + 16):
            for g in range(G):
                m = min(K, n, int(rng.integers(1, K + 1)))
                sel[t, g, :m] = rng.choice(union, size=m, replace=False)
    sel[-16:] = -1
    pos = rng.integers(0, 4000, size=T)
    local = sel[:, 0, 0].copy()
    ll = (pos % S + 1).astype(np.int32)
    pt = rng.permutation(pool)[:nkeys].astype(np.int32)
    q = jnp.asarray(rng.standard_normal((T, G * HG, D)) * qscale, jnp.bfloat16)
    nib = jnp.asarray(rng.integers(0, 16, size=(pool, S, G * 2 * D)), jnp.uint8)
    kv = msa_fp4.pack_rows(nib)
    scv = rng.uniform(0.02, 0.4, size=(pool, msa_fp4.scale_rows(G), S))
    scv[3, :, 5:9] = np.nan
    sc = jnp.asarray(scv, jnp.float32)
    return (q, kv, jnp.asarray(sel), jnp.asarray(local), jnp.asarray(ll), jnp.asarray(pt)), sc


def oracle(args, sc, G):
    q, kv, sel, lk, ll, pt = [np.asarray(jax.device_get(x)) for x in args]
    nib = np.asarray(jax.device_get(msa_fp4.unpack_rows(args[1]))).astype(np.int64)
    sgn = np.where(nib >= 8, nib - 16, nib).astype(np.float64)
    scv = np.asarray(sc, np.float64)
    scv = np.where(np.isfinite(scv), scv, 0.0)
    T = q.shape[0]
    out = np.zeros((G * HG, T, D))
    lse = np.full((G * HG, T), -np.inf)
    sm = 1 / np.sqrt(D)
    qf = q.astype(np.float64)
    for t in range(T):
        for g in range(G):
            ks, vs = [], []
            for key in np.unique(sel[t, g][sel[t, g] >= 0]):
                p = pt[key]
                n = ll[t] if key == lk[t] else S
                ks.append(sgn[p, :n, g * 2 * D : g * 2 * D + D] * scv[p, 2 * g, :n, None])
                vs.append(sgn[p, :n, g * 2 * D + D : (g + 1) * 2 * D] * scv[p, 2 * g + 1, :n, None])
            if not ks:
                continue
            kk, vv = np.concatenate(ks), np.concatenate(vs)
            for h in range(HG):
                s = kk @ qf[t, g * HG + h] * sm
                m = s.max()
                e = np.exp(s - m)
                out[g * HG + h, t] = e @ vv / e.sum()
                lse[g * HG + h, t] = m + np.log(e.sum())
    return out, lse


def run(args, sc, precision, **kw):
    base = dict(sm_scale=1 / np.sqrt(D), block_q=16, pages_per_step=16,
                interpret=INTERPRET, head_major=True, prefill_precision=precision)
    base.update(kw)
    o, l = msa_attention_lse.msa_sparse_attention_lse_heads(*args, scale_cache=sc, **base)
    return np.asarray(o.astype(jnp.float32)), np.asarray(l)


def errors(new, ora):
    (o1, l1), (oo, lo) = new, ora
    fin = np.isfinite(lo)
    assert np.isfinite(o1).all()
    assert np.array_equal(fin, np.isfinite(l1))
    rel = np.sqrt(np.mean((o1 - oo) ** 2)) / np.sqrt(np.mean(oo**2))
    lmax = float(np.max(np.abs(np.where(fin, l1, 0) - np.where(fin, lo, 0))))
    return float(rel), lmax


CASES = (
    (2, 128, False, dict(page_cache_slots=512)),
    # c64 head-parallel config (8:64::1024, 16 query heads a tile)
    (1, 128, True, dict(page_cache_slots=1024, block_q=8, pages_per_step=64)),
    (2, 96, True, dict(page_cache_slots=512, pages_per_step=32,
                       step_widths=(2, 4, 8, 12, 16, 20, 24))),
)


@pytest.mark.parametrize("case", range(len(CASES)))
def test_levers_vs_oracle(case):
    G, T, wide, kw = CASES[case]
    args, sc = make(T, 11 + T + G, G, wide)
    ora = oracle(args, sc, G)
    inc = run(args, sc, "", **kw)
    got = {}
    for spec in BOUNDS:
        new = run(args, sc, spec, **kw)
        got[spec] = errors(new, ora)
        print(f"case {case} {spec or 'incumbent'}: out rel rms {got[spec][0]:.3e} "
              f"lse max {got[spec][1]:.3e}")
        if not spec:
            for a, b in zip(inc, new):
                np.testing.assert_array_equal(a, b)
    outs = {spec: run(args, sc, spec, **kw) for spec in SAME_BITS}
    for spec, ref in SAME_BITS.items():
        for a, b in zip(outs[spec], run(args, sc, ref, **kw)):
            np.testing.assert_array_equal(a, b)
    for spec, bound in BOUNDS.items():
        rel, lmax = got[spec]
        assert rel <= bound, spec
        assert lmax <= LSE_BOUNDS.get(spec, 3e-2), spec


@pytest.mark.parametrize("qscale", (8.0, 12.0))
def test_msub_sharp_scores(qscale):
    """Sharper scores (large q): msub's shift estimate stays finite and close.
    (At q x 24 the row lse reaches ~240 log2 units above the first page's
    estimate and exp2 overflows: msub assumes the qk-normed served range.)"""
    G, T = 1, 64
    args, sc = make(T, 7, G, True, qscale=qscale)
    ora = oracle(args, sc, G)
    kw = dict(page_cache_slots=1024, block_q=8, pages_per_step=64)
    base = errors(run(args, sc, "qk8", **kw), ora)
    new = errors(run(args, sc, "qk8,msub", **kw), ora)
    print(f"q x{qscale}: qk8 {base} qk8,msub {new}")
    assert new[0] <= 1.5 * base[0] + 1e-3
    assert new[1] <= 1.5 * base[1] + 1e-3


def test_decode_ignores_levers():
    G, T = 2, 32
    args, sc = make(T, 5, G)
    a = run(args, sc, "")
    b = run(args, sc, "qk8,pv8")
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_bad_lever():
    with pytest.raises(ValueError):
        msa_attention_lse.prefill_precision_levers("fp4")
