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
"""TPU_MINIMAX_M3_MSA_PREFILL_FP8_OPT: fp8 MSA prefill attention (transposed
body, block_q 16 / 32) with the "exp16" lever against a float64 oracle
(interpret), and the "bias" and "lanemask" page masks bit for bit against the select.

Checks: no lever is the incumbent bit for bit; "exp16" keeps the -inf lse
pattern, finite outputs and errors within a bf16-level bound; unknown levers
raise."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse

INTERPRET = os.environ.get("INTERPRET", "1") == "1"
S, D, K, HG = 128, 128, 16, 16


def make(T, seed):
    rng = np.random.default_rng(seed)
    pool, nkeys = 64, 48
    sel = np.full((T, K), -1, np.int32)
    for t0 in range(0, T, 16):
        union = rng.choice(nkeys, size=(t0 // 16) % 12 + 4, replace=False)
        for t in range(t0, t0 + 16):
            m = int(rng.integers(1, len(union) + 1))
            sel[t, :m] = rng.choice(union, size=m, replace=False)
    sel[-8:] = -1
    pos = rng.integers(0, 4000, size=T)
    local = sel[:, 0].copy()
    ll = (pos % S + 1).astype(np.int32)
    pt = rng.permutation(pool)[:nkeys].astype(np.int32)
    q = jnp.asarray(rng.standard_normal((HG, T, D)) * 1.5, jnp.bfloat16)
    x = rng.standard_normal((pool, S, 2 * D)) * 0.7
    kv = jnp.asarray(x, jnp.float32).astype(jnp.float8_e4m3fn).reshape(pool, S // 4, 4, 2 * D)
    return q, kv, jnp.asarray(sel), jnp.asarray(local), jnp.asarray(ll), jnp.asarray(pt)


def oracle(q, kv, sel, local, ll, pt):
    q = np.asarray(q.astype(jnp.float32), np.float64)  # [H, T, D]
    rows = np.asarray(kv.astype(jnp.float32), np.float64).reshape(kv.shape[0], S, 2 * D)
    sel, local, ll, pt = map(np.asarray, (sel, local, ll, pt))
    H, T, _ = q.shape
    out = np.zeros((H, T, D)); lse = np.full((H, T), -np.inf)
    for t in range(T):
        keys = np.unique(sel[t][sel[t] >= 0])
        if keys.size == 0:
            continue
        ks, vs = [], []
        for k in keys:
            lim = ll[t] if k == local[t] else S
            ks.append(rows[pt[k], :lim, :D]); vs.append(rows[pt[k], :lim, D:])
        kk = np.concatenate(ks); vv = np.concatenate(vs)
        s = q[:, t] @ kk.T / np.sqrt(D)
        m = s.max(axis=1, keepdims=True)
        p = np.exp(s - m)
        out[:, t] = (p @ vv) / p.sum(axis=1, keepdims=True)
        lse[:, t] = m[:, 0] + np.log(p.sum(axis=1))
    return out, lse


def run(args, block_q, spec):
    return msa_attention_lse.msa_sparse_attention_lse(
        *args, sm_scale=1.0 / np.sqrt(D), block_q=block_q, pages_per_step=16,
        head=0, head_major=True, q_head_major=True, interpret=INTERPRET,
        prefill_precision=spec or None,
    )


@pytest.mark.parametrize("block_q", [16, 32])
def test_exp16_close_and_off_is_incumbent(block_q):
    args = make(64, seed=block_q)
    o0, l0 = run(args, block_q, "")
    ref_o, ref_l = oracle(*args)
    fin = np.isfinite(ref_l)
    assert np.array_equal(np.isfinite(np.asarray(l0)), fin)
    o1, l1 = run(args, block_q, "exp16")
    o1 = np.asarray(o1, np.float64); l1 = np.asarray(l1, np.float64)
    assert np.array_equal(np.isfinite(l1), fin)
    assert np.all(np.isfinite(o1))
    rms = lambda a: np.sqrt(np.mean(a ** 2))
    err0 = rms(np.asarray(o0, np.float64) - ref_o) / rms(ref_o)
    err1 = rms(o1 - ref_o) / rms(ref_o)
    assert err1 < max(2e-2, 3 * err0), (err0, err1)
    assert np.max(np.abs(np.where(fin, l1 - ref_l, 0))) < 5e-2


@pytest.mark.parametrize("block_q", [16, 32])
@pytest.mark.parametrize("other", ["", "exp16"])
@pytest.mark.parametrize("lever", ["bias", "lanemask"])
def test_mask_levers_are_bitwise(block_q, other, lever):
    args = make(64, seed=7 + block_q)
    o0, l0 = run(args, block_q, other)
    o1, l1 = run(args, block_q, ",".join(x for x in (other, lever) if x))
    assert np.array_equal(np.asarray(o0), np.asarray(o1))
    assert np.array_equal(np.asarray(l0), np.asarray(l1))


@pytest.mark.parametrize("block_q", [16, 32])
@pytest.mark.parametrize("lever", ["vt", "lmxu", "exp16,lmxu", "mdot", "lmxu,mdot"])
def test_vt_close(block_q, lever):
    args = make(64, seed=11 + block_q)
    o0, l0 = run(args, block_q, "")
    o1, l1 = run(args, block_q, lever)
    fin = np.isfinite(np.asarray(l0))
    assert np.max(np.abs(np.where(fin, np.asarray(l1) - np.asarray(l0), 0))) < 2e-2
    o0 = np.asarray(o0, np.float64); o1 = np.asarray(o1, np.float64)
    assert np.array_equal(np.isfinite(np.asarray(l0)), np.isfinite(np.asarray(l1)))
    assert np.sqrt(np.mean((o1 - o0) ** 2)) / np.sqrt(np.mean(o0 ** 2)) < 1e-2


@pytest.mark.parametrize("lever", ["lmxu", "lmxu,mdot"])
@pytest.mark.parametrize("q_scale", [1.0, 8.0])
def test_mdot_against_oracle(lever, q_scale):
    """"mdot" shifts by the first page's raw max (+ headroom), not the running
    max: still the float64 oracle within bf16 bounds, also on sharp scores
    (queries x8: row maxima tens of log2 units above the first page's)."""
    q, *rest = make(64, seed=23)
    args = (q * q_scale, *rest)
    ref_o, ref_l = oracle(*args)
    fin = np.isfinite(ref_l)
    o0, l0 = run(args, 32, "lmxu")
    o1, l1 = run(args, 32, lever)
    o1 = np.asarray(o1, np.float64); l1 = np.asarray(l1, np.float64)
    assert np.array_equal(np.isfinite(l1), fin)
    assert np.all(np.isfinite(o1))
    rms = lambda a: np.sqrt(np.mean(a ** 2))
    err0 = rms(np.asarray(o0, np.float64) - ref_o) / rms(ref_o)
    err1 = rms(o1 - ref_o) / rms(ref_o)
    assert err1 < max(2e-2, 2 * err0), (err0, err1)
    assert np.max(np.abs(np.where(fin, l1 - ref_l, 0))) < 5e-2


def test_unknown_lever_raises():
    with pytest.raises(ValueError):
        msa_attention_lse.fp8_prefill_levers("qk9")
    assert msa_attention_lse.fp8_prefill_levers("") == frozenset()
