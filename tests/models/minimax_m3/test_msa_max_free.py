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
"""TPU_MINIMAX_M3_MSA_MAX_FREE (+ TPU_MINIMAX_M3_MSA_MXU_MASK): MSA attention
softmax with no running max, against a float64 oracle (interpret) and against
the running-max kernel, on fp8 / bf16 / 4-bit caches, transposed (prefill)
and plain (decode, 4-bit) bodies, single- and multi-group tiles, and composed
with the fp8 FP8_OPT levers and the 4-bit page cache / PRECISION levers.

Checks: the same -inf lse pattern, finite outputs, errors within a bf16-level
bound; scores near the bound stay finite; "pv8" calls keep the running max
(bit for bit); the bound computation and its refusal past 75; off adds no
lever to any kernel."""

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse, msa_fp4

INTERPRET = os.environ.get("INTERPRET", "1") == "1"
S, D, K, HG = 128, 128, 16, 16


@pytest.fixture
def max_free(monkeypatch):
    """``set(on, mxu)``: the configured state, read when a program traces
    (the jitted entry points cache on statics only, so caches are cleared)."""

    def set_state(on, mxu=False):
        monkeypatch.setattr(msa_attention_lse, "MAX_FREE", on)
        monkeypatch.setattr(msa_attention_lse, "MXU_MASK", on and mxu)
        jax.clear_caches()

    yield set_state
    jax.clear_caches()


# ------------------------------------------------------------ 8/16-bit caches
def make(T, seed, dtype=jnp.float8_e4m3fn, q_gain=1.5, union_max=12):
    rng = np.random.default_rng(seed)
    pool, nkeys = 64, 48
    sel = np.full((T, K), -1, np.int32)
    for t0 in range(0, T, 16):
        union = rng.choice(nkeys, size=(t0 // 16) % union_max + 4, replace=False)
        for t in range(t0, t0 + 16):
            m = int(rng.integers(1, min(K, len(union)) + 1))
            sel[t, :m] = rng.choice(union, size=m, replace=False)
    sel[-8:] = -1
    pos = rng.integers(0, 4000, size=T)
    local = sel[:, 0].copy()
    ll = (pos % S + 1).astype(np.int32)
    pt = rng.permutation(pool)[:nkeys].astype(np.int32)
    q = jnp.asarray(rng.standard_normal((HG, T, D)) * q_gain, jnp.bfloat16)
    x = jnp.asarray(rng.standard_normal((pool, S, 2 * D)) * 0.7, jnp.float32)
    packing = 4 // jnp.dtype(dtype).itemsize
    kv = x.astype(dtype).reshape(pool, S // packing, packing, 2 * D)
    return q, kv, jnp.asarray(sel), jnp.asarray(local), jnp.asarray(ll), jnp.asarray(pt)


def oracle(q, kv, sel, local, ll, pt):
    """Exact softmax over each token's visible keys (float64), and the
    largest |score| it saw."""
    q = np.asarray(q.astype(jnp.float32), np.float64)  # [H, T, D]
    rows = np.asarray(kv.astype(jnp.float32), np.float64).reshape(kv.shape[0], S, 2 * D)
    sel, local, ll, pt = map(np.asarray, (sel, local, ll, pt))
    H, T, _ = q.shape
    out = np.zeros((H, T, D))
    lse = np.full((H, T), -np.inf)
    peak = 0.0
    for t in range(T):
        keys = np.unique(sel[t][sel[t] >= 0])
        if keys.size == 0:
            continue
        ks, vs = [], []
        for k in keys:
            lim = ll[t] if k == local[t] else S
            ks.append(rows[pt[k], :lim, :D])
            vs.append(rows[pt[k], :lim, D:])
        kk, vv = np.concatenate(ks), np.concatenate(vs)
        s = q[:, t] @ kk.T / np.sqrt(D)
        peak = max(peak, float(np.abs(s).max()))
        m = s.max(axis=1, keepdims=True)
        p = np.exp(s - m)
        out[:, t] = (p @ vv) / p.sum(axis=1, keepdims=True)
        lse[:, t] = m[:, 0] + np.log(p.sum(axis=1))
    return out, lse, peak


def run(args, block_q, spec="", pages_per_step=16):
    o, l = msa_attention_lse.msa_sparse_attention_lse(
        *args, sm_scale=1.0 / np.sqrt(D), block_q=block_q,
        pages_per_step=pages_per_step, head=0, head_major=True,
        q_head_major=True, interpret=INTERPRET, prefill_precision=spec or None,
    )
    return np.asarray(o.astype(jnp.float32), np.float64), np.asarray(l, np.float64)


def rel_rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b**2)))


def check(new, ora, bound=2e-2, lse_bound=5e-2):
    (o, l), (oo, lo) = new, ora[:2]
    fin = np.isfinite(lo)
    assert np.isfinite(o).all()
    assert np.array_equal(np.isfinite(l), fin)
    assert np.all(np.where(fin[..., None], True, o == 0))  # unseen rows: zero output
    err = rel_rms(o, oo)
    lerr = float(np.max(np.abs(np.where(fin, l - lo, 0))))
    assert err < bound and lerr < lse_bound, (err, lerr)
    return err, lerr


# (block_q, pages_per_step): 16 / 32 run the transposed body, 8 the plain one
# (decode tiles); 4-page groups split most tiles into several groups.
SHAPES = [(16, 16), (32, 16), (16, 4), (8, 16), (8, 4)]


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.bfloat16], ids=["fp8", "bf16"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"bq{s[0]}p{s[1]}")
@pytest.mark.parametrize("mxu", [False, True], ids=["sel", "mxu"])
def test_max_free_vs_oracle(max_free, dtype, shape, mxu):
    block_q, pps = shape
    args = make(64, seed=block_q + pps, dtype=dtype)
    ora = oracle(*args)
    max_free(False)
    base = run(args, block_q, pages_per_step=pps)
    e0 = check(base, ora)
    max_free(True, mxu)
    new = run(args, block_q, pages_per_step=pps)
    e1 = check(new, ora)
    assert e1[0] < max(1e-2, 2 * e0[0]), (e0, e1)
    assert rel_rms(new[0], base[0]) < 1e-2


@pytest.mark.parametrize("block_q", [16, 32])
@pytest.mark.parametrize("spec", ["exp16", "bias", "lanemask", "vt", "lmxu",
                                  "exp16,lmxu", "bias,lanemask,exp16,vt,lmxu"])
@pytest.mark.parametrize("mxu", [False, True], ids=["sel", "mxu"])
def test_max_free_composes_with_fp8_levers(max_free, block_q, spec, mxu):
    args = make(64, seed=31 + block_q)
    ora = oracle(*args)
    max_free(False)
    base = run(args, block_q, spec)
    max_free(True, mxu)
    new = run(args, block_q, spec)
    check(new, ora)
    assert rel_rms(new[0], base[0]) < 1.5e-2


@pytest.mark.parametrize("mxu", [False, True], ids=["sel", "mxu"])
def test_max_free_mask_levers_bitwise(max_free, mxu):
    """Under max-free the "bias" / "lanemask" masks still match the select
    bit for bit (with the MXU mask they are superseded: same program)."""
    args = make(64, seed=5)
    max_free(True, mxu)
    a = run(args, 32, "")
    for spec in ("bias", "lanemask"):
        b = run(args, 32, spec)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def test_large_scores_stay_finite(max_free):
    """Scores up to ~60 (natural units, under the 75 limit) neither overflow
    nor lose the softmax: the result tracks the oracle."""
    args = make(64, seed=9, q_gain=1.5 * 9)
    ora = oracle(*args)
    assert 40 < ora[2] < 75, ora[2]
    max_free(True, True)
    for block_q, pps in ((32, 16), (16, 4), (8, 4)):
        check(run(args, block_q, pages_per_step=pps), ora, bound=3e-2, lse_bound=0.2)


def test_off_adds_no_lever(max_free):
    max_free(False)
    assert msa_attention_lse.max_free_levers() == frozenset()
    max_free(True)
    assert msa_attention_lse.max_free_levers() == {"maxfree"}
    max_free(True, True)
    assert msa_attention_lse.max_free_levers() == {"maxfree", "mxumask"}


# ------------------------------------------------------------ 4-bit caches
def make4(T, seed, G):
    rng = np.random.default_rng(seed)
    pool, nkeys = 64, 48
    sel = np.full((T, G, K), -1, np.int32)
    for t0 in range(0, T, 16):
        n = (t0 // 16) % 16 + 1
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
    q = jnp.asarray(rng.standard_normal((T, G * HG, D)) * 1.5, jnp.bfloat16)
    nib = jnp.asarray(rng.integers(0, 16, size=(pool, S, G * 2 * D)), jnp.uint8)
    kv = msa_fp4.pack_rows(nib)
    scv = rng.uniform(0.02, 0.4, size=(pool, msa_fp4.scale_rows(G), S))
    scv[3, :, 5:9] = np.nan
    sc = jnp.asarray(scv, jnp.float32)
    return (q, kv, jnp.asarray(sel), jnp.asarray(local), jnp.asarray(ll), jnp.asarray(pt)), sc


def oracle4(args, sc, G):
    q, kv, sel, lk, ll, pt = [np.asarray(jax.device_get(x)) for x in args]
    nib = np.asarray(jax.device_get(msa_fp4.unpack_rows(args[1]))).astype(np.int64)
    sgn = np.where(nib >= 8, nib - 16, nib).astype(np.float64)
    scv = np.asarray(sc, np.float64)
    scv = np.where(np.isfinite(scv), scv, 0.0)
    T = q.shape[0]
    out = np.zeros((G * HG, T, D))
    lse = np.full((G * HG, T), -np.inf)
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
                s = kk @ qf[t, g * HG + h] / np.sqrt(D)
                m = s.max()
                e = np.exp(s - m)
                out[g * HG + h, t] = e @ vv / e.sum()
                lse[g * HG + h, t] = m + np.log(e.sum())
    return out, lse


def run4(args, sc, precision="", **kw):
    base = dict(sm_scale=1 / np.sqrt(D), block_q=16, pages_per_step=16,
                interpret=INTERPRET, head_major=True, prefill_precision=precision)
    base.update(kw)
    o, l = msa_attention_lse.msa_sparse_attention_lse_heads(*args, scale_cache=sc, **base)
    return np.asarray(o.astype(jnp.float32), np.float64), np.asarray(l, np.float64)


FP4_CASES = {
    "decode_bq8": (32, dict(block_q=8)),
    "stream_bq16": (64, dict()),
    "groups_p4": (64, dict(pages_per_step=4)),
    "page_cache": (64, dict(page_cache_slots=512)),
    "page_cache_bf16": (64, dict(page_cache_slots=512, page_cache_bf16=True)),
    "page_cache_qk8": (64, dict(page_cache_slots=512, precision="qk8")),
}


@pytest.mark.parametrize("case", list(FP4_CASES))
def test_max_free_fp4_vs_oracle(max_free, case):
    T, kw = FP4_CASES[case]
    kw = dict(kw)
    precision = kw.pop("precision", "")
    G = 2
    args, sc = make4(T, 3 + T, G)
    ora = oracle4(args, sc, G)
    bound = 6e-2 if precision else 1e-2
    lse_bound = 0.3 if precision else 1e-4
    max_free(False)
    base = run4(args, sc, precision, **kw)
    check(base, ora, bound, lse_bound)
    max_free(True, True)  # the MXU mask has no say in the 4-bit body
    new = run4(args, sc, precision, **kw)
    check(new, ora, bound, lse_bound)
    assert rel_rms(new[0], base[0]) < (3e-2 if precision else 1e-2)


@pytest.mark.parametrize("precision", ["pv8", "qk8,pv8"])
def test_pv8_keeps_running_max(max_free, precision):
    """p enters the "pv8" dot in fp8 (max 448): those calls refuse max-free
    and stay bit for bit the running-max kernel."""
    args, sc = make4(64, 17, 2)
    kw = dict(page_cache_slots=512)
    max_free(False)
    a = run4(args, sc, precision, **kw)
    max_free(True)
    b = run4(args, sc, precision, **kw)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


# ------------------------------------------------------------ the bound
def test_score_bound():
    wq = np.full(D, 0.5, np.float32)
    wq[3] = -2.5  # |1 + w| = 1.5
    wk = np.zeros(D, np.float32)
    wk[7] = 1.0  # 2.0
    b = msa_attention_lse.score_bound(wq, wk, D)
    assert b == pytest.approx(np.sqrt(D) * 1.5 * 2.0)
    bt = msa_attention_lse.score_bound(torch.tensor(wq), torch.tensor(wk), D, 1.25)
    assert bt == pytest.approx(np.sqrt(D) * 1.5 * 2.0 * 1.25)
    assert msa_attention_lse.KEY_ROUNDING == {"int4": 1.25, "fp8": 1.0625, "bf16": 1.0}


def test_configure_refuses_past_limit(monkeypatch):
    monkeypatch.setattr(msa_attention_lse, "MAX_FREE", False)
    monkeypatch.setattr(msa_attention_lse, "MXU_MASK", False)
    assert msa_attention_lse.configure_max_free(74.9, enabled=True, mxu_mask=True)
    assert msa_attention_lse.MAX_FREE and msa_attention_lse.MXU_MASK
    assert not msa_attention_lse.configure_max_free(75.1, enabled=True, mxu_mask=True)
    assert not msa_attention_lse.MAX_FREE and not msa_attention_lse.MXU_MASK
    assert not msa_attention_lse.configure_max_free(10.0, enabled=False, mxu_mask=True)
    assert not msa_attention_lse.MXU_MASK
    assert msa_attention_lse.configure_max_free(10.0, enabled=True, mxu_mask=False)
    assert not msa_attention_lse.MXU_MASK
    assert msa_attention_lse.max_free_levers() == {"maxfree"}


def test_configure_reads_env(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(msa_attention_lse, "MAX_FREE", False)
    monkeypatch.setattr(msa_attention_lse, "MXU_MASK", False)
    monkeypatch.delenv("TPU_MINIMAX_M3_MSA_MAX_FREE", raising=False)
    monkeypatch.delenv("TPU_MINIMAX_M3_MSA_MXU_MASK", raising=False)
    assert envs.TPU_MINIMAX_M3_MSA_MAX_FREE is False
    assert not msa_attention_lse.configure_max_free(10.0)
    monkeypatch.setenv("TPU_MINIMAX_M3_MSA_MAX_FREE", "1")
    monkeypatch.setenv("TPU_MINIMAX_M3_MSA_MXU_MASK", "1")
    assert msa_attention_lse.configure_max_free(10.0)
    assert msa_attention_lse.MXU_MASK


def test_model_hook_off_is_noop(monkeypatch):
    """The model's load hook computes nothing with the switch off, and with it
    on takes the largest layer bound."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.models.minimax_m3 import model as model_mod

    calls = []

    class Attn(model_mod.MiniMaxM3SparseAttention):
        def __init__(self, b):  # no nn.Module state needed
            object.__setattr__(self, "b", b)

        def score_bound(self):
            calls.append(self.b)
            return self.b

    class Layer:
        def __init__(self, b):
            self.self_attn = Attn(b)

    fake = type("M", (), {})()
    fake.layers = [Layer(50.0), Layer(70.0), object()]
    got = []
    monkeypatch.setattr(
        msa_attention_lse, "configure_max_free", lambda b: got.append(b) or True
    )
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_MSA_MAX_FREE", False, raising=False)
    model_mod.MiniMaxM3Model._configure_max_free(fake)
    assert calls == [] and got == []
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_MSA_MAX_FREE", True, raising=False)
    model_mod.MiniMaxM3Model._configure_max_free(fake)
    assert calls == [50.0, 70.0] and got == [70.0]


# ------------------------------------------------------------ Mosaic lowering
COMPILE_CASES = {
    # c4 / c16 prefill: fp8, transposed body, 32-token tiles of 32-page groups.
    "fp8_prefill_bq32": (jnp.float8_e4m3fn, 2048, dict(block_q=32, pages_per_step=32)),
    "fp8_prefill_levers": (
        jnp.float8_e4m3fn, 2048,
        dict(block_q=32, pages_per_step=32, prefill_precision="exp16,lmxu"),
    ),
    "fp8_decode_bq8": (jnp.float8_e4m3fn, 32, dict(block_q=8, pages_per_step=16)),
    # c48 / c64 prefill: 4-bit page cache.
    "fp4_prefill_cache": (jnp.uint8, 2048, dict(page_cache_slots=512, pages_per_step=32)),
    "fp4_decode_bq8": (jnp.uint8, 32, dict(block_q=8, pages_per_step=16)),
}


@pytest.mark.parametrize("case", list(COMPILE_CASES))
@pytest.mark.parametrize("mxu", [False, True], ids=["sel", "mxu"])
def test_compiles_for_tpu7x(max_free, case, mxu):
    """Mosaic lowers the max-free kernels (deviceless tpu7x, one chip, TP 8
    shapes: 8 query heads of one KV head, top-16, 128-token pages)."""
    from jax.experimental import topologies
    from jax.sharding import SingleDeviceSharding

    try:
        topo = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    dev = SingleDeviceSharding(topo.devices[0])
    dtype, T, kw = COMPILE_CASES[case]
    kw = dict(kw)
    fp4 = dtype == jnp.uint8
    hg, pages, nkeys = 8, 4096, 4096

    def s(shape, dt):
        return jax.ShapeDtypeStruct(shape, dt, sharding=dev)

    kv = s((pages, 16, 4, 2 * D), jnp.uint8) if fp4 else s((pages, 32, 4, 2 * D), dtype)
    args = [
        s((T, 1, K), jnp.int32) if fp4 else s((T, K), jnp.int32),
        s((T,), jnp.int32),
        s((T,), jnp.int32),
        s((nkeys,), jnp.int32),
    ]
    max_free(True, mxu)
    mesh = jax.sharding.Mesh(np.array(topo.devices[:1]), ("x",))
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        _lower(fp4, kw, s, kv, args, hg, T, pages).compile()


def _lower(fp4, kw, s, kv, args, hg, T, pages):
    if fp4:
        f = functools.partial(
            msa_attention_lse.msa_sparse_attention_lse_heads,
            sm_scale=D**-0.5, head_major=True, interpret=False, **kw,
        )
        q = s((T, hg, D), jnp.bfloat16)
        sc = s((pages, msa_fp4.scale_rows(1), S), jnp.float32)
        lowered = jax.jit(lambda q, kv, *a, sc: f(q, kv, *a, scale_cache=sc)).lower(
            q, kv, *args, sc=sc
        )
    else:
        f = functools.partial(
            msa_attention_lse.msa_sparse_attention_lse,
            sm_scale=D**-0.5, head=0, head_major=True, q_head_major=True,
            interpret=False, **kw,
        )
        lowered = jax.jit(f).lower(s((hg, T, D), jnp.bfloat16), kv, *args)
    return lowered
