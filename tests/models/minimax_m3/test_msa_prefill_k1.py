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
"""TPU_MINIMAX_M3_MSA_PREFILL_K1: the head-parallel prefill call shape (one KV
head, 16 query heads, 8-token tiles of 64-page groups, 4-bit pages, page
cache, qk8, max-free + MXU mask), interpret mode.

- unset / "0": no lever, the incumbent bit for bit;
- "1" = "split32+w56" (w56: head-parallel calls with no configured widths
  add widths 40 / 56, `test_hp_widths`);
- "split32": groups over 32 pages softmax in page chunks: same -inf
  pattern, close to the incumbent and to a float64 oracle (bitwise on TPU7x,
  not under interpret, whose f32 sums run in another order); groups of at
  most 32 pages run the incumbent body, bit for bit, cached or streamed."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse, msa_dcp_heads, msa_fp4

INTERPRET = os.environ.get("INTERPRET", "1") == "1"
S, D, K, HG = 128, 128, 16, 16
ENV = msa_attention_lse.K1_ENV


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(msa_attention_lse, "MAX_FREE", True)
    monkeypatch.setattr(msa_attention_lse, "MXU_MASK", True)
    monkeypatch.delenv(ENV, raising=False)
    jax.clear_caches()

    def set_k1(spec):
        if spec is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, spec)
        jax.clear_caches()

    yield set_k1
    jax.clear_caches()


def make(T, seed, union=48, pool=96):
    """8-token tiles over a union of `union` pages (wider than 32 pages, so
    groups split), each token selecting up to 16 of them."""
    rng = np.random.default_rng(seed)
    nkeys = pool - 8
    sel = np.full((T, K), -1, np.int32)
    for t0 in range(0, T, 8):
        u = rng.choice(nkeys, size=min(union, nkeys), replace=False)
        for t in range(t0, t0 + 8):
            m = int(rng.integers(1, K + 1))
            sel[t, :m] = rng.choice(u, size=m, replace=False)
    sel[-8:] = -1  # a tile of padding tokens
    pos = rng.integers(0, 4000, size=T)
    local = sel[:, 0].copy()
    ll = (pos % S + 1).astype(np.int32)
    pt = rng.permutation(pool)[:nkeys].astype(np.int32)
    q = jnp.asarray(rng.standard_normal((T, HG, D)) * 1.5, jnp.bfloat16)
    kv = msa_fp4.pack_rows(jnp.asarray(rng.integers(0, 16, size=(pool, S, 2 * D)), jnp.uint8))
    scv = rng.uniform(0.02, 0.4, size=(pool, msa_fp4.scale_rows(1), S))
    scv[3, :, 5:9] = np.nan  # unwritten slots: non-finite scales read as 0
    args = (q, kv, jnp.asarray(sel), jnp.asarray(local), jnp.asarray(ll), jnp.asarray(pt))
    return args, jnp.asarray(scv, jnp.float32)


def oracle(args, sc):
    q, kv, sel, lk, ll, pt = [np.asarray(jax.device_get(x)) for x in args]
    nib = np.asarray(jax.device_get(msa_fp4.unpack_rows(args[1]))).astype(np.int64)
    sgn = np.where(nib >= 8, nib - 16, nib).astype(np.float64)
    scv = np.asarray(sc, np.float64)
    scv = np.where(np.isfinite(scv), scv, 0.0)
    T = q.shape[0]
    out = np.zeros((HG, T, D))
    lse = np.full((HG, T), -np.inf)
    qf = q.astype(np.float64)
    for t in range(T):
        ks, vs = [], []
        for key in np.unique(sel[t][sel[t] >= 0]):
            p = pt[key]
            n = ll[t] if key == lk[t] else S
            ks.append(sgn[p, :n, :D] * scv[p, 0, :n, None])
            vs.append(sgn[p, :n, D:] * scv[p, 1, :n, None])
        if not ks:
            continue
        kk, vv = np.concatenate(ks), np.concatenate(vs)
        for h in range(HG):
            s = kk @ qf[t, h] / np.sqrt(D)
            m = s.max()
            e = np.exp(s - m)
            out[h, t] = e @ vv / e.sum()
            lse[h, t] = m + np.log(e.sum())
    return out, lse


def run(args, sc, slots=128):
    o, l = msa_attention_lse.msa_sparse_attention_lse(
        *args, sm_scale=1 / np.sqrt(D), block_q=8, pages_per_step=64, head=0,
        scale_cache=sc, k_scale_row=0, v_scale_row=1, head_major=True,
        page_cache_slots=slots, page_cache_tile_pages=64, prefill_precision="qk8",
        interpret=INTERPRET,
    )
    return np.asarray(o.astype(jnp.float32)), np.asarray(l)


def test_levers_parse(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert msa_attention_lse.k1_levers() == frozenset()
    assert msa_attention_lse.k1_levers("0") == frozenset()
    assert msa_attention_lse.k1_levers("1") == {"k1", "split32", "w56"}
    assert msa_attention_lse.k1_levers("split24") == {"k1", "split24"}
    with pytest.raises(ValueError):
        msa_attention_lse.k1_levers("split32+bogus")


@pytest.mark.parametrize("spec", [None, "0"])
def test_off_is_incumbent(served, spec):
    args, sc = make(64, 0)
    served(None)
    base = run(args, sc)
    served(spec)
    new = run(args, sc)
    assert base[0].tobytes() == new[0].tobytes()
    assert base[1].tobytes() == new[1].tobytes()


@pytest.mark.parametrize("slots", [128, 16])  # 16: too few slots, the call streams
def test_narrow_groups_bitwise(served, slots):
    args, sc = make(64, 1, union=24)
    served(None)
    base = run(args, sc, slots)
    served("1")
    new = run(args, sc, slots)
    assert base[0].tobytes() == new[0].tobytes()
    assert base[1].tobytes() == new[1].tobytes()


@pytest.mark.parametrize("spec", ["1", "split16"])
def test_split_close(served, spec):
    args, sc = make(64, 2)
    served(None)
    base = run(args, sc)
    served(spec)
    new = run(args, sc)
    ora = oracle(args, sc)
    fin = np.isfinite(base[1])
    assert np.array_equal(fin, np.isfinite(new[1]))
    assert np.isfinite(new[0]).all()
    rel = np.sqrt(np.mean((new[0] - base[0]) ** 2)) / np.sqrt(np.mean(base[0] ** 2))
    assert rel < 1e-2, rel
    assert np.max(np.abs(new[1][fin] - base[1][fin])) < 1e-3
    # Same distance to the oracle as the incumbent (qk8 rounding dominates).
    def to_oracle(x):
        return np.sqrt(np.mean((x[0] - ora[0]) ** 2)) / np.sqrt(np.mean(ora[0] ** 2))
    assert to_oracle(new) < 6e-2
    assert to_oracle(new) < 1.2 * to_oracle(base) + 1e-3


def test_hp_widths(monkeypatch):
    monkeypatch.setenv("TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG", "8:64::1024")
    monkeypatch.delenv(ENV, raising=False)
    assert "step_widths" not in msa_dcp_heads.hp_prefill_kernel_kwargs()
    monkeypatch.setenv(ENV, "1")
    kw = msa_dcp_heads.hp_prefill_kernel_kwargs()
    assert kw["step_widths"] == (4, 8, 16, 24, 32, 40, 48, 56)
    assert (kw["block_q"], kw["pages_per_step"], kw["page_cache_slots"]) == (8, 64, 1024)
    monkeypatch.setenv(ENV, "split32")
    assert "step_widths" not in msa_dcp_heads.hp_prefill_kernel_kwargs()
    # Explicit widths win.
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG", "8:64:8/16:1024")
    assert msa_dcp_heads.hp_prefill_kernel_kwargs()["step_widths"] == (8, 16)
