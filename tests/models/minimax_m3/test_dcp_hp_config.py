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
"""TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG: the head-parallel prefill attention
kernel config, and its numerics against the default config (interpret)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_attention_lse,
    msa_dcp_heads,
    msa_fp4,
)

ENV = msa_dcp_heads.HP_PREFILL_CONFIG_ENV


def test_unset_is_kernel_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert msa_dcp_heads.hp_prefill_kernel_kwargs() == dict(
        block_q=msa_attention.DEFAULT_BLOCK_Q,
        pages_per_step=msa_attention.DEFAULT_PAGES_PER_STEP,
    )


def test_parse(monkeypatch):
    monkeypatch.setenv(ENV, "32:16:12/2/4/6/8:512")
    kw = msa_dcp_heads.hp_prefill_kernel_kwargs()
    assert kw["block_q"] == 32 and kw["pages_per_step"] == 16
    assert kw["step_widths"] == (2, 4, 6, 8, 12)
    assert kw["page_cache_slots"] == 512
    monkeypatch.setenv(ENV, "16:16::0")
    assert msa_dcp_heads.hp_prefill_kernel_kwargs() == dict(block_q=16, pages_per_step=16)
    monkeypatch.setenv(ENV, "16:16")
    with pytest.raises(ValueError):
        msa_dcp_heads.hp_prefill_kernel_kwargs()


def _inputs(T=64, pool=40, heads=8, d=128, s=128, topk=16, seed=0):
    rng = np.random.default_rng(seed)
    nib = jnp.asarray(rng.integers(0, 16, (pool, s, 2 * d)), jnp.uint8)
    kv = msa_fp4.pack_rows(nib)
    sc = jnp.asarray(rng.uniform(0.01, 0.03, (pool, msa_fp4.SCALE_ROWS, s)), jnp.float32)
    # Request-coherent selections: each token picks 6..16 pages of a 24-page window.
    sel = np.full((T, topk), -1, np.int32)
    for t in range(T):
        n = rng.integers(6, topk + 1)
        sel[t, :n] = rng.choice(24, n, replace=False) + (t // 32)
    sel = -np.sort(-sel, axis=1)
    lk = sel[:, 0].copy()
    pos = rng.integers(0, s, T)
    q = jnp.asarray(rng.standard_normal((T, heads, d)) * 0.5, jnp.bfloat16)
    pt = jnp.asarray(rng.permutation(pool), jnp.int32)
    return q, kv, jnp.asarray(sel), jnp.asarray(lk), jnp.asarray(pos % s + 1, jnp.int32), pt, sc


def _attend(args, **kw):
    q, kv, sel, lk, ll, pt, sc = args
    return msa_attention_lse.msa_sparse_attention_lse(
        q, kv, sel, lk, ll, pt, sm_scale=1 / math.sqrt(128), head=0,
        scale_cache=sc, k_scale_row=0, v_scale_row=1, head_major=True,
        interpret=True, **kw,
    )


@pytest.mark.parametrize("spec", ["16:16:2/4/6/8/12:0", "16:16:2/4/6/8/12:512", "32:16:2/4/6/8/12:512", "8:16::0"])
def test_numerics_vs_default(monkeypatch, spec):
    args = _inputs()
    monkeypatch.delenv(ENV, raising=False)
    o0, l0 = _attend(args, **msa_dcp_heads.hp_prefill_kernel_kwargs())
    monkeypatch.setenv(ENV, spec)
    o1, l1 = _attend(args, **msa_dcp_heads.hp_prefill_kernel_kwargs())
    o0, o1 = np.asarray(o0, np.float32), np.asarray(o1, np.float32)
    l0, l1 = np.asarray(l0), np.asarray(l1)
    assert np.array_equal(np.isfinite(l0), np.isfinite(l1))
    fin = np.isfinite(l0)
    np.testing.assert_allclose(l1[fin], l0[fin], atol=1e-4)
    np.testing.assert_allclose(o1, o0, atol=2e-2, rtol=2e-2)
    if spec.startswith("16:"):
        # Same 16-token tiles: widths and the page cache change only which
        # columns run, not the sums (bitwise, as on the msa_attend_dcp path).
        assert np.array_equal(o1, o0) and np.array_equal(l1, l0)
