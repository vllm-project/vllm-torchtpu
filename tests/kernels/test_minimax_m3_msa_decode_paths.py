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
"""MSA decode paths behind default-off switches give BASE's bits:
the work-list kernel (`_packed_work_kernel`) and one call for every KV head."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_attention_lse,
    msa_fp4,
)

TOPK = 16
PPS = 16
D = 128
HG = 4
PAGE = 128


def _keys(rng, calls, tokens, block_q, keys_per_req, fill):
    """Page keys [C, T, K] of 8-token requests, -1 padded, shared within a
    request with probability `fill`."""
    sel = np.full((calls, tokens, TOPK), -1, np.int32)
    for c in range(calls):
        for t0 in range(0, tokens, 8):
            pool = rng.choice(keys_per_req, size=TOPK + 8, replace=False) + t0 * 1000
            for t in range(t0, min(t0 + 8, tokens)):
                n = rng.integers(0, TOPK + 1)
                picks = [p for p in pool if rng.random() < fill][:n]
                sel[c, t, : len(picks)] = picks
    return jnp.asarray(sel)


@pytest.mark.parametrize(
    "calls,tokens,block_q,fill",
    [(1, 8, 8, 0.7), (1, 48, 8, 0.9), (2, 128, 8, 0.5), (1, 24, 8, 0.0), (1, 160, 16, 0.8)],
)
def test_work_list_kernel_matches_xla(calls, tokens, block_q, fill):
    rng = np.random.default_rng(tokens + calls)
    sel = _keys(rng, calls, tokens, block_q, 64, fill)
    ref, tps = msa_attention_lse._packed_work(sel, block_q, PPS)
    got, tps2, offs = msa_attention_lse._packed_work_kernel(
        sel, block_q, PPS, interpret=True
    )
    assert tps == tps2
    tiles = tokens // block_q
    gpt = msa_attention._groups_per_tile(block_q, TOPK, PPS)
    u, m, s = tiles * gpt * PPS, tiles * gpt, tiles // tps + 1
    ref, got = np.asarray(ref), np.asarray(got)
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(a[:u], b[offs[0] : offs[0] + u])
        np.testing.assert_array_equal(a[u : u + m], b[offs[1] : offs[1] + m])
        np.testing.assert_array_equal(a[u + m : u + 2 * m], b[offs[2] : offs[2] + m])
        np.testing.assert_array_equal(a[u + 2 * m : u + 2 * m + s], b[offs[3] : offs[3] + s])


def _attention_inputs(rng, heads, reqs, fp4):
    tokens = reqs * 8
    pages = reqs * 6 + 4
    pps = 8  # page keys per sequence
    table = rng.permutation(pages)[: reqs * pps].astype(np.int32)
    seq = np.arange(tokens) // 8
    sel = np.full((tokens, heads, TOPK), -1, np.int32)
    for t in range(tokens):
        for h in range(heads):
            n = rng.integers(1, pps)
            sel[t, h, :n] = seq[t] * pps + rng.choice(pps, n, replace=False)
    local = (seq * pps + rng.integers(0, pps, tokens)).astype(np.int32)
    limit = rng.integers(1, PAGE + 1, tokens).astype(np.int32)
    q = rng.standard_normal((tokens, heads * HG, D)).astype(np.float32)
    if fp4:
        nib = rng.integers(0, 16, (pages, PAGE, heads * 2 * D), dtype=np.uint8)
        kv = msa_fp4.pack_rows(jnp.asarray(np.where(nib == 8, 0, nib).astype(np.uint8)))
        sc = jnp.asarray(
            rng.uniform(0.05, 0.2, (pages, msa_fp4.scale_rows(heads), PAGE)), jnp.float32
        )
    else:
        kv = jnp.asarray(
            rng.standard_normal((pages, PAGE // 4, 4, heads * 2 * D)), jnp.float8_e4m3fn
        )
        sc = None
    return (
        jnp.asarray(q, jnp.bfloat16),
        kv,
        jnp.asarray(sel),
        jnp.asarray(local),
        jnp.asarray(limit),
        jnp.asarray(table),
        sc,
    )


@pytest.mark.parametrize(
    "env,heads,reqs,fp4",
    [
        (msa_attention_lse.WORK_LIST_KERNEL_ENV, 1, 3, False),
        (msa_attention_lse.ONE_CALL_ENV, 4, 2, True),
        (msa_attention_lse.ONE_CALL_ENV, 2, 3, False),
    ],
)
def test_decode_switch_same_bits(monkeypatch, env, heads, reqs, fp4):
    q, kv, sel, local, limit, table, sc = _attention_inputs(
        np.random.default_rng(heads * 10 + reqs), heads, reqs, fp4
    )

    def run():
        jax.clear_caches()
        return msa_attention_lse.msa_sparse_attention_lse_heads(
            q, kv, sel, local, limit, table, sm_scale=D**-0.5, block_q=8,
            scale_cache=sc, interpret=True, head_major=True,
        )

    monkeypatch.delenv(env, raising=False)
    ref = run()
    monkeypatch.setenv(env, "1")
    got = run()
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
