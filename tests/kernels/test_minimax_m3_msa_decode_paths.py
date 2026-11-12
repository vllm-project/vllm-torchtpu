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
"""r6 MSA lse kernel (`msa_attention_lse_r6`, key minimax_msa_lse_r6) paths
behind default-off keys give the default path's bits: the decode work-list
kernel (`_packed_work_kernel`, key minimax_msa_decode_work_list_kernel), one
call for every KV head (key minimax_msa_decode_one_call), and the prefill
knobs (page cache, tile-union cap, bf16 page cache, row split, step widths)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.minimax_m3.kernels import msa_attention, msa_fp4
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention_lse_r6 as msa_attention_lse,
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
    [(1, 8, 8, 0.7), (1, 48, 8, 0.9), (2, 128, 8, 0.5), (1, 24, 8, 0.0), (1, 160, 16, 0.8), (1, 2048, 16, 0.3)],
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
    "flags,heads,reqs,fp4",
    [
        (("DECODE_WORK_LIST_KERNEL",), 1, 3, False),
        (("DECODE_ONE_CALL",), 4, 2, True),
        (("DECODE_ONE_CALL",), 2, 3, False),
        (("DECODE_ONE_CALL", "DECODE_WORK_LIST_KERNEL"), 4, 3, True),
    ],
)
def test_decode_switch_same_bits(monkeypatch, flags, heads, reqs, fp4):
    q, kv, sel, local, limit, table, sc = _attention_inputs(
        np.random.default_rng(heads * 10 + reqs), heads, reqs, fp4
    )

    def run():
        jax.clear_caches()
        return msa_attention_lse.msa_sparse_attention_lse_heads(
            q, kv, sel, local, limit, table, sm_scale=D**-0.5, block_q=8,
            scale_cache=sc, interpret=True, head_major=True,
        )

    for f in flags:
        assert getattr(msa_attention_lse, f) is False
    ref = run()
    for f in flags:
        monkeypatch.setattr(msa_attention_lse, f, True)
    got = run()
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize(
    "knobs",
    [
        dict(page_cache_slots=64),
        dict(page_cache_slots=64, page_cache_bf16=True),
        dict(page_cache_slots=64, tile_pages=64),
        dict(page_cache_slots=4),  # too few slots: the call streams
        dict(row_split=2),
        dict(step_widths=(4, 8, 12)),
        dict(step_widths=(2, 4, 6, 8, 12), page_cache_slots=64, row_split=2),
    ],
)
def test_prefill_knobs_same_bits(monkeypatch, knobs):
    """The prefill knobs `msa_dcp.msa_attend_dcp` passes (`prefill_options`)
    keep the bits of the default call."""
    heads, reqs = 2, 6
    q, kv, sel, local, limit, table, sc = _attention_inputs(
        np.random.default_rng(99), heads, reqs, True
    )
    knobs = dict(knobs)
    tile_pages = knobs.pop("tile_pages", None)

    def run(**kw):
        jax.clear_caches()
        return msa_attention_lse.msa_sparse_attention_lse_heads(
            q, kv, sel, local, limit, table, sm_scale=D**-0.5, block_q=16,
            scale_cache=sc, interpret=True, head_major=True, **kw,
        )

    ref = run()
    if tile_pages is not None:
        monkeypatch.setattr(msa_attention_lse, "PAGE_CACHE_TILE_PAGES", tile_pages)
    got = run(**knobs)
    for a, b in zip(ref, got):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_prefill_options_empty_by_default(monkeypatch):
    assert msa_attention_lse.prefill_options(True) == {}
    assert msa_attention_lse.prefill_options(False) == {}
    monkeypatch.setattr(msa_attention_lse, "PREFILL_PAGE_CACHE_SLOTS", 512)
    monkeypatch.setattr(msa_attention_lse, "PREFILL_PAGE_CACHE_BF16", True)
    monkeypatch.setattr(msa_attention_lse, "PREFILL_STEP_WIDTHS", (4, 8, 12))
    assert msa_attention_lse.prefill_options(False) == {"step_widths": (4, 8, 12)}
    assert msa_attention_lse.prefill_options(True) == {
        "step_widths": (4, 8, 12),
        "page_cache_slots": 512,
        "page_cache_bf16": True,
    }
