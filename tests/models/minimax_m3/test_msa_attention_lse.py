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
"""MSA sparse attention with log-sum-exp (interpret mode) against a dense
reference."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import CACHE_DTYPES, pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention, msa_fp4
from vllm_torchtpu.models.minimax_m3.kernels.msa_attention_lse import (
    msa_sparse_attention_lse,
    msa_sparse_attention_lse_heads,
)

Q_PER_KV = 16
HEAD_DIM = 128
PAGE = 128
PAGES = 12
TOPK = 4
SCALE = HEAD_DIM**-0.5


def _reference(q, kv_rows, keys, local_key, local_limit, table, head):
    """Per token: out and lse over its selected pages (keys -> pages)."""
    out = np.zeros_like(q)
    lse = np.full(q.shape[:2], -np.inf, np.float32)
    lo = 2 * HEAD_DIM * head
    for t in range(q.shape[0]):
        k_rows, v_rows = [], []
        for key in keys[t]:
            if key < 0:
                continue
            visible = local_limit[t] if key == local_key[t] else PAGE
            rows = kv_rows[table[key], :visible, lo : lo + 2 * HEAD_DIM]
            k_rows.append(rows[:, :HEAD_DIM])
            v_rows.append(rows[:, HEAD_DIM:])
        if not k_rows:
            continue
        s = q[t] @ np.concatenate(k_rows).T * SCALE
        m = s.max(-1, keepdims=True)
        w = np.exp(s - m)
        out[t] = (w / w.sum(-1, keepdims=True)) @ np.concatenate(v_rows)
        lse[t] = (m + np.log(w.sum(-1, keepdims=True)))[:, 0]
    return out, lse


def _inputs(rng, num_tokens, num_real, num_kv_heads, dtype):
    q = jnp.asarray(
        rng.standard_normal((num_tokens, num_kv_heads * Q_PER_KV, HEAD_DIM))
    ).astype(jnp.bfloat16)
    kv_rows = rounded(
        rng.standard_normal((PAGES, PAGE, num_kv_heads * 2 * HEAD_DIM)), dtype
    )
    table = rng.permutation(PAGES).astype(np.int32)
    keys = np.full((num_tokens, num_kv_heads, TOPK), -1, np.int32)
    local_key = np.full(num_tokens, -1, np.int32)
    local_limit = np.zeros(num_tokens, np.int32)
    for t in range(num_real):
        local_key[t] = rng.integers(PAGES)
        local_limit[t] = rng.integers(1, PAGE + 1)
        for h in range(num_kv_heads):
            chosen = rng.choice(PAGES, rng.integers(0, TOPK + 1), replace=False)
            keys[t, h, : len(chosen)] = chosen
    return q, kv_rows, pack(kv_rows, dtype), table, keys, local_key, local_limit


def _check(got, want):
    out, lse = got
    want_out, want_lse = want
    np.testing.assert_allclose(np.asarray(out.astype(jnp.float32)), want_out, atol=2e-2)
    seen = np.isfinite(want_lse)
    np.testing.assert_array_equal(np.isfinite(np.asarray(lse)), seen)
    np.testing.assert_allclose(np.asarray(lse)[seen], want_lse[seen], atol=2e-2)


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
@pytest.mark.parametrize(
    "num_tokens,num_real,block_q,tokens_per_call,sorted_unions,pages_per_step",
    [
        (37, 35, 16, None, False, 16),  # last tile part padding
        (24, 24, 8, None, False, 16),  # 8-token tiles
        (150, 140, 16, 64, False, 16),  # several kernel calls (the chunk loop)
        (150, 140, 16, 64, True, 16),  # the prefill (sorted) work list
        (150, 140, 16, 64, True, 5),  # groups per tile padded to a power of two
    ],
)
def test_lse_matches_dense_reference(
    num_tokens,
    num_real,
    block_q,
    tokens_per_call,
    sorted_unions,
    pages_per_step,
    dtype,
    monkeypatch,
):
    if sorted_unions:
        monkeypatch.setattr(msa_attention, "_DENSE_WORK_LIST_ELEMS", 0)
    if tokens_per_call is not None:
        monkeypatch.setattr(msa_attention, "MAX_TOKENS_PER_CALL", tokens_per_call)
    rng = np.random.default_rng(num_tokens)
    q, kv_rows, kv, table, keys, local_key, local_limit = _inputs(
        rng, num_tokens, num_real, 1, dtype
    )
    got = msa_sparse_attention_lse(
        q,
        kv,
        jnp.asarray(keys[:, 0]),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        jnp.asarray(table),
        sm_scale=SCALE,
        block_q=block_q,
        pages_per_step=pages_per_step,
        interpret=True,
    )
    want = _reference(
        np.asarray(q.astype(jnp.float32)),
        kv_rows,
        keys[:, 0],
        local_key,
        local_limit,
        table,
        0,
    )
    _check(got, want)


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_lse_heads_matches_dense_reference(dtype, monkeypatch):
    """Every KV head of multi-head cache rows, over several kernel calls."""
    monkeypatch.setattr(msa_attention, "MAX_TOKENS_PER_CALL", 32)
    num_kv_heads, num_tokens = 2, 70
    rng = np.random.default_rng(5)
    q, kv_rows, kv, table, keys, local_key, local_limit = _inputs(
        rng, num_tokens, num_tokens - 3, num_kv_heads, dtype
    )
    got = msa_sparse_attention_lse_heads(
        q,
        kv,
        jnp.asarray(keys),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        jnp.asarray(table),
        sm_scale=SCALE,
        interpret=True,
    )
    q32 = np.asarray(q.astype(jnp.float32))
    outs, lses = [], []
    for h in range(num_kv_heads):
        o, lse_h = _reference(
            q32[:, h * Q_PER_KV : (h + 1) * Q_PER_KV],
            kv_rows,
            keys[:, h],
            local_key,
            local_limit,
            table,
            h,
        )
        outs.append(o)
        lses.append(lse_h)
    _check(got, (np.concatenate(outs, axis=1), np.concatenate(lses, axis=1)))


def test_lse_empty_batch():
    out, lse = msa_sparse_attention_lse(
        jnp.zeros((0, Q_PER_KV, HEAD_DIM), jnp.bfloat16),
        pack(np.zeros((PAGES, PAGE, 2 * HEAD_DIM), np.float32), jnp.bfloat16),
        jnp.zeros((0, TOPK), jnp.int32),
        jnp.zeros((0,), jnp.int32),
        jnp.zeros((0,), jnp.int32),
        jnp.arange(PAGES, dtype=jnp.int32),
        sm_scale=SCALE,
        interpret=True,
    )
    assert out.shape == (0, Q_PER_KV, HEAD_DIM) and lse.shape == (0, Q_PER_KV)


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_lse_unselected_nan_pages_do_not_leak(dtype):
    """Bytes no token may see hold NaN (pages nobody selects, slots past a
    local page's newest token): masked scores and zeroed value rows keep
    them out, as they must for page-buffer slots past a group's pages."""
    rng = np.random.default_rng(9)
    q, kv_rows, _, table, keys, local_key, local_limit = _inputs(rng, 70, 66, 1, dtype)
    # Keys 0..4 are selected whole; key 5 is every real token's local page,
    # whose slots past its newest token no token sees; keys 6..11 nobody
    # selects.
    keys = np.where(keys >= 0, keys % 5, -1)
    for row in keys[:, 0]:  # a page counts once per token
        _, first = np.unique(row, return_index=True)
        row[np.setdiff1d(np.arange(row.size), first)] = -1
    real = local_key >= 0
    keys[real, 0, -1] = 5
    local_key[real] = 5
    local_limit[real] = rng.integers(1, PAGE // 2, real.sum())
    poisoned = kv_rows.copy()
    poisoned[table[6:]] = np.nan
    poisoned[table[5], local_limit.max() :] = np.nan
    got = msa_sparse_attention_lse(
        q,
        pack(poisoned, dtype),
        jnp.asarray(keys[:, 0]),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        jnp.asarray(table),
        sm_scale=SCALE,
        interpret=True,
    )
    assert np.isfinite(np.asarray(got[0].astype(jnp.float32))).all()
    want = _reference(
        np.asarray(q.astype(jnp.float32)),
        kv_rows,
        keys[:, 0],
        local_key,
        local_limit,
        table,
        0,
    )
    _check(got, want)


def _fp4_cache(rng, num_kv_heads):
    """A 4-bit cache, its scale pages (head ``h``: K / V rows ``2h`` /
    ``2h + 1``) and the f32 rows they hold."""
    x = rng.standard_normal((PAGES, PAGE, num_kv_heads, 2, HEAD_DIM))
    nib, scale = msa_fp4.quantize(jnp.asarray(x, jnp.float32))
    rows = np.asarray(msa_fp4.dequantize(nib, scale)).reshape(PAGES, PAGE, -1)
    kv = msa_fp4.pack_rows(nib.reshape(PAGES, PAGE, -1))
    scales = np.zeros((PAGES, msa_fp4.SCALE_ROWS, PAGE), np.float32)
    scales[:, : 2 * num_kv_heads] = (
        np.asarray(scale).reshape(PAGES, PAGE, -1).transpose(0, 2, 1)
    )
    return kv, scales, rows


@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_fp4_lse_matches_dense_reference(num_kv_heads, monkeypatch):
    """4-bit pages and their scale pages, copied by the kernel beside each
    page: single-head calls (static scale rows) and every head at once
    (rows of the cache-row head). Scale pages nobody selects hold NaN."""
    monkeypatch.setattr(msa_attention, "MAX_TOKENS_PER_CALL", 32)
    num_tokens = 70
    rng = np.random.default_rng(11 + num_kv_heads)
    q, _, _, table, keys, local_key, local_limit = _inputs(
        rng, num_tokens, num_tokens - 3, num_kv_heads, jnp.bfloat16
    )
    keys = np.where(keys >= 0, keys % 10, -1)  # keys 10, 11: nobody selects
    for t in range(num_tokens):  # a page counts once per token and head
        for h in range(num_kv_heads):
            _, first = np.unique(keys[t, h], return_index=True)
            keys[t, h, np.setdiff1d(np.arange(TOPK), first)] = -1
    local_key = np.where(local_key >= 0, local_key % 10, -1)
    kv, scales, rows = _fp4_cache(rng, num_kv_heads)
    scales[table[10:]] = np.nan
    common = dict(sm_scale=SCALE, scale_cache=jnp.asarray(scales), interpret=True)
    args = (jnp.asarray(local_key), jnp.asarray(local_limit), jnp.asarray(table))
    if num_kv_heads == 1:
        got = msa_sparse_attention_lse(q, kv, jnp.asarray(keys[:, 0]), *args, **common)
    else:
        got = msa_sparse_attention_lse_heads(q, kv, jnp.asarray(keys), *args, **common)
    assert np.isfinite(np.asarray(got[0].astype(jnp.float32))).all()
    q32 = np.asarray(q.astype(jnp.float32))
    outs, lses = [], []
    for h in range(num_kv_heads):
        o, lse_h = _reference(
            q32[:, h * Q_PER_KV : (h + 1) * Q_PER_KV],
            rows,
            keys[:, h],
            local_key,
            local_limit,
            table,
            h,
        )
        outs.append(o)
        lses.append(lse_h)
    _check(got, (np.concatenate(outs, axis=1), np.concatenate(lses, axis=1)))
