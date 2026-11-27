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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import CACHE_DTYPES, pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels import (
    msa_attention,
    msa_attention_lse,
    msa_fp4,
)
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
        (24, 24, 8, None, False, 32),  # 8-token tiles, the DCP decode group size
        (150, 140, 16, 64, False, 16),  # several kernel calls (the chunk loop)
        (150, 140, 16, 64, True, 16),  # the prefill (sorted) work list
        (150, 140, 16, 64, True, 5),  # groups per tile padded to a power of two
        (150, 140, 32, 64, True, 16),  # prefill (32-token) tiles
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


# (head-major results, head-major q)
LAYOUTS = [(False, False), (True, False), (True, True)]


def _head_major_call(fn, q, *args, head_major, q_head_major=False, **kwargs):
    """`fn`'s token-major results, or its head-major ones transposed back;
    with `q_head_major`, `fn` takes q head-major."""
    if q_head_major:
        q = q.transpose(1, 0, 2)
        kwargs["q_head_major"] = True
    if not head_major:
        return fn(q, *args, **kwargs)
    out, lse = fn(q, *args, head_major=True, **kwargs)
    return out.transpose(1, 0, 2), lse.T


@pytest.mark.parametrize("head_major,q_head_major", LAYOUTS)
@pytest.mark.parametrize("dtype", CACHE_DTYPES)
def test_lse_heads_matches_dense_reference(
    dtype, head_major, q_head_major, monkeypatch
):
    """Every KV head of multi-head cache rows, over several kernel calls."""
    monkeypatch.setattr(msa_attention, "MAX_TOKENS_PER_CALL", 32)
    num_kv_heads, num_tokens = 2, 70
    rng = np.random.default_rng(5)
    q, kv_rows, kv, table, keys, local_key, local_limit = _inputs(
        rng, num_tokens, num_tokens - 3, num_kv_heads, dtype
    )
    got = _head_major_call(
        msa_sparse_attention_lse_heads,
        q,
        kv,
        jnp.asarray(keys),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        jnp.asarray(table),
        sm_scale=SCALE,
        interpret=True,
        head_major=head_major,
        q_head_major=q_head_major,
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
    # 8-bit pages enter the MXU as stored (`DIRECT_T`, as the decode tiles'
    # `direct` path): they rely on `msa_write` zeroing the slots past a
    # page's newest token, so only wider pages hold NaN there.
    eight_bit = np.dtype(dtype).itemsize == 1 and msa_attention_lse.DIRECT_T
    poisoned[table[5], local_limit.max() :] = 0.0 if eight_bit else np.nan
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


@pytest.mark.parametrize(
    "limit_table,kernel_limits", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("head_major,q_head_major", LAYOUTS)
@pytest.mark.parametrize("num_kv_heads", [1, 2])
def test_fp4_lse_matches_dense_reference(
    num_kv_heads, head_major, q_head_major, limit_table, kernel_limits, monkeypatch
):
    """4-bit pages and their scale pages, copied by the kernel beside each
    page: single-head calls (static scale rows) and every head at once
    (rows of the cache-row head). Scale pages nobody selects hold NaN.
    `limit_table`: the calls take their page limits from the caller's table
    (prefill-size calls); `kernel_limits`: without it, from the kernel's
    per-group compare (decode-size calls), else per page."""
    monkeypatch.setattr(msa_attention_lse, "KERNEL_LIMITS", kernel_limits)
    monkeypatch.setattr(msa_attention, "MAX_TOKENS_PER_CALL", 32)
    monkeypatch.setattr(
        msa_attention_lse, "_LIMIT_TABLE_MIN_TOKENS", 32 if limit_table else 1 << 30
    )
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
    common.update(head_major=head_major, q_head_major=q_head_major)
    if num_kv_heads == 1:
        got = _head_major_call(
            msa_sparse_attention_lse, q, kv, jnp.asarray(keys[:, 0]), *args, **common
        )
    else:
        got = _head_major_call(
            msa_sparse_attention_lse_heads, q, kv, jnp.asarray(keys), *args, **common
        )
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


@pytest.mark.parametrize("dtype", CACHE_DTYPES)
@pytest.mark.parametrize("block_q,pages_per_step", [(8, 32), (32, 32), (8, 64)])
def test_lse_wide_groups(block_q, pages_per_step, dtype, monkeypatch):
    """Unions of tens of pages: 32- and 64-page groups, their remainders at
    the 24-page step width and full-group steps."""
    monkeypatch.setitem(globals(), "PAGES", 96)
    monkeypatch.setitem(globals(), "TOPK", 16)
    num_tokens = 64
    rng = np.random.default_rng(block_q + pages_per_step)
    q, kv_rows, kv, table, keys, local_key, local_limit = _inputs(
        rng, num_tokens, num_tokens - 2, 1, dtype
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


@pytest.mark.parametrize(
    "block_q,pages_per_step,q_per_kv,num_tokens",
    [
        (8, 32, 16, 64),
        (8, 64, 16, 64),
        (8, 64, 8, 64),
        (8, 64, 8, 48),
        (32, 32, 16, 64),
        (16, 32, 16, 64),
    ],
)
def test_fp4_lse_wide_groups(
    block_q, pages_per_step, q_per_kv, num_tokens, monkeypatch
):
    """4-bit pages and their scale pages in 32-page groups (the 4-bit prefill
    calls) and 64-page ones of 8 query heads (the head-parallel calls, whose
    8-row output stores merge into 16-row tiles: 48 tokens take 2 tiles a
    grid step, where 3 would leave a half window)."""
    monkeypatch.setitem(globals(), "Q_PER_KV", q_per_kv)
    monkeypatch.setitem(globals(), "PAGES", 96)
    monkeypatch.setitem(globals(), "TOPK", 16)
    rng = np.random.default_rng(7 + pages_per_step)
    q, _, _, table, keys, local_key, local_limit = _inputs(
        rng, num_tokens, num_tokens - 2, 1, jnp.bfloat16
    )
    kv, scales, rows = _fp4_cache(rng, 1)
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
        scale_cache=jnp.asarray(scales),
        interpret=True,
    )
    want = _reference(
        np.asarray(q.astype(jnp.float32)),
        rows,
        keys[:, 0],
        local_key,
        local_limit,
        table,
        0,
    )
    _check(got, want)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("dtype", [*CACHE_DTYPES, "int4"])
@pytest.mark.parametrize(
    "num_tokens,num_real,pages_per_step,most_tiles",
    [
        (40, 30, 16, 64),  # one step of 5 tiles (padded to 6): a unit and singles
        (56, 56, 16, 2),  # 2-tile steps: units shorter than DECODE_UNIT_GROUPS
        (88, 61, 32, 4),  # padding tiles (no pages) beside wide groups
        (8, 5, 16, 64),  # one tile, unpadded: the plain group loop
        (64, 37, 16, 64),  # trailing padding tiles of a bucketed batch, alone
        (32, 24, 32, 64),  # 3 tiles + 1 padding tile: a unit of the remainder
    ],
)
def test_decode_units_match_reference(
    num_tokens, num_real, pages_per_step, most_tiles, dtype, ordered, monkeypatch
):
    """8-token tiles of calls over several KV heads run in units of
    `DECODE_UNIT_GROUPS` groups (`ordered`: their tile pairs in width order,
    `_width_order`): group counts per step that are not multiples of it,
    steps shorter than a unit and its remainder, units holding padding tiles
    and tiles of several groups, paired and unpaired finishes, and page
    copies issued across steps and calls (`run_unit`'s count)."""
    if ordered:
        monkeypatch.setattr(msa_attention_lse, "WIDTH_ORDER_MIN_TILES", 2)
    monkeypatch.setitem(globals(), "PAGES", 96)
    monkeypatch.setitem(globals(), "TOPK", 16)
    monkeypatch.setattr(msa_attention_lse, "UNIT_TILES_PER_STEP", most_tiles)
    jax.clear_caches()
    num_kv_heads = 2
    rng = np.random.default_rng(num_tokens + pages_per_step)
    fp4 = dtype == "int4"
    q, kv_rows, kv, table, keys, local_key, local_limit = _inputs(
        rng, num_tokens, num_real, num_kv_heads, jnp.bfloat16 if fp4 else dtype
    )
    extra = {}
    if fp4:
        kv, scales, kv_rows = _fp4_cache(rng, num_kv_heads)
        extra = dict(scale_cache=jnp.asarray(scales))
    got = msa_sparse_attention_lse_heads(
        q,
        kv,
        jnp.asarray(keys),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        jnp.asarray(table),
        sm_scale=SCALE,
        block_q=8,
        pages_per_step=pages_per_step,
        interpret=True,
        **extra,
    )
    jax.clear_caches()
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


@pytest.mark.parametrize("num_tokens,most_tiles", [(96, 64), (256, 8), (48, 64)])
def test_width_order_keeps_tiles_and_pairs_whole(num_tokens, most_tiles):
    """`_packed_work`'s width order only permutes each grid step's groups:
    unions and step starts unchanged, every tile's groups contiguous and in
    order (first .. last), tile pairs (2k, 2k + 1) adjacent, and the pairs
    of a step in non-decreasing width, tiles of no page last."""
    rng = np.random.default_rng(num_tokens)
    block_q, topk, pages_per_step = 8, 16, 32
    num_tiles = num_tokens // block_q
    selected = np.full((2, num_tokens, topk), -1, np.int32)
    for c in range(2):
        for t in range(num_tiles):
            union = rng.choice([0, 1, 3, 9, 20, 31, 45, 70])
            pool = t * 1000 + rng.permutation(500)[: max(union, 1)]
            for i in range(block_q):
                n = min(topk, union)
                selected[c, t * block_q + i, :n] = rng.choice(pool, n, replace=False)
    args = (jnp.asarray(selected), block_q, pages_per_step)
    kw = dict(align=16, most=most_tiles)
    plain, tiles_per_step = msa_attention_lse._packed_work(*args, **kw, interpret=True)
    ordered, _ = msa_attention_lse._packed_work(
        *args, **kw, width_order=True, interpret=True
    )
    plain, ordered = np.asarray(plain), np.asarray(ordered)
    groups_per_tile = msa_attention._groups_per_tile(block_q, topk, pages_per_step)
    num_groups = num_tiles * groups_per_tile
    unions = num_groups * pages_per_step
    meta, counts = (
        slice(unions, unions + num_groups),
        slice(unions + num_groups, unions + 2 * num_groups),
    )
    np.testing.assert_array_equal(ordered[:, :unions], plain[:, :unions])
    np.testing.assert_array_equal(
        ordered[:, unions + 2 * num_groups :], plain[:, unions + 2 * num_groups :]
    )
    widths = [n for n in msa_attention_lse._STEP_WIDTHS if n < pages_per_step]
    for c in range(2):
        starts = plain[c, unions + 2 * num_groups :]
        for step in range(num_tiles // tiles_per_step):
            lo, hi = starts[step], starts[step + 1]
            group = slice(meta.start + lo, meta.start + hi)
            n_group = slice(counts.start + lo, counts.start + hi)
            old = sorted(zip(plain[c, group], plain[c, n_group]))
            new = list(zip(ordered[c, group], ordered[c, n_group]))
            assert sorted(new) == old
            tiles = [int(msa_attention._meta_tile(m, groups_per_tile)) for m, _ in new]
            runs = [tiles[0]] + [b for a, b in zip(tiles, tiles[1:]) if b != a]
            assert len(runs) == len(set(runs)), "a tile's groups split"
            assert all(
                a % 2 == 0 and b == a + 1 for a, b in zip(runs[0::2], runs[1::2])
            )
            assert [m for m, _ in new] == sorted(
                (m for m, _ in new),
                key=lambda m: (
                    runs.index(int(msa_attention._meta_tile(m, groups_per_tile))),
                    m,
                ),
            )
            pages = {t: 0 for t in runs}
            for t, (_, n) in zip(tiles, new):
                pages[t] += int(n)

            def width(t, pages=pages):
                p = pages[t]
                return (
                    -1 if p == 0 else sum(p > w for w in widths) + (p > pages_per_step)
                )

            keys = [max(width(a), width(b)) for a, b in zip(runs[0::2], runs[1::2])]
            keys = [len(widths) + 2 if k < 0 else k for k in keys]
            assert keys == sorted(keys)


@pytest.mark.parametrize("pages_per_step", [16, 32])
def test_decode_tiles_read_unwritten_slots_through_the_write(pages_per_step):
    """8-bit decode tiles feed pages to the MXU unmasked: a local page's
    slots past its newest token must have been zeroed by the KV write
    (`msa_write`), whatever bytes (NaN) they held before; stale ring slots
    past a group's pages are zeroed by the kernel."""
    from vllm_torchtpu.models.minimax_m3.kernels import msa

    dtype = jnp.float8_e4m3fn
    rng = np.random.default_rng(21)
    q, kv_rows, _, table, keys, local_key, local_limit = _inputs(rng, 24, 24, 1, dtype)
    newest = 63
    keys[keys == 5] = -1  # a page counts once per token
    keys[:, 0, -1] = 5  # every token's local page, written up to slot `newest`
    local_key[:] = 5
    local_limit[:] = rng.integers(1, newest + 2, 24)
    poisoned = kv_rows.copy()
    poisoned[table[5], newest:] = np.nan
    cache = msa.write_rows(
        pack(poisoned, dtype),
        jnp.asarray(kv_rows[table[5], newest : newest + 1]).astype(jnp.bfloat16),
        jnp.asarray([table[5]], jnp.int32),
        jnp.asarray([newest], jnp.int32),
        max_num_seqs=1,
        interpret=True,
    )
    got = msa_sparse_attention_lse(
        q,
        cache,
        jnp.asarray(keys[:, 0]),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        jnp.asarray(table),
        sm_scale=SCALE,
        block_q=8,
        pages_per_step=pages_per_step,
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


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize(
    "mode,max_union", [("auto", 0.0), ("auto", 1e9), ("on", 0.0)]
)  # regular / wide tiles taken
def test_adaptive_tiles_match_reference(fp4, mode, max_union, monkeypatch):
    """Prefill-size calls past `WIDE_TILE_MIN_TOKENS` tokens: the wide-tile
    branch (64-token tiles, `attend_t`) and the regular one give the
    reference, with tokens padded to whole wide tiles and several calls."""
    monkeypatch.setattr(msa_attention_lse, "WIDE_TILES", mode)
    monkeypatch.setattr(msa_attention_lse, "WIDE_TILE_MIN_TOKENS", 64)
    monkeypatch.setattr(
        msa_attention_lse, "_WIDE_TILE_MAX_RATIO", {"fp4": max_union, "8bit": max_union}
    )
    monkeypatch.setattr(msa_attention, "MAX_TOKENS_PER_CALL", 128)
    num_tokens = 150
    rng = np.random.default_rng(23 + fp4)
    dtype = jnp.float8_e4m3fn
    q, kv_rows, kv, table, keys, local_key, local_limit = _inputs(
        rng, num_tokens, num_tokens - 4, 1, dtype
    )
    extra = {}
    if fp4:
        kv, scales, kv_rows = _fp4_cache(rng, 1)
        extra = dict(scale_cache=jnp.asarray(scales))
    got = msa_sparse_attention_lse(
        q,
        kv,
        jnp.asarray(keys[:, 0]),
        jnp.asarray(local_key),
        jnp.asarray(local_limit),
        jnp.asarray(table),
        sm_scale=SCALE,
        block_q=8 if fp4 else 32,
        pages_per_step=32,
        interpret=True,
        **extra,
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


@pytest.mark.parametrize("num_tokens", [8, 48, 216, 384, 1024])
@pytest.mark.parametrize("width_order", [False, True])
def test_work_kernel_matches_xla_work_list(num_tokens, width_order, monkeypatch):
    """`_work_kernel`'s packed rows equal the XLA work list's word for word:
    tiles of shared, dispersed (four 32-page groups) and no pages, padding
    tiles at the end."""
    rng = np.random.default_rng(num_tokens)
    block_q, topk, pages_per_step = 8, 16, 32
    num_tiles = num_tokens // block_q
    selected = np.full((2, num_tokens, topk), -1, np.int32)
    for c in range(2):
        for t in range(num_tiles - num_tiles // 8):
            union = rng.choice([0, 1, 4, 5, 9, 17, 33, 128])
            pool = t * 512 + rng.permutation(512)[: max(union, 1)]
            for i in range(block_q):
                n = min(topk, union)
                selected[c, t * block_q + i, :n] = rng.choice(pool, n, replace=False)
    args = (jnp.asarray(selected), block_q, pages_per_step)
    kw = dict(align=16, most=64, width_order=width_order)
    calls = []
    kernel_work = msa_attention_lse._kernel_work
    monkeypatch.setattr(
        msa_attention_lse,
        "_kernel_work",
        lambda *a, **k: calls.append(1) or kernel_work(*a, **k),
    )
    got, got_tiles = msa_attention_lse._packed_work(*args, **kw, interpret=True)
    assert calls, "the work list kernel did not build the work list"
    monkeypatch.setattr(msa_attention_lse, "_work_kernel_fits", lambda *a: False)
    want, want_tiles = msa_attention_lse._packed_work(*args, **kw)
    assert got_tiles == want_tiles
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
