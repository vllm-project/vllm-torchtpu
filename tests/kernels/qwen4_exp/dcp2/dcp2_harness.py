# SPDX-License-Identifier: Apache-2.0
"""CPU equality harness for main-KV DCP-2 (VLLM_TPU_QSA_MAIN_KV_DCP=2).

Layout (docs: ~/q38f/dcp/CONTRACT.md, integration lead 2d):
  D = 2, B = pool.shape[1]. A DCP main-KV table entry covers D*B tokens:
  token t of a request lives at page ``table[req, t // (D*B)]``, row ``t % B``
  on owner rank ``(t // B) % D``. The replicated layout keeps the same token at
  ``full_table[req, t // B]``, row ``t % B``. Logical B-page j is wholly owned
  by rank ``j % D``, so a rank pool is the replicated pool's owned pages copied
  under the DCP table, with every other page NaN-poisoned.

Everything below is adapter-based. An adapter has the signature of the
production function it stands in for (see README.md), so plugging the real
kernel in is a one-line change:
  write adapter    = pooled_qsa.qsa_cache_writes(..., dcp=(rank_i32[1], 2))
  metadata adapter = qsa_rpa.virtual_page_metadata(..., dcp=(rank_i32[1], 2))
  attend adapter   = (q, gate, blocks, lengths, table, cache, *, kv_heads,
                      ratio, k_scale, v_scale, dcp) -> (out f32, lse f32)
  combine adapter  = qsa_rpa.dcp_combine(out_pair, lse_pair, gate, rank_i32[1])

Tolerance policy: writes, ownership and per-rank compacted metadata are
compared EXACTLY (raw bytes / int equality). Attention outputs are not
bit-equal (the split softmax reorders the accumulation): partial and combined
pre-gate outputs must agree with the float32 reference at fp32 level
(``Tolerance.rtol``/``atol``) and gated BF16 outputs within
``Tolerance.max_ulp`` BF16 ulps of the replicated oracle; every check reports
max abs / rel diff and the bit-identical fraction. Any NaN in an output fails.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

D = 2
FP8 = jnp.float8_e4m3fn
FP8_NAN_BYTE = 0x7F
# 448, the largest finite E4M3. Poison for rows a correct kernel may LOAD but
# must mask (rows past kv_len inside a fetched micro-page / tile, the null
# page 0 that invalid page columns point at): p = 0 times 448 is 0, while an
# unmasked read swamps the softmax. NaN there would flag every kernel that
# masks by multiplication, which the replicated kernels do too.
FP8_SENTINEL_BYTE = 0x7E


# ----------------------------------------------------------------------------
# Ownership


def owner_of_token(t, page_size):
    """Contract owner: B-page parity."""
    return (np.asarray(t) // page_size) % D


def owner_of_group(g, page_size, ratio):
    return owner_of_token(np.asarray(g) * ratio, page_size)


# Deliberately wrong owner rules, used by the fault controls.
def owner_flipped(t, page_size):
    return 1 - owner_of_token(t, page_size)


def owner_token_parity(t, page_size):
    return np.asarray(t) % D


def rank_operand(rank):
    return jnp.asarray([rank], jnp.int32)


# ----------------------------------------------------------------------------
# Pool layout


def random_fp8(rng, shape):
    """Finite FP8 E4M3 values with |x| in [2^-4, 3.75] (random bytes, fast:
    a float64 normal of a 300 MB pool costs tens of seconds)."""
    sign = rng.integers(0, 2, shape, dtype=np.uint8) << 7
    exp = rng.integers(3, 9, shape, dtype=np.uint8) << 3
    mant = rng.integers(0, 8, shape, dtype=np.uint8)
    return from_raw(sign | exp | mant)


def raw(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint8))


def from_raw(bytes_, dtype=FP8):
    return jax.lax.bitcast_convert_type(jnp.asarray(bytes_, jnp.uint8), dtype)


@dataclasses.dataclass
class Layout:
    """Page tables of one step. Pages of the two layouts never alias."""

    page_size: int
    full_table: np.ndarray  # [R, cols_full]  replicated, B tokens per entry
    dcp_table: np.ndarray  # [R, cols_dcp]   DCP, D*B tokens per entry
    alloc_tokens: np.ndarray  # [R] tokens each request has storage for
    poison_page: int  # a page no table entry owns (NaN in rank pools)
    free_pages: np.ndarray  # pages owned by neither main layout


def build_layout(rng, alloc_tokens, page_size, num_pages, *, reserve=0):
    """Random disjoint page ids for both layouts (never page 0, the null
    page); unused pages are returned in ``free_pages`` (compressed/ring pages
    of the unified pool)."""
    alloc_tokens = np.asarray(alloc_tokens, np.int64)
    cols_full = max(1, int(-(-alloc_tokens.max(initial=1) // page_size)))
    cols_dcp = -(-cols_full // D)
    need_full = [int(-(-n // page_size)) for n in alloc_tokens]
    need_dcp = [-(-n // D) for n in need_full]
    total = sum(need_full) + sum(need_dcp) + 2 + reserve
    if total > num_pages:
        raise ValueError(f"layout needs {total} pages, pool has {num_pages}")
    # Page 0 is the null page (idle rows / invalid columns point there).
    free = iter(1 + rng.permutation(num_pages - 1))
    poison = int(next(free))
    full = np.full((len(alloc_tokens), cols_full), poison, np.int32)
    dcp = np.full((len(alloc_tokens), cols_dcp), poison, np.int32)
    for r in range(len(alloc_tokens)):
        for j in range(need_full[r]):
            full[r, j] = next(free)
        for j in range(need_dcp[r]):
            dcp[r, j] = next(free)
    rest = np.fromiter(free, np.int64).astype(np.int32)
    return Layout(page_size, full, dcp, alloc_tokens, poison, rest)


def project_rank_pool(full_bytes, layout, rank, *, keep_pages=(), base=None):
    """Rank ``rank``'s pool (raw bytes): the replicated pool's owned B-pages
    under the DCP table, ``keep_pages`` (replicated regions: compressed keys,
    rings) copied verbatim, everything else NaN (or ``base``'s bytes)."""
    if base is None:
        pool = np.full_like(full_bytes, FP8_NAN_BYTE)
    else:
        pool = np.array(base, copy=True)
    keep = np.asarray(list(keep_pages), np.int64)
    if len(keep):
        pool[keep] = full_bytes[keep]
    cols_alloc = -(-layout.alloc_tokens // layout.page_size)
    for r in range(len(layout.alloc_tokens)):
        for j in range(rank, int(cols_alloc[r]), D):
            pool[layout.dcp_table[r, j // D]] = full_bytes[layout.full_table[r, j]]
    return pool


def poison_tail_rows(pool_bytes, layout, rank, seq_tokens, *, value=FP8_SENTINEL_BYTE):
    """Poison owned rows at or beyond each request's sequence length (rows no
    query of this step may attend). Default 0x7E (448, see
    FP8_SENTINEL_BYTE); 0x7F = NaN for the strict variant."""
    pool = np.array(pool_bytes, copy=True)
    b = layout.page_size
    cols_alloc = -(-layout.alloc_tokens // b)
    for r, n in enumerate(np.asarray(seq_tokens)):
        for j in range(rank, int(cols_alloc[r]), D):
            lo = j * b
            if lo + b > n:
                pool[layout.dcp_table[r, j // D], max(0, n - lo) :] = value
    return pool


# ----------------------------------------------------------------------------
# Selection: global token lists and the per-rank compacted metadata


def selected_token_ids(blocks, lengths, ratio):
    """[N, (budget + 1) * ratio] global token ids (-1 = none) attended by each
    row: the first ``min(len // ratio, budget)`` blocks plus the partial tail
    group, exactly as ``virtual_page_metadata`` lays them out."""
    blocks = np.asarray(blocks)
    lengths = np.asarray(lengths)
    n, budget = blocks.shape
    complete = lengths // ratio
    count = np.minimum(complete, budget)
    groups = np.concatenate([blocks, np.zeros((n, 1), blocks.dtype)], 1)
    cols = np.arange(budget + 1)[None]
    groups = np.where(cols == count[:, None], complete[:, None], groups)
    tokens = groups[:, :, None] * ratio + np.arange(ratio)
    valid = (cols < count[:, None])[..., None] | (
        (cols == count[:, None])[..., None]
        & (np.arange(ratio) < (lengths % ratio)[:, None, None])
    )
    return np.where(valid, tokens, -1).reshape(n, -1).astype(np.int64)


def reference_metadata(blocks, lengths, table, *, ratio=4, page_size, dcp=None,
                       order="ascending"):
    """Expected ``virtual_page_metadata`` output: (pages [N, budget + 1],
    kv_lens [N], counts [N] of owned complete groups, tail_owned [N]).

    ``dcp=None``: the replicated layout (B tokens per entry, input order kept).
    ``dcp=(rank, D)``: owned complete groups compacted to the front in
    ``order`` ("ascending" group id per the contract, or "stable" input
    order), then the partial tail group if owned; kv_lens = owned tokens.
    Invalid columns are 0, as in the replicated function.
    """
    blocks, lengths, table = map(np.asarray, (blocks, lengths, table))
    n, budget = blocks.shape
    micro = page_size // ratio
    pages = np.zeros((n, budget + 1), np.int32)
    kv_lens = np.zeros(n, np.int32)
    counts = np.zeros(n, np.int32)
    tail_owned = np.zeros(n, bool)
    rank, size = (0, 1) if dcp is None else (int(np.asarray(dcp[0]).reshape(())), dcp[1])
    span = size * page_size
    for i in range(n):
        complete = int(lengths[i]) // ratio
        groups = [int(g) for g in blocks[i, : min(complete, budget)]]
        own = [g for g in groups if (g * ratio // page_size) % size == rank]
        if dcp is not None and order == "ascending":
            own = sorted(own)
        tail = int(lengths[i]) % ratio
        has_tail = tail > 0 and (complete * ratio // page_size) % size == rank
        cols = own + ([complete] if has_tail else [])
        for c, g in enumerate(cols):
            t = g * ratio
            pages[i, c] = table[i, t // span] * micro + (t % page_size) // ratio
        counts[i] = len(own)
        tail_owned[i] = has_tail
        kv_lens[i] = len(own) * ratio + (tail if has_tail else 0)
    return pages, kv_lens, counts, tail_owned


def check_metadata(actual, expected):
    """Exact: owned pages in compacted order and owned kv_lens per row."""
    pages, kv_lens = np.asarray(actual[0]), np.asarray(actual[1])
    e_pages, e_lens, e_counts, e_tail = expected
    np.testing.assert_array_equal(kv_lens, e_lens, "owned kv_lens")
    for i in range(len(e_lens)):
        used = int(e_counts[i] + e_tail[i])
        np.testing.assert_array_equal(
            pages[i, :used], e_pages[i, :used], f"row {i} owned pages"
        )


# ----------------------------------------------------------------------------
# Reference attention (float32, gather + where: poison never reaches the math)


@functools.partial(jax.jit, static_argnames=("kv_heads", "kv_head_offset"))
def _partial_chunk(q, rows, live, cache, k_scale, v_scale, *, kv_heads, kv_head_offset):
    combined = cache.reshape(-1, cache.shape[2] * cache.shape[3], cache.shape[-1])
    sel = combined[rows]  # [n, S, slots, Dc]
    lo = 2 * kv_head_offset
    key = sel[:, :, lo : lo + 2 * kv_heads : 2].astype(jnp.float32) * k_scale
    value = sel[:, :, lo + 1 : lo + 2 * kv_heads : 2].astype(jnp.float32) * v_scale
    key = jnp.where(live[..., None, None], key, 0)
    value = jnp.where(live[..., None, None], value, 0)
    n, heads, dim = q.shape
    per = heads // kv_heads
    qq = q.astype(jnp.float32).reshape(n, kv_heads, per, dim)
    qq = jnp.pad(qq, ((0, 0), (0, 0), (0, 0), (0, key.shape[-1] - dim)))
    score = jnp.einsum(
        "nhqd,nshd->nhqs", qq, key, precision=jax.lax.Precision.HIGHEST
    ) * (dim**-0.5)
    score = jnp.where(live[:, None, None, :], score, -jnp.inf)
    top = jnp.max(score, -1, keepdims=True)
    safe = jnp.where(jnp.isfinite(top), top, 0)
    ex = jnp.where(live[:, None, None, :], jnp.exp(score - safe), 0)
    mass = ex.sum(-1, keepdims=True)
    out = jnp.einsum("nhqs,nshd->nhqd", ex, value, precision=jax.lax.Precision.HIGHEST)
    out = out / jnp.where(mass > 0, mass, 1)
    lse = jnp.where(mass > 0, safe + jnp.log(jnp.where(mass > 0, mass, 1)), -jnp.inf)
    out = out[..., :dim].reshape(n, heads, dim)
    return out, lse.reshape(n, heads)


def reference_attend(q, gate, blocks, lengths, table, cache, *, kv_heads, ratio,
                     k_scale=1.0, v_scale=1.0, dcp=None, kv_head_offset=0,
                     owner=owner_of_token, chunk=64):
    """Attention over the row's selected tokens that this rank owns.

    Signature of the DCP attend adapters. Returns ``(out [N, H, D] f32
    normalized over owned keys, ungated; lse [N, H] f32 natural log)``; a row
    with no owned key gets out = 0, lse = -inf. ``dcp=None`` is the replicated
    reference (every token, B tokens per table entry). ``owner`` is the
    token -> rank rule (overridden only by the fault controls).
    """
    del gate
    b = cache.shape[1]
    tokens = selected_token_ids(blocks, lengths, ratio)
    table = np.asarray(table)
    if dcp is None:
        live = tokens >= 0
        span = b
    else:
        rank, size = int(np.asarray(dcp[0]).reshape(())), dcp[1]
        assert size == D
        live = (tokens >= 0) & (owner(np.maximum(tokens, 0), b) == rank)
        span = size * b
    safe = np.maximum(tokens, 0)
    col = np.minimum(safe // span, table.shape[1] - 1)
    page = np.take_along_axis(table, col, 1)
    rows = np.where(live, page.astype(np.int64) * b + safe % b, 0)
    n = len(tokens)
    pad = (-n) % chunk
    qp = jnp.pad(jnp.asarray(q), ((0, pad), (0, 0), (0, 0)))
    rows_p = np.pad(rows, ((0, pad), (0, 0)))
    live_p = np.pad(live, ((0, pad), (0, 0)))
    outs, lses = [], []
    for s in range(0, n + pad, chunk):
        o, l = _partial_chunk(
            qp[s : s + chunk], jnp.asarray(rows_p[s : s + chunk], jnp.int32),
            jnp.asarray(live_p[s : s + chunk]), cache, jnp.float32(k_scale),
            jnp.float32(v_scale), kv_heads=kv_heads, kv_head_offset=kv_head_offset,
        )
        outs.append(o)
        lses.append(l)
    out = jnp.concatenate(outs)[:n]
    lse = jnp.concatenate(lses)[:n]
    # NaN query rows (padding) own no key: keep them 0 / -inf.
    empty = ~live.any(1)
    out = jnp.where(empty[:, None, None], 0, out)
    lse = jnp.where(empty[:, None], -jnp.inf, lse)
    return out, lse


def reference_combine(out_pair, lse_pair, gate, rank, *, out_dtype=jnp.bfloat16):
    """Pure-numpy-equivalent pair LSE merge + gate (qsa_rpa.dcp_combine contract)."""
    size, _, heads_all, _ = out_pair.shape
    heads = heads_all // size
    r = int(np.asarray(rank).reshape(()))
    o = jnp.asarray(out_pair, jnp.float32)[:, :, r * heads : (r + 1) * heads]
    l = jnp.asarray(lse_pair, jnp.float32)[:, :, r * heads : (r + 1) * heads]
    m = jnp.max(l, 0)
    finite = jnp.isfinite(m)
    w = jnp.where(finite[None], jnp.exp(l - jnp.where(finite, m, 0)[None]), 0)
    total = w.sum(0)
    out = jnp.where(w[..., None] > 0, w[..., None] * o, 0).sum(0)
    out = out / jnp.where(total > 0, total, 1)[..., None]
    return gate_output(out, gate, finite).astype(out_dtype)


def combine_lse(lse_pair):
    m = jnp.max(lse_pair, 0)
    finite = jnp.isfinite(m)
    s = jnp.where(finite[None], jnp.exp(lse_pair - jnp.where(finite, m, 0)[None]), 0).sum(0)
    return jnp.where(finite, jnp.where(finite, m, 0) + jnp.log(jnp.where(s > 0, s, 1)), -jnp.inf)


def gate_output(out, gate, live):
    """The replicated kernels' gate rounding (qsa_rpa / block_sparse)."""
    gated = bf16(
        bf16(out).astype(jnp.float32)
        * bf16(jax.nn.sigmoid(jnp.asarray(gate).astype(jnp.float32))).astype(jnp.float32)
    )
    return jnp.where(live[..., None], gated, 0)


def bf16(x):
    from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16 as _bf16

    return _bf16(x)


# Fault-control adapters (each MUST be detected).
def fault_attend(owner):
    return functools.partial(reference_attend, owner=owner)


def combine_rank0_only(out_pair, lse_pair, gate, rank):
    heads = out_pair.shape[2] // out_pair.shape[0]
    r = int(np.asarray(rank).reshape(()))
    o = out_pair[0, :, r * heads : (r + 1) * heads]
    live = jnp.isfinite(lse_pair[0, :, r * heads : (r + 1) * heads])
    return gate_output(o, gate, live).astype(jnp.bfloat16)


def combine_plain_average(out_pair, lse_pair, gate, rank):
    heads = out_pair.shape[2] // out_pair.shape[0]
    r = int(np.asarray(rank).reshape(()))
    o = out_pair[:, :, r * heads : (r + 1) * heads].mean(0)
    live = jnp.isfinite(lse_pair[:, :, r * heads : (r + 1) * heads]).any(0)
    return gate_output(o, gate, live).astype(jnp.bfloat16)


# ----------------------------------------------------------------------------
# Checks


@dataclasses.dataclass(frozen=True)
class Tolerance:
    rtol: float = 1e-5  # f32 pre-gate outputs vs the f32 reference
    atol: float = 1e-5
    lse_atol: float = 1e-5  # scaled by max(1, |lse|)
    # Gated BF16 vs the replicated oracle. 2, not 1: the kernels round twice
    # (bf16(out), then bf16(bf16(out) * bf16(sigmoid(gate)))), so a 1-ulp
    # flip of bf16(out) from f32 reordering can reach 2 ulps of the gated
    # value. The f32 pre-gate outputs are held to rtol/atol above.
    max_ulp: float = 2.0
    # ulps are measured at max(|x|, max|head vector| / ulp_floor_ratio): a
    # near-zero element produced by cancellation is judged at its vector's
    # scale, not its own (f32 summation order moves it by ~1e-7 x scale).
    ulp_floor_ratio: float = 64.0


DEFAULT_TOL = Tolerance()


def bf16_ulp(x):
    x = np.abs(np.asarray(x, np.float32))
    exp = np.floor(np.log2(np.maximum(x, np.float32(2.0**-126))))
    return np.float32(2.0) ** (exp - 7)


def diff_report(actual, expected):
    a = np.asarray(actual, np.float64)
    e = np.asarray(expected, np.float64)
    both = np.isfinite(a) & np.isfinite(e)
    d = np.abs(a - e)[both]
    rel = d / np.maximum(np.abs(e[both]), 1e-30)
    return dict(
        max_abs=float(d.max(initial=0)),
        max_rel=float(rel.max(initial=0)),
        bit_identical=float(np.mean(a == e)) if a.size else 1.0,
    )


def assert_no_nan(x, what):
    x = np.asarray(x, np.float32)
    bad = np.argwhere(np.isnan(x))
    assert not len(bad), f"{what}: NaN leaked at {bad[:5].tolist()} ({len(bad)} total)"


def check_partial(out, lse, ref_out, ref_lse, kv_lens, tol=DEFAULT_TOL, what="partial"):
    """One rank's (out, lse) vs the f32 reference restricted to owned keys."""
    out = np.asarray(out)
    lse = np.asarray(lse)
    assert out.dtype == np.float32 and lse.dtype == np.float32, (out.dtype, lse.dtype)
    assert_no_nan(out, f"{what} out")
    assert not np.isnan(lse).any(), f"{what} lse NaN"
    empty = np.asarray(kv_lens) == 0
    assert np.all(out[empty] == 0), f"{what}: rows with no owned key must be 0"
    assert np.all(np.isneginf(lse[empty])), f"{what}: rows with no owned key need lse=-inf"
    assert np.all(np.isfinite(lse[~empty])), f"{what}: owned rows need finite lse"
    ref_out, ref_lse = np.asarray(ref_out), np.asarray(ref_lse)
    np.testing.assert_allclose(out, ref_out, rtol=tol.rtol, atol=tol.atol, err_msg=what)
    fin = np.isfinite(ref_lse)
    np.testing.assert_array_equal(np.isfinite(lse), fin, f"{what} lse support")
    err = np.abs(lse[fin] - ref_lse[fin]) / np.maximum(1, np.abs(ref_lse[fin]))
    assert err.max(initial=0) <= tol.lse_atol, f"{what} lse err {err.max()}"
    return diff_report(out, ref_out)


def check_gated(actual, oracle, live, tol=DEFAULT_TOL, what="gated"):
    """Gated BF16 DCP output vs the replicated oracle: no NaN, exact zeros on
    rows no rank owns, |diff| <= max_ulp BF16 ulps elementwise."""
    a = np.asarray(actual, np.float32)
    e = np.asarray(oracle, np.float32)
    assert a.shape == e.shape, (a.shape, e.shape)
    assert_no_nan(a, what)
    live = np.asarray(live, bool)
    assert np.all(a[~live] == 0), f"{what}: rows with no key anywhere must be 0"
    mag = np.maximum(np.abs(e), np.abs(a))
    floor = np.abs(e).max(-1, keepdims=True) / tol.ulp_floor_ratio
    ulps = np.abs(a - e) / bf16_ulp(np.maximum(mag, floor))
    rep = diff_report(a, e)
    rep["max_ulp"] = float(ulps.max(initial=0))
    rep["max_ulp_elementwise"] = float((np.abs(a - e) / bf16_ulp(mag)).max(initial=0))
    assert rep["max_ulp"] <= tol.max_ulp, f"{what}: {rep}"
    return rep


def check_combined_prenorm(out_pair, lse_pair, ref_out, ref_lse, tol=DEFAULT_TOL):
    """f32 merge of the partials vs the f32 full reference (pre-gate)."""
    o = np.asarray(out_pair, np.float64)
    l = np.asarray(lse_pair, np.float64)
    m = l.max(0)
    fin = np.isfinite(m)
    w = np.where(fin[None], np.exp(l - np.where(fin, m, 0)[None]), 0)
    merged = (w[..., None] * np.where(w[..., None] > 0, o, 0)).sum(0) / np.where(
        w.sum(0) > 0, w.sum(0), 1
    )[..., None]
    np.testing.assert_allclose(merged, ref_out, rtol=tol.rtol, atol=tol.atol,
                               err_msg="combined pre-gate")
    lse = np.asarray(combine_lse(jnp.asarray(lse_pair)))
    rf = np.isfinite(ref_lse)
    np.testing.assert_array_equal(np.isfinite(lse), rf, "combined lse support")
    err = np.abs(lse[rf] - np.asarray(ref_lse)[rf]) / np.maximum(1, np.abs(ref_lse[rf]))
    assert err.max(initial=0) <= tol.lse_atol, f"combined lse err {err.max()}"
    return diff_report(merged, ref_out)


# ----------------------------------------------------------------------------
# Attention cases: decode verify rows and prefill chunks


@dataclasses.dataclass
class AttentionCase:
    q: jax.Array  # [N, D*H, Dh] both pair ranks' heads, rank 0 first
    gate: jax.Array  # [N, D*H, Dh]
    blocks: np.ndarray  # [N, budget] replicated selection (unsorted, -1 suffix)
    lengths: np.ndarray  # [N] causal kv length per row (0 = idle / padding)
    row_req: np.ndarray  # [N]
    starts: np.ndarray  # [R + 1] request row ranges
    layout: Layout
    full_pool: jax.Array  # replicated pool (finite everywhere)
    rank_pools: list  # [D] poisoned per-rank pools
    heads: int  # H per TP rank
    ratio: int
    k_scale: float
    v_scale: float

    @property
    def full_rows(self):
        return self.layout.full_table[self.row_req]

    @property
    def dcp_rows(self):
        return self.layout.dcp_table[self.row_req]


def random_blocks(rng, lengths, ratio, budget):
    """Per row: min(complete, budget) distinct complete groups, random order."""
    lengths = np.asarray(lengths)
    blocks = np.full((len(lengths), budget), -1, np.int32)
    for i, n in enumerate(lengths):
        complete = int(n) // ratio
        count = min(complete, budget)
        if count:
            blocks[i, :count] = (
                rng.choice(complete, count, replace=False)
                if complete > count
                else rng.permutation(complete)
            )
    return blocks


def make_case(rng, *, row_lengths, row_req, seq_tokens, page_size=512, heads=3,
              head_dim=256, ratio=4, budget=512, k_scale=0.375, v_scale=2.5,
              num_pages=None, poison_tail=FP8_SENTINEL_BYTE,
              null_page=FP8_SENTINEL_BYTE, nan_rows=()):
    """Build both layouts and pools for the given rows.

    ``seq_tokens[r]``: KV rows request r has written (the max row length);
    rows of the pools at or past it are stale (finite) in the replicated pool
    and ``poison_tail`` in the rank pools (None = leave the projection).
    Unallocated pages and the replicated layout's pages are NaN in the rank
    pools; the null page 0 is ``null_page``.
    ``nan_rows``: query rows (padding) whose q/gate are NaN; their length
    must be 0.
    """
    row_lengths = np.asarray(row_lengths, np.int32)
    row_req = np.asarray(row_req, np.int32)
    seq_tokens = np.asarray(seq_tokens, np.int64)
    alloc = np.maximum(seq_tokens, 1)
    if num_pages is None:
        num_pages = int(sum(-(-alloc // page_size)) * 2 + len(alloc) * 2 + 4)
    layout = build_layout(rng, alloc, page_size, num_pages)
    full = random_fp8(rng, (num_pages, page_size, 1, 4, head_dim))
    full_bytes = raw(full)
    pools = []
    for r in range(D):
        p = project_rank_pool(full_bytes, layout, r)
        if poison_tail is not None:
            p = poison_tail_rows(p, layout, r, seq_tokens, value=poison_tail)
        p[0] = null_page
        pools.append(from_raw(p))
    n = len(row_lengths)
    q = rng.normal(0, 1, (n, D * heads, head_dim)).astype(np.float32)
    g = rng.normal(0, 1, (n, D * heads, head_dim)).astype(np.float32)
    for i in nan_rows:
        assert row_lengths[i] == 0
        q[i] = np.nan
        g[i] = np.nan
    starts = np.searchsorted(row_req, np.arange(len(alloc) + 1)).astype(np.int32)
    return AttentionCase(
        q=jnp.asarray(q, jnp.bfloat16), gate=jnp.asarray(g, jnp.bfloat16),
        blocks=random_blocks(rng, row_lengths, ratio, budget), lengths=row_lengths,
        row_req=row_req, starts=starts, layout=layout, full_pool=full,
        rank_pools=pools, heads=heads, ratio=ratio, k_scale=k_scale, v_scale=v_scale,
    )


def decode_case(rng, base_lengths, *, rows_per_seat=4, pad_rows=0, **kw):
    """MTP verify: seat s holds rows of lengths base+0..base+3 (base 0 = idle
    seat); ``pad_rows`` NaN padding rows at the end (bucket padding)."""
    lengths, req = [], []
    for s, base in enumerate(base_lengths):
        for i in range(rows_per_seat):
            lengths.append(base + i if base else 0)
            req.append(s)
    seq = [b + rows_per_seat - 1 if b else 0 for b in base_lengths]
    nan_rows = list(range(len(lengths), len(lengths) + pad_rows))
    lengths += [0] * pad_rows
    req += [len(base_lengths) - 1] * pad_rows
    return make_case(rng, row_lengths=lengths, row_req=req, seq_tokens=seq,
                     nan_rows=nan_rows, **kw)


def prefill_case(rng, chunks, **kw):
    """``chunks`` = [(cached_prefix, chunk_tokens), ...]: one request each."""
    lengths, req, seq = [], [], []
    for r, (prefix, count) in enumerate(chunks):
        lengths += list(range(prefix + 1, prefix + count + 1))
        req += [r] * count
        seq.append(prefix + count)
    return make_case(rng, row_lengths=lengths, row_req=req, seq_tokens=seq, **kw)


@dataclasses.dataclass
class AttentionResult:
    partial: list  # per rank (out, lse)
    gated: list  # per TP rank gated bf16 [N, H, Dh]
    reports: dict


def run_dcp_attention(case, attend=reference_attend, combine=None, tol=DEFAULT_TOL,
                      check_partials=True):
    """Run both ranks through ``attend`` on their poisoned pools, combine for
    each TP rank's heads with ``combine`` (default qsa_rpa.dcp_combine) and
    check every stage against the references. Returns the gated outputs."""
    if combine is None:
        from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

        combine = qsa_rpa.dcp_combine
    opts = dict(kv_heads=1, ratio=case.ratio, k_scale=case.k_scale, v_scale=case.v_scale)
    b = case.layout.page_size
    reports = {}
    partial = []
    for r in range(D):
        dcp = (rank_operand(r), D)
        out, lse = attend(case.q, case.gate, case.blocks, case.lengths, case.dcp_rows,
                          case.rank_pools[r], dcp=dcp, **opts)[:2]
        partial.append((out, lse))
        if check_partials:
            if attend is reference_attend:
                ref_out, ref_lse = out, lse  # same computation; contract checks only
            else:
                ref_out, ref_lse = reference_attend(
                    case.q, case.gate, case.blocks, case.lengths, case.dcp_rows,
                    case.rank_pools[r], dcp=dcp, **opts)
            _, kv_lens, _, _ = reference_metadata(
                case.blocks, case.lengths, case.dcp_rows, ratio=case.ratio,
                page_size=b, dcp=(r, D))
            reports[f"partial{r}"] = check_partial(
                out, lse, ref_out, ref_lse, kv_lens, tol, what=f"rank {r} partial")
    out_pair = jnp.stack([jnp.asarray(p[0], jnp.float32) for p in partial])
    lse_pair = jnp.stack([jnp.asarray(p[1], jnp.float32) for p in partial])
    full_out, full_lse = reference_attend(
        case.q, case.gate, case.blocks, case.lengths, case.full_rows, case.full_pool, **opts)
    if check_partials:
        reports["combined_f32"] = check_combined_prenorm(
            out_pair, lse_pair, np.asarray(full_out), np.asarray(full_lse), tol)
    h = case.heads
    gated = []
    for r in range(D):
        gate_r = case.gate[:, r * h : (r + 1) * h]
        gated.append(combine(out_pair, lse_pair, gate_r, rank_operand(r)))
    return AttentionResult(partial, gated, reports), (full_out, full_lse)


def merge_pair(out_pair, lse_pair, rank, heads):
    """Float32 pre-gate merge of the pair for TP rank ``rank``'s heads (the
    math of dcp_combine before its gate)."""
    o = jnp.asarray(out_pair, jnp.float32)[:, :, rank * heads : (rank + 1) * heads]
    l = jnp.asarray(lse_pair, jnp.float32)[:, :, rank * heads : (rank + 1) * heads]
    m = jnp.max(l, 0)
    fin = jnp.isfinite(m)
    w = jnp.where(fin[None], jnp.exp(l - jnp.where(fin, m, 0)[None]), 0)
    total = w.sum(0)
    out = jnp.where(w[..., None] > 0, w[..., None] * o, 0).sum(0)
    return out / jnp.where(total > 0, total, 1)[..., None], combine_lse(l)


def check_pregate(merged_out, merged_lse, rep_out, rep_lse, live, tol=DEFAULT_TOL,
                  what="pre-gate"):
    """Merged f32 DCP attention vs the replicated route's f32 pre-gate output
    (``return_lse=True`` without dcp): fp32-level agreement, same lse."""
    a, e = np.asarray(merged_out, np.float32), np.asarray(rep_out, np.float32)
    assert_no_nan(a, what)
    live = np.asarray(live, bool)
    a, e = np.where(live[..., None], a, 0), np.where(live[..., None], e, 0)
    rep = diff_report(a, e)
    scale = np.abs(e).max(-1, keepdims=True)
    rep["max_err_over_vector_max"] = float((np.abs(a - e) / np.maximum(scale, 1e-30)).max(initial=0))
    la, le = np.asarray(merged_lse)[live], np.asarray(rep_lse)[live]
    rep["lse_max_abs"] = float(np.abs(la - le).max(initial=0))
    assert rep["max_err_over_vector_max"] <= tol.rtol * 10 + tol.atol, f"{what}: {rep}"
    assert rep["lse_max_abs"] <= tol.lse_atol * max(1.0, float(np.abs(le).max(initial=1))), \
        f"{what}: {rep}"
    return rep


def replicated_gated(case, full_out, full_lse):
    """Gated BF16 of the f32 replicated reference, per TP rank."""
    h = case.heads
    live = np.isfinite(np.asarray(full_lse))
    return [gate_output(full_out[:, r * h : (r + 1) * h], case.gate[:, r * h : (r + 1) * h],
                        jnp.asarray(live[:, r * h : (r + 1) * h])).astype(jnp.bfloat16)
            for r in range(D)]
