# SPDX-License-Identifier: Apache-2.0
"""Head-split DCP draft KV (`cp_draft_head_split`, additional_config
draft_kv_head_split) vs the float64 oracle and vs the current DCP-sharded
draft path (`cp_draft_attention`).

Served shape: MiniMax-M3.1 DSpark draft, 4 KV heads and 64 query heads over
TP8 (2 TP replicas per KV head, 8 query heads per rank), fp8 KV cache, DCP 2
(4 DCP groups of 2 ranks = one KV head each) or DCP 8 (one group of all 8
ranks, 4 KV heads). Ranks are simulated in a Python loop; the collectives
(query all-gather, `cp_lse_ag_out_rs`) are emulated in numpy, the same way
for both layouts. Kernels run in the Pallas interpreter.
"""

import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter import cp_draft_attention as cda
from vllm_torchtpu.layers.adapter import cp_draft_head_split as hs

PAGE = 128
HEAD_DIM = 128
TP = 8
KV_HEADS = 4
REPLICAS = TP // KV_HEADS
QH = 8  # query heads per rank
Q_BLOCK = 7
SM_SCALE = HEAD_DIM**-0.5


def _cache(pages, heads, dtype):
    """A page shape for `heads` KV heads (the KV cache spec of the group)."""
    packing = 4 // jnp.dtype(dtype).itemsize
    words = -(-2 * heads // packing)
    return jnp.zeros((pages, PAGE, words, packing, HEAD_DIM), dtype)


def _stored(x, dtype, scale):
    x = jnp.asarray(x)
    if scale is None:
        return np.asarray(x.astype(dtype).astype(jnp.float32), np.float64)
    return np.asarray((x / scale).astype(dtype).astype(jnp.float32), np.float64) * scale


def _cast(x, dtype, scale):
    """The cache entries (unscaled) a write stores for ``x``."""
    x = jnp.asarray(x) / (1.0 if scale is None else scale)
    return np.asarray(x.astype(dtype).astype(jnp.float32))


def _tables(rng, num_seqs, logical):
    # Shuffled pages, so a wrong page lookup reads someone else's KV.
    perm = rng.permutation(num_seqs * logical + 1)[: num_seqs * logical]
    return jnp.asarray(perm.reshape(num_seqs, -1), jnp.int32)


# ---------------------------------------------------------------------------
# Write layout
# ---------------------------------------------------------------------------


def _read_token_heads(cache, dcp, head_ranks):
    """Head-split cache -> per (page, token-head slot) K, V ``[pages, n, D]``."""
    num_pages = cache.shape[0]
    halves = hs._halves(cache.shape[1:], dcp // head_ranks)
    kv = np.asarray(cache.astype(jnp.float32)).reshape(
        num_pages, -1, cache.shape[3] // halves, HEAD_DIM
    )
    return kv[:, :, 0], kv[:, :, 1]


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
@pytest.mark.parametrize("dcp,head_ranks", [(2, 2), (8, 2), (8, 1), (4, 4)])
@pytest.mark.parametrize("prefill", [False, True])
def test_write_kv_places_own_head_tokens(dcp, head_ranks, dtype, prefill):
    """Every rank stores exactly the tokens of global blocks b with
    b % head_ranks == rank % head_ranks, at slot (b % dcp) // head_ranks *
    PAGE + pos % PAGE of page tables[seq, b // dcp]; nothing else changes."""
    rng = np.random.default_rng(dcp * 7 + head_ranks)
    if prefill:  # >= 1024 tokens: the per-run grid writer
        ctx = np.array([300, 0, 2000, 129], np.int32)
        q_len = np.array([900, 300, 1100, 5], np.int32)
    else:
        ctx = np.array([1000, 37, 600, 0, 255, 127], np.int32)
        q_len = np.array([3, 3, 3, 3, 2, 1], np.int32)
    num_seqs = len(ctx)
    total = ctx + q_len
    logical = -(-int(total.max()) // (PAGE * dcp)) + 1
    tables = _tables(rng, num_seqs, logical)
    num_tokens = int(q_len.sum()) + 3  # padding tokens
    scale = None if dtype == jnp.bfloat16 else 0.5
    key = rng.standard_normal((num_tokens, 1, HEAD_DIM)).astype(np.float32)
    value = rng.standard_normal((num_tokens, 1, HEAD_DIM)).astype(np.float32)
    cu = np.concatenate([[0], np.cumsum(q_len)]).astype(np.int32)
    heads = dcp // head_ranks
    sentinel = 3.0
    for rank in range(dcp):
        init = jnp.full_like(_cache(num_seqs * logical + 1, heads, dtype), sentinel)
        cache = hs.write_kv(
            init,
            jnp.asarray(key),
            jnp.asarray(value),
            jnp.asarray(total),
            tables.reshape(-1),
            jnp.asarray(cu),
            dcp=dcp,
            head_ranks=head_ranks,
            rank=jnp.int32(rank),
            k_scale=scale,
            v_scale=scale,
            interpret=True,
        )
        k, v = _read_token_heads(cache, dcp, head_ranks)
        expect_k = np.full(k.shape, sentinel)
        expect_v = np.full(v.shape, sentinel)
        t_np = np.asarray(tables)
        written = set()
        for s in range(num_seqs):
            for j in range(int(q_len[s])):
                tok = cu[s] + j
                pos = int(ctx[s]) + j
                b = pos // PAGE
                if b % head_ranks != rank % head_ranks:
                    continue
                page = t_np[s, b // dcp]
                slot = (b % dcp) // head_ranks * PAGE + pos % PAGE
                expect_k[page, slot] = _cast(key[tok, 0], dtype, scale)
                expect_v[page, slot] = _cast(value[tok, 0], dtype, scale)
                written.add((page, slot))
        # Two token-heads per word (`write_pairs`): a run ending at an even
        # slot fills the word's other half (the next, not yet written
        # position) with the next row; positions are written in order.
        check = np.ones(k.shape[:2], bool)
        if hs._halves(cache.shape[1:], heads) == 2:
            for page, slot in written:
                if slot % 2 == 0 and (page, slot + 1) not in written:
                    check[page, slot + 1] = False
        assert (~check).sum() <= 2 * num_seqs
        np.testing.assert_array_equal(
            k[check], expect_k[check], err_msg=f"rank {rank} K"
        )
        np.testing.assert_array_equal(
            v[check], expect_v[check], err_msg=f"rank {rank} V"
        )


def test_write_pairs_keeps_earlier_half_on_odd_start():
    """Two decode steps of one request at DCP 8 / fp8 (two token-heads per
    word): the second write starts at an odd half and must keep the first
    write's token in the shared word."""
    dcp, head_ranks = 8, 2
    rng = np.random.default_rng(3)
    tables = jnp.asarray([[2, 0, 1]], jnp.int32)
    init = _cache(4, dcp // head_ranks, jnp.float8_e4m3fn)
    k = rng.standard_normal((4, 1, HEAD_DIM)).astype(np.float32)
    v = rng.standard_normal((4, 1, HEAD_DIM)).astype(np.float32)
    # Step 1 writes positions 130..132 (3 tokens), step 2 writes 133..133.
    c1 = hs.write_kv(
        init,
        jnp.asarray(k[:3]),
        jnp.asarray(v[:3]),
        jnp.asarray([133]),
        tables.reshape(-1),
        jnp.asarray([0, 3]),
        dcp=dcp,
        head_ranks=head_ranks,
        rank=jnp.int32(1),
        interpret=True,
    )
    c2 = hs.write_kv(
        c1,
        jnp.asarray(k[3:]),
        jnp.asarray(v[3:]),
        jnp.asarray([134]),
        tables.reshape(-1),
        jnp.asarray([0, 1]),
        dcp=dcp,
        head_ranks=head_ranks,
        rank=jnp.int32(1),
        interpret=True,
    )
    got_k, got_v = _read_token_heads(c2, dcp, head_ranks)
    for i, pos in enumerate(range(130, 134)):
        slot = (1 % dcp) // head_ranks * PAGE + pos % PAGE  # block 1 -> slot group 0
        want = np.asarray(
            jnp.asarray(k[i, 0]).astype(jnp.float8_e4m3fn).astype(jnp.float32)
        )
        np.testing.assert_array_equal(got_k[2, slot], want, err_msg=f"pos {pos}")
        want = np.asarray(
            jnp.asarray(v[i, 0]).astype(jnp.float8_e4m3fn).astype(jnp.float32)
        )
        np.testing.assert_array_equal(got_v[2, slot], want, err_msg=f"pos {pos}")


# ---------------------------------------------------------------------------
# Window attention: kernel vs XLA reference vs float64 oracle
# ---------------------------------------------------------------------------


def _hs_case(dcp, head_ranks, dtype, seed, ctx, q_len, window):
    rng = np.random.default_rng(seed)
    ctx = np.asarray(ctx, np.int32)
    q_len = np.asarray(q_len, np.int32)
    num_seqs = len(ctx)
    total = ctx + q_len
    logical = -(-int(total.max()) // (PAGE * dcp)) + 1
    tables = _tables(rng, num_seqs, logical)
    scale = None if dtype == jnp.bfloat16 else 0.5
    ck = rng.standard_normal((int(ctx.sum()), 1, HEAD_DIM)).astype(np.float32)
    cv = rng.standard_normal((int(ctx.sum()), 1, HEAD_DIM)).astype(np.float32)
    num_tokens = int(q_len.sum()) + 5
    q = jnp.asarray(rng.standard_normal((num_tokens, 2 * QH, HEAD_DIM)), jnp.bfloat16)
    cu_q = jnp.asarray(np.concatenate([[0], np.cumsum(q_len)]), jnp.int32)
    cu_ctx = jnp.asarray(np.concatenate([[0], np.cumsum(ctx)]), jnp.int32)
    caches = [
        hs.write_kv(
            _cache(num_seqs * logical + 1, dcp // head_ranks, dtype),
            jnp.asarray(ck),
            jnp.asarray(cv),
            jnp.asarray(ctx),
            tables.reshape(-1),
            cu_ctx,
            dcp=dcp,
            head_ranks=head_ranks,
            rank=jnp.int32(r),
            k_scale=scale,
            v_scale=scale,
            interpret=True,
        )
        for r in range(dcp)
    ]
    return dict(
        ctx=ctx,
        q_len=q_len,
        total=jnp.asarray(total),
        tables=tables.reshape(-1),
        cu_q=cu_q,
        q=q,
        caches=caches,
        scale=scale,
        keys=_stored(ck, dtype, scale),
        values=_stored(cv, dtype, scale),
        cu_ctx=np.asarray(cu_ctx),
        window=window,
        dcp=dcp,
        head_ranks=head_ranks,
    )


def _oracle(case, rank):
    q = np.asarray(case["q"], np.float64)
    num_tokens, heads, _ = q.shape
    out = np.zeros(q.shape)
    lse = np.full((num_tokens, heads), -np.inf)
    cu_q = np.asarray(case["cu_q"])
    for s, c in enumerate(case["ctx"]):
        lo, hi = int(cu_q[s]), int(cu_q[s + 1])
        if hi == lo or c == 0:
            continue
        k = case["keys"][case["cu_ctx"][s] : case["cu_ctx"][s + 1], 0]
        v = case["values"][case["cu_ctx"][s] : case["cu_ctx"][s + 1], 0]
        k_pos = np.arange(c)
        q_pos = c + np.arange(hi - lo)
        mine = (k_pos // PAGE) % case["head_ranks"] == rank % case["head_ranks"]
        mask = mine[None, :] & (q_pos[:, None] - k_pos[None, :] < case["window"])
        sc = np.einsum("qhd,kd->hqk", q[lo:hi], k) * SM_SCALE
        sc = np.where(mask[None], sc, -np.inf)
        m = sc.max(-1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        p = np.exp(sc - m)
        l = p.sum(-1, keepdims=True)
        o = np.einsum("hqk,kd->qhd", p, v) / np.where(l > 0, l, 1.0).transpose(1, 0, 2)
        out[lo:hi] = np.where((l > 0).transpose(1, 0, 2), o, 0.0)
        l0 = l[..., 0]
        lse[lo:hi] = np.where(
            l0 > 0, m[..., 0] + np.log(np.where(l0 > 0, l0, 1)), -np.inf
        ).T
    return out, lse


def _bf16_ulp(*xs):
    mag = np.maximum(np.max([np.abs(x) for x in xs], axis=0), 2.0**-126)
    return 2.0 ** (np.floor(np.log2(mag)) - 7)


def _hs_attend(case, cache_rank, fn=hs.attend_cached, **over):
    kw = dict(
        head_ranks=case["head_ranks"],
        max_query_len=Q_BLOCK,
        sliding_window=case["window"],
        sm_scale=SM_SCALE,
        dcp=case["dcp"],
        rank=jnp.int32(cache_rank),
        k_scale=case["scale"],
        v_scale=case["scale"],
    )
    if fn is hs.attend_cached:
        kw["interpret"] = True
    kw.update(over)
    o, l = fn(
        case["q"],
        case["caches"][cache_rank],
        case["total"],
        case["tables"],
        case["cu_q"],
        **kw,
    )
    return np.asarray(o, np.float64), np.asarray(l, np.float64)


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.bfloat16])
@pytest.mark.parametrize("dcp,head_ranks", [(2, 2), (8, 2), (8, 1)])
def test_attend_matches_oracle_and_reference(dcp, head_ranks, dtype):
    case = _hs_case(
        dcp,
        head_ranks,
        dtype,
        seed=dcp + head_ranks,
        ctx=[1000, 37, 600, 0, 2000, 130],
        q_len=[7, 7, 3, 7, 0, 7],
        window=512,
    )
    for r in range(dcp):
        got = _hs_attend(case, r)
        ref = _hs_attend(case, r, hs.attend_cached_reference)
        want_o, want_l = _oracle(case, r)
        live = np.isfinite(want_l)
        assert (live == np.isfinite(got[1])).all(), r
        assert (live == np.isfinite(ref[1])).all(), r
        tol = _bf16_ulp(got[0], want_o) + 1e-5
        assert (np.abs(got[0] - want_o) <= tol).all(), (
            r,
            (np.abs(got[0] - want_o) - tol).max(),
        )
        assert np.abs(got[1] - want_l)[live].max(initial=0.0) < 1e-4
        tol = _bf16_ulp(got[0], ref[0]) + 1e-5
        assert (np.abs(got[0] - ref[0]) <= tol).all(), r


@pytest.mark.parametrize("fault", ["wrong_rank", "window_minus_one_page", "no_v_scale"])
def test_attend_fault_controls(fault):
    """The oracle comparison catches each injected fault (positive controls)."""
    case = _hs_case(
        8,
        2,
        jnp.float8_e4m3fn,
        seed=11,
        ctx=[1000, 37, 600, 2000],
        q_len=[7, 7, 3, 7],
        window=512,
    )
    rank, over = 0, {}
    if fault == "wrong_rank":
        over = dict(rank=jnp.int32(1))
    elif fault == "window_minus_one_page":
        over = dict(sliding_window=512 - PAGE)
    elif fault == "no_v_scale":
        over = dict(v_scale=None)
    o, l = _hs_attend(case, rank, **over)
    want_o, want_l = _oracle(case, rank)
    live = np.isfinite(want_l) & np.isfinite(l)
    err_o = np.abs(o - want_o) / (_bf16_ulp(o, want_o) + 1e-5)
    assert (
        err_o[live].max(initial=0) > 1.0
        or np.abs(l - want_l)[live].max(initial=0) >= 1e-4
    )


# ---------------------------------------------------------------------------
# Whole draft attention layer at the served TP8 / 4-KV-head shape:
# head-split vs the current sharded path
# ---------------------------------------------------------------------------


def _ag_rs_merge(parts, my_index):
    """`dcp_comm.cp_lse_ag_out_rs` emulated: parts = per group rank
    (out [T, n * QH, D] bf16, lse [T, n * QH] f32); returns rank
    `my_index`'s heads (bf16 out [T, QH, D], f32 lse [T, QH])."""
    lses = np.stack([np.asarray(l, np.float32) for _, l in parts])
    with np.errstate(divide="ignore", invalid="ignore"):
        m = lses.max(0)
        m_safe = np.where(np.isfinite(m), m, 0.0)
        merged = m_safe + np.log(np.exp(lses - m_safe).sum(0))
        weighted = None
        for o, l in parts:
            w = np.exp(np.asarray(l, np.float32) - merged)[..., None]
            term = np.where(w > 0, np.asarray(o.astype(jnp.float32)) * w, 0.0).astype(
                np.float32
            )
            weighted = (
                term if weighted is None else (weighted + term).astype(np.float32)
            )
    sl = slice(my_index * QH, (my_index + 1) * QH)
    out = jnp.asarray(weighted[:, sl]).astype(jnp.bfloat16)
    return out, jnp.asarray(merged[:, sl], jnp.float32)


@functools.cache
def _jit_old_attend(dcp, kv_heads, window):
    return jax.jit(
        functools.partial(
            cda.attend_cached_flash,
            kv_heads=kv_heads,
            max_query_len=Q_BLOCK,
            sliding_window=window,
            sm_scale=SM_SCALE,
            dcp=dcp,
            k_scale=0.5,
            v_scale=0.5,
            interpret=True,
        )
    )


@functools.cache
def _jit_new_attend(dcp, head_ranks, window):
    return jax.jit(
        functools.partial(
            hs.attend_cached,
            head_ranks=head_ranks,
            max_query_len=Q_BLOCK,
            sliding_window=window,
            sm_scale=SM_SCALE,
            dcp=dcp,
            k_scale=0.5,
            v_scale=0.5,
            interpret=True,
        )
    )


@functools.cache
def _jit_block():
    def f(q, k, v, total, cu_q, ctx_out, ctx_lse):
        out, lse = cda.attend_block(
            q, k, v, total, cu_q, max_query_len=Q_BLOCK, sm_scale=SM_SCALE, causal=False
        )
        return cda.merge(ctx_out.astype(out.dtype), ctx_lse, out, lse)[0]

    return jax.jit(f)


def _layer_case(num_reqs, dcp, seed, window):
    """The served layer: all 8 TP ranks' caches in both layouts, written
    from the same context K/V (per KV head), and every rank's queries and
    block K/V."""
    rng = np.random.default_rng(seed)
    ctx = rng.integers(1, 3 * window, num_reqs).astype(np.int32)
    ctx[rng.random(num_reqs) < 0.1] = 0  # requests with no context yet
    q_len = np.full(num_reqs, Q_BLOCK, np.int32)
    total = ctx + q_len
    logical = -(-int(total.max()) // (PAGE * dcp)) + 1
    tables = _tables(rng, num_reqs, logical).reshape(-1)
    num_ctx = int(ctx.sum())
    num_tokens = num_reqs * Q_BLOCK + 3
    cu_ctx = jnp.asarray(np.concatenate([[0], np.cumsum(ctx)]), jnp.int32)
    cu_q = jnp.asarray(np.concatenate([[0], np.cumsum(q_len)]), jnp.int32)
    ck = rng.standard_normal((max(num_ctx, 1), KV_HEADS, HEAD_DIM)).astype(np.float32)
    cv = rng.standard_normal((max(num_ctx, 1), KV_HEADS, HEAD_DIM)).astype(np.float32)
    ck, cv = jnp.asarray(ck[:num_ctx]), jnp.asarray(cv[:num_ctx])
    q = jnp.asarray(
        rng.standard_normal((TP, num_tokens, QH, HEAD_DIM)) * 1.5, jnp.bfloat16
    )
    bk = rng.standard_normal((num_tokens, KV_HEADS, HEAD_DIM)).astype(np.float32)
    bv = rng.standard_normal((num_tokens, KV_HEADS, HEAD_DIM)).astype(np.float32)
    group_heads = dcp // REPLICAS
    head_ranks = min(dcp, REPLICAS)
    old_caches, new_caches = [], []
    num_pages = num_reqs * logical + 1
    for t in range(TP):
        g, r = divmod(t, dcp)
        first_head = g * dcp // REPLICAS
        old_caches.append(
            cda.write_kv(
                _cache(num_pages, group_heads, jnp.float8_e4m3fn),
                ck[:, first_head : first_head + group_heads],
                cv[:, first_head : first_head + group_heads],
                jnp.asarray(ctx),
                tables,
                cu_ctx,
                dcp=dcp,
                rank=jnp.int32(r),
                k_scale=0.5,
                v_scale=0.5,
            )
        )
        head = t // REPLICAS
        new_caches.append(
            hs.write_kv(
                _cache(num_pages, group_heads, jnp.float8_e4m3fn),
                ck[:, head : head + 1],
                cv[:, head : head + 1],
                jnp.asarray(ctx),
                tables,
                cu_ctx,
                dcp=dcp,
                head_ranks=head_ranks,
                rank=jnp.int32(r),
                k_scale=0.5,
                v_scale=0.5,
                interpret=True,
            )
        )
    return SimpleNamespace(
        ck=np.asarray(ck),
        cv=np.asarray(cv),
        cu_ctx=np.asarray(cu_ctx),
        ctx=ctx,
        total=jnp.asarray(total),
        tables=tables,
        cu_q=cu_q,
        q=q,
        bk=bk,
        bv=bv,
        old=old_caches,
        new=new_caches,
        dcp=dcp,
        head_ranks=head_ranks,
        window=window,
        num_tokens=num_tokens,
    )


def _block_kv(case, t):
    head = t // REPLICAS
    k = (jnp.asarray(case.bk[:, head : head + 1]) / 0.5).astype(jnp.float8_e4m3fn)
    v = (jnp.asarray(case.bv[:, head : head + 1]) / 0.5).astype(jnp.float8_e4m3fn)
    k = (k.astype(jnp.float32) * 0.5).astype(jnp.bfloat16)
    v = (v.astype(jnp.float32) * 0.5).astype(jnp.bfloat16)
    return k, v


def _layer_outputs(case, layout):
    """Every TP rank's draft attention output [TP, T, QH, D] (bf16)."""
    dcp = case.dcp
    if layout == "old":
        size, attend = dcp, _jit_old_attend(dcp, dcp // REPLICAS, case.window)
        caches = case.old
    else:
        size, attend = (
            case.head_ranks,
            _jit_new_attend(dcp, case.head_ranks, case.window),
        )
        caches = case.new
    parts = {}
    for t in range(TP):
        lo = t - t % size
        q_all = jnp.concatenate([case.q[u] for u in range(lo, lo + size)], axis=1)
        parts[t] = attend(
            q_all,
            caches[t],
            case.total,
            case.tables,
            case.cu_q,
            rank=jnp.int32(t % dcp),
        )
    outs = []
    for t in range(TP):
        lo = t - t % size
        ctx_out, ctx_lse = _ag_rs_merge(
            [parts[u] for u in range(lo, lo + size)], t % size
        )
        k, v = _block_kv(case, t)
        outs.append(
            _jit_block()(case.q[t], k, v, case.total, case.cu_q, ctx_out, ctx_lse)
        )
    return np.stack([np.asarray(o.astype(jnp.float32)) for o in outs])


def _fp8(x):
    """The values an fp8 cache with scale 0.5 holds for ``x`` (float64)."""
    x = (jnp.asarray(x, jnp.float32) / 0.5).astype(jnp.float8_e4m3fn)
    return np.asarray(x.astype(jnp.float32), np.float64) * 0.5


def _layer_oracle(case):
    """float64 draft attention of every TP rank's query heads over the
    window of its KV head's cached context plus the (non-causal) block."""
    ck, cv = _fp8(case.ck), _fp8(case.cv)
    bk, bv = _fp8(case.bk), _fp8(case.bv)
    q = np.asarray(case.q.astype(jnp.float32), np.float64)
    out = np.zeros(q.shape)
    for s, c in enumerate(case.ctx):
        lo, hi = s * Q_BLOCK, (s + 1) * Q_BLOCK
        k_pos = np.arange(c)
        q_pos = c + np.arange(Q_BLOCK)
        win = q_pos[:, None] - k_pos[None, :] < case.window
        mask = np.concatenate([win, np.ones((Q_BLOCK, Q_BLOCK), bool)], axis=1)
        for t in range(TP):
            head = t // REPLICAS
            k = np.concatenate(
                [ck[case.cu_ctx[s] : case.cu_ctx[s + 1], head], bk[lo:hi, head]]
            )
            v = np.concatenate(
                [cv[case.cu_ctx[s] : case.cu_ctx[s + 1], head], bv[lo:hi, head]]
            )
            sc = np.einsum("qhd,kd->hqk", q[t, lo:hi], k) * SM_SCALE
            sc = np.where(mask[None], sc, -np.inf)
            p = np.exp(sc - sc.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            out[t, lo:hi] = np.einsum("hqk,kd->qhd", p, v)
    return out


def _compare(case, old, new, seed):
    """Both paths against the float64 oracle: elementwise error, and draft
    tokens (the 64 heads through a random LM head, argmax) on every row
    whose float64 top-2 logit gap exceeds both paths' logit errors."""
    num = len(case.ctx) * Q_BLOCK
    old, new = old[:, :num], new[:, :num]
    ref = _layer_oracle(case)[:, :num]
    err_old, err_new = np.abs(old - ref), np.abs(new - ref)
    rng = np.random.default_rng(seed)
    head = rng.standard_normal((TP * QH * HEAD_DIM, 512)) / np.sqrt(TP * QH * HEAD_DIM)

    def logits(x):
        return x.transpose(1, 0, 2, 3).reshape(num, -1) @ head

    lg_ref, lg_old, lg_new = logits(ref), logits(old), logits(new)
    noise = max(np.abs(lg_old - lg_ref).max(), np.abs(lg_new - lg_ref).max())
    srt = np.sort(lg_ref, axis=-1)
    clear = srt[:, -1] - srt[:, -2] > 2 * noise
    top = {
        k: v.argmax(-1) for k, v in (("ref", lg_ref), ("old", lg_old), ("new", lg_new))
    }
    return dict(
        rows=num,
        bitwise=round(float((old == new).mean()), 4),
        max_diff=float(np.abs(old - new).max()),
        max_err_old=float(err_old.max()),
        max_err_new=float(err_new.max()),
        mean_err_old=float(err_old.mean()),
        mean_err_new=float(err_new.mean()),
        tok_agree=float((top["old"] == top["new"]).mean()),
        tok_old_ref=float((top["old"] == top["ref"]).mean()),
        tok_new_ref=float((top["new"] == top["ref"]).mean()),
        clear_rows=int(clear.sum()),
        clear_ok=bool(
            (top["old"] == top["ref"])[clear].all()
            and (top["new"] == top["ref"])[clear].all()
        ),
    )


def _check(res):
    # Not bitwise: the old path rounds dcp partials to bf16 before the merge,
    # head-split head_ranks (fewer), and the window kernels sum differently.
    # Head-split must be no less accurate, and the draft tokens equal
    # wherever the reference argmax is clear of both paths' noise.
    assert res["max_err_new"] <= 1.25 * res["max_err_old"] + 1e-6, res
    assert res["mean_err_new"] <= 1.05 * res["mean_err_old"], res
    assert res["clear_ok"], res
    assert res["tok_new_ref"] >= res["tok_old_ref"] - 2 / res["rows"], res


@pytest.mark.parametrize("dcp", [2, 8])
@pytest.mark.parametrize("num_reqs", [1, 4, 16, 64])
def test_head_split_layer_matches_sharded(num_reqs, dcp):
    """Head-split vs current sharded draft attention (write + window attend
    + merge + block), all 8 TP ranks, both against float64 (`_check`)."""
    window = 1024 if num_reqs <= 16 else 512
    case = _layer_case(num_reqs, dcp, seed=num_reqs * 10 + dcp, window=window)
    old = _layer_outputs(case, "old")
    new = _layer_outputs(case, "new")
    res = _compare(case, old, new, seed=num_reqs)
    print(f"R={num_reqs} DCP{dcp}:", res)
    _check(res)


def test_head_split_layer_served_window():
    """The served 4096-token window at DCP 8, R = 4."""
    case = _layer_case(4, 8, seed=99, window=4096)
    res = _compare(
        case, _layer_outputs(case, "old"), _layer_outputs(case, "new"), seed=1
    )
    print("R=4 DCP8 window 4096:", res)
    _check(res)


# ---------------------------------------------------------------------------
# All-layer write == per-layer writes; the model's head-split context write
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dcp", [2, 8])
def test_write_layers_equals_per_layer(dcp):
    rng = np.random.default_rng(dcp)
    ctx = np.array([500, 0, 129], np.int32)
    q_len = np.array([20, 9, 300], np.int32)
    total = ctx + q_len
    logical = -(-int(total.max()) // (PAGE * dcp)) + 1
    tables = _tables(rng, 3, logical).reshape(-1)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(q_len)]), jnp.int32)
    num_tokens = int(q_len.sum()) + 4
    keys = jnp.asarray(rng.standard_normal((3, num_tokens, 1, HEAD_DIM)), jnp.bfloat16)
    values = jnp.asarray(
        rng.standard_normal((3, num_tokens, 1, HEAD_DIM)), jnp.bfloat16
    )
    caches = [_cache(3 * logical + 1, dcp // 2, jnp.float8_e4m3fn) for _ in range(3)]
    for rank in range(dcp):
        kw = dict(
            dcp=dcp,
            head_ranks=2,
            rank=jnp.int32(rank),
            k_scale=0.7,
            v_scale=0.3,
            interpret=True,
        )
        got = hs.write_layers(
            caches, keys, values, jnp.asarray(total), tables, cu, **kw
        )
        for i in range(3):
            want = hs.write_kv(
                caches[i], keys[i], values[i], jnp.asarray(total), tables, cu, **kw
            )
            np.testing.assert_array_equal(
                np.asarray(got[i].astype(jnp.float32)),
                np.asarray(want.astype(jnp.float32)),
            )


def test_model_head_split_context_write_matches_per_layer(monkeypatch):
    """`MiniMaxM3DSparkModel.store_context_kv` with the all-layer op hands the
    op each layer's own-head K/V exactly as the per-layer path writes it."""
    from vllm_torchtpu.models.minimax_m3 import dspark

    monkeypatch.setattr(dspark, "SPLIT_FC", False)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 1)
    torch.manual_seed(0)
    hidden, num_tokens, layers = 64, 9, 3

    class _Norm(torch.nn.Module):
        def forward(self, x):
            return x * 0.5

    class _Rope:
        def __call__(self, positions, k, q):
            return k + positions[:, None].to(k.dtype), q

    seen = {"per_layer": [], "op": None}

    def make_layer(i):
        attn = torch.nn.Module()
        w = torch.randn(hidden, 3 * HEAD_DIM)
        attn.qkv_proj = lambda x, w=w: (x @ w, None)
        attn.q_size, attn.kv_size, attn.head_dim, attn.num_kv_heads = (
            HEAD_DIM,
            HEAD_DIM,
            HEAD_DIM,
            1,
        )
        attn.k_norm = _Norm()
        attn.rotary_emb = _Rope()

        def write(layer, k, v, cache, md, i=i):
            seen["per_layer"].append((i, k.clone(), v.clone()))

        impl = SimpleNamespace(
            sharded_draft_dcp=dict(
                dcp=8, kv_head_replicas=2, head_split=True, k_scale=1.0, v_scale=1.0
            ),
            write_sharded_draft_kv=write,
        )
        attn.attn = SimpleNamespace(
            impl=impl,
            kv_cache=torch.zeros(1),
            draft_dcp_rank=torch.tensor([0], dtype=torch.int32),
        )
        return SimpleNamespace(self_attn=attn)

    model = SimpleNamespace(
        layers=[make_layer(i) for i in range(layers)],
        fc=lambda x: x[:, :hidden],
        hidden_norm=_Norm(),
        rank_rows_context_kv=None,
        _head_split_kv_write_op=None,
        _head_split_context=False,
    )
    for name in (
        "prepare_draft_kv_write",
        "_write_head_split_context_kv",
        "store_context_kv",
    ):
        setattr(model, name, getattr(dspark.MiniMaxM3DSparkModel, name).__get__(model))
    monkeypatch.setattr(
        dspark.cp_draft_head_split,
        "write_layers_op",
        lambda *cfg: ("op", cfg),
    )

    def fake_write_layers(*, keys, values, kv_caches, attn_metadata, dcp_rank, op):
        seen["op"] = (op, keys.clone(), values.clone())

    monkeypatch.setattr(
        dspark.cp_draft_head_split, "write_layers_with_dcp", fake_write_layers
    )
    aux = torch.randn(num_tokens, 2 * hidden)
    positions = torch.arange(num_tokens)
    md = SimpleNamespace()
    # Per-layer path (no all-layer op yet).
    model.store_context_kv(aux, positions, (md,) * layers)
    model.prepare_draft_kv_write()
    assert model._head_split_kv_write_op == ("op", (8, 2, 1.0, 1.0, layers))
    model.store_context_kv(aux, positions, (md,) * layers)
    op, keys, values = seen["op"]
    assert keys.shape == (layers, num_tokens, 1, HEAD_DIM)
    for i, k, v in seen["per_layer"]:
        torch.testing.assert_close(
            keys[i], k.reshape(num_tokens, 1, HEAD_DIM), rtol=0, atol=0
        )
        torch.testing.assert_close(
            values[i], v.reshape(num_tokens, 1, HEAD_DIM), rtol=0, atol=0
        )
    # Distinct metadata per layer: the per-layer path.
    seen["per_layer"].clear()
    model.store_context_kv(
        aux, positions, tuple(SimpleNamespace() for _ in range(layers))
    )
    assert len(seen["per_layer"]) == layers


@pytest.mark.parametrize("dcp,head_ranks", [(2, 2), (8, 2)])
@pytest.mark.parametrize("num_tokens,max_seqs", [(32, 4), (512, 73), (8192, 146)])
def test_head_split_compiles_for_tpu7x(dcp, head_ranks, num_tokens, max_seqs):
    """Deviceless tpu7x AOT compile of the write and the attention at the
    served draft shapes (fp8 cache, 4096 window)."""
    from jax.experimental import topologies
    from jax.sharding import (
        AbstractDevice,
        AbstractMesh,
        SingleDeviceSharding,
        use_abstract_mesh,
    )

    try:
        topology = topologies.get_topology_desc("tpu7x:2x2x1", "tpu")
    except Exception as e:  # no libtpu
        pytest.skip(f"no deviceless tpu7x topology: {e}")
    sharding = SingleDeviceSharding(topology.devices[0])

    def spec(shape, dt):
        return jax.ShapeDtypeStruct(shape, dt, sharding=sharding)

    words = -(-2 * (dcp // head_ranks) // 4)
    cache = spec((65536, PAGE, words, 4, HEAD_DIM), jnp.float8_e4m3fn)
    meta = (
        spec((max_seqs,), jnp.int32),
        spec((max_seqs * 64,), jnp.int32),
        spec((max_seqs + 1,), jnp.int32),
        spec((1,), jnp.int32),
    )

    def write(caches, keys, values, seq_lens, tables, cu_q, rank):
        return hs.write_layers(
            caches,
            keys,
            values,
            seq_lens,
            tables,
            cu_q,
            dcp=dcp,
            head_ranks=head_ranks,
            rank=rank[0],
            k_scale=0.7,
            v_scale=0.3,
        )

    def attend(q, cache, seq_lens, tables, cu_q, rank):
        return hs.attend_cached(
            q,
            cache,
            seq_lens,
            tables,
            cu_q,
            head_ranks=head_ranks,
            max_query_len=Q_BLOCK,
            sliding_window=4096,
            sm_scale=SM_SCALE,
            dcp=dcp,
            rank=rank[0],
            k_scale=0.7,
            v_scale=0.3,
        )

    tpu7x = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(tpu7x):
        kv = spec((5, num_tokens, 1, HEAD_DIM), jnp.bfloat16)
        lw = jax.jit(write).lower((cache,) * 5, kv, kv, *meta)
        if num_tokens <= 1024:
            q = spec((num_tokens, head_ranks * QH, HEAD_DIM), jnp.bfloat16)
            la = jax.jit(attend).lower(q, cache, *meta)
    assert "tpu_custom_call" in lw.compile().as_text()
    if num_tokens <= 1024:
        assert "tpu_custom_call" in la.compile().as_text()
