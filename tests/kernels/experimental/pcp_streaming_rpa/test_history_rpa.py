# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""CPU checks of the TPU_PCP_HISTORY_RPA composition (history_rpa.py).

The PCP collectives run under ``jax.vmap(axis_name="pcp")``, and the two
Pallas kernels are replaced by dense f32 stand-ins with the contracts the
production kernels have (read from their code):

* current pass: online state of each local row over the history-boundary
  partial virtual page plus the causal current chunk;
* RPAm CACHE_ONLY: attention of each row over its sequence's first
  ``kv_len - q_len`` cached tokens, read through the page table, with LSE.

The composed output must equal dense causal attention in global order. That
checks the per-source metadata, the local page selection (block-table mod),
the whole-virtual-page split with the current pass, the all-gather /
all-to-all routing and the LSE merge.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import history_rpa
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
    pcp_local_token_counts,
)

PCP = 2
PAGE = 16
KV_HEADS = 2
Q_PER_KV = 3  # padded to 4 in the state, as bf16 Q packing does
Q_PAD = 4
HEAD_DIM = 16
LANES = 4
SM_SCALE = HEAD_DIM**-0.5


def _case(interleave):
    vpage = PAGE * PCP
    # (history tokens, query tokens): no history, a partial boundary page,
    # an exact virtual-page boundary, and history shorter than one page.
    reqs = [(0, 37), (5 * vpage + 13, 50), (7 * vpage, 24), (vpage - 3, 9)]
    max_seqs = 6  # two inactive slots
    pages_per_seq = 12
    rng = np.random.default_rng(7)
    q_lens = np.array([q for _, q in reqs])
    hist = np.array([h for h, _ in reqs])
    kv_lens = np.zeros(max_seqs, np.int32)
    kv_lens[: len(reqs)] = hist + q_lens
    cu = np.zeros(max_seqs + 1, np.int32)
    cu[1 : len(reqs) + 1] = np.cumsum(q_lens)
    cu[len(reqs) + 1 :] = cu[len(reqs)]
    total = int(q_lens.sum())
    padded = int(np.ceil(total / (PCP * 32)) * PCP * 32)
    local_t = padded // PCP

    keys = [rng.normal(size=(h + q, KV_HEADS, HEAD_DIM)) for h, q in reqs]
    vals = [rng.normal(size=(h + q, KV_HEADS, HEAD_DIM)) for h, q in reqs]
    qs = [rng.normal(size=(q, KV_HEADS * Q_PER_KV, HEAD_DIM)) * 2 for _, q in reqs]

    # Physical local pages: request i's local page j -> distinct slot; the
    # block table stores slot + k * num_blocks to exercise the mod.
    num_blocks = max_seqs * pages_per_seq
    perm = rng.permutation(num_blocks)
    block_tables = np.zeros((max_seqs, pages_per_seq), np.int32)
    for i in range(max_seqs):
        for j in range(pages_per_seq):
            slot = perm[i * pages_per_seq + j]
            block_tables[i, j] = slot + num_blocks * rng.integers(0, 3)
    k_cache = np.full((PCP, num_blocks, PAGE, KV_HEADS, HEAD_DIM), np.nan)
    v_cache = np.full((PCP, num_blocks, PAGE, KV_HEADS, HEAD_DIM), np.nan)
    cycle = PCP * interleave
    for i, (h, q) in enumerate(reqs):
        for pos in range(h + q):  # the current chunk is written back too
            rank = (pos // interleave) % PCP
            local = (pos // cycle) * interleave + pos % interleave
            j, off = divmod(local, PAGE)
            slot = block_tables[i, j] % num_blocks
            k_cache[rank, slot, off] = keys[i][pos]
            v_cache[rank, slot, off] = vals[i][pos]

    token_order, _ = build_pcp_rank_major_token_order(
        q_lens, PCP, interleave, padded, token_owner_start_offsets_per_req=cu[:-1][: len(reqs)]
    )
    req_of = np.repeat(np.arange(len(reqs)), q_lens)
    tok_of = np.concatenate([np.arange(q) for q in q_lens])
    return dict(
        reqs=reqs, max_seqs=max_seqs, kv_lens=kv_lens, cu=cu, local_t=local_t,
        keys=keys, vals=vals, qs=qs, block_tables=block_tables,
        k_cache=k_cache, v_cache=v_cache, token_order=token_order,
        req_of=req_of, tok_of=tok_of, interleave=interleave,
    )


def _attend(q, k, v):
    """q [H, D], k/v [N, KV, D] -> (o [H, D], lse [H])."""
    k = np.repeat(k, Q_PER_KV, axis=1)
    v = np.repeat(v, Q_PER_KV, axis=1)
    s = np.einsum("hd,nhd->hn", q, k) * SM_SCALE
    m = s.max(-1, keepdims=True)
    p = np.exp(s - m)
    l = p.sum(-1, keepdims=True)
    return np.einsum("hn,nhd->hd", p / l, v), (m + np.log(l))[:, 0], m[:, 0], l[:, 0], np.einsum("hn,nhd->hd", p, v)


def _local_inputs(c):
    """Per-rank Q rows, current-pass state stand-in and the reference."""
    vpage = PAGE * PCP
    lt = c["local_t"]
    q_local = np.zeros((PCP, lt, KV_HEADS * Q_PER_KV, HEAD_DIM))
    m_state = np.full((PCP, KV_HEADS, lt, Q_PAD, LANES), np.nan)
    l_state = np.full((PCP, KV_HEADS, lt, Q_PAD, LANES), np.nan)
    acc_state = np.full((PCP, KV_HEADS, lt, Q_PAD, HEAD_DIM), np.nan)
    expected = np.full((PCP, lt, KV_HEADS * Q_PER_KV, HEAD_DIM), np.nan)
    valid = np.zeros((PCP, lt), bool)
    for dst, src in enumerate(c["token_order"]):
        if src < 0:
            continue
        rank, row = divmod(dst, lt)
        i, t = c["req_of"][src], c["tok_of"][src]
        h, _ = c["reqs"][i]
        pos = h + t
        q = c["qs"][i][t]
        q_local[rank, row] = q
        valid[rank, row] = True
        expected[rank, row] = _attend(q, c["keys"][i][: pos + 1], c["vals"][i][: pos + 1])[0]
        h_full = (h // vpage) * vpage
        _, _, m, l, acc = _attend(q, c["keys"][i][h_full : pos + 1], c["vals"][i][h_full : pos + 1])
        m = m.reshape(KV_HEADS, Q_PER_KV)
        l = l.reshape(KV_HEADS, Q_PER_KV)
        acc = acc.reshape(KV_HEADS, Q_PER_KV, HEAD_DIM)
        m_state[rank, :, row, :Q_PER_KV] = m[..., None]
        l_state[rank, :, row, :Q_PER_KV] = l[..., None]
        acc_state[rank, :, row, :Q_PER_KV] = acc
    return q_local, m_state, l_state, acc_state, expected, valid


def _ref_history_fn(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, num_seqs):
    """Dense stand-in for RPAm CACHE_ONLY with LSE over local pages."""
    del k, v
    k_cache, v_cache = kv_cache
    max_seqs = kv_lens.shape[0]
    pages = page_indices.reshape(max_seqs, -1)
    rows = jnp.arange(q.shape[0])
    seq = jnp.clip(jnp.searchsorted(cu_q_lens, rows, side="right") - 1, 0, max_seqs - 1)
    in_seq = jnp.logical_and(rows < cu_q_lens[jnp.minimum(num_seqs, max_seqs)], seq < num_seqs)
    cache_len = kv_lens - (cu_q_lens[1:] - cu_q_lens[:-1])
    kk = k_cache[pages[seq]].reshape(q.shape[0], -1, KV_HEADS, HEAD_DIM)
    vv = v_cache[pages[seq]].reshape(q.shape[0], -1, KV_HEADS, HEAD_DIM)
    tok_valid = jnp.arange(kk.shape[1])[None, :] < cache_len[seq][:, None]
    kk = jnp.where(tok_valid[..., None, None], kk, 0.0)
    vv = jnp.where(tok_valid[..., None, None], vv, 0.0)
    kk = jnp.repeat(kk, Q_PER_KV, axis=2)
    vv = jnp.repeat(vv, Q_PER_KV, axis=2)
    s = jnp.einsum("thd,tnhd->thn", q, kk) * SM_SCALE
    s = jnp.where(tok_valid[:, None, :], s, -jnp.inf)
    m = jnp.max(s, axis=-1, keepdims=True)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.exp(s - m_safe)
    l = jnp.sum(p, axis=-1, keepdims=True)
    o = jnp.einsum("thn,tnhd->thd", p / jnp.where(l > 0, l, 1.0), vv)
    lse = jnp.where(l[..., 0] > 0, m_safe[..., 0] + jnp.log(jnp.where(l[..., 0] > 0, l[..., 0], 1.0)), -jnp.inf)
    has = jnp.logical_and(in_seq[:, None], l[..., 0] > 0)
    # Like RPAm: rows that attend nothing keep the aliased Q and lse -inf.
    o = jnp.where(has[..., None], o, q)
    lse = jnp.where(has, lse, -jnp.inf)
    return o, kv_cache, lse


def _run(c, history_fn=_ref_history_fn):
    q_local, m_state, l_state, acc_state, expected, valid = _local_inputs(c)
    k_dummy = jnp.zeros((PCP, c["local_t"], KV_HEADS, HEAD_DIM))
    distribution = jnp.array([0, 0, len(c["reqs"])], jnp.int32)

    def body(q, k, kc, vc, m, l, acc):
        out, _ = history_rpa.pcp_history_attention(
            q, k, k, (kc, vc), m, l, acc,
            jnp.asarray(c["kv_lens"]),
            jnp.asarray(c["block_tables"].reshape(-1)),
            jnp.asarray(c["cu"]),
            distribution,
            pcp_size=PCP,
            interleave_size=c["interleave"],
            page_size=PAGE,
            pcp_axis_name="pcp",
            history_fn=history_fn,
        )
        return out

    out = jax.vmap(body, axis_name="pcp")(
        jnp.asarray(q_local, jnp.float32),
        k_dummy,
        jnp.asarray(np.nan_to_num(c["k_cache"]), jnp.float32),
        jnp.asarray(np.nan_to_num(c["v_cache"]), jnp.float32),
        jnp.asarray(m_state, jnp.float32),
        jnp.asarray(l_state, jnp.float32),
        jnp.asarray(acc_state, jnp.float32),
    )
    out = np.asarray(out)[:, :, :, :Q_PER_KV, :].reshape(expected.shape)
    return out, expected, valid


@pytest.mark.parametrize("pair_exchange", ["0", "1"])
@pytest.mark.parametrize("interleave", [8, 16])
def test_history_rpa_composition_matches_dense_causal_attention(interleave, pair_exchange, monkeypatch):
    monkeypatch.setenv("TPU_PCP_RPA_PAIR_EXCHANGE", pair_exchange)
    c = _case(interleave)
    out, expected, valid = _run(c)
    assert valid.sum() == sum(q for _, q in c["reqs"])
    np.testing.assert_allclose(out[valid], expected[valid], rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("interleave", [8, 16])
def test_history_rpa_metadata_matches_layout_counts(interleave):
    c = _case(interleave)
    cu_src, kv_src, hist_local = history_rpa.history_rpa_metadata(
        jnp.asarray(c["kv_lens"]),
        jnp.asarray(c["cu"]),
        jnp.array([0, 0, len(c["reqs"])], jnp.int32),
        pcp_size=PCP,
        interleave_size=interleave,
        page_size=PAGE,
    )
    cu_src, kv_src, hist_local = map(np.asarray, (cu_src, kv_src, hist_local))
    vpage = PAGE * PCP
    for i, (h, q) in enumerate(c["reqs"]):
        counts = pcp_local_token_counts(
            [q], PCP, interleave, token_owner_start_offsets_per_req=[c["cu"][i]]
        )
        assert hist_local[i] == (h // vpage) * PAGE
        for s in range(PCP):
            assert cu_src[s, i + 1] - cu_src[s, i] == counts[s]
            expect_kv = (h // vpage) * PAGE + counts[s] if counts[s] else 0
            assert kv_src[s, i] == expect_kv
    # Inactive slots contribute no rows and no KV.
    assert np.all(kv_src[:, len(c["reqs"]) :] == 0)
    assert np.all(cu_src[:, len(c["reqs"]) :] == cu_src[:, len(c["reqs"]) : len(c["reqs"]) + 1])


def test_merge_ignores_partials_that_attended_nothing():
    """A partial with lse -inf and a non-finite o must not change the output."""
    rng = np.random.default_rng(1)
    t, heads = 3, KV_HEADS * Q_PER_KV
    m = rng.normal(size=(KV_HEADS, t, Q_PAD, LANES)).astype(np.float32)
    l = rng.uniform(1, 2, size=(KV_HEADS, t, Q_PAD, LANES)).astype(np.float32)
    acc = rng.normal(size=(KV_HEADS, t, Q_PAD, HEAD_DIM)).astype(np.float32)
    o = np.full((2, t, heads, HEAD_DIM), np.inf, np.float32)
    lse = np.full((2, t, heads), -np.inf, np.float32)
    out = np.asarray(
        history_rpa.merge_current_state_with_partials(m, l, acc, o, lse)
    )
    expect = np.transpose(acc / l[..., :1], (1, 0, 2, 3))
    np.testing.assert_allclose(out, expect, rtol=1e-6)


# ---------------------------------------------------------------------------
# TPU_PCP_CURRENT_RPA: current chunk on RPA too (pcp_attention_all_rpa).
# Cache pages here use the token-last layout of the HND cache:
# [blocks, 2 (k/v), kv_heads, head_dim, page].
# ---------------------------------------------------------------------------


def _paged_attend(q, pool, kv_lens, page_indices, cu_q_lens, num_seqs, causal):
    """Dense stand-in for RPA over a token-last page pool, with LSE."""
    max_seqs = kv_lens.shape[0]
    pages = page_indices.reshape(max_seqs, -1)
    rows = jnp.arange(q.shape[0])
    seq = jnp.clip(jnp.searchsorted(cu_q_lens, rows, side="right") - 1, 0, max_seqs - 1)
    in_seq = jnp.logical_and(rows < cu_q_lens[jnp.minimum(num_seqs, max_seqs)], seq < num_seqs)
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    kvt = pool[pages[seq]]  # [T, W, 2, KV, D, page]
    kvt = jnp.moveaxis(kvt, -1, 2).reshape(q.shape[0], -1, 2, KV_HEADS, HEAD_DIM)
    n = jnp.arange(kvt.shape[1])[None, :]
    if causal:  # queries are the last q_len KV positions of their sequence
        qpos = kv_lens[seq] - q_lens[seq] + (rows - cu_q_lens[seq])
        ok = n <= qpos[:, None]
    else:
        ok = n < (kv_lens - q_lens)[seq][:, None]
    ok = jnp.logical_and(ok, n < kv_lens[seq][:, None])
    kk = jnp.repeat(kvt[:, :, 0], Q_PER_KV, axis=2)
    vv = jnp.repeat(kvt[:, :, 1], Q_PER_KV, axis=2)
    s = jnp.einsum("thd,tnhd->thn", q, jnp.where(ok[..., None, None], kk, 0.0)) * SM_SCALE
    s = jnp.where(ok[:, None, :], s, -jnp.inf)
    m = jnp.max(s, axis=-1, keepdims=True)
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    p = jnp.exp(s - m_safe)
    l = jnp.sum(p, axis=-1, keepdims=True)
    o = jnp.einsum("thn,tnhd->thd", p / jnp.where(l > 0, l, 1.0), jnp.where(ok[..., None, None], vv, 0.0))
    has = jnp.logical_and(in_seq[:, None], l[..., 0] > 0)
    lse = jnp.where(has, m_safe[..., 0] + jnp.log(jnp.where(l[..., 0] > 0, l[..., 0], 1.0)), -jnp.inf)
    return jnp.where(has[..., None], o, q), lse


def _ref_current_fn(q, k, v, pool, kv_lens, page_indices, cu_q_lens, num_seqs):
    del k, v
    return _paged_attend(q, pool, kv_lens, page_indices, cu_q_lens, num_seqs, causal=True)


def _ref_history_fn_tokenlast(q, k, v, cache, kv_lens, page_indices, cu_q_lens, num_seqs):
    del k, v
    o, lse = _paged_attend(q, cache, kv_lens, page_indices, cu_q_lens, num_seqs, causal=False)
    return o, cache, lse


def _run_all_rpa(c, broken=None):
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import wrapper as pcpw

    q_local, _, _, _, expected, valid = _local_inputs(c)
    lt = c["local_t"]
    lt_pad = -(-lt // PAGE) * PAGE
    # Token-last caches and the per-rank fresh KV of each local row.
    cache = np.stack((np.nan_to_num(c["k_cache"]), np.nan_to_num(c["v_cache"])), axis=2)
    cache = np.moveaxis(cache, 3, -1)  # [P, blocks, 2, KV, D, page]
    fresh = np.zeros((PCP, 2, KV_HEADS, HEAD_DIM, lt_pad))
    pos_l = np.zeros((PCP, lt), np.int32)
    req_l = np.zeros((PCP, lt), np.int32)
    val_l = np.zeros((PCP, lt), bool)
    for dst, src in enumerate(c["token_order"]):
        if src < 0:
            continue
        rank, row = divmod(dst, lt)
        i, t = c["req_of"][src], c["tok_of"][src]
        pos = c["reqs"][i][0] + t
        fresh[rank, 0, :, :, row] = c["keys"][i][pos]
        fresh[rank, 1, :, :, row] = c["vals"][i][pos]
        pos_l[rank, row], req_l[rank, row], val_l[rank, row] = pos, i, True
    kv_lens = jnp.asarray(c["kv_lens"])
    cu = jnp.asarray(c["cu"])
    distribution = jnp.array([0, 0, len(c["reqs"])], jnp.int32)
    rm = pcpw._compute_pcp_rank_major_query_row_mapping_from_metadata(
        kv_lens, cu, distribution, local_padded_tokens=lt, pcp_size=PCP,
        interleave_size=c["interleave"],
    )
    wb = pcpw._compute_pcp_kv_writeback_mapping(
        rm, jnp.asarray(c["block_tables"]), local_kv_cache_num_blocks=cache.shape[1],
        page_size=PAGE, pcp_size=PCP, interleave_size=c["interleave"],
    )
    captured = {}

    def body(q, fr, kc, pos, req, val, rank):
        def write_fn(cache_, cap):
            captured["x"] = cap
            return cache_

        out, _ = history_rpa.pcp_attention_all_rpa(
            q, jnp.zeros((lt, KV_HEADS, HEAD_DIM)), jnp.zeros((lt, KV_HEADS, HEAD_DIM)),
            fr, kc, kv_lens, jnp.asarray(c["block_tables"].reshape(-1)), cu, distribution,
            (pos, req, val), (wb.capture_offsets, wb.cache_owner_ranks, rm.is_valid, rank),
            pcp_size=PCP, interleave_size=c["interleave"], page_size=PAGE, pcp_axis_name="pcp",
            current_fn=broken or _ref_current_fn, history_fn=_ref_history_fn_tokenlast,
            write_fn=write_fn,
        )
        return out, captured["x"]

    out, cap = jax.vmap(body, axis_name="pcp")(
        jnp.asarray(q_local, jnp.float32), jnp.asarray(fresh, jnp.float32),
        jnp.asarray(cache, jnp.float32), jnp.asarray(pos_l), jnp.asarray(req_l),
        jnp.asarray(val_l), jnp.arange(PCP, dtype=jnp.int32),
    )
    return np.asarray(out), np.asarray(cap), expected, valid


def _expected_capture(c, rank, size):
    """Owned fresh tokens, request-major then position order (ring-capture order)."""
    buf = np.zeros((2, KV_HEADS, HEAD_DIM, size))
    k = 0
    for i, (h, q) in enumerate(c["reqs"]):
        for pos in range(h, h + q):
            if (pos // c["interleave"]) % PCP == rank:
                buf[0, :, :, k] = c["keys"][i][pos]
                buf[1, :, :, k] = c["vals"][i][pos]
                k += 1
    return buf


@pytest.mark.parametrize("pair_exchange", ["0", "1"])
@pytest.mark.parametrize("interleave", [8, 16])
def test_all_rpa_composition_matches_dense_causal_attention(interleave, pair_exchange, monkeypatch):
    monkeypatch.setenv("TPU_PCP_RPA_PAIR_EXCHANGE", pair_exchange)
    c = _case(interleave)
    out, cap, expected, valid = _run_all_rpa(c)
    np.testing.assert_allclose(out[valid], expected[valid], rtol=2e-4, atol=2e-4)
    for r in range(PCP):
        np.testing.assert_allclose(cap[r], _expected_capture(c, r, cap.shape[-1]), atol=1e-6)


def test_all_rpa_detects_a_non_causal_current_pass():
    """Mutant: the current call ignores causality, which must change the output."""
    c = _case(16)

    def acausal(q, k, v, pool, kv_lens, page_indices, cu_q_lens, num_seqs):
        q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
        # Attend the whole sequence including later rows of the chunk.
        return _paged_attend(q, pool, kv_lens + 0 * q_lens, page_indices, cu_q_lens, num_seqs, causal=False)

    out, _, expected, valid = _run_all_rpa(c, broken=acausal)
    assert not np.allclose(out[valid], expected[valid], rtol=2e-4, atol=2e-4)


def test_fused_merge_matches_stacked_merge_with_large_and_empty_partials():
    """Fused merge == reference merge, incl. lse 100+ (needs the max shift) and -inf."""
    rng = np.random.default_rng(5)
    t, h, d = 7, 6, 16
    o = [rng.normal(size=(t, h, d)).astype(np.float32) for _ in range(3)]
    lse = [rng.normal(size=(t, h)).astype(np.float32) * 3 + 100 for _ in range(3)]
    lse[2][:3] = -np.inf
    o[2][:3] = np.nan  # an empty partial's o is undefined
    ref = np.asarray(history_rpa.merge_partials(jnp.stack(o), jnp.stack(lse)))
    out = np.asarray(history_rpa.merge_partials_fused(o, lse, jnp.float32))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    out_bf16 = np.asarray(history_rpa.merge_partials_fused(o, lse, jnp.bfloat16).astype(jnp.float32))
    np.testing.assert_allclose(out_bf16, ref, rtol=1e-2, atol=1e-2)
