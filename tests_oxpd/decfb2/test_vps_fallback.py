# SPDX-License-Identifier: Apache-2.0
"""PD DEC-FB-2: VPS per-row full-vocabulary fallback and per-candidate draws (vp_sampler.vp_verify_fb), torch CPU.

References are the PRODUCTION stock verify (`tpu_runner._verify_fused_impl` extracted by ast, top_k_top_p.py and
rejection_sampler.py loaded from the tree; the harness is tests/q38mm_dstock/test_vp_sampler.py's). The TP ranks run
in threads with a barrier all-gather, so every rank's own computation is exercised and the outputs must be replicated.
Run: python -m pytest -q -p no:cacheprovider --noconftest tests/decfb2/test_vps_fallback.py
Mutation-checked by tests/decfb2/q38_decfb2_vpsfb_mutrun.py (Q38MM_SRC_ROOT points at a mutated copy).
"""

import os
import sys
import threading

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q38mm_dstock"))
import test_vp_sampler as base  # noqa: E402  (loads the stock modules and vp_sampler from Q38MM_SRC_ROOT or the tree)

VP, STOCK, GOLDEN, problem = base.VP, base.STOCK, base.GOLDEN, base.problem


@pytest.fixture(autouse=True, scope="module")
def _harness_modules_installed():
    """test_vp_sampler's stubs only while these tests run (sys.modules is left as found otherwise)."""
    with base.HARNESS.active():
        yield


def stock_full(p, rates=GOLDEN, synthetic=True, bonus_u=None, recover_u=None):
    """The stock fused verify on the full vocabulary path (topk_candidates=0)."""
    return STOCK(p["logits"], p["bonus_idx"], p["target_idx"], p["temps"], p["top_k"], p["top_p"],
                 p["bonus_u"] if bonus_u is None else bonus_u, p["accept_u"],
                 p["recover_u"] if recover_u is None else recover_u, p["draft"], p["lengths"], p["seg"], p["grp"],
                 rates, p["k"], synthetic, 0)


def run_tp(tp, fn):
    """fn(rank, all_gather) on tp threads; all_gather concatenates the ranks' tensors along the last dim."""
    slots, results, errors = [None] * tp, [None] * tp, []
    bar = threading.Barrier(tp)

    def ag_for(r):
        def ag(x):
            slots[r] = x
            bar.wait()
            out = torch.cat(list(slots), dim=-1)
            bar.wait()
            return out
        return ag

    def body(r):
        try:
            results[r] = fn(r, ag_for(r))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
            bar.abort()

    th = [threading.Thread(target=body, args=(r,)) for r in range(tp)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        raise errors[0]
    return results


def vpfb(p, tp=8, kc=32, block=32, F=0, num_active=None, cand=False, bonus_u=None, recover_u=None, fb_u=None,
         rates=GOLDEN, synthetic=True):
    V = p["logits"].shape[1]
    vs = V // tp
    na = torch.tensor(p["bonus_idx"].shape[0] if num_active is None else num_active, dtype=torch.int32)

    def fn(r, ag):
        return VP.vp_verify_fb(
            p["logits"][:, r * vs:(r + 1) * vs], r * vs, ag, p["bonus_idx"], p["target_idx"], p["temps"],
            p["top_k"], p["top_p"], p["bonus_u"] if bonus_u is None else bonus_u, p["accept_u"],
            p["recover_u"] if recover_u is None else recover_u, p["draft"], p["lengths"], p["seg"], p["grp"],
            rates, p["k"], synthetic, na, fallback_rows=F, cand_draws=cand, fb_u=fb_u, kc=kc, block=block)

    outs = run_tp(tp, fn)
    for t, s in outs[1:]:
        assert torch.equal(t, outs[0][0]) and torch.equal(s, outs[0][1]), "ranks disagree"
    return outs[0]


def rows_of(p, req):
    """Logits rows of request `req`: its k target rows and its bonus row."""
    k = p["k"]
    return list(range(req * (k + 1), (req + 1) * (k + 1)))


# ---- 1. the extended candidates: first kc = local_candidates, sentinel = the true (kc+1)-th value ----
@pytest.mark.parametrize("seed", range(4))
def test_ext_candidates_and_sentinel(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(-6, 6, (12, 400), generator=g).to(torch.bfloat16)      # heavy ties
    v, i = VP.local_candidates(x, 1000, 8, 20)
    v2, i2, s = VP.local_candidates_ext(x, 1000, 8, 20)
    assert torch.equal(v, v2) and torch.equal(i, i2)
    ref = torch.sort(x.float(), dim=-1, descending=True, stable=True).values[:, 8]
    assert torch.equal(s.squeeze(-1).float(), ref)


# ---- 2. the completeness test is exact: tie-flagged <=> the candidates miss a token the top-k keeps ----
def _truth_incomplete(logits, cand_ids, top_k, temps):
    """Brute force on the full rows: does the stock top-k keep a token that is not a candidate?"""
    scaled = logits.float() / (temps.float() + VP.TEMPERATURE_EPS)
    kth = torch.sort(scaled, dim=-1, descending=True).values.gather(1, (top_k.long() - 1))
    kept = scaled >= kth
    inc = torch.zeros(logits.shape[0], dtype=torch.long)
    for r in range(logits.shape[0]):
        inc[r] = len(set(torch.nonzero(kept[r]).flatten().tolist()) - set(cand_ids[r].tolist()))
    return inc > 0


@pytest.mark.parametrize("seed", range(6))
def test_tie_flags_exact(seed):
    g = torch.Generator().manual_seed(seed)
    tp, kc, block, V, R = 4, 4, 2, 64, 40
    logits = torch.randint(0, 5, (R, V), generator=g).to(torch.bfloat16)   # many exact ties
    top_k = torch.randint(1, kc + 1, (R, 1), generator=g).to(torch.int32)
    temps = torch.full((R, 1), 0.7, dtype=torch.bfloat16)
    vs = V // tp
    parts = [VP.local_candidates_ext(logits[:, r * vs:(r + 1) * vs], r * vs, kc, block) for r in range(tp)]
    cv = torch.cat([a for a, _, _ in parts], -1)
    ci = torch.cat([b for _, b, _ in parts], -1)
    cs = torch.cat([c for _, _, c in parts], -1)
    rng, tie = VP._need_full(VP._scaled_target(cv, temps), VP._scaled_target(cs, temps),
                             torch.zeros(R, dtype=torch.bool), top_k, torch.ones(R, dtype=torch.bool), kc)
    truth = _truth_incomplete(logits, ci, top_k, temps)
    assert not rng.any()
    assert torch.equal(tie, truth)
    assert truth.any() and (~truth).any()      # both kinds occur, so the check is not vacuous


# ---- 3. range rows (top_k unset or > KC) are exact through the fallback; stats count them ----
def _mixed(seed, B=8, V=10240, unset=(1,), big=(4,), top_p=0.9):
    p = problem(seed, B=B, V=V)
    for r in unset:
        p["top_k"][r] = 0
    for r in big:
        p["top_k"][r] = 50
    p["top_p"][:] = top_p
    return p


@pytest.mark.parametrize("seed", range(3))
def test_range_rows_exact(seed):
    p = _mixed(seed)
    flagged = 2 * (p["k"] + 1)                       # two requests, bonus + 3 target rows each
    out, stats = vpfb(p, F=8)
    assert stats.tolist() == [flagged, flagged, 0]
    assert torch.equal(out, stock_full(p))


def test_range_rows_without_fallback_are_counted_and_differ():
    """Positive control: unset top_k on flat rows samples from the whole vocabulary, so the candidates-only answer is
    wrong without the fallback (and right with it)."""
    p = _mixed(11, unset=(0, 1, 2, 3), big=(4, 5), top_p=1.0)
    g = torch.Generator().manual_seed(1)
    for req in (0, 1, 2, 3):
        for r in rows_of(p, req):
            p["logits"][r] = (torch.randn(p["logits"].shape[1], generator=g) * 0.3).to(torch.bfloat16)
    n = 6 * (p["k"] + 1)
    out, stats = vpfb(p, F=0, rates=None, synthetic=False)
    assert stats.tolist() == [n, 0, n]
    ref = stock_full(p, rates=None, synthetic=False)
    assert not torch.equal(out[:4], ref[:4])
    out2, stats2 = vpfb(p, F=n, rates=None, synthetic=False)
    assert stats2.tolist() == [n, n, 0] and torch.equal(out2, ref)


def test_overflow_counts_and_covers_first_rows():
    p = _mixed(5, unset=(0, 1, 2), big=())
    out, stats = vpfb(p, F=5)
    assert stats.tolist() == [12, 5, 7]
    # The first 5 flagged rows in priority order are bonus rows of requests 0, 1, 2 then targets of request 0.
    ref = stock_full(p)
    assert torch.equal(out[3:], ref[3:])             # unflagged requests are exact


def test_unflagged_chunk_equals_vp_verify():
    """No flagged row: vp_verify_fb (any F) is bit for bit MM's vp_verify."""
    p = problem(4)
    a, sa = vpfb(p, F=0)
    b, sb = vpfb(p, F=4)
    cv, ci = base.shards(p, 8, 32, 40)
    ref = VP.vp_verify(cv, ci, p["bonus_idx"], p["target_idx"], p["temps"], p["top_k"], p["top_p"], p["bonus_u"],
                       p["accept_u"], p["recover_u"], p["draft"], p["lengths"], p["seg"], p["grp"], GOLDEN, p["k"],
                       True)
    assert torch.equal(a, ref) and torch.equal(b, ref)
    assert sa.tolist() == [0, 0, 0] and sb.tolist() == [0, 0, 0]


def test_real_acceptance_mode():
    p = _mixed(7)
    out, stats = vpfb(p, F=8, rates=None, synthetic=False)
    assert stats[0] == 8 and stats[2] == 0
    assert torch.equal(out, stock_full(p, rates=None, synthetic=False))


# ---- 4. greedy, padded and undeclared rows are never flagged ----
def test_greedy_and_padding_not_flagged():
    p = problem(8)
    p["top_k"][:] = 0                                 # every request unset ...
    p["temps"][:4] = 0.0                              # ... requests 0-3 greedy
    out, stats = vpfb(p, F=0, num_active=6)          # requests 6, 7 are padding
    assert stats.tolist() == [2 * (p["k"] + 1), 0, 2 * (p["k"] + 1)]


def test_undeclared_slots_not_flagged():
    p = problem(9, invalid_every=2)                  # requests 0, 2, 4, 6 declare fewer drafts
    p["top_k"][:] = 0
    declared = int(p["lengths"].sum())
    out, stats = vpfb(p, F=0)
    assert stats[0] == p["bonus_idx"].shape[0] + declared


def test_tiny_temperature_bonus_is_greedy_target_is_not():
    p = problem(10)
    p["top_k"][2] = 0
    p["temps"][2] = 1e-6                              # bonus: <= SAMPLING_EPS (greedy); target: != 0 (sampled)
    out, stats = vpfb(p, F=4)
    assert stats.tolist() == [p["k"], p["k"], 0]
    assert torch.equal(out, stock_full(p))


# ---- 5. tie rows: candidates miss kept tokens, the fallback fixes them ----
def _tie_problem(seed, B=4, V=10240, tp=8, ties=120):
    p = problem(seed, B=B, V=V)
    vs = V // tp
    # Request 1 (all its rows): `ties` exact ties at a high value inside shard 3, above everything else but the top 3.
    for r in rows_of(p, 1):
        row = p["logits"][r].float()
        hi = row.max()
        row[3 * vs:3 * vs + ties] = hi + 1.0
        row[5 * vs:5 * vs + 3] = hi + 4.0
        p["logits"][r] = row.to(torch.bfloat16)
    p["top_k"][:] = 20
    p["top_p"][:] = 1.0
    return p


def test_tie_rows_fixed_by_fallback():
    p = _tie_problem(12)
    n = p["k"] + 1
    out0, s0 = vpfb(p, F=0)
    out1, s1 = vpfb(p, F=4)
    ref = stock_full(p)
    assert s0.tolist() == [n, 0, n] and s1.tolist() == [n, n, 0]
    assert torch.equal(out1, ref)
    assert not torch.equal(out0, ref)                 # the gap is real on these rows (positive control)


ALL_ACCEPT = torch.ones(3)                            # synthetic rates 1, 1, 1: every draft accepted, bonus emitted


def _flatten(p, reqs, seed=1, sd=0.3):
    g = torch.Generator().manual_seed(seed)
    for req in reqs:
        for r in rows_of(p, req):
            p["logits"][r] = (torch.randn(p["logits"].shape[1], generator=g) * sd).to(torch.bfloat16)


def test_range_rows_take_slots_before_tie_rows():
    p = _tie_problem(13)
    p["top_k"][2] = 0                                 # request 2: a range request on flat rows (its bonus is emitted)
    _flatten(p, (2,))
    out, stats = vpfb(p, F=4, rates=ALL_ACCEPT)
    n = p["k"] + 1
    assert stats.tolist() == [2 * n, 4, n]
    ref = stock_full(p, rates=ALL_ACCEPT)
    assert torch.equal(out[2], ref[2])                # the range request is exact; the tie request is uncovered
    out0, _ = vpfb(p, F=0, rates=ALL_ACCEPT)
    assert not torch.equal(out0[2], ref[2])           # and it needed the slot


# ---- 5b. each fallback output is load-bearing: bonus token, accept decision, completeness boundary ----
def test_fallback_bonus_tokens_exact():
    """Flat rows, unset top_k (requests 0-3 with top_p 0.5, 4-5 with 1.0), every draft accepted: each emitted bonus
    token must be the full-vocabulary sample."""
    p = problem(17)
    p["top_k"][:6] = 0
    p["top_p"][:4] = 0.5
    p["top_p"][4:] = 1.0
    _flatten(p, range(6))
    n = 6 * (p["k"] + 1)
    ref = stock_full(p, rates=ALL_ACCEPT)
    out, stats = vpfb(p, F=n, rates=ALL_ACCEPT)
    assert stats.tolist() == [n, n, 0] and torch.equal(out, ref)
    out0, _ = vpfb(p, F=0, rates=ALL_ACCEPT)
    assert int((out0[:6, -1] != ref[:6, -1]).sum()) >= 3     # candidates-only bonus tokens are wrong here


def test_fallback_accept_decisions_exact():
    """One dominant token (+8) over a flat 10,240 background: p(draft) is about 0.2 on the full vocabulary and about
    0.8 on the 256 candidates, so the accept test must use the full-vocabulary probability."""
    p = problem(18)
    B, k = p["bonus_idx"].shape[0], p["k"]
    _flatten(p, range(B))
    hot = torch.randint(0, p["logits"].shape[1], (B,), generator=torch.Generator().manual_seed(2))
    for req in range(B):
        for r in rows_of(p, req):
            p["logits"][r, hot[req]] = 8.0
    p["draft"] = hot.repeat_interleave(k).to(torch.int32)
    p["top_k"][:] = 0
    p["top_p"][:] = 1.0
    ref = stock_full(p, rates=None, synthetic=False)
    out, stats = vpfb(p, F=B * (k + 1), rates=None, synthetic=False)
    assert stats[2] == 0 and torch.equal(out, ref)
    out0, _ = vpfb(p, F=0, rates=None, synthetic=False)
    acc = lambda o: (o[:, :k] == p["draft"].view(B, k)).sum()  # noqa: E731
    assert acc(out0) > acc(ref)                      # candidates-only overestimates p(draft)


def test_completeness_boundary_count_equals_k():
    """Hand-built row (TP2, KC 4, block 1): exactly k candidates exceed every sentinel -> complete; k + 1 -> not."""
    s0 = [9, 8, 7, 1, 0, 0, 0, 0]
    s1 = [6, 5, 0, 0, 0, 0, 0, 0]
    row = torch.tensor([s0 + s1], dtype=torch.bfloat16)
    parts = [VP.local_candidates_ext(row[:, r * 8:(r + 1) * 8], r * 8, 4, 1) for r in range(2)]
    cv = torch.cat([a for a, _, _ in parts], -1)
    ci = torch.cat([b for _, b, _ in parts], -1)
    cs = torch.cat([c for _, _, c in parts], -1)
    temps = torch.full((1, 1), 1.0, dtype=torch.bfloat16)
    one, no = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    for k, want in ((6, False), (7, True), (2, False)):
        top_k = torch.tensor([[k]], dtype=torch.int32)
        _, tie = VP._need_full(VP._scaled_target(cv, temps), VP._scaled_target(cs, temps), no, top_k, one, 8)
        assert bool(tie[0]) == want, k
        assert bool(_truth_incomplete(row, ci, top_k, temps)[0]) == want, k


# ---- 6. VPB: per-candidate draws = the stock arithmetic on an equivalent full-width draw ----
def _expand_uniforms(p, cand_u, idx, tp=8, kc=32, block=32, fill_seed=99):
    """A full-width uniform tensor whose value at each candidate id is the per-candidate draw."""
    cv, ci = base.shards(p, tp, kc, block)
    rows_ci = torch.index_select(ci, 0, idx).long()
    full = torch.rand(rows_ci.shape[0], p["logits"].shape[1], generator=torch.Generator().manual_seed(fill_seed))
    return full.scatter(1, rows_ci, cand_u)


def test_vpb_equals_stock_on_equivalent_draws():
    p = problem(14)
    W = 8 * 32
    g = torch.Generator().manual_seed(3)
    bu = torch.rand(p["bonus_idx"].shape[0], W, generator=g)
    ru = torch.rand(p["target_idx"].shape[0], W, generator=g)
    out, stats = vpfb(p, F=0, cand=True, bonus_u=bu, recover_u=ru)
    bfull = _expand_uniforms(p, bu, p["bonus_idx"])
    rfull = _expand_uniforms(p, ru, p["target_idx"])
    assert stats.tolist() == [0, 0, 0]
    ref = STOCK(p["logits"], p["bonus_idx"], p["target_idx"], p["temps"], p["top_k"], p["top_p"], bfull,
                p["accept_u"], rfull, p["draft"], p["lengths"], p["seg"], p["grp"], GOLDEN, p["k"], True, 64)
    assert torch.equal(out, ref)


def test_vpb_fallback_rows_use_fb_draws():
    p = _mixed(15, unset=(1,), big=())
    W, V, n = 8 * 32, p["logits"].shape[1], p["k"] + 1
    g = torch.Generator().manual_seed(4)
    bu = torch.rand(p["bonus_idx"].shape[0], W, generator=g)
    ru = torch.rand(p["target_idx"].shape[0], W, generator=g)
    fbu = torch.rand(n, V, generator=g)
    out, stats = vpfb(p, F=n, cand=True, bonus_u=bu, recover_u=ru, fb_u=fbu)
    assert stats.tolist() == [n, n, 0]
    # Slot order: request 1's bonus row (verify row 1), then its targets (verify rows B + 3, B + 4, B + 5).
    bfull = _expand_uniforms(p, bu, p["bonus_idx"])
    rfull = _expand_uniforms(p, ru, p["target_idx"])
    bfull[1] = fbu[0]
    rfull[3:6] = fbu[1:4]
    assert torch.equal(out, stock_full(p, bonus_u=bfull, recover_u=rfull))


# ---- 7. served geometry ----
def test_served_geometry_tp8_with_range_rows():
    """V = 248,320 over TP8 (31,040-wide shards), block 160, KC 32, two range requests."""
    p = _mixed(16, B=4, V=248320, unset=(0,), big=(2,))
    out, stats = vpfb(p, F=8, block=VP.VP_BLOCK, kc=VP.VP_KC)
    assert stats.tolist() == [8, 8, 0]
    assert torch.equal(out, stock_full(p))
