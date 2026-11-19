# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v6: the H3 v6 acceptance bar (reports/prereg/q38_l4_h3_v6_bar_DRAFT.md, LOCKED by L1 14:07:44Z, sha256 e7c7dad0;
variant 2 with the four binding edits of its section 6) on ATTNWIRE's T2 world. Not collected (no test_ prefix). CPU only.
V6COMP-BUILD (helper of L4-E).

Per kind pass and rank the shadow (`BarShadow`) runs L1's oracle (today's draft pass, `test_l1h3_draft_attn_call._stock`
over every rank, on copies of the pool) three times with the batched-RPA stand-in's V transformed in float32 after its
cast and scale (``MODE``): "id" (today's attention, the reference), "abs" (|v_i|) and "ones" (v_i = 1), then the real
kind on the live pool. The stand-in's softmax weights p_i are float32, normalized (sum 1) and never BF16-quantized; the
oracle pipeline after it (the DCP LSE merge, the gate) is the same code in all three runs, so:
  * "abs" gives T_g = bf16(bf16(sum_i p_i |v_i|) * g), g = bf16(sigmoid(gate)) (the merge is linear in v);
  * "ones" gives g itself (sum_i p_i = 1);
  * the term of edit 2, S = sum_i p_i |v_i| (float32 accumulation in the stand-in), is read back as T_g / g. It carries
    the oracle pipeline's two BF16 output roundings (relative <= ~2^-8 of S; I, disclosed); the gated term T_g ~ g * S is
    the same bound on the gated element the kernel outputs, and leg B is also printed against it ("B-tight").
Legs (per test; the prereg's text):
  A  pool bytes byte-identical to today's after EVERY pass, on every rank;
  B  every live element: err <= max(2 ulp(want) + 1e-6, u * S), u = 2^-8, c = 1 (edit 1);
  C  bitwise share >= 99.9 % aggregated over all passes and ranks of the test (per-pass shares printed, descriptive);
  D  engagement (checked by the tests: loop passes only, at B = p, R_d = p under DRAFT-CAP, the per-B build line, packed
     at 64, agreement all attached, no build after the first);
  E  meta_i32 on / off byte-identical (tests that run both).
Descriptive (edit 4): the count and share of live elements passing only through the second term and their maximum
cancellation ratio S / |out|.
Test-side mutants (Q38_H3BAR_MUT, applied to the recorded kernel output of the evaluated passes before the legs):
  elt    one element of one pass, ratio <= 1.5, |want| >= 1e-3, moved by 4 ulp(want): leg B must FAIL;
  agg    0.2 % of the live elements of every pass (bitwise ones) moved by 1 ulp: leg C must FAIL;
  elthi  the live element of one pass with the largest ratio moved by 2 x max(2 ulp + 1e-6, u * S): leg B must FAIL;
  elthitight  the same element moved by 2 x max(2 ulp + 1e-6, u * T_g) (the gated form's bound);
  gatedonly   (the lead's mutant) one live element with g < 1 whose literal bound exceeds its gated bound, moved to
              want + (gated bound + literal bound) / 2: beyond the gated bound, inside the literal one, so leg B (gated,
              the locked form) must FAIL while the literal count stays 0.
Leg-B form (L4-E 2026-11-14 ~14:3xZ: L1 rules literal u S vs gated g u S before any verdict): both counts are always
printed. Addendum A1 (reports/prereg/q38_l4_h3_v6_bar_addendumA1_L1.md, sha256 dbd78f3d, L1 14:27Z) rules the GATED form
binding: leg B asserts err <= max(2 ulp + 1e-6, g u S) = max(.., u T_g) by default; Q38_H3BAR_BFORM=literal asserts u S. The tests defer
every assertion to the end (`defer=True`) so each test prints all of its legs raw (H3BAR_RAW) before failing.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

import test_l1h3_draft_attn_call as l1c

U = 2.0 ** -8  # edit 1: u = 2^-8, c = 1
# D > 1 (lead's ruling 2026-11-14 ~18:1xZ): L1's oracle routes its DCP decode attention to dcp2_harness.reference_attend
# (test_l1h3_draft_attn_call._route_any_dcp), not to the batched-RPA stand-in, so the V transform must also reach that
# path: install() patches dcp2_harness._partial_chunk (the reference's float32 chunk) with the same per-MODE transform.
# CAPTURE (a list, or None) records each "id"-mode reference chunk's inputs with its DCP rank, for the independent check.
CAPTURE = {"on": None, "rank": None}
C = 1.0
EXACT_BOUND = 0.999
MODE = {"v": "id"}


def modal_rpa(queries, keys, values, kv_cache, kv_lens, page_indices, cu_q_lens, distribution, *, sm_scale, k_scale,
              v_scale, out_dtype, kv_layout, skip_kv_update, return_lse=False, **_):
    """The dcp2 conftest's reference_rpa_lse (decode-only HEAD_ALONG_SUBLANE RPA, combined head 2h = K, 2h + 1 = V, the
    ``return_lse`` output), statement for statement, with the float32 V transformed per MODE after its cast and scale
    ("id": unchanged; "abs": |v|; "ones": 1). The softmax weights are float32 and normalized; nothing is BF16 until the
    output cast."""
    import jax
    import jax.numpy as jnp

    assert skip_kv_update
    mode = MODE["v"]
    kv_heads = keys.shape[1]
    per_kv = queries.shape[1] // kv_heads
    pages_per_seq = len(page_indices) // len(kv_lens)
    page_size = kv_cache.shape[1]
    combined = kv_cache.reshape(*kv_cache.shape[:2], -1, kv_cache.shape[-1])
    out = jnp.zeros(queries.shape, jnp.float32)
    lse_out = jnp.full(queries.shape[:2], -jnp.inf, jnp.float32)
    for seq in range(len(kv_lens)):
        pages = page_indices[seq * pages_per_seq : (seq + 1) * pages_per_seq]
        cache = combined[pages].reshape(pages_per_seq * page_size, -1, combined.shape[-1])
        key = cache[:, 0 : 2 * kv_heads : 2].astype(jnp.float32) * k_scale
        value = cache[:, 1 : 2 * kv_heads : 2].astype(jnp.float32) * v_scale
        if mode == "abs":
            value = jnp.abs(value)
        elif mode == "ones":
            value = jnp.ones_like(value)
        visible = jnp.arange(len(cache)) < kv_lens[seq]
        row = jnp.minimum(cu_q_lens[seq], len(queries) - 1)
        q = queries[row].astype(jnp.float32).reshape(kv_heads, per_kv, -1)
        score = jnp.einsum("hqd,thd->hqt", q, key) * sm_scale
        score = jnp.where(visible[None, None], score, -jnp.inf)
        prob = jax.nn.softmax(score, axis=-1)
        prob = jnp.where(jnp.isnan(prob), 0, prob)
        result = jnp.einsum("hqt,thd->hqd", prob, value).reshape(-1, q.shape[-1])
        lse = jax.nn.logsumexp(score, axis=-1).reshape(-1)
        live = cu_q_lens[seq + 1] > cu_q_lens[seq]
        out = out.at[row].set(jnp.where(live, result, out[row]))
        lse_out = lse_out.at[row].set(jnp.where(live, lse, lse_out[row]))
    if return_lse:
        return out.astype(out_dtype), kv_cache, lse_out
    return out.astype(out_dtype), kv_cache


def install(monkeypatch):
    """The modal stand-in in place of the conftest's (call after the rpa_stand_in fixture), and the modal DCP reference
    chunk (unless Q38_H3BAR_NO_DCP_MODAL=1: the old routing, for the oracle-fix mutant)."""
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper

    import dcp2_harness

    MODE["v"] = "id"
    monkeypatch.setattr(wrapper, "ragged_paged_attention", modal_rpa)
    if os.environ.get("Q38_H3BAR_NO_DCP_MODAL", "") != "1":
        orig_chunk = dcp2_harness.__dict__.get("_q38_orig_partial_chunk", dcp2_harness._partial_chunk)
        orig_attend = dcp2_harness.__dict__.get("_q38_orig_reference_attend", dcp2_harness.reference_attend)
        monkeypatch.setattr(dcp2_harness, "_q38_orig_partial_chunk", orig_chunk, raising=False)
        monkeypatch.setattr(dcp2_harness, "_q38_orig_reference_attend", orig_attend, raising=False)

        def chunk(q, rows, live, cache, k_scale, v_scale, *, kv_heads, kv_head_offset):
            mode = MODE["v"]
            if mode == "id":
                if CAPTURE["on"] is not None:
                    CAPTURE["on"].append(dict(rank=CAPTURE["rank"], q=np.asarray(q, np.float32), rows=np.asarray(rows),
                                              live=np.asarray(live), cache=np.asarray(cache).astype(np.float32),
                                              k_scale=float(k_scale), v_scale=float(v_scale), kv_heads=kv_heads,
                                              kv_head_offset=kv_head_offset))
                return orig_chunk(q, rows, live, cache, k_scale, v_scale, kv_heads=kv_heads,
                                  kv_head_offset=kv_head_offset)
            return _partial_chunk_modal(q, rows, live, cache, k_scale, v_scale, kv_heads=kv_heads,
                                        kv_head_offset=kv_head_offset, mode=mode)

        def attend(*a, dcp=None, **k):
            CAPTURE["rank"] = None if dcp is None else int(np.asarray(dcp[0]).reshape(()))
            return orig_attend(*a, dcp=dcp, **k)

        monkeypatch.setattr(dcp2_harness, "_partial_chunk", chunk)
        monkeypatch.setattr(dcp2_harness, "reference_attend", attend)


def _modal_chunk_impl(q, rows, live, cache, k_scale, v_scale, *, kv_heads, kv_head_offset, mode):
    """dcp2_harness._partial_chunk statement for statement, with the float32 V transformed per `mode` after its cast
    and scale ("abs": |v|, "ones": 1), before its live mask."""
    import jax
    import jax.numpy as jnp

    combined = cache.reshape(-1, cache.shape[2] * cache.shape[3], cache.shape[-1])
    sel = combined[rows]
    lo = 2 * kv_head_offset
    key = sel[:, :, lo : lo + 2 * kv_heads : 2].astype(jnp.float32) * k_scale
    value = sel[:, :, lo + 1 : lo + 2 * kv_heads : 2].astype(jnp.float32) * v_scale
    if mode == "abs":
        value = jnp.abs(value)
    elif mode == "ones":
        value = jnp.ones_like(value)
    key = jnp.where(live[..., None, None], key, 0)
    value = jnp.where(live[..., None, None], value, 0)
    n, heads, dim = q.shape
    per = heads // kv_heads
    qq = q.astype(jnp.float32).reshape(n, kv_heads, per, dim)
    qq = jnp.pad(qq, ((0, 0), (0, 0), (0, 0), (0, key.shape[-1] - dim)))
    score = jnp.einsum("nhqd,nshd->nhqs", qq, key, precision=jax.lax.Precision.HIGHEST) * (dim**-0.5)
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


def _partial_chunk_modal(q, rows, live, cache, k_scale, v_scale, *, kv_heads, kv_head_offset, mode):
    import jax

    fn = _JIT.get("chunk")
    if fn is None:
        fn = _JIT["chunk"] = jax.jit(_modal_chunk_impl, static_argnames=("kv_heads", "kv_head_offset", "mode"))
    return fn(q, rows, live, cache, k_scale, v_scale, kv_heads=kv_heads, kv_head_offset=kv_head_offset, mode=mode)


_JIT: dict = {}


def independent_dcp_term(caps, nh, dim):
    """The check of the fixed oracle: from the reference's own inputs (CAPTURE: one id-mode chunk per DCP group member,
    q over the group's heads), float32 numpy p_i (normalized softmax over the member's owned keys) and v_i, the member's
    sum_i p_i |v_i| and its LSE, merged over the members with LSE weights (dcp_combine's rule) for the heads of group
    rank 0: S [n, nh * dim]."""
    parts = []
    for c in caps:
        cache, rows, live = c["cache"], c["rows"], c["live"].astype(bool)
        combined = cache.reshape(-1, cache.shape[2] * cache.shape[3], cache.shape[-1])
        sel = combined[rows]
        lo, kvh = 2 * c["kv_head_offset"], c["kv_heads"]
        key = sel[:, :, lo:lo + 2 * kvh:2] * np.float32(c["k_scale"])
        val = sel[:, :, lo + 1:lo + 2 * kvh:2] * np.float32(c["v_scale"])
        key = np.where(live[..., None, None], key, 0).astype(np.float32)
        val = np.where(live[..., None, None], val, 0).astype(np.float32)
        q = c["q"]
        n, heads, d = q.shape
        per = heads // kvh
        qq = q.reshape(n, kvh, per, d)
        qq = np.pad(qq, ((0, 0), (0, 0), (0, 0), (0, key.shape[-1] - d)))
        score = np.einsum("nhqd,nshd->nhqs", qq, key).astype(np.float32) * np.float32(d ** -0.5)
        score = np.where(live[:, None, None, :], score, -np.inf)
        top = score.max(-1, keepdims=True)
        safe = np.where(np.isfinite(top), top, 0)
        ex = np.where(live[:, None, None, :], np.exp(score - safe), 0).astype(np.float32)
        mass = ex.sum(-1, keepdims=True)
        p = ex / np.where(mass > 0, mass, 1)
        s_k = np.einsum("nhqs,nshd->nhqd", p, np.abs(val))[..., :d].reshape(n, heads, d)
        lse = np.where(mass > 0, safe + np.log(np.where(mass > 0, mass, 1)), -np.inf).reshape(n, heads)
        parts.append((s_k[:, :nh], lse[:, :nh]))
    lses = np.stack([l for _, l in parts])
    top = lses.max(0)
    fin = np.isfinite(top)
    w = np.where(fin[None], np.exp(lses - np.where(fin, top, 0)[None]), 0)
    tot = w.sum(0)
    S = (w[..., None] * np.stack([s for s, _ in parts])).sum(0) / np.where(tot > 0, tot, 1)[..., None]
    return S.reshape(S.shape[0], nh * dim)


class BarShadow:
    """mk_attn as the predictor sees it (ATTNWIRE's _Shadow, extended): L1's oracle in modes id / abs / ones on copies of
    the pool, then the real kind on the live pool; per call the attention, the oracle's reference, T_g and g, the pool
    comparison in place (byte-difference count + digest; whole-pool copies are never kept: ATTNWIRE's 42ae764d9)."""

    def __init__(self, real, stock, rec):
        self.real, self.stock, self.rec = real, stock, rec

    def _mode(self, mode, block, pos, md, attn, pool0):
        # every rank sets the same mode before the oracle's rank-0 run (Spmd.run's barriers order it)
        MODE["v"] = mode
        try:
            return self.stock(block, pos, md, attn, pool0.clone())
        finally:
            MODE["v"] = "id"

    def __call__(self, block, pos, md, attn):
        pool0 = attn.cache_owners[0].clone()
        s_att, s_pool = self._mode("id", block, pos, md, attn, pool0)
        t_g, _ = self._mode("abs", block, pos, md, attn, pool0)
        g, _ = self._mode("ones", block, pos, md, attn, pool0)
        del pool0
        out = self.real(block, pos, md, attn)
        live_pool = attn.cache_owners[0].view(torch.uint8)
        self.rec.append(dict(
            rows=int(block.shape[0]), pos_dtype=pos.dtype, ctx_dtype=md.context_lens.dtype,
            att=out.float().clone(), s_att=s_att.float(), t_g=t_g.float(), g=g.float(),
            pool_diff=int((live_pool != s_pool.view(torch.uint8)).sum()),
            pool_sha=hashlib.sha256(live_pool.contiguous().numpy().tobytes()).hexdigest(),
            live=int((md.query_start_loc[1:] > md.query_start_loc[:-1]).sum())))
        return out


def _mutate(recs, mut):
    """Apply a test-side mutant (see the module docstring) to the recorded kernel outputs; returns a description."""
    flat = [(r, k, c) for r, rs in enumerate(recs) for k, c in enumerate(rs)]
    if not flat:
        return "no passes"
    if mut == "agg":
        n_tot = 0
        for _r, _k, c in flat:
            live = np.arange(c["rows"]) < c["live"]
            a, w = c["att"].numpy(), c["s_att"].numpy()
            idx = np.argwhere(live[:, None] & (a == w) & (w != 0))
            n = max(1, int(np.ceil(0.002 * live.sum() * a.shape[1])))
            pick = idx[np.linspace(0, len(idx) - 1, n).astype(int)] if len(idx) else idx
            for i, j in pick:
                a[i, j] = w[i, j] + l1c._ulp(np.float32(w[i, j]))
            c["att"] = torch.from_numpy(a)
            n_tot += len(pick)
        return f"agg: {n_tot} bitwise live elements moved by 1 ulp over {len(flat)} passes"
    if mut == "gatedonly":
        best = None
        for r2, k2, c2 in flat:
            live2 = np.arange(c2["rows"]) < c2["live"]
            w2, g2, a2 = c2["s_att"].numpy(), c2["g"].numpy(), c2["att"].numpy()
            t1 = 2 * l1c._ulp(w2) + 1e-6
            bg = np.maximum(t1, U * c2["t_g"].numpy())
            bl = np.maximum(t1, U * _term(c2))
            ok = live2[:, None] & (g2 > 0) & (g2 < 1) & (bl > bg) & (np.abs(a2 - w2) <= bg)
            gap = np.where(ok, bl - bg, -1.0)
            i2, j2 = np.unravel_index(int(np.argmax(gap)), gap.shape)
            if gap[i2, j2] > 0 and (best is None or gap[i2, j2] > best[0]):
                best = (float(gap[i2, j2]), r2, k2, c2, i2, j2, float(bg[i2, j2]), float(bl[i2, j2]), float(g2[i2, j2]))
        if best is None:
            raise AssertionError("gatedonly: no live element with g < 1 and a literal bound above its gated bound")
        _gap, r, k, c, i, j, bg_, bl_, g_ = best
        a = c["att"].numpy()
        a[i, j] = c["s_att"].numpy()[i, j] + 0.5 * (bg_ + bl_)
        c["att"] = torch.from_numpy(a)
        return (f"gatedonly: rank {r} pass {k} element ({i},{j}) g {g_:.4f} moved to want + {0.5 * (bg_ + bl_):.3e} "
                f"(gated bound {bg_:.3e} < err <= literal bound {bl_:.3e})")
    r, k, c = flat[len(flat) // 2]
    live = np.arange(c["rows"]) < c["live"]
    a, w = c["att"].numpy(), c["s_att"].numpy()
    S = _term(c)
    ratio = np.where(np.abs(w) > 0, S * np.where(c["g"].numpy() > 0, c["g"].numpy(), 0) / np.maximum(np.abs(w), 1e-30), 0)
    m = live[:, None] & np.ones_like(w, bool)
    if mut == "elt":
        # L1's ruling (2026-11-14, no rule change): exactly as written -- 4 ulps on a live element whose cancellation
        # ratio is <= 1.5 (V6COMP's stated filter: |want| >= 1e-3); NO substitute: without one this test reports the
        # mutant NOT APPLICABLE (not a kill) with its minimum ratio
        cen = ratio_census(world_recs=recs)
        if cen["hit"] is None:
            raise AssertionError(f"M-elt NOT APPLICABLE: no live element at ratio <= 1.5 with |want| >= 1e-3 in this "
                                 f"test (minimum ratio {cen['min']:.4f} at rank {cen['at'][0]} pass {cen['at'][1]})")
        rv, wv, r, k, i, j = cen["hit"]
        c = recs[r][k]
        a, w = c["att"].numpy(), c["s_att"].numpy()
        a[i, j] = w[i, j] + 4 * l1c._ulp(np.float32(w[i, j]))
        c["att"] = torch.from_numpy(a)
        return (f"elt: rank {r} pass {k} element ({i},{j}) ratio {rv:.4f} |want| {wv:.4e} moved by 4 ulp "
                f"({cen['n15']} live elements of the test qualify)")
    elif mut in ("elthi", "elthitight"):
        rr = np.where(m, ratio, -1)
        i, j = np.unravel_index(int(np.argmax(rr)), rr.shape)
        second = U * float(S[i, j]) if mut == "elthi" else U * float(c["t_g"].numpy()[i, j])
        bound = max(2 * float(l1c._ulp(np.float32(w[i, j]))) + 1e-6, second)
        sign = 1.0 if a[i, j] >= w[i, j] else -1.0
        a[i, j] = a[i, j] + sign * 2 * bound
        what = f"{mut}: rank {r} pass {k} element ({i},{j}) ratio {ratio[i, j]:.3f} moved by 2 x {bound:.3e}"
    else:
        raise ValueError(mut)
    c["att"] = torch.from_numpy(a)
    return what


def ratio_census(world_recs):
    """Per test: the minimum cancellation ratio S g / |want| (= S / |ungated out|) over live elements with |want| >= 1e-3,
    where it is, how many are <= 1.5, and the first qualifying element (ratio <= 1.5) in rank / pass order."""
    best, at, n15, hit = np.inf, (None, None), 0, None
    for r, rs in enumerate(world_recs):
        for k, c in enumerate(rs):
            live = np.arange(c["rows"]) < c["live"]
            w, g = c["s_att"].numpy(), c["g"].numpy()
            rat = np.where(np.abs(w) > 0, _term(c) * np.where(g > 0, g, 0) / np.maximum(np.abs(w), 1e-30), np.inf)
            rat = np.where(live[:, None] & (np.abs(w) >= 1e-3), rat, np.inf)
            m = float(rat.min()) if rat.size else np.inf
            if m < best:
                best, at = m, (r, k)
            q = np.argwhere(rat <= 1.5)
            n15 += len(q)
            if hit is None and len(q):
                i, j = q[0]
                hit = (float(rat[i, j]), float(abs(w[i, j])), r, k, int(i), int(j))
    return dict(min=best, at=at, n15=n15, hit=hit)


def _term(c):
    """S = sum_i p_i |v_i| per element (T_g / g), 0 where the element has no key (g == 0)."""
    t, g = c["t_g"].numpy(), c["g"].numpy()
    return np.where(g > 0, t / np.where(g > 0, g, 1), 0).astype(np.float32)


def evaluate(world, tag, mutant=None, defer=False):
    """Legs A, B, C on every recorded pass of every rank (the caller checks D and E). Returns the stats dict; raises
    AssertionError with 'LEG A' / 'LEG B' / 'LEG C' in the message on a failed leg. `mutant` defaults to Q38_H3BAR_MUT."""
    mut = mutant if mutant is not None else os.environ.get("Q38_H3BAR_MUT", "")
    cen = ratio_census(world.recs)
    print(f"H3BAR_MINRATIO {tag}: minimum cancellation ratio among live elements with |want| >= 1e-3 {cen['min']:.4f} "
          f"(rank {cen['at'][0]} pass {cen['at'][1]}); elements at ratio <= 1.5: {cen['n15']}", flush=True)
    note = _mutate(world.recs, mut) if mut else ""
    if note:
        print(f"H3BAR_MUTANT {tag}: {note}", flush=True)
    n_live = n_exact = n_bad = n_bad_tight = n_second = n_pass = n_pool = 0
    max_ratio2 = 0.0
    shares = []
    worst_b = (0.0, None)
    for r, recs in enumerate(world.recs):
        for k, c in enumerate(recs):
            live = np.arange(c["rows"]) < c["live"]
            a, w = c["att"].numpy()[live], c["s_att"].numpy()[live]
            S = _term(c)[live]
            Tg = c["t_g"].numpy()[live]
            assert np.isfinite(c["att"].numpy()).all(), f"{tag}: non-finite kernel output rank {r} pass {k}"
            assert (c["att"].numpy()[~live] == 0).all(), f"{tag}: idle rows not zero rank {r} pass {k}"
            err = np.abs(a - w)
            t1 = 2 * l1c._ulp(w) + 1e-6
            bound = np.maximum(t1, U * C * S)
            bad = err > bound
            n_bad += int(bad.sum())
            n_bad_tight += int((err > np.maximum(t1, U * C * Tg)).sum())
            second = (err > t1) & ~bad
            n_second += int(second.sum())
            if second.any():
                ratio = S / np.maximum(np.abs(w) / np.where(c["g"].numpy()[live] > 0, c["g"].numpy()[live], 1), 1e-30)
                max_ratio2 = max(max_ratio2, float(ratio[second].max()))
            if bad.any():
                x = float((err / bound)[bad].max())
                if x > worst_b[0]:
                    worst_b = (x, (r, k))
            ex = int((err == 0).sum())
            n_exact += ex
            n_live += err.size
            shares.append(ex / max(1, err.size))
            n_pool += int(c["pool_diff"] != 0)
            n_pass += 1
    share = n_exact / max(1, n_live)
    st = dict(passes=n_pass, live=n_live, exact=n_exact, share=share, min_pass_share=min(shares) if shares else -1.0,
              bad=n_bad, bad_tight=n_bad_tight, second=n_second, second_share=n_second / max(1, n_live),
              max_ratio_second=max_ratio2, pool_fail=n_pool, mutant=note)
    print(f"H3BAR {tag}: passes {n_pass} live elements {n_live}; A pools differing in {n_pool} passes; B bad {n_bad} "
          f"(B-tight, gated term: {n_bad_tight}); C share {share:.6f} (per-pass min {st['min_pass_share']:.6f}); "
          f"second-term-only {n_second} ({st['second_share']:.2e} of live), max ratio among them {max_ratio2:.2f}"
          + (f"; MUTANT {note}" if note else ""), flush=True)
    print(f"H3BAR {tag} per-pass shares " + " ".join(f"{s:.5f}" for s in shares), flush=True)
    # addendum A1 (L1 14:27Z, sha256 dbd78f3d): leg B gates on the GATED form, max(2 ulp + 1e-6, g u S) = u T_g here;
    # the literal u S count is descriptive (Q38_H3BAR_BFORM=literal asserts it instead, for the mutant record only)
    form = os.environ.get("Q38_H3BAR_BFORM", "tight")
    fails = []
    if n_pass == 0:
        fails.append(f"{tag}: no kind pass recorded")
    if n_pool:
        fails.append(f"LEG A FAIL {tag}: pool bytes differ from today's in {n_pool} passes")
    nb = n_bad_tight if form == "tight" else n_bad
    if nb:
        fails.append(f"LEG B FAIL (form {form}) {tag}: {nb} live elements beyond the bound (literal {n_bad}, gated "
                     f"{n_bad_tight}; literal worst x{worst_b[0]:.2f} at {worst_b[1]})")
    if share < EXACT_BOUND:
        fails.append(f"LEG C FAIL {tag}: aggregate bitwise share {share:.6f} < {EXACT_BOUND}")
    st["fails"], st["form"] = fails, form
    if fails and not defer:
        raise AssertionError(" ; ".join(fails))
    return st
