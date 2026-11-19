# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v6: the FRESH H3 composed run under the locked v6 bar (reports/prereg/q38_l4_h3_v6_bar_DRAFT.md, sha256
e7c7dad0, variant 2 + section 6's edits; the legs and their evaluation: h3bar_harness.py). CPU, Pallas interpret.
V6COMP-BUILD (helper of L4-E). Imports ATTNWIRE's and V6COMP's harnesses; edits no L1 / ATTNWIRE test file.

CP     ATTNWIRE's DH-D leg (test_attnwire_compose.World, its seeds 7000 + 100 D + p and live counts) on the composed tree,
       p 8 / 16 / 32 / 48 / 64 x (R, D) (2, 1) / (2, 2) / (4, 4), 8 steps, meta_i32 on and off: legs A, B, C per world;
       D: mk_attn on loop passes only at B = p (K - 1 per DH-D call, never step 0), the per-B build line only, one
       build per rank (none after the first step), packed at 64, agreement all attached; E: meta on / off drafts,
       source, every pass's attention and pool digest byte-identical. Descriptive: draft tokens vs a stock world.
PATHS  ATTNWIRE's window-switched paths (stage A / DRAFT-ASM loop / plain loop / DH-D at p = 16 after the DH-D warm-up)
       under the bar: legs A-C, D with 0 builds after the warm-up.
HC     DRAFT-CAP's R_d program (V6World.install_cap + draft_cap's program class) at (2, 2) p 16 / 48 and (4, 4) p 16,
       8 steps, meta on and off: legs A-E, every pass at B = p over R_d = p request slots.
HK     T1-K1FUSE's program (capacity and DRAFT-CAP R_d) at (2, 2) p 16 / 48, 8 steps (K1 traces under meta_i32
       always): legs A-D, every pass at B = p over its slot count.
Run (CDK CPU pod): JAX_PLATFORMS=cpu VLLM_TARGET_DEVICE=cpu python -m pytest -q -s -p q38_cpu_device
  tests/kernels/qwen4_exp/dcp2/test_v6c_h3bar.py
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import h3bar_harness as hb  # noqa: E402
import test_attnwire_compose as cw  # noqa: E402
import test_v6c_h3_compose as v6  # noqa: E402

K, CAP = cw.K, cw.CAP
STEPS = 8
VERDICTS: list = []


def _bar(monkeypatch, caplog):
    cw_setup_level = logging.INFO
    caplog.set_level(cw_setup_level)
    hb.install(monkeypatch)
    monkeypatch.setattr(cw, "_Shadow", hb.BarShadow)


def _leg_d(world, caplog, p, R, tag, slots=None):
    msgs = [x.getMessage() for x in caplog.records]
    builds = [m for m in msgs if "MTP draft attention kernel built at" in m]
    assert builds and all(m == cw.BUILD_LINE % p for m in builds), f"LEG D FAIL {tag}: build lines {sorted(set(builds))}"
    assert sorted(set(world.builds)) == [p] and len(world.builds) == R, f"LEG D FAIL {tag}: builds {world.builds}"
    agree = [m for m in msgs if "draft attention agreement" in m]
    assert agree and all(f"attached {R}, refused 0 of {R}" in m for m in agree), f"LEG D FAIL {tag}: agreement {agree}"
    for r, recs in enumerate(world.recs):
        assert recs and all(c["rows"] == p for c in recs), f"LEG D FAIL {tag}: rank {r} pass rows"
    for r in range(R):
        assert world.ranks[r].pred.mk_attn.inner.real.pack_rows == max(cw.DRIVER_BUCKETS), f"LEG D FAIL {tag}: pack"
    if slots is not None:
        want = (p, slots, slots * world.cols)
        assert all(s == want for rs in world.shapes for s in rs), f"LEG D FAIL {tag}: shapes {world.shapes[0][:2]} != {want}"
    return f"D ok (B = {p} on {sum(len(x) for x in world.recs)} passes, build line x{len(builds)}, {len(world.builds)} builds)"


def _leg_e(wa, wb, oa, ob, tag):
    for r in range(len(oa)):
        for a, b in zip(oa[r], ob[r]):
            for x, y in zip(a, b):
                assert torch.equal(x, y), f"LEG E FAIL {tag}: rank {r} drafts / source differ meta on vs off"
    for r, (ra, rb) in enumerate(zip(wa.recs, wb.recs)):
        assert len(ra) == len(rb), f"LEG E FAIL {tag}: pass counts"
        for k, (ca, cb) in enumerate(zip(ra, rb)):
            assert torch.equal(ca["att"], cb["att"]) and ca["pool_sha"] == cb["pool_sha"], f"LEG E FAIL {tag}: pass {k}"
            assert ca["ctx_dtype"] == torch.int32 and cb["ctx_dtype"] == torch.int64, f"LEG E FAIL {tag}: dtypes"
    return "E ok (meta on / off byte-identical)"


def _drafts_vs_stock(monkeypatch, R, D, seed, p, n, outs):
    world = cw.World(monkeypatch, R, D, seed=seed, mk=False)

    def body(r):
        world.setup(r)
        return [world.dhd(r, world.step_inputs(i, p, n), p, meta=True) for i in range(STEPS)]

    ref = world.spmd.threads(body)[0]
    agree = total = 0
    for a, b in zip(outs[0], ref):
        for x, y in zip(a[:K], b[:K]):
            agree += int((x[:n] == y[:n]).sum())
            total += n
    return agree, total


def _try(fn, *a, **k):
    try:
        return fn(*a, **k), None
    except AssertionError as exc:
        return None, str(exc)


def _verdict(tag, st, d, e, fails=()):
    """One raw line per test (no verdict word: L4-E holds the H3 verdict for L1's leg-B ruling), then the deferred
    assertion."""
    line = (f"H3BAR_RAW {tag}: A pools differing in {st['pool_fail']} of {st['passes']} passes; B literal (u S) beyond "
            f"{st['bad']}; B gated (u T_g) beyond {st['bad_tight']}; C share {st['share']:.6f} (per-pass min "
            f"{st['min_pass_share']:.6f}); D {d}; E {e}; second-term-only {st['second']} ({st['second_share']:.2e} of "
            f"{st['live']} live), max ratio {st['max_ratio_second']:.2f}")
    VERDICTS.append(line)
    print(line, flush=True)
    fails = list(st.get("fails", [])) + [f for f in fails if f]
    if fails:
        raise AssertionError(" ; ".join(fails))


@pytest.mark.parametrize("p", [8, 16, 32, 48, 64])
@pytest.mark.parametrize("R,D", [(2, 1), (2, 2), (4, 4)])
def test_cp_bar(rpa_stand_in, monkeypatch, caplog, R, D, p):
    cw._setup_dcp(monkeypatch, D)
    _bar(monkeypatch, caplog)
    n = cw._live_for(p)
    seed = 7000 + 100 * D + p
    res = {}
    for meta in (True, False):
        caplog.clear()
        world = v6.V6World(monkeypatch, R, D, seed=seed)

        def body(r, world=world, meta=meta):
            world.setup(r)
            return [world.dhd(r, world.step_inputs(i, p, n), p, meta=meta) for i in range(STEPS)]

        outs = world.spmd.threads(body)
        for r in range(1, R):
            for a, b in zip(outs[0], outs[r]):
                for x, y in zip(a, b):
                    assert torch.equal(x, y), "ranks drafted differently"
        tag = f"CP p={p} R={R} D={D} meta={int(meta)}"
        st = hb.evaluate(world, tag, defer=True)
        d, dfail = _try(_leg_d, world, caplog, p, R, tag, slots=CAP)
        res[meta] = (outs, world, st, d or dfail, dfail)
    e, efail = _try(_leg_e, res[True][1], res[False][1], res[True][0], res[False][0], f"CP p={p} R={R} D={D}")
    agree, total = _drafts_vs_stock(monkeypatch, R, D, seed, p, n, res[True][0])
    print(f"H3BAR_DRAFTS CP p={p} R={R} D={D} (descriptive): kind drafts equal stock on {agree} / {total} draft tokens "
          f"({agree / max(1, total):.4f})", flush=True)
    errs = []
    for meta in (True, False):
        errs += _try(_verdict, f"CP p={p} R={R} D={D} meta={int(meta)}", res[meta][2], res[meta][3], e or efail,
                     (res[meta][4], efail))[1:]
    errs = [x for x in errs if x]
    assert not errs, " || ".join(errs)


def test_paths_bar(rpa_stand_in, monkeypatch, caplog):
    from vllm_torchtpu.spec_decode import draft_asm, draft_one
    from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop

    R, D, p = 2, 1, 16
    cw._setup_dcp(monkeypatch, D)
    _bar(monkeypatch, caplog)
    world = v6.V6World(monkeypatch, R, D, seed=8800)
    n = 11

    def body(r):
        world.setup(r)
        world.dhd(r, world.step_inputs(0, p, n), p, meta=True)  # warm-up: builds B = p once per rank
        after_warm = len(world.builds)
        counts = {}
        for i, path in enumerate(("stageA", "draftasm", "plain", "dhd")):
            si = world.step_inputs(1 + i, p, n)
            n0 = len(world.recs[r])
            if path == "dhd":
                world.dhd(r, si, p, meta=True)
            else:
                bufs = world.install(r, si)
                for b, v in zip(bufs, si["md1"]):
                    b.copy_(v)
                prog, rk = world.prog(r, p), world.ranks[r]
                h0 = torch.from_numpy(np.random.default_rng(i).normal(0, 0.5, (CAP, world.g.H)).astype(np.float32)).to(
                    torch.bfloat16)
                tok0 = torch.from_numpy(np.random.default_rng(i + 9).integers(1, cw.V, p).astype(np.int32))
                if path == "stageA":
                    draft_one.carry_loop_body(prog, si["pos1"], h0, h0, bufs[0], torch.zeros(CAP, dtype=torch.int32),
                                              torch.tensor(n), CAP)
                elif path == "draftasm":
                    draft_asm._MtpFusedLoopAsm.forward(prog, tok0, si["pos1"], h0[:p], bufs[0],
                                                       torch.zeros(CAP, dtype=torch.int32), torch.tensor(n))
                else:
                    _MtpFusedLoop.forward(prog, tok0, si["pos1"], h0[:p], bufs[0])
            counts[path] = [c["rows"] for c in world.recs[r][n0:]]
        return after_warm, counts

    outs = world.spmd.threads(body)
    st = hb.evaluate(world, "PATHS p=16 R=2 D=1", defer=True)
    dfail = None
    for after_warm, counts in outs:
        for path, rows in counts.items():
            if rows != [p] * (K - 1):
                dfail = f"LEG D FAIL PATHS: {path} rows {rows}"
    if len(world.builds) != R or set(world.builds) != {p}:
        dfail = f"LEG D FAIL PATHS: builds after warm-up {world.builds}"
    _verdict("PATHS p=16 R=2 D=1", st, dfail or f"ok (stage A / DRAFT-ASM / plain / DH-D at B = {p}; 0 builds after the "
             f"warm-up)", "n/a", (dfail,))


def _dhd_cap(world, r, si, p, meta):
    from vllm_torchtpu.spec_decode import draft_one

    rk = world.ranks[r]
    bufs, flat, tbuf = world.install_cap(r, si, p)
    n0 = len(world.recs[r])
    args = (si["input_ids"], si["positions"], si["hidden"], si["draft_rows"][:p], si["pos1"], *si["md0"], *si["md1"],
            *bufs, torch.zeros(CAP, dtype=torch.int32), torch.tensor(si["n"]), flat, None, None, None, tbuf, None, None,
            None)
    with draft_one._meta_i32(rk.pred) if meta else cw._null():
        out = tuple(world.cap_prog(r, p)(*args))
    calls = world.recs[r][n0:]
    assert len(calls) == K - 1 and all(c["rows"] == p for c in calls), f"LEG D FAIL: {[c['rows'] for c in calls]}"
    return out


@pytest.mark.parametrize("R,D,p", [(2, 2, 16), (2, 2, 48), (4, 4, 16)])
def test_hc_bar(rpa_stand_in, monkeypatch, caplog, R, D, p):
    cw._setup_dcp(monkeypatch, D)
    _bar(monkeypatch, caplog)
    n = cw._live_for(p)
    res = {}
    for meta in (True, False):
        caplog.clear()
        world = v6.V6World(monkeypatch, R, D, seed=9100 + 100 * D + p)

        def body(r, world=world, meta=meta):
            world.setup(r)
            return [_dhd_cap(world, r, world.step_inputs(i, p, n), p, meta) for i in range(STEPS)]

        outs = world.spmd.threads(body)
        tag = f"HC p={p} R={R} D={D} meta={int(meta)}"
        st = hb.evaluate(world, tag, defer=True)
        d, dfail = _try(_leg_d, world, caplog, p, R, tag, slots=min(p, CAP))
        res[meta] = (outs, world, st, d or dfail, dfail)
    e, efail = _try(_leg_e, res[True][1], res[False][1], res[True][0], res[False][0], f"HC p={p} R={R} D={D}")
    errs = []
    for meta in (True, False):
        errs += _try(_verdict, f"HC p={p} R={R} D={D} meta={int(meta)}", res[meta][2], res[meta][3], e or efail,
                     (res[meta][4], efail))[1:]
    errs = [x for x in errs if x]
    assert not errs, " || ".join(errs)


@pytest.mark.parametrize("cap", [False, True])
@pytest.mark.parametrize("p", [16, 48])
def test_hk_bar(rpa_stand_in, monkeypatch, caplog, p, cap):
    R, D = 2, 2
    cw._setup_dcp(monkeypatch, D)
    v6._gather_identity(monkeypatch)
    _bar(monkeypatch, caplog)
    n = cw._live_for(p)
    world = v6.V6World(monkeypatch, R, D, seed=9300 + p + 7 * cap, vocab=v6.VT)

    def body(r):
        world.setup(r)
        res = []
        for i in range(STEPS):
            si = world.step_inputs(i, p, n)
            si["positions"] = world.positions_for(si)
            res.append(world.k1(r, si, world.verify_inputs(si, i, p), p, cap))
        return res

    world.spmd.threads(body)
    tag = f"HK p={p} cap={int(cap)} R={R} D={D}"
    st = hb.evaluate(world, tag, defer=True)
    d, dfail = _try(_leg_d, world, caplog, p, R, tag, slots=min(p, CAP) if cap else CAP)
    _verdict(tag, st, d or dfail, "n/a (K1 traces under meta_i32 always)", (dfail,))


def test_dcp_oracle_term_matches_the_reference_sum(rpa_stand_in, monkeypatch, caplog):
    """D > 1 oracle fix (the lead's ruling, 2026-11-14): L1's DCP oracle path (_route_any_dcp ->
    dcp2_harness.reference_attend) now takes the |v| / 1 transform too. CP (2, 2) p = 8, one DH-D step: on rank 0's first
    kind pass the oracle's S (T_g / g) equals an independent float32 numpy sum_i p_i |v_i| built from the reference's own
    inputs (its q, owned rows, live mask, cache and scales: p_i the normalized softmax over each member's owned keys,
    merged over the DCP group by LSE) within A1's rounding slack (relative 2^-6 + 1e-6 absolute) on every live element.
    With the old routing (Q38_H3BAR_NO_DCP_MODAL=1: S = g = want) it fails."""
    R, D, p = 2, 2, 8
    cw._setup_dcp(monkeypatch, D)
    _bar(monkeypatch, caplog)
    hb.CAPTURE["on"] = []
    try:
        world = v6.V6World(monkeypatch, R, D, seed=7000 + 100 * D + p)
        n = cw._live_for(p)

        def body(r):
            world.setup(r)
            return world.dhd(r, world.step_inputs(0, p, n), p, meta=True)

        world.spmd.threads(body)
        caps = list(hb.CAPTURE["on"])
    finally:
        hb.CAPTURE["on"] = None
    c = world.recs[0][0]
    live = np.arange(c["rows"]) < c["live"]
    s_or = hb._term(c)[live]
    first = [x for x in caps[:R] if x["rank"] is not None]
    assert len(first) == D and [x["rank"] for x in first] == list(range(D)), \
        f"DCP ORACLE TERM MISMATCH: reference chunks of the first pass {[x['rank'] for x in caps[:R]]} (old routing?)"
    s_in = hb.independent_dcp_term(first, cw.aw.NH_RANK, cw.aw.HD)[: c["rows"]][live]
    rel = np.abs(s_or - s_in) / np.maximum(np.abs(s_in), 1e-6)
    bad = (np.abs(s_or - s_in) > 2.0 ** -6 * np.abs(s_in) + 1e-6)
    print(f"H3BAR_DCPTERM CP p={p} R={R} D={D} rank 0 pass 0: {s_in.size} live elements; oracle S vs independent float32 "
          f"sum p|v|: max relative error {float(rel.max()):.3e}, beyond 2^-6 + 1e-6: {int(bad.sum())}; S range "
          f"{float(s_in.min()):.3e}..{float(s_in.max()):.3e}", flush=True)
    assert not bad.any(), f"DCP ORACLE TERM MISMATCH: {int(bad.sum())} of {s_in.size} elements (max rel {float(rel.max()):.3e})"


def test_zz_verdicts():
    print("H3BAR_SUMMARY " + str(len(VERDICTS)) + " raw lines", flush=True)
