# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT-CAP inside DH-D (VLLM_TPU_DRAFT_ASM_CAP) through the real ``Eagle3Proposer.propose`` and the real
async spec substitution (tests/q38_draftcap/draftcap_harness.py: per-group draft metadata, a stub draft that reads
every group's block-table row per request slot).

A. Bitwise over 8 serving steps: DH-D at R_d = p == DH-D at R_d = capacity == the base path (drafts, step 1's
   positions / seq_lens / first token, source, rejected count, index dicts) at 2 / 3 / 8 / 12 / 16 / 20 / 32 / 48 / 64
   live requests, fallback rows on and off; fewer-draft requests; R_d = capacity when p > capacity.
B. Per-pass metadata: every draft pass inside the R_d program reads exactly the first R_d request slots of what
   the capacity program's pass reads (seq_lens, query_start_loc, request_distribution, every group's block table,
   step 0's draft_rows), at the R_d shapes, over steps whose live count moves inside one loop bucket.
C. Census: the program's arguments at R_d (draft_rows [R_d], buffers [R_d] / [R_d + 1], table buffers
   [R_d * cols]) where DH-D's are at the capacity; 2 compiled draft programs; 0 eager ops, 0 deferred inputs.
D. Programs: one per loop bucket (code objects of their own), compiling only its token buckets; the warm-up
   records exactly the 12 steady (token, loop) pairs.
E. Window-switched steps over DH-D (R_d) / stage A / DRAFT-ASM / base never compile after the warm-up; bitwise.
F. The verify metadata and its block tables are never written; the R_d buffers persist across steps; md0 / md1
   and the tables are read only inside the body.
G. Configuration: the flag needs DH-D; default off, compile-hashed.
H. Tables: more distinct tables than the program takes keeps DH-D off (stage A folds); two groups sharing one
   table tensor share one buffer.
I. Agreement: R_d and capacity programs have different warm-key fingerprints (ranks that differ in the flag end
   on the base path).
J. CHECK compares the R_d program beside the base path and catches a perturbed output.
K. Engagement line once per R_d in serving, never in warm-up; a build failure stops.
"""

from collections import Counter

import pytest
import torch

import draftcap_harness as h
from vllm_torchtpu.spec_decode import draft_asm, draft_cap, draft_one

ROWS = [2, 3, 8, 12, 16, 20, 32, 48, 64]
DRAFT_PROGRAMS = ("seed", "seed_asm", "seed_cl", "draft_fwd0", "carry", "carry_fresh", "carry_asm",
                  "carry_fresh_asm", "loop", "loop_asm", "carry_loop", "one")
T_BUF = slice(16 + draft_cap.TABLE_SLOTS, 16 + 2 * draft_cap.TABLE_SLOTS)
T_SRC = slice(16, 16 + draft_cap.TABLE_SLOTS)


def _draft_calls(calls):
    return sum(v for k, v in calls.items() if k in DRAFT_PROGRAMS)


def _p(R, cap=64):
    from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

    return _get_padded_token_len(h.PADDINGS, R)


def _shape(key, i):
    return None if key[i] is None else key[i][0]


# ---------------------------------------------------------------- A. bitwise over 8 steps


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("R", ROWS)
def test_cap_is_bitwise_capacity_dhd_and_base_over_8_steps(monkeypatch, R, fallback):
    seeds = [51000 + 100 * R + 10 * fallback + i for i in range(8)]
    kw = {"step_kw": {"fallback": fallback}}
    base, _ = h.serve(monkeypatch, "base", seeds, R, **kw)
    full, fp = h.serve(monkeypatch, "one", seeds, R, **kw)
    capo, cp = h.serve(monkeypatch, "cap", seeds, R, **kw)
    r_d = min(_p(R), 64)
    for i, (b, f, c) in enumerate(zip(base, full, capo)):
        assert not b.folded and f.folded and c.folded, (i, cp._draft_asm.counts(), dict(cp._draft_asm.reasons))
        assert c.calls["one"] == 1 and c.calls["draft_fwd0"] == 0 and c.calls["carry_loop"] == 0, c.calls
        kc, kf = h.one_key(c), h.one_key(f)
        assert _shape(kc, 3) == (r_d,) and _shape(kf, 3) == (64,), (kc[3], kf[3])
        h.same_out(c, f, f"R_d={r_d} vs capacity DH-D R={R} step {i}")
        h.same_out(c, b, f"R_d={r_d} vs base R={R} step {i}")
    assert cp._draft_asm.counts()["folded_one"] == 8 and cp._draft_asm.one_cap
    print(f"\nDRAFTCAP_BITWISE R={R} p={_p(R)} R_d={r_d} fallback={int(fallback)} steps=8 drafts "
          f"{tuple(capo[0].drafts.shape)} source[{capo[0].src.numel()}] rejected[{capo[0].rejected.numel()}]: "
          f"equal to DH-D at capacity 64 and to the base path on every step")


@pytest.mark.parametrize("R", [12, 32])
def test_cap_bitwise_with_fewer_drafts(monkeypatch, R):
    seeds = [52000 + 100 * R + i for i in range(8)]
    kw = {"warm_each": True, "step_kw": {"short": True}}
    base, _ = h.serve(monkeypatch, "base", seeds, R, **kw)
    capo, cp = h.serve(monkeypatch, "cap", seeds, R, **kw)
    for i, (b, c) in enumerate(zip(base, capo)):
        h.same_out(c, b, f"short R={R} step {i}")
    c = cp._draft_asm.counts()
    assert c["folded_one"] >= 1, c
    print(f"\nDRAFTCAP_BITWISE short R={R} steps=8 fewer-draft requests equal to base (folded {c['folded']}: "
          f"R_d DH-D {c['folded_one']}, stage A {c['folded_cl']})")


@pytest.mark.parametrize("R", [33, 40])
def test_rd_is_the_capacity_when_the_loop_bucket_exceeds_it(monkeypatch, R):
    """Capacity 40 (max-num-seqs 40): loop bucket 48 > 40, so R_d = 40 and step 0's rows are the capacity rows."""
    seeds = [53000 + 100 * R + i for i in range(8)]
    base, _ = h.serve(monkeypatch, "base", seeds, R, cap=40)
    full, _ = h.serve(monkeypatch, "one", seeds, R, cap=40)
    capo, cp = h.serve(monkeypatch, "cap", seeds, R, cap=40)
    for i, (b, f, c) in enumerate(zip(base, full, capo)):
        assert c.folded and c.calls["one"] == 1, (c.calls, dict(cp._draft_asm.reasons))
        k = h.one_key(c)
        assert _shape(k, 3) == (40,) and _shape(k, 11) == (40,) and _shape(k, 12) == (41,), k[3]
        h.same_out(c, f, f"cap40 R={R} step {i} vs capacity DH-D")
        h.same_out(c, b, f"cap40 R={R} step {i} vs base")
    print(f"\nDRAFTCAP_BITWISE R={R} p=48 capacity=40 R_d=40 steps=8: equal to DH-D at capacity and to base")


# ---------------------------------------------------------------- B. per-pass metadata


def _rows_of(t, slots):
    return t.reshape(slots, -1)


@pytest.mark.parametrize("rs,cap", [((12, 3, 16, 5, 9, 1, 14, 2), 64), ((20, 31, 17, 32, 25), 64),
                                    ((40, 33, 48, 35), 64), ((60, 49, 64, 51), 64), ((36, 33, 40), 40)])
def test_each_pass_reads_the_rd_prefix_of_the_capacity_pass(monkeypatch, rs, cap):
    seeds = [54000 + 37 * i + 1000 * len(rs) for i in range(len(rs))]
    full, fp = h.serve(monkeypatch, "one", seeds, rs[0], rs=list(rs), cap=cap, mtp_cls=h.RecMtpCap)
    capo, cp = h.serve(monkeypatch, "cap", seeds, rs[0], rs=list(rs), cap=cap, mtp_cls=h.RecMtpCap)
    nf, nc = fp.draft_model.seen, cp.draft_model.seen
    # each serving step's DH-D forwards are its last K records; serving steps come after every warm-up step
    nf, nc = nf[len(nf) - h.K * len(rs):], nc[len(nc) - h.K * len(rs):]
    checked = Counter()
    for j, R in enumerate(rs):
        r_d = min(_p(R), cap)
        for k in range(h.K):
            (sf, seqf, qslf, rdf, tbf, rowsf), (sc, seqc, qslc, rdc, tbc, rowsc) = nf[h.K * j + k], nc[h.K * j + k]
            tag = f"R={R} step {j} pass {k}"
            assert sf == sc == k, tag
            assert seqc.shape == (r_d,) and qslc.shape == (r_d + 1,), (tag, seqc.shape, qslc.shape)
            h.same(seqc, seqf[:r_d], f"{tag} seq_lens")
            h.same(qslc, qslf[: r_d + 1], f"{tag} query_start_loc")
            h.same(rdc, rdf, f"{tag} request_distribution")
            assert set(tbc) == set(tbf) == set(h.NAMES), (tag, sorted(tbc))
            for name in h.NAMES:
                cols = tbf[name].shape[0] // cap
                assert tbc[name].shape == (r_d * cols,), (tag, name, tbc[name].shape)
                h.same(_rows_of(tbc[name], r_d), _rows_of(tbf[name], cap)[:r_d], f"{tag} {name} block table")
            if k == 0:
                assert rowsc is not None and rowsc.shape == (r_d,) and rowsf.shape == (cap,), (tag, rowsc.shape)
                h.same(rowsc, rowsf[:r_d], f"{tag} draft_rows")
            else:
                assert rowsc is None and rowsf is None, tag
            checked[r_d] += 1
    for i, (f, c) in enumerate(zip(full, capo)):
        h.same_out(c, f, f"rs={rs} step {i}")
    print(f"\nDRAFTCAP_PAGES rs={list(rs)} capacity={cap}: every pass reads the R_d prefix of the capacity pass "
          f"(seq_lens, query_start_loc, request_distribution, {len(h.NAMES)} layer tables over {len(h.COLS)} groups, "
          f"draft_rows) at the R_d shapes; passes per R_d {dict(sorted(checked.items()))}")


# ---------------------------------------------------------------- C. census


@pytest.mark.parametrize("R", [2, 12, 20, 40, 64])
def test_census_rd_shapes_two_draft_programs(monkeypatch, R):
    (f,), _ = h.serve(monkeypatch, "one", [55000 + R], R)
    (c,), cp = h.serve(monkeypatch, "cap", [55000 + R], R)
    r_d = min(_p(R), 64)
    assert c.eager == [] and c.deferred == [], (c.eager, c.deferred)
    assert c.calls["spec_tail"] == 0 and c.calls["seed_cl"] == 1 and c.calls["one"] == 1
    assert _draft_calls(c.calls) == _draft_calls(f.calls) == 2, (c.calls, f.calls)
    kc, kf = h.one_key(c), h.one_key(f)
    assert len(kc) == 16 + 2 * draft_cap.TABLE_SLOTS and len(kf) == 16
    assert [_shape(kc, i) for i in (3, 11, 12)] == [(r_d,), (r_d,), (r_d + 1,)], kc
    assert [_shape(kf, i) for i in (3, 11, 12)] == [(64,), (64,), (65,)], kf
    srcs = [_shape(kc, i) for i in range(T_SRC.start, T_SRC.stop)]
    bufs = [_shape(kc, i) for i in range(T_BUF.start, T_BUF.stop)]
    want = sorted(64 * c_ for c_ in h.COLS.values())
    assert sorted(s[0] for s in srcs if s) == want and srcs.count(None) == draft_cap.TABLE_SLOTS - len(h.COLS)
    assert sorted(b[0] for b in bufs if b) == sorted(r_d * c_ for c_ in h.COLS.values()), bufs
    assert all(_shape(kc, i) == _shape(kf, i) for i in (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15))
    print(f"\nDRAFTCAP_CENSUS R={R} p={_p(R)} R_d={r_d}: draft_rows {_shape(kc, 3)} seq_lens buf {_shape(kc, 11)} "
          f"qsl buf {_shape(kc, 12)} table bufs {[b for b in bufs if b]} (capacity program: draft_rows "
          f"{_shape(kf, 3)} bufs {_shape(kf, 11)} / {_shape(kf, 12)}, tables {[s for s in srcs if s]}); "
          f"draft_programs=2 eager=0 deferred=0")


# ---------------------------------------------------------------- D. programs


def test_one_rd_program_per_loop_bucket_compiling_only_its_token_buckets(monkeypatch):
    rec = {"info": [], "error": []}
    h.install(monkeypatch, fold=True, cl=True, one=True, cap=True)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 1)
    with h.quiet_logs(monkeypatch, rec):
        p = h.make_proposer()
        for R in range(1, h.MAX_REQS + 1):
            h.run_step(p, h.make_step(56000 + R, R), serving=False)
        h.run_step(p, h.make_step(56100, 9), serving=True)
    ctl = p._draft_asm
    mods = ctl.one_modules
    assert sorted(mods) == [16, 32, 48, 64], sorted(mods)
    codes = {type(m.module).forward.__code__ for m in mods.values()}
    assert len(codes) == 4 and draft_cap._DraftCapProgram.forward.__code__ not in codes
    assert draft_one._DraftOneProgram.forward.__code__ not in codes
    assert all(isinstance(m.module, draft_cap._DraftCapProgram) for m in mods.values())
    assert sorted(m.prefix for m in mods.values()) == sorted(f"draft_cap_k{h.K}_p{q}" for q in (16, 32, 48, 64))
    from vllm_torchtpu.compilation import bucket_filter

    steady = {(t, lp) for t, _w, lp, _c in ctl.decode_shapes()}
    for lp, m in mods.items():
        pred = bucket_filter.assigned(m)
        assert pred is not None
        assert {t for t in h.PADDINGS if pred(t)} == {t for t, q in steady if q == lp}, lp
    keys = {(k[0][0][0], k[4][0][0]) for k in ctl.warm["one"]}
    assert keys == steady and len(ctl.warm["one"]) == len(steady) == 12, (sorted(keys), sorted(steady))
    rd = {(k[4][0][0], k[3][0][0]) for k in ctl.warm["one"]}
    assert rd == {(q, q) for q in (16, 32, 48, 64)}, rd
    assert sorted(h.cap_bufs(ctl)) == [16, 32, 48, 64]
    print(f"\nDRAFTCAP_PROGRAMS {len(mods)} loop buckets, {len(ctl.warm['one'])} (token, loop) keys = the steady pairs "
          f"{sorted(steady)}; R_d per loop bucket {sorted(rd)}; buffer sets {sorted(h.cap_bufs(ctl))}")


# ---------------------------------------------------------------- E. window switching


def test_window_switching_with_rd_programs_never_compiles_and_stays_bitwise(monkeypatch):
    from test_draft1b_dhd import _FakeL6

    sched = {"draftasmcl": True, "draftasmone": True}
    fake = _FakeL6(["draftasmcl", "draftasmone"], steps=1, on=lambda lever: sched[lever])
    fake.install(monkeypatch)
    p = h.variant(monkeypatch, "cap", cl_ab="ab:1", one_ab="ab:1")
    fake.install(monkeypatch)
    ctl = p._draft_asm
    assert ctl.dha_needed and ctl.cl_conf and ctl.one_conf and ctl.one_cap
    ctl.precompile()
    rs = list(range(1, h.MAX_REQS + 1))
    warm_keys = set()
    for R in rs:
        warm_keys.update(h.run_step(p, h.make_step(57000 + R, R), serving=False).keys)
    warm_keys.update(("seed_cl", k) for k in ctl.warm["seed"])
    warm_keys.update(("carry_fresh_asm", k) for k in ctl.warm["carry"])
    warm_keys.update(("carry_asm", k) for k in ctl.warm["carry"])
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(fold=False)
    h.run_step(bp, h.make_step(57001, 1), serving=False)
    new, used = set(), Counter()
    for i, R in enumerate(rs):
        kind = ("one", "cl", "dha", "base")[i % 4]
        sched["draftasmone"] = kind == "one"
        sched["draftasmcl"] = kind in ("one", "cl")
        ctl.window_on = kind != "base"
        h.install(monkeypatch, fold=True, cl=True, one=True, cap=True, cl_ab="ab:1", one_ab="ab:1")
        fake.install(monkeypatch)
        f = h.run_step(p, h.make_step(58000 + R, R), serving=True)
        h.install(monkeypatch, fold=False)
        b = h.run_step(bp, h.make_step(58000 + R, R), serving=True)
        h.same_out(f, b, f"window step R={R} ({kind})")
        got = ("base" if not f.folded else "one" if f.calls["one"] else "cl" if f.calls["carry_loop"] else "dha")
        assert got == kind, (R, kind, got, f.calls, dict(ctl.reasons))
        if kind == "one":
            assert _shape(h.one_key(f), 3) == (_p(R),)
        used[got] += 1
        new.update(k for k in f.keys if k not in warm_keys)
    assert not new, sorted(new)[:4]
    c = ctl.counts()
    assert c["cold"] == 0 and c["folded_one"] == used["one"] and c["folded_cl"] == used["cl"]
    print(f"\nDRAFTCAP_COMPILE warm_keys={len(warm_keys)} new_after_warmup={len(new)} steps={len(rs)} "
          f"dhd_rd={used['one']} stageA={used['cl']} draftasm={used['dha']} base={used['base']} warm={c['warm_keys']}")


def test_cold_rd_key_falls_to_stage_a(monkeypatch):
    (b,), _ = h.serve(monkeypatch, "base", [58501], 20)
    p = h.variant(monkeypatch, "cap")
    h.run_step(p, h.make_step(58500, 20), serving=False)
    p._draft_asm.warm["one"].clear()
    o = h.run_step(p, h.make_step(58501, 20), serving=True)
    assert o.folded and o.calls["one"] == 0 and o.calls["carry_loop"] == 1, o.calls
    assert p._draft_asm.reasons["cold_one"] == 1
    h.same_out(o, b, "cold R_d DH-D -> stage A")


# ---------------------------------------------------------------- F. buffers, read-only metadata


def test_verify_metadata_and_tables_untouched_and_buffers_persist(monkeypatch):
    p = h.variant(monkeypatch, "cap")
    ctl = p._draft_asm
    h.run_step(p, h.make_step(59000, 24), serving=False)
    first = {r: (tuple(map(id, v[0])), tuple(map(id, v[1]))) for r, v in h.cap_bufs(ctl).items()}
    assert sorted(first) == [32]
    for i in range(3):
        step, ref = h.make_step(59001 + i, 24), h.make_step(59001 + i, 24)
        o = h.run_step(p, step, serving=True)
        assert o.calls["one"] == 1
        for f in ("seq_lens", "query_start_loc", "request_distribution"):
            h.same(getattr(step.chunk.attn_ctx, f), getattr(ref.chunk.attn_ctx, f), f"verify {f} after R_d DH-D")
        for name in h.NAMES:
            h.same(step.chunk.attn_metadata[name].block_tables, ref.chunk.attn_metadata[name].block_tables,
                   f"{name} block table after R_d DH-D")
        now = {r: (tuple(map(id, v[0])), tuple(map(id, v[1]))) for r, v in h.cap_bufs(ctl).items()}
        assert now == first, "R_d buffers re-allocated"
        k = h.one_key(o)
        assert _shape(k, 11) == (32,)
    print("\nDRAFTCAP_BUFS R=24: verify seq_lens / query_start_loc / request_distribution and 3 group tables unchanged "
          "after 3 R_d steps; one persistent buffer set for R_d 32")


def test_body_copies_prefixes_and_never_writes_its_sources(monkeypatch):
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
    from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop

    g = torch.Generator().manual_seed(5)
    T, cap, r_d, n = 48, 64, 16, 12
    mtp = h.RecMtpCap()
    loop = _MtpFusedLoop(vllm_config=None, prefix="t", mtp=mtp, num_steps=h.K - 1)
    md0 = (torch.randint(100, 999, (cap,), generator=g, dtype=torch.int32),
           torch.arange(cap + 1, dtype=torch.int32).clamp(max=n) * 4, torch.tensor([0, n, n], dtype=torch.int32))
    md1 = (torch.randint(100, 999, (cap,), generator=g, dtype=torch.int32),
           torch.arange(cap + 1, dtype=torch.int32).clamp(max=n), torch.tensor([n] * 3, dtype=torch.int32))
    tables = tuple(torch.randint(1, 9999, (cap * c,), generator=g, dtype=torch.int32) for c in h.COLS.values())
    keep0, keep1, keept = [t.clone() for t in md0], [t.clone() for t in md1], [t.clone() for t in tables]
    bufs = (torch.full((r_d,), 7, dtype=torch.int32), torch.full((r_d + 1,), 7, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32))
    tbufs = tuple(torch.full((r_d * c,), 7, dtype=torch.int32) for c in h.COLS.values())
    md = {name: AttentionMetadata(input_positions=torch.zeros(T, dtype=torch.int32), block_tables=tb,
                                  seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2])
          for name, tb in zip(h.GROUPS, tbufs)}
    md[h.MAIN2] = md[h.MAIN]
    rows = torch.arange(r_d, dtype=torch.int64) * 3 + 2
    with h.h0.fake_forward_context(md):
        out = draft_cap.dh_cap_body(
            loop, input_ids=torch.randint(1, 90, (T,), generator=g, dtype=torch.int32),
            positions=torch.randint(0, 9999, (T,), generator=g, dtype=torch.int32),
            target_hidden=torch.randn(T, h.H, generator=g), draft_rows=rows,
            positions_step1=torch.randint(0, 9999, (r_d,), generator=g, dtype=torch.int32),
            md0=md0, md1=md1, bufs=bufs, tables=(*tables, None), table_bufs=(*tbufs, None),
            tail_bonus=torch.zeros(h.MAX_REQS, dtype=torch.int32), num_reqs=torch.tensor(n), max_reqs=h.MAX_REQS)
    assert len(out) == h.K + 1 and out[-1].shape == (h.MAX_REQS * (1 + h.K),)
    for a, b, what in [*zip(md0, keep0, ["md0"] * 3), *zip(md1, keep1, ["md1"] * 3), *zip(tables, keept, ["table"] * 3)]:
        h.same(a, b, f"{what} untouched")
    for tb, src, c in zip(tbufs, tables, h.COLS.values()):
        h.same(tb, src[: r_d * c], "table buffer = request prefix")
    seen = mtp.seen
    h.same(seen[0][1], keep0[0][:r_d], "step 0 seq_lens prefix")
    h.same(seen[0][2], keep0[1][: r_d + 1], "step 0 query_start_loc prefix")
    h.same(seen[1][1], keep1[0][:r_d], "step 1 seq_lens prefix")
    h.same(seen[1][2], keep1[1][: r_d + 1], "step 1 query_start_loc prefix")
    h.same(seen[2][1], keep1[0][:r_d] + 1, "step 2 seq_lens prefix + 1")
    h.same(bufs[0], keep1[0][:r_d] + (h.K - 2), "seq_lens buffer ends at step K-1's")


# ---------------------------------------------------------------- G. configuration


def test_flag_needs_dhd(monkeypatch):
    rec = {"info": [], "error": [], "warning": []}
    with h.quiet_logs(monkeypatch, rec):
        for cl, one, one_ab, cap, want in [(True, True, None, True, True), (True, True, None, False, False),
                                           (True, False, None, True, False), (True, True, "0", True, False),
                                           (False, True, None, True, True), (True, True, "ab:1", True, True)]:
            h.install(monkeypatch, fold=True, cl=cl, one=one, cap=cap, one_ab=one_ab)
            p = h.make_proposer()
            assert isinstance(p._draft_asm, draft_one.DraftOne)
            assert p._draft_asm.one_cap is want, (cl, one, one_ab, cap)
    warns = [w for w in rec["warning"] if "DRAFT-CAP" in w]
    assert len(warns) == 2 and all("needs DH-D" in w for w in warns), rec["warning"]


def test_flag_default_off_registered_and_hashed(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    monkeypatch.delenv(h.CAP_ENV, raising=False)
    assert h.CAP_ENV in envs.environment_variables and envs.VLLM_TPU_DRAFT_ASM_CAP is False
    assert h.CAP_ENV not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.setenv(h.CAP_ENV, "1")
    assert envs.VLLM_TPU_DRAFT_ASM_CAP is True
    print("\nDRAFTCAP_FLAG VLLM_TPU_DRAFT_ASM_CAP default off, registered, compile-hashed (not ignored); on-value 1")


# ---------------------------------------------------------------- H. tables


def test_more_tables_than_slots_keeps_dhd_off(monkeypatch):
    groups = {f"draft.attn.0.g{i}": 1 + i for i in range(draft_cap.TABLE_SLOTS + 1)}
    names = tuple(groups) + (h.MAIN2,)
    groups = {h.MAIN: 2, **groups}
    names = (h.MAIN, *names)
    seeds = [59500, 59501]
    kw = {"step_kw": {"groups": groups}, "names": names}
    base, _ = h.serve(monkeypatch, "base", seeds, 20, **kw)
    capo, cp = h.serve(monkeypatch, "cap", seeds, 20, **kw)
    for b, c in zip(base, capo):
        assert c.folded and c.calls["one"] == 0 and c.calls["carry_loop"] == 1, c.calls
        h.same_out(c, b, "too many tables -> stage A")
    assert cp._draft_asm.reasons["one_cap_tables"] == 2 and cp._draft_asm.reasons["warm_skip_one_cap_tables"] == 1


def test_groups_sharing_one_table_share_one_buffer(monkeypatch):
    kw = {"step_kw": {"share": {h.RING: h.COMP}}}
    seeds = [59600 + i for i in range(3)]
    base, _ = h.serve(monkeypatch, "base", seeds, 12, **kw)
    capo, cp = h.serve(monkeypatch, "cap", seeds, 12, **kw)
    for b, c in zip(base, capo):
        assert c.calls["one"] == 1
        h.same_out(c, b, "shared table")
        k = h.one_key(c)
        assert sum(1 for i in range(T_SRC.start, T_SRC.stop) if k[i] is not None) == 2
        assert sum(1 for i in range(T_BUF.start, T_BUF.stop) if k[i] is not None) == 2
    (_bufs, tbufs), = h.cap_bufs(cp._draft_asm).values()
    assert len(tbufs) == 2


# ---------------------------------------------------------------- I. agreement


def test_rd_and_capacity_programs_never_agree(monkeypatch):
    """The R_d shapes enter the DH-D warm keys and so DRAFT-ASM's fingerprint: a rank with the flag and one
    without disagree at the boundary, and every rank keeps the base path."""
    pf = h.variant(monkeypatch, "one")
    pc = h.variant(monkeypatch, "cap")
    for p in (pf, pc):
        for R in (3, 20, 40):
            h.run_step(p, h.make_step(59700 + R, R), serving=False)
    ff, fc = pf._draft_asm._fingerprint(), pc._draft_asm._fingerprint()
    assert ff != fc and len(pf._draft_asm.warm["one"]) == len(pc._draft_asm.warm["one"]) == 3
    ctl = pc._draft_asm
    ctl.rank = pf._draft_asm.rank = 0
    vec_c, vec_f = ctl.local_vector(), pf._draft_asm.local_vector()
    assert vec_c[0] == vec_f[0] and vec_c[-1] != vec_f[-1]
    ctl.apply([vec_c, vec_f])
    assert not ctl.agreed and not ctl.one_agreed
    ctl2 = h.variant(monkeypatch, "cap")._draft_asm
    ctl2.rank = 0
    for R in (3, 20, 40):
        h.run_step(ctl2.p, h.make_step(59700 + R, R), serving=False)
    v2 = ctl2.local_vector()
    ctl2.apply([v2, list(v2)])
    assert ctl2.agreed and ctl2.one_agreed
    print(f"\nDRAFTCAP_AGREE fingerprints R_d {fc:x} vs capacity {ff:x}: mixed ranks keep the base path; equal ranks agree")


# ---------------------------------------------------------------- J. CHECK


def test_check_mode_compares_rd_dhd_then_arms_it(monkeypatch):
    rec = {"info": [], "error": []}
    h.install(monkeypatch, fold=True, cl=True, one=True, cap=True, check_steps=3)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 4)
    with h.quiet_logs(monkeypatch, rec):
        p = h.make_proposer()
        h.run_step(p, h.make_step(59801, 20), serving=False)
        outs = [h.run_step(p, h.make_step(59802 + i, 20), serving=True) for i in range(5)]
    assert [o.folded for o in outs] == [False, False, False, True, True]
    assert all(o.calls["one"] == 1 and o.calls["spec_tail"] == 1 for o in outs[:3])
    c = p._draft_asm.counts()
    assert c["checked"] == 3 and c["mismatches"] == 0 and c["folded_one"] == 2, (c, rec["error"])
    print("\nDRAFTCAP_CHECK clean: checked 3 (R_d DH-D beside the base path) mismatches 0, then R_d DH-D folded 2")


def test_check_mode_catches_a_perturbed_rd_output(monkeypatch):
    rec = {"info": [], "error": []}
    h.install(monkeypatch, fold=True, cl=True, one=True, cap=True, check_steps=2)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 3)
    real = draft_cap.dh_cap_body
    monkeypatch.setattr(draft_cap, "dh_cap_body", lambda *a, **k: (lambda o: (o[0], o[1] + 1, *o[2:]))(real(*a, **k)))
    with h.quiet_logs(monkeypatch, rec):
        p = h.make_proposer()
        h.run_step(p, h.make_step(59901, 16), serving=False)
        outs = [h.run_step(p, h.make_step(59902 + i, 16), serving=True) for i in range(4)]
    assert not any(o.folded for o in outs)
    c = p._draft_asm.counts()
    assert c["mismatches"] >= 2 and p._draft_asm.failed, c
    assert "DH-D" in rec["error"][0], rec["error"]


# ---------------------------------------------------------------- K. markers, build failure


def test_engagement_line_once_per_rd_in_serving_only(monkeypatch):
    rec = {"info": [], "error": []}
    h.install(monkeypatch, fold=True, cl=True, one=True, cap=True)
    with h.quiet_logs(monkeypatch, rec):
        p = h.make_proposer()
        for R in (3, 20):
            h.run_step(p, h.make_step(60000 + R, R), serving=False)
        warm_lines = [m for m in rec["info"] if m.startswith("q38 DRAFT-CAP: on")]
        for i, R in enumerate((3, 3, 20, 20, 3)):
            h.run_step(p, h.make_step(60100 + i, R), serving=True)
    lines = [m for m in rec["info"] if m.startswith("q38 DRAFT-CAP: on")]
    assert warm_lines == [] and len(lines) == 2, rec["info"]
    assert "loop bucket 16: R_d 16 of capacity 64" in lines[0] and "loop bucket 32: R_d 32" in lines[1], lines
    assert "3 block tables" in lines[0]
    print(f"\nDRAFTCAP_MARKER {lines[0]} | {lines[1]}")


def test_a_build_failure_stops(monkeypatch):
    p = h.variant(monkeypatch, "cap")

    def boom(**kw):
        raise ValueError("no")

    monkeypatch.setattr(draft_cap, "build_draft_cap_program", boom)
    with pytest.raises(RuntimeError, match="VLLM_TPU_DRAFT_ASM_CAP=0"):
        h.run_step(p, h.make_step(60200, 12), serving=False)

