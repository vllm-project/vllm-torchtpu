# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT1-B DH-D (VLLM_TPU_DRAFT_ASM_ONE, H1 card DH-D DRAFT1-B, with X-D14-DRAFT-3 (a)) through the
real ``Eagle3Proposer.propose`` and the real async spec substitution (tests/q38_draft1b/draft1b_harness.py).

A. Bitwise over 8 serving steps: DH-D == the base path for the drafts, step 1's positions / seq_lens / first
   token, the source, the rejected count and the index dicts; rows 4 / 8 / 12 / 16 / 32 / 48 / 64, K = 3,
   fallback rows on and off; fewer-draft requests.
B. Per-pass metadata: the draft's forwards inside DH-D read exactly the (step index, seq_lens,
   query_start_loc, request_distribution, draft_rows) sequence the base path's forwards read.
C. Census: a DH-D step runs 2 compiled draft programs (seed + the one program; 1 with the verify's seed)
   where stage A runs 3 and DRAFT-ASM 4; 0 eager ops, 0 deferred inputs, no spec tail.
D. Programs: one per loop bucket, each with its own forward code object, compiling only its token buckets;
   the warm-up records exactly the steady (token, loop) bucket pairs; RSS line per rank.
E. Fallbacks: a token bucket its program does not compile, a never-warmed key, the debug checks: stage A.
F. SAB: per-step windows over DH-D / stage A / DRAFT-ASM / base never compile after the warm-up; bitwise.
G. CHECK compares DH-D on its steps and catches perturbed DH-D outputs.
H. The body (KLOOP interface): md0 / md1 never written; buffers switched in place; the int32 switch is set
   only inside DH-D calls; the MTP glue under it passes int32 metadata with the int64 path's values.
I. Agreement: one rank's DH-D wish off turns DH-D off on every rank at the next boundary, stage A stays.
"""

import sys
import threading
from collections import Counter
from types import ModuleType, SimpleNamespace

import pytest
import torch

import draft1b_harness as h
from vllm_torchtpu.spec_decode import draft_asm, draft_one

ROWS = [4, 8, 12, 16, 32, 48, 64]
DRAFT_PROGRAMS = ("seed", "seed_asm", "seed_cl", "draft_fwd0", "carry", "carry_fresh", "carry_asm",
                  "carry_fresh_asm", "loop", "loop_asm", "carry_loop", "one")


def _draft_calls(calls):
    return sum(v for k, v in calls.items() if k in DRAFT_PROGRAMS)


def _variant(monkeypatch, name, cap=64, mtp_cls=None, **kw):
    """base (fold off), dha (DRAFT-ASM), cl (+ stage A), one (+ DH-D, stage A configured beside it)."""
    h.install(monkeypatch, fold=name != "base", cl=name in ("cl", "one"), one=name == "one", **kw)
    return h.make_proposer(fold=name != "base", cap=cap, mtp_cls=mtp_cls)


def _same_one(a, b, tag):
    h.same(a.drafts, b.drafts, f"{tag} drafts")
    h.same(a.src, b.src, f"{tag} source")
    h.same(a.rejected, b.rejected, f"{tag} rejected count")
    assert a.idx == b.idx and a.ndpr == b.ndpr, f"{tag} index dicts"
    for k in ("seq_lens", "positions", "input_ids"):
        h.same(a.loop_entry[k], b.loop_entry[k], f"{tag} loop-entry {k}")


def _serve(monkeypatch, name, seeds, R, warm_each=False, cap=64, step_kw=None, mtp_cls=None, **kw):
    step_kw = step_kw or {}
    p = _variant(monkeypatch, name, cap=cap, mtp_cls=mtp_cls, **kw)
    if not warm_each:
        h.run_step(p, h.make_step(seeds[0] - 1, R, cap=cap, **step_kw), serving=False)
    outs = []
    for s in seeds:
        if warm_each:
            h.run_step(p, h.make_step(s, R, cap=cap, **step_kw), serving=False)
        outs.append(h.run_step(p, h.make_step(s, R, cap=cap, **step_kw), serving=True))
    return outs, p


# ---------------------------------------------------------------- A. bitwise over 8 steps


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("R", ROWS)
def test_dhd_is_bitwise_base_over_8_steps(monkeypatch, R, fallback):
    seeds = [31000 + 100 * R + 10 * fallback + i for i in range(8)]
    kw = {"step_kw": {"fallback": fallback}}
    base, _ = _serve(monkeypatch, "base", seeds, R, **kw)
    one, op = _serve(monkeypatch, "one", seeds, R, **kw)
    for i, (b, o) in enumerate(zip(base, one)):
        assert not b.folded and o.folded, (i, op._draft_asm.counts(), dict(op._draft_asm.reasons))
        assert o.calls["one"] == 1 and o.calls["draft_fwd0"] == 0 and o.calls["carry_loop"] == 0, o.calls
        _same_one(o, b, f"DH-D vs base R={R} step {i}")
    assert op._draft_asm.counts()["folded_one"] == 8
    print(f"\nDRAFT1B_BITWISE dhd R={R} fallback={int(fallback)} steps=8 drafts {tuple(one[0].drafts.shape)} "
          f"positions/seq_lens/token0 at step 1, source[{one[0].src.numel()}], rejected[{one[0].rejected.numel()}], "
          f"index dicts equal to base on every step")


@pytest.mark.parametrize("R", [12, 32])
def test_dhd_bitwise_with_fewer_drafts(monkeypatch, R):
    seeds = [32000 + 100 * R + i for i in range(8)]
    kw = {"warm_each": True, "step_kw": {"short": True}}
    base, _ = _serve(monkeypatch, "base", seeds, R, **kw)
    one, op = _serve(monkeypatch, "one", seeds, R, **kw)
    for i, (b, o) in enumerate(zip(base, one)):
        _same_one(o, b, f"short R={R} step {i}")
    c = op._draft_asm.counts()
    print(f"\nDRAFT1B_BITWISE dhd_short R={R} steps=8 fewer-draft requests equal to base "
          f"(folded {c['folded']}: DH-D {c['folded_one']}, stage A {c['folded_cl']})")


# ---------------------------------------------------------------- B. per-pass metadata


@pytest.mark.parametrize("R", [12, 64])
def test_each_pass_reads_the_base_paths_metadata(monkeypatch, R):
    (b,), bp = _serve(monkeypatch, "base", [33000 + R], R, mtp_cls=h.RecMtp)
    (o,), op = _serve(monkeypatch, "one", [33000 + R], R, mtp_cls=h.RecMtp)
    assert o.calls["one"] == 1
    # the serving step's forwards are the last 1 + (K - 1) of each record
    nb, no = bp.draft_model.seen[-h.K:], op.draft_model.seen[-h.K:]
    assert [s[0] for s in nb] == [s[0] for s in no] == list(range(h.K))
    for (sb, *vb), (so, *vo) in zip(nb, no):
        for what, x, y in zip(("seq_lens", "query_start_loc", "request_distribution", "draft_rows"), vb, vo):
            if x is None or y is None:
                assert x is None and y is None, (sb, what)
            else:
                h.same(y, x, f"pass {sb} {what}")
    print(f"\nDRAFT1B_PAGES dhd R={R} passes={h.K}: step 0 reads the verify metadata and draft_rows, loop steps "
          f"read step 1's seq_lens (+1 per step) and the loop query_start_loc / request_distribution, as the base path")


def test_dhd_leaves_the_verify_metadata_untouched(monkeypatch):
    """md0 is read, never written: after a DH-D step the verify chunk's seq_lens / query_start_loc /
    request_distribution hold their values (the program writes only its own buffers)."""
    p = _variant(monkeypatch, "one")
    h.run_step(p, h.make_step(33500, 24), serving=False)
    step, ref = h.make_step(33501, 24), h.make_step(33501, 24)
    o = h.run_step(p, step, serving=True)
    assert o.calls["one"] == 1
    for f in ("seq_lens", "query_start_loc", "request_distribution"):
        h.same(getattr(step.chunk.attn_ctx, f), getattr(ref.chunk.attn_ctx, f), f"verify {f} after DH-D")


# ---------------------------------------------------------------- C. census


@pytest.mark.parametrize("R", [8, 32, 48, 64])
def test_census_two_draft_programs(monkeypatch, R):
    (d,), _ = _serve(monkeypatch, "dha", [34000 + R], R)
    (c,), _ = _serve(monkeypatch, "cl", [34000 + R], R)
    (o,), _ = _serve(monkeypatch, "one", [34000 + R], R)
    assert o.eager == [] and o.deferred == [], (o.eager, o.deferred)
    assert o.calls["spec_tail"] == 0 and o.calls["seed_cl"] == 1 and o.calls["one"] == 1
    nd, nc, no = _draft_calls(d.calls), _draft_calls(c.calls), _draft_calls(o.calls)
    assert (nd, nc, no) == (4, 3, 2), (d.calls, c.calls, o.calls)
    print(f"\nDRAFT1B_CENSUS R={R} DRAFT-ASM draft_programs={nd} stageA draft_programs={nc} "
          f"dhd draft_programs={no} (seed + one; 1 with the VFY-ONE value 2 seed) ({dict(sorted(o.calls.items()))}) "
          f"eager=0 deferred=0 spec_tail=0")


def test_census_one_draft_program_with_the_verify_seed(monkeypatch):
    p = _variant(monkeypatch, "one")
    p._vfy_seed_take = lambda chunk, nt, impl, rest: impl(nt, *rest)
    h.run_step(p, h.make_step(34900, 20), serving=False)
    o = h.run_step(p, h.make_step(34901, 20), serving=True)
    assert o.folded and _draft_calls(o.calls) == 1 and o.calls["one"] == 1, o.calls
    print(f"\nDRAFT1B_CENSUS R=20 dhd_vfy draft_programs=1 ({dict(sorted(o.calls.items()))})")


# ---------------------------------------------------------------- D. programs


def test_one_program_per_loop_bucket_compiling_only_its_token_buckets(monkeypatch):
    rec = {"info": [], "error": []}
    h.install(monkeypatch, fold=True, cl=True, one=True)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 1)
    with h.quiet_logs(monkeypatch, rec):
        p = h.make_proposer()
        for R in range(1, h.MAX_REQS + 1):
            h.run_step(p, h.make_step(35000 + R, R), serving=False)
        h.run_step(p, h.make_step(35100, 9), serving=True)
    ctl = p._draft_asm
    mods = ctl.one_modules
    assert sorted(mods) == [16, 32, 48, 64], sorted(mods)
    codes = {type(m.module).forward.__code__ for m in mods.values()}
    assert len(codes) == 4 and draft_one._DraftOneProgram.forward.__code__ not in codes
    from vllm_torchtpu.compilation import bucket_filter

    steady = {(t, lp) for t, _w, lp, _c in ctl.decode_shapes()}
    for lp, m in mods.items():
        pred = bucket_filter.assigned(m)
        assert pred is not None
        assert {t for t in h.PADDINGS if pred(t)} == {t for t, q in steady if q == lp}, lp
    keys = {(k[0][0][0], k[4][0][0]) for k in ctl.warm["one"]}
    assert keys == steady, (sorted(keys), sorted(steady))
    rss = [m for m in rec["info"] if "DH-D program keys" in m]
    assert len(rss) == 1 and "loop buckets" in rss[0], rec["info"]
    print(f"\nDRAFT1B_RSS dhd programs: {len(mods)} loop buckets, {len(ctl.warm['one'])} (token, loop) keys = the "
          f"steady pairs {sorted(steady)}; {rss[0]}")


# ---------------------------------------------------------------- E. fallbacks


def test_cold_dhd_key_falls_to_stage_a(monkeypatch):
    (b,), _ = _serve(monkeypatch, "base", [36001], 20)
    p = _variant(monkeypatch, "one")
    h.run_step(p, h.make_step(36000, 20), serving=False)
    p._draft_asm.warm["one"].clear()
    o = h.run_step(p, h.make_step(36001, 20), serving=True)
    assert o.folded and o.calls["one"] == 0 and o.calls["carry_loop"] == 1, o.calls
    assert p._draft_asm.reasons["cold_one"] == 1
    _same_one(o, b, "cold DH-D -> stage A")


def test_debug_checks_keep_dhd_off(monkeypatch):
    p = _variant(monkeypatch, "one")
    h.run_step(p, h.make_step(36100, 12), serving=False)
    p._draft_top_checks_left = 1
    p._check_draft_top_tokens = lambda hidden, tokens: None
    o = h.run_step(p, h.make_step(36101, 12), serving=True)
    assert o.calls["one"] == 0 and p._draft_asm.reasons["one_debug_checks"] == 1, (o.calls, p._draft_asm.reasons)


def test_unserved_token_bucket_is_never_called(monkeypatch):
    """A (token, loop) pair outside the steady set (short drafts at 20 requests: T 48 with loop bucket 32)
    is not compiled by loop bucket 32's program: DH-D is skipped, stage A folds."""
    p = _variant(monkeypatch, "one")
    ctl = p._draft_asm
    for s in range(40):
        step = h.make_step(36200 + s, 20, short=True)
        if step.chunk.input_ids.shape[0] not in ctl._one_serves(32):
            break
    else:
        pytest.skip("no unserved short step found")
    h.run_step(p, step, serving=False)
    assert ctl.reasons["warm_skip_one_bucket"] == 1 and not ctl.warm["one"], dict(ctl.reasons)


# ---------------------------------------------------------------- F. SAB windows


class _FakeL6:
    def __init__(self, levers, steps=1, on=None):
        self.LEVERS = {k: "x" for k in levers}
        self.STEPS = steps
        self.on_fn = on or (lambda lever: True)

    def install(self, monkeypatch):
        import vllm_torchtpu.core as core

        mod = ModuleType("vllm_torchtpu.core.l6_ab")
        mod.LEVERS, mod.STEPS = self.LEVERS, self.STEPS
        mod.on = lambda lever: self.on_fn(lever)
        mod.note = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.core.l6_ab", mod)
        monkeypatch.setattr(core, "l6_ab", mod, raising=False)
        monkeypatch.setattr(draft_asm, "_L6_AB", [])


def test_window_switching_over_four_paths_never_compiles_and_stays_bitwise(monkeypatch):
    sched = {"draftasmcl": True, "draftasmone": True}
    fake = _FakeL6(["draftasmcl", "draftasmone"], steps=1, on=lambda lever: sched[lever])
    fake.install(monkeypatch)
    p = _variant(monkeypatch, "one", cl_ab="ab:1", one_ab="ab:1")
    fake.install(monkeypatch)
    ctl = p._draft_asm
    assert ctl.dha_needed and ctl.cl_conf and ctl.one_conf
    ctl.precompile()
    rs = list(range(1, h.MAX_REQS + 1))
    warm_keys = set()
    for R in rs:
        warm_keys.update(h.run_step(p, h.make_step(37000 + R, R), serving=False).keys)
    warm_keys.update(("seed_cl", k) for k in ctl.warm["seed"])
    warm_keys.update(("carry_fresh_asm", k) for k in ctl.warm["carry"])
    warm_keys.update(("carry_asm", k) for k in ctl.warm["carry"])
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(fold=False)
    h.run_step(bp, h.make_step(37001, 1), serving=False)
    new, used = set(), Counter()
    for i, R in enumerate(rs):
        kind = ("one", "cl", "dha", "base")[i % 4]
        sched["draftasmone"] = kind == "one"
        sched["draftasmcl"] = kind in ("one", "cl")
        ctl.window_on = kind != "base"
        h.install(monkeypatch, fold=True, cl=True, one=True, cl_ab="ab:1", one_ab="ab:1")
        fake.install(monkeypatch)
        f = h.run_step(p, h.make_step(38000 + R, R), serving=True)
        h.install(monkeypatch, fold=False)
        b = h.run_step(bp, h.make_step(38000 + R, R), serving=True)
        _same_one(f, b, f"window step R={R} ({kind})")
        got = ("base" if not f.folded else "one" if f.calls["one"] else "cl" if f.calls["carry_loop"] else "dha")
        assert got == kind, (R, kind, got, f.calls, dict(ctl.reasons))
        used[got] += 1
        new.update(k for k in f.keys if k not in warm_keys)
    assert not new, sorted(new)[:4]
    c = ctl.counts()
    assert c["cold"] == 0 and c["folded_one"] == used["one"] and c["folded_cl"] == used["cl"]
    print(f"\nDRAFT1B_COMPILE dhd warm_keys={len(warm_keys)} new_after_warmup={len(new)} steps={len(rs)} "
          f"dhd={used['one']} stageA={used['cl']} draftasm={used['dha']} base={used['base']} warm={c['warm_keys']}")


# ---------------------------------------------------------------- G. CHECK


def test_check_mode_compares_dhd_then_arms_it(monkeypatch):
    rec = {"info": [], "error": []}
    h.install(monkeypatch, fold=True, cl=True, one=True, check_steps=3)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 4)
    with h.quiet_logs(monkeypatch, rec):
        p = h.make_proposer()
        h.run_step(p, h.make_step(39001, 20), serving=False)
        outs = [h.run_step(p, h.make_step(39002 + i, 20), serving=True) for i in range(5)]
    assert [o.folded for o in outs] == [False, False, False, True, True]
    assert all(o.calls["one"] == 1 and o.calls["carry_loop"] == 1 and o.calls["spec_tail"] == 1 for o in outs[:3])
    c = p._draft_asm.counts()
    assert c["checked"] == 3 and c["mismatches"] == 0 and c["folded_one"] == 2, (c, rec["error"])
    print("\nDRAFT1B_CHECK dhd clean: checked 3 (DH-D and stage A beside the base path) mismatches 0, then DH-D folded 2")


@pytest.mark.parametrize("what", ["source", "token0", "loop_token"])
def test_check_mode_catches_a_perturbed_dhd_output(monkeypatch, what):
    rec = {"info": [], "error": []}
    h.install(monkeypatch, fold=True, cl=True, one=True, check_steps=2)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 3)
    real = draft_one.dh_one_body
    if what == "source":
        monkeypatch.setattr(draft_one, "dh_one_body", lambda *a, **k: (lambda o: (*o[:-1], o[-1] + 1))(real(*a, **k)))
    elif what == "token0":
        monkeypatch.setattr(draft_one, "dh_one_body", lambda *a, **k: (lambda o: (o[0] + 1, *o[1:]))(real(*a, **k)))
    else:
        monkeypatch.setattr(draft_one, "dh_one_body", lambda *a, **k: (lambda o: (o[0], o[1] + 1, *o[2:]))(real(*a, **k)))
    with h.quiet_logs(monkeypatch, rec):
        p = h.make_proposer()
        h.run_step(p, h.make_step(39101, 16), serving=False)
        outs = [h.run_step(p, h.make_step(39102 + i, 16), serving=True) for i in range(4)]
    assert not any(o.folded for o in outs), [o.folded for o in outs]
    c = p._draft_asm.counts()
    assert c["mismatches"] >= 2 and p._draft_asm.failed, c
    errs = rec["error"]
    assert "CHECK mismatch" in errs[0] and "DH-D" in errs[0] and "every rank keeps the base path" in errs[-1], errs
    print(f"\nDRAFT1B_CHECK dhd perturbed {what}: caught ({errs[0][:90]}), fold off for the run")


# ---------------------------------------------------------------- H. the body, the int32 switch


def test_body_switches_buffers_in_place_and_never_writes_md0_md1(monkeypatch):
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
    from vllm_torchtpu.spec_decode.mtp_fused_loop import _MtpFusedLoop

    g = torch.Generator().manual_seed(3)
    T, cap, p_rows, n = 48, 64, 16, 12
    mtp = h.RecMtp()
    loop = _MtpFusedLoop(vllm_config=None, prefix="t", mtp=mtp, num_steps=h.K - 1)
    md0 = (torch.randint(100, 999, (cap,), generator=g, dtype=torch.int32),
           torch.arange(cap + 1, dtype=torch.int32) * 4, torch.tensor([0, n, n], dtype=torch.int32))
    md1 = (torch.randint(100, 999, (cap,), generator=g, dtype=torch.int32),
           torch.arange(cap + 1, dtype=torch.int32).clamp(max=n), torch.tensor([n] * 3, dtype=torch.int32))
    keep0, keep1 = [t.clone() for t in md0], [t.clone() for t in md1]
    bufs = tuple(torch.zeros_like(t) for t in md0)
    md = {"d": AttentionMetadata(input_positions=torch.zeros(T, dtype=torch.int32), block_tables=torch.arange(4),
                                 seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2])}
    rows = torch.full((cap,), -1, dtype=torch.int64)
    rows[:n] = torch.arange(n) * 4 + 3
    with h.h.fake_forward_context(md):
        out = draft_one.dh_one_body(
            loop, input_ids=torch.randint(1, 90, (T,), generator=g, dtype=torch.int32),
            positions=torch.randint(0, 9999, (T,), generator=g, dtype=torch.int32),
            target_hidden=torch.randn(T, h.H, generator=g), draft_rows=rows,
            positions_step1=torch.randint(0, 9999, (p_rows,), generator=g, dtype=torch.int32),
            md0=md0, md1=md1, bufs=bufs, tail_bonus=torch.zeros(h.MAX_REQS, dtype=torch.int32),
            num_reqs=torch.tensor(n), max_reqs=h.MAX_REQS)
    assert len(out) == h.K + 1 and out[-1].shape == (h.MAX_REQS * (1 + h.K),)
    for a, b in zip(md0, keep0):
        h.same(a, b, "md0 untouched")
    for a, b in zip(md1, keep1):
        h.same(a, b, "md1 untouched")
    h.same(mtp.seen[0][1], keep0[0], "step 0 seq_lens")
    h.same(mtp.seen[0][2], keep0[1], "step 0 query_start_loc")
    h.same(mtp.seen[1][1], keep1[0], "step 1 seq_lens")
    h.same(mtp.seen[2][1], keep1[0] + 1, "step 2 seq_lens")
    h.same(mtp.seen[1][3], keep1[2], "step 1 request_distribution")
    h.same(bufs[0], keep1[0] + (h.K - 2), "buffer ends at step K-1's seq_lens")


def test_int32_switch_is_set_only_inside_dhd_calls(monkeypatch):
    seen = []

    class Model:
        meta_i32 = False

    class SwitchMtp(h.FakeMtp2):
        def __init__(self):
            super().__init__()
            self.model = Model()

        def forward(self, *a, **k):
            seen.append(self.model.meta_i32)
            return super().forward(*a, **k)

    (o,), op = _serve(monkeypatch, "one", [39301], 9, mtp_cls=SwitchMtp)
    assert o.calls["one"] == 1 and op.draft_model.model.meta_i32 is False
    # warm-up: DH-D (True x K, twice for its first call), then stage A and the base path (False); serving: DH-D
    assert seen[-h.K:] == [True] * h.K and False in seen, seen


@pytest.fixture
def cpu_vllm_config():
    """As tests/models/vllm/test_qwen4_exp_mtp.py's: the predictor's norms are vLLM CustomOps, which read the
    current vLLM config when they are built."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield


@pytest.mark.usefixtures("cpu_vllm_config")
def test_mtp_glue_int32_metadata_has_the_int64_values(monkeypatch):
    from types import SimpleNamespace as NS

    from torch import nn

    from tests.models.vllm.qwen4_exp_test_utils import q4exp_text_config
    from vllm_torchtpu.models.vllm.qwen4_exp import mtp

    text = q4exp_text_config(hidden_size=8, hc_count=4, hc_lowrank=4, vocab_size=16, num_hidden_layers=2)
    seen = {}

    class Layer(nn.Module):
        def __init__(self, cfg, index, prefix, *, layer_type):
            super().__init__()
            self.self_attn = NS(prefix=prefix + ".self_attn")

        def forward(self, hidden, positions, input_ids, starts, ngram, slots, context, valid, main, compressed):
            seen.setdefault("all", []).append((positions, input_ids, context, valid))
            return hidden * 2

        def forward_rows(self, hidden, positions, rows, starts, slots, context, main, compressed):
            seen.setdefault("rows", []).append((positions, rows, context))
            return hidden[rows.clamp_min(0)] * 2

    monkeypatch.setattr(mtp, "Qwen4ExpDecoderLayer", Layer)
    monkeypatch.setattr(mtp, "VocabParallelEmbedding", nn.Embedding)
    for module in ("parameter", "layers.linear"):
        for name, value in (("get_tensor_model_parallel_rank", 0), ("get_tensor_model_parallel_world_size", 1)):
            monkeypatch.setattr(f"vllm.model_executor.{module}.{name}", lambda value=value: value, raising=False)
    cfg = NS(model_config=NS(hf_text_config=text), additional_config={})
    with torch.device("cpu"):
        predictor = mtp.Qwen4ExpMultiTokenPredictor(cfg, ["mtp.layers.2"])
    assert predictor.meta_i32 is False
    predictor = predictor.to(torch.bfloat16)  # the draft's fc weights are BF16 (as the existing glue test)
    for prm in predictor.parameters():
        prm.data.normal_(0, 0.2)
    ids = torch.tensor([3, 5, 7], dtype=torch.int32)
    pos = torch.tensor([7, 8, 0], dtype=torch.int32)
    target = torch.randn(3, 32).to(torch.bfloat16)
    metadata = NS(query_start_loc=torch.tensor([0, 2, 2, 2], dtype=torch.int32),
                  seq_lens=torch.tensor([9, 5, 0], dtype=torch.int32), block_tables=torch.zeros(3, dtype=torch.int32))
    monkeypatch.setattr(mtp, "get_forward_context", lambda: NS(attn_metadata={"mtp.layers.2.self_attn": metadata}))
    rows = torch.tensor([1, -1, 2], dtype=torch.int64)
    for flag in (False, True):
        predictor.meta_i32 = flag
        predictor(ids, pos, target, None, 0)
        predictor(ids, pos, target, None, 0, rows=rows)
    (p64, i64, c64, v64), (p32, i32, c32, v32) = seen["all"]
    (rp64, r64, rc64), (rp32, r32, rc32) = seen["rows"]
    assert p64.dtype == i64.dtype == c64.dtype == r64.dtype == rp64.dtype == torch.int64
    assert p32.dtype == i32.dtype == c32.dtype == r32.dtype == rp32.dtype == torch.int32
    for a, b in ((p32, p64), (i32, i64), (c32, c64), (r32, r64), (rp32, rp64), (rc32, rc64)):
        assert a.tolist() == b.tolist()
    assert v32.tolist() == v64.tolist() == [True, True, False]
    print("\nDRAFT1B_DTYPE mtp glue: meta_i32 on passes int32 positions / input_ids / context / rows (int64 off), "
          "same values; valid mask equal")


# ---------------------------------------------------------------- I. agreement


def test_one_ranks_dhd_wish_off_turns_dhd_off_everywhere(monkeypatch):
    from test_draft1b_multirank import FakeGroup

    WORLD, R = 8, 6
    h.install(monkeypatch, fold=True, cl=True, one=True)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", 4)
    quiet = SimpleNamespace(info=lambda *a: None, error=lambda *a: None, warning=lambda *a, **k: None)
    monkeypatch.setattr(draft_asm, "logger", quiet)
    monkeypatch.setattr(draft_one, "logger", quiet)
    group = FakeGroup(WORLD)
    tl = threading.local()
    # q38 L4 AGREE port (v5 composition, V5COMP-BUILD; tests only): the boundary gather is l4_agree's.
    from vllm_torchtpu.runner import l4_agree

    monkeypatch.setattr(l4_agree, "gather", lambda vec: group.gather(tl.rank, vec))
    real = draft_one.DraftOne._stage_window
    monkeypatch.setattr(draft_one.DraftOne, "_stage_window",
                        lambda self, lever, ab: False if (lever == "draftasmone" and getattr(self, "_q38_off", False))
                        else real(self, lever, ab))
    ranks = []
    for r in range(WORLD):
        p = h.make_proposer()
        h.run_step(p, h.make_step(500, R), serving=False)
        p._draft_asm.rank = r
        ranks.append(p)
    paths = [[None] * 16 for _ in range(WORLD)]
    errors = []

    def worker(r):
        tl.rank = r
        ctl = ranks[r]._draft_asm
        ranks[r].runner.warmup_done = True
        try:
            for i in range(16):
                ctl._q38_off = r == 2 and 5 <= i <= 9
                step = h.make_step(600 + i, R)
                ranks[r].runner._l4_agree.tick(ranks[r].runner)  # q38 L4 AGREE port: the engine step's clock tick
                ctl.begin([step.chunk], [step.nt], SimpleNamespace(has_structured_output_requests=False), True)
                st = ctl.step
                paths[r][i] = ("b" if st is None else
                               "D" if ctl._stage_on(st, "one") else "A" if ctl._stage_on(st, "cl") else "f")
                ctl.step = None
        except Exception as exc:  # noqa: BLE001
            errors.append((r, repr(exc)))
            group.barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(WORLD)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not errors, errors
    for i in range(16):
        assert len({paths[r][i] for r in range(WORLD)}) == 1, (i, [paths[r][i] for r in range(WORLD)])
    path = "".join(paths[0])
    assert path == "D" * 7 + "A" * 4 + "D" * 5, path
    print(f"\nDRAFT1B_AGREE dhd_wish rank2 off calls 6-10: all {WORLD} ranks {path}")


# ---------------------------------------------------------------- J. the frozen K-step interface (draft_body)


def _body_inputs(p, step):
    c, R = step.chunk, step.R
    cap = c.attn_ctx.seq_lens.shape[0]
    target_md = p._target_draft_attn_metadata(c)
    md0 = (c.attn_ctx.seq_lens, c.attn_ctx.query_start_loc, c.attn_ctx.request_distribution)
    bufs = tuple(torch.zeros_like(t) for t in md0)
    import dataclasses

    md = {k: dataclasses.replace(m, seq_lens=bufs[0], query_start_loc=bufs[1], request_distribution=bufs[2])
          for k, m in target_md.items()}
    kw = dict(
        next_tokens=step.nt, num_draft=c.draft_lengths, query_start_loc=md0[1],
        num_reqs=torch.tensor(R, dtype=torch.int64), input_ids=c.input_ids, positions=c.position_ids,
        seq_lens=md0[0], request_distribution=md0[2], target_hidden=p._mtp_target_hidden_states(c),
        loop_query_start_loc=torch.arange(cap + 1, dtype=torch.int32).clamp(max=R),
        loop_request_distribution=torch.tensor([R] * 3, dtype=torch.int32), bufs=bufs,
        p=p._loop_bucket(R), num_rows=cap, max_reqs=h.MAX_REQS,
    )
    return kw, md


@pytest.mark.parametrize("R", [5, 12, 40, 64])
def test_draft_body_is_the_base_paths_drafts_source_and_rejected(monkeypatch, R):
    (b,), _ = _serve(monkeypatch, "base", [41000 + R], R)
    p = _variant(monkeypatch, "one")
    step = h.make_step(41000 + R, R)
    kw, md = _body_inputs(p, step)
    with h.h.fake_forward_context(md):
        drafts, src, rejected = draft_one.draft_body(p._mtp_fused_loop.module, **kw)
    assert drafts.shape == (kw["p"], h.K) and drafts.dtype == torch.int32
    h.same(drafts[:R], b.drafts, f"draft_body drafts R={R}")
    h.same(src, b.src, f"draft_body source R={R}")
    h.same(rejected, b.rejected, f"draft_body rejected R={R}")
    print(f"\nDRAFT1B_KSTEP draft_body R={R}: drafts[{kw['p']}, {h.K}] int32 (rows < R equal the base path's), "
          f"source[{src.numel()}] and rejected[{rejected.numel()}] equal the spec tail's")


def test_draft_body_traces_with_no_host_read(monkeypatch):
    """No .item() / .tolist() / .cpu() / .numpy() and no value-dependent shape: under a dispatch mode that
    refuses aten._local_scalar_dense (every host scalar read goes through it) with the host-copy methods
    patched to raise, and as ONE graph under torch.compile(fullgraph=True) (a host read is a graph break)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    p = _variant(monkeypatch, "one")
    step = h.make_step(42000, 20)
    kw, md = _body_inputs(p, step)
    loop = p._mtp_fused_loop.module

    class NoHostRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func.overloadpacket.__name__ != "_local_scalar_dense", "host scalar read in draft_body"
            return func(*args, **(kwargs or {}))

    def refuse(name):
        def f(*a, **k):
            raise AssertionError(f"Tensor.{name} in draft_body")
        return f

    with h.h.fake_forward_context(md):
        want = draft_one.draft_body(loop, **kw)
        with pytest.MonkeyPatch.context() as m:
            for name in ("item", "tolist", "cpu", "numpy"):
                m.setattr(torch.Tensor, name, refuse(name))
            with NoHostRead():
                got = draft_one.draft_body(loop, **kw)
    for a, b in zip(got, want):
        h.same(a, b, "draft_body under the no-host-read mode")
    compiled = torch.compile(draft_one.draft_body, backend="eager", fullgraph=True, dynamic=False)
    kw2, md2 = _body_inputs(p, h.make_step(42000, 20))
    with h.h.fake_forward_context(md2):
        got2 = compiled(loop, **kw2)
    for a, b in zip(got2, want):
        h.same(a, b, "draft_body compiled as one graph")
    print("\nDRAFT1B_KSTEP draft_body: no host scalar read (aten._local_scalar_dense 0), no .item/.tolist/.cpu/"
          ".numpy, one graph under torch.compile(fullgraph=True); outputs equal")
