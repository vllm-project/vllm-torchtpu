# SPDX-License-Identifier: Apache-2.0
"""q38 L4 v5 composition (q38-l4-v5c), drafter side: DRAFT-ASM + DRAFT1-B DH-D with VFY-ONE value 2's seed taken
from the verify program. CDK CPU pod only. V5COMP-BUILD (helper of L4-E).

Through the real Eagle3Proposer.propose and the real async spec substitution (DRAFT1-B's harness,
tests/q38_draft1b/draft1b_harness.py, over DRAFT-ASM's: every compiled program a counted region). The stash the
runner's `_vfy_verify` leaves is built as the VFY program computes it (spec.fn(next_tokens, *spec.args); the
program's bitwise identity to the D83 chain is tests/l4vfy and tests/q38_l4_v5c/test_v5c_runner.py), as
tests/l4vfy/test_vfy_dha_compose.py does for DRAFT-ASM alone.

D1  8 serving steps at 8 / 32 / 48 / 64 rows, fallback rows off and on: DH-D folds every step through ONE draft
    program with the verify's superset seed taken (0 seed programs of any kind), bitwise the flag-off base path
    (drafts, source, rejected count, index dicts, step 1's positions / seq_lens / first token).
D2  per-step windows over DH-D / stage A / DRAFT-ASM / base with the verify seed on every step: after the warm-up no
    step uses a program key it did not warm (no compile), no step runs a seed program, bitwise the base path.
D3  DRAFT-ASM's agreement entry 0 keeps one bit per piece (fold 1, stage A 2, DH-D 4): a rank without one stage's
    wish turns that stage, and only that stage, off on every rank.
Run: python -m pytest -q -p no:cacheprovider -p no:logging -s tests/q38_l4_v5c/test_v5c_drafter.py
"""

import pathlib
import sys
from collections import Counter
from types import ModuleType

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "q38_draft1b"))
import draft1b_harness as h  # noqa: E402
from vllm_torchtpu.runner import vfy_one  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm, draft_one  # noqa: E402

ROWS = [8, 32, 48, 64]
SEEDS = ("seed", "seed_asm", "seed_cl")
COUNTS: Counter = Counter()


def _inject(p, step):
    """The stash the runner's VFY step (value 2) leaves for this chunk, and the runner's controller."""
    runner = p.runner
    if runner.__dict__.get("_vfy") is None:
        runner._vfy = vfy_one.VfyOne(2, 0, "")
    spec = p.vfy_seed_spec(step.chunk, step.R)
    assert spec is not None
    outs = tuple(spec.fn(step.nt, *spec.args))
    runner._vfy_seed = (spec, step.nt, outs, vfy_one.versions((step.nt, *spec.args, *spec.base_args)))
    return spec


def _same(a, b, tag):
    h.same(a.drafts, b.drafts, f"{tag} drafts")
    h.same(a.src, b.src, f"{tag} source")
    h.same(a.rejected, b.rejected, f"{tag} rejected count")
    assert a.idx == b.idx and a.ndpr == b.ndpr, f"{tag} index dicts"


def _base(monkeypatch, seeds, R, **kw):
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(False)
    h.run_step(bp, h.make_step(seeds[0] - 1, R, **kw), serving=False)
    return [h.run_step(bp, h.make_step(s, R, **kw), serving=True) for s in seeds]


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("R", ROWS)
def test_d1_dhd_with_the_verify_seed_is_bitwise_base_over_8_steps(monkeypatch, R, fallback):
    seeds = [41000 + 100 * R + 10 * fallback + i for i in range(8)]
    kw = {"fallback": fallback}
    base = _base(monkeypatch, seeds, R, **kw)
    h.install(monkeypatch, fold=True, cl=True, one=True)
    p = h.make_proposer(True)
    assert isinstance(p._draft_asm, draft_one.DraftOne)
    h.run_step(p, h.make_step(seeds[0] - 1, R, **kw), serving=False)
    for i, (s_, b) in enumerate(zip(seeds, base)):
        s = h.make_step(s_, R, **kw)
        spec = _inject(p, s)
        assert spec.fn is draft_one._seed_cl_impl, spec.fn
        o = h.run_step(p, s, serving=True)
        assert o.folded and o.calls["one"] == 1, (i, o.calls, dict(p._draft_asm.reasons))
        assert sum(o.calls[k] for k in SEEDS) == 0, (i, o.calls)  # one seed fold per step: the verify's
        assert o.calls["draft_fwd0"] == 0 and o.calls["carry_loop"] == 0 and o.calls["spec_tail"] == 0, o.calls
        _same(o, b, f"DH-D+VFY vs base R={R} step {i}")
    assert p.runner._vfy.stats["seed_taken"] == 8 and p._draft_asm.counts()["folded_one"] == 8
    COUNTS["d1_cases"] += 1
    print(f"\nV5C_DRAFTER D1 R={R} fallback={int(fallback)} steps=8 DH-D folded 8/8 with the verify seed taken "
          f"8/8 (seed programs 0; one program 1 per step); bitwise base")


class _FakeL6:
    def __init__(self, levers, on):
        self.LEVERS = {k: "x" for k in levers}
        self.STEPS = 1
        self.on_fn = on

    def install(self, monkeypatch):
        import vllm_torchtpu.core as core

        mod = ModuleType("vllm_torchtpu.core.l6_ab")
        mod.LEVERS, mod.STEPS = self.LEVERS, self.STEPS
        mod.on = lambda lever: self.on_fn(lever)
        mod.note = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.core.l6_ab", mod)
        monkeypatch.setattr(core, "l6_ab", mod, raising=False)
        monkeypatch.setattr(draft_asm, "_L6_AB", [])


def test_d2_window_switching_with_the_verify_seed_never_compiles(monkeypatch):
    sched = {"draftasmcl": True, "draftasmone": True}
    fake = _FakeL6(["draftasmcl", "draftasmone"], on=lambda lever: sched[lever])
    fake.install(monkeypatch)
    h.install(monkeypatch, fold=True, cl=True, one=True, cl_ab="ab:1", one_ab="ab:1")
    p = h.make_proposer(True)
    fake.install(monkeypatch)
    ctl = p._draft_asm
    assert ctl.cl_conf and ctl.one_conf
    ctl.precompile()
    rs = list(range(1, h.MAX_REQS + 1))
    warm = set()
    for R in rs:
        warm.update(h.run_step(p, h.make_step(47000 + R, R), serving=False).keys)
    warm.update(("seed_cl", k) for k in ctl.warm["seed"])
    warm.update(("carry_fresh_asm", k) for k in ctl.warm["carry"])
    warm.update(("carry_asm", k) for k in ctl.warm["carry"])
    h.install(monkeypatch, fold=False)
    bp = h.make_proposer(False)
    h.run_step(bp, h.make_step(47001, 1), serving=False)
    new, used = set(), Counter()
    for i, R in enumerate(rs):
        kind = ("one", "cl", "dha", "base")[i % 4]
        sched["draftasmone"] = kind == "one"
        sched["draftasmcl"] = kind in ("one", "cl")
        ctl.window_on = kind != "base"
        h.install(monkeypatch, fold=True, cl=True, one=True, cl_ab="ab:1", one_ab="ab:1")
        fake.install(monkeypatch)
        s = h.make_step(48000 + R, R)
        _inject(p, s)
        f = h.run_step(p, s, serving=True)
        h.install(monkeypatch, fold=False)
        b = h.run_step(bp, h.make_step(48000 + R, R), serving=True)
        _same(f, b, f"window step R={R} ({kind})")
        got = "base" if not f.folded else "one" if f.calls["one"] else "cl" if f.calls["carry_loop"] else "dha"
        assert got == kind, (R, kind, got, f.calls, dict(ctl.reasons))
        assert sum(f.calls[k] for k in SEEDS) == 0, (R, kind, f.calls)
        used[got] += 1
        new.update(k for k in f.keys if k not in warm)
    assert not new, sorted(new)[:4]
    c = ctl.counts()
    assert c["cold"] == 0 and c["folded_one"] == used["one"] and c["folded_cl"] == used["cl"]
    assert p.runner._vfy.stats["seed_taken"] == len(rs), dict(p.runner._vfy.stats)
    COUNTS["d2_steps"] = len(rs)
    print(f"\nV5C_DRAFTER D2 warm_keys={len(warm)} new_after_warmup=0 steps={len(rs)} dhd={used['one']} "
          f"stageA={used['cl']} draftasm={used['dha']} base={used['base']} seed programs 0, verify seed taken "
          f"{p.runner._vfy.stats['seed_taken']}")


def test_d3_agreement_bits_one_per_piece(monkeypatch):
    assert (draft_one.BIT_FOLD, draft_one.BIT_CL, draft_one.BIT_ONE) == (1, 2, 4)
    h.install(monkeypatch, fold=True, cl=True, one=True)
    p = h.make_proposer(True)
    ctl = p._draft_asm
    fp = ctl._fingerprint()
    rows = lambda wishes: [[w, 0, 0, 0, fp] for w in wishes]  # noqa: E731  (the documented layout, literal bits)
    cases = [
        ([7, 7, 7, 7], True, True, True),
        ([7, 7, 7, 3], True, True, False),  # one rank without the DH-D wish: DH-D off everywhere, stage A stays
        ([7, 7, 7, 5], True, False, True),  # one rank without the stage-A wish: stage A off, DH-D stays
        ([7, 7, 7, 6], False, False, False),  # one rank without the fold wish: everything off
    ]
    for wishes, fold, cl, one in cases:
        ctl.failed = False
        ctl.check_pending = False
        ctl.apply(rows(wishes))
        assert (ctl.agreed, ctl.cl_agreed, ctl.one_agreed) == (fold, cl, one), (wishes, ctl.agreed, ctl.cl_agreed,
                                                                                ctl.one_agreed)
    COUNTS["d3_cases"] = len(cases)
    print(f"\nV5C_DRAFTER D3 agreement entry 0 bits fold=1 stageA=2 dhd=4; {len(cases)} wish patterns decide each "
          f"piece on its own bit")


def test_zz_counts():
    print("\nV5C_DRAFTER_COUNTS " + " ".join(f"{k}={v}" for k, v in sorted(COUNTS.items())))
