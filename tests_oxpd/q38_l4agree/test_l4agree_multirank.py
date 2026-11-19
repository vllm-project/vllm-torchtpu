# SPDX-License-Identifier: Apache-2.0
"""q38 L4 AGREE (runner/l4_agree.py): DRAFT-ASM and VFY-ONE on one engine-step clock with one gather, 8 TP ranks, CPU
only (CDK CPU pod). AGREE-BUILD (helper of L4-E).

Each simulated rank is one thread with its own runner, its own real DraftAsm controller (registered by DraftAsm's
__init__, built through DRAFT-ASM's harness) and its own real VfyOne controller (registered on the same runner as
tpu_runner registers it), so one Agreement per rank carries both levers. Per engine step every rank runs
execute_model's two hooks (VfyOne.begin_step, then Agreement.tick) and then the step's K calls of each lever's per-call
entry point (VfyOne.decide and DraftAsm.begin, plus the folded seed on fold steps), as the K inner steps of a K-step
burst would. Gathers meet in a barrier stand-in for the TP group's CPU all_gather; core/l6_ab.py is a stand-in whose
`on` is per rank and whose window class follows the engine step (or, for the wall-clock case, each rank's own clock).

Checked on every case: every rank records the same choice on every call; each lever's agreed state changes only on a
boundary step; the boundary steps are the same on every rank and equal the stated schedule; and every rank makes
exactly one gather per boundary for both levers together (each lever's boundary count = the Agreement's).

  1. one gather per boundary for both levers together (DRAFT-ASM every 4, VFY-ONE every 6 engine steps);
  2. K-step bursts (K varies 1..4 per engine step, so propose / decide calls != engine steps): the boundaries stay on
     engine steps, and a local disable made between two inner calls takes effect at the next boundary on every rank;
  3. L6's two desync paths through core/l6_ab.py, for each lever (the other lever uniform): (a) a rank's lockstep write
     fails at k*S - D, so its `on` drops at once and every other rank's at k*S - 1; (b) a rank's confirm read fails
     for one step, once on a boundary and once between boundaries;
  4. a rank-local disable, unequal warm keys and a CHECK mismatch on one rank, for each lever: only that lever leaves
     its fast path, on every rank, at the next boundary (the CHECK sum only at a boundary); the other lever stays on;
  5. wall-clock l6_ab windows whose class rolls on different steps on different ranks never make a boundary;
  6. unequal lever sets (one rank without VFY-ONE): every rank raises at the first boundary (fail-stop, no hang);
  7. the registration hook: a DraftAsm subclass that packs a stage bit into DRAFT-ASM's vector (DraftOne's pattern)
     joins with no change, and a third lever (VRNG's shape) rides the same gather.
"""

import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "q38_draftasm"))
import draftasm_harness as h  # noqa: E402
from vllm_torchtpu.runner import l4_agree, vfy_one  # noqa: E402
from vllm_torchtpu.spec_decode import draft_asm  # noqa: E402

WORLD = 8
R = 6
TIMEOUT = 30
QUIET = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None, warning=lambda *a, **k: None)
COUNTS: dict = {}


class FakeGroup:
    """All_gather over WORLD threads: each deposits its vector, all read every vector; calls counted per rank."""

    def __init__(self, world):
        self.barrier = threading.Barrier(world, timeout=TIMEOUT)
        self.slots = [None] * world
        self.calls = [0] * world

    def gather(self, rank, vec):
        self.slots[rank] = list(vec)
        self.calls[rank] += 1
        self.barrier.wait()
        out = [list(s) for s in self.slots]
        try:
            # Only guards slot reuse by the next gather: every rank has read its rows here. A rank that raised
            # after this gather and aborted the barrier must not turn a released sibling's return into an error.
            self.barrier.wait()
        except threading.BrokenBarrierError:
            pass
        return out


class FakeL6:
    """core/l6_ab.py stand-in: `on` per (lever, rank, engine step); window class per (rank, engine step)."""

    def __init__(self, tl, steps, on_fn=None, cls_fn=None, levers=("draftasm", "vfyone")):
        self.LEVERS = {name: "X" for name in levers}
        self.STEPS = steps
        self.ACTIVE = False
        self._tl, self._on, self._cls = tl, on_fn, cls_fn

    def on(self, lever):
        return bool(self._on(lever, getattr(self._tl, "rank", 0), getattr(self._tl, "step", 0))) if self._on else False

    def current_class(self):
        return self._cls(getattr(self._tl, "rank", 0), getattr(self._tl, "step", 0)) if self._cls else None

    def note(self, *a, **k):
        pass

    def check_allowed(self):
        return True


def _vps_runner(runner):
    """The runner attributes VfyOne.static / decide read (a VPS fallback runner with D83's fold)."""
    runner._vps_head, runner._vps_fb_path, runner._vps_ro_fold = object(), True, True
    runner.mamba_slot_read_offsets, runner._vps_checks_left = object(), 0


def _setup(monkeypatch, *, da_ab=None, da_check=0, vf_ab="", vf_check=0, da_every=4, vf_every=6, fake=None,
           prep=None, skip_vfy=(), proposer=None):
    """WORLD ranks, each: runner + DraftAsm (registered by its __init__) + VfyOne (registered as tpu_runner does)."""
    h.install(monkeypatch, fold=True, check_steps=da_check, ab=da_ab)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", da_every)
    monkeypatch.setattr(vfy_one, "AGREE_EVERY", vf_every)
    for mod in (draft_asm, vfy_one, l4_agree):
        monkeypatch.setattr(mod, "logger", QUIET)
    monkeypatch.setattr(draft_asm, "_L6_AB", [fake])
    monkeypatch.setattr(vfy_one, "_L6_AB", [fake])
    group = FakeGroup(WORLD)
    ranks = []
    for r in range(WORLD):
        p = proposer() if proposer is not None else h.make_proposer(True)
        _vps_runner(p.runner)
        h.run_step(p, h.make_step(500, R), serving=False)  # warm-up: the clock does not count it
        p._draft_asm.rank = r
        agree = p.runner._l4_agree
        assert agree.step == 0 and [s.name for s in agree.slots] == ["draftasm"]
        v = vfy_one.VfyOne(1, vf_check, vf_ab)
        v.warm, v.rank = {("k", 1)}, r
        if r not in skip_vfy:
            l4_agree.register(p.runner, v)
        agree._l6[:] = [fake]
        agree.transport = lambda vec, r=r: group.gather(r, vec)
        if prep is not None:
            prep(r, p._draft_asm, v)
        ranks.append((p, v))
    return ranks, group


def _run(ranks, group, steps, tl, burst=None, inject=None, inner=None):
    """`steps` engine steps on every rank; per step execute_model's hooks, then burst[step] calls of each lever's
    per-call entry point. Returns per rank: calls [(step, j, vfy_on, vfy_mode, draft_mode)], states
    [(draft_agreed, draft_failed, vfy_engaged, vfy_failed)] per step, and the boundary steps; and the errors."""
    world = len(ranks)
    step_barrier = threading.Barrier(world, timeout=TIMEOUT)
    calls = [[] for _ in range(world)]
    states = [[] for _ in range(world)]
    bounds = [[] for _ in range(world)]
    errors = []
    so = SimpleNamespace(has_structured_output_requests=False)

    def worker(r):
        tl.rank = r
        p, v = ranks[r]
        ctl, runner = p._draft_asm, p.runner
        runner.warmup_done = True
        try:
            for step in range(1, steps + 1):
                tl.step = step
                if inject is not None:
                    inject(r, step, None, ctl, v)
                v.begin_step(runner)  # execute_model hook 1 (VFY-ONE's step count)
                if runner._l4_agree.tick(runner):  # execute_model hook 2 (the engine-step clock)
                    bounds[r].append(step)
                states[r].append((ctl.agreed, ctl.failed, v.engaged, v.failed))
                for j in range((burst or {}).get(step, 1)):
                    if inner is not None:
                        inner(r, step, j, ctl, v)
                    von = v.decide(runner, True)
                    s = h.make_step(600 + step, R)
                    ctl.begin([s.chunk], [s.nt], so, True)
                    if ctl.step is not None and ctl.step.mode == "fold":
                        ctl.seed(s.chunk, s.nt, s.chunk.draft_lengths, R, [])
                    mode = "base" if ctl.step is None else ctl.step.mode
                    ctl.step = None
                    calls[r].append((step, j, von, v.step_mode, mode))
                step_barrier.wait()
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            errors.append((r, type(exc).__name__, str(exc)[:300]))
            step_barrier.abort()
            group.barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10 * TIMEOUT)
    assert not any(t.is_alive() for t in threads), "a rank hung"
    return calls, states, bounds, errors


def _check(ranks, group, calls, states, bounds, errors, want_bounds):
    """The checks every case shares (see the module docstring)."""
    assert not errors, errors
    world = len(ranks)
    for i in range(len(calls[0])):
        row = {calls[r][i] for r in range(world)}
        assert len(row) == 1, ("ranks diverge on call", calls[0][i][:2], [calls[r][i] for r in range(world)])
    assert all(b == bounds[0] for b in bounds), bounds
    assert bounds[0] == want_bounds, (bounds[0], want_bounds)
    for r in range(world):
        assert states[r] == states[0], ("agreed states diverge", r)
    for k in range(1, len(states[0])):
        if states[0][k] != states[0][k - 1]:
            assert (k + 1) in bounds[0], ("an agreed state changed off a boundary at engine step", k + 1)
    for r, (p, v) in enumerate(ranks):
        agree = p.runner._l4_agree
        nb = len(bounds[r])
        assert group.calls[r] == agree.gathers == agree.boundaries == nb, (r, group.calls[r], agree.counts(), nb)
        assert p._draft_asm.boundaries == nb, (r, p._draft_asm.boundaries, nb)
        if v.agreement is agree:
            assert v.boundaries == nb, (r, v.boundaries, nb)
    COUNTS["lockstep_cases"] = COUNTS.get("lockstep_cases", 0) + 1


def _schedule(steps, periods, rolls=()):
    """The boundary steps the module docstring states: step 1, multiples of each period, the given roll steps."""
    return [s for s in range(1, steps + 1) if s == 1 or any(p and s % p == 0 for p in periods) or s in rolls]


def _per_step(states, k):
    return [st[k] for st in states[0]]


def _model(bounds, steps, wish):
    """Reference: the agreed state = MIN over ranks of wish(rank, s) at the last boundary s <= step."""
    out, state = [], False
    for s in range(1, steps + 1):
        if s in bounds:
            state = all(wish(r, s) for r in range(WORLD))
        out.append(state)
    return out


# ---------------------------------------------------------------- 1. one gather per boundary


def test_one_gather_per_boundary_for_both_levers(monkeypatch):
    tl = threading.local()
    ranks, group = _setup(monkeypatch)
    calls, states, bounds, errors = _run(ranks, group, 24, tl)
    want = _schedule(24, (4, 6))
    _check(ranks, group, calls, states, bounds, errors, want)
    assert want == [1, 4, 6, 8, 12, 16, 18, 20, 24]
    assert all(c[2] and c[3] == "vfy" and c[4] == "fold" for c in calls[0]), calls[0][:3]
    assert ranks[0][0].runner._l4_agree.counts()["levers"] == ["draftasm", "vfyone"]
    COUNTS["gathers_per_rank"] = group.calls[0]
    print(f"\nQ38_L4AGREE one_gather: {WORLD} ranks x {len(want)} boundaries {want}: gathers per rank {group.calls}")


# ---------------------------------------------------------------- 2. K-step bursts: calls != engine steps

BURST = {s: [1, 2, 4, 3, 1, 4, 2, 2][s % 8] for s in range(1, 25)}


def test_kstep_bursts_keep_the_boundaries_on_engine_steps(monkeypatch):
    tl = threading.local()
    ranks, group = _setup(monkeypatch)

    def inner(r, step, j, ctl, v):
        # rank 5 disables DRAFT-ASM between inner calls 1 and 2 of engine step 9 (a K = 2 burst)
        if r == 5 and step == 9 and j == 1:
            ctl.disable_locally("injected inside a burst")

    calls, states, bounds, errors = _run(ranks, group, 24, tl, burst=BURST, inner=inner)
    want = _schedule(24, (4, 6))
    _check(ranks, group, calls, states, bounds, errors, want)
    n_calls = sum(BURST.values())
    p0, v0 = ranks[0]
    assert p0.runner._l4_agree.step == 24 and p0._draft_asm.serving_calls == n_calls == len(calls[0]) != 24
    assert v0.serving == 24
    # every call of one engine step makes the same choice (the agreement of the burst's engine step)
    for step in range(1, 25):
        assert len({c[2:] for c in calls[0] if c[0] == step}) == 1, step
    # DRAFT-ASM: fold through engine step 11, base from boundary 12 on (the disable at step 9 waited for it)
    draft = {c[0]: c[4] for c in calls[0]}
    assert [draft[s] for s in range(1, 25)] == ["fold"] * 11 + ["base"] * 13, draft
    assert _per_step(states, 1) == [False] * 11 + [True] * 13
    assert all(c[3] == "vfy" for c in calls[0])  # VFY-ONE unaffected
    COUNTS["kstep_calls"] = n_calls
    print(f"\nQ38_L4AGREE kstep: 24 engine steps, {n_calls} propose / decide calls per rank, boundaries {bounds[0]} "
          f"(engine steps), gathers per rank {group.calls[0]}")


# ---------------------------------------------------------------- 3. L6's two desync paths, each lever

S = 6  # l6_ab step window; class = (step // S) % 2, ab:1 -> on in odd windows; window 0 never confirmed


def _cls(rank, step):
    return str((step // S) % 2)


def _base_on(step):
    return (step // S) % 2 == 1 and step // S >= 1


def _windowed(lever, monkeypatch, tl, on):
    """The lever under test on ab:1 step windows (with `on`), the other uniform on."""
    fake = FakeL6(tl, S, on_fn=lambda lv, r, s: on(r, s) if lv == lever else True, cls_fn=_cls)
    kw = dict(fake=fake, da_every=4, vf_every=5)
    if lever == "draftasm":
        kw["da_ab"] = "ab:1"
    else:
        kw["vf_ab"] = "ab:1"
    ranks, group = _setup(monkeypatch, **kw)
    return ranks, group


def _sched_for(lever, steps):
    # DRAFT-ASM windowed: period S plus rolls (= multiples of S here); VFY-ONE windowed: rolls only (multiples of S).
    # The uniform lever keeps its own period (VFY-ONE 5, DRAFT-ASM 4).
    other = 5 if lever == "draftasm" else 4
    return _schedule(steps, (S, other))


@pytest.mark.parametrize("lever", ["draftasm", "vfyone"])
def test_l6_write_failure_on_one_rank_never_splits_the_ranks(monkeypatch, lever):
    # path (a), k = 2: rank 4's write for window 2 fails at 2S - D = 9 (D = 3): its `on` drops at once; every other
    # rank finds the entry missing at 2S - 1 = 11 and drops too; all stay off for the rest of the run.
    def on(r, s):
        return _base_on(s) and s < (9 if r == 4 else 11)

    tl = threading.local()
    ranks, group = _windowed(lever, monkeypatch, tl, on)
    calls, states, bounds, errors = _run(ranks, group, 30, tl)
    want = _sched_for(lever, 30)
    _check(ranks, group, calls, states, bounds, errors, want)
    k = 0 if lever == "draftasm" else 2
    got = _per_step(states, k)
    assert got == _model(want, 30, on), (got, want)
    # on in window 1 from its roll (6) to the first boundary inside the desync (rank 4 off from 9), then off
    first_off = min(b for b in want if b >= 9)
    assert got == [False] * 5 + [True] * (first_off - 6) + [False] * (31 - first_off), (got, first_off)
    assert not any(_per_step(states, k + 1)), "a wish drop is not a failure"
    other = 2 if lever == "draftasm" else 0
    assert all(_per_step(states, other)), "the other lever stays on"
    COUNTS[f"l6_write_{lever}"] = first_off
    print(f"\nQ38_L4AGREE l6_write_fail {lever}: rank 4 off at 9, others at 11; all {WORLD} ranks off from boundary "
          f"{first_off}; boundaries {want}")


@pytest.mark.parametrize("lever", ["draftasm", "vfyone"])
def test_l6_one_step_read_failure_on_one_rank_never_splits_the_ranks(monkeypatch, lever):
    # path (b): rank 1's confirm read fails for one step, in an ON window: at 21 (no boundary of either lever) and at 30
    # (the roll into window 5, a boundary of both)
    def on(r, s):
        return _base_on(s) and not (r == 1 and s in (21, 30))

    tl = threading.local()
    ranks, group = _windowed(lever, monkeypatch, tl, on)
    calls, states, bounds, errors = _run(ranks, group, 36, tl)
    want = _sched_for(lever, 36)
    assert 21 not in want and 30 in want
    _check(ranks, group, calls, states, bounds, errors, want)
    k = 0 if lever == "draftasm" else 2
    got = _per_step(states, k)
    assert got == _model(want, 36, on), got
    # window 3 (18..23) on everywhere despite the dip at 21; window 4 (24..29) off; window 5 starts off on every rank
    # (rank 1's dip on its boundary), then comes back on at its next boundary (35 / 32) on every rank
    nxt = min(b for b in want if b > 30)
    assert got[17:23] == [True] * 6 and not any(got[23:30]), got
    assert got[30:35] == [s >= nxt for s in range(31, 36)], (got[30:35], nxt)
    assert not any(_per_step(states, k + 1)), "a wish drop is not a failure"
    COUNTS[f"l6_read_{lever}"] = nxt
    print(f"\nQ38_L4AGREE l6_read_fail {lever}: dip at 21 ignored; dip at boundary 30 holds every rank off to {nxt}; "
          f"lockstep on {WORLD} ranks")


# ---------------------------------------------------------------- 4. rank-local events, each lever


@pytest.mark.parametrize("lever", ["draftasm", "vfyone"])
def test_a_local_disable_on_one_rank_moves_only_its_lever_at_the_next_boundary(monkeypatch, lever):
    tl = threading.local()
    ranks, group = _setup(monkeypatch)

    def inject(r, step, _, ctl, v):
        if r == 5 and step == 9:
            (ctl if lever == "draftasm" else v).disable_locally("injected")

    calls, states, bounds, errors = _run(ranks, group, 24, tl, inject=inject)
    want = _schedule(24, (4, 6))
    _check(ranks, group, calls, states, bounds, errors, want)
    k, other = (0, 2) if lever == "draftasm" else (2, 0)
    assert _per_step(states, k) == [True] * 11 + [False] * 13  # next boundary after 9 is 12
    assert _per_step(states, k + 1) == [False] * 11 + [True] * 13  # failed: base for the rest of the run
    assert all(_per_step(states, other)) and not any(_per_step(states, other + 1))


@pytest.mark.parametrize("lever", ["draftasm", "vfyone"])
def test_unequal_warm_keys_on_one_rank_keep_only_its_lever_on_base(monkeypatch, lever):
    def prep(r, ctl, v):
        if r == 6:
            if lever == "draftasm":
                ctl.warm["seed"].clear()
            else:
                v.warm = {("k", 2)}

    tl = threading.local()
    ranks, group = _setup(monkeypatch, prep=prep)
    calls, states, bounds, errors = _run(ranks, group, 12, tl)
    _check(ranks, group, calls, states, bounds, errors, _schedule(12, (4, 6)))
    k, other = (0, 2) if lever == "draftasm" else (2, 0)
    assert not any(_per_step(states, k)) and all(_per_step(states, k + 1))
    assert all(_per_step(states, other))


@pytest.mark.parametrize("lever", ["draftasm", "vfyone"])
def test_a_check_mismatch_on_one_rank_is_summed_at_the_boundary(monkeypatch, lever):
    tl = threading.local()
    kw = dict(da_check=3) if lever == "draftasm" else dict(vf_check=3)
    ranks, group = _setup(monkeypatch, **kw)

    def inject(r, step, _, ctl, v):
        if r == 2 and step == 2:
            (ctl if lever == "draftasm" else v).check_mismatches += 1  # what each lever's CHECK counts

    calls, states, bounds, errors = _run(ranks, group, 12, tl, inject=inject)
    _check(ranks, group, calls, states, bounds, errors, _schedule(12, (4, 6)))
    if lever == "draftasm":
        modes = [c[4] for c in calls[0]]
        assert modes == ["check"] * 3 + ["base"] * 9, modes
        assert _per_step(states, 1) == [False] * 3 + [True] * 9  # the sum fails it at boundary 4, not before
        assert all(c[3] == "vfy" for c in calls[0])
    else:
        modes = [c[3] for c in calls[0]]
        assert modes == ["check"] * 3 + [None] * 9, modes
        assert _per_step(states, 3) == [False] * 3 + [True] * 9
        assert all(c[4] == "fold" for c in calls[0])


# ---------------------------------------------------------------- 5. wall-clock windows never make a boundary


def test_wall_clock_window_rolls_never_make_a_boundary(monkeypatch):
    """l6_ab on wall-clock windows (STEPS = 0): each rank's class rolls on its own step (here rank r rolls 2r steps
    late). The old VFY-ONE rule (a boundary on any class change) would send some ranks into the gather while others
    run the step. Both levers windowed: no roll boundary; the boundaries are step 1 and the periods only."""
    tl = threading.local()
    fake = FakeL6(tl, 0, on_fn=lambda lv, r, s: True, cls_fn=lambda r, s: str(((s + 2 * r) // 5) % 2))
    ranks, group = _setup(monkeypatch, fake=fake, da_ab="ab:1", vf_ab="ab:1", da_every=7, vf_every=6)
    calls, states, bounds, errors = _run(ranks, group, 20, tl)
    # DRAFT-ASM ab: without step windows has the uniform period (7) and refuses its window; VFY-ONE ab: has no period
    _check(ranks, group, calls, states, bounds, errors, _schedule(20, (7,)))
    assert ranks[0][0].runner._l4_agree.roll is True  # VFY-ONE asks for rolls; wall-clock windows give none
    assert not any(_per_step(states, 0)) and not any(_per_step(states, 2))  # both refuse ab: without step windows
    COUNTS["wallclock_bounds"] = len(bounds[0])


# ---------------------------------------------------------------- 6. unequal lever sets: fail-stop


def test_unequal_lever_sets_fail_stop_on_every_rank(monkeypatch):
    tl = threading.local()
    ranks, group = _setup(monkeypatch, skip_vfy=(3,))
    calls, states, bounds, errors = _run(ranks, group, 4, tl)
    assert sorted(r for r, *_ in errors) == list(range(WORLD)), errors
    assert all(kind == "RuntimeError" and "headers differ" in msg for _, kind, msg in errors), errors
    assert group.calls == [1] * WORLD  # every rank reached the one gather (equal sizes: no hang), then stopped
    COUNTS["failstop_ranks"] = len(errors)


# ---------------------------------------------------------------- 7. the registration hook


BIT_STAGE = 4


class StageAsm(draft_asm.DraftAsm):
    """DraftOne's pattern (q38-l4-draft1b): a DraftAsm subclass that packs a stage wish into entry 0 of DRAFT-ASM's
    vector and decides the stage on it. It has no agreement code of its own: DraftAsm.__init__ registers it."""

    stage_wish = True
    stage_agreed = False

    def local_vector(self):
        vec = super().local_vector()
        vec[0] = int(vec[0]) | (BIT_STAGE if self.stage_wish else 0)
        return vec

    def apply(self, rows):
        super().apply([[int(r[0]) & 1, *r[1:]] for r in rows])
        self.stage_agreed = self.agreed and all(int(r[0]) & BIT_STAGE for r in rows)


class VrngShape:
    """VRNG-VPS's shape (q38-l4-samp2): wish, disable, warm-key fingerprint, counter word."""

    AGREE_NAME, AGREE_WIDTH = "vrng", 4

    def __init__(self, counter):
        self.counter, self.agreed, self.failed, self.boundaries, self.agreement = counter, False, False, 0, None

    def local_vector(self):
        return [1, 0, 77, self.counter]

    def apply(self, rows):
        self.boundaries += 1
        if len({r[3] for r in rows}) != 1:
            self.failed = True
        self.agreed = not self.failed and all(r[0] for r in rows)

    def agree_period(self):
        return 10


def test_a_subclass_and_a_third_lever_join_through_the_hook(monkeypatch):
    tl = threading.local()
    vrngs = []

    def proposer():
        p = h.make_proposer(False)
        p._draft_asm = StageAsm(p)  # registered by DraftAsm.__init__
        return p

    def prep(r, ctl, v):
        g = VrngShape(counter=5 if r != 7 else 6)  # rank 7's counter differs: only VRNG fails
        l4_agree.register(ctl.p.runner, g)
        vrngs.append(g)

    def inject(r, step, _, ctl, v):
        if r == 2 and step == 7:
            ctl.stage_wish = False  # rank 2 drops the stage wish mid-run

    ranks, group = _setup(monkeypatch, proposer=proposer, prep=prep)
    calls, states, bounds, errors = _run(ranks, group, 24, tl, inject=inject)
    want = _schedule(24, (4, 6, 10))
    _check(ranks, group, calls, states, bounds, errors, want)
    agree = ranks[0][0].runner._l4_agree
    assert agree.counts()["levers"] == ["draftasm", "vfyone", "vrng"] and type(ranks[0][0]._draft_asm) is StageAsm
    assert all(c.stage_agreed is False for c in (p._draft_asm for p, _ in ranks))  # off from boundary 8 on
    assert all(p._draft_asm.agreed for p, _ in ranks) and all(v.engaged for _, v in ranks)
    assert all(g.failed and not g.agreed and g.boundaries == len(want) for g in vrngs)
    assert all(g.agreement is p.runner._l4_agree for g, (p, _) in zip(vrngs, ranks))
    COUNTS["hook_levers"] = 3


def test_flag_off_builds_no_agreement(monkeypatch):
    h.install(monkeypatch, fold=False)
    p = h.make_proposer(False)
    assert "_l4_agree" not in p.runner.__dict__ and getattr(p, "_draft_asm", None) is None
    h.run_step(p, h.make_step(950, R), serving=True)  # the harness tick finds no agreement: nothing runs
    assert "_l4_agree" not in p.runner.__dict__


def test_zz_counts():
    out = os.environ.get("Q38_L4AGREE_MR_COUNTS")
    if out:
        with open(out, "w") as f:
            json.dump(COUNTS, f, sort_keys=True)
    print("Q38_L4AGREE_MR_COUNTS " + json.dumps(COUNTS, sort_keys=True))
