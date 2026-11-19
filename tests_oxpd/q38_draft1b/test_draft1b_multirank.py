# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT1-B all-rank agreement of the stage bits (lead safety rule), CPU only.

Eight simulated TP ranks (one thread each, each with its own proposer and DraftOne controller) step
through the same serving calls; DRAFT-ASM's boundary all_gather meets in a barrier-backed stand-in (on the v5
composition, q38 L4 AGREE's: the driver ticks the runner's engine-step clock once per call and routes
l4_agree.gather to the stand-in, as AGREE ported tests/q38_draftasm/test_draftasm_multirank.py). Each
call's path per rank is base / check / fold, and a fold call's variant is stage A ("A") or DRAFT-ASM
("f") from the agreed stage bit. On every call all ranks must hold the same path and variant, and a
rank-local event may change either only at a boundary, on every rank at once:

  * one rank's stage-A window wish drops between boundaries (the fold stays on);
  * one rank without stage A configured (its window "0"): unequal warm-key sets, base for the run;
  * L6's desync paths on the stage lever "draftasmcl": a rank's lockstep write fails (its `on` drops for
    D steps) and a rank's confirm read fails for one step;
  * a local disable and a CHECK mismatch on one rank only: base for the run.
"""

import sys
import threading
from types import ModuleType, SimpleNamespace

import draft1b_harness as h
from vllm_torchtpu.runner import l4_agree
from vllm_torchtpu.spec_decode import draft_asm, draft_one

WORLD = 8
R = 6


class FakeGroup:
    """All_gather over WORLD threads (DRAFT-ASM's stand-in): each deposits its vector, all read every one."""

    def __init__(self, world):
        self.world = world
        self.barrier = threading.Barrier(world, timeout=60)
        self.slots = [None] * world
        self.calls = 0

    def gather(self, rank, vec):
        self.slots[rank] = list(vec)
        self.barrier.wait()
        out = [list(s) for s in self.slots]
        if rank == 0:
            self.calls += 1
        self.barrier.wait()
        return out


class FakeL6:
    def __init__(self, steps, tl, on_fn):
        self.LEVERS = {"draftasmcl": "VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB"}
        self.STEPS = steps
        self._tl, self._on_fn = tl, on_fn

    def on(self, lever):
        return self._on_fn(self._tl.rank, self._tl.i)


def _run(monkeypatch, steps=16, period=4, check_steps=0, cl_ab=None, l6_on=None, prep=None, inject=None,
         per_rank_ab=None):
    h.install(monkeypatch, fold=True, cl=True, check_steps=check_steps, cl_ab=cl_ab)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", period)
    quiet = SimpleNamespace(info=lambda *a: None, error=lambda *a: None, warning=lambda *a, **k: None)
    monkeypatch.setattr(draft_asm, "logger", quiet)
    monkeypatch.setattr(draft_one, "logger", quiet)
    group = FakeGroup(WORLD)
    tl = threading.local()
    # q38 L4 AGREE port (v5 composition, V5COMP-BUILD; tests only): the boundary gather is l4_agree's.
    monkeypatch.setattr(l4_agree, "gather", lambda vec: group.gather(tl.rank, vec))
    if l6_on is not None:
        import vllm_torchtpu.core as core

        fake = FakeL6(period, tl, l6_on)
        mod = ModuleType("vllm_torchtpu.core.l6_ab")
        mod.LEVERS, mod.STEPS, mod.on, mod.note = fake.LEVERS, fake.STEPS, fake.on, lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.core.l6_ab", mod)
        monkeypatch.setattr(core, "l6_ab", mod, raising=False)
    ranks = []
    for r in range(WORLD):
        if per_rank_ab is not None and r in per_rank_ab:
            monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB", per_rank_ab[r])
        elif cl_ab is not None:
            monkeypatch.setenv("VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB", cl_ab)
        else:
            monkeypatch.delenv("VLLM_TPU_DRAFT_ASM_CARRY_LOOP_AB", raising=False)
        p = h.make_proposer()
        h.run_step(p, h.make_step(500, R), serving=False)  # warm-up (no agreement before serving)
        p._draft_asm.rank = r
        if prep is not None:
            prep(r, p._draft_asm)
        ranks.append(p)
    paths = [[None] * steps for _ in range(WORLD)]
    errors = []

    def worker(r):
        tl.rank = r
        p = ranks[r]
        ctl = p._draft_asm
        p.runner.warmup_done = True
        try:
            for i in range(steps):
                tl.i = i
                if inject is not None:
                    inject(r, i, ctl)
                step = h.make_step(600 + i, R)
                p.runner._l4_agree.tick(p.runner)  # q38 L4 AGREE port: the engine step's clock tick (execute_model)
                ctl.begin([step.chunk], [step.nt], SimpleNamespace(has_structured_output_requests=False), True)
                st = ctl.step
                if st is not None and st.mode == "fold":
                    ctl.seed(step.chunk, step.nt, step.chunk.draft_lengths, R, [])
                st = ctl.step
                if st is None:
                    paths[r][i] = "b"
                elif st.mode == "check":
                    paths[r][i] = "c"
                else:
                    paths[r][i] = "A" if ctl._stage_on(st, "cl") else "f"
                ctl.step = None
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append((r, repr(exc)))
            group.barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(WORLD)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not errors, errors
    for i in range(steps):
        col = {paths[r][i] for r in range(WORLD)}
        assert len(col) == 1, f"call {i + 1}: ranks diverge {[paths[r][i] for r in range(WORLD)]}"
    return "".join(paths[0]), [p._draft_asm for p in ranks], group


def test_stage_wish_changes_only_at_boundaries_on_every_rank(monkeypatch):
    # rank 3's stage-A wish drops on calls 6..10 (1-based); boundaries every 4: 1, 4, 8, 12, 16
    real = draft_one.DraftOne._stage_window

    def inject(r, i, ctl):
        ctl._q38_off = r == 3 and 5 <= i <= 9

    monkeypatch.setattr(draft_one.DraftOne, "_stage_window",
                        lambda self, lever, ab: False if getattr(self, "_q38_off", False) else real(self, lever, ab))
    path, ctls, group = _run(monkeypatch, inject=inject)
    assert path == "A" * 7 + "f" * 4 + "A" * 5, path  # the fold stays; stage A off from boundary 8 to 12
    assert group.calls == 5 and not any(c.failed for c in ctls)
    print(f"\nDRAFT1B_AGREE stage_wish rank3 off calls 6-10: all {WORLD} ranks {path} (boundaries {group.calls})")


def test_a_rank_without_stage_a_keeps_every_rank_on_base(monkeypatch):
    path, ctls, _ = _run(monkeypatch, per_rank_ab={6: "0"})
    assert not ctls[6].cl_conf and all(c.cl_conf for i, c in enumerate(ctls) if i != 6)
    assert path == "b" * 16, path
    assert all(c.failed for c in ctls)
    print(f"\nDRAFT1B_AGREE config_mismatch rank6 stage window 0: all {WORLD} ranks {path}")


def test_l6_lockstep_write_failure_on_the_stage_lever(monkeypatch):
    def on(rank, i):
        return not (rank == 4 and 5 <= i < 5 + 16)

    path, ctls, _ = _run(monkeypatch, steps=24, cl_ab="ab:1", l6_on=on)
    assert path == "A" * 7 + "f" * 16 + "A", path  # boundaries 8..20 see rank 4 off, 24 sees it back
    print(f"\nDRAFT1B_AGREE l6_write_fail draftasmcl rank4: all {WORLD} ranks {path}")


def test_l6_confirm_read_failure_on_the_stage_lever(monkeypatch):
    def on(rank, i):
        return not (rank == 1 and i in (5, 7))

    path, ctls, _ = _run(monkeypatch, cl_ab="ab:1", l6_on=on)
    assert path == "A" * 7 + "f" * 4 + "A" * 5, path  # call 6 (no boundary) ignored, call 8 (boundary) agreed
    print(f"\nDRAFT1B_AGREE l6_read_fail draftasmcl rank1: all {WORLD} ranks {path}")


def test_local_disable_on_one_rank_moves_every_rank_to_base(monkeypatch):
    def inject(r, i, ctl):
        if r == 5 and i == 5:
            ctl.disable_locally("injected stage A anomaly")

    path, ctls, _ = _run(monkeypatch, inject=inject)
    assert path == "A" * 7 + "b" * 9, path
    assert all(c.failed and not c.cl_agreed for c in ctls)
    print(f"\nDRAFT1B_AGREE local_disable rank5 call 6: all {WORLD} ranks {path}")


def test_a_check_mismatch_on_one_rank_keeps_every_rank_on_base(monkeypatch):
    def inject(r, i, ctl):
        if r == 2 and i == 1:
            ctl.check_mismatches += 1

    path, ctls, _ = _run(monkeypatch, check_steps=3, inject=inject)
    assert path == "c" * 3 + "b" * 13, path
    assert all(c.failed for c in ctls)
    print(f"\nDRAFT1B_AGREE check_mismatch rank2: all {WORLD} ranks {path}")
