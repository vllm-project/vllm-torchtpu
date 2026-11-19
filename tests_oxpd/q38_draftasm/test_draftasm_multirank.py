# SPDX-License-Identifier: Apache-2.0
"""q38 L4 DRAFT-ASM all-rank agreement (lead safety rule), CPU only.

Eight simulated TP ranks, one thread each, each with its own proposer and DraftAsm controller, step through
the same serving calls; their boundary agreements meet in a barrier-backed stand-in for the TP group's CPU
all_gather. One rank gets a rank-local event; on every call all eight ranks must pick the same path (fold,
check or base), and the event may change the path only at a boundary, on every rank at once:

  * a rank's window wish flips between boundaries, and at a boundary;
  * a rank disables locally (a fold-path anomaly);
  * a rank's warm keys differ (a key cold on that rank only);
  * a CHECK mismatch on one rank only;
  * L6's desync paths through core/l6_ab.py: a rank's lockstep write fails (its own `on` drops for D steps)
    and a rank's confirm read fails for one step.

Plus fail-stop: a gather that fails raises, and so does a folded-loop build failure.

q38 L4 AGREE (q38-l4-agree): the boundaries are the runner's engine-step clock's (runner/l4_agree.py) and the gather
is its one all_gather; each simulated serving call here is one engine step, so the driver ticks the runner's clock
once before each call (execute_model's hook) and routes l4_agree.gather to the barrier stand-in. Assertions unchanged.
"""

import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch

import draftasm_harness as h
from vllm_torchtpu.runner import l4_agree
from vllm_torchtpu.spec_decode import draft_asm

WORLD = 8
R = 6


class FakeGroup:
    """All_gather over WORLD threads: each deposits its vector, all read every vector."""

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
    """core/l6_ab.py with a per-rank (thread-local) `on`."""

    def __init__(self, steps, tl, on_fn):
        self.LEVERS = {"draftasm": "VLLM_TPU_DRAFT_ASM_AB"}
        self.STEPS = steps
        self._tl, self._on_fn = tl, on_fn

    def on(self, lever):
        return self._on_fn(self._tl.rank, self._tl.i)

    def note(self, lever, key, n=1):
        pass


def _run(monkeypatch, steps=16, period=4, check_steps=0, ab=None, l6_on=None, prep=None, inject=None):
    """Drive WORLD ranks through `steps` serving calls; returns modes[rank][i] in fold / check / base."""
    h.install(monkeypatch, fold=True, check_steps=check_steps, ab=ab)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", period)
    monkeypatch.setattr(draft_asm, "logger", SimpleNamespace(info=lambda *a: None, error=lambda *a: None,
                                                           warning=lambda *a, **k: None))
    group = FakeGroup(WORLD)
    tl = threading.local()
    monkeypatch.setattr(l4_agree, "gather", lambda vec: group.gather(tl.rank, vec))
    if l6_on is not None:
        import sys

        import vllm_torchtpu.core as core

        fake = FakeL6(period, tl, l6_on)
        mod = ModuleType("vllm_torchtpu.core.l6_ab")
        mod.LEVERS, mod.STEPS, mod.on, mod.note = fake.LEVERS, fake.STEPS, fake.on, fake.note
        monkeypatch.setitem(sys.modules, "vllm_torchtpu.core.l6_ab", mod)
        monkeypatch.setattr(core, "l6_ab", mod, raising=False)
    ranks = []
    for r in range(WORLD):
        p = h.make_proposer(True)
        h.run_step(p, h.make_step(500, R), serving=False)  # warm-up (no agreement before serving)
        p._draft_asm.rank = r
        if prep is not None:
            prep(r, p._draft_asm)
        ranks.append(p)
    modes = [[None] * steps for _ in range(WORLD)]
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
                p.runner._l4_agree.tick(p.runner)  # the engine step's clock tick (execute_model)
                ctl.begin([step.chunk], [step.nt], SimpleNamespace(has_structured_output_requests=False), True)
                st = ctl.step
                if st is not None and st.mode == "fold":
                    ctl.seed(step.chunk, step.nt, step.chunk.draft_lengths, R, [])
                st = ctl.step
                modes[r][i] = "base" if st is None else st.mode
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
        col = {modes[r][i] for r in range(WORLD)}
        assert len(col) == 1, f"call {i + 1}: ranks diverge {[modes[r][i] for r in range(WORLD)]}"
    return modes[0], [p._draft_asm for p in ranks], group


def _boundary_index(i, period):
    return i == 0 or (i + 1) % period == 0


def test_window_wish_changes_only_at_boundaries_on_every_rank(monkeypatch):
    # rank 3's wish drops on calls 6..10 (1-based); boundaries every 4 calls: 1, 4, 8, 12, 16
    def inject(r, i, ctl):
        ctl.window_on = not (r == 3 and 5 <= i <= 9)

    path, ctls, group = _run(monkeypatch, inject=inject)
    want = ["fold"] * 7 + ["base"] * 4 + ["fold"] * 5  # off from boundary 8 to boundary 12
    assert path == want, path
    assert group.calls == 5
    print(f"\nDRAFTASM_AGREE window rank3 off calls 6-10: all {WORLD} ranks {''.join(m[0] for m in path)} "
          f"(boundaries {group.calls})")


def test_local_disable_on_one_rank_moves_every_rank_at_the_next_boundary(monkeypatch):
    def inject(r, i, ctl):
        if r == 5 and i == 5:
            ctl.disable_locally("injected fold-path anomaly")

    path, ctls, _ = _run(monkeypatch, inject=inject)
    assert path == ["fold"] * 7 + ["base"] * 9, path
    assert all(c.failed for c in ctls)
    print(f"\nDRAFTASM_AGREE local_disable rank5 call 6: all {WORLD} ranks {''.join(m[0] for m in path)}")


def test_a_key_cold_on_one_rank_only_keeps_every_rank_on_base(monkeypatch):
    def prep(r, ctl):
        if r == 6:
            ctl.warm["seed"].clear()

    path, ctls, _ = _run(monkeypatch, prep=prep)
    assert path == ["base"] * 16, path
    assert all(c.failed for c in ctls)
    print(f"\nDRAFTASM_AGREE cold_one_rank rank6: all {WORLD} ranks {''.join(m[0] for m in path)}")


def test_a_check_mismatch_on_one_rank_keeps_every_rank_on_base(monkeypatch):
    def inject(r, i, ctl):
        if r == 2 and i == 1:
            ctl.check_mismatches += 1  # what _mismatch counts

    path, ctls, _ = _run(monkeypatch, check_steps=3, inject=inject)
    assert path == ["check"] * 3 + ["base"] * 13, path
    assert all(c.failed for c in ctls)
    print(f"\nDRAFTASM_AGREE check_mismatch rank2: all {WORLD} ranks {''.join(m[0] for m in path)}")


def test_l6_lockstep_write_failure_on_one_rank_does_not_split_the_ranks(monkeypatch):
    # L6 path 1: rank 4 fails its fingerprint write at step k*S - D; its own `on` drops at once for D steps
    def on(rank, i):
        return not (rank == 4 and 5 <= i < 5 + 16)

    path, ctls, _ = _run(monkeypatch, steps=24, ab="ab:1", l6_on=on)
    want = ["fold"] * 7 + ["base"] * 16 + ["fold"]  # boundaries 8..20 see rank 4 off, 24 sees it back
    assert path == want, path
    print(f"\nDRAFTASM_AGREE l6_write_fail rank4: all {WORLD} ranks {''.join(m[0] for m in path)}")


def test_l6_confirm_read_failure_on_one_rank_does_not_split_the_ranks(monkeypatch):
    # L6 path 2: rank 1's confirm read fails for one step; once between boundaries, once on a boundary
    def on(rank, i):
        return not (rank == 1 and i in (5, 7))

    path, ctls, _ = _run(monkeypatch, ab="ab:1", l6_on=on)
    want = ["fold"] * 7 + ["base"] * 4 + ["fold"] * 5  # call 6 (no boundary) ignored, call 8 (boundary) agreed
    assert path == want, path
    print(f"\nDRAFTASM_AGREE l6_read_fail rank1: all {WORLD} ranks {''.join(m[0] for m in path)}")


def test_a_failed_gather_stops_instead_of_diverging(monkeypatch):
    import torch.distributed as dist

    group = SimpleNamespace(world_size=WORLD, cpu_group=object())
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group", lambda: group)

    def boom(*a, **k):
        raise RuntimeError("gloo timeout")

    monkeypatch.setattr(dist, "all_gather", boom)
    with pytest.raises(RuntimeError, match="could not agree on the lever paths"):
        l4_agree.gather([1, 0, 0, 0, 7])
    # one rank (no TP group): its own vector
    monkeypatch.setattr("vllm.distributed.parallel_state.get_tp_group",
                        lambda: (_ for _ in ()).throw(AssertionError("no TP group")))
    assert l4_agree.gather([1, 0, 0, 0, 7]) == [[1, 0, 0, 0, 7]]


def test_a_folded_loop_build_failure_stops(monkeypatch):
    h.install(monkeypatch, fold=True)

    def broken(**kw):
        raise ValueError("support_torch_compile refused the module")

    monkeypatch.setattr(draft_asm, "build_mtp_fused_loop_asm", broken)
    fp = h.make_proposer(True)
    with pytest.raises(RuntimeError, match="could not build the folded draft loop"):
        h.run_step(fp, h.make_step(700, R), serving=False)
