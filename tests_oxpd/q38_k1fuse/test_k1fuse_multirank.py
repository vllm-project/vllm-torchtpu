# SPDX-License-Identifier: Apache-2.0
"""q38 L4 T1-K1FUSE all-rank agreement (lead safety rule), CPU only. Eight simulated TP ranks (one thread each, each
with its own proposer, stub runner and DraftK1 controller) step through the same serving calls; DRAFT-ASM's
boundary all_gather meets in DRAFT1-B's barrier-backed stand-in (tests/q38_draft1b/test_draft1b_multirank.py).

  * one rank without VLLM_TPU_K1FUSE (a plain DraftOne): every K1 rank raises at the first boundary (the run stops);
  * one K1 rank with a different warm-key set: every K1 rank raises at the first boundary;
  * one rank's K1 window wish off between boundaries: K1 goes off on every rank at the next boundary only, and back;
  * a K1 CHECK mismatch on one rank only: K1 off on every rank for the run when the window closes (no raise).
"""

import threading
from types import SimpleNamespace

import pytest

import k1fuse_harness as kh
from test_draft1b_multirank import FakeGroup
from vllm_torchtpu.spec_decode import draft_asm, draft_one, k1fuse

WORLD = 8
R = 6


def _ranks(monkeypatch, kinds, check=0, period=4):
    quiet = SimpleNamespace(info=lambda *a: None, error=lambda *a: None, warning=lambda *a, **k: None)
    ranks = []
    for r, kind in enumerate(kinds):
        kh.install(monkeypatch, k1=(kind == "k1"), check=check)
        p = kh.make_pair_proposer(kind, F=0, seed=r)
        kh.warm(p, [R], seed0=40000)
        p._draft_asm.rank = r
        ranks.append(p)
    kh.install(monkeypatch, k1=True, check=check)
    monkeypatch.setattr(draft_asm, "AGREE_EVERY", period)
    for mod in (draft_asm, draft_one, k1fuse):
        monkeypatch.setattr(mod, "logger", quiet)
    return ranks


def _drive(monkeypatch, ranks, calls=16, each=None):
    group = FakeGroup(len(ranks))
    tl = threading.local()
    # q38 L4 AGREE port (v6 composition, V6COMP-BUILD; tests only): the boundary gather is l4_agree's.
    from vllm_torchtpu.runner import l4_agree

    monkeypatch.setattr(l4_agree, "gather", lambda vec: group.gather(tl.rank, vec))
    paths = [[None] * calls for _ in ranks]
    errors = {}

    def worker(r):
        tl.rank = r
        p = ranks[r]
        ctl = p._draft_asm
        p.runner.warmup_done = True
        try:
            for i in range(calls):
                if each is not None:
                    each(r, i, ctl)
                step = kh.h.make_step(41000 + i, R, cap=kh.CAP)
                p.runner._l4_agree.tick(p.runner)  # q38 L4 AGREE port: the engine step's clock tick (execute_model)
                ctl.begin([step.chunk], [step.nt], SimpleNamespace(has_structured_output_requests=False), True)
                mode = ctl._k1_mode() if isinstance(ctl, k1fuse.DraftK1) else None
                paths[r][i] = {"fold": "K", "check": "c", None: "-"}[mode]
                if isinstance(ctl, k1fuse.DraftK1) and mode == "check":
                    ctl.k1_check_left -= 1
                ctl.step = None
        except Exception as exc:  # noqa: BLE001
            errors[r] = (len([x for x in paths[r] if x is not None]), repr(exc))
            # break the barrier only after the other ranks have left this gather (an immediate abort would turn
            # their own raise at this boundary into a BrokenBarrierError); a rank that goes on waits at its next one
            threading.Timer(3.0, group.barrier.abort).start()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(len(ranks))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    return paths, errors


def test_one_rank_without_the_flag_stops_the_run(monkeypatch):
    kinds = ["k1"] * WORLD
    kinds[3] = "one"
    ranks = _ranks(monkeypatch, kinds)
    paths, errors = _drive(monkeypatch, ranks, calls=4)
    k1_ranks = [r for r in range(WORLD) if r != 3]
    for r in k1_ranks:
        assert r in errors and "configured on TP ranks" in errors[r][1] and errors[r][0] == 0, (r, errors.get(r))
    print(f"\nK1FUSE_AGREE flag off on rank 3: every K1 rank raised at the first boundary ({len(k1_ranks)} of "
          f"{len(k1_ranks)}: {errors[0][1][:90]}...)")


def test_one_rank_with_another_warm_key_set_stops_the_run(monkeypatch):
    ranks = _ranks(monkeypatch, ["k1"] * WORLD)
    ranks[5]._draft_asm.warm["k1"].add(("extra",))
    paths, errors = _drive(monkeypatch, ranks, calls=4)
    assert sorted(errors) == list(range(WORLD)), errors
    assert all("warm-key sets differ" in e[1] and e[0] == 0 for e in errors.values()), errors
    print("\nK1FUSE_AGREE rank 5 with an extra K1 warm key: all 8 ranks raised at the first boundary")


def test_one_ranks_wish_off_turns_k1_off_everywhere_at_boundaries_only(monkeypatch):
    ranks = _ranks(monkeypatch, ["k1"] * WORLD)
    real = k1fuse.DraftK1._stage_window

    def window(self, lever, ab):
        if lever == k1fuse.LEVER and getattr(self, "_q38_off", False):
            return False
        return real(self, lever, ab)

    monkeypatch.setattr(k1fuse.DraftK1, "_stage_window", window)

    def each(r, i, ctl):
        ctl._q38_off = r == 2 and 5 <= i <= 9

    paths, errors = _drive(monkeypatch, ranks, calls=16, each=each)
    assert not errors, errors
    for i in range(16):
        assert len({paths[r][i] for r in range(WORLD)}) == 1, (i, [paths[r][i] for r in range(WORLD)])
    path = "".join(paths[0])
    # boundaries at calls 1, 4, 8, 12, 16 (1-based): rank 2's wish is off at the call-8 boundary only
    assert path == "K" * 7 + "-" * 4 + "K" * 5, path
    print(f"\nK1FUSE_AGREE rank 2 wish off on calls 6-10: all {WORLD} ranks {path}")


def test_a_check_mismatch_on_one_rank_turns_k1_off_on_every_rank(monkeypatch):
    ranks = _ranks(monkeypatch, ["k1"] * WORLD, check=2)

    def each(r, i, ctl):
        if r == 6 and i == 1:
            ctl.k1_mismatches += 1  # as _k1_check_draft_side counts a mismatch

    paths, errors = _drive(monkeypatch, ranks, calls=12, each=each)
    assert not errors, errors
    for i in range(12):
        assert len({paths[r][i] for r in range(WORLD)}) == 1, (i, [paths[r][i] for r in range(WORLD)])
    path = "".join(paths[0])
    assert path.startswith("cc") and "K" not in path, path
    assert all(p._draft_asm.k1_failed for p in ranks)
    print(f"\nK1FUSE_AGREE CHECK mismatch on rank 6 only: all {WORLD} ranks {path}, K1 off for the run, no raise")
