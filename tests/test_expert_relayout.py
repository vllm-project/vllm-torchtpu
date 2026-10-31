# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The dynamic re-layout controller, on the host, with fakes for the tap, the
mover and the per-step all-reduce (expert_relayout.py)."""

import numpy as np
import pytest
import torch

from vllm_torchtpu import eplb_policy
from vllm_torchtpu import expert_mover as mv
from vllm_torchtpu import expert_relayout as rl

pytestmark = pytest.mark.cpu_test

SPEC = dict(experts=8, active=4, staging=1, replica_cap=2)
LAYOUT = [[0, 1, 2, 3, rl.FREE], [4, 5, 6, 7, rl.FREE]]


def _layout():
    return rl.LayerLayout([list(r) for r in LAYOUT])


def _rows(hot=(), steps=3):
    """[steps, experts] rows with the `hot` experts at 200 and the rest at 10.
    Two hot experts on one rank is what a repacking can fix: with no
    redundant active slot a single hot expert cannot be split."""
    rows = np.full((steps, 8), 10.0)
    for e in hot:
        rows[:, e] = 200.0
    return rows


# ---------------------------------------------------------------------------
# Layouts, windows, plans


def test_layout_slot_table_and_window_rows():
    lay = _layout()
    assert lay.ranks() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    table = lay.slot_table(8, 2)
    assert table[0] == [0, -1] and table[4] == [5, -1]  # rank 1's slots start at 5
    assert lay.free_slots(0) == [4] and lay.holders(6) == [1]
    delta = np.array([7, 1, 1, 1, 0, 2, 2, 2, 2, 0])  # per physical slot
    rows = rl.window_rows(delta, lay, 8)
    assert rows.tolist() == [7, 1, 1, 1, 2, 2, 2, 2]
    with pytest.raises(ValueError, match="more than 1 copies"):
        rl.LayerLayout([[0, 1, 2, 3, 0], [4, 5, 6, 7, -1]]).slot_table(8, 1)


def test_plan_hash_is_positive_deterministic_and_sensitive():
    a = rl.Plan(1, 0, {("target", 0): _layout()})
    b = rl.Plan(1, 0, {("target", 0): _layout()})
    assert a.hash62 == b.hash62 > 0 and a.hash62 < 1 << rl.HASH_BITS
    moved = _layout()
    moved.held[0][4] = 7
    assert rl.Plan(1, 0, {("target", 0): moved}).hash62 != a.hash62
    assert rl.Plan(2, 0, {("target", 0): _layout()}).hash62 != a.hash62


def _copies(ranks, experts=8):
    return np.bincount([e for r in ranks for e in r], minlength=experts)


def test_policy_copy_cap_bounds_the_replicas():
    rows = np.asarray(_rows(hot=(0,)), dtype=np.float32)[:, None, :]  # [K, 1, E]
    counts = _copies(eplb_policy.rebalance_experts(rows, 16, 4, max_copies=2)[0])
    assert int(counts.max()) <= 2 and int(counts.sum()) == 16
    counts3 = _copies(eplb_policy.rebalance_experts(rows, 16, 4)[0])
    assert int(counts3.max()) > 2  # the cap was what held it
    with pytest.raises(ValueError, match="max_copies"):
        eplb_policy.rebalance_experts(rows, 16, 4, max_copies=0)


def test_slowest_rank_load_splits_rows_over_copies():
    lay = rl.LayerLayout([[0, 1, 0], [2, 3, rl.FREE]])
    rows = np.array([10.0, 1.0, 4.0, 2.0])
    # Rank 0: 10/2 + 1 + 10/2 = 11; rank 1: 4 + 2 = 6.
    assert rl.slowest_rank_load(rows, lay, 4) == 11.0


# ---------------------------------------------------------------------------
# Solve: matching, budget, hysteresis


def test_rank_matching_removes_renaming_moves():
    lay = _layout()
    swapped = [[4, 5, 6, 7], [0, 1, 2, 3]]  # the policy's ranks, relabelled
    assert rl._match_ranks(swapped, lay) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    nxt = rl._budgeted_next(lay, rl._match_ranks(swapped, lay), np.ones(8), 1)
    assert nxt.held == lay.held


def test_budget_limits_adds_and_keeps_every_expert_held():
    lay = _layout()
    # The target swaps two experts between the ranks: 2 adds per rank, budget 1.
    targets = [[0, 1, 6, 7], [4, 5, 2, 3]]
    load = np.array([1, 1, 5, 2, 1, 1, 9, 3], dtype=float)
    nxt = rl._budgeted_next(lay, targets, load, staging=1)
    # Rank 0 gained its hottest wanted expert (6) and let one departing go.
    assert nxt.held[0].count(6) == 1 and nxt.held[0].count(rl.FREE) == 1
    assert nxt.held[1].count(2) == 1 and nxt.held[1].count(rl.FREE) == 1
    held = sorted(e for row in nxt.held for e in row if e != rl.FREE)
    assert held == list(range(8))  # complete: nobody orphaned
    moves = rl._moves_between("target", 0, lay, nxt)
    assert {(m.expert, m.src_rank, m.dst_rank) for m in moves} == {(6, 1, 0), (2, 0, 1)}
    assert all(m.src_slot == lay.held[m.src_rank].index(m.expert) for m in moves)
    assert all(nxt.held[m.dst_rank][m.dst_slot] == m.expert for m in moves)
    # A second generation toward the same target finishes the swap.
    nxt2 = rl._budgeted_next(nxt, targets, load, staging=1)
    assert sorted(e for e in nxt2.held[0] if e != rl.FREE) == [0, 1, 6, 7]
    assert sorted(e for e in nxt2.held[1] if e != rl.FREE) == [2, 3, 4, 5]


def test_budget_keeps_a_moving_hot_expert_under_the_replica_cap():
    """DP8 `eplb-dyn-dp8-e` generation 3: ranks 0 and 1 hold the hot expert 0
    (at the cap). Their targets drop it and one light expert each but the
    budget admits one add, so they remove the light one (held elsewhere too)
    and keep 0, while ranks 2 and 3 add it: four copies against a cap of two,
    which the slot table cannot hold. With the cap, the adds of 0 are blocked
    and the layout stays valid."""
    old = rl.LayerLayout(
        [[0, 1, 2, rl.FREE], [0, 3, 4, rl.FREE], [5, 6, 1, rl.FREE], [7, 3, 5, rl.FREE]]
    )
    targets = [[2, 5, 6], [4, 6, 7], [5, 0, 1], [7, 3, 0]]
    load = np.array([100, 1, 5, 1, 5, 5, 5, 5], dtype=float)
    loose = rl._budgeted_next(old, targets, load, staging=1)
    assert loose.copies(8).max() == 4  # the step that crashed the install
    nxt = rl._budgeted_next(old, targets, load, staging=1, cap=2)
    copies = nxt.copies(8)
    assert copies.max() <= 2 and copies.min() >= 1
    assert all(len(r) == 3 for r in nxt.ranks())
    nxt.slot_table(8, 2)  # what install builds; must not raise
    assert 0 not in nxt.held[2] and 0 not in nxt.held[3]
    assert nxt.held != old.held  # the other adds still happen


def test_solve_group_layouts_always_fit_the_slot_table():
    """Random windows over random valid layouts: every planned layout holds
    each expert at least once and at most `replica_cap` times, with the
    active count per rank unchanged, and every move names a live source."""
    rng = np.random.default_rng(7)
    checked = 0
    for shards, experts, active, cap in ((2, 8, 4, 2), (4, 12, 4, 2), (4, 14, 4, 3)):
        spec = dict(experts=experts, active=active, staging=1, replica_cap=cap)
        for _ in range(12):
            seed_rows = rng.integers(1, 200, size=(3, 1, experts)).astype(np.float32)
            try:
                lists = eplb_policy.rebalance_experts(
                    seed_rows, shards * active, active, max_copies=cap
                )
            except ValueError:  # the greedy packer's known refusal on odd shapes
                continue
            held = [list(r) + [rl.FREE] for r in lists[0]]
            old = rl.LayerLayout(held)
            rows = rng.integers(1, 300, size=(4, experts)).astype(float)
            rows[:, rng.integers(experts)] *= 20  # one hot expert
            try:
                layouts, moves, _ = rl.solve_group(
                    "target", {0: old}, {0: rows}, min_gain=0.0, **spec
                )
            except ValueError:
                continue
            checked += 1
            for lay in layouts.values():
                copies = lay.copies(experts)
                assert copies.min() >= 1 and copies.max() <= cap
                assert all(len(r) == active for r in lay.ranks())
                lay.slot_table(experts, cap)
            for m in moves:
                assert old.held[m.src_rank][m.src_slot] == m.expert
    assert checked >= 12


def test_a_refused_layer_is_skipped_and_the_others_still_solve(monkeypatch):
    """DP8 `eplb-dyn-dp8-f`: the greedy packer refused 6 windows ("fewer
    replicas than ranks are needed") and the whole batched solve died with
    them. The layers are then packed one at a time and the refused one is
    skipped."""
    real = eplb_policy.rebalance_experts

    def picky(rows, *args, **kwargs):
        # Refuse any batch that contains layer 1's window (its expert 7 is the
        # hot one); a batch of layer 0 alone goes through.
        if bool((rows[:, :, 7] > 100).any()):
            raise ValueError("no rank can take a copy without holding the expert twice")
        return real(rows, *args, **kwargs)

    monkeypatch.setattr(eplb_policy, "rebalance_experts", picky)
    lay0, lay1 = _layout(), _layout()
    windows = {0: _rows(hot=(0, 1)), 1: _rows(hot=(7,))}
    layouts, moves, _ = rl.solve_group(
        "target", {0: lay0, 1: lay1}, windows, min_gain=0.0, **SPEC
    )
    assert 1 not in layouts
    assert 0 in layouts and moves and all(m.layer == 0 for m in moves)


def test_removes_never_orphan_a_single_copy():
    lay = _layout()
    # Rank 0 wants 7 and would drop 3, but 3 has no other holder in the
    # target either: the add is given back rather than orphaning 3.
    targets = [[0, 1, 2, 7], [4, 5, 6, 7]]
    nxt = rl._budgeted_next(lay, targets, np.ones(8), staging=1)
    held = sorted(e for row in nxt.held for e in row if e != rl.FREE)
    assert held == list(range(8)) or held == sorted(list(range(8)) + [7])
    assert 3 in nxt.held[0] or 3 in nxt.held[1]


def test_solve_group_moves_for_a_hot_expert_and_skips_a_flat_window():
    lay = _layout()
    flat = {0: _rows()}
    layouts, moves, _ = rl.solve_group("target", {0: lay}, flat, min_gain=0.03, **SPEC)
    assert layouts == {} and moves == []
    hot = {0: _rows(hot=(0, 1))}
    layouts, moves, predicted = rl.solve_group(
        "target", {0: lay}, hot, min_gain=0.03, **SPEC
    )
    assert set(layouts) == {0} and moves
    before, after = predicted[0]
    assert after < before
    held = sorted(e for row in layouts[0].held for e in row if e != rl.FREE)
    assert set(held) == set(range(8))
    for row in layouts[0].held:
        assert row.count(rl.FREE) == 1  # the budget's spare slot survives


# ---------------------------------------------------------------------------
# The protocol over a fake all-reduce


class _Tap:
    """Replicated counts: one hot expert, identical on every rank."""

    def __init__(self):
        self.step = 0

    def advance(self, steps=1):
        self.step += steps

    def snapshot(self):
        counts = np.full(10, 10 * self.step, dtype=np.int64)
        counts[0] = counts[1] = 200 * self.step  # experts 0 and 1, both on rank 0
        counts[4] = counts[9] = 0  # staging slots
        return {("target", 0): counts.copy()}


class _Mover:
    def __init__(self, generation=0, auto=True):
        self.staged = generation
        self.auto = auto
        self.started: list[rl.Plan] = []
        self.cancelled = 0
        self.drained: list[float] = []
        self._failed = False

    def drain(self, timeout_s):
        self.drained.append(timeout_s)
        return True

    def start(self, plan, modules):
        self.started.append(plan)
        if self.auto:
            self.staged = plan.generation

    def pump(self):
        pass

    def staged_generation(self):
        return self.staged

    def failed(self):
        return self._failed

    def cancel(self):
        self.cancelled += 1

    def settle(self, generation):
        self.staged = generation


def _controller(rank, tap, mover, installs, **kw):
    return rl.Controller(
        rank=rank,
        groups={"target": SPEC},
        layouts={("target", 0): _layout()},
        modules={("target", 0): f"module-{rank}"},
        snapshot=tap.snapshot,
        install=lambda module, table: installs.append((module, table)),
        mover=mover,
        window_steps=4,
        sub_windows=2,
        warmup_steps=0,
        **{"synchronous_solve": True, **kw},
    )


def _run(controllers, tap, steps, start=0):
    for step in range(start, start + steps):
        fields = [c.fields() for c in controllers]
        reduced = tuple(max(f[i] for f in fields) for i in range(3))
        for c in controllers:
            c.on_step(step, reduced)
            c.after_dispatch()
        tap.advance()


def test_two_ranks_reach_the_same_plan_and_install_in_the_same_step():
    tap = _Tap()
    installs = []
    movers = [_Mover(), _Mover()]
    cs = [_controller(r, tap, movers[r], installs) for r in range(2)]
    _run(cs, tap, 4)  # step 0 takes the baseline snapshot, nothing else
    assert all(c.state == rl.IDLE for c in cs) and installs == []
    _run(cs, tap, 1, start=4)  # trigger: both solve the same plan
    assert all(c.state == rl.PENDING for c in cs)
    assert cs[0].pending.hash62 == cs[1].pending.hash62
    _run(cs, tap, 1, start=5)  # hashes agree -> staging starts (auto-staged)
    assert all(c.state == rl.STAGING for c in cs)
    assert [len(m.started) for m in movers] == [1, 1]
    _run(cs, tap, 1, start=6)  # both staged -> both install this step
    assert all(c.state == rl.IDLE and c.generation == 1 for c in cs)
    assert [m for m, _ in installs] == ["module-0", "module-1"]
    table = installs[0][1]
    assert len(table) == 8 and all(len(r) == 2 for r in table)
    assert cs[0].layouts[("target", 0)].held == cs[1].layouts[("target", 0)].held
    assert cs[0].stats["installed"] == 1 and len(cs[0].history) == 1


def test_install_waits_for_the_slowest_rank():
    tap = _Tap()
    installs = []
    movers = [_Mover(), _Mover(auto=False)]
    cs = [_controller(r, tap, movers[r], installs) for r in range(2)]
    _run(cs, tap, 7)
    assert all(c.state == rl.STAGING for c in cs) and installs == []
    _run(cs, tap, 1, start=7)
    assert installs == []  # rank 1 still staging
    movers[1].staged = 1
    _run(cs, tap, 1, start=8)
    assert all(c.generation == 1 for c in cs) and len(installs) == 2


def test_hash_disagreement_drops_the_generation_everywhere():
    tap = _Tap()
    installs = []
    cs = [_controller(r, tap, _Mover(), installs) for r in range(2)]
    _run(cs, tap, 5)
    assert all(c.state == rl.PENDING for c in cs)
    cs[1].pending.generation = 7  # a rank that solved something else
    _run(cs, tap, 1, start=5)
    assert all(c.state == rl.IDLE and c.generation == 0 for c in cs)
    assert all(c.stats["aborted"] == 1 for c in cs) and installs == []


def test_staging_timeout_and_mover_failure_abort():
    tap = _Tap()
    installs = []
    slow = _Mover(auto=False)
    cs = [_controller(0, tap, slow, installs, timeout_windows=1)]
    _run(cs, tap, 6)
    assert cs[0].state == rl.STAGING
    _run(cs, tap, 5, start=6)  # past the timeout (window 4 x 1)
    assert (
        cs[0].state == rl.IDLE and cs[0].stats["aborted"] == 1 and slow.cancelled == 1
    )
    # Every rank times out at the same step: the exchange is drained first.
    assert slow.drained == [rl.DRAIN_S]
    failing = _Mover(auto=False)
    cs = [_controller(0, tap, failing, installs)]
    _run(cs, tap, 6)
    failing._failed = True
    _run(cs, tap, 1, start=6)
    assert cs[0].state == rl.IDLE and cs[0].stats["aborted"] == 1
    # A rank-local failure cannot count on its peers: no drain.
    assert failing.drained == [] and failing.cancelled == 1


def test_flat_traffic_never_triggers_a_move():
    class Flat(_Tap):
        def snapshot(self):
            counts = np.full(10, 10 * self.step, dtype=np.int64)
            counts[4] = counts[9] = 0
            return {("target", 0): counts}

    tap = Flat()
    installs = []
    cs = [_controller(0, tap, _Mover(), installs)]
    _run(cs, tap, 13)
    assert cs[0].stats["installed"] == 0 and cs[0].stats["skipped"] >= 2


# ---------------------------------------------------------------------------
# Arming in a server


_CPU = mv.CpuTransport()


def _arm_env(monkeypatch, **over):
    from vllm_torchtpu import envs

    values = dict(
        TPU_MOE_EPLB_DYNAMIC=True,
        TPU_MOE_EPLB_WINDOW_STEPS=16,
        TPU_MOE_EPLB_MIN_GAIN=0.05,
    )
    values.update(over)
    for k, v in values.items():
        monkeypatch.setitem(vars(envs), k, v)


def _fake_placed_world(monkeypatch, staging=1):
    from types import SimpleNamespace

    from vllm_torchtpu import expert_mover, expert_placement
    from vllm_torchtpu.layers.adapter import expert_load

    group = SimpleNamespace(
        experts=8,
        active=4,
        staging=staging,
        slots=4 + staging,
        replica_cap=2,
        ranks=lambda layer: ((0, 1, 2, 3), (4, 5, 6, 7)),
    )
    placement = SimpleNamespace(group=lambda name: group if name == "target" else None)
    module = SimpleNamespace(
        layer_name="layers.0.mlp.experts",
        _tpu_expert_slot=torch.zeros((8, 2), dtype=torch.int32),
    )
    monkeypatch.setattr(expert_placement, "current", lambda vllm_config=None: placement)
    monkeypatch.setattr(
        expert_placement, "applied_layers", lambda: {("target", 0): module}
    )
    monkeypatch.setattr(expert_load, "registered_layers", lambda: 1)
    monkeypatch.setattr(
        expert_mover, "make_gloo_group", lambda ranks: ("gloo", tuple(ranks))
    )
    return module


def test_arm_builds_a_controller_from_the_placed_layers(monkeypatch):
    _arm_env(monkeypatch)
    _fake_placed_world(monkeypatch)
    c = rl.arm(ep_rank=1, ep_world=2, ep_global_ranks=[4, 5], transport=_CPU)
    assert c is not None and c.rank == 1 and c.window_steps == 16
    assert c.groups == {"target": dict(experts=8, active=4, staging=1, replica_cap=2)}
    assert c.layouts[("target", 0)].held == [
        [0, 1, 2, 3, rl.FREE],
        [4, 5, 6, 7, rl.FREE],
    ]
    assert c.mover.p2p.group == ("gloo", (4, 5)) and c.mover.p2p._global(1) == 5


def test_arm_refuses_without_the_knob_staging_or_lockstep(monkeypatch):
    _fake_placed_world(monkeypatch)
    _arm_env(monkeypatch, TPU_MOE_EPLB_DYNAMIC=False)
    assert rl.arm(ep_rank=0, ep_world=2, transport=_CPU) is None
    _arm_env(monkeypatch)
    assert rl.arm(ep_rank=0, ep_world=2, lockstep=False, transport=_CPU) is None
    _fake_placed_world(monkeypatch, staging=0)
    assert rl.arm(ep_rank=0, ep_world=2, transport=_CPU) is None


def test_the_tap_exists_under_the_dynamic_knob(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter import expert_load

    monkeypatch.setitem(vars(envs), "TPU_MOE_EPLB_DYNAMIC", True)
    assert expert_load.enabled()
    monkeypatch.setitem(vars(envs), "TPU_MOE_EPLB_DYNAMIC", False)
    assert not expert_load.enabled()


# ---------------------------------------------------------------------------
# Plan -> mover -> table, end to end on the host


def test_a_solved_generation_moves_exactly_the_bytes_its_table_names():
    """Solve a generation from a hot window, run the mover for every rank over
    an in-process mailbox on slot-major weight tensors, then check that every
    slot the new table names holds its expert's original bytes -- the
    consistency the device bit-exact test relies on."""
    from types import SimpleNamespace

    experts, shards, active, staging = 8, 2, 4, 1
    lay = _layout()
    rows = {0: _rows(hot=(0, 1))}
    layouts, moves, _ = rl.solve_group("target", {0: lay}, rows, min_gain=0.0, **SPEC)
    assert moves, "the hot window must move something"
    nxt = layouts[0]
    plan = rl.Plan(1, 0, layouts, moves)
    # Each rank's weights: slot s of rank r holds bytes that spell the expert.
    k, n = 32, 32  # a slot of int32 is one 4 KiB block

    def weights(rank):
        w = torch.zeros((active + staging, k, n), dtype=torch.int32)
        for j, e in enumerate(lay.held[rank]):
            if e != rl.FREE:
                w[j] = 1000 * rank + e  # distinct per (origin rank, expert)
        return w

    mods = {
        r: SimpleNamespace(local_num_experts=active + staging, w13_weight=weights(r))
        for r in range(shards)
    }
    exchange = mv.LocalExchange()
    movers = [
        mv.SlabMover(r, mv.CpuTransport(), exchange.endpoint(r)) for r in range(shards)
    ]
    import time

    for r, m in enumerate(movers):
        m.start(plan, {("target", 0): mods[r]})
    for _ in range(400):
        for m in movers:
            m.pump()
        if all(m.staged_generation() == 1 or m.failed() for m in movers):
            break
        time.sleep(0.005)
    assert all(m.staged_generation() == 1 for m in movers)
    table = nxt.slot_table(experts, SPEC["replica_cap"])
    for e, slots in enumerate(table):
        for s in slots:
            if s == rl.FREE:
                continue
            r, j = divmod(s, active + staging)
            value = int(mods[r].w13_weight[j, 0, 0])
            assert value % 1000 == e, f"slot {s} should hold expert {e}, holds {value}"
    # Every expert is still held somewhere and nobody holds one twice.
    held = [e for row in nxt.held for e in row if e != rl.FREE]
    assert set(held) == set(range(experts))
    for row in nxt.held:
        live = [e for e in row if e != rl.FREE]
        assert len(live) == len(set(live)) == active


def test_the_threaded_solve_publishes_when_done_and_peers_wait_for_it():
    """With the solve on a thread a rank publishes 0 until its plan exists;
    the others wait (min hash 0) instead of dropping the generation."""
    import time

    tap = _Tap()
    installs = []
    cs = [
        _controller(0, tap, _Mover(), installs, synchronous_solve=False),
        _controller(1, tap, _Mover(), installs, synchronous_solve=True),
    ]
    _run(cs, tap, 5)  # step 4 triggers both; rank 1 solved inline
    assert cs[1].state == rl.PENDING
    assert cs[0].state in (rl.SOLVING, rl.PENDING)
    for _ in range(200):  # let rank 0's thread finish
        if cs[0]._solve_thread is None or not cs[0]._solve_thread.is_alive():
            break
        time.sleep(0.01)
    # Rank 1 published its hash while rank 0 published 0: nobody dropped.
    assert all(c.stats["aborted"] == 0 for c in cs)
    _run(cs, tap, 4, start=5)
    assert all(c.generation == 1 for c in cs) and len(installs) == 2


def test_a_rank_that_never_publishes_times_the_generation_out():
    tap = _Tap()
    installs = []
    c = _controller(0, tap, _Mover(), installs, timeout_windows=1)
    _run([c], tap, 5)
    assert c.state == rl.PENDING
    # Fake a peer that keeps publishing 0: max is ours, -min is 0.
    for step in range(5, 12):
        c.on_step(step, (c.pending.hash62 if c.pending else 0, 0, 0))
        c.after_dispatch()
        tap.advance()
    assert c.state == rl.IDLE and c.stats["aborted"] == 1 and installs == []


def test_the_per_rank_cap_keeps_a_generation_within_what_the_budget_moves():
    """D8 (j-608af38f): ~1 GB of slabs in plus out per rank through a 16 MB
    step budget needs ~62 pumps against a 64-step timeout, so every
    generation timed out. The controller now caps each rank's in-plus-out
    bytes at half of budget x timeout, keeping layers by saving per byte."""
    tap = _Tap()
    c = _controller(0, tap, _Mover(), [], step_budget_bytes=100, timeout_windows=1)
    assert c.timeout_steps == 4 and c.rank_bytes_cap == 100 * 4 // 2
    c.move_bytes = {"target": 50}
    plan = rl.Plan(generation=1, trigger_step=0)
    lay = _layout()
    # Layer 0: two moves into rank 1 (rank 1: 100 in; rank 0: 100 out).
    # Layer 1: one move 2 -> 3. Layer 2: two moves into rank 1 again.
    for lid, saving, moves in (
        (0, 40.0, [(0, 1), (0, 1)]),
        (1, 10.0, [(2, 3)]),
        (2, 30.0, [(2, 1), (2, 1)]),
    ):
        plan.layouts[("target", lid)] = lay
        plan.predicted[("target", lid)] = (100.0, 100.0 - saving)
        for i, (src, dst) in enumerate(moves):
            plan.moves.append(rl.Move("target", lid, i, src, i, dst, 4))
    out = c._cap(plan)
    # Cap 200: layer 0 then layer 2 put rank 1 at exactly 200, layer 1 puts
    # ranks 2 and 3 at 150 and 50: everything fits.
    assert set(out.layouts) == {("target", 0), ("target", 1), ("target", 2)}
    c.rank_bytes_cap = 150
    plan2 = rl.Plan(generation=1, trigger_step=0)
    for lid, saving, moves in (
        (0, 40.0, [(0, 1), (0, 1)]),
        (1, 10.0, [(2, 3)]),
        (2, 30.0, [(2, 1), (2, 1)]),
    ):
        plan2.layouts[("target", lid)] = lay
        plan2.predicted[("target", lid)] = (100.0, 100.0 - saving)
        for i, (src, dst) in enumerate(moves):
            plan2.moves.append(rl.Move("target", lid, i, src, i, dst, 4))
    out2 = c._cap(plan2)
    # Layer 0 puts rank 1 at 100; layer 2 would take it to 200 > 150: dropped;
    # layer 1 (ranks 2 and 3 at 50) still fits.
    assert set(out2.layouts) == {("target", 0), ("target", 1)}
    assert all((m.group, m.layer) in out2.layouts for m in out2.moves)


def test_direct_staged_next_breaks_single_copy_cycle_and_chains_decode():
    """When the policy target only replicates hot experts without permuting
    single-copy experts, `_direct_staged_next` still finds a valid 1-for-1
    swap between the hottest and coldest ranks, and decode multi-generation
    chaining converges and freezes once no layer clears `min_gain`."""
    # Rank 0 holds two medium-hot experts (0, 1) and two warm experts (2, 3);
    # Rank 1 holds four cold experts (4, 5, 6, 7).
    lay = _layout()
    load = np.array([120.0, 110.0, 80.0, 70.0, 10.0, 10.0, 10.0, 10.0])
    before = rl.slowest_rank_load(load, lay, 8)
    nxt = rl._direct_staged_next(lay, load, staging=1, cap=2)
    after = rl.slowest_rank_load(load, nxt, 8)
    assert after < before
    copies = nxt.copies(8)
    assert int(copies.min()) >= 1 and int(copies.max()) <= 2
    assert all(len(r) == 4 for r in nxt.ranks())

    # Verify decode chaining and convergence freeze on Controller:
    tap = _Tap()
    installs = []
    c = _controller(0, tap, _Mover(), installs, min_gain=0.005)
    # Drive decode phase steps until first decode window triggers and installs,
    # then chains until converged.
    for step in range(1, 30):
        h, nh, ns = c.fields()
        c.on_step(step, (h, nh, ns), phase="decode")
        c.after_dispatch()
        tap.advance()
    assert c.generation >= 1
    assert c._decode_converged is True


def test_window_deltas_wrap_cleanly_across_int32_overflow():
    """The on-device load accumulator is int32; when a hot slot wraps past
    2^31 - 1 into negative int32 values, the window delta must still be the
    positive unsigned difference modulo 2^32."""
    tap = _Tap()
    installs = []
    c = _controller(0, tap, _Mover(), installs)
    snap_a = {
        ("target", 0): np.array(
            [2_147_483_600, 100, 100, 100, 0, 100, 100, 100, 100, 0], dtype=np.int64
        )
    }
    # Slot 0 adds 200 rows, wrapping signed int32 to -2_147_483_496.
    wrapped = int(np.int32(np.int64(2_147_483_600 + 200)))
    snap_b = {
        ("target", 0): np.array(
            [wrapped, 110, 110, 110, 0, 110, 110, 110, 110, 0], dtype=np.int64
        )
    }
    logical = c._compute_logical_windows([snap_a, snap_b], c.layouts)
    assert logical[("target", 0)][0].tolist() == [
        200.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ]
