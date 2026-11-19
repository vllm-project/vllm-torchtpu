# SPDX-License-Identifier: Apache-2.0
"""q38 L6-HOST JITTER levers gcfreeze / switch / mainpin, and L6-SYNC's syncgc / mainpin2 (core/l6_jitter.py).
Host-only; CDK CPU pod. The syncgc / mainpin2 tests use a fake sysfs tree and fake affinity calls."""

import gc
import os
import sys

import pytest

S = 4


@pytest.fixture
def jt(monkeypatch):
    from vllm_torchtpu.core import l6_ab, l6_jitter

    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS",
                                                  "VLLM_TPU_SERVED_AB"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(l6_ab, "_rank_world", lambda: (0, 1))
    lines = []
    for mod in (l6_ab, l6_jitter):
        monkeypatch.setattr(mod.logger, "info", lambda fmt, *a: lines.append(fmt % a))
        monkeypatch.setattr(mod.logger, "error", lambda fmt, *a: lines.append("ERROR " + fmt % a))
    prev = sys.getswitchinterval()
    sys.setswitchinterval(0.002)
    gc.unfreeze()
    l6_jitter.restore()

    def setup(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        return l6_ab, l6_jitter

    yield setup, lines, monkeypatch
    l6_jitter.restore()
    gc.unfreeze()
    sys.setswitchinterval(prev)
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def _steps(ab, jit, n):
    out = []
    for _ in range(n):
        ab.begin_step()
        jit.on_step()
        out.append((gc.get_freeze_count() > 0, round(sys.getswitchinterval() * 1e6)))
    return out


def test_gcfreeze_follows_the_windows(jt):
    setup, lines, _ = jt
    ab, jit = setup(VLLM_TPU_L6_AB_STEPS=str(S), VLLM_TPU_L6_AB_CONFIRM="1", VLLM_TPU_L6_GC_FREEZE="ab:1")
    st = _steps(ab, jit, 3 * S - 1)  # windows 0 (3 steps), 1, 2
    frozen = [f for f, _ in st]
    assert frozen == [False] * 3 + [True] * 4 + [False] * 4, frozen
    assert ab._state["counts"].get("gcfreeze.unfreeze") == 1


def test_switch_sets_and_restores_the_interval(jt):
    setup, lines, _ = jt
    ab, jit = setup(VLLM_TPU_L6_AB_STEPS=str(S), VLLM_TPU_L6_AB_CONFIRM="1", VLLM_TPU_L6_SWITCH="ab:1",
                    VLLM_TPU_L6_SWITCH_US="300")
    jit.SWITCH_US = 300
    st = _steps(ab, jit, 3 * S - 1)
    assert [us for _, us in st] == [2000] * 3 + [300] * 4 + [2000] * 4, st


def test_gcfreeze_refused_next_to_d1e_freeze(jt):
    setup, lines, mp = jt
    ab, jit = setup(VLLM_TPU_L6_GC_FREEZE="1")
    mp.setattr(jit, "D1E_FREEZE", True)
    st = _steps(ab, jit, 3)
    assert not any(f for f, _ in st)
    assert sum(ln.startswith("ERROR q38 L6 gcfreeze refused") for ln in lines) == 1


def test_uniform_gcfreeze_from_the_first_step(jt):
    setup, lines, _ = jt
    ab, jit = setup(VLLM_TPU_L6_GC_FREEZE="1")
    st = _steps(ab, jit, 2)
    assert all(f for f, _ in st)


def _aff(tid):
    return os.sched_getaffinity(tid)


def test_mainpin_isolates_the_main_thread_and_restores(jt):
    import threading

    setup, lines, _ = jt
    allowed = os.sched_getaffinity(0)
    assert len(allowed) >= 2, allowed  # the CDK CPU pod has 8 CPUs
    stop = threading.Event()
    workers = [threading.Thread(target=stop.wait, daemon=True) for _ in range(3)]
    for t in workers:
        t.start()
    try:
        main = threading.get_native_id()
        before = {t.native_id: _aff(t.native_id) for t in workers}
        before[main] = _aff(main)
        ab, jit = setup(VLLM_TPU_L6_AB_STEPS=str(S), VLLM_TPU_L6_AB_CONFIRM="1", VLLM_TPU_L6_MAIN_PIN="ab:1")
        cpu = max(allowed)
        for _ in range(S - 1):  # window 0: off
            ab.begin_step(); jit.on_step()
        assert _aff(main) == before[main]
        ab.begin_step(); jit.on_step()  # first step of window 1: on
        assert _aff(main) == {cpu}
        for t in workers:
            assert cpu not in _aff(t.native_id), (t.native_id, _aff(t.native_id))
        for _ in range(S):  # rest of window 1, then the first step of window 2: off
            ab.begin_step(); jit.on_step()
        assert _aff(main) == before[main]
        for t in workers:
            assert _aff(t.native_id) == before[t.native_id]
        assert ab._state["counts"].get("mainpin.unpin") == 1
    finally:
        stop.set()


def test_mainpin_failure_turns_it_off_once(jt):
    setup, lines, mp = jt
    ab, jit = setup(VLLM_TPU_L6_MAIN_PIN="1")
    mp.setattr(jit, "PIN_CPU", 100000)  # not an allowed CPU
    before = os.sched_getaffinity(0)
    for _ in range(3):
        ab.begin_step(); jit.on_step()
    assert os.sched_getaffinity(0) == before
    assert sum(ln.startswith("ERROR q38 L6 mainpin off") for ln in lines) == 1


def test_mainpin_ranks_pick_distinct_cpus_and_reserve_all():
    from vllm_torchtpu.core import l6_jitter

    allowed = set(range(100, 112))  # one NUMA node's CPUs under VLLM_TPU_CPU_PIN=numa
    picks = [l6_jitter._pick(allowed, r, 8) for r in range(4)]  # ranks 0-3 share the node
    cpus = [c for c, _ in picks]
    assert len(set(cpus)) == 4 and all(res == set(range(104, 112)) for _, res in picks), picks
    assert cpus == [111, 110, 109, 108]
    picks1 = [l6_jitter._pick(set(range(0, 12)), r, 8)[0] for r in range(4, 8)]  # node 1, ranks 4-7
    assert len(set(picks1)) == 4


def test_mainpin_rescans_on_consecutive_on_windows_and_reserves_for_every_rank(jt):
    import threading

    setup, lines, mp = jt
    allowed = os.sched_getaffinity(0)
    assert len(allowed) >= 4, allowed
    from vllm_torchtpu.core import l6_ab

    stop = threading.Event()
    w = threading.Thread(target=stop.wait, daemon=True)
    w.start()
    try:
        ab, jit = setup(VLLM_TPU_L6_AB_STEPS=str(S), VLLM_TPU_L6_AB_CONFIRM="1", VLLM_TPU_L6_AB_CLASSES="4",
                        VLLM_TPU_L6_MAIN_PIN="ab:1,2")
        top = sorted(allowed)[-3:]
        main = threading.get_native_id()
        ab.begin_step(); jit.on_step()  # l6_ab caches its own (rank, world) = (0, 1) here: no lockstep files
        # mainpin asks torch.distributed for the rank on every pin: rank 1 of 3 -> second-highest CPU
        mp.setattr(l6_ab, "_rank_world", lambda: (1, 3))
        for _ in range(S - 2 + S + 1):  # rest of window 0, window 1, first step of window 2 (consecutive on-windows)
            ab.begin_step(); jit.on_step()
        assert _aff(main) == {sorted(allowed)[-2]}
        assert not set(top) & _aff(w.native_id), _aff(w.native_id)
        assert not any(ln.startswith("ERROR q38 L6 mainpin off") for ln in lines), lines
    finally:
        stop.set()


# ================================================================ L6-SYNC: SYNC-GC (syncgc)


@pytest.fixture
def sg(monkeypatch):
    """syncgc: real gc module, the propose hook stubbed (its own test below), every log line kept."""
    from vllm_torchtpu.core import l6_ab, l6_jitter

    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS",
                                                  "VLLM_TPU_SERVED_AB"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(l6_ab, "_rank_world", lambda: (0, 1))
    lines = []
    for mod in (l6_ab, l6_jitter):
        monkeypatch.setattr(mod.logger, "info", lambda fmt, *a: lines.append(fmt % a))
        monkeypatch.setattr(mod.logger, "warning", lambda fmt, *a: lines.append("WARNING " + fmt % a))
        monkeypatch.setattr(mod.logger, "error", lambda fmt, *a: lines.append("ERROR " + fmt % a))
    monkeypatch.setattr(l6_jitter, "_install_sync_point", lambda cls=None: True)
    monkeypatch.setattr(l6_jitter, "SYNC_EVERY", 100)
    monkeypatch.setattr(l6_jitter, "SYNC_VALVE", 10**9)
    monkeypatch.setattr(l6_jitter, "D1E_FREEZE", False)
    th = gc.get_threshold()
    gc.enable()
    gc.unfreeze()
    l6_jitter.restore()

    def setup(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        return l6_ab, l6_jitter

    yield setup, lines, monkeypatch
    l6_jitter.restore()
    gc.enable()
    gc.unfreeze()
    gc.set_threshold(*th)
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def _sg_run(ab, jit, n, skip_propose=()):
    """n worker steps: begin_step, the jitter levers, then the draft propose's sync point
    (absent on the steps in skip_propose). Returns [(j, gc enabled after the step)]."""
    out = []
    for _ in range(n):
        ab.begin_step()
        jit.on_step()
        if ab.step_seq() not in skip_propose:
            jit.sync_point()
        out.append((ab.step_seq(), gc.isenabled()))
    return out


def test_syncgc_collects_only_on_the_lockstep_step_index(sg):
    """Collections run at l6_ab step indices that are multiples of N (the first propose at or after
    one, if that step had none), whatever step the lever engaged on, so every rank pauses on the same
    step; automatic GC is off exactly in the on-windows."""
    setup, lines, mp = sg
    ab, jit = setup(VLLM_TPU_L6_AB_STEPS="8", VLLM_TPU_L6_AB_CONFIRM="1", VLLM_TPU_L6_SYNC_GC="ab:1")
    mp.setattr(jit, "SYNC_EVERY", 3)
    got = []
    mp.setattr(jit, "_gc_collect", lambda g: got.append((ab.step_seq(), g)))
    jit.mark_warm()
    st = _sg_run(ab, jit, 31, skip_propose={12})  # windows 0 (j 1-7, off), 1 (8-15, on), 2 (off), 3 (24-31, on)
    assert [j for j, _ in got] == [9, 13, 15, 24, 27, 30], got  # 12 had no propose: 13 is late, 15 is on time
    assert all(g == 0 for _, g in got)
    assert [en for j, en in st] == [not (8 <= j <= 15 or 24 <= j <= 31) for j, _ in st], st
    assert not any(ln.startswith("ERROR") for ln in lines), lines


def test_syncgc_valve_fires_above_its_bound(sg):
    setup, lines, mp = sg
    ab, jit = setup(VLLM_TPU_L6_SYNC_GC="1")
    mp.setattr(jit, "SYNC_EVERY", 10**6)  # no lockstep collection in this test
    mp.setattr(jit, "SYNC_VALVE", 5000)
    jit.mark_warm()
    ab.begin_step(); jit.on_step()
    assert not gc.isenabled()
    small = [[] for _ in range(50)]
    jit.sync_point()
    assert "syncgc.valve" not in ab._state["counts"]
    junk = []
    for _ in range(10000):
        x = []
        x.append(x)  # a reference cycle: only the cyclic GC frees it
        junk.append(x)
    del junk, x
    assert gc.get_count()[0] > 5000
    jit.sync_point()
    assert ab._state["counts"].get("syncgc.valve") == 1, ab._state["counts"]
    assert gc.get_count()[0] < 5000
    assert sum(ln.startswith("WARNING q38 L6 syncgc valve #1") for ln in lines) == 1, lines
    jit.sync_point()
    assert ab._state["counts"].get("syncgc.valve") == 1
    del small


def test_syncgc_off_window_restores_gc_and_keeps_the_warmup_freeze(sg):
    """Off-window: automatic GC back on, D1e's threshold (served: threshold mode) untouched, and the
    worker's warm-up freeze (vLLM freeze_gc_heap) never thawed."""
    setup, lines, mp = sg

    class Warm:
        pass

    def frozen(o):  # in the permanent generation: tracked, but in no generation gc.get_objects() lists
        return gc.is_tracked(o) and not any(x is o for x in gc.get_objects())

    keep = [Warm() for _ in range(200)]
    gc.freeze()  # stands in for the worker's freeze_gc_heap() at the end of warm-up
    base = gc.get_freeze_count()
    assert base > 0 and frozen(keep[0])
    gc.set_threshold(50000, 10, 10)  # D1e VLLM_TPU_HOST_GC=threshold, as served
    ab, jit = setup(VLLM_TPU_L6_AB_STEPS="4", VLLM_TPU_L6_AB_CONFIRM="1", VLLM_TPU_L6_SYNC_GC="ab:1")
    jit.mark_warm()
    st = _sg_run(ab, jit, 3)  # window 0: off
    assert all(en for _, en in st)
    st = _sg_run(ab, jit, 4)  # window 1: on
    assert not any(en for _, en in st)
    frozen_on = gc.get_freeze_count()
    assert frozen_on >= base
    st = _sg_run(ab, jit, 1)  # first step of window 2: off
    assert st[0][1] is True
    # not thawed (gc.unfreeze would move it back to generation 2); the count only drops by the few
    # frozen objects freed by refcount meanwhile
    assert frozen(keep[0]) and gc.get_freeze_count() > base // 2
    assert gc.get_threshold() == (50000, 10, 10)
    assert ab._state["counts"].get("syncgc.off") == 1
    st = _sg_run(ab, jit, 3 + 4)  # rest of window 2, window 3 (on again: the freeze is not repeated)
    assert [en for _, en in st] == [True] * 3 + [False] * 4
    assert ab._state["counts"].get("syncgc.freeze") is None
    del keep


def test_syncgc_error_restores_gc_once_and_stays_off(sg):
    setup, lines, mp = sg
    ab, jit = setup(VLLM_TPU_L6_SYNC_GC="1")
    mp.setattr(jit, "SYNC_EVERY", 2)

    def boom(g):
        raise RuntimeError("injected collect failure")

    mp.setattr(jit, "_gc_collect", boom)
    jit.mark_warm()
    st = _sg_run(ab, jit, 1)  # j=1: engaged, nothing due
    assert st == [(1, False)]
    st = _sg_run(ab, jit, 4)  # j=2: the collect fails -> gc back on, lever off for good
    assert all(en for _, en in st), st
    assert sum(ln.startswith("ERROR q38 L6 syncgc off for the rest of the run (automatic gc re-enabled)")
               for ln in lines) == 1, lines


@pytest.mark.parametrize("case", ["d1e_freeze", "gcfreeze"])
def test_syncgc_refused_next_to_another_freeze_owner(sg, case):
    setup, lines, mp = sg
    env = {"VLLM_TPU_L6_SYNC_GC": "1"}
    if case == "gcfreeze":
        env["VLLM_TPU_L6_GC_FREEZE"] = "ab:1"
        env.update(VLLM_TPU_L6_AB_STEPS="4", VLLM_TPU_L6_AB_CONFIRM="1")
    ab, jit = setup(**env)
    if case == "d1e_freeze":
        mp.setattr(jit, "D1E_FREEZE", True)
    jit.mark_warm()
    st = _sg_run(ab, jit, 3)
    assert all(en for _, en in st)
    assert sum(ln.startswith("ERROR q38 L6 syncgc refused") for ln in lines) == 1, lines


def test_syncgc_needs_gc_enabled_when_it_engages(sg):
    setup, lines, mp = sg
    ab, jit = setup(VLLM_TPU_L6_SYNC_GC="1")
    jit.mark_warm()
    gc.disable()  # someone else owns a disabled gc
    try:
        _sg_run(ab, jit, 2)
        assert not gc.isenabled()  # left as found, not re-enabled by syncgc
        assert sum("gc is already disabled" in ln for ln in lines if ln.startswith("ERROR")) == 1, lines
    finally:
        gc.enable()


def test_syncgc_escalates_with_cpython_counters(sg):
    """A collection reaches gen 1 / gen 2 when CPython's own counters exceed thresholds 1 / 2, so
    objects that survived a gen-0 pass are still reclaimed with automatic GC off."""
    setup, lines, mp = sg
    ab, jit = setup(VLLM_TPU_L6_SYNC_GC="1")
    mp.setattr(jit, "SYNC_EVERY", 1)
    gens = []
    real = gc.collect
    mp.setattr(jit, "_gc_collect", lambda g: (gens.append(g), real(g)))
    gc.set_threshold(700, 2, 2)
    real()  # every generation count back to 0
    jit.mark_warm()
    _sg_run(ab, jit, 16)
    # CPython checks generation 2 first: gen 2 right after the third gen-1 collection (count 2 = 3 > 2)
    assert gens == [0, 0, 0, 1] * 3 + [2, 0, 0, 0], gens


def test_syncgc_times_every_collection_in_the_window_line(sg):
    setup, lines, mp = sg
    ab, jit = setup(VLLM_TPU_L6_AB_STEPS="4", VLLM_TPU_L6_AB_CONFIRM="1", VLLM_TPU_L6_SYNC_GC="ab:1")
    ab.begin_step(); jit.on_step()  # an off window, before warm-up, no D1e AB period: the timer is in
    assert jit._gc_timer in gc.callbacks
    gc.collect()
    c = ab._state["counts"]
    assert c.get("gc.n", 0) >= 1 and c.get("gc.g2", 0) >= 1 and c.get("gc.us", -1) >= 0, c


def test_syncgc_waits_for_warm_up(sg):
    setup, lines, mp = sg
    ab, jit = setup(VLLM_TPU_L6_SYNC_GC="1")
    st = _sg_run(ab, jit, 3)
    assert all(en for _, en in st)
    jit.mark_warm()
    st = _sg_run(ab, jit, 1)
    assert st[0][1] is False


# The real hook installer, captured before the sg fixture stubs the module attribute.
from vllm_torchtpu.core import l6_jitter as _l6j_at_import  # noqa: E402

_INSTALL_REAL = _l6j_at_import._install_sync_point


def test_syncgc_hook_runs_before_propose(sg):
    setup, lines, mp = sg
    ab, jit = setup(VLLM_TPU_L6_SYNC_GC="1")
    order = []

    class Drafter:
        def propose(self, x):
            order.append("propose")
            return x + 1

    mp.setattr(jit, "sync_point", lambda: order.append("sync"))
    assert _INSTALL_REAL(Drafter) and _INSTALL_REAL(Drafter)  # idempotent: one wrapper
    jit._sg["armed"] = True
    assert Drafter().propose(1) == 2
    assert order == ["sync", "propose"], order
    jit._sg["armed"] = False
    order.clear()
    Drafter().propose(1)
    assert order == ["propose"], order


def test_syncgc_hook_wraps_the_eagle3_proposer(monkeypatch):
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    orig = Eagle3Proposer.propose
    monkeypatch.setattr(Eagle3Proposer, "propose", orig)  # undone at teardown
    assert _INSTALL_REAL()
    assert getattr(Eagle3Proposer.propose, "_q38_syncgc", False)
    assert Eagle3Proposer.propose.__wrapped__ is orig


def test_worker_marks_warm_after_the_cpu_pin():
    import pathlib

    from vllm_torchtpu.core import l6_jitter

    src = (pathlib.Path(l6_jitter.__file__).resolve().parents[1] / "worker" / "tpu_worker.py").read_text()
    pin = src.index("cpu_pinning.apply(self.rank, self.local_rank)")
    warm = src.index("l6_jitter.mark_warm()")
    assert pin < warm < src.index("return CompilationTimes(language_model=compilation_time", pin)


def test_new_levers_are_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED
    from vllm_torchtpu.core import l6_ab

    assert l6_ab.LEVERS["syncgc"] == "VLLM_TPU_L6_SYNC_GC"
    assert l6_ab.LEVERS["mainpin2"] == "VLLM_TPU_L6_MAIN_PIN2"
    for name in ("VLLM_TPU_L6_SYNC_GC", "VLLM_TPU_L6_SYNC_GC_EVERY", "VLLM_TPU_L6_SYNC_GC_VALVE",
                 "VLLM_TPU_L6_MAIN_PIN2", "VLLM_TPU_L6_NUMA_BIND"):
        assert name in envs.environment_variables and name in _TPU_COMPILE_ENV_IGNORED, name


# ================================================================ L6-SYNC: MAINPIN-2 (mainpin2)

# Fake host: 2 NUMA nodes x 12 physical cores x 2 SMT siblings (CPU c and c + 24), like a v7x host's
# 56 cores per node with sibling c + 112 (tests/worker/test_cpu_pinning.py), scaled down.
NODE0 = set(range(0, 12)) | set(range(24, 36))
NODE1 = set(range(12, 24)) | set(range(36, 48))


def _fake_sysfs(root):
    from vllm_torchtpu.core.l6_jitter import fmt_cpulist

    for n, cpus in ((0, NODE0), (1, NODE1)):
        d = root / "node" / f"node{n}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(fmt_cpulist(cpus) + "\n")
    for c in range(48):
        t = root / "cpu" / f"cpu{c}" / "topology"
        t.mkdir(parents=True)
        base = c % 24
        (t / "thread_siblings_list").write_text(f"{base},{base + 24}\n")
    (root / "cpu" / "online").write_text("0-47\n")
    return str(root)


class FakeAff:
    """Affinity calls on a fake thread table. A request made only of `deny` CPUs (outside the cgroup)
    fails with EINVAL, as the kernel's does."""

    def __init__(self, masks, deny=()):
        self.masks = {t: set(m) for t, m in masks.items()}
        self.orig = {t: set(m) for t, m in masks.items()}
        self.deny = set(deny)
        self.calls = []

    def get(self, tid):
        if tid not in self.masks:
            raise ProcessLookupError(tid)
        return set(self.masks[tid])

    def set(self, tid, cpus):
        cpus = set(cpus)
        self.calls.append((tid, frozenset(cpus)))
        if tid not in self.masks:
            raise ProcessLookupError(tid)
        if not cpus - self.deny:
            raise OSError(22, "Invalid argument")
        self.masks[tid] = cpus

    def tids(self):
        return list(self.masks)


MAIN = 100


@pytest.fixture
def p2(monkeypatch, tmp_path):
    """mainpin2 on a fake sysfs tree and fake affinity calls (no real sysfs, no real affinity)."""
    from vllm_torchtpu.core import l6_ab, l6_jitter

    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS",
                                                  "VLLM_TPU_SERVED_AB"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(l6_ab, "_rank_world", lambda: (0, 1))
    lines = []
    for mod in (l6_ab, l6_jitter):
        monkeypatch.setattr(mod.logger, "info", lambda fmt, *a: lines.append(fmt % a))
        monkeypatch.setattr(mod.logger, "error", lambda fmt, *a: lines.append("ERROR " + fmt % a))
    monkeypatch.setattr(l6_jitter, "SYS", _fake_sysfs(tmp_path / "system"))
    monkeypatch.setattr(l6_jitter, "CPU_PIN_MODE", "numa")
    monkeypatch.setattr(l6_jitter, "_main_tid", lambda: MAIN)
    monkeypatch.setattr(l6_jitter, "_chip_node", lambda: None)
    l6_jitter.restore()
    state = {}

    def setup(masks, deny=(), **env):
        fa = FakeAff(masks, deny)
        monkeypatch.setattr(l6_jitter, "_getaff", fa.get)
        monkeypatch.setattr(l6_jitter, "_setaff", fa.set)
        monkeypatch.setattr(l6_jitter, "_tids", fa.tids)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        # l6_ab's own window lockstep stays single-rank (no fingerprint files); mainpin2 reads the
        # rank through l6_ab._rank_world() itself, which a test may patch.
        l6_ab._state["rank_world"] = (0, 1)
        state["fa"] = fa
        return l6_ab, l6_jitter, fa

    yield setup, lines, monkeypatch
    l6_jitter.restore()  # the fakes are still in place here
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def _p2_steps(ab, jit, n):
    for _ in range(n):
        ab.begin_step()
        jit.on_step()


def _others_calls(fa, since=0):
    return [c for c in fa.calls[since:] if c[0] != MAIN]


def test_mainpin2_plan_gives_distinct_cores_with_idle_siblings(p2):
    setup, lines, mp = p2
    ab, jit, fa = setup({MAIN: NODE0})
    picks = {r: jit.plan2(0 if r < 4 else 1, r, 8) for r in range(8)}  # ranks 0-3 on node 0, 4-7 on node 1
    mains = [cpu for cpu, _, _ in picks.values()]
    assert len(set(mains)) == 8, picks
    assert mains[:4] == [11, 10, 9, 8] and mains[4:] == [19, 18, 17, 16]
    for r, (cpu, idle, reserved) in picks.items():
        assert idle == {cpu + 24}, (r, idle)  # its SMT sibling
        assert cpu in reserved and idle <= reserved
        assert not idle & set(mains)  # no rank's main thread sits on another core's sibling
    res0 = picks[0][2]
    assert res0 == set(range(4, 12)) | set(range(28, 36))  # the 8 highest cores of node 0, both siblings
    assert jit.fmt_cpulist(jit.complement(8)) == "0-3,12-15,24-27,36-39"
    with pytest.raises(OSError):
        jit.plan2(0, 0, 12)  # 12 reserved cores need 13 on the node


def test_mainpin2_complement_cli(p2, capsys):
    setup, lines, mp = p2
    ab, jit, fa = setup({MAIN: NODE0})
    assert jit._main(["complement", "8"]) == 0
    assert capsys.readouterr().out.strip() == "0-3,12-15,24-27,36-39"
    assert jit._main(["complement"]) == 2


def test_mainpin2_static_exclusion_once_only_the_main_mask_toggles(p2):
    setup, lines, mp = p2
    others = {t: set(NODE0) for t in range(101, 106)}
    ab, jit, fa = setup({MAIN: NODE0, **others}, VLLM_TPU_L6_AB_STEPS="4", VLLM_TPU_L6_AB_CONFIRM="1",
                        VLLM_TPU_L6_MAIN_PIN2="ab:1")
    ab.begin_step(); jit.on_step()  # j=1, not warm: nothing
    assert fa.calls == []
    mp.setattr(ab, "_rank_world", lambda: (1, 8))  # mainpin2 reads the torch.distributed rank: 1 of 8
    jit.mark_warm()
    reserved = set(range(4, 12)) | set(range(28, 36))
    _p2_steps(ab, jit, 2)  # j=2-3, window 0 (off): the static exclusion, once, in an off window
    assert ab._state["counts"].get("mainpin2.excl") == 5
    assert {t for t, _ in fa.calls} == set(others)
    for t in others:
        assert fa.masks[t] == NODE0 - reserved, t
    assert fa.masks[MAIN] == NODE0
    n0 = len(fa.calls)
    _p2_steps(ab, jit, 4)  # window 1 (on)
    assert fa.masks[MAIN] == {10}  # rank 1 -> the second-highest core (10, sibling 34 idle)
    assert all(34 not in fa.masks[t] and 10 not in fa.masks[t] for t in others)
    assert ab._state["counts"].get("mainpin2.pinned") == 4 and ab._state["counts"].get("mainpin2.toggle") == 1
    _p2_steps(ab, jit, 4)  # window 2 (off): only the main mask goes back
    assert fa.masks[MAIN] == NODE0
    assert all(fa.masks[t] == NODE0 - reserved for t in others)
    _p2_steps(ab, jit, 4)  # window 3 (on)
    assert fa.masks[MAIN] == {10}
    assert _others_calls(fa, n0) == [], _others_calls(fa, n0)  # known threads never touched again
    assert [c for c in fa.calls[n0:]] == [(MAIN, frozenset({10})), (MAIN, frozenset(NODE0)),
                                         (MAIN, frozenset({10}))]
    assert not any(ln.startswith("ERROR") for ln in lines), lines
    jit.restore()
    assert fa.masks == fa.orig  # every saved mask back


def test_mainpin2_new_threads_are_excluded_at_the_next_toggle(p2):
    setup, lines, mp = p2
    ab, jit, fa = setup({MAIN: NODE0, 101: NODE0}, VLLM_TPU_L6_AB_STEPS="4", VLLM_TPU_L6_AB_CONFIRM="1",
                        VLLM_TPU_L6_MAIN_PIN2="ab:1")
    mp.setattr(ab, "_rank_world", lambda: (0, 8))
    jit.mark_warm()
    reserved = set(range(4, 12)) | set(range(28, 36))
    _p2_steps(ab, jit, 3 + 1)  # window 0, first step of window 1: main pinned to CPU 11
    assert fa.masks[MAIN] == {11}
    fa.masks[102] = {11}  # created by the main thread while pinned: inherits its CPU
    fa.orig[102] = set(NODE0)  # its restore target: the process's allowed set, not CPU 11
    n0 = len(fa.calls)
    _p2_steps(ab, jit, 3 + 1)  # rest of window 1, first step of window 2 (toggle off)
    assert fa.masks[102] == NODE0 - reserved
    assert ab._state["counts"].get("mainpin2.newthr") == 1
    assert [t for t, _ in _others_calls(fa, n0)] == [102]  # 101 is not touched again
    jit.restore()
    assert fa.masks == fa.orig


def test_mainpin2_failure_restores_every_mask_once(p2):
    setup, lines, mp = p2
    others = {t: set(NODE0) for t in range(101, 104)}
    ab, jit, fa = setup({MAIN: NODE0, **others}, deny={11}, VLLM_TPU_L6_AB_STEPS="4",
                        VLLM_TPU_L6_AB_CONFIRM="1", VLLM_TPU_L6_MAIN_PIN2="ab:1")
    mp.setattr(ab, "_rank_world", lambda: (0, 8))
    jit.mark_warm()
    _p2_steps(ab, jit, 3)  # window 0: excluded
    assert all(fa.masks[t] != NODE0 for t in others)
    _p2_steps(ab, jit, 1)  # window 1: pinning CPU 11 is refused by the (fake) cgroup
    assert fa.masks == fa.orig
    assert ab._state["counts"].get("mainpin2.fail") == 1
    n0 = len(fa.calls)
    _p2_steps(ab, jit, 3 + 4 + 4)
    assert fa.calls[n0:] == [] and fa.masks == fa.orig
    assert sum(ln.startswith("ERROR q38 L6 mainpin2 off for the rest of the run") for ln in lines) == 1, lines


@pytest.mark.parametrize("case", ["mainpin", "cpu_pin_slice"])
def test_mainpin2_refused_next_to_mainpin_and_cpu_pin_slice(p2, case):
    setup, lines, mp = p2
    env = {"VLLM_TPU_L6_MAIN_PIN2": "1"}
    if case == "mainpin":
        env["VLLM_TPU_L6_MAIN_PIN"] = "ab:1"
        env.update(VLLM_TPU_L6_AB_STEPS="4", VLLM_TPU_L6_AB_CONFIRM="1")
    ab, jit, fa = setup({MAIN: NODE0, 101: NODE0}, **env)
    if case == "cpu_pin_slice":
        mp.setattr(jit, "CPU_PIN_MODE", "slice")
    jit.mark_warm()
    _p2_steps(ab, jit, 3)
    assert fa.calls == [] and fa.masks == fa.orig
    assert sum(ln.startswith("ERROR q38 L6 mainpin2 refused") for ln in lines) == 1, lines


def test_mainpin2_waits_for_warm_up(p2):
    setup, lines, mp = p2
    ab, jit, fa = setup({MAIN: NODE0, 101: NODE0}, VLLM_TPU_L6_MAIN_PIN2="1")
    _p2_steps(ab, jit, 3)
    assert fa.calls == []
    jit.mark_warm()
    _p2_steps(ab, jit, 1)
    assert fa.masks[MAIN] == {11}


def test_mainpin2_takes_the_chip_node_when_the_allowed_set_spans_nodes(p2):
    setup, lines, mp = p2
    every = NODE0 | NODE1
    ab, jit, fa = setup({MAIN: every, 101: every}, VLLM_TPU_L6_MAIN_PIN2="1")
    mp.setattr(jit, "_chip_node", lambda: 1)
    jit.mark_warm()
    _p2_steps(ab, jit, 1)
    assert fa.masks[MAIN] == {23}  # rank 0 of 1: node 1's highest core
    assert 47 not in fa.masks[101] and 23 not in fa.masks[101]
    jit.restore()
    mp.setattr(jit, "_chip_node", lambda: None)  # chip node unknown: refuses to guess, restores
    fa.calls.clear()
    jit.mark_warm()
    ab._resolve()
    ab._state["rank_world"] = (0, 1)
    _p2_steps(ab, jit, 2)
    assert fa.masks == fa.orig
    assert sum("cannot tell this worker's NUMA node" in ln for ln in lines if ln.startswith("ERROR")) == 1


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
