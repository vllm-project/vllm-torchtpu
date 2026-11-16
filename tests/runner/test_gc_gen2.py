# SPDX-License-Identifier: Apache-2.0
"""Engine-core generation-2 policy: freeze point, threshold, passes at waits.

See `vllm_torchtpu.runner.gc_at_waits`.
"""

import gc
import threading
import time

import pytest
from vllm.distributed.device_communicators.shm_broadcast import SpinCondition

from vllm_torchtpu.runner import gc_at_waits


class _Sched:
    def __init__(self, n: int = 0):
        self.requests = {f"r{i}": object() for i in range(n)}
        self.running = list(self.requests)
        self.waiting = []


class _Engine:
    def __init__(self, n: int = 0):
        self.scheduler = _Sched(n)


@pytest.fixture
def fresh(monkeypatch):
    """A fresh engine-core state; restores thresholds, callbacks and freezes."""
    old_threshold = gc.get_threshold()
    old_callbacks = list(gc.callbacks)
    monkeypatch.setattr(gc_at_waits, "_G", gc_at_waits._EngineGC())
    for name in ("ENABLED", "FREEZE", "UNTRACK_TOKEN_LISTS", "CENSUS"):
        monkeypatch.setattr(gc_at_waits, name, False if name != "CENSUS" else "")
    monkeypatch.setattr(gc_at_waits, "GEN2_THRESHOLD", 0)
    monkeypatch.setattr(gc_at_waits, "GEN2_PERIOD_S", 0.0)
    yield gc_at_waits._G
    gc.set_threshold(*old_threshold)
    gc.callbacks[:] = old_callbacks
    gc.unfreeze()


class _Clock:
    def __init__(self, ns: int = 10**12):
        self.ns = ns

    def __call__(self) -> int:
        return self.ns


def _recording_collect(monkeypatch) -> list:
    calls = []
    monkeypatch.setattr(gc_at_waits.gc, "collect", lambda g=2: calls.append(g) or 0)
    return calls


def test_start_freezes_once_and_sets_policy(fresh, monkeypatch):
    monkeypatch.setattr(gc_at_waits, "FREEZE", True)
    monkeypatch.setattr(gc_at_waits, "GEN2_THRESHOLD", 1000)
    monkeypatch.setattr(gc_at_waits, "GEN2_PERIOD_S", 30.0)
    held = [object() for _ in range(10)]
    frozen_before = gc.get_freeze_count()
    engine = _Engine(3)
    gc_at_waits.start_engine_gc(engine)
    assert gc.get_freeze_count() > frozen_before
    assert gc.get_threshold()[2] == 1000
    assert fresh.period_ns == 30 * 10**9 and fresh.engine is engine
    assert gc_at_waits._gen2_callback in gc.callbacks
    assert gc_at_waits.held_requests() == 3
    # Once only: a second call (a DP subclass's super().__init__) does nothing.
    monkeypatch.setattr(gc_at_waits, "GEN2_THRESHOLD", 5)
    gc_at_waits.start_engine_gc(_Engine())
    assert gc.get_threshold()[2] == 1000 and fresh.engine is engine
    del held


def test_start_without_flags_keeps_cpython_policy(fresh):
    before = (gc.get_threshold(), gc.get_freeze_count(), list(gc.callbacks))
    gc_at_waits.start_engine_gc(_Engine())
    assert (gc.get_threshold(), gc.get_freeze_count(), list(gc.callbacks)) == before


class _Cond:
    """Stands in for a SpinCondition: carries the wait-start attribute."""


MS = 1_000_000
PERIOD = 30 * 10**9


def test_gen2_runs_in_a_wait_that_reached_2ms_after_the_period(fresh, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(gc_at_waits.time, "monotonic_ns", clock)
    calls = _recording_collect(monkeypatch)
    young = []
    monkeypatch.setattr(gc_at_waits, "collect_due", lambda: young.append(1))
    fresh.period_ns = PERIOD
    fresh.last_ns = clock.ns
    cond = _Cond()
    gc_at_waits.at_wait(cond)
    assert calls == [] and young == []  # GC_AT_WAITS off: nothing younger either
    clock.ns += PERIOD - 1
    gc_at_waits.at_wait(cond)
    assert calls == []
    clock.ns += 1
    gc_at_waits.at_wait(cond)  # the period has passed: this wait starts the clock
    assert calls == []
    clock.ns += 2 * MS - 1
    gc_at_waits.at_wait(cond)
    assert calls == []
    clock.ns += 1
    gc_at_waits.at_wait(cond)
    assert calls == [2] and fresh.last_ns == clock.ns
    gc_at_waits.at_wait(cond)
    clock.ns += 5 * MS
    gc_at_waits.at_wait(cond)
    assert calls == [2]


def test_a_read_ends_the_wait(fresh, monkeypatch):
    """Short waits separated by reads never add up to 2 ms."""
    from vllm.distributed.device_communicators.shm_broadcast import (
        SpinCondition as Real,
    )

    monkeypatch.setattr(Real, "wait", lambda self, timeout_ms=None: None)
    monkeypatch.setattr(Real, "record_read", lambda self: None)
    monkeypatch.setattr(gc_at_waits, "GEN2_PERIOD_S", 30.0)
    gc_at_waits.patch_gc_at_waits()
    assert getattr(Real.record_read, gc_at_waits._PATCH_FLAG, False)
    clock = _Clock()
    monkeypatch.setattr(gc_at_waits.time, "monotonic_ns", clock)
    calls = _recording_collect(monkeypatch)
    fresh.period_ns = PERIOD
    fresh.last_ns = clock.ns - PERIOD
    cond = object.__new__(Real)
    for _ in range(10):
        Real.wait(cond)
        clock.ns += MS
        Real.record_read(cond)
        clock.ns += 10_000
    assert calls == []
    Real.wait(cond)
    clock.ns += 2 * MS
    Real.wait(cond)
    assert calls == [2]


def test_gen2_runs_at_any_wait_after_two_periods(fresh, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(gc_at_waits.time, "monotonic_ns", clock)
    calls = _recording_collect(monkeypatch)
    fresh.period_ns = PERIOD
    fresh.last_ns = clock.ns
    clock.ns += 2 * PERIOD - 1
    gc_at_waits.at_wait(_Cond())
    assert calls == []
    clock.ns += 1
    gc_at_waits.at_wait(_Cond())
    assert calls == [2]


def test_gen2_wait_comes_before_young_collections(fresh, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(gc_at_waits.time, "monotonic_ns", clock)
    calls = _recording_collect(monkeypatch)
    young = []
    monkeypatch.setattr(gc_at_waits, "ENABLED", True)
    monkeypatch.setattr(gc_at_waits, "collect_due", lambda: young.append(1))
    fresh.period_ns = PERIOD
    fresh.last_ns = clock.ns
    cond = _Cond()
    gc_at_waits.at_wait(cond)
    assert calls == [] and young == [1]
    clock.ns += PERIOD
    gc_at_waits.at_wait(cond)  # due, wait too young: younger collections run
    assert calls == [] and young == [1, 1]
    clock.ns += 2 * MS
    gc_at_waits.at_wait(cond)
    assert calls == [2] and young == [1, 1]
    gc_at_waits.at_wait(cond)
    assert young == [1, 1, 1]


def test_automatic_gen2_restarts_the_period(fresh, monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(gc_at_waits.time, "monotonic_ns", clock)
    fresh.period_ns = PERIOD
    fresh.last_ns = clock.ns
    clock.ns += 29 * 10**9
    gc_at_waits._gen2_callback("start", {"generation": 2})
    gc_at_waits._gen2_callback("stop", {"generation": 2, "collected": 0})
    assert fresh.last_ns == clock.ns
    gc_at_waits._gen2_callback("stop", {"generation": 1, "collected": 0})
    assert fresh.last_ns == clock.ns
    calls = _recording_collect(monkeypatch)
    clock.ns += 29 * 10**9
    cond = _Cond()
    gc_at_waits.at_wait(cond)
    clock.ns += 5 * MS
    gc_at_waits.at_wait(cond)
    assert calls == []


def test_real_gen2_pass_at_wait_is_logged_and_counted(fresh, monkeypatch):
    lines = []
    monkeypatch.setattr(
        gc_at_waits.logger, "info", lambda msg, *a: lines.append(msg % a)
    )
    fresh.period_ns = 1
    fresh.last_ns = 0
    fresh.log_gen2 = True
    gc.callbacks.append(gc_at_waits._gen2_callback)
    before = gc_at_waits.counts[2]
    gc_at_waits.at_wait(_Cond())  # two periods overdue: runs at once
    assert gc_at_waits.counts[2] == before + 1
    assert any(
        line.startswith("GC_GEN2 ") and "cause=scheduled" in line for line in lines
    )
    assert fresh.cause == "automatic"


def test_gen2_only_on_the_main_thread(fresh, monkeypatch):
    calls = _recording_collect(monkeypatch)
    fresh.period_ns = 1
    fresh.last_ns = 0
    t = threading.Thread(target=gc_at_waits.at_wait, args=(_Cond(),))
    t.start()
    t.join()
    assert calls == []


def test_threshold_makes_automatic_gen2_rare(fresh):
    """With a high generation-2 threshold, generation-1 passes do not escalate."""
    gc.collect()
    gc.set_threshold(700, 10, 1000)
    passes = []

    def cb(phase, info):
        if phase == "stop":
            passes.append(info["generation"])

    gc.callbacks.append(cb)
    keep = []
    for _ in range(200):
        keep.append([[] for _ in range(800)])  # one automatic gen0 at least
    gc.callbacks.remove(cb)
    assert 2 not in passes and passes.count(1) >= 10
    del keep


def test_patch_installs_wait_hook_with_period_only(fresh, monkeypatch):
    seen = []
    monkeypatch.setattr(
        SpinCondition, "wait", lambda self, timeout_ms=None: seen.append("wait")
    )
    monkeypatch.setattr(gc_at_waits, "GEN2_PERIOD_S", 30.0)
    monkeypatch.setattr(gc_at_waits, "at_wait", lambda c: seen.append("at_wait"))
    monkeypatch.setattr(SpinCondition, "record_read", SpinCondition.record_read)
    gc_at_waits.patch_gc_at_waits()
    SpinCondition.wait(object(), 1)
    assert seen == ["at_wait", "wait"]


def test_engine_patch_wraps_engine_core_proc(fresh, monkeypatch):
    import vllm.v1.engine.core as core

    built = []

    class _Proc:
        def __init__(self, n):
            built.append(n)
            self.scheduler = _Sched(n)

    monkeypatch.setattr(core, "EngineCoreProc", _Proc)
    monkeypatch.setattr(gc_at_waits, "GEN2_THRESHOLD", 777)
    gc_at_waits.patch_engine_core_gc()
    proc = _Proc(4)
    assert built == [4] and fresh.engine is proc
    assert gc.get_threshold()[2] == 777
    wrapped = _Proc.__init__
    gc_at_waits.patch_engine_core_gc()
    assert _Proc.__init__ is wrapped


def test_engine_patch_off_without_flags(fresh, monkeypatch):
    import vllm.v1.engine.core as core

    class _Proc:
        def __init__(self):
            pass

    init = _Proc.__init__
    monkeypatch.setattr(core, "EngineCoreProc", _Proc)
    gc_at_waits.patch_engine_core_gc()
    assert _Proc.__init__ is init


def test_census_thresholds_parse_colons_and_commas(fresh, monkeypatch):
    monkeypatch.setattr(gc_at_waits, "CENSUS", "90:60,120")
    gc_at_waits.start_engine_gc(_Engine())
    assert fresh.census_at == [60, 90, 120] and fresh.log_gen2


def test_census_runs_once_per_threshold(fresh, monkeypatch):
    seen = []
    monkeypatch.setattr(
        gc_at_waits, "census", lambda label, **kw: seen.append(label) or []
    )
    calls = _recording_collect(monkeypatch)
    engine = _Engine(0)
    fresh.engine = engine
    fresh.census_at = [2, 5, 6]
    cond = _Cond()
    gc_at_waits.at_wait(cond)
    assert seen == []
    engine.scheduler = _Sched(2)
    gc_at_waits.at_wait(cond)
    gc_at_waits.at_wait(cond)
    assert seen == ["held>=2"]
    engine.scheduler = _Sched(7)  # crosses 5 and 6 at once: one census
    gc_at_waits.at_wait(cond)
    gc_at_waits.at_wait(cond)
    assert seen == ["held>=2", "held>=6"] and fresh.census_at == []
    assert calls == []


# --- token lists and the census on real vLLM requests ------------------------


def _request(n_tokens: int):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.request import Request
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    ecr = EngineCoreRequest(
        request_id="r",
        prompt_token_ids=list(range(1000, 1000 + n_tokens)),
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    ecr = MsgpackDecoder(EngineCoreRequest).decode(MsgpackEncoder().encode(ecr))
    return Request.from_engine_core_request(ecr, None)


@pytest.fixture
def untrack_patch(monkeypatch):
    from vllm.v1.request import Request

    monkeypatch.setattr(Request, "__init__", Request.__init__)
    gc_at_waits._patch_request_untrack()
    yield


def test_untrack_int_list():
    values = list(range(10_000, 20_000))
    assert gc.is_tracked(values)
    gc_at_waits.untrack_int_list(values)
    assert not gc.is_tracked(values)
    gc_at_waits.untrack_int_list(values)  # idempotent
    values.append(5)
    assert not gc.is_tracked(values) and values[-1] == 5 and len(values) == 10_001
    gc_at_waits.untrack_int_list((1, 2))  # not a list: ignored
    gc.collect()
    assert values[:2] == [10_000, 10_001]


def test_request_token_lists_untracked(untrack_patch):
    req = _request(5000)
    assert not gc.is_tracked(req.prompt_token_ids)
    assert not gc.is_tracked(req._all_token_ids)
    req.append_output_token_ids([7, 8])
    assert list(req.all_token_ids[-2:]) == [7, 8]
    assert req.num_tokens == 5002 and req.num_prompt_tokens == 5000
    assert list(req.output_token_ids) == [7, 8]
    gc.collect()
    assert req.prompt_token_ids[0] == 1000 and req._all_token_ids[4999] == 5999


def test_untracked_request_is_freed_by_refcount(untrack_patch):
    import weakref

    req = _request(5000)
    ref = weakref.ref(req)
    del req
    assert ref() is None


def test_census_names_the_token_lists(fresh, untrack_patch, monkeypatch):
    from vllm.v1.request import Request

    req_tracked = None
    monkeypatch.setattr(Request, "__init__", Request.__init__.__wrapped__)
    req_tracked = _request(6000)
    lines = gc_at_waits.census("t", time_gen2=False)
    owners = [line for line in lines if " owner: " in line]
    assert any("Request.prompt_token_ids (list)" in line for line in owners)
    assert any("Request._all_token_ids (list)" in line for line in owners)
    assert lines[0].startswith("t held=0 ") and "gen2_ms=-1.0" in lines[0]
    del req_tracked
    gc_at_waits._patch_request_untrack()
    req = _request(6000)
    lines = gc_at_waits.census("u", time_gen2=True)
    assert not any("Request.prompt_token_ids" in line for line in lines)
    assert "gen2_ms=-1.0" not in lines[0]
    del req


def test_census_failure_is_logged_not_raised(fresh, monkeypatch):
    warnings = []
    monkeypatch.setattr(gc_at_waits.logger, "warning", lambda *a: warnings.append(a))

    def boom(label, **kw):
        raise RuntimeError("census")

    monkeypatch.setattr(gc_at_waits, "census", boom)
    fresh.engine = _Engine(1)
    fresh.census_at = [1]
    gc_at_waits.at_wait(_Cond())
    assert warnings and fresh.census_at == []


def test_census_survives_odd_heap_objects(fresh):
    import weakref

    class _Thing:
        pass

    thing = _Thing()
    proxy = weakref.proxy(thing)
    holder = [proxy, [proxy]]
    del thing  # the proxy is dead: o.__class__ raises ReferenceError
    lines = gc_at_waits.census("odd", time_gen2=False)
    assert lines[0].startswith("odd held=0 ")
    del holder
