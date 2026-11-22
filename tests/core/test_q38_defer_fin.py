# SPDX-License-Identifier: Apache-2.0
"""A finished request's block free and teardown run in the engine core's waits.

See `vllm_torchtpu.core.defer_fin` (q38 SV-EC DEFER-FIN). With the key off nothing is patched. With it on, a
``_free_request_blocks`` call made inside ``update_from_output`` on the main thread is queued with the request (so the
request outlives ``update_from_output``); calls outside ``update_from_output`` or off the main thread free at once. The
queue is worked off once per ``SpinCondition.wait()`` after the main thread's wait has lasted ``OPEN_AFTER_NS``, all at
once before the main thread blocks on an empty input queue, and at ``schedule()`` for entries ``MAX_LAG`` schedules old
or whose request id is back; every request is freed exactly once, in order, by vLLM's own ``_free_request_blocks``.
"""

import ast
import pathlib
import queue
import threading
import time
import types
import weakref

import pytest
from vllm.distributed.device_communicators import shm_broadcast
from vllm.distributed.device_communicators.shm_broadcast import SpinCondition
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.request import Request

from vllm_torchtpu import envs
from vllm_torchtpu.core import defer_fin as df

KEYS = ("VLLM_TPU_DEFER_FIN", "VLLM_TPU_DEFER_FIN_AB_MS")
STOCK_FREE = Scheduler.__dict__["_free_request_blocks"]


class _KVM:
    """Records which request ids the KV cache manager frees (ids only, so it holds no request)."""

    def __init__(self):
        self.freed = []

    def free(self, request):
        self.freed.append(request.request_id)


def _sched():
    s = object.__new__(Scheduler)
    s.defer_block_free = False
    s.kv_cache_manager = _KVM()
    s.requests = {}
    return s


def _req(i, n=2000):
    return Request("req-%d" % i, list(range(n)), SamplingParams(temperature=0.0, max_tokens=4), None)


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(df, "ENABLED", True)
    monkeypatch.setattr(df, "AB_MS", 0)
    monkeypatch.setattr(df, "stats", {k: 0 for k in df.stats})
    monkeypatch.setattr(df, "_state", {"in_update": 0, "seq": 0, "ab_window": None})
    monkeypatch.setitem(df._ORIG, "free", STOCK_FREE)
    df.PENDING.clear()
    yield
    df.PENDING.clear()


def _free_w():
    return df.make_free_request_blocks(STOCK_FREE)


def _update(free_w):
    """A stand-in update_from_output: builds requests in its own frame and frees them, like vLLM's finish path."""

    refs = []

    def update_from_output(self, n, start=0):
        reqs = [_req(start + i) for i in range(n)]
        refs.extend(weakref.ref(r) for r in reqs)
        for r in reqs:
            free_w(self, r)
        return None

    return df.make_update_from_output(update_from_output), refs


def test_key_off_patches_nothing(fresh, monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    assert envs.VLLM_TPU_DEFER_FIN is False and envs.VLLM_TPU_DEFER_FIN_AB_MS == 0
    monkeypatch.setattr(df, "ENABLED", False)
    before = (Scheduler._free_request_blocks, Scheduler.update_from_output, Scheduler.schedule,
              SpinCondition.wait, SpinCondition.record_read, EngineCoreProc.__init__)
    df.patch_defer_fin()
    after = (Scheduler._free_request_blocks, Scheduler.update_from_output, Scheduler.schedule,
             SpinCondition.wait, SpinCondition.record_read, EngineCoreProc.__init__)
    assert all(x is y for x, y in zip(before, after))


def test_key_on_installs_once(fresh, monkeypatch):
    for owner, name in ((Scheduler, "_free_request_blocks"), (Scheduler, "update_from_output"), (Scheduler, "schedule"),
                        (SpinCondition, "wait"), (SpinCondition, "record_read"), (EngineCoreProc, "__init__")):
        monkeypatch.setattr(owner, name, owner.__dict__[name])
    orig = (Scheduler._free_request_blocks, Scheduler.update_from_output, Scheduler.schedule)
    df.patch_defer_fin()
    df.patch_defer_fin()
    for fn in (Scheduler._free_request_blocks, Scheduler.update_from_output, Scheduler.schedule,
               SpinCondition.wait, SpinCondition.record_read, EngineCoreProc.__init__):
        assert getattr(fn, df._PATCH_FLAG, False)
    assert (Scheduler._free_request_blocks.__wrapped__, Scheduler.update_from_output.__wrapped__,
            Scheduler.schedule.__wrapped__) == orig
    assert df._ORIG["free"] is orig[0]


def test_refused_with_the_spinloop_extension(fresh, monkeypatch):
    monkeypatch.setattr(shm_broadcast, "SPINLOOP_EXT_ENABLED", True, raising=False)
    monkeypatch.setattr(Scheduler, "_free_request_blocks", STOCK_FREE)
    df.patch_defer_fin()
    assert Scheduler._free_request_blocks is STOCK_FREE


def test_frees_inside_update_are_queued_and_outside_are_immediate(fresh):
    s, free_w = _sched(), _free_w()
    upd, _ = _update(free_w)
    upd(s, 3)
    assert s.kv_cache_manager.freed == [] and len(df.PENDING) == 3 and df.stats["deferred"] == 3
    free_w(s, _req(9))  # e.g. a preemption or an abort: not deferred
    assert s.kv_cache_manager.freed == ["req-9"] and len(df.PENDING) == 3


def test_off_main_thread_frees_immediately(fresh):
    s, free_w = _sched(), _free_w()
    upd, _ = _update(free_w)
    t = threading.Thread(target=upd, args=(s, 2))
    t.start()
    t.join()
    assert s.kv_cache_manager.freed == ["req-0", "req-1"] and not df.PENDING


def test_request_outlives_update_until_worked_off(fresh, monkeypatch):
    s, free_w = _sched(), _free_w()
    upd, refs = _update(free_w)
    upd(s, 2)
    assert all(r() is not None for r in refs)  # the teardown moved out of update_from_output
    df.flush_all()
    assert all(r() is None for r in refs) and s.kv_cache_manager.freed == ["req-0", "req-1"]
    monkeypatch.setattr(df, "AB_MS", 1000)
    monkeypatch.setattr(df.time, "time_ns", lambda: 1_500_000_000)  # odd window: the stock path
    upd2, refs2 = _update(free_w)
    upd2(s, 1, start=5)
    assert refs2[0]() is None and s.kv_cache_manager.freed[-1] == "req-5" and not df.PENDING


def test_long_main_thread_wait_works_off_one_entry_per_call(fresh, monkeypatch):
    s, free_w = _sched(), _free_w()
    upd, _ = _update(free_w)
    upd(s, 3)
    t0 = 10_000_000
    now = [t0]
    monkeypatch.setattr(df.time, "monotonic_ns", lambda: now[0])
    cond = types.SimpleNamespace()
    df.note_wait(cond)
    now[0] = t0 + 1_900_000
    df.note_wait(cond)
    assert s.kv_cache_manager.freed == []
    now[0] = t0 + 2_100_000
    df.note_wait(cond)
    assert s.kv_cache_manager.freed == ["req-0"]
    df.note_wait(cond)
    assert s.kv_cache_manager.freed == ["req-0", "req-1"]
    df.note_read(cond)
    now[0] = t0 + 3_000_000
    df.note_wait(cond)
    now[0] = t0 + 4_000_000
    df.note_wait(cond)
    assert s.kv_cache_manager.freed == ["req-0", "req-1"]  # a new wait starts its own clock
    other = types.SimpleNamespace()

    def off_main():
        df.note_wait(other)
        now[0] = t0 + 90_000_000
        df.note_wait(other)

    t = threading.Thread(target=off_main)
    t.start()
    t.join()
    assert s.kv_cache_manager.freed == ["req-0", "req-1"] and len(df.PENDING) == 1
    assert df.stats["in_wait"] == 2


def test_everything_is_worked_off_before_blocking_on_an_empty_input_queue(fresh):
    s, free_w = _sched(), _free_w()
    upd, _ = _update(free_w)
    q = queue.Queue()
    df.wrap_input_queue(q)
    df.wrap_input_queue(q)  # idempotent
    upd(s, 2)
    q.put("x")
    assert q.get() == "x" and len(df.PENDING) == 2  # not empty: no flush
    with pytest.raises(queue.Empty):
        q.get_nowait()
    assert len(df.PENDING) == 2  # non-blocking: no flush
    threading.Timer(0.2, q.put, args=("y",)).start()
    assert q.get() == "y"
    assert not df.PENDING and s.kv_cache_manager.freed == ["req-0", "req-1"] and df.stats["idle"] == 2


def test_schedule_works_off_entries_max_lag_old(fresh):
    s, free_w = _sched(), _free_w()
    sch = df.make_schedule(lambda self: "SO")
    upd, _ = _update(free_w)
    upd(s, 1)
    for i in range(df.MAX_LAG - 1):
        assert sch(s) == "SO"
        assert s.kv_cache_manager.freed == []
    assert sch(s) == "SO"
    assert s.kv_cache_manager.freed == ["req-0"] and df.stats["stale"] == 1


def test_schedule_works_off_a_readded_request_id(fresh):
    s, free_w = _sched(), _free_w()
    sch = df.make_schedule(lambda self: "SO")
    upd, _ = _update(free_w)
    upd(s, 3)
    s.requests["req-1"] = object()
    sch(s)
    assert s.kv_cache_manager.freed == ["req-1"] and df.stats["readded"] == 1
    assert [r.request_id for _, r, _ in df.PENDING] == ["req-0", "req-2"]


def test_each_request_is_freed_once_in_order(fresh):
    s, free_w = _sched(), _free_w()
    upd, _ = _update(free_w)
    upd(s, 5)
    df.flush_all()
    df.flush_all()
    assert s.kv_cache_manager.freed == ["req-%d" % i for i in range(5)]


def test_ab_windows_follow_the_wall_clock(fresh, monkeypatch):
    monkeypatch.setattr(df, "AB_MS", 1000)
    assert df.ab_on(0) and not df.ab_on(1_500_000_000) and df.ab_on(2_100_000_000)
    s, free_w = _sched(), _free_w()
    monkeypatch.setattr(df.time, "time_ns", lambda: 1_500_000_000)
    upd, _ = _update(free_w)
    upd(s, 2)
    assert s.kv_cache_manager.freed == ["req-0", "req-1"] and not df.PENDING and df.stats["stock_ab"] == 2
    monkeypatch.setattr(df.time, "time_ns", lambda: 2_100_000_000)
    upd(s, 1, start=7)
    assert len(df.PENDING) == 1 and df.stats["deferred"] == 1


def test_spin_condition_wrappers_note_first(fresh, monkeypatch):
    seen = []
    monkeypatch.setattr(SpinCondition, "wait", lambda self, timeout_ms=None: seen.append(("wait", timeout_ms)))
    monkeypatch.setattr(SpinCondition, "record_read", lambda self: seen.append("read"))
    for owner, name in ((Scheduler, "_free_request_blocks"), (Scheduler, "update_from_output"), (Scheduler, "schedule"),
                        (EngineCoreProc, "__init__")):
        monkeypatch.setattr(owner, name, owner.__dict__[name])
    monkeypatch.setattr(df, "note_wait", lambda c: seen.append("note_wait"))
    monkeypatch.setattr(df, "note_read", lambda c: seen.append("note_read"))
    df.patch_defer_fin()
    SpinCondition.wait(object(), 5)
    SpinCondition.record_read(object())
    assert seen == ["note_wait", ("wait", 5), "note_read", "read"]


def test_engine_core_logs_the_engagement_line_once_built(fresh, monkeypatch):
    lines = []
    monkeypatch.setattr(df.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    for owner, name in ((Scheduler, "_free_request_blocks"), (Scheduler, "update_from_output"), (Scheduler, "schedule"),
                        (SpinCondition, "wait"), (SpinCondition, "record_read")):
        monkeypatch.setattr(owner, name, owner.__dict__[name])
    monkeypatch.setattr(EngineCoreProc, "__init__", lambda self, *a, **k: setattr(self, "input_queue", queue.Queue()))
    df.patch_defer_fin()
    lines.clear()
    eng = object.__new__(EngineCoreProc)
    EngineCoreProc.__init__(eng)
    assert lines == ["DEFER_FIN engaged: finished requests freed in the engine core's waits "
                     "(after 2.0 ms of a wait, at idle, or within 2 schedules; AB 0 ms)"]
    assert getattr(eng.input_queue.get, df._PATCH_FLAG, False)


def test_keys_are_registered_and_out_of_the_compile_key():
    assert all(k in envs.environment_variables for k in KEYS)
    src = pathlib.Path(df.__file__).parents[1] / "compilation" / "tpu_compiler.py"
    tree = ast.parse(src.read_text())
    ignored = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_TPU_COMPILE_ENV_IGNORED" for t in node.targets
        ):
            ignored = {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
    assert ignored is not None and all(k in ignored for k in KEYS)


def test_registry_installs_it_at_engine_core_after_preproc():
    from vllm_torchtpu import patch_registry

    targets = [p.target for p in patch_registry.PATCHES]
    t = "vllm_torchtpu.core.defer_fin:patch_defer_fin"
    assert t in targets
    assert patch_registry.PATCHES[targets.index(t)].stages == ("engine_core",)
    assert targets.index(t) > targets.index("vllm_torchtpu.core.preproc_in_wait:patch_preproc_in_wait")
