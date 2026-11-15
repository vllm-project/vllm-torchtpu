# SPDX-License-Identifier: Apache-2.0
"""Due garbage collections run at waits, on the main thread only.

See `vllm_torchtpu.runner.gc_at_waits`.
"""

import gc
import threading

import pytest
from vllm.distributed.device_communicators.shm_broadcast import SpinCondition

from vllm_torchtpu.runner import gc_at_waits


@pytest.fixture
def thresholds():
    old = gc.get_threshold()
    gc.collect()
    gc.set_threshold(700, 10, 10)
    yield
    gc.set_threshold(*old)


def _allocate(n: int) -> list:
    return [[] for _ in range(n)]


def test_collects_when_half_the_threshold_is_reached(thresholds, monkeypatch):
    calls = []
    monkeypatch.setattr(gc_at_waits.gc, "collect", lambda g: calls.append(g))
    monkeypatch.setattr(gc_at_waits.gc, "get_count", lambda: (349, 0, 0))
    gc_at_waits.collect_due()
    assert calls == []
    monkeypatch.setattr(gc_at_waits.gc, "get_count", lambda: (350, 8, 0))
    gc_at_waits.collect_due()
    assert calls == [0]
    monkeypatch.setattr(gc_at_waits.gc, "get_count", lambda: (350, 9, 3))
    gc_at_waits.collect_due()
    assert calls == [0, 1]


def test_runs_real_collections_and_resets_the_count(thresholds):
    keep = _allocate(500)
    assert gc.get_count()[0] >= 350
    gc_at_waits.collect_due()
    assert gc.get_count()[0] < 350
    del keep


def test_other_threads_never_collect(thresholds, monkeypatch):
    calls = []
    monkeypatch.setattr(gc_at_waits.gc, "collect", lambda g: calls.append(g))
    monkeypatch.setattr(gc_at_waits.gc, "get_count", lambda: (10_000, 9, 9))
    t = threading.Thread(target=gc_at_waits.collect_due)
    t.start()
    t.join()
    assert calls == []


def test_disabled_gc_threshold_never_collects(monkeypatch):
    calls = []
    monkeypatch.setattr(gc_at_waits.gc, "collect", lambda g: calls.append(g))
    monkeypatch.setattr(gc_at_waits.gc, "get_threshold", lambda: (0, 10, 10))
    monkeypatch.setattr(gc_at_waits.gc, "get_count", lambda: (10_000, 0, 0))
    gc_at_waits.collect_due()
    assert calls == []


def test_spin_condition_wait_collects_first(monkeypatch):
    seen = []
    monkeypatch.setattr(
        SpinCondition,
        "wait",
        lambda self, timeout_ms=None: seen.append(("wait", timeout_ms)),
    )
    monkeypatch.setattr(gc_at_waits, "ENABLED", True)
    monkeypatch.setattr(gc_at_waits, "collect_due", lambda: seen.append("collect"))
    gc_at_waits.patch_gc_at_waits()
    SpinCondition.wait(object(), 5)
    assert seen == ["collect", ("wait", 5)]
    patched = SpinCondition.wait
    gc_at_waits.patch_gc_at_waits()
    assert SpinCondition.wait is patched


def test_flag_off_installs_nothing(monkeypatch):
    before = SpinCondition.wait
    monkeypatch.setattr(gc_at_waits, "ENABLED", False)
    gc_at_waits.patch_gc_at_waits()
    assert SpinCondition.wait is before


class _Event:
    def __init__(self, done=None, raises=False):
        self.done, self.raises = done, raises

    def query(self):
        if self.raises:
            raise RuntimeError("no query")
        return self.done


def test_event_pending():
    assert gc_at_waits.event_pending(_Event(done=False))
    assert not gc_at_waits.event_pending(_Event(done=True))
    assert not gc_at_waits.event_pending(_Event(raises=True))


def test_copy_wait_collects_only_while_pending(monkeypatch):
    from vllm_torchtpu.runner.tpu_runner_async_output import AsyncTPUCopyState

    seen = []

    class _SyncEvent(_Event):
        def synchronize(self):
            seen.append("sync")

    monkeypatch.setattr(gc_at_waits, "collect_due", lambda: seen.append("collect"))
    for enabled, done, expect in (
        (True, False, ["collect", "sync"]),
        (True, True, ["sync"]),
        (False, False, ["sync"]),
    ):
        seen.clear()
        monkeypatch.setattr(gc_at_waits, "ENABLED", enabled)
        state = AsyncTPUCopyState(
            chunks_cpu=[], chunk_real_lens=[], copy_ready_event=_SyncEvent(done=done)
        )
        state.wait()
        assert seen == expect and state.completed
