# SPDX-License-Identifier: Apache-2.0
"""Debug modes for TPU tensors freed on torch_tpu compile threads."""

import gc
import threading

import pytest
import torch

from vllm_torchtpu.compilation import compile_thread_gc_diag as diag
from vllm_torchtpu.compilation import moe_schedule


@pytest.fixture
def gc_enabled():
    was_enabled = gc.isenabled()
    gc.enable()
    yield
    if was_enabled:
        gc.enable()
    else:
        gc.disable()


@pytest.fixture
def records(monkeypatch):
    seen = []

    class Sink:
        def info(self, msg, *args):
            seen.append(msg % args)

    monkeypatch.setattr(diag, "logger", Sink())
    monkeypatch.setattr(diag, "_pass_threads", set())
    # The GC watch is process-wide; remove it after each test.
    callbacks = list(gc.callbacks)
    monkeypatch.setattr(diag, "_watch_installed", False)
    yield seen
    gc.callbacks[:] = callbacks


def _rewrite_records_gc(monkeypatch):
    states = []

    def rewrite(blob):
        states.append(gc.isenabled())

    monkeypatch.setattr(moe_schedule, "rewrite_moe_schedule", rewrite)
    return states


@pytest.mark.cpu_test
def test_off_registers_the_guarded_pass(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_DEBUG_COMPILE_THREAD_GC", "off")
    assert moe_schedule._pass_callback() is moe_schedule.rewrite_moe_schedule_without_gc


@pytest.mark.cpu_test
def test_invalid_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_TPU_DEBUG_COMPILE_THREAD_GC", "on")
    with pytest.raises(ValueError):
        moe_schedule._pass_callback()


@pytest.mark.cpu_test
def test_observe_keeps_guard_and_names_pass_thread(gc_enabled, monkeypatch, records):
    monkeypatch.setenv("VLLM_TPU_DEBUG_COMPILE_THREAD_GC", "observe")
    states = _rewrite_records_gc(monkeypatch)
    callback = moe_schedule._pass_callback()

    thread = threading.Thread(target=callback, args=(b"",))
    thread.start()
    thread.join(timeout=10)
    callback(b"")

    assert states == [False, False]
    assert gc.isenabled()
    passes = [r for r in records if "HLO pass on thread" in r]
    assert len(passes) == 2
    assert any("main=False" in r for r in passes)
    assert any("main=True" in r for r in passes)


@pytest.mark.cpu_test
def test_collect_in_pass_drops_guard_and_collects(gc_enabled, monkeypatch, records):
    monkeypatch.setenv("VLLM_TPU_DEBUG_COMPILE_THREAD_GC", "collect_in_pass")
    states = _rewrite_records_gc(monkeypatch)
    callback = moe_schedule._pass_callback()

    thread = threading.Thread(target=callback, args=(b"",))
    thread.start()
    thread.join(timeout=10)

    assert states == [True]
    assert any("forced collection freed" in r for r in records)
    # The watch logs the forced gen2 collection on the pass thread.
    assert any("gen2 collection" in r and "pass_thread=True" in r for r in records)


@pytest.mark.cpu_test
def test_dump_names_tensor_held_by_cycle(gc_enabled, records):
    class Holder:
        pass

    holder = Holder()
    holder.self_ref = holder
    holder.tensor = torch.zeros(3, 5)
    del holder

    assert diag.dump_tpu_garbage("unit", device_type="cpu") >= 1
    summary = next(r for r in records if "phase=unit" in r)
    assert "cpu_tensors=" in summary
    assert any("Tensor(3, 5) torch.float32 cpu" in r for r in records)
    assert any("<- " in r for r in records)
    # The saved cycles were released by the dump, not left for another thread.
    assert not gc.garbage
    assert not gc.get_debug() & gc.DEBUG_SAVEALL


@pytest.mark.cpu_test
def test_dump_survives_dead_weakref_proxies(gc_enabled, records):
    import weakref

    class Target:
        pass

    class Holder:
        pass

    target = Target()
    holder = Holder()
    holder.self_ref = holder
    holder.proxy = weakref.proxy(target)
    holder.tensor = torch.zeros(2)
    del holder, target

    # The proxy is dead inside the saved garbage; inspecting it must not raise.
    assert diag.dump_tpu_garbage("proxy", device_type="cpu") >= 1
    assert any("phase=proxy" in r for r in records)
    assert not gc.garbage


@pytest.mark.cpu_test
def test_dump_is_a_no_op_off_the_main_thread(records):
    result = []
    thread = threading.Thread(
        target=lambda: result.append(diag.dump_tpu_garbage("bg", device_type="cpu"))
    )
    thread.start()
    thread.join(timeout=10)
    assert result == [0]
    assert not records
