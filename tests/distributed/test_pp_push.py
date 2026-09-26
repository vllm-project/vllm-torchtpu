# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed import pp_push


@pytest.fixture(autouse=True)
def fresh_ring_minimum(monkeypatch):
    monkeypatch.setattr(pp_push, "_ring_chunks_needed", 0)


@pytest.fixture
def queue(monkeypatch):
    """A stand-in for vLLM's MessageQueue, unpatched for each test."""
    from vllm.distributed.device_communicators import shm_broadcast

    class _Queue:
        def __init__(self, n_reader, n_local_reader, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(shm_broadcast, "MessageQueue", _Queue)
    return _Queue


class _Executor:
    def __init__(self):
        self.output_rank = 7
        self.calls = []

    def collective_rpc(self, method, **kwargs):
        self.calls.append((method, kwargs))


def _done(value):
    f = Future()
    f.set_result(value)
    return f


def _futures(executor, wave, count):
    """Steps 1..count dispatched back to back."""
    futures = []
    for _ in range(count):
        step = wave.dispatched()
        futures.append(
            pp_push._PushFuture(_done(step), executor, wave, step, wave.real_steps)
        )
    return futures


def test_pushes_exactly_the_missing_later_steps():
    executor = _Executor()
    wave = pp_push._WaveState(stages=8)
    futures = _futures(executor, wave, 9)
    # step 1 with 8 later in flight: nothing to do
    assert futures[0].result() == 1
    assert executor.calls == []
    # step 5 with 4 later in flight needs 6: two pushes, replies not awaited
    assert futures[4].result() == 5
    assert (
        executor.calls == [("pp_push", {"non_block": True, "unique_reply_rank": 7})] * 2
    )
    assert wave.steps == 11 and wave.pushes == 2
    # the pushes count as later steps: step 6 now has 5 later, one push
    assert futures[5].result() == 6
    assert len(executor.calls) == 3
    assert wave.steps == 12
    # step 9 is the newest step the scheduler produced: nothing else is in
    # flight, so the wave settles instead of pushing
    assert futures[8].result() == 9
    assert [c[0] for c in executor.calls] == ["pp_push"] * 3 + ["pp_settle"]
    assert wave.open is False and wave.settles == 1
    # waiting again on an older step sends nothing more
    assert futures[7].result() == 8
    assert len(executor.calls) == 4


def test_settles_when_the_awaited_step_is_the_newest():
    executor = _Executor()
    wave = pp_push._WaveState(stages=4)
    first, second = _futures(executor, wave, 2)
    # step 1 with step 2 behind it: one push
    assert first.result() == 1
    assert [c[0] for c in executor.calls] == ["pp_push"]
    assert wave.open
    # step 2 is the newest: the settle carries it, no push
    assert second.result() == 2
    assert [c[0] for c in executor.calls] == ["pp_push", "pp_settle"]
    assert not wave.open
    # the next step opens a new burst; alone, it settles too
    (third,) = _futures(executor, wave, 1)
    assert wave.open
    assert third.result() == 4
    assert [c[0] for c in executor.calls] == ["pp_push", "pp_settle", "pp_settle"]


def test_the_future_forwards_to_the_executors_future():
    executor = _Executor()
    wave = pp_push._WaveState(stages=2)
    inner = Future()
    future = pp_push._PushFuture(inner, executor, wave, 1, 1)
    seen = []
    future.add_done_callback(seen.append)
    assert not future.done() and not future.running()
    inner.set_result("out")
    assert future.done() and seen == [future]
    assert future.result() == "out"
    assert future.exception(timeout=0) is None

    failed = Future()
    future = pp_push._PushFuture(failed, executor, wave, 1, 1)
    failed.set_exception(RuntimeError("boom"))
    assert isinstance(future.exception(), RuntimeError)
    with pytest.raises(RuntimeError, match="boom"):
        future.result()

    pending = Future()
    future = pp_push._PushFuture(pending, executor, wave, 1, 1)
    assert future.cancel() and future.cancelled() and pending.cancelled()


def test_waiting_for_the_exception_settles_too():
    executor = _Executor()
    wave = pp_push._WaveState(stages=8)
    step = wave.dispatched()
    assert (
        pp_push._PushFuture(_done(1), executor, wave, step, wave.real_steps).exception()
        is None
    )
    assert [c[0] for c in executor.calls] == ["pp_settle"]


def test_rpc_ring_holds_the_pipeline_lookahead():
    assert pp_push.rpc_ring_chunks(8) == 40
    assert pp_push.rpc_ring_chunks(2) == 16


def test_widening_applies_to_every_ring(queue):
    pp_push._widen_rpc_ring(8)
    assert queue(8, 8, max_chunk_bytes=1).kwargs["max_chunks"] == 40
    assert queue(8, 8, max_chunks=64).kwargs["max_chunks"] == 64
    assert queue(1, 1).kwargs["max_chunks"] == 40


def test_a_deeper_pipeline_widens_the_rings_a_shallower_one_keeps_them(queue):
    pp_push._widen_rpc_ring(2)
    assert queue(1, 1).kwargs["max_chunks"] == 16
    pp_push._widen_rpc_ring(4)
    assert queue(1, 1).kwargs["max_chunks"] == 24
    pp_push._widen_rpc_ring(2)
    assert queue(1, 1).kwargs["max_chunks"] == 24


def _config(pp):
    return SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=pp))


def test_worker_side_widening_follows_the_pipeline_size(queue):
    pp_push.widen_message_rings(_config(1))
    assert "max_chunks" not in queue(1, 1).kwargs
    pp_push.widen_message_rings(_config(4))
    assert queue(1, 1).kwargs["max_chunks"] == 24


def test_patch_is_skipped_without_a_pipeline():
    pp_push.patch_executor_for_pp_wave(None)
    pp_push.patch_executor_for_pp_wave(_config(1))


def _install_on_fake_executor(monkeypatch, stages):
    from vllm.v1.executor import multiproc_executor

    class _FakeExecutor:
        def __init__(self, stages=stages):
            self.vllm_config = _config(stages)
            self.output_rank = stages - 1
            self.calls = []

        def execute_model(self, scheduler_output, non_block=False):
            future = _done(("forward", scheduler_output.step))
            return future if non_block else future.result()

        def sample_tokens(self, grammar_output, non_block=False):
            future = _done(("sample", grammar_output))
            return future if non_block else future.result()

        def collective_rpc(self, method, **kwargs):
            self.calls.append((method, kwargs))

    monkeypatch.setattr(multiproc_executor, "MultiprocExecutor", _FakeExecutor)
    pp_push.patch_executor_for_pp_wave(_config(stages))
    return _FakeExecutor()


def _step(tokens, step):
    return SimpleNamespace(total_num_scheduled_tokens=tokens, step=step)


def test_forward_futures_push_like_sampling_futures(monkeypatch, queue):
    # A pooling model waits on execute_model's future, never on sampling.
    executor = _install_on_fake_executor(monkeypatch, stages=4)
    futures = [
        executor.execute_model(_step(16, i), non_block=True) for i in range(1, 4)
    ]
    assert all(isinstance(f, pp_push._PushFuture) for f in futures)
    # step 1 with 2 later in flight: the wave carries it, no push
    assert futures[0].result() == ("forward", 1)
    assert executor.calls == []
    # step 2 with 1 later in flight: one push
    assert futures[1].result() == ("forward", 2)
    assert [c[0] for c in executor.calls] == ["pp_push"]
    # step 3 is the newest: the settle carries it
    assert futures[2].result() == ("forward", 3)
    assert [c[0] for c in executor.calls] == ["pp_push", "pp_settle"]


def test_sampling_waits_for_its_own_step(monkeypatch, queue):
    executor = _install_on_fake_executor(monkeypatch, stages=4)
    executor.execute_model(_step(16, 1), non_block=True)
    sample = executor.sample_tokens("grammar", non_block=True)
    assert sample.result() == ("sample", "grammar")
    # the sampling belongs to the lone step, which settles
    assert [c[0] for c in executor.calls] == ["pp_settle"]


def test_blocking_forward_call_returns_the_output(monkeypatch, queue):
    executor = _install_on_fake_executor(monkeypatch, stages=4)
    assert executor.execute_model(_step(16, 1)) == ("forward", 1)
    assert [c[0] for c in executor.calls] == ["pp_settle"]


def test_empty_steps_do_not_join_the_wave(monkeypatch, queue):
    executor = _install_on_fake_executor(monkeypatch, stages=4)
    future = executor.execute_model(_step(0, 1), non_block=True)
    assert not isinstance(future, pp_push._PushFuture)
    assert future.result() == ("forward", 1)
    assert executor.calls == []
    assert executor._tpu_wave_state.steps == 0


def test_each_executor_pushes_for_its_own_pipeline_size(monkeypatch, queue):
    two = _install_on_fake_executor(monkeypatch, stages=2)
    four = type(two)(stages=4)
    pp_push.patch_executor_for_pp_wave(four.vllm_config)
    # two stages: every launch pairs on its own, so neither push nor settle
    assert two.execute_model(_step(16, 1)) == ("forward", 1)
    assert two.calls == []
    # four stages: the same lone step settles as well
    assert four.execute_model(_step(16, 1)) == ("forward", 1)
    assert [c[0] for c in four.calls] == ["pp_settle"]
    assert queue(1, 1).kwargs["max_chunks"] == 24
