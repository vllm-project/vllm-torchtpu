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
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from vllm_torchtpu.distributed import pp_engine


@pytest.fixture(autouse=True)
def fresh_ring_minimum(monkeypatch):
    monkeypatch.setattr(pp_engine, "_ring_chunks_needed", 0)


@pytest.fixture(autouse=True)
def fresh_steps_in_flight(monkeypatch):
    from vllm.config import VllmConfig

    # The engine side raises vLLM's steps in flight once per process: each
    # test starts from vLLM's property, unmarked, and both come back
    # afterwards.
    monkeypatch.setattr(
        VllmConfig, "max_concurrent_batches", VllmConfig.max_concurrent_batches
    )
    monkeypatch.setattr(VllmConfig, "_tpu_pp_steps_patch", False, raising=False)
    monkeypatch.delattr(VllmConfig, "_tpu_pp_steps_patch")


@pytest.fixture
def queue(monkeypatch):
    """A stand-in for vLLM's MessageQueue, unpatched for each test."""
    from vllm.distributed.device_communicators import shm_broadcast

    class _Queue:
        def __init__(self, n_reader, n_local_reader, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(shm_broadcast, "MessageQueue", _Queue)
    return _Queue


def _done(value):
    f = Future()
    f.set_result(value)
    return f


def test_rpc_ring_holds_the_pipeline_lookahead():
    assert pp_engine.rpc_ring_chunks(8) == 40
    assert pp_engine.rpc_ring_chunks(2) == 16


def test_widening_applies_to_every_ring(queue):
    pp_engine._widen_rpc_ring(8)
    assert queue(8, 8, max_chunk_bytes=1).kwargs["max_chunks"] == 40
    assert queue(8, 8, max_chunks=64).kwargs["max_chunks"] == 64
    assert queue(1, 1).kwargs["max_chunks"] == 40


def test_a_deeper_pipeline_widens_the_rings_a_shallower_one_keeps_them(queue):
    pp_engine._widen_rpc_ring(2)
    assert queue(1, 1).kwargs["max_chunks"] == 16
    pp_engine._widen_rpc_ring(4)
    assert queue(1, 1).kwargs["max_chunks"] == 24
    pp_engine._widen_rpc_ring(2)
    assert queue(1, 1).kwargs["max_chunks"] == 24


def _config(pp):
    return SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=pp))


def test_worker_side_widening_follows_the_pipeline_size(queue):
    pp_engine.widen_message_rings(_config(1))
    assert "max_chunks" not in queue(1, 1).kwargs
    pp_engine.widen_message_rings(_config(4))
    assert queue(1, 1).kwargs["max_chunks"] == pp_engine.rpc_ring_chunks(4, 2)


def test_patch_is_skipped_without_a_pipeline():
    pp_engine.patch_executor_for_pp_buffered(None)
    pp_engine.patch_executor_for_pp_buffered(_config(1))


def _step(tokens, step):
    return SimpleNamespace(total_num_scheduled_tokens=tokens, step=step)


def test_the_engine_keeps_half_the_stages_again_in_flight():
    extra = [pp_engine.buffered_extra_steps(stages) for stages in (2, 4, 8)]
    assert extra == [1, 2, 4]


def test_the_rings_hold_the_extra_steps(monkeypatch, queue):
    assert pp_engine.rpc_ring_chunks(8, 4) == 4 * 12 + 8
    _bare_executor_cls(monkeypatch)
    pp_engine.patch_executor_for_pp_buffered(_config(8))
    assert queue(1, 1).kwargs["max_chunks"] == pp_engine.rpc_ring_chunks(8, 4)


def test_the_worker_rings_hold_the_extra_steps(monkeypatch, queue):
    from vllm.config import VllmConfig

    pp_engine.widen_message_rings(_config(4))
    assert queue(1, 1).kwargs["max_chunks"] == pp_engine.rpc_ring_chunks(
        4, pp_engine.buffered_extra_steps(4)
    )
    # the worker's config counts the extra steps as the engine's does
    steps = VllmConfig.max_concurrent_batches.fget
    assert steps(_engine_config(4, async_scheduling=False)) == 4 + 2
    # the depth caps a stage's send queue, not the engine's steps in flight
    monkeypatch.setenv("TPU_PP_HANDOFF_DEPTH", "4")
    pp_engine.widen_message_rings(_config(8))
    assert queue(1, 1).kwargs["max_chunks"] == pp_engine.rpc_ring_chunks(
        8, pp_engine.buffered_extra_steps(8)
    )


def _engine_config(pp, async_scheduling):
    """A VllmConfig stand-in carrying what vLLM's max_concurrent_batches
    reads."""
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        use_v2_model_runner=False,
    )


def test_a_pipeline_engine_keeps_extra_steps_in_flight(monkeypatch, queue):
    from vllm.config import VllmConfig

    vllm_steps = VllmConfig.max_concurrent_batches.fget
    _bare_executor_cls(monkeypatch)
    pp_engine.patch_executor_for_pp_buffered(_config(8))
    steps = VllmConfig.max_concurrent_batches.fget
    assert steps(_engine_config(8, async_scheduling=False)) == 8 + 4
    # async scheduling sizes the queue for its own overlap
    overlapped = _engine_config(8, async_scheduling=True)
    assert steps(overlapped) == vllm_steps(overlapped) == 8
    # an engine of one stage keeps vLLM's count
    single = _engine_config(1, async_scheduling=False)
    assert steps(single) == vllm_steps(single) == 1


def test_the_steps_in_flight_are_raised_once(monkeypatch, queue):
    from vllm.config import VllmConfig

    _bare_executor_cls(monkeypatch)
    pp_engine.patch_executor_for_pp_buffered(_config(8))
    raised = VllmConfig.max_concurrent_batches
    # a later call keeps the one patch; each config counts its own stages
    pp_engine.patch_executor_for_pp_buffered(_config(8))
    pp_engine.patch_executor_for_pp_buffered(_config(4))
    assert VllmConfig.max_concurrent_batches is raised
    assert raised.fget(_engine_config(8, async_scheduling=False)) == 8 + 4
    assert raised.fget(_engine_config(4, async_scheduling=False)) == 4 + 2
    assert raised.fget(_engine_config(2, async_scheduling=False)) == 2 + 1


def _bare_executor_cls(monkeypatch):
    """A MultiprocExecutor stand-in that the patch has not touched yet."""
    from vllm.v1.executor import multiproc_executor

    class _FakeExecutor:
        def __init__(self):
            self.vllm_config = _config(4)
            self.output_rank = 3
            self.calls = []

        def execute_model(self, scheduler_output, non_block=False):
            future = _done(("forward", scheduler_output.step))
            return future if non_block else future.result()

        def sample_tokens(self, grammar_output, non_block=False):
            future = _done(("sample", grammar_output))
            return future if non_block else future.result()

        def take_draft_token_ids(self):
            raise AssertionError("asked the workers for draft tokens")

        def collective_rpc(self, method, **kwargs):
            self.calls.append((method, kwargs))

    monkeypatch.setattr(multiproc_executor, "MultiprocExecutor", _FakeExecutor)
    return _FakeExecutor


def test_the_executor_keeps_vllms_forward_and_sampling_calls(monkeypatch, queue):
    from vllm.config import VllmConfig

    dispatch_depth = VllmConfig.max_concurrent_batches
    executor_cls = _bare_executor_cls(monkeypatch)
    execute_model = executor_cls.execute_model
    sample_tokens = executor_cls.sample_tokens
    pp_engine.patch_executor_for_pp_buffered(_config(4))
    assert executor_cls.execute_model is execute_model
    assert executor_cls.sample_tokens is sample_tokens
    executor = executor_cls()
    future = executor.execute_model(_step(16, 1), non_block=True)
    assert future.result() == ("forward", 1)
    assert executor.take_draft_token_ids() is None
    assert executor.calls == []
    assert queue(1, 1).kwargs["max_chunks"] == pp_engine.rpc_ring_chunks(4, 2)
    # the engine keeps more steps in flight than vLLM's one per stage
    assert VllmConfig.max_concurrent_batches is not dispatch_depth


def test_a_single_stage_keeps_its_draft_tokens(monkeypatch, queue):
    executor_cls = _bare_executor_cls(monkeypatch)
    executor_cls.take_draft_token_ids = lambda self: "drafts"
    pp_engine.patch_executor_for_pp_buffered(_config(4))
    single = executor_cls()
    single.vllm_config = _config(1)
    assert single.take_draft_token_ids() == "drafts"
    assert executor_cls().take_draft_token_ids() is None
