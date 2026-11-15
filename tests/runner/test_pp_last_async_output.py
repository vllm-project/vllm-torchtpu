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
"""VLLM_TPU_PP_LAST_ASYNC_OUTPUT: the last pipeline stage replies from the
worker's output thread in call order, its per-step hooks run in the same order
as without the flag, and the sampled tokens it did not wait for come back
through the scheduler's new_token_ids with the writes the synchronous path
makes."""

import random
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import AsyncModelRunnerOutput

from vllm_torchtpu.runner import pp_last_async_output


class _ReplyRing:
    def __init__(self):
        self.replies = []
        self.cond = threading.Condition()

    def enqueue(self, item):
        with self.cond:
            self.replies.append(item)
            self.cond.notify_all()

    def wait_for(self, n, timeout=10.0):
        with self.cond:
            assert self.cond.wait_for(lambda: len(self.replies) >= n, timeout)
            return list(self.replies)


class _StepOutput:
    def __init__(self, step, tokens=None):
        self.step, self.tokens = step, tokens

    def __eq__(self, other):
        return (self.step, self.tokens) == (other.step, other.tokens)

    def __repr__(self):
        return f"out({self.step}, {self.tokens})"


class _Unwaited(AsyncModelRunnerOutput):
    """Stands in for AsyncTPUModelRunnerOutput: get_output blocks until the
    step's device work (here a timer) is done."""

    def __init__(self, step, ready_at, log):
        self.step, self.ready_at, self.log = step, ready_at, log

    def get_output(self):
        delay = self.ready_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.log.append(("get_output", self.step, threading.current_thread().name))
        return _StepOutput(self.step, tokens=[1000 + self.step])


class _Worker:
    """A last-stage worker: execute_model runs the per-step hooks in order (ring
    receive post, connector bind and load), sample_tokens polls the connector and
    returns the step's output, unwaited when ``unwaited`` is set."""

    def __init__(self, unwaited, delays):
        self.unwaited, self.delays = unwaited, delays
        self.hooks = []
        self.last_forward_end = time.monotonic()

    def init_rpc(self, i):
        self.hooks.append(("init", i))
        return f"init{i}"

    def execute_model(self, step):
        self.hooks.append(("begin_step", step))
        self.hooks.append(("bind_connector_metadata", step))
        self.hooks.append(("start_load_kv", step))
        if step == 3:
            # A step without tokens: a concrete output (connector reports only).
            return _StepOutput(step)
        return None

    def sample_tokens(self, step):
        self.hooks.append(("get_finished", step))
        if step == 3:
            return None
        if step == 5:
            raise RuntimeError("sampler failed")
        # Forwards run back to back on the device, each `delay` long.
        self.last_forward_end = max(self.last_forward_end, time.monotonic()) + self.delays[step]
        out = _Unwaited(step, self.last_forward_end, self.log)
        return out if self.unwaited else out.get_output()


def _calls(steps):
    calls = [("init_rpc", 0), ("init_rpc", 1)]
    for step in range(steps):
        calls += [("execute_model", step), ("sample_tokens", step)]
    return calls


def _run(unwaited, steps, delays):
    proc = object.__new__(WorkerProc)
    proc.rank = 7
    proc.use_async_scheduling = False
    proc.worker_response_mq = _ReplyRing()
    worker = _Worker(unwaited, delays)
    worker.log = []
    proc.worker = worker
    calls = _calls(steps)
    t0 = time.monotonic()
    for method, arg in calls:
        proc._execute_worker_rpc((method, (arg,), {}, None))
    main_done = time.monotonic() - t0
    replies = proc.worker_response_mq.wait_for(len(calls))
    return proc, worker, replies, main_done


@pytest.fixture
def patched():
    saved = WorkerProc.__dict__.get("handle_output")
    pp_last_async_output.install_worker_output_thread()
    yield
    WorkerProc.handle_output = saved
    WorkerProc._tpu_pp_last_async_output_patch = False


def test_replies_keep_call_order_through_the_output_thread(patched):
    random.seed(0)
    steps = 12
    delays = [random.uniform(0.005, 0.03) for _ in range(steps)]
    proc, worker, replies, main_done = _run(True, steps, delays)
    _, ref_worker, ref_replies, ref_main_done = _run(False, steps, delays)

    # Same replies, same order, as the synchronous worker; failures stay in place.
    assert replies == ref_replies
    ok, fail = WorkerProc.ResponseStatus.SUCCESS, WorkerProc.ResponseStatus.FAILURE
    expected = [(ok, "init0"), (ok, "init1")]
    for step in range(steps):
        expected.append((ok, _StepOutput(3) if step == 3 else None))
        if step == 3:
            expected.append((ok, None))
        elif step == 5:
            expected.append((fail, "sampler failed"))
        else:
            expected.append((ok, _StepOutput(step, [1000 + step])))
    assert replies == expected

    # The hooks (ring receive post, connector bind / load / poll) run on the main
    # thread in call order, the same sequence as without the thread.
    assert worker.hooks == ref_worker.hooks

    # The output thread started at the first unwaited output and waited on every
    # copy; replies before it were sent synchronously.
    assert proc.use_async_scheduling and proc.async_output_copy_thread.name == "WorkerAsyncOutputCopy"
    assert {name for kind, _, name in worker.log} == {"WorkerAsyncOutputCopy"}
    assert {name for kind, _, name in ref_worker.log} == {threading.current_thread().name}

    # The main thread did not wait on the copies: it dispatched every call well
    # before the last forward ended, while the synchronous worker waited each one.
    assert main_done < 0.5 * sum(delays) < ref_main_done


def test_next_steps_hooks_run_before_the_previous_reply(patched):
    """The point of the flag: step N+1's hooks run while step N's copy is pending."""
    delays = [0.05] * 4
    proc = object.__new__(WorkerProc)
    proc.rank = 7
    proc.use_async_scheduling = False
    ring = proc.worker_response_mq = _ReplyRing()
    worker = proc.worker = _Worker(True, delays)
    worker.log = []
    events = []
    orig_enqueue = ring.enqueue

    def enqueue(item):
        events.append(("reply", len(ring.replies)))
        orig_enqueue(item)

    ring.enqueue = enqueue
    for method, arg in [("execute_model", 0), ("sample_tokens", 0), ("execute_model", 1)]:
        proc._execute_worker_rpc((method, (arg,), {}, None))
        events.append(("returned", method, arg))
    ring.wait_for(3)
    # execute_model(1) returned (its hooks ran) before sample_tokens(0)'s reply left.
    assert events.index(("returned", "execute_model", 1)) < events.index(("reply", 1))


def test_inert_without_an_unwaited_output(patched):
    proc = object.__new__(WorkerProc)
    proc.rank = 0
    proc.use_async_scheduling = False
    proc.worker_response_mq = _ReplyRing()
    proc.handle_output(None)
    proc.handle_output(_StepOutput(0))
    assert not proc.use_async_scheduling
    assert not hasattr(proc, "async_output_copy_thread")
    assert proc.worker_response_mq.replies == [
        (WorkerProc.ResponseStatus.SUCCESS, None),
        (WorkerProc.ResponseStatus.SUCCESS, _StepOutput(0)),
    ]


# -- which workers turn it on ------------------------------------------------------------------------------


def _config(pp=8, tp=1, pcp=1, async_scheduling=False, spec=None, kv=object()):
    return NS(
        parallel_config=NS(
            pipeline_parallel_size=pp, tensor_parallel_size=tp, prefill_context_parallel_size=pcp
        ),
        scheduler_config=NS(async_scheduling=async_scheduling),
        speculative_config=spec,
        kv_transfer_config=kv,
    )


def test_decide(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "VLLM_TPU_PP_LAST_ASYNC_OUTPUT", False, raising=False)
    assert pp_last_async_output.decide(_config(), True, False) == (False, "flag off")
    monkeypatch.setattr(envs, "VLLM_TPU_PP_LAST_ASYNC_OUTPUT", True, raising=False)
    assert pp_last_async_output.decide(_config(), True, False) == (True, "on")
    assert pp_last_async_output.decide(_config(kv=None), True, False) == (True, "on")
    assert not pp_last_async_output.decide(_config(), False, False)[0]
    assert not pp_last_async_output.decide(_config(pp=1), True, False)[0]
    assert not pp_last_async_output.decide(_config(async_scheduling=True), True, False)[0]
    assert not pp_last_async_output.decide(_config(spec=object()), True, False)[0]
    assert not pp_last_async_output.decide(_config(), True, True)[0]
    # A last stage of several ranks needs every rank's reply to reach the thread.
    assert not pp_last_async_output.decide(_config(tp=2, kv=None), True, False)[0]
    assert pp_last_async_output.decide(_config(tp=2), True, False)[0]


# -- the tokens it did not wait for ------------------------------------------------------------------------


class _Req:
    def __init__(self, prompt, outputs=()):
        self.prompt_token_ids = list(prompt)
        self.num_prompt_tokens = len(prompt)
        self.output_token_ids = list(outputs)

    @property
    def num_tokens(self):
        return self.num_prompt_tokens + len(self.output_token_ids)


def _batch(requests, order):
    width = 64
    batch = NS(
        req_id_to_index={},
        token_ids_cpu=np.zeros((len(order), width), dtype=np.int32),
        num_tokens_no_spec=np.zeros(len(order), dtype=np.int32),
    )
    for idx, req_id in enumerate(order):
        req = requests[req_id]
        batch.req_id_to_index[req_id] = idx
        toks = req.prompt_token_ids + req.output_token_ids
        batch.token_ids_cpu[idx, : len(toks)] = toks
        batch.num_tokens_no_spec[idx] = len(toks)
    return batch


def _sync_update(requests, batch, sampled):
    """What the synchronous sample_tokens writes once the tokens reach the host."""
    for req_id, tokens in sampled.items():
        idx = batch.req_id_to_index.get(req_id)
        if idx is not None:
            start = batch.num_tokens_no_spec[idx]
            batch.token_ids_cpu[idx, start : start + len(tokens)] = tokens
            batch.num_tokens_no_spec[idx] = start + len(tokens)
        requests[req_id].output_token_ids.extend(tokens)


def _state(requests, batch):
    return (
        {r: list(q.output_token_ids) for r, q in requests.items()},
        batch.token_ids_cpu.copy().tolist(),
        batch.num_tokens_no_spec.copy().tolist(),
    )


def test_scheduled_token_ids_match_the_synchronous_update():
    def setup():
        reqs = {
            "done": _Req(range(10, 20)),        # prompt finished in step N: sampled 777
            "chunk": _Req(range(30, 60)),       # mid-prompt chunk: nothing sampled
            "decode": _Req(range(5), [41]),     # already decoding: sampled 88
            "resumed": _Req(range(70, 74)),     # outside the batch, sampled 99 before preemption
        }
        return reqs, _batch(reqs, ["done", "chunk", "decode"])

    sync_reqs, sync_batch = setup()
    _sync_update(sync_reqs, sync_batch, {"done": [777], "decode": [88], "resumed": [99]})

    reqs, batch = setup()
    # Step N+1's cached-request data as vLLM's scheduler builds it under PP without async
    # scheduling: each request's scheduled tokens from num_computed_tokens on.
    req_data = NS(
        req_ids=["done", "chunk", "decode", "resumed"],
        new_token_ids=[[777], list(range(40, 50)), [88], list(range(70, 74)) + [99]],
        num_computed_tokens=[10, 10, 6, 0],
    )
    applied = pp_last_async_output.apply_scheduled_token_ids(reqs, batch, req_data)
    assert applied == 3
    assert _state(reqs, batch) == _state(sync_reqs, sync_batch)

    # A resumed request recomputing only part of its prompt gets its token later.
    reqs, batch = setup()
    req_data = NS(req_ids=["resumed"], new_token_ids=[[70, 71]], num_computed_tokens=[0])
    assert pp_last_async_output.apply_scheduled_token_ids(reqs, batch, req_data) == 0
    assert reqs["resumed"].output_token_ids == []

    # Async scheduling sends no tokens: nothing to apply.
    assert pp_last_async_output.apply_scheduled_token_ids(
        reqs, batch, NS(req_ids=["done"], new_token_ids=[], num_computed_tokens=[10])
    ) == 0
