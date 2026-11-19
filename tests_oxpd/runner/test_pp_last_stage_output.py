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
"""The last pipeline stage under synchronous scheduling returns its step
without waiting for the device (TPU_PP_LAST_STAGE_ASYNC_OUTPUT)."""

import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner import pp_last_stage_output as plo
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.runner.tpu_runner_async_output import (
    AsyncTPUCopyState,
    AsyncTPUModelRunnerOutput,
)


class _Event:
    """A device event that has not completed until released."""

    def __init__(self, done: bool = False):
        self.done = done
        self.waits = 0

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.waits += 1
        self.done = True


def _copy(tokens: list[int], done: bool = False) -> AsyncTPUCopyState:
    return AsyncTPUCopyState(
        chunks_cpu=[torch.tensor(tokens, dtype=torch.int32)],
        chunk_real_lens=[len(tokens)],
        copy_ready_event=_Event(done),
    )


def _runner(*, pp: int = 8, last: bool = True, spec=None, rows=None):
    """The runner state sample_tokens reads on the last of ``pp`` stages of a
    producer: requests ``rows`` = [(req_id, num_computed, num_scheduled,
    num_tokens)], greedy, no drafter. A row whose chunk ends before its
    prompt does (a partial prefill) has its sample thrown away."""
    rows = rows or [("r0", 0, 1, 1)]
    n = len(rows)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={r[0]: r[2] for r in rows},
        finished_req_ids=set(),
        has_structured_output_requests=False,
        total_num_scheduled_tokens=sum(r[2] for r in rows),
    )
    runner = SimpleNamespace(
        _pp_wave=None,
        _pp_is_last=last,
        parallel_config=SimpleNamespace(pipeline_parallel_size=pp),
        execute_model_state=SimpleNamespace(
            scheduler_output=scheduler_output,
            draft_chunks=[],
            logits_list=[torch.zeros(n, 16)],
            num_reqs_list=[n],
            spec_decode_metadata_list=[None],
            mamba_state_indices_list=None,
            pooler_output_list=[],
            top_tokens=False,
        ),
        input_batch=SimpleNamespace(
            max_num_logprobs=None,
            num_reqs=n,
            req_ids=[r[0] for r in rows],
            req_id_to_index={r[0]: i for i, r in enumerate(rows)},
            all_greedy=True,
            generators={},
            num_tokens_no_spec=[r[3] for r in rows],
            token_ids_cpu=np.zeros((n, 16), dtype=np.int32),
        ),
        requests={
            r[0]: SimpleNamespace(
                num_computed_tokens=r[1], num_tokens=r[3], output_token_ids=[]
            )
            for r in rows
        },
        speculative_config=spec,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _is_async_drafter=False,
        _mtp_prefill_only=False,
        _fence_spec_kv_under_token_copy=False,
        _is_dspark_pd_producer=False,
        is_pooling_model=False,
        _get_sampling_generator=lambda: None,
        # Row i samples token 7 + i.
        _greedy_sample=lambda logits: torch.arange(7, 7 + logits.shape[0]),
        _update_mamba_slot_read_offsets=lambda *_args: None,
        _update_num_xla_graphs=lambda *_args: None,
        _modify_prev_results=lambda: None,
        _stage_timing=MagicMock(),
    )
    for name in ("maybe_wait_for_kv_save", "get_finished_kv_transfers"):
        setattr(runner, name, getattr(TPUModelRunner, name).__get__(runner))
    return runner


def _sample(runner):
    copies = []

    def host_copy(chunks, lengths, spec_logprobs=None):
        state = AsyncTPUCopyState(
            chunks_cpu=[c.to(torch.int32) for c in chunks],
            chunk_real_lens=lengths,
            copy_ready_event=_Event(),
        )
        copies.append(state)
        return state

    with (
        patch("vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=False),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)
    return output, copies[0]


def _flush_all(runner, scheduled):
    return plo.flush_pending(
        runner.__dict__["_pp_last_out_pending"],
        scheduled,
        lambda p, toks: plo.commit_sampled_tokens(
            runner.input_batch, runner.requests, p, toks
        ),
    )


def test_last_stage_returns_its_step_without_waiting(monkeypatch):
    monkeypatch.setenv(plo.ENV, "1")
    runner = _runner()

    output, copy = _sample(runner)

    # Nothing waited for the device: the step's output goes back unresolved
    # and the token is not committed yet.
    assert isinstance(output, AsyncTPUModelRunnerOutput)
    assert copy.copy_ready_event.waits == 0
    assert runner.input_batch.num_tokens_no_spec == [1]
    assert runner.requests["r0"].output_token_ids == []
    pending = runner.__dict__["_pp_last_out_pending"]
    assert len(pending) == 1

    # A later step that schedules r0 commits the token first, waiting for it.
    assert _flush_all(runner, {"r0": 1}) == 1 and not pending
    assert copy.copy_ready_event.waits >= 1
    assert runner.input_batch.num_tokens_no_spec == [2]
    assert runner.input_batch.token_ids_cpu[0, 1] == 7
    assert runner.requests["r0"].output_token_ids == [7]
    # The worker's output thread reports the same token to the engine.
    assert output.get_output().sampled_token_ids == [[7]]


@pytest.mark.parametrize("value", ["0", None])
def test_off_keeps_the_synchronous_path(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(plo.ENV, raising=False)
    else:
        monkeypatch.setenv(plo.ENV, value)
    runner = _runner()

    output, copy = _sample(runner)

    assert not isinstance(output, AsyncTPUModelRunnerOutput)
    assert output.sampled_token_ids == [[7]]
    assert copy.copy_ready_event.waits == 1
    assert runner.input_batch.num_tokens_no_spec == [2]
    assert runner.requests["r0"].output_token_ids == [7]
    assert "_pp_last_out_pending" not in runner.__dict__


@pytest.mark.parametrize(
    "change",
    [
        {"pp": 1},
        {"last": False},
        {"spec": SimpleNamespace(method="mtp", num_speculative_tokens=1)},
    ],
)
def test_other_runners_keep_the_synchronous_path(monkeypatch, change):
    """Only the last stage of a pipeline without a drafter defers."""
    monkeypatch.setenv(plo.ENV, "1")
    runner = _runner(**change)
    if change.get("spec") is not None:
        # sample_tokens's drafter call is not part of this double; the mode
        # decision alone is checked.
        assert plo.mode_for_runner(runner) == plo.OFF
        return
    output, copy = _sample(runner)
    assert not isinstance(output, AsyncTPUModelRunnerOutput)
    assert copy.copy_ready_event.waits == 1
    assert "_pp_last_out_pending" not in runner.__dict__


def test_mode_applies_to_the_last_stage_of_a_synchronous_pipeline(monkeypatch):
    base = dict(
        pipeline_parallel_size=8,
        is_last_stage=True,
        async_scheduling=False,
        is_pooling_model=False,
        has_drafter=False,
    )
    monkeypatch.setenv(plo.ENV, "1")
    assert plo.runner_mode(**base) == plo.ON
    for change in (
        {"pipeline_parallel_size": 1},
        {"is_last_stage": False},
        {"async_scheduling": True},
        {"is_pooling_model": True},
        {"has_drafter": True},
    ):
        assert plo.runner_mode(**{**base, **change}) == plo.OFF, change
    monkeypatch.setenv(plo.ENV, "0")
    assert plo.runner_mode(**base) == plo.OFF
    monkeypatch.setenv(plo.ENV, "bogus")
    with pytest.raises(ValueError):
        plo.runner_mode(**base)


def test_deferred_commit_is_identical_to_the_synchronous_path(monkeypatch):
    """After the commit, the runner state and the engine-visible output equal
    the synchronous path's, for a step with a finished prefill (kept sample)
    and a partial prefill (thrown-away sample)."""
    rows = [("done", 3, 2, 5), ("partial", 0, 4, 9)]
    monkeypatch.setenv(plo.ENV, "0")
    sync_runner = _runner(rows=rows)
    sync_out, _ = _sample(sync_runner)
    monkeypatch.setenv(plo.ENV, "1")
    async_runner = _runner(rows=rows)
    async_out, copy = _sample(async_runner)
    assert isinstance(async_out, AsyncTPUModelRunnerOutput)
    # Only the partial prefill is scheduled next: its sample was thrown away,
    # so nothing has to wait; the landed copy is committed anyway.
    assert _flush_all(async_runner, {"partial": 4}) == 0
    assert copy.copy_ready_event.waits == 0
    copy.copy_ready_event.done = True
    assert _flush_all(async_runner, {"partial": 4}) == 1

    a, b = sync_runner, async_runner
    assert a.input_batch.num_tokens_no_spec == b.input_batch.num_tokens_no_spec == [6, 9]
    assert np.array_equal(a.input_batch.token_ids_cpu, b.input_batch.token_ids_cpu)
    assert a.input_batch.token_ids_cpu[0, 5] == 7
    for req in ("done", "partial"):
        assert a.requests[req].output_token_ids == b.requests[req].output_token_ids
    assert a.requests["done"].output_token_ids == [7]
    assert a.requests["partial"].output_token_ids == []
    got = async_out.get_output()
    assert got.sampled_token_ids == sync_out.sampled_token_ids == [[7], []]
    assert got.req_ids == sync_out.req_ids


def _pending(req_id: str, token: int, done: bool):
    state = SimpleNamespace(output_token_ids=[])
    return plo.PendingSyncOutput(
        copy_state=_copy([token], done=done),
        request_seq_lens=[(0, state, 1, req_id)],
        discard_indices=[],
    )


def test_flush_waits_only_up_to_a_scheduled_request():
    a, b, c = _pending("a", 1, False), _pending("b", 2, False), _pending("c", 3, True)
    committed = []
    commit = lambda p, toks: committed.append((p.request_seq_lens[0][3], toks))  # noqa: E731

    # Nothing scheduled and the oldest copy still in flight: nothing waits,
    # and the landed copy behind it keeps its place.
    pending = deque([a, b, c])
    assert plo.flush_pending(pending, {"x": 1}, commit) == 0
    assert committed == [] and len(pending) == 3
    assert a.copy_state.copy_ready_event.waits == 0

    # b scheduled: a and b are committed (waiting), then c, which landed.
    assert plo.flush_pending(pending, {"b": 1}, commit) == 3
    assert committed == [("a", [[1]]), ("b", [[2]]), ("c", [[3]])]
    assert a.copy_state.copy_ready_event.waits == 1
    assert b.copy_state.copy_ready_event.waits == 1
    assert not pending


def test_flush_commits_landed_copies_without_waiting():
    a, b = _pending("a", 1, True), _pending("b", 2, False)
    committed = []
    pending = deque([a, b])
    assert plo.flush_pending(pending, set(), lambda p, t: committed.append(t)) == 1
    assert committed == [[[1]]] and list(pending) == [b]
    assert b.copy_state.copy_ready_event.waits == 0
    assert plo.flush_pending(pending, None, lambda p, t: committed.append(t)) == 1
    assert b.copy_state.copy_ready_event.waits == 1


def test_discarded_rows_are_not_committed_or_waited_for():
    state = SimpleNamespace(output_token_ids=[])
    p = plo.PendingSyncOutput(
        copy_state=_copy([5], done=False),
        request_seq_lens=[(0, state, 1, "r")],
        discard_indices=[0],
    )
    assert p.req_ids() == []
    pending = deque([p])
    # r is scheduled again (its next chunk) but its sample was thrown away.
    assert plo.flush_pending(pending, {"r": 1}, lambda *_: None) == 0
    assert p.copy_state.copy_ready_event.waits == 0


def test_deferred_commit_skips_requests_that_left_the_batch():
    batch = SimpleNamespace(
        req_id_to_index={"kept": 0, "readded": 1},
        num_tokens_no_spec=[3, 3],
        token_ids_cpu=np.zeros((2, 8), dtype=np.int32),
    )
    kept = SimpleNamespace(output_token_ids=[])
    stale = SimpleNamespace(output_token_ids=[])
    gone = SimpleNamespace(output_token_ids=[])
    requests = {"kept": kept, "readded": SimpleNamespace(output_token_ids=[])}
    p = plo.PendingSyncOutput(
        copy_state=None,
        request_seq_lens=[(0, kept, 4, "kept"), (1, stale, 4, "readded"), (2, gone, 4, "gone")],
        discard_indices=[],
    )
    plo.commit_sampled_tokens(batch, requests, p, [[11], [12], [13]])
    assert kept.output_token_ids == [11] and batch.token_ids_cpu[0, 3] == 11
    assert batch.num_tokens_no_spec == [4, 3]
    assert stale.output_token_ids == [] and gone.output_token_ids == []
    assert requests["readded"].output_token_ids == []


def test_valid_tokens_match_get_output_for_multi_token_rows():
    """Rows wider than one token (padded with INVALID_TOKEN_ID) are read as
    get_output reads them."""
    cs = AsyncTPUCopyState(
        chunks_cpu=[torch.tensor([[4, 5, -1], [6, -1, -1], [8, 9, 10]], dtype=torch.int32)],
        chunk_real_lens=[3],
        copy_ready_event=_Event(True),
    )
    out = AsyncTPUModelRunnerOutput(
        model_runner_output=SimpleNamespace(sampled_token_ids=None, logprobs=None),
        copy_state=cs,
        discard_sampled_tokens_req_indices=[1],
    ).get_output()
    assert plo.valid_sampled_token_ids(cs, [1]) == out.sampled_token_ids == [[4, 5], [], [8, 9, 10]]


def test_worker_replies_from_a_thread_in_call_order(monkeypatch):
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.outputs import AsyncModelRunnerOutput

    monkeypatch.setattr(WorkerProc, "handle_output", WorkerProc.handle_output)
    monkeypatch.setattr(WorkerProc, "_tpu_pp_output_thread", False, raising=False)
    plo.install_worker_output_thread()

    release = threading.Event()

    class Slow(AsyncModelRunnerOutput):
        def get_output(self):
            assert release.wait(10)
            return "step"

    replies = []
    got_all = threading.Event()

    def enqueue(result):
        replies.append(result)
        if len(replies) == 3:
            got_all.set()

    proc = object.__new__(WorkerProc)
    proc.use_async_scheduling = False
    proc.worker_response_mq = SimpleNamespace(enqueue=enqueue)

    proc.handle_output("before")  # no asynchronous output yet: direct
    assert [r[1] for r in replies] == ["before"]
    proc.handle_output(Slow())  # returns at once, the thread waits
    proc.handle_output("after")  # queued behind the step, not ahead of it
    assert [r[1] for r in replies] == ["before"]
    release.set()
    assert got_all.wait(10)
    assert [r[1] for r in replies] == ["before", "step", "after"]
    assert all(r[0] == WorkerProc.ResponseStatus.SUCCESS for r in replies)


def test_worker_with_async_scheduling_keeps_vllms_path(monkeypatch):
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.outputs import AsyncModelRunnerOutput

    calls = []
    monkeypatch.setattr(WorkerProc, "handle_output", lambda self, o: calls.append(o))
    monkeypatch.setattr(WorkerProc, "_tpu_pp_output_thread", False, raising=False)
    plo.install_worker_output_thread()

    class Out(AsyncModelRunnerOutput):
        def get_output(self):
            return "x"

    proc = object.__new__(WorkerProc)
    proc.use_async_scheduling = True
    out = Out()
    proc.handle_output(out)
    assert calls == [out]
    assert "_tpu_pp_output_queue" not in proc.__dict__


class _Stop(Exception):
    pass


@pytest.mark.parametrize("value,installs", [("1", True), ("0", False), (None, False)])
def test_worker_init_installs_the_output_thread_only_when_on(monkeypatch, value, installs):
    from vllm_torchtpu.worker import tpu_worker

    if value is None:
        monkeypatch.delenv(plo.ENV, raising=False)
    else:
        monkeypatch.setenv(plo.ENV, value)
    calls = []
    monkeypatch.setattr(plo, "install_worker_output_thread", lambda: calls.append(1))
    monkeypatch.setattr("vllm_torchtpu.patch_registry.apply", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "vllm_torchtpu.distributed.pp_push.widen_message_rings", lambda *_a: None
    )

    def stop(*_a, **_k):
        raise _Stop

    monkeypatch.setattr(tpu_worker.WorkerBase, "__init__", stop)
    with pytest.raises(_Stop):
        tpu_worker.TPUWorker(
            vllm_config=SimpleNamespace(),
            local_rank=0,
            rank=0,
            distributed_init_method="",
        )
    assert calls == ([1] if installs else [])


@pytest.mark.parametrize("schedules_r0", [True, False])
def test_execute_model_commits_before_update_states(monkeypatch, schedules_r0):
    """The next step commits the deferred token before _update_states when
    it schedules the request again, and does not wait for it otherwise."""
    monkeypatch.setenv(plo.ENV, "1")
    runner = _runner()
    _output, copy = _sample(runner)
    seen = {}

    def update_states(_scheduler_output):
        seen["num_tokens"] = list(runner.input_batch.num_tokens_no_spec)
        seen["waits"] = copy.copy_ready_event.waits
        raise _Stop

    runner._update_states = update_states
    scheduled = {"r0": 1} if schedules_r0 else {"r1": 1}
    with pytest.raises(_Stop):
        TPUModelRunner.execute_model(
            runner,
            SimpleNamespace(num_scheduled_tokens=scheduled, total_num_scheduled_tokens=1),
        )
    if schedules_r0:
        assert seen == {"num_tokens": [2], "waits": 1}
        assert not runner.__dict__["_pp_last_out_pending"]
    else:
        assert seen == {"num_tokens": [1], "waits": 0}
        assert len(runner.__dict__["_pp_last_out_pending"]) == 1


class _Connector:
    """A PD producer's connector: records save, poll and clear, and refuses any
    of them after this step's metadata was cleared."""

    def __init__(self, events):
        self.events, self.bound = events, True

    def _log(self, name):
        assert self.bound, f"{name} after the metadata clear"
        self.events.append(name)

    def wait_for_save(self):
        self._log("save")

    def get_transfer_results(self, _finished):
        self._log("poll")
        return SimpleNamespace(finished_sending={"sent"}, finished_recving=None, failed_recving=None)

    def get_block_ids_with_load_errors(self):
        return set()

    def build_connector_worker_meta(self):
        return None

    def get_kv_connector_stats(self):
        return None

    def clear_connector_metadata(self):
        self._log("clear")
        self.bound = False


@pytest.mark.parametrize("value", ["1", "0"])
def test_connector_calls_and_report_are_unchanged_under_the_deferred_return(monkeypatch, value):
    """A PD producer's last stage: save, poll and clear still run inside
    sample_tokens, in that order, before it returns; the step's report rides
    the (deferred) output exactly as on the synchronous path."""
    monkeypatch.setenv(plo.ENV, value)
    runner = _runner()
    runner._stage_timing.take_samples.return_value = ([], [])
    events: list = []
    connector = _Connector(events)
    copies = []

    def host_copy(chunks, lengths, spec_logprobs=None):
        state = AsyncTPUCopyState(
            chunks_cpu=[c.to(torch.int32) for c in chunks],
            chunk_real_lens=lengths,
            copy_ready_event=_Event(),
        )
        copies.append(state)
        return state

    with (
        patch("vllm_torchtpu.runner.tpu_runner.has_kv_transfer_group", return_value=True),
        patch("vllm_torchtpu.runner.tpu_runner.get_kv_transfer_group", return_value=connector),
        patch.object(AsyncTPUCopyState, "from_device_chunks", side_effect=host_copy),
    ):
        output = TPUModelRunner.sample_tokens(runner, None)
    assert events == ["save", "poll", "clear"]
    deferred = value == "1"
    assert isinstance(output, AsyncTPUModelRunnerOutput) == deferred
    assert copies[0].copy_ready_event.waits == (0 if deferred else 1)
    final = output.get_output() if deferred else output
    assert final.kv_connector_output.finished_sending == {"sent"}
    assert final.sampled_token_ids == [[7]]


def test_engagement_lines_go_to_vllms_logger():
    """The engagement lines (V3 in the benches) must reach vLLM's handler: the
    module logs under the 'vllm.' tree, as vllm_torchtpu.logger arranges. A
    logger named outside it (vllm.logger.init_logger(__name__)) drops INFO."""
    assert getattr(plo.logger, "_name", "").startswith("vllm.vllm_torchtpu.")
