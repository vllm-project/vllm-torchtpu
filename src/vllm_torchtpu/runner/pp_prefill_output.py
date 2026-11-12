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
"""Automatic PP producer output overlap, without asynchronous scheduling.

Only the last stage owns this path. One immutable output may await device
completion while the next execute prepares and submits its forward. Sampling
then retires that output before creating another. Upstream WorkerProc retains
ownership of response ordering, serialization and error transport.
"""

import contextvars
import functools
import os
import queue
import threading

from vllm.v1.outputs import AsyncModelRunnerOutput

_installed = False


class PendingOutput(AsyncModelRunnerOutput):
    def __init__(self, output, drafts, request_ids):
        self.output = output
        self.kv_connector_output = output.get_kv_connector_output()
        self.context = contextvars.copy_context()
        self.drafts = drafts
        self.request_ids = frozenset(request_ids)
        self.lock = threading.Lock()
        self.completed = False
        self.result = None
        self.error = None

    def get_kv_connector_output(self):
        return self.kv_connector_output

    def get_output(self):
        # Non-reply ranks retire through the owner; reply ranks may already
        # be materializing on WorkerProc's output thread. Exactly one reader.
        with self.lock:
            if not self.completed:
                try:
                    self.result = self.context.run(self.output.get_output)
                except BaseException as exc:
                    self.error = exc
                finally:
                    self.completed = True
                    self.output = None
                    self.drafts = None
            if self.error is not None:
                raise self.error
            return self.result


class ResponseQueue(queue.Queue):
    """Notify after the upstream output loop initializes its thread device."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def get(self, *args, **kwargs):
        with self.owner.replies:
            self.owner.dispatch_ready = True
            self.owner.replies.notify_all()
        return super().get(*args, **kwargs)


class PrefillOutput:
    def __init__(self, runner):
        self.runner = runner
        self.active = False
        self.pending = None
        self.dispatch_started = False
        self.dispatch_ready = False
        self.replies = threading.Condition()
        self.submitted = 0
        self.enqueued = 0
        self.dispatch_error = None

    def check_dispatch(self):
        if self.dispatch_error is not None:
            raise RuntimeError(
                "PP prefill reply dispatch failed"
            ) from self.dispatch_error

    def begin_execute(self, scheduled):
        self.check_dispatch()
        # _update_states mutates shared host staging before _prepare_inputs.
        # This fence covers uploads preceding the old main, not its MTP end.
        self.runner._wait_input_staging_fence()
        invalidated = (
            scheduled.finished_req_ids
            | (scheduled.preempted_req_ids or set())
            | scheduled.scheduled_cached_reqs.resumed_req_ids
        )
        if self.pending is not None and invalidated & self.pending.request_ids:
            self.retire()
        self.active = scheduled.total_num_scheduled_tokens > 0

    def retire(self):
        if self.pending is not None:
            pending, self.pending = self.pending, None
            pending.get_output()

    def wrap(self, output, drafts, request_ids):
        assert self.pending is None, "Previous PP prefill output was not retired"
        self.pending = PendingOutput(output, drafts, request_ids)
        return self.pending

    def enable_dispatch(self, worker):
        self.check_dispatch()
        if self.dispatch_started:
            return
        # This WorkerProc flag controls only its reply dispatch. Neither
        # EngineCore nor the runner's scheduler_config is changed.
        assert not worker.use_async_scheduling
        # Bound device results through retire(), not a blocking Queue.put:
        # a dead consumer must never strand the worker on a full queue.
        worker.async_output_queue = ResponseQueue(self)
        worker.use_async_scheduling = True

        def run():
            try:
                worker.async_output_busy_loop()
            except BaseException as exc:
                with self.replies:
                    self.dispatch_error = exc
                    self.replies.notify_all()

        worker.async_output_copy_thread = threading.Thread(
            target=run,
            daemon=True,
            name="PPPrefillOutputCopy",
        )
        self.dispatch_started = True
        worker.async_output_copy_thread.start()
        with self.replies:
            self.replies.wait_for(
                lambda: self.dispatch_ready or self.dispatch_error is not None
            )
            self.check_dispatch()

    def drain(self):
        self.retire()
        # Materialization completion alone does not prove the FIFO replies
        # reached their MQ. The upstream loop has no queue.task_done().
        with self.replies:
            self.replies.wait_for(
                lambda: self.enqueued == self.submitted
                or self.dispatch_error is not None
            )
            if self.dispatch_error is not None:
                raise RuntimeError(
                    "PP prefill reply dispatch failed"
                ) from self.dispatch_error


def create(runner):
    config = runner.vllm_config
    is_ring_producer = (
        config.parallel_config.pipeline_parallel_size > 1
        and os.environ.get("TPU_PP_HANDOFF") == "ring"
        and config.kv_transfer_config is not None
        and config.kv_transfer_config.kv_role == "kv_producer"
    )
    if not is_ring_producer or not runner._mtp_prefill_only:
        return None
    if (
        config.parallel_config.data_parallel_size != 1
        or config.scheduler_config.async_scheduling
        or runner.is_pooling_model
    ):
        raise ValueError(
            "PP prefill output requires a synchronous single-DP MTP producer"
        )
    if not runner._pp_is_last:
        return None
    if not runner._fence_spec_kv_under_token_copy:
        raise ValueError("PP prefill output requires the draft KV completion fence")
    install_worker()
    return PrefillOutput(runner)


def install_worker():
    global _installed
    if _installed:
        return
    from vllm.v1.executor.multiproc_executor import WorkerProc

    from vllm_torchtpu.worker.tpu_worker import TPUWorker

    execute = TPUWorker.execute_model
    sample = TPUWorker.sample_tokens
    handle = WorkerProc.handle_output
    enqueue = WorkerProc.enqueue_output
    rpc = WorkerProc._execute_worker_rpc

    def owner(worker):
        return worker.worker.worker.model_runner._pp_prefill_output

    @functools.wraps(execute)
    def execute_model(self, scheduler_output):
        output = self.model_runner._pp_prefill_output
        if output is not None:
            output.begin_execute(scheduler_output)
        return execute(self, scheduler_output)

    @functools.wraps(sample)
    def sample_tokens(self, grammar_output):
        output = self.model_runner._pp_prefill_output
        if output is None:
            return sample(self, grammar_output)
        # The next execute has already prepared and submitted its main.
        try:
            output.retire()
            return sample(self, grammar_output)
        finally:
            output.active = False

    @functools.wraps(handle)
    def handle_output(self, result):
        output = owner(self)
        if output is not None:
            output.enable_dispatch(self)
            with output.replies:
                output.submitted += 1
        return handle(self, result)

    @functools.wraps(enqueue)
    def enqueue_output(self, result):
        output = owner(self)
        if output is None:
            return enqueue(self, result)
        try:
            return enqueue(self, result)
        except BaseException as exc:
            with output.replies:
                output.dispatch_error = exc
            raise
        finally:
            with output.replies:
                output.enqueued += 1
                output.replies.notify_all()

    @functools.wraps(rpc)
    def execute_rpc(self, request):
        output = owner(self)
        if output is not None and request[0] not in ("execute_model", "sample_tokens"):
            try:
                output.drain()
            except Exception as exc:
                # Use upstream's error transport for this control reply too.
                if request[3] is None or self.rank == request[3]:
                    self.handle_output(exc)
                return
        return rpc(self, request)

    # Preserve upstream process-death shutdown: it closes MQs before device
    # teardown. Waiting for reply delivery here can deadlock on a dead reader.
    # The owner keeps pending sources alive through that teardown; normal
    # control RPCs already use drain() above.

    TPUWorker.execute_model = execute_model
    TPUWorker.sample_tokens = sample_tokens
    WorkerProc.handle_output = handle_output
    WorkerProc.enqueue_output = enqueue_output
    WorkerProc._execute_worker_rpc = execute_rpc
    _installed = True
