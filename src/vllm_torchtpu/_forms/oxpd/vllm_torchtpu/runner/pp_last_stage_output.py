# SPDX-License-Identifier: Apache-2.0
"""Keeps the last pipeline stage's host off its own device's critical path.

A pipeline producer runs with synchronous scheduling (``--no-async-scheduling``),
so the last stage's ``sample_tokens`` waits for its step's sampled-token copy
(``AsyncTPUModelRunnerOutput.get_output``) before it returns. That wait lasts
until the step's forward has run on the device. Only then does the worker
reply, read the next step's ``execute_model`` (unpickle), add its requests,
update its states and prepare and launch its forward, while its device sits
idle. On a pipeline whose last stage binds, that host time is part of the
pipeline period: measured 16.0 ms of a 68.3 ms period per 6,400-token step at
FR7 kernels (fr7s j-b6e3865c, plecap j-3de85f34, m16 j-61d4e8ad).

``TPU_PP_LAST_STAGE_ASYNC_OUTPUT=1`` removes that wait on the last stage of a
pipeline with synchronous scheduling. ``sample_tokens`` returns the step's
``AsyncTPUModelRunnerOutput`` without waiting. The worker hands it to an
output thread (``install_worker_output_thread``), which waits for the copy and
replies to the engine in call order, while the worker's main thread goes on to
the next step. The runner's own bookkeeping of the sampled tokens
(``token_ids_cpu``, ``num_tokens_no_spec``, ``output_token_ids``) is kept as a
``PendingSyncOutput`` and committed by ``flush_pending`` at the start of a
later ``execute_model``: before ``_update_states`` whenever that step
schedules one of its requests (the commit then waits for the copy, so the step
sees exactly what the synchronous path left), and otherwise as soon as its
copy has landed.

The engine sees a step's tokens at the same moment as before (the output
thread replies once the copy has landed, and the copy's ready event keeps any
fence the runner put on it); only the worker's next-step host work moves
earlier, under the device's current forward. The ring hand-off bounds how far
the last stage runs ahead (``pp_ring``: end_forward blocks at its depth).

Port of IX's item 6 (``ix/xfer/pcpperf/item6_last_stage_async_output.diff``)
to the q38 runner. Differences: the mode applies only to a last stage without
a drafter (the q38 producer has no speculative config; the drafts part of the
IX change is not ported), and the deferred tokens are read with the q38 copy
state's own accessors.

``0`` (the default) keeps the synchronous path.
"""

import os
import queue
import threading
from collections import deque
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu.logger import init_logger

from vllm_torchtpu.runner.tpu_runner_async_output import (
    INVALID_TOKEN_ID,
    AsyncTPUCopyState,
)

logger = init_logger(__name__)

ENV = "TPU_PP_LAST_STAGE_ASYNC_OUTPUT"

OFF = 0
ON = 1


def mode_from_env() -> int:
    value = os.environ.get(ENV, "0").strip().lower()
    if value in ("", "0", "off", "false"):
        return OFF
    if value in ("1", "on", "true"):
        return ON
    raise ValueError(f"{ENV}={value!r}: expected 0 or 1")


def runner_mode(
    *,
    pipeline_parallel_size: int,
    is_last_stage: bool,
    async_scheduling: bool,
    is_pooling_model: bool,
    has_drafter: bool,
) -> int:
    """The mode that applies to one runner: only the last stage of a pipeline
    with synchronous scheduling, never a pooling model, and never a runner
    with a drafter (its synchronous path waits for host drafts; not ported)."""
    mode = mode_from_env()
    if (
        mode == OFF
        or pipeline_parallel_size <= 1
        or not is_last_stage
        or async_scheduling
        or is_pooling_model
    ):
        return OFF
    if has_drafter:
        logger.warning(
            "%s=%s ignored: the last stage has a drafter (not supported)",
            ENV,
            os.environ.get(ENV),
        )
        return OFF
    return mode


def mode_for_runner(runner: Any) -> int:
    """``runner_mode`` for a TPUModelRunner, resolved once per runner. An
    object without the configs (a test double) is off."""
    cached = runner.__dict__.get("_pp_last_out_mode")
    if cached is not None:
        return cached
    parallel_config = getattr(runner, "parallel_config", None)
    scheduler_config = getattr(runner, "scheduler_config", None)
    if parallel_config is None or scheduler_config is None:
        mode = OFF
    else:
        mode = runner_mode(
            pipeline_parallel_size=int(parallel_config.pipeline_parallel_size),
            is_last_stage=bool(getattr(runner, "_pp_is_last", False)),
            async_scheduling=bool(scheduler_config.async_scheduling),
            is_pooling_model=bool(getattr(runner, "is_pooling_model", False)),
            has_drafter=bool(getattr(runner, "_is_async_drafter", False))
            or getattr(runner, "speculative_config", None) is not None,
        )
        if mode != OFF:
            logger.info(
                "Last pipeline stage: %s=%s (sampled tokens returned without "
                "waiting, committed later)",
                ENV,
                os.environ.get(ENV),
            )
    runner.__dict__["_pp_last_out_mode"] = mode
    return mode


@dataclass
class PendingSyncOutput:
    """A step whose sampled tokens the runner has not committed yet.

    ``request_seq_lens`` and ``discard_indices`` are what the synchronous path
    of ``sample_tokens`` computed for the step: ``(row, req_state, seq_len,
    req_id)`` per request that keeps its sample, and the rows whose sample is
    thrown away (partial prefills).
    """

    copy_state: AsyncTPUCopyState | None
    request_seq_lens: list[tuple[int, Any, int, str]]
    discard_indices: list[int]

    def req_ids(self) -> list[str]:
        discard = set(self.discard_indices)
        return [r[3] for r in self.request_seq_lens if r[0] not in discard]

    def ready(self) -> bool:
        cs = self.copy_state
        if cs is None or cs.completed:
            return True
        return bool(cs.copy_ready_event.query())


def valid_sampled_token_ids(
    copy_state: AsyncTPUCopyState | None, discard_indices: Collection[int]
) -> list[list[int]]:
    """The per-row tokens ``AsyncTPUModelRunnerOutput.get_output`` reports,
    computed without touching the output object (the worker's output thread
    may be sending it). Waits for the copy."""
    if copy_state is None:
        return []
    copy_state.wait()
    t = copy_state.sampled_token_ids_cpu
    if t.numel() == 0:
        return []
    if t.dim() == 1:
        t = t.unsqueeze(-1)
    if t.shape[-1] == 1:
        out = t.tolist()
    else:
        out = [[tok for tok in row if tok != INVALID_TOKEN_ID] for row in t.tolist()]
    for i in discard_indices:
        out[i] = []
    return out


def flush_pending(
    pending: "deque[PendingSyncOutput]",
    scheduled_req_ids: Collection[str] | None,
    commit: Callable[[PendingSyncOutput, list[list[int]]], None],
) -> int:
    """Commits pending steps, oldest first, and returns how many.

    Every step up to the newest one that holds a request in
    ``scheduled_req_ids`` is committed, waiting for its copy if needed;
    ``None`` commits them all. After that, steps are committed while their
    copies have already landed, so nothing else waits.
    """
    if not pending:
        return 0
    if scheduled_req_ids is None:
        last_needed = len(pending) - 1
    else:
        last_needed = -1
        for i, p in enumerate(pending):
            if any(r in scheduled_req_ids for r in p.req_ids()):
                last_needed = i
    done = 0
    while pending and (done <= last_needed or pending[0].ready()):
        p = pending.popleft()
        commit(p, valid_sampled_token_ids(p.copy_state, p.discard_indices))
        done += 1
    return done


def commit_sampled_tokens(
    input_batch: Any,
    requests: dict[str, Any],
    pending: PendingSyncOutput,
    sampled_token_ids: list[list[int]],
) -> None:
    """The persistent-batch update the synchronous path of ``sample_tokens``
    makes for a step, applied later. Requests that left the batch since the
    step, or were added again, are skipped."""
    discard = pending.discard_indices
    for i, req_state, _seq_len, req_id in pending.request_seq_lens:
        if i in discard:
            continue
        valid_tokens = sampled_token_ids[i]
        if not valid_tokens:
            continue
        if requests.get(req_id) is not req_state:
            continue
        req_idx = input_batch.req_id_to_index.get(req_id)
        if req_idx is None:
            continue
        start_tok_idx = input_batch.num_tokens_no_spec[req_idx]
        end_tok_idx = start_tok_idx + len(valid_tokens)
        input_batch.token_ids_cpu[req_idx, start_tok_idx:end_tok_idx] = valid_tokens
        input_batch.num_tokens_no_spec[req_idx] = end_tok_idx
        req_state.output_token_ids.extend(valid_tokens)


def install_worker_output_thread() -> None:
    """Routes a worker's replies through an output thread once it returns an
    ``AsyncModelRunnerOutput`` under synchronous scheduling.

    vLLM's ``WorkerProc`` has such a thread only with async scheduling;
    without it, ``enqueue_output`` calls ``get_output()`` on the worker's main
    thread, which waits for the step. The thread starts at the first
    asynchronous output, and from then on every reply goes through its queue,
    so replies keep the order of the calls. Workers that never return an
    asynchronous output (every stage but the last) never start it.
    """
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.outputs import AsyncModelRunnerOutput

    if getattr(WorkerProc, "_tpu_pp_output_thread", False):
        return
    orig = WorkerProc.handle_output

    def handle_output(self, output: Any) -> None:
        q = self.__dict__.get("_tpu_pp_output_queue")
        if q is None:
            if getattr(self, "use_async_scheduling", False) or not isinstance(
                output, AsyncModelRunnerOutput
            ):
                orig(self, output)
                return
            q = queue.Queue()
            self._tpu_pp_output_queue = q
            threading.Thread(
                target=_output_loop,
                args=(self, q),
                daemon=True,
                name="WorkerPPOutputCopy",
            ).start()
            logger.info("Last pipeline stage: replies go through an output thread")
        q.put(output)

    WorkerProc.handle_output = handle_output
    WorkerProc._tpu_pp_output_thread = True


def _output_loop(worker_proc: Any, q: "queue.Queue[Any]") -> None:
    while True:
        worker_proc.enqueue_output(q.get())
