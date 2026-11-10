# SPDX-License-Identifier: Apache-2.0
"""Engine step that keeps empty scheduler steps out of the batch queue (W7 P1b).

vLLM's ``EngineCore.step_with_batch_queue`` queues every scheduler output,
including steps without tokens, which the scheduler returns when everything
it can run is in flight. Each one holds a slot of the batch queue until it
reaches the front, and the engine stops scheduling and blocks on the oldest
step once the queue is full. At the start of a burst, when requests reach
the engine core one at a time, empty steps fill the queue between the first
real steps: the engine then waits out the first step's whole pipeline
latency with only two or three real steps queued, and stage 0 runs dry (the
round-start bubble of the ring hand-off; the wave's settles hide it).

With ``TPU_PP_EMPTY_STEP_GRACE_MS`` > 0 under ``TPU_PP_HANDOFF=ring`` this
step replaces vLLM's. A step without tokens that changes no worker state
(``pp_lazy_step._carries_state``; its scheduler stats come with the next
step) takes no slot: it is neither dispatched nor queued, as the lazy push
loop drops it. If a request reached the engine within the grace period,
the step returns without blocking so the engine core reads its input queue
and schedules the next request; otherwise, and once the grace has passed
with no new input, it waits for the oldest step as vLLM does. Every other
step goes through vLLM's path unchanged.
"""

import time
from collections.abc import Callable
from typing import Any, cast

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Poll interval while waiting for input during the grace period.
_POLL_S = 0.0005
# SchedulerOutput fields that hold configuration rather than worker state on a
# step without tokens: vLLM 0.31 sets num_spec_tokens_to_schedule to the draft
# length on every output, so with MTP every empty step looked stateful.
_CONFIG_FIELDS = frozenset({"num_spec_tokens_to_schedule"})
# Connector metadata fields that hold state only when they carry work: a KV
# connector sets its metadata on every output (our TPUConnectorMetadata is a
# dataclass of request dicts, truthy even when every dict is empty), and the
# steps with tokens in flight poll the connector's finished transfers too.
_CONNECTOR_META_FIELDS = frozenset({"kv_connector_metadata"})


def _empty_metadata(value: Any) -> bool:
    """A dataclass connector metadata whose every field is empty."""
    import dataclasses

    return dataclasses.is_dataclass(value) and not any(
        getattr(value, field.name) for field in dataclasses.fields(value)
    )
# sha256[:16] of the vLLM step this one re-implements (vLLM 0.31.0, both the
# PCP shadow image and the R12p image); any other source keeps vLLM's step.
_VLLM_STEP_SHA = "65c2c7342d075e70"


def vllm_step_matches() -> bool:
    import hashlib
    import inspect

    from vllm.v1.engine.core import EngineCore

    src = inspect.getsource(EngineCore.step_with_batch_queue)
    return hashlib.sha256(src.encode()).hexdigest()[:16] == _VLLM_STEP_SHA


def grace_ms() -> float:
    """TPU_PP_EMPTY_STEP_GRACE_MS: 0 (default) keeps vLLM's step."""
    import os

    value = os.environ.get("TPU_PP_EMPTY_STEP_GRACE_MS", "0").strip()
    try:
        return max(float(value), 0.0)
    except ValueError as e:
        raise ValueError(f"TPU_PP_EMPTY_STEP_GRACE_MS must be a number, not {value!r}") from e


class EmptyStepGrace:
    """Per engine core state of the replacement step."""

    def __init__(self, core: Any, grace_s: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.core = core
        self.grace_s = grace_s
        self.clock = clock
        self.sleep = sleep
        self.known = len(getattr(core.scheduler, "requests", {}))
        self.last_arrival = -1e18
        self.dropped = 0
        self.yielded = 0
        self.kept = 0

    def _note_arrivals(self) -> None:
        known = len(getattr(self.core.scheduler, "requests", {}))
        if known > self.known:
            self.last_arrival = self.clock()
        self.known = known

    def _input_pending(self) -> bool:
        queue = getattr(self.core, "input_queue", None)
        return queue is not None and not queue.empty()

    def _stateful_fields(self, scheduler_output: Any) -> list[str]:
        """The fields that make a step without tokens matter to the workers:
        ``pp_lazy_step._carries_state``'s rule without the configuration
        fields."""
        import dataclasses

        from vllm_torchtpu.core.pp_lazy_step import _STATELESS_FIELDS

        fields = []
        if scheduler_output.scheduled_cached_reqs.req_ids:
            fields.append("scheduled_cached_reqs")
        for field in dataclasses.fields(scheduler_output):
            name = field.name
            if name in _STATELESS_FIELDS or name in _CONFIG_FIELDS:
                continue
            value = getattr(scheduler_output, name)
            if name in _CONNECTOR_META_FIELDS and _empty_metadata(value):
                continue
            if value:
                fields.append(name)
        return fields

    def step(self) -> tuple[dict[int, Any] | None, bool]:

        core = self.core
        batch_queue = core.batch_queue
        assert batch_queue is not None and len(batch_queue) < core.batch_queue_size
        self._note_arrivals()
        model_executed = False
        deferred_scheduler_output = None
        exec_future = None
        if core.scheduler.has_requests():
            scheduler_output = core.scheduler.schedule(core._should_throttle_prefills())
            stateful = (
                self._stateful_fields(scheduler_output)
                if batch_queue and scheduler_output.total_num_scheduled_tokens <= 0
                else None
            )
            if stateful:
                self.kept += 1
                if self.kept <= 5:
                    logger.info(
                        "PP empty-step grace: kept an empty step for its state: %s",
                        ", ".join(stateful),
                    )
            if (
                stateful is not None
                and not stateful
                and not getattr(core.scheduler, "grammar_compile_error_reqs", None)
            ):
                # An empty step holds nothing for the workers: no slot for it.
                self.dropped += 1
                if self.dropped <= 5 or self.dropped % 1000 == 0:
                    logger.info(
                        "PP empty-step grace: %d empty steps kept out of the batch "
                        "queue, %d yields to new input",
                        self.dropped,
                        self.yielded,
                    )
                deadline = self.last_arrival + self.grace_s
                while self.clock() < deadline:
                    if self._input_pending():
                        self.yielded += 1
                        return None, False
                    self.sleep(_POLL_S)
                if self._input_pending():
                    self.yielded += 1
                    return None, False
                # Nothing new arrived: wait for the oldest step, as vLLM does.
            else:
                with core.log_error_detail(scheduler_output):
                    exec_future = core.model_executor.execute_model(
                        scheduler_output, non_block=True
                    )
                if not core.is_mm_encoder_only:
                    model_executed = scheduler_output.total_num_scheduled_tokens > 0
                if core.is_pooling_model or not model_executed:
                    future = exec_future
                elif not scheduler_output.pending_structured_output_tokens:
                    grammar_output = core.scheduler.get_grammar_bitmask(scheduler_output)
                    future = core.model_executor.sample_tokens(grammar_output, non_block=True)
                else:
                    deferred_scheduler_output = scheduler_output
                if not deferred_scheduler_output:
                    batch_queue.appendleft((future, scheduler_output, exec_future))
                    if len(batch_queue) < core.batch_queue_size and (
                        model_executed or core.scheduler.has_requests()
                    ):
                        return None, model_executed
        elif not batch_queue:
            return None, False
        # Block until the next result is available (vLLM's path).
        future, scheduler_output, exec_model_fut = batch_queue.pop()
        with (
            core.capture_iteration_details(scheduler_output) as iteration_details,
            core.log_error_detail(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                exec_model_fut.result()
                raise RuntimeError("unexpected error")
        core._process_aborts_queue()
        engine_core_outputs = core.scheduler.update_from_output(scheduler_output, model_output)
        core._attach_iteration_details(engine_core_outputs, iteration_details)
        if deferred_scheduler_output:
            if core.check_for_draft_tokens:
                draft_token_ids = core.model_executor.take_draft_token_ids()
                if draft_token_ids is not None:
                    core.scheduler.update_draft_token_ids_in_output(
                        draft_token_ids, deferred_scheduler_output
                    )
            grammar_output = core.scheduler.get_grammar_bitmask(deferred_scheduler_output)
            future = core.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))
        return engine_core_outputs, model_executed


def install_empty_step_grace(core: Any) -> bool:
    """Swap the engine core's batch-queue step for ``EmptyStepGrace.step``
    when TPU_PP_EMPTY_STEP_GRACE_MS > 0, the stages hand off with the ring and
    the core runs a batch queue. Returns whether it did."""
    from vllm_torchtpu.distributed.pp_push import ring_handoff

    g = grace_ms()
    if g <= 0:
        return False
    if not ring_handoff():
        logger.warning("TPU_PP_EMPTY_STEP_GRACE_MS ignored: the hand-off is not the ring.")
        return False
    if getattr(core, "batch_queue", None) is None or getattr(core, "async_scheduling", False):
        logger.warning("TPU_PP_EMPTY_STEP_GRACE_MS ignored: no batch queue.")
        return False
    if not vllm_step_matches():
        logger.warning(
            "TPU_PP_EMPTY_STEP_GRACE_MS ignored: vLLM's step_with_batch_queue is not "
            "the one this step re-implements."
        )
        return False
    stepper = EmptyStepGrace(core, g / 1000.0)
    core._tpu_empty_step_grace = stepper
    core.step_fn = cast(Any, stepper.step)
    logger.info(
        "PP empty-step grace: empty steps take no batch-queue slot; %.1f ms grace "
        "for new input (TPU_PP_EMPTY_STEP_GRACE_MS)",
        g,
    )
    return True
