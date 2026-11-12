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
"""Engine-side part of the pipeline hand-off.

Stages pass activations along with collective launches that all stages issue
together (see ``pp_wave``). Stage 0 launches once after each forward it runs
and every other stage launches once before each forward it runs, so a
forward moves one stage further each time the stages run another forward,
and reaches the last stage once ``stages - 2`` later forwards have run. The
engine dispatches scheduler steps, not forwards: a step with tokens runs at
least one forward on every stage (more when the runner splits it into token
chunks), so a step reaches the last stage once ``stages - 2`` later steps
with tokens have been dispatched, and this module counts steps. While the
engine keeps dispatching, this happens by itself. When the engine stops to
wait for a result before that, the launches that would carry the step the
rest of the way are never issued. This module wraps the executor so that,
in that case, the engine first tells every worker to issue them, one
``pp_push`` per missing step (sent without waiting for a reply), and then
waits. A push is a forward that carries nothing, so the forwards in flight
keep their places and the next real step follows them; nothing drains and
nothing restarts. A step without tokens runs no forward, but the workers
answer it only after every call before it, so a wait for it sends what the
last step with tokens dispatched before it still lacks. With
``TPU_PP_LAZY_PUSH`` the engine core sends the pushes itself, each at the
moment stage 0 would otherwise sit idle (``core/pp_lazy_step.py``), and a
wait only settles.

The engine and its workers exchange messages through shared-memory ring
buffers, vLLM's ``MessageQueue``, which hold ten messages by default. The
engine's calls go out on one ring that every worker reads; the last stage
reads a step's calls only after several later steps have been dispatched,
so a short ring would block the engine. Each worker's replies come back on
its own ring, which the engine reads in the order it made the calls; with a
KV connector attached every worker replies to every call, so while the
engine waits for the last stage's reply to an old step, the other stages
keep replying to later steps and a short ring fills and blocks them before
they reach the push message. Both rings are widened to
``rpc_ring_chunks`` messages.

The push is installed on ``MultiprocExecutor`` before the executor is
built: from the platform's config check for engines built in this process,
and from the engine-core process otherwise. It is installed once per
process; every executor keeps its own step count and reads its own pipeline
size, so engines of different depths can share a process. Each process
creates its own rings, the engine its broadcast ring and every worker its
reply ring, so the widening is installed in the engine process and, through
``widen_message_rings``, in every worker process; the ring size is the
widest any pipeline in the process asked for.

With ``TPU_PP_HANDOFF=ring`` the stages hand each forward on point to point
as soon as it is done (``pp_ring``), so no step waits for later ones and
the engine sends no pushes or settles. The engine keeps
``ring_extra_steps`` more steps in flight than the pipeline has stages, so
stage 0's next forward is already queued while the engine core prepares
the following step, and the rings are widened for that count.
``patch_executor_for_pp_handoff`` installs the engine side the flag
selects.
"""

import contextlib
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, cast

from vllm.logger import init_logger

logger = init_logger(__name__)


class _WaveState:
    """One executor's count of the steps it dispatched to the pipeline."""

    def __init__(self, stages: int):
        self.stages = stages
        # Steps with tokens dispatched since startup, pushes included, and
        # the number of the last one the scheduler produced.
        self.steps = 0
        self.last_step = 0
        # Steps the scheduler produced since startup.
        self.real_steps = 0
        # A step was dispatched since the last settle.
        self.open = False
        self.pushes = 0
        self.settles = 0
        # Steps the last settle covered: every launch up to it has paired.
        self.settled = 0
        # The engine core times the pushes itself (core/pp_lazy_step.py).
        self.lazy = False

    def dispatched(self) -> int:
        """Records a scheduler step and returns its number."""
        self.steps += 1
        self.last_step = self.steps
        self.real_steps += 1
        self.open = True
        return self.steps

    def push(self, executor: Any) -> tuple[int, Future]:
        """Sends one ``pp_push`` and returns its step number and the last
        stage's reply."""
        self.steps += 1
        self.open = True
        self.pushes += 1
        if self.pushes <= 5 or self.pushes % 100 == 0:
            logger.debug("PP wave push %d as step %d", self.pushes, self.steps)
        return self.steps, executor.collective_rpc(
            "pp_push", non_block=True, unique_reply_rank=executor.output_rank
        )

    def settle(self, executor: Any) -> Future | None:
        """Sends the ``pp_settle`` that closes the burst and returns the last
        stage's reply; None when no step was dispatched since the last
        settle."""
        if not self.open:
            return None
        self.open = False
        self.settles += 1
        self.settled = self.steps
        if self.settles <= 5 or self.settles % 100 == 0:
            logger.debug(
                "PP wave settle %d after step %d", self.settles, self.last_step
            )
        return executor.collective_rpc(
            "pp_settle", non_block=True, unique_reply_rank=executor.output_rank
        )

    def before_waiting(self, executor: Any, step: int, real_step: int) -> None:
        """Send what carries ``step`` to the last stage before the engine
        waits for it: the settle when it is the newest step the scheduler
        produced, since nothing else is in flight and the stages can rest
        afterwards; otherwise one ``pp_push`` for every later step it still
        lacks, counting the pushes as later steps for the waits that
        follow. With lazy pushes the engine core sends the pushes, and a
        wait settles. Nothing is sent for a step a settle has carried."""
        if not self.open or step <= self.settled:
            return
        # With two stages the launches pair without pushes; the settle still
        # goes out and issues stage 0's queued hand-off.
        if self.lazy or real_step == self.real_steps:
            self.settle(executor)
            return
        need = self.stages - 2 - (self.steps - step)
        if need <= 0:
            return
        self.pushes += need
        if self.pushes <= 5 or self.pushes % 100 == 0:
            logger.debug(
                "PP wave: %d pushes (%d total) before waiting on step %d of %d",
                need,
                self.pushes,
                step,
                self.steps,
            )
        for _ in range(need):
            executor.collective_rpc(
                "pp_push", non_block=True, unique_reply_rank=executor.output_rank
            )
        self.steps += need


class _PushFuture(Future):
    """The result of one forward, sampling or empty-step call.

    Every method forwards to the executor's own future. The two that wait
    for it, ``result`` and ``exception``, first send what carries this
    step to the last stage: the settle when nothing else is in flight,
    pushes otherwise. With lazy pushes the engine core reads finished steps
    only, and reading one sends nothing.
    """

    def __init__(
        self, inner: Future, executor: Any, wave: _WaveState, step: int, real_step: int
    ):
        super().__init__()
        self._inner = inner
        self._executor = executor
        self._wave = wave
        self._step = step
        self._real_step = real_step

    def _before_waiting(self) -> None:
        if not (self._wave.lazy and self._inner.done()):
            self._wave.before_waiting(self._executor, self._step, self._real_step)

    def result(self, timeout=None):
        self._before_waiting()
        return self._inner.result(timeout)

    def exception(self, timeout=None):
        self._before_waiting()
        # The executor's future resolves only inside result(); a stored
        # exception raises there and is read back below.
        with contextlib.suppress(Exception):
            self._inner.result(timeout)
        return self._inner.exception(timeout=0)

    def done(self):
        return self._inner.done()

    def running(self):
        return self._inner.running()

    def cancelled(self):
        return self._inner.cancelled()

    def cancel(self):
        return self._inner.cancel()

    def add_done_callback(self, fn):
        self._inner.add_done_callback(lambda _: fn(self))


def wave_state(executor: Any) -> _WaveState:
    """The executor's step count, created on first use."""
    state = getattr(executor, "_tpu_wave_state", None)
    if state is None:
        state = _WaveState(
            int(executor.vllm_config.parallel_config.pipeline_parallel_size)
        )
        executor._tpu_wave_state = state
    return state


def ring_extra_steps(stages: int) -> int:
    """Steps the engine keeps in flight beyond the stage count under the
    ring.

    Stage 0 is the ring's feed: when the last stage samples a step, the
    step that refills stage 0 reaches it only after the engine core has
    updated, scheduled and sent it and the worker has prepared its inputs,
    and stage 0 sits idle for that time every step. Each extra step queues
    one more forward on stage 0's device, so that time overlaps a forward.
    Half the stage count reached the floor on an eight-stage prefill
    engine; one extra step recovered half of the gain.
    """
    return max(1, stages // 2)


def rpc_ring_chunks(stages: int, extra: int = 0) -> int:
    """Messages each ring buffer between the engine and its workers must hold.

    The engine keeps up to ``stages + extra`` steps in flight, each of them
    two calls (execute_model and sample_tokens), so a worker reads a step's
    calls up to that many steps late and, with a KV connector, holds up to
    that many unread replies; a wait adds at most ``stages - 2`` pushes
    or one settle.
    """
    return 4 * (stages + extra) + 8


# Messages every ring created in this process must hold: the widest any
# pipeline in the process asked for.
_ring_chunks_needed = 0


def _widen_rpc_ring(stages: int, extra: int = 0) -> None:
    """Give every ``MessageQueue`` created from now on, the engine's
    broadcast to its workers and each worker's reply ring, at least
    ``rpc_ring_chunks(stages, extra)`` messages; a deeper pipeline raises
    the minimum, a shallower one leaves it."""
    global _ring_chunks_needed
    _ring_chunks_needed = max(_ring_chunks_needed, rpc_ring_chunks(stages, extra))
    from vllm.distributed.device_communicators import shm_broadcast

    cls = shm_broadcast.MessageQueue
    if getattr(cls, "_tpu_pp_wave_patch", False):
        return
    orig_init = cls.__init__

    def __init__(self, n_reader, n_local_reader, *args, **kwargs):
        kwargs["max_chunks"] = max(
            int(kwargs.get("max_chunks", 10)), _ring_chunks_needed
        )
        orig_init(self, n_reader, n_local_reader, *args, **kwargs)

    cls.__init__ = __init__
    cls._tpu_pp_wave_patch = True


def widen_message_rings(vllm_config: Any) -> None:
    """Widen the message rings this process will create for a pipeline.

    A worker calls it before vLLM builds the worker's reply ring; without a
    pipeline it does nothing. Under the ring the rings also hold the extra
    steps the engine keeps in flight (``ring_extra_steps``), and the
    worker's config counts them as the engine's does.
    """
    if vllm_config is None:
        return
    stages = int(vllm_config.parallel_config.pipeline_parallel_size)
    if stages > 1:
        extra = ring_extra_steps(stages) if ring_handoff() else 0
        _widen_rpc_ring(stages, extra)
        if extra:
            _raise_steps_in_flight(extra)


def ring_handoff() -> bool:
    """Whether pipeline stages hand forwards on point to point (``pp_ring``)
    rather than with the collective wave (``pp_wave``)."""
    from vllm_torchtpu import envs

    return envs.TPU_PP_HANDOFF == "ring"


def _install_take_draft_token_ids(executor_cls: Any) -> None:
    """Have a pipeline executor's ``take_draft_token_ids`` return None
    without asking the workers; an executor without a pipeline keeps the
    class's own method.

    A pipeline admits prefill-only MTP alone (pp_validation): its drafts
    exist to fill the draft layer's KV, and no step consumes them. The
    engine asks with a blocking RPC right after dispatching a step, which
    the last stage answers only once that step is done; under the wave,
    asking the workers would also deadlock, as a stage can still sit in that
    step's hand-off, waiting for a push that only a wait on the step's
    future sends.
    """
    if getattr(executor_cls, "_tpu_pp_take_draft_patch", False):
        return
    orig = getattr(executor_cls, "take_draft_token_ids", None)

    def take_draft_token_ids(self: Any) -> Any:
        if int(self.vllm_config.parallel_config.pipeline_parallel_size) > 1:
            return None
        return orig(self) if orig is not None else None

    executor_cls.take_draft_token_ids = take_draft_token_ids
    executor_cls._tpu_pp_take_draft_patch = True


def _rides_the_wave(executor: Any) -> bool:
    """Whether an executor's steps go through the wave: a pipeline whose
    stages hand off with it. Executors of other depths share the class."""
    stages = int(executor.vllm_config.parallel_config.pipeline_parallel_size)
    return stages > 1 and not ring_handoff()


def _raise_steps_in_flight(extra: int) -> None:
    """Have ``VllmConfig.max_concurrent_batches`` count ``extra`` more steps
    for a pipeline, so the engine keeps that many steps in flight beyond
    its stages.

    The engine core reads the property when it is built, so this runs from
    the same places as the executor patch, before it. Pipelines of one
    stage, and engines with async scheduling, which size the queue for
    their own overlap, keep vLLM's count. Installed once per process: the
    first call's ``extra`` stands for every engine the process builds.
    """
    from vllm.config import VllmConfig

    if getattr(VllmConfig, "_tpu_pp_extra_steps", None) is not None:
        return
    original = cast(property, VllmConfig.max_concurrent_batches)
    original_fget = cast(Callable[[Any], int], original.fget)

    def max_concurrent_batches(self: Any) -> int:
        steps = original_fget(self)
        if (
            self.parallel_config.pipeline_parallel_size > 1
            and not self.scheduler_config.async_scheduling
        ):
            return steps + extra
        return steps

    # vLLM declares the property through its getter; pyrefly types the class
    # attribute as that function.
    # pyrefly: ignore[bad-assignment]
    VllmConfig.max_concurrent_batches = property(
        max_concurrent_batches, doc=original.__doc__
    )
    VllmConfig._tpu_pp_extra_steps = extra


def patch_executor_for_pp_handoff(vllm_config: Any) -> None:
    """Install the engine side of the hand-off ``TPU_PP_HANDOFF`` selects."""
    if ring_handoff():
        patch_executor_for_pp_ring(vllm_config)
    else:
        patch_executor_for_pp_wave(vllm_config)


def patch_executor_for_pp_ring(vllm_config: Any) -> None:
    """Install the ring hand-off's engine side: the extra steps in flight,
    the wider ring buffers, and draft tokens taken without asking the
    workers.

    Does nothing without a pipeline; safe to call more than once per
    process.
    """
    if vllm_config is None:
        return
    stages = int(vllm_config.parallel_config.pipeline_parallel_size)
    if stages <= 1:
        return
    extra = ring_extra_steps(stages)
    _widen_rpc_ring(stages, extra)
    _raise_steps_in_flight(extra)
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    if getattr(MultiprocExecutor, "_tpu_pp_ring_patch", False):
        return
    _install_take_draft_token_ids(MultiprocExecutor)
    MultiprocExecutor._tpu_pp_ring_patch = True
    logger.info(
        "PP ring: engine side installed, %d steps in flight, message ring %d chunks",
        stages + extra,
        _ring_chunks_needed,
    )


def patch_executor_for_pp_wave(vllm_config: Any) -> None:
    """Install the push, the settle and the wider ring buffer on
    ``MultiprocExecutor``.

    Does nothing without a pipeline; safe to call more than once per
    process, and a later call for a deeper pipeline widens the rings.
    """
    if vllm_config is None:
        return
    stages = int(vllm_config.parallel_config.pipeline_parallel_size)
    if stages <= 1:
        return
    _widen_rpc_ring(stages)
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    if getattr(MultiprocExecutor, "_tpu_pp_wave_patch", False):
        return
    orig_execute = MultiprocExecutor.execute_model
    orig_sample = MultiprocExecutor.sample_tokens

    def execute_model(self, scheduler_output, non_block=False):
        if not _rides_the_wave(self):
            return orig_execute(self, scheduler_output, non_block=non_block)
        state = wave_state(self)
        if scheduler_output.total_num_scheduled_tokens <= 0:
            # No forward, but the workers answer it after every earlier
            # call: the wait carries the last step with tokens so far.
            inner = cast(Future, orig_execute(self, scheduler_output, non_block=True))
            future = _PushFuture(inner, self, state, state.steps, state.real_steps)
            return future if non_block else future.result()
        step = state.dispatched()
        # For pooling models and for steps without sampling the engine
        # waits on this future directly, so it pushes like the sampling one.
        inner = cast(Future, orig_execute(self, scheduler_output, non_block=True))
        future = _PushFuture(inner, self, state, step, state.real_steps)
        return future if non_block else future.result()

    def sample_tokens(self, grammar_output, non_block=False):
        if not _rides_the_wave(self):
            return orig_sample(self, grammar_output, non_block=non_block)
        state = wave_state(self)
        inner = cast(Future, orig_sample(self, grammar_output, non_block=True))
        # Sampling belongs to the last dispatched step, whatever pushes
        # went out since.
        future = _PushFuture(inner, self, state, state.last_step, state.real_steps)
        return future if non_block else future.result()

    MultiprocExecutor.execute_model = execute_model
    MultiprocExecutor.sample_tokens = sample_tokens
    _install_take_draft_token_ids(MultiprocExecutor)
    MultiprocExecutor._tpu_pp_wave_patch = True
    logger.info(
        "PP wave: engine push and settle installed, message ring %d chunks",
        _ring_chunks_needed,
    )
