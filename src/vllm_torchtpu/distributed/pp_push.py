# SPDX-License-Identifier: Apache-2.0
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
nothing restarts.

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
"""

import contextlib
from concurrent.futures import Future
from typing import Any

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

    def dispatched(self) -> int:
        """Records a scheduler step and returns its number."""
        self.steps += 1
        self.last_step = self.steps
        self.real_steps += 1
        self.open = True
        return self.steps

    def before_waiting(self, executor: Any, step: int, real_step: int) -> None:
        """Send what carries ``step`` to the last stage before the engine
        waits for it: the settle when it is the newest step the scheduler
        produced, since nothing else is in flight and the stages can rest
        afterwards; otherwise one ``pp_push`` for every later step it still
        lacks, counting the pushes as later steps for the waits that
        follow."""
        # With one or two stages every launch pairs on its own: stage 1's
        # launch before forward m is stage 0's launch after it.
        if not self.open or self.stages <= 2:
            return
        if real_step == self.real_steps:
            self.open = False
            self.settles += 1
            if self.settles <= 5 or self.settles % 100 == 0:
                logger.debug(
                    "PP wave settle %d before waiting on step %d", self.settles, step
                )
            executor.collective_rpc(
                "pp_settle", non_block=True, unique_reply_rank=executor.output_rank
            )
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
    """The result of one forward or sampling call.

    Every method forwards to the executor's own future. The two that wait
    for it, ``result`` and ``exception``, first send what carries this
    step to the last stage: the settle when nothing else is in flight,
    pushes otherwise.
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

    def result(self, timeout=None):
        self._wave.before_waiting(self._executor, self._step, self._real_step)
        return self._inner.result(timeout)

    def exception(self, timeout=None):
        self._wave.before_waiting(self._executor, self._step, self._real_step)
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


def rpc_ring_chunks(stages: int) -> int:
    """Messages each ring buffer between the engine and its workers must hold.

    The engine keeps up to ``stages`` steps in flight, each of them two
    calls (execute_model and sample_tokens), so a worker reads a step's
    calls up to that many steps late and, with a KV connector, holds up to
    that many unread replies; a wait adds at most ``stages - 2`` pushes
    or one settle.
    """
    return 4 * stages + 8


# Messages every ring created in this process must hold: the widest any
# pipeline in the process asked for.
_ring_chunks_needed = 0


def _widen_rpc_ring(stages: int) -> None:
    """Give every ``MessageQueue`` created from now on, the engine's
    broadcast to its workers and each worker's reply ring, at least
    ``rpc_ring_chunks(stages)`` messages; a deeper pipeline raises the
    minimum, a shallower one leaves it."""
    global _ring_chunks_needed
    _ring_chunks_needed = max(_ring_chunks_needed, rpc_ring_chunks(stages))
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
    pipeline it does nothing.
    """
    if vllm_config is None:
        return
    stages = int(vllm_config.parallel_config.pipeline_parallel_size)
    if stages > 1:
        _widen_rpc_ring(stages)


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

    def _state(executor) -> _WaveState:
        state = getattr(executor, "_tpu_wave_state", None)
        if state is None:
            state = _WaveState(
                int(executor.vllm_config.parallel_config.pipeline_parallel_size)
            )
            executor._tpu_wave_state = state
        return state

    def execute_model(self, scheduler_output, non_block=False):
        state = _state(self)
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return orig_execute(self, scheduler_output, non_block=non_block)
        step = state.dispatched()
        # For pooling models and for steps without sampling the engine
        # waits on this future directly, so it pushes like the sampling one.
        inner = orig_execute(self, scheduler_output, non_block=True)
        future = _PushFuture(inner, self, state, step, state.real_steps)
        return future if non_block else future.result()

    def sample_tokens(self, grammar_output, non_block=False):
        state = _state(self)
        inner = orig_sample(self, grammar_output, non_block=True)
        # Sampling belongs to the last dispatched step, whatever pushes
        # went out since.
        future = _PushFuture(inner, self, state, state.last_step, state.real_steps)
        return future if non_block else future.result()

    MultiprocExecutor.execute_model = execute_model
    MultiprocExecutor.sample_tokens = sample_tokens
    MultiprocExecutor._tpu_pp_wave_patch = True
    logger.info(
        "PP wave: engine push and settle installed, message ring %d chunks",
        _ring_chunks_needed,
    )
