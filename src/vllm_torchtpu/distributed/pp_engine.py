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
"""Engine-side part of the pipeline hand-off (``pp_buffered``).

The engine keeps ``buffered_extra_steps`` more steps in flight than the
pipeline has stages, so stage 0's next forward is already queued while the
engine core prepares the following step, and a pipeline engine takes no
draft tokens from its workers.

The engine and its workers exchange messages through shared-memory ring
buffers, vLLM's ``MessageQueue``, which hold ten messages by default: the
engine's calls go out on one ring that every worker reads, and each
worker's replies come back on its own ring, which the engine reads in the
order it made the calls. A worker reads a step's calls, and with a KV
connector holds its replies, up to as many steps late as the engine keeps
in flight, so both rings are widened to ``rpc_ring_chunks`` messages.

``patch_executor_for_pp_buffered`` installs this engine side before the
executor is built: from the platform's config check for engines built in
this process, and from the engine-core process otherwise. Every worker
process widens its own reply ring (``widen_message_rings``); the ring size
is the widest any pipeline in the process asked for.
"""

from collections.abc import Callable
from typing import Any, cast

from vllm.logger import init_logger

logger = init_logger(__name__)


def buffered_extra_steps(stages: int) -> int:
    """Steps the engine keeps in flight beyond the stage count under the
    buffered hand-off.

    Stage 0 feeds the pipeline: when the last stage samples a step, the
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
    that many unread replies; eight more messages are headroom.
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
    if getattr(cls, "_tpu_pp_rings_widened", False):
        return
    orig_init = cls.__init__

    def __init__(self, n_reader, n_local_reader, *args, **kwargs):
        kwargs["max_chunks"] = max(
            int(kwargs.get("max_chunks", 10)), _ring_chunks_needed
        )
        orig_init(self, n_reader, n_local_reader, *args, **kwargs)

    cls.__init__ = __init__
    cls._tpu_pp_rings_widened = True


def widen_message_rings(vllm_config: Any) -> None:
    """Widen the message rings this process will create for a pipeline.

    A worker calls it before vLLM builds the worker's reply ring; without a
    pipeline it does nothing. The rings also hold the extra steps the
    engine keeps in flight (``buffered_extra_steps``), and the worker's
    config counts them as the engine's does.
    """
    if vllm_config is None:
        return
    stages = int(vllm_config.parallel_config.pipeline_parallel_size)
    if stages > 1:
        _widen_rpc_ring(stages, buffered_extra_steps(stages))
        _raise_steps_in_flight()


def _install_take_draft_token_ids(executor_cls: Any) -> None:
    """Have a pipeline executor's ``take_draft_token_ids`` return None
    without asking the workers; an executor without a pipeline keeps the
    class's own method.

    A pipeline admits prefill-only MTP alone (pp_validation): its drafts
    exist to fill the draft layer's KV, and no step consumes them. The
    engine asks with a blocking RPC right after dispatching a step, which
    the last stage answers only once that step is done.
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


def _raise_steps_in_flight() -> None:
    """Have ``VllmConfig.max_concurrent_batches`` count
    ``buffered_extra_steps`` more steps for a pipeline, so the engine keeps
    that many steps in flight beyond its stages.

    The engine core reads the property when it is built, so this runs from
    the same places as the executor patch, before it. Pipelines of one
    stage, and engines with async scheduling, which size the queue for
    their own overlap, keep vLLM's count. Installed once per process; each
    config counts the extra steps for its own stage count.
    """
    from vllm.config import VllmConfig

    if getattr(VllmConfig, "_tpu_pp_steps_patch", False):
        return
    original = cast(property, VllmConfig.max_concurrent_batches)
    original_fget = cast(Callable[[Any], int], original.fget)

    def max_concurrent_batches(self: Any) -> int:
        steps = original_fget(self)
        stages = self.parallel_config.pipeline_parallel_size
        if stages > 1 and not self.scheduler_config.async_scheduling:
            return steps + buffered_extra_steps(stages)
        return steps

    # vLLM declares the property through its getter; pyrefly types the class
    # attribute as that function.
    # pyrefly: ignore[bad-assignment]
    VllmConfig.max_concurrent_batches = property(
        max_concurrent_batches, doc=original.__doc__
    )
    VllmConfig._tpu_pp_steps_patch = True


def patch_executor_for_pp_buffered(vllm_config: Any) -> None:
    """Install the buffered hand-off's engine side: the extra steps in flight,
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
    extra = buffered_extra_steps(stages)
    _widen_rpc_ring(stages, extra)
    _raise_steps_in_flight()
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    if getattr(MultiprocExecutor, "_tpu_pp_buffered_patch", False):
        return
    _install_take_draft_token_ids(MultiprocExecutor)
    MultiprocExecutor._tpu_pp_buffered_patch = True
    logger.info(
        "PP buffered: engine side installed, %d steps in flight, "
        "message ring %d chunks",
        stages + extra,
        _ring_chunks_needed,
    )
