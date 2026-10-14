# SPDX-License-Identifier: Apache-2.0
"""Hand-off of activations between pipeline stages over ICI.

One hand-off is a single collective-permute that all stages run together:
each stage sends one buffer to the next stage and receives one from the
previous stage (the last stage's buffer goes to stage 0, which ignores it).
Because every stage takes part in every hand-off, launch number p on one
stage pairs with launch number p on every other stage. A stage of several
devices (a context-parallel group) hands off once per device: each device
holds its own rows of the chunk and permutes them with the devices holding
the same rows in the other stages, its column.

What travels is the sum of hidden states and residual, one tensor in the
model dtype. The receiving stage's first norm adds hidden states and
residual in float32, normalizes that sum, and keeps it rounded to the model
dtype as the new residual. Given the rounded sum and a zero residual, the
norm keeps the identical residual but normalizes a sum rounded once more
than in a single-stage model: a relative difference of at most 2^-8 per
element for bfloat16. Sending the sum halves the bytes on the link; sending
both tensors would reproduce the single-stage arithmetic exactly.

A model may need side data to travel with the activations: a sparse-attention
model whose later layers reuse the top-k indices an earlier layer chose has to
carry that table across a stage boundary, because the layer that would
recompute it lives on another chip. Such a tensor rides the same launch as a
further tensor. A stage that does not recompute a tensor passes on what it
received.

A forward here is one runner forward over one token chunk; the runner splits
a scheduler step into as many forwards as its token budget needs, and each
forward is one hand-off. Launch numbers count from 0 within a burst.

Stage 0 launches once right after each forward it runs; every other stage
launches once right before each forward, because it needs the incoming
activations first, and that launch carries the output of the stage's
previous forward (zeros when there is none). Prologue launches of zeros fill
the numbers before a stage's first real one: one on stage 0, r on stage r.
So the output of stage 0's forward m goes out on launch m+1, stage r receives
it on launch m+r, right before its own forward m, and it reaches the last of
S stages on launch m+S-1. A stage that is ahead of the others queues its
launches on its device; its host never waits.

Three stages, two forwards, then two pushes that carry forward 1 to the
last stage. Each row is one launch number and each entry the buffer that
stage sends on it:

    launch  stage 0             stage 1               stage 2
    0       prologue: zeros     prologue: zeros       prologue: zeros
    1       after fwd 0: out 0  before fwd 0: zeros   prologue: zeros
    2       after fwd 1: out 1  before fwd 1: out 0   before fwd 0: zeros
    3       after push: zeros   before push: out 1    before fwd 1: zeros
    4       after push: zeros   before push: zeros    before push: zeros

On launch p stage r receives what stage r-1 sent on launch p: stage 1 gets
stage 0's output 0 on launch 1, stage 2 gets stage 1's output 0 on launch 2
and its output 1 on launch 3.

Launches pair up only while every stage keeps launching, which they do
while the engine keeps dispatching steps. When the engine stops to wait
for a result, forwards can be stuck part-way, because the launches that
would carry them further are never issued. A push (``push``) is a forward
that carries nothing: every stage issues the one launch it would issue for
a forward, sending what it still holds and then zeros, so every forward in
flight moves one stage further and the numbering stays as it is. The
engine side (``pp_push``) sends as many pushes as the waited-for step still
needs, counting scheduler steps. A stage that is ahead keeps its unpaired
launches queued on its device until the others catch up; once nothing is
in flight the engine settles (``settle``): each stage launches whatever it
still holds, then enough zeros to bring all stages to the same count, so
the devices are quiet while the engine is idle. The forwards between two
settles form a burst; the next burst starts with the prologue again.

Two rules keep the launches pairing across stages: every launch runs the
same compiled program (one buffer padded to the largest token bucket), and
every launch is submitted to the device right away, even on a stage that
never reads the result.

A stage that spans more than one chip drains its device and waits for every
worker of the slice before each hand-off: with a stage program and a hand-off
running at the same time, a core of the stage stops inside a stage collective
and the runtime halts the device. A one-chip stage never showed this and
launches the hand-off without waiting.

Stage 0 launches its hand-off in end_forward, after its forward. On a
multi-chip stage that must drain before each hand-off, draining on the worker
thread would block host preparation for the next step (scheduling,
KV-connector setup, input staging) until the forward pass finishes on the
device. Stage 0 therefore offloads the drain to a background helper thread:
- end_forward flushes the outgoing tensors, records a device fence, and queues
  the hand-off so the worker thread can immediately prepare the next step.
- The helper thread only waits on the fence and the slice barrier; it never
  traces ops while the worker thread is inside a runner step, because flushing
  from another thread could split the lazy graph at different points across
  ranks.
- Once drained, the helper launches the hand-off immediately if the worker
  thread is between runner calls. If the worker thread is inside a step, the
  worker issues the queued launch itself at the next start_forward, push, or
  settle. Either way the launch goes out without a further engine action,
  which pp_push's accounting relies on: it counts stage 0's launch as issued
  once execute_model returns and may wait without sending a push.
- Before starting its next forward, stage 0 explicitly waits for the previous
  hand-off to finish on the device, since stage 0 does not read the received
  buffers and has no data dependency to order the forward behind the hand-off.
- Pushes and settles always issue any queued hand-off first. Even on a
  two-stage pipeline where stage 0 issues no settle launches of its own, settle
  waits on the helper thread so any drain or launch failure is raised at the
  engine's next wait.
"""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.distributed.pp_shift import get_or_create_pp_mesh, pp_permute_op
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.tracing.annotation import TraceAnnotation

if TYPE_CHECKING:
    from torch_tpu._internal.stream.streams import TpuEvent

logger = init_logger(__name__)

# The intermediate tensors every stage passes on.
HIDDEN_STATES = "hidden_states"
RESIDUAL = "residual"


def pp_rank_flags(parallel_config: Any) -> tuple[bool, bool]:
    """(is_first_stage, is_last_stage) for this worker."""
    if parallel_config.pipeline_parallel_size <= 1:
        return True, True
    from vllm.distributed.parallel_state import get_pp_group

    group = get_pp_group()
    return bool(group.is_first_rank), bool(group.is_last_rank)


def stage_chips() -> int:
    """How many chips this worker's pipeline stage spans."""
    from vllm_torchtpu.distributed.chip_topology import get_chip_topology
    from vllm_torchtpu.distributed.pcp import get_pcp_group_layout

    topology = get_chip_topology()
    if topology is None:
        return 1
    device_ids = set(get_pcp_group_layout().device_ids)
    return sum(1 for chip in topology.chips if device_ids & set(chip))


def _drain_slice(fence: "TpuEvent | None" = None) -> None:
    """Waits for local device work (or the given fence) to complete, then joins
    the slice barrier so no stage forward is running when the hand-off starts."""
    import torch.distributed as dist
    from torch_tpu._internal.sync import sync
    from vllm.distributed.parallel_state import get_world_group

    if fence is None:
        sync.synchronize(None, wait=True)
    else:
        fence.synchronize()
    dist.barrier(group=get_world_group().cpu_group)


def _launch_now(op, carried: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Run one hand-off as its own device program, submitted immediately.

    vLLM runs torch_tpu in fused-eager mode, where ops are deferred, fused
    with their neighbours and submitted only when a result is read. Pending
    ops are flushed first, then the launch alone; neither flush waits.
    """
    from torch_tpu._internal.sync import sync

    sync.synchronize(None, wait=False)
    out = op(*carried)
    out = (out,) if isinstance(out, torch.Tensor) else tuple(out)
    sync.synchronize(list(out), wait=False)
    return out


def _flush(tensors: tuple[torch.Tensor, ...]) -> None:
    """Submits the pending ops producing ``tensors`` without blocking."""
    from torch_tpu._internal.sync import sync

    sync.synchronize(list(tensors), wait=False)


def _wait(tensors: tuple[torch.Tensor, ...]) -> None:
    """Blocks until ``tensors`` are materialized on the device."""
    from torch_tpu._internal.sync import sync

    sync.synchronize(list(tensors), wait=True)


def _fence() -> "TpuEvent":
    """Records a device event for all work submitted so far."""
    from torch_tpu._internal.stream.streams import TpuEvent

    event = TpuEvent()
    event.record()
    return event


class PPWave:
    def __init__(
        self,
        device: torch.device,
        max_rows: int,
        template: dict[str, tuple[tuple[int, ...], torch.dtype]],
    ):
        from vllm.distributed.parallel_state import get_pp_group

        from vllm_torchtpu.distributed.pcp import get_pcp_rank, get_pcp_world_size

        group = get_pp_group()
        self.rank = int(group.rank_in_group)
        self.size = int(group.world_size)
        self.last = self.size - 1
        # This device's column in its stage: the same column of every stage
        # holds the same token rows.
        self.column = get_pcp_rank()
        self.columns = get_pcp_world_size()
        extras = [key for key in template if key not in (HIDDEN_STATES, RESIDUAL)]
        if (
            set(template) != {HIDDEN_STATES, RESIDUAL, *extras}
            or template[HIDDEN_STATES] != template[RESIDUAL]
            or any(len(template[key][0]) != 1 for key in (HIDDEN_STATES, *extras))
        ):
            raise ValueError(
                "the pipeline hand-off carries hidden_states and residual, "
                "two [tokens, hidden] tensors of one dtype, plus any number "
                "of [tokens, width] side tensors; the model's intermediate "
                f"tensors are {template}"
            )
        # The first is the activation sum; the rest are side data a stage
        # passes on unchanged. Every stage builds from the same template,
        # so the order matches across stages.
        self.carried = (HIDDEN_STATES, *extras)
        self.max_rows = int(max_rows)
        self.op = pp_permute_op(get_or_create_pp_mesh(), num_tensors=len(self.carried))
        # See the module docstring.
        self.drain_before_handoff = stage_chips() > 1
        self.async_handoff = (
            self.rank == 0 and self.size > 1 and self.drain_before_handoff
        )
        # State for at most one in-flight async hand-off on stage 0, moving from
        # queued (_async_queued) to draining (_async_busy), drained
        # (_async_drained), issuing (_async_issuing), and completed (_async_out).
        self._async_cv = threading.Condition()
        self._async_queued: tuple[tuple[torch.Tensor, ...], TpuEvent | None] | None = (
            None
        )
        self._async_busy = False
        self._async_drained: tuple[torch.Tensor, ...] | None = None
        self._async_issuing = False
        self._async_out: tuple[torch.Tensor, ...] | None = None
        self._async_error: BaseException | None = None
        self._in_step = 0
        self._async_thread: threading.Thread | None = None
        self.zeros = tuple(
            torch.zeros(
                (self.max_rows, *template[key][0]),
                dtype=template[key][1],
                device=device,
            )
            for key in self.carried
        )
        self.zero = self.zeros[0]
        self.launches = 0
        self.bursts = 0
        self.burst_open = False
        self.burst_launches = 0
        self.forwards = 0
        self.pending: tuple[torch.Tensor, ...] | None = None
        logger.info(
            "PP wave: rank %d/%d column %d/%d, hand-off carries %s%s",
            self.rank,
            self.size,
            self.column,
            self.columns,
            ", ".join(
                f"{key}[{self.max_rows}, {template[key][0][0]}] {template[key][1]}"
                for key in self.carried
            ),
            ", launched from a helper thread" if self.async_handoff else "",
        )

    def _launch_blocking(
        self, carried: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        with TraceAnnotation("PP:Handoff"):
            if self.drain_before_handoff:
                _drain_slice()
            return _launch_now(self.op, carried)

    def _launch(self, carried: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        self._issue_queued()
        out = self._launch_blocking(carried)
        self.launches += 1
        self.burst_launches += 1
        return out

    def _async_idle(self) -> bool:
        return (
            self._async_queued is None
            and not self._async_busy
            and self._async_drained is None
            and not self._async_issuing
            and self._async_out is None
        )

    @contextmanager
    def step(self) -> Iterator[None]:
        """Marks a worker-thread region that traces device ops so the helper
        thread does not launch a hand-off concurrently."""
        with self._async_cv:
            self._async_cv.wait_for(lambda: not self._async_issuing)
            self._in_step += 1
        try:
            yield
        finally:
            with self._async_cv:
                self._in_step -= 1
                self._async_cv.notify_all()

    def _launch_async(self, carried: tuple[torch.Tensor, ...]) -> None:
        """Flushes the outgoing tensors and queues the drain on the helper
        thread; the hand-off is issued by whichever thread is free first."""
        _flush(carried)
        fence = _fence() if self.drain_before_handoff else None
        self.launches += 1
        self.burst_launches += 1
        if self._async_thread is None:
            self._async_thread = threading.Thread(
                target=self._async_loop, name="pp-wave-handoff", daemon=True
            )
            self._async_thread.start()
        with self._async_cv:
            self._async_cv.wait_for(self._async_idle)
            self._async_queued = (carried, fence)
            self._async_cv.notify_all()

    def _async_loop(self) -> None:
        while True:
            with self._async_cv:
                self._async_cv.wait_for(lambda: self._async_queued is not None)
                queued = self._async_queued
                assert queued is not None
                carried, fence = queued
                self._async_queued = None
                self._async_busy = True
            error: BaseException | None = None
            try:
                if self.drain_before_handoff:
                    with TraceAnnotation("PP:Drain"):
                        _drain_slice(fence)
            except BaseException as exc:  # re-raised on the worker thread
                logger.exception(
                    "PP wave rank %d: the hand-off thread failed to drain",
                    self.rank,
                )
                error = exc
            with self._async_cv:
                self._async_busy = False
                if error is not None:
                    self._async_error = error
                    self._async_cv.notify_all()
                    continue
                self._async_drained = carried
                self._async_cv.notify_all()
                # Launch immediately if the worker thread is outside a step;
                # otherwise leave it for the worker's next launch point.
                self._async_cv.wait_for(
                    lambda: self._async_drained is None or self._in_step == 0
                )
                if self._async_drained is None:
                    continue
                self._async_drained = None
                self._async_issuing = True
            out: tuple[torch.Tensor, ...] | None = None
            try:
                with TraceAnnotation("PP:Handoff"):
                    out = _launch_now(self.op, carried)
            except BaseException as exc:  # re-raised on the worker thread
                logger.exception(
                    "PP wave rank %d: the hand-off thread failed to launch",
                    self.rank,
                )
                error = exc
            with self._async_cv:
                self._async_issuing = False
                self._async_out = out
                if error is not None:
                    self._async_error = error
                self._async_cv.notify_all()

    def _issue_queued(self) -> tuple[torch.Tensor, ...] | None:
        """Waits for any queued drain, issues the hand-off if the helper has not,
        propagates helper errors, and returns the received buffers."""
        if not self.async_handoff:
            return None
        with self._async_cv:
            # Wait without a separate timeout: process-group barrier timeouts
            # surface via _async_error, and device hangs are handled by the engine.
            self._async_cv.wait_for(
                lambda: self._async_queued is None
                and not self._async_busy
                and not self._async_issuing
            )
            if self._async_error is not None:
                error, self._async_error = self._async_error, None
                raise RuntimeError(
                    f"PP wave rank {self.rank}: the hand-off thread failed"
                ) from error
            carried, self._async_drained = self._async_drained, None
            out, self._async_out = self._async_out, None
        if carried is not None:
            with TraceAnnotation("PP:Handoff"):
                out = _launch_now(self.op, carried)
        return out

    def _pad(
        self, carried: tuple[torch.Tensor, ...], rows: int
    ) -> tuple[torch.Tensor, ...]:
        if rows == self.max_rows:
            return carried
        return tuple(torch.cat([t, z[rows:]]) for t, z in zip(carried, self.zeros))

    def warmup(self) -> None:
        """One launch outside any burst on every device, carrying a value
        known by its stage and column; checks that it arrived on the same
        column of the next stage."""
        assert not self.burst_open and self.launches == 0
        sent = self.rank * self.columns + self.column + 1
        out = self._launch(tuple(torch.full_like(z, sent) for z in self.zeros))
        if self.rank > 0:
            want = float(sent - self.columns)
            for key, x in zip(self.carried, out):
                got = (float(x[0, 0].item()), float(x[-1, -1].item()))
                if any(g != want for g in got):
                    raise RuntimeError(
                        f"PP wave rank {self.rank}: hand-off self-test tensor "
                        f"{key!r} received {got}, expected {want}"
                    )
        else:
            _ = out[0][0, 0].item()
        self.burst_launches = 0

    def _open_burst(self) -> None:
        """Stage r's launch before its forward m is number m+r and stage 0's
        launch after its forward m is number m+1; empty launches fill the
        numbers before the first real one."""
        if self.burst_open:
            return
        self.burst_open = True
        self.bursts += 1
        self.burst_launches = 0
        self.forwards = 0
        with TraceAnnotation("PP:Prologue"):
            for _ in range(max(self.rank, 1)):
                self._launch(self.zeros)

    def _check_count(self, extra: int = 0) -> None:
        expected = self.forwards + max(self.rank, 1) + extra
        if self.burst_launches != expected:
            raise RuntimeError(
                f"PP wave rank {self.rank}: burst of {self.forwards} forwards "
                f"issued {self.burst_launches} launches, expected {expected}"
            )

    def start_forward(self, rows: int) -> IntermediateTensors | None:
        """Called before forward number m of the burst with its padded token
        count. Returns the incoming activations (None on the first stage)."""
        self._open_burst()
        if self.rank == 0:
            out = self._issue_queued()
            if out is not None:
                # Stage 0 does not consume the received buffers, so explicitly
                # wait to keep the next forward from overlapping the hand-off.
                _wait(out)
            return None
        send = self.pending if self.pending is not None else self.zeros
        self.pending = None
        out = self._launch(send)
        # Copies: the next launch overwrites the received buffers. The
        # residual is zero: the sum arrived as the hidden states.
        tensors = {key: x[:rows].clone() for key, x in zip(self.carried, out)}
        tensors[RESIDUAL] = self.zero[:rows].clone()
        return IntermediateTensors(tensors)

    def end_forward(self, tensors: dict[str, torch.Tensor] | None, rows: int) -> None:
        """Called after the forward with its outgoing activations (None on
        the last stage)."""
        self.forwards += 1
        if self.rank == self.last:
            self._check_count()
            return
        assert tensors is not None
        missing = [key for key in self.carried if key not in tensors]
        if missing:
            raise RuntimeError(
                f"PP wave rank {self.rank}: the forward produced no "
                f"{missing} to hand on"
            )
        # Summed in the model dtype; see the module docstring.
        total = tensors[HIDDEN_STATES] + tensors[RESIDUAL]
        outgoing = (total, *(tensors[key] for key in self.carried[1:]))
        padded = self._pad(outgoing, rows)
        if self.rank != 0:
            self.pending = padded
        elif self.async_handoff:
            self._launch_async(padded)
        else:
            self._launch(padded)
        self._check_count()

    def push(self) -> None:
        """A forward that carries nothing: the one launch this stage issues
        for a forward, sending what it still holds (zeros otherwise), so
        every forward in flight moves one stage further."""
        self._open_burst()
        with TraceAnnotation("PP:Push"):
            if self.rank == 0:
                self._launch(self.zeros)
            else:
                send = self.pending if self.pending is not None else self.zeros
                self.pending = None
                self._launch(send)
        self.forwards += 1
        if 0 < self.rank < self.last:
            self.pending = self.zeros
        self._check_count()

    def settle(self) -> None:
        """Bring every stage to the same launch count and close the burst;
        called once nothing is in flight."""
        if not self.burst_open:
            return
        with TraceAnnotation("PP:Settle"):
            if self.rank == 0:
                # Issue any queued hand-off and surface helper errors even on
                # a two-stage pipeline where stage 0 issues no settle launches.
                self._issue_queued()
            if 0 < self.rank < self.last:
                assert self.pending is not None
                self._launch(self.pending)
                self.pending = None
            zero_launches = self.size - 2 - self.rank if self.rank < self.last else 0
            for _ in range(zero_launches):
                self._launch(self.zeros)
        # Every stage now holds forwards + size - 1 launches.
        self._check_count(self.size - 1 - max(self.rank, 1))
        if self.bursts <= 5 or self.bursts % 100 == 0:
            logger.debug(
                "PP wave rank %d: burst %d settled after %d forwards, %d "
                "launches (%d total)",
                self.rank,
                self.bursts,
                self.forwards,
                self.burst_launches,
                self.launches,
            )
        self.burst_open = False
        self.forwards = 0
        self.burst_launches = 0
