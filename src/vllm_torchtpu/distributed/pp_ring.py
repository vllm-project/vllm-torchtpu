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
"""Point-to-point hand-off of activations between pipeline stages.

Each device sends the output of a forward to the device that holds the same
token rows in the next stage, its column, with ``torch.distributed.isend``,
and that device takes it with ``irecv``. A transfer involves only the two
devices at its ends and runs no program on either: the receiver posts a
landing buffer, and once the sender's data is ready the runtime copies it
there. So a stage waits only for its own input, never for the other stages
to reach the same point, and a forward moves on as soon as it is done.

What travels is what ``pp_wave`` carries: the sum of hidden states and
residual in the model dtype, plus any side tensor such as a sparse model's
top-k table. The next stage starts from that sum and a zero residual.

A forward is one runner forward over one token chunk. Every stage runs the
same sequence of forwards, so a count of transfers kept by each process
numbers them alike on every stage; the warmup is transfer 1 and each
forward takes the next number, which, with the tensor's place, gives the
tag it travels with. A step without tokens runs no forward and transfers
nothing.

A receive names the exact shape it takes. Every stage gets a scheduler step
before it runs any forward of it, and the padded token count of each
forward follows from the step, so at the start of the step (``begin_step``)
a stage posts one receive per forward with those counts, before it prepares
its inputs; a forward started without one (the warmup forwards) posts its
own. The posted tensor is bound to a fresh landing buffer. The receiver does
not wait for the data on the host: the forward reads the tensor, and the
runtime runs the forward once the data has landed.

The sender hands its output on from a helper thread. end_forward submits the
ops that produce the outgoing tensors, records a device fence and queues the
transfer; the helper waits on the fence, calls ``isend`` for each tensor and
waits until the copies are done, keeping the tensors alive until then.
``isend`` returns only once the receiver has posted the matching receive:
that is the credit, a stage cannot send a forward before the next stage has
taken up its step. At most ``depth`` forwards wait in a stage's queue, the
one being sent included; end_forward blocks until there is room, which
bounds the memory a stage that runs ahead holds for its outputs. The engine
keeps vLLM's ``stages`` steps in flight, as with the wave (``pp_push``).

Unlike the wave, nothing here needs the stages to issue the same launches:
there is no prologue, the engine sends no pushes and no settles (``push``
and ``settle`` do nothing), a stage of several chips neither drains its
device nor waits at a world barrier before a hand-off, and the only programs
a stage runs across its devices are its forwards.
"""

import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.distributed.pp_wave import HIDDEN_STATES, RESIDUAL, _fence, _flush
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.tracing.annotation import TraceAnnotation

if TYPE_CHECKING:
    from torch_tpu._internal.stream.streams import TpuEvent

logger = init_logger(__name__)

# Tags are C ints; transfer numbers wrap well before they would overflow one.
_TAG_SPAN = 1 << 30

_LIBTPU_REQUIREMENT = (
    "TPU_PP_HANDOFF=ring needs a libtpu whose transfers between hosts land in "
    "the posted receive, as the libtpu nightly of 2026-10-27 with the "
    "torch_tpu nightly of 2026-10-28 does; TPU_PP_HANDOFF=wave does not"
)


def _first_and_last(x: torch.Tensor) -> tuple[float, float]:
    """Reads a received tensor's first and last element on the host; the
    runtime runs the read after the data has landed."""
    return float(x[0, 0].item()), float(x[-1, -1].item())


class PPRing:
    # Read by the worker's pp_drains_before_handoff: no stage waits for the
    # slice before a hand-off.
    drain_before_handoff = False

    def __init__(
        self,
        device: torch.device,
        max_rows: int,
        template: dict[str, tuple[tuple[int, ...], torch.dtype]],
        depth: int = 2,
    ):
        from vllm.distributed.parallel_state import get_pp_group

        from vllm_torchtpu.distributed.pcp import get_pcp_rank, get_pcp_world_size

        group = get_pp_group()
        self.rank = int(group.rank_in_group)
        self.size = int(group.world_size)
        self.last = self.size - 1
        # The pipeline group of a device holds its column of every stage:
        # the devices that hold the same token rows.
        ranks = [int(rank) for rank in group.ranks]
        self.global_rank = ranks[self.rank]
        self.prev = ranks[self.rank - 1] if self.rank > 0 else None
        self.next = ranks[self.rank + 1] if self.rank < self.last else None
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
        if depth < 1:
            raise ValueError(f"the ring needs a depth of at least 1, got {depth}")
        # The first is the activation sum; the rest are side data a stage
        # passes on unchanged. Every stage builds from the same template,
        # so the order matches across stages.
        self.carried = (HIDDEN_STATES, *extras)
        self.template = dict(template)
        self.device = device
        self.max_rows = int(max_rows)
        self.depth = int(depth)
        # Transfers started: the warmup's, then one per forward.
        self.transfers = 0
        # The last transfer whose receives this stage has posted.
        self.posted_through = 0
        self.forwards = 0
        self.sent = 0
        # Receives posted ahead of their forwards: (transfer, rows, tensors).
        self._posted: deque[tuple[int, int, tuple[torch.Tensor, ...]]] = deque()
        # The transfer of the forward between start_forward and end_forward.
        self._current = 0
        # Sends queued for the helper thread, the one in progress first:
        # (transfer, tensors, fence).
        self._cv = threading.Condition()
        self._sends: deque[tuple[int, tuple[torch.Tensor, ...], TpuEvent]] = deque()
        self._closing = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        logger.info(
            "PP ring: rank %d/%d column %d/%d (global rank %d, from %s, to %s), "
            "depth %d, hand-off carries %s",
            self.rank,
            self.size,
            self.column,
            self.columns,
            self.global_rank,
            self.prev,
            self.next,
            self.depth,
            ", ".join(
                f"{key}[rows, {template[key][0][0]}] {template[key][1]}"
                for key in self.carried
            ),
        )

    def _tag(self, transfer: int, index: int) -> int:
        return (transfer * len(self.carried) + index) % _TAG_SPAN

    def _post(self, transfer: int, rows: int) -> tuple[torch.Tensor, ...]:
        """Posts the receives of one transfer of ``rows`` rows and returns
        the tensors they land in."""
        import torch.distributed as dist

        if not 0 < rows <= self.max_rows:
            raise RuntimeError(
                f"PP ring rank {self.rank}: transfer {transfer} of {rows} rows, "
                f"the hand-off holds 1 to {self.max_rows}"
            )
        if transfer != self.posted_through + 1:
            raise RuntimeError(
                f"PP ring rank {self.rank}: posting transfer {transfer} after "
                f"{self.posted_through}"
            )
        received = []
        for index, key in enumerate(self.carried):
            shape, dtype = self.template[key]
            tensor = torch.empty((rows, *shape), dtype=dtype, device=self.device)
            # The returned work does not wait for the data; the forward that
            # reads the tensor runs after it lands.
            dist.irecv(tensor, src=self.prev, tag=self._tag(transfer, index))
            received.append(tensor)
        self.posted_through = transfer
        return tuple(received)

    def _raise_send_failure(self) -> None:
        with self._cv:
            self._raise_send_failure_locked()

    def _raise_send_failure_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"PP ring rank {self.rank}: the send thread failed"
            ) from self._error

    @contextmanager
    def step(self) -> Iterator[None]:
        """Marks a runner call that traces device ops. The send thread traces
        none, so there is nothing to keep out of it."""
        yield

    def warmup(self) -> None:
        """One small transfer of every carried tensor from each device to
        the same column of the next stage, carrying the sender's global
        rank; checks that what arrived came from the same column of the
        previous stage."""
        import torch.distributed as dist

        assert self.transfers == 0 and not self._posted
        self.transfers = 1
        rows = 2
        try:
            received: tuple[torch.Tensor, ...] = ()
            # Every stage posts before it sends, and a send waits only for
            # the next stage's post.
            if self.prev is not None:
                received = self._post(self.transfers, rows)
            if self.next is not None:
                sent = tuple(
                    torch.full(
                        (rows, *self.template[key][0]),
                        self.global_rank + 1,
                        dtype=self.template[key][1],
                        device=self.device,
                    )
                    for key in self.carried
                )
                _flush(sent)
                works = [
                    dist.isend(x, dst=self.next, tag=self._tag(self.transfers, i))
                    for i, x in enumerate(sent)
                ]
                for work in works:
                    work.wait()
            got = [_first_and_last(x) for x in received]
        except Exception as exc:
            raise RuntimeError(
                f"PP ring rank {self.rank}: the hand-off self-test failed. "
                f"{_LIBTPU_REQUIREMENT}."
            ) from exc
        if self.prev is None:
            return
        for key, values in zip(self.carried, got):
            dtype = self.template[key][1]
            want = float(torch.tensor(self.prev + 1).to(dtype).item())
            if any(value != want for value in values):
                raise RuntimeError(
                    f"PP ring rank {self.rank}: hand-off self-test tensor "
                    f"{key!r} received {values}, expected {want} from global "
                    f"rank {self.prev}. {_LIBTPU_REQUIREMENT}."
                )

    def begin_step(self, rows_per_forward: list[int]) -> None:
        """Called before the first forward of a scheduler step with the
        padded token count of each of its forwards; posts their receives on
        every stage but the first."""
        self._raise_send_failure()
        if self._posted:
            raise RuntimeError(
                f"PP ring rank {self.rank}: {len(self._posted)} receives were "
                "posted for forwards that never ran"
            )
        if self.rank == 0 or not rows_per_forward:
            return
        with TraceAnnotation("PP:PostReceives"):
            for rows in rows_per_forward:
                transfer = self.posted_through + 1
                self._posted.append((transfer, rows, self._post(transfer, rows)))

    def start_forward(self, rows: int) -> IntermediateTensors | None:
        """Called before each forward with its padded token count. Returns
        the incoming activations (None on the first stage)."""
        self._raise_send_failure()
        self.transfers += 1
        self.forwards += 1
        self._current = self.transfers
        if self.rank == 0:
            return None
        if self._posted:
            transfer, posted_rows, received = self._posted.popleft()
        else:
            transfer, posted_rows = self.transfers, rows
            received = self._post(transfer, rows)
        if transfer != self.transfers or posted_rows != rows:
            raise RuntimeError(
                f"PP ring rank {self.rank}: forward {self.transfers} of {rows} "
                f"rows found the receive of transfer {transfer} posted for "
                f"{posted_rows} rows"
            )
        tensors = dict(zip(self.carried, received))
        # The residual is zero: the sum arrived as the hidden states.
        shape, dtype = self.template[RESIDUAL]
        tensors[RESIDUAL] = torch.zeros((rows, *shape), dtype=dtype, device=self.device)
        return IntermediateTensors(tensors)

    def end_forward(self, tensors: dict[str, torch.Tensor] | None, rows: int) -> None:
        """Called after the forward with its outgoing activations (None on
        the last stage); queues them for the next stage."""
        self._raise_send_failure()
        if self.rank == self.last:
            return
        assert tensors is not None
        missing = [key for key in self.carried if key not in tensors]
        if missing:
            raise RuntimeError(
                f"PP ring rank {self.rank}: the forward produced no "
                f"{missing} to hand on"
            )
        # Summed in the model dtype; see pp_wave's module docstring.
        total = tensors[HIDDEN_STATES] + tensors[RESIDUAL]
        outgoing = (total, *(tensors[key] for key in self.carried[1:]))
        # Sending a view would build its copy on the send thread; the copy is
        # made here, on the thread that traces the step.
        outgoing = tuple(x if x._base is None else x.clone() for x in outgoing)
        wrong = {
            key: tuple(x.shape)
            for key, x in zip(self.carried, outgoing)
            if x.shape[0] != rows
        }
        if wrong:
            raise RuntimeError(
                f"PP ring rank {self.rank}: forward {self._current} of {rows} "
                f"rows hands on {wrong}"
            )
        _flush(outgoing)
        self._queue_send(self._current, outgoing, _fence())

    def _queue_send(
        self, transfer: int, outgoing: tuple[torch.Tensor, ...], fence: "TpuEvent"
    ) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._send_loop, name="pp-ring-send", daemon=True
            )
            self._thread.start()
        with self._cv:
            if len(self._sends) >= self.depth and self._error is None:
                with TraceAnnotation("PP:RingFull"):
                    self._cv.wait_for(
                        lambda: len(self._sends) < self.depth or self._error is not None
                    )
            self._raise_send_failure_locked()
            self._sends.append((transfer, outgoing, fence))
            self._cv.notify_all()

    def _send(
        self, transfer: int, outgoing: tuple[torch.Tensor, ...], fence: "TpuEvent"
    ) -> None:
        """Sends one transfer once its tensors are computed and returns when
        the copies are done; runs on the send thread."""
        import torch.distributed as dist

        with TraceAnnotation("PP:Send"):
            fence.synchronize()
            works = [
                dist.isend(x, dst=self.next, tag=self._tag(transfer, i))
                for i, x in enumerate(outgoing)
            ]
            for work in works:
                work.wait()

    def _send_loop(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: bool(self._sends) or self._closing)
                if not self._sends:
                    return
                # Stays queued, and counts against the depth, until sent.
                transfer = self._sends[0][0]
            error: BaseException | None = None
            try:
                self._send(*self._sends[0])
            except BaseException as exc:  # re-raised on the worker thread
                logger.exception(
                    "PP ring rank %d: sending transfer %d failed", self.rank, transfer
                )
                error = exc
            with self._cv:
                if error is not None:
                    self._error = error
                    self._sends.clear()
                    self._cv.notify_all()
                    return
                # Releases the tensors: the copies are done.
                self._sends.popleft()
                self.sent += 1
                self._cv.notify_all()

    def push(self) -> None:
        """Does nothing: every forward moves on as soon as it is done, and
        the engine sends no pushes with the ring."""

    def settle(self) -> None:
        """Does nothing: no stage holds launches for the others to match."""

    def close(self) -> None:
        """Waits until every queued send is done, stops the send thread, and
        raises a send failure."""
        thread = self._thread
        if thread is not None:
            with self._cv:
                self._closing = True
                self._cv.notify_all()
            thread.join()
            self._thread = None
            self._closing = False
        self._raise_send_failure()
