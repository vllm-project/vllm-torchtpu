# SPDX-License-Identifier: Apache-2.0
"""Event-gated, parked Stage-3 send registration on a pipeline producer (q38
PSTEP-L2; ``VLLM_TPU_STAGE3_PARK_SEND_REGISTRATION``, default off).

Today a pipeline producer's ``process_send_load`` runs, in the step after a
request finishes, ``synchronize_tensors(kv, wait=True)`` and then registers the
step's sends with the Raiden controller. The sync materializes the deferred KV
writes (including the drafter's) and blocks the host until the device has run
every program enqueued so far, so the device idles while the host then
registers and prepares the next forward.

With the key on, three things change, and nothing else:

1. The sync enqueues the same materialization without waiting
   (``synchronize_tensors(kv, wait=False)``: graph traversal, compile and
   enqueue still happen on the host; only the wait for the device is dropped),
   and a ``torch.tpu.Event`` is recorded behind it. The event covers every
   program enqueued before it, which includes every KV write of the finished
   requests (they are not scheduled again, and their pages stay allocated until
   the send finishes: vLLM's delay-free blocks).
2. The step's registration (its ``reqs_to_send`` and ``reqs_to_release``) is
   parked with that event instead of running.
3. The next ``process_send_load`` call on the worker, which every
   ``execute_model`` makes (a forward step before its forward, a no-forward
   step through ``kv_connector_no_forward``), first registers every parked
   step in order. It calls ``synchronize()`` on the step's event before the
   registration: when ``query()`` already reported completion this returns at
   once (and surfaces any deferred device error, as
   ``offload/raiden_connector._event_is_retired`` does); otherwise the host
   waits there, a bounded wait for work enqueued before the previous step's
   forward. A finish the scheduler announced while the step was parked
   (``finished_req_ids``) is applied right after its registration, as
   ``get_finished`` applies it to a registration made ahead of the finish.

Raiden cannot read a source before its registration, and a registration runs
only after the event (so every write enqueued up to the materialization point)
has completed: the invariant the blocking sync gives today. Only the moment the
host learns of completion moves, by at most one ``execute_model`` call.

Host-side only: no traced program reads the key.
"""

from __future__ import annotations

import collections
import dataclasses
import time
from collections.abc import Callable, Iterable
from typing import Any

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ENV = "VLLM_TPU_STAGE3_PARK_SEND_REGISTRATION"

# Engagement (the reader's markers): one line at a worker's first park, then a
# counter line at its first drain and every REPORT_EVERY drains after it.
REPORT_EVERY = 64


def enabled() -> bool:
    return bool(envs.VLLM_TPU_STAGE3_PARK_SEND_REGISTRATION)


@dataclasses.dataclass
class ParkedStep:
    """One step's parked registration and the event that gates it."""

    metadata: Any
    event: Any
    req_ids: frozenset[str]
    parked_at: float


class Stage3SendParking:
    """FIFO of parked send registrations on one producer worker."""

    def __init__(self, transfer_rank: int) -> None:
        self.transfer_rank = int(transfer_rank)
        self._parked: collections.deque[ParkedStep] = collections.deque()
        self._finished: set[str] = set()
        self.counts = {
            "parks": 0,
            "sends": 0,
            "drains": 0,
            "ready": 0,
            "waited": 0,
            "finishes": 0,
        }
        self.max_parked_s = 0.0

    def __len__(self) -> int:
        return len(self._parked)

    def parked_req_ids(self) -> set[str]:
        out: set[str] = set()
        for step in self._parked:
            out |= step.req_ids
        return out

    def park(self, metadata: Any, event: Any) -> None:
        """Parks one step's registration behind ``event`` (recorded after the
        step's KV materialization was enqueued)."""
        req_ids = frozenset(str(r) for r in metadata.reqs_to_send)
        self._parked.append(ParkedStep(metadata, event, req_ids, time.monotonic()))
        self.counts["parks"] += 1
        self.counts["sends"] += len(req_ids)
        if self.counts["parks"] == 1:
            logger.info(
                "q38 PSTEP-L2 stage-3 send park ENGAGED: transfer rank %d, sends %d (VLLM_TPU_STAGE3_PARK_SEND_REGISTRATION)",  # noqa: E501
                self.transfer_rank,
                len(req_ids),
            )

    def note_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Remembers the scheduler's finish of a request whose registration is
        still parked, to apply it once the registration has run."""
        if not self._parked:
            return
        parked = self.parked_req_ids()
        for req_id in finished_req_ids:
            if str(req_id) in parked:
                self._finished.add(str(req_id))

    def drain(
        self,
        register: Callable[[Any], None],
        arm_finished: Callable[[str], None],
    ) -> int:
        """Registers every parked step, oldest first, each only after its
        event has completed (waiting on it if ``query()`` says it has not)."""
        drained = 0
        while self._parked:
            step = self._parked[0]
            ready = bool(step.event.query())
            # Returns at once when ready; otherwise the bounded wait.
            step.event.synchronize()
            self._parked.popleft()
            register(step.metadata)
            for req_id in sorted(step.req_ids & self._finished):
                arm_finished(req_id)
                self._finished.discard(req_id)
                self.counts["finishes"] += 1
            self.counts["ready" if ready else "waited"] += 1
            self.max_parked_s = max(
                self.max_parked_s, time.monotonic() - step.parked_at
            )
            drained += 1
        if drained:
            self.counts["drains"] += 1
            n = self.counts["drains"]
            if n == 1 or n % REPORT_EVERY == 0:
                c = self.counts
                logger.info(
                    "q38 PSTEP-L2 stage-3 send park: transfer rank %d, parks %d, sends %d, drained ready %d, waited %d, finishes %d, max parked %.1f ms",  # noqa: E501
                    self.transfer_rank,
                    c["parks"],
                    c["sends"],
                    c["ready"],
                    c["waited"],
                    c["finishes"],
                    self.max_parked_s * 1e3,
                )
        return drained
