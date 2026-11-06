# SPDX-License-Identifier: Apache-2.0
"""Batched Stage-3 destination admission (QWEN4_EXP_PD_BATCHED_ADMISSION).

A `_submit_stage3_loads` pass admits loads in order: parked loads (oldest first),
then the step's new ones. Today every load that reaches admission runs its own
CPU collectives over the TP group: the destination's ``reserve`` all-reduces the
staging free map (MIN: allocate only pages free on every rank), and a load that
does not fit all-reduces its expiry flag (MAX: any rank expired expires it on all).
N parked loads cost 2N gloo all-reduces per decode step, on the critical path
before the forward is dispatched (FR1b j-e6ef4291 c48: 8 parked loads, 16
all-reduces, ~14 ms per step on rank 0).

The batched pass runs ONE collective per pass: a MIN all-reduce of
``concat(free map, 1 - expired flag of every candidate load)``. Then each load is
decided locally, in the same order, by the same rules:

- **Reservation:** allocate from the reduced map, zeroing the allocated pages in
  it as well as locally. Reservations are identical on every rank and no staging
  page is released during the pass (releases happen in ``repack_landed`` before
  the pass and in ``get_finished`` after it), so the reduced map at each load
  equals what today's per-load MIN all-reduce returns there.
- **Expiry:** MIN of (1 - expired) is 1 - MAX(expired), the same any-rank rule.
  The deadline is the parked load's, or ``now + budget`` for a load parking now,
  with ``now`` read once per pass instead of once per load (microseconds apart).

S2: no collective when no load reaches admission (as today), and no per-load
allocation work for a load that needs more pages than the reduced map still has.

Framework-free (torch + torch.distributed) so a multi-process CPU gloo test can
drive the real decision code (tests/distributed/test_stage3_batched_admission.py).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def _world(group: Any) -> int:
    return int(getattr(group, "world_size", 1) or 1) if group is not None else 1


class BatchedAdmission:
    """One pass's admission decisions, made locally after the single collective."""

    def __init__(
        self,
        reduced_free: torch.Tensor | None,
        not_expired: dict[str, bool],
        deadlines: dict[str, float],
        collectives: int,
    ) -> None:
        self.reduced_free = reduced_free
        self.not_expired = not_expired
        self.deadlines = deadlines
        self.collectives = collectives
        self.available = int(torch.count_nonzero(reduced_free)) if reduced_free is not None else 0
        self.decisions: list[tuple[str, str]] = []

    def __contains__(self, dest: str) -> bool:
        return dest in self.not_expired

    def reserve(
        self,
        dest: str,
        meta: Any,
        reserve_fn: Callable[[Any, torch.Tensor | None], bool],
        pages_needed_fn: Callable[[Any], int] | None = None,
    ) -> bool:
        """``reserve_fn(meta, reduced_free)`` allocates from (and zeroes in) the
        reduced map; without a free map (no model-owned destination) it decides
        alone, as the per-load path does. ``pages_needed_fn`` (S2) skips the
        allocation work when the load cannot fit."""
        if self.reduced_free is not None and pages_needed_fn is not None:
            if pages_needed_fn(meta) > self.available:
                self.decisions.append((dest, "no_room"))
                return False
        ok = bool(reserve_fn(meta, self.reduced_free))
        if self.reduced_free is not None and ok:
            self.available = int(torch.count_nonzero(self.reduced_free))
        self.decisions.append((dest, "reserved" if ok else "no_room"))
        return ok

    def expired(
        self,
        dest: str,
        meta: Any,
        waits: dict[str, tuple[Any, float]],
        on_expired: Callable[[str], None] | None = None,
    ) -> bool:
        """Today's ``_stage3_destination_wait_expired`` with the pass's reduced flag:
        records the parked load's deadline, and drops it from ``waits`` (and calls
        ``on_expired``) when any rank's deadline has passed."""
        waits[dest] = (meta, self.deadlines[dest])
        expired = not self.not_expired[dest]
        if expired:
            waits.pop(dest, None)
            if on_expired is not None:
                on_expired(dest)
        self.decisions.append((dest, "expired" if expired else "parked"))
        return expired


def prepare(
    candidates: Sequence[tuple[str, Any]],
    local_free: torch.Tensor | None,
    waits: dict[str, tuple[Any, float]],
    budget_s: float,
    now: float,
    group: Any,
) -> BatchedAdmission:
    """The pass's single collective: MIN all-reduce of the free map and of every
    candidate's not-expired flag over ``group.cpu_group`` (no collective with no
    group, a group of one, or nothing to decide)."""
    deadlines: dict[str, float] = {}
    flags = []
    for dest, _ in candidates:
        entry = waits.get(dest)
        deadline = entry[1] if entry is not None else now + budget_s
        deadlines[dest] = deadline
        flags.append(0 if now >= deadline else 1)
    parts = []
    if local_free is not None:
        parts.append(local_free.to(torch.int32).clone())
    parts.append(torch.tensor(flags, dtype=torch.int32))
    payload = torch.cat(parts) if len(parts) > 1 else parts[0]
    collectives = 0
    if candidates and _world(group) > 1 and payload.numel():
        import torch.distributed as dist

        dist.all_reduce(payload, op=dist.ReduceOp.MIN, group=group.cpu_group)
        collectives = 1
    n_free = 0 if local_free is None else int(local_free.numel())
    reduced_free = payload[:n_free].to(local_free.dtype) if local_free is not None else None
    reduced_flags = payload[n_free:].tolist()
    not_expired = {dest: bool(f) for (dest, _), f in zip(candidates, reduced_flags)}
    return BatchedAdmission(reduced_free, not_expired, deadlines, collectives)


def ab_mode(passes: int, ab_passes: int, flag: bool) -> bool:
    """The switch for pass number ``passes`` (1-based, counted identically on every
    rank): ``ab_passes > 0`` alternates off / on every ``ab_passes`` passes, off
    first, so all ranks flip on the same pass; otherwise ``flag``."""
    if ab_passes > 0:
        return ((passes - 1) // ab_passes) % 2 == 1
    return bool(flag)
