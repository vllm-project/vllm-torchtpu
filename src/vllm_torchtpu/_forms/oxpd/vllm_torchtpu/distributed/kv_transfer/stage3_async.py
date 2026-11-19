# SPDX-License-Identifier: Apache-2.0
"""q38 L6-S3ASYNC: Stage-3 load coordination off the model-runner thread.

Lever ``s3async`` (``VLLM_TPU_S3_ASYNC_SUBMIT``: 0 / 1 / ab:<classes>, read once per
submit pass through core/l6_ab.py). Default off: today's path, unchanged.

The problem (S3-SUBMIT, decider j-7968ae01). On a sharded Stage-3 consumer every
arrival runs, on the model-runner thread of every TP rank and before the step's
forward is dispatched: the batched admission's MIN all-reduce of the staging free
map, the staging reservation, the transfer staging and an ``all_gather_object``
ready agreement. In v7 that pass costs 9.75 ms per submit on all 8 ranks (the
``Stage-3 admission A/B stats`` pass_ms regressed on the TP0 submit count), and
the device idles for it.

What this module does. The model-runner thread keeps every scheduler-visible
decision that needs no collective (release-only, fail-only, request-id binding,
duplicate replays) and records the load as submitted, exactly as today's loop does
before admission. The loads that reach admission go, as one *job* per pass, to a
per-rank coordinator thread, which runs today's decision code in today's order on
its own gloo group:

1. ``stage3_admission.prepare``: one MIN all-reduce of (free map, not-expired flags);
2. per load, in order: the reservation against the reduced map, or the park /
   expiry decision. The reservation is ``fast_reserve``: the same pages, tables and
   map updates as ``StagedConsumer.reserve(reduced_free=...)``, without its
   Python list of every free page and its two list-indexed writes per owner;
3. per reserved load: the transfer staging (the real ``_stage_stage3_submit_task``),
   then ONE MIN all-reduce of the job's ready flags (all-rank agreement, where
   today's pass runs an ``all_gather_object`` per load); only for a load every
   rank staged does every rank mark it in flight (its native terminals park) and
   TP0 enqueue the controller RPC on its existing submit worker.

Why the two cheaper forms (CPU microbench mb2, j-cce428c2): run as-is on a
coordinator thread, the pass's Python work (the 65,536-entry ``tolist`` and the
per-owner writes, the object all-gather's pickling) competes with the model
thread for the GIL, which at v7's 200 us switch interval cost the model thread
about as much as it saved. Decisions are unchanged: ``fast_reserve`` is proven
equal to ``reserve`` (tests), and MIN over 0/1 flags is "every rank ready".

The decisions come from collectives, so they are identical on every rank. Every
rank applies job j's decisions to its tables at the same submit pass,
``pass(j) + LAG`` (it blocks there only if its coordinator has not finished job j;
``JOIN_TIMEOUT_S`` turns a stuck job into an error instead of a hang). TP0 holds
the RPC outcome in its mailbox until that pass, so every collective the model
thread runs (the TP submit-state broadcast in ``get_finished``) keeps a membership
that is identical on every rank at every pass. A pass whose lever state is off
first applies every outstanding job, then runs today's path, so the two paths never
reserve concurrently.

What stays exactly as today. The transfer call (source and destination units,
pages, tags, counts, clips), so the KV bytes are bitwise identical; completion
(native terminals on each rank, the scheduler's all-rank aggregation, the sync
repack named only after every rank landed), so no request decodes before its KV is
resident; the abort paths (an abort while a load is in a job resolves through the
same pending-cleanup and pre-RPC checks as an abort while its RPC is queued).

Collectives (methodology rule 8). Only gloo CPU collectives, only on the
coordinator's own group (``torch.distributed.new_group(backend="gloo")`` over the
TP ranks, created once on the model thread at the first on pass, every rank at the
same pass). No device collective anywhere.

Framework-free (torch + torch.distributed + the admission module), so a
multi-process CPU gloo test drives the real decision code
(tests/distributed/test_stage3_async_coord.py).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Protocol

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.kv_transfer import stage3_admission as adm

LEVER = "s3async"

# Read once (envs reads are not free); tests set these module attributes.
SUBMARKS = bool(envs.VLLM_TPU_S3_SUBMARKS)
LAG = max(1, int(envs.VLLM_TPU_S3_ASYNC_LAG))
JOIN_TIMEOUT_S = float(envs.VLLM_TPU_S3_ASYNC_JOIN_TIMEOUT_S)
STATS_EVERY = int(envs.VLLM_TPU_S3_ASYNC_STATS_EVERY)

# Decision kinds a job hands back, per load, in candidate order.
DISPATCHED = "dispatched"  # reserved, staged on every rank, in flight (TP0 RPC enqueued)
PARKED = "parked"  # no destination room yet; payload = the parked deadline
EXPIRED = "expired"  # no room within the wait budget on some rank: the load fails
STAGEFAIL = "stagefail"  # reserved, but some rank's local staging failed: the load fails
KINDS = (DISPATCHED, PARKED, EXPIRED, STAGEFAIL)


# Resolved on first use (core/l6_ab.py reads the flag at import); tests set it.
CONFIGURED: bool | None = None


def configured() -> bool:
    """Whether the lever's flag is set (1 or ab:<classes>)."""
    global CONFIGURED
    if CONFIGURED is None:
        from vllm_torchtpu.core import l6_ab

        CONFIGURED = bool(l6_ab.configured(LEVER))
    return CONFIGURED


def lever_on() -> bool:
    """The lever's state for the current step (identical on every rank)."""
    from vllm_torchtpu.core import l6_ab

    return l6_ab.on(LEVER)


# S3-DEFER-AR (lever s3defer): the single-threaded alternative to s3async; see tpu_connector s3d_*.
DEFER_LEVER = "s3defer"
DEFER_CONFIGURED: bool | None = None


def defer_configured() -> bool:
    global DEFER_CONFIGURED
    if DEFER_CONFIGURED is None:
        from vllm_torchtpu.core import l6_ab

        DEFER_CONFIGURED = bool(l6_ab.configured(DEFER_LEVER))
    return DEFER_CONFIGURED


def defer_on() -> bool:
    """s3defer's state for the current step (identical on every rank)."""
    from vllm_torchtpu.core import l6_ab

    return l6_ab.on(DEFER_LEVER)


def note(key: str, n: int = 1) -> None:
    from vllm_torchtpu.core import l6_ab

    l6_ab.note(LEVER, key, n)


def note_lever(lever: str, key: str, n: int = 1) -> None:
    from vllm_torchtpu.core import l6_ab

    l6_ab.note(lever, key, n)


class Host(Protocol):
    """What a job needs from the connector worker (thread-safe for these calls)."""

    def s3a_free_map(self) -> torch.Tensor | None: ...

    def s3a_budget_s(self) -> float: ...

    def s3a_now(self) -> float: ...

    def s3a_reserve(self, meta: Any, reduced_free: torch.Tensor | None) -> bool: ...

    def s3a_pages_needed(self, meta: Any) -> int: ...

    def s3a_stage(self, job: "Job", dest: str, meta: Any) -> Any | None: ...

    def s3a_dispatch(self, job: "Job", dest: str, staged: Any) -> None: ...


@dataclass
class Job:
    """One pass's loads that reach admission, and what the coordinator decided."""

    seq: int
    apply_pass: int
    # (destination req id, load metadata, parked deadline or None), in today's order:
    # parked loads (oldest first), then the pass's new ones.
    candidates: list[tuple[str, Any, float | None]]
    dst_controller_address: str = ""
    decisions: list[tuple[str, str, Any]] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    collectives: int = 0
    ms: dict[str, float] = field(default_factory=dict)
    # Wall clock (time.time) of the hand-off and of the job's start / end on the
    # coordinator, for joining with device steps (the S3ASYNC_JOB line).
    t_handoff: float = 0.0
    t_begin: float = 0.0
    t_end: float = 0.0


def make_group(tp_group: Any) -> tuple[Any | None, str]:
    """The coordinator's own gloo group over the TP ranks, as ``(group, "")``, or
    ``(None, why)``. Collective over the default group: every rank calls it at the
    same submit pass. Only when the TP group is the whole world, contiguous from 0
    (the decode engines this lever serves); otherwise the lever stays off."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None, "torch.distributed is not initialized"
    world = dist.get_world_size()
    tp_world = int(getattr(tp_group, "world_size", 1) or 1)
    ranks = list(getattr(tp_group, "ranks", range(tp_world)))
    if tp_world != world or sorted(ranks) != list(range(world)):
        return None, f"TP group {ranks} is not the whole world of {world} ranks"
    rank_in_group = getattr(tp_group, "rank_in_group", dist.get_rank())
    if rank_in_group != dist.get_rank():
        return None, f"TP rank {rank_in_group} != global rank {dist.get_rank()}"
    pg = dist.new_group(ranks=list(range(world)), backend="gloo")
    return SimpleNamespace(world_size=world, cpu_group=pg), ""


def fast_reserve(
    staging: Any,
    uuid: int,
    num_tokens: int,
    *,
    skip_tokens: int = 0,
    reduced_free: torch.Tensor,
    draft_zero_until: int = 0,
) -> bool:
    """``StagedConsumer.reserve(uuid, num_tokens, skip_tokens=..., reduced_free=...,
    draft_zero_until=...)`` with the same result in every table: the same pages per
    owner (the first free pages of the reduced map, in manifest order), the same
    zeroes in the local and the reduced map, the same reservations / skips / tokens
    / draft_zero entries, and False with nothing changed when the pages do not fit.
    Cost is in the pages taken, not in the map size (one ``nonzero`` in C, a
    ``tolist`` of the taken pages, two tensor-indexed writes)."""
    if uuid in staging.reservations:
        return True
    first, owners, counts = staging._owner_counts(num_tokens, skip_tokens)
    total = sum(counts)
    available = torch.nonzero(reduced_free, as_tuple=False).flatten()
    if int(available.numel()) < total:
        return False
    taken = available[:total].tolist()
    pages: dict[str, list[int]] = {}
    cursor = 0
    for name, count in zip(owners, counts):
        pages[name] = taken[cursor : cursor + count]
        cursor += count
    # What reserve zeroes: the pages kept in the dict (an owner name repeated in the
    # manifest keeps only its last slice there, as in reserve).
    held = [p for owner_pages in pages.values() for p in owner_pages]
    if held:
        index = torch.tensor(held, dtype=torch.long)
        staging.free[index] = 0
        reduced_free[index] = 0
    staging.reservations[uuid] = pages
    staging.skips[uuid] = (skip_tokens, first)
    staging.__dict__.setdefault("tokens", {})[uuid] = int(num_tokens)
    if draft_zero_until > 0:
        staging.__dict__.setdefault("draft_zero", {})[uuid] = int(draft_zero_until)
    return True


def run_job(host: Host, job: Job, group: Any) -> None:
    """Today's decisions for one pass's admission candidates, on ``group``.

    Same order and same rules as ``_submit_stage3_loads``'s batched pass: one
    admission collective, then per load reserve / park / expire, then the staging
    of every reserved load and one ready agreement over all of them. Fills
    ``job.decisions`` in candidate order."""
    perf = time.perf_counter
    ms = job.ms
    for key in ("adm", "res", "stage", "ag", "disp"):
        ms[key] = 0.0
    t_start = perf()
    job.t_begin = time.time()
    world = int(getattr(group, "world_size", 1) or 1) if group is not None else 1
    loads = [(dest, meta) for dest, meta, _ in job.candidates]
    waits = {
        dest: (meta, deadline)
        for dest, meta, deadline in job.candidates
        if deadline is not None
    }
    t0 = perf()
    admission = adm.prepare(
        loads, host.s3a_free_map(), waits, host.s3a_budget_s(), host.s3a_now(), group
    )
    ms["adm"] += (perf() - t0) * 1e3
    job.collectives += admission.collectives
    outcome: dict[str, tuple[str, Any]] = {}
    reserved: list[tuple[str, Any]] = []
    for dest, meta, _ in job.candidates:
        t0 = perf()
        ok = admission.reserve(dest, meta, host.s3a_reserve, host.s3a_pages_needed)
        ms["res"] += (perf() - t0) * 1e3
        if ok:
            reserved.append((dest, meta))
            continue
        deadline = admission.deadlines[dest]
        expired = admission.expired(dest, meta, waits)
        outcome[dest] = (EXPIRED if expired else PARKED, deadline)
    staged: dict[str, Any] = {}
    for dest, meta in reserved:
        # Let the model-runner thread take the GIL between loads.
        time.sleep(0)
        t0 = perf()
        staged[dest] = host.s3a_stage(job, dest, meta)
        ms["stage"] += (perf() - t0) * 1e3
    flags = [1 if staged[dest] is not None else 0 for dest, _ in reserved]
    if reserved and world > 1:
        t0 = perf()
        agreed = torch.tensor(flags, dtype=torch.int32)
        torch.distributed.all_reduce(
            agreed, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
        )
        job.collectives += 1
        ms["ag"] += (perf() - t0) * 1e3
        flags = agreed.tolist()
    for (dest, _), ok in zip(reserved, flags):
        if not ok:
            outcome[dest] = (STAGEFAIL, None)
            continue
        t0 = perf()
        host.s3a_dispatch(job, dest, staged[dest])
        ms["disp"] += (perf() - t0) * 1e3
        outcome[dest] = (DISPATCHED, staged[dest])
    job.decisions = [(dest, *outcome[dest]) for dest, _, _ in job.candidates]
    ms["total"] = (perf() - t_start) * 1e3
    job.t_end = time.time()


class Coordinator:
    """One daemon thread per rank running jobs in hand-off order."""

    def __init__(self, host: Host, group: Any, name: str) -> None:
        self.host = host
        self.group = group
        self._queue: queue.Queue[Job] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self.thread.start()

    def submit(self, job: Job) -> None:
        self._queue.put(job)

    def _loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                run_job(self.host, job, self.group)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the model thread
                job.error = exc
            finally:
                job.done.set()


@dataclass
class Stats:
    """Per-rank counters for the S3ASYNC engagement line (model-runner thread only)."""

    passes_on: int = 0
    passes_off: int = 0
    ms_on: list[float] = field(default_factory=list)
    ms_off: list[float] = field(default_factory=list)
    jobs: int = 0
    loads: int = 0
    kinds: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KINDS, 0))
    join_waits: int = 0
    join_ms: float = 0.0
    drains: int = 0
    job_ms: dict[str, float] = field(default_factory=dict)
    job_total: list[float] = field(default_factory=list)
    collectives: int = 0

    def add_job(self, job: Job) -> None:
        self.jobs += 1
        self.loads += len(job.candidates)
        self.collectives += job.collectives
        for _, kind, _ in job.decisions:
            self.kinds[kind] += 1
        for key, value in job.ms.items():
            if key == "total":
                self.job_total.append(value)
            else:
                self.job_ms[key] = self.job_ms.get(key, 0.0) + value

    def due(self) -> bool:
        return STATS_EVERY > 0 and self.passes_on + self.passes_off >= STATS_EVERY

    def line(self, rank: int) -> str:
        def q(values: list[float]) -> str:
            if not values:
                return "-"
            v = sorted(values)
            n = len(v)
            return "%.3f/%.3f/%.3f" % (sum(v) / n, v[n // 2], v[min(n - 1, int(0.9 * n))])

        n = self.passes_on + self.passes_off
        jt = sorted(self.job_total)
        parts = " ".join(
            "%s=%.3f" % (key, self.job_ms.get(key, 0.0) / max(1, self.jobs))
            for key in ("adm", "res", "stage", "ag", "disp")
        )
        return (
            "S3ASYNC rank%d passes=%d on=%d share_on=%.3f pass_ms_on(mean/p50/p90)=%s "
            "pass_ms_off(mean/p50/p90)=%s jobs=%d loads=%d dispatched=%d parked=%d "
            "expired=%d stagefail=%d collectives=%d join_waits=%d join_ms=%.3f drains=%d "
            "job_ms(mean/p90)=%s/%s job_parts_ms(mean per job) %s lag=%d"
            % (
                rank,
                n,
                self.passes_on,
                self.passes_on / max(1, n),
                q(self.ms_on),
                q(self.ms_off),
                self.jobs,
                self.loads,
                self.kinds[DISPATCHED],
                self.kinds[PARKED],
                self.kinds[EXPIRED],
                self.kinds[STAGEFAIL],
                self.collectives,
                self.join_waits,
                self.join_ms,
                self.drains,
                "%.3f" % (sum(jt) / len(jt)) if jt else "-",
                "%.3f" % jt[min(len(jt) - 1, int(0.9 * len(jt)))] if jt else "-",
                parts,
                LAG,
            )
        )

    def reset(self) -> None:
        fresh = Stats()
        self.__dict__.update(fresh.__dict__)
