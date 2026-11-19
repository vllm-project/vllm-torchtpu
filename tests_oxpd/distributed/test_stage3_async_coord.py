# SPDX-License-Identifier: Apache-2.0
"""q38 L6-S3ASYNC coordinator (distributed/kv_transfer/stage3_async.py): four CPU
gloo ranks.

1. Exactness: for the batched-admission scenarios of
   test_stage3_batched_admission.py (parked and new loads, prefix-aware skips,
   deadlines expired on no rank / one rank / all ranks, rank-specific free maps),
   one coordinator job per pass, run on the coordinator thread over its own gloo
   group, makes on every rank the same decisions, in the same order, with the same
   staging pages and the same leftover staging state as today's batched pass run on
   the model thread's group; and every rank agrees. Meanwhile the "model thread"
   runs its own collectives on the default group (the TP submit-state broadcast's
   group), which must neither hang nor see a wrong value.
2. Agreement: a load whose staging fails on any one rank is dispatched on no rank
   (STAGEFAIL everywhere, no RPC on TP0).
3. Bytes: every dispatched load's simulated landing (source bytes -> its staging
   pages -> the destination) is bitwise identical to the landing through today's
   pass.
4. Concurrency: 60 passes with rank-specific random delays, the model thread
   releasing staging pages while jobs run (each rank at its own pass, as receive
   terminals do), a fixed apply lag, and model-thread collectives every pass: no
   deadlock, identical decisions on every rank, never a page reserved twice, and
   every rank's free map consistent with its live reservations.

The reference pass uses the real ``StagedConsumer.reserve`` (pd_staging); the
coordinator uses ``stage3_async.fast_reserve``, and test_fast_reserve_equals_reserve
proves the two leave every table equal. The admission is the real
``stage3_admission``; the coordinator is the real ``Coordinator`` / ``run_job``.
"""

import os
import random
import tempfile
import threading
import time
import types

import torch
import torch.multiprocessing as mp

from . import test_stage3_batched_admission as tba  # noqa: F401  (stubs vLLM on a bare CPU env)
from vllm_torchtpu.distributed.kv_transfer import stage3_admission as adm  # noqa: E402
from vllm_torchtpu.distributed.kv_transfer import stage3_async as s3a  # noqa: E402

WORLD = tba.WORLD
NOW = tba.NOW
BUDGET = tba.BUDGET
PAGE = 4  # simulated bytes per staging page (int32 words)


class _Host:
    """stage3_async.Host over a real StagedConsumer; records dispatches."""

    def __init__(self, staging, rank, fail=()):
        self.st = staging
        self.rank = rank
        self.fail = set(fail)
        self.dispatched = []
        self.rpcs = []

    def s3a_free_map(self):
        return self.st.free

    def s3a_budget_s(self):
        return BUDGET

    def s3a_now(self):
        return NOW

    def s3a_reserve(self, meta, reduced_free):
        # The coordinator's reservation (the Qwen worker's s3a_reserve); the
        # reference pass (tba._batched) runs the real StagedConsumer.reserve.
        return s3a.fast_reserve(self.st, meta.uuid, meta.num_tokens, skip_tokens=meta.skip_tokens,
                                reduced_free=reduced_free)

    def s3a_pages_needed(self, meta):
        return self.st.pages_needed(meta.uuid, meta.num_tokens, skip_tokens=meta.skip_tokens)

    def s3a_stage(self, job, dest, meta):
        if dest in self.fail:
            return None
        return {"uuid": meta.uuid, "pages": {k: list(v) for k, v in self.st.reservations[meta.uuid].items()}}

    def s3a_dispatch(self, job, dest, staged):
        self.dispatched.append(dest)
        if self.rank == 0:
            self.rpcs.append((dest, staged))


def _land(pages_by_owner, uuid):
    """Simulated landing: the load's source bytes through its staging pages."""
    pages = [p for owner in sorted(pages_by_owner) for p in pages_by_owner[owner]]
    staging = torch.zeros(4096 * PAGE, dtype=torch.int32)
    src = torch.arange(len(pages) * PAGE, dtype=torch.int32) * 7919 + uuid
    for i, p in enumerate(pages):
        staging[p * PAGE:(p + 1) * PAGE] = src[i * PAGE:(i + 1) * PAGE]
    return torch.cat([staging[p * PAGE:(p + 1) * PAGE] for p in pages]) if pages else torch.zeros(0, dtype=torch.int32)


def _mt_collective(group, rank, tag):
    """A model-thread collective on the default group, checked."""
    import torch.distributed as dist

    x = torch.tensor([rank + 1 + tag], dtype=torch.int64)
    dist.all_reduce(x, group=group)
    want = sum(r + 1 + tag for r in range(WORLD))
    return int(x.item()) == want


def _exact(rank, group, tp, seeds, fails):
    import torch.distributed as dist

    stats = {"jobs": 0, "dispatched": 0, "parked": 0, "expired": 0, "stagefail": 0, "mt_ok": 0}
    for seed in seeds:
        free, waits, loads = tba._exact_fit(rank) if seed == -1 else tba._scenario(seed, rank)
        ref, st_r, w_r, _ = tba._batched(free, waits, loads, tp)
        fail_dest = set()
        if seed > 0 and seed % 7 == 0:
            reserved = [d for d, what, _ in ref if what == "reserved"]
            if reserved and rank == seed % WORLD:
                fail_dest = {reserved[0]}
        st_a = tba._staging(free)
        host = _Host(st_a, rank, fail_dest)
        coord = s3a.Coordinator(host, group, f"t-s3a-{rank}-{seed}")
        job = s3a.Job(seq=seed, apply_pass=seed + 1,
                      candidates=[(d, m, waits[d][1] if d in waits else None) for d, m in loads])
        coord.submit(job)
        # The model thread keeps running its own collectives while the job runs.
        if _mt_collective(dist.group.WORLD, rank, seed):
            stats["mt_ok"] += 1
        if not job.done.wait(120):
            fails.append((seed, "job did not finish"))
            return stats
        if job.error is not None:
            fails.append((seed, "job error", repr(job.error)))
            continue
        stats["jobs"] += 1
        any_fail = [None] * WORLD
        dist.all_gather_object(any_fail, sorted(fail_dest))
        failed_somewhere = {d for f in any_fail for d in f}
        got = []
        for dest, kind, payload in job.decisions:
            stats[kind] += 1
            if kind == s3a.DISPATCHED:
                got.append((dest, "reserved", payload["pages"]))
            elif kind == s3a.STAGEFAIL:
                got.append((dest, "reserved", st_a.reservations[next(m.uuid for d, m in loads if d == dest)]))
            else:
                got.append((dest, kind, None))
        if ref != got:
            fails.append((seed, "decisions", ref, got))
        if not torch.equal(st_r.free, st_a.free) or st_r.reservations != st_a.reservations or st_r.skips != st_a.skips:
            fails.append((seed, "staging state"))
        # Agreement: a load staged-failed on any rank is dispatched on none.
        for dest, kind, _ in job.decisions:
            if dest in failed_somewhere and kind != s3a.STAGEFAIL:
                fails.append((seed, "dispatched despite a failed rank", dest))
            if dest not in failed_somewhere and kind == s3a.STAGEFAIL:
                fails.append((seed, "stagefail without a failed rank", dest))
        if rank == 0 and any(d in failed_somewhere for d, _ in host.rpcs):
            fails.append((seed, "RPC for a load a rank failed to stage"))
        # Bytes: the dispatched loads land identically through today's pass.
        for dest, kind, payload in job.decisions:
            if kind == s3a.DISPATCHED:
                uuid = payload["uuid"]
                if not torch.equal(_land(payload["pages"], uuid), _land(st_r.reservations[uuid], uuid)):
                    fails.append((seed, "landed bytes differ", dest))
        every = [None] * WORLD
        dist.all_gather_object(every, [(d, k) for d, k, _ in job.decisions])
        if any(e != every[0] for e in every):
            fails.append((seed, "ranks disagree"))
    return stats


def _concurrent(rank, group, passes, fails, lag):
    """Many passes, apply lag, rank-specific delays and releases during jobs."""
    import torch.distributed as dist

    pages = 160
    free = torch.ones(pages, dtype=torch.int32)
    free[0] = 0
    st = tba._staging(free)
    host = _Host(st, rank)
    coord = s3a.Coordinator(host, group, f"t-s3a-conc-{rank}")
    g = random.Random(4242)  # identical on every rank: the scheduler's loads
    r = random.Random(1000 + rank)  # rank-local timing and release delays
    jobs, applied, live = [], [], {}  # live: uuid -> release pass on this rank
    release_at = {}
    stop = threading.Event()
    stats = {"mt_ok": 0, "decisions": 0, "dispatched": 0, "parked": 0, "max_live": 0}

    def releaser():
        # Releases run on another thread, racing the coordinator's reservations,
        # as get_finished / repack releases race them on the model thread.
        while not stop.is_set():
            time.sleep(r.uniform(0, 0.002))
            for uuid in [u for u, p in list(release_at.items()) if p <= pass_no[0]]:
                st.release(uuid)
                release_at.pop(uuid, None)

    pass_no = [0]
    th = threading.Thread(target=releaser, daemon=True)
    th.start()
    seq = 0
    waits = {}
    for p in range(1, passes + 1):
        pass_no[0] = p
        # The model thread's per-step quarantine count-down (start_load_kv), also
        # racing the coordinator.
        st.advance_quarantine()
        while jobs and jobs[0].apply_pass <= p:
            job = jobs.pop(0)
            if not job.done.wait(120):
                fails.append((p, "join timed out"))
                stop.set()
                return stats
            if job.error is not None:
                fails.append((p, "job error", repr(job.error)))
            applied.append([(d, k) for d, k, _ in job.decisions])
            for d, k, payload in job.decisions:
                stats["decisions"] += 1
                if k == s3a.DISPATCHED:
                    stats["dispatched"] += 1
                    waits.pop(d, None)
                    release_at[payload["uuid"]] = p + 2 + r.randint(0, 3)  # each rank at its own terminal
                elif k == s3a.PARKED:
                    stats["parked"] += 1
                    waits[d] = payload
                else:
                    waits.pop(d, None)
        inflight = {d for j in jobs for d, _, _ in j.candidates}
        cands = [(d, m, waits[d]) for d, m in sorted(parked_meta.items()) if d in waits and d not in inflight] if p > 1 else []
        for i in range(g.choice([0, 0, 1, 1, 2, 3])):
            tokens = g.choice([16, 40, 64, 100, 160, 200, 320])
            dest = f"c{p}_{i}"
            meta = tba.Meta(uuid=p * 100 + i, num_tokens=tokens, skip_tokens=0)
            parked_meta[dest] = meta
            cands.append((dest, meta, None))
        if cands:
            seq += 1
            job = s3a.Job(seq=seq, apply_pass=p + lag, candidates=cands)
            jobs.append(job)
            coord.submit(job)
        if _mt_collective(dist.group.WORLD, rank, p):
            stats["mt_ok"] += 1
        time.sleep(r.uniform(0, 0.004))
        reserved_pages = [q for v in list(st.reservations.values()) for pages_ in v.values() for q in pages_]
        if len(reserved_pages) != len(set(reserved_pages)):
            fails.append((p, "a page reserved twice"))
        if reserved_pages and int(st.free[reserved_pages].sum()) != 0:
            fails.append((p, "a reserved page is marked free"))
        stats["max_live"] = max(stats["max_live"], len(st.reservations))
    for job in jobs:
        if not job.done.wait(120):
            fails.append(("drain", "job did not finish"))
            stop.set()
            return stats
        applied.append([(d, k) for d, k, _ in job.decisions])
    stop.set()
    th.join(5)
    every = [None] * WORLD
    dist.all_gather_object(every, applied)
    if any(e != every[0] for e in every):
        fails.append(("concurrent", "ranks applied different decisions"))
    return stats


parked_meta = {}


def _rank(rank, init_file, seeds, passes, q):
    import torch.distributed as dist

    try:
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
        tp = types.SimpleNamespace(world_size=WORLD, ranks=list(range(WORLD)), rank_in_group=rank,
                                   cpu_group=dist.group.WORLD)
        group, why = s3a.make_group(tp)
        fails = []
        if group is None:
            q.put((rank, [("make_group", why)], {}))
            return
        if group.cpu_group is dist.group.WORLD:
            fails.append(("make_group", "the coordinator group is the default group"))
        stats = _exact(rank, group, tp, seeds, fails)
        if not fails:
            stats.update({"c_" + k: v for k, v in _concurrent(rank, group, passes, fails, lag=2).items()})
        q.put((rank, fails, stats))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as exc:  # noqa: BLE001
        q.put((rank, [("exception", repr(exc))], {}))


def _run(seeds, passes, timeout=None):
    timeout = timeout or int(os.environ.get("Q38_S3A_TIMEOUT", "420"))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, "init")
        procs = [ctx.Process(target=_rank, args=(r, init, seeds, passes, q)) for r in range(WORLD)]
        for p in procs:
            p.start()
        res = []
        deadline = time.time() + timeout
        try:
            for _ in procs:
                res.append(q.get(timeout=max(1, deadline - time.time())))
        except Exception as exc:  # noqa: BLE001 - a hang is a failure, not a pass
            res.append((-1, [("timeout", f"{len(res)} of {WORLD} ranks reported: {exc!r}")], {}))
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
    return sorted(res, key=lambda x: x[0])


SEEDS = [-1] + list(range(1, 61))


def test_coordinator_matches_todays_pass_agrees_and_survives_concurrency():
    res = _run(SEEDS, passes=60)
    assert len(res) == WORLD, res
    for rank, fails, stats in res:
        assert not fails, (rank, fails[:3])
    stats = res[0][2]
    print("Q38S3A_COORD", stats)
    # Every branch ran: dispatched, parked, expired, a staging failure, and the
    # model thread's collectives all returned the right value.
    assert stats["dispatched"] > 40 and stats["parked"] > 15 and stats["expired"] > 8, stats
    assert stats["stagefail"] >= 3, stats
    assert stats["mt_ok"] == len(SEEDS), stats
    assert stats["c_mt_ok"] == 60 and stats["c_dispatched"] > 20, stats


def test_job_without_group_and_stats_line():
    """World of one (no collective) and the S3ASYNC line's fields."""
    free = torch.ones(32, dtype=torch.int32)
    free[0] = 0
    st = tba._staging(free)
    host = _Host(st, 0)
    job = s3a.Job(seq=1, apply_pass=3, candidates=[
        ("a", tba.Meta(uuid=1, num_tokens=48, skip_tokens=0), None),
        ("b", tba.Meta(uuid=2, num_tokens=320, skip_tokens=0), NOW - 1),
    ])
    s3a.run_job(host, job, None)
    assert [(d, k) for d, k, _ in job.decisions] == [("a", s3a.DISPATCHED), ("b", s3a.EXPIRED)]
    assert job.collectives == 0 and host.dispatched == ["a"]
    stats = s3a.Stats()
    stats.add_job(job)
    stats.passes_on, stats.ms_on = 3, [0.1, 0.2, 0.3]
    line = stats.line(5)
    for field in ("S3ASYNC rank5", "passes=3", "on=3", "share_on=1.000", "jobs=1", "loads=2",
                  "dispatched=1", "expired=1", "join_waits=0", "lag="):
        assert field in line, (field, line)


def test_make_group_refuses_without_a_process_group():
    tp = types.SimpleNamespace(world_size=2, ranks=[0, 1], rank_in_group=0)
    import torch.distributed as dist

    if not dist.is_initialized():
        group, why = s3a.make_group(tp)
        assert group is None and "not initialized" in why


def _fr_staging(free, names=("fa0", "gdn0", "fa1")):
    st = tba._staging(free)
    if tuple(names) != ("fa0", "gdn0", "fa1"):
        st.manifest = types.SimpleNamespace(pools=[types.SimpleNamespace(layer_name=n) for n in names])
        st._src_owner = {n: (1 if n.startswith("gdn") else 0) for n in names}
    return st


def test_fast_reserve_equals_reserve():
    """fast_reserve against StagedConsumer.reserve(reduced_free=...): 300 random
    sequences of loads (sizes, prefix skips, draft-zero prefixes, maps with holes,
    loads that do not fit, repeats of a reserved uuid, a manifest repeating an
    owner name): the same return values and the same free map, reduced map,
    reservations, skips, tokens and draft_zero after every call."""
    import copy as _copy

    calls = fits = misses = 0
    for seed in range(300):
        g = random.Random(seed)
        pages = g.choice([16, 40, 96, 300])
        free = torch.ones(pages, dtype=torch.int32)
        free[0] = 0
        for p in g.sample(range(1, pages), g.randint(0, pages // 2)):
            free[p] = 0
        names = ("fa0", "gdn0", "fa1") if seed % 10 else ("fa0", "gdn0", "fa0")
        a, b = _fr_staging(free, names), _fr_staging(free, names)
        rf_a = free.clone() if seed % 3 else (free * (torch.rand(pages, generator=torch.Generator().manual_seed(seed)) > 0.2).to(torch.int32))
        rf_b = rf_a.clone()
        for i in range(g.randint(1, 8)):
            tokens = g.choice([16, 40, 64, 100, 160, 320, 1000])
            skip = g.choice([0, 0, 16, 32]) if tokens > 48 else 0
            dz = g.choice([0, 0, 48])
            uuid = seed * 100 + (i if g.random() > 0.15 else 0)  # sometimes a repeat
            ra = a.reserve(uuid, tokens, skip_tokens=skip, reduced_free=rf_a, draft_zero_until=dz)
            rb = s3a.fast_reserve(b, uuid, tokens, skip_tokens=skip, reduced_free=rf_b, draft_zero_until=dz)
            calls += 1
            fits += ra
            misses += not ra
            assert ra == rb, (seed, i)
            assert torch.equal(a.free, b.free) and torch.equal(rf_a, rf_b), (seed, i)
            assert a.reservations == b.reservations and a.skips == b.skips, (seed, i)
            assert a.__dict__.get("tokens", {}) == b.__dict__.get("tokens", {}), (seed, i)
            assert a.__dict__.get("draft_zero", {}) == b.__dict__.get("draft_zero", {}), (seed, i)
    assert fits > 300 and misses > 100, (calls, fits, misses)
