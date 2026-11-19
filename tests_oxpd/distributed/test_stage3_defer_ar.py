# SPDX-License-Identifier: Apache-2.0
"""q38 L6-S3ASYNC S3-DEFER-AR on four CPU gloo ranks: the early-issued admission
all-reduce (stage3_admission.prepare_async / finish) and the lean deferred pass.

Every rank runs the model thread's program order on ONE CPU group, as served:
issue the admission all-reduce (async) in "start_load_kv", then other CPU-group
collectives of the step (the TP submit-state broadcast's kind), then "the deferred
pass": finish (wait), reserve with stage3_async.fast_reserve, stage, Benliang
Wang's one-int32 all_staged (credited), with rank-specific delays at every point.

On every rank and for the batched-admission scenarios of
test_stage3_batched_admission.py: finish returns exactly what today's synchronous
prepare returns (reduced map, not-expired flags, deadlines); the pass makes the same
decisions with the same pages and leftover staging state as today's batched pass;
a load that failed staging on any rank is dispatched on none; the interleaved
collectives all return the right values; every rank agrees.
The proof's mutants (issue on one rank only, issue out of order on one rank, no
wait, MAX instead of MIN in all_staged) must make this test fail or time out.
"""

import os
import random
import tempfile
import time
import types

import torch
import torch.multiprocessing as mp

from . import test_stage3_batched_admission as tba  # noqa: F401  (stubs vLLM on a bare CPU env)
from vllm_torchtpu.distributed.kv_transfer import stage3_admission as adm  # noqa: E402
from vllm_torchtpu.distributed.kv_transfer import stage3_async as s3a  # noqa: E402

WORLD = tba.WORLD
NOW, BUDGET = tba.NOW, tba.BUDGET


def _mt(group, rank, tag):
    """One of the step's other CPU-group collectives, checked."""
    import torch.distributed as dist

    x = torch.tensor([rank + 1 + tag], dtype=torch.int64)
    dist.all_reduce(x, group=group)
    return int(x.item()) == sum(r + 1 + tag for r in range(WORLD))


def _rank(rank, init_file, seeds, q):
    import torch.distributed as dist

    try:
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
        tp = types.SimpleNamespace(world_size=WORLD, cpu_group=dist.group.WORLD)
        rnd = random.Random(100 + rank)
        fails, stats = [], {"passes": 0, "mt_ok": 0, "dispatched": 0, "parked": 0, "expired": 0, "stagefail": 0,
                            "collectives": 0}
        for seed in seeds:
            free, waits, loads = tba._exact_fit(rank) if seed == -1 else tba._scenario(seed, rank)
            # today's reference on the same group (synchronous prepare + real reserve + per-load ready)
            ref, st_r, w_r, _ = tba._batched(free, waits, loads, tp)
            sync = adm.prepare(loads, tba._staging(free).free, dict(waits), BUDGET, NOW, tp)
            # the served order: issue in start_load_kv ...
            st = tba._staging(free)
            w = dict(waits)
            time.sleep(rnd.uniform(0, 0.004) if rank == 3 else 0)
            issued = adm.prepare_async(loads, st.free, w, BUDGET, NOW, tp)
            # ... the step's other CPU-group collectives (forward dispatch, sampling) ...
            if _mt(dist.group.WORLD, rank, seed):
                stats["mt_ok"] += 1
            time.sleep(rnd.uniform(0, 0.003))
            # ... then the deferred pass
            a = adm.finish(issued)
            if not (torch.equal(a.reduced_free, sync.reduced_free) and a.not_expired == sync.not_expired
                    and a.deadlines == sync.deadlines and a.collectives == sync.collectives):
                fails.append((seed, "finish != prepare"))
            fail_dest = set()
            if seed > 0 and seed % 7 == 0 and rank == seed % WORLD:
                res = [d for d, what, _ in ref if what == "reserved"]
                fail_dest = set(res[:1])
            out = []
            for dest, meta in loads:
                ok = a.reserve(dest, meta,
                               lambda m, rf: s3a.fast_reserve(st, m.uuid, m.num_tokens, skip_tokens=m.skip_tokens,
                                                              reduced_free=rf),
                               lambda m: st.pages_needed(m.uuid, m.num_tokens, skip_tokens=m.skip_tokens))
                if ok:
                    w.pop(dest, None)
                    staged = dest not in fail_dest
                    every = adm.all_staged(staged, tp, True)
                    stats["collectives"] += 1
                    out.append((dest, "reserved", st.reservations[meta.uuid]))
                    stats["dispatched" if every else "stagefail"] += 1
                    if every and not staged:
                        fails.append((seed, "dispatched although this rank failed staging", dest))
                elif not a.expired(dest, meta, w):
                    out.append((dest, "parked", None))
                    stats["parked"] += 1
                else:
                    out.append((dest, "expired", None))
                    stats["expired"] += 1
            stats["passes"] += 1
            if out != ref:
                fails.append((seed, "decisions", ref, out))
            if not torch.equal(st.free, st_r.free) or st.reservations != st_r.reservations or st.skips != st_r.skips:
                fails.append((seed, "staging state"))
            every = [None] * WORLD
            dist.all_gather_object(every, (out, stats["dispatched"], stats["stagefail"]))
            if any(e != every[0] for e in every):
                fails.append((seed, "ranks disagree"))
        q.put((rank, fails, stats))
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as exc:  # noqa: BLE001
        q.put((rank, [("exception", repr(exc)[:300])], {}))


def _run(seeds):
    timeout = int(os.environ.get("Q38_S3A_TIMEOUT", "420"))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, "init")
        procs = [ctx.Process(target=_rank, args=(r, init, seeds, q)) for r in range(WORLD)]
        for p in procs:
            p.start()
        res, deadline = [], time.time() + timeout
        try:
            for _ in procs:
                res.append(q.get(timeout=max(1, deadline - time.time())))
        except Exception as exc:  # noqa: BLE001 - a hang is a failure
            res.append((-1, [("timeout", f"{len(res)} of {WORLD} ranks reported: {exc!r}")], {}))
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
    return sorted(res, key=lambda x: x[0])


def test_issued_admission_equals_todays_pass_on_every_rank():
    res = _run([-1] + list(range(1, 61)))
    assert len(res) == WORLD, res
    for rank, fails, stats in res:
        assert not fails, (rank, fails[:3])
    stats = res[0][2]
    print("Q38S3D_PROTOCOL", stats)
    assert stats["passes"] == 61 and stats["mt_ok"] == 61, stats
    assert stats["dispatched"] > 40 and stats["parked"] > 15 and stats["expired"] > 8 and stats["stagefail"] >= 3, stats


def test_all_staged_without_a_group_is_local():
    """World of one: no collective (prepare_async / finish behave like prepare)."""
    free = torch.ones(16, dtype=torch.int32)
    loads = [("x", tba.Meta(uuid=1, num_tokens=16, skip_tokens=0))]
    a = adm.finish(adm.prepare_async(loads, free, {}, BUDGET, NOW, None))
    b = adm.prepare(loads, free, {}, BUDGET, NOW, None)
    assert a.collectives == b.collectives == 0 and a.not_expired == b.not_expired
    assert torch.equal(a.reduced_free, b.reduced_free)
