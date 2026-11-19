#!/usr/bin/env python3
"""q38 L6-SYNC: CPU microbench of the cyclic-GC pause per collection, on a heap shaped like the decode
worker's, with the real SYNC-GC lever (core/l6_ab.py + core/l6_jitter.py) driving the sync arms.

CDK CPU pod only (run by tests/q38_l6/run_cpu_sync.sh). The heap is synthetic: its shape follows
reports/host_levers_h1.md T-5 (the previous step's DraftChunkInputs, about 74 walk outputs and 59
metadata objects released each step, C tpu_runner :6101) and the worker code; it is not measured on
the worker, so mapping these pauses onto served steps is inference (I).

Heap:
* static: --static objects (dicts, lists, small instances), then gc.collect + gc.freeze, as vLLM's
  freeze_gc_heap() does at the end of warm-up (both arms);
* per step: --meta metadata-like instances (14 attributes incl. a list, a tuple and a dict), --walk
  walk outputs (a tuple of two lists), --tensors small tensor stand-ins, one DraftChunkInputs-like holder,
  and a SchedulerOutput-like dict over the running requests. The previous step's set is dropped at the
  start of the next step. A fraction --cyc of the metadata objects hold a back-reference to their holder
  (a reference cycle), so only the cyclic GC frees them;
* requests: --req running requests, each a small object graph, replaced in a stream so each lives
  --life steps.

Arms (--steps worker steps each after --warm):
* base: automatic GC, D1e served thresholds (--gen0, 10, 10);
* sync N (each of --every): VLLM_TPU_L6_SYNC_GC=1 through l6_ab.begin_step / l6_jitter.on_step /
  l6_jitter.sync_point (the propose point, placed mid-step), with SYNC_EVERY = N and the valve.

Every collection is timed by a gc.callbacks timer. A sync arm must show 0 automatic collections and
every lockstep collection on a step index that is a multiple of N (valve collections are reported
separately); otherwise the last line is BENCH_SYNCGC_DONE rc=1.
"""
import argparse
import gc
import os
import sys
import time


class Meta:
    pass


class Holder:
    pass


class Req:
    pass


def pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def build_static(n):
    keep = []
    for i in range(n // 4):
        d = {"k": i, "v": [i, i + 1]}
        o = Meta()
        o.a, o.b = i, d
        keep.append((d, o, [o]))
    return keep


def new_req(i):
    r = Req()
    r.rid = f"req-{i}"
    r.tokens = list(range(64))
    r.blocks = [list(range(8)) for _ in range(4)]
    r.params = {"temperature": 0.0, "top_p": 1.0, "stop": [1, 2]}
    return r


def step_objects(args, j, reqs, cyc_every):
    h = Holder()
    metas = []
    for m in range(args.meta):
        o = Meta()
        o.num_reqs, o.max_len, o.pos = len(reqs), 4096, j
        o.block_tables = list(range(16))
        o.seq_lens = tuple(range(8))
        o.extra = {"layer": m, "kind": "full"}
        o.q, o.k, o.v, o.a1, o.a2, o.a3, o.a4 = 1, 2, 3, 4, 5, 6, 7
        if cyc_every and m % cyc_every == 0:
            o.parent = h  # cycle: holder -> metas -> o -> holder
        metas.append(o)
    walk = [([j, w], [w, w + 1, w + 2]) for w in range(args.walk)]
    tensors = [Meta() for _ in range(args.tensors)]
    h.metas, h.walk, h.tensors = metas, walk, tensors
    h.sched = {r.rid: 1 + (j % 3) for r in reqs}
    return h


def run_arm(args, sync_every, cyc_every, l6_ab, l6_jitter):
    # the arm's own stream of requests
    reqs = [new_req(i) for i in range(args.req)]
    nxt = args.req
    pauses = []  # (gen, us, ours)
    ours = {"on": False}
    t0 = {"t": 0}

    def timer(phase, info):
        if phase == "start":
            t0["t"] = time.perf_counter_ns()
        else:
            pauses.append((info.get("generation", 0), (time.perf_counter_ns() - t0["t"]) / 1000, ours["on"]))

    collects = []  # (j, gen, kind)
    real_collect = gc.collect

    def lever_collect(gen):
        ours["on"] = True
        try:
            return real_collect(gen)
        finally:
            ours["on"] = False

    for k in [k for k in os.environ if k.startswith("VLLM_TPU_L6_")]:
        del os.environ[k]
    l6_jitter.restore()
    if sync_every:
        os.environ["VLLM_TPU_L6_SYNC_GC"] = "1"
        l6_jitter.SYNC_EVERY = sync_every
        l6_jitter.SYNC_VALVE = args.valve
        l6_jitter._install_sync_point = lambda cls=None: True  # the propose point is called directly below
        l6_jitter._gc_collect = lever_collect
        orig_note = l6_ab.note

        def note(lever, key, n=1):
            if lever == "syncgc" and key in ("collect", "valve"):
                collects.append((l6_ab.step_seq(), key))
            return orig_note(lever, key, n)

        l6_ab.note = note
    l6_ab._resolve()
    l6_jitter.mark_warm()
    gc.enable()
    gc.set_threshold(args.gen0, 10, 10)
    gc.collect()
    prev = None
    steps_us = []
    gc.callbacks.append(timer)
    try:
        for j in range(1, args.warm + args.steps + 1):
            if j == args.warm + 1:
                pauses.clear()
                collects.clear()
            t = time.perf_counter_ns()
            l6_ab.begin_step()
            l6_jitter.on_step()
            prev = None  # the previous step's objects released (tpu_runner :6101)
            if j % max(1, args.life // args.req) == 0:
                reqs.pop(0)
                reqs.append(new_req(nxt))
                nxt += 1
            h = step_objects(args, j, reqs, cyc_every)
            l6_jitter.sync_point()  # the draft propose (just before dp_entry)
            prev = h
            if j > args.warm:
                steps_us.append((time.perf_counter_ns() - t) / 1000)
    finally:
        gc.callbacks.remove(timer)
        if sync_every:
            l6_ab.note = orig_note
            l6_jitter._gc_collect = real_collect
        l6_jitter.restore()
        gc.enable()
        for k in [k for k in os.environ if k.startswith("VLLM_TPU_L6_")]:
            del os.environ[k]
        l6_ab._resolve()
    del prev
    return steps_us, pauses, collects


def report(args, name, cyc, sync_every, steps_us, pauses, collects):
    ok = True
    n = len(steps_us)
    auto = [p for p in pauses if not p[2]]
    mine = [p for p in pauses if p[2]]
    per_gen = []
    for g in (0, 1, 2):
        xs = [us for gen, us, _ in pauses if gen == g]
        per_gen.append(f"g{g}:n={len(xs)},p50={pct(xs, .5):.0f},p99={pct(xs, .99):.0f},max={max(xs, default=0):.0f}")
    gc_us = sum(us for _, us, _ in pauses)
    line = (f"BENCH arm={name} cyc={cyc:.2f} N={sync_every or '-'} steps={n} step_us p50={pct(steps_us, .5):.1f} "
            f"p99={pct(steps_us, .99):.1f} max={max(steps_us):.0f} gc_us_per_step={gc_us / n:.2f} "
            f"collections={len(pauses)} auto={len(auto)} lever={len(mine)} " + " ".join(per_gen))
    if sync_every:
        lock = [j for j, kind in collects if kind == "collect"]
        valve = [j for j, kind in collects if kind == "valve"]
        off = [j for j in lock if j % sync_every]
        line += f" lockstep={len(lock)} valve={len(valve)} off_index={len(off)}"
        if auto or off or not lock:
            ok = False
            line += " CHECK=FAIL"
        else:
            line += " CHECK=ok"
    print(line, flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", type=int, default=1_000_000)
    ap.add_argument("--meta", type=int, default=59)
    ap.add_argument("--walk", type=int, default=74)
    ap.add_argument("--tensors", type=int, default=40)
    ap.add_argument("--req", type=int, default=32)
    ap.add_argument("--life", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--gen0", type=int, default=50000)
    ap.add_argument("--valve", type=int, default=100000)
    ap.add_argument("--every", default="50,100,200")
    ap.add_argument("--cyc", default="0,0.1")
    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from vllm_torchtpu.core import l6_ab, l6_jitter

    t = time.perf_counter()
    static = build_static(args.static)
    gc.collect()
    gc.freeze()
    print(f"BENCH heap static={args.static} frozen={gc.get_freeze_count()} meta={args.meta} walk={args.walk} "
          f"tensors={args.tensors} req={args.req} life={args.life} steps={args.steps} warm={args.warm} "
          f"gen0={args.gen0} valve={args.valve} python={sys.version.split()[0]} build_s={time.perf_counter() - t:.1f}",
          flush=True)
    ok = True
    for cyc in [float(x) for x in args.cyc.split(",")]:
        cyc_every = int(round(1 / cyc)) if cyc > 0 else 0
        steps_us, pauses, collects = run_arm(args, 0, cyc_every, l6_ab, l6_jitter)
        ok &= report(args, "base", cyc, 0, steps_us, pauses, collects)
        for n in [int(x) for x in args.every.split(",")]:
            steps_us, pauses, collects = run_arm(args, n, cyc_every, l6_ab, l6_jitter)
            ok &= report(args, "sync", cyc, n, steps_us, pauses, collects)
    del static
    print(f"BENCH_SYNCGC_DONE rc={0 if ok else 1}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
