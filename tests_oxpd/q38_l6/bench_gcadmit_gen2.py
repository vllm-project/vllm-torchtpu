#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""q38 L6 GC-ADMIT-FREEZE: gen-2 walk time on a synthetic worker heap of about 10^7 references, three arms
(CDK CPU pod only; never the Mac or the VM).

Usage (from a tree root, PYTHONPATH=src):  python3 tests/q38_l6/bench_gcadmit_gen2.py
Last line: BENCH_GCADMIT_DONE rc=0 | rc=1 (a child failed or a sanity check failed).

Each arm runs in its own subprocess (fresh heap). Common to all arms: vLLM / torch imported, a static heap of
STATIC dicts, vLLM's freeze_gc_heap() (the worker's warm-up freeze), then R requests of T prompt tokens
(R x T = 10^7 references) admitted ADMIT per step, each unpickled from an engine-side payload and built as
vLLM's CachedRequestState (SamplingParams, two KV groups' block ids, a 64-token output list):
  a  base       prompt ids as Python lists (what the worker holds today without a PromptTail);
  b  gcadmit    as a, plus the real lever (core/l6_gcadmit.after_update_states, flag 1) after each admission step;
  c  untracked  prompt ids shipped and held as int32 numpy arrays (not GC-tracked), as the existing
                VLLM_TPU_PROMPT_IDS_NDARRAY=1 does (core/prompt_ids_ndarray.py; engine-side substitution).
Then, per arm:
  * forced collections: gc.collect(2) x REPS (the gen-2 walk), gc.collect(1) and gc.collect(0) x 5 (µs);
  * a decode loop of LOOP steps with automatic GC (CPython default thresholds 700 / 10 / 10): per step one token
    appended to every request's output list, ~200 short-lived tracked objects, 4 small garbage cycles and 2
    objects kept in a 1,000-slot ring (so some reach generation 2); every collection timed by gc.callbacks;
  * admission-step µs (unpickle + state build, and the lever's collect + freeze in arm b);
  * gc.get_freeze_count() µs (CPython 3.12 walks the permanent generation) and RSS.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import random
import statistics
import subprocess
import sys
import time

R = 100
T = 100_000
ADMIT = 4
STATIC = 300_000
REPS = 7
LOOP = 3000
VOCAB = 151_000
BLOCK = 256


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20


def child(arm: str) -> int:
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k == "VLLM_TPU_HOST_GC":
            del os.environ[k]
    if arm == "b":
        os.environ["VLLM_TPU_L6_GC_ADMIT_FREEZE"] = "1"
    from types import SimpleNamespace as NS

    import numpy as np
    from vllm.sampling_params import SamplingParams
    from vllm.utils.gc_utils import freeze_gc_heap
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    from vllm_torchtpu.core import l6_ab, l6_gcadmit, prompt_ids_ndarray

    l6_ab._resolve()
    l6_gcadmit._resolve()
    assert l6_gcadmit.ON == (arm == "b")
    rng = random.Random(7)
    payloads = []
    for i in range(R):
        ids = [rng.randrange(VOCAB) for _ in range(T)]
        p = prompt_ids_ndarray.to_array(ids) if arm == "c" else ids
        payloads.append(pickle.dumps({"req_id": f"r{i}", "prompt_token_ids": p}, protocol=5))
        del ids, p
    static = [{"k": [i, i + 1], "t": (i, str(i)), "s": {i}} for i in range(STATIC)]
    freeze_gc_heap()
    t_build = time.perf_counter()
    requests: dict = {}
    runner = NS(warmup_done=True)
    admit_us, lever_us = [], []
    for s0 in range(0, R, ADMIT):
        t0 = time.perf_counter_ns()
        new = []
        for i in range(s0, min(R, s0 + ADMIT)):
            nrd = pickle.loads(payloads[i])
            ids = nrd["prompt_token_ids"]
            nblk = -(-len(ids) // BLOCK)
            requests[nrd["req_id"]] = CachedRequestState(
                req_id=nrd["req_id"], prompt_token_ids=ids, mm_features=[],
                sampling_params=SamplingParams(temperature=1.0, top_k=20, top_p=0.95, max_tokens=4096),
                generator=None, block_ids=(list(range(nblk)), list(range(nblk, 2 * nblk))),
                num_computed_tokens=len(ids), output_token_ids=[rng.randrange(VOCAB) for _ in range(64)])
            new.append(NS(req_id=nrd["req_id"]))
        t1 = time.perf_counter_ns()
        if arm == "b":
            l6_gcadmit.after_update_states(runner, NS(scheduled_new_reqs=new, total_num_scheduled_tokens=len(new),
                                                      num_scheduled_tokens={}))
        t2 = time.perf_counter_ns()
        admit_us.append((t1 - t0) // 1000)
        lever_us.append((t2 - t1) // 1000)
    del payloads
    build_s = time.perf_counter() - t_build
    st = next(iter(requests.values()))
    tracked = gc.is_tracked(st.prompt_token_ids)
    refs = sum(len(r.prompt_token_ids) for r in requests.values()) if arm != "c" else 0
    walked_refs_est = refs + sum(len(r.output_token_ids) + 2 * len(r.block_ids[0]) for r in requests.values())

    gen = {0: [], 1: [], 2: []}
    t0s = [0]

    def timer(phase, info):
        if phase == "start":
            t0s[0] = time.perf_counter_ns()
        elif t0s[0]:
            gen[min(int(info.get("generation", 0)), 2)].append((time.perf_counter_ns() - t0s[0]) // 1000)
            t0s[0] = 0

    def forced(g, n):
        out = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            gc.collect(g)
            out.append((time.perf_counter_ns() - t0) // 1000)
        return out

    f2, f1, f0 = forced(2, REPS), forced(1, 5), forced(0, 5)
    gc.collect()
    gc.callbacks.append(timer)
    ring = [None] * 1000
    tok = 0
    t_loop = time.perf_counter()
    for j in range(LOOP):
        for r in requests.values():
            tok = (tok + 7919) % VOCAB
            r.output_token_ids.append(tok)
        short = [{"i": i, "v": [i]} for i in range(100)]
        for _ in range(4):
            a, b = [], []
            a.append(b)
            b.append(a)
        ring[j % 1000] = ({"j": j}, [j])
        del short, a, b
    loop_s = time.perf_counter() - t_loop
    gc.callbacks.remove(timer)
    fcu = []
    for _ in range(5):
        t0 = time.perf_counter_ns()
        fc = gc.get_freeze_count()
        fcu.append((time.perf_counter_ns() - t0) // 1000)

    def summ(xs):
        if not xs:
            return {"n": 0, "p50": 0, "max": 0, "sum": 0}
        ys = sorted(xs)
        return {"n": len(ys), "p50": ys[len(ys) // 2], "max": ys[-1], "sum": sum(ys)}

    counts = dict(l6_ab._state.get("counts", {}))
    out = {"arm": arm, "tracked_prompt": tracked, "prompt_refs": refs, "walked_refs_est": walked_refs_est,
           "forced_g2": f2, "forced_g1": f1, "forced_g0": f0, "auto": {g: summ(v) for g, v in gen.items()},
           "admit_us": summ(admit_us), "lever_us": summ(lever_us), "freeze_count": fc,
           "freeze_count_us": statistics.median(fcu), "rss_mb": rss_mb(), "build_s": build_s, "loop_s": loop_s,
           "counts": counts, "static": len(static), "python": sys.version.split()[0],
           "np_tracked": gc.is_tracked(np.zeros(4, dtype=np.int32))}
    print("BENCH_CHILD_JSON " + json.dumps(out), flush=True)
    return 0


def parent() -> int:
    print(f"BENCH_GCADMIT heap R={R} T={T} refs={R * T} admit_per_step={ADMIT} static={STATIC} reps={REPS} "
          f"loop={LOOP}", flush=True)
    res, rc = {}, 0
    for arm in ("a", "b", "c"):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "child", arm], capture_output=True,
                           text=True, timeout=3000, env=dict(os.environ))
        js = [ln for ln in p.stdout.splitlines() if ln.startswith("BENCH_CHILD_JSON ")]
        if p.returncode != 0 or not js:
            print(f"BENCH arm={arm} CHILD FAILED rc={p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
            rc = 1
            continue
        d = json.loads(js[-1][len("BENCH_CHILD_JSON "):])
        res[arm] = d
        f2 = sorted(d["forced_g2"])
        au = d["auto"]
        print(f"BENCH arm={arm} python={d['python']} prompt_tracked={d['tracked_prompt']} prompt_refs={d['prompt_refs']} "
              f"forced_g2_us p50={f2[len(f2) // 2]} min={f2[0]} max={f2[-1]} "
              f"forced_g1_us p50={sorted(d['forced_g1'])[2]} forced_g0_us p50={sorted(d['forced_g0'])[2]} "
              f"auto g0 n={au['0']['n']} p50={au['0']['p50']} max={au['0']['max']} "
              f"g1 n={au['1']['n']} p50={au['1']['p50']} max={au['1']['max']} "
              f"g2 n={au['2']['n']} p50={au['2']['p50']} max={au['2']['max']} sum={au['2']['sum']} "
              f"loop_s={d['loop_s']:.2f} admit_step_us p50={d['admit_us']['p50']} max={d['admit_us']['max']} "
              f"lever_us p50={d['lever_us']['p50']} max={d['lever_us']['max']} "
              f"freeze_count={d['freeze_count']} get_freeze_count_us={d['freeze_count_us']} "
              f"rss_mb={d['rss_mb']:.0f} counts={d['counts']}", flush=True)
    if set(res) == {"a", "b", "c"}:
        g2 = {a: sorted(res[a]["forced_g2"])[REPS // 2] for a in res}
        ok = (res["a"]["tracked_prompt"] and res["b"]["tracked_prompt"] and not res["c"]["tracked_prompt"]
              and not res["c"]["np_tracked"] and res["b"]["counts"].get("gcadmit.freeze") == -(-R // ADMIT))
        ns_per_ref = (g2["a"] - g2["c"]) * 1000 / (R * T) if g2["a"] > g2["c"] else float("nan")
        print(f"BENCH_GEN2 forced gen-2 p50 us: a(base)={g2['a']} b(gcadmit)={g2['b']} c(untracked)={g2['c']} "
              f"b/a={g2['b'] / max(g2['a'], 1):.4f} c/a={g2['c'] / max(g2['a'], 1):.4f} "
              f"token-list walk ns/ref={ns_per_ref:.2f}", flush=True)
        if ns_per_ref == ns_per_ref:
            lo, hi = 13.1e9 / ns_per_ref, 19.4e9 / ns_per_ref
            print(f"BENCH_SCALE T98o's 13.1-19.4 s gen-2 at this pod's {ns_per_ref:.2f} ns per list reference would "
                  f"be {lo:.3g}-{hi:.3g} references (I: CPU pod, not the v7x host)", flush=True)
        print(f"BENCH_SANITY {'ok' if ok else 'BAD'} (a/b prompts tracked, c untracked, b froze once per step)")
        rc |= 0 if ok else 1
    print(f"BENCH_GCADMIT_DONE rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "child":
        sys.exit(child(sys.argv[2]))
    sys.exit(parent())
