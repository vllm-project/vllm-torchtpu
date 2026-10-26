# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark of the mixed-step routing plan + combine, per cell.

Clock (write-tpu-kernels M1/M2): every (arm, cell, kind, R) callable is built
and compiled before any timing. A call is chained R times inside one jitted
fori_loop; the loop carry is the previous call's OUTPUTS, barriered together
with the loop-invariant inputs, so XLA can neither hoist the call nor narrow
it to the elements a reduction would read. Per-call cost = least-squares
slope over R; the intercept (host dispatch + loop entry) is printed apart.
Hoist check: T(R_max)/T(R_min) must be well above 1.

Each arm's compiled entry computation is censused by opcode (the region's XLA
op breakdown, M4): incumbent and candidate regions are compared from this.

usage: bench.py --cells mix128,mix256 --kinds served,served_pad --arms plan,combine
                [--out DIR] [--trace] [--reps 7] [--rs 4,12,24]
Arms: plan (incumbent XLA glue), combine (incumbent XLA/Pallas combine),
      plus any arm registered in CANDIDATES (filled after the freeze gate).
"""

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec  # noqa: E402

# name -> fn(cell, exchange, local_blocks, block_hist, me) -> dict like incumbent_plan.
CANDIDATES = {
    "a1": lambda c, e, l, b, m: spec.candidate_plan(c, e, l, b, m),
    # timing-only arms (wrong outputs allowed; M8 removal arms / lever pairs)
    "a1_nohi": lambda c, e, l, b, m: spec.candidate_plan(c, e, l, b, m, dc_hi=False),
    "a1_rm_s3": lambda c, e, l, b, m: spec.candidate_plan(c, e, l, b, m, fault="rm_s3"),
    "a1_rm_ydot": lambda c, e, l, b, m: spec.candidate_plan(c, e, l, b, m, fault="rm_ydot"),
}


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def census(compiled_text):
    """Opcode counts of the ENTRY computation, plus custom-call targets."""
    entry = compiled_text.split("ENTRY", 1)[1]
    body = entry.split("\n}\n", 1)[0]
    ops = collections.Counter()
    targets = collections.Counter()
    for line in body.splitlines():
        m = re.match(r"\s*(?:ROOT\s+)?%?[\w.\-]+\s*=\s*.*?\s([a-z][\w\-]*)\(", line)
        if not m:
            continue
        op = m.group(1)
        ops[op] += 1
        if op == "custom-call":
            t = re.search(r'custom_call_target="([^"]+)"', line)
            targets[t.group(1) if t else "?"] += 1
        if op == "fusion":
            k = re.search(r"kind=(\w+)", line)
            ops["fusion:" + (k.group(1) if k else "?")] += 1
    return dict(ops), dict(targets)


def build_plan_inputs(cell, kind, seed, me_mode):
    pairs = spec.make_pairs(cell, kind, seed)
    sb = spec.make_scale_bits(cell, seed)
    me = spec.busiest_shard(cell, pairs) if me_mode == "max" else int(me_mode)
    ex = spec.make_exchange(cell, pairs, sb)
    lb, bh = spec.local_inputs(cell, pairs, me)
    return pairs, me, (ex, lb, bh, jnp.int32(me))


def chained(fn, R):
    """R chained calls; carry = previous outputs, barriered with the inputs."""

    def run(first_args, out0):
        def body(i, carry):
            args, prev = carry
            args, prev = lax.optimization_barrier((args, prev))
            return args, fn(*args)

        _, out = lax.fori_loop(0, R, body, (first_args, out0))
        return out

    return jax.jit(run)


def time_arm(fn, args, rs, reps):
    out0 = jax.jit(fn)(*args)
    jax.block_until_ready(out0)
    fns, compile_s = {}, {}
    for R in rs:
        t0 = time.perf_counter()
        f = chained(fn, R)
        jax.block_until_ready(f(args, out0))
        compile_s[R] = time.perf_counter() - t0
        fns[R] = f
    times = {R: [] for R in rs}
    for _ in range(reps):
        for R in rs:
            t0 = time.perf_counter()
            jax.block_until_ready(fns[R](args, out0))
            times[R].append(time.perf_counter() - t0)
    med = {R: float(np.median(v[1:])) for R, v in times.items()}  # rep 0 held out
    x = np.array(rs, float)
    y = np.array([med[R] for R in rs])
    slope, icpt = np.polyfit(x, y, 1)
    return dict(
        per_call_us=slope * 1e6,
        intercept_us=icpt * 1e6,
        median_s=med,
        hoist_ratio=med[rs[-1]] / med[rs[0]],
        compile_s=compile_s,
        late_rep_vs_rep0=[times[R][-1] / times[R][0] for R in rs],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="mix128,mix256")
    ap.add_argument("--kinds", default="served,served_pad")
    ap.add_argument("--arms", default="plan,combine")
    ap.add_argument("--me", default="max")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--rs", default="4,12,24")
    ap.add_argument("--out", default=os.environ.get("CDK_OUTPUT_DIR", "/tmp") + "/kplan_bench")
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rs = [int(r) for r in a.rs.split(",")]
    here = os.path.dirname(os.path.abspath(__file__))
    import vllm_torchtpu.kernels.fused_moe.v2.host as host_mod
    import vllm_torchtpu.kernels.fused_moe.v2.layer as layer_mod

    meta = dict(
        devices=str(jax.devices()[0]),
        jax=jax.__version__,
        md5={p: md5(p) for p in (host_mod.__file__, layer_mod.__file__, spec.__file__, __file__)},
    )
    print("META", json.dumps(meta), flush=True)
    results = []
    jobs = []
    # M1: build and compile everything first.
    for cn in a.cells.split(","):
        cell = spec.CELLS[cn]
        for kind in a.kinds.split(","):
            pairs, me, pargs = build_plan_inputs(cell, kind, a.seed, a.me)
            for arm in a.arms.split(","):
                if arm == "plan":
                    fn = lambda e, l, b, m, c=cell: spec.incumbent_plan(c, e, l, b, m)
                    args = pargs
                elif arm == "combine" or arm == "pk128" or arm.split("@")[0] in spec.COMBINE_VARIANTS:
                    plan = jax.jit(lambda e, l, b, m, c=cell: spec.incumbent_plan(c, e, l, b, m))(*pargs)
                    arr, scl, w, routes = spec.make_combine_inputs(cell, pairs, me, a.seed)
                    args = (arr, scl, plan["pos"], plan["mirror_pos"], w, routes)
                    if arm == "combine":
                        fn = spec.incumbent_combine
                    elif arm == "pk128":
                        fn = spec.forced_pallas_combine
                    else:  # stage A2: <variant>[@opt;opt] (opts = timing-only removal arms)
                        v, _, o = arm.partition("@")
                        opts = tuple(x for x in o.split(";") if x)
                        fn = lambda *xs, v=v, opts=opts: spec.candidate_combine(*xs, variant=v, opts=opts)
                        from vllm_torchtpu.kernels.fused_moe.v2 import mixed_combine_kernel as mck
                        print("DRIVER", arm, cn, json.dumps(mck.descriptors(
                            cell.t_local, cell.topk, cell.recv_rows,
                            spec.COMBINE_VARIANTS[v]["gather"])), flush=True)
                elif arm.startswith("a1@"):  # timing-only: a1 with fault/skip=<arm[3:]> (";" for ",")
                    fn = lambda e, l, b, m, c=cell, f=arm[3:].replace(";", ","): spec.candidate_plan(c, e, l, b, m, fault=f)
                    args = pargs
                elif arm in CANDIDATES:
                    fn = lambda e, l, b, m, c=cell, f=CANDIDATES[arm]: f(c, e, l, b, m)
                    args = pargs
                else:
                    raise ValueError(arm)
                txt = jax.jit(fn).lower(*args).compile().as_text()
                ops, targets = census(txt)
                jobs.append((cn, kind, me, arm, fn, args, ops, targets))
    for cn, kind, me, arm, fn, args, ops, targets in jobs:
        r = time_arm(fn, args, rs, a.reps)
        r.update(cell=cn, kind=kind, me=me, arm=arm, entry_ops=ops, custom_calls=targets)
        results.append(r)
        print(
            f"RESULT {arm:10s} {cn:7s} {kind:10s} me={me:2d} per_call={r['per_call_us']:8.2f} us "
            f"icpt={r['intercept_us']:8.1f} hoist={r['hoist_ratio']:.2f} "
            f"entry_ops={sum(v for k, v in ops.items() if ':' not in k)} cc={targets}",
            flush=True,
        )
        print("CENSUS", arm, cn, kind, json.dumps(dict(sorted(ops.items(), key=lambda x: -x[1]))), flush=True)
    if a.trace:
        tdir = os.path.join(a.out, "trace")
        with jax.profiler.trace(tdir):
            for cn, kind, me, arm, fn, args, ops, targets in jobs:
                out0 = jax.jit(fn)(*args)
                f = chained(fn, 8)
                for _ in range(3):
                    jax.block_until_ready(f(args, out0))
        print("TRACE", tdir, flush=True)
    json.dump(dict(meta=meta, results=results), open(os.path.join(a.out, "bench.json"), "w"), indent=1, default=str)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
