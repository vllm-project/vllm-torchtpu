# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Kernel microbenchmark for moe_tp_decode and its variants (multi-controller; 16 ranks on tpu7x 2x2x2).

Arms: "F" = the incumbent module verbatim (vllm_torchtpu.kernels.experimental.moe_tp_decode.kernel); further arms come
from --plugin FILE, a module with ARMS {name: dele flags} and fn(name) -> callable with moe_tp_decode's signature.

Clock (M1/M2): every (arm, dep, m) program is compiled before any timing (compile seconds printed). A program runs
`nsteps` chained layer calls inside one jit: lax.fori_loop(0, nsteps, body, x) with nsteps a traced operand (one
compile serves every M) and body(x) = x.at[0, 0].set(kernel(x)[0, 0]) — the loop carry flows through the kernel, so
the call cannot be hoisted; weights are HBM operands re-read by the kernel's own DMAs every call. Checked per program:
exactly one tpu_custom_call, inside a while body; the while body's non-kernel ops (printed, diffed against F: M4).
Per-call cost = least-squares slope of wall time over M in MS; intercept reported; hoist check T(M_hi)/T(M_lo) >= 3.
Rounds interleave every (cell, arm) with F first and last in each cell (within-round A/A); round 0 is held out.

Output lines (process 0): PROG (compile / census), DRV (sweep drivers), T (raw), SLOPE (per round), SUMMARY.
"""

import argparse
import importlib.util
import json
import os
import re
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec  # noqa: E402
import tpu_common as tc  # noqa: E402

from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as K  # noqa: E402


def load_plugin(path):
    if not path:
        return None
    s = importlib.util.spec_from_file_location("bench_plugin", path)
    mod = importlib.util.module_from_spec(s)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    s.loader.exec_module(mod)
    return mod


def chain_program(msh, fn_body):
    def loop(nsteps, x, scores, bias, nv, rank, *rest):
        def body(_, xc):
            out = fn_body(xc, scores, bias, nv, rank, *rest)
            return xc.at[0, 0].set(out[0, 0])

        return lax.fori_loop(0, nsteps[0], body, x)

    specs = (tc.REP,) + tc.KERNEL_SPECS
    return jax.jit(jax.shard_map(loop, mesh=msh, in_specs=specs, out_specs=tc.PER, check_vma=False))


def census(hlo):
    calls = hlo.count('custom_call_target="tpu_custom_call"')
    names = sorted(set(re.findall(r"moe_tp_decode-[A-Za-z0-9_\-]+", hlo)))
    has_while = " while(" in hlo or "= while(" in hlo or "while(" in hlo
    # opcodes of every non-entry computation's instructions (coarse M4 region print)
    ops = re.findall(r"=\s*[a-z0-9\[\]{},:\s]*?\b([a-z][a-z0-9\-]*)\(", hlo)
    hist = {}
    for o in ops:
        hist[o] = hist.get(o, 0) + 1
    return calls, names, has_while, dict(sorted(hist.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps", default="tp16,ep16")
    ap.add_argument("--ms", default="64,112,192")
    ap.add_argument("--kinds", default="indep")
    ap.add_argument("--arms", default="F", help="comma list, or 'plugin' for plugin.arms_for(dep)")
    ap.add_argument("--plugin", default="")
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--msteps", default="2,8,32")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--smoke", action="store_true", help="CPU harness smoke: skip the custom-call check")
    a = ap.parse_args()
    pid = tc.init()
    say = (lambda *s: print(*s, flush=True)) if pid == 0 else (lambda *s: None)
    msh = tc.mesh()
    n = len(msh.devices.ravel())
    plug = load_plugin(a.plugin)
    MS = [int(v) for v in a.msteps.split(",")]
    deps = [spec.DEPS[d] for d in a.deps.split(",")]
    ms = [int(v) for v in a.ms.split(",")]
    kinds = a.kinds.split(",")
    say("BENCH", json.dumps(dict(tag=a.tag, n_ranks=n, procs=jax.process_count(), deps=a.deps, ms=ms, kinds=kinds,
                                 arms=a.arms, rounds=a.rounds, msteps=MS, jax=jax.__version__,
                                 kernel_file=K.__file__, env={k: v for k, v in os.environ.items() if k.startswith("TPU_MOE")})))

    def arm_list(dep):
        return list(plug.arms_for(dep.name)) if a.arms == "plugin" else a.arms.split(",")

    weights = {dep.name: tc.rank_weights_global(msh, dep, seed=11) for dep in deps}
    progs, inputs = {}, {}
    for dep in deps:
        for m in ms:
            for kind in kinds:
                args, (x, scores, bias, p) = tc.cell_inputs(msh, dep, m, kind, m)
                inputs[(dep.name, m, kind)] = args
                say("DRV", json.dumps(dict(dep=dep.name, m=m, kind=kind, **spec.drivers(p, m, dep))))
            for arm in arm_list(dep):
                fn = K.moe_tp_decode if arm == "F" else plug.fn(arm)
                prog = chain_program(msh, tc.kernel_call(fn, dep, n))
                args = inputs[(dep.name, m, kinds[0])]
                nst = tc.global_from_host(msh, np.array([1], np.int32), tc.REP)
                t0 = time.perf_counter()
                comp = prog.lower(nst, *tc.kernel_args(args, weights[dep.name])).compile()
                dt = time.perf_counter() - t0
                calls, names, has_while, hist = census(comp.as_text())
                ok = (calls == 1 and has_while) or a.smoke
                say("PROG", json.dumps(dict(dep=dep.name, m=m, arm=arm, compile_s=round(dt, 2), custom_calls=calls,
                                            while_loop=has_while, clock_ok=ok, names=names, ops=hist)))
                if not ok:
                    say("VOID clock check failed", dep.name, m, arm)
                    sys.exit(4)
                progs[(dep.name, m, arm)] = comp
    nsteps = {M: tc.global_from_host(msh, np.array([M], np.int32), tc.REP) for M in MS}
    slopes = {}

    def timed(comp, args, w, M):
        t0 = time.perf_counter()
        out = comp(nsteps[M], *tc.kernel_args(args, w))
        out.block_until_ready()
        return time.perf_counter() - t0

    for r in range(a.rounds):
        for dep in deps:
            arms = arm_list(dep)
            seq = ["F"] + [x for x in arms if x != "F"] + ["F"] if "F" in arms else arms
            for m in ms:
                for kind in kinds:
                    args = inputs[(dep.name, m, kind)]
                    for j, arm in enumerate(seq):
                        comp = progs[(dep.name, m, arm)]
                        ts = [timed(comp, args, weights[dep.name], M) for M in MS]
                        A = np.vstack([np.array(MS, float), np.ones(len(MS))]).T
                        slope, icpt = np.linalg.lstsq(A, np.array(ts), rcond=None)[0]
                        key = (dep.name, m, kind, arm + ("@last" if (arm == "F" and j == len(seq) - 1) else ""))
                        say("T", json.dumps(dict(r=r, dep=dep.name, m=m, kind=kind, arm=key[3],
                                                 t_us=[round(t * 1e6, 2) for t in ts], slope_us=round(slope * 1e6, 3),
                                                 icpt_us=round(icpt * 1e6, 2), hoist=round(ts[-1] / ts[0], 2))))
                        if r > 0:
                            slopes.setdefault(key, []).append((slope * 1e6, icpt * 1e6, ts[-1] / ts[0]))
    for key, v in sorted(slopes.items()):
        s = [x[0] for x in v]
        med = statistics.median(s)
        say("SUMMARY", json.dumps(dict(dep=key[0], m=key[1], kind=key[2], arm=key[3], per_call_us=round(med, 3),
                                       spread_pct=round(100 * (max(s) - min(s)) / med, 2) if med else None,
                                       n=len(s), icpt_us=round(statistics.median([x[1] for x in v]), 1),
                                       hoist_min=round(min(x[2] for x in v), 2))))
    say("BENCH_DONE", a.tag)


if __name__ == "__main__":
    main()
