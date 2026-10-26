"""K-MLAM microbenchmark: mixed-step HCA attention, per cell, per arm.

  python bench_mixed.py --cells a,b --arms stock[,stock_kvp128,mod:fn] [--profile]

Clock (M1/M2): every (arm, cell, M) program is built and compiled BEFORE timing; M calls UNROLLED in one jit,
chained through the loop carry (q <- previous out, swa <- previous aliased SWA cache), so no call is invariant.
Per call = least-squares slope over M in MS (wall), rep 0 held out, median of REPS; hoist ratio printed;
custom-call count of the M=1 program printed. --profile: jax.profiler traces of M=1 and M=12 programs ->
tools/devtime.py gives the device per-layer time (busy12-busy1)/11 and the per-layer op template (M4).
Arms: 'stock' (served env), 'stock_<knob>' (incumbent tuning knobs, see STOCK_KNOBS), 'module:fn' where
fn(d) -> (out, swa) like mlam_common.stock_attn().
"""
import argparse
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlam_common as MC  # noqa: E402

import numpy as np  # noqa: E402

MS = tuple(int(x) for x in os.environ.get("KMLAM_MS", "1,4,12").split(","))
REPS = int(os.environ.get("REPS", "5"))
PROF_MS = (1, 12)
STOCK_KNOBS = {
    "stock": {},
    "stock_kvp128": dict(kvp=(8, 128, 128)),
    "stock_kvp32": dict(kvp=(8, 32, 32)),
    "stock_bq16": dict(nq=(1, 16, 16)),
    "stock_bq64": dict(nq=(1, 64, 64)),
}


def chained(f, M):
    import jax

    def run(d):
        q, swa = d["q"], d["swa"]
        for _ in range(M):
            out, swa = f(dict(d, q=q, swa=swa))
            q = out.astype(q.dtype)
        return q, swa

    return jax.jit(run, donate_argnums=())


def count_custom_calls(text):
    import re
    return len(re.findall(r'custom_call_target="([^"]+)"', text))


def time_fn(fn, d):
    import jax
    ts = []
    for _ in range(REPS + 1):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(d))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts[1:]))


def resolve(arm):
    if arm in STOCK_KNOBS:
        return MC.stock_attn(**STOCK_KNOBS[arm])
    if arm == "stock_prefetch":
        f0 = MC.stock_attn()

        def f(d):  # env read at trace time; restored so later arms are unaffected
            old = os.environ.get("TPU_DSV4_HCA_PREFETCH_EARLY")
            os.environ["TPU_DSV4_HCA_PREFETCH_EARLY"] = "1"
            try:
                return f0(d)
            finally:
                if old is None:
                    os.environ.pop("TPU_DSV4_HCA_PREFETCH_EARLY")
                else:
                    os.environ["TPU_DSV4_HCA_PREFETCH_EARLY"] = old
        return f
    m, fn = arm.split(":")
    return getattr(importlib.import_module(m), fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(MC.PHASE_CELLS))
    ap.add_argument("--arms", default="stock")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "."), "bench.jsonl"))
    a = ap.parse_args()
    MC.stock_env()
    import jax
    print(jax.devices()[0], jax.__version__, flush=True)
    out_dir = os.path.dirname(a.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    fo = open(a.out, "a")
    for name in a.cells.split(","):
        cell = MC.CELLS[name]
        inp = MC.build(cell)
        d = MC.device(inp)
        for arm in a.arms.split(","):
            try:
                f = resolve(arm)
                fns, comp_s = {}, {}
                for M in sorted(set(MS) | set(PROF_MS if a.profile else ())):
                    t0 = time.perf_counter()
                    fns[M] = chained(f, M).lower(d).compile()
                    comp_s[M] = round(time.perf_counter() - t0, 1)
                ncc = count_custom_calls(fns[1].as_text())
                times = {}
                for M in MS:
                    jax.block_until_ready(fns[M](d))
                    times[M] = time_fn(fns[M], d)
                slope, icpt = np.polyfit(MS, [times[m] for m in MS], 1)
                rec = dict(cell=name, arm=arm, T=cell.T, H=cell.heads, reqs=[list(x) for x in cell.reqs],
                           per_call_us=slope * 1e6, intercept_us=icpt * 1e6, t_us={m: times[m] * 1e6 for m in MS},
                           hoist_ratio=times[MS[-1]] / times[MS[0]], compile_s=comp_s, custom_calls=ncc)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                rec = dict(cell=name, arm=arm, error=repr(ex)[:800])
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                print("BENCH " + json.dumps(rec), flush=True)
                continue
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print("BENCH " + json.dumps(rec), flush=True)
            if a.profile:
                pdir = os.path.join(out_dir, "xprof", f"{name}.{arm.replace(':', '_')}")
                for pm in PROF_MS:
                    jax.block_until_ready(fns[pm](d))
                    jax.profiler.start_trace(pdir + f".M{pm}")
                    for _ in range(3):
                        jax.block_until_ready(fns[pm](d))
                    jax.profiler.stop_trace()
        del d


if __name__ == "__main__":
    main()
