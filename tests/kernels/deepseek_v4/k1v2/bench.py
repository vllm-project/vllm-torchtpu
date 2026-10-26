"""K1v2 microbenchmark: per-cell, per-arm chained device clock + xprof capture.

  python bench.py --cells a,b --arms inc,udx[,mod:fn] [--profile]

Clock (M1/M2): every (arm, cell, M) callable is built + compiled BEFORE timing. M calls are UNROLLED in one jit
(served-like: the layer-invariant index glue is CSE'd once per program, STRAT-DSV4-02), chained q <- previous output
(no call is loop-invariant). Per call = least-squares slope over M in MS (rep 0 held out, median of REPS).
Hoist guards: T(M_hi)/T(1) and the program's custom-call count (expected to scale with M) printed; the top-k is passed
through an optimization_barrier with the chained q each call so the gather path cannot be CSE'd across calls.
--profile: jax.profiler trace at M=1 and M=12 -> per-layer DEVICE time = (dev12 - dev1)/11 from tools/devtime.py
(the decision clock: K-HCA measured the wall slope noisy at +-2 us on a ~1 ms harness intercept).
Drivers (M6): gathered rows (incumbent), union rows/tiles, bytes, per cell -- printed from the same inputs.
"""
import argparse, importlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import k1v2_common as C

MS = tuple(int(x) for x in os.environ.get("K1V2_MS", "1,4,12").split(","))
REPS = int(os.environ.get("REPS", "7"))
PROF_MS = (1, 12)


def chained(f, M):
    import jax

    def run(d):
        q = d["q"]
        for _ in range(M):
            # M2: the per-layer top-k must be a distinct value per call (as in the served model, where every CSA layer
            # has its own indexer output); otherwise XLA CSEs the gather path (page ids, SC gather, union plan) once
            # per program and the chain times the attention kernel alone (j-c694e3a0: 1 custom call per inc call).
            q, topk = jax.lax.optimization_barrier((q, d["topk"]))
            q = f(dict(d, topk=topk), q).astype(q.dtype)
        return q
    return jax.jit(run)


def count_custom_calls(txt):
    import re
    return len(re.findall(r'custom_call_target="tpu_custom_call"', txt)), sorted(set(re.findall(r'kernel_name="?([A-Za-z0-9_\-\.]+)', txt)))[:30]


def time_fn(fn, d):
    import jax
    ts = []
    for _ in range(REPS + 1):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(d))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(C.PHASE_CELLS))
    ap.add_argument("--arms", default="inc")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "."), "bench.jsonl"))
    a = ap.parse_args()
    import jax
    print(jax.devices()[0], jax.__version__, flush=True)
    out_dir = os.path.dirname(a.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    fo = open(a.out, "a")
    for name in a.cells.split(","):
        cell = C.CELLS[name]
        inp = C.build_inputs(cell)
        drv = C.drivers(inp)
        print("DRIVERS " + json.dumps(dict(cell=name, **drv, ctx=inp["ctx"])), flush=True)
        d, _ = C.device_inputs(inp, cache="random")
        for arm in a.arms.split(","):
            if arm in C.ARMS:
                f = C.ARMS[arm](cell.conc)
            else:
                m, fn = arm.split(":")
                f = getattr(importlib.import_module(m), fn)(cell.conc)
            try:
                fns, comp_s = {}, {}
                for M in sorted(set(MS) | (set(PROF_MS) if a.profile else set())):
                    t0 = time.perf_counter()
                    fns[M] = chained(f, M).lower(d).compile()
                    comp_s[M] = round(time.perf_counter() - t0, 1)
                ncc = {M: count_custom_calls(fns[M].as_text())[0] for M in (1, max(MS))}
                kn = count_custom_calls(fns[1].as_text())[1]
                times = {}
                for M in MS:
                    jax.block_until_ready(fns[M](d))
                    times[M] = time_fn(fns[M], d)
                slope, icpt = np.polyfit(MS, [times[m] for m in MS], 1)
                rec = dict(cell=name, arm=arm, T=cell.T, H=cell.heads, R=cell.R, per_call_us=round(slope * 1e6, 2),
                           intercept_us=round(icpt * 1e6, 1), t_us={m: round(times[m] * 1e6, 1) for m in MS},
                           hoist_ratio=round(times[max(MS)] / times[1], 3), compile_s=comp_s, custom_calls=ncc, kernels=kn,
                           union=drv["union"], union_tiles=drv["union_tiles"], gathered_rows_inc=drv["gathered_rows_incumbent"])
            except Exception as e:  # noqa
                rec = dict(cell=name, arm=arm, error=repr(e)[:1500])
                print("BENCH " + json.dumps(rec), flush=True)
                fo.write(json.dumps(rec) + "\n"); fo.flush()
                continue
            fo.write(json.dumps(rec) + "\n"); fo.flush()
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
