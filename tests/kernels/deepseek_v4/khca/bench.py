"""K-HCA microbenchmark: per-cell, per-region chained device-loop clock + xprof capture.

  python bench.py --cells a,b --regions full,attn,comp,rope [--arms stock[,mod:fn]] [--filler] [--profile]

Clock (M1/M2): every (arm, region, cell, M) callable is built and compiled BEFORE timing; M calls are UNROLLED in one jit (served-like CSE of layer-invariant glue);
lax.fori_loop runs M calls with the LOOP CARRY = (q, swa, hca, st) (the call's output q feeds the next
call; caches are its aliased outputs), so no call is loop-invariant. Per call = least-squares slope over
M in MS; rep 0 held out; median of REPS. Hoist guards: T(M_hi)/T(1) printed; HLO custom-call count of the
loop body printed per unit. --filler adds a layer-sized bf16 dot (x[T,7168] @ W[7168,7168]) between calls
(MOE-11: a back-to-back chain can misstate a call); its own slope is measured and printed.
--profile writes a jax.profiler trace of M=PROF_M calls per (arm, region, cell) for the XLA op breakdown (M4).
"""
import argparse
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hca_common as C  # noqa: E402

import numpy as np  # noqa: E402

MS = tuple(int(x) for x in os.environ.get('KHCA_MS', '1,4,12').split(','))
REPS = int(os.environ.get("REPS", "7"))
PROF_MS = (1, 12)
REGIONS = {"full": ("rope", "comp", "attn"), "attn": ("attn",), "comp": ("comp",), "rope": ("rope",),
           "rope_attn": ("rope", "attn")}


def chained(f, M, filler):
    """M calls UNROLLED in one jit (served-like: layer-invariant glue is CSE'd once per program, STRAT-DSV4-02).
    Chain: q <- previous out, kv <- previous RoPE'd kv (stock rope is pure; an invariant kv would be CSE'd),
    caches <- previous aliased caches, hidden <- filler dot output when --filler."""
    import jax
    import jax.numpy as jnp

    def run(d, cs, fw):
        q, kv, swa, hca, st, x = d["q"], d["kv"], d["swa"], d["hca"], d["st"], d["hidden"]
        for _ in range(M):
            if filler:
                x = jnp.dot(x, fw, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
            dd = dict(d, q=q, kv=kv, swa=swa, hca=hca, st=st, hidden=x)
            out, swa, hca, st, kv = f(dd, cs, return_kv=True)
            q = out.astype(q.dtype)
        return q, swa, hca, st, x

    return jax.jit(run)


def filler_only(M):
    import jax
    import jax.numpy as jnp

    def run(x, fw):
        return jax.lax.fori_loop(0, M, lambda i, x: jnp.dot(x, fw, preferred_element_type=jnp.float32).astype(jnp.bfloat16), x)
    return jax.jit(run)


def count_custom_calls(compiled_text):
    import re
    names = re.findall(r'custom_call_target="([^"]+)"', compiled_text)
    kn = re.findall(r'kernel_name[^"]*"([^"]+)"', compiled_text)
    return len(names), sorted(set(kn))[:20]


def time_fn(fn, args):
    import jax
    ts = []
    for _ in range(REPS + 1):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts[1:])), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(C.PHASE_CELLS))
    ap.add_argument("--regions", default="full,attn,comp,rope")
    ap.add_argument("--arms", default="stock")
    ap.add_argument("--filler", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "."), "bench.jsonl"))
    a = ap.parse_args()
    C.stock_env()
    import jax
    import jax.numpy as jnp
    print(jax.devices()[0], jax.__version__, flush=True)
    arms = {}
    for arm in a.arms.split(","):
        if arm == "stock":
            arms[arm] = lambda parts: C.stock_region(parts)[0]
        else:
            m, fn = arm.split(":")
            arms[arm] = getattr(importlib.import_module(m), fn)
    cs = jnp.asarray(C.cos_sin_np())
    out_dir = os.path.dirname(a.out) or "."
    fo = open(a.out, "a")
    for name in a.cells.split(","):
        cell = C.CELLS[name]
        inp = C.build_inputs(cell)
        d = C.device_inputs(inp)
        fw = (jax.random.normal(jax.random.PRNGKey(7), (C.HIDDEN, C.HIDDEN), jnp.float32) * C.HIDDEN ** -0.5).astype(jnp.bfloat16)
        fill = {}
        if a.filler:
            for M in MS:
                fn = filler_only(M)
                jax.block_until_ready(fn(d["hidden"], fw))
                fill[M] = time_fn(fn, (d["hidden"], fw))[0]
            fs = np.polyfit(MS, [fill[m] for m in MS], 1)[0] * 1e6
            print(f"FILLER cell={name} slope_us={fs:.2f}", flush=True)
        for arm, mk in arms.items():
            for reg in a.regions.split(","):
                f = mk(REGIONS[reg])
                # M1: build + compile every M first
                fns, comp_s = {}, {}
                for M in MS:
                    t0 = time.perf_counter()
                    fn = chained(f, M, a.filler)
                    lowered = fn.lower(d, cs, fw)
                    compiled = lowered.compile()
                    comp_s[M] = time.perf_counter() - t0
                    fns[M] = compiled
                ncc, kn = count_custom_calls(fns[1].as_text())
                times = {}
                for M in MS:
                    jax.block_until_ready(fns[M](d, cs, fw))
                    times[M] = time_fn(fns[M], (d, cs, fw))[0]
                slope, icpt = np.polyfit(MS, [times[m] for m in MS], 1)
                rec = dict(cell=name, arm=arm, region=reg, filler=a.filler, T=cell.T, H=cell.heads, R=cell.R,
                           ctx=list(cell.ctx), per_call_us=slope * 1e6, intercept_us=icpt * 1e6,
                           t_us={m: times[m] * 1e6 for m in MS}, hoist_ratio=times[MS[-1]] / times[1],
                           compile_s={m: round(comp_s[m], 1) for m in MS}, custom_calls_in_program=ncc, kernels=kn)
                if a.filler:
                    rec["filler_slope_us"] = float(np.polyfit(MS, [fill[m] for m in MS], 1)[0] * 1e6)
                    rec["per_call_net_us"] = rec["per_call_us"] - rec["filler_slope_us"]
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                print("BENCH " + json.dumps(rec), flush=True)
                if a.profile:
                    pdir = os.path.join(out_dir, "xprof", f"{name}.{arm.replace(':', '_')}.{reg}{'.fill' if a.filler else ''}")
                    for pm in PROF_MS:
                        fnp = fns[pm] if pm in fns else chained(f, pm, a.filler)
                        jax.block_until_ready(fnp(d, cs, fw))
                        jax.profiler.start_trace(pdir + f".M{pm}")
                        for _ in range(3):
                            jax.block_until_ready(fnp(d, cs, fw))
                        jax.profiler.stop_trace()
        del d


if __name__ == "__main__":
    main()
