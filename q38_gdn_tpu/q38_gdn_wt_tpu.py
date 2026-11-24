#!/usr/bin/env python3
"""DEC-FB-2 GDN window_tile, on device (one TensorCore of a 4-chip v7x pod): the bitwise gate, its mutants, then timing.

The x86 CPU interpreter could not adjudicate bitwise for this kernel pair (CPU job j-c53ec814 failed bitwise; the frozen
secondary check passed); this job decides bitwise ON THE DEVICE (prereg reports/prereg/q38_decfb2_gdnwt_prereg_DRAFT.md).

  V1  pytest tests/kernels/test_gdn_v3_window_tile.py -k on_tpu with Q38_GDN_WT_BACKEND=tpu: window_tile vs inner_kernel,
      unpatched (no test aids), bit for bit on outputs and every pool byte, 15 cases: TP8 tiles 1-4 and the served
      dynamic tile (8), TP2 tile 1 and the served dynamic tile (2), TP2 on 1664-token pool pages, the PR's HND/BF16 case.
  V2  the same on-device test with a mutated copy of src (PYTHONPATH): each mutant must FAIL it (a mismatch on device):
        rec_ckpt_slot_offset    the recurrent checkpoint stored to the next window slot (TP8 tile 4, TP2 tile 1)
        tp8_pad_tap_store_off   the conv tap stored one row off in the 10 -> 16 row padding (TP8 tiles 2 and 3)
  T   only if V1 and V2 hold: GDN verify call us per layer, flag off vs on, interleaved, median of --samples, at O1 TP8
      B16 / 28 / 48 / 64 and O4 TP2 B7 / 12 / 16 (per-rank shapes, served 1664-token fp8 pool pages, dynamic tiling as
      served, K = 3), the pool donated.
Writes result.json (ok, v1, v2, timing, cases) in --out; ok = V1 and V2 hold and every timing case ran.
  PYTHONPATH=src python3 q38_gdn_tpu/q38_gdn_wt_tpu.py [--out DIR] [--samples 11] [--iters 20]
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
T = "tests/kernels/test_gdn_v3_window_tile.py"
WT = "src/vllm_torchtpu/kernels/gdn/v3/window_tile.py"
MUTANTS = [
    ("rec_ckpt_slot_offset",
     "                    vmem_ldst.store_state_head(rec_out_ref, cfg, idx, pos, head, updated)\n",
     "                    vmem_ldst.store_state_head(rec_out_ref, cfg, idx, min(pos + 1, cfg.window_size - 1), head, updated)\n",
     ["test_window_tile_bitwise_on_tpu[tp8-verify_partial-t4]", "test_window_tile_bitwise_on_tpu[tp2-verify_full-t1]"]),
    ("tp8_pad_tap_store_off",
     "            view[pl.ds(j * rows, rows)] = tap.astype(dtype)\n",
     "            view[pl.ds(j * (rows + 1), rows)] = tap.astype(dtype)\n",
     ["test_window_tile_bitwise_on_tpu[tp8-verify_full-t2]", "test_window_tile_bitwise_on_tpu[tp8-verify_partial-t3]"]),
]
TIMING = [("o1_tp8", b) for b in (16, 28, 48, 64)] + [("o4_tp2", b) for b in (7, 12, 16)]


EXTRA_PATH = []


def ensure_pytest():
    """The bench image may lack pytest: install it into a private dir (never into the image's site-packages)."""
    try:
        import pytest  # noqa: F401
        return "image"
    except ImportError:
        d = "/tmp/q38_gdn_pytest"
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", d, "pytest"], check=True)
        EXTRA_PATH.append(d)
        # timing() imports the test module in THIS process (its line 62 imports pytest); T71 j-d342e536 died there
        # with "ModuleNotFoundError: No module named 'pytest'" because the dir reached only the V1/V2 subprocesses.
        sys.path.insert(0, d)
        return d


def pytest_on_tpu(src_root, ids=None, timeout=3600):
    path = ":".join([f"{src_root}/src", f"{ROOT}/tests/kernels", *EXTRA_PATH])
    env = dict(os.environ, Q38_GDN_WT_BACKEND="tpu", PYTHONPATH=path, PYTHONDONTWRITEBYTECODE="1")
    env.pop("VLLM_TPU_GDN_WINDOW_TILE", None)
    sel = [f"{ROOT}/{T}::{i}" for i in ids] if ids else [f"{ROOT}/{T}", "-k", "on_tpu"]
    cmd = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest", "-rfE", *sel]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"rc": 124, "summary": f"HANG after {timeout} s", "s": round(time.time() - t0)}
    lines = r.stdout.strip().splitlines() or ["?"]
    return {"rc": r.returncode, "summary": lines[-1], "failed": [ln for ln in lines if ln.startswith("FAILED")][:20],
            "s": round(time.time() - t0), "tail": "\n".join(lines[-40:])}


LOWERING_ERRORS = ("NotImplementedError", "LoweringException", "MosaicError", "Mosaic failed", "INTERNAL: Mosaic")


def classify_failure(r):
    """'mismatch' only for a bitwise assertion failure (np.testing "Arrays are not equal") with no lowering or
    compile error in the output; 'lowering' if a Mosaic / lowering error appears; else 'other' / 'passed'."""
    tail = r.get("tail", "")
    if r.get("rc") == 0:
        return "passed"
    if any(e in tail for e in LOWERING_ERRORS):
        return "lowering"
    if "Arrays are not equal" in tail and r.get("failed"):
        return "mismatch"
    return "other"


def v2_mutant(name, old, new, ids):
    """GDNWT re-run (addendum A1): each listed case runs alone, and the mutant counts as detected only if EVERY case
    fails by a real bitwise mismatch -- a lowering error (T44: tp8_pad_tap_store_off was "caught" by the TP8 Mosaic
    rejection) does not count."""
    d = tempfile.mkdtemp(prefix="q38_gdn_tpu_mut_")
    try:
        shutil.copytree(os.path.join(ROOT, "src"), os.path.join(d, "src"), ignore=shutil.ignore_patterns("__pycache__"))
        p = os.path.join(d, WT)
        s = open(p).read()
        assert s.count(old) == 1, (name, s.count(old))
        open(p, "w").write(s.replace(old, new))
        per = {}
        for i in ids:
            ri = pytest_on_tpu(d, [i])
            per[i] = {"rc": ri["rc"], "summary": ri["summary"], "class": classify_failure(ri), "tail": ri.get("tail", "")[-1500:]}
        r = {"cases": per, "summary": "; ".join(f"{i.split('[')[-1].rstrip(']')}: {v['class']}" for i, v in per.items())}
        r["mismatch_detected"] = all(v["class"] == "mismatch" for v in per.values())
        r["rc"] = 1 if r["mismatch_detected"] else 0
        return r
    finally:
        shutil.rmtree(d, ignore_errors=True)


def timing(samples, iters):
    os.environ["Q38_GDN_WT_BACKEND"] = "tpu"
    os.environ["TPU_ENABLE_GDN_DYNAMIC_TILING"] = "1"
    sys.path.insert(0, os.path.join(ROOT, "tests", "kernels"))
    import jax
    import numpy as np
    import test_gdn_v3_window_tile as tw   # the geometries, pools, steps and plans of V1 (C: the same helpers)
    from vllm_torchtpu.kernels.gdn.v3 import wrapper
    geos = {"o1_tp8": tw.TP8_P1664, "o4_tp2": tw.TP2_P1664}
    out = []
    for name, b in TIMING:
        geo = geos[name]
        step = tw.make_step(geo, 100 + b, 3, [4] * b, (b, b, b))
        weights = tw.make_weights(geo, 200 + b)
        pool0 = tw.make_pool(geo, 300 + b, 1 + b * 4 + 8)
        plan = tw.plan_for(geo, pool0)
        fns = {}
        for wt in (False, True):
            def call(pool, qkv, wt=wt):
                res = wrapper.fused_conv1d_gdn(conv_state=None, recurrent_state=None, state_source=pool, state_plan=plan,
                                               n_kq=geo.n_kq, n_v=geo.n_v, d_k=tw.D, d_v=tw.D, kernel_size=tw.KERNEL_SIZE,
                                               window_tile_mode=wt, **{**step, "qkv": qkv}, **weights)
                return res[0], res[1]
            fns[wt] = jax.jit(call, donate_argnums=(0,))
        pools = {wt: jax.device_put(np.asarray(pool0)) for wt in (False, True)}
        for wt in (False, True):
            pools[wt], o = fns[wt](pools[wt], step["qkv"])
            jax.block_until_ready(o)
        xs = {False: [], True: []}
        for s in range(samples):
            for wt in ((False, True) if s % 2 == 0 else (True, False)):
                t0 = time.perf_counter()
                for _ in range(iters):
                    pools[wt], o = fns[wt](pools[wt], step["qkv"])
                jax.block_until_ready(o)
                xs[wt].append((time.perf_counter() - t0) / iters * 1e6)
        rec = {"shape": name, "B": b, "us_off": round(statistics.median(xs[False]), 2),
               "us_on": round(statistics.median(xs[True]), 2), "samples_off": [round(x, 2) for x in xs[False]],
               "samples_on": [round(x, 2) for x in xs[True]]}
        rec["saving_us"] = round(rec["us_off"] - rec["us_on"], 2)
        out.append(rec)
        print("Q38_GDN_WT_TIME " + json.dumps({k: rec[k] for k in ("shape", "B", "us_off", "us_on", "saving_us")}), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"), "q38f-bench",
                                               "script", "run01_A"))
    ap.add_argument("--samples", type=int, default=11)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = {"bench": "q38-decfb2-gdnwt", "ok": False, "cases": [], "v1": None, "v2": {}, "timing": None,
           "pytest": ensure_pytest()}
    res["v1"] = pytest_on_tpu(ROOT)
    v1 = res["v1"]["rc"] == 0 and " passed" in res["v1"]["summary"] and "failed" not in res["v1"]["summary"]
    print(f"Q38_GDN_WT_V1 rc={res['v1']['rc']} {res['v1']['summary']} -> {'PASS' if v1 else 'FAIL'}", flush=True)
    for name, old, new, ids in MUTANTS:
        res["v2"][name] = v2_mutant(name, old, new, ids)
        print(f"Q38_GDN_WT_V2 {name}: rc={res['v2'][name]['rc']} {res['v2'][name]['summary']} -> "
              f"{'MISMATCH DETECTED' if res['v2'][name]['mismatch_detected'] else 'NOT DETECTED'}", flush=True)
    v2 = all(r["mismatch_detected"] for r in res["v2"].values())
    res["v1_pass"], res["v2_pass"] = v1, v2
    if v1 and v2:
        res["timing"] = timing(a.samples, a.iters)
        res["cases"] = [{"shape": r["shape"], "B": r["B"], "us_off": r["us_off"], "us_on": r["us_on"]} for r in res["timing"]]
        res["ok"] = len(res["cases"]) == len(TIMING)
    else:
        res["cases"] = [{"v1_pass": v1, "v2_pass": v2}]
        print("Q38_GDN_WT timing skipped (V1 or V2 failed): the microbench reads NO VERDICT / INVALID per the prereg",
              flush=True)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print(f"Q38_GDN_WT DONE ok={res['ok']} v1={v1} v2={v2} -> {a.out}", flush=True)
    return 0 if res["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
