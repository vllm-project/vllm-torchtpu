#!/usr/bin/env python3
"""q38_gdn_wt_aot.py (DEC-FB-2, GDNWT re-run pre-lock): deviceless TPU7x compile of the window_tile kernel at every
on-device gate case (test_gdn_v3_window_tile.TPU_CASES: TP8 tiles 1-4 and its served dynamic tile, TP2 tile 1 and its
served dynamic tile, the 1664-token pages, the PR's BF16 case), every step of each schedule, flag ON.

No TPU: jax.experimental.topologies gives a tpu7x:2x2x1 device and lowered.compile() runs the real TPU compiler, Mosaic
included, so a construct Mosaic rejects (T44 GDNWT j-bae75c90: "Reshape 1D ref is not supported" at TP8) fails here.
The x86 interpreter and the CPU suite's view shim cannot see it.

  q38_gdn_wt_aot.py <tree> [case-id ...]      driver: one python process per case (bounded RSS), summary line
  q38_gdn_wt_aot.py <tree> --child <case-id>  one case: one Q38_GDN_WT_AOT json line per step
Case ids are the test's (_id): e.g. tp8-verify_full-t2, tp2-verify_wide-tdyn. Exit 0 only if every step of every case
compiled with window_tile traced at the expected tile.
"""

import importlib.util
import json
import os
import resource
import subprocess
import sys
import time

tree = os.path.abspath(sys.argv[1])
args = sys.argv[2:]

def _versions():
    from importlib import metadata
    out = {}
    for d in ("jax", "jaxlib", "libtpu", "torch-tpu", "torch"):
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = None
    return out


if "--child" not in args:
    print("Q38_AOT_VERSIONS " + json.dumps(_versions()), flush=True)
    sys.path.insert(0, os.path.join(tree, "src"))
    sys.path.insert(0, tree)
    # Case ids without importing jax here (the child imports everything).
    ids = args or None
    if ids is None:
        r = subprocess.run([sys.executable, __file__, tree, "--child", "--list"], capture_output=True, text=True)
        ids = [l.split(" ", 1)[1] for l in r.stdout.splitlines() if l.startswith("CASE ")]
        if not ids:
            print(r.stdout[-2000:], r.stderr[-4000:])
            print("Q38_GDN_WT_AOT_SUMMARY " + json.dumps({"ok": False, "error": "no cases listed"}))
            sys.exit(2)
    results = []
    for cid in ids:
        t0 = time.time()
        r = subprocess.run([sys.executable, __file__, tree, "--child", cid], capture_output=True, text=True)
        lines = [l for l in r.stdout.splitlines() if l.startswith("Q38_GDN_WT_AOT ")]
        for l in lines:
            print(l, flush=True)
        ok = r.returncode == 0 and lines and all(json.loads(l.split(" ", 1)[1]).get("compiled") for l in lines)
        if not lines:
            print(f"Q38_GDN_WT_AOT_CHILD_ERROR {cid} rc={r.returncode}\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}", flush=True)
        results.append({"case": cid, "ok": bool(ok), "steps": len(lines), "rc": r.returncode, "s": round(time.time() - t0, 1)})
        print("Q38_GDN_WT_AOT_CASE " + json.dumps(results[-1]), flush=True)
    summ = {"ok": all(x["ok"] for x in results), "cases": len(results), "cases_ok": sum(x["ok"] for x in results),
            "steps": sum(x["steps"] for x in results)}
    print("Q38_GDN_WT_AOT_SUMMARY " + json.dumps(summ), flush=True)
    sys.exit(0 if summ["ok"] else 1)

# ------------------------------------------------------------------ child
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.pop("VLLM_KV_CACHE_LAYOUT", None)
os.environ.setdefault("Q38_GDN_WT_BACKEND", "cpu")
sys.path.insert(0, os.path.join(tree, "src"))
sys.path.insert(0, tree)

import jax  # noqa: E402
from jax._src import tpu_info as ti  # noqa: E402
from jax._src.pallas.mosaic import tpu_info as pti  # noqa: E402
from jax.experimental import topologies  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

INFO = ti.get_tpu_info_for_chip(ti.ChipVersion.TPU_7X, 1)
for mod in (pltpu, ti, pti):
    mod.get_tpu_info = lambda: INFO
for mod in (ti, pti):
    mod.is_tpu_device = lambda *a, **k: True

spec = importlib.util.spec_from_file_location("q38_wt_tests", os.path.join(tree, "tests", "kernels", "test_gdn_v3_window_tile.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.gdn.v3 import window_tile, wrapper  # noqa: E402

CASES = {T._id(c): c for c in T.TPU_CASES}
if "--list" in args:
    for cid in CASES:
        print("CASE " + cid)
    sys.exit(0)
cid = args[args.index("--child") + 1]
geo, schedule, tile = CASES[cid]

topo = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1")
sh = jax.sharding.SingleDeviceSharding(topo.devices[0])

traced = []
_run = window_tile.run


def recorded(*a, **k):
    traced.append(k["cfg"])
    return _run(*a, **k)


window_tile.run = recorded
envs.TPU_ENABLE_GDN_DYNAMIC_TILING = tile == "dyn"
tile_arg = None if tile == "dyn" else tile
want_tile = {"tp8": 8, "tp2": 2}.get(geo.name) if tile == "dyn" else tile

seqs, steps = T.SCHEDULES[schedule]
weights = T.make_weights(geo, 1, bias=geo.name == "tp8")      # as run_schedule (seed 0)
pool = T.make_pool(geo, 2, 1 + seqs * T.W + 1)
plan = T.plan_for(geo, pool)
ok_all = True
for i, stepspec in enumerate(steps):
    step = T.make_step(geo, 10 + i, 3, **stepspec)
    arr_keys = [k for k, v in step.items() if isinstance(v, jax.Array)]
    w_keys = [k for k, v in weights.items() if v is not None]
    static = {k: v for k, v in step.items() if k not in arr_keys}
    nones = {k: None for k, v in weights.items() if v is None}

    def f(pool_, *arrs, _ak=tuple(arr_keys), _wk=tuple(w_keys)):
        kw = dict(zip(_ak, arrs[: len(_ak)]))
        kw.update(zip(_wk, arrs[len(_ak):]))
        extra = {} if tile_arg is None else {"decode_tile_size": tile_arg}
        return wrapper.fused_conv1d_gdn(
            conv_state=None, recurrent_state=None, state_source=pool_, state_plan=plan,
            n_kq=geo.n_kq, n_v=geo.n_v, d_k=T.D, d_v=T.D, kernel_size=T.KERNEL_SIZE, window_tile_mode=True,
            **kw, **static, **nones, **extra,
        )

    specs = [jax.ShapeDtypeStruct(pool.shape, pool.dtype, sharding=sh)]
    specs += [jax.ShapeDtypeStruct(step[k].shape, step[k].dtype, sharding=sh) for k in arr_keys]
    specs += [jax.ShapeDtypeStruct(weights[k].shape, weights[k].dtype, sharding=sh) for k in w_keys]
    row = {"case": cid, "step": i, "geo": geo.name, "dim": geo.dim, "bias": weights["conv_bias"] is not None,
           "tile": tile, "num_tokens": int(step["qkv"].shape[0])}
    jax.clear_caches()   # the inner jitted wrapper would reuse step 0's trace at equal shapes (no window_tile.run call)
    before = len(traced)
    t0 = time.time()
    try:
        lowered = jax.jit(f).trace(*specs).lower()
        cfgs = traced[before:]
        row["window_tile_traced"] = bool(cfgs) and all(c.window_tile for c in cfgs)
        row["tiles"] = sorted({c.seq_tile_size for c in cfgs})
        compiled = lowered.compile()
        ma = compiled.memory_analysis()
        row["compiled"] = True
        row["temp_bytes"] = getattr(ma, "temp_size_in_bytes", None)
    except Exception as e:  # noqa: BLE001
        row["compiled"] = False
        row["error"] = (type(e).__name__ + ": " + str(e))[:1200]
    row["s"] = round(time.time() - t0, 1)
    row["maxrss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    if row.get("compiled") and not (row.get("window_tile_traced") and row.get("tiles") == [want_tile]):
        row["compiled"] = False
        row["error"] = f"window_tile not traced at the expected tile {want_tile}: traced={row.get('window_tile_traced')} tiles={row.get('tiles')}"
    ok_all &= bool(row["compiled"])
    print("Q38_GDN_WT_AOT " + json.dumps(row), flush=True)
sys.exit(0 if ok_all else 1)
