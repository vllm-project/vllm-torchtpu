# SPDX-License-Identifier: Apache-2.0
"""q38 AR-FIX ARB1: the decode TP8 all-reduce of bf16 [rows, 2560] on one 4-chip v7x host (2x2x1, 8 TensorCores).

One process per libtpu configuration (libtpu parses LIBTPU_INIT_ARGS once, at client creation), chosen by --cfg and
applied to os.environ BEFORE jax is imported (T39's mistake: its XLA arm ran SparseCore-offloaded because the client
existed before vllm_torchtpu.env_override raised the offload threshold; M, trace rows64_xlae):

  tc        the engine's libtpu set: the pod's LIBTPU_INIT_ARGS + env_override's additions (SparseCore all-reduce /
            all-gather offload threshold 64 MiB, dynamic SMEM). Arms: xtc (lax.psum = the engine's TensorCore
            all-reduce), h8, h8x (the q38-arfix op body, tp8_ar_decode_fn.global_fn, engine mesh order).
  scdef     the pod's LIBTPU_INIT_ARGS only (libtpu's default offload threshold; T39's condition). Arm: xsc.
  schier16  + --xla_tpu_enable_sparse_core_hierarchical_all_reduce=true, offload min size 16 tokens x 2560 x 2 B.
  schier4   the same at 4 tokens.
  scon      + --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true (IX golden recipe), min size 16 tokens.

Per arm and row count: a chain of C calls (48 and 8, each call followed by a 1-op elementwise filler so consecutive
all-reduces are separated as in the decode forward), timed 60 times each, interleaved across arms; per-call cost =
(median long - median short) / 40 ("diff"); a jax.profiler trace of 3 long runs per arm for device op times.
Exactness (single call): every member equal; h8 bitwise its numpy order model, h8x bitwise the single-rounding f32
sum; ulp distances to the exact sum for every arm; a positive control (one planted element moves exactly that
output element on all members). Engagement from the compiled HLO text: kernel name and all-reduce counts, and
whether the all-reduce is an async (SparseCore-offloaded) call.

Writes --out (JSON) for the reader q38_arfix_arb1_read.py. Exit 0 when every case it was asked for ran.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
import warnings

import numpy as np

N, H = 8, 2560
ENGINE_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)
VMEM_64MIB = 67108864
CFGS = {
    "tc": {"arms": ("xtc", "h8", "h8x"),
           "add": [f"--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes={VMEM_64MIB}",
                   f"--xla_tpu_sparse_core_all_gather_offload_min_size_in_bytes={VMEM_64MIB}",
                   "--xla_tpu_use_dynamic_smem_negotiation=true"]},
    "scdef": {"arms": ("xsc",), "add": []},
    "schier16": {"arms": ("xsc",), "add": ["--xla_tpu_enable_sparse_core_hierarchical_all_reduce=true",
                                           f"--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes={16 * H * 2}"]},
    "schier4": {"arms": ("xsc",), "add": ["--xla_tpu_enable_sparse_core_hierarchical_all_reduce=true",
                                          f"--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes={4 * H * 2}"]},
    "scon": {"arms": ("xsc",), "add": ["--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true",
                                       f"--xla_tpu_sparse_core_all_reduce_offload_min_size_in_bytes={16 * H * 2}"]},
}
WIRE = {"h8": "bf16", "h8x": "f32"}
CHAIN_SCALE = -112


def apply_cfg(cfg: str) -> str:
    """Set LIBTPU_INIT_ARGS for this process (before jax is imported); return the final string."""
    base = os.environ.get("LIBTPU_INIT_ARGS", "")
    names = {a.split("=")[0] for a in CFGS[cfg]["add"]}
    kept = [t for t in base.split() if t.split("=")[0] not in names]
    final = " ".join(kept + CFGS[cfg]["add"]).strip()
    os.environ["LIBTPU_INIT_ARGS"] = final
    return final


def _stub_vllm_logger():
    import importlib.util
    import logging
    import types
    if importlib.util.find_spec("vllm") is not None:
        return
    vllm = types.ModuleType("vllm")
    vlog = types.ModuleType("vllm.logger")
    vlog.init_logger = logging.getLogger
    vllm.logger = vlog
    sys.modules["vllm"], sys.modules["vllm.logger"] = vllm, vlog


def build(arm, mesh, pairs, calls, filler=True, interpret=False):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.collectives import tp8_ar_decode_fn as fn

    if arm in ("xtc", "xsc"):
        def one(v):
            return lax.psum(v, "d")
    else:
        def one(v):
            return fn.local_body(v, axis="d", pairs=pairs, wire=WIRE[arm], max_rows_=256, interpret=interpret)

    def step(v):
        y = one(v)
        # Filler: one elementwise op between calls (the forward has the HC combine / norm there). A power-of-two
        # scale is exact on the chain's 2^-112-scaled inputs and cannot be simplified away.
        return y * jnp.asarray(0.5, y.dtype) if filler else y

    def body(a):
        v = a[0]
        y = step(v) if calls == 1 else lax.fori_loop(0, calls, lambda i, u: step(u), v)
        return y[None]

    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P("d"),), out_specs=P("d"), check_vma=False))


def hlo_counts(txt):
    return {"all-reduce": len(re.findall(r"\ball-reduce(?:-start)?\(", txt)),
            "all-reduce-async": len(re.findall(r"all-reduce[\w.]*(?:call-start|-start)\(|async-start\([^\n]*all-reduce", txt)),
            "h8": len(re.findall(r"tp8_hier_all_reduce_\w", txt)),
            "while": len(re.findall(r"\bwhile\(", txt))}


def engaged(arm, cnt):
    if arm in ("xtc", "xsc"):
        return cnt["all-reduce"] >= 1 and cnt["h8"] == 0
    return cnt["h8"] >= 1 and cnt["all-reduce"] == 0


def make_inputs(rows, seed):
    from vllm_torchtpu.kernels.collectives import tp8_hier_all_reduce as h8
    rng = np.random.default_rng(seed)
    nrm = h8._bf16_round(rng.standard_normal((N, rows, H)).astype(np.float32))
    sign = np.where(rng.random((N, rows, H)) < 0.5, -1.0, 1.0)
    chain = h8._bf16_round(np.ldexp(sign * (1.0 + rng.random((N, rows, H))), CHAIN_SCALE).astype(np.float32))
    pos = nrm.copy()
    pos[5, min(7, rows - 1), 1000] += 64.0
    return {"nrm": nrm, "chain": chain, "posctl": pos}


def _bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def exactness(arm, outs, x, pairs):
    from vllm_torchtpu.kernels.collectives import tp8_hier_all_reduce as h8
    o = outs["nrm"]
    exact = h8._bf16_round(np.asarray(x["nrm"], np.float64).sum(0).astype(np.float32))
    d = h8.ulp_distance(o[0], exact)
    rep = {"ranks_equal": bool(all(np.array_equal(_bits(o[0]), _bits(o[i])) for i in range(N))),
           "finite": bool(np.isfinite(o).all()), "frac_le1ulp_vs_exact": float((d <= 1).mean()),
           "frac_le2ulp_vs_exact": float((d <= 2).mean()), "max_ulp_vs_exact": int(d.max())}
    if arm in WIRE:
        rep["mismatch_vs_model"] = int((_bits(o[0]) != _bits(h8.reference(x["nrm"], pairs, WIRE[arm]))).sum())
    moved = _bits(outs["posctl"]) != _bits(o)
    r = min(7, o.shape[1] - 1)
    rep["posctl_changed"] = int(moved.sum())
    rep["posctl_ok"] = bool(moved[:, r, 1000].all() and int(moved.sum()) == N)
    rep["ok"] = bool(rep["ranks_equal"] and rep["finite"] and rep["posctl_ok"]
                     and rep.get("mismatch_vs_model", 0) == 0)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, choices=sorted(CFGS))
    ap.add_argument("--rows", default="16,32,48,64,128,256")
    ap.add_argument("--calls", type=int, default=48)
    ap.add_argument("--calls-short", type=int, default=8)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace-dir")
    ap.add_argument("--aot", action="store_true", help="deviceless tpu7x:2x2x1 compile of every program, no timing")
    ap.add_argument("--interpret", action="store_true", help="CPU harness check only (8 host devices)")
    a = ap.parse_args(argv)
    inherited = os.environ.get("LIBTPU_INIT_ARGS", "")
    final = apply_cfg(a.cfg)
    _stub_vllm_logger()
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.collectives import tp8_ar_decode_fn as fn

    arms = CFGS[a.cfg]["arms"]
    rows_list = [int(v) for v in a.rows.split(",")]
    if a.aot:
        from jax.experimental import topologies
        devs = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices
    else:
        devs = jax.devices()
    by_id = {d.id: d for d in devs}
    assert len(by_id) >= N, f"need {N} devices, got {len(by_id)}"
    mesh = Mesh(np.array([by_id[i] for i in ENGINE_ORDER]), ("d",))
    pairs = fn.pairs_from_device_ids(ENGINE_ORDER)
    res = {"host": {"node": platform.node(), "jax": jax.__version__, "kind": devs[0].device_kind, "cfg": a.cfg,
                    "libtpu_inherited": inherited, "libtpu_final": final, "pairs": pairs, "argv": sys.argv,
                    "aot": a.aot, "interpret": a.interpret,
                    "coords": [(d.id, tuple(getattr(d, "coords", ())), getattr(d, "core_on_chip", None)) for d in devs]},
           "cases": {}, "ok": False}
    print(f"ARB1 cfg={a.cfg} LIBTPU_INIT_ARGS={final}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def dump():
        with open(a.out + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(a.out + ".tmp", a.out)

    if a.aot:
        bad = 0
        for rows in rows_list:
            case = {}
            for arm in arms:
                arg = jax.ShapeDtypeStruct((N, rows, H), jnp.bfloat16, sharding=NamedSharding(mesh, P("d")))
                try:
                    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                        txt = build(arm, mesh, pairs, a.calls).lower(arg).compile().as_text()
                    cnt = hlo_counts(txt)
                    case[arm] = {"compiled": True, "hlo": cnt, "engaged": engaged(arm, cnt),
                                 "ar_lines": [l.strip()[:300] for l in txt.splitlines() if "all-reduce" in l][:4]}
                    bad += not case[arm]["engaged"]
                except Exception as e:  # noqa: BLE001
                    bad += 1
                    case[arm] = {"compiled": False, "error": str(e)[:2000]}
                print(f"ARB1 AOT cfg={a.cfg} rows {rows} {arm}: {json.dumps(case[arm])[:600]}", flush=True)
            res["cases"][str(rows)] = case
        res["ok"] = bad == 0
        dump()
        print("ARB1 AOT", "PASS" if not bad else f"FAIL ({bad})", flush=True)
        return 0 if not bad else 1

    for rows in rows_list:
        x = make_inputs(rows, a.seed * 1000 + rows)
        progs, rep = {}, {}
        for arm in arms:
            t0 = time.time()
            args = {k: jax.device_put(jnp.asarray(v, jnp.bfloat16), NamedSharding(mesh, P("d"))) for k, v in x.items()}
            c1 = build(arm, mesh, pairs, 1, filler=False, interpret=a.interpret).lower(args["nrm"]).compile()
            cl = build(arm, mesh, pairs, a.calls, interpret=a.interpret).lower(args["chain"]).compile()
            cs = build(arm, mesh, pairs, a.calls_short, interpret=a.interpret).lower(args["chain"]).compile()
            cnt = hlo_counts(cl.as_text())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outs = {k: np.asarray(jax.device_get(c1(args[k]))).astype(np.float32) for k in ("nrm", "posctl")}
            rep[arm] = exactness(arm, outs, x, pairs)
            rep[arm].update({"hlo": cnt, "engaged": engaged(arm, cnt), "compile_s": round(time.time() - t0, 1)})
            progs[arm] = (cl, cs, args["chain"])
            print(f"ARB1 cfg={a.cfg} rows {rows} {arm}: {json.dumps(rep[arm])}", flush=True)
        for arm in arms:
            for _ in range(3):
                jax.block_until_ready(progs[arm][0](progs[arm][2]))
                jax.block_until_ready(progs[arm][1](progs[arm][2]))
        long_s, short_s = {arm: [] for arm in arms}, {arm: [] for arm in arms}
        for it in range(a.iters):
            for arm in arms[it % len(arms):] + arms[:it % len(arms)]:
                cl, cs, arg = progs[arm]
                t0 = time.perf_counter()
                jax.block_until_ready(cl(arg))
                long_s[arm].append((time.perf_counter() - t0) * 1e6)
                t0 = time.perf_counter()
                jax.block_until_ready(cs(arg))
                short_s[arm].append((time.perf_counter() - t0) * 1e6)
        if a.trace_dir:
            for arm in arms:
                d = os.path.join(a.trace_dir, f"{a.cfg}_rows{rows}_{arm}")
                os.makedirs(d, exist_ok=True)
                with jax.profiler.trace(d):
                    for _ in range(3):
                        jax.block_until_ready(progs[arm][0](progs[arm][2]))
        case = {}
        for arm in arms:
            ml, ms = statistics.median(long_s[arm]), statistics.median(short_s[arm])
            per = sorted(v / a.calls for v in long_s[arm])
            case[arm] = {"us_per_call_diff": round((ml - ms) / (a.calls - a.calls_short), 3),
                         "us_per_call_median": round(statistics.median(per), 3),
                         "long_median_us": round(ml, 2), "short_median_us": round(ms, 2),
                         "samples_long_us": [round(v, 2) for v in long_s[arm]],
                         "samples_short_us": [round(v, 2) for v in short_s[arm]], "exact": rep[arm]}
            print(f"ARB1 cfg={a.cfg} rows {rows:3d} {arm:4s}: diff {case[arm]['us_per_call_diff']:7.2f} us/call "
                  f"(median/call {case[arm]['us_per_call_median']:.2f}) exact ok {rep[arm]['ok']} "
                  f"engaged {rep[arm]['engaged']} hlo {rep[arm]['hlo']}", flush=True)
        res["cases"][str(rows)] = case
        dump()
    res["ok"] = len(res["cases"]) == len(rows_list)
    dump()
    print(f"ARB1 DONE cfg={a.cfg} -> {a.out}", flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
