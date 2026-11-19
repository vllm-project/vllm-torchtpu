# SPDX-License-Identifier: Apache-2.0
"""DEC-COLL DCB1: O1's DCP combine on one host's 8 TensorCores (4 v7x chips = 2 KV-head groups x DCP4), today's
two all-gathers + slice against head-slice all-to-alls (TPU), or a deviceless compile of every program (--aot).

Per device: partial attention ``out`` [T, 12, 256] f32 and ``lse`` [T, 12] f32 over the DCP group's 12 heads, and
this rank's gate [T, 3, 256] bf16 (O1: TP8, DCP4, H = 3 heads per rank, D = 256). Mesh ("kv", "dcp") = (2, 4) in the
engine's group order: devices {0,1,4,5} and {6,7,2,3} (C: the served HLO's replica groups).

Arms (all inside shard_map; the merge is ``qsa_rpa.dcp_combine`` / ``dcp_combine_heads``, the engine's code):
  ag    today: all-gather out, all-gather lse, slice this rank's heads, merge (qsa_op._dcp_finish)
  agp   one all-gather of the packed [T, 12*256 + 128] (output + LSE), slice, merge (24 -> 12 collectives only)
  a2ax  pack head slices + LSE, XLA all-to-all over the DCP axis, merge (1/4 of the bytes)
  a2ad  the same with the Pallas remote-DMA all-to-all (kernels/collectives/dcp_head_a2a.py)
  rs    reference, NOT bitwise: all-gather lse, scale own partials, reduce-scatter over heads, divide (the vLLM
        cp_lse_ag_out_rs pattern)
Rows T = 64 / 128 / 192 / 256 (the decode row buckets: B16 / B32 / B48 / B64 at MTP k=3).

Timing: a sample is one jitted program chaining ``--calls`` combines (each call's inputs pass through an
optimization_barrier with the previous call's output, so calls run in order and are not CSE'd), wall time / calls;
a second program of ``--calls-short`` gives the launch-free estimate. Arms interleave with a rotating start.
Exactness, inline: every arm's per-device output on nrm and adv inputs is compared bit for bit with ag's; a positive
control (one partial element moved) must change exactly the owner rank's one output element on every bitwise arm.
Engagement: compiled HLO collective counts per arm. Optional --trace-dir: a jax.profiler trace of 3 samples per arm.
Result: run01_A/result.json with "ok" and "cases" (keyed by T), rewritten after every T (ok false until the last).
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

import numpy as np

KV, SIZE, H, D = 2, 4, 3, 256
ENGINE_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)
ARMS = ("ag", "agp", "a2ax", "a2ad", "rs")
BITWISE = ("agp", "a2ax", "a2ad")


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


def make_mesh(devs):
    import jax
    by_id = {d.id: d for d in devs}
    return jax.sharding.Mesh(np.asarray([by_id[i] for i in ENGINE_ORDER]).reshape(KV, SIZE), ("kv", "dcp"))


def one_call(arm, interpret=False):
    """(out [T,12,D] f32, lse [T,12] f32, gate [T,H,D] bf16) -> [T,H,D] bf16, per device inside shard_map."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from vllm_torchtpu.kernels.collectives import dcp_head_a2a as A
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import bf16, dcp_combine, dcp_combine_heads

    def rank():
        return lax.axis_index("dcp").reshape(1).astype(jnp.int32)

    if arm == "ag":
        def f(out, lse, gate):
            return dcp_combine(lax.all_gather(out, "dcp"), lax.all_gather(lse, "dcp"), gate, rank())
    elif arm == "agp":
        def f(out, lse, gate):
            t = out.shape[0]
            packed = jnp.concatenate([out.reshape(t, SIZE * H * D), jnp.pad(lse, ((0, 0), (0, 128 - SIZE * H)))], axis=-1)
            g = lax.all_gather(packed, "dcp")
            return dcp_combine(g[:, :, : SIZE * H * D].reshape(SIZE, t, SIZE * H, D),
                               g[:, :, SIZE * H * D: SIZE * H * D + SIZE * H], gate, rank())
    elif arm in ("a2ax", "a2ad"):
        def f(out, lse, gate):
            x = A.pack_head_slices(out, lse, SIZE)
            recv = A.xla_head_a2a_local(x, axis="dcp") if arm == "a2ax" else \
                A.dcp_head_a2a_local(x, axis="dcp", size=SIZE, interpret=interpret)
            o, s = A.unpack_head_slices(recv, H, D)
            return dcp_combine_heads(o, s, gate)
    elif arm == "rs":
        def f(out, lse, gate):
            t = out.shape[0]
            lse_all = lax.all_gather(lse, "dcp")  # [size, T, 12]
            top = jnp.max(lse_all, axis=0)
            finite = jnp.isfinite(top)
            safe = jnp.where(finite, top, 0)
            w_own = jnp.where(finite, jnp.exp(lse - safe), 0)
            total = jnp.sum(jnp.where(finite[None], jnp.exp(lse_all - safe[None]), 0), axis=0)
            contrib = (w_own[..., None] * out).reshape(t, SIZE, H, D).transpose(1, 0, 2, 3)
            summed = lax.psum_scatter(contrib, "dcp", scatter_dimension=0, tiled=False)  # [T, H, D]
            start = rank().reshape(()) * H
            tot = lax.dynamic_slice_in_dim(total, start, H, axis=1)
            fin = lax.dynamic_slice_in_dim(finite, start, H, axis=1)
            o = summed / jnp.where(tot > 0, tot, 1)[..., None]
            gated = bf16(bf16(o).astype(jnp.float32) * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32))
            return jnp.where(fin[..., None], gated, 0).astype(jnp.bfloat16)
    else:
        raise ValueError(arm)
    return f


def build_program(arm, mesh, calls, interpret=False):
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P
    f = one_call(arm, interpret)
    spec = P("kv", "dcp")

    def body(out, lse, gate):
        out, lse, gate = out[0, 0], lse[0, 0], gate[0, 0]
        if calls == 1:
            return f(out, lse, gate)[None, None]

        def step(i, prev):
            o, s, g, _ = lax.optimization_barrier((out, lse, gate, prev))
            return f(o, s, g)
        y = lax.fori_loop(0, calls, step, jnp.zeros(gate.shape, jnp.bfloat16))
        return y[None, None]

    return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False))


def abstract_args(mesh, tokens):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    sh = NamedSharding(mesh, P("kv", "dcp"))
    return (jax.ShapeDtypeStruct((KV, SIZE, tokens, SIZE * H, D), jnp.float32, sharding=sh),
            jax.ShapeDtypeStruct((KV, SIZE, tokens, SIZE * H), jnp.float32, sharding=sh),
            jax.ShapeDtypeStruct((KV, SIZE, tokens, H, D), jnp.bfloat16, sharding=sh))


def make_inputs(tokens, seed):
    rng = np.random.default_rng(seed)
    out = rng.standard_normal((KV, SIZE, tokens, SIZE * H, D)).astype(np.float32)
    lse = (rng.standard_normal((KV, SIZE, tokens, SIZE * H)) * 4).astype(np.float32)
    lse[:, 3, 1::5, :] = -np.inf  # a member owns no key of these rows
    gate = rng.standard_normal((KV, SIZE, tokens, H, D)).astype(np.float32)
    nrm = (out, lse, gate)
    out2, lse2 = out.copy(), lse.copy()
    lse2[:, :, 2::7, 5] = -np.inf  # nobody owns a key -> output 0
    lse2[:, :, 3::11, :] = 80.0  # ties at a large LSE
    lse2[:, 1, 4::13, :] = -80.0  # a large spread
    out2[:, 0, 6::17] *= 1e30
    out2[:, 2, 6::17] *= 1e-30
    out2[:, :, 8::19, :, ::2] = -0.0
    adv = (out2, lse2, gate)
    pos = out.copy()
    pos[0, 2, 5, 7, 11] += 64.0  # group 0, member 2, row 5, head 7 (owned by member 2), lane 11
    return {"nrm": nrm, "adv": adv, "posctl": (pos, lse, gate)}


def put(mesh, x):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    sh = NamedSharding(mesh, P("kv", "dcp"))
    out, lse, gate = x
    return (jax.device_put(jnp.asarray(out), sh), jax.device_put(jnp.asarray(lse), sh),
            jax.device_put(jnp.asarray(gate, jnp.bfloat16), sh))


def fetch(y):
    import jax
    return np.asarray(jax.device_get(y)).view(np.uint16)  # bf16 bits [KV, SIZE, T, H, D]


def hlo_counts(txt):
    return {"all-gather": len(re.findall(r"\ball-gather(?:-start)?\(", txt)),
            "all-to-all": len(re.findall(r"\ball-to-all\(", txt)),
            "reduce-scatter": len(re.findall(r"\breduce-scatter(?:-start)?\(", txt)),
            "all-reduce": len(re.findall(r"\ball-reduce(?:-start)?\(", txt)),
            # the Pallas call's name, not the module path (which also appears in op metadata)
            "dcp_head_a2a": len(re.findall(r"dcp_head_a2a-n_\d", txt)),
            "while": len(re.findall(r"\bwhile\(", txt))}


EXPECT = {  # per call in the body: (all-gather, all-to-all, reduce-scatter, kernel)
    "ag": (2, 0, 0, 0), "agp": (1, 0, 0, 0), "a2ax": (0, 1, 0, 0), "a2ad": (0, 0, 0, 1), "rs": (1, 0, 1, 0)}


def engaged(arm, cnt):
    ag, a2a, rs, k = EXPECT[arm]
    return (cnt["all-gather"] >= 1) == (ag > 0) and (cnt["all-to-all"] >= 1) == (a2a > 0) \
        and (cnt["reduce-scatter"] >= 1) == (rs > 0) and (cnt["dcp_head_a2a"] >= 1) == (k > 0) and cnt["all-reduce"] == 0


def aot_check(tokens_list, arms, calls):
    _stub_vllm_logger()
    import jax
    from jax.experimental import topologies
    devs = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x:2x2x1").devices
    mesh = make_mesh(devs)
    print(f"AOT topology tpu7x:2x2x1 devices {[(d.id, tuple(d.coords), d.core_on_chip) for d in devs]}", flush=True)
    bad = 0
    for t in tokens_list:
        for arm in arms:
            for nc in (1, calls):
                t0 = time.time()
                try:
                    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                        comp = build_program(arm, mesh, nc).lower(*abstract_args(mesh, t)).compile()
                    cnt = hlo_counts(comp.as_text())
                    ok = engaged(arm, cnt) and (cnt["while"] > 0 or nc == 1)
                    m = comp.memory_analysis()
                    bad += not ok
                    print(f"AOT T {t:3d} {arm:4s} calls {nc:3d}: COMPILED{'' if ok else ' BUT HLO CHECK FAILED'} "
                          f"arg {m.argument_size_in_bytes / 2**20:.2f} MiB temp {m.temp_size_in_bytes / 2**20:.2f} MiB "
                          f"hlo {cnt} [{time.time() - t0:.1f} s]", flush=True)
                except Exception as e:  # noqa: BLE001
                    bad += 1
                    print(f"AOT T {t} {arm} calls {nc}: FAILED {str(e)[:3000]}", flush=True)
    print("AOT PASS" if not bad else f"AOT FAIL ({bad})", flush=True)
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="64,128,192,256")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--calls", type=int, default=48)
    ap.add_argument("--calls-short", type=int, default=8)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--trace-dir")
    ap.add_argument("--aot", action="store_true")
    ap.add_argument("--interpret", action="store_true", help="CPU harness check only (a2ad in the Pallas interpreter)")
    a = ap.parse_args(argv)
    arms = a.arms.split(",")
    tokens_list = [int(v) for v in a.rows.split(",")]
    assert all(arm in ARMS for arm in arms) and a.calls > a.calls_short >= 1
    if a.aot:
        return aot_check(tokens_list, arms, a.calls)
    assert a.out and arms[0] == "ag"
    _stub_vllm_logger()
    import jax
    devs = sorted(jax.devices(), key=lambda d: d.id)
    assert len(devs) == KV * SIZE, f"need {KV * SIZE} devices, got {len(devs)}"
    mesh = make_mesh(devs)
    host = {"node": platform.node(), "jax": jax.__version__, "devices": len(devs), "kind": devs[0].device_kind,
            "coords": [(d.id, tuple(getattr(d, "coords", ())), getattr(d, "core_on_chip", None)) for d in devs],
            "mesh_order": ENGINE_ORDER, "interpret": a.interpret, "seed": a.seed, "calls": a.calls,
            "calls_short": a.calls_short, "iters": a.iters, "argv": sys.argv}
    print(f"DCB1 host {host}", flush=True)
    res = {"host": host, "cases": {}, "ok": False}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    def dump():
        with open(a.out + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(a.out + ".tmp", a.out)

    for t in tokens_list:
        x = make_inputs(t, a.seed * 1000 + t)
        progs, comp_s, outs, cnts = {}, {}, {}, {}
        for arm in arms:
            t0 = time.time()
            args = {k: put(mesh, v) for k, v in x.items()}
            c1 = build_program(arm, mesh, 1, a.interpret).lower(*args["nrm"]).compile()
            cl = build_program(arm, mesh, a.calls, a.interpret).lower(*args["nrm"]).compile()
            cs = build_program(arm, mesh, a.calls_short, a.interpret).lower(*args["nrm"]).compile()
            comp_s[arm] = round(time.time() - t0, 1)
            cnts[arm] = hlo_counts(cl.as_text())
            outs[arm] = {k: fetch(c1(*args[k])) for k in ("nrm", "adv", "posctl")}
            outs[arm]["chain"] = fetch(cl(*args["nrm"]))
            progs[arm] = (cl, cs, args["nrm"])
        rep = {}
        for arm in arms:
            r = {"hlo": cnts[arm], "engaged": engaged(arm, cnts[arm])}
            for k in ("nrm", "adv"):
                r[f"mismatch_vs_ag_{k}"] = int((outs[arm][k] != outs["ag"][k]).sum())
            r["mismatch_vs_ag_chain"] = int((outs[arm]["chain"] != outs["ag"]["chain"]).sum())
            r["chain_equals_single"] = bool(np.array_equal(outs[arm]["chain"], outs[arm]["nrm"]))
            moved = outs[arm]["posctl"] != outs[arm]["nrm"]
            where = np.argwhere(moved).tolist()
            r["posctl_changed"] = int(moved.sum())
            r["posctl_ok"] = bool(where == [[0, 2, 5, 7 - 2 * H, 11]])
            ob = outs[arm]["nrm"].view(np.uint16).astype(np.uint32) << 16
            r["finite"] = bool(np.isfinite(ob.view(np.float32)).all())
            r["elements"] = int(outs[arm]["nrm"].size)
            r["bitwise_vs_ag"] = bool(r["mismatch_vs_ag_nrm"] == 0 and r["mismatch_vs_ag_adv"] == 0
                                      and r["mismatch_vs_ag_chain"] == 0)
            rep[arm] = r
            print(f"DCB1 T {t} {arm:4s}: compiled {comp_s[arm]} s; {json.dumps(r)}", flush=True)
        for arm in arms:
            for _ in range(3):
                jax.block_until_ready(progs[arm][0](*progs[arm][2]))
                jax.block_until_ready(progs[arm][1](*progs[arm][2]))
        long_s, short_s = {arm: [] for arm in arms}, {arm: [] for arm in arms}
        for it in range(a.iters):
            for arm in arms[it % len(arms):] + arms[:it % len(arms)]:
                cl, cs, arg = progs[arm]
                t0 = time.perf_counter()
                jax.block_until_ready(cl(*arg))
                long_s[arm].append((time.perf_counter() - t0) * 1e6)
                t0 = time.perf_counter()
                jax.block_until_ready(cs(*arg))
                short_s[arm].append((time.perf_counter() - t0) * 1e6)
        if a.trace_dir:
            for arm in arms:
                d = os.path.join(a.trace_dir, f"T{t}_{arm}")
                os.makedirs(d, exist_ok=True)
                with jax.profiler.trace(d):
                    for _ in range(3):
                        jax.block_until_ready(progs[arm][0](*progs[arm][2]))
        case = {}
        for arm in arms:
            per = sorted(v / a.calls for v in long_s[arm])
            med_l, med_s = statistics.median(long_s[arm]), statistics.median(short_s[arm])
            case[arm] = {"us_per_call_median": round(statistics.median(per), 3), "us_per_call_min": round(per[0], 3),
                         "us_per_call_p10": round(per[int(0.1 * (len(per) - 1))], 3),
                         "us_per_call_p90": round(per[int(0.9 * (len(per) - 1))], 3),
                         "us_per_call_diff": round((med_l - med_s) / (a.calls - a.calls_short), 3),
                         "samples_long_us": [round(v, 2) for v in long_s[arm]],
                         "samples_short_us": [round(v, 2) for v in short_s[arm]],
                         "compile_s": comp_s[arm], "exact": rep[arm]}
        base = case[arms[0]]["us_per_call_median"]
        for arm in arms:
            c = case[arm]
            c["saving_vs_ag_us"] = round(base - c["us_per_call_median"], 3)
            print(f"DCB1 T {t:3d} {arm:4s}: median {c['us_per_call_median']:7.2f} us/call (p10 {c['us_per_call_p10']:.2f}, "
                  f"p90 {c['us_per_call_p90']:.2f}, diff-est {c['us_per_call_diff']:.2f}) saving vs ag {c['saving_vs_ag_us']:.2f} "
                  f"bitwise {rep[arm]['bitwise_vs_ag']} posctl {rep[arm]['posctl_ok']} engaged {rep[arm]['engaged']}", flush=True)
        res["cases"][str(t)] = case
        dump()
    res["ok"] = len(res["cases"]) == len(tokens_list)
    dump()
    print(f"DCB1 DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
