# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU tier of the accuracy instrument (multi-controller, all ranks; GLM-5.2 per-rank shapes).

Per cell (dep, m, kind, n_valid): the kernel(s) under test vs
  - the f32 HIGHEST oracle (dequantized weights, unquantized x, exact routing weights), psum over ranks:
    rel L2 <= 0.06, cos >= 0.998 (the branch tests' TOLERANCE);
  - the native reference (refs.native_partial per rank, bf16 partials summed in rank order as the all-reduce does):
    rel L2 <= 1e-3 (EXPECTED_REL_L2), bitwise fraction reported;
  - ranks identical, padding rows +0, no NaN, kernel name census.
Also, per cell: the row-gathered algorithm (refs.rg_partial) vs native (algebra on TPU XLA) and the fault controls
F1..F8 (catch = fails the native 1e-3 gate) with catch rate per routing kind; one f64 spot check of the f32 oracle
(rank 0 partial, 4 rows) on host.
Kernels: "F" = incumbent module; --plugin adds more (fn(name)); each non-F kernel also reports vs_incumbent (bitwise).
Run with XLA_FLAGS=--xla_allow_excess_precision=false so the jnp references round bf16 per op like the kernel; if the
native reference still misses the incumbent by > 1e-3 the fault catch rate prints VOID (a saturated metric is not a
catch). Output: ACC lines (process 0), ACC_CATCH, ACC_DONE.
"""

import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refs  # noqa: E402
import spec  # noqa: E402
import tpu_common as tc  # noqa: E402
from bench import load_plugin  # noqa: E402

from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as K  # noqa: E402

F32, BF16 = jnp.float32, jnp.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deps", default="tp16,ep16")
    ap.add_argument("--ms", default="64,112,192")
    ap.add_argument("--kinds", default="indep,mtp0.8,zipf0.6")
    ap.add_argument("--kernels", default="F")
    ap.add_argument("--plugin", default="")
    ap.add_argument("--faults", default="112,192", help="m values that also run the rg algebra + fault controls")
    ap.add_argument("--r-slab", type=int, default=16)
    a = ap.parse_args()
    pid = tc.init()
    say = (lambda *s: print(*s, flush=True)) if pid == 0 else (lambda *s: None)
    msh = tc.mesh()
    n = len(msh.devices.ravel())
    plug = load_plugin(a.plugin)
    fault_ms = {int(v) for v in a.faults.split(",") if v}
    PER, REP = tc.PER, tc.REP

    def per_rank(fn, out_spec=PER, **kw):
        def body(x, coef, e0, *w):
            return fn(x, coef, e0.reshape(()), *w, routed_scale=spec.ROUTED_SCALE, **kw)

        return jax.jit(jax.shard_map(body, mesh=msh, in_specs=(PER, REP, PER) + (PER,) * 8, out_specs=out_spec,
                                     check_vma=False))

    def ar_sum(partials_global, m):
        """bf16 partials [n*m, h] -> rank-order f32 sum -> bf16 [m, h] (replicated)."""
        def body(p):
            g = jax.lax.all_gather(p, "d")  # [n, m, h]
            acc = g[0].astype(F32)
            for r in range(1, n):
                acc = acc + g[r].astype(F32)
            return acc.astype(BF16)

        return jax.jit(jax.shard_map(body, mesh=msh, in_specs=(PER,), out_specs=REP, check_vma=False))(partials_global)

    def psum_f32(partials_global):
        def body(p):
            return jax.lax.psum(p, "d")

        return jax.jit(jax.shard_map(body, mesh=msh, in_specs=(PER,), out_specs=REP, check_vma=False))(partials_global)

    def ranks_check(out_global):
        def body(o):
            g = jax.lax.all_gather(o, "d")
            d = jnp.max(jnp.abs(g.astype(F32) - g[0:1].astype(F32)))
            return jax.lax.pmax(d, "d").reshape(1)

        # Replicated, so every process can read it (only process 0 holds rank 0's shard).
        return jax.jit(jax.shard_map(body, mesh=msh, in_specs=(PER,), out_specs=REP, check_vma=False))(out_global)

    def host(a_):
        return np.asarray(jax.device_get(a_)) if a_.is_fully_addressable else tc.local_shard(a_)

    catch = {}
    for dep in [spec.DEPS[d] for d in a.deps.split(",")]:
        w = tc.rank_weights_global(msh, dep, seed=11)
        progs = {}
        for name in a.kernels.split(","):
            fn = K.moe_tp_decode if name == "F" else plug.fn(name)
            progs[name] = jax.jit(jax.shard_map(tc.kernel_call(fn, dep, n), mesh=msh, in_specs=tc.KERNEL_SPECS,
                                                out_specs=PER, check_vma=False))
        for m in [int(v) for v in a.ms.split(",")]:
            for kind in a.kinds.split(","):
                for n_valid in (m, m - 13):
                    t0 = time.time()
                    kargs, (x, scores, bias, p) = tc.cell_inputs(msh, dep, m, kind, n_valid)
                    valid = (np.arange(m) < n_valid)[:, None]
                    coef = np.where(valid, np.asarray(refs.routing_coef(jnp.asarray(scores), jnp.asarray(bias), spec.TOPK)), 0)
                    coefx = np.where(valid, np.asarray(refs.routing_coef(jnp.asarray(scores), jnp.asarray(bias), spec.TOPK, exact=True)), 0)
                    cg = tc.global_from_host(msh, coef.astype(np.float32), REP)
                    cxg = tc.global_from_host(msh, coefx.astype(np.float32), REP)
                    xg, e0g = kargs[0], kargs[5]
                    nat = ar_sum(per_rank(refs.native_partial)(xg, cg, e0g, *w), m)
                    natv = np.where(valid, tc.as_f32(host(nat)), 0.0)
                    orc_parts = per_rank(refs.oracle_partial_f32)(xg, cxg, e0g, *w)
                    orc = tc.as_f32(host(psum_f32(orc_parts)))
                    rec = dict(dep=dep.name, m=m, kind=kind, n_valid=n_valid, drivers=spec.drivers(p, n_valid, dep, a.r_slab))
                    nat_vs_orc = refs.compare(natv, orc, n_valid)
                    rec["native_vs_oracle"] = nat_vs_orc
                    f_out = None
                    # Calibration of the native reference (jnp under XLA) against the incumbent kernel: the native
                    # 1e-3 gate and the fault catch rates are meaningful only if this is itself <= 1e-3.
                    for name, prog in progs.items():
                        out = prog(*tc.kernel_args(kargs, w))
                        txt = ""
                        rd = float(np.max(host(ranks_check(out))))
                        if pid != 0:  # rank 0's shard lives on process 0; host-side checks only there
                            continue
                        o0 = tc.local_shard(out, 0)
                        c_or = refs.compare(o0, orc, n_valid)
                        c_nat = refs.compare(o0, orc, n_valid, ref=natv.astype(np.float32))
                        rec[name] = dict(vs_oracle=c_or, vs_native=dict(bitwise=c_nat["bitwise_vs_ref"],
                                                                      rel_l2=c_nat["rel_l2_vs_ref"]),
                                         ranks_max_diff=rd, passes=bool(refs.passes(c_or, c_nat) and rd == 0.0))
                        if name == "F":
                            f_out = o0
                        elif "F" in progs and f_out is not None:
                            vf = refs.compare(o0, orc, n_valid, ref=f_out)
                            rec[name]["vs_incumbent"] = dict(bitwise=vf["bitwise_vs_ref"], rel_l2=vf["rel_l2_vs_ref"])
                    if m in fault_ms and n_valid == m:
                        rg = ar_sum(per_rank(refs.rg_partial, r_slab=a.r_slab)(xg, cg, e0g, *w), m)
                        rgc = refs.compare(np.where(valid, tc.as_f32(host(rg)), 0), natv, n_valid, ref=natv.astype(np.float32))
                        rec["rg_algebra_vs_native"] = dict(bitwise=rgc["bitwise_vs_ref"], rel_l2=rgc["rel_l2_vs_ref"])
                        fl = {}
                        for f in refs.FAULTS:
                            if f == "F4" and not dep.ep:
                                continue
                            got = ar_sum(per_rank(refs.rg_partial, fault=f, r_slab=a.r_slab)(xg, cg, e0g, *w), m)
                            fc = refs.compare(np.where(valid, tc.as_f32(host(got)), 0), natv, n_valid, ref=natv.astype(np.float32))
                            hit = fc["rel_l2_vs_ref"] > refs.TOL_NATIVE or fc["nan"] > 0
                            if rgc["rel_l2"] > refs.TOL_NATIVE:  # the clean algorithm already fails: catch saturated
                                hit = None
                            fl[f] = dict(caught=hit if hit is None else bool(hit), rel_l2=fc["rel_l2_vs_ref"])
                            k = (f, dep.name, kind)
                            catch.setdefault(k, [0, 0])
                            if hit is not None:
                                catch[k][0] += int(hit)
                                catch[k][1] += 1
                        rec["faults"] = fl
                    if dep.name == "tp16" and m == 64 and kind == "indep" and n_valid == m and pid == 0:
                        rows = [0, 17, 40, 63]
                        w0 = [tc.local_shard(t, 0) for t in w]
                        part0 = tc.local_shard(orc_parts, 0)
                        ref64 = refs.oracle_partial_f64(x, coefx, 0, *[jnp.asarray(t) for t in w0],
                                                        routed_scale=spec.ROUTED_SCALE, rows=rows)
                        rec["oracle_f32_vs_f64"] = refs.compare(part0[rows], ref64, len(rows))
                    rec["seconds"] = round(time.time() - t0, 1)
                    say("ACC", json.dumps(rec))
    say("ACC_CATCH", json.dumps({"|".join(k): (f"{v[0]}/{v[1]}" if v[1] else "VOID: clean reference fails the gate")
                                 for k, v in sorted(catch.items())}))
    say("ACC_ENV", json.dumps(dict(XLA_FLAGS=os.environ.get("XLA_FLAGS", ""))))
    say("ACC_DONE")


if __name__ == "__main__":
    main()
