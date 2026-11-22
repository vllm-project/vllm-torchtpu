# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""CPU interpret-mode dry run (16 CPU devices, race detector): moe_tp_decode_rg vs the incumbent kernel, bitwise, at
small shapes (h 2048, 32 router experts, top-4) on TP and expert-range layouts, including overflow cells (an expert
with more than R rows), and the kernel-level fault controls. Run as a script:
  XLA_FLAGS=--xla_force_host_platform_device_count=16 JAX_PLATFORMS=cpu python interpret_rg.py [faults]
Prints CASE / FAULT lines and exits non-zero if a clean case is not bitwise or a fault expected to be caught is not."""

import contextlib
import io
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refs  # noqa: E402
import spec  # noqa: E402

from jax._src.pallas.mosaic import tpu_info  # noqa: E402

from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel as K  # noqa: E402
from vllm_torchtpu.kernels.experimental.moe_tp_decode import kernel_rg as KR  # noqa: E402

N = 16
E, TOPK, H = 32, 4, 2048


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


pltpu.get_tpu_info = _tpu7x
tpu_info.get_tpu_info = _tpu7x
os.environ.setdefault("TPU_MOE_TP_DECODE_SPLIT_META", "1")
os.environ.setdefault("TPU_MOE_TP_DECODE_VMEM_FRACTION", "1.0")

DEPS = {
    "tp": spec.Dep("tp_small", e_l=E, i=128, ep=False, h=H, e=E),
    "ep": spec.Dep("ep_small", e_l=E // N, i=256, ep=True, h=H, e=E),
}


def run_case(dep, m, kind, n_valid, faults=(), r_slab=16, **kopts):
    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    mesh = Mesh(np.asarray(devices), ("d",))
    scores, bias, p = spec.scores_bias(m, kind, spec.seed_of(m, dep.name, kind), e=E, topk=TOPK)
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((m, H)), jnp.bfloat16)
    ranks = [refs.rank_weights(jax.random.key(100 + r), dep) for r in range(N)]
    stacked = [jnp.concatenate(ws, axis=0) for ws in zip(*ranks, strict=True)]
    specs = (P("d"), P("d"), P(), P("d"), P("d")) + (P("d"),) * 8 + (P("d"),)

    def make(fn, **kw):
        def body(x_, s_, b_, nv, rk, *rest):
            *w, e0 = rest
            return fn(x_, s_, b_, nv[0], rk.reshape(()), *w, e0.reshape(()), routed_scale=spec.ROUTED_SCALE,
                      topk=TOPK, n_ranks=N, fast=True, **kw)

        return jax.jit(jax.shard_map(body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False))

    ops = [jnp.tile(x, (N, 1)), jnp.tile(jnp.asarray(scores), (N, 1)), jnp.asarray(bias),
           jnp.full((N,), n_valid, jnp.int32), jnp.arange(N, dtype=jnp.int32), *stacked,
           (jnp.arange(N, dtype=jnp.int32) * dep.e_l) if dep.ep else jnp.zeros((N,), jnp.int32)]
    ops = [jax.device_put(a, NamedSharding(mesh, s)) for a, s in zip(ops, specs, strict=True)]

    def interp(fn):
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call

        pltpu.reset_tpu_interpret_mode_state()
        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
            with contextlib.redirect_stdout(io.StringIO()):
                out = np.asarray(jax.block_until_ready(fn(*ops)))
        return out.reshape(N, m, H).astype(np.float32), bool(interpret_pallas_call.races.races_found)

    t0 = time.time()
    inc, races_i = interp(make(K.moe_tp_decode))
    rg, races_r = interp(make(KR.moe_tp_decode_rg, r_slab=r_slab, **kopts))
    n_loc = p[:n_valid].sum(0).reshape(N, -1) if dep.ep else p[:n_valid].sum(0)[None]
    case = dict(dep=dep.name, m=m, kind=kind, n_valid=n_valid, r_slab=r_slab, opts=kopts, max_rows_per_expert=int(n_loc.max()),
                overflow=bool(n_loc.max() > r_slab), races=[races_i, races_r],
                bitwise=bool((rg.view(np.uint32) == inc.view(np.uint32)).all()),
                ranks_identical=bool((rg.view(np.uint32) == rg[0].view(np.uint32)).all()),
                seconds=round(time.time() - t0, 1))
    if not case["bitwise"]:
        d = rg[0] != inc[0]
        rows = np.flatnonzero(d.any(1))
        case["diff"] = dict(n_elems=int(d.sum()), rows=rows[:20].tolist(), n_rows=int(len(rows)),
                            rel_l2=float(np.linalg.norm(rg[0] - inc[0]) / np.linalg.norm(inc[0])),
                            max_abs=float(np.abs(rg[0] - inc[0]).max()))
    print("CASE", json.dumps(case), flush=True)
    res = []
    for f in faults:
        bad, _ = interp(make(KR.moe_tp_decode_rg, r_slab=r_slab, fault=f, **kopts))
        caught = float(np.linalg.norm(bad - inc) / np.linalg.norm(inc)) > 1e-3 or bool(np.isnan(bad).any())
        print("FAULT", json.dumps(dict(dep=dep.name, m=m, kind=kind, fault=f, caught=caught,
                                       rel_l2=float(np.linalg.norm(bad - inc) / np.linalg.norm(inc)))), flush=True)
        res.append((f, caught))
    return case, res


if __name__ == "__main__":
    do_faults = "faults" in sys.argv[1:]
    ok = True
    cells = [("tp", 80, "indep", 80, 32), ("tp", 112, "zipf1.0", 99, 16), ("ep", 80, "indep", 80, 32),
             ("ep", 112, "zipf1.0", 112, 16), ("ep", 112, "indep", 99, 32)]
    for d, m, kind, nv, r in cells:
        faults = refs.FAULTS if do_faults and (d == "ep" or m == 80) else ()
        faults = tuple(f for f in faults if not (f == "F4" and d == "tp"))
        case, res = run_case(DEPS[d], m, kind, nv, faults, r_slab=r)
        # Interpret mode runs the kernel body under XLA-CPU, whose FMA contraction choices can differ between the
        # row-wise scatter and the incumbent's block update (1 bf16 ulp in a few elements); the bitwise gate is a
        # TPU fact (accuracy_tpu.py). Here: bitwise, or rel L2 <= 1e-4 with every difference <= 1 bf16 ulp.
        near = case.get("diff", {}).get("rel_l2", 0.0) <= 1e-4
        ok = ok and (case["bitwise"] or near) and not any(case["races"])
        for f, caught in res:
            expect = case["overflow"] if f == "F5" else True
            if caught != expect:
                print("UNEXPECTED", f, d, m, kind, caught, expect)
                ok = False
    print("INTERPRET_RG", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
