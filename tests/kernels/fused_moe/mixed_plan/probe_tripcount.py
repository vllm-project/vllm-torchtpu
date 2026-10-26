# SPDX-License-Identifier: Apache-2.0
"""Design-time FACT probe 03: the cheapest way to get the unit loop's trip count.

Probe 02 F-a (HBM->SMEM DMA of the [16,384] count section + nested scalar
fori_loops over 16 sources x 24 local experts) lowered and was correct but cost
8.0 us per call. Arms (each then runs the same dummy unit loop):
  T1  whole exchange [16, W] as a scalar-prefetch operand, nested fori_loops (isolates loop cost from the DMA)
  T2  whole exchange as scalar prefetch, Python-unrolled straight-line sum (384 SMEM loads)
  T3  vector path: exchange in VMEM, masked lane sum of aligned counts -> jnp.sum -> scalar -> fori bound
  T4  in-kernel HBM->SMEM DMA (as F-a) + unrolled sum
"""

import functools
import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

EP, E, G = 16, 384, 24
CA = 768 + 128  # count section offset at mix128
W = CA + E


def _units(out_ref, n_units):
    out_ref[...] = jnp.zeros_like(out_ref)

    def unit(u, carry):
        out_ref[pl.ds(u, 1), :, :] = jnp.full((1, 1, 128), u, jnp.int32)
        return carry

    lax.fori_loop(0, n_units, unit, 0)


def _al(c):
    return ((c + 1) // 2) * 2


def k_t1(me_ref, ex_sm, out_ref):
    first = me_ref[0] * G

    def per_src(d, acc):
        return lax.fori_loop(0, G, lambda g, a: a + _al(ex_sm[d, CA + first + g]), acc)

    rows = lax.fori_loop(0, EP, per_src, jnp.int32(0))
    _units(out_ref, (rows + 127) // 128)


def k_t2(me_ref, ex_sm, out_ref):
    first = me_ref[0] * G
    rows = jnp.int32(0)
    for d in range(EP):
        for g in range(G):
            rows = rows + _al(ex_sm[d, CA + first + g])
    _units(out_ref, (rows + 127) // 128)


def k_t3(me_ref, ex_ref, out_ref):
    first = me_ref[0] * G
    cnt = ex_ref[:, CA:CA + E]  # [16, 384]
    lane = lax.broadcasted_iota(jnp.int32, (EP, E), 1)
    mine = (lane >= first) & (lane < first + G)
    rows = jnp.sum(jnp.where(mine, _al(cnt), 0))
    _units(out_ref, (rows + 127) // 128)


def k_t4(me_ref, ex_hbm, out_ref, sm, sem):
    cp = pltpu.make_async_copy(ex_hbm.at[:, pl.ds(CA, E)], sm, sem)
    cp.start()
    cp.wait()
    first = me_ref[0] * G
    rows = jnp.int32(0)
    for d in range(EP):
        for g in range(G):
            rows = rows + _al(sm[d, first + g])
    _units(out_ref, (rows + 127) // 128)


def call(arm, me, ex):
    vmem = pltpu.MemorySpace.VMEM
    out = jax.ShapeDtypeStruct((112, 1, 128), jnp.int32)
    if arm in ("t1", "t2"):
        gs = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2, grid=(), in_specs=[],
                                          out_specs=pl.BlockSpec(memory_space=vmem))
        return pl.pallas_call({"t1": k_t1, "t2": k_t2}[arm], grid_spec=gs, out_shape=out, name=f"kplan_p03_{arm}")(me, ex)
    if arm == "t3":
        gs = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(), in_specs=[pl.BlockSpec(memory_space=vmem)],
                                          out_specs=pl.BlockSpec(memory_space=vmem))
        return pl.pallas_call(k_t3, grid_spec=gs, out_shape=out, name="kplan_p03_t3")(me, ex)
    gs = pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, grid=(), in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
                                      out_specs=pl.BlockSpec(memory_space=vmem),
                                      scratch_shapes=[pltpu.SMEM((EP, E), jnp.int32), pltpu.SemaphoreType.DMA])
    return pl.pallas_call(k_t4, grid_spec=gs, out_shape=out, name="kplan_p03_t4")(me, ex)


def chained(fn, R):
    def run(args, out0):
        def body(i, carry):
            a, prev = lax.optimization_barrier(carry)
            return a, fn(*a)
        return lax.fori_loop(0, R, body, (args, out0))[1]
    return jax.jit(run)


def clock(fn, args, rs=(16, 96, 256)):
    out0 = jax.jit(fn)(*args)
    fns = {R: chained(fn, R) for R in rs}
    for f in fns.values():
        jax.block_until_ready(f(args, out0))
    med = {}
    for R in rs:
        ts = []
        for _ in range(6):
            t0 = time.perf_counter()
            jax.block_until_ready(fns[R](args, out0))
            ts.append(time.perf_counter() - t0)
        med[R] = float(np.median(ts[1:]))
    s, i = np.polyfit(np.array(rs, float), np.array([med[R] for R in rs]), 1)
    return out0, dict(per_call_us=s * 1e6, icpt_us=i * 1e6, hoist=med[rs[-1]] / med[rs[0]])


def main():
    rng = np.random.default_rng(0)
    ex = rng.integers(0, 9, (EP, W)).astype(np.int32)
    res = {}
    for arm in ("t1", "t2", "t3", "t4"):
        for me in (0, 7):
            first = me * G
            nu = (int(_al(ex[:, CA + first:CA + first + G]).sum()) + 127) // 128
            want = np.zeros((112, 1, 128), np.int64)
            for u in range(nu):
                want[u] = u
            key = f"{arm}_me{me}"
            try:
                out, t = clock(functools.partial(call, arm), (jnp.array([me], jnp.int32), jnp.asarray(ex)))
                res[key] = dict(lowers=True, n_units=nu, correct=bool(np.array_equal(np.asarray(out), want)), **t)
            except Exception as e:  # noqa: BLE001
                res[key] = dict(lowers=False, error=str(e)[:500])
            print("PROBE03", key, json.dumps(res[key]), flush=True)
    json.dump(res, open(os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "kplan_probe03.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
