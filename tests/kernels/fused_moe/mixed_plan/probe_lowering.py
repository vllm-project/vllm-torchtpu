# SPDX-License-Identifier: Apache-2.0
"""Design-time FACT probe 02: two lowering questions the stage-A design rests on.

F-a  trip count from data: DMA the exchange's count section [ep, E] i32 from an
     HBM (ANY) operand into SMEM scratch, close this shard's live slab rows with a
     scalar loop (sum over G local experts x ep sources of the ROWBLK-aligned
     count), use ceil(rows / 128) as a lax.fori_loop bound. Lowers? Cost?
F-b  1-D output: the token gather is consumed as a 1-D s32[N] array. Arms:
     (b1) out_shape 1-D HBM [N], written by DMA from a 1-D VMEM scratch [N];
     (b2) out_shape 2-D HBM [N/128, 128] DMA'd from a [N/128,128] scratch, then
          .reshape(N) in XLA -- does XLA insert a relayout copy (HLO census)?
     N = 13056 (mix128 gather rows, N/128 = 102, not a multiple of 8).
Also times every arm with the chained clock (R in {16, 96, 256}, carry barriered).
"""

import functools
import json
import os
import re
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

EP, E, G = 16, 384, 24
N = 13056


def _fa_kernel(me_ref, ex_hbm, out_ref, cnt_sm, sem, *, count_at):
    cp = pltpu.make_async_copy(ex_hbm.at[:, pl.ds(count_at, E)], cnt_sm, sem)
    cp.start()
    cp.wait()
    first = me_ref[0] * G

    def per_src(d, acc):
        def per_g(g, a):
            c = cnt_sm[d, first + g]
            return a + ((c + 1) // 2) * 2
        return lax.fori_loop(0, G, per_g, acc)

    rows = lax.fori_loop(0, EP, per_src, jnp.int32(0))
    n_units = (rows + 127) // 128

    def unit(u, carry):
        out_ref[pl.ds(u, 1), :, :] = jnp.full((1, 1, 128), u, jnp.int32)
        return carry

    out_ref[...] = jnp.zeros_like(out_ref)
    lax.fori_loop(0, n_units, unit, 0)


def fa_call(me, ex, count_at):
    return pl.pallas_call(
        functools.partial(_fa_kernel, count_at=count_at),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM),
            scratch_shapes=[pltpu.SMEM((EP, E), jnp.int32), pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct((112, 1, 128), jnp.int32),
        name="kplan_probe_fa",
    )(me, ex)


def _fb1_kernel(x_ref, out_hbm, scr, sem):
    scr[...] = x_ref[...] + 1
    cp = pltpu.make_async_copy(scr, out_hbm, sem)
    cp.start()
    cp.wait()


def fb1_call(x1d):
    return pl.pallas_call(
        _fb1_kernel,
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=[pltpu.VMEM((N,), jnp.int32), pltpu.SemaphoreType.DMA],
        out_shape=jax.ShapeDtypeStruct((N,), jnp.int32),
        name="kplan_probe_fb1",
    )(x1d)


def fb2_call(x1d):
    x2 = x1d.reshape(N // 128, 128)
    y = pl.pallas_call(
        _fb1_kernel,
        in_specs=[pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        scratch_shapes=[pltpu.VMEM((N // 128, 128), jnp.int32), pltpu.SemaphoreType.DMA],
        out_shape=jax.ShapeDtypeStruct((N // 128, 128), jnp.int32),
        name="kplan_probe_fb2",
    )(x2)
    return y.reshape(N)


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


def ops_after(fn, args):
    txt = jax.jit(fn).lower(*args).compile().as_text()
    entry = txt.split("ENTRY", 1)[1]
    return sorted(set(re.findall(r"=\s*\S+\s+(copy|transpose|fusion|bitcast|reshape|custom-call)\(", entry)))


def main():
    res = {}
    rng = np.random.default_rng(0)
    count_at = 768 + 128
    ex = rng.integers(0, 9, (EP, count_at + E)).astype(np.int32)
    for me in (0, 7):
        try:
            out, t = clock(functools.partial(fa_call, count_at=count_at), (jnp.array([me], jnp.int32), jnp.asarray(ex)))
            first = me * G
            rows = int((((ex[:, count_at + first:count_at + first + G] + 1) // 2) * 2).sum())
            nu = (rows + 127) // 128
            want = np.zeros((112, 1, 128), np.int64)
            for u in range(nu):
                want[u] = u
            res[f"fa_me{me}"] = dict(lowers=True, n_units=nu, correct=bool(np.array_equal(np.asarray(out), want)), **t)
        except Exception as e:  # noqa: BLE001
            res[f"fa_me{me}"] = dict(lowers=False, error=str(e)[:600])
        print("PROBE02", f"fa_me{me}", json.dumps(res[f"fa_me{me}"]), flush=True)
    x = jnp.asarray(rng.integers(0, 1000, N).astype(np.int32))
    for name, fn in (("fb1_1d_scratch", fb1_call), ("fb2_2d_then_reshape", fb2_call)):
        try:
            out, t = clock(fn, (x,))
            res[name] = dict(lowers=True, correct=bool(np.array_equal(np.asarray(out), np.asarray(x) + 1)),
                             entry_ops=ops_after(fn, (x,)), **t)
        except Exception as e:  # noqa: BLE001
            res[name] = dict(lowers=False, error=str(e)[:600])
        print("PROBE02", name, json.dumps(res[name]), flush=True)
    json.dump(res, open(os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp"), "kplan_probe02.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
