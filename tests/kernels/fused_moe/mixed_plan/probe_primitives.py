# SPDX-License-Identifier: Apache-2.0
"""Design-time FACT probe (write-tpu-kernels rule 1): do the gather-formulation
primitives lower on this toolchain, and what does each cost at the served
shapes? Not a candidate: it computes only the slab-row token index, with no
correctness claim beyond a self-check against numpy.

Primitives:
  P-NT   : S''^T[t, (g,d)] = dot_general(Q[t,P], M[(g,d),P], NT), bf16 0/1 -> f32
  P-PRE  : C''^T = Lstrict[t,t] @ S''^T  (exclusive prefix over tokens)
  P-LOOP : lax.fori_loop over a trip count read from SMEM (dynamic)
  P-UNIT : per 128 slab rows: run id by compare-count over 384 run starts,
           OH^T one-hot, Cseg^T = C''^T @ OH^T, tok = sum_t[Cseg <= j] - 1
Arms sweep n_units (static upper bound 32, dynamic trip = n) at mix128 and
mix256 shapes: slope = per-unit cost, intercept = setup (P-NT + P-PRE + launch).
Clock: chained fori_loop of R calls, carry = previous output barriered with the
inputs (STRAT-SC-03), slope over R; custom-call count checked in the HLO.
"""

import functools
import json
import os
import re
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

G, EP, K = 24, 16, 6
RUNS = G * EP
UNIT = 128
MAX_UNITS = 32


def _iota(shape, dim):
    return lax.broadcasted_iota(jnp.int32, shape, dim)


def _kernel(n_ref, me_ref, pairs_ref, starts_ref, out_ref, *, t_local):
    P = t_local * K
    first = me_ref[0] * G
    pairs = pairs_ref[...]  # [EP, P] i32
    g_col = _iota((G, 1, 1), 0) + first
    m = (pairs[None, :, :] == g_col).astype(jnp.bfloat16).reshape(RUNS, P)  # (g,d) rows
    q = (_iota((t_local, P), 1) // K == _iota((t_local, P), 0)).astype(jnp.bfloat16)
    s_t = lax.dot_general(q, m, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    lower = (_iota((t_local, t_local), 1) < _iota((t_local, t_local), 0)).astype(jnp.bfloat16)
    c_t = jnp.dot(lower, s_t.astype(jnp.bfloat16), preferred_element_type=jnp.float32)
    c_t = c_t.astype(jnp.bfloat16)  # values <= t_local - 1 < 256: exact
    starts = starts_ref[...]  # [RUNS, 1] i32
    k_iota = _iota((RUNS, UNIT), 0)

    def unit(u, carry):
        r = u * UNIT + _iota((1, UNIT), 1)
        n = jnp.sum((starts <= r).astype(jnp.int32), axis=0, keepdims=True)
        k = n - 1
        hit = k_iota == k
        oh = hit.astype(jnp.bfloat16)
        start = jnp.sum(jnp.where(hit, starts, 0), axis=0, keepdims=True)
        j = (r - start).astype(jnp.float32)
        cseg = jnp.dot(c_t, oh, preferred_element_type=jnp.float32)  # [t, UNIT]
        tok = jnp.sum((cseg <= j).astype(jnp.int32), axis=0, keepdims=True) - 1
        out_ref[pl.ds(u, 1), :, :] = tok.reshape(1, 1, UNIT)
        return carry

    out_ref[...] = jnp.zeros_like(out_ref)
    lax.fori_loop(0, n_ref[0], unit, 0)


@functools.lru_cache(None)
def build(t_local):
    kern = functools.partial(_kernel, t_local=t_local)
    vmem = pltpu.MemorySpace.VMEM

    def call(n, me, pairs, starts):
        return pl.pallas_call(
            kern,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=(),
                in_specs=[pl.BlockSpec(memory_space=vmem)] * 2,
                out_specs=pl.BlockSpec(memory_space=vmem),
            ),
            out_shape=jax.ShapeDtypeStruct((MAX_UNITS, 1, UNIT), jnp.int32),
            name=f"kplan_probe_t{t_local}",
            interpret=os.environ.get("KPLAN_INTERPRET") == "1",
        )(n, me, pairs, starts)

    return call


def numpy_ref(t_local, pairs, starts, me, n):
    first = me * G
    out = np.zeros((MAX_UNITS, 1, UNIT), np.int64)
    P = t_local * K
    for u in range(n):
        for li in range(UNIT):
            r = u * UNIT + li
            k = int((starts[:, 0] <= r).sum()) - 1
            g, d = divmod(k, EP)
            j = r - starts[k, 0]
            col = np.array([(pairs[d, t * K:(t + 1) * K] == first + g).any() for t in range(t_local)], np.int64)
            c = np.cumsum(col) - col
            out[u, 0, li] = int((c <= j).sum()) - 1
    return out


def chained(fn, R):
    def run(args, out0):
        def body(i, carry):
            a, prev = lax.optimization_barrier(carry)
            return a, fn(*a)
        return lax.fori_loop(0, R, body, (args, out0))[1]
    return jax.jit(run)


def main():
    rs = [int(r) for r in os.environ.get('KPLAN_RS', '4,12,24').split(',')]
    res = []
    rng = np.random.default_rng(0)
    for t_local in (128, 256):
        P = t_local * K
        pairs = rng.integers(0, 384, (EP, P)).astype(np.int32)
        lens = rng.integers(0, 6, RUNS)
        starts = (np.cumsum(lens) - lens).astype(np.int32).reshape(RUNS, 1)
        call = build(t_local)
        for n in (0, 1, 4, 8, 16):
            args = (jnp.array([n], jnp.int32), jnp.array([5], jnp.int32), jnp.asarray(pairs), jnp.asarray(starts))
            out0 = jax.jit(call)(*args)
            ok = None
            if n in (1, 4):
                ok = bool(np.array_equal(np.asarray(out0), numpy_ref(t_local, pairs, starts, 5, n)))
            hlo = jax.jit(call).lower(*args).compile().as_text()
            ncc = len(re.findall(r"custom-call\(", hlo))
            fns = {R: chained(call, R) for R in rs}
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
            slope, icpt = np.polyfit(np.array(rs, float), np.array([med[R] for R in rs]), 1)
            r = dict(t_local=t_local, n_units=n, per_call_us=slope * 1e6, icpt_us=icpt * 1e6,
                     hoist=med[rs[-1]] / med[rs[0]], custom_calls=ncc, selfcheck=ok)
            res.append(r)
            print("PROBE", json.dumps(r), flush=True)
    out = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    json.dump(res, open(os.path.join(out, os.environ.get("KPLAN_PROBE_OUT", "kplan_probe00.json")), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
