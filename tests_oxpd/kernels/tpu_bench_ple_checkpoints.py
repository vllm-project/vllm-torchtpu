# SPDX-License-Identifier: Apache-2.0
"""On-TPU time of `pooled_short_conv_speculative`, request-major vs token-major checkpoints.

Served shapes: 32 request rows, window 4, 9-step state, 10240 channels. Each
variant is chained on device with the pool as the carry (M-sweep slope), and
the two final pools are compared byte for byte.
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.qwen4_exp import pool as P
from vllm_torchtpu.kernels.qwen4_exp.ple import short_conv

COUNT, WINDOW, S, C = 32, 4, 9, 10240


def variant(token_major):
    def f(x, weight, pool, starts, ids, initial, read_ids, ckpt_ids, windowed):
        history = P.read_region(pool, read_ids, elements=S * C, dtype=jnp.bfloat16).reshape(COUNT, S, C)
        lengths = starts[1:] - starts[:-1]
        active = (ids > 0) & (lengths > 0)
        slots = jnp.where(active, jnp.arange(COUNT, dtype=jnp.int32), -1)
        output, final = short_conv(x, weight, history, starts, slots, initial, dilation=3,
                                   request_owned_history=True)
        if token_major:
            v, t = P.token_checkpoints(x, history, starts, initial, active, ckpt_ids, windowed)
        else:
            v, t = P.request_checkpoints(x, history, final, starts, initial, active, ckpt_ids, windowed)
        return P.write_region(pool, v, t), output
    return f


def slope(f, pool, args, chains=(8, 32)):
    best = {}
    for n in chains:
        g = jax.jit(lambda p, a, n=n: jax.lax.fori_loop(0, n, lambda _, q: f(a[0], a[1], q, *a[2:])[0], p))
        jax.block_until_ready(g(pool, args))
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            jax.block_until_ready(g(pool, args))
            ts.append(time.perf_counter() - t0)
        best[n] = min(ts)
    k, _ = np.polyfit(np.array(list(best), float), np.array(list(best.values()), float), 1)
    return round(k * 1e6, 1)


def main():
    rng = np.random.default_rng(0)
    nblk = 1 + COUNT * WINDOW + 8
    for T, live in ((4, 4), (8, 2), (16, 4), (32, 8)):
        per = T // live
        lengths = np.zeros(COUNT, np.int32)
        lengths[:live] = per
        starts = jnp.asarray(np.concatenate([[0], np.cumsum(lengths)]), jnp.int32)
        ids = jnp.asarray(np.where(lengths > 0, 1 + np.arange(COUNT), 0), jnp.int32)
        ckpt = jnp.asarray((1 + np.arange(COUNT * WINDOW)).reshape(COUNT, WINDOW), jnp.int32)
        read_ids = ckpt[:, 0] * (ids > 0)
        windowed = jnp.ones((COUNT,), bool)
        initial = jnp.asarray(lengths > 0)
        x = jnp.asarray(rng.normal(size=(T, C)), jnp.bfloat16)
        weight = jnp.asarray(rng.normal(size=(C, 4)), jnp.bfloat16)
        pool = jnp.asarray(rng.normal(size=(nblk, 1664, 1, 2, 256)), jnp.bfloat16)
        args = (x, weight, starts, ids, initial, read_ids, ckpt, windowed)
        out = {"T": T, "live_requests": live}
        pools = {}
        for name, tm in (("request_major", False), ("token_major", True)):
            f = variant(tm)
            pools[name] = np.asarray(jax.jit(lambda p, f=f: f(x, weight, p, *args[2:]))(pool)[0]).view(np.uint16)
            out[name + "_us"] = slope(f, pool, args)
        out["pool_bytes_differ"] = int((pools["request_major"] != pools["token_major"]).sum())
        print("RESULT", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
