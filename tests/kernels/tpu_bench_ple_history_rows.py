# SPDX-License-Identifier: Apache-2.0
"""On-TPU time of the speculative PLE short conv, VLLM_TPU_PLE_HISTORY_ROW_KERNEL off and on.

The call is chained on device with the pool and the output as the loop carry,
and the per-call time is the slope over 8, 32 and 64 chained calls (min of 7).
Layouts and inputs are those of test_qwen4_exp_ple_history_rows_tpu.py. Prints
one PLEROWS JSON line per measurement.
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_qwen4_exp_ple_history_rows_tpu as T  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp import pool as ple_pool  # noqa: E402


def slope(step, carry, consts, chains=(8, 32, 64), repeats=7):
    best = {}
    for n in chains:
        g = jax.jit(
            lambda c, k, n=n: jax.lax.fori_loop(0, n, lambda _, x: step(x, k), c)
        )
        jax.block_until_ready(g(carry, consts))
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(g(carry, consts))
            times.append(time.perf_counter() - t0)
        best[n] = min(times)
    k, _ = np.polyfit(
        np.array(list(best), float), np.array(list(best.values()), float), 1
    )
    return round(k * 1e6, 2)


def main():
    names = sys.argv[1:] or [
        "decode24_t96",
        "decode48_t192",
        "decode64_t256",
        "decode80_t320",
        "decode96_t384",
        "decode128_t512",
        "short_t96",
        "prefill_t1024",
    ]
    for name in names:
        args, _ = T._inputs(name, seed=1)
        pool = args[2]
        consts = tuple(args[:2]) + tuple(args[3:])
        for flag in (False, True):
            os.environ[T.FLAG] = "1" if flag else "0"

            def step(c, k):
                x, weight = k[:2]
                return ple_pool.pooled_short_conv_speculative(
                    x, weight, c[0], *k[2:], dilation=T.PLE_DIL, state_len=T.PLE_S
                )

            us = slope(step, (pool, jnp.zeros_like(args[0])), consts)
            row = dict(layout=name, flag=flag, us=us)
            print("PLEROWS " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
