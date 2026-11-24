# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: B023
"""Microbench: the c64 decode QKV GEMM (MXFP8 per-channel e4m3 weights, one
TensorCore) with and without every index-query head (`qkv_index_heads`,
+4 x 128 output columns), and whether the default columns come out bit for
bit the same in the wider GEMM.

    PYTHONPATH=src python scripts/bench_qkv_index_heads.py --tokens 192,256,512
"""

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

HID, D, NQ = 6144, 128, 8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="192,256,320,384,448,512")
    p.add_argument("--layers", type=int, default=64)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    base_cols = (NQ + 4) * D  # q | k | v | index_q | index_k
    w_all = rng.standard_normal((HID, base_cols + 4 * D)).astype(np.float32) * 0.05
    # wide layout: q | index_q (4) | k | v | own index_q (head 1) | index_k
    q_c, k_c, v_c = (
        slice(0, NQ * D),
        slice(NQ * D, (NQ + 1) * D),
        slice((NQ + 1) * D, (NQ + 2) * D),
    )
    iq_own, ik = slice((NQ + 2) * D, (NQ + 3) * D), slice((NQ + 3) * D, (NQ + 4) * D)
    iq_all = w_all[:, base_cols:]
    iq_all[:, D : 2 * D] = w_all[:, iq_own]
    base = w_all[:, :base_cols]
    wide = np.concatenate(
        [
            base[:, q_c],
            iq_all,
            base[:, k_c],
            base[:, v_c],
            base[:, iq_own],
            base[:, ik],
        ],
        1,
    )
    wb = jnp.asarray(base, jnp.float8_e4m3fn)
    ww = jnp.asarray(wide, jnp.float8_e4m3fn)
    sb = jnp.asarray(2.0 ** rng.integers(-3, 3, base.shape[1]), jnp.bfloat16)
    sw = jnp.concatenate(
        [sb[q_c], jnp.ones((4 * D,), jnp.bfloat16) * 0.5, sb[NQ * D :]]
    )
    sw = sw.at[NQ * D + D : NQ * D + 2 * D].set(sb[iq_own])

    def apply(x, w, s):  # MiniMaxM3Mxfp8LinearMethod.apply below 4096 rows
        return jnp.matmul(x, w.astype(x.dtype)) * s

    for T in (int(t) for t in args.tokens.split(",")):
        x = jnp.asarray(rng.standard_normal((T, HID)), jnp.bfloat16)
        ob = np.asarray(jax.jit(apply)(x, wb, sb))
        ow = np.asarray(jax.jit(apply)(x, ww, sw))
        same = (
            np.array_equal(ow[:, : NQ * D], ob[:, q_c])
            and np.array_equal(ow[:, (NQ + 4) * D :], ob[:, NQ * D :])
            and np.array_equal(ow[:, (NQ + 1) * D : (NQ + 2) * D], ob[:, iq_own])
        )

        def chain(n, w, s):
            def f(x):
                def body(_, c):
                    y = apply(c, w, s)
                    return c + (y[:, :1] * 0).astype(c.dtype)

                return jax.lax.fori_loop(0, n, body, x)

            return jax.jit(f)

        res = {}
        for name, w, s in (("base", wb, sb), ("wide", ww, sw)):
            t = {}
            for n in (args.layers // 4, args.layers):
                fn = chain(n, w, s)
                jax.block_until_ready(fn(x))
                samples = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    jax.block_until_ready([fn(x) for _ in range(4)])
                    samples.append((time.perf_counter() - t0) / 4)
                t[n] = statistics.median(samples)
            res[name] = (
                (t[args.layers] - t[args.layers // 4]) / (args.layers * 3 / 4) * 1e6
            )
        print(
            f"QKV T{T} base {res['base']:.2f} wide {res['wide']:.2f} us/layer "
            f"(+{res['wide'] - res['base']:.2f}); default columns bitwise {same}",
            flush=True,
        )


if __name__ == "__main__":
    main()
