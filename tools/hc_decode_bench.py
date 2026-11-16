# SPDX-License-Identifier: Apache-2.0
"""Times the fused decode HC kernel per call on one TPU core.

For each decode bucket and TPU_QWEN4_HC_DECODE_TILE_ROWS cap it runs a chain of
calls inside one jit, each call's outputs feeding the next as in the decoder
(combined streams -> next residual, mixed -> next block, injection -> next
gate), so the time per call excludes dispatch:

* folded: one combine+mix call (``hc_decode_mix`` with ``block``), the decode
  path with TPU_QWEN4_HC_DECODE_FOLD;
* unfolded: the XLA combine followed by the mix call, the path without it.

usage: python tools/hc_decode_bench.py [--caps 0,224,256,512] [--tokens 96,...]
Prints one line per case and writes JSON to $CDK_OUTPUT_DIR/hc-decode-bench.json
when that is set.
"""

import argparse
import json
import os
import statistics
import time

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.qwen4_exp.hc_decode import decode_kernel_tile, hc_decode_mix

BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6
WIDTH = BRANCHES * HIDDEN


def _combine(residual, block, injection):
    residual = residual.astype(jnp.float32)
    block = block.astype(jnp.float32)
    parts = [
        residual[:, b * HIDDEN : (b + 1) * HIDDEN] + block * injection[:, b : b + 1]
        for b in range(BRANCHES)
    ]
    return jnp.concatenate(parts, axis=-1).astype(jnp.bfloat16)


def _chain(weights, cap, folded, iters):
    static = dict(branches=BRANCHES, hidden=HIDDEN, eps=EPS, max_tile=cap)

    def step(_, carry):
        residual, block, injection = carry
        if folded:
            mixed, injection, combined = hc_decode_mix(
                residual, *weights, block=block, prev_injection=injection, **static
            )
        else:
            combined = _combine(residual, block, injection)
            mixed, injection = hc_decode_mix(combined, *weights, **static)
        return combined, mixed, injection

    return jax.jit(lambda *c: jax.lax.fori_loop(0, iters, step, c))


def _time(fn, args, repeats):
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return min(samples), statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", default="0,224,256,512")
    parser.add_argument(
        "--tokens", default="96,112,128,144,160,192,208,224,240,256,320,384,448"
    )
    parser.add_argument("--iters", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    opts = parser.parse_args()
    keys = jax.random.split(jax.random.key(0), 8)
    bf16 = jnp.bfloat16
    weights = [
        (0.3 * jax.random.normal(keys[3], (WIDTH,))).astype(bf16),
        (0.02 * jax.random.normal(keys[4], (LOWRANK, WIDTH))).astype(bf16),
        (0.05 * jax.random.normal(keys[5], (WIDTH, LOWRANK))).astype(bf16),
        (0.02 * jax.random.normal(keys[6], (BRANCHES, WIDTH))).astype(bf16),
    ]
    rows = []
    print(f"device {jax.devices()[0].device_kind}, {opts.iters} chained calls per run")
    for tokens in [int(t) for t in opts.tokens.split(",")]:
        residual = jax.random.normal(keys[0], (tokens, WIDTH)).astype(bf16)
        block = jax.random.normal(keys[1], (tokens, HIDDEN)).astype(bf16)
        gate = 2.0 * jax.nn.sigmoid(jax.random.normal(keys[2], (tokens, BRANCHES)))
        args = (residual, block, gate.astype(jnp.float32))
        for cap in [int(c) for c in opts.caps.split(",")]:
            for folded in (True, False):
                fn = _chain(weights, cap, folded, opts.iters)
                best, median = _time(fn, args, opts.repeats)
                row = dict(
                    tokens=tokens,
                    cap=cap,
                    tile=decode_kernel_tile(tokens, cap),
                    path="folded" if folded else "unfolded",
                    us_per_call_min=1e6 * best / opts.iters,
                    us_per_call_median=1e6 * median / opts.iters,
                )
                rows.append(row)
                print(
                    f"tokens {tokens:4d} cap {cap:3d} tile {row['tile']:4d} "
                    f"{row['path']:8s} {row['us_per_call_min']:8.2f} us/call "
                    f"(median {row['us_per_call_median']:.2f})",
                    flush=True,
                )
    out = os.environ.get("CDK_OUTPUT_DIR")
    if out:
        with open(os.path.join(out, "hc-decode-bench.json"), "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
