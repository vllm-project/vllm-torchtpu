# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark of VLLM_TPU_QSA_PREFILL_UNION_QK_AHEAD at TP8 prefill shapes.

The served union walk call (group 128, KV tile 1024, cap 12288, min context
32768, lean) on bench_prefill_vpu's TP8 rank inputs (3 query heads, 1 KV head,
FP8 paired 512-token pages, top-512 from a per-block hot set), flag off vs on.
Prints one BENCH line per (shape, variant): device us per call of the whole op
and whether the output is bit-identical to the flag-off run. BENCH_POISON=1
NaN-poisons every unwritten cache row (see bench_prefill_vpu.make_inputs).

Usage (one TPU chip): python bench_union_walk_ahead.py [--shapes ...]
"""

import argparse
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from bench_prefill_vpu import device_time, make_inputs

SHAPES = {
    # name: (rows, context)
    "r1024_c108k": (1024, 108_000),
    "r3072_c108k": (3072, 108_000),
    "r3072_c60k": (3072, 60_000),
    "r3072_c160k": (3072, 160_000),
    "r3072_c233k": (3072, 233_000),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args()
    from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud

    print("devices", jax.devices(), flush=True)
    for name in args.shapes.split(","):
        rows, context = SHAPES[name]
        (query, blocks, lengths, table, cache), _ = make_inputs(rows, context, 3, 512, 1)
        gate = jax.random.uniform(
            jax.random.PRNGKey(3), query.shape, jnp.float32, -3, 3
        ).astype(jnp.bfloat16)
        starts = jnp.asarray([0, rows], jnp.int32)
        base = None
        for rep in range(args.reps):
            for variant, flag in (("off", "0"), ("ahead", "1")):
                os.environ["VLLM_TPU_QSA_PREFILL_UNION_QK_AHEAD"] = flag
                jax.clear_caches()
                fn = jax.jit(functools.partial(
                    ud.union_walk_prefill, kv_heads=1, ratio=4, group=128,
                    kv_tile=1024, union_cap=12288, min_context=32768, lean=True,
                ))
                us, (out, _) = device_time(
                    fn, (query, gate, blocks, lengths, table, cache, starts), args.iters
                )
                bits = np.asarray(out).view(np.uint16)
                if base is None:
                    base = bits
                print(
                    f"BENCH shape={name} rep={rep} variant={variant} total_us={us:.1f} "
                    f"bitexact={bool(np.array_equal(bits, base))} "
                    f"finite={bool(np.isfinite(np.asarray(out, np.float32)).all())}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
