"""Prefill relayout items of the c48 8K chunk: per item, the current layout vs the fix.

qgate_pair_major: a DCP pair's q/gate rows enter the QSA step either joined by a
dim -1 all-gather (the join runs as its own copy, `as_strided bf16[2,8192,1536]`
in the trace) or rank-major (VLLM_TPU_QSA_DCP_QGATE_PAIR_MAJOR) with the join
inside the step. Arms time `produce` (the step's first qgate consumer):
  joined    - pre-joined input, the join not timed (floor)
  copy_join - explicit materialized join + produce (today's cost)
  in_step   - rank-major input, joined inside the step (the fix)
and check in_step's outputs are bit-identical to joined's.
Prints RLY lines.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp.qsa import produce  # noqa: E402

T, PAIR, HEADS, D, INDEX_HEADS, INDEX_DIM, ROTARY = 8192, 2, 3, 256, 4, 128, 64
ITERS = 20


def timed(fn, *args):
    out = jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            r = fn(*args)
        jax.block_until_ready(r)
        samples.append((time.perf_counter() - t0) / ITERS * 1e3)
    return float(np.median(samples)), out


def bits_equal(a, b):
    return all(
        bool(jnp.array_equal(jax.lax.bitcast_convert_type(x, jnp.uint16 if x.dtype == jnp.bfloat16 else jnp.uint32),
                             jax.lax.bitcast_convert_type(y, jnp.uint16 if y.dtype == jnp.bfloat16 else jnp.uint32)))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


def qgate_pair_major():
    rng = np.random.default_rng(0)

    def normal(*shape):
        return jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)

    width = HEADS * 2 * D  # one rank's q/gate row
    pair_major = normal(PAIR, T, width)
    rest = (
        normal(T, D), normal(T, D), normal(T, (INDEX_HEADS + 1) * INDEX_DIM),
        normal(D), normal(D), normal(INDEX_DIM),
        jnp.asarray(rng.integers(0, 60000, T), jnp.int32),
        jnp.pad(normal(65536, ROTARY), ((0, 0), (0, 128 - ROTARY))),
    )

    def join(x):
        return jnp.moveaxis(x, 0, 1).reshape(x.shape[1], -1)

    def run(qgate, *rest):
        return produce(qgate, *rest[:7], rest[7], 1e-6, ROTARY)

    joined = jax.block_until_ready(jax.jit(join)(pair_major))
    floor = jax.jit(run)
    copy_join = jax.jit(lambda x, *r: run(jax.lax.optimization_barrier(join(x)), *r))
    in_step = jax.jit(lambda x, *r: run(join(x), *r))
    rows = []
    ms_floor, ref = timed(floor, joined, *rest)
    ms_copy, out_copy = timed(copy_join, pair_major, *rest)
    ms_fix, out_fix = timed(in_step, pair_major, *rest)
    for arm, ms, out in (("joined", ms_floor, ref), ("copy_join", ms_copy, out_copy),
                         ("in_step", ms_fix, out_fix)):
        rows.append(dict(item="qgate_pair_major", arm=arm, ms_per_layer=round(ms, 4),
                         identical=bits_equal(out, ref)))
    return rows


ITEMS = {"qgate_pair_major": qgate_pair_major}


def main():
    names = sys.argv[1:] or list(ITEMS)
    for name in names:
        for row in ITEMS[name]():
            print("RLY", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
