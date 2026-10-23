"""QSA ring/tag state I/O microbench on the 512-token FP8 pool page.

Qwen3.8-Flash TP8: compress ratio 4 -> 8 ring rows x 128 lanes (1024 BF16
elements) + 16 int32 tags per request, 32 request rows, FP8 unified pool with
(512, 1, 4, 256) pages. The tuned DMA batch table has no entry for that page,
so reads and writes issue one serial DMA per request. Sweeps
VLLM_TPU_QSA_RING_DMA_BATCH and checks reads and the written pool match.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp import qsa_state  # noqa: E402

RING, TAGS, ITERS = 1024, 16, 50


def timed(fn, *args):
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - t0) / ITERS * 1e6)
    return float(np.median(samples))


def test_ring_io():
    rng = np.random.default_rng(0)
    pages = 8192
    pool = jax.random.normal(jax.random.key(0), (pages, 512, 1, 4, 256)).astype(
        jnp.float8_e4m3fn
    )
    results = []
    for live in (4, 8, 32):
        ids = np.zeros(32, np.int32)
        ids[:live] = rng.choice(np.arange(1, pages), live, replace=False)
        ids = jnp.asarray(ids)
        ring = jnp.asarray(rng.standard_normal((32, RING)), jnp.bfloat16)
        tags = jnp.asarray(rng.integers(-5, 1 << 20, (32, TAGS)), jnp.int32)
        ref = None
        for batch in (0, 8, 16, 32):
            os.environ["VLLM_TPU_QSA_RING_DMA_BATCH"] = str(batch)

            @jax.jit
            def read_loop(pool, ids):
                def body(i, acc):
                    r, t = qsa_state.read_ring_and_tags(
                        pool, ids + (acc * 0).astype(jnp.int32), RING, TAGS
                    )
                    return acc + r[0, 0].astype(jnp.float32) * 0 + t[0, 0] * 0
                return jax.lax.fori_loop(0, ITERS, body, jnp.float32(0))

            @jax.jit
            def write_loop(pool, ring, tags, ids):
                def body(i, pool):
                    return qsa_state.write_ring_and_tags(pool, ring, tags, ids)
                return jax.lax.fori_loop(0, ITERS, body, pool)

            read_us = timed(read_loop, pool, ids)
            write_us = timed(write_loop, pool, ring, tags, ids)
            r, t = jax.jit(
                lambda p, i: qsa_state.read_ring_and_tags(p, i, RING, TAGS)
            )(pool, ids)
            written = jax.jit(qsa_state.write_ring_and_tags)(pool, ring, tags, ids)
            back = jax.jit(
                lambda p, i: qsa_state.read_ring_and_tags(p, i, RING, TAGS)
            )(written, ids)
            got = (np.asarray(r).view(np.uint16), np.asarray(t),
                   np.asarray(back[0]).view(np.uint16), np.asarray(back[1]))
            if ref is None:
                ref = got
            same = all(np.array_equal(a, b) for a, b in zip(got, ref))
            row = dict(live=live, batch=batch, read_us=read_us, write_us=write_us,
                       same=same)
            print("RING", json.dumps(row), flush=True)
            results.append(row)
    out = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    Path(out, "bench_qsa_ring_io.json").write_text(json.dumps(results, indent=1))
    assert all(r["same"] for r in results)


if __name__ == "__main__":
    test_ring_io()
