"""QSA decode indexer microbench for one-token request rows (MTP draft steps).

Qwen3.8-Flash TP8 geometry: 4 index heads x 128, FP8 unified pool with
512-token pages, ratio 4 (32 compressed keys per 128 tokens), top-512
groups, shared-stream indexer (VLLM_TPU_QSA_DECODE_QUERY_BLOCK=4). Sweeps
VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK and checks the selection is unchanged.
"""

import functools
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa  # noqa: E402

RATIO, TOPK_GROUPS, PAGE, POOL_PAGES = 4, 512, 512, 4096
ITERS = 20


def make(rng, rows, max_ctx):
    cols = max_ctx // PAGE
    context = rng.integers(max_ctx // 4, max_ctx - 8, size=rows).astype(np.int32)
    table = np.stack([rng.permutation(POOL_PAGES)[:cols] for _ in range(rows)])
    query = rng.standard_normal((rows, 4, 128)).astype(np.float32)
    return dict(
        query=jnp.asarray(query, jnp.bfloat16),
        lengths=jnp.asarray(context + 1),
        table=jnp.asarray(table, jnp.int32),
        req=jnp.arange(rows, dtype=jnp.int32),
        starts=jnp.arange(rows + 1, dtype=jnp.int32),
        context=jnp.asarray(context),
    )


def run(pool, args, block):
    os.environ["VLLM_TPU_QSA_DECODE_QUERY_BLOCK"] = "4"
    os.environ["VLLM_TPU_QSA_DECODE_KV_BLOCK"] = "1024"
    os.environ["VLLM_TPU_QSA_DECODE_SINGLE_KV_BLOCK"] = str(block)
    select = functools.partial(
        pooled_qsa.select_blocks, ratio=RATIO, block_topk=TOPK_GROUPS
    )

    @jax.jit
    def loop(pool, query):
        def body(i, acc):
            sel = select(query=query + acc.astype(query.dtype) * 0, pool=pool,
                         **{k: v for k, v in args.items() if k != "query"})
            return acc + jnp.sum(sel).astype(jnp.float32) * 0
        return jax.lax.fori_loop(0, ITERS, body, jnp.float32(0))

    jax.block_until_ready(loop(pool, args["query"]))
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(loop(pool, args["query"]))
        samples.append((time.perf_counter() - t0) / ITERS * 1e6)
    sel = jax.jit(lambda pool, query: select(
        query=query, pool=pool, **{k: v for k, v in args.items() if k != "query"}
    ))(pool, args["query"])
    return float(np.median(samples)), np.sort(np.asarray(sel), axis=-1)


def test_draft_indexer():
    rng = np.random.default_rng(0)
    pool = (
        jax.random.normal(jax.random.key(0), (POOL_PAGES, PAGE, 1, 4, 256))
        .astype(jnp.float8_e4m3fn)
    )
    results = []
    for rows, max_ctx in itertools.product((4, 8, 32), (16384, 65536)):
        args = make(rng, rows, max_ctx)
        ref = None
        for block in (256, 512, 1024, 2048):
            try:
                us, sel = run(pool, args, block)
                if ref is None:
                    ref = sel
                same = bool(np.array_equal(sel, ref))
            except Exception as e:  # noqa: BLE001
                us, same = repr(e)[:200], None
            row = dict(rows=rows, max_ctx=max_ctx, block=block, us=us, same=same)
            print("IDX", json.dumps(row), flush=True)
            results.append(row)
    out = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    Path(out, "bench_qsa_draft_indexer.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    test_draft_indexer()
