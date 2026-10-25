"""QSA pool writes in an 8192-token prefill chunk: where the time goes.

One pooled QSA layer at the Qwen3.8-Flash TP8 geometry (3 query heads x 256,
one KV head, indexer 4 x 128, ratio 4, top-2048 tokens, FP8 unified pool with
512-token pages, paired main KV slots), on a c32-like mixed chunk: 28 decode
requests of 4 tokens and 4 prefill requests sharing the rest of 8192 tokens.
The indexer selection and the attention are stubbed, so the step is the cache
writes and their operand math. Each writer is then knocked out in turn (main
KV rows, compressed keys, ring/tag read+write) to split the write time, and a
traced run gives the op breakdown.
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa  # noqa: E402

TOKENS, PAGE, PAGES = 8192, 512, 6144
HEADS, HEAD_DIM, INDEX_HEADS, INDEX_DIM = 3, 256, 4, 128
RATIO, TOPK, ROTARY = 4, 2048, 64
DECODE, DECODE_LEN, PREFILL = 28, 4, 4
COLUMNS = 64  # page-table columns per request (32k tokens)
ITERS = 10


def make_step(rng, case):
    if case == "prefill_8192":
        query_lens = [DECODE_LEN] * DECODE
        rest = TOKENS - DECODE * DECODE_LEN
        share = rest // PREFILL
        query_lens += [share] * (PREFILL - 1) + [rest - share * (PREFILL - 1)]
        contexts = list(rng.integers(1000, 16000, DECODE)) + [0, 3000, 8000, 12000]
    else:  # c8 MTP verify step: 8 requests x 4 tokens
        query_lens = [DECODE_LEN] * 8
        contexts = list(rng.integers(1000, 16000, 8))
    requests = len(query_lens)
    starts = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    free = iter(rng.permutation(np.arange(1, PAGES)))
    main = np.zeros((requests, COLUMNS), np.int32)
    comp = np.zeros((requests, COLUMNS), np.int32)
    ring = np.zeros(requests, np.int32)
    for r in range(requests):
        last = (contexts[r] + query_lens[r] - 1) // PAGE
        for c in range(last + 1):
            main[r, c] = next(free)
        for c in range(last // RATIO + 1):
            comp[r, c] = next(free)
        ring[r] = next(free)
    tokens = int(starts[-1])
    positions = np.zeros(tokens, np.int32)
    for r in range(requests):
        positions[starts[r] : starts[r + 1]] = contexts[r] + np.arange(query_lens[r])

    def normal(*shape):
        return jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)

    pool = jax.random.normal(jax.random.key(0), (PAGES, PAGE, 1, 4, HEAD_DIM)).astype(
        jnp.float8_e4m3fn
    )
    rope = jnp.pad(normal(65536, ROTARY), ((0, 0), (0, 128 - ROTARY)))
    args = (
        normal(tokens, 2 * HEADS * HEAD_DIM),
        normal(tokens, HEAD_DIM),
        normal(tokens, HEAD_DIM),
        normal(tokens, (INDEX_HEADS + 1) * INDEX_DIM),
        normal(HEAD_DIM),
        normal(HEAD_DIM),
        normal(INDEX_DIM),
        normal(INDEX_DIM),
        jnp.asarray(positions),
        rope,
        pool,
        jnp.asarray(starts),
        jnp.asarray(contexts, jnp.int32),
        jnp.asarray(ring),
        jnp.asarray(main.reshape(-1)),
        jnp.asarray(comp.reshape(-1)),
    )
    return args, max(query_lens)


def no_selection(query, *args, block_topk, **kwargs):
    return jnp.zeros((len(query), block_topk), jnp.int32)


def no_attention(q, gate, blocks, lengths, table, pool, *args, **kwargs):
    return jnp.zeros(q.shape, jnp.bfloat16), pool


def keep_pool(pool, *args, **kwargs):
    return pool


def zero_ring(pool, ids, ring_elements, tag_elements):
    return (
        jnp.zeros((len(ids), ring_elements), jnp.bfloat16),
        jnp.zeros((len(ids), tag_elements), jnp.int32),
    )


ARMS = {
    "all_writes": {},
    "run_write": {},
    "no_main": {"write_main_cache": keep_pool},
    "no_compressed": {"write_compressed": keep_pool},
    "no_ring": {"read_ring_and_tags": zero_ring, "write_ring_and_tags": keep_pool},
    "no_writes": {
        "write_main_cache": keep_pool,
        "write_compressed": keep_pool,
        "read_ring_and_tags": zero_ring,
        "write_ring_and_tags": keep_pool,
    },
}


def bench(args, max_query_len, patches, main_slot, run_write=False):
    os.environ["VLLM_TPU_QSA_COMPRESSED_RUN_WRITE"] = "1" if run_write else "0"
    jax.clear_caches()
    with mock.patch.multiple(
        pooled_qsa, select_blocks=no_selection, qsa_attention=no_attention, **patches
    ):

        # One jitted step per call: inside a fori_loop the SparseCore offload
        # of the step fails to lower (RET_CHECK buffer_found).
        step = jax.jit(
            lambda pool, *rest: pooled_qsa.pooled_qsa_step(
                *rest[:10],
                pool,
                *rest[10:],
                compress_ratio=RATIO,
                token_topk=TOPK,
                norm_eps=1e-6,
                rotary_dim=ROTARY,
                main_slot=main_slot,
                compressed_slot=2,
                max_query_len=max_query_len,
            )[0],
            donate_argnums=0,
        )
        rest = args[:10] + args[11:]

        def loop(pool):
            return step(pool, *rest)

        pool = jnp.copy(args[10])
        pool = loop(pool)
        jax.block_until_ready(pool)
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(ITERS):
                pool = loop(pool)
            jax.block_until_ready(pool)
            samples.append((time.perf_counter() - t0) / ITERS * 1e3)
        return float(np.median(samples)), loop, pool


def test_prefill_write():
    # The serving recipe's ring settings at conc >= 16 (where 8192-token chunks run).
    os.environ.setdefault("VLLM_TPU_QSA_RING_SELECT", "1")
    os.environ.setdefault("VLLM_TPU_QSA_RING_DMA_BATCH", "32")
    rng = np.random.default_rng(0)
    out = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    results = []
    for case in ("prefill_8192", "decode_32"):
        args, max_query_len = make_step(rng, case)
        for name, patches in ARMS.items():
            ms, loop, pool = bench(
                args, max_query_len, patches, 1, run_write=name == "run_write"
            )
            row = dict(case=case, arm=name, ms_per_layer=ms)
            print("PFW", json.dumps(row), flush=True)
            results.append(row)
            if name in ("all_writes", "run_write"):
                with jax.profiler.trace(os.path.join(out, "xprof", f"{case}_{name}")):
                    for _ in range(3):
                        pool = loop(pool)
                    jax.block_until_ready(pool)
    Path(out, "bench_qsa_prefill_write.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    test_prefill_write()
