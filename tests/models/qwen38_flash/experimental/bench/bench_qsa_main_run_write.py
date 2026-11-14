# Provenance: copied from scripts/bench_qsa_main_run_write.py
#   at ee1071f71152a5e7cb0f3c6cda22a50bb70799da.
"""Main KV writes of one QSA layer in an 8192-token prefill chunk: tiled vs run writer.

Qwen3.8-Flash TP8 per-rank geometry: FP8 unified pool, 512-token pages, paired
main KV slots (one KV head per layer, two layers per page). Steps:

- c48 / c32: 44 / 28 MTP verify requests (4 tokens) at 20-90K context, plus 4
  prefill requests sharing the rest of the chunk at ~90K context (unaligned).
- long: one 8192-token prefill at context 90001 (unaligned) / 90112 (aligned).

Each step runs without DCP and with main-KV DCP-2 (both ranks). Per arm: ms per
call (one layer) and the written pool's bytes against the tiled writer's.
Prints MRW lines and writes $CDK_OUTPUT_DIR/bench_qsa_main_run_write.json.
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

from vllm_torchtpu.models.qwen38_flash.experimental import settings  # noqa: E402
from vllm_torchtpu.models.qwen38_flash.experimental.kernels import (
    pooled_qsa,  # noqa: E402
)

TOKENS, PAGE, PAGES, WIDTH = 8192, 512, 9216, 256
ITERS = int(os.environ.get("ITERS", "20"))
TILES = [int(t) for t in os.environ.get("TILES", "256 512 1024").split()]


def composition(rng, case):
    if case == "long_unaligned":
        return [TOKENS], [90001]
    if case == "long_aligned":
        return [TOKENS], [90112]
    decode = {"c48": 44, "c32": 28}[case]
    lens = [4] * decode
    rest = TOKENS - 4 * decode
    share = rest // 4
    lens += [share] * 3 + [rest - 3 * share]
    contexts = list(rng.integers(20000, 90000, decode)) + [90001, 88003, 61077, 89411]
    return lens, contexts


def destinations(rng, lens, contexts, dcp):
    size, rank = dcp or (1, 0)
    span = size * PAGE
    free = iter(rng.permutation(np.arange(PAGES)))
    page = np.full(TOKENS, PAGES, np.int32)
    logical = np.zeros(TOKENS, np.int32)
    valid = np.zeros(TOKENS, bool)
    i = 0
    for n, ctx in zip(lens, contexts):
        table = {}
        for t in range(ctx, ctx + n):
            if t // span not in table:
                table[t // span] = next(free)
            logical[i] = t
            if (t // PAGE) % size == rank:
                valid[i], page[i] = True, table[t // span]
            i += 1
    return jnp.asarray(page), jnp.asarray(logical), jnp.asarray(valid)


@jax.jit
def same_bytes(a, b):
    bits = lambda x: jax.lax.bitcast_convert_type(x, jnp.uint8)  # noqa: E731
    return jnp.array_equal(bits(a), bits(b))


def time_call(fn, pool, rows):
    pool = fn(pool, rows)
    jax.block_until_ready(pool)
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            pool = fn(pool, rows)
        jax.block_until_ready(pool)
        samples.append((time.perf_counter() - t0) / ITERS * 1e3)
    return float(np.median(samples)), pool


def arm(run_write, tile, page, logical, valid, slot, pool0, rows):
    settings.VLLM_TPU_QSA_MAIN_RUN_WRITE = run_write  # read at trace time
    settings.VLLM_TPU_QSA_MAIN_RUN_WRITE_MIN_TOKENS = 0
    settings.VLLM_TPU_QSA_MAIN_RUN_WRITE_TILE = tile
    jax.clear_caches()
    fn = jax.jit(
        lambda pool, rows: pooled_qsa.write_main_cache(
            pool, rows, page, logical, valid, slot=slot
        ),
        donate_argnums=0,
    )
    once = fn(jnp.copy(pool0), rows)
    ms, _ = time_call(fn, jnp.copy(pool0), rows)
    return ms, once


def main():
    rng = np.random.default_rng(0)
    out = os.environ.get("CDK_OUTPUT_DIR", "/tmp")
    pool0 = jax.random.normal(jax.random.key(0), (PAGES, PAGE, 1, 4, WIDTH)).astype(
        jnp.float8_e4m3fn
    )
    raw = rng.normal(0, 1, (TOKENS, 1, 4, WIDTH))
    raw[:, :, 2:] = 0  # pack_rows: K, V, 0, 0
    rows = jnp.asarray(raw, jnp.float8_e4m3fn)
    results = []
    for case in ("c48", "c32", "long_unaligned", "long_aligned"):
        lens, contexts = composition(rng, case)
        for dcp in (None, (2, 0), (2, 1)):
            page, logical, valid = destinations(rng, lens, contexts, dcp)
            for slot in (0, 1):
                base_ms, base = arm(False, 512, page, logical, valid, slot, pool0, rows)
                row = dict(
                    case=case,
                    dcp=str(dcp),
                    slot=slot,
                    arm="tiled",
                    ms=base_ms,
                    wrote=not bool(same_bytes(base, pool0)),
                )
                print("MRW", json.dumps(row), flush=True)
                results.append(row)
                for tile in TILES:
                    ms, got = arm(True, tile, page, logical, valid, slot, pool0, rows)
                    row = dict(
                        case=case,
                        dcp=str(dcp),
                        slot=slot,
                        arm=f"run{tile}",
                        ms=ms,
                        speedup=base_ms / ms,
                        identical=bool(same_bytes(got, base)),
                    )
                    del got
                    print("MRW", json.dumps(row), flush=True)
                    results.append(row)
    bad = [r for r in results if r.get("identical") is False]
    print("MRW_SUMMARY", json.dumps({"rows": len(results), "not_identical": len(bad)}))
    Path(out, "bench_qsa_main_run_write.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
