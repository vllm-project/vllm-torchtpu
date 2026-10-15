# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native packed candidate baseline versus reusable request-union pipeline."""

import argparse
import functools
import importlib.metadata
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from bench_decode_overlap import fixture, load_kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--width", type=int, default=98304)
    parser.add_argument("--seed", type=int, default=824)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--tiles", type=int, nargs="+", default=[4, 8, 12, 16, 48])
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--capacity", type=int, default=40960)
    parser.add_argument("--candidate-file", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    native, old_topk, _, _ = load_kernels(root)
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import candidate

    inputs, _, _ = fixture(
        args.requests, 6, args.width, args.width - 1, 128, True, "random", 2, args.seed
    )
    q, weights, pool, records, lengths = inputs[:5]
    print("fixture", [(getattr(x, "shape", None)) for x in inputs], flush=True)
    rng = np.random.default_rng(args.seed + 1)
    common = 1434
    candidates = []
    for _ in range(args.requests):
        ids = rng.permutation(args.width // 8)
        candidates.append(
            np.stack(
                [
                    np.concatenate(
                        (
                            ids[:common],
                            ids[
                                common + i * (2048 - common) : common
                                + (i + 1) * (2048 - common)
                            ],
                        )
                    )
                    for i in range(6)
                ]
            )
        )
    candidates = (
        np.load(args.candidate_file)["candidates"]
        if args.candidate_file
        else np.stack(candidates)
    )
    if candidates.shape != (args.requests, 6, 2048):
        raise ValueError("Expected candidate IDs [requests, 6, 2048]")
    counts = [len(np.unique(r[(r >= 0) & (r < args.width // 8)])) for r in candidates]
    capacity = args.capacity
    if capacity % 4096 or max(counts) * 8 > capacity:
        raise ValueError("Capacity must cover every union and be divisible by 4096")
    candidates = jnp.asarray(candidates, jnp.int32)

    def prepare(pool, records, candidates):
        meta = candidate.union_metadata(
            candidates, logical_blocks=args.width // 8, capacity_blocks=capacity // 8
        )
        starts, bits = candidate.union_record_starts(
            meta,
            records,
            block_size=8,
            pool_records=native["_cache_num_rows"](pool) * 2,
        )
        words = native["gather_cache_page_words"](pool, starts, 8, 128)
        words = words.reshape(args.requests, -1, 128)
        keys = words if args.packed else candidate.unpack_candidate_words(words)
        return keys, meta.ids, bits

    def baseline(q, weights, pool, records, candidates, lengths):
        blocks = args.width // 8
        ids = jnp.sort(
            jnp.where((candidates >= 0) & (candidates < blocks), candidates, blocks),
            axis=-1,
        ).reshape(-1, 2048)
        unique = jnp.concatenate(
            (jnp.ones_like(ids[:, :1], dtype=jnp.bool_), ids[:, 1:] != ids[:, :-1]),
            axis=1,
        )
        logical = (ids[..., None] * 8 + jnp.arange(8)).reshape(-1, 16384)
        req = jnp.repeat(jnp.arange(args.requests), 6)
        compact = records[req[:, None], jnp.minimum(logical, args.width - 1)]
        valid = (logical < lengths.reshape(-1, 1)) & (logical < args.width)
        valid &= jnp.repeat(unique, 8, axis=1)
        compact = jnp.where(valid, compact, -1)
        scores = native["paged_index_scores_jax"](
            q.reshape(-1, 32, 128),
            pool,
            compact,
            weights.reshape(-1, 32),
            jnp.full((args.requests * 6,), 16384, jnp.int32),
            page_records=8,
        )
        scores = jnp.where(compact >= 0, scores, -jnp.inf)
        selected = old_topk(
            scores,
            k=512,
            row_lengths=jnp.full((args.requests * 6,), 16384, jnp.int32),
        )
        logical = jnp.take_along_axis(logical, jnp.maximum(selected, 0), axis=1)
        logical = jnp.sort(jnp.where(selected >= 0, logical, args.width), axis=1)
        return jnp.where(logical < args.width, logical, -1)

    prepare = jax.jit(prepare)
    baseline = jax.jit(baseline)
    prepared = prepare(pool, records, candidates)
    jax.block_until_ready(prepared)
    split_q = native["split_halves"](q)
    expected = baseline(q, weights, pool, records, candidates, lengths)
    jax.block_until_ready(expected)
    variants = {
        "baseline": lambda: baseline(q, weights, pool, records, candidates, lengths)
    }
    for tile in args.tiles:
        fn = functools.partial(
            candidate.candidate_indexer, block_size=8, request_tile=tile
        )

        def invoke(fn=fn):
            return fn(
                split_q, weights.reshape(args.requests, 6, 32), *prepared, lengths
            )

        actual = invoke()
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        variants[f"union_tile{tile}"] = invoke
    variants["prepare"] = lambda: prepare(pool, records, candidates)
    # Four different Q/weights calls; runtime prepare remains in the timed chain.
    raw_qs = [q] + [
        jnp.asarray(rng.normal(size=q.shape) * 0.25, jnp.bfloat16) for _ in range(3)
    ]
    qs = [native["split_halves"](qi) for qi in raw_qs]
    ws = [weights.reshape(args.requests, 6, 32)] + [
        jnp.asarray(rng.normal(size=(args.requests, 6, 32)) * 0.02, jnp.float32)
        for _ in range(3)
    ]

    def baseline_chain():
        return tuple(
            baseline(qi, wi, pool, records, candidates, lengths)
            for qi, wi in zip(raw_qs, ws)
        )

    references = baseline_chain()
    jax.block_until_ready(references)
    variants["four_layers_baseline"] = baseline_chain
    for tile in args.tiles:

        def chain(tile=tile):
            cache = prepare(pool, records, candidates)
            return tuple(
                candidate.candidate_indexer(
                    qi, wi, *cache, lengths, block_size=8, request_tile=tile
                )
                for qi, wi in zip(qs, ws)
            )

        outputs = chain()
        for actual, expected in zip(outputs, references):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        variants[f"four_layers_tile{tile}"] = chain
    samples = {name: [] for name in variants}
    names = list(variants)
    for i in range(args.repeats):
        order = names[i % len(names) :] + names[: i % len(names)]
        if i % 2:
            order.reverse()
        for name in order:
            start = time.perf_counter()
            jax.block_until_ready(variants[name]())
            samples[name].append((time.perf_counter() - start) * 1000)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "metrics.jsonl").open("w") as f:
        for name, values in samples.items():
            row = dict(
                variant=name,
                seed=args.seed,
                requests=args.requests,
                width=args.width,
                capacity=capacity,
                candidate_source=str(args.candidate_file or "synthetic_70pct_core"),
                union_blocks=counts,
                representation="packed" if args.packed else "bf16",
                latency_ms=float(np.median(values)),
                samples_ms=values,
                runtime={
                    p: importlib.metadata.version(p)
                    for p in ("jax", "jaxlib", "libtpu")
                },
            )
            print(json.dumps(row), flush=True)
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
