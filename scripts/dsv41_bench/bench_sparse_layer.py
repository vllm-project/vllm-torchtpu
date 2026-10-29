# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One candidate-reuse indexer layer, without model weights or a serving engine.

Synthetic per-query candidates preserve the serving geometry but are not a
capture of model activations. Measure preparation and scoring separately, then
measure their compiled composition. Physical cache pages contain 128 tokens;
the eight-token candidate fragments address subsets of those pages.
"""

import argparse
import functools
import json
import statistics
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.profiler import ProfileData

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PackedFP8Layout,
    quantize_index_rows,
    read_fp8_records,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.candidates import (
    PreparedCandidates,
    gather_candidate_keys,
    prepare_candidate_metadata,
    prepare_candidates,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import indexer_impl
from vllm_torchtpu.models.vllm.deepseek_v41.v2.indexer_impl import _score_prepared


def pack_fixture(keys, layout):
    """Initialize pages without the incremental writer's SMEM index limit."""
    values, scales, _ = quantize_index_rows(keys)
    bits = jax.lax.bitcast_convert_type(values, jnp.uint8).astype(jnp.uint32)
    bits = bits.reshape(-1, 32, 4)
    words = bits[..., 0] | (bits[..., 1] << 8)
    words |= (bits[..., 2] << 16) | (bits[..., 3] << 24)
    records = jnp.concatenate(
        (
            words.astype(jnp.int32),
            jax.lax.bitcast_convert_type(scales, jnp.int32)[:, None],
        ),
        axis=1,
    )
    pages = records.reshape(layout.pages, layout.page_size, 33).transpose(0, 2, 1)
    pages = pages.reshape(layout.pages, -1)
    return jnp.pad(
        pages, ((0, 0), (0, layout.page_rows * 128 - pages.shape[1]))
    ).reshape(-1, 128)


def measure(name, fn, inputs, root, repeats, dimensions):
    started = time.perf_counter()
    lowered = jax.jit(fn).lower(*inputs)
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - started
    hlo = root / "rank-0/compiler/hlo"
    hlo.mkdir(parents=True, exist_ok=True)
    (hlo / f"{name}.txt").write_text(executable.as_text())
    for _ in range(3):
        jax.block_until_ready(executable(*inputs))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        jax.block_until_ready(executable(*inputs))
        samples.append((time.perf_counter_ns() - started) / 1e6)
    metric = dict(
        dimensions,
        variant=name,
        latency_ms=statistics.median(samples),
        sample_statistics={"min_ms": min(samples), "max_ms": max(samples)},
        samples_ms=samples,
        compile_time_s=compile_seconds,
    )
    with (root / "rank-0/benchmark/metrics.jsonl").open("a") as out:
        out.write(json.dumps(metric) + "\n")
    print(json.dumps(metric), flush=True)
    profile = root / "rank-0/profiling/xprof" / name
    profile.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(profile)):
        for _ in range(5):
            jax.block_until_ready(executable(*inputs))
    device_samples = []
    for path in profile.rglob("*.xplane.pb"):
        data = ProfileData.from_file(str(path))
        plane = next(p for p in data.planes if p.name == "/device:TPU:0")
        for line in plane.lines:
            if line.name == "XLA Modules":
                device_samples.extend(e.duration_ns / 1000 for e in line.events)
    device_metric = dict(
        variant=name,
        device_median_us=statistics.median(device_samples),
        device_samples_us=device_samples,
    )
    with (root / "rank-0/benchmark/device.jsonl").open("a") as out:
        out.write(json.dumps(device_metric) + "\n")
    print(json.dumps(device_metric), flush=True)
    return executable(*inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--queries-per-request", type=int, default=6)
    parser.add_argument("--width", type=int, default=98816)
    parser.add_argument("--candidates", type=int, default=2048)
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.7)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if not 0 <= args.overlap <= 1 or args.width % args.page_size:
        raise ValueError("Use overlap in [0,1] and a page-aligned width")
    if args.queries_per_request % 2:
        raise ValueError("This paired-candidate benchmark requires even query counts")
    root = args.out
    (root / "rank-0/benchmark").mkdir(parents=True, exist_ok=True)
    dimensions = dict(
        requests=args.requests,
        queries_per_request=args.queries_per_request,
        tokens=args.requests * args.queries_per_request,
        seq_len=args.width,
        candidate_blocks=args.candidates,
        block_size=args.block,
        page_size=args.page_size,
        overlap=args.overlap,
        topk=args.topk,
    )
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                workflow="operator-optimization",
                operator_family="indexer",
                operator_name="paired_candidate_layer",
                dimensions=dimensions,
                source_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                input_kind="synthetic, fixed seed; not captured model activations",
                timing="compiled warm runs; host dispatch plus device completion",
            ),
            indent=2,
        )
    )
    rng = np.random.default_rng(826)
    tokens = dimensions["tokens"]
    pages_per_request = args.width // args.page_size
    layout = PackedFP8Layout(
        args.requests * pages_per_request, page_size=args.page_size
    )
    pages = jnp.asarray(rng.permutation(layout.pages).reshape(args.requests, -1))
    owners = jnp.repeat(
        jnp.arange(args.requests, dtype=jnp.int32), args.queries_per_request
    )
    candidate_rows = []
    for _ in range(args.requests):
        common = rng.choice(
            args.width // args.block, int(args.candidates * args.overlap), replace=False
        )
        for _ in range(args.queries_per_request):
            extra = rng.choice(
                args.width // args.block, args.candidates - len(common), replace=False
            )
            candidate_rows.append(np.concatenate((common, extra)))
    candidates = jnp.asarray(np.array(candidate_rows), jnp.int32)
    metadata = measure(
        "prepare_metadata",
        functools.partial(
            prepare_candidate_metadata,
            layout=layout,
            block=args.block,
            width=args.width,
        ),
        (pages, candidates, owners),
        root,
        args.repeats,
        dimensions,
    )
    host_candidates = np.asarray(candidates)
    host_ids = np.asarray(metadata.block_ids)
    host_physical = np.asarray(metadata.physical)
    flags = np.asarray(metadata.block_owners)
    for group in range(tokens // 2):
        first = group * 2
        union = np.unique(host_candidates[first : first + 2])
        np.testing.assert_array_equal(host_ids[first][host_ids[first] >= 0], union)
        for query in range(2):
            actual = (flags[group] & (1 << query)) != 0
            expected = np.isin(host_ids[first], host_candidates[first + query]) & (
                host_ids[first] >= 0
            )
            np.testing.assert_array_equal(actual, expected)
    keys = jax.random.normal(
        jax.random.key(826), (layout.pages * layout.page_size, 128), jnp.bfloat16
    )
    pool = jax.jit(functools.partial(pack_fixture, layout=layout))(keys)
    banks = tuple(jnp.split(pool, 2))
    jax.block_until_ready(banks)
    del pool, keys
    q = jax.random.normal(jax.random.key(827), (tokens, 32, 128)).astype(
        jnp.float8_e4m3fn
    )
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32)
    lengths = jnp.asarray(
        args.width
        - np.tile(np.arange(args.queries_per_request - 1, -1, -1), args.requests),
        jnp.int32,
    )
    gather = functools.partial(gather_candidate_keys, block=args.block, layout=layout)
    run = functools.partial(
        measure, root=root, repeats=args.repeats, dimensions=dimensions
    )
    gathered = run("gather", gather, (banks, metadata.physical, metadata.offsets))
    prepare = functools.partial(
        prepare_candidates, layout=layout, block=args.block, width=args.width
    )
    run("metadata_gather", prepare, (banks, pages, candidates, owners))
    slots = np.flatnonzero(host_physical.reshape(-1) >= 0)[::97]
    target = (slots[:, None] * args.block + np.arange(args.block)).reshape(-1)
    source = (
        host_physical.reshape(-1)[slots, None] * args.block + np.arange(args.block)
    ).reshape(-1)
    np.testing.assert_array_equal(
        np.asarray(
            read_fp8_records(
                (gathered,),
                jnp.asarray(target),
                PackedFP8Layout(gathered.shape[0] // 33),
            )
        ),
        np.asarray(read_fp8_records(banks, jnp.asarray(source), layout)),
    )
    prepared = PreparedCandidates(gathered, *metadata[1:])
    compact_lengths = jnp.sum(
        jnp.where(
            prepared.block_ids >= 0,
            jnp.clip(lengths[:, None] - prepared.block_ids * args.block, 0, args.block),
            0,
        ),
        axis=1,
    )

    def logits(q, weights, keys, compact_lengths, block_owners, offsets):
        return indexer_impl.fp8_gathered_mqa_logits(
            q,
            keys,
            weights,
            compact_lengths,
            block_owners,
            query_start_loc=offsets,
            membership_block_size=args.block,
        )

    raw_scores = run(
        "logits",
        logits,
        (
            q,
            weights,
            gathered,
            compact_lengths,
            prepared.block_owners,
            prepared.offsets,
        ),
    )
    topk = functools.partial(
        indexer_impl._select_candidate_tokens,
        k=args.topk,
        block_size=args.block,
        unordered=True,
    )
    split = run("topk", topk, (raw_scores, prepared.block_ids))
    score = functools.partial(
        _score_prepared,
        topk=args.topk,
        block=args.block,
        unordered=True,
        width=args.width,
    )
    selected = run("logits_topk", score, (q, weights, lengths, prepared))

    def gather_score(q, weights, lengths, banks, metadata):
        keys = gather(banks, metadata.physical, metadata.offsets)
        return score(q, weights, lengths, PreparedCandidates(keys, *metadata[1:]))

    combined = run(
        "gather_logits_topk", gather_score, (q, weights, lengths, banks, metadata)
    )

    def layer(q, weights, lengths, banks, pages, candidates, owners):
        prepared = prepare_candidates(
            banks, pages, candidates, owners, layout, args.block, args.width
        )
        return score(q, weights, lengths, prepared)

    full = run(
        "metadata_gather_logits_topk",
        layer,
        (q, weights, lengths, banks, pages, candidates, owners),
    )
    for result in (split, combined, full):
        np.testing.assert_array_equal(
            np.sort(np.asarray(selected), axis=1), np.sort(np.asarray(result), axis=1)
        )
    scores_host = np.asarray(raw_scores)
    ids_host = np.asarray(selected)
    for row in range(tokens):
        chosen = ids_host[row]
        assert len(np.unique(chosen)) == args.topk
        assert np.all(chosen >= 0) and np.all(chosen < int(lengths[row]))
        assert np.all(np.isin(chosen // args.block, host_candidates[row]))
        slots = (
            np.searchsorted(host_ids[row][host_ids[row] >= 0], chosen // args.block)
            * args.block
            + chosen % args.block
        )
        threshold = np.partition(scores_host[row], -args.topk)[-args.topk]
        assert np.min(scores_host[row, slots]) >= threshold
    (root / "correctness.json").write_text(
        json.dumps(
            {
                "packed_ownership_numpy": "passed",
                "sampled_gather_readback": "passed",
                "split_vs_combined_topk": "passed",
                "topk_numpy_threshold_and_candidate_membership": "passed",
            },
            indent=2,
        )
    )
    print("SPARSE_LAYER_BENCHMARK_PASSED", flush=True)


if __name__ == "__main__":
    with jax.default_device(jax.devices("tpu")[0]):
        main()
