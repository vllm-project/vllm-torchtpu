# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolate sparse MQA gather and logits for field-major and token-major K.

Persistent cache packing and candidate metadata are outside timing. Every variant
uses the same FP8 bits, scales, candidate windows and output-column mapping.
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
from sparse_mqa_layouts import gather_token_major, token_major_mqa_logits

from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PackedFP8Layout,
    quantize_index_rows,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.candidates import (
    gather_candidate_keys,
    prepare_candidate_metadata,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.sparse_logits import (
    candidate_windows,
    fp8_gathered_mqa_logits,
)

VARIANTS = {
    "field": (0, 64, 4),
    "tm33": (33, 64, 4),
    "tm32": (32, 64, 4),
    "tm33-g128": (33, 128, 2),
    "tm32-g128": (32, 128, 2),
    "tm33-sc64": (33, 64, 2),
    "tm33-sc32": (33, 32, 3),
    "tm32-d1": (32, 64, 4),
    "tm32-loop4k": (32, 64, 4),
    "tm32-loop8k": (32, 64, 4),
    "tm32-byte4k": (32, 64, 4),
    "tm32-byte-static": (32, 64, 4),
    "tm4-native": (32, 64, 4),
    "tm4-loop4k": (32, 64, 4),
    "tm32-lanes4k": (32, 64, 4),
    "tm32-lanes": (32, 64, 4),
    "tm32-butterfly": (32, 64, 4),
    "tm32-butterfly-d4": (32, 64, 4),
    "tm32-pack": (32, 64, 4),
    "tm4-fused-control": (33, 64, 4),
}


def measure(name, fn, inputs, root, repeats):
    started = time.perf_counter()
    executable = jax.jit(fn).lower(*inputs).compile()
    compile_s = time.perf_counter() - started
    hlo = root / "rank-0/compiler/hlo"
    hlo.mkdir(parents=True, exist_ok=True)
    (hlo / f"{name}.txt").write_text(executable.as_text())
    for _ in range(3):
        jax.block_until_ready(executable(*inputs))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        jax.block_until_ready(executable(*inputs))
        samples.append((time.perf_counter_ns() - started) / 1000)
    profile = root / "rank-0/profiling/xprof" / name
    with jax.profiler.trace(str(profile)):
        for _ in range(5):
            jax.block_until_ready(executable(*inputs))
    device = []
    for path in profile.rglob("*.xplane.pb"):
        plane = next(
            p
            for p in ProfileData.from_file(str(path)).planes
            if p.name == "/device:TPU:0"
        )
        for line in plane.lines:
            if line.name == "XLA Modules":
                device.extend(e.duration_ns / 1000 for e in line.events)
    metric = dict(
        variant=name,
        latency_ms=statistics.median(device) / 1000,
        device_median_us=statistics.median(device),
        device_samples_us=device,
        host_median_us=statistics.median(samples),
        host_samples_us=samples,
        compile_time_s=compile_s,
    )
    with (root / "rank-0/benchmark/metrics.jsonl").open("a") as f:
        f.write(json.dumps(metric) + "\n")
    print(json.dumps(metric), flush=True)
    return executable(*inputs)


@jax.jit
def pack(keys):
    values, scales, _ = quantize_index_rows(keys)
    bits = (
        jax.lax.bitcast_convert_type(values, jnp.uint8)
        .astype(jnp.uint32)
        .reshape(-1, 32, 4)
    )
    words = (
        bits[..., 0] | (bits[..., 1] << 8) | (bits[..., 2] << 16) | (bits[..., 3] << 24)
    )
    records = jnp.concatenate(
        (
            words.astype(jnp.int32),
            jax.lax.bitcast_convert_type(scales, jnp.int32)[:, None],
        ),
        axis=1,
    )
    field = records.reshape(-1, 128, 33).transpose(0, 2, 1).reshape(-1, 128)
    grouped = bits.reshape(-1, 4, 128)
    native = (
        grouped[:, 0]
        | (grouped[:, 1] << 8)
        | (grouped[:, 2] << 16)
        | (grouped[:, 3] << 24)
    )
    return (
        field,
        records.reshape(-1, 128),
        words.astype(jnp.int32).reshape(-1, 128),
        records[:, 32].reshape(-1, 128),
        native.astype(jnp.int32),
    )


def check_scores(actual, expected):
    actual = np.asarray(actual)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=0.01, atol=0.01)
    return float(np.max(np.abs(actual[finite] - expected[finite]), initial=0))


def check_words(packed, field, fields, physical, repack=False, native=False):
    live = np.flatnonzero(np.asarray(physical).reshape(-1) >= 0)
    slots = live[:: max(1, len(live) // 4096)]
    tokens = jnp.asarray((slots[:, None] * 8 + np.arange(8)).reshape(-1))
    expected = field.reshape(-1, 33, 128)[tokens // 128, :, tokens % 128]
    if native:
        bits = (
            packed[0][tokens // 4].astype(jnp.uint32) >> ((tokens % 4) * 8)[:, None]
            & 255
        )
        bits = bits.reshape(-1, 32, 4)
        words = (
            bits[..., 0]
            | (bits[..., 1] << 8)
            | (bits[..., 2] << 16)
            | (bits[..., 3] << 24)
        )
        actual = jnp.concatenate(
            (words.astype(jnp.int32), packed[1].reshape(-1)[tokens, None]), axis=1
        )
    elif repack:
        actual = packed[0].reshape(-1, 33, 128)[tokens // 128, :, tokens % 128]
    elif fields == 33:
        actual = packed[0].reshape(-1, 33)[tokens]
    else:
        actual = jnp.concatenate(
            (packed[0].reshape(-1, 32)[tokens], packed[1].reshape(-1)[tokens, None]),
            axis=1,
        )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--edge-case", choices=("offsets", "singleton"))
    parser.add_argument("--variants", default="field,tm4-native,tm32-lanes")
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    root = args.out
    (root / "rank-0/benchmark").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(826)
    requests, queries, width, count = (
        (4, 2, 4096, 128) if args.small else (48, 6, 98816, 2048)
    )
    if args.edge_case:
        requests, queries, width, count = (
            (16, 2, 4096, 128) if args.edge_case == "offsets" else (1, 1, 4096, 128)
        )
    tokens = requests * queries
    layout = PackedFP8Layout(requests * width // 128)
    pages = jnp.asarray(rng.permutation(layout.pages).reshape(requests, -1), jnp.int32)
    owners = jnp.repeat(jnp.arange(requests, dtype=jnp.int32), queries)
    candidates = []
    for request in range(requests):
        common = rng.choice(width // 8, int(count * 0.7), replace=False)
        for query in range(queries):
            extra = rng.choice(width // 8, count - len(common), replace=False)
            candidates.append(
                np.arange(count) + request * query
                if args.edge_case == "offsets"
                else np.concatenate((common, extra))
            )
    metadata = prepare_candidate_metadata(
        pages, jnp.asarray(np.array(candidates), jnp.int32), owners, layout, 8, width
    )
    lengths = width - jnp.tile(jnp.arange(queries - 1, -1, -1), requests)
    if (args.small or args.edge_case) and tokens > 4:
        lengths = lengths.at[2].set(0).at[4].set(width // 3)
    windows = candidate_windows(
        metadata.candidate_block_ids, metadata.block_spans, lengths, 8
    )
    caches = pack(
        jax.random.normal(jax.random.key(826), (layout.pages * 128, 128), jnp.bfloat16)
    )
    caches = tuple(tuple(jnp.split(cache, 2)) for cache in caches)
    q = jax.random.normal(jax.random.key(827), (tokens, 32, 128)).astype(
        jnp.float8_e4m3fn
    )
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32)
    dimensions = dict(
        tokens=tokens,
        requests=requests,
        queries=queries,
        context=width,
        candidate_blocks=count,
        block_size=8,
        page_size=128,
        overlap=0.7,
        heads=32,
        dim=128,
        bytes_per_token=132,
        banks=2,
        edge_case=args.edge_case,
    )
    (root / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                workflow="operator-optimization",
                operator_family="indexer",
                operator_name="sparse_mqa_layouts",
                dimensions=dimensions,
                source_commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                input_kind="synthetic seeded, persistent cache packing excluded",
                timing="five XPlane module samples after warmed host runs",
            ),
            indent=2,
        )
    )
    field_gather = functools.partial(gather_candidate_keys, layout=layout, block=8)
    field_score = functools.partial(
        fp8_gathered_mqa_logits,
        query_start_loc=metadata.offsets,
        output_width=count * 8,
    )
    field = jax.jit(field_gather)(
        caches[0], metadata.physical_block_ids, metadata.offsets
    )
    expected = np.asarray(field_score(q, field, weights, windows))

    def repacked_score(q, packed, weights, windows):
        return field_score(q, packed[0], weights, windows)

    validation = {}
    for name in args.variants.split(","):
        fields, group, buffers = VARIANTS[name]
        repack = name.startswith("tm33-sc")
        native = name.startswith("tm4-")
        fused = name == "tm4-fused-control"
        if fields:
            pools = (
                (caches[1],)
                if fields == 33
                else (caches[4] if native else caches[2], caches[3])
            )
            gather = functools.partial(
                gather_token_major,
                fields=fields,
                group=group,
                buffers=buffers,
                repack=repack,
            )
            score = functools.partial(
                token_major_mqa_logits,
                fields=fields,
                query_start_loc=metadata.offsets,
                output_width=count * 8,
                issue_depth={
                    "tm32-d1": 1,
                    "tm32-loop4k": 4,
                    "tm32-loop8k": 8,
                    "tm32-byte4k": 4,
                    "tm4-loop4k": 4,
                    "tm32-lanes4k": 4,
                    "tm32-butterfly-d4": 4,
                }.get(name, 16),
                loop_tile={
                    "tm32-loop4k": 4096,
                    "tm32-loop8k": 8192,
                    "tm32-byte4k": 4096,
                    "tm4-loop4k": 4096,
                    "tm32-lanes4k": 4096,
                }.get(name, 0),
                byte_transpose=name.startswith("tm32-byte"),
                native_packing=native,
                lane_repack=(
                    "butterfly"
                    if name.startswith("tm32-butterfly")
                    else "pack"
                    if name == "tm32-pack"
                    else "direct"
                    if name.startswith("tm32-lanes")
                    else ""
                ),
            )
            if repack:
                score = repacked_score
            if fused:
                pools = caches[4], caches[3]

                def gather(pool, ids, offsets, group=group, buffers=buffers):
                    k, scale = gather_token_major(
                        pool, ids, offsets, fields=32, group=group, buffers=buffers
                    )
                    return (
                        jnp.concatenate(
                            (k.reshape(-1, 32, 128), scale[:, None, :]), axis=1
                        ).reshape(-1, 128),
                    )
        else:
            pools, gather, score = caches[0], field_gather, field_score
        packed = measure(
            name + ".gather",
            gather,
            (pools, metadata.physical_block_ids, metadata.offsets),
            root,
            args.repeats,
        )
        if fields:
            checked = packed
            if fused:
                pages = packed[0].reshape(-1, 33, 128)
                checked = pages[:, :32].reshape(-1, 128), pages[:, 32]
            check_words(
                checked, field, fields, metadata.physical_block_ids, repack, native
            )
        scores = measure(
            name + ".logits", score, (q, packed, weights, windows), root, args.repeats
        )
        error = check_scores(scores, expected)

        def sparse_mqa(
            pool, q, weights, windows, ids, offsets, score=score, gather=gather
        ):
            return score(q, gather(pool, ids, offsets), weights, windows)

        combined = measure(
            name + ".sparse_mqa",
            sparse_mqa,
            (pools, q, weights, windows, metadata.physical_block_ids, metadata.offsets),
            root,
            args.repeats,
        )
        error = max(error, check_scores(combined, expected))
        validation[name] = dict(
            packed_words="exact",
            finite_mask="exact",
            logits_max_abs_error=error,
            output_shape=list(scores.shape),
            member_b_page_offsets=np.unique(
                np.asarray(windows)[1::2, 0] % 128
            ).tolist(),
            group_sizes=np.unique(np.diff(np.asarray(metadata.offsets))).tolist(),
        )
        (root / "correctness.json").write_text(json.dumps(validation, indent=2))
        del packed, scores, combined
    print("SPARSE_MQA_LAYOUT_BENCHMARK_PASSED", flush=True)


if __name__ == "__main__":
    main()
