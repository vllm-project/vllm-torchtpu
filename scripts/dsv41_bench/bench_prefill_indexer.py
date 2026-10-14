# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B the production prefill route, including masks, sorting and dispatch.

Run with the same installed torch-tpu/JAX/libtpu environment as serving.
"""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v41.prefill_index_scores import (
    supports_chunked_prefill,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    paged_indexer_jax,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", type=int, choices=[1, 8], default=1)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--length", type=int, default=50176)
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["baseline", "radix"],
        default=["baseline", "radix"],
    )
    parser.add_argument(
        "--distribution", choices=["random", "ties", "ascending"], default="random"
    )
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assert jax.default_backend() == "tpu"
    test_lengths = args.lengths or [args.length]
    assert all(0 <= n <= args.capacity for n in test_lengths)
    assert args.capacity % 128 == 0
    assert args.repeats >= 1
    assert len(jax.devices()) >= args.devices
    modes = args.modes
    assert modes[0] == "baseline"
    rng = np.random.default_rng(42)
    rows, width, page = 2048, args.capacity, 128
    queries = jnp.asarray(rng.normal(size=(rows, 32, 128)) * 0.25, jnp.bfloat16)
    if args.distribution == "ties":
        queries = jnp.zeros_like(queries)
    weights = jnp.asarray(rng.normal(size=(rows, 32)) * 0.02, jnp.float32)
    keys = jnp.asarray(rng.normal(size=(width, 128)) * 0.25, jnp.bfloat16)
    if args.distribution == "ascending":
        queries = jnp.ones_like(queries)
        weights = jnp.full_like(weights, 1 / 32)
        keys = jnp.broadcast_to(
            (jnp.arange(width, dtype=jnp.float32) / width).astype(jnp.bfloat16)[
                :, None
            ],
            (width, 128),
        )
    words = jax.lax.bitcast_convert_type(keys.reshape(width, 64, 2), jnp.int32)
    words = words.reshape(width // 2, 128)
    split = (width // (2 * page)) * (page // 2)
    pool = (words[:split], words[split:])
    assert supports_chunked_prefill(queries, pool, page)
    order = np.arange(width // page, dtype=np.int32)
    if args.shuffled:
        rng.shuffle(order)
    records = (order[:, None] * page + np.arange(page)).reshape(1, width)
    # Exercise invalid physical pages independently of the causal suffix.
    records[:, 3 * page : 4 * page] = -1
    records = jnp.asarray(records, jnp.int32)
    candidates = jnp.zeros((rows, 0), jnp.int32)
    requests = jnp.zeros(rows, jnp.int32)
    logical_keys = keys.reshape(-1, page, 128)[order].reshape(width, 128)
    check_rows = jnp.asarray([0, 1024, 2043, 2047])
    reference = jnp.einsum(
        "thd,kd->thk",
        queries[check_rows],
        logical_keys,
        preferred_element_type=jnp.float32,
    )
    reference = (jnp.maximum(reference, 0) * weights[check_rows, :, None]).sum(1)
    runtime = {
        name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "libtpu")
    }
    compiled, outputs, compile_seconds = {}, {}, {}
    print("RUNTIME", json.dumps(runtime), flush=True)
    mesh = Mesh(np.asarray(jax.devices()[: args.devices]), ("dp",))
    for mode in modes:

        def run(*xs, mode=mode):
            return paged_indexer_jax(
                *xs, 512, 1, False, False, True, page_records=page, prefill_mode=mode
            )

        fn = jax.jit(
            jax.shard_map(
                run,
                mesh=mesh,
                in_specs=(P(),) * 7,
                out_specs=(P(), P()),
                check_vma=False,
            )
        )
        compiled[mode] = fn

    metrics = []
    memory = {}
    for length in test_lengths:
        lengths = jnp.maximum(
            length - jnp.maximum(rows - 5 - jnp.arange(rows), 0) // 2, 0
        )
        lengths = lengths.at[-4:].set(0)
        inputs = (queries, pool, records, weights, lengths, candidates, requests)
        # Fixture placement is not part of indexer execution. In DP8, starting
        # each call with device-0 arrays measures cache replication as well.
        inputs = jax.device_put(inputs, NamedSharding(mesh, P()))
        jax.block_until_ready(inputs)
        valid = (jnp.arange(width)[None] < lengths[check_rows, None]) & (records >= 0)
        values, ids = jax.lax.top_k(jnp.where(valid, reference, -jnp.inf), 512)
        expected = jnp.sort(jnp.where(values > -jnp.inf, ids, width), axis=1)
        expected = np.asarray(jnp.where(expected == width, -1, expected))
        for mode in modes:
            started = time.perf_counter()
            result = jax.block_until_ready(compiled[mode](*inputs))
            compile_seconds[mode] = time.perf_counter() - started
            if mode not in memory:
                stats = compiled[mode].lower(*inputs).compile().memory_analysis()
                memory[mode] = {
                    "argument_bytes": stats.argument_size_in_bytes,
                    "output_bytes": stats.output_size_in_bytes,
                    "temporary_bytes": stats.temp_size_in_bytes,
                }
                (args.out / "memory.json").write_text(json.dumps(memory, indent=2))
            outputs[mode] = np.asarray(result[0])
            np.testing.assert_array_equal(
                outputs[mode][np.asarray(check_rows)], expected
            )
            np.testing.assert_array_equal(outputs[mode], outputs["baseline"])
            for shard in result[0].addressable_shards:
                np.testing.assert_array_equal(
                    np.asarray(shard.data), outputs["baseline"]
                )
            for _ in range(5):
                jax.block_until_ready(compiled[mode](*inputs))
            print("VERIFIED", mode, length, compile_seconds[mode], flush=True)
        samples = {mode: [] for mode in modes}
        for iteration in range(args.repeats):
            offset = iteration % len(modes)
            for mode in modes[offset:] + modes[:offset]:
                started = time.perf_counter_ns()
                jax.block_until_ready(compiled[mode](*inputs))
                samples[mode].append((time.perf_counter_ns() - started) / 1e6)
        for mode in modes:
            metric = dict(
                variant=mode,
                runtime=runtime,
                timing="synchronized_host_latency",
                input_placement="resident_replicated",
                devices=args.devices,
                tokens_per_device=rows,
                score_capacity=width,
                length=length,
                effective_key_limit=int(lengths.max()),
                shuffled=args.shuffled,
                distribution=args.distribution,
                correct=True,
                compile_seconds=compile_seconds[mode],
                latency_ms=float(np.median(samples[mode])),
                samples_ms=samples[mode],
            )
            metrics.append(metric)
            print(json.dumps(metric), flush=True)
            with (args.out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metric) + "\n")
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if args.profile and length in (50176, 65536, 65537, width):
            for mode in modes:
                with jax.profiler.trace(str(args.out / "xprof" / f"{mode}-{length}")):
                    for _ in range(3):
                        jax.block_until_ready(compiled[mode](*inputs))


if __name__ == "__main__":
    main()
