# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone request-major decode score/topk A/B; no serving registration.

The baseline loads the unchanged JAX scorer/cache helpers from the checkout's
AST. Only the torch/vLLM package startup and single-device bridge are bypassed.
Both paths include split-order conversion, masking and sorted-index adapters.
"""

import argparse
import ast
import functools
import importlib.metadata
import json
import logging
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np


def load_kernels(root: Path):
    for name in ("vllm_torchtpu", "vllm_torchtpu.kernels"):
        package = types.ModuleType(name)
        package.__path__ = [str(root / "src" / name.replace(".", "/"))]
        sys.modules[name] = package
    # Standalone runs bypass vLLM logging as well as package registration.
    logger_module = types.ModuleType("vllm_torchtpu.logger")
    logger = logging.getLogger("decode_benchmark")
    logger.warning_once = functools.lru_cache(None)(logger.warning)
    logger_module.init_logger = lambda name: logger
    sys.modules[logger_module.__name__] = logger_module
    from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk
    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import packed_scores
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.decode import (
        decode_indexer,
    )
    from vllm_torchtpu.kernels.deepseek_v41.index_selection.decode_scores import (
        grouped_decode_scores,
    )

    namespace = dict(
        jax=jax,
        jnp=jnp,
        ROW_WORDS=128,
        index_key_layout=index_key_layout,
        packed_scores=packed_scores,
    )
    source = root / "src/vllm_torchtpu/models/vllm/deepseek_v41/v2"
    cache = ast.parse((source / "cache.py").read_text())
    helpers = [
        node
        for node in cache.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_cache_num_rows",
            "_gather_raw_rows",
            "_permute",
            "piece_order",
            "split_halves",
            "gather_cache_page_words",
        }
    ]
    exec(
        compile(
            ast.Module(body=helpers, type_ignores=[]), str(source / "cache.py"), "exec"
        ),
        namespace,
    )
    indexer = ast.parse((source / "indexer_impl.py").read_text())
    definitions = [
        node
        for node in indexer.body
        if (isinstance(node, ast.FunctionDef) and node.name == "paged_index_scores_jax")
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id.startswith("_INDEX_")
                for t in node.targets
            )
        )
    ]
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(source / "indexer_impl.py"),
            "exec",
        ),
        namespace,
    )
    namespace["_score_word_rows"] = packed_scores.score_word_rows
    return namespace, sparsecore_topk, decode_indexer, grouped_decode_scores


def fixture(
    requests, queries, width, length, page, shuffled, distribution, banks, seed
):
    rng = np.random.default_rng(seed)
    # Distinct physical storage per request; no small repeated-cache surrogate.
    bits = rng.integers(0, 65536, (requests * width, 128), dtype=np.uint16)
    bits &= np.uint16(0x807F)
    bits |= np.uint16(0x3E00)
    keys = bits.view(ml_dtypes.bfloat16)
    from vllm_torchtpu.kernels.deepseek_v41 import index_key_layout

    encoded = index_key_layout.encode_records(jnp.asarray(keys))
    words = (encoded[::2] | (encoded[1::2] << 16)).astype(jnp.int32)
    del bits
    q = rng.standard_normal((requests, queries, 32, 128), dtype=np.float32) * 0.25
    w = rng.standard_normal((requests, queries, 32), dtype=np.float32) * 0.02
    if distribution == "ties":
        q.fill(0)
    order = np.arange(requests * width // page, dtype=np.int32).reshape(requests, -1)
    if shuffled:
        for row in order:
            rng.shuffle(row)
    pages = order * page
    pages[0, 3] = -1
    lengths = (
        np.broadcast_to(length - np.arange(queries)[::-1], (requests, queries))
        .copy()
        .astype(np.int32)
    )
    lengths[-1, -1] = 0
    records = (pages[:, :, None] + np.arange(page, dtype=np.int32)).reshape(
        requests, width
    )
    records = np.where(np.repeat(pages < 0, page, axis=1), -1, records)
    inputs = (
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(w),
        jnp.asarray(words),
        jnp.asarray(records),
        jnp.asarray(lengths),
    )
    if banks == 2:
        pool = inputs[2]
        split = (pool.shape[0] // (page // 2) // 2) * (page // 2)
        inputs = (*inputs[:2], (pool[:split], pool[split:]), *inputs[3:])
    checks = [
        0,
        min(7, requests * queries - 1),
        requests * queries - 2,
        requests * queries - 1,
    ]
    expected = []
    flat_q, flat_w = np.asarray(inputs[0]).reshape(-1, 32, 128), w.reshape(-1, 32, 1)
    # Independent FP32 dense reference, only four rows to bound host work.
    for t in checks:
        physical = records[t // queries]
        k = keys[np.maximum(physical, 0)].astype(np.float32)
        logits = flat_q[t].astype(np.float32) @ k.T
        s = (np.maximum(logits, 0) * flat_w[t]).sum(0)
        valid = (np.arange(width) < lengths.reshape(-1)[t]) & (physical >= 0)
        s[~valid] = -np.inf
        ids = np.argsort(-s, kind="stable")[:512]
        ids = np.sort(np.where(np.isfinite(s[ids]), ids, width))
        expected.append(np.where(ids < width, ids, -1))
    return inputs, checks, np.asarray(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=65536)
    parser.add_argument("--length", type=int, default=49408)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--page", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument(
        "--modes", nargs="+", default=["baseline", "serial8", "overlap8"]
    )
    parser.add_argument("--shuffled", action="store_true")
    parser.add_argument("--banks", type=int, choices=[1, 2], default=2)
    parser.add_argument("--distribution", choices=["random", "ties"], default="random")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--variants", type=Path, help="JSON mapping names to decode kwargs"
    )
    parser.add_argument("--seed", type=int, default=517)
    parser.add_argument(
        "--tune-request-tiles",
        nargs="+",
        type=int,
        help="Sweep only request_tile; write a suggested Python table entry",
    )
    args = parser.parse_args()
    if args.tune_request_tiles and args.variants:
        parser.error("--tune-request-tiles and --variants are mutually exclusive")
    if args.repeats < 1 or args.requests < 1:
        parser.error("requests and repeats must be positive")
    variants = json.loads(args.variants.read_text()) if args.variants else {}
    if args.tune_request_tiles:
        if any(t < 1 for t in args.tune_request_tiles):
            parser.error("Every request_tile must be positive")
        variants = {f"tile{t}": {"request_tile": t} for t in args.tune_request_tiles}
    if variants:
        args.modes = list(variants)
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=True)
    ns, old_topk, decode, _ = load_kernels(root)
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import tuned_block_sizes

    runtime = {
        name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "libtpu")
    }
    print("RUNTIME", runtime, jax.devices(), flush=True)
    inputs, checks, expected = fixture(
        args.requests,
        args.queries,
        args.width,
        args.length,
        args.page,
        args.shuffled,
        args.distribution,
        args.banks,
        args.seed,
    )
    jax.block_until_ready(inputs)

    def baseline(q, w, pool, records, lengths):
        rows = q.shape[0] * q.shape[1]
        q, w, lengths = (
            q.reshape(rows, 32, 128),
            w.reshape(rows, 32),
            lengths.reshape(-1),
        )
        request_ids = jnp.repeat(
            jnp.arange(records.shape[0], dtype=jnp.int32), args.queries
        )
        scores = ns["paged_index_scores_jax"](
            q, pool, records, w, lengths, request_ids, args.page
        )
        valid = (jnp.arange(args.width)[None] < lengths[:, None]) & (
            records[request_ids] >= 0
        )
        scores = jnp.where(valid, scores, -jnp.inf)
        selected = old_topk(scores, k=512, row_lengths=lengths)
        ordered = jnp.sort(jnp.where(selected >= 0, selected, args.width), axis=1)
        return jnp.where(ordered < args.width, ordered, -1)

    def candidate(q, w, pool, records, lengths, *, tile, overlap, **options):
        q = ns["split_halves"](q)
        return decode(
            q,
            w,
            pool,
            records[:, :: args.page],
            lengths,
            page_records=args.page,
            request_tile=tile,
            overlap=overlap,
            **options,
        )

    compiled, outputs, compilation, memory = {}, {}, {}, {}
    effective_configs = {}
    for mode in args.modes:
        if mode == "baseline":
            fn = baseline
            effective_configs[mode] = None
        else:
            if mode in variants:
                options = dict(variants[mode])
                tile = options.pop("request_tile", None)
                overlap = options.pop("overlap", True)
            else:
                options = {}
                overlap = mode.startswith("overlap")
                tile = int(mode.removeprefix("overlap").removeprefix("serial"))
            score_options = options.pop("score_config", None)
            if score_options is not None and tile is None:
                parser.error("score_config overrides require an explicit request_tile")
            score_config = (
                tuned_block_sizes.get_persistent_score_config(
                    (args.queries, 32, 128, args.width, args.banks)
                )
                if score_options is None
                else tuned_block_sizes.PersistentScoreConfig(**score_options)
            )
            if tile is None:
                tile = tuned_block_sizes.get_decode_request_tile(
                    tuned_block_sizes.DecodeTuningKey(
                        jax.devices()[0].device_kind,
                        args.queries,
                        32,
                        args.width,
                        args.page,
                        options.get("k", 512),
                        args.banks,
                    ),
                    args.requests,
                )
            tile = min(tile, args.requests)
            effective_configs[mode] = dict(
                request_tile=tile,
                overlap=overlap,
                score_config=asdict(score_config),
                **options,
            )
            fn = functools.partial(
                candidate,
                tile=tile,
                overlap=overlap,
                score_config=score_config,
                **options,
            )
        t = time.perf_counter()
        executable = jax.jit(fn).lower(*inputs).compile()
        compilation[mode] = time.perf_counter() - t
        result = jax.block_until_ready(executable(*inputs))
        outputs[mode] = np.asarray(result)
        np.testing.assert_array_equal(outputs[mode][checks], expected)
        if mode != args.modes[0]:
            np.testing.assert_array_equal(outputs[mode], outputs[args.modes[0]])
        stats = executable.memory_analysis()
        memory[mode] = dict(
            temporary_bytes=stats.temp_size_in_bytes,
            argument_bytes=stats.argument_size_in_bytes,
            output_bytes=stats.output_size_in_bytes,
        )
        compiled[mode] = executable
        for _ in range(5):
            jax.block_until_ready(executable(*inputs))
        print("VERIFIED", mode, compilation[mode], memory[mode], flush=True)
    samples = {mode: [] for mode in args.modes}
    for iteration in range(args.repeats):
        offset = iteration % len(args.modes)
        for mode in args.modes[offset:] + args.modes[:offset]:
            t = time.perf_counter_ns()
            jax.block_until_ready(compiled[mode](*inputs))
            samples[mode].append((time.perf_counter_ns() - t) / 1e6)
    for mode in args.modes:
        metric = dict(
            variant=mode,
            runtime=runtime,
            timing="synchronized_host_latency",
            devices=1,
            requests=args.requests,
            queries=args.queries,
            score_capacity=args.width,
            length=args.length,
            shuffled=args.shuffled,
            banks=args.banks,
            distribution=args.distribution,
            seed=args.seed,
            device_kind=jax.devices()[0].device_kind,
            correct=True,
            latency_ms=float(np.median(samples[mode])),
            samples_ms=samples[mode],
            compile_seconds=compilation[mode],
            config=variants.get(mode, {}),
            effective_config=effective_configs[mode],
            **memory[mode],
        )
        with (args.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)
    if args.tune_request_tiles:
        winner = min(samples, key=lambda mode: np.median(samples[mode]))
        key = tuned_block_sizes.DecodeTuningKey(
            jax.devices()[0].device_kind,
            args.queries,
            32,
            args.width,
            args.page,
            512,
            args.banks,
        )
        tile = variants[winner]["request_tile"]
        suggestion = (
            f"# seed={args.seed}; effective_length={args.length}; "
            f"shuffled={args.shuffled}; runtime={runtime!r}\n"
            f"# Verify the winner on a separate run before installing.\n"
            f"{key!r}: {tile},\n"
        )
        (args.out / "tuned_request_tile.py.txt").write_text(suggestion)
        print("TUNING_WINNER", winner, flush=True)
    if args.profile:
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 2
        options.python_tracer_level = 0
        options.advanced_configuration = dict(
            tpu_num_chips_to_profile_per_task=1,
            tpu_num_sparse_cores_to_trace=2,
            tpu_num_sparse_core_tiles_to_trace=1,
        )
        for mode, executable in compiled.items():
            with jax.profiler.trace(
                str(args.out / "xprof" / mode), profiler_options=options
            ):
                for step in range(5):
                    with jax.profiler.StepTraceAnnotation(mode, step_num=step):
                        jax.block_until_ready(executable(*inputs))


if __name__ == "__main__":
    main()
