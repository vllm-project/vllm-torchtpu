# SPDX-License-Identifier: Apache-2.0
"""Benchmark compaction candidates against the frozen production baseline.

CPU mode validates the candidate's unchanged JAX body, not TPU lowering.
TPU mode compiles outside synchronized ABBA and captures isolated 16-call traces.
"""

import argparse
import ast
import contextlib
import functools
import hashlib
import importlib
import inspect
import json
import platform
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest import mock

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHAPES = (96, 192, 240, 256, 288, 384)


def load_kernels():
    """Load actual source modules, bypassing unrelated torch/vLLM startup."""
    for suffix in (
        "",
        ".kernels",
        ".kernels.attention_preprocessing",
        ".kernels.experimental",
    ):
        name = "vllm_torchtpu" + suffix
        module = types.ModuleType(name)
        module.__path__ = [str(ROOT / "src" / name.replace(".", "/"))]
        sys.modules[name] = module
    compress = importlib.import_module(
        "vllm_torchtpu.kernels.attention_preprocessing.compress_rope"
    )
    decode = importlib.import_module(
        "vllm_torchtpu.kernels.experimental.decode_preparation"
    )
    return compress, decode


def reference(records: np.ndarray, capacity: int, tb: int):
    """Independent scalar oracle: stable prefix, zero tail, tile prefix counts."""
    n = len(records)
    assert n % tb == 0
    pairs = np.zeros((n, 2), np.int32)
    offsets = [0]
    count = 0
    for row, record in enumerate(records):
        if 0 <= int(record) < capacity:
            pairs[count] = (row, record)
            count += 1
        if (row + 1) % tb == 0:
            offsets.append(count)
    return pairs.reshape(-1), np.array(offsets, np.int32)


def fixtures(n: int, capacity: int, tb: int) -> dict[str, np.ndarray]:
    empty = np.full(n, -1, np.int32)
    full = np.arange(n, dtype=np.int32) % capacity
    boundary = empty.copy()
    for edge in range(0, n + 1, tb):
        for row in (edge - 1, edge):
            if 0 <= row < n:
                boundary[row] = (n - row) % capacity
    mixed = np.resize(
        np.array([-1, 0, capacity - 1, capacity, -2, capacity + 17, 7, 7], np.int32), n
    )
    padded = full.copy()
    padded[-15:] = -1  # production pads unaligned T to 16 with -1, not zero
    random = np.random.default_rng(n).integers(-2, capacity + 2, n, dtype=np.int32)
    random[np.arange(n) % 3 == 0] = -1
    return {
        "empty": empty,
        "full": full,
        "duplicates": np.full(n, 7, np.int32),
        "out_of_range": np.resize(
            np.array([-1, -2, capacity, capacity + 1], np.int32), n
        ),
        "mixed": mixed,
        "tile_boundary": boundary,
        "padded_tail": padded,
        "random": random,
    }


def check(actual, expected) -> None:
    for got, want in zip(jax.device_get(actual), expected, strict=True):
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, want)


def source_identity(fn) -> dict[str, str]:
    source = inspect.getsource(fn)
    return {"sha256": hashlib.sha256(source.encode()).hexdigest(), "source": source}


def same_function(left, right) -> bool:
    return ast.dump(ast.parse(inspect.getsource(left))) == ast.dump(
        ast.parse(inspect.getsource(right))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "tpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", choices=("actual", "frozen"), default="actual")
    parser.add_argument(
        "--production",
        action="store_true",
        help="compare actual production _compact against frozen baseline",
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--profiles", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.warmup < 1:
        parser.error("rounds and warmup must be positive")
    if jax.default_backend() != args.mode:
        parser.error(f"requested {args.mode}, got {jax.default_backend()}")
    if args.profiles and args.mode != "tpu":
        parser.error("profiles require TPU; CPU mode is correctness only")
    import compact_baseline

    compress, decode = load_kernels()
    unchanged = same_function(compress._compact, compact_baseline._compact)
    if args.production:
        args.baseline = "frozen"
    if args.baseline == "actual" and not unchanged:
        parser.error(
            "production _compact changed; review drift and use --baseline frozen"
        )
    baseline = (
        compress._compact if args.baseline == "actual" else compact_baseline._compact
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": args.mode,
        "baseline": args.baseline,
        "production": args.production,
        "production_matches_frozen": unchanged,
        "jax": jax.__version__,
        "python": platform.python_version(),
        "devices": [str(d) for d in jax.devices()],
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "sources": {
            "baseline": source_identity(baseline),
            "candidate": source_identity(
                compress._compact if args.production else decode.compact_records
            ),
            "compact_records": source_identity(decode.compact_records),
            "prefix_sum": source_identity(decode.prefix_sum),
            "fuse_function": source_identity(decode.fuse_function),
            "default_config": source_identity(compress.default_config),
        },
        "timing": "host dispatch + block_until_ready, ABBA; no profiler",
        "cpu_scope": "production CPU fallback"
        if args.production
        else ("unchanged candidate JAX body, NOT Pallas lowering"),
        "shapes": [],
    }
    # Direct-candidate CPU mode replaces only the Pallas boundary. Production
    # mode keeps real dispatch; the lowering spy below forwards calls unchanged.
    context = (
        mock.patch.object(
            decode, "fuse_function", side_effect=lambda fn, args, **kw: jax.jit(fn)
        )
        if args.mode == "cpu" and not args.production
        else contextlib.nullcontext()
    )
    try:
        with context:
            cases = [(n, compress.default_config(n).token_block) for n in SHAPES]
            if args.production:
                cases += [(288, 288), (128, 128), (384, 384)]
            for n, tb in cases:
                case_id = f"n{n}-tb{tb}"
                capacity = 2 * n
                inputs = fixtures(n, capacity, tb)
                entry = {
                    "n": n,
                    "tb": tb,
                    "tiles": n // tb,
                    "control": n == 288,
                    "fallback_control": args.production
                    and (n, tb) in ((128, 128), (384, 384)),
                    "capacity": capacity,
                    "baseline_path": "compact_records"
                    if n == 288 and args.mode == "tpu"
                    else "scatter",
                    "fixtures": [],
                }
                result["shapes"].append(entry)
                candidate = (
                    functools.partial(compress._compact, tiles=n // tb)
                    if args.production
                    else decode.compact_records
                )
                functions = {
                    "baseline": jax.jit(
                        functools.partial(
                            baseline, total_records=capacity, tiles=n // tb, tb=tb
                        )
                    ),
                    "candidate": jax.jit(
                        functools.partial(candidate, total_records=capacity, tb=tb)
                    ),
                }
                sample = jax.device_put(inputs["mixed"])
                jax.block_until_ready(sample)
                compiled = {}
                for name, fn in functions.items():
                    with mock.patch.object(
                        decode, "compact_records", wraps=decode.compact_records
                    ) as dispatch:
                        lowered = fn.lower(sample)
                    path = (
                        "compact_records"
                        if dispatch.called
                        or (name == "candidate" and not args.production)
                        else "scatter"
                    )
                    entry[f"{name}_path"] = path
                    if args.production:
                        expected_fast = args.mode == "tpu" and (
                            n == 288
                            or (
                                name == "candidate"
                                and (n, tb)
                                in (
                                    (96, 96),
                                    (192, 192),
                                    (240, 240),
                                    (256, 256),
                                    (384, 192),
                                )
                            )
                        )
                        assert dispatch.call_count == int(expected_fast), (
                            case_id,
                            name,
                            path,
                        )
                    compiled[name] = lowered.compile()
                    (args.output / f"{case_id}-{name}.stablehlo.txt").write_text(
                        str(lowered.compiler_ir()), encoding="utf-8"
                    )
                    (args.output / f"{case_id}-{name}.hlo.txt").write_text(
                        compiled[name].as_text(), encoding="utf-8"
                    )
                for label, records in inputs.items():
                    expected = reference(records, capacity, tb)
                    device_records = jax.device_put(records)
                    for fn in compiled.values():
                        check(fn(device_records), expected)
                    entry["fixtures"].append(label)
                # Also validate capacity=0, for which every record must drop.
                for fn in (
                    functools.partial(baseline, total_records=0, tiles=n // tb, tb=tb),
                    functools.partial(candidate, total_records=0, tb=tb),
                ):
                    check(jax.jit(fn)(sample), reference(inputs["mixed"], 0, tb))
                entry["zero_capacity"] = "passed"
                if args.mode == "cpu":
                    continue
                for _ in range(args.warmup):
                    for fn in compiled.values():
                        jax.block_until_ready(fn(sample))
                samples = {name: [] for name in compiled}
                for _ in range(args.rounds):
                    for name in ("baseline", "candidate", "candidate", "baseline"):
                        start = time.perf_counter_ns()
                        jax.block_until_ready(compiled[name](sample))
                        samples[name].append((time.perf_counter_ns() - start) / 1000)
                entry["host_us"] = samples
                entry["median_us"] = {
                    k: statistics.median(v) for k, v in samples.items()
                }
                entry["baseline_over_candidate"] = (
                    entry["median_us"]["baseline"] / entry["median_us"]["candidate"]
                )
                if args.profiles:
                    entry["profiles"] = {}
                    for name, fn in compiled.items():
                        trace_dir = args.output / f"{case_id}-{name}-profile"
                        with jax.profiler.trace(str(trace_dir)):
                            for step in range(16):
                                with jax.profiler.StepTraceAnnotation(
                                    name, step_num=step
                                ):
                                    jax.block_until_ready(fn(sample))
                        entry["profiles"][name] = {
                            "path": str(trace_dir),
                            "samples": 16,
                        }
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "shapes": len(result["shapes"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
