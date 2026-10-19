# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-runtime baseline/candidate route A/B, including sorted-index adapters.

The baseline is an explicit checkout with its own scorer and tuning modules.
Shared score/top-k helpers must match. Serving registration is bypassed using
the existing standalone loader.
"""

import argparse
import functools
import importlib.metadata
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
from bench_decode_overlap import fixture, load_kernels


def load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", default=25, type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    native, _, _, _ = load_kernels(root)
    from vllm_torchtpu.kernels.deepseek_v41.index_selection import candidate, decode

    relative = Path("src/vllm_torchtpu/kernels/deepseek_v41/index_selection")
    for name in ("scores.py", "streaming_topk.py", "union.py"):
        assert (root / relative / name).read_bytes() == (
            args.baseline_root / relative / name
        ).read_bytes(), name
    old_env = load(
        "route_baseline_env", args.baseline_root / "src/vllm_torchtpu/envs.py"
    )
    old_candidate = load(
        "route_baseline_candidate", args.baseline_root / relative / "candidate.py"
    )
    old_decode = load(
        "route_baseline_decode", args.baseline_root / relative / "decode.py"
    )
    old_candidate.envs = old_decode.envs = old_env
    old_candidate.decode = old_decode
    old_candidate.tuned_block_sizes = load(
        "route_baseline_tuning", args.baseline_root / relative / "tuned_block_sizes.py"
    )
    old_decode.DecodeTuningKey = old_candidate.tuned_block_sizes.DecodeTuningKey
    old_decode.get_decode_request_tile = (
        old_candidate.tuned_block_sizes.get_decode_request_tile
    )
    args.out.mkdir(parents=True, exist_ok=True)
    runtime = {p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")}
    (args.out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    (args.out / "source.json").write_text(
        json.dumps(
            {
                "baseline": subprocess.check_output(
                    ["git", "-C", str(args.baseline_root), "rev-parse", "HEAD"],
                    text=True,
                ).strip(),
                "candidate_base": subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
                ).strip(),
                "devices_visible": len(jax.devices()),
                "devices_used": 1,
                "device_kind": jax.devices()[0].device_kind,
            },
            indent=2,
        )
    )
    (args.out / "candidate.patch").write_bytes(
        subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--binary"])
    )

    def check_bench(name, calls):
        results = [jax.block_until_ready(fn()) for fn in calls]
        jax.tree.map(np.testing.assert_array_equal, results[0], results[1])
        for _ in range(5):
            for fn in calls:
                jax.block_until_ready(fn())
        samples = [[], []]
        for i in range(args.repeats):
            for arm in (i % 2, 1 - i % 2):
                start = time.perf_counter()
                jax.block_until_ready(calls[arm]())
                samples[arm].append((time.perf_counter() - start) * 1000)
        row = dict(
            case=name,
            correctness="exact",
            samples_ms=samples,
            median_ms=[statistics.median(s) for s in samples],
            runtime=runtime,
        )
        with (args.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if args.profile:
            for arm, fn in enumerate(calls):
                with jax.profiler.trace(str(args.out / "profiles" / name / str(arm))):
                    for i in range(3):
                        with jax.profiler.StepTraceAnnotation("route", step_num=i):
                            jax.block_until_ready(fn())

    for requests, queries, width in [
        (48, 6, 65536),
        (48, 6, 131072),
        (8, 3, 65536),
        (3, 5, 8192),
    ]:
        inputs, _, _ = fixture(
            requests, queries, width, width - 1, 128, True, "random", 2, 519
        )
        q, weights, pool, records, lengths = inputs
        q = native["split_halves"](q)
        calls = [
            functools.partial(
                m.decode_indexer,
                q,
                weights,
                pool,
                records[:, ::128],
                lengths,
                page_records=128,
                key_tile=4096,
            )
            for m in (old_decode, decode)
        ]
        check_bench(f"decode-r{requests}-q{queries}-w{width}", calls)
        jax.clear_caches()

    rng = np.random.default_rng(825)
    for requests, queries, width, packed, key_tile in [
        (48, 6, 40960, False, 4096),
        (48, 6, 49152, False, 4096),
        (48, 6, 49152, False, None),
        (48, 6, 49152, True, None),
        (8, 3, 49152, True, None),
        (3, 5, 8192, True, None),
        (7, 7, 8192, True, None),
    ]:
        # Halfplane packed keys: each int32 holds two BF16 records per lane.
        bits = rng.integers(0, 65536, (requests, width // 2, 128), dtype=np.uint32)
        bits = (bits & 0x807F) | 0x3E00
        high = rng.integers(0, 65536, bits.shape, dtype=np.uint32)
        high = (high & 0x807F) | 0x3E00
        keys = jnp.asarray((bits | (high << 16)).view(np.int32))
        if not packed:
            keys = jax.lax.bitcast_convert_type(keys, jnp.bfloat16).reshape(
                requests, width, 128
            )
        ids = np.broadcast_to(np.arange(width // 8), (requests, width // 8)).copy()
        ids += 20000  # Logical IDs well beyond the compact union capacity.
        ids[0, -128:] = -1
        members = jnp.full(ids.shape, (1 << queries) - 1, jnp.int32)
        ids = jnp.asarray(ids, jnp.int32)
        lengths = jnp.full((requests, queries), 524288, jnp.int32).at[-1, -1].set(0)
        weights = jnp.asarray(rng.normal(size=(requests, queries, 32)), jnp.float32)
        for ties in (False, True):
            q = jnp.asarray(rng.normal(size=(requests, queries, 32, 128)), jnp.bfloat16)
            if ties:
                q = jnp.zeros_like(q)
            calls = [
                functools.partial(
                    m.candidate_indexer,
                    q,
                    weights,
                    keys,
                    ids,
                    members,
                    lengths,
                    block_size=8,
                    key_tile=key_tile,
                    halfplane=packed,
                )
                for m in (old_candidate, candidate)
            ]
            check_bench(
                f"candidate-r{requests}-q{queries}-w{width}"
                f"-packed{packed}-tile{key_tile}-ties{ties}",
                calls,
            )
            jax.clear_caches()


if __name__ == "__main__":
    main()
