# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare an unmodified StreamIndex module against the imported source tree.

Synthetic FP8 index keys/queries, no model, no service and no model KV allocation.
Both variants receive identical dynamic device arguments; compilation and transfer
are excluded. Each timed invocation blocks on its output. Input banks alternate
outside the compiled executable, preventing capture of constant query tensors.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def stats(values):
    return {
        "mean_us": statistics.mean(values),
        "std_population_us": statistics.pstdev(values),
        **{f"p{p}_us": float(np.percentile(values, p)) for p in (50, 90, 99)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 4, 64])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--context-tokens", type=int, default=65536)
    parser.add_argument("--tuned-decode", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    name = "vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk"
    candidate = importlib.import_module(name)
    spec = importlib.util.spec_from_file_location(name + "_baseline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    # dataclasses resolve their defining module through sys.modules.
    import sys

    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    device = jax.devices()[0]
    assert device.platform == "tpu", device
    manifest = {
        "synthetic": True,
        "context_tokens": args.context_tokens,
        "tuned_decode": args.tuned_decode,
        "compression_ratio": 4,
        "page_size": 256,
        "heads": 64,
        "head_dim": 128,
        "k": 512,
        "metadata_sequence_capacity": 256,
        "warmup": 20,
        "samples": 100,
        "rounds": 3,
        "input_banks": 8,
        "synchronization": "block_until_ready per call",
        "devices": [str(d) for d in jax.devices()],
        "jax": jax.__version__,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("TPU_", "LIBTPU_", "XLA_", "JAX_", "PYTHONPATH"))
        },
        "source": {},
        "results": [],
    }
    for label, module in [("baseline", baseline), ("optimized", candidate)]:
        p = Path(module.__file__)
        manifest["source"][label] = {
            "path": str(p),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "environment"}),
        flush=True,
    )
    rng = np.random.default_rng(1641)
    for batch in args.tokens:
        # 65 pages include the partial page near the end of a 64K decode request.
        pages_per_seq, page_size = 65, 256
        lengths = np.zeros(256, np.int32)
        lengths[:batch] = args.context_tokens + np.arange(batch) % 16
        if np.max(lengths) > pages_per_seq * page_size * 4:
            raise ValueError("Context exceeds the 66560-token cache capacity")
        tables = np.zeros((256, pages_per_seq), np.int32)
        physical = rng.permutation(batch * pages_per_seq).reshape(batch, pages_per_seq)
        tables[:batch] = physical
        cu = np.minimum(np.arange(257), batch).astype(np.int32)
        keys = rng.normal(size=(batch * pages_per_seq, page_size, 128)).astype(
            np.float32
        )
        # A common exact power-of-two scale keeps cache packing outside timings.
        key_fp8 = jax.device_put(keys, device).astype(jnp.float8_e4m3fn)
        packed = np.zeros((batch * pages_per_seq, page_size, 256), np.uint8)
        packed[:, :, :128] = np.asarray(
            jax.lax.bitcast_convert_type(key_fp8, jnp.uint8)
        )
        packed[:, :, 128] = 127  # UE8M0 scale = 1.
        cache = jax.device_put(packed.reshape(-1, 64, 4, 256), device)
        common = (
            cache,
            jax.device_put(lengths, device),
            jax.device_put(tables.flatten(), device),
            jax.device_put(cu, device),
            jax.device_put(np.array([batch, batch, batch], np.int32), device),
        )
        banks = []
        for _ in range(8):
            q = jax.device_put(
                rng.normal(size=(batch, 64, 128)).astype(np.float32), device
            ).astype(jnp.float8_e4m3fn)
            w = jax.device_put(
                rng.uniform(-1, 1, size=(batch, 64)).astype(np.float32), device
            )
            banks.append((q, w, *common))
        jax.block_until_ready(banks)
        executables = {}
        for label, module in [("baseline", baseline), ("optimized", candidate)]:

            def run(
                q,
                weights,
                cache,
                lengths,
                pages,
                cu,
                distribution,
                module=module,
                label=label,
            ):
                decode_pages = 3
                if args.tuned_decode and label == "optimized":
                    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
                        tuned_block_sizes,
                    )

                    decode_pages = tuned_block_sizes.get_decode_kv_pages_per_block(
                        q, cache, pages.reshape(256, -1), k=512, compression_ratio=4
                    )
                return module.streamindex_topk(
                    q,
                    weights,
                    cache,
                    lengths,
                    pages,
                    cu,
                    distribution,
                    k=512,
                    compression_ratio=4,
                    num_kv_pages_per_block=(decode_pages, 2, 2),
                    num_queries_per_block=(1, 128, 128),
                )

            run.__name__ = label + "_streamindex"
            exe = jax.jit(run).lower(*banks[0]).compile()
            executables[label] = exe
            (args.output / f"b{batch}-{label}.hlo.txt").write_text(exe.as_text())
        checks = []
        for inputs in banks:
            before = np.asarray(executables["baseline"](*inputs))
            after = np.asarray(executables["optimized"](*inputs))
            np.testing.assert_array_equal(
                np.sort(before, axis=1), np.sort(after, axis=1)
            )
            assert np.all(
                (after == -1) | ((after >= 0) & (after < lengths[:batch, None] // 4))
            )
            for row, length in zip(after, lengths[:batch], strict=True):
                valid = row[row >= 0]
                assert len(valid) == min(512, length // 4)
                assert len(np.unique(valid)) == len(valid)
            checks.append({"equal_topk_sets": True, "valid_unique_indices": True})
        rounds = []
        for repeat in range(3):
            order = (
                ["baseline", "optimized"]
                if repeat % 2 == 0
                else ["optimized", "baseline"]
            )
            rd = {"round": repeat, "order": order}
            for label in order:
                exe = executables[label]
                for i in range(20):
                    exe(*banks[i % 8]).block_until_ready()
                times = []
                for i in range(100):
                    inputs = banks[i % 8]
                    start = time.perf_counter_ns()
                    exe(*inputs).block_until_ready()
                    times.append((time.perf_counter_ns() - start) / 1000)
                rd[label] = {**stats(times), "raw_us": times}
            rd["speedup_p50"] = rd["baseline"]["p50_us"] / rd["optimized"]["p50_us"]
            rounds.append(rd)
        result = {"tokens_per_dp": batch, "checks": checks, "rounds": rounds}
        for label in executables:
            result[label] = stats([t for rd in rounds for t in rd[label]["raw_us"]])
        result["speedup_p50"] = (
            result["baseline"]["p50_us"] / result["optimized"]["p50_us"]
        )
        manifest["results"].append(result)
        (args.output / "results.json").write_text(json.dumps(manifest, indent=2))
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("checks", "rounds")}
            ),
            flush=True,
        )
        if args.trace and batch == 1:
            with jax.profiler.trace(str(args.output / "trace")):
                for repeat in range(3):
                    order = ["baseline", "optimized"]
                    if repeat % 2:
                        order.reverse()
                    for label in order:
                        with jax.profiler.TraceAnnotation(f"{label}-round-{repeat}"):
                            for i in range(20):
                                executables[label](*banks[i % 8]).block_until_ready()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
