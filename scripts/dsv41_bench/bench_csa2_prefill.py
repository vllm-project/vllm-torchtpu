# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure ragged CSA2 prefill, including tile planning and output compaction."""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np
from bench_csa2_decode import make_inputs, verify

from vllm_torchtpu.kernels.deepseek_v41.csa2.prefill import csa2_prefill


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--bq", type=int, default=16)
    args = parser.parse_args()
    if any(tokens < 6 for tokens in args.tokens):
        parser.error("--tokens must be at least 6")
    args.out.mkdir(parents=True, exist_ok=True)
    device = jax.devices("tpu")[0]
    for tokens in args.tokens:
        batch = (tokens + 5) // 6
        host = make_inputs(batch, args.heads, 64)
        for key in ("q", "main_kv", "main_kv_scale", "rope_cache", "main_lengths"):
            host[key] = host[key][:tokens]
        positions = np.arange(2048, 2048 + tokens, dtype=np.int32)
        base = (positions[0] - 127) // 64 * 64
        columns = (positions[-1] + 1 - base + 63) // 64
        rng = np.random.default_rng(17)
        host["swa_kv"] = (rng.normal(size=(columns + 1, 64, 512)) * 0.2).astype(
            host["q"].dtype
        )
        host["swa_kv"][0] = 0
        metadata = np.array([[base, 0, *range(1, columns + 1)]], np.int32)
        inputs = (
            host["q"],
            host["main_kv"],
            host["main_kv_scale"],
            host["swa_kv"],
            np.array([0, tokens], np.int32),
            positions,
            host["sink"],
            host["rope_cache"],
        )
        inputs, kwargs = jax.device_put(
            (inputs, dict(swa_metadata=metadata, main_lengths=host["main_lengths"])),
            device,
        )
        compiled = csa2_prefill.lower(*inputs, **kwargs, bq=args.bq).compile()
        output = np.asarray(compiled(*inputs, **kwargs))
        assert np.isfinite(output.astype(np.float32)).all()
        host["positions"] = 2048 + np.arange(batch, dtype=np.int32) * 6
        host["swa_page"] = np.repeat(metadata, batch, axis=0)
        correctness = verify(host, output)
        for _ in range(20):
            compiled(*inputs, **kwargs).block_until_ready()
        host_times = []
        for _ in range(5):
            start = time.perf_counter()
            outputs = [compiled(*inputs, **kwargs) for _ in range(100)]
            jax.block_until_ready(outputs)
            host_times.append((time.perf_counter() - start) * 1e6 / 100)
        directory = args.out / f"t{tokens}"
        with jax.profiler.trace(str(directory), create_perfetto_link=False):
            outputs = [compiled(*inputs, **kwargs) for _ in range(90)]
            jax.block_until_ready(outputs)
        device_times = []
        for path in directory.rglob("*.xplane.pb"):
            for plane in jax.profiler.ProfileData.from_file(str(path)).planes:
                if (
                    not plane.name.startswith("/device:TPU:")
                    or "SparseCore" in plane.name
                ):
                    continue
                for line in plane.lines:
                    if line.name == "XLA Modules":
                        for event in line.events:
                            if "jit_csa2_prefill(" in event.name:
                                device_times.append(float(event.duration_ns) / 1000)
        assert len(device_times) == 90, len(device_times)
        row = dict(
            tokens=tokens,
            heads=args.heads,
            bq=args.bq,
            host_continuous_mean_us=float(np.mean(host_times)),
            device_module_median_us=float(np.median(device_times)),
            device_module_p10_us=float(np.percentile(device_times, 10)),
            device_module_p90_us=float(np.percentile(device_times, 90)),
            correctness=correctness,
            timing_basis=(
                "Full XLA module and host averages include planning and compaction"
            ),
        )
        with (args.out / "metrics.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
