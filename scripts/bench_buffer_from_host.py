#!/usr/bin/env python3
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
"""Measure direct TorchTPU CPU-to-device BufferFromHostBuffer latency."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.profiler
import torch_tpu  # noqa: F401 - registers the PrivateUse1 TPU backend.
from torch_tpu._internal.profiler import TpuProfilerConfig


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values_us: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values_us),
        "min_us": min(values_us),
        "p50_us": percentile(values_us, 0.50),
        "p90_us": percentile(values_us, 0.90),
        "p99_us": percentile(values_us, 0.99),
        "max_us": max(values_us),
        "mean_us": statistics.fmean(values_us),
    }


def copy_once(source: torch.Tensor,
              device: torch.device) -> tuple[torch.Tensor, float]:
    started_ns = time.perf_counter_ns()
    result = source.to(device, non_blocking=True)
    elapsed_us = (time.perf_counter_ns() - started_ns) / 1_000.0
    return result, elapsed_us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="tpu:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    sources = {
        32: torch.zeros(32, dtype=torch.int32, device="cpu"),
        8704: torch.zeros(8704, dtype=torch.int32, device="cpu"),
    }

    init = torch.zeros(1, dtype=torch.int32, device=device)
    torch.tpu.synchronize()
    del init
    for source in sources.values():
        for iteration in range(args.warmup):
            source[0] = iteration
            result, _ = copy_once(source, device)
            torch.tpu.synchronize()
            del result

    trace_dir = args.output_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    handler = torch.profiler.tensorboard_trace_handler(dir_name=str(trace_dir),
                                                       use_gzip=True)
    config = TpuProfilerConfig(
        run_dir=str(trace_dir),
        host_tracer_level=2,
        device_tracer_level=1,
        python_tracer_level=0,
        experimental_options={},
    )
    wall_times: dict[int, list[float]] = {size: [] for size in sources}
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ],
            on_trace_ready=handler,
            experimental_config=config,
    ):
        for size, source in sources.items():
            for iteration in range(args.iterations):
                source[0] = iteration
                with torch.profiler.record_function(f"bench_s32_{size}"):
                    result, elapsed_us = copy_once(source, device)
                wall_times[size].append(elapsed_us)
                torch.tpu.synchronize()
                del result

    output = {
        "device": str(device),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "tpu_premapped_buffer_size": os.getenv("TPU_PREMAPPED_BUFFER_SIZE"),
        "wall_api_time": {
            f"s32[{size}]": summarize(values)
            for size, values in wall_times.items()
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)
    print(f"trace_dir={trace_dir}", flush=True)


if __name__ == "__main__":
    main()
