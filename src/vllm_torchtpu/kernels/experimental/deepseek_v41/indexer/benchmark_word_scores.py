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
"""TPU7x microbenchmark for `score_word_rows` across 4K..128K V4.1 contexts.

Reports both hardware XProf on-device latency (`Device us` from `/device:TPU:0`
`XLA Modules` via `jax.profiler.ProfileData`) and single-call dispatch-inclusive
host latency (`Host us`).
"""

import argparse
import functools
import glob
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.profiler
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer.word_scores import (
    DIM,
    HALF_ROW_WORDS,
    HIGH_HALF_SHIFT,
    ROW_WORDS,
    SLOTS,
    WORD_BYTES,
    score_word_rows,
)

BF16_PER_WORD = 2
"""BF16 elements per 32-bit word (`even` in low 16, `odd` in high 16)."""

SCORE_BYTES = WORD_BYTES
"""Bytes per FP32 output score element."""

FLOPS_PER_FMA = 2.0
"""Floating-point operations per multiply-accumulate in MXU dot-product contraction."""

BYTES_PER_MIB = 1 << 20
"""Bytes per MiB (`1,048,576`)."""

BYTES_PER_GB = 1e9
"""Bytes per decimal gigabyte for effective HBM bandwidth reporting."""

FLOPS_PER_TFLOP = 1e12
"""FLOPs per teraFLOP for MXU throughput reporting."""

US_PER_SEC = 1e6
"""Microseconds per second."""

NS_PER_US = 1e3
"""Nanoseconds per microsecond."""

TPU_DEVICE_PLANE = "/device:TPU:0"
"""XProf plane name for TPU device 0."""

XLA_MODULES_LINE = "XLA Modules"
"""XProf line name recording on-device XLA module execution events."""

KEY_INIT_SCALE = 0.1
"""Deterministic initialization scale for synthetic BF16 index keys."""

TPU_WORKLOADS: tuple[tuple[int, int, int, int, str], ...] = (
    (48, 48, 2048, 32, "Decode (1 tok/seq, B=48, 4K keys)"),
    (48, 48, 4096, 32, "Decode (1 tok/seq, B=48, 8K keys)"),
    (48, 48, 6144, 32, "Decode (1 tok/seq, B=48, 12K keys)"),
    (288, 48, 2048, 32, "MTP (6 tok/seq, B=48, 4K keys)"),
    (288, 48, 4096, 32, "MTP (6 tok/seq, B=48, 8K keys)"),
    (288, 48, 6144, 32, "MTP (6 tok/seq, B=48, 12K keys)"),
    (8, 8, 32768, 32, "Decode 128k-2x (1 tok/seq, B=8, 64K keys)"),
    (48, 8, 32768, 32, "MTP 128k-2x (6 tok/seq, B=8, 64K keys)"),
    (288, 48, 32768, 32, "MTP 128k-2x (6 tok/seq, B=48, 64K keys)"),
    (512, 8, 32768, 32, "Prefill 128k-2x (64 tok/seq, B=8, 64K keys)"),
    (4, 4, 65536, 32, "Decode 128k-1x (1 tok/seq, B=4, 128K keys)"),
    (24, 4, 65536, 32, "MTP 128k-1x (6 tok/seq, B=4, 128K keys)"),
    (96, 16, 65536, 32, "MTP 128k-1x (6 tok/seq, B=16, 128K keys)"),
    (512, 4, 65536, 32, "Prefill 128k-1x (128 tok/seq, B=4, 128K keys)"),
)
INTERPRET_WORKLOADS: tuple[tuple[int, int, int, int, str], ...] = (
    (2, 2, 8, 4, "Interpret smoke (1 tok/seq, B=2, 16 keys)"),
)


def _pack_keys_to_words(keys: jax.Array) -> jax.Array:
    """Pack `bf16[groups, 2 * rows, 128]` keys into `int32[groups, rows, 128]` words."""
    groups, num_keys, _ = keys.shape
    paired = keys.reshape(
        groups, num_keys // SLOTS, SLOTS, HALF_ROW_WORDS, BF16_PER_WORD
    )
    word_bits = jax.lax.bitcast_convert_type(paired, jnp.uint16).astype(jnp.uint32)
    packed = word_bits[..., 0] | (word_bits[..., 1] << HIGH_HALF_SHIFT)
    return packed.reshape(groups, num_keys // SLOTS, ROW_WORDS).astype(jnp.int32)


def _measure_device_and_host_us(
    bench_fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    interpret: bool = False,
) -> tuple[float, float]:
    """Measure median on-device XLA module latency and host latency in microseconds."""
    for _ in range(warmup):
        jax.block_until_ready(bench_fn())
    host_samples_us: list[float] = []
    if interpret or jax.default_backend() != "tpu":
        for _ in range(iters):
            start_time = time.perf_counter()
            jax.block_until_ready(bench_fn())
            host_samples_us.append((time.perf_counter() - start_time) * US_PER_SEC)
        host_us = float(statistics.median(host_samples_us))
        return host_us, host_us

    with tempfile.TemporaryDirectory() as tmpdir:
        with jax.profiler.trace(tmpdir):
            for _ in range(iters):
                start_time = time.perf_counter()
                jax.block_until_ready(bench_fn())
                host_samples_us.append((time.perf_counter() - start_time) * US_PER_SEC)
        host_us = float(statistics.median(host_samples_us))
        xplane_files = glob.glob(
            os.path.join(tmpdir, "**", "*.xplane.pb"), recursive=True
        )
        profile_data = jax.profiler.ProfileData.from_file(xplane_files[0])
        device_samples_us: list[float] = []
        for plane in profile_data.planes:
            if plane.name == TPU_DEVICE_PLANE:
                for line in plane.lines:
                    if line.name == XLA_MODULES_LINE:
                        device_samples_us = [
                            event.duration_ns / NS_PER_US for event in line.events
                        ]
                        break
        device_us = (
            float(statistics.median(device_samples_us))
            if device_samples_us
            else host_us
        )
        return device_us, host_us


def main(argv: list[str] | None = None) -> int:
    """Execute the `score_word_rows` benchmark suite and print the roofline table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--interpret", action="store_true")
    args = parser.parse_args(argv)

    peak_bw = float(pltpu.get_tpu_info().mem_bw_bytes_per_second)
    score_jit = jax.jit(functools.partial(score_word_rows, interpret=args.interpret))
    sys.stdout.write(
        f"Devices: {jax.devices()} (peak_bw={peak_bw / BYTES_PER_GB:.1f} GB/s)\n"
    )
    header = (
        f"{'(T, G, R, H)':<22} {'Workload':<45} {'HBM MiB':>8} "
        f"{'AI (F/B)':>9} {'Device us':>10} {'Host us':>10} "
        f"{'BW GB/s':>9} {'MXU TF/s':>9}\n"
    )
    sys.stdout.write(header + "-" * (len(header) - 1) + "\n")

    workloads = INTERPRET_WORKLOADS if args.interpret else TPU_WORKLOADS
    rng = jax.random.key(0)
    for tokens, groups, rows, heads, workload in workloads:
        rng_q, rng_weights, rng = jax.random.split(rng, 3)
        num_keys = SLOTS * rows
        q = jax.random.normal(rng_q, (tokens, heads, DIM), dtype=jnp.bfloat16)
        weights = jax.random.uniform(rng_weights, (tokens, heads, 1), dtype=jnp.float32)
        words = _pack_keys_to_words(
            jnp.ones((groups, num_keys, DIM), jnp.bfloat16) * KEY_INIT_SCALE
        )
        jax.block_until_ready((q, weights, words))
        device_us, host_us = _measure_device_and_host_us(
            functools.partial(score_jit, q, weights, words),
            warmup=args.warmup,
            iters=args.iters,
            interpret=args.interpret,
        )
        total_bytes = (
            q.nbytes + weights.nbytes + words.nbytes + tokens * num_keys * SCORE_BYTES
        )
        total_flops = FLOPS_PER_FMA * tokens * heads * num_keys * DIM
        elapsed_sec = device_us / US_PER_SEC
        sys.stdout.write(
            f"{str((tokens, groups, rows, heads)):<22} {workload:<45} "
            f"{total_bytes / BYTES_PER_MIB:>8.2f} "
            f"{total_flops / total_bytes:>9.2f} "
            f"{device_us:>10.1f} "
            f"{host_us:>10.1f} "
            f"{(total_bytes / BYTES_PER_GB) / elapsed_sec:>9.1f} "
            f"{(total_flops / FLOPS_PER_TFLOP) / elapsed_sec:>9.2f}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
