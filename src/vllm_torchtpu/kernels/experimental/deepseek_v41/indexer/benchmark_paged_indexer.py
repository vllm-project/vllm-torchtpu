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
"""Standalone TPU7x benchmark for `paged_indexer_topk` across 4K..128K contexts."""

import argparse
import functools
import glob
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.profiler
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer.paged_indexer import (
    paged_indexer_topk,
)
from vllm_torchtpu.kernels.experimental.deepseek_v41.indexer.word_scores import (
    BF16_BYTES,
    DIM,
    ROW_WORDS,
    SLOTS,
    SUB_BLOCK_ROWS,
    SUBLANES,
    TPU7X_GENERATION,
    WEIGHT_BYTES_PER_HEAD,
    WORD_BYTES,
)

# Size the pool at 2x VMEM capacity so paged gathers exercise HBM bandwidth.
POOL_VMEM_MULTIPLIER = 2
INDEXER_HEADS = 32
ROW_ALIGNMENT = SUB_BLOCK_ROWS
ROW_BYTES = ROW_WORDS * WORD_BYTES
QUERY_WEIGHT_BYTES_PER_HEAD = BF16_BYTES * DIM + WEIGHT_BYTES_PER_HEAD
# Working traffic per key: 4 B score write + 4 B masked read + 4 B top-k read.
TOPK_WORKING_BYTES_PER_KEY = 12
INDEX_BYTES = WORD_BYTES
GATHER_READ_WRITE_FACTOR = 2
QUERY_INIT_SCALE = 0.1
BYTES_PER_GB = 1e9
US_PER_SEC = 1e6
NS_PER_US = 1e3
TPU_DEVICE_PLANE = "/device:TPU:0"
XLA_MODULES_LINE = "XLA Modules"
PERCENT_SCALE = 100.0

TPU_WORKLOADS: tuple[tuple[str, int, int, int, int, int, int, int], ...] = (
    ("decode_b48_q1_ctx4k_k2048", 48, 1, 32, 64, 2048, 0, 2),
    ("decode_b48_q1_ctx8k_k2048", 48, 1, 32, 128, 2048, 0, 2),
    ("mtp_b48_q6_ctx4k_k2048", 48, 6, 32, 64, 2048, 0, 2),
    ("mtp_b48_q6_ctx4k_cand_k2048", 48, 6, 32, 64, 2048, 2048, 2),
    ("decode_128k_2x_b8_q1_k512", 8, 1, 32, 1024, 512, 0, 2),
    ("mtp_128k_2x_b8_q6_k512", 8, 6, 32, 1024, 512, 0, 2),
    ("prefill_128k_2x_b2_t512_k512", 2, 256, 32, 1024, 512, 0, 2),
    ("decode_128k_1x_b4_q1_k2048", 4, 1, 64, 1024, 2048, 0, 1),
    ("mtp_128k_1x_b4_q6_cand16k_k2048", 4, 6, 64, 1024, 2048, 16384, 1),
    ("prefill_128k_1x_b2_t2048_k2048", 2, 1024, 64, 1024, 2048, 0, 1),
)


def _measure_device_and_host_us(
    bench_fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        jax.block_until_ready(bench_fn())
    host_samples_us: list[float] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with jax.profiler.trace(tmpdir):
            for _ in range(repeat):
                start_time = time.perf_counter()
                jax.block_until_ready(bench_fn())
                host_samples_us.append((time.perf_counter() - start_time) * US_PER_SEC)
        host_us = float(statistics.median(host_samples_us))
        xplane_files = glob.glob(
            os.path.join(tmpdir, "**", "*.xplane.pb"), recursive=True
        )
        if not xplane_files:
            raise RuntimeError(
                f"No .xplane.pb profile file was produced under {tmpdir}."
            )
        profile_data = jax.profiler.ProfileData.from_file(xplane_files[0])
        tpu_plane = next(
            (p for p in profile_data.planes if p.name == TPU_DEVICE_PLANE),
            None,
        )
        if tpu_plane is None:
            raise RuntimeError(
                f"Missing profiler plane {TPU_DEVICE_PLANE!r} in {xplane_files[0]}."
            )
        xla_line = next(
            (line for line in tpu_plane.lines if line.name == XLA_MODULES_LINE),
            None,
        )
        if xla_line is None:
            raise RuntimeError(
                f"Missing profiler line {XLA_MODULES_LINE!r} on plane "
                f"{TPU_DEVICE_PLANE!r} in {xplane_files[0]}."
            )
        device_samples_us = [event.duration_ns / NS_PER_US for event in xla_line.events]
        if not device_samples_us:
            raise RuntimeError(
                f"Empty device event samples on {TPU_DEVICE_PLANE!r} -> "
                f"{XLA_MODULES_LINE!r} in {xplane_files[0]}."
            )
        return float(statistics.median(device_samples_us)), host_us


def _check_tpu7x(func_name: str) -> None:
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"{func_name} requires physical TPU7x execution, "
            f"got backend={jax.default_backend()!r}"
        )
    tpu_info = pltpu.get_tpu_info()
    if (
        tpu_info.generation != TPU7X_GENERATION
        or tpu_info.num_lanes != ROW_WORDS
        or tpu_info.num_sublanes != SUBLANES
    ):
        raise RuntimeError(
            f"{func_name} requires TPU7x ({SUBLANES}x{ROW_WORDS} lanes), got "
            f"generation={tpu_info.generation}, num_lanes={tpu_info.num_lanes}, "
            f"num_sublanes={tpu_info.num_sublanes}"
        )


def run_benchmarks(
    *,
    warmup: int = 3,
    repeat: int = 10,
) -> list[tuple[str, float, float, float, float]]:
    _check_tpu7x("run_benchmarks")
    tpu_info = pltpu.get_tpu_info()
    peak_bw_gbps = float(tpu_info.mem_bw_bytes_per_second) / BYTES_PER_GB
    pool_rows = (POOL_VMEM_MULTIPLIER * int(tpu_info.vmem_capacity_bytes)) // ROW_BYTES

    results: list[tuple[str, float, float, float, float]] = []
    header = (
        f"{'Kernel Workload':<34} | {'Device (us)':>12} | "
        f"{'Host (us)':>11} | {'BW (GB/s)':>10} | {'Roofline (%)':>12}\n"
    )
    sys.stdout.write(header + "-" * (len(header) - 1) + "\n")
    for (
        label,
        num_seqs,
        query_len,
        page_rows,
        max_pages,
        topk,
        num_candidates,
        compress_ratio,
    ) in TPU_WORKLOADS:
        num_pages = pool_rows // page_rows
        num_tokens = num_seqs * query_len
        padded_rows = (
            ((max_pages * page_rows) + ROW_ALIGNMENT - 1) // ROW_ALIGNMENT
        ) * ROW_ALIGNMENT
        total_keys = SLOTS * padded_rows
        queries = (
            jnp.ones((num_tokens, INDEXER_HEADS, DIM), dtype=jnp.bfloat16)
            * QUERY_INIT_SCALE
        )
        weights = jnp.ones((num_tokens, INDEXER_HEADS, 1), dtype=jnp.float32)
        k_pool = jnp.zeros((num_pages, page_rows, ROW_WORDS), dtype=jnp.int32)
        block_tables = (
            jnp.arange(num_seqs * max_pages, dtype=jnp.int32) % num_pages
        ).reshape(num_seqs, max_pages)
        seq_lens = jnp.full(
            (num_seqs,),
            max_pages * page_rows * SLOTS * compress_ratio,
            dtype=jnp.int32,
        )
        cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32) * query_len
        candidate_indices = (
            jnp.broadcast_to(
                jnp.arange(num_candidates, dtype=jnp.int32),
                (num_tokens, num_candidates),
            )
            if num_candidates > 0
            else None
        )
        workload_fn = functools.partial(
            paged_indexer_topk,
            queries,
            weights,
            k_pool,
            block_tables,
            seq_lens,
            cu_q_lens,
            topk=topk,
            candidate_indices=candidate_indices,
            compress_ratio=compress_ratio,
        )
        gather_score_bytes = (
            GATHER_READ_WRITE_FACTOR * num_seqs * padded_rows * ROW_BYTES
        )
        topk_stage_bytes = num_tokens * (
            INDEXER_HEADS * QUERY_WEIGHT_BYTES_PER_HEAD
            + total_keys * TOPK_WORKING_BYTES_PER_KEY
            + (topk + num_candidates) * INDEX_BYTES
        )
        total_bytes = gather_score_bytes + topk_stage_bytes
        device_us, host_us = _measure_device_and_host_us(
            workload_fn,
            warmup=warmup,
            repeat=repeat,
        )
        bw_gbps = (total_bytes / BYTES_PER_GB) / (device_us / US_PER_SEC)
        roofline_pct = PERCENT_SCALE * bw_gbps / peak_bw_gbps
        results.append((label, device_us, host_us, bw_gbps, roofline_pct))
        sys.stdout.write(
            f"{label:<34} | {device_us:12.2f} | {host_us:11.2f} | "
            f"{bw_gbps:10.2f} | {roofline_pct:11.2f}%\n"
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args(argv)
    run_benchmarks(
        warmup=args.warmup,
        repeat=args.repeat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
