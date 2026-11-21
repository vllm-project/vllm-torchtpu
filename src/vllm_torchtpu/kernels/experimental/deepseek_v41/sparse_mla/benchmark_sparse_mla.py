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
"""Standalone TPU7x microbenchmark for Sparse MLA + SWA gather and attention."""

from __future__ import annotations

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

from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla import (
    kernel as mla,
)

POOL_VMEM_MULTIPLIER = 8
PAGE_SIZE = 16
ROPE_ROWS_PER_PAGE = PAGE_SIZE // mla.ROPE_TOKENS_PER_ROW
ROW_BYTES = mla.ROW_WORDS * mla.WORD_BYTES
ROPE_BYTES_PER_TOKEN = mla.ROPE_WORDS * mla.WORD_BYTES
INDEX_BYTES_PER_KEY = mla.WORD_BYTES
GATHER_HBM_BYTES_PER_KEY = 2 * (ROW_BYTES + ROPE_BYTES_PER_TOKEN) + INDEX_BYTES_PER_KEY
ATTN_HBM_BYTES_PER_SPARSE_KEY = ROW_BYTES + ROPE_BYTES_PER_TOKEN + INDEX_BYTES_PER_KEY
ATTN_HBM_BYTES_PER_TOKEN_HEAD = 2 * mla.HEAD_DIM * mla.BF16_BYTES
MXU_FLOPS_PER_TOKEN_HEAD_KEY = 4.0 * mla.HEAD_DIM
DEFAULT_SM_SCALE = 0.0625
TPU_SWA_WINDOW = 128
TPU_UNCOMPRESSED_MAX_PAGES_PER_SEQ = 4096
RANDOM_WORD_UPPER_BOUND = 1000
TPU_DEVICE_PLANE_NAME = "/device:TPU:0"
XLA_MODULES_LINE_NAME = "XLA Modules"
NS_PER_US = 1e3
US_PER_SECOND = 1e6
BYTES_PER_GB = 1e9
FLOPS_PER_TFLOP = 1e12

WORKLOADS: tuple[tuple[str, int, int, int, int, int], ...] = (
    ("decode_h64_k512_q1", 48, 1, 64, 512, 2),
    ("mtp_h64_k512_q6", 48, 6, 64, 512, 2),
    ("prefill_h64_k512_t2048", 2, 1024, 64, 512, 2),
    ("decode_h64_k2048_q1", 48, 1, 64, 2048, 1),
    ("mtp_h64_k2048_q6", 48, 6, 64, 2048, 1),
    ("prefill_h64_k2048_t2048", 2, 1024, 64, 2048, 1),
    ("decode_h32_k2048_q1", 48, 1, 32, 2048, 1),
    ("mtp_h32_k2048_q6", 48, 6, 32, 2048, 1),
)


def measure_device_and_host_us(
    step_fn: Callable[[], Any], *, warmup: int, repeat: int
) -> tuple[float, float]:
    warmup_steps = max(1, warmup)
    repeat_steps = max(1, repeat)
    for _ in range(warmup_steps):
        jax.tree_util.tree_map(lambda arr: arr.block_until_ready(), step_fn())
    host_samples_us: list[float] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with jax.profiler.trace(tmpdir):
            for _ in range(repeat_steps):
                start_time = time.perf_counter()
                jax.tree_util.tree_map(lambda arr: arr.block_until_ready(), step_fn())
                host_samples_us.append(
                    (time.perf_counter() - start_time) * US_PER_SECOND
                )
        host_us = float(statistics.median(host_samples_us))
        xplane_paths = glob.glob(
            os.path.join(tmpdir, "**", "*.xplane.pb"), recursive=True
        )
        if not xplane_paths:
            raise RuntimeError(f"No .xplane.pb file produced under {tmpdir}.")
        profile_data = jax.profiler.ProfileData.from_file(xplane_paths[0])
        tpu_plane = next(
            (
                plane
                for plane in profile_data.planes
                if plane.name == TPU_DEVICE_PLANE_NAME
            ),
            None,
        )
        if tpu_plane is None:
            raise RuntimeError(
                f"Missing profiler plane {TPU_DEVICE_PLANE_NAME!r} "
                f"in {xplane_paths[0]}."
            )
        xla_line = next(
            (line for line in tpu_plane.lines if line.name == XLA_MODULES_LINE_NAME),
            None,
        )
        if xla_line is None:
            raise RuntimeError(
                f"Missing profiler line {XLA_MODULES_LINE_NAME!r} on plane "
                f"{TPU_DEVICE_PLANE_NAME!r} in {xplane_paths[0]}."
            )
        device_samples_us = [event.duration_ns / NS_PER_US for event in xla_line.events]
        if not device_samples_us:
            raise RuntimeError(
                f"Empty device event samples on {TPU_DEVICE_PLANE_NAME!r} -> "
                f"{XLA_MODULES_LINE_NAME!r} in {xplane_paths[0]}."
            )
        return float(statistics.median(device_samples_us)), host_us


def _check_tpu7x(func_name: str) -> None:
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            f"{func_name} requires physical TPU backend, got {jax.default_backend()!r}."
        )
    tpu_info = pltpu.get_tpu_info()
    if (
        tpu_info.generation != mla.TPU7X_GENERATION
        or tpu_info.num_lanes != mla.ROW_WORDS
        or tpu_info.num_sublanes != mla.SUBLANES
    ):
        raise RuntimeError(
            f"{func_name} requires TPU7x ({mla.SUBLANES}x{mla.ROW_WORDS} lanes), "
            f"got generation={tpu_info.generation}, "
            f"num_lanes={tpu_info.num_lanes}, "
            f"num_sublanes={tpu_info.num_sublanes}."
        )


def run_benchmarks(
    *, warmup: int = 3, repeat: int = 10
) -> list[tuple[str, float, float, float, float]]:
    _check_tpu7x("run_benchmarks")
    tpu_info = pltpu.get_tpu_info()
    # Avoid XLA prefetching KV pool into 64 MiB VMEM: size pool to 8x VMEM (512 MiB).
    pool_rows = (POOL_VMEM_MULTIPLIER * int(tpu_info.vmem_capacity_bytes)) // ROW_BYTES
    num_pages = pool_rows // PAGE_SIZE
    peak_bw_gbps = float(tpu_info.mem_bw_bytes_per_second) / BYTES_PER_GB
    rng_keys = jax.random.split(jax.random.PRNGKey(0), 4)
    kv_pool = jax.random.randint(
        rng_keys[0],
        (num_pages, PAGE_SIZE, mla.ROW_WORDS),
        0,
        RANDOM_WORD_UPPER_BOUND,
        dtype=jnp.int32,
    )
    rope_pool = jax.random.randint(
        rng_keys[1],
        (num_pages, ROPE_ROWS_PER_PAGE, mla.ROW_WORDS),
        0,
        RANDOM_WORD_UPPER_BOUND,
        dtype=jnp.int32,
    )
    gather_jit = jax.jit(
        functools.partial(mla.gather_sparse_fp8_kv, page_size=PAGE_SIZE)
    )
    header_prefix = (
        f"{'Kernel Workload':<45} | {'Device (us)':>11} | {'Host (us)':>9} | "
    )
    table_header = (
        header_prefix + f"{'AI (F/B)':>8} | {'BW (GB/s)':>9} | {'MXU (TFLOP/s)':>13} | "
        f"{'BW Roof (%)':>11}\n"
    )
    sys.stdout.write(table_header + "-" * (len(table_header) - 1) + "\n")
    results: list[tuple[str, float, float, float, float]] = []
    for (
        label,
        num_seqs,
        query_len,
        num_heads,
        topk,
        compress_ratio,
    ) in WORKLOADS:
        num_tokens = num_seqs * query_len
        max_pages_per_seq = TPU_UNCOMPRESSED_MAX_PAGES_PER_SEQ // compress_ratio
        block_tables = (
            jnp.arange(num_seqs * max_pages_per_seq, dtype=jnp.int32).reshape(
                num_seqs, max_pages_per_seq
            )
            % num_pages
        )
        seq_lens = jnp.full(
            (num_seqs,),
            max_pages_per_seq * PAGE_SIZE * compress_ratio,
            jnp.int32,
        )
        cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32) * query_len
        attention_sinks = jnp.linspace(-0.5, 0.5, num_heads, dtype=jnp.float32)
        query = (
            jax.random.normal(
                rng_keys[2],
                (num_tokens, num_heads, mla.HEAD_DIM),
                dtype=jnp.bfloat16,
            )
            * 0.1
        )
        swa_shape = (
            (num_tokens, TPU_SWA_WINDOW, mla.HEAD_DIM)
            if query_len == 1
            else (num_seqs * TPU_SWA_WINDOW + num_tokens, mla.HEAD_DIM)
        )
        swa_kv = jax.random.normal(rng_keys[3], swa_shape, dtype=jnp.bfloat16) * 0.1
        topk_indices = jnp.arange(num_tokens * topk, dtype=jnp.int32).reshape(
            num_tokens, topk
        ) % (max_pages_per_seq * PAGE_SIZE)
        gathered_nope, gathered_rope = gather_jit(
            kv_pool,
            rope_pool,
            topk_indices,
            block_tables,
            cu_q_lens,
        )
        attn_jit = jax.jit(
            functools.partial(
                mla.sparse_mla_swa_attention,
                sm_scale=DEFAULT_SM_SCALE,
                swa_window=TPU_SWA_WINDOW,
                kv_compress_ratio=compress_ratio,
            )
        )
        base_attn_bytes = num_tokens * (
            topk * ATTN_HBM_BYTES_PER_SPARSE_KEY
            + num_heads * ATTN_HBM_BYTES_PER_TOKEN_HEAD
        )
        attn_bytes = base_attn_bytes + swa_kv.size * mla.BF16_BYTES
        attn_flops = (
            MXU_FLOPS_PER_TOKEN_HEAD_KEY
            * num_tokens
            * num_heads
            * (topk + TPU_SWA_WINDOW)
        )
        gather_bytes = num_tokens * topk * GATHER_HBM_BYTES_PER_KEY
        bench_rows: list[tuple[str, Callable[[], Any], int, float]] = [
            (
                f"gather_sparse_fp8_kv({label})",
                functools.partial(
                    gather_jit,
                    kv_pool,
                    rope_pool,
                    topk_indices,
                    block_tables,
                    cu_q_lens,
                ),
                gather_bytes,
                0.0,
            ),
            (
                f"sparse_mla_swa_attention({label})",
                functools.partial(
                    attn_jit,
                    query,
                    gathered_nope,
                    gathered_rope,
                    topk_indices,
                    attention_sinks,
                    seq_lens,
                    cu_q_lens,
                    swa_kv=swa_kv,
                ),
                attn_bytes,
                attn_flops,
            ),
        ]
        if query_len > 1:
            swa_kv_3d = (
                jax.random.normal(
                    rng_keys[3],
                    (num_tokens, TPU_SWA_WINDOW, mla.HEAD_DIM),
                    dtype=jnp.bfloat16,
                )
                * 0.1
            )
            attn_3d_bytes = base_attn_bytes + swa_kv_3d.size * mla.BF16_BYTES
            bench_rows.append(
                (
                    f"sparse_mla_swa_attention_3d({label})",
                    functools.partial(
                        attn_jit,
                        query,
                        gathered_nope,
                        gathered_rope,
                        topk_indices,
                        attention_sinks,
                        seq_lens,
                        cu_q_lens,
                        swa_kv=swa_kv_3d,
                    ),
                    attn_3d_bytes,
                    attn_flops,
                )
            )
        for row_name, step_fn, total_bytes, total_flops in bench_rows:
            device_us, host_us = measure_device_and_host_us(
                step_fn, warmup=warmup, repeat=repeat
            )
            bandwidth_gbps = (total_bytes / BYTES_PER_GB) / (device_us / US_PER_SECOND)
            achieved_tflops = (total_flops / FLOPS_PER_TFLOP) / (
                device_us / US_PER_SECOND
            )
            roofline_pct = 100.0 * bandwidth_gbps / peak_bw_gbps
            results.append((row_name, device_us, host_us, bandwidth_gbps, roofline_pct))
            row_prefix = f"{row_name:<45} | {device_us:11.2f} | {host_us:9.2f} | "
            sys.stdout.write(
                row_prefix
                + f"{total_flops / total_bytes:8.2f} | {bandwidth_gbps:9.2f} | "
                f"{achieved_tflops:13.2f} | {roofline_pct:10.2f}%\n"
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    run_benchmarks(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
