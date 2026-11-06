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
"""Standalone TPU7x microbenchmark for DeepSeek-V4.1 Sparse MLA + SWA kernels.

Sizes the HBM KV cache bank to ``BANK_VMEM_MULTIPLIER = 8`` times TPU7x VMEM
capacity (``512 MiB`` NoPE + ``128 MiB`` RoPE per bank) so SparseCore gathers
cannot be prefetched into VMEM, and measures both on-device hardware latency
(``Device (us)`` via ``jax.profiler.trace`` + ``jax.profiler.ProfileData``) and
host dispatch-inclusive latency (``Host (us)``) across decode, MTP verify, and
chunked prefill workloads (including compact 2D vs materialized 3D ``swa_kv``).
"""

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

BANK_VMEM_MULTIPLIER = 8
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
INTERPRET_SWA_WINDOW = 16
INTERPRET_PAGES_PER_BANK = 8
INTERPRET_MAX_PAGES_PER_SEQ = 4
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
INTERPRET_WORKLOADS: tuple[tuple[str, int, int, int, int, int], ...] = (
    ("decode_q1", 2, 1, 8, 32, 1),
    ("mtp_q6", 2, 2, 8, 32, 2),
)


def _time_fn(
    step_fn: Callable[[], Any], *, interpret: bool, warmup: int, repeat: int
) -> tuple[float, float]:
    """Measure median ``(device_us, host_us)`` latency for ``step_fn``."""
    warmup_steps = max(1, warmup)
    repeat_steps = max(1, repeat)
    for _ in range(warmup_steps):
        jax.tree_util.tree_map(lambda arr: arr.block_until_ready(), step_fn())
    host_samples_us: list[float] = []
    if interpret:
        for _ in range(repeat_steps):
            start_time = time.perf_counter()
            jax.tree_util.tree_map(lambda arr: arr.block_until_ready(), step_fn())
            host_samples_us.append((time.perf_counter() - start_time) * US_PER_SECOND)
        host_us = float(statistics.median(host_samples_us))
        return host_us, host_us
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
        profile_data = jax.profiler.ProfileData.from_file(xplane_paths[0])
        device_samples_us = [
            event.duration_ns / NS_PER_US
            for plane in profile_data.planes
            if plane.name == TPU_DEVICE_PLANE_NAME
            for line in plane.lines
            if line.name == XLA_MODULES_LINE_NAME
            for event in line.events
        ]
        return float(statistics.median(device_samples_us)), host_us


def run_benchmarks(
    *, interpret: bool = False, warmup: int = 3, repeat: int = 10
) -> list[tuple[str, float, float, float, float]]:
    """Run V4.1 Sparse MLA + SWA benchmarks across decode, MTP, and prefill.

    Args:
        interpret: Run kernels in CPU Pallas interpret mode when ``True``.
        warmup: Number of untimed warmup calls before profiling.
        repeat: Number of timed calls captured under ``jax.profiler.trace``.

    Returns:
        List of ``(workload_name, device_us, host_us, bandwidth_gbps, roofline_pct)``
        rows.

    Data Lifecycle:
        - Created: Synthetic HBM KV pools ``kv_pool_lo``, ``rope_pool_lo`` and
          per-workload query/index/SWA inputs.
        - Used: Passed as dynamic arguments to JIT-compiled
          ``gather_sparse_fp8_kv`` and ``sparse_mla_swa_attention``.
        - Changed: Writes formatted table rows to ``sys.stdout``.
    """
    tpu_info = pltpu.get_tpu_info()
    bank_rows = (BANK_VMEM_MULTIPLIER * int(tpu_info.vmem_capacity_bytes)) // ROW_BYTES
    num_pages = INTERPRET_PAGES_PER_BANK if interpret else bank_rows // PAGE_SIZE
    swa_window = INTERPRET_SWA_WINDOW if interpret else TPU_SWA_WINDOW
    peak_bw_gbps = float(tpu_info.mem_bw_bytes_per_second) / BYTES_PER_GB
    rng_keys = jax.random.split(jax.random.PRNGKey(0), 4)
    kv_pool_lo = jax.random.randint(
        rng_keys[0],
        (num_pages, PAGE_SIZE, mla.ROW_WORDS),
        0,
        RANDOM_WORD_UPPER_BOUND,
        dtype=jnp.int32,
    )
    rope_pool_lo = jax.random.randint(
        rng_keys[1],
        (num_pages, ROPE_ROWS_PER_PAGE, mla.ROW_WORDS),
        0,
        RANDOM_WORD_UPPER_BOUND,
        dtype=jnp.int32,
    )
    gather_jit = jax.jit(
        functools.partial(
            mla.gather_sparse_fp8_kv,
            page_size=PAGE_SIZE,
            interpret=interpret,
        )
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
    workloads = INTERPRET_WORKLOADS if interpret else WORKLOADS
    for (
        label,
        num_seqs,
        query_len,
        num_heads,
        topk,
        compress_ratio,
    ) in workloads:
        num_tokens = num_seqs * query_len
        max_pages_per_seq = (
            INTERPRET_MAX_PAGES_PER_SEQ
            if interpret
            else TPU_UNCOMPRESSED_MAX_PAGES_PER_SEQ // compress_ratio
        )
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
            (num_tokens, swa_window, mla.HEAD_DIM)
            if query_len == 1
            else (num_seqs * swa_window + num_tokens, mla.HEAD_DIM)
        )
        swa_kv = jax.random.normal(rng_keys[3], swa_shape, dtype=jnp.bfloat16) * 0.1
        topk_indices = jnp.arange(num_tokens * topk, dtype=jnp.int32).reshape(
            num_tokens, topk
        ) % (max_pages_per_seq * PAGE_SIZE)
        gathered_nope, gathered_rope = gather_jit(
            kv_pool_lo,
            rope_pool_lo,
            topk_indices,
            block_tables,
            cu_q_lens,
        )
        attn_jit = jax.jit(
            functools.partial(
                mla.sparse_mla_swa_attention,
                sm_scale=DEFAULT_SM_SCALE,
                swa_window=swa_window,
                kv_compress_ratio=compress_ratio,
                interpret=interpret,
            )
        )
        base_attn_bytes = num_tokens * (
            topk * ATTN_HBM_BYTES_PER_SPARSE_KEY
            + num_heads * ATTN_HBM_BYTES_PER_TOKEN_HEAD
        )
        attn_bytes = base_attn_bytes + swa_kv.size * mla.BF16_BYTES
        attn_flops = (
            MXU_FLOPS_PER_TOKEN_HEAD_KEY * num_tokens * num_heads * (topk + swa_window)
        )
        gather_bytes = num_tokens * topk * GATHER_HBM_BYTES_PER_KEY
        bench_rows: list[tuple[str, Callable[[], Any], int, float]] = [
            (
                f"gather_sparse_fp8_kv({label})",
                functools.partial(
                    gather_jit,
                    kv_pool_lo,
                    rope_pool_lo,
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
                    (num_tokens, swa_window, mla.HEAD_DIM),
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
            device_us, host_us = _time_fn(
                step_fn, interpret=interpret, warmup=warmup, repeat=repeat
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
    """CLI entrypoint for ``benchmark_sparse_mla``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    run_benchmarks(**vars(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
