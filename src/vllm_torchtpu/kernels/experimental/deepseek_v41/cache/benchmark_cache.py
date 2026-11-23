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
"""Standalone TPU7x microbenchmark for DeepSeek-V4.1 word-row cache kernels."""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.profiler
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.deepseek_v41 import cache as ck

TPU7X_GENERATION = 7
# Avoid XLA prefetching the pool into 64 MiB VMEM: size pool at 8x VMEM (512 MiB).
POOL_VMEM_MULTIPLIER = 8
TPU_DEVICE_PLANE = "/device:TPU:0"
XLA_MODULES_LINE = "XLA Modules"
NANOSECONDS_PER_MICROSECOND = 1e3

DEFAULT_WARMUP = 5
DEFAULT_REPEAT = 50
PT8_PAGE_TOKENS = 8
SWA_WINDOW_TOKENS = 128
SWA_ROWS_PER_TOKEN = 2
SWA_ROWS_PER_PAGE = ck.DEFAULT_PAGE_TOKENS * SWA_ROWS_PER_TOKEN
SWA_ROWS_PER_SEQ = SWA_WINDOW_TOKENS * SWA_ROWS_PER_TOKEN
SWA_PAGES_PER_WINDOW = (
    SWA_ROWS_PER_SEQ + 2 * SWA_ROWS_PER_PAGE - 2
) // SWA_ROWS_PER_PAGE
PACK_FP8_INPUT_BYTES_PER_TOKEN = ck.LATENT_DIM * 2
PACK_FP8_OUTPUT_BYTES_PER_TOKEN = (ck.ROW_WORDS + ck.ROPE_WORDS) * ck.WORD_BYTES
PACK_FP8_HBM_BYTES_PER_TOKEN = (
    PACK_FP8_INPUT_BYTES_PER_TOKEN + PACK_FP8_OUTPUT_BYTES_PER_TOKEN
)
GATHER_HBM_BYTES_PER_ROW = 2 * ck.ROW_BYTES
SWA_WINDOW_HBM_BYTES_PER_SEQ = (
    SWA_PAGES_PER_WINDOW * SWA_ROWS_PER_PAGE + SWA_ROWS_PER_SEQ
) * ck.ROW_BYTES
COMPRESSED_RATIO_2X = 2

SMALL_BATCH_SEQS = 8
NON_POW2_BATCH_SEQS = 13
DECODE_BATCH_SEQS = 48
MTP_VERIFY_TOKENS = 6
MTP_BATCH_TOKENS = DECODE_BATCH_SEQS * MTP_VERIFY_TOKENS
NON_POW2_PREFILL_TOKENS = 300
PREFILL_2K_TOKENS = 2048
PREFILL_4K_TOKENS = 4096
MIXED_PREFILL_SEQS = 4
RANDOM_GATHER_QUERIES = DECODE_BATCH_SEQS * DECODE_BATCH_SEQS
ALIGNED_BURST_QUERIES_B48 = DECODE_BATCH_SEQS * SWA_ROWS_PER_SEQ

# Avoid sequential page allocation or aligned SWA windows: coprime strides scatter
# pages across HBM and straddle 9 non-contiguous 16-token pages per window.
SEQ_PAGE_HASH_PRIME = 104_729
BLOCK_HASH_PRIME = 7_919
PAGE_HASH_OFFSET = 17
UNALIGNED_BASE_CONTEXT_POS = 261
SEQ_CONTEXT_STRIDE = 37

MICROSECONDS_PER_SECOND = 1e6
BYTES_PER_GIGABYTE = 1e9
PERCENT_SCALE = 100.0

BENCHMARK_REGIMES = (
    (DECODE_BATCH_SEQS, 1, "dec"),
    (MTP_BATCH_TOKENS, MTP_VERIFY_TOKENS, "mtp"),
    (NON_POW2_PREFILL_TOKENS, NON_POW2_PREFILL_TOKENS, "p300"),
    (PREFILL_2K_TOKENS, PREFILL_2K_TOKENS, "p2k"),
    (PREFILL_4K_TOKENS, PREFILL_4K_TOKENS, "p4k"),
)
SWA_WINDOW_SEQ_COUNTS = (
    (SMALL_BATCH_SEQS, "s8"),
    (NON_POW2_BATCH_SEQS, "s13"),
    (DECODE_BATCH_SEQS, "dec"),
    (MTP_BATCH_TOKENS, "mtp"),
)


def _check_tpu7x() -> Any:
    if jax.default_backend() != "tpu":
        raise RuntimeError(
            "benchmark_cache requires a physical TPU7x backend; "
            f"got {jax.default_backend()!r}."
        )
    tpu_info = pltpu.get_tpu_info()
    if tpu_info.generation != TPU7X_GENERATION or tpu_info.num_lanes != ck.ROW_WORDS:
        raise RuntimeError(
            f"Requires TPU7x (generation={TPU7X_GENERATION}, "
            f"num_lanes={ck.ROW_WORDS}), got generation={tpu_info.generation}, "
            f"num_lanes={tpu_info.num_lanes}."
        )
    return tpu_info


def measure_device_and_host_us(
    workload_fn: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    """Return median on-device XProf latency and host wall-clock latency in us."""
    for _ in range(warmup):
        jax.tree_util.tree_map(lambda arr: arr.block_until_ready(), workload_fn())
    host_samples_us: list[float] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        with jax.profiler.trace(tmpdir):
            for _ in range(repeat):
                start_time = time.perf_counter()
                jax.tree_util.tree_map(
                    lambda arr: arr.block_until_ready(), workload_fn()
                )
                elapsed_us = (
                    time.perf_counter() - start_time
                ) * MICROSECONDS_PER_SECOND
                host_samples_us.append(elapsed_us)
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
        device_samples_us = [
            event.duration_ns / NANOSECONDS_PER_MICROSECOND for event in xla_line.events
        ]
        if not device_samples_us:
            raise RuntimeError(
                f"Empty device event samples on {TPU_DEVICE_PLANE!r} -> "
                f"{XLA_MODULES_LINE!r} in {xplane_files[0]}."
            )
        return float(statistics.median(device_samples_us)), host_us


def _paged_slots(
    num_tokens: int,
    tokens_per_seq: int,
    total_pool_slots: int,
    page_tokens: int = ck.DEFAULT_PAGE_TOKENS,
) -> jax.Array:
    token_indices = jnp.arange(num_tokens, dtype=jnp.int32)
    num_pages = max(1, total_pool_slots // page_tokens)
    seq_indices = token_indices // tokens_per_seq
    block_in_seq = (token_indices % tokens_per_seq) // page_tokens
    physical_page = (
        seq_indices * SEQ_PAGE_HASH_PRIME
        + block_in_seq * BLOCK_HASH_PRIME
        + PAGE_HASH_OFFSET
    ) % num_pages
    return physical_page * page_tokens + (token_indices % page_tokens)


def _swa_rows(num_seqs: int, total_pool_rows: int) -> jax.Array:
    seq_indices = jnp.arange(num_seqs, dtype=jnp.int32)[:, None]
    window_offsets = jnp.arange(SWA_WINDOW_TOKENS, dtype=jnp.int32)
    kv_positions = (
        UNALIGNED_BASE_CONTEXT_POS
        + seq_indices * SEQ_CONTEXT_STRIDE
        - SWA_WINDOW_TOKENS
        + window_offsets
    )
    num_swa_pages = max(1, total_pool_rows // SWA_ROWS_PER_PAGE)
    physical_pages = (
        seq_indices * SEQ_PAGE_HASH_PRIME
        + (kv_positions // ck.DEFAULT_PAGE_TOKENS) * BLOCK_HASH_PRIME
        + PAGE_HASH_OFFSET
    ) % num_swa_pages
    token_records = (
        physical_pages * ck.DEFAULT_PAGE_TOKENS
        + (kv_positions % ck.DEFAULT_PAGE_TOKENS)
    )[:, :, None]
    row_pair_offsets = jnp.arange(SWA_ROWS_PER_TOKEN, dtype=jnp.int32)
    return (token_records * SWA_ROWS_PER_TOKEN + row_pair_offsets).reshape(-1)


def _paged_swa_window_inputs(
    num_seqs: int, pool_rows: int
) -> tuple[jax.Array, jax.Array]:
    num_pages = max(1, pool_rows // SWA_ROWS_PER_PAGE)
    seq_indices = jnp.arange(num_seqs, dtype=jnp.int32)
    window_starts = (
        UNALIGNED_BASE_CONTEXT_POS
        + seq_indices * SEQ_CONTEXT_STRIDE
        - SWA_WINDOW_TOKENS
    )
    first_logical_blocks = window_starts // ck.DEFAULT_PAGE_TOKENS
    logical_blocks = (
        first_logical_blocks[:, None]
        + jnp.arange(SWA_PAGES_PER_WINDOW, dtype=jnp.int32)[None, :]
    )
    page_ids = (
        seq_indices[:, None] * SEQ_PAGE_HASH_PRIME
        + logical_blocks * BLOCK_HASH_PRIME
        + PAGE_HASH_OFFSET
    ) % num_pages
    phase_rows = (window_starts % ck.DEFAULT_PAGE_TOKENS) * SWA_ROWS_PER_TOKEN
    return page_ids.astype(jnp.int32), phase_rows.astype(jnp.int32)


def _aligned_swa_rows(num_queries: int, total_pool_rows: int) -> jax.Array:
    num_chunks = max(1, num_queries // SWA_ROWS_PER_PAGE)
    num_swa_pages = max(1, total_pool_rows // SWA_ROWS_PER_PAGE)
    chunk_indices = jnp.arange(num_chunks, dtype=jnp.int32)[:, None]
    physical_pages = (
        chunk_indices * BLOCK_HASH_PRIME + PAGE_HASH_OFFSET
    ) % num_swa_pages
    intra_page_offsets = jnp.arange(SWA_ROWS_PER_PAGE, dtype=jnp.int32)[None, :]
    return (physical_pages * SWA_ROWS_PER_PAGE + intra_page_offsets).reshape(-1)[
        :num_queries
    ]


def run_benchmarks(
    *,
    warmup: int = DEFAULT_WARMUP,
    repeat: int = DEFAULT_REPEAT,
) -> list[tuple[str, float, float, float, float]]:
    """Return (workload_name, device_us, host_us, achieved_gbps, peak_bw_pct) tuples."""
    tpu_info = _check_tpu7x()
    pool_rows = (
        POOL_VMEM_MULTIPLIER * int(tpu_info.vmem_capacity_bytes)
    ) // ck.ROW_BYTES
    peak_hbm_bw_gbps = float(tpu_info.mem_bw_bytes_per_second) / BYTES_PER_GIGABYTE
    pool = jnp.zeros((pool_rows, ck.ROW_WORDS), jnp.int32)
    write_jit = jax.jit(
        ck.write_sub_rows,
        donate_argnums=0,
        static_argnames=("page_tokens",),
    )
    pack_jit = jax.jit(ck.pack_fp8_kv_rows)
    gather_jit = jax.jit(ck.gather_cache_rows)
    swa_window_jit = jax.jit(
        partial(
            ck.gather_swa_windows,
            window_rows=SWA_ROWS_PER_SEQ,
            rows_per_page=SWA_ROWS_PER_PAGE,
            page_row_stride=SWA_ROWS_PER_PAGE,
        )
    )

    results: list[tuple[str, float, float, float, float]] = []
    header = (
        f"{'Workload':<32} | {'Device(us)':>10} | "
        f"{'Host(us)':>9} | {'BW(GB/s)':>9} | {'Peak(%)':>8}\n"
    )
    sys.stdout.write(header + "-" * (len(header) - 1) + "\n")

    def _record_workload(
        workload_name: str,
        workload_fn: Callable[[], Any],
        modelled_hbm_bytes: int,
    ) -> None:
        device_us, host_us = measure_device_and_host_us(
            workload_fn,
            warmup=warmup,
            repeat=repeat,
        )
        achieved_gbps = (modelled_hbm_bytes / BYTES_PER_GIGABYTE) / (
            device_us / MICROSECONDS_PER_SECOND
        )
        peak_bw_pct = PERCENT_SCALE * achieved_gbps / peak_hbm_bw_gbps
        results.append((workload_name, device_us, host_us, achieved_gbps, peak_bw_pct))
        sys.stdout.write(
            f"{workload_name:<32} | {device_us:10.2f} | "
            f"{host_us:9.2f} | {achieved_gbps:9.2f} | {peak_bw_pct:7.2f}%\n"
        )

    def _bench_write(
        regime_tag: str,
        num_tokens: int,
        tokens_per_seq: int,
        slice_words: int,
        page_tokens: int = ck.DEFAULT_PAGE_TOKENS,
    ) -> None:
        nonlocal pool
        sub_slots_per_row = ck.ROW_WORDS // slice_words
        slot_mapping = _paged_slots(
            num_tokens,
            tokens_per_seq,
            pool_rows * sub_slots_per_row,
            page_tokens,
        )
        updates = jnp.ones((num_tokens, slice_words), dtype=jnp.int32)
        _, sub_slot_counts = jnp.unique(
            slot_mapping // sub_slots_per_row, return_counts=True
        )
        num_touched_rows = int(sub_slot_counts.shape[0])
        num_partial_rows = int(jnp.sum(sub_slot_counts < sub_slots_per_row))
        modelled_hbm_bytes = (
            num_tokens * (slice_words + 1) * ck.WORD_BYTES
            + (num_touched_rows + num_partial_rows) * ck.ROW_BYTES
        )

        def _write_once() -> jax.Array:
            nonlocal pool
            pool = write_jit(
                pool,
                updates,
                slot_mapping,
                page_tokens=page_tokens,
            )
            return pool

        _record_workload(
            f"write_sw{slice_words}_{regime_tag}(N={num_tokens})",
            _write_once,
            modelled_hbm_bytes,
        )

    def _bench_gather(
        workload_name: str,
        gather_indices: jax.Array,
    ) -> None:
        num_gathered = int(gather_indices.shape[0])
        _record_workload(
            workload_name,
            partial(gather_jit, pool, gather_indices),
            num_gathered * GATHER_HBM_BYTES_PER_ROW,
        )

    for num_tokens, tokens_per_seq, regime_tag in BENCHMARK_REGIMES:
        num_seqs = max(1, num_tokens // tokens_per_seq)
        swa_tokens_per_seq = SWA_ROWS_PER_TOKEN * min(tokens_per_seq, SWA_WINDOW_TOKENS)
        _bench_write(
            f"swa_{regime_tag}",
            num_seqs * swa_tokens_per_seq,
            swa_tokens_per_seq,
            ck.ROW_WORDS,
            SWA_ROWS_PER_PAGE,
        )
        for slice_words in reversed(ck.VALID_SLICE_WORDS):
            _bench_write(regime_tag, num_tokens, tokens_per_seq, slice_words)
        compressed_tokens = max(1, num_tokens // COMPRESSED_RATIO_2X)
        compressed_tokens_per_seq = max(1, tokens_per_seq // COMPRESSED_RATIO_2X)
        _bench_write(
            f"pt8_{regime_tag}",
            compressed_tokens,
            compressed_tokens_per_seq,
            ck.INDEXER_KEY_WORDS,
            PT8_PAGE_TOKENS,
        )
        latent_bf16 = jnp.ones((num_tokens, ck.LATENT_DIM), dtype=jnp.bfloat16)
        _record_workload(
            f"pack_fp8_{regime_tag}(T={num_tokens})",
            partial(pack_jit, latent_bf16),
            num_tokens * PACK_FP8_HBM_BYTES_PER_TOKEN,
        )
        swa_seq_batches = (
            ((num_seqs, ""), (MIXED_PREFILL_SEQS, "_b4"))
            if num_tokens >= PREFILL_2K_TOKENS
            else ((num_seqs, ""),)
        )
        for batch_seqs, batch_suffix in swa_seq_batches:
            gather_indices = _swa_rows(batch_seqs, pool_rows)
            num_gathered = int(gather_indices.shape[0])
            _bench_gather(
                f"gather_swa_{regime_tag}{batch_suffix}(Q={num_gathered})",
                gather_indices,
            )

    for num_seqs, seq_tag in SWA_WINDOW_SEQ_COUNTS:
        page_ids, phase_rows = _paged_swa_window_inputs(num_seqs, pool_rows)
        _record_workload(
            f"gather_swa_win_{seq_tag}(S={num_seqs})",
            partial(swa_window_jit, pool, page_ids, phase_rows),
            num_seqs * SWA_WINDOW_HBM_BYTES_PER_SEQ,
        )

    aligned_b48_indices = _aligned_swa_rows(ALIGNED_BURST_QUERIES_B48, pool_rows)
    _bench_gather(
        f"gather_swa_aligned_b48(Q={ALIGNED_BURST_QUERIES_B48})",
        aligned_b48_indices,
    )
    aligned_large_indices = _aligned_swa_rows(pool_rows, pool_rows)
    _bench_gather(
        f"gather_swa_aligned_p4k(Q={pool_rows})",
        aligned_large_indices,
    )
    random_indices = (
        jnp.arange(RANDOM_GATHER_QUERIES, dtype=jnp.int32) * SEQ_PAGE_HASH_PRIME
    ) % pool_rows
    _bench_gather(
        f"gather_rand(Q={RANDOM_GATHER_QUERIES})",
        random_indices,
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    args = parser.parse_args(argv)
    run_benchmarks(warmup=args.warmup, repeat=args.repeat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
