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
"""Standalone TPU7x microbenchmark for DeepSeek-V4.1 word-row cache kernels.

Sizes each HBM bank at ``BANK_VMEM_MULTIPLIER = 8`` times TPU7x VMEM capacity
(``8 * 64 MiB = 512 MiB`` per bank, ``1.0 GiB`` across 2 banks) so the compiler
cannot prefetch cache pools into ``64 MiB`` VMEM and all measurements reflect
true HBM-resident bandwidth. Reports both hardware XProf on-device latency
(``Device(us)`` from ``/device:TPU:0`` ``XLA Modules`` via
``jax.profiler.ProfileData``) and single-call dispatch-inclusive host latency
(``Host(us)``).
"""

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

# HBM pool sizing: 8x VMEM per bank (512 MiB/bank, 1.0 GiB across 2 banks on TPU7x).
BANK_VMEM_MULTIPLIER = 8
SMOKE_BANK_ROWS = 64
SMOKE_GATHER_LIMIT = 64
SMOKE_RANDOM_QUERIES = 32
NUM_CACHE_BANKS = 2
TPU_DEVICE_PLANE = "/device:TPU:0"
XLA_MODULES_LINE = "XLA Modules"
NANOSECONDS_PER_MICROSECOND = 1e3

# DeepSeek-V4.1 SWA & cache row byte-accounting constants.
SWA_WINDOW_TOKENS = 128
SWA_ROWS_PER_TOKEN = 2
SWA_ROWS_PER_PAGE = ck.DEFAULT_PAGE_TOKENS * SWA_ROWS_PER_TOKEN
SWA_ROWS_PER_SEQ = SWA_WINDOW_TOKENS * SWA_ROWS_PER_TOKEN
PACK_FP8_INPUT_BYTES_PER_TOKEN = ck.LATENT_DIM * 2
PACK_FP8_OUTPUT_BYTES_PER_TOKEN = (ck.ROW_WORDS + ck.ROPE_WORDS) * ck.WORD_BYTES
PACK_FP8_HBM_BYTES_PER_TOKEN = (
    PACK_FP8_INPUT_BYTES_PER_TOKEN + PACK_FP8_OUTPUT_BYTES_PER_TOKEN
)
GATHER_HBM_BYTES_PER_ROW = 2 * ck.ROW_BYTES
COMPRESSED_RATIO_2X = 2

# Representative DeepSeek-V4.1 serving batch regimes (TP=1, DP=8 per die).
DECODE_BATCH_SEQS = 48
MTP_VERIFY_TOKENS = 6
MTP_BATCH_TOKENS = DECODE_BATCH_SEQS * MTP_VERIFY_TOKENS
PREFILL_2K_TOKENS = 2048
PREFILL_4K_TOKENS = 4096
SMOKE_BATCH_TOKENS = 16
MIXED_PREFILL_SEQS = 4
RANDOM_GATHER_QUERIES = DECODE_BATCH_SEQS * DECODE_BATCH_SEQS
ALIGNED_BURST_QUERIES_B48 = DECODE_BATCH_SEQS * SWA_ROWS_PER_SEQ

# Coprime hash strides and non-aligned offsets to model realistic paged tables
# across the full 2-bank HBM pool without synthetic sequential prefetching:
#   - SEQ_PAGE_HASH_PRIME (104729) & BLOCK_HASH_PRIME (7919) scatter logical
#     (seq_idx, block_idx) pairs across physical pages, with PAGE_HASH_OFFSET (17)
#     avoiding physical page 0.
#   - UNALIGNED_BASE_CONTEXT_POS (261, where 261 % 16 = 5) and SEQ_CONTEXT_STRIDE
#     (37, coprime to 16) ensure each 128-token SWA window straddles 9 non-
#     contiguous 16-token pages (11 tail tokens + 7 full pages + 5 head tokens).
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
    (PREFILL_2K_TOKENS, PREFILL_2K_TOKENS, "p2k"),
    (PREFILL_4K_TOKENS, PREFILL_4K_TOKENS, "p4k"),
)
SMOKE_REGIMES = ((SMOKE_BATCH_TOKENS, SMOKE_BATCH_TOKENS, "smoke"),)


def _measure_device_and_host_us(
    workload_fn: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
    interpret: bool = False,
) -> tuple[float, float]:
    """Measure median on-device XLA module latency and host latency in microseconds.

    Args:
        workload_fn: Zero-argument callable returning one or more JAX arrays.
        warmup: Number of warmup iterations before timing.
        repeat: Number of timed iterations.
        interpret: When ``True`` (or on CPU backend), fall back to host wall time.

    Returns:
        Tuple of ``(device_us, host_us)`` median latencies across ``repeat`` calls.
    """
    for _ in range(warmup):
        jax.tree_util.tree_map(lambda arr: arr.block_until_ready(), workload_fn())
    host_samples_us: list[float] = []
    if interpret or jax.default_backend() != "tpu":
        for _ in range(repeat):
            start_time = time.perf_counter()
            jax.tree_util.tree_map(lambda arr: arr.block_until_ready(), workload_fn())
            elapsed_us = (time.perf_counter() - start_time) * MICROSECONDS_PER_SECOND
            host_samples_us.append(elapsed_us)
        host_us = float(statistics.median(host_samples_us))
        return host_us, host_us

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
        profile_data = jax.profiler.ProfileData.from_file(xplane_files[0])
        device_samples_us: list[float] = []
        for plane in profile_data.planes:
            if plane.name == TPU_DEVICE_PLANE:
                for line in plane.lines:
                    if line.name == XLA_MODULES_LINE:
                        device_samples_us = [
                            event.duration_ns / NANOSECONDS_PER_MICROSECOND
                            for event in line.events
                        ]
                        break
        device_us = (
            float(statistics.median(device_samples_us))
            if device_samples_us
            else host_us
        )
        return device_us, host_us


def _paged_slots(
    num_tokens: int,
    tokens_per_seq: int,
    total_bank_slots: int,
    page_tokens: int = ck.DEFAULT_PAGE_TOKENS,
) -> jax.Array:
    """Generate realistic paged sub-slot indices across ``total_bank_slots``.

    Maps contiguous intra-page token offsets ``token_idx % page_tokens`` onto
    pseudo-randomly allocated physical pages so full pages remain contiguous
    within a page while distinct logical blocks map to non-adjacent pages.

    Args:
        num_tokens: Total token updates in the batch.
        tokens_per_seq: Query tokens per sequence in the batch.
        total_bank_slots: Total sub-slots available in the target cache bank.
        page_tokens: Number of sub-slots per physical page.

    Returns:
        ``int32[num_tokens]`` sub-slot mapping array.
    """
    token_indices = jnp.arange(num_tokens, dtype=jnp.int32)
    num_pages = max(1, total_bank_slots // page_tokens)
    seq_indices = token_indices // tokens_per_seq
    block_in_seq = (token_indices % tokens_per_seq) // page_tokens
    physical_page = (
        seq_indices * SEQ_PAGE_HASH_PRIME
        + block_in_seq * BLOCK_HASH_PRIME
        + PAGE_HASH_OFFSET
    ) % num_pages
    return physical_page * page_tokens + (token_indices % page_tokens)


def _swa_rows(num_seqs: int, total_pool_rows: int) -> jax.Array:
    """Generate phase-unaligned 128-token SWA window row indices (2 rows/token).

    Args:
        num_seqs: Number of sequences in the batch.
        total_pool_rows: Total 128-word rows across the 2-bank HBM pool.

    Returns:
        ``int32[num_seqs * 256]`` row indices straddling non-contiguous pages.
    """
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


def _aligned_swa_rows(num_queries: int, total_pool_rows: int) -> jax.Array:
    """Generate 32-row page-aligned burst gather indices across 2 HBM banks.

    Args:
        num_queries: Total 128-word rows to gather (multiple of 32).
        total_pool_rows: Total 128-word rows across the 2-bank HBM pool.

    Returns:
        ``int32[num_queries]`` row indices organized in 32-row contiguous pages.
    """
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
    interpret: bool = False,
    warmup: int = 5,
    repeat: int = 20,
) -> list[tuple[str, float, float, float, float]]:
    """Run DeepSeek-V4.1 cache kernel benchmarks across 2-bank HBM pools.

    Args:
        interpret: Run in CPU interpret smoke mode when ``True``.
        warmup: Number of warmup calls per workload.
        repeat: Number of timed calls per workload.

    Returns:
        List of ``(workload_name, device_us, host_us, achieved_gbps, peak_bw_pct)``
        tuples.
    """
    tpu_info = pltpu.get_tpu_info()
    full_bank_rows = (
        BANK_VMEM_MULTIPLIER * int(tpu_info.vmem_capacity_bytes)
    ) // ck.ROW_BYTES
    bank_rows = SMOKE_BANK_ROWS if interpret else full_bank_rows
    regimes = SMOKE_REGIMES if interpret else BENCHMARK_REGIMES
    peak_hbm_bw_gbps = float(tpu_info.mem_bw_bytes_per_second) / BYTES_PER_GIGABYTE
    pool_lo = jnp.zeros((bank_rows, ck.ROW_WORDS), jnp.int32)
    pool_hi = jnp.ones((bank_rows, ck.ROW_WORDS), jnp.int32)
    write_jit = jax.jit(
        partial(ck.write_sub_rows, interpret=interpret),
        donate_argnums=0,
        static_argnames=("page_tokens",),
    )
    pack_jit = jax.jit(partial(ck.pack_fp8_kv_rows, interpret=interpret))
    gather_jit = jax.jit(partial(ck.gather_cache_rows, interpret=interpret))

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
        device_us, host_us = _measure_device_and_host_us(
            workload_fn,
            warmup=warmup,
            repeat=repeat,
            interpret=interpret,
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
        nonlocal pool_lo
        sub_slots_per_row = ck.ROW_WORDS // slice_words
        slot_mapping = _paged_slots(
            num_tokens,
            tokens_per_seq,
            bank_rows * sub_slots_per_row,
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
            nonlocal pool_lo
            pool_lo = write_jit(
                pool_lo,
                updates,
                slot_mapping,
                page_tokens=page_tokens,
            )
            return pool_lo

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
            partial(gather_jit, pool_lo, gather_indices, pool_hi=pool_hi),
            num_gathered * GATHER_HBM_BYTES_PER_ROW,
        )

    total_pool_rows = NUM_CACHE_BANKS * bank_rows
    for num_tokens, tokens_per_seq, regime_tag in regimes:
        num_seqs = num_tokens // tokens_per_seq
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
            ck.COMPRESSED_PAGE_TOKENS,
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
            max_gather_queries = (
                SMOKE_GATHER_LIMIT if interpret else SWA_ROWS_PER_SEQ * batch_seqs
            )
            gather_indices = _swa_rows(batch_seqs, total_pool_rows)[:max_gather_queries]
            num_gathered = int(gather_indices.shape[0])
            _bench_gather(
                f"gather_swa_{regime_tag}{batch_suffix}(Q={num_gathered})",
                gather_indices,
            )

    aligned_b48_queries = SMOKE_GATHER_LIMIT if interpret else ALIGNED_BURST_QUERIES_B48
    aligned_b48_indices = _aligned_swa_rows(aligned_b48_queries, total_pool_rows)
    _bench_gather(
        f"gather_swa_aligned_b48(Q={aligned_b48_queries})",
        aligned_b48_indices,
    )
    aligned_large_queries = SMOKE_GATHER_LIMIT if interpret else bank_rows
    aligned_large_indices = _aligned_swa_rows(aligned_large_queries, total_pool_rows)
    _bench_gather(
        f"gather_swa_aligned_p4k(Q={aligned_large_queries})",
        aligned_large_indices,
    )
    random_queries = SMOKE_RANDOM_QUERIES if interpret else RANDOM_GATHER_QUERIES
    random_indices = (
        jnp.arange(random_queries, dtype=jnp.int32) * SEQ_PAGE_HASH_PRIME
    ) % total_pool_rows
    _bench_gather(
        f"gather_rand(Q={random_queries})",
        random_indices,
    )
    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for DeepSeek-V4.1 cache kernel microbenchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args(argv)
    run_benchmarks(
        interpret=args.interpret,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
