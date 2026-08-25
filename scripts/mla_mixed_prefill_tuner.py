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
"""Tune the MLA v2 Pallas kernel for a mixed-prefill invocation.

The upstream kernel tuner currently generates batched-decode cases only.  This
small, fault-isolated driver exercises the mixed path directly.  Each candidate
runs in a fresh subprocess so a compiler OOM cannot poison the rest of a sweep.

The defaults reproduce Kimi-K3 TP32's 8K prefill tuning key on TPU v7x:
three local query heads, BF16 Q, FP8 KV, 16-token pages, and metadata padded to
eight sequences with 520 page-table entries per sequence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MIB = 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate",
                        action="append",
                        default=[],
                        help="BQ,BKVP,Q_SPLIT,VMEM_MIB; repeat as needed")
    parser.add_argument("--bq", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--bkvp", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--q-split", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--vmem-mib", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--lkv-dim", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--kv-packing", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--pages-per-seq", type=int, default=520)
    parser.add_argument("--total-pages", type=int, default=2112)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output",
                        type=Path,
                        help="Write aggregate JSON results here")
    return parser


def _parse_candidate(value: str) -> tuple[int, int, int, int]:
    fields = tuple(int(field) for field in value.split(","))
    if len(fields) != 4:
        raise ValueError(
            f"candidate must be BQ,BKVP,Q_SPLIT,VMEM_MIB: {value}")
    bq, bkvp, q_split, vmem_mib = fields
    if min(fields) <= 0 or bq % q_split:
        raise ValueError(f"invalid candidate: {value}")
    return bq, bkvp, q_split, vmem_mib


def _default_candidates() -> list[tuple[int, int, int, int]]:
    # Establish the principal tiles, then probe the high-throughput/VMEM
    # boundary.  The latter is important for 8K: q splitting makes the largest
    # viable KV-page tile fit even when q_split=1 does not.
    tiles = itertools.product((16, 32, 64, 128, 256, 512), (1, 2, 4, 8, 16))
    base = [(bq, bkvp, 1, 60) for bq, bkvp in tiles]
    split = [(bq, bkvp, q_split, 60) for bq in (256, 512)
             for bkvp in (16, 32, 64, 128, 256, 512) for q_split in (2, 4, 8)
             if bq % q_split == 0]
    vmem_boundary = [(512, 256, 4, vmem_mib) for vmem_mib in (48, 56, 64)]
    return base + split + vmem_boundary


def _run_child(args: argparse.Namespace) -> None:
    # Keep all JAX imports out of the parent process.  TPU runtime failures are
    # therefore isolated to this one candidate.
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.mla.v2.kernel import mla_ragged_paged_attention

    if args.result_path is None:
        raise ValueError("--result-path is required in child mode")
    if args.page_size % args.kv_packing:
        raise ValueError("page size must be divisible by KV packing")
    needed_pages = (args.tokens + args.page_size - 1) // args.page_size
    if needed_pages > args.pages_per_seq or needed_pages > args.total_pages:
        raise ValueError("page-table/cache capacity is too small")

    ql_nope = jnp.ones((args.num_heads, args.tokens, args.lkv_dim),
                       jnp.bfloat16)
    q_pe = jnp.ones((args.tokens, args.num_heads, args.rope_dim), jnp.bfloat16)
    new_kv_c = jnp.ones((args.tokens, args.lkv_dim), jnp.float8_e4m3fn)
    new_k_pe = jnp.ones((args.tokens, args.rope_dim), jnp.float8_e4m3fn)
    padded_kv_dim = ((args.lkv_dim + 127) // 128 * 128 +
                     (args.rope_dim + 127) // 128 * 128)
    cache_kv = jnp.zeros((args.total_pages, args.page_size // args.kv_packing,
                          args.kv_packing, padded_kv_dim), jnp.float8_e4m3fn)
    kv_lens = jnp.asarray([args.tokens] + [0] * (args.max_num_seqs - 1),
                          jnp.int32)
    page_row = list(
        range(needed_pages)) + [-1] * (args.pages_per_seq - needed_pages)
    page_indices = jnp.asarray(
        page_row + [-1] * (args.pages_per_seq * (args.max_num_seqs - 1)),
        jnp.int32)
    cu_q_lens = jnp.asarray([0, args.tokens] + [args.tokens] *
                            (args.max_num_seqs - 1), jnp.int32)
    distribution = jnp.asarray([0, 0, 1], jnp.int32)

    def invoke(cache):
        return mla_ragged_paged_attention(
            ql_nope=ql_nope,
            q_pe=q_pe,
            new_kv_c=new_kv_c,
            new_k_pe=new_k_pe,
            cache_kv=cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            num_kv_pages_per_block=(3, 1, args.bkvp),
            num_queries_per_block=(1, 16, args.bq),
            vmem_limit_bytes=args.vmem_mib * MIB,
            mixed_q_split=args.q_split,
            s_dtype=jnp.bfloat16,
            p_same_dtype_as_v=True,
        )

    result = {
        "bq": args.bq,
        "bkvp": args.bkvp,
        "q_split": args.q_split,
        "vmem_mib": args.vmem_mib,
        "status": "unknown_error",
    }
    try:
        # The first invocation includes compilation.  Warmups and measurements
        # reuse that executable and synchronize both aliased outputs.
        compile_start = time.perf_counter()
        output, cache_kv = invoke(cache_kv)
        jax.block_until_ready((output, cache_kv))
        result["compile_and_first_s"] = time.perf_counter() - compile_start
        for _ in range(args.warmups):
            output, cache_kv = invoke(cache_kv)
            jax.block_until_ready((output, cache_kv))
        samples_ms = []
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            output, cache_kv = invoke(cache_kv)
            jax.block_until_ready((output, cache_kv))
            samples_ms.append((time.perf_counter_ns() - start) / 1e6)

        # All values use V=1, so causal attention must produce one everywhere.
        # This cheaply rejects candidates with silent numerical corruption.
        host_output = np.asarray(output, dtype=np.float32)
        result.update({
            "status":
            "success",
            "samples_ms":
            samples_ms,
            "median_ms":
            statistics.median(samples_ms),
            "mean_ms":
            statistics.fmean(samples_ms),
            "min_ms":
            min(samples_ms),
            "max_abs_error_from_one":
            float(np.max(np.abs(host_output - 1.0))),
            "finite":
            bool(np.isfinite(host_output).all()),
        })
        if not result["finite"] or result["max_abs_error_from_one"] > 0.02:
            result["status"] = "incorrect"
    except Exception as error:  # A structured result is more useful in a sweep.
        result["status"] = "oom" if "RESOURCE_EXHAUSTED" in str(
            error) else "error"
        result["error"] = repr(error)
    args.result_path.write_text(json.dumps(result, indent=2) + "\n")


def _run_parent(args: argparse.Namespace) -> int:
    candidates = ([_parse_candidate(value) for value in args.candidate]
                  if args.candidate else _default_candidates())
    common = [
        f"--tokens={args.tokens}",
        f"--num-heads={args.num_heads}",
        f"--lkv-dim={args.lkv_dim}",
        f"--rope-dim={args.rope_dim}",
        f"--page-size={args.page_size}",
        f"--kv-packing={args.kv_packing}",
        f"--max-num-seqs={args.max_num_seqs}",
        f"--pages-per-seq={args.pages_per_seq}",
        f"--total-pages={args.total_pages}",
        f"--warmups={args.warmups}",
        f"--iterations={args.iterations}",
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="mla-mixed-tuner-") as temp_dir:
        for index, (bq, bkvp, q_split, vmem_mib) in enumerate(candidates):
            result_path = Path(temp_dir) / f"result-{index}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                f"--bq={bq}",
                f"--bkvp={bkvp}",
                f"--q-split={q_split}",
                f"--vmem-mib={vmem_mib}",
                f"--result-path={result_path}",
                *common,
            ]
            print(
                f"[{index + 1}/{len(candidates)}] bq={bq} bkvp={bkvp} "
                f"q_split={q_split} vmem={vmem_mib}MiB",
                flush=True)
            completed = subprocess.run(command,
                                       env=os.environ.copy(),
                                       check=False)
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                result = {
                    "bq": bq,
                    "bkvp": bkvp,
                    "q_split": q_split,
                    "vmem_mib": vmem_mib,
                    "status": "process_error",
                    "returncode": completed.returncode,
                }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            if args.output:
                args.output.write_text(json.dumps(results, indent=2) + "\n")

    valid = [result for result in results if result["status"] == "success"]
    valid.sort(key=lambda result: result["median_ms"])
    print("\nRanked successful candidates:")
    for rank, result in enumerate(valid, 1):
        print(f"{rank:2d}. median={result['median_ms']:.3f} ms "
              f"bq={result['bq']} bkvp={result['bkvp']} "
              f"q_split={result['q_split']} vmem={result['vmem_mib']}MiB")
    return 0 if valid else 1


def main() -> int:
    args = _parser().parse_args()
    if args.child:
        _run_child(args)
        return 0
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
