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
"""Multi-Chip v82 CSA + Indexer Microbenchmark & Parity Suite.

Covers:
1. Chunked Prefill up to 100k:
   - L_kv in {4096, 8192, 16384, 32768, 65536, 100000}
   - q_len in {512, 1024, 2048, 4096}
2. Decode (q_len=1):
   - B in {16, 64, 128}
   - L_kv in {8192, 32768, 100000}

Supports 4-chip parallel execution on a single node via TPU_VISIBLE_CHIPS=0,1,2,3.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

PREFILL_L_KVS = [4096, 8192, 16384, 32768, 65536, 100000]
PREFILL_Q_LENS = [512, 1024, 2048, 4096]
DECODE_BS = [16, 64, 128]
DECODE_L_KVS = [8192, 32768, 100000]


def _make_random_csa_caches(rng, total_pages: int, page_size: int):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout

    fp8 = jnp.asarray(
        rng.standard_normal((total_pages, page_size, 448), dtype=np.float32) * 0.25
    ).astype(jnp.float8_e4m3fn)
    scales = jnp.asarray(
        2.0 ** rng.integers(-2, 1, (total_pages, page_size, 64)), dtype=jnp.float32
    ).astype(jnp.float8_e8m0fnu)
    nope_slabs = jnp.concatenate(
        [
            jax.lax.bitcast_convert_type(fp8, jnp.uint8),
            jax.lax.bitcast_convert_type(scales, jnp.uint8),
        ],
        axis=-1,
    ).reshape(total_pages, page_size, 4, 128)
    rope = jnp.asarray(
        rng.standard_normal((total_pages * page_size, 64), dtype=np.float32) * 0.25
    ).astype(jnp.bfloat16)
    return (
        csa_cache_layout.slabs_to_words(nope_slabs),
        csa_cache_layout.encode_rope(rope).reshape(total_pages, page_size // 4, 128),
    )


def _time_jax_fn(fn, args: tuple[Any, ...], kwargs: dict[str, Any], warmup: int, iters: int) -> tuple[Any, float]:
    import jax

    out = None
    for _ in range(warmup):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0 / iters


def _build_prefill_topk_indices(rng, q_len: int, kv_comp_len: int, topk: int, group_sz: int = 64):
    """Realistic causal top-k indices for prefill queries in [kv_comp_len - q_len//2, kv_comp_len).

    Nearby query tokens within a sequence share a strong fraction of top-k blocks
    plus query-specific local/recent blocks, matching real Indexer outputs.
    """
    import numpy as np

    rows = []
    q_comp_start = max(0, kv_comp_len - (q_len // 2))
    num_groups = (q_len + group_sz - 1) // group_sz
    for g in range(num_groups):
        g_start = g * group_sz
        g_end = min(q_len, (g + 1) * group_sz)
        vis_base = max(topk, min(kv_comp_len, q_comp_start + (g_start // 2) + 1))
        shared_pool = rng.permutation(vis_base)[: min(vis_base, int(topk * 1.25))]
        for t in range(g_start, g_end):
            vis = max(1, min(kv_comp_len, q_comp_start + (t // 2) + 1))
            valid_pool = shared_pool[shared_pool < vis]
            if len(valid_pool) >= topk:
                offset = (t - g_start) % max(1, len(valid_pool) - topk + 1)
                picked = np.sort(valid_pool[offset : offset + topk])
            else:
                picked = np.sort(rng.permutation(vis)[: min(vis, topk)])
            rows.append(
                np.pad(picked, (0, topk - len(picked)), constant_values=-1).astype(
                    np.int32
                )
            )
    return np.stack(rows, axis=0)


def run_worker(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.kernels.deepseek_v41 import compressor as v41_comp
    from vllm_torchtpu.kernels.deepseek_v41 import sparse_mla as v41_mla

    rng = np.random.default_rng(42)
    num_q_heads = args.num_q_heads
    head_dim = 512
    topk = 512
    page_size = 512  # compressed KV tokens per page (1024 tokens at compress_ratio=2)
    compress_ratio = 2
    n_idx_heads = 64
    idx_head_dim = 128

    # Build task list across prefill and decode
    tasks: list[dict[str, Any]] = []
    if args.mode in ("all", "prefill"):
        for l_kv in PREFILL_L_KVS:
            for q_len in PREFILL_Q_LENS:
                if q_len > l_kv:
                    continue
                tasks.append(
                    {
                        "kind": "prefill",
                        "name": f"prefill_L{l_kv}_Q{q_len}",
                        "B": 1,
                        "q_len": q_len,
                        "L_kv": l_kv,
                    }
                )
    if args.mode in ("all", "decode"):
        for l_kv in DECODE_L_KVS:
            for b in DECODE_BS:
                tasks.append(
                    {
                        "kind": "decode",
                        "name": f"decode_B{b}_L{l_kv}",
                        "B": b,
                        "q_len": 1,
                        "L_kv": l_kv,
                    }
                )

    my_tasks = [
        t for idx, t in enumerate(tasks) if idx % args.num_workers == args.worker_id
    ]
    print(
        f"[Worker {args.worker_id}/{args.num_workers} on devices={jax.devices()}] "
        f"Running {len(my_tasks)} benchmark tasks...",
        flush=True,
    )

    cos_sin_cache = jnp.asarray(
        rng.standard_normal((131072, 64), dtype=np.float32), dtype=jnp.float32
    )
    results: dict[str, Any] = {}

    for t in my_tasks:
        kind = t["kind"]
        name = t["name"]
        B = t["B"]
        q_len = t["q_len"]
        L_kv = t["L_kv"]
        num_tokens = B * q_len
        kv_comp_len = L_kv // compress_ratio
        pages_per_seq = max(1, (kv_comp_len + page_size - 1) // page_size)
        # Keep total pages realistic while fitting comfortably in single-chip HBM
        total_pages = min(4096, max(B * pages_per_seq, 256))

        cache_kv_nope, cache_kv_rope = _make_random_csa_caches(
            rng, total_pages, page_size
        )
        page_indices_np = rng.integers(
            0, total_pages, size=(B * pages_per_seq,), dtype=np.int32
        )
        page_indices = jnp.asarray(page_indices_np, dtype=jnp.int32)

        q_lens = np.full((B,), q_len, dtype=np.int32)
        cu_q_lens = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), dtype=jnp.int32)
        seq_lens = jnp.full((B,), L_kv, dtype=jnp.int32)

        if kind == "decode":
            distribution = jnp.array([B, B, B], dtype=jnp.int32)
            rows = []
            for _ in range(B):
                picked = np.sort(rng.permutation(kv_comp_len)[: min(kv_comp_len, topk)])
                rows.append(
                    np.pad(picked, (0, topk - len(picked)), constant_values=-1).astype(
                        np.int32
                    )
                )
            topk_indices_raw = jnp.asarray(np.stack(rows, axis=0), dtype=jnp.int32)
            positions = jnp.full((B,), L_kv - 1, dtype=jnp.int32)
        else:
            distribution = jnp.array([0, 0, B], dtype=jnp.int32)
            topk_indices_raw = jnp.asarray(
                _build_prefill_topk_indices(rng, q_len, kv_comp_len, topk),
                dtype=jnp.int32,
            )
            positions = jnp.arange(L_kv - q_len, L_kv, dtype=jnp.int32)

        encoded_topk = v41_mla.encode_csa_gather_indices(
            topk_indices_raw, page_indices, cu_q_lens, total_pages, page_size
        )
        jax.block_until_ready(encoded_topk)

        q = jnp.asarray(
            rng.standard_normal((num_tokens, num_q_heads, head_dim), dtype=np.float32)
            * 0.2,
            dtype=jnp.bfloat16,
        )
        attention_sinks = jnp.asarray(
            rng.standard_normal((num_q_heads,), dtype=np.float32) * 0.1,
            dtype=jnp.float32,
        )
        swa_acc = jnp.asarray(
            rng.standard_normal((num_tokens, num_q_heads, head_dim), dtype=np.float32)
            * 0.1,
            dtype=jnp.bfloat16,
        )
        swa_l = jnp.asarray(
            np.abs(rng.standard_normal((num_tokens, num_q_heads), dtype=np.float32))
            + 1.0,
            dtype=jnp.float32,
        )
        swa_m = jnp.asarray(
            rng.standard_normal((num_tokens, num_q_heads), dtype=np.float32) * 0.1,
            dtype=jnp.float32,
        )

        # 1. Benchmark CSA (`sparse_ragged_paged_attention`)
        csa_args = (
            q,
            cache_kv_nope,
            cache_kv_rope,
            encoded_topk,
            cu_q_lens,
            distribution,
            attention_sinks,
            swa_acc,
            swa_l,
            swa_m,
        )
        csa_kwargs: dict[str, Any] = {"sm_scale": 0.1352337788608801}
        csa_out, csa_ms = _time_jax_fn(
            v41_mla.sparse_ragged_paged_attention,
            csa_args,
            csa_kwargs,
            warmup=args.warmup,
            iters=args.iters,
        )

        rec: dict[str, Any] = {
            "kind": kind,
            "B": B,
            "q_len": q_len,
            "L_kv": L_kv,
            "csa_ms": round(csa_ms, 4),
            "csa_out_mean": float(jnp.mean(csa_out.astype(jnp.float32))),
            "csa_out_std": float(jnp.std(csa_out.astype(jnp.float32))),
        }
        if kind == "prefill" and hasattr(v41_mla, "precompute_csa_union_and_bitmask"):
            pre_union = v41_mla.precompute_csa_union_and_bitmask(
                encoded_topk, total_cache_rows=total_pages * page_size
            )
            jax.block_until_ready(pre_union[0])
            _, consumer_ms = _time_jax_fn(
                v41_mla.sparse_ragged_paged_attention,
                csa_args,
                {**csa_kwargs, "precomputed_union": pre_union},
                warmup=args.warmup,
                iters=args.iters,
            )
            rec["csa_consumer_ms"] = round(consumer_ms, 4)
        if args.save_outputs_dir:
            os.makedirs(args.save_outputs_dir, exist_ok=True)
            np.save(
                os.path.join(args.save_outputs_dir, f"{name}_csa.npy"),
                np.asarray(csa_out, dtype=np.float32),
            )
        if args.compare_outputs_dir:
            ref_path = os.path.join(args.compare_outputs_dir, f"{name}_csa.npy")
            if os.path.exists(ref_path):
                ref_arr = np.load(ref_path)
                diff = float(np.max(np.abs(np.asarray(csa_out, dtype=np.float32) - ref_arr)))
                rec["csa_max_abs_diff"] = round(diff, 6)

        # 2. Benchmark Indexer (`indexer_produce_k_jax` + `indexer_v41_jax`) if enabled
        if not args.skip_indexer:
            k_cache = jnp.zeros((total_pages, 33, 4, page_size), dtype=jnp.uint8)
            k_normed = jnp.asarray(
                rng.standard_normal((num_tokens, idx_head_dim), dtype=np.float32) * 0.2,
                dtype=jnp.bfloat16,
            )
            prod_k_fn = jax.jit(
                v41_comp.indexer_produce_k_jax,
                static_argnames=("compress_ratio",),
            )
            _, prod_k_ms = _time_jax_fn(
                prod_k_fn,
                (
                    k_normed,
                    cos_sin_cache,
                    positions,
                    page_indices,
                    cu_q_lens,
                    distribution,
                    k_cache,
                ),
                {"compress_ratio": compress_ratio},
                warmup=args.warmup,
                iters=args.iters,
            )
            rec["indexer_produce_k_ms"] = round(prod_k_ms, 4)

            idx_q = jnp.asarray(
                rng.standard_normal(
                    (num_tokens, n_idx_heads, idx_head_dim), dtype=np.float32
                )
                * 0.2,
                dtype=jnp.bfloat16,
            )
            idx_weights = jnp.asarray(
                np.abs(rng.standard_normal((num_tokens, n_idx_heads), dtype=np.float32))
                * 0.05,
                dtype=jnp.bfloat16,
            )
            idx_fn = jax.jit(
                v41_comp.indexer_v41_jax,
                static_argnames=(
                    "k",
                    "compression_ratio",
                    "softmax_scale",
                    "n_head",
                    "cand_mode",
                    "cand_blocks",
                    "cand_block_size",
                    "has_precomputed_meta",
                ),
            )
            _, idx_ms = _time_jax_fn(
                idx_fn,
                (
                    idx_q,
                    positions,
                    cos_sin_cache,
                    idx_weights,
                    k_cache,
                    seq_lens,
                    page_indices,
                    cu_q_lens,
                    distribution,
                    page_indices,
                ),
                {
                    "k": topk,
                    "compression_ratio": compress_ratio,
                    "softmax_scale": 0.1352337788608801,
                    "n_head": n_idx_heads,
                },
                warmup=args.warmup,
                iters=args.iters,
            )
            rec["indexer_topk_ms"] = round(idx_ms, 4)

        results[name] = rec
        print(f"[Worker {args.worker_id}] {name}: {rec}", flush=True)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="v82 CSA + Indexer Multi-Chip Benchmark")
    parser.add_argument("--mode", choices=["all", "prefill", "decode"], default="all")
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--skip-indexer", action="store_true")
    parser.add_argument("--worker-id", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--save-outputs-dir", type=str, default=None)
    parser.add_argument("--compare-outputs-dir", type=str, default=None)
    args = parser.parse_args()

    if args.worker_id is not None:
        run_worker(args)
        return

    # Orchestrate 4 parallel processes on TPU_VISIBLE_CHIPS=0,1,2,3
    procs = []
    tmp_jsons = []
    for w in range(args.num_workers):
        out_j = f"/tmp/bench_v82_worker_{w}.json"
        tmp_jsons.append(out_j)
        if os.path.exists(out_j):
            os.remove(out_j)
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--mode",
            args.mode,
            "--num-q-heads",
            str(args.num_q_heads),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--worker-id",
            str(w),
            "--num-workers",
            str(args.num_workers),
            "--output-json",
            out_j,
        ]
        if args.skip_indexer:
            cmd.append("--skip-indexer")
        if args.save_outputs_dir:
            cmd.extend(["--save-outputs-dir", args.save_outputs_dir])
        if args.compare_outputs_dir:
            cmd.extend(["--compare-outputs-dir", args.compare_outputs_dir])
        env = os.environ.copy()
        env["TPU_VISIBLE_CHIPS"] = str(w)
        env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
        env["TPU_PROCESS_BOUNDS"] = "1,1,1"
        env["TPU_WORKER_ID"] = "0"
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(1.5)

    failed = False
    for w, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            print(f"[ERROR] Worker {w} failed with exit code {rc}", file=sys.stderr)
            failed = True
    if failed:
        sys.exit(1)

    merged: dict[str, Any] = {}
    for out_j in tmp_jsons:
        if os.path.exists(out_j):
            with open(out_j) as f:
                merged.update(json.load(f))

    print("\n================ MERGED BENCHMARK SUMMARY ================")
    for k in sorted(merged.keys()):
        print(f"  {k:24s}: {merged[k]}")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(merged, f, indent=2)


if __name__ == "__main__":
    main()
