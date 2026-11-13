#!/usr/bin/env python3
"""Multi-chip standalone benchmark & numerical parity suite for DeepSeek-V4.1 SWA kernel.

Benchmarks `_run_swa` and `mla_sliding_window_ragged_paged_attention` across:
  1. Chunked Prefill up to 100k:
     L_kv in {4096, 8192, 16384, 32768, 65536, 100000}
     q_len in {512, 1024, 2048, 4096}
     num_q_heads in {16, 64, 8} (DP=8 / TP=8 head-sharded)
  2. Decode (q_len=1 per sequence):
     B in {16, 64, 128} at L_kv in {8192, 32768, 100000}
     B = 512 at L_kv = 8192

Supports 4-chip parallel execution on tpu7x-8 (2x2x1) via TPU_VISIBLE_CHIPS={0,1,2,3}.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np


def _make_inputs(
    jax,
    jnp,
    *,
    num_seqs: int,
    q_len: int,
    kv_len: int,
    num_q_heads: int,
    head_dim: int = 512,
    logical_page_size: int = 16,
    sliding_window: int = 128,
    max_num_seqs_padded: int = 512,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    num_tokens = num_seqs * q_len
    pages_per_seq = (kv_len + logical_page_size - 1) // logical_page_size + 4
    # Allocate realistic sw_cache pages (matching DeepSeek-V4.1-Flash CSA/SWA host layout)
    # Use enough physical pages so ring wrapping works cleanly without excessive HBM waste
    window_pages_per_seq = (q_len + sliding_window + logical_page_size - 1) // logical_page_size + 4
    total_phys_pages = max(256, num_seqs * window_pages_per_seq + 16)

    q_np = rng.standard_normal((num_tokens, num_q_heads, head_dim), dtype=np.float32) * 0.2
    kv_np = rng.standard_normal((num_tokens, head_dim), dtype=np.float32) * 0.2
    sinks_np = rng.standard_normal((num_q_heads,), dtype=np.float32) * 0.1

    kv_lens_np = np.zeros((max_num_seqs_padded,), dtype=np.int32)
    kv_lens_np[:num_seqs] = kv_len

    cu_q_lens_np = np.zeros((max_num_seqs_padded + 1,), dtype=np.int32)
    for i in range(num_seqs):
        cu_q_lens_np[i + 1] = (i + 1) * q_len
    cu_q_lens_np[num_seqs + 1 :] = num_tokens

    page_table_np = np.zeros((max_num_seqs_padded, pages_per_seq), dtype=np.int32)
    for s in range(num_seqs):
        for p in range(pages_per_seq):
            page_table_np[s, p] = 1 + ((s * pages_per_seq + p) % (total_phys_pages - 1))

    is_decode = q_len == 1
    if is_decode:
        dist_np = np.array([num_seqs, num_seqs, num_seqs], dtype=np.int32)
    else:
        # In _run_swa / DeepSeek-V4.1, chunk_prefill_size is None so prefill runs in MIXED slot [dist[1]:dist[2]]
        dist_np = np.array([0, 0, num_seqs], dtype=np.int32)

    q = jnp.asarray(q_np, dtype=jnp.bfloat16)
    new_kv = jnp.asarray(kv_np, dtype=jnp.bfloat16)
    # Initialize CSA host int32 [total_phys_pages, 128, 128] with valid bf16 values (no NaNs!)
    cache_bf16 = jnp.asarray(
        rng.standard_normal((total_phys_pages * 64, head_dim), dtype=np.float32) * 0.2,
        dtype=jnp.bfloat16,
    )
    from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import prepare_kv_inputs

    sw_cache_u8 = prepare_kv_inputs(cache_bf16).reshape(total_phys_pages, 128, 4, 128)
    sw_cache_init = jax.lax.bitcast_convert_type(
        jnp.moveaxis(sw_cache_u8, 2, -1), jnp.int32
    )
    swa_kv_lens = jnp.asarray(kv_lens_np, dtype=jnp.int32)
    swa_page_indices = jnp.asarray(page_table_np.reshape(-1), dtype=jnp.int32)
    swa_cu_q_lens = jnp.asarray(cu_q_lens_np, dtype=jnp.int32)
    swa_distribution = jnp.asarray(dist_np, dtype=jnp.int32)
    attention_sinks = jnp.asarray(sinks_np, dtype=jnp.float32)

    return {
        "q": q,
        "new_kv": new_kv,
        "sw_cache": sw_cache_init,
        "swa_kv_lens": swa_kv_lens,
        "swa_page_indices": swa_page_indices,
        "swa_cu_q_lens": swa_cu_q_lens,
        "swa_distribution": swa_distribution,
        "attention_sinks": attention_sinks,
    }


def _baseline_run_swa(
    jax,
    jnp,
    inputs: dict[str, Any],
    *,
    sliding_window: int = 128,
    logical_page_size: int = 16,
    sm_scale: float = 512 ** -0.5,
):
    """Exact original v81/v82 baseline before any v82 optimization."""
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_flash_attention_op import (
        _prepare_swa_metadata,
    )
    from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import (
        mla_sliding_window_ragged_paged_attention,
    )
    from vllm_torchtpu.utils import align_to

    q = inputs["q"]
    new_kv = inputs["new_kv"]
    sw_cache = inputs["sw_cache"]
    swa_kv_lens = inputs["swa_kv_lens"]
    swa_page_indices = inputs["swa_page_indices"]
    swa_cu_q_lens = inputs["swa_cu_q_lens"]
    swa_distribution = inputs["swa_distribution"]
    attention_sinks = inputs["attention_sinks"]

    max_num_seqs = swa_kv_lens.shape[0]
    eff_max_seqs = min(max_num_seqs, max(8, align_to(q.shape[0], 8)))
    if eff_max_seqs < max_num_seqs:
        pages_per_seq = swa_page_indices.size // max_num_seqs
        swa_kv_lens = swa_kv_lens[:eff_max_seqs]
        swa_page_indices = swa_page_indices[: eff_max_seqs * pages_per_seq]
        swa_cu_q_lens = swa_cu_q_lens[: eff_max_seqs + 1]
        swa_distribution = jnp.minimum(swa_distribution, eff_max_seqs)
    swa_kv_lens, swa_page_indices = _prepare_swa_metadata(
        swa_kv_lens,
        swa_page_indices,
        swa_cu_q_lens,
        num_tokens=q.shape[0],
        sw_cache_num_pages=sw_cache.shape[0],
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
    )
    return mla_sliding_window_ragged_paged_attention(
        q=q,
        new_kv=new_kv,
        cache_kv=sw_cache,
        kv_lens=swa_kv_lens,
        page_indices=swa_page_indices,
        cu_q_lens=swa_cu_q_lens,
        distribution=swa_distribution,
        attention_sinks=attention_sinks,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        logical_page_size=logical_page_size,
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 32, 8 if q.shape[0] <= 384 else 32),
        q_compute_block_size=4,
        decode_seq_batch_size=32,
        unnormalized_output=False,
        ced_decoder=False,
    )


def _worker_main(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_flash_attention_op import (
        _prepare_swa_metadata,
        _run_swa,
    )
    from vllm_torchtpu.kernels.deepseek_v4.core_attention.mla_swa import (
        mla_sliding_window_ragged_paged_attention,
    )
    from vllm_torchtpu.utils import align_to

    cases = json.loads(args.cases_json)
    results = []

    for case in cases:
        mode = case["mode"]  # "prefill" or "decode" or "sweep_prefill" or "sweep_decode"
        num_seqs = case["num_seqs"]
        q_len = case["q_len"]
        kv_len = case["kv_len"]
        num_q_heads = case["num_q_heads"]
        inp = _make_inputs(
            jax,
            jnp,
            num_seqs=num_seqs,
            q_len=q_len,
            kv_len=kv_len,
            num_q_heads=num_q_heads,
        )

        # Reference output from baseline params
        ref_fn = jax.jit(
            lambda q, kv, c, kl, pi, cq, d, s: _baseline_run_swa(
                jax,
                jnp,
                {
                    "q": q,
                    "new_kv": kv,
                    "sw_cache": c,
                    "swa_kv_lens": kl,
                    "swa_page_indices": pi,
                    "swa_cu_q_lens": cq,
                    "swa_distribution": d,
                    "attention_sinks": s,
                },
            ),
            donate_argnums=(2,),
        )
        ref_out, ref_cache, ref_l, ref_m = ref_fn(
            inp["q"],
            inp["new_kv"],
            inp["sw_cache"].copy(),
            inp["swa_kv_lens"],
            inp["swa_page_indices"],
            inp["swa_cu_q_lens"],
            inp["swa_distribution"],
            inp["attention_sinks"],
        )
        jax.block_until_ready(ref_out)

        # Benchmark baseline
        for _ in range(3):
            o_b, _, _, _ = ref_fn(
                inp["q"],
                inp["new_kv"],
                inp["sw_cache"].copy(),
                inp["swa_kv_lens"],
                inp["swa_page_indices"],
                inp["swa_cu_q_lens"],
                inp["swa_distribution"],
                inp["attention_sinks"],
            )
            jax.block_until_ready(o_b)

        c_bufs = [inp["sw_cache"].copy() for _ in range(args.iters)]
        jax.block_until_ready(c_bufs[-1])
        t0 = time.perf_counter()
        for it in range(args.iters):
            o_b, c_b, _, _ = ref_fn(
                inp["q"],
                inp["new_kv"],
                c_bufs[it],
                inp["swa_kv_lens"],
                inp["swa_page_indices"],
                inp["swa_cu_q_lens"],
                inp["swa_distribution"],
                inp["attention_sinks"],
            )
        jax.block_until_ready(o_b)
        base_us = (time.perf_counter() - t0) * 1e6 / args.iters

        if case.get("custom_params"):
            cp = case["custom_params"]

            def _custom_fn(q, kv, c, kl, pi, cq, d, s):
                max_num_seqs = kl.shape[0]
                eff_max_seqs = min(max_num_seqs, max(8, align_to(q.shape[0], 8)))
                if eff_max_seqs < max_num_seqs:
                    pages_per_seq = pi.size // max_num_seqs
                    kl = kl[:eff_max_seqs]
                    pi = pi[: eff_max_seqs * pages_per_seq]
                    cq = cq[: eff_max_seqs + 1]
                    d = jnp.minimum(d, eff_max_seqs)
                kl, pi = _prepare_swa_metadata(
                    kl,
                    pi,
                    cq,
                    num_tokens=q.shape[0],
                    sw_cache_num_pages=c.shape[0],
                    sliding_window=128,
                    logical_page_size=16,
                )
                return mla_sliding_window_ragged_paged_attention(
                    q=q,
                    new_kv=kv,
                    cache_kv=c,
                    kv_lens=kl,
                    page_indices=pi,
                    cu_q_lens=cq,
                    distribution=d,
                    attention_sinks=s,
                    sm_scale=512 ** -0.5,
                    sliding_window=128,
                    logical_page_size=16,
                    num_kv_pages_per_block=tuple(cp["num_kv_pages_per_block"]),
                    num_queries_per_block=tuple(cp["num_queries_per_block"]),
                    q_compute_block_size=cp["q_compute_block_size"],
                    decode_seq_batch_size=cp.get("decode_seq_batch_size", 32),
                    unnormalized_output=False,
                    ced_decoder=False,
                )

            opt_fn = jax.jit(_custom_fn, donate_argnums=(2,))
        else:
            opt_fn = jax.jit(
                lambda q, kv, c, kl, pi, cq, d, s: _run_swa(
                    q=q,
                    new_kv=kv,
                    sw_cache=c,
                    swa_kv_lens=kl,
                    swa_page_indices=pi,
                    swa_cu_q_lens=cq,
                    swa_distribution=d,
                    attention_sinks=s,
                    sm_scale=512 ** -0.5,
                    sliding_window=128,
                    logical_page_size=16,
                    swa_only=True,
                    ced_decoder=False,
                ),
                donate_argnums=(2,),
            )

        opt_out, opt_cache, _, _ = opt_fn(
            inp["q"],
            inp["new_kv"],
            inp["sw_cache"].copy(),
            inp["swa_kv_lens"],
            inp["swa_page_indices"],
            inp["swa_cu_q_lens"],
            inp["swa_distribution"],
            inp["attention_sinks"],
        )
        jax.block_until_ready(opt_out)
        max_abs_diff = float(jnp.max(jnp.abs(opt_out.astype(jnp.float32) - ref_out.astype(jnp.float32))))
        cache_match = bool(jnp.array_equal(opt_cache, ref_cache))

        for _ in range(3):
            o_o, _, _, _ = opt_fn(
                inp["q"],
                inp["new_kv"],
                inp["sw_cache"].copy(),
                inp["swa_kv_lens"],
                inp["swa_page_indices"],
                inp["swa_cu_q_lens"],
                inp["swa_distribution"],
                inp["attention_sinks"],
            )
            jax.block_until_ready(o_o)

        c_bufs2 = [inp["sw_cache"].copy() for _ in range(args.iters)]
        jax.block_until_ready(c_bufs2[-1])
        t1 = time.perf_counter()
        for it in range(args.iters):
            o_o, _, _, _ = opt_fn(
                inp["q"],
                inp["new_kv"],
                c_bufs2[it],
                inp["swa_kv_lens"],
                inp["swa_page_indices"],
                inp["swa_cu_q_lens"],
                inp["swa_distribution"],
                inp["attention_sinks"],
            )
        jax.block_until_ready(o_o)
        opt_us = (time.perf_counter() - t1) * 1e6 / args.iters

        res = {
            **case,
            "base_us": round(base_us, 2),
            "opt_us": round(opt_us, 2),
            "speedup": round(base_us / max(opt_us, 1e-6), 3),
            "max_abs_diff": max_abs_diff,
            "cache_match": cache_match,
        }
        results.append(res)

    print("BENCH_JSON_RESULT:" + json.dumps(results), flush=True)


def _run_on_chip(chip_id: int, cases: list[dict[str, Any]], iters: int) -> list[dict[str, Any]]:
    env = os.environ.copy()
    env["TPU_VISIBLE_CHIPS"] = str(chip_id)
    env["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
    env["TPU_HOST_BOUNDS"] = "1,1,1"
    for k in (
        "TPU_WORKER_ID",
        "TPU_PROCESS_ADDRESSES",
        "TPU_PROCESS_PORT",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    ):
        env.pop(k, None)
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--iters",
        str(iters),
        "--cases-json",
        json.dumps(cases),
    ]
    out = subprocess.check_output(cmd, env=env, text=True)
    for line in out.splitlines():
        if line.startswith("BENCH_JSON_RESULT:"):
            return json.loads(line[len("BENCH_JSON_RESULT:") :])
    raise RuntimeError(f"Worker on chip {chip_id} failed:\n{out}")


def build_standard_suite(num_q_heads_list: list[int]) -> list[dict[str, Any]]:
    suite: list[dict[str, Any]] = []
    # 1. Chunked Prefill up to 100k
    for nqh in num_q_heads_list:
        for kv_len in [4096, 8192, 16384, 32768, 65536, 100000]:
            for q_len in [512, 1024, 2048, 4096]:
                if q_len > kv_len:
                    continue
                suite.append(
                    {
                        "mode": "prefill",
                        "num_seqs": 1,
                        "q_len": q_len,
                        "kv_len": kv_len,
                        "num_q_heads": nqh,
                    }
                )
    # 2. Decode B in {16, 64, 128} at L_kv in {8192, 32768, 100000} and B=512 at 8k
    for nqh in num_q_heads_list:
        for kv_len in [8192, 32768, 100000]:
            for b in [16, 64, 128]:
                suite.append(
                    {
                        "mode": "decode",
                        "num_seqs": b,
                        "q_len": 1,
                        "kv_len": kv_len,
                        "num_q_heads": nqh,
                    }
                )
        suite.append(
            {
                "mode": "decode",
                "num_seqs": 512,
                "q_len": 1,
                "kv_len": 8192,
                "num_q_heads": nqh,
            }
        )
    return suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cases-json", type=str, default="[]")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--num-chips", type=int, default=4)
    parser.add_argument("--heads", type=str, default="16,64,8")
    parser.add_argument("--sweep", choices=["none", "prefill", "decode"], default="none")
    parser.add_argument("--out-json", type=str, default="")
    args = parser.parse_args()

    if args.worker:
        _worker_main(args)
        return

    heads_list = [int(h) for h in args.heads.split(",") if h.strip()]
    if args.sweep == "prefill":
        cases = []
        for nqh in heads_list:
            for q_len in [512, 2048, 4096]:
                for bq_sz in [64, 128, 256]:
                    for bkv_p in [4, 8, 12, 16]:
                        for q_cbs in [16, 32, 64, None]:
                            if q_cbs is not None and q_cbs > bq_sz:
                                continue
                            eff_cbs = bq_sz if q_cbs is None else q_cbs
                            single_bkv = ((bq_sz + 128 - 1 + (bkv_p * 16) - 1) // (bkv_p * 16)) == 1
                            # TPU v7x physical VMEM = 63.94 MiB; single_bkv eliminates 16 MiB acc_scratch
                            bkv_bytes = 2 * (bkv_p * 16) * 1024
                            bq_bytes = 2 * bq_sz * max(nqh, 8) * 512 * 2
                            acc_bytes = 0 if single_bkv else bq_sz * max(nqh, 8) * 512 * 4
                            vreg_spill = eff_cbs * max(nqh, 8) * 512 * 4
                            if bkv_bytes + 2 * bq_bytes + acc_bytes + vreg_spill > 48 * 1024 * 1024:
                                continue
                            cases.append(
                                {
                                    "mode": "sweep_prefill",
                                    "num_seqs": 1,
                                    "q_len": q_len,
                                    "kv_len": 32768,
                                    "num_q_heads": nqh,
                                    "custom_params": {
                                        "num_kv_pages_per_block": [2, bkv_p, bkv_p],
                                        "num_queries_per_block": [1, bq_sz, bq_sz],
                                        "q_compute_block_size": q_cbs,
                                        "decode_seq_batch_size": 32,
                                    },
                                }
                            )
    elif args.sweep == "decode":
        cases = []
        for nqh in heads_list:
            bq_m = 64 if nqh > 32 else 128
            for b in [16, 64, 128, 512]:
                for bkv_p_dec in [8]:
                    for seq_bsz in [8, 16, 32, 64]:
                        cases.append(
                            {
                                "mode": "sweep_decode",
                                "num_seqs": b,
                                "q_len": 1,
                                "kv_len": 8192 if b == 512 else 32768,
                                "num_q_heads": nqh,
                                "custom_params": {
                                    "num_kv_pages_per_block": [bkv_p_dec, 16, 16],
                                    "num_queries_per_block": [1, bq_m, bq_m],
                                    "q_compute_block_size": 32,
                                    "decode_seq_batch_size": seq_bsz,
                                },
                            }
                        )
    else:
        cases = build_standard_suite(heads_list)

    shards = [cases[i :: args.num_chips] for i in range(args.num_chips)]
    all_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_chips) as ex:
        futs = [
            ex.submit(_run_on_chip, chip_id, shard, args.iters)
            for chip_id, shard in enumerate(shards)
            if shard
        ]
        for fut in concurrent.futures.as_completed(futs):
            all_results.extend(fut.result())

    all_results.sort(
        key=lambda r: (
            r["mode"],
            r["num_q_heads"],
            r["kv_len"],
            r["q_len"],
            r["num_seqs"],
            str(r.get("custom_params", "")),
        )
    )
    print(f"\n=== SWA BENCHMARK RESULTS ({len(all_results)} cases across {args.num_chips} TPU chips) ===")
    for r in all_results:
        extra = ""
        if "custom_params" in r:
            cp = r["custom_params"]
            extra = (
                f" | bkv_p={cp['num_kv_pages_per_block'][2] if 'prefill' in r['mode'] else cp['num_kv_pages_per_block'][0]}"
                f" bq={cp['num_queries_per_block'][2]}"
                f" cbs={cp['q_compute_block_size']}"
                f" dec_bsz={cp['decode_seq_batch_size']}"
            )
        print(
            f"[{r['mode']:13s}] H={r['num_q_heads']:2d} B={r['num_seqs']:3d} q={r['q_len']:4d} L_kv={r['kv_len']:6d}{extra}"
            f" -> base={r['base_us']:7.1f} us | opt={r['opt_us']:7.1f} us | speedup={r['speedup']:5.2f}x"
            f" | diff={r['max_abs_diff']:.5f} | cache_eq={r['cache_match']}"
        )
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
