#!/usr/bin/env python3
"""Parity and A/B/C microbenchmark for DeepSeek-V4 MLA (HCA) and Sparse MLA (CSA) kernels on TPU7x."""

import functools
import importlib.util
import os
import sys
import time
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import numpy as np


def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom == 0:
        return 1.0
    return float(np.dot(a_f, b_f) / denom)


def create_hca_cache(rng, total_pages: int, page_size: int, head_dim: int):
    cache_kv_base = jnp.array(
        rng.standard_normal(size=(total_pages, page_size // 2, 2, head_dim), dtype=np.float32)
        * 0.1
    ).astype(jnp.bfloat16)
    kernel_cache = cache_kv_base.reshape(total_pages, page_size, 4, 128)
    kernel_cache = jax.lax.bitcast_convert_type(kernel_cache, jnp.uint8)
    kernel_cache = kernel_cache.transpose(0, 1, 2, 4, 3)
    kernel_cache = kernel_cache.reshape(total_pages, page_size * 2, 4, 128)
    return cache_kv_base, kernel_cache


def create_csa_cache(rng, total_pages: int, page_size: int, head_dim: int):
    from tests.kernels.deepseek_v4 import sparse_mla_test

    cache_kv_base, cache_kv_nope, cache_kv_rope = sparse_mla_test.create_cache(
        rng, total_pages, page_size, head_dim, jnp.bfloat16
    )
    return cache_kv_base, cache_kv_nope, cache_kv_rope


def run_parity_suite(mla_base, mla_opt_a, mla_opt_b, sparse_base, sparse_opt):
    print("=" * 100)
    print("PART 1: NUMERICAL PARITY & CORRECTNESS SUITE")
    print("=" * 100)
    from tests.kernels.deepseek_v4 import mla_test, sparse_mla_test

    print("\n[1.1] Running mla_test.CorrectnessTest on optimized mla.py (OptA) ...")
    t_mla = mla_test.CorrectnessTest()
    t_mla.test_correctness()
    print("  -> mla_test.test_correctness PASSED")
    t_mla.test_static_verify_segment()
    print("  -> mla_test.test_static_verify_segment (incl. bit-exact mixed == static) PASSED")

    print("\n[1.2] Running sparse_mla_test.CorrectnessTest on optimized sparse_mla.py ...")
    t_sparse = sparse_mla_test.CorrectnessTest()
    t_sparse.test_correctness()
    print("  -> sparse_mla_test.test_correctness PASSED")

    print("\n[1.3] Detailed HCA (mla.py) Parity: Base vs OptA vs OptB vs Ref")
    rng = np.random.default_rng(20261027)
    page_size = 16
    head_dim = 512
    sm_scale = 0.13523377886
    pages_per_seq = 128

    for num_heads in [16, 64]:
        q_lens = np.array([1, 1, 7, 7, 16, 32], dtype=np.int32)
        batch_size = len(q_lens)
        kv_lens = jnp.array([16, 95, 140, 210, 80, 175], dtype=np.int32)
        cu_q_lens = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(jnp.array(q_lens, dtype=jnp.int32))]
        )
        total_tokens = int(np.sum(q_lens))
        q = jnp.array(rng.standard_normal((total_tokens, num_heads, head_dim), dtype=np.float32)).astype(
            jnp.bfloat16
        )
        kv_lens_to_attend = []
        for i in range(batch_size):
            for t in range(int(q_lens[i])):
                kv_lens_to_attend.append(int(kv_lens[i]) - int(q_lens[i]) + t + 1)
        kv_lens_to_attend = jnp.array(kv_lens_to_attend, dtype=jnp.int32)

        total_pages = batch_size * pages_per_seq
        cache_kv_base, kernel_cache = create_hca_cache(rng, total_pages, page_size, head_dim)
        attention_sinks = jnp.array(rng.random((num_heads,), dtype=np.float32))
        swa_acc = jnp.array(
            rng.standard_normal((total_tokens, num_heads, head_dim), dtype=np.float32) * 0.1
        ).astype(jnp.bfloat16)
        swa_l = jnp.array(rng.random((total_tokens, num_heads), dtype=np.float32) + 0.5)
        swa_m = jnp.array(rng.random((total_tokens, num_heads), dtype=np.float32) - 1.0)

        perm = rng.permutation(total_pages)
        pt = jnp.array(perm[: batch_size * pages_per_seq], dtype=jnp.int32)
        dist = jnp.array([2, 4, batch_size], dtype=jnp.int32)
        out_ref = mla_test.ref_implementation(
            q, cache_kv_base, kv_lens, kv_lens_to_attend, pt, cu_q_lens, dist,
            attention_sinks, swa_acc, swa_l, swa_m, sm_scale=sm_scale,
        )
        out_ref.block_until_ready()
        ref_np = np.asarray(out_ref, dtype=np.float32)

        for fast_dma in [False, True]:
            for static_q_len in [None, 7]:
                d_use = jnp.array([2, 2, batch_size] if static_q_len is None else [2, 4, batch_size], dtype=jnp.int32)
                common_kw = dict(
                    sm_scale=sm_scale,
                    num_kv_pages_per_block=4,
                    num_queries_per_block=(1, 16, 16),
                    chunk_prefill_size=static_q_len,
                    fast_dma=fast_dma,
                )
                args = (q, kernel_cache, kv_lens, kv_lens_to_attend, pt, cu_q_lens, d_use, attention_sinks, swa_acc, swa_l, swa_m)
                out_b = mla_base.mla_ragged_paged_attention(*args, **common_kw)
                out_oa = mla_opt_a.mla_ragged_paged_attention(*args, **common_kw)
                out_ob = mla_opt_b.mla_ragged_paged_attention(*args, **common_kw)
                out_b.block_until_ready()
                out_oa.block_until_ready()
                out_ob.block_until_ready()
                b_np = np.asarray(out_b, dtype=np.float32)
                oa_np = np.asarray(out_oa, dtype=np.float32)
                ob_np = np.asarray(out_ob, dtype=np.float32)

                max_a = float(np.max(np.abs(b_np - oa_np)))
                cos_a = cosine_sim(b_np, oa_np)
                max_b = float(np.max(np.abs(b_np - ob_np)))
                cos_b = cosine_sim(b_np, ob_np)
                print(
                    f"  HCA heads={num_heads:2d} fast_dma={str(fast_dma):5s} static_q={str(static_q_len):4s} | "
                    f"OptA vs Base: max_diff={max_a:.5f} cos={cos_a:.8f} | "
                    f"OptB vs Base: max_diff={max_b:.5f} cos={cos_b:.8f}"
                )
                assert cos_a > 0.9999 and cos_b > 0.9999
                assert max_a <= 0.05 and max_b <= 0.05
                np.testing.assert_allclose(ref_np, oa_np, rtol=0.1, atol=0.1)
                np.testing.assert_allclose(ref_np, ob_np, rtol=0.1, atol=0.1)


def bench_fn(fn, *args, warmup: int = 5, rep: int = 30):
    for _ in range(warmup):
        out = fn(*args)
        out.block_until_ready()
    times_us = []
    for _ in range(rep):
        t0 = time.perf_counter()
        out = fn(*args)
        out.block_until_ready()
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)
    return float(np.median(times_us)), float(np.mean(times_us)), float(np.min(times_us))


def run_hca_suite(title, mla_base, mla_opt_a, mla_opt_b, *, num_heads, page_size, max_seqs, pages_per_seq, bkv_p, bq_blocks, cases):
    print(f"\n--- {title} ---")
    print(
        f"{'Case':<26} {'Ctx':<6} {'KV_c':<6} {'fast_dma':<8} {'Contig':<6} | "
        f"{'Base(us)':>9} {'OptA(us)':>9} {'Spd_A':>7} | {'OptB(us)':>9} {'Spd_B':>7}"
    )
    print("-" * 110)

    head_dim = 512
    sm_scale = 0.13523377886
    # Use a compact physical pool of 8192 pages and wrap page indices into it so HBM alloc is fast
    phys_pages = 8192
    cache_kv_base = jnp.zeros((phys_pages, page_size * 2, 4, 128), dtype=jnp.uint8) + jnp.uint8(1)
    cache_kv_base.block_until_ready()

    for case_name, bsz, q_len, static_q_len, mode_seg, ctx_lbl, kv_c_len, fast_dma_list in cases:
        total_tokens = max(bsz * q_len, 16)
        q = jnp.ones((total_tokens, num_heads, head_dim), dtype=jnp.bfloat16)
        kv_lens = jnp.full((max_seqs,), kv_c_len, dtype=jnp.int32)
        cu_q = [0]
        for i in range(max_seqs):
            cu_q.append(cu_q[-1] + (q_len if i < bsz else 0))
        cu_q_lens = jnp.array(cu_q, dtype=jnp.int32)
        kv_lens_to_attend = jnp.full((total_tokens,), kv_c_len, dtype=jnp.int32)

        if mode_seg == "decode":
            dist = jnp.array([bsz, bsz, bsz], dtype=jnp.int32)
        elif mode_seg == "prefill":
            dist = jnp.array([0, bsz, bsz], dtype=jnp.int32)
        else:
            dist = jnp.array([0, 0, bsz], dtype=jnp.int32)

        attention_sinks = jnp.zeros((num_heads,), dtype=jnp.float32)
        swa_acc = jnp.zeros((total_tokens, num_heads, head_dim), dtype=jnp.bfloat16)
        swa_l = jnp.ones((total_tokens, num_heads), dtype=jnp.float32)
        swa_m = jnp.zeros((total_tokens, num_heads), dtype=jnp.float32)

        pt_contig_np = np.zeros((max_seqs, pages_per_seq), dtype=np.int32)
        n_alloc = cdiv(kv_c_len, page_size)
        for i in range(max_seqs):
            base_p = (i * 128 + 1) % (phys_pages - pages_per_seq - 2)
            pt_contig_np[i, :n_alloc] = np.arange(base_p, base_p + n_alloc, dtype=np.int32)
        pt_contig = jnp.array(pt_contig_np.ravel(), dtype=jnp.int32)

        for fast_dma in fast_dma_list:
            common_kw = dict(
                sm_scale=sm_scale,
                num_kv_pages_per_block=bkv_p,
                num_queries_per_block=bq_blocks,
                chunk_prefill_size=static_q_len,
                fast_dma=fast_dma,
            )
            fn_base = jax.jit(lambda *a: mla_base.mla_ragged_paged_attention(*a, **common_kw))
            fn_opta = jax.jit(lambda *a: mla_opt_a.mla_ragged_paged_attention(*a, **common_kw))
            fn_optb = jax.jit(lambda *a: mla_opt_b.mla_ragged_paged_attention(*a, **common_kw))
            args = (
                q, cache_kv_base, kv_lens, kv_lens_to_attend, pt_contig, cu_q_lens,
                dist, attention_sinks, swa_acc, swa_l, swa_m,
            )
            med_b, _, _ = bench_fn(fn_base, *args)
            med_a, _, _ = bench_fn(fn_opta, *args)
            med_ob, _, _ = bench_fn(fn_optb, *args)
            spd_a = med_b / med_a
            spd_b = med_b / med_ob
            print(
                f"{case_name:<26} {ctx_lbl:<6} {kv_c_len:<6} {str(fast_dma):<8} {'True':<6} | "
                f"{med_b:9.1f} {med_a:9.1f} {spd_a:6.2f}x | {med_ob:9.1f} {spd_b:6.2f}x"
            )


def run_csa_suite(sparse_base, sparse_opt):
    print("\n--- [2.3] CSA (sparse_mla.py) Production DSv4 AgentX Geometry (num_heads=64, topk=2048) ---")
    print("  (a) Full sparse_ragged_paged_attention (SparseCore Gather + TensorCore Attention)")
    print(
        f"{'Case':<26} {'Masked':<8} {'Tokens':<8} | "
        f"{'Base(us)':>9} {'Opt(us)':>9} {'Speedup':>8}"
    )
    print("-" * 75)

    num_heads = 64
    head_dim = 512
    sm_scale = 0.13523377886
    max_seqs = 16
    topk = 2048
    total_pages_csa = 4096
    cache_nope = jnp.ones((total_pages_csa, 16, 128), dtype=jnp.int32)
    cache_rope = jnp.ones((total_pages_csa, 4, 128), dtype=jnp.int32)
    page_indices_csa = jnp.arange(max_seqs * 256, dtype=jnp.int32)

    csa_cases = [
        ("verify_bsz1_q7", 1, 7, False),
        ("verify_bsz4_q7", 4, 7, False),
        ("verify_bsz4_q7_masked", 4, 7, True),
        ("verify_bsz10_q7", 10, 7, False),
        ("prefill_bsz1_q128", 1, 128, False),
        ("prefill_bsz1_q512", 1, 512, False),
    ]
    for case_name, bsz, q_len, is_masked in csa_cases:
        total_tokens = max(bsz * q_len, 8)
        q = jnp.ones((total_tokens, num_heads, head_dim), dtype=jnp.bfloat16)
        cu_q = [0]
        for i in range(max_seqs):
            cu_q.append(cu_q[-1] + (q_len if i < bsz else 0))
        cu_q_lens = jnp.array(cu_q, dtype=jnp.int32)
        dist = jnp.array([0, 0, bsz], dtype=jnp.int32)

        idx_np = np.broadcast_to(np.arange(topk, dtype=np.int32)[None, :], (total_tokens, topk)).copy()
        if is_masked:
            idx_np[:, topk // 2 :] = -1
        topk_indices = jnp.array(idx_np, dtype=jnp.int32)

        attention_sinks = jnp.zeros((num_heads,), dtype=jnp.float32)
        swa_acc = jnp.zeros((total_tokens, num_heads, head_dim), dtype=jnp.bfloat16)
        swa_l = jnp.ones((total_tokens, num_heads), dtype=jnp.float32)
        swa_m = jnp.zeros((total_tokens, num_heads), dtype=jnp.float32)

        chunk_sz = 32 if total_tokens <= 128 else 64
        fn_csa_base = jax.jit(
            lambda *a: sparse_base.sparse_ragged_paged_attention(
                *a, sm_scale=sm_scale, attention_kernel_batch_size=8, gather_and_attention_chunk_size=chunk_sz
            )
        )
        fn_csa_opt = jax.jit(
            lambda *a: sparse_opt.sparse_ragged_paged_attention(
                *a, sm_scale=sm_scale, attention_kernel_batch_size=8, gather_and_attention_chunk_size=chunk_sz
            )
        )
        args = (
            q, cache_nope, cache_rope, topk_indices, page_indices_csa,
            cu_q_lens, dist, attention_sinks, swa_acc, swa_l, swa_m,
        )
        med_b, _, _ = bench_fn(fn_csa_base, *args)
        med_o, _, _ = bench_fn(fn_csa_opt, *args)
        speedup = med_b / med_o
        print(
            f"{case_name:<26} {str(is_masked):<8} {bsz * q_len:<8} | "
            f"{med_b:9.1f} {med_o:9.1f} {speedup:7.2f}x"
        )


def main():
    print(f"JAX version: {jax.__version__}, devices: {jax.devices()}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mla_base = load_module_from_path("mla_baseline", os.path.join(base_dir, "mla_baseline.py"))
    mla_opt_b = load_module_from_path("mla_opt_b", os.path.join(base_dir, "mla_opt_b.py"))
    sparse_base = load_module_from_path(
        "sparse_mla_baseline", os.path.join(base_dir, "sparse_mla_baseline.py")
    )
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import mla as mla_opt_a
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import sparse_mla as sparse_opt

    run_parity_suite(mla_base, mla_opt_a, mla_opt_b, sparse_base, sparse_opt)

    print("\n" + "=" * 100)
    print("PART 2: HARDWARE KERNEL MICROBENCHMARKS ON TPU7x")
    print("=" * 100)

    # 2.1: DSv4 AgentX Production Geometry (num_heads=64, page_size=8, max_seqs=64, pages_per_seq=1024, bkv_p=(32, 64, 64), bq=(1, 32, 32))
    prod_cases = [
        ("verify_bsz1_q7_static", 1, 7, 7, "prefill", "80k", 625, [False, True]),
        ("verify_bsz4_q7_static", 4, 7, 7, "prefill", "80k", 625, [False, True]),
        ("verify_bsz4_q7_static", 4, 7, 7, "prefill", "256k", 2000, [False, True]),
        ("verify_bsz4_q7_static", 4, 7, 7, "prefill", "480k", 3750, [False, True]),
        ("verify_bsz10_q7_static", 10, 7, 7, "prefill", "80k", 625, [False]),
        ("verify_bsz10_q7_static", 10, 7, 7, "prefill", "160k", 1250, [False]),
        ("decode_bsz4_q1", 4, 1, None, "decode", "80k", 625, [False]),
        ("decode_bsz4_q1", 4, 1, None, "decode", "256k", 2000, [False]),
        ("decode_bsz4_q1", 4, 1, None, "decode", "480k", 3750, [False]),
        ("batched_decode_bsz8_q1", 8, 1, None, "decode", "80k", 625, [False]),
        ("batched_decode_bsz8_q1", 8, 1, None, "decode", "256k", 2000, [False]),
        ("prefill_bsz1_q128", 1, 128, None, "mixed", "100k", 782, [False, True]),
        ("prefill_bsz1_q128", 1, 128, None, "mixed", "256k", 2000, [False]),
        ("prefill_bsz1_q128", 1, 128, None, "mixed", "480k", 3750, [False]),
        ("prefill_bsz1_q512", 1, 512, None, "mixed", "100k", 782, [False]),
        ("prefill_bsz1_q512", 1, 512, None, "mixed", "256k", 2000, [False]),
    ]
    run_hca_suite(
        "[2.1] HCA (mla.py) Production DSv4 AgentX Geometry (num_heads=64, page_size=8, bkv_p=(32,64,64), bq=(1,32,32), table=64x1024)",
        mla_base, mla_opt_a, mla_opt_b,
        num_heads=64, page_size=8, max_seqs=64, pages_per_seq=1024,
        bkv_p=(32, 64, 64), bq_blocks=(1, 32, 32), cases=prod_cases,
    )

    # 2.2: TP=8 Geometry (num_heads=16, page_size=16, max_seqs=8, pages_per_seq=1024, bkv_p=(32, 64, 64), bq=(1, 16, 16))
    tp8_cases = [
        ("decode_bsz4_q1", 4, 1, None, "decode", "80k", 1250, [False]),
        ("decode_bsz4_q1", 4, 1, None, "decode", "256k", 4000, [False]),
        ("decode_bsz4_q1", 4, 1, None, "decode", "480k", 7500, [False]),
        ("verify_bsz1_q7_static", 1, 7, 7, "prefill", "80k", 1250, [False]),
        ("verify_bsz4_q7_static", 4, 7, 7, "prefill", "80k", 1250, [False, True]),
        ("verify_bsz4_q7_static", 4, 7, 7, "prefill", "256k", 4000, [False, True]),
        ("verify_bsz4_q7_static", 4, 7, 7, "prefill", "480k", 7500, [False, True]),
        ("verify_bsz4_q7_mixed", 4, 7, None, "mixed", "80k", 1250, [False]),
        ("verify_bsz4_q7_mixed", 4, 7, None, "mixed", "480k", 7500, [False]),
        ("prefill_bsz1_q512", 1, 512, None, "mixed", "80k", 1250, [False, True]),
        ("prefill_bsz1_q2048", 1, 2048, None, "mixed", "80k", 1250, [False, True]),
    ]
    run_hca_suite(
        "[2.2] HCA (mla.py) TP=8 Geometry (num_heads=16, page_size=16, bkv_p=(32,64,64), bq=(1,16,16), table=8x1024)",
        mla_base, mla_opt_a, mla_opt_b,
        num_heads=16, page_size=16, max_seqs=8, pages_per_seq=1024,
        bkv_p=(32, 64, 64), bq_blocks=(1, 16, 16), cases=tp8_cases,
    )

    run_csa_suite(sparse_base, sparse_opt)
    print("\nALL PARITY CHECKS AND MICROBENCHMARKS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
