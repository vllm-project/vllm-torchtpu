# SPDX-License-Identifier: Apache-2.0
"""Crossover benchmark: masked-dense vs sparse (gather) MLA attention.

Both kernels compute GLM-5.2's DSA attention over the same paged
`(nope, rope)` uint8 caches, by opposite strategies:

* `mla/sparse` gathers each token's `topk` selected KV rows on SparseCore and
  attends over that dense buffer. Cost is flat in `kv_len`.
* `mla/masked_dense` streams the whole cache in blocks and masks the scores
  with the indexer's selection. Cost is O(kv_len), but it issues no gather --
  and the gather hot-spots when many tokens select overlapping rows, which is
  exactly what happens while a sequence is short.

The production dispatcher considers masked-dense only for prefill-only
batches. This script measures the crossover at GLM-5.2's serving geometry and
also times the complete enabled-dispatch wrapper against direct sparse for
prefill, decode, and mixed steps. Decode-sized token buckets bypass dispatch
statically; larger mixed buckets exercise the runtime sparse fallback.

    prefill step:  1 sequence  x 1024 query tokens
    decode step:  64 sequences x    1 query token

Three arms are timed per point:

* `analytic` -- masked-dense with the causal mask computed in registers.
  Available only while `kv_len <= topk`, where the indexer necessarily
  selected every causal position, so no bitmap is built at all.
* `bitmap`   -- masked-dense including `generate_mask_sc`.
* `gather`   -- the sparse kernel including `dsa_gather`.

`--prefill-replay-requests N` measures N complete 8K prefill requests, each as
eight 1,024-token prefill-only steps. No decode request is ever inserted into
that workload. Compilation and warmup are excluded.

Run:  python scripts/benchmark_masked_dense_vs_sparse_mla.py
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import statistics
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vllm_torchtpu.kernels.mla import dispatch as mla_dispatch  # noqa: E402
from vllm_torchtpu.kernels.mla import \
    kv_cache_utils as mla_kv_cache  # noqa: E402
from vllm_torchtpu.kernels.mla.masked_dense import csa_mask  # noqa: E402
from vllm_torchtpu.kernels.mla.masked_dense import \
    kernel as masked_dense  # noqa: E402
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse  # noqa: E402

# GLM-5.2-FP8, read off the checkpoint config.
LKV_DIM = 512  # kv_lora_rank
ROPE_DIM = 64  # qk_rope_head_dim
TOPK = 2048  # index_topk
# TENSOR_PARALLELISM=1, so every rank carries all 64 attention heads.
NUM_HEADS = 64
# PallasMLAttentionBackend.get_page_size returns 1024 for this model, and
# MAX_MODEL_LEN=9216 gives 9 pages per sequence.
PAGE_SIZE = 1024
PAGES_PER_SEQ = 9
MAX_MODEL_LEN = PAGE_SIZE * PAGES_PER_SEQ

MAX_NUM_BATCHED_TOKENS = 1024
MAX_NUM_SEQS = 64

KV_LENS = (512, 1024, 2048, 3072, 4096, 6144, 8192, 9216)


def _time(fn, *args, warmup: int = 3, iters: int = 10) -> float:
    """Median wall time of `fn` in milliseconds, device-synchronized."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _time_pair(first,
               second,
               *args,
               warmup: int = 3,
               iters: int = 10) -> tuple[float, float]:
    """Paired medians with alternating arm order to reduce TPU drift."""
    for _ in range(warmup):
        jax.block_until_ready(first(*args))
        jax.block_until_ready(second(*args))
    samples = ([], [])
    for iteration in range(iters):
        arms = ((first, samples[0]), (second, samples[1]))
        if iteration % 2:
            arms = arms[::-1]
        for fn, values in arms:
            start = time.perf_counter()
            jax.block_until_ready(fn(*args))
            values.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples[0]), statistics.median(samples[1])


def _build_caches(rng, total_pages):
    """Random uint8 (nope, rope) caches in the native split sparse layout."""
    nope = jnp.asarray(
        rng.integers(0,
                     200,
                     size=(total_pages, PAGE_SIZE, 4, 128),
                     dtype=np.uint8))
    rope = jnp.asarray(
        rng.integers(0,
                     200,
                     size=(total_pages, PAGE_SIZE // 4, 4, 128),
                     dtype=np.uint8))
    return nope, rope


def _topk_indices(rng, positions, kv_len):
    """Causal top-k selection: token at absolute `pos` picks from `[0, pos]`."""
    rows = np.full((len(positions), TOPK), -1, np.int32)
    for i, pos in enumerate(positions):
        n = min(pos + 1, TOPK)
        if pos + 1 <= TOPK:
            rows[i, :n] = np.arange(n)
        else:
            rows[i, :n] = np.sort(rng.choice(pos + 1, size=n, replace=False))
    return jnp.asarray(rows)


def _case(shape: str, kv_len: int):
    """Returns the inputs for one (step shape, kv_len) point."""
    rng = np.random.default_rng(0)
    if shape == "prefill":
        num_seqs = 1
        q_len = min(MAX_NUM_BATCHED_TOKENS, kv_len)
        num_tokens = q_len
        cu_q_lens = np.array([0, q_len], np.int32)
        distribution = np.array([0, 0, 1], np.int32)
        # The chunk being prefilled ends at kv_len.
        positions = list(range(kv_len - q_len, kv_len))
    elif shape == "decode":
        num_seqs = MAX_NUM_SEQS
        num_tokens = num_seqs
        cu_q_lens = np.arange(num_seqs + 1, dtype=np.int32)
        distribution = np.array([num_seqs, num_seqs, num_seqs], np.int32)
        positions = [kv_len - 1] * num_seqs
    else:
        # Representative nightly mixed step: 63 one-token decodes and one
        # prefill request consuming the remaining 961 real token rows. PR 1
        # must send this entire batch through the unchanged sparse-only op.
        num_decode = MAX_NUM_SEQS - 1
        prefill_q = MAX_NUM_BATCHED_TOKENS - num_decode
        num_seqs = MAX_NUM_SEQS
        num_tokens = MAX_NUM_BATCHED_TOKENS
        q_lens = np.array([1] * num_decode + [prefill_q], np.int32)
        cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        distribution = np.array([num_decode, num_decode, num_seqs], np.int32)
        decode_kv = MAX_MODEL_LEN
        positions = [decode_kv - 1] * num_decode
        positions.extend(range(kv_len - prefill_q, kv_len))

    total_pages = num_seqs * PAGES_PER_SEQ
    nope, rope = _build_caches(rng, total_pages)
    page_indices = jnp.asarray(rng.permutation(total_pages).astype(np.int32))
    q = jnp.asarray(rng.standard_normal(
        (num_tokens, NUM_HEADS, LKV_DIM + ROPE_DIM)).astype(np.float32),
                    dtype=jnp.bfloat16)
    return dict(
        q=q,
        nope=nope,
        rope=rope,
        kv_lens=jnp.asarray(
            ([MAX_MODEL_LEN] * (MAX_NUM_SEQS - 1) +
             [kv_len]) if shape == "mixed" else [kv_len] * num_seqs,
            jnp.int32),
        topk_indices=_topk_indices(rng, positions, kv_len),
        page_indices=page_indices,
        cu_q_lens=jnp.asarray(cu_q_lens),
        distribution=jnp.asarray(distribution),
        num_tokens=num_tokens,
    )


def _wrapper_args(c):
    ql_nope = c["q"][..., :LKV_DIM]
    q_pe = c["q"][..., LKV_DIM:]
    return (
        ql_nope,
        q_pe,
        ql_nope[:, 0, :].astype(jnp.float8_e4m3fn),
        q_pe[:, 0, :].astype(jnp.float8_e4m3fn),
        c["nope"],
        c["rope"],
        c["topk_indices"],
        c["kv_lens"],
        c["page_indices"],
        c["cu_q_lens"],
        c["distribution"],
    )


def _build_wrapper(enable_dispatch: bool):
    """Stable production-equivalent cache-update + attention executable."""
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("model", ))
    sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)
    limits = (mla_dispatch.MASKED_DENSE_ANALYTIC_LIMIT,
              mla_dispatch.MASKED_DENSE_LIMIT)

    def integrated(ql_nope, q_pe, kv_c_normed, k_pe, nope, rope, topk,
                   seq_lens, pages, starts, distribution):
        kv_packing = sparse.get_dtype_packing(kv_c_normed.dtype)
        nope_spec = mla_kv_cache.SparseMLAKVCacheSpec.create(
            mla_kv_cache.KVCacheType.NOPE,
            mla_kv_cache.KVCacheLayout.TENSORCORE, nope.shape[0], PAGE_SIZE,
            LKV_DIM, kv_packing)
        rope_spec = mla_kv_cache.SparseMLAKVCacheSpec.create(
            mla_kv_cache.KVCacheType.ROPE,
            mla_kv_cache.KVCacheLayout.TENSORCORE, rope.shape[0], PAGE_SIZE,
            ROPE_DIM, kv_packing)
        nope, rope = mla_kv_cache.update_sparse_mla_kv_cache(
            nope,
            rope,
            kv_c_normed,
            k_pe,
            seq_lens,
            pages,
            starts,
            nope_spec=nope_spec,
            rope_spec=rope_spec)
        q = jnp.concatenate((ql_nope, q_pe), axis=-1)

        def gather(_):
            return sparse.sparse_ragged_paged_attention(
                q,
                nope,
                rope,
                topk,
                pages,
                starts,
                distribution,
                sm_scale=sm_scale,
                k_scale=1.0,
            )

        if (not enable_dispatch
                or q.shape[0] < mla_dispatch.MASKED_DENSE_MIN_TOKEN_BUCKET):
            output = gather(None)
        else:

            def dense(max_kv_len, _):
                return masked_dense.masked_dense_ragged_paged_attention(
                    q,
                    nope,
                    rope,
                    seq_lens,
                    topk,
                    pages,
                    starts,
                    distribution,
                    sm_scale=sm_scale,
                    k_scale=1.0,
                    max_kv_len=max_kv_len,
                    num_kv_pages_per_block=(1, 1, 1),
                    num_queries_per_block=(1, 32, 32),
                )

            tier = mla_dispatch.masked_dense_prefill_mla_tier(
                seq_lens, distribution, starts, limits)
            output = jax.lax.switch(
                tier,
                (functools.partial(dense, limits[0]),
                 functools.partial(dense, limits[1]), gather),
                operand=None,
            )
        return output[..., :LKV_DIM], nope, rope

    in_specs = (P(None, None, None), P(None, None,
                                       None), P(None, None), P(None, None),
                P(None), P(None), P(None), P(None), P(None), P(None), P(None))
    out_specs = (P(None, None, None), P(None), P(None))
    return jax.jit(
        shard_map.shard_map(integrated,
                            mesh=mesh,
                            in_specs=in_specs,
                            out_specs=out_specs,
                            check_rep=False))


def _time_prefill_requests(num_requests: int):
    """Paired medians for complete 8K prefill-only requests."""
    stages = [_case("prefill", kv_len) for kv_len in range(1024, 8193, 1024)]
    assert all(int(c["distribution"][0]) == 0 for c in stages)
    args = [_wrapper_args(c) for c in stages]
    feature = _build_wrapper(True)
    sparse_only = _build_wrapper(False)

    def run_request(fn):
        start = time.perf_counter()
        for stage_args in args:
            jax.block_until_ready(fn(*stage_args))
        return (time.perf_counter() - start) * 1e3

    for _ in range(2):
        run_request(feature)
        run_request(sparse_only)
    samples = {"prefill_dispatch_ms": [], "sparse_only_ms": []}
    for request in range(num_requests):
        arms = ((feature, samples["prefill_dispatch_ms"]),
                (sparse_only, samples["sparse_only_ms"]))
        if request % 2:
            arms = arms[::-1]
        for fn, values in arms:
            values.append(run_request(fn))
    feature_median = statistics.median(samples["prefill_dispatch_ms"])
    sparse_median = statistics.median(samples["sparse_only_ms"])
    return {
        "scenario": "8k_input_batch1_prefill_only",
        "num_requests": num_requests,
        "steps_per_request": len(stages),
        "decode_requests": 0,
        **samples,
        "prefill_dispatch_median_ms": feature_median,
        "sparse_only_median_ms": sparse_median,
        "ratio": feature_median / sparse_median,
    }


def _run_point(shape: str, kv_len: int, bkv_p: int, bq_sz: int, iters: int,
               measure_wrapper: bool):
    c = _case(shape, kv_len)
    sm_scale = 1.0 / math.sqrt(LKV_DIM + ROPE_DIM)

    def dense_fn(max_kv_len=None):
        return masked_dense.masked_dense_ragged_paged_attention(
            c["q"],
            c["nope"],
            c["rope"],
            c["kv_lens"],
            c["topk_indices"],
            c["page_indices"],
            c["cu_q_lens"],
            c["distribution"],
            sm_scale=sm_scale,
            k_scale=1.0,
            max_kv_len=max_kv_len,
            num_kv_pages_per_block=bkv_p,
            num_queries_per_block=bq_sz,
        )

    def sparse_fn():
        return sparse.sparse_ragged_paged_attention(
            c["q"],
            c["nope"],
            c["rope"],
            c["topk_indices"],
            c["page_indices"],
            c["cu_q_lens"],
            c["distribution"],
            sm_scale=sm_scale,
            k_scale=1.0,
        )

    analytic_bound = min(mla_dispatch.MASKED_DENSE_ANALYTIC_LIMIT,
                         MAX_MODEL_LEN)
    bitmap_bound = min(mla_dispatch.MASKED_DENSE_LIMIT, MAX_MODEL_LEN)

    def mask_fn():
        return csa_mask.generate_mask_sc(c["topk_indices"], bitmap_bound)

    row = dict(
        shape=shape,
        kv_len=kv_len,
        bkv_p=bkv_p,
        bq_sz=bq_sz,
        num_tokens=c["num_tokens"],
        bitmap_ms=(_time(functools.partial(dense_fn, bitmap_bound),
                         iters=iters) if shape == "prefill"
                   and 0 < kv_len <= bitmap_bound else float("nan")),
        gather_ms=_time(sparse_fn, iters=iters),
        mask_only_ms=(_time(mask_fn, iters=iters) if shape == "prefill"
                      and 0 < kv_len <= bitmap_bound else float("nan")),
    )
    # The analytic mask needs its static bound at or under `topk`, and the
    # bound must cover every sequence in the call.
    if shape == "prefill" and 0 < kv_len <= analytic_bound:
        row["analytic_ms"] = _time(functools.partial(dense_fn, analytic_bound),
                                   iters=iters)
    else:
        row["analytic_ms"] = float("nan")
    row["wrapper_dispatch_ms"] = float("nan")
    row["wrapper_sparse_ms"] = float("nan")
    if measure_wrapper:
        dispatch_wrapper = _build_wrapper(True)
        sparse_wrapper = _build_wrapper(False)
        (row["wrapper_dispatch_ms"],
         row["wrapper_sparse_ms"]) = _time_pair(dispatch_wrapper,
                                                sparse_wrapper,
                                                *_wrapper_args(c),
                                                iters=iters)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes",
                        nargs="+",
                        default=["prefill", "decode"],
                        choices=["prefill", "decode", "mixed"])
    parser.add_argument("--kv-lens", nargs="+", type=int, default=KV_LENS)
    parser.add_argument("--num-kv-pages-per-block", type=int, default=1)
    parser.add_argument("--num-queries-per-block", type=int, default=32)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--measure-wrapper",
        action="store_true",
        help="time cache update plus the production static dispatch path",
    )
    parser.add_argument(
        "--prefill-replay-requests",
        type=int,
        default=0,
        help=("also measure this many complete 8K-input, batch-1, "
              "prefill-only requests; no decode requests are scheduled"),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.prefill_replay_requests < 0:
        parser.error("--prefill-replay-requests must be nonnegative")

    print(f"device: {jax.devices()[0]}")
    print(f"heads={NUM_HEADS} head_dim={LKV_DIM + ROPE_DIM} topk={TOPK} "
          f"page_size={PAGE_SIZE} pages_per_seq={PAGES_PER_SEQ}")
    print(f"bkv_sz={args.num_kv_pages_per_block * PAGE_SIZE} "
          f"bq_sz={args.num_queries_per_block}\n")

    rows = []
    for shape in args.shapes:
        print(f"{shape:>8} | {'kv_len':>7} | {'analytic':>9} | "
              f"{'bitmap':>9} | {'gather':>9} | {'mask':>8} | "
              f"{'best/gather':>11} | win")
        print("-" * 88)
        for kv_len in args.kv_lens:
            if shape == "mixed" and kv_len < 1024:
                continue
            row = _run_point(shape, kv_len, args.num_kv_pages_per_block,
                             args.num_queries_per_block, args.iters,
                             args.measure_wrapper)
            best_ms, win = min((m, n)
                               for m, n in ((row["analytic_ms"], "analytic"),
                                            (row["bitmap_ms"], "bitmap"),
                                            (row["gather_ms"], "gather"))
                               if not math.isnan(m))
            row["ratio"] = best_ms / row["gather_ms"]
            row["win"] = win
            rows.append(row)
            print(f"{shape:>8} | {kv_len:>7} | {row['analytic_ms']:>9.3f} | "
                  f"{row['bitmap_ms']:>9.3f} | {row['gather_ms']:>9.3f} | "
                  f"{row['mask_only_ms']:>8.3f} | {row['ratio']:>11.3f} | "
                  f"{win}")
            if args.measure_wrapper:
                wrapper_ratio = (row["wrapper_dispatch_ms"] /
                                 row["wrapper_sparse_ms"])
                print("         wrapper: "
                      f"dispatch={row['wrapper_dispatch_ms']:.3f} ms "
                      f"sparse={row['wrapper_sparse_ms']:.3f} ms "
                      f"ratio={wrapper_ratio:.3f}")
        print()

    for shape in args.shapes:
        for tier in ("analytic", "bitmap"):
            wins = [
                r["kv_len"] for r in rows
                if r["shape"] == shape and r["win"] == tier
            ]
            print(f"{shape}: {tier} wins up to kv_len="
                  f"{max(wins) if wins else 'never'}")

    replay = None
    if args.prefill_replay_requests:
        replay = _time_prefill_requests(args.prefill_replay_requests)
        print("\n8K/1 prefill-only replay "
              f"({replay['num_requests']} requests, "
              f"{replay['decode_requests']} decode): "
              f"dispatch={replay['prefill_dispatch_median_ms']:.3f} ms "
              f"sparse={replay['sparse_only_median_ms']:.3f} ms "
              f"ratio={replay['ratio']:.3f}")

    if args.out:
        args.out.write_text(
            json.dumps({
                "points": rows,
                "replay": replay
            }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
