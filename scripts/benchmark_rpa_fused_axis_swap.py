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
"""Microbenchmark for RPA v3 with the query/output axis swap fused into the
kernel DMAs (`fuse_non_tiling_axis_swap=True`) against the XLA transposes it
replaces (`False`).

Times the whole `ragged_paged_attention` call, so the unfused number includes
the two HBM transposes around the kernels. Each row also checks that both
paths return bit-identical outputs and KV caches. The timed jit does not donate
its inputs, so both paths also pay one KV cache copy that serving does not.

Usage:
  python scripts/benchmark_rpa_fused_axis_swap.py
  python scripts/benchmark_rpa_fused_axis_swap.py --models=llama3-8b \
      --kv-dtypes=bfloat16 --scenarios=prefill --json=/tmp/rpa.json
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.ragged_paged_attention.v3 import kernel

# Per-chip attention shapes: (num_q_heads, num_kv_heads, head_dim).
_MODELS = {
    "llama3-8b": (32, 8, 128),
    "llama3-70b-tp8": (8, 1, 128),
    "qwen2.5-32b-tp2": (20, 4, 128),
    "mha-16": (16, 16, 128),
}
_KV_DTYPES = {"bfloat16": jnp.bfloat16, "float8_e4m3fn": jnp.float8_e4m3fn}
_PAGE_SIZE = 16


def _scenarios() -> dict[
    str, list[tuple[str, list[int], list[int], tuple, int | None]]
]:
    """(name, q_lens, kv_lens, distribution, chunk_prefill_size) per group."""
    prefill = [
        (f"prefill 1x{t}", [t], [t], (0, 0, 1), None)
        for t in (512, 1024, 2048, 4096, 8192)
    ] + [
        ("prefill 4x1024", [1024] * 4, [1024] * 4, (0, 0, 4), None),
        ("prefill 16x256", [256] * 16, [256] * 16, (0, 0, 16), None),
    ]
    chunked = [
        (f"chunked 4x{c} (history 2048)", [c] * 4, [2048 + c] * 4, (0, 4, 4), c)
        for c in (256, 512, 1024)
    ]
    decode = [
        (f"decode {b} seqs kv={kv}", [1] * b, [kv] * b, (b, b, b), None)
        for b, kv in ((8, 2048), (32, 2048), (64, 1024), (128, 1024), (256, 512))
    ]
    mixed = [
        (
            "mixed 32 decode + 1x2048",
            [1] * 32 + [2048],
            [1024] * 32 + [2048],
            (32, 32, 33),
            None,
        ),
        (
            "mixed 64 decode + 2x512 chunk + 1x1000",
            [1] * 64 + [512] * 2 + [1000],
            [1024] * 64 + [1536] * 2 + [1000],
            (64, 66, 67),
            512,
        ),
    ]
    return {"prefill": prefill, "chunked": chunked, "decode": decode, "mixed": mixed}


def _make_inputs(heads, kv_dtype, q_lens, kv_lens, distribution, seed=0):
    num_q_heads, num_kv_heads, head_dim = heads
    rng = np.random.default_rng(seed)
    num_seqs = len(q_lens)
    pages_per_seq = -(-max(kv_lens) // _PAGE_SIZE)
    total_num_pages = num_seqs * pages_per_seq + 8
    num_tokens = sum(q_lens)
    cache_shape = kernel.get_kv_cache_shape(
        total_num_pages, _PAGE_SIZE, num_kv_heads, head_dim, kv_dtype
    )
    key = jax.random.key(seed)
    kq, kk, kv, kc = jax.random.split(key, 4)
    return (
        jax.random.normal(kq, (num_tokens, num_q_heads, head_dim), jnp.bfloat16),
        jax.random.normal(kk, (num_tokens, num_kv_heads, head_dim)).astype(kv_dtype),
        jax.random.normal(kv, (num_tokens, num_kv_heads, head_dim)).astype(kv_dtype),
        jax.random.normal(kc, cache_shape).astype(kv_dtype),
        jnp.asarray(kv_lens, jnp.int32),
        jnp.asarray(
            rng.permutation(total_num_pages)[: num_seqs * pages_per_seq], jnp.int32
        ),
        jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]), jnp.int32),
        jnp.asarray(distribution, jnp.int32),
    )


def _time_us(fn, args, iters, reps):
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(iters):
            out = fn(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / iters * 1e6)
    return float(np.median(samples))


def _bit_identical(a, b):
    return all(
        np.array_equal(
            np.asarray(x.astype(jnp.float32)), np.asarray(y.astype(jnp.float32))
        )
        for x, y in zip(a, b, strict=True)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=",".join(_MODELS))
    # fp8 KV caches are native on v7x only; earlier chips emulate them.
    parser.add_argument("--kv-dtypes", default="bfloat16")
    parser.add_argument("--scenarios", default="prefill,chunked,decode,mixed")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--json", default=None, help="Also write rows to this path.")
    args = parser.parse_args()

    groups = _scenarios()
    rows = []
    print(f"device: {jax.devices()[0].device_kind}")
    print(
        "| model | kv dtype | scenario | unfused (us) | fused (us) | saved (us) "
        "| speedup | bit-identical |"
    )
    print("|---|---|---|---:|---:|---:|---:|:---:|")
    for model in args.models.split(","):
        heads = _MODELS[model]
        for dtype_name in args.kv_dtypes.split(","):
            kv_dtype = _KV_DTYPES[dtype_name]
            for group in args.scenarios.split(","):
                for name, q_lens, kv_lens, dist, chunk in groups[group]:
                    inputs = _make_inputs(heads, kv_dtype, q_lens, kv_lens, dist)
                    times, outs = {}, {}
                    for fuse in (False, True):
                        # No donation, so the same inputs are reused every call.
                        fn = jax.jit(
                            lambda *a, fuse=fuse, chunk=chunk, hd=heads[2]: (
                                kernel.ragged_paged_attention(
                                    *a,
                                    sm_scale=hd**-0.5,
                                    chunk_prefill_size=chunk,
                                    fuse_non_tiling_axis_swap=fuse,
                                )
                            )
                        )
                        times[fuse] = _time_us(fn, inputs, args.iters, args.reps)
                        outs[fuse] = fn(*inputs)
                    row = dict(
                        model=model,
                        kv_dtype=dtype_name,
                        group=group,
                        scenario=name,
                        unfused_us=round(times[False], 1),
                        fused_us=round(times[True], 1),
                        saved_us=round(times[False] - times[True], 1),
                        speedup=round(times[False] / times[True], 3),
                        bit_identical=_bit_identical(outs[False], outs[True]),
                    )
                    rows.append(row)
                    same = "yes" if row["bit_identical"] else "NO"
                    print(
                        f"| {model} | {dtype_name} | {name} | {row['unfused_us']} "
                        f"| {row['fused_us']} | {row['saved_us']} | "
                        f"{row['speedup']}x | {same} |",
                        flush=True,
                    )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(device=jax.devices()[0].device_kind, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
