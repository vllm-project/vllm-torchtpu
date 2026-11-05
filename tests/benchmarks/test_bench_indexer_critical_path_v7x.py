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
"""Single-host TPU v7x (`8` chips = `16` TensorCores = `DCP=16`) benchmark for
GLM-5.2 Indexer Critical-Path Optimizations:
1. `TPU_DCP_INDEXER_A2A_SMALL_COLL`: Pallas remote-DMA `all_to_all` + `all_gather`
   over flat 1-D `(1024,)`-tiled HBM score/slot buffers.
2. `TPU_INDEXER_OVERLAP_QB_ABSORB`: Prequantized `q_b` + `W_UK` `absorb_q`
   scheduled under local SparseCore top-k and Stage-2 SparseCore merge waits.

Run directly on a single TPU v7x host (`16` devices) via:
    python tests/benchmarks/test_bench_indexer_critical_path_v7x.py
or via pytest:
    pytest -m multichip tests/benchmarks/test_bench_indexer_critical_path_v7x.py -s
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    KVLayout,
    get_kv_cache_shape,
    streamindex_topk_dcp,
)
from vllm_torchtpu.kernels.mla.absorb_q import absorb_q
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor as jax_quantize_tensor,
)
from vllm_torchtpu.kernels.quantized_matmul.util import (
    xla_prequantized_matmul,
)

LANE_AXIS = "dcp_lane"


def run_v7x_benchmark(
    *,
    num_tokens: int = 64,
    dcp_size: int = 16,
    seq_len: int = 32768,
    topk: int = 2048,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, dict[str, float]]:
    devices = jax.devices()
    if len(devices) < dcp_size or jax.default_backend() != "tpu":
        raise RuntimeError(
            f"Requires TPU backend with >= {dcp_size} devices, got "
            f"backend={jax.default_backend()!r} devices={len(devices)}"
        )

    mesh = Mesh(
        np.asarray(devices[:dcp_size]).reshape(1, dcp_size), (LANE_AXIS, DCP_AXIS_NAME)
    )
    rep = NamedSharding(mesh, P())
    sharded = NamedSharding(mesh, P(DCP_AXIS_NAME))

    # GLM-5.2 indexer & MLA dimensions (`DCP=16` query-replicated).
    h_idx, d_idx = 32, 128
    q_lora_rank = 1536
    h_qrep, nope_dim, rope_dim, kv_lora_rank = 64, 192, 64, 512
    q_out_dim = h_qrep * (nope_dim + rope_dim)
    page_size = 64
    virtual_page_size = page_size * dcp_size
    pages_per_seq = (seq_len + virtual_page_size - 1) // virtual_page_size
    local_blocks = max(num_tokens * pages_per_seq + 8, 64)
    cache_shape_1 = get_kv_cache_shape(
        local_blocks, page_size, d_idx, KVLayout.HEAD_ALONG_SUBLANE
    )
    global_cache_shape = (dcp_size * cache_shape_1[0], *cache_shape_1[1:])

    k0, k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 6)
    q_idx = jax.device_put(
        jax.random.normal(k0, (num_tokens, h_idx, d_idx), jnp.float32).astype(
            jnp.float8_e4m3fn
        ),
        rep,
    )
    weights = jax.device_put(
        jax.random.uniform(
            k1, (num_tokens, h_idx), jnp.bfloat16, minval=0.1, maxval=1.0
        ),
        rep,
    )
    cache_kv = jax.device_put(
        jax.random.randint(k2, global_cache_shape, 0, 127, jnp.uint8),
        sharded,
    )
    seq_lens = jax.device_put(jnp.full((num_tokens,), seq_len, jnp.int32), rep)
    page_indices = jax.device_put(
        jnp.arange(num_tokens * pages_per_seq, dtype=jnp.int32) % local_blocks, rep
    )
    cu_q_lens = jax.device_put(jnp.arange(num_tokens + 1, dtype=jnp.int32), rep)
    dist = jax.device_put(
        jnp.array([num_tokens, num_tokens, num_tokens], jnp.int32), rep
    )
    rank_ids = jax.device_put(jnp.arange(dcp_size, dtype=jnp.int32), sharded)
    lane_ids = jax.device_put(jnp.zeros((dcp_size,), dtype=jnp.int32), sharded)

    q_c = jax.random.normal(k3, (num_tokens, q_lora_rank), jnp.bfloat16)
    qb_xq, qb_xs = jax_quantize_tensor(q_c, jnp.float8_e4m3fn, dim=-1)
    qb_xq = jax.device_put(qb_xq, rep)
    qb_xs = jax.device_put(qb_xs, rep)
    qb_w = jax.device_put(
        jax.random.normal(k4, (q_lora_rank, q_out_dim), jnp.float32).astype(
            jnp.float8_e4m3fn
        ),
        rep,
    )
    qb_s = jax.device_put(
        jax.random.uniform(k5, (q_out_dim,), jnp.float32, minval=0.01, maxval=0.1), rep
    )
    w_uk_t = jax.device_put(
        jax.random.normal(
            jax.random.key(7), (h_qrep, nope_dim, kv_lora_rank), jnp.bfloat16
        ),
        rep,
    )

    def _qb_pq(xq, xs, wq, ws):
        return xla_prequantized_matmul(xq, xs, wq, ws, jnp.bfloat16)

    def _absorb_fn(q_out, wuk):
        qn = q_out.reshape(q_out.shape[0], h_qrep, nope_dim + rope_dim)[..., :nope_dim]
        return q_out, absorb_q(qn, wuk.astype(qn.dtype))

    kv_pages_per_block = (
        (2, 2, 32 if num_tokens <= 8 else 16) if num_tokens <= 64 else (2, 2, 2)
    )
    queries_per_block = (1, 128, 4) if num_tokens <= 64 else (1, 128, 128)

    def _make_step(*, small_coll: bool, overlap: bool):
        if not overlap:

            @jax.jit
            def step(q_i, w_i, c_kv, sl, pi, cq, ds, rk, ln, xq, xs, wq, ws, wuk):
                q_out = _qb_pq(xq, xs, wq, ws)
                topk_idx = streamindex_topk_dcp(
                    q_i,
                    w_i,
                    c_kv,
                    sl,
                    pi,
                    cq,
                    ds,
                    mesh=mesh,
                    k=topk,
                    compression_ratio=1,
                    dcp_size=dcp_size,
                    interleave_size=1,
                    num_kv_pages_per_block=kv_pages_per_block,
                    num_queries_per_block=queries_per_block,
                    a2a_merge=True,
                    a2a_small_coll=small_coll,
                    dcp_rank_ids=rk,
                    dcp_lane_ids=ln,
                    decode_pack=True,
                    decode_ring=6,
                )
                # Mirror Python custom-op boundary: absorb_q starts after indexer.
                _, q_after = jax.lax.optimization_barrier((topk_idx[:1, :1], q_out))
                _, ql_nope = _absorb_fn(q_after, wuk)
                return topk_idx, q_out, ql_nope

            return step

        @jax.jit
        def step_ov(q_i, w_i, c_kv, sl, pi, cq, ds, rk, ln, xq, xs, wq, ws, wuk):
            topk_idx, (q_out, ql_nope) = streamindex_topk_dcp(
                q_i,
                w_i,
                c_kv,
                sl,
                pi,
                cq,
                ds,
                mesh=mesh,
                k=topk,
                compression_ratio=1,
                dcp_size=dcp_size,
                interleave_size=1,
                num_kv_pages_per_block=kv_pages_per_block,
                num_queries_per_block=queries_per_block,
                a2a_merge=True,
                a2a_small_coll=small_coll,
                dcp_rank_ids=rk,
                dcp_lane_ids=ln,
                decode_pack=True,
                decode_ring=6,
                overlap_fn=_qb_pq,
                overlap_args=(xq, xs, wq, ws),
                merge_overlap_fn=_absorb_fn,
                merge_overlap_args=(wuk,),
                overlap_pin_max_tokens=num_tokens,
                overlap_out_specs=(P(), P()),
            )
            return topk_idx, q_out, ql_nope

        return step_ov

    all_args = (
        q_idx,
        weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        dist,
        rank_ids,
        lane_ids,
        qb_xq,
        qb_xs,
        qb_w,
        qb_s,
        w_uk_t,
    )
    configs = [
        ("baseline", False, False),
        ("a2a_small_coll", True, False),
        ("overlap_qb_absorb", False, True),
        ("combined", True, True),
    ]

    ref_outputs = None
    summary: dict[str, dict[str, float]] = {}
    for name, sc, ov in configs:
        fn = _make_step(small_coll=sc, overlap=ov)
        for _ in range(warmup):
            out = fn(*all_args)
        jax.tree.map(lambda x: x.block_until_ready(), out)

        got = tuple(np.asarray(x) for x in out)
        if ref_outputs is None:
            ref_outputs = got
        else:
            assert np.array_equal(got[0], ref_outputs[0]), f"{name}: topk mismatch"
            assert np.array_equal(
                got[1].view(np.uint16), ref_outputs[1].view(np.uint16)
            ), f"{name}: q_out mismatch"
            assert np.array_equal(
                got[2].view(np.uint16), ref_outputs[2].view(np.uint16)
            ), f"{name}: ql_nope mismatch"

        times_us = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn(*all_args)
            jax.tree.map(lambda x: x.block_until_ready(), out)
            times_us.append((time.perf_counter() - t0) * 1e6)

        med = float(np.median(times_us))
        p90 = float(np.percentile(times_us, 90))
        summary[name] = {
            "median_us": med,
            "p90_us": p90,
            "step_61_layers_ms": med * 61.0 / 1000.0,
        }

    base_us = summary["baseline"]["median_us"]
    print(
        f"\n### TPU v7x Indexer Critical-Path Benchmark "
        f"(T={num_tokens}, DCP={dcp_size}, seq_len={seq_len}, topk={topk})\n"
    )
    print(
        "| Mode | Median (us/layer) | p90 (us/layer) "
        "| 61-layer (ms/step) | Delta vs Baseline (ms/step) |"
    )
    print("| :--- | ---: | ---: | ---: | ---: |")
    for name, stats in summary.items():
        delta_ms = (stats["median_us"] - base_us) * 61.0 / 1000.0
        print(
            f"| `{name}` | {stats['median_us']:.2f} | {stats['p90_us']:.2f} "
            f"| {stats['step_61_layers_ms']:.3f} | {delta_ms:+.3f} |"
        )
    return summary


def test_bench_indexer_critical_path_v7x():
    import json

    if jax.default_backend() != "tpu" or len(jax.devices()) < 8:
        pytest.skip("Requires TPU v7x host (>= 8 TPU cores)")
    ndev = len(jax.devices())
    dcp = 16 if ndev >= 16 else 8
    token_cases = (16, 64) if dcp == 16 else (8, 32, 64)
    all_results = {}
    for t in token_cases:
        summary = run_v7x_benchmark(
            num_tokens=t, dcp_size=dcp, seq_len=32768, warmup=10, iters=50
        )
        assert set(summary) == {
            "baseline",
            "a2a_small_coll",
            "overlap_qb_absorb",
            "combined",
        }
        all_results[f"dcp{dcp}_t{t}"] = summary
    print("RESULT " + json.dumps(all_results, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--dcp", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    cli_args = parser.parse_args()
    run_v7x_benchmark(
        num_tokens=cli_args.tokens,
        dcp_size=cli_args.dcp,
        seq_len=cli_args.seq_len,
        warmup=cli_args.warmup,
        iters=cli_args.iters,
    )
