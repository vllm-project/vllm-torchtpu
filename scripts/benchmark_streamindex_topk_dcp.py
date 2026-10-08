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
"""Microbenchmark: `streamindex_topk_dcp` all-gather merge vs `a2a_merge=True`.

Times the whole DCP indexer top-k call (local scoring + merge + owner pack) at
the GLM-5.2-FP8 AgentX `2x2x2/agentx/c16` call-site configuration, both merge
arms interleaved in one process, and asserts the two arms are bit-identical at
every shape it times.

Constants, derived from production rather than inherited from a test:
  - indexer heads 32, head dim 128, top-k 2048: `config.json` `index_n_heads`,
    `index_head_dim`, `index_topk` (knowledge pin U-annyan-260909-07).
  - compression_ratio=1, num_kv_pages_per_block=(2, 2, 2),
    num_queries_per_block=(1, 128, 128), decode_req_batch_size default:
    `layers/adapter/custom_ops/mla_attention_op.py` `_build_streamidx_dcp_op`.
  - SEQ_ALONG_LANE cache `[pages, 33, 4, 256]`: `--block-size=256` with the
    default `VLLM_TPU_INDEXER_KV_CACHE_MODE=sal` and no Raiden GLM admission.
  - interleave 4 (`--cp-kv-cache-interleave-size=4`), 8 sequence slots
    (`--max-num-seqs=8`), virtual pages per seq = 1M / (256 * dcp)
    (`--max-model-len=1048576`), chunk_tokens=256
    (`TPU_STREAMIDX_CHUNK_TOKENS=256` in the AgentX image).
  - distribution = (num_decodes, num_decodes, num_seqs): the call site stacks
    `rd[0], rd[0], rd[2]`.

Per-call time is reported two ways, both per (T, local KV length, arm):
  - `slope`: (t(n=N) - t(n=1)) / (N - 1) of a jitted `fori_loop` that chains
    `n` calls (traced `n`, so one compile serves both): device time per call
    with dispatch amortised away.
  - `wall`: median wall time of one call (`n=1`) incl. dispatch.
Each is the median over `--reps` interleaved repetitions (arm order alternates
per rep); `spread` is (max - min) / median of the per-rep values.

Multi-host: when `NPROC > 1` (simple-run-command-2node), calls
`jax.distributed.initialize(COORD, NPROC, TPU_WORKER_ID)` and builds every
input with `make_array_from_callback`; only process 0 prints.

Usage:
  python scripts/benchmark_streamindex_topk_dcp.py --dcp 8
  python scripts/benchmark_streamindex_topk_dcp.py --dcp 16 --tokens 16,64
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

H_IDX, D_IDX, TOPK = 32, 128, 2048
PAGE_SIZE, INTERLEAVE, MAX_SEQS = 256, 4, 8
MAX_MODEL_LEN = 1 << 20
CHUNK_TOKENS = 256
KV_PACKING = 4
SAL_GROUPS = D_IDX // KV_PACKING + 1  # 33


def _init_distributed():
    nproc = int(os.environ.get("NPROC", "1"))
    if nproc > 1:
        jax.distributed.initialize(
            coordinator_address=os.environ["COORD"],
            num_processes=nproc,
            process_id=int(os.environ.get("TPU_WORKER_ID", os.environ.get("JOBSET_JOB_INDEX", "0"))),
        )


def _log(*a):
    if jax.process_index() == 0:
        print(*a, flush=True)


def _replicated(mesh, arr):
    P = jax.sharding.PartitionSpec
    arr = np.asarray(arr)
    sharding = jax.sharding.NamedSharding(mesh, P())
    return jax.make_array_from_callback(arr.shape, sharding, lambda idx: arr[idx])


def _fp8_bytes(rng, shape):
    """Random e4m3fn bytes with the two NaN encodings (0x7f, 0xff) removed."""
    b = rng.integers(0, 256, size=shape, dtype=np.uint8)
    b[(b & 0x7F) == 0x7F] = 0x3F
    return b


def _cache(mesh, dcp, max_local_len, axis):
    """Per-rank SEQ_ALONG_LANE cache: seq s owns local pages [s*P, (s+1)*P)."""
    P = jax.sharding.PartitionSpec
    pages_per_seq_local = max_local_len // PAGE_SIZE
    local_pages = MAX_SEQS * pages_per_seq_local
    shape = (dcp * local_pages, SAL_GROUPS, KV_PACKING, PAGE_SIZE)
    sharding = jax.sharding.NamedSharding(mesh, P(axis))

    def cb(idx):
        rank = idx[0].start // local_pages
        rng = np.random.default_rng(1000 + rank)
        shard = np.zeros((local_pages, SAL_GROUPS * KV_PACKING, PAGE_SIZE), np.uint8)
        shard[:, :D_IDX, :] = _fp8_bytes(rng, (local_pages, D_IDX, PAGE_SIZE))
        shard[:, D_IDX, :] = 127  # UE8M0 scale of exactly 1.0
        return shard.reshape(local_pages, SAL_GROUPS, KV_PACKING, PAGE_SIZE)

    return jax.make_array_from_callback(shape, sharding, cb), pages_per_seq_local


def _batch(mesh, dcp, T, local_len, pages_per_seq_local, seed):
    """T=16: 8 decodes (q_len 1). T>16: 7 decodes + one prefill chunk of T-7."""
    rng = np.random.default_rng(seed)
    if T == 16:
        q_lens = [1] * MAX_SEQS
        num_decodes = MAX_SEQS
    else:
        q_lens = [1] * (MAX_SEQS - 1) + [T - (MAX_SEQS - 1)]
        num_decodes = MAX_SEQS - 1
    assert sum(q_lens) <= T
    glob = local_len * dcp
    seq_lens = np.array([glob - 7 * s for s in range(MAX_SEQS - 1)] + [glob], np.int32)
    vpages_per_seq = MAX_MODEL_LEN // (PAGE_SIZE * dcp)
    page_indices = np.zeros((MAX_SEQS, vpages_per_seq), np.int32)
    for s in range(MAX_SEQS):
        page_indices[s, :pages_per_seq_local] = s * pages_per_seq_local + np.arange(
            pages_per_seq_local
        )
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    q = (rng.standard_normal((T, H_IDX, D_IDX)).astype(np.float32) * 0.5).astype(
        jnp.float8_e4m3fn
    )
    w = rng.standard_normal((T, H_IDX)).astype(np.float32).astype(jnp.bfloat16)
    dist = np.array([num_decodes, num_decodes, MAX_SEQS], np.int32)
    return tuple(
        _replicated(mesh, a)
        for a in (q, w, None, seq_lens, page_indices.reshape(-1), cu, dist)
        if a is not None
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp", type=int, required=True)
    ap.add_argument("--tokens", default="16,64,256,1024,4096")
    ap.add_argument("--local-kv", default="5120,18432,65536")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20, help="wall samples per rep")
    ap.add_argument("--loop", type=int, default=11, help="N for the slope")
    args = ap.parse_args()

    _init_distributed()
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
        DCP_AXIS_NAME,
        KVLayout,
        streamindex_topk_dcp,
    )
    import vllm_torchtpu

    dcp = args.dcp
    _log(f"vllm_torchtpu.__file__ = {vllm_torchtpu.__file__}")
    _log(
        f"jax {jax.__version__} | processes {jax.process_count()} | "
        f"devices {jax.device_count()} | dcp {dcp}"
    )
    _log(f"LIBTPU_INIT_ARGS={os.environ.get('LIBTPU_INIT_ARGS', '')}")
    assert jax.device_count() >= dcp, (jax.device_count(), dcp)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:dcp]), (DCP_AXIS_NAME,))
    P = jax.sharding.PartitionSpec
    out_sharding = jax.sharding.NamedSharding(mesh, P(DCP_AXIS_NAME))

    local_kvs = [int(x) for x in args.local_kv.split(",")]
    tokens = [int(x) for x in args.tokens.split(",")]
    cache, ppsl = _cache(mesh, dcp, max(local_kvs), DCP_AXIS_NAME)

    kw = dict(
        mesh=mesh,
        k=TOPK,
        compression_ratio=1,
        dcp_size=dcp,
        interleave_size=INTERLEAVE,
        num_kv_pages_per_block=(2, 2, 2),
        num_queries_per_block=(1, 128, 128),
        kv_layout=KVLayout.SEQ_ALONG_LANE,
        chunk_tokens=CHUNK_TOKENS,
    )

    def make_loop(a2a, T):
        @jax.jit
        def loop(n, q, w, c, sl, pi, cu, dist):
            def body(_, carry):
                q_, out = carry
                # The barrier ties q_ to the previous output, so the call is
                # loop-variant and cannot be hoisted or CSE'd.
                q_, out = lax.optimization_barrier((q_, out))
                out = streamindex_topk_dcp(
                    q_, w, c, sl, pi, cu, dist, a2a_merge=a2a, **kw
                )
                return q_, out

            init = lax.with_sharding_constraint(
                jnp.zeros((dcp * T, TOPK), jnp.int32), out_sharding
            )
            return lax.fori_loop(0, n, body, (q, init))[1]

        return loop

    def local_np(x):
        return [np.asarray(s.data) for s in sorted(x.addressable_shards, key=lambda s: s.index[0].start or 0)]

    def run(fn, n, inputs):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(n, *inputs))
        return time.perf_counter() - t0, out

    _log(
        "| dcp | T | local_kv | arm | slope_us | slope_spread | wall_us | wall_spread | bitexact |"
    )
    _log("|---|---|---|---|---|---|---|---|---|")
    n1 = _replicated(mesh, np.int32(1))
    nN = _replicated(mesh, np.int32(args.loop))
    summary = []
    for T in tokens:
        assert T % dcp == 0 and T % 16 == 0, T
        loops = {a: make_loop(a, T) for a in (False, True)}
        for L in local_kvs:
            inputs = _batch(mesh, dcp, T, L, ppsl, seed=T * 7 + L)
            inputs = (inputs[0], inputs[1], cache) + inputs[2:]
            # Correctness: both arms, bit-identical, at this exact shape.
            outs = {}
            for a in (False, True):
                _, o = run(loops[a], n1, inputs)  # also compiles
                run(loops[a], nN, inputs)
                outs[a] = local_np(o)
            exact = all(np.array_equal(x, y) for x, y in zip(outs[False], outs[True]))
            filled = int(sum((x >= 0).sum() for x in outs[False]))
            wall = {False: [], True: []}
            slope = {False: [], True: []}
            for rep in range(args.reps):
                order = (False, True) if rep % 2 == 0 else (True, False)
                for a in order:
                    w1 = [run(loops[a], n1, inputs)[0] for _ in range(args.iters)]
                    wn = [run(loops[a], nN, inputs)[0] for _ in range(max(3, args.iters // 4))]
                    t1, tn = float(np.median(w1)), float(np.median(wn))
                    wall[a].append(t1 * 1e6)
                    slope[a].append((tn - t1) / (args.loop - 1) * 1e6)
            res = {}
            for a in (False, True):
                s, w = np.array(slope[a]), np.array(wall[a])
                res[a] = (np.median(s), np.ptp(s) / np.median(s), np.median(w), np.ptp(w) / np.median(w))
                _log(
                    f"| {dcp} | {T} | {L} | {'a2a' if a else 'gather'} | "
                    f"{res[a][0]:.1f} | {100 * res[a][1]:.1f}% | {res[a][2]:.1f} | "
                    f"{100 * res[a][3]:.1f}% | {exact} (local filled={filled}) |"
                )
            summary.append((T, L, res, exact))
            if not exact:
                _log(f"MISMATCH at dcp={dcp} T={T} L={L}")
    _log("SUMMARY dcp T local_kv gather_slope a2a_slope delta% gather_wall a2a_wall delta% bitexact")
    for T, L, res, exact in summary:
        g, a = res[False], res[True]
        _log(
            f"SUMMARY {dcp} {T} {L} {g[0]:.1f} {a[0]:.1f} {100 * (a[0] - g[0]) / g[0]:+.1f} "
            f"{g[2]:.1f} {a[2]:.1f} {100 * (a[2] - g[2]) / g[2]:+.1f} {exact}"
        )
    if not all(e for *_, e in summary):
        raise SystemExit("a2a output != all-gather output at some shape")


if __name__ == "__main__":
    main()
