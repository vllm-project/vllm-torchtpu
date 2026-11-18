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
"""Head-parallel MSA DCP prefill microbench (c64, TPU v7 2x2x1): the replica
row split (TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT) and the exchange round capacity
(TPU_MINIMAX_M3_DCP_HP_ROUND_CAPACITY) against the served path.

One SPMD program over the DCP mesh runs L chained MSA layers as the 4-bit
DCP op does for several KV heads per group (K/V/index-query all-gather,
`msa_write_and_candidates_dcp`, `msa_dcp_heads.attend_and_merge`); the
per-layer device time is ``(t(L) - t(1)) / (L - 1)``. The program's wall
time is the slowest rank's, so balance shows up in it. Variants set the env
switches before tracing; outputs are compared to the first variant per row
class (decode / prefill).

Served MSA keys of c64 (ER8_64S + qk8) are set before the kernels import.
Use --dcp 8 --kv-heads 4 --hp 16:16::1024 (c64 DCP8) or --dcp 4 --kv-heads 2
--hp 8:64::1024 (c64 DCP4: one DCP group of 4 cores, two KV heads, two
replicas each).

CPU smoke (interpret mode, tiny shapes):
  python scripts/bench_hp_balance.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

PAGE = 128
HEAD_DIM = 128

# Served c64 MSA keys (launch_unified.sh E_C64 + QK8); read at import.
C64_ENV = {
    "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE": "512",
    "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES": "192",
    "TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q": "32",
    "TPU_MINIMAX_M3_MSA_PREFILL_PAGES_PER_STEP": "16",
    "TPU_MINIMAX_M3_MSA_PREFILL_WIDTHS": "2/4/6/8/12",
    "TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT": "packed",
    "TPU_MINIMAX_M3_MSA_PREFILL_PRECISION": "qk8",
    "TPU_MINIMAX_M3_MSA_DECODE_WORK_LIST_KERNEL": "1",
    "TPU_MINIMAX_M3_MSA_DECODE_ONE_CALL": "1",
    "TPU_MINIMAX_M3_INDEXER_LARGE_PAGES_PER_STEP": "64",
    "TPU_TIGHT_VMEM": "1",
}

# name -> (decode requests, decode ctx, [(prefill tokens, prefill ctx), ...]);
# a None prefill length takes the rest of the step. Contexts follow the
# c64 xprof window (kv_len avg 137-150K, max 231K).
CASES = {
    # One long prefill chunk: one round at any capacity.
    "chunk150k": (0, 0, [(None, 150000)]),
    # Decode + two long prefills: two rounds at capacity 1.
    "mix2x140k": (56, 140000, [(3872, 140000), (None, 200000)]),
    # Decode + four agentic turns (short new tokens, long prefixes): 3-4 rounds.
    "mix4x120k": (
        56,
        140000,
        [(1936, 120000), (1936, 150000), (1936, 90000), (None, 180000)],
    ),
}
DEFAULT_CASES = "chunk150k,mix2x140k,mix4x120k"
# variant -> (row split, round capacity)
VARIANTS = {
    "base": ("", "1"),
    "split": ("1", "1"),
    "cap4": ("", "4"),
    "split+cap4": ("1", "4"),
}


def _parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cases", default=DEFAULT_CASES)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--dcp", type=int, default=8)
    p.add_argument(
        "--kv-heads", type=int, default=4, help="KV heads per DCP group's TP span"
    )
    p.add_argument("--q-heads", type=int, default=8, help="query heads per rank")
    p.add_argument("--max-seqs", type=int, default=64)
    p.add_argument("--decode-rows", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=262144)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument(
        "--hp", default="16:16::1024", help="TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"
    )
    p.add_argument("--q-noise", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default="")
    p.add_argument(
        "--xprof",
        default="",
        help="jax.profiler trace dir (one L-layer run per variant)",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.tokens, args.layers, args.iters = 96, 2, 1
        args.max_seqs, args.max_model_len, args.decode_rows = 4, 4096, 2
        args.dcp, args.kv_heads, args.q_heads, args.hp = 4, 2, 2, "8:16::0"
        CASES.clear()
        CASES["smoke"] = (1, 1000, [(40, 1500), (None, 900)])
        args.cases = "smoke"
    return args


ARGS = _parse() if __name__ == "__main__" else None
if ARGS is not None:
    for k, v in C64_ENV.items():
        os.environ.setdefault(k, v)
    os.environ["TPU_MINIMAX_M3_DCP_HP_PREFILL_CONFIG"] = ARGS.hp
    if ARGS.smoke:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_force_host_platform_device_count={ARGS.dcp}"
        )
        for k in ("TPU_MINIMAX_M3_MSA_PREFILL_PRECISION", "TPU_TIGHT_VMEM"):
            os.environ.pop(k, None)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.models.minimax_m3.kernels import (  # noqa: E402
    msa_dcp,
    msa_dcp_heads,
    msa_fp4,
)

AXIS = "dcp"


def build_step(args, case, rng):
    num_decode, decode_ctx, prefills = CASES[case]
    q_lens = [args.decode_rows] * num_decode
    ctx = (
        list((decode_ctx * rng.uniform(0.8, 1.2, num_decode)).astype(int))
        if num_decode
        else []
    )
    rest = args.tokens - sum(q_lens)
    for length, c in prefills:
        if length is None:
            length = rest - sum(n for n, _ in prefills if n is not None)
        q_lens.append(length)
        ctx.append(c)
    assert sum(q_lens) == args.tokens and len(q_lens) <= args.max_seqs, (case, q_lens)
    num_seqs = len(q_lens)
    seq_lens = np.zeros(args.max_seqs, np.int32)
    seq_lens[:num_seqs] = np.add(ctx, q_lens)
    assert seq_lens.max() <= args.max_model_len, case
    cu_q = np.full(args.max_seqs + 1, args.tokens, np.int32)
    cu_q[: num_seqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    width = -(-args.max_model_len // (PAGE * args.dcp))
    need = -(-seq_lens // (PAGE * args.dcp))
    num_pages = -(-(int(need.sum()) + 1) // 8) * 8
    perm = rng.permutation(np.arange(1, num_pages))
    tables = np.zeros((args.max_seqs, width), np.int32)
    at = 0
    for s in range(num_seqs):
        tables[s, : need[s]] = perm[at : at + need[s]]
        at += need[s]
    meta = dict(
        seq_lens=seq_lens,
        tables=tables.reshape(-1),
        cu_q=cu_q,
        dist=np.asarray([num_decode, num_decode, num_seqs], np.int32),
    )
    rows = np.zeros(args.tokens, np.int8)
    rows[: num_decode * args.decode_rows] = 1
    rows[num_decode * args.decode_rows :] = 2
    blocks = need[num_decode:num_seqs]
    return meta, num_pages, q_lens, rows, int(blocks.sum()), width


def build_inputs(args, mesh, meta, num_pages, q_lens):
    dcp, layers, T = args.dcp, args.layers, args.tokens
    replicas = max(1, dcp // args.kv_heads)
    G = dcp // replicas
    seq_of_row = np.repeat(np.arange(len(q_lens)), q_lens)

    def make(key):
        key = jax.random.fold_in(key, jax.lax.axis_index(AXIS))
        ks = jax.random.split(key, 10)
        kv = jax.random.bits(
            ks[0], (num_pages, PAGE // 8, 4, G * 2 * HEAD_DIM), jnp.uint8
        )
        nib = jax.random.randint(ks[1], (num_pages, PAGE, HEAD_DIM), 0, 16).astype(
            jnp.uint8
        )
        iscale = jax.random.uniform(ks[2], (num_pages, PAGE), jnp.float32, 0.02, 0.2)
        index = msa_fp4.pack_rows(nib, scales=iscale)
        scale = jax.random.uniform(
            ks[3], (num_pages, msa_fp4.scale_rows(G), PAGE), jnp.float32, 0.02, 0.2
        )

        def bf16(k, shape):
            return jax.random.normal(k, shape, jnp.float32).astype(jnp.bfloat16)

        q = bf16(ks[4], (layers, T, args.q_heads, HEAD_DIM))
        k = bf16(ks[5], (layers, T, 1, HEAD_DIM))
        v = bf16(ks[6], (layers, T, 1, HEAD_DIM))
        base = jax.random.normal(ks[7], (layers, len(q_lens), 1, HEAD_DIM))
        noise = jax.random.normal(ks[9], (layers, T, 1, HEAD_DIM))
        iq = (base[:, seq_of_row] + args.q_noise * noise).astype(jnp.bfloat16)
        ik = bf16(ks[8], (layers, T, HEAD_DIM))
        return tuple(x[None] for x in (kv, index, scale, q, k, v, iq, ik))

    out = jax.jit(
        jax.shard_map(make, mesh=mesh, in_specs=P(), out_specs=(P(AXIS),) * 8)
    )(jax.random.key(args.seed))
    rank = jax.device_put(np.arange(dcp, dtype=np.int32), NamedSharding(mesh, P(AXIS)))
    rep = NamedSharding(mesh, P())
    meta_dev = tuple(
        jax.device_put(jnp.asarray(meta[k]), rep)
        for k in ("seq_lens", "tables", "cu_q", "dist")
    )
    return rank, out[:3], out[3:], meta_dev, G, replicas


def program(args, mesh, num_layers, replicas, interpret):
    dcp = args.dcp

    def body(rank, kv, index, scale, q, k, v, iq, ik, seq_lens, tables, cu, dist):
        kv, index, scale = kv[0], index[0], scale[0]
        q, k, v, iq, ik = q[0], k[0], v[0], iq[0], ik[0]
        T = q.shape[1]
        outs = []
        for layer in range(num_layers):
            per_rank = jax.lax.all_gather(
                jnp.concatenate([k[layer], v[layer], iq[layer]], axis=1),
                AXIS,
                axis=1,
                tiled=True,
            )
            per_head = per_rank.reshape(T, dcp, 3, HEAD_DIM)[:, ::replicas]
            kk, vv, qq = (per_head[:, :, i] for i in range(3))
            kv, index, scale, cand = msa_dcp.msa_write_and_candidates_dcp(
                kk,
                vv,
                qq,
                ik[layer],
                kv,
                index,
                seq_lens,
                tables,
                cu,
                dist,
                topk=args.topk,
                init_blocks=0,
                local_blocks=1,
                rank=rank,
                dcp=dcp,
                scale_cache=scale,
                interpret=interpret,
            )
            outs.append(
                msa_dcp_heads.attend_and_merge(
                    rank,
                    msa_fp4.rotate_queries(q[layer]),
                    kv,
                    scale,
                    cand,
                    seq_lens,
                    tables,
                    cu,
                    dist,
                    axis=AXIS,
                    dcp=dcp,
                    kv_head_replicas=replicas,
                    sm_scale=HEAD_DIM**-0.5,
                    topk=args.topk,
                    decode_rows=args.decode_rows,
                    rs_out_precision=True,
                    head_parallel_min_tokens=0
                    if args.smoke
                    else msa_dcp_heads.HEAD_PARALLEL_MIN_TOKENS,
                    interpret=interpret,
                )
            )
        return kv[None], index[None], scale[None], jnp.stack(outs)[None]

    s = P(AXIS)
    fn = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(s,) * 9 + (P(),) * 4,
        out_specs=(s,) * 4,
        check_vma=False,
    )
    return jax.jit(fn, donate_argnums=(1, 2, 3))


def compare(out, ref, rows):
    o, r = np.asarray(out).astype(np.float32), np.asarray(ref).astype(np.float32)
    res = {}
    for cls, name in ((1, "decode"), (2, "prefill")):
        sel = rows == cls
        if sel.any():
            a, b = o[:, :, sel], r[:, :, sel]
            scale = float(np.abs(b).max()) or 1.0
            res[name] = (bool(np.array_equal(a, b)), float(np.abs(a - b).max()) / scale)
    return res


def main():
    args = ARGS
    interpret = args.smoke
    mesh = Mesh(np.asarray(jax.devices()[: args.dcp]), (AXIS,))
    print(
        f"devices {jax.devices()[: args.dcp]}; DCP {args.dcp}, "
        f"KV heads {args.kv_heads}, "
        f"T {args.tokens}, L {args.layers}, HP {args.hp}",
        flush=True,
    )
    results = []
    for case in args.cases.split(","):
        rng = np.random.default_rng(args.seed + len(case))
        meta, num_pages, q_lens, rows, hp_blocks, width = build_step(args, case, rng)
        rank, caches, layer_in, meta_dev, G, replicas = build_inputs(
            args, mesh, meta, num_pages, q_lens
        )
        print(
            f"{case}: prefill blocks {hp_blocks} (round {width} x capacity), "
            f"q_lens tail {q_lens[-4:]}",
            flush=True,
        )
        base = base_slope = None
        for name in args.variants.split(","):
            split, cap = VARIANTS[name]
            os.environ["TPU_MINIMAX_M3_DCP_HP_ROW_SPLIT"] = split
            os.environ["TPU_MINIMAX_M3_DCP_HP_ROUND_CAPACITY"] = cap
            jax.clear_caches()
            p1 = program(args, mesh, 1, replicas, interpret)
            pl = program(args, mesh, args.layers, replicas, interpret)
            state = list(caches)

            def run(prog):
                nonlocal state
                *state, out = prog(rank, *state, *layer_in, *meta_dev)  # noqa: B023
                return out

            def timed(prog):
                jax.block_until_ready(run(prog))
                samples = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    jax.block_until_ready(run(prog))
                    samples.append(time.perf_counter() - t0)
                return statistics.median(samples) * 1e3

            t1, tl = timed(p1), timed(pl)
            if args.xprof:
                with jax.profiler.trace(os.path.join(args.xprof, f"{case}_{name}")):
                    jax.block_until_ready(run(pl))
            out = np.asarray(run(pl))
            caches = state
            slope = (tl - t1) / max(args.layers - 1, 1) * 1e3
            cmp = {} if base is None else compare(out, base, rows)
            if base is None:
                base, base_slope = out, slope
            print(
                f"  {name:<11} t1 {t1:8.3f} tL {tl:8.3f} us/layer {slope:9.1f} "
                f"vs base {slope - base_slope:+8.1f} ({slope / base_slope:.3f}) {cmp}",
                flush=True,
            )
            results.append(
                dict(
                    case=case,
                    variant=name,
                    t1_ms=t1,
                    tl_ms=tl,
                    us_per_layer=slope,
                    vs_base_us=slope - base_slope,
                    cmp=cmp,
                )
            )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    print("RESULT " + json.dumps(results))


if __name__ == "__main__":
    main()
