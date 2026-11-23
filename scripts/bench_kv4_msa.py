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
"""Microbench: one DCP rank's MSA work per layer for fp8 vs 4-bit caches,
and 4-bit with a shared scale page (TPU_MINIMAX_M3_KV4_SHARED_SCALES).

Two halves per layer, on one TensorCore, as `msa_dcp` runs them:
"score" = `msa_write_and_candidates_dcp` (KV / index-key writes, the
indexer over this rank's index pages, the local top-k) and "attend" =
`msa_attend_dcp` (candidate merge, attention over this rank's selected
pages; the cross-rank merge collectives are not included). L chained layers
per program (each layer's inputs depend on the previous layer's output by
a tiny add, caches threaded through); per-layer cost = ``(t_L - t_1) / (L -
1)``. Requests: decode steps of 8 verify rows per request (T = 16: 2
requests, T = 128: 16, T = 512: 64) or one 8192-token prefill chunk, over
100-150K tokens of context. DCP 2 = one KV head per rank (c16 / c48
layout), DCP 4 = two (c64). Also prints the KV pool from the per-token
bytes of each cache layout.

    PYTHONPATH=.:src python scripts/bench_kv4_msa.py --dcp 2,4 --tokens 16,128,8192
"""

import argparse
import json
import os
import statistics
import time

# Served c48 / c64 MSA settings (the reference commands' env).
for _k, _v in {
    "TPU_TIGHT_VMEM": "1",
    "TPU_MINIMAX_M3_MSA_DECODE_WORK_LIST_KERNEL": "1",
    "TPU_MINIMAX_M3_MSA_DECODE_ONE_CALL": "1",
    "TPU_MINIMAX_M3_INDEXER_LARGE_PAGES_PER_STEP": "64",
    "TPU_MINIMAX_M3_INDEXER_DECODE_OPT": "noclamp+defer",
    "TPU_MINIMAX_M3_INDEX_FOLD_OPT": "gather+compose",
    "TPU_MINIMAX_M3_MSA_PREFILL_BLOCK_Q": "32",
    "TPU_MINIMAX_M3_MSA_PREFILL_PAGES_PER_STEP": "16",
    "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE": "512",
    "TPU_MINIMAX_M3_MSA_PREFILL_PAGE_CACHE_TILE_PAGES": "192",
    "TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT": "packed",
    "TPU_MINIMAX_M3_MSA_PREFILL_WIDTHS": "2/4/6/8/12",
}.items():
    os.environ.setdefault(_k, _v)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4  # noqa: E402

PAGE, D, TOPK, HG = 128, 128, 16, 8  # HG query heads per KV head (TP 8)
ROWS = 8  # verify rows per decode request
LAYERS, LAYERS_PER_FWD = 60, 60
DRAFT_BYTES = 5 * 512  # 5 DSpark layers, fp8 K/V of 1 head padded to 4 heads


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="16,128,512,8192")
    p.add_argument("--dcp", default="2,4")
    p.add_argument("--kv", default="fp8,int4,int4sh")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--ctx", default="100000:150000")
    p.add_argument("--hbm-gib", type=float, default=55.6, help="KV HBM per device")
    p.add_argument("--json", default="")
    p.add_argument("--smoke", action="store_true", help="CPU interpret, tiny shapes")
    return p.parse_args()


def bytes_per_token(kv: str, g: int) -> dict:
    """Per-rank bytes of one token (on its owner rank) by cache, 60 layers."""
    if kv == "fp8":
        per = {"kv": 2 * g * D, "index": D, "scale": 0}
    else:
        rows = msa_fp4.scale_rows(g) * 4  # f32 rows of one [R, 128] page per layer
        if kv == "int4sh":
            rows = rows / msa_fp4.layers_per_scale_page(g)
        per = {"kv": g * D, "index": D // 2 + 4, "scale": rows}
    out = {k: v * LAYERS for k, v in per.items()}
    out["draft"] = DRAFT_BYTES
    out["total"] = sum(out.values())
    return out


def pool_tokens(kv: str, dcp: int, hbm_gib: float) -> int:
    g = max(1, dcp // 2)
    per_rank = hbm_gib * (1 << 30) / bytes_per_token(kv, g)["total"]
    return int(per_rank // (PAGE * dcp) * PAGE * dcp * dcp)


def build_step(num_tokens, dcp, g, ctx_range, rng):
    """Tables, metadata and per-layer candidates of one step."""
    lo, hi = ctx_range
    if num_tokens >= 1024:  # one prefill chunk
        q_lens = np.array([num_tokens])
    else:
        q_lens = np.full(num_tokens // ROWS, ROWS)
    reqs = len(q_lens)
    ctx = rng.integers(lo, hi + 1, reqs)
    seq_lens = ctx + q_lens
    logical = -(-seq_lens // (dcp * PAGE))
    max_seqs = max(reqs, 16)
    pps = int(-(-(hi + q_lens.max()) // (dcp * PAGE)))
    num_pages = int(logical.sum()) + 1
    tables = np.zeros((max_seqs, pps), np.int32)
    perm = rng.permutation(np.arange(1, num_pages))
    at = 0
    for j in range(reqs):
        tables[j, : logical[j]] = perm[at : at + logical[j]]
        at += logical[j]
    sl = np.zeros(max_seqs, np.int32)
    sl[:reqs] = seq_lens
    cu = np.full(max_seqs + 1, num_tokens, np.int32)
    cu[: reqs + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    dist = np.array([0, 0, reqs], np.int32)
    # Per token: init block 0, its own block, 14 picks from a pool its tile
    # (8 rows / 128 prefill tokens) shares; split by owner rank.
    req = np.repeat(np.arange(reqs), q_lens)
    off = np.concatenate([np.arange(n) for n in q_lens])
    pos = ctx[req] + off
    own = pos // PAGE
    tile = num_tokens >= 1024 and PAGE or ROWS
    cands = np.zeros((dcp, num_tokens, g * 2 * TOPK), np.float32)
    for h in range(g):
        blocks = np.zeros((num_tokens, TOPK), np.int64)
        blocks[:, 1] = own
        for t0 in range(0, num_tokens, tile):
            hi_b = max(int(own[t0]) - 1, 2)
            pool = rng.choice(np.arange(1, hi_b), min(40, hi_b - 1), replace=False)
            for t in range(t0, min(t0 + tile, num_tokens)):
                blocks[t, 2:] = rng.choice(pool, TOPK - 2, replace=False)
        scores = rng.standard_normal((num_tokens, TOPK)).astype(np.float32)
        scores[:, :2] = [3e38, 2e38]
        for r in range(dcp):
            mine = blocks % dcp == r
            s = np.where(mine, scores, -np.inf)
            order = np.argsort(-s, axis=1, kind="stable")
            s = np.take_along_axis(s, order, 1)
            b = np.where(np.isfinite(s), np.take_along_axis(blocks, order, 1), -1)
            base = h * 2 * TOPK
            cands[r, :, base : base + TOPK] = s
            cands[r, :, base + TOPK : base + 2 * TOPK] = b
    meta = tuple(jnp.asarray(x) for x in (sl, tables.reshape(-1), cu))
    return meta, jnp.asarray(dist), num_pages, jnp.asarray(cands)


def build_caches(kv, num_pages, g, rng):
    key = jax.random.PRNGKey(int(rng.integers(1 << 30)))
    if kv == "fp8":
        kv_c = jax.random.normal(key, (num_pages, PAGE // 4, 4, 2 * g * D)).astype(
            jnp.float8_e4m3fn
        )
        idx = jax.random.normal(key, (num_pages, PAGE // 4, 4, D)).astype(
            jnp.float8_e4m3fn
        )
        return kv_c, idx, None
    kv_c = jax.random.randint(key, (num_pages, PAGE // 8, 4, 2 * g * D), 0, 256).astype(
        jnp.uint8
    )
    nib = jax.random.randint(key, (num_pages, PAGE, D), 0, 16).astype(jnp.uint8)
    idx = jax.jit(msa_fp4.pack_rows)(
        nib, jax.random.uniform(key, (num_pages, PAGE), jnp.float32, 0.05, 0.2)
    )
    sc = jax.random.uniform(
        key, (num_pages, msa_fp4.SCALE_ROWS, PAGE), jnp.float32, 0.02, 0.1
    )
    return kv_c, idx, sc


def programs(kv, dcp, g, num_layers, interpret):
    fp4 = kv != "fp8"
    row0 = 2 * g * (msa_fp4.layers_per_scale_page(g) - 1) if kv == "int4sh" else 0

    def score(k, v, iq, ik, kv_c, idx, sc, seq_lens, tables, cu, dist):
        link = None
        for _ in range(num_layers):
            if link is not None:
                k = k + link
            out = msa_dcp.msa_write_and_candidates_dcp(
                k, v, iq, ik, kv_c, idx, seq_lens, tables, cu, dist,
                topk=TOPK, init_blocks=0, local_blocks=1, rank=0, dcp=dcp,
                scale_cache=msa_fp4.with_row0(sc, row0) if fp4 else None,
                interpret=interpret,
            )  # fmt: skip
            kv_c, idx = out[0], out[1]
            if fp4:
                sc = msa_fp4.pages_of(out[2])
            link = (out[-1][0, 0] * 0.0).astype(k.dtype)
        return kv_c, idx, sc, out[-1]

    def attend(q, kv_c, sc, cands, seq_lens, tables, cu):
        outs, link = [], None
        for _ in range(num_layers):
            c = cands if link is None else cands + link
            out, lse = msa_dcp.msa_attend_dcp(
                q, kv_c, c, seq_lens, tables, cu,
                sm_scale=D**-0.5, topk=TOPK, rank=0, dcp=dcp,
                scale_cache=msa_fp4.with_row0(sc, row0) if fp4 else None,
                q_head_major=True, interpret=interpret,
            )  # fmt: skip
            link = out[0, 0, 0].astype(jnp.float32) * 0.0
            outs.append(lse[:, :8])
        return out, jnp.stack(outs)

    return (
        jax.jit(score, donate_argnums=(4, 5, 6)),
        jax.jit(attend),
    )


def timed(fn, make_args, iters):
    jax.block_until_ready(fn(*make_args()))
    samples = []
    for _ in range(iters):
        args = make_args()
        jax.block_until_ready(args)
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def main():
    args = _parse()
    if args.smoke:
        args.layers, args.iters, args.ctx = 2, 1, "3000:5000"
        args.tokens = "16"
    interpret = args.smoke
    ctx = tuple(int(x) for x in args.ctx.split(":"))
    print(f"devices {jax.devices()[:1]}; layers {args.layers}", flush=True)
    results = []
    for dcp in (int(x) for x in args.dcp.split(",")):
        g = max(1, dcp // 2)
        for kv in args.kv.split(","):
            b = bytes_per_token(kv, g)
            print(
                f"POOL dcp{dcp} {kv}: {b} B/token/rank -> "
                f"{pool_tokens(kv, dcp, args.hbm_gib):,} tokens",
                flush=True,
            )
        for T in (int(x) for x in args.tokens.split(",")):
            rng = np.random.default_rng(1000 * dcp + T)
            meta, dist, num_pages, cands = build_step(T, dcp, g, ctx, rng)

            def bf16(*shape):
                return jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)

            k, v, iq = bf16(T, g, D), bf16(T, g, D), bf16(T, g, D)
            ik = bf16(T, D)
            q = bf16(dcp * HG, T, D)
            outs, int4 = {}, None
            for kv in args.kv.split(","):
                if kv == "int4sh":
                    # int4's caches, its scale rows moved to this layer's rows
                    # of a shared page (other rows: other layers' values).
                    kv_c, idx, sc = int4 or build_caches("int4", num_pages, g, rng)
                    row0 = 2 * g * (msa_fp4.layers_per_scale_page(g) - 1)
                    sc = jnp.roll(sc, 1, axis=0).at[:, row0 : row0 + 2 * g].set(
                        sc[:, : 2 * g]
                    )
                    caches = (kv_c, idx, sc)
                else:
                    caches = build_caches(kv, num_pages, g, rng)
                if kv == "int4":
                    int4 = caches
                row = {"dcp": dcp, "tokens": T, "kv": kv, "pages": num_pages}
                for L in (1, args.layers):
                    score, attend = programs(kv, dcp, g, L, interpret)

                    def score_args(caches=caches):
                        kv_c, idx, sc = (
                            None if c is None else jnp.copy(c) for c in caches
                        )
                        return (k, v, iq, ik, kv_c, idx, sc, *meta, dist)

                    qq = msa_fp4.rotate_queries(q) if kv != "fp8" else q

                    def attend_args(caches=caches, qq=qq):
                        return (qq, caches[0], caches[2], cands, *meta)

                    row[f"score_t{L}"] = timed(score, score_args, args.iters)
                    row[f"attend_t{L}"] = timed(attend, attend_args, args.iters)
                    if L == args.layers:
                        outs[kv] = np.asarray(
                            attend(*attend_args())[0].astype(jnp.float32)
                        )
                n = max(args.layers - 1, 1)
                for part in ("score", "attend"):
                    row[f"{part}_us"] = (
                        (row[f"{part}_t{args.layers}"] - row[f"{part}_t1"]) / n * 1e3
                    )
                row["fwd60_ms"] = (row["score_us"] + row["attend_us"]) * 60 / 1e3
                print(
                    f"RES dcp{dcp} T{T:5d} {kv:6s} score {row['score_us']:8.1f} us  "
                    f"attend {row['attend_us']:8.1f} us  per-60-layer fwd "
                    f"{row['fwd60_ms']:7.2f} ms  (pages {num_pages})",
                    flush=True,
                )
                results.append(row)
                del caches
            int4 = None
            if "int4" in outs and "int4sh" in outs:
                same = np.array_equal(outs["int4"], outs["int4sh"])
                print(f"BITWISE dcp{dcp} T{T} int4 vs int4sh: {same}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
