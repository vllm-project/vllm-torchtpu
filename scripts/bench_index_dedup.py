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
"""Microbench + on-device exactness: the MSA index cache once per TP group
(TPU_MINIMAX_M3_DCP_INDEX_DEDUP) vs once per DCP group, 4-bit caches.

Per layer, one TensorCore, DCP rank 0: "score" = `msa_write_and_candidates_dcp`
(K/V + index writes, the indexer, the local top-k): the current path scores
the group's G index heads over 128-token index pages; the dedup path scores
every lane's ``lanes * G`` heads over ``128 / lanes``-token pages, plus the
lane merge (`msa_dcp.merge_lane_candidates`). Chained layers as
`bench_kv4_msa.py`. Exactness: for one step per size, every lane's dedup
candidates merged equal the current path's candidates of each lane, bit for
bit, on the device. Collectives (index-query all-gather and candidate
all-to-all over the lane axis) are timed separately over all devices with a
(lane, dcp) mesh at the step's shapes. Also prints the KV pool.

    PYTHONPATH=.:src python scripts/bench_index_dedup.py --dcp 2,4 --tokens 16,128,512,8192
"""

import argparse
import os
import statistics
import time

for _k, _v in {
    "TPU_TIGHT_VMEM": "1",
    "TPU_MINIMAX_M3_INDEXER_LARGE_PAGES_PER_STEP": "64",
    "TPU_MINIMAX_M3_INDEXER_DECODE_OPT": "noclamp+defer",
    "TPU_MINIMAX_M3_INDEX_FOLD_OPT": "gather+compose",
}.items():
    os.environ.setdefault(_k, _v)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp, msa_fp4  # noqa: E402

PAGE, D, TOPK, ROWS, TP = 128, 128, 16, 8, 8
LAYERS = 60
DRAFT_BYTES = 5 * 512


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="16,128,512,8192")
    p.add_argument("--dcp", default="2,4")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--ctx", default="100000:150000")
    p.add_argument("--hbm-gib", type=float, default=55.6)
    p.add_argument("--smoke", action="store_true", help="CPU interpret, tiny shapes")
    return p.parse_args()


def bytes_per_token(dcp: int, lanes: int) -> dict:
    """Per-rank bytes of one token, 60 layers, 4-bit + shared scale pages."""
    g = max(1, dcp // 2)
    scale = msa_fp4.scale_rows(g) * 4 / msa_fp4.layers_per_scale_page(g)
    index = (PAGE // lanes // 8 + 1) * 4 * D / PAGE  # page bytes / block tokens
    out = {
        k: v * LAYERS for k, v in {"kv": g * D, "index": index, "scale": scale}.items()
    }
    out["draft"] = DRAFT_BYTES
    out["total"] = sum(out.values())
    return out


def pool_tokens(dcp: int, lanes: int, hbm_gib: float) -> int:
    per_rank = hbm_gib * (1 << 30) / bytes_per_token(dcp, lanes)["total"]
    return int(per_rank // (PAGE * dcp) * PAGE * dcp * dcp)


def build_step(num_tokens, dcp, ctx_range, rng):
    lo, hi = ctx_range
    q_lens = (
        np.array([num_tokens])
        if num_tokens >= 1024
        else np.full(num_tokens // ROWS, ROWS)
    )
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
    meta = tuple(jnp.asarray(x) for x in (sl, tables.reshape(-1), cu))
    return meta, jnp.asarray(np.array([0, 0, reqs], np.int32)), num_pages


@jax.jit
def _unpack(idx):
    return msa_fp4.unpack_rows(idx), msa_fp4.folded_scales(idx)


def split_index(idx, lanes, lane):
    nib, scales = _unpack(idx)
    sub = PAGE // lanes
    part = slice(lane * sub, (lane + 1) * sub)
    pad = jnp.full((scales.shape[0], PAGE - sub), 0.125, jnp.float32)
    return jax.jit(msa_fp4.pack_rows)(
        nib[:, part], jnp.concatenate([scales[:, part], pad], axis=1)
    )


def score_program(dcp, lanes, lane, num_layers, interpret):
    def score(k, v, iq, ik, kv_c, idx, sc, seq_lens, tables, cu, dist):
        link = None
        for _ in range(num_layers):
            if link is not None:
                k = k + link
            out = msa_dcp.msa_write_and_candidates_dcp(
                k, v, iq, ik, kv_c, idx, seq_lens, tables, cu, dist,
                topk=TOPK, init_blocks=0, local_blocks=1, rank=0, dcp=dcp,
                scale_cache=sc, interpret=interpret,
                **({"lanes": lanes, "lane": lane} if lanes > 1 else {}),
            )  # fmt: skip
            kv_c, idx, sc = out[0], out[1], msa_fp4.pages_of(out[2])
            link = (out[-1][0, 0] * 0.0).astype(k.dtype)
        return kv_c, idx, sc, out[-1]

    return jax.jit(score, donate_argnums=(4, 5, 6))


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


def collectives(num_tokens, dcp, lanes, g, iters):
    """us of the index-query all-gather and the candidate all-to-all over the
    lane axis of a (lane, dcp) mesh of all devices (None if too few)."""
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    devices = jax.devices()
    if len(devices) < lanes * dcp:
        return None
    mesh = Mesh(np.asarray(devices[: lanes * dcp]).reshape(lanes, dcp), ("lane", "dcp"))
    iq = jnp.zeros((lanes * dcp * num_tokens, g, D), jnp.bfloat16)
    cand = jnp.zeros((lanes * dcp * lanes, num_tokens, g * 2 * TOPK), jnp.float32)

    def gather(x):
        return jax.lax.all_gather(x, "lane", axis=1, tiled=True)

    def a2a(x):
        return jax.lax.all_to_all(x, "lane", 0, 0, tiled=True)

    spec = P(("lane", "dcp"))
    out = {}
    for name, fn, x in (("iq_gather", gather, iq), ("cand_a2a", a2a, cand)):
        f = jax.jit(shard_map(fn, mesh=mesh, in_specs=spec, out_specs=spec))
        out[name] = timed(f, lambda x=x: (x,), iters) * 1e3
    return out


def main():
    args = _parse()
    if args.smoke:
        args.layers, args.iters, args.ctx, args.tokens = 2, 1, "3000:5000", "16"
    interpret = args.smoke
    ctx = tuple(int(x) for x in args.ctx.split(":"))
    print(
        f"devices {len(jax.devices())} x {jax.devices()[0]}; layers {args.layers}",
        flush=True,
    )
    for dcp in (int(x) for x in args.dcp.split(",")):
        g, lanes = max(1, dcp // 2), TP // dcp
        for n in (1, lanes):
            print(
                f"POOL dcp{dcp} lanes{n}: {bytes_per_token(dcp, n)} B/token/rank -> "
                f"{pool_tokens(dcp, n, args.hbm_gib):,} tokens",
                flush=True,
            )
        for T in (int(x) for x in args.tokens.split(",")):
            rng = np.random.default_rng(1000 * dcp + T)
            meta, dist, num_pages = build_step(T, dcp, ctx, rng)
            key = jax.random.PRNGKey(T + dcp)
            kv_c = jax.random.randint(
                key, (num_pages, PAGE // 8, 4, 2 * g * D), 0, 256
            ).astype(jnp.uint8)
            nib = jax.random.randint(key, (num_pages, PAGE, D), 0, 16).astype(jnp.uint8)
            idx = jax.jit(msa_fp4.pack_rows)(
                nib, jax.random.uniform(key, (num_pages, PAGE), jnp.float32, 0.05, 0.2)
            )
            sc = jax.random.uniform(
                key, (num_pages, msa_fp4.SCALE_ROWS, PAGE), jnp.float32, 0.02, 0.1
            )
            lane_idx = [split_index(idx, lanes, lane) for lane in range(lanes)]

            def bf16(*shape):
                return jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)

            k, v, ik = bf16(T, g, D), bf16(T, g, D), bf16(T, D)
            iq = bf16(T, lanes * g, D)
            row = {}
            for mode in ("base", "dedup"):
                n = lanes if mode == "dedup" else 1
                index = lane_idx[0] if mode == "dedup" else idx
                q_heads = iq if mode == "dedup" else iq[:, :g]
                for L in (1, args.layers):
                    prog = score_program(dcp, n, 0, L, interpret)

                    def make(index=index, q_heads=q_heads):
                        return (k, v, q_heads, ik, jnp.copy(kv_c), jnp.copy(index),
                                jnp.copy(sc), *meta, dist)  # fmt: skip

                    row[mode, L] = timed(prog, make, args.iters)
            per = {m: (row[m, args.layers] - row[m, 1]) / max(args.layers - 1, 1) * 1e3
                   for m in ("base", "dedup")}  # fmt: skip
            # Exactness on the device, and the merge's cost.
            base1 = score_program(dcp, 1, 0, 1, interpret)
            parts = []
            for lane in range(lanes):
                prog = score_program(dcp, lanes, lane, 1, interpret)
                parts.append(prog(k, v, iq, ik, jnp.copy(kv_c), jnp.copy(lane_idx[lane]),
                                  jnp.copy(sc), *meta, dist)[3])  # fmt: skip
            width = g * 2 * TOPK
            same = True
            merge = jax.jit(lambda c: msa_dcp.merge_lane_candidates(c, TOPK))
            for m in range(lanes):
                want = base1(k, v, iq[:, m * g : (m + 1) * g], ik, jnp.copy(kv_c),
                             jnp.copy(idx), jnp.copy(sc), *meta, dist)[3]  # fmt: skip
                stacked = jnp.stack([p[:, m * width : (m + 1) * width] for p in parts])
                got = merge(stacked)
                same &= bool(np.array_equal(np.asarray(got), np.asarray(want)))
            merge_us = timed(merge, lambda: (stacked,), args.iters) * 1e3
            coll = collectives(T, dcp, lanes, g, args.iters) if not interpret else None
            print(
                f"RES dcp{dcp} T{T:5d} score base {per['base']:8.1f} us  dedup "
                f"{per['dedup']:8.1f} us  merge {merge_us:6.1f} us  coll {coll}  "
                f"BITWISE {same}  (pages {num_pages})",
                flush=True,
            )


if __name__ == "__main__":
    main()
