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
# ruff: noqa: B023, E501
"""Microbench: the whole MSA decode chain of one layer at the served c48 /
c64 step composition (4-bit KV, DCP 2 / DCP 4), as one SPMD program over
the DCP group's TensorCores.

Per layer, as `ops.msa` runs a decode step:
- c48 (DCP 2, one KV head per pair): `msa_write_and_candidates_dcp` (KV and
  index-key writes, indexer, local top-k) then the attention half,
  `msa_dcp.attend_and_merge` (served) or a fused variant (``--variants``);
- c64 (DCP 4, two KV heads per group): the replica-split gather of the
  group's k / v / index queries, the scoring op with both heads, then
  `msa_dcp_heads.attend_and_merge` (served) or a fused variant.

L chained layers per program (the caches are threaded through and each
layer's queries depend on the previous output by a tiny add); per-layer cost
= ``(t_L - t_1) / (L - 1)``. Each decode request has 8 verify rows over
`--ctx` tokens of context. The index keys are random 4-bit values; a pool
of `--pool` blocks per request carries 10x larger index scales, so each
row's top-k comes from that pool (served c48 tiles hold ~20-45 distinct
pages). Also prints the score-half parts (write, indexer, local top-k) and
the floors: index-key and KV-page bytes at HBM rate.

    PYTHONPATH=src python scripts/bench_decode_msa.py --conc 48 --tokens 32,64,128
"""

import argparse
import functools
import hashlib
import json
import os
import statistics
import time

SERVED_ENV = {
    # Both concs (reference commands + the gap round's max-free softmax).
    "common": {
        "TPU_MINIMAX_M3_INDEXER_LARGE_PAGES_PER_STEP": "64",
        "TPU_MINIMAX_M3_INDEXER_DECODE_OPT": "noclamp+defer",
        "TPU_MINIMAX_M3_MSA_PAGE_CACHE_SORT": "packed",
        "TPU_MINIMAX_M3_MSA_MAX_FREE": "1",
        "TPU_MINIMAX_M3_MSA_MXU_MASK": "1",
    },
    48: {},
    64: {
        "TPU_TIGHT_VMEM": "1",
        "TPU_MINIMAX_M3_MSA_DECODE_ONE_CALL": "1",
        "TPU_MINIMAX_M3_MSA_DECODE_WORK_LIST_KERNEL": "1",
        "TPU_MINIMAX_M3_INDEX_FOLD_OPT": "gather+compose",
    },
}


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--conc", type=int, default=48, choices=(48, 64))
    p.add_argument("--tokens", default="32,64,128")
    p.add_argument("--variants", default="served")
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--ctx", default="100000:200000")
    p.add_argument("--pool", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parts", action="store_true", help="also time score parts")
    p.add_argument(
        "--idx-split",
        action="store_true",
        help="one TC: indexer of the served heads over all of a rank's pages vs "
        "all 4 index heads over 1 / (4 / G) of them",
    )
    p.add_argument("--env", default="", help="extra A=1,B=2 env")
    p.add_argument(
        "--tp8",
        action="store_true",
        help="whole TP group (8 TCs): scoring served vs split over the DCP groups",
    )
    p.add_argument("--attn", default="served", help="attention half for --tp8")
    p.add_argument(
        "--comp",
        action="store_true",
        help="one TC: write / indexer / top-k parts, served vs split shapes",
    )
    p.add_argument(
        "--cases",
        default="served:served,split:served",
        help="--tp8 cases <served|split|served_score|split_score|split_score_nocoll>:<attn>",
    )
    p.add_argument("--json", default="")
    p.add_argument("--smoke", action="store_true", help="CPU interpret, tiny shapes")
    return p.parse_args()


ARGS = _parse()
for _k, _v in {**SERVED_ENV["common"], **SERVED_ENV[ARGS.conc]}.items():
    os.environ.setdefault(_k, _v)
for _item in filter(None, ARGS.env.split(",")):
    _k, _v = _item.split("=", 1)
    os.environ[_k] = _v

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.models.minimax_m3.kernels import (  # noqa: E402
    msa,
    msa_dcp,
    msa_dcp_fused,
    msa_dcp_heads,
    msa_fp4,
)

AXIS = "dcp"
PAGE, D, HL, TOPK, ROWS = 128, 128, 8, 16, 8
HBM = 3.69e12  # bytes/s per TensorCore
VARIANTS = {}  # name -> attention-half fn (rank, q, kv, sc, cands, meta..., had)


def variant(name):
    def deco(fn):
        VARIANTS[name] = fn
        return fn

    return deco


def _group(dcp):
    """DCP 2: the two TensorCores of one chip; DCP 4: two chips."""
    devs = jax.devices()
    if devs[0].platform != "tpu":
        return devs[:dcp]
    by_chip = {}
    for d in devs:
        by_chip.setdefault(tuple(d.coords), []).append(d)
    chips = [sorted(v, key=lambda d: d.id) for v in by_chip.values()]
    out = [d for chip in chips for d in chip]
    return out[:dcp]


def build_step(T, dcp, ctx_range, pool, rng):
    reqs = T // ROWS
    lo, hi = ctx_range
    ctx = rng.integers(lo, hi + 1, reqs)
    seq_lens = ctx + ROWS
    logical = -(-seq_lens // (dcp * PAGE))
    max_seqs = max(reqs, 16)
    pps = int(-(-(hi + ROWS) // (dcp * PAGE)))
    num_pages = int(logical.sum()) + 1
    tables = np.zeros((max_seqs, pps), np.int32)
    perm = rng.permutation(np.arange(1, num_pages))
    at = 0
    for j in range(reqs):
        tables[j, : logical[j]] = perm[at : at + logical[j]]
        at += logical[j]
    sl = np.zeros(max_seqs, np.int32)
    sl[:reqs] = seq_lens
    cu = np.full(max_seqs + 1, T, np.int32)
    cu[: reqs + 1] = np.arange(reqs + 1) * ROWS
    dist = np.array([0, 0, reqs], np.int32)
    # Index scales per rank page: pool blocks 10x.
    hot = np.zeros((dcp, num_pages), bool)
    for j in range(reqs):
        nb = int(-(-ctx[j] // PAGE))
        picks = rng.choice(np.arange(1, nb - 1), min(pool, nb - 2), replace=False)
        for b in picks:
            hot[b % dcp, tables[j, b // dcp]] = True
    keys_per_rank = int(np.sum(-(-ctx // PAGE) // dcp)) * PAGE
    return (sl, tables.reshape(-1), cu, dist), num_pages, hot, keys_per_rank


def build_caches(dcp, g, num_pages, hot, rng):
    kv = rng.integers(0, 256, (dcp * num_pages, PAGE // 8, 4, 2 * g * D), np.uint8)
    nib = jnp.asarray(rng.integers(0, 16, (dcp * num_pages, PAGE, D), np.uint8))
    s = np.where(hot.reshape(-1, 1), 0.5, 0.05) * rng.uniform(
        0.8, 1.2, (dcp * num_pages, PAGE)
    )
    idx = jax.jit(msa_fp4.pack_rows)(nib, jnp.asarray(s, jnp.float32))
    sc = rng.uniform(0.02, 0.1, (dcp * num_pages, msa_fp4.SCALE_ROWS, PAGE))
    return jnp.asarray(kv), idx, jnp.asarray(sc, jnp.float32)


# ---- attention halves -------------------------------------------------------


@variant("served")
def _served(rank, q, kv, sc, cands, sl, tb, cu, dist, had, *, dcp, g, interpret):
    if g == 1:
        return msa_dcp.attend_and_merge(
            rank, q.transpose(1, 0, 2), kv, cands, sl, tb, cu, sc, had,
            axis=AXIS, dcp=dcp, sm_scale=D**-0.5, topk=TOPK, interpret=interpret,
        ).reshape(q.shape)  # fmt: skip
    return msa_dcp_heads.attend_and_merge(
        rank, q, kv, sc, cands, sl, tb, cu, dist,
        axis=AXIS, dcp=dcp, kv_head_replicas=dcp // g, sm_scale=D**-0.5,
        topk=TOPK, decode_rows=ROWS, rs_out_precision=True, interpret=interpret,
    )  # fmt: skip


@variant("dattn")
def _dattn(rank, q, kv, sc, cands, sl, tb, cu, dist, had, *, dcp, g, interpret):
    assert g == 1 and dcp == 2
    return msa_dcp_fused.fused_attend_and_merge(
        rank, q.transpose(1, 0, 2), kv, cands, sl, tb, cu, sc, had,
        axis=AXIS, dcp=dcp, sm_scale=D**-0.5, topk=TOPK, interpret=interpret,
    ).reshape(q.shape)  # fmt: skip


def _gather_kv(k, v, iq, rank, dcp, g):
    """`ops.msa.gather_group_kv` (replica-split all-gather)."""
    T = k.shape[0]
    reps = dcp // g
    rows = jnp.stack([k, v, iq], axis=1)  # [T, 3, D]
    part = T // reps
    mine = jax.lax.dynamic_index_in_dim(
        rows.reshape(reps, part, 3, D), rank[0] % reps, 0, False
    )
    parts = jax.lax.all_gather(mine, AXIS, axis=0)  # [dcp, part, 3, D]
    per_head = (
        parts.reshape(g, reps, part, 3, D).transpose(1, 2, 0, 3, 4).reshape(T, g, 3, D)
    )
    return per_head[:, :, 0], per_head[:, :, 1], per_head[:, :, 2]


def program(mesh, num_layers, attend, *, dcp, g, what, interpret):
    def body(rank, q, k, v, iq, ik, kv, idx, sc, sl, tb, cu, dist, had):
        link, outs = None, []
        for _ in range(num_layers):
            qq = q if link is None else q + link
            if g > 1:
                kk, vv, ii = _gather_kv(k, v, iq, rank, dcp, g)
            else:
                kk, vv, ii = k[:, None], v[:, None], iq[:, None]
            if what == "write":
                kv, idx, sc = msa_dcp.msa_write_dcp(
                    kk + (0 if link is None else link), vv, ik, kv, idx, sl, tb,
                    cu, rank=rank, dcp=dcp, scale_cache=sc, interpret=interpret,
                )  # fmt: skip
                link = (kv[0, 0, 0, 0].astype(jnp.float32) * 0.0).astype(q.dtype)
                outs.append(kv[0, 0, 0, :8])
                continue
            kv, idx, sc, cands = msa_dcp.msa_write_and_candidates_dcp(
                kk, vv, ii + (0 if link is None else link), ik, kv, idx, sl,
                tb, cu, dist, topk=TOPK, init_blocks=1, local_blocks=1, rank=rank,
                dcp=dcp, scale_cache=sc, interpret=interpret,
            )  # fmt: skip
            if what == "score":
                link = (cands[0, 0] * 0.0).astype(q.dtype)
                outs.append(cands[0, :8])
                continue
            out = attend(
                rank, msa_fp4.rotate_queries(qq), kv, sc, cands, sl, tb, cu, dist,
                had, dcp=dcp, g=g, interpret=interpret,
            )  # fmt: skip
            link = out[0, 0, 0] * 0.0
            outs.append(out[:, 0, :8].astype(jnp.float32))
        return jnp.stack(outs)[None], kv, idx, sc

    s, r = P(AXIS), P()
    # q, k, v, iq, ik per rank: [dcp * T, ...] sharded on the token axis.
    in_specs = (s,) + (s,) * 5 + (s, s, s) + (r,) * 4 + (r,)
    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=(s, s, s, s),
            check_vma=False,
        ),
        donate_argnums=(6, 7, 8),
    )


def idx_split(args, T, dcp, g, ctx, interpret):
    """Per-layer indexer time on one TensorCore: served (G heads, all of the
    rank's pages) vs split (4 heads, a 1 / (4 / G) share of each request's
    pages), chained L calls."""
    rng = np.random.default_rng(args.seed + T)
    meta, num_pages, hot, keys = build_step(T, dcp, ctx, args.pool, rng)
    _, idx, sc = build_caches(1, g, num_pages, hot[:1], rng)
    sl, tb, cu, dist = (jnp.asarray(m) for m in meta)
    lens = msa_dcp.local_seq_lens(sl, block_size=PAGE, rank=0, dcp=dcp)
    srow = msa_fp4.index_scale_row(g)
    iq4 = jnp.asarray(rng.standard_normal((T, 4, D)), jnp.bfloat16)
    out = {}
    for name, heads, share in (("served", g, 1), ("split", 4, 4 // g)):
        part = -(-(lens // PAGE) // share) * PAGE

        def run(L, heads=heads, part=part):
            def f(iq, idx, sc):
                link, acc = 0.0, []
                for _ in range(L):
                    q = msa_fp4.rotate_queries(iq[:, :heads] + link)
                    if heads == 1:
                        s = msa._block_scores(q, idx, part, tb, cu, dist[2], interpret,
                                              sc, srow)  # fmt: skip
                    else:
                        s = msa.partial_block_scores(q, idx, part, tb, cu, dist[2],
                                                     shards=1, interpret=interpret,
                                                     scale_cache=sc, scale_row=srow)  # fmt: skip
                    link = (s.reshape(-1)[0] * 0.0).astype(iq.dtype)
                    acc.append(s.reshape(-1)[:8])
                return jnp.stack(acc)

            fn = jax.jit(f)
            jax.block_until_ready(fn(iq4, idx, sc))
            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(iq4, idx, sc))
                samples.append(time.perf_counter() - t0)
            return statistics.median(samples) * 1e3

        t1, tl = run(1), run(args.layers)
        out[name] = (tl - t1) / max(args.layers - 1, 1) * 1e3
    row = dict(conc=args.conc, tokens=T, keys_per_rank=keys,
               idx_floor_us=keys * (D // 2 + 4) / HBM * 1e6,
               idx_served_us=out["served"], idx_split_us=out["split"])  # fmt: skip
    print("IDX", json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                             for k, v in row.items()}), flush=True)  # fmt: skip
    return row


GAXIS = "grp"


def _tp8_devices(dcp):
    """[groups, dcp] devices: a DCP group = consecutive TCs (pairs = chips)."""
    devs = _group(8)
    return np.asarray(devs).reshape(8 // dcp, dcp)


def tp8_program(
    mesh, num_layers, attend, *, dcp, g, split, interpret, score_only=False, coll=True,
    attn_only="",
):
    """`num_layers` layers in one `fori_loop` (the program size does not grow
    with the layer count, so long chains average out the host's noise)."""
    groups = 8 // dcp

    def body(rank, grp, q, k, v, iq, ik, kv, idx, sc, sl, tb, cu, dist, had):
        if attn_only:
            kk, vv, ii = (_gather_kv(k, v, iq, rank, dcp, g) if g > 1
                          else (k[:, None], v[:, None], iq[:, None]))  # fmt: skip
            kv, idx, sc, cands0 = msa_dcp.msa_write_and_candidates_dcp(
                kk, vv, ii, ik, kv, idx, sl, tb, cu, dist, topk=TOPK, init_blocks=1,
                local_blocks=1, rank=rank, dcp=dcp, scale_cache=sc, interpret=interpret,
            )  # fmt: skip

            def a_layer(_, carry):
                link, acc = carry
                qr = msa_fp4.rotate_queries(q + link)
                c = cands0 + link.astype(jnp.float32)
                if attn_only == "kernel":  # attention kernel(s) alone, no collectives
                    out, _ = msa_dcp.msa_attend_dcp(
                        jnp.concatenate([qr] * dcp, axis=1), kv, jnp.stack([c] * dcp),
                        sl, tb, cu, sm_scale=D**-0.5, topk=TOPK, rank=rank, dcp=dcp,
                        scale_cache=sc, interpret=interpret,
                    )  # fmt: skip
                    v_ = out[0, 0, :8].astype(jnp.float32)
                else:
                    out = attend(rank, qr, kv, sc, c, sl, tb, cu, dist, had, dcp=dcp,
                                 g=g, interpret=interpret)  # fmt: skip
                    v_ = out[0, 0, :8].astype(jnp.float32)
                return (v_[0] * 0).astype(q.dtype), acc + v_

            _, acc = jax.lax.fori_loop(
                0, num_layers, a_layer, (jnp.zeros((), q.dtype), jnp.zeros((8,), jnp.float32))
            )
            return acc[None], kv, idx, sc

        def layer(_, carry):
            link, acc, kv, idx, sc = carry
            qq, iql = q + link, iq + link
            if g > 1:
                kk, vv, ii = _gather_kv(k, v, iql, rank, dcp, g)
            else:
                kk, vv, ii = k[:, None], v[:, None], iql[:, None]
            common = dict(topk=TOPK, init_blocks=1, local_blocks=1, rank=rank,
                          dcp=dcp, interpret=interpret)  # fmt: skip
            if split:
                kv, idx, sc = msa_dcp.msa_write_dcp(
                    kk, vv, ik, kv, idx, sl, tb, cu, rank=rank, dcp=dcp,
                    scale_cache=sc, interpret=interpret,
                )  # fmt: skip
                if coll:
                    iq_all = jax.lax.all_gather(
                        ii, GAXIS, axis=1, tiled=True
                    )  # [T, 4, D]
                else:  # cost probe: no exchange (wrong values)
                    iq_all = jnp.concatenate([ii] * groups, axis=1)
                parts = msa_dcp.split_candidates(
                    iq_all, idx, sl, tb, cu, dist, part=grp, parts=groups,
                    cache_heads=g, scale_cache=sc, **common,
                )  # fmt: skip
                if coll:
                    parts = jax.lax.all_gather(parts, GAXIS, axis=0)  # [NG, H, T, 2K]
                else:
                    parts = jnp.stack([parts] * groups)
                cands = msa_dcp.split_merge(parts, grp, g, TOPK)
            else:
                kv, idx, sc, cands = msa_dcp.msa_write_and_candidates_dcp(
                    kk, vv, ii, ik, kv, idx, sl, tb, cu, dist, scale_cache=sc, **common
                )
            if score_only:
                return ((cands[0, 0] * 0.0).astype(q.dtype),
                        acc + jnp.sum(cands[:, :8], axis=0), kv, idx, sc)  # fmt: skip
            out = attend(
                rank, msa_fp4.rotate_queries(qq), kv, sc, cands, sl, tb, cu, dist,
                had, dcp=dcp, g=g, interpret=interpret,
            )  # fmt: skip
            return (out[0, 0, 0] * 0.0, acc + jnp.sum(out[:, 0, :8].astype(jnp.float32), axis=0),
                    kv, idx, sc)  # fmt: skip

        init = (jnp.zeros((), q.dtype), jnp.zeros((8,), jnp.float32), kv, idx, sc)
        _, acc, kv, idx, sc = jax.lax.fori_loop(0, num_layers, layer, init)
        return acc[None], kv, idx, sc

    s, r = P((GAXIS, AXIS)), P()
    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(s, s) + (s,) * 5 + (s, s, s) + (r,) * 5,
            out_specs=(s, s, s, s),
            check_vma=False,
        ),
        donate_argnums=(7, 8, 9),
    )


def tp8_main(args, dcp, g, ctx, interpret):
    groups = 8 // dcp
    devs = (
        _tp8_devices(dcp)
        if not interpret
        else np.asarray(jax.devices()[:8]).reshape(groups, dcp)
    )
    mesh = Mesh(devs, (GAXIS, AXIS))
    print(
        f"tp8 conc {args.conc} dcp {dcp} g {g} groups {groups} mesh {mesh.devices.tolist()}",
        flush=True,
    )
    sh, rep = NamedSharding(mesh, P((GAXIS, AXIS))), NamedSharding(mesh, P())
    results = []
    for T in (int(x) for x in args.tokens.split(",")):
        rng = np.random.default_rng(args.seed + T)
        meta, num_pages, hot, keys = build_step(T, dcp, ctx, args.pool, rng)
        kv, idx, sc = build_caches(dcp, g, num_pages, hot, rng)
        # Every group holds the same caches (index keys are one head).
        caches = tuple(
            jnp.tile(c, (groups,) + (1,) * (c.ndim - 1)) for c in (kv, idx, sc)
        )
        heads_q = rng.standard_normal((groups, 1, T, HL, D))
        q = np.broadcast_to(heads_q, (groups, dcp, T, HL, D)).reshape(8 * T, HL, D)
        # One index-query head per KV head: both / all replicas of a head
        # hold its rows (c64: rank i = h * replicas + j holds head h).
        iq_heads = rng.standard_normal((groups * g, T, D))
        reps = dcp // g
        iq = np.stack([iq_heads[(i // dcp) * g + (i % dcp) // reps] for i in range(8)])
        k = rng.standard_normal((8, T, D))
        ik1 = rng.standard_normal((T, D))
        ik = np.broadcast_to(ik1, (8, T, D))

        def put(x):
            return jax.device_put(
                jnp.asarray(
                    np.ascontiguousarray(x).reshape((8 * T,) + x.shape[2:])
                    if x.ndim == 3 and x.shape[0] == 8
                    else x,
                    jnp.bfloat16,
                ),
                sh,
            )

        fixed = (
            jax.device_put(jnp.asarray(np.arange(8) % dcp, jnp.int32), sh),
            jax.device_put(jnp.asarray(np.arange(8) // dcp, jnp.int32), sh),
            jax.device_put(jnp.asarray(q, jnp.bfloat16), sh),
            put(k),
            put(k),
            put(iq),
            put(ik),
        )
        tail = (
            *(jax.device_put(jnp.asarray(m), rep) for m in meta),
            jax.device_put(jnp.asarray(msa_fp4.hadamard(D), jnp.bfloat16), rep),
        )

        def make():
            return (*fixed, *(jax.device_put(jnp.copy(c), sh) for c in caches), *tail)

        def timed(prog, batch=4):
            """(median, min) ms per call over `args.iters` samples, each the
            mean of `batch` back-to-back calls (device-bound, host-hidden)."""
            jax.block_until_ready(prog(*make()))
            samples = []
            for _ in range(args.iters):
                sets = [make() for _ in range(batch)]
                jax.block_until_ready(sets)
                t0 = time.perf_counter()
                outs_ = [prog(*a) for a in sets]
                jax.block_until_ready(outs_)
                samples.append((time.perf_counter() - t0) / batch)
                del outs_, sets
            return statistics.median(samples) * 1e3, min(samples) * 1e3

        row = dict(conc=args.conc, tokens=T, keys_per_rank=keys, attn=args.attn,
                   idx_floor_us=keys * (D // 2 + 4) / HBM * 1e6)  # fmt: skip
        outs = {}
        cases = []
        for spec in args.cases.split(","):
            mode, attn = spec.split(":")
            cases.append((spec, mode.startswith("split"), "score" in mode,
                          "nocoll" not in mode, attn,
                          {"attnonly": "attend", "kernelonly": "kernel"}.get(mode, "")))  # fmt: skip
        lo, hi = args.layers // 4, args.layers
        for name, split, score_only, coll, attn, attn_only in cases:
            mk = functools.partial(tp8_program, mesh, attend=VARIANTS[attn], dcp=dcp,
                                   g=g, split=split, interpret=interpret,
                                   score_only=score_only, coll=coll,
                                   attn_only=attn_only)  # fmt: skip
            p_hi = mk(hi)
            (t1, t1m), (tl, tlm) = timed(mk(lo)), timed(p_hi)
            n = max(hi - lo, 1)
            row[f"{name}_us"] = (tl - t1) / n * 1e3
            row[f"{name}_min_us"] = (tlm - t1m) / n * 1e3
            if not score_only and coll and not attn_only:
                outs[name] = np.asarray(p_hi(*make())[0])
                # Same seed -> same inputs: compare runs with different env.
                row[f"{name}_sha"] = hashlib.sha1(outs[name].tobytes()).hexdigest()[:12]
        names = list(outs)
        for n_ in names[1:]:
            row[f"{n_}_bitwise"] = bool(np.array_equal(outs[names[0]], outs[n_]))
        row["finite"] = bool(np.isfinite(outs[names[0]]).all())
        print("TP8", json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                 for k, v in row.items()}), flush=True)  # fmt: skip
        results.append(row)
    return results


def comp_main(args, dcp, g, ctx, interpret):
    """Per-layer time of the score half's parts on one TensorCore (rank 0),
    each in a `fori_loop` chain: write, indexer (served G heads over all
    pages / split 4 heads over 1 / (4 / G)), local top-k (served / split
    shapes), merge of the split parts."""
    groups = 4 // g
    rows = []
    for T in (int(x) for x in args.tokens.split(",")):
        rng = np.random.default_rng(args.seed + T)
        meta, num_pages, hot, keys = build_step(T, dcp, ctx, args.pool, rng)
        kv, idx, sc = build_caches(1, g, num_pages, hot[:1], rng)
        sl, tb, cu, dist = (jnp.asarray(m) for m in meta)
        _, pos, _ = msa.token_positions(T, sl, cu)
        rank = jnp.zeros((1,), jnp.int32)
        lens = msa_dcp.local_seq_lens(sl, block_size=PAGE, rank=0, dcp=dcp)
        srow = msa_fp4.index_scale_row(g)
        iq4 = jnp.asarray(rng.standard_normal((T, 4, D)), jnp.bfloat16)
        kk = jnp.asarray(rng.standard_normal((T, g, D)), jnp.bfloat16)
        ik = jnp.asarray(rng.standard_normal((T, D)), jnp.bfloat16)
        cand_kw = dict(block_size=PAGE, topk=TOPK, init_blocks=1, local_blocks=1)

        def served_scores(q, idx, sc):
            q = msa_fp4.rotate_queries(q[:, :g])
            if g == 1:
                return msa._block_scores(q, idx, lens, tb, cu, dist[2], interpret, sc, srow)
            return msa.partial_block_scores(q, idx, lens, tb, cu, dist[2], shards=1,
                                            interpret=interpret, scale_cache=sc,
                                            scale_row=srow).reshape(g * T, 1, -1)

        def split_scores(q, idx, sc):
            q = msa_fp4.rotate_queries(q)
            vl = msa_dcp.local_seq_lens(sl, block_size=PAGE, rank=0, dcp=groups * dcp)
            t = msa_dcp.split_part_tables(tb, sl.shape[0], 0, groups)
            return msa.partial_block_scores(q, idx, vl, t, cu, dist[2], shards=1,
                                            interpret=interpret, scale_cache=sc,
                                            scale_row=srow).reshape(4 * T, 1, -1)

        s_served = jax.jit(served_scores)(iq4, idx, sc)
        s_split = jax.jit(split_scores)(iq4, idx, sc)
        parts = jnp.asarray(rng.standard_normal((groups, 4, T, 2 * TOPK)), jnp.float32)

        def chain(step):
            def f(n, x, kv, idx, sc):
                def body(_, c):
                    link, acc, kv, idx, sc = c
                    y, kv, idx, sc = step(x + link, kv, idx, sc)
                    v = y.reshape(-1)[0].astype(jnp.float32)
                    return (v * 0).astype(x.dtype), acc + v, kv, idx, sc
                init = (jnp.zeros((), x.dtype), jnp.zeros((), jnp.float32), kv, idx, sc)
                return jax.lax.fori_loop(0, n, body, init)[1:]
            return jax.jit(f, static_argnums=0, donate_argnums=(2, 3, 4))

        steps = {
            "write": (kk, lambda x, kv, idx, sc: (lambda o: (o[0][0, 0, 0, :1], *o))(
                msa_dcp.msa_write_dcp(x, x, ik, kv, idx, sl, tb, cu, rank=rank, dcp=dcp,
                                      scale_cache=sc, interpret=interpret))),
            "idx_served": (iq4, lambda x, kv, idx, sc: (served_scores(x, idx, sc), kv, idx, sc)),
            "idx_split": (iq4, lambda x, kv, idx, sc: (split_scores(x, idx, sc), kv, idx, sc)),
            "topk_served": (s_served, lambda x, kv, idx, sc: (msa_dcp.local_candidates(
                x, jnp.tile(pos, g), rank=rank, dcp=dcp, interpret=interpret, **cand_kw)[0],
                kv, idx, sc)),
            "topk_split": (s_split, lambda x, kv, idx, sc: (msa_dcp.local_candidates(
                x, jnp.tile(pos, 4), rank=rank, dcp=groups * dcp, interpret=interpret,
                **cand_kw)[0], kv, idx, sc)),
            "merge_split": (parts, lambda x, kv, idx, sc: (msa_dcp.split_merge(
                x, 0, g, TOPK), kv, idx, sc)),
        }
        row = dict(conc=args.conc, tokens=T, keys_per_rank=keys,
                   idx_floor_us=keys * (D // 2 + 4) / HBM * 1e6)  # fmt: skip
        lo, hi = args.layers // 4, args.layers
        for name, (x, step) in steps.items():
            fn = chain(step)

            def timed(n, batch=4):
                def mk():
                    return [jnp.copy(c) for c in (kv, idx, sc)]
                jax.block_until_ready(fn(n, x, *mk()))
                best = []
                for _ in range(args.iters):
                    sets = [mk() for _ in range(batch)]
                    jax.block_until_ready(sets)
                    t0 = time.perf_counter()
                    o = [fn(n, x, *a) for a in sets]
                    jax.block_until_ready(o)
                    best.append((time.perf_counter() - t0) / batch)
                return statistics.median(best)

            row[name] = (timed(hi) - timed(lo)) / (hi - lo) * 1e6
        print("COMP", json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                  for k, v in row.items()}), flush=True)  # fmt: skip
        rows.append(row)
    return rows


def main():
    args = ARGS
    if args.smoke:
        args.layers, args.iters, args.ctx, args.pool = 2, 1, "4000:6000", 8
    interpret = args.smoke
    dcp = 2 if args.conc == 48 else 4
    g = dcp // 2
    ctx = tuple(int(x) for x in args.ctx.split(":"))
    if args.comp:
        rows = comp_main(args, dcp, g, ctx, interpret)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
        return
    if args.tp8:
        rows = tp8_main(args, dcp, g, ctx, interpret)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
        return
    if args.idx_split:
        rows = [
            idx_split(args, int(T), dcp, g, ctx, interpret)
            for T in args.tokens.split(",")
        ]
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1)
        return
    mesh = Mesh(np.asarray(_group(dcp)), (AXIS,))
    print(f"conc {args.conc} dcp {dcp} g {g} devices {list(mesh.devices)}", flush=True)
    print(
        "env", {k: v for k, v in os.environ.items() if k.startswith("TPU_")}, flush=True
    )
    sh, rep = NamedSharding(mesh, P(AXIS)), NamedSharding(mesh, P())
    results = []
    for T in (int(x) for x in args.tokens.split(",")):
        rng = np.random.default_rng(args.seed + T)
        meta, num_pages, hot, keys = build_step(T, dcp, ctx, args.pool, rng)
        caches = build_caches(dcp, g, num_pages, hot, rng)

        def bf16(*shape):
            return jax.device_put(
                jnp.asarray(
                    rng.standard_normal((dcp * shape[0],) + shape[1:]), jnp.bfloat16
                ),
                sh,
            )

        acts = (bf16(T, HL, D), bf16(T, D), bf16(T, D), bf16(T, D), bf16(T, D))
        fixed = (
            jax.device_put(jnp.arange(dcp, dtype=jnp.int32), sh),
            *acts,
        )
        tail = (
            *(jax.device_put(jnp.asarray(m), rep) for m in meta),
            jax.device_put(jnp.asarray(msa_fp4.hadamard(D), jnp.bfloat16), rep),
        )

        def make(caches=caches):
            return (*fixed, *(jax.device_put(jnp.copy(c), sh) for c in caches), *tail)

        def timed(prog):
            jax.block_until_ready(prog(*make()))
            samples = []
            for _ in range(args.iters):
                a = make()
                jax.block_until_ready(a)
                t0 = time.perf_counter()
                jax.block_until_ready(prog(*a))
                samples.append(time.perf_counter() - t0)
            return statistics.median(samples) * 1e3

        def per_layer(attend, what):
            t1 = timed(
                program(mesh, 1, attend, dcp=dcp, g=g, what=what, interpret=interpret)
            )
            tl = timed(
                program(
                    mesh,
                    args.layers,
                    attend,
                    dcp=dcp,
                    g=g,
                    what=what,
                    interpret=interpret,
                )
            )
            return (tl - t1) / max(args.layers - 1, 1) * 1e3

        # Floors per rank and layer: the index keys it scores, the KV pages
        # of its selections (the distinct pages per request, ~pool / dcp + 2).
        idx_bytes = keys * (D // 2 + 4)
        kv_pages = (T // ROWS) * (args.pool + 2) / dcp
        kv_bytes = kv_pages * PAGE * (g * D + 2 * g * 4)
        row = dict(
            conc=args.conc, tokens=T, pages=num_pages, keys_per_rank=keys,
            idx_floor_us=idx_bytes / HBM * 1e6, kv_floor_us=kv_bytes / HBM * 1e6,
        )  # fmt: skip
        if args.parts:
            row["write_us"] = per_layer(None, "write")
            row["score_us"] = per_layer(None, "score")
        outs = {}
        for name in args.variants.split(","):
            row[f"{name}_us"] = per_layer(VARIANTS[name], "all")
            prog = program(mesh, args.layers, VARIANTS[name], dcp=dcp, g=g, what="all",
                           interpret=interpret)  # fmt: skip
            outs[name] = np.asarray(prog(*make())[0])
        names = list(outs)
        for n in names[1:]:
            row[f"{n}_bitwise"] = bool(np.array_equal(outs[names[0]], outs[n]))
        print("RES", json.dumps({k: (round(v, 2) if isinstance(v, float) else v)
                                 for k, v in row.items()}), flush=True)  # fmt: skip
        results.append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
