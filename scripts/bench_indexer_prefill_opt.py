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
"""Microbench: MSA indexer prefill block scores, TPU_MINIMAX_M3_INDEXER_PREFILL_OPT
and the indexer tile knobs.

One DCP rank's `msa_write_and_candidates_dcp` (writes + block scores + local
top-k) on an 8192-token prefill chunk at the served c48 layout: DCP 4, two
KV heads per rank (folded index heads), 4-bit caches with folded index
scales, 48 request slots, 2048 logical blocks per request (1M tokens).
Each arm is ``name=ENV:VAL;ENV:VAL`` (``PPS:N`` sets the large class's pages
per step, ``BQ:N`` its tile rows). L chained layers per program, per-layer
``(t_L - t_1) / (L - 1)``; with `--xprof` each arm's L-layer program is
traced and its device ops summed by name. Every arm's candidates and live
large-class block scores are compared bit for bit with the first arm's.

    PYTHONPATH=src python scripts/bench_indexer_prefill_opt.py --smoke
"""

import argparse
import glob
import os
import statistics
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.models.minimax_m3.kernels import msa, msa_dcp, msa_fp4, msa_indexer

PAGE, D = 128, 128
OPT = "TPU_MINIMAX_M3_INDEXER_PREFILL_OPT"
BQ = "TPU_MINIMAX_M3_KV4_INDEXER_LARGE_BLOCK_Q"


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="1x8192@230000,1x8192@100000,1x8192@400000")
    p.add_argument(
        "--arms",
        default="base=,unpair=OPT:unpair,unpair16=OPT:unpair+chunk16,c16=OPT:chunk16",
    )
    p.add_argument("--dcp", type=int, default=4)
    p.add_argument("--max-seqs", type=int, default=48)
    p.add_argument("--max-len", type=int, default=1 << 20)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--xprof", default="", help="trace dir (per-op sums)")
    p.add_argument("--smoke", action="store_true", help="CPU interpret, tiny")
    return p.parse_args()


def build_step(case, dcp, max_seqs, max_len, rng):
    n, rest = case.split("x")
    q, ctxs = rest.split("@")
    n, q = int(n), int(q)
    ctx = [int(c) for c in ctxs.split("/")]
    ctx = (ctx * n)[:n]
    q_lens = np.full(n, q)
    seq_lens = np.asarray(ctx) + q_lens
    logical = -(-seq_lens // (dcp * PAGE))
    pps = max_len // (dcp * PAGE)
    num_pages = int(logical.sum()) + 1
    tables = np.zeros((max_seqs, pps), np.int32)
    perm = rng.permutation(np.arange(1, num_pages))
    at = 0
    for j in range(n):
        tables[j, : logical[j]] = perm[at : at + logical[j]]
        at += logical[j]
    sl = np.zeros(max_seqs, np.int32)
    sl[:n] = seq_lens
    num_tokens = int(q_lens.sum())
    cu = np.full(max_seqs + 1, num_tokens, np.int32)
    cu[: n + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    meta = tuple(jnp.asarray(x) for x in (sl, tables.reshape(-1), cu))
    return meta, jnp.asarray([0, 0, n], jnp.int32), num_pages, num_tokens


def build_caches(num_pages, g, seed):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    kv_c = jax.random.randint(k1, (num_pages, PAGE // 8, 4, 2 * g * D), 0, 256).astype(
        jnp.uint8
    )
    nib = jax.random.randint(k2, (num_pages, PAGE, D), 0, 16).astype(jnp.uint8)
    idx = jax.jit(msa_fp4.pack_rows)(
        nib, jax.random.uniform(k3, (num_pages, PAGE), jnp.float32, 0.05, 0.2)
    )
    sc = jax.random.uniform(
        k4, (num_pages, msa_fp4.scale_rows(g), PAGE), jnp.float32, 0.02, 0.1
    )
    return kv_c, idx, sc


def program(args, num_layers, interpret):
    def score(k, v, iq, ik, kv_c, idx, sc, seq_lens, tables, cu, dist, rank):
        link = None
        for _ in range(num_layers):
            if link is not None:
                k = k + link
            out = msa_dcp.msa_write_and_candidates_dcp(
                k, v, iq, ik, kv_c, idx, seq_lens, tables, cu, dist,
                topk=args.topk, init_blocks=0, local_blocks=1, rank=rank,
                dcp=args.dcp, scale_cache=sc, interpret=interpret,
            )  # fmt: skip
            kv_c, idx, sc = out[0], out[1], out[2]
            link = (out[-1][0, 0] * 0.0).astype(k.dtype)
        return kv_c, idx, sc, out[-1]

    return jax.jit(score, donate_argnums=(4, 5, 6))


def large_scores(interpret):
    """The large class's raw block scores on a written index cache."""

    def f(iq, idx, local_lens, tables, cu, num_seqs):
        g = iq.shape[1]
        folded, _ = msa.fold_index_heads(msa_fp4.rotate_queries(iq), local_lens, cu)
        plan = msa_indexer.class_plan(
            folded[:, None], idx, local_lens, tables, g * cu, num_seqs,
            **msa.block_score_options(g), interpret=interpret,
        )  # fmt: skip
        sc = plan.large()
        return sc.scores, sc.num_tiles, plan.large_block_q

    return jax.jit(f)


def timed(fn, make_args, iters):
    jax.block_until_ready(fn(*make_args()))
    samples = []
    for _ in range(iters):
        a = make_args()
        jax.block_until_ready(a)
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a))
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def op_sums(trace_dir, layers):
    from vllm_torchtpu.tools import xplane_pb2

    sums = {}
    for path in glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True):
        space = xplane_pb2.XSpace()
        with open(path, "rb") as f:
            space.ParseFromString(f.read())
        for plane in space.planes:
            if not plane.name.startswith("/device:TPU:0"):
                continue
            for line in plane.lines:
                if line.name != "XLA Ops":
                    continue
                for ev in line.events:
                    name = plane.event_metadata[ev.metadata_id].name
                    name = name.split(" = ")[0].lstrip("%").split(".")[0]
                    sums[name] = sums.get(name, 0.0) + ev.duration_ps / 1e6 / layers
    return sorted(sums.items(), key=lambda kv: -kv[1])


def parse_arms(spec):
    arms = []
    for item in spec.split(","):
        name, _, sets = item.partition("=")
        kv = {}
        for s in filter(None, sets.split(";")):
            k, v = s.split(":", 1)
            kv[{"OPT": OPT, "BQ": BQ}.get(k, k)] = v
        arms.append((name, kv))
    return arms


_SET = set()


def apply_arm(kv, base_env, base_pps):
    for k in _SET | {OPT, BQ}:
        os.environ.pop(k, None)
        if k in base_env:
            os.environ[k] = base_env[k]
    _SET.clear()
    msa_indexer.DEFAULT_PAGES_PER_STEP = base_pps
    for k, v in kv.items():
        if k == "PPS":
            msa_indexer.DEFAULT_PAGES_PER_STEP = int(v)
        else:
            base_env.setdefault(k, os.environ[k]) if k in os.environ else None
            _SET.add(k)
            os.environ[k] = v


def main():
    args = _parse()
    interpret = args.smoke
    if args.smoke:
        args.layers, args.iters, args.max_len, args.max_seqs = 2, 1, 1 << 15, 4
        args.cases = "1x1024@3000"
    g = max(1, args.dcp // 2)
    base_env = {k: os.environ[k] for k in (OPT, BQ) if k in os.environ}
    base_pps = msa_indexer.DEFAULT_PAGES_PER_STEP
    arms = parse_arms(args.arms)
    print(f"devices {jax.devices()[:1]}; dcp {args.dcp} g {g} topk {args.topk} "
          f"max_seqs {args.max_seqs} max_len {args.max_len} env {base_env} "
          f"pps {base_pps}; arms {arms}", flush=True)  # fmt: skip
    for case in args.cases.split(","):
        rng = np.random.default_rng(sum(map(ord, case)))
        meta, dist, num_pages, T = build_step(
            case, args.dcp, args.max_seqs, args.max_len, rng
        )

        def bf16(*shape):
            return jnp.asarray(rng.standard_normal(shape), jnp.bfloat16)

        k, v, iq, ik = bf16(T, g, D), bf16(T, g, D), bf16(T, g, D), bf16(T, D)
        caches = build_caches(num_pages, g, 7)
        rank = jnp.asarray([args.dcp - 1], jnp.int32)  # the rank owning the last blocks

        def make_args():
            kv_c, idx, sc = (jnp.copy(c) for c in caches)
            return (k, v, iq, ik, kv_c, idx, sc, *meta, dist, rank)

        # Local lengths of the rank (as msa_write_and_candidates_dcp).
        seq_lens, tables, cu = meta
        local = msa_dcp.local_seq_lens(
            seq_lens, block_size=PAGE, rank=rank, dcp=args.dcp
        )
        num_seqs = dist[2]
        ref = None
        for name, kv in arms:
            apply_arm(kv, base_env, base_pps)
            res = {}
            for L in (1, args.layers):
                fn = program(args, L, interpret)  # a new trace per arm (env at trace)
                res[L] = timed(fn, make_args, args.iters)
                if L == args.layers:
                    final = fn(*make_args())
                    cands = np.asarray(final[-1])
                    if args.xprof:
                        d = tempfile.mkdtemp(dir=args.xprof)
                        a = make_args()
                        jax.block_until_ready(a)
                        with jax.profiler.trace(d):
                            jax.block_until_ready(fn(*a))
                        top = op_sums(d, args.layers)
                        total = sum(t for _, t in top)
                        bs = sum(t for n_, t in top if "block_scores" in n_)
                        print(f"OPS {case} {name}: device {total:.1f} block_scores {bs:.1f} "
                              f"us/layer", flush=True)  # fmt: skip
                        for n_, t in top[:6]:
                            print(f"OPS   {t:8.1f}  {n_[:100]}", flush=True)
            scores, nt, bq = large_scores(interpret)(
                iq, final[1], local, tables, cu, num_seqs
            )
            nt = int(nt)
            live_pages = int(-(-int(local[0]) // PAGE))
            sc = np.asarray(scores)[:live_pages, : nt * bq]
            per = (res[args.layers] - res[1]) / max(args.layers - 1, 1) * 1e3
            if ref is None:
                ref = (per, cands, sc)
                same_c = same_s = True
            else:
                same_c = cands.view(np.uint32).tobytes() == ref[1].view(np.uint32).tobytes()
                same_s = sc.shape == ref[2].shape and (
                    sc.view(np.uint32).tobytes() == ref[2].view(np.uint32).tobytes()
                )
            print(f"RES {case:24s} T{T} {name:14s} {per:8.1f} us/layer "
                  f"({(per / ref[0] - 1) * 100:+.1f}%) cands_bitwise {same_c} "
                  f"scores_bitwise {same_s} live {sc.shape}", flush=True)  # fmt: skip
        del caches


if __name__ == "__main__":
    main()
