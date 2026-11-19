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
"""Microbench: the DCP-2 decode attention half, unfused (`msa_dcp.attend_and_merge`
as served at c4 / c16) vs fused (`msa_dcp_fused.fused_attend_and_merge`).

One SPMD program over the two TensorCores of a chip runs L chained layers
(each layer's candidates depend on the previous layer's output through a
tiny add, the same in both variants); the per-layer cost is the slope
``(t_L - t_1) / (L - 1)``. Requests are 8 verify rows each over 100-150K
tokens of context, fp8 (c4 / c16) or 4-bit KV, top-16 blocks with the rows of a request
sharing most blocks (as served: ~20 distinct blocks per request). Reports
us per layer, the saving per 60-layer forward, and whether the outputs
match bit for bit.

    PYTHONPATH=.:src python scripts/bench_dcp_fused_decode.py --tokens 8,16,32,128
"""

import argparse
import json
import os
import statistics
import time

# The served c4 / c16 decode settings of the unfused path.
os.environ.setdefault("TPU_MINIMAX_M3_MSA_DECODE_WORK_LIST_KERNEL", "1")
os.environ.setdefault("TPU_TIGHT_VMEM", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.models.minimax_m3.kernels import (  # noqa: E402
    msa_dcp,
    msa_dcp_fused,
    msa_fp4,
)

AXIS = "dcp"
PAGE, D, HL, TOPK = 128, 128, 8, 16
ROWS = 8  # verify rows per request


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", default="8,16,32,128")
    p.add_argument("--layers", type=int, default=30)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--ctx", default="100000:150000")
    p.add_argument(
        "--pool", type=int, default=20, help="blocks a request's rows pick from"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--kv", default="fp8", choices=("fp8", "fp4"), help="c4 / c16 serve fp8"
    )
    p.add_argument("--json", default="")
    p.add_argument("--xprof", default="")
    p.add_argument("--smoke", action="store_true", help="CPU interpret, tiny shapes")
    return p.parse_args()


def _pair():
    """Two TensorCores of one chip (the served DCP pair)."""
    devs = jax.devices()
    if devs[0].platform != "tpu":
        return devs[:2]
    first = devs[0]
    mate = [d for d in devs[1:] if getattr(d, "coords", None) == first.coords]
    return [first, mate[0]] if mate else devs[:2]


def build_step(args, num_tokens, rng):
    """Per-rank caches and tables, and per-layer queries and candidates."""
    reqs = num_tokens // ROWS
    lo, hi = (int(x) for x in args.ctx.split(":"))
    ctx = rng.integers(lo, hi + 1, reqs)
    seq_lens = ctx + ROWS
    logical = -(-seq_lens // (2 * PAGE))
    max_seqs = max(reqs, 16)
    pps = int(-(-(hi + ROWS) // (2 * PAGE)))
    num_pages = int(logical.sum()) + 1
    tables = np.zeros((max_seqs, pps), np.int32)
    pages = rng.permutation(np.arange(1, num_pages))
    at = 0
    for j in range(reqs):
        tables[j, : logical[j]] = pages[at : at + logical[j]]
        at += logical[j]
    sl = np.zeros(max_seqs, np.int32)
    sl[:reqs] = seq_lens
    cu = np.full(max_seqs + 1, num_tokens, np.int32)
    cu[: reqs + 1] = np.arange(reqs + 1) * ROWS
    meta = (sl, tables.reshape(-1), cu)

    def candidates():
        """[2, T, 2 * topk] packed candidates of each rank for one layer."""
        out = np.zeros((2, num_tokens, 2 * TOPK), np.float32)
        out[:, :, :TOPK] = -np.inf
        out[:, :, TOPK:] = -1
        for j in range(reqs):
            nb = int(-(-seq_lens[j] // PAGE))
            pool = rng.choice(np.arange(1, nb - 1), args.pool, replace=False)
            for i in range(ROWS):
                own = (ctx[j] + i) // PAGE
                picks = rng.choice(pool, TOPK - 2, replace=False)
                blocks = np.concatenate([[0, own], picks])
                scores = np.concatenate([[3e38, 2e38], rng.standard_normal(TOPK - 2)])
                for r in range(2):
                    m = blocks % 2 == r
                    b, s = blocks[m], scores[m]
                    order = np.lexsort((b, -s))
                    t = j * ROWS + i
                    out[r, t, : m.sum()] = s[order]
                    out[r, t, TOPK : TOPK + m.sum()] = b[order]
        return out

    L = args.layers
    q = rng.standard_normal((L, 2 * HL, num_tokens, D)).astype(np.float32)
    cands = np.stack([candidates() for _ in range(L)])  # [L, 2, T, 2K]
    cands = cands.reshape(L, 2 * num_tokens, 2 * TOPK)
    return meta, num_pages, q, cands


def build_inputs(mesh, meta, num_pages, q, cands, rng, fp4):
    sh = NamedSharding(mesh, P(AXIS))
    lsh = NamedSharding(mesh, P(None, AXIS))
    rep = NamedSharding(mesh, P())
    qd = jax.device_put(jnp.asarray(q, jnp.bfloat16), lsh)
    if fp4:
        kv = jnp.asarray(
            rng.integers(0, 256, (2 * num_pages, PAGE // 8, 4, 2 * D), dtype=np.uint8)
        )
        scale = rng.uniform(0.02, 0.1, (2 * num_pages, msa_fp4.scale_rows(1), PAGE))
        extra = (
            jax.device_put(jnp.asarray(scale, jnp.float32), sh),
            jax.device_put(jnp.asarray(msa_fp4.hadamard(D), jnp.bfloat16), rep),
        )
        qd = jax.jit(msa_fp4.rotate_queries, out_shardings=lsh)(qd)
    else:
        kv = jax.jit(
            lambda key: jax.random.normal(
                key, (2 * num_pages, PAGE // 4, 4, 2 * D)
            ).astype(jnp.float8_e4m3fn),
            out_shardings=sh,
        )(jax.random.PRNGKey(int(rng.integers(1 << 30))))
        extra = ()
    return (
        jax.device_put(jnp.arange(2, dtype=jnp.int32), sh),
        qd,
        jax.device_put(kv, sh),
        jax.device_put(jnp.asarray(cands), lsh),
        *(jax.device_put(jnp.asarray(m), rep) for m in meta),
        *extra,
    )


def program(mesh, num_layers, fused, interpret, fp4):
    fn = msa_dcp_fused.fused_attend_and_merge if fused else msa_dcp.attend_and_merge

    def body(rank, q, kv, cands, seq_lens, tables, cu, *extra):
        outs = []
        link = None
        for layer in range(num_layers):
            c = cands[layer]
            if link is not None:  # chain the layers (same tiny op in both variants)
                c = c + link
            out = fn(
                rank,
                q[layer],
                kv,
                c,
                seq_lens,
                tables,
                cu,
                *extra,
                axis=AXIS,
                dcp=2,
                sm_scale=D**-0.5,
                topk=TOPK,
                interpret=interpret,
            )
            link = out[0, 0].astype(jnp.float32) * 0.0
            outs.append(out)
        return jnp.stack(outs)

    s, ls, r = P(AXIS), P(None, AXIS), P()
    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(s, ls, s, ls, r, r, r) + ((s, r) if fp4 else ()),
            out_specs=P(None, AXIS),
            check_vma=False,
        )
    )


def main():
    args = _parse()
    if args.smoke:
        args.layers, args.iters, args.ctx = 3, 1, "3000:6000"
    mesh = Mesh(np.asarray(_pair()), (AXIS,))
    print(
        f"devices {list(mesh.devices)}; layers {args.layers}; kv {args.kv}", flush=True
    )
    results = []
    for T in (int(x) for x in args.tokens.split(",")):
        rng = np.random.default_rng(args.seed + T)
        meta, num_pages, q, cands = build_step(args, T, rng)
        fp4 = args.kv == "fp4"
        inputs = build_inputs(mesh, meta, num_pages, q, cands, rng, fp4)
        row = {"tokens": T, "kv": args.kv, "pages_per_rank": num_pages}
        outs = {}
        for name, fused in (("unfused", False), ("fused", True)):
            p1 = program(mesh, 1, fused, args.smoke, fp4)
            pl_ = program(mesh, args.layers, fused, args.smoke, fp4)

            def timed(prog, inputs=inputs):
                jax.block_until_ready(prog(*inputs))
                samples = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    jax.block_until_ready(prog(*inputs))
                    samples.append(time.perf_counter() - t0)
                return statistics.median(samples) * 1e3, min(samples) * 1e3

            (t1, t1m), (tl, tlm) = timed(p1), timed(pl_)
            slope = (tl - t1) / max(args.layers - 1, 1) * 1e3
            slope_min = (tlm - t1m) / max(args.layers - 1, 1) * 1e3
            outs[name] = np.asarray(pl_(*inputs).astype(jnp.float32))
            row[name] = dict(
                t1_ms=t1, tl_ms=tl, us_per_layer=slope, us_per_layer_min=slope_min
            )
            print(
                f"T {T:4d} {name:8s} t1 {t1:8.3f} tL {tl:8.3f} ms  "
                f"us/layer {slope:7.2f} (min-based {slope_min:7.2f})",
                flush=True,
            )
            if args.xprof:
                with jax.profiler.trace(os.path.join(args.xprof, f"t{T}_{name}")):
                    jax.block_until_ready(pl_(*inputs))
        a, b = outs["unfused"], outs["fused"]
        row["bitwise"] = bool(np.array_equal(a, b))
        row["differ"] = int(np.sum(a != b))
        row["max_abs"] = float(np.nanmax(np.abs(a - b)))
        row["finite"] = bool(np.isfinite(a).all())
        save = row["unfused"]["us_per_layer"] - row["fused"]["us_per_layer"]
        row["saving_us_per_layer"] = save
        row["saving_ms_per_fwd60"] = save * 60 / 1e3
        print(
            f"T {T:4d} saving {save:6.2f} us/layer = {save * 0.06:6.3f} ms per "
            f"60-layer fwd; bitwise {row['bitwise']} (differ {row['differ']}, "
            f"max {row['max_abs']:.3g}, finite {row['finite']})",
            flush=True,
        )
        results.append(row)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
