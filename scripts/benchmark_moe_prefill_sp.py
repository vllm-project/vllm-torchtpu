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
"""One GLM-5.2 MoE layer of a prefill step: today's path vs the
sequence-parallel split of TPU_MOE_PREFILL_SEQUENCE_PARALLEL.

Over an EP mesh of every device (GLM-5.2 geometry: hidden 6144, 256 experts
of inter 2048, top-8 sigmoid routing with a score-correction bias and a
routed scale of 2.5, one shared expert of inter 2048, per-channel fp8
weights), with random weights and activations:

  * today: every rank holds all T rows, routes them, runs its experts through
    the grouped-matmul path (`fused_moe_func`) and its shard of the TP shared
    expert, and the partial sums are all-reduced;
  * split: each rank holds its T/ep rows, routes them through the fused EP
    kernel (own tokens, bf16 result wire unless --wire fp8) and runs the
    replicated shared expert (block-scaled down projection, one block per TP
    shard) on them;
  * both are checked against an f32 reference (dequantized weights, the same
    routing) on rows spread over every rank, and timed (median of --iters,
    --chain layers per call so the dispatch cost is amortized);
  * the o_proj-side collectives each path pays are timed on [T, hidden] bf16:
    all-reduce (today) vs reduce-scatter + all-gather (split).

Usage (all local devices, or both hosts of a slice with COORD/NPROC set):
  python3 scripts/benchmark_moe_prefill_sp.py --tokens 4096
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.fused_moe.grouped_shared_expert import grouped_shared_expert
from vllm_torchtpu.kernels.fused_moe.v2 import WeightFormat, fused_ep_moe_v2
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    fused_moe_func,
    resolve_onehot_permute_threshold,
)

AX = "d"
H, INTER, E, K = 6144, 2048, 256, 8
ROUTED_SCALE = 2.5
SHARED_GROUP = 4  # as moe_prefill_sp.SHARED_GROUP
FP8 = jnp.float8_e4m3fn
FP8_MAX = float(jnp.finfo(FP8).max)


def _timeit(fn, *args, iters):
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3


def _host(x):
    if x.is_fully_addressable:
        return np.asarray(x)
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _silu_mul(x):
    d = x.shape[-1] // 2
    return jax.nn.silu(x[..., :d]) * x[..., d:]


def _route(logits, bias):
    """Top-8 by biased sigmoid score; unbiased weights renormalized."""
    scores = jax.nn.sigmoid(logits.astype(jnp.float32))
    _, idx = lax.top_k(scores + bias[None, :], K)
    w = jnp.take_along_axis(scores, idx, axis=-1)
    return w / jnp.maximum(w.sum(-1, keepdims=True), 1e-20), idx.astype(jnp.int32)


def make_weights(mesh, ep, key):
    """Random fp8 weights in the served layouts, built on device."""
    rep, shd = NamedSharding(mesh, P()), NamedSharding(mesh, P(AX))
    ks = jax.random.split(key, 10)
    tp_i = INTER // ep

    def fp8(k, shape):
        return (jax.random.normal(k, shape, jnp.float32) * 48).astype(FP8)

    def scale(k, shape):
        return 0.02 / 48 * jax.random.uniform(k, shape, jnp.float32, 0.5, 1.5)

    def experts():
        return (
            fp8(ks[0], (E, H, 2 * INTER)),
            fp8(ks[1], (E, INTER, H)),
            scale(ks[2], (E, 2 * INTER)),
            scale(ks[3], (E, H)),
        )

    w13, w2, s13, s2 = jax.jit(experts, out_shardings=(shd,) * 4)()

    def router_and_shared():
        gate = (jax.random.normal(ks[4], (E, H)) * 0.01).astype(jnp.bfloat16)
        bias = jax.random.normal(ks[5], (E,)) * 0.1
        w13s = fp8(ks[6], (H, 2 * INTER))
        s13s = scale(ks[7], (2 * INTER,))
        w2f = jax.random.normal(ks[8], (INTER, H), jnp.float32) * 0.02
        blk = w2f.reshape(ep, tp_i, H)
        s2s = jnp.max(jnp.abs(blk), axis=1) / FP8_MAX  # [ep, H]: per TP shard
        w2s = (blk / s2s[:, None, :]).astype(FP8).reshape(INTER, H)
        return gate, bias, w13s, s13s, w2s, s2s

    gate, bias, w13s, s13s, w2s, s2s = jax.jit(
        router_and_shared, out_shardings=(rep,) * 6
    )()

    def tp_shards(w13s, s13s, w2s, s2s):
        # Shard r of the merged gate/up: [gate_r | up_r]; of down: rows r.
        w13_tp = w13s.reshape(H, 2, ep, tp_i).transpose(2, 0, 1, 3).reshape(ep, H, -1)
        s13_tp = s13s.reshape(2, ep, tp_i).transpose(1, 0, 2).reshape(ep, -1)
        return w13_tp, s13_tp, w2s.reshape(ep, tp_i, H), s2s

    shared_tp = jax.jit(tp_shards, out_shardings=(shd,) * 4)(w13s, s13s, w2s, s2s)

    def group_shards(w13s, s13s, w2s, s2s):
        # Rank r holds member r % SHARED_GROUP's channels (moe_prefill_sp.member_slice).
        c, per = INTER // SHARED_GROUP, ep // SHARED_GROUP
        parts = []
        for m in (r % SHARED_GROUP for r in range(ep)):
            cols = jnp.r_[m * c : (m + 1) * c, INTER + m * c : INTER + (m + 1) * c]
            parts.append(
                (
                    w13s[:, cols],
                    s13s[cols],
                    w2s[m * c : (m + 1) * c],
                    s2s[m * per : (m + 1) * per],
                )
            )
        return tuple(jnp.concatenate(p) for p in zip(*parts))

    shared_group = jax.jit(group_shards, out_shardings=(shd,) * 4)(w13s, s13s, w2s, s2s)
    return dict(
        w13=w13,
        w2=w2,
        s13=s13,
        s2=s2,
        gate=gate,
        bias=bias,
        shared_rep=(w13s, s13s, w2s, s2s.reshape(1, ep, 1, H)),
        shared_tp=shared_tp,
        shared_group=shared_group,
    )


def today_fn(mesh, ep):
    """Today's MoE: all rows on every rank, grouped matmuls, TP shared
    expert, one all-reduce. x and logits replicated; returns replicated."""
    g = E // ep
    threshold = resolve_onehot_permute_threshold()

    def local(x, logits, bias, w13, w2, s13, s2, sw13, ss13, sw2, ss2):
        tw, ti = _route(logits, bias)
        start = (lax.axis_index(AX) * g).astype(jnp.int32)
        routed = fused_moe_func(
            x,
            w13,
            w2,
            s13[:, None, None, :],
            s2[:, None, None, :],
            None,
            None,
            tw.astype(x.dtype),
            ti,
            start,
            topk=K,
            activation="silu",
            use_ep=True,
            use_sparse_core=envs.USE_MOE_SPARSE_CORE,
            onehot_moe_permute_threshold=threshold,
            skip_padded_tokens=False,
            onehot_sortfree=envs.TPU_MOE_ONEHOT_SORTFREE,
        )
        act = _silu_mul(xla_quantized_matmul(x, sw13[0], ss13[0]))
        shared = xla_quantized_matmul(act, sw2[0], ss2[0])
        return lax.psum(routed * ROUTED_SCALE + shared, AX)

    sm = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(), P(), P()) + (P(AX),) * 8,
        out_specs=P(),
        check_vma=False,
    )
    return lambda x, logits, w: sm(
        x, logits, w["bias"], w["w13"], w["w2"], w["s13"], w["s2"], *w["shared_tp"]
    )


def split_fn(mesh, ep, *, wire, sharded_plan):
    """The split: own rows in, own rows out (x, logits sharded on rows)."""

    groups = tuple(
        tuple(range(g, g + SHARED_GROUP)) for g in range(0, ep, SHARED_GROUP)
    )
    shared = jax.shard_map(
        lambda *a: grouped_shared_expert(*a, axis=AX, groups=groups),
        mesh=mesh,
        in_specs=(P(AX),) * 5,
        out_specs=P(AX),
        check_vma=False,
    )
    kw = {} if wire == "fp8" else {"wire_dtype": jnp.bfloat16}

    def fn(x, logits, w):
        routed = fused_ep_moe_v2(
            x,
            w["w13"],
            w["w2"],
            w["s13"],
            w["s2"],
            logits,
            topk=K,
            renormalize=True,
            mesh=mesh,
            capacity=128,
            weight_format=WeightFormat.FP8,
            act_fn="silu",
            sharded_plan=sharded_plan,
            scoring_fn="sigmoid",
            score_bias=w["bias"],
            routed_scaling_factor=1.0,
            **kw,
        )
        return shared(x, *w["shared_group"]) + routed * ROUTED_SCALE

    return fn


def reference_fn(mesh, ep):
    """f32 reference on a few rows (replicated in, replicated out)."""
    g = E // ep

    def local(x, logits, bias, w13, w2, s13, s2, w13s, s13s, w2s, s2s):
        x = x.astype(jnp.float32)
        tw, ti = _route(logits, bias)
        base = lax.axis_index(AX) * g

        def one(e, acc):
            a = x @ (w13[e].astype(jnp.float32) * s13[e][None, :])
            y = _silu_mul(a) @ (w2[e].astype(jnp.float32) * s2[e][None, :])
            wgt = jnp.sum(jnp.where(ti == base + e, tw, 0.0), axis=-1, keepdims=True)
            return acc + wgt * y

        routed = lax.psum(lax.fori_loop(0, g, one, jnp.zeros_like(x)), AX)
        w2d = w2s.astype(jnp.float32).reshape(ep, -1, H) * s2s.reshape(ep, 1, H)
        act = _silu_mul(x @ (w13s.astype(jnp.float32) * s13s[None, :]))
        return routed * ROUTED_SCALE + act @ w2d.reshape(-1, H)

    sm = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P(), P(), P()) + (P(AX),) * 4 + (P(),) * 4,
        out_specs=P(),
        check_vma=False,
    )
    return jax.jit(
        lambda x, logits, w: sm(
            x, logits, w["bias"], w["w13"], w["w2"], w["s13"], w["s2"], *w["shared_rep"]
        )
    )


def chained(fn, chain):
    """`chain` layers per call, each fed the previous output, renormalized."""

    def run(x, logits, w):
        for _ in range(chain):
            y = fn(x, logits, w).astype(jnp.float32)
            x = (y * lax.rsqrt(jnp.mean(y * y, -1, keepdims=True) + 1e-6)).astype(
                x.dtype
            )
        return x

    return jax.jit(run)


def rel_l2(a, ref):
    a, ref = np.asarray(a, np.float32), np.asarray(ref, np.float32)
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[4096])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--check-rows", type=int, default=8, help="per rank")
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if os.environ.get("NPROC"):
        jax.distributed.initialize(
            coordinator_address=os.environ["COORD"],
            num_processes=int(os.environ["NPROC"]),
            process_id=int(os.environ["TPU_WORKER_ID"]),
        )
    devices = jax.devices()
    ep = len(devices)
    mesh = Mesh(np.array(devices), (AX,))
    rep, shd = NamedSharding(mesh, P()), NamedSharding(mesh, P(AX))
    wts = make_weights(mesh, ep, jax.random.key(0))
    results = {"ep": ep, "devices": str(devices[0].device_kind)}
    for T in a.tokens:
        t = T // ep
        r = {}
        x = jax.jit(
            lambda key, n=T: jax.random.normal(key, (n, H)).astype(jnp.bfloat16),
            out_shardings=rep,
        )(jax.random.key(T))
        logits = jax.jit(
            lambda x, gate: x.astype(jnp.float32) @ gate.astype(jnp.float32).T,
            out_shardings=rep,
        )(x, wts["gate"])
        x_s, logits_s = jax.jit(lambda *v: v, out_shardings=(shd, shd))(x, logits)

        today = today_fn(mesh, ep)
        splits = {
            f"split_{w}_plan{int(p)}": split_fn(mesh, ep, wire=w, sharded_plan=p)
            for w, p in (("bf16", False), ("bf16", True), ("fp8", False))
        }
        rows = (np.arange(ep)[:, None] * t + np.arange(a.check_rows)[None, :]).ravel()
        pick = jax.jit(lambda x, lg, r=rows: (x[r], lg[r]), out_shardings=(rep, rep))
        ref = _host(reference_fn(mesh, ep)(*pick(x, logits), wts))
        y_today = _host(jax.jit(today)(x, logits, wts))[rows]
        r["rel_l2_today"] = rel_l2(y_today, ref)
        for name, fn in splits.items():
            y = _host(jax.jit(fn)(x_s, logits_s, wts))[rows]
            r[f"rel_l2_{name}"] = rel_l2(y, ref)
            r[f"rel_l2_{name}_vs_today"] = rel_l2(y, y_today)

        r["today_ms"] = (
            _timeit(chained(today, a.chain), x, logits, wts, iters=a.iters) / a.chain
        )
        for name, fn in splits.items():
            r[f"{name}_ms"] = (
                _timeit(chained(fn, a.chain), x_s, logits_s, wts, iters=a.iters)
                / a.chain
            )

        def coll(f, spec_in, spec_out):
            sm = jax.shard_map(
                f, mesh=mesh, in_specs=spec_in, out_specs=spec_out, check_vma=False
            )

            def run(x):
                for _ in range(a.chain):
                    x = sm(x)
                return x

            return jax.jit(run)

        ar = coll(lambda v: lax.psum(v, AX), P(AX), P(AX))
        rs_ag = coll(
            lambda v: lax.all_gather(
                lax.psum_scatter(v, AX, tiled=True), AX, tiled=True
            ),
            P(AX),
            P(AX),
        )
        # Each rank its own [T, H].
        xf = jax.jit(lambda v: jnp.tile(v, (ep, 1)), out_shardings=shd)(x)
        r["all_reduce_ms"] = _timeit(ar, xf, iters=a.iters) / a.chain
        r["rs_plus_ag_ms"] = _timeit(rs_ag, xf, iters=a.iters) / a.chain
        if a.trace_dir:
            jax.profiler.start_trace(os.path.join(a.trace_dir, f"t{T}"))
            for fn, args in (
                (chained(today, a.chain), (x, logits, wts)),
                (chained(splits["split_bf16_plan0"], a.chain), (x_s, logits_s, wts)),
            ):
                for _ in range(3):
                    jax.block_until_ready(fn(*args))
            jax.profiler.stop_trace()
        results[f"T{T}"] = r
        if jax.process_index() == 0:
            print(f"T={T} " + json.dumps(r), flush=True)
    if a.out and jax.process_index() == 0:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
