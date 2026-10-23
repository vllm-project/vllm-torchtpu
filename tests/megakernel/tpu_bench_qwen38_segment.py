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
"""On-TPU Qwen3.8 segment kernel: compile, parity vs the reference, latency.

Run on a v7x 2x2x1 host (8 TensorCores):

    python tests/megakernel/tpu_bench_qwen38_segment.py --rows 8 16 32 64

Per verify bucket (T=4 tokens per sequence) it builds the full-geometry
segment kernel (and the one-layer tail), generates random weights sharded on
device, checks the kernel against `segment_reference`, and times chained
calls whose residual output feeds the next call's input (the loop carry), at
several chain lengths; the per-call cost is the least-squares slope.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel, planner
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import (
    pack_segment,
    random_segment_inputs,
)
from vllm_torchtpu.megakernel.models.qwen38_reference import segment_reference


def tile_profile(name: str, B: int = 8) -> Any:
    """Named tile choices for bucket `B`; `None` = library defaults fitted to VMEM."""
    from vllm_torchtpu.megakernel.exec_spec import Tiles

    if name == "default":
        return None
    if name in ("pipe_ep", "pipe_ep_h1"):  # e13 T=2, e2 T=1: software-pipelined EP experts
        hier = 1 if (name == "pipe_ep_h1" or B > 32) else 0
        nbe, nbd = (12, 10) if B <= 16 else ((6, 10) if B <= 32 else (3, 6))
        return Tiles(ebank=2 << 20, nbe=nbe, nbd=nbd, vmem_mib=62, hier=hier).with_blocks(
            {"moe": {"pipe_ep": True, "moe_unroll": True}}
        )
    if name == "pipe":  # single-tile experts
        return Tiles(ebank=3584 << 10, nbe=6 if B <= 16 else 4, nbd=10, vmem_mib=62).with_blocks(
            {"moe": {"pipe": True}}
        )
    raise ValueError(name)


def _shardings(tree: Any, R: int, mesh: Mesh) -> Any:
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    return jax.tree.map(
        lambda s: tp if len(s.shape) and s.shape[0] == R else rep, tree
    )


def _fold_put(x: Any, mesh: Mesh) -> Any:
    tp = NamedSharding(mesh, P("tp"))
    return jax.tree.map(lambda a: jax.device_put(kernel.fold(a), tp), x)


def run_bucket(
    g: Any,
    B: int,
    T: int,
    L: int,
    mesh: Mesh,
    chains: list[int],
    tiles: Any = None,
    gain: float = 1.0,
    zero: tuple[str, ...] = (),
    router_gain: float = 1.0,
    interpret: bool = False,
    profile: str = "default",
    ref_precision: str = "highest",
    same_experts: bool = False,
) -> dict:
    R = mesh.size
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=interpret, tiles=tiles)
    spec = qwen38.segment_spec(g, R, L=L)

    def gen(key):
        inp = random_segment_inputs(
            key, g, ex, L, gain=gain, zero=zero, router_gain=router_gain
        )
        if same_experts:
            # All routed experts identical: parity no longer depends on top-k picks.
            w = dict(inp.w_l)
            for k in ("e13", "e2", "e13_s", "e2_s"):
                w[k] = jnp.broadcast_to(w[k][:1, :, :1], w[k].shape)
            inp.w_l = w
        return inp

    shapes = jax.eval_shape(gen, jax.random.PRNGKey(0))
    inp = jax.jit(gen, out_shardings=_shardings(shapes, R, mesh))(jax.random.PRNGKey(B))

    # The library references index with concrete values, so this runs eagerly.
    moe_in: list = []
    # TPU's default f32 matmul precision is a single bf16 pass; the reference
    # must not be less precise than the kernel it checks.
    with jax.default_matmul_precision(ref_precision):
        ref, _ = segment_reference(
            g, ex, inp.x, inp.hc, inp.w_l, inp.st_l, inp.attn, inp.o_proj, L=L,
            moe_inputs=moe_in,
        )
    # Per token, the smallest gap between the TOPK-th and next router logit over
    # the segment's MoE layers: a near-tie the kernel may resolve differently.
    gaps = []
    for j, m in enumerate(moe_in):
        r = inp.w_l["router"][j] if inp.w_l["router"].ndim == 3 else inp.w_l["router"]
        lg = jnp.dot(m.astype(jnp.bfloat16), r.astype(jnp.bfloat16),
                     preferred_element_type=jnp.float32)
        top = jax.lax.top_k(lg, g.TOPK + 1)[0]
        gaps.append(top[:, g.TOPK - 1] - top[:, g.TOPK])
    min_gap = np.asarray(jnp.min(jnp.stack(gaps), axis=0), np.float32)

    def pack(i):
        return pack_segment(spec, g, ex, i)

    p_shapes = jax.eval_shape(pack, inp)
    w_mk, states, attn = jax.jit(pack, out_shardings=_shardings(p_shapes, R, mesh))(inp)
    hid = jnp.broadcast_to(
        jnp.pad(inp.x, ((0, ex.M - B), (0, 0)))[None], (R, ex.M, inp.x.shape[-1])
    )
    w_mk, states, attn, hid = (_fold_put(t, mesh) for t in (w_mk, states, attn, hid))
    pos = jax.device_put(inp.pos, NamedSharding(mesh, P()))

    step, info = kernel.make_step(mesh, spec, ex)
    mp = planner.plan(spec, info["exec"]).block_params["moe"]
    moe_path = {k: getattr(mp, k) for k in ("T13", "T2", "TE", "NBE", "PIPE", "PIPE_EP", "CH", "IR", "ISR", "LAZY_MOE", "topk_packed", "route_vec", "unroll")}
    t0 = time.perf_counter()
    _, outs = step(pos, hid, w_mk, states, attn)
    res = outs[RESIDUAL_OUT].block_until_ready()
    compile_s = time.perf_counter() - t0

    res0 = np.asarray(res.reshape(R, ex.M, -1)[0, :B], np.float32)
    ref = np.asarray(ref, np.float32)
    d = np.abs(res0 - ref)
    scale = float(np.abs(ref).mean())
    row_max = d.max(axis=1) / scale
    # A near-tie in MoE top-k can route one token to a different expert in the
    # kernel and the reference; that shows up as a whole-row outlier.
    parity = {
        "mean_rel": float(d.mean() / scale),
        "q999_rel": float(np.quantile(d, 0.999) / scale),
        "max_rel": float(d.max() / scale),
        "row_max_rel": [round(float(v), 3) for v in row_max],
        "rows_over_10pct": int((row_max > 0.1).sum()),
        "min_gap_outlier_rows": [round(float(v), 4) for v in min_gap[row_max > 0.1]][:16],
        "min_gap_good_rows_p10": round(float(np.quantile(min_gap[row_max <= 0.1], 0.1)), 4)
        if (row_max <= 0.1).any() else None,
        "moved_std": float(np.std(ref - np.asarray(inp.x, np.float32))),
    }
    parity["ok"] = parity["mean_rel"] < 1e-2 and parity["rows_over_10pct"] <= max(1, B // 32)
    # Where an outlier row deviates: per residual stream, and the last MoE
    # layer's routed experts (with owner ranks) for that row.
    h = g.H
    diag = []
    r_last = inp.w_l["router"][-1] if inp.w_l["router"].ndim == 3 else inp.w_l["router"]
    lg = jnp.dot(moe_in[-1].astype(jnp.bfloat16), r_last.astype(jnp.bfloat16),
                 preferred_element_type=jnp.float32)
    top = np.asarray(jax.lax.top_k(lg, g.TOPK)[1])
    for i in np.nonzero(row_max > 0.1)[0][:6]:
        per_stream = [round(float(d[i, b * h : (b + 1) * h].max() / scale), 3) for b in range(g.NHC)]
        diag.append({
            "row": int(i),
            "streams": per_stream,
            "experts": [int(e) for e in top[i]],
            "ranks": sorted({int(e) // (g.E // R) for e in top[i]}),
            "ranks_peak": int(np.bincount(top[i] // (g.E // R), minlength=R).max()),
        })
    good = [i for i in range(B) if row_max[i] <= 0.1]
    parity["outlier_diag"] = diag
    parity["good_rows_ranks_peak"] = [
        int(np.bincount(top[i] // (g.E // R), minlength=R).max()) for i in good[:16]
    ]

    def chain(n: int) -> float:
        h = hid
        t = time.perf_counter()
        for _ in range(n):
            _, o = step(pos, h, w_mk, states, attn)
            h = o[RESIDUAL_OUT]
        h.block_until_ready()
        return time.perf_counter() - t

    # Device clock: n chained calls inside one jit; the residual is the loop
    # carry, so XLA cannot hoist the kernel out of the loop.
    def dev_chain(n: int):
        # Weights, states and inputs are arguments, not closure constants
        # (closing over them embeds GBs of constants in the program).
        def run(h, p, w, st, a):
            return jax.lax.fori_loop(
                0, n, lambda _, x: step(p, x, w, st, a)[1][RESIDUAL_OUT], h
            )

        return jax.jit(run)

    dev_fns = {n: dev_chain(n) for n in chains}
    for f in dev_fns.values():
        f(hid, pos, w_mk, states, attn).block_until_ready()

    def dev_time(n: int) -> float:
        t = time.perf_counter()
        dev_fns[n](hid, pos, w_mk, states, attn).block_until_ready()
        return time.perf_counter() - t

    dbest = {n: min(dev_time(n) for _ in range(5)) for n in chains}
    d_slope, d_icpt = np.polyfit(
        np.array(list(dbest), np.float64), np.array(list(dbest.values()), np.float64), 1
    )

    chain(2)
    # Host dispatch jitter is one-sided, so fit the minimum of 5 reps per length.
    best = {n: min(chain(n) for _ in range(5)) for n in chains}
    n_arr = np.array(list(best), np.float64)
    t_arr = np.array(list(best.values()), np.float64)
    slope, icpt = np.polyfit(n_arr, t_arr, 1)
    resid = t_arr - (slope * n_arr + icpt)
    return {
        "rows": B,
        "T": T,
        "L": L,
        "gain": gain,
        "router_gain": router_gain,
        "zero": list(zero),
        "profile": profile,
        "ref_precision": ref_precision,
        "same_experts": same_experts,
        "H": g.H,
        "I": g.I,
        "moe_path": moe_path,
        "tiles": str(info["exec"].resolved_tiles),
        "compile_s": round(compile_s, 1),
        "parity": parity,
        "dev_us_per_call": round(d_slope * 1e6, 1),
        "dev_intercept_us": round(d_icpt * 1e6, 1),
        "us_per_call": round(slope * 1e6, 1),
        "intercept_us": round(icpt * 1e6, 1),
        "fit_resid_us": round(float(np.std(resid)) * 1e6, 1),
        "hoist_ratio": round(
            float(np.median(t_arr[n_arr == max(chains)]) / np.median(t_arr[n_arr == min(chains)])), 2
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 1])
    ap.add_argument("--chains", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--tiny", action="store_true", help="CPU interpret smoke run")
    ap.add_argument("--gain", type=float, nargs="+", default=[1.0])
    ap.add_argument("--zero", nargs="+", default=["none"], help="none | gdn | moe")
    ap.add_argument("--router-gain", type=float, nargs="+", default=[1.0])
    ap.add_argument("--geom", nargs="+", default=["2560:640"], help="H:I geometry overrides")
    ap.add_argument("--profile", nargs="+", default=["default"], help="default | pipe_ep | pipe")
    ap.add_argument("--ref-precision", nargs="+", default=["highest"], help="highest | default")
    ap.add_argument("--same-experts", nargs="+", type=int, default=[0], help="0 | 1")
    args = ap.parse_args()
    g, tiny_tiles = qwen38.FULL, None
    devs = jax.devices()
    if args.tiny:
        from vllm_torchtpu.megakernel.exec_spec import Tiles

        g, devs = qwen38.TINY, devs[:2]
        tiny_tiles = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
    print("devices", len(devs), devs[0].device_kind, "jax", jax.__version__, flush=True)
    mesh = Mesh(np.array(devs), ("tp",))
    out = []
    arms = [(gn, z, rg) for gn in args.gain for z in args.zero for rg in args.router_gain]
    geoms = []
    for gi_ in args.geom:
        h_, i_ = (int(v) for v in gi_.split(":"))
        geoms.append(g if (h_, i_) == (g.H, g.I) or args.tiny else dataclasses.replace(g, H=h_, I=i_))
    for g in geoms:
      for L in args.layers:
        for B in args.rows:
          for (gn, z, rg), prof, rp, se in [
              (a, pr, rp, se)
              for a in arms
              for pr in args.profile
              for rp in args.ref_precision
              for se in args.same_experts
          ]:
            tiles = tiny_tiles if args.tiny else tile_profile(prof, B)
            try:
                r = run_bucket(
                    g, B, args.T, L, mesh, args.chains, tiles, gn,
                    () if z == "none" else (z,), rg, args.tiny, prof, rp, bool(se),
                )
            except Exception as e:  # noqa: BLE001 — report and continue to the next bucket
                r = {"rows": B, "L": L, "error": f"{type(e).__name__}: {e}"[:2000]}
            print("RESULT", json.dumps(r), flush=True)
            out.append(r)
    dst = os.environ.get("CDK_OUTPUT_DIR")
    if dst:
        with open(os.path.join(dst, "qwen38_segment_bench.json"), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
