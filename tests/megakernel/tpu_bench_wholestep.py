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
"""On-TPU whole-step prototype: compile, parity vs the chained reference, latency.

Run on a v7x 2x2x1 host (8 TensorCores):

    python tests/megakernel/tpu_bench_wholestep.py --rows 8 --layers 4 8

Full Qwen3.8 geometry, `L` layers of `[GDN, GDN, GDN, QSA]` groups (every QSA layer in-kernel:
HC mix, projection, norms, RoPE, sparse attention over a shuffled FP8 pool, gate, o_proj).
Parity uses identical routed experts (strict: router flips cannot move the output) and random
experts (statistical). Latency: n chained calls inside one jit, residual as the loop carry,
per-call cost = least-squares slope; the per-group cost is the slope across `L`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.wholestep import inputs as wi
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec


def tiles_for(B: int, nbd: int | None, nbe: int | None) -> Tiles:
    """The segment bench's `pipe_ep` profile, rings trimmed for QSAFull's ~3.4 MiB of scratch."""
    d_nbe, d_nbd = (12, 10) if B <= 16 else ((6, 10) if B <= 32 else (3, 6))
    return Tiles(
        ebank=2 << 20, nbe=nbe or d_nbe - 2, nbd=nbd or d_nbd - 1, vmem_mib=62,
        hier=1 if B > 32 else 0,
    ).with_blocks({"moe": {"pipe_ep": True, "moe_unroll": True}})


def _shard(tree: Any, R: int, mesh: Mesh) -> Any:
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    return jax.tree.map(lambda s: tp if len(s.shape) and s.shape[0] == R else rep, tree)


def _fold_put(x: Any, mesh: Mesh) -> Any:
    tp = NamedSharding(mesh, P("tp"))
    return jax.tree.map(lambda a: jax.device_put(kernel.fold(a), tp), x)


def run(g, B, T, L, mesh, chains, budget, identical, nbd, nbe, tiny=False, slots=8, gain=1.0,
        small=False) -> dict:
    R = mesh.size
    tiles = (
        Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
        if (tiny or small) else tiles_for(B, nbd, nbe)
    )
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, tiles=tiles, interpret=tiny)
    spec = wholestep_spec(g, R, L=L, qsa=dict(budget=budget, slots=slots))
    n_pages = -(-(ex.M * (budget + 1) * 4) // 128) + 1

    def gen(key):
        d = wi.random_wholestep_inputs(
            7, g, ex, L, budget=budget, n_pages=n_pages, identical_experts=identical, gain=gain
        )
        d.pop("off")
        d.pop("budget")
        return d

    shapes = jax.eval_shape(gen, 0)
    d = jax.jit(gen, out_shardings=_shard(shapes, R, mesh), static_argnums=0)(0)
    d["off"], d["budget"] = 1, budget
    with jax.default_matmul_precision("highest"):
        ref, kv_ref = wi.wholestep_reference(g, ex, d, L)

    def pack(dd):
        return wi.pack_wholestep(spec, g, ex, {**dd, "off": 1, "budget": budget}, L)

    dd = {k: v for k, v in d.items() if k not in ("off", "budget")}
    w_mk, states = jax.jit(pack, out_shardings=_shard(jax.eval_shape(pack, dd), R, mesh))(dd)
    hid = jnp.broadcast_to(
        jnp.pad(d["inp"].x, ((0, ex.M - B), (0, 0)))[None], (R, ex.M, d["inp"].x.shape[-1])
    )
    w_mk, states, hid = (_fold_put(t, mesh) for t in (w_mk, states, hid))
    pos = jax.device_put(d["inp"].pos, NamedSharding(mesh, P()))
    xs = wi.extras(d, mesh, R)
    assert [e.name for e in kernel.extra_inputs(spec, ex)] == wi.EXTRA_NAMES

    step, info = kernel.make_step(mesh, spec, ex)
    t0 = time.perf_counter()
    _, outs = step(pos, hid, w_mk, states, *xs)
    res = outs[RESIDUAL_OUT].block_until_ready()
    compile_s = time.perf_counter() - t0

    res0 = np.asarray(res.reshape(R, ex.M, -1)[0, :B], np.float32)
    refn = np.asarray(ref, np.float32)
    dlt = np.abs(res0 - refn)
    scale = float(np.abs(refn).mean())
    row_max = dlt.max(axis=1) / scale
    parity = {
        "mean_rel": float(dlt.mean() / scale),
        "q999_rel": float(np.quantile(dlt, 0.999) / scale),
        "max_rel": float(dlt.max() / scale),
        "rows_over_10pct": int((row_max > 0.1).sum()),
        "moved_std": float(np.std(refn - np.asarray(d["inp"].x[:B], np.float32))),
    }
    parity["ok"] = parity["mean_rel"] < 1e-2 and (
        parity["max_rel"] < 0.1 if identical else parity["rows_over_10pct"] <= max(1, B // 32)
    )

    def dev_chain(n):
        def body(h, p, w, st, *e):
            return jax.lax.fori_loop(0, n, lambda _, x: step(p, x, w, st, *e)[1][RESIDUAL_OUT], h)

        return jax.jit(body)

    fns = {n: dev_chain(n) for n in chains}
    for f in fns.values():
        f(hid, pos, w_mk, states, *xs).block_until_ready()

    def t(n):
        t1 = time.perf_counter()
        fns[n](hid, pos, w_mk, states, *xs).block_until_ready()
        return time.perf_counter() - t1

    best = {n: min(t(n) for _ in range(5)) for n in chains}
    slope, icpt = np.polyfit(np.array(list(best), float), np.array(list(best.values()), float), 1)
    return {
        "rows": B, "T": T, "L": L, "budget": budget, "identical_experts": identical, "slots": slots,
        "gain": gain, "geom": "tiny" if (tiny or small) else "full",
        "tiles": str(info["exec"].resolved_tiles), "compile_s": round(compile_s, 1),
        "parity": parity, "dev_us_per_call": round(slope * 1e6, 1),
        "dev_intercept_us": round(icpt * 1e6, 1),
        "hoist_ratio": round(best[max(chains)] / best[min(chains)], 2),
    }


def run_step(g, B, T, L, mesh, chains, budget, max_groups, ctx, small, with_ref, gain=1.0,
             identical=True, manager_pages=0) -> dict:
    """Stage C on TPU: in-kernel indexer over the FP8 carrier + the step's own tokens."""
    R = mesh.size
    tiles = (Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64) if small
             else tiles_for(B, None, None))
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, tiles=tiles)
    n_pages = 8

    def gen(key):
        d = wi.random_wholestep_inputs(7, g, ex, L, budget=budget, n_pages=n_pages,
                                       identical_experts=identical, gain=gain)
        for k in ("off", "budget", "first"):
            d.pop(k)
        return d

    shapes = jax.eval_shape(gen, 0)
    d = jax.jit(gen, out_shardings=_shard(shapes, R, mesh), static_argnums=0)(0)
    d.update(off=1, budget=budget, first="gdn")
    d = wi.add_indexer_inputs(7, g, ex, d, L, max_groups=max_groups, ctx=ctx)
    d = wi.add_step_inputs(7, g, ex, d, L, ctx=ctx)
    d = wi.add_carrier(7, d, mp=manager_pages)
    d["carrier"] = wi.fp8_carrier(d["carrier"])
    ref = None
    if with_ref:
        with jax.default_matmul_precision("highest"):
            ref, _ = wi.wholestep_reference(g, ex, d, L)
    d.pop("_step_out", None)
    pps = d["mtab"].shape[-1]
    idx = dict(max_groups=max_groups, pages_per_seq=pps, carrier="fp8", chunk_pages=52 if manager_pages else 32,
               manager_pages=manager_pages)
    spec = wholestep_spec(g, R, L=L, qsa=dict(budget=budget), step=True, indexed=idx)
    meta = ("off", "budget", "first", "max_groups")

    def pack(dd):
        return wi.pack_wholestep(spec, g, ex, {**dd, **{k: d[k] for k in meta}}, L)

    dd = {k: v for k, v in d.items() if k not in meta and k != "carrier"}
    w_mk, states = jax.jit(pack, out_shardings=_shard(jax.eval_shape(pack, dd), R, mesh))(dd)
    hid = jnp.broadcast_to(
        jnp.pad(d["inp"].x, ((0, ex.M - B), (0, 0)))[None], (R, ex.M, d["inp"].x.shape[-1])
    )
    w_mk, states, hid = (_fold_put(t, mesh) for t in (w_mk, states, hid))
    pos = jax.device_put(d["inp"].pos, NamedSharding(mesh, P()))
    xs = wi.extras_c(d, mesh, R)
    step, info = kernel.make_step(mesh, spec, ex)
    t0 = time.perf_counter()
    _, outs = step(pos, hid, w_mk, states, *xs)
    res = outs[RESIDUAL_OUT].block_until_ready()
    compile_s = time.perf_counter() - t0
    out = {"rows": B, "L": L, "stage": "C", "max_groups": max_groups, "geom": "tiny" if small else "full",
           "manager_pages": manager_pages, "compile_s": round(compile_s, 1)}
    if ref is not None:
        res0 = np.asarray(res.reshape(R, ex.M, -1)[0, :B], np.float32)
        refn = np.asarray(ref, np.float32)
        dl = np.abs(res0 - refn)
        sc = float(np.abs(refn).mean())
        sel_all = slice(T, 2 * T)  # request 1: select-all rows (selection-insensitive)
        out["parity"] = {
            "mean_rel": float(dl.mean() / sc),
            "selectall_mean_rel": float(dl[sel_all].mean() / sc),
            "selectall_max_rel": float(dl[sel_all].max() / sc),
            "row_max_rel": [round(float(v), 3) for v in dl.max(axis=1) / sc],
        }

    def dev_chain(n):
        def body(h, p, w, st, *e):
            return jax.lax.fori_loop(0, n, lambda _, x: step(p, x, w, st, *e)[1][RESIDUAL_OUT], h)

        return jax.jit(body)

    fns = {n: dev_chain(n) for n in chains}
    for f in fns.values():
        f(hid, pos, w_mk, states, *xs).block_until_ready()

    def tm(n):
        t1 = time.perf_counter()
        fns[n](hid, pos, w_mk, states, *xs).block_until_ready()
        return time.perf_counter() - t1

    best = {n: min(tm(n) for _ in range(5)) for n in chains}
    slope, icpt = np.polyfit(np.array(list(best), float), np.array(list(best.values()), float), 1)
    out.update(dev_us_per_call=round(slope * 1e6, 1), dev_intercept_us=round(icpt * 1e6, 1),
               tiles=str(info["exec"].resolved_tiles))
    return out


def compile_only(g, B, T, L, mesh, budget, nbd, nbe, slots=8) -> dict:
    """Lower + compile the L-layer kernel from shapes only (no weights allocated): compile
    time and whether the program fits, e.g. at L = 48 (the whole model)."""
    R = mesh.size
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, tiles=tiles_for(B, nbd, nbe))
    spec = wholestep_spec(g, R, L=L, qsa=dict(budget=budget, slots=slots))
    n_pages = -(-(ex.M * (budget + 1) * 4) // 128) + 1

    def build(key):
        d = wi.random_wholestep_inputs(7, g, ex, L, budget=budget, n_pages=n_pages)
        w, st = wi.pack_wholestep(spec, g, ex, d, L)
        hid = jnp.broadcast_to(
            jnp.pad(d["inp"].x, ((0, ex.M - B), (0, 0)))[None], (R, ex.M, d["inp"].x.shape[-1])
        )
        ext = (
            kernel.fold(d["pool"]), d["pages"], d["kv_len"], jnp.asarray([1], jnp.int32),
            jnp.asarray([1.0, 1.0], jnp.float32), kernel.fold(jnp.broadcast_to(d["rope"][None], (R,) + d["rope"].shape)),
        )
        return d["inp"].pos, kernel.fold(hid), jax.tree.map(kernel.fold, w), jax.tree.map(kernel.fold, st), ext

    shapes = jax.eval_shape(build, 0)
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    pos_s, hid_s, w_s, st_s, ext_s = shapes
    with_sh = lambda t, sh: jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sh), t)  # noqa: E731
    ext_sh = tuple(
        with_sh(e, tp if i in (0, 5) else rep) for i, e in enumerate(ext_s)
    )
    step, info = kernel.make_step(mesh, spec, ex)
    t0 = time.perf_counter()
    lowered = jax.jit(step).lower(
        with_sh(pos_s, rep), with_sh(hid_s, tp), with_sh(w_s, tp), with_sh(st_s, tp), *ext_sh
    )
    t1 = time.perf_counter()
    lowered.compile()
    t2 = time.perf_counter()
    return {
        "rows": B, "L": L, "compile_only": True, "lower_s": round(t1 - t0, 1),
        "compile_s": round(t2 - t1, 1), "tiles": str(info["exec"].resolved_tiles),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[8])
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--chains", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--identical", type=int, nargs="+", default=[1, 0])
    ap.add_argument("--nbd", type=int, default=None)
    ap.add_argument("--nbe", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="CPU interpret smoke run")
    ap.add_argument("--slots", type=int, default=8, help="attention ring slots")
    ap.add_argument("--compile-only", action="store_true", help="shapes only: compile time / fit")
    ap.add_argument("--gain", type=float, default=1.0, help="sublayer output gain (smaller at H=2560)")
    ap.add_argument("--small", action="store_true",
                    help="TINY geometry compiled on 2 real TPU cores (the CPU test's config)")
    ap.add_argument("--stage-c", action="store_true", help="in-kernel indexer + step tokens")
    ap.add_argument("--max-groups", type=int, default=1024)
    ap.add_argument("--ctx", type=int, nargs="+", default=[3001, 100])
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--manager-pages", type=int, default=0)
    a = ap.parse_args()
    devs = jax.devices()[:2] if (a.tiny or a.small) else jax.devices()
    g = qwen38.TINY if (a.tiny or a.small) else qwen38.FULL
    print("devices", len(devs), devs[0].device_kind, "jax", jax.__version__, flush=True)
    mesh = Mesh(np.array(devs), ("tp",))
    out = []
    for L in a.layers:
        for B in a.rows:
            if a.stage_c:
                try:
                    r = run_step(g, B, a.T, L, mesh, a.chains, a.budget, a.max_groups, a.ctx,
                                 a.small, not a.no_ref, a.gain, bool(a.identical[0]),
                                 a.manager_pages)
                except Exception as e:  # noqa: BLE001
                    r = {"rows": B, "L": L, "stage": "C", "error": f"{type(e).__name__}: {e}"[:3000]}
                print("RESULT", json.dumps(r), flush=True)
                out.append(r)
                continue
            if a.compile_only:
                try:
                    r = compile_only(g, B, a.T, L, mesh, a.budget, a.nbd, a.nbe, a.slots)
                except Exception as e:  # noqa: BLE001
                    r = {"rows": B, "L": L, "compile_only": True, "error": f"{type(e).__name__}: {e}"[:3000]}
                print("RESULT", json.dumps(r), flush=True)
                out.append(r)
                continue
            for ident in a.identical:
                try:
                    r = run(g, B, a.T, L, mesh, a.chains, a.budget, bool(ident), a.nbd, a.nbe, a.tiny, a.slots,
                            a.gain, a.small)
                except Exception as e:  # noqa: BLE001 — report and continue
                    r = {"rows": B, "L": L, "identical": ident, "error": f"{type(e).__name__}: {e}"[:3000]}
                print("RESULT", json.dumps(r), flush=True)
                out.append(r)
    dst = os.environ.get("CDK_OUTPUT_DIR")
    if dst:
        with open(os.path.join(dst, "wholestep_bench.json"), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
