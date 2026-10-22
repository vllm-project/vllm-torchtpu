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
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import (
    pack_segment,
    random_segment_inputs,
)
from vllm_torchtpu.megakernel.models.qwen38_reference import segment_reference


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
) -> dict:
    R = mesh.size
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=tiles is not None, tiles=tiles)
    spec = qwen38.segment_spec(g, R, L=L)

    def gen(key):
        return random_segment_inputs(key, g, ex, L, gain=gain, zero=zero)

    shapes = jax.eval_shape(gen, jax.random.PRNGKey(0))
    inp = jax.jit(gen, out_shardings=_shardings(shapes, R, mesh))(jax.random.PRNGKey(B))

    # The library references index with concrete values, so this runs eagerly.
    ref, _ = segment_reference(
        g, ex, inp.x, inp.hc, inp.w_l, inp.st_l, inp.attn, inp.o_proj, L=L
    )

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
        "moved_std": float(np.std(ref - np.asarray(inp.x, np.float32))),
    }
    parity["ok"] = parity["mean_rel"] < 1e-2 and parity["rows_over_10pct"] <= max(1, B // 32)

    def chain(n: int) -> float:
        h = hid
        t = time.perf_counter()
        for _ in range(n):
            _, o = step(pos, h, w_mk, states, attn)
            h = o[RESIDUAL_OUT]
        h.block_until_ready()
        return time.perf_counter() - t

    chain(2)
    samples = []
    for _ in range(3):
        for n in chains:
            samples.append((n, chain(n)))
    n_arr = np.array([s[0] for s in samples], np.float64)
    t_arr = np.array([s[1] for s in samples], np.float64)
    slope, icpt = np.polyfit(n_arr, t_arr, 1)
    resid = t_arr - (slope * n_arr + icpt)
    return {
        "rows": B,
        "T": T,
        "L": L,
        "gain": gain,
        "zero": list(zero),
        "tiles": str(info["exec"].resolved_tiles),
        "compile_s": round(compile_s, 1),
        "parity": parity,
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
    args = ap.parse_args()
    g, tiles = qwen38.FULL, None
    devs = jax.devices()
    if args.tiny:
        from vllm_torchtpu.megakernel.exec_spec import Tiles

        g, devs = qwen38.TINY, devs[:2]
        tiles = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
    print("devices", len(devs), devs[0].device_kind, "jax", jax.__version__, flush=True)
    mesh = Mesh(np.array(devs), ("tp",))
    out = []
    arms = [(gn, z) for gn in args.gain for z in args.zero]
    for L in args.layers:
        for B in args.rows:
          for gn, z in arms:
            try:
                r = run_bucket(
                    g, B, args.T, L, mesh, args.chains, tiles, gn,
                    () if z == "none" else (z,),
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
