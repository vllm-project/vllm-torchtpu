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
"""On-TPU parity and latency: layer-stacked vs batched whole-step QSA cache writes.

    python tests/megakernel/tpu_bench_stacked_qsa_writes.py --reqs 1 4 16

The 11 QSA layers of the whole-step kernel (3, 7, .., 43) at the served decode
geometry: FP8 pool with paired main K/V (two layers per main group) and packed
compressed keys (four layers per compressed page), `T = 4` tokens per request (MTP
K=3). Parity: both writers from the same pool over three steps, bitwise. Latency: mean
of `--iters` donated calls, per writer.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import (
    QSALayerCache,
    wholestep_qsa_writes,
)

PAGE_ROWS, WIDTH, HEAD_DIM, INDEX_DIM, RATIO, ROTARY, T = 512, 256, 256, 128, 4, 64, 4
N_LAYERS = 11
LAYERS = [
    QSALayerCache(
        compressed_slot=i % 4,
        main_slot=i % 2,
        k_scale=0.25 + 0.05 * i,
        v_scale=1.0 + 0.1 * i,
        main_group=f"main{i // 2}",
    )
    for i in range(N_LAYERS)
]


def raw(x):
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))


def build(rng, nreq, pages):
    """Distinct pages per main group, compressed group and ring."""
    free = iter(rng.permutation(np.arange(1, pages)))
    cols = 8
    n_main, n_comp = (N_LAYERS + 1) // 2, (N_LAYERS + 3) // 4
    mains = [np.array([[next(free) for _ in range(cols)] for _ in range(nreq)], np.int32)
             for _ in range(n_main)]
    comps = [np.array([[next(free) for _ in range(cols)] for _ in range(nreq)], np.int32)
             for _ in range(n_comp)]
    rings = [np.array([[next(free)] + [0] * 7 for _ in range(nreq)], np.int32)
             for _ in range(N_LAYERS)]
    return [
        (jnp.asarray(rings[i].reshape(-1)), jnp.asarray(mains[i // 2].reshape(-1)),
         jnp.asarray(comps[i // 4].reshape(-1)))
        for i in range(N_LAYERS)
    ]


def step(rng, ctx):
    nreq = len(ctx)
    starts = jnp.asarray(np.arange(nreq + 1) * T, jnp.int32)
    positions = jnp.asarray((ctx[:, None] + np.arange(T)[None]).reshape(-1), jnp.int32)

    def normal(*shape):
        return jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)

    rows = [(normal(nreq * T, HEAD_DIM), normal(nreq * T, HEAD_DIM),
             normal(nreq * T, INDEX_DIM)) for _ in LAYERS]
    norms = [(normal(HEAD_DIM), normal(INDEX_DIM)) for _ in LAYERS]
    return positions, starts, jnp.asarray(ctx), rows, norms


def writer(stacked):
    def fn(pool, rows, norms, tabs, positions, trig, starts, context):
        envs.VLLM_TPU_MK_STACKED_QSA_WRITES = stacked
        return wholestep_qsa_writes(
            pool, rows, norms, tabs, LAYERS, positions, trig, starts, context,
            compress_ratio=RATIO, norm_eps=1e-6, rotary_dim=ROTARY, max_query_len=T,
        )

    return jax.jit(fn, donate_argnums=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqs", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--pages", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--interpret", action="store_true", help="CPU smoke run")
    a = ap.parse_args()
    if a.interpret:
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        original = pl.pallas_call
        pl.pallas_call = lambda *x, **k: original(*x, **{**k, "interpret": True})
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        pltpu.get_tpu_info = lambda: info
    print("devices", jax.devices()[:1], flush=True)
    out = {}
    for nreq in a.reqs:
        rng = np.random.default_rng(nreq)
        tabs = build(rng, nreq, a.pages)
        trig = jnp.pad(jnp.asarray(rng.normal(0, 1, (8192, ROTARY)), jnp.bfloat16),
                       ((0, 0), (0, 128 - ROTARY)))
        pool0 = np.asarray(rng.integers(0, 256, (a.pages, PAGE_ROWS, 1, 4, WIDTH), np.uint8))
        mk = lambda: jax.lax.bitcast_convert_type(jnp.asarray(pool0), jnp.float8_e4m3fn)
        f_b, f_s = writer(False), writer(True)
        p_b, p_s = mk(), mk()
        ctx = rng.integers(0, 3000, nreq).astype(np.int32)
        ok = True
        for _ in range(3):
            args = step(rng, ctx)
            positions, starts, context, rows, norms = args
            p_b = f_b(p_b, rows, norms, tabs, positions, trig, starts, context)
            p_s = f_s(p_s, rows, norms, tabs, positions, trig, starts, context)
            ok &= bool(np.array_equal(raw(p_b), raw(p_s)))
            ctx = ctx + rng.integers(1, T + 1, nreq).astype(np.int32)
        changed = not np.array_equal(raw(p_b), pool0)
        res = {"bitwise": ok, "changed": changed}
        for name, f, p in (("batched", f_b, p_b), ("stacked", f_s, p_s)):
            positions, starts, context, rows, norms = step(rng, ctx)
            for _ in range(3):
                p = f(p, rows, norms, tabs, positions, trig, starts, context)
            p.block_until_ready()
            t0 = time.perf_counter()
            for _ in range(a.iters):
                p = f(p, rows, norms, tabs, positions, trig, starts, context)
            p.block_until_ready()
            res[f"{name}_us"] = round((time.perf_counter() - t0) / a.iters * 1e6, 1)
            n_ops = len(f.lower(p, rows, norms, tabs, positions, trig, starts, context)
                        .compile().as_text().splitlines())
            res[f"{name}_hlo_lines"] = n_ops
        out[nreq] = res
        print("RESULT", nreq, json.dumps(res), flush=True)
    print("SUMMARY", json.dumps(out), flush=True)
    assert all(r["bitwise"] and r["changed"] for r in out.values()), out


if __name__ == "__main__":
    main()
