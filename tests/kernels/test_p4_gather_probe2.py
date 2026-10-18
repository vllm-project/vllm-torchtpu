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
"""Per-expert compute floor of the routed-expert tile on one TensorCore (phase 4, A2), probe 2.

One expert's weights stay resident in VMEM; the loop over NE experts times the compute only.
Every variant adds a small fixed tail (out[0:8] += ...) so the loop body is never empty:

- base: the loop overhead and the per-expert scale-row load alone;
- expert: the 12 expert tiles on CAP rows (fp8 weights, bf16 rows), CAP 16 .. 80;
- expert_bf16w: the same with bf16 weights (no fp8 dequantization on the VPU);
- expert_pipe: two experts interleaved as in MoEBlock.experts_piped_ep (one w2 tile of expert
  i next to two w13 tiles of expert i+1 per step);
- list: the row list of one expert from its wnT row (dynamic-row load, lane prefix sum,
  one-hot [CAP, B], per-slot weights);
- gather: one-hot MXU gather of CAP rows out of B;
- scatter: transposed one-hot MXU scatter-add of CAP weighted rows into part [B, H] f32
  (hi/lo weight split);
- full: list + gather + expert + scatter.
"""
from __future__ import annotations

import json
import os
import time

import pytest

if not os.environ.get("P3_BENCH"):
    pytest.skip("phase 4 probe; set P3_BENCH=1 to run", allow_module_level=True)

import jax  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.qwen35_megakernel.config import BF16, F32, FP8, I32, NN, rnd  # noqa: E402

H, IR, E = 4096, 1024, 512
T13, T2 = 8, 4
TK13, TK2 = H // T13, IR // T2
NE = 64
OUT = os.environ.get("MK_BENCH_OUT", "/tmp/p4-probe2")


def _record(row):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "probe2.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print("MK_PROBE2 " + json.dumps(row), flush=True)


def mm(x, w):
    return lax.dot_general(x, w, NN, preferred_element_type=F32)


def build(variant: str, B: int, CAP: int, wdt):
    def kernel(x_ref, wnT_ref, idxv_ref, wt_ref, w13, w2, es_ref, out_ref,
               acc13, acc, mid, xg, xg2, acc13b, oh):
        out_ref[...] = jnp.zeros((B, H), F32)
        xg[...] = jnp.zeros((CAP, H), BF16)
        xg2[...] = jnp.zeros((CAP, H), BF16)
        lane_b = lax.broadcasted_iota(I32, (1, 128), 1)
        row_c = lax.broadcasted_iota(I32, (CAP, 1), 0)
        NQ = B // 128

        def scan(v):
            for sh in (1, 2, 4, 8, 16, 32, 64):
                v = v + jnp.where(lane_b >= sh, pltpu.roll(v, sh, 1), 0)
            return v

        def list_build(e):
            col = wnT_ref[pl.ds(e, 1), :]  # [1, B]
            nz = col != 0.0
            nzi = nz.astype(I32)
            segs, carry = [], jnp.zeros((1, 1), I32)
            for q in range(NQ):
                inc = scan(nzi[:, q * 128:(q + 1) * 128])
                segs.append(inc + carry)
                carry = carry + inc[:, 127:128]
            pos = (jnp.concatenate(segs, axis=1) if NQ > 1 else segs[0]) - nzi
            g = jnp.logical_and(pos == row_c, nz)
            w = jnp.sum(jnp.where(g, col, 0.0), axis=1, keepdims=True)
            return g.astype(BF16), w

        def expert(xin, es, a13, a2):
            for t in range(T13):
                m13 = mm(xin[:, t * TK13:(t + 1) * TK13], w13[t])
                if t == 0:
                    a13[...] = m13
                else:
                    a13[...] += m13
            h = a13[...] * es[:, :2 * IR]
            mid[...] = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)
            for t in range(T2):
                m2 = mm(mid[:, t * TK2:(t + 1) * TK2], w2[t])
                if t == 0:
                    a2[...] = m2
                else:
                    a2[...] += m2
            return rnd(a2[...] * es[:, 2 * IR:])

        def body(e, carry):
            es = es_ref[e]
            if variant == "base":
                out_ref[0:8, :] += es[:, 0:H]
                return carry
            if variant in ("expert", "expert_bf16w"):
                y = expert(xg, es, acc13, acc)
                out_ref[0:8, :] += y[0:8]
                return carry
            if variant == "expert_pipe":
                # steady state of experts_piped_ep: w2 tile of expert e (mid) next to 2 w13
                # tiles of expert e+1 (xg2) per step
                for s in range(T2):
                    m2 = mm(mid[:, s * TK2:(s + 1) * TK2], w2[s])
                    if s == 0:
                        acc[...] = m2
                    else:
                        acc[...] += m2
                    for t in (2 * s, 2 * s + 1):
                        m13 = mm(xg2[:, t * TK13:(t + 1) * TK13], w13[t])
                        if t == 0:
                            acc13b[...] = m13
                        else:
                            acc13b[...] += m13
                y = rnd(acc[...] * es[:, 2 * IR:])
                out_ref[0:8, :] += y[0:8]
                h = acc13b[...] * es[:, :2 * IR]
                mid[...] = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)
                return carry
            if variant == "list":
                g, w = list_build(e)
                oh[...] = g
                out_ref[0:8, 0:1] += w[0:8]
                return carry
            if variant == "gather":
                idx_c = idxv_ref[e][:, 0:1]
                g = (idx_c == lax.broadcasted_iota(I32, (CAP, B), 1)).astype(BF16)
                xg[...] = mm(g, x_ref[...]).astype(BF16)
                out_ref[0:8, :] += xg[0:8].astype(F32)
                return carry
            if variant == "scatter":
                idx_c = idxv_ref[e][:, 0:1]
                g = (idx_c == lax.broadcasted_iota(I32, (CAP, B), 1)).astype(F32)
                w = wt_ref[e][:, 0:1]
                w_hi = w.astype(BF16).astype(F32)
                gw = jnp.concatenate([(g * w_hi).astype(BF16), (g * (w - w_hi)).astype(BF16)], axis=0)
                y = xg[...]
                y2 = jnp.concatenate([y, y], axis=0)
                out_ref[...] += lax.dot_general(gw, y2, (((0,), (0,)), ((), ())),
                                                preferred_element_type=F32)
                return carry
            # full
            g, w = list_build(e)
            xg[...] = mm(g, x_ref[...]).astype(BF16)
            y = expert(xg, es, acc13, acc)
            w_hi = w.astype(BF16).astype(F32)
            gf = g.astype(F32)
            gw = jnp.concatenate([(gf * w_hi).astype(BF16), (gf * (w - w_hi)).astype(BF16)], axis=0)
            y2 = jnp.concatenate([y, y], axis=0).astype(BF16)
            out_ref[...] += lax.dot_general(gw, y2, (((0,), (0,)), ((), ())),
                                            preferred_element_type=F32)
            return carry

        lax.fori_loop(0, NE, body, 0)

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H), F32),
        in_specs=[vmem] * 7,
        out_specs=vmem,
        scratch_shapes=[
            pltpu.VMEM((CAP, 2 * IR), F32),
            pltpu.VMEM((CAP, H), F32),
            pltpu.VMEM((CAP, IR), BF16),
            pltpu.VMEM((CAP, H), BF16),
            pltpu.VMEM((CAP, H), BF16),
            pltpu.VMEM((CAP, 2 * IR), F32),
            pltpu.VMEM((CAP, B), BF16),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 << 20),
        name=f"p4_probe2_{variant}_b{B}_cap{CAP}",
    )


def _inputs(B, CAP, wdt, seed=0):
    rng = np.random.default_rng(seed)
    xb = jnp.asarray(rng.standard_normal((B, H)).astype(np.float32), BF16)
    wn = np.zeros((B, E), np.float32)
    idx = np.zeros((NE, CAP), np.int32)
    wt = np.zeros((NE, CAP), np.float32)
    rows = [[] for _ in range(NE)]
    for r in range(B):
        picks = rng.choice(E, 10, replace=False)
        w = rng.random(10).astype(np.float32)
        w /= w.sum()
        for p, wv in zip(picks, w):
            wn[r, p] = wv
            if p < NE:
                rows[p].append(r)
    mx = max(len(rr) for rr in rows)
    for e, rr in enumerate(rows):
        rr = rr[:CAP]
        idx[e, :len(rr)] = rr
        wt[e, :len(rr)] = wn[rr, e]
    wnT = np.ascontiguousarray(wn.T)
    w13 = jnp.asarray(rng.standard_normal((T13, TK13, 2 * IR)).astype(np.float32) * 0.02, wdt)
    w2 = jnp.asarray(rng.standard_normal((T2, TK2, H)).astype(np.float32) * 0.02, wdt)
    es = jnp.asarray(rng.uniform(0.5, 1.5, (NE, 1, 2 * IR + H)).astype(np.float32))
    idxv = np.broadcast_to(idx[:, :, None], (NE, CAP, 128)).copy()
    wtv = np.broadcast_to(wt[:, :, None], (NE, CAP, 128)).copy()
    return (xb, jnp.asarray(wnT), jnp.asarray(idxv), jnp.asarray(wtv), w13, w2, es), mx


VARIANTS = ["base", "expert", "expert_bf16w", "expert_pipe", "list", "gather", "scatter", "full"]
CASES = [(256, 16), (256, 32), (256, 64), (256, 80), (512, 32), (768, 32)]


def _sel(name, default):
    v = os.environ.get(name)
    return [x for x in v.split(",") if x] if v else default


@pytest.mark.parametrize("B,CAP", CASES)
@pytest.mark.parametrize("variant", _sel("PROBE_VARIANTS", VARIANTS))
def test_probe2(variant, B, CAP):
    if variant in ("base",) and (B, CAP) != (256, 32):
        pytest.skip("one base measurement")
    devs = [d for d in jax.devices() if d.platform == "tpu"]
    if not devs:
        pytest.skip("needs a TPU")
    dev = devs[0]
    wdt = BF16 if variant == "expert_bf16w" else FP8
    args, mx = _inputs(B, CAP, wdt)
    args = tuple(jax.device_put(a, dev) for a in args)
    row = {"variant": variant, "B": B, "CAP": CAP, "NE": NE, "max_rows_per_expert": int(mx)}
    try:
        fn = jax.jit(build(variant, B, CAP, wdt))
        t0 = time.time()
        out = fn(*args)
        jax.block_until_ready(out)
        row["compile_s"] = round(time.time() - t0, 1)
    except Exception as ex:  # noqa: BLE001
        row["error"] = f"{type(ex).__name__}: {str(ex)[:400]}"
        _record(row)
        pytest.fail(row["error"])
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    row.update({"median_us": round(times[3], 1), "min_us": round(times[0], 1),
                "us_per_expert": round(times[3] / NE, 2), "finite": bool(jnp.isfinite(out).all())})
    _record(row)
