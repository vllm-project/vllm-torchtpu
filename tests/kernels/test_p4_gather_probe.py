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
"""Per-expert compute cost of the gather-form MoE inner loop on one TensorCore (phase 4, A2).

Weights of one expert stay resident in VMEM (no HBM stream), so the loop over NE experts
times only the compute of the expert-major formulation at B_all rows:

- gather_mxu: xg [CAP, H] = onehot [CAP, B] @ xb [B, H] (bf16 MXU), the expert's 12 tiles,
  scatter_mxu: part [B, H] += onehotT [B, CAP] @ yw [CAP, H] (the V2 prototype's gather mode).
- gather_dyn: xg rows by dynamic single-row loads xb[pl.ds(idx, 1)] (f32 or bf16 source),
  the expert, scatter_dyn: part[pl.ds(idx, 1)] += w * y (f32 row read-modify-write).
- list: per-expert row list from a transposed weight table wnT [E, B]: dynamic-row load, lane
  prefix sum, one-hot [CAP, B] and per-slot weights [CAP, 1].
- expert_only: the 12 expert tiles on CAP rows (the compute floor of one expert).

Collected only with P3_BENCH=1. Results: one MK_PROBE json line per variant with us per expert.
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

from vllm_torchtpu.kernels.qwen35_megakernel.config import BF16, F32, FP8, I32, NN, NT, rnd  # noqa: E402

H, IR, E = 4096, 1024, 512
T13, T2 = 8, 4
TK13, TK2 = H // T13, IR // T2
NE = 64
OUT = os.environ.get("MK_BENCH_OUT", "/tmp/p4-probe")


def _record(row):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "probe.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print("MK_PROBE " + json.dumps(row), flush=True)


def _expert(mm, xin, w13, w2, es, acc13, acc, mid, cap):
    for t in range(T13):
        m13 = mm(xin[:, t * TK13:(t + 1) * TK13], w13[t])
        if t == 0:
            acc13[...] = m13
        else:
            acc13[...] += m13
    h = acc13[...] * es[:, :2 * IR]
    mid[...] = rnd(rnd(jax.nn.silu(rnd(h[:, :IR]))) * rnd(h[:, IR:])).astype(BF16)
    for t in range(T2):
        m2 = mm(mid[:, t * TK2:(t + 1) * TK2], w2[t])
        if t == 0:
            acc[...] = m2
        else:
            acc[...] += m2
    return rnd(acc[...] * es[:, 2 * IR:])  # [cap, H]


def build(variant: str, B: int, CAP: int, src_dtype):
    """pallas_call: NE experts of one layer on B rows in the given formulation."""
    def mm(x, w):
        return lax.dot_general(x, w, NN, preferred_element_type=F32)

    def kernel(x_ref, xf_ref, wnT_ref, idx_ref, idxv_ref, wt_ref, w13, w2, es_ref,
               out_ref, acc13, acc, mid, xg, xgf, oh, cnt):
        out_ref[...] = jnp.zeros((B, H), F32)
        xg[...] = jnp.zeros((CAP, H), BF16)
        lane_b = lax.broadcasted_iota(I32, (1, B), 1)
        row_c = lax.broadcasted_iota(I32, (CAP, 1), 0)

        def list_build(e):
            # row list of expert e from its wnT row: nonzero lanes, exclusive prefix count
            col = wnT_ref[pl.ds(e, 1), :]  # [1, B]
            nz = (col != 0.0).astype(I32)
            ps = nz
            k = 1
            while k < B:
                sh = pltpu.roll(ps, k, 1)
                ps = ps + jnp.where(lane_b >= k, sh, 0)
                k *= 2
            pos = ps - nz  # exclusive count
            g = jnp.logical_and(pos == row_c, nz != 0)  # [CAP, B]
            oh[...] = g.astype(BF16)
            w = jnp.sum(jnp.where(g, col, 0.0), axis=1, keepdims=True)  # [CAP, 1]
            return w

        def body(e, carry):
            es = es_ref[e]
            if variant == "expert_only":
                y = _expert(mm, xg, w13, w2, es, acc13, acc, mid, CAP)
                out_ref[0:CAP, :] += y
                return carry
            if variant in ("list", "list_gather_mxu", "list_full_mxu"):
                w = list_build(e)
                if variant == "list":
                    out_ref[0:CAP, 0:1] += w
                    return carry
                xg[...] = mm(oh[...], x_ref[...]).astype(BF16)
                if variant == "list_gather_mxu":
                    out_ref[0:CAP, :] += xg[...].astype(F32)
                    return carry
                y = _expert(mm, xg, w13, w2, es, acc13, acc, mid, CAP)
                yw = (y * w).astype(BF16)
                out_ref[...] += lax.dot_general(oh[...], yw, (((0,), (0,)), ((), ())),
                                                preferred_element_type=F32)
                return carry
            if variant in ("gather_mxu", "full_mxu"):
                idx_c = idxv_ref[e][:, 0:1]  # [CAP, 1]
                g = (idx_c == lax.broadcasted_iota(I32, (CAP, B), 1)).astype(BF16)
                xg[...] = mm(g, x_ref[...]).astype(BF16)
                if variant == "gather_mxu":
                    out_ref[0:CAP, :] += xg[...].astype(F32)
                    return carry
                y = _expert(mm, xg, w13, w2, es, acc13, acc, mid, CAP)
                yw = (y * wt_ref[e][:, 0:1]).astype(BF16)
                out_ref[...] += lax.dot_general(g, yw, (((0,), (0,)), ((), ())),
                                                preferred_element_type=F32)
                return carry
            # dynamic single-row variants: indices from SMEM
            for i in range(CAP):
                r = idx_ref[e, i]
                if src_dtype == F32:
                    xgf[pl.ds(i, 1), :] = xf_ref[pl.ds(r, 1), :]
                else:
                    xg[pl.ds(i, 1), :] = x_ref[pl.ds(r, 1), :]
            if src_dtype == F32:
                xg[...] = xgf[...].astype(BF16)
            if variant == "gather_dyn":
                out_ref[0:CAP, :] += xg[...].astype(F32)
                return carry
            y = _expert(mm, xg, w13, w2, es, acc13, acc, mid, CAP)
            if variant == "full_dyn":
                for i in range(CAP):
                    r = idx_ref[e, i]
                    out_ref[pl.ds(r, 1), :] += y[i:i + 1] * wt_ref[e][i:i + 1, 0:1]
            else:  # full_dyn_mxu_scatter
                idx_c = idxv_ref[e][:, 0:1]
                g = (idx_c == lax.broadcasted_iota(I32, (CAP, B), 1)).astype(BF16)
                yw = (y * wt_ref[e][:, 0:1]).astype(BF16)
                out_ref[...] += lax.dot_general(g, yw, (((0,), (0,)), ((), ())),
                                                preferred_element_type=F32)
            return carry

        lax.fori_loop(0, NE, body, 0)

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, H), F32),
        in_specs=[vmem, vmem, vmem, smem, vmem, vmem, vmem, vmem, vmem],
        out_specs=vmem,
        scratch_shapes=[
            pltpu.VMEM((CAP, 2 * IR), F32),
            pltpu.VMEM((CAP, H), F32),
            pltpu.VMEM((CAP, IR), BF16),
            pltpu.VMEM((CAP, H), BF16),
            pltpu.VMEM((CAP, H), F32),
            pltpu.VMEM((CAP, B), BF16),
            pltpu.SMEM((8,), I32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 << 20),
        name=f"p4_probe_{variant}_b{B}_cap{CAP}",
    )


def _inputs(B, CAP, src_dtype, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, H)).astype(np.float32)
    xb = jnp.asarray(x, BF16)
    xf = xb.astype(F32)
    # routing: each row picks 10 of E; this core owns experts 0..NE-1
    wn = np.zeros((B, E), np.float32)
    idx = np.full((NE, CAP), 0, np.int32)
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
    wnT = np.ascontiguousarray(wn.T)  # [E, B]
    w13 = jnp.asarray(rng.standard_normal((T13, TK13, 2 * IR)).astype(np.float32) * 0.02, FP8)
    w2 = jnp.asarray(rng.standard_normal((T2, TK2, H)).astype(np.float32) * 0.02, FP8)
    es = jnp.asarray(rng.uniform(0.5, 1.5, (NE, 1, 2 * IR + H)).astype(np.float32))
    idxv = np.broadcast_to(idx[:, :, None], (NE, CAP, 128)).copy()
    wtv = np.broadcast_to(wt[:, :, None], (NE, CAP, 128)).copy()
    return (xb, xf, jnp.asarray(wnT), jnp.asarray(idx), jnp.asarray(idxv), jnp.asarray(wtv),
            w13, w2, es), mx


VARIANTS = ["expert_only", "gather_mxu", "full_mxu", "list", "list_gather_mxu", "list_full_mxu",
            "gather_dyn", "full_dyn", "full_dyn_mxu_scatter"]
CASES = [(256, 16), (256, 32), (384, 32), (512, 32), (768, 32), (768, 64)]  # B: lane multiples of 128


def _sel(name, default):
    v = os.environ.get(name)
    return [x for x in v.split(",") if x] if v else default


@pytest.mark.parametrize("B,CAP", [c for c in CASES
                                    if f"{c[0]}:{c[1]}" in _sel("PROBE_CASES", [f"{b}:{c}" for b, c in CASES])])
@pytest.mark.parametrize("variant", _sel("PROBE_VARIANTS", VARIANTS))
@pytest.mark.parametrize("src", ["f32", "bf16"])
def test_probe(variant, B, CAP, src):
    if src == "bf16" and not variant.endswith("dyn"):
        pytest.skip("source dtype matters only for the dynamic-row gather")
    devs = [d for d in jax.devices() if d.platform == "tpu"]
    if not devs:
        pytest.skip("needs a TPU")
    dev = devs[0]
    src_dtype = F32 if src == "f32" else BF16
    args, mx = _inputs(B, CAP, src_dtype)
    args = tuple(jax.device_put(a, dev) for a in args)
    row = {"variant": variant, "B": B, "CAP": CAP, "src": src, "NE": NE, "max_rows_per_expert": int(mx)}
    try:
        fn = jax.jit(build(variant, B, CAP, src_dtype))
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
