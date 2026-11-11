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
"""On-TPU gate: an expert phase with FP4 (e2m1) weights vs FP8 (e4m3) weights.

    python tests/megakernel/tpu_bench_fp4_widen.py [--rows 4 16 64 96] [--interpret]

One TensorCore streams `D` routed experts of Qwen3.8 geometry (w13 [2560, 1280], w2 [640,
2560]) per "layer", the expert count of the MK / stock decode at each row bucket (EP8, top
10 of 512: 64 (1 - (1 - 10/512)^rows), max-TC at 4 rows), through a BlockSpec pipeline
(the DMA of expert d + 1 runs under the compute of expert d), and does gate/up, SiLU, down
for `rows` rows. Variants:

  fp8            e4m3 weights, per-output-channel scales (today's MK / fused_moe_decode).
  fp4_bits_pc    e2m1 packed 8 per u32, widened by shifts/masks to e4m3 (exact: the e4m3
                 byte `s 0 0 e e m 0 0` is the e2m1 value times 2^-6, subnormal included;
                 2^6 folds into the scale), per-output-channel scales.
  fp4_bits_qb    the same widen, per-contraction-block scales (w13 block 128, w2 block 64).
  fp4_astype_qb  the existing path's widen (`bitcast(u32 -> float4_e2m1fn).astype(fp8)`),
                 per-contraction-block scales (j-4160f23d measured this at 1.44-1.54x fp8).

Packing of the bit widen: u32 word (r, n) of a K-row matrix holds rows 4r..4r+3 (byte b,
low nibble: row 4r + b) and K/2 + 4r..4r+3 (high nibble), so the two widened halves are the
rows 0..K/2-1 and K/2..K-1 in order: no shuffle, the contraction splits in two halves.

Activations: `--x bf16` (the MK: bf16 rows x fp8 weights) or `fp8` (fused_moe_decode).
Timing: the grid repeats the D experts `--rep` times so the call is milliseconds; per-layer
time = call / rep. Also checks the bit widen against an e2m1 decode table (exact).
"""

from __future__ import annotations

import argparse
import functools
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

H, I = 2560, 640
N13 = 2 * I
FP8 = jnp.float8_e4m3fn
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)


def experts_for(rows: int) -> int:
    if rows <= 4:
        return 8  # max-TC experts at 4 rows (mean 4.9)
    return int(round(64 * (1 - (1 - 10 / 512) ** rows)))


# ----------------------------------------------------------------------------- widen
def widen_bits(w):
    """u32 [R, N] (8 e2m1 per word) -> (lo, hi) e4m3 [4R, N] each, value = e2m1 * 2^-6."""
    lo = ((w << 4) & jnp.uint32(0x80808080)) | ((w << 2) & jnp.uint32(0x1C1C1C1C))
    hi = (w & jnp.uint32(0x80808080)) | ((w >> 2) & jnp.uint32(0x1C1C1C1C))
    return pltpu.bitcast(lo, FP8), pltpu.bitcast(hi, FP8)


def pack_bits(codes: np.ndarray) -> np.ndarray:
    """e2m1 codes uint8 [K, N] (0..15) -> u32 [K/8, N] in the split-halves layout."""
    K, N = codes.shape
    lo, hi = codes[: K // 2], codes[K // 2 :]
    byte = (lo | (hi << 4)).astype(np.uint32).reshape(K // 8, 4, N)
    return (byte[:, 0] | (byte[:, 1] << 8) | (byte[:, 2] << 16) | (byte[:, 3] << 24)).astype(
        np.uint32
    )


def decode(codes: np.ndarray) -> np.ndarray:
    v = E2M1[codes & 7]
    return np.where(codes & 8, -v, v)


# ----------------------------------------------------------------------------- kernels
def _mm(x, w):
    return jax.lax.dot_general(x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)


def _ffn_tail(h, xdt):
    mid = jax.nn.silu(h[:, :I]) * h[:, I:]
    return mid.astype(xdt)


def make_kernel(variant: str, rows: int, D: int, rep: int, xdt, qb13=128, qb2=64, G=1, nbuf=2):
    nb13, nb2 = H // qb13, I // qb2

    def body(x_ref, w13_g, s13_g, w2_g, s2_g, o_ref):
        @pl.when((pl.program_id(0) == 0) & (pl.program_id(1) == 0))
        def _():
            o_ref[...] = jnp.zeros(o_ref.shape, jnp.float32)

        for g in range(G):
            one(x_ref, w13_g.at[pl.ds(g, 1)], s13_g.at[pl.ds(g, 1)], w2_g.at[pl.ds(g, 1)],
                s2_g.at[pl.ds(g, 1)], o_ref)

    def one(x_ref, w13_ref, s13_ref, w2_ref, s2_ref, o_ref):
        x = x_ref[...]
        if variant == "fp8":
            h = _mm(x, w13_ref[0]) * s13_ref[0]
            mid = _ffn_tail(h, xdt)
            o_ref[...] += _mm(mid, w2_ref[0]) * s2_ref[0]
            return
        if variant == "fp4_bits_pc":
            lo, hi = widen_bits(w13_ref[0])
            h = (_mm(x[:, : H // 2], lo) + _mm(x[:, H // 2 :], hi)) * s13_ref[0]
            mid = _ffn_tail(h, xdt)
            lo, hi = widen_bits(w2_ref[0])
            o_ref[...] += (_mm(mid[:, : I // 2], lo) + _mm(mid[:, I // 2 :], hi)) * s2_ref[0]
            return

        def blocks(w_ref, s_ref, src, K, qb):
            """sum_b (src[:, blk_b] @ W_b) * s_b over contraction blocks of qb rows."""
            acc = None
            if variant == "fp4_bits_qb":
                pr = qb // 4  # packed rows per widened block (per half)
                for half in range(2):
                    for b in range(K // 2 // qb):
                        lo, hi = widen_bits(w_ref[0, pl.ds(b * pr, pr), :])
                        blk = lo if half == 0 else hi
                        k0 = half * (K // 2) + b * qb
                        p = _mm(src[:, k0 : k0 + qb], blk) * s_ref[0, k0 // qb : k0 // qb + 1, :]
                        acc = p if acc is None else acc + p
            else:  # fp4_astype_qb: u32 [qb/8, N] -> e2m1 [qb, N] -> e4m3
                pr = qb // 8
                for b in range(K // qb):
                    blk = pltpu.bitcast(w_ref[0, pl.ds(b * pr, pr), :], jnp.float4_e2m1fn).astype(FP8)
                    p = _mm(src[:, b * qb : (b + 1) * qb], blk) * s_ref[0, b : b + 1, :]
                    acc = p if acc is None else acc + p
            return acc

        h = blocks(w13_ref, s13_ref, x, H, qb13)
        mid = _ffn_tail(h, xdt)
        o_ref[...] += blocks(w2_ref, s2_ref, mid, I, qb2)

    if variant == "fp8":
        w13s, w2s = (D, H, N13), (D, I, H)
        wdt = FP8
        s13s, s2s = (D, 1, N13), (D, 1, H)
    else:
        w13s, w2s = (D, H // 8, N13), (D, I // 8, H)
        wdt = jnp.uint32
        if variant == "fp4_bits_pc":
            s13s, s2s = (D, 1, N13), (D, 1, H)
        else:
            s13s, s2s = (D, nb13, N13), (D, nb2, H)

    def wspec(shape):
        kw = {"pipeline_mode": pl.Buffered(nbuf)} if nbuf != 2 else {}
        return pl.BlockSpec((G,) + shape[1:], lambda r, d: (d,) + (0,) * (len(shape) - 1), **kw)

    call = pl.pallas_call(
        body,
        grid=(rep, D // G),
        in_specs=[
            pl.BlockSpec((rows, H), lambda r, d: (0, 0)),
            wspec(w13s),
            wspec(s13s),
            wspec(w2s),
            wspec(s2s),
        ],
        out_specs=pl.BlockSpec((rows, H), lambda r, d: (0, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, H), jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"), vmem_limit_bytes=64 * 1024 * 1024
        ),
        name=f"fp4gate_{variant}",
    )
    return call, (w13s, wdt), (s13s,), (w2s, wdt), (s2s,)


def inputs(variant, rows, D, xdt, rng):
    call_shapes = {
        "fp8": ((D, H, N13), (D, I, H), (D, 1, N13), (D, 1, H)),
        "fp4_bits_pc": ((D, H // 8, N13), (D, I // 8, H), (D, 1, N13), (D, 1, H)),
        "fp4_bits_qb": ((D, H // 8, N13), (D, I // 8, H), (D, H // 128, N13), (D, I // 64, H)),
        "fp4_astype_qb": ((D, H // 8, N13), (D, I // 8, H), (D, H // 128, N13), (D, I // 64, H)),
    }[variant]
    w13s, w2s, s13s, s2s = call_shapes
    x = jnp.asarray(rng.normal(0, 1, (rows, H)), jnp.float32).astype(xdt)
    if variant == "fp8":
        mk = lambda s: jax.random.normal(jax.random.PRNGKey(s[0] + s[1]), s, jnp.float32).astype(FP8)  # noqa: E731
        w13, w2 = mk(w13s), mk(w2s)
    else:
        mk = lambda s: jax.random.bits(jax.random.PRNGKey(s[1]), s, jnp.uint32)  # noqa: E731
        w13, w2 = mk(w13s), mk(w2s)
    s13 = jnp.full(s13s, 1e-2, jnp.float32)
    s2 = jnp.full(s2s, 1e-2, jnp.float32)
    return x, w13, s13, w2, s2


def check_widen(interpret: bool) -> dict:
    """Bit widen vs the e2m1 table (all 16 codes, both halves, every byte lane)."""
    K, N = 64, 128
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, (K, N)).astype(np.uint8)
    codes[:16, 0] = np.arange(16)
    packed = jnp.asarray(pack_bits(codes))

    def k(w_ref, o_ref):
        lo, hi = widen_bits(w_ref[...])
        o_ref[: K // 2, :] = lo.astype(jnp.float32)
        o_ref[K // 2 :, :] = hi.astype(jnp.float32)

    out = pl.pallas_call(
        k, out_shape=jax.ShapeDtypeStruct((K, N), jnp.float32), interpret=interpret
    )(packed)
    got = np.asarray(out) * 64.0
    want = decode(codes)
    return {"exact": bool(np.array_equal(got, want)), "max_abs_err": float(np.max(np.abs(got - want)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[4, 16, 64, 96])
    ap.add_argument("--variants", nargs="+",
                    default=["fp8", "fp4_bits_pc", "fp4_bits_qb", "fp4_astype_qb"])
    ap.add_argument("--x", nargs="+", default=["bf16", "fp8"])
    ap.add_argument("--rep", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--group", type=int, nargs="+", default=[1], help="experts per grid step")
    ap.add_argument("--nbuf", type=int, nargs="+", default=[2], help="weight pipeline buffers")
    a = ap.parse_args()
    if a.interpret:
        original = pl.pallas_call
        pl.pallas_call = functools.partial(original, interpret=True)
    print("devices", jax.devices()[:1], flush=True)
    print("WIDEN_CHECK", json.dumps(check_widen(a.interpret)), flush=True)
    rng = np.random.default_rng(0)
    res = {}
    for rows in a.rows:
        for G in a.group:
          for nbuf in a.nbuf:
            D = -(-experts_for(rows) // G) * G
            for xname in a.x:
                xdt = jnp.bfloat16 if xname == "bf16" else FP8
                base = None
                for v in a.variants:
                    key = f"r{rows}_g{G}_b{nbuf}_x{xname}_{v}"
                    try:
                        call = make_kernel(v, max(rows, 8), D, a.rep, xdt, G=G, nbuf=nbuf)[0]
                        args = inputs(v, max(rows, 8), D, xdt, rng)
                        f = jax.jit(call)
                        jax.block_until_ready(f(*args))
                        ts = []
                        for _ in range(a.iters):
                            t0 = time.perf_counter()
                            jax.block_until_ready(f(*args))
                            ts.append(time.perf_counter() - t0)
                        layer_us = float(np.median(ts)) / a.rep * 1e6
                        if v == "fp8":
                            base = layer_us
                        floor = (4.92e6 if v == "fp8" else 2.61e6) * D / 3.37e12 * 1e6
                        r = {"rows": rows, "G": G, "nbuf": nbuf, "D": D, "x": xname, "variant": v,
                             "layer_us": round(layer_us, 2), "per_expert_us": round(layer_us / D, 3),
                             "hbm_floor_frac": round(floor / layer_us, 3),
                             "ratio_vs_fp8": round(layer_us / base, 3) if base else None}
                    except Exception as e:  # noqa: BLE001
                        r = {"rows": rows, "G": G, "nbuf": nbuf, "x": xname, "variant": v,
                             "error": f"{type(e).__name__}: {e}"[:300]}
                    res[key] = r
                    print("RESULT", json.dumps(r), flush=True)
    print("SUMMARY", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
