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
"""Standalone Multi-Chip mHC Microbenchmark & Parity Suite (DeepSeek-V4.1-Flash)."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.mhc import utils
from vllm_torchtpu.kernels.deepseek_v41.mhc import (
    mhc_fused_post_pre_v41,
    mhc_pre_v41,
)

HC_MULT = 4
HIDDEN_SIZE = 5120
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20

ALL_SHAPES = [
    ("decode", 16),
    ("decode", 64),
    ("decode", 128),
    ("decode", 512),
    ("prefill", 512),
    ("prefill", 1024),
    ("prefill", 2048),
    ("prefill", 4096),
]


def _ref_rms_norm(x_bf16: jax.Array, nw_bf16: jax.Array, eps: float) -> jax.Array:
    xf = x_bf16.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (xf * rstd).astype(jnp.bfloat16) * nw_bf16


def _ref_mhc_pre_v41(
    residual: jax.Array,
    pre_mix_in: jax.Array,
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    use_stream0: bool,
):
    *outer_shape, hc_mult, hidden_size = residual.shape
    num_tokens = residual.size // (hc_mult * hidden_size)
    res2d = residual.reshape(num_tokens, hc_mult * hidden_size)
    fn_f32 = hc_fn.astype(jnp.float32)
    x_f32 = res2d.astype(jnp.float32)
    mixes = jax.lax.dot_general(
        x_f32,
        fn_f32,
        dimension_numbers=(((1,), (1,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
    )
    sqr = jnp.sum(x_f32 * x_f32, axis=-1, keepdims=True)
    next_pre_mix, post_mix, comb_mix = utils.mhc_pre_gates(
        mixes,
        sqr,
        hc_mult,
        hidden_size,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    if use_stream0:
        collapsed_bf16 = res2d[:, :hidden_size]
    else:
        pre_2d = pre_mix_in.reshape(num_tokens, hc_mult).astype(jnp.float32)
        res_3d = residual.reshape(num_tokens, hc_mult, hidden_size).astype(jnp.float32)
        collapsed_bf16 = jnp.sum(pre_2d[:, :, None] * res_3d, axis=1).astype(jnp.bfloat16)
    normed = _ref_rms_norm(collapsed_bf16, norm_weight.reshape(1, hidden_size).astype(jnp.bfloat16), rms_eps)
    return (
        post_mix.reshape(*outer_shape, hc_mult, 1),
        comb_mix.reshape(*outer_shape, hc_mult, hc_mult),
        normed.reshape(*outer_shape, hidden_size),
        next_pre_mix.reshape(*outer_shape, hc_mult),
    )


def _ref_mhc_fused_post_pre_v41(
    x: jax.Array,
    residual: jax.Array,
    post_layer_mix: jax.Array,
    comb_res_mix: jax.Array,
    pre_mix_in: jax.Array,
    norm_weight: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    *outer_shape, hc_mult, hidden_size = residual.shape
    num_tokens = x.size // hidden_size
    x2d = x.reshape(num_tokens, hidden_size).astype(jnp.float32)
    res3d = residual.reshape(num_tokens, hc_mult, hidden_size).astype(jnp.float32)
    post3d = post_layer_mix.reshape(num_tokens, hc_mult, 1).astype(jnp.float32)
    comb3d = comb_res_mix.reshape(num_tokens, hc_mult, hc_mult).astype(jnp.float32)

    mixed_res = jnp.einsum(
        "tij,tih->tjh",
        comb3d,
        res3d,
        precision=jax.lax.Precision.HIGHEST,
    )
    new_res = (mixed_res + post3d * x2d[:, None, :]).astype(jnp.bfloat16)
    post_mix, comb_mix, normed, next_pre_mix = _ref_mhc_pre_v41(
        new_res.reshape(*outer_shape, hc_mult, hidden_size),
        pre_mix_in,
        norm_weight,
        hc_fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        use_stream0=False,
    )
    return (
        new_res.reshape(*outer_shape, hc_mult, hidden_size),
        post_mix,
        comb_mix,
        normed,
        next_pre_mix,
    )


def _max_abs_diff(a: jax.Array, b: jax.Array) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def run_worker(num_tokens_list: list[int], iters: int = 100) -> dict:
    rng = np.random.default_rng(42)
    mix_hc = HC_MULT * (HC_MULT + 2)
    hc_dim = HC_MULT * HIDDEN_SIZE

    fn_np = (rng.standard_normal((mix_hc, hc_dim), dtype=np.float32) * 0.02).astype(np.float32)
    scale_np = (1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32)
    base_np = (0.5 * rng.standard_normal(mix_hc)).astype(np.float32)
    nw_np = (1.0 + 0.1 * rng.standard_normal(HIDDEN_SIZE)).astype(np.float32)

    fn_jax = jnp.asarray(fn_np, dtype=jnp.float32)
    scale_jax = jnp.asarray(scale_np, dtype=jnp.float32)
    base_jax = jnp.asarray(base_np, dtype=jnp.float32)
    nw_jax = jnp.asarray(nw_np, dtype=jnp.bfloat16)

    pre_fn = jax.jit(
        lambda res, pre_in: mhc_pre_v41(
            res,
            pre_in,
            nw_jax,
            fn_jax,
            scale_jax,
            base_jax,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_PRE_EPS,
            hc_sinkhorn_eps=HC_SINKHORN_EPS,
            hc_post_mult_value=HC_POST_MULT_VALUE,
            sinkhorn_repeat=SINKHORN_REPEAT,
            use_stream0=False,
        )
    )
    fused_fn = jax.jit(
        lambda x, res, post_m, comb_m, pre_in: mhc_fused_post_pre_v41(
            x,
            res,
            post_m,
            comb_m,
            pre_in,
            nw_jax,
            fn_jax,
            scale_jax,
            base_jax,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_PRE_EPS,
            hc_sinkhorn_eps=HC_SINKHORN_EPS,
            hc_post_mult_value=HC_POST_MULT_VALUE,
            sinkhorn_repeat=SINKHORN_REPEAT,
        )
    )

    results = {}
    for T in num_tokens_list:
        res_np = rng.standard_normal((T, HC_MULT, HIDDEN_SIZE), dtype=np.float32) * 0.5
        x_np = rng.standard_normal((T, HIDDEN_SIZE), dtype=np.float32) * 0.5
        post_np = jax.nn.sigmoid(jnp.asarray(rng.standard_normal((T, HC_MULT, 1), dtype=np.float32))) * 2.0
        comb_raw = jnp.asarray(rng.standard_normal((T, HC_MULT, HC_MULT), dtype=np.float32))
        comb_np = jax.nn.softmax(comb_raw, axis=-1)
        pre_in_np = jax.nn.sigmoid(jnp.asarray(rng.standard_normal((T, HC_MULT), dtype=np.float32)))

        res_jax = jnp.asarray(res_np, dtype=jnp.bfloat16)
        x_jax = jnp.asarray(x_np, dtype=jnp.bfloat16)

        # Warmup + parity check for mhc_pre_v41
        out_pre = pre_fn(res_jax, pre_in_np)
        jax.block_until_ready(out_pre)
        ref_pre = _ref_mhc_pre_v41(
            res_jax,
            pre_in_np,
            nw_jax,
            fn_jax,
            scale_jax,
            base_jax,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_PRE_EPS,
            hc_sinkhorn_eps=HC_SINKHORN_EPS,
            hc_post_mult_value=HC_POST_MULT_VALUE,
            sinkhorn_repeat=SINKHORN_REPEAT,
            use_stream0=False,
        )
        jax.block_until_ready(ref_pre)

        pre_post_diff = _max_abs_diff(out_pre[0], ref_pre[0])
        pre_comb_diff = _max_abs_diff(out_pre[1], ref_pre[1])
        pre_norm_diff = _max_abs_diff(out_pre[2], ref_pre[2])
        pre_next_diff = _max_abs_diff(out_pre[3], ref_pre[3])

        t0 = time.perf_counter()
        for _ in range(iters):
            o = pre_fn(res_jax, pre_in_np)
            jax.block_until_ready(o)
        pre_us = (time.perf_counter() - t0) / iters * 1e6

        # Warmup + parity check for mhc_fused_post_pre_v41
        out_fused = fused_fn(x_jax, res_jax, post_np, comb_np, pre_in_np)
        jax.block_until_ready(out_fused)
        ref_fused = _ref_mhc_fused_post_pre_v41(
            x_jax,
            res_jax,
            post_np,
            comb_np,
            pre_in_np,
            nw_jax,
            fn_jax,
            scale_jax,
            base_jax,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_PRE_EPS,
            hc_sinkhorn_eps=HC_SINKHORN_EPS,
            hc_post_mult_value=HC_POST_MULT_VALUE,
            sinkhorn_repeat=SINKHORN_REPEAT,
        )
        jax.block_until_ready(ref_fused)

        fused_res_diff = _max_abs_diff(out_fused[0], ref_fused[0])
        fused_post_diff = _max_abs_diff(out_fused[1], ref_fused[1])
        fused_comb_diff = _max_abs_diff(out_fused[2], ref_fused[2])
        fused_norm_diff = _max_abs_diff(out_fused[3], ref_fused[3])
        fused_next_diff = _max_abs_diff(out_fused[4], ref_fused[4])

        t0 = time.perf_counter()
        for _ in range(iters):
            o = fused_fn(x_jax, res_jax, post_np, comb_np, pre_in_np)
            jax.block_until_ready(o)
        fused_us = (time.perf_counter() - t0) / iters * 1e6

        results[str(T)] = {
            "pre_us": round(pre_us, 2),
            "fused_us": round(fused_us, 2),
            "pre_gate_max_diff": max(pre_post_diff, pre_comb_diff, pre_next_diff),
            "pre_bf16_max_diff": pre_norm_diff,
            "fused_gate_max_diff": max(fused_post_diff, fused_comb_diff, fused_next_diff),
            "fused_bf16_max_diff": max(fused_res_diff, fused_norm_diff),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 64, 128, 512, 1024, 2048, 4096])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--parallel-4chips", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.parallel_4chips:
        buckets = [
            [16, 4096],
            [64, 2048],
            [128, 1024],
            [512],
        ]
        merged = {}

        def _spawn(chip_idx: int, toks: list[int]):
            env = os.environ.copy()
            env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
            env["TPU_PROCESS_BOUNDS"] = "1,1,1"
            env["TPU_VISIBLE_DEVICES"] = str(chip_idx)
            env["TPU_VISIBLE_CHIPS"] = str(chip_idx)
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--tokens",
                *[str(t) for t in toks],
                "--iters",
                str(args.iters),
                "--json",
            ]
            out = subprocess.check_output(cmd, env=env, text=True)
            for line in out.splitlines():
                if line.startswith("JSON_RESULT:"):
                    return json.loads(line[len("JSON_RESULT:") :])
            raise RuntimeError(f"Chip {chip_idx} failed:\n{out}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_spawn, i, b) for i, b in enumerate(buckets)]
            for f in futs:
                merged.update(f.result())

        print("| Mode | Tokens | `mhc_pre_v41` (us) | `mhc_fused_post_pre_v41` (us) | Pre Gate MaxDiff | Pre BF16 MaxDiff | Fused Gate MaxDiff | Fused BF16 MaxDiff |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for mode, T in ALL_SHAPES:
            r = merged[str(T)]
            print(
                f"| {mode} | {T} | {r['pre_us']:.2f} | {r['fused_us']:.2f} | "
                f"{r['pre_gate_max_diff']:.2e} | {r['pre_bf16_max_diff']:.2e} | "
                f"{r['fused_gate_max_diff']:.2e} | {r['fused_bf16_max_diff']:.2e} |"
            )
        return

    res = run_worker(args.tokens, iters=args.iters)
    if args.json:
        print("JSON_RESULT:" + json.dumps(res), flush=True)
    else:
        for k, v in res.items():
            print(f"T={k}: {v}", flush=True)


if __name__ == "__main__":
    main()
