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
"""TPU microbench of the pooled GDN v3 verify call at the c48 geometry.

One Qwen3.8-Flash TP8 rank (2 kq / 6 v heads, head dim 128) on the served
NHD fp8 pool (512-token pages, BF16 recurrent, FP32 conv state), MTP K=3,
served GDN flags (raw qkv / b / a, in-kernel zero-fill). Arms:

  base    the current kernels
  window  VLLM_TPU_GDN_V3_WINDOW_VERIFY=1
  single  VLLM_TPU_GDN_V3_SINGLE_LAUNCH=1
  both    both switches

For each arm and row count it first checks outputs and pool bytes against
`base` from the same inputs (bitwise on TPU, as upstream checks), then
times the jitted call (median of --iters, after --warmup).

  VLLM_KV_CACHE_LAYOUT=NHD python tests/kernels/gdn/v3/bench_window_verify.py
"""

import argparse
import contextlib
import time
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.gdn.v3 import wrapper
from vllm_torchtpu.layers.core import gdn_attention

N_KQ, N_V, D, KS = 2, 6, 128, 4
DIM = 2 * N_KQ * D + N_V * D
PAGE_TOKENS = 512
PAGE = (PAGE_TOKENS, 1, 4, 256)
SPEC = 3
W = SPEC + 1
SERVED = ("TPU_GDN_RAW_QKV", "TPU_GDN_RAW_BA", "TPU_GDN_KERNEL_ZERO_OUT")
SWITCHES = ("VLLM_TPU_GDN_V3_WINDOW_VERIFY", "VLLM_TPU_GDN_V3_SINGLE_LAUNCH")
ARMS = {
    "base": (),
    "window": ("VLLM_TPU_GDN_V3_WINDOW_VERIFY",),
    "single": ("VLLM_TPU_GDN_V3_SINGLE_LAUNCH",),
    "both": SWITCHES,
}


def _inputs(seqs: int, seed: int):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 0x3B, size=(1 + seqs * W + 1,) + PAGE, dtype=np.uint8)
    raw |= (rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7).astype(np.uint8)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    keys = iter(jax.random.split(jax.random.key(seed), 7))
    tokens = seqs * W
    step = dict(
        qkv=jax.random.normal(next(keys), (tokens, DIM)).astype(jnp.bfloat16),
        b=jax.random.normal(next(keys), (tokens, N_V)).astype(jnp.bfloat16),
        a=jax.random.normal(next(keys), (tokens, N_V)).astype(jnp.bfloat16),
        conv_weight=(0.5 * jax.random.normal(next(keys), (DIM, 1, KS))).astype(
            jnp.bfloat16
        ),
        conv_bias=(0.1 * jax.random.normal(next(keys), (DIM,))).astype(jnp.bfloat16),
        a_log=jax.random.normal(next(keys), (N_V,)),
        dt_bias=jax.random.normal(next(keys), (N_V,)),
        query_start_loc=jnp.arange(seqs + 1, dtype=jnp.int32) * W,
        distribution=jnp.full((3,), seqs, jnp.int32),
        seq_lens=jnp.full((seqs,), 1000 + W, jnp.int32),
        state_indices=jnp.asarray(1 + np.arange(seqs) * W, jnp.int32),
        ckpt_indices=jnp.asarray(
            1 + np.arange(seqs)[:, None] * W + np.arange(W)[None], jnp.int32
        ),
        read_offsets=jnp.asarray(np.arange(seqs) % W, jnp.int32),
    )
    return pool, step


def _make_fn(pool, flags):
    plan = gdn_attention._build_v3_pool_state_plan(
        pool,
        conv_dim=DIM,
        n_v=N_V,
        d_k=D,
        d_v=D,
        kernel_size=KS,
        pool_block_tokens=PAGE_TOKENS,
        qk_pair_layout=False,
        recurrent_state_dtype=jnp.bfloat16,
        conv_state_dtype=jnp.float32,
    )

    def fn(pool, step):
        with contextlib.ExitStack() as stack:
            for name in SERVED + SWITCHES:
                stack.enter_context(
                    mock.patch.object(envs, name, name in SERVED + flags)
                )
            return wrapper.fused_conv1d_gdn(
                conv_state=None,
                recurrent_state=None,
                state_source=pool,
                state_plan=plan,
                n_kq=N_KQ,
                n_v=N_V,
                d_k=D,
                d_v=D,
                kernel_size=KS,
                num_spec_tokens=SPEC,
                **step,
            )

    return jax.jit(fn, donate_argnums=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[96, 128])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    for rows in args.rows:
        seqs = rows // W
        pool0, step = _inputs(seqs, seed=rows)
        fns = {arm: _make_fn(pool0, flags) for arm, flags in ARMS.items()}
        ref_pool, ref_out = fns["base"](jnp.array(pool0, copy=True), step)
        ref = [np.asarray(x).view(np.uint8) for x in (ref_out, ref_pool)]
        for arm, fn in fns.items():
            pool, out = fn(jnp.array(pool0, copy=True), step)
            got = [np.asarray(x).view(np.uint8) for x in (out, pool)]
            diffs = [int((g != r).sum()) for g, r in zip(got, ref, strict=True)]
            pool = jnp.array(pool0, copy=True)
            for _ in range(args.warmup):
                pool, out = fn(pool, step)
            jax.block_until_ready(out)
            times = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                pool, out = fn(pool, step)
                jax.block_until_ready(out)
                times.append(time.perf_counter() - t0)
            print(
                f"rows={rows} arm={arm:6s} median_us={1e6 * np.median(times):8.1f} "
                f"p10_us={1e6 * np.percentile(times, 10):8.1f} "
                f"diff_bytes(out,pool)={diffs}",
                flush=True,
            )


if __name__ == "__main__":
    main()
