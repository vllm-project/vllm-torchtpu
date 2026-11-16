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
"""Device time of the pooled GDN v3 decode kernel at the served Flash geometry.

Times the BATCHED kernel alone (batched_only) for 0 to 48 verify windows of
4 tokens, in SELECT and LOG mode, at the dynamic tile size and at tile 4, and
the full pooled call (BATCHED plus the empty PER_SEQ kernel, with and without
the output zero fill), each as the mean of N calls run back to back inside
one jitted loop. Prints one line
per case and a least-squares fit of the time against the number of tiles.

usage: python tests/kernels/gdn/v3/bench_gdn_fixed_cost.py [iters]
"""

import dataclasses
import functools
import os
import sys
import time

os.environ.setdefault("TPU_ENABLE_GDN_DYNAMIC_TILING", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from vllm_torchtpu.kernels.gdn.v3 import config, tiling, wrapper  # noqa: E402
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput  # noqa: E402

N_KQ, N_V, D, K, DIM, KS = 2, 6, 128, 3, 1280, 4
W = K + 1
F32 = jnp.dtype(jnp.float32)
SELECT_PLAN = config.StateSourcePlan(
    stride=1,
    conv=config.StateRegion(0, 1, 384, 16, F32, 1, 15),
    recurrent=config.StateRegion(0, 1, 0, 384, F32, 2, 768),
)
LOG_PLAN = dataclasses.replace(
    SELECT_PLAN,
    log=config.StateRegion(0, 1, 400, 14, F32, 2, 27),
    log_conv=config.StateRegion(0, 1, 414, 15, F32, 1, 15),
)
MAX_SEQS = 64
NUM_BLOCKS = 1 + MAX_SEQS * W


def _make_loop(plan, iters, batched_only, zero_out):
    def step(_, carry):
        pool, inputs = carry
        qkv, b, a, conv_w, a_log, dt_bias, qsl, si, dist, sl, ro, ck = inputs
        pool, out = wrapper.fused_conv1d_gdn(
            Conv1DInput(qkv, jnp.bfloat16), b, a, None, None, conv_w, None,
            a_log, dt_bias, qsl, si, dist, sl, ro, ck,
            n_kq=N_KQ, n_v=N_V, d_k=D, d_v=D, kernel_size=KS,
            state_source=pool, state_plan=plan, num_spec_tokens=K,
            conv_weight_kmajor=True, batched_only=batched_only,
            zero_initialize_out=zero_out,
        )
        # Feed one output element back so no call is dead code.
        qkv = qkv.at[0, 0, 0].add(out[0, 0].astype(jnp.float32) * 0)
        return pool, (qkv, b, a, conv_w, a_log, dt_bias, qsl, si, dist, sl, ro, ck)

    @functools.partial(jax.jit, donate_argnums=0)
    def run(pool, inputs):
        pool, _ = jax.lax.fori_loop(0, iters, step, (pool, inputs))
        return pool

    return run


def _inputs(rng, num_seqs, tokens):
    qlens = np.zeros(MAX_SEQS, np.int32)
    qlens[:num_seqs] = W
    qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    groups = np.arange(1, 1 + MAX_SEQS * W, dtype=np.int32).reshape(MAX_SEQS, W)
    si = groups[:, 0].copy()
    sl = np.where(qlens > 0, 1000 + qlens, 0).astype(np.int32)
    ro = rng.integers(0, W, MAX_SEQS).astype(np.int32)
    dist = np.array([num_seqs, num_seqs, num_seqs], np.int32)
    return (
        jnp.asarray(rng.normal(0, 1, (tokens, 1, DIM)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, (tokens, N_V)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, (tokens, N_V)), jnp.float32),
        jnp.asarray(rng.normal(0, 0.5, (KS, 1, DIM)), jnp.float32),
        jnp.asarray(np.log(rng.uniform(1, 16, N_V)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, N_V), jnp.bfloat16),
        jnp.asarray(qsl),
        jnp.asarray(si),
        jnp.asarray(dist),
        jnp.asarray(sl),
        jnp.asarray(ro),
        jnp.asarray(groups),
    )


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(NUM_BLOCKS, 512, 1, 4, 256), dtype=np.uint8)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    del raw
    tokens = MAX_SEQS * W  # the 256-row decode bucket
    real_get_tile_sizes = tiling.get_tile_sizes
    rows = []
    cases = []
    for mode, plan in (("select", SELECT_PLAN), ("log", LOG_PLAN)):
        for tile in ("dyn", 4):
            for num_seqs in (0, 8, 16, 24, 32, 48):
                cases.append((mode, plan, tile, num_seqs, True, False))
        for num_seqs in (0, 24, 48):
            cases.append((mode, plan, "dyn", num_seqs, False, False))
            cases.append((mode, plan, "dyn", num_seqs, False, True))
    for mode, plan, tile, num_seqs, batched_only, zero_out in cases:
        if tile == "dyn":
            tiling.get_tile_sizes = real_get_tile_sizes
        else:
            tiling.get_tile_sizes = (
                lambda *a, _t=tile, **k: (_t, real_get_tile_sizes(*a, **k)[1])
            )
        run = _make_loop(plan, iters, batched_only, zero_out)
        inputs = _inputs(rng, num_seqs, tokens)
        pool = run(pool, inputs)  # compile and warm
        pool.block_until_ready()
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            pool = run(pool, inputs)
            pool.block_until_ready()
            best = min(best, time.perf_counter() - t0)
        us = best / iters * 1e6
        name = "batched" if batched_only else ("call+zero" if zero_out else "call")
        print(f"GDN_BENCH mode={mode} tile={tile} seqs={num_seqs} kind={name} us_per_call={us:.2f}", flush=True)
        rows.append((mode, tile, num_seqs, name, us))
    tiling.get_tile_sizes = real_get_tile_sizes
    for mode in ("select", "log"):
        for tile in ("dyn", 4):
            pts = [(n, us) for m, t, n, k, us in rows if m == mode and t == tile and k == "batched" and n > 0]
            tile_n = 8 if tile == "dyn" else tile
            x = np.array([n / tile_n for n, _ in pts])
            y = np.array([us for _, us in pts])
            slope, intercept = np.polyfit(x, y, 1)
            print(f"GDN_FIT mode={mode} tile={tile} intercept_us={intercept:.2f} us_per_tile={slope:.2f} "
                  f"us_per_request={slope / tile_n:.3f}", flush=True)


if __name__ == "__main__":
    main()
