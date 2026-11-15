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
"""Splits the pooled GDN v3 decode kernel's time at the served Flash geometry.

Part 1: device time per BATCHED call (wrapper included, SELECT mode as served)
for verify windows of 1, 2 and 4 tokens at 0 to 48 running, with the state
checkpoint stores as served, without their VMEM stores (store_state_region),
without their HBM writes (the state out-refs' DMAs), and without both.
Part 2: the empty call. An xprof capture of the empty BATCHED call and of 48
running at window 4 (per-op device time: the kernel event and the XLA ops
around it), and synthetic pallas calls that add the GDN kernel's operand set
piece by piece: launch, SMEM metadata, VMEM weights, HBM operands.

Each time is the mean of N calls run back to back inside one jitted loop.

usage: python tests/kernels/gdn/v3/bench_gdn_split.py [iters]
"""

import contextlib
import functools
import os
import re
import sys
import time

os.environ.setdefault("TPU_ENABLE_GDN_DYNAMIC_TILING", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

from vllm_torchtpu.kernels.gdn.v3 import config, memory_ref, vmem_ldst, wrapper  # noqa: E402
from vllm_torchtpu.kernels.gdn.v3.input_layout import Conv1DInput  # noqa: E402

N_KQ, N_V, D, DIM, KS = 2, 6, 128, 1280, 4
F32 = jnp.dtype(jnp.float32)
PLAN = config.StateSourcePlan(
    stride=1,
    conv=config.StateRegion(0, 1, 384, 16, F32, 1, 15),
    recurrent=config.StateRegion(0, 1, 0, 384, F32, 2, 768),
)
MAX_SEQS = 64
MAX_W = 4
NUM_BLOCKS = 1 + MAX_SEQS * MAX_W
TOKENS = MAX_SEQS * MAX_W  # the 256-row decode bucket
OUT_DIR = os.path.join(os.environ.get("CDK_OUTPUT_DIR", "/tmp/cdk_output"), "q38f-bench", "script")


@contextlib.contextmanager
def _stores(vmem: bool, hbm: bool):
    """Turns the state checkpoint stores off: their VMEM stores, their HBM writes."""
    saved = (
        vmem_ldst.store_state_region,
        memory_ref.ExternalStateOutBufferedRef.copy_out,
        memory_ref.ExternalStateOutBufferedRef.wait_out,
    )
    if not vmem:
        vmem_ldst.store_state_region = lambda *a, **k: None
    if not hbm:
        memory_ref.ExternalStateOutBufferedRef.copy_out = lambda self, *a, **k: None
        memory_ref.ExternalStateOutBufferedRef.wait_out = lambda self, *a, **k: None
    jax.clear_caches()
    try:
        yield
    finally:
        (
            vmem_ldst.store_state_region,
            memory_ref.ExternalStateOutBufferedRef.copy_out,
            memory_ref.ExternalStateOutBufferedRef.wait_out,
        ) = saved
        jax.clear_caches()


def _gdn_loop(window, iters):
    def call(pool, inputs):
        qkv, b, a, conv_w, a_log, dt_bias, qsl, si, dist, sl, ro, ck = inputs
        return wrapper.fused_conv1d_gdn(
            Conv1DInput(qkv, jnp.bfloat16), b, a, None, None, conv_w, None,
            a_log, dt_bias, qsl, si, dist, sl, ro, ck,
            n_kq=N_KQ, n_v=N_V, d_k=D, d_v=D, kernel_size=KS,
            state_source=pool, state_plan=PLAN, num_spec_tokens=window - 1,
            conv_weight_kmajor=True, batched_only=True, zero_initialize_out=False,
        )

    def step(_, carry):
        pool, inputs = carry
        pool, out = call(pool, inputs)
        qkv = inputs[0].at[0, 0, 0].add(out[0, 0].astype(jnp.float32) * 0)
        return pool, (qkv,) + inputs[1:]

    @functools.partial(jax.jit, donate_argnums=0)
    def run(pool, inputs):
        return jax.lax.fori_loop(0, iters, step, (pool, inputs))[0]

    return run, jax.jit(call)


def _inputs(rng, num_seqs, window):
    qlens = np.zeros(MAX_SEQS, np.int32)
    qlens[:num_seqs] = window
    qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
    groups = np.arange(1, 1 + MAX_SEQS * MAX_W, dtype=np.int32).reshape(MAX_SEQS, MAX_W)
    return (
        jnp.asarray(rng.normal(0, 1, (TOKENS, 1, DIM)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, (TOKENS, N_V)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, (TOKENS, N_V)), jnp.float32),
        jnp.asarray(rng.normal(0, 0.5, (KS, 1, DIM)), jnp.float32),
        jnp.asarray(np.log(rng.uniform(1, 16, N_V)), jnp.float32),
        jnp.asarray(rng.normal(0, 1, N_V), jnp.bfloat16),
        jnp.asarray(qsl),
        jnp.asarray(groups[:, 0].copy()),
        jnp.asarray(np.array([num_seqs] * 3, np.int32)),
        jnp.asarray(np.where(qlens > 0, 1000 + qlens, 0).astype(np.int32)),
        jnp.asarray(rng.integers(0, window, MAX_SEQS).astype(np.int32)),
        jnp.asarray(groups[:, :window].copy()),
    )


def _time(run, pool, args, iters):
    pool = run(pool, *args)
    pool.block_until_ready()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        pool = run(pool, *args)
        pool.block_until_ready()
        best = min(best, time.perf_counter() - t0)
    return pool, best / iters * 1e6


def _tile(call, pool, inputs):
    text = call.lower(pool, inputs).as_text()
    m = re.search(r"fused_conv1d_gdn_batched_b(\d+)_c(\d+)", text)
    return int(m.group(1)) if m else -1


def part1(pool, rng, iters):
    rows = []
    for window in (1, 2, 4):
        for vmem, hbm in ((True, True), (False, True), (True, False), (False, False)):
            with _stores(vmem, hbm):
                run, call = _gdn_loop(window, iters)
                tile = None
                pts = []
                for num_seqs in (0, 8, 16, 24, 32, 48):
                    inputs = _inputs(rng, num_seqs, window)
                    if tile is None and num_seqs:
                        tile = _tile(call, pool, inputs)
                    pool, us = _time(run, pool, (inputs,), iters)
                    pts.append((num_seqs, us))
                    print(f"GDN_SPLIT window={window} vmem_store={int(vmem)} hbm_store={int(hbm)} "
                          f"seqs={num_seqs} us_per_call={us:.2f}", flush=True)
                x = np.array([n for n, _ in pts if n])
                y = np.array([u for n, u in pts if n])
                slope, icpt = np.polyfit(x, y, 1)
                print(f"GDN_SPLIT_FIT window={window} vmem_store={int(vmem)} hbm_store={int(hbm)} tile={tile} "
                      f"empty_us={pts[0][1]:.2f} intercept_us={icpt:.2f} us_per_request={slope:.3f} "
                      f"us_per_tile={slope * tile:.2f} us_per_token={slope / window:.3f}", flush=True)
                rows.append((window, vmem, hbm, tile, pts[0][1], icpt, slope))
    return pool, rows


def part2_profile(pool, rng, iters):
    for name, num_seqs, window in (("empty_w4", 0, 4), ("seqs48_w4", 48, 4)):
        run, _ = _gdn_loop(window, iters)
        inputs = _inputs(rng, num_seqs, window)
        pool = run(pool, inputs)
        pool.block_until_ready()
        path = os.path.join(OUT_DIR, "xprof", name)
        os.makedirs(path, exist_ok=True)
        jax.profiler.start_trace(path)
        pool = run(pool, inputs)
        pool.block_until_ready()
        jax.profiler.stop_trace()
        print(f"GDN_PROFILE name={name} iters={iters} dir={path}", flush=True)
    return pool


def _synthetic(level, iters, pool):
    """A pallas call with the GDN BATCHED kernel's operands up to `level`:
    0 launch only, 1 + SMEM metadata, 2 + VMEM weights, 3 + HBM operands."""
    smem = [jnp.zeros((), jnp.int32)] + [jnp.zeros((MAX_SEQS,), jnp.int32)] * 8
    smem += [jnp.zeros((MAX_SEQS, MAX_W), jnp.int32)]
    vmem = [jnp.zeros((KS, 1, DIM), jnp.float32), jnp.zeros((N_V,), jnp.float32),
            jnp.zeros((N_V,), jnp.bfloat16)]
    hbm = [jnp.zeros((TOKENS, 1, DIM), jnp.float32), jnp.zeros((TOKENS, 1, 128), jnp.float32),
           jnp.zeros((TOKENS, 1, 128), jnp.float32)]
    ops = []
    specs = []
    if level >= 1:
        ops += smem
        specs += [pl.BlockSpec(memory_space=pltpu.SMEM)] * len(smem)
    if level >= 2:
        ops += vmem
        specs += [pl.BlockSpec(memory_space=pltpu.VMEM)] * len(vmem)
    if level >= 3:
        ops += hbm
        specs += [pl.BlockSpec(memory_space=pltpu.HBM)] * len(hbm)
    n_ops = len(ops)

    def kernel(*refs):
        *ins, pool_in, out_ref, pool_out = refs
        del pool_in, pool_out
        acc = jnp.zeros((8, 128), jnp.int32)
        if level >= 1:
            acc = acc + ins[0][...] + ins[1][0]
        out_ref[...] = acc

    def one(pool, ops):
        out, pool = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((8, 128), jnp.int32),
                       jax.ShapeDtypeStruct(pool.shape, pool.dtype)),
            in_specs=specs + [pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=(pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pltpu.HBM)),
            input_output_aliases={n_ops: 1},
        )(*ops, pool)
        return pool, out

    def step(_, carry):
        pool, ops, acc = carry
        pool, out = one(pool, ops)
        return pool, ops, acc + out[0, 0]

    @functools.partial(jax.jit, donate_argnums=0)
    def run(pool, ops):
        return jax.lax.fori_loop(0, iters, step, (pool, ops, jnp.int32(0)))[0]

    return _time(run, pool, (ops,), iters)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(NUM_BLOCKS, 512, 1, 4, 256), dtype=np.uint8)
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    del raw
    pool, _ = part1(pool, rng, iters)
    pool = part2_profile(pool, rng, iters)
    names = ("launch", "+smem_metadata", "+vmem_weights", "+hbm_operands")
    for level, name in enumerate(names):
        pool, us = _synthetic(level, iters, pool)
        print(f"GDN_EMPTY_SYNTH level={level} {name} us_per_call={us:.2f}", flush=True)


if __name__ == "__main__":
    main()
