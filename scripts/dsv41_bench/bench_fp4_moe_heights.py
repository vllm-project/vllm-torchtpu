# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare FP4 MoE height ladders on the served EP8 decode geometry.

Each configuration compiles independently, verifies every output against the
four-rung reference, and runs in rotating order. Host timings include dispatch;
the optional XProf capture measures device execution.
"""

import argparse
import functools
import importlib.metadata
import json
import os
import random
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import host, kernel, layer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1020)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--local-tokens", type=int, default=288)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.local_tokens < 1:
        parser.error("repeats and local-tokens must be positive")
    if args.experts < 8 or args.experts % 8 or not 1 <= args.topk <= args.experts:
        parser.error("experts must divide over EP8 and topk must be in [1, experts]")
    if len(jax.devices()) < 8:
        parser.error("this benchmark requires eight TPU devices")

    args.out.mkdir(parents=True, exist_ok=False)
    mesh = Mesh(np.asarray(jax.devices()[:8]), (kernel.AXIS,))
    hidden, inter, active, qb = 5120, 2560, 2304, 512
    experts = args.experts // mesh.size
    tokens = args.local_tokens * mesh.size
    spec = P(kernel.AXIS)
    sharding = NamedSharding(mesh, spec)

    @functools.partial(jax.jit, out_shardings=(sharding,) * 4)
    @functools.partial(
        jax.shard_map, mesh=mesh, in_specs=(), out_specs=(spec,) * 4, check_vma=False
    )
    def make_weights():
        rank = lax.axis_index(kernel.AXIS)
        keys = jax.random.split(jax.random.fold_in(jax.random.key(args.seed), rank), 4)
        outputs = []
        shapes = ((experts, hidden, 2 * inter), (experts, inter, hidden))
        for index, (key, shape) in enumerate(zip(keys[:2], shapes)):
            values = jax.random.randint(key, shape, -12, 13).astype(jnp.float32) / 2
            if index == 0:
                cols = jnp.arange(2 * inter)
                valid = (cols < active) | ((cols >= inter) & (cols < inter + active))
                values = jnp.where(valid[None, None, :], values, 0)
            else:
                values = jnp.where(jnp.arange(inter)[None, :, None] < active, values, 0)
            outputs.append(values.astype(jnp.float4_e2m1fn))
        scale_shapes = (
            (experts, hidden // qb, 2 * inter),
            (experts, inter // qb, hidden),
        )
        for key, shape in zip(keys[2:], scale_shapes):
            values = jax.random.uniform(key, shape, minval=0.005, maxval=0.025)
            rows = host.align_up(shape[1], host.U32_SUBLANE_TILE)
            values = jnp.pad(values, ((0, 0), (0, rows - shape[1]), (0, 0)))
            outputs.append(values.reshape(experts * rows, shape[2]))
        return tuple(outputs)

    weights = jax.block_until_ready(make_weights())
    rng = np.random.default_rng(args.seed + 1)
    x = jax.device_put(
        (rng.normal(size=(tokens, hidden)) * 0.1).astype(jnp.bfloat16), sharding
    )
    gating = jax.device_put(
        rng.normal(size=(tokens, args.experts)).astype(np.float32), sharding
    )
    inputs = (x, *weights, gating)
    compiled, checks = {}, {}
    reference = None
    override = "MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE"
    saved_override = os.environ.get(override)
    try:
        for name, rungs in (("reference4", 4), ("automatic", None), ("rungs2", 2)):
            if rungs is None:
                os.environ.pop(override, None)
            else:
                os.environ[override] = str(rungs)
            kernel._BUILD_CACHE.clear()
            layer._LAYER_SM_CACHE.clear()

            def forward(*values):
                return layer.fused_ep_moe_v2(
                    *values,
                    mesh=mesh,
                    topk=args.topk,
                    renormalize=True,
                    capacity=128,
                    weight_format=host.WeightFormat.FP4,
                    rhs_qb=qb,
                    act_fn="silu_and_mul_with_clamp",
                    sharded_plan=True,
                    logical_intermediate_size=active,
                )

            started = time.monotonic()
            executable = jax.jit(forward).lower(*inputs).compile()
            actual = np.asarray(executable(*inputs)).astype(np.float32)
            if reference is None:
                reference = actual
            np.testing.assert_array_equal(
                actual.view(np.uint32), reference.view(np.uint32), err_msg=name
            )
            checks[name] = {
                "bit_exact": True,
                "compile_seconds": time.monotonic() - started,
            }
            compiled[name] = executable
            (args.out / f"{name}.hlo.txt").write_text(executable.as_text())
            for _ in range(5):
                jax.block_until_ready(executable(*inputs))
    finally:
        if saved_override is None:
            os.environ.pop(override, None)
        else:
            os.environ[override] = saved_override
        kernel._BUILD_CACHE.clear()
        layer._LAYER_SM_CACHE.clear()

    samples = {name: [] for name in compiled}
    order_rng = random.Random(args.seed)
    for _ in range(args.repeats):
        order = list(compiled)
        order_rng.shuffle(order)
        for name in order:
            started = time.perf_counter_ns()
            jax.block_until_ready(compiled[name](*inputs))
            samples[name].append((time.perf_counter_ns() - started) / 1e6)
    results = {
        "seed": args.seed,
        "local_tokens": args.local_tokens,
        "experts": args.experts,
        "topk": args.topk,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "libtpu")
        },
        "checks": checks,
        "timing": "synchronized_host_latency",
        "timings": {
            name: {"median_ms": statistics.median(values), "samples_ms": values}
            for name, values in samples.items()
        },
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results), flush=True)
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile")):
            for _ in range(5):
                for name, executable in compiled.items():
                    with jax.profiler.TraceAnnotation(name):
                        jax.block_until_ready(executable(*inputs))


if __name__ == "__main__":
    main()
