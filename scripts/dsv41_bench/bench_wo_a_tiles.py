# SPDX-License-Identifier: Apache-2.0
"""Check WO_A row-chunk tuning against the previous 96-row decode kernel."""

import argparse
import functools
import json
import random
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v4.o_projection import wo_a_projection


@functools.partial(jax.jit, static_argnames=("seed",))
def make_data(seed):
    keys = jax.random.split(jax.random.key(seed), 3)
    return (
        jax.random.normal(keys[0], (288, 64, 512), dtype=jnp.bfloat16)
        * jnp.bfloat16(0.2),
        jax.random.uniform(keys[1], (4096, 8192), minval=-1.0, maxval=1.0).astype(
            jnp.float8_e4m3fn
        ),
        jax.random.uniform(keys[2], (8192,), minval=0.01, maxval=0.03),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1033)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "tpu":
        parser.error("TPU is required for timing")
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = jax.block_until_ready(make_data(args.seed))
    executables, checks = {}, {}
    reference = None
    for name, sub_t in (("reference96", 96), ("auto", None), ("whole288", 288)):
        fn = jax.jit(
            functools.partial(
                wo_a_projection,
                cos_sin=None,
                tile_t=288,
                tile_r=1024,
                sub_t=sub_t,
                quantize_activations=True,
            )
        )
        try:
            executable = fn.lower(*inputs).compile()
            actual = np.asarray(executable(*inputs)).astype(np.float32)
            if reference is None:
                reference = actual
            same = actual.view(np.uint32) == reference.view(np.uint32)
            checks[name] = {
                "bit_exact": bool(np.all(same)),
                "different_bits": int(np.count_nonzero(~same)),
                "max_abs": float(np.max(np.abs(actual - reference))),
                "memory": str(executable.memory_analysis()),
            }
            (args.out / (name + ".hlo.txt")).write_text(executable.as_text())
            if checks[name]["bit_exact"]:
                executables[name] = executable
                for _ in range(5):
                    jax.block_until_ready(executable(*inputs))
        except Exception as error:
            if name == "reference96":
                raise
            checks[name] = {"error": str(error)[:1500]}
        print(name, json.dumps(checks[name]), flush=True)
    samples = {name: [] for name in executables}
    rng = random.Random(args.seed)
    for _ in range(41):
        names = list(executables)
        rng.shuffle(names)
        for name in names:
            start = time.perf_counter_ns()
            jax.block_until_ready(executables[name](*inputs))
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    results = {
        "seed": args.seed,
        "device_kind": jax.devices()[0].device_kind,
        "checks": checks,
        "timings": {
            name: {"median_ms": statistics.median(values), "samples_ms": values}
            for name, values in samples.items()
        },
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile")):
            for _ in range(5):
                for name, executable in executables.items():
                    with jax.profiler.TraceAnnotation(name):
                        jax.block_until_ready(executable(*inputs))
    assert all(checks[name].get("bit_exact") for name in checks), checks


if __name__ == "__main__":
    main()
