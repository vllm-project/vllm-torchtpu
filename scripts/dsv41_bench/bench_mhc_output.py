# SPDX-License-Identifier: Apache-2.0
"""Compare mHC output placement, including its following FP8 input projection.

Pass the baseline kernel source explicitly. Numerical checks include all mHC
outputs, quantized activations, scales and projection outputs. Timings retain
the BF16 value used by the compressor and other attention consumers.
"""

import argparse
import functools
import hashlib
import importlib.util
import json
import random
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.deepseek_v41.mhc import shifted_post_pre_rms as candidate
from vllm_torchtpu.kernels.quantized_matmul.util import quantize_tensor


@functools.partial(jax.jit, static_argnames=("seed", "tokens"))
def _make_data(seed: int, tokens: int) -> tuple[jax.Array, ...]:
    keys = jax.random.split(jax.random.key(seed), 9)
    streams, hidden = 4, 5120
    return (
        jax.random.normal(keys[0], (tokens, hidden), dtype=jnp.bfloat16),
        jax.random.normal(keys[1], (tokens, streams * hidden), dtype=jnp.bfloat16),
        jax.nn.sigmoid(jax.random.normal(keys[2], (tokens, streams))) * 2,
        jax.nn.softmax(jax.random.normal(keys[3], (tokens, streams, streams)), axis=-1),
        candidate.pack_mix_weight(
            jax.random.normal(keys[4], (24, streams * hidden))
            / (streams * hidden) ** 0.5
        ),
        jnp.full((3,), 0.5, jnp.float32),
        jnp.zeros((24,), jnp.float32),
        jax.nn.sigmoid(jax.random.normal(keys[5], (tokens, streams))),
        (1 + 0.1 * jax.random.normal(keys[6], (hidden,))).astype(jnp.bfloat16),
        (jax.random.normal(keys[7], (hidden, 1792)) * 40).astype(jnp.float8_e4m3fn),
        jax.random.uniform(keys[8], (1792,), minval=0.002, maxval=0.02),
    )


def _forward(module, *values, expose_quantization=False):
    result = module.shifted_post_pre_rms(*values[:8], norm_weight=values[8])
    quantized, scale = quantize_tensor(result[3], values[9].dtype)
    dot = jax.lax.dot_general(
        quantized,
        values[9],
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    projection = (dot * scale * values[10][None, :]).astype(result[3].dtype)
    extra = (quantized, scale) if expose_quantization else ()
    return (*result, *extra, projection)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1045)
    parser.add_argument("--tokens", type=int, default=288)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if jax.default_backend() != "tpu":
        parser.error("TPU is required for this hardware-specific kernel")
    args.out.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location(
        "mhc_output_reference", args.reference_source
    )
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    data = jax.block_until_ready(_make_data(args.seed, args.tokens))
    expected = jax.tree.map(
        lambda x: np.asarray(x).astype(np.float32),
        jax.jit(functools.partial(_forward, reference, expose_quantization=True))(
            *data
        ),
    )
    checks, executables, memory = {}, {}, {}
    for name, module in (("reference", reference), ("candidate", candidate)):
        check = jax.jit(functools.partial(_forward, module, expose_quantization=True))
        actual = jax.tree.map(lambda x: np.asarray(x).astype(np.float32), check(*data))
        differences = [
            int(np.count_nonzero(x.view(np.uint32) != y.view(np.uint32)))
            for x, y in zip(expected, actual, strict=True)
        ]
        checks[name] = {
            "bit_exact": not any(differences),
            "different_bits": differences,
        }
        (args.out / "checks.json").write_text(json.dumps(checks, indent=2))
        assert checks[name]["bit_exact"], checks[name]
        executable = jax.jit(functools.partial(_forward, module)).lower(*data).compile()
        (args.out / f"{name}.hlo.txt").write_text(executable.as_text())
        stats = executable.memory_analysis()
        memory[name] = {
            key: getattr(stats, key)
            for key in (
                "generated_code_size_in_bytes",
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
            )
        }
        executables[name] = executable
        for _ in range(5):
            jax.block_until_ready(executable(*data))
    samples = {name: [] for name in executables}
    rng = random.Random(args.seed)
    for _ in range(31):
        order = list(executables)
        rng.shuffle(order)
        for name in order:
            start = time.perf_counter_ns()
            jax.block_until_ready(executables[name](*data))
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    result = {
        "seed": args.seed,
        "tokens": args.tokens,
        "reference_sha256": hashlib.sha256(
            args.reference_source.read_bytes()
        ).hexdigest(),
        "checks": checks,
        "memory": memory,
        "timings": {
            name: {"median_ms": statistics.median(times), "samples_ms": times}
            for name, times in samples.items()
        },
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": checks, "memory": memory}), flush=True)
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile")):
            for _ in range(5):
                for name, executable in executables.items():
                    with jax.profiler.TraceAnnotation(name):
                        jax.block_until_ready(executable(*data))


if __name__ == "__main__":
    main()
