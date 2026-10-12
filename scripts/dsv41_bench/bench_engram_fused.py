# SPDX-License-Identifier: Apache-2.0
"""Compare packed Engram fusion with the unfused JAX numerical reference.

Run on one TPU device; gathered head shards are inputs to both variants.
The serving A/B separately measures CPU lookup, H2D and the TP collective.
"""

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.engram.kernel import (
    dequantize_rows,
    fused_engram,
    gated_residual,
)
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul


def make_inputs(tokens: int, tp: int, dim: int, width: int) -> tuple:
    rng = np.random.default_rng(20261011)
    streams = 4
    raw_bytes = (
        np.asarray(
            jnp.asarray(
                rng.normal(size=(tp, tokens * tp, width // tp)), jnp.float32
            ).astype(jnp.float8_e4m3fn)
        )
        .view(np.uint8)
        .copy()
    )
    scales = rng.integers(
        123, 129, size=(tp, tokens * tp, width // tp // 32), dtype=np.uint8
    )
    scales[:, ::7] = 255
    raw_bytes[:, ::7] = 127
    return (
        jnp.asarray(raw_bytes.view(np.int32)),
        jnp.asarray(scales.view(np.int32)),
        jnp.asarray(rng.normal(size=(tokens, streams, dim)), jnp.bfloat16),
        jnp.asarray(rng.normal(size=(width, (streams + 1) * dim)), jnp.float8_e4m3fn),
        jnp.asarray(rng.uniform(0.005, 0.03, size=((streams + 1) * dim,)), jnp.float32),
        jnp.asarray(rng.normal(size=(streams, dim)), jnp.bfloat16),
        jnp.asarray(rng.normal(size=(streams, dim)), jnp.bfloat16),
        jnp.arange(tokens) % 3 != 0,
    )


def reference(raw, scales, hidden, weight, weight_scale, q, k, mask, *, rank):
    tokens = hidden.shape[0]
    start = rank * tokens
    embedded = (
        dequantize_rows(
            raw[:, start : start + tokens], scales[:, start : start + tokens]
        )
        .transpose(1, 0, 2)
        .reshape(tokens, weight.shape[0])
    )
    projection = xla_quantized_matmul(embedded, weight, weight_scale)
    return gated_residual(hidden, projection, q, k, mask, 1e-20, 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interpret", action="store_true")
    args = parser.parse_args()
    root = args.output
    out = root / "rank-0"
    for subdir in ("benchmark", "compiler/hlo", "profiling/xprof"):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "operator-optimization",
                "operator_family": "engram",
                "operator_name": "engram_offload_fused",
                "dimensions": {"tp_size": 8, "scope": "after_packed_all_gather"},
            },
            indent=2,
        )
    )
    cases = (
        [(32, 2, 128, 512)]
        if args.interpret
        else [
            (256, 8, 5120, 6144),
            (1024, 8, 5120, 6144),
        ]
    )
    with jax.default_device(jax.devices()[0]):
        for tokens, tp, dim, width in cases:
            inputs = make_inputs(tokens, tp, dim, width)
            rank = tp - 1
            baseline = jax.jit(functools.partial(reference, rank=rank))
            candidate = jax.jit(
                functools.partial(
                    fused_engram,
                    rank=rank,
                    eps=1e-20,
                    clamp_value=1e-6,
                    interpret=args.interpret,
                )
            )
            results = {}
            for variant, fn in (("baseline", baseline), ("candidate", candidate)):
                print(f"COMPILE {variant} tokens={tokens}", flush=True)
                start = time.perf_counter()
                lowered = fn.lower(*inputs)
                executable = lowered.compile()
                compile_s = time.perf_counter() - start
                compiler_path = out / "compiler/hlo" / f"{variant}-{tokens}"
                try:
                    compiler_path.with_suffix(".txt").write_text(executable.as_text())
                except jax.errors.JaxRuntimeError as error:
                    # The pinned libtpu may emit async-update HLO that this
                    # jaxlib cannot deserialize. Preserve the error and the
                    # pre-optimization IR, then measure the compiled program.
                    compiler_path.with_suffix(".inspection-error.txt").write_text(
                        str(error)
                    )
                    compiler_path.with_suffix(".stablehlo.mlir").write_text(
                        lowered.as_text()
                    )
                    print(f"HLO_INSPECTION_FALLBACK {variant} tokens={tokens}: {error}")
                for _ in range(3):
                    output = executable(*inputs).block_until_ready()
                samples = []
                for _ in range(20):
                    start = time.perf_counter_ns()
                    output = executable(*inputs).block_until_ready()
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                results[variant] = np.asarray(output.astype(jnp.float32))
                record = {
                    "variant": variant,
                    "tokens": tokens,
                    "tp_size": tp,
                    "hidden_size": dim,
                    "input_width": width,
                    "dtype": "fp8",
                    "latency_ms": float(np.median(samples)),
                    "compile_time_s": compile_s,
                    "samples_ms": samples,
                }
                print(json.dumps(record), flush=True)
                with (out / "benchmark/metrics.jsonl").open("a") as file:
                    file.write(json.dumps(record) + "\n")
                if not args.interpret:
                    from torch_tpu._internal import profiler

                    # Use the wheel's profiler ABI, as the serving harness does.
                    with profiler.profile(
                        activities=[
                            profiler.ProfilerActivity.CPU,
                            profiler.ProfilerActivity.TPU,
                        ],
                        on_trace_ready=profiler.xprof_trace_handler(
                            dir_name=str(out / f"profiling/xprof/{variant}-{tokens}")
                        ),
                    ):
                        for _ in range(10):
                            executable(*inputs).block_until_ready()
            delta = results["candidate"] - results["baseline"]
            check = {
                "tokens": tokens,
                "max_abs_error": float(np.max(np.abs(delta))),
                "relative_l2": float(
                    np.linalg.norm(delta) / np.linalg.norm(results["baseline"])
                ),
            }
            (out / f"benchmark/correctness-{tokens}.json").write_text(json.dumps(check))
            print("NUMERICS", json.dumps(check), flush=True)
            assert np.isfinite(results["candidate"]).all()
            assert check["relative_l2"] < 0.005, check
            np.testing.assert_array_equal(
                results["candidate"][::3],
                np.asarray(inputs[2].astype(jnp.float32))[::3],
            )


if __name__ == "__main__":
    main()
