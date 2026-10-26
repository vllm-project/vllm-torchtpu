# SPDX-License-Identifier: Apache-2.0
"""Compare output-projection candidates with this checkout's JAX references."""

import argparse
import importlib.metadata
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

import jax
import ml_dtypes
import numpy as np
from jax._src.profiler import ProfileData

from vllm_torchtpu.kernels.deepseek_v4.o_projection import wo_a_projection
from vllm_torchtpu.kernels.quantized_matmul.buffered import buffered_matmul
from vllm_torchtpu.kernels.quantized_matmul.tuned_block_sizes import get_tpu_version
from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=(240, 288), default=288)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4919)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.samples > 0
    # CI images exclude .git; run_in_docker.sh forwards the image's build SHA.
    git_sha = (
        os.environ.get("BUILDKITE_COMMIT")
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    )
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise ValueError("Benchmark source requires a full Git commit SHA")
    assert get_tpu_version() == 7, "These candidates are qualified on TPU v7."
    args.output.mkdir(parents=True, exist_ok=True)
    device = jax.devices()[0]
    rng = np.random.default_rng(args.seed)

    def random(shape, dtype):
        return jax.device_put(rng.standard_normal(shape).astype(dtype), device)

    tokens = args.tokens
    oa_inputs = (
        random((tokens, 64, 512), ml_dtypes.bfloat16),
        random((4096, 8192), ml_dtypes.float8_e4m3fn),
        jax.device_put(rng.uniform(0.005, 0.03, 8192).astype(np.float32), device),
    )
    ob_inputs = (
        random((tokens, 8192), ml_dtypes.bfloat16),
        random((8192, 5120), ml_dtypes.float8_e4m3fn),
        jax.device_put(rng.uniform(0.005, 0.03, 5120).astype(np.float32), device),
    )
    chain_inputs = (*oa_inputs, *ob_inputs[1:])

    def oa_reference(x, w, scale):
        # Explicit original sub-tiling bypasses automatic buffered dispatch.
        sub_t = next(n for n in range(128, 0, -1) if x.shape[0] % n == 0)
        return wo_a_projection(x, w, scale, None, sub_t=sub_t)

    def oa_candidate(x, w, scale):
        return wo_a_projection(x, w, scale, None)

    def chain_reference(x, wa, sa, wb, sb):
        return xla_quantized_matmul(oa_reference(x, wa, sa), wb, sb)

    def chain_candidate(x, wa, sa, wb, sb):
        return buffered_matmul(oa_candidate(x, wa, sa), wb, sb)

    cases = {
        "oa_reference": (oa_reference, oa_inputs),
        "oa_candidate": (oa_candidate, oa_inputs),
        "ob_reference": (xla_quantized_matmul, ob_inputs),
        "ob_candidate": (buffered_matmul, ob_inputs),
        "chain_reference": (chain_reference, chain_inputs),
        "chain_candidate": (chain_candidate, chain_inputs),
    }
    executables, outputs = {}, {}
    for name, (function, inputs) in cases.items():

        def named(*values, _function=function):
            return _function(*values)

        named.__name__ = name
        lowered = jax.jit(named).lower(*inputs)
        (args.output / f"{name}.stablehlo.mlir").write_text(lowered.as_text())
        executable = lowered.compile()
        (args.output / f"{name}.hlo.txt").write_text(executable.as_text())
        actual = np.asarray(executable(*inputs))
        if name.endswith("candidate"):
            expected = outputs[name.replace("candidate", "reference")]
            np.testing.assert_array_equal(
                actual.view(np.uint16), expected.view(np.uint16)
            )
        assert np.isfinite(actual.astype(np.float32)).all()
        outputs[name] = actual
        for _ in range(8):
            jax.block_until_ready(executable(*inputs))
        executables[name] = (executable, inputs)

    with jax.profiler.trace(str(args.output / "profile")):
        for repeat in range(args.samples):
            names = list(executables)
            if repeat % 2:
                names.reverse()
            for name in names:
                executable, inputs = executables[name]
                with jax.profiler.TraceAnnotation(name, step_num=repeat):
                    jax.block_until_ready(executable(*inputs))

    profile = next((args.output / "profile").rglob("*.xplane.pb"))
    data = ProfileData.from_file(str(profile))
    plane = next(p for p in data.planes if p.name == "/device:TPU:0")
    modules = next(line for line in plane.lines if line.name == "XLA Modules")
    results = {}
    for name in cases:
        samples = [
            event.duration_ns / 1000
            for event in modules.events
            if event.name == f"jit_{name}" or event.name.startswith(f"jit_{name}(")
        ]
        assert len(samples) == args.samples, (name, len(samples), args.samples)
        results[name] = {
            "median_us": statistics.median(samples),
            "samples_us": samples,
            "bitwise_equal": True,
        }
    result = {
        "scope": "complete XLA module device latency; not serving E2E",
        "tokens": tokens,
        "seed": args.seed,
        "git_sha": git_sha,
        "device": str(device),
        "libtpu_flags": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "runtime": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "libtpu")
        },
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({name: value["median_us"] for name, value in results.items()}))


if __name__ == "__main__":
    main()
