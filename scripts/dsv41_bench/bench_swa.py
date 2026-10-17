# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare pure SWA against the XLA implementation on identical inputs."""

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch_tpu._internal.profiler import TpuProfilerConfig

from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention_kernels import (
    _attend_window,
    _attention_state,
    paged_attention_jax,
)
from vllm_torchtpu.models.vllm.deepseek_v41.v2.cache import _pack_bf16_words


def baseline(q, pool, records, sink):
    maximum, denominator, numerator = _attend_window(
        q, pool, records, _attention_state(q), q.shape[-1] ** -0.5
    )
    denominator += jnp.exp(sink[None] - maximum)
    return (numerator / denominator[..., None]).astype(q.dtype)


def candidate(q, pool, records, sink):
    return paged_attention_jax(
        q,
        pool,
        pool,
        records,
        jnp.empty((q.shape[0], 0), jnp.int32),
        sink,
        q.shape[-1] ** -0.5,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--heads", type=int, choices=(8, 16, 32, 64), default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Initialize the torch-tpu client before JAX, as the serving worker does.
    torch.empty((1,), device="tpu").cpu()
    reports = []
    for mode, tokens in (("decode", 288), ("prefill", 2048), ("mixed", 2048)):
        rng = np.random.default_rng(92)
        heads, dim, rows = args.heads, 512, 4096
        q = jnp.asarray(rng.normal(size=(tokens, heads, dim)), jnp.bfloat16)
        values = jnp.asarray(rng.normal(size=(rows, dim)), jnp.bfloat16)
        pool = _pack_bf16_words(values).reshape(-1, 128)
        ends = np.arange(tokens) + 127
        if mode == "decode":
            ends = rng.integers(127, rows, tokens)
        elif mode == "mixed":
            ends[1019:] = np.arange(tokens - 1019) + 3000
            ends = np.minimum(ends, rows - 1)
        records = jnp.asarray(ends[:, None] - 127 + np.arange(128)[None], jnp.int32)
        sink = jnp.asarray(rng.normal(size=heads), jnp.float32)
        inputs = (q, pool, records, sink)
        compiled = {}
        outputs = {}
        for name, function in (("baseline", baseline), ("candidate", candidate)):
            lowered = jax.jit(function).lower(*inputs)
            exe = lowered.compile()
            compiled[name] = exe
            (args.out / f"{mode}-{name}.stablehlo.txt").write_text(lowered.as_text())
            for _ in range(5):
                outputs[name] = jax.block_until_ready(exe(*inputs))
        np.testing.assert_allclose(
            np.asarray(outputs["candidate"], np.float32),
            np.asarray(outputs["baseline"], np.float32),
            atol=3e-3,
            rtol=1e-2,
        )
        samples = {name: [] for name in compiled}
        for iteration in range(args.iterations):
            order = list(compiled) if iteration % 2 else list(reversed(compiled))
            for name in order:
                start = time.perf_counter_ns()
                jax.block_until_ready(compiled[name](*inputs))
                samples[name].append((time.perf_counter_ns() - start) / 1000)
        medians = {name: statistics.median(sample) for name, sample in samples.items()}
        report = dict(
            mode=mode,
            tokens=tokens,
            heads=heads,
            median_us=medians,
            speedup=medians["baseline"] / medians["candidate"],
            samples_us=samples,
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
        (args.out / "results.json").write_text(json.dumps(reports, indent=2))
        if args.profile:
            capture = str(args.out / f"profile-{mode}")
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.PrivateUse1,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    capture, use_gzip=True
                ),
                experimental_config=TpuProfilerConfig(
                    run_dir=capture,
                    host_tracer_level=2,
                    device_tracer_level=1,
                    experimental_options={"tpu_trace_mode": "TRACE_COMPUTE"},
                ),
            ):
                for name, exe in compiled.items():
                    for step in range(5):
                        with torch.profiler.record_function(f"{name}_{step}"):
                            jax.block_until_ready(exe(*inputs))


if __name__ == "__main__":
    main()
