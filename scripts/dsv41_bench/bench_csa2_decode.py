# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark CSA2 decode: 48 requests x 6 queries, with pre-gathered main KV.

Run on a TPU in the project environment:
    python scripts/dsv41_bench/bench_csa2_decode.py --out /tmp/csa2-decode --profile

For tracing and compiler inspection, add:
    --trace-mode TRACE_COMPUTE --dump-llo /tmp/csa2-compiler
Profiling enables libtpu custom-call region tracing for the kernel's named scopes.
Compiler dumps are in /tmp/csa2-compiler/rank0, indexed by compiler-dumps.json
in the output directory. Search those files for csa2_swa_csa_inverse_rope.

Q is reshaped from [requests, queries, heads, 512] to [tokens, heads, 512].
Request boundaries and consecutive positions preserve six-token causal decode.
Only the compiled CSA2 call is timed, including host dispatch/synchronization;
FP8 encoding, KV gathering, compilation and warmup are outside the samples.
"""

import argparse
import functools
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--heads", type=int, choices=range(8, 129, 8), default=64)
    parser.add_argument("--context-length", type=int, default=98304)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--trace-mode",
        type=str.upper,
        choices=("TRACE_COMPUTE", "TRACE_ONLY_HOST"),
        default=None,
        help="libtpu trace mode; implies --profile (default: TRACE_COMPUTE)",
    )
    parser.add_argument(
        "--dump-llo",
        type=Path,
        help="dump HLO/LLO into DIR/rank0 and disable compilation caches",
    )
    args = parser.parse_args()
    if min(args.requests, args.queries, args.iterations) < 1:
        parser.error("--requests, --queries and --iterations must be positive")
    if args.context_length < 1024:
        parser.error("--context-length must be >= 1024 for 512 valid compressed KV")

    if args.dump_llo is not None:
        from common import _llo_env

        dump_env = dict(os.environ)
        _llo_env(args.dump_llo.resolve(), 0, dump_env)
        dump_env["LIBTPU_INIT_ARGS"] += (
            " --xla_jf_dump_to_filter=module=.*csa2_swa_csa_inverse_rope.*"
        )
        os.environ.update(dump_env)

    if args.profile or args.trace_mode is not None:
        os.environ["LIBTPU_INIT_ARGS"] = (
            os.environ.get("LIBTPU_INIT_ARGS", "")
            + " --xla_enable_custom_call_region_trace=false"
            + " --xla_xprof_enable_custom_call_tracing=true"
        ).strip()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal.profiler import TpuProfilerConfig

    from vllm_torchtpu.kernels.deepseek_v41.csa2 import csa2
    from vllm_torchtpu.kernels.deepseek_v41.csa2_cache import encode_csa2_kv

    if args.dump_llo is not None:
        jax.config.update("jax_enable_compilation_cache", False)

    # Match the serving worker's client initialization order.
    torch.empty((1,), device="tpu").cpu()
    if jax.default_backend() != "tpu":
        raise RuntimeError("Run this benchmark in the project environment on a TPU")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    tokens = args.requests * args.queries
    dim, selected, page_size = 512, 512, 32

    def random_bf16(shape: tuple[int, ...]) -> jax.Array:
        return jnp.asarray(
            rng.standard_normal(shape, dtype=np.float32) * 0.2, dtype=jnp.bfloat16
        )

    q = random_bf16((args.requests, args.queries, args.heads, dim)).reshape(
        tokens, args.heads, dim
    )
    starts = jnp.arange(args.requests + 1, dtype=jnp.int32) * args.queries
    positions = jnp.tile(
        args.context_length + jnp.arange(args.queries, dtype=jnp.int32),
        args.requests,
    )
    # Independent request caches; all six queries reuse the same selected rows.
    words, scales = encode_csa2_kv(random_bf16((args.requests * selected, dim)))
    records = jnp.repeat(
        jnp.arange(args.requests * selected).reshape(args.requests, selected),
        args.queries,
        axis=0,
    )
    main_kv, main_scales = words[records], scales[records]

    first_page_position = (args.context_length - 127) // page_size * page_size
    columns = (
        args.context_length + args.queries - first_page_position + page_size - 1
    ) // page_size
    pages = 1 + args.requests * columns
    swa = random_bf16((pages, page_size, dim)).at[0].set(0)
    bits = jax.lax.bitcast_convert_type(swa, jnp.uint16).astype(jnp.uint32)
    bits = bits.reshape(pages, page_size, 2, 2, 128)
    packed = bits[:, :, :, 0] | (bits[:, :, :, 1] << 16)
    swa_words = jax.lax.bitcast_convert_type(packed, jnp.int32).transpose(0, 2, 1, 3)
    metadata = jnp.concatenate(
        (
            jnp.full((args.requests, 1), first_page_position, jnp.int32),
            jnp.zeros((args.requests, 1), jnp.int32),
            jnp.arange(1, pages, dtype=jnp.int32).reshape(args.requests, columns),
        ),
        axis=1,
    )
    angles = jnp.asarray(rng.standard_normal((tokens, 32)), jnp.float32)
    cos = jnp.repeat(jnp.cos(angles), 2, axis=-1)
    sin = jnp.repeat(jnp.sin(angles), 2, axis=-1)
    sin *= jnp.tile(jnp.array([1, -1], jnp.float32), 32)
    rope = jnp.stack(
        (
            jnp.concatenate((jnp.ones_like(cos), cos), axis=-1),
            jnp.concatenate((jnp.zeros_like(sin), sin), axis=-1),
        ),
        axis=1,
    )
    sink = jnp.asarray(rng.standard_normal(args.heads), jnp.float32)
    inputs = (q, main_kv, main_scales, swa_words, starts, positions, sink, rope)
    kwargs = dict(
        swa_metadata=metadata,
        main_lengths=jnp.full((tokens,), selected, jnp.int32),
    )
    jax.block_until_ready((inputs, kwargs))
    begin = time.perf_counter()
    lowered = jax.jit(functools.partial(csa2, bq=6)).lower(*inputs, **kwargs)
    executable = lowered.compile()
    compile_s = time.perf_counter() - begin
    (args.out / "csa2.stablehlo.txt").write_text(lowered.as_text())
    (args.out / "csa2.optimized-hlo.txt").write_text(executable.as_text())
    if args.dump_llo is not None:
        dump_dir = args.dump_llo.resolve() / "rank0"
        dump_files = sorted(str(path) for path in dump_dir.rglob("*") if path.is_file())
        (args.out / "compiler-dumps.json").write_text(
            json.dumps({"directory": str(dump_dir), "files": dump_files}, indent=2)
            + "\n"
        )
        print(f"HLO/LLO dumps: {dump_dir}; index: {args.out / 'compiler-dumps.json'}")
    for _ in range(5):
        output = jax.block_until_ready(executable(*inputs, **kwargs))
    assert output.shape == q.shape
    if not np.isfinite(np.asarray(output, dtype=np.float32)).all():
        raise RuntimeError("CSA2 produced non-finite output")

    samples_us = []
    for _ in range(args.iterations):
        begin_ns = time.perf_counter_ns()
        jax.block_until_ready(executable(*inputs, **kwargs))
        samples_us.append((time.perf_counter_ns() - begin_ns) / 1000)
    memory = executable.memory_analysis()
    report = dict(
        kernel="csa2",
        device=jax.devices()[0].device_kind,
        requests=args.requests,
        queries_per_request=args.queries,
        tokens=tokens,
        q_shape=list(q.shape),
        main_kv_shape=list(main_kv.shape),
        swa_shape=list(swa_words.shape),
        context_length=args.context_length,
        bq=6,
        causal=True,
        compile_s=compile_s,
        median_us=float(np.median(samples_us)),
        p90_us=float(np.percentile(samples_us, 90)),
        min_us=min(samples_us),
        samples_us=samples_us,
        temp_bytes=memory.temp_size_in_bytes,
        input_bytes=memory.argument_size_in_bytes,
        output_bytes=memory.output_size_in_bytes,
        timing="host wall time with synchronization; excludes input preparation",
        trace_mode=(args.trace_mode or "TRACE_COMPUTE")
        if args.profile or args.trace_mode is not None
        else None,
    )
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if args.profile or args.trace_mode is not None:
        capture = str(args.out / "profile")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.PrivateUse1,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(capture),
            experimental_config=TpuProfilerConfig(
                run_dir=capture,
                experimental_options={
                    "tpu_trace_mode": args.trace_mode or "TRACE_COMPUTE"
                },
            ),
        ):
            for step in range(5):
                with torch.profiler.record_function(f"csa2_decode_{step}"):
                    jax.block_until_ready(executable(*inputs, **kwargs))


if __name__ == "__main__":
    main()
