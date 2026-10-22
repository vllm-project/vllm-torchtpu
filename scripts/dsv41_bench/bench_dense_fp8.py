# SPDX-License-Identifier: Apache-2.0
"""Compare experimental Pallas FP8 GEMMs with the serving XLA implementation.

This benchmark does not change serving dispatch. Only bit-exact candidates are
included in timings; failed numerical checks and compilation errors are retained.
Shapes come from the 7b971e97 main-model HLO in Falcon exp-qg29pd6unq.
"""

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
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)


def _pallas_matmul(
    x, w, scale, *, batch=288, columns=1024, fused_quant=True, interpret=False
):
    m, k = x.shape
    n = w.shape[1]
    assert m % batch == 0 and n % columns == 0
    if fused_quant:
        inputs = (x, w, scale)
    else:
        xq, xs = quantize_tensor(x, w.dtype)
        inputs = (xq, w, scale, xs)

    def kernel(x_ref, w_ref, scale_ref, *refs):
        if fused_quant:
            out_ref, xq_ref, xs_ref = refs

            @pl.when(pl.program_id(1) == 0)
            def quantize():
                # Keep exactly the reference's scale, clipping, and operation
                # order. TPU checks still decide whether compilation preserves it.
                q, s = quantize_tensor(x_ref[...], w.dtype)
                xq_ref[...] = q
                xs_ref[...] = s

            q, s = xq_ref[...], xs_ref[...]
        else:
            xs_ref, out_ref = refs
            q, s = x_ref[...], xs_ref[...]
        acc = jax.lax.dot_general(
            q,
            w_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out_ref[...] = (acc * s * scale_ref[...][None, :]).astype(x.dtype)

    specs = [
        pl.BlockSpec((batch, k), lambda b, o: (b, 0)),
        pl.BlockSpec((k, columns), lambda b, o: (0, o)),
        pl.BlockSpec((columns,), lambda b, o: (o,)),
    ]
    if not fused_quant:
        specs.append(pl.BlockSpec((batch, 1), lambda b, o: (b, 0)))
    return pl.pallas_call(
        kernel,
        grid=(m // batch, n // columns),
        in_specs=specs,
        out_specs=pl.BlockSpec((batch, columns), lambda b, o: (b, o)),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        scratch_shapes=(
            (pltpu.VMEM((batch, k), w.dtype), pltpu.VMEM((batch, 1), jnp.float32))
            if fused_quant
            else ()
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=48 * 1024 * 1024,
        ),
        name="decode_dense_fp8",
        interpret=interpret,
    )(*inputs)


@functools.partial(jax.jit, static_argnames=("k", "n", "seed", "rows"))
def _make_data(k, n, seed, rows=288):
    keys = jax.random.split(jax.random.key(seed + k + n), 3)
    return (
        jax.random.normal(keys[0], (rows, k), dtype=jnp.bfloat16),
        (jax.random.normal(keys[1], (k, n)) * 40).astype(jnp.float8_e4m3fn),
        jax.random.uniform(keys[2], (n,), minval=0.002, maxval=0.02),
    )


def _interpret_check(out, seed):
    args = _make_data(128, 512, seed, rows=32)
    reference = np.asarray(xla_quantized_matmul(*args)).astype(np.float32)
    checks = {}
    for fused in (False, True):
        actual = np.asarray(
            _pallas_matmul(
                *args, batch=16, columns=128, fused_quant=fused, interpret=True
            )
        ).astype(np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
        checks[str(fused)] = {"bit_exact": True}
    (out / "interpret-check.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1028)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--interpret-check", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    if a.interpret_check:
        _interpret_check(a.out, a.seed)
        return
    if jax.default_backend() != "tpu":
        p.error("TPU is required for timing; use --interpret-check for the CPU check")
    executables, checks = {}, {}
    # Shapes observed in the compiled main model, not inferred from another model.
    for k, n in [(5120, 1792), (1280, 32768), (8192, 5120), (5120, 4608), (2304, 5120)]:
        args = jax.block_until_ready(_make_data(k, n, a.seed))
        reference = np.asarray(jax.jit(xla_quantized_matmul)(*args)).astype(np.float32)
        variants = [("reference", xla_quantized_matmul)]
        for fused in [False, True]:
            for batch in [96, 288]:
                for columns in [256, 512, 1024, 2048]:
                    if n % columns:
                        continue
                    name = f"{'fused' if fused else 'prequant'}_m{batch}_n{columns}"
                    variants.append(
                        (
                            name,
                            functools.partial(
                                _pallas_matmul,
                                batch=batch,
                                columns=columns,
                                fused_quant=fused,
                            ),
                        )
                    )
        for variant, fn in variants:
            name = f"k{k}_n{n}_{variant}"
            try:
                e = jax.jit(fn).lower(*args).compile()
                actual = np.asarray(e(*args)).astype(np.float32)
                same_bits = actual.view(np.uint32) == reference.view(np.uint32)
                exact = bool(np.all(same_bits))
                checks[name] = {
                    "bit_exact": exact,
                    "max_abs": float(np.max(np.abs(actual - reference))),
                    "rel_l2": float(
                        np.linalg.norm(actual - reference) / np.linalg.norm(reference)
                    ),
                    "different_elements": int(np.count_nonzero(~same_bits)),
                }
                (a.out / (name + ".hlo.txt")).write_text(e.as_text())
                if exact:
                    executables[name] = (e, args)
                    for _ in range(4):
                        jax.block_until_ready(e(*args))
                print("CHECK", name, json.dumps(checks[name]), flush=True)
            except Exception as error:
                if variant == "reference":
                    raise
                checks[name] = {"error": str(error)[:1500]}
                print("ERROR", name, str(error)[:400], flush=True)
        (a.out / "checks.json").write_text(json.dumps(checks, indent=2))
    samples = {name: [] for name in executables}
    rng = random.Random(a.seed)
    for _ in range(31):
        names = list(executables)
        rng.shuffle(names)
        for name in names:
            e, args = executables[name]
            start = time.perf_counter_ns()
            jax.block_until_ready(e(*args))
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    (a.out / "results.json").write_text(
        json.dumps(
            {
                "checks": checks,
                "timings": {
                    k: {"median_ms": statistics.median(v), "samples_ms": v}
                    for k, v in samples.items()
                },
            },
            indent=2,
        )
    )
    if a.profile:
        with jax.profiler.trace(str(a.out / "profile")):
            for _ in range(5):
                for name, (e, args) in executables.items():
                    with jax.profiler.TraceAnnotation(name):
                        jax.block_until_ready(e(*args))


if __name__ == "__main__":
    main()
