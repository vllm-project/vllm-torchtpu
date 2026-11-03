# SPDX-License-Identifier: Apache-2.0
"""Gate output trimming with original shifted mHC producer/consumer contracts."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/vllm_torchtpu/kernels/deepseek_v41/mhc/shifted_post_pre_rms.py"
spec = importlib.util.spec_from_file_location("mhc_original", SOURCE)
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)


def compact_gate(out, *, hbm_output=False, early_output=False):
    """Change only the output storage/trim of the original gate kernel."""
    node = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_pre_gates"
    )
    source = ast.unparse(node).replace("def _pre_gates(", "def compact_pre_gates(")
    source = source.replace(
        "def gate_kernel(",
        "out_block = 288\n    assert t == block == 384\n\n    def gate_kernel(",
    )
    source = source.replace(
        "pre_out[...] = pre_mix", "pre_out[...] = pre_mix[:, :out_block]"
    )
    source = source.replace(
        "post_out[...] = post_mix", "post_out[...] = post_mix[:, :out_block]"
    )
    source = source.replace(
        "comb_out[...] = comb_mix", "comb_out[...] = comb_mix[:, :, :out_block]"
    )
    source = source.replace("pl.BlockSpec((m, block)", "pl.BlockSpec((m, out_block)")
    source = source.replace(
        "pl.BlockSpec((m, m, block)", "pl.BlockSpec((m, m, out_block)"
    )
    source = source.replace(
        "((m, t), (m, t), (m, m, t))",
        "((m, out_block), (m, out_block), (m, m, out_block))",
    )
    source = source.replace(
        "compiler_params=pltpu.CompilerParams(",
        'name="dsv41_mhc_compact_gates", compiler_params=pltpu.CompilerParams(',
    )
    if hbm_output:
        source = source.replace(
            "pre_out, post_out, comb_out):",
            "pre_out, post_out, comb_out, pre_buf, post_buf, comb_buf, sems):",
        )
        for prefix in ("pre", "post", "comb"):
            source = source.replace(f"{prefix}_out[...] =", f"{prefix}_buf[...] =")
        source = source.replace(
            "    pre, post, comb = pl.pallas_call(",
            """        copies = [
            pltpu.make_async_copy(src, dst, sems.at[i])
            for i, (src, dst) in enumerate(
                ((pre_buf, pre_out), (post_buf, post_out), (comb_buf, comb_out))
            )
        ]
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    pre, post, comb = pl.pallas_call(""",
        )
        # The unparsed call uses one line; replace output specs only.
        source = source.replace(
            "out_specs=[pl.BlockSpec((m, out_block), lambda i: (0, i)), "
            "pl.BlockSpec((m, out_block), lambda i: (0, i)), "
            "pl.BlockSpec((m, m, out_block), lambda i: (0, 0, i))]",
            "out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 3",
        )
        source = source.replace(
            'name="dsv41_mhc_compact_gates",',
            'name="dsv41_mhc_compact_gates_hbm", '
            "scratch_shapes=[pltpu.VMEM((m, out_block), jnp.float32), "
            "pltpu.VMEM((m, out_block), jnp.float32), "
            "pltpu.VMEM((m, m, out_block), jnp.float32), "
            "pltpu.SemaphoreType.DMA((3,))],",
        )
        source = source.replace(
            "jax.ShapeDtypeStruct(shape, jnp.float32)",
            "pltpu.HBM(shape, jnp.float32)",
        )
        assert "pre_buf[...]" in source and "copy.start()" in source
        assert "out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 3" in source
    if early_output:
        assert hbm_output
        for prefix in ("pre", "post"):
            source = source.replace(
                f"        {prefix}_buf[...] = {prefix}_mix[:, :out_block]\n", ""
            )
        source = source.replace(
            "        comb_logits =",
            """        pre_buf[...] = pre_mix[:, :out_block]
        post_buf[...] = post_mix[:, :out_block]
        pre_copy = pltpu.make_async_copy(pre_buf, pre_out, sems.at[0])
        post_copy = pltpu.make_async_copy(post_buf, post_out, sems.at[1])
        pre_copy.start()
        post_copy.start()
        comb_logits =""",
        )
        begin = source.index("        copies = [")
        end = source.index("\n    pre, post, comb = pl.pallas_call(", begin)
        source = (
            source[:begin]
            + """        comb_copy = pltpu.make_async_copy(
            comb_buf, comb_out, sems.at[2]
        )
        comb_copy.start()
        pre_copy.wait()
        post_copy.wait()
        comb_copy.wait()
"""
            + source[end:]
        )
        source = source.replace(
            'name="dsv41_mhc_compact_gates_hbm"',
            'name="dsv41_mhc_compact_gates_hbm_early"',
        )
    (out / "candidate-gate.py").write_text(source + "\n")
    namespace = dict(vars(original))
    exec(compile(source, str(out / "candidate-gate.py"), "exec"), namespace)
    return namespace["compact_pre_gates"]


def fixture(kind, seed=42):
    rng = np.random.default_rng(seed)
    mix = rng.normal(0, 0.5, (24, 384)).astype(np.float32)
    square = rng.uniform(10000, 30000, (1, 384)).astype(np.float32)
    scale = rng.uniform(0.7, 1.3, 3).astype(np.float32)
    base = rng.normal(0, 0.5, 24).astype(np.float32)
    if kind == "zero":
        mix[:] = 0
        square[:] = 0
        base[:] = 0
    elif kind == "tiny":
        mix *= 1e-24
        square *= 1e-40
    elif kind == "saturated":
        mix *= 80
    elif kind == "large":
        mix *= 1e8
        square *= 1e16
    elif kind == "uniform":
        mix[:] = 0.5
        base[:] = 0.25
    return mix, square, scale, base


def reference(xs):
    mix, square, scale, base = (a.astype(np.float64) for a in xs)
    z = mix[:, :288] / np.sqrt(square[:, :288] / 20480 + 1e-20)
    pre_logits = z[:4] * scale[0] + base[:4, None]
    post_logits = z[4:8] * scale[1] + base[4:8, None]

    def sigmoid(x):
        result = np.empty_like(x)
        positive = x >= 0
        result[positive] = 1 / (1 + np.exp(-x[positive]))
        e = np.exp(x[~positive])
        result[~positive] = e / (1 + e)
        return result

    pre = sigmoid(pre_logits) + 1e-6
    post = sigmoid(post_logits) * 2
    # Independent per-token matrix equations, with the original row/column order.
    comb = np.empty((4, 4, 288), np.float64)
    for t in range(288):
        logits = (z[8:, t] * scale[2] + base[8:]).reshape(4, 4)
        values = np.exp(logits - logits.max(axis=1, keepdims=True))
        values = values / values.sum(axis=1, keepdims=True) + 1e-6
        values = values / (values.sum(axis=0, keepdims=True) + 1e-6)
        for _ in range(19):
            values = values / (values.sum(axis=1, keepdims=True) + 1e-6)
            values = values / (values.sum(axis=0, keepdims=True) + 1e-6)
        comb[:, :, t] = values
    return pre, post, comb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", default=200, type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    compact = compact_gate(args.out)

    def adapter(fn, *xs):
        pre, post, comb = fn(*xs, hidden_size=5120)
        return pre[:288].T, post[:288].T, comb[:288].transpose(1, 2, 0)

    def native(*xs):
        return adapter(original._pre_gates, *xs)

    def candidate(*xs):
        return adapter(compact, *xs)

    funcs = {"native": jax.jit(native), "pallas": jax.jit(candidate)}
    result = {
        "runtime": {
            n: importlib.metadata.version(n)
            for n in ("jax", "jaxlib", "libtpu", "numpy")
        },
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "original gate producer + trim/transpose to shifted post/pre "
            "consumer input layout; large consumer not yet included"
        ),
        "correctness": [],
        "timings": {},
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    xs = tuple(jax.device_put(x) for x in fixture("normal"))
    for name, fn in funcs.items():
        compiled = fn.lower(*xs).compile()
        (args.out / f"{name}-optimized.txt").write_text(compiled.as_text())
    for kind, seed in [
        ("normal", 42),
        ("normal", 137),
        ("zero", 42),
        ("tiny", 42),
        ("saturated", 42),
        ("large", 42),
        ("uniform", 42),
    ]:
        cpu = fixture(kind, seed)
        want = reference(cpu)
        xs = tuple(jax.device_put(x) for x in cpu)
        actual = {n: jax.block_until_ready(fn(*xs)) for n, fn in funcs.items()}
        errors = {}
        for name, values in actual.items():
            errors[name] = []
            for a, b in zip(values, want):
                np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-6)
                errors[name].append(float(np.max(np.abs(np.asarray(a) - b))))
        for a, b in zip(actual["native"], actual["pallas"]):
            np.testing.assert_array_equal(a, b)
        row = {
            "kind": kind,
            "seed": seed,
            "three_way_pass": True,
            "native_candidate_exact": True,
            "max_abs_vs_float64": errors,
        }
        result["correctness"].append(row)
        save()
        print(json.dumps(row), flush=True)
    result["precision_passed"] = True
    xs = tuple(jax.device_put(x) for x in fixture("normal"))
    for _ in range(20):
        for fn in funcs.values():
            jax.block_until_ready(fn(*xs))
    samples = {name: [] for name in funcs}
    for i in range(args.samples):
        order = list(funcs.items())
        for name, fn in order if i % 2 == 0 else order[::-1]:
            start = time.perf_counter_ns()
            jax.block_until_ready(fn(*xs))
            samples[name].append((time.perf_counter_ns() - start) / 1000)
    result["timings"]["gate_adapter"] = {
        n: {"median_us": float(np.median(s)), "samples_us": s}
        for n, s in samples.items()
    }
    save()
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile-gate")):
            for i in range(16):
                order = list(funcs.items())
                for name, fn in order if i % 2 == 0 else order[::-1]:
                    with jax.profiler.TraceAnnotation(name, step_num=i):
                        jax.block_until_ready(fn(*xs))
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
