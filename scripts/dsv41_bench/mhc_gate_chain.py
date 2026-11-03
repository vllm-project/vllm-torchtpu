# SPDX-License-Identifier: Apache-2.0
"""Two real shifted mHC calls; candidate changes only intermediate gate stores."""

import argparse
import hashlib
import importlib.metadata
import json
import time
import types
from pathlib import Path

import jax
import mhc_gate_layout as gate
import ml_dtypes
import numpy as np


def bf16(x):
    return np.asarray(x).astype(ml_dtypes.bfloat16)


def pack(fn):
    high = (fn.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    remainder = fn - high
    middle = (remainder.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    low = fn - high - middle
    parts = bf16(np.concatenate((high, middle, low))).T
    np.testing.assert_array_equal(
        parts[:, :24].astype(np.float32)
        + parts[:, 24:48].astype(np.float32)
        + parts[:, 48:72].astype(np.float32),
        fn.T,
    )
    return np.pad(parts, ((0, 0), (0, 56)))


def fixture(kind, seed=42):
    rng = np.random.default_rng(seed)
    x0 = bf16(rng.standard_normal((288, 5120)))
    flat = bf16(rng.standard_normal((288, 20480)))
    x1 = bf16(rng.standard_normal((288, 5120)))
    post = rng.uniform(0, 2, (288, 4, 1)).astype(np.float32)
    comb = rng.uniform(0, 1, (288, 4, 4)).astype(np.float32)
    comb /= comb.sum(axis=-1, keepdims=True)
    pre = rng.uniform(0, 1, (288, 4)).astype(np.float32)
    if kind == "zero":
        x0[:] = flat[:] = x1[:] = 0
    elif kind == "tiny":
        x0, flat, x1 = (bf16(a.astype(np.float32) * 1e-15) for a in (x0, flat, x1))
    elif kind == "identity":
        post[:] = 0
        comb[:] = np.eye(4, dtype=np.float32)
        pre[:] = [1, 0, 0, 0]
    elif kind == "skew":
        comb[:] = 0
        comb[:, :, 0] = 1
        pre[:] = [1, 0, 0, 0]
    fn0, fn1 = [rng.normal(0, 0.01, (24, 20480)).astype(np.float32) for _ in range(2)]
    scale0, scale1 = [rng.uniform(0.8, 1.2, 3).astype(np.float32) for _ in range(2)]
    base0, base1 = [rng.normal(0, 0.1, 24).astype(np.float32) for _ in range(2)]
    norm0, norm1 = [bf16(rng.uniform(0.8, 1.2, 5120)) for _ in range(2)]
    xs = (
        x0,
        flat,
        post,
        comb,
        pre,
        pack(fn0),
        scale0,
        base0,
        norm0,
        x1,
        pack(fn1),
        scale1,
        base1,
        norm1,
    )
    return xs, (fn0, fn1)


def reference_one(x, flat, post, comb, pre, fn, scale, base, norm):
    x, flat, post, comb, pre, fn, norm = (
        a.astype(np.float64) for a in (x, flat, post, comb, pre, fn, norm)
    )
    # Dense independent equations, retaining BF16 residual/collapse boundaries.
    residual = np.einsum("tij,tid->tjd", comb, flat.reshape(288, 4, 5120))
    residual += post * x[:, None, :]
    residual = bf16(residual).astype(np.float64)
    collapsed = bf16(np.sum(pre[..., None] * residual, axis=1)).astype(np.float64)
    value = bf16(
        collapsed
        / np.sqrt(np.mean(collapsed**2, axis=-1, keepdims=True) + 1e-20)
        * norm
    )
    new_flat = residual.reshape(288, 20480)
    mix = new_flat @ fn.T
    square = np.sum(new_flat**2, axis=1, keepdims=True)
    next_pre, next_post, next_comb = gate.reference(
        (
            np.pad(mix.T, ((0, 0), (0, 96))),
            np.pad(square.T, ((0, 0), (0, 96))),
            scale,
            base,
        )
    )
    return (
        bf16(new_flat),
        next_post.T[..., None],
        next_comb.transpose(2, 0, 1),
        value,
        next_pre.T,
    )


def reference(xs, weights):
    x0, flat, post, comb, pre, _, scale0, base0, norm0, x1, _, scale1, base1, norm1 = xs
    first = reference_one(x0, flat, post, comb, pre, weights[0], scale0, base0, norm0)
    second = reference_one(
        x1, first[0], first[1], first[2], first[4], weights[1], scale1, base1, norm1
    )
    return (first[3], *second)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", default=1000, type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    compact = gate.compact_gate(args.out)
    original = gate.original.shifted_post_pre_rms
    namespace = dict(original.__globals__)
    namespace["_pre_gates"] = compact
    modified = types.FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    modified.__kwdefaults__ = original.__kwdefaults__

    def chain(fn, *xs):
        (
            x0,
            flat,
            post,
            comb,
            pre,
            w0,
            scale0,
            base0,
            norm0,
            x1,
            w1,
            scale1,
            base1,
            norm1,
        ) = xs
        first = fn(
            x0,
            flat,
            post,
            comb,
            w0,
            scale0,
            base0,
            pre,
            norm_weight=norm0,
            norm_eps=1e-20,
        )
        second = fn(
            x1,
            first[0],
            first[1],
            first[2],
            w1,
            scale1,
            base1,
            first[4],
            norm_weight=norm1,
            norm_eps=1e-20,
        )
        return (first[3], *second)

    def native(*xs):
        return chain(original, *xs)

    def candidate(*xs):
        return chain(modified, *xs)

    funcs = {"native": jax.jit(native), "pallas": jax.jit(candidate)}
    source_paths = [Path(__file__), Path(gate.__file__), gate.SOURCE]
    result = {
        "runtime": {
            n: importlib.metadata.version(n)
            for n in ("jax", "jaxlib", "libtpu", "numpy")
        },
        "source_sha256": {
            str(p.relative_to(gate.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "scope": (
            "two original shifted_post_pre_rms calls, including all gate adapters, "
            "residual/collapse/projection/RMS; x1 is an independent sublayer input"
        ),
        "reference_tolerances": {
            "gate": [5e-3, 1e-5],
            "bf16": [2e-2, 2e-2],
            "origin": "tests/models/vllm/test_dsv41_common_mhc.py",
        },
        "correctness": [],
        "timings": {},
        "compile_llo": {},
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    cpu, weights = fixture("normal")
    xs = tuple(jax.device_put(x) for x in cpu)
    for name, fn in funcs.items():
        before = set((args.out / "compiler").glob("*post-finalize-llo.txt"))
        compiled = fn.lower(*xs).compile()
        (args.out / f"{name}-optimized.txt").write_text(compiled.as_text())
        added = set((args.out / "compiler").glob("*post-finalize-llo.txt")) - before
        result["compile_llo"][name] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in added
        }
        save()
    for kind, seed in [
        ("normal", 42),
        ("normal", 137),
        ("zero", 42),
        ("tiny", 42),
        ("identity", 42),
        ("skew", 42),
    ]:
        cpu, weights = fixture(kind, seed)
        want = reference(cpu, weights)
        xs = tuple(jax.device_put(x) for x in cpu)
        actual = {n: jax.block_until_ready(fn(*xs)) for n, fn in funcs.items()}
        errors = {}
        for name, values in actual.items():
            errors[name] = []
            for i, (a, b) in enumerate(zip(values, want)):
                rtol, atol = (2e-2, 2e-2) if i in (0, 1, 4) else (5e-3, 1e-5)
                a, b = (
                    np.asarray(a).astype(np.float64),
                    np.asarray(b).astype(np.float64),
                )
                np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
                errors[name].append(float(np.max(np.abs(a - b))))
        for a, b in zip(actual["native"], actual["pallas"]):
            np.testing.assert_array_equal(
                np.ascontiguousarray(a).view(np.uint8),
                np.ascontiguousarray(b).view(np.uint8),
            )
        row = {
            "kind": kind,
            "seed": seed,
            "three_way_pass": True,
            "native_candidate_exact": True,
            "max_abs_vs_reference": errors,
        }
        result["correctness"].append(row)
        save()
        print(json.dumps(row), flush=True)
    result["precision_passed"] = True
    cpu, _ = fixture("normal")
    xs = tuple(jax.device_put(x) for x in cpu)
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
    result["timings"]["chain"] = {
        n: {"median_us": float(np.median(s)), "samples_us": s}
        for n, s in samples.items()
    }
    save()
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile-chain")):
            for i in range(16):
                order = list(funcs.items())
                for name, fn in order if i % 2 == 0 else order[::-1]:
                    with jax.profiler.TraceAnnotation(name, step_num=i):
                        jax.block_until_ready(fn(*xs))
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
