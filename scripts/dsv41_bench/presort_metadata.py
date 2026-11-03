# SPDX-License-Identifier: Apache-2.0
"""Count-based presort metadata fusion; preserve the original FP8 gather."""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import logging
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax

ROOT = Path(__file__).resolve().parents[2]
for name in ("vllm_torchtpu", "vllm_torchtpu.kernels"):
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "src" / name.replace(".", "/"))]
    sys.modules[name] = package
logger_module = types.ModuleType("vllm_torchtpu.logger")
logger = logging.getLogger("presort_metadata")
logger.warning_once = functools.lru_cache(None)(logger.warning)
logger_module.init_logger = lambda name: logger
sys.modules[logger_module.__name__] = logger_module

from vllm_torchtpu.kernels.experimental.decode_preparation import (  # noqa: E402
    fuse_function,
    prefix_sum,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe import host  # noqa: E402


def metadata_candidate(pairs, counts, *, revision=1):
    """Retain count semantics, excluding the rejected activation matmul."""
    topk, tokens = (6, pairs.size // 6) if revision == 3 else pairs.shape
    experts = counts.size
    bins = jnp.arange(experts, dtype=jnp.int32)
    selection = jnp.zeros((tokens, experts), bool)
    for k in range(topk):
        if revision == 3:
            column = lax.gather(
                pairs,
                (jnp.arange(tokens, dtype=jnp.int32) * topk + k)[:, None],
                lax.GatherDimensionNumbers(
                    offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
                ),
                slice_sizes=(1,),
                mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
            )
        else:
            column = pairs[k]
        selection = selection | (column[:, None] == bins[None, :])
    cumulative = prefix_sum(selection.astype(jnp.int32), axis=0)
    end = prefix_sum(counts)
    start = end - counts
    jj = jnp.arange(tokens * topk, dtype=jnp.int32)
    expert = jnp.minimum(jnp.sum(end[None, :] <= jj[:, None], axis=1), experts - 1)
    onehot = expert[:, None] == bins[None, :]
    offset = jj - jnp.sum(jnp.where(onehot, start[None, :], 0), axis=1)
    if revision == 1:
        col = jnp.dot(
            onehot.astype(jnp.float32),
            cumulative.T.astype(jnp.float32),
            precision=lax.Precision.HIGHEST,
        ).astype(jnp.int32)
    else:
        # One-hot rows select one integer in [0, 288]. Both base-256 digits
        # are exactly representable in BF16; FP32 products/sums are exact.
        hot = onehot.astype(jnp.bfloat16)
        low = (cumulative & 255).T.astype(jnp.bfloat16)
        high = (cumulative >> 8).T.astype(jnp.bfloat16)
        col = (
            jnp.dot(hot, low, preferred_element_type=jnp.float32)
            + 256 * jnp.dot(hot, high, preferred_element_type=jnp.float32)
        ).astype(jnp.int32)
    order = jnp.minimum(jnp.sum(col <= offset[:, None], axis=1), tokens - 1)
    return order, start


def native(pairs, counts, q):
    return host.local_sort_activations(pairs, counts, q, e_total=384, topk=6)


def fixture(kind, seed=42):
    rng = np.random.default_rng(seed)
    pairs = np.stack(
        [
            rng.choice(12 if kind == "skew" else 384, 6, replace=False)
            for _ in range(288)
        ]
    ).astype(np.int32)
    if kind == "empty":
        pairs[:] = 384
    elif kind == "edge":
        pairs[::3] = 384
        pairs[0] = [0, 1, 2, 381, 382, 383]
    elif kind == "duplicates":
        pairs[:, 1] = pairs[:, 0]
    elif kind == "full_count":
        pairs[:] = np.arange(6, dtype=np.int32)
    elif kind == "sentinel":
        pairs[::5, ::2] = 10000
    flat = pairs.reshape(-1)
    counts = np.bincount(flat[flat < 384], minlength=384).astype(np.int32)
    raw = rng.integers(0, 256, (288, 40, 128), dtype=np.uint8)
    raw.reshape(-1)[:256] = np.arange(256, dtype=np.uint8)
    return flat, counts, raw.view(ml_dtypes.float8_e4m3fn)


def reference(xs):
    pairs, counts, q = xs
    tokens, experts = q.shape[0], counts.size
    pairs = pairs.reshape(tokens, -1)
    end = np.cumsum(counts, dtype=np.int32)
    start = end - counts
    order = []
    # A per-row search/reference deliberately independent of the matrix kernel.
    for j in range(pairs.size):
        expert = min(int(np.searchsorted(end, j, side="right")), experts - 1)
        seen = np.cumsum(np.any(pairs == expert, axis=1))
        order.append(min(int(np.count_nonzero(seen <= j - start[expert])), tokens - 1))
    return q[np.asarray(order)], start


def assert_equal(a, b):
    np.testing.assert_array_equal(
        np.asarray(a[0]).view(np.uint8), np.asarray(b[0]).view(np.uint8)
    )
    np.testing.assert_array_equal(a[1], b[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", default=200, type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--revision", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    initial = tuple(jax.device_put(x) for x in fixture("uniform"))
    fused = fuse_function(
        functools.partial(metadata_candidate, revision=args.revision),
        (
            initial[0] if args.revision == 3 else initial[0].reshape(288, 6).T,
            initial[1],
        ),
        name=f"dsv41_presort_metadata_v{args.revision}",
    )

    def candidate(pairs, counts, q):
        order, start = fused(
            pairs if args.revision == 3 else pairs.reshape(288, 6).T, counts
        )
        return q[order], start

    if args.production:
        from vllm_torchtpu.kernels.experimental.push_fused_moe import layer, presort

        assert layer.local_sort_activations is presort.local_sort_activations
        assert presort.supports_local_sort(*initial, e_total=384, topk=6)

        def candidate(pairs, counts, q):
            return layer.local_sort_activations(pairs, counts, q, e_total=384, topk=6)

    funcs = {"native": jax.jit(native), "pallas": jax.jit(candidate)}
    source_paths = [
        Path(__file__),
        ROOT / "src/vllm_torchtpu/kernels/experimental/decode_preparation.py",
        ROOT / "src/vllm_torchtpu/kernels/experimental/push_fused_moe/host.py",
        ROOT / "src/vllm_torchtpu/kernels/experimental/push_fused_moe/layer.py",
    ]
    if args.production:
        source_paths.append(
            ROOT / "src/vllm_torchtpu/kernels/experimental/push_fused_moe/presort.py"
        )
    result = {
        "revision": args.revision,
        "production": args.production,
        "runtime": {
            "versions": {
                n: importlib.metadata.version(n)
                for n in ("jax", "jaxlib", "libtpu", "numpy", "ml_dtypes")
            },
            "devices": [str(d) for d in jax.devices()],
            "device_kind": jax.devices()[0].device_kind,
        },
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "scope": (
            "original local_sort_activations versus metadata Pallas including "
            "input transpose, outputs and unchanged native FP8 gather; "
            "communication excluded at existing optimization barriers"
        ),
        "correctness": [],
        "timings": {},
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    for name, fn in funcs.items():
        compiled = fn.lower(*initial).compile()
        (args.out / f"{name}-optimized.txt").write_text(compiled.as_text())
        (args.out / f"{name}-stablehlo.txt").write_text(
            str(fn.lower(*initial).compiler_ir())
        )
    for kind, seed in [
        ("uniform", 42),
        ("uniform", 137),
        ("skew", 42),
        ("empty", 42),
        ("edge", 42),
        ("duplicates", 42),
        ("sentinel", 42),
        ("full_count", 42),
    ]:
        cpu = fixture(kind, seed)
        xs = tuple(jax.device_put(x) for x in cpu)
        np.testing.assert_array_equal(
            np.asarray(xs[2]).view(np.uint8), cpu[2].view(np.uint8)
        )
        want = reference(cpu)
        outputs = {n: jax.block_until_ready(fn(*xs)) for n, fn in funcs.items()}
        for output in outputs.values():
            assert_equal(output, want)
        assert_equal(outputs["native"], outputs["pallas"])
        row = {
            "case": kind,
            "seed": seed,
            "three_way_exact": True,
            "fp8_all_256_patterns": True,
            "outputs": 2,
        }
        result["correctness"].append(row)
        save()
        print(json.dumps(row), flush=True)
    if args.production:
        result["fallbacks"] = []
        base = fixture("uniform")
        for kind in ("tokens", "dtype", "topk", "experts", "width", "index_dtype"):
            p, c, q = (x.copy() for x in base)
            e, k = 384, 6
            if kind == "tokens":
                q, p = q[:256], p[: 256 * k]
            elif kind == "dtype":
                q = q.astype(ml_dtypes.bfloat16)
            elif kind == "topk":
                p, k = p.reshape(288, 6)[:, :4].reshape(-1), 4
            elif kind == "experts":
                e = 128
            elif kind == "width":
                q = q[:, :20]
            elif kind == "index_dtype":
                p = p.astype(np.int16)
            c = np.bincount(p[p < e], minlength=e).astype(np.int32)
            xs = tuple(jax.device_put(x) for x in (p, c, q))
            assert not presort.supports_local_sort(*xs, e_total=e, topk=k)
            original = jax.jit(
                functools.partial(host.local_sort_activations, e_total=e, topk=k)
            )
            actual = jax.jit(
                functools.partial(layer.local_sort_activations, e_total=e, topk=k)
            )
            want = reference((p, c, q))
            a, b = jax.block_until_ready((original(*xs), actual(*xs)))
            assert_equal(a, want)
            assert_equal(b, want)
            assert_equal(a, b)
            hlo = actual.lower(*xs).compile().as_text()
            assert "dsv41_presort_metadata" not in hlo
            (args.out / f"fallback-{kind}-optimized.txt").write_text(hlo)
            result["fallbacks"].append(
                {"case": kind, "three_way_exact": True, "no_pallas": True}
            )
            save()
    result["precision_passed"] = True
    for kind in ("uniform", "skew"):
        xs = tuple(jax.device_put(x) for x in fixture(kind))
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
        result["timings"][kind] = {
            name: {
                "median_us": float(np.median(xs)),
                "p10_p90_us": np.percentile(xs, [10, 90]).tolist(),
                "samples_us": xs,
            }
            for name, xs in samples.items()
        }
        save()
        print(
            json.dumps(
                {
                    "case": kind,
                    "medians_us": {n: float(np.median(s)) for n, s in samples.items()},
                }
            ),
            flush=True,
        )
        if args.profile:
            with jax.profiler.trace(str(args.out / ("profile-" + kind))):
                for i in range(16):
                    order = list(funcs.items())
                    for name, fn in order if i % 2 == 0 else order[::-1]:
                        with jax.profiler.TraceAnnotation(name, step_num=i):
                            jax.block_until_ready(fn(*xs))
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
