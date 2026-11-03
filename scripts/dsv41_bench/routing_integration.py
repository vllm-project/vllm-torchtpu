# SPDX-License-Identifier: Apache-2.0
"""Replay the actual push-MoE caller between prepush and TC compute cuts."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import routing_complete as bench

from vllm_torchtpu.kernels.experimental.push_fused_moe import layer

ROOT = Path(__file__).resolve().parents[2]
LAYER = ROOT / "src/vllm_torchtpu/kernels/experimental/push_fused_moe/layer.py"


def extract(path, stride, out):
    source = path.read_text()
    compute = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "compute_local"
    )
    selected, active = [], False
    for node in compute.body:
        targets = {
            n.id
            for target in getattr(node, "targets", [])
            for n in ast.walk(target)
            if isinstance(n, ast.Name)
        }
        if "plan_kw" in targets:
            active = True
        if "arrivals" in targets:
            active = False
        if active and not targets.intersection({"recv_rows", "w1b_k", "w2b_k"}):
            selected.append(ast.unparse(node))
        # The old caller computes these after kfn; the new caller receives
        # the same two live outputs from prepare_routing.
        if not active and targets.intersection({"pos", "mirror_pos"}):
            selected.append(ast.unparse(node))
    text = (
        "def call(pairs, local, hist, true, aligned, starts, rows, rank, bits, slab):\n"
        "    me = rank[0]\n"
        "    pairs_g, local_blocks, block_hist = pairs, local, hist\n"
        "    rows_by_dest, runs = true.T, (true, aligned, starts, rows)\n"
        "    scale_bits_g = bits\n"
        + "\n".join("    " + line for part in selected for line in part.splitlines())
        + "\n    return (pos, mirror_pos, *kernel_tables, expert_rows, slab_base, "
        "counts, scale_slab, slab)\n"
    )
    (out / (path.stem + "-extracted.py")).write_text(text)
    namespace = dict(vars(layer))
    namespace.update(
        build_routing_tables_sharded=bench.push.build_routing_tables_sharded,
        RoutingTables=bench.host.RoutingTables,
        ShardRoutingTables=bench.host.ShardRoutingTables,
        T=bench.EP * bench.TOKENS,
        topk=bench.TOPK,
        ep=bench.EP,
        e_total=bench.EXPERTS,
        t_local=bench.TOKENS,
        block=bench.BLOCK,
        capacity=128,
        ragged_stride=stride,
        sharded_plan=True,
        form=types.SimpleNamespace(quantized_activations=True),
    )
    exec(compile(text, str(path), "exec"), namespace)
    return namespace["call"], hashlib.sha256(source.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.fallback:
        bench.STRIDE += 128
        bench.meta.STRIDE = bench.STRIDE
    funcs, hashes = {}, {}
    for name, path in (("baseline", args.baseline_source), ("candidate", LAYER)):
        funcs[name], hashes[name] = extract(path, bench.STRIDE, args.out)
    # Same shape as the actual FP8 q_sorted/prepush slab; payload is supplied
    # at this cut. No collective or SC prepush body is part of the timing.
    # The cut carries ownership forward; donation avoids a synthetic 144MB copy.
    slab = jnp.zeros(
        (
            bench.STRIDE,
            5120 // bench.host.HIDDEN_LANE_BLOCK,
            bench.host.HIDDEN_LANE_BLOCK,
        ),
        jnp.float8_e4m3fn,
    )
    bits = bench.scale_fixture()
    arrays = (*bench.fixture("uniform", 0), bits)
    xs = (*tuple(jax.device_put(x) for x in arrays), slab)
    compiled, results = (
        {},
        {
            "source_sha256": hashes,
            "runtime": {
                p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")
            },
            "fallback": args.fallback,
            "stride": bench.STRIDE,
            "scope": (
                "actual routing caller and scale scatter; supplied prepush slab; "
                "before TC MoE/communication/combine"
            ),
            "correctness": [],
            "timings": {},
        },
    )
    for name, fn in funcs.items():
        lowered = jax.jit(fn, donate_argnums=(9,)).lower(*xs)
        (args.out / f"{name}.stablehlo.txt").write_text(lowered.as_text())
        compiled[name] = lowered.compile()
        (args.out / f"{name}.hlo.txt").write_text(compiled[name].as_text())
        results[name + "_memory"] = str(compiled[name].memory_analysis())
    labels = (*bench.LABELS[:2], *bench.LABELS[3:], "scale_slab")
    for case in ("uniform", "skew", "empty", "edge"):
        for rank in (0, 7, 15):
            arrays = bench.fixture(case, rank)
            expected = bench.scale_reference(bench.reference(*arrays), bits)
            inputs = (*tuple(jax.device_put(x) for x in (*arrays, bits)), slab)
            for name, fn in compiled.items():
                outputs = fn(*inputs)
                actual = jax.device_get(outputs[:-1])
                for label, got, want in zip(labels, actual, expected, strict=True):
                    if got.dtype == np.float32:
                        got, want = got.view(np.uint32), want.view(np.uint32)
                    np.testing.assert_array_equal(
                        got, want, err_msg=f"{case}/{rank}/{name}/{label}"
                    )
                # The supplied prepush payload must survive the unchanged barrier.
                assert bool(jax.device_get(jnp.all(outputs[-1] == 0)))
                slab = outputs[-1]
                inputs = (*inputs[:-1], slab)
            results["correctness"].append({"case": case, "rank": rank, "exact": True})
    xs = (*xs[:-1], slab)
    for fn in compiled.values():
        for _ in range(20):
            outputs = jax.block_until_ready(fn(*xs))
            xs = (*xs[:-1], outputs[-1])
    raw = {name: [] for name in compiled}
    for i in range(args.samples):
        for name in (
            ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
        ):
            start = time.perf_counter_ns()
            outputs = jax.block_until_ready(compiled[name](*xs))
            raw[name].append((time.perf_counter_ns() - start) / 1000)
            xs = (*xs[:-1], outputs[-1])
    for name, values in raw.items():
        results["timings"][name] = {
            "median_us": float(np.median(values)),
            "p10_us": float(np.percentile(values, 10)),
            "p90_us": float(np.percentile(values, 90)),
            "raw_us": values,
        }
    results["passed"] = True
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile-routing")):
            for _ in range(16):
                for fn in compiled.values():
                    outputs = jax.block_until_ready(fn(*xs))
                    xs = (*xs[:-1], outputs[-1])
    print(
        {
            name: {k: v for k, v in timing.items() if k != "raw_us"}
            for name, timing in results["timings"].items()
        }
    )


if __name__ == "__main__":
    main()
