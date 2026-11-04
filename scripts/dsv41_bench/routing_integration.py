# SPDX-License-Identifier: Apache-2.0
"""Replay the actual push-MoE caller between prepush and TC compute cuts."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import time
import types
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import routing_complete as bench

from vllm_torchtpu.kernels.experimental.push_fused_moe import layer

ROOT = Path(__file__).resolve().parents[2]
LAYER = ROOT / "src/vllm_torchtpu/kernels/experimental/push_fused_moe/layer.py"


ROUTING = LAYER.with_name("routing.py")
TOKEN_CHOICES = (96, 192, 256, 288, 384)
ROUTING_CALL = "dsv41_routing_position_metadata"
BASELINE_ROUTING = Path(__file__).with_name("_routing_baseline_09b82cc91.py")
BASELINE_SHA256 = "08aba11b235b2ce31e45cf2d0272e8796f6c9af527c13dbbb0c917269567e5af"


def load_production_baseline(path: Path = BASELINE_ROUTING):
    """Load the byte-validated production snapshot without requiring Git."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError("Frozen routing baseline does not match revision 09b82cc91")
    return load_routing(path)


def load_routing(path: Path, *, shape_candidate: bool = False):
    """Execute a private snapshot so its functions share private module globals."""
    source = path.read_text()
    if shape_candidate:
        gate = "(t_local, gathered.topk, e_total, ep, tile_m) == (288, 6, 384, 16, 128)"
        if source.count(gate) != 1:
            raise ValueError("Routing shape gate changed; review exploratory patch")
        source = source.replace(
            gate,
            f"t_local in {TOKEN_CHOICES!r} "
            "and (gathered.topk, e_total, ep, tile_m) == (6, 384, 16, 128)",
        )
    module = types.ModuleType(
        "shape_candidate_routing" if shape_candidate else "snapshot_routing"
    )
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), vars(module))
    module.source_sha256 = hashlib.sha256(source.encode()).hexdigest()
    return module


def configure_shape(tokens: int, *, fallback: bool = False) -> None:
    """Set every fixture-dependent dimension before constructing any arrays."""
    bench.TOKENS = tokens
    bench.BLOCK = bench.host.routing_block(tokens, 6)
    bench.STRIDE = bench.host.ragged_stride_bound(16 * tokens, 6, 384, 128)
    if fallback:
        bench.STRIDE += 128
    bench.meta.STRIDE = bench.STRIDE


def extract(path, stride, out, routing=None, tag=None, *, native=False):
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
    (out / ((tag or path.stem) + "-extracted.py")).write_text(text)
    namespace = dict(vars(layer))
    if routing is not None:
        namespace["prepare_routing"] = partial(
            routing.prepare_routing, fuse_metadata=not native
        )
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


def smem_consumer(fn):
    """Feed the five kernel tables through scalar prefetch, as kfn does.

    The push MoE kernel reads them from SMEM, which pins a row-major
    operand layout; returning them straight from jit would not, and hides
    any relayout the producer leaves behind. Values pass through unchanged.
    """
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    def consume(*tables):
        n = len(tables)

        def kernel(*refs):
            for src, dst in zip(refs[:n], refs[n:], strict=True):
                rows, cols = src.shape

                def body(k, carry, src=src, dst=dst, cols=cols):
                    dst[k // cols, k % cols] = src[k // cols, k % cols]
                    return carry

                lax.fori_loop(0, rows * cols, body, jnp.int32(0))

        smem = pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM)
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=n, in_specs=[], out_specs=[smem] * n, grid=()
            ),
            out_shape=[jax.ShapeDtypeStruct(t.shape, t.dtype) for t in tables],
            name="routing_tables_smem_consumer",
        )(*tables)

    def call(*inputs):
        outputs = fn(*inputs)
        return (*outputs[:2], *consume(*outputs[2:7]), *outputs[7:])

    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument(
        "--mode",
        choices=("exploratory", "integrated"),
        default="exploratory",
        help="integrated: actual production versus frozen 09b82cc91 routing",
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--tokens", type=int, choices=TOKEN_CHOICES, default=288)
    parser.add_argument(
        "--shape-candidate",
        action="store_true",
        help="exploratory private routing gate; does not change production support",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument(
        "--consumer",
        action="store_true",
        help="route the kernel tables through an SMEM scalar-prefetch consumer",
    )
    parser.add_argument(
        "--baseline-routing",
        type=Path,
        help="routing.py snapshot for the baseline; the layer source is shared",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_shape(args.tokens, fallback=args.fallback)
    funcs, hashes = {}, {}
    integrated = args.mode == "integrated"
    if integrated and (args.shape_candidate or args.baseline_source):
        parser.error("integrated mode uses the unmodified candidate and shared layer")
    if not integrated and args.baseline_source is None:
        parser.error("exploratory mode requires --baseline-source")
    baseline_path = args.baseline_routing or BASELINE_ROUTING
    baseline_routing = (
        load_production_baseline(baseline_path)
        if integrated
        else load_routing(baseline_path)
    )
    candidate_routing = load_routing(
        BASELINE_ROUTING if args.shape_candidate else ROUTING,
        shape_candidate=args.shape_candidate,
    )
    baseline_layer = LAYER if integrated else args.baseline_source
    for name, path, routing in (
        ("baseline", baseline_layer, baseline_routing),
        ("candidate", LAYER, candidate_routing),
    ):
        funcs[name], hashes[name] = extract(
            path,
            bench.STRIDE,
            args.out,
            routing=routing,
            tag=name,
            native=not integrated and name == "baseline",
        )
    hashes.update(
        baseline_routing=baseline_routing.source_sha256,
        candidate_routing=candidate_routing.source_sha256,
        production_routing=hashlib.sha256(ROUTING.read_bytes()).hexdigest(),
        harness=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        decode_preparation=hashlib.sha256(
            Path(bench.prep.__file__).read_bytes()
        ).hexdigest(),
    )
    for module in (bench, bench.meta, bench.meta.original, bench.host, bench.push):
        hashes[module.__name__] = hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
    if args.consumer:
        funcs = {name: smem_consumer(fn) for name, fn in funcs.items()}
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
            "mode": args.mode,
            "baseline_revision": "09b82cc91" if integrated else None,
            "fallback": args.fallback,
            "consumer": args.consumer,
            "shape_candidate": args.shape_candidate,
            "shape": {
                "tokens": bench.TOKENS,
                "ep": bench.EP,
                "topk": bench.TOPK,
                "experts": bench.EXPERTS,
            },
            "block": bench.BLOCK,
            "stride": bench.STRIDE,
            "source_paths": {
                "baseline": str(baseline_layer.resolve()),
                "baseline_routing": str(baseline_path.resolve()),
                "candidate": str(LAYER),
                "candidate_routing": str(ROUTING),
            },
            "compile": {},
            "profiles": {},
            "passed": False,
            "scope": (
                "actual routing caller and scale scatter; supplied prepush slab; "
                "before TC MoE/communication/combine"
            ),
            "correctness": [],
            "timings": {},
        },
    )

    def save_results():
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    save_results()
    for name, fn in funcs.items():
        start = time.perf_counter()
        try:
            lowered = jax.jit(fn, donate_argnums=(9,)).lower(*xs)
            stablehlo = lowered.as_text()
            (args.out / f"{name}.stablehlo.txt").write_text(stablehlo)
            compiled[name] = lowered.compile()
            hlo = compiled[name].as_text()
            (args.out / f"{name}.hlo.txt").write_text(hlo)
            hit = any(
                ROUTING_CALL in line and "custom-call" in line
                for line in hlo.splitlines()
            )
            expected = not args.fallback and (
                name == "candidate" or (integrated and args.tokens == 288)
            )
            assert hit == expected, (
                f"{name}: routing Pallas call present={hit}, expected={expected}"
            )
            results["compile"][name] = {
                "passed": True,
                "seconds": time.perf_counter() - start,
                "routing_pallas_call": hit,
            }
            results[name + "_memory"] = str(compiled[name].memory_analysis())
        except Exception as exc:
            results["compile"][name] = {
                "passed": False,
                "seconds": time.perf_counter() - start,
                "error": repr(exc),
            }
            raise
        finally:
            save_results()
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
        for name, fn in compiled.items():
            profile = args.out / f"profile-{name}-t{bench.TOKENS}-b{bench.BLOCK}"
            with jax.profiler.trace(str(profile)):
                for _ in range(16):
                    outputs = jax.block_until_ready(fn(*xs))
                    xs = (*xs[:-1], outputs[-1])
            results["profiles"][name] = str(profile.resolve())
        save_results()
    print(
        {
            name: {k: v for k, v in timing.items() if k != "raw_us"}
            for name, timing in results["timings"].items()
        }
    )


if __name__ == "__main__":
    main()
