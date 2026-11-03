# SPDX-License-Identifier: Apache-2.0
"""Compile-only real compressor consumer layout check; never allocate KV banks.

The original Pallas compressor body is unchanged. The packed wrapper replaces
only its two external compact calls with already computed tables. This is JAX
compiler evidence, not a Torch correctness or performance measurement.
"""

import argparse
import ast
import copy
import functools
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from attention_state import ROOT, load_preparation, packed_function, pallas_function


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def consumer_functions():
    base = ROOT / "src/vllm_torchtpu/kernels/attention_preprocessing"
    source = base / "compress_rope.py"
    rope = load_module(base / "rope_gather.py", "compile_rope_gather")
    tuning = load_module(base / "tuned_block_sizes.py", "compile_tuning")
    tree = ast.parse(source.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("vllm_torchtpu.")
        )
    ]
    ns = {
        "rope_gather": rope,
        "_LANE": rope._LANE,
        "_lane_gather": rope._lane_gather,
        "COMPRESS_ROPE_TUNING": tuning.COMPRESS_ROPE_TUNING,
        "CompressConfig": tuning.CompressConfig,
    }
    exec(compile(tree, str(source), "exec"), ns)
    prep = load_preparation()
    ns["_compact"] = lambda records, total, tiles, tb: prep.compact_records(
        records, total, tb
    )
    fn = copy.deepcopy(
        next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "fused_compress_rope"
        )
    )
    fn.name = "compress_with_prepacked_tables"
    fn.args.kwonlyargs.append(ast.arg(arg="prefetch_override"))
    fn.args.kw_defaults.append(None)
    assignments = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "prefetch"
        and isinstance(n.value, ast.List)
        and len(n.value.elts) == 2
    ]
    assert len(assignments) == 1
    assert all(isinstance(n, ast.Starred) for n in assignments[0].value.elts)
    assignments[0].value = ast.Name(id="prefetch_override", ctx=ast.Load())
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(source), "exec"), ns)
    return ns, tuning.COMPRESS_ROPE_TUNING[(7, 288, 5120, 2)], source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ns, config, source = consumer_functions()
    address = pallas_function(16, 16, 0)
    packed = packed_function(16, 16, 0)

    def chain(mode, table, starts, positions, minimum, x, weight, gamma, rope, a, b):
        metadata = (table, starts, starts, positions, minimum)
        if mode == "retained":
            prev, valid, write, read = address(*metadata)
            return ns["fused_compress_rope"](
                x, weight, gamma, rope, prev, valid, (a, b), write, read, config=config
            )
        prev, valid, *tables = packed(*metadata)
        shape_only = jnp.empty((288,), jnp.int32)
        return ns["compress_with_prepacked_tables"](
            x,
            weight,
            gamma,
            rope,
            prev,
            valid,
            (a, b),
            shape_only,
            shape_only,
            config=config,
            prefetch_override=tables,
        )

    shapes = (
        ((48, 8), jnp.int32),
        ((49,), jnp.int32),
        ((288,), jnp.int32),
        ((48,), jnp.int32),
        ((288, 5120), jnp.bfloat16),
        ((5120, 1024), jnp.bfloat16),
        ((512,), jnp.bfloat16),
        ((288, 64), jnp.float32),
        ((33423360, 128), jnp.int32),
        ((28652288, 128), jnp.int32),
    )
    signature = tuple(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in shapes)
    result = {
        "scope": "TPU compile only, JAX chain; no tensors allocated or executed",
        "geometry": {"capacity": 16, "stride": 16, "offset": 0},
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                source,
                Path(__file__),
                Path(__file__).with_name("attention_state.py"),
            )
        },
        "modes": {},
    }
    for mode in ("retained", "packed"):
        fn = jax.jit(functools.partial(chain, mode), donate_argnums=(8, 9))
        lower = fn.lower(*signature)
        (args.out / f"{mode}.stablehlo.txt").write_text(lower.as_text())
        compiled = lower.compile()
        text = compiled.as_text()
        (args.out / f"{mode}.hlo.txt").write_text(text)
        result["modes"][mode] = {
            "compiled": True,
            "memory_analysis": str(compiled.memory_analysis()),
            "hlo_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(mode, result["modes"][mode], flush=True)


if __name__ == "__main__":
    main()
