# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only state-packed gate widening; no combined-address experiments.

CPU: python state_packed_shapes.py --cpu-test
TPU7: python state_packed_shapes.py --out /tmp/state-packed [--profile]
Timing is source-translated native metadata + frozen actual _compact vs packed.
This is JAX exploration, NOT Torch dispatcher/native production speedup evidence.
NOT the compressor consumer. Actual compress_rope integration must follow.
"""

import argparse
import ast
import hashlib
import json
import time
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "src/vllm_torchtpu/kernels"
PREP = KERNELS / "experimental/decode_preparation.py"
COMPRESS = KERNELS / "attention_preprocessing/compress_rope.py"
TUNING = COMPRESS.with_name("tuned_block_sizes.py")
ATTENTION = ROOT / "src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention.py"
DISPATCH = ATTENTION.with_name("preprocess_ops.py")
STATE = Path(__file__).with_name("attention_state.py")
BASELINE = Path(__file__).with_name("compact_baseline.py")
COMPACT_AST_SHA256 = "c8ced9787dd398a144358b1e1f2188b9296b8f86c6128fce384ac31f189213a5"
TOKENS = (96, 192, 240, 256, 288)
RECORDS = 7759456
CASES = ("uniform", "ragged", "padding", "empty", "distinct", "long", "bounds")


def function(path, name):
    return next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def extract(path, names, namespace):
    nodes = [function(path, name) for name in names]
    exec(compile(ast.Module(nodes, []), str(path), "exec"), namespace)
    return namespace


def candidate_namespace(cpu=False):
    """Load the integrated packed kernel without rewriting its shape guard."""
    tree = ast.parse(PREP.read_text())
    packed = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compressor_state_packed"
    )
    if not cpu:
        ns = {"__name__": "state_packed_private"}
        exec(compile(tree, str(PREP), "exec"), ns)
        return ns
    # CPU exercises original address and compact expressions, not Pallas lowering.
    compact = next(
        n for n in packed.body if isinstance(n, ast.FunctionDef) and n.name == "compact"
    )
    ns = {"jnp": np, "prefix_sum": np.cumsum, "total_records": RECORDS}
    exec(
        compile(
            ast.Module([function(PREP, "_compressor_state_addresses"), compact], []),
            str(PREP),
            "exec",
        ),
        ns,
    )
    return ns


class TorchToJax(ast.NodeTransformer):
    """Translate only the API subset used by the extracted native helpers.

    Keep native bucketize/gathers (not packed candidate's reduction algorithm).
    x64 is required: Torch arange, bucketize and .long() use int64. Floor
    division/remainder retain negative-slot semantics; masked_fill is where.
    """

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute):
            return node
        obj, method = node.func.value, node.func.attr

        def call(name, args, keywords=()):
            return ast.Call(ast.parse(name, mode="eval").body, args, list(keywords))

        if isinstance(obj, ast.Name) and obj.id == "torch":
            if method == "arange":
                return call(
                    "jnp.arange",
                    node.args,
                    [ast.keyword("dtype", ast.parse("jnp.int64", mode="eval").body)],
                )
            if method == "bucketize":
                assert ast.literal_eval(node.keywords[0].value) is True
                value = call(
                    "jnp.searchsorted",
                    node.args[::-1],
                    [ast.keyword("side", ast.Constant("right"))],
                )
                return call(
                    "jnp.asarray", [value, ast.parse("jnp.int64", mode="eval").body]
                )
            if method in ("zeros_like", "full_like"):
                return call("jnp." + method, node.args, node.keywords)
            raise ValueError(f"Unreviewed Torch API: {method}")
        if method in ("int", "long"):
            dtype = "jnp.int32" if method == "int" else "jnp.int64"
            return call("jnp.asarray", [obj, ast.parse(dtype, mode="eval").body])
        if method == "numel":
            return ast.Attribute(obj, "size", ast.Load())
        if method == "clamp":
            assert len(node.keywords) == 1 and node.keywords[0].arg == "max"
            return call("jnp.minimum", [obj, node.keywords[0].value])
        if method == "masked_fill":
            return call("jnp.where", [node.args[0], node.args[1], obj])
        if method == "reshape":
            return call("jnp.reshape", [obj, ast.Tuple(node.args, ast.Load())])
        raise ValueError(f"Unreviewed tensor API: {method}")


def native_jax():
    """Extract actual native helpers AND attention_state's operand assembly."""
    import jax
    import jax.numpy as jnp

    if not jax.config.x64_enabled:
        raise RuntimeError("Enable JAX x64 to preserve Torch int64 intermediates")
    names = ("_block_table", "_token_requests", "_replayed", "_cache_slots")
    nodes = [function(ATTENTION, name) for name in names]
    cache_class = next(
        n
        for n in ast.parse(ATTENTION.with_name("cache.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "PagedCacheRows"
    )
    nodes.append(
        next(
            n
            for n in cache_class.body
            if isinstance(n, ast.FunctionDef) and n.name == "row_indices"
        )
    )
    wrapper = function(STATE, "native_function")
    nodes.append(
        next(
            n
            for n in wrapper.body
            if isinstance(n, ast.FunctionDef) and n.name == "native"
        )
    )
    tree = ast.fix_missing_locations(TorchToJax().visit(ast.Module(nodes, [])))
    assert not any(isinstance(n, ast.Name) and n.id == "torch" for n in ast.walk(tree))
    ns = {
        "jnp": jnp,
        "Metadata": namedtuple("Metadata", "block_tables seq_lens query_start_loc"),
        "cache": SimpleNamespace(shape=(1, 16), page_stride=16, row_offset=0),
        "capacity": 16,
        "window": 128,
    }
    ns["namespace"] = ns
    exec(compile(tree, str(STATE) + ":jax_translation", "exec"), ns)
    return ns["native"]


def frozen_compact(prep):
    """Exact a8bf0aeb7 _compact; resolve only its local dependency import."""
    import jax
    import jax.numpy as jnp

    node = function(BASELINE, "_compact")
    fingerprint = hashlib.sha256(
        ast.dump(node, include_attributes=False).encode()
    ).hexdigest()
    assert fingerprint == COMPACT_AST_SHA256, "Frozen _compact source changed"

    class ResolveImport(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            assert (
                node.module == "vllm_torchtpu.kernels.experimental.decode_preparation"
            )
            assert [(n.name, n.asname) for n in node.names] == [
                ("compact_records", None)
            ]
            return None

    tree = ast.fix_missing_locations(ResolveImport().visit(ast.Module([node], [])))
    ns = {"jax": jax, "jnp": jnp, "compact_records": prep["compact_records"]}
    exec(compile(tree, str(BASELINE), "exec"), ns)
    return ns["_compact"]


def compile_baseline(prep, values):
    """Compile metadata and compact separately under their required x64 modes.

    The baseline has two device dispatches, unlike the single packed dispatch.
    This is deliberate staging, not a Torch graph performance comparison.
    """
    import jax

    with jax.enable_x64(True):
        native = jax.jit(native_jax()).lower(*values).compile()
        operands = jax.eval_shape(native_jax(), *values)
    compact = frozen_compact(prep)

    def pack(prev, valid, write, read):
        t = write.shape[0]
        assert consumer_config(t).token_block == t
        return (
            prev,
            valid,
            *compact(write, RECORDS, 1, t),
            *compact(read, RECORDS, 1, t),
        )

    # The context MUST enclose lower AND compile: Pallas lowering itself uses
    # JAX reductions, whose default integer width depends on this context.
    with jax.enable_x64(False):
        pack_compiled = jax.jit(pack).lower(*operands).compile()

    def baseline(*xs):
        return pack_compiled(*native(*xs))

    return baseline


def consumer_config(tokens):
    """Execute actual select_config with a TPU7 version dependency stub."""
    # select_config's sole dynamic import is runtime TPU version lookup.
    select = function(COMPRESS, "select_config")
    assert isinstance(select.body[0], ast.ImportFrom)
    select.body = select.body[1:]
    ns = {"get_tpu_version": lambda: 7}
    exec(compile(ast.parse(TUNING.read_text()), str(TUNING), "exec"), ns)
    nodes = [function(COMPRESS, n) for n in ("_largest_divisor", "default_config")]
    ns["_ROW_ALIGN"] = 16
    exec(compile(ast.Module([*nodes, select], []), str(COMPRESS), "exec"), ns)
    return ns["select_config"](tokens, 5120, 2)


def allowed(tokens, shape=(48, 8), capacity=16, stride=16, offset=0, window=128):
    return tokens in TOKENS and (shape, capacity, stride, offset, window) == (
        (48, 8),
        16,
        16,
        0,
        128,
    )


def inputs(tokens, kind):
    rng = np.random.default_rng(20261104 + tokens)
    lengths = np.full(48, tokens // 48, np.int32)
    lengths[: tokens % 48] += 1
    if kind in ("ragged", "distinct", "bounds"):
        lengths = rng.multinomial(tokens - 7, np.ones(48) / 48).astype(np.int32)
    elif kind == "padding":
        lengths[29:] = 0
    elif kind == "empty":
        lengths[:] = 0
    elif kind == "long":
        lengths[:] = 0
        lengths[0], lengths[3] = 31, tokens - 32
    starts = np.r_[np.int32(0), lengths.cumsum(dtype=np.int32)]
    kv = starts.copy()
    if kind == "distinct":
        kv = np.r_[
            np.int32(0),
            rng.multinomial(tokens - 3, np.ones(48) / 48).cumsum(dtype=np.int32),
        ]
    positions = np.empty(tokens, np.int32)
    for r in range(48):
        positions[starts[r] : starts[r + 1]] = (
            np.arange(lengths[r], dtype=np.int32) + 2 * r
        )
    positions[starts[-1] :] = np.arange(tokens - starts[-1])
    positions[:8] = [0, 1, 127, 128, 129, 255, 98303, 98304]
    minimum = rng.choice(np.array([0, 1, 128, 256, 98304], np.int32), 48)
    table = rng.integers(0, 8192, (48, 8), dtype=np.int32)
    if kind == "bounds":
        table[:9, 0] = [
            -1,
            0,
            1,
            RECORDS // 16 - 1,
            RECORDS // 16,
            RECORDS // 16 + 1,
            261119,
            261120,
            261121,
        ]
    return table, starts, kv, positions, minimum


def oracle(arrays):
    table, starts, kv, positions, minimum = arrays
    n = len(positions)
    prev, valid = np.zeros(n, np.int32), np.zeros(n, np.int32)
    write, read = np.full(n, -1, np.int32), np.full(n, -1, np.int32)
    for t, position in enumerate(positions):
        r = min(sum(t >= b for b in starts[1:]), 47)
        k = min(sum(t >= b for b in kv[1:]), 47)
        prev[t] = t != starts[r]
        replay = minimum[k] > 0 and minimum[k] <= position < minimum[k] + 128
        valid[t] = t < kv[-1] and (position + 1) % 2 == 0 and not replay
        if t < starts[-1] and table[r, 0] >= 0:
            if t >= starts[r + 1] - 16:
                write[t] = table[r, 0] * 16 + position % 16
            if not prev[t]:
                read[t] = table[r, 0] * 16 + (position - 1) % 16
    result = [prev[:, None], valid[:, None]]
    for addresses in (write, read):
        pairs = np.zeros((n, 2), np.int32)
        count = 0
        for t, address in enumerate(addresses):
            if 0 <= address < RECORDS:
                pairs[count] = t, address
                count += 1
        result.extend((pairs.ravel(), np.array([0, count], np.int32)))
    return (prev, valid, write, read), result


def check(actual, expected):
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(got), want)


def cpu_test():
    import jax
    import jax.numpy as jnp

    ns = candidate_namespace(cpu=True)
    prep = candidate_namespace()
    native = jax.jit(native_jax())
    # Execute extracted candidate arithmetic with JAX, not only NumPy.
    candidate_math = {
        "jnp": jnp,
        "prefix_sum": prep["prefix_sum"],
        "total_records": RECORDS,
    }
    packed_node = function(PREP, "compressor_state_packed")
    compact_node = next(
        n
        for n in packed_node.body
        if isinstance(n, ast.FunctionDef) and n.name == "compact"
    )
    exec(
        compile(
            ast.Module(
                [function(PREP, "_compressor_state_addresses"), compact_node], []
            ),
            str(PREP),
            "exec",
        ),
        candidate_math,
    )
    checks = []
    for t in TOKENS:
        config = consumer_config(t)
        assert config.token_block == t
        baseline = compile_baseline(
            prep, tuple(jnp.asarray(a) for a in inputs(t, "uniform"))
        )
        for kind in CASES:
            arrays = inputs(t, kind)
            addresses, expected = oracle(arrays)
            values = tuple(jnp.asarray(a) for a in arrays)
            check(
                native(*values),
                (addresses[0][:, None], addresses[1][:, None], *addresses[2:]),
            )
            check(baseline(*values), expected)
            with jax.enable_x64(False):
                jax_addresses = candidate_math["_compressor_state_addresses"](*values)
                check(jax_addresses, addresses)
                jax_packed = [jax_addresses[0][:, None], jax_addresses[1][:, None]]
                for address in jax_addresses[2:]:
                    pairs, offsets = candidate_math["compact"](address)
                    jax_packed.extend((pairs.ravel(), offsets))
            assert all(x.dtype == jnp.int32 for x in jax_packed)
            check(jax_packed, expected)
            actual = ns["_compressor_state_addresses"](*arrays)
            check(actual, addresses)
            packed = [actual[0][:, None], actual[1][:, None]]
            for a in actual[2:]:
                pairs, offsets = ns["compact"](a)
                packed.extend((pairs.ravel(), offsets))
            check(packed, expected)
            checks.append({"tokens": t, "case": kind, "exact": True})
    assert not allowed(384) and not allowed(128) and not allowed(96, stride=32)
    assert consumer_config(384).token_block != 384
    return checks


def benchmark(args, result):
    import jax

    if jax.default_backend() != "tpu":
        raise RuntimeError("This benchmark requires TPU7; CPU mode is --cpu-test")
    if not all(
        "TPU7" in d.device_kind or "TPU v7" in d.device_kind for d in jax.devices()
    ):
        raise RuntimeError(f"Expected TPU7, got {jax.devices()}")
    private = candidate_namespace()
    original = {"__name__": "state_packed_original"}
    exec(compile(PREP.read_text(), str(PREP), "exec"), original)

    def packed(table, starts, kv, positions, minimum):
        assert allowed(positions.shape[0])
        fn = original if positions.shape[0] == 288 else private
        with jax.enable_x64(False):
            return fn["compressor_state_packed"](
                table, starts, kv, positions, minimum, total_records=RECORDS
            )

    result["baseline"]["dispatch"] = "two-stage compiled metadata then compact"
    result["baseline"]["original_latency_comparable"] = False
    for t in TOKENS:
        values = tuple(jax.device_put(a) for a in inputs(t, "uniform"))
        baseline = compile_baseline(original, values)
        with jax.enable_x64(False):
            candidate = jax.jit(packed).lower(*values).compile()
        for kind in CASES:
            arrays = inputs(t, kind)
            values = tuple(jax.device_put(a) for a in arrays)
            expected = oracle(arrays)[1]
            for fn in (baseline, candidate):
                check(jax.device_get(fn(*values)), expected)
        values = tuple(jax.device_put(a) for a in inputs(t, "uniform"))

        def run(fn, values=values):
            output = fn(*values)
            jax.block_until_ready(output)

        for _ in range(20):
            run(baseline)
            run(candidate)
        samples = {"native": [], "packed": []}
        for _ in range(args.samples):
            for name, fn in (
                ("native", baseline),
                ("packed", candidate),
                ("packed", candidate),
                ("native", baseline),
            ):
                start = time.perf_counter_ns()
                run(fn)
                samples[name].append((time.perf_counter_ns() - start) / 1000)
        result["timings"].append(
            {"tokens": t, "control_original": t == 288, "synchronous_us": samples}
        )
        if args.profile:
            with jax.profiler.trace(str(args.out / f"profile_T{t}")):
                for i in range(16):
                    for name, fn in (("native", baseline), ("packed", candidate)):
                        with jax.profiler.TraceAnnotation(name, step_num=i):
                            run(fn)


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not args.cpu_test and args.out is None:
        parser.error("--out required for TPU benchmark")
    result = {
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                PREP,
                COMPRESS,
                TUNING,
                ATTENTION,
                DISPATCH,
                ATTENTION.with_name("cache.py"),
                STATE,
                BASELINE,
                Path(__file__),
            )
        },
        "geometry": {
            "tokens": TOKENS,
            "batch": 48,
            "table": [48, 8],
            "capacity": 16,
            "stride": 16,
            "offset": 0,
            "window": 128,
        },
        "consumer": {
            "version": 7,
            "hidden": 5120,
            "ratio": 2,
            "whole_tile_verified_from_source": True,
            "integration": "MUST FOLLOW: actual compressor same weights/inputs",
        },
        "scope": (
            "source-translated native JAX metadata+frozen actual compact "
            "vs private packed; not end-to-end"
        ),
        "baseline": {
            "compact_revision": "a8bf0aeb7",
            "compact_ast_sha256": COMPACT_AST_SHA256,
            "metadata": "AST-extracted Torch helpers with reviewed JAX API translation",
            "torch_dispatcher_equivalence": False,
            "production_speedup_evidence": False,
            "x64_intermediates": True,
        },
        "runtime": {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()]},
        "cpu_checks": cpu_test(),
        "timings": [],
    }
    if not args.cpu_test:
        args.out.mkdir(parents=True, exist_ok=True)
        benchmark(args, result)
    text = json.dumps(result, separators=(",", ":"))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "state_packed_results.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
