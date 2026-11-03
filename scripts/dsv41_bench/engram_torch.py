# SPDX-License-Identifier: Apache-2.0
"""Actual Torch Engram tail versus Pallas, after wkv and before consumers."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax
from jax.experimental import pallas as pl

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/vllm_torchtpu/models/vllm/deepseek_v41/common/engram.py"
TOKENS, MULT, DIM = 288, 4, 5120
EPS, CLAMP = 1e-20, 1e-6
RTOL, ATOL = 1 / 128, 2e-6


def original_tail(out, source=SOURCE, *, dispatch=False, namespace=None):
    class BindAttributes(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                names = {"q_weight": "qw", "k_weight": "kw"}
                constants = {
                    "hc_mult": MULT,
                    "dim": DIM,
                    "eps": EPS,
                    "clamp_value": CLAMP,
                }
                if node.attr in names:
                    return ast.Name(id=names[node.attr], ctx=ast.Load())
                return ast.Constant(value=constants[node.attr])
            return self.generic_visit(node)

    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Engram"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    selected, active = [], False
    for node in forward.body:
        if (
            dispatch
            and isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "kv" for t in node.targets)
        ):
            active = True
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "keys" for t in node.targets
        ):
            active = True
        if active:
            selected.append(ast.unparse(BindAttributes().visit(node)))
    assert selected and selected[-1].startswith("return ")
    text = (
        "def tail(hidden_states, kv, qw, kw, token_mask):\n"
        + "\n".join("    " + line for part in selected for line in part.splitlines())
        + "\n"
    )
    (out / ("candidate-tail.py" if dispatch else "original-tail.py")).write_text(text)
    import torch

    scope = {"torch": torch, **(namespace or {})}
    exec(compile(text, str(source), "exec"), scope)
    return scope["tail"]


def fused_tail(
    hidden: jax.Array,
    kv: jax.Array,
    qw: jax.Array,
    kw: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Tile32 candidate with the actual model constants and BF16 inputs."""
    tile = 32

    def kernel(hr, kr, qr, wr, mr, output):
        h = hr[...].astype(jnp.float32)
        keys = kr[:, : MULT * DIM].reshape(tile, MULT, DIM).astype(jnp.float32)
        values = kr[:, MULT * DIM :].astype(jnp.float32)
        q, w = qr[...].astype(jnp.float32), wr[...].astype(jnp.float32)
        dot = jnp.sum(h * q * w * keys, axis=-1)
        dot *= lax.rsqrt(jnp.mean(h * h, axis=-1) + EPS)
        dot *= lax.rsqrt(jnp.mean(keys * keys, axis=-1) + EPS) * DIM**-0.5
        root = jnp.sqrt(jnp.maximum(jnp.abs(dot), CLAMP))
        gate = jax.nn.sigmoid(jnp.where(dot < 0, -root, root)) * mr[...]
        output[...] = (h + gate[:, :, None] * values[:, None, :]).astype(hidden.dtype)

    return pl.pallas_call(
        kernel,
        grid=(TOKENS // tile,),
        in_specs=[
            pl.BlockSpec((tile, MULT, DIM), lambda i: (i, 0, 0)),
            pl.BlockSpec((tile, (MULT + 1) * DIM), lambda i: (i, 0)),
            pl.BlockSpec(qw.shape, lambda i: (0, 0)),
            pl.BlockSpec(kw.shape, lambda i: (0, 0)),
            pl.BlockSpec((tile, 1), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec((tile, MULT, DIM), lambda i: (i, 0, 0)),
        out_shape=jax.ShapeDtypeStruct(hidden.shape, hidden.dtype),
        name="dsv41_engram_torch_gate_residual",
    )(hidden, kv, qw, kw, mask[:, None])


def fixture(kind, seed=42):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(TOKENS, MULT, DIM)).astype(np.float32)
    kv = rng.normal(size=(TOKENS, (MULT + 1) * DIM)).astype(np.float32)
    q = rng.normal(size=(MULT, DIM)).astype(np.float32)
    k = rng.normal(size=(MULT, DIM)).astype(np.float32)
    mask = np.ones(TOKENS, np.float32)
    if kind == "masked":
        mask[::3] = 0
    elif kind == "empty":
        mask[:] = 0
    elif kind == "zero":
        h[:] = 0
        kv[:, : MULT * DIM] = 0
    elif kind == "tiny":
        h *= 1e-12
        kv[:, : MULT * DIM] *= 1e-12
    elif kind == "cancellation":
        h[:] = 1
        kv[:, : MULT * DIM] = np.tile(np.array([1, -1], np.float32), MULT * DIM // 2)
        kv[:, MULT * DIM :] = -2
        q[:] = 1
        k[:] = 1
    elif kind == "clamp_boundary":
        h[:] = 1
        kv[:, : MULT * DIM] = 1
        q[:] = 1
        k[:] = (
            np.array([-1.01, -0.99, 0.99, 1.01], np.float32)[:, None] * CLAMP / DIM**0.5
        )
    return tuple(
        x.astype(ml_dtypes.bfloat16).astype(np.float32) for x in (h, kv, q, k)
    ) + (mask,)


def reference(xs, masked):
    h, kv, q, k, mask = [x.astype(np.float64) for x in xs]
    keys = kv[:, : MULT * DIM].reshape(TOKENS, MULT, DIM)
    values = kv[:, MULT * DIM :]
    dot = np.sum(h * q * k * keys, axis=-1)
    dot /= np.sqrt(np.mean(h * h, axis=-1) + EPS)
    dot *= 1 / np.sqrt(np.mean(keys * keys, axis=-1) + EPS) / DIM**0.5
    signed_root = np.copysign(
        np.sqrt(np.maximum(np.abs(dot), CLAMP)), np.where(dot < 0, -1, 1)
    )
    gate = 1 / (1 + np.exp(-signed_root))
    if masked:
        gate *= mask[:, None]
    return (
        (h + gate[:, :, None] * values[:, None, :])
        .astype(ml_dtypes.bfloat16)
        .astype(np.float32)
    )


def compare(actual, expected):
    delta = np.abs(actual - expected)
    allowed = ATOL + RTOL * np.abs(expected)
    good = np.isfinite(actual) & np.isfinite(expected) & (delta <= allowed)
    return {
        "passed": bool(np.all(good)),
        "failed_elements": int(np.count_nonzero(~good)),
        "different_elements": int(np.count_nonzero(delta)),
        "max_abs": float(np.max(delta)),
        "max_tolerance_ratio": float(np.max(delta / allowed)),
        "failures": [
            {
                "index": list(index),
                "actual": float(actual[index]),
                "expected": float(expected[index]),
                "absolute_error": float(delta[index]),
                "allowed_error": float(allowed[index]),
            }
            for index in (
                tuple(int(i) for i in coordinate)
                for coordinate in np.argwhere(~good)[:8]
            )
        ],
    }


def main():
    global EPS, TOKENS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--normal-seeds", default="")
    parser.add_argument("--with-mhc-post", action="store_true")
    parser.add_argument("--production-caller", action="store_true")
    parser.add_argument("--compare-post-block32", action="store_true")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--fallback-eps", type=float)
    parser.add_argument("--tokens", type=int, default=288)
    args = parser.parse_args()
    if args.production_caller and args.baseline_source is None:
        parser.error("--production-caller requires --baseline-source")
    if args.compare_post_block32 and not (
        args.production_caller and args.with_mhc_post and args.tokens == 288
    ):
        parser.error("post-block32 requires the production caller, post and T288")
    TOKENS = args.tokens
    if args.fallback_eps is not None:
        EPS = args.fallback_eps
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import pallas, sync

    torch.set_num_threads(1)
    torch.empty(1, device="tpu").cpu()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    custom = pallas.jax_op(
        "pallas::dsv41_engram_torch_gate_residual_bench",
        fused_tail,
        mesh=mesh,
        input_partition_specs=(jax.sharding.PartitionSpec(),) * 5,
    )

    @custom.register_fake
    def fake(h, kv, qw, kw, mask):
        return torch.empty_like(h)

    native = original_tail(args.out, args.baseline_source or SOURCE)

    def candidate(h, kv, qw, kw, mask):
        if mask is None:
            mask = torch.ones(h.shape[0], dtype=torch.float32, device=h.device)
        return custom(h, kv, qw, kw, mask)

    integration_checks = {}
    if args.production_caller:
        sys.path.insert(0, str(ROOT / "src"))
        from types import ModuleType, SimpleNamespace

        # The kernel runtime does not install vLLM's unrelated model registry.
        # Preserve the real module search paths and load the bridge/context
        # unchanged, without executing only those two parent initializers.
        for name in ("vllm_torchtpu.models", "vllm_torchtpu.models.vllm"):
            if name not in sys.modules:
                package = ModuleType(name)
                package.__path__ = [str(ROOT / "src" / name.replace(".", "/"))]
                sys.modules[name] = package

        from vllm_torchtpu.models.vllm.deepseek_v41.common import engram_gate as bridge
        from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
            set_vllm_model_wrapper_context,
        )

        # Exercise the exact registration calls in the existing model setup
        # context; full model construction and weight loading remain outside scope.
        hooks = []
        with set_vllm_model_wrapper_context(mesh=mesh):
            for relative in ("v1/model.py", "v2/model.py", "v2/dspark.py"):
                source = SOURCE.parents[1] / relative
                tree = ast.parse(source.read_text())
                calls = [
                    n
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Name)
                    and n.value.func.id == "initialize_engram_gate"
                ]
                assert len(calls) == 1, relative
                exec(
                    compile(
                        ast.Module(body=calls, type_ignores=[]), str(source), "exec"
                    ),
                    vars(bridge),
                )
                hooks.append(relative)
            first = bridge._gate_op
            bridge.initialize_engram_gate()
            assert first is not None and first is bridge._gate_op
        integration_checks["model_registration_hooks"] = hooks
        integration_checks["registration_cache"] = True
        integration_checks["unrelated_model_registry_bypassed"] = True

        def tensor(shape, dtype=torch.bfloat16, device="tpu"):
            return SimpleNamespace(
                shape=shape, dtype=dtype, device=torch.device(device)
            )

        nominal = [
            tensor((288, 4, 5120)),
            tensor((288, 25600)),
            tensor((4, 5120)),
            tensor((4, 5120)),
            tensor((288,), torch.float32),
            1e-20,
            1e-6,
        ]
        assert bridge.supports_engram_gate(*nominal)
        assert bridge.supports_engram_gate(*nominal[:4], None, *nominal[5:])
        rejected = []
        for name, index, value in (
            ("token_count", 0, tensor((287, 4, 5120))),
            ("hidden_rank", 0, tensor((288, 20480))),
            ("kv_width", 1, tensor((288, 25599))),
            ("q_dtype", 2, tensor((4, 5120), torch.float32)),
            ("k_shape", 3, tensor((1, 5120))),
            ("hidden_dtype", 0, tensor((288, 4, 5120), torch.float32)),
            ("kv_dtype", 1, tensor((288, 25600), torch.float32)),
            ("mask_dtype", 4, tensor((288,), torch.bool)),
            ("mask_shape", 4, tensor((288, 1), torch.float32)),
            ("cpu", 0, tensor((288, 4, 5120), device="cpu")),
            ("eps", 5, 1e-6),
            ("clamp", 6, 1e-4),
        ):
            values = nominal.copy()
            values[index] = value
            assert not bridge.supports_engram_gate(*values), name
            rejected.append(name)
        bridge._gate_op = None
        assert not bridge.supports_engram_gate(*nominal)
        bridge._gate_op = first
        integration_checks["rejected_contracts"] = rejected + ["uninitialized"]
        candidate = original_tail(args.out, dispatch=True, namespace=vars(bridge))

    post_op = None
    candidate_post_op = None
    if args.with_mhc_post:
        sys.path.insert(0, str(ROOT / "src"))
        from vllm_torchtpu.kernels.deepseek_v4.mhc import post_kernel

        def post_jax(
            x: jax.Array,
            residual: jax.Array,
            post: jax.Array,
            comb: jax.Array,
        ) -> jax.Array:
            return post_kernel.mhc_post(x, residual, post, comb)

        post_op = pallas.jax_op(
            "pallas::dsv41_engram_original_mhc_post_bench",
            post_jax,
            mesh=mesh,
            input_partition_specs=(jax.sharding.PartitionSpec(),) * 4,
        )

        @post_op.register_fake
        def fake_post(x, residual, post, comb):
            return torch.empty_like(residual)

        candidate_post_op = post_op
        if args.compare_post_block32:
            # Keep the already-adopted Engram path identical on both sides.
            native = candidate

            def post32_jax(
                x: jax.Array,
                residual: jax.Array,
                post: jax.Array,
                comb: jax.Array,
            ) -> jax.Array:
                return post_kernel.mhc_post(
                    x, residual, post, comb, token_block_size=32
                )

            candidate_post_op = pallas.jax_op(
                "pallas::dsv41_engram_post32_bench",
                post32_jax,
                mesh=mesh,
                input_partition_specs=(jax.sharding.PartitionSpec(),) * 4,
            )
            candidate_post_op.register_fake(fake_post)

        def with_producer(fn, producer):
            def call(h, kv, qw, kw, mask, x, post, comb):
                return fn(producer(x, h, post, comb), kv, qw, kw, mask)

            return call

        native = with_producer(native, post_op)
        candidate = with_producer(candidate, candidate_post_op)
    funcs = {
        name: torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
        for name, fn in (("native", native), ("pallas", candidate))
    }
    result = {
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (SOURCE, Path(__file__))
        },
        "runtime": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torch-tpu", "jax", "jaxlib", "libtpu", "numpy")
        },
        "contract": {
            "shape": [TOKENS, MULT, DIM],
            "eps": EPS,
            "clamp": CLAMP,
            "rtol": RTOL,
            "atol": ATOL,
            "inputs": "BF16 h/kv/q/k; F32 mask or None",
        },
        "correctness": [],
        "timings": {},
        "with_mhc_post": args.with_mhc_post,
        "production_caller": args.production_caller,
        "compare_post_block32": args.compare_post_block32,
        "integration_checks": integration_checks,
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    def inputs(xs, masked):
        values = tuple(
            torch.from_numpy(a).to(dtype=torch.bfloat16, device="tpu") for a in xs[:4]
        ) + (torch.from_numpy(xs[4]).to("tpu") if masked else None,)
        if args.with_mhc_post:
            # Dynamic identity fixture: execute the unchanged full producer
            # while making its BF16 output independently known, including trim.
            extras = (
                torch.zeros((TOKENS, DIM), dtype=torch.bfloat16, device="tpu"),
                torch.zeros((TOKENS, MULT, 1), dtype=torch.float32, device="tpu"),
                torch.from_numpy(
                    np.tile(np.eye(MULT, dtype=np.float32), (TOKENS, 1, 1))
                ).to("tpu"),
            )
            values += extras
        return values

    save()
    cases = (
        ("normal", 42),
        ("normal", 137),
        ("masked", 42),
        ("empty", 42),
        ("zero", 42),
        ("tiny", 42),
        ("cancellation", 42),
        ("clamp_boundary", 42),
    )
    if args.diagnose:
        cases = (("normal", 137),)
    if args.normal_seeds:
        cases = tuple(("normal", int(seed)) for seed in args.normal_seeds.split(","))
    for kind, seed in cases:
        xs = fixture(kind, seed)
        for masked in (True, False):
            values = inputs(xs, masked)
            if args.production_caller:
                selected = bridge.supports_engram_gate(*values[:5], EPS, CLAMP)
                assert selected == (EPS == 1e-20 and TOKENS == 288)
            if post_op is not None:
                produced = post_op(values[5], values[0], values[6], values[7])
                np.testing.assert_array_equal(produced.cpu().float().numpy(), xs[0])
                if args.compare_post_block32:
                    produced = candidate_post_op(
                        values[5], values[0], values[6], values[7]
                    )
                    np.testing.assert_array_equal(produced.cpu().float().numpy(), xs[0])
            want = reference(xs, masked)
            outputs = {
                name: fn(*values).cpu().float().numpy() for name, fn in funcs.items()
            }
            row = {"kind": kind, "seed": seed, "masked": masked}
            row.update(
                {
                    name + "_vs_reference": compare(actual, want)
                    for name, actual in outputs.items()
                }
            )
            row["pallas_vs_native"] = compare(outputs["pallas"], outputs["native"])
            if args.compare_post_block32:
                np.testing.assert_array_equal(outputs["pallas"], outputs["native"])
                row["native_candidate_exact"] = True
            if args.diagnose:
                detailed = []
                failed_coordinates = {
                    tuple(failure["index"])
                    for metric in ("native_vs_reference", "pallas_vs_reference")
                    for failure in row[metric]["failures"]
                }
                for t, m, d in sorted(failed_coordinates):
                    h = xs[0][t, m].astype(np.longdouble)
                    keys = (
                        xs[1][t, : MULT * DIM]
                        .reshape(MULT, DIM)[m]
                        .astype(np.longdouble)
                    )
                    q, k = [a[m].astype(np.longdouble) for a in xs[2:4]]
                    dot = np.sum(h * q * k * keys)
                    dot /= np.sqrt(np.mean(h * h) + np.longdouble(EPS))
                    dot /= np.sqrt(np.mean(keys * keys) + np.longdouble(EPS)) * np.sqrt(
                        np.longdouble(DIM)
                    )
                    gate = 1 / (
                        1
                        + np.exp(
                            -np.copysign(
                                np.sqrt(max(abs(dot), np.longdouble(CLAMP))),
                                -1 if dot < 0 else 1,
                            )
                        )
                    )
                    value = np.longdouble(xs[1][t, MULT * DIM + d])
                    exact = h[d] + gate * value
                    detailed.append(
                        {
                            "index": [t, m, d],
                            "float64_reference": float(want[t, m, d]),
                            "longdouble_residual": float(exact),
                            "longdouble_gate": float(gate),
                            "longdouble_dot": float(dot),
                            "hidden": float(h[d]),
                            "value": float(value),
                            "native": float(outputs["native"][t, m, d]),
                            "pallas": float(outputs["pallas"][t, m, d]),
                        }
                    )
                row["diagnostic"] = detailed
            result["correctness"].append(row)
            save()
            print(json.dumps(row), flush=True)
    result["precision_passed"] = all(
        c[metric]["passed"]
        for c in result["correctness"]
        for metric in ("native_vs_reference", "pallas_vs_reference", "pallas_vs_native")
    )
    save()
    if args.diagnose:
        result["completed"] = True
        save()
        return
    # Failed precision is recorded first; timings remain diagnostic and never
    # promote a numerically failing candidate into a production default.
    for masked in (True, False):
        values = inputs(fixture("normal"), masked)

        def run(fn, values=values):
            output = fn(*values)
            sync.synchronize(tensors=[output], wait=True)

        for _ in range(20):
            for fn in funcs.values():
                run(fn)
        samples = {name: [] for name in funcs}
        for i in range(args.samples):
            order = list(funcs.items())
            for name, fn in order if i % 2 == 0 else order[::-1]:
                start = time.perf_counter_ns()
                run(fn)
                samples[name].append((time.perf_counter_ns() - start) / 1000)
        label = "masked" if masked else "unmasked"
        result["timings"][label] = {
            name: {
                "median_us": float(np.median(xs)),
                "p10_us": float(np.percentile(xs, 10)),
                "p90_us": float(np.percentile(xs, 90)),
                "samples_us": xs,
            }
            for name, xs in samples.items()
        }
        save()
        if args.profile:
            with jax.profiler.trace(str(args.out / ("profile-" + label))):
                for i in range(16):
                    for name, fn in funcs.items():
                        with jax.profiler.TraceAnnotation(name, step_num=i):
                            run(fn)
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
