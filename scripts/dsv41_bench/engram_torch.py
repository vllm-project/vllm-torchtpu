# SPDX-License-Identifier: Apache-2.0
"""Actual Torch Engram tail versus Pallas, after wkv and before consumers."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
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


def original_tail(out):
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

    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Engram"
    )
    forward = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    selected, active = [], False
    for node in forward.body:
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
    (out / "original-tail.py").write_text(text)
    import torch

    scope = {"torch": torch}
    exec(compile(text, str(SOURCE), "exec"), scope)
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
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
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

    native = original_tail(args.out)

    def candidate(h, kv, qw, kw, mask):
        if mask is None:
            mask = torch.ones(h.shape[0], dtype=torch.float32, device=h.device)
        return custom(h, kv, qw, kw, mask)

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
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    def inputs(xs, masked):
        return tuple(
            torch.from_numpy(a).to(dtype=torch.bfloat16, device="tpu") for a in xs[:4]
        ) + (torch.from_numpy(xs[4]).to("tpu") if masked else None,)

    save()
    for kind, seed in (
        ("normal", 42),
        ("normal", 137),
        ("masked", 42),
        ("empty", 42),
        ("zero", 42),
        ("tiny", 42),
        ("cancellation", 42),
        ("clamp_boundary", 42),
    ):
        xs = fixture(kind, seed)
        for masked in (True, False):
            values = inputs(xs, masked)
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
            result["correctness"].append(row)
            save()
            print(json.dumps(row), flush=True)
    result["precision_passed"] = all(
        c[metric]["passed"]
        for c in result["correctness"]
        for metric in ("native_vs_reference", "pallas_vs_reference", "pallas_vs_native")
    )
    save()
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
