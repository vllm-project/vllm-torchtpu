# SPDX-License-Identifier: Apache-2.0
"""Actual shifted mHC Torch bridge with public and terminal-post boundaries."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import sys
import time
import types
from pathlib import Path

import jax
import mhc_gate_chain as chain
import mhc_gate_layout as gate
import ml_dtypes
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", default=1000, type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--candidate",
        choices=("gate", "post32", "post3d", "post3d-tile"),
        default="gate",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(gate.ROOT / "src"))
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import pallas, sync

    torch.set_num_threads(1)
    torch.empty(1, device="tpu").cpu()
    # Match the production load-time packing, including contiguous storage.
    pack_source = gate.ROOT / "src/vllm_torchtpu/models/vllm/deepseek_v41/common/mhc.py"
    pack_node = next(
        n
        for n in ast.parse(pack_source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "pack_mix_weight"
    )
    pack_namespace = {"torch": torch}
    exec(
        compile(ast.Module([pack_node], type_ignores=[]), str(pack_source), "exec"),
        pack_namespace,
    )
    pack_weight = pack_namespace["pack_mix_weight"]
    # Skip unrelated model/layer registries; import the target bridge unchanged.
    for name in (
        "vllm_torchtpu.models",
        "vllm_torchtpu.models.vllm",
        "vllm_torchtpu.layers",
        "vllm_torchtpu.layers.adapter.custom_ops",
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(gate.ROOT / "src" / name.replace(".", "/"))]
            sys.modules[name] = package
    from vllm_torchtpu.kernels.deepseek_v4.mhc import post_kernel
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41 import (
        deepseek_v41_mhc_op as bridge,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    with set_vllm_model_wrapper_context(mesh=mesh):
        bridge.initialize_shifted_mhc()
    native_op = bridge.get_shifted_post_pre_rms_op()
    compact = gate.compact_gate(args.out, hbm_output=True)
    original = gate.original.shifted_post_pre_rms
    namespace = dict(original.__globals__)
    namespace["_pre_gates"] = compact
    modified = types.FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__
    )
    modified.__kwdefaults__ = original.__kwdefaults__
    bridge_namespace = dict(bridge._post_pre_rms_jax.__globals__)
    bridge_namespace["shifted_post_pre_rms"] = types.SimpleNamespace(
        is_supported_shape=gate.original.is_supported_shape,
        shifted_post_pre_rms=modified,
    )
    candidate_jax = types.FunctionType(
        bridge._post_pre_rms_jax.__code__, bridge_namespace, "candidate_bridge"
    )
    candidate_jax.__annotations__ = bridge._post_pre_rms_jax.__annotations__
    candidate_op = pallas.jax_op(
        "pallas::dsv41_mhc_gate_layout_torch_candidate",
        candidate_jax,
        mesh=mesh,
        input_partition_specs=(jax.sharding.PartitionSpec(),) * 9 + (None,) * 6,
    )

    @candidate_op.register_fake
    def fake(x, flat, post, comb, weight, scale, base, pre, *args):
        tokens, mult = pre.shape
        return (
            torch.empty_like(flat),
            scale.new_empty((tokens, mult, 1)),
            scale.new_empty((tokens, mult, mult)),
            flat.new_empty((tokens, flat.shape[1] // mult)),
            scale.new_empty((tokens, mult)),
        )

    def post_jax(
        x: jax.Array, residual: jax.Array, post: jax.Array, comb: jax.Array
    ) -> jax.Array:
        return post_kernel.mhc_post(x, residual, post, comb)

    post_op = pallas.jax_op(
        "pallas::dsv41_mhc_gate_original_terminal_post",
        post_jax,
        mesh=mesh,
        input_partition_specs=(jax.sharding.PartitionSpec(),) * 4,
    )

    @post_op.register_fake
    def fake_post(x, residual, post, comb):
        return torch.empty_like(residual)

    if args.candidate != "gate":
        from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import (
            deepseek_v4_mhc_op as post_bridge,
        )

        with set_vllm_model_wrapper_context(mesh=mesh):
            post_op = post_bridge.get_mhc_post_op()

        def post32_jax(
            x: jax.Array, residual: jax.Array, post: jax.Array, comb: jax.Array
        ) -> jax.Array:
            if args.candidate in ("post3d", "post3d-tile"):
                import mhc_post_layout

                return mhc_post_layout.post3d(
                    x, residual, post, comb, whole_tile=args.candidate == "post3d-tile"
                )
            return post_kernel.mhc_post(x, residual, post, comb, token_block_size=32)

        post32_op = pallas.jax_op(
            "pallas::dsv41_mhc_terminal_post32_bench",
            post32_jax,
            mesh=mesh,
            input_partition_specs=(jax.sharding.PartitionSpec(),) * 4,
        )
        post32_op.register_fake(fake_post)

    def wrap(op, terminal, terminal_op):
        def call(*xs):
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
            first = op(
                x0,
                flat,
                post,
                comb,
                w0,
                scale0,
                base0,
                pre,
                norm0,
                1e-20,
                1e-6,
                1e-6,
                2.0,
                20,
                1e-20,
            )
            second = op(
                x1,
                first[0],
                first[1],
                first[2],
                w1,
                scale1,
                base1,
                first[4],
                norm1,
                1e-20,
                1e-6,
                1e-6,
                2.0,
                20,
                1e-20,
            )
            if terminal:
                residual = terminal_op(
                    x0, second[0].view(288, 4, 5120), second[1], second[2]
                )
                return first[3], second[3], residual, second[4]
            return (first[3], *second)

        return call

    variants = (
        (("native", native_op, post_op), ("pallas", candidate_op, post_op))
        if args.candidate == "gate"
        else (("native", native_op, post_op), ("pallas", native_op, post32_op))
    )
    boundaries = (
        (("public", False), ("terminal", True))
        if args.candidate == "gate"
        else (("terminal", True),)
    )
    funcs = {
        label: {
            name: torch.compile(
                wrap(op, terminal, terminal_op),
                backend="tpu",
                fullgraph=True,
                dynamic=False,
            )
            for name, op, terminal_op in variants
        }
        for label, terminal in boundaries
    }
    result = {
        "runtime": {
            n: importlib.metadata.version(n)
            for n in ("jax", "jaxlib", "libtpu", "numpy", "torch", "torch-tpu")
        },
        "scope": (
            "two actual Torch shifted mHC bridge calls, with either all public "
            "outputs or original terminal mhc_post consuming post/comb; "
            "sublayer inputs are independent, no attention/MoE or full model"
        ),
        "candidate": args.candidate,
        "source_sha256": {
            str(p.relative_to(gate.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(chain.__file__),
                Path(gate.__file__),
                gate.SOURCE,
                Path(bridge.__file__),
                Path(post_kernel.__file__),
                pack_source,
            )
        },
        "correctness": [],
        "timings": {},
        "compile_llo": {},
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    def inputs(cpu, weights):
        tensors = []
        layouts = []
        for i, a in enumerate(cpu):
            if i in (5, 10):
                tensor = pack_weight(torch.from_numpy(weights[0 if i == 5 else 1]))
                np.testing.assert_array_equal(
                    tensor.float().numpy(), a.astype(np.float32)
                )
                assert tensor.is_contiguous()
            else:
                tensor = torch.from_numpy(a.astype(np.float32))
            tensor = tensor.to(
                device="tpu",
                dtype=(
                    torch.bfloat16 if a.dtype == ml_dtypes.bfloat16 else torch.float32
                ),
            )
            tensors.append(tensor)
            layouts.append(
                {
                    "index": i,
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                }
            )
        result["input_layouts"] = layouts
        result["weight_packing"] = (
            "original pack_mix_weight, exact independent pack match"
        )
        return tuple(tensors)

    save()
    for kind, seed in (
        ("normal", 42),
        ("normal", 137),
        ("zero", 42),
        ("tiny", 42),
        ("identity", 42),
        ("skew", 42),
    ):
        cpu, weights = chain.fixture(kind, seed)
        want = chain.reference(cpu, weights)
        residual = want[1].astype(np.float64).reshape(288, 4, 5120)
        terminal_ref = chain.bf16(
            np.einsum("tij,tid->tjd", want[3], residual)
            + want[2] * cpu[0].astype(np.float64)[:, None, :]
        )
        wanted = {"public": want, "terminal": (want[0], want[4], terminal_ref, want[5])}
        xs = inputs(cpu, weights)
        for label, variants in funcs.items():
            actual = {}
            errors = {}
            for name, fn in variants.items():
                before = set((args.out / "compiler").glob("*post-finalize-llo.txt"))
                actual[name] = [x.cpu().float().numpy() for x in fn(*xs)]
                added = (
                    set((args.out / "compiler").glob("*post-finalize-llo.txt")) - before
                )
                if added:
                    result["compile_llo"][f"{label}/{name}"] = {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in added
                    }
                errors[name] = []
                for i, (a, b) in enumerate(zip(actual[name], wanted[label])):
                    is_bf16 = i in ((0, 1, 4) if label == "public" else (0, 1, 2))
                    rtol, atol = (2e-2, 2e-2) if is_bf16 else (5e-3, 1e-5)
                    b = b.astype(np.float64)
                    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
                    errors[name].append(float(np.max(np.abs(a - b))))
            for a, b in zip(actual["native"], actual["pallas"]):
                np.testing.assert_array_equal(
                    np.ascontiguousarray(a).view(np.uint8),
                    np.ascontiguousarray(b).view(np.uint8),
                )
            row = {
                "label": label,
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
    xs = inputs(*chain.fixture("normal"))
    for label, variants in funcs.items():

        def run(fn):
            sync.synchronize(tensors=list(fn(*xs)), wait=True)

        for _ in range(20):
            for fn in variants.values():
                run(fn)
        samples = {name: [] for name in variants}
        for i in range(args.samples):
            ordered = list(variants.items())
            for name, fn in ordered if i % 2 == 0 else ordered[::-1]:
                start = time.perf_counter_ns()
                run(fn)
                samples[name].append((time.perf_counter_ns() - start) / 1000)
        result["timings"][label] = {
            n: {"median_us": float(np.median(v)), "samples_us": v}
            for n, v in samples.items()
        }
        save()
        if args.profile:
            with jax.profiler.trace(str(args.out / f"profile-{label}")):
                for i in range(16):
                    ordered = list(variants.items())
                    for name, fn in ordered if i % 2 == 0 else ordered[::-1]:
                        with jax.profiler.TraceAnnotation(name, step_num=i):
                            run(fn)
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
