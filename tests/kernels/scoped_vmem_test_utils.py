# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""StableHLO and bitwise parity helpers for scoped attention VMEM tests."""

import base64
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.lib import tpu as mosaic_tpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh
from jaxlib.mlir import ir
from jaxlib.mlir.passmanager import PassManager


def _walk_ops(op, fn):
    fn(op)
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                _walk_ops(child.operation, fn)


def _vmem_memref_bytes(ty) -> int:
    try:
        memref = ir.MemRefType(ty)
    except (TypeError, ValueError):
        return 0
    if "vmem" not in str(memref.memory_space):
        return 0
    num_elems = 1
    for dim in memref.shape:
        num_elems *= dim
    return num_elems * max(1, memref.element_type.width // 8)


def _declared_vmem_bytes(body_b64: str) -> int:
    ctx = ir.Context()
    mosaic_tpu.register_dialect(ctx)
    ctx.allow_unregistered_dialects = True
    with ctx, ir.Location.unknown():
        module = ir.Module.parse(base64.b64decode(body_b64))
        PassManager.parse("builtin.module(mosaic-serde{serialize=false})").run(
            module.operation
        )
        total = 0

        def visit(op):
            nonlocal total
            if op.name == "func.func":
                fn_type = ir.FunctionType(
                    ir.TypeAttr(op.attributes["function_type"]).value
                )
                total += sum(_vmem_memref_bytes(t) for t in fn_type.inputs)
            elif op.name == "memref.alloca":
                total += _vmem_memref_bytes(op.results[0].type)

        _walk_ops(module.operation, visit)
        return total


def _extract_tpu_custom_calls(fn, args) -> list[tuple[str, dict]]:
    lowered = jax.jit(fn).trace(*args).lower(lowering_platforms=("tpu",))
    module = lowered.compiler_ir("stablehlo")
    found: list[tuple[str, dict]] = []

    def visit(op):
        if op.name != "stablehlo.custom_call":
            return
        target = ir.StringAttr(op.attributes["call_target_name"]).value
        if target != "tpu_custom_call":
            return
        name = (
            ir.StringAttr(op.attributes["kernel_name"]).value
            if "kernel_name" in op.attributes
            else "?"
        )
        cfg = json.loads(ir.StringAttr(op.attributes["backend_config"]).value)
        found.append((name, cfg))

    _walk_ops(module.operation, visit)
    return found


def _subjaxprs(eqn):
    for param in eqn.params.values():
        for value in param if isinstance(param, tuple | list) else (param,):
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                yield inner


def _eqns(jaxpr, primitive: str):
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == primitive:
            yield eqn
        for inner in _subjaxprs(eqn):
            yield from _eqns(inner, primitive)


def _scoped_vmem_bytes(cfg: dict) -> int | None:
    sizes = [
        entry["size"]
        for entry in cfg.get("scoped_memory_configs", [])
        if entry.get("memory_space") == 1
    ]
    return sizes[0] if sizes else None


def _abstract_tpu(device_kind: str):
    return use_abstract_mesh(
        AbstractMesh((), (), (), abstract_device=AbstractDevice(device_kind, 1, "tpu"))
    )


def _sds(shape, dtype):
    return jax.ShapeDtypeStruct(tuple(shape), jnp.dtype(dtype))


def _require_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip(f"Requires TPU backend, found {jax.default_backend()!r}.")


def _assert_bitwise_equal(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(
        actual.view(np.uint8).reshape(-1), expected.view(np.uint8).reshape(-1)
    )
