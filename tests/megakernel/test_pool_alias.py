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
"""Unit tests for `pool_alias`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import export
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel import pool_alias
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8


def test_pool_alias_storage_size():
    """Verify TPU 8x128 tile storage calculation."""
    assert pool_alias.storage_size((128, 256), jnp.dtype(FP8)) == 128 * 256 * 1
    assert pool_alias.storage_size((128, 256), jnp.dtype(BF16)) == 128 * 256 * 2
    assert pool_alias.storage_size((1, 256), jnp.dtype(F32)) == 8 * 256 * 4
    assert pool_alias.storage_size((8, 192), jnp.dtype(F32)) == 8 * 256 * 4


def test_pool_alias_reinterpret_mosaic_lowering(monkeypatch):
    """Verify `pool_alias.Reinterpret` lowers to TPU MLIR ops (`jax==0.11.2`).

    Exercises the `jax._src.pallas.mosaic.lowering._reshape_memref` hook (pinned
    against `jax==0.11.2` / `jaxlib==0.11.2` in `pyproject.toml`). If a future JAX
    upgrade changes the signature or behavior of `_reshape_memref`,
    `tpu.reinterpret_cast`, or `tpu.erase_memref_layout`, this test will fail on CPU.
    """
    from jax._src.lib import tpu
    from jax._src.lib.mlir import ir
    from jax._src.pallas.mosaic import lowering

    # Start from a fresh install: earlier tests may already have installed the
    # hook, and the frozen reference kernel (loaded by other test modules)
    # installs its own pool_alias hook on top. monkeypatch restores all three.
    monkeypatch.setattr(pool_alias, "_installed", False)
    monkeypatch.setattr(pool_alias, "_original", pool_alias._original)
    monkeypatch.setattr(lowering, "_reshape_memref", lowering._reshape_memref)
    pool_alias._install()
    pool_alias._install()
    assert pool_alias._installed
    assert lowering._reshape_memref is pool_alias._lower

    cases = [
        (F32, (16, 256), "#tpu.tiled<(8,128),[2,1]>"),
        (BF16, (32, 256), "#tpu.tiled<(16,128)(2,1),[2,1]>"),
        (FP8, (64, 256), "#tpu.tiled<(32,128)(4,1),[2,1]>"),
        (BF16, (2, 16, 256), "#tpu.tiled<(16,128)(2,1),[2,2,1]>"),
        (BF16, (2, 32, 256), "#tpu.tiled<(16,128)(2,1),[4,2,1]>"),
        (BF16, (2, 3, 16, 256), "#tpu.tiled<(16,128)(2,1),[6,2,2,1]>"),
        (F32, (2, 2, 8, 128), "#tpu.tiled<(8,128),[2,1,1,1]>"),
    ]
    for dtype, target_shape, expected_layout in cases:
        nbytes = pool_alias.storage_size(target_shape, jnp.dtype(dtype))
        itemsize = jnp.dtype(dtype).itemsize
        flat_elems = nbytes // itemsize
        flat_shape = (1, 8, flat_elems // 8)

        with ir.Context() as ctx, ir.Location.unknown():
            tpu.register_dialect(ctx)
            module = ir.Module.create()
            vmem_attr = ir.Attribute.parse("#tpu.memory_space<vmem>")
            if dtype == F32:
                elem_type = ir.F32Type.get()
            elif dtype == BF16:
                elem_type = ir.BF16Type.get()
            else:
                elem_type = ir.Float8E4M3FNType.get()
            src_memref = ir.MemRefType.get(
                flat_shape, elem_type, memory_space=vmem_attr
            )
            with ir.InsertionPoint(module.body):
                func_type = ir.FunctionType.get([src_memref], [])
                func_op = ir.Operation.create(
                    "func.func",
                    attributes={
                        "sym_name": ir.StringAttr.get("test_reinterpret"),
                        "function_type": ir.TypeAttr.get(func_type),
                    },
                    regions=1,
                )
                block = func_op.regions[0].blocks.append(src_memref)
                with ir.InsertionPoint(block):
                    transform = pool_alias.Reinterpret(target_shape)
                    aval = jax.ShapeDtypeStruct(flat_shape, dtype)
                    out_val, out_shape = lowering._reshape_memref(
                        block.arguments[0], transform, aval, flat_shape
                    )
                    assert out_shape == target_shape
                    out_type = ir.MemRefType(out_val.type)
                    assert tuple(out_type.shape) == target_shape
                    assert out_type.memory_space == vmem_attr
                    rc_op = block.operations[0]
                    assert rc_op.name == "tpu.reinterpret_cast"
                    rc_type = ir.MemRefType(rc_op.result.type)
                    assert str(rc_type.layout) == expected_layout
                asm = str(func_op)
                assert "tpu.reinterpret_cast" in asm
                assert "tpu.erase_memref_layout" in asm
                assert expected_layout in asm


def test_pool_alias_hook_chains_to_hook_installed_after_import(monkeypatch):
    """`_install()` chains the `_reshape_memref` hook current at install time.

    Another hook installed after `pool_alias` is imported but before the first
    `view()` (e.g. a reference kernel's own pool_alias copy) must keep handling
    non-`Reinterpret` transforms; `Reinterpret` transforms reach `_lower`.
    """
    from jax._src.lib import tpu
    from jax._src.lib.mlir import ir
    from jax._src.pallas.mosaic import lowering

    calls = []

    def stand_in(ref, transform, aval, block_shape):
        calls.append(transform)
        return "stand-in", block_shape

    # monkeypatch restores the hook and pool_alias's install state afterwards.
    monkeypatch.setattr(pool_alias, "_installed", False)
    monkeypatch.setattr(pool_alias, "_original", pool_alias._original)
    monkeypatch.setattr(lowering, "_reshape_memref", stand_in)
    pool_alias._install()
    assert lowering._reshape_memref is pool_alias._lower
    assert pool_alias._original is stand_in

    other = object()
    assert lowering._reshape_memref("ref", other, None, (8, 128)) == (
        "stand-in",
        (8, 128),
    )
    assert calls == [other]

    target_shape, flat_shape = (16, 256), (1, 8, 512)
    with ir.Context() as ctx, ir.Location.unknown():
        tpu.register_dialect(ctx)
        module = ir.Module.create()
        vmem_attr = ir.Attribute.parse("#tpu.memory_space<vmem>")
        src_memref = ir.MemRefType.get(
            flat_shape, ir.F32Type.get(), memory_space=vmem_attr
        )
        with ir.InsertionPoint(module.body):
            func_op = ir.Operation.create(
                "func.func",
                attributes={
                    "sym_name": ir.StringAttr.get("test_chain"),
                    "function_type": ir.TypeAttr.get(
                        ir.FunctionType.get([src_memref], [])
                    ),
                },
                regions=1,
            )
            block = func_op.regions[0].blocks.append(src_memref)
            with ir.InsertionPoint(block):
                _, out_shape = lowering._reshape_memref(
                    block.arguments[0],
                    pool_alias.Reinterpret(target_shape),
                    jax.ShapeDtypeStruct(flat_shape, F32),
                    flat_shape,
                )
            assert out_shape == target_shape
            assert "tpu.reinterpret_cast" in str(func_op)
    assert calls == [other]


@pytest.mark.parametrize(
    "pool_shape,shape,dtype",
    [
        ((8, 128), (16, 64), F32),  # minor dim < 128: zero lane-tile stride
        ((12, 128), (8, 192), F32),  # minor dim not a multiple of 128
        ((16, 128), (2, 4, 256), F32),  # padded rows: 16 KiB view, 8 KiB slice
        ((16, 128), (2, 8, 128), BF16),  # bf16 tiles are 16 rows
    ],
)
def test_pool_alias_view_rejects_untileable_shapes(pool_shape, shape, dtype):
    """Element counts match, but the tiled view would alias storage past the slice."""

    def kernel(o_ref, pool_ref):
        del o_ref
        pool_alias.view(pool_ref, shape, dtype)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), F32),
        scratch_shapes=[pltpu.VMEM(pool_shape, dtype)],
        interpret=True,
    )
    with pytest.raises(ValueError, match="pads to"):
        jax.make_jaxpr(call)()


def test_pool_alias_view_reaches_hook_in_tpu_pallas_lowering(monkeypatch):
    """A TPU `pallas_call` using `view()` lowers `Reinterpret` through the hook.

    Cross-lowers for TPU on CPU via `jax.export`, so a JAX change to
    `_transform_ref` dispatch that bypasses `_reshape_memref` fails here instead of
    silently lowering to `tpu.memref_reshape`. libtpu layout legality is not checked.
    """
    from jax._src.lib.mlir import ir
    from jax._src.pallas.mosaic import lowering

    pool_alias._install()
    chained = lowering._reshape_memref
    layouts = []

    def spy(ref, transform, aval, block_shape):
        out = chained(ref, transform, aval, block_shape)
        if isinstance(transform, pool_alias.Reinterpret):
            alias = out[0].owner.operands[0]
            layouts.append(str(ir.MemRefType(alias.type).layout))
        return out

    monkeypatch.setattr(lowering, "_reshape_memref", spy)
    shape = (2, 32, 128)

    def kernel(x_ref, o_ref, pool_ref, sem):
        v = pool_alias.view(pool_ref, shape, BF16)
        for src, dst in ((x_ref, v), (v, o_ref)):
            cp = pltpu.make_async_copy(src, dst, sem)
            cp.start()
            cp.wait()

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, BF16),
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[pltpu.VMEM((128, 128), FP8), pltpu.SemaphoreType.DMA(())],
    )
    export.export(jax.jit(call), platforms=["tpu"])(jax.ShapeDtypeStruct(shape, BF16))
    assert layouts and set(layouts) == {"#tpu.tiled<(16,128)(2,1),[2,1,1]>"}
