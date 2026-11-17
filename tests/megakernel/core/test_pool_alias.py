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

import base64
import json

import jax
import jax.numpy as jnp
import pytest
from jax import export, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.extend.mlir import ir

from vllm_torchtpu.megakernel.core import pool_alias
from vllm_torchtpu.megakernel.core.ops import BF16, F32, FP8


def _extract_mosaic_mlir(exported: export.Exported) -> str:
    """Decode the embedded `@tpu_custom_call` Mosaic MLIR module text."""
    from jax._src.interpreters import mlir
    from jax._src.lib import tpu

    with mlir.make_ir_context() as ctx:
        ctx.allow_unregistered_dialects = True
        tpu.register_dialect(ctx)
        hlo_mod = ir.Module.parse(exported.mlir_module(), context=ctx)
        bodies: list[str] = []

        def _visit(region: ir.Region) -> None:
            for block in region.blocks:
                for op in block.operations:
                    if op.name == "stablehlo.custom_call" and (
                        "backend_config" in op.attributes
                    ):
                        cfg = json.loads(
                            ir.StringAttr(op.attributes["backend_config"]).value
                        )
                        body_b64 = cfg.get("custom_call_config", {}).get("body")
                        if body_b64:
                            bodies.append(
                                str(
                                    ir.Module.parse(
                                        base64.b64decode(body_b64), context=ctx
                                    )
                                )
                            )
                    for sub in op.regions:
                        _visit(sub)

        _visit(hlo_mod.operation.regions[0])
        assert len(bodies) == 1
        return bodies[0]


def test_pool_alias_storage_size():
    """Verify TPU 8x128 tile storage calculation."""
    assert pool_alias.storage_size((128, 256), jnp.dtype(FP8)) == 128 * 256 * 1
    assert pool_alias.storage_size((128, 256), jnp.dtype(BF16)) == 128 * 256 * 2
    assert pool_alias.storage_size((1, 256), jnp.dtype(F32)) == 8 * 256 * 4
    assert pool_alias.storage_size((8, 192), jnp.dtype(F32)) == 8 * 256 * 4


def test_pool_alias_does_not_monkeypatch_mosaic_lowering():
    """`pool_alias` registers its own primitive without patching `_reshape_memref`."""
    from jax._src.pallas.mosaic import core as tpu_core
    from jax._src.pallas.mosaic import lowering

    assert lowering._reshape_memref.__module__ == lowering.__name__
    assert (
        pool_alias.reinterpret_cast_p in lowering.lowering_rules[tpu_core.CoreType.TC]
    )


@pytest.mark.parametrize(
    "dtype,target_shape,expected_layout",
    [
        (F32, (16, 256), "#tpu.tiled<(8,128),[2,1]>"),
        (BF16, (32, 256), "#tpu.tiled<(16,128)(2,1),[2,1]>"),
        (FP8, (64, 256), "#tpu.tiled<(32,128)(4,1),[2,1]>"),
        (BF16, (2, 16, 256), "#tpu.tiled<(16,128)(2,1),[2,2,1]>"),
        (BF16, (2, 32, 256), "#tpu.tiled<(16,128)(2,1),[4,2,1]>"),
        (BF16, (2, 3, 16, 256), "#tpu.tiled<(16,128)(2,1),[6,2,2,1]>"),
        (F32, (2, 2, 8, 128), "#tpu.tiled<(8,128),[2,1,1,1]>"),
    ],
)
def test_pool_alias_reinterpret_mosaic_lowering(dtype, target_shape, expected_layout):
    """`pool_alias.view` lowers via `tpu.reinterpret_cast` with `#tpu.tiled`."""
    pool_rows = pool_alias.storage_size(target_shape, jnp.dtype(dtype)) // 128

    def kernel(x_ref, o_ref, pool_ref, sem):
        v = pool_alias.view(pool_ref, target_shape, dtype)
        for src, dst in ((x_ref, v), (v, o_ref)):
            cp = pltpu.make_async_copy(src, dst, sem)
            cp.start()
            cp.wait()

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(target_shape, dtype),
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[
            pltpu.VMEM((pool_rows, 128), jnp.uint8),
            pltpu.SemaphoreType.DMA(()),
        ],
    )
    exported = export.export(jax.jit(call), platforms=["tpu"])(
        jax.ShapeDtypeStruct(target_shape, dtype)
    )
    mosaic_ir = _extract_mosaic_mlir(exported)
    assert "tpu.reinterpret_cast" in mosaic_ir
    assert "tpu.erase_memref_layout" in mosaic_ir
    assert expected_layout in mosaic_ir


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


def test_pool_alias_view_rejects_mismatched_source_bytes():
    """`pool_alias.view` rejects a source slice whose size mismatches the target."""

    def kernel(o_ref, pool_ref):
        del o_ref
        pool_alias.view(pool_ref, (32, 128), BF16)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), F32),
        scratch_shapes=[pltpu.VMEM((32, 128), jnp.uint8)],
        interpret=True,
    )
    with pytest.raises(ValueError, match="contiguous tiled bytes"):
        jax.make_jaxpr(call)()


def test_pool_alias_dynamic_ring_and_control_flow_lowering():
    """`pool_alias.view` lowers dynamic 2D/3D ring slices across control flow."""
    shape_bf16, shape_fp8 = (2, 32, 128), (64, 128)
    slot_rows, num_slots = 128, 4

    def kernel(x_bf16_ref, x_fp8_ref, o_ref, pool2d, pool3d, sem):
        def body(step, _):
            slot = lax.rem(step, num_slots)
            v_bf16 = pool_alias.view(
                pool2d.at[pl.ds(slot * slot_rows, slot_rows)], shape_bf16, BF16
            )
            pltpu.make_async_copy(x_bf16_ref, v_bf16, sem.at[slot]).start()
            pltpu.make_async_copy(x_bf16_ref, v_bf16, sem.at[slot]).wait()
            v_bf16[...] = v_bf16[...] + jnp.bfloat16(1)

            v_fp8 = pool_alias.view(pool3d.at[slot, :64], shape_fp8, FP8)
            pltpu.make_async_copy(x_fp8_ref, v_fp8, sem.at[slot]).start()
            pltpu.make_async_copy(x_fp8_ref, v_fp8, sem.at[slot]).wait()

            @pl.when(step == num_slots - 1)
            def _():
                pltpu.make_async_copy(v_bf16, o_ref, sem.at[slot]).start()
                pltpu.make_async_copy(v_bf16, o_ref, sem.at[slot]).wait()

            return None

        lax.fori_loop(0, num_slots, body, None)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape_bf16, BF16),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=[
            pltpu.VMEM((num_slots * slot_rows, 128), jnp.uint8),
            pltpu.VMEM((num_slots, slot_rows, 128), jnp.uint8),
            pltpu.SemaphoreType.DMA((num_slots,)),
        ],
    )
    exported = export.export(jax.jit(call), platforms=["tpu"])(
        jax.ShapeDtypeStruct(shape_bf16, BF16),
        jax.ShapeDtypeStruct(shape_fp8, FP8),
    )
    mosaic_ir = _extract_mosaic_mlir(exported)
    assert "tpu.memref_slice" in mosaic_ir
    assert "#tpu.tiled<(16,128)(2,1),[2,1,1]>" in mosaic_ir
    assert "#tpu.tiled<(32,128)(4,1),[1,1]>" in mosaic_ir
