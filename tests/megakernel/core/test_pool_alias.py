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
import itertools
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import export, lax
from jax._src import core as jax_core
from jax._src import state
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib import tpu
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.pallas.mosaic import lowering
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.extend.mlir import ir

from vllm_torchtpu.megakernel.core import pool_alias
from vllm_torchtpu.megakernel.core.ops import BF16, F32, FP8


def _require_tpu() -> jax.Device:
    """Return the first TPU device or skip when running on CPU."""
    try:
        tpu_devs = jax.devices("tpu")
    except RuntimeError:
        tpu_devs = []
    if not tpu_devs:
        pytest.skip("Requires TPU device")
    return tpu_devs[0]


def _sync_copy(src, dst, sem) -> None:
    """Issue and await a synchronous Pallas DMA copy."""
    cp = pltpu.make_async_copy(src, dst, sem)
    cp.start()
    cp.wait()


def _extract_mosaic_mlir(exported: export.Exported) -> str:
    """Decode the embedded `@tpu_custom_call` Mosaic MLIR module text."""
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


@pytest.mark.parametrize(
    "bad_dtype",
    [jnp.int4, jnp.uint4, jnp.float4_e2m1fn, jnp.bool_, jnp.float64, jnp.complex64],
)
def test_pool_alias_rejects_unsupported_dtypes(bad_dtype):
    """Sub-byte, >32-bit, and non-numeric dtypes raise ValueError explicitly."""
    with pytest.raises(ValueError, match="8-, 16-, or 32-bit numeric dtypes"):
        pool_alias.storage_size((32, 128), bad_dtype)

    def kernel(o_ref, pool_ref):
        del o_ref
        pool_alias.view(pool_ref, (32, 128), bad_dtype)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), F32),
        scratch_shapes=[pltpu.VMEM((32, 128), jnp.uint8)],
        interpret=True,
    )
    with pytest.raises(ValueError, match="8-, 16-, or 32-bit numeric dtypes"):
        jax.make_jaxpr(call)()


@pytest.mark.parametrize("bad_dtype", [jnp.int4, jnp.uint4, jnp.bool_])
def test_pool_alias_view_rejects_unsupported_source_dtypes(bad_dtype):
    """Source refs with sub-byte or non-numeric dtypes raise ValueError explicitly."""

    def kernel(o_ref, pool_ref):
        del o_ref
        pool_alias.view(pool_ref, (8, 128), F32)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), F32),
        scratch_shapes=[pltpu.VMEM((32, 128), bad_dtype)],
        interpret=True,
    )
    with pytest.raises(ValueError, match="8-, 16-, or 32-bit numeric dtypes"):
        jax.make_jaxpr(call)()


def test_pool_alias_rejects_rank1_and_negative_shapes():
    """Rank-1 or negative shapes raise ValueError rather than IndexError."""
    with pytest.raises(ValueError, match=">= 2 non-negative dims"):
        pool_alias.storage_size((128,), jnp.uint8)
    with pytest.raises(ValueError, match=">= 2 non-negative dims"):
        pool_alias.storage_size((-8, 128), F32)

    def kernel(o_ref, pool_ref):
        del o_ref
        pool_alias.view(pool_ref, (8, 128), F32)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), F32),
        scratch_shapes=[pltpu.VMEM((128,), jnp.uint8)],
        interpret=True,
    )
    with pytest.raises(ValueError, match="source ref with >= 2 positive dims"):
        jax.make_jaxpr(call)()


def test_pool_alias_effects_pass_check_jaxpr_and_survive_dce():
    """`ref_allocating` passes `check_jaxpr`, propagates effects, and defeats DCE."""

    def kernel(x_ref, o_ref, pool_ref):
        v_write = pool_alias.view(pool_ref, (32, 128), BF16)
        v_write[...] = x_ref[...]
        v_read = pool_alias.view(pool_ref, (32, 128), BF16)
        o_ref[...] = v_read[...]

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((32, 128), BF16),
        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=[pltpu.VMEM((64, 128), jnp.uint8)],
    )
    with jax.enable_checks(True):
        outer = jax.make_jaxpr(call)(jax.ShapeDtypeStruct((32, 128), BF16))
        inner = outer.eqns[0].params["jaxpr"]
        jax_core.check_jaxpr(inner)

    pool_var = inner.invars[2]
    assert state.ReadEffect(pool_var) in inner.effects
    assert state.WriteEffect(pool_var) in inner.effects

    before_names = [eqn.primitive.name for eqn in inner.eqns]
    dce_inner, used_inputs = pe.dce_jaxpr(inner, used_outputs=[])
    assert [eqn.primitive.name for eqn in dce_inner.eqns] == before_names
    assert before_names.count(pool_alias.reinterpret_cast_p.name) == 2
    assert used_inputs == [True, True, True]


@pytest.mark.tpu_test
@pytest.mark.parametrize(
    "dtype,target_shape",
    [
        (F32, (16, 256)),
        (BF16, (32, 256)),
        (FP8, (64, 256)),
        (BF16, (2, 16, 256)),
        (BF16, (2, 3, 16, 256)),
        (F32, (2, 2, 8, 128)),
    ],
)
def test_pool_alias_reinterpret_roundtrip_on_tpu(dtype, target_shape):
    """Cross-dtype `tpu.reinterpret_cast` compiles and aliases VMEM on real TPU."""
    tpu_dev = _require_tpu()
    pool_rows = pool_alias.storage_size(target_shape, jnp.dtype(dtype)) // 128

    def kernel(x_ref, o_ref, pool_ref, sem):
        v_write = pool_alias.view(pool_ref, target_shape, dtype)
        _sync_copy(x_ref, v_write, sem)
        if dtype == FP8:
            for r in range(0, target_shape[-2], 32):
                for c in range(0, target_shape[-1], 128):
                    chunk = v_write[r : r + 32, c : c + 128].astype(F32) + jnp.float32(
                        1.0
                    )
                    v_write[r : r + 32, c : c + 128] = chunk.astype(FP8)
        else:
            for slice_id, idx in enumerate(
                itertools.product(*(range(d) for d in target_shape[:-2]))
            ):
                offset = jnp.float32(slice_id + 1)
                v_write[idx] = (v_write[idx].astype(F32) + offset).astype(dtype)
        # Read back through a fresh alias to prove both alias the same VMEM.
        v_read = pool_alias.view(pool_ref, target_shape, dtype)
        _sync_copy(v_read, o_ref, sem)

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

    flat_vals = np.arange(int(np.prod(target_shape)), dtype=np.int32)
    base_np = (flat_vals % (7 if dtype == FP8 else 32)).reshape(target_shape)
    x_np = base_np.astype(np.float32)
    expected_np = x_np.copy()
    if dtype == FP8:
        expected_np += 1.0
    else:
        for slice_id, idx in enumerate(
            itertools.product(*(range(d) for d in target_shape[:-2]))
        ):
            expected_np[idx] += float(slice_id + 1)

    with jax.default_device(tpu_dev):
        x = jnp.asarray(x_np, dtype=dtype)
        out = jax.block_until_ready(jax.jit(call)(x))

    expected = expected_np.astype(jnp.dtype(dtype))
    np.testing.assert_array_equal(np.asarray(out), expected)


@pytest.mark.tpu_test
def test_pool_alias_multi_view_and_dynamic_ring_on_tpu():
    """Reuse one pool across dtypes/shapes and verify dynamic 2D/3D ring slices."""
    tpu_dev = _require_tpu()
    shape_f32 = (32, 128)
    shape_bf16 = (2, 32, 128)
    shape_fp8 = (64, 128)
    slot_rows, num_slots = 128, 4

    def kernel(
        x_f32,
        x_bf16,
        x_fp8,
        ring_bf16,
        ring_fp8,
        o_f32,
        o_bf16,
        o_fp8,
        o_ring_bf16,
        o_ring_fp8,
        shared_pool,
        pool2d,
        pool3d,
        sem,
    ):
        # 1) Reuse a single 16 KiB uint8 pool across F32 -> BF16 -> FP8 views.
        v_f32 = pool_alias.view(shared_pool, shape_f32, F32)
        _sync_copy(x_f32, v_f32, sem.at[0])
        v_f32[...] = v_f32[...] + jnp.float32(1.0)
        _sync_copy(pool_alias.view(shared_pool, shape_f32, F32), o_f32, sem.at[0])

        v_bf16 = pool_alias.view(shared_pool, shape_bf16, BF16)
        _sync_copy(x_bf16, v_bf16, sem.at[0])
        for i in range(2):
            v_bf16[i] = (v_bf16[i].astype(F32) + jnp.float32(2.0)).astype(BF16)
        _sync_copy(v_bf16, o_bf16, sem.at[0])

        v_fp8 = pool_alias.view(shared_pool, (128, 128), FP8)
        _sync_copy(x_fp8, v_fp8, sem.at[0])
        for r in range(0, 128, 32):
            v_fp8[r : r + 32, :] = (
                v_fp8[r : r + 32, :].astype(F32) + jnp.float32(1.0)
            ).astype(FP8)
        _sync_copy(v_fp8, o_fp8, sem.at[0])

        # 2) Write all dynamic ring slots via write-only aliases, then read back
        # via fresh aliases to verify slot offsets do not collide.
        def write_slot(step, _):
            v2 = pool_alias.view(
                pool2d.at[pl.ds(step * slot_rows, slot_rows)], shape_bf16, BF16
            )
            _sync_copy(ring_bf16.at[step], v2, sem.at[step])
            step_val = (step + 1).astype(F32)
            for i in range(2):
                v2[i] = (v2[i].astype(F32) + step_val).astype(BF16)

            v3 = pool_alias.view(pool3d.at[step, :64], shape_fp8, FP8)
            _sync_copy(ring_fp8.at[step], v3, sem.at[step])
            for r in range(0, 64, 32):
                v3[r : r + 32, :] = (
                    v3[r : r + 32, :].astype(F32) + jnp.float32(1.0)
                ).astype(FP8)
            return None

        def read_slot(step, _):
            v2 = pool_alias.view(
                pool2d.at[pl.ds(step * slot_rows, slot_rows)], shape_bf16, BF16
            )
            _sync_copy(v2, o_ring_bf16.at[step], sem.at[step])
            v3 = pool_alias.view(pool3d.at[step, :64], shape_fp8, FP8)
            _sync_copy(v3, o_ring_fp8.at[step], sem.at[step])
            return None

        lax.fori_loop(0, num_slots, write_slot, None)
        lax.fori_loop(0, num_slots, read_slot, None)

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(shape_f32, F32),
            jax.ShapeDtypeStruct(shape_bf16, BF16),
            jax.ShapeDtypeStruct((128, 128), FP8),
            jax.ShapeDtypeStruct((num_slots, *shape_bf16), BF16),
            jax.ShapeDtypeStruct((num_slots, *shape_fp8), FP8),
        ),
        in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 5,
        out_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 5,
        scratch_shapes=[
            pltpu.VMEM((128, 128), jnp.uint8),
            pltpu.VMEM((num_slots * slot_rows, 128), jnp.uint8),
            pltpu.VMEM((num_slots, slot_rows, 128), jnp.uint8),
            pltpu.SemaphoreType.DMA((num_slots,)),
        ],
    )

    x_f32_np = (np.arange(int(np.prod(shape_f32)), dtype=np.float32) % 32).reshape(
        shape_f32
    )
    x_bf16_np = (np.arange(int(np.prod(shape_bf16)), dtype=np.float32) % 16).reshape(
        shape_bf16
    )
    x_fp8_np = (np.arange(128 * 128, dtype=np.int32) % 7).reshape((128, 128))
    ring_bf16_np = np.stack(
        [
            ((np.arange(int(np.prod(shape_bf16))) + 3 * s) % 16).reshape(shape_bf16)
            for s in range(num_slots)
        ]
    ).astype(np.float32)
    ring_fp8_np = np.stack(
        [
            ((np.arange(int(np.prod(shape_fp8))) + s) % 7).reshape(shape_fp8)
            for s in range(num_slots)
        ]
    ).astype(np.float32)

    with jax.default_device(tpu_dev):
        outs = jax.block_until_ready(
            jax.jit(call)(
                jnp.asarray(x_f32_np, dtype=F32),
                jnp.asarray(x_bf16_np, dtype=BF16),
                jnp.asarray(x_fp8_np, dtype=FP8),
                jnp.asarray(ring_bf16_np, dtype=BF16),
                jnp.asarray(ring_fp8_np, dtype=FP8),
            )
        )

    out_f32, out_bf16, out_fp8, out_ring_bf16, out_ring_fp8 = outs
    np.testing.assert_array_equal(np.asarray(out_f32), x_f32_np + 1.0)
    np.testing.assert_array_equal(
        np.asarray(out_bf16), (x_bf16_np + 2.0).astype(jnp.dtype(BF16))
    )
    np.testing.assert_array_equal(
        np.asarray(out_fp8), (x_fp8_np.astype(np.float32) + 1.0).astype(jnp.dtype(FP8))
    )
    expected_ring_bf16 = ring_bf16_np + np.arange(1, num_slots + 1)[:, None, None, None]
    np.testing.assert_array_equal(
        np.asarray(out_ring_bf16), expected_ring_bf16.astype(jnp.dtype(BF16))
    )
    np.testing.assert_array_equal(
        np.asarray(out_ring_fp8), (ring_fp8_np + 1.0).astype(jnp.dtype(FP8))
    )
