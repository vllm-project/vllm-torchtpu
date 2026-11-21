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
"""Parity, Scoped VMEM sizing, and admission tests for the row-tiled fused
QKV+gate + Q/K RMSNorm + RoPE kernel against vLLM's eager path in plain JAX."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels.qk_norm_rope_gate import (
    MIN_PALLAS_QKV_BYTES,
    can_run_qk_norm_rope_gate,
    legal_row_tile,
    qk_norm_rope_gate,
    qk_norm_rope_gate_rows,
    reference,
    scoped_vmem_limit_bytes,
    unfused_qk_norm_rope_gate_rows,
)
from vllm_torchtpu.kernels.qk_norm_rope_gate.kernel import _vmem_tile_bytes

_VMEM_16M = 16 * 1024 * 1024
_VMEM_64M = 64 * 1024 * 1024
_VMEM_128M = 128 * 1024 * 1024


def _on_tpu():
    return pltpu.is_tpu_device()


def _operands(
    seed, rows, num_heads, num_kv_heads, head_dim, rotary_dim, dtype=jnp.bfloat16
):
    rng = np.random.default_rng(seed)
    total_cols = (2 * num_heads + 2 * num_kv_heads) * head_dim
    qkv = jnp.asarray(rng.standard_normal((rows, total_cols)), dtype)
    q_weight = jnp.asarray(1.0 + 0.1 * rng.standard_normal(head_dim), jnp.float32)
    k_weight = jnp.asarray(1.0 + 0.1 * rng.standard_normal(head_dim), jnp.float32)
    angles = rng.random((4096, rotary_dim // 2)) * 2 * np.pi
    cos_sin_cache = jnp.asarray(
        np.concatenate([np.cos(angles), np.sin(angles)], axis=1), jnp.bfloat16
    )
    positions = jnp.asarray(rng.integers(0, 4096, size=(rows,)), jnp.int32)
    return qkv, q_weight, k_weight, cos_sin_cache, positions


def _assert_parity(
    rows,
    geometry,
    dtype=jnp.bfloat16,
    *,
    interpret: bool,
    vmem_capacity_bytes: int | None = None,
    num_sublanes: int | None = None,
):
    num_heads, num_kv_heads, head_dim, rotary_dim = geometry
    operands = _operands(
        rows * 3 + num_heads,
        rows,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        dtype=dtype,
    )
    kwargs = dict(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=1e-6,
    )
    got = qk_norm_rope_gate(
        *operands,
        vmem_capacity_bytes=vmem_capacity_bytes,
        num_sublanes=num_sublanes,
        interpret=interpret,
        **kwargs,
    )
    want = reference(*operands, **kwargs)
    tol = 2e-2 if dtype == jnp.bfloat16 else 1e-5
    for name, actual, expected in zip(("q", "k", "v", "gate"), got, want):
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected.dtype, name
        if name == "v":
            assert jnp.array_equal(actual, expected)
        else:
            np.testing.assert_allclose(
                np.asarray(actual, np.float32),
                np.asarray(expected, np.float32),
                atol=tol,
                rtol=tol,
                err_msg=name,
            )


def test_legal_row_tile_and_admission():
    assert legal_row_tile(8, 32) == 8
    assert legal_row_tile(24, 32) == 24
    assert legal_row_tile(40, 32) == 8
    assert legal_row_tile(48, 32) == 24
    assert legal_row_tile(384, 32) == 32
    assert legal_row_tile(48, 64, num_sublanes=16) == 48
    assert legal_row_tile(48, 32, num_sublanes=16) == 16
    with pytest.raises(ValueError, match="num_sublanes must be a positive multiple"):
        legal_row_tile(16, 32, num_sublanes=4)
    with pytest.raises(ValueError, match="rows must be a positive multiple of 8"):
        legal_row_tile(13, 32)
    with pytest.raises(ValueError, match="rows must be a positive multiple of 16"):
        legal_row_tile(24, 32, num_sublanes=16)
    with pytest.raises(ValueError, match="max_tile must be a positive multiple of 8"):
        legal_row_tile(32, 12)

    # Aligned large prompt rows fit in constant tiled VMEM across v5e/v6e/v7x.
    assert can_run_qk_norm_rope_gate(
        8192, 32, 2, 256, 64, dtype_itemsize=2, vmem_capacity_bytes=_VMEM_64M
    )
    assert can_run_qk_norm_rope_gate(
        2048, 32, 2, 256, 64, dtype_itemsize=2, vmem_capacity_bytes=_VMEM_16M
    )
    # Small unaligned rows (<= max(32, 4 * num_sublanes)) fit in whole-tensor VMEM:
    # <= 32 on 8-sublane chips, <= 64 on 16-sublane TPU_8T (admitting rows=40).
    assert can_run_qk_norm_rope_gate(
        13, 32, 2, 256, 64, dtype_itemsize=2, vmem_capacity_bytes=_VMEM_16M
    )
    assert not can_run_qk_norm_rope_gate(
        33, 4, 1, 256, 64, dtype_itemsize=2, vmem_capacity_bytes=_VMEM_64M
    )
    assert can_run_qk_norm_rope_gate(
        40,
        4,
        1,
        256,
        64,
        dtype_itemsize=2,
        vmem_capacity_bytes=_VMEM_128M,
        num_sublanes=16,
        chip_version=pltpu.ChipVersion.TPU_8T,
    )
    assert not can_run_qk_norm_rope_gate(
        65,
        4,
        1,
        256,
        64,
        dtype_itemsize=2,
        vmem_capacity_bytes=_VMEM_128M,
        num_sublanes=16,
        chip_version=pltpu.ChipVersion.TPU_8T,
    )
    assert not can_run_qk_norm_rope_gate(
        2049, 4, 1, 256, 64, dtype_itemsize=2, vmem_capacity_bytes=_VMEM_64M
    )
    assert not can_run_qk_norm_rope_gate(
        16, 4, 1, 64, 64, dtype_itemsize=2, vmem_capacity_bytes=_VMEM_64M
    )
    assert not can_run_qk_norm_rope_gate(
        16,
        4,
        1,
        256,
        64,
        dtype_itemsize=2,
        vmem_capacity_bytes=_VMEM_64M,
        num_sublanes=4,
    )


def test_scoped_vmem_limit_bytes_and_multi_chip_abstract_lowering():
    bytes_8 = _vmem_tile_bytes(
        8, 32, 2, 256, 64, 2, buffer_count=1, num_sublanes=8, stacked=True
    )
    bytes_384 = _vmem_tile_bytes(
        64, 32, 2, 256, 64, 2, buffer_count=2, num_sublanes=8, stacked=True
    )
    assert scoped_vmem_limit_bytes(bytes_8, _VMEM_64M) == 4 * 1024 * 1024
    limit_384 = scoped_vmem_limit_bytes(
        bytes_384, _VMEM_64M, chip_version=pltpu.ChipVersion.TPU_7X
    )
    assert limit_384 is not None and 16 * 1024 * 1024 < limit_384 < 32 * 1024 * 1024
    # TPU v6e defaults to 32 MiB Scoped VMEM (128 MiB physical VMEM), so a ~16.7 MiB
    # cap is emitted instead of falling back to the full 32 MiB default.
    assert (
        scoped_vmem_limit_bytes(
            bytes_384, _VMEM_128M, chip_version=pltpu.ChipVersion.TPU_V6E
        )
        == limit_384
    )
    # TPU v5p (64 MiB physical) and TPU v5e (128 MiB physical) both default to
    # 16 MiB Scoped VMEM, returning None when need >= 16 MiB.
    for v5_chip, v5_vmem in (
        (pltpu.ChipVersion.TPU_V5P, _VMEM_64M),
        (pltpu.ChipVersion.TPU_V5E, _VMEM_128M),
    ):
        assert (
            scoped_vmem_limit_bytes(bytes_8, v5_vmem, chip_version=v5_chip)
            == 4 * 1024 * 1024
        )
        assert scoped_vmem_limit_bytes(bytes_384, v5_vmem, chip_version=v5_chip) is None
    # Gen 8 chips default to >= 64 MiB Scoped VMEM, emitting proportional caps > 32 MiB.
    limit_gen8 = scoped_vmem_limit_bytes(
        28 * 1024 * 1024, _VMEM_128M, chip_version=pltpu.ChipVersion.TPU_8T
    )
    assert limit_gen8 is not None and 32 * 1024 * 1024 < limit_gen8 < 64 * 1024 * 1024
    with pytest.raises(ValueError, match="positive"):
        scoped_vmem_limit_bytes(0, _VMEM_64M)

    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh), pl.pallas_export_experimental(dynamic_shapes=False):
        for rows, expected_limit in (
            (4, 4 * 1024 * 1024),
            (8, 4 * 1024 * 1024),
            (384, limit_384),
        ):
            total_cols = (2 * 32 + 2 * 2) * 256
            exported = jax.export.export(
                jax.jit(
                    lambda a, b, c, d: qk_norm_rope_gate_rows(
                        a,
                        b,
                        c,
                        d,
                        num_heads=32,
                        num_kv_heads=2,
                        head_dim=256,
                        rotary_dim=64,
                        eps=1e-6,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((rows, total_cols), jnp.bfloat16),
                jax.ShapeDtypeStruct((256,), jnp.float32),
                jax.ShapeDtypeStruct((256,), jnp.float32),
                jax.ShapeDtypeStruct((rows, 64), jnp.bfloat16),
            )
            mlir_str = str(exported.mlir_module())
            assert re.search(
                rf'scoped_memory_configs.*size(?:\\22|")?:\s*(?:\\22|")?{expected_limit}\b',
                mlir_str,
            ), (rows, expected_limit)

        # Verify Scoped-VMEM-capped tile downshifting across v5p, v5e, and v6e so
        # tile_rows never exceeds the chip's 16 MiB (v5p/v5e) or 32 MiB (v6e) cap.
        for chip, vmem, rows, heads, kv_heads, dtype, expected_tile in (
            (pltpu.ChipVersion.TPU_V5E, _VMEM_128M, 384, 32, 2, jnp.float32, 32),
            (pltpu.ChipVersion.TPU_V5E, _VMEM_128M, 64, 64, 4, jnp.bfloat16, 32),
            (pltpu.ChipVersion.TPU_V5P, _VMEM_64M, 384, 64, 4, jnp.bfloat16, 32),
            (pltpu.ChipVersion.TPU_V6E, _VMEM_128M, 64, 128, 8, jnp.bfloat16, 32),
        ):
            cols = (2 * heads + 2 * kv_heads) * 256
            exported_chip = jax.export.export(
                jax.jit(
                    lambda a, b, c, d, _chip=chip, _vmem=vmem, _h=heads, _kh=kv_heads: (
                        qk_norm_rope_gate_rows(
                            a,
                            b,
                            c,
                            d,
                            num_heads=_h,
                            num_kv_heads=_kh,
                            head_dim=256,
                            rotary_dim=64,
                            eps=1e-6,
                            vmem_capacity_bytes=_vmem,
                            num_sublanes=8,
                            chip_version=_chip,
                        )
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((rows, cols), dtype),
                jax.ShapeDtypeStruct((256,), jnp.float32),
                jax.ShapeDtypeStruct((256,), jnp.float32),
                jax.ShapeDtypeStruct((rows, 64), dtype),
            )
            chip_mlir = str(exported_chip.mlir_module())
            assert f"qk_norm_rope_gate_t{expected_tile}" in chip_mlir, chip
            assert "qk_norm_rope_gate_t64" not in chip_mlir, chip


def test_qk_norm_rope_gate_rows_rejects_bad_inputs():
    with pytest.raises(ValueError, match="must be 2D"):
        qk_norm_rope_gate_rows(
            jnp.zeros((8, 1, 1024), jnp.bfloat16),
            jnp.ones((128,), jnp.float32),
            jnp.ones((128,), jnp.float32),
            jnp.zeros((8, 64), jnp.float32),
            num_heads=2,
            num_kv_heads=2,
            head_dim=128,
            rotary_dim=64,
            eps=1e-6,
            interpret=True,
        )
    with pytest.raises(ValueError, match="width"):
        qk_norm_rope_gate_rows(
            jnp.zeros((8, 512), jnp.bfloat16),
            jnp.ones((128,), jnp.float32),
            jnp.ones((128,), jnp.float32),
            jnp.zeros((8, 64), jnp.float32),
            num_heads=2,
            num_kv_heads=2,
            head_dim=128,
            rotary_dim=64,
            eps=1e-6,
            interpret=True,
        )
    with pytest.raises(ValueError, match="max_tile"):
        qk_norm_rope_gate_rows(
            jnp.zeros((4, 1024), jnp.bfloat16),
            jnp.ones((128,), jnp.float32),
            jnp.ones((128,), jnp.float32),
            jnp.zeros((4, 64), jnp.float32),
            num_heads=2,
            num_kv_heads=2,
            head_dim=128,
            rotary_dim=64,
            eps=1e-6,
            max_tile=4,
            interpret=True,
        )
    with pytest.raises(ValueError, match="q_weight and k_weight"):
        qk_norm_rope_gate_rows(
            jnp.zeros((8, 1024), jnp.bfloat16),
            jnp.ones((64,), jnp.float32),
            jnp.ones((128,), jnp.float32),
            jnp.zeros((8, 64), jnp.float32),
            num_heads=2,
            num_kv_heads=2,
            head_dim=128,
            rotary_dim=64,
            eps=1e-6,
            interpret=True,
        )


# (4, 1, 256, 64) is Qwen3.5-397B at TP8; (1, 1, 256, 64) exercises single-head
# Q and K; (2, 2, 128, 128) rotates the full head_dim. Includes unaligned (1, 4, 13),
# odd-8 (8, 24, 40), and multi-tile (64, 96) row counts.
@pytest.mark.parametrize("rows", [1, 4, 8, 13, 24, 40, 64, 96])
@pytest.mark.parametrize(
    "geometry", [(4, 1, 256, 64), (1, 1, 256, 64), (2, 2, 128, 128)]
)
def test_parity(rows, geometry):
    _assert_parity(rows, geometry, interpret=True)


@pytest.mark.parametrize("rows", [8, 40, 64])
def test_parity_tp1_397b_and_float32(rows):
    _assert_parity(rows, (32, 2, 256, 64), dtype=jnp.bfloat16, interpret=True)
    _assert_parity(rows, (4, 1, 256, 64), dtype=jnp.float32, interpret=True)
    _assert_parity(
        rows,
        (4, 1, 256, 64),
        dtype=jnp.bfloat16,
        num_sublanes=16,
        vmem_capacity_bytes=_VMEM_128M,
        interpret=True,
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("rows", [1, 4, 8, 13, 24, 40, 64, 384, 2048])
def test_tpu_parity_prompt_and_steady_rows(rows):
    if not _on_tpu():
        pytest.skip("requires TPU")
    _assert_parity(rows, (32, 2, 256, 64), dtype=jnp.bfloat16, interpret=False)


@pytest.mark.tpu_test
def test_tpu_parity_full_rotary_float32_and_vmem_downshift():
    if not _on_tpu():
        pytest.skip("requires TPU")
    _assert_parity(24, (2, 2, 128, 128), dtype=jnp.bfloat16, interpret=False)
    _assert_parity(24, (4, 1, 256, 64), dtype=jnp.float32, interpret=False)
    _assert_parity(
        64,
        (32, 2, 256, 64),
        dtype=jnp.bfloat16,
        vmem_capacity_bytes=_VMEM_16M,
        interpret=False,
    )


def test_min_pallas_qkv_bytes_crossover_and_validation():
    """Below MIN_PALLAS_QKV_BYTES, qk_norm_rope_gate_rows lowers inline without
    pallas_call (including unaligned small TP=8 shapes); at or above the cutoff
    it emits the row-tiled Pallas kernel."""
    assert MIN_PALLAS_QKV_BYTES == 1024 * 1024
    assert not can_run_qk_norm_rope_gate(
        16,
        32,
        2,
        256,
        64,
        vmem_capacity_bytes=_VMEM_64M,
        min_pallas_qkv_bytes=-1,
    )
    # 33 rows at TP=8 (4 Q, 1 KV, D=256 bf16 = 168,960 B < 1 MiB) is admitted via
    # unfused_qk_norm_rope_gate_rows even though 33 > MAX_UNALIGNED_ROWS (32).
    assert can_run_qk_norm_rope_gate(
        33,
        4,
        1,
        256,
        64,
        vmem_capacity_bytes=_VMEM_64M,
        min_pallas_qkv_bytes=MIN_PALLAS_QKV_BYTES,
    )
    # 209 rows at TP=8 (1,070,080 B >= 1 MiB) is unaligned > 32, so rejected.
    assert not can_run_qk_norm_rope_gate(
        209,
        4,
        1,
        256,
        64,
        vmem_capacity_bytes=_VMEM_64M,
        min_pallas_qkv_bytes=MIN_PALLAS_QKV_BYTES,
    )

    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh), pl.pallas_export_experimental(dynamic_shapes=False):
        for rows, expect_pallas in ((24, False), (32, True)):
            qkv, q_w, k_w, cos_sin_cache, pos = _operands(
                seed=100 + rows,
                rows=rows,
                num_heads=32,
                num_kv_heads=2,
                head_dim=256,
                rotary_dim=64,
                dtype=jnp.bfloat16,
            )
            exported = jax.export.export(
                jax.jit(
                    lambda a, b, c, d: qk_norm_rope_gate_rows(
                        a,
                        b,
                        c,
                        d,
                        num_heads=32,
                        num_kv_heads=2,
                        head_dim=256,
                        rotary_dim=64,
                        eps=1e-6,
                        vmem_capacity_bytes=_VMEM_64M,
                        num_sublanes=8,
                        chip_version=pltpu.ChipVersion.TPU_7X,
                        min_pallas_qkv_bytes=MIN_PALLAS_QKV_BYTES,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct(qkv.shape, jnp.bfloat16),
                jax.ShapeDtypeStruct((256,), jnp.float32),
                jax.ShapeDtypeStruct((256,), jnp.float32),
                jax.ShapeDtypeStruct((rows, 64), jnp.bfloat16),
            )
            mlir_str = str(exported.mlir_module())
            has_pallas = (
                "qk_norm_rope_gate_t" in mlir_str
                or "qk_norm_rope_gate_whole" in mlir_str
            )
            assert has_pallas is expect_pallas

    for rows in (24, 32):
        qkv, q_w, k_w, cos_sin_cache, pos = _operands(
            seed=100 + rows,
            rows=rows,
            num_heads=32,
            num_kv_heads=2,
            head_dim=256,
            rotary_dim=64,
            dtype=jnp.bfloat16,
        )
        got = qk_norm_rope_gate(
            qkv,
            q_w,
            k_w,
            cos_sin_cache,
            pos,
            num_heads=32,
            num_kv_heads=2,
            head_dim=256,
            rotary_dim=64,
            eps=1e-6,
            vmem_capacity_bytes=_VMEM_64M,
            num_sublanes=8,
            min_pallas_qkv_bytes=MIN_PALLAS_QKV_BYTES,
            interpret=True,
        )
        want = reference(
            qkv,
            q_w,
            k_w,
            cos_sin_cache,
            pos,
            num_heads=32,
            num_kv_heads=2,
            head_dim=256,
            rotary_dim=64,
            eps=1e-6,
        )
        for actual, expected in zip(got, want):
            np.testing.assert_allclose(
                np.asarray(actual, np.float32),
                np.asarray(expected, np.float32),
                atol=3e-2,
                rtol=2e-2,
            )

    with pytest.raises(ValueError, match="min_pallas_qkv_bytes"):
        qk_norm_rope_gate_rows(
            jnp.zeros((8, 1024), jnp.bfloat16),
            jnp.ones((128,), jnp.float32),
            jnp.ones((128,), jnp.float32),
            jnp.zeros((8, 64), jnp.float32),
            num_heads=2,
            num_kv_heads=2,
            head_dim=128,
            rotary_dim=64,
            eps=1e-6,
            min_pallas_qkv_bytes=-1,
            interpret=True,
        )
    with pytest.raises(ValueError, match="width"):
        unfused_qk_norm_rope_gate_rows(
            jnp.zeros((8, 512), jnp.bfloat16),
            jnp.ones((128,), jnp.float32),
            jnp.ones((128,), jnp.float32),
            jnp.zeros((8, 64), jnp.float32),
            num_heads=2,
            num_kv_heads=2,
            head_dim=128,
            rotary_dim=64,
            eps=1e-6,
        )
