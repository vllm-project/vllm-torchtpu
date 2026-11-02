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
"""CPU checks for affine VMEM tile capacity and automatic tile policy."""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import prefill_tiles as tiles


def shapes(heads=24, kv_heads=2, dim=256, dtype=jnp.bfloat16):
    packing = 4 // jnp.dtype(dtype).itemsize
    query = jax.ShapeDtypeStruct((256, heads, dim), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct(
        (2, 1664, (2 * kv_heads + packing - 1) // packing, packing, dim), dtype
    )
    return query, cache


@pytest.fixture
def vmem(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return info.vmem_capacity_bytes * 31 // 32


@pytest.mark.parametrize(
    "heads,dim,kv_heads,packing,item_bytes",
    [
        (24, 256, 2, 2, 2),
        (48, 256, 4, 4, 1),
        (8, 128, 4, 2, 2),
    ],
)
@pytest.mark.parametrize("kv_tile", [128, 256])
def test_working_set_has_fixed_kv_buffers_and_per_query_row_state(
    heads, dim, kv_heads, packing, item_bytes, kv_tile
):
    native_heads = (2 * kv_heads + packing - 1) // packing * packing
    fixed = 2 * kv_tile * native_heads * dim * item_bytes
    per_row = 2 * heads * ((2 * dim + 2 * kv_tile + 2) * 4 + dim * 4)
    for rows in (0, 1, 8, 128, 256):
        assert (
            tiles._paired_working_set(
                heads, kv_heads, dim, packing, item_bytes, rows, kv_tile
            )
            == fixed + per_row * rows
        )


@pytest.mark.parametrize("rows", [8, 16, 32, 64, 128, 256])
def test_generic_capacity_exact_boundary_and_one_byte_below(monkeypatch, rows):
    query, cache = shapes(heads=8, kv_heads=4, dim=128)
    fixed = tiles._paired_working_set(8, 4, 128, 2, 2, 0, 256)
    per_row = tiles._paired_working_set(8, 4, 128, 2, 2, 1, 256) - fixed
    boundary = fixed + rows * per_row
    monkeypatch.setattr(tiles, "prefill_vmem_limit", lambda: boundary)
    assert tiles.get_sparse_prefill_tiles(query, cache, kv_heads=4, ratio=8) == (
        rows,
        256,
    )
    monkeypatch.setattr(tiles, "prefill_vmem_limit", lambda: boundary - 1)
    if rows == 8:
        with pytest.raises(ValueError, match="smallest query tile"):
            tiles.get_sparse_prefill_tiles(query, cache, kv_heads=4, ratio=8)
    else:
        assert tiles.get_sparse_prefill_tiles(query, cache, kv_heads=4, ratio=8) == (
            rows // 2,
            256,
        )


@pytest.mark.parametrize(
    "affordable,expected", [(8, 8), (15, 8), (33, 32), (255, 128), (511, 256)]
)
def test_power_of_two_and_subtile_cap_policy(monkeypatch, affordable, expected):
    query, cache = shapes(heads=8, kv_heads=4, dim=128)
    fixed = tiles._paired_working_set(8, 4, 128, 2, 2, 0, 256)
    per_row = tiles._paired_working_set(8, 4, 128, 2, 2, 1, 256) - fixed
    monkeypatch.setattr(
        tiles, "prefill_vmem_limit", lambda: fixed + affordable * per_row
    )
    qb, kt = tiles.get_sparse_prefill_tiles(query, cache, kv_heads=4, ratio=8)
    assert (qb, kt) == (expected, 256)
    assert qb % 8 == 0


@pytest.mark.parametrize("available_rows", [-1, 0, 7])
def test_capacity_below_existing_minimum_rejects(monkeypatch, available_rows):
    query, cache = shapes(heads=8, kv_heads=4, dim=128)
    fixed = tiles._paired_working_set(8, 4, 128, 2, 2, 0, 256)
    per_row = tiles._paired_working_set(8, 4, 128, 2, 2, 1, 256) - fixed
    monkeypatch.setattr(
        tiles, "prefill_vmem_limit", lambda: fixed + available_rows * per_row
    )
    with pytest.raises(ValueError, match="smallest query tile"):
        tiles.get_sparse_prefill_tiles(query, cache, kv_heads=4, ratio=8)


@pytest.mark.parametrize(
    "heads,kv_heads,dtype,expected",
    [
        (24, 2, jnp.bfloat16, (256, 256)),
        (24, 2, jnp.float8_e4m3fn, (256, 256)),
        (48, 4, jnp.bfloat16, (128, 128)),
        (48, 4, jnp.float8_e4m3fn, (128, 256)),
        (28, 2, jnp.bfloat16, (256, 128)),
        (64, 4, jnp.bfloat16, (64, 256)),
    ],
)
def test_larger_query_tile_then_paired_tie_policy(
    vmem, heads, kv_heads, dtype, expected
):
    query, cache = shapes(heads=heads, kv_heads=kv_heads, dtype=dtype)
    chosen = tiles.get_sparse_prefill_tiles(query, cache, kv_heads=kv_heads, ratio=4)
    assert chosen == expected
    assert (
        tiles._paired_working_set(
            heads, kv_heads, 256, cache.shape[3], jnp.dtype(dtype).itemsize, *chosen
        )
        <= vmem
    )


@pytest.mark.parametrize("query_block", [0, 7, 9, 257])
def test_paired_alignment_and_maximum_contract_stays_explicit(vmem, query_block):
    query, cache = shapes()
    assert not tiles.supports_paired_prefill(
        query, cache, kv_heads=2, ratio=4, query_block=query_block
    )


def test_paired_dtype_and_packing_guards_are_preserved(vmem):
    query, cache = shapes()
    wrong_query = jax.ShapeDtypeStruct(query.shape, jnp.float32)
    wrong_cache = jax.ShapeDtypeStruct((2, 1664, 1, 2, 256), jnp.bfloat16)
    assert not tiles.supports_paired_prefill(
        wrong_query, cache, kv_heads=2, ratio=4, query_block=128
    )
    assert not tiles.supports_paired_prefill(
        query, wrong_cache, kv_heads=2, ratio=4, query_block=128
    )


@pytest.mark.parametrize("which", ["heads", "dim", "packing", "kv_heads", "ratio"])
def test_positive_geometry_for_capacity_inversion(vmem, which):
    query, cache = shapes()
    kv_heads, ratio = 2, 4
    if which == "heads":
        query = jax.ShapeDtypeStruct((256, 0, 256), jnp.bfloat16)
    elif which == "dim":
        query = jax.ShapeDtypeStruct((256, 24, 0), jnp.bfloat16)
    elif which == "packing":
        cache = jax.ShapeDtypeStruct((2, 1664, 2, 0, 256), jnp.bfloat16)
    elif which == "kv_heads":
        kv_heads = 0
    else:
        ratio = 0
    with pytest.raises(ValueError, match="positive"):
        tiles.get_sparse_prefill_tiles(query, cache, kv_heads=kv_heads, ratio=ratio)
