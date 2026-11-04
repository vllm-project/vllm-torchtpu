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
"""Whole-query metadata storage retains chunk-local transport and fallback."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info

from vllm_torchtpu.kernels.sparse_gqa_prefill import grouped, owner_window, slab_gather


@pytest.fixture
def info(monkeypatch):
    value = SimpleNamespace(
        num_lanes=128,
        num_sublanes=8,
        sparse_core=SimpleNamespace(
            num_lanes=16, num_cores=4, num_subcores=2, vmem_capacity_bytes=512 * 1024
        ),
    )
    monkeypatch.setattr(grouped.pltpu, "get_tpu_info", lambda: value)
    monkeypatch.setattr(tpu_info, "get_tpu_info", lambda: value)
    return value


@pytest.mark.parametrize("base", [0, 64, 128])
def test_whole_output_is_one_chunk_and_descriptor_is_64_bytes(monkeypatch, info, base):
    seen = {}

    def build(kernel, *, out_type, **kwargs):
        seen["shape"] = out_type.shape

        def call(*inputs):
            seen["inputs"] = inputs
            return jnp.zeros(out_type.shape, out_type.dtype)

        return call

    monkeypatch.setattr(slab_gather.pl, "kernel", build)
    cache = jnp.zeros((4, 128, 1, 4, 128), jnp.float8_e4m3fn)
    slab = jnp.zeros((192, 8, 128), jnp.int32)
    result = slab_gather._gather_from_slab(
        slab,
        cache,
        budget=17,
        table_cols=7,
        ratio=4,
        query_group=8,
        tile_k=256,
        padding_rows=128,
        query_base=jnp.int32(base),
        chunk_queries=64,
    )
    assert result.shape == (64 * 256, 1, 4, 128)
    assert seen["inputs"][1] is slab
    descriptor = np.asarray(seen["inputs"][2])
    assert descriptor.nbytes == 64
    np.testing.assert_array_equal(descriptor, base)


@pytest.mark.parametrize("failure", ["nonshared", "resource"])
def test_unsupported_does_not_pack_and_preserves_original_kwargs(
    monkeypatch, info, failure
):
    if failure == "nonshared":
        info.sparse_core.num_lanes = 96
    else:
        info.sparse_core.vmem_capacity_bytes = 1

    def no_pack(*args):
        raise AssertionError("Unsupported path packed whole-query metadata")

    monkeypatch.setattr(slab_gather, "pack_query_metadata", no_pack)
    seen = []
    marker = object()
    monkeypatch.setattr(
        owner_window.adaptive, "is_supported_by_sc_window", lambda *a, **k: True
    )
    monkeypatch.setattr(
        owner_window.adaptive, "_tc", lambda *a, **k: seen.append((a, k)) or marker
    )
    q = jax.ShapeDtypeStruct((64, 24, 128), jnp.bfloat16)
    blocks = jax.ShapeDtypeStruct((64, 17), jnp.int32)
    n = jax.ShapeDtypeStruct((64,), jnp.int32)
    table = jax.ShapeDtypeStruct((64, 7), jnp.int32)
    cache = jax.ShapeDtypeStruct((4, 128, 1, 4, 128), jnp.float8_e4m3fn)
    starts = jax.ShapeDtypeStruct((3,), jnp.int32)
    args = (q, q, blocks, n, table, cache, starts)
    assert owner_window.owner_window_attention(*args, kv_heads=2, ratio=4) is marker
    assert seen == [(args, dict(kv_heads=2, ratio=4, kv_tile=256))]


@pytest.mark.parametrize("slack,expected", [(-1, False), (0, True)])
def test_whole_descriptor_resource_boundary(info, slack, expected):
    blocks = jax.ShapeDtypeStruct((8192, 512), jnp.int32)
    table = jax.ShapeDtypeStruct((8192, 158), jnp.int32)
    cache = jax.ShapeDtypeStruct((1265, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    required = slab_gather.buffer_bytes(512, 158, 256)["total"]
    descriptor = info.sparse_core.num_lanes * jnp.dtype(jnp.int32).itemsize
    info.sparse_core.vmem_capacity_bytes = required + descriptor + slack
    assert (
        grouped._supports_whole_slab(
            blocks, table, cache, query_group=8, tile_k=256, padding_rows=1664
        )
        is expected
    )


def test_measured_whole_geometry_still_admitted(info):
    blocks = jax.ShapeDtypeStruct((8192, 512), jnp.int32)
    table = jax.ShapeDtypeStruct((8192, 158), jnp.int32)
    cache = jax.ShapeDtypeStruct((1265, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    assert info.sparse_core.vmem_capacity_bytes == 512 * 1024
    assert grouped._supports_whole_slab(
        blocks, table, cache, query_group=8, tile_k=256, padding_rows=1664
    )


@pytest.mark.parametrize("group_delta,expected", [(-1, True), (0, True), (1, False)])
def test_whole_metadata_tensor_abi_boundary(info, group_delta, expected):
    from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
        SC_GATHER_MAX_CACHE_BYTES,
    )

    query_group = 8
    words = slab_gather.field_layout(512, 158)["stride_words"]
    row_bytes = words * jnp.dtype(jnp.int32).itemsize
    assert SC_GATHER_MAX_CACHE_BYTES % row_bytes == 0
    queries = SC_GATHER_MAX_CACHE_BYTES // row_bytes + group_delta * query_group
    blocks = jax.ShapeDtypeStruct((queries, 512), jnp.int32)
    table = jax.ShapeDtypeStruct((queries, 158), jnp.int32)
    cache = jax.ShapeDtypeStruct((2, 1664, 1, 4, 256), jnp.float8_e4m3fn)
    assert (
        grouped._supports_whole_slab(
            blocks, table, cache, query_group=query_group, tile_k=256, padding_rows=1664
        )
        is expected
    )
