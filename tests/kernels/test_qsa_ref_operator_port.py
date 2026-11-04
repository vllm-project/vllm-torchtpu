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
"""Functional and owned-Ref entries share packing, geometry and math bindings."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info

from vllm_torchtpu.kernels.sparse_gqa_prefill import grouped, slab_gather


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


@pytest.mark.parametrize("batch", [8, 16, 24])
def test_block_id_tile_is_actual_parallel_program(batch, info):
    ids = np.asarray(grouped._block_id_data(batch, 8))
    assert ids.shape == (batch // 8, info.num_sublanes, info.num_lanes)
    assert ids.dtype == np.int32
    spec = grouped._block_id_spec()
    for b in range(batch // 8):
        assert spec.index_map(b) == (b, 0, 0)
        np.testing.assert_array_equal(ids[b], b)


def test_ref_reader_loads_parallel_id_and_passes_owned_view(monkeypatch, info):
    seen = []
    view = object()
    destination = SimpleNamespace(
        shape=(4096, 4, 128), dtype=jnp.float8_e4m3fn, reshape=lambda *shape: view
    )
    monkeypatch.setattr(
        grouped, "_attention_block", lambda *a, **kw: seen.append((a, kw))
    )

    def build(kernel, **kw):
        def call(lengths, ids, q, gate):
            kernel(lengths, ids[1], q[1], gate[1], *[object() for _ in range(6)])
            return jnp.zeros(kw["out_shape"].shape, jnp.bfloat16)

        return call

    monkeypatch.setattr(grouped.pl, "pallas_call", build)
    q = jnp.zeros((16, 24, 128), jnp.bfloat16)
    grouped._attend_compact_ref(
        q,
        q,
        jnp.ones(16, jnp.int32),
        destination,
        kv_heads=2,
        ratio=4,
        query_group=8,
        source_group=8,
        tile_k=256,
        capacity=256,
        budget=17,
    )
    assert int(seen[0][0][0]) == 1
    assert seen[0][0][4] is view
    assert seen[0][1]["query_group"] == 8


@pytest.mark.parametrize("heads,kv_heads", [(24, 2), (10, 1)])
def test_shared_launch_keeps_cost_and_packing_with_only_ref_id_bytes(
    monkeypatch, info, heads, kv_heads
):
    calls = []

    def build(kernel, **kwargs):
        def call(*args):
            calls.append((kernel, kwargs, args))
            return jnp.zeros(kwargs["out_shape"].shape, kwargs["out_shape"].dtype)

        return call

    monkeypatch.setattr(grouped.pl, "pallas_call", build)
    q = jnp.arange(16 * heads * 128, dtype=jnp.float32).reshape(16, heads, 128)
    q = q.astype(jnp.bfloat16)
    gate = -q
    lengths = jnp.arange(16, dtype=jnp.int32)
    packing = 4
    head_axes = (2 * kv_heads + packing - 1) // packing
    compact = jnp.zeros((16 * 256, head_axes, packing, 128), jnp.float8_e4m3fn)
    destination = SimpleNamespace(
        shape=(16 * 256 * head_axes, packing, 128), dtype=compact.dtype
    )
    kwargs = dict(
        kv_heads=kv_heads,
        ratio=4,
        query_group=8,
        source_group=8,
        tile_k=256,
        capacity=256,
        budget=17,
    )
    ref = grouped._attend_compact_ref(q, gate, lengths, destination, **kwargs)
    assert ref.shape == q.shape
    (new,) = calls
    assert new[1]["name"] == "destination_ref_block_id_tc_attention"
    assert new[1]["grid_spec"].grid == (2,)
    # Derive QK/PV cardinality from the actual packed query operand.
    padded_heads = new[2][2].shape[-2] // kwargs["query_group"]
    score_elements = q.shape[0] * kv_heads * padded_heads * kwargs["capacity"]
    alpha_elements = q.shape[0] * kv_heads * padded_heads
    output_elements = q.shape[0] * kv_heads * padded_heads * q.shape[-1]
    cost = new[1]["cost_estimate"]
    assert cost.flops == 4 * score_elements * q.shape[-1]
    assert cost.transcendentals == score_elements + alpha_elements + output_elements
    assert cost.bytes_accessed == (
        sum(x.nbytes for x in new[2]) + compact.nbytes + output_elements * 2
    )
    assert new[2][0] is lengths
    np.testing.assert_array_equal(new[2][1], grouped._block_id_data(16, 8))


def test_external_writer_has_empty_output_and_same_source_inputs(monkeypatch, info):
    calls = []

    def build(kernel, **kwargs):
        def call(*args):
            calls.append((kernel, kwargs, args))
            if kwargs["out_type"] == ():
                return ()
            return jnp.zeros(kwargs["out_type"].shape, kwargs["out_type"].dtype)

        return call

    monkeypatch.setattr(slab_gather.pl, "kernel", build)
    slab = jnp.zeros((192, 8, 128), jnp.int32)
    cache = jax.ShapeDtypeStruct((4, 128, 1, 4, 128), jnp.float8_e4m3fn)
    kw = dict(
        budget=17,
        table_cols=7,
        ratio=4,
        query_group=8,
        tile_k=256,
        padding_rows=128,
        query_base=jnp.int32(64),
        chunk_queries=64,
    )
    ordinary = slab_gather._gather_from_slab(slab, cache, **kw)
    destination = object()
    assert slab_gather._gather_from_slab_ref(destination, slab, cache, **kw) is None
    assert ordinary.shape == (64 * 256, 1, 4, 128)
    old, new = calls
    assert new[1]["out_type"] == ()
    assert old[1]["name"] == "sc_prefill_whole_slab_window_gather"
    assert new[1]["name"] == "destination_ref_slab_writer"
    assert old[2][0] is new[2][0] is cache
    assert old[2][1] is new[2][1] is slab
    np.testing.assert_array_equal(old[2][2], new[2][2])
