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
"""Static dispatch and metadata preservation for the explicit slab producer."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.sparse_gqa_prefill import grouped, slab_gather


@pytest.fixture
def sc_info(monkeypatch):
    info = SimpleNamespace(
        num_lanes=128,
        num_sublanes=8,
        sparse_core=SimpleNamespace(num_lanes=16, vmem_capacity_bytes=512 * 1024),
    )
    monkeypatch.setattr(grouped.pltpu, "get_tpu_info", lambda: info)
    return info


def test_slab_fields_and_zero_padding(sc_info):
    blocks = jnp.arange(3 * 17, dtype=jnp.int32).reshape(3, 17)
    lengths = jnp.asarray([0, 5, 100], jnp.int32)
    table = jnp.arange(3 * 7, dtype=jnp.int32).reshape(3, 7)
    layout = slab_gather.field_layout(17, 7)
    packed = np.asarray(slab_gather.pack_query_metadata(blocks, lengths, table))
    flat = packed.reshape(3, -1)
    np.testing.assert_array_equal(flat[:, 0], lengths)
    np.testing.assert_array_equal(
        flat[:, layout["block_offset"] : layout["block_offset"] + 17], blocks
    )
    np.testing.assert_array_equal(
        flat[:, layout["table_offset"] : layout["table_offset"] + 7], table
    )
    occupied = np.zeros(flat.shape[1], bool)
    occupied[0] = True
    occupied[layout["block_offset"] : layout["block_offset"] + 17] = True
    occupied[layout["table_offset"] : layout["table_offset"] + 7] = True
    np.testing.assert_array_equal(flat[:, ~occupied], 0)
    assert packed.shape[1] % sc_info.num_sublanes == 0
    assert packed.shape[2] == sc_info.num_lanes
