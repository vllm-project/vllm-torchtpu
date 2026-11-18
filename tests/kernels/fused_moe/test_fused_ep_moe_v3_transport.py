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
"""Fused EP MoE v3's transport start program (``transport.start_transport``)
traced on one CPU device.

The program quantizes a shard's rows a QUANT_TILE_ROWS tile at a time and
writes each tile's scale column into the routing message; a shard whose rows
are not a whole number of tiles ends in a partial tile. The program returns
DMA semaphores, which the TPU interpreter cannot do, so these tests trace it:
the shapes are what is under test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v3 import transport
from vllm_torchtpu.kernels.fused_moe.v3.layout import MESH_AXIS

pytestmark = pytest.mark.cpu_test

HIDDEN, PAIRS = 256, 1024


def _trace_start(tokens):
    """The start program over `tokens` rows on a one-shard mesh, traced."""
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), (MESH_AXIS,))
    message_rows = transport.message_rows(tokens, PAIRS)

    def start(rows, message):
        return transport.start_transport(
            rows,
            message,
            scale_base=transport.scale_row_offset(PAIRS),
            num_shards=1,
        )[3]

    program = jax.shard_map(
        start,
        mesh=mesh,
        in_specs=(P(MESH_AXIS), P(MESH_AXIS)),
        out_specs=P(MESH_AXIS),
        check_vma=False,
    )
    return jax.make_jaxpr(program)(
        jax.ShapeDtypeStruct((tokens, HIDDEN), jnp.bfloat16),
        jax.ShapeDtypeStruct((message_rows, transport.MESSAGE_LANES), jnp.int32),
    )


@pytest.mark.parametrize("tokens", [128, 136, 192])
def test_start_traces_with_a_partial_last_tile(tokens):
    """136 and 192 rows end in a partial tile of 8 and 64 rows; each tile's
    scale column is a whole tile tall and fills one message row."""
    assert transport.quant_tiles(tokens) == -(-tokens // transport.QUANT_TILE_ROWS)
    _trace_start(tokens)
