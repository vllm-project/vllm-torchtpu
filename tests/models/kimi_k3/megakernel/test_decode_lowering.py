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
"""Deviceless TPU v7x lowering of the Kimi-K3 decode megakernel per mode."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import topologies
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel import config as kernel_config
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import decode_megakernel, load

LAYERS = 4
PAGES = 16
BATCH = 8
RANKS = 32
DCP_POOL_LAYERS = 24
# Weights stored as packed MXFP4 words and their per-block scales.
_UINT32_WEIGHTS = ("expert_gu", "expert_down")
_UINT8_WEIGHTS = ("expert_gus", "expert_ds")

# Decode options per mode: (attention_token_scatter, moe_token_scatter_output,
# moe_routed_reduce_scatter, sequence_parallel_pre_attention, aux_hidden_layers).
MODES = {
    "plain": (False, False, False, False, ()),
    "scatter": (True, False, False, False, ()),
    "scatter_moe_out": (True, True, False, False, ()),
    "scatter_moe_reduce_scatter": (True, True, True, False, ()),
    "full_sequence_parallel": (True, True, True, True, ()),
    "aux_hidden": (False, False, False, False, (2,)),
}


@pytest.fixture(scope="module")
def tp_mesh() -> Mesh:
    """Build a 32-device TPU v7x 2x2x4 mesh without TPU hardware."""
    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    return Mesh(devices, ("tp",))


def _abstract_weights() -> dict[str, jax.ShapeDtypeStruct]:
    """Per-rank weight shapes of a LAYERS-layer prefix in the kernel layout."""
    weights = {}
    for name, shape in load.selected_local_shapes(layers=LAYERS).items():
        if name in _UINT32_WEIGHTS:
            dtype = jnp.uint32
        elif name in _UINT8_WEIGHTS:
            dtype = jnp.uint8
        else:
            dtype = jnp.bfloat16
        weights[name] = jax.ShapeDtypeStruct(shape, dtype)
    return weights


@pytest.mark.parametrize("mode", sorted(MODES))
def test_decode_megakernel_lowers_to_tpu_custom_call(tp_mesh: Mesh, mode: str) -> None:
    """Each decode mode lowers to one Mosaic custom call on TPU v7x."""
    active = kernel_config.active()
    assert (active.attention_dp, active.dcp_group) == (1, RANKS), active
    scatter, moe_out, moe_reduce_scatter, sequence_parallel, aux_layers = MODES[mode]
    weights = _abstract_weights()

    def step(*arrays: jax.Array) -> object:
        tokens, positions, conv_state, rec_state, pool, block_tables = arrays[:6]
        return decode_megakernel.decode_local(
            dict(zip(weights, arrays[6:], strict=True)),
            (conv_state, rec_state, pool),
            tokens,
            positions,
            batch_size=BATCH,
            sequence_rows=1,
            layers=LAYERS,
            block_tables=block_tables,
            dcp_pool_map=(0,) * DCP_POOL_LAYERS,
            dcp_owner_table=tuple(range(RANKS)),
            attention_dp=1,
            attention_token_scatter=scatter,
            moe_token_scatter_output=moe_out,
            moe_routed_reduce_scatter=moe_reduce_scatter,
            sequence_parallel_pre_attention=sequence_parallel,
            aux_hidden_layers=aux_layers,
        )

    inputs = (
        jax.ShapeDtypeStruct((BATCH,), jnp.int32),
        jax.ShapeDtypeStruct((BATCH,), jnp.int32),
        jax.ShapeDtypeStruct((BATCH, LAYERS, 3, 3, 4, 128), jnp.float32),
        jax.ShapeDtypeStruct((BATCH, LAYERS, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((PAGES, 128, 4, 640), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((BATCH, PAGES), jnp.int32),
    )
    sharded = jax.shard_map(
        step,
        mesh=tp_mesh,
        in_specs=(P(),) * (len(inputs) + len(weights)),
        out_specs=P(),
        check_vma=False,
    )
    hlo = jax.jit(sharded).lower(*inputs, *weights.values()).as_text()
    assert "tpu_custom_call" in hlo
