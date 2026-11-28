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

import gc
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import topologies
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel import config as kernel_config
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import decode_megakernel, load

if TYPE_CHECKING:
    from collections.abc import Callable

LAYERS = 4
PAGES = 16
BATCH = 8
# reuse_row_plan activates at batch_size in (32, 64, 96, 128), exercising the
# int16 page-offset gathers in stage_planned_rows.
ROW_PLAN_BATCH_SIZE = 32
# Smallest KDA+MoE prefix that lowers and compiles stage_planned_rows on CPU.
ROW_PLAN_LAYERS = 2
# Decode recipe uses an MLA cache batch tile of 2 at batch 32.
ROW_PLAN_MLA_BATCH_TILE = 2
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


def _abstract_weights(layers: int = LAYERS) -> dict[str, jax.ShapeDtypeStruct]:
    """Per-rank weight shapes of a ``layers``-layer prefix in kernel layout."""
    weights = {}
    for name, shape in load.selected_local_shapes(layers=layers).items():
        if name in _UINT32_WEIGHTS:
            dtype = jnp.uint32
        elif name in _UINT8_WEIGHTS:
            dtype = jnp.uint8
        else:
            dtype = jnp.bfloat16
        weights[name] = jax.ShapeDtypeStruct(shape, dtype)
    return weights


def _abstract_inputs(batch_size: int, layers: int) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Abstract decode inputs and KV/recurrent state buffers for one step."""
    return (
        jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        jax.ShapeDtypeStruct((batch_size, layers, 3, 3, 4, 128), jnp.float32),
        jax.ShapeDtypeStruct((batch_size, layers, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((PAGES, 128, 4, 640), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((batch_size, PAGES), jnp.int32),
    )


def _make_sharded_decode(
    tp_mesh: Mesh,
    weights: dict[str, jax.ShapeDtypeStruct],
    batch_size: int,
    layers: int,
    decode_kwargs: dict[str, object],
) -> tuple[Callable[..., object], tuple[jax.ShapeDtypeStruct, ...]]:
    """Build abstract inputs and wrap ``decode_local`` in ``jax.shard_map``."""
    inputs = _abstract_inputs(batch_size, layers)

    def step(*arrays: jax.Array) -> object:
        """Run one local decode step inside ``jax.shard_map``."""
        tokens, positions, conv_state, rec_state, pool, block_tables = arrays[:6]
        return decode_megakernel.decode_local(
            dict(zip(weights, arrays[6:], strict=True)),
            (conv_state, rec_state, pool),
            tokens,
            positions,
            batch_size=batch_size,
            sequence_rows=1,
            layers=layers,
            block_tables=block_tables,
            dcp_pool_map=(0,) * DCP_POOL_LAYERS,
            dcp_owner_table=tuple(range(RANKS)),
            attention_dp=1,
            **decode_kwargs,
        )

    sharded = jax.shard_map(
        step,
        mesh=tp_mesh,
        in_specs=(P(),) * (len(inputs) + len(weights)),
        out_specs=P(),
        check_vma=False,
    )
    return sharded, inputs


@pytest.mark.parametrize("mode", sorted(MODES))
def test_decode_megakernel_lowers_to_tpu_custom_call(tp_mesh: Mesh, mode: str) -> None:
    """Each decode mode lowers to one Mosaic custom call on TPU v7x."""
    active = kernel_config.active()
    assert (active.attention_dp, active.dcp_group) == (1, RANKS), active
    scatter, moe_out, moe_reduce_scatter, sequence_parallel, aux_layers = MODES[mode]
    weights = _abstract_weights()
    sharded, inputs = _make_sharded_decode(
        tp_mesh,
        weights,
        BATCH,
        LAYERS,
        {
            "attention_token_scatter": scatter,
            "moe_token_scatter_output": moe_out,
            "moe_routed_reduce_scatter": moe_reduce_scatter,
            "sequence_parallel_pre_attention": sequence_parallel,
            "aux_hidden_layers": aux_layers,
        },
    )
    hlo = jax.jit(sharded).lower(*inputs, *weights.values()).as_text()
    assert "tpu_custom_call" in hlo


def test_decode_megakernel_batch32_compiles_with_int16_gather(
    tp_mesh: Mesh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch-32 decode compiles through Mosaic using int16 page-offset gathers."""
    orig_take_along_axis = jnp.take_along_axis
    gather_calls: list[tuple[jnp.dtype, object]] = []

    def _spy_take_along_axis(
        arr: jax.Array,
        indices: jax.Array,
        *args: object,
        **kwargs: object,
    ) -> jax.Array:
        """Record index dtype and wrap_negative_indices while forwarding to JAX."""
        gather_calls.append((indices.dtype, kwargs.get("wrap_negative_indices")))
        return orig_take_along_axis(arr, indices, *args, **kwargs)

    monkeypatch.setattr(jnp, "take_along_axis", _spy_take_along_axis)
    weights = _abstract_weights(layers=ROW_PLAN_LAYERS)
    sharded, inputs = _make_sharded_decode(
        tp_mesh,
        weights,
        ROW_PLAN_BATCH_SIZE,
        ROW_PLAN_LAYERS,
        {
            "attention_token_scatter": True,
            "moe_token_scatter_output": True,
            "moe_routed_reduce_scatter": True,
            "sequence_parallel_pre_attention": True,
            "residual_reduction_order": "relaxed",
            "mla_cache_batch_tile_size": ROW_PLAN_MLA_BATCH_TILE,
            "alias_expert_scales": True,
        },
    )
    try:
        jax.jit(sharded).lower(*inputs, *weights.values()).compile()
    finally:
        # Release the compiled 32-device executable before later CPU tests run.
        jax.clear_caches()
        gc.collect()
    assert any(dtype == jnp.int16 and wrap is False for dtype, wrap in gather_calls), (
        f"Expected int16 gather with wrap_negative_indices=False, saw {gather_calls}"
    )
