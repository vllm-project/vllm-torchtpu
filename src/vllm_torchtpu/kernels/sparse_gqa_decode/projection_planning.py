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
"""Projection layout and VMEM budgets for the resident-weight decode kernel."""

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from .planning import VMEM_USABLE_FRACTION, DecodeLaunch

_Array = jax.Array | jax.ShapeDtypeStruct


def projection_width(
    query: _Array, weight: _Array, residual: _Array, hc_gate: _Array
) -> int:
    """Validate the tiled weight/HC ABI and return the unpadded output width."""
    if (
        query.ndim != 3
        or weight.ndim != 3
        or min(weight.shape) <= 0
        or weight.dtype != jnp.bfloat16
        or weight.shape[1] != math.prod(query.shape[1:])
        or residual.ndim != 2
        or residual.shape[0] != len(query)
        or residual.dtype != jnp.bfloat16
        or hc_gate.ndim != 2
        or hc_gate.shape[0] != len(query)
        or hc_gate.shape[1] <= 0
        or hc_gate.dtype != jnp.float32
        or residual.shape[1] <= 0
        or residual.shape[1] % hc_gate.shape[1]
    ):
        raise ValueError(
            "Projection requires BF16 tiled [tiles,K,N] weights/residual "
            "and FP32 [batch,branches] HC gates"
        )
    width = residual.shape[1] // hc_gate.shape[1]
    if weight.shape[0] != math.ceil(width / weight.shape[2]):
        raise ValueError("Projection tiles must cover the logical HC output width")
    return width


@dataclasses.dataclass(frozen=True)
class ProjectionResources:
    weight_bytes: int
    attended_bytes: int
    residual_bytes: int
    gate_bytes: int
    output_bytes: int
    working_bytes: int

    @property
    def vmem_bytes(self) -> int:
        return sum(dataclasses.astuple(self))


def projection_resources(
    query: _Array, weight: _Array, hc_gate: _Array, info: pltpu.TpuInfo
) -> ProjectionResources:
    """Include vector layout padding, resident buffers and FP32 dot/HC planes."""

    def storage(shape, dtype):
        rows, columns = shape[-2:]
        row_tile = info.get_sublane_tiling(dtype)
        return (
            math.prod(shape[:-2])
            * math.ceil(rows / row_tile)
            * row_tile
            * math.ceil(columns / info.num_lanes)
            * info.num_lanes
            * jnp.dtype(dtype).itemsize
        )

    batch, heads, dim = query.shape
    branches = hc_gate.shape[1]
    padded_width = weight.shape[0] * weight.shape[2]
    return ProjectionResources(
        weight_bytes=storage(weight.shape, jnp.bfloat16),
        # RPA's FP32 reader uses [KV,batch,heads/KV,1,dim]; account the
        # singleton row padding conservatively without assuming KV grouping.
        attended_bytes=storage((batch, heads, 1, dim), jnp.float32),
        residual_bytes=storage((batch, branches * padded_width), jnp.bfloat16),
        gate_bytes=storage(hc_gate.shape, hc_gate.dtype),
        output_bytes=storage((batch, branches * padded_width), jnp.bfloat16),
        working_bytes=(
            storage((batch, heads * dim), jnp.bfloat16)
            + 4 * storage((batch, weight.shape[2]), jnp.float32)
        ),
    )


def projection_fits(
    launch: DecodeLaunch,
    query: _Array,
    weight: _Array,
    hc_gate: _Array,
    info: pltpu.TpuInfo,
) -> bool:
    if launch.chunk_queries < len(query) or weight.shape[2] % info.num_lanes:
        return False
    resources = projection_resources(query, weight, hc_gate, info)
    return (
        launch.resources.vmem_estimate_bytes + resources.vmem_bytes
        <= info.vmem_capacity_bytes * VMEM_USABLE_FRACTION
    )


def plan_projected_decode(
    query: _Array,
    gate: _Array,
    blocks: _Array,
    lengths: _Array,
    table: _Array,
    cache: _Array,
    weight: _Array,
    hc_gate: _Array,
    *,
    kv_heads: int,
    ratio: int,
    k_scale: object,
    v_scale: object,
) -> DecodeLaunch | None:
    """Reduce pipeline concurrency only as needed to fit all resident buffers."""
    from .eligibility import plan_decode
    from .tuned_block_sizes import DEFAULT_POLICY

    info = pltpu.get_tpu_info()
    lanes = DEFAULT_POLICY.pipeline_lanes
    while lanes:
        launch = plan_decode(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            kv_heads=kv_heads,
            ratio=ratio,
            k_scale=k_scale,
            v_scale=v_scale,
            policy=dataclasses.replace(DEFAULT_POLICY, pipeline_lanes=lanes),
        )
        if launch is not None and projection_fits(launch, query, weight, hc_gate, info):
            return launch
        lanes //= 2
    return None
