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
"""TPU_DCP_MLA_TOKEN_SPLIT: the sparse MLA of a large prefill bucket with each
DCP lane attending its block of the tokens with every head, instead of every
token with its lane's heads.

With tp // dcp lanes, every lane holds a replica of the same KV shards, so
under head split each lane gathers the same top-k KV rows for every token.
Under token split lane `l` takes tokens `[l * T / L, (l + 1) * T / L)` and all
`L * dcp * h` heads (h = heads per rank), so each row is gathered once per
lane block instead of once per lane.

The kernel heads run in `head_order`: position `p = (d * L + l) * h + j`
holds global head `(l * dcp + d) * h + j`. The DCP merge reduce-scatters the
head axis into blocks `d`, which leaves rank `(l, d)` with its lane's tokens
for heads `(l', d, j)`; a lane all-to-all (split heads on `l'`, join tokens)
then gives every rank all T tokens of its own o_proj heads `(l, d, j)`, in
token order. The 64-head q_b / W_UK_T / W_UV copies are built at load in this
order, so nothing is permuted at run time.
"""

from typing import Any

import jax
import torch
from jax import lax
from jax.sharding import PartitionSpec as P

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter import cp_mla_attention as _cp


def head_order(dcp_size: int, num_lanes: int, heads_per_rank: int) -> list[int]:
    """The global head each token-split kernel head position holds."""
    return [
        (lane * dcp_size + d) * heads_per_rank + j
        for d in range(dcp_size)
        for lane in range(num_lanes)
        for j in range(heads_per_rank)
    ]


def q_column_index(
    dcp_size: int, num_lanes: int, heads_per_rank: int, head_dim: int
) -> torch.Tensor:
    """Output columns, in `head_order`, of a query-replicated q projection
    all-gathered over the TP group: rank `r` holds lane `r // dcp_size`'s
    `dcp_size * heads_per_rank` heads, `head_dim` columns each."""
    group = dcp_size * heads_per_rank
    cols_per_rank = group * head_dim
    pieces = []
    for g in head_order(dcp_size, num_lanes, heads_per_rank):
        lane, within = divmod(g, group)
        start = lane * dcp_size * cols_per_rank + within * head_dim
        pieces.append(torch.arange(start, start + head_dim))
    return torch.cat(pieces)


def lane_rows(x: torch.Tensor, lane: torch.Tensor, num_lanes: int) -> torch.Tensor:
    """Block `lane` (an i32 [1] tensor) of `num_lanes` equal blocks of `x`'s rows."""
    return x.unflatten(0, (num_lanes, -1)).index_select(0, lane).squeeze(0)


def lane_all_to_all_local(x: jax.Array, lane_axis: str) -> jax.Array:
    """Inside a shard_map: this rank's `[T / L, L * h, V]` (its lane's tokens,
    head block `l'` for lane `l'`) -> `[T, h, V]`, every lane's tokens of its
    own heads, lane-major (= token order)."""
    return lax.all_to_all(x, lane_axis, split_axis=1, concat_axis=0, tiled=True)


def lane_all_to_all(x: jax.Array, mesh: Any, lane_axis: str) -> jax.Array:
    """`lane_all_to_all_local` over `mesh`; `x` is replicated in the spec but
    holds each lane's own rows, as the torch bridge hands them."""
    return jax.shard_map(
        lambda v: lane_all_to_all_local(v, lane_axis),
        mesh=mesh,
        in_specs=P(),
        out_specs=P(),
        check_vma=False,
    )(x)


_A2A_OPS: dict[tuple, Any] = {}


def get_lane_all_to_all_op(mesh: Any, lane_axis: str):
    """The torch op around `lane_all_to_all`. Build at load, outside Dynamo."""
    from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
        pcp_streaming_jax_op,
    )

    lanes = int(mesh.shape[lane_axis])
    key = (tuple(mesh.axis_names), tuple(mesh.shape.values()), lane_axis)
    op = _A2A_OPS.get(key)
    if op is not None:
        return op

    def fn(x: jax.Array) -> jax.Array:
        return lane_all_to_all(x, mesh, lane_axis)

    op = pcp_streaming_jax_op(
        f"pallas::dcp_token_split_lane_a2a_l{lanes}",
        fn,
        donate_argnums=None,
        mesh=mesh,
        input_partition_specs=(P(),),
        output_partition_specs=(P(),),
    )

    def fake(x):
        return x.new_empty((x.shape[0] * lanes, x.shape[1] // lanes, *x.shape[2:]))

    op.register_fake(fake)
    _A2A_OPS[key] = op
    return op


def merge_scatter_heads(
    partial: torch.Tensor,
    lse: torch.Tensor,
    w_uv: torch.Tensor,
    w_uv_scale: torch.Tensor | None,
    wire_dtype: torch.dtype,
    a2a_op: Any,
) -> torch.Tensor:
    """This lane's `[T / L, L * dcp * h, C]` partial -> `[T, h, V]`: the W_UV
    DCP merge the head-split backbone runs (V-space chunked where it applies),
    then the lane all-to-all."""
    if _cp.vspace_fused_applies(partial.shape[0], True):
        merged = _cp.merge_lse_partials_scatter_heads_vspace(
            partial, lse, w_uv, w_uv_scale, wire_dtype=wire_dtype
        )
    else:
        narrow = envs.TPU_DCP_RS_OUT_PRECISION
        projected = _cp.project_uv_heads(
            partial,
            w_uv,
            w_uv_scale,
            out_dtype=wire_dtype if narrow else torch.float32,
        )
        # Rows are lane-local, so no real-row count (query_start_loc) here.
        merged = _cp.merge_lse_partials_scatter_heads(
            projected, lse, is_backbone=True, wire_dtype=wire_dtype
        )
    return a2a_op(merged.contiguous())
