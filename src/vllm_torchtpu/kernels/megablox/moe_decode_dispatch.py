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
"""Decode-sized EP MoE dispatch in one Pallas kernel.

From the local routes of a few tokens it builds everything the grouped GEMM
and the one-hot combine need: the permuted rows, the group sizes, the combine
matrix and the computed-row mask. As XLA ops this is a counting sort, a
handful of mask and relayout fusions and a one-hot matmul, each a separate op
of fixed cost at decode sizes; the GEMM's two operands are also written
straight to HBM here, where XLA would add a copy for each.

The routes sort by expert, stably (ties keep their flat order), as
`fused_moe_gmm.prepare_routed_gmm_inputs` sorts them; invalid routes take the
sentinel key and sort after every expert. Every output value is exact: each
matmul output has at most one nonzero term.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Routes are laid out along one row of lanes.
MAX_ROUTES = 128


def _iota(shape: tuple[int, int], axis: int) -> jax.Array:
    return jax.lax.broadcasted_iota(jnp.int32, shape, axis)


def _owner(shape: tuple[int, int], route_axis: int, topk: int) -> jax.Array:
    """Route ``j`` belongs to token ``j // topk``, as compares, with routes on
    `route_axis` and tokens on the other axis."""
    first = _iota(shape, 1 - route_axis) * topk
    route = _iota(shape, route_axis)
    return (route >= first) & (route < first + topk)


def _flatten(x: jax.Array, lanes: int) -> jax.Array:
    """``[T, K]`` -> ``[1, lanes]`` in flat ``t * K + k`` order (zero past
    ``T * K``), exact for values bf16 holds: ``x @ [K, lanes]`` spreads column
    ``k`` to every lane ``j`` with ``j % K == k``, then each lane keeps its own
    token's row."""
    num_tokens, topk = x.shape
    lane = _iota((topk, lanes), 1)
    col = _iota((topk, lanes), 0)
    spread = functools.reduce(
        jnp.logical_or, [lane == t * topk + col for t in range(num_tokens)]
    )
    wide = jnp.dot(
        x.astype(jnp.bfloat16),
        spread.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
    return jnp.sum(
        jnp.where(_owner((num_tokens, lanes), 1, topk), wide, 0),
        axis=0,
        keepdims=True,
    )


def _transpose_row(row: jax.Array) -> jax.Array:
    """``[1, L]`` -> ``[L, 1]`` through one aligned ``(8, L)`` transpose."""
    return jnp.transpose(jnp.broadcast_to(row, (8, row.shape[1])))[:, :1]


def _dispatch_kernel(
    ids_ref,
    weights_ref,
    hidden_ref,
    rows_hbm,
    sizes_hbm,
    combine_ref,
    computed_ref,
    rows_vmem,
    sizes_vmem,
    sems,
    *,
    num_experts: int,
    skip_padded_tokens: bool,
):
    num_tokens, topk = ids_ref.shape
    num_routes = num_tokens * topk
    lanes = MAX_ROUTES
    ids, weights = ids_ref[...], weights_ref[...]
    valid = ids >= 0
    if skip_padded_tokens:
        valid &= weights != 0
    flat = _iota((1, lanes), 1)
    # Padding takes a key past the sentinel, so it sorts after every route.
    key_row = jnp.where(
        flat < num_routes,
        _flatten(jnp.where(valid, ids, num_experts), lanes).astype(jnp.int32),
        num_experts + 1,
    )
    weight_row = _flatten(jnp.where(valid, weights, 0), lanes)
    key_col = _transpose_row(key_row)
    weight_col = _transpose_row(weight_row)

    lane = _iota((lanes, lanes), 1)
    sub = _iota((lanes, lanes), 0)
    # [i, j]: route i (sublanes) sorts before route j (lanes), and the reverse.
    before = (key_col < key_row) | ((key_col == key_row) & (sub < lane))
    after = (key_row < key_col) | ((key_row == key_col) & (lane < sub))
    pos_row = jnp.sum(before.astype(jnp.int32), axis=0, keepdims=True)
    pos_col = jnp.sum(after.astype(jnp.int32), axis=1, keepdims=True)

    sizes = jnp.sum((key_col == lane).astype(jnp.int32), axis=0, keepdims=True)
    sizes_vmem[...] = sizes[0, :num_experts]
    sizes_copy = pltpu.make_async_copy(sizes_vmem, sizes_hbm, sems.at[0])
    sizes_copy.start()

    # [row, route]: the route landing on each sorted row.
    lands = pos_row == _iota((num_routes, lanes), 0)
    permute = jnp.dot(
        lands.astype(jnp.bfloat16),
        _owner((lanes, num_tokens), 0, topk).astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    ).astype(hidden_ref.dtype)
    rows_vmem[...] = jnp.dot(
        permute, hidden_ref[...], preferred_element_type=jnp.float32
    ).astype(rows_vmem.dtype)
    rows_copy = pltpu.make_async_copy(rows_vmem, rows_hbm, sems.at[1])
    rows_copy.start()

    # [route, row]: each route's weight at its sorted row, summed per token.
    on_row = pos_col == _iota((lanes, num_routes), 1)
    computed_ref[...] = jnp.sum(
        (on_row & (key_col < num_experts)).astype(jnp.int32), axis=0, keepdims=True
    )
    weighted = jnp.where(on_row, weight_col, 0).astype(combine_ref.dtype)
    combine_ref[...] = jnp.dot(
        _owner((num_tokens, lanes), 1, topk).astype(combine_ref.dtype),
        weighted,
        preferred_element_type=jnp.float32,
    ).astype(combine_ref.dtype)
    sizes_copy.wait()
    rows_copy.wait()


def decode_dispatch(
    hidden_states: jax.Array,
    topk_ids: jax.Array,
    topk_weights: jax.Array,
    *,
    num_experts: int,
    skip_padded_tokens: bool,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Permute, group and combine metadata of decode-sized local routes.

    `topk_ids` / `topk_weights` are ``[T, topk]`` local expert ids (``-1``
    for a route this shard does not own) and their bf16 weights. Returns
    ``(x [T * topk, H], group_sizes [num_experts] int32, combine [T, T *
    topk], computed [T * topk] bool)``: the expert-sorted rows of
    `hidden_states`, each expert's row count, the routing weight of each
    token at each sorted row, and which rows the grouped GEMM computes.
    """
    num_tokens, topk = topk_ids.shape
    num_routes = num_tokens * topk
    assert num_routes <= MAX_ROUTES and num_experts + 1 < MAX_ROUTES
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    rows, sizes, combine, computed = pl.pallas_call(
        functools.partial(
            _dispatch_kernel,
            num_experts=num_experts,
            skip_padded_tokens=skip_padded_tokens,
        ),
        out_shape=[
            # The GEMM reads both from HBM: placed there, XLA adds no copy.
            pltpu.HBM((num_routes, hidden_states.shape[1]), hidden_states.dtype),
            pltpu.HBM((num_experts,), jnp.int32),
            jax.ShapeDtypeStruct((num_tokens, num_routes), topk_weights.dtype),
            jax.ShapeDtypeStruct((1, num_routes), jnp.int32),
        ],
        in_specs=[vmem, vmem, vmem],
        out_specs=[hbm, hbm, vmem, vmem],
        scratch_shapes=[
            pltpu.VMEM((num_routes, hidden_states.shape[1]), hidden_states.dtype),
            pltpu.VMEM((num_experts,), jnp.int32),
            pltpu.SemaphoreType.DMA((2,)),
        ],
        interpret=interpret,
        name="moe_decode_dispatch",
    )(topk_ids, topk_weights, hidden_states)
    return rows, sizes, combine, computed.reshape(num_routes) != 0
