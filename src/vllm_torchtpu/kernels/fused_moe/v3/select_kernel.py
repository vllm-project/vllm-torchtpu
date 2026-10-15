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
"""Top-k expert selection: an in-VMEM max-and-mask pass over the router
logits, with the renormalized weights computed in place.

A row with no finite logit gets weight zero on every slot. As a routing
message it carries the index one past the last expert, which the tables
kernel allocates no routed row; as plain indices it carries expert zero,
and the layer's weights zero it.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3.layout import LANES

# Below any real logit or softmax score, without being infinite.
UNROUTABLE_LOGIT = -3.0e38
# A selected maximum at or below this is the unroutable logit: the row
# routes nowhere.
UNROUTABLE_MAX = -1e38


def _lane_sum_in_xla_order(values, top_k):
    """The sum of values' top_k lanes in the order XLA's lane reduction adds
    them (a halving tree over the row padded with zeros to a power of two),
    so the renormalized weights match the layer's XLA form bit for bit."""
    columns = [values[:, k : k + 1] for k in range(top_k)]
    width = 1
    while width < top_k:
        width *= 2
    columns += [None] * (width - top_k)
    while width > 1:
        width //= 2
        columns = [
            a if b is None else b if a is None else a + b
            for a, b in zip(columns[:width], columns[width : 2 * width])
        ]
    return columns[0]


def _select_body(
    logits_ref,
    weights_ref,
    indices_ref,
    *scratch,
    top_k,
    num_experts,
    num_rows,
    renormalize,
    drop_nan_rows,
    message_rows,
    pair_rows,
):
    """One block of rows: the top_k by repeated max-and-mask.

    With `renormalize` the weights are the top_k-wide softmax of the
    selected logits, else the selected values themselves. With `message_rows`
    the indices go out slot-major as the routing message's index half (word
    k * rows + r is row r's slot k): each block's [rows, top_k] index tile
    is written column by column into VMEM, transposed a lane block of rows
    at a time, and each slot's lane rows stored at their sublanes. The rows
    behind the indices are zeroed on the first step.
    """
    logits = jnp.where(jnp.isnan(logits_ref[...]), UNROUTABLE_LOGIT, logits_ref[...])
    expert_iota = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
    index_out = scratch[0] if message_rows else indices_ref
    if renormalize:
        selected = jnp.full((logits.shape[0], top_k), UNROUTABLE_LOGIT, jnp.float32)
        slot_iota = jax.lax.broadcasted_iota(jnp.int32, selected.shape, 1)
    else:
        selected = slot_iota = None
    routes = None
    for k in range(top_k):
        row_max = jnp.max(logits, axis=1, keepdims=True)
        if k == 0:
            routes = row_max > jnp.float32(UNROUTABLE_MAX)
        is_max = logits == row_max
        # The lowest matching column. A non-matching column reads as the
        # last expert, so every index returned is a real expert.
        index = jnp.min(jnp.where(is_max, expert_iota, num_experts - 1), axis=1)
        if renormalize:
            assert selected is not None and slot_iota is not None
            selected = jnp.where(slot_iota == k, row_max, selected)
        else:
            weights_ref[:, k] = row_max[:, 0]
        index_out[:, k] = index.astype(jnp.int32)
        logits = jnp.where(expert_iota == index[:, None], UNROUTABLE_LOGIT, logits)
    assert routes is not None  # set on the first of top_k >= 1 passes
    if renormalize:
        assert selected is not None
        if drop_nan_rows:
            routes = jnp.logical_and(
                jnp.logical_not(
                    jnp.any(jnp.isnan(logits_ref[...]), axis=1, keepdims=True)
                ),
                routes,
            )
        peak = jnp.max(selected, axis=1, keepdims=True)
        exponentials = jnp.exp(selected - jnp.where(routes, peak, jnp.float32(0.0)))
        denominator = _lane_sum_in_xla_order(exponentials, top_k)
        weights = exponentials / jnp.where(routes, denominator, jnp.float32(1.0))
        weights_ref[...] = jnp.where(routes, weights, jnp.float32(0.0))
    if message_rows:
        # A row that routes nowhere carries the index one past the last
        # expert on every slot: the tables allocate it no routed row and
        # the combine gives it weight zero. Padded rows arrive this way.
        index_out[...] = jnp.where(routes, index_out[...], jnp.int32(num_experts))
        step = pl.program_id(0)
        block_rows, lanes = index_out.shape
        rows_per_slot = num_rows // lanes
        halves = block_rows // lanes
        index_rows = top_k * rows_per_slot
        if message_rows > index_rows:
            # The pair list is padded to whole tables steps with pairs that
            # route nowhere (rows up to pair_rows); the scale rows after
            # them start zero.
            @pl.when(step == 0)
            def _():
                if pair_rows > index_rows:
                    indices_ref[index_rows:pair_rows, :] = jnp.full(
                        (pair_rows - index_rows, lanes), num_experts, jnp.int32
                    )
                if message_rows > pair_rows:
                    indices_ref[pair_rows:, :] = jnp.zeros(
                        (message_rows - pair_rows, lanes), jnp.int32
                    )

        tiles = [index_out[h * lanes : (h + 1) * lanes, :].T for h in range(halves)]
        for k in range(top_k):
            rows = jnp.concatenate([t[k : k + 1, :] for t in tiles], axis=0)
            indices_ref[pl.ds(k * rows_per_slot + step * halves, halves), :] = rows


def select_top_k(
    logits,
    *,
    top_k,
    block_rows,
    renormalize=False,
    drop_nan_rows=False,
    message_rows=None,
    pair_rows=None,
):
    """Top-k over [rows, experts] scores.

    Returns (weights f32 [rows, top_k], indices int32 [rows, top_k]). With
    `message_rows` the indices come back as the routing message's index half,
    [message_rows, LANES] slot-major; the rows after the real pairs up to
    `pair_rows` (the index rows when None) carry the unrouted index. The
    scores and both outputs are pinned to HBM so the blocks stream through
    VMEM from here.
    """
    num_rows, num_experts = logits.shape
    if pair_rows is None:
        pair_rows = -(-num_rows * top_k // LANES)
    if num_rows % block_rows:
        raise ValueError(
            f"{num_rows} rows is not a whole number of {block_rows}-row blocks"
        )
    hbm = pltpu.MemorySpace.HBM
    lanes = LANES
    if message_rows:
        if block_rows % lanes or (num_rows * top_k) % lanes:
            raise ValueError(
                f"the routing message takes whole {lanes}-lane rows: "
                f"a block of {block_rows} rows and {num_rows} x "
                f"{top_k} selections have to be multiples of it"
            )
        if pair_rows * lanes < num_rows * top_k or message_rows < pair_rows:
            raise ValueError(
                f"a routing message of {message_rows} rows with "
                f"{pair_rows} pair rows holds fewer than the "
                f"{num_rows * top_k} selections"
            )
        index_shape = hbm((message_rows, lanes), jnp.int32)
        index_spec = pl.BlockSpec((message_rows, lanes), lambda i: (0, 0))
        scratch = [pltpu.VMEM((block_rows, lanes), jnp.int32)]
    else:
        index_shape = hbm((num_rows, top_k), jnp.int32)
        index_spec = pl.BlockSpec((block_rows, top_k), lambda i: (i, 0))
        scratch = []
    return pl.pallas_call(
        functools.partial(
            _select_body,
            top_k=top_k,
            num_experts=num_experts,
            num_rows=num_rows,
            renormalize=renormalize,
            drop_nan_rows=drop_nan_rows,
            message_rows=message_rows,
            pair_rows=pair_rows,
        ),
        grid=(num_rows // block_rows,),
        in_specs=[pl.BlockSpec((block_rows, num_experts), lambda i: (i, 0))],
        out_specs=[pl.BlockSpec((block_rows, top_k), lambda i: (i, 0)), index_spec],
        out_shape=[hbm((num_rows, top_k), jnp.float32), index_shape],
        scratch_shapes=scratch,
        name=f"fused_ep_select_{'renormalized' if renormalize else 'raw'}",
    )(pltpu.with_memory_space_constraint(logits, hbm))
