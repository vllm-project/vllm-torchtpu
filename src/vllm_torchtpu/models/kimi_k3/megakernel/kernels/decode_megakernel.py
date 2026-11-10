# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

"""Kimi K3 decode megakernel components."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import math
import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.megakernel.core import pool_alias

from .. import kernels as kimi
from ..common import collectives32
from ..common.vmem_pool import VmemPool
from ..config import active as _config
from ..layout import CONV_HEAD_ROWS, cache_layout
from . import mla_dcp

logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from collections.abc import Callable

    from jax._src.lib import hlo as jax_hlo
    from jax._src.pallas import core as pl_core
    from jax._src.state import types as state_types

    type TensorLike = (
        jax.Array
        | pl_core.MemoryRef
        | state_types.TransformedRef
        | jax.ShapeDtypeStruct
        | tpu.SemaphoreType
        | jax_hlo.HloModule
    )
else:
    type TensorLike = jax.Array
_NONE_STAGE_VAL: TensorLike = getattr(None, "", None)
_PAIR_TUPLE_LEN = 2
_DCP_HEAD_COUNT = 3
_TP_RING_SIZE = 4
_LAST_EXPERT_SLOT = 7
_ROUTED_EXPERTS_PER_TOKEN = 8
_EXPERT_COL_PACKING = 12
_EXPERT_GROUP_STRIDE = 16
_KDA_GROUP_WIDTH = 32
_GATED_GROUP_WIDTH = 64
_MAX_STAGE_INDEX = 92
_GATED_GROUP_STRIDE = 96
_VREG_LANE_COUNT = 128
_ATTN_OUT_WIDTH = 384
_LATENT_SUB_WIDTH = 768
_FULL_SEQ_PAD_LIMIT = 896
_KV_COMMIT_TILE_TOKENS = 1024
_GATE_PROJ_WIDTH = 1664
_HIDDEN_SIZE = 7168
_ROPE_BASE_DIM = 8192
_DATACLASS_REGISTRATION_STATE: list[bool] = []


def _ensure_dataclasses_registered() -> None:
    """Compute Pallas VMEM tensors for ensure stage.

    Register decode megakernel PyTree dataclasses lazily on first
    instantiation.
    """
    if _DATACLASS_REGISTRATION_STATE:
        return
    for cls in (
        _PaddedBf16Rows,
        AttentionCommunicationWorkspace,
        MixtureOfExpertsWeights,
        MixtureOfExpertsWorkspace,
        KimiDeltaAttentionWeights,
        MultiHeadLatentAttentionWeights,
        MultiHeadLatentAttentionWorkspace,
    ):
        jax.tree_util.register_dataclass(cls)
    _DATACLASS_REGISTRATION_STATE.append(True)


@dataclasses.dataclass(frozen=True)
class _PaddedBf16Rows(pool_alias.Reinterpret):
    """Expose active rows while preserving the full native row capacity."""

    if TYPE_CHECKING:

        def __init__(self, shape: tuple[int, ...]) -> None:
            """Initialize padded bfloat16 rows transform."""

    def __post_init__(self) -> None:
        """Compute and stage Pallas VMEM state for post init.

        Register PyTree dataclasses lazily and chain parent
        initialization.
        """
        _ensure_dataclasses_registered()
        super_post_init = getattr(super(), "__post_init__", None)
        if super_post_init is not None:
            super_post_init()

    def _validate_shape(self, input_shape: tuple[int, ...]) -> None:
        """Compute and stage Pallas VMEM state for validate shape.

        Validate that backing storage is a multiple of 16 rows and 128
        columns.
        """
        rows, width = self.shape
        backing_rows = math.prod(input_shape) // width
        np.testing.assert_(
            len(self.shape) == _PAIR_TUPLE_LEN
            and width % _VREG_LANE_COUNT == 0
            and (1 <= rows <= backing_rows),
            "Invalid padded bf16 row view shape",
        )
        np.testing.assert_(
            backing_rows * width == math.prod(input_shape)
            and backing_rows % _EXPERT_GROUP_STRIDE == 0,
            str((input_shape, self.shape)),
        )


PHASE_SCOPES = _config().phase_scopes


def _phase(name: str) -> contextlib.AbstractContextManager[TensorLike]:
    """Return a named XProf trace scope when phase scopes are active."""
    return jax.named_scope(name) if PHASE_SCOPES else contextlib.nullcontext()


def _phase_start(name: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for phase start."""
    scope = _phase(name)
    scope.__enter__()
    return scope


def _phase_end(scope: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for phase end."""
    scope.__exit__(None, None, None)


def _scoped(name: TensorLike) -> TensorLike:
    """Decorator form of _phase for pl.when bodies."""

    def wrap(function: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for wrap."""

        @functools.wraps(function)
        def scoped(*args: TensorLike, **kwargs: TensorLike) -> TensorLike:
            """Compute and stage Pallas VMEM state for scoped."""
            with _phase(name):
                return function(*args, **kwargs)

        return scoped

    return wrap


MAX_BATCH = 128
SCATTER_BATCHES = (8, 12, 16, 20, 24, 28, 32, 64, 96, 128)


def _bf16_rows_view(ref: TensorLike, rows: int, width: int) -> TensorLike:
    """View a scratch ref as padded bfloat16 rows."""
    ref = ref.bitcast(jnp.bfloat16)
    return pool_alias.types.TransformedRef(
        ref.ref, (*ref.transforms, _PaddedBf16Rows((rows, width)))
    )


def _dot(x: TensorLike, w: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for dot."""
    batch_size = x.shape[0]
    np.testing.assert_(
        bool(1 <= batch_size <= MAX_BATCH), f"_dot supports 1 to {MAX_BATCH} rows"
    )
    if batch_size == 1:
        tiled_x = jnp.broadcast_to(x, (16, x.shape[-1]))
    elif batch_size % 16:
        tiled_x = jnp.pad(x, ((0, -batch_size % 16), (0, 0)))
    else:
        tiled_x = x
    return jax.lax.dot_general(
        tiled_x, w, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )[:batch_size]


def _attention_scores(queries: TensorLike, keys: TensorLike) -> TensorLike:
    """Multiply partial query rows by a transposed key tile in FP32."""
    row_count = queries.shape[0]
    padded_queries = jnp.pad(queries, ((0, -row_count % 16), (0, 0)))
    return jax.lax.dot_general(
        padded_queries,
        keys,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:row_count]


def _gated_activation(gate: TensorLike, value: TensorLike) -> TensorLike:
    """Kimi K3 shared/routed expert activation, rounded to BF16."""
    return (
        4 * jnp.tanh(gate / 4) * jax.nn.sigmoid(gate) * 25 * jnp.tanh(value / 25)
    ).astype(jnp.bfloat16)


def _route_hidden_states(
    hidden_states: TensorLike,
    routing_projection_vmem: TensorLike,
    routing_bias_ref: TensorLike,
    routing_scores_vmem: TensorLike,
    send_semaphore: TensorLike,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> TensorLike:
    """Compute and stage Pallas VMEM state for route hidden states.

    Project and select local expert routes through one shared B1/B2
    path.

    ``routing_scores_vmem`` is [batch_size, 1024]. Paired ranks
    exchange their 512-score halves directly in that padded workspace,
    then every top-k iteration advances all batch rows together.
    Selected local routes are returned as [batch_size, top_k] values
    for scalar SMEM materialization by the caller.
    """
    (
        receive_semaphore,
        selected_expert_count,
        local_expert_count,
        expert_tensor_parallel,
        on_local_selection,
        after_selection,
        after_scores,
        after_score_send,
        live_rows,
    ) = (
        (*step_args[0:1], stage_kwargs.get("receive_semaphore", _NONE_STAGE_VAL))[0],
        stage_kwargs["selected_expert_count"],
        stage_kwargs["local_expert_count"],
        stage_kwargs["expert_tensor_parallel"],
        stage_kwargs["on_local_selection"],
        stage_kwargs["after_selection"],
        stage_kwargs.get("after_scores", _NONE_STAGE_VAL),
        stage_kwargs.get("after_score_send", _NONE_STAGE_VAL),
        stage_kwargs.get("live_rows", _NONE_STAGE_VAL),
    )
    rank = jax.lax.axis_index("tp")
    batch_size = hidden_states.shape[0]
    expert_count = routing_bias_ref.shape[-1]
    np.testing.assert_(
        not (
            expert_count != _FULL_SEQ_PAD_LIMIT
            or hidden_states.shape[-1] > _HIDDEN_SIZE
        ),
        "Host-shared router requires 896 expert scores",
    )
    np.testing.assert_(
        routing_scores_vmem.shape[-1] == _KV_COMMIT_TILE_TOKENS,
        "Routing score workspace must have padded width 1024",
    )
    router_lane = rank % 2
    local_scores = _dot(hidden_states, routing_projection_vmem[0]).reshape(
        batch_size, 512
    )
    column = jax.lax.broadcasted_iota(jnp.int32, local_scores.shape, 1)
    local_scores = jnp.where(
        (router_lane == 0) | (column < _ATTN_OUT_WIDTH), local_scores, 0.0
    )
    score_half = routing_scores_vmem.at[:, pl.ds(router_lane * 512, 512)]
    score_half[...] = local_scores
    if after_scores is not None:
        after_scores()
    copy = tpu.make_async_remote_copy(
        score_half,
        score_half,
        send_semaphore,
        receive_semaphore,
        device_id=(rank ^ 1,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()
    if after_score_send is not None:
        after_score_send()
    copy.wait()
    raw_scores = routing_scores_vmem[:, :expert_count]
    uncorrected_probabilities = jax.nn.sigmoid(raw_scores)
    routing_scores_vmem[:, :expert_count] = (
        uncorrected_probabilities + routing_bias_ref[...]
    )
    selected_experts = jnp.zeros((batch_size, selected_expert_count), jnp.int32)
    selected_probabilities = jnp.zeros((batch_size, selected_expert_count), jnp.float32)
    local_route_counts = jnp.zeros((batch_size,), jnp.int32)
    probability_totals = jnp.zeros((batch_size,), jnp.float32)
    expert_ids = jnp.arange(expert_count)[None, :]
    route_slots = jnp.arange(selected_expert_count)[None, :]
    for selection in range(selected_expert_count):
        winners = jnp.argmax(routing_scores_vmem[:, :expert_count], axis=1).astype(
            jnp.int32
        )
        winner_mask = expert_ids == winners[:, None]
        probabilities = jnp.sum(
            jnp.where(winner_mask, uncorrected_probabilities, 0), axis=1
        )
        probability_totals += probabilities
        routing_scores_vmem[:, :expert_count] = jnp.where(
            winner_mask, -jnp.inf, routing_scores_vmem[:, :expert_count]
        )
        counts_before_selection = local_route_counts
        owned_locally = winners // local_expert_count == rank // expert_tensor_parallel
        if live_rows is not None:
            owned_locally &= live_rows
        destination_mask = owned_locally[:, None] & (
            route_slots == counts_before_selection[:, None]
        )
        selected_experts = jnp.where(
            destination_mask, (winners % local_expert_count)[:, None], selected_experts
        )
        selected_probabilities = jnp.where(
            destination_mask, probabilities[:, None], selected_probabilities
        )
        local_route_counts = counts_before_selection + owned_locally.astype(jnp.int32)
        on_local_selection(
            winners, owned_locally, counts_before_selection, selected_experts
        )
        after_selection(selection)
    return (
        selected_experts,
        selected_probabilities,
        local_route_counts,
        probability_totals
        if live_rows is None
        else jnp.where(live_rows, probability_totals, 1.0),
    )


def _route_owned_rows(
    owned_hidden_ref: TensorLike,
    batch_size: TensorLike,
    routing_projection_vmem: TensorLike,
    routing_bias_ref: TensorLike,
    routing_scores_vmem: TensorLike,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> TensorLike:
    """Compute and stage Pallas VMEM state for route owned rows.

    ``_route_hidden_states`` with the top-k sharded over the local
    rank pairs.

    Each rank pair scores and selects only the rows it owns
    (``owned_hidden_ref`` [batch / 4, hidden]); the pairs exchange
    their selections (global expert ids and probabilities, one FP32
    row per token) and every rank derives its local routes from all of
    them. The per-row arithmetic and order are those of the replicated
    router, so the results are identical; only the top-k loop shrinks
    with the owned rows. Uses send/receive semaphores 1..4.
    """
    (
        selection_source_vmem,
        selections_vmem,
        send_semaphores,
        receive_semaphores,
        selected_expert_count,
        local_expert_count,
        expert_tensor_parallel,
        on_local_selection,
        after_selection,
        after_scores,
        live_rows,
    ) = (
        (*step_args[0:1], stage_kwargs.get("selection_source_vmem", _NONE_STAGE_VAL))[
            0
        ],
        (*step_args[1:2], stage_kwargs.get("selections_vmem", _NONE_STAGE_VAL))[0],
        (*step_args[2:3], stage_kwargs.get("send_semaphores", _NONE_STAGE_VAL))[0],
        (*step_args[3:4], stage_kwargs.get("receive_semaphores", _NONE_STAGE_VAL))[0],
        stage_kwargs["selected_expert_count"],
        stage_kwargs["local_expert_count"],
        stage_kwargs["expert_tensor_parallel"],
        stage_kwargs["on_local_selection"],
        stage_kwargs["after_selection"],
        stage_kwargs.get("after_scores", _NONE_STAGE_VAL),
        stage_kwargs.get("live_rows", _NONE_STAGE_VAL),
    )
    rank = jax.lax.axis_index("tp")
    rows = owned_hidden_ref.shape[0]
    pair = rank % 8 // 2
    expert_count = routing_bias_ref.shape[-1]
    np.testing.assert_(
        not (rows * 4 != batch_size or expert_count != _FULL_SEQ_PAD_LIMIT),
        "Sharded routing needs one row block per local rank pair",
    )
    router_lane = rank % 2
    local_scores = _dot(owned_hidden_ref[...], routing_projection_vmem[0]).reshape(
        rows, 512
    )
    k = _route_owned_rows_stage_a(
        (
            (after_scores, after_selection, expert_count, local_scores, pair),
            (rank, receive_semaphores, router_lane, routing_bias_ref),
            (routing_scores_vmem, rows, selected_expert_count),
            (selection_source_vmem, selections_vmem, send_semaphores),
        )
    )
    gathered = selections_vmem[...]
    all_winners = gathered[:, :k].astype(jnp.int32)
    all_probabilities = gathered[:, k : 2 * k]
    selected_experts = jnp.zeros((batch_size, k), jnp.int32)
    selected_probabilities = jnp.zeros((batch_size, k), jnp.float32)
    local_route_counts = jnp.zeros((batch_size,), jnp.int32)
    route_slots = jnp.arange(k)[None, :]
    for selection in range(k):
        winners = all_winners[:, selection]
        probabilities = all_probabilities[:, selection]
        counts_before_selection = local_route_counts
        owned_locally = winners // local_expert_count == rank // expert_tensor_parallel
        if live_rows is not None:
            owned_locally &= live_rows
        destination_mask = owned_locally[:, None] & (
            route_slots == counts_before_selection[:, None]
        )
        selected_experts = jnp.where(
            destination_mask, (winners % local_expert_count)[:, None], selected_experts
        )
        selected_probabilities = jnp.where(
            destination_mask, probabilities[:, None], selected_probabilities
        )
        local_route_counts = counts_before_selection + owned_locally.astype(jnp.int32)
        on_local_selection(
            winners, owned_locally, counts_before_selection, selected_experts
        )
    return (
        selected_experts,
        selected_probabilities,
        local_route_counts,
        gathered[:, 2 * k]
        if live_rows is None
        else jnp.where(live_rows, gathered[:, 2 * k], 1.0),
    )


def _route_owned_rows_stage_a(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for route owned rows stage."""
    (
        (after_scores, after_selection, expert_count, local_scores, pair),
        (rank, receive_semaphores, router_lane, routing_bias_ref),
        (routing_scores_vmem, rows, selected_expert_count),
        (selection_source_vmem, selections_vmem, send_semaphores),
    ) = stage_ctx
    column = jax.lax.broadcasted_iota(jnp.int32, local_scores.shape, 1)
    local_scores = jnp.where(
        (router_lane == 0) | (column < _ATTN_OUT_WIDTH), local_scores, 0.0
    )
    score_half = routing_scores_vmem.at[pl.ds(0, rows), pl.ds(router_lane * 512, 512)]
    score_half[...] = local_scores
    if after_scores is not None:
        after_scores()
    copy = tpu.make_async_remote_copy(
        score_half,
        score_half,
        send_semaphores.at[1],
        receive_semaphores.at[1],
        device_id=(rank ^ 1,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()
    copy.wait()
    scores = routing_scores_vmem.at[pl.ds(0, rows), pl.ds(0, expert_count)]
    uncorrected_probabilities = jax.nn.sigmoid(scores[...])
    scores[...] = uncorrected_probabilities + routing_bias_ref[...]
    expert_ids = jnp.arange(expert_count)[None, :]
    winners_by_slot, probabilities_by_slot = ([], [])
    probability_totals = jnp.zeros((rows,), jnp.float32)
    for selection in range(selected_expert_count):
        winners = jnp.argmax(scores[...], axis=1).astype(jnp.int32)
        winner_mask = expert_ids == winners[:, None]
        probabilities = jnp.sum(
            jnp.where(winner_mask, uncorrected_probabilities, 0), axis=1
        )
        probability_totals += probabilities
        scores[...] = jnp.where(winner_mask, -jnp.inf, scores[...])
        winners_by_slot.append(winners)
        probabilities_by_slot.append(probabilities)
        after_selection(selection)
    k = selected_expert_count
    selection_source_vmem[...] = jnp.concatenate(
        (
            jnp.stack(winners_by_slot, axis=1).astype(jnp.float32),
            jnp.stack(probabilities_by_slot, axis=1),
            probability_totals[:, None],
            jnp.zeros((rows, 128 - 2 * k - 1), jnp.float32),
        ),
        axis=1,
    )
    for owner in range(4):

        @pl.when(pair == owner)
        def gather_selections(owner: TensorLike = owner) -> None:
            """Compute and stage Pallas VMEM state for gather selections."""
            destination = selections_vmem.at[pl.ds(owner * rows, rows)]
            destination[...] = selection_source_vmem[...]
            copies = []
            for offset in range(3, 0, -1):
                copy = tpu.make_async_remote_copy(
                    selection_source_vmem,
                    destination,
                    send_semaphores.at[1 + offset],
                    receive_semaphores.at[1 + offset],
                    device_id=(rank ^ 2 * offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()

    return k


def _start_initial_expert_weight_copies(
    selected_experts: TensorLike,
    local_route_counts: TensorLike,
    copy_expert: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Prefetch the first routed experts in their execution order.

    This uses the same cross-row ordering as the expert stream, so it
    works for both shared and divergent routes without specializing on
    batch size.
    """
    slot_count, already_started, group_pending_routes = (
        stage_kwargs["slot_count"],
        stage_kwargs["already_started"],
        stage_kwargs["group_pending_routes"],
    )
    selected_expert_count = selected_experts.shape[1]
    route_slots = jnp.arange(selected_expert_count)[None, :]
    processed = jnp.zeros(selected_experts.shape, jnp.bool_)

    def next_expert(completed: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for next expert."""
        pending = (route_slots < local_route_counts[:, None]) & ~completed
        if group_pending_routes:
            minimum_expert = jnp.min(
                jnp.where(pending, selected_experts.astype(jnp.float32), jnp.inf)
            )
            has_expert = jnp.any(pending)
            expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
            consumed = pending & (selected_experts == expert)
        else:
            rows_with_routes = jnp.any(pending, axis=1)
            current_slots = jnp.sum(completed.astype(jnp.int32), axis=1)
            current_slot_mask = route_slots == current_slots[:, None]
            current_experts = jnp.sum(
                jnp.where(current_slot_mask, selected_experts, 0), axis=1
            )
            minimum_expert = jnp.min(
                jnp.where(
                    rows_with_routes, current_experts.astype(jnp.float32), jnp.inf
                )
            )
            has_expert = jnp.any(rows_with_routes)
            expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
            consumed = (rows_with_routes & (current_experts == expert))[
                :, None
            ] & current_slot_mask
        return (expert, consumed, has_expert)

    for slot in range(slot_count):
        expert, consumed, has_expert = next_expert(processed)

        @pl.when(has_expert & (slot >= already_started))
        def start_copy(expert: TensorLike = expert, slot: TensorLike = slot) -> None:
            """Compute and stage Pallas VMEM state for start copy."""
            _start_async_copies(copy_expert(expert, slot))

        processed |= consumed


def _quantized_expert_dot(
    source: TensorLike,
    packed: TensorLike,
    scales: TensorLike,
    accumulator: TensorLike,
    contraction_size: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Multiply by one resident packed MXFP4 matrix."""
    block_major, dot_tile = (
        stage_kwargs.get("block_major", False),
        stage_kwargs.get("dot_tile", _NONE_STAGE_VAL),
    )
    dot_tile = MXFP4_DOT_TILE or 0 if dot_tile is None else dot_tile
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)
    np.testing.assert_(
        packed.dtype == jnp.uint32, "Expert weights must use packed MXFP4"
    )
    split_small_contraction = contraction_size == _LATENT_SUB_WIDTH and block_major
    block_size = 256 if split_small_contraction else min(MXFP4_BLOCK, contraction_size)
    _static_tiles, _tile = _quantized_expert_dot_stage_c(
        (
            (accumulator, block_size, contraction_size, dot_tile, packed, scales),
            (source, split_small_contraction),
        )
    )


def _quantized_expert_dot_stage_c(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for quantized stage."""
    (
        (accumulator, block_size, contraction_size, dot_tile, packed, scales),
        (source, split_small_contraction),
    ) = stage_ctx
    static_tiles = MXFP4_STATIC_TILE > 0 and (not split_small_contraction)

    def tile(start: TensorLike, size: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for tile."""
        return _quantized_expert_dot_tile_stage_b(
            (
                (accumulator, dot_tile, packed, scales, source),
                (split_small_contraction, static_tiles),
            ),
            (start, size),
        )

    if static_tiles:
        for start in range(0, contraction_size, MXFP4_STATIC_TILE):
            tile(start, min(MXFP4_STATIC_TILE, contraction_size - start))
    elif split_small_contraction:
        for start in range(0, contraction_size, block_size):
            tile(start, block_size)
    else:

        @pl.loop(0, contraction_size // block_size)
        def full_tile(index: TensorLike) -> None:
            """Compute and stage Pallas VMEM state for full tile."""
            tile(index * block_size, block_size)

        if contraction_size % block_size:
            tile(
                contraction_size // block_size * block_size,
                contraction_size % block_size,
            )
    return (static_tiles, tile)


def _quantized_expert_dot_tile_stage_b_stage_d(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for quantized stage."""
    ((packed, scales, size, source, split_small_contraction, start), static_tiles) = (
        stage_ctx
    )
    if static_tiles:
        values = tpu.bitcast(
            packed[start // 8 : (start + size) // 8, :], jnp.float4_e2m1fn
        ).astype(jnp.float8_e4m3fn)
    else:
        values = tpu.bitcast(
            packed[pl.ds(pl.multiple_of(start // 8, 8), size // 8), :],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
    if static_tiles or split_small_contraction:
        exponents = scales[...][start // 32 : (start + size) // 32, :].astype(
            jnp.uint32
        )
    else:
        exponents = scales[
            pl.ds(pl.multiple_of(start // 32, 32), size // 32), :
        ].astype(jnp.uint32)
    scale = jax.lax.bitcast_convert_type(exponents << 23, jnp.float32)
    activation_alignment = 128 if split_small_contraction else 512
    if static_tiles:
        activation = source[:, start : start + size].astype(jnp.bfloat16)
    else:
        activation = source[
            :, pl.ds(pl.multiple_of(start, activation_alignment), size)
        ].astype(jnp.bfloat16)
    return (activation, scale, values)


def _quantized_expert_dot_tile_stage_b(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for tile."""
    (
        (accumulator, dot_tile, packed, scales, source),
        (split_small_contraction, static_tiles),
    ) = stage_ctx
    start, size = stage_args
    left = partial = _NONE_STAGE_VAL
    activation, scale, values = _quantized_expert_dot_tile_stage_b_stage_d(
        ((packed, scales, size, source, split_small_contraction, start), static_tiles)
    )
    if dot_tile and (not split_small_contraction) and (size > dot_tile):
        partials = []
        for offset in range(0, size, dot_tile):
            width = min(dot_tile, size - offset)
            mask = jnp.arange(width // 32)[:, None] == jnp.arange(width)[None, :] // 32
            left = jnp.where(
                mask[None],
                activation[:, None, offset : offset + width],
                jnp.bfloat16(0),
            )
            partials.append(
                jax.lax.dot_general(
                    left,
                    values[offset : offset + width],
                    (((2,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
            )
        partial = jnp.concatenate(partials, axis=1)
        accumulator[...] += jnp.sum(partial * scale[None], axis=1)
        return
    scale_block_mask = (
        jnp.arange(size // 32)[:, None] == jnp.arange(size)[None, :] // 32
    )
    if split_small_contraction and activation.shape[0] == 1:
        left = jnp.where(scale_block_mask, activation, jnp.bfloat16(0))
        scaled = (
            jax.lax.dot_general(
                left,
                values,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            * scale
        )
        total = scaled[0:1]
        for block in range(1, size // 32):
            total = total + scaled[block : block + 1]
        accumulator[...] += total
    elif split_small_contraction:
        left = jnp.where(
            scale_block_mask[:, None, :], activation[None, :, :], jnp.bfloat16(0)
        )
        partial = jax.lax.dot_general(
            left, values, (((2,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        accumulator[...] += jnp.sum(partial * scale[:, None, :], axis=0)
    else:
        left = jnp.where(
            scale_block_mask[None], activation[:, None, :], jnp.bfloat16(0)
        )
        partial = jax.lax.dot_general(
            left, values, (((2,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        accumulator[...] += jnp.sum(partial * scale[None], axis=1)


def _paired_expert_gate_up(
    source: TensorLike,
    gate_up_vmem: TensorLike,
    gate_up_scale_vmem: TensorLike,
    accumulator: TensorLike,
    first_slot: TensorLike,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Run two resident expert gate/up projections.

    With separate compact MXFP4 sources, only the matching row/weight
    products are consumed. Keep cross-expert entries zero while
    preserving each diagonal's scale reduction and tile accumulation
    order.
    """
    second_slot, contraction_size, second_source, active_capacity = (
        (*step_args[0:1], stage_kwargs.get("second_slot", _NONE_STAGE_VAL))[0],
        (*step_args[1:2], stage_kwargs.get("contraction_size", _NONE_STAGE_VAL))[0],
        stage_kwargs.get("second_source", _NONE_STAGE_VAL),
        stage_kwargs.get("active_capacity", _NONE_STAGE_VAL),
    )
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)

    def tile(start: TensorLike, size: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for tile."""
        packed = jnp.concatenate(
            (
                gate_up_vmem[first_slot, pl.ds(start // 8, size // 8), :],
                gate_up_vmem[second_slot, pl.ds(start // 8, size // 8), :],
            ),
            axis=1,
        )
        values = tpu.bitcast(packed, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)
        exponents = jnp.concatenate(
            (
                gate_up_scale_vmem[first_slot, pl.ds(start // 32, size // 32), :],
                gate_up_scale_vmem[second_slot, pl.ds(start // 32, size // 32), :],
            ),
            axis=1,
        ).astype(jnp.uint32)
        scale = jax.lax.bitcast_convert_type(exponents << 23, jnp.float32)
        activation = source[:, pl.ds(start, size)].astype(jnp.bfloat16)
        if second_source is not None:
            activation = jnp.concatenate(
                (activation, second_source[:, pl.ds(start, size)].astype(jnp.bfloat16)),
                axis=0,
            )
        scale_block_mask = (
            jnp.arange(size // 32)[:, None] == jnp.arange(size)[None, :] // 32
        )
        left = jnp.where(
            scale_block_mask[None], activation[:, None, :], jnp.bfloat16(0)
        )
        if second_source is not None:
            rows = source.shape[0]
            active = rows if active_capacity is None else active_capacity
            columns = values.shape[1] // 2
            for half in range(2):
                partial = jax.lax.dot_general(
                    left[half * rows : half * rows + active],
                    values[:, half * columns : (half + 1) * columns],
                    (((2,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                result = jnp.sum(
                    partial * scale[None, :, half * columns : (half + 1) * columns],
                    axis=1,
                )
                accumulator[
                    half * rows : (half + 1) * rows,
                    half * columns : (half + 1) * columns,
                ] += (
                    result
                    if active_capacity is None
                    else jnp.pad(result, ((0, rows - active), (0, 0)))
                )
        else:
            partial = jax.lax.dot_general(
                left,
                values,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            accumulator[...] += jnp.sum(partial * scale[None], axis=1)

    tile_size = MXFP4_PAIR_TILE
    for start in range(0, contraction_size, tile_size):
        tile(start, min(tile_size, contraction_size - start))


def _paired_order_expert_gate_up(
    source: TensorLike,
    packed: TensorLike,
    scales: TensorLike,
    accumulator: TensorLike,
    contraction_size: TensorLike,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Compute Pallas VMEM tensors for paired stage.

    Project one expert in the paired gate/up's FP32 accumulation
    order.

    Pair membership changes when null routes disappear. Tail and
    overflow projections must retain the paired tile's scale reduction
    and accumulation order so an unrelated row cannot change a live
    expert's rounded result.
    """
    tile_size: TensorLike = (
        *step_args[0:1],
        stage_kwargs.get("tile_size", _NONE_STAGE_VAL),
    )[0]
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)
    for start in range(0, contraction_size, tile_size):
        size = min(tile_size, contraction_size - start)
        values = tpu.bitcast(
            packed[pl.ds(start // 8, size // 8), :], jnp.float4_e2m1fn
        ).astype(jnp.float8_e4m3fn)
        exponents = scales[pl.ds(start // 32, size // 32), :].astype(jnp.uint32)
        scale = jax.lax.bitcast_convert_type(exponents << 23, jnp.float32)
        activation = source[:, pl.ds(start, size)].astype(jnp.bfloat16)
        mask = jnp.arange(size // 32)[:, None] == jnp.arange(size)[None, :] // 32
        left = jnp.where(mask[None], activation[:, None, :], jnp.bfloat16(0))
        partial = jax.lax.dot_general(
            left, values, (((2,), (0,)), ((), ())), preferred_element_type=jnp.float32
        )
        accumulator[...] += jnp.sum(partial * scale[None], axis=1)


def _dot_scoped(function: TensorLike, name_of: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for dot scoped.

    Profiling: trace each routed-expert product as its own e_dot_*
    phase.
    """

    @functools.wraps(function)
    def scoped(*args: TensorLike, **kwargs: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for scoped."""
        with _phase(name_of(*args, **kwargs)):
            return function(*args, **kwargs)

    return scoped


if PHASE_SCOPES:
    _quantized_expert_dot = _dot_scoped(
        _quantized_expert_dot, lambda *_args, **_kwargs: "e_dot_mxfp4"
    )
    _paired_expert_gate_up = _dot_scoped(
        _paired_expert_gate_up, lambda *_args, **_kwargs: "e_dot_pair"
    )


def _run_routed_expert_stream(
    source: TensorLike,
    selected_experts: TensorLike,
    selected_probabilities: TensorLike,
    local_route_counts: TensorLike,
    probability_totals: TensorLike,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Execute the next route shared by any ready batch rows.

    Rows retain their own routing order. When several rows have the
    same next expert, its weights are loaded once and its projections
    carry the batch dimension. The loop is bounded by the distinct
    experts (grouped routes) or the route count, never by batch size.
    """
    (
        expert_gate_up_vmem,
        expert_gate_up_scale_vmem,
        expert_down_vmem,
        expert_down_scale_vmem,
        expert_output,
        gate_up_accumulator,
        down_accumulator,
        processed_routes_vmem,
        first_compact_source_vmem,
        second_compact_source_vmem,
        copy_expert,
        expert_input_size,
        expert_width,
        expert_output_size,
        group_pending_routes,
        progress,
    ) = (
        (*step_args[0:1], stage_kwargs.get("expert_gate_up_vmem", _NONE_STAGE_VAL))[0],
        (
            *step_args[1:2],
            stage_kwargs.get("expert_gate_up_scale_vmem", _NONE_STAGE_VAL),
        )[0],
        (*step_args[2:3], stage_kwargs.get("expert_down_vmem", _NONE_STAGE_VAL))[0],
        (*step_args[3:4], stage_kwargs.get("expert_down_scale_vmem", _NONE_STAGE_VAL))[
            0
        ],
        (*step_args[4:5], stage_kwargs.get("expert_output", _NONE_STAGE_VAL))[0],
        (*step_args[5:6], stage_kwargs.get("gate_up_accumulator", _NONE_STAGE_VAL))[0],
        (*step_args[6:7], stage_kwargs.get("down_accumulator", _NONE_STAGE_VAL))[0],
        (*step_args[7:8], stage_kwargs.get("processed_routes_vmem", _NONE_STAGE_VAL))[
            0
        ],
        (
            *step_args[8:9],
            stage_kwargs.get("first_compact_source_vmem", _NONE_STAGE_VAL),
        )[0],
        (
            *step_args[9:10],
            stage_kwargs.get("second_compact_source_vmem", _NONE_STAGE_VAL),
        )[0],
        (*step_args[10:11], stage_kwargs.get("copy_expert", _NONE_STAGE_VAL))[0],
        stage_kwargs["expert_input_size"],
        stage_kwargs["expert_width"],
        stage_kwargs["expert_output_size"],
        stage_kwargs["group_pending_routes"],
        stage_kwargs.get("progress", _NONE_STAGE_VAL),
    )
    batch_size, selected_expert_count = selected_experts.shape
    (_dot_tile, expert_dot, singleton_block, stage_blocks) = (
        _run_routed_expert_stream_stage_m(batch_size, expert_input_size)
    )
    if PHASE_SCOPES:
        unscoped_copy_expert = copy_expert

        def copy_expert(
            expert: TensorLike, slot: TensorLike, **expert_kwargs: TensorLike
        ) -> TensorLike:
            """Compute and stage Pallas VMEM state for copy expert."""
            copies = unscoped_copy_expert(expert, slot, **expert_kwargs)
            return SimpleNamespace(
                **{
                    name: None if copy is None else _ScopedWait(copy)
                    for name, copy in vars(copies).items()
                }
            )

    next_expert, route_slots = _run_routed_expert_stream_stage_n(
        (
            (expert_output, group_pending_routes, local_route_counts),
            (processed_routes_vmem, selected_expert_count, selected_experts),
        )
    )

    def finish_expert(
        active_rows: TensorLike,
        current_slot_mask: TensorLike,
        copies: TensorLike,
        column: TensorLike,
        slot: TensorLike,
        *step_args: TensorLike,
        **stage_kwargs: TensorLike,
    ) -> None:
        """Compute and stage Pallas VMEM state for finish expert."""
        output_weights: TensorLike = (
            *step_args[0:1],
            stage_kwargs.get("output_weights", _NONE_STAGE_VAL),
        )[0]
        gate_up = (
            gate_up_accumulator[
                :,
                pl.ds(pl.multiple_of(column, 128), 2 * expert_width)
                if share_full_finish
                else slice(column, column + 2 * expert_width),
            ]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        middle = _gated_activation(gate_up[:, :expert_width], gate_up[:, expert_width:])
        gate_up_accumulator[:, :expert_width] = middle.astype(jnp.float32)
        copies.down.wait()
        copies.down_scales.wait()
        expert_dot(
            gate_up_accumulator.at[:, :expert_width],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator,
            expert_width,
            block_major=batch_size >= _ROUTED_EXPERTS_PER_TOKEN,
        )
        if output_weights is None:
            route_probabilities = (
                jnp.sum(
                    jnp.where(
                        current_slot_mask, selected_probabilities, jnp.float32(0)
                    ),
                    axis=1,
                )
                / probability_totals
            )
            output_weights = jnp.where(active_rows, route_probabilities, jnp.float32(0))
        expert_output[:, :expert_output_size] += (
            down_accumulator[...].astype(jnp.bfloat16).astype(jnp.float32)
            * output_weights[:, None]
        )

    (
        (compact_capacity, finish_singleton_expert, first_single_source),
        (row_bitmask_words, second_single_source, stage_singleton_source),
    ) = _run_routed_expert_stream_stage_i(
        (
            (batch_size, down_accumulator, expert_dot, expert_down_scale_vmem),
            (expert_down_vmem, expert_input_size, expert_output),
            (expert_output_size, expert_width, first_compact_source_vmem),
            (gate_up_accumulator, probability_totals, second_compact_source_vmem),
            (selected_probabilities, singleton_block, source),
        )
    )
    (
        (compact_route_data, compact_row_plan, compact_row_positions),
        (reuse_row_plan, stage_compact_rows, _stage_planned_rows),
    ) = _run_routed_expert_stream_stage_l(
        (
            (batch_size, compact_capacity, expert_input_size, probability_totals),
            (row_bitmask_words, selected_probabilities, source, stage_blocks),
        )
    )

    def finish_compact_expert(
        active_rows: TensorLike,
        current_slot_mask: TensorLike,
        copies: TensorLike,
        first_row: TensorLike,
        column: TensorLike,
        *step_args: TensorLike,
        **stage_kwargs: TensorLike,
    ) -> None:
        """Compute and stage Pallas VMEM state for finish compact expert."""
        slot, offset, wait_down, route_data = (
            (*step_args[0:1], stage_kwargs.get("slot", _NONE_STAGE_VAL))[0],
            (*step_args[1:2], stage_kwargs.get("offset", 0))[0],
            stage_kwargs.get("wait_down", True),
            stage_kwargs.get("route_data", _NONE_STAGE_VAL),
        )
        return _run_routed_expert_stream_finish_comp_stage_s(
            (
                (compact_capacity, compact_route_data, down_accumulator, expert_dot),
                (expert_down_scale_vmem, expert_down_vmem, expert_output),
                (expert_output_size, expert_width, gate_up_accumulator),
                share_compact_finish,
            ),
            (
                (active_rows, current_slot_mask, copies, first_row, column, slot),
                (offset, wait_down, route_data),
            ),
        )

    compact_overflow = _use_compact_overflow(
        batch_size, group_pending_routes=group_pending_routes
    )
    compact_only = gate_up_accumulator.shape[0] < batch_size or (
        compact_overflow and down_accumulator.shape[0] < batch_size
    )
    share_compact_finish = _run_routed_expert_stream_stage_o(
        batch_size, compact_capacity, compact_only, compact_overflow
    )
    share_full_finish = _run_routed_expert_stream_stage_q(
        compact_only, share_compact_finish
    )
    (pair_count, routes_are_shared, run_expert_in_passes) = (
        _run_routed_expert_stream_stage_k(
            (
                (batch_size, compact_capacity, compact_only, compact_route_data),
                (compact_row_plan, compact_row_positions, copy_expert),
                (down_accumulator, expert_dot, expert_down_scale_vmem),
                (expert_down_vmem, expert_gate_up_scale_vmem, expert_gate_up_vmem),
                (expert_input_size, expert_output, expert_output_size, expert_width),
                (finish_compact_expert, first_compact_source_vmem),
                (gate_up_accumulator, group_pending_routes, local_route_counts),
                (probability_totals, reuse_row_plan, route_slots),
                (selected_expert_count, selected_experts, selected_probabilities),
                (source, stage_compact_rows),
            )
        )
    )
    _run_routed_expert_stream_stage_r(
        (
            (batch_size, compact_capacity, compact_only, compact_overflow),
            (compact_route_data, compact_row_plan, copy_expert, expert_dot),
            (expert_gate_up_scale_vmem, expert_gate_up_vmem, expert_input_size),
            (expert_width, finish_compact_expert, finish_expert),
            (finish_singleton_expert, first_compact_source_vmem),
            (first_single_source, gate_up_accumulator, group_pending_routes),
            (next_expert, pair_count, probability_totals, processed_routes_vmem),
            (progress, reuse_row_plan, routes_are_shared, run_expert_in_passes),
            (second_compact_source_vmem, second_single_source),
            (selected_expert_count, selected_experts, selected_probabilities),
            (share_compact_finish, share_full_finish, source, stage_compact_rows),
            stage_singleton_source,
        )
    )


def _run_routed_expert_stream_stage_m(
    batch_size: TensorLike, expert_input_size: TensorLike
) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    dot_tile = (
        MXFP4_DOT_TILE
        if MXFP4_DOT_TILE is not None
        else 1024
        if batch_size == _GATED_GROUP_STRIDE
        else 512
        if batch_size in (1, 2, 4, 8, 16, 32, 64, 128)
        else 0
    )
    singleton_block = (
        SINGLETON_BLOCK
        if SINGLETON_BLOCK is not None
        else batch_size in (16, 32, 64, 96, 128)
    )
    stage_blocks = (
        STAGE_BLOCKS
        if STAGE_BLOCKS is not None
        else batch_size in (16, 32, 64, 96, 128)
    )

    def expert_dot(
        source: TensorLike,
        packed: TensorLike,
        scales: TensorLike,
        accumulator: TensorLike,
        contraction_size: TensorLike,
        **expert_kwargs: TensorLike,
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for expert dot."""
        if batch_size > 1 and contraction_size == expert_input_size:
            return _paired_order_expert_gate_up(
                source, packed, scales, accumulator, contraction_size, MXFP4_PAIR_TILE
            )
        return _quantized_expert_dot(
            source,
            packed,
            scales,
            accumulator,
            contraction_size,
            dot_tile=dot_tile,
            **expert_kwargs,
        )

    return (dot_tile, expert_dot, singleton_block, stage_blocks)


def _run_routed_expert_stream_stage_n(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (expert_output, group_pending_routes, local_route_counts),
        (processed_routes_vmem, selected_expert_count, selected_experts),
    ) = stage_ctx
    route_slots = jnp.arange(selected_expert_count)[None, :]
    processed_routes_vmem[...] = jnp.zeros(processed_routes_vmem.shape, jnp.int32)
    expert_output[...] = jnp.zeros(expert_output.shape, jnp.float32)

    def next_expert(processed: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for next expert."""
        active_rows = expert = has_expert = _NONE_STAGE_VAL
        minimum_expert = _NONE_STAGE_VAL
        pending = (route_slots < local_route_counts[:, None]) & ~processed
        if group_pending_routes:
            minimum_expert = jnp.min(
                jnp.where(pending, selected_experts.astype(jnp.float32), jnp.inf)
            )
            has_expert = jnp.any(pending)
            expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
            selected_slot_mask = pending & (selected_experts == expert)
            active_rows = jnp.any(selected_slot_mask, axis=1)
            return (
                expert,
                active_rows,
                selected_slot_mask,
                selected_slot_mask,
                has_expert,
            )
        rows_with_routes = jnp.any(pending, axis=1)
        current_slots = jnp.sum(processed.astype(jnp.int32), axis=1)
        current_slot_mask = route_slots == current_slots[:, None]
        current_experts = jnp.sum(
            jnp.where(current_slot_mask, selected_experts, 0), axis=1
        )
        minimum_expert = jnp.min(
            jnp.where(rows_with_routes, current_experts.astype(jnp.float32), jnp.inf)
        )
        has_expert = jnp.any(rows_with_routes)
        expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
        active_rows = rows_with_routes & (current_experts == expert)
        consumed = active_rows[:, None] & current_slot_mask
        return (expert, active_rows, current_slot_mask, consumed, has_expert)

    return (next_expert, route_slots)


def _run_routed_expert_stream_stage_i(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (batch_size, down_accumulator, expert_dot, expert_down_scale_vmem),
        (expert_down_vmem, expert_input_size, expert_output),
        (expert_output_size, expert_width, first_compact_source_vmem),
        (gate_up_accumulator, probability_totals, second_compact_source_vmem),
        (selected_probabilities, singleton_block, source),
    ) = stage_ctx

    def finish_singleton_expert(
        active_rows: TensorLike,
        current_slot_mask: TensorLike,
        copies: TensorLike,
        compact_row: TensorLike,
        column: TensorLike,
        *step_args: TensorLike,
        **stage_kwargs: TensorLike,
    ) -> None:
        """Finish one expert whose route is active for exactly one batch row."""
        slot: TensorLike = (*step_args[0:1], stage_kwargs.get("slot", _NONE_STAGE_VAL))[
            0
        ]
        gate_up = (
            gate_up_accumulator[
                compact_row : compact_row + 1, column : column + 2 * expert_width
            ]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        middle = _gated_activation(gate_up[:, :expert_width], gate_up[:, expert_width:])
        gate_up_accumulator[compact_row : compact_row + 1, :expert_width] = (
            middle.astype(jnp.float32)
        )
        copies.down.wait()
        copies.down_scales.wait()
        expert_dot(
            gate_up_accumulator.at[compact_row : compact_row + 1, :expert_width],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator.at[:1],
            expert_width,
            block_major=True,
        )
        route_probabilities = (
            jnp.sum(
                jnp.where(current_slot_mask, selected_probabilities, jnp.float32(0)),
                axis=1,
            )
            / probability_totals
        )
        singleton_output = down_accumulator[:1].astype(jnp.bfloat16).astype(jnp.float32)
        row = jnp.sum(jnp.where(active_rows, jnp.arange(batch_size), 0))
        probability = jnp.sum(
            jnp.where(active_rows, route_probabilities, jnp.float32(0))
        )
        output_row = expert_output.at[pl.ds(row, 1), :expert_output_size]
        output_row[...] += singleton_output * probability

    compact_capacity = first_compact_source_vmem.shape[0]
    if compact_capacity > 1:
        first_single_source = first_compact_source_vmem.at[:1]
        second_single_source = second_compact_source_vmem.at[:1]
    else:
        first_single_source = first_compact_source_vmem
        second_single_source = second_compact_source_vmem

    def stage_singleton_source(
        active_rows: TensorLike, destination: TensorLike
    ) -> None:
        """Compute and stage Pallas stage singleton source."""
        active_index = _NONE_STAGE_VAL
        if singleton_block and batch_size % 16 == 0:
            active_index = jnp.sum(jnp.where(active_rows, jnp.arange(batch_size), 0))
            block = source[
                pl.ds(pl.multiple_of(active_index // 16 * 16, 16), 16),
                :expert_input_size,
            ].astype(jnp.float32)
            within = (
                jax.lax.broadcasted_iota(jnp.int32, block.shape, 0) == active_index % 16
            )
            destination[...] = jnp.sum(
                jnp.where(within, block, 0.0), axis=0, keepdims=True
            ).astype(destination.dtype)
            return
        active_index = jnp.sum(jnp.where(active_rows, jnp.arange(batch_size), 0))
        for row in range(batch_size):

            @pl.when(active_index == row)
            def copy_indexed_row(row: TensorLike = row) -> None:
                """Compute and stage Pallas copy indexed row."""
                destination[...] = source[row : row + 1, :expert_input_size].astype(
                    destination.dtype
                )

    def row_bitmask_words(active_rows: TensorLike) -> TensorLike:
        """The active rows as scalar bit words (32 rows per word)."""
        index = jnp.arange(batch_size)
        bits = active_rows.astype(jnp.int32) << index % 32
        return [
            jnp.sum(jnp.where(index // 32 == word, bits, 0))
            for word in range(-(-batch_size // 32))
        ]

    return (
        (compact_capacity, finish_singleton_expert, first_single_source),
        (row_bitmask_words, second_single_source, stage_singleton_source),
    )


def _run_routed_expert_stream_stage_l(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (batch_size, compact_capacity, expert_input_size, probability_totals),
        (row_bitmask_words, selected_probabilities, source, stage_blocks),
    ) = stage_ctx
    reuse_row_plan = stage_blocks and batch_size in (32, 64, 96, 128)

    def compact_row_positions(active_rows: TensorLike) -> TensorLike:
        """Compute and stage Pallas compact row positions."""
        count_dtype = jnp.float32 if reuse_row_plan else jnp.int32
        index = jnp.arange(batch_size)
        return jnp.sum(
            jnp.where(
                index[None, :] < index[:, None],
                active_rows[None, :].astype(count_dtype),
                0,
            ),
            axis=1,
        ).astype(jnp.int32)

    (compact_row_plan, stage_planned_rows) = _run_routed_expert_stream_stage_j(
        batch_size, compact_capacity, compact_row_positions, expert_input_size, source
    )

    def stage_compact_rows(
        active_rows: TensorLike,
        destination: TensorLike,
        offset: TensorLike = 0,
        row_plan: TensorLike = None,
    ) -> None:
        """Compute and stage Pallas VMEM state for stage compact rows."""
        return _run_routed_expert_stream_stage_compa_stage_f(
            (
                (batch_size, compact_capacity, compact_row_plan, expert_input_size),
                (reuse_row_plan, row_bitmask_words, source, stage_blocks),
                stage_planned_rows,
            ),
            ((active_rows, destination, offset, row_plan),),
        )

    def compact_route_data(
        active_rows: TensorLike,
        current_slot_mask: TensorLike,
        offset: TensorLike = 0,
        row_plan: TensorLike = None,
    ) -> TensorLike:
        """Compute and stage Pallas compact route data."""
        route_probabilities = (
            jnp.sum(
                jnp.where(current_slot_mask, selected_probabilities, jnp.float32(0)),
                axis=1,
            )
            / probability_totals
        )
        if reuse_row_plan:
            if row_plan is None:
                row_plan = compact_row_plan(active_rows, offset)
            indices, valid = row_plan
            selected = (jnp.arange(batch_size)[:, None] == indices[None, :]) & valid[
                None, :
            ]
            selector = selected.astype(jnp.bfloat16)
            in_pass = jnp.any(selected, axis=1)
        else:
            before = compact_row_positions(active_rows)
            selector = (
                (before[:, None] - offset == jnp.arange(compact_capacity)[None, :])
                & active_rows[:, None]
            ).astype(jnp.bfloat16)
            in_pass = (
                active_rows & (before >= offset) & (before < offset + compact_capacity)
            )
        return (selector, jnp.where(in_pass, route_probabilities, jnp.float32(0)))

    return (
        (compact_route_data, compact_row_plan, compact_row_positions),
        (reuse_row_plan, stage_compact_rows, stage_planned_rows),
    )


def _run_routed_expert_stream_stage_j(
    batch_size: TensorLike,
    compact_capacity: TensorLike,
    compact_row_positions: TensorLike,
    expert_input_size: TensorLike,
    source: TensorLike,
) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""

    def compact_row_plan(
        active_rows: TensorLike, offset: TensorLike = 0, before: TensorLike = None
    ) -> TensorLike:
        """Compute and stage Pallas compact row plan."""
        if before is None:
            before = compact_row_positions(active_rows)
        selected = active_rows[None, :] & (
            before[None, :] == offset + jnp.arange(compact_capacity)[:, None]
        )
        indices = jnp.sum(
            jnp.where(
                selected, jnp.arange(batch_size).astype(jnp.float32)[None, :], 0.0
            ),
            axis=1,
        ).astype(jnp.int32)
        return (indices, jnp.any(selected, axis=1))

    def stage_planned_rows(row_plan: TensorLike, destination: TensorLike) -> None:
        """Compute and stage Pallas stage planned rows."""
        indices, valid = row_plan
        if batch_size in (64, 96, 128):
            one_hot = (
                (indices[:, None] == jnp.arange(batch_size)[None, :]) & valid[:, None]
            ).astype(jnp.bfloat16)
            destination[...] = _dot(one_hot, source[:, :expert_input_size])
            return
        destination[...] = jnp.zeros(destination.shape, destination.dtype)
        if batch_size == _KDA_GROUP_WIDTH:

            @pl.loop(0, batch_size // 16)
            def stage_block(block: TensorLike) -> None:
                """Compute and stage Pallas VMEM state for stage block."""
                in_block = valid & (indices // 16 == block)

                @pl.when(jnp.any(in_block))
                def copy_block() -> None:
                    """Compute and stage Pallas VMEM state for copy block."""
                    rows = source[
                        pl.ds(pl.multiple_of(block * 16, 16), 16), :expert_input_size
                    ]
                    gather_indices = jnp.broadcast_to(
                        (indices % 16).astype(jnp.int16)[:, None],
                        (compact_capacity, expert_input_size),
                    )
                    selected_rows = jnp.take_along_axis(
                        rows,
                        gather_indices,
                        axis=0,
                        mode="promise_in_bounds",
                        wrap_negative_indices=False,
                    ).astype(jnp.float32)
                    destination[...] = jnp.where(
                        in_block[:, None], selected_rows, destination[...]
                    )

            return

        @pl.loop(0, jnp.sum(valid.astype(jnp.float32)).astype(jnp.int32))
        def copy_row(position: TensorLike) -> None:
            """Compute and stage Pallas VMEM state for copy row."""
            index = jnp.sum(
                jnp.where(
                    jnp.arange(compact_capacity) == position,
                    indices.astype(jnp.float32),
                    0.0,
                )
            ).astype(jnp.int32)
            rows = source[
                pl.ds(pl.multiple_of(index // 16 * 16, 16), 16), :expert_input_size
            ]
            gather_indices = jnp.full((1, expert_input_size), index % 16, jnp.int16)
            selected_row = jnp.take_along_axis(
                rows,
                gather_indices,
                axis=0,
                mode="promise_in_bounds",
                wrap_negative_indices=False,
            ).astype(jnp.float32)
            destination[pl.ds(position, 1), :] = selected_row

    return (compact_row_plan, stage_planned_rows)


def _run_routed_expert_stream_stage_compa_stage_f(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for stage compact rows."""
    (
        (batch_size, compact_capacity, compact_row_plan, expert_input_size),
        (reuse_row_plan, row_bitmask_words, source, stage_blocks),
        stage_planned_rows,
    ) = stage_ctx
    ((active_rows, destination, offset, row_plan),) = stage_args
    if reuse_row_plan:
        if row_plan is None:
            row_plan = compact_row_plan(active_rows, offset)
        stage_planned_rows(row_plan, destination)
        return
    destination[...] = jnp.zeros(destination.shape, destination.dtype)
    if stage_blocks and batch_size == _EXPERT_GROUP_STRIDE:
        _run_routed_expert_stream_stage_compa_stage_w(
            (
                (active_rows, batch_size, compact_capacity, destination),
                (expert_input_size, offset, row_bitmask_words, source),
            )
        )
        return
    if stage_blocks and batch_size % 16 == 0:
        index = jnp.arange(batch_size)
        active = active_rows.astype(jnp.int32)

        @pl.loop(0, batch_size // 16)
        def stage_block(block: TensorLike) -> None:
            """Compute and stage Pallas VMEM state for stage block."""
            first = pl.multiple_of(block * 16, 16)
            bits = jnp.sum(jnp.where(index // 16 == block, active << index % 16, 0))
            position = jnp.sum(jnp.where(index < block * 16, active, 0))
            for row in range(16):
                bit = bits >> row & 1
                take = (
                    (bit != 0)
                    & (position >= offset)
                    & (position < offset + compact_capacity)
                )

                @pl.when(take)
                def copy_block_row(
                    row: TensorLike = row, position: TensorLike = position
                ) -> None:
                    """Compute and stage Pallas copy block row."""
                    rows = source[pl.ds(first, 16), :expert_input_size]
                    destination[pl.ds(position - offset, 1), :] = rows[
                        row : row + 1
                    ].astype(jnp.float32)

                position = position + bit

        return
    _run_routed_expert_stream_stage_compa_stage_v(
        (
            (active_rows, batch_size, compact_capacity, destination),
            (expert_input_size, offset, row_bitmask_words, source),
        )
    )


def _run_routed_expert_stream_stage_compa_stage_w(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (active_rows, batch_size, compact_capacity, destination),
        (expert_input_size, offset, row_bitmask_words, source),
    ) = stage_ctx
    (word,) = row_bitmask_words(active_rows)

    def stage_block(block: TensorLike, position: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for stage block."""
        first = pl.multiple_of(block * 16, 16)
        bits = word >> block * 16 & 65535

        def stage_nonempty(position: TensorLike) -> TensorLike:
            """Compute and stage Pallas VMEM state for stage nonempty."""
            for row in range(16):
                bit = bits >> row & 1
                take = (
                    (bit != 0)
                    & (position >= offset)
                    & (position < offset + compact_capacity)
                )

                @pl.when(take)
                def copy_block_row(
                    row: TensorLike = row, position: TensorLike = position
                ) -> None:
                    """Compute and stage Pallas copy block row."""
                    rows = source[pl.ds(first, 16), :expert_input_size]
                    destination[pl.ds(position - offset, 1), :] = rows[
                        row : row + 1
                    ].astype(jnp.float32)

                position = position + bit
            return position

        return stage_nonempty(position)

    jax.lax.fori_loop(0, batch_size // 16, stage_block, jnp.int32(0))
    return word


def _run_routed_expert_stream_stage_compa_stage_v(stage_ctx: TensorLike) -> None:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (active_rows, batch_size, compact_capacity, destination),
        (expert_input_size, offset, row_bitmask_words, source),
    ) = stage_ctx
    words = row_bitmask_words(active_rows)
    position = jnp.int32(0)
    for row in range(batch_size):
        bit = words[row // 32] >> row % 32 & 1
        if isinstance(offset, int) and offset == 0:
            take = bit != 0
        else:
            take = (
                (bit != 0)
                & (position >= offset)
                & (position < offset + compact_capacity)
            )

        @pl.when(take)
        def copy_active_row(
            row: TensorLike = row, position: TensorLike = position
        ) -> None:
            """Compute and stage Pallas copy active row."""
            destination[pl.ds(position - offset, 1), :] = source[
                row : row + 1, :expert_input_size
            ].astype(jnp.float32)

        position = position + bit


def _run_routed_expert_stream_finish_comp_stage_s(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for finish compact expert."""
    (
        (compact_capacity, compact_route_data, down_accumulator, expert_dot),
        (expert_down_scale_vmem, expert_down_vmem, expert_output),
        (expert_output_size, expert_width, gate_up_accumulator),
        share_compact_finish,
    ) = stage_ctx
    (
        (active_rows, current_slot_mask, copies, first_row, column, slot),
        (offset, wait_down, route_data),
    ) = stage_args
    rows = (
        pl.ds(pl.multiple_of(first_row, 8), compact_capacity)
        if share_compact_finish
        else slice(first_row, first_row + compact_capacity)
    )
    gate_up = (
        gate_up_accumulator[
            rows,
            pl.ds(pl.multiple_of(column, 128), 2 * expert_width)
            if share_compact_finish
            else slice(column, column + 2 * expert_width),
        ]
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    middle = _gated_activation(gate_up[:, :expert_width], gate_up[:, expert_width:])
    gate_up_accumulator[rows, :expert_width] = middle.astype(jnp.float32)
    if wait_down:
        copies.down.wait()
        copies.down_scales.wait()
    expert_dot(
        gate_up_accumulator.at[rows, :expert_width],
        expert_down_vmem.at[slot],
        expert_down_scale_vmem.at[slot],
        down_accumulator.at[:compact_capacity],
        expert_width,
        block_major=True,
    )
    if route_data is None:
        route_data = compact_route_data(active_rows, current_slot_mask, offset)
    selector, output_weights = route_data
    selected = _dot(selector, down_accumulator[:compact_capacity].astype(jnp.bfloat16))
    expert_output[:, :expert_output_size] += selected * output_weights[:, None]


def _run_routed_expert_stream_stage_o(
    batch_size: TensorLike,
    compact_capacity: TensorLike,
    compact_only: TensorLike,
    compact_overflow: TensorLike,
) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    return (
        SHARED_EXPERT_FINISHERS
        and batch_size in (16, 32, 64, 96, 128)
        and (compact_capacity == _ROUTED_EXPERTS_PER_TOKEN)
        and (not compact_only or compact_overflow)
    )


def _run_routed_expert_stream_stage_q(
    compact_only: TensorLike, share_compact_finish: TensorLike
) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    return share_compact_finish and (not compact_only)


def _run_routed_expert_stream_stage_k(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (batch_size, compact_capacity, compact_only, compact_route_data),
        (compact_row_plan, compact_row_positions, copy_expert),
        (down_accumulator, expert_dot, expert_down_scale_vmem),
        (expert_down_vmem, expert_gate_up_scale_vmem, expert_gate_up_vmem),
        (expert_input_size, expert_output, expert_output_size, expert_width),
        (finish_compact_expert, first_compact_source_vmem),
        (gate_up_accumulator, group_pending_routes, local_route_counts),
        (probability_totals, reuse_row_plan, route_slots),
        (selected_expert_count, selected_experts, selected_probabilities),
        (source, stage_compact_rows),
    ) = stage_ctx

    def run_expert_in_passes(
        active_rows: TensorLike,
        current_slot_mask: TensorLike,
        copies: TensorLike,
        row_count: TensorLike,
        slot: TensorLike,
        **stage_kwargs: TensorLike,
    ) -> None:
        """One expert with more than C rows, C rows per pass."""
        paired_gate: TensorLike = stage_kwargs.get("paired_gate", False)
        copies.gate_up.wait()
        copies.gate_up_scales.wait()
        copies.down.wait()
        copies.down_scales.wait()
        before = compact_row_positions(active_rows) if reuse_row_plan else None

        @pl.loop(0, (row_count + compact_capacity - 1) // compact_capacity)
        def run_pass(index: TensorLike) -> None:
            """Compute and stage Pallas VMEM state for run pass."""
            offset = index * compact_capacity
            row_plan = (
                compact_row_plan(active_rows, offset, before)
                if reuse_row_plan
                else None
            )
            stage_compact_rows(active_rows, first_compact_source_vmem, offset, row_plan)
            if paired_gate:
                _paired_order_expert_gate_up(
                    first_compact_source_vmem,
                    expert_gate_up_vmem.at[slot],
                    expert_gate_up_scale_vmem.at[slot],
                    gate_up_accumulator.at[:compact_capacity, : 2 * expert_width],
                    expert_input_size,
                    MXFP4_PAIR_TILE,
                )
            else:
                expert_dot(
                    first_compact_source_vmem,
                    expert_gate_up_vmem.at[slot],
                    expert_gate_up_scale_vmem.at[slot],
                    gate_up_accumulator.at[:compact_capacity, : 2 * expert_width],
                    expert_input_size,
                )
            finish_compact_expert(
                active_rows,
                current_slot_mask,
                copies,
                0,
                0,
                slot,
                offset=offset,
                wait_down=False,
                route_data=compact_route_data(
                    active_rows, current_slot_mask, offset, row_plan
                ),
            )

    valid_routes = route_slots < local_route_counts[:, None]
    routes_are_shared = (
        (not group_pending_routes)
        & jnp.all(local_route_counts == local_route_counts[0])
        & jnp.all(~valid_routes | (selected_experts == selected_experts[0:1]))
    )

    @pl.when(routes_are_shared & (not compact_only))
    def run_shared_routes() -> None:
        """Compute and stage Pallas run shared routes."""
        return _run_routed_expert_stream_run_shared_stage_g(
            (
                (batch_size, compact_only, copy_expert, down_accumulator, expert_dot),
                (expert_down_scale_vmem, expert_down_vmem, expert_gate_up_scale_vmem),
                (expert_gate_up_vmem, expert_input_size, expert_output),
                (expert_output_size, expert_width, gate_up_accumulator),
                (local_route_counts, probability_totals, route_slots),
                (selected_experts, selected_probabilities, source),
            )
        )

    if group_pending_routes:
        used = jnp.zeros((1, 128), jnp.bool_)
        for route_slot in range(selected_expert_count):
            used |= jnp.any(
                valid_routes[:, route_slot : route_slot + 1]
                & (
                    selected_experts[:, route_slot : route_slot + 1]
                    == jnp.arange(128)[None, :]
                ),
                axis=0,
                keepdims=True,
            )
        pair_count = (jnp.sum(used.astype(jnp.int32)) + 1) // 2
    else:
        pair_count = (jnp.sum(local_route_counts) + 1) // 2
    return (pair_count, routes_are_shared, run_expert_in_passes)


def _run_routed_expert_stream_run_shared_stage_g(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas run shared routes."""
    (
        (batch_size, compact_only, copy_expert, down_accumulator, expert_dot),
        (expert_down_scale_vmem, expert_down_vmem, expert_gate_up_scale_vmem),
        (expert_gate_up_vmem, expert_input_size, expert_output),
        (expert_output_size, expert_width, gate_up_accumulator),
        (local_route_counts, probability_totals, route_slots),
        (selected_experts, selected_probabilities, source),
    ) = stage_ctx
    if compact_only:
        return

    def shared_expert_at(index: TensorLike) -> TensorLike:
        """Compute and stage Pallas shared expert at."""
        return jnp.sum(
            jnp.where(route_slots[0] == index, selected_experts[0], jnp.int32(0))
        ).astype(jnp.int32)

    def shared_probabilities_at(index: TensorLike) -> TensorLike:
        """Compute and stage Pallas shared probabilities at."""
        return jnp.sum(
            jnp.where(route_slots == index, selected_probabilities, jnp.float32(0)),
            axis=1,
        )

    def wait_for_gate_up(index: TensorLike, slot: TensorLike) -> None:
        """Compute and stage Pallas wait for gate up."""
        copies = copy_expert(shared_expert_at(index), slot, down=False)
        copies.gate_up.wait()
        copies.gate_up_scales.wait()

    def wait_for_down(index: TensorLike, slot: TensorLike) -> None:
        """Compute and stage Pallas wait for down."""
        copies = copy_expert(shared_expert_at(index), slot, gate_up=False)
        copies.down.wait()
        copies.down_scales.wait()

    def finish_shared_expert(
        index: TensorLike, slot: TensorLike, column: TensorLike
    ) -> None:
        """Compute and stage Pallas finish shared expert."""
        gate_up = (
            gate_up_accumulator[:, column : column + 2 * expert_width]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
        middle = _gated_activation(gate_up[:, :expert_width], gate_up[:, expert_width:])
        gate_up_accumulator[:, :expert_width] = middle.astype(jnp.float32)
        wait_for_down(index, slot)
        expert_dot(
            gate_up_accumulator.at[:, :expert_width],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator,
            expert_width,
            block_major=batch_size >= _ROUTED_EXPERTS_PER_TOKEN,
        )
        route_weights = (shared_probabilities_at(index) / probability_totals)[:, None]
        expert_output[:, :expert_output_size] += (
            down_accumulator[...].astype(jnp.bfloat16).astype(jnp.float32)
            * route_weights
        )

        @pl.when(index + 4 < local_route_counts[0])
        def refill_expert() -> None:
            """Compute and stage Pallas VMEM state for refill expert."""
            _start_async_copies(copy_expert(shared_expert_at(index + 4), slot))

    @pl.loop(0, local_route_counts[0] // 2)
    def run_shared_pair(pair: TensorLike) -> None:
        """Compute and stage Pallas run shared pair."""
        first = 2 * pair
        first_slot = first % 4
        second_slot = (first + 1) % 4
        wait_for_gate_up(first, first_slot)
        wait_for_gate_up(first + 1, second_slot)
        _paired_expert_gate_up(
            source,
            expert_gate_up_vmem,
            expert_gate_up_scale_vmem,
            gate_up_accumulator,
            first_slot,
            second_slot,
            expert_input_size,
        )
        finish_shared_expert(first, first_slot, 0)
        finish_shared_expert(first + 1, second_slot, 2 * expert_width)

    @pl.when(local_route_counts[0] % 2 == 1)
    def run_shared_tail() -> None:
        """Compute and stage Pallas run shared tail."""
        index = local_route_counts[0] - 1
        slot = index % 4
        wait_for_gate_up(index, slot)
        expert_dot(
            source,
            expert_gate_up_vmem.at[slot],
            expert_gate_up_scale_vmem.at[slot],
            gate_up_accumulator.at[:, : 2 * expert_width],
            expert_input_size,
        )
        finish_shared_expert(index, slot, 0)


def _run_routed_expert_stream_stage_r(stage_ctx: TensorLike) -> None:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (batch_size, compact_capacity, compact_only, compact_overflow),
        (compact_route_data, compact_row_plan, copy_expert, expert_dot),
        (expert_gate_up_scale_vmem, expert_gate_up_vmem, expert_input_size),
        (expert_width, finish_compact_expert, finish_expert),
        (finish_singleton_expert, first_compact_source_vmem),
        (first_single_source, gate_up_accumulator, group_pending_routes),
        (next_expert, pair_count, probability_totals, processed_routes_vmem),
        (progress, reuse_row_plan, routes_are_shared, run_expert_in_passes),
        (second_compact_source_vmem, second_single_source),
        (selected_expert_count, selected_experts, selected_probabilities),
        (share_compact_finish, share_full_finish, source, stage_compact_rows),
        stage_singleton_source,
    ) = stage_ctx

    @pl.when(~routes_are_shared)
    def run_divergent_routes() -> None:
        """Compute and stage Pallas run divergent routes."""
        return _run_routed_expert_stream_run_diverge_stage_h(
            (
                (batch_size, compact_capacity, compact_only, compact_overflow),
                (compact_route_data, compact_row_plan, copy_expert, expert_dot),
                (expert_gate_up_scale_vmem, expert_gate_up_vmem, expert_input_size),
                (expert_width, finish_compact_expert, finish_expert),
                (finish_singleton_expert, first_compact_source_vmem),
                (first_single_source, gate_up_accumulator, group_pending_routes),
                (next_expert, pair_count, probability_totals, processed_routes_vmem),
                (progress, reuse_row_plan, run_expert_in_passes),
                (second_compact_source_vmem, second_single_source),
                (selected_expert_count, selected_experts, selected_probabilities),
                (share_compact_finish, share_full_finish, source, stage_compact_rows),
                stage_singleton_source,
            )
        )


def _run_routed_expert_stream_run_diverge_stage_h(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas run divergent routes."""
    (
        (batch_size, compact_capacity, compact_only, compact_overflow),
        (compact_route_data, compact_row_plan, copy_expert, expert_dot),
        (expert_gate_up_scale_vmem, expert_gate_up_vmem, expert_input_size),
        (expert_width, finish_compact_expert, finish_expert),
        (finish_singleton_expert, first_compact_source_vmem),
        (first_single_source, gate_up_accumulator, group_pending_routes),
        (next_expert, pair_count, probability_totals, processed_routes_vmem),
        (progress, reuse_row_plan, run_expert_in_passes),
        (second_compact_source_vmem, second_single_source),
        (selected_expert_count, selected_experts, selected_probabilities),
        (share_compact_finish, share_full_finish, source, stage_compact_rows),
        stage_singleton_source,
    ) = stage_ctx
    initial_first = initial_second = _NONE_STAGE_VAL
    carry_routes = (batch_size in (2, 4) and (not group_pending_routes)) or (
        batch_size in (8, 16, 32, 64, 96, 128) and group_pending_routes
    )

    def pack(choice: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for pack."""
        return tuple(value.astype(jnp.int32) for value in choice)

    def unpack(choice: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for unpack."""
        return (choice[0], *(value != 0 for value in choice[1:]))

    if carry_routes:
        initial_processed = jnp.zeros(selected_experts.shape, jnp.bool_)
        initial_first = next_expert(initial_processed)
        initial_second = next_expert(initial_processed | initial_first[3])

    def run_next_expert_pair(
        iteration: TensorLike, carry: TensorLike = None
    ) -> TensorLike:
        """Compute and stage Pallas run next expert pair."""
        return _run_routed_expert_stream_run_diverge_stage_e(
            (
                (batch_size, carry_routes, compact_capacity, compact_only),
                (compact_overflow, compact_route_data, compact_row_plan, copy_expert),
                (expert_dot, expert_gate_up_scale_vmem, expert_gate_up_vmem),
                (expert_input_size, expert_width, finish_compact_expert),
                (finish_expert, finish_singleton_expert, first_compact_source_vmem),
                (first_single_source, gate_up_accumulator, next_expert, pack),
                (probability_totals, processed_routes_vmem, progress, reuse_row_plan),
                (run_expert_in_passes, second_compact_source_vmem),
                (second_single_source, selected_expert_count, selected_probabilities),
                (share_compact_finish, share_full_finish, source, stage_compact_rows),
                (stage_singleton_source, unpack),
            ),
            (iteration, carry),
        )

    if carry_routes:
        jax.lax.fori_loop(
            0,
            pair_count,
            run_next_expert_pair,
            (pack(initial_first), pack(initial_second)),
        )
    else:

        @pl.loop(0, pair_count)
        def run_original_expert_pair(iteration: TensorLike) -> None:
            """Compute and stage Pallas run original expert pair."""
            run_next_expert_pair(iteration)


def _run_routed_expert_stream_run_diverge_stage_e(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> TensorLike:
    """Compute and stage Pallas run next expert pair."""
    (
        (batch_size, carry_routes, compact_capacity, compact_only),
        (compact_overflow, compact_route_data, compact_row_plan, copy_expert),
        (expert_dot, expert_gate_up_scale_vmem, expert_gate_up_vmem),
        (expert_input_size, expert_width, finish_compact_expert),
        (finish_expert, finish_singleton_expert, first_compact_source_vmem),
        (first_single_source, gate_up_accumulator, next_expert, pack),
        (probability_totals, processed_routes_vmem, progress, reuse_row_plan),
        (run_expert_in_passes, second_compact_source_vmem),
        (second_single_source, selected_expert_count, selected_probabilities),
        (share_compact_finish, share_full_finish, source, stage_compact_rows),
        (stage_singleton_source, unpack),
    ) = stage_ctx
    iteration, carry = stage_args
    if progress is not None:
        progress(iteration)
    with _phase("e_iteration"):
        (
            (current_base_slot, execute_full_pair, execute_full_tail, first_consumed),
            (first_copies, first_expert, first_rows, first_slot_mask, fourth_consumed),
            (_fourth_copies, fourth_expert, fourth_rows, fourth_slot_mask, has_first),
            (has_fourth, has_second, has_third, second_consumed, second_copies),
            (
                second_expert,
                second_rows,
                second_slot_mask,
                third_consumed,
                _third_copies,
            ),
            (third_expert, third_rows, third_slot_mask),
        ) = _run_routed_expert_stream_run_diverge_stage_z(
            (
                (carry, carry_routes, copy_expert, expert_dot),
                (expert_gate_up_scale_vmem, expert_gate_up_vmem, expert_input_size),
                (expert_width, finish_expert, gate_up_accumulator, iteration),
                (next_expert, processed_routes_vmem, selected_expert_count),
                (share_full_finish, source, unpack),
            )
        )
        if batch_size >= _ROUTED_EXPERTS_PER_TOKEN:
            (_first_plan, _first_row_count, _second_plan, _second_row_count) = (
                _run_routed_expert_stream_run_diverge_stage_t(
                    (
                        (
                            compact_capacity,
                            compact_only,
                            compact_overflow,
                            compact_route_data,
                        ),
                        (
                            compact_row_plan,
                            copy_expert,
                            current_base_slot,
                            execute_full_pair,
                        ),
                        (execute_full_tail, expert_dot, expert_gate_up_scale_vmem),
                        (expert_gate_up_vmem, expert_input_size, expert_width),
                        (finish_compact_expert, finish_expert, finish_singleton_expert),
                        (
                            first_compact_source_vmem,
                            first_copies,
                            first_expert,
                            first_rows,
                        ),
                        (
                            first_single_source,
                            first_slot_mask,
                            gate_up_accumulator,
                            has_first,
                        ),
                        (
                            has_second,
                            probability_totals,
                            reuse_row_plan,
                            run_expert_in_passes,
                        ),
                        (second_compact_source_vmem, second_copies, second_expert),
                        (second_rows, second_single_source, second_slot_mask),
                        (
                            selected_probabilities,
                            share_compact_finish,
                            share_full_finish,
                        ),
                        (stage_compact_rows, stage_singleton_source),
                    )
                )
            )
        else:

            @pl.when(has_first & has_second)
            def execute_pair() -> None:
                """Compute and stage Pallas VMEM state for execute pair."""
                execute_full_pair()

            @pl.when(has_first & ~has_second)
            def execute_tail() -> None:
                """Compute and stage Pallas VMEM state for execute tail."""
                execute_full_tail()

        processed_routes_vmem[:, :selected_expert_count] = jnp.where(
            first_consumed | second_consumed,
            jnp.int32(1),
            processed_routes_vmem[:, :selected_expert_count],
        )
        if carry_routes:
            return (
                pack(
                    (
                        third_expert,
                        third_rows,
                        third_slot_mask,
                        third_consumed,
                        has_third,
                    )
                ),
                pack(
                    (
                        fourth_expert,
                        fourth_rows,
                        fourth_slot_mask,
                        fourth_consumed,
                        has_fourth,
                    )
                ),
            )
        return None


def _run_routed_expert_stream_run_diverge_stage_z(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (carry, carry_routes, copy_expert, expert_dot),
        (expert_gate_up_scale_vmem, expert_gate_up_vmem, expert_input_size),
        (expert_width, finish_expert, gate_up_accumulator, iteration),
        (next_expert, processed_routes_vmem, selected_expert_count),
        (share_full_finish, source, unpack),
    ) = stage_ctx
    select_scope = _phase_start("e_select")
    processed = processed_routes_vmem[:, :selected_expert_count] != 0
    first_expert, first_rows, first_slot_mask, first_consumed, has_first = (
        unpack(carry[0]) if carry_routes else next_expert(processed)
    )
    second_expert, second_rows, second_slot_mask, second_consumed, has_second = (
        unpack(carry[1]) if carry_routes else next_expert(processed | first_consumed)
    )
    processed_after_pair = processed | first_consumed | second_consumed
    third_expert, third_rows, third_slot_mask, third_consumed, has_third = next_expert(
        processed_after_pair
    )
    fourth_expert, fourth_rows, fourth_slot_mask, fourth_consumed, has_fourth = (
        next_expert(processed_after_pair | third_consumed)
    )
    current_base_slot = iteration % 2 * 2
    next_base_slot = (iteration + 1) % 2 * 2
    first_copies = copy_expert(first_expert, current_base_slot)
    second_copies = copy_expert(second_expert, current_base_slot + 1)
    third_copies = copy_expert(third_expert, next_base_slot)
    fourth_copies = copy_expert(fourth_expert, next_base_slot + 1)
    should_prefetch_third = has_third & (iteration != 0)
    should_prefetch_fourth = has_fourth & (iteration != 0)

    @pl.when(should_prefetch_third)
    def prefetch_third() -> None:
        """Compute and stage Pallas VMEM state for prefetch third."""
        _start_async_copies(third_copies)

    @pl.when(should_prefetch_fourth)
    def prefetch_fourth() -> None:
        """Compute and stage Pallas VMEM state for prefetch fourth."""
        _start_async_copies(fourth_copies)

    _phase_end(select_scope)

    def execute_full_pair() -> None:
        """Compute and stage Pallas execute full pair."""
        first_copies.gate_up.wait()
        first_copies.gate_up_scales.wait()
        second_copies.gate_up.wait()
        second_copies.gate_up_scales.wait()
        _paired_expert_gate_up(
            source,
            expert_gate_up_vmem,
            expert_gate_up_scale_vmem,
            gate_up_accumulator,
            current_base_slot,
            current_base_slot + 1,
            expert_input_size,
        )
        if not share_full_finish:
            finish_expert(
                first_rows, first_slot_mask, first_copies, 0, current_base_slot
            )
            finish_expert(
                second_rows,
                second_slot_mask,
                second_copies,
                2 * expert_width,
                current_base_slot + 1,
            )

    def execute_full_tail() -> None:
        """Compute and stage Pallas execute full tail."""
        first_copies.gate_up.wait()
        first_copies.gate_up_scales.wait()
        expert_dot(
            source,
            expert_gate_up_vmem.at[current_base_slot],
            expert_gate_up_scale_vmem.at[current_base_slot],
            gate_up_accumulator.at[:, : 2 * expert_width],
            expert_input_size,
        )
        if not share_full_finish:
            finish_expert(
                first_rows, first_slot_mask, first_copies, 0, current_base_slot
            )

    return (
        (current_base_slot, execute_full_pair, execute_full_tail, first_consumed),
        (first_copies, first_expert, first_rows, first_slot_mask, fourth_consumed),
        (fourth_copies, fourth_expert, fourth_rows, fourth_slot_mask, has_first),
        (has_fourth, has_second, has_third, second_consumed, second_copies),
        (second_expert, second_rows, second_slot_mask, third_consumed, third_copies),
        (third_expert, third_rows, third_slot_mask),
    )


def _run_routed_expert_stream_run_diverge_stage_t(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (compact_capacity, compact_only, compact_overflow, compact_route_data),
        (compact_row_plan, copy_expert, current_base_slot, execute_full_pair),
        (execute_full_tail, expert_dot, expert_gate_up_scale_vmem),
        (expert_gate_up_vmem, expert_input_size, expert_width),
        (finish_compact_expert, finish_expert, finish_singleton_expert),
        (first_compact_source_vmem, first_copies, first_expert, first_rows),
        (first_single_source, first_slot_mask, gate_up_accumulator, has_first),
        (has_second, probability_totals, reuse_row_plan, run_expert_in_passes),
        (second_compact_source_vmem, second_copies, second_expert),
        (second_rows, second_single_source, second_slot_mask),
        (selected_probabilities, share_compact_finish, share_full_finish),
        (stage_compact_rows, stage_singleton_source),
    ) = stage_ctx
    first_row_count = jnp.sum(first_rows.astype(jnp.int32))
    second_row_count = jnp.sum(second_rows.astype(jnp.int32))
    compact_pair = (
        has_first & has_second & (first_row_count == 1) & (second_row_count == 1)
    )
    first_plan = compact_row_plan(first_rows) if reuse_row_plan else None
    multi_pair, second_plan = _run_routed_expert_stream_run_diverge_stage_x(
        (
            (compact_capacity, compact_pair, compact_row_plan, current_base_slot),
            (expert_gate_up_scale_vmem, expert_gate_up_vmem, expert_input_size),
            (expert_width, finish_compact_expert, finish_singleton_expert),
            (first_compact_source_vmem, first_copies, first_plan, first_row_count),
            (first_rows, first_single_source, first_slot_mask),
            (gate_up_accumulator, has_first, has_second, reuse_row_plan),
            (second_compact_source_vmem, second_copies, second_row_count),
            (second_rows, second_single_source, second_slot_mask),
            (share_compact_finish, stage_compact_rows, stage_singleton_source),
        )
    )

    @pl.when(has_first & has_second & ~compact_pair & ~multi_pair)
    @_scoped("e_pass_pair")
    def execute_noncompact_pair() -> None:
        """Compute and stage Pallas execute noncompact pair."""
        if compact_only:
            run_expert_in_passes(
                first_rows,
                first_slot_mask,
                first_copies,
                first_row_count,
                current_base_slot,
                paired_gate=compact_overflow,
            )
            run_expert_in_passes(
                second_rows,
                second_slot_mask,
                second_copies,
                second_row_count,
                current_base_slot + 1,
                paired_gate=compact_overflow,
            )
        else:
            execute_full_pair()

    compact_tail = has_first & ~has_second & (first_row_count == 1)

    @pl.when(compact_tail)
    @_scoped("e_single_tail")
    def execute_compact_tail() -> None:
        """Compute and stage Pallas execute compact tail."""
        first_copies.gate_up.wait()
        first_copies.gate_up_scales.wait()
        stage_singleton_source(first_rows, first_single_source)
        expert_dot(
            first_single_source,
            expert_gate_up_vmem.at[current_base_slot],
            expert_gate_up_scale_vmem.at[current_base_slot],
            gate_up_accumulator.at[:1, : 2 * expert_width],
            expert_input_size,
        )
        finish_singleton_expert(
            first_rows, first_slot_mask, first_copies, 0, 0, current_base_slot
        )

    if compact_capacity > 1:
        multi_tail = (
            has_first
            & ~has_second
            & (first_row_count != 1)
            & (first_row_count <= compact_capacity)
        )

        @pl.when(multi_tail)
        def execute_compact_multi_tail() -> None:
            """Compute and stage Pallas execute compact multi tail."""
            first_copies.gate_up.wait()
            first_copies.gate_up_scales.wait()
            stage_compact_rows(
                first_rows, first_compact_source_vmem, row_plan=first_plan
            )
            expert_dot(
                first_compact_source_vmem,
                expert_gate_up_vmem.at[current_base_slot],
                expert_gate_up_scale_vmem.at[current_base_slot],
                gate_up_accumulator.at[:compact_capacity, : 2 * expert_width],
                expert_input_size,
            )
            if not share_compact_finish:
                finish_compact_expert(
                    first_rows, first_slot_mask, first_copies, 0, 0, current_base_slot
                )
    else:
        multi_tail = jnp.bool_(bool(0))

    @pl.when(has_first & ~has_second & (first_row_count != 1) & ~multi_tail)
    def execute_noncompact_tail() -> None:
        """Compute and stage Pallas execute noncompact tail."""
        if compact_only:
            run_expert_in_passes(
                first_rows,
                first_slot_mask,
                first_copies,
                first_row_count,
                current_base_slot,
            )
        else:
            execute_full_tail()

    _run_routed_expert_stream_run_diverge_stage_y(
        (
            (compact_capacity, compact_pair, compact_route_data, compact_tail),
            (copy_expert, current_base_slot, expert_width, finish_compact_expert),
            (finish_expert, first_expert, first_plan, first_rows, first_slot_mask),
            (has_first, has_second, multi_pair, multi_tail, probability_totals),
            (second_expert, second_plan, second_rows, second_slot_mask),
            (selected_probabilities, share_compact_finish, share_full_finish),
        )
    )
    return (first_plan, first_row_count, second_plan, second_row_count)


def _run_routed_expert_stream_run_diverge_stage_x(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (compact_capacity, compact_pair, compact_row_plan, current_base_slot),
        (expert_gate_up_scale_vmem, expert_gate_up_vmem, expert_input_size),
        (expert_width, finish_compact_expert, finish_singleton_expert),
        (first_compact_source_vmem, first_copies, first_plan, first_row_count),
        (first_rows, first_single_source, first_slot_mask),
        (gate_up_accumulator, has_first, has_second, reuse_row_plan),
        (second_compact_source_vmem, second_copies, second_row_count),
        (second_rows, second_single_source, second_slot_mask),
        (share_compact_finish, stage_compact_rows, stage_singleton_source),
    ) = stage_ctx
    second_plan = compact_row_plan(second_rows) if reuse_row_plan else None

    @pl.when(compact_pair)
    @_scoped("e_single_pair")
    def execute_compact_pair() -> None:
        """Compute and stage Pallas execute compact pair."""
        first_copies.gate_up.wait()
        first_copies.gate_up_scales.wait()
        second_copies.gate_up.wait()
        second_copies.gate_up_scales.wait()
        stage_singleton_source(first_rows, first_single_source)
        stage_singleton_source(second_rows, second_single_source)
        _paired_expert_gate_up(
            first_single_source,
            expert_gate_up_vmem,
            expert_gate_up_scale_vmem,
            gate_up_accumulator.at[:2],
            current_base_slot,
            current_base_slot + 1,
            expert_input_size,
            second_source=second_single_source,
        )
        finish_singleton_expert(
            first_rows, first_slot_mask, first_copies, 0, 0, current_base_slot
        )
        finish_singleton_expert(
            second_rows,
            second_slot_mask,
            second_copies,
            1,
            2 * expert_width,
            current_base_slot + 1,
        )

    if compact_capacity > 1:
        multi_pair = (
            has_first
            & has_second
            & ~compact_pair
            & (first_row_count <= compact_capacity)
            & (second_row_count <= compact_capacity)
        )

        @pl.when(multi_pair)
        @_scoped("e_multi_pair")
        def execute_compact_multi_pair() -> None:
            """Compute and stage Pallas execute compact multi pair."""
            first_copies.gate_up.wait()
            first_copies.gate_up_scales.wait()
            second_copies.gate_up.wait()
            second_copies.gate_up_scales.wait()
            stage_compact_rows(
                first_rows, first_compact_source_vmem, row_plan=first_plan
            )
            stage_compact_rows(
                second_rows, second_compact_source_vmem, row_plan=second_plan
            )

            def project_pair(capacity: TensorLike = None) -> None:
                """Compute and stage Pallas VMEM state for project pair."""
                _paired_expert_gate_up(
                    first_compact_source_vmem,
                    expert_gate_up_vmem,
                    expert_gate_up_scale_vmem,
                    gate_up_accumulator.at[: 2 * compact_capacity],
                    current_base_slot,
                    current_base_slot + 1,
                    expert_input_size,
                    second_source=second_compact_source_vmem,
                    active_capacity=capacity,
                )

            if share_compact_finish:
                largest = jnp.maximum(first_row_count, second_row_count)
                previous = 1
                for capacity in (2, 4, 8):

                    @pl.when((largest > previous) & (largest <= capacity))
                    @_scoped(f"e_capacity_{capacity}")
                    def execute_capacity(capacity: TensorLike = capacity) -> None:
                        """Compute and stage Pallas VMEM state for execute capacity."""
                        project_pair(capacity)

                    previous = capacity
            else:
                project_pair()
                finish_compact_expert(
                    first_rows, first_slot_mask, first_copies, 0, 0, current_base_slot
                )
                finish_compact_expert(
                    second_rows,
                    second_slot_mask,
                    second_copies,
                    compact_capacity,
                    2 * expert_width,
                    current_base_slot + 1,
                )
    else:
        multi_pair = jnp.bool_(bool(0))
    return (multi_pair, second_plan)


def _run_routed_expert_stream_run_diverge_stage_y(stage_ctx: TensorLike) -> None:
    """Compute Pallas VMEM tensors for run stage."""
    (
        (compact_capacity, compact_pair, compact_route_data, compact_tail),
        (copy_expert, current_base_slot, expert_width, finish_compact_expert),
        (finish_expert, first_expert, first_plan, first_rows, first_slot_mask),
        (has_first, has_second, multi_pair, multi_tail, probability_totals),
        (second_expert, second_plan, second_rows, second_slot_mask),
        (selected_probabilities, share_compact_finish, share_full_finish),
    ) = stage_ctx
    if share_full_finish:

        @pl.when(has_first & ~compact_pair & ~multi_pair & ~compact_tail & ~multi_tail)
        @_scoped("e_pass_finish")
        def execute_full_finish() -> None:
            """Compute and stage Pallas execute full finish."""

            def route_weights(rows: TensorLike, mask: TensorLike) -> TensorLike:
                """Compute and stage Pallas VMEM state for route weights."""
                probability = (
                    jnp.sum(
                        jnp.where(mask, selected_probabilities, jnp.float32(0)), axis=1
                    )
                    / probability_totals
                )
                return jnp.where(rows, probability, jnp.float32(0))

            first_weights = route_weights(first_rows, first_slot_mask)
            second_weights = route_weights(second_rows, second_slot_mask)

            @pl.loop(0, jnp.where(has_second, 2, 1))
            def selected_expert(index: TensorLike) -> None:
                """Compute and stage Pallas VMEM state for selected expert."""
                first = index == 0
                output_weights = jnp.where(first, first_weights, second_weights)
                expert = jnp.where(first, first_expert, second_expert)
                slot = current_base_slot + index
                copies = copy_expert(expert, slot, gate_up=False)
                finish_expert(
                    None, None, copies, index * 2 * expert_width, slot, output_weights
                )

    if share_compact_finish:

        @pl.when(multi_pair | multi_tail)
        @_scoped("e_multi_finish")
        def execute_compact_finish() -> None:
            """Compute and stage Pallas execute compact finish."""
            first_selector, first_weights = compact_route_data(
                first_rows, first_slot_mask, row_plan=first_plan
            )
            second_selector, second_weights = compact_route_data(
                second_rows, second_slot_mask, row_plan=second_plan
            )

            def finish_selected() -> None:
                """Compute and stage Pallas VMEM state for finish selected."""

                @pl.loop(0, jnp.where(has_second, 2, 1))
                def selected_expert(index: TensorLike) -> None:
                    """Compute and stage Pallas VMEM state for selected expert."""
                    first = index == 0
                    selector = jnp.where(first, first_selector, second_selector)
                    output_weights = jnp.where(first, first_weights, second_weights)
                    expert = jnp.where(first, first_expert, second_expert)
                    slot = current_base_slot + index
                    copies = copy_expert(expert, slot, gate_up=False)
                    finish_compact_expert(
                        None,
                        None,
                        copies,
                        index * compact_capacity,
                        index * 2 * expert_width,
                        slot,
                        route_data=(selector, output_weights),
                    )

            finish_selected()


def _staging_slot_rows(payload_rows: int) -> int:
    """A staged per-pair payload's slot: whole 16-row BF16 tiles."""
    return -(-payload_rows // 16) * 16


def projection_staging_scratch(
    batch_size: int, attention_dp: int = 1
) -> tuple[TensorLike, ...]:
    """Compute Pallas VMEM tensors for projection stage.

    Above B32: send/receive staging of the attention projection's four
    per-pair payloads ([2, 4 * slot, 128] BF16).
    """
    if batch_size <= _KDA_GROUP_WIDTH:
        return ()
    payload_rows = batch_size // attention_dp // 4 * 3584 // 128
    return (tpu.VMEM((2, 4 * _staging_slot_rows(payload_rows), 128), jnp.bfloat16),)


def routed_send_scratch(batch_size: int) -> tuple[TensorLike, ...]:
    """Compute and stage Pallas VMEM state for routed send scratch.

    Above B32: the routed reduce-scatter's send side (eight lanes'
    tokens).
    """
    if batch_size <= _KDA_GROUP_WIDTH:
        return ()
    return (tpu.VMEM((8 * 14 * -(-batch_size // 8), 128), jnp.float32),)


def communication_scratch(
    batch_size: int = 1, attention_dp: int = 1
) -> tuple[TensorLike, ...]:
    """Compute and stage Pallas VMEM state for communication scratch.

    Shared for the entire fused stack, including the reduction
    sequence.
    """
    wide_reduction_rows = max(
        64, 28 * (min(batch_size, 8) if batch_size <= _EXPERT_GROUP_STRIDE else 16)
    )
    host_reduction_rows = max(
        wide_reduction_rows,
        7 * batch_size // attention_dp if batch_size > _KDA_GROUP_WIDTH else 0,
    )
    wide_reduction_rows = max(wide_reduction_rows, 4 * batch_size)
    host_reduction_rows = max(host_reduction_rows, 4 * batch_size)
    if attention_dp > 1:
        wide_reduction_rows = max(wide_reduction_rows, 7 * batch_size)
    padded_tokens = (
        8 * -(-batch_size // 8) if batch_size in (6, *SCATTER_BATCHES) else batch_size
    )
    expert_reduction_rows = max(32, 14 * padded_tokens)
    expert_host_rows = expert_reduction_rows
    if batch_size > _KDA_GROUP_WIDTH:
        expert_reduction_rows = 14 * -(-batch_size // 8)
        expert_host_rows = max(expert_reduction_rows, 8 * batch_size)
    (
        wide_reduction_rows,
        expert_reduction_rows,
        host_reduction_rows,
        expert_host_rows,
    ) = (
        -(-rows // 8) * 8
        for rows in (
            wide_reduction_rows,
            expert_reduction_rows,
            host_reduction_rows,
            expert_host_rows,
        )
    )
    return (
        tpu.VMEM((8, wide_reduction_rows, 128), jnp.float32),
        tpu.VMEM((8, expert_reduction_rows, 128), jnp.float32),
        tpu.VMEM((4, host_reduction_rows, 128), jnp.float32),
        tpu.VMEM((4, expert_host_rows, 128), jnp.float32),
        tpu.VMEM(
            (max(16, -(-batch_size // 16) * 16), 1024 * attention_dp), jnp.bfloat16
        ),
        tpu.VMEM(
            (
                max(
                    16,
                    -(
                        -(
                            batch_size // 4
                            if batch_size >= _GATED_GROUP_WIDTH
                            else batch_size
                        )
                        // 16
                    )
                    * 16,
                ),
                7168,
            ),
            jnp.bfloat16,
        ),
        tpu.SemaphoreType.DMA((16,)),
        tpu.SemaphoreType.DMA((16,)),
        tpu.SMEM((1,), jnp.int32),
        tpu.VMEM((batch_size, 4096), jnp.bfloat16),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionCommunicationWorkspace:
    """Collective transport storage shared by attention and MoE layers."""

    shared_local_reduction_vmem: TensorLike
    expert_local_reduction_vmem: TensorLike
    shared_host_reduction_vmem: TensorLike
    expert_host_reduction_vmem: TensorLike
    attention_features_vmem: TensorLike
    attention_output_vmem: TensorLike
    send_semaphores: TensorLike
    receive_semaphores: TensorLike
    reduction_phase: TensorLike
    latent_gather_vmem: TensorLike
    projection_staging_vmem: TensorLike = None
    routed_send_vmem: TensorLike = None
    attention_gather_vmem: TensorLike = None
    routed_consumed: TensorLike = None
    pair_semaphores: TensorLike = None

    def __post_init__(self) -> None:
        """Register PyTree dataclasses lazily on first instantiation."""
        _ensure_dataclasses_registered()


class _ScopedWait:
    """An async copy whose wait is traced as the e_wait phase."""

    def __init__(self, copy: TensorLike) -> None:
        """Initialize instance state."""
        self.copy = copy

    def start(self) -> None:
        """Compute and stage Pallas VMEM state for start."""
        self.copy.start()

    def wait(self) -> None:
        """Compute and stage Pallas VMEM state for wait."""
        with _phase("e_wait"):
            self.copy.wait()


def _start_async_copies(copies: SimpleNamespace) -> None:
    """Compute and stage Pallas start async copies."""
    for copy in vars(copies).values():
        if copy is not None:
            copy.start()


def _gather_output_shards(
    output_vmem: TensorLike,
    send_semaphores: TensorLike,
    receive_semaphores: TensorLike,
    semaphore_slot: TensorLike,
) -> None:
    """Compute and stage Pallas gather output shards."""
    rank = jax.lax.axis_index("tp")
    local_shard = output_vmem.at[:, pl.ds(rank % 8 * 1024, 1024)]
    for offset in (7, 6, 5, 3, 4, 2, 1):
        tpu.make_async_remote_copy(
            local_shard,
            local_shard,
            send_semaphores.at[semaphore_slot],
            receive_semaphores.at[semaphore_slot],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    received_shards = output_vmem.at[:, : 7 * 1024]
    tpu.make_async_copy(
        received_shards, received_shards, send_semaphores.at[semaphore_slot]
    ).wait()
    tpu.make_async_copy(
        received_shards, received_shards, receive_semaphores.at[semaphore_slot]
    ).wait()


@dataclasses.dataclass(frozen=True, slots=True)
class MixtureOfExpertsWeights:
    """Compute and stage Pallas VMEM state for MixtureOfExpertsWeights.

    MoE parameters and persistent resources for asynchronous weight
    copies.
    """

    routing_projection: TensorLike
    routing_projection_vmem: TensorLike
    routing_copy_semaphores: TensorLike
    routing_bias: TensorLike
    latent_input_projection: TensorLike
    latent_output_projection: TensorLike
    latent_output_projection_vmem: TensorLike
    latent_output_copy_semaphore: TensorLike
    latent_normalization: TensorLike
    shared_gate_up_projection: TensorLike
    shared_down_projection: TensorLike
    expert_gate_up_projection: TensorLike
    expert_gate_up_projection_vmem: TensorLike
    expert_gate_up_scales: TensorLike
    expert_gate_up_scales_vmem: TensorLike
    expert_down_projection: TensorLike
    expert_down_projection_vmem: TensorLike
    expert_down_scales: TensorLike
    expert_down_scales_vmem: TensorLike
    expert_copy_semaphores: TensorLike

    def __post_init__(self) -> None:
        """Register PyTree dataclasses lazily on first instantiation."""
        _ensure_dataclasses_registered()

    def _copy_routing_projection(
        self, *, wait: bool, initialize_padding: bool = True
    ) -> None:
        """Compute and stage Pallas copy routing projection."""
        lane = jax.lax.axis_index("tp") % 2

        @pl.when(lane == 0)
        def copy_full_half() -> None:
            """Compute and stage Pallas copy full half."""
            copy = tpu.make_async_copy(
                self.routing_projection.at[:, :512],
                self.routing_projection_vmem.at[0],
                self.routing_copy_semaphores.at[0],
            )
            copy.wait() if wait else copy.start()

        @pl.when(lane == 1)
        def copy_tail() -> None:
            """Compute and stage Pallas VMEM state for copy tail."""
            if not wait and initialize_padding:
                self.routing_projection_vmem[0, :, 384:] = jnp.zeros(
                    (self.routing_projection.shape[0], 128), jnp.bfloat16
                )
            copy = tpu.make_async_copy(
                self.routing_projection.at[:, pl.ds(512, 384)],
                self.routing_projection_vmem.at[0, :, :384],
                self.routing_copy_semaphores.at[0],
            )
            copy.wait() if wait else copy.start()

    def start_routing_projection_copy(self, *, initialize_padding: bool = True) -> None:
        """Compute and stage Pallas start routing projection copy."""
        self._copy_routing_projection(wait=False, initialize_padding=initialize_padding)

    def wait_for_routing_projection_copy(self) -> None:
        """Compute and stage Pallas wait for routing projection copy."""
        self._copy_routing_projection(wait=True)

    def make_latent_output_copy(self) -> TensorLike:
        """Compute and stage Pallas make latent output copy."""
        return tpu.make_async_copy(
            self.latent_output_projection,
            self.latent_output_projection_vmem,
            self.latent_output_copy_semaphore,
        )

    def make_expert_copies(
        self,
        expert: TensorLike,
        slot: TensorLike,
        *,
        gate_up: bool = True,
        down: bool = True,
    ) -> SimpleNamespace:
        """Compute and stage Pallas make expert copies."""
        return SimpleNamespace(
            gate_up=tpu.make_async_copy(
                self.expert_gate_up_projection.at[expert],
                self.expert_gate_up_projection_vmem.at[slot],
                self.expert_copy_semaphores.at[slot, 0],
            )
            if gate_up
            else None,
            gate_up_scales=tpu.make_async_copy(
                self.expert_gate_up_scales.at[expert],
                self.expert_gate_up_scales_vmem.at[slot],
                self.expert_copy_semaphores.at[slot, 1],
            )
            if gate_up
            else None,
            down=tpu.make_async_copy(
                self.expert_down_projection.at[expert],
                self.expert_down_projection_vmem.at[slot],
                self.expert_copy_semaphores.at[slot, 2],
            )
            if down
            else None,
            down_scales=tpu.make_async_copy(
                self.expert_down_scales.at[expert],
                self.expert_down_scales_vmem.at[slot],
                self.expert_copy_semaphores.at[slot, 3],
            )
            if down
            else None,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class MixtureOfExpertsWorkspace:
    """Persistent VMEM and semaphores used by one MoE invocation."""

    selected_experts_vmem: TensorLike
    selected_probabilities_vmem: TensorLike
    local_route_counts_vmem: TensorLike
    probability_totals_vmem: TensorLike
    routing_scores_vmem: TensorLike
    shared_outputs_vmem: TensorLike
    routed_outputs_vmem: TensorLike
    gate_up_accumulator_vmem: TensorLike
    down_accumulator_vmem: TensorLike
    processed_routes_vmem: TensorLike
    first_compact_source_vmem: TensorLike
    second_compact_source_vmem: TensorLike
    shared_send_semaphores: TensorLike
    shared_receive_semaphores: TensorLike
    output_vmem: TensorLike
    output_send_semaphores: TensorLike
    output_receive_semaphores: TensorLike
    latent_send_semaphores: TensorLike
    latent_receive_semaphores: TensorLike
    selection_source_vmem: TensorLike = None
    selections_vmem: TensorLike = None

    def __post_init__(self) -> None:
        """Register PyTree dataclasses lazily on first instantiation."""
        _ensure_dataclasses_registered()


def moe(
    hidden_ref: TensorLike, output_ref: TensorLike, **stage_kwargs: TensorLike
) -> None:
    """Execute one mixture-of-experts layer inside a Pallas program.

    ``token_scatter_output`` leaves each of the
    ``token_scatter_owners`` local row owners (4: rank pairs; 8:
    ranks) with its batch // owners rows. ``owned_hidden_ref`` (this
    rank pair's rows of ``hidden_ref``) shards the router's top-k over
    the rank pairs (_route_owned_rows).

    ``live_rows`` masks null serving slots before route
    prefetch/compaction; the router scores and shared expert still use
    the compiled batch shape.

    ``prepare_weights_vmem`` optionally supplies caller-owned VMEM for
    the latent-down, shared gate/up and shared down projections
    instead of a scoped allocation.
    """
    (
        weights,
        workspace,
        communication_workspace,
        selected_expert_count,
        epsilon,
        latent_feature_count,
        synchronize_devices,
        routing_projection_is_prefetched,
        group_pending_expert_routes,
        token_scatter_output,
        routed_reduce_scatter,
        after_experts,
        after_routing_scores,
        overlap_router_latent,
        overlap_shared_reduction,
        prepare_weights_vmem,
        shared_gate_up_stream,
        token_scatter_owners,
        owned_hidden_ref,
        live_rows,
    ) = (
        stage_kwargs["weights"],
        stage_kwargs["workspace"],
        stage_kwargs["communication_workspace"],
        stage_kwargs.get("selected_expert_count", 16),
        stage_kwargs.get("epsilon", 1e-05),
        stage_kwargs.get("latent_feature_count", 3584),
        stage_kwargs.get("synchronize_devices", True),
        stage_kwargs.get("routing_projection_is_prefetched", False),
        stage_kwargs.get("group_pending_expert_routes", False),
        stage_kwargs.get("token_scatter_output", False),
        stage_kwargs.get("routed_reduce_scatter", False),
        stage_kwargs.get("after_experts", _NONE_STAGE_VAL),
        stage_kwargs.get("after_routing_scores", _NONE_STAGE_VAL),
        stage_kwargs.get("overlap_router_latent", False),
        stage_kwargs.get("overlap_shared_reduction", False),
        stage_kwargs.get("prepare_weights_vmem", _NONE_STAGE_VAL),
        stage_kwargs.get("shared_gate_up_stream", _NONE_STAGE_VAL),
        stage_kwargs.get("token_scatter_owners", 4),
        stage_kwargs.get("owned_hidden_ref", _NONE_STAGE_VAL),
        stage_kwargs.get("live_rows", _NONE_STAGE_VAL),
    )
    lane = _NONE_STAGE_VAL
    batch_size, hidden_size = hidden_ref.shape
    (
        (expert_input_size, expert_output_size, expert_prefetch_slot_count),
        (expert_tensor_parallel, expert_width, latent_size, local_expert_count),
        shared_width,
    ) = _moe_stage_al(
        (
            (batch_size, hidden_size, routed_reduce_scatter, token_scatter_output),
            (token_scatter_owners, weights),
        )
    )
    (
        (down_accumulator_vmem, expert_down_scale_vmem, expert_down_vmem),
        (expert_gate_up_scale_vmem, expert_gate_up_vmem, first_compact_source_vmem),
        (gate_up_accumulator_vmem, local_route_counts_vmem, output_vmem),
        (probability_totals_vmem, processed_routes_vmem, routed_outputs_vmem),
        (second_compact_source_vmem, selected_experts_vmem),
        (selected_probabilities_vmem, shared_down_weights, shared_outputs_vmem),
        (shared_receive_semaphores, shared_send_semaphores, shared_sharded),
    ) = _moe_stage_af(
        (
            (batch_size, communication_workspace, expert_input_size),
            (expert_output_size, expert_prefetch_slot_count, expert_width),
            (group_pending_expert_routes, hidden_ref, hidden_size, latent_size),
            (output_ref, prepare_weights_vmem, token_scatter_output),
            (token_scatter_owners, weights, workspace),
        )
    )
    (
        (copy_expert, expert_host_reduction_vmem, expert_local_reduction_vmem),
        (hidden, latent_copy, latent_output_copy, _latent_receive_semaphores),
        (_latent_send_semaphores, _latent_slot, latent_vmem),
        (output_receive_semaphores, output_send_semaphores, overlap_latent_gather),
        (overlap_router_latent, overlap_shared_routing, rank, receive_semaphores),
        (reduction_phase, routed_consumed, routed_send_vmem, send_semaphores),
        (shared_host_reduction_vmem, shared_local_reduction_vmem),
        wait_for_latent_gather,
    ) = _moe_stage_ag(
        (
            (communication_workspace, hidden_ref, overlap_router_latent),
            (owned_hidden_ref, routing_projection_is_prefetched),
            (selected_expert_count, shared_sharded, synchronize_devices, weights),
            workspace,
        )
    )
    (
        (decode_expert, decode_gathered_experts, decode_shared, decode_shared_shard),
        (encode_expert, encode_shared, encode_shared_shard, expert_payload),
        (_expert_rows, lane_payload, reduce_scatter_routed_output),
        (reduce_scatter_shared_output, shared_copy, shared_payload, _shared_rows),
        (shared_shard_payload, _shared_shard_rows, shared_shard_width),
        (_start_latent_gather, tokens_per_lane),
    ) = _moe_stage_am(
        (
            (after_routing_scores, batch_size, communication_workspace),
            (copy_expert, expert_output_size, expert_prefetch_slot_count),
            (expert_tensor_parallel, group_pending_expert_routes, hidden),
            (hidden_size, latent_copy, latent_vmem, live_rows, local_expert_count),
            (local_route_counts_vmem, overlap_latent_gather),
            (overlap_router_latent, overlap_shared_routing, owned_hidden_ref),
            (prepare_weights_vmem, probability_totals_vmem, rank),
            (receive_semaphores, routed_outputs_vmem, routed_reduce_scatter),
            (selected_expert_count, selected_experts_vmem),
            (selected_probabilities_vmem, send_semaphores, shared_gate_up_stream),
            (shared_host_reduction_vmem, shared_local_reduction_vmem),
            (shared_outputs_vmem, shared_receive_semaphores),
            (shared_send_semaphores, shared_sharded, shared_width),
            (wait_for_latent_gather, weights, workspace),
        )
    )
    (
        (begin_shared_reduction, expert_copy, expert_shard_copy),
        (expert_shard_receive_view, expert_shard_send_view, finish_shared_reduction),
        (routed_send_storage, _shared_shard_copy, _shared_shard_receive_view),
        _shared_shard_send_view,
    ) = _moe_stage_ak(
        (
            (batch_size, decode_shared, decode_shared_shard, encode_shared),
            (encode_shared_shard, expert_host_reduction_vmem),
            (expert_local_reduction_vmem, lane_payload, rank, receive_semaphores),
            (reduce_scatter_shared_output, routed_send_vmem, send_semaphores),
            (shared_copy, shared_down_weights, shared_host_reduction_vmem),
            (shared_local_reduction_vmem, shared_payload),
            (shared_receive_semaphores, shared_send_semaphores),
            (shared_shard_payload, shared_shard_width, shared_sharded),
        )
    )

    def finish_shared_host_reduction(
        payload: TensorLike, reduced_shared_outputs: TensorLike
    ) -> None:
        """Compute and stage Pallas finish shared host reduction."""
        for offset in range(3, 0, -1):
            shared_copy(payload, offset, across_hosts=True).wait()
        if reduce_scatter_shared_output:
            lane = rank % 8

            @pl.when(lane < _LAST_EXPERT_SLOT)
            def store_shared_shard() -> None:
                """Compute and stage Pallas store shared shard."""
                (
                    reduced_shared_outputs
                    if shared_sharded
                    else reduced_shared_outputs.at[
                        :, pl.ds(lane * shared_shard_width, shared_shard_width)
                    ]
                )[...] = decode_shared_shard(shared_host_reduction_vmem[:, :payload, :])
        else:
            reduced_shared_outputs[...] = decode_shared(
                shared_host_reduction_vmem[:, :payload, :]
            )

    lane = _moe_stage_an(
        (
            (after_experts, batch_size, begin_shared_reduction, copy_expert),
            (decode_expert, decode_gathered_experts, down_accumulator_vmem),
            (encode_expert, epsilon, expert_copy, expert_down_scale_vmem),
            (expert_down_vmem, expert_gate_up_scale_vmem, expert_gate_up_vmem),
            (expert_host_reduction_vmem, expert_input_size),
            (expert_local_reduction_vmem, expert_output_size, expert_payload),
            (expert_shard_copy, expert_shard_receive_view, expert_shard_send_view),
            (expert_width, finish_shared_host_reduction, finish_shared_reduction),
            (first_compact_source_vmem, gate_up_accumulator_vmem),
            (group_pending_expert_routes, hidden_ref, hidden_size, lane),
            (lane_payload, latent_feature_count, latent_output_copy, latent_vmem),
            (local_route_counts_vmem, output_receive_semaphores, output_ref),
            (output_send_semaphores, output_vmem, overlap_latent_gather),
            (overlap_shared_reduction, probability_totals_vmem),
            (processed_routes_vmem, rank, reduce_scatter_routed_output),
            (reduce_scatter_shared_output, reduction_phase, routed_consumed),
            (routed_outputs_vmem, routed_send_storage, routed_send_vmem),
            (second_compact_source_vmem, selected_expert_count),
            (selected_experts_vmem, selected_probabilities_vmem),
            (shared_local_reduction_vmem, shared_outputs_vmem, shared_payload),
            (shared_receive_semaphores, shared_send_semaphores),
            (shared_shard_payload, shared_sharded, shared_width),
            (token_scatter_output, token_scatter_owners, tokens_per_lane),
            (wait_for_latent_gather, weights),
        )
    )


def _moe_stage_al(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe stage al."""
    (
        (batch_size, hidden_size, routed_reduce_scatter, token_scatter_output),
        (token_scatter_owners, weights),
    ) = stage_ctx
    expert_count = weights.routing_projection.shape[1]
    latent_size = weights.latent_normalization.shape[-1]
    expert_output_size = weights.expert_down_projection.shape[-1]
    local_expert_count = weights.expert_gate_up_projection.shape[0]
    expert_tensor_parallel = 32 * local_expert_count // expert_count
    expert_width = weights.expert_down_projection.shape[1] * 8
    expert_input_size = weights.expert_gate_up_projection.shape[1] * 8
    shared_width = weights.shared_down_projection.shape[0]
    expert_prefetch_slot_count = 4
    np.testing.assert_(
        bool(1 <= batch_size <= MAX_BATCH), f"MoE supports 1 to {MAX_BATCH} rows"
    )
    np.testing.assert_(
        not (
            token_scatter_output
            and batch_size not in SCATTER_BATCHES
            and (not (batch_size in (2, 4) and token_scatter_owners == 1))
        ),
        str.join(
            "",
            (
                "MoE token-scattered output requires batch size ",
                "8 or 16, or B2/B4 to one owner",
            ),
        ),
    )
    _moe_stage_aj(
        (
            (batch_size, expert_count, expert_output_size, expert_tensor_parallel),
            (expert_width, hidden_size, latent_size, routed_reduce_scatter),
            weights,
        )
    )
    np.testing.assert_(
        not (
            expert_count * expert_tensor_parallel != 32 * local_expert_count
            or expert_output_size > latent_size
            or expert_input_size > latent_size
        ),
        "Expert ownership or latent padding is inconsistent",
    )
    np.testing.assert_(
        not (expert_input_size % 512 or expert_output_size % 128),
        "Expert dimensions must satisfy MXFP4 alignment",
    )
    return (
        (expert_input_size, expert_output_size, expert_prefetch_slot_count),
        (expert_tensor_parallel, expert_width, latent_size, local_expert_count),
        shared_width,
    )


def _moe_stage_aj(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for moe stage aj."""
    (
        (batch_size, expert_count, expert_output_size, expert_tensor_parallel),
        (expert_width, hidden_size, latent_size, routed_reduce_scatter),
        weights,
    ) = stage_ctx
    np.testing.assert_(
        not (routed_reduce_scatter and batch_size not in (6, *SCATTER_BATCHES)),
        "MoE routed reduce-scatter requires batch size 6, 8 or 16",
    )
    np.testing.assert_(
        not (
            weights.expert_gate_up_projection.dtype != jnp.uint32
            or weights.expert_down_projection.dtype != jnp.uint32
        ),
        "Routed-expert weights must be packed MXFP4",
    )
    np.testing.assert_(
        not expert_tensor_parallel not in (1, 2, 4),
        "Expert tensor parallelism must be 1, 2, or 4",
    )
    np.testing.assert_(
        not (
            hidden_size % 128
            or expert_count % 128
            or latent_size % 256
            or expert_width % 256
            or (hidden_size > _ROPE_BASE_DIM)
            or (latent_size > _ROPE_BASE_DIM)
        ),
        "Unsupported MoE tile dimensions",
    )
    np.testing.assert_(
        not (
            weights.latent_input_projection.shape != (hidden_size, 128)
            or weights.latent_output_projection.shape != (expert_output_size, 1024)
        ),
        "Latent projections require padded TP32 shards",
    )


def _moe_stage_af(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe stage af."""
    (
        (batch_size, communication_workspace, expert_input_size),
        (expert_output_size, expert_prefetch_slot_count, expert_width),
        (group_pending_expert_routes, hidden_ref, hidden_size, latent_size),
        (output_ref, prepare_weights_vmem, token_scatter_output),
        (token_scatter_owners, weights, workspace),
    ) = stage_ctx
    np.testing.assert_(
        not (
            weights.expert_gate_up_projection.shape[1:]
            != (expert_input_size // 8, 2 * expert_width)
            or weights.expert_gate_up_scales.shape[1:]
            != (expert_input_size // 32, 2 * expert_width)
            or weights.expert_gate_up_scales.dtype != jnp.uint8
        ),
        "MXFP4 gate/up weights have inconsistent shapes",
    )
    np.testing.assert_(
        not (
            weights.expert_down_projection.shape[1] * 8 != expert_width
            or weights.expert_down_scales.dtype != jnp.uint8
        ),
        "MXFP4 down weights or scales have inconsistent shapes",
    )
    expected_output_shape = (
        (batch_size // token_scatter_owners, hidden_size)
        if token_scatter_output
        else hidden_ref.shape
    )
    np.testing.assert_(
        output_ref.shape == expected_output_shape,
        "MoE output has the wrong shape for its communication mode",
    )
    accumulator_rows = _expert_accumulator_rows(
        batch_size, group_pending_routes=group_pending_expert_routes
    )
    expected_workspace_shapes = (
        (workspace.selected_experts_vmem.shape, (batch_size, 128)),
        (workspace.selected_probabilities_vmem.shape, (batch_size, 128)),
        (workspace.local_route_counts_vmem.shape, (batch_size, 128)),
        (workspace.probability_totals_vmem.shape, (batch_size, 128)),
        (workspace.routing_scores_vmem.shape, (batch_size, 1024)),
        (
            workspace.shared_outputs_vmem.shape,
            (batch_size, 1024 if _shared_shards(batch_size) else hidden_size),
        ),
        (workspace.routed_outputs_vmem.shape, (batch_size, latent_size)),
        (
            workspace.gate_up_accumulator_vmem.shape,
            (accumulator_rows[0], 4 * expert_width),
        ),
        (
            workspace.down_accumulator_vmem.shape,
            (accumulator_rows[1], expert_output_size),
        ),
        (workspace.processed_routes_vmem.shape, (batch_size, 128)),
        (
            workspace.first_compact_source_vmem.shape,
            (max(1, _compact_rows(batch_size)), expert_input_size),
        ),
        (
            workspace.second_compact_source_vmem.shape,
            (max(1, _compact_rows(batch_size)), expert_input_size),
        ),
    )
    np.testing.assert_(
        not any((actual != expected for actual, expected in expected_workspace_shapes)),
        "MoE workspace has an invalid shape",
    )
    expected_output_vmem_shape = (
        (
            token_scatter_owners + 8,
            batch_size // token_scatter_owners * 1024 // 128,
            128,
        )
        if token_scatter_output
        else (batch_size, 8192)
    )
    np.testing.assert_(
        workspace.output_vmem.shape == expected_output_vmem_shape,
        "MoE output communication workspace has the wrong shape",
    )
    np.testing.assert_(
        communication_workspace.latent_gather_vmem.shape == (batch_size, 4096),
        "MoE latent-gather workspace must match the batch size",
    )
    np.testing.assert_(
        weights.routing_projection_vmem.shape == (1, hidden_size, 512),
        "MoE routing projection workspace has the wrong shape",
    )
    expert_workspace_shapes = (
        (
            weights.expert_gate_up_projection_vmem.shape,
            (expert_prefetch_slot_count, *weights.expert_gate_up_projection.shape[1:]),
        ),
        (
            weights.expert_gate_up_scales_vmem.shape,
            (expert_prefetch_slot_count, *weights.expert_gate_up_scales.shape[1:]),
        ),
        (
            weights.expert_down_projection_vmem.shape,
            (expert_prefetch_slot_count, *weights.expert_down_projection.shape[1:]),
        ),
        (
            weights.expert_down_scales_vmem.shape,
            (expert_prefetch_slot_count, *weights.expert_down_scales.shape[1:]),
        ),
    )
    np.testing.assert_(
        not any((actual != expected for actual, expected in expert_workspace_shapes)),
        "MoE expert weight workspace has an invalid shape",
    )
    expert_gate_up_vmem = weights.expert_gate_up_projection_vmem
    expert_gate_up_scale_vmem = weights.expert_gate_up_scales_vmem
    expert_down_vmem = weights.expert_down_projection_vmem
    expert_down_scale_vmem = weights.expert_down_scales_vmem
    selected_experts_vmem = workspace.selected_experts_vmem
    selected_probabilities_vmem = workspace.selected_probabilities_vmem
    local_route_counts_vmem = workspace.local_route_counts_vmem
    probability_totals_vmem = workspace.probability_totals_vmem
    shared_outputs_vmem = workspace.shared_outputs_vmem
    routed_outputs_vmem = workspace.routed_outputs_vmem
    shared_sharded = shared_outputs_vmem.shape[1] < hidden_size
    shared_down_weights = (
        prepare_weights_vmem[2] if prepare_weights_vmem is not None else None
    )
    np.testing.assert_(
        not shared_sharded or shared_down_weights is not None,
        "shared shards need pooled weights",
    )
    gate_up_accumulator_vmem = workspace.gate_up_accumulator_vmem
    down_accumulator_vmem = workspace.down_accumulator_vmem
    processed_routes_vmem = workspace.processed_routes_vmem
    first_compact_source_vmem = workspace.first_compact_source_vmem
    second_compact_source_vmem = workspace.second_compact_source_vmem
    shared_send_semaphores = workspace.shared_send_semaphores
    shared_receive_semaphores = workspace.shared_receive_semaphores
    output_vmem = workspace.output_vmem
    return (
        (down_accumulator_vmem, expert_down_scale_vmem, expert_down_vmem),
        (expert_gate_up_scale_vmem, expert_gate_up_vmem, first_compact_source_vmem),
        (gate_up_accumulator_vmem, local_route_counts_vmem, output_vmem),
        (probability_totals_vmem, processed_routes_vmem, routed_outputs_vmem),
        (second_compact_source_vmem, selected_experts_vmem),
        (selected_probabilities_vmem, shared_down_weights, shared_outputs_vmem),
        (shared_receive_semaphores, shared_send_semaphores, shared_sharded),
    )


def _moe_stage_ag(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe stage ag."""
    (
        (communication_workspace, hidden_ref, overlap_router_latent),
        (owned_hidden_ref, routing_projection_is_prefetched),
        (selected_expert_count, shared_sharded, synchronize_devices, weights),
        workspace,
    ) = stage_ctx
    output_send_semaphores = workspace.output_send_semaphores
    output_receive_semaphores = workspace.output_receive_semaphores
    latent_send_semaphores = workspace.latent_send_semaphores
    latent_receive_semaphores = workspace.latent_receive_semaphores
    shared_local_reduction_vmem = communication_workspace.shared_local_reduction_vmem
    expert_local_reduction_vmem = communication_workspace.expert_local_reduction_vmem
    shared_host_reduction_vmem = communication_workspace.shared_host_reduction_vmem
    expert_host_reduction_vmem = communication_workspace.expert_host_reduction_vmem
    routed_send_vmem = communication_workspace.routed_send_vmem
    routed_consumed = communication_workspace.routed_consumed
    send_semaphores = communication_workspace.send_semaphores
    receive_semaphores = communication_workspace.receive_semaphores
    reduction_phase = communication_workspace.reduction_phase
    latent_vmem = communication_workspace.latent_gather_vmem
    rank = jax.lax.axis_index("tp")
    if synchronize_devices:
        collectives32.barrier()
        reduction_phase[0] = 0
    hidden = hidden_ref[...]

    def copy_expert(
        expert: TensorLike,
        slot: TensorLike,
        *,
        gate_up: TensorLike = True,
        down: TensorLike = True,
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for copy expert."""
        return weights.make_expert_copies(expert, slot, gate_up=gate_up, down=down)

    if not routing_projection_is_prefetched:
        weights.start_routing_projection_copy()
    latent_output_copy = weights.make_latent_output_copy()
    overlap_latent_gather = selected_expert_count >= _EXPERT_COL_PACKING
    overlap_router_latent = (
        overlap_router_latent and overlap_latent_gather and (owned_hidden_ref is None)
    )
    overlap_shared_routing = overlap_router_latent and shared_sharded
    latent_slot = reduction_phase[0] % 2

    def latent_copy(offset: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for latent copy."""
        view = latent_vmem.at[:, pl.ds(rank * 128, 128)]
        return tpu.make_async_remote_copy(
            view,
            view,
            latent_send_semaphores.at[latent_slot],
            latent_receive_semaphores.at[latent_slot],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def wait_for_latent_gather() -> None:
        """Compute and stage Pallas wait for latent gather."""
        view = latent_vmem.at[:, : 31 * 128]
        tpu.make_async_copy(view, view, latent_send_semaphores.at[latent_slot]).wait()
        tpu.make_async_copy(
            view, view, latent_receive_semaphores.at[latent_slot]
        ).wait()

    return (
        (copy_expert, expert_host_reduction_vmem, expert_local_reduction_vmem),
        (hidden, latent_copy, latent_output_copy, latent_receive_semaphores),
        (latent_send_semaphores, latent_slot, latent_vmem),
        (output_receive_semaphores, output_send_semaphores, overlap_latent_gather),
        (overlap_router_latent, overlap_shared_routing, rank, receive_semaphores),
        (reduction_phase, routed_consumed, routed_send_vmem, send_semaphores),
        (shared_host_reduction_vmem, shared_local_reduction_vmem),
        wait_for_latent_gather,
    )


def _moe_stage_am(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe stage am."""
    (
        (after_routing_scores, batch_size, communication_workspace),
        (copy_expert, expert_output_size, expert_prefetch_slot_count),
        (expert_tensor_parallel, group_pending_expert_routes, hidden),
        (hidden_size, latent_copy, latent_vmem, live_rows, local_expert_count),
        (local_route_counts_vmem, overlap_latent_gather),
        (overlap_router_latent, overlap_shared_routing, owned_hidden_ref),
        (prepare_weights_vmem, probability_totals_vmem, rank),
        (receive_semaphores, routed_outputs_vmem, routed_reduce_scatter),
        (selected_expert_count, selected_experts_vmem),
        (selected_probabilities_vmem, send_semaphores, shared_gate_up_stream),
        (shared_host_reduction_vmem, shared_local_reduction_vmem),
        (shared_outputs_vmem, shared_receive_semaphores),
        (shared_send_semaphores, shared_sharded, shared_width),
        (wait_for_latent_gather, weights, workspace),
    ) = stage_ctx

    def start_latent_gather(
        latent_input_projection_vmem: TensorLike, latent_input_copy: TensorLike
    ) -> None:
        """Compute and stage Pallas start latent gather."""
        latent_input_copy.wait()
        latent_vmem[:, pl.ds(rank * 128, 128)] = _dot(
            hidden, latent_input_projection_vmem[...]
        ).astype(jnp.bfloat16)
        for offset in (
            31,
            30,
            29,
            27,
            23,
            15,
            28,
            26,
            25,
            22,
            21,
            19,
            14,
            13,
            11,
            7,
            24,
            20,
            18,
            17,
            12,
            10,
            9,
            6,
            5,
            3,
            16,
            8,
            4,
            2,
            1,
        ):
            latent_copy(offset).start()

    def prepare_batch(
        latent_input_projection_vmem: TensorLike,
        latent_input_copy_semaphore: TensorLike,
        shared_gate_up_projection_vmem: TensorLike,
        shared_down_projection_vmem: TensorLike,
        dense_copy_semaphores: TensorLike,
    ) -> None:
        """Compute and stage Pallas VMEM state for prepare batch."""
        return _moe_prepare_batch_stage_ab(
            (
                (after_routing_scores, batch_size, communication_workspace),
                (copy_expert, expert_prefetch_slot_count, expert_tensor_parallel),
                (group_pending_expert_routes, hidden, live_rows, local_expert_count),
                (local_route_counts_vmem, overlap_latent_gather),
                (overlap_router_latent, overlap_shared_routing, owned_hidden_ref),
                (probability_totals_vmem, receive_semaphores, routed_outputs_vmem),
                (selected_expert_count, selected_experts_vmem),
                (selected_probabilities_vmem, send_semaphores, shared_gate_up_stream),
                (shared_outputs_vmem, shared_sharded, shared_width),
                (start_latent_gather, wait_for_latent_gather, weights, workspace),
            ),
            (
                (latent_input_projection_vmem, latent_input_copy_semaphore),
                (shared_gate_up_projection_vmem, shared_down_projection_vmem),
                dense_copy_semaphores,
            ),
        )

    if prepare_weights_vmem is not None:
        latent_down_vmem, shared_gate_up_vmem, shared_down_vmem = prepare_weights_vmem
        pl.run_scoped(
            lambda latent_semaphore, dense_semaphores: prepare_batch(
                latent_down_vmem,
                latent_semaphore,
                shared_gate_up_vmem,
                shared_down_vmem,
                dense_semaphores,
            ),
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA((2,)),
        )
    else:
        pl.run_scoped(
            prepare_batch,
            tpu.VMEM(
                weights.latent_input_projection.shape,
                weights.latent_input_projection.dtype,
            ),
            tpu.SemaphoreType.DMA,
            tpu.VMEM(
                weights.shared_gate_up_projection.shape,
                weights.shared_gate_up_projection.dtype,
            ),
            tpu.VMEM(
                weights.shared_down_projection.shape,
                weights.shared_down_projection.dtype,
            ),
            tpu.SemaphoreType.DMA((2,)),
        )
    shared_rows = (hidden_size + 1023) // 1024 * 8
    (
        (decode_expert, decode_gathered_experts, decode_shared, decode_shared_shard),
        (encode_expert, encode_shared, encode_shared_shard, expert_payload),
        (expert_rows, lane_payload, reduce_scatter_routed_output),
        (reduce_scatter_shared_output, shared_payload, shared_shard_payload),
        (shared_shard_rows, shared_shard_width, tokens_per_lane),
    ) = _moe_stage_ad(
        batch_size,
        expert_output_size,
        group_pending_expert_routes,
        routed_reduce_scatter,
        shared_rows,
    )

    def shared_copy(
        payload: TensorLike, offset: TensorLike, *, across_hosts: bool = False
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for shared copy."""
        lane, host = (rank % 8, rank // 8)
        view = (
            shared_host_reduction_vmem.at[host, :payload, :]
            if across_hosts
            else shared_local_reduction_vmem.at[lane, :payload, :]
        )
        semaphore = 6 + offset if across_hosts else offset - 1
        peer = rank ^ (8 * offset if across_hosts else offset)
        return tpu.make_async_remote_copy(
            view,
            view,
            shared_send_semaphores.at[semaphore],
            shared_receive_semaphores.at[semaphore],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    return (
        (decode_expert, decode_gathered_experts, decode_shared, decode_shared_shard),
        (encode_expert, encode_shared, encode_shared_shard, expert_payload),
        (expert_rows, lane_payload, reduce_scatter_routed_output),
        (reduce_scatter_shared_output, shared_copy, shared_payload, shared_rows),
        (shared_shard_payload, shared_shard_rows, shared_shard_width),
        (start_latent_gather, tokens_per_lane),
    )


def _moe_prepare_batch_stage_ab(stage_ctx: TensorLike, stage_args: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for prepare batch."""
    (
        (after_routing_scores, batch_size, communication_workspace),
        (copy_expert, expert_prefetch_slot_count, expert_tensor_parallel),
        (group_pending_expert_routes, hidden, live_rows, local_expert_count),
        (local_route_counts_vmem, overlap_latent_gather),
        (overlap_router_latent, overlap_shared_routing, owned_hidden_ref),
        (probability_totals_vmem, receive_semaphores, routed_outputs_vmem),
        (selected_expert_count, selected_experts_vmem),
        (selected_probabilities_vmem, send_semaphores, shared_gate_up_stream),
        (shared_outputs_vmem, shared_sharded, shared_width),
        (start_latent_gather, wait_for_latent_gather, weights, workspace),
    ) = stage_ctx
    (
        (latent_input_projection_vmem, latent_input_copy_semaphore),
        (shared_gate_up_projection_vmem, shared_down_projection_vmem),
        dense_copy_semaphores,
    ) = stage_args
    latent_input_copy = tpu.make_async_copy(
        weights.latent_input_projection,
        latent_input_projection_vmem,
        latent_input_copy_semaphore,
    )
    latent_input_copy.start()
    dense_copies = SimpleNamespace(
        gate_up=None
        if shared_gate_up_stream is not None
        else tpu.make_async_copy(
            weights.shared_gate_up_projection,
            shared_gate_up_projection_vmem,
            dense_copy_semaphores.at[0],
        ),
        down=tpu.make_async_copy(
            weights.shared_down_projection,
            shared_down_projection_vmem,
            dense_copy_semaphores.at[1],
        ),
    )
    _start_async_copies(dense_copies)
    if shared_gate_up_stream is not None:
        shared_gate_up_stream.start()
    weights.wait_for_routing_projection_copy()
    shared_route_prefix = jnp.int32(1)
    prefetched_expert_count = jnp.int32(0)

    def prefetch_shared_route_prefix(
        winners: TensorLike,
        owned_locally: TensorLike,
        route_counts: TensorLike,
        _unused_selected_experts: TensorLike,
    ) -> None:
        """Compute and stage Pallas prefetch shared route prefix."""
        if group_pending_expert_routes:
            return
        nonlocal shared_route_prefix, prefetched_expert_count
        shared_route_prefix *= jnp.all(winners == winners[0]).astype(jnp.int32)
        should_prefetch = (
            shared_route_prefix
            * owned_locally.astype(jnp.int32)[0]
            * (route_counts[0] < expert_prefetch_slot_count).astype(jnp.int32)
        )

        @pl.when(should_prefetch != 0)
        def start_expert_copy() -> None:
            """Compute and stage Pallas start expert copy."""
            _start_async_copies(
                copy_expert(winners[0] % local_expert_count, route_counts[0])
            )

        prefetched_expert_count += should_prefetch

    (overlap_router_score_exchange, prepare_shared_compute) = (
        _moe_prepare_batch_stage_ab_stage_ar(
            (
                (communication_workspace, dense_copies, hidden, latent_input_copy),
                (latent_input_projection_vmem, overlap_shared_routing),
                (routed_outputs_vmem, shared_down_projection_vmem),
                (shared_gate_up_projection_vmem, shared_gate_up_stream),
                (shared_outputs_vmem, shared_sharded, shared_width),
                start_latent_gather,
            )
        )
    )
    (local_route_counts, selected_experts) = _moe_prepare_batch_stage_ab_stage_aq(
        (
            (after_routing_scores, batch_size, expert_tensor_parallel, hidden),
            (latent_input_copy, latent_input_projection_vmem, live_rows),
            (local_expert_count, local_route_counts_vmem, overlap_latent_gather),
            (overlap_router_latent, overlap_router_score_exchange),
            (overlap_shared_routing, owned_hidden_ref),
            (prefetch_shared_route_prefix, prepare_shared_compute),
            (probability_totals_vmem, receive_semaphores, selected_expert_count),
            (selected_experts_vmem, selected_probabilities_vmem, send_semaphores),
            (start_latent_gather, weights, workspace),
        )
    )
    _start_initial_expert_weight_copies(
        selected_experts,
        local_route_counts,
        copy_expert,
        slot_count=expert_prefetch_slot_count,
        already_started=prefetched_expert_count,
        group_pending_routes=group_pending_expert_routes,
    )
    if not overlap_latent_gather:
        start_latent_gather(latent_input_projection_vmem, latent_input_copy)
        wait_for_latent_gather()
    if overlap_shared_routing:
        dense_copies.down.wait()
    else:
        prepare_shared_compute()


def _moe_prepare_batch_stage_ab_stage_ar(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe prepare batch stage."""
    (
        (communication_workspace, dense_copies, hidden, latent_input_copy),
        (latent_input_projection_vmem, overlap_shared_routing),
        (routed_outputs_vmem, shared_down_projection_vmem),
        (shared_gate_up_projection_vmem, shared_gate_up_stream),
        (shared_outputs_vmem, shared_sharded, shared_width),
        start_latent_gather,
    ) = stage_ctx

    def overlap_router_score_exchange() -> None:
        """Compute and stage Pallas overlap router score exchange."""
        start_latent_gather(latent_input_projection_vmem, latent_input_copy)

    def prepare_shared_compute() -> None:
        """Compute and stage Pallas prepare shared compute."""
        if shared_gate_up_stream is not None:
            shared_gate_up = (
                shared_gate_up_stream.dot(
                    hidden, communication_workspace.reduction_phase[0] >= 0
                )
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
        else:
            dense_copies.gate_up.wait()
            shared_gate_up = (
                _dot(hidden, shared_gate_up_projection_vmem[...])
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
        middle = _gated_activation(
            shared_gate_up[:, :shared_width], shared_gate_up[:, shared_width:]
        )
        if not overlap_shared_routing:
            dense_copies.down.wait()
        if shared_sharded:
            routed_outputs_vmem[:, :shared_width] = middle.astype(jnp.float32)
        else:
            shared_outputs_vmem[...] = _dot(middle, shared_down_projection_vmem[...])

    return (overlap_router_score_exchange, prepare_shared_compute)


def _moe_prepare_batch_stage_ab_stage_aq(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe prepare batch stage."""
    (
        (after_routing_scores, batch_size, expert_tensor_parallel, hidden),
        (latent_input_copy, latent_input_projection_vmem, live_rows),
        (local_expert_count, local_route_counts_vmem, overlap_latent_gather),
        (overlap_router_latent, overlap_router_score_exchange),
        (overlap_shared_routing, owned_hidden_ref),
        (prefetch_shared_route_prefix, prepare_shared_compute),
        (probability_totals_vmem, receive_semaphores, selected_expert_count),
        (selected_experts_vmem, selected_probabilities_vmem, send_semaphores),
        (start_latent_gather, weights, workspace),
    ) = stage_ctx

    def overlap_routing_with_latent_gather(selection: TensorLike) -> None:
        """Compute and stage Pallas overlap routing with latent gather."""
        if overlap_shared_routing and selection == _LAST_EXPERT_SLOT:
            prepare_shared_compute()
        if (
            selection == _DCP_HEAD_COUNT
            and overlap_latent_gather
            and (not overlap_router_latent)
        ):
            start_latent_gather(latent_input_projection_vmem, latent_input_copy)

    with _phase("p3_route"):
        (
            selected_experts,
            selected_probabilities,
            local_route_counts,
            selected_probability_totals,
        ) = (
            _route_owned_rows(
                owned_hidden_ref,
                batch_size,
                weights.routing_projection_vmem,
                weights.routing_bias,
                workspace.routing_scores_vmem,
                workspace.selection_source_vmem,
                workspace.selections_vmem,
                send_semaphores,
                receive_semaphores,
                selected_expert_count=selected_expert_count,
                local_expert_count=local_expert_count,
                expert_tensor_parallel=expert_tensor_parallel,
                on_local_selection=prefetch_shared_route_prefix,
                after_selection=overlap_routing_with_latent_gather,
                after_scores=after_routing_scores,
                live_rows=live_rows,
            )
            if owned_hidden_ref is not None
            else _route_hidden_states(
                hidden,
                weights.routing_projection_vmem,
                weights.routing_bias,
                workspace.routing_scores_vmem,
                send_semaphores.at[0],
                receive_semaphores.at[0],
                selected_expert_count=selected_expert_count,
                local_expert_count=local_expert_count,
                expert_tensor_parallel=expert_tensor_parallel,
                on_local_selection=prefetch_shared_route_prefix,
                after_selection=overlap_routing_with_latent_gather,
                after_scores=after_routing_scores,
                after_score_send=overlap_router_score_exchange
                if overlap_router_latent
                else None,
                live_rows=live_rows,
            )
        )
    selected_experts_vmem[...] = jnp.pad(
        selected_experts, ((0, 0), (0, 128 - selected_expert_count))
    )
    selected_probabilities_vmem[...] = jnp.pad(
        selected_probabilities, ((0, 0), (0, 128 - selected_expert_count))
    )
    local_route_counts_vmem[...] = jnp.broadcast_to(
        local_route_counts[:, None], local_route_counts_vmem.shape
    )
    probability_totals_vmem[...] = jnp.broadcast_to(
        selected_probability_totals[:, None], probability_totals_vmem.shape
    )
    return (local_route_counts, selected_experts)


def _moe_stage_ad(
    batch_size: TensorLike,
    expert_output_size: TensorLike,
    group_pending_expert_routes: TensorLike,
    routed_reduce_scatter: TensorLike,
    shared_rows: TensorLike,
) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe stage ad."""
    shared_payload = shared_rows // 2
    expert_rows = expert_output_size // 128
    expert_payload = expert_rows // 2
    tokens_per_lane = -(-batch_size // 8)
    lane_payload = tokens_per_lane * expert_payload
    reduce_scatter_shared_output = (
        batch_size in (6, *SCATTER_BATCHES) and group_pending_expert_routes
    )
    reduce_scatter_routed_output = routed_reduce_scatter
    shared_shard_width = 1024
    shared_shard_rows = shared_shard_width // 128
    shared_shard_payload = shared_shard_rows // 2

    def encode_shared(value: TensorLike, destination: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for encode shared."""
        transport_batch_size = value.shape[0]
        packed = tpu.bitcast(
            value.reshape(transport_batch_size * shared_rows, 128).astype(jnp.bfloat16),
            jnp.uint32,
        )
        destination[: transport_batch_size * shared_payload, :] = (
            jax.lax.bitcast_convert_type(packed, jnp.float32)
        )

    def decode_shared(value: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for decode shared."""
        transport_batch_size = value.shape[1] // shared_payload
        decoded = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(decoded, axis=0).reshape(transport_batch_size, shared_rows * 128)

    def encode_shared_shard(value: TensorLike, destination: TensorLike) -> None:
        """Compute and stage Pallas encode shared shard."""
        transport_batch_size = value.shape[0]
        packed = tpu.bitcast(
            value.reshape(transport_batch_size * shared_shard_rows, 128).astype(
                jnp.bfloat16
            ),
            jnp.uint32,
        )
        destination[: transport_batch_size * shared_shard_payload, :] = (
            jax.lax.bitcast_convert_type(packed, jnp.float32)
        )

    def decode_shared_shard(value: TensorLike) -> TensorLike:
        """Compute and stage Pallas decode shared shard."""
        transport_batch_size = value.shape[1] // shared_shard_payload
        decoded = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(decoded, axis=0).reshape(
            transport_batch_size, shared_shard_width
        )

    def encode_expert(value: TensorLike, destination: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for encode expert."""
        transport_batch_size = value.shape[0]
        bits = tpu.bitcast(
            value[:, :expert_output_size]
            .reshape(transport_batch_size * expert_rows, 128)
            .astype(jnp.bfloat16),
            jnp.uint32,
        )
        destination[: transport_batch_size * expert_payload, :] = (
            jax.lax.bitcast_convert_type(bits, jnp.float32)
        )

    def decode_expert(value: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for decode expert."""
        transport_batch_size = value.shape[1] // expert_payload
        values = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(values, axis=0).reshape(transport_batch_size, expert_output_size)

    def decode_gathered_experts(value: TensorLike) -> TensorLike:
        """Compute and stage Pallas decode gathered experts."""
        values = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return values.reshape(8 * tokens_per_lane, expert_output_size)[:batch_size]

    return (
        (decode_expert, decode_gathered_experts, decode_shared, decode_shared_shard),
        (encode_expert, encode_shared, encode_shared_shard, expert_payload),
        (expert_rows, lane_payload, reduce_scatter_routed_output),
        (reduce_scatter_shared_output, shared_payload, shared_shard_payload),
        (shared_shard_rows, shared_shard_width, tokens_per_lane),
    )


def _moe_stage_ak(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe stage ak."""
    (
        (batch_size, decode_shared, decode_shared_shard, encode_shared),
        (encode_shared_shard, expert_host_reduction_vmem),
        (expert_local_reduction_vmem, lane_payload, rank, receive_semaphores),
        (reduce_scatter_shared_output, routed_send_vmem, send_semaphores),
        (shared_copy, shared_down_weights, shared_host_reduction_vmem),
        (shared_local_reduction_vmem, shared_payload),
        (shared_receive_semaphores, shared_send_semaphores),
        (shared_shard_payload, shared_shard_width, shared_sharded),
    ) = stage_ctx

    def shared_shard_send_view(target_lane: TensorLike) -> TensorLike:
        """Compute and stage Pallas shared shard send view."""
        send_rows = (
            expert_host_reduction_vmem.shape[0]
            * expert_host_reduction_vmem.shape[1]
            // 8
        )
        return pool_alias.view(
            expert_host_reduction_vmem, (8, send_rows, 128), jnp.float32
        ).at[target_lane, : batch_size * shared_shard_payload, :]

    def shared_shard_receive_view(source_lane: TensorLike) -> TensorLike:
        """Compute and stage Pallas shared shard receive view."""
        return shared_local_reduction_vmem.at[
            source_lane, : batch_size * shared_shard_payload, :
        ]

    (expert_copy, expert_shard_send_view, routed_send_storage), shared_shard_copy = (
        _moe_stage_ai(
            (
                (expert_host_reduction_vmem, expert_local_reduction_vmem),
                (lane_payload, rank, receive_semaphores, routed_send_vmem),
                (send_semaphores, shared_receive_semaphores, shared_send_semaphores),
                (shared_shard_receive_view, shared_shard_send_view),
            )
        )
    )
    (begin_shared_reduction, expert_shard_copy, expert_shard_receive_view) = (
        _moe_stage_ah(
            (
                (batch_size, encode_shared, encode_shared_shard),
                (expert_local_reduction_vmem, expert_shard_send_view, lane_payload),
                (rank, receive_semaphores, reduce_scatter_shared_output),
                (send_semaphores, shared_copy, shared_down_weights),
                (shared_local_reduction_vmem, shared_payload, shared_shard_copy),
                (shared_shard_payload, shared_shard_receive_view),
                (shared_shard_send_view, shared_shard_width, shared_sharded),
            )
        )
    )

    def finish_shared_reduction(payload: TensorLike) -> None:
        """Compute and stage Pallas finish shared reduction."""
        offset = _NONE_STAGE_VAL
        if reduce_scatter_shared_output:
            for offset in range(7, 0, -1):
                shared_shard_copy(offset).wait()
            encode_shared_shard(
                decode_shared_shard(shared_local_reduction_vmem[:, :payload, :]),
                shared_host_reduction_vmem.at[rank // 8],
            )
        else:
            for offset in range(7, 0, -1):
                shared_copy(payload, offset).wait()
            encode_shared(
                decode_shared(shared_local_reduction_vmem[:, :payload, :]),
                shared_host_reduction_vmem.at[rank // 8],
            )
        for offset in range(3, 0, -1):
            shared_copy(payload, offset, across_hosts=True).start()

    return (
        (begin_shared_reduction, expert_copy, expert_shard_copy),
        (expert_shard_receive_view, expert_shard_send_view, finish_shared_reduction),
        (routed_send_storage, shared_shard_copy, shared_shard_receive_view),
        shared_shard_send_view,
    )


def _moe_stage_ai(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe stage ai."""
    (
        (expert_host_reduction_vmem, expert_local_reduction_vmem),
        (lane_payload, rank, receive_semaphores, routed_send_vmem),
        (send_semaphores, shared_receive_semaphores, shared_send_semaphores),
        (shared_shard_receive_view, shared_shard_send_view),
    ) = stage_ctx

    def shared_shard_copy(offset: TensorLike) -> TensorLike:
        """Compute and stage Pallas shared shard copy."""
        source_lane = rank % 8
        target_lane = source_lane ^ offset
        return tpu.make_async_remote_copy(
            shared_shard_send_view(target_lane),
            shared_shard_receive_view(source_lane),
            shared_send_semaphores.at[offset - 1],
            shared_receive_semaphores.at[offset - 1],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def expert_copy(
        payload: TensorLike, offset: TensorLike, *, across_hosts: bool = False
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for expert copy."""
        view = (
            expert_host_reduction_vmem.at[rank // 8, :payload, :]
            if across_hosts
            else expert_local_reduction_vmem.at[rank % 8, :payload, :]
        )
        semaphore = 6 + offset if across_hosts else offset - 1
        peer = rank ^ (8 * offset if across_hosts else offset)
        return tpu.make_async_remote_copy(
            view,
            view,
            send_semaphores.at[semaphore],
            receive_semaphores.at[semaphore],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def routed_send_storage() -> TensorLike:
        """Compute and stage Pallas routed send storage."""
        if routed_send_vmem is not None:
            return routed_send_vmem.at[: 8 * lane_payload, :]
        return expert_local_reduction_vmem.at[rank % 8, : 8 * lane_payload, :]

    def expert_shard_send_view(target_lane: TensorLike) -> TensorLike:
        """Compute and stage Pallas expert shard send view."""
        return routed_send_storage().reshape(8, lane_payload, 128).at[target_lane]

    return (
        (expert_copy, expert_shard_send_view, routed_send_storage),
        shared_shard_copy,
    )


def _moe_stage_ah(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe stage ah."""
    (
        (batch_size, encode_shared, encode_shared_shard),
        (expert_local_reduction_vmem, expert_shard_send_view, lane_payload),
        (rank, receive_semaphores, reduce_scatter_shared_output),
        (send_semaphores, shared_copy, shared_down_weights),
        (shared_local_reduction_vmem, shared_payload, shared_shard_copy),
        (shared_shard_payload, shared_shard_receive_view),
        (shared_shard_send_view, shared_shard_width, shared_sharded),
    ) = stage_ctx

    def expert_shard_receive_view(source_lane: TensorLike) -> TensorLike:
        """Compute and stage Pallas expert shard receive view."""
        return expert_local_reduction_vmem.at[source_lane, :lane_payload, :]

    def expert_shard_copy(offset: TensorLike) -> TensorLike:
        """Compute and stage Pallas expert shard copy."""
        source_lane = rank % 8
        target_lane = source_lane ^ offset
        return tpu.make_async_remote_copy(
            expert_shard_send_view(target_lane),
            expert_shard_receive_view(source_lane),
            send_semaphores.at[offset - 1],
            receive_semaphores.at[offset - 1],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def begin_shared_reduction(local_shared_outputs: TensorLike) -> None:
        """Compute and stage Pallas begin shared reduction."""
        if reduce_scatter_shared_output:
            lane = rank % 8
            for target_lane in range(7):
                columns = slice(
                    target_lane * shared_shard_width,
                    (target_lane + 1) * shared_shard_width,
                )
                encode_shared_shard(
                    _dot(local_shared_outputs, shared_down_weights[:, columns])
                    if shared_sharded
                    else local_shared_outputs[:, columns],
                    shared_shard_send_view(target_lane),
                )
            shared_shard_send_view(7)[...] = jnp.zeros(
                (batch_size * shared_shard_payload, 128), jnp.float32
            )
            shared_shard_receive_view(lane)[...] = shared_shard_send_view(lane)[...]
            for offset in range(7, 0, -1):
                shared_shard_copy(offset).start()
        else:
            payload = local_shared_outputs.shape[0] * shared_payload
            np.testing.assert_(payload <= shared_local_reduction_vmem.shape[1])
            encode_shared(
                local_shared_outputs, shared_local_reduction_vmem.at[rank % 8]
            )
            for offset in range(7, 0, -1):
                shared_copy(payload, offset).start()

    return (begin_shared_reduction, expert_shard_copy, expert_shard_receive_view)


def _moe_stage_an(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe stage an."""
    (
        (after_experts, batch_size, begin_shared_reduction, copy_expert),
        (decode_expert, decode_gathered_experts, down_accumulator_vmem),
        (encode_expert, epsilon, expert_copy, expert_down_scale_vmem),
        (expert_down_vmem, expert_gate_up_scale_vmem, expert_gate_up_vmem),
        (expert_host_reduction_vmem, expert_input_size),
        (expert_local_reduction_vmem, expert_output_size, expert_payload),
        (expert_shard_copy, expert_shard_receive_view, expert_shard_send_view),
        (expert_width, finish_shared_host_reduction, finish_shared_reduction),
        (first_compact_source_vmem, gate_up_accumulator_vmem),
        (group_pending_expert_routes, hidden_ref, hidden_size, lane),
        (lane_payload, latent_feature_count, latent_output_copy, latent_vmem),
        (local_route_counts_vmem, output_receive_semaphores, output_ref),
        (output_send_semaphores, output_vmem, overlap_latent_gather),
        (overlap_shared_reduction, probability_totals_vmem),
        (processed_routes_vmem, rank, reduce_scatter_routed_output),
        (reduce_scatter_shared_output, reduction_phase, routed_consumed),
        (routed_outputs_vmem, routed_send_storage, routed_send_vmem),
        (second_compact_source_vmem, selected_expert_count),
        (selected_experts_vmem, selected_probabilities_vmem),
        (shared_local_reduction_vmem, shared_outputs_vmem, shared_payload),
        (shared_receive_semaphores, shared_send_semaphores),
        (shared_shard_payload, shared_sharded, shared_width),
        (token_scatter_output, token_scatter_owners, tokens_per_lane),
        (wait_for_latent_gather, weights),
    ) = stage_ctx

    def finish_reductions(
        local_expert_outputs: TensorLike,
        reduced_expert_outputs: TensorLike,
        reduced_shared_outputs: TensorLike,
    ) -> None:
        """Compute and stage Pallas VMEM state for finish reductions."""
        return _moe_finish_reductions_stage_aa(
            (
                (after_experts, batch_size, decode_expert, decode_gathered_experts),
                (encode_expert, expert_copy, expert_host_reduction_vmem),
                (expert_local_reduction_vmem, expert_output_size, expert_payload),
                (expert_shard_copy, expert_shard_receive_view, expert_shard_send_view),
                (finish_shared_host_reduction, finish_shared_reduction, lane_payload),
                (latent_output_copy, overlap_shared_reduction, rank),
                (reduce_scatter_routed_output, reduce_scatter_shared_output),
                (reduction_phase, routed_consumed, routed_send_storage),
                (routed_send_vmem, shared_local_reduction_vmem, shared_payload),
                (shared_shard_payload, tokens_per_lane),
            ),
            ((local_expert_outputs, reduced_expert_outputs, reduced_shared_outputs),),
        )

    if shared_sharded:
        np.testing.assert_(reduce_scatter_shared_output)
        begin_shared_reduction(
            routed_outputs_vmem[:, :shared_width].astype(jnp.bfloat16)
        )
    else:
        begin_shared_reduction(shared_outputs_vmem[...])
    if overlap_latent_gather:
        wait_for_latent_gather()
    t7, up = _moe_stage_ae(
        (
            (batch_size, copy_expert, down_accumulator_vmem, epsilon),
            (expert_down_scale_vmem, expert_down_vmem, expert_gate_up_scale_vmem),
            (expert_gate_up_vmem, expert_input_size, expert_output_size),
            (expert_width, finish_reductions, finish_shared_reduction),
            (first_compact_source_vmem, gate_up_accumulator_vmem),
            (group_pending_expert_routes, latent_feature_count),
            (latent_output_copy, latent_vmem, local_route_counts_vmem),
            (overlap_shared_reduction, probability_totals_vmem),
            (processed_routes_vmem, reduce_scatter_shared_output),
            (routed_outputs_vmem, second_compact_source_vmem),
            (selected_expert_count, selected_experts_vmem),
            (selected_probabilities_vmem, shared_outputs_vmem, shared_payload),
            (shared_receive_semaphores, shared_send_semaphores),
            (shared_shard_payload, weights),
        )
    )
    local_output = (
        up
        + (
            shared_outputs_vmem[...]
            if shared_sharded
            else shared_outputs_vmem[:, pl.ds(rank % 8 * 1024, 1024)]
        )
    ).astype(jnp.bfloat16)
    _phase_end(t7)
    t8 = _phase_start("t8_output_transport")
    if token_scatter_output:
        owners = token_scatter_owners
        tokens_per_pair = batch_size // owners
        payload_rows = tokens_per_pair * 1024 // 128
        lane = rank % 8
        pair = lane // (8 // owners)
        send = output_vmem.at[pl.ds(0, owners)]
        receive = output_vmem.at[pl.ds(owners, 8)]
        for target_pair in range(owners):
            row_start = target_pair * tokens_per_pair
            send[target_pair] = local_output[
                row_start : row_start + tokens_per_pair
            ].reshape(payload_rows, 128)
        receive[lane] = send[pair]
        copies = []
        for offset in range(7, 0, -1):
            target_lane = lane ^ offset
            target_pair = target_lane // (8 // owners)
            copy = tpu.make_async_remote_copy(
                send.at[target_pair],
                receive.at[lane],
                output_send_semaphores.at[offset - 1],
                output_receive_semaphores.at[offset - 1],
                device_id=(rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        for source_lane in range(7):
            block = receive[source_lane].reshape(tokens_per_pair, 1024)
            output_ref[:, pl.ds(source_lane * 1024, 1024)] = block
    else:
        output_vmem[:, pl.ds(rank % 8 * 1024, 1024)] = local_output
        _gather_output_shards(
            output_vmem,
            output_send_semaphores,
            output_receive_semaphores,
            reduction_phase[0] % 2,
        )
        output_ref[...] = output_vmem[:batch_size, :hidden_size].astype(
            hidden_ref.dtype
        )
    _phase_end(t8)
    reduction_phase[0] += 1
    return lane


def _moe_finish_reductions_stage_aa(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for finish reductions."""
    (
        (after_experts, batch_size, decode_expert, decode_gathered_experts),
        (encode_expert, expert_copy, expert_host_reduction_vmem),
        (expert_local_reduction_vmem, expert_output_size, expert_payload),
        (expert_shard_copy, expert_shard_receive_view, expert_shard_send_view),
        (finish_shared_host_reduction, finish_shared_reduction, lane_payload),
        (latent_output_copy, overlap_shared_reduction, rank),
        (reduce_scatter_routed_output, reduce_scatter_shared_output),
        (reduction_phase, routed_consumed, routed_send_storage),
        (routed_send_vmem, shared_local_reduction_vmem, shared_payload),
        (shared_shard_payload, tokens_per_lane),
    ) = stage_ctx
    ((local_expert_outputs, reduced_expert_outputs, reduced_shared_outputs),) = (
        stage_args
    )
    offset = _NONE_STAGE_VAL
    transport_batch_size = local_expert_outputs.shape[0]
    shared_transport_payload = transport_batch_size * (
        shared_shard_payload if reduce_scatter_shared_output else shared_payload
    )
    expert_transport_payload = transport_batch_size * expert_payload
    np.testing.assert_(shared_transport_payload <= shared_local_reduction_vmem.shape[1])
    np.testing.assert_(
        reduce_scatter_routed_output
        or expert_transport_payload <= expert_local_reduction_vmem.shape[1]
    )
    _moe_finish_reductions_stage_aa_stage_ap(
        (
            (after_experts, batch_size, encode_expert, expert_copy),
            (expert_local_reduction_vmem, expert_shard_copy),
            (expert_transport_payload, finish_shared_reduction),
            (latent_output_copy, local_expert_outputs, overlap_shared_reduction),
            (rank, reduce_scatter_routed_output, routed_consumed),
            (routed_send_storage, routed_send_vmem, shared_transport_payload),
            tokens_per_lane,
        )
    )
    if reduce_scatter_routed_output:
        _moe_finish_reductions_stage_aa_stage_ao(
            (
                (decode_expert, decode_gathered_experts, encode_expert, expert_copy),
                (expert_host_reduction_vmem, expert_local_reduction_vmem),
                (expert_output_size, expert_shard_copy, expert_shard_receive_view),
                (expert_shard_send_view, finish_shared_host_reduction, lane_payload),
                (overlap_shared_reduction, rank, reduced_expert_outputs),
                (reduced_shared_outputs, reduction_phase, routed_consumed),
                shared_transport_payload,
            )
        )
        return
    for offset in range(7, 0, -1):
        expert_copy(expert_transport_payload, offset).wait()
    encode_expert(
        decode_expert(expert_local_reduction_vmem[:, :expert_transport_payload, :]),
        expert_host_reduction_vmem.at[rank // 8],
    )
    for offset in range(3, 0, -1):
        expert_copy(expert_transport_payload, offset, across_hosts=True).start()
    for offset in range(3, 0, -1):
        expert_copy(expert_transport_payload, offset, across_hosts=True).wait()
    reduced_expert_outputs[...] = jnp.zeros(reduced_expert_outputs.shape, jnp.float32)
    reduced_expert_outputs[:, :expert_output_size] = decode_expert(
        expert_host_reduction_vmem[:, :expert_transport_payload, :]
    )
    finish_shared_host_reduction(shared_transport_payload, reduced_shared_outputs)
    reduction_phase[0] += 2


def _moe_finish_reductions_stage_aa_stage_ap(stage_ctx: TensorLike) -> None:
    """Compute Pallas VMEM tensors for moe stage."""
    (
        (after_experts, batch_size, encode_expert, expert_copy),
        (expert_local_reduction_vmem, expert_shard_copy),
        (expert_transport_payload, finish_shared_reduction),
        (latent_output_copy, local_expert_outputs, overlap_shared_reduction),
        (rank, reduce_scatter_routed_output, routed_consumed),
        (routed_send_storage, routed_send_vmem, shared_transport_payload),
        tokens_per_lane,
    ) = stage_ctx
    np.testing.assert_(routed_send_vmem is None or reduce_scatter_routed_output)
    np.testing.assert_(routed_consumed is not None or not reduce_scatter_routed_output)
    latent_output_copy.start()
    if reduce_scatter_routed_output:
        t1 = _phase_start("t1_routed_send")
        routed_scatter_storage = routed_send_storage()
        if batch_size < 8 * tokens_per_lane:
            routed_scatter_storage[...] = jnp.zeros(
                routed_scatter_storage.shape, jnp.float32
            )
        encode_expert(local_expert_outputs, routed_scatter_storage)
        for offset in range(7, 0, -1):
            expert_shard_copy(offset).start()
        _phase_end(t1)
    else:
        encode_expert(local_expert_outputs, expert_local_reduction_vmem.at[rank % 8])
        for offset in range(7, 0, -1):
            expert_copy(expert_transport_payload, offset).start()
    if after_experts is not None:
        after_experts()
    t2 = _phase_start("t2_shared_local")
    if not overlap_shared_reduction:
        finish_shared_reduction(shared_transport_payload)
    _phase_end(t2)


def _moe_finish_reductions_stage_aa_stage_ao(stage_ctx: TensorLike) -> None:
    """Compute Pallas VMEM tensors for moe stage."""
    (
        (decode_expert, decode_gathered_experts, encode_expert, expert_copy),
        (expert_host_reduction_vmem, expert_local_reduction_vmem),
        (expert_output_size, expert_shard_copy, expert_shard_receive_view),
        (expert_shard_send_view, finish_shared_host_reduction, lane_payload),
        (overlap_shared_reduction, rank, reduced_expert_outputs),
        (reduced_shared_outputs, reduction_phase, routed_consumed),
        shared_transport_payload,
    ) = stage_ctx
    t3 = _phase_start("t3_routed_local")
    lane = rank % 8
    for offset in range(7, 0, -1):
        expert_shard_copy(offset).wait()
    owned_shard = expert_shard_send_view(lane)[...]
    expert_shard_receive_view(lane)[...] = owned_shard
    encode_expert(
        decode_expert(expert_local_reduction_vmem[:, :lane_payload, :]),
        expert_host_reduction_vmem.at[rank // 8],
    )
    for offset in range(1, 8):
        pl.semaphore_signal(
            routed_consumed,
            1,
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    _phase_end(t3)
    t4 = _phase_start("t4_routed_host")
    for offset in range(3, 0, -1):
        expert_copy(lane_payload, offset, across_hosts=True).start()
    for offset in range(3, 0, -1):
        expert_copy(lane_payload, offset, across_hosts=True).wait()
    _phase_end(t4)
    t5 = _phase_start("t5_routed_gather")
    reduced_token = decode_expert(expert_host_reduction_vmem[:, :lane_payload, :])
    encode_expert(reduced_token, expert_local_reduction_vmem.at[lane])
    pl.semaphore_wait(routed_consumed, 7)
    for offset in range(7, 0, -1):
        expert_copy(lane_payload, offset).start()
    if overlap_shared_reduction:
        with _phase("t6_shared_during_routed_gather"):
            finish_shared_host_reduction(
                shared_transport_payload, reduced_shared_outputs
            )
    for offset in range(7, 0, -1):
        expert_copy(lane_payload, offset).wait()
    reduced_expert_outputs[:, :expert_output_size] = decode_gathered_experts(
        expert_local_reduction_vmem[:, :lane_payload, :]
    )
    _phase_end(t5)
    if not overlap_shared_reduction:
        t6 = _phase_start("t6_shared_host")
        finish_shared_host_reduction(shared_transport_payload, reduced_shared_outputs)
        _phase_end(t6)
    reduction_phase[0] += 2


def _moe_stage_ae(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe stage ae."""
    (
        (batch_size, copy_expert, down_accumulator_vmem, epsilon),
        (expert_down_scale_vmem, expert_down_vmem, expert_gate_up_scale_vmem),
        (expert_gate_up_vmem, expert_input_size, expert_output_size),
        (expert_width, finish_reductions, finish_shared_reduction),
        (first_compact_source_vmem, gate_up_accumulator_vmem),
        (group_pending_expert_routes, latent_feature_count),
        (latent_output_copy, latent_vmem, local_route_counts_vmem),
        (overlap_shared_reduction, probability_totals_vmem),
        (processed_routes_vmem, reduce_scatter_shared_output),
        (routed_outputs_vmem, second_compact_source_vmem),
        (selected_expert_count, selected_experts_vmem),
        (selected_probabilities_vmem, shared_outputs_vmem, shared_payload),
        (shared_receive_semaphores, shared_send_semaphores),
        (shared_shard_payload, weights),
    ) = stage_ctx

    def run_with_shared_progress(progress_state: TensorLike) -> None:
        """Compute and stage Pallas run with shared progress."""
        progress_state[0] = 0
        payload = batch_size * (
            shared_shard_payload if reduce_scatter_shared_output else shared_payload
        )
        expected = payload * 128 * 4 // 32

        def poll_shared(iteration: TensorLike) -> None:
            """Compute and stage Pallas VMEM state for poll shared."""

            @pl.when((progress_state[0] == 0) & (iteration % 4 == 0))
            def try_progress() -> None:
                """Compute and stage Pallas VMEM state for try progress."""
                ready = jnp.bool_(bool(1))
                for peer in range(7):
                    ready = ready & (
                        pl.semaphore_read(shared_send_semaphores.at[peer]) >= expected
                    )
                    ready = ready & (
                        pl.semaphore_read(shared_receive_semaphores.at[peer])
                        >= expected
                    )

                @pl.when(ready)
                def advance() -> None:
                    """Compute and stage Pallas VMEM state for advance."""
                    with _phase("t2_shared_local_polled"):
                        finish_shared_reduction(payload)
                    progress_state[0] = 1

        with _phase("p4_experts"):
            _run_routed_expert_stream(
                latent_vmem,
                selected_experts_vmem[:, :selected_expert_count],
                selected_probabilities_vmem[:, :selected_expert_count],
                local_route_counts_vmem[:, 0],
                probability_totals_vmem[:, 0],
                expert_gate_up_vmem,
                expert_gate_up_scale_vmem,
                expert_down_vmem,
                expert_down_scale_vmem,
                routed_outputs_vmem,
                gate_up_accumulator_vmem,
                down_accumulator_vmem,
                processed_routes_vmem,
                first_compact_source_vmem,
                second_compact_source_vmem,
                copy_expert,
                expert_input_size=expert_input_size,
                expert_width=expert_width,
                expert_output_size=expert_output_size,
                group_pending_routes=group_pending_expert_routes,
                progress=poll_shared,
            )

        @pl.when(progress_state[0] == 0)
        def drain_shared() -> None:
            """Compute and stage Pallas VMEM state for drain shared."""
            with _phase("t2_shared_local_drain"):
                finish_shared_reduction(payload)
            progress_state[0] = 1

    if overlap_shared_reduction:
        pl.run_scoped(
            run_with_shared_progress, progress_state=tpu.SMEM((1,), jnp.int32)
        )
    else:
        with _phase("p4_experts"):
            _run_routed_expert_stream(
                latent_vmem,
                selected_experts_vmem[:, :selected_expert_count],
                selected_probabilities_vmem[:, :selected_expert_count],
                local_route_counts_vmem[:, 0],
                probability_totals_vmem[:, 0],
                expert_gate_up_vmem,
                expert_gate_up_scale_vmem,
                expert_down_vmem,
                expert_down_scale_vmem,
                routed_outputs_vmem,
                gate_up_accumulator_vmem,
                down_accumulator_vmem,
                processed_routes_vmem,
                first_compact_source_vmem,
                second_compact_source_vmem,
                copy_expert,
                expert_input_size=expert_input_size,
                expert_width=expert_width,
                expert_output_size=expert_output_size,
                group_pending_routes=group_pending_expert_routes,
                progress=None,
            )
    finish_reductions(
        routed_outputs_vmem[...], routed_outputs_vmem, shared_outputs_vmem
    )
    t7 = _phase_start("t7_latent_up")
    routed = routed_outputs_vmem[...].astype(jnp.bfloat16).astype(jnp.float32)
    normalized = (
        routed
        * jax.lax.rsqrt(
            jnp.sum(routed * routed, axis=1, keepdims=True) / latent_feature_count
            + epsilon
        )
        * weights.latent_normalization[...]
    ).astype(jnp.bfloat16)
    latent_output_copy.wait()
    up = _dot(
        normalized[:, :expert_output_size], weights.latent_output_projection_vmem[...]
    )
    return (t7, up)


STREAMED_WEIGHTS = {
    "k_projection": ("columns", 0),
    "k_o": ("rows", 4),
    "m_qb": ("columns", 0),
    "m_kb": ("columns", 4),
    "m_gate": ("columns", 8),
    "m_o": ("rows", 12),
    "m_qa": ("columns", 16),
    "shared_gu": ("columns", 20),
}
WEIGHT_STREAMS = {name: tuple(value) for name, value in _config().weight_streams}
np.testing.assert_(
    not set(WEIGHT_STREAMS) - set(STREAMED_WEIGHTS),
    str.join(
        "",
        (
            "KernelConfig.weight_streams names unknown ",
            "weights ",
            f"{sorted(set(WEIGHT_STREAMS) - set(STREAMED_WEIGHTS))}",
        ),
    ),
)
DEEP_STREAM_BASE = 24


def _stream_base(name: TensorLike, depth: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stream base."""
    return STREAMED_WEIGHTS[name][1] if depth <= _TP_RING_SIZE else DEEP_STREAM_BASE


def _stream_semaphore_count(streams: TensorLike) -> int:
    """Compute and stage Pallas VMEM state for stream semaphore count.

    Stream semaphores for a configuration (each weight owns 4 from its
    base). A minimum of 8 keeps the scalar schedule stable.
    """
    return max(
        [8]
        + [
            _stream_base(name, depth) + depth
            for name, (chunks, depth) in streams.items()
            if chunks > 1
        ]
    )


DCP_B32_WEIGHT_STREAMS = {
    "m_qa": (12, 4),
    "k_projection": (13, 4),
    "shared_gu": (3, 3),
    "m_gate": (3, 2),
}
DCP_B32_DECODE_RINGS = {"m_qa": (12, 7), "k_projection": (13, 7)}
ATTENTION_DP_WEIGHTS = {
    "k_projection": ("concat", 1),
    "k_o": ("concat", 0),
    "k_fb": ("concat", 1),
    "k_conv": ("stack", None),
    "k_a_log": ("stack", None),
    "k_dt": ("stack", None),
    "m_qb": ("concat", 1),
    "m_kb": ("concat", 1),
    "m_gate": ("concat", 1),
    "m_o": ("concat", 0),
}
DP_WEIGHT_STREAMS = {
    2: {
        "k_projection": (2, 1),
        "k_o": (4, 2),
        "m_qb": (2, 1),
        "m_kb": (2, 1),
        "m_gate": (2, 1),
        "m_o": (4, 2),
    },
    4: {
        "k_projection": (4, 1),
        "k_o": (8, 2),
        "m_qb": (4, 1),
        "m_kb": (4, 1),
        "m_gate": (4, 1),
        "m_o": (8, 2),
    },
}


def _dp_dcp_weight_streams(dp: int, group_rows: int) -> dict[str, tuple[int, int]]:
    """Compute and stage Pallas VMEM state for dp dcp weight streams.

    Attention DP under DCP: the KDA input projection (13 chunks per
    held shard) and MLA q-down (12) stream so the DCP pages and
    packets fit, while key-value-up stays resident. Ring depths shrink
    as the group rows grow; with one group row (DP2 B2) MLA query-up
    and gate are also copied whole.
    """
    if dp != _PAIR_TUPLE_LEN:
        depths = {8: (13, 12), 16: (7, 7)}.get(group_rows, (6, 4))
    elif group_rows <= _ROUTED_EXPERTS_PER_TOKEN:
        depths = (13, 12)
    else:
        depths = {16: (10, 9), 32: (7, 7)}.get(group_rows, (4, 4))
    streams = {
        "k_projection": (13 * dp, depths[0]),
        "m_qa": (12, depths[1]),
        "m_kb": (dp, dp),
    }
    if dp == _PAIR_TUPLE_LEN and group_rows == 1:
        streams.update(m_qb=(dp, dp), m_gate=(dp, dp))
    return streams


def _attention_projection_streams(dp: int, schedule: str) -> dict[str, tuple[int, int]]:
    """Compute Pallas VMEM tensors for attention stage.

    Static stream overrides for the opt-in attention-DP projection
    schedule.

    ``mla_up_double`` double-buffers MLA query-up and KV-up without
    changing the projection dot shapes; the gate projection stays
    single-buffered.
    """
    choices = ("default", "mla_up_double")
    np.testing.assert_(
        not schedule not in choices,
        f"attention_projection_schedule must be one of {choices}",
    )
    if schedule == "default":
        return {}
    np.testing.assert_(
        not dp not in (2, 4),
        "Projection schedule experiments require attention DP2/DP4",
    )
    return dict.fromkeys(("m_qb", "m_kb"), (dp, 2))


def _chunk_sizes(extent: int, chunks: int, unit: int) -> tuple[int, ...]:
    """Compute and stage Pallas VMEM state for chunk sizes.

    Split ``extent`` into ``chunks`` whole ``unit`` blocks, larger
    first.
    """
    blocks, remainder = divmod(extent // unit, chunks)
    np.testing.assert_(
        not (extent % unit or not 1 <= chunks <= extent // unit),
        f"Cannot split {extent} into {chunks} chunks of whole {unit}-blocks",
    )
    return tuple(unit * (blocks + (index < remainder)) for index in range(chunks))


class ChunkedWeightStream:
    """A weight copied from HBM in chunks, through a ring of VMEM slots.

    ``source`` [K, N] is split along ``axis`` ("columns": output
    columns; "rows": the contraction) into ``sizes``. Position i of
    the stream holds chunk ``order(i)`` (default i; may be a traced
    value when all chunks have one size), in slot i % depth,
    signalling semaphore ``base + i % depth``. With ``depth`` equal to
    the chunk count the slots are the chunks' own parts of ``full``
    (the weight's whole VMEM view); a smaller depth reuses slots
    inside ``full``'s pool range (``pool`` from byte ``offset``).

    ``start`` issues the first ``depth`` positions. Consumers ``wait``
    for a position, read ``view``, then ``release`` it, which issues
    the position ``depth`` ahead into the freed slot; ``dot`` does
    this for column streams. Chunked products are not bit-identical to
    the whole-weight product (the compiler's MXU accumulation split
    depends on the shape).
    """

    def __init__(
        self,
        source: TensorLike,
        full: TensorLike,
        semaphores: TensorLike,
        **stage_kwargs: TensorLike,
    ) -> None:
        """Initialize instance state."""
        axis, sizes, depth, base, pool, offset, order = (
            stage_kwargs["axis"],
            stage_kwargs["sizes"],
            stage_kwargs["depth"],
            stage_kwargs.get("base", 0),
            stage_kwargs.get("pool", _NONE_STAGE_VAL),
            stage_kwargs.get("offset", _NONE_STAGE_VAL),
            stage_kwargs.get("order", _NONE_STAGE_VAL),
        )
        self.shards: TensorLike = 1
        np.testing.assert_(
            not axis not in ("columns", "rows"),
            "Stream axis must be 'columns' or 'rows'",
        )
        self.axis, self.sizes, self.depth, self.chunks = (
            axis,
            tuple(sizes),
            depth,
            len(sizes),
        )
        extent = source.shape[1 if axis == "columns" else 0]
        np.testing.assert_(
            sum(self.sizes) == extent, "Chunk sizes must cover the streamed axis"
        )
        np.testing.assert_(
            not (not 1 <= depth <= self.chunks or base + depth > semaphores.shape[0]),
            "Stream depth must fit the chunks and the semaphores",
        )
        np.testing.assert_(
            not (order is not None and len(set(self.sizes)) != 1),
            "A reordered stream needs equal chunk sizes",
        )
        self.source, self.full, self.semaphores, self.base, self.order = (
            source,
            full,
            semaphores,
            base,
            order,
        )
        self.slots = None
        if depth < self.chunks:
            np.testing.assert_(
                pool is not None, "A ring stream needs its weight's pool range"
            )
            size = max(self.sizes)
            shape = (
                (full.shape[0], size) if axis == "columns" else (size, full.shape[1])
            )
            slot_bytes = pool_alias.storage_size(shape, full.dtype)
            np.testing.assert_(
                not depth * slot_bytes
                > pool_alias.storage_size(full.shape, full.dtype),
                "Ring slots exceed the weight's pool range",
            )
            self.slots = tuple(
                pool_alias.view(
                    pool.at[
                        pl.ds((offset + slot * slot_bytes) // 128, slot_bytes // 128),
                        :,
                    ],
                    shape,
                    full.dtype,
                )
                for slot in range(depth)
            )

    def _chunk(self, position: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for chunk."""
        return position if self.order is None else self.order(position)

    def _extent(self, position: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for extent."""
        chunk = self._chunk(position)
        if self.order is None:
            return (sum(self.sizes[:chunk]), self.sizes[chunk])
        return (chunk * self.sizes[0], self.sizes[0])

    def _part(self, ref: TensorLike, start: TensorLike, size: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for part."""
        if self.axis == "columns":
            return ref.at[:, pl.ds(start, size)]
        return ref.at[pl.ds(start, size)]

    def view(self, position: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for view."""
        start, size = self._extent(position)
        if self.chunks == 1:
            return self.full
        if self.slots is None:
            return self._part(self.full, start, size)
        return self._part(self.slots[position % self.depth], 0, size)

    def _copy(self, position: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for copy."""
        start, size = self._extent(position)
        source = (
            self.source if self.chunks == 1 else self._part(self.source, start, size)
        )
        return tpu.make_async_copy(
            source,
            self.view(position),
            self.semaphores.at[self.base + position % self.depth],
        )

    def start(self) -> None:
        """Compute and stage Pallas VMEM state for start."""
        for position in range(self.depth):
            self._copy(position).start()

    def wait(self, position: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for wait."""
        self._copy(position).wait()

    def wait_all(self) -> None:
        """Compute and stage Pallas VMEM state for wait all."""
        np.testing.assert_(
            not self.slots is not None,
            "Only a stream with a slot per chunk can wait for all chunks",
        )
        for position in range(self.chunks):
            self.wait(position)

    def release(self, position: TensorLike, guard: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for release.

        Refill the slot freed by ``position``. ``guard`` is a runtime-true
        condition that keeps the refill in its own region, after the loads
        of the chunk it overwrites.
        """
        if position + self.depth < self.chunks:

            @pl.when(guard)
            def refill() -> None:
                """Compute and stage Pallas VMEM state for refill."""
                self._copy(position + self.depth).start()

    def dot_position(
        self, x: TensorLike, position: TensorLike, guard: TensorLike
    ) -> TensorLike:
        """``x`` times one column chunk (FP32): wait, multiply, release."""
        np.testing.assert_(self.axis == "columns", "dot_position is for column streams")
        self.wait(position)
        result = _dot(x, self.view(position)[...])
        self.release(position, guard)
        return result

    def dot(self, x: TensorLike, guard: TensorLike) -> TensorLike:
        """``x @ source`` in FP32 for a column stream."""
        np.testing.assert_(self.axis == "columns", "dot is for column streams")
        if self.slots is None:
            self.wait_all()
        parts = []
        for position in range(self.chunks):
            if self.slots is not None:
                self.wait(position)
            parts.append(_dot(x, self.view(position)[...]))
            self.release(position, guard)
        return parts[0] if self.chunks == 1 else jnp.concatenate(parts, axis=1)


class KdaStateStream:
    """Compute and stage Pallas VMEM state for KdaStateStream.

    KDA state streamed through three rolling or four private VMEM
    buffers.

    Normally buffer c % 3 is reused only after its write has landed.
    B32's unrolled path uses four buffers: prime chunks 0/1 before
    projection, then start chunks 2/3 at recurrence entry. Final
    writes drain on exit. With slots, read each sequence's initial
    state directly from the bank; subsequent verifier rows are
    produced by the recurrence before use.
    """

    def __init__(
        self,
        history_hbm: TensorLike,
        state_hbm: TensorLike,
        history_output_hbm: TensorLike,
        state_output_hbm: TensorLike,
        *step_args: TensorLike,
        **stage_kwargs: TensorLike,
    ) -> None:
        """Initialize instance state."""
        (
            history_buffers,
            state_buffers,
            read_semaphores,
            write_semaphores,
            chunk,
            chunks,
            slots,
            sequence_rows,
        ) = (
            (*step_args[0:1], stage_kwargs.get("history_buffers", _NONE_STAGE_VAL))[0],
            (*step_args[1:2], stage_kwargs.get("state_buffers", _NONE_STAGE_VAL))[0],
            (*step_args[2:3], stage_kwargs.get("read_semaphores", _NONE_STAGE_VAL))[0],
            (*step_args[3:4], stage_kwargs.get("write_semaphores", _NONE_STAGE_VAL))[0],
            stage_kwargs["chunk"],
            stage_kwargs["chunks"],
            stage_kwargs.get("slots", _NONE_STAGE_VAL),
            stage_kwargs.get("sequence_rows", 1),
        )
        self.history_hbm, self.state_hbm = (history_hbm, state_hbm)
        self.history_output_hbm, self.state_output_hbm = (
            history_output_hbm,
            state_output_hbm,
        )
        self.history_buffers, self.state_buffers = (history_buffers, state_buffers)
        self.read_semaphores, self.write_semaphores = (
            read_semaphores,
            write_semaphores,
        )
        self.chunk, self.chunks = (chunk, chunks)
        self.slots = slots
        self.sequence_rows = sequence_rows
        self.pending_writes = []
        self.buffer_count = history_buffers.shape[0]
        self.four_buffers = self.buffer_count == _TP_RING_SIZE
        np.testing.assert_(
            not self.four_buffers
            or (
                self.chunks == _TP_RING_SIZE
                and self.chunk == _ROUTED_EXPERTS_PER_TOKEN
                and (not KDA_STAGED)
            )
        )

    def _reads(self, index: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for reads."""
        rows, slot = (pl.ds(index * self.chunk, self.chunk), index % self.buffer_count)
        if self.slots is not None:
            return tuple(
                (
                    tpu.make_async_copy(
                        source.at[pl.ds(self.slots[index * self.chunk + row], 1)],
                        buffer.at[slot, pl.ds(row, 1)],
                        self.read_semaphores.at[slot, kind],
                    )
                    for kind, (source, buffer) in enumerate(
                        (
                            (self.history_hbm, self.history_buffers),
                            (self.state_hbm, self.state_buffers),
                        )
                    )
                    for row in range(0, self.chunk, self.sequence_rows)
                )
            )
        return (
            tpu.make_async_copy(
                self.history_hbm.at[rows],
                self.history_buffers.at[slot],
                self.read_semaphores.at[slot, 0],
            ),
            tpu.make_async_copy(
                self.state_hbm.at[rows],
                self.state_buffers.at[slot],
                self.read_semaphores.at[slot, 1],
            ),
        )

    def _writes(self, index: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for writes."""
        rows, slot = (pl.ds(index * self.chunk, self.chunk), index % self.buffer_count)
        return (
            tpu.make_async_copy(
                self.history_buffers.at[slot],
                self.history_output_hbm.at[rows],
                self.write_semaphores.at[slot, 0],
            ),
            tpu.make_async_copy(
                self.state_buffers.at[slot],
                self.state_output_hbm.at[rows],
                self.write_semaphores.at[slot, 1],
            ),
        )

    def prime(self) -> None:
        """Compute and stage Pallas VMEM state for prime."""
        for index in range(min(2, self.chunks)):
            for copy in self._reads(index):
                copy.start()

    def acquire(self, index: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for acquire."""
        if self.four_buffers and index == 0:
            for tail in range(2, self.chunks):
                for copy in self._reads(tail):
                    copy.start()
        for copy in self._reads(index):
            copy.wait()
        slot = index % self.buffer_count
        return (self.history_buffers.at[slot], self.state_buffers.at[slot])

    def release(self, index: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for release."""
        for copy in self._writes(index):
            copy.start()
        self.pending_writes.append(index)
        if self.four_buffers:
            return
        ahead = index + 2
        if ahead < self.chunks:
            if ahead >= _DCP_HEAD_COUNT:
                for copy in self._writes(ahead - 3):
                    copy.wait()
                self.pending_writes.remove(ahead - 3)
            for copy in self._reads(ahead):
                copy.start()

    def finish(self) -> None:
        """Compute and stage Pallas VMEM state for finish."""
        for index in list(self.pending_writes):
            for copy in self._writes(index):
                copy.wait()
        self.pending_writes.clear()

    def acquire_rolled(self, index: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for acquire rolled."""
        for copy in self._reads(index):
            copy.wait()
        slot = index % 3
        return (self.history_buffers.at[slot], self.state_buffers.at[slot])

    def release_rolled(self, index: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for release rolled."""
        for copy in self._writes(index):
            copy.start()
        ahead = index + 2

        @pl.when(ahead < self.chunks)
        def refill() -> None:
            """Compute and stage Pallas VMEM state for refill."""

            @pl.when(ahead >= _DCP_HEAD_COUNT)
            def drain() -> None:
                """Compute and stage Pallas VMEM state for drain."""
                for copy in self._writes(ahead - 3):
                    copy.wait()

            for copy in self._reads(ahead):
                copy.start()

    def finish_rolled(self) -> None:
        """Compute and stage Pallas VMEM state for finish rolled."""
        for index in range(max(0, self.chunks - 3), self.chunks):
            for copy in self._writes(index):
                copy.wait()


class KdaStateStreamSlice:
    """One head shard's rows in a shared, flattened KDA state stream.

    The caller primes the parent once and drains it after all head
    shards. Local KDA calls use shard-local chunk indices and finish
    calls, while the parent keeps its lookahead and buffer ownership
    across shard boundaries.
    """

    def __init__(self, parent: TensorLike, *, first_chunk: int, chunks: int) -> None:
        """Initialize instance state."""
        np.testing.assert_(
            not (
                isinstance(first_chunk, int)
                and (
                    first_chunk < 0
                    or chunks < 1
                    or first_chunk + chunks > parent.chunks
                )
            ),
            "A state-stream slice must fit within its parent",
        )
        self.parent = parent
        self.first_chunk, self.chunks = (first_chunk, chunks)
        self.chunk, self.slots = (parent.chunk, parent.slots)

    def acquire(self, index: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for acquire."""
        with _phase("t1_kda_state_read_wait"):
            return self.parent.acquire(self.first_chunk + index)

    def release(self, index: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for release."""
        self.parent.release(self.first_chunk + index)

    def acquire_rolled(self, index: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for acquire rolled."""
        with _phase("t1_kda_state_read_wait"):
            return self.parent.acquire_rolled(self.first_chunk + index)

    def release_rolled(self, index: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for release rolled."""
        self.parent.release_rolled(self.first_chunk + index)

    def finish(self) -> None:
        """Compute and stage Pallas VMEM state for finish."""

    def finish_rolled(self) -> None:
        """Compute and stage Pallas VMEM state for finish rolled."""


def _shard_dot(
    stream: TensorLike, x: TensorLike, shard: TensorLike, guard: TensorLike
) -> TensorLike:
    """Compute and stage Pallas VMEM state for shard dot.

    ``x`` times one held shard's columns of a column stream (FP32):
    the shard's chunks in order, concatenated.
    """
    per_shard = stream.chunks // stream.shards
    parts = [
        stream.dot_position(x, shard * per_shard + index, guard)
        for index in range(per_shard)
    ]
    return parts[0] if per_shard == 1 else jnp.concatenate(parts, axis=1)


def _traced_dot_position(
    stream: TensorLike,
    x_ref: TensorLike,
    position: TensorLike,
    guard: TensorLike,
    out_ref: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for traced dot position.

    ``dot_position`` at a traced ``position`` of a ring column stream
    with equal chunks: wait, multiply ``x_ref`` by the chunk into
    ``out_ref`` (BF16), refill the slot ``depth`` positions ahead.
    """
    np.testing.assert_(
        not (
            stream.slots is None
            or stream.order is not None
            or len(set(stream.sizes)) != 1
        ),
        "A traced stream position needs a ring of equal chunks",
    )
    size = stream.sizes[0]
    slot = position % stream.depth

    def copy(
        at: TensorLike, destination: TensorLike, semaphore: TensorLike
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for copy."""
        source = stream.source.at[:, pl.ds(pl.multiple_of(at * size, 128), size)]
        return tpu.make_async_copy(source, destination, semaphore)

    for index, destination in enumerate(stream.slots):

        @pl.when(slot == index)
        def consume(
            destination: TensorLike = destination,
            semaphore: TensorLike = stream.semaphores.at[stream.base + index],
        ) -> None:
            """Compute and stage Pallas VMEM state for consume."""
            copy(position, destination, semaphore).wait()
            out_ref[...] = _dot(x_ref[...], destination[...]).astype(jnp.bfloat16)

            @pl.when(guard & (position + stream.depth < stream.chunks))
            def refill() -> None:
                """Compute and stage Pallas VMEM state for refill."""
                copy(position + stream.depth, destination, semaphore).start()


class _LazyProjectionCopies:
    """Lazily instantiate projection DMA descriptors on first attribute access."""

    def __init__(
        self,
        specs: dict[str, tuple[TensorLike, TensorLike, TensorLike]],
    ) -> None:
        """Store pre-sliced ``(src_ref, dst_ref, sem_ref)`` tuples per projection."""
        self._specs = specs
        self._cache: dict[str, TensorLike] = {}

    def __getattr__(self, name: str) -> TensorLike:
        """Create and cache the async copy descriptor for ``name``."""
        if name not in self._cache:
            if name not in self._specs:
                raise AttributeError(name)
            src_ref, dst_ref, sem_ref = self._specs[name]
            self._cache[name] = tpu.make_async_copy(src_ref, dst_ref, sem_ref)
        return self._cache[name]


@dataclasses.dataclass(frozen=True, slots=True)
class KimiDeltaAttentionWeights:
    """KDA parameters and their persistent projection-copy resources."""

    gate_projection: TensorLike
    gate_projection_vmem: TensorLike
    input_projection: TensorLike
    input_projection_vmem: TensorLike
    feature_gate_projection: TensorLike
    feature_gate_projection_vmem: TensorLike
    output_projection: TensorLike
    output_projection_vmem: TensorLike
    convolution_filter: TensorLike
    exp_log_decay_rate: TensorLike
    time_step_bias: TensorLike
    output_normalization: TensorLike
    projection_copy_semaphores: TensorLike

    def __post_init__(self) -> None:
        """Register PyTree dataclasses lazily on first instantiation."""
        _ensure_dataclasses_registered()

    def make_projection_copies(self) -> _LazyProjectionCopies:
        """Create the asynchronous projection-weight copies on first use."""
        return _LazyProjectionCopies(
            {
                "gate": (
                    self.gate_projection,
                    self.gate_projection_vmem,
                    self.projection_copy_semaphores.at[0],
                ),
                "feature_gate": (
                    self.feature_gate_projection,
                    self.feature_gate_projection_vmem,
                    self.projection_copy_semaphores.at[1],
                ),
                "input": (
                    self.input_projection,
                    self.input_projection_vmem,
                    self.projection_copy_semaphores.at[2],
                ),
                "output": (
                    self.output_projection,
                    self.output_projection_vmem,
                    self.projection_copy_semaphores.at[3],
                ),
            }
        )


def _kda_row_chunk(batch_size: int, attention_dp: int = 1) -> int:
    """Compute and stage Pallas VMEM state for kda row chunk.

    Rows per KDA recurrence chunk above B8 (smaller batches run
    whole). TP32 B128 and above use four: eight-row chunks exceed the
    compiler's VMEM although _vmem_budget accepts them. Attention DP's
    B128 peaks outside KDA and fits eight.
    """
    return (
        8
        if batch_size < _VREG_LANE_COUNT
        or (attention_dp > 1 and batch_size == _VREG_LANE_COUNT)
        else 4
    )


def kda(
    hidden_ref: TensorLike,
    output_ref: TensorLike,
    updated_convolution_history_ref: TensorLike,
    updated_recurrent_state_ref: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Execute one Kimi Delta Attention layer inside a Pallas program.

    ``state_stream`` (above eight rows; see KdaStateStream) supplies
    each recurrence chunk's state from HBM instead of whole-batch
    state refs.

    ``streams`` optionally maps "input" and "output" to
    ChunkedWeightStreams that own the copies of those projections.

    Attention DP runs one call per held TP32 shard (``shard_index``);
    ``attended_out`` ([batch, 384]) receives its gated features for
    the caller's output projection, and ``gate_features_vmem`` (at
    least [batch, 128] BF16) shares the gate features between shards.
    ``stage_projections`` stages a DP shard's projections above eight
    rows (rolled over a state stream's tiles). ``projected_rows``
    ([batch, 1664] BF16) is the caller's input projection for the DP2
    B2/B4 shard swap, which passes a traced ``shard_index`` and
    ``first_call``.
    """
    (
        weights,
        convolution_history_ref,
        recurrent_state_ref,
        communication_workspace,
        epsilon,
        decay_lower_bound,
        synchronize_devices,
        weights_are_prefetched,
        token_scatter_output,
        sequence_rows,
        wait_for_state,
        on_state_updated,
        streams,
        shard_index,
        attended_out,
        gate_features_vmem,
        state_stream,
        stage_projections,
        projected_rows,
        first_call,
    ) = (
        stage_kwargs["weights"],
        stage_kwargs["convolution_history_ref"],
        stage_kwargs["recurrent_state_ref"],
        stage_kwargs["communication_workspace"],
        stage_kwargs.get("epsilon", 1e-05),
        stage_kwargs.get("decay_lower_bound", -5.0),
        stage_kwargs.get("synchronize_devices", True),
        stage_kwargs.get("weights_are_prefetched", False),
        stage_kwargs.get("token_scatter_output", False),
        stage_kwargs.get("sequence_rows", 1),
        stage_kwargs.get("wait_for_state", _NONE_STAGE_VAL),
        stage_kwargs.get("on_state_updated", _NONE_STAGE_VAL),
        stage_kwargs.get("streams", _NONE_STAGE_VAL),
        stage_kwargs.get("shard_index", _NONE_STAGE_VAL),
        stage_kwargs.get("attended_out", _NONE_STAGE_VAL),
        stage_kwargs.get("gate_features_vmem", _NONE_STAGE_VAL),
        stage_kwargs.get("state_stream", _NONE_STAGE_VAL),
        stage_kwargs.get("stage_projections", False),
        stage_kwargs.get("projected_rows", _NONE_STAGE_VAL),
        stage_kwargs.get("first_call", _NONE_STAGE_VAL),
    )
    streams = streams or {}
    first_shard = shard_index in (None, 0) if first_call is None else first_call
    traced_shard = shard_index is not None and (not isinstance(shard_index, int))
    np.testing.assert_(
        (shard_index is None) == (attended_out is None),
        "Per-shard KDA calls must take attended_out",
    )
    batch_size, hidden_size = hidden_ref.shape
    head_count, channels_per_head = (3, 128)
    state_batch = (
        convolution_history_ref.shape[0]
        if state_stream is not None and state_stream.slots is not None
        else batch_size
    )
    _kda_stage_au(
        (
            (channels_per_head, convolution_history_ref, head_count, hidden_ref),
            (hidden_size, output_ref, recurrent_state_ref, shard_index),
            (state_batch, weights),
        )
    )
    (chunk_count, projection_width, rolled, row_chunk, staged) = _kda_stage_at(
        (
            (batch_size, communication_workspace, sequence_rows, shard_index),
            (stage_projections, state_stream, synchronize_devices, weights),
        )
    )
    _kda_stage_av(
        (
            (attended_out, batch_size, channels_per_head, chunk_count),
            (communication_workspace, convolution_history_ref, decay_lower_bound),
            (epsilon, first_shard, gate_features_vmem, head_count, hidden_ref),
            (on_state_updated, output_ref, projected_rows, projection_width),
            (recurrent_state_ref, rolled, row_chunk, sequence_rows, shard_index),
            (staged, state_stream, streams, token_scatter_output, traced_shard),
            (updated_convolution_history_ref, updated_recurrent_state_ref),
            (wait_for_state, weights, weights_are_prefetched),
        )
    )
    if state_stream is not None and (not rolled) and (batch_size == _KDA_GROUP_WIDTH):
        state_stream.finish()


def _kda_stage_au(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for kda stage au."""
    (
        (channels_per_head, convolution_history_ref, head_count, hidden_ref),
        (hidden_size, output_ref, recurrent_state_ref, shard_index),
        (state_batch, weights),
    ) = stage_ctx
    state_shape = (state_batch, head_count, channels_per_head, channels_per_head)
    history_shape = (state_batch, 3, 3, CONV_HEAD_ROWS, channels_per_head)
    np.testing.assert_(
        not (
            output_ref.shape != hidden_ref.shape
            or weights.gate_projection.shape != (hidden_size, 128)
            or weights.input_projection.shape[0] != hidden_size
            or weights.input_projection.shape[1] % 1664
            or (
                shard_index is None
                and weights.input_projection.shape[1] != _GATE_PROJ_WIDTH
            )
            or (weights.exp_log_decay_rate.shape != (3, 1))
            or (convolution_history_ref.shape != history_shape)
            or (recurrent_state_ref.shape != state_shape)
        ),
        "Kimi Delta Attention inputs and state must match the input batch",
    )


def _kda_stage_at(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for kda stage at."""
    (
        (batch_size, communication_workspace, sequence_rows, shard_index),
        (stage_projections, state_stream, synchronize_devices, weights),
    ) = stage_ctx
    np.testing.assert_(
        not batch_size % sequence_rows, "sequence_rows must divide the KDA batch"
    )
    if synchronize_devices:
        collectives32.barrier()
        communication_workspace.reduction_phase[0] = 0
    staged = (
        (stage_projections or (KDA_STAGED and shard_index is None))
        and batch_size > _ROUTED_EXPERTS_PER_TOKEN
        and (sequence_rows == 1)
    )
    projection_width = (
        1664 if shard_index is not None else weights.input_projection.shape[1]
    )
    rolled = staged and state_stream is not None
    row_chunk = (
        state_stream.chunk if state_stream is not None else _kda_row_chunk(batch_size)
    )
    chunk_count = batch_size // row_chunk
    return (chunk_count, projection_width, rolled, row_chunk, staged)


def _kda_stage_av(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for kda stage av."""
    (
        (attended_out, batch_size, channels_per_head, chunk_count),
        (communication_workspace, convolution_history_ref, decay_lower_bound),
        (epsilon, first_shard, gate_features_vmem, head_count, hidden_ref),
        (on_state_updated, output_ref, projected_rows, projection_width),
        (recurrent_state_ref, rolled, row_chunk, sequence_rows, shard_index),
        (staged, state_stream, streams, token_scatter_output, traced_shard),
        (updated_convolution_history_ref, updated_recurrent_state_ref),
        (wait_for_state, weights, weights_are_prefetched),
    ) = stage_ctx

    def run(
        reduction_vmem: TensorLike,
        attended_vmem: TensorLike = None,
        projected_vmem: TensorLike = None,
        gate_vmem: TensorLike = None,
        output_gate_vmem: TensorLike = None,
    ) -> None:
        """Compute and stage Pallas VMEM state for run."""
        return _kda_run_stage_as(
            (
                (attended_out, batch_size, channels_per_head, chunk_count),
                (communication_workspace, convolution_history_ref, decay_lower_bound),
                (epsilon, first_shard, gate_features_vmem, head_count, hidden_ref),
                (on_state_updated, output_ref, projected_rows, recurrent_state_ref),
                (rolled, row_chunk, sequence_rows, shard_index, staged, state_stream),
                (streams, token_scatter_output, traced_shard),
                (updated_convolution_history_ref, updated_recurrent_state_ref),
                (wait_for_state, weights, weights_are_prefetched),
            ),
            (
                (reduction_vmem, attended_vmem, projected_vmem, gate_vmem),
                output_gate_vmem,
            ),
        )

    pl.run_scoped(
        run,
        tpu.VMEM(
            (batch_size, 3584 if batch_size > _KDA_GROUP_WIDTH else 8192), jnp.float32
        ),
        *(
            (tpu.VMEM((batch_size, head_count, channels_per_head), jnp.float32),)
            if batch_size > _ROUTED_EXPERTS_PER_TOKEN
            and (sequence_rows == 1 or state_stream is not None)
            else ()
        ),
        *(
            (
                tpu.VMEM((chunk_count, row_chunk, projection_width), jnp.bfloat16),
                tpu.VMEM(
                    (chunk_count, row_chunk, head_count * channels_per_head),
                    jnp.bfloat16,
                ),
                tpu.VMEM((batch_size, head_count * channels_per_head), jnp.bfloat16),
            )
            if rolled
            else (
                tpu.VMEM((batch_size, projection_width), jnp.bfloat16),
                tpu.VMEM((batch_size, head_count * channels_per_head), jnp.bfloat16),
            )
            if staged
            else ()
        ),
    )


def _kda_run_stage_as_stage_az(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for kda run stage as."""
    (
        (channels_per_head, first_shard, gate_features_vmem, head_count),
        (hidden_ref, streams, weights, weights_are_prefetched),
    ) = stage_ctx
    kda_scope = [_phase_start("q3_kda_projections")]
    weight_copies = weights.make_projection_copies()
    if not weights_are_prefetched:
        weight_copies.gate.start()
        weight_copies.feature_gate.start()
        for name in ("input", "output"):
            if name in streams:
                streams[name].start()
            else:
                getattr(weight_copies, name).start()
    attention_width = head_count * channels_per_head
    if first_shard:
        weight_copies.gate.wait()
    if first_shard or gate_features_vmem is None:
        gate_features = _dot(hidden_ref[...], weights.gate_projection_vmem[...]).astype(
            jnp.bfloat16
        )
        if gate_features_vmem is not None:
            gate_features_vmem[:, :128] = gate_features
    else:
        gate_features = gate_features_vmem[:, :128]
    if first_shard:
        weight_copies.feature_gate.wait()
    feature_gate_vmem = weights.feature_gate_projection_vmem
    return (
        (attention_width, feature_gate_vmem, gate_features, kda_scope),
        weight_copies,
    )


def _kda_run_stage_as_stage_bc(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for kda run stage as."""
    (
        (attention_width, batch_size, channels_per_head, chunk_count),
        (communication_workspace, feature_gate_vmem, gate_features, gate_vmem),
        (head_count, hidden_ref, projected_rows, raw_recurrent_gate, rolled),
        (row_chunk, shard_index, staged, streams, traced_shard, weight_copies),
        weights,
    ) = stage_ctx
    if shard_index is not None and (not traced_shard):
        feature_gate_vmem = feature_gate_vmem.at[
            :, pl.ds(shard_index * attention_width, attention_width)
        ]
    raw_recurrent_gate = _kda_run_stage_as_stage_bb(
        (
            (attention_width, batch_size, channels_per_head, chunk_count),
            (feature_gate_vmem, gate_features, gate_vmem, head_count),
            (raw_recurrent_gate, rolled, row_chunk, shard_index, staged),
            traced_shard,
        )
    )
    if projected_rows is not None:
        projected = projected_rows
    elif shard_index is not None:
        projected = _shard_dot(
            streams["input"],
            hidden_ref[...],
            shard_index,
            communication_workspace.reduction_phase[0] >= 0,
        ).astype(jnp.bfloat16)
    elif "input" in streams:
        projected = (
            streams["input"]
            .dot(hidden_ref[...], communication_workspace.reduction_phase[0] >= 0)
            .astype(jnp.bfloat16)
        )
    else:
        weight_copies.input.wait()
        projected = _dot(hidden_ref[...], weights.input_projection_vmem[...]).astype(
            jnp.bfloat16
        )
    return (projected, raw_recurrent_gate)


def _kda_run_stage_as_stage_bb(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for kda run stage as."""
    (
        (attention_width, batch_size, channels_per_head, chunk_count),
        (feature_gate_vmem, gate_features, gate_vmem, head_count),
        (raw_recurrent_gate, rolled, row_chunk, shard_index, staged),
        traced_shard,
    ) = stage_ctx
    if rolled and traced_shard and (feature_gate_vmem.shape[1] > 2 * attention_width):
        raw_gate = _dot(gate_features, feature_gate_vmem[...])
        selected = raw_gate[:, :attention_width]
        for held in range(1, feature_gate_vmem.shape[1] // attention_width):
            selected = jnp.where(
                shard_index == held,
                raw_gate[:, held * attention_width : (held + 1) * attention_width],
                selected,
            )
        gate_vmem[...] = selected.astype(jnp.bfloat16).reshape(
            chunk_count, row_chunk, -1
        )
    elif rolled:
        gate_vmem[...] = (
            _dot(gate_features, feature_gate_vmem[...])
            .astype(jnp.bfloat16)
            .reshape(chunk_count, row_chunk, -1)
        )
    elif staged:
        gate_vmem[...] = _dot(gate_features, feature_gate_vmem[...]).astype(
            jnp.bfloat16
        )
    else:
        raw_recurrent_gate = _dot(gate_features, feature_gate_vmem[...])
        if traced_shard:
            raw_recurrent_gate = jnp.where(
                shard_index == 0,
                raw_recurrent_gate[:, :attention_width],
                raw_recurrent_gate[:, attention_width:],
            )
        raw_recurrent_gate = (
            raw_recurrent_gate.astype(jnp.bfloat16)
            .reshape(batch_size, head_count, channels_per_head)
            .astype(jnp.float32)
        )
    return raw_recurrent_gate


def _kda_run_stage_as_stage_ba(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for kda run stage as."""
    (
        (attention_width, batch_size, channels_per_head, chunk_count),
        (decay_lower_bound, gate_vmem, head_count, output_gate),
        (output_gate_vmem, projected, projected_vmem, raw_recurrent_gate),
        (rolled, row_chunk, staged, wait_for_state, weights),
    ) = stage_ctx
    if rolled:
        projected_vmem[...] = projected.reshape(chunk_count, row_chunk, -1)
        output_gate_vmem[...] = projected[:, 3 * attention_width : 4 * attention_width]
        queries_keys_values = _RowFunction(
            lambda index: projected_vmem[index][:, : 3 * attention_width].reshape(
                row_chunk, 3, head_count, channels_per_head
            )
        )
        update_rate = _RowFunction(
            lambda index: projected_vmem[index][
                :, 4 * attention_width : 4 * attention_width + head_count
            ]
            .reshape(row_chunk, head_count, 1)
            .astype(jnp.float32)
        )
        decay = _RowFunction(
            lambda index: decay_lower_bound
            * jax.nn.sigmoid(
                jnp.exp(weights.exp_log_decay_rate[...])[None]
                * (
                    gate_vmem[index]
                    .reshape(row_chunk, head_count, channels_per_head)
                    .astype(jnp.float32)
                    + weights.time_step_bias[...][None]
                )
            )
        )
        if wait_for_state is not None:
            wait_for_state()
    elif staged:
        projected_vmem[...] = projected

        def chunk_rows(rows: TensorLike) -> TensorLike:
            """Compute and stage Pallas VMEM state for chunk rows."""
            return rows.stop - rows.start

        queries_keys_values = _RowFunction(
            lambda rows: projected_vmem[rows, : 3 * attention_width].reshape(
                chunk_rows(rows), 3, head_count, channels_per_head
            )
        )
        update_rate = _RowFunction(
            lambda rows: projected_vmem[
                rows, 4 * attention_width : 4 * attention_width + head_count
            ]
            .reshape(chunk_rows(rows), head_count, 1)
            .astype(jnp.float32)
        )
        decay = _RowFunction(
            lambda rows: decay_lower_bound
            * jax.nn.sigmoid(
                jnp.exp(weights.exp_log_decay_rate[...])[None]
                * (
                    gate_vmem[rows]
                    .reshape(chunk_rows(rows), head_count, channels_per_head)
                    .astype(jnp.float32)
                    + weights.time_step_bias[...][None]
                )
            )
        )
        if wait_for_state is not None:
            wait_for_state()
    else:
        queries_keys_values = projected[:, : 3 * attention_width].reshape(
            batch_size, 3, head_count, channels_per_head
        )
        projected_tail = projected[:, 3 * attention_width :]
        output_gate = projected_tail[:, :attention_width].reshape(
            batch_size, head_count, channels_per_head
        )
        update_rate = (
            projected_tail[:, attention_width : attention_width + head_count]
            .reshape(batch_size, head_count, 1)
            .astype(jnp.float32)
        )
        if wait_for_state is not None:
            wait_for_state()
        decay = decay_lower_bound * jax.nn.sigmoid(
            jnp.exp(weights.exp_log_decay_rate[...])[None]
            * (raw_recurrent_gate + weights.time_step_bias[...][None])
        )
    return (decay, output_gate, queries_keys_values, update_rate)


def _kda_run_stage_as(stage_ctx: TensorLike, stage_args: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for run."""
    (
        (attended_out, batch_size, channels_per_head, chunk_count),
        (communication_workspace, convolution_history_ref, decay_lower_bound),
        (epsilon, first_shard, gate_features_vmem, head_count, hidden_ref),
        (on_state_updated, output_ref, projected_rows, recurrent_state_ref),
        (rolled, row_chunk, sequence_rows, shard_index, staged, state_stream),
        (streams, token_scatter_output, traced_shard),
        (updated_convolution_history_ref, updated_recurrent_state_ref),
        (wait_for_state, weights, weights_are_prefetched),
    ) = stage_ctx
    ((reduction_vmem, attended_vmem, projected_vmem, gate_vmem), output_gate_vmem) = (
        stage_args
    )
    chunk = history_chunk = state_chunk = _NONE_STAGE_VAL
    output_gate = raw_recurrent_gate = _NONE_STAGE_VAL
    (attention_width, feature_gate_vmem, gate_features, kda_scope), weight_copies = (
        _kda_run_stage_as_stage_az(
            (
                (channels_per_head, first_shard, gate_features_vmem, head_count),
                (hidden_ref, streams, weights, weights_are_prefetched),
            )
        )
    )
    projected, raw_recurrent_gate = _kda_run_stage_as_stage_bc(
        (
            (attention_width, batch_size, channels_per_head, chunk_count),
            (communication_workspace, feature_gate_vmem, gate_features, gate_vmem),
            (head_count, hidden_ref, projected_rows, raw_recurrent_gate, rolled),
            (row_chunk, shard_index, staged, streams, traced_shard, weight_copies),
            weights,
        )
    )
    (decay, output_gate, queries_keys_values, update_rate) = _kda_run_stage_as_stage_ba(
        (
            (attention_width, batch_size, channels_per_head, chunk_count),
            (decay_lower_bound, gate_vmem, head_count, output_gate),
            (output_gate_vmem, projected, projected_vmem, raw_recurrent_gate),
            (rolled, row_chunk, staged, wait_for_state, weights),
        )
    )
    if sequence_rows == 1:
        (attended, chunk, history_chunk, _recurrence, state_chunk) = (
            _kda_run_stage_as_stage_ax(
                (
                    (attended_vmem, batch_size, channels_per_head, chunk, chunk_count),
                    (
                        communication_workspace,
                        convolution_history_ref,
                        decay,
                        head_count,
                    ),
                    (
                        history_chunk,
                        kda_scope,
                        queries_keys_values,
                        recurrent_state_ref,
                    ),
                    (rolled, row_chunk, state_chunk, state_stream, update_rate),
                    (updated_convolution_history_ref, updated_recurrent_state_ref),
                    weights,
                )
            )
        )
    else:
        (_convolution_filter, run_sequence) = _kda_run_stage_as_stage_ay(
            (
                (channels_per_head, decay, queries_keys_values, recurrent_state_ref),
                (sequence_rows, update_rate, weights),
            )
        )
        if state_stream is None:
            attended = _kda_run_stage_as_stage_bg(
                (
                    (
                        batch_size,
                        convolution_history_ref,
                        head_count,
                        recurrent_state_ref,
                    ),
                    (run_sequence, sequence_rows, updated_convolution_history_ref),
                    updated_recurrent_state_ref,
                )
            )
        else:
            chunk = state_stream.chunk
            np.testing.assert_(
                not chunk % sequence_rows,
                "Streamed KDA chunks must hold whole sequences",
            )
            for index, base in enumerate(range(0, batch_size, chunk)):

                @pl.when(communication_workspace.reduction_phase[0] >= 0)
                def run_sequence_chunk(
                    index: TensorLike = index, base: TensorLike = base
                ) -> None:
                    """Compute and stage Pallas run sequence chunk."""
                    history_chunk, state_chunk = state_stream.acquire(index)
                    rows = []
                    for local in range(0, chunk, sequence_rows):

                        def store(
                            offset: TensorLike,
                            history: TensorLike,
                            state: TensorLike,
                            local: TensorLike = local,
                        ) -> None:
                            """Compute and stage Pallas VMEM state for store."""
                            history_chunk[local + offset, :, :, :head_count] = history
                            state_chunk[local + offset] = state

                        rows += run_sequence(
                            base + local,
                            history_chunk[local, :, :, :head_count],
                            state_chunk[local],
                            store,
                        )
                    attended_vmem[base : base + chunk] = jnp.concatenate(rows, axis=0)
                    state_stream.release(index)

            if batch_size != _KDA_GROUP_WIDTH:
                state_stream.finish()
            attended = attended_vmem[...]
    attended = _kda_run_stage_as_stage_bd(
        (
            (attended, attention_width, batch_size, channels_per_head, epsilon),
            (head_count, kda_scope, on_state_updated, output_gate),
            (output_gate_vmem, projected_vmem, rolled, staged, weights),
        )
    )
    if attended_out is not None:
        attended_out[...] = attended.reshape(batch_size, attention_width)
        return
    _kda_run_stage_as_stage_be(
        (
            (attended, attention_width, batch_size, communication_workspace),
            (output_ref, reduction_vmem, streams, token_scatter_output),
            (weight_copies, weights),
        )
    )


def _kda_run_stage_as_stage_ax(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for kda run stage as."""
    (
        (attended_vmem, batch_size, channels_per_head, chunk, chunk_count),
        (communication_workspace, convolution_history_ref, decay, head_count),
        (history_chunk, kda_scope, queries_keys_values, recurrent_state_ref),
        (rolled, row_chunk, state_chunk, state_stream, update_rate),
        (updated_convolution_history_ref, updated_recurrent_state_ref),
        weights,
    ) = stage_ctx

    def recurrence(
        rows: TensorLike, history_ref: TensorLike = None, state_ref: TensorLike = None
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for recurrence."""
        convolution_history = (
            convolution_history_ref[rows, :, :, :head_count]
            if history_ref is None
            else history_ref[:, :, :, :head_count]
        )
        recurrent_state = (
            recurrent_state_ref[rows] if state_ref is None else state_ref[...]
        ).astype(jnp.float32)
        convolution_window = jnp.concatenate(
            (convolution_history, queries_keys_values[rows][:, None]), axis=1
        )
        filtered_queries_keys_values = jax.nn.silu(
            jnp.sum(
                convolution_window.astype(jnp.float32)
                * weights.convolution_filter[...].astype(jnp.float32)[None],
                axis=1,
            )
        ).astype(jnp.bfloat16)
        query, key, value = (
            filtered_queries_keys_values[:, 0].astype(jnp.float32),
            filtered_queries_keys_values[:, 1].astype(jnp.float32),
            filtered_queries_keys_values[:, 2].astype(jnp.float32),
        )
        query *= jax.lax.rsqrt(
            jnp.sum(query * query, axis=-1, keepdims=True) + 1e-06
        ) * channels_per_head ** (-0.5)
        key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-06)
        next_recurrent_state = recurrent_state * jnp.exp(decay[rows])[..., None]
        predicted_value = jnp.sum(key[..., None] * next_recurrent_state, axis=2)
        state_update = jax.nn.sigmoid(update_rate[rows]) * (value - predicted_value)
        next_recurrent_state += key[..., None] * state_update[..., None, :]
        attended_rows = jnp.sum(query[..., None] * next_recurrent_state, axis=2)
        if history_ref is None:
            updated_convolution_history_ref[rows, :, :, :head_count] = (
                convolution_window[:, 1:]
            )
            updated_recurrent_state_ref[rows] = next_recurrent_state.astype(
                updated_recurrent_state_ref.dtype
            )
        else:
            history_ref[:, :, :, :head_count] = convolution_window[:, 1:]
            state_ref[...] = next_recurrent_state.astype(state_ref.dtype)
        return attended_rows

    _phase_end(kda_scope.pop())
    kda_scope.append(_phase_start("q4_kda_recurrence"))
    if batch_size <= _ROUTED_EXPERTS_PER_TOKEN:
        if state_stream is None:
            attended = recurrence(slice(None))
        else:
            np.testing.assert_(
                state_stream.chunks == 1 and state_stream.chunk == batch_size
            )
            history_chunk, state_chunk = state_stream.acquire(0)
            attended = recurrence(slice(None), history_chunk, state_chunk)
            state_stream.release(0)
            state_stream.finish()
    else:
        attended, chunk = _kda_run_stage_as_stage_aw(
            (
                (attended_vmem, batch_size, chunk_count, communication_workspace),
                (recurrence, rolled, row_chunk, state_stream),
            )
        )
    return (attended, chunk, history_chunk, recurrence, state_chunk)


def _kda_run_stage_as_stage_aw(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for kda run stage as."""
    (
        (attended_vmem, batch_size, chunk_count, communication_workspace),
        (recurrence, rolled, row_chunk, state_stream),
    ) = stage_ctx
    chunk = row_chunk
    if rolled:

        @pl.loop(0, chunk_count)
        def run_rolled_chunk(index: TensorLike) -> None:
            """Compute and stage Pallas run rolled chunk."""
            history_chunk, state_chunk = state_stream.acquire_rolled(index)
            attended_vmem[pl.ds(pl.multiple_of(index * chunk, chunk), chunk)] = (
                recurrence(index, history_chunk, state_chunk)
            )
            state_stream.release_rolled(index)

        state_stream.finish_rolled()
    for index, first in () if rolled else enumerate(range(0, batch_size, chunk)):

        @pl.when(communication_workspace.reduction_phase[0] >= 0)
        def run_chunk(index: TensorLike = index, first: TensorLike = first) -> None:
            """Compute and stage Pallas VMEM state for run chunk."""
            if state_stream is None:
                attended_vmem[first : first + chunk] = recurrence(
                    slice(first, first + chunk)
                )
            else:
                history_chunk, state_chunk = state_stream.acquire(index)
                attended_vmem[first : first + chunk] = recurrence(
                    slice(first, first + chunk), history_chunk, state_chunk
                )
                state_stream.release(index)

    if state_stream is not None and (not rolled) and (batch_size != _KDA_GROUP_WIDTH):
        state_stream.finish()
    attended = attended_vmem[...]
    return (attended, chunk)


def _kda_run_stage_as_stage_ay(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for kda run stage as."""
    (
        (channels_per_head, decay, queries_keys_values, recurrent_state_ref),
        (sequence_rows, update_rate, weights),
    ) = stage_ctx
    convolution_filter = weights.convolution_filter[...].astype(jnp.float32)

    def run_sequence(
        first: TensorLike, history: TensorLike, state: TensorLike, store: TensorLike
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for run sequence.

        The rows of the sequence at ``first`` from its input conv
        ``history`` and recurrent ``state``; ``store(offset, history,
        state)`` writes row ``first + offset``'s post-states.
        """
        window = jnp.concatenate(
            (history, queries_keys_values[first : first + sequence_rows]), axis=0
        )
        state = state.astype(jnp.float32)
        rows = []
        for offset in range(sequence_rows):
            row = first + offset
            filtered = jax.nn.silu(
                jnp.sum(
                    window[offset : offset + 4].astype(jnp.float32)
                    * convolution_filter,
                    axis=0,
                )
            ).astype(jnp.bfloat16)
            query = filtered[0].astype(jnp.float32)
            key = filtered[1].astype(jnp.float32)
            value = filtered[2].astype(jnp.float32)
            query *= jax.lax.rsqrt(
                jnp.sum(query * query, axis=-1, keepdims=True) + 1e-06
            ) * channels_per_head ** (-0.5)
            key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-06)
            state = state * jnp.exp(decay[row])[..., None]
            predicted_value = jnp.sum(key[..., None] * state, axis=1)
            state_update = jax.nn.sigmoid(update_rate[row]) * (value - predicted_value)
            state = state + key[..., None] * state_update[:, None, :]
            rows.append(jnp.sum(query[..., None] * state, axis=1)[None])
            saved_state = state.astype(recurrent_state_ref.dtype)
            store(offset, window[offset + 1 : offset + 4], saved_state)
            state = saved_state.astype(jnp.float32)
        return rows

    return (convolution_filter, run_sequence)


def _kda_run_stage_as_stage_bg(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for kda run stage as."""
    (
        (batch_size, convolution_history_ref, head_count, recurrent_state_ref),
        (run_sequence, sequence_rows, updated_convolution_history_ref),
        updated_recurrent_state_ref,
    ) = stage_ctx
    attended_rows = []
    for first in range(0, batch_size, sequence_rows):

        def store(
            offset: TensorLike,
            history: TensorLike,
            state: TensorLike,
            first: TensorLike = first,
        ) -> None:
            """Compute and stage Pallas VMEM state for store."""
            updated_convolution_history_ref[first + offset, :, :, :head_count] = history
            updated_recurrent_state_ref[first + offset] = state

        attended_rows += run_sequence(
            first,
            convolution_history_ref[first, :, :, :head_count],
            recurrent_state_ref[first],
            store,
        )
    return jnp.concatenate(attended_rows, axis=0)


def _kda_run_stage_as_stage_bd(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for kda run stage as."""
    (
        (attended, attention_width, batch_size, channels_per_head, epsilon),
        (head_count, kda_scope, on_state_updated, output_gate),
        (output_gate_vmem, projected_vmem, rolled, staged, weights),
    ) = stage_ctx
    _phase_end(kda_scope.pop())
    kda_scope.append(_phase_start("q5_kda_output_gate"))
    attended = attended.astype(jnp.bfloat16).astype(jnp.float32)
    attended *= jax.lax.rsqrt(
        jnp.mean(attended * attended, axis=-1, keepdims=True) + epsilon
    )
    normalization = (
        weights.output_normalization[...]
        if len(weights.output_normalization.shape) == 1
        else weights.output_normalization[0]
    )
    if rolled:
        output_gate = output_gate_vmem[...].reshape(
            batch_size, head_count, channels_per_head
        )
    elif staged:
        output_gate = projected_vmem[
            :, 3 * attention_width : 4 * attention_width
        ].reshape(batch_size, head_count, channels_per_head)
    attended = (attended * normalization).astype(jnp.bfloat16) * jax.nn.sigmoid(
        output_gate.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    if on_state_updated is not None:
        on_state_updated()
    _phase_end(kda_scope.pop())
    return attended


def _kda_run_stage_as_stage_be(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for kda run stage as."""
    (
        (attended, attention_width, batch_size, communication_workspace),
        (output_ref, reduction_vmem, streams, token_scatter_output),
        (weight_copies, weights),
    ) = stage_ctx
    output_stream = streams.get("output")
    if output_stream is None:
        weight_copies.output.wait()
    elif output_stream.chunks == 1:
        output_stream.wait_all()
        output_stream = None
    with _phase("p2_attn_project_reduce"):
        _project_attention_2d(
            attended.reshape(batch_size, attention_width),
            weights.output_projection_vmem,
            output_ref,
            reduction_vmem,
            communication_workspace.shared_local_reduction_vmem,
            communication_workspace.shared_host_reduction_vmem,
            communication_workspace.attention_features_vmem,
            communication_workspace.attention_output_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            communication_workspace.reduction_phase,
            _dot,
            token_scatter_output=token_scatter_output,
            weight_stream=output_stream,
            staging=communication_workspace.projection_staging_vmem,
            gather_wire=communication_workspace.attention_gather_vmem,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class MultiHeadLatentAttentionWeights:
    """MLA parameters and their persistent projection-copy resources."""

    query_down_projection: TensorLike
    query_down_projection_vmem: TensorLike
    key_value_down_projection: TensorLike
    key_value_down_projection_vmem: TensorLike
    query_up_projection: TensorLike
    query_up_projection_vmem: TensorLike
    key_value_up_projection: TensorLike
    key_value_up_projection_vmem: TensorLike
    output_gate_projection: TensorLike
    output_gate_projection_vmem: TensorLike
    output_projection: TensorLike
    output_projection_vmem: TensorLike
    query_normalization: TensorLike
    key_value_normalization: TensorLike
    projection_copy_semaphores: TensorLike

    def __post_init__(self) -> None:
        """Register PyTree dataclasses lazily on first instantiation."""
        _ensure_dataclasses_registered()

    def make_projection_copies(self) -> _LazyProjectionCopies:
        """Create the asynchronous projection-weight copies on first use."""
        return _LazyProjectionCopies(
            {
                "query_down": (
                    self.query_down_projection,
                    self.query_down_projection_vmem,
                    self.projection_copy_semaphores.at[0],
                ),
                "key_value_down": (
                    self.key_value_down_projection,
                    self.key_value_down_projection_vmem,
                    self.projection_copy_semaphores.at[1],
                ),
                "query_up": (
                    self.query_up_projection,
                    self.query_up_projection_vmem,
                    self.projection_copy_semaphores.at[2],
                ),
                "key_value_up": (
                    self.key_value_up_projection,
                    self.key_value_up_projection_vmem,
                    self.projection_copy_semaphores.at[3],
                ),
                "output_gate": (
                    self.output_gate_projection,
                    self.output_gate_projection_vmem,
                    self.projection_copy_semaphores.at[4],
                ),
                "output": (
                    self.output_projection,
                    self.output_projection_vmem,
                    self.projection_copy_semaphores.at[5],
                ),
            }
        )


PAGE_TOKENS = 256
MLA_LAYERS = 24


@dataclasses.dataclass(frozen=True, slots=True)
class MultiHeadLatentAttentionWorkspace:
    """Persistent VMEM and semaphores used by one MLA invocation."""

    key_cache_vmem: TensorLike
    value_cache_vmem: TensorLike
    cache_copy_semaphores: TensorLike
    query_vmem: TensorLike
    output_gate_values_vmem: TensorLike
    new_key_vmem: TensorLike
    new_value_vmem: TensorLike
    maximum_vmem: TensorLike
    denominator_vmem: TensorLike
    accumulator_vmem: TensorLike

    def __post_init__(self) -> None:
        """Register PyTree dataclasses lazily on first instantiation."""
        _ensure_dataclasses_registered()


def _mla_stage_bk(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for mla stage bk."""
    (
        (communication_workspace, dcp, hidden_ref, new_key_ref, new_value_ref),
        (output_ref, position_ref, sequence_rows, synchronize_devices),
        (weights, workspace),
    ) = stage_ctx
    batch_size, hidden_size = hidden_ref.shape
    context_length = (
        dcp[0].page_tokens * dcp[0].pages * (1 if dcp[0].global_pages else dcp[0].group)
    )
    np.testing.assert_(
        bool(1 <= batch_size <= MAX_BATCH), f"MLA supports 1 to {MAX_BATCH} batch rows"
    )
    np.testing.assert_(
        output_ref.shape == hidden_ref.shape,
        "MLA output must match the hidden-state shape",
    )
    np.testing.assert_(
        not (not 1 <= sequence_rows <= batch_size or batch_size % sequence_rows),
        "sequence_rows must divide the MLA batch",
    )
    _mla_stage_bj(
        ((batch_size, context_length, hidden_size, position_ref, weights), workspace)
    )
    np.testing.assert_(
        not (
            new_key_ref.shape[0] != batch_size
            or new_key_ref.shape[1] < _DCP_HEAD_COUNT
            or new_key_ref.shape[2:] != (256,)
            or (new_value_ref.shape[0] != batch_size)
            or (new_value_ref.shape[1] < _DCP_HEAD_COUNT)
            or (new_value_ref.shape[2:] != (128,))
        ),
        "MLA key/value outputs have invalid shapes",
    )
    if synchronize_devices:
        collectives32.barrier()
        communication_workspace.reduction_phase[0] = 0
    return batch_size


def _mla_stage_bj(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for mla stage bj."""
    ((batch_size, context_length, hidden_size, position_ref, weights), workspace) = (
        stage_ctx
    )
    np.testing.assert_(
        position_ref.shape == (batch_size,), "Positions must match the MLA batch size"
    )
    np.testing.assert_(
        not (
            weights.query_down_projection.shape != (hidden_size, 1536)
            or weights.key_value_down_projection.shape != (hidden_size, 640)
            or weights.query_up_projection.shape != (1536, 768)
            or (weights.key_value_up_projection.shape != (512, 768))
            or (weights.output_gate_projection.shape != (hidden_size, 384))
            or (weights.query_normalization.shape != (1, 1536))
            or (weights.key_value_normalization.shape != (1, 512))
            or (context_length < _ATTN_OUT_WIDTH)
            or context_length % 128
        ),
        "MLA projection and normalization shapes are invalid",
    )
    expected_workspace_shapes = (
        (workspace.query_vmem.shape, (batch_size, 3, 256)),
        (workspace.output_gate_values_vmem.shape, (batch_size, 384)),
        (workspace.new_key_vmem.shape, (batch_size, 3, 256)),
        (workspace.new_value_vmem.shape, (batch_size, 3, 128)),
        (workspace.maximum_vmem.shape, (batch_size, 3, 1)),
        (workspace.denominator_vmem.shape, (batch_size, 3, 1)),
        (workspace.accumulator_vmem.shape, (batch_size, 3, 128)),
    )
    np.testing.assert_(
        not any((actual != expected for actual, expected in expected_workspace_shapes)),
        "MLA workspace has an invalid shape",
    )


def mla(
    position_ref: TensorLike,
    hidden_ref: TensorLike,
    output_ref: TensorLike,
    new_key_ref: TensorLike,
    new_value_ref: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for mla.

    Execute one multi-head latent-attention layer inside a Pallas
    program.

    ``dcp`` (``(mla_dcp.DcpCache, mla_dcp.DcpWorkspace)``) is the
    cache: decode context parallel attention (see ``mla_dcp``) over
    this rank's token shard of the latent pages in vLLM's unified
    pools, attended in latent space for the whole DCP group's heads.
    The new token's latent row ``[512 normalized | 64 rope | 64
    zero]`` is handed out through ``new_key_ref`` rows 0..2 (lanes
    0:256, 256:512, then 512:640 and 128 zero lanes) for the commit;
    ``new_value_ref`` is zero.

    ``streams`` optionally maps "query_up", "key_value_up",
    "output_gate" and "output" to ChunkedWeightStreams that own those
    projections' copies.

    Attention DP runs :func:`mla_dcp_shards` instead.

    Consecutive groups of ``sequence_rows`` rows belong to one
    sequence and share one cache entry. Within a group, rows represent
    consecutive tokens and attend causally to earlier rows from the
    same invocation.
    """
    (
        weights,
        workspace,
        communication_workspace,
        dcp,
        epsilon,
        synchronize_devices,
        weights_are_prefetched,
        token_scatter_output,
        sequence_rows,
        output_projection_after_query_down,
        streams,
    ) = (
        stage_kwargs["weights"],
        stage_kwargs["workspace"],
        stage_kwargs["communication_workspace"],
        stage_kwargs["dcp"],
        stage_kwargs.get("epsilon", 1e-05),
        stage_kwargs.get("synchronize_devices", True),
        stage_kwargs.get("weights_are_prefetched", False),
        stage_kwargs.get("token_scatter_output", False),
        stage_kwargs.get("sequence_rows", 1),
        stage_kwargs.get("output_projection_after_query_down", False),
        stage_kwargs.get("streams", _NONE_STAGE_VAL),
    )
    batch_size = _mla_stage_bk(
        (
            (communication_workspace, dcp, hidden_ref, new_key_ref, new_value_ref),
            (output_ref, position_ref, sequence_rows, synchronize_devices),
            (weights, workspace),
        )
    )
    streams = streams or {}
    weight_copies = weights.make_projection_copies()
    mla_scope = [_phase_start("q6_mla_projections")]

    def start_copy(name: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for start copy."""
        if name in streams:
            streams[name].start()
        else:
            getattr(weight_copies, name).start()

    (dcp_attention, new_latent, projection) = _mla_stage_bi(
        (
            (batch_size, communication_workspace, dcp, epsilon, hidden_ref),
            (mla_scope, new_key_ref, new_value_ref),
            (output_projection_after_query_down, position_ref, sequence_rows),
            (start_copy, streams, weight_copies, weights, weights_are_prefetched),
            workspace,
        )
    )
    workspace.maximum_vmem[...] = jnp.full((batch_size, 3, 1), -jnp.inf, jnp.float32)
    (_attention, _output_stream, project_output) = _mla_stage_bh(
        (
            (batch_size, communication_workspace, dcp, dcp_attention, hidden_ref),
            (mla_scope, new_latent, output_ref, projection, streams),
            (token_scatter_output, weight_copies, weights, workspace),
        )
    )
    pl.run_scoped(
        project_output,
        tpu.VMEM(
            (batch_size, 3584 if batch_size > _KDA_GROUP_WIDTH else 8192), jnp.float32
        ),
    )


def _mla_stage_bi(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for mla stage bi."""
    (
        (batch_size, communication_workspace, dcp, epsilon, hidden_ref),
        (mla_scope, new_key_ref, new_value_ref),
        (output_projection_after_query_down, position_ref, sequence_rows),
        (start_copy, streams, weight_copies, weights, weights_are_prefetched),
        workspace,
    ) = stage_ctx

    def projection(name: TensorLike, x: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for projection.

        ``x`` times a projection in FP32, through its stream if it has
        one.
        """
        guard = communication_workspace.reduction_phase[0] >= 0
        if name in streams:
            return streams[name].dot(x, guard)
        getattr(weight_copies, name).wait()
        return _dot(x, getattr(weights, f"{name}_projection_vmem")[...])

    if not weights_are_prefetched:
        weight_copies.query_down.start()
        weight_copies.key_value_down.start()
    start_copy("query_up")
    start_copy("key_value_up")
    start_copy("output_gate")
    if not output_projection_after_query_down:
        start_copy("output")
    dcp_skip = _config().dcp_skip
    dcp_cache, dcp_workspace = dcp
    dcp_attention = mla_dcp.DcpAttention(
        dcp_cache,
        dcp_workspace,
        position_ref,
        batch_size,
        sequence_rows=sequence_rows,
        skip=dcp_skip,
        cache_verify_rows=sequence_rows > 1 and dcp[0].dtype == jnp.float8_e4m3fn,
    )
    dcp_attention.prime()
    if "query_down" in streams:
        query_latent = (
            streams["query_down"]
            .dot(hidden_ref[...], communication_workspace.reduction_phase[0] >= 0)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
    else:
        weight_copies.query_down.wait()
        query_latent = (
            _dot(hidden_ref[...], weights.query_down_projection_vmem[...])
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
    if output_projection_after_query_down:

        @pl.when(communication_workspace.reduction_phase[0] >= 0)
        def start_output_projection_copy() -> None:
            """Compute and stage Pallas start output projection copy."""
            start_copy("output")

    query_latent = (
        query_latent
        * jax.lax.rsqrt(
            jnp.mean(query_latent * query_latent, axis=1, keepdims=True) + epsilon
        )
        * weights.query_normalization[...]
    ).astype(jnp.bfloat16)
    weight_copies.key_value_down.wait()
    compressed = _dot(
        hidden_ref[...], weights.key_value_down_projection_vmem[...]
    ).astype(jnp.bfloat16)
    key_value_latent = compressed[:, :512].astype(jnp.float32)
    key_value_latent = (
        key_value_latent
        * jax.lax.rsqrt(
            jnp.mean(key_value_latent * key_value_latent, axis=1, keepdims=True)
            + epsilon
        )
        * weights.key_value_normalization[...]
    ).astype(jnp.bfloat16)
    workspace.query_vmem[...] = (
        projection("query_up", query_latent)
        .astype(jnp.bfloat16)
        .reshape(batch_size, 3, 256)
    )
    new_latent = jnp.concatenate(
        (
            key_value_latent,
            compressed[:, 512:576],
            jnp.zeros((batch_size, 64), jnp.bfloat16),
        ),
        axis=1,
    )
    new_key_ref[:, 0] = new_latent[:, :256]
    new_key_ref[:, 1] = new_latent[:, 256:512]
    new_key_ref[:, 2] = jnp.concatenate(
        (new_latent[:, 512:], jnp.zeros((batch_size, 128), jnp.bfloat16)), axis=1
    )
    new_value_ref[:, :3] = jnp.zeros((batch_size, 3, 128), jnp.bfloat16)
    _phase_end(mla_scope.pop())
    mla_scope.append(_phase_start("q7_mla_attend"))
    return (dcp_attention, new_latent, projection)


def _mla_stage_bh(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for mla stage bh."""
    (
        (batch_size, communication_workspace, dcp, dcp_attention, hidden_ref),
        (mla_scope, new_latent, output_ref, projection, streams),
        (token_scatter_output, weight_copies, weights, workspace),
    ) = stage_ctx
    workspace.denominator_vmem[...] = jnp.zeros((batch_size, 3, 1), jnp.float32)
    workspace.accumulator_vmem[...] = jnp.zeros((batch_size, 3, 128), jnp.float32)

    def attend_dcp() -> None:
        """Compute and stage Pallas VMEM state for attend dcp.

        DCP attention: W_UK absorbed into this rank's queries (score scale
        folded in), latent-space attention over the group's token shards
        (``mla_dcp.DcpAttention``, the output gate projected while the
        query gather is in flight), then W_UV per head. The heads' outputs
        land in the accumulator with a unit denominator.
        """
        if "key_value_up" in streams:
            streams["key_value_up"].wait_all()
        else:
            weight_copies.key_value_up.wait()
        up = weights.key_value_up_projection_vmem
        query = workspace.query_vmem[...]
        scale = 192 ** (-0.5)
        absorbed = []
        for head in range(3):
            latent_query = jax.lax.dot_general(
                query[:, head, :128],
                up[:, head * 256 : head * 256 + 128],
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            absorbed.append(
                jnp.concatenate(
                    (
                        latent_query,
                        query[:, head, 128:192].astype(jnp.float32),
                        jnp.zeros((batch_size, 64), jnp.float32),
                    ),
                    axis=1,
                )
                * scale
            )

        def output_gate() -> None:
            """Compute and stage Pallas VMEM state for output gate."""
            workspace.output_gate_values_vmem[...] = projection(
                "output_gate", hidden_ref[...]
            ).astype(jnp.bfloat16)

        merged = dcp_attention.attend(
            jnp.stack(absorbed, axis=1),
            new_latent.astype(jnp.float32),
            dcp[0].layer % 2,
            between=output_gate,
        )
        outputs = []
        for head in range(3):
            value_up = up[:, head * 256 + 128 : head * 256 + 256]
            latent = merged[:, head]
            high = latent.astype(jnp.bfloat16)
            low = (latent - high.astype(jnp.float32)).astype(jnp.bfloat16)
            outputs.append(
                sum(
                    jax.lax.dot_general(
                        part,
                        value_up,
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    for part in (high, low)
                )
            )
        workspace.accumulator_vmem[...] = jnp.stack(outputs, axis=1)
        workspace.denominator_vmem[...] = jnp.ones((batch_size, 3, 1), jnp.float32)

    attend_dcp()
    attention = (
        workspace.accumulator_vmem[...] / workspace.denominator_vmem[...]
    ).reshape(batch_size, 384).astype(jnp.bfloat16) * jax.nn.sigmoid(
        workspace.output_gate_values_vmem[...].astype(jnp.float32)
    ).astype(jnp.bfloat16)
    _phase_end(mla_scope.pop())
    output_stream = streams.get("output")
    if output_stream is None:
        weight_copies.output.wait()
    elif output_stream.chunks == 1:
        output_stream.wait_all()
        output_stream = None

    def project_output(reduction_vmem: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for project output."""
        with _phase("p2_attn_project_reduce"):
            _project_attention_2d(
                attention,
                weights.output_projection_vmem,
                output_ref,
                reduction_vmem,
                communication_workspace.shared_local_reduction_vmem,
                communication_workspace.shared_host_reduction_vmem,
                communication_workspace.attention_features_vmem,
                communication_workspace.attention_output_vmem,
                communication_workspace.send_semaphores,
                communication_workspace.receive_semaphores,
                communication_workspace.reduction_phase,
                _dot,
                token_scatter_output=token_scatter_output,
                weight_stream=output_stream,
                staging=communication_workspace.projection_staging_vmem,
                gather_wire=communication_workspace.attention_gather_vmem,
            )

    return (attention, output_stream, project_output)


def mla_dcp_shards(
    position_ref: TensorLike,
    hidden_ref: TensorLike,
    new_key_ref: TensorLike,
    new_value_ref: TensorLike,
    attended_out: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for mla dcp shards.

    One MLA layer of attention DP under DCP: the rank's held TP32
    shards (3 heads each) attend in one ``mla_dcp.DcpAttention`` call
    over the DP group's latent pages, so a page is read once for every
    held head. ``sequence_rows`` as in :func:`mla`, over the group's
    rows.

    The q/kv latents are computed once; each shard's queries come from
    its chunk of the query-up stream and are absorbed with its W_UK
    columns (the key-value-up stream holds every shard); the output
    gates are projected while the query gather is in flight. W_UV and
    the gate leave each shard's gated attention in
    ``attended_out[shard]`` for the caller's output projection. The
    new latent row is handed out through ``new_key_ref`` rows 0..2 as
    in :func:`mla`.
    """
    weights, streams, communication_workspace, dcp, epsilon, sequence_rows = (
        stage_kwargs["weights"],
        stage_kwargs["streams"],
        stage_kwargs["communication_workspace"],
        stage_kwargs["dcp"],
        stage_kwargs.get("epsilon", 1e-05),
        stage_kwargs.get("sequence_rows", 1),
    )
    rows = hidden_ref.shape[0]
    shards = attended_out.shape[0]
    up_stream = streams["key_value_up"]
    np.testing.assert_(
        not (up_stream.slots is not None or up_stream.chunks != shards),
        "DCP attention DP needs every shard's key-value-up columns resident",
    )
    weight_copies = weights.make_projection_copies()
    guard = communication_workspace.reduction_phase[0] >= 0

    def start_copy(name: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for start copy."""
        if name in streams:
            streams[name].start()
        else:
            getattr(weight_copies, name).start()

    for name in ("query_up", "key_value_up", "output_gate"):
        start_copy(name)
    dcp_cache, dcp_workspace = dcp
    dcp_attention = mla_dcp.DcpAttention(
        dcp_cache,
        dcp_workspace,
        position_ref,
        rows,
        skip=_config().dcp_skip,
        heads=3 * shards,
        sequence_rows=sequence_rows,
    )
    dcp_attention.prime()
    scope = _phase_start("q6_mla_projections")
    if "query_down" in streams:
        query_latent = streams["query_down"].dot(hidden_ref[...], guard)
    else:
        weight_copies.query_down.wait()
        query_latent = _dot(hidden_ref[...], weights.query_down_projection_vmem[...])
    query_latent = query_latent.astype(jnp.bfloat16).astype(jnp.float32)
    _mla_dcp_shards_stage_bl(
        (
            (attended_out, dcp, dcp_attention, epsilon, guard, hidden_ref),
            (new_key_ref, new_value_ref, query_latent, rows, scope, shards),
            (start_copy, streams, up_stream, weight_copies, weights),
        )
    )


def _mla_dcp_shards_stage_bl(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for mla dcp shards stage."""
    (
        (attended_out, dcp, dcp_attention, epsilon, guard, hidden_ref),
        (new_key_ref, new_value_ref, query_latent, rows, scope, shards),
        (start_copy, streams, up_stream, weight_copies, weights),
    ) = stage_ctx

    @pl.when(guard)
    def start_output_projection_copy() -> None:
        """Compute and stage Pallas start output projection copy."""
        start_copy("output")

    query_latent = (
        query_latent
        * jax.lax.rsqrt(
            jnp.mean(query_latent * query_latent, axis=1, keepdims=True) + epsilon
        )
        * weights.query_normalization[...]
    ).astype(jnp.bfloat16)
    weight_copies.key_value_down.wait()
    compressed = _dot(
        hidden_ref[...], weights.key_value_down_projection_vmem[...]
    ).astype(jnp.bfloat16)
    key_value_latent = compressed[:, :512].astype(jnp.float32)
    key_value_latent = (
        key_value_latent
        * jax.lax.rsqrt(
            jnp.mean(key_value_latent * key_value_latent, axis=1, keepdims=True)
            + epsilon
        )
        * weights.key_value_normalization[...]
    ).astype(jnp.bfloat16)
    new_latent = jnp.concatenate(
        (key_value_latent, compressed[:, 512:576], jnp.zeros((rows, 64), jnp.bfloat16)),
        axis=1,
    )
    new_key_ref[:, 0] = new_latent[:, :256]
    new_key_ref[:, 1] = new_latent[:, 256:512]
    new_key_ref[:, 2] = jnp.concatenate(
        (new_latent[:, 512:], jnp.zeros((rows, 128), jnp.bfloat16)), axis=1
    )
    new_value_ref[:, :3] = jnp.zeros((rows, 3, 128), jnp.bfloat16)
    up_stream.wait_all()
    scale = 192 ** (-0.5)
    absorbed = []
    for shard in range(shards):
        query = (
            _shard_dot(streams["query_up"], query_latent, shard, guard)
            .astype(jnp.bfloat16)
            .reshape(rows, 3, 256)
        )
        up = up_stream.view(shard)
        for head in range(3):
            latent_query = jax.lax.dot_general(
                query[:, head, :128],
                up[:, head * 256 : head * 256 + 128],
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            absorbed.append(
                jnp.concatenate(
                    (
                        latent_query,
                        query[:, head, 128:192].astype(jnp.float32),
                        jnp.zeros((rows, 64), jnp.float32),
                    ),
                    axis=1,
                )
                * scale
            )
    _phase_end(scope)
    scope = _phase_start("q7_mla_attend")

    def output_gates() -> None:
        """Compute and stage Pallas VMEM state for output gates."""
        for shard in range(shards):
            attended_out[shard] = _shard_dot(
                streams["output_gate"], hidden_ref[...], shard, guard
            ).astype(jnp.bfloat16)

    merged = dcp_attention.attend(
        jnp.stack(absorbed, axis=1),
        new_latent.astype(jnp.float32),
        dcp[0].layer % 2,
        between=output_gates,
    )
    for shard in range(shards):
        up = up_stream.view(shard)
        outputs = []
        for head in range(3):
            latent = merged[:, 3 * shard + head]
            high = latent.astype(jnp.bfloat16)
            low = (latent - high.astype(jnp.float32)).astype(jnp.bfloat16)
            outputs.append(
                sum(
                    jax.lax.dot_general(
                        part,
                        up[:, head * 256 + 128 : head * 256 + 256],
                        (((1,), (0,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                    for part in (high, low)
                )
            )
        attended_out[shard] = jnp.stack(outputs, axis=1).reshape(rows, 384).astype(
            jnp.bfloat16
        ) * jax.nn.sigmoid(attended_out[shard].astype(jnp.float32)).astype(jnp.bfloat16)
    _phase_end(scope)


B4_ATTENTION_GATHER = _config().b4_attention_gather


def _project_attention_2d(
    attended: TensorLike,
    weight: TensorLike,
    out: TensorLike,
    comm: TensorLike,
    local: TensorLike,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for project attention 2d.

    ``weight_stream`` (optional) is a row ChunkedWeightStream of
    ``weight`` whose positions 0 and 1 are this rank's and its pair
    peer's feature rows.
    """
    (
        host_reduction,
        feature_backing,
        output_backing,
        sends,
        recvs,
        phase,
        dot,
        token_scatter_output,
        weight_stream,
        staging,
        gather_wire,
    ) = (
        (*step_args[0:1], stage_kwargs.get("host_reduction", _NONE_STAGE_VAL))[0],
        (*step_args[1:2], stage_kwargs.get("feature_backing", _NONE_STAGE_VAL))[0],
        (*step_args[2:3], stage_kwargs.get("output_backing", _NONE_STAGE_VAL))[0],
        (*step_args[3:4], stage_kwargs.get("sends", _NONE_STAGE_VAL))[0],
        (*step_args[4:5], stage_kwargs.get("recvs", _NONE_STAGE_VAL))[0],
        (*step_args[5:6], stage_kwargs.get("phase", _NONE_STAGE_VAL))[0],
        (*step_args[6:7], stage_kwargs.get("dot", _NONE_STAGE_VAL))[0],
        stage_kwargs.get("token_scatter_output", False),
        stage_kwargs.get("weight_stream", _NONE_STAGE_VAL),
        stage_kwargs.get("staging", _NONE_STAGE_VAL),
        stage_kwargs.get("gather_wire", _NONE_STAGE_VAL),
    )
    rank = jax.lax.axis_index("tp")
    slot = phase[0] % 2
    batch_size = attended.shape[0]
    ordered_full_output = (
        B4_ATTENTION_GATHER
        and batch_size == _TP_RING_SIZE
        and (not token_scatter_output)
    )
    width = attended.shape[-1]
    g = weight.shape[0] // width
    np.testing.assert_(g == _PAIR_TUPLE_LEN)
    np.testing.assert_(
        1 <= batch_size <= MAX_BATCH
        and out.shape[0] == batch_size
        and (comm.shape[0] == batch_size)
    )
    h = out.shape[-1]
    half = h // g
    nr = 32 // g

    def compute(features: TensorLike, output: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for compute."""
        return _project_attention_2d_compute_stage_bm(
            (
                (attended, batch_size, comm, dot, g, gather_wire, h, half),
                (host_reduction, local, nr, ordered_full_output, out, output_backing),
                (phase, rank, recvs, sends, slot, staging, token_scatter_output),
                (weight, weight_stream, width),
            ),
            (features, output),
        )

    np.testing.assert_(g * width <= _KV_COMMIT_TILE_TOKENS and h == _HIDDEN_SIZE)
    feature_store = _bf16_rows_view(feature_backing, batch_size, 1024)
    output_store = _bf16_rows_view(output_backing, batch_size, 7168)
    compute(feature_store.at[:, : g * width], output_store)


def _project_attention_2d_compute_stage_bm(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for compute."""
    (
        (attended, batch_size, comm, dot, g, gather_wire, h, half),
        (host_reduction, local, nr, ordered_full_output, out, output_backing),
        (phase, rank, recvs, sends, slot, staging, token_scatter_output),
        (weight, weight_stream, width),
    ) = stage_ctx
    features, output = stage_args
    view = features.at[:, pl.ds(rank % g * width, width)]
    view[...] = attended
    for offset in range(g - 1, 0, -1):
        tpu.make_async_remote_copy(
            view,
            view,
            sends.at[10 + slot],
            recvs.at[10 + slot],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    comm[...] = jnp.zeros(comm.shape, jnp.float32)
    if weight_stream is None:
        own_weight = weight[pl.ds(rank % 2 * width, width), :]
    else:
        weight_stream.wait(0)
        own_weight = weight_stream.view(0)[...]
    comm[:, :half] = dot(attended, own_weight)
    if weight_stream is not None:
        weight_stream.release(0, phase[0] >= 0)
    count = features.at[:, : (g - 1) * width]
    tpu.make_async_copy(count, count, sends.at[10 + slot]).wait()
    tpu.make_async_copy(count, count, recvs.at[10 + slot]).wait()
    peer_lane = rank % 2 ^ 1
    if weight_stream is None:
        peer_weight = weight[pl.ds(peer_lane * width, width), :]
    else:
        weight_stream.wait(1)
        peer_weight = weight_stream.view(1)[...]
    comm[:, :half] += dot(features[:, pl.ds(peer_lane * width, width)], peer_weight)
    if token_scatter_output or ordered_full_output:
        _project_attention_2d_compute_stage_b_stage_bn(
            (
                (batch_size, comm, gather_wire, h, half, host_reduction, local),
                (ordered_full_output, out, output_backing, rank, recvs, sends),
                staging,
            )
        )
    else:
        rows = (half + 255) // 256 * 2
        payload = rows // 2
        batched_payload = batch_size * payload
        receive = local.reshape(nr, local.shape[1] * 8 // nr, 128)
        np.testing.assert_(batched_payload <= receive.shape[1])
        bits = tpu.bitcast(
            comm[:, : rows * 128].reshape(batch_size * rows, 128).astype(jnp.bfloat16),
            jnp.uint32,
        )
        receive[rank // g, :batched_payload, :] = jax.lax.bitcast_convert_type(
            bits, jnp.float32
        )
        view = receive.at[rank // g, :batched_payload, :]
        for offset in range(nr - 1, 0, -1):
            tpu.make_async_remote_copy(
                view,
                view,
                sends.at[14 + slot],
                recvs.at[14 + slot],
                device_id=(rank ^ g * offset,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
        count = receive.at[: nr - 1, :batched_payload, :]
        tpu.make_async_copy(count, count, sends.at[14 + slot]).wait()
        tpu.make_async_copy(count, count, recvs.at[14 + slot]).wait()
        values = tpu.bitcast(
            jax.lax.bitcast_convert_type(receive[:, :batched_payload, :], jnp.uint32),
            jnp.bfloat16,
        ).astype(jnp.float32)
        comm[:, : rows * 128] = jnp.sum(values, axis=0).reshape(batch_size, rows * 128)
        view = output.at[:, pl.ds(rank % g * half, half)]
        view[...] = comm[:, :half].astype(jnp.bfloat16)
        for offset in range(g - 1, 0, -1):
            tpu.make_async_remote_copy(
                view,
                view,
                sends.at[12 + slot],
                recvs.at[12 + slot],
                device_id=(rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
        count = output.at[:, : (g - 1) * half]
        tpu.make_async_copy(count, count, sends.at[12 + slot]).wait()
        tpu.make_async_copy(count, count, recvs.at[12 + slot]).wait()
        out[...] = output[...]
    phase[0] += 1


def _project_attention_2d_compute_stage_b_stage_bn(stage_ctx: TensorLike) -> None:
    """Compute Pallas VMEM tensors for project stage."""
    (
        (batch_size, comm, gather_wire, h, half, host_reduction, local),
        (ordered_full_output, out, output_backing, rank, recvs, sends),
        staging,
    ) = stage_ctx
    np.testing.assert_(batch_size in (4, *SCATTER_BATCHES))
    tokens_per_pair = batch_size // 4
    payload_rows = tokens_per_pair * half // 128
    host = rank // 8
    pair = rank % 8 // 2
    parity = rank % 2
    np.testing.assert_(
        staging is not None or 2 * local.shape[1] >= 4 * payload_rows,
        "wide reduction rows too small for the payloads",
    )
    local_send = (
        (local.at[0].bitcast(jnp.bfloat16) if staging is None else staging.at[0])
        .at[pl.ds(0, 4 * payload_rows), :]
        .reshape(4, payload_rows, 128)
    )
    host_receive, owned_output = _project_attention_2d_compute_stage_b_stage_bp(
        (
            (comm, h, half, host, host_reduction, local, local_send),
            (ordered_full_output, output_backing, pair, payload_rows, rank, recvs),
            (sends, staging, tokens_per_pair),
        )
    )
    if ordered_full_output:
        rank_partials = (
            host_receive[...].reshape(16, payload_rows, 128).astype(jnp.float32)
        )
        reduced_half = jnp.sum(rank_partials, axis=0).astype(jnp.bfloat16)
    else:
        reduced_half = jnp.sum(host_receive[...], axis=0).astype(jnp.bfloat16)
    pair_send = local.at[2].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    pair_receive = local.at[3].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    pair_send[...] = reduced_half
    pair_copy = tpu.make_async_remote_copy(
        pair_send,
        pair_receive,
        sends.at[6],
        recvs.at[6],
        device_id=(rank ^ 1,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    pair_copy.start()
    pair_copy.wait()
    sent_half = pair_send[...].reshape(tokens_per_pair, half)
    received_half = pair_receive[...].reshape(tokens_per_pair, half)
    owned_output[:, :half] = jnp.where(parity == 0, sent_half, received_half)
    owned_output[:, half:] = jnp.where(parity == 0, received_half, sent_half)
    if ordered_full_output:
        np.testing.assert_(gather_wire is not None)
        _gather_local_pair_rows(owned_output, out, sends, recvs, gather_wire, owners=4)


def _project_attention_2d_compute_stage_b_stage_bp(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for project stage."""
    (
        (comm, h, half, host, host_reduction, local, local_send),
        (ordered_full_output, output_backing, pair, payload_rows, rank, recvs),
        (sends, staging, tokens_per_pair),
    ) = stage_ctx
    local_receive = (
        (local.at[1].bitcast(jnp.bfloat16) if staging is None else staging.at[1])
        .at[pl.ds(0, 4 * payload_rows), :]
        .reshape(4, payload_rows, 128)
    )
    owned_output = _bf16_rows_view(output_backing, tokens_per_pair, h)
    for target_pair in range(4):
        if tokens_per_pair == 1:
            partial = comm[target_pair, :half][None, :]
        else:
            partial = comm[pl.ds(target_pair * tokens_per_pair, tokens_per_pair), :half]
        local_send[target_pair] = partial.astype(jnp.bfloat16).reshape(
            payload_rows, 128
        )
    local_receive[pair] = local_send[pair]
    local_copies = []
    for offset in range(3, 0, -1):
        target_pair = pair ^ offset
        copy = tpu.make_async_remote_copy(
            local_send.at[target_pair],
            local_receive.at[pair],
            sends.at[offset - 1],
            recvs.at[offset - 1],
            device_id=(rank ^ 2 * offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        local_copies.append(copy)
    for copy in local_copies:
        copy.wait()
    if ordered_full_output:
        host_wire_rows = 4 * payload_rows
        np.testing.assert_(host_wire_rows <= 2 * host_reduction.shape[1])
        host_receive = host_reduction.bitcast(jnp.bfloat16).at[
            :, pl.ds(0, host_wire_rows), :
        ]
        host_receive[host] = local_receive[...].reshape(host_wire_rows, 128)
    else:
        local_sum = jnp.sum(local_receive[...].astype(jnp.float32), axis=0)
        host_receive = host_reduction.at[:, pl.ds(0, payload_rows), :]
        host_receive[host] = local_sum
    host_copies = []
    host_view = host_receive.at[host]
    for offset in range(3, 0, -1):
        copy = tpu.make_async_remote_copy(
            host_view,
            host_view,
            sends.at[3 + offset - 1],
            recvs.at[3 + offset - 1],
            device_id=(rank ^ 8 * offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        host_copies.append(copy)
    for copy in host_copies:
        copy.wait()
    return (host_receive, owned_output)


def _project_attention_dp(
    attended: TensorLike,
    weight: TensorLike,
    out: TensorLike,
    comm: TensorLike,
    local: TensorLike,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Project DP2/TP16 or DP4/TP8 attention back to DP row ownership.

    ``attended`` [dp, 4r, 384] holds each logical TP32 head shard for
    the DP group's rows. The projection first reduces over four local
    rank pairs; DP2 also reduces over its second host. Rank pairs
    assemble the hidden dimension, so ``out`` [r, 7168] holds this
    rank pair's r rows of its DP group. With one row owner (DP2 B2/B4,
    ``attended`` [2, r, 384]) each pair's partial of its group's r
    rows goes to the 15 same-lane ranks of both groups in one hop, so
    ``out`` [2r, 7168] is the whole batch on every rank. Output weight
    chunks are all own-lane shard halves, then peer halves.
    """
    (
        host_reduction,
        feature_backing,
        sends,
        recvs,
        phase,
        dot,
        weight_stream,
        staging,
        owners,
    ) = (
        (*step_args[0:1], stage_kwargs.get("host_reduction", _NONE_STAGE_VAL))[0],
        (*step_args[1:2], stage_kwargs.get("feature_backing", _NONE_STAGE_VAL))[0],
        (*step_args[2:3], stage_kwargs.get("sends", _NONE_STAGE_VAL))[0],
        (*step_args[3:4], stage_kwargs.get("recvs", _NONE_STAGE_VAL))[0],
        (*step_args[4:5], stage_kwargs.get("phase", _NONE_STAGE_VAL))[0],
        (*step_args[5:6], stage_kwargs.get("dot", _NONE_STAGE_VAL))[0],
        stage_kwargs.get("weight_stream", _NONE_STAGE_VAL),
        stage_kwargs.get("staging", _NONE_STAGE_VAL),
        stage_kwargs.get("owners", 4),
    )
    copy = _NONE_STAGE_VAL
    rank = jax.lax.axis_index("tp")
    slot = phase[0] % 2
    shards, rows, width = attended.shape
    hidden = out.shape[-1]
    half = hidden // 2
    np.testing.assert_(shards in (2, 4) and owners in (1, 4) and (rows % owners == 0))
    np.testing.assert_(owners == _TP_RING_SIZE or shards == _PAIR_TUPLE_LEN)
    lane, pair, host = (rank % 2, rank % 8 // 2, rank // 8)
    payload_rows, rows_per_pair = _project_attention_dp_stage_br(
        (
            (attended, comm, dot, feature_backing, half, lane, owners, phase),
            (rank, recvs, rows, sends, shards, slot, weight, weight_stream, width),
        )
    )
    if owners == 1:
        host_slots = host_reduction.bitcast(jnp.bfloat16)

        def pair_partials(source_host: TensorLike) -> TensorLike:
            """Compute and stage Pallas VMEM state for pair partials."""
            return (
                host_slots.at[source_host]
                .at[pl.ds(0, 4 * payload_rows), :]
                .reshape(4, payload_rows, 128)
            )

        own_partial = pair_partials(host).at[pair]
        own_partial[...] = (
            comm[:, :half].astype(jnp.bfloat16).reshape(payload_rows, 128)
        )
        copies = []
        for offset in range(15, 0, -1):
            copy = tpu.make_async_remote_copy(
                own_partial,
                own_partial,
                sends.at[12],
                recvs.at[12],
                device_id=(rank ^ 2 * offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        host_sums = [
            jnp.sum(pair_partials(source_host)[...].astype(jnp.float32), axis=0)
            for source_host in range(4)
        ]
        reduced_half = jnp.concatenate(
            [host_sums[2 * group] + host_sums[2 * group + 1] for group in range(2)],
            axis=0,
        ).astype(jnp.bfloat16)
        payload_rows *= 2
        rows_per_pair *= 2
    else:
        slot_rows = (
            payload_rows if staging is None else _staging_slot_rows(payload_rows)
        )
        local_send = (
            (local.at[0].bitcast(jnp.bfloat16) if staging is None else staging.at[0])
            .at[pl.ds(0, 4 * slot_rows), :]
            .reshape(4, slot_rows, 128)
        )
        reduced_half = _project_attention_dp_stage_bq_stage_bs(
            (
                (comm, half, host, host_reduction, local, local_send, owners, pair),
                (payload_rows, rank, recvs, rows_per_pair, sends, shards, slot_rows),
                staging,
            )
        )
    pair_send = local.at[2].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    pair_receive = local.at[3].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    pair_send[...] = reduced_half
    copy = tpu.make_async_remote_copy(
        pair_send,
        pair_receive,
        sends.at[6],
        recvs.at[6],
        device_id=(rank ^ 1,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()
    copy.wait()
    sent = pair_send[...].reshape(rows_per_pair, half)
    received_half = pair_receive[...].reshape(rows_per_pair, half)
    row = jnp.concatenate(
        (
            jnp.where(lane == 0, sent, received_half),
            jnp.where(lane == 0, received_half, sent),
        ),
        axis=1,
    )
    np.testing.assert_(out.shape == (rows_per_pair, hidden))
    out[...] = row
    phase[0] += 1


def _project_attention_dp_stage_br(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for project stage."""
    (
        (attended, comm, dot, feature_backing, half, lane, owners, phase),
        (rank, recvs, rows, sends, shards, slot, weight, weight_stream, width),
    ) = stage_ctx
    rows_per_pair = rows // owners
    np.testing.assert_(
        weight_stream is not None or weight.shape == (shards * 2 * width, half)
    )
    guard = phase[0] >= 0

    def weight_rows(position: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for weight rows."""
        shard, peer = (position % shards, position // shards)
        if weight_stream is not None:
            weight_stream.wait(position)
            return weight_stream.view(position)[...]
        return weight[pl.ds((shard * 2 + (lane ^ peer)) * width, width), :]

    features = _bf16_rows_view(feature_backing, rows, feature_backing.shape[-1])
    np.testing.assert_(features.shape[-1] >= 2 * width * shards)
    for shard in range(shards):
        view = features.at[:, pl.ds(shard * 2 * width + lane * width, width)]
        view[...] = attended[shard]
        tpu.make_async_remote_copy(
            view,
            view,
            sends.at[10 + slot],
            recvs.at[10 + slot],
            device_id=(rank ^ 1,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    comm[...] = jnp.zeros(comm.shape, jnp.float32)
    for shard in range(shards):
        comm[:, :half] += dot(attended[shard], weight_rows(shard))
        if weight_stream is not None:
            weight_stream.release(shard, guard)
    received = features.at[:, pl.ds(0, shards * width)]
    tpu.make_async_copy(received, received, sends.at[10 + slot]).wait()
    tpu.make_async_copy(received, received, recvs.at[10 + slot]).wait()
    peer_lane = lane ^ 1
    for shard in range(shards):
        peer_features = features[:, pl.ds(shard * 2 * width + peer_lane * width, width)]
        comm[:, :half] += dot(peer_features, weight_rows(shards + shard))
        if weight_stream is not None:
            weight_stream.release(shards + shard, guard)
    payload_rows = rows_per_pair * half // 128
    return (payload_rows, rows_per_pair)


def _project_attention_dp_stage_bq_stage_bs(stage_ctx: TensorLike) -> TensorLike:
    """Compute Pallas VMEM tensors for project stage."""
    (
        (comm, half, host, host_reduction, local, local_send, owners, pair),
        (payload_rows, rank, recvs, rows_per_pair, sends, shards, slot_rows),
        staging,
    ) = stage_ctx
    local_receive = (
        (local.at[1].bitcast(jnp.bfloat16) if staging is None else staging.at[1])
        .at[pl.ds(0, 4 * slot_rows), :]
        .reshape(4, slot_rows, 128)
    )
    for target_pair in range(4):
        wire = (
            comm[
                pl.ds(
                    target_pair * rows_per_pair if owners == _TP_RING_SIZE else 0,
                    rows_per_pair,
                ),
                :half,
            ]
            .astype(jnp.bfloat16)
            .reshape(payload_rows, 128)
        )
        if slot_rows != payload_rows:
            wire = jnp.concatenate(
                [wire, jnp.zeros((slot_rows - payload_rows, 128), jnp.bfloat16)]
            )
        local_send[target_pair] = wire
    local_receive[pair] = local_send[pair]
    copies = []
    for offset in range(3, 0, -1):
        copy = tpu.make_async_remote_copy(
            local_send.at[pair ^ offset],
            local_receive.at[pair],
            sends.at[offset - 1],
            recvs.at[offset - 1],
            device_id=(rank ^ 2 * offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
    local_sum = jnp.sum(local_receive[...].astype(jnp.float32), axis=0)
    if slot_rows != payload_rows:
        local_sum = local_sum[:payload_rows]
    if shards == _PAIR_TUPLE_LEN:
        host_receive = host_reduction.at[:2, pl.ds(0, payload_rows), :]
        host_receive[host % 2] = local_sum
        host_view = host_receive.at[host % 2]
        copy = tpu.make_async_remote_copy(
            host_view,
            host_view,
            sends.at[3],
            recvs.at[3],
            device_id=(rank ^ 8,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copy.wait()
        reduced_half = jnp.sum(host_receive[...], axis=0).astype(jnp.bfloat16)
    else:
        reduced_half = local_sum.astype(jnp.bfloat16)
    return reduced_half


def _gather_dp_moe_rows_stage_bu(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for gather dp moe rows."""
    ((local, pair_semaphores, recvs, row, sends, shards),) = stage_ctx
    copy = half = k = _NONE_STAGE_VAL
    o = slot = start = _NONE_STAGE_VAL
    rank = jax.lax.axis_index("tp")
    group_size = 32 // shards
    group, pair = (rank // group_size, rank % 8 // 2)
    rows, hidden = row.shape
    payload = rows * hidden // 128
    stride = -(-payload // 16) * 16
    slot_count = 2 * local.shape[0] * local.shape[1] // stride
    base = slot_count // 2
    np.testing.assert_(base >= 4 * shards)
    wires = pool_alias.view(local, (slot_count, stride, 128), jnp.bfloat16)
    if pair_semaphores is not None:
        half = -(-payload // 32) * 16
        start = pl.multiple_of(rank % 2 * (payload - half), 16)

    def wire(o: TensorLike, k: TensorLike, *, whole: TensorLike = False) -> TensorLike:
        """Compute and stage Pallas VMEM state for wire."""
        if pair_semaphores is None or whole:
            return wires.at[base + 4 * o + k, pl.ds(0, payload)]
        return wires.at[base + 4 * o + k, pl.ds(start, half)]

    def send(
        source: TensorLike, o: TensorLike, k: TensorLike, semaphore: TensorLike
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for send."""
        copy = tpu.make_async_remote_copy(
            source,
            wire(o, k),
            sends.at[semaphore],
            recvs.at[semaphore],
            device_id=(rank ^ group_size * o ^ 2 * k,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        return copy

    wire(0, 0, whole=True)[...] = row.reshape(payload, 128)
    _gather_dp_moe_rows_stage_bt(((rank, recvs, rows, send, sends, shards, wire),))
    if pair_semaphores is not None:
        swaps = []
        for slot in range(1, 4 * shards):
            o, k = divmod(slot, 4)
            swaps.append(
                tpu.make_async_remote_copy(
                    wire(o, k),
                    wire(o, k),
                    *pair_semaphores,
                    device_id=(rank ^ 1,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            )
            swaps[-1].start()
        for copy in swaps:
            copy.wait()
    return (
        (base, group, group_size, half, hidden, pair, payload, rank, rows, start),
        (wire, wires),
    )


def _gather_dp_moe_rows_stage_bt(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for gather dp moe rows."""
    ((rank, recvs, rows, send, sends, shards, wire),) = stage_ctx
    if rows <= _PAIR_TUPLE_LEN and shards == _PAIR_TUPLE_LEN:
        _gather_dp_moe_rows_stage_bt_stage_bv(send, shards, wire)
    else:
        local_copies = [send(wire(0, 0), 0, k, 7) for k in range(1, 4)]
        group_copies = [
            send(wire(0, 0), o, 0, (8, 10, 11)[o - 1]) for o in range(1, shards)
        ]
        forwarded = []
        for o, copy in enumerate(group_copies, 1):
            copy.wait_recv()
            for k in range(1, 4):
                forwarded.append(
                    tpu.make_async_remote_copy(
                        wire(o, 0),
                        wire(o, k),
                        sends.at[9],
                        recvs.at[9],
                        device_id=(rank ^ 2 * k,),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                )
                forwarded[-1].start()
        for copy in group_copies:
            copy.wait_send()
        for copy in (*local_copies, *forwarded):
            copy.wait()


def _gather_dp_moe_rows_stage_bt_stage_bv(
    send: TensorLike, shards: TensorLike, wire: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for gather dp moe rows."""
    direct = [
        send(wire(0, 0), o, k, 7 if o == 0 else 8)
        for o in range(shards)
        for k in range(4)
        if o or k
    ]
    for copy in direct:
        copy.wait()


def _gather_dp_moe_rows(
    row: TensorLike,
    owned_out: TensorLike,
    out: TensorLike,
    local: TensorLike,
    sends: TensorLike,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Gather the DP groups' rows and the local pairs' rows, overlapped.

    Each rank holds [batch / (4 * DP), hidden] for its (DP group,
    local pair) and ends with the whole batch in ``out`` and its
    pair's rows of all groups in ``owned_out``. Own rows go to the
    local pairs and the other groups together, so the inter-group
    exchange does not delay the local one; each group's wire is
    forwarded as soon as it lands. Flat wires use lanes 4..7 of
    ``local``: the wire from relative position (group ^ o, pair ^ k)
    lands in slot (o, k) on every rank. With ``pair_semaphores`` the
    pair's two ranks (one chip) each carry half of every wire and then
    swap halves.
    """
    recvs, shards, pair_semaphores = (
        (*step_args[0:1], stage_kwargs.get("recvs", _NONE_STAGE_VAL))[0],
        (*step_args[1:2], stage_kwargs.get("shards", _NONE_STAGE_VAL))[0],
        (*step_args[2:3], stage_kwargs.get("pair_semaphores", _NONE_STAGE_VAL))[0],
    )
    (
        (base, group, _group_size, _half, hidden, pair, payload, _rank, rows, _start),
        (_wire, wires),
    ) = _gather_dp_moe_rows_stage_bu(
        ((local, pair_semaphores, recvs, row, sends, shards),)
    )
    for d in range(shards):
        for p in range(4):
            slot = base + 4 * (group ^ d) + (pair ^ p)
            first = (p * shards + d) * rows
            out[first : first + rows, :] = wires[slot, pl.ds(0, payload)].reshape(
                rows, hidden
            )
        owned_out[d * rows : (d + 1) * rows, :] = wires[
            base + 4 * (group ^ d), pl.ds(0, payload)
        ].reshape(rows, hidden)


def _router_copy(
    weight: TensorLike,
    buffer: TensorLike,
    sems: TensorLike,
    *,
    wait: bool,
    initialize_padding: bool = True,
) -> None:
    """Compute and stage Pallas VMEM state for router copy."""
    lane = jax.lax.axis_index("tp") % 2

    @pl.when(lane < 1)
    def full() -> None:
        """Compute and stage Pallas VMEM state for full."""
        copy = tpu.make_async_copy(
            weight.at[:, pl.ds(lane * 512, 512)], buffer.at[0], sems.at[0]
        )
        copy.wait() if wait else copy.start()

    @pl.when(lane == 1)
    def tail() -> None:
        """Compute and stage Pallas VMEM state for tail."""
        if not wait and initialize_padding:
            buffer[0, :, 384:] = jnp.zeros((weight.shape[0], 128), jnp.bfloat16)
        copy = tpu.make_async_copy(
            weight.at[:, pl.ds(512, 384)], buffer.at[0, :, :384], sems.at[0]
        )
        copy.wait() if wait else copy.start()


def _dense_scatter_rows(
    partial: TensorLike,
    output: TensorLike,
    local: TensorLike,
    hosts: TensorLike,
    sends: TensorLike,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Reduce the dense layer directly to its token owners.

    Preserve the original BF16 wire format, local eight-rank sum and
    four-host sum, including their rounding points. Each owner
    consumes eight rows per round instead of every rank reducing the
    entire batch and discarding most rows. Pack each owner's source
    once; the two ranks in each owner pair send and receive identical
    logical row ranges.
    """
    recvs, owners = (
        (*step_args[0:1], stage_kwargs.get("recvs", _NONE_STAGE_VAL))[0],
        (*step_args[1:2], stage_kwargs.get("owners", _NONE_STAGE_VAL))[0],
    )
    rank = jax.lax.axis_index("tp")
    lane, host = (rank % 8, rank // 8)
    owned_rows = partial.shape[0] // owners
    hidden = partial.shape[1]
    wire_rows = 8 * hidden // 128
    payload = wire_rows // 2
    np.testing.assert_(owned_rows % 8 == 0 and 8 % owners == 0)
    np.testing.assert_(payload <= local.shape[1] and payload <= hosts.shape[1])

    def reduce_owned(wires: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for reduce owned."""
        return _dense_scatter_rows_reduce_owned_stage_bw(
            (
                (hidden, host, hosts, lane, local, output, owned_rows, owners),
                (partial, payload, rank, recvs, sends, wire_rows),
            ),
            wires,
        )

    pl.run_scoped(reduce_owned, tpu.VMEM((owners, payload, 128), jnp.float32))


def _dense_scatter_rows_reduce_owned_stage_bw(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for reduce owned."""
    (
        (hidden, host, hosts, lane, local, output, owned_rows, owners),
        (partial, payload, rank, recvs, sends, wire_rows),
    ) = stage_ctx
    wires = stage_args

    def encode(value: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for encode."""
        packed = tpu.bitcast(
            value.reshape(wire_rows, 128).astype(jnp.bfloat16), jnp.uint32
        )
        return jax.lax.bitcast_convert_type(packed, jnp.float32)

    def decode(value: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for decode."""
        unpacked = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(unpacked, axis=0).reshape(8, hidden)

    for first in range(0, owned_rows, 8):
        if first:
            collectives32.barrier()
        for owner in range(owners):
            start = owner * owned_rows + first
            wires[owner] = encode(partial[start : start + 8])
        local[lane, :payload] = wires[lane // (8 // owners)]
        copies = []
        for offset in range(7, 0, -1):
            copy = tpu.make_async_remote_copy(
                wires.at[(lane ^ offset) // (8 // owners)],
                local.at[lane, :payload],
                sends.at[offset - 1],
                recvs.at[offset - 1],
                device_id=(rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        hosts[host, :payload] = encode(decode(local[:, :payload]))
        copies = []
        for offset in range(3, 0, -1):
            view = hosts.at[host, :payload]
            copy = tpu.make_async_remote_copy(
                view,
                view,
                sends.at[6 + offset],
                recvs.at[6 + offset],
                device_id=(rank ^ 8 * offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        output[first : first + 8] = decode(hosts[:, :payload]).astype(jnp.bfloat16)


def _all_reduce_rows(
    partial: TensorLike,
    local: TensorLike,
    hosts: TensorLike,
    sends: TensorLike,
    recvs: TensorLike,
) -> TensorLike:
    """TP32 sum of an FP32 ``[batch, hidden]`` partial on every rank.

    The shared-expert transport of the mixture-of-experts operation:
    the partial is rounded to BF16 and packed in pairs into 32-bit
    words, each lane sends its packet to the seven other local lanes,
    the local sum is packed again and exchanged among the four hosts.
    ``local`` and ``hosts`` are the wide reduction buffers of the
    communication workspace (``[8, rows, 128]`` and ``[4, rows, 128]``
    FP32); semaphores 0..6 and 7..9 are used and left idle.
    """
    rank = jax.lax.axis_index("tp")
    lane, host = (rank % 8, rank // 8)
    batch_size, hidden = partial.shape
    rows = batch_size * hidden // 128
    payload = rows // 2
    np.testing.assert_(
        rows % 2 == 0 and payload <= local.shape[1] and (payload <= hosts.shape[1])
    )

    def encode(value: TensorLike, destination: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for encode."""
        packed = tpu.bitcast(value.reshape(rows, 128).astype(jnp.bfloat16), jnp.uint32)
        destination[:payload, :] = jax.lax.bitcast_convert_type(packed, jnp.float32)

    def decode(buffer: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for decode."""
        decoded = tpu.bitcast(
            jax.lax.bitcast_convert_type(buffer, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(decoded, axis=0).reshape(batch_size, hidden)

    def exchange(
        buffer: TensorLike,
        index: TensorLike,
        offsets: TensorLike,
        stride: TensorLike,
        semaphore_base: TensorLike,
    ) -> None:
        """Compute and stage Pallas VMEM state for exchange."""
        view = buffer.at[index, :payload, :]
        copies = []
        for offset in offsets:
            copy = tpu.make_async_remote_copy(
                view,
                view,
                sends.at[semaphore_base + offset],
                recvs.at[semaphore_base + offset],
                device_id=(rank ^ stride * offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

    encode(partial, local.at[lane])
    exchange(local, lane, range(7, 0, -1), 1, -1)
    encode(decode(local[:, :payload, :]), hosts.at[host])
    exchange(hosts, host, range(3, 0, -1), 8, 6)
    return decode(hosts[:, :payload, :])


def _gather_sp8_local_rows(
    row: TensorLike,
    out: TensorLike,
    local: TensorLike,
    sends: TensorLike,
    recvs: TensorLike,
) -> None:
    """Eight owners directly feed the existing pre-attention all-gather."""
    rank = jax.lax.axis_index("tp")
    owner = rank % 8
    rows, hidden = row.shape
    payload = rows * hidden // 128
    np.testing.assert_(rows == _ROUTED_EXPERTS_PER_TOKEN and payload % 16 == 0)
    slots = 2 * local.shape[0] * local.shape[1] // payload
    np.testing.assert_(slots >= _KDA_GROUP_WIDTH)
    wires = pool_alias.view(local, (slots, payload, 128), jnp.bfloat16)
    source = wires.at[owner]
    source[...] = row.reshape(payload, 128)
    copies = []
    for offset in range(7, 0, -1):
        copy = tpu.make_async_remote_copy(
            source,
            source,
            sends.at[offset - 1],
            recvs.at[offset - 1],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
    for pair in range(4):
        out[pair * 16 : (pair + 1) * 16] = jnp.concatenate(
            (
                wires[2 * pair].reshape(rows, hidden),
                wires[2 * pair + 1].reshape(rows, hidden),
            ),
            axis=0,
        )


def _gather_sp8_dp_moe_rows(
    row: TensorLike,
    owned_out: TensorLike,
    out: TensorLike,
    local: TensorLike,
    sends: TensorLike,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for gather sp8 dp moe.

    DP2 MoE-input all-gather, eight owners, existing pair-major ABI
    out.

    Payload slots are relative (DP group XOR, local core XOR). The
    upper half of the existing transport allocation is disjoint from
    pre gather. No separate pair reconstruction collective is needed.
    """
    recvs: TensorLike = (*step_args[0:1], stage_kwargs.get("recvs", _NONE_STAGE_VAL))[0]
    rank = jax.lax.axis_index("tp")
    group, owner = (rank // 16, rank % 8)
    rows, hidden = row.shape
    payload = rows * hidden // 128
    np.testing.assert_(rows == _ROUTED_EXPERTS_PER_TOKEN and payload % 16 == 0)
    slots = 2 * local.shape[0] * local.shape[1] // payload
    base = slots // 2
    np.testing.assert_(base >= _EXPERT_GROUP_STRIDE)
    wires = pool_alias.view(local, (slots, payload, 128), jnp.bfloat16)

    def wire(o: TensorLike, k: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for wire."""
        return wires.at[base + 8 * o + k]

    def send(
        source: TensorLike,
        o: TensorLike,
        k: TensorLike,
        semaphore: TensorLike,
        *,
        local_only: TensorLike = False,
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for send."""
        copy = tpu.make_async_remote_copy(
            source,
            wire(o, k),
            sends.at[semaphore],
            recvs.at[semaphore],
            device_id=(rank ^ (0 if local_only else 16 * o) ^ k,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        return copy

    wire(0, 0)[...] = row.reshape(payload, 128)
    local_copies = [send(wire(0, 0), 0, k, 7) for k in range(1, 8)]
    group_copy = send(wire(0, 0), 1, 0, 8)
    group_copy.wait_recv()
    forwarded = [send(wire(1, 0), 1, k, 9, local_only=True) for k in range(1, 8)]
    group_copy.wait_send()
    for copy in (*local_copies, *forwarded):
        copy.wait()
    for d in range(2):
        for p in range(4):
            even = wire(group ^ d, owner ^ 2 * p)[...].reshape(rows, hidden)
            odd = wire(group ^ d, owner ^ 2 * p + 1)[...].reshape(rows, hidden)
            first = (2 * p + d) * 16
            out[first : first + 16] = jnp.concatenate((even, odd), axis=0)
        own = wire(group ^ d, 0)[...].reshape(rows, hidden)
        peer = wire(group ^ d, 1)[...].reshape(rows, hidden)
        owned_out[d * 16 : (d + 1) * 16] = jnp.where(
            owner % 2 == 0,
            jnp.concatenate((own, peer), axis=0),
            jnp.concatenate((peer, own), axis=0),
        )


def _gather_local_pair_rows(
    source: TensorLike,
    destination: TensorLike,
    sends: TensorLike,
    recvs: TensorLike,
    scratch: TensorLike = None,
    *step_args: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """All-gather the token shards of the ``owners`` local row owners.

    Owner o (4: the local rank pairs; 8: the local ranks) holds rows o
    * r .. o * r + r - 1 in ``source`` [r, hidden]; every rank ends
    with all of them in ``destination``. Each rank sends to its
    counterpart in every other owner (rank ^ k * 8 / owners).
    ``packed`` stages through flat transport wires when a source row
    slice crosses a native BF16 tile.
    """
    owners, packed = (
        (*step_args[0:1], stage_kwargs.get("owners", 4))[0],
        stage_kwargs.get("packed", False),
    )
    rank = jax.lax.axis_index("tp")
    stride = 8 // owners
    owner = rank % 8 // stride
    rows_per_owner = source.shape[0]
    if packed or rows_per_owner == 1 or rows_per_owner & rows_per_owner - 1:
        np.testing.assert_(scratch is not None)
        payload_rows = rows_per_owner * source.shape[1] // 128
        source_wire = (
            scratch.at[owner].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
        )
        source_wire[...] = source[...].reshape(payload_rows, 128)
        copies = []
        for offset in range(owners - 1, 0, -1):
            copy = tpu.make_async_remote_copy(
                source_wire,
                source_wire,
                sends.at[offset - 1],
                recvs.at[offset - 1],
                device_id=(rank ^ stride * offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        wires = tuple(
            scratch.at[source_owner].bitcast(jnp.bfloat16)[:payload_rows]
            for source_owner in range(owners)
        )
        if rows_per_owner == 1:
            destination[...] = jnp.stack(
                tuple(wire.reshape(source.shape[1]) for wire in wires)
            )
        else:
            destination[...] = jnp.concatenate(
                tuple(wire.reshape(rows_per_owner, source.shape[1]) for wire in wires),
                axis=0,
            )
        return
    for owner_index in range(owners):

        @pl.when(owner == owner_index)
        def gather_owner(owner_index: TensorLike = owner_index) -> None:
            """Compute and stage Pallas VMEM state for gather owner."""
            row_slice = pl.ds(owner_index * rows_per_owner, rows_per_owner)
            destination[row_slice] = source[...]
            copies = []
            for offset in range(owners - 1, 0, -1):
                copy = tpu.make_async_remote_copy(
                    source,
                    destination.at[row_slice],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ stride * offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()


MOE = (
    "router",
    "router_bias",
    "latent_down",
    "latent_up",
    "latent_norm",
    "shared_gu",
    "shared_down",
    "expert_gu",
    "expert_gus",
    "expert_down",
    "expert_ds",
)
KDA = ("k_gate", "k_projection", "k_fb", "k_o", "k_conv", "k_a_log", "k_dt", "k_norm")
MLA = ("m_qa", "m_ka", "m_qb", "m_kb", "m_gate", "m_o", "m_qnorm", "m_knorm")
NORMS = ("attn_res", "mlp_res", "attn_norm", "ffn_norm")
EPILOGUE = ("output_res", "final_norm")
DENSE = ("dense_gu", "dense_down")
NAMES = MOE + KDA + MLA + NORMS


def _residual_mixture_row(
    block_values: TensorLike,
    prefix_value: TensorLike,
    folded: TensorLike,
    active_block_count: TensorLike,
    **stage_kwargs: TensorLike,
) -> TensorLike:
    """Compute one row of the learned residual mixture."""
    epsilon, relaxed_reduction_order = (
        stage_kwargs["epsilon"],
        stage_kwargs.get("relaxed_reduction_order", False),
    )

    def normalized_scores(values: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for normalized scores."""
        scores = jnp.sum(values * folded[...], axis=1, keepdims=True)
        return scores * jax.lax.rsqrt(
            jnp.mean(values * values, axis=1, keepdims=True) + epsilon
        )

    if relaxed_reduction_order:
        block_values = block_values.astype(jnp.float32)
        prefix_value = prefix_value.astype(jnp.float32)
        scores = jnp.concatenate(
            (normalized_scores(block_values), normalized_scores(prefix_value)), axis=0
        )
        values = block_values
    else:
        values = jnp.concatenate((block_values, prefix_value), axis=0).astype(
            jnp.float32
        )
        scores = jnp.sum(values * folded[...], axis=1, keepdims=True)
        scores *= jax.lax.rsqrt(
            jnp.mean(values * values, axis=1, keepdims=True) + epsilon
        )
    residual_indices = jax.lax.broadcasted_iota(jnp.int32, (9, 1), 0)
    active = (residual_indices < active_block_count) | (
        residual_indices == _ROUTED_EXPERTS_PER_TOKEN
    )
    probabilities = jax.nn.softmax(jnp.where(active, scores, -jnp.inf), axis=0)
    if relaxed_reduction_order:
        mixed = (
            jnp.sum(block_values * probabilities[:8], axis=0, keepdims=True)
            + prefix_value * probabilities[8:9]
        )
    else:
        mixed = jnp.sum(values * probabilities, axis=0, keepdims=True)
    return mixed.astype(jnp.bfloat16)


def _residual_mixture_by_row(
    blocks: TensorLike,
    prefix: TensorLike,
    destination: TensorLike,
    folded: TensorLike,
    active_block_count: TensorLike,
    **stage_kwargs: TensorLike,
) -> None:
    """Compute a residual mixture for a batch-leading input."""
    batch_size, epsilon, relaxed_reduction_order = (
        stage_kwargs["batch_size"],
        stage_kwargs["epsilon"],
        stage_kwargs.get("relaxed_reduction_order", False),
    )
    for batch_index in range(batch_size):
        destination[batch_index : batch_index + 1] = _residual_mixture_row(
            blocks[batch_index],
            prefix[batch_index : batch_index + 1],
            folded,
            active_block_count,
            epsilon=epsilon,
            relaxed_reduction_order=relaxed_reduction_order,
        )


POOLED_WEIGHTS = _config().pooled_weights
COMPACT_ROWS = 8
SINGLETON_BLOCK = _config().singleton_block
STAGE_BLOCKS = _config().stage_blocks
MXFP4_DOT_TILE = _config().mxfp4_dot_tile
np.testing.assert_(
    not (MXFP4_DOT_TILE is not None and (MXFP4_DOT_TILE < 0 or MXFP4_DOT_TILE % 128)),
    "KernelConfig.mxfp4_dot_tile must be zero or a positive multiple of 128",
)
MXFP4_BLOCK = _config().mxfp4_block
MXFP4_PAIR_TILE = _config().mxfp4_pair_tile
MXFP4_STATIC_TILE = _config().mxfp4_static_tile
B16_COMPACT_EXPERTS = _config().b16_compact_experts


def _compact_rows(batch_size: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for compact rows."""
    if (
        batch_size == _EXPERT_GROUP_STRIDE
        and B16_COMPACT_EXPERTS
        and (not COMPACT_ONLY)
    ):
        return 8
    return COMPACT_ROWS if batch_size >= _KDA_GROUP_WIDTH else 0


COMPACT_ONLY = _config().compact_only
SHARED_EXPERT_FINISHERS = _config().shared_expert_finishers
COMPACT_OVERFLOW = _config().compact_overflow
SHARED_SHARDS = _config().shared_shards
KDA_STAGED = _config().kda_staged
DEBUG_SKIP = set(_config().debug_skip)


class _RowFunction:
    """Compute and stage Pallas VMEM state for RowFunction.

    ``f[rows]`` evaluates ``f(rows)``: a per-chunk value computed from
    refs.
    """

    def __init__(self, function: TensorLike) -> None:
        """Initialize instance state."""
        self.function = function

    def __getitem__(self, rows: TensorLike) -> TensorLike:
        """Evaluate indexed item."""
        return self.function(rows)


def _shared_shards(batch_size: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for shared shards."""
    if SHARED_SHARDS is None:
        return batch_size == _VREG_LANE_COUNT
    return SHARED_SHARDS and batch_size > _KDA_GROUP_WIDTH


def _use_compact_overflow(
    batch_size: TensorLike, *, group_pending_routes: TensorLike
) -> TensorLike:
    """Compute and stage Pallas use compact overflow."""
    return (
        COMPACT_OVERFLOW
        and (not COMPACT_ONLY)
        and SHARED_EXPERT_FINISHERS
        and (
            batch_size in (32, 64, 96, 128)
            or (batch_size == _EXPERT_GROUP_STRIDE and B16_COMPACT_EXPERTS)
        )
        and (COMPACT_ROWS == _ROUTED_EXPERTS_PER_TOKEN)
        and group_pending_routes
    )


def _expert_accumulator_rows(
    batch_size: TensorLike, *, group_pending_routes: TensorLike = False
) -> TensorLike:
    """Compute and stage Pallas VMEM state for expert accumulator rows.

    (gate/up, down) accumulator rows: the batch, or only the compact
    tiles.
    """
    capacity = _compact_rows(batch_size)
    if capacity and (
        COMPACT_ONLY
        or _use_compact_overflow(batch_size, group_pending_routes=group_pending_routes)
    ):
        return (2 * capacity, capacity)
    return (batch_size, batch_size)


MOE_PREPARE_WEIGHTS = ("latent_down", "shared_gu", "shared_down")
MLA_EXTRA_WEIGHTS = ("m_qb", "m_kb", "m_gate")


def _pool_layout(local_weights: TensorLike, **stage_kwargs: TensorLike) -> TensorLike:
    """Byte offsets of every view in the projection/expert pool.

    Region A (from 0) holds, one phase at a time: the next layer's
    attention projections (KDA k_gate..k_o, or MLA q/kv-down),
    prefetched after the experts finish; the MLA output projection
    over the consumed q-down projection; the four routed-expert slots
    during the MoE. Region B (after A) holds the MLA q/kv-up and gate
    projections during attention, the latent-down and shared-expert
    weights during routing, and latent-up in the MoE tail (which
    overlaps the next layer's attention prefetch into A).
    """
    alias_expert_scales, pooled, managed, dense_bytes, arena_extra = (
        stage_kwargs["alias_expert_scales"],
        stage_kwargs["pooled"],
        stage_kwargs.get("managed", False),
        stage_kwargs.get("dense_bytes", 0),
        stage_kwargs.get("arena_extra", 0),
    )

    def storage(name: str, *, experts: bool = False) -> int:
        """Compute and stage Pallas VMEM state for storage."""
        value = local_weights[name]
        shape = (4, *value.shape[1:]) if experts else value.shape
        return pool_alias.storage_size(shape, value.dtype)

    expert_names = (
        ("expert_gu", "expert_gus", "expert_down", "expert_ds")
        if alias_expert_scales
        else ("expert_gu", "expert_down")
    )
    expert_bytes = sum(storage(name, experts=True) for name in expert_names)
    attention_bytes = max(
        sum(
            math.prod(local_weights[name].shape) * local_weights[name].dtype.itemsize
            for name in names
        )
        for names in (KDA[:4], MLA[:2])
    )
    layout = {"pool_bytes": max(expert_bytes, attention_bytes)}
    if not pooled:
        return layout
    region_b = -(-layout["pool_bytes"] // 4096) * 4096
    layout["attention_bytes"] = max(
        sum(storage(name) for name in names) for names in (KDA[:4], MLA[:2])
    )
    if managed:
        region_b = (
            -(-max(expert_bytes, layout["attention_bytes"], dense_bytes) // 4096) * 4096
        )
        region_b += arena_extra
        layout["arena_bytes"] = region_b
    end = region_b
    for names in (MLA_EXTRA_WEIGHTS, MOE_PREPARE_WEIGHTS, ("latent_up",)):
        offset = region_b
        for name in names:
            layout[name] = offset
            offset += storage(name)
        end = max(end, offset)
    layout["m_o"] = 0
    np.testing.assert_(
        not storage("m_o") > storage("m_qa"),
        "The MLA output projection must fit over the q-down projection",
    )
    layout["pool_bytes"] = end
    return layout


def _vmem_bytes(spec: TensorLike) -> int:
    """Tiled VMEM bytes of a scratch spec (0 for semaphores and SMEM)."""
    if str(getattr(spec, "memory_space", "")) != "vmem":
        return 0
    shape, itemsize = (tuple(spec.shape), jnp.dtype(spec.dtype).itemsize)
    if len(shape) < _PAIR_TUPLE_LEN:
        return -(-math.prod(shape) * itemsize // 512) * 512
    packing = max(1, 4 // itemsize)
    rows, tile_rows = (shape[-2], 8 * packing)
    if rows < tile_rows:
        padded = max(packing, 1 << (rows - 1).bit_length())
    else:
        padded = -(-rows // tile_rows) * tile_rows
    return math.prod(shape[:-2]) * padded * (-(-shape[-1] // 128) * 128) * itemsize


VMEM_LIMIT_BYTES = _config().vmem_limit_mb * 1024**2
VMEM_SPILL_RESERVE_BYTES = int(_config().vmem_spill_reserve_mb * 1024**2)


def _vmem_budget(
    persistent: TensorLike,
    regions: TensorLike,
    *,
    limit: TensorLike = None,
    spill_reserve: TensorLike = None,
) -> TensorLike:
    """Check a static VMEM plan and render it as a table.

    ``persistent`` is [(name, spec)], live for the whole kernel.
    ``regions`` is {region: ([(name, spec)] held for the region,
    [[(name, spec)], ...] nested alternatives)}; regions never overlap
    each other, and a region's nested alternatives never overlap each
    other. Returns (report, peak bytes).
    """
    limit = VMEM_LIMIT_BYTES if limit is None else limit
    spill_reserve = VMEM_SPILL_RESERVE_BYTES if spill_reserve is None else spill_reserve

    def mib(value: int) -> str:
        """Format a byte count in MiB."""
        return f"{value / 2**20:8.2f}"

    lines = ["persistent VMEM (MiB):"]
    persistent_bytes = 0
    for name, spec in persistent:
        size = _vmem_bytes(spec)
        persistent_bytes += size
        if size >= 64 * 1024:
            lines.append(
                str.join(
                    "",
                    (
                        "  ",
                        f"{mib(size)}",
                        "  ",
                        f"{name}",
                        " ",
                        f"{tuple(spec.shape)}",
                        " ",
                        f"{jnp.dtype(spec.dtype).name}",
                    ),
                )
            )
    lines.append(f"  {mib(persistent_bytes)}  total (entries under 64 KiB included)")
    peaks = {}
    for region, (held, nested) in regions.items():
        held_bytes = sum((_vmem_bytes(spec) for _, spec in held))
        nested_bytes = [
            sum((_vmem_bytes(spec) for _, spec in alternative))
            for alternative in nested
        ]
        peaks[region] = held_bytes + max(nested_bytes, default=0)
        lines.append(f"scoped region {region}: peak {mib(peaks[region]).strip()} MiB")
        for name, spec in held:
            if _vmem_bytes(spec) >= 64 * 1024:
                lines.append(f"  {mib(_vmem_bytes(spec))}  {name}")
        for alternative, size in zip(nested, nested_bytes):
            if size:
                lines.append(
                    str.join(
                        "",
                        (
                            "  ",
                            f"{mib(size)}",
                            "  nested: ",
                            f"{', '.join((name for name, _ in alternative))}",
                        ),
                    )
                )
    worst = max(peaks, key=lambda region_key: peaks[region_key]) if peaks else None
    peak = persistent_bytes + (peaks[worst] if worst else 0)
    lines.append(
        str.join(
            "",
            (
                "peak ",
                f"{mib(peak).strip()}",
                " MiB (persistent + ",
                f"{worst}",
                ") + spill reserve ",
                f"{mib(spill_reserve).strip()}",
                " of ",
                f"{mib(limit).strip()}",
                " MiB",
            ),
        )
    )
    return ("\n".join(lines), peak)


def _mla_scoped_scratch(
    local_weights: TensorLike, **stage_kwargs: TensorLike
) -> TensorLike:
    """Compute and stage Pallas VMEM state for mla scoped scratch.

    Transient MLA resources allocated once around each layer
    invocation (placeholders for the workspace's cache fields,
    ``mla_dcp.scoped_scratch`` last; ``dcp_heads`` per rank).
    """
    batch_size, sequence_rows, dcp_heads, dcp_dtype, dcp_page_tokens = (
        stage_kwargs["batch_size"],
        stage_kwargs["sequence_rows"],
        stage_kwargs.get("dcp_heads", 3),
        stage_kwargs.get("dcp_dtype", jnp.bfloat16),
        stage_kwargs.get("dcp_page_tokens", 256),
    )
    cache = (
        tpu.VMEM((8, 128), jnp.bfloat16),
        tpu.VMEM((8, 128), jnp.bfloat16),
        tpu.SemaphoreType.DMA((1,)),
    )
    return (
        *(
            ()
            if POOLED_WEIGHTS
            else (
                tpu.VMEM(local_weights[name].shape, local_weights[name].dtype)
                for name in MLA[2:6]
            )
        ),
        *cache,
        tpu.VMEM((batch_size, 3, 256), jnp.bfloat16),
        tpu.VMEM((batch_size, 384), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 256), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 128), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 1), jnp.float32),
        tpu.VMEM((batch_size, 3, 1), jnp.float32),
        tpu.VMEM((batch_size, 3, 128), jnp.float32),
        *mla_dcp.scoped_scratch(
            batch_size,
            heads=dcp_heads,
            cache_dtype=dcp_dtype,
            page_tokens=dcp_page_tokens,
            sequence_rows=sequence_rows,
            **mla_dcp.page_plan(batch_size, dcp_heads, sequence_rows, dcp_dtype),
        ),
    )


def _moe_scoped_scratch(
    local_weights: TensorLike, **stage_kwargs: TensorLike
) -> TensorLike:
    """Compute and stage Pallas VMEM state for moe scoped scratch.

    Transient MoE resources not held in the shared projection/expert
    pool.
    """
    (
        batch_size,
        alias_expert_scales,
        token_scatter_output,
        token_scatter_owners,
        shard_routing,
        group_pending_routes,
    ) = (
        stage_kwargs["batch_size"],
        stage_kwargs["alias_expert_scales"],
        stage_kwargs["token_scatter_output"],
        stage_kwargs.get("token_scatter_owners", 4),
        stage_kwargs.get("shard_routing", False),
        stage_kwargs.get("group_pending_routes", False),
    )
    gate_up = local_weights["expert_gu"]
    down = local_weights["expert_down"]
    latent_size = local_weights["latent_norm"].shape[-1]
    expert_input_size = gate_up.shape[1] * 8
    expert_width = down.shape[1] * 8
    expert_output_size = down.shape[-1]
    accumulator_rows = _expert_accumulator_rows(
        batch_size, group_pending_routes=group_pending_routes
    )
    scale_scratch = (
        ()
        if alias_expert_scales
        else (
            tpu.VMEM(
                (4, *local_weights["expert_gus"].shape[1:]),
                local_weights["expert_gus"].dtype,
            ),
            tpu.VMEM(
                (4, *local_weights["expert_ds"].shape[1:]),
                local_weights["expert_ds"].dtype,
            ),
        )
    )
    owners = token_scatter_owners
    output_shape = (
        (owners + 8, batch_size // owners * 1024 // 128, 128)
        if token_scatter_output
        else (batch_size, 8192)
    )
    return (
        *(
            ()
            if POOLED_WEIGHTS
            else (
                tpu.VMEM(
                    local_weights["latent_up"].shape, local_weights["latent_up"].dtype
                ),
            )
        ),
        tpu.SemaphoreType.DMA,
        tpu.SemaphoreType.DMA((4, 4)),
        *scale_scratch,
        tpu.VMEM((batch_size, 128), jnp.int32),
        tpu.VMEM((batch_size, 128), jnp.float32),
        tpu.VMEM((batch_size, 128), jnp.int32),
        tpu.VMEM((batch_size, 128), jnp.float32),
        tpu.VMEM((batch_size, 1024), jnp.float32),
        tpu.VMEM(
            (batch_size, 1024 if _shared_shards(batch_size) else 7168), jnp.float32
        ),
        tpu.VMEM((batch_size, latent_size), jnp.float32),
        tpu.VMEM((accumulator_rows[0], 4 * expert_width), jnp.float32),
        tpu.VMEM((accumulator_rows[1], expert_output_size), jnp.float32),
        tpu.VMEM((batch_size, 128), jnp.int32),
        *(
            (tpu.VMEM((_compact_rows(batch_size), expert_input_size), jnp.float32),) * 2
            if _compact_rows(batch_size)
            else (tpu.VMEM((1, expert_input_size), jnp.bfloat16),) * 2
        ),
        tpu.SemaphoreType.DMA((10,)),
        tpu.SemaphoreType.DMA((10,)),
        tpu.VMEM(output_shape, jnp.bfloat16),
        tpu.SemaphoreType.DMA((7,) if token_scatter_output else (2,)),
        tpu.SemaphoreType.DMA((7,) if token_scatter_output else (2,)),
        tpu.SemaphoreType.DMA((2,)),
        tpu.SemaphoreType.DMA((2,)),
        *(
            (
                tpu.VMEM((batch_size // 4, 128), jnp.float32),
                tpu.VMEM((batch_size, 128), jnp.float32),
            )
            if shard_routing
            else ()
        ),
    )


SHARD_ROUTING = _config().shard_routing


def _stack_stage_cs(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cs."""
    (
        (attention_dp, batch_size, block_tables, dcp_pool_map),
        (include_layer_zero, sequence_rows, state_row, states, weights),
    ) = stage_ctx
    np.testing.assert_(
        bool(1 <= batch_size <= MAX_BATCH),
        str.join(
            "",
            (
                "The fused decoder stack supports batch sizes ",
                "from 1 to ",
                f"{MAX_BATCH}",
            ),
        ),
    )
    np.testing.assert_(
        not (include_layer_zero and any(name not in weights for name in DENSE)),
        "include_layer_zero needs the dense MLP weights",
    )
    select_state_row = state_row is not None
    np.testing.assert_(
        not (block_tables is None or dcp_pool_map is None),
        "The stack takes block_tables and dcp_pool_map (vLLM's unified pools)",
    )
    pool_map = tuple(dcp_pool_map)
    np.testing.assert_(
        not (
            len(dcp_pool_map) != MLA_LAYERS or len(states) != 2 + 1 + max(dcp_pool_map)
        ),
        "The stack takes (conv, recurrent, *pools) and a pool per MLA layer",
    )
    layout = cache_layout(str(states[2].dtype))
    np.testing.assert_(
        not any(
            pool.shape[1:] != layout.block_shape or pool.dtype != states[2].dtype
            for pool in states[2:]
        ),
        f"unified pools must match {layout.block_shape} {layout.dtype}",
    )
    np.testing.assert_(
        not (
            block_tables.shape[0] != batch_size or block_tables.ndim != _PAIR_TUPLE_LEN
        ),
        "block_tables must be [batch_size, pages]",
    )
    np.testing.assert_(
        not (attention_dp != 1 and mla_dcp.GROUP * attention_dp != _KDA_GROUP_WIDTH),
        str.join(
            "",
            (
                "Attention DP needs DCP over each DP group ",
                "(KernelConfig.dcp_group = 32 / dp)",
            ),
        ),
    )
    attention_dp_ownership = attention_dp > 1
    paged_pages = int(block_tables.shape[1])
    np.testing.assert_(
        not (select_state_row and sequence_rows != batch_size),
        "state_row selection needs one sequence per tile",
    )
    first_layer = 0 if include_layer_zero else 1
    return (
        (attention_dp_ownership, first_layer, paged_pages, pool_map),
        select_state_row,
    )


def _stack_stage_ct(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage ct."""
    (
        (attention_token_scatter, aux_capture, aux_hidden_layers, batch_size),
        (include_layer_zero, layers, mla_cache_batch_tile_size),
        (residual_reduction_order, sequence_rows),
    ) = stage_ctx
    dense_names = DENSE if include_layer_zero else ()
    np.testing.assert_(
        not (not 1 <= sequence_rows <= batch_size or batch_size % sequence_rows),
        "sequence_rows must divide the batch",
    )
    if sequence_rows > 1:
        mla_cache_batch_tile_size = batch_size
    aux_hidden_layers = tuple(aux_hidden_layers)
    np.testing.assert_(
        not (
            any(not 1 <= index <= layers - 1 for index in aux_hidden_layers)
            or len(set(aux_hidden_layers)) != len(aux_hidden_layers)
        ),
        str.join(
            "",
            (
                "aux_hidden_layers are distinct layer ids in ",
                "1..layers-1; id L captures the residual stream ",
                "entering layer L",
            ),
        ),
    )
    np.testing.assert_(
        not aux_capture not in ("layer", "attention", "mixture_input"),
        "aux_capture must be 'layer', 'attention' or 'mixture_input'",
    )
    np.testing.assert_(
        not (aux_capture != "layer" and attention_token_scatter),
        "Attention-point aux capture needs complete prefix rows",
    )
    b4_attention_gather = (
        B4_ATTENTION_GATHER
        and batch_size == _TP_RING_SIZE
        and (not attention_token_scatter)
    )
    np.testing.assert_(
        not residual_reduction_order not in ("reference", "row", "relaxed"),
        "residual_reduction_order must be 'reference', 'row', or 'relaxed'",
    )
    mla_cache_batch_tile_size = mla_cache_batch_tile_size or batch_size
    np.testing.assert_(
        bool(1 <= mla_cache_batch_tile_size <= batch_size),
        "MLA cache batch tile size must be within the batch",
    )
    np.testing.assert_(
        not batch_size % mla_cache_batch_tile_size,
        "MLA cache batch tile size must divide the batch",
    )
    return (
        (aux_hidden_layers, b4_attention_gather, dense_names),
        mla_cache_batch_tile_size,
    )


def _stack_stage_dc(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage dc."""
    (
        (attention_dp, attention_dp_ownership, attention_token_scatter),
        (batch_size, mla_cache_batch_tile_size, mla_commit_tile_size),
        (moe_routed_reduce_scatter, moe_token_scatter_output),
        (select_state_row, sequence_parallel_pre_attention),
        (sequence_parallel_size, sequence_rows, shard_routing),
    ) = stage_ctx
    np.testing.assert_(
        not (
            attention_token_scatter
            and batch_size not in (4, *SCATTER_BATCHES)
            and (not (attention_dp > 1 and batch_size == _PAIR_TUPLE_LEN))
        ),
        str.join(
            "",
            (
                "Attention token scatter requires batch size 4, ",
                "8 or 16, or attention DP's B2",
            ),
        ),
    )
    (owned_rows, shard_routing, sp_owners, tokens_per_pair) = _stack_stage_cw(
        (
            (attention_dp, attention_token_scatter, batch_size),
            (moe_routed_reduce_scatter, moe_token_scatter_output),
            (sequence_parallel_pre_attention, sequence_parallel_size),
            shard_routing,
        )
    )
    _stack_stage_db(
        (
            (attention_dp, batch_size, select_state_row),
            (sequence_parallel_pre_attention, sequence_rows, sp_owners),
        )
    )
    (commit_rows, dp_held_rows, group_rows, group_rows_per_pair, sp8_residual) = (
        _stack_stage_cx(
            (
                (attention_dp, attention_dp_ownership, batch_size),
                (mla_cache_batch_tile_size, mla_commit_tile_size, sequence_rows),
                sp_owners,
            )
        )
    )
    pair_split_gather = (
        attention_dp_ownership
        and (not sp8_residual)
        and (sp_owners != 1)
        and (dp_held_rows > _PAIR_TUPLE_LEN)
        and (sequence_rows == 1)
    )
    return (
        (commit_rows, dp_held_rows, group_rows, group_rows_per_pair, owned_rows),
        (pair_split_gather, shard_routing, sp8_residual, sp_owners, tokens_per_pair),
    )


def _stack_stage_cw(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cw."""
    (
        (attention_dp, attention_token_scatter, batch_size),
        (moe_routed_reduce_scatter, moe_token_scatter_output),
        (sequence_parallel_pre_attention, sequence_parallel_size),
        shard_routing,
    ) = stage_ctx
    np.testing.assert_(
        not (
            moe_token_scatter_output
            and (
                not (
                    attention_token_scatter
                    and (
                        batch_size in SCATTER_BATCHES
                        or sequence_parallel_size < _TP_RING_SIZE
                    )
                )
            )
        ),
        "MoE token-scattered output requires B8 attention token scatter",
    )
    np.testing.assert_(
        not (moe_routed_reduce_scatter and batch_size not in (6, *SCATTER_BATCHES)),
        "MoE routed reduce-scatter requires batch size 6, 8 or 16",
    )
    sp_owners, tokens_per_pair = _stack_stage_cu(
        (
            (attention_dp, attention_token_scatter, batch_size),
            (moe_token_scatter_output, sequence_parallel_pre_attention),
            sequence_parallel_size,
        )
    )
    owned_rows = batch_size // sp_owners if attention_token_scatter else 0
    shard_routing = SHARD_ROUTING if shard_routing is None else shard_routing
    shard_routing = (
        shard_routing and moe_token_scatter_output and (sp_owners == _TP_RING_SIZE)
    )
    np.testing.assert_(
        not attention_dp not in (1, 2, 4), "Attention DP supports dp=1, dp=2 or dp=4"
    )
    return (owned_rows, shard_routing, sp_owners, tokens_per_pair)


def _stack_stage_cu(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cu."""
    (
        (attention_dp, attention_token_scatter, batch_size),
        (moe_token_scatter_output, sequence_parallel_pre_attention),
        sequence_parallel_size,
    ) = stage_ctx
    np.testing.assert_(
        not (sequence_parallel_pre_attention and (not moe_token_scatter_output)),
        "Sequence-parallel pre-attention requires B8 MoE token scatter",
    )
    sp_owners = sequence_parallel_size
    tokens_per_pair = (
        batch_size // (sp_owners if attention_dp > 1 else 4)
        if attention_token_scatter
        else 0
    )
    np.testing.assert_(
        not sp_owners not in (1, 4, 8),
        str.join(
            "",
            (
                "sequence_parallel_size must be 1 (attention DP2 ",
                "B2/B4), 4 (rank pairs) or 8 (ranks)",
            ),
        ),
    )
    np.testing.assert_(
        not (
            sp_owners == 1
            and (not (attention_dp == _PAIR_TUPLE_LEN and batch_size in (2, 4)))
        ),
        "One row owner is attention DP2's B2/B4 layout",
    )
    np.testing.assert_(
        not (
            sp_owners == _ROUTED_EXPERTS_PER_TOKEN
            and (
                not (
                    attention_token_scatter
                    and batch_size % 8 == 0
                    and (attention_dp == 1)
                )
            )
        ),
        str.join(
            "",
            (
                "Eight row owners need the token-scatter layout, ",
                "a multiple of 8 rows and no attention DP",
            ),
        ),
    )
    return (sp_owners, tokens_per_pair)


def _stack_stage_db(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for stack stage db."""
    (
        (attention_dp, batch_size, select_state_row),
        (sequence_parallel_pre_attention, sequence_rows, sp_owners),
    ) = stage_ctx
    np.testing.assert_(
        not (
            attention_dp > 1
            and (
                not (
                    batch_size in (2, 4, 8, 16, 32, 64, 96, 128)
                    and batch_size % (sp_owners * attention_dp) == 0
                    and sequence_parallel_pre_attention
                    and (batch_size // attention_dp % sequence_rows == 0)
                    and (not select_state_row)
                    and POOLED_WEIGHTS
                )
            )
        ),
        str.join(
            "",
            (
                "Attention DP needs a supported batch divisible ",
                "by 4*DP, the sequence-parallel token-scatter ",
                "layout, whole sequences in each group's rows ",
                "and pooled weights",
            ),
        ),
    )


def _stack_stage_cx(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cx."""
    (
        (attention_dp, attention_dp_ownership, batch_size),
        (mla_cache_batch_tile_size, mla_commit_tile_size, sequence_rows),
        sp_owners,
    ) = stage_ctx
    np.testing.assert_(
        not (
            attention_dp > 1
            and (
                batch_size not in (32, 64, 96, 128)
                and (
                    not (
                        attention_dp == _PAIR_TUPLE_LEN and batch_size in (2, 4, 8, 16)
                    )
                )
            )
        ),
        "Attention DP supports DP2 at B2..B128 or DP4 at B32..B128",
    )
    commit_rows = mla_commit_tile_size or mla_cache_batch_tile_size
    np.testing.assert_(
        not batch_size % commit_rows, "mla_commit_tile_size must divide the batch"
    )
    group_rows = batch_size // attention_dp
    group_rows_per_pair = group_rows // sp_owners
    dp_held_rows = batch_size if sp_owners == 1 else group_rows_per_pair
    sp8_residual = (
        attention_dp_ownership
        and attention_dp == _PAIR_TUPLE_LEN
        and (batch_size == _VREG_LANE_COUNT)
        and (sequence_rows == 1)
    )
    return (commit_rows, dp_held_rows, group_rows, group_rows_per_pair, sp8_residual)


def _stack_stage_cv(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cv."""
    (
        (attention_dp, attention_dp_ownership, attention_projection_schedule),
        (batch_size, commit_rows, group_rows, mla_cache_batch_tile_size),
        (prefix, sequence_rows, sp_owners),
    ) = stage_ctx
    default_streams = (
        {
            **DCP_B32_WEIGHT_STREAMS,
            **(DCP_B32_DECODE_RINGS if sequence_rows == 1 else {}),
        }
        if batch_size in (32, 64)
        and attention_dp == 1
        and (not _config().weight_streams)
        else {}
    )
    projection_streams = _attention_projection_streams(
        attention_dp, attention_projection_schedule
    )
    conflicts = projection_streams.keys() & WEIGHT_STREAMS.keys()
    np.testing.assert_(
        not conflicts,
        str.join(
            "",
            (
                "Projection schedule conflicts with ",
                "KernelConfig.weight_streams: ",
                f"{sorted(conflicts)}",
            ),
        ),
    )
    kda_shard_swap = attention_dp_ownership and sp_owners == 1
    stream_config = {
        **DP_WEIGHT_STREAMS.get(attention_dp, {}),
        **(
            _dp_dcp_weight_streams(attention_dp, group_rows) if attention_dp > 1 else {}
        ),
        **({"k_projection": (attention_dp, 1)} if kda_shard_swap else {}),
        **default_streams,
        **WEIGHT_STREAMS,
        **projection_streams,
    }
    cache_handoff_head_count = max(
        3, 8 // math.gcd(mla_cache_batch_tile_size, commit_rows, 8)
    )
    hidden_size = prefix.shape[-1]
    blocks_block_major = (
        batch_size > _ROUTED_EXPERTS_PER_TOKEN or attention_dp_ownership
    )
    blocks_owned = batch_size > _EXPERT_GROUP_STRIDE
    return (
        (blocks_block_major, blocks_owned, cache_handoff_head_count, hidden_size),
        (kda_shard_swap, stream_config),
    )


def _stack_stage_cz(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cz."""
    (
        (attention_dp, attention_dp_ownership, attention_token_scatter),
        (batch_size, blocks_block_major, blocks_owned, dp_held_rows),
        (moe_token_scatter_output, owned_rows, select_state_row),
        (sequence_parallel_pre_attention, state_slots),
    ) = stage_ctx
    compact_outputs = (
        batch_size > _EXPERT_GROUP_STRIDE
        and attention_token_scatter
        and moe_token_scatter_output
    )
    block_rows_held = (
        dp_held_rows
        if attention_dp_ownership
        else owned_rows
        if blocks_owned
        else batch_size
    )
    owner_generic = blocks_owned and owned_rows % 16 == 0
    pool_managed = POOLED_WEIGHTS
    np.testing.assert_(
        not (blocks_block_major and (not sequence_parallel_pre_attention)),
        "Above eight rows the decoder needs sequence-parallel pre-attention",
    )
    (kda_row_chunk, persist_kda_state, slot_reads, stream_kda_state) = _stack_stage_cy(
        attention_dp, batch_size, select_state_row, state_slots
    )
    overlap_kda_state_reads = batch_size in (6, *SCATTER_BATCHES) or attention_dp > 1
    overlap_kda_state_writes = batch_size in (6, *SCATTER_BATCHES) or attention_dp > 1
    return (
        (block_rows_held, compact_outputs, kda_row_chunk, overlap_kda_state_reads),
        (overlap_kda_state_writes, owner_generic, persist_kda_state, pool_managed),
        (slot_reads, stream_kda_state),
    )


def _stack_stage_cy(
    attention_dp: TensorLike,
    batch_size: TensorLike,
    select_state_row: TensorLike,
    state_slots: TensorLike,
) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cy."""
    persist_kda_state = batch_size <= _TP_RING_SIZE and attention_dp == 1
    stream_kda_state = batch_size > _EXPERT_GROUP_STRIDE
    kda_row_chunk = _kda_row_chunk(batch_size, attention_dp)
    slot_reads = state_slots is not None
    np.testing.assert_(
        not (
            slot_reads
            and (
                not (
                    (attention_dp == 1 and (select_state_row or not persist_kda_state))
                    or (
                        attention_dp == _PAIR_TUPLE_LEN
                        and batch_size in (2, 4, 8, 16, 32, 64, 96, 128)
                    )
                    or (
                        attention_dp == _TP_RING_SIZE
                        and batch_size in (32, 64, 96, 128)
                    )
                )
            )
        ),
        str.join(
            "",
            (
                "state_slots needs DP1 with state_row or batch > ",
                "4, DP2 at B2..B128 or DP4 at B32..B128",
            ),
        ),
    )
    return (kda_row_chunk, persist_kda_state, slot_reads, stream_kda_state)


def _stack_stage_cr(stream_config: TensorLike, weights: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cr."""

    def shape(value: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for shape."""
        return jax.ShapeDtypeStruct(value.shape, value.dtype)

    local_weights = {
        name: jax.ShapeDtypeStruct(value.shape[1:], value.dtype)
        for name, value in weights.items()
    }

    def pooled_shape(name: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for pooled shape.

        Pool footprint of a layer weight: a ring stream's slot region
        (depth x its largest chunk along the streamed axis), else the
        weight.
        """
        shape = local_weights[name].shape
        if name not in STREAMED_WEIGHTS or name not in stream_config:
            return shape
        chunks, depth = stream_config[name]
        if chunks == depth:
            return shape
        axis = 1 if STREAMED_WEIGHTS[name][0] == "columns" else 0
        largest = _chunk_sizes(shape[axis], chunks, 128 if axis == 1 else 16)[0]
        return tuple(
            (
                depth * largest if index == axis else size
                for index, size in enumerate(shape)
            )
        )

    pool_weights = {
        name: jax.ShapeDtypeStruct(pooled_shape(name), value.dtype)
        for name, value in local_weights.items()
    }
    return (local_weights, pool_weights, shape)


def stack(
    prefix: TensorLike,
    embedding: TensorLike,
    weights: TensorLike,
    states: TensorLike,
    positions: TensorLike,
    **stage_kwargs: TensorLike,
) -> TensorLike:
    """Fuse the decoder layers while keeping a native batch tile in VMEM.

    The MLA cache is vLLM's unified pools, read in place with decode
    context parallelism (``mla_dcp``): ``states`` is ``(conv,
    recurrent, *pools)``, MLA layer ``m`` in
    ``pools[dcp_pool_map[m]]``, ``block_tables`` (int32 ``[batch_size,
    pages]``) the rows' tables. The pools stay read-only: in place of
    an updated cache the kernel returns each row's new latent for
    every MLA layer, ``[batch, 24, H, 256]`` / ``[batch, 24, H, 128]``
    (``H >= 3`` padded heads; ``paged.commit_latent_rows`` commits
    them). Nothing is aliased: XLA copies every operand a
    communicating kernel updates in place, so the KDA states come back
    as fresh outputs too.

    With the token-scatter / sequence-parallel layout,
    ``sequence_parallel_size`` local row owners (4: rank pairs; 8:
    single ranks; 1: every rank, at attention DP2's B2 and B4),
    replicated across the hosts, each own batch_size / size rows for
    the per-token work.

    Layers 1..L-1 by default; with ``include_layer_zero`` the loop
    starts at layer 0 (``prefix`` is then the embedding; its KDA runs
    on the common attention path and a dense MLP streamed through the
    weight pool replaces the mixture of experts), which needs
    ``weights["dense_gu"]`` and ``weights["dense_down"]``.
    ``state_row`` (a traced int32 scalar, one sequence per tile only)
    selects the row of the KDA input states to read for the sequence's
    first row instead of a pre-selected state copy.

    ``state_slots`` (int32 ``[batch_size]``): ``states[:2]`` are the
    whole KDA slot arrays ``[slots, 69, ...]`` and row ``b`` reads its
    input state from slot ``state_slots[b]`` (DP2 indices include the
    head-shard bank offset and follow shard-major group rows; with
    ``state_row``, slot ``state_slots[state_row]``), instead of the
    caller gathering the rows first. The updated states are still
    returned per row. Needs ``state_row`` or ``batch_size`` > 4 with
    DP1, B2..B128 with DP2 or B32..B128 with DP4.
    """
    (
        batch_size,
        layers,
        eps,
        residual_reduction_order,
        mla_cache_batch_tile_size,
        alias_expert_scales,
        attention_token_scatter,
        moe_token_scatter_output,
        moe_routed_reduce_scatter,
        sequence_parallel_pre_attention,
        sequence_rows,
        aux_hidden_layers,
        aux_capture,
        include_layer_zero,
        state_row,
        attention_dp,
        attention_projection_schedule,
        mla_commit_tile_size,
        sequence_parallel_size,
        shard_routing,
        block_tables,
        dcp_pool_map,
        dcp_owner_table,
        state_slots,
        live_rows,
    ) = (
        stage_kwargs["batch_size"],
        stage_kwargs.get("layers", 93),
        stage_kwargs.get("eps", 1e-05),
        stage_kwargs.get("residual_reduction_order", "reference"),
        stage_kwargs.get("mla_cache_batch_tile_size", _NONE_STAGE_VAL),
        stage_kwargs.get("alias_expert_scales", False),
        stage_kwargs.get("attention_token_scatter", False),
        stage_kwargs.get("moe_token_scatter_output", False),
        stage_kwargs.get("moe_routed_reduce_scatter", False),
        stage_kwargs.get("sequence_parallel_pre_attention", False),
        stage_kwargs.get("sequence_rows", 1),
        stage_kwargs.get("aux_hidden_layers", ()),
        stage_kwargs.get("aux_capture", "layer"),
        stage_kwargs.get("include_layer_zero", False),
        stage_kwargs.get("state_row", _NONE_STAGE_VAL),
        stage_kwargs.get("attention_dp", 1),
        stage_kwargs.get("attention_projection_schedule", "default"),
        stage_kwargs.get("mla_commit_tile_size", _NONE_STAGE_VAL),
        stage_kwargs.get("sequence_parallel_size", 4),
        stage_kwargs.get("shard_routing", _NONE_STAGE_VAL),
        stage_kwargs.get("block_tables", _NONE_STAGE_VAL),
        stage_kwargs.get("dcp_pool_map", _NONE_STAGE_VAL),
        stage_kwargs.get("dcp_owner_table", _NONE_STAGE_VAL),
        stage_kwargs.get("state_slots", _NONE_STAGE_VAL),
        stage_kwargs.get("live_rows", _NONE_STAGE_VAL),
    )
    (attention_dp_ownership, first_layer, paged_pages, pool_map), select_state_row = (
        _stack_stage_cs(
            (
                (attention_dp, batch_size, block_tables, dcp_pool_map),
                (include_layer_zero, sequence_rows, state_row, states, weights),
            )
        )
    )
    (aux_hidden_layers, b4_attention_gather, dense_names), mla_cache_batch_tile_size = (
        _stack_stage_ct(
            (
                (attention_token_scatter, aux_capture, aux_hidden_layers, batch_size),
                (include_layer_zero, layers, mla_cache_batch_tile_size),
                (residual_reduction_order, sequence_rows),
            )
        )
    )
    (
        (commit_rows, dp_held_rows, group_rows, group_rows_per_pair, owned_rows),
        (pair_split_gather, shard_routing, sp8_residual, sp_owners, tokens_per_pair),
    ) = _stack_stage_dc(
        (
            (attention_dp, attention_dp_ownership, attention_token_scatter),
            (batch_size, mla_cache_batch_tile_size, mla_commit_tile_size),
            (moe_routed_reduce_scatter, moe_token_scatter_output),
            (select_state_row, sequence_parallel_pre_attention),
            (sequence_parallel_size, sequence_rows, shard_routing),
        )
    )
    (
        (blocks_block_major, blocks_owned, cache_handoff_head_count, hidden_size),
        (kda_shard_swap, stream_config),
    ) = _stack_stage_cv(
        (
            (attention_dp, attention_dp_ownership, attention_projection_schedule),
            (batch_size, commit_rows, group_rows, mla_cache_batch_tile_size),
            (prefix, sequence_rows, sp_owners),
        )
    )
    (
        (block_rows_held, compact_outputs, kda_row_chunk, overlap_kda_state_reads),
        (overlap_kda_state_writes, owner_generic, persist_kda_state, pool_managed),
        (slot_reads, stream_kda_state),
    ) = _stack_stage_cz(
        (
            (attention_dp, attention_dp_ownership, attention_token_scatter),
            (batch_size, blocks_block_major, blocks_owned, dp_held_rows),
            (moe_token_scatter_output, owned_rows, select_state_row),
            (sequence_parallel_pre_attention, state_slots),
        )
    )
    (local_weights, pool_weights, shape) = _stack_stage_cr(stream_config, weights)
    mla_scoped_scratch = _mla_scoped_scratch(
        local_weights,
        batch_size=batch_size // attention_dp,
        sequence_rows=sequence_rows,
        dcp_dtype=states[2].dtype,
        dcp_page_tokens=states[2].shape[-3] * states[2].shape[-2],
        dcp_heads=3 * attention_dp,
    )
    moe_scoped_scratch = _moe_scoped_scratch(
        local_weights,
        batch_size=batch_size,
        alias_expert_scales=alias_expert_scales,
        token_scatter_output=moe_token_scatter_output,
        token_scatter_owners=sp_owners,
        shard_routing=shard_routing,
        group_pending_routes=batch_size in (6, *SCATTER_BATCHES)
        and residual_reduction_order == "relaxed",
    )

    def body(
        position_ref: TensorLike, state_row_ref: TensorLike, *refs: TensorLike
    ) -> None:
        """Compute and stage Pallas VMEM state for body."""
        return _stack_body_stage_cm(
            (
                (alias_expert_scales, attention_dp, attention_dp_ownership),
                (attention_token_scatter, aux_capture, aux_hidden_layers),
                (b4_attention_gather, batch_size, block_rows_held, blocks_block_major),
                (blocks_owned, cache_handoff_head_count, compact_outputs),
                (dcp_owner_table, dense_names, dense_slabs, dp_held_rows, eps),
                (expert_pool_bytes, first_layer, group_rows, group_rows_per_pair),
                (hidden_size, include_layer_zero, kda_row_chunk, kda_shard_swap),
                (layers, live_rows, mla_scoped_scratch, moe_routed_reduce_scatter),
                (moe_scoped_scratch, moe_token_scatter_output),
                (overlap_kda_state_reads, overlap_kda_state_writes, owned_rows),
                (owner_generic, paged_pages, pair_split_gather, persist_kda_state),
                (pool_layout, pool_managed, pool_map, pool_weights),
                (residual_reduction_order, select_state_row),
                (sequence_parallel_pre_attention, sequence_rows, shard_routing),
                (slot_reads, sp8_residual, sp_owners, states, stream_config),
                (stream_kda_state, tokens_per_pair),
            ),
            (position_ref, state_row_ref, refs),
        )

    args = (
        prefix,
        embedding,
        *(weights[name] for name in NAMES + EPILOGUE + dense_names),
        *states,
    )
    state_row_scalar = (
        jnp.asarray(state_row, jnp.int32).reshape(1)
        if select_state_row
        else jnp.zeros((1,), jnp.int32)
    )
    state_output_shapes = (
        jax.ShapeDtypeStruct((batch_size, *states[0].shape[1:]), states[0].dtype),
        jax.ShapeDtypeStruct((batch_size, *states[1].shape[1:]), states[1].dtype),
        jax.ShapeDtypeStruct(
            (batch_size, MLA_LAYERS, cache_handoff_head_count, 256), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct(
            (batch_size, MLA_LAYERS, cache_handoff_head_count, 128), jnp.bfloat16
        ),
    )
    output_shapes = (
        shape(prefix),
        jax.ShapeDtypeStruct((8, 128), jnp.uint8)
        if sequence_parallel_pre_attention
        else jax.ShapeDtypeStruct((batch_size, 8, hidden_size), jnp.bfloat16),
        *state_output_shapes,
        jax.ShapeDtypeStruct(
            (max(1, len(aux_hidden_layers)), batch_size, hidden_size), jnp.bfloat16
        ),
    )
    (dense_slab_bytes, dense_slabs, expert_pool_bytes, pool_bytes, pool_layout) = (
        _stack_stage_cq(
            (
                (alias_expert_scales, batch_size, hidden_size, include_layer_zero),
                (local_weights, pool_managed, pool_weights, weights),
            )
        )
    )
    scratch = _stack_stage_cp(
        (
            (attention_dp, attention_dp_ownership, attention_token_scatter),
            (b4_attention_gather, batch_size, block_rows_held, blocks_block_major),
            (compact_outputs, group_rows, hidden_size, local_weights),
            (moe_routed_reduce_scatter, owned_rows, pair_split_gather),
            (persist_kda_state, pool_bytes, pool_managed, prefix, sequence_rows),
            (states, stream_config, weights),
        )
    )
    (persistent, projection_reduction, scratch_names) = _stack_stage_da(
        (
            (attention_dp, b4_attention_gather, batch_size, group_rows),
            (moe_routed_reduce_scatter, output_shapes, pair_split_gather, scratch),
        )
    )
    regions = _stack_stage_cn(
        (
            (alias_expert_scales, attention_dp, batch_size),
            (cache_handoff_head_count, group_rows, kda_row_chunk, local_weights),
            (mla_scoped_scratch, moe_scoped_scratch, persist_kda_state),
            (projection_reduction, shard_routing, states, stream_kda_state),
        )
    )
    if include_layer_zero and batch_size > _ROUTED_EXPERTS_PER_TOKEN:
        regions["dense layer 0"] = (
            [
                (
                    "gate/up accumulator",
                    tpu.VMEM((batch_size, weights["dense_gu"].shape[1]), jnp.float32),
                ),
                ("down accumulator", tpu.VMEM((batch_size, hidden_size), jnp.float32)),
            ],
            [],
        )
    pool_layout, scratch = _stack_stage_co(
        (
            (alias_expert_scales, batch_size, dense_slab_bytes, dense_slabs),
            (hidden_size, persistent, pool_layout, pool_managed, pool_weights),
            (regions, scratch, scratch_names),
        )
    )
    return pl.pallas_call(
        body,
        name="kimi_decoder_stack",
        out_shape=output_shapes,
        input_output_aliases={},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3 + int(slot_reads) + int(live_rows is not None),
            in_specs=[pl.BlockSpec(memory_space=tpu.HBM) for _ in range(len(args))],
            out_specs=[
                pl.BlockSpec(memory_space=tpu.HBM),
                pl.BlockSpec(memory_space=tpu.VMEM),
                *(pl.BlockSpec(memory_space=tpu.HBM) for _ in range(5)),
            ],
            scratch_shapes=scratch,
        ),
        compiler_params=tpu.CompilerParams(
            collective_id=21,
            vmem_limit_bytes=_config().vmem_limit_mb * 1024**2,
            disable_bounds_checks=not _config().bounds_checks,
            shape_invariant_numerics=True,
            internal_scratch_in_bytes=_config().internal_scratch_kb * 1024
            if _config().internal_scratch_kb
            else None,
        ),
    )(
        positions,
        state_row_scalar,
        jnp.asarray(block_tables, jnp.int32).reshape(-1),
        *(
            (jnp.asarray(state_slots, jnp.int32).reshape(batch_size),)
            if slot_reads
            else ()
        ),
        *(
            (jnp.asarray(live_rows, jnp.int32).reshape(batch_size),)
            if live_rows is not None
            else ()
        ),
        *args,
    )


def _stack_body_stage_cm_stage_ew(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    ((batch_size, dense_names, live_rows, pool_map, refs, slot_reads), states) = (
        stage_ctx
    )
    dp_attended_vmem = dp_compressed_vmem = group_input_vmem = _NONE_STAGE_VAL
    group_positions_smem = _NONE_STAGE_VAL
    table_ref, refs = (refs[0], refs[1:])
    slots_ref = None
    if slot_reads:
        slots_ref, refs = (refs[0], refs[1:])
    active_rows = None
    if live_rows is not None:
        live_ref, refs = (refs[0], refs[1:])
        active_rows = jnp.stack([live_ref[b] for b in range(batch_size)]) != 0
    prefix_ref, embedding_ref, refs = (refs[0], refs[1], refs[2:])
    layer_weights = dict(zip(NAMES, refs[: len(NAMES)], strict=True))
    output_residual_ref, final_normalization_ref = refs[
        len(NAMES) : len(NAMES) + len(EPILOGUE)
    ]
    dense_start = len(NAMES) + len(EPILOGUE)
    dense_refs = dict(
        zip(
            dense_names, refs[dense_start : dense_start + len(dense_names)], strict=True
        )
    )
    state_start = dense_start + len(dense_names)
    state_count = len(states)
    convolution_ref, recurrent_ref = refs[state_start : state_start + 2]
    raw_pool_refs = refs[state_start + 2 : state_start + state_count]
    pool_ref = tuple(raw_pool_refs[index] for index in pool_map)
    output_refs = refs[state_start + state_count : state_start + state_count + 7]
    (
        output_ref,
        blocks_output_ref,
        convolution_output_ref,
        recurrent_output_ref,
        key_output_ref,
        value_output_ref,
        aux_output_ref,
    ) = output_refs
    return (
        (active_rows, aux_output_ref, blocks_output_ref, convolution_output_ref),
        (convolution_ref, dense_refs, dp_attended_vmem, dp_compressed_vmem),
        (embedding_ref, final_normalization_ref, group_input_vmem),
        (group_positions_smem, key_output_ref, layer_weights, output_ref),
        (output_residual_ref, pool_ref, prefix_ref, recurrent_output_ref),
        (recurrent_ref, refs, slots_ref, state_count, state_start, table_ref),
        value_output_ref,
    )


def _stack_body_stage_cm_stage_ex(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (attention_dp, b4_attention_gather, compact_outputs, dp_attended_vmem),
        (dp_compressed_vmem, group_input_vmem, group_positions_smem),
        (moe_routed_reduce_scatter, pair_split_gather, refs, select_state_row),
        (slot_reads, slots_ref, state_count, state_row_ref, state_start),
    ) = stage_ctx
    (
        prefix_vmem,
        blocks_vmem,
        normalized_vmem,
        attention_vmem,
        mixture_vmem,
        attention_residual_vmem,
        mlp_residual_vmem,
        attention_norm_vmem,
        ffn_norm_vmem,
        router_vmem,
        router_semaphores,
        projection_expert_pool_vmem,
        pool_semaphores,
        norm_semaphores,
        convolution_filter_vmem,
        exp_log_decay_rate_vmem,
        time_step_bias_vmem,
        output_normalization_vmem,
        convolution_history_vmem,
        recurrent_state_vmem,
        auxiliary_semaphores,
        query_norms_vmem,
        key_value_norms_vmem,
        routing_biases_vmem,
        latent_norms_vmem,
        metadata_semaphores,
        owned_rows_vmem,
        aux_semaphore,
        dense_semaphores,
        stream_semaphores,
        *common,
    ) = refs[state_start + state_count + 7 :]
    if attention_dp > 1:
        (
            group_input_vmem,
            dp_attended_vmem,
            group_positions_smem,
            _dp_query_latent_vmem,
            dp_compressed_vmem,
        ) = common[:5]
        common = common[5:]
    dcp_persistent = common[: len(mla_dcp.PERSISTENT_NAMES)]
    common = common[len(mla_dcp.PERSISTENT_NAMES) :]
    if _config().vmem_pad_kb:
        headroom_pad = common[-1]
        headroom_pad[...] = jnp.zeros(headroom_pad.shape, jnp.float32)
        common = common[:-1]
    else:
        headroom_pad = None
    (attention_vmem, communication_workspace, kda_state_source), selected_state_row = (
        _stack_body_stage_cm_stage_ef(
            (
                (attention_vmem, b4_attention_gather, common, compact_outputs),
                (moe_routed_reduce_scatter, normalized_vmem, pair_split_gather),
                (select_state_row, slot_reads, slots_ref, state_row_ref),
            )
        )
    )
    return (
        (attention_norm_vmem, attention_residual_vmem, attention_vmem),
        (aux_semaphore, auxiliary_semaphores, blocks_vmem, communication_workspace),
        (convolution_filter_vmem, convolution_history_vmem, dcp_persistent),
        (dense_semaphores, dp_attended_vmem, dp_compressed_vmem),
        (exp_log_decay_rate_vmem, ffn_norm_vmem, group_input_vmem),
        (group_positions_smem, headroom_pad, kda_state_source, key_value_norms_vmem),
        (latent_norms_vmem, metadata_semaphores, mixture_vmem, mlp_residual_vmem),
        (norm_semaphores, normalized_vmem, output_normalization_vmem),
        (owned_rows_vmem, pool_semaphores, prefix_vmem, projection_expert_pool_vmem),
        (query_norms_vmem, recurrent_state_vmem, router_semaphores, router_vmem),
        (routing_biases_vmem, selected_state_row, stream_semaphores),
        time_step_bias_vmem,
    )


def _stack_body_stage_cm_stage_ef(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (attention_vmem, b4_attention_gather, common, compact_outputs),
        (moe_routed_reduce_scatter, normalized_vmem, pair_split_gather),
        (select_state_row, slot_reads, slots_ref, state_row_ref),
    ) = stage_ctx
    gather_wire = None
    if b4_attention_gather:
        *common, gather_wire = common
    pair_semaphores = None
    if pair_split_gather:
        *common, pair_send, pair_receive = common
        pair_semaphores = (pair_send.at[0], pair_receive.at[0])
    routed_consumed = None
    if moe_routed_reduce_scatter:
        *common, routed_consumed = common
    communication_workspace = AttentionCommunicationWorkspace(
        *common,
        attention_gather_vmem=gather_wire,
        routed_consumed=routed_consumed,
        pair_semaphores=pair_semaphores,
    )
    if compact_outputs:
        attention_vmem = normalized_vmem
    selected_state_row = state_row_ref[0] if select_state_row else None

    def kda_state_source(ref: TensorLike, state_index: TensorLike) -> TensorLike:
        """Compute and stage Pallas kda state source."""
        if select_state_row:
            row = slots_ref[selected_state_row] if slot_reads else selected_state_row
            return ref.at[pl.ds(row, 1), state_index]
        return ref.at[:, state_index]

    return (
        (attention_vmem, communication_workspace, kda_state_source),
        selected_state_row,
    )


def _stack_body_stage_cm(stage_ctx: TensorLike, stage_args: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for body."""
    (
        (alias_expert_scales, attention_dp, attention_dp_ownership),
        (attention_token_scatter, aux_capture, aux_hidden_layers),
        (b4_attention_gather, batch_size, block_rows_held, blocks_block_major),
        (blocks_owned, cache_handoff_head_count, compact_outputs),
        (dcp_owner_table, dense_names, dense_slabs, dp_held_rows, eps),
        (expert_pool_bytes, first_layer, group_rows, group_rows_per_pair),
        (hidden_size, include_layer_zero, kda_row_chunk, kda_shard_swap),
        (layers, live_rows, mla_scoped_scratch, moe_routed_reduce_scatter),
        (moe_scoped_scratch, moe_token_scatter_output),
        (overlap_kda_state_reads, overlap_kda_state_writes, owned_rows),
        (owner_generic, paged_pages, pair_split_gather, persist_kda_state),
        (pool_layout, pool_managed, pool_map, pool_weights),
        (residual_reduction_order, select_state_row),
        (sequence_parallel_pre_attention, sequence_rows, shard_routing),
        (slot_reads, sp8_residual, sp_owners, states, stream_config),
        (stream_kda_state, tokens_per_pair),
    ) = stage_ctx
    position_ref, state_row_ref, refs = stage_args
    copy = _NONE_STAGE_VAL
    (
        (active_rows, aux_output_ref, blocks_output_ref, convolution_output_ref),
        (convolution_ref, dense_refs, dp_attended_vmem, dp_compressed_vmem),
        (embedding_ref, final_normalization_ref, group_input_vmem),
        (group_positions_smem, key_output_ref, layer_weights, output_ref),
        (output_residual_ref, pool_ref, prefix_ref, recurrent_output_ref),
        (recurrent_ref, refs, slots_ref, state_count, state_start, table_ref),
        value_output_ref,
    ) = _stack_body_stage_cm_stage_ew(
        ((batch_size, dense_names, live_rows, pool_map, refs, slot_reads), states)
    )
    (
        (attention_norm_vmem, attention_residual_vmem, attention_vmem),
        (aux_semaphore, auxiliary_semaphores, blocks_vmem, communication_workspace),
        (convolution_filter_vmem, convolution_history_vmem, dcp_persistent),
        (dense_semaphores, dp_attended_vmem, dp_compressed_vmem),
        (exp_log_decay_rate_vmem, ffn_norm_vmem, group_input_vmem),
        (group_positions_smem, headroom_pad, kda_state_source, key_value_norms_vmem),
        (latent_norms_vmem, metadata_semaphores, mixture_vmem, mlp_residual_vmem),
        (norm_semaphores, normalized_vmem, output_normalization_vmem),
        (owned_rows_vmem, pool_semaphores, prefix_vmem, projection_expert_pool_vmem),
        (query_norms_vmem, recurrent_state_vmem, router_semaphores, router_vmem),
        (routing_biases_vmem, _selected_state_row, stream_semaphores),
        time_step_bias_vmem,
    ) = _stack_body_stage_cm_stage_ex(
        (
            (attention_dp, b4_attention_gather, compact_outputs, dp_attended_vmem),
            (dp_compressed_vmem, group_input_vmem, group_positions_smem),
            (moe_routed_reduce_scatter, pair_split_gather, refs, select_state_row),
            (slot_reads, slots_ref, state_count, state_row_ref, state_start),
        )
    )

    def kda_state_reads(
        ref: TensorLike,
        vmem: TensorLike,
        semaphore: TensorLike,
        state_index: TensorLike,
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for kda state reads.

        The copies of a layer's input KDA states into ``vmem``: one, or
        one per row from its slot (``state_slots``).
        """
        if not slot_reads or select_state_row:
            return (
                tpu.make_async_copy(
                    kda_state_source(ref, state_index),
                    kda_state_destination(vmem),
                    semaphore,
                ),
            )
        return tuple(
            tpu.make_async_copy(
                ref.at[pl.ds(slots_ref[b], 1), state_index],
                vmem.at[pl.ds(b, 1)],
                semaphore,
            )
            for b in range(batch_size)
        )

    (
        (arena_router, attention_views, expert_views, kda_state_destination, pool),
        (pool_view, _pool_views, request_router),
    ) = _stack_body_stage_cm_stage_ep(
        (
            (alias_expert_scales, hidden_size, layer_weights, pool_layout),
            (pool_managed, pool_weights, projection_expert_pool_vmem),
            (router_semaphores, select_state_row),
        )
    )

    def request_attention(index: TensorLike, predicate: TensorLike = None) -> None:
        """Queue layer ``index``'s attention input projections."""
        pool.request(
            "attention",
            pool_layout["attention_bytes"],
            lambda offset, index: prefetch_attention(index, base=offset),
            index=index,
            predicate=predicate,
        )

    (
        (auxiliary_vmem, block_rows, dp_owned_vmem, _for_dp_owned_rows),
        (_initialize_blocks, metadata_vmem, normalize, owner_block_row),
        (owner_indices, owner_prefix_row, owner_start, owner_when, residual),
        (sp8_owned_value, sp8_rows_vmem, store_mixed_rows),
    ) = _stack_body_stage_cm_stage_ev(
        (
            (attention_dp, attention_dp_ownership, batch_size, block_rows_held),
            (blocks_block_major, blocks_owned, blocks_vmem),
            (communication_workspace, convolution_filter_vmem),
            (convolution_history_vmem, dp_held_rows, embedding_ref, eps),
            (exp_log_decay_rate_vmem, group_positions_smem, group_rows),
            (group_rows_per_pair, hidden_size, include_layer_zero),
            (key_value_norms_vmem, latent_norms_vmem, output_normalization_vmem),
            (owned_rows, owner_generic, position_ref, prefix_ref, prefix_vmem),
            (query_norms_vmem, recurrent_state_vmem, residual_reduction_order),
            (routing_biases_vmem, sequence_rows, sp8_residual, sp_owners),
            (time_step_bias_vmem, tokens_per_pair),
        )
    )
    mix_dp_rows, normalize_dp_rows = _stack_body_stage_cm_stage_es(
        (
            (attention_dp, batch_size, block_rows, blocks_vmem),
            (communication_workspace, dp_held_rows, dp_owned_vmem, eps),
            (group_rows_per_pair, hidden_size, kda_shard_swap, normalize),
            (normalized_vmem, owned_rows_vmem, prefix_vmem),
            (residual_reduction_order, sp8_residual, sp8_rows_vmem),
            store_mixed_rows,
        )
    )

    def swap_kda_projections(stream: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for swap kda projections.

        DP2 B2/B4's shard swap: this rank projects its own group's held
        shard for every row, sends the other group's rows to its
        counterpart there (which projects the other shard) and gets this
        group's rows of that shard back. Returns ``rows(order)``: the
        group's rows of the own shard's input projection (order 0), then,
        after the exchange, of the other shard's (order 1).
        """
        rank = jax.lax.axis_index("tp")
        group = rank // (32 // attention_dp)
        projected = stream.dot(
            normalized_vmem[...], communication_workspace.reduction_phase[0] >= 0
        )
        projected = projected.astype(jnp.bfloat16)
        own = jnp.where(group == 0, projected[:group_rows], projected[group_rows:])
        other = jnp.where(group == 0, projected[group_rows:], projected[:group_rows])
        used = group_rows * projected.shape[1] // 128
        payload = used + used % 2
        local = communication_workspace.shared_local_reduction_vmem
        send = local.at[4].bitcast(jnp.bfloat16).at[pl.ds(0, payload), :]
        receive = local.at[5].bitcast(jnp.bfloat16).at[pl.ds(0, payload), :]
        wire = other.reshape(used, 128)
        if payload != used:
            wire = jnp.concatenate(
                [wire, jnp.zeros((payload - used, 128), jnp.bfloat16)]
            )
        send[...] = wire
        copy = tpu.make_async_remote_copy(
            send,
            receive,
            communication_workspace.send_semaphores.at[14],
            communication_workspace.receive_semaphores.at[14],
            device_id=(rank ^ 32 // attention_dp,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()

        def rows(order: TensorLike) -> TensorLike:
            """Compute and stage Pallas VMEM state for rows."""
            if order == 0:
                return own
            copy.wait()
            return receive[...][:used].reshape(own.shape)

        return rows

    gather_dp_rows, prefetch_norms = _stack_body_stage_cm_stage_em(
        (
            (attention_dp, attention_norm_vmem, attention_residual_vmem),
            (communication_workspace, dp_owned_vmem, ffn_norm_vmem, layer_weights),
            (mlp_residual_vmem, norm_semaphores, owned_rows_vmem, sp8_owned_value),
            sp_owners,
        )
    )
    (
        (kda_streams, project_attention_dp, query_down_weight_stream, _storage),
        weight_stream,
    ) = _stack_body_stage_cm_stage_eh(
        (
            (attention_dp, attention_views, communication_workspace),
            (dp_attended_vmem, dp_held_rows, group_rows, hidden_size),
            (kda_shard_swap, layer_weights, pool_weights),
            (projection_expert_pool_vmem, sp_owners, stream_config),
            stream_semaphores,
        )
    )
    prefetch_attention = _stack_body_stage_cm_stage_eu(
        attention_views,
        kda_streams,
        layer_weights,
        pool_semaphores,
        query_down_weight_stream,
    )
    (_prefetch_auxiliary, _prefetch_metadata, _prefetch_router) = (
        _stack_body_stage_cm_stage_eo(
            (
                (active_rows, alias_expert_scales, arena_router, attention_dp),
                (attention_dp_ownership, attention_norm_vmem, attention_residual_vmem),
                (attention_token_scatter, attention_views, attention_vmem),
                (aux_capture, aux_hidden_layers, aux_output_ref, aux_semaphore),
                (auxiliary_semaphores, auxiliary_vmem, batch_size, block_rows),
                (block_rows_held, blocks_block_major, blocks_vmem),
                (cache_handoff_head_count, communication_workspace),
                (convolution_history_vmem, convolution_output_ref, convolution_ref),
                (dcp_owner_table, dcp_persistent, dense_refs, dense_semaphores),
                (dense_slabs, dp_attended_vmem, dp_compressed_vmem, dp_held_rows),
                (dp_owned_vmem, eps, expert_pool_bytes, expert_views, ffn_norm_vmem),
                (first_layer, gather_dp_rows, group_input_vmem, group_positions_smem),
                (group_rows, group_rows_per_pair, hidden_size, include_layer_zero),
                (kda_row_chunk, kda_shard_swap, kda_state_destination),
                (kda_state_reads, kda_state_source, kda_streams, key_output_ref),
                (key_value_norms_vmem, latent_norms_vmem, layer_weights, layers),
                (metadata_semaphores, metadata_vmem, mix_dp_rows, mixture_vmem),
                (mla_scoped_scratch, mlp_residual_vmem, moe_routed_reduce_scatter),
                (moe_scoped_scratch, moe_token_scatter_output, norm_semaphores),
                (normalize, normalize_dp_rows, normalized_vmem),
                (overlap_kda_state_reads, overlap_kda_state_writes, owned_rows),
                (owned_rows_vmem, owner_block_row, owner_indices, owner_prefix_row),
                (owner_start, owner_when, paged_pages, persist_kda_state, pool),
                (pool_layout, pool_managed, pool_ref, pool_semaphores, pool_view),
                (position_ref, prefetch_attention, prefetch_norms, prefix_vmem),
                (project_attention_dp, projection_expert_pool_vmem),
                (query_down_weight_stream, query_norms_vmem, recurrent_output_ref),
                (recurrent_ref, recurrent_state_vmem, request_attention),
                (request_router, residual, residual_reduction_order),
                (router_semaphores, router_vmem, routing_biases_vmem),
                (sequence_parallel_pre_attention, sequence_rows, shard_routing),
                (slot_reads, slots_ref, sp8_owned_value, sp8_residual, sp_owners),
                (states, store_mixed_rows, stream_kda_state, swap_kda_projections),
                (table_ref, tokens_per_pair, value_output_ref, weight_stream),
            )
        )
    )
    _stack_body_stage_cm_stage_et(headroom_pad, prefix_vmem)
    if sequence_parallel_pre_attention:
        copy, _final_block_count = _stack_body_stage_cm_stage_ec(
            (
                (attention_dp_ownership, attention_norm_vmem, attention_residual_vmem),
                (block_rows, blocks_output_ref, communication_workspace, copy),
                (dp_owned_vmem, eps, final_normalization_ref, gather_dp_rows, layers),
                (norm_semaphores, normalize, normalize_dp_rows, normalized_vmem),
                (output_ref, output_residual_ref, owned_rows, owned_rows_vmem),
                (owner_block_row, owner_indices, owner_prefix_row, owner_start),
                (owner_when, prefix_vmem, sp_owners, store_mixed_rows),
            )
        )
    else:
        _stack_body_stage_cm_stage_ey(
            blocks_output_ref, blocks_vmem, output_ref, prefix_vmem
        )


def _stack_body_stage_cm_stage_ep(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (alias_expert_scales, hidden_size, layer_weights, pool_layout),
        (pool_managed, pool_weights, projection_expert_pool_vmem),
        (router_semaphores, select_state_row),
    ) = stage_ctx

    def kda_state_destination(vmem: TensorLike) -> TensorLike:
        """Compute and stage Pallas kda state destination."""
        return vmem.at[pl.ds(0, 1)] if select_state_row else vmem

    (attention_views, expert_views, pool_view, pool_views) = (
        _stack_body_stage_cm_stage_ed(
            alias_expert_scales, pool_layout, pool_weights, projection_expert_pool_vmem
        )
    )
    pool = (
        VmemPool(
            projection_expert_pool_vmem,
            pool_layout["arena_bytes"],
            report=_config().pool_report,
        )
        if pool_managed
        else None
    )

    def managed_pool() -> VmemPool:
        """Return the arena pool; router arena copies exist only when pool-managed."""
        if pool is None:
            msg = "router arena copies need a pool-managed layout"
            raise RuntimeError(msg)
        return pool

    def arena_router(offset: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for arena router."""
        return managed_pool().view(
            offset, (1, hidden_size, 512), jnp.dtype(jnp.bfloat16)
        )

    def request_router(index: TensorLike, predicate: TensorLike = None) -> None:
        """Queue layer ``index``'s router copy into the arena."""
        managed_pool().request(
            "router",
            pool_alias.storage_size((1, hidden_size, 512), jnp.dtype(jnp.bfloat16)),
            lambda offset, index: _router_copy(
                layer_weights["router"].at[index - 1],
                arena_router(offset),
                router_semaphores,
                wait=False,
                initialize_padding=False,
            ),
            index=index,
            predicate=predicate,
        )

    return (
        (arena_router, attention_views, expert_views, kda_state_destination, pool),
        (pool_view, pool_views, request_router),
    )


def _stack_body_stage_cm_stage_ed(
    alias_expert_scales: TensorLike,
    pool_layout: TensorLike,
    pool_weights: TensorLike,
    projection_expert_pool_vmem: TensorLike,
) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""

    def pool_view(name: TensorLike, base: TensorLike = 0) -> TensorLike:
        """Compute and stage Pallas VMEM state for pool view.

        The pooled view of one layer weight at its _pool_layout offset
        (from ``base``: the MLA output projection is placed relative to
        its layer's attention weights).
        """
        value = pool_weights[name]
        size = math.prod(value.shape) * value.dtype.itemsize
        return pool_alias.view(
            projection_expert_pool_vmem.at[
                pl.ds((base + pool_layout[name]) // 128, size // 128), :
            ],
            value.shape,
            value.dtype,
        )

    def pool_views(
        names: TensorLike, *, experts: TensorLike = False, base: TensorLike = 0
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for pool views."""
        views = []
        offset = base
        for name in names:
            value = pool_weights[name]
            logical_shape = value.shape[1:] if experts else value.shape
            view_shape = (4, *logical_shape) if experts else logical_shape
            size = math.prod(view_shape) * value.dtype.itemsize
            storage_size = pool_alias.storage_size(view_shape, value.dtype)
            slice_ref = projection_expert_pool_vmem.at[
                pl.ds(offset // 128, size // 128), :
            ]
            if storage_size == size:
                views.append(pool_alias.view(slice_ref, view_shape, value.dtype))
            else:
                # offset += storage_size reserves the padded tiled footprint,
                # while ReshapeTransform._validate_shape requires slice_ref
                # to match math.prod(view_shape) logical bytes.
                bitcast_ref = slice_ref.bitcast(value.dtype)
                reinterpret = cast(
                    "Callable[[tuple[int, ...]], state_types.Transform]",
                    pool_alias.Reinterpret,
                )
                views.append(
                    pool_alias.types.TransformedRef(
                        bitcast_ref.ref,
                        (
                            *bitcast_ref.transforms,
                            reinterpret(tuple(view_shape)),
                        ),
                    )
                )
            offset += storage_size
        return views

    def attention_views(base: TensorLike = 0) -> TensorLike:
        """The KDA and MLA input projections of one layer from ``base``."""
        return (pool_views(KDA[:4], base=base), pool_views(MLA[:2], base=base))

    def expert_views(base: TensorLike = 0) -> TensorLike:
        """Compute and stage Pallas VMEM state for expert views."""
        if alias_expert_scales:
            views = pool_views(
                ("expert_gu", "expert_down", "expert_gus", "expert_ds"),
                experts=True,
                base=base,
            )
            return (views[0], views[2], views[1], views[3])
        return pool_views(("expert_gu", "expert_down"), experts=True, base=base)

    return (attention_views, expert_views, pool_view, pool_views)


def _stack_body_stage_cm_stage_ev(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (attention_dp, attention_dp_ownership, batch_size, block_rows_held),
        (blocks_block_major, blocks_owned, blocks_vmem),
        (communication_workspace, convolution_filter_vmem),
        (convolution_history_vmem, dp_held_rows, embedding_ref, eps),
        (exp_log_decay_rate_vmem, group_positions_smem, group_rows),
        (group_rows_per_pair, hidden_size, include_layer_zero),
        (key_value_norms_vmem, latent_norms_vmem, output_normalization_vmem),
        (owned_rows, owner_generic, position_ref, prefix_ref, prefix_vmem),
        (query_norms_vmem, recurrent_state_vmem, residual_reduction_order),
        (routing_biases_vmem, sequence_rows, sp8_residual, sp_owners),
        (time_step_bias_vmem, tokens_per_pair),
    ) = stage_ctx
    auxiliary_vmem = (
        convolution_filter_vmem,
        exp_log_decay_rate_vmem,
        time_step_bias_vmem,
        output_normalization_vmem,
        convolution_history_vmem,
        recurrent_state_vmem,
    )
    metadata_vmem = (
        query_norms_vmem,
        key_value_norms_vmem,
        routing_biases_vmem,
        latent_norms_vmem,
    )
    collectives32.barrier()
    communication_workspace.reduction_phase[0] = 0
    if attention_dp > 1:
        data_group = jax.lax.axis_index("tp") // (32 // attention_dp)
        for row in range(group_rows):
            pair_index, within = divmod(row, group_rows_per_pair)
            group_positions_smem[row] = position_ref[
                tokens_per_pair * pair_index + data_group * group_rows_per_pair + within
            ]
    for_dp_owned_rows, load_prefix = _stack_body_stage_cm_stage_ee(
        (
            (attention_dp, attention_dp_ownership, dp_held_rows),
            (group_rows_per_pair, prefix_ref, prefix_vmem, sp_owners),
            tokens_per_pair,
        )
    )
    if attention_dp_ownership:
        pl.run_scoped(
            load_prefix,
            tpu.VMEM(prefix_ref.shape, prefix_ref.dtype),
            tpu.SemaphoreType.DMA,
        )
    else:
        pl.run_scoped(
            functools.partial(load_prefix, prefix_vmem), tpu.SemaphoreType.DMA
        )
    (initialize_blocks, owner_indices) = _stack_body_stage_cm_stage_ek(
        (
            (attention_dp, attention_dp_ownership, batch_size, blocks_block_major),
            (blocks_owned, blocks_vmem, dp_held_rows, embedding_ref),
            (for_dp_owned_rows, include_layer_zero, owned_rows, owner_generic),
            (sequence_rows, sp_owners),
        )
    )
    (owner_block_row, owner_prefix_row, owner_start, owner_when) = (
        _stack_body_stage_cm_stage_ei(owner_generic, prefix_vmem)
    )

    def block_rows(row: TensorLike, count: TensorLike = 8) -> TensorLike:
        """The residual snapshots of one batch row, ``[count, hidden]``."""
        if blocks_block_major:
            row = row % block_rows_held
            return blocks_vmem[row // 2, pl.ds(row % 2 * 8, count)]
        return (
            blocks_vmem[row]
            if count == _ROUTED_EXPERTS_PER_TOKEN
            else blocks_vmem[row, :count]
        )

    (dp_owned_vmem, normalize, residual, store_mixed_rows) = (
        _stack_body_stage_cm_stage_ej(
            (
                (attention_dp_ownership, batch_size, blocks_vmem),
                (communication_workspace, dp_held_rows, eps, hidden_size, prefix_vmem),
                (residual_reduction_order, sp_owners),
            )
        )
    )
    sp8_rows_vmem = dp_owned_vmem if sp8_residual else None

    def sp8_owned_value() -> TensorLike:
        """Compute and stage Pallas sp8 owned value."""
        if sp8_rows_vmem is None:
            msg = "SP8-owned rows exist only with the SP8 residual layout"
            raise RuntimeError(msg)
        value = sp8_rows_vmem[...]
        return jnp.where(jax.lax.axis_index("tp") % 2 == 0, value[:8], value[8:16])

    return (
        (auxiliary_vmem, block_rows, dp_owned_vmem, for_dp_owned_rows),
        (initialize_blocks, metadata_vmem, normalize, owner_block_row),
        (owner_indices, owner_prefix_row, owner_start, owner_when, residual),
        (sp8_owned_value, sp8_rows_vmem, store_mixed_rows),
    )


def _stack_body_stage_cm_stage_ee(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (attention_dp, attention_dp_ownership, dp_held_rows),
        (group_rows_per_pair, prefix_ref, prefix_vmem, sp_owners),
        tokens_per_pair,
    ) = stage_ctx

    def for_dp_owned_rows(run: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for for dp owned rows.

        ``run(first)`` with this rank's first batch row (static per row
        owner and DP group).
        """
        if sp_owners == 1:
            run(0)
            return
        rank = jax.lax.axis_index("tp")
        pair, group = (rank % 8 // (8 // sp_owners), rank // (32 // attention_dp))
        for owner in range(sp_owners):
            for data_group in range(attention_dp):

                @pl.when((pair == owner) & (group == data_group))
                def owned_region(
                    owner: TensorLike = owner, data_group: TensorLike = data_group
                ) -> None:
                    """Compute and stage Pallas VMEM state for owned region."""
                    run(owner * tokens_per_pair + data_group * group_rows_per_pair)

    def load_prefix(staged: TensorLike, semaphore: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for load prefix."""
        copy = tpu.make_async_copy(prefix_ref, staged, semaphore)
        copy.start()
        copy.wait()
        if attention_dp_ownership:

            def take_owned_rows(first: TensorLike) -> None:
                """Compute and stage Pallas take owned rows."""
                prefix_vmem[:dp_held_rows] = staged[first : first + dp_held_rows]

            for_dp_owned_rows(take_owned_rows)

    return (for_dp_owned_rows, load_prefix)


def _stack_body_stage_cm_stage_ek(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (attention_dp, attention_dp_ownership, batch_size, blocks_block_major),
        (blocks_owned, blocks_vmem, dp_held_rows, embedding_ref),
        (for_dp_owned_rows, include_layer_zero, owned_rows, owner_generic),
        (sequence_rows, sp_owners),
    ) = stage_ctx
    blocks_vmem[...] = jnp.zeros(blocks_vmem.shape, blocks_vmem.dtype)

    def initialize_blocks(embedding_ref: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for initialize blocks."""
        return _stack_body_initialize_blocks_stage_ck(
            (
                (attention_dp_ownership, batch_size, blocks_block_major, blocks_owned),
                (blocks_vmem, dp_held_rows, for_dp_owned_rows, owned_rows, sp_owners),
            ),
            embedding_ref,
        )

    def staged_blocks(embedding_vmem: TensorLike, semaphore: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for staged blocks."""
        copy = tpu.make_async_copy(embedding_ref, embedding_vmem, semaphore)
        copy.start()
        copy.wait()
        initialize_blocks(embedding_vmem)

    if not (
        include_layer_zero
        and attention_dp == _PAIR_TUPLE_LEN
        and (batch_size in (64, 128))
        and (sequence_rows == 1)
    ):
        pl.run_scoped(
            staged_blocks,
            tpu.VMEM(embedding_ref.shape, embedding_ref.dtype),
            tpu.SemaphoreType.DMA,
        )

    def owner_indices(pair: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for owner indices."""
        return (pair,) if owner_generic else range(sp_owners)

    return (initialize_blocks, owner_indices)


def _stack_body_initialize_blocks_stage_ck(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for initialize blocks."""
    (
        (attention_dp_ownership, batch_size, blocks_block_major, blocks_owned),
        (blocks_vmem, dp_held_rows, for_dp_owned_rows, owned_rows, sp_owners),
    ) = stage_ctx
    embedding_ref = stage_args
    if attention_dp_ownership:

        def initialize_owned_blocks(first: TensorLike) -> None:
            """Compute and stage Pallas initialize owned blocks."""
            for local in range(dp_held_rows):
                blocks_vmem[local // 2, pl.ds(local % 2 * 8, 1)] = embedding_ref[
                    pl.ds(first + local, 1)
                ].astype(jnp.bfloat16)

        for_dp_owned_rows(initialize_owned_blocks)
    elif blocks_owned:
        owner_of_rank = jax.lax.axis_index("tp") % 8 // (8 // sp_owners)
        for owner_index in range(sp_owners):

            @pl.when(owner_of_rank == owner_index)
            def initialize_owned_blocks(owner_index: TensorLike = owner_index) -> None:
                """Compute and stage Pallas initialize owned blocks."""
                for local in range(owned_rows):
                    blocks_vmem[local // 2, pl.ds(local % 2 * 8, 1)] = embedding_ref[
                        pl.ds(owner_index * owned_rows + local, 1)
                    ].astype(jnp.bfloat16)
    elif blocks_block_major:
        for row in range(batch_size):
            blocks_vmem[row // 2, pl.ds(row % 2 * 8, 1)] = embedding_ref[
                pl.ds(row, 1)
            ].astype(jnp.bfloat16)
    else:
        blocks_vmem[:, 0] = embedding_ref[...].astype(jnp.float32)


def _stack_body_stage_cm_stage_ei(
    owner_generic: TensorLike, prefix_vmem: TensorLike
) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""

    def owner_when(pair: TensorLike, owner: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for owner when."""
        if owner_generic:
            return lambda function: function()
        return pl.when(pair == owner)

    def owner_start(start: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for owner start."""
        return pl.multiple_of(start, 16) if owner_generic else start

    def owner_block_row(global_row: TensorLike, local_row: TensorLike) -> TensorLike:
        """Compute and stage Pallas owner block row."""
        return local_row if owner_generic else global_row

    def owner_prefix_row(
        global_row: TensorLike, row_slice: TensorLike, local_row: TensorLike
    ) -> TensorLike:
        """Compute and stage Pallas owner prefix row."""
        if owner_generic:
            return prefix_vmem[row_slice][local_row : local_row + 1]
        return prefix_vmem[pl.ds(global_row, 1)]

    return (owner_block_row, owner_prefix_row, owner_start, owner_when)


def _stack_body_stage_cm_stage_ej(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (attention_dp_ownership, batch_size, blocks_vmem),
        (communication_workspace, dp_held_rows, eps, hidden_size, prefix_vmem),
        (residual_reduction_order, sp_owners),
    ) = stage_ctx

    def residual(
        destination: TensorLike, folded: TensorLike, active_block_count: TensorLike
    ) -> None:
        """Compute and stage Pallas VMEM state for residual."""
        _residual_mixture_by_row(
            blocks_vmem[...],
            prefix_vmem[...],
            destination,
            folded,
            active_block_count,
            batch_size=batch_size,
            epsilon=eps,
            relaxed_reduction_order=residual_reduction_order == "relaxed",
        )

    def store_mixed_rows(compute_row: TensorLike, destination: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for store mixed rows.

        Write the destination's owned rows. Above two rows per pair (B16)
        each row runs in its own region: unrolled rows would keep every
        row's 8 x hidden FP32 blocks live and spill. DP2 B4's four rows
        (one owner) stay unrolled.
        """
        rows = destination.shape[0]
        if rows <= _PAIR_TUPLE_LEN or sp_owners == 1:
            destination[...] = jnp.concatenate(
                [compute_row(local_row) for local_row in range(rows)], axis=0
            )
            return
        for local_row in range(rows):

            @pl.when(communication_workspace.reduction_phase[0] >= 0)
            def store_row(local_row: TensorLike = local_row) -> None:
                """Compute and stage Pallas VMEM state for store row."""
                destination[pl.ds(local_row, 1)] = compute_row(local_row)

    def normalize(destination: TensorLike, scale: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for normalize."""
        value = destination[...].astype(jnp.float32)
        destination[...] = (
            value
            * jax.lax.rsqrt(jnp.mean(value * value, axis=1, keepdims=True) + eps)
            * scale[...]
        ).astype(jnp.bfloat16)

    dp_owned_vmem = (
        _bf16_rows_view(
            communication_workspace.attention_output_vmem, dp_held_rows, hidden_size
        )
        if attention_dp_ownership
        else None
    )
    return (dp_owned_vmem, normalize, residual, store_mixed_rows)


def _stack_body_stage_cm_stage_es(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (attention_dp, batch_size, block_rows, blocks_vmem),
        (communication_workspace, dp_held_rows, dp_owned_vmem, eps),
        (group_rows_per_pair, hidden_size, kda_shard_swap, normalize),
        (normalized_vmem, owned_rows_vmem, prefix_vmem),
        (residual_reduction_order, sp8_residual, sp8_rows_vmem),
        store_mixed_rows,
    ) = stage_ctx

    def mix_dp_rows(
        folded: TensorLike,
        active_blocks: TensorLike,
        *,
        layer_index: TensorLike,
        attention_output: TensorLike = False,
        start_block: TensorLike = False,
    ) -> None:
        """Compute and stage Pallas mix dp rows."""
        return _stack_body_mix_dp_rows_stage_ch(
            (
                (block_rows, blocks_vmem, communication_workspace, dp_held_rows),
                (dp_owned_vmem, eps, hidden_size, kda_shard_swap, normalized_vmem),
                (prefix_vmem, residual_reduction_order, sp8_residual, sp8_rows_vmem),
                store_mixed_rows,
            ),
            ((folded, active_blocks, layer_index, attention_output, start_block),),
        )

    def normalize_dp_rows(scale: TensorLike, *, split: TensorLike = False) -> None:
        """Compute and stage Pallas normalize dp rows."""
        return _stack_body_normalize_dp_rows_stage_ci(
            (
                (attention_dp, batch_size, dp_owned_vmem, group_rows_per_pair),
                (hidden_size, normalize, owned_rows_vmem, sp8_rows_vmem),
            ),
            (scale, split),
        )

    return (mix_dp_rows, normalize_dp_rows)


def _stack_body_mix_dp_rows_stage_ch(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas mix dp rows."""
    (
        (block_rows, blocks_vmem, communication_workspace, dp_held_rows),
        (dp_owned_vmem, eps, hidden_size, kda_shard_swap, normalized_vmem),
        (prefix_vmem, residual_reduction_order, sp8_residual, sp8_rows_vmem),
        store_mixed_rows,
    ) = stage_ctx
    ((folded, active_blocks, layer_index, attention_output, start_block),) = stage_args
    rows = slice(0, dp_held_rows)
    if attention_output:
        prefix_vmem[rows] = jnp.where(
            layer_index % 12 == 0,
            dp_owned_vmem[...],
            prefix_vmem[rows] + dp_owned_vmem[...],
        ).astype(jnp.bfloat16)

    def mix_row(row: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for mix row."""
        return _residual_mixture_row(
            block_rows(row),
            prefix_vmem[row : row + 1],
            folded,
            active_blocks,
            epsilon=eps,
            relaxed_reduction_order=residual_reduction_order == "relaxed",
        )

    if sp8_residual:
        sp8_rows_vmem[...] = jnp.zeros((16, hidden_size), jnp.bfloat16)
        for lane in range(2):

            @pl.when(jax.lax.axis_index("tp") % 2 == lane)
            def mix_half(lane: TensorLike = lane) -> None:
                """Compute and stage Pallas VMEM state for mix half."""
                for local_row in range(8):

                    @pl.when(communication_workspace.reduction_phase[0] >= 0)
                    def store_row(local_row: TensorLike = local_row) -> None:
                        """Compute and stage Pallas VMEM state for store row."""
                        row = lane * 8 + local_row
                        sp8_rows_vmem[pl.ds(row, 1)] = mix_row(row)
    else:
        store_mixed_rows(
            mix_row,
            normalized_vmem if start_block and kda_shard_swap else dp_owned_vmem,
        )
    _stack_body_mix_dp_rows_stage_ch_stage_eb(
        blocks_vmem, dp_held_rows, layer_index, prefix_vmem, start_block
    )


def _stack_body_mix_dp_rows_stage_ch_stage_eb(
    blocks_vmem: TensorLike,
    dp_held_rows: TensorLike,
    layer_index: TensorLike,
    prefix_vmem: TensorLike,
    start_block: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for stack body mix dp."""
    if start_block:

        @pl.when(layer_index % 12 == 0)
        def snapshot() -> None:
            """Compute and stage Pallas VMEM state for snapshot."""
            for row in range(dp_held_rows):
                half = blocks_vmem.at[row // 2, pl.ds(row % 2 * 8, 8)]
                mask = jax.lax.broadcasted_iota(jnp.int32, half.shape, 0)
                half[...] = jnp.where(
                    mask == layer_index // 12, prefix_vmem[row : row + 1], half[...]
                )


def _stack_body_normalize_dp_rows_stage_ci(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas normalize dp rows."""
    (
        (attention_dp, batch_size, dp_owned_vmem, group_rows_per_pair),
        (hidden_size, normalize, owned_rows_vmem, sp8_rows_vmem),
    ) = stage_ctx
    scale, split = stage_args
    if split:
        normalize(sp8_rows_vmem, scale)
        return
    if batch_size != _GATED_GROUP_WIDTH:
        normalize(dp_owned_vmem, scale)
        return
    group = jax.lax.axis_index("tp") // (32 // attention_dp)
    for data_group in range(attention_dp):

        @pl.when(group == data_group)
        def stage_native_rows(data_group: TensorLike = data_group) -> None:
            """Compute and stage Pallas stage native rows."""
            first = data_group * group_rows_per_pair
            parts = []
            if first:
                parts.append(jnp.zeros((first, hidden_size), jnp.bfloat16))
            parts.append(dp_owned_vmem[...])
            tail = 16 - first - group_rows_per_pair
            if tail:
                parts.append(jnp.zeros((tail, hidden_size), jnp.bfloat16))
            owned_rows_vmem[...] = jnp.concatenate(tuple(parts), axis=0)

    normalize(owned_rows_vmem, scale)
    native_rows = owned_rows_vmem[...]
    for data_group in range(attention_dp):

        @pl.when(group == data_group)
        def select_normalized_rows(data_group: TensorLike = data_group) -> None:
            """Compute and stage Pallas select normalized rows."""
            first = data_group * group_rows_per_pair
            dp_owned_vmem[...] = native_rows[first : first + group_rows_per_pair]


def _stack_body_stage_cm_stage_em(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (attention_dp, attention_norm_vmem, attention_residual_vmem),
        (communication_workspace, dp_owned_vmem, ffn_norm_vmem, layer_weights),
        (mlp_residual_vmem, norm_semaphores, owned_rows_vmem, sp8_owned_value),
        sp_owners,
    ) = stage_ctx

    def gather_dp_rows(destination: TensorLike, *, split: TensorLike = False) -> None:
        """Compute and stage Pallas gather dp rows."""
        if split:
            _gather_sp8_dp_moe_rows(
                sp8_owned_value(),
                owned_rows_vmem,
                destination,
                communication_workspace.shared_local_reduction_vmem,
                communication_workspace.send_semaphores,
                communication_workspace.receive_semaphores,
            )
            return
        if sp_owners == 1:
            destination[...] = dp_owned_vmem[...]
            return
        _gather_dp_moe_rows(
            dp_owned_vmem[...],
            owned_rows_vmem,
            destination,
            communication_workspace.shared_local_reduction_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            attention_dp,
            pair_semaphores=communication_workspace.pair_semaphores,
        )

    def prefetch_norms(index: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for prefetch norms."""
        destinations = (
            attention_residual_vmem,
            mlp_residual_vmem,
            attention_norm_vmem,
            ffn_norm_vmem,
        )
        for slot, (name, destination) in enumerate(
            zip(NORMS, destinations, strict=True)
        ):
            tpu.make_async_copy(
                layer_weights[name].at[index], destination, norm_semaphores.at[slot]
            ).start()

    return (gather_dp_rows, prefetch_norms)


def _stack_body_stage_cm_stage_eh(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (attention_dp, attention_views, communication_workspace),
        (dp_attended_vmem, dp_held_rows, group_rows, hidden_size),
        (kda_shard_swap, layer_weights, pool_weights),
        (projection_expert_pool_vmem, sp_owners, stream_config),
        stream_semaphores,
    ) = stage_ctx

    def storage(name: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for storage."""
        value = pool_weights[name]
        return pool_alias.storage_size(value.shape, value.dtype)

    def weight_stream(
        name: TensorLike,
        source: TensorLike,
        full: TensorLike,
        *,
        offset: TensorLike = None,
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for weight stream."""
        return _stack_body_weight_stream_stage_cj(
            (
                (attention_dp, projection_expert_pool_vmem, stream_config),
                stream_semaphores,
            ),
            (name, source, full, offset),
        )

    @_scoped("t9_dp_project")
    def project_attention_dp(weight_view: TensorLike, stream: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for project attention dp.

        Attention DP: project every shard's gated features into this
        rank's owned rows (the token-scatter attention output).
        """
        owned = _bf16_rows_view(
            communication_workspace.attention_output_vmem, dp_held_rows, hidden_size
        )
        pl.run_scoped(
            lambda comm: _project_attention_dp(
                dp_attended_vmem,
                weight_view,
                owned,
                comm,
                communication_workspace.shared_local_reduction_vmem,
                communication_workspace.shared_host_reduction_vmem,
                communication_workspace.attention_features_vmem,
                communication_workspace.send_semaphores,
                communication_workspace.receive_semaphores,
                communication_workspace.reduction_phase,
                _dot,
                weight_stream=stream,
                staging=communication_workspace.projection_staging_vmem,
                owners=sp_owners,
            ),
            tpu.VMEM((group_rows, 8192), jnp.float32),
        )

    def query_down_weight_stream(
        state_index: TensorLike, base: TensorLike = 0
    ) -> TensorLike:
        """Compute Pallas VMEM tensors for query stage.

        MLA's q-down projection (at the attention weights' ``base``) as a
        column stream, if configured.
        """
        return weight_stream(
            "m_qa",
            layer_weights["m_qa"].at[state_index],
            attention_views(base)[1][0],
            offset=base if POOLED_WEIGHTS else None,
        )

    def kda_streams(state_index: TensorLike, base: TensorLike = 0) -> TensorLike:
        """Compute and stage Pallas VMEM state for kda streams."""
        kda_views = attention_views(base)[0]
        source = layer_weights["k_projection"].at[state_index]
        if kda_shard_swap:
            width = source.shape[1] // attention_dp
            group = jax.lax.axis_index("tp") // (32 // attention_dp)
            input_stream = ChunkedWeightStream(
                source.at[:, pl.ds(pl.multiple_of(group * width, 128), width)],
                kda_views[1],
                stream_semaphores,
                axis="columns",
                sizes=(width,),
                depth=1,
            )
        else:
            input_stream = weight_stream(
                "k_projection", source, kda_views[1], offset=base + storage("k_gate")
            )
        streams = {
            "input": input_stream,
            "output": weight_stream(
                "k_o",
                layer_weights["k_o"].at[state_index],
                kda_views[3],
                offset=base
                + storage("k_gate")
                + storage("k_projection")
                + storage("k_fb"),
            ),
        }
        return {key: stream for key, stream in streams.items() if stream is not None}

    return (
        (kda_streams, project_attention_dp, query_down_weight_stream, storage),
        weight_stream,
    )


def _stack_body_weight_stream_stage_cj(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> TensorLike:
    """Compute and stage Pallas VMEM state for weight stream."""
    (attention_dp, projection_expert_pool_vmem, stream_config), stream_semaphores = (
        stage_ctx
    )
    name, source, full, offset = stage_args
    axis = STREAMED_WEIGHTS[name][0]
    chunks, depth = stream_config.get(name, (1, 1))
    if chunks == 1:
        return None
    base = _stream_base(name, depth)
    order = None
    if axis == "rows":
        np.testing.assert_(
            chunks == 2 * attention_dp,
            f"{name} streams as its {2 * attention_dp} pair halves",
        )
        lane = jax.lax.axis_index("tp") % 2

        def _pair_half_order(
            position: TensorLike, *, _lane: TensorLike = lane
        ) -> TensorLike:
            """Order rank-pair feature-row halves with own rows first."""
            return position % attention_dp * 2 + (_lane ^ position // attention_dp)

        order = _pair_half_order
    extent = source.shape[1 if axis == "columns" else 0]
    unit = 128 if axis == "columns" else 16
    if axis == "columns" and attention_dp > 1:
        np.testing.assert_(
            not chunks % attention_dp,
            f"{name}: chunks must be a multiple of the attention DP degree",
        )
        sizes = (
            _chunk_sizes(extent // attention_dp, chunks // attention_dp, unit)
            * attention_dp
        )
    else:
        sizes = _chunk_sizes(extent, chunks, unit)
    stream = ChunkedWeightStream(
        source,
        full,
        stream_semaphores,
        axis=axis,
        sizes=sizes,
        depth=depth,
        base=base,
        order=order,
        pool=None if offset is None else projection_expert_pool_vmem,
        offset=offset,
    )
    stream.shards = attention_dp
    return stream


def _stack_body_stage_cm_stage_eu(
    attention_views: TensorLike,
    kda_streams: TensorLike,
    layer_weights: TensorLike,
    pool_semaphores: TensorLike,
    query_down_weight_stream: TensorLike,
) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""

    def prefetch_attention(index: TensorLike, base: TensorLike = 0) -> None:
        """Compute and stage Pallas VMEM state for prefetch attention."""
        kda_projection_vmem, mla_projection_vmem = attention_views(base)

        def prefetch_kda() -> None:
            """Compute and stage Pallas VMEM state for prefetch kda."""
            state_index = index - index // 4
            projection_indices = (0, 2, 1, 3)
            streams = kda_streams(state_index, base)
            streamed = {
                name: streams[key]
                for key, name in (("input", "k_projection"), ("output", "k_o"))
                if key in streams
            }
            for slot, projection_index in enumerate(projection_indices):
                if KDA[projection_index] in streamed:
                    streamed[KDA[projection_index]].start()
                    continue
                tpu.make_async_copy(
                    layer_weights[KDA[projection_index]].at[state_index],
                    kda_projection_vmem[projection_index],
                    pool_semaphores.at[slot],
                ).start()

        def prefetch_mla() -> None:
            """Compute and stage Pallas VMEM state for prefetch mla."""
            state_index = index // 4
            query_down_stream = query_down_weight_stream(state_index, base)
            for slot, (name, destination) in enumerate(
                zip(MLA[:2], mla_projection_vmem, strict=True)
            ):
                if name == "m_qa" and query_down_stream is not None:
                    query_down_stream.start()
                    continue
                tpu.make_async_copy(
                    layer_weights[name].at[state_index],
                    destination,
                    pool_semaphores.at[slot],
                ).start()

        jax.lax.cond(
            (index % 4 == _DCP_HEAD_COUNT) | (index == _MAX_STAGE_INDEX),
            prefetch_mla,
            prefetch_kda,
        )

    return prefetch_attention


def _stack_body_stage_cm_stage_eo(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (active_rows, alias_expert_scales, arena_router, attention_dp),
        (attention_dp_ownership, attention_norm_vmem, attention_residual_vmem),
        (attention_token_scatter, attention_views, attention_vmem),
        (aux_capture, aux_hidden_layers, aux_output_ref, aux_semaphore),
        (auxiliary_semaphores, auxiliary_vmem, batch_size, block_rows),
        (block_rows_held, blocks_block_major, blocks_vmem),
        (cache_handoff_head_count, communication_workspace),
        (convolution_history_vmem, convolution_output_ref, convolution_ref),
        (dcp_owner_table, dcp_persistent, dense_refs, dense_semaphores),
        (dense_slabs, dp_attended_vmem, dp_compressed_vmem, dp_held_rows),
        (dp_owned_vmem, eps, expert_pool_bytes, expert_views, ffn_norm_vmem),
        (first_layer, gather_dp_rows, group_input_vmem, group_positions_smem),
        (group_rows, group_rows_per_pair, hidden_size, include_layer_zero),
        (kda_row_chunk, kda_shard_swap, kda_state_destination),
        (kda_state_reads, kda_state_source, kda_streams, key_output_ref),
        (key_value_norms_vmem, latent_norms_vmem, layer_weights, layers),
        (metadata_semaphores, metadata_vmem, mix_dp_rows, mixture_vmem),
        (mla_scoped_scratch, mlp_residual_vmem, moe_routed_reduce_scatter),
        (moe_scoped_scratch, moe_token_scatter_output, norm_semaphores),
        (normalize, normalize_dp_rows, normalized_vmem),
        (overlap_kda_state_reads, overlap_kda_state_writes, owned_rows),
        (owned_rows_vmem, owner_block_row, owner_indices, owner_prefix_row),
        (owner_start, owner_when, paged_pages, persist_kda_state, pool),
        (pool_layout, pool_managed, pool_ref, pool_semaphores, pool_view),
        (position_ref, prefetch_attention, prefetch_norms, prefix_vmem),
        (project_attention_dp, projection_expert_pool_vmem),
        (query_down_weight_stream, query_norms_vmem, recurrent_output_ref),
        (recurrent_ref, recurrent_state_vmem, request_attention),
        (request_router, residual, residual_reduction_order),
        (router_semaphores, router_vmem, routing_biases_vmem),
        (sequence_parallel_pre_attention, sequence_rows, shard_routing),
        (slot_reads, slots_ref, sp8_owned_value, sp8_residual, sp_owners),
        (states, store_mixed_rows, stream_kda_state, swap_kda_projections),
        (table_ref, tokens_per_pair, value_output_ref, weight_stream),
    ) = stage_ctx

    def prefetch_auxiliary(index: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for prefetch auxiliary."""

        @pl.when((index % 4 != _DCP_HEAD_COUNT) & (index != _MAX_STAGE_INDEX))
        def prefetch_kda_auxiliary() -> None:
            """Compute and stage Pallas prefetch kda auxiliary."""
            state_index = index - index // 4
            sources = tuple(layer_weights[name].at[state_index] for name in KDA[4:])
            destinations = list(auxiliary_vmem[: len(sources)])
            if persist_kda_state:
                sources += (
                    kda_state_source(convolution_ref, state_index),
                    kda_state_source(recurrent_ref, state_index),
                )
                destinations += [
                    kda_state_destination(auxiliary_vmem[4]),
                    kda_state_destination(auxiliary_vmem[5]),
                ]
            for slot, (source, destination) in enumerate(
                zip(sources, destinations, strict=True)
            ):
                tpu.make_async_copy(
                    source, destination, auxiliary_semaphores.at[slot]
                ).start()

    prefetch_metadata = _stack_body_stage_cm_stage_en(
        layer_weights, metadata_semaphores, metadata_vmem
    )
    prefetch_router = _stack_body_stage_cm_stage_eg(
        (
            (first_layer, include_layer_zero, layer_weights, pool, pool_managed),
            (prefetch_attention, prefetch_auxiliary, prefetch_metadata),
            (prefetch_norms, request_attention, request_router, router_semaphores),
            router_vmem,
        )
    )

    def layer(layer_index: TensorLike, *, dense_layer: TensorLike = False) -> None:
        """Compute and stage Pallas VMEM state for layer."""
        return _stack_body_layer_stage_cg(
            (
                (active_rows, alias_expert_scales, arena_router, attention_dp),
                (attention_dp_ownership, attention_norm_vmem, attention_residual_vmem),
                (attention_token_scatter, attention_views, attention_vmem),
                (aux_capture, aux_hidden_layers, aux_output_ref, aux_semaphore),
                (auxiliary_semaphores, auxiliary_vmem, batch_size, block_rows),
                (block_rows_held, blocks_block_major, blocks_vmem),
                (cache_handoff_head_count, communication_workspace),
                (convolution_history_vmem, convolution_output_ref, convolution_ref),
                (dcp_owner_table, dcp_persistent, dense_refs, dense_semaphores),
                (dense_slabs, dp_attended_vmem, dp_compressed_vmem, dp_held_rows),
                (dp_owned_vmem, eps, expert_pool_bytes, expert_views, ffn_norm_vmem),
                (gather_dp_rows, group_input_vmem, group_positions_smem, group_rows),
                (group_rows_per_pair, hidden_size, kda_row_chunk, kda_shard_swap),
                (kda_state_destination, kda_state_reads, kda_state_source),
                (kda_streams, key_output_ref, key_value_norms_vmem, latent_norms_vmem),
                (layer_weights, layers, metadata_semaphores, mix_dp_rows),
                (mixture_vmem, mla_scoped_scratch, mlp_residual_vmem),
                (moe_routed_reduce_scatter, moe_scoped_scratch),
                (moe_token_scatter_output, norm_semaphores, normalize),
                (normalize_dp_rows, normalized_vmem, overlap_kda_state_reads),
                (overlap_kda_state_writes, owned_rows, owned_rows_vmem),
                (owner_block_row, owner_indices, owner_prefix_row, owner_start),
                (owner_when, paged_pages, persist_kda_state, pool, pool_layout),
                (pool_ref, pool_semaphores, pool_view, position_ref),
                (prefetch_attention, prefetch_auxiliary, prefetch_metadata),
                (prefetch_norms, prefetch_router, prefix_vmem, project_attention_dp),
                (projection_expert_pool_vmem, query_down_weight_stream),
                (query_norms_vmem, recurrent_output_ref, recurrent_ref),
                (recurrent_state_vmem, request_attention, request_router, residual),
                (residual_reduction_order, router_semaphores, router_vmem),
                (routing_biases_vmem, sequence_parallel_pre_attention, sequence_rows),
                (shard_routing, slot_reads, slots_ref, sp8_owned_value, sp8_residual),
                (sp_owners, states, store_mixed_rows, stream_kda_state),
                (swap_kda_projections, table_ref, tokens_per_pair, value_output_ref),
                weight_stream,
            ),
            (layer_index, dense_layer),
        )

    if include_layer_zero:
        layer(jnp.int32(0), dense_layer=True)
    if pool is not None:
        pool.loop(1, layers)(layer)
        if pool.report:
            logger.info("%s", pool.timeline())
    else:
        pl.loop(1, layers)(layer)
    return (prefetch_auxiliary, prefetch_metadata, prefetch_router)


def _stack_body_stage_cm_stage_en(
    layer_weights: TensorLike,
    metadata_semaphores: TensorLike,
    metadata_vmem: TensorLike,
) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""

    def prefetch_metadata(
        index: TensorLike, *, mixture_metadata: TensorLike = True
    ) -> None:
        """Compute and stage Pallas VMEM state for prefetch metadata."""
        buffer_slot = index % 2

        @pl.when((index % 4 == _DCP_HEAD_COUNT) | (index == _MAX_STAGE_INDEX))
        def prefetch_mla_norms() -> None:
            """Compute and stage Pallas prefetch mla norms."""
            for slot, name in enumerate(("m_qnorm", "m_knorm")):
                tpu.make_async_copy(
                    layer_weights[name].at[index // 4],
                    metadata_vmem[slot].at[buffer_slot],
                    metadata_semaphores.at[buffer_slot, slot],
                ).start()

        if not mixture_metadata:
            return
        for slot, name in enumerate(("router_bias", "latent_norm"), 2):
            tpu.make_async_copy(
                layer_weights[name].at[index - 1],
                metadata_vmem[slot].at[buffer_slot],
                metadata_semaphores.at[buffer_slot, slot],
            ).start()

    return prefetch_metadata


def _stack_body_stage_cm_stage_eg(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (first_layer, include_layer_zero, layer_weights, pool, pool_managed),
        (prefetch_attention, prefetch_auxiliary, prefetch_metadata),
        (prefetch_norms, request_attention, request_router, router_semaphores),
        router_vmem,
    ) = stage_ctx

    def prefetch_router(index: TensorLike, *, initialize_padding: bool = False) -> None:
        """Compute and stage Pallas VMEM state for prefetch router."""
        if pool_managed:
            return
        _router_copy(
            layer_weights["router"].at[index - 1],
            router_vmem,
            router_semaphores,
            wait=False,
            initialize_padding=initialize_padding,
        )

    if include_layer_zero:
        if not pool_managed:

            @pl.when(jax.lax.axis_index("tp") % 2 == 1)
            def initialize_router_padding() -> None:
                """Compute and stage Pallas initialize router padding."""
                router_vmem[0, :, 384:] = jnp.zeros(
                    (router_vmem.shape[1], 128), jnp.bfloat16
                )
    else:
        prefetch_router(jnp.int32(1), initialize_padding=True)
    prefetch_norms(jnp.int32(first_layer))
    if pool is not None:
        request_attention(jnp.int32(first_layer))
    else:
        prefetch_attention(jnp.int32(first_layer))
    prefetch_auxiliary(jnp.int32(first_layer))
    prefetch_metadata(jnp.int32(first_layer), mixture_metadata=not include_layer_zero)
    if pool is not None and (not include_layer_zero):
        request_router(jnp.int32(1))
    return prefetch_router


def _stack_body_layer_stage_cg_stage_dy(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer stage."""
    (
        (attention_dp, attention_dp_ownership, attention_norm_vmem),
        (attention_residual_vmem, attention_token_scatter, attention_views),
        (attention_vmem, aux_hidden_layers, aux_output_ref, aux_semaphore),
        (auxiliary_semaphores, auxiliary_vmem, batch_size, block_rows),
        (block_rows_held, blocks_block_major, blocks_vmem),
        (cache_handoff_head_count, communication_workspace),
        (convolution_history_vmem, convolution_output_ref, convolution_ref),
        (dcp_owner_table, dcp_persistent, dp_attended_vmem),
        (dp_compressed_vmem, dp_owned_vmem, eps, ffn_norm_vmem),
        (gather_dp_rows, group_input_vmem, group_positions_smem, group_rows),
        (group_rows_per_pair, kda_row_chunk, kda_shard_swap),
        (kda_state_destination, kda_state_reads, kda_state_source),
        (kda_streams, key_output_ref, key_value_norms_vmem, layer_index),
        (layer_weights, metadata_semaphores, mix_dp_rows, mla_scoped_scratch),
        (mlp_residual_vmem, norm_semaphores, normalize, normalize_dp_rows),
        (normalized_vmem, overlap_kda_state_reads, overlap_kda_state_writes),
        (owned_rows, owned_rows_vmem, owner_block_row, owner_indices),
        (owner_prefix_row, owner_start, owner_when, paged_pages),
        (persist_kda_state, pool, pool_layout, pool_ref, pool_semaphores),
        (pool_view, position_ref, prefix_vmem, project_attention_dp),
        (query_down_weight_stream, query_norms_vmem, recurrent_output_ref),
        (recurrent_ref, recurrent_state_vmem, residual),
        (residual_reduction_order, sequence_parallel_pre_attention),
        (sequence_rows, slot_reads, slots_ref, sp8_owned_value, sp8_residual),
        (sp_owners, states, store_mixed_rows, stream_kda_state),
        (swap_kda_projections, table_ref, tokens_per_pair, value_output_ref),
        weight_stream,
    ) = stage_ctx
    mixture_values = _NONE_STAGE_VAL
    prefix_values = _NONE_STAGE_VAL
    attention_base = pool.offset("attention") if pool is not None else 0
    kda_projection_vmem, mla_projection_vmem = attention_views(attention_base)

    def capture_aux_hidden(
        source: TensorLike = None, *, gather_owned_rows: bool = False
    ) -> None:
        """Compute and stage Pallas capture aux hidden."""
        return _stack_body_layer_capture_aux_hidden_stage_cf(
            (
                (attention_dp_ownership, aux_hidden_layers, aux_output_ref),
                (aux_semaphore, communication_workspace, gather_dp_rows, layer_index),
                (normalized_vmem, owned_rows_vmem, prefix_vmem, sp_owners),
            ),
            (source, gather_owned_rows),
        )

    norm_copies = [
        tpu.make_async_copy(
            layer_weights[name].at[layer_index], destination, norm_semaphores.at[index]
        )
        for index, (name, destination) in enumerate(
            (
                ("attn_res", attention_residual_vmem),
                ("mlp_res", mlp_residual_vmem),
                ("attn_norm", attention_norm_vmem),
                ("ffn_norm", ffn_norm_vmem),
            )
        )
    ]
    _stack_body_layer_stage_cg_stage_dt(
        (
            (attention_dp, attention_dp_ownership, attention_norm_vmem),
            (attention_residual_vmem, block_rows, block_rows_held),
            (blocks_block_major, blocks_vmem, communication_workspace),
            (dp_owned_vmem, eps, group_input_vmem, group_rows, kda_shard_swap),
            (layer_index, mix_dp_rows, norm_copies, normalize, normalize_dp_rows),
            (normalized_vmem, owned_rows, owned_rows_vmem, owner_block_row),
            (owner_indices, owner_prefix_row, owner_start, owner_when),
            (prefix_vmem, residual, residual_reduction_order),
            (sequence_parallel_pre_attention, sp8_owned_value, sp8_residual),
            (sp_owners, store_mixed_rows),
        )
    )

    def run_delta_attention() -> None:
        """Compute and stage Pallas run delta attention."""
        return _stack_body_layer_run_delta_attention_stage_cc(
            (
                (attention_base, attention_dp, attention_token_scatter),
                (attention_vmem, auxiliary_semaphores, auxiliary_vmem, batch_size),
                (communication_workspace, convolution_history_vmem),
                (convolution_output_ref, convolution_ref, dp_attended_vmem),
                (dp_compressed_vmem, eps, group_input_vmem, group_rows),
                (kda_projection_vmem, kda_row_chunk, kda_shard_swap),
                (kda_state_destination, kda_state_reads, kda_state_source),
                (kda_streams, layer_index, layer_weights, normalized_vmem),
                (overlap_kda_state_reads, overlap_kda_state_writes, persist_kda_state),
                (pool_semaphores, project_attention_dp, recurrent_output_ref),
                (recurrent_ref, recurrent_state_vmem, sequence_rows, slot_reads),
                (slots_ref, states, stream_kda_state, swap_kda_projections),
            )
        )

    def run_latent_attention() -> None:
        """Compute and stage Pallas run latent attention."""
        state_index = layer_index // 4
        buffer_slot = layer_index % 2
        query_norm_vmem = query_norms_vmem.at[buffer_slot]
        key_value_norm_vmem = key_value_norms_vmem.at[buffer_slot]
        layer_metadata_semaphores = metadata_semaphores.at[buffer_slot]
        dp_output = None
        if attention_dp > 1:
            output_view = pool_view("m_o", base=attention_base)
            dp_output = (
                output_view,
                weight_stream(
                    "m_o",
                    layer_weights["m_o"].at[state_index],
                    output_view,
                    offset=pool_layout["m_o"] + attention_base,
                ),
            )

        def compute(
            new_key_vmem: TensorLike, new_value_vmem: TensorLike, *workspace: TensorLike
        ) -> None:
            """Compute and stage Pallas VMEM state for compute."""
            return _stack_body_layer_run_latent_attentio_stage_bz(
                (
                    (attention_base, attention_dp, attention_token_scatter),
                    (attention_vmem, communication_workspace, dcp_owner_table),
                    (
                        dcp_persistent,
                        dp_attended_vmem,
                        dp_output,
                        eps,
                        group_input_vmem,
                    ),
                    (group_positions_smem, group_rows, group_rows_per_pair),
                    (key_output_ref, key_value_norm_vmem, layer_metadata_semaphores),
                    (layer_weights, mla_projection_vmem, normalized_vmem, paged_pages),
                    (pool_layout, pool_ref, pool_semaphores, pool_view, position_ref),
                    (query_down_weight_stream, query_norm_vmem, sequence_rows),
                    (state_index, table_ref, tokens_per_pair, value_output_ref),
                    weight_stream,
                ),
                ((new_key_vmem, new_value_vmem, workspace),),
            )

        pl.run_scoped(
            compute,
            tpu.VMEM((batch_size, cache_handoff_head_count, 256), jnp.bfloat16),
            tpu.VMEM((batch_size, cache_handoff_head_count, 128), jnp.bfloat16),
            *mla_scoped_scratch,
        )
        if dp_output is not None:
            project_attention_dp(*dp_output)

    if pool is not None:
        pool.acquire("attention")
    with _phase("p1_attention"):
        jax.lax.cond(
            (layer_index % 4 == _DCP_HEAD_COUNT) | (layer_index == _MAX_STAGE_INDEX),
            run_latent_attention,
            run_delta_attention,
        )
    if pool is not None:
        pool.release("attention")
    post_scope = _phase_start("q9_post_mix")
    with _phase("q9a_post_norm_wait"):
        norm_copies[1].wait()
        norm_copies[3].wait()
    return (
        (attention_base, capture_aux_hidden, kda_projection_vmem, mixture_values),
        (mla_projection_vmem, post_scope, prefix_values),
    )


def _stack_body_layer_capture_aux_hidden_stage_cf(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas capture aux hidden."""
    (
        (attention_dp_ownership, aux_hidden_layers, aux_output_ref),
        (aux_semaphore, communication_workspace, gather_dp_rows, layer_index),
        (normalized_vmem, owned_rows_vmem, prefix_vmem, sp_owners),
    ) = stage_ctx
    source, gather_owned_rows = stage_args
    source = prefix_vmem if source is None else source
    if gather_owned_rows and attention_dp_ownership:
        source = normalized_vmem
    for aux_slot, aux_layer in enumerate(aux_hidden_layers):

        @pl.when(layer_index == aux_layer - 1)
        def capture(aux_slot: TensorLike = aux_slot) -> None:
            """Compute and stage Pallas VMEM state for capture."""
            if gather_owned_rows:
                if attention_dp_ownership:
                    gather_dp_rows(normalized_vmem)
                else:
                    _gather_local_pair_rows(
                        owned_rows_vmem,
                        prefix_vmem,
                        communication_workspace.send_semaphores,
                        communication_workspace.receive_semaphores,
                        communication_workspace.shared_local_reduction_vmem,
                        owners=sp_owners,
                    )
            aux_copy = tpu.make_async_copy(
                source, aux_output_ref.at[aux_slot], aux_semaphore.at[0]
            )
            aux_copy.start()
            aux_copy.wait()


def _stack_body_layer_stage_cg_stage_dt(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for stack body layer stage."""
    (
        (attention_dp, attention_dp_ownership, attention_norm_vmem),
        (attention_residual_vmem, block_rows, block_rows_held),
        (blocks_block_major, blocks_vmem, communication_workspace),
        (dp_owned_vmem, eps, group_input_vmem, group_rows, kda_shard_swap),
        (layer_index, mix_dp_rows, norm_copies, normalize, normalize_dp_rows),
        (normalized_vmem, owned_rows, owned_rows_vmem, owner_block_row),
        (owner_indices, owner_prefix_row, owner_start, owner_when),
        (prefix_vmem, residual, residual_reduction_order),
        (sequence_parallel_pre_attention, sp8_owned_value, sp8_residual),
        (sp_owners, store_mixed_rows),
    ) = stage_ctx
    norm_copies[0].wait()
    norm_copies[2].wait()
    layer_scope = [_phase_start("q1_pre_mix")]
    if attention_dp_ownership:
        mix_dp_rows(
            attention_residual_vmem,
            (layer_index - 1) // 12 + 1,
            layer_index=layer_index,
            start_block=True,
        )
    elif sequence_parallel_pre_attention:
        rank = jax.lax.axis_index("tp")
        pair = rank % 8 // (8 // sp_owners)
        for owner_pair in owner_indices(pair):
            _stack_body_layer_stage_cg_stage_dq(
                (
                    (attention_residual_vmem, block_rows, block_rows_held),
                    (blocks_block_major, blocks_vmem, eps, layer_index, owned_rows),
                    (owned_rows_vmem, owner_block_row, owner_pair, owner_prefix_row),
                    (
                        owner_start,
                        owner_when,
                        pair,
                        prefix_vmem,
                        residual_reduction_order,
                    ),
                    store_mixed_rows,
                )
            )
    else:
        residual(normalized_vmem, attention_residual_vmem, (layer_index - 1) // 12 + 1)

        @pl.when(layer_index % 12 == 0)
        def start_residual_block() -> None:
            """Compute and stage Pallas start residual block."""
            block_mask = (
                jax.lax.broadcasted_iota(jnp.int32, blocks_vmem.shape, 1)
                == layer_index // 12
            )
            blocks_vmem[...] = jnp.where(
                block_mask,
                prefix_vmem[...][:, None, :].astype(jnp.float32),
                blocks_vmem[...],
            )

    _phase_end(layer_scope.pop())
    layer_scope.append(_phase_start("q2_pre_norm_gather"))
    if kda_shard_swap:
        normalize(normalized_vmem, attention_norm_vmem)
        data_group = jax.lax.axis_index("tp") // (32 // attention_dp)
        rows = normalized_vmem[...]
        group_input_vmem[...] = jnp.where(
            data_group == 0, rows[:group_rows], rows[group_rows:]
        )
    elif attention_dp_ownership:
        normalize_dp_rows(attention_norm_vmem, split=sp8_residual)
        with _phase("t0_dp_group_gather"):
            if sp8_residual:
                _gather_sp8_local_rows(
                    sp8_owned_value(),
                    group_input_vmem,
                    communication_workspace.shared_local_reduction_vmem,
                    communication_workspace.send_semaphores,
                    communication_workspace.receive_semaphores,
                )
            else:
                _gather_local_pair_rows(
                    dp_owned_vmem,
                    group_input_vmem,
                    communication_workspace.send_semaphores,
                    communication_workspace.receive_semaphores,
                    communication_workspace.shared_local_reduction_vmem,
                    packed=True,
                )
    elif sequence_parallel_pre_attention:
        normalize(owned_rows_vmem, attention_norm_vmem)
        _gather_local_pair_rows(
            owned_rows_vmem,
            normalized_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            communication_workspace.shared_local_reduction_vmem,
            owners=sp_owners,
        )
    else:
        normalize(normalized_vmem, attention_norm_vmem)
    _phase_end(layer_scope.pop())


def _stack_body_layer_stage_cg_stage_dq(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for stack body layer stage."""
    (
        (attention_residual_vmem, block_rows, block_rows_held),
        (blocks_block_major, blocks_vmem, eps, layer_index, owned_rows),
        (owned_rows_vmem, owner_block_row, owner_pair, owner_prefix_row),
        (owner_start, owner_when, pair, prefix_vmem, residual_reduction_order),
        store_mixed_rows,
    ) = stage_ctx

    @owner_when(pair, owner_pair)
    def prepare_owned_attention_rows(*, owner_pair: int = owner_pair) -> None:
        """Compute and stage Pallas prepare owned attention rows."""
        row_offset = owner_pair * owned_rows
        row_slice = pl.ds(owner_start(row_offset), owned_rows)

        def mix_row(local_row: TensorLike) -> TensorLike:
            """Compute and stage Pallas VMEM state for mix row."""
            global_row = row_offset + local_row
            return _residual_mixture_row(
                block_rows(owner_block_row(global_row, local_row)),
                owner_prefix_row(global_row, row_slice, local_row),
                attention_residual_vmem,
                (layer_index - 1) // 12 + 1,
                epsilon=eps,
                relaxed_reduction_order=residual_reduction_order == "relaxed",
            )

        store_mixed_rows(mix_row, owned_rows_vmem)

        @pl.when(layer_index % 12 == 0 if blocks_block_major else False)
        def start_owned_residual_block_major() -> None:
            """Compute and stage Pallas start owned residual block major."""
            for local_row in range(owned_rows):
                global_row = row_offset + local_row
                row = owner_block_row(global_row, local_row) % block_rows_held
                half = blocks_vmem.at[row // 2, pl.ds(row % 2 * 8, 8)]
                block_mask = (
                    jax.lax.broadcasted_iota(jnp.int32, half.shape, 0)
                    == layer_index // 12
                )
                half[...] = jnp.where(
                    block_mask,
                    owner_prefix_row(global_row, row_slice, local_row),
                    half[...],
                )

        @pl.when(False if blocks_block_major else layer_index % 12 == 0)
        def start_owned_residual_block() -> None:
            """Compute and stage Pallas start owned residual block."""
            block_mask = (
                jax.lax.broadcasted_iota(jnp.int32, blocks_vmem[row_slice].shape, 1)
                == layer_index // 12
            )
            blocks_vmem[row_slice] = jnp.where(
                block_mask,
                prefix_vmem[row_slice][:, None, :].astype(jnp.float32),
                blocks_vmem[row_slice],
            )


def _stack_body_layer_run_delta_attention_stage_cc(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas run delta attention."""
    (
        (attention_base, attention_dp, attention_token_scatter),
        (attention_vmem, auxiliary_semaphores, auxiliary_vmem, batch_size),
        (communication_workspace, convolution_history_vmem),
        (convolution_output_ref, convolution_ref, dp_attended_vmem),
        (dp_compressed_vmem, eps, group_input_vmem, group_rows),
        (kda_projection_vmem, kda_row_chunk, kda_shard_swap),
        (kda_state_destination, kda_state_reads, kda_state_source),
        (kda_streams, layer_index, layer_weights, normalized_vmem),
        (overlap_kda_state_reads, overlap_kda_state_writes, persist_kda_state),
        (pool_semaphores, project_attention_dp, recurrent_output_ref),
        (recurrent_ref, recurrent_state_vmem, sequence_rows, slot_reads),
        (slots_ref, states, stream_kda_state, swap_kda_projections),
    ) = stage_ctx
    state_index = layer_index - layer_index // 4
    auxiliary_sources = tuple(layer_weights[name].at[state_index] for name in KDA[4:])
    pending = tuple(
        (
            tpu.make_async_copy(source, destination, auxiliary_semaphores.at[index])
            for index, (source, destination) in enumerate(
                zip(auxiliary_sources, auxiliary_vmem[:4], strict=True)
            )
        )
    )
    for copy in pending:
        copy.wait()

    def execute(
        history_vmem: TensorLike,
        state_vmem: TensorLike,
        state_semaphores: TensorLike,
        state_reads: TensorLike = (),
    ) -> None:
        """Compute and stage Pallas VMEM state for execute."""
        return _stack_body_layer_run_delta_attention_stage_by(
            (
                (attention_base, attention_dp, attention_token_scatter),
                (attention_vmem, auxiliary_vmem, communication_workspace),
                (convolution_output_ref, dp_attended_vmem, dp_compressed_vmem, eps),
                (group_input_vmem, group_rows, kda_projection_vmem, kda_shard_swap),
                (kda_streams, layer_weights, normalized_vmem, overlap_kda_state_reads),
                (overlap_kda_state_writes, persist_kda_state, pool_semaphores),
                (project_attention_dp, recurrent_output_ref, sequence_rows),
                (state_index, swap_kda_projections),
            ),
            ((history_vmem, state_vmem, state_semaphores, state_reads),),
        )

    if persist_kda_state:
        state_sources = (
            kda_state_source(convolution_ref, state_index),
            kda_state_source(recurrent_ref, state_index),
        )
        state_pending = tuple(
            (
                tpu.make_async_copy(
                    source, destination, auxiliary_semaphores.at[index + 4]
                )
                for index, (source, destination) in enumerate(
                    zip(
                        state_sources,
                        (
                            kda_state_destination(convolution_history_vmem),
                            kda_state_destination(recurrent_state_vmem),
                        ),
                        strict=True,
                    )
                )
            )
        )
        for copy in state_pending:
            copy.wait()
        execute(convolution_history_vmem, recurrent_state_vmem, auxiliary_semaphores)
    elif stream_kda_state:
        _stack_body_layer_run_delta_attention_stage_do(
            (
                (attention_base, attention_dp, attention_token_scatter),
                (attention_vmem, auxiliary_vmem, batch_size, communication_workspace),
                (convolution_output_ref, convolution_ref, dp_attended_vmem),
                (dp_compressed_vmem, eps, group_input_vmem, group_rows),
                (kda_projection_vmem, kda_row_chunk, kda_streams, layer_weights),
                (normalized_vmem, pool_semaphores, project_attention_dp),
                (recurrent_output_ref, recurrent_ref, sequence_rows, slot_reads),
                (slots_ref, state_index, states),
            )
        )
    else:

        def execute_scoped(
            history_vmem: TensorLike,
            state_vmem: TensorLike,
            state_semaphores: TensorLike,
        ) -> None:
            """Compute and stage Pallas VMEM state for execute scoped."""
            state_copies = (
                *kda_state_reads(
                    convolution_ref, history_vmem, state_semaphores.at[0], state_index
                ),
                *kda_state_reads(
                    recurrent_ref, state_vmem, state_semaphores.at[1], state_index
                ),
            )
            for copy in state_copies:
                copy.start()
            if not overlap_kda_state_reads:
                for copy in state_copies:
                    copy.wait()
            execute(history_vmem, state_vmem, state_semaphores, state_copies)

        pl.run_scoped(
            execute_scoped,
            tpu.VMEM((batch_size, *states[0].shape[2:]), states[0].dtype),
            tpu.VMEM((batch_size, *states[1].shape[2:]), states[1].dtype),
            tpu.SemaphoreType.DMA((2,)),
        )


def _stack_body_layer_run_delta_attention_stage_by(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for execute."""
    (
        (attention_base, attention_dp, attention_token_scatter),
        (attention_vmem, auxiliary_vmem, communication_workspace),
        (convolution_output_ref, dp_attended_vmem, dp_compressed_vmem, eps),
        (group_input_vmem, group_rows, kda_projection_vmem, kda_shard_swap),
        (kda_streams, layer_weights, normalized_vmem, overlap_kda_state_reads),
        (overlap_kda_state_writes, persist_kda_state, pool_semaphores),
        (project_attention_dp, recurrent_output_ref, sequence_rows),
        (state_index, swap_kda_projections),
    ) = stage_ctx
    ((history_vmem, state_vmem, state_semaphores, state_reads),) = stage_args
    state_writes = (
        tpu.make_async_copy(
            history_vmem,
            convolution_output_ref.at[:, state_index],
            state_semaphores.at[0],
        ),
        tpu.make_async_copy(
            state_vmem, recurrent_output_ref.at[:, state_index], state_semaphores.at[1]
        ),
    )

    def wait_for_state() -> None:
        """Compute and stage Pallas wait for state."""
        for copy in state_reads:
            copy.wait()

    _stack_body_layer_run_delta_attention_stage_dh(
        (
            (attention_base, attention_dp, attention_token_scatter),
            (attention_vmem, auxiliary_vmem, communication_workspace),
            (dp_attended_vmem, dp_compressed_vmem, eps, group_input_vmem),
            (group_rows, history_vmem, kda_projection_vmem, kda_shard_swap),
            (kda_streams, layer_weights, normalized_vmem, overlap_kda_state_reads),
            (overlap_kda_state_writes, pool_semaphores, project_attention_dp),
            (sequence_rows, state_index, state_vmem, state_writes),
            (swap_kda_projections, wait_for_state),
        )
    )
    if not persist_kda_state:
        for copy in state_writes:
            copy.wait()


def _stack_body_layer_run_delta_attention_stage_dh(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    (
        (attention_base, attention_dp, attention_token_scatter),
        (attention_vmem, auxiliary_vmem, communication_workspace),
        (dp_attended_vmem, dp_compressed_vmem, eps, group_input_vmem),
        (group_rows, history_vmem, kda_projection_vmem, kda_shard_swap),
        (kda_streams, layer_weights, normalized_vmem, overlap_kda_state_reads),
        (overlap_kda_state_writes, pool_semaphores, project_attention_dp),
        (sequence_rows, state_index, state_vmem, state_writes),
        (swap_kda_projections, wait_for_state),
    ) = stage_ctx

    def start_state_writes() -> None:
        """Compute and stage Pallas start state writes."""
        for copy in state_writes:
            copy.start()

    if attention_dp > 1:
        streams = kda_streams(state_index, attention_base)
        swapped = swap_kda_projections(streams["input"]) if kda_shard_swap else None
        for order in range(attention_dp):
            _stack_body_layer_run_delta_attention_stage_dg(
                (
                    (attention_dp, auxiliary_vmem, communication_workspace),
                    (dp_attended_vmem, dp_compressed_vmem, eps, group_input_vmem),
                    (group_rows, history_vmem, kda_projection_vmem, kda_shard_swap),
                    (layer_weights, order, overlap_kda_state_reads),
                    (overlap_kda_state_writes, pool_semaphores, sequence_rows),
                    (start_state_writes, state_index, state_vmem, streams, swapped),
                    wait_for_state,
                )
            )
        project_attention_dp(kda_projection_vmem[3], streams.get("output"))
    else:
        kda(
            normalized_vmem,
            attention_vmem,
            history_vmem,
            state_vmem,
            weights=KimiDeltaAttentionWeights(
                layer_weights["k_gate"].at[state_index],
                kda_projection_vmem[0],
                layer_weights["k_projection"].at[state_index],
                kda_projection_vmem[1],
                layer_weights["k_fb"].at[state_index],
                kda_projection_vmem[2],
                layer_weights["k_o"].at[state_index],
                kda_projection_vmem[3],
                auxiliary_vmem[0],
                auxiliary_vmem[1],
                auxiliary_vmem[2],
                auxiliary_vmem[3],
                pool_semaphores,
            ),
            convolution_history_ref=history_vmem,
            recurrent_state_ref=state_vmem,
            communication_workspace=communication_workspace,
            epsilon=eps,
            synchronize_devices=False,
            weights_are_prefetched=True,
            token_scatter_output=attention_token_scatter,
            sequence_rows=sequence_rows,
            wait_for_state=wait_for_state if overlap_kda_state_reads else None,
            on_state_updated=start_state_writes if overlap_kda_state_writes else None,
            streams=kda_streams(state_index, attention_base),
        )
    if not overlap_kda_state_writes:
        start_state_writes()


def _stack_body_layer_run_delta_attention_stage_dg(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    (
        (attention_dp, auxiliary_vmem, communication_workspace),
        (dp_attended_vmem, dp_compressed_vmem, eps, group_input_vmem),
        (group_rows, history_vmem, kda_projection_vmem, kda_shard_swap),
        (layer_weights, order, overlap_kda_state_reads),
        (overlap_kda_state_writes, pool_semaphores, sequence_rows),
        (start_state_writes, state_index, state_vmem, streams, swapped),
        wait_for_state,
    ) = stage_ctx
    shard = order
    if kda_shard_swap:
        own_shard = jax.lax.axis_index("tp") // (32 // attention_dp)
        shard = (own_shard + order) % attention_dp
    rows = pl.ds(shard * group_rows, group_rows)
    kda(
        group_input_vmem,
        group_input_vmem,
        history_vmem.at[rows],
        state_vmem.at[rows],
        weights=KimiDeltaAttentionWeights(
            layer_weights["k_gate"].at[state_index],
            kda_projection_vmem[0],
            layer_weights["k_projection"].at[state_index],
            kda_projection_vmem[1],
            layer_weights["k_fb"].at[state_index],
            kda_projection_vmem[2],
            layer_weights["k_o"].at[state_index],
            kda_projection_vmem[3],
            auxiliary_vmem[0].at[shard],
            auxiliary_vmem[1].at[shard],
            auxiliary_vmem[2].at[shard],
            auxiliary_vmem[3],
            pool_semaphores,
        ),
        convolution_history_ref=history_vmem.at[rows],
        recurrent_state_ref=state_vmem.at[rows],
        communication_workspace=communication_workspace,
        epsilon=eps,
        synchronize_devices=False,
        weights_are_prefetched=True,
        wait_for_state=wait_for_state
        if overlap_kda_state_reads and order == 0
        else None,
        on_state_updated=start_state_writes
        if overlap_kda_state_writes and order == attention_dp - 1
        else None,
        sequence_rows=sequence_rows,
        streams=streams,
        shard_index=shard,
        attended_out=dp_attended_vmem.at[shard],
        gate_features_vmem=dp_compressed_vmem,
        stage_projections=True,
        projected_rows=None if swapped is None else swapped(order),
        first_call=order == 0,
    )


def _stack_body_layer_run_delta_attention_stage_do(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    (
        (attention_base, attention_dp, attention_token_scatter),
        (attention_vmem, auxiliary_vmem, batch_size, communication_workspace),
        (convolution_output_ref, convolution_ref, dp_attended_vmem),
        (dp_compressed_vmem, eps, group_input_vmem, group_rows),
        (kda_projection_vmem, kda_row_chunk, kda_streams, layer_weights),
        (normalized_vmem, pool_semaphores, project_attention_dp),
        (recurrent_output_ref, recurrent_ref, sequence_rows, slot_reads),
        (slots_ref, state_index, states),
    ) = stage_ctx

    def execute_streamed_dp(
        history_buffers: TensorLike,
        state_buffers: TensorLike,
        read_semaphores: TensorLike,
        write_semaphores: TensorLike,
    ) -> None:
        """Compute and stage Pallas execute streamed dp."""
        return _stack_body_layer_run_delta_attention_stage_bx(
            (
                (attention_base, attention_dp, auxiliary_vmem),
                (communication_workspace, convolution_output_ref, convolution_ref),
                (dp_attended_vmem, dp_compressed_vmem, eps, group_input_vmem),
                (group_rows, kda_projection_vmem, kda_row_chunk, kda_streams),
                (layer_weights, pool_semaphores, project_attention_dp),
                (recurrent_output_ref, recurrent_ref, sequence_rows, slot_reads),
                (slots_ref, state_index),
            ),
            ((history_buffers, state_buffers, read_semaphores, write_semaphores),),
        )

    def execute_streamed(
        history_buffers: TensorLike,
        state_buffers: TensorLike,
        read_semaphores: TensorLike,
        write_semaphores: TensorLike,
    ) -> None:
        """Compute and stage Pallas VMEM state for execute streamed."""
        state_stream = KdaStateStream(
            convolution_ref.at[:, state_index],
            recurrent_ref.at[:, state_index],
            convolution_output_ref.at[:, state_index],
            recurrent_output_ref.at[:, state_index],
            history_buffers,
            state_buffers,
            read_semaphores,
            write_semaphores,
            chunk=kda_row_chunk,
            chunks=batch_size // kda_row_chunk,
            slots=slots_ref if slot_reads else None,
            sequence_rows=sequence_rows,
        )
        state_stream.prime()
        kda(
            normalized_vmem,
            attention_vmem,
            convolution_output_ref.at[:, state_index],
            recurrent_output_ref.at[:, state_index],
            weights=KimiDeltaAttentionWeights(
                layer_weights["k_gate"].at[state_index],
                kda_projection_vmem[0],
                layer_weights["k_projection"].at[state_index],
                kda_projection_vmem[1],
                layer_weights["k_fb"].at[state_index],
                kda_projection_vmem[2],
                layer_weights["k_o"].at[state_index],
                kda_projection_vmem[3],
                auxiliary_vmem[0],
                auxiliary_vmem[1],
                auxiliary_vmem[2],
                auxiliary_vmem[3],
                pool_semaphores,
            ),
            convolution_history_ref=convolution_ref.at[:, state_index],
            recurrent_state_ref=recurrent_ref.at[:, state_index],
            communication_workspace=communication_workspace,
            epsilon=eps,
            synchronize_devices=False,
            weights_are_prefetched=True,
            token_scatter_output=attention_token_scatter,
            sequence_rows=sequence_rows,
            streams=kda_streams(state_index, attention_base),
            state_stream=state_stream,
        )

    state_buffer_count = (
        4
        if batch_size == _KDA_GROUP_WIDTH and attention_dp == 1 and (not KDA_STAGED)
        else 3
    )
    pl.run_scoped(
        execute_streamed_dp if attention_dp > 1 else execute_streamed,
        tpu.VMEM(
            (state_buffer_count, kda_row_chunk, *states[0].shape[2:]), states[0].dtype
        ),
        tpu.VMEM(
            (state_buffer_count, kda_row_chunk, *states[1].shape[2:]), states[1].dtype
        ),
        tpu.SemaphoreType.DMA((state_buffer_count, 2)),
        tpu.SemaphoreType.DMA((state_buffer_count, 2)),
    )


def _stack_body_layer_run_delta_attention_stage_dd(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    (
        (attention_base, attention_dp, convolution_output_ref),
        (convolution_ref, group_rows, history_buffers, kda_row_chunk),
        (kda_streams, read_semaphores, recurrent_output_ref, recurrent_ref),
        (sequence_rows, slot_reads, slots_ref, state_buffers, state_index),
        write_semaphores,
    ) = stage_ctx
    streams = kda_streams(state_index, attention_base)
    chunks_per_shard = group_rows // kda_row_chunk
    shared_state_stream = KdaStateStream(
        convolution_ref.at[:, state_index],
        recurrent_ref.at[:, state_index],
        convolution_output_ref.at[:, state_index],
        recurrent_output_ref.at[:, state_index],
        history_buffers,
        state_buffers,
        read_semaphores,
        write_semaphores,
        chunk=kda_row_chunk,
        chunks=attention_dp * chunks_per_shard,
        slots=slots_ref if slot_reads else None,
        sequence_rows=sequence_rows,
    )
    shared_state_stream.prime()
    input_stream = streams.get("input")
    rolled_shards = (
        attention_dp == _TP_RING_SIZE
        and group_rows in (24, 32)
        and (input_stream is not None)
        and (input_stream.slots is not None)
        and slot_reads
        and ("kda" not in DEBUG_SKIP)
    )
    per_shard = chunk_columns = 0
    if rolled_shards:
        np.testing.assert_(input_stream is not None)
        per_shard = input_stream.chunks // attention_dp
        chunk_columns = input_stream.sizes[0]
    return (
        (chunk_columns, chunks_per_shard, input_stream, per_shard, rolled_shards),
        (shared_state_stream, streams),
    )


def _stack_body_layer_run_delta_attention_stage_bx(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas execute streamed dp."""
    (
        (attention_base, attention_dp, auxiliary_vmem),
        (communication_workspace, convolution_output_ref, convolution_ref),
        (dp_attended_vmem, dp_compressed_vmem, eps, group_input_vmem),
        (group_rows, kda_projection_vmem, kda_row_chunk, kda_streams),
        (layer_weights, pool_semaphores, project_attention_dp),
        (recurrent_output_ref, recurrent_ref, sequence_rows, slot_reads),
        (slots_ref, state_index),
    ) = stage_ctx
    ((history_buffers, state_buffers, read_semaphores, write_semaphores),) = stage_args
    (
        (chunk_columns, chunks_per_shard, input_stream, per_shard, rolled_shards),
        (shared_state_stream, streams),
    ) = _stack_body_layer_run_delta_attention_stage_dd(
        (
            (attention_base, attention_dp, convolution_output_ref),
            (convolution_ref, group_rows, history_buffers, kda_row_chunk),
            (kda_streams, read_semaphores, recurrent_output_ref, recurrent_ref),
            (sequence_rows, slot_reads, slots_ref, state_buffers, state_index),
            write_semaphores,
        )
    )

    def run_rolled_shards(projection: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for run rolled shards.

        One KDA body for every held shard (traced index): the shared gate
        features first, then per shard its input projection chunks and the
        recurrence.
        """

        def shard_weights(shard: TensorLike) -> TensorLike:
            """Compute and stage Pallas VMEM state for shard weights."""
            return KimiDeltaAttentionWeights(
                layer_weights["k_gate"].at[state_index],
                kda_projection_vmem[0],
                layer_weights["k_projection"].at[state_index],
                kda_projection_vmem[1],
                layer_weights["k_fb"].at[state_index],
                kda_projection_vmem[2],
                layer_weights["k_o"].at[state_index],
                kda_projection_vmem[3],
                auxiliary_vmem[0].at[shard],
                auxiliary_vmem[1].at[shard],
                auxiliary_vmem[2].at[shard],
                auxiliary_vmem[3],
                pool_semaphores,
            )

        copies = shard_weights(0).make_projection_copies()
        copies.gate.wait()
        copies.feature_gate.wait()
        dp_compressed_vmem[:, :128] = _dot(
            group_input_vmem[...], kda_projection_vmem[0][...]
        ).astype(jnp.bfloat16)
        guard = communication_workspace.reduction_phase[0] >= 0

        @pl.loop(0, attention_dp)
        def shard_loop(shard: TensorLike) -> None:
            """Compute and stage Pallas VMEM state for shard loop."""

            @pl.loop(0, per_shard)
            def project(position: TensorLike) -> None:
                """Compute and stage Pallas VMEM state for project."""
                _traced_dot_position(
                    input_stream,
                    group_input_vmem,
                    shard * per_shard + position,
                    guard,
                    projection.at[position],
                )

            rows = pl.ds(pl.multiple_of(shard * group_rows, 8), group_rows)
            kda(
                group_input_vmem,
                group_input_vmem,
                convolution_output_ref.at[:, state_index].at[rows],
                recurrent_output_ref.at[:, state_index].at[rows],
                weights=shard_weights(shard),
                convolution_history_ref=convolution_ref.at[:, state_index],
                recurrent_state_ref=recurrent_ref.at[:, state_index],
                communication_workspace=communication_workspace,
                epsilon=eps,
                synchronize_devices=False,
                weights_are_prefetched=True,
                sequence_rows=sequence_rows,
                streams=streams,
                shard_index=shard,
                attended_out=dp_attended_vmem.at[shard],
                gate_features_vmem=dp_compressed_vmem,
                stage_projections=True,
                state_stream=KdaStateStreamSlice(
                    shared_state_stream,
                    first_chunk=shard * chunks_per_shard,
                    chunks=chunks_per_shard,
                ),
                projected_rows=jnp.concatenate(
                    [projection[c] for c in range(per_shard)], axis=1
                ),
                first_call=False,
            )

    _stack_body_layer_run_delta_attention_stage_df(
        chunk_columns, group_rows, per_shard, rolled_shards, run_rolled_shards
    )
    for shard in range(0 if rolled_shards else attention_dp):
        rows = pl.ds(shard * group_rows, group_rows)
        history_rows = convolution_ref.at[:, state_index]
        state_rows = recurrent_ref.at[:, state_index]
        if not slot_reads:
            history_rows = history_rows.at[rows]
            state_rows = state_rows.at[rows]
        state_stream = KdaStateStreamSlice(
            shared_state_stream,
            first_chunk=shard * chunks_per_shard,
            chunks=chunks_per_shard,
        )
        if "kda" in DEBUG_SKIP:
            for index in range(state_stream.chunks):
                state_stream.acquire(index)
                state_stream.release(index)
            state_stream.finish()
            continue
        kda(
            group_input_vmem,
            group_input_vmem,
            convolution_output_ref.at[:, state_index].at[rows],
            recurrent_output_ref.at[:, state_index].at[rows],
            weights=KimiDeltaAttentionWeights(
                layer_weights["k_gate"].at[state_index],
                kda_projection_vmem[0],
                layer_weights["k_projection"].at[state_index],
                kda_projection_vmem[1],
                layer_weights["k_fb"].at[state_index],
                kda_projection_vmem[2],
                layer_weights["k_o"].at[state_index],
                kda_projection_vmem[3],
                auxiliary_vmem[0].at[shard],
                auxiliary_vmem[1].at[shard],
                auxiliary_vmem[2].at[shard],
                auxiliary_vmem[3],
                pool_semaphores,
            ),
            convolution_history_ref=history_rows,
            recurrent_state_ref=state_rows,
            communication_workspace=communication_workspace,
            epsilon=eps,
            synchronize_devices=False,
            weights_are_prefetched=True,
            sequence_rows=sequence_rows,
            streams=streams,
            shard_index=shard,
            attended_out=dp_attended_vmem.at[shard],
            gate_features_vmem=dp_compressed_vmem,
            stage_projections=True,
            state_stream=state_stream,
        )
    _stack_body_layer_run_delta_attention_stage_de(
        group_rows,
        kda_projection_vmem,
        project_attention_dp,
        shared_state_stream,
        streams,
    )


def _stack_body_layer_run_delta_attention_stage_df(
    chunk_columns: TensorLike,
    group_rows: TensorLike,
    per_shard: TensorLike,
    rolled_shards: TensorLike,
    run_rolled_shards: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    if rolled_shards:
        pl.run_scoped(
            run_rolled_shards,
            tpu.VMEM((per_shard, group_rows, chunk_columns), jnp.bfloat16),
        )


def _stack_body_layer_run_delta_attention_stage_de(
    group_rows: TensorLike,
    kda_projection_vmem: TensorLike,
    project_attention_dp: TensorLike,
    shared_state_stream: TensorLike,
    streams: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    if group_rows > _ROUTED_EXPERTS_PER_TOKEN and "kda" not in DEBUG_SKIP:
        shared_state_stream.finish_rolled()
    else:
        shared_state_stream.finish()
    if "project" not in DEBUG_SKIP:
        project_attention_dp(kda_projection_vmem[3], streams.get("output"))


def _stack_body_layer_run_latent_attentio_stage_dj(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    (
        (attention_base, key_value_norm_vmem, layer_metadata_semaphores),
        (layer_weights, pool_view, query_norm_vmem, state_index, workspace),
    ) = stage_ctx
    count = len(mla_dcp.SCOPED_NAMES)
    workspace, dcp_scoped = (workspace[:-count], workspace[-count:])
    metadata_copies = (
        tpu.make_async_copy(
            layer_weights["m_qnorm"].at[state_index],
            query_norm_vmem,
            layer_metadata_semaphores.at[0],
        ),
        tpu.make_async_copy(
            layer_weights["m_knorm"].at[state_index],
            key_value_norm_vmem,
            layer_metadata_semaphores.at[1],
        ),
    )
    for copy in metadata_copies:
        copy.wait()
    if POOLED_WEIGHTS:
        projection_vmem = tuple(
            pool_view(name, base=attention_base if name == "m_o" else 0)
            for name in MLA[2:6]
        )
    else:
        projection_vmem, workspace = (workspace[:4], workspace[4:])
    return (dcp_scoped, projection_vmem, workspace)


def _stack_body_layer_run_latent_attentio_stage_bz(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for compute."""
    (
        (attention_base, attention_dp, attention_token_scatter),
        (attention_vmem, communication_workspace, dcp_owner_table),
        (dcp_persistent, dp_attended_vmem, dp_output, eps, group_input_vmem),
        (group_positions_smem, group_rows, group_rows_per_pair),
        (key_output_ref, key_value_norm_vmem, layer_metadata_semaphores),
        (layer_weights, mla_projection_vmem, normalized_vmem, paged_pages),
        (pool_layout, pool_ref, pool_semaphores, pool_view, position_ref),
        (query_down_weight_stream, query_norm_vmem, sequence_rows),
        (state_index, table_ref, tokens_per_pair, value_output_ref),
        weight_stream,
    ) = stage_ctx
    ((new_key_vmem, new_value_vmem, workspace),) = stage_args
    (dcp_scoped, projection_vmem, workspace) = (
        _stack_body_layer_run_latent_attentio_stage_dj(
            (
                (attention_base, key_value_norm_vmem, layer_metadata_semaphores),
                (layer_weights, pool_view, query_norm_vmem, state_index, workspace),
            )
        )
    )
    mla_streams = {
        key: stream
        for key, name, view in zip(
            ("query_up", "key_value_up", "output_gate", "output"),
            MLA[2:6],
            projection_vmem,
            strict=True,
        )
        if (
            stream := (
                dp_output[1]
                if dp_output and name == "m_o"
                else weight_stream(
                    name,
                    layer_weights[name].at[state_index],
                    view,
                    offset=pool_layout[name] + (attention_base if name == "m_o" else 0)
                    if POOLED_WEIGHTS
                    else None,
                )
            )
        )
        is not None
    }
    _stack_body_layer_run_latent_attentio_stage_dk(
        attention_base, mla_streams, query_down_weight_stream, state_index
    )
    mla_options = {
        "weights": MultiHeadLatentAttentionWeights(
            layer_weights["m_qa"].at[state_index],
            mla_projection_vmem[0],
            layer_weights["m_ka"].at[state_index],
            mla_projection_vmem[1],
            layer_weights["m_qb"].at[state_index],
            projection_vmem[0],
            layer_weights["m_kb"].at[state_index],
            projection_vmem[1],
            layer_weights["m_gate"].at[state_index],
            projection_vmem[2],
            layer_weights["m_o"].at[state_index],
            projection_vmem[3],
            query_norm_vmem,
            key_value_norm_vmem,
            pool_semaphores,
        ),
        "workspace": MultiHeadLatentAttentionWorkspace(*workspace),
        "output_projection_after_query_down": POOLED_WEIGHTS,
        "streams": mla_streams,
        "communication_workspace": communication_workspace,
        "epsilon": eps,
        "synchronize_devices": False,
        "weights_are_prefetched": True,
        "token_scatter_output": attention_token_scatter,
        "sequence_rows": sequence_rows,
    }
    if attention_dp > 1:
        data_group = jax.lax.axis_index("tp") // (32 // attention_dp)

        def group_table_row(row: TensorLike) -> TensorLike:
            """Compute and stage Pallas group table row."""
            return (
                row // group_rows_per_pair * tokens_per_pair
                + data_group * group_rows_per_pair
                + row % group_rows_per_pair
            )

        group_handoff = pl.ds(0, group_rows)
        mla_dcp_shards(
            group_positions_smem,
            group_input_vmem,
            new_key_vmem.at[group_handoff],
            new_value_vmem.at[group_handoff],
            dp_attended_vmem,
            weights=mla_options["weights"],
            streams=mla_streams,
            communication_workspace=communication_workspace,
            dcp=(
                mla_dcp.DcpCache(
                    pool_ref,
                    table_ref,
                    paged_pages,
                    state_index,
                    owner=None
                    if dcp_owner_table is None
                    else mla_dcp.owner_index(dcp_owner_table) % mla_dcp.GROUP,
                    table_row=group_table_row,
                ),
                mla_dcp.DcpWorkspace(*dcp_persistent, *dcp_scoped),
            ),
            epsilon=eps,
            sequence_rows=sequence_rows,
        )
    else:
        mla(
            position_ref,
            normalized_vmem,
            attention_vmem,
            new_key_vmem,
            new_value_vmem,
            dcp=(
                mla_dcp.DcpCache(
                    pool_ref,
                    table_ref,
                    paged_pages,
                    state_index,
                    owner=None
                    if dcp_owner_table is None
                    else mla_dcp.owner_index(dcp_owner_table),
                ),
                mla_dcp.DcpWorkspace(*dcp_persistent, *dcp_scoped),
            ),
            **mla_options,
        )
    _stack_body_layer_run_latent_attentio_stage_di(
        key_output_ref, new_key_vmem, new_value_vmem, state_index, value_output_ref
    )


def _stack_body_layer_run_latent_attentio_stage_dk(
    attention_base: TensorLike,
    mla_streams: TensorLike,
    query_down_weight_stream: TensorLike,
    state_index: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    query_down_stream = query_down_weight_stream(state_index, attention_base)
    if query_down_stream is not None:
        mla_streams["query_down"] = query_down_stream


def _stack_body_layer_run_latent_attentio_stage_di(
    key_output_ref: TensorLike,
    new_key_vmem: TensorLike,
    new_value_vmem: TensorLike,
    state_index: TensorLike,
    value_output_ref: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    commit_scope = _phase_start("q8_mla_commit")

    def hand_out_rows(semaphores: TensorLike) -> None:
        """Compute and stage Pallas hand out rows."""
        copies = (
            tpu.make_async_copy(
                new_key_vmem, key_output_ref.at[:, state_index], semaphores.at[0]
            ),
            tpu.make_async_copy(
                new_value_vmem, value_output_ref.at[:, state_index], semaphores.at[1]
            ),
        )
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    pl.run_scoped(hand_out_rows, tpu.SemaphoreType.DMA((2,)))
    _phase_end(commit_scope)


def _stack_body_layer_stage_cg(stage_ctx: TensorLike, stage_args: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for layer."""
    (
        (active_rows, alias_expert_scales, arena_router, attention_dp),
        (attention_dp_ownership, attention_norm_vmem, attention_residual_vmem),
        (attention_token_scatter, attention_views, attention_vmem),
        (aux_capture, aux_hidden_layers, aux_output_ref, aux_semaphore),
        (auxiliary_semaphores, auxiliary_vmem, batch_size, block_rows),
        (block_rows_held, blocks_block_major, blocks_vmem),
        (cache_handoff_head_count, communication_workspace),
        (convolution_history_vmem, convolution_output_ref, convolution_ref),
        (dcp_owner_table, dcp_persistent, dense_refs, dense_semaphores),
        (dense_slabs, dp_attended_vmem, dp_compressed_vmem, dp_held_rows),
        (dp_owned_vmem, eps, expert_pool_bytes, expert_views, ffn_norm_vmem),
        (gather_dp_rows, group_input_vmem, group_positions_smem, group_rows),
        (group_rows_per_pair, hidden_size, kda_row_chunk, kda_shard_swap),
        (kda_state_destination, kda_state_reads, kda_state_source),
        (kda_streams, key_output_ref, key_value_norms_vmem, latent_norms_vmem),
        (layer_weights, layers, metadata_semaphores, mix_dp_rows),
        (mixture_vmem, mla_scoped_scratch, mlp_residual_vmem),
        (moe_routed_reduce_scatter, moe_scoped_scratch),
        (moe_token_scatter_output, norm_semaphores, normalize),
        (normalize_dp_rows, normalized_vmem, overlap_kda_state_reads),
        (overlap_kda_state_writes, owned_rows, owned_rows_vmem),
        (owner_block_row, owner_indices, owner_prefix_row, owner_start),
        (owner_when, paged_pages, persist_kda_state, pool, pool_layout),
        (pool_ref, pool_semaphores, pool_view, position_ref),
        (prefetch_attention, prefetch_auxiliary, prefetch_metadata),
        (prefetch_norms, prefetch_router, prefix_vmem, project_attention_dp),
        (projection_expert_pool_vmem, query_down_weight_stream),
        (query_norms_vmem, recurrent_output_ref, recurrent_ref),
        (recurrent_state_vmem, request_attention, request_router, residual),
        (residual_reduction_order, router_semaphores, router_vmem),
        (routing_biases_vmem, sequence_parallel_pre_attention, sequence_rows),
        (shard_routing, slot_reads, slots_ref, sp8_owned_value, sp8_residual),
        (sp_owners, states, store_mixed_rows, stream_kda_state),
        (swap_kda_projections, table_ref, tokens_per_pair, value_output_ref),
        weight_stream,
    ) = stage_ctx
    layer_index, dense_layer = stage_args
    owned_prefix_vmem = scattered_mixture = _NONE_STAGE_VAL
    (
        (_attention_base, capture_aux_hidden, _kda_projection_vmem, mixture_values),
        (_mla_projection_vmem, post_scope, prefix_values),
    ) = _stack_body_layer_stage_cg_stage_dy(
        (
            (attention_dp, attention_dp_ownership, attention_norm_vmem),
            (attention_residual_vmem, attention_token_scatter, attention_views),
            (attention_vmem, aux_hidden_layers, aux_output_ref, aux_semaphore),
            (auxiliary_semaphores, auxiliary_vmem, batch_size, block_rows),
            (block_rows_held, blocks_block_major, blocks_vmem),
            (cache_handoff_head_count, communication_workspace),
            (convolution_history_vmem, convolution_output_ref, convolution_ref),
            (dcp_owner_table, dcp_persistent, dp_attended_vmem),
            (dp_compressed_vmem, dp_owned_vmem, eps, ffn_norm_vmem),
            (gather_dp_rows, group_input_vmem, group_positions_smem, group_rows),
            (group_rows_per_pair, kda_row_chunk, kda_shard_swap),
            (kda_state_destination, kda_state_reads, kda_state_source),
            (kda_streams, key_output_ref, key_value_norms_vmem, layer_index),
            (layer_weights, metadata_semaphores, mix_dp_rows, mla_scoped_scratch),
            (mlp_residual_vmem, norm_semaphores, normalize, normalize_dp_rows),
            (normalized_vmem, overlap_kda_state_reads, overlap_kda_state_writes),
            (owned_rows, owned_rows_vmem, owner_block_row, owner_indices),
            (owner_prefix_row, owner_start, owner_when, paged_pages),
            (persist_kda_state, pool, pool_layout, pool_ref, pool_semaphores),
            (pool_view, position_ref, prefix_vmem, project_attention_dp),
            (query_down_weight_stream, query_norms_vmem, recurrent_output_ref),
            (recurrent_ref, recurrent_state_vmem, residual),
            (residual_reduction_order, sequence_parallel_pre_attention),
            (sequence_rows, slot_reads, slots_ref, sp8_owned_value, sp8_residual),
            (sp_owners, states, store_mixed_rows, stream_kda_state),
            (swap_kda_projections, table_ref, tokens_per_pair, value_output_ref),
            weight_stream,
        )
    )
    if attention_dp_ownership:
        with _phase("q9b_post_rows"):
            mix_dp_rows(
                mlp_residual_vmem,
                layer_index // 12 + 1,
                layer_index=layer_index,
                attention_output=True,
            )
        normalize_dp_rows(ffn_norm_vmem, split=sp8_residual)
        _phase_end(post_scope)
        with _phase("t10_moe_input_gather"):
            gather_dp_rows(normalized_vmem, split=sp8_residual)
    elif attention_token_scatter:
        (
            (owned_attention_vmem, owned_normalized_vmem, pair, pair_split),
            prefix_values,
        ) = _stack_body_layer_stage_cg_stage_dr(
            (
                (communication_workspace, hidden_size, owned_rows, owned_rows_vmem),
                (prefix_values, prefix_vmem, sp_owners, tokens_per_pair),
            )
        )
        for owner_pair in owner_indices(pair):

            @owner_when(pair, owner_pair)
            def prepare_owned_rows(owner_pair: TensorLike = owner_pair) -> None:
                """Compute and stage Pallas prepare owned rows."""
                row_slice = _NONE_STAGE_VAL
                attention_rows = (
                    owned_attention_vmem
                    if pair_split == 1
                    else owned_attention_vmem.at[
                        pl.ds(
                            owner_start(owner_pair % pair_split * owned_rows),
                            owned_rows,
                        )
                    ]
                )
                row_offset = owner_pair * owned_rows
                if owned_rows == 1:
                    prefix_rows = prefix_values[owner_pair : owner_pair + 1]
                else:
                    row_slice = pl.ds(owner_start(row_offset), owned_rows)
                    prefix_rows = prefix_vmem[row_slice]
                updated_prefix = jnp.where(
                    layer_index % 12 == 0,
                    attention_rows[...],
                    prefix_rows + attention_rows[...],
                ).astype(jnp.bfloat16)
                if owned_rows == 1:
                    row_indices = jax.lax.broadcasted_iota(
                        jnp.int32, prefix_vmem.shape, 0
                    )
                    prefix_vmem[...] = jnp.where(
                        row_indices == owner_pair, updated_prefix, prefix_values
                    )
                else:
                    prefix_vmem[row_slice] = updated_prefix

                def mix_row(local_row: TensorLike) -> TensorLike:
                    """Compute and stage Pallas VMEM state for mix row."""
                    global_row = row_offset + local_row
                    return _residual_mixture_row(
                        block_rows(owner_block_row(global_row, local_row)),
                        updated_prefix[local_row : local_row + 1]
                        if owned_rows <= _PAIR_TUPLE_LEN
                        else owner_prefix_row(global_row, row_slice, local_row),
                        mlp_residual_vmem,
                        layer_index // 12 + 1,
                        epsilon=eps,
                        relaxed_reduction_order=residual_reduction_order == "relaxed",
                    )

                with _phase("q9b_post_rows"):
                    store_mixed_rows(mix_row, owned_normalized_vmem)

        normalize(owned_normalized_vmem, ffn_norm_vmem)
        _phase_end(post_scope)
        t10 = _phase_start("t10_moe_input_gather")
        _gather_local_pair_rows(
            owned_normalized_vmem,
            normalized_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            communication_workspace.shared_local_reduction_vmem,
            owners=sp_owners,
        )
        _phase_end(t10)
    else:
        _stack_body_layer_stage_cg_stage_ea(
            (
                (attention_vmem, aux_capture, capture_aux_hidden, ffn_norm_vmem),
                (layer_index, mlp_residual_vmem, normalize, normalized_vmem),
                (post_scope, prefix_vmem, residual),
            )
        )
    (_after_experts, run_dense_mlp, run_mixture) = _stack_body_layer_stage_cg_stage_dw(
        (
            (active_rows, alias_expert_scales, arena_router, attention_dp),
            (aux_capture, auxiliary_semaphores, batch_size, capture_aux_hidden),
            (communication_workspace, convolution_history_vmem),
            (convolution_output_ref, dense_layer, dense_refs, dense_semaphores),
            (dense_slabs, eps, expert_pool_bytes, expert_views, hidden_size),
            (latent_norms_vmem, layer_index, layer_weights, layers),
            (metadata_semaphores, mixture_vmem, moe_routed_reduce_scatter),
            (moe_token_scatter_output, normalized_vmem, owned_rows),
            (owned_rows_vmem, persist_kda_state, pool, pool_layout, pool_view),
            (prefetch_attention, prefetch_auxiliary, prefetch_metadata),
            (prefetch_norms, prefetch_router, projection_expert_pool_vmem),
            (recurrent_output_ref, recurrent_state_vmem, request_attention),
            (request_router, residual_reduction_order, router_semaphores),
            (router_vmem, routing_biases_vmem, sequence_rows, shard_routing),
            (sp_owners, weight_stream),
        )
    )
    (mixture_values, owned_prefix_vmem, prefix_values, scattered_mixture) = (
        _stack_body_layer_stage_cg_stage_du(
            (
                (attention_dp, attention_dp_ownership, attention_token_scatter),
                (aux_capture, capture_aux_hidden, communication_workspace),
                (dense_layer, dp_held_rows, dp_owned_vmem, group_rows_per_pair),
                (hidden_size, mixture_values, mixture_vmem, moe_scoped_scratch),
                (moe_token_scatter_output, owned_prefix_vmem, owned_rows),
                (owned_rows_vmem, owner_indices, owner_start, owner_when),
                (prefix_values, prefix_vmem, run_dense_mlp, run_mixture),
                (scattered_mixture, sequence_parallel_pre_attention, sp_owners),
            )
        )
    )


def _stack_body_layer_stage_cg_stage_dr(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer stage."""
    (
        (communication_workspace, hidden_size, owned_rows, owned_rows_vmem),
        (prefix_values, prefix_vmem, sp_owners, tokens_per_pair),
    ) = stage_ctx
    rank = jax.lax.axis_index("tp")
    pair = rank % 8 // (8 // sp_owners)
    owned_attention_vmem = _bf16_rows_view(
        communication_workspace.attention_output_vmem, tokens_per_pair, hidden_size
    )
    owned_normalized_vmem = owned_rows_vmem
    pair_split = sp_owners // 4
    if owned_rows == 1:
        prefix_values = prefix_vmem[...]
    return (
        (owned_attention_vmem, owned_normalized_vmem, pair, pair_split),
        prefix_values,
    )


def _stack_body_layer_stage_cg_stage_ea(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for stack body layer stage."""
    (
        (attention_vmem, aux_capture, capture_aux_hidden, ffn_norm_vmem),
        (layer_index, mlp_residual_vmem, normalize, normalized_vmem),
        (post_scope, prefix_vmem, residual),
    ) = stage_ctx
    prefix_vmem[...] = jnp.where(
        layer_index % 12 == 0,
        attention_vmem[...],
        prefix_vmem[...] + attention_vmem[...],
    )
    if aux_capture == "attention":
        capture_aux_hidden()
    residual(normalized_vmem, mlp_residual_vmem, layer_index // 12 + 1)
    normalize(normalized_vmem, ffn_norm_vmem)
    _phase_end(post_scope)


def _stack_body_layer_stage_cg_stage_dw(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer stage."""
    (
        (active_rows, alias_expert_scales, arena_router, attention_dp),
        (aux_capture, auxiliary_semaphores, batch_size, capture_aux_hidden),
        (communication_workspace, convolution_history_vmem),
        (convolution_output_ref, dense_layer, dense_refs, dense_semaphores),
        (dense_slabs, eps, expert_pool_bytes, expert_views, hidden_size),
        (latent_norms_vmem, layer_index, layer_weights, layers),
        (metadata_semaphores, mixture_vmem, moe_routed_reduce_scatter),
        (moe_token_scatter_output, normalized_vmem, owned_rows),
        (owned_rows_vmem, persist_kda_state, pool, pool_layout, pool_view),
        (prefetch_attention, prefetch_auxiliary, prefetch_metadata),
        (prefetch_norms, prefetch_router, projection_expert_pool_vmem),
        (recurrent_output_ref, recurrent_state_vmem, request_attention),
        (request_router, residual_reduction_order, router_semaphores),
        (router_vmem, routing_biases_vmem, sequence_rows, shard_routing),
        (sp_owners, weight_stream),
    ) = stage_ctx

    def after_experts() -> None:
        """Compute and stage Pallas VMEM state for after experts."""
        return _stack_body_layer_after_experts_stage_ce(
            (
                (auxiliary_semaphores, convolution_history_vmem),
                (convolution_output_ref, dense_layer, layer_index, layers),
                (persist_kda_state, pool, prefetch_attention, prefetch_auxiliary),
                (prefetch_metadata, prefetch_norms, prefetch_router),
                (recurrent_output_ref, recurrent_state_vmem, request_attention),
                request_router,
            )
        )

    def run_mixture(*resources: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for run mixture."""
        return _stack_body_layer_run_mixture_stage_cd(
            (
                (active_rows, after_experts, alias_expert_scales, arena_router),
                (attention_dp, batch_size, communication_workspace, eps),
                (expert_pool_bytes, expert_views, latent_norms_vmem, layer_index),
                (layer_weights, metadata_semaphores, mixture_vmem),
                (moe_routed_reduce_scatter, moe_token_scatter_output, normalized_vmem),
                (owned_rows_vmem, pool, pool_layout, pool_view),
                (residual_reduction_order, router_semaphores, router_vmem),
                (routing_biases_vmem, sequence_rows, shard_routing, sp_owners),
                weight_stream,
            ),
            resources,
        )

    def run_dense_mlp() -> None:
        """Compute and stage Pallas VMEM state for run dense mlp."""
        return _stack_body_layer_run_dense_mlp_stage_cb(
            (
                (after_experts, attention_dp, batch_size, communication_workspace),
                (dense_refs, dense_semaphores, dense_slabs, hidden_size, mixture_vmem),
                (moe_token_scatter_output, normalized_vmem, owned_rows),
                (owned_rows_vmem, pool, projection_expert_pool_vmem, sequence_rows),
                sp_owners,
            )
        )

    if aux_capture == "mixture_input":
        capture_aux_hidden(normalized_vmem)
    return (after_experts, run_dense_mlp, run_mixture)


def _stack_body_layer_after_experts_stage_ce(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for after experts."""
    (
        (auxiliary_semaphores, convolution_history_vmem),
        (convolution_output_ref, dense_layer, layer_index, layers),
        (persist_kda_state, pool, prefetch_attention, prefetch_auxiliary),
        (prefetch_metadata, prefetch_norms, prefetch_router),
        (recurrent_output_ref, recurrent_state_vmem, request_attention),
        request_router,
    ) = stage_ctx
    if persist_kda_state:

        @pl.when(
            (layer_index % 4 != _DCP_HEAD_COUNT) & (layer_index != _MAX_STAGE_INDEX)
        )
        def drain_state_writes() -> None:
            """Compute and stage Pallas drain state writes."""
            state_index = layer_index - layer_index // 4
            state_writes = (
                tpu.make_async_copy(
                    convolution_history_vmem,
                    convolution_output_ref.at[:, state_index],
                    auxiliary_semaphores.at[0],
                ),
                tpu.make_async_copy(
                    recurrent_state_vmem,
                    recurrent_output_ref.at[:, state_index],
                    auxiliary_semaphores.at[1],
                ),
            )
            for copy in state_writes:
                copy.wait()

    if pool is not None:
        if not dense_layer:
            pool.release("experts")
        has_next = layer_index + 1 < layers
        pl.when(has_next)(lambda: prefetch_norms(layer_index + 1))
        request_attention(layer_index + 1, predicate=has_next)

        @pl.when(has_next)
        def prefetch_next_layer_metadata() -> None:
            """Compute and stage Pallas prefetch next layer metadata."""
            prefetch_auxiliary(layer_index + 1)
            prefetch_metadata(layer_index + 1)

        request_router(layer_index + 1, predicate=has_next)
        return

    @pl.when(layer_index + 1 < layers)
    def prefetch_next_layer() -> None:
        """Compute and stage Pallas prefetch next layer."""
        prefetch_norms(layer_index + 1)
        prefetch_attention(layer_index + 1)
        prefetch_auxiliary(layer_index + 1)
        prefetch_metadata(layer_index + 1)
        prefetch_router(layer_index + 1)


def _stack_body_layer_run_mixture_stage_c_stage_dp(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    (
        (alias_expert_scales, arena_router, expert_pool_bytes, expert_views),
        (latent_norms_vmem, layer_index, layer_weights, metadata_semaphores),
        (mixture_vmem, moe_token_scatter_output, owned_rows_vmem, pool),
        (pool_view, resources, router_vmem, routing_biases_vmem),
    ) = stage_ctx
    latent_output_vmem = _NONE_STAGE_VAL
    buffer_slot = layer_index % 2
    bias_vmem = routing_biases_vmem.at[buffer_slot]
    latent_norm_vmem = latent_norms_vmem.at[buffer_slot]
    metadata_copies = (
        tpu.make_async_copy(
            layer_weights["router_bias"].at[layer_index - 1],
            bias_vmem,
            metadata_semaphores.at[buffer_slot, 2],
        ),
        tpu.make_async_copy(
            layer_weights["latent_norm"].at[layer_index - 1],
            latent_norm_vmem,
            metadata_semaphores.at[buffer_slot, 3],
        ),
    )
    for copy in metadata_copies:
        copy.wait()
    if POOLED_WEIGHTS:
        latent_output_vmem = pool_view("latent_up")
        resources = (latent_output_vmem, *resources)
    latent_output_vmem, latent_output_semaphore, expert_semaphores = resources[:3]
    layer_router_vmem, layer_expert_vmem = (router_vmem, expert_views())
    after_routing_scores = None
    if pool is not None:
        layer_router_vmem = arena_router(pool.acquire("router"))
        layer_expert_vmem = expert_views(
            pool.allocate("experts", expert_pool_bytes, replaces=("router",))
        )

        def after_routing_scores() -> None:
            """Compute and stage Pallas after routing scores."""
            pool.release("router")
            pool.acquire("experts")

    if alias_expert_scales:
        gate_up_vmem, gate_up_scales_vmem, down_vmem, down_scales_vmem = (
            layer_expert_vmem
        )
        workspace_resources = resources[3:]
    else:
        gate_up_vmem, down_vmem = layer_expert_vmem
        gate_up_scales_vmem, down_scales_vmem = resources[3:5]
        workspace_resources = resources[5:]
    mixture_output_vmem = owned_rows_vmem if moe_token_scatter_output else mixture_vmem
    return (
        (after_routing_scores, bias_vmem, down_scales_vmem, down_vmem),
        (expert_semaphores, gate_up_scales_vmem, gate_up_vmem, latent_norm_vmem),
        (latent_output_semaphore, latent_output_vmem, layer_router_vmem),
        (mixture_output_vmem, workspace_resources),
    )


def _stack_body_layer_run_mixture_stage_cd(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for run mixture."""
    (
        (active_rows, after_experts, alias_expert_scales, arena_router),
        (attention_dp, batch_size, communication_workspace, eps),
        (expert_pool_bytes, expert_views, latent_norms_vmem, layer_index),
        (layer_weights, metadata_semaphores, mixture_vmem),
        (moe_routed_reduce_scatter, moe_token_scatter_output, normalized_vmem),
        (owned_rows_vmem, pool, pool_layout, pool_view),
        (residual_reduction_order, router_semaphores, router_vmem),
        (routing_biases_vmem, sequence_rows, shard_routing, sp_owners),
        weight_stream,
    ) = stage_ctx
    resources = stage_args
    (
        (after_routing_scores, bias_vmem, down_scales_vmem, down_vmem),
        (expert_semaphores, gate_up_scales_vmem, gate_up_vmem, latent_norm_vmem),
        (latent_output_semaphore, latent_output_vmem, layer_router_vmem),
        (mixture_output_vmem, workspace_resources),
    ) = _stack_body_layer_run_mixture_stage_c_stage_dp(
        (
            (alias_expert_scales, arena_router, expert_pool_bytes, expert_views),
            (latent_norms_vmem, layer_index, layer_weights, metadata_semaphores),
            (mixture_vmem, moe_token_scatter_output, owned_rows_vmem, pool),
            (pool_view, resources, router_vmem, routing_biases_vmem),
        )
    )
    moe(
        normalized_vmem,
        mixture_output_vmem,
        weights=MixtureOfExpertsWeights(
            layer_weights["router"].at[layer_index - 1],
            layer_router_vmem,
            router_semaphores,
            bias_vmem,
            layer_weights["latent_down"].at[layer_index - 1],
            layer_weights["latent_up"].at[layer_index - 1],
            latent_output_vmem,
            latent_output_semaphore,
            latent_norm_vmem,
            layer_weights["shared_gu"].at[layer_index - 1],
            layer_weights["shared_down"].at[layer_index - 1],
            layer_weights["expert_gu"].at[layer_index - 1],
            gate_up_vmem,
            layer_weights["expert_gus"].at[layer_index - 1],
            gate_up_scales_vmem,
            layer_weights["expert_down"].at[layer_index - 1],
            down_vmem,
            layer_weights["expert_ds"].at[layer_index - 1],
            down_scales_vmem,
            expert_semaphores,
        ),
        workspace=MixtureOfExpertsWorkspace(*workspace_resources),
        communication_workspace=communication_workspace,
        epsilon=eps,
        synchronize_devices=False,
        routing_projection_is_prefetched=True,
        group_pending_expert_routes=batch_size in (6, *SCATTER_BATCHES)
        and residual_reduction_order == "relaxed",
        after_experts=after_experts,
        after_routing_scores=after_routing_scores,
        overlap_router_latent=attention_dp == _PAIR_TUPLE_LEN
        and batch_size == _VREG_LANE_COUNT
        and (sequence_rows == 1),
        overlap_shared_reduction=attention_dp == _PAIR_TUPLE_LEN
        and batch_size in (64, 128)
        and (sequence_rows == 1),
        token_scatter_output=moe_token_scatter_output,
        routed_reduce_scatter=moe_routed_reduce_scatter,
        prepare_weights_vmem=tuple(pool_view(name) for name in MOE_PREPARE_WEIGHTS)
        if POOLED_WEIGHTS
        else None,
        shared_gate_up_stream=weight_stream(
            "shared_gu",
            layer_weights["shared_gu"].at[layer_index - 1],
            pool_view("shared_gu"),
            offset=pool_layout["shared_gu"],
        )
        if POOLED_WEIGHTS
        else None,
        token_scatter_owners=sp_owners,
        owned_hidden_ref=owned_rows_vmem if shard_routing else None,
        live_rows=active_rows,
    )


def _stack_body_layer_run_dense_mlp_stage_cb(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for run dense mlp."""
    (
        (after_experts, attention_dp, batch_size, communication_workspace),
        (dense_refs, dense_semaphores, dense_slabs, hidden_size, mixture_vmem),
        (moe_token_scatter_output, normalized_vmem, owned_rows),
        (owned_rows_vmem, pool, projection_expert_pool_vmem, sequence_rows),
        sp_owners,
    ) = stage_ctx
    gu_ref, down_ref = (dense_refs["dense_gu"], dense_refs["dense_down"])
    (
        (accumulator, _block_copy, _down_block, _gu_blocks, _gu_width, _k_block, _slab),
        (_slab_base, _slab_rows, _slabs, _total),
    ) = _stack_body_layer_run_dense_mlp_stage_stage_dm(
        (
            (after_experts, batch_size, dense_semaphores, dense_slabs, down_ref),
            (gu_ref, hidden_size, normalized_vmem, pool),
            projection_expert_pool_vmem,
        )
    )
    reduce_half = _stack_body_layer_run_dense_mlp_stage_stage_dn(
        accumulator, communication_workspace
    )
    if (
        moe_token_scatter_output
        and attention_dp == _PAIR_TUPLE_LEN
        and (sp_owners == _TP_RING_SIZE)
        and (batch_size in (64, 128))
        and (sequence_rows == 1)
    ):
        _dense_scatter_rows(
            accumulator,
            owned_rows_vmem,
            communication_workspace.shared_local_reduction_vmem,
            communication_workspace.shared_host_reduction_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            sp_owners,
        )
    else:
        reduced = (
            jnp.concatenate(
                [reduce_half(first) for first in range(0, batch_size, 8)], axis=0
            )
            if batch_size > _ROUTED_EXPERTS_PER_TOKEN
            else _all_reduce_rows(
                accumulator,
                communication_workspace.shared_local_reduction_vmem,
                communication_workspace.shared_host_reduction_vmem,
                communication_workspace.send_semaphores,
                communication_workspace.receive_semaphores,
            )
        )
        if moe_token_scatter_output:
            rank = jax.lax.axis_index("tp")
            pair = rank % 8 // (8 // sp_owners)
            for owner_pair in range(sp_owners):

                @pl.when(pair == owner_pair)
                def store_owned_dense_rows(*, owner_pair: int = owner_pair) -> None:
                    """Compute and stage Pallas store owned dense rows."""
                    owned_rows_vmem[...] = reduced[
                        owner_pair * owned_rows : (owner_pair + 1) * owned_rows
                    ].astype(jnp.bfloat16)
        else:
            mixture_vmem[...] = reduced.astype(jnp.bfloat16)


def _stack_body_layer_run_dense_mlp_stage_stage_dm(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    (
        (after_experts, batch_size, dense_semaphores, dense_slabs, down_ref),
        (gu_ref, hidden_size, normalized_vmem, pool),
        projection_expert_pool_vmem,
    ) = stage_ctx
    k_block, down_block = (1024, 384)
    gu_width = gu_ref.shape[1]
    gu_blocks = hidden_size // k_block
    down_blocks = down_ref.shape[0] // down_block
    np.testing.assert_(gu_ref.shape == (gu_blocks * k_block, gu_width))
    np.testing.assert_(down_ref.shape == (down_blocks * down_block, hidden_size))
    np.testing.assert_(gu_width == 2 * down_blocks * down_block)
    slab_rows = (
        max(
            pool_alias.storage_size((k_block, gu_width), gu_ref.dtype),
            pool_alias.storage_size((down_block, hidden_size), down_ref.dtype),
        )
        // 128
    )
    slabs = dense_slabs
    np.testing.assert_(slabs * slab_rows <= projection_expert_pool_vmem.shape[0])
    total = gu_blocks + down_blocks
    slab_base = 0
    if pool is not None:
        slab_base = pool.allocate("dense", slabs * slab_rows * 128) // 128
        pool.acquire("dense")

    def slab(index: TensorLike, shape: TensorLike, dtype: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for slab."""
        return pool_alias.view(
            projection_expert_pool_vmem.at[
                pl.ds(
                    slab_base + index * slab_rows,
                    pool_alias.storage_size(shape, dtype) // 128,
                ),
                :,
            ],
            shape,
            dtype,
        )

    def block_copy(k: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for block copy."""
        s = k % slabs
        if k < gu_blocks:
            return tpu.make_async_copy(
                gu_ref.at[pl.ds(k * k_block, k_block), :],
                slab(s, (k_block, gu_width), gu_ref.dtype),
                dense_semaphores.at[s],
            )
        r = k - gu_blocks
        return tpu.make_async_copy(
            down_ref.at[pl.ds(r * down_block, down_block), :],
            slab(s, (down_block, hidden_size), down_ref.dtype),
            dense_semaphores.at[s],
        )

    for k in range(min(slabs, total)):
        block_copy(k).start()

    def stream_blocks(
        projection_vmem: TensorLike = None, accumulator_vmem: TensorLike = None
    ) -> TensorLike:
        """Compute and stage Pallas VMEM state for stream blocks."""
        return _stack_body_layer_run_dense_mlp_strea_stage_ca(
            (
                (block_copy, down_block, down_ref, gu_blocks, gu_ref, gu_width),
                (hidden_size, k_block, normalized_vmem, slab, slabs, total),
            ),
            (projection_vmem, accumulator_vmem),
        )

    if batch_size <= _ROUTED_EXPERTS_PER_TOKEN:
        accumulator = stream_blocks()
    else:
        accumulator = pl.run_scoped(
            lambda projection_vmem, accumulator_vmem: (
                stream_blocks(projection_vmem, accumulator_vmem),
                accumulator_vmem[...],
            )[1],
            tpu.VMEM((batch_size, gu_width), jnp.float32),
            tpu.VMEM((batch_size, hidden_size), jnp.float32),
        )
    if pool is not None:
        pool.release("dense")
    after_experts()
    return (
        (accumulator, block_copy, down_block, gu_blocks, gu_width, k_block, slab),
        (slab_base, slab_rows, slabs, total),
    )


def _stack_body_layer_run_dense_mlp_strea_stage_ca(
    stage_ctx: TensorLike, stage_args: TensorLike
) -> TensorLike:
    """Compute and stage Pallas VMEM state for stream blocks."""
    (
        (block_copy, down_block, down_ref, gu_blocks, gu_ref, gu_width),
        (hidden_size, k_block, normalized_vmem, slab, slabs, total),
    ) = stage_ctx
    projection_vmem, accumulator_vmem = stage_args
    hidden_in = normalized_vmem[...]
    projection, activation, accumulator = (None, None, None)
    for k in range(total):
        s = k % slabs
        block_copy(k).wait()
        if k < gu_blocks:
            activation, projection = _stack_body_layer_run_dense_mlp_strea_stage_dl(
                (
                    (activation, gu_blocks, gu_ref, gu_width, hidden_in, k, k_block),
                    (projection, projection_vmem, s, slab),
                )
            )
        else:
            if activation is None:
                msg = "down blocks need the gate/up activation"
                raise RuntimeError(msg)
            r = k - gu_blocks
            if projection_vmem is not None:
                activation_block = projection_vmem[
                    :, r * down_block : (r + 1) * down_block
                ].astype(activation.dtype)
            else:
                activation_block = activation[:, r * down_block : (r + 1) * down_block]
            part = _dot(
                activation_block,
                slab(s, (down_block, hidden_size), down_ref.dtype)[...],
            )
            if accumulator_vmem is not None:
                if r == 0:
                    accumulator_vmem[...] = part
                else:
                    accumulator_vmem[...] += part
            else:
                accumulator = part if accumulator is None else accumulator + part
        if k + slabs < total:
            block_copy(k + slabs).start()
    return accumulator


def _stack_body_layer_run_dense_mlp_strea_stage_dl(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer run."""
    (
        (activation, gu_blocks, gu_ref, gu_width, hidden_in, k, k_block),
        (projection, projection_vmem, s, slab),
    ) = stage_ctx
    part = _dot(
        hidden_in[:, k * k_block : (k + 1) * k_block],
        slab(s, (k_block, gu_width), gu_ref.dtype)[...],
    )
    if projection_vmem is not None:
        if k == 0:
            projection_vmem[...] = part
        else:
            projection_vmem[...] += part
    else:
        projection = part if projection is None else projection + part
    if k == gu_blocks - 1:
        if projection_vmem is not None:
            projection = projection_vmem[...]
        half = gu_width // 2
        activation = _gated_activation(
            projection[:, :half].astype(jnp.bfloat16),
            projection[:, half:].astype(jnp.bfloat16),
        )
        if projection_vmem is not None:
            projection_vmem[:, :half] = activation.astype(jnp.float32)
    return (activation, projection)


def _stack_body_layer_run_dense_mlp_stage_stage_dn(
    accumulator: TensorLike, communication_workspace: TensorLike
) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer run."""

    def reduce_half(first: TensorLike) -> TensorLike:
        """Compute and stage Pallas VMEM state for reduce half."""
        if first:
            collectives32.barrier()
        return _all_reduce_rows(
            accumulator[first : first + 8],
            communication_workspace.shared_local_reduction_vmem,
            communication_workspace.shared_host_reduction_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
        )

    return reduce_half


def _stack_body_layer_stage_cg_stage_du(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer stage."""
    (
        (attention_dp, attention_dp_ownership, attention_token_scatter),
        (aux_capture, capture_aux_hidden, communication_workspace),
        (dense_layer, dp_held_rows, dp_owned_vmem, group_rows_per_pair),
        (hidden_size, mixture_values, mixture_vmem, moe_scoped_scratch),
        (moe_token_scatter_output, owned_prefix_vmem, owned_rows),
        (owned_rows_vmem, owner_indices, owner_start, owner_when),
        (prefix_values, prefix_vmem, run_dense_mlp, run_mixture),
        (scattered_mixture, sequence_parallel_pre_attention, sp_owners),
    ) = stage_ctx
    if dense_layer:
        if "dense" not in DEBUG_SKIP:
            run_dense_mlp()
    else:
        with _phase("p0_moe"):
            if "moe" not in DEBUG_SKIP:
                pl.run_scoped(run_mixture, *moe_scoped_scratch)
    finish_scope = _phase_start("q0_finish_rows")
    if attention_dp_ownership:
        group = jax.lax.axis_index("tp") // (32 // attention_dp)
        mixture = owned_rows_vmem[:dp_held_rows]
        for data_group in range(1, attention_dp if sp_owners > 1 else 1):
            first = data_group * group_rows_per_pair
            mixture = jnp.where(
                group == data_group,
                owned_rows_vmem[first : first + group_rows_per_pair],
                mixture,
            )
        finished = (prefix_vmem[:dp_held_rows] + mixture).astype(jnp.bfloat16)
        prefix_vmem[:dp_held_rows] = finished
        dp_owned_vmem[...] = finished
    elif attention_token_scatter:
        (mixture_values, owned_prefix_vmem, prefix_values, scattered_mixture) = (
            _stack_body_layer_stage_cg_stage_ds(
                (
                    (
                        communication_workspace,
                        hidden_size,
                        mixture_values,
                        mixture_vmem,
                    ),
                    (
                        moe_token_scatter_output,
                        owned_rows,
                        owned_rows_vmem,
                        owner_indices,
                    ),
                    (owner_start, owner_when, prefix_values, prefix_vmem),
                    (sequence_parallel_pre_attention, sp_owners),
                )
            )
        )
    else:
        prefix_vmem[...] += mixture_vmem[...]
    _phase_end(finish_scope)
    if aux_capture == "layer":
        capture_aux_hidden(gather_owned_rows=sequence_parallel_pre_attention)
    return (mixture_values, owned_prefix_vmem, prefix_values, scattered_mixture)


def _stack_body_layer_stage_cg_stage_ds(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body layer stage."""
    (
        (communication_workspace, hidden_size, mixture_values, mixture_vmem),
        (moe_token_scatter_output, owned_rows, owned_rows_vmem, owner_indices),
        (owner_start, owner_when, prefix_values, prefix_vmem),
        (sequence_parallel_pre_attention, sp_owners),
    ) = stage_ctx
    rank = jax.lax.axis_index("tp")
    pair = rank % 8 // (8 // sp_owners)
    owned_prefix_vmem = owned_rows_vmem
    scattered_mixture = owned_rows_vmem[...] if moe_token_scatter_output else None
    if owned_rows == 1:
        prefix_values = prefix_vmem[...]
        if not moe_token_scatter_output:
            mixture_values = mixture_vmem[...]
    _stack_body_layer_stage_cg_stage_ds_stage_ez(
        (
            (communication_workspace, hidden_size, mixture_values, mixture_vmem),
            (moe_token_scatter_output, owned_prefix_vmem, owned_rows),
            (owner_indices, owner_start, owner_when, pair, prefix_values),
            (prefix_vmem, scattered_mixture, sequence_parallel_pre_attention),
            sp_owners,
        )
    )
    return (mixture_values, owned_prefix_vmem, prefix_values, scattered_mixture)


def _stack_body_layer_stage_cg_stage_ds_stage_ez(stage_ctx: TensorLike) -> None:
    """Compute and stage Pallas VMEM state for stack body layer stage."""
    (
        (communication_workspace, hidden_size, mixture_values, mixture_vmem),
        (moe_token_scatter_output, owned_prefix_vmem, owned_rows),
        (owner_indices, owner_start, owner_when, pair, prefix_values),
        (prefix_vmem, scattered_mixture, sequence_parallel_pre_attention),
        sp_owners,
    ) = stage_ctx
    for owner_pair in owner_indices(pair):

        @owner_when(pair, owner_pair)
        def finish_owned_rows(*, owner_pair: int = owner_pair) -> None:
            """Compute and stage Pallas finish owned rows."""
            row_offset = owner_pair * owned_rows
            if owned_rows == 1:
                finished_prefix = (
                    prefix_values[owner_pair : owner_pair + 1]
                    + (
                        scattered_mixture
                        if moe_token_scatter_output
                        else mixture_values[owner_pair : owner_pair + 1]
                    )
                ).astype(jnp.bfloat16)
            else:
                row_slice = pl.ds(owner_start(row_offset), owned_rows)
                finished_prefix = (
                    prefix_vmem[row_slice]
                    + (
                        scattered_mixture
                        if moe_token_scatter_output
                        else mixture_vmem[row_slice]
                    )
                ).astype(jnp.bfloat16)
            owned_prefix_vmem[:, : hidden_size // 2] = finished_prefix[
                :, : hidden_size // 2
            ]
            owned_prefix_vmem[:, hidden_size // 2 :] = finished_prefix[
                :, hidden_size // 2 :
            ]

    if sequence_parallel_pre_attention:
        for owner_pair in owner_indices(pair):

            @owner_when(pair, owner_pair)
            def store_owned_prefix_rows(*, owner_pair: int = owner_pair) -> None:
                """Compute and stage Pallas store owned prefix rows."""
                row_offset = owner_pair * owned_rows
                row_slice = pl.ds(owner_start(row_offset), owned_rows)
                prefix_vmem[row_slice] = owned_prefix_vmem[...]
    else:
        _gather_local_pair_rows(
            owned_prefix_vmem,
            prefix_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            communication_workspace.shared_local_reduction_vmem,
            owners=sp_owners,
        )


def _stack_body_stage_cm_stage_et(
    headroom_pad: TensorLike, prefix_vmem: TensorLike
) -> None:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    if headroom_pad is not None:
        prefix_vmem[0:8, 0:128] += jnp.sum(
            headroom_pad[...].reshape(-1, 8, 128), axis=0
        ).astype(jnp.bfloat16)


def _stack_body_stage_cm_stage_ec(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack body stage cm."""
    (
        (attention_dp_ownership, attention_norm_vmem, attention_residual_vmem),
        (block_rows, blocks_output_ref, communication_workspace, copy),
        (dp_owned_vmem, eps, final_normalization_ref, gather_dp_rows, layers),
        (norm_semaphores, normalize, normalize_dp_rows, normalized_vmem),
        (output_ref, output_residual_ref, owned_rows, owned_rows_vmem),
        (owner_block_row, owner_indices, owner_prefix_row, owner_start),
        (owner_when, prefix_vmem, sp_owners, store_mixed_rows),
    ) = stage_ctx
    final_weight_copies = (
        tpu.make_async_copy(
            output_residual_ref, attention_residual_vmem, norm_semaphores.at[0]
        ),
        tpu.make_async_copy(
            final_normalization_ref, attention_norm_vmem, norm_semaphores.at[2]
        ),
    )
    for copy in final_weight_copies:
        copy.start()
    for copy in final_weight_copies:
        copy.wait()
    if attention_dp_ownership:
        final_block_count = (layers - 1) // 12 + 1

        def mix_row(row: TensorLike) -> TensorLike:
            """Compute and stage Pallas VMEM state for mix row."""
            values = jnp.concatenate(
                (block_rows(row, final_block_count), prefix_vmem[row : row + 1]), axis=0
            ).astype(jnp.float32)
            scores = jnp.sum(values * attention_residual_vmem[0], axis=1)
            scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + eps)
            return jnp.sum(
                jax.nn.softmax(scores)[:, None] * values, axis=0, keepdims=True
            ).astype(jnp.bfloat16)

        store_mixed_rows(mix_row, dp_owned_vmem)
        normalize_dp_rows(attention_norm_vmem)
        gather_dp_rows(normalized_vmem)
    else:
        rank = jax.lax.axis_index("tp")
        pair = rank % 8 // (8 // sp_owners)
        final_block_count = (layers - 1) // 12 + 1
        for owner_pair in owner_indices(pair):

            @owner_when(pair, owner_pair)
            def finish_owned_decoder_rows(*, owner_pair: int = owner_pair) -> None:
                """Compute and stage Pallas finish owned decoder rows."""
                row_offset = owner_pair * owned_rows
                row_slice = pl.ds(owner_start(row_offset), owned_rows)

                def mix_row(local_row: TensorLike) -> TensorLike:
                    """Compute and stage Pallas VMEM state for mix row."""
                    global_row = row_offset + local_row
                    values = jnp.concatenate(
                        (
                            block_rows(
                                owner_block_row(global_row, local_row),
                                final_block_count,
                            ),
                            owner_prefix_row(global_row, row_slice, local_row),
                        ),
                        axis=0,
                    ).astype(jnp.float32)
                    scores = jnp.sum(values * attention_residual_vmem[0], axis=1)
                    scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + eps)
                    return jnp.sum(
                        jax.nn.softmax(scores)[:, None] * values, axis=0, keepdims=True
                    ).astype(jnp.bfloat16)

                store_mixed_rows(mix_row, owned_rows_vmem)

        normalize(owned_rows_vmem, attention_norm_vmem)
        _gather_local_pair_rows(
            owned_rows_vmem,
            normalized_vmem,
            communication_workspace.send_semaphores,
            communication_workspace.receive_semaphores,
            communication_workspace.shared_local_reduction_vmem,
            owners=sp_owners,
        )

    def store_output(semaphore: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for store output."""
        copy = tpu.make_async_copy(normalized_vmem, output_ref, semaphore)
        copy.start()
        copy.wait()

    pl.run_scoped(store_output, tpu.SemaphoreType.DMA)
    blocks_output_ref[...] = jnp.zeros(blocks_output_ref.shape, jnp.uint8)
    return (copy, final_block_count)


def _stack_body_stage_cm_stage_ey(
    blocks_output_ref: TensorLike,
    blocks_vmem: TensorLike,
    output_ref: TensorLike,
    prefix_vmem: TensorLike,
) -> None:
    """Compute and stage Pallas VMEM state for stack body stage cm."""

    def store_output(semaphore: TensorLike) -> None:
        """Compute and stage Pallas VMEM state for store output."""
        copy = tpu.make_async_copy(prefix_vmem, output_ref, semaphore)
        copy.start()
        copy.wait()

    pl.run_scoped(store_output, tpu.SemaphoreType.DMA)
    blocks_output_ref[...] = blocks_vmem[...].astype(jnp.bfloat16)


def _stack_stage_cq(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cq."""
    (
        (alias_expert_scales, batch_size, hidden_size, include_layer_zero),
        (local_weights, pool_managed, pool_weights, weights),
    ) = stage_ctx
    pooled_expert_names = (
        ("expert_gu", "expert_gus", "expert_down", "expert_ds")
        if alias_expert_scales
        else ("expert_gu", "expert_down")
    )
    expert_pool_bytes = sum(
        pool_alias.storage_size(
            (4, *local_weights[name].shape[1:]), local_weights[name].dtype
        )
        for name in pooled_expert_names
    )
    dense_slabs = 4 if batch_size <= _GATED_GROUP_WIDTH else 3
    dense_slab_bytes = (
        max(
            pool_alias.storage_size(
                (1024, weights["dense_gu"].shape[1]), weights["dense_gu"].dtype
            ),
            pool_alias.storage_size((384, hidden_size), weights["dense_down"].dtype),
        )
        if include_layer_zero
        else 0
    )
    pool_layout = _pool_layout(
        pool_weights,
        alias_expert_scales=alias_expert_scales,
        pooled=POOLED_WEIGHTS,
        managed=pool_managed,
        dense_bytes=dense_slabs * dense_slab_bytes,
    )
    pool_bytes = pool_layout["pool_bytes"]
    if _config().vmem_report:

        def _mib(name: str, *, experts: bool = False) -> float:
            """Compute and stage Pallas VMEM state for mib."""
            value = pool_weights[name]
            shape = (4, *value.shape[1:]) if experts else value.shape
            return pool_alias.storage_size(shape, value.dtype) / 2**20

        region_a = {
            "kda": sum(_mib(n) for n in KDA[:4]),
            "mla": sum(_mib(n) for n in MLA[:2]),
            "experts": sum(_mib(n, experts=True) for n in pooled_expert_names),
        }
        region_b = {
            "mla_extra": sum(_mib(n) for n in MLA_EXTRA_WEIGHTS),
            "moe_prepare": sum(_mib(n) for n in MOE_PREPARE_WEIGHTS),
            "latent_up": _mib("latent_up"),
        }
        logger.info(
            "pool region A (MiB): %s", {k: round(v, 2) for k, v in region_a.items()}
        )
        if pool_managed:
            logger.info(
                "arena (managed region A): %.2f MiB, attention %.2f, dense slabs %.2f",
                pool_layout["arena_bytes"] / 2**20,
                pool_layout["attention_bytes"] / 2**20,
                dense_slabs * dense_slab_bytes / 2**20,
            )
        logger.info(
            "pool region B (MiB): %s %s",
            {k: round(v, 2) for k, v in region_b.items()},
            {
                n: round(_mib(n), 2)
                for n in (*MLA_EXTRA_WEIGHTS, *MOE_PREPARE_WEIGHTS, "latent_up")
            },
        )
    np.testing.assert_(pool_bytes >= expert_pool_bytes and pool_bytes % 128 == 0)
    return (dense_slab_bytes, dense_slabs, expert_pool_bytes, pool_bytes, pool_layout)


def _stack_stage_cp(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cp."""
    (
        (attention_dp, attention_dp_ownership, attention_token_scatter),
        (b4_attention_gather, batch_size, block_rows_held, blocks_block_major),
        (compact_outputs, group_rows, hidden_size, local_weights),
        (moe_routed_reduce_scatter, owned_rows, pair_split_gather),
        (persist_kda_state, pool_bytes, pool_managed, prefix, sequence_rows),
        (states, stream_config, weights),
    ) = stage_ctx
    return (
        tpu.VMEM(
            (-(-block_rows_held // 16) * 16, hidden_size)
            if attention_dp_ownership
            else prefix.shape,
            jnp.bfloat16,
        ),
        tpu.VMEM(
            (-(-block_rows_held // 2), 16, hidden_size)
            if blocks_block_major
            else (batch_size, 8, hidden_size),
            jnp.bfloat16 if blocks_block_major else jnp.float32,
        ),
        tpu.VMEM(prefix.shape, jnp.bfloat16),
        *(
            (tpu.VMEM((16, 128), jnp.bfloat16), tpu.VMEM((16, 128), jnp.bfloat16))
            if compact_outputs
            else (
                tpu.VMEM(prefix.shape, jnp.bfloat16),
                tpu.VMEM(prefix.shape, jnp.bfloat16),
            )
        ),
        *(tpu.VMEM((1, hidden_size), weights[name].dtype) for name in NORMS),
        tpu.VMEM((8, 128) if pool_managed else (1, hidden_size, 512), jnp.bfloat16),
        tpu.SemaphoreType.DMA((2,)),
        tpu.VMEM((pool_bytes // 128, 128), jnp.uint8),
        tpu.SemaphoreType.DMA((6,)),
        tpu.SemaphoreType.DMA((4,)),
        *(
            tpu.VMEM(local_weights[name].shape, local_weights[name].dtype)
            for name in KDA[4:]
        ),
        *(
            (
                tpu.VMEM((batch_size, *states[0].shape[2:]), states[0].dtype),
                tpu.VMEM((batch_size, *states[1].shape[2:]), states[1].dtype),
            )
            if persist_kda_state
            else (tpu.VMEM((1,), jnp.uint8), tpu.VMEM((1,), jnp.uint8))
        ),
        tpu.SemaphoreType.DMA((6,)),
        *(
            tpu.VMEM((2, *local_weights[name].shape), local_weights[name].dtype)
            for name in ("m_qnorm", "m_knorm", "router_bias", "latent_norm")
        ),
        tpu.SemaphoreType.DMA((2, 4)),
        tpu.VMEM(
            (owned_rows, hidden_size) if attention_token_scatter else (1,), jnp.bfloat16
        ),
        tpu.SemaphoreType.DMA((1,)),
        tpu.SemaphoreType.DMA((4,)),
        tpu.SemaphoreType.DMA((_stream_semaphore_count(stream_config),)),
        *(
            (
                tpu.VMEM((group_rows, hidden_size), jnp.bfloat16),
                tpu.VMEM((attention_dp, group_rows, 384), jnp.bfloat16),
                tpu.SMEM((group_rows,), jnp.int32),
                tpu.VMEM((group_rows, 1536), jnp.bfloat16),
                tpu.VMEM((group_rows, 640), jnp.bfloat16),
            )
            if attention_dp > 1
            else ()
        ),
        *mla_dcp.persistent_scratch(
            group_rows,
            heads=3 * attention_dp,
            cache_dtype=states[2].dtype,
            sequence_rows=sequence_rows,
        ),
        *communication_scratch(batch_size, attention_dp)[:-1],
        tpu.VMEM((batch_size, 4096), jnp.bfloat16),
        *projection_staging_scratch(batch_size, attention_dp),
        *routed_send_scratch(batch_size),
        *((tpu.SemaphoreType.REGULAR,) if moe_routed_reduce_scatter else ()),
        *(
            (tpu.SemaphoreType.DMA((1,)), tpu.SemaphoreType.DMA((1,)))
            if pair_split_gather
            else ()
        ),
        *((tpu.VMEM((4, 32, 128), jnp.float32),) if b4_attention_gather else ()),
        *(
            (tpu.VMEM((_config().vmem_pad_kb * 2, 128), jnp.float32),)
            if _config().vmem_pad_kb
            else ()
        ),
    )


def _stack_stage_da(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage da."""
    (
        (attention_dp, b4_attention_gather, batch_size, group_rows),
        (moe_routed_reduce_scatter, output_shapes, pair_split_gather, scratch),
    ) = stage_ctx
    scratch_names = (
        "prefix",
        "blocks",
        "normalized",
        "attention",
        "mixture",
        *NORMS,
        "router",
        "router_semaphores",
        "projection_expert_pool",
        "pool_semaphores",
        "norm_semaphores",
        *KDA[4:],
        "kda_history",
        "kda_state",
        "auxiliary_semaphores",
        "m_qnorm",
        "m_knorm",
        "router_bias",
        "latent_norm",
        "metadata_semaphores",
        "owned_rows",
        "aux_semaphore",
        "dense_semaphores",
        "stream_semaphores",
        *(
            (
                "dp_group_input",
                "dp_attended",
                "dp_group_positions",
                "dp_query_latent",
                "dp_compressed",
            )
            if attention_dp > 1
            else ()
        ),
        *mla_dcp.PERSISTENT_NAMES,
        "shared_local_reduction",
        "expert_local_reduction",
        "shared_host_reduction",
        "expert_host_reduction",
        "attention_features",
        "attention_output",
        "send_semaphores",
        "receive_semaphores",
        "reduction_phase",
        "latent_gather",
        *(
            ("projection_staging", "routed_send")
            if batch_size > _KDA_GROUP_WIDTH
            else ()
        ),
        *(("routed_consumed",) if moe_routed_reduce_scatter else ()),
        *(
            ("pair_send_semaphore", "pair_receive_semaphore")
            if pair_split_gather
            else ()
        ),
        *(("attention_gather",) if b4_attention_gather else ()),
        "headroom_pad",
    )
    persistent = [
        ("output window 1", tpu.VMEM(output_shapes[1].shape, output_shapes[1].dtype)),
        *zip(scratch_names, scratch),
    ]
    projection_reduction = (
        "output-projection reduction",
        tpu.VMEM((group_rows, 8192), jnp.float32)
        if attention_dp > 1
        else tpu.VMEM(
            (batch_size, 3584 if batch_size > _KDA_GROUP_WIDTH else 8192), jnp.float32
        ),
    )
    return (persistent, projection_reduction, scratch_names)


def _stack_stage_cn(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage cn."""
    (
        (alias_expert_scales, attention_dp, batch_size),
        (cache_handoff_head_count, group_rows, kda_row_chunk, local_weights),
        (mla_scoped_scratch, moe_scoped_scratch, persist_kda_state),
        (projection_reduction, shard_routing, states, stream_kda_state),
    ) = stage_ctx
    return {
        "MLA": (
            [
                (
                    "new-key handoff",
                    tpu.VMEM((batch_size, cache_handoff_head_count, 256), jnp.bfloat16),
                ),
                (
                    "new-value handoff",
                    tpu.VMEM((batch_size, cache_handoff_head_count, 128), jnp.bfloat16),
                ),
                *zip(
                    (
                        *(() if POOLED_WEIGHTS else MLA[2:6]),
                        "key cache tiles",
                        "value cache tiles",
                        "cache semaphores",
                        "query",
                        "output gate values",
                        "new key",
                        "new value",
                        "softmax maximum",
                        "softmax denominator",
                        "accumulator",
                        *mla_dcp.SCOPED_NAMES,
                    ),
                    mla_scoped_scratch,
                    strict=True,
                ),
            ],
            [
                *([] if attention_dp > 1 else [[projection_reduction]]),
                [("new-row hand-out semaphores", tpu.SemaphoreType.DMA((2,)))],
            ],
        ),
        **({"DP projection": ([projection_reduction], [])} if attention_dp > 1 else {}),
        "KDA": (
            []
            if persist_kda_state
            else [
                (
                    "kda history chunks",
                    tpu.VMEM((3, kda_row_chunk, *states[0].shape[2:]), states[0].dtype),
                ),
                (
                    "kda state chunks",
                    tpu.VMEM((3, kda_row_chunk, *states[1].shape[2:]), states[1].dtype),
                ),
            ]
            if stream_kda_state
            else [
                (
                    "kda history",
                    tpu.VMEM((batch_size, *states[0].shape[2:]), states[0].dtype),
                ),
                (
                    "kda state",
                    tpu.VMEM((batch_size, *states[1].shape[2:]), states[1].dtype),
                ),
            ],
            [
                [
                    projection_reduction,
                    *(
                        (
                            (
                                "DP staged projection",
                                tpu.VMEM(
                                    (group_rows // kda_row_chunk, kda_row_chunk, 1664),
                                    jnp.bfloat16,
                                ),
                            ),
                            (
                                "DP staged recurrence gate",
                                tpu.VMEM(
                                    (group_rows // kda_row_chunk, kda_row_chunk, 384),
                                    jnp.bfloat16,
                                ),
                            ),
                            (
                                "DP staged output gate",
                                tpu.VMEM((group_rows, 384), jnp.bfloat16),
                            ),
                        )
                        if attention_dp > 1 and group_rows > _ROUTED_EXPERTS_PER_TOKEN
                        else ()
                    ),
                    *(
                        (
                            (
                                "chunked attended",
                                tpu.VMEM((batch_size, 3, 128), jnp.float32),
                            ),
                        )
                        if batch_size > _ROUTED_EXPERTS_PER_TOKEN
                        else ()
                    ),
                ]
            ],
        ),
        "MoE": (
            list(
                zip(
                    (
                        *(() if POOLED_WEIGHTS else ("latent_up",)),
                        "latent-up semaphore",
                        "expert semaphores",
                        *(
                            ()
                            if alias_expert_scales
                            else ("gate/up scale slots", "down scale slots")
                        ),
                        "selected experts",
                        "selected probabilities",
                        "local route counts",
                        "probability totals",
                        "routing scores",
                        "shared outputs",
                        "routed outputs",
                        "gate/up accumulator",
                        "down accumulator",
                        "processed routes",
                        "compact source 1",
                        "compact source 2",
                        "shared send semaphores",
                        "shared receive semaphores",
                        "output transport",
                        "output send semaphores",
                        "output receive semaphores",
                        "latent send semaphores",
                        "latent receive semaphores",
                        *(
                            ("routing selection source", "routing selections")
                            if shard_routing
                            else ()
                        ),
                    ),
                    moe_scoped_scratch,
                    strict=True,
                )
            ),
            []
            if POOLED_WEIGHTS
            else [
                [
                    (
                        name,
                        tpu.VMEM(local_weights[name].shape, local_weights[name].dtype),
                    )
                    for name in MOE_PREPARE_WEIGHTS
                ]
            ],
        ),
    }


def _stack_stage_co(stage_ctx: TensorLike) -> TensorLike:
    """Compute and stage Pallas VMEM state for stack stage co."""
    (
        (alias_expert_scales, batch_size, dense_slab_bytes, dense_slabs),
        (hidden_size, persistent, pool_layout, pool_managed, pool_weights),
        (regions, scratch, scratch_names),
    ) = stage_ctx
    vmem_report, vmem_peak = _vmem_budget(persistent, regions)
    if pool_managed:
        demand = pool_layout["attention_bytes"] + pool_alias.storage_size(
            (1, hidden_size, 512), jnp.dtype(jnp.bfloat16)
        )
        shortfall = max(0, -(-(demand - pool_layout["arena_bytes"]) // 4096) * 4096)
        available = (
            (VMEM_LIMIT_BYTES - VMEM_SPILL_RESERVE_BYTES - vmem_peak) // 4096 * 4096
        )
        spare = (
            shortfall if shortfall <= available and batch_size < _VREG_LANE_COUNT else 0
        )
        if spare > 0:
            pool_layout = _pool_layout(
                pool_weights,
                alias_expert_scales=alias_expert_scales,
                pooled=POOLED_WEIGHTS,
                managed=True,
                dense_bytes=dense_slabs * dense_slab_bytes,
                arena_extra=spare,
            )
            pool_bytes = pool_layout["pool_bytes"]
            pool_spec = tpu.VMEM((pool_bytes // 128, 128), jnp.uint8)
            index = scratch_names.index("projection_expert_pool")
            scratch = (*scratch[:index], pool_spec, *scratch[index + 1 :])
            persistent = [
                (name, pool_spec if name == "projection_expert_pool" else spec)
                for name, spec in persistent
            ]
            vmem_report, vmem_peak = _vmem_budget(persistent, regions)
        if _config().vmem_report:
            logger.info(
                "arena with spare VMEM: %.2f MiB", pool_layout["arena_bytes"] / 2**20
            )
    if _config().vmem_report:
        logger.info("VMEM plan, batch %s:\n%s", batch_size, vmem_report)
    np.testing.assert_(
        not vmem_peak > VMEM_LIMIT_BYTES,
        f"The decoder's static VMEM plan exceeds the limit:\n{vmem_report}",
    )
    if vmem_peak + VMEM_SPILL_RESERVE_BYTES > VMEM_LIMIT_BYTES:
        warnings.warn(
            str.join(
                "",
                (
                    "The decoder's VMEM plan leaves less than the ",
                    "spill reserve:\n",
                    f"{vmem_report}",
                ),
            ),
            stacklevel=2,
        )
    return (pool_layout, scratch)


def decode_local(
    weights: TensorLike,
    states: TensorLike,
    tokens: TensorLike,
    positions: TensorLike,
    **stage_kwargs: TensorLike,
) -> TensorLike:
    """Decode one native batch tile with local TP32 weights and caches.

    ``states`` is ``(conv, recurrent, *pools)`` over vLLM's
    unified-pool tensors ``[blocks, 128, 2, 640]`` (read in place,
    never written here), MLA layer ``m`` in
    ``pools[dcp_pool_map[m]]``, ``block_tables`` (``[batch_size,
    pages]`` int32) the rows' tables, ``dcp_owner_table`` (mesh
    position -> page owner index, e.g. vLLM TP rank) which pages each
    rank holds. The returned states are ``(conv, recurrent, new_keys,
    new_values)``: each row's new latent for every MLA layer
    (``[batch, 24, H, 256]`` / ``[batch, 24, H, 128]``), for
    ``paged.commit_latent_rows`` to commit.

    Layer zero (its KDA and dense MLP) runs inside the fused stack
    after the XLA embedding gather. ``state_row`` (int32 scalar,
    requires ``sequence_rows == batch_size``) names the row of the KDA
    input states to start the sequence from, replacing a pre-selected
    state copy.

    ``sequence_rows`` groups consecutive rows into one sequence of
    consecutive tokens (speculative verification): KDA and MLA states
    lead with rows and sequences respectively, every row's KDA
    post-state is returned, and MLA pools receive all rows.
    ``aux_hidden_layers`` additionally returns the residual stream
    entering those layers as ``[len, batch, hidden]``.
    """
    (
        batch_size,
        layers,
        eps,
        residual_reduction_order,
        mla_cache_batch_tile_size,
        alias_expert_scales,
        attention_token_scatter,
        moe_routed_reduce_scatter,
        moe_token_scatter_output,
        sequence_parallel_pre_attention,
        gate_first_kda_projection,
        batched_decoder_boundaries,
        sequence_rows,
        aux_hidden_layers,
        aux_capture,
        state_row,
        attention_dp,
        attention_projection_schedule,
        mla_commit_tile_size,
        sequence_parallel_size,
        block_tables,
        dcp_pool_map,
        dcp_owner_table,
        state_slots,
        live_rows,
    ) = (
        stage_kwargs["batch_size"],
        stage_kwargs.get("layers", 93),
        stage_kwargs.get("eps", 1e-05),
        stage_kwargs.get("residual_reduction_order", "reference"),
        stage_kwargs.get("mla_cache_batch_tile_size", _NONE_STAGE_VAL),
        stage_kwargs.get("alias_expert_scales", False),
        stage_kwargs.get("attention_token_scatter", False),
        stage_kwargs.get("moe_routed_reduce_scatter", False),
        stage_kwargs.get("moe_token_scatter_output", False),
        stage_kwargs.get("sequence_parallel_pre_attention", False),
        stage_kwargs.get("gate_first_kda_projection", _NONE_STAGE_VAL),
        stage_kwargs.get("batched_decoder_boundaries", _NONE_STAGE_VAL),
        stage_kwargs.get("sequence_rows", 1),
        stage_kwargs.get("aux_hidden_layers", ()),
        stage_kwargs.get("aux_capture", "layer"),
        stage_kwargs.get("state_row", _NONE_STAGE_VAL),
        stage_kwargs.get("attention_dp", 1),
        stage_kwargs.get("attention_projection_schedule", "default"),
        stage_kwargs.get("mla_commit_tile_size", _NONE_STAGE_VAL),
        stage_kwargs.get("sequence_parallel_size", 4),
        stage_kwargs.get("block_tables", _NONE_STAGE_VAL),
        stage_kwargs.get("dcp_pool_map", _NONE_STAGE_VAL),
        stage_kwargs.get("dcp_owner_table", _NONE_STAGE_VAL),
        stage_kwargs.get("state_slots", _NONE_STAGE_VAL),
        stage_kwargs.get("live_rows", _NONE_STAGE_VAL),
    )
    np.testing.assert_(
        gate_first_kda_projection is not False,
        "The clean decoder always uses the split KDA gate projection",
    )
    if batched_decoder_boundaries is None:
        batched_decoder_boundaries = batch_size in SCATTER_BATCHES or attention_dp > 1
    np.testing.assert_(
        bool(1 <= batch_size <= MAX_BATCH),
        f"The native decoder supports batch sizes from 1 to {MAX_BATCH}",
    )
    aux_hidden_layers = tuple(aux_hidden_layers)
    rank = jax.lax.axis_index("tp")
    vocabulary = weights["embedding"].shape[0]
    if batched_decoder_boundaries:
        local_ids = tokens - rank * vocabulary
        local_embeddings = weights["embedding"][jnp.clip(local_ids, 0, vocabulary - 1)]
        embedding = kimi.tp_sum(
            jnp.where(
                ((local_ids >= 0) & (local_ids < vocabulary))[:, None],
                local_embeddings.astype(jnp.float32),
                0,
            )
        ).astype(jnp.bfloat16)
    else:

        def prepare_first_layer_row(_: TensorLike, token_id: TensorLike) -> TensorLike:
            """Compute and stage Pallas prepare first layer row."""
            local_id = token_id - rank * vocabulary
            embedding_row = weights["embedding"][jnp.clip(local_id, 0, vocabulary - 1)][
                None
            ]
            embedding_row = kimi.tp_sum(
                jnp.where(
                    (local_id >= 0) & (local_id < vocabulary),
                    embedding_row.astype(jnp.float32),
                    0,
                )
            ).astype(jnp.bfloat16)
            return ((), embedding_row[0])

        _, embedding = jax.lax.scan(prepare_first_layer_row, (), tokens)
    np.testing.assert_(
        not (state_row is not None and sequence_rows != batch_size),
        "state_row needs one sequence per tile",
    )
    prefix = embedding
    convolution_states, recurrent_states = states[:2]
    cache_states = tuple(states[2:])
    (
        prefix,
        blocks,
        convolution_states,
        recurrent_states,
        new_keys,
        new_values,
        aux_hidden,
    ) = stack(
        prefix,
        embedding,
        weights,
        (convolution_states, recurrent_states, *cache_states),
        positions,
        batch_size=batch_size,
        layers=layers,
        eps=eps,
        residual_reduction_order=residual_reduction_order,
        mla_cache_batch_tile_size=mla_cache_batch_tile_size,
        alias_expert_scales=alias_expert_scales,
        attention_token_scatter=attention_token_scatter,
        moe_token_scatter_output=moe_token_scatter_output,
        moe_routed_reduce_scatter=moe_routed_reduce_scatter,
        sequence_parallel_pre_attention=sequence_parallel_pre_attention,
        sequence_rows=sequence_rows,
        aux_hidden_layers=aux_hidden_layers,
        aux_capture=aux_capture,
        include_layer_zero=True,
        state_row=state_row,
        attention_dp=attention_dp,
        attention_projection_schedule=attention_projection_schedule,
        mla_commit_tile_size=mla_commit_tile_size,
        sequence_parallel_size=sequence_parallel_size,
        block_tables=block_tables,
        dcp_pool_map=dcp_pool_map,
        dcp_owner_table=dcp_owner_table,
        state_slots=state_slots,
        live_rows=live_rows,
    )
    if sequence_parallel_pre_attention and batched_decoder_boundaries:
        logits = jnp.dot(prefix, weights["lm_head"], preferred_element_type=jnp.float32)
    elif sequence_parallel_pre_attention:

        def project_decoder_row(_: TensorLike, hidden_row: TensorLike) -> TensorLike:
            """Compute and stage Pallas project decoder row."""
            logits_row = jnp.dot(
                hidden_row[None], weights["lm_head"], preferred_element_type=jnp.float32
            )
            return ((), logits_row[0])

        _, logits = jax.lax.scan(project_decoder_row, (), prefix)
    else:
        active_block_count = (layers - 1) // 12 + 1

        def finish_decoder(_: TensorLike, inputs: TensorLike) -> TensorLike:
            """Compute and stage Pallas VMEM state for finish decoder."""
            prefix_row, block_rows = inputs
            hidden_row = kimi.attention_residual(
                prefix_row,
                block_rows[:active_block_count],
                weights["output_res"][0],
                eps,
            )[None]
            hidden_row = kimi.rms(hidden_row, weights["final_norm"], eps)
            logits_row = jnp.dot(
                hidden_row, weights["lm_head"], preferred_element_type=jnp.float32
            )
            return ((), logits_row[0])

        _, logits = jax.lax.scan(finish_decoder, (), (prefix, blocks))
    new_states = (convolution_states, recurrent_states, new_keys, new_values)
    if aux_hidden_layers:
        return (logits[None], aux_hidden, new_states)
    return (logits[None], new_states)
