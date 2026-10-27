# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 decode megakernel components."""

from __future__ import annotations

import dataclasses
import functools
import math
from types import SimpleNamespace
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import typing as jax_typing
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.megakernel.common import collectives32, pool_alias
from vllm_torchtpu.megakernel.common.vmem_pool import VmemPool
from vllm_torchtpu.megakernel.kimi_k3 import kernels as kimi
from vllm_torchtpu.megakernel.kimi_k3.config import active as _config
from vllm_torchtpu.megakernel.kimi_k3.kernels import mla_dcp
from vllm_torchtpu.megakernel.kimi_k3.layout import CONV_HEAD_ROWS

_COMMUNICATION_SCRATCH_SPEC_COUNT = 10


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class _PaddedBf16Rows(pool_alias.Reinterpret):
    """Expose active rows while preserving the full native row capacity (a
    multiple of 16 rows)."""

    def _validate_shape(self, shape):
        rows, width = self.shape
        backing_rows = math.prod(shape) // width
        assert len(self.shape) == 2 and width % 128 == 0 and 1 <= rows <= backing_rows
        assert backing_rows * width == math.prod(shape) and backing_rows % 16 == 0, (shape, self.shape)


# Named phase scopes for XProf custom-call tracing (KernelConfig.phase_scopes=True).
PHASE_SCOPES = _config().phase_scopes


def _phase(name):
    import contextlib

    return jax.named_scope(name) if PHASE_SCOPES else contextlib.nullcontext()


def _phase_start(name):
    scope = _phase(name)
    scope.__enter__()
    return scope


def _phase_end(scope):
    scope.__exit__(None, None, None)


def _scoped(name):
    """Decorator form of _phase for pl.when bodies."""
    def wrap(function):
        @functools.wraps(function)
        def scoped(*args, **kwargs):
            with _phase(name):
                return function(*args, **kwargs)
        return scoped
    return wrap


# Largest supported decode batch; SCATTER_BATCHES lists the token-scatter batches.
MAX_BATCH = 128
# Hot-expert copies: [(layer weight refs, layer index, fp8 slot-pair views)] while a layer's MoE traces.
_HOT: list = []
_HOT_BASE: list = []
HOT_LOCAL_EXPERTS = 112  # EP8 local experts; copy slot j is local id 112 + j (family entries 112 + 2j, 113 + 2j)
# MXFP4 hot-expert replicas (KernelConfig.kd_replica_plan): the expert families carry S extra entries,
# replica slot j = local expert id 112 + j, run by the MXFP4 expert loop like a local expert.
REPLICAS = bool(_config().kd_replica_plan)
REPLICA_LANE = 3584  # replica table in latent_norm's zero padding lanes (see rank_loader._replica_table)
SCATTER_BATCHES = (8, 12, 16, 20, 24, 28, 32, 64, 96, 128)


def _bf16_rows_view(ref, rows, width):
    ref = ref.bitcast(jnp.bfloat16)
    return pool_alias.types.TransformedRef(
        ref.ref,
        (*ref.transforms, _PaddedBf16Rows((rows, width))),
    )


def _dot(x, w):
    batch_size = x.shape[0]
    if not 1 <= batch_size <= MAX_BATCH:
        raise ValueError(f"_dot supports 1 to {MAX_BATCH} rows")
    if batch_size == 1:
        # B1 broadcasts into one 16-row tile instead of zero-padding.
        tiled_x = jnp.broadcast_to(x, (16, x.shape[-1]))
    elif batch_size % 16:
        # Whole native 16-row MXU tiles.
        tiled_x = jnp.pad(x, ((0, -batch_size % 16), (0, 0)))
    else:
        tiled_x = x
    return jax.lax.dot_general(
        tiled_x,
        w,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:batch_size]


def _attention_scores(queries, keys):
    """Multiply partial query rows by a transposed key tile in FP32."""
    row_count = queries.shape[0]
    padded_queries = jnp.pad(queries, ((0, -row_count % 16), (0, 0)))
    return jax.lax.dot_general(
        padded_queries,
        keys,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:row_count]


def _gated_activation(gate, value):
    """Kimi K3 shared/routed expert activation, rounded to BF16."""
    return (
        4
        * jnp.tanh(gate / 4)
        * jax.nn.sigmoid(gate)
        * 25
        * jnp.tanh(value / 25)
    ).astype(jnp.bfloat16)
def _mxfp4_visit_cost(rows):
    """Modeled expert-loop time of one MXFP4 visit (us): per visit, per row, per extra 8-row pass (fit to measured
    per-group expert time, R^2 0.82)."""
    return jnp.where(rows > 0, 1.74 + 0.31 * rows + 0.67 * (jnp.ceil(rows / 8.0) - 1.0), 0.0)


def _replica_dispatch(winners, table, entries, local_expert_count, counts):
    """Which routes move to a replica. ``winners`` [B, k] global expert ids, ``table`` [1, 256] (this layer's
    replica lanes: ids, holder groups, local slots), ``counts`` [1, E] the step's per-expert route counts.
    Greedy over the entries in table order: each replicated expert's routes stay home, split by row parity (odd
    rows move) or all move, whichever minimises the modeled slowest group after the move. Every per-entry value
    stays a [1, 1] vector: holding the entries as scalars overflows the scalar unit for large tables.
    Returns to_copy [B, k] bool, holder group and local slot id [B, k] int32."""
    def entry(lane):
        return table[:, lane : lane + 1].astype(jnp.int32)

    def total(x):
        return jnp.sum(x, axis=(0, 1), keepdims=True)

    row = jax.lax.broadcasted_iota(jnp.int32, winners.shape, 0)
    group_lane = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    expert_lane = jax.lax.broadcasted_iota(jnp.int32, counts.shape, 1)
    replicated = jnp.zeros(counts.shape, jnp.bool_)
    for i in range(entries):
        replicated = replicated | (expert_lane == entry(i))
    cost = jnp.where(replicated, 0.0, _mxfp4_visit_cost(counts))
    expert_group = jnp.floor((expert_lane.astype(jnp.float32) + 0.5) / local_expert_count)
    load = jnp.zeros((1, 128), jnp.float32)
    for g in range(8):
        load = load + jnp.where(group_lane == g, total(jnp.where(expert_group == g, cost, 0.0)), 0.0)
    to_copy = jnp.zeros(winners.shape, jnp.bool_)
    holder = jnp.zeros(winners.shape, jnp.int32)
    slot = jnp.zeros(winners.shape, jnp.int32)
    odd = (row % 2) == 1
    for i in range(entries):
        expert = entry(i)
        routes = winners == expert
        rows = total(jnp.where(expert_lane == expert, counts, 0.0))
        home_group = expert // local_expert_count
        copy_group = entry(entries + i)
        best_metric = best_choice = best_add = None
        for choice, moved_rows in ((0, jnp.zeros_like(rows)), (1, jnp.floor(rows / 2.0)), (2, rows)):
            add = (
                jnp.where(group_lane == home_group, _mxfp4_visit_cost(rows - moved_rows), 0.0)
                + jnp.where(group_lane == copy_group, _mxfp4_visit_cost(moved_rows), 0.0)
            )
            metric = jnp.max(load + add, axis=1, keepdims=True) + 1e-3 * total(add)
            if best_metric is None:
                best_metric, best_choice, best_add = metric, jnp.full((1, 1), choice, jnp.int32), add
            else:
                better = metric < best_metric
                best_metric = jnp.where(better, metric, best_metric)
                best_choice = jnp.where(better, choice, best_choice)
                best_add = jnp.where(better, add, best_add)
        load = load + best_add
        moved = routes & ((best_choice == 2) | ((best_choice == 1) & odd))
        to_copy = to_copy | moved
        holder = jnp.where(moved, copy_group, holder)
        slot = jnp.where(moved, entry(2 * entries + i), slot)
    return to_copy, holder, slot


def _route_hidden_states(
    hidden_states,
    routing_projection_vmem,
    routing_bias_ref,
    routing_scores_vmem,
    send_semaphore,
    receive_semaphore,
    *,
    selected_expert_count,
    local_expert_count,
    expert_tensor_parallel,
    on_local_selection,
    after_selection,
    after_scores=None,
    after_score_send=None,
    live_rows=None,
    replica_table=None,
    replica_entries=0,
):
    """Project and select local expert routes through one shared B1/B2 path.

    ``routing_scores_vmem`` is [batch_size, 1024]. Paired ranks exchange their
    512-score halves directly in that padded workspace, then every top-k
    iteration advances all batch rows together. Selected local routes are
    returned as [batch_size, top_k] values for scalar SMEM materialization by
    the caller.
    """
    rank = jax.lax.axis_index("tp")
    batch_size = hidden_states.shape[0]
    expert_count = routing_bias_ref.shape[-1]
    if expert_count != 896 or hidden_states.shape[-1] > 7168:
        raise ValueError("Host-shared router requires 896 expert scores")
    if routing_scores_vmem.shape[-1] != 1024:
        raise ValueError("Routing score workspace must have padded width 1024")

    router_lane = rank % 2
    local_scores = _dot(hidden_states, routing_projection_vmem[0]).reshape(
        batch_size, 512
    )
    # Lane 1's last 128 router columns are padding; in the shared arena they
    # hold stale bytes, so zero their scores.
    column = jax.lax.broadcasted_iota(jnp.int32, local_scores.shape, 1)
    local_scores = jnp.where((router_lane == 0) | (column < 384), local_scores, 0.0)
    score_half = routing_scores_vmem.at[
        :, pl.ds(router_lane * 512, 512)
    ]
    score_half[...] = local_scores
    if after_scores is not None:
        after_scores()  # the router projection is consumed
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
    selected_experts = jnp.zeros(
        (batch_size, selected_expert_count), jnp.int32
    )
    selected_probabilities = jnp.zeros(
        (batch_size, selected_expert_count), jnp.float32
    )
    local_route_counts = jnp.zeros((batch_size,), jnp.int32)
    probability_totals = jnp.zeros((batch_size,), jnp.float32)
    expert_ids = jnp.arange(expert_count)[None, :]
    route_slots = jnp.arange(selected_expert_count)[None, :]
    if replica_table is not None:
        # Replicas: all selections first, then the replica dispatch, then this rank's local routes.
        winners_by_slot, probabilities_by_slot = [], []
        expert_counts = jnp.zeros((1, expert_count), jnp.float32)
        for selection in range(selected_expert_count):
            winners = jnp.argmax(routing_scores_vmem[:, :expert_count], axis=1).astype(jnp.int32)
            winner_mask = expert_ids == winners[:, None]
            probabilities = jnp.sum(jnp.where(winner_mask, uncorrected_probabilities, 0), axis=1)
            probability_totals += probabilities
            routing_scores_vmem[:, :expert_count] = jnp.where(
                winner_mask, -jnp.inf, routing_scores_vmem[:, :expert_count]
            )
            winners_by_slot.append(winners)
            probabilities_by_slot.append(probabilities)
            # (A 1-D boolean -> [B, 1] reshape does not lower; mask the ids of dead rows instead.)
            live_mask = (
                winner_mask if live_rows is None
                else expert_ids == jnp.where(live_rows, winners, -1)[:, None]
            )
            expert_counts = expert_counts + jnp.sum(live_mask.astype(jnp.float32), axis=0, keepdims=True)
            after_selection(selection)
        all_winners = jnp.stack(winners_by_slot, axis=1)
        all_probabilities = jnp.stack(probabilities_by_slot, axis=1)
        to_copy, holder, slot = _replica_dispatch(
            all_winners, replica_table, replica_entries, local_expert_count, expert_counts
        )
        group = rank // expert_tensor_parallel
        # (Boolean selects do not lower in Mosaic; combine with logic ops.)
        owned_all = (to_copy & (holder == group)) | (
            ~to_copy & (all_winners // local_expert_count == group)
        )
        local_all = jnp.where(to_copy, slot, all_winners % local_expert_count)
        for selection in range(selected_expert_count):
            counts_before_selection = local_route_counts
            owned_locally = owned_all[:, selection]
            if live_rows is not None:
                owned_locally &= live_rows
            destination_mask = owned_locally[:, None] & (route_slots == counts_before_selection[:, None])
            selected_experts = jnp.where(destination_mask, local_all[:, selection][:, None], selected_experts)
            selected_probabilities = jnp.where(
                destination_mask, all_probabilities[:, selection][:, None], selected_probabilities
            )
            local_route_counts = counts_before_selection + owned_locally.astype(jnp.int32)
        return (
            selected_experts,
            selected_probabilities,
            local_route_counts,
            probability_totals if live_rows is None else jnp.where(live_rows, probability_totals, 1.0),
        )
    # Only top-k rank is sequential. Every score reduction and ownership
    # operation carries the leading batch dimension with no per-row control.
    for selection in range(selected_expert_count):
        winners = jnp.argmax(
            routing_scores_vmem[:, :expert_count], axis=1
        ).astype(jnp.int32)
        winner_mask = expert_ids == winners[:, None]
        probabilities = jnp.sum(
            jnp.where(winner_mask, uncorrected_probabilities, 0), axis=1
        )
        probability_totals += probabilities
        routing_scores_vmem[:, :expert_count] = jnp.where(
            winner_mask, -jnp.inf, routing_scores_vmem[:, :expert_count]
        )

        counts_before_selection = local_route_counts
        owned_locally = (
            winners // local_expert_count
            == rank // expert_tensor_parallel
        )
        if live_rows is not None:
            owned_locally &= live_rows
        destination_mask = (
            owned_locally[:, None]
            & (route_slots == counts_before_selection[:, None])
        )
        selected_experts = jnp.where(
            destination_mask,
            (winners % local_expert_count)[:, None],
            selected_experts,
        )
        selected_probabilities = jnp.where(
            destination_mask,
            probabilities[:, None],
            selected_probabilities,
        )
        local_route_counts = (
            counts_before_selection + owned_locally.astype(jnp.int32)
        )
        on_local_selection(
            winners, owned_locally, counts_before_selection, selected_experts
        )
        after_selection(selection)

    return (
        selected_experts,
        selected_probabilities,
        local_route_counts,
        probability_totals if live_rows is None else jnp.where(live_rows, probability_totals, 1.0),
    )


def _route_owned_rows(
    owned_hidden_ref,
    batch_size,
    routing_projection_vmem,
    routing_bias_ref,
    routing_scores_vmem,
    selection_source_vmem,
    selections_vmem,
    send_semaphores,
    receive_semaphores,
    *,
    selected_expert_count,
    local_expert_count,
    expert_tensor_parallel,
    on_local_selection,
    after_selection,
    after_scores=None,
    live_rows=None,
):
    """``_route_hidden_states`` with the top-k sharded over the local rank pairs.

    Each rank pair scores and selects only the rows it owns
    (``owned_hidden_ref`` [batch / 4, hidden]); the pairs exchange their
    selections (global expert ids and probabilities, one FP32 row per token)
    and every rank derives its local routes from all of them. The per-row
    arithmetic and order are those of the replicated router, so the results
    are identical; only the top-k loop shrinks with the owned rows.
    Uses send/receive semaphores 1..4.
    """
    rank = jax.lax.axis_index("tp")
    rows = owned_hidden_ref.shape[0]
    pair = (rank % 8) // 2
    expert_count = routing_bias_ref.shape[-1]
    if rows * 4 != batch_size or expert_count != 896:
        raise ValueError("Sharded routing needs one row block per local rank pair")

    router_lane = rank % 2
    local_scores = _dot(owned_hidden_ref[...], routing_projection_vmem[0]).reshape(rows, 512)
    # Lane 1's last 128 router columns are padding; in the shared arena they
    # hold stale bytes, so zero their scores.
    column = jax.lax.broadcasted_iota(jnp.int32, local_scores.shape, 1)
    local_scores = jnp.where((router_lane == 0) | (column < 384), local_scores, 0.0)
    score_half = routing_scores_vmem.at[pl.ds(0, rows), pl.ds(router_lane * 512, 512)]
    score_half[...] = local_scores
    if after_scores is not None:
        after_scores()  # the router projection is consumed
    copy = tpu.make_async_remote_copy(
        score_half, score_half, send_semaphores.at[1], receive_semaphores.at[1],
        device_id=(rank ^ 1,), device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()
    copy.wait()

    scores = routing_scores_vmem.at[pl.ds(0, rows), pl.ds(0, expert_count)]
    uncorrected_probabilities = jax.nn.sigmoid(scores[...])
    scores[...] = uncorrected_probabilities + routing_bias_ref[...]
    expert_ids = jnp.arange(expert_count)[None, :]
    winners_by_slot, probabilities_by_slot = [], []
    probability_totals = jnp.zeros((rows,), jnp.float32)
    for selection in range(selected_expert_count):
        winners = jnp.argmax(scores[...], axis=1).astype(jnp.int32)
        winner_mask = expert_ids == winners[:, None]
        probabilities = jnp.sum(jnp.where(winner_mask, uncorrected_probabilities, 0), axis=1)
        probability_totals += probabilities
        scores[...] = jnp.where(winner_mask, -jnp.inf, scores[...])
        winners_by_slot.append(winners)
        probabilities_by_slot.append(probabilities)
        after_selection(selection)

    # One FP32 row per token: winners (exact small integers), probabilities,
    # the probability total.
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
        def gather_selections(owner=owner):
            destination = selections_vmem.at[pl.ds(owner * rows, rows)]
            destination[...] = selection_source_vmem[...]
            copies = []
            for offset in range(3, 0, -1):
                copy = tpu.make_async_remote_copy(
                    selection_source_vmem, destination,
                    send_semaphores.at[1 + offset], receive_semaphores.at[1 + offset],
                    device_id=(rank ^ (2 * offset),), device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()

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
        on_local_selection(winners, owned_locally, counts_before_selection, selected_experts)
    return (
        selected_experts,
        selected_probabilities,
        local_route_counts,
        gathered[:, 2 * k] if live_rows is None else jnp.where(live_rows, gathered[:, 2 * k], 1.0),
    )


def _start_initial_expert_weight_copies(
    selected_experts,
    local_route_counts,
    copy_expert,
    *,
    slot_count,
    already_started,
    group_pending_routes,
    skip_copies=False,
):
    """Prefetch the first routed experts in their execution order.

    This uses the same cross-row ordering as the expert stream, so it works for
    both shared and divergent routes without specializing on batch size.
    """
    selected_expert_count = selected_experts.shape[1]
    route_slots = jnp.arange(selected_expert_count)[None, :]
    processed = (selected_experts >= HOT_LOCAL_EXPERTS) if skip_copies else jnp.zeros(selected_experts.shape, jnp.bool_)

    def next_expert(completed):
        pending = (route_slots < local_route_counts[:, None]) & ~completed
        if group_pending_routes:
            minimum_expert = jnp.min(
                jnp.where(
                    pending,
                    selected_experts.astype(jnp.float32),
                    jnp.inf,
                )
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
                    rows_with_routes,
                    current_experts.astype(jnp.float32),
                    jnp.inf,
                )
            )
            has_expert = jnp.any(rows_with_routes)
            expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
            consumed = (
                rows_with_routes & (current_experts == expert)
            )[:, None] & current_slot_mask
        return expert, consumed, has_expert

    for slot in range(slot_count):
        expert, consumed, has_expert = next_expert(processed)

        @pl.when(has_expert & (slot >= already_started))
        def start_copy(expert=expert, slot=slot):
            _start_async_copies(copy_expert(expert, slot))

        processed |= consumed


def _hot_home_routes(latent_norm, selected_experts, local_route_counts, slot_count):
    """Home-only copy routing: a route to one of this group's hot experts takes its copy (local id
    112 + j) when the expert has at least hot_min_rows rows this step. The table is in the
    zero padding lanes of latent_norm (3584 + g S + j: global expert id)."""
    group = jax.lax.axis_index("tp") // 4
    table = latent_norm[0:1, 3584 : 3584 + 8 * slot_count]
    lanes = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    entry_lanes = jax.lax.broadcasted_iota(jnp.int32, table.shape, 1)
    count = selected_experts.shape[1]
    valid = jnp.arange(count)[None, :] < local_route_counts[:, None]
    rows = jnp.zeros((1, 128), jnp.float32)
    for slot in range(count):
        rows += jnp.sum(
            (valid[:, slot : slot + 1] & (selected_experts[:, slot : slot + 1] == lanes)).astype(jnp.float32),
            axis=0, keepdims=True,
        )
    copy_of = jnp.full((1, 128), -1, jnp.int32)
    for j in range(slot_count):
        # A [1, 1] vector, not a scalar: per-slot scalars kept live overflow the scalar unit for large plans.
        expert = jnp.sum(jnp.where(entry_lanes == group * slot_count + j, table, 0.0), axis=1, keepdims=True).astype(jnp.int32)
        local = expert - HOT_LOCAL_EXPERTS * group
        home = (expert >= 0) & (local >= 0) & (local < HOT_LOCAL_EXPERTS)
        copy_of = jnp.where(home & (lanes == local), HOT_LOCAL_EXPERTS + j, copy_of)
    copy_of = jnp.where(rows >= _config().hot_min_rows, copy_of, -1)
    columns = []
    for slot in range(count):
        target = jnp.sum(
            jnp.where(selected_experts[:, slot : slot + 1] == lanes, copy_of, 0) + 0, axis=1, keepdims=True
        ) + jnp.sum(jnp.where(selected_experts[:, slot : slot + 1] == lanes, 0, 0), axis=1, keepdims=True)
        hit = jnp.any((selected_experts[:, slot : slot + 1] == lanes) & (copy_of >= 0), axis=1, keepdims=True)
        columns.append(jnp.where(valid[:, slot : slot + 1] & hit, target, selected_experts[:, slot : slot + 1]))
    return jnp.concatenate(columns, axis=1)


def _run_hot_experts(hot, source, selected_experts, selected_probabilities, valid_routes, probability_totals,
                     expert_output, slots, *, expert_input_size, expert_width, expert_output_size):
    """Every routed copy slot (local id 112 + j) as one fp8 product per 32 of its rows; the slot's
    entries stream from HBM into MXFP4 slot pairs (0, 1) and (2, 3), double-buffered."""
    batch_size = selected_experts.shape[0]
    lanes = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
    lane_values = lanes.astype(jnp.float32)
    refs, layer = hot.refs, hot.layer - 1

    def lowest(remaining):
        expert = jnp.min(jnp.where(remaining, lane_values, 128.0)).astype(jnp.int32)
        return expert, remaining & (lanes != expert)

    def copies(expert, buffer):
        entry = HOT_LOCAL_EXPERTS + 2 * jnp.clip(expert - HOT_LOCAL_EXPERTS, 0, hot.slots - 1)
        semaphores = hot.semaphores
        gate_up, gate_up_scales, down, down_scales = slots
        return (
            tpu.make_async_copy(refs["expert_gu"].at[layer, pl.ds(entry, 2)],
                                gate_up.at[pl.ds(2 * buffer, 2)], semaphores.at[2 * buffer, 0]),
            tpu.make_async_copy(refs["expert_gus"].at[layer, entry],
                                gate_up_scales.at[2 * buffer], semaphores.at[2 * buffer, 1]),
            tpu.make_async_copy(refs["expert_down"].at[layer, pl.ds(entry, 2)],
                                down.at[pl.ds(2 * buffer, 2)], semaphores.at[2 * buffer, 2]),
            tpu.make_async_copy(refs["expert_ds"].at[layer, entry],
                                down_scales.at[2 * buffer], semaphores.at[2 * buffer, 3]),
        )

    def start(expert, buffer):
        @pl.when(expert < 128)
        def _():
            for copy in copies(expert, buffer):
                copy.start()

    index = jnp.arange(batch_size)
    chunk = 32

    def compute(expert, buffer):
        @pl.when(expert < 128)
        def _():
            view = hot.views[buffer]
            for copy in copies(expert, buffer):
                copy.wait()
            mask = valid_routes & (selected_experts == expert)
            active = jnp.any(mask, axis=1)
            weights = jnp.where(
                active, jnp.sum(jnp.where(mask, selected_probabilities, 0.0), axis=1) / probability_totals, 0.0
            )
            before = jnp.sum(
                jnp.where(index[None, :] < index[:, None], active[None, :].astype(jnp.float32), 0.0), axis=1
            )
            rows = jnp.sum(active.astype(jnp.int32))

            @pl.loop(0, (rows + chunk - 1) // chunk)
            def run_chunk(part):
                positions = (part * chunk + jnp.arange(chunk)).astype(jnp.float32)
                pick = active[None, :] & (before[None, :] == positions[:, None])
                staged = _dot(pick.astype(jnp.bfloat16), source[:, :expert_input_size])
                gate_up = jax.lax.dot_general(
                    staged.astype(jnp.bfloat16), view.gu[...], (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                ) * view.gs[...]
                gate_up = gate_up.astype(jnp.bfloat16).astype(jnp.float32)
                middle = _gated_activation(gate_up[:, :expert_width], gate_up[:, expert_width:])
                product = jax.lax.dot_general(
                    middle, view.down[...], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32,
                ) * view.ds[...]
                put = active[:, None] & (before[:, None] == positions[None, :])
                expert_output[:, :expert_output_size] += (
                    _dot(put.astype(jnp.bfloat16), product.astype(jnp.bfloat16)) * weights[:, None]
                )

    first, remaining = lowest(hot.lanes)
    second, remaining = lowest(remaining)
    start(first, 0)
    start(second, 1)

    def pair(iteration, carry):
        first, second, remaining = carry
        compute(first, 0)
        third, remaining = lowest(remaining != 0)
        start(third, 0)
        compute(second, 1)
        fourth, remaining = lowest(remaining)
        start(fourth, 1)
        return third, fourth, remaining.astype(jnp.int32)

    jax.lax.fori_loop(
        0, (jnp.sum(hot.lanes.astype(jnp.int32)) + 1) // 2, pair, (first, second, remaining.astype(jnp.int32))
    )


def _quantized_expert_dot(
    source,
    packed,
    scales,
    accumulator,
    contraction_size,
    output_size,
    *,
    block_major=False,
    dot_tile=None,
):
    """Multiply by one resident MXFP4 matrix or reblocked FP8 matrix.

    Reblocked FP8 stores one BF16 power-of-two scale for each contiguous
    contraction block. The scale-row count determines the static block width.
    This trades a larger HBM representation for removing MXFP4 unpack and its
    per-32-element scaled partial reduction.
    """
    dot_tile = (MXFP4_DOT_TILE or 0) if dot_tile is None else dot_tile
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)

    if packed.dtype == jnp.float8_e4m3fn:
        scale_block_count = scales.shape[0]
        if contraction_size % scale_block_count:
            raise ValueError("FP8 scale blocks must divide the contraction")
        scale_block_size = contraction_size // scale_block_count
        if scale_block_size % 128:
            raise ValueError("FP8 scale blocks must be MXU aligned")

        # Pallas traces and unrolls this fixed-size loop. Each FP8 tile is
        # consumed directly by MXU and scaled only after the dot product.
        for scale_block in range(scale_block_count):
            start = scale_block * scale_block_size
            activation = source[
                :, pl.ds(pl.multiple_of(start, 128), scale_block_size)
            ].astype(jnp.bfloat16)
            values = packed[start : start + scale_block_size, :]
            partial = jax.lax.dot_general(
                jnp.pad(
                    activation,
                    ((0, -activation.shape[0] % 16), (0, 0)),
                ),
                values,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )[: activation.shape[0]]
            accumulator[...] += (
                partial * scales[scale_block : scale_block + 1].astype(jnp.float32)
            )
        return

    if packed.dtype != jnp.uint32:
        raise ValueError("Expert weights must use packed MXFP4 or reblocked FP8")

    split_small_contraction = contraction_size == 768 and block_major
    block_size = 256 if split_small_contraction else min(MXFP4_BLOCK, contraction_size)

    static_tiles = MXFP4_STATIC_TILE > 0 and not split_small_contraction

    def tile(start, size):
        if static_tiles:
            values = tpu.bitcast(
                packed[start // 8 : (start + size) // 8, :], jnp.float4_e2m1fn
            ).astype(jnp.float8_e4m3fn)
        else:
            values = tpu.bitcast(
                packed[
                    pl.ds(pl.multiple_of(start // 8, 8), size // 8), :
                ],
                jnp.float4_e2m1fn,
            ).astype(jnp.float8_e4m3fn)
        if static_tiles:
            exponents = scales[...][
                start // 32 : (start + size) // 32, :
            ].astype(jnp.uint32)
        elif split_small_contraction:
            # The full 24-row view is aligned even though the second static
            # 512-wide tile begins halfway through a 32-row VMEM tile.
            exponents = scales[...][
                start // 32 : (start + size) // 32, :
            ].astype(jnp.uint32)
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
        if dot_tile and not split_small_contraction and size > dot_tile:
            # Split only the masked MXU products. Keep all scale-block partials
            # in block order and perform the same FP32 reduction as
            # the unsplit tile; reducing each subtile separately changes that
            # association and can perturb routing over many decode steps.
            partials = []
            for offset in range(0, size, dot_tile):
                width = min(dot_tile, size - offset)
                mask = (
                    jnp.arange(width // 32)[:, None]
                    == jnp.arange(width)[None, :] // 32
                )
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
            jnp.arange(size // 32)[:, None]
            == jnp.arange(size)[None, :] // 32
        )
        if split_small_contraction and activation.shape[0] == 1:
            # One row: keep the scale blocks in sublanes instead of padding a
            # [blocks, 1, N] partial. Add them sequentially, the order of the
            # major-axis sum below, so both shapes round identically.
            left = jnp.where(scale_block_mask, activation, jnp.bfloat16(0))
            scaled = jax.lax.dot_general(
                left,
                values,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ) * scale
            total = scaled[0:1]
            for block in range(1, size // 32):
                total = total + scaled[block : block + 1]
            accumulator[...] += total
        elif split_small_contraction:
            left = jnp.where(
                scale_block_mask[:, None, :],
                activation[None, :, :],
                jnp.bfloat16(0),
            )
            partial = jax.lax.dot_general(
                left,
                values,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            accumulator[...] += jnp.sum(
                partial * scale[:, None, :], axis=0
            )
        else:
            left = jnp.where(
                scale_block_mask[None], activation[:, None, :], jnp.bfloat16(0)
            )
            partial = jax.lax.dot_general(
                left,
                values,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            accumulator[...] += jnp.sum(partial * scale[None], axis=1)

    if static_tiles:
        for start in range(0, contraction_size, MXFP4_STATIC_TILE):
            tile(start, min(MXFP4_STATIC_TILE, contraction_size - start))
    elif split_small_contraction:
        for start in range(0, contraction_size, block_size):
            tile(start, block_size)
    else:
        @pl.loop(0, contraction_size // block_size)
        def full_tile(index):
            tile(index * block_size, block_size)

        if contraction_size % block_size:
            tile(
                contraction_size // block_size * block_size,
                contraction_size % block_size,
            )


def _paired_expert_gate_up(
    source,
    gate_up_vmem,
    gate_up_scale_vmem,
    accumulator,
    first_slot,
    second_slot,
    contraction_size,
    *,
    second_source=None,
    active_capacity=None,
):
    """Run two resident expert gate/up projections.

    With separate compact MXFP4 sources, only the matching row/weight
    products are consumed. Keep cross-expert entries zero while preserving
    each diagonal's scale reduction and tile accumulation order.
    """
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)

    if gate_up_vmem.dtype == jnp.float8_e4m3fn:
        scale_block_count = gate_up_scale_vmem.shape[1]
        if contraction_size % scale_block_count:
            raise ValueError("FP8 gate/up scales must divide the contraction")
        scale_block_size = contraction_size // scale_block_count
        if scale_block_size % 128:
            raise ValueError("FP8 gate/up scale blocks must be MXU aligned")
        for scale_block in range(scale_block_count):
            start = scale_block * scale_block_size
            values = jnp.concatenate(
                (
                    gate_up_vmem[
                        first_slot,
                        pl.ds(pl.multiple_of(start, 128), scale_block_size),
                        :,
                    ],
                    gate_up_vmem[
                        second_slot,
                        pl.ds(pl.multiple_of(start, 128), scale_block_size),
                        :,
                    ],
                ),
                axis=1,
            )
            scales = jnp.concatenate(
                (
                    gate_up_scale_vmem[
                        first_slot, scale_block : scale_block + 1, :
                    ],
                    gate_up_scale_vmem[
                        second_slot, scale_block : scale_block + 1, :
                    ],
                ),
                axis=1,
            ).astype(jnp.float32)
            activation = source[
                :, pl.ds(pl.multiple_of(start, 128), scale_block_size)
            ].astype(jnp.bfloat16)
            if second_source is not None:
                activation = jnp.concatenate(
                    (
                        activation,
                        second_source[
                            :,
                            pl.ds(
                                pl.multiple_of(start, 128), scale_block_size
                            ),
                        ].astype(jnp.bfloat16),
                    ),
                    axis=0,
                )
            partial = jax.lax.dot_general(
                jnp.pad(
                    activation,
                    ((0, -activation.shape[0] % 16), (0, 0)),
                ),
                values,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )[: activation.shape[0]]
            accumulator[...] += partial * scales
        return

    def tile(start, size):
        packed = jnp.concatenate(
            (
                gate_up_vmem[
                    first_slot, pl.ds(start // 8, size // 8), :
                ],
                gate_up_vmem[
                    second_slot, pl.ds(start // 8, size // 8), :
                ],
            ),
            axis=1,
        )
        values = tpu.bitcast(packed, jnp.float4_e2m1fn).astype(
            jnp.float8_e4m3fn
        )
        exponents = jnp.concatenate(
            (
                gate_up_scale_vmem[
                    first_slot, pl.ds(start // 32, size // 32), :
                ],
                gate_up_scale_vmem[
                    second_slot, pl.ds(start // 32, size // 32), :
                ],
            ),
            axis=1,
        ).astype(jnp.uint32)
        scale = jax.lax.bitcast_convert_type(exponents << 23, jnp.float32)
        activation = source[:, pl.ds(start, size)].astype(jnp.bfloat16)
        if second_source is not None:
            activation = jnp.concatenate(
                (
                    activation,
                    second_source[:, pl.ds(start, size)].astype(jnp.bfloat16),
                ),
                axis=0,
            )
        scale_block_mask = (
            jnp.arange(size // 32)[:, None]
            == jnp.arange(size)[None, :] // 32
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
                    left[half * rows:half * rows + active],
                    values[:, half * columns:(half + 1) * columns],
                    (((2,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                result = jnp.sum(
                    partial * scale[None, :, half * columns:(half + 1) * columns], axis=1
                )
                accumulator[
                    half * rows:(half + 1) * rows,
                    half * columns:(half + 1) * columns,
                ] += (result if active_capacity is None else
                      jnp.pad(result, ((0, rows - active), (0, 0))))
        else:
            partial = jax.lax.dot_general(
                left,
                values,
                (((2,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            accumulator[...] += jnp.sum(partial * scale[None], axis=1)

    # Narrow scaled partials (MXFP4_PAIR_TILE) shorten register live ranges.
    tile_size = MXFP4_PAIR_TILE
    for start in range(0, contraction_size, tile_size):
        tile(start, min(tile_size, contraction_size - start))


def _paired_order_expert_gate_up(source, packed, scales, accumulator, contraction_size, tile_size):
    """Project one expert in the paired gate/up's FP32 accumulation order.

    Pair membership changes when null routes disappear. Tail and overflow
    projections must retain the paired tile's scale reduction and accumulation
    order so an unrelated row cannot change a live expert's rounded result.
    """
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)
    for start in range(0, contraction_size, tile_size):
        size = min(tile_size, contraction_size - start)
        values = tpu.bitcast(packed[pl.ds(start // 8, size // 8), :],
                             jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)
        exponents = scales[pl.ds(start // 32, size // 32), :].astype(jnp.uint32)
        scale = jax.lax.bitcast_convert_type(exponents << 23, jnp.float32)
        activation = source[:, pl.ds(start, size)].astype(jnp.bfloat16)
        mask = jnp.arange(size // 32)[:, None] == jnp.arange(size)[None, :] // 32
        left = jnp.where(mask[None], activation[:, None, :], jnp.bfloat16(0))
        partial = jax.lax.dot_general(left, values, (((2,), (0,)), ((), ())),
                                      preferred_element_type=jnp.float32)
        accumulator[...] += jnp.sum(partial * scale[None], axis=1)


def _dot_scoped(function, name_of):
    """Profiling: trace each routed-expert product as its own e_dot_* phase."""
    @functools.wraps(function)
    def scoped(*args, **kwargs):
        with _phase(name_of(*args, **kwargs)):
            return function(*args, **kwargs)
    return scoped


if PHASE_SCOPES:
    _quantized_expert_dot = _dot_scoped(
        _quantized_expert_dot,
        lambda source, packed, *args, **kwargs: (
            "e_dot_fp8" if packed.dtype == jnp.float8_e4m3fn else "e_dot_mxfp4"
        ),
    )
    _paired_expert_gate_up = _dot_scoped(_paired_expert_gate_up, lambda *args, **kwargs: "e_dot_pair")


def _run_routed_expert_stream(
    source,
    selected_experts,
    selected_probabilities,
    local_route_counts,
    probability_totals,
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
    *,
    expert_input_size,
    expert_width,
    expert_output_size,
    group_pending_routes,
    progress=None,
    hot=None,
):
    """Execute the next route shared by any ready batch rows.

    Rows retain their own routing order. When several rows have the same next
    expert, its weights are loaded once and its projections carry the batch
    dimension. The loop is bounded by the distinct experts (grouped routes)
    or the route count, never by batch size.
    """
    batch_size, selected_expert_count = selected_experts.shape
    # Narrow masked products avoid zero MACs and register spills while keeping
    # the unsplit scale reduction tree. KernelConfig.mxfp4_dot_tile overrides
    # the per-batch default (0: unsplit dots).
    dot_tile = (
        MXFP4_DOT_TILE if MXFP4_DOT_TILE is not None
        else 1024 if batch_size == 96
        else 512 if batch_size in (1, 2, 4, 8, 16, 32, 64, 128) else 0
    )
    # Aligned block staging avoids a separate source-slice branch for every
    # batch row. On by default at B16-B128; KernelConfig.singleton_block
    # and stage_blocks override it.
    singleton_block = (
        SINGLETON_BLOCK if SINGLETON_BLOCK is not None
        else batch_size in (16, 32, 64, 96, 128)
    )
    stage_blocks = (
        STAGE_BLOCKS if STAGE_BLOCKS is not None
        else batch_size in (16, 32, 64, 96, 128)
    )
    def expert_dot(source, packed, scales, accumulator, contraction_size, output_size, **options):
        if (batch_size > 1 and contraction_size == expert_input_size
                and packed.dtype == jnp.uint32):
            # A live expert may switch between a pair and the unpaired tail
            # as other rows enter or leave. Match the paired gate's scale
            # reduction tree, including its final partial tile.
            return _paired_order_expert_gate_up(
                source, packed, scales, accumulator, contraction_size, MXFP4_PAIR_TILE
            )
        return _quantized_expert_dot(
            source, packed, scales, accumulator, contraction_size, output_size,
            dot_tile=dot_tile, **options,
        )
    if PHASE_SCOPES:
        # Profiling: time spent waiting for expert weights (e_wait).
        unscoped_copy_expert = copy_expert

        def copy_expert(expert, slot, **options):
            copies = unscoped_copy_expert(expert, slot, **options)
            return SimpleNamespace(**{
                name: None if copy is None else _ScopedWait(copy)
                for name, copy in vars(copies).items()
            })
    route_slots = jnp.arange(selected_expert_count)[None, :]
    processed_routes_vmem[...] = jnp.zeros(
        processed_routes_vmem.shape, jnp.int32
    )
    expert_output[...] = jnp.zeros(expert_output.shape, jnp.float32)

    def next_expert(processed):
        pending = (
            route_slots < local_route_counts[:, None]
        ) & ~processed
        if group_pending_routes:
            minimum_expert = jnp.min(
                jnp.where(
                    pending,
                    selected_experts.astype(jnp.float32),
                    jnp.inf,
                )
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
            jnp.where(
                rows_with_routes,
                current_experts.astype(jnp.float32),
                jnp.inf,
            )
        )
        has_expert = jnp.any(rows_with_routes)
        expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
        active_rows = rows_with_routes & (current_experts == expert)
        consumed = active_rows[:, None] & current_slot_mask
        return expert, active_rows, current_slot_mask, consumed, has_expert

    def finish_expert(active_rows, current_slot_mask, copies, column, slot, output_weights=None):
        gate_up = gate_up_accumulator[
            :, (pl.ds(pl.multiple_of(column, 128), 2 * expert_width)
                if share_full_finish else slice(column, column + 2 * expert_width))
        ].astype(jnp.bfloat16).astype(jnp.float32)
        middle = _gated_activation(
            gate_up[:, :expert_width], gate_up[:, expert_width:]
        )
        gate_up_accumulator[:, :expert_width] = middle.astype(jnp.float32)
        copies.down.wait()
        copies.down_scales.wait()
        expert_dot(
            gate_up_accumulator.at[:, :expert_width],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator,
            expert_width,
            expert_output_size,
            # Keep the down projection's scale reductions independent of
            # how many rows happen to route to this expert.
            block_major=batch_size >= 8,
        )
        if output_weights is None:
            route_probabilities = jnp.sum(
                jnp.where(
                    current_slot_mask,
                    selected_probabilities,
                    jnp.float32(0),
                ),
                axis=1,
            ) / probability_totals
            output_weights = jnp.where(active_rows, route_probabilities, jnp.float32(0))
        expert_output[:, :expert_output_size] += (
            down_accumulator[...]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
            * output_weights[:, None]
        )

    def finish_singleton_expert(
        active_rows,
        current_slot_mask,
        copies,
        compact_row,
        column,
        slot,
    ):
        """Finish one expert whose route is active for exactly one batch row."""
        gate_up = gate_up_accumulator[
            compact_row : compact_row + 1,
            column : column + 2 * expert_width,
        ].astype(jnp.bfloat16).astype(jnp.float32)
        middle = _gated_activation(
            gate_up[:, :expert_width], gate_up[:, expert_width:]
        )
        gate_up_accumulator[
            compact_row : compact_row + 1, :expert_width
        ] = middle.astype(jnp.float32)
        copies.down.wait()
        copies.down_scales.wait()
        expert_dot(
            gate_up_accumulator.at[
                compact_row : compact_row + 1, :expert_width
            ],
            expert_down_vmem.at[slot],
            expert_down_scale_vmem.at[slot],
            down_accumulator.at[:1],
            expert_width,
            expert_output_size,
            # Use the 256-wide block-major accumulation of the compact/full
            # paths so a singleton row rounds the same as in a group.
            block_major=True,
        )
        route_probabilities = jnp.sum(
            jnp.where(
                current_slot_mask,
                selected_probabilities,
                jnp.float32(0),
            ),
            axis=1,
        ) / probability_totals
        singleton_output = (
            down_accumulator[:1]
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
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

    def stage_singleton_source(active_rows, destination):
        if singleton_block and batch_size % 16 == 0:
            active_index = jnp.sum(jnp.where(active_rows, jnp.arange(batch_size), 0))
            block = source[
                pl.ds(pl.multiple_of(active_index // 16 * 16, 16), 16), :expert_input_size
            ].astype(jnp.float32)
            within = jax.lax.broadcasted_iota(jnp.int32, block.shape, 0) == active_index % 16
            destination[...] = jnp.sum(
                jnp.where(within, block, 0.0), axis=0, keepdims=True
            ).astype(destination.dtype)
            return
        active_index = jnp.sum(jnp.where(active_rows, jnp.arange(batch_size), 0))
        for row in range(batch_size):
            @pl.when(active_index == row)
            def copy_indexed_row(row=row):
                destination[...] = source[
                    row : row + 1, :expert_input_size
                ].astype(destination.dtype)

    def row_bitmask_words(active_rows):
        """The active rows as scalar bit words (32 rows per word)."""
        index = jnp.arange(batch_size)
        bits = active_rows.astype(jnp.int32) << (index % 32)
        return [
            jnp.sum(jnp.where(index // 32 == word, bits, 0))
            for word in range(-(-batch_size // 32))
        ]

    reuse_row_plan = stage_blocks and batch_size in (32, 64, 96, 128)

    def compact_row_positions(active_rows):
        # Counts and row indices are < 128, hence exact in FP32. This avoids
        # the split reductions used for unrestricted int32.
        count_dtype = jnp.float32 if reuse_row_plan else jnp.int32
        index = jnp.arange(batch_size)
        return jnp.sum(
            jnp.where(index[None, :] < index[:, None],
                      active_rows[None, :].astype(count_dtype), 0),
            axis=1,
        ).astype(jnp.int32)

    def compact_row_plan(active_rows, offset=0, before=None):
        if before is None:
            before = compact_row_positions(active_rows)
        # One ordered index list serves both input staging and output
        # selection. Invalid trailing entries must not select source row 0.
        selected = active_rows[None, :] & (
            before[None, :] == offset + jnp.arange(compact_capacity)[:, None]
        )
        indices = jnp.sum(
            jnp.where(selected, jnp.arange(batch_size).astype(jnp.float32)[None, :], 0.0), axis=1
        ).astype(jnp.int32)
        return indices, jnp.any(selected, axis=1)

    def stage_planned_rows(row_plan, destination):
        indices, valid = row_plan
        if batch_size in (64, 96, 128):
            # A one-hot product selects the planned rows (exact: one BF16
            # term per entry, zero rows past the plan) without a scalar
            # read and a gather per row.
            one_hot = (
                (indices[:, None] == jnp.arange(batch_size)[None, :]) & valid[:, None]
            ).astype(jnp.bfloat16)
            destination[...] = _dot(one_hot, source[:, :expert_input_size])
            return
        destination[...] = jnp.zeros(destination.shape, destination.dtype)

        if batch_size == 32:
            @pl.loop(0, batch_size // 16)
            def stage_block(block):
                in_block = valid & (indices // 16 == block)

                @pl.when(jnp.any(in_block))
                def copy_block():
                    rows = source[
                        pl.ds(pl.multiple_of(block * 16, 16), 16), :expert_input_size
                    ]
                    gather_indices = jnp.broadcast_to(
                        (indices % 16).astype(jnp.int16)[:, None],
                        (compact_capacity, expert_input_size),
                    )
                    selected_rows = jnp.take_along_axis(
                        rows, gather_indices, axis=0, mode="promise_in_bounds",
                    ).astype(jnp.float32)
                    destination[...] = jnp.where(
                        in_block[:, None], selected_rows, destination[...]
                    )
            return

        # Each gather stays inside one native 16-row BF16 tile. Matching
        # int16 indices avoid unsupported mixed-width Mosaic gathers.
        @pl.loop(0, jnp.sum(valid.astype(jnp.float32)).astype(jnp.int32))
        def copy_row(position):
            index = jnp.sum(
                jnp.where(jnp.arange(compact_capacity) == position,
                          indices.astype(jnp.float32), 0.0)
            ).astype(jnp.int32)
            rows = source[
                pl.ds(pl.multiple_of(index // 16 * 16, 16), 16), :expert_input_size
            ]
            gather_indices = jnp.full((1, expert_input_size), index % 16, jnp.int16)
            selected_row = jnp.take_along_axis(
                rows, gather_indices, axis=0, mode="promise_in_bounds",
            ).astype(jnp.float32)
            destination[pl.ds(position, 1), :] = selected_row

    def stage_compact_rows(active_rows, destination, offset=0, row_plan=None):
        """Copy active rows offset .. offset + C - 1 (in row order) into
        ``destination`` [C, K] FP32 (the rest zero)."""
        if reuse_row_plan:
            if row_plan is None:
                row_plan = compact_row_plan(active_rows, offset)
            stage_planned_rows(row_plan, destination)
            return
        destination[...] = jnp.zeros(destination.shape, destination.dtype)
        if stage_blocks and batch_size == 16:
            # B16 stages rows with a scalar scan over the row bitmask.
            word, = row_bitmask_words(active_rows)

            def stage_block(block, position):
                first = pl.multiple_of(block * 16, 16)
                bits = (word >> (block * 16)) & 65535
                def stage_nonempty(position):
                    for row in range(16):
                        bit = (bits >> row) & 1
                        take = (bit != 0) & (position >= offset) & (position < offset + compact_capacity)

                        @pl.when(take)
                        def copy_block_row(row=row, position=position):
                            rows = source[pl.ds(first, 16), :expert_input_size]
                            destination[pl.ds(position - offset, 1), :] = (
                                rows[row : row + 1].astype(jnp.float32)
                            )

                        position = position + bit
                    return position

                return stage_nonempty(position)

            jax.lax.fori_loop(0, batch_size // 16, stage_block, jnp.int32(0))
            return
        if stage_blocks and batch_size % 16 == 0:
            index = jnp.arange(batch_size)
            active = active_rows.astype(jnp.int32)

            @pl.loop(0, batch_size // 16)
            def stage_block(block):
                first = pl.multiple_of(block * 16, 16)
                bits = jnp.sum(jnp.where(index // 16 == block, active << (index % 16), 0))
                position = jnp.sum(jnp.where(index < block * 16, active, 0))
                for row in range(16):
                    bit = (bits >> row) & 1
                    take = (bit != 0) & (position >= offset) & (position < offset + compact_capacity)

                    @pl.when(take)
                    def copy_block_row(row=row, position=position):
                        rows = source[pl.ds(first, 16), :expert_input_size]
                        destination[pl.ds(position - offset, 1), :] = (
                            rows[row : row + 1].astype(jnp.float32)
                        )

                    position = position + bit
            return
        words = row_bitmask_words(active_rows)
        position = jnp.int32(0)
        for row in range(batch_size):
            bit = (words[row // 32] >> (row % 32)) & 1
            if isinstance(offset, int) and offset == 0:  # static first pass
                take = bit != 0
            else:
                take = (bit != 0) & (position >= offset) & (position < offset + compact_capacity)

            @pl.when(take)
            def copy_active_row(row=row, position=position):
                destination[pl.ds(position - offset, 1), :] = source[
                    row : row + 1, :expert_input_size
                ].astype(jnp.float32)

            position = position + bit

    def compact_route_data(active_rows, current_slot_mask, offset=0, row_plan=None):
        route_probabilities = jnp.sum(
            jnp.where(current_slot_mask, selected_probabilities, jnp.float32(0)),
            axis=1,
        ) / probability_totals
        if reuse_row_plan:
            if row_plan is None:
                row_plan = compact_row_plan(active_rows, offset)
            indices, valid = row_plan
            selected = (jnp.arange(batch_size)[:, None] == indices[None, :]) & valid[None, :]
            selector = selected.astype(jnp.bfloat16)
            in_pass = jnp.any(selected, axis=1)
        else:
            before = compact_row_positions(active_rows)
            selector = (
                (before[:, None] - offset == jnp.arange(compact_capacity)[None, :])
                & active_rows[:, None]
            ).astype(jnp.bfloat16)
            in_pass = active_rows & (before >= offset) & (before < offset + compact_capacity)
        return selector, jnp.where(in_pass, route_probabilities, jnp.float32(0))

    def finish_compact_expert(
        active_rows, current_slot_mask, copies, first_row, column, slot,
        offset=0, wait_down=True, route_data=None,
    ):
        """Finish one expert whose active rows offset .. offset + C - 1 sit
        compacted in accumulator rows [first_row, first_row + C)."""
        rows = (pl.ds(pl.multiple_of(first_row, 8), compact_capacity) if share_compact_finish
                else slice(first_row, first_row + compact_capacity))
        gate_up = gate_up_accumulator[
            rows, (pl.ds(pl.multiple_of(column, 128), 2 * expert_width)
                   if share_compact_finish else slice(column, column + 2 * expert_width))
        ].astype(jnp.bfloat16).astype(jnp.float32)
        middle = _gated_activation(
            gate_up[:, :expert_width], gate_up[:, expert_width:]
        )
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
            expert_output_size,
            block_major=True,
        )
        if route_data is None:
            route_data = compact_route_data(active_rows, current_slot_mask, offset)
        selector, output_weights = route_data
        selected = _dot(
            selector,
            down_accumulator[:compact_capacity].astype(jnp.bfloat16),
        )
        expert_output[:, :expert_output_size] += selected * output_weights[:, None]


    compact_overflow = _use_compact_overflow(
        batch_size, expert_gate_up_vmem.dtype, group_pending_routes=group_pending_routes
    )
    # At B16, two compact gate tiles fill 16 rows but the down tile has 8.
    # Only compact overflow tests the down tile (not KernelConfig.compact_only
    # or b16_compact_experts=False).
    compact_only = gate_up_accumulator.shape[0] < batch_size or (
        compact_overflow and down_accumulator.shape[0] < batch_size
    )
    share_compact_finish = (
        SHARED_EXPERT_FINISHERS and batch_size in (16, 32, 64, 96, 128) and compact_capacity == 8
        and (not compact_only or compact_overflow)
        and expert_gate_up_vmem.dtype == jnp.uint32
    )
    share_full_finish = share_compact_finish and not compact_only

    def run_expert_in_passes(active_rows, current_slot_mask, copies, row_count, slot, paired_gate=False):
        """One expert with more than C rows, C rows per pass."""
        copies.gate_up.wait()
        copies.gate_up_scales.wait()
        copies.down.wait()
        copies.down_scales.wait()

        before = compact_row_positions(active_rows) if reuse_row_plan else None
        @pl.loop(0, (row_count + compact_capacity - 1) // compact_capacity)
        def run_pass(index):
            offset = index * compact_capacity
            row_plan = compact_row_plan(active_rows, offset, before) if reuse_row_plan else None
            stage_compact_rows(active_rows, first_compact_source_vmem, offset, row_plan)
            if paired_gate:
                _paired_order_expert_gate_up(
                    first_compact_source_vmem,
                    expert_gate_up_vmem.at[slot],
                    expert_gate_up_scale_vmem.at[slot],
                    gate_up_accumulator.at[:compact_capacity, : 2 * expert_width],
                    expert_input_size, MXFP4_PAIR_TILE,
                )
            else:
                expert_dot(
                    first_compact_source_vmem,
                    expert_gate_up_vmem.at[slot],
                    expert_gate_up_scale_vmem.at[slot],
                    gate_up_accumulator.at[:compact_capacity, : 2 * expert_width],
                    expert_input_size,
                    2 * expert_width,
                )
            # Every pass owns its rows; finish it before reusing the compact
            # accumulator. Weight completions are consumed once, above.
            finish_compact_expert(
                active_rows, current_slot_mask, copies, 0, 0, slot,
                offset=offset, wait_down=False,
                route_data=compact_route_data(active_rows, current_slot_mask, offset, row_plan),
            )

    valid_routes = route_slots < local_route_counts[:, None]
    routes_are_shared = (not group_pending_routes) & jnp.all(
        local_route_counts == local_route_counts[0]
    ) & jnp.all(
        ~valid_routes
        | (selected_experts == selected_experts[0:1])
    )

    @pl.when(routes_are_shared & (not compact_only))
    def run_shared_routes():
        if compact_only:
            return
        def shared_expert_at(index):
            return jnp.sum(
                jnp.where(
                    route_slots[0] == index,
                    selected_experts[0],
                    jnp.int32(0),
                )
            ).astype(jnp.int32)

        def shared_probabilities_at(index):
            return jnp.sum(
                jnp.where(
                    route_slots == index,
                    selected_probabilities,
                    jnp.float32(0),
                ),
                axis=1,
            )

        def wait_for_gate_up(index, slot):
            copies = copy_expert(
                shared_expert_at(index), slot, down=False
            )
            copies.gate_up.wait()
            copies.gate_up_scales.wait()

        def wait_for_down(index, slot):
            copies = copy_expert(
                shared_expert_at(index), slot, gate_up=False
            )
            copies.down.wait()
            copies.down_scales.wait()

        def finish_shared_expert(index, slot, column):
            gate_up = gate_up_accumulator[
                :, column : column + 2 * expert_width
            ].astype(jnp.bfloat16).astype(jnp.float32)
            middle = _gated_activation(
                gate_up[:, :expert_width], gate_up[:, expert_width:]
            )
            gate_up_accumulator[:, :expert_width] = middle.astype(
                jnp.float32
            )
            wait_for_down(index, slot)
            expert_dot(
                gate_up_accumulator.at[:, :expert_width],
                expert_down_vmem.at[slot],
                expert_down_scale_vmem.at[slot],
                down_accumulator,
                expert_width,
                expert_output_size,
                block_major=batch_size >= 8,
            )
            route_weights = (
                shared_probabilities_at(index) / probability_totals
            )[:, None]
            expert_output[:, :expert_output_size] += (
                down_accumulator[...]
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
                * route_weights
            )

            @pl.when(index + 4 < local_route_counts[0])
            def refill_expert():
                _start_async_copies(
                    copy_expert(shared_expert_at(index + 4), slot)
                )

        @pl.loop(0, local_route_counts[0] // 2)
        def run_shared_pair(pair):
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
            finish_shared_expert(
                first + 1, second_slot, 2 * expert_width
            )

        @pl.when(local_route_counts[0] % 2 == 1)
        def run_shared_tail():
            index = local_route_counts[0] - 1
            slot = index % 4
            wait_for_gate_up(index, slot)
            expert_dot(
                source,
                expert_gate_up_vmem.at[slot],
                expert_gate_up_scale_vmem.at[slot],
                gate_up_accumulator.at[:, : 2 * expert_width],
                expert_input_size,
                2 * expert_width,
            )
            finish_shared_expert(index, slot, 0)

    if group_pending_routes:
        # Each iteration serves every pending route of its two experts, so the
        # loop needs one iteration per two distinct experts, not per two
        # routes (local expert ids are < 128).
        used = jnp.zeros((1, 128), jnp.bool_)
        for route_slot in range(selected_expert_count):
            used |= jnp.any(
                valid_routes[:, route_slot : route_slot + 1]
                & (selected_experts[:, route_slot : route_slot + 1] == jnp.arange(128)[None, :]),
                axis=0,
                keepdims=True,
            )
        if hot is not None:
            copy_lanes = used & (jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1) >= HOT_LOCAL_EXPERTS)
            hot = SimpleNamespace(**vars(hot), lanes=copy_lanes)
            used = used & ~copy_lanes
        pair_count = (jnp.sum(used.astype(jnp.int32)) + 1) // 2
    else:
        # Rows advance through their own route order, so one expert can be
        # visited more than once: bound the loop by the route count.
        pair_count = (jnp.sum(local_route_counts) + 1) // 2

    @pl.when(~routes_are_shared)
    def run_divergent_routes():
        # Reuse the choices already made for the next four-slot prefetch.
        # Other batch/routing combinations use the selector loop.
        carry_routes = (
            (batch_size in (2, 4) and not group_pending_routes)
            or (batch_size in (8, 16, 32, 64, 96, 128) and group_pending_routes)
        )
        def pack(choice):
            return tuple(value.astype(jnp.int32) for value in choice)

        def unpack(choice):
            return (choice[0], *(value != 0 for value in choice[1:]))

        # Grouped routes visit the used experts in ascending id order, each
        # with all its routes, so the next expert is the lowest one not yet
        # taken from ``used`` (one 128-lane row) rather than a reduction over
        # every pending route.
        plan_routes = carry_routes and group_pending_routes
        expert_lanes = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)

        def plan_lowest(remaining):
            # Expert ids are < 128, exact in FP32 (as next_expert's minimum);
            # 128: the plan is exhausted.
            lowest = jnp.min(jnp.where(remaining, expert_lanes.astype(jnp.float32), 128.0)).astype(jnp.int32)
            return lowest, remaining & (expert_lanes != lowest)

        def plan_choice(lowest):
            has_expert = lowest < 128
            expert = jnp.where(has_expert, lowest, 0)
            selected_slot_mask = valid_routes & (selected_experts == lowest)
            active_rows = jnp.any(selected_slot_mask, axis=1)
            return (expert, active_rows, selected_slot_mask, selected_slot_mask, has_expert)

        if plan_routes:
            # The carry holds only the next two expert ids and the plan; their
            # route masks are rebuilt when they run (no [B, 16] carries).
            initial_first, initial_remaining = plan_lowest(used)
            initial_second, initial_remaining = plan_lowest(initial_remaining)
        elif carry_routes:
            initial_processed = jnp.zeros(selected_experts.shape, jnp.bool_)
            initial_first = next_expert(initial_processed)
            initial_second = next_expert(initial_processed | initial_first[3])

        def run_next_expert_pair(iteration, carry=None):
          if progress is not None:
            progress(iteration)
          with _phase("e_iteration"):
            select_scope = _phase_start("e_select")
            processed = None if plan_routes else processed_routes_vmem[:, :selected_expert_count] != 0
            (
                first_expert,
                first_rows,
                first_slot_mask,
                first_consumed,
                has_first,
            ) = plan_choice(carry[0]) if plan_routes else unpack(carry[0]) if carry_routes else next_expert(processed)
            (
                second_expert,
                second_rows,
                second_slot_mask,
                second_consumed,
                has_second,
            ) = plan_choice(carry[1]) if plan_routes else unpack(carry[1]) if carry_routes else next_expert(processed | first_consumed)
            if plan_routes:
                third_lowest, remaining = plan_lowest(carry[2] != 0)
                fourth_lowest, remaining = plan_lowest(remaining)
                third_choice = (jnp.where(third_lowest < 128, third_lowest, 0), None, None, None, third_lowest < 128)
                fourth_choice = (jnp.where(fourth_lowest < 128, fourth_lowest, 0), None, None, None, fourth_lowest < 128)
            else:
                processed_after_pair = processed | first_consumed | second_consumed
                third_choice = next_expert(processed_after_pair)
                fourth_choice = next_expert(processed_after_pair | third_choice[3])
            (
                third_expert,
                third_rows,
                third_slot_mask,
                third_consumed,
                has_third,
            ) = third_choice
            (
                fourth_expert,
                fourth_rows,
                fourth_slot_mask,
                fourth_consumed,
                has_fourth,
            ) = fourth_choice

            current_base_slot = (iteration % 2) * 2
            next_base_slot = ((iteration + 1) % 2) * 2
            first_copies = copy_expert(first_expert, current_base_slot)
            second_copies = copy_expert(second_expert, current_base_slot + 1)
            third_copies = copy_expert(third_expert, next_base_slot)
            fourth_copies = copy_expert(fourth_expert, next_base_slot + 1)

            should_prefetch_third = has_third & (iteration != 0)
            should_prefetch_fourth = has_fourth & (iteration != 0)

            @pl.when(should_prefetch_third)
            def prefetch_third():
                _start_async_copies(third_copies)

            @pl.when(should_prefetch_fourth)
            def prefetch_fourth():
                _start_async_copies(fourth_copies)

            _phase_end(select_scope)

            def execute_full_pair():
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
                        first_rows,
                        first_slot_mask,
                        first_copies,
                        0,
                        current_base_slot,
                    )
                    finish_expert(
                        second_rows,
                        second_slot_mask,
                        second_copies,
                        2 * expert_width,
                        current_base_slot + 1,
                    )

            def execute_full_tail():
                first_copies.gate_up.wait()
                first_copies.gate_up_scales.wait()
                expert_dot(
                    source,
                    expert_gate_up_vmem.at[current_base_slot],
                    expert_gate_up_scale_vmem.at[current_base_slot],
                    gate_up_accumulator.at[:, : 2 * expert_width],
                    expert_input_size,
                    2 * expert_width,
                )
                if not share_full_finish:
                    finish_expert(
                        first_rows,
                        first_slot_mask,
                        first_copies,
                        0,
                        current_base_slot,
                    )

            # From B8, staging a singleton route's one active row is cheaper
            # than a full padded MXU tile. At B2/B4 the selection overhead
            # offsets the saved padded work.
            if batch_size >= 8:
                first_row_count = jnp.sum(first_rows.astype(jnp.int32))
                second_row_count = jnp.sum(second_rows.astype(jnp.int32))
                compact_pair = (
                    has_first
                    & has_second
                    & (first_row_count == 1)
                    & (second_row_count == 1)
                )

                first_plan = compact_row_plan(first_rows) if reuse_row_plan else None
                second_plan = compact_row_plan(second_rows) if reuse_row_plan else None

                @pl.when(compact_pair)
                @_scoped("e_single_pair")
                def execute_compact_pair():
                    first_copies.gate_up.wait()
                    first_copies.gate_up_scales.wait()
                    second_copies.gate_up.wait()
                    second_copies.gate_up_scales.wait()
                    stage_singleton_source(
                        first_rows, first_single_source
                    )
                    stage_singleton_source(
                        second_rows, second_single_source
                    )
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
                        first_rows,
                        first_slot_mask,
                        first_copies,
                        0,
                        0,
                        current_base_slot,
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
                        has_first & has_second & ~compact_pair
                        & (first_row_count <= compact_capacity)
                        & (second_row_count <= compact_capacity)
                    )

                    @pl.when(multi_pair)
                    @_scoped("e_multi_pair")
                    def execute_compact_multi_pair():
                        first_copies.gate_up.wait()
                        first_copies.gate_up_scales.wait()
                        second_copies.gate_up.wait()
                        second_copies.gate_up_scales.wait()
                        stage_compact_rows(first_rows, first_compact_source_vmem, row_plan=first_plan)
                        stage_compact_rows(second_rows, second_compact_source_vmem, row_plan=second_plan)
                        def project_pair(capacity=None):
                            _paired_expert_gate_up(
                                first_compact_source_vmem,
                                expert_gate_up_vmem,
                                expert_gate_up_scale_vmem,
                                gate_up_accumulator.at[:2 * compact_capacity],
                                current_base_slot,
                                current_base_slot + 1,
                                expert_input_size,
                                second_source=second_compact_source_vmem,
                                active_capacity=capacity,
                            )

                        if share_compact_finish:
                            # Keep native eight-row staging and finishers. Narrow only the register
                            # operands of gate/up, preserving its scale and FP32 reduction order.
                            largest = jnp.maximum(first_row_count, second_row_count)
                            previous = 1
                            for capacity in (2, 4, 8):
                                @pl.when((largest > previous) & (largest <= capacity))
                                @_scoped(f"e_capacity_{capacity}")
                                def execute_capacity(capacity=capacity):
                                    project_pair(capacity)
                                previous = capacity
                        else:
                            project_pair()
                            finish_compact_expert(
                                first_rows, first_slot_mask, first_copies,
                                0, 0, current_base_slot,
                            )
                            finish_compact_expert(
                                second_rows, second_slot_mask, second_copies,
                                compact_capacity, 2 * expert_width, current_base_slot + 1,
                            )
                else:
                    multi_pair = jnp.bool_(False)

                @pl.when(has_first & has_second & ~compact_pair & ~multi_pair)
                @_scoped("e_pass_pair")
                def execute_noncompact_pair():
                    if compact_only:
                        run_expert_in_passes(
                            first_rows, first_slot_mask, first_copies,
                            first_row_count, current_base_slot,
                            paired_gate=compact_overflow,
                        )
                        run_expert_in_passes(
                            second_rows, second_slot_mask, second_copies,
                            second_row_count, current_base_slot + 1,
                            paired_gate=compact_overflow,
                        )
                    else:
                        execute_full_pair()

                compact_tail = (
                    has_first & ~has_second & (first_row_count == 1)
                )

                @pl.when(compact_tail)
                @_scoped("e_single_tail")
                def execute_compact_tail():
                    first_copies.gate_up.wait()
                    first_copies.gate_up_scales.wait()
                    stage_singleton_source(
                        first_rows, first_single_source
                    )
                    expert_dot(
                        first_single_source,
                        expert_gate_up_vmem.at[current_base_slot],
                        expert_gate_up_scale_vmem.at[current_base_slot],
                        gate_up_accumulator.at[:1, : 2 * expert_width],
                        expert_input_size,
                        2 * expert_width,
                    )
                    finish_singleton_expert(
                        first_rows,
                        first_slot_mask,
                        first_copies,
                        0,
                        0,
                        current_base_slot,
                    )

                if compact_capacity > 1:
                    multi_tail = (
                        has_first & ~has_second & (first_row_count != 1)
                        & (first_row_count <= compact_capacity)
                    )

                    @pl.when(multi_tail)
                    def execute_compact_multi_tail():
                        first_copies.gate_up.wait()
                        first_copies.gate_up_scales.wait()
                        stage_compact_rows(first_rows, first_compact_source_vmem, row_plan=first_plan)
                        expert_dot(
                            first_compact_source_vmem,
                            expert_gate_up_vmem.at[current_base_slot],
                            expert_gate_up_scale_vmem.at[current_base_slot],
                            gate_up_accumulator.at[:compact_capacity, : 2 * expert_width],
                            expert_input_size,
                            2 * expert_width,
                        )
                        if not share_compact_finish:
                            finish_compact_expert(
                                first_rows, first_slot_mask, first_copies,
                                0, 0, current_base_slot,
                            )
                else:
                    multi_tail = jnp.bool_(False)

                @pl.when(
                    has_first
                    & ~has_second
                    & (first_row_count != 1)
                    & ~multi_tail
                )
                def execute_noncompact_tail():
                    if compact_only:
                        run_expert_in_passes(
                            first_rows, first_slot_mask, first_copies,
                            first_row_count, current_base_slot,
                        )
                    else:
                        execute_full_tail()
                if share_full_finish:
                    @pl.when(has_first & ~compact_pair & ~multi_pair & ~compact_tail & ~multi_tail)
                    @_scoped("e_pass_finish")
                    def execute_full_finish():
                        def route_weights(rows, mask):
                            probability = jnp.sum(jnp.where(mask, selected_probabilities, jnp.float32(0)), axis=1) / probability_totals
                            return jnp.where(rows, probability, jnp.float32(0))
                        first_weights = route_weights(first_rows, first_slot_mask)
                        second_weights = route_weights(second_rows, second_slot_mask)
                        @pl.loop(0, jnp.where(has_second, 2, 1))
                        def selected_expert(index):
                            first = index == 0
                            output_weights = jnp.where(first, first_weights, second_weights)
                            expert = jnp.where(first, first_expert, second_expert)
                            slot = current_base_slot + index
                            copies = copy_expert(expert, slot)
                            finish_expert(None, None, copies, index * 2 * expert_width, slot, output_weights)
                if share_compact_finish:
                    @pl.when(multi_pair | multi_tail)
                    @_scoped("e_multi_finish")
                    def execute_compact_finish():
                        first_selector, first_weights = compact_route_data(first_rows, first_slot_mask, row_plan=first_plan)
                        second_selector, second_weights = compact_route_data(second_rows, second_slot_mask, row_plan=second_plan)
                        def finish_selected():
                            @pl.loop(0, jnp.where(has_second, 2, 1))
                            def selected_expert(index):
                                first = index == 0
                                selector = jnp.where(first, first_selector, second_selector)
                                output_weights = jnp.where(first, first_weights, second_weights)
                                expert = jnp.where(first, first_expert, second_expert)
                                slot = current_base_slot + index
                                copies = copy_expert(expert, slot)
                                finish_compact_expert(None, None, copies, index * compact_capacity,
                                                      index * 2 * expert_width, slot,
                                                      route_data=(selector, output_weights))
                        finish_selected()
            else:
                @pl.when(has_first & has_second)
                def execute_pair():
                    execute_full_pair()

                @pl.when(has_first & ~has_second)
                def execute_tail():
                    execute_full_tail()

            if not plan_routes:
                processed_routes_vmem[:, :selected_expert_count] = jnp.where(
                    first_consumed | second_consumed,
                    jnp.int32(1),
                    processed_routes_vmem[:, :selected_expert_count],
                )
            if plan_routes:
                return (third_lowest, fourth_lowest, remaining.astype(jnp.int32))
            if carry_routes:
                return (
                    pack((third_expert, third_rows, third_slot_mask, third_consumed, has_third)),
                    pack((fourth_expert, fourth_rows, fourth_slot_mask, fourth_consumed, has_fourth)),
                )

        if plan_routes:
            jax.lax.fori_loop(
                0, pair_count, run_next_expert_pair,
                (initial_first, initial_second, initial_remaining.astype(jnp.int32)),
            )
        elif carry_routes:
            jax.lax.fori_loop(0, pair_count, run_next_expert_pair, (pack(initial_first), pack(initial_second)))
        else:
            @pl.loop(0, pair_count)
            def run_original_expert_pair(iteration):
                run_next_expert_pair(iteration)

    if hot is not None and group_pending_routes:
        with _phase("e_hot"):
            _run_hot_experts(
                hot, source, selected_experts, selected_probabilities, valid_routes, probability_totals,
                expert_output,
                (expert_gate_up_vmem, expert_gate_up_scale_vmem, expert_down_vmem, expert_down_scale_vmem),
                expert_input_size=expert_input_size, expert_width=expert_width,
                expert_output_size=expert_output_size,
            )


def projection_staging_scratch(batch_size: int, attention_dp: int = 1) -> tuple[Any, ...]:
    """Above B32: send/receive staging of the attention projection's four
    per-pair payloads ([2, 4 * payload, 128] BF16)."""
    if batch_size <= 32:
        return ()
    payload_rows = (batch_size // attention_dp // 4) * 3584 // 128
    return (tpu.VMEM((2, 4 * payload_rows, 128), jnp.bfloat16),)


def routed_send_scratch(batch_size: int) -> tuple[Any, ...]:
    """Above B32: the routed reduce-scatter's send side (eight lanes' tokens)."""
    if batch_size <= 32:
        return ()
    return (tpu.VMEM((8 * 14 * -(-batch_size // 8), 128), jnp.float32),)


def communication_scratch(batch_size: int = 1, attention_dp: int = 1) -> tuple[Any, ...]:
    """Shared for the entire fused stack, including the reduction sequence."""
    # FP32 transport rows per token: 28 for shared outputs and attention
    # projections, 14 for routed outputs, in separate buffers. Above B8 only
    # layer 0's dense all-reduce carries the whole batch; it runs in 8-row
    # halves, so B16 keeps the B8 capacity.
    wide_reduction_rows = max(64, 28 * (min(batch_size, 8) if batch_size <= 16 else 16))
    # Above B32 the host exchange grows with the attention projection's
    # per-pair payloads (staged in projection_staging_scratch).
    host_reduction_rows = max(
        wide_reduction_rows, 7 * batch_size // attention_dp if batch_size > 32 else 0
    )
    # The shared-expert reduce-scatter receives 4 rows per token in each lane
    # and host stage.
    wide_reduction_rows = max(wide_reduction_rows, 4 * batch_size)
    host_reduction_rows = max(host_reduction_rows, 4 * batch_size)
    if attention_dp > 1:
        # The DP MoE-input gather's 4 * DP row wires use lanes 4..7.
        wide_reduction_rows = max(wide_reduction_rows, 7 * batch_size)
    # The eight-lane routed reduce-scatter pads the batch to whole tokens per lane.
    padded_tokens = 8 * -(-batch_size // 8) if batch_size in (6, *SCATTER_BATCHES) else batch_size
    expert_reduction_rows = max(32, 14 * padded_tokens)
    expert_host_rows = expert_reduction_rows
    if batch_size > 32:
        # One lane's tokens per source lane (send side: routed_send_scratch);
        # the host buffer also carries the shared-expert shard sends.
        expert_reduction_rows = 14 * -(-batch_size // 8)
        expert_host_rows = max(expert_reduction_rows, 8 * batch_size)
    # Whole 8-row FP32 tiles: padded strides make the reductions' ref reshapes
    # fail to compile ("reshape on a non-contiguous memref").
    wide_reduction_rows, expert_reduction_rows, host_reduction_rows, expert_host_rows = (
        -(-rows // 8) * 8
        for rows in (wide_reduction_rows, expert_reduction_rows, host_reduction_rows, expert_host_rows))
    return (
        tpu.VMEM((8, wide_reduction_rows, 128), jnp.float32),
        tpu.VMEM((8, expert_reduction_rows, 128), jnp.float32),
        tpu.VMEM((4, host_reduction_rows, 128), jnp.float32),
        tpu.VMEM((4, expert_host_rows, 128), jnp.float32),
        # Attention's group-gather features of every held shard, in native
        # 16-row backing (active rows via _bf16_rows_view).
        tpu.VMEM((max(16, -(-batch_size // 16) * 16), 1024 * attention_dp), jnp.bfloat16),
        # Above B32 (always token-scattered) attention writes only each
        # pair's batch / 4 rows here.
        tpu.VMEM(
            (max(16, -(-(batch_size // 4 if batch_size >= 64 else batch_size) // 16) * 16), 7168),
            jnp.bfloat16,
        ),
        tpu.SemaphoreType.DMA((16,)),
        tpu.SemaphoreType.DMA((16,)),
        tpu.SMEM((1,), jnp.int32),
        tpu.VMEM((batch_size, 4096), jnp.bfloat16),
    )


TensorLike = jax_typing.ArrayLike | jax.ShapeDtypeStruct


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class AttentionCommunicationWorkspace:
    """Collective transport storage shared by attention and MoE layers."""

    shared_local_reduction_vmem: Any  # [8, rows, 128] FP32; rows: see communication_scratch
    expert_local_reduction_vmem: Any  # [8, rows, 128] FP32
    shared_host_reduction_vmem: Any  # [4, rows, 128] FP32
    expert_host_reduction_vmem: Any  # [4, rows, 128] FP32
    attention_features_vmem: Any  # [16 * n, 1024 * attention_dp] BF16
    attention_output_vmem: Any  # [16 * n, 7168] BF16
    send_semaphores: Any  # [16]
    receive_semaphores: Any  # [16]
    reduction_phase: Any  # [1]
    latent_gather_vmem: Any  # [batch_size, 4096]
    projection_staging_vmem: Any = None  # above B32: [2, 4 * payload, 128] BF16
    routed_send_vmem: Any = None  # above B32: [8 * lane payload, 128] FP32
    attention_gather_vmem: Any = None  # B4: private final-gather packets


class _ScopedWait:
    """An async copy whose wait is traced as the e_wait phase."""

    def __init__(self, copy):
        self.copy = copy

    def start(self):
        self.copy.start()

    def wait(self):
        with _phase("e_wait"):
            self.copy.wait()


def _start_async_copies(copies: SimpleNamespace) -> None:
    for copy in vars(copies).values():
        if copy is not None:
            copy.start()


def _gather_output_shards(
    output_vmem,
    send_semaphores,
    receive_semaphores,
    semaphore_slot,
) -> None:
    rank = jax.lax.axis_index("tp")
    local_shard = output_vmem.at[:, pl.ds((rank % 8) * 1024, 1024)]
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
        received_shards,
        received_shards,
        send_semaphores.at[semaphore_slot],
    ).wait()
    tpu.make_async_copy(
        received_shards,
        received_shards,
        receive_semaphores.at[semaphore_slot],
    ).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class MixtureOfExpertsWeights:
    """MoE parameters and persistent resources for asynchronous weight copies."""

    routing_projection: TensorLike  # [hidden_size, 896]
    routing_projection_vmem: Any  # [1, hidden_size, 512]
    routing_copy_semaphores: Any  # [2]
    routing_bias: TensorLike  # [1, 896]
    latent_input_projection: TensorLike  # [hidden_size, 128]
    latent_output_projection: TensorLike  # [expert_output_size, 1024]
    latent_output_projection_vmem: Any  # [expert_output_size, 1024]
    latent_output_copy_semaphore: Any  # scalar DMA semaphore
    latent_normalization: TensorLike  # [1, padded_latent_size]
    shared_gate_up_projection: TensorLike  # [hidden_size, 2 * shared_width]
    shared_down_projection: TensorLike  # [shared_width, hidden_size]
    # Packed MXFP4: [local_experts, expert_input_size / 8, 2 * expert_width].
    # Reblocked FP8: [local_experts, expert_input_size, 2 * expert_width].
    expert_gate_up_projection: TensorLike
    expert_gate_up_projection_vmem: Any  # [4, ...]
    expert_gate_up_scales: TensorLike
    expert_gate_up_scales_vmem: Any  # [4, ...]
    # Packed MXFP4: [local_experts, expert_width / 8, expert_output_size].
    # Reblocked FP8: [local_experts, expert_width, expert_output_size].
    expert_down_projection: TensorLike
    expert_down_projection_vmem: Any  # [4, ...]
    expert_down_scales: TensorLike
    expert_down_scales_vmem: Any  # [4, ...]
    expert_copy_semaphores: Any  # [4, 4]

    def _copy_routing_projection(
        self,
        *,
        wait: bool,
        initialize_padding: bool = True,
    ) -> None:
        lane = jax.lax.axis_index("tp") % 2

        @pl.when(lane == 0)
        def copy_full_half():
            copy = tpu.make_async_copy(
                self.routing_projection.at[:, :512],
                self.routing_projection_vmem.at[0],
                self.routing_copy_semaphores.at[0],
            )
            copy.wait() if wait else copy.start()

        @pl.when(lane == 1)
        def copy_tail():
            if not wait and initialize_padding:
                self.routing_projection_vmem[0, :, 384:] = jnp.zeros(
                    (self.routing_projection.shape[0], 128),
                    jnp.bfloat16,
                )
            copy = tpu.make_async_copy(
                self.routing_projection.at[:, pl.ds(512, 384)],
                self.routing_projection_vmem.at[0, :, :384],
                self.routing_copy_semaphores.at[0],
            )
            copy.wait() if wait else copy.start()

    def start_routing_projection_copy(
        self, *, initialize_padding: bool = True
    ) -> None:
        self._copy_routing_projection(
            wait=False,
            initialize_padding=initialize_padding,
        )

    def wait_for_routing_projection_copy(self) -> None:
        self._copy_routing_projection(wait=True)

    def make_latent_output_copy(self):
        return tpu.make_async_copy(
            self.latent_output_projection,
            self.latent_output_projection_vmem,
            self.latent_output_copy_semaphore,
        )

    def make_expert_copies(
        self,
        expert,
        slot,
        *,
        gate_up: bool = True,
        down: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            gate_up=(
                tpu.make_async_copy(
                    self.expert_gate_up_projection.at[expert],
                    self.expert_gate_up_projection_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 0],
                )
                if gate_up
                else None
            ),
            gate_up_scales=(
                tpu.make_async_copy(
                    self.expert_gate_up_scales.at[expert],
                    self.expert_gate_up_scales_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 1],
                )
                if gate_up
                else None
            ),
            down=(
                tpu.make_async_copy(
                    self.expert_down_projection.at[expert],
                    self.expert_down_projection_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 2],
                )
                if down
                else None
            ),
            down_scales=(
                tpu.make_async_copy(
                    self.expert_down_scales.at[expert],
                    self.expert_down_scales_vmem.at[slot],
                    self.expert_copy_semaphores.at[slot, 3],
                )
                if down
                else None
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class MixtureOfExpertsWorkspace:
    """Persistent VMEM and semaphores used by one MoE invocation."""

    selected_experts_vmem: Any  # [batch_size, 128]
    selected_probabilities_vmem: Any  # [batch_size, 128]
    local_route_counts_vmem: Any  # [batch_size, 128]
    probability_totals_vmem: Any  # [batch_size, 128]
    routing_scores_vmem: Any  # [batch_size, 1024]
    shared_outputs_vmem: Any  # [batch_size, hidden_size]
    routed_outputs_vmem: Any  # [batch_size, padded_latent_size]
    gate_up_accumulator_vmem: Any  # [rows, 4 * expert_width]; rows: _expert_accumulator_rows
    down_accumulator_vmem: Any  # [rows, expert_output_size]
    processed_routes_vmem: Any  # [batch_size, 128]
    first_compact_source_vmem: Any  # [max(1, _compact_rows), expert_input_size]
    second_compact_source_vmem: Any  # [max(1, _compact_rows), expert_input_size]
    shared_send_semaphores: Any  # [10]
    shared_receive_semaphores: Any  # [10]
    output_vmem: Any  # [batch_size, 8192] or token-scatter transport shape
    output_send_semaphores: Any  # [2] or [7]
    output_receive_semaphores: Any  # [2] or [7]
    latent_send_semaphores: Any  # [2]
    latent_receive_semaphores: Any  # [2]
    # Sharded routing only: this pair's selection rows and everyone's.
    selection_source_vmem: Any = None  # [batch_size / 4, 128] FP32
    selections_vmem: Any = None  # [batch_size, 128] FP32


def _consume_completed_routed_chunks(
    *, batch_size, tokens_per_lane, lane, wait_for_peer, consume
):
    """Consume native 16-row tiles after all of their source lanes arrive.

    All seven peer copies must already be in flight. Each wait drains both
    the send to and receive from that peer. No transport source is changed
    until every chunk has been consumed, so later chunks can remain in flight
    while the caller computes on an earlier tile's complete latent vectors.
    """
    if batch_size not in (32, 64) or tokens_per_lane != batch_size // 8:
        raise ValueError("Routed token pipeline needs B32/B64 over eight lanes")
    chunk_rows = 16
    lanes_per_chunk = chunk_rows // tokens_per_lane
    for first in range(0, batch_size, chunk_rows):
        first_lane = first // tokens_per_lane
        for source_lane in range(first_lane, first_lane + lanes_per_chunk):
            @pl.when(lane != source_lane)
            def wait(source_lane=source_lane):
                wait_for_peer(lane ^ source_lane)
        consume(first, chunk_rows, first_lane, lanes_per_chunk)


def moe(
    hidden_ref: TensorLike,  # [batch_size, hidden_size]
    output_ref: TensorLike,  # [batch_size, hidden_size] or owned token rows
    *,
    weights: MixtureOfExpertsWeights,
    workspace: MixtureOfExpertsWorkspace,
    communication_workspace: AttentionCommunicationWorkspace,
    selected_expert_count: int = 16,
    epsilon: float = 1e-5,
    latent_feature_count: int = 3584,
    synchronize_devices: bool = True,
    routing_projection_is_prefetched: bool = False,
    group_pending_expert_routes: bool = False,
    token_scatter_output: bool = False,
    routed_reduce_scatter: bool = False,
    token_pipeline: bool = False,
    after_experts=None,
    after_routing_scores=None,
    overlap_router_latent: bool = False,
    overlap_shared_reduction: bool = False,
    prepare_weights_vmem=None,
    shared_gate_up_stream=None,
    token_scatter_owners: int = 4,
    owned_hidden_ref=None,
    live_rows=None,
) -> None:
    """Execute one mixture-of-experts layer inside a Pallas program.

    ``token_scatter_output`` leaves each of the ``token_scatter_owners`` local
    row owners (4: rank pairs; 8: ranks) with its batch // owners rows.
    ``owned_hidden_ref`` (this rank pair's rows of ``hidden_ref``) shards the
    router's top-k over the rank pairs (_route_owned_rows).

    ``live_rows`` masks null serving slots before route prefetch/compaction;
    the router scores and shared expert still use the compiled batch shape.

    ``prepare_weights_vmem`` optionally supplies caller-owned VMEM for the
    latent-down, shared gate/up and shared down projections instead of a
    scoped allocation.
    """
    batch_size, hidden_size = hidden_ref.shape
    expert_count = weights.routing_projection.shape[1]
    latent_size = weights.latent_normalization.shape[-1]
    expert_output_size = weights.expert_down_projection.shape[-1]
    local_expert_count = weights.expert_gate_up_projection.shape[0]
    # Replicas: entries 112.. of the local expert families are replica slots.
    replica_slots = local_expert_count - expert_count // 8 if REPLICAS else 0
    if replica_slots:
        local_expert_count = expert_count // 8
        if not 0 < replica_slots <= 16:
            raise ValueError("replica slots must be 1..16 (local ids < 128)")
    expert_tensor_parallel = 32 * local_expert_count // expert_count
    down_is_reblocked_fp8 = (
        weights.expert_down_projection.dtype == jnp.float8_e4m3fn
    )
    expert_width = (
        weights.expert_down_projection.shape[1]
        if down_is_reblocked_fp8
        else weights.expert_down_projection.shape[1] * 8
    )
    expert_input_size = (
        weights.expert_gate_up_projection.shape[1] * 8
        if weights.expert_gate_up_projection.dtype == jnp.uint32
        else weights.expert_gate_up_projection.shape[1]
    )
    shared_width = weights.shared_down_projection.shape[0]
    expert_prefetch_slot_count = 4
    hot_info = _HOT.pop() if _HOT else None
    if not (group_pending_expert_routes and batch_size in (32, 64, 96, 128)):
        hot_info = None

    if not 1 <= batch_size <= MAX_BATCH:
        raise ValueError(f"MoE supports 1 to {MAX_BATCH} rows")
    if token_scatter_output and batch_size not in SCATTER_BATCHES and not (
        batch_size in (2, 4) and token_scatter_owners == 1
    ):
        raise ValueError("MoE token-scattered output requires batch size 8 or 16, or B2/B4 to one owner")
    if routed_reduce_scatter and batch_size not in (6, *SCATTER_BATCHES):
        raise ValueError("MoE routed reduce-scatter requires batch size 6, 8 or 16")
    if token_pipeline and (
        batch_size not in (32, 64)
        or not routed_reduce_scatter
        or not token_scatter_output
        or token_scatter_owners != 4
        or expert_tensor_parallel != 4
    ):
        raise ValueError("MoE token pipeline needs B32/B64, EP8/TP4 and pair-owned routed reduce-scatter")
    if weights.expert_gate_up_projection.dtype not in (
        jnp.uint32,
        jnp.float8_e4m3fn,
    ) or weights.expert_down_projection.dtype not in (
        jnp.uint32,
        jnp.float8_e4m3fn,
    ):
        raise ValueError("Unsupported routed-expert weight storage")
    if expert_tensor_parallel not in (1, 2, 4):
        raise ValueError("Expert tensor parallelism must be 1, 2, or 4")
    if (
        hidden_size % 128
        or expert_count % 128
        or latent_size % 256
        or expert_width % 256
        or hidden_size > 8192
        or latent_size > 8192
    ):
        raise ValueError("Unsupported MoE tile dimensions")
    if (
        weights.latent_input_projection.shape != (hidden_size, 128)
        or weights.latent_output_projection.shape
        != (expert_output_size, 1024)
    ):
        raise ValueError("Latent projections require padded TP32 shards")
    if (
        expert_count * expert_tensor_parallel != 32 * local_expert_count
        or expert_output_size > latent_size
        or expert_input_size > latent_size
    ):
        raise ValueError("Expert ownership or latent padding is inconsistent")
    if expert_input_size % 512 or expert_output_size % 128:
        raise ValueError("Expert dimensions must satisfy MXFP4 alignment")

    if weights.expert_gate_up_projection.dtype == jnp.uint32:
        if (
            weights.expert_gate_up_projection.shape[1:]
            != (expert_input_size // 8, 2 * expert_width)
            or weights.expert_gate_up_scales.shape[1:]
            != (expert_input_size // 32, 2 * expert_width)
            or weights.expert_gate_up_scales.dtype != jnp.uint8
        ):
            raise ValueError("MXFP4 gate/up weights have inconsistent shapes")
    elif (
        weights.expert_gate_up_projection.shape[1:]
        != (expert_input_size, 2 * expert_width)
        or weights.expert_gate_up_scales.dtype != jnp.bfloat16
        or not weights.expert_gate_up_scales.shape[1]
        or expert_input_size % weights.expert_gate_up_scales.shape[1]
    ):
        raise ValueError("FP8 gate/up weights have inconsistent shapes")

    if down_is_reblocked_fp8:
        if weights.expert_down_projection.shape[1] != expert_width:
            raise ValueError("Reblocked FP8 down weights have the wrong width")
        if weights.expert_down_scales.dtype != jnp.bfloat16:
            raise ValueError("Reblocked FP8 down scales must be BF16")
        if (
            not weights.expert_down_scales.shape[1]
            or expert_width % weights.expert_down_scales.shape[1]
        ):
            raise ValueError("FP8 down scales must partition the contraction")
    elif (
        weights.expert_down_projection.shape[1] * 8 != expert_width
        or weights.expert_down_scales.dtype != jnp.uint8
    ):
        raise ValueError("MXFP4 down weights or scales have inconsistent shapes")

    expected_output_shape = (
        (batch_size // token_scatter_owners, hidden_size)
        if token_scatter_output
        else hidden_ref.shape
    )
    if output_ref.shape != expected_output_shape:
        raise ValueError("MoE output has the wrong shape for its communication mode")

    accumulator_rows = _expert_accumulator_rows(
        batch_size, weights.expert_gate_up_projection.dtype,
        group_pending_routes=group_pending_expert_routes,
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
    if any(actual != expected for actual, expected in expected_workspace_shapes):
        raise ValueError("MoE workspace has an invalid shape")

    expected_output_vmem_shape = (
        (
            token_scatter_owners + 8,
            batch_size // token_scatter_owners * 1024 // 128,
            128,
        )
        if token_scatter_output
        else (batch_size, 8192)
    )
    if workspace.output_vmem.shape != expected_output_vmem_shape:
        raise ValueError("MoE output communication workspace has the wrong shape")
    if communication_workspace.latent_gather_vmem.shape != (batch_size, 4096):
        raise ValueError("MoE latent-gather workspace must match the batch size")
    if weights.routing_projection_vmem.shape != (1, hidden_size, 512):
        raise ValueError("MoE routing projection workspace has the wrong shape")
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
    if any(actual != expected for actual, expected in expert_workspace_shapes):
        raise ValueError("MoE expert weight workspace has an invalid shape")

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
    # The pooled shared-down weight stays resident in region B until the
    # MoE tail, so the shards are projected when the reduce-scatter starts.
    shared_down_weights = prepare_weights_vmem[2] if prepare_weights_vmem is not None else None
    assert not shared_sharded or shared_down_weights is not None, "shared shards need pooled weights"
    gate_up_accumulator_vmem = workspace.gate_up_accumulator_vmem
    down_accumulator_vmem = workspace.down_accumulator_vmem
    processed_routes_vmem = workspace.processed_routes_vmem
    first_compact_source_vmem = workspace.first_compact_source_vmem
    second_compact_source_vmem = workspace.second_compact_source_vmem
    shared_send_semaphores = workspace.shared_send_semaphores
    shared_receive_semaphores = workspace.shared_receive_semaphores
    output_vmem = workspace.output_vmem
    output_send_semaphores = workspace.output_send_semaphores
    output_receive_semaphores = workspace.output_receive_semaphores
    latent_send_semaphores = workspace.latent_send_semaphores
    latent_receive_semaphores = workspace.latent_receive_semaphores
    shared_local_reduction_vmem = (
        communication_workspace.shared_local_reduction_vmem
    )
    expert_local_reduction_vmem = (
        communication_workspace.expert_local_reduction_vmem
    )
    shared_host_reduction_vmem = (
        communication_workspace.shared_host_reduction_vmem
    )
    expert_host_reduction_vmem = (
        communication_workspace.expert_host_reduction_vmem
    )
    routed_send_vmem = communication_workspace.routed_send_vmem
    send_semaphores = communication_workspace.send_semaphores
    receive_semaphores = communication_workspace.receive_semaphores
    reduction_phase = communication_workspace.reduction_phase
    latent_vmem = communication_workspace.latent_gather_vmem
    rank = jax.lax.axis_index("tp")
    if synchronize_devices:
        collectives32.barrier()
        reduction_phase[0] = 0
    hidden = hidden_ref[...]

    def copy_expert(expert, slot, *, gate_up=True, down=True):
        return weights.make_expert_copies(
            expert,
            slot,
            gate_up=gate_up,
            down=down,
        )

    if not routing_projection_is_prefetched:
        weights.start_routing_projection_copy()
    latent_output_copy = weights.make_latent_output_copy()
    overlap_latent_gather = selected_expert_count >= 12
    overlap_router_latent = (
        overlap_router_latent and overlap_latent_gather and owned_hidden_ref is None
    )
    overlap_shared_routing = overlap_router_latent and shared_sharded
    latent_slot = reduction_phase[0] % 2

    def latent_copy(offset):
        view = latent_vmem.at[:, pl.ds(rank * 128, 128)]
        return tpu.make_async_remote_copy(
            view,
            view,
            latent_send_semaphores.at[latent_slot],
            latent_receive_semaphores.at[latent_slot],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def wait_for_latent_gather():
        view = latent_vmem.at[:, : 31 * 128]
        tpu.make_async_copy(
            view, view, latent_send_semaphores.at[latent_slot]
        ).wait()
        tpu.make_async_copy(
            view, view, latent_receive_semaphores.at[latent_slot]
        ).wait()

    def start_latent_gather(latent_input_projection_vmem, latent_input_copy):
        latent_input_copy.wait()
        latent_vmem[:, pl.ds(rank * 128, 128)] = _dot(
            hidden, latent_input_projection_vmem[...]
        ).astype(jnp.bfloat16)
        for offset in (
            31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 19,
            14, 13, 11, 7, 24, 20, 18, 17, 12, 10, 9, 6, 5, 3,
            16, 8, 4, 2, 1,
        ):
            latent_copy(offset).start()

    def prepare_batch(
        latent_input_projection_vmem,
        latent_input_copy_semaphore,
        shared_gate_up_projection_vmem,
        shared_down_projection_vmem,
        dense_copy_semaphores,
    ):
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
            winners,
            owned_locally,
            route_counts,
            unused_selected_experts,
        ):
            # Router selection order is not the sorted expert-id order of
            # grouped execution. Its weight slots can only be populated once
            # all routes are known; otherwise a wait can consume another
            # expert's correctly sized DMA without detecting the mismatch.
            if group_pending_expert_routes:
                return
            del unused_selected_experts
            nonlocal shared_route_prefix, prefetched_expert_count
            shared_route_prefix *= jnp.all(winners == winners[0]).astype(
                jnp.int32
            )
            should_prefetch = (
                shared_route_prefix
                * owned_locally.astype(jnp.int32)[0]
                * (
                    route_counts[0] < expert_prefetch_slot_count
                ).astype(jnp.int32)
            )

            @pl.when(should_prefetch != 0)
            def start_expert_copy():
                _start_async_copies(
                    copy_expert(
                        winners[0] % local_expert_count,
                        route_counts[0],
                    )
                )

            prefetched_expert_count += should_prefetch

        def overlap_router_score_exchange():
            # Latent projection reads hidden states, not router scores. Its
            # MXU work can run while the score halves move between chip cores.
            start_latent_gather(latent_input_projection_vmem, latent_input_copy)

        def prepare_shared_compute():
            if shared_gate_up_stream is not None:
                shared_gate_up = shared_gate_up_stream.dot(
                    hidden, communication_workspace.reduction_phase[0] >= 0
                ).astype(jnp.bfloat16).astype(jnp.float32)
            else:
                dense_copies.gate_up.wait()
                shared_gate_up = _dot(
                    hidden, shared_gate_up_projection_vmem[...]
                ).astype(jnp.bfloat16).astype(jnp.float32)
            middle = _gated_activation(
                shared_gate_up[:, :shared_width],
                shared_gate_up[:, shared_width:],
            )
            if not overlap_shared_routing:
                dense_copies.down.wait()
            if shared_sharded:
                # The routed outputs are zeroed by the expert stream; until then
                # they hold the BF16 activation (exact in FP32).
                routed_outputs_vmem[:, :shared_width] = middle.astype(jnp.float32)
            else:
                shared_outputs_vmem[...] = _dot(
                    middle, shared_down_projection_vmem[...]
                )

        def overlap_routing_with_latent_gather(selection):
            if overlap_shared_routing and selection == 7:
                prepare_shared_compute()
            if selection == 3 and overlap_latent_gather and not overlap_router_latent:
                start_latent_gather(
                    latent_input_projection_vmem,
                    latent_input_copy,
                )

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
                after_score_send=(
                    overlap_router_score_exchange if overlap_router_latent else None
                ),
                live_rows=live_rows,
                replica_table=(
                    weights.latent_normalization[:, REPLICA_LANE : REPLICA_LANE + 256] if replica_slots else None
                ),
                replica_entries=8 * replica_slots,
            )
        )
        if hot_info is not None:
            selected_experts = _hot_home_routes(
                weights.latent_normalization, selected_experts, local_route_counts, hot_info[3]
            )
        selected_experts_vmem[...] = jnp.pad(
            selected_experts,
            ((0, 0), (0, 128 - selected_expert_count)),
        )
        selected_probabilities_vmem[...] = jnp.pad(
            selected_probabilities,
            ((0, 0), (0, 128 - selected_expert_count)),
        )
        local_route_counts_vmem[...] = jnp.broadcast_to(
            local_route_counts[:, None], local_route_counts_vmem.shape
        )
        probability_totals_vmem[...] = jnp.broadcast_to(
            selected_probability_totals[:, None],
            probability_totals_vmem.shape,
        )
        _start_initial_expert_weight_copies(
            selected_experts,
            local_route_counts,
            copy_expert,
            slot_count=expert_prefetch_slot_count,
            already_started=prefetched_expert_count,
            group_pending_routes=group_pending_expert_routes,
            skip_copies=hot_info is not None,
        )

        if not overlap_latent_gather:
            start_latent_gather(
                latent_input_projection_vmem,
                latent_input_copy,
            )
            wait_for_latent_gather()

        if overlap_shared_routing:
            dense_copies.down.wait()
        else:
            prepare_shared_compute()

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

    shared_rows = ((hidden_size + 1023) // 1024) * 8
    shared_payload = shared_rows // 2
    expert_rows = expert_output_size // 128
    expert_payload = expert_rows // 2
    # The routed reduce-scatter gives each local lane ceil(batch / 8) tokens
    # (B6 pads to eight).
    tokens_per_lane = -(-batch_size // 8)
    lane_payload = tokens_per_lane * expert_payload
    # Each lane consumes only its 1024-column shared-expert output shard;
    # reduce-scattering it avoids replicating all 7168 columns, with the same
    # BF16 rounding points.
    reduce_scatter_shared_output = (
        batch_size in (6, *SCATTER_BATCHES) and group_pending_expert_routes
    )
    # The routed payload is eight contiguous token shards: each local lane
    # reduces one, then a local all-gather restores the batch for latent-up.
    reduce_scatter_routed_output = routed_reduce_scatter
    shared_shard_width = 1024
    shared_shard_rows = shared_shard_width // 128
    shared_shard_payload = shared_shard_rows // 2

    def encode_shared(value, destination):
        transport_batch_size = value.shape[0]
        packed = tpu.bitcast(
            value.reshape(transport_batch_size * shared_rows, 128).astype(
                jnp.bfloat16
            ),
            jnp.uint32,
        )
        destination[: transport_batch_size * shared_payload, :] = (
            jax.lax.bitcast_convert_type(packed, jnp.float32)
        )

    def decode_shared(value):
        transport_batch_size = value.shape[1] // shared_payload
        decoded = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(decoded, axis=0).reshape(
            transport_batch_size, shared_rows * 128
        )

    def encode_shared_shard(value, destination):
        transport_batch_size = value.shape[0]
        packed = tpu.bitcast(
            value.reshape(
                transport_batch_size * shared_shard_rows, 128
            ).astype(jnp.bfloat16),
            jnp.uint32,
        )
        destination[: transport_batch_size * shared_shard_payload, :] = (
            jax.lax.bitcast_convert_type(packed, jnp.float32)
        )

    def decode_shared_shard(value):
        transport_batch_size = value.shape[1] // shared_shard_payload
        decoded = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(decoded, axis=0).reshape(
            transport_batch_size, shared_shard_width
        )

    def encode_expert(value, destination):
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

    def decode_expert(value):
        transport_batch_size = value.shape[1] // expert_payload
        values = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(values, axis=0).reshape(
            transport_batch_size, expert_output_size
        )

    def decode_gathered_experts(value):
        values = tpu.bitcast(
            jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return values.reshape(8 * tokens_per_lane, expert_output_size)[:batch_size]

    def shared_copy(payload, offset, across_hosts=False):
        lane, host = rank % 8, rank // 8
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

    def shared_shard_send_view(target_lane):
        # Reuse the expert host buffer until the routed local reduction
        # finishes and needs it for its normal role.
        send_rows = (
            expert_host_reduction_vmem.shape[0]
            * expert_host_reduction_vmem.shape[1]
            // 8
        )
        return pool_alias.view(
            expert_host_reduction_vmem,
            (8, send_rows, 128),
            jnp.float32,
        ).at[target_lane, : batch_size * shared_shard_payload, :]

    def shared_shard_receive_view(source_lane):
        return shared_local_reduction_vmem.at[
            source_lane, : batch_size * shared_shard_payload, :
        ]

    def shared_shard_copy(offset):
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

    def expert_copy(payload, offset, across_hosts=False):
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

    def routed_send_storage():
        if routed_send_vmem is not None:
            return routed_send_vmem.at[: 8 * lane_payload, :]
        return expert_local_reduction_vmem.at[rank % 8, : 8 * lane_payload, :]

    def expert_shard_send_view(target_lane):
        return routed_send_storage().reshape(8, lane_payload, 128).at[target_lane]

    def expert_shard_receive_view(source_lane):
        return expert_local_reduction_vmem.at[
            source_lane, :lane_payload, :
        ]

    def expert_shard_copy(offset):
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

    def begin_shared_reduction(local_shared_outputs):
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
                (batch_size * shared_shard_payload, 128),
                jnp.float32,
            )
            shared_shard_receive_view(lane)[...] = (
                shared_shard_send_view(lane)[...]
            )
            for offset in range(7, 0, -1):
                shared_shard_copy(offset).start()

        else:
            payload = local_shared_outputs.shape[0] * shared_payload
            assert payload <= shared_local_reduction_vmem.shape[1]
            encode_shared(
                local_shared_outputs,
                shared_local_reduction_vmem.at[rank % 8],
            )
            for offset in range(7, 0, -1):
                shared_copy(payload, offset).start()

    def finish_shared_reduction(payload):
        if reduce_scatter_shared_output:
            for offset in range(7, 0, -1):
                shared_shard_copy(offset).wait()
            encode_shared_shard(
                decode_shared_shard(
                    shared_local_reduction_vmem[:, :payload, :]
                ),
                shared_host_reduction_vmem.at[rank // 8],
            )
        else:
            for offset in range(7, 0, -1):
                shared_copy(payload, offset).wait()
            encode_shared(
                decode_shared(
                    shared_local_reduction_vmem[:, :payload, :]
                ),
                shared_host_reduction_vmem.at[rank // 8],
            )
        for offset in range(3, 0, -1):
            shared_copy(payload, offset, True).start()

    def finish_shared_host_reduction(payload, reduced_shared_outputs):
        for offset in range(3, 0, -1):
            shared_copy(payload, offset, True).wait()
        if reduce_scatter_shared_output:
            lane = rank % 8

            @pl.when(lane < 7)
            def store_shared_shard():
                (
                    reduced_shared_outputs
                    if shared_sharded
                    else reduced_shared_outputs.at[
                        :, pl.ds(lane * shared_shard_width, shared_shard_width)
                    ]
                )[...] = decode_shared_shard(
                    shared_host_reduction_vmem[:, :payload, :]
                )
        else:
            reduced_shared_outputs[...] = decode_shared(
                shared_host_reduction_vmem[:, :payload, :]
            )

    def finish_reductions(
        local_expert_outputs,
        reduced_expert_outputs,
        reduced_shared_outputs,
    ):
        transport_batch_size = local_expert_outputs.shape[0]
        shared_transport_payload = transport_batch_size * (
            shared_shard_payload if reduce_scatter_shared_output else shared_payload
        )
        expert_transport_payload = transport_batch_size * expert_payload
        assert shared_transport_payload <= shared_local_reduction_vmem.shape[1]
        assert reduce_scatter_routed_output or expert_transport_payload <= expert_local_reduction_vmem.shape[1]
        assert routed_send_vmem is None or reduce_scatter_routed_output

        latent_output_copy.start()
        if reduce_scatter_routed_output:
            t1 = _phase_start("t1_routed_send")
            routed_scatter_storage = routed_send_storage()
            if batch_size < 8 * tokens_per_lane:
                # Zero the padding tokens of the last lanes.
                routed_scatter_storage[...] = jnp.zeros(
                    routed_scatter_storage.shape, jnp.float32
                )
            encode_expert(local_expert_outputs, routed_scatter_storage)
            for offset in range(7, 0, -1):
                expert_shard_copy(offset).start()
            _phase_end(t1)
        else:
            encode_expert(
                local_expert_outputs,
                expert_local_reduction_vmem.at[rank % 8],
            )
            for offset in range(7, 0, -1):
                expert_copy(expert_transport_payload, offset).start()

        if after_experts is not None:
            after_experts()

        # The polled path completes this during or immediately after experts.
        t2 = _phase_start("t2_shared_local")
        if not overlap_shared_reduction:
            finish_shared_reduction(shared_transport_payload)
        _phase_end(t2)

        if reduce_scatter_routed_output:
            t3 = _phase_start("t3_routed_local")
            lane = rank % 8
            for offset in range(7, 0, -1):
                expert_shard_copy(offset).wait()
            owned_shard = expert_shard_send_view(lane)[...]
            expert_shard_receive_view(lane)[...] = owned_shard
            encode_expert(
                decode_expert(
                    expert_local_reduction_vmem[:, :lane_payload, :]
                ),
                expert_host_reduction_vmem.at[rank // 8],
            )
            _phase_end(t3)
            t4 = _phase_start("t4_routed_host")
            for offset in range(3, 0, -1):
                expert_copy(lane_payload, offset, True).start()
            for offset in range(3, 0, -1):
                expert_copy(lane_payload, offset, True).wait()
            _phase_end(t4)
            t5 = _phase_start("t5_routed_gather")

            reduced_token = decode_expert(
                expert_host_reduction_vmem[:, :lane_payload, :]
            )
            encode_expert(
                reduced_token,
                expert_local_reduction_vmem.at[lane],
            )
            for offset in range(7, 0, -1):
                expert_copy(lane_payload, offset).start()
            if overlap_shared_reduction and not token_pipeline:
                # Shared-output conversion/reduction is independent of the
                # routed gather and uses a separate transport buffer.
                with _phase("t6_shared_during_routed_gather"):
                    finish_shared_host_reduction(
                        shared_transport_payload, reduced_shared_outputs
                    )
            if token_pipeline:
                # The shared contribution must be ready before a completed
                # routed tile can finish its latent-up/add. Its local sends
                # were drained above, before expert_host_reduction_vmem was
                # reused for the routed host reduction.
                t6 = _phase_start("t6_shared_host")
                finish_shared_host_reduction(
                    shared_transport_payload, reduced_shared_outputs
                )
                _phase_end(t6)
                latent_output_copy.wait()

                def consume(first, rows, first_lane, lane_count):
                    # Each gathered lane contains fully reduced latent
                    # vectors, not feature fragments. Same BF16 rounding point
                    # and token order as the unpipelined gather.
                    packed = expert_local_reduction_vmem[
                        first_lane:first_lane + lane_count, :lane_payload, :
                    ]
                    values = tpu.bitcast(
                        jax.lax.bitcast_convert_type(packed, jnp.uint32),
                        jnp.bfloat16,
                    ).astype(jnp.float32).reshape(rows, expert_output_size)
                    reduced_expert_outputs[
                        pl.ds(first, rows), :expert_output_size
                    ] = values
                    consume_routed_chunk(first, rows)

                _consume_completed_routed_chunks(
                    batch_size=batch_size,
                    tokens_per_lane=tokens_per_lane,
                    lane=lane,
                    wait_for_peer=lambda offset: expert_copy(lane_payload, offset).wait(),
                    consume=consume,
                )
                _phase_end(t5)
            else:
                for offset in range(7, 0, -1):
                    expert_copy(lane_payload, offset).wait()
                reduced_expert_outputs[:, :expert_output_size] = (
                    decode_gathered_experts(
                        expert_local_reduction_vmem[:, :lane_payload, :]
                    )
                )
                _phase_end(t5)
                if not overlap_shared_reduction:
                    t6 = _phase_start("t6_shared_host")
                    finish_shared_host_reduction(
                        shared_transport_payload, reduced_shared_outputs
                    )
                    _phase_end(t6)
            reduction_phase[0] += 2
            return
        for offset in range(7, 0, -1):
            expert_copy(expert_transport_payload, offset).wait()
        encode_expert(
            decode_expert(
                expert_local_reduction_vmem[:, :expert_transport_payload, :]
            ),
            expert_host_reduction_vmem.at[rank // 8],
        )
        for offset in range(3, 0, -1):
            expert_copy(
                expert_transport_payload, offset, True
            ).start()
        for offset in range(3, 0, -1):
            expert_copy(expert_transport_payload, offset, True).wait()
        reduced_expert_outputs[...] = jnp.zeros(
            reduced_expert_outputs.shape, jnp.float32
        )
        reduced_expert_outputs[:, :expert_output_size] = decode_expert(
            expert_host_reduction_vmem[:, :expert_transport_payload, :]
        )

        finish_shared_host_reduction(
            shared_transport_payload, reduced_shared_outputs
        )
        reduction_phase[0] += 2


    if shared_sharded:
        assert reduce_scatter_shared_output
        begin_shared_reduction(
            routed_outputs_vmem[:, :shared_width].astype(jnp.bfloat16)
        )
    else:
        begin_shared_reduction(shared_outputs_vmem[...])
    if overlap_latent_gather:
        wait_for_latent_gather()
    def run_with_shared_progress(progress_state):
        progress_state[0] = 0
        payload = batch_size * (shared_shard_payload if reduce_scatter_shared_output else shared_payload)
        # Seven copies, each with its own send/receive semaphore. DMA counters
        # count 32-byte units; both ends must be complete before reusing sources.
        expected = payload * 128 * 4 // 32

        def poll_shared(iteration):
            @pl.when((progress_state[0] == 0) & (iteration % 4 == 0))
            def try_progress():
                ready = jnp.bool_(True)
                for peer in range(7):
                    ready = ready & (pl.semaphore_read(shared_send_semaphores.at[peer]) >= expected)
                    ready = ready & (pl.semaphore_read(shared_receive_semaphores.at[peer]) >= expected)
                @pl.when(ready)
                def advance():
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
            hot=(None if hot_info is None else SimpleNamespace(
                refs=hot_info[0], layer=hot_info[1], views=hot_info[2], slots=hot_info[3],
                semaphores=weights.expert_copy_semaphores)),
            )
        @pl.when(progress_state[0] == 0)
        def drain_shared():
            with _phase("t2_shared_local_drain"):
                finish_shared_reduction(payload)
            progress_state[0] = 1

    if overlap_shared_reduction:
        pl.run_scoped(run_with_shared_progress, progress_state=tpu.SMEM((1,), jnp.int32))
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
            hot=(None if hot_info is None else SimpleNamespace(
                refs=hot_info[0], layer=hot_info[1], views=hot_info[2], slots=hot_info[3],
                semaphores=weights.expert_copy_semaphores)),
            )

    def consume_routed_chunk(first, rows):
        with _phase("t7_pipeline_latent_up"):
            row_slice = pl.ds(first, rows)
            routed = routed_outputs_vmem[row_slice].astype(jnp.bfloat16).astype(jnp.float32)
            normalized = (
                routed
                * jax.lax.rsqrt(
                    jnp.sum(routed * routed, axis=1, keepdims=True)
                    / latent_feature_count
                    + epsilon
                )
                * weights.latent_normalization[...]
            ).astype(jnp.bfloat16)
            up = _dot(
                normalized[:, :expert_output_size],
                weights.latent_output_projection_vmem[...],
            )
            local_output = (
                up
                + (
                    shared_outputs_vmem[row_slice]
                    if shared_sharded
                    else shared_outputs_vmem[
                        row_slice, pl.ds((rank % 8) * 1024, 1024)
                    ]
                )
            ).astype(jnp.bfloat16)
            # Expert input latents are dead after the expert stream. This
            # storage is disjoint from the still-live routed gather packets.
            latent_vmem[row_slice, :1024] = local_output

    finish_reductions(
        routed_outputs_vmem[...],
        routed_outputs_vmem,
        shared_outputs_vmem,
    )
    t7 = _phase_start("t7_latent_up")
    if token_pipeline:
        local_output = latent_vmem[:, :1024]
    else:
        routed = routed_outputs_vmem[...].astype(jnp.bfloat16).astype(jnp.float32)
        normalized = (
            routed
            * jax.lax.rsqrt(
                jnp.sum(routed * routed, axis=1, keepdims=True)
                / latent_feature_count
                + epsilon
            )
            * weights.latent_normalization[...]
        ).astype(jnp.bfloat16)
        latent_output_copy.wait()
        # The contraction is independent across batch rows. Let Mosaic lower
        # it as one native row tile so the resident weight is consumed once.
        up = _dot(
            normalized[:, :expert_output_size],
            weights.latent_output_projection_vmem[...],
        )

        local_output = (
            up
            + (
                shared_outputs_vmem[...]
                if shared_sharded
                else shared_outputs_vmem[:, pl.ds((rank % 8) * 1024, 1024)]
            )
        ).astype(jnp.bfloat16)
    _phase_end(t7)
    t8 = _phase_start("t8_output_transport")
    if token_scatter_output:
        # Each lane owns 1024 hidden columns for all tokens. Send each peer
        # only the rows owned by its local row owner (rank pair or rank).
        # All seven transfers are independent messages, so the token /
        # feature transpose completes in one communication phase.
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
            ].reshape(
                payload_rows, 128
            )
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
            block = receive[source_lane].reshape(
                tokens_per_pair, 1024
            )
            output_ref[:, pl.ds(source_lane * 1024, 1024)] = block
    else:
        output_vmem[:, pl.ds((rank % 8) * 1024, 1024)] = local_output
        _gather_output_shards(
            output_vmem,
            output_send_semaphores,
            output_receive_semaphores,
            reduction_phase[0] % 2,
        )
        output_ref[...] = output_vmem[
            :batch_size, :hidden_size
        ].astype(hidden_ref.dtype)
    _phase_end(t8)
    reduction_phase[0] += 1


# Weight streaming. Every large head-sharded attention weight is copied
# through a ChunkedWeightStream; KernelConfig.weight_streams (JSON {name: [chunks,
# depth]}) splits one into chunks through a ring of ``depth`` slots. The
# default, one chunk, is exactly the whole-weight copy and product.
STREAMED_WEIGHTS = {
    # name: (axis, semaphore base in the stream semaphores when chunked)
    "k_projection": ("columns", 0),
    "k_o": ("rows", 4),
    "m_qb": ("columns", 0),
    "m_kb": ("columns", 4),
    "m_gate": ("columns", 8),
    "m_o": ("rows", 12),
    "m_qa": ("columns", 16),
    "shared_gu": ("columns", 20),
}
WEIGHT_STREAMS = {
    name: tuple(value)
    for name, value in _config().weight_streams
}
if set(WEIGHT_STREAMS) - set(STREAMED_WEIGHTS):
    raise ValueError(f"KernelConfig.weight_streams names unknown weights {sorted(set(WEIGHT_STREAMS) - set(STREAMED_WEIGHTS))}")


# Rings deeper than four slots (attention DP's small batches) signal from
# here, past every weight's own four: a KDA and an MLA layer's streams never
# overlap in time.
DEEP_STREAM_BASE = 24


def _stream_base(name, depth):
    return STREAMED_WEIGHTS[name][1] if depth <= 4 else DEEP_STREAM_BASE


def _stream_semaphore_count(streams) -> int:
    """Stream semaphores for a configuration (each weight owns 4 from its
    base). A minimum of 8 keeps the scalar schedule stable."""
    return max(
        [8]
        + [
            _stream_base(name, depth) + depth
            for name, (chunks, depth) in streams.items()
            if chunks > 1
        ]
    )


STREAM_SEMAPHORES = _stream_semaphore_count(WEIGHT_STREAMS)

# B32/B64 paged DCP needs these chunks to leave room for communication scratch.
# These do not alter arithmetic or the resident weight representation.
DCP_B32_WEIGHT_STREAMS = {
    "m_qa": (12, 4),
    "k_projection": (13, 4),
    "shared_gu": (3, 3),
    "m_gate": (3, 2),
}
# Plain decode's input-projection rings fill the arena that layer 0's dense
# slabs size.
DCP_B32_DECODE_RINGS = {"m_qa": (12, 7), "k_projection": (13, 7)}


# Attention data parallelism: with DP groups of 32 / dp ranks, every rank holds
# the logical TP32 shards t, t + 32 / dp, ... (t = rank % (32 / dp)) of each
# head-sharded attention weight, concatenated along the axis it streams on
# ("concat", per-layer axis) or stacked on a new leading axis ("stack").
# Replicated weights (k_gate, m_qa, m_ka, norms) are unchanged.
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
# Stream defaults per attention-DP degree: shard-sized chunks through one slot,
# output projections as rank-pair halves (own halves first) through two slots.
DP_WEIGHT_STREAMS = {
    2: {
        # Whole shards through one slot; half-shard chunks through two slots
        # would need a wider slot.
        "k_projection": (2, 1),
        "k_o": (4, 2),
        # Shard chunks through one slot keep MLA's region-B share (q/kv-up
        # and gate) under the MoE shared-expert weights it alternates with.
        "m_qb": (2, 1),
        "m_kb": (2, 1),
        "m_gate": (2, 1),
        "m_o": (4, 2),
    },
    4: {
        # A TP8 KDA input projection is 91 MiB. Keep only one TP32
        # head-shard (22.75 MiB) resident and refill after each shard.
        "k_projection": (4, 1),
        "k_o": (8, 2),
        "m_qb": (4, 1),
        "m_kb": (4, 1),
        "m_gate": (4, 1),
        "m_o": (8, 2),
    },
}


def _dp_dcp_weight_streams(dp: int, group_rows: int) -> dict[str, tuple[int, int]]:
    """Attention DP under DCP: the KDA input projection (13 chunks per held
    shard) and MLA q-down (12) stream so the DCP pages and packets fit, while
    key-value-up stays resident. DP2 ring depths shrink as the group rows
    grow; with one group row (B2) MLA query-up and gate are also copied whole."""
    if dp != 2:
        depths = (4, 4)
    elif group_rows <= 8:
        depths = (13, 12)
    else:
        depths = {16: (10, 9), 32: (7, 7)}.get(group_rows, (4, 4))
    streams = {"k_projection": (13 * dp, depths[0]), "m_qa": (12, depths[1]), "m_kb": (dp, dp)}
    if dp == 2 and group_rows == 1:
        streams.update(m_qb=(dp, dp), m_gate=(dp, dp))
    return streams


def _attention_projection_streams(dp: int, schedule: str) -> dict[str, tuple[int, int]]:
    """Static stream overrides for the opt-in attention-DP projection schedule.

    ``mla_up_double`` double-buffers MLA query-up and KV-up without changing
    the projection dot shapes; the gate projection stays single-buffered.
    """
    choices = ("default", "mla_up_double")
    if schedule not in choices:
        raise ValueError(f"attention_projection_schedule must be one of {choices}")
    if schedule == "default":
        return {}
    if dp not in (2, 4):
        raise ValueError("Projection schedule experiments require attention DP2/DP4")
    return {name: (dp, 2) for name in ("m_qb", "m_kb")}

def attention_dp_shape(name, shape, dp):
    """The per-rank ``[layers, ...]`` shape of ``name`` under attention DP."""
    if dp == 1 or name not in ATTENTION_DP_WEIGHTS:
        return tuple(shape)
    kind, axis = ATTENTION_DP_WEIGHTS[name]
    shape = list(shape)
    if kind == "stack":
        return (shape[0], dp, *shape[1:])
    shape[1 + axis] *= dp
    return tuple(shape)


def to_attention_dp(weights, mesh, dp):
    """Rearrange TP32 weights (``[32, layers, ...]`` per family) for attention
    DP: rank r gathers the shards of ranks r % (32 / dp) + k * (32 / dp)."""
    if dp == 1:
        return weights
    from jax.sharding import PartitionSpec as P

    if dp not in (2, 4):
        raise ValueError("Attention DP supports dp=1, dp=2 or dp=4")
    group = 32 // dp

    def rearrange(name):
        kind, axis = ATTENTION_DP_WEIGHTS[name]

        def local(value):
            value = value[0]
            # XOR partners collect every held TP32 shard. Canonical shard
            # order is independent of the physical DP group owning this rank.
            parts = (value,) + tuple(
                jax.lax.ppermute(
                    value, "tp", perm=[(rank, rank ^ (offset * group)) for rank in range(32)]
                )
                for offset in range(1, dp)
            )
            data_group = jax.lax.axis_index("tp") // group
            ordered = []
            for shard in range(dp):
                part = parts[0]
                for offset in range(1, dp):
                    part = jnp.where((data_group ^ offset) == shard, parts[offset], part)
                ordered.append(part)
            if kind == "stack":
                return jnp.stack(ordered, axis=1)[None]
            return jnp.concatenate(ordered, axis=1 + axis)[None]

        return jax.jit(
            jax.shard_map(local, mesh=mesh, in_specs=P("tp"), out_specs=P("tp"), check_vma=False)
        )

    return {
        name: rearrange(name)(value) if name in ATTENTION_DP_WEIGHTS else value
        for name, value in weights.items()
    }


def attention_dp_abstract(weights, dp):
    """``to_attention_dp`` applied to a ShapeDtypeStruct tree."""
    return {
        name: jax.ShapeDtypeStruct(
            (value.shape[0], *attention_dp_shape(name, value.shape[1:], dp)),
            value.dtype,
            sharding=getattr(value, "sharding", None),
        )
        for name, value in weights.items()
    }


def _chunk_sizes(extent: int, chunks: int, unit: int) -> tuple[int, ...]:
    """Split ``extent`` into ``chunks`` whole ``unit`` blocks, larger first."""
    blocks, remainder = divmod(extent // unit, chunks)
    if extent % unit or not 1 <= chunks <= extent // unit:
        raise ValueError(f"Cannot split {extent} into {chunks} chunks of whole {unit}-blocks")
    return tuple(unit * (blocks + (index < remainder)) for index in range(chunks))


class ChunkedWeightStream:
    """A weight copied from HBM in chunks, through a ring of VMEM slots.

    ``source`` [K, N] is split along ``axis`` ("columns": output columns;
    "rows": the contraction) into ``sizes``. Position i of the stream holds
    chunk ``order(i)`` (default i; may be a traced value when all chunks have
    one size), in slot i % depth, signalling semaphore ``base + i % depth``.
    With ``depth`` equal to the chunk count the slots are the chunks' own
    parts of ``full`` (the weight's whole VMEM view); a smaller depth reuses
    slots inside ``full``'s pool range (``pool`` from byte ``offset``).

    ``start`` issues the first ``depth`` positions. Consumers ``wait`` for a
    position, read ``view``, then ``release`` it, which issues the position
    ``depth`` ahead into the freed slot; ``dot`` does this for column streams.
    Chunked products are not bit-identical to the whole-weight product (the
    compiler's MXU accumulation split depends on the shape).
    """

    def __init__(self, source, full, semaphores, *, axis, sizes, depth, base=0,
                 pool=None, offset=None, order=None):
        if axis not in ("columns", "rows"):
            raise ValueError("Stream axis must be 'columns' or 'rows'")
        self.axis, self.sizes, self.depth, self.chunks = axis, tuple(sizes), depth, len(sizes)
        extent = source.shape[1 if axis == "columns" else 0]
        if sum(self.sizes) != extent:
            raise ValueError("Chunk sizes must cover the streamed axis")
        if not 1 <= depth <= self.chunks or base + depth > semaphores.shape[0]:
            raise ValueError("Stream depth must fit the chunks and the semaphores")
        if order is not None and len(set(self.sizes)) != 1:
            raise ValueError("A reordered stream needs equal chunk sizes")
        self.source, self.full, self.semaphores, self.base, self.order = (
            source, full, semaphores, base, order,
        )
        self.slots = None
        if depth < self.chunks:
            if pool is None:
                raise ValueError("A ring stream needs its weight's pool range")
            size = max(self.sizes)
            shape = (full.shape[0], size) if axis == "columns" else (size, full.shape[1])
            slot_bytes = pool_alias.storage_size(shape, full.dtype)
            if depth * slot_bytes > pool_alias.storage_size(full.shape, full.dtype):
                raise ValueError("Ring slots exceed the weight's pool range")
            self.slots = tuple(
                pool_alias.view(
                    pool.at[pl.ds((offset + slot * slot_bytes) // 128, slot_bytes // 128), :],
                    shape,
                    full.dtype,
                )
                for slot in range(depth)
            )

    def _chunk(self, position):
        return position if self.order is None else self.order(position)

    def _extent(self, position):
        chunk = self._chunk(position)
        if self.order is None:
            return sum(self.sizes[:chunk]), self.sizes[chunk]
        return chunk * self.sizes[0], self.sizes[0]

    def _part(self, ref, start, size):
        if self.axis == "columns":
            return ref.at[:, pl.ds(start, size)]
        return ref.at[pl.ds(start, size)]

    def view(self, position):
        start, size = self._extent(position)
        if self.chunks == 1:
            return self.full
        if self.slots is None:
            return self._part(self.full, start, size)
        return self._part(self.slots[position % self.depth], 0, size)

    def _copy(self, position):
        start, size = self._extent(position)
        source = self.source if self.chunks == 1 else self._part(self.source, start, size)
        return tpu.make_async_copy(
            source, self.view(position), self.semaphores.at[self.base + position % self.depth]
        )

    def start(self):
        for position in range(self.depth):
            self._copy(position).start()

    def wait(self, position):
        self._copy(position).wait()

    def wait_all(self):
        if self.slots is not None:
            raise ValueError("Only a stream with a slot per chunk can wait for all chunks")
        for position in range(self.chunks):
            self.wait(position)

    def release(self, position, guard):
        """Refill the slot freed by ``position``. ``guard`` is a runtime-true
        condition that keeps the refill in its own region, after the loads of
        the chunk it overwrites."""
        if position + self.depth < self.chunks:
            @pl.when(guard)
            def refill():
                self._copy(position + self.depth).start()

    def dot_position(self, x, position, guard):
        """``x`` times one column chunk (FP32): wait, multiply, release."""
        if self.axis != "columns":
            raise ValueError("dot_position is for column streams")
        self.wait(position)
        result = _dot(x, self.view(position)[...])
        self.release(position, guard)
        return result

    def dot(self, x, guard):
        """``x @ source`` in FP32 for a column stream."""
        if self.axis != "columns":
            raise ValueError("dot is for column streams")
        if self.slots is None:
            # A DMA wait is a scheduling boundary: with a slot per chunk,
            # resolve every transfer before the products so they overlap.
            self.wait_all()
        parts = []
        for position in range(self.chunks):
            if self.slots is not None:
                self.wait(position)
            parts.append(_dot(x, self.view(position)[...]))
            self.release(position, guard)
        return parts[0] if self.chunks == 1 else jnp.concatenate(parts, axis=1)


class _ConditionalCopy:
    """A DMA started and waited only where ``condition`` holds."""

    def __init__(self, copy, condition):
        self.copy, self.condition = copy, condition

    def start(self):
        pl.when(self.condition)(self.copy.start)

    def wait(self):
        pl.when(self.condition)(self.copy.wait)


class KdaStateStream:
    """KDA state streamed through three rolling or four private VMEM buffers.

    Normally buffer c % 3 is reused only after its write has landed. B32's
    unrolled path uses four buffers: prime chunks 0/1 before projection,
    then start chunks 2/3 at recurrence entry. Final writes drain on exit.
    With slots, read each sequence's initial state directly from the bank;
    subsequent verifier rows are produced by the recurrence before use.
    """

    def __init__(self, history_hbm, state_hbm, history_output_hbm, state_output_hbm,
                 history_buffers, state_buffers, read_semaphores, write_semaphores, *, chunk, chunks, slots=None, sequence_rows=1,
                 inplace_period=None):
        self.history_hbm, self.state_hbm = history_hbm, state_hbm
        self.history_output_hbm, self.state_output_hbm = history_output_hbm, state_output_hbm
        self.history_buffers, self.state_buffers = history_buffers, state_buffers
        self.read_semaphores, self.write_semaphores = read_semaphores, write_semaphores
        self.chunk, self.chunks = chunk, chunks
        self.slots = slots
        self.sequence_rows = sequence_rows
        # In place: each row's post-states go to its own slot of the input banks;
        # null-slot rows (slot % inplace_period == 0) are not written.
        self.inplace_period = inplace_period
        assert inplace_period is None or (slots is not None and sequence_rows == 1)
        self.pending_writes = []
        self.buffer_count = history_buffers.shape[0]
        self.four_buffers = self.buffer_count == 4
        assert not self.four_buffers or (self.chunks == 4 and self.chunk == 8 and not KDA_STAGED)

    def _reads(self, index):
        rows, slot = pl.ds(index * self.chunk, self.chunk), index % self.buffer_count
        if self.slots is not None:
            return tuple(
                tpu.make_async_copy(source.at[pl.ds(self.slots[index * self.chunk + row], 1)],
                                    buffer.at[slot, pl.ds(row, 1)],
                                    self.read_semaphores.at[slot, kind])
                for kind, (source, buffer) in enumerate((
                    (self.history_hbm, self.history_buffers),
                    (self.state_hbm, self.state_buffers),
                ))
                for row in range(0, self.chunk, self.sequence_rows)
            )
        return (
            tpu.make_async_copy(self.history_hbm.at[rows], self.history_buffers.at[slot],
                                self.read_semaphores.at[slot, 0]),
            tpu.make_async_copy(self.state_hbm.at[rows], self.state_buffers.at[slot],
                                self.read_semaphores.at[slot, 1]),
        )

    def _writes(self, index):
        rows, slot = pl.ds(index * self.chunk, self.chunk), index % self.buffer_count
        if self.inplace_period is not None:
            copies = []
            for row in range(self.chunk):
                target = self.slots[index * self.chunk + row]
                for kind, (buffer, destination) in enumerate((
                    (self.history_buffers, self.history_hbm),
                    (self.state_buffers, self.state_hbm),
                )):
                    copies.append(_ConditionalCopy(
                        tpu.make_async_copy(buffer.at[slot, pl.ds(row, 1)],
                                            destination.at[pl.ds(target, 1)],
                                            self.write_semaphores.at[slot, kind]),
                        target % self.inplace_period != 0))
            return tuple(copies)
        return (
            tpu.make_async_copy(self.history_buffers.at[slot], self.history_output_hbm.at[rows],
                                self.write_semaphores.at[slot, 0]),
            tpu.make_async_copy(self.state_buffers.at[slot], self.state_output_hbm.at[rows],
                                self.write_semaphores.at[slot, 1]),
        )

    def prime(self):
        for index in range(min(2, self.chunks)):
            for copy in self._reads(index):
                copy.start()

    def acquire(self, index):
        if self.four_buffers and index == 0:
            for tail in range(2, self.chunks):
                for copy in self._reads(tail):
                    copy.start()
        for copy in self._reads(index):
            copy.wait()
        slot = index % self.buffer_count
        return self.history_buffers.at[slot], self.state_buffers.at[slot]

    def release(self, index):
        for copy in self._writes(index):
            copy.start()
        self.pending_writes.append(index)
        if self.four_buffers:
            return
        ahead = index + 2
        if ahead < self.chunks:
            if ahead >= 3:
                # Buffer ahead % 3 last held chunk ahead - 3.
                for copy in self._writes(ahead - 3):
                    copy.wait()
                self.pending_writes.remove(ahead - 3)
            for copy in self._reads(ahead):
                copy.start()

    def finish(self):
        for index in list(self.pending_writes):
            for copy in self._writes(index):
                copy.wait()
        self.pending_writes.clear()

    # Traced-index variants for a rolled chunk loop: chunk c's writes are
    # drained when chunk c + 3 is read into its buffer, the last three in
    # finish_rolled.
    def acquire_rolled(self, index):
        for copy in self._reads(index):
            copy.wait()
        slot = index % 3
        return self.history_buffers.at[slot], self.state_buffers.at[slot]

    def release_rolled(self, index):
        for copy in self._writes(index):
            copy.start()
        ahead = index + 2

        @pl.when(ahead < self.chunks)
        def refill():
            @pl.when(ahead >= 3)
            def drain():
                for copy in self._writes(ahead - 3):
                    copy.wait()

            for copy in self._reads(ahead):
                copy.start()

    def finish_rolled(self):
        for index in range(max(0, self.chunks - 3), self.chunks):
            for copy in self._writes(index):
                copy.wait()


class KdaStateStreamSlice:
    """One head shard's rows in a shared, flattened KDA state stream.

    The caller primes the parent once and drains it after all head shards. Local
    KDA calls use shard-local chunk indices and finish calls, while the
    parent keeps its lookahead and buffer ownership across shard boundaries.
    """

    def __init__(self, parent, *, first_chunk: int, chunks: int):
        if first_chunk < 0 or chunks < 1 or first_chunk + chunks > parent.chunks:
            raise ValueError("A state-stream slice must fit within its parent")
        self.parent = parent
        self.first_chunk, self.chunks = first_chunk, chunks
        self.chunk, self.slots = parent.chunk, parent.slots

    def acquire(self, index):
        with _phase("t1_kda_state_read_wait"):
            return self.parent.acquire(self.first_chunk + index)

    def release(self, index):
        self.parent.release(self.first_chunk + index)

    def acquire_rolled(self, index):
        with _phase("t1_kda_state_read_wait"):
            return self.parent.acquire_rolled(self.first_chunk + index)

    def release_rolled(self, index):
        self.parent.release_rolled(self.first_chunk + index)

    def finish(self):
        # The next shard can already have reads in flight in these buffers.
        # Only the owner of the flattened stream can drain its final writes.
        pass

    def finish_rolled(self):
        pass


def _shard_dot(stream, x, shard, guard):
    """``x`` times one held shard's columns of a column stream (FP32): the
    shard's chunks in order, concatenated."""
    per_shard = stream.chunks // stream.shards
    parts = [
        stream.dot_position(x, shard * per_shard + index, guard)
        for index in range(per_shard)
    ]
    return parts[0] if per_shard == 1 else jnp.concatenate(parts, axis=1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class KimiDeltaAttentionWeights:
    """KDA parameters and their persistent projection-copy resources."""

    gate_projection: TensorLike  # [hidden_size, 128]
    gate_projection_vmem: Any  # [hidden_size, 128]
    input_projection: TensorLike  # [hidden_size, 1664]
    input_projection_vmem: Any  # [hidden_size, 1664]
    feature_gate_projection: TensorLike  # [128, 384]
    feature_gate_projection_vmem: Any  # [128, 384]
    # [384, hidden_size], or [768, hidden_size / 2] after paired-rank repartitioning.
    output_projection: TensorLike
    output_projection_vmem: Any  # [384, hidden_size] or [768, hidden_size / 2]
    convolution_filter: TensorLike  # [4, 3, 3, 128]
    exp_log_decay_rate: TensorLike  # [3, 1]
    time_step_bias: TensorLike  # [3, 128]
    output_normalization: TensorLike  # [128] or [8, 128]
    projection_copy_semaphores: Any  # [4]

    def make_projection_copies(self) -> SimpleNamespace:
        """Create the asynchronous projection-weight copies."""
        return SimpleNamespace(
            gate=tpu.make_async_copy(
                self.gate_projection,
                self.gate_projection_vmem,
                self.projection_copy_semaphores.at[0],
            ),
            feature_gate=tpu.make_async_copy(
                self.feature_gate_projection,
                self.feature_gate_projection_vmem,
                self.projection_copy_semaphores.at[1],
            ),
            input=tpu.make_async_copy(
                self.input_projection,
                self.input_projection_vmem,
                self.projection_copy_semaphores.at[2],
            ),
            output=tpu.make_async_copy(
                self.output_projection,
                self.output_projection_vmem,
                self.projection_copy_semaphores.at[3],
            ),
        )


def _kda_row_chunk(batch_size: int, attention_dp: int = 1) -> int:
    """Rows per KDA recurrence chunk above B8 (smaller batches run whole).
    TP32 B128 and above use four: eight-row chunks exceed the compiler's VMEM
    although _vmem_budget accepts them. Attention DP2's B128 peaks outside KDA
    and fits eight."""
    return 8 if batch_size < 128 or attention_dp == 2 and batch_size == 128 else 4


def kda(
    hidden_ref: TensorLike,  # [batch_size, hidden_size]
    output_ref: TensorLike,  # [batch_size, hidden_size]
    updated_convolution_history_ref: TensorLike,  # [batch_size, 3, 3, 4, 128]
    updated_recurrent_state_ref: TensorLike,  # [batch_size, 3, 128, 128]
    *,
    weights: KimiDeltaAttentionWeights,
    convolution_history_ref: TensorLike,  # [batch_size, 3, 3, 4, 128]
    recurrent_state_ref: TensorLike,  # [batch_size, 3, 128, 128]
    communication_workspace: AttentionCommunicationWorkspace,
    epsilon: float = 1e-5,
    decay_lower_bound: float = -5.0,
    synchronize_devices: bool = True,
    weights_are_prefetched: bool = False,
    token_scatter_output: bool = False,
    sequence_rows: int = 1,
    wait_for_state=None,
    on_state_updated=None,
    streams: dict | None = None,
    shard_index: int | None = None,
    attended_out=None,
    gate_features_vmem=None,
    state_stream=None,
    stage_projections: bool = False,
    projected_rows=None,
    first_call: bool | None = None,
) -> None:
    """Execute one Kimi Delta Attention layer inside a Pallas program.

    ``state_stream`` (above eight rows; see KdaStateStream) supplies each
    recurrence chunk's state from HBM instead of whole-batch state refs.

    ``streams`` optionally maps "input" and "output" to ChunkedWeightStreams
    that own the copies of those projections.

    Attention DP runs one call per held TP32 shard (``shard_index``);
    ``attended_out`` ([batch, 384]) receives its gated features for the
    caller's output projection, and ``gate_features_vmem`` (at least
    [batch, 128] BF16) shares the gate features between shards.
    ``stage_projections`` stages a DP shard's projections above eight rows
    (rolled over a state stream's tiles). ``projected_rows`` ([batch, 1664]
    BF16) is the caller's input projection for the DP2 B2/B4 shard swap, which
    passes a traced ``shard_index`` and ``first_call``.
    """
    streams = streams or {}
    first_shard = shard_index in (None, 0) if first_call is None else first_call
    traced_shard = shard_index is not None and not isinstance(shard_index, int)
    if (shard_index is None) != (attended_out is None):
        raise ValueError("Per-shard KDA calls must take attended_out")
    batch_size, hidden_size = hidden_ref.shape
    head_count, channels_per_head = 3, 128
    state_batch = (convolution_history_ref.shape[0]
                   if state_stream is not None and state_stream.slots is not None
                   else batch_size)
    state_shape = (
        state_batch,
        head_count,
        channels_per_head,
        channels_per_head,
    )
    history_shape = (state_batch, 3, 3, CONV_HEAD_ROWS, channels_per_head)
    if (
        output_ref.shape != hidden_ref.shape
        or weights.gate_projection.shape != (hidden_size, 128)
        or weights.input_projection.shape[0] != hidden_size
        or weights.input_projection.shape[1] % 1664
        or (shard_index is None and weights.input_projection.shape[1] != 1664)
        or weights.exp_log_decay_rate.shape != (3, 1)
        or convolution_history_ref.shape != history_shape
        or recurrent_state_ref.shape != state_shape
    ):
        raise ValueError(
            "Kimi Delta Attention inputs and state must match the input batch"
        )
    if batch_size % sequence_rows:
        raise ValueError("sequence_rows must divide the KDA batch")
    if synchronize_devices:
        collectives32.barrier()
        communication_workspace.reduction_phase[0] = 0

    # Large DP batches keep each shard's projections in VMEM and reload only
    # a recurrence tile, instead of keeping projected/decay vectors live across
    # every state-stream region. DP1 stages only with KernelConfig.kda_staged.
    staged = (
        (stage_projections or (KDA_STAGED and shard_index is None))
        and batch_size > 8 and sequence_rows == 1
    )
    projection_width = 1664 if shard_index is not None else weights.input_projection.shape[1]
    rolled = staged and state_stream is not None
    row_chunk = state_stream.chunk if state_stream is not None else _kda_row_chunk(batch_size)
    chunk_count = batch_size // row_chunk

    def run(reduction_vmem, attended_vmem=None, projected_vmem=None, gate_vmem=None,
            output_gate_vmem=None):
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
            gate_features = _dot(
                hidden_ref[...], weights.gate_projection_vmem[...]
            ).astype(jnp.bfloat16)
            if gate_features_vmem is not None:
                gate_features_vmem[:, :128] = gate_features
        else:
            gate_features = gate_features_vmem[:, :128]

        if first_shard:
            weight_copies.feature_gate.wait()
        feature_gate_vmem = weights.feature_gate_projection_vmem
        if shard_index is not None and not traced_shard:
            feature_gate_vmem = feature_gate_vmem.at[
                :, pl.ds(shard_index * attention_width, attention_width)
            ]
        if rolled:
            gate_vmem[...] = (
                _dot(gate_features, feature_gate_vmem[...])
                .astype(jnp.bfloat16)
                .reshape(chunk_count, row_chunk, -1)
            )
        elif staged:
            gate_vmem[...] = _dot(gate_features, feature_gate_vmem[...]).astype(jnp.bfloat16)
        else:
            raw_recurrent_gate = _dot(gate_features, feature_gate_vmem[...])
            if traced_shard:
                raw_recurrent_gate = jnp.where(
                    shard_index == 0,
                    raw_recurrent_gate[:, :attention_width],
                    raw_recurrent_gate[:, attention_width:],
                )
            raw_recurrent_gate = (
                raw_recurrent_gate
                .astype(jnp.bfloat16)
                .reshape(batch_size, head_count, channels_per_head)
                .astype(jnp.float32)
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
            projected = streams["input"].dot(
                hidden_ref[...],
                communication_workspace.reduction_phase[0] >= 0,
            ).astype(jnp.bfloat16)
        else:
            weight_copies.input.wait()
            projected = _dot(
                hidden_ref[...], weights.input_projection_vmem[...]
            ).astype(jnp.bfloat16)
        if rolled:
            projected_vmem[...] = projected.reshape(chunk_count, row_chunk, -1)
            output_gate_vmem[...] = projected[:, 3 * attention_width : 4 * attention_width]
            del projected

            queries_keys_values = _RowFunction(
                lambda index: projected_vmem[index][:, : 3 * attention_width].reshape(
                    row_chunk, 3, head_count, channels_per_head
                )
            )
            update_rate = _RowFunction(
                lambda index: projected_vmem[index][
                    :, 4 * attention_width : 4 * attention_width + head_count
                ].reshape(row_chunk, head_count, 1).astype(jnp.float32)
            )
            decay = _RowFunction(
                lambda index: decay_lower_bound * jax.nn.sigmoid(
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
            del projected

            def chunk_rows(rows):
                return rows.stop - rows.start

            queries_keys_values = _RowFunction(
                lambda rows: projected_vmem[rows, : 3 * attention_width].reshape(
                    chunk_rows(rows), 3, head_count, channels_per_head
                )
            )
            update_rate = _RowFunction(
                lambda rows: projected_vmem[
                    rows, 4 * attention_width : 4 * attention_width + head_count
                ].reshape(chunk_rows(rows), head_count, 1).astype(jnp.float32)
            )
            decay = _RowFunction(
                lambda rows: decay_lower_bound * jax.nn.sigmoid(
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
        if sequence_rows == 1:
            def recurrence(rows, history_ref=None, state_ref=None):
                # ``history_ref``/``state_ref``: a streamed chunk's buffers
                # (updated in place), else the whole-batch state refs.
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
                query *= (
                    jax.lax.rsqrt(jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
                    * channels_per_head**-0.5
                )
                key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
                next_recurrent_state = recurrent_state * jnp.exp(decay[rows])[..., None]
                predicted_value = jnp.sum(key[..., None] * next_recurrent_state, axis=2)
                state_update = jax.nn.sigmoid(update_rate[rows]) * (value - predicted_value)
                next_recurrent_state += key[..., None] * state_update[..., None, :]
                attended_rows = jnp.sum(query[..., None] * next_recurrent_state, axis=2)
                if history_ref is None:
                    updated_convolution_history_ref[rows, :, :, :head_count] = convolution_window[:, 1:]
                    updated_recurrent_state_ref[rows] = next_recurrent_state.astype(updated_recurrent_state_ref.dtype)
                else:
                    history_ref[:, :, :, :head_count] = convolution_window[:, 1:]
                    state_ref[...] = next_recurrent_state.astype(state_ref.dtype)
                return attended_rows

            _phase_end(kda_scope.pop())
            kda_scope.append(_phase_start("q4_kda_recurrence"))
            if batch_size <= 8:
                if state_stream is None:
                    attended = recurrence(slice(None))
                else:
                    # DP4 B32 has one eight-row state chunk per head shard.
                    # Its refs are HBM refs, so consume the prefetched stream
                    # even though the recurrence itself needs no row tiling.
                    assert state_stream.chunks == 1 and state_stream.chunk == batch_size
                    history_chunk, state_chunk = state_stream.acquire(0)
                    attended = recurrence(slice(None), history_chunk, state_chunk)
                    state_stream.release(0)
                    state_stream.finish()
            else:
                # Above eight rows the whole-batch state value (3 MB at B16)
                # spills; run row chunks in separate regions so only one
                # chunk's state is live.
                chunk = row_chunk
                if rolled:
                    @pl.loop(0, chunk_count)
                    def run_rolled_chunk(index):
                        history_chunk, state_chunk = state_stream.acquire_rolled(index)
                        attended_vmem[pl.ds(pl.multiple_of(index * chunk, chunk), chunk)] = (
                            recurrence(index, history_chunk, state_chunk)
                        )
                        state_stream.release_rolled(index)

                    state_stream.finish_rolled()
                for index, first in (() if rolled else enumerate(range(0, batch_size, chunk))):
                    # A runtime condition (never false) so the chunk is its own region.
                    @pl.when(communication_workspace.reduction_phase[0] >= 0)
                    def run_chunk(index=index, first=first):
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
                if state_stream is not None and not rolled and batch_size != 32:
                    state_stream.finish()
                attended = attended_vmem[...]
        else:
            convolution_filter = weights.convolution_filter[...].astype(jnp.float32)

            def run_sequence(first, history, state, store):
                """The rows of the sequence at ``first`` from its input conv
                ``history`` and recurrent ``state``; ``store(offset, history,
                state)`` writes row ``first + offset``'s post-states."""
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
                    query *= (
                        jax.lax.rsqrt(
                            jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6
                        )
                        * channels_per_head**-0.5
                    )
                    key *= jax.lax.rsqrt(
                        jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6
                    )
                    state = state * jnp.exp(decay[row])[..., None]
                    predicted_value = jnp.sum(key[..., None] * state, axis=1)
                    state_update = jax.nn.sigmoid(update_rate[row]) * (
                        value - predicted_value
                    )
                    state = state + key[..., None] * state_update[:, None, :]
                    rows.append(jnp.sum(query[..., None] * state, axis=1)[None])
                    # Match separate decode steps: the current output uses
                    # FP32 updated state, but the next token sees stored state.
                    saved_state = state.astype(recurrent_state_ref.dtype)
                    store(offset, window[offset + 1 : offset + 4], saved_state)
                    state = saved_state.astype(jnp.float32)
                return rows

            if state_stream is None:
                attended_rows = []
                for first in range(0, batch_size, sequence_rows):
                    def store(offset, history, state, first=first):
                        updated_convolution_history_ref[first + offset, :, :, :head_count] = history
                        updated_recurrent_state_ref[first + offset] = state

                    attended_rows += run_sequence(
                        first,
                        convolution_history_ref[first, :, :, :head_count],
                        recurrent_state_ref[first],
                        store,
                    )
                attended = jnp.concatenate(attended_rows, axis=0)
            else:
                # Streamed states (above 16 rows): each row chunk holds whole
                # sequences, each starting from its first row's state; every
                # row's post-states go into the chunk buffers, written back in
                # place on release.
                chunk = state_stream.chunk
                if chunk % sequence_rows:
                    raise ValueError("Streamed KDA chunks must hold whole sequences")
                for index, base in enumerate(range(0, batch_size, chunk)):
                    # A runtime condition (never false) so the chunk is its own region.
                    @pl.when(communication_workspace.reduction_phase[0] >= 0)
                    def run_sequence_chunk(index=index, base=base):
                        history_chunk, state_chunk = state_stream.acquire(index)
                        rows = []
                        for local in range(0, chunk, sequence_rows):
                            def store(offset, history, state, local=local):
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
                if batch_size != 32:  # B32 drains after the output work (below)
                    state_stream.finish()
                attended = attended_vmem[...]
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
            output_gate = output_gate_vmem[...].reshape(batch_size, head_count, channels_per_head)
        elif staged:
            output_gate = projected_vmem[
                :, 3 * attention_width : 4 * attention_width
            ].reshape(batch_size, head_count, channels_per_head)
        attended = (
            (attended * normalization).astype(jnp.bfloat16)
            * jax.nn.sigmoid(output_gate.astype(jnp.float32)).astype(jnp.bfloat16)
        )
        if on_state_updated is not None:
            on_state_updated()
        _phase_end(kda_scope.pop())

        if attended_out is not None:
            attended_out[...] = attended.reshape(batch_size, attention_width)
            return
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

    pl.run_scoped(
        run,
        tpu.VMEM((batch_size, 3584 if batch_size > 32 else 8192), jnp.float32),
        *(
            (tpu.VMEM((batch_size, head_count, channels_per_head), jnp.float32),)
            if batch_size > 8 and (sequence_rows == 1 or state_stream is not None)
            else ()
        ),
        *(
            (
                tpu.VMEM((chunk_count, row_chunk, projection_width), jnp.bfloat16),
                tpu.VMEM((chunk_count, row_chunk, head_count * channels_per_head), jnp.bfloat16),
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
    if state_stream is not None and not rolled and batch_size == 32:
        state_stream.finish()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class MultiHeadLatentAttentionWeights:
    """MLA parameters and their persistent projection-copy resources."""

    query_down_projection: TensorLike  # [hidden_size, 1536]
    query_down_projection_vmem: Any  # [hidden_size, 1536]
    key_value_down_projection: TensorLike  # [hidden_size, 640]
    key_value_down_projection_vmem: Any  # [hidden_size, 640]
    query_up_projection: TensorLike  # [1536, 768]
    query_up_projection_vmem: Any  # [1536, 768]
    key_value_up_projection: TensorLike  # [512, 768]
    key_value_up_projection_vmem: Any  # [512, 768]
    output_gate_projection: TensorLike  # [hidden_size, 384]
    output_gate_projection_vmem: Any  # [hidden_size, 384]
    # [384, hidden_size], or [768, hidden_size / 2] after paired-rank repartitioning.
    output_projection: TensorLike
    output_projection_vmem: Any  # [384, hidden_size] or [768, hidden_size / 2]
    query_normalization: TensorLike  # [1, 1536]
    key_value_normalization: TensorLike  # [1, 512]
    projection_copy_semaphores: Any  # [6]

    def make_projection_copies(self) -> SimpleNamespace:
        """Create the asynchronous projection-weight copies."""
        return SimpleNamespace(
            query_down=tpu.make_async_copy(
                self.query_down_projection,
                self.query_down_projection_vmem,
                self.projection_copy_semaphores.at[0],
            ),
            key_value_down=tpu.make_async_copy(
                self.key_value_down_projection,
                self.key_value_down_projection_vmem,
                self.projection_copy_semaphores.at[1],
            ),
            query_up=tpu.make_async_copy(
                self.query_up_projection,
                self.query_up_projection_vmem,
                self.projection_copy_semaphores.at[2],
            ),
            key_value_up=tpu.make_async_copy(
                self.key_value_up_projection,
                self.key_value_up_projection_vmem,
                self.projection_copy_semaphores.at[3],
            ),
            output_gate=tpu.make_async_copy(
                self.output_gate_projection,
                self.output_gate_projection_vmem,
                self.projection_copy_semaphores.at[4],
            ),
            output=tpu.make_async_copy(
                self.output_projection,
                self.output_projection_vmem,
                self.projection_copy_semaphores.at[5],
            ),
        )


PAGE_TOKENS = 256  # tokens per paged MLA cache page (two 128-token read tiles)
MLA_LAYERS = 24


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class MultiHeadLatentAttentionWorkspace:
    """Persistent VMEM and semaphores used by one MLA invocation."""

    # Unused placeholders ([8, 128] and one DMA semaphore): the DCP attention
    # keeps its pages in mla_dcp's workspace.
    key_cache_vmem: Any
    value_cache_vmem: Any
    cache_copy_semaphores: Any
    query_vmem: Any  # [batch_size, 3, 256]
    output_gate_values_vmem: Any  # [batch_size, 384]
    new_key_vmem: Any  # [batch_size, 3, 256]
    new_value_vmem: Any  # [batch_size, 3, 128]
    maximum_vmem: Any  # [batch_size, 3, 1]
    denominator_vmem: Any  # [batch_size, 3, 1]
    accumulator_vmem: Any  # [batch_size, 3, 128]


def mla(
    position_ref: TensorLike,  # [batch_size]
    hidden_ref: TensorLike,  # [batch_size, hidden_size]
    output_ref: TensorLike,  # [batch_size, hidden_size]
    new_key_ref: TensorLike,  # [batch_size, padded_head_count, 256]
    new_value_ref: TensorLike,  # [batch_size, padded_head_count, 128]
    *,
    weights: MultiHeadLatentAttentionWeights,
    workspace: MultiHeadLatentAttentionWorkspace,
    communication_workspace: AttentionCommunicationWorkspace,
    dcp,
    epsilon: float = 1e-5,
    synchronize_devices: bool = True,
    weights_are_prefetched: bool = False,
    token_scatter_output: bool = False,
    sequence_rows: int = 1,
    output_projection_after_query_down: bool = False,
    streams: dict | None = None,
) -> None:
    """Execute one multi-head latent-attention layer inside a Pallas program.

    ``dcp`` (``(mla_dcp.DcpCache, mla_dcp.DcpWorkspace)``) is the cache: decode
    context parallel attention (see ``mla_dcp``) over this rank's token shard
    of the latent pages in vLLM's unified pools, attended in latent space for
    the whole DCP group's heads. The new token's latent row ``[512 normalized
    | 64 rope | 64 zero]`` is handed out through ``new_key_ref`` rows 0..2
    (lanes 0:256, 256:512, then 512:640 and 128 zero lanes) for the commit;
    ``new_value_ref`` is zero.

    ``streams`` optionally maps "query_up", "key_value_up", "output_gate" and
    "output" to ChunkedWeightStreams that own those projections' copies.

    Attention DP runs :func:`mla_dcp_shards` instead.

    Consecutive groups of ``sequence_rows`` rows belong to one sequence and
    share one cache entry. Within a group, rows represent consecutive tokens
    and attend causally to earlier rows from the same invocation.
    """
    batch_size, hidden_size = hidden_ref.shape
    context_length = dcp[0].page_tokens * dcp[0].pages * (1 if dcp[0].global_pages else dcp[0].group)
    if not 1 <= batch_size <= MAX_BATCH:
        raise ValueError(f"MLA supports 1 to {MAX_BATCH} batch rows")
    if output_ref.shape != hidden_ref.shape:
        raise ValueError("MLA output must match the hidden-state shape")
    if not 1 <= sequence_rows <= batch_size or batch_size % sequence_rows:
        raise ValueError("sequence_rows must divide the MLA batch")
    if position_ref.shape != (batch_size,):
        raise ValueError("Positions must match the MLA batch size")
    if (
        weights.query_down_projection.shape != (hidden_size, 1536)
        or weights.key_value_down_projection.shape != (hidden_size, 640)
        or weights.query_up_projection.shape != (1536, 768)
        or weights.key_value_up_projection.shape != (512, 768)
        or weights.output_gate_projection.shape != (hidden_size, 384)
        or weights.query_normalization.shape != (1, 1536)
        or weights.key_value_normalization.shape != (1, 512)
        or context_length < 384
        or context_length % 128
    ):
        raise ValueError("MLA projection and normalization shapes are invalid")
    expected_workspace_shapes = (
        (workspace.query_vmem.shape, (batch_size, 3, 256)),
        (workspace.output_gate_values_vmem.shape, (batch_size, 384)),
        (workspace.new_key_vmem.shape, (batch_size, 3, 256)),
        (workspace.new_value_vmem.shape, (batch_size, 3, 128)),
        (workspace.maximum_vmem.shape, (batch_size, 3, 1)),
        (workspace.denominator_vmem.shape, (batch_size, 3, 1)),
        (workspace.accumulator_vmem.shape, (batch_size, 3, 128)),
    )
    if any(actual != expected for actual, expected in expected_workspace_shapes):
        raise ValueError("MLA workspace has an invalid shape")
    if (
        new_key_ref.shape[0] != batch_size
        or new_key_ref.shape[1] < 3
        or new_key_ref.shape[2:] != (256,)
        or new_value_ref.shape[0] != batch_size
        or new_value_ref.shape[1] < 3
        or new_value_ref.shape[2:] != (128,)
    ):
        raise ValueError("MLA key/value outputs have invalid shapes")
    if synchronize_devices:
        collectives32.barrier()
        communication_workspace.reduction_phase[0] = 0

    streams = streams or {}
    weight_copies = weights.make_projection_copies()
    mla_scope = [_phase_start("q6_mla_projections")]

    def start_copy(name):
        if name in streams:
            streams[name].start()
        else:
            getattr(weight_copies, name).start()

    def projection(name, x):
        """``x`` times a projection in FP32, through its stream if it has one."""
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

    # Diagnostic (KernelConfig.dcp_skip): skip the DCP query gather, local
    # attention and/or partial combine, e.g. "gather,combine". Results are
    # invalid; timing only.
    dcp_skip = _config().dcp_skip
    dcp_attention = mla_dcp.DcpAttention(*dcp, position_ref, batch_size, sequence_rows=sequence_rows,
                                         skip=dcp_skip,
                                         cache_verify_rows=sequence_rows > 1 and dcp[0].dtype == jnp.float8_e4m3fn)
    dcp_attention.prime()

    if "query_down" in streams:
        query_latent = streams["query_down"].dot(
            hidden_ref[...], communication_workspace.reduction_phase[0] >= 0
        ).astype(jnp.bfloat16).astype(jnp.float32)
    else:
        weight_copies.query_down.wait()
        query_latent = _dot(
            hidden_ref[...], weights.query_down_projection_vmem[...]
        ).astype(jnp.bfloat16).astype(jnp.float32)
    if output_projection_after_query_down:
        # The pool views are reinterpret casts, so the compiler need not see
        # that this copy aliases the q-down loads above; a region (on a runtime
        # condition that is never false) keeps it from being hoisted over them.
        @pl.when(communication_workspace.reduction_phase[0] >= 0)
        def start_output_projection_copy():
            start_copy("output")
    query_latent = (
        query_latent
        * jax.lax.rsqrt(
            jnp.mean(query_latent * query_latent, axis=1, keepdims=True)
            + epsilon
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
            jnp.mean(
                key_value_latent * key_value_latent,
                axis=1,
                keepdims=True,
            )
            + epsilon
        )
        * weights.key_value_normalization[...]
    ).astype(jnp.bfloat16)
    workspace.query_vmem[...] = projection("query_up", query_latent).astype(
        jnp.bfloat16
    ).reshape(batch_size, 3, 256)

    # The new token's latent row, handed out for the owner's commit.
    new_latent = jnp.concatenate(
        (key_value_latent, compressed[:, 512:576], jnp.zeros((batch_size, 64), jnp.bfloat16)),
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
    workspace.maximum_vmem[...] = jnp.full(
        (batch_size, 3, 1), -jnp.inf, jnp.float32
    )
    workspace.denominator_vmem[...] = jnp.zeros(
        (batch_size, 3, 1), jnp.float32
    )
    workspace.accumulator_vmem[...] = jnp.zeros(
        (batch_size, 3, 128), jnp.float32
    )

    def attend_dcp():
        """DCP attention: W_UK absorbed into this rank's queries (score scale
        folded in), latent-space attention over the group's token shards
        (``mla_dcp.DcpAttention``, the output gate projected while the query
        gather is in flight), then W_UV per head. The heads' outputs land in
        the accumulator with a unit denominator."""
        if "key_value_up" in streams:
            streams["key_value_up"].wait_all()
        else:
            weight_copies.key_value_up.wait()
        # m_kb [512, 768]: head h's key columns h*256 : +128, value columns +128 : +256.
        up = weights.key_value_up_projection_vmem
        query = workspace.query_vmem[...]  # [B, 3, 256]: nope 128 | rope 64 | 64
        scale = 192**-0.5
        absorbed = []
        for head in range(3):
            latent_query = jax.lax.dot_general(
                query[:, head, :128], up[:, head * 256 : head * 256 + 128],
                (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
            )
            absorbed.append(jnp.concatenate(
                (latent_query, query[:, head, 128:192].astype(jnp.float32),
                 jnp.zeros((batch_size, 64), jnp.float32)),
                axis=1,
            ) * scale)

        def output_gate():
            workspace.output_gate_values_vmem[...] = projection(
                "output_gate", hidden_ref[...]
            ).astype(jnp.bfloat16)

        merged = dcp_attention.attend(
            jnp.stack(absorbed, axis=1), new_latent.astype(jnp.float32), dcp[0].layer % 2,
            between=output_gate,
        )
        outputs = []
        for head in range(3):
            value_up = up[:, head * 256 + 128 : head * 256 + 256]
            latent = merged[:, head]
            high = latent.astype(jnp.bfloat16)
            low = (latent - high.astype(jnp.float32)).astype(jnp.bfloat16)
            outputs.append(sum(
                jax.lax.dot_general(part, value_up, (((1,), (0,)), ((), ())),
                                    preferred_element_type=jnp.float32)
                for part in (high, low)
            ))
        workspace.accumulator_vmem[...] = jnp.stack(outputs, axis=1)
        workspace.denominator_vmem[...] = jnp.ones((batch_size, 3, 1), jnp.float32)

    # sequence_rows > 1: DcpAttention folds each sequence's rows in causally.
    attend_dcp()

    attention = (
        (workspace.accumulator_vmem[...] / workspace.denominator_vmem[...])
        .reshape(batch_size, 384)
        .astype(jnp.bfloat16)
        * jax.nn.sigmoid(
            workspace.output_gate_values_vmem[...].astype(jnp.float32)
        ).astype(jnp.bfloat16)
    )
    _phase_end(mla_scope.pop())
    output_stream = streams.get("output")
    if output_stream is None:
        weight_copies.output.wait()
    elif output_stream.chunks == 1:
        output_stream.wait_all()
        output_stream = None

    def project_output(reduction_vmem):
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

    pl.run_scoped(
        project_output,
        tpu.VMEM((batch_size, 3584 if batch_size > 32 else 8192), jnp.float32),
    )


def mla_dcp_shards(
    position_ref: TensorLike,  # [rows]
    hidden_ref: TensorLike,  # [rows, hidden_size]
    new_key_ref: TensorLike,  # [rows, padded_head_count, 256]
    new_value_ref: TensorLike,  # [rows, padded_head_count, 128]
    attended_out: TensorLike,  # [shards, rows, 384]
    *,
    weights: MultiHeadLatentAttentionWeights,
    streams: dict,
    communication_workspace: AttentionCommunicationWorkspace,
    dcp,
    epsilon: float = 1e-5,
    sequence_rows: int = 1,
) -> None:
    """One MLA layer of attention DP under DCP: the rank's held TP32 shards
    (3 heads each) attend in one ``mla_dcp.DcpAttention`` call over the DP
    group's latent pages, so a page is read once for every held head.
    ``sequence_rows`` as in :func:`mla`, over the group's rows.

    The q/kv latents are computed once; each shard's queries come from its
    chunk of the query-up stream and are absorbed with its W_UK columns (the
    key-value-up stream holds every shard); the output gates are projected
    while the query gather is in flight. W_UV and the gate leave each shard's
    gated attention in ``attended_out[shard]`` for the caller's output
    projection. The new latent row is handed out through ``new_key_ref``
    rows 0..2 as in :func:`mla`."""
    rows = hidden_ref.shape[0]
    shards = attended_out.shape[0]
    up_stream = streams["key_value_up"]
    if up_stream.slots is not None or up_stream.chunks != shards:
        raise ValueError("DCP attention DP needs every shard's key-value-up columns resident")
    weight_copies = weights.make_projection_copies()
    guard = communication_workspace.reduction_phase[0] >= 0

    def start_copy(name):
        if name in streams:
            streams[name].start()
        else:
            getattr(weight_copies, name).start()

    for name in ("query_up", "key_value_up", "output_gate"):
        start_copy(name)
    dcp_attention = mla_dcp.DcpAttention(*dcp, position_ref, rows, skip=_config().dcp_skip, heads=3 * shards,
                                         sequence_rows=sequence_rows,
                                         chunk_rows=_verify_chunk(rows, 3 * shards, sequence_rows))
    dcp_attention.prime()
    scope = _phase_start("q6_mla_projections")
    if "query_down" in streams:
        query_latent = streams["query_down"].dot(hidden_ref[...], guard)
    else:
        weight_copies.query_down.wait()
        query_latent = _dot(hidden_ref[...], weights.query_down_projection_vmem[...])
    query_latent = query_latent.astype(jnp.bfloat16).astype(jnp.float32)

    # The pooled output projection overwrites the q-down projection.
    @pl.when(guard)
    def start_output_projection_copy():
        start_copy("output")

    query_latent = (
        query_latent
        * jax.lax.rsqrt(jnp.mean(query_latent * query_latent, axis=1, keepdims=True) + epsilon)
        * weights.query_normalization[...]
    ).astype(jnp.bfloat16)
    weight_copies.key_value_down.wait()
    compressed = _dot(hidden_ref[...], weights.key_value_down_projection_vmem[...]).astype(jnp.bfloat16)
    key_value_latent = compressed[:, :512].astype(jnp.float32)
    key_value_latent = (
        key_value_latent
        * jax.lax.rsqrt(jnp.mean(key_value_latent * key_value_latent, axis=1, keepdims=True) + epsilon)
        * weights.key_value_normalization[...]
    ).astype(jnp.bfloat16)
    new_latent = jnp.concatenate(
        (key_value_latent, compressed[:, 512:576], jnp.zeros((rows, 64), jnp.bfloat16)), axis=1
    )
    new_key_ref[:, 0] = new_latent[:, :256]
    new_key_ref[:, 1] = new_latent[:, 256:512]
    new_key_ref[:, 2] = jnp.concatenate((new_latent[:, 512:], jnp.zeros((rows, 128), jnp.bfloat16)), axis=1)
    new_value_ref[:, :3] = jnp.zeros((rows, 3, 128), jnp.bfloat16)
    up_stream.wait_all()
    scale = 192**-0.5
    absorbed = []
    for shard in range(shards):
        query = _shard_dot(streams["query_up"], query_latent, shard, guard).astype(
            jnp.bfloat16
        ).reshape(rows, 3, 256)
        up = up_stream.view(shard)  # [512, 768]: head h's key columns h*256 : +128, value +128 : +256
        for head in range(3):
            latent_query = jax.lax.dot_general(
                query[:, head, :128], up[:, head * 256 : head * 256 + 128],
                (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
            )
            absorbed.append(jnp.concatenate(
                (latent_query, query[:, head, 128:192].astype(jnp.float32), jnp.zeros((rows, 64), jnp.float32)),
                axis=1,
            ) * scale)
    _phase_end(scope)
    scope = _phase_start("q7_mla_attend")

    def output_gates():
        for shard in range(shards):
            attended_out[shard] = _shard_dot(streams["output_gate"], hidden_ref[...], shard, guard).astype(
                jnp.bfloat16
            )

    merged = dcp_attention.attend(
        jnp.stack(absorbed, axis=1), new_latent.astype(jnp.float32), dcp[0].layer % 2, between=output_gates
    )
    for shard in range(shards):
        up = up_stream.view(shard)
        outputs = []
        for head in range(3):
            latent = merged[:, 3 * shard + head]
            high = latent.astype(jnp.bfloat16)
            low = (latent - high.astype(jnp.float32)).astype(jnp.bfloat16)
            outputs.append(sum(
                jax.lax.dot_general(part, up[:, head * 256 + 128 : head * 256 + 256],
                                    (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
                for part in (high, low)
            ))
        attended_out[shard] = jnp.stack(outputs, axis=1).reshape(rows, 384).astype(jnp.bfloat16) * jax.nn.sigmoid(
            attended_out[shard].astype(jnp.float32)
        ).astype(jnp.bfloat16)
    _phase_end(scope)


# At B4, scatter the projection rows and gather the completed output once.
# Keep all 16 BF16 rank-pair partials until the final FP32 sum: intermediate
# host sums can change long-context logits. The final gather has private
# persistent storage so adjacent phases cannot overwrite unread packets.
B4_ATTENTION_GATHER = _config().b4_attention_gather


def _project_attention_2d(
    attended, weight, out, comm, local, host_reduction, feature_backing,
    output_backing,
    sends, recvs, phase, dot, *, token_scatter_output=False, weight_stream=None,
    staging=None, gather_wire=None,
):
    """``weight_stream`` (optional) is a row ChunkedWeightStream of ``weight``
    whose positions 0 and 1 are this rank's and its pair peer's feature rows."""
    rank=jax.lax.axis_index('tp');slot=phase[0]%2
    batch_size=attended.shape[0]
    ordered_full_output = (
        B4_ATTENTION_GATHER and batch_size == 4 and not token_scatter_output
    )
    width=attended.shape[-1];g=weight.shape[0]//width
    assert g==2
    assert 1 <= batch_size <= MAX_BATCH and out.shape[0]==batch_size and comm.shape[0]==batch_size
    h=out.shape[-1];half=h//g;nr=32//g
    def compute(features,output):
        view=features.at[:,pl.ds((rank%g)*width,width)]
        view[...] = attended
        for offset in range(g-1,0,-1):
            tpu.make_async_remote_copy(view,view,sends.at[10+slot],recvs.at[10+slot],device_id=(rank^offset,),device_id_type=pl.DeviceIdType.MESH).start()
        comm[...] = jnp.zeros(comm.shape,jnp.float32)
        if weight_stream is None:
            own_weight = weight[pl.ds((rank%2)*width,width),:]
        else:
            weight_stream.wait(0)
            own_weight = weight_stream.view(0)[...]
        comm[:,:half] = dot(attended,own_weight)
        if weight_stream is not None:
            weight_stream.release(0, phase[0] >= 0)
        count=features.at[:,: (g-1)*width]
        tpu.make_async_copy(count,count,sends.at[10+slot]).wait()
        tpu.make_async_copy(count,count,recvs.at[10+slot]).wait()
        peer_lane=(rank%2)^1
        if weight_stream is None:
            peer_weight = weight[pl.ds(peer_lane*width,width),:]
        else:
            weight_stream.wait(1)
            peer_weight = weight_stream.view(1)[...]
        comm[:,:half] += dot(features[:,pl.ds(peer_lane*width,width)],peer_weight)
        if token_scatter_output or ordered_full_output:
            assert batch_size in (4, *SCATTER_BATCHES)
            tokens_per_pair = batch_size // 4
            payload_rows = tokens_per_pair * half // 128
            host = rank // 8
            pair = (rank % 8) // 2
            parity = rank % 2

            # Reuse attention/MoE communication storage after the input gather.
            # Keep each asynchronous role in a distinct physical lane. This
            # prevents the compiler from reusing aliased transformed refs
            # before every remote send has retired.
            assert staging is not None or 2 * local.shape[1] >= 4 * payload_rows, "wide reduction rows too small for the payloads"
            local_send = (local.at[0].bitcast(jnp.bfloat16) if staging is None else staging.at[0]).at[
                pl.ds(0, 4 * payload_rows), :
            ].reshape(4, payload_rows, 128)
            local_receive = (local.at[1].bitcast(jnp.bfloat16) if staging is None else staging.at[1]).at[
                pl.ds(0, 4 * payload_rows), :
            ].reshape(
                4, payload_rows, 128
            )
            # ``output`` is backed by a native 16-row allocation; only the
            # rows owned by this local pair are live during the reduction.
            owned_output = _bf16_rows_view(
                output_backing, tokens_per_pair, h
            )
            for target_pair in range(4):
                if tokens_per_pair == 1:
                    partial = comm[target_pair, :half][None, :]
                else:
                    partial = comm[
                        pl.ds(target_pair * tokens_per_pair, tokens_per_pair),
                        :half,
                    ]
                local_send[target_pair] = partial.astype(
                    jnp.bfloat16
                ).reshape(payload_rows, 128)
            local_receive[pair] = local_send[pair]
            local_copies = []
            for offset in range(3, 0, -1):
                target_pair = pair ^ offset
                copy = tpu.make_async_remote_copy(
                    local_send.at[target_pair],
                    local_receive.at[pair],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ (2 * offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                local_copies.append(copy)
            for copy in local_copies:
                copy.wait()

            if ordered_full_output:
                # Host h owns rank-pair lanes 4*h .. 4*h+3. Transport those
                # unreduced partials without changing their reduction order.
                host_wire_rows = 4 * payload_rows
                assert host_wire_rows <= 2 * host_reduction.shape[1]
                host_receive = host_reduction.bitcast(jnp.bfloat16).at[
                    :, pl.ds(0, host_wire_rows), :
                ]
                host_receive[host] = local_receive[...].reshape(host_wire_rows, 128)
            else:
                local_sum = jnp.sum(
                    local_receive[...].astype(jnp.float32), axis=0
                )
                # The caller supplies the four-lane FP32 host workspace
                # used by MoE after attention.
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
                    device_id=(rank ^ (8 * offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                host_copies.append(copy)
            for copy in host_copies:
                copy.wait()

            if ordered_full_output:
                rank_partials = host_receive[...].reshape(
                    16, payload_rows, 128
                ).astype(jnp.float32)
                reduced_half = jnp.sum(rank_partials, axis=0).astype(jnp.bfloat16)
            else:
                reduced_half = jnp.sum(host_receive[...], axis=0).astype(
                    jnp.bfloat16
                )
            pair_send = local.at[2].bitcast(jnp.bfloat16).at[
                pl.ds(0, payload_rows), :
            ]
            pair_receive = local.at[3].bitcast(jnp.bfloat16).at[
                pl.ds(0, payload_rows), :
            ]
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
            owned_output[:, :half] = jnp.where(
                parity == 0, sent_half, received_half
            )
            owned_output[:, half:] = jnp.where(
                parity == 0, received_half, sent_half
            )
            if ordered_full_output:
                # Full-batch rows for residual mixing and MoE; the packets are
                # private since peers may be a phase apart.
                assert gather_wire is not None
                _gather_local_pair_rows(
                    owned_output, out, sends, recvs, gather_wire, owners=4
                )
        else:
            rows=((half+255)//256)*2;payload=rows//2
            batched_payload=batch_size*payload
            receive=local.reshape(nr,local.shape[1]*8//nr,128)
            assert batched_payload<=receive.shape[1]
            bits=tpu.bitcast(comm[:,:rows*128].reshape(batch_size*rows,128).astype(jnp.bfloat16),jnp.uint32)
            receive[rank//g,:batched_payload,:] = jax.lax.bitcast_convert_type(bits,jnp.float32)
            view=receive.at[rank//g,:batched_payload,:]
            for offset in range(nr-1,0,-1):
                tpu.make_async_remote_copy(view,view,sends.at[14+slot],recvs.at[14+slot],device_id=(rank^(g*offset),),device_id_type=pl.DeviceIdType.MESH).start()
            count=receive.at[:nr-1,:batched_payload,:]
            tpu.make_async_copy(count,count,sends.at[14+slot]).wait()
            tpu.make_async_copy(count,count,recvs.at[14+slot]).wait()
            values=tpu.bitcast(jax.lax.bitcast_convert_type(receive[:,:batched_payload,:],jnp.uint32),jnp.bfloat16).astype(jnp.float32)
            comm[:,:rows*128] = jnp.sum(values,axis=0).reshape(batch_size,rows*128)
            view=output.at[:,pl.ds((rank%g)*half,half)]
            view[...] = comm[:,:half].astype(jnp.bfloat16)
            for offset in range(g-1,0,-1):
                tpu.make_async_remote_copy(view,view,sends.at[12+slot],recvs.at[12+slot],device_id=(rank^offset,),device_id_type=pl.DeviceIdType.MESH).start()
            count=output.at[:,: (g-1)*half]
            tpu.make_async_copy(count,count,sends.at[12+slot]).wait()
            tpu.make_async_copy(count,count,recvs.at[12+slot]).wait()
            out[...] = output[...]
        phase[0]+=1
    # Persistent addresses cannot alias preceding MoE temporaries on a slow peer.
    # Reserve 16x1024 BF16 for features and 16x7168 for outputs, including
    # native row padding. Total physical storage is exactly 256 KiB.
    assert g*width<=1024 and h==7168
    feature_store=_bf16_rows_view(feature_backing,batch_size,1024)
    output_store=_bf16_rows_view(output_backing,batch_size,7168)
    compute(feature_store.at[:,:g*width],output_store)

def _project_attention_dp(
    attended, weight, out, comm, local, host_reduction, feature_backing,
    sends, recvs, phase, dot, *, weight_stream=None, staging=None,
    keep_dp_rows=False, owners=4,
):
    """Project DP2/TP16 or DP4/TP8 attention back to MoE row ownership.

    ``attended`` [dp, 4r, 384] holds each logical TP32 head shard for the
    DP group's rows. The projection first reduces over four local rank
    pairs; DP2 also reduces over its second host. Rank pairs assemble the
    hidden dimension, then DP groups gather their disjoint row subsets so
    ``out`` [dp*r, 7168] has the pair-owned rows in sequence order.
    With one row owner (DP2 B2/B4, ``attended`` [2, r, 384]) each pair's
    partial of its group's r rows goes to the 15 same-lane ranks of both
    groups in one hop, so ``out`` [2r, 7168] is the whole batch on every rank.
    Output weight chunks are all own-lane shard halves, then peer halves.
    """
    rank = jax.lax.axis_index("tp")
    slot = phase[0] % 2
    shards, rows, width = attended.shape
    hidden = out.shape[-1]
    half = hidden // 2
    assert shards in (2, 4) and owners in (1, 4) and rows % owners == 0
    assert owners == 4 or shards == 2
    lane, pair, host = rank % 2, (rank % 8) // 2, rank // 8
    rows_per_pair = rows // owners
    assert weight_stream is not None or weight.shape == (shards * 2 * width, half)
    guard = phase[0] >= 0

    def weight_rows(position):
        shard, peer = position % shards, position // shards
        if weight_stream is not None:
            weight_stream.wait(position)
            return weight_stream.view(position)[...]
        return weight[pl.ds((shard * 2 + (lane ^ peer)) * width, width), :]

    features = _bf16_rows_view(feature_backing, rows, feature_backing.shape[-1])
    assert features.shape[-1] >= 2 * width * shards
    for shard in range(shards):
        view = features.at[:, pl.ds(shard * 2 * width + lane * width, width)]
        view[...] = attended[shard]
        tpu.make_async_remote_copy(
            view, view, sends.at[10 + slot], recvs.at[10 + slot],
            device_id=(rank ^ 1,), device_id_type=pl.DeviceIdType.MESH,
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
    if owners == 1:
        # Every rank needs both groups' sums: this pair's partial goes to the
        # 15 same-lane ranks in one hop; each rank adds the pairs host by
        # host, then each group's two hosts.
        host_slots = host_reduction.bitcast(jnp.bfloat16)

        def pair_partials(source_host):
            return host_slots.at[source_host].at[pl.ds(0, 4 * payload_rows), :].reshape(
                4, payload_rows, 128
            )

        own_partial = pair_partials(host).at[pair]
        own_partial[...] = comm[:, :half].astype(jnp.bfloat16).reshape(payload_rows, 128)
        copies = []
        for offset in range(15, 0, -1):
            copy = tpu.make_async_remote_copy(
                own_partial, own_partial, sends.at[12], recvs.at[12],
                device_id=(rank ^ (2 * offset),), device_id_type=pl.DeviceIdType.MESH,
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
            [host_sums[2 * group] + host_sums[2 * group + 1] for group in range(2)], axis=0,
        ).astype(jnp.bfloat16)
        payload_rows *= 2
        rows_per_pair *= 2
    else:
        # Reduce-scatter the group's rows: pair q of each local host owns rows
        # q * r .. q * r + r - 1.
        local_send = (local.at[0].bitcast(jnp.bfloat16) if staging is None else staging.at[0]).at[
            pl.ds(0, 4 * payload_rows), :
        ].reshape(4, payload_rows, 128)
        local_receive = (local.at[1].bitcast(jnp.bfloat16) if staging is None else staging.at[1]).at[
            pl.ds(0, 4 * payload_rows), :
        ].reshape(4, payload_rows, 128)
        for target_pair in range(4):
            local_send[target_pair] = comm[
                pl.ds(target_pair * rows_per_pair if owners == 4 else 0, rows_per_pair), :half
            ].astype(jnp.bfloat16).reshape(payload_rows, 128)
        local_receive[pair] = local_send[pair]
        copies = []
        for offset in range(3, 0, -1):
            copy = tpu.make_async_remote_copy(
                local_send.at[pair ^ offset], local_receive.at[pair],
                sends.at[offset - 1], recvs.at[offset - 1],
                device_id=(rank ^ (2 * offset),), device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        local_sum = jnp.sum(local_receive[...].astype(jnp.float32), axis=0)

        if shards == 2:
            # TP16 spans two hosts; TP8 finishes its reduction on one host.
            host_receive = host_reduction.at[:2, pl.ds(0, payload_rows), :]
            host_receive[host % 2] = local_sum
            host_view = host_receive.at[host % 2]
            copy = tpu.make_async_remote_copy(
                host_view, host_view, sends.at[3], recvs.at[3],
                device_id=(rank ^ 8,), device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copy.wait()
            reduced_half = jnp.sum(host_receive[...], axis=0).astype(jnp.bfloat16)
        else:
            reduced_half = local_sum.astype(jnp.bfloat16)

    # Assemble the full row across the rank pair.
    pair_send = local.at[2].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    pair_receive = local.at[3].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    pair_send[...] = reduced_half
    copy = tpu.make_async_remote_copy(
        pair_send, pair_receive, sends.at[6], recvs.at[6],
        device_id=(rank ^ 1,), device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()
    copy.wait()
    sent = pair_send[...].reshape(rows_per_pair, half)
    received_half = pair_receive[...].reshape(rows_per_pair, half)
    row = jnp.concatenate(
        (jnp.where(lane == 0, sent, received_half), jnp.where(lane == 0, received_half, sent)),
        axis=1,
    )

    if keep_dp_rows:
        assert out.shape == (rows_per_pair, hidden)
        out[...] = row
    else:
        _gather_attention_dp_rows(row, out, local, sends, recvs, shards)
    phase[0] += 1


def _gather_attention_dp_rows(row, out, local, sends, recvs, shards):
    """Restore a pair's DP subsets in canonical order, without arithmetic.

    Each rank has only [batch / (4 * DP), hidden]. EP8/TP4 routing still
    needs every pair's rows on every host. Keeping this exchange separate
    lets residual mixing and normalization execute before replication.
    """
    rank = jax.lax.axis_index("tp")
    group_size = 32 // shards
    group = rank // group_size
    rows_per_pair, hidden = row.shape
    assert out.shape == (shards * rows_per_pair, hidden)
    # Pack into flat wires so a one-row subset (DP4 B16) is DMA aligned.
    group_payload = rows_per_pair * hidden // 128
    group_send = local.at[4].bitcast(jnp.bfloat16).at[pl.ds(0, group_payload), :]
    group_send[...] = row.reshape(group_payload, 128)
    group_receives, group_copies = [], []
    for offset in range(1, shards):
        received_group = local.at[4 + offset].bitcast(jnp.bfloat16).at[
            pl.ds(0, group_payload), :
        ]
        copy = tpu.make_async_remote_copy(
            group_send, received_group, sends.at[6 + offset], recvs.at[6 + offset],
            device_id=(rank ^ (group_size * offset),), device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        group_receives.append(received_group)
        group_copies.append(copy)
    for copy in group_copies:
        copy.wait()
    for target_group in range(shards):
        group_row = row
        for offset, received_group in enumerate(group_receives, start=1):
            group_row = jnp.where(
                (group ^ offset) == target_group,
                received_group[...].reshape(rows_per_pair, hidden),
                group_row,
            )
        out[target_group * rows_per_pair : (target_group + 1) * rows_per_pair, :] = group_row


# Hidden-row gathers that travel as FP8 (KernelConfig.fp8_wires).
FP8_WIRES = frozenset(_config().fp8_wires)


def _fp8_wire_size(rows, hidden):
    """32-bit wire rows of ``rows`` FP8 hidden rows: the e4m3 bytes, four per
    word, then one 8-row tile carrying the row scales."""
    return rows * hidden // 512 + 8


def _fp8_wire_slots(local, payload, minimum):
    """(slot count, stride) of int32 FP8 wire slots tiling ``local`` exactly:
    the most slots (a power of two, at least ``minimum``) whose 8-row aligned
    stride still holds ``payload`` rows; None if ``minimum`` slots do not fit
    (the gather then keeps its BF16 wires)."""
    total = local.shape[0] * local.shape[1]

    def fits(slots):
        return total % slots == 0 and total // slots >= payload and (total // slots) % 8 == 0

    if not fits(minimum):
        return None
    slots = minimum
    while fits(2 * slots):
        slots *= 2
    return slots, total // slots


def _fp8_wire_encode(value):
    """[rows, hidden] -> int32 [_fp8_wire_size, 128]: each row scaled by its
    absolute maximum / 448 and rounded to e4m3; row ``i``'s FP32 scale bits sit
    in lane ``i`` of the tail tile."""
    rows, hidden = value.shape
    assert rows % 4 == 0 and rows <= 128 and hidden % 128 == 0
    value = value.astype(jnp.float32)
    maximum = jnp.max(jnp.abs(value), axis=1, keepdims=True)
    scale = jnp.where(maximum > 0, maximum / 448.0, 1.0)
    quantized = (value / scale).reshape(rows * hidden // 128, 128).astype(jnp.float8_e4m3fn)
    words = tpu.bitcast(quantized, jnp.int32)
    bits = jax.lax.bitcast_convert_type(scale, jnp.int32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, (rows, 128), 1)
    sublanes = jax.lax.broadcasted_iota(jnp.int32, (rows, 128), 0)
    tail = jnp.sum(jnp.where(lanes == sublanes, bits, 0), axis=0, keepdims=True)
    return jnp.concatenate([words, jnp.broadcast_to(tail, (8, 128))], axis=0)


def _fp8_wire_decode(words, rows, hidden):
    """The BF16 rows of an ``_fp8_wire_encode`` wire (int32 value)."""
    body = rows * hidden // 512
    quantized = tpu.bitcast(words[:body], jnp.float8_e4m3fn)
    value = quantized.astype(jnp.float32).reshape(rows, hidden)
    lanes = jax.lax.broadcasted_iota(jnp.int32, (rows, 128), 1)
    sublanes = jax.lax.broadcasted_iota(jnp.int32, (rows, 128), 0)
    tail = jnp.broadcast_to(words[body : body + 1], (rows, 128))
    bits = jnp.sum(jnp.where(lanes == sublanes, tail, 0), axis=1, keepdims=True)
    scale = jax.lax.bitcast_convert_type(bits, jnp.float32)
    return (value * scale).astype(jnp.bfloat16)


def _gather_dp_moe_rows(row, owned_out, out, local, sends, recvs, shards):
    """``_gather_attention_dp_rows`` and the local pair gather, overlapped.

    Each rank holds [batch / (4 * DP), hidden] for its (DP group, local pair)
    and ends with the whole batch in ``out`` and its pair's rows of all groups
    in ``owned_out``. Own rows go to the local pairs and the other groups
    together, so the inter-group exchange does not delay the local one. Flat
    wires use lanes 4..7 of ``local``: the wire from relative position
    (group ^ o, pair ^ k) lands in slot (o, k) on every rank.
    """
    rank = jax.lax.axis_index("tp")
    group_size = 32 // shards
    group, pair = rank // group_size, (rank % 8) // 2
    rows, hidden = row.shape
    tiling = (_fp8_wire_slots(local, _fp8_wire_size(rows, hidden), 8 * shards)
              if "t10" in FP8_WIRES and rows % 4 == 0 else None)
    fp8 = tiling is not None
    if fp8:
        payload = _fp8_wire_size(rows, hidden)
        slot_count, stride = tiling
        wires = pool_alias.view(local, (slot_count, stride, 128), jnp.int32)
    else:
        payload = rows * hidden // 128
        # A one-row wire (B8 DP2: 56 BF16 rows) starts its slot on a whole tile.
        stride = -(-payload // 16) * 16
        slot_count = 2 * local.shape[0] * local.shape[1] // stride
        wires = pool_alias.view(local, (slot_count, stride, 128), jnp.bfloat16)
    base = slot_count // 2
    assert base >= 4 * shards

    def wire(o, k):
        return wires.at[base + 4 * o + k, pl.ds(0, payload)]

    def received(slot):
        value = wires[slot, pl.ds(0, payload)]
        return _fp8_wire_decode(value, rows, hidden) if fp8 else value.reshape(rows, hidden)

    def send(source, o, k, semaphore):
        # Peer (o, k) sees this rank at the same relative position.
        copy = tpu.make_async_remote_copy(
            source, wire(o, k), sends.at[semaphore], recvs.at[semaphore],
            device_id=(rank ^ (group_size * o) ^ (2 * k),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        return copy

    wire(0, 0)[...] = _fp8_wire_encode(row) if fp8 else row.reshape(payload, 128)
    if rows <= 2:
        # Short wires (DP2 B8/B16) go to every peer directly: one hop.
        direct = [send(wire(0, 0), o, k, 7 if o == 0 else 8)
                  for o in range(shards) for k in range(4) if o or k]
        for copy in direct:
            copy.wait()
    else:
        local_copies = [send(wire(0, 0), 0, k, 7) for k in range(1, 4)]
        group_copies = [send(wire(0, 0), o, 0, 8) for o in range(1, shards)]
        # The groups' wires share one semaphore: forward only once all landed.
        for copy in group_copies:
            copy.wait_recv()
        forwarded = []
        for o in range(1, shards):
            for k in range(1, 4):
                forwarded.append(tpu.make_async_remote_copy(
                    wire(o, 0), wire(o, k), sends.at[9], recvs.at[9],
                    device_id=(rank ^ (2 * k),), device_id_type=pl.DeviceIdType.MESH,
                ))
                forwarded[-1].start()
        for copy in group_copies:
            copy.wait_send()
        for copy in (*local_copies, *forwarded):
            copy.wait()
    for d in range(shards):
        for p in range(4):
            slot = base + 4 * (group ^ d) + (pair ^ p)
            first = (p * shards + d) * rows
            out[first : first + rows, :] = received(slot)
        owned_out[d * rows : (d + 1) * rows, :] = received(base + 4 * (group ^ d))


def _gather_latent_host(x,sends,recvs):
    rank=jax.lax.axis_index('tp')
    view=x.at[:,pl.ds((rank%8)*512,512)]
    copies=[]
    for offset in range(7,0,-1):
        cp=tpu.make_async_remote_copy(view,view,sends.at[offset-1],recvs.at[offset-1],device_id=(rank^offset,),device_id_type=pl.DeviceIdType.MESH)
        cp.start();copies.append(cp)
    for cp in copies:cp.wait()


def _router_copy(weight, buffer, sems, wait, *, initialize_padding=True):
    lane = jax.lax.axis_index("tp") % 2

    @pl.when(lane < 1)
    def full():
        copy = tpu.make_async_copy(
            weight.at[:, pl.ds(lane * 512, 512)], buffer.at[0], sems.at[0]
        )
        copy.wait() if wait else copy.start()

    @pl.when(lane == 1)
    def tail():
        if not wait and initialize_padding:
            buffer[0, :, 384:] = jnp.zeros((weight.shape[0], 128), jnp.bfloat16)
        copy = tpu.make_async_copy(
            weight.at[:, pl.ds(512, 384)], buffer.at[0, :, :384], sems.at[0]
        )
        copy.wait() if wait else copy.start()


def _dense_scatter_rows(partial, output, local, hosts, sends, recvs, owners):
    """Reduce the dense layer directly to its token owners.

    Preserve the original BF16 wire format, local eight-rank sum and four-host
    sum, including their rounding points. Each owner consumes eight rows per
    round instead of every rank reducing the entire batch and discarding most
    rows. Pack each owner's source once; the two ranks in each owner pair send
    and receive identical logical row ranges.
    """
    rank = jax.lax.axis_index("tp")
    lane, host = rank % 8, rank // 8
    owned_rows = partial.shape[0] // owners
    hidden = partial.shape[1]
    wire_rows = 8 * hidden // 128
    payload = wire_rows // 2
    assert owned_rows % 8 == 0 and 8 % owners == 0
    assert payload <= local.shape[1] and payload <= hosts.shape[1]

    def reduce_owned(wires):
        def encode(value):
            packed = tpu.bitcast(
                value.reshape(wire_rows, 128).astype(jnp.bfloat16), jnp.uint32
            )
            return jax.lax.bitcast_convert_type(packed, jnp.float32)

        def decode(value):
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
                    device_id=(rank ^ (8 * offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()
            output[first : first + 8] = decode(hosts[:, :payload]).astype(jnp.bfloat16)

    pl.run_scoped(reduce_owned, tpu.VMEM((owners, payload, 128), jnp.float32))


def _all_reduce_rows(partial, local, hosts, sends, recvs):
    """TP32 sum of an FP32 ``[batch, hidden]`` partial on every rank.

    The shared-expert transport of the mixture-of-experts operation: the
    partial is rounded to BF16 and packed in pairs into 32-bit words, each
    lane sends its packet to the seven other local lanes, the local sum is
    packed again and exchanged among the four hosts. ``local`` and ``hosts``
    are the wide reduction buffers of the communication workspace
    (``[8, rows, 128]`` and ``[4, rows, 128]`` FP32); semaphores 0..6 and
    7..9 are used and left idle.
    """
    rank = jax.lax.axis_index("tp")
    lane, host = rank % 8, rank // 8
    batch_size, hidden = partial.shape
    rows = batch_size * hidden // 128
    payload = rows // 2
    assert rows % 2 == 0 and payload <= local.shape[1] and payload <= hosts.shape[1]

    def encode(value, destination):
        packed = tpu.bitcast(value.reshape(rows, 128).astype(jnp.bfloat16), jnp.uint32)
        destination[:payload, :] = jax.lax.bitcast_convert_type(packed, jnp.float32)

    def decode(buffer):
        decoded = tpu.bitcast(
            jax.lax.bitcast_convert_type(buffer, jnp.uint32), jnp.bfloat16
        ).astype(jnp.float32)
        return jnp.sum(decoded, axis=0).reshape(batch_size, hidden)

    def exchange(buffer, index, offsets, stride, semaphore_base):
        view = buffer.at[index, :payload, :]
        copies = []
        for offset in offsets:
            copy = tpu.make_async_remote_copy(
                view,
                view,
                sends.at[semaphore_base + offset],
                recvs.at[semaphore_base + offset],
                device_id=(rank ^ (stride * offset),),
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


def _gather_sp8_local_rows(row, out, local, sends, recvs):
    """Eight owners directly feed the existing pre-attention all-gather."""
    rank = jax.lax.axis_index("tp")
    owner = rank % 8
    rows, hidden = row.shape
    tiling = _fp8_wire_slots(local, _fp8_wire_size(rows, hidden), 8) if "t0" in FP8_WIRES else None
    fp8 = tiling is not None
    if fp8:
        payload = _fp8_wire_size(rows, hidden)
        slots, stride = tiling
        wires = pool_alias.view(local, (slots, stride, 128), jnp.int32).at[:, pl.ds(0, payload)]
    else:
        payload = rows * hidden // 128
        slots = 2 * local.shape[0] * local.shape[1] // payload
        wires = pool_alias.view(local, (slots, payload, 128), jnp.bfloat16)
    assert rows == 8 and payload % (8 if fp8 else 16) == 0
    assert slots >= (8 if fp8 else 32)
    source = wires.at[owner]
    source[...] = _fp8_wire_encode(row) if fp8 else row.reshape(payload, 128)
    copies = []
    for offset in range(7, 0, -1):
        copy = tpu.make_async_remote_copy(
            source, source, sends.at[offset - 1], recvs.at[offset - 1],
            device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
    def received(slot):
        return _fp8_wire_decode(wires[slot], rows, hidden) if fp8 else wires[slot].reshape(rows, hidden)

    for pair in range(4):
        out[pair * 16:(pair + 1) * 16] = jnp.concatenate(
            (received(2 * pair), received(2 * pair + 1)), axis=0)


def _gather_sp8_dp_moe_rows(row, owned_out, out, local, sends, recvs):
    """DP2 MoE-input all-gather, eight owners, existing pair-major ABI out.

    Payload slots are relative (DP group XOR, local core XOR). The upper
    half of the existing transport allocation is disjoint from pre gather.
    No separate pair reconstruction collective is needed.
    """
    rank = jax.lax.axis_index("tp")
    group, owner = rank // 16, rank % 8
    rows, hidden = row.shape
    tiling = _fp8_wire_slots(local, _fp8_wire_size(rows, hidden), 32) if "t10" in FP8_WIRES else None
    fp8 = tiling is not None
    if fp8:
        payload = _fp8_wire_size(rows, hidden)
        slots, stride = tiling
        wires = pool_alias.view(local, (slots, stride, 128), jnp.int32)
    else:
        payload = rows * hidden // 128
        slots = 2 * local.shape[0] * local.shape[1] // payload
        wires = pool_alias.view(local, (slots, payload, 128), jnp.bfloat16)
    assert rows == 8 and payload % (8 if fp8 else 16) == 0
    base = slots // 2
    assert base >= 16

    def wire(o, k):
        return wires.at[base + 8 * o + k, pl.ds(0, payload)] if fp8 else wires.at[base + 8 * o + k]

    def received(o, k):
        value = wire(o, k)[...]
        return _fp8_wire_decode(value, rows, hidden) if fp8 else value.reshape(rows, hidden)

    def send(source, o, k, semaphore, *, local_only=False):
        copy = tpu.make_async_remote_copy(
            source, wire(o, k), sends.at[semaphore], recvs.at[semaphore],
            device_id=(rank ^ (0 if local_only else 16 * o) ^ k,), device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        return copy

    wire(0, 0)[...] = _fp8_wire_encode(row) if fp8 else row.reshape(payload, 128)
    local_copies = [send(wire(0, 0), 0, k, 7) for k in range(1, 8)]
    group_copy = send(wire(0, 0), 1, 0, 8)
    group_copy.wait_recv()
    forwarded = [send(wire(1, 0), 1, k, 9, local_only=True) for k in range(1, 8)]
    group_copy.wait_send()
    for copy in (*local_copies, *forwarded):
        copy.wait()
    for d in range(2):
        for p in range(4):
            even = received(group ^ d, owner ^ (2 * p))
            odd = received(group ^ d, owner ^ (2 * p + 1))
            first = (2 * p + d) * 16
            out[first:first + 16] = jnp.concatenate((even, odd), axis=0)
        own = received(group ^ d, 0)
        peer = received(group ^ d, 1)
        owned_out[d * 16:(d + 1) * 16] = jnp.where(
            owner % 2 == 0, jnp.concatenate((own, peer), axis=0),
            jnp.concatenate((peer, own), axis=0))


def _gather_local_pair_rows(source, destination, sends, recvs, scratch=None, owners=4,
                            *, packed=False, fp8=False):
    """All-gather the token shards of the ``owners`` local row owners.

    Owner o (4: the local rank pairs; 8: the local ranks) holds rows
    o * r .. o * r + r - 1 in ``source`` [r, hidden]; every rank ends with all
    of them in ``destination``. Each rank sends to its counterpart in every
    other owner (rank ^ k * 8 / owners). ``packed`` stages through flat
    transport wires when a source row slice crosses a native BF16 tile.
    """
    rank = jax.lax.axis_index("tp")
    stride = 8 // owners
    owner = (rank % 8) // stride
    rows_per_owner = source.shape[0]
    if (fp8 and rows_per_owner % 4 == 0 and scratch is not None
            and _fp8_wire_size(rows_per_owner, source.shape[1]) <= scratch.shape[1]):
        # FP8 wires (``_fp8_wire_encode``) through the transport scratch.
        assert scratch is not None
        hidden = source.shape[1]
        payload_rows = _fp8_wire_size(rows_per_owner, hidden)
        source_wire = scratch.at[owner].bitcast(jnp.int32).at[pl.ds(0, payload_rows), :]
        source_wire[...] = _fp8_wire_encode(source[...])
        copies = []
        for offset in range(owners - 1, 0, -1):
            copy = tpu.make_async_remote_copy(
                source_wire,
                source_wire,
                sends.at[offset - 1],
                recvs.at[offset - 1],
                device_id=(rank ^ (stride * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()
        destination[...] = jnp.concatenate(
            tuple(
                _fp8_wire_decode(
                    scratch.at[source_owner].bitcast(jnp.int32)[:payload_rows],
                    rows_per_owner, hidden,
                )
                for source_owner in range(owners)
            ),
            axis=0,
        )
        return
    # One row, or a row count whose slice is not tile-aligned for a DMA
    # (3, 5, 6, 7 rows at B12..B28): send a flattened wire through the
    # transport scratch and assemble the destination from values.
    if packed or rows_per_owner == 1 or rows_per_owner & (rows_per_owner - 1):
        assert scratch is not None
        payload_rows = rows_per_owner * source.shape[1] // 128
        source_wire = scratch.at[owner].bitcast(jnp.bfloat16).at[
            pl.ds(0, payload_rows), :
        ]
        source_wire[...] = source[...].reshape(payload_rows, 128)
        copies = []
        for offset in range(owners - 1, 0, -1):
            copy = tpu.make_async_remote_copy(
                source_wire,
                source_wire,
                sends.at[offset - 1],
                recvs.at[offset - 1],
                device_id=(rank ^ (stride * offset),),
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
            destination[...] = jnp.stack(tuple(wire.reshape(source.shape[1]) for wire in wires))
        else:
            destination[...] = jnp.concatenate(
                tuple(wire.reshape(rows_per_owner, source.shape[1]) for wire in wires), axis=0
            )
        return
    for owner_index in range(owners):
        @pl.when(owner == owner_index)
        def gather_owner(owner_index=owner_index):
            row_slice = pl.ds(
                owner_index * rows_per_owner, rows_per_owner
            )
            destination[row_slice] = source[...]
            copies = []
            for offset in range(owners - 1, 0, -1):
                copy = tpu.make_async_remote_copy(
                    source,
                    destination.at[row_slice],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ (stride * offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()


# Repeating-stack composition.

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
KDA = (
    "k_gate",
    "k_projection",
    "k_fb",
    "k_o",
    "k_conv",
    "k_a_log",
    "k_dt",
    "k_norm",
)
MLA = ("m_qa", "m_ka", "m_qb", "m_kb", "m_gate", "m_o", "m_qnorm", "m_knorm")
NORMS = ("attn_res", "mlp_res", "attn_norm", "ffn_norm")
EPILOGUE = ("output_res", "final_norm")
DENSE = ("dense_gu", "dense_down")  # layer 0's MLP, used when the stack includes layer 0
NAMES = MOE + KDA + MLA + NORMS


def local_shapes(
    layers=93,
    hidden=7168,
    *,
    full_model=False,
    vocab=163840,
    expert_storage="mxfp4",
    split_kda_gate_projection: bool = True,
):
    """Per-rank weight shapes. Layer and expert indices are never replicated/tied."""
    if not split_kda_gate_projection:
        raise ValueError("The clean decoder always uses the split KDA gate projection")
    if not 2 <= layers <= 93:
        raise ValueError("Kimi stack supports a prefix of 2..93 layers")
    m = max(1, layers // 4 + (layers == 93))
    k = layers - (layers // 4 + (layers == 93))
    h = hidden
    shapes = (
        (layers - 1, h, 896),
        (layers - 1, 1, 896),
        (layers - 1, h, 128),
        (layers - 1, 128, h),
        (layers - 1, 1, 4096),
        (layers - 1, h, 512),
        (layers - 1, 256, h),
        (layers - 1, 112, 448, 1536),
        (layers - 1, 112, 112, 1536),
        (layers - 1, 112, 96, 3584),
        (layers - 1, 112, 24, 3584),
        (k, h, 128),
        (k, h, 1664 if split_kda_gate_projection else 1792),
        (k, 128, 384),
        (k, 384, h),
        (k, 4, 3, 3, 128),
        (k, 3, 1),
        (k, 3, 128),
        (k, 8, 128),
        (m, h, 1536),
        (m, h, 640),
        (m, 1536, 768),
        (m, 512, 768),
        (m, h, 384),
        (m, 384, h),
        (m, 1, 1536),
        (m, 1, 512),
        *((layers, 1, h) for _ in range(4)),
    )
    result = dict(zip(NAMES, shapes, strict=True))
    if expert_storage != "mxfp4":
        raise ValueError("Routed experts require resident packed FP4")
    if full_model:
        if vocab % 32:
            raise ValueError("Vocabulary must divide into 32 shards")
        result.update(
            embedding=(vocab // 32, h),
            lm_head=(h, vocab // 32),
            dense_gu=(h, 2304),
            dense_down=(1152, h),
            output_res=(1, h),
            final_norm=(1, h),
        )
    return result





def _residual_mixture_row(
    block_values,
    prefix_value,
    folded,
    active_block_count,
    *,
    epsilon,
    relaxed_reduction_order=False,
):
    """Compute one row of the learned residual mixture."""
    def normalized_scores(values):
        scores = jnp.sum(values * folded[...], axis=1, keepdims=True)
        return scores * jax.lax.rsqrt(
            jnp.mean(values * values, axis=1, keepdims=True) + epsilon
        )

    if relaxed_reduction_order:
        block_values = block_values.astype(jnp.float32)
        prefix_value = prefix_value.astype(jnp.float32)

        # Keeping the hidden-sized prefix separate avoids materializing and
        # reducing a nine-row tensor. This changes FP32 association and is
        # therefore intentionally restricted to the relaxed mode.
        scores = jnp.concatenate(
            (
                normalized_scores(block_values),
                normalized_scores(prefix_value),
            ),
            axis=0,
        )
        values = None
    else:
        values = jnp.concatenate(
            (
                block_values,
                prefix_value,
            ),
            axis=0,
        ).astype(jnp.float32)
        scores = jnp.sum(values * folded[...], axis=1, keepdims=True)
        scores *= jax.lax.rsqrt(
            jnp.mean(values * values, axis=1, keepdims=True) + epsilon
        )

    residual_indices = jax.lax.broadcasted_iota(jnp.int32, (9, 1), 0)
    active = (residual_indices < active_block_count) | (
        residual_indices == 8
    )
    probabilities = jax.nn.softmax(
        jnp.where(active, scores, -jnp.inf), axis=0
    )
    if relaxed_reduction_order:
        mixed = jnp.sum(
            block_values * probabilities[:8], axis=0, keepdims=True
        ) + prefix_value * probabilities[8:9]
    else:
        mixed = jnp.sum(values * probabilities, axis=0, keepdims=True)
    return mixed.astype(jnp.bfloat16)


def _residual_mixture_by_row(
    blocks,
    prefix,
    destination,
    folded,
    active_block_count,
    *,
    batch_size,
    epsilon,
    relaxed_reduction_order=False,
):
    """Compute a residual mixture for a batch-leading input."""
    for batch_index in range(batch_size):
        destination[batch_index : batch_index + 1] = _residual_mixture_row(
            blocks[batch_index],
            prefix[batch_index : batch_index + 1],
            folded,
            active_block_count,
            epsilon=epsilon,
            relaxed_reduction_order=relaxed_reduction_order,
        )


# Stage fixed-size layer weights in the persistent projection/expert pool
# (KernelConfig.pooled_weights=False: scoped VMEM buffers).
POOLED_WEIGHTS = _config().pooled_weights
# From B32 a routed expert used by 2..COMPACT_ROWS rows runs gate/up and down on
# a compact tile of those rows, not the whole batch (the FP8 path's
# per-scale-block accumulation scales with it). C=4 does not compile.
COMPACT_ROWS = 8
# A routed expert used by one row (from B8) stages it via scalar compares and
# updates only that output row. KernelConfig.singleton_block (batch a multiple
# of 16) selects it from its aligned 16-row block instead of one branch per row;
# it and stage_blocks default to on at B16-B128.
SINGLETON_BLOCK = _config().singleton_block
STAGE_BLOCKS = _config().stage_blocks
# MXFP4 tiles: single-expert contraction block (multiple of 1024: scale rows
# stay 32-aligned), paired gate/up tile, and split MXU dot tile (multiple of
# 128; 0 = unsplit); split tiles keep the scale-reduction order.
MXFP4_DOT_TILE = _config().mxfp4_dot_tile
if MXFP4_DOT_TILE is not None and (MXFP4_DOT_TILE < 0 or MXFP4_DOT_TILE % 128):
    raise ValueError("KernelConfig.mxfp4_dot_tile must be zero or a positive multiple of 128")
MXFP4_BLOCK = _config().mxfp4_block
MXFP4_PAIR_TILE = _config().mxfp4_pair_tile
# KernelConfig.mxfp4_static_tile=T: unroll the single-expert MXFP4 product over
# static T-row tiles instead of a dynamic loop, keeping the unpacked tile small.
MXFP4_STATIC_TILE = _config().mxfp4_static_tile


# B16 runs routed experts on compact 8-row tiles (MXFP4 only).
B16_COMPACT_EXPERTS = (
    _config().b16_compact_experts
    and _config().expert_gate_up == "mxfp4"
)


def _compact_rows(batch_size):
    if batch_size == 16 and B16_COMPACT_EXPERTS and not COMPACT_ONLY:
        return 8
    return COMPACT_ROWS if batch_size >= 32 else 0


COMPACT_ONLY = _config().compact_only

# Shared finishers keep the B16-B128 expert loop and its successor within
# IMEM (decode runs these batches without deduplicated HLO calls).
SHARED_EXPERT_FINISHERS = _config().shared_expert_finishers
# Grouped MXFP4 routes at B16-B128 keep the 2/4/8-row gate/up pairs but
# finish overflowing experts in eight-row passes; without the full-batch
# fallback the expert loop fits in IMEM next to its predecessor.
COMPACT_OVERFLOW = _config().compact_overflow
SHARED_SHARDS = _config().shared_shards
# KernelConfig.kda_staged=True (above B8, no attention DP): stage the KDA input
# and feature-gate projections in VMEM; with a state stream (above B16) the
# chunk loop is rolled into one pl.loop.
# TODO: derive separate staging/rolling rules for B32/B64 with 8-row chunks.
KDA_STAGED = _config().kda_staged
# Diagnostic: skip named parts of each layer; results are invalid (timing/hang
# bisection only). Every rank skips the same parts, so collectives stay paired.
DEBUG_SKIP = set(_config().debug_skip)


class _RowFunction:
    """``f[rows]`` evaluates ``f(rows)``: a per-chunk value computed from refs."""

    def __init__(self, function):
        self.function = function

    def __getitem__(self, rows):
        return self.function(rows)


def _shared_shards(batch_size):
    # B128 keeps only each lane's shared-expert output shard (VMEM).
    if SHARED_SHARDS is None:
        return batch_size == 128
    return SHARED_SHARDS and batch_size > 32


def _use_compact_overflow(batch_size, gate_up_dtype, *, group_pending_routes):
    return (
        COMPACT_OVERFLOW and not COMPACT_ONLY and SHARED_EXPERT_FINISHERS
        and (batch_size in (32, 64, 96, 128) or (batch_size == 16 and B16_COMPACT_EXPERTS))
        and COMPACT_ROWS == 8
        and gate_up_dtype == jnp.uint32 and group_pending_routes
    )


def _expert_accumulator_rows(batch_size, gate_up_dtype=None, *, group_pending_routes=False):
    """(gate/up, down) accumulator rows: the batch, or only the compact tiles."""
    capacity = _compact_rows(batch_size)
    if capacity and (COMPACT_ONLY or _use_compact_overflow(
        batch_size, gate_up_dtype, group_pending_routes=group_pending_routes
    )):
        return 2 * capacity, capacity
    return batch_size, batch_size
MOE_PREPARE_WEIGHTS = ("latent_down", "shared_gu", "shared_down")
MLA_EXTRA_WEIGHTS = ("m_qb", "m_kb", "m_gate")


def _pool_layout(
    local_weights, *, alias_expert_scales: bool, pooled: bool,
    managed: bool = False, dense_bytes: int = 0, arena_extra: int = 0,
):
    """Byte offsets of every view in the projection/expert pool.

    Region A (from 0) holds, one phase at a time: the next layer's attention
    projections (KDA k_gate..k_o, or MLA q/kv-down), prefetched after the
    experts finish; the MLA output projection over the consumed q-down
    projection; the four routed-expert slots during the MoE. Region B (after
    A) holds the MLA q/kv-up and gate projections during attention, the
    latent-down and shared-expert weights during routing, and latent-up in the
    MoE tail (which overlaps the next layer's attention prefetch into A).
    """
    def storage(name, experts=False):
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
    # One layer's attention input projections, as laid out by pool_views.
    layout["attention_bytes"] = max(
        sum(storage(name) for name in names) for names in (KDA[:4], MLA[:2])
    )
    if managed:
        # Region A is the arena the manager places the router, attention
        # projections, expert slots and dense slabs in.
        region_b = -(-max(expert_bytes, layout["attention_bytes"], dense_bytes) // 4096) * 4096
        region_b += arena_extra
        layout["arena_bytes"] = region_b
    end = region_b
    # Three disjoint lifetimes share region B: MLA attention, MoE routing and
    # shared expert, MoE tail.
    for names in (MLA_EXTRA_WEIGHTS, MOE_PREPARE_WEIGHTS, ("latent_up",)):
        offset = region_b
        for name in names:
            layout[name] = offset
            offset += storage(name)
        end = max(end, offset)
    layout["m_o"] = 0
    if storage("m_o") > storage("m_qa"):
        raise ValueError("The MLA output projection must fit over the q-down projection")
    layout["pool_bytes"] = end
    return layout


def _vmem_bytes(spec) -> int:
    """Tiled VMEM bytes of a scratch spec (0 for semaphores and SMEM)."""
    if str(getattr(spec, "memory_space", "")) != "vmem":
        return 0
    shape, itemsize = tuple(spec.shape), jnp.dtype(spec.dtype).itemsize
    if len(shape) < 2:
        return -(-math.prod(shape) * itemsize // 512) * 512
    packing = max(1, 4 // itemsize)
    rows, tile_rows = shape[-2], 8 * packing
    if rows < tile_rows:
        # Short second-minor dimensions use smaller native tiles (1 row for
        # 32-bit types, 2 or 4 for BF16), as in the compiler's reports.
        padded = max(packing, 1 << (rows - 1).bit_length())
    else:
        padded = -(-rows // tile_rows) * tile_rows
    return math.prod(shape[:-2]) * padded * (-(-shape[-1] // 128) * 128) * itemsize


VMEM_LIMIT_BYTES = _config().vmem_limit_mb * 1024**2
# Register-allocator spill slots are one program-wide VMEM area the compiler
# sizes; the budget only warns against this reserve.
VMEM_SPILL_RESERVE_BYTES = int(_config().vmem_spill_reserve_mb * 1024**2)


def _vmem_budget(persistent, regions, *, limit=None, spill_reserve=None):
    """Check a static VMEM plan and render it as a table.

    ``persistent`` is [(name, spec)], live for the whole kernel. ``regions`` is
    {region: ([(name, spec)] held for the region, [[(name, spec)], ...] nested
    alternatives)}; regions never overlap each other, and a region's nested
    alternatives never overlap each other. Returns (report, peak bytes).
    """
    limit = VMEM_LIMIT_BYTES if limit is None else limit
    spill_reserve = VMEM_SPILL_RESERVE_BYTES if spill_reserve is None else spill_reserve
    mib = lambda value: f"{value / 2**20:8.2f}"  # noqa: E731
    lines = ["persistent VMEM (MiB):"]
    persistent_bytes = 0
    for name, spec in persistent:
        size = _vmem_bytes(spec)
        persistent_bytes += size
        if size >= 64 * 1024:
            lines.append(f"  {mib(size)}  {name} {tuple(spec.shape)} {jnp.dtype(spec.dtype).name}")
    lines.append(f"  {mib(persistent_bytes)}  total (entries under 64 KiB included)")
    peaks = {}
    for region, (held, nested) in regions.items():
        held_bytes = sum(_vmem_bytes(spec) for _, spec in held)
        nested_bytes = [sum(_vmem_bytes(spec) for _, spec in alternative) for alternative in nested]
        peaks[region] = held_bytes + max(nested_bytes, default=0)
        lines.append(f"scoped region {region}: peak {mib(peaks[region]).strip()} MiB")
        for name, spec in held:
            if _vmem_bytes(spec) >= 64 * 1024:
                lines.append(f"  {mib(_vmem_bytes(spec))}  {name}")
        for alternative, size in zip(nested, nested_bytes):
            if size:
                lines.append(f"  {mib(size)}  nested: {', '.join(name for name, _ in alternative)}")
    worst = max(peaks, key=peaks.get) if peaks else None
    peak = persistent_bytes + (peaks[worst] if worst else 0)
    lines.append(
        f"peak {mib(peak).strip()} MiB (persistent + {worst}) + spill reserve "
        f"{mib(spill_reserve).strip()} of {mib(limit).strip()} MiB"
    )
    return "\n".join(lines), peak


VERIFY_CHUNK = 4  # kdecode P3.a: explicit DCP chunk rows (ring 3) for attention-DP verifies


def _verify_chunk(rows, heads, sequence_rows):
    if sequence_rows == 4 and heads == 6 and rows >= 16 and rows % VERIFY_CHUNK == 0:
        return VERIFY_CHUNK
    return None


def _mla_scoped_scratch(
    local_weights,
    *,
    batch_size: int,
    sequence_rows: int,
    dcp_heads: int = 3,
    dcp_dtype=jnp.bfloat16,
    dcp_page_tokens=256,
):
    """Transient MLA resources allocated once around each layer invocation
    (placeholders for the workspace's cache fields, ``mla_dcp.scoped_scratch``
    last; ``dcp_heads`` per rank)."""
    cache = (
        tpu.VMEM((8, 128), jnp.bfloat16),
        tpu.VMEM((8, 128), jnp.bfloat16),
        tpu.SemaphoreType.DMA((1,)),
    )
    return (
        *(
            ()
            if POOLED_WEIGHTS
            else (tpu.VMEM(local_weights[name].shape, local_weights[name].dtype) for name in MLA[2:6])
        ),
        *cache,
        tpu.VMEM((batch_size, 3, 256), jnp.bfloat16),
        tpu.VMEM((batch_size, 384), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 256), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 128), jnp.bfloat16),
        tpu.VMEM((batch_size, 3, 1), jnp.float32),
        tpu.VMEM((batch_size, 3, 1), jnp.float32),
        tpu.VMEM((batch_size, 3, 128), jnp.float32),
        # The DCP page buffers by batch (mla_dcp.page_plan; DcpAttention
        # reads the span and the buffers from the shape).
        *mla_dcp.scoped_scratch(batch_size, heads=dcp_heads,
                                cache_dtype=dcp_dtype, page_tokens=dcp_page_tokens, sequence_rows=sequence_rows,
                                chunk_rows=_verify_chunk(batch_size, dcp_heads, sequence_rows),
                                **mla_dcp.page_plan(batch_size, dcp_heads, sequence_rows, dcp_dtype)),
    )


def _moe_scoped_scratch(
    local_weights,
    *,
    batch_size: int,
    alias_expert_scales: bool,
    token_scatter_output: bool,
    token_scatter_owners: int = 4,
    shard_routing: bool = False,
    group_pending_routes: bool = False,
):
    """Transient MoE resources not held in the shared projection/expert pool."""
    gate_up = local_weights["expert_gu"]
    down = local_weights["expert_down"]
    latent_size = local_weights["latent_norm"].shape[-1]
    expert_input_size = (
        gate_up.shape[1] * 8 if gate_up.dtype == jnp.uint32 else gate_up.shape[1]
    )
    expert_width = down.shape[1] * 8 if down.dtype == jnp.uint32 else down.shape[1]
    expert_output_size = down.shape[-1]
    accumulator_rows = _expert_accumulator_rows(
        batch_size, gate_up.dtype, group_pending_routes=group_pending_routes
    )
    scale_scratch = () if alias_expert_scales else (
        tpu.VMEM((4, *local_weights["expert_gus"].shape[1:]), local_weights["expert_gus"].dtype),
        tpu.VMEM((4, *local_weights["expert_ds"].shape[1:]), local_weights["expert_ds"].dtype),
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
            else (tpu.VMEM(local_weights["latent_up"].shape, local_weights["latent_up"].dtype),)
        ),
        tpu.SemaphoreType.DMA,
        tpu.SemaphoreType.DMA((4, 4)),
        *scale_scratch,
        tpu.VMEM((batch_size, 128), jnp.int32),
        tpu.VMEM((batch_size, 128), jnp.float32),
        tpu.VMEM((batch_size, 128), jnp.int32),
        tpu.VMEM((batch_size, 128), jnp.float32),
        tpu.VMEM((batch_size, 1024), jnp.float32),
        tpu.VMEM((batch_size, 1024 if _shared_shards(batch_size) else 7168), jnp.float32),
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


# Shard the MoE router's top-k over the local rank pairs in the four-owner
# token-scatter layout (KernelConfig.shard_routing=True).
SHARD_ROUTING = _config().shard_routing


def kda_inplace_writes(*, batch_size, attention_dp, sequence_rows, slot_reads, state_row):
    """Whether the stack writes KDA post-states into their bank slots itself
    (KernelConfig.e2_kda_inplace: DP2 plain decode, streamed states, slot reads)."""
    return (_config().kda_inplace_writes and slot_reads and attention_dp == 2 and batch_size > 16
            and sequence_rows == 1 and state_row is None)


def stack(
    prefix,
    embedding,
    weights,
    states,
    positions,
    *,
    batch_size,
    layers=93,
    eps=1e-5,
    residual_reduction_order: Literal["reference", "row", "relaxed"] = "reference",
    mla_cache_batch_tile_size: int | None = None,
    alias_expert_scales: bool = False,
    attention_token_scatter: bool = False,
    moe_token_scatter_output: bool = False,
    moe_routed_reduce_scatter: bool = False,
    sequence_parallel_pre_attention: bool = False,
    gate_first_kda_projection: bool = True,
    sequence_rows: int = 1,
    aux_hidden_layers: tuple[int, ...] = (),
    aux_capture: Literal["layer", "attention", "mixture_input"] = "layer",
    include_layer_zero: bool = False,
    state_row=None,
    attention_dp: int = 1,
    attention_schedule: Literal["auto", "pooled", "legacy", "overlap"] = "auto",
    attention_state_pipeline: bool = False,
    attention_dp_ownership: bool = False,
    attention_projection_schedule: Literal["default", "mla_up_double"] = "default",
    moe_token_pipeline: bool = False,
    dp_moe_gather: Literal["staged", "split"] = "staged",
    mla_commit_tile_size: int | None = None,
    sequence_parallel_size: int = 4,
    shard_routing: bool | None = None,
    block_tables=None,
    dcp_pool_map: tuple[int, ...] | None = None,
    dcp_owner_table: tuple[int, ...] | None = None,
    state_slots=None,
    live_rows=None,
):
    """Fuse the decoder layers while keeping a native batch tile in VMEM.

    The MLA cache is vLLM's unified pools, read in place with decode context
    parallelism (``mla_dcp``): ``states`` is ``(conv, recurrent, *pools)``,
    MLA layer ``m`` in ``pools[dcp_pool_map[m]]``, ``block_tables`` (int32
    ``[batch_size, pages]``) the rows' tables. The pools stay read-only: in
    place of an updated cache the kernel returns each row's new latent for
    every MLA layer, ``[batch, 24, H, 256]`` / ``[batch, 24, H, 128]`` (``H >=
    3`` padded heads; ``paged.commit_latent_rows`` commits them). Nothing is
    aliased: XLA copies every operand a communicating kernel updates in
    place, so the KDA states come back as fresh outputs too.

    With the token-scatter / sequence-parallel layout, ``sequence_parallel_size``
    local row owners (4: rank pairs; 8: single ranks; 1: every rank, at
    attention DP2's B2 and B4), replicated across the hosts, each own batch_size /
    size rows for the per-token work.

    Layers 1..L-1 by default; with ``include_layer_zero`` the loop starts at
    layer 0 (``prefix`` is then the embedding; its KDA runs on the common
    attention path and a dense MLP streamed through the weight pool replaces
    the mixture of experts), which needs ``weights["dense_gu"]`` and
    ``weights["dense_down"]``. ``state_row`` (a traced int32 scalar, one
    sequence per tile only) selects the row of the KDA input states to read
    for the sequence's first row instead of a pre-selected state copy.

    ``state_slots`` (int32 ``[batch_size]``): ``states[:2]`` are the
    whole KDA slot arrays ``[slots, 69, ...]`` and row ``b`` reads its input
    state from slot ``state_slots[b]`` (DP2 indices include the head-shard
    bank offset and follow shard-major group rows; with ``state_row``, slot
    ``state_slots[state_row]``), instead of the caller gathering the rows
    first. The updated states are still returned per row. Needs
    ``state_row`` or ``batch_size`` > 4 with DP1, or B2..B128 with DP2.
    """
    if not 1 <= batch_size <= MAX_BATCH:
        raise ValueError(f"The fused decoder stack supports batch sizes from 1 to {MAX_BATCH}")
    if include_layer_zero and any(name not in weights for name in DENSE):
        raise ValueError("include_layer_zero needs the dense MLP weights")
    select_state_row = state_row is not None
    if block_tables is None or dcp_pool_map is None:
        raise ValueError("The stack takes block_tables and dcp_pool_map (vLLM's unified pools)")
    pool_map = tuple(dcp_pool_map)
    if len(dcp_pool_map) != MLA_LAYERS or len(states) != 2 + 1 + max(dcp_pool_map):
        raise ValueError("The stack takes (conv, recurrent, *pools) and a pool per MLA layer")
    from vllm_torchtpu.megakernel.kimi_k3.layout import cache_layout
    layout = cache_layout(str(states[2].dtype))
    if any(pool.shape[1:] != layout.block_shape or pool.dtype != states[2].dtype for pool in states[2:]):
        raise ValueError(f"unified pools must match {layout.block_shape} {layout.dtype}")
    if block_tables.shape[0] != batch_size or block_tables.ndim != 2:
        raise ValueError("block_tables must be [batch_size, pages]")
    if attention_dp != 1 and not (mla_dcp.GROUP * attention_dp == 32 and attention_dp_ownership):
        raise ValueError(
            "Attention DP needs DCP over each DP group (KernelConfig.dcp_group = 32 / dp) "
            "and DP row ownership"
        )
    paged_pages = int(block_tables.shape[1])
    if select_state_row and sequence_rows != batch_size:
        raise ValueError("state_row selection needs one sequence per tile")
    first_layer = 0 if include_layer_zero else 1
    dense_names = DENSE if include_layer_zero else ()
    if not 1 <= sequence_rows <= batch_size or batch_size % sequence_rows:
        raise ValueError("sequence_rows must divide the batch")
    if sequence_rows > 1:
        # Rows of one sequence share one MLA cache entry, so the cache is never
        # tiled by rows; the other batch options act on rows and stay valid.
        mla_cache_batch_tile_size = batch_size
    aux_hidden_layers = tuple(aux_hidden_layers)
    if any(not 1 <= index <= layers - 1 for index in aux_hidden_layers) or len(
        set(aux_hidden_layers)
    ) != len(aux_hidden_layers):
        raise ValueError(
            "aux_hidden_layers are distinct layer ids in 1..layers-1; id L "
            "captures the residual stream entering layer L"
        )
    if aux_capture not in ("layer", "attention", "mixture_input"):
        raise ValueError("aux_capture must be 'layer', 'attention' or 'mixture_input'")
    if aux_capture != "layer" and attention_token_scatter:
        raise ValueError("Attention-point aux capture needs complete prefix rows")
    b4_attention_gather = (
        B4_ATTENTION_GATHER and batch_size == 4 and not attention_token_scatter
    )
    if residual_reduction_order not in ("reference", "row", "relaxed"):
        raise ValueError(
            "residual_reduction_order must be 'reference', 'row', or 'relaxed'"
        )
    mla_cache_batch_tile_size = mla_cache_batch_tile_size or batch_size
    if not 1 <= mla_cache_batch_tile_size <= batch_size:
        raise ValueError("MLA cache batch tile size must be within the batch")
    if batch_size % mla_cache_batch_tile_size:
        raise ValueError("MLA cache batch tile size must divide the batch")
    if attention_token_scatter and batch_size not in (4, *SCATTER_BATCHES) and not (
        attention_dp > 1 and batch_size == 2
    ):
        raise ValueError("Attention token scatter requires batch size 4, 8 or 16, or attention DP's B2")
    if moe_token_scatter_output and not (
        attention_token_scatter and (batch_size in SCATTER_BATCHES or sequence_parallel_size < 4)
    ):
        raise ValueError(
            "MoE token-scattered output requires B8 attention token scatter"
        )
    if moe_routed_reduce_scatter and batch_size not in (6, *SCATTER_BATCHES):
        raise ValueError("MoE routed reduce-scatter requires batch size 6, 8 or 16")
    if sequence_parallel_pre_attention and not moe_token_scatter_output:
        raise ValueError(
            "Sequence-parallel pre-attention requires B8 MoE token scatter"
        )
    sp_owners = sequence_parallel_size
    # Rows of each local row owner of attention DP's layout (a rank pair, or
    # at DP2 B2/B4 every rank: all rows).
    tokens_per_pair = (
        batch_size // (sp_owners if attention_dp > 1 else 4) if attention_token_scatter else 0
    )
    if sp_owners not in (1, 4, 8):
        raise ValueError("sequence_parallel_size must be 1 (attention DP2 B2/B4), 4 (rank pairs) or 8 (ranks)")
    if sp_owners == 1 and not (attention_dp == 2 and attention_dp_ownership and batch_size in (2, 4)):
        raise ValueError("One row owner is attention DP2's B2/B4 layout")
    if sp_owners == 8 and not (
        attention_token_scatter and batch_size % 8 == 0 and attention_dp == 1
    ):
        raise ValueError(
            "Eight row owners need the token-scatter layout, a multiple of 8 rows "
            "and no attention DP"
        )
    owned_rows = batch_size // sp_owners if attention_token_scatter else 0
    shard_routing = SHARD_ROUTING if shard_routing is None else shard_routing
    # The router splits its columns over each rank pair, so the top-k shards
    # over pairs: four row owners.
    shard_routing = shard_routing and moe_token_scatter_output and sp_owners == 4
    if attention_schedule not in ("auto", "pooled", "legacy", "overlap"):
        raise ValueError("attention_schedule must be auto, pooled, legacy, or overlap")
    if attention_dp not in (1, 2, 4):
        raise ValueError("Attention DP supports dp=1, dp=2 or dp=4")
    if attention_dp > 1 and not (
        batch_size in (2, 4, 8, 16, 32, 64, 96, 128)
        and batch_size % (sp_owners * attention_dp) == 0
        and sequence_parallel_pre_attention
        and (batch_size // attention_dp) % sequence_rows == 0
        and not select_state_row
        and POOLED_WEIGHTS
    ):
        raise ValueError(
            "Attention DP needs a supported batch divisible by 4*DP, the "
            "sequence-parallel token-scatter layout, whole sequences in each "
            "group's rows and pooled weights"
        )
    if attention_dp_ownership and (
        attention_dp == 1
        or batch_size not in (32, 64, 96)
        and not (attention_dp == 2 and batch_size in (2, 4, 8, 16, 128))
    ):
        raise ValueError("DP row ownership requires attention DP2 at B2..B128 or DP4 at B32/B64/B96")
    if moe_token_pipeline and (
        attention_dp not in (2, 4)
        or batch_size not in (32, 64)
        or not moe_routed_reduce_scatter
        or not moe_token_scatter_output
        or sp_owners != 4
    ):
        raise ValueError("MoE token pipeline requires attention DP2/DP4 at B32/B64 with pair-owned routed reduce-scatter")
    if dp_moe_gather not in ("staged", "split"):
        raise ValueError("dp_moe_gather must be staged or split")
    if dp_moe_gather == "split" and not attention_dp_ownership:
        raise ValueError("The split DP MoE-input gather requires DP row ownership")
    # The new-row hand-out's padded head count (cache_handoff_head_count)
    # follows these tile sizes.
    commit_rows = mla_commit_tile_size or mla_cache_batch_tile_size
    if batch_size % commit_rows:
        raise ValueError("mla_commit_tile_size must divide the batch")
    group_rows = batch_size // attention_dp
    # Each local row owner holds tokens_per_pair rows; group d attends r of them.
    group_rows_per_pair = group_rows // sp_owners
    # The residual-stream rows a rank holds under DP row ownership: its own r,
    # or with one owner every row.
    dp_held_rows = batch_size if sp_owners == 1 else group_rows_per_pair
    default_streams = (
        {**DCP_B32_WEIGHT_STREAMS, **(DCP_B32_DECODE_RINGS if sequence_rows == 1 else {})}
        if (batch_size in (32, 64) and attention_dp == 1
            and not _config().weight_streams)
        else {}
    )
    projection_streams = _attention_projection_streams(
        attention_dp, attention_projection_schedule,
    )
    conflicts = projection_streams.keys() & WEIGHT_STREAMS.keys()
    if conflicts:
        raise ValueError(
            "Projection schedule conflicts with KernelConfig.weight_streams: "
            f"{sorted(conflicts)}"
        )
    # DP2 B2/B4 (one row owner): each rank projects only its own group's held
    # shard of the KDA input projection, for every row (``swap_kda_projections``).
    kda_shard_swap = attention_dp_ownership and sp_owners == 1
    stream_config = {
        **DP_WEIGHT_STREAMS.get(attention_dp, {}),
        **(_dp_dcp_weight_streams(attention_dp, group_rows) if attention_dp > 1 else {}),
        **({"k_projection": (attention_dp, 1)} if kda_shard_swap else {}),
        **default_streams,
        **WEIGHT_STREAMS,
        **projection_streams,
    }
    cache_handoff_head_count = max(
        3, 8 // math.gcd(mla_cache_batch_tile_size, commit_rows, 8)
    )
    hidden_size = prefix.shape[-1]
    # DP row ownership keeps its rows' snapshots block-major at every batch.
    blocks_block_major = batch_size > 8 or attention_dp_ownership
    # Above B16 each row owner keeps only its own rows' snapshots.
    blocks_owned = batch_size > 16
    # Above B16 with token-scattered attention and MoE outputs, the
    # whole-batch attention and mixture outputs are never written.
    compact_outputs = batch_size > 16 and attention_token_scatter and moe_token_scatter_output
    # DP row ownership: a rank's residual stream (prefix and snapshots) is
    # only its own group_rows_per_pair rows, held from row 0.
    block_rows_held = (
        dp_held_rows if attention_dp_ownership
        else owned_rows if blocks_owned else batch_size
    )
    # One copy of each per-owner row region with a traced owner index (program
    # size O(1) in the owner count). BF16 row slices at a traced offset must be
    # provably 16-aligned, so owners need a multiple of 16 rows (B64 at sp=4,
    # B128 at sp=8); smaller owners use unrolled copies.
    owner_generic = blocks_owned and owned_rows % 16 == 0
    # The trace-time arena manager (``VmemPool``) places the router, attention
    # input projections, routed-expert slots and layer 0's dense slabs; under
    # DP the router reuses the drained stream slots. "legacy" keeps the fixed
    # DP layout.
    pool_managed = POOLED_WEIGHTS and (
        attention_dp == 1 or attention_schedule != "legacy"
    )
    if blocks_block_major and not sequence_parallel_pre_attention:
        raise ValueError("Above eight rows the decoder needs sequence-parallel pre-attention")
    persist_kda_state = batch_size <= 4 and attention_dp == 1
    # Above B16 the KDA state streams through row-chunk buffers.
    stream_kda_state = batch_size > 16
    kda_row_chunk = _kda_row_chunk(batch_size, attention_dp)
    slot_reads = state_slots is not None
    if slot_reads and not (
            attention_dp == 1 and (select_state_row or not persist_kda_state)
            or attention_dp == 2 and batch_size in (2, 4, 8, 16, 32, 64, 96, 128)):
        raise ValueError("state_slots needs DP1 with state_row or batch > 4, or DP2 at B2..B128")
    kda_inplace = kda_inplace_writes(batch_size=batch_size, attention_dp=attention_dp,
                                     sequence_rows=sequence_rows, slot_reads=slot_reads,
                                     state_row=state_row)
    kda_inplace_period = states[0].shape[0] // attention_dp if kda_inplace else None
    overlap_kda_state_reads = batch_size in (6, *SCATTER_BATCHES) or attention_dp > 1
    overlap_kda_state_writes = batch_size in (6, *SCATTER_BATCHES) or attention_dp > 1

    def shape(value):
        return jax.ShapeDtypeStruct(value.shape, value.dtype)

    local_weights = {
        name: jax.ShapeDtypeStruct(value.shape[1:], value.dtype)
        for name, value in weights.items()
    }

    def pooled_shape(name):
        """Pool footprint of a layer weight: a ring stream's slot region
        (depth x its largest chunk along the streamed axis), else the weight."""
        shape = local_weights[name].shape
        if name not in STREAMED_WEIGHTS or name not in stream_config:
            return shape
        chunks, depth = stream_config[name]
        if chunks == depth:
            return shape
        axis = 1 if STREAMED_WEIGHTS[name][0] == "columns" else 0
        largest = _chunk_sizes(shape[axis], chunks, 128 if axis == 1 else 16)[0]
        return tuple(depth * largest if index == axis else size for index, size in enumerate(shape))

    pool_weights = {
        name: jax.ShapeDtypeStruct(pooled_shape(name), value.dtype)
        for name, value in local_weights.items()
    }
    mla_scoped_scratch = _mla_scoped_scratch(
        local_weights,
        # Attention DP runs MLA on the group's rows, one shard at a time.
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
        group_pending_routes=(
            batch_size in (6, *SCATTER_BATCHES)
            and residual_reduction_order == "relaxed"
        ),
    )
    def body(position_ref, state_row_ref, *refs):
        table_ref, *refs = refs
        slots_ref = None
        if slot_reads:
            slots_ref, *refs = refs
        active_rows = None
        if live_rows is not None:
            live_ref, *refs = refs
            # Stack scalar integers before comparison: Mosaic cannot concatenate
            # scalar predicates into a vector through bitcast_vreg.
            active_rows = jnp.stack([live_ref[b] for b in range(batch_size)]) != 0
        prefix_ref, embedding_ref, *refs = refs
        layer_weights = dict(zip(NAMES, refs[: len(NAMES)], strict=True))
        output_residual_ref, final_normalization_ref = refs[
            len(NAMES) : len(NAMES) + len(EPILOGUE)
        ]
        dense_start = len(NAMES) + len(EPILOGUE)
        dense_refs = dict(
            zip(dense_names, refs[dense_start : dense_start + len(dense_names)], strict=True)
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
                dp_query_latent_vmem,
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
        gather_wire = None
        if b4_attention_gather:
            *common, gather_wire = common
        communication_workspace = AttentionCommunicationWorkspace(
            *common, attention_gather_vmem=gather_wire
        )
        if compact_outputs:
            # Attention's whole-batch output argument is shape-checked but
            # unused with token scatter; the normalized input has its shape.
            attention_vmem = normalized_vmem

        # In sequence-row mode the chained recurrence reads only the first
        # row's input state; with a state row the accepted row of the
        # previous step is read directly into that slot.
        selected_state_row = state_row_ref[0] if select_state_row else None

        def kda_state_source(ref, state_index):
            if select_state_row:
                row = slots_ref[selected_state_row] if slot_reads else selected_state_row
                return ref.at[pl.ds(row, 1), state_index]
            return ref.at[:, state_index]

        def kda_state_reads(ref, vmem, semaphore, state_index):
            """The copies of a layer's input KDA states into ``vmem``: one,
            or one per row from its slot (``state_slots``)."""
            if not slot_reads or select_state_row:
                return (tpu.make_async_copy(kda_state_source(ref, state_index),
                                            kda_state_destination(vmem), semaphore), )
            return tuple(tpu.make_async_copy(ref.at[pl.ds(slots_ref[b], 1), state_index],
                                             vmem.at[pl.ds(b, 1)], semaphore)
                         for b in range(batch_size))

        def kda_state_destination(vmem):
            return vmem.at[pl.ds(0, 1)] if select_state_row else vmem

        def pool_view(name, base=0):
            """The pooled view of one layer weight at its _pool_layout offset
            (from ``base``: the MLA output projection is placed relative to its
            layer's attention weights)."""
            value = pool_weights[name]
            size = math.prod(value.shape) * value.dtype.itemsize
            return pool_alias.view(
                projection_expert_pool_vmem.at[
                    pl.ds((base + pool_layout[name]) // 128, size // 128), :
                ],
                value.shape,
                value.dtype,
            )

        def pool_views(names, *, experts=False, base=0):
            views = []
            offset = base
            for name in names:
                value = pool_weights[name]
                logical_shape = value.shape[1:] if experts else value.shape
                view_shape = (4, *logical_shape) if experts else logical_shape
                size = math.prod(view_shape) * value.dtype.itemsize
                storage_size = pool_alias.storage_size(view_shape, value.dtype)
                views.append(
                    pool_alias.view(
                        projection_expert_pool_vmem.at[
                            pl.ds(offset // 128, size // 128), :
                        ],
                        view_shape,
                        value.dtype,
                    )
                )
                offset += storage_size
            return views

        def hot_fp8_views():
            """fp8 gate/up, down and column-scale views over MXFP4 slot pairs (0, 1) and (2, 3)."""
            base = _HOT_BASE[-1] if _HOT_BASE else 0
            names = ("expert_gu", "expert_down", "expert_gus", "expert_ds")
            sizes = [pool_alias.storage_size((4, *pool_weights[n].shape[1:]), pool_weights[n].dtype) for n in names]
            starts = [base + sum(sizes[:i]) for i in range(4)]
            gate_up_rows = pool_weights["expert_gu"].shape[1] * 8
            down_rows = pool_weights["expert_down"].shape[1] * 8
            gate_up_columns = pool_weights["expert_gu"].shape[2]
            down_columns = pool_weights["expert_down"].shape[2]

            def view(start, shape, dtype):
                size = math.prod(shape) * jnp.dtype(dtype).itemsize
                return pool_alias.view(
                    projection_expert_pool_vmem.at[pl.ds(start // 128, size // 128), :], shape, jnp.dtype(dtype)
                )

            return [
                SimpleNamespace(
                    gu=view(starts[0] + k * (sizes[0] // 2), (gate_up_rows, gate_up_columns), jnp.float8_e4m3fn),
                    down=view(starts[1] + k * (sizes[1] // 2), (down_rows, down_columns), jnp.float8_e4m3fn),
                    gs=view(starts[2] + k * (sizes[2] // 2), (1, gate_up_columns), jnp.float32),
                    ds=view(starts[3] + k * (sizes[3] // 2), (1, down_columns), jnp.float32),
                )
                for k in range(2)
            ]

        def attention_views(base=0):
            """The KDA and MLA input projections of one layer from ``base``."""
            return pool_views(KDA[:4], base=base), pool_views(MLA[:2], base=base)

        def expert_views(base=0):
            _HOT_BASE[:] = [base]
            if alias_expert_scales:
                views = pool_views(
                    ("expert_gu", "expert_down", "expert_gus", "expert_ds"),
                    experts=True,
                    base=base,
                )
                return views[0], views[2], views[1], views[3]
            return pool_views(("expert_gu", "expert_down"), experts=True, base=base)

        # With the arena manager, where each layer's weights land (and when
        # their copies start) follows from the order of requests and releases.
        pool = (
            VmemPool(projection_expert_pool_vmem, pool_layout["arena_bytes"], report=_config().pool_report)
            if pool_managed
            else None
        )

        def arena_router(offset):
            return pool.view(offset, (1, hidden_size, 512), jnp.dtype(jnp.bfloat16))

        def request_router(index, predicate=None):
            """Queue layer ``index``'s router copy into the arena."""
            pool.request(
                "router",
                pool_alias.storage_size((1, hidden_size, 512), jnp.dtype(jnp.bfloat16)),
                lambda offset, index: _router_copy(
                    layer_weights["router"].at[index - 1],
                    arena_router(offset),
                    router_semaphores,
                    False,
                    initialize_padding=False,
                ),
                index=index,
                predicate=predicate,
            )

        def request_attention(index, predicate=None):
            """Queue layer ``index``'s attention input projections."""
            pool.request(
                "attention",
                pool_layout["attention_bytes"],
                lambda offset, index: prefetch_attention(index, base=offset),
                index=index,
                predicate=predicate,
            )
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
            # DP group d attends, of the rows owner p holds, the r rows
            # tokens_per_pair * p + d * r + j (group row p * r + j).
            data_group = jax.lax.axis_index("tp") // (32 // attention_dp)
            for row in range(group_rows):
                pair_index, within = divmod(row, group_rows_per_pair)
                group_positions_smem[row] = position_ref[
                    tokens_per_pair * pair_index + data_group * group_rows_per_pair + within
                ]
        def for_dp_owned_rows(run):
            """``run(first)`` with this rank's first batch row (static per
            row owner and DP group)."""
            if sp_owners == 1:
                run(0)
                return
            rank = jax.lax.axis_index("tp")
            pair, group = (rank % 8) // (8 // sp_owners), rank // (32 // attention_dp)
            for owner in range(sp_owners):
                for data_group in range(attention_dp):
                    @pl.when((pair == owner) & (group == data_group))
                    def owned_region(owner=owner, data_group=data_group):
                        run(owner * tokens_per_pair + data_group * group_rows_per_pair)

        # The prefix and embedding inputs and the hidden output stay in HBM:
        # one DMA each at entry/exit instead of whole-batch VMEM windows live
        # for the entire kernel.
        def load_prefix(staged, semaphore):
            copy = tpu.make_async_copy(prefix_ref, staged, semaphore)
            copy.start()
            copy.wait()
            if attention_dp_ownership:
                def take_owned_rows(first):
                    prefix_vmem[:dp_held_rows] = staged[first : first + dp_held_rows]

                for_dp_owned_rows(take_owned_rows)

        if attention_dp_ownership:
            pl.run_scoped(load_prefix, tpu.VMEM(prefix_ref.shape, prefix_ref.dtype),
                          tpu.SemaphoreType.DMA)
        else:
            pl.run_scoped(functools.partial(load_prefix, prefix_vmem), tpu.SemaphoreType.DMA)
        blocks_vmem[...] = jnp.zeros(blocks_vmem.shape, blocks_vmem.dtype)
        def initialize_blocks(embedding_ref):
            if attention_dp_ownership:
                def initialize_owned_blocks(first):
                    for local in range(dp_held_rows):
                        blocks_vmem[local // 2, pl.ds((local % 2) * 8, 1)] = embedding_ref[
                            pl.ds(first + local, 1)
                        ].astype(jnp.bfloat16)

                for_dp_owned_rows(initialize_owned_blocks)
            elif blocks_owned:
                owner_of_rank = (jax.lax.axis_index("tp") % 8) // (8 // sp_owners)
                for owner_index in range(sp_owners):
                    @pl.when(owner_of_rank == owner_index)
                    def initialize_owned_blocks(owner_index=owner_index):
                        for local in range(owned_rows):
                            blocks_vmem[local // 2, pl.ds((local % 2) * 8, 1)] = embedding_ref[
                                pl.ds(owner_index * owned_rows + local, 1)
                            ].astype(jnp.bfloat16)
            elif blocks_block_major:
                for row in range(batch_size):
                    blocks_vmem[row // 2, pl.ds((row % 2) * 8, 1)] = embedding_ref[
                        pl.ds(row, 1)
                    ].astype(jnp.bfloat16)
            else:
                blocks_vmem[:, 0] = embedding_ref[...].astype(jnp.float32)

        def staged_blocks(embedding_vmem, semaphore):
            copy = tpu.make_async_copy(embedding_ref, embedding_vmem, semaphore)
            copy.start()
            copy.wait()
            initialize_blocks(embedding_vmem)

        # With layer zero included, its pre-mixture has no active snapshots.
        # That layer seeds snapshot zero from prefix before any later read.
        # Keep all snapshot slots zero-initialized so masked arithmetic stays
        # finite; only omit the redundant embedding DMA and row insertion.
        if not (include_layer_zero and attention_dp == 2
                and batch_size in (64, 128) and sequence_rows == 1):
            pl.run_scoped(
                staged_blocks,
                tpu.VMEM(embedding_ref.shape, embedding_ref.dtype),
                tpu.SemaphoreType.DMA,
            )

        def owner_indices(pair):
            return (pair,) if owner_generic else range(sp_owners)

        def owner_when(pair, owner):
            if owner_generic:
                return lambda function: function()
            return pl.when(pair == owner)

        def owner_start(start):
            return pl.multiple_of(start, 16) if owner_generic else start

        def owner_block_row(global_row, local_row):
            # Above B16 each owner holds only its own rows' snapshots.
            return local_row if owner_generic else global_row

        def owner_prefix_row(global_row, row_slice, local_row):
            if owner_generic:
                return prefix_vmem[row_slice][local_row : local_row + 1]
            return prefix_vmem[pl.ds(global_row, 1)]

        def block_rows(row, count=8):
            """The residual snapshots of one batch row, ``[count, hidden]``."""
            if blocks_block_major:
                row = row % block_rows_held  # owned rows are held locally
                return blocks_vmem[row // 2, pl.ds((row % 2) * 8, count)]
            return blocks_vmem[row] if count == 8 else blocks_vmem[row, :count]

        def residual(destination, folded, active_block_count):
            _residual_mixture_by_row(
                blocks_vmem[...],
                prefix_vmem[...],
                destination,
                folded,
                active_block_count,
                batch_size=batch_size,
                epsilon=eps,
                relaxed_reduction_order=(
                    residual_reduction_order == "relaxed"
                ),
            )

        def store_mixed_rows(compute_row, destination):
            """Write the destination's owned rows. Above two rows per pair
            (B16) each row runs in its own region: unrolled rows would keep
            every row's 8 x hidden FP32 blocks live and spill. DP2 B4's four
            rows (one owner) stay unrolled."""
            rows = destination.shape[0]
            if rows <= 2 or sp_owners == 1:
                destination[...] = jnp.concatenate(
                    [compute_row(local_row) for local_row in range(rows)],
                    axis=0,
                )
                return
            for local_row in range(rows):
                # A runtime condition (never false) so the row is its own region.
                @pl.when(communication_workspace.reduction_phase[0] >= 0)
                def store_row(local_row=local_row):
                    destination[pl.ds(local_row, 1)] = compute_row(local_row)

        def normalize(destination, scale):
            value = destination[...].astype(jnp.float32)
            destination[...] = (
                value
                * jax.lax.rsqrt(
                    jnp.mean(value * value, axis=1, keepdims=True) + eps
                )
                * scale[...]
            ).astype(jnp.bfloat16)

        dp_owned_vmem = _bf16_rows_view(
            communication_workspace.attention_output_vmem,
            dp_held_rows,
            hidden_size,
        ) if attention_dp_ownership else None

        sp8_residual = (attention_dp_ownership and attention_dp == 2
                        and batch_size == 128 and sequence_rows == 1)
        # Expert/projection ABIs retain their existing pair ownership. Only
        # mixture rows and the following gathers use eight independent owners.
        sp8_rows_vmem = dp_owned_vmem if sp8_residual else None

        def sp8_owned_value():
            value = sp8_rows_vmem[...]
            return jnp.where(jax.lax.axis_index("tp") % 2 == 0,
                             value[:8], value[8:16])

        def mix_dp_rows(folded, active_blocks, *, layer_index,
                        attention_output=False, start_block=False):
            rows = slice(0, dp_held_rows)
            if attention_output:
                prefix_vmem[rows] = jnp.where(
                    layer_index % 12 == 0,
                    dp_owned_vmem[...],
                    prefix_vmem[rows] + dp_owned_vmem[...],
                ).astype(jnp.bfloat16)

            def mix_row(row):
                return _residual_mixture_row(
                    block_rows(row),
                    prefix_vmem[row : row + 1],
                    folded,
                    active_blocks,
                    epsilon=eps,
                    relaxed_reduction_order=residual_reduction_order == "relaxed",
                )

            if sp8_residual:
                # RMS keeps a native 16-row geometry; initialize unused rows
                # rather than reading undefined scratch on the first layer.
                sp8_rows_vmem[...] = jnp.zeros((16, hidden_size), jnp.bfloat16)
                for lane in range(2):
                    @pl.when(jax.lax.axis_index("tp") % 2 == lane)
                    def mix_half(lane=lane):
                        for local_row in range(8):
                            @pl.when(communication_workspace.reduction_phase[0] >= 0)
                            def store_row(local_row=local_row):
                                row = lane * 8 + local_row
                                # Index the full transformed view directly,
                                # avoiding an unsupported nested half-row view.
                                sp8_rows_vmem[pl.ds(row, 1)] = mix_row(row)
            else:
                store_mixed_rows(mix_row, normalized_vmem if start_block and kda_shard_swap else dp_owned_vmem)
            if start_block:
                @pl.when(layer_index % 12 == 0)
                def snapshot():
                    for row in range(dp_held_rows):
                        half = blocks_vmem.at[row // 2, pl.ds((row % 2) * 8, 8)]
                        mask = jax.lax.broadcasted_iota(jnp.int32, half.shape, 0)
                        half[...] = jnp.where(
                            mask == layer_index // 12, prefix_vmem[row : row + 1], half[...],
                        )

        def normalize_dp_rows(scale, *, split=False):
            if split:
                # Preserve the 16-row normalization geometry. Unowned rows
                # are not communicated or consumed by any later computation.
                normalize(sp8_rows_vmem, scale)
                return
            if batch_size != 64:
                normalize(dp_owned_vmem, scale)
                return
            # Normalize in a native 16-row tile at the group's row lanes
            # (owned_rows_vmem is free until the following gather).
            group = jax.lax.axis_index("tp") // (32 // attention_dp)
            for data_group in range(attention_dp):
                @pl.when(group == data_group)
                def stage_native_rows(data_group=data_group):
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
                def select_normalized_rows(data_group=data_group):
                    first = data_group * group_rows_per_pair
                    dp_owned_vmem[...] = native_rows[first:first + group_rows_per_pair]

        def swap_kda_projections(stream):
            """DP2 B2/B4's shard swap: this rank projects its own group's held
            shard for every row, sends the other group's rows to its
            counterpart there (which projects the other shard) and gets this
            group's rows of that shard back. Returns ``rows(order)``: the
            group's rows of the own shard's input projection (order 0), then,
            after the exchange, of the other shard's (order 1)."""
            rank = jax.lax.axis_index("tp")
            group = rank // (32 // attention_dp)
            projected = stream.dot(normalized_vmem[...], communication_workspace.reduction_phase[0] >= 0)
            projected = projected.astype(jnp.bfloat16)
            own = jnp.where(group == 0, projected[:group_rows], projected[group_rows:])
            other = jnp.where(group == 0, projected[group_rows:], projected[:group_rows])
            used = group_rows * projected.shape[1] // 128
            # The BF16 view of the FP32 lanes slices in row pairs: pad B2's
            # 13-row payload.
            payload = used + used % 2
            local = communication_workspace.shared_local_reduction_vmem
            send = local.at[4].bitcast(jnp.bfloat16).at[pl.ds(0, payload), :]
            receive = local.at[5].bitcast(jnp.bfloat16).at[pl.ds(0, payload), :]
            wire = other.reshape(used, 128)
            if payload != used:
                wire = jnp.concatenate([wire, jnp.zeros((payload - used, 128), jnp.bfloat16)])
            send[...] = wire
            copy = tpu.make_async_remote_copy(
                send, receive,
                communication_workspace.send_semaphores.at[14],
                communication_workspace.receive_semaphores.at[14],
                device_id=(rank ^ (32 // attention_dp),), device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()

            def rows(order):
                if order == 0:
                    return own
                copy.wait()
                return receive[...][:used].reshape(own.shape)

            return rows

        def gather_dp_rows(destination, *, split=False):
            if split:
                _gather_sp8_dp_moe_rows(
                    sp8_owned_value(), owned_rows_vmem, destination,
                    communication_workspace.shared_local_reduction_vmem,
                    communication_workspace.send_semaphores,
                    communication_workspace.receive_semaphores)
                return
            # Restore the router's owned input as well as the complete MoE
            # input. No arithmetic occurs after normalization. Scratch lanes
            # 4..7 for DP transport are disjoint from local-gather lanes 0..3.
            if sp_owners == 1:
                destination[...] = dp_owned_vmem[...]
                return
            if dp_moe_gather == "split":
                _gather_dp_moe_rows(
                    dp_owned_vmem[...], owned_rows_vmem, destination,
                    communication_workspace.shared_local_reduction_vmem,
                    communication_workspace.send_semaphores,
                    communication_workspace.receive_semaphores,
                    attention_dp,
                )
                return
            with _phase("t10a_dp_normalized_gather"):
                _gather_attention_dp_rows(
                    dp_owned_vmem[...], owned_rows_vmem,
                    communication_workspace.shared_local_reduction_vmem,
                    communication_workspace.send_semaphores,
                    communication_workspace.receive_semaphores,
                    attention_dp,
                )
            with _phase("t10b_moe_local_gather"):
                _gather_local_pair_rows(
                    owned_rows_vmem, destination,
                    communication_workspace.send_semaphores,
                    communication_workspace.receive_semaphores,
                    communication_workspace.shared_local_reduction_vmem,
                    owners=sp_owners,
                )

        def prefetch_norms(index):
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
                    layer_weights[name].at[index],
                    destination,
                    norm_semaphores.at[slot],
                ).start()

        def storage(name):
            value = pool_weights[name]
            return pool_alias.storage_size(value.shape, value.dtype)

        def weight_stream(name, source, full, *, offset=None):
            """The stream that owns ``name``'s copy, or None when the weight is
            one chunk (plain whole-weight copy)."""
            axis = STREAMED_WEIGHTS[name][0]
            chunks, depth = stream_config.get(name, (1, 1))
            if chunks == 1:
                return None
            base = _stream_base(name, depth)
            order = None
            if axis == "rows":
                # Output projections stream in rank-pair feature-row halves,
                # own rows first: per shard, all own halves before the peer's.
                if chunks != 2 * attention_dp:
                    raise ValueError(f"{name} streams as its {2 * attention_dp} pair halves")
                lane = jax.lax.axis_index("tp") % 2
                order = lambda position: (  # noqa: E731
                    (position % attention_dp) * 2 + (lane ^ (position // attention_dp))
                )
            extent = source.shape[1 if axis == "columns" else 0]
            unit = 128 if axis == "columns" else 16
            if axis == "columns" and attention_dp > 1:
                # Chunks never straddle two held shards.
                if chunks % attention_dp:
                    raise ValueError(f"{name}: chunks must be a multiple of the attention DP degree")
                sizes = _chunk_sizes(extent // attention_dp, chunks // attention_dp, unit) * attention_dp
            else:
                sizes = _chunk_sizes(extent, chunks, unit)
            stream = ChunkedWeightStream(
                source, full, stream_semaphores, axis=axis, sizes=sizes,
                depth=depth, base=base, order=order,
                pool=None if offset is None else projection_expert_pool_vmem, offset=offset,
            )
            stream.shards = attention_dp
            return stream

        @_scoped("t9_dp_project")
        def project_attention_dp(weight_view, stream):
            """Attention DP: project every shard's gated features into this
            rank's owned rows (the token-scatter attention output)."""
            owned = _bf16_rows_view(
                communication_workspace.attention_output_vmem,
                dp_held_rows if attention_dp_ownership else tokens_per_pair,
                hidden_size,
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
                    keep_dp_rows=attention_dp_ownership,
                    owners=sp_owners,
                ),
                tpu.VMEM((group_rows, 8192), jnp.float32),
            )

        def query_down_weight_stream(state_index, base=0):
            """MLA's q-down projection (at the attention weights' ``base``) as a
            column stream, if configured."""
            return weight_stream(
                "m_qa", layer_weights["m_qa"].at[state_index], attention_views(base)[1][0],
                offset=base if POOLED_WEIGHTS else None,
            )

        def kda_streams(state_index, base=0):
            kda_views = attention_views(base)[0]
            source = layer_weights["k_projection"].at[state_index]
            if kda_shard_swap:
                # One copy of this rank's own group's held shard.
                width = source.shape[1] // attention_dp
                group = jax.lax.axis_index("tp") // (32 // attention_dp)
                input_stream = ChunkedWeightStream(
                    source.at[:, pl.ds(pl.multiple_of(group * width, 128), width)], kda_views[1],
                    stream_semaphores, axis="columns", sizes=(width,), depth=1,
                )
            else:
                input_stream = weight_stream(
                    "k_projection", source, kda_views[1], offset=base + storage("k_gate"),
                )
            streams = {
                "input": input_stream,
                "output": weight_stream(
                    "k_o", layer_weights["k_o"].at[state_index], kda_views[3],
                    offset=base + storage("k_gate") + storage("k_projection") + storage("k_fb"),
                ),
            }
            return {key: stream for key, stream in streams.items() if stream is not None}

        def prefetch_attention(index, base=0):
            kda_projection_vmem, mla_projection_vmem = attention_views(base)

            def prefetch_kda():
                state_index = index - index // 4
                # Semaphore slots follow execution dependency order; the VMEM
                # tuple follows schema order, so permute both together.
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

            def prefetch_mla():
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
                (index % 4 == 3) | (index == 92),
                prefetch_mla,
                prefetch_kda,
            )

        def prefetch_auxiliary(index):
            @pl.when((index % 4 != 3) & (index != 92))
            def prefetch_kda_auxiliary():
                state_index = index - index // 4
                sources = tuple(
                    layer_weights[name].at[state_index] for name in KDA[4:]
                )
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
                        source,
                        destination,
                        auxiliary_semaphores.at[slot],
                    ).start()

        def prefetch_metadata(index, *, mixture_metadata=True):
            buffer_slot = index % 2

            @pl.when((index % 4 == 3) | (index == 92))
            def prefetch_mla_norms():
                for slot, name in enumerate(("m_qnorm", "m_knorm")):
                    tpu.make_async_copy(
                        layer_weights[name].at[index // 4],
                        metadata_vmem[slot].at[buffer_slot],
                        metadata_semaphores.at[buffer_slot, slot],
                    ).start()

            if not mixture_metadata:
                return  # layer 0 has no mixture of experts (dense MLP instead)
            for slot, name in enumerate(("router_bias", "latent_norm"), 2):
                tpu.make_async_copy(
                    layer_weights[name].at[index - 1],
                    metadata_vmem[slot].at[buffer_slot],
                    metadata_semaphores.at[buffer_slot, slot],
                ).start()

        def prefetch_router(index, *, initialize_padding=False):
            if pool_managed:
                return  # queued in the arena
            _router_copy(
                layer_weights["router"].at[index - 1],
                router_vmem,
                router_semaphores,
                False,
                initialize_padding=initialize_padding,
            )

        # The lane-1 projection uses 384 of its 512 MXU columns. This persistent
        # tail is never written elsewhere, so initialize it once rather than on
        # every layer prefetch.
        if include_layer_zero:
            # Layer 0 has no router; layer 1's is prefetched after the dense
            # MLP.
            if not pool_managed:
                @pl.when(jax.lax.axis_index("tp") % 2 == 1)
                def initialize_router_padding():
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
        if pool is not None and not include_layer_zero:
            request_router(jnp.int32(1))

        # Layer 0 (dense MLP) is peeled out of the loop as straight-line code
        # so the loop body stays the routed-expert body without a conditional
        # around the mixture (a control-flow region there costs the scheduler
        # the overlap of the mixture's start and tail with attention work).
        def layer(layer_index, *, dense_layer=False):
            # This layer's attention projections: placed by the arena manager
            # when its copy was issued, else at the start of the pool.
            attention_base = pool.offset("attention") if pool is not None else 0
            kda_projection_vmem, mla_projection_vmem = attention_views(attention_base)

            def capture_aux_hidden(source=None, gather_owned_rows=False):
                source = prefix_vmem if source is None else source
                if gather_owned_rows and attention_dp_ownership:
                    # DP ranks hold only their own prefix rows.
                    source = normalized_vmem
                for aux_slot, aux_layer in enumerate(aux_hidden_layers):
                    # Layer id L names the residual stream entering layer L,
                    # which is the prefix after layer L - 1 completes (or, with
                    # aux_capture="attention", after its attention).
                    # The rows go to HBM so the capture costs no VMEM.
                    @pl.when(layer_index == aux_layer - 1)
                    def capture(aux_slot=aux_slot):
                        if gather_owned_rows:
                            # Sequence-parallel layers keep only this pair's
                            # prefix rows; complete them for the capture.
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

            norm_copies = [
                tpu.make_async_copy(
                    layer_weights[name].at[layer_index],
                    destination,
                    norm_semaphores.at[index],
                )
                for index, (name, destination) in enumerate((
                    ("attn_res", attention_residual_vmem),
                    ("mlp_res", mlp_residual_vmem),
                    ("attn_norm", attention_norm_vmem),
                    ("ffn_norm", ffn_norm_vmem),
                ))
            ]
            norm_copies[0].wait()
            norm_copies[2].wait()
            layer_scope = [_phase_start("q1_pre_mix")]
            if attention_dp_ownership:
                mix_dp_rows(
                    attention_residual_vmem, (layer_index - 1) // 12 + 1,
                    layer_index=layer_index, start_block=True,
                )
            elif sequence_parallel_pre_attention:
                rank = jax.lax.axis_index("tp")
                pair = (rank % 8) // (8 // sp_owners)
                for owner_pair in owner_indices(pair):
                    @owner_when(pair, owner_pair)
                    def prepare_owned_attention_rows():
                        row_offset = owner_pair * owned_rows
                        row_slice = pl.ds(owner_start(row_offset), owned_rows)

                        def mix_row(local_row):
                            global_row = row_offset + local_row
                            return _residual_mixture_row(
                                block_rows(owner_block_row(global_row, local_row)),
                                owner_prefix_row(global_row, row_slice, local_row),
                                attention_residual_vmem,
                                (layer_index - 1) // 12 + 1,
                                epsilon=eps,
                                relaxed_reduction_order=(
                                    residual_reduction_order == "relaxed"
                                ),
                            )

                        store_mixed_rows(mix_row, owned_rows_vmem)

                        @pl.when((layer_index % 12 == 0) if blocks_block_major else False)
                        def start_owned_residual_block_major():
                            for local_row in range(owned_rows):
                                global_row = row_offset + local_row
                                row = owner_block_row(global_row, local_row) % block_rows_held
                                half = blocks_vmem.at[row // 2, pl.ds((row % 2) * 8, 8)]
                                block_mask = (
                                    jax.lax.broadcasted_iota(jnp.int32, half.shape, 0)
                                    == layer_index // 12
                                )
                                half[...] = jnp.where(
                                    block_mask, owner_prefix_row(global_row, row_slice, local_row), half[...]
                                )

                        @pl.when(False if blocks_block_major else (layer_index % 12 == 0))
                        def start_owned_residual_block():
                            block_mask = (
                                jax.lax.broadcasted_iota(
                                    jnp.int32,
                                    blocks_vmem[row_slice].shape,
                                    1,
                                )
                                == layer_index // 12
                            )
                            blocks_vmem[row_slice] = jnp.where(
                                block_mask,
                                prefix_vmem[row_slice][
                                    :, None, :
                                ].astype(jnp.float32),
                                blocks_vmem[row_slice],
                            )
            else:
                residual(
                    normalized_vmem,
                    attention_residual_vmem,
                    (layer_index - 1) // 12 + 1,
                )

                @pl.when(layer_index % 12 == 0)
                def start_residual_block():
                    block_mask = (
                        jax.lax.broadcasted_iota(
                            jnp.int32, blocks_vmem.shape, 1
                        )
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
                # Every row is the swapped projection's input; attention reads
                # the group's.
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
                            sp8_owned_value(), group_input_vmem,
                            communication_workspace.shared_local_reduction_vmem,
                            communication_workspace.send_semaphores,
                            communication_workspace.receive_semaphores)
                    else:
                        _gather_local_pair_rows(
                            dp_owned_vmem, group_input_vmem,
                            communication_workspace.send_semaphores,
                            communication_workspace.receive_semaphores,
                            communication_workspace.shared_local_reduction_vmem,
                            packed=True, fp8="t0" in FP8_WIRES)
            elif sequence_parallel_pre_attention and attention_dp > 1:
                # The group's attention input: each local pair contributes its
                # owned row 2p + d.
                normalize(owned_rows_vmem, attention_norm_vmem)
                t0 = _phase_start("t0_dp_group_gather")
                data_group = jax.lax.axis_index("tp") // (32 // attention_dp)
                if group_rows_per_pair == 1:
                    owned = owned_rows_vmem[...]
                    selected = owned[0:1]
                    for group in range(1, attention_dp):
                        selected = jnp.where(data_group == group, owned[group : group + 1], selected)
                    group_input_vmem[0:1] = selected
                    _gather_local_pair_rows(
                        group_input_vmem.at[pl.ds(0, 1)],
                        group_input_vmem,
                        communication_workspace.send_semaphores,
                        communication_workspace.receive_semaphores,
                        communication_workspace.shared_local_reduction_vmem,
                    )
                elif "gather" not in DEBUG_SKIP:
                    for group in range(attention_dp):
                        @pl.when(data_group == group)
                        def gather_group_rows(group=group):
                            _gather_local_pair_rows(
                                owned_rows_vmem.at[
                                    group * group_rows_per_pair : (group + 1) * group_rows_per_pair
                                ],
                                group_input_vmem,
                                communication_workspace.send_semaphores,
                                communication_workspace.receive_semaphores,
                                communication_workspace.shared_local_reduction_vmem,
                                # B64 DP2 selects eight rows from a native
                                # 16-row BF16 owner buffer. Pack this half-tile
                                # view so its DMA byte count/layout is explicit.
                                packed=group_rows_per_pair == 8,
                            )
                _phase_end(t0)
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

            def release_kda_inputs():
                if attention_schedule == "overlap":
                    assert pool is not None
                    # Every head shard has consumed the gate/input/feature-gate
                    # weights. The output ring still occupies the suffix, so
                    # retain it while the queued router DMA reuses the prefix.
                    pool.release_prefix(
                        "attention",
                        storage("k_gate") + storage("k_projection") + storage("k_fb"),
                    )

            def run_delta_attention():
                state_index = layer_index - layer_index // 4
                auxiliary_sources = tuple(
                    layer_weights[name].at[state_index] for name in KDA[4:]
                )
                pending = tuple(
                    tpu.make_async_copy(
                        source,
                        destination,
                        auxiliary_semaphores.at[index],
                    )
                    for index, (source, destination) in enumerate(
                        zip(auxiliary_sources, auxiliary_vmem[:4], strict=True)
                    )
                )
                for copy in pending:
                    copy.wait()

                def execute(
                    history_vmem, state_vmem, state_semaphores, state_reads=()
                ):
                    state_writes = (
                        tpu.make_async_copy(
                            history_vmem,
                            convolution_output_ref.at[:, state_index],
                            state_semaphores.at[0],
                        ),
                        tpu.make_async_copy(
                            state_vmem,
                            recurrent_output_ref.at[:, state_index],
                            state_semaphores.at[1],
                        ),
                    )

                    def wait_for_state():
                        for copy in state_reads:
                            copy.wait()

                    def start_state_writes():
                        for copy in state_writes:
                            copy.start()

                    if attention_dp > 1:
                        streams = kda_streams(state_index, attention_base)
                        swapped = swap_kda_projections(streams["input"]) if kda_shard_swap else None
                        for order in range(attention_dp):
                            shard = order
                            if kda_shard_swap:
                                # The swap runs this rank's own shard first.
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
                                    # Per-shard filter, decay and time-step bias; shared norm.
                                    *(auxiliary_vmem[index].at[shard] for index in range(3)),
                                    auxiliary_vmem[3],
                                    pool_semaphores,
                                ),
                                convolution_history_ref=history_vmem.at[rows],
                                recurrent_state_ref=state_vmem.at[rows],
                                communication_workspace=communication_workspace,
                                epsilon=eps,
                                synchronize_devices=False,
                                weights_are_prefetched=True,
                                wait_for_state=(
                                    wait_for_state
                                    if overlap_kda_state_reads and order == 0
                                    else None
                                ),
                                on_state_updated=(
                                    start_state_writes
                                    if overlap_kda_state_writes and order == attention_dp - 1
                                    else None
                                ),
                                sequence_rows=sequence_rows,
                                streams=streams,
                                shard_index=shard,
                                attended_out=dp_attended_vmem.at[shard],
                                gate_features_vmem=dp_compressed_vmem,
                                stage_projections=attention_schedule in ("auto", "overlap"),
                                projected_rows=None if swapped is None else swapped(order),
                                first_call=order == 0,
                            )
                        release_kda_inputs()
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
                                *auxiliary_vmem[:4],
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
                            wait_for_state=(
                                wait_for_state if overlap_kda_state_reads else None
                            ),
                            on_state_updated=(
                                start_state_writes if overlap_kda_state_writes else None
                            ),
                            streams=kda_streams(state_index, attention_base),
                        )
                    if not overlap_kda_state_writes:
                        start_state_writes()
                    if not persist_kda_state:
                        for copy in state_writes:
                            copy.wait()

                if persist_kda_state:
                    state_sources = (
                        kda_state_source(convolution_ref, state_index),
                        kda_state_source(recurrent_ref, state_index),
                    )
                    state_pending = tuple(
                        tpu.make_async_copy(
                            source,
                            destination,
                            auxiliary_semaphores.at[index + 4],
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
                    for copy in state_pending:
                        copy.wait()
                    execute(
                        convolution_history_vmem,
                        recurrent_state_vmem,
                        auxiliary_semaphores,
                    )
                elif stream_kda_state:
                    def execute_streamed_dp(history_buffers, state_buffers, read_semaphores, write_semaphores):
                        streams = kda_streams(state_index, attention_base)
                        chunks_per_shard = group_rows // kda_row_chunk
                        shared_state_stream = None
                        if attention_state_pipeline:
                            # to_attention_dp orders state rows as head shard,
                            # then the group's sequences. One flattened stream
                            # keeps its two-chunk lookahead across those shards
                            # without allocating another state buffer.
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
                                inplace_period=kda_inplace_period,
                            )
                            shared_state_stream.prime()
                        for shard in range(attention_dp):
                            rows = pl.ds(shard * group_rows, group_rows)
                            history_rows = convolution_ref.at[:, state_index]
                            state_rows = recurrent_ref.at[:, state_index]
                            if not slot_reads:
                                history_rows = history_rows.at[rows]
                                state_rows = state_rows.at[rows]
                            if shared_state_stream is None:
                                state_stream = KdaStateStream(
                                    history_rows,
                                    state_rows,
                                    convolution_output_ref.at[:, state_index].at[rows],
                                    recurrent_output_ref.at[:, state_index].at[rows],
                                    history_buffers,
                                    state_buffers,
                                    read_semaphores,
                                    write_semaphores,
                                    chunk=kda_row_chunk,
                                    chunks=chunks_per_shard,
                                    slots=slots_ref.at[rows] if slot_reads else None,
                                    sequence_rows=sequence_rows,
                                    inplace_period=kda_inplace_period,
                                )
                                state_stream.prime()
                            else:
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
                                    *(auxiliary_vmem[index].at[shard] for index in range(3)),
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
                                stage_projections=attention_schedule in ("auto", "overlap"),
                                state_stream=state_stream,
                            )
                        if shared_state_stream is not None:
                            # All KDA consumers have finished. The final three
                            # writes are the only outstanding DMAs: every read
                            # was acquired and earlier writes drained at reuse.
                            if (attention_schedule in ("auto", "overlap")
                                    and group_rows > 8 and "kda" not in DEBUG_SKIP):
                                shared_state_stream.finish_rolled()
                            else:
                                shared_state_stream.finish()
                        if "project" not in DEBUG_SKIP:
                            release_kda_inputs()
                            project_attention_dp(kda_projection_vmem[3], streams.get("output"))

                    def execute_streamed(history_buffers, state_buffers, read_semaphores, write_semaphores):
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
                                *auxiliary_vmem[:4],
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

                    state_buffer_count = 4 if batch_size == 32 and attention_dp == 1 and not KDA_STAGED else 3
                    pl.run_scoped(
                        execute_streamed_dp if attention_dp > 1 else execute_streamed,
                        tpu.VMEM((state_buffer_count, kda_row_chunk, *states[0].shape[2:]), states[0].dtype),
                        tpu.VMEM((state_buffer_count, kda_row_chunk, *states[1].shape[2:]), states[1].dtype),
                        tpu.SemaphoreType.DMA((state_buffer_count, 2)),
                        tpu.SemaphoreType.DMA((state_buffer_count, 2)),
                    )
                else:
                    def execute_scoped(history_vmem, state_vmem, state_semaphores):
                        state_copies = (
                            *kda_state_reads(convolution_ref, history_vmem, state_semaphores.at[0], state_index),
                            *kda_state_reads(recurrent_ref, state_vmem, state_semaphores.at[1], state_index),
                        )
                        for copy in state_copies:
                            copy.start()
                        if not overlap_kda_state_reads:
                            for copy in state_copies:
                                copy.wait()
                        execute(
                            history_vmem, state_vmem, state_semaphores, state_copies
                        )

                    pl.run_scoped(
                        execute_scoped,
                        tpu.VMEM(
                            (batch_size, *states[0].shape[2:]), states[0].dtype
                        ),
                        tpu.VMEM(
                            (batch_size, *states[1].shape[2:]), states[1].dtype
                        ),
                        tpu.SemaphoreType.DMA((2,)),
                    )

            def run_latent_attention():
                state_index = layer_index // 4
                buffer_slot = layer_index % 2
                query_norm_vmem = query_norms_vmem.at[buffer_slot]
                key_value_norm_vmem = key_value_norms_vmem.at[buffer_slot]
                layer_metadata_semaphores = metadata_semaphores.at[buffer_slot]
                # Attention DP projects the output once the MLA workspace is
                # released, so its reduction buffer does not add to the
                # workspace's peak (pooled weights: the projection persists).
                dp_output = None
                if attention_dp > 1:
                    output_view = pool_view("m_o", base=attention_base)
                    dp_output = (output_view, weight_stream(
                        "m_o", layer_weights["m_o"].at[state_index], output_view,
                        offset=pool_layout["m_o"] + attention_base,
                    ))

                def compute(new_key_vmem, new_value_vmem, *workspace):
                    count = len(mla_dcp.SCOPED_NAMES)
                    workspace, dcp_scoped = workspace[:-count], workspace[-count:]
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
                        projection_vmem, workspace = workspace[:4], workspace[4:]
                    mla_streams = {
                        key: stream
                        for key, name, view in zip(
                            ("query_up", "key_value_up", "output_gate", "output"),
                            MLA[2:6],
                            projection_vmem,
                            strict=True,
                        )
                        if (
                            stream := dp_output[1] if dp_output and name == "m_o" else weight_stream(
                                name,
                                layer_weights[name].at[state_index],
                                view,
                                offset=(
                                    pool_layout[name] + (attention_base if name == "m_o" else 0)
                                    if POOLED_WEIGHTS
                                    else None
                                ),
                            )
                        )
                        is not None
                    }
                    query_down_stream = query_down_weight_stream(state_index, attention_base)
                    if query_down_stream is not None:
                        mla_streams["query_down"] = query_down_stream
                    mla_options = dict(
                        weights=MultiHeadLatentAttentionWeights(
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
                        workspace=MultiHeadLatentAttentionWorkspace(*workspace),
                        # The pooled output projection overwrites the q-down
                        # projection, so it may only load once that is consumed.
                        output_projection_after_query_down=POOLED_WEIGHTS,
                        streams=mla_streams,
                        communication_workspace=communication_workspace,
                        epsilon=eps,
                        synchronize_devices=False,
                        weights_are_prefetched=True,
                        token_scatter_output=attention_token_scatter,
                        sequence_rows=sequence_rows,
                    )
                    if attention_dp > 1:
                        data_group = jax.lax.axis_index("tp") // (32 // attention_dp)

                        def group_table_row(row):
                            # Group row p is the batch row its positions came from.
                            return (row // group_rows_per_pair * tokens_per_pair
                                    + data_group * group_rows_per_pair + row % group_rows_per_pair)

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
                                mla_dcp.DcpCache(pool_ref, table_ref, paged_pages, state_index,
                                                 owner=(None if dcp_owner_table is None
                                                        else mla_dcp.owner_index(dcp_owner_table) % mla_dcp.GROUP),
                                                 table_row=group_table_row),
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
                                mla_dcp.DcpCache(pool_ref, table_ref, paged_pages, state_index,
                                                 owner=(None if dcp_owner_table is None
                                                        else mla_dcp.owner_index(dcp_owner_table))),
                                mla_dcp.DcpWorkspace(*dcp_persistent, *dcp_scoped),
                            ),
                            **mla_options,
                        )

                    commit_scope = _phase_start("q8_mla_commit")
                    # The pool stays read-only: hand this layer's new rows out.
                    def hand_out_rows(semaphores):
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

                pl.run_scoped(
                    compute,
                    # Pad the transient handoff so each runtime tile advances
                    # by a multiple of eight vector rows.
                    tpu.VMEM(
                        (batch_size, cache_handoff_head_count, 256),
                        jnp.bfloat16,
                    ),
                    tpu.VMEM(
                        (batch_size, cache_handoff_head_count, 128),
                        jnp.bfloat16,
                    ),
                    *mla_scoped_scratch,
                )
                if dp_output is not None:
                    project_attention_dp(*dp_output)

            if pool is not None:
                pool.acquire("attention")  # the attention code waits for its copies
            with _phase("p1_attention"):
                if attention_dp > 1 and attention_schedule == "overlap":
                    assert pool is not None

                    def consume_attention(run):
                        def consume():
                            run()
                            pool.release("attention")
                        return consume

                    # KDA frees its consumed prefix before the output projection;
                    # MLA reuses its q-down region for m_o and retains it until
                    # the whole attention operation completes. Both branches
                    # must restore the same arena state for the layer loop.
                    pool.cond(
                        (layer_index % 4 == 3) | (layer_index == 92),
                        consume_attention(run_latent_attention),
                        consume_attention(run_delta_attention),
                    )
                else:
                    jax.lax.cond(
                        (layer_index % 4 == 3) | (layer_index == 92),
                        run_latent_attention,
                        run_delta_attention,
                    )
            if pool is not None and not (attention_dp > 1 and attention_schedule == "overlap"):
                pool.release("attention")
            post_scope = _phase_start("q9_post_mix")
            with _phase("q9a_post_norm_wait"):
                norm_copies[1].wait()
                norm_copies[3].wait()
            if attention_dp_ownership:
                with _phase("q9b_post_rows"):
                    mix_dp_rows(
                        mlp_residual_vmem, layer_index // 12 + 1,
                        layer_index=layer_index, attention_output=True,
                    )
                normalize_dp_rows(ffn_norm_vmem, split=sp8_residual)
                _phase_end(post_scope)
                with _phase("t10_moe_input_gather"):
                    gather_dp_rows(normalized_vmem, split=sp8_residual)
            elif attention_token_scatter:
                rank = jax.lax.axis_index("tp")
                pair = (rank % 8) // (8 // sp_owners)
                owned_attention_vmem = _bf16_rows_view(
                    communication_workspace.attention_output_vmem,
                    tokens_per_pair,
                    hidden_size,
                )
                owned_normalized_vmem = owned_rows_vmem
                # With one owner per rank each rank keeps its half of the
                # pair's attention rows.
                pair_split = sp_owners // 4

                # Each row owner holds ``owned_rows`` rows. Static branches keep
                # source slices aligned while calculations use a purpose-sized
                # workspace.
                if owned_rows == 1:
                    prefix_values = prefix_vmem[...]
                for owner_pair in owner_indices(pair):
                    @owner_when(pair, owner_pair)
                    def prepare_owned_rows(owner_pair=owner_pair):
                        attention_rows = (
                            owned_attention_vmem
                            if pair_split == 1
                            else owned_attention_vmem.at[
                                pl.ds(owner_start((owner_pair % pair_split) * owned_rows), owned_rows)
                            ]
                        )
                        row_offset = owner_pair * owned_rows
                        if owned_rows == 1:
                            prefix_rows = prefix_values[
                                owner_pair : owner_pair + 1
                            ]
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
                                row_indices == owner_pair,
                                updated_prefix,
                                prefix_values,
                            )
                        else:
                            prefix_vmem[row_slice] = updated_prefix
                        def mix_row(local_row):
                            global_row = row_offset + local_row
                            return _residual_mixture_row(
                                block_rows(owner_block_row(global_row, local_row)),
                                (
                                    updated_prefix[local_row : local_row + 1]
                                    if owned_rows <= 2
                                    # stored above; re-read inside the row region
                                    else owner_prefix_row(global_row, row_slice, local_row)
                                ),
                                mlp_residual_vmem,
                                layer_index // 12 + 1,
                                epsilon=eps,
                                relaxed_reduction_order=(
                                    residual_reduction_order == "relaxed"
                                ),
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
                prefix_vmem[...] = jnp.where(
                    layer_index % 12 == 0,
                    attention_vmem[...],
                    prefix_vmem[...] + attention_vmem[...],
                )
                if aux_capture == "attention":
                    capture_aux_hidden()
                residual(
                    normalized_vmem,
                    mlp_residual_vmem,
                    layer_index // 12 + 1,
                )
                normalize(normalized_vmem, ffn_norm_vmem)
                _phase_end(post_scope)

            def after_experts():
                if persist_kda_state:
                    @pl.when((layer_index % 4 != 3) & (layer_index != 92))
                    def drain_state_writes():
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
                    # The expert slots are free: queue the next layer's weights
                    # (issued now if the arena has room, else by the release
                    # that makes it).
                    if not dense_layer:
                        pool.release("experts")
                    has_next = layer_index + 1 < layers
                    pl.when(has_next)(lambda: prefetch_norms(layer_index + 1))
                    request_attention(layer_index + 1, predicate=has_next)

                    @pl.when(has_next)
                    def prefetch_next_layer_metadata():
                        prefetch_auxiliary(layer_index + 1)
                        prefetch_metadata(layer_index + 1)

                    request_router(layer_index + 1, predicate=has_next)
                    return

                @pl.when(layer_index + 1 < layers)
                def prefetch_next_layer():
                    prefetch_norms(layer_index + 1)
                    prefetch_attention(layer_index + 1)
                    prefetch_auxiliary(layer_index + 1)
                    prefetch_metadata(layer_index + 1)
                    prefetch_router(layer_index + 1)

            def run_mixture(*resources):
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
                latent_output_vmem, latent_output_semaphore, expert_semaphores = (
                    resources[:3]
                )
                layer_router_vmem, layer_expert_vmem = router_vmem, expert_views()
                after_routing_scores = None
                if pool is not None:
                    layer_router_vmem = arena_router(pool.acquire("router"))  # moe waits
                    # The expert slots may overlap the router: its scores are
                    # computed before the first expert copy starts.
                    layer_expert_vmem = expert_views(
                        pool.allocate("experts", expert_pool_bytes, replaces=("router",))
                    )

                    def after_routing_scores():
                        pool.release("router")
                        pool.acquire("experts")

                if alias_expert_scales:
                    gate_up_vmem, gate_up_scales_vmem, down_vmem, down_scales_vmem = layer_expert_vmem
                    workspace_resources = resources[3:]
                else:
                    gate_up_vmem, down_vmem = layer_expert_vmem
                    gate_up_scales_vmem, down_scales_vmem = resources[3:5]
                    workspace_resources = resources[5:]
                mixture_output_vmem = (
                    owned_rows_vmem if moe_token_scatter_output else mixture_vmem
                )
                hot_slots = 0 if REPLICAS else (local_weights["expert_gu"].shape[0] - HOT_LOCAL_EXPERTS) // 2
                if hot_slots > 16:
                    raise ValueError("hot-expert copies: at most 16 slots (local ids < 128)")
                _HOT[:] = (
                    [(layer_weights, layer_index, hot_fp8_views(), hot_slots)]
                    if hot_slots > 0 and alias_expert_scales and POOLED_WEIGHTS and batch_size in (32, 64, 96, 128)
                    else []
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
                        (layer_weights["expert_gu"].at[layer_index - 1, :HOT_LOCAL_EXPERTS] if hot_slots > 0
                         else layer_weights["expert_gu"].at[layer_index - 1]),
                        gate_up_vmem,
                        (layer_weights["expert_gus"].at[layer_index - 1, :HOT_LOCAL_EXPERTS] if hot_slots > 0
                         else layer_weights["expert_gus"].at[layer_index - 1]),
                        gate_up_scales_vmem,
                        (layer_weights["expert_down"].at[layer_index - 1, :HOT_LOCAL_EXPERTS] if hot_slots > 0
                         else layer_weights["expert_down"].at[layer_index - 1]),
                        down_vmem,
                        (layer_weights["expert_ds"].at[layer_index - 1, :HOT_LOCAL_EXPERTS] if hot_slots > 0
                         else layer_weights["expert_ds"].at[layer_index - 1]),
                        down_scales_vmem,
                        expert_semaphores,
                    ),
                    workspace=MixtureOfExpertsWorkspace(*workspace_resources),
                    communication_workspace=communication_workspace,
                    epsilon=eps,
                    synchronize_devices=False,
                    routing_projection_is_prefetched=True,
                    group_pending_expert_routes=(
                        batch_size in (6, *SCATTER_BATCHES)
                        and residual_reduction_order == "relaxed"
                    ),
                    after_experts=after_experts,
                    after_routing_scores=after_routing_scores,
                    overlap_router_latent=(attention_dp == 2 and batch_size == 128 and sequence_rows == 1),
                    # Verify steps of 64 rows overlap it too; at 128 verify rows it is slower.
                    overlap_shared_reduction=(attention_dp == 2 and (batch_size == 64 or batch_size == 128 and sequence_rows == 1)),
                    token_scatter_output=moe_token_scatter_output,
                    routed_reduce_scatter=moe_routed_reduce_scatter,
                    token_pipeline=moe_token_pipeline,
                    prepare_weights_vmem=(
                        tuple(pool_view(name) for name in MOE_PREPARE_WEIGHTS)
                        if POOLED_WEIGHTS
                        else None
                    ),
                    shared_gate_up_stream=(
                        weight_stream(
                            "shared_gu",
                            layer_weights["shared_gu"].at[layer_index - 1],
                            pool_view("shared_gu"),
                            offset=pool_layout["shared_gu"],
                        )
                        if POOLED_WEIGHTS
                        else None
                    ),
                    token_scatter_owners=sp_owners,
                    owned_hidden_ref=owned_rows_vmem if shard_routing else None,
                    live_rows=active_rows,
                )

            def run_dense_mlp():
                """Layer 0's dense MLP with the weights streamed through the
                pool as contiguous row blocks (``dense_slabs`` in flight): the
                gate/up projection accumulates over 1024-row K blocks of
                ``dense_gu`` (a column block would be 7168 short DMA
                segments), the down projection over 384-row blocks of
                ``dense_down``; TP32 sum with the shared-expert transport;
                output in the mixture's layout."""
                gu_ref, down_ref = dense_refs["dense_gu"], dense_refs["dense_down"]
                k_block, down_block = 1024, 384
                gu_width = gu_ref.shape[1]
                gu_blocks = hidden_size // k_block
                down_blocks = down_ref.shape[0] // down_block
                assert gu_ref.shape == (gu_blocks * k_block, gu_width)
                assert down_ref.shape == (down_blocks * down_block, hidden_size)
                assert gu_width == 2 * down_blocks * down_block
                slab_rows = max(
                    pool_alias.storage_size((k_block, gu_width), gu_ref.dtype),
                    pool_alias.storage_size((down_block, hidden_size), down_ref.dtype),
                ) // 128
                slabs = dense_slabs
                assert slabs * slab_rows <= projection_expert_pool_vmem.shape[0]
                total = gu_blocks + down_blocks
                slab_base = 0
                if pool is not None:
                    slab_base = pool.allocate("dense", slabs * slab_rows * 128) // 128
                    pool.acquire("dense")

                def slab(index, shape, dtype):
                    return pool_alias.view(
                        projection_expert_pool_vmem.at[
                            pl.ds(slab_base + index * slab_rows, pool_alias.storage_size(shape, dtype) // 128), :
                        ],
                        shape,
                        dtype,
                    )

                def block_copy(k):
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

                def stream_blocks(projection_vmem=None, accumulator_vmem=None):
                    # Above B8 the unrolled partial sums spill; accumulate
                    # them in VMEM instead.
                    hidden_in = normalized_vmem[...]
                    projection, activation, accumulator = None, None, None
                    for k in range(total):
                        s = k % slabs
                        block_copy(k).wait()
                        if k < gu_blocks:
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
                                # The mathematical reference rounds the projections to BF16.
                                half = gu_width // 2
                                activation = _gated_activation(
                                    projection[:, :half].astype(jnp.bfloat16),
                                    projection[:, half:].astype(jnp.bfloat16),
                                )
                                if projection_vmem is not None:
                                    projection_vmem[:, :half] = activation.astype(jnp.float32)
                        else:
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

                if batch_size <= 8:
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
                # The pool is free again: stage layer 1 while the sum travels.
                if pool is not None:
                    pool.release("dense")
                after_experts()
                def reduce_half(first):
                    if first:
                        # Both halves reuse the transport buffers; a peer may
                        # still be decoding the first half.
                        collectives32.barrier()
                    return _all_reduce_rows(
                        accumulator[first : first + 8],
                        communication_workspace.shared_local_reduction_vmem,
                        communication_workspace.shared_host_reduction_vmem,
                        communication_workspace.send_semaphores,
                        communication_workspace.receive_semaphores,
                    )

                if (moe_token_scatter_output and attention_dp == 2
                        and sp_owners == 4 and batch_size in (64, 128) and sequence_rows == 1):
                    _dense_scatter_rows(accumulator,owned_rows_vmem,
                        communication_workspace.shared_local_reduction_vmem,
                        communication_workspace.shared_host_reduction_vmem,
                        communication_workspace.send_semaphores,
                        communication_workspace.receive_semaphores,sp_owners)
                else:
                    reduced = jnp.concatenate(
                        [reduce_half(first) for first in range(0, batch_size, 8)],
                        axis=0,
                    ) if batch_size > 8 else _all_reduce_rows(
                        accumulator,
                        communication_workspace.shared_local_reduction_vmem,
                        communication_workspace.shared_host_reduction_vmem,
                        communication_workspace.send_semaphores,
                        communication_workspace.receive_semaphores,
                    )
                    if moe_token_scatter_output:
                        rank = jax.lax.axis_index("tp")
                        pair = (rank % 8) // (8 // sp_owners)
                        for owner_pair in range(sp_owners):
                            @pl.when(pair == owner_pair)
                            def store_owned_dense_rows():
                                owned_rows_vmem[...] = reduced[
                                    owner_pair * owned_rows : (owner_pair + 1) * owned_rows
                                ].astype(jnp.bfloat16)
                    else:
                        mixture_vmem[...] = reduced.astype(jnp.bfloat16)

            if aux_capture == "mixture_input":
                capture_aux_hidden(normalized_vmem)
            if dense_layer:
                if "dense" not in DEBUG_SKIP:
                    run_dense_mlp()
            else:
                with _phase("p0_moe"):
                    if "moe" not in DEBUG_SKIP:
                        pl.run_scoped(run_mixture, *moe_scoped_scratch)
            finish_scope = _phase_start("q0_finish_rows")
            if attention_dp_ownership:
                # The MoE delivered the pair's rows of every DP group (with
                # one owner, every row).
                group = jax.lax.axis_index("tp") // (32 // attention_dp)
                mixture = owned_rows_vmem[:dp_held_rows]
                for data_group in range(1, attention_dp if sp_owners > 1 else 1):
                    first = data_group * group_rows_per_pair
                    mixture = jnp.where(
                        group == data_group, owned_rows_vmem[first : first + group_rows_per_pair], mixture
                    )
                finished = (prefix_vmem[:dp_held_rows] + mixture).astype(jnp.bfloat16)
                prefix_vmem[:dp_held_rows] = finished
                dp_owned_vmem[...] = finished
            elif attention_token_scatter:
                rank = jax.lax.axis_index("tp")
                pair = (rank % 8) // (8 // sp_owners)
                owned_prefix_vmem = owned_rows_vmem
                scattered_mixture = (
                    owned_rows_vmem[...] if moe_token_scatter_output else None
                )
                if owned_rows == 1:
                    prefix_values = prefix_vmem[...]
                    if not moe_token_scatter_output:
                        mixture_values = mixture_vmem[...]
                for owner_pair in owner_indices(pair):
                    @owner_when(pair, owner_pair)
                    def finish_owned_rows():
                        row_offset = owner_pair * owned_rows
                        if owned_rows == 1:
                            finished_prefix = (
                                prefix_values[
                                    owner_pair : owner_pair + 1
                                ]
                                + (
                                    # The MoE delivered this owner's row.
                                    scattered_mixture
                                    if moe_token_scatter_output
                                    else mixture_values[
                                        owner_pair : owner_pair + 1
                                    ]
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
                        owned_prefix_vmem[:, : hidden_size // 2] = (
                            finished_prefix[:, : hidden_size // 2]
                        )
                        owned_prefix_vmem[:, hidden_size // 2 :] = (
                            finished_prefix[:, hidden_size // 2 :]
                        )
                if sequence_parallel_pre_attention:
                    for owner_pair in owner_indices(pair):
                        @owner_when(pair, owner_pair)
                        def store_owned_prefix_rows():
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
            else:
                prefix_vmem[...] += mixture_vmem[...]
            _phase_end(finish_scope)
            if aux_capture == "layer":
                capture_aux_hidden(gather_owned_rows=sequence_parallel_pre_attention)

        if include_layer_zero:
            layer(jnp.int32(0), dense_layer=True)
        if pool is not None:
            pool.loop(1, layers)(layer)
            if pool.report:
                print(pool.timeline(), flush=True)
        else:
            pl.loop(1, layers)(layer)
        if headroom_pad is not None:
            # Keep the pad live through the whole program (it holds zeros).
            prefix_vmem[0:8, 0:128] += jnp.sum(
                headroom_pad[...].reshape(-1, 8, 128), axis=0
            ).astype(jnp.bfloat16)

        if sequence_parallel_pre_attention:
            final_weight_copies = (
                tpu.make_async_copy(
                    output_residual_ref,
                    attention_residual_vmem,
                    norm_semaphores.at[0],
                ),
                tpu.make_async_copy(
                    final_normalization_ref,
                    attention_norm_vmem,
                    norm_semaphores.at[2],
                ),
            )
            for copy in final_weight_copies:
                copy.start()
            for copy in final_weight_copies:
                copy.wait()

            if attention_dp_ownership:
                final_block_count = (layers - 1) // 12 + 1
                def mix_row(row):
                    values = jnp.concatenate(
                        (block_rows(row, final_block_count), prefix_vmem[row : row + 1]),
                        axis=0,
                    ).astype(jnp.float32)
                    scores = jnp.sum(values * attention_residual_vmem[0], axis=1)
                    scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + eps)
                    return jnp.sum(
                        jax.nn.softmax(scores)[:, None] * values,
                        axis=0, keepdims=True,
                    ).astype(jnp.bfloat16)
                store_mixed_rows(mix_row, dp_owned_vmem)
                normalize_dp_rows(attention_norm_vmem)
                gather_dp_rows(normalized_vmem)
            else:
                rank = jax.lax.axis_index("tp")
                pair = (rank % 8) // (8 // sp_owners)
                final_block_count = (layers - 1) // 12 + 1
                for owner_pair in owner_indices(pair):
                    @owner_when(pair, owner_pair)
                    def finish_owned_decoder_rows():
                        row_offset = owner_pair * owned_rows
                        row_slice = pl.ds(owner_start(row_offset), owned_rows)

                        def mix_row(local_row):
                            global_row = row_offset + local_row
                            values = jnp.concatenate(
                                (
                                    block_rows(owner_block_row(global_row, local_row), final_block_count),
                                    owner_prefix_row(global_row, row_slice, local_row),
                                ),
                                axis=0,
                            ).astype(jnp.float32)
                            scores = jnp.sum(
                                values * attention_residual_vmem[0], axis=1
                            )
                            scores *= jax.lax.rsqrt(
                                jnp.mean(values * values, axis=1) + eps
                            )
                            return jnp.sum(
                                jax.nn.softmax(scores)[:, None] * values,
                                axis=0,
                                keepdims=True,
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
            def store_output(semaphore):
                copy = tpu.make_async_copy(normalized_vmem, output_ref, semaphore)
                copy.start()
                copy.wait()

            pl.run_scoped(store_output, tpu.SemaphoreType.DMA)
            blocks_output_ref[...] = jnp.zeros(
                blocks_output_ref.shape, jnp.uint8
            )
        else:
            def store_output(semaphore):
                copy = tpu.make_async_copy(prefix_vmem, output_ref, semaphore)
                copy.start()
                copy.wait()

            pl.run_scoped(store_output, tpu.SemaphoreType.DMA)
            blocks_output_ref[...] = blocks_vmem[...].astype(jnp.bfloat16)

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
    # (With state_slots the inputs are the whole slot arrays; the updated
    # states are per row.)
    state_output_shapes = (
        jax.ShapeDtypeStruct((batch_size, *states[0].shape[1:]), states[0].dtype),
        jax.ShapeDtypeStruct((batch_size, *states[1].shape[1:]), states[1].dtype),
        jax.ShapeDtypeStruct((batch_size, MLA_LAYERS, cache_handoff_head_count, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((batch_size, MLA_LAYERS, cache_handoff_head_count, 128), jnp.bfloat16),
    )
    output_shapes = (
        shape(prefix),
        (
            jax.ShapeDtypeStruct((8, 128), jnp.uint8)
            if sequence_parallel_pre_attention
            else jax.ShapeDtypeStruct(
                (batch_size, 8, hidden_size), jnp.bfloat16
            )
        ),
        *state_output_shapes,
        jax.ShapeDtypeStruct(
            (max(1, len(aux_hidden_layers)), batch_size, hidden_size), jnp.bfloat16
        ),
    )
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
    # Above B64 three slabs keep the arena at the expert slots' size (VMEM).
    dense_slabs = 4 if batch_size <= 64 else 3
    dense_slab_bytes = (
        max(
            pool_alias.storage_size((1024, weights["dense_gu"].shape[1]), weights["dense_gu"].dtype),
            pool_alias.storage_size((384, hidden_size), weights["dense_down"].dtype),
        )
        if include_layer_zero
        else 0
    )
    pool_layout = _pool_layout(
        pool_weights, alias_expert_scales=alias_expert_scales, pooled=POOLED_WEIGHTS,
        managed=pool_managed, dense_bytes=dense_slabs * dense_slab_bytes,
    )
    pool_bytes = pool_layout["pool_bytes"]
    if _config().vmem_report:
        def _mib(name, experts=False):
            value = pool_weights[name]
            shape = (4, *value.shape[1:]) if experts else value.shape
            return pool_alias.storage_size(shape, value.dtype) / 2**20
        region_a = {
            "kda": sum(_mib(n) for n in KDA[:4]),
            "mla": sum(_mib(n) for n in MLA[:2]),
            "experts": sum(_mib(n, True) for n in pooled_expert_names),
        }
        region_b = {
            "mla_extra": sum(_mib(n) for n in MLA_EXTRA_WEIGHTS),
            "moe_prepare": sum(_mib(n) for n in MOE_PREPARE_WEIGHTS),
            "latent_up": _mib("latent_up"),
        }
        print("pool region A (MiB):", {k: round(v, 2) for k, v in region_a.items()}, flush=True)
        if pool_managed:
            print(f"arena (managed region A): {pool_layout['arena_bytes'] / 2**20:.2f} MiB, "
                  f"attention {pool_layout['attention_bytes'] / 2**20:.2f}, dense slabs "
                  f"{dense_slabs * dense_slab_bytes / 2**20:.2f}", flush=True)
        print("pool region B (MiB):", {k: round(v, 2) for k, v in region_b.items()},
              {n: round(_mib(n), 2) for n in (*MLA_EXTRA_WEIGHTS, *MOE_PREPARE_WEIGHTS, "latent_up")}, flush=True)
    assert pool_bytes >= expert_pool_bytes and pool_bytes % 128 == 0
    scratch = (
        tpu.VMEM(
            (-(-block_rows_held // 16) * 16, hidden_size) if attention_dp_ownership else prefix.shape,
            jnp.bfloat16,
        ),
        # Residual snapshots are exact BF16 values. Up to B8: FP32
        # [batch, 8, hidden], no unpack per mix (a two-row allocation for the
        # owned rows miscompiles across loop iterations). Above B8 or with DP
        # row ownership: BF16, two rows per native (16, 128) tile.
        tpu.VMEM(
            (-(-block_rows_held // 2), 16, hidden_size) if blocks_block_major else (batch_size, 8, hidden_size),
            jnp.bfloat16 if blocks_block_major else jnp.float32,
        ),
        tpu.VMEM(prefix.shape, jnp.bfloat16),
        *(
            (tpu.VMEM((16, 128), jnp.bfloat16), tpu.VMEM((16, 128), jnp.bfloat16))
            if compact_outputs
            else (tpu.VMEM(prefix.shape, jnp.bfloat16), tpu.VMEM(prefix.shape, jnp.bfloat16))
        ),
        *(tpu.VMEM((1, hidden_size), weights[name].dtype) for name in NORMS),
        # With the router in the arena this persistent slot is a stub.
        tpu.VMEM((8, 128) if pool_managed else (1, hidden_size, 512), jnp.bfloat16),
        tpu.SemaphoreType.DMA((2,)),
        tpu.VMEM((pool_bytes // 128, 128), jnp.uint8),
        tpu.SemaphoreType.DMA((6,)),
        tpu.SemaphoreType.DMA((4,)),
        *(tpu.VMEM(local_weights[name].shape, local_weights[name].dtype) for name in KDA[4:]),
        *(
            (
                tpu.VMEM((batch_size, *states[0].shape[2:]), states[0].dtype),
                tpu.VMEM((batch_size, *states[1].shape[2:]), states[1].dtype),
            )
            if persist_kda_state
            else (
                tpu.VMEM((1,), jnp.uint8),
                tpu.VMEM((1,), jnp.uint8),
            )
        ),
        tpu.SemaphoreType.DMA((6,)),
        *(
            tpu.VMEM((2, *local_weights[name].shape), local_weights[name].dtype)
            for name in ("m_qnorm", "m_knorm", "router_bias", "latent_norm")
        ),
        tpu.SemaphoreType.DMA((2, 4)),
        tpu.VMEM(
            (owned_rows, hidden_size)
            if attention_token_scatter
            else (1,),
            jnp.bfloat16,
        ),
        tpu.SemaphoreType.DMA((1,)),
        tpu.SemaphoreType.DMA((4,)),
        tpu.SemaphoreType.DMA((_stream_semaphore_count(stream_config),)),
        *(
            (
                # Attention DP: the group's attention input rows, each shard's
                # gated attention output, the group rows' positions, and the
                # MLA latents shared by the shards.
                tpu.VMEM((group_rows, hidden_size), jnp.bfloat16),
                tpu.VMEM((attention_dp, group_rows, 384), jnp.bfloat16),
                tpu.SMEM((group_rows,), jnp.int32),
                tpu.VMEM((group_rows, 1536), jnp.bfloat16),
                tpu.VMEM((group_rows, 640), jnp.bfloat16),
            )
            if attention_dp > 1
            else ()
        ),
        # DCP: buffers group peers' remote copies land in.
        *mla_dcp.persistent_scratch(group_rows, heads=3 * attention_dp,
                                    cache_dtype=states[2].dtype, sequence_rows=sequence_rows,
                                    chunk_rows=_verify_chunk(group_rows, 3 * attention_dp, sequence_rows)),
        *communication_scratch(batch_size, attention_dp)[:-1],
        tpu.VMEM((batch_size, 4096), jnp.bfloat16),
        *projection_staging_scratch(batch_size, attention_dp),
        *routed_send_scratch(batch_size),
        # Four 7168-element BF16 wires, padded to 64 BF16 transport rows.
        # Private storage (see B4_ATTENTION_GATHER).
        *((tpu.VMEM((4, 32, 128), jnp.float32),)
          if b4_attention_gather else ()),
        # Diagnostic: vmem_pad_kb reserves unused VMEM to probe headroom.
        *(
            (tpu.VMEM((_config().vmem_pad_kb * 2, 128), jnp.float32),)
            if _config().vmem_pad_kb
            else ()
        ),
    )
    scratch_names = (
        "prefix", "blocks", "normalized", "attention", "mixture", *NORMS,
        "router", "router_semaphores", "projection_expert_pool", "pool_semaphores",
        "norm_semaphores", *KDA[4:], "kda_history", "kda_state", "auxiliary_semaphores",
        "m_qnorm", "m_knorm", "router_bias", "latent_norm", "metadata_semaphores",
        "owned_rows", "aux_semaphore", "dense_semaphores", "stream_semaphores",
        *(
            ("dp_group_input", "dp_attended", "dp_group_positions", "dp_query_latent", "dp_compressed")
            if attention_dp > 1
            else ()
        ),
        *mla_dcp.PERSISTENT_NAMES,
        "shared_local_reduction",
        "expert_local_reduction", "shared_host_reduction", "expert_host_reduction",
        "attention_features", "attention_output", "send_semaphores",
        "receive_semaphores", "reduction_phase", "latent_gather",
        *(("projection_staging", "routed_send") if batch_size > 32 else ()),
        *(("attention_gather",) if b4_attention_gather else ()),
        "headroom_pad",
    )
    persistent = [
        ("output window 1", tpu.VMEM(output_shapes[1].shape, output_shapes[1].dtype)),
        *zip(scratch_names, scratch),
    ]
    # Attention DP projects the group's rows (after the MLA workspace).
    projection_reduction = (
        "output-projection reduction",
        tpu.VMEM((group_rows, 8192), jnp.float32) if attention_dp > 1
        else tpu.VMEM((batch_size, 3584 if batch_size > 32 else 8192), jnp.float32),
    )
    regions = {
        "MLA": (
            [
                ("new-key handoff", tpu.VMEM((batch_size, cache_handoff_head_count, 256), jnp.bfloat16)),
                ("new-value handoff", tpu.VMEM((batch_size, cache_handoff_head_count, 128), jnp.bfloat16)),
                *zip(
                    (
                        *(() if POOLED_WEIGHTS else MLA[2:6]),
                        "key cache tiles", "value cache tiles", "cache semaphores",
                        "query", "output gate values", "new key", "new value",
                        "softmax maximum", "softmax denominator", "accumulator",
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
                ("kda history chunks", tpu.VMEM((3, kda_row_chunk, *states[0].shape[2:]), states[0].dtype)),
                ("kda state chunks", tpu.VMEM((3, kda_row_chunk, *states[1].shape[2:]), states[1].dtype)),
            ]
            if stream_kda_state
            else [
                ("kda history", tpu.VMEM((batch_size, *states[0].shape[2:]), states[0].dtype)),
                ("kda state", tpu.VMEM((batch_size, *states[1].shape[2:]), states[1].dtype)),
            ],
            [[
                projection_reduction,
                *(
                    (
                        ("DP staged projection", tpu.VMEM((group_rows // kda_row_chunk, kda_row_chunk, 1664), jnp.bfloat16)),
                        ("DP staged recurrence gate", tpu.VMEM((group_rows // kda_row_chunk, kda_row_chunk, 384), jnp.bfloat16)),
                        ("DP staged output gate", tpu.VMEM((group_rows, 384), jnp.bfloat16)),
                    )
                    if attention_dp > 1 and attention_schedule in ("auto", "overlap") and group_rows > 8
                    else ()
                ),
                *(
                    (("chunked attended", tpu.VMEM((batch_size, 3, 128), jnp.float32)),)
                    if batch_size > 8
                    else ()
                ),
            ]],
        ),
        "MoE": (
            list(
                zip(
                    (
                        *(() if POOLED_WEIGHTS else ("latent_up",)),
                        "latent-up semaphore", "expert semaphores",
                        *(() if alias_expert_scales else ("gate/up scale slots", "down scale slots")),
                        "selected experts", "selected probabilities", "local route counts",
                        "probability totals", "routing scores", "shared outputs",
                        "routed outputs", "gate/up accumulator", "down accumulator",
                        "processed routes", "compact source 1", "compact source 2",
                        "shared send semaphores", "shared receive semaphores",
                        "output transport", "output send semaphores",
                        "output receive semaphores", "latent send semaphores",
                        "latent receive semaphores",
                        *(("routing selection source", "routing selections") if shard_routing else ()),
                    ),
                    moe_scoped_scratch,
                    strict=True,
                )
            ),
            []
            if POOLED_WEIGHTS
            else [[
                (name, tpu.VMEM(local_weights[name].shape, local_weights[name].dtype))
                for name in MOE_PREPARE_WEIGHTS
            ]],
        ),
    }
    if include_layer_zero and batch_size > 8:
        regions["dense layer 0"] = (
            [
                ("gate/up accumulator", tpu.VMEM((batch_size, weights["dense_gu"].shape[1]), jnp.float32)),
                ("down accumulator", tpu.VMEM((batch_size, hidden_size), jnp.float32)),
            ],
            [],
        )
    vmem_report, vmem_peak = _vmem_budget(persistent, regions)
    if pool_managed:
        # Grow the arena, all or nothing, into VMEM above the spill reserve so
        # the next layer's attention projections and router (queued together
        # once the experts finish) fit without waiting for a release.
        # TODO: derive the demand from a dry-run trace of the requests.
        demand = pool_layout["attention_bytes"] + pool_alias.storage_size(
            (1, hidden_size, 512), jnp.dtype(jnp.bfloat16)
        )
        shortfall = max(0, -(-(demand - pool_layout["arena_bytes"]) // 4096) * 4096)
        available = (VMEM_LIMIT_BYTES - VMEM_SPILL_RESERVE_BYTES - vmem_peak) // 4096 * 4096
        # B128's register spills exceed the reserve: its arena does not grow.
        spare = shortfall if shortfall <= available and batch_size < 128 else 0
        if spare > 0:
            pool_layout = _pool_layout(
                pool_weights, alias_expert_scales=alias_expert_scales, pooled=POOLED_WEIGHTS,
                managed=True, dense_bytes=dense_slabs * dense_slab_bytes, arena_extra=spare,
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
            print(f"arena with spare VMEM: {pool_layout['arena_bytes'] / 2**20:.2f} MiB", flush=True)
    if _config().vmem_report:
        print(f"VMEM plan, batch {batch_size}:\n{vmem_report}", flush=True)
    if vmem_peak > VMEM_LIMIT_BYTES:
        raise ValueError(f"The decoder's static VMEM plan exceeds the limit:\n{vmem_report}")
    if vmem_peak + VMEM_SPILL_RESERVE_BYTES > VMEM_LIMIT_BYTES:
        import warnings

        warnings.warn(
            f"The decoder's VMEM plan leaves less than the spill reserve:\n{vmem_report}",
            stacklevel=2,
        )
    return pl.pallas_call(
        body,
        name="kimi_decoder_stack",
        out_shape=output_shapes,
        input_output_aliases={},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3 + int(slot_reads) + int(live_rows is not None),
            in_specs=[
                pl.BlockSpec(memory_space=tpu.HBM) for _ in range(len(args))
            ],
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
            **({"internal_scratch_in_bytes": _config().internal_scratch_kb * 1024} if _config().internal_scratch_kb else {}),
        ),
    )(
        positions,
        state_row_scalar,
        jnp.asarray(block_tables, jnp.int32).reshape(-1),
        *((jnp.asarray(state_slots, jnp.int32).reshape(batch_size),) if slot_reads else ()),
        *((jnp.asarray(live_rows, jnp.int32).reshape(batch_size),) if live_rows is not None else ()),
        *args,
    )




def decode_local(
    weights,
    states,
    tokens: TensorLike,  # [batch_size]
    positions: TensorLike,  # [batch_size]
    *,
    batch_size: int,
    layers: int = 93,
    eps: float = 1e-5,
    residual_reduction_order: Literal["reference", "row", "relaxed"] = "reference",
    mla_cache_batch_tile_size: int | None = None,
    alias_expert_scales: bool = False,
    attention_token_scatter: bool = False,
    moe_routed_reduce_scatter: bool = False,
    moe_token_scatter_output: bool = False,
    sequence_parallel_pre_attention: bool = False,
    gate_first_kda_projection: bool | None = None,
    batched_decoder_boundaries: bool | None = None,
    sequence_rows: int = 1,
    aux_hidden_layers: tuple[int, ...] = (),
    aux_capture: Literal["layer", "attention", "mixture_input"] = "layer",
    state_row=None,
    attention_dp: int = 1,
    attention_schedule: Literal["auto", "pooled", "legacy", "overlap"] = "auto",
    attention_state_pipeline: bool = False,
    attention_dp_ownership: bool = False,
    attention_projection_schedule: Literal["default", "mla_up_double"] = "default",
    moe_token_pipeline: bool = False,
    dp_moe_gather: Literal["staged", "split"] = "staged",
    mla_commit_tile_size: int | None = None,
    sequence_parallel_size: int = 4,
    block_tables=None,
    dcp_pool_map: tuple[int, ...] | None = None,
    dcp_owner_table: tuple[int, ...] | None = None,
    state_slots=None,
    live_rows=None,
):
    """Decode one native batch tile with local TP32 weights and caches.

    ``states`` is ``(conv, recurrent, *pools)`` over vLLM's unified-pool
    tensors ``[blocks, 128, 2, 640]`` (read in place, never written here),
    MLA layer ``m`` in ``pools[dcp_pool_map[m]]``, ``block_tables`` (``[batch_size,
    pages]`` int32) the rows' tables, ``dcp_owner_table`` (mesh position ->
    page owner index, e.g. vLLM TP rank) which pages each rank holds. The
    returned states are ``(conv, recurrent, new_keys, new_values)``: each
    row's new latent for every MLA layer (``[batch, 24, H, 256]`` / ``[batch,
    24, H, 128]``), for ``paged.commit_latent_rows`` to commit.

    Layer zero (its KDA and dense MLP) runs inside the fused stack after the
    XLA embedding gather. ``state_row`` (int32 scalar, requires
    ``sequence_rows == batch_size``) names the row of the KDA input states
    to start the sequence from, replacing a pre-selected state copy.

    ``sequence_rows`` groups consecutive rows into one sequence of consecutive
    tokens (speculative verification): KDA and MLA states lead with rows and
    sequences respectively, every row's KDA post-state is returned, and MLA
    pools receive all rows. ``aux_hidden_layers`` additionally returns the
    residual stream entering those layers as ``[len, batch, hidden]``.
    """
    if gate_first_kda_projection is False:
        raise ValueError("The clean decoder always uses the split KDA gate projection")
    gate_first_kda_projection = True
    if batched_decoder_boundaries is None:
        # The embedding gather, the dense MLP and (sequence-parallel) lm_head
        # run on the whole tile instead of row by row; the collectives then
        # carry batch-sized payloads, so the results differ from the per-row
        # path at the ULP level.
        batched_decoder_boundaries = batch_size in SCATTER_BATCHES or attention_dp > 1
    if not 1 <= batch_size <= MAX_BATCH:
        raise ValueError(f"The native decoder supports batch sizes from 1 to {MAX_BATCH}")
    aux_hidden_layers = tuple(aux_hidden_layers)

    rank = jax.lax.axis_index("tp")
    vocabulary = weights["embedding"].shape[0]

    if batched_decoder_boundaries:
        local_ids = tokens - rank * vocabulary
        local_embeddings = weights["embedding"][
            jnp.clip(local_ids, 0, vocabulary - 1)
        ]
        embedding = kimi.tp_sum(
            jnp.where(
                ((local_ids >= 0) & (local_ids < vocabulary))[:, None],
                local_embeddings.astype(jnp.float32),
                0,
            )
        ).astype(jnp.bfloat16)

    else:
        # Per-row embedding gather and TP sum (see batched_decoder_boundaries).
        def prepare_first_layer_row(_, token):
            local_id = token - rank * vocabulary
            embedding_row = weights["embedding"][
                jnp.clip(local_id, 0, vocabulary - 1)
            ][None]
            embedding_row = kimi.tp_sum(
                jnp.where(
                    (local_id >= 0) & (local_id < vocabulary),
                    embedding_row.astype(jnp.float32),
                    0,
                )
            ).astype(jnp.bfloat16)
            return (), embedding_row[0]

        _, embedding = jax.lax.scan(
            prepare_first_layer_row, (), tokens
        )
    if state_row is not None and sequence_rows != batch_size:
        raise ValueError("state_row needs one sequence per tile")
    # Layer zero runs inside the fused stack and starts from the embedding.
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
    ) = (
        stack(
            prefix,
            embedding,
            weights,
            (
                convolution_states,
                recurrent_states,
                *cache_states,
            ),
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
            gate_first_kda_projection=gate_first_kda_projection,
            sequence_rows=sequence_rows,
            aux_hidden_layers=aux_hidden_layers,
            aux_capture=aux_capture,
            include_layer_zero=True,
            state_row=state_row,
            attention_dp=attention_dp,
            attention_schedule=attention_schedule,
            attention_state_pipeline=attention_state_pipeline,
            attention_dp_ownership=attention_dp_ownership,
            attention_projection_schedule=attention_projection_schedule,
            moe_token_pipeline=moe_token_pipeline,
            dp_moe_gather=dp_moe_gather,
            mla_commit_tile_size=mla_commit_tile_size,
            sequence_parallel_size=sequence_parallel_size,
            block_tables=block_tables,
            dcp_pool_map=dcp_pool_map,
            dcp_owner_table=dcp_owner_table,
            state_slots=state_slots,
            live_rows=live_rows,
        )
    )

    if sequence_parallel_pre_attention and batched_decoder_boundaries:
        logits = jnp.dot(
            prefix, weights["lm_head"], preferred_element_type=jnp.float32
        )
    elif sequence_parallel_pre_attention:
        def project_decoder_row(_, hidden_row):
            logits_row = jnp.dot(
                hidden_row[None],
                weights["lm_head"],
                preferred_element_type=jnp.float32,
            )
            return (), logits_row[0]

        _, logits = jax.lax.scan(project_decoder_row, (), prefix)
    else:
        active_block_count = (layers - 1) // 12 + 1

        def finish_decoder(_, inputs):
            prefix_row, block_rows = inputs
            hidden_row = kimi.attention_residual(
                prefix_row,
                block_rows[:active_block_count],
                weights["output_res"][0],
                eps,
            )[None]
            hidden_row = kimi.rms(hidden_row, weights["final_norm"], eps)
            logits_row = jnp.dot(
                hidden_row,
                weights["lm_head"],
                preferred_element_type=jnp.float32,
            )
            return (), logits_row[0]

        _, logits = jax.lax.scan(finish_decoder, (), (prefix, blocks))
    new_states = (
        convolution_states,
        recurrent_states,
        new_keys,
        new_values,
    )
    if aux_hidden_layers:
        return logits[None], aux_hidden, new_states
    return logits[None], new_states
