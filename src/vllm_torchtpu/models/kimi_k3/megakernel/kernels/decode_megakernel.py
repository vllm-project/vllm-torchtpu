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

# Source provenance: inferact/k3-megakernel@30c9ba966b
#   src/vllm_torchtpu/megakernel/kimi_k3/kernels/decode_megakernel.py
"""Kimi K3 decode megakernel components."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, Protocol, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.core.collectives.spmd import mesh_axis_index
from vllm_torchtpu.models.kimi_k3.megakernel import kernels as kimi
from vllm_torchtpu.models.kimi_k3.megakernel.common import collectives32, pool_alias
from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_TP,
    BITS_PER_WORD,
    CONV_HEAD_ROWS,
    CONV_TAPS,
    DP2_GROUPS,
    DP4_GROUPS,
    DP_VALID_BATCHES,
    EXPERT_FAMILIES,
    HEAD_DIM,
    HIDDEN_SIZE,
    KDA_HEADS,
    KDA_QKV_PARTS,
    MIB_BYTES,
    MLA_LATENT_ROW_LANES,
    MLA_LAYER_MODULO,
    MLA_LAYER_STRIDE,
    NUM_LAYERS,
    PAGE_TOKENS,
    POWER_OF_TWO_BATCHES,
    ROUTED_EXPERTS,
    SUBLANE_ROWS,
    TOKEN_SCATTER_BATCHES,
    TP_PEERS,
    TP_SIZE,
    VREG_LANE_COUNT,
)
from vllm_torchtpu.models.kimi_k3.megakernel.common.vmem_pool import VmemPool
from vllm_torchtpu.models.kimi_k3.megakernel.config import active as _config
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import mla_dcp
from vllm_torchtpu.models.kimi_k3.megakernel.layout import (
    MLA_LAYERS,
    cache_layout,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy as np
    from jax import typing as jax_typing

    from vllm_torchtpu.models.kimi_k3.megakernel.common.protocols import (
        ArenaRefProtocol,
        AsyncCopyProtocol,
        PallasRefProtocol,
    )

type TensorLike = jax.Array | PallasRefProtocol
type PallasRef = PallasRefProtocol

_BF16_VIEW_NDIM = 2
_BF16_TILE_ROWS = 16
_ROUTER_LANE_SCORE_WIDTH = 512
_PADDED_ROUTER_SCORE_WIDTH = 1024
_ODD_LANE_ROUTER_COLUMNS = 384
_LOCAL_RANK_PAIRS = 4
_PAIR_PEER_OFFSETS = 3
_MXFP4_VALUES_PER_WORD = 8
_MXFP4_SCALE_BLOCK = 32
_FP32_EXPONENT_SHIFT = 23
_SMALL_EXPERT_CONTRACTION = 768
_SPLIT_CONTRACTION_BLOCK = 256
_DEFAULT_ACTIVATION_ALIGNMENT = 512
_WIDE_DOT_TILE_BATCH = 96
_WIDE_BATCH_DOT_TILE = 1024
_DEFAULT_SPLIT_DOT_TILE = 512
_BLOCK_STAGED_BATCHES = (16, 32, 64, 96, 128)
_PLANNED_STAGE_BATCHES = (32, 64, 96, 128)
_ONE_HOT_STAGE_BATCHES = (64, 96, 128)
_BLOCK_GATHER_BATCH = 32
_SCALAR_SCAN_BATCH = 16
_SCALAR_SCAN_MASK_16 = 65535
_COMPACT_ROUTE_MIN_BATCH = 8
_SHARED_FINISHER_COMPACT_CAPACITY = 8
_EXPERT_PREFETCH_SLOTS = 4
_MAX_LOCAL_EXPERTS = 128
_SMALL_UNGROUPED_CARRY_BATCHES = (2, 4)
_GROUPED_CARRY_BATCHES = (8, 16, 32, 64, 96, 128)
_COMPACT_PAIR_CAPACITIES = (2, 4, 8)
_INLINE_TRANSPORT_MAX_BATCH = 32
_SINGLE_PASS_REDUCTION_MAX_BATCH = 16
_DENSE_HALF_BATCH_ROWS = 8
_SPECULATIVE_B6_BATCH = 6
_PAIR_OWNED_OUTPUT_MIN_BATCH = 64
_ROUTED_TRANSPORT_ROWS_PER_TOKEN = 14
_SHARED_TRANSPORT_ROWS_PER_TOKEN = 28
_MIN_WIDE_REDUCTION_ROWS = 64
_MIN_EXPERT_REDUCTION_ROWS = 32
_SHARED_SHARD_ROWS_PER_TOKEN = 4
_DP_GATHER_ROWS_PER_TOKEN = 7
_HOST_SHARED_SHARD_ROWS_PER_TOKEN = 8
_ATTENTION_FEATURE_SHARD_WIDTH = 1024
_LATENT_GATHER_WIDTH = 4096
_TRANSPORT_SEMAPHORE_COUNT = 16
_BF16_PER_FP32_WORD = 2
_BF16_ELEMENTS_PER_FP32_ROW = _BF16_PER_FP32_WORD * VREG_LANE_COUNT
_RANK_PAIR_SIZE = 2
_DOUBLE_BUFFER_SLOTS = 2
_COMPACT_INPUT_BUFFERS = 2
_HIDDEN_HALVES = 2
_KDA_PROJECTION_COUNT = 4
_MLA_INPUT_PROJECTION_COUNT = 2
_MLA_NORM_COUNT = 2
_ARENA_ALIGN_BYTES = 4096
_HALF_HIDDEN_REDUCTION_WIDTH = HIDDEN_SIZE // 2
_WIDE_REDUCTION_WIDTH = 8192
_WIDE_REDUCTION_MAX_BATCH = 32
_KDA_PROJECTION_WIDTH = 1664
_MLA_QUERY_LORA_RANK = 1536
_MLA_KEY_HANDOFF_LANES = 256
_KDA_HEAD_OUT_WIDTH = KDA_HEADS * HEAD_DIM
_MLA_HEAD_OUT_WIDTH = mla_dcp.HEADS_PER_RANK * HEAD_DIM
_B4_BATCH_SIZE = 4
_ROLLING_STATE_BUFFERS = 3
_UNROLLED_B32_STATE_BUFFERS = 4
_SMALL_DP_GROUP_ROWS = 8
_MAX_WHOLE_KDA_BATCH = 8
_KIB_BYTES = 1024
_logger = init_logger(__name__)

_PYTREE_LOCK = threading.Lock()
_DATACLASS_REGISTRATION_DONE: list[bool] = [False]


def ensure_dataclasses_registered() -> None:
    """Register megakernel PyTree dataclasses lazily on first instantiation."""
    if _DATACLASS_REGISTRATION_DONE[0]:
        return
    with _PYTREE_LOCK:
        if _DATACLASS_REGISTRATION_DONE[0]:
            return
        for cls in (
            PaddedBf16Rows,
            AttentionCommunicationWorkspace,
            MixtureOfExpertsWeights,
            MixtureOfExpertsWorkspace,
            KimiDeltaAttentionWeights,
            MultiHeadLatentAttentionWeights,
            MultiHeadLatentAttentionWorkspace,
        ):
            jax.tree_util.register_dataclass(cls)
        _DATACLASS_REGISTRATION_DONE[0] = True


@dataclasses.dataclass(frozen=True, slots=True)
class PaddedBf16Rows(pool_alias.Reinterpret):
    """Expose active rows while preserving native 16-row backing capacity."""

    if TYPE_CHECKING:
        shape: tuple[int, ...]

        def __init__(self, shape: tuple[int, ...]) -> None:
            """Initialize the padded 2D BF16 row view transform."""
            ...

    def __post_init__(self) -> None:
        """Register megakernel PyTree dataclasses on first instantiation."""
        ensure_dataclasses_registered()

    def _validate_shape(self, input_shape: tuple[int, ...]) -> None:
        """Validate that ``shape`` fits inside a 16-row-aligned backing shape."""
        rows, width = self.shape
        backing_rows = math.prod(input_shape) // width
        if not (
            len(self.shape) == _BF16_VIEW_NDIM
            and width % VREG_LANE_COUNT == 0
            and 1 <= rows <= backing_rows
        ):
            msg = f"Invalid padded BF16 row shape {self.shape} for {input_shape}"
            raise ValueError(msg)
        if not (
            backing_rows * width == math.prod(input_shape)
            and backing_rows % _BF16_TILE_ROWS == 0
        ):
            msg = (
                f"Backing shape {input_shape} must cover a 16-row multiple "
                f"of {self.shape}"
            )
            raise ValueError(msg)


# Named phase scopes for custom-call tracing (KernelConfig.phase_scopes=True).
PHASE_SCOPES = _config().phase_scopes


class PhaseScopeProtocol(Protocol):
    """Structural protocol for named tracing scope context managers."""

    def __enter__(self) -> object:
        """Enter the tracing scope."""
        ...

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
        /,
    ) -> bool | None:
        """Exit the tracing scope."""
        ...


class _BitcastRefViewProtocol(Protocol):
    """Structural protocol for bitcast references carrying transform state."""

    @property
    def ref(self) -> object:
        """Return the underlying state reference."""
        ...

    @property
    def transforms(self) -> tuple[pool_alias.types.Transform, ...]:
        """Return accumulated reference transforms."""
        ...


class _ExpertCopyCallable(Protocol):
    """Structural protocol for building per-slot asynchronous expert copies."""

    def __call__(
        self,
        expert: int | jax.Array,
        slot: int | jax.Array,
        *,
        gate_up: bool = ...,
        down: bool = ...,
    ) -> SimpleNamespace:
        """Return async copy descriptors for ``expert`` into ``slot``."""
        ...


type _LocalSelectionHook = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array],
    None,
]
type _SelectionStepHook = Callable[[int], None]
type _PhaseCallback = Callable[[], None]
type _ProgressHook = Callable[[jax.Array], None]
type _RouteChoice = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]
type _PackedRouteChoice = tuple[jax.Array, ...]
type _RowPlan = tuple[jax.Array, jax.Array]
type _CompactRouteData = tuple[jax.Array, jax.Array]


def _phase(name: str) -> PhaseScopeProtocol:
    """Create a named tracing context when phase scopes are enabled."""
    return jax.named_scope(name) if PHASE_SCOPES else contextlib.nullcontext()


def phase_start(name: str) -> PhaseScopeProtocol:
    """Enter a named tracing scope when phase scopes are enabled."""
    scope = _phase(name)
    scope.__enter__()
    return scope


def phase_end(scope: PhaseScopeProtocol) -> None:
    """Exit a tracing scope opened by ``phase_start``."""
    scope.__exit__(None, None, None)


def _scoped[**P, R](name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a decorator wrapping ``pl.when`` callbacks in ``_phase(name)``."""

    def wrap(function: Callable[P, R]) -> Callable[P, R]:
        """Wrap ``function`` inside the named phase scope."""

        @functools.wraps(function)
        def scoped(*args: P.args, **kwargs: P.kwargs) -> R:
            """Execute ``function`` inside ``name``."""
            with _phase(name):
                return function(*args, **kwargs)

        return scoped

    return wrap


# Largest supported decode batch; SCATTER_BATCHES lists the token-scatter batches.
MAX_BATCH = 128
SCATTER_BATCHES = TOKEN_SCATTER_BATCHES


def _bf16_rows_view(
    ref: PallasRefProtocol,
    rows: int,
    width: int,
) -> PallasRefProtocol:
    """Expose ``rows`` active BF16 rows of ``width`` columns over ``ref``."""
    aliased = cast("_BitcastRefViewProtocol", ref.bitcast(jnp.bfloat16))
    return cast(
        "PallasRefProtocol",
        pool_alias.types.TransformedRef(
            aliased.ref,
            (*aliased.transforms, PaddedBf16Rows((rows, width))),
        ),
    )


def dot(x: jax.Array | np.ndarray, w: jax.Array | np.ndarray) -> jax.Array:
    """Multiply up to ``MAX_BATCH`` activation rows by ``w`` in FP32."""
    batch_size = x.shape[0]
    if not 1 <= batch_size <= MAX_BATCH:
        msg = f"dot supports 1 to {MAX_BATCH} rows, got batch_size={batch_size}"
        raise ValueError(msg)
    if batch_size == 1:
        # B1 broadcasts into one 16-row tile instead of zero-padding.
        tiled_x = jnp.broadcast_to(x, (_BF16_TILE_ROWS, x.shape[-1]))
    elif batch_size % _BF16_TILE_ROWS:
        # Whole native 16-row matrix tiles.
        tiled_x = jnp.pad(x, ((0, -batch_size % _BF16_TILE_ROWS), (0, 0)))
    else:
        tiled_x = x
    return jax.lax.dot_general(
        tiled_x,
        w,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:batch_size]


def attention_scores(
    queries: jax.Array | np.ndarray,
    keys: jax.Array | np.ndarray,
) -> jax.Array:
    """Multiply partial query rows by a transposed key tile in FP32."""
    row_count = queries.shape[0]
    padded_queries = jnp.pad(
        queries,
        ((0, -row_count % _BF16_TILE_ROWS), (0, 0)),
    )
    return jax.lax.dot_general(
        padded_queries,
        keys,
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )[:row_count]


def _gated_activation(gate: jax.Array, value: jax.Array) -> jax.Array:
    """Compute the Kimi K3 shared and routed expert activation in BF16."""
    return (
        4 * jnp.tanh(gate / 4) * jax.nn.sigmoid(gate) * 25 * jnp.tanh(value / 25)
    ).astype(jnp.bfloat16)


def _route_hidden_states(
    hidden_states: jax.Array,
    routing_projection_vmem: PallasRefProtocol,
    routing_bias_ref: TensorLike,
    routing_scores_vmem: PallasRefProtocol,
    send_semaphore: PallasRefProtocol,
    receive_semaphore: PallasRefProtocol,
    *,
    selected_expert_count: int,
    local_expert_count: int,
    expert_tensor_parallel: int,
    on_local_selection: _LocalSelectionHook,
    after_selection: _SelectionStepHook,
    after_scores: _PhaseCallback | None = None,
    after_score_send: _PhaseCallback | None = None,
    live_rows: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project and select local expert routes through one shared B1/B2 path.

    ``routing_scores_vmem`` is ``[batch_size, 1024]``. Paired ranks exchange
    their 512-score halves directly in that padded workspace, then every top-k
    iteration advances all batch rows together. Selected local routes are
    returned as ``[batch_size, top_k]`` values for scalar memory
    materialization by the caller.
    """
    rank = jax.lax.axis_index(AXIS_TP)
    batch_size = hidden_states.shape[0]
    expert_count = routing_bias_ref.shape[-1]
    if expert_count != ROUTED_EXPERTS or hidden_states.shape[-1] > HIDDEN_SIZE:
        msg = "Host-shared router requires 896 expert scores"
        raise ValueError(msg)
    if routing_scores_vmem.shape[-1] != _PADDED_ROUTER_SCORE_WIDTH:
        msg = "Routing score workspace must have padded width 1024"
        raise ValueError(msg)

    router_lane = rank % 2
    local_scores = dot(hidden_states, routing_projection_vmem[0]).reshape(
        batch_size,
        _ROUTER_LANE_SCORE_WIDTH,
    )
    # Lane 1's last 128 router columns are padding; in the shared arena they
    # hold stale bytes, so zero their scores.
    column = jax.lax.broadcasted_iota(jnp.int32, local_scores.shape, 1)
    local_scores = jnp.where(
        (router_lane == 0) | (column < _ODD_LANE_ROUTER_COLUMNS),
        local_scores,
        0.0,
    )
    score_half = routing_scores_vmem.at[
        :,
        pl.ds(
            router_lane * _ROUTER_LANE_SCORE_WIDTH,
            _ROUTER_LANE_SCORE_WIDTH,
        ),
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
        (batch_size, selected_expert_count),
        jnp.int32,
    )
    selected_probabilities = jnp.zeros(
        (batch_size, selected_expert_count),
        jnp.float32,
    )
    local_route_counts = jnp.zeros((batch_size,), jnp.int32)
    probability_totals = jnp.zeros((batch_size,), jnp.float32)
    expert_ids = jnp.arange(expert_count)[None, :]
    route_slots = jnp.arange(selected_expert_count)[None, :]
    # Only top-k rank is sequential. Every score reduction and ownership
    # operation carries the leading batch dimension with no per-row control.
    for selection in range(selected_expert_count):
        winners = jnp.argmax(
            routing_scores_vmem[:, :expert_count],
            axis=1,
        ).astype(jnp.int32)
        winner_mask = expert_ids == winners[:, None]
        probabilities = jnp.sum(
            jnp.where(winner_mask, uncorrected_probabilities, 0),
            axis=1,
        )
        probability_totals += probabilities
        routing_scores_vmem[:, :expert_count] = jnp.where(
            winner_mask,
            -jnp.inf,
            routing_scores_vmem[:, :expert_count],
        )

        counts_before_selection = local_route_counts
        owned_locally = winners // local_expert_count == rank // expert_tensor_parallel
        if live_rows is not None:
            owned_locally &= live_rows
        destination_mask = owned_locally[:, None] & (
            route_slots == counts_before_selection[:, None]
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
        local_route_counts = counts_before_selection + owned_locally.astype(jnp.int32)
        on_local_selection(
            winners,
            owned_locally,
            counts_before_selection,
            selected_experts,
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


@dataclasses.dataclass(frozen=True, slots=True)
class _OwnedRouteSelectionBuffers:
    """Scratch references and semaphores for pair-sharded route exchanges."""

    selection_source_vmem: PallasRefProtocol
    selections_vmem: PallasRefProtocol
    send_semaphores: PallasRefProtocol
    receive_semaphores: PallasRefProtocol


@dataclasses.dataclass(frozen=True, slots=True)
class _OwnedRouteDerivationSpec:
    """Ownership geometry and selection callback for gathered route rows."""

    batch_size: int
    selected_expert_count: int
    local_expert_count: int
    expert_tensor_parallel: int
    on_local_selection: _LocalSelectionHook


def _gather_owned_route_selections(
    rank: jax.Array,
    pair: jax.Array,
    rows: int,
    buffers: _OwnedRouteSelectionBuffers,
) -> jax.Array:
    """Broadcast each local rank pair's owned selection block across pairs."""
    for owner in range(_LOCAL_RANK_PAIRS):

        @pl.when(pair == owner)
        def gather_selections(owner: int = owner) -> None:
            """Copy ``owner`` pair's route selections to the three peer pairs."""
            destination = buffers.selections_vmem.at[pl.ds(owner * rows, rows)]
            destination[...] = buffers.selection_source_vmem[...]
            copies = []
            for offset in range(_PAIR_PEER_OFFSETS, 0, -1):
                copy = tpu.make_async_remote_copy(
                    buffers.selection_source_vmem,
                    destination,
                    buffers.send_semaphores.at[1 + offset],
                    buffers.receive_semaphores.at[1 + offset],
                    device_id=(rank ^ (2 * offset),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()

    return buffers.selections_vmem[...]


def _derive_local_routes_from_gathered(
    gathered: jax.Array,
    rank: jax.Array,
    spec: _OwnedRouteDerivationSpec,
    live_rows: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Extract this rank's owned expert routes from all gathered tokens."""
    selected_count = spec.selected_expert_count
    all_winners = gathered[:, :selected_count].astype(jnp.int32)
    all_probabilities = gathered[:, selected_count : 2 * selected_count]
    selected_experts = jnp.zeros((spec.batch_size, selected_count), jnp.int32)
    selected_probabilities = jnp.zeros(
        (spec.batch_size, selected_count),
        jnp.float32,
    )
    local_route_counts = jnp.zeros((spec.batch_size,), jnp.int32)
    route_slots = jnp.arange(selected_count)[None, :]
    for selection in range(selected_count):
        winners = all_winners[:, selection]
        probabilities = all_probabilities[:, selection]
        counts_before_selection = local_route_counts
        owned_locally = (
            winners // spec.local_expert_count == rank // spec.expert_tensor_parallel
        )
        if live_rows is not None:
            owned_locally &= live_rows
        destination_mask = owned_locally[:, None] & (
            route_slots == counts_before_selection[:, None]
        )
        selected_experts = jnp.where(
            destination_mask,
            (winners % spec.local_expert_count)[:, None],
            selected_experts,
        )
        selected_probabilities = jnp.where(
            destination_mask,
            probabilities[:, None],
            selected_probabilities,
        )
        local_route_counts = counts_before_selection + owned_locally.astype(jnp.int32)
        spec.on_local_selection(
            winners,
            owned_locally,
            counts_before_selection,
            selected_experts,
        )
    totals = gathered[:, 2 * selected_count]
    return (
        selected_experts,
        selected_probabilities,
        local_route_counts,
        totals if live_rows is None else jnp.where(live_rows, totals, 1.0),
    )


def _route_owned_rows(
    owned_hidden_ref: TensorLike,
    batch_size: int,
    routing_projection_vmem: PallasRefProtocol,
    routing_bias_ref: TensorLike,
    routing_scores_vmem: PallasRefProtocol,
    selection_source_vmem: PallasRefProtocol | None,
    selections_vmem: PallasRefProtocol | None,
    send_semaphores: PallasRefProtocol,
    receive_semaphores: PallasRefProtocol,
    *,
    selected_expert_count: int,
    local_expert_count: int,
    expert_tensor_parallel: int,
    on_local_selection: _LocalSelectionHook,
    after_selection: _SelectionStepHook,
    after_scores: _PhaseCallback | None = None,
    live_rows: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project and select local expert routes with pair-sharded top-k.

    Each rank pair scores and selects only the rows it owns
    (``owned_hidden_ref`` ``[batch / 4, hidden]``); the pairs exchange their
    selections (global expert ids and probabilities, one FP32 row per token)
    and every rank derives its local routes from all of them. The per-row
    arithmetic and order match the replicated router, so the results are
    identical; only the top-k loop shrinks with the owned rows. Uses
    send/receive semaphores 1..4.
    """
    rank = jax.lax.axis_index(AXIS_TP)
    rows = owned_hidden_ref.shape[0]
    pair = (rank % mla_dcp.HOST_LANES) // 2
    expert_count = routing_bias_ref.shape[-1]
    if rows * _LOCAL_RANK_PAIRS != batch_size or expert_count != ROUTED_EXPERTS:
        msg = "Sharded routing needs one row block per local rank pair"
        raise ValueError(msg)
    if selection_source_vmem is None or selections_vmem is None:
        msg = "Sharded routing requires selection_source and selections scratch"
        raise ValueError(msg)

    router_lane = rank % 2
    local_scores = dot(owned_hidden_ref[...], routing_projection_vmem[0]).reshape(
        rows,
        _ROUTER_LANE_SCORE_WIDTH,
    )
    # Lane 1's last 128 router columns are padding; in the shared arena they
    # hold stale bytes, so zero their scores.
    column = jax.lax.broadcasted_iota(jnp.int32, local_scores.shape, 1)
    local_scores = jnp.where(
        (router_lane == 0) | (column < _ODD_LANE_ROUTER_COLUMNS),
        local_scores,
        0.0,
    )
    score_half = routing_scores_vmem.at[
        pl.ds(0, rows),
        pl.ds(
            router_lane * _ROUTER_LANE_SCORE_WIDTH,
            _ROUTER_LANE_SCORE_WIDTH,
        ),
    ]
    score_half[...] = local_scores
    if after_scores is not None:
        after_scores()  # the router projection is consumed
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
    winners_by_slot: list[jax.Array] = []
    probabilities_by_slot: list[jax.Array] = []
    probability_totals = jnp.zeros((rows,), jnp.float32)
    for selection in range(selected_expert_count):
        winners = jnp.argmax(scores[...], axis=1).astype(jnp.int32)
        winner_mask = expert_ids == winners[:, None]
        probabilities = jnp.sum(
            jnp.where(winner_mask, uncorrected_probabilities, 0),
            axis=1,
        )
        probability_totals += probabilities
        scores[...] = jnp.where(winner_mask, -jnp.inf, scores[...])
        winners_by_slot.append(winners)
        probabilities_by_slot.append(probabilities)
        after_selection(selection)

    # One FP32 row per token: winners (exact small integers), probabilities,
    # and the probability total.
    selection_source_vmem[...] = jnp.concatenate(
        (
            jnp.stack(winners_by_slot, axis=1).astype(jnp.float32),
            jnp.stack(probabilities_by_slot, axis=1),
            probability_totals[:, None],
            jnp.zeros(
                (rows, VREG_LANE_COUNT - 2 * selected_expert_count - 1),
                jnp.float32,
            ),
        ),
        axis=1,
    )
    gathered = _gather_owned_route_selections(
        rank,
        pair,
        rows,
        _OwnedRouteSelectionBuffers(
            selection_source_vmem=selection_source_vmem,
            selections_vmem=selections_vmem,
            send_semaphores=send_semaphores,
            receive_semaphores=receive_semaphores,
        ),
    )
    return _derive_local_routes_from_gathered(
        gathered,
        rank,
        _OwnedRouteDerivationSpec(
            batch_size=batch_size,
            selected_expert_count=selected_expert_count,
            local_expert_count=local_expert_count,
            expert_tensor_parallel=expert_tensor_parallel,
            on_local_selection=on_local_selection,
        ),
        live_rows,
    )


def _start_initial_expert_weight_copies(
    selected_experts: jax.Array,
    local_route_counts: jax.Array,
    copy_expert: _ExpertCopyCallable,
    *,
    slot_count: int,
    already_started: int | jax.Array,
    group_pending_routes: bool,
) -> None:
    """Prefetch the first routed experts in their execution order.

    Uses the same cross-row ordering as the expert stream, so it works for both
    shared and divergent routes without specializing on batch size.
    """
    selected_expert_count = selected_experts.shape[1]
    route_slots = jnp.arange(selected_expert_count)[None, :]
    processed = jnp.zeros(selected_experts.shape, jnp.bool_)

    def next_expert(
        completed: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Select the next pending expert and mark consumed route slots."""
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
                jnp.where(current_slot_mask, selected_experts, 0),
                axis=1,
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
            consumed = (rows_with_routes & (current_experts == expert))[
                :, None
            ] & current_slot_mask
        return expert, consumed, has_expert

    for slot in range(slot_count):
        expert, consumed, has_expert = next_expert(processed)

        @pl.when(has_expert & (slot >= already_started))
        def start_copy(
            expert: jax.Array = expert,
            slot: int = slot,
        ) -> None:
            """Issue async weight copies for ``expert`` into ``slot``."""
            _start_async_copies(copy_expert(expert, slot))

        processed |= consumed


@dataclasses.dataclass(frozen=True, slots=True)
class _Mxfp4DotOperands:
    """Operands and tiling mode for one resident MXFP4 expert projection."""

    source: TensorLike
    packed: TensorLike
    scales: TensorLike
    static_tiles: bool
    split_small_contraction: bool


def _load_mxfp4_dot_tile(
    operands: _Mxfp4DotOperands,
    start: int | jax.Array,
    size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Unpack one contraction slice of MXFP4 weights, scales, and activations."""
    word_start = start // _MXFP4_VALUES_PER_WORD
    word_size = size // _MXFP4_VALUES_PER_WORD
    if operands.static_tiles:
        packed_slice = operands.packed[word_start : word_start + word_size, :]
    else:
        packed_slice = operands.packed[
            pl.ds(
                pl.multiple_of(cast("jax.Array", word_start), _MXFP4_VALUES_PER_WORD),
                word_size,
            ),
            :,
        ]
    values = tpu.bitcast(packed_slice, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)
    scale_start = start // _MXFP4_SCALE_BLOCK
    scale_size = size // _MXFP4_SCALE_BLOCK
    if operands.static_tiles or operands.split_small_contraction:
        # For split 768-wide contractions, the full 24-row scale view is
        # aligned even when a subtile starts halfway through a 32-row tile.
        exponents = operands.scales[...][
            scale_start : scale_start + scale_size,
            :,
        ].astype(jnp.uint32)
    else:
        exponents = operands.scales[
            pl.ds(
                pl.multiple_of(cast("jax.Array", scale_start), _MXFP4_SCALE_BLOCK),
                scale_size,
            ),
            :,
        ].astype(jnp.uint32)
    scale = jax.lax.bitcast_convert_type(
        exponents << _FP32_EXPONENT_SHIFT,
        jnp.float32,
    )
    activation_alignment = (
        VREG_LANE_COUNT
        if operands.split_small_contraction
        else _DEFAULT_ACTIVATION_ALIGNMENT
    )
    if operands.static_tiles:
        activation = operands.source[:, start : start + size].astype(jnp.bfloat16)
    else:
        activation = operands.source[
            :,
            pl.ds(
                pl.multiple_of(cast("jax.Array", start), activation_alignment),
                size,
            ),
        ].astype(jnp.bfloat16)
    return values, scale, activation


def _accumulate_split_mxfp4_subtiles(
    activation: jax.Array,
    values: jax.Array,
    scale: jax.Array,
    accumulator: PallasRefProtocol,
    dot_tile: int,
) -> None:
    """Accumulate split masked subtile products with block-order FP32 sum."""
    size = activation.shape[1]
    partials = []
    for offset in range(0, size, dot_tile):
        width = min(dot_tile, size - offset)
        mask = (
            jnp.arange(width // _MXFP4_SCALE_BLOCK)[:, None]
            == jnp.arange(width)[None, :] // _MXFP4_SCALE_BLOCK
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


def _accumulate_mxfp4_tile(
    activation: jax.Array,
    values: jax.Array,
    scale: jax.Array,
    accumulator: PallasRefProtocol,
    *,
    split_small_contraction: bool,
) -> None:
    """Accumulate one unsplit MXFP4 contraction block into ``accumulator``."""
    size = activation.shape[1]
    scale_blocks = size // _MXFP4_SCALE_BLOCK
    scale_block_mask = (
        jnp.arange(scale_blocks)[:, None]
        == jnp.arange(size)[None, :] // _MXFP4_SCALE_BLOCK
    )
    if split_small_contraction and activation.shape[0] == 1:
        # One row: keep the scale blocks in sublanes instead of padding a
        # [blocks, 1, N] partial. Add them sequentially in the order of the
        # major-axis sum below so both shapes round identically.
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
        for block in range(1, scale_blocks):
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
        accumulator[...] += jnp.sum(partial * scale[:, None, :], axis=0)
    else:
        left = jnp.where(
            scale_block_mask[None],
            activation[:, None, :],
            jnp.bfloat16(0),
        )
        partial = jax.lax.dot_general(
            left,
            values,
            (((2,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        accumulator[...] += jnp.sum(partial * scale[None], axis=1)


def _quantized_expert_dot(
    source: TensorLike,
    packed: TensorLike,
    scales: TensorLike,
    accumulator: PallasRefProtocol,
    contraction_size: int,
    *,
    block_major: bool = False,
    dot_tile: int | None = None,
) -> None:
    """Multiply ``source`` by one resident packed MXFP4 matrix."""
    resolved_dot_tile = (MXFP4_DOT_TILE or 0) if dot_tile is None else dot_tile
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)

    if packed.dtype != jnp.uint32:
        msg = "Expert weights must use packed MXFP4"
        raise ValueError(msg)

    split_small_contraction = (
        contraction_size == _SMALL_EXPERT_CONTRACTION and block_major
    )
    block_size = (
        _SPLIT_CONTRACTION_BLOCK
        if split_small_contraction
        else min(MXFP4_BLOCK, contraction_size)
    )
    static_tiles = MXFP4_STATIC_TILE > 0 and not split_small_contraction
    operands = _Mxfp4DotOperands(
        source=source,
        packed=packed,
        scales=scales,
        static_tiles=static_tiles,
        split_small_contraction=split_small_contraction,
    )

    def tile(start: int | jax.Array, size: int) -> None:
        """Unpack and accumulate one contraction block."""
        values, scale, activation = _load_mxfp4_dot_tile(operands, start, size)
        if (
            resolved_dot_tile
            and not split_small_contraction
            and size > resolved_dot_tile
        ):
            # Split only the masked matrix products. Keep all scale-block
            # partials in block order and perform the same FP32 reduction as
            # the unsplit tile so association stays invariant across steps.
            _accumulate_split_mxfp4_subtiles(
                activation,
                values,
                scale,
                accumulator,
                resolved_dot_tile,
            )
            return
        _accumulate_mxfp4_tile(
            activation,
            values,
            scale,
            accumulator,
            split_small_contraction=split_small_contraction,
        )

    if static_tiles:
        for start in range(0, contraction_size, MXFP4_STATIC_TILE):
            tile(start, min(MXFP4_STATIC_TILE, contraction_size - start))
    elif split_small_contraction:
        for start in range(0, contraction_size, block_size):
            tile(start, block_size)
    else:

        @pl.loop(0, contraction_size // block_size)
        def full_tile(index: jax.Array) -> None:
            """Accumulate one full contraction block."""
            tile(index * block_size, block_size)

        if contraction_size % block_size:
            tile(
                contraction_size // block_size * block_size,
                contraction_size % block_size,
            )


quantized_expert_dot = _quantized_expert_dot


def _paired_expert_gate_up(
    source: TensorLike,
    gate_up_vmem: PallasRefProtocol,
    gate_up_scale_vmem: PallasRefProtocol,
    accumulator: PallasRefProtocol,
    first_slot: int | jax.Array,
    second_slot: int | jax.Array,
    contraction_size: int,
    *,
    second_source: TensorLike | None = None,
    active_capacity: int | None = None,
) -> None:
    """Run two resident expert gate/up projections simultaneously.

    With separate compact MXFP4 sources, only the matching row/weight products
    are consumed. Keep cross-expert entries zero while preserving each
    diagonal's scale reduction and tile accumulation order.
    """
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)

    def tile(start: int, size: int) -> None:
        """Unpack and accumulate one paired gate/up contraction tile."""
        packed = jnp.concatenate(
            (
                gate_up_vmem[
                    first_slot,
                    pl.ds(
                        start // _MXFP4_VALUES_PER_WORD,
                        size // _MXFP4_VALUES_PER_WORD,
                    ),
                    :,
                ],
                gate_up_vmem[
                    second_slot,
                    pl.ds(
                        start // _MXFP4_VALUES_PER_WORD,
                        size // _MXFP4_VALUES_PER_WORD,
                    ),
                    :,
                ],
            ),
            axis=1,
        )
        values = tpu.bitcast(packed, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)
        exponents = jnp.concatenate(
            (
                gate_up_scale_vmem[
                    first_slot,
                    pl.ds(
                        start // _MXFP4_SCALE_BLOCK,
                        size // _MXFP4_SCALE_BLOCK,
                    ),
                    :,
                ],
                gate_up_scale_vmem[
                    second_slot,
                    pl.ds(
                        start // _MXFP4_SCALE_BLOCK,
                        size // _MXFP4_SCALE_BLOCK,
                    ),
                    :,
                ],
            ),
            axis=1,
        ).astype(jnp.uint32)
        scale = jax.lax.bitcast_convert_type(
            exponents << _FP32_EXPONENT_SHIFT,
            jnp.float32,
        )
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
            jnp.arange(size // _MXFP4_SCALE_BLOCK)[:, None]
            == jnp.arange(size)[None, :] // _MXFP4_SCALE_BLOCK
        )
        left = jnp.where(
            scale_block_mask[None],
            activation[:, None, :],
            jnp.bfloat16(0),
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

    # Narrow scaled partials (MXFP4_PAIR_TILE) shorten register live ranges.
    tile_size = MXFP4_PAIR_TILE
    for start in range(0, contraction_size, tile_size):
        tile(start, min(tile_size, contraction_size - start))


def _paired_order_expert_gate_up(
    source: TensorLike,
    packed: TensorLike,
    scales: TensorLike,
    accumulator: PallasRefProtocol,
    contraction_size: int,
    tile_size: int,
) -> None:
    """Project one expert in the paired gate/up FP32 accumulation order.

    Pair membership changes when null routes disappear. Tail and overflow
    projections retain the paired tile's scale reduction and accumulation order
    so an unrelated row cannot change a live expert's rounded result.
    """
    accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)
    for start in range(0, contraction_size, tile_size):
        size = min(tile_size, contraction_size - start)
        values = tpu.bitcast(
            packed[
                pl.ds(
                    start // _MXFP4_VALUES_PER_WORD,
                    size // _MXFP4_VALUES_PER_WORD,
                ),
                :,
            ],
            jnp.float4_e2m1fn,
        ).astype(jnp.float8_e4m3fn)
        exponents = scales[
            pl.ds(start // _MXFP4_SCALE_BLOCK, size // _MXFP4_SCALE_BLOCK),
            :,
        ].astype(jnp.uint32)
        scale = jax.lax.bitcast_convert_type(
            exponents << _FP32_EXPONENT_SHIFT,
            jnp.float32,
        )
        activation = source[:, pl.ds(start, size)].astype(jnp.bfloat16)
        mask = (
            jnp.arange(size // _MXFP4_SCALE_BLOCK)[:, None]
            == jnp.arange(size)[None, :] // _MXFP4_SCALE_BLOCK
        )
        left = jnp.where(mask[None], activation[:, None, :], jnp.bfloat16(0))
        partial = jax.lax.dot_general(
            left,
            values,
            (((2,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        accumulator[...] += jnp.sum(partial * scale[None], axis=1)


def _dot_scoped[**P, R](
    function: Callable[P, R],
    phase_name: str,
) -> Callable[P, R]:
    """Wrap a routed-expert product in its ``phase_name`` tracing scope."""

    @functools.wraps(function)
    def scoped(*args: P.args, **kwargs: P.kwargs) -> R:
        """Execute ``function`` inside ``phase_name``."""
        with _phase(phase_name):
            return function(*args, **kwargs)

    return scoped


if PHASE_SCOPES:
    _quantized_expert_dot = _dot_scoped(_quantized_expert_dot, "e_dot_mxfp4")
    quantized_expert_dot = _quantized_expert_dot
    _paired_expert_gate_up = _dot_scoped(_paired_expert_gate_up, "e_dot_pair")


@dataclasses.dataclass(frozen=True, slots=True)
class _RoutedExpertStreamContext:
    """Resolved buffers, geometry, and schedule flags for routed experts."""

    source: TensorLike
    selected_experts: jax.Array
    selected_probabilities: jax.Array
    local_route_counts: jax.Array
    probability_totals: jax.Array
    route_slots: jax.Array
    expert_gate_up_vmem: PallasRefProtocol
    expert_gate_up_scale_vmem: PallasRefProtocol
    expert_down_vmem: PallasRefProtocol
    expert_down_scale_vmem: PallasRefProtocol
    expert_output: PallasRefProtocol
    gate_up_accumulator: PallasRefProtocol
    down_accumulator: PallasRefProtocol
    processed_routes_vmem: PallasRefProtocol
    first_compact_source_vmem: PallasRefProtocol
    second_compact_source_vmem: PallasRefProtocol
    first_single_source: PallasRefProtocol
    second_single_source: PallasRefProtocol
    copy_expert: _ExpertCopyCallable
    batch_size: int
    selected_expert_count: int
    compact_capacity: int
    expert_input_size: int
    expert_width: int
    expert_output_size: int
    dot_tile: int
    singleton_block: bool
    stage_blocks: bool
    reuse_row_plan: bool
    compact_overflow: bool
    compact_only: bool
    share_compact_finish: bool
    share_full_finish: bool
    group_pending_routes: bool
    progress: _ProgressHook | None


@dataclasses.dataclass(frozen=True, slots=True)
class _ExpertSlotTarget:
    """Route masks, copy handles, and accumulator placement for one expert."""

    active_rows: jax.Array | None
    current_slot_mask: jax.Array | None
    copies: SimpleNamespace
    column: int | jax.Array
    slot: int | jax.Array


@dataclasses.dataclass(frozen=True, slots=True)
class _ExpertPairIteration:
    """Selected expert pair, row masks, and copy descriptors for one step."""

    first_expert: jax.Array
    first_rows: jax.Array
    first_slot_mask: jax.Array
    first_consumed: jax.Array
    has_first: jax.Array
    second_expert: jax.Array
    second_rows: jax.Array
    second_slot_mask: jax.Array
    second_consumed: jax.Array
    has_second: jax.Array
    current_base_slot: jax.Array
    first_copies: SimpleNamespace
    second_copies: SimpleNamespace


def _resolve_expert_dot_tile(batch_size: int) -> int:
    """Return the split MXFP4 contraction subtile width for ``batch_size``.

    Narrow masked products avoid zero MACs and register spills while keeping
    the unsplit scale reduction tree. ``KernelConfig.mxfp4_dot_tile`` overrides
    the per-batch default (0: unsplit dots).
    """
    if MXFP4_DOT_TILE is not None:
        return MXFP4_DOT_TILE
    if batch_size == _WIDE_DOT_TILE_BATCH:
        return _WIDE_BATCH_DOT_TILE
    if batch_size in POWER_OF_TWO_BATCHES:
        return _DEFAULT_SPLIT_DOT_TILE
    return 0


def _wrap_scoped_expert_copy(
    copy_expert: _ExpertCopyCallable,
) -> _ExpertCopyCallable:
    """Wrap ``copy_expert`` descriptors so waits trace inside ``e_wait``."""

    def scoped_copy_expert(
        expert: int | jax.Array,
        slot: int | jax.Array,
        *,
        gate_up: bool = True,
        down: bool = True,
    ) -> SimpleNamespace:
        """Build phase-scoped async copy handles for ``expert`` in ``slot``."""
        copies = copy_expert(expert, slot, gate_up=gate_up, down=down)
        return SimpleNamespace(
            **{
                name: None if copy is None else _ScopedWait(copy)
                for name, copy in vars(copies).items()
            }
        )

    return scoped_copy_expert


def _expert_gate_up_dot(
    ctx: _RoutedExpertStreamContext,
    source: TensorLike,
    slot: int | jax.Array,
    accumulator: PallasRefProtocol,
) -> None:
    """Project ``source`` through one expert's gate/up weights in ``slot``."""
    if ctx.batch_size > 1:
        # A live expert may switch between a pair and the unpaired tail as
        # other rows enter or leave; match the paired reduction tree.
        _paired_order_expert_gate_up(
            source,
            ctx.expert_gate_up_vmem.at[slot],
            ctx.expert_gate_up_scale_vmem.at[slot],
            accumulator,
            ctx.expert_input_size,
            MXFP4_PAIR_TILE,
        )
        return
    _quantized_expert_dot(
        source,
        ctx.expert_gate_up_vmem.at[slot],
        ctx.expert_gate_up_scale_vmem.at[slot],
        accumulator,
        ctx.expert_input_size,
        dot_tile=ctx.dot_tile,
    )


def _expert_down_dot(
    ctx: _RoutedExpertStreamContext,
    source: TensorLike,
    slot: int | jax.Array,
    accumulator: PallasRefProtocol,
    *,
    block_major: bool,
) -> None:
    """Project ``source`` through one expert's down weights in ``slot``."""
    if ctx.batch_size > 1 and ctx.expert_width == ctx.expert_input_size:
        _paired_order_expert_gate_up(
            source,
            ctx.expert_down_vmem.at[slot],
            ctx.expert_down_scale_vmem.at[slot],
            accumulator,
            ctx.expert_width,
            MXFP4_PAIR_TILE,
        )
        return
    _quantized_expert_dot(
        source,
        ctx.expert_down_vmem.at[slot],
        ctx.expert_down_scale_vmem.at[slot],
        accumulator,
        ctx.expert_width,
        dot_tile=ctx.dot_tile,
        block_major=block_major,
    )


def _select_next_routed_expert(
    ctx: _RoutedExpertStreamContext,
    processed: jax.Array,
) -> _RouteChoice:
    """Select the lowest-numbered ready expert and its active row masks."""
    pending = (ctx.route_slots < ctx.local_route_counts[:, None]) & ~processed
    if ctx.group_pending_routes:
        minimum_expert = jnp.min(
            jnp.where(
                pending,
                ctx.selected_experts.astype(jnp.float32),
                jnp.inf,
            )
        )
        has_expert = jnp.any(pending)
        expert = jnp.where(has_expert, minimum_expert, 0).astype(jnp.int32)
        selected_slot_mask = pending & (ctx.selected_experts == expert)
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
    current_slot_mask = ctx.route_slots == current_slots[:, None]
    current_experts = jnp.sum(
        jnp.where(current_slot_mask, ctx.selected_experts, 0),
        axis=1,
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


def _finish_full_expert(
    ctx: _RoutedExpertStreamContext,
    target: _ExpertSlotTarget,
    output_weights: jax.Array | None = None,
) -> None:
    """Activate, down-project, and accumulate a full-batch routed expert."""
    column_slice = (
        pl.ds(
            pl.multiple_of(cast("jax.Array", target.column), VREG_LANE_COUNT),
            2 * ctx.expert_width,
        )
        if ctx.share_full_finish
        else slice(target.column, target.column + 2 * ctx.expert_width)
    )
    gate_up = (
        ctx.gate_up_accumulator[:, column_slice]
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    middle = _gated_activation(
        gate_up[:, : ctx.expert_width],
        gate_up[:, ctx.expert_width :],
    )
    ctx.gate_up_accumulator[:, : ctx.expert_width] = middle.astype(jnp.float32)
    target.copies.down.wait()
    target.copies.down_scales.wait()
    _expert_down_dot(
        ctx,
        ctx.gate_up_accumulator.at[:, : ctx.expert_width],
        target.slot,
        ctx.down_accumulator,
        # Keep the down projection's scale reductions independent of how many
        # rows happen to route to this expert.
        block_major=ctx.batch_size >= _COMPACT_ROUTE_MIN_BATCH,
    )
    if output_weights is None:
        if target.active_rows is None or target.current_slot_mask is None:
            msg = "Active rows and slot mask are required without output_weights"
            raise ValueError(msg)
        route_probabilities = (
            jnp.sum(
                jnp.where(
                    target.current_slot_mask,
                    ctx.selected_probabilities,
                    jnp.float32(0),
                ),
                axis=1,
            )
            / ctx.probability_totals
        )
        output_weights = jnp.where(
            target.active_rows,
            route_probabilities,
            jnp.float32(0),
        )
    ctx.expert_output[:, : ctx.expert_output_size] += (
        ctx.down_accumulator[...].astype(jnp.bfloat16).astype(jnp.float32)
        * output_weights[:, None]
    )


def _finish_singleton_expert(
    ctx: _RoutedExpertStreamContext,
    target: _ExpertSlotTarget,
    compact_row: int,
) -> None:
    """Finish one expert whose route is active for exactly one batch row."""
    if target.active_rows is None or target.current_slot_mask is None:
        msg = "Singleton expert finish requires active_rows and current_slot_mask"
        raise ValueError(msg)
    gate_up = (
        ctx.gate_up_accumulator[
            compact_row : compact_row + 1,
            target.column : target.column + 2 * ctx.expert_width,
        ]
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    middle = _gated_activation(
        gate_up[:, : ctx.expert_width],
        gate_up[:, ctx.expert_width :],
    )
    ctx.gate_up_accumulator[
        compact_row : compact_row + 1,
        : ctx.expert_width,
    ] = middle.astype(jnp.float32)
    target.copies.down.wait()
    target.copies.down_scales.wait()
    _expert_down_dot(
        ctx,
        ctx.gate_up_accumulator.at[
            compact_row : compact_row + 1,
            : ctx.expert_width,
        ],
        target.slot,
        ctx.down_accumulator.at[:1],
        # Use the 256-wide block-major accumulation of the compact/full paths
        # so a singleton row rounds the same as in a group.
        block_major=True,
    )
    route_probabilities = (
        jnp.sum(
            jnp.where(
                target.current_slot_mask,
                ctx.selected_probabilities,
                jnp.float32(0),
            ),
            axis=1,
        )
        / ctx.probability_totals
    )
    singleton_output = ctx.down_accumulator[:1].astype(jnp.bfloat16).astype(jnp.float32)
    row = jnp.sum(jnp.where(target.active_rows, jnp.arange(ctx.batch_size), 0))
    probability = jnp.sum(
        jnp.where(target.active_rows, route_probabilities, jnp.float32(0))
    )
    output_row = ctx.expert_output.at[pl.ds(row, 1), : ctx.expert_output_size]
    output_row[...] += singleton_output * probability


def _stage_singleton_source(
    ctx: _RoutedExpertStreamContext,
    active_rows: jax.Array,
    destination: PallasRefProtocol,
) -> None:
    """Copy a singleton route's one active row into ``destination``."""
    if ctx.singleton_block and ctx.batch_size % _BF16_TILE_ROWS == 0:
        active_index = jnp.sum(jnp.where(active_rows, jnp.arange(ctx.batch_size), 0))
        block = ctx.source[
            pl.ds(
                pl.multiple_of(
                    active_index // _BF16_TILE_ROWS * _BF16_TILE_ROWS,
                    _BF16_TILE_ROWS,
                ),
                _BF16_TILE_ROWS,
            ),
            : ctx.expert_input_size,
        ].astype(jnp.float32)
        within = (
            jax.lax.broadcasted_iota(jnp.int32, block.shape, 0)
            == active_index % _BF16_TILE_ROWS
        )
        destination[...] = jnp.sum(
            jnp.where(within, block, 0.0),
            axis=0,
            keepdims=True,
        ).astype(destination.dtype)
        return
    active_index = jnp.sum(jnp.where(active_rows, jnp.arange(ctx.batch_size), 0))
    for row in range(ctx.batch_size):

        @pl.when(active_index == row)
        def copy_indexed_row(row: int = row) -> None:
            """Copy active ``row`` into the singleton staging buffer."""
            destination[...] = ctx.source[
                row : row + 1,
                : ctx.expert_input_size,
            ].astype(destination.dtype)


def _row_bitmask_words(
    batch_size: int,
    active_rows: jax.Array,
) -> list[jax.Array]:
    """Pack active row flags into 32-bit scalar bitmask words."""
    index = jnp.arange(batch_size)
    bits = active_rows.astype(jnp.int32) << (index % BITS_PER_WORD)
    return [
        jnp.sum(jnp.where(index // BITS_PER_WORD == word, bits, 0))
        for word in range(-(-batch_size // BITS_PER_WORD))
    ]


def _compact_row_positions(
    ctx: _RoutedExpertStreamContext,
    active_rows: jax.Array,
) -> jax.Array:
    """Compute exclusive prefix counts of ``active_rows`` across the batch."""
    # Counts and row indices are < 128, hence exact in FP32, avoiding split
    # reductions used for unrestricted int32.
    count_dtype = jnp.float32 if ctx.reuse_row_plan else jnp.int32
    index = jnp.arange(ctx.batch_size)
    return jnp.sum(
        jnp.where(
            index[None, :] < index[:, None],
            active_rows[None, :].astype(count_dtype),
            0,
        ),
        axis=1,
    ).astype(jnp.int32)


def _compact_row_plan(
    ctx: _RoutedExpertStreamContext,
    active_rows: jax.Array,
    offset: int | jax.Array = 0,
    before: jax.Array | None = None,
) -> _RowPlan:
    """Build ordered batch-row indices and validity mask for one pass."""
    if before is None:
        before = _compact_row_positions(ctx, active_rows)
    # One ordered index list serves both input staging and output selection.
    # Invalid trailing entries must not select source row 0.
    selected = active_rows[None, :] & (
        before[None, :] == offset + jnp.arange(ctx.compact_capacity)[:, None]
    )
    indices = jnp.sum(
        jnp.where(
            selected,
            jnp.arange(ctx.batch_size).astype(jnp.float32)[None, :],
            0.0,
        ),
        axis=1,
    ).astype(jnp.int32)
    return indices, jnp.any(selected, axis=1)


def _stage_planned_rows(
    ctx: _RoutedExpertStreamContext,
    row_plan: _RowPlan,
    destination: PallasRefProtocol,
) -> None:
    """Stage compact active rows described by ``row_plan`` into ``destination``."""
    indices, valid = row_plan
    if ctx.batch_size in _ONE_HOT_STAGE_BATCHES:
        # A one-hot product selects the planned rows (exact: one BF16 term per
        # entry, zero rows past the plan) without per-row scalar gathers.
        one_hot = (
            (indices[:, None] == jnp.arange(ctx.batch_size)[None, :]) & valid[:, None]
        ).astype(jnp.bfloat16)
        destination[...] = dot(one_hot, ctx.source[:, : ctx.expert_input_size])
        return
    destination[...] = jnp.zeros(destination.shape, destination.dtype)

    if ctx.batch_size == _BLOCK_GATHER_BATCH:

        @pl.loop(0, ctx.batch_size // _BF16_TILE_ROWS)
        def stage_block(block: jax.Array) -> None:
            """Gather planned rows belonging to aligned 16-row ``block``."""
            in_block = valid & (indices // _BF16_TILE_ROWS == block)

            @pl.when(jnp.any(in_block))
            def copy_block() -> None:
                """Gather in-bounds rows from ``block`` into ``destination``."""
                rows = ctx.source[
                    pl.ds(
                        pl.multiple_of(
                            block * _BF16_TILE_ROWS,
                            _BF16_TILE_ROWS,
                        ),
                        _BF16_TILE_ROWS,
                    ),
                    : ctx.expert_input_size,
                ]
                gather_indices = jnp.broadcast_to(
                    (indices % _BF16_TILE_ROWS).astype(jnp.int16)[:, None],
                    (ctx.compact_capacity, ctx.expert_input_size),
                )
                selected_rows = jnp.take_along_axis(
                    rows,
                    gather_indices,
                    axis=0,
                    mode="promise_in_bounds",
                    wrap_negative_indices=False,
                ).astype(jnp.float32)
                destination[...] = jnp.where(
                    in_block[:, None],
                    selected_rows,
                    destination[...],
                )

        return

    # Each gather stays inside one native 16-row BF16 tile; matching int16
    # indices avoid unsupported mixed-width gathers.
    @pl.loop(0, jnp.sum(valid.astype(jnp.float32)).astype(jnp.int32))
    def copy_row(position: jax.Array) -> None:
        """Gather single planned row at ``position`` into ``destination``."""
        index = jnp.sum(
            jnp.where(
                jnp.arange(ctx.compact_capacity) == position,
                indices.astype(jnp.float32),
                0.0,
            )
        ).astype(jnp.int32)
        rows = ctx.source[
            pl.ds(
                pl.multiple_of(
                    index // _BF16_TILE_ROWS * _BF16_TILE_ROWS,
                    _BF16_TILE_ROWS,
                ),
                _BF16_TILE_ROWS,
            ),
            : ctx.expert_input_size,
        ]
        gather_indices = jnp.full(
            (1, ctx.expert_input_size),
            index % _BF16_TILE_ROWS,
            jnp.int16,
        )
        selected_row = jnp.take_along_axis(
            rows,
            gather_indices,
            axis=0,
            mode="promise_in_bounds",
            wrap_negative_indices=False,
        ).astype(jnp.float32)
        destination[pl.ds(position, 1), :] = selected_row


def _stage_b16_bitmask_rows(
    ctx: _RoutedExpertStreamContext,
    active_rows: jax.Array,
    destination: PallasRefProtocol,
    offset: int | jax.Array,
) -> None:
    """Stage compact rows at B16 using a scalar scan over the row bitmask."""
    (word,) = _row_bitmask_words(ctx.batch_size, active_rows)

    def stage_block(block: jax.Array, position: jax.Array) -> jax.Array:
        """Scan 16-bit row mask for ``block`` and copy selected rows."""
        first = pl.multiple_of(block * _BF16_TILE_ROWS, _BF16_TILE_ROWS)
        bits = (word >> (block * _BF16_TILE_ROWS)) & _SCALAR_SCAN_MASK_16

        def stage_nonempty(current_position: jax.Array) -> jax.Array:
            """Advance ``current_position`` across 16 rows in ``block``."""
            for row in range(_BF16_TILE_ROWS):
                bit = (bits >> row) & 1
                take = (
                    (bit != 0)
                    & (current_position >= offset)
                    & (current_position < offset + ctx.compact_capacity)
                )

                @pl.when(take)
                def copy_block_row(
                    row: int = row,
                    position: jax.Array = current_position,
                ) -> None:
                    """Copy ``row`` from ``block`` into compact slot."""
                    rows = ctx.source[
                        pl.ds(first, _BF16_TILE_ROWS),
                        : ctx.expert_input_size,
                    ]
                    destination[pl.ds(position - offset, 1), :] = rows[
                        row : row + 1
                    ].astype(jnp.float32)

                current_position = current_position + bit
            return current_position

        return stage_nonempty(position)

    jax.lax.fori_loop(
        0,
        ctx.batch_size // _BF16_TILE_ROWS,
        stage_block,
        jnp.int32(0),
    )


def _stage_aligned_block_rows(
    ctx: _RoutedExpertStreamContext,
    active_rows: jax.Array,
    destination: PallasRefProtocol,
    offset: int | jax.Array,
) -> None:
    """Stage compact rows across 16-row aligned batch blocks."""
    index = jnp.arange(ctx.batch_size)
    active = active_rows.astype(jnp.int32)

    @pl.loop(0, ctx.batch_size // _BF16_TILE_ROWS)
    def stage_block(block: jax.Array) -> None:
        """Stage active rows belonging to 16-row ``block``."""
        first = pl.multiple_of(block * _BF16_TILE_ROWS, _BF16_TILE_ROWS)
        bits = jnp.sum(
            jnp.where(
                index // _BF16_TILE_ROWS == block,
                active << (index % _BF16_TILE_ROWS),
                0,
            )
        )
        position = jnp.sum(jnp.where(index < block * _BF16_TILE_ROWS, active, 0))
        for row in range(_BF16_TILE_ROWS):
            bit = (bits >> row) & 1
            take = (
                (bit != 0)
                & (position >= offset)
                & (position < offset + ctx.compact_capacity)
            )

            @pl.when(take)
            def copy_block_row(
                row: int = row,
                position: jax.Array = position,
            ) -> None:
                """Copy ``row`` from ``block`` into compact slot."""
                rows = ctx.source[
                    pl.ds(first, _BF16_TILE_ROWS),
                    : ctx.expert_input_size,
                ]
                destination[pl.ds(position - offset, 1), :] = rows[
                    row : row + 1
                ].astype(jnp.float32)

            position = position + bit


def _stage_scanned_bitmask_rows(
    ctx: _RoutedExpertStreamContext,
    active_rows: jax.Array,
    destination: PallasRefProtocol,
    offset: int | jax.Array,
) -> None:
    """Stage compact rows via per-row bitmask scan for unaligned batches."""
    words = _row_bitmask_words(ctx.batch_size, active_rows)
    position = jnp.int32(0)
    static_zero_offset = isinstance(offset, int) and offset == 0
    for row in range(ctx.batch_size):
        bit = (words[row // BITS_PER_WORD] >> (row % BITS_PER_WORD)) & 1
        if static_zero_offset:
            take = bit != 0
        else:
            take = (
                (bit != 0)
                & (position >= offset)
                & (position < offset + ctx.compact_capacity)
            )

        @pl.when(take)
        def copy_active_row(
            row: int = row,
            position: jax.Array = position,
        ) -> None:
            """Copy active ``row`` into its compacted destination slot."""
            destination[pl.ds(position - offset, 1), :] = ctx.source[
                row : row + 1,
                : ctx.expert_input_size,
            ].astype(jnp.float32)

        position = position + bit


def _stage_compact_rows(
    ctx: _RoutedExpertStreamContext,
    active_rows: jax.Array,
    destination: PallasRefProtocol,
    offset: int | jax.Array = 0,
    row_plan: _RowPlan | None = None,
) -> None:
    """Copy active rows ``offset .. offset + C - 1`` into ``destination``."""
    if ctx.reuse_row_plan:
        if row_plan is None:
            row_plan = _compact_row_plan(ctx, active_rows, offset)
        _stage_planned_rows(ctx, row_plan, destination)
        return
    destination[...] = jnp.zeros(destination.shape, destination.dtype)
    if ctx.stage_blocks and ctx.batch_size == _SCALAR_SCAN_BATCH:
        _stage_b16_bitmask_rows(ctx, active_rows, destination, offset)
        return
    if ctx.stage_blocks and ctx.batch_size % _BF16_TILE_ROWS == 0:
        _stage_aligned_block_rows(ctx, active_rows, destination, offset)
        return
    _stage_scanned_bitmask_rows(ctx, active_rows, destination, offset)


def _compact_route_data(
    ctx: _RoutedExpertStreamContext,
    active_rows: jax.Array,
    current_slot_mask: jax.Array,
    offset: int | jax.Array = 0,
    row_plan: _RowPlan | None = None,
) -> _CompactRouteData:
    """Build output row selector matrix and normalized weights for one pass."""
    route_probabilities = (
        jnp.sum(
            jnp.where(
                current_slot_mask,
                ctx.selected_probabilities,
                jnp.float32(0),
            ),
            axis=1,
        )
        / ctx.probability_totals
    )
    if ctx.reuse_row_plan:
        if row_plan is None:
            row_plan = _compact_row_plan(ctx, active_rows, offset)
        indices, valid = row_plan
        selected = (jnp.arange(ctx.batch_size)[:, None] == indices[None, :]) & valid[
            None, :
        ]
        selector = selected.astype(jnp.bfloat16)
        in_pass = jnp.any(selected, axis=1)
    else:
        before = _compact_row_positions(ctx, active_rows)
        selector = (
            (before[:, None] - offset == jnp.arange(ctx.compact_capacity)[None, :])
            & active_rows[:, None]
        ).astype(jnp.bfloat16)
        in_pass = (
            active_rows & (before >= offset) & (before < offset + ctx.compact_capacity)
        )
    return selector, jnp.where(in_pass, route_probabilities, jnp.float32(0))


def _finish_compact_expert(
    ctx: _RoutedExpertStreamContext,
    target: _ExpertSlotTarget,
    first_row: int | jax.Array,
    route_data: _CompactRouteData | None = None,
    *,
    wait_down: bool = True,
) -> None:
    """Finish one expert whose active rows sit compacted from ``first_row``."""
    rows = (
        pl.ds(
            pl.multiple_of(cast("jax.Array", first_row), SUBLANE_ROWS),
            ctx.compact_capacity,
        )
        if ctx.share_compact_finish
        else slice(first_row, first_row + ctx.compact_capacity)
    )
    columns = (
        pl.ds(
            pl.multiple_of(cast("jax.Array", target.column), VREG_LANE_COUNT),
            2 * ctx.expert_width,
        )
        if ctx.share_compact_finish
        else slice(target.column, target.column + 2 * ctx.expert_width)
    )
    gate_up = (
        ctx.gate_up_accumulator[rows, columns].astype(jnp.bfloat16).astype(jnp.float32)
    )
    middle = _gated_activation(
        gate_up[:, : ctx.expert_width],
        gate_up[:, ctx.expert_width :],
    )
    ctx.gate_up_accumulator[rows, : ctx.expert_width] = middle.astype(jnp.float32)
    if wait_down:
        target.copies.down.wait()
        target.copies.down_scales.wait()
    _expert_down_dot(
        ctx,
        ctx.gate_up_accumulator.at[rows, : ctx.expert_width],
        target.slot,
        ctx.down_accumulator.at[: ctx.compact_capacity],
        block_major=True,
    )
    if route_data is None:
        if target.active_rows is None or target.current_slot_mask is None:
            msg = "Active rows and slot mask are required without route_data"
            raise ValueError(msg)
        route_data = _compact_route_data(
            ctx,
            target.active_rows,
            target.current_slot_mask,
            0,
        )
    selector, output_weights = route_data
    selected = dot(
        selector,
        ctx.down_accumulator[: ctx.compact_capacity].astype(jnp.bfloat16),
    )
    ctx.expert_output[:, : ctx.expert_output_size] += selected * output_weights[:, None]


def _run_expert_in_passes(
    ctx: _RoutedExpertStreamContext,
    target: _ExpertSlotTarget,
    row_count: jax.Array,
    *,
    paired_gate: bool = False,
) -> None:
    """Execute one overflowing expert in passes of ``compact_capacity`` rows."""
    if target.active_rows is None or target.current_slot_mask is None:
        msg = "Multi-pass expert execution requires active_rows and slot mask"
        raise ValueError(msg)
    active_rows = target.active_rows
    current_slot_mask = target.current_slot_mask
    target.copies.gate_up.wait()
    target.copies.gate_up_scales.wait()
    target.copies.down.wait()
    target.copies.down_scales.wait()

    before = _compact_row_positions(ctx, active_rows) if ctx.reuse_row_plan else None

    @pl.loop(0, (row_count + ctx.compact_capacity - 1) // ctx.compact_capacity)
    def run_pass(index: jax.Array) -> None:
        """Stage, project, and scatter one compact pass of active rows."""
        offset = index * ctx.compact_capacity
        row_plan = (
            _compact_row_plan(ctx, active_rows, offset, before)
            if ctx.reuse_row_plan
            else None
        )
        _stage_compact_rows(
            ctx,
            active_rows,
            ctx.first_compact_source_vmem,
            offset,
            row_plan,
        )
        accumulator_slice = ctx.gate_up_accumulator.at[
            : ctx.compact_capacity,
            : 2 * ctx.expert_width,
        ]
        if paired_gate:
            _paired_order_expert_gate_up(
                ctx.first_compact_source_vmem,
                ctx.expert_gate_up_vmem.at[target.slot],
                ctx.expert_gate_up_scale_vmem.at[target.slot],
                accumulator_slice,
                ctx.expert_input_size,
                MXFP4_PAIR_TILE,
            )
        else:
            _expert_gate_up_dot(
                ctx,
                ctx.first_compact_source_vmem,
                target.slot,
                accumulator_slice,
            )
        # Every pass owns its rows; finish before reusing the compact
        # accumulator. Weight completions were consumed once above.
        _finish_compact_expert(
            ctx,
            _ExpertSlotTarget(
                active_rows=active_rows,
                current_slot_mask=current_slot_mask,
                copies=target.copies,
                column=0,
                slot=target.slot,
            ),
            0,
            _compact_route_data(
                ctx,
                active_rows,
                current_slot_mask,
                offset,
                row_plan,
            ),
            wait_down=False,
        )


def _shared_route_expert_at(
    ctx: _RoutedExpertStreamContext,
    index: jax.Array,
) -> jax.Array:
    """Return the expert id at route slot ``index`` when routes are shared."""
    return jnp.sum(
        jnp.where(
            ctx.route_slots[0] == index,
            ctx.selected_experts[0],
            jnp.int32(0),
        )
    ).astype(jnp.int32)


def _finish_shared_route_expert(
    ctx: _RoutedExpertStreamContext,
    index: jax.Array,
    slot: jax.Array,
    column: int,
) -> None:
    """Finish one shared-route expert and prefetch the expert four slots ahead."""
    gate_up = (
        ctx.gate_up_accumulator[:, column : column + 2 * ctx.expert_width]
        .astype(jnp.bfloat16)
        .astype(jnp.float32)
    )
    middle = _gated_activation(
        gate_up[:, : ctx.expert_width],
        gate_up[:, ctx.expert_width :],
    )
    ctx.gate_up_accumulator[:, : ctx.expert_width] = middle.astype(jnp.float32)
    down_copies = ctx.copy_expert(
        _shared_route_expert_at(ctx, index),
        slot,
        gate_up=False,
    )
    down_copies.down.wait()
    down_copies.down_scales.wait()
    _expert_down_dot(
        ctx,
        ctx.gate_up_accumulator.at[:, : ctx.expert_width],
        slot,
        ctx.down_accumulator,
        block_major=ctx.batch_size >= _COMPACT_ROUTE_MIN_BATCH,
    )
    probabilities = jnp.sum(
        jnp.where(
            ctx.route_slots == index,
            ctx.selected_probabilities,
            jnp.float32(0),
        ),
        axis=1,
    )
    route_weights = (probabilities / ctx.probability_totals)[:, None]
    ctx.expert_output[:, : ctx.expert_output_size] += (
        ctx.down_accumulator[...].astype(jnp.bfloat16).astype(jnp.float32)
        * route_weights
    )

    @pl.when(index + _EXPERT_PREFETCH_SLOTS < ctx.local_route_counts[0])
    def refill_expert() -> None:
        """Prefetch the shared route four positions ahead into ``slot``."""
        _start_async_copies(
            ctx.copy_expert(
                _shared_route_expert_at(ctx, index + _EXPERT_PREFETCH_SLOTS),
                slot,
            )
        )


def _execute_shared_routes(ctx: _RoutedExpertStreamContext) -> None:
    """Execute paired and tail experts when all batch rows share routes."""
    if ctx.compact_only:
        return

    def wait_for_gate_up(index: jax.Array, slot: jax.Array) -> None:
        """Wait for gate/up weights and scales of shared route ``index``."""
        copies = ctx.copy_expert(
            _shared_route_expert_at(ctx, index),
            slot,
            down=False,
        )
        copies.gate_up.wait()
        copies.gate_up_scales.wait()

    @pl.loop(0, ctx.local_route_counts[0] // 2)
    def run_shared_pair(pair: jax.Array) -> None:
        """Execute one pair of shared-route experts across all batch rows."""
        first = 2 * pair
        first_slot = first % _EXPERT_PREFETCH_SLOTS
        second_slot = (first + 1) % _EXPERT_PREFETCH_SLOTS
        wait_for_gate_up(first, first_slot)
        wait_for_gate_up(first + 1, second_slot)
        _paired_expert_gate_up(
            ctx.source,
            ctx.expert_gate_up_vmem,
            ctx.expert_gate_up_scale_vmem,
            ctx.gate_up_accumulator,
            first_slot,
            second_slot,
            ctx.expert_input_size,
        )
        _finish_shared_route_expert(ctx, first, first_slot, 0)
        _finish_shared_route_expert(
            ctx,
            first + 1,
            second_slot,
            2 * ctx.expert_width,
        )

    @pl.when(ctx.local_route_counts[0] % 2 == 1)
    def run_shared_tail() -> None:
        """Execute the trailing odd shared-route expert when present."""
        index = ctx.local_route_counts[0] - 1
        slot = index % _EXPERT_PREFETCH_SLOTS
        wait_for_gate_up(index, slot)
        _expert_gate_up_dot(
            ctx,
            ctx.source,
            slot,
            ctx.gate_up_accumulator.at[:, : 2 * ctx.expert_width],
        )
        _finish_shared_route_expert(ctx, index, slot, 0)


def _count_divergent_expert_pairs(
    ctx: _RoutedExpertStreamContext,
    valid_routes: jax.Array,
) -> jax.Array:
    """Compute the pair loop bound for grouped or row-ordered routes."""
    if ctx.group_pending_routes:
        # Each iteration serves every pending route of its two experts, so the
        # loop needs one iteration per two distinct experts (local ids < 128).
        used = jnp.zeros((1, _MAX_LOCAL_EXPERTS), jnp.bool_)
        for route_slot in range(ctx.selected_expert_count):
            slot_valid = valid_routes[:, route_slot : route_slot + 1]
            slot_experts = ctx.selected_experts[:, route_slot : route_slot + 1]
            used |= jnp.any(
                slot_valid & (slot_experts == jnp.arange(_MAX_LOCAL_EXPERTS)[None, :]),
                axis=0,
                keepdims=True,
            )
        return (jnp.sum(used.astype(jnp.int32)) + 1) // 2
    # Rows advance through their own route order, so one expert can be visited
    # more than once: bound the loop by the total route count.
    return (jnp.sum(ctx.local_route_counts) + 1) // 2


def _pack_route_choice(choice: _RouteChoice) -> _PackedRouteChoice:
    """Cast a route selection tuple to int32 arrays for loop carry."""
    return tuple(value.astype(jnp.int32) for value in choice)


def _unpack_route_choice(choice: _PackedRouteChoice) -> _RouteChoice:
    """Restore boolean masks from an int32-packed route selection tuple."""
    return (
        choice[0],
        choice[1] != 0,
        choice[2] != 0,
        choice[3] != 0,
        choice[4] != 0,
    )


def _select_and_prefetch_expert_pair(
    ctx: _RoutedExpertStreamContext,
    iteration: jax.Array,
    carry: tuple[_PackedRouteChoice, _PackedRouteChoice] | None,
    *,
    carry_routes: bool,
) -> tuple[_ExpertPairIteration, _RouteChoice, _RouteChoice]:
    """Select the active expert pair and prefetch the next pair's weights."""
    select_scope = phase_start("e_select")
    processed = ctx.processed_routes_vmem[:, : ctx.selected_expert_count] != 0
    if carry_routes:
        if carry is None:
            msg = "Carried route state is required when carry_routes is enabled"
            raise ValueError(msg)
        (
            first_expert,
            first_rows,
            first_slot_mask,
            first_consumed,
            has_first,
        ) = _unpack_route_choice(carry[0])
        (
            second_expert,
            second_rows,
            second_slot_mask,
            second_consumed,
            has_second,
        ) = _unpack_route_choice(carry[1])
    else:
        (
            first_expert,
            first_rows,
            first_slot_mask,
            first_consumed,
            has_first,
        ) = _select_next_routed_expert(ctx, processed)
        (
            second_expert,
            second_rows,
            second_slot_mask,
            second_consumed,
            has_second,
        ) = _select_next_routed_expert(ctx, processed | first_consumed)
    processed_after_pair = processed | first_consumed | second_consumed
    third_choice = _select_next_routed_expert(ctx, processed_after_pair)
    fourth_choice = _select_next_routed_expert(
        ctx,
        processed_after_pair | third_choice[3],
    )

    current_base_slot = (iteration % 2) * 2
    next_base_slot = ((iteration + 1) % 2) * 2
    first_copies = ctx.copy_expert(first_expert, current_base_slot)
    second_copies = ctx.copy_expert(second_expert, current_base_slot + 1)
    third_copies = ctx.copy_expert(third_choice[0], next_base_slot)
    fourth_copies = ctx.copy_expert(fourth_choice[0], next_base_slot + 1)

    should_prefetch_third = third_choice[4] & (iteration != 0)
    should_prefetch_fourth = fourth_choice[4] & (iteration != 0)

    @pl.when(should_prefetch_third)
    def prefetch_third() -> None:
        """Start async weight copies for the first expert of the next pair."""
        _start_async_copies(third_copies)

    @pl.when(should_prefetch_fourth)
    def prefetch_fourth() -> None:
        """Start async weight copies for the second expert of the next pair."""
        _start_async_copies(fourth_copies)

    phase_end(select_scope)
    step = _ExpertPairIteration(
        first_expert=first_expert,
        first_rows=first_rows,
        first_slot_mask=first_slot_mask,
        first_consumed=first_consumed,
        has_first=has_first,
        second_expert=second_expert,
        second_rows=second_rows,
        second_slot_mask=second_slot_mask,
        second_consumed=second_consumed,
        has_second=has_second,
        current_base_slot=current_base_slot,
        first_copies=first_copies,
        second_copies=second_copies,
    )
    return step, third_choice, fourth_choice


def _execute_full_expert_pair(
    ctx: _RoutedExpertStreamContext,
    step: _ExpertPairIteration,
) -> None:
    """Project and optionally finish a full-batch pair of routed experts."""
    step.first_copies.gate_up.wait()
    step.first_copies.gate_up_scales.wait()
    step.second_copies.gate_up.wait()
    step.second_copies.gate_up_scales.wait()
    _paired_expert_gate_up(
        ctx.source,
        ctx.expert_gate_up_vmem,
        ctx.expert_gate_up_scale_vmem,
        ctx.gate_up_accumulator,
        step.current_base_slot,
        step.current_base_slot + 1,
        ctx.expert_input_size,
    )
    if not ctx.share_full_finish:
        _finish_full_expert(
            ctx,
            _ExpertSlotTarget(
                active_rows=step.first_rows,
                current_slot_mask=step.first_slot_mask,
                copies=step.first_copies,
                column=0,
                slot=step.current_base_slot,
            ),
        )
        _finish_full_expert(
            ctx,
            _ExpertSlotTarget(
                active_rows=step.second_rows,
                current_slot_mask=step.second_slot_mask,
                copies=step.second_copies,
                column=2 * ctx.expert_width,
                slot=step.current_base_slot + 1,
            ),
        )


def _execute_full_expert_tail(
    ctx: _RoutedExpertStreamContext,
    step: _ExpertPairIteration,
) -> None:
    """Project and optionally finish a single full-batch trailing expert."""
    step.first_copies.gate_up.wait()
    step.first_copies.gate_up_scales.wait()
    _expert_gate_up_dot(
        ctx,
        ctx.source,
        step.current_base_slot,
        ctx.gate_up_accumulator.at[:, : 2 * ctx.expert_width],
    )
    if not ctx.share_full_finish:
        _finish_full_expert(
            ctx,
            _ExpertSlotTarget(
                active_rows=step.first_rows,
                current_slot_mask=step.first_slot_mask,
                copies=step.first_copies,
                column=0,
                slot=step.current_base_slot,
            ),
        )


def _run_singleton_expert_pair(
    ctx: _RoutedExpertStreamContext,
    step: _ExpertPairIteration,
) -> None:
    """Stage, project, and finish two singleton-row routed experts together."""
    step.first_copies.gate_up.wait()
    step.first_copies.gate_up_scales.wait()
    step.second_copies.gate_up.wait()
    step.second_copies.gate_up_scales.wait()
    _stage_singleton_source(ctx, step.first_rows, ctx.first_single_source)
    _stage_singleton_source(ctx, step.second_rows, ctx.second_single_source)
    _paired_expert_gate_up(
        ctx.first_single_source,
        ctx.expert_gate_up_vmem,
        ctx.expert_gate_up_scale_vmem,
        ctx.gate_up_accumulator.at[:2],
        step.current_base_slot,
        step.current_base_slot + 1,
        ctx.expert_input_size,
        second_source=ctx.second_single_source,
    )
    _finish_singleton_expert(
        ctx,
        _ExpertSlotTarget(
            active_rows=step.first_rows,
            current_slot_mask=step.first_slot_mask,
            copies=step.first_copies,
            column=0,
            slot=step.current_base_slot,
        ),
        0,
    )
    _finish_singleton_expert(
        ctx,
        _ExpertSlotTarget(
            active_rows=step.second_rows,
            current_slot_mask=step.second_slot_mask,
            copies=step.second_copies,
            column=2 * ctx.expert_width,
            slot=step.current_base_slot + 1,
        ),
        1,
    )


def _run_compact_multi_pair(
    ctx: _RoutedExpertStreamContext,
    step: _ExpertPairIteration,
    row_counts: tuple[jax.Array, jax.Array],
    plans: tuple[_RowPlan | None, _RowPlan | None],
) -> None:
    """Stage and project two compact multi-row routed experts."""
    first_row_count, second_row_count = row_counts
    first_plan, second_plan = plans
    step.first_copies.gate_up.wait()
    step.first_copies.gate_up_scales.wait()
    step.second_copies.gate_up.wait()
    step.second_copies.gate_up_scales.wait()
    _stage_compact_rows(
        ctx,
        step.first_rows,
        ctx.first_compact_source_vmem,
        row_plan=first_plan,
    )
    _stage_compact_rows(
        ctx,
        step.second_rows,
        ctx.second_compact_source_vmem,
        row_plan=second_plan,
    )

    def project_pair(capacity: int | None = None) -> None:
        """Run paired gate/up for ``capacity`` active rows per expert."""
        _paired_expert_gate_up(
            ctx.first_compact_source_vmem,
            ctx.expert_gate_up_vmem,
            ctx.expert_gate_up_scale_vmem,
            ctx.gate_up_accumulator.at[: 2 * ctx.compact_capacity],
            step.current_base_slot,
            step.current_base_slot + 1,
            ctx.expert_input_size,
            second_source=ctx.second_compact_source_vmem,
            active_capacity=capacity,
        )

    if ctx.share_compact_finish:
        # Keep native eight-row staging and finishers. Narrow only the
        # operands of gate/up, preserving its scale and FP32 reduction order.
        largest = jnp.maximum(first_row_count, second_row_count)
        previous = 1
        for capacity in _COMPACT_PAIR_CAPACITIES:

            @pl.when((largest > previous) & (largest <= capacity))
            @_scoped(f"e_capacity_{capacity}")
            def execute_capacity(capacity: int = capacity) -> None:
                """Project the compact pair at bucket ``capacity``."""
                project_pair(capacity)

            previous = capacity
    else:
        project_pair()
        _finish_compact_expert(
            ctx,
            _ExpertSlotTarget(
                active_rows=step.first_rows,
                current_slot_mask=step.first_slot_mask,
                copies=step.first_copies,
                column=0,
                slot=step.current_base_slot,
            ),
            0,
        )
        _finish_compact_expert(
            ctx,
            _ExpertSlotTarget(
                active_rows=step.second_rows,
                current_slot_mask=step.second_slot_mask,
                copies=step.second_copies,
                column=2 * ctx.expert_width,
                slot=step.current_base_slot + 1,
            ),
            ctx.compact_capacity,
        )


def _dispatch_compact_expert_pairs(
    ctx: _RoutedExpertStreamContext,
    step: _ExpertPairIteration,
    first_row_count: jax.Array,
    second_row_count: jax.Array,
) -> tuple[jax.Array, jax.Array, _RowPlan | None, _RowPlan | None]:
    """Dispatch singleton, compact multi-row, and overflow expert pairs.

    From B8, staging a singleton route's one active row is cheaper than a full
    padded MXU tile. At B2/B4 the selection overhead offsets the saved padded
    work.
    """
    compact_pair = (
        step.has_first
        & step.has_second
        & (first_row_count == 1)
        & (second_row_count == 1)
    )
    first_plan = _compact_row_plan(ctx, step.first_rows) if ctx.reuse_row_plan else None
    second_plan = (
        _compact_row_plan(ctx, step.second_rows) if ctx.reuse_row_plan else None
    )

    @pl.when(compact_pair)
    @_scoped("e_single_pair")
    def execute_compact_pair() -> None:
        """Execute two singleton routed experts as one compact pair."""
        _run_singleton_expert_pair(ctx, step)

    if ctx.compact_capacity > 1:
        multi_pair = (
            step.has_first
            & step.has_second
            & ~compact_pair
            & (first_row_count <= ctx.compact_capacity)
            & (second_row_count <= ctx.compact_capacity)
        )

        @pl.when(multi_pair)
        @_scoped("e_multi_pair")
        def execute_compact_multi_pair() -> None:
            """Execute two compact multi-row routed experts."""
            _run_compact_multi_pair(
                ctx,
                step,
                (first_row_count, second_row_count),
                (first_plan, second_plan),
            )
    else:
        multi_pair = jnp.bool_(0)

    @pl.when(step.has_first & step.has_second & ~compact_pair & ~multi_pair)
    @_scoped("e_pass_pair")
    def execute_noncompact_pair() -> None:
        """Execute an overflowing expert pair in passes or full batch."""
        if ctx.compact_only:
            _run_expert_in_passes(
                ctx,
                _ExpertSlotTarget(
                    active_rows=step.first_rows,
                    current_slot_mask=step.first_slot_mask,
                    copies=step.first_copies,
                    column=0,
                    slot=step.current_base_slot,
                ),
                first_row_count,
                paired_gate=ctx.compact_overflow,
            )
            _run_expert_in_passes(
                ctx,
                _ExpertSlotTarget(
                    active_rows=step.second_rows,
                    current_slot_mask=step.second_slot_mask,
                    copies=step.second_copies,
                    column=0,
                    slot=step.current_base_slot + 1,
                ),
                second_row_count,
                paired_gate=ctx.compact_overflow,
            )
        else:
            _execute_full_expert_pair(ctx, step)

    return compact_pair, multi_pair, first_plan, second_plan


def _dispatch_compact_expert_tails(
    ctx: _RoutedExpertStreamContext,
    step: _ExpertPairIteration,
    first_row_count: jax.Array,
    first_plan: _RowPlan | None,
) -> tuple[jax.Array, jax.Array]:
    """Dispatch singleton, compact multi-row, and overflow trailing experts."""
    compact_tail = step.has_first & ~step.has_second & (first_row_count == 1)

    @pl.when(compact_tail)
    @_scoped("e_single_tail")
    def execute_compact_tail() -> None:
        """Execute a trailing singleton routed expert."""
        step.first_copies.gate_up.wait()
        step.first_copies.gate_up_scales.wait()
        _stage_singleton_source(ctx, step.first_rows, ctx.first_single_source)
        _expert_gate_up_dot(
            ctx,
            ctx.first_single_source,
            step.current_base_slot,
            ctx.gate_up_accumulator.at[:1, : 2 * ctx.expert_width],
        )
        _finish_singleton_expert(
            ctx,
            _ExpertSlotTarget(
                active_rows=step.first_rows,
                current_slot_mask=step.first_slot_mask,
                copies=step.first_copies,
                column=0,
                slot=step.current_base_slot,
            ),
            0,
        )

    if ctx.compact_capacity > 1:
        multi_tail = (
            step.has_first
            & ~step.has_second
            & (first_row_count != 1)
            & (first_row_count <= ctx.compact_capacity)
        )

        @pl.when(multi_tail)
        def execute_compact_multi_tail() -> None:
            """Execute a trailing compact multi-row routed expert."""
            step.first_copies.gate_up.wait()
            step.first_copies.gate_up_scales.wait()
            _stage_compact_rows(
                ctx,
                step.first_rows,
                ctx.first_compact_source_vmem,
                row_plan=first_plan,
            )
            _expert_gate_up_dot(
                ctx,
                ctx.first_compact_source_vmem,
                step.current_base_slot,
                ctx.gate_up_accumulator.at[
                    : ctx.compact_capacity,
                    : 2 * ctx.expert_width,
                ],
            )
            if not ctx.share_compact_finish:
                _finish_compact_expert(
                    ctx,
                    _ExpertSlotTarget(
                        active_rows=step.first_rows,
                        current_slot_mask=step.first_slot_mask,
                        copies=step.first_copies,
                        column=0,
                        slot=step.current_base_slot,
                    ),
                    0,
                )
    else:
        multi_tail = jnp.bool_(0)

    @pl.when(step.has_first & ~step.has_second & (first_row_count != 1) & ~multi_tail)
    def execute_noncompact_tail() -> None:
        """Execute an overflowing trailing expert in passes or full batch."""
        if ctx.compact_only:
            _run_expert_in_passes(
                ctx,
                _ExpertSlotTarget(
                    active_rows=step.first_rows,
                    current_slot_mask=step.first_slot_mask,
                    copies=step.first_copies,
                    column=0,
                    slot=step.current_base_slot,
                ),
                first_row_count,
            )
        else:
            _execute_full_expert_tail(ctx, step)

    return compact_tail, multi_tail


def _dispatch_shared_expert_finishers(
    ctx: _RoutedExpertStreamContext,
    step: _ExpertPairIteration,
    predicates: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    plans: tuple[_RowPlan | None, _RowPlan | None],
) -> None:
    """Run deduplicated full-batch and compact expert finishers when enabled."""
    compact_pair, multi_pair, compact_tail, multi_tail = predicates
    first_plan, second_plan = plans
    if ctx.share_full_finish:

        @pl.when(
            step.has_first & ~compact_pair & ~multi_pair & ~compact_tail & ~multi_tail
        )
        @_scoped("e_pass_finish")
        def execute_full_finish() -> None:
            """Finish one or two full-batch experts via a shared loop."""

            def route_weights(rows: jax.Array, mask: jax.Array) -> jax.Array:
                """Compute normalized route weights for ``rows``."""
                probability = (
                    jnp.sum(
                        jnp.where(
                            mask,
                            ctx.selected_probabilities,
                            jnp.float32(0),
                        ),
                        axis=1,
                    )
                    / ctx.probability_totals
                )
                return jnp.where(rows, probability, jnp.float32(0))

            first_weights = route_weights(step.first_rows, step.first_slot_mask)
            second_weights = route_weights(
                step.second_rows,
                step.second_slot_mask,
            )

            @pl.loop(0, jnp.where(step.has_second, 2, 1))
            def selected_expert(index: jax.Array) -> None:
                """Finish full-batch expert ``index`` in the active pair."""
                first = index == 0
                output_weights = jnp.where(
                    first,
                    first_weights,
                    second_weights,
                )
                expert = jnp.where(
                    first,
                    step.first_expert,
                    step.second_expert,
                )
                slot = step.current_base_slot + index
                copies = ctx.copy_expert(expert, slot)
                _finish_full_expert(
                    ctx,
                    _ExpertSlotTarget(
                        active_rows=None,
                        current_slot_mask=None,
                        copies=copies,
                        column=index * 2 * ctx.expert_width,
                        slot=slot,
                    ),
                    output_weights,
                )

    if ctx.share_compact_finish:

        @pl.when(multi_pair | multi_tail)
        @_scoped("e_multi_finish")
        def execute_compact_finish() -> None:
            """Finish one or two compact experts via a shared loop."""
            first_selector, first_weights = _compact_route_data(
                ctx,
                step.first_rows,
                step.first_slot_mask,
                row_plan=first_plan,
            )
            second_selector, second_weights = _compact_route_data(
                ctx,
                step.second_rows,
                step.second_slot_mask,
                row_plan=second_plan,
            )

            def finish_selected() -> None:
                """Loop over active compact experts in the pair."""

                @pl.loop(0, jnp.where(step.has_second, 2, 1))
                def selected_expert(index: jax.Array) -> None:
                    """Finish compact expert ``index`` in the active pair."""
                    first = index == 0
                    selector = jnp.where(
                        first,
                        first_selector,
                        second_selector,
                    )
                    output_weights = jnp.where(
                        first,
                        first_weights,
                        second_weights,
                    )
                    expert = jnp.where(
                        first,
                        step.first_expert,
                        step.second_expert,
                    )
                    slot = step.current_base_slot + index
                    copies = ctx.copy_expert(expert, slot)
                    _finish_compact_expert(
                        ctx,
                        _ExpertSlotTarget(
                            active_rows=None,
                            current_slot_mask=None,
                            copies=copies,
                            column=index * 2 * ctx.expert_width,
                            slot=slot,
                        ),
                        index * ctx.compact_capacity,
                        (selector, output_weights),
                    )

            finish_selected()


def _step_divergent_expert_pair(
    ctx: _RoutedExpertStreamContext,
    iteration: jax.Array,
    carry: tuple[_PackedRouteChoice, _PackedRouteChoice] | None,
    *,
    carry_routes: bool,
) -> tuple[_PackedRouteChoice, _PackedRouteChoice] | None:
    """Execute one pair step of divergent routed experts."""
    if ctx.progress is not None:
        ctx.progress(iteration)
    with _phase("e_iteration"):
        step, third_choice, fourth_choice = _select_and_prefetch_expert_pair(
            ctx,
            iteration,
            carry,
            carry_routes=carry_routes,
        )
        # From B8, staging a singleton route's one active row is cheaper than a
        # full padded matrix tile. At B2/B4 selection overhead offsets the gain.
        if ctx.batch_size >= _COMPACT_ROUTE_MIN_BATCH:
            first_row_count = jnp.sum(step.first_rows.astype(jnp.int32))
            second_row_count = jnp.sum(step.second_rows.astype(jnp.int32))
            compact_pair, multi_pair, first_plan, second_plan = (
                _dispatch_compact_expert_pairs(
                    ctx,
                    step,
                    first_row_count,
                    second_row_count,
                )
            )
            compact_tail, multi_tail = _dispatch_compact_expert_tails(
                ctx,
                step,
                first_row_count,
                first_plan,
            )
            _dispatch_shared_expert_finishers(
                ctx,
                step,
                (compact_pair, multi_pair, compact_tail, multi_tail),
                (first_plan, second_plan),
            )
        else:

            @pl.when(step.has_first & step.has_second)
            def execute_pair() -> None:
                """Execute a full-batch expert pair below B8."""
                _execute_full_expert_pair(ctx, step)

            @pl.when(step.has_first & ~step.has_second)
            def execute_tail() -> None:
                """Execute a full-batch trailing expert below B8."""
                _execute_full_expert_tail(ctx, step)

        ctx.processed_routes_vmem[:, : ctx.selected_expert_count] = jnp.where(
            step.first_consumed | step.second_consumed,
            jnp.int32(1),
            ctx.processed_routes_vmem[:, : ctx.selected_expert_count],
        )
        if carry_routes:
            return (
                _pack_route_choice(third_choice),
                _pack_route_choice(fourth_choice),
            )
        return None


def _execute_divergent_routes(
    ctx: _RoutedExpertStreamContext,
    pair_count: jax.Array,
) -> None:
    """Advance divergent expert routes across batch rows in pairs."""
    # Reuse the choices already made for the next four-slot prefetch.
    # Other batch/routing combinations use the selector loop.
    carry_routes = (
        ctx.batch_size in _SMALL_UNGROUPED_CARRY_BATCHES
        and not ctx.group_pending_routes
    ) or (ctx.batch_size in _GROUPED_CARRY_BATCHES and ctx.group_pending_routes)
    initial_first: _RouteChoice | None = None
    initial_second: _RouteChoice | None = None
    if carry_routes:
        initial_processed = jnp.zeros(ctx.selected_experts.shape, jnp.bool_)
        initial_first = _select_next_routed_expert(ctx, initial_processed)
        initial_second = _select_next_routed_expert(
            ctx,
            initial_processed | initial_first[3],
        )

    if carry_routes:
        if initial_first is None or initial_second is None:
            msg = "Initial route choices must be set when carry_routes is on"
            raise ValueError(msg)

        def run_next_expert_pair(
            iteration: jax.Array,
            carry: tuple[_PackedRouteChoice, _PackedRouteChoice],
        ) -> tuple[_PackedRouteChoice, _PackedRouteChoice]:
            """Execute one carried pair of divergent routed experts."""
            next_carry = _step_divergent_expert_pair(
                ctx,
                iteration,
                carry,
                carry_routes=True,
            )
            if next_carry is None:
                msg = "Carried expert pair step must return next carry tuple"
                raise ValueError(msg)
            return next_carry

        jax.lax.fori_loop(
            0,
            pair_count,
            run_next_expert_pair,
            (
                _pack_route_choice(initial_first),
                _pack_route_choice(initial_second),
            ),
        )
    else:

        @pl.loop(0, pair_count)
        def run_original_expert_pair(iteration: jax.Array) -> None:
            """Execute one uncarried pair of divergent routed experts."""
            _step_divergent_expert_pair(
                ctx,
                iteration,
                None,
                carry_routes=False,
            )


def _run_routed_expert_stream(
    source: TensorLike,
    selected_experts: jax.Array,
    selected_probabilities: jax.Array,
    local_route_counts: jax.Array,
    probability_totals: jax.Array,
    expert_gate_up_vmem: PallasRefProtocol,
    expert_gate_up_scale_vmem: PallasRefProtocol,
    expert_down_vmem: PallasRefProtocol,
    expert_down_scale_vmem: PallasRefProtocol,
    expert_output: PallasRefProtocol,
    gate_up_accumulator: PallasRefProtocol,
    down_accumulator: PallasRefProtocol,
    processed_routes_vmem: PallasRefProtocol,
    first_compact_source_vmem: PallasRefProtocol,
    second_compact_source_vmem: PallasRefProtocol,
    copy_expert: _ExpertCopyCallable,
    *,
    expert_input_size: int,
    expert_width: int,
    expert_output_size: int,
    group_pending_routes: bool,
    progress: _ProgressHook | None = None,
) -> None:
    """Execute the next route shared by any ready batch rows.

    Rows retain their own routing order. When several rows have the same next
    expert, its weights are loaded once and its projections carry the batch
    dimension. The loop is bounded by the distinct experts (grouped routes) or
    the route count, never by batch size.
    """
    batch_size, selected_expert_count = selected_experts.shape
    # Narrow masked products avoid zero MACs and register spills while keeping
    # the unsplit scale reduction tree. KernelConfig.mxfp4_dot_tile overrides
    # the per-batch default (0: unsplit dots).
    dot_tile = _resolve_expert_dot_tile(batch_size)
    # Aligned block staging avoids a separate source-slice branch for every
    # batch row. On by default at B16-B128; KernelConfig.singleton_block
    # and stage_blocks override it.
    singleton_block = (
        SINGLETON_BLOCK
        if SINGLETON_BLOCK is not None
        else batch_size in _BLOCK_STAGED_BATCHES
    )
    stage_blocks = (
        STAGE_BLOCKS
        if STAGE_BLOCKS is not None
        else batch_size in _BLOCK_STAGED_BATCHES
    )
    resolved_copy_expert = (
        _wrap_scoped_expert_copy(copy_expert) if PHASE_SCOPES else copy_expert
    )
    route_slots = jnp.arange(selected_expert_count)[None, :]
    processed_routes_vmem[...] = jnp.zeros(
        processed_routes_vmem.shape,
        jnp.int32,
    )
    expert_output[...] = jnp.zeros(expert_output.shape, jnp.float32)

    compact_capacity = first_compact_source_vmem.shape[0]
    first_single_source = (
        first_compact_source_vmem.at[:1]
        if compact_capacity > 1
        else first_compact_source_vmem
    )
    second_single_source = (
        second_compact_source_vmem.at[:1]
        if compact_capacity > 1
        else second_compact_source_vmem
    )
    reuse_row_plan = stage_blocks and batch_size in _PLANNED_STAGE_BATCHES
    compact_overflow = _use_compact_overflow(
        batch_size,
        group_pending_routes=group_pending_routes,
    )
    # At B16, two compact gate tiles fill 16 rows but the down tile has 8.
    # Only compact overflow tests the down tile.
    compact_only = gate_up_accumulator.shape[0] < batch_size or (
        compact_overflow and down_accumulator.shape[0] < batch_size
    )
    share_compact_finish = (
        SHARED_EXPERT_FINISHERS
        and batch_size in _BLOCK_STAGED_BATCHES
        and compact_capacity == _SHARED_FINISHER_COMPACT_CAPACITY
        and (not compact_only or compact_overflow)
    )
    share_full_finish = share_compact_finish and not compact_only

    ctx = _RoutedExpertStreamContext(
        source=source,
        selected_experts=selected_experts,
        selected_probabilities=selected_probabilities,
        local_route_counts=local_route_counts,
        probability_totals=probability_totals,
        route_slots=route_slots,
        expert_gate_up_vmem=expert_gate_up_vmem,
        expert_gate_up_scale_vmem=expert_gate_up_scale_vmem,
        expert_down_vmem=expert_down_vmem,
        expert_down_scale_vmem=expert_down_scale_vmem,
        expert_output=expert_output,
        gate_up_accumulator=gate_up_accumulator,
        down_accumulator=down_accumulator,
        processed_routes_vmem=processed_routes_vmem,
        first_compact_source_vmem=first_compact_source_vmem,
        second_compact_source_vmem=second_compact_source_vmem,
        first_single_source=first_single_source,
        second_single_source=second_single_source,
        copy_expert=resolved_copy_expert,
        batch_size=batch_size,
        selected_expert_count=selected_expert_count,
        compact_capacity=compact_capacity,
        expert_input_size=expert_input_size,
        expert_width=expert_width,
        expert_output_size=expert_output_size,
        dot_tile=dot_tile,
        singleton_block=singleton_block,
        stage_blocks=stage_blocks,
        reuse_row_plan=reuse_row_plan,
        compact_overflow=compact_overflow,
        compact_only=compact_only,
        share_compact_finish=share_compact_finish,
        share_full_finish=share_full_finish,
        group_pending_routes=group_pending_routes,
        progress=progress,
    )

    valid_routes = route_slots < local_route_counts[:, None]
    routes_are_shared = (
        (not group_pending_routes)
        & jnp.all(local_route_counts == local_route_counts[0])
        & jnp.all(~valid_routes | (selected_experts == selected_experts[0:1]))
    )

    @pl.when(routes_are_shared & (not compact_only))
    def run_shared_routes() -> None:
        """Execute shared expert routes across all batch rows."""
        _execute_shared_routes(ctx)

    pair_count = _count_divergent_expert_pairs(ctx, valid_routes)

    @pl.when(~routes_are_shared)
    def run_divergent_routes() -> None:
        """Execute divergent expert routes across batch rows in pairs."""
        _execute_divergent_routes(ctx, pair_count)


def _staging_slot_rows(payload_rows: int) -> int:
    """Round a per-pair staged payload row count up to 16-row BF16 tiles."""
    return -(-payload_rows // _BF16_TILE_ROWS) * _BF16_TILE_ROWS


def projection_staging_scratch(
    batch_size: int,
    attention_dp: int = 1,
) -> tuple[pl.MemoryRef, ...]:
    """Allocate send/receive staging scratch for attention projections above B32."""
    if batch_size <= _INLINE_TRANSPORT_MAX_BATCH:
        return ()
    payload_rows = (
        (batch_size // attention_dp // _LOCAL_RANK_PAIRS)
        * _HALF_HIDDEN_REDUCTION_WIDTH
        // VREG_LANE_COUNT
    )
    return (
        tpu.VMEM(
            (
                2,
                _LOCAL_RANK_PAIRS * _staging_slot_rows(payload_rows),
                VREG_LANE_COUNT,
            ),
            jnp.bfloat16,
        ),
    )


def routed_send_scratch(batch_size: int) -> tuple[pl.MemoryRef, ...]:
    """Allocate send-side scratch for routed expert reduce-scatter above B32."""
    if batch_size <= _INLINE_TRANSPORT_MAX_BATCH:
        return ()
    return (
        tpu.VMEM(
            (
                mla_dcp.HOST_LANES
                * _ROUTED_TRANSPORT_ROWS_PER_TOKEN
                * -(-batch_size // mla_dcp.HOST_LANES),
                VREG_LANE_COUNT,
            ),
            jnp.float32,
        ),
    )


def communication_scratch(
    batch_size: int = 1,
    attention_dp: int = 1,
) -> tuple[pl.MemoryRef, ...]:
    """Allocate shared collective transport scratch for the fused decoder stack."""
    # FP32 transport rows per token: 28 for shared outputs and attention
    # projections, 14 for routed outputs, in separate buffers. Above B8 only
    # layer 0's dense all-reduce carries the whole batch; it runs in 8-row
    # halves, so B16 keeps the B8 capacity.
    dense_batch_cap = (
        min(batch_size, _DENSE_HALF_BATCH_ROWS)
        if batch_size <= _SINGLE_PASS_REDUCTION_MAX_BATCH
        else _SINGLE_PASS_REDUCTION_MAX_BATCH
    )
    wide_reduction_rows = max(
        _MIN_WIDE_REDUCTION_ROWS,
        _SHARED_TRANSPORT_ROWS_PER_TOKEN * dense_batch_cap,
    )
    # Above B32 the host exchange grows with the attention projection's
    # per-pair payloads (staged in projection_staging_scratch).
    host_reduction_rows = max(
        wide_reduction_rows,
        _DP_GATHER_ROWS_PER_TOKEN * batch_size // attention_dp
        if batch_size > _INLINE_TRANSPORT_MAX_BATCH
        else 0,
    )
    # The shared-expert reduce-scatter receives 4 rows per token in each lane
    # and host stage.
    wide_reduction_rows = max(
        wide_reduction_rows,
        _SHARED_SHARD_ROWS_PER_TOKEN * batch_size,
    )
    host_reduction_rows = max(
        host_reduction_rows,
        _SHARED_SHARD_ROWS_PER_TOKEN * batch_size,
    )
    if attention_dp > 1:
        # The DP MoE-input gather's 4 * DP row wires use lanes 4..7.
        wide_reduction_rows = max(
            wide_reduction_rows,
            _DP_GATHER_ROWS_PER_TOKEN * batch_size,
        )
    # The eight-lane routed reduce-scatter pads the batch to whole tokens per lane.
    padded_tokens = (
        mla_dcp.HOST_LANES * -(-batch_size // mla_dcp.HOST_LANES)
        if batch_size in (_SPECULATIVE_B6_BATCH, *SCATTER_BATCHES)
        else batch_size
    )
    expert_reduction_rows = max(
        _MIN_EXPERT_REDUCTION_ROWS,
        _ROUTED_TRANSPORT_ROWS_PER_TOKEN * padded_tokens,
    )
    expert_host_rows = expert_reduction_rows
    if batch_size > _INLINE_TRANSPORT_MAX_BATCH:
        # One lane's tokens per source lane (send side: routed_send_scratch);
        # the host buffer also carries the shared-expert shard sends.
        expert_reduction_rows = _ROUTED_TRANSPORT_ROWS_PER_TOKEN * -(
            -batch_size // mla_dcp.HOST_LANES
        )
        expert_host_rows = max(
            expert_reduction_rows,
            _HOST_SHARED_SHARD_ROWS_PER_TOKEN * batch_size,
        )
    # Whole 8-row FP32 tiles: padded strides make the reductions' ref reshapes
    # fail to compile ("reshape on a non-contiguous memref").
    (
        wide_reduction_rows,
        expert_reduction_rows,
        host_reduction_rows,
        expert_host_rows,
    ) = (
        -(-rows // SUBLANE_ROWS) * SUBLANE_ROWS
        for rows in (
            wide_reduction_rows,
            expert_reduction_rows,
            host_reduction_rows,
            expert_host_rows,
        )
    )
    output_rows = (
        batch_size // _LOCAL_RANK_PAIRS
        if batch_size >= _PAIR_OWNED_OUTPUT_MIN_BATCH
        else batch_size
    )
    return (
        tpu.VMEM(
            (mla_dcp.HOST_LANES, wide_reduction_rows, VREG_LANE_COUNT),
            jnp.float32,
        ),
        tpu.VMEM(
            (mla_dcp.HOST_LANES, expert_reduction_rows, VREG_LANE_COUNT),
            jnp.float32,
        ),
        tpu.VMEM(
            (_LOCAL_RANK_PAIRS, host_reduction_rows, VREG_LANE_COUNT),
            jnp.float32,
        ),
        tpu.VMEM(
            (_LOCAL_RANK_PAIRS, expert_host_rows, VREG_LANE_COUNT),
            jnp.float32,
        ),
        # Attention's group-gather features of every held shard, in native
        # 16-row backing (active rows via _bf16_rows_view).
        tpu.VMEM(
            (
                max(
                    _BF16_TILE_ROWS,
                    -(-batch_size // _BF16_TILE_ROWS) * _BF16_TILE_ROWS,
                ),
                _ATTENTION_FEATURE_SHARD_WIDTH * attention_dp,
            ),
            jnp.bfloat16,
        ),
        # Above B32 (always token-scattered) attention writes only each
        # pair's batch / 4 rows here.
        tpu.VMEM(
            (
                max(
                    _BF16_TILE_ROWS,
                    -(-output_rows // _BF16_TILE_ROWS) * _BF16_TILE_ROWS,
                ),
                HIDDEN_SIZE,
            ),
            jnp.bfloat16,
        ),
        tpu.SemaphoreType.DMA((_TRANSPORT_SEMAPHORE_COUNT,)),
        tpu.SemaphoreType.DMA((_TRANSPORT_SEMAPHORE_COUNT,)),
        tpu.SMEM((1,), jnp.int32),
        tpu.VMEM((batch_size, _LATENT_GATHER_WIDTH), jnp.bfloat16),
    )


_SINGLE_OWNER_SCATTER_BATCHES: tuple[int, ...] = (2, 4)
_ROUTED_SCATTER_BATCHES: tuple[int, ...] = (6, *SCATTER_BATCHES)
_SUPPORTED_EXPERT_TP_DEGREES: tuple[int, ...] = (1, 2, 4)
_MOE_GATHER_PACKET_COLS: int = 8192
_MAX_MOE_LATENT_SIZE: int = 8192
_HIDDEN_SHARD_COLS: int = 1024
_ACTIVE_HIDDEN_SHARD_LANES: int = 7
_MIN_LATENT_OVERLAP_EXPERTS: int = 12
_LATENT_GATHER_OVERLAP_STEP: int = 3
_SHARED_ROUTING_OVERLAP_STEP: int = 7
_OUTPUT_GATHER_PEER_OFFSETS: tuple[int, ...] = (7, 6, 5, 3, 4, 2, 1)
_LATENT_GATHER_PEER_OFFSETS: tuple[int, ...] = (
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
)


@dataclasses.dataclass(frozen=True, slots=True)
class AttentionCommunicationWorkspace:
    """Collective transport storage shared by attention and MoE layers."""

    shared_local_reduction_vmem: PallasRefProtocol  # [8, rows, 128] FP32
    expert_local_reduction_vmem: PallasRefProtocol  # [8, rows, 128] FP32
    shared_host_reduction_vmem: PallasRefProtocol  # [4, rows, 128] FP32
    expert_host_reduction_vmem: PallasRefProtocol  # [4, rows, 128] FP32
    attention_features_vmem: PallasRefProtocol  # [16 * n, 1024 * attention_dp] BF16
    attention_output_vmem: PallasRefProtocol  # [16 * n, 7168] BF16
    send_semaphores: PallasRefProtocol  # [16]
    receive_semaphores: PallasRefProtocol  # [16]
    reduction_phase: PallasRefProtocol  # [1]
    latent_gather_vmem: PallasRefProtocol  # [batch_size, 4096]
    projection_staging_vmem: PallasRefProtocol | None = None
    routed_send_vmem: PallasRefProtocol | None = None
    attention_gather_vmem: PallasRefProtocol | None = None
    routed_consumed: PallasRefProtocol | None = None
    pair_semaphores: tuple[PallasRefProtocol, PallasRefProtocol] | None = None

    def __post_init__(self) -> None:
        """Register megakernel PyTree dataclasses on first instantiation."""
        ensure_dataclasses_registered()


class _ScopedWait:
    """Async copy wrapper whose wait is traced under the ``e_wait`` phase."""

    def __init__(self, copy: AsyncCopyProtocol) -> None:
        """Store the underlying async copy descriptor."""
        self.copy = copy

    def start(self) -> None:
        """Start the underlying async transfer."""
        self.copy.start()

    def wait(self) -> None:
        """Wait for transfer completion inside the ``e_wait`` phase."""
        with _phase("e_wait"):
            self.copy.wait()


def _start_async_copies(copies: SimpleNamespace) -> None:
    """Start every non-null async copy stored on ``copies``."""
    for copy in vars(copies).values():
        if copy is not None:
            copy.start()


def _gather_output_shards(
    output_vmem: PallasRefProtocol,
    send_semaphores: PallasRefProtocol,
    receive_semaphores: PallasRefProtocol,
    semaphore_slot: int | jax.Array,
) -> None:
    """All-gather the eight local 1024-column output shards across host peers."""
    rank = jax.lax.axis_index(AXIS_TP)
    local_shard = output_vmem.at[
        :,
        pl.ds((rank % mla_dcp.HOST_LANES) * _HIDDEN_SHARD_COLS, _HIDDEN_SHARD_COLS),
    ]
    for offset in _OUTPUT_GATHER_PEER_OFFSETS:
        tpu.make_async_remote_copy(
            local_shard,
            local_shard,
            send_semaphores.at[semaphore_slot],
            receive_semaphores.at[semaphore_slot],
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    received_shards = output_vmem.at[
        :, : _ACTIVE_HIDDEN_SHARD_LANES * _HIDDEN_SHARD_COLS
    ]
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


@dataclasses.dataclass(frozen=True, slots=True)
class MixtureOfExpertsWeights:
    """MoE parameters and persistent resources for asynchronous weight copies."""

    routing_projection: PallasRefProtocol  # [hidden_size, 896]
    routing_projection_vmem: PallasRefProtocol  # [1, hidden_size, 512]
    routing_copy_semaphores: PallasRefProtocol  # [2]
    routing_bias: TensorLike  # [1, 896]
    latent_input_projection: TensorLike  # [hidden_size, 128]
    latent_output_projection: TensorLike  # [expert_output_size, 1024]
    latent_output_projection_vmem: PallasRefProtocol  # [expert_output_size, 1024]
    latent_output_copy_semaphore: PallasRefProtocol  # scalar DMA semaphore
    latent_normalization: TensorLike  # [1, padded_latent_size]
    shared_gate_up_projection: TensorLike  # [hidden_size, 2 * shared_width]
    shared_down_projection: TensorLike  # [shared_width, hidden_size]
    # Packed MXFP4: [local_experts, expert_input_size / 8, 2 * expert_width].
    expert_gate_up_projection: PallasRefProtocol
    expert_gate_up_projection_vmem: PallasRefProtocol  # [4, ...]
    expert_gate_up_scales: PallasRefProtocol
    expert_gate_up_scales_vmem: PallasRefProtocol  # [4, ...]
    # Packed MXFP4: [local_experts, expert_width / 8, expert_output_size].
    expert_down_projection: PallasRefProtocol
    expert_down_projection_vmem: PallasRefProtocol  # [4, ...]
    expert_down_scales: PallasRefProtocol
    expert_down_scales_vmem: PallasRefProtocol  # [4, ...]
    expert_copy_semaphores: PallasRefProtocol  # [4, 4]

    def __post_init__(self) -> None:
        """Register megakernel PyTree dataclasses on first instantiation."""
        ensure_dataclasses_registered()

    def _copy_routing_projection(
        self,
        *,
        wait: bool,
        initialize_padding: bool = True,
    ) -> None:
        """Start or wait on this rank's half of the router projection copy."""
        lane = jax.lax.axis_index(AXIS_TP) % 2

        @pl.when(lane == 0)
        def copy_full_half() -> None:
            """Transfer the leading 512 router columns on even lanes."""
            copy = tpu.make_async_copy(
                self.routing_projection.at[:, :_ROUTER_LANE_SCORE_WIDTH],
                self.routing_projection_vmem.at[0],
                self.routing_copy_semaphores.at[0],
            )
            if wait:
                copy.wait()
            else:
                copy.start()

        @pl.when(lane == 1)
        def copy_tail() -> None:
            """Transfer trailing 384 router columns and zero padding on odd lanes."""
            if not wait and initialize_padding:
                self.routing_projection_vmem[0, :, _ODD_LANE_ROUTER_COLUMNS:] = (
                    jnp.zeros(
                        (self.routing_projection.shape[0], VREG_LANE_COUNT),
                        jnp.bfloat16,
                    )
                )
            copy = tpu.make_async_copy(
                self.routing_projection.at[
                    :,
                    pl.ds(_ROUTER_LANE_SCORE_WIDTH, _ODD_LANE_ROUTER_COLUMNS),
                ],
                self.routing_projection_vmem.at[0, :, :_ODD_LANE_ROUTER_COLUMNS],
                self.routing_copy_semaphores.at[0],
            )
            if wait:
                copy.wait()
            else:
                copy.start()

    def start_routing_projection_copy(self, *, initialize_padding: bool = True) -> None:
        """Start asynchronous copy of the router projection into local memory."""
        self._copy_routing_projection(
            wait=False,
            initialize_padding=initialize_padding,
        )

    def wait_for_routing_projection_copy(self) -> None:
        """Wait for the asynchronous router projection copy to complete."""
        self._copy_routing_projection(wait=True)

    def make_latent_output_copy(self) -> AsyncCopyProtocol:
        """Create the async copy descriptor for the latent output projection."""
        return tpu.make_async_copy(
            self.latent_output_projection,
            self.latent_output_projection_vmem,
            self.latent_output_copy_semaphore,
        )

    def make_expert_copies(
        self,
        expert: int | jax.Array,
        slot: int | jax.Array,
        *,
        gate_up: bool = True,
        down: bool = True,
    ) -> SimpleNamespace:
        """Create async copy descriptors for one routed expert's projections."""
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


@dataclasses.dataclass(frozen=True, slots=True)
class MixtureOfExpertsWorkspace:
    """Persistent local memory and semaphores used by one MoE invocation."""

    selected_experts_vmem: PallasRefProtocol  # [batch_size, 128]
    selected_probabilities_vmem: PallasRefProtocol  # [batch_size, 128]
    local_route_counts_vmem: PallasRefProtocol  # [batch_size, 128]
    probability_totals_vmem: PallasRefProtocol  # [batch_size, 128]
    routing_scores_vmem: PallasRefProtocol  # [batch_size, 1024]
    shared_outputs_vmem: PallasRefProtocol  # [batch_size, hidden_size]
    routed_outputs_vmem: PallasRefProtocol  # [batch_size, padded_latent_size]
    gate_up_accumulator_vmem: PallasRefProtocol  # [rows, 4 * expert_width]
    down_accumulator_vmem: PallasRefProtocol  # [rows, expert_output_size]
    processed_routes_vmem: PallasRefProtocol  # [batch_size, 128]
    first_compact_source_vmem: PallasRefProtocol  # [max(1, _compact_rows), K]
    second_compact_source_vmem: PallasRefProtocol  # [max(1, _compact_rows), K]
    shared_send_semaphores: PallasRefProtocol  # [10]
    shared_receive_semaphores: PallasRefProtocol  # [10]
    output_vmem: PallasRefProtocol  # [batch_size, 8192] or scatter shape
    output_send_semaphores: PallasRefProtocol  # [2] or [7]
    output_receive_semaphores: PallasRefProtocol  # [2] or [7]
    latent_send_semaphores: PallasRefProtocol  # [2]
    latent_receive_semaphores: PallasRefProtocol  # [2]
    # Sharded routing only: this pair's selection rows and everyone's.
    selection_source_vmem: PallasRefProtocol | None = None  # [batch_size / 4, 128]
    selections_vmem: PallasRefProtocol | None = None  # [batch_size, 128]

    def __post_init__(self) -> None:
        """Register megakernel PyTree dataclasses on first instantiation."""
        ensure_dataclasses_registered()


@dataclasses.dataclass(frozen=True, slots=True)
class _MoeDimensions:
    """Derived tensor and parallelism dimensions for one MoE layer."""

    batch_size: int
    hidden_size: int
    expert_count: int
    latent_size: int
    expert_output_size: int
    local_expert_count: int
    expert_tensor_parallel: int
    expert_width: int
    expert_input_size: int
    shared_width: int


@dataclasses.dataclass(frozen=True, slots=True)
class _MoeLatentGatherContext:
    """Rank and local-memory handles for the MoE latent input all-gather."""

    rank: jax.Array
    latent_slot: jax.Array
    latent_vmem: PallasRefProtocol
    latent_send_semaphores: PallasRefProtocol
    latent_receive_semaphores: PallasRefProtocol


@dataclasses.dataclass(frozen=True, slots=True)
class _MoePrepareResources:
    """Scoped local-memory buffers and DMA semaphores for MoE batch preparation."""

    latent_input_projection_vmem: PallasRefProtocol
    latent_input_copy_semaphore: PallasRefProtocol
    shared_gate_up_projection_vmem: PallasRefProtocol
    shared_down_projection_vmem: PallasRefProtocol
    dense_copy_semaphores: PallasRefProtocol


@dataclasses.dataclass(frozen=True, slots=True)
class _MoePrepareContext:
    """Static configuration and workspace handles for MoE batch preparation."""

    dims: _MoeDimensions
    hidden: jax.Array
    owned_hidden_ref: PallasRefProtocol | None
    live_rows: jax.Array | None
    weights: MixtureOfExpertsWeights
    workspace: MixtureOfExpertsWorkspace
    communication_workspace: AttentionCommunicationWorkspace
    latent_ctx: _MoeLatentGatherContext
    shared_gate_up_stream: ChunkedWeightStream | None
    after_routing_scores: Callable[[], None] | None
    selected_expert_count: int
    group_pending_expert_routes: bool
    overlap_latent_gather: bool
    overlap_router_latent: bool
    overlap_shared_routing: bool
    shared_sharded: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _MoeReductionContext:
    """Transport geometry and collective buffers for MoE output reductions."""

    dims: _MoeDimensions
    workspace: MixtureOfExpertsWorkspace
    communication_workspace: AttentionCommunicationWorkspace
    rank: jax.Array
    shared_down_weights: PallasRefProtocol | None
    shared_rows: int
    shared_payload: int
    expert_rows: int
    expert_payload: int
    tokens_per_lane: int
    lane_payload: int
    shared_shard_rows: int
    shared_shard_payload: int
    token_scatter_owners: int
    shared_sharded: bool
    reduce_scatter_shared_output: bool
    reduce_scatter_routed_output: bool
    overlap_shared_reduction: bool
    token_scatter_output: bool


def _derive_moe_dimensions(
    hidden_ref: PallasRefProtocol,
    weights: MixtureOfExpertsWeights,
) -> _MoeDimensions:
    """Extract batch, hidden, latent, and routed-expert dimensions."""
    batch_size, hidden_size = hidden_ref.shape
    expert_count = weights.routing_projection.shape[1]
    latent_size = weights.latent_normalization.shape[-1]
    expert_output_size = weights.expert_down_projection.shape[-1]
    local_expert_count = weights.expert_gate_up_projection.shape[0]
    expert_tensor_parallel = TP_SIZE * local_expert_count // expert_count
    expert_width = weights.expert_down_projection.shape[1] * 8
    expert_input_size = weights.expert_gate_up_projection.shape[1] * 8
    shared_width = weights.shared_down_projection.shape[0]
    return _MoeDimensions(
        batch_size=batch_size,
        hidden_size=hidden_size,
        expert_count=expert_count,
        latent_size=latent_size,
        expert_output_size=expert_output_size,
        local_expert_count=local_expert_count,
        expert_tensor_parallel=expert_tensor_parallel,
        expert_width=expert_width,
        expert_input_size=expert_input_size,
        shared_width=shared_width,
    )


def _validate_moe_modes_and_dims(
    dims: _MoeDimensions,
    output_shape: tuple[int, ...],
    *,
    token_scatter_output: bool,
    routed_reduce_scatter: bool,
    token_scatter_owners: int,
) -> None:
    """Validate MoE communication modes, tile alignment, and output shape."""
    if not 1 <= dims.batch_size <= MAX_BATCH:
        msg = f"MoE supports 1 to {MAX_BATCH} rows, got batch_size={dims.batch_size}"
        raise ValueError(msg)
    if (
        token_scatter_output
        and dims.batch_size not in SCATTER_BATCHES
        and not (
            dims.batch_size in _SINGLE_OWNER_SCATTER_BATCHES
            and token_scatter_owners == 1
        )
    ):
        msg = (
            "MoE token-scattered output requires batch size 8 or 16, "
            "or B2/B4 to one owner, "
            f"got batch_size={dims.batch_size}, owners={token_scatter_owners}"
        )
        raise ValueError(msg)
    if routed_reduce_scatter and dims.batch_size not in _ROUTED_SCATTER_BATCHES:
        msg = (
            "MoE routed reduce-scatter requires batch size 6, 8 or 16, "
            f"got batch_size={dims.batch_size}"
        )
        raise ValueError(msg)
    if dims.expert_tensor_parallel not in _SUPPORTED_EXPERT_TP_DEGREES:
        msg = (
            "Expert tensor parallelism must be 1, 2, or 4, "
            f"got expert_tensor_parallel={dims.expert_tensor_parallel}"
        )
        raise ValueError(msg)
    if (
        dims.hidden_size % VREG_LANE_COUNT
        or dims.expert_count % VREG_LANE_COUNT
        or dims.latent_size % 256
        or dims.expert_width % 256
        or dims.hidden_size > _MOE_GATHER_PACKET_COLS
        or dims.latent_size > _MAX_MOE_LATENT_SIZE
    ):
        msg = (
            "Unsupported MoE tile dimensions: "
            f"hidden_size={dims.hidden_size}, expert_count={dims.expert_count}, "
            f"latent_size={dims.latent_size}, expert_width={dims.expert_width}"
        )
        raise ValueError(msg)
    if (
        dims.expert_count * dims.expert_tensor_parallel
        != TP_SIZE * dims.local_expert_count
        or dims.expert_output_size > dims.latent_size
        or dims.expert_input_size > dims.latent_size
    ):
        msg = (
            "Expert ownership or latent padding is inconsistent: "
            f"expert_count={dims.expert_count}, tp={dims.expert_tensor_parallel}, "
            f"local_experts={dims.local_expert_count}, "
            f"input={dims.expert_input_size}, output={dims.expert_output_size}, "
            f"latent={dims.latent_size}"
        )
        raise ValueError(msg)
    if dims.expert_input_size % 512 or dims.expert_output_size % VREG_LANE_COUNT:
        msg = (
            "Expert dimensions must satisfy MXFP4 alignment, "
            f"got expert_input_size={dims.expert_input_size}, "
            f"expert_output_size={dims.expert_output_size}"
        )
        raise ValueError(msg)
    expected_output_shape = (
        (dims.batch_size // token_scatter_owners, dims.hidden_size)
        if token_scatter_output
        else (dims.batch_size, dims.hidden_size)
    )
    if output_shape != expected_output_shape:
        msg = (
            "MoE output has the wrong shape for its communication mode: "
            f"got {output_shape}, expected {expected_output_shape}"
        )
        raise ValueError(msg)


def _validate_moe_weight_tensors(
    dims: _MoeDimensions,
    weights: MixtureOfExpertsWeights,
) -> None:
    """Validate MoE weight dtypes, packed MXFP4 shapes, and prefetch buffers."""
    if (
        weights.expert_gate_up_projection.dtype != jnp.uint32
        or weights.expert_down_projection.dtype != jnp.uint32
    ):
        msg = (
            "Routed-expert weights must be packed MXFP4 (uint32), "
            f"got gate_up={weights.expert_gate_up_projection.dtype}, "
            f"down={weights.expert_down_projection.dtype}"
        )
        raise ValueError(msg)
    if weights.latent_input_projection.shape != (
        dims.hidden_size,
        VREG_LANE_COUNT,
    ) or weights.latent_output_projection.shape != (
        dims.expert_output_size,
        _HIDDEN_SHARD_COLS,
    ):
        msg = (
            "Latent projections require padded TP32 shards, "
            f"got latent_in={weights.latent_input_projection.shape}, "
            f"latent_out={weights.latent_output_projection.shape}"
        )
        raise ValueError(msg)
    if (
        weights.expert_gate_up_projection.shape[1:]
        != (dims.expert_input_size // 8, 2 * dims.expert_width)
        or weights.expert_gate_up_scales.shape[1:]
        != (dims.expert_input_size // 32, 2 * dims.expert_width)
        or weights.expert_gate_up_scales.dtype != jnp.uint8
    ):
        msg = (
            "MXFP4 gate/up weights have inconsistent shapes: "
            f"weights={weights.expert_gate_up_projection.shape}, "
            f"scales={weights.expert_gate_up_scales.shape} "
            f"({weights.expert_gate_up_scales.dtype})"
        )
        raise ValueError(msg)
    if (
        weights.expert_down_projection.shape[1] * 8 != dims.expert_width
        or weights.expert_down_scales.dtype != jnp.uint8
    ):
        msg = (
            "MXFP4 down weights or scales have inconsistent shapes: "
            f"weights={weights.expert_down_projection.shape}, "
            f"scales_dtype={weights.expert_down_scales.dtype}, "
            f"expert_width={dims.expert_width}"
        )
        raise ValueError(msg)
    if weights.routing_projection_vmem.shape != (
        1,
        dims.hidden_size,
        _ROUTER_LANE_SCORE_WIDTH,
    ):
        msg = (
            "MoE routing projection workspace has the wrong shape: "
            f"got {weights.routing_projection_vmem.shape}, "
            f"expected (1, {dims.hidden_size}, {_ROUTER_LANE_SCORE_WIDTH})"
        )
        raise ValueError(msg)
    expert_workspace_shapes = (
        (
            weights.expert_gate_up_projection_vmem.shape,
            (
                _EXPERT_PREFETCH_SLOTS,
                *weights.expert_gate_up_projection.shape[1:],
            ),
        ),
        (
            weights.expert_gate_up_scales_vmem.shape,
            (_EXPERT_PREFETCH_SLOTS, *weights.expert_gate_up_scales.shape[1:]),
        ),
        (
            weights.expert_down_projection_vmem.shape,
            (_EXPERT_PREFETCH_SLOTS, *weights.expert_down_projection.shape[1:]),
        ),
        (
            weights.expert_down_scales_vmem.shape,
            (_EXPERT_PREFETCH_SLOTS, *weights.expert_down_scales.shape[1:]),
        ),
    )
    if any(actual != expected for actual, expected in expert_workspace_shapes):
        msg = (
            "MoE expert weight workspace has an invalid shape: "
            f"pairs={expert_workspace_shapes}"
        )
        raise ValueError(msg)


def _validate_moe_workspaces(
    dims: _MoeDimensions,
    workspace: MixtureOfExpertsWorkspace,
    communication_workspace: AttentionCommunicationWorkspace,
    expected_output_vmem_shape: tuple[int, ...],
    *,
    group_pending_expert_routes: bool,
) -> None:
    """Validate persistent local-memory shapes for MoE compute and transport."""
    accumulator_rows = _expert_accumulator_rows(
        dims.batch_size,
        group_pending_routes=group_pending_expert_routes,
    )
    compact_rows = max(1, _compact_rows(dims.batch_size))
    shared_cols = (
        _HIDDEN_SHARD_COLS if _shared_shards(dims.batch_size) else dims.hidden_size
    )
    expected_workspace_shapes = (
        (workspace.selected_experts_vmem.shape, (dims.batch_size, VREG_LANE_COUNT)),
        (
            workspace.selected_probabilities_vmem.shape,
            (dims.batch_size, VREG_LANE_COUNT),
        ),
        (
            workspace.local_route_counts_vmem.shape,
            (dims.batch_size, VREG_LANE_COUNT),
        ),
        (
            workspace.probability_totals_vmem.shape,
            (dims.batch_size, VREG_LANE_COUNT),
        ),
        (
            workspace.routing_scores_vmem.shape,
            (dims.batch_size, _HIDDEN_SHARD_COLS),
        ),
        (workspace.shared_outputs_vmem.shape, (dims.batch_size, shared_cols)),
        (workspace.routed_outputs_vmem.shape, (dims.batch_size, dims.latent_size)),
        (
            workspace.gate_up_accumulator_vmem.shape,
            (accumulator_rows[0], 4 * dims.expert_width),
        ),
        (
            workspace.down_accumulator_vmem.shape,
            (accumulator_rows[1], dims.expert_output_size),
        ),
        (workspace.processed_routes_vmem.shape, (dims.batch_size, VREG_LANE_COUNT)),
        (
            workspace.first_compact_source_vmem.shape,
            (compact_rows, dims.expert_input_size),
        ),
        (
            workspace.second_compact_source_vmem.shape,
            (compact_rows, dims.expert_input_size),
        ),
    )
    if any(actual != expected for actual, expected in expected_workspace_shapes):
        msg = "MoE workspace has an invalid shape"
        raise ValueError(msg)
    if workspace.output_vmem.shape != expected_output_vmem_shape:
        msg = "MoE output communication workspace has the wrong shape"
        raise ValueError(msg)
    if communication_workspace.latent_gather_vmem.shape != (
        dims.batch_size,
        _LATENT_GATHER_WIDTH,
    ):
        msg = "MoE latent-gather workspace must match the batch size"
        raise ValueError(msg)


def _start_moe_latent_gather(
    hidden: jax.Array,
    latent_input_projection_vmem: PallasRefProtocol,
    latent_input_copy: AsyncCopyProtocol,
    latent_ctx: _MoeLatentGatherContext,
) -> None:
    """Project this rank's latent input shard and start the 32-way all-gather."""
    latent_input_copy.wait()
    latent_ctx.latent_vmem[
        :, pl.ds(latent_ctx.rank * VREG_LANE_COUNT, VREG_LANE_COUNT)
    ] = dot(hidden, latent_input_projection_vmem[...]).astype(jnp.bfloat16)
    for offset in _LATENT_GATHER_PEER_OFFSETS:
        view = latent_ctx.latent_vmem.at[
            :, pl.ds(latent_ctx.rank * VREG_LANE_COUNT, VREG_LANE_COUNT)
        ]
        tpu.make_async_remote_copy(
            view,
            view,
            latent_ctx.latent_send_semaphores.at[latent_ctx.latent_slot],
            latent_ctx.latent_receive_semaphores.at[latent_ctx.latent_slot],
            device_id=(latent_ctx.rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()


def _wait_for_moe_latent_gather(latent_ctx: _MoeLatentGatherContext) -> None:
    """Wait for all 31 remote peers in the MoE latent input all-gather."""
    view = latent_ctx.latent_vmem.at[:, : TP_PEERS * VREG_LANE_COUNT]
    tpu.make_async_copy(
        view, view, latent_ctx.latent_send_semaphores.at[latent_ctx.latent_slot]
    ).wait()
    tpu.make_async_copy(
        view, view, latent_ctx.latent_receive_semaphores.at[latent_ctx.latent_slot]
    ).wait()


def _compute_moe_shared_expert(
    ctx: _MoePrepareContext,
    resources: _MoePrepareResources,
    gate_up_copy: AsyncCopyProtocol | None,
    down_copy: AsyncCopyProtocol,
) -> None:
    """Project and activate the shared expert during batch preparation."""
    if ctx.shared_gate_up_stream is not None:
        shared_gate_up = (
            ctx.shared_gate_up_stream.dot(
                ctx.hidden,
                ctx.communication_workspace.reduction_phase[0] >= 0,
            )
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
    else:
        if gate_up_copy is None:
            msg = "Shared gate/up async copy missing without stream"
            raise ValueError(msg)
        gate_up_copy.wait()
        shared_gate_up = (
            dot(ctx.hidden, resources.shared_gate_up_projection_vmem[...])
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
    middle = _gated_activation(
        shared_gate_up[:, : ctx.dims.shared_width],
        shared_gate_up[:, ctx.dims.shared_width :],
    )
    if not ctx.overlap_shared_routing:
        down_copy.wait()
    if ctx.shared_sharded:
        # The routed outputs are zeroed by the expert stream; until then
        # they hold the BF16 activation (exact in FP32).
        ctx.workspace.routed_outputs_vmem[:, : ctx.dims.shared_width] = middle.astype(
            jnp.float32
        )
    else:
        ctx.workspace.shared_outputs_vmem[...] = dot(
            middle, resources.shared_down_projection_vmem[...]
        )


def _route_and_prefetch_moe_batch(
    ctx: _MoePrepareContext,
    on_local_selection: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], None],
    after_selection: Callable[[int], None],
    after_score_send: Callable[[], None] | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run top-k router selection and return selected experts and probabilities."""
    with _phase("p3_route"):
        if ctx.owned_hidden_ref is not None:
            if (
                ctx.workspace.selection_source_vmem is None
                or ctx.workspace.selections_vmem is None
            ):
                msg = "Sharded routing requires selection scratch buffers"
                raise ValueError(msg)
            return _route_owned_rows(
                ctx.owned_hidden_ref,
                ctx.dims.batch_size,
                ctx.weights.routing_projection_vmem,
                ctx.weights.routing_bias,
                ctx.workspace.routing_scores_vmem,
                ctx.workspace.selection_source_vmem,
                ctx.workspace.selections_vmem,
                ctx.communication_workspace.send_semaphores,
                ctx.communication_workspace.receive_semaphores,
                selected_expert_count=ctx.selected_expert_count,
                local_expert_count=ctx.dims.local_expert_count,
                expert_tensor_parallel=ctx.dims.expert_tensor_parallel,
                on_local_selection=on_local_selection,
                after_selection=after_selection,
                after_scores=ctx.after_routing_scores,
                live_rows=ctx.live_rows,
            )
        return _route_hidden_states(
            ctx.hidden,
            ctx.weights.routing_projection_vmem,
            ctx.weights.routing_bias,
            ctx.workspace.routing_scores_vmem,
            ctx.communication_workspace.send_semaphores.at[0],
            ctx.communication_workspace.receive_semaphores.at[0],
            selected_expert_count=ctx.selected_expert_count,
            local_expert_count=ctx.dims.local_expert_count,
            expert_tensor_parallel=ctx.dims.expert_tensor_parallel,
            on_local_selection=on_local_selection,
            after_selection=after_selection,
            after_scores=ctx.after_routing_scores,
            after_score_send=after_score_send,
            live_rows=ctx.live_rows,
        )


def _store_and_prefetch_moe_routes(
    ctx: _MoePrepareContext,
    routed: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    prefetched_expert_count: jax.Array,
) -> None:
    """Write router outputs to local memory and start initial expert weight copies."""
    (
        selected_experts,
        selected_probabilities,
        local_route_counts,
        selected_probability_totals,
    ) = routed
    pad_cols = VREG_LANE_COUNT - ctx.selected_expert_count
    ctx.workspace.selected_experts_vmem[...] = jnp.pad(
        selected_experts,
        ((0, 0), (0, pad_cols)),
    )
    ctx.workspace.selected_probabilities_vmem[...] = jnp.pad(
        selected_probabilities,
        ((0, 0), (0, pad_cols)),
    )
    ctx.workspace.local_route_counts_vmem[...] = jnp.broadcast_to(
        local_route_counts[:, None],
        ctx.workspace.local_route_counts_vmem.shape,
    )
    ctx.workspace.probability_totals_vmem[...] = jnp.broadcast_to(
        selected_probability_totals[:, None],
        ctx.workspace.probability_totals_vmem.shape,
    )

    def copy_expert(
        expert: int | jax.Array,
        slot: int | jax.Array,
        *,
        gate_up: bool = True,
        down: bool = True,
    ) -> SimpleNamespace:
        """Create async copies for one routed expert slot."""
        return ctx.weights.make_expert_copies(
            expert,
            slot,
            gate_up=gate_up,
            down=down,
        )

    _start_initial_expert_weight_copies(
        selected_experts,
        local_route_counts,
        copy_expert,
        slot_count=_EXPERT_PREFETCH_SLOTS,
        already_started=prefetched_expert_count,
        group_pending_routes=ctx.group_pending_expert_routes,
    )


def _start_moe_prepare_copies(
    ctx: _MoePrepareContext,
    resources: _MoePrepareResources,
) -> tuple[AsyncCopyProtocol, AsyncCopyProtocol | None, AsyncCopyProtocol]:
    """Start latent-input and shared-expert weight copies for batch preparation."""
    latent_input_copy = tpu.make_async_copy(
        ctx.weights.latent_input_projection,
        resources.latent_input_projection_vmem,
        resources.latent_input_copy_semaphore,
    )
    latent_input_copy.start()
    gate_up_copy = (
        None
        if ctx.shared_gate_up_stream is not None
        else tpu.make_async_copy(
            ctx.weights.shared_gate_up_projection,
            resources.shared_gate_up_projection_vmem,
            resources.dense_copy_semaphores.at[0],
        )
    )
    down_copy = tpu.make_async_copy(
        ctx.weights.shared_down_projection,
        resources.shared_down_projection_vmem,
        resources.dense_copy_semaphores.at[1],
    )
    dense_copies = SimpleNamespace(gate_up=gate_up_copy, down=down_copy)
    _start_async_copies(dense_copies)
    if ctx.shared_gate_up_stream is not None:
        ctx.shared_gate_up_stream.start()
    ctx.weights.wait_for_routing_projection_copy()
    return latent_input_copy, gate_up_copy, down_copy


def _prepare_moe_batch(
    ctx: _MoePrepareContext,
    resources: _MoePrepareResources,
) -> None:
    """Run MoE batch preparation: router top-k, latent gather, and shared expert."""
    latent_input_copy, gate_up_copy, down_copy = _start_moe_prepare_copies(
        ctx, resources
    )
    shared_route_prefix = jnp.int32(1)
    prefetched_expert_count = jnp.int32(0)

    def prefetch_shared_route_prefix(
        winners: jax.Array,
        owned_locally: jax.Array,
        route_counts: jax.Array,
        _selected_experts: jax.Array,
    ) -> None:
        """Prefetch expert weights while all batch rows share a route prefix."""
        # Router selection order is not the sorted expert-id order of
        # grouped execution. Its weight slots can only be populated once
        # all routes are known; otherwise a wait can consume another
        # expert's correctly sized DMA without detecting the mismatch.
        if ctx.group_pending_expert_routes:
            return
        nonlocal shared_route_prefix, prefetched_expert_count
        shared_route_prefix *= jnp.all(winners == winners[0]).astype(jnp.int32)
        should_prefetch = (
            shared_route_prefix
            * owned_locally.astype(jnp.int32)[0]
            * (route_counts[0] < _EXPERT_PREFETCH_SLOTS).astype(jnp.int32)
        )

        @pl.when(should_prefetch != 0)
        def start_expert_copy() -> None:
            """Start async weight copies for a shared-prefix local expert."""
            _start_async_copies(
                ctx.weights.make_expert_copies(
                    winners[0] % ctx.dims.local_expert_count,
                    route_counts[0],
                )
            )

        prefetched_expert_count += should_prefetch

    def overlap_router_score_exchange() -> None:
        """Start latent input gather while router score halves are exchanged."""
        # Latent projection reads hidden states, not router scores. Its
        # MXU work can run while the score halves move between chip cores.
        _start_moe_latent_gather(
            ctx.hidden,
            resources.latent_input_projection_vmem,
            latent_input_copy,
            ctx.latent_ctx,
        )

    def overlap_routing_with_latent_gather(selection: int) -> None:
        """Overlap shared-expert compute or latent gather with top-k steps."""
        if ctx.overlap_shared_routing and selection == _SHARED_ROUTING_OVERLAP_STEP:
            _compute_moe_shared_expert(ctx, resources, gate_up_copy, down_copy)
        if (
            selection == _LATENT_GATHER_OVERLAP_STEP
            and ctx.overlap_latent_gather
            and not ctx.overlap_router_latent
        ):
            _start_moe_latent_gather(
                ctx.hidden,
                resources.latent_input_projection_vmem,
                latent_input_copy,
                ctx.latent_ctx,
            )

    routed = _route_and_prefetch_moe_batch(
        ctx,
        prefetch_shared_route_prefix,
        overlap_routing_with_latent_gather,
        overlap_router_score_exchange if ctx.overlap_router_latent else None,
    )
    _store_and_prefetch_moe_routes(ctx, routed, prefetched_expert_count)

    if not ctx.overlap_latent_gather:
        _start_moe_latent_gather(
            ctx.hidden,
            resources.latent_input_projection_vmem,
            latent_input_copy,
            ctx.latent_ctx,
        )
        _wait_for_moe_latent_gather(ctx.latent_ctx)

    if ctx.overlap_shared_routing:
        down_copy.wait()
    else:
        _compute_moe_shared_expert(ctx, resources, gate_up_copy, down_copy)


def _run_scoped_moe_prepare(
    prepare_ctx: _MoePrepareContext,
    prepare_weights_vmem: Sequence[PallasRefProtocol] | None,
) -> None:
    """Allocate or bind dense preparation buffers and execute batch preparation."""
    weights = prepare_ctx.weights
    if prepare_weights_vmem is not None:
        latent_down_vmem = prepare_weights_vmem[0]
        shared_gate_up_vmem = prepare_weights_vmem[1]
        shared_down_vmem = prepare_weights_vmem[2]
        pl.run_scoped(
            lambda latent_semaphore, dense_semaphores: _prepare_moe_batch(
                prepare_ctx,
                _MoePrepareResources(
                    latent_input_projection_vmem=latent_down_vmem,
                    latent_input_copy_semaphore=latent_semaphore,
                    shared_gate_up_projection_vmem=shared_gate_up_vmem,
                    shared_down_projection_vmem=shared_down_vmem,
                    dense_copy_semaphores=dense_semaphores,
                ),
            ),
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA((2,)),
        )
    else:

        def prepare_batch(
            latent_input_projection_vmem: PallasRefProtocol,
            latent_input_copy_semaphore: PallasRefProtocol,
            shared_gate_up_projection_vmem: PallasRefProtocol,
            shared_down_projection_vmem: PallasRefProtocol,
            dense_copy_semaphores: PallasRefProtocol,
        ) -> None:
            """Stage dense projections and route the current decode batch."""
            _prepare_moe_batch(
                prepare_ctx,
                _MoePrepareResources(
                    latent_input_projection_vmem=latent_input_projection_vmem,
                    latent_input_copy_semaphore=latent_input_copy_semaphore,
                    shared_gate_up_projection_vmem=shared_gate_up_projection_vmem,
                    shared_down_projection_vmem=shared_down_projection_vmem,
                    dense_copy_semaphores=dense_copy_semaphores,
                ),
            )

        pl.run_scoped(
            prepare_batch,
            tpu.VMEM(
                tuple(weights.latent_input_projection.shape),
                weights.latent_input_projection.dtype,
            ),
            tpu.SemaphoreType.DMA,
            tpu.VMEM(
                tuple(weights.shared_gate_up_projection.shape),
                weights.shared_gate_up_projection.dtype,
            ),
            tpu.VMEM(
                tuple(weights.shared_down_projection.shape),
                weights.shared_down_projection.dtype,
            ),
            tpu.SemaphoreType.DMA((2,)),
        )


def _encode_moe_bf16_rows(
    value: jax.Array,
    destination: PallasRefProtocol,
    rows_per_token: int,
    payload_per_token: int,
) -> None:
    """Pack BF16 row pairs into FP32 collective transport slots."""
    transport_batch_size = value.shape[0]
    packed = tpu.bitcast(
        value.reshape(transport_batch_size * rows_per_token, VREG_LANE_COUNT).astype(
            jnp.bfloat16
        ),
        jnp.uint32,
    )
    destination[: transport_batch_size * payload_per_token, :] = (
        jax.lax.bitcast_convert_type(packed, jnp.float32)
    )


def _decode_moe_bf16_sum(
    value: jax.Array,
    payload_per_token: int,
    output_width: int,
) -> jax.Array:
    """Unpack FP32 transport slots as BF16, sum across peers, and reshape."""
    transport_batch_size = value.shape[1] // payload_per_token
    decoded = tpu.bitcast(
        jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
    ).astype(jnp.float32)
    return jnp.sum(decoded, axis=0).reshape(transport_batch_size, output_width)


def _encode_moe_expert(
    ctx: _MoeReductionContext,
    value: jax.Array,
    destination: PallasRefProtocol,
) -> None:
    """Pack active expert output columns as BF16 pairs into FP32 transport."""
    transport_batch_size = value.shape[0]
    bits = tpu.bitcast(
        value[:, : ctx.dims.expert_output_size]
        .reshape(transport_batch_size * ctx.expert_rows, VREG_LANE_COUNT)
        .astype(jnp.bfloat16),
        jnp.uint32,
    )
    destination[: transport_batch_size * ctx.expert_payload, :] = (
        jax.lax.bitcast_convert_type(bits, jnp.float32)
    )


def _decode_moe_gathered_experts(
    ctx: _MoeReductionContext,
    value: jax.Array,
) -> jax.Array:
    """Unpack all-gathered routed-expert token shards back to batch order."""
    values = tpu.bitcast(
        jax.lax.bitcast_convert_type(value, jnp.uint32), jnp.bfloat16
    ).astype(jnp.float32)
    return values.reshape(
        mla_dcp.HOST_LANES * ctx.tokens_per_lane, ctx.dims.expert_output_size
    )[: ctx.dims.batch_size]


def _make_moe_shared_copy(
    ctx: _MoeReductionContext,
    payload: int,
    offset: int,
    *,
    across_hosts: bool = False,
) -> AsyncCopyProtocol:
    """Build a local or cross-host remote copy for shared-expert reduction."""
    lane = ctx.rank % mla_dcp.HOST_LANES
    host = ctx.rank // mla_dcp.HOST_LANES
    comm = ctx.communication_workspace
    view = (
        comm.shared_host_reduction_vmem.at[host, :payload, :]
        if across_hosts
        else comm.shared_local_reduction_vmem.at[lane, :payload, :]
    )
    semaphore = 6 + offset if across_hosts else offset - 1
    peer = ctx.rank ^ (mla_dcp.HOST_LANES * offset if across_hosts else offset)
    return tpu.make_async_remote_copy(
        view,
        view,
        ctx.workspace.shared_send_semaphores.at[semaphore],
        ctx.workspace.shared_receive_semaphores.at[semaphore],
        device_id=(peer,),
        device_id_type=pl.DeviceIdType.MESH,
    )


def _moe_shared_shard_send_view(
    ctx: _MoeReductionContext,
    target_lane: int | jax.Array,
) -> PallasRefProtocol:
    """Return the aliased send view for a target lane's shared-expert shard."""
    # Reuse the expert host buffer until the routed local reduction
    # finishes and needs it for its normal role.
    host_buf = ctx.communication_workspace.expert_host_reduction_vmem
    send_rows = host_buf.shape[0] * host_buf.shape[1] // mla_dcp.HOST_LANES
    return cast(
        "PallasRefProtocol",
        pool_alias.view(
            host_buf,
            (mla_dcp.HOST_LANES, send_rows, VREG_LANE_COUNT),
            jnp.float32,
        ).at[target_lane, : ctx.dims.batch_size * ctx.shared_shard_payload, :],
    )


def _moe_shared_shard_receive_view(
    ctx: _MoeReductionContext,
    source_lane: int | jax.Array,
) -> PallasRefProtocol:
    """Return the local receive view for a source lane's shared-expert shard."""
    return ctx.communication_workspace.shared_local_reduction_vmem.at[
        source_lane, : ctx.dims.batch_size * ctx.shared_shard_payload, :
    ]


def _make_moe_shared_shard_copy(
    ctx: _MoeReductionContext,
    offset: int,
) -> AsyncCopyProtocol:
    """Build a local reduce-scatter copy for one shared-expert shard peer."""
    source_lane = ctx.rank % mla_dcp.HOST_LANES
    target_lane = source_lane ^ offset
    return tpu.make_async_remote_copy(
        _moe_shared_shard_send_view(ctx, target_lane),
        _moe_shared_shard_receive_view(ctx, source_lane),
        ctx.workspace.shared_send_semaphores.at[offset - 1],
        ctx.workspace.shared_receive_semaphores.at[offset - 1],
        device_id=(ctx.rank ^ offset,),
        device_id_type=pl.DeviceIdType.MESH,
    )


def _make_moe_expert_copy(
    ctx: _MoeReductionContext,
    payload: int,
    offset: int,
    *,
    across_hosts: bool = False,
) -> AsyncCopyProtocol:
    """Build a local or cross-host remote copy for routed-expert reduction."""
    comm = ctx.communication_workspace
    view = (
        comm.expert_host_reduction_vmem.at[ctx.rank // mla_dcp.HOST_LANES, :payload, :]
        if across_hosts
        else comm.expert_local_reduction_vmem.at[
            ctx.rank % mla_dcp.HOST_LANES, :payload, :
        ]
    )
    semaphore = 6 + offset if across_hosts else offset - 1
    peer = ctx.rank ^ (mla_dcp.HOST_LANES * offset if across_hosts else offset)
    return tpu.make_async_remote_copy(
        view,
        view,
        comm.send_semaphores.at[semaphore],
        comm.receive_semaphores.at[semaphore],
        device_id=(peer,),
        device_id_type=pl.DeviceIdType.MESH,
    )


def _moe_routed_send_storage(ctx: _MoeReductionContext) -> PallasRefProtocol:
    """Return the send buffer backing the 8-lane routed reduce-scatter."""
    comm = ctx.communication_workspace
    total_rows = mla_dcp.HOST_LANES * ctx.lane_payload
    if comm.routed_send_vmem is not None:
        return comm.routed_send_vmem.at[:total_rows, :]
    return comm.expert_local_reduction_vmem.at[
        ctx.rank % mla_dcp.HOST_LANES, :total_rows, :
    ]


def _moe_expert_shard_send_view(
    ctx: _MoeReductionContext,
    target_lane: int | jax.Array,
) -> PallasRefProtocol:
    """Return the routed reduce-scatter send slice for ``target_lane``."""
    return (
        _moe_routed_send_storage(ctx)
        .reshape(mla_dcp.HOST_LANES, ctx.lane_payload, VREG_LANE_COUNT)
        .at[target_lane]
    )


def _moe_expert_shard_receive_view(
    ctx: _MoeReductionContext,
    source_lane: int | jax.Array,
) -> PallasRefProtocol:
    """Return the routed reduce-scatter receive slice from ``source_lane``."""
    return ctx.communication_workspace.expert_local_reduction_vmem.at[
        source_lane, : ctx.lane_payload, :
    ]


def _make_moe_expert_shard_copy(
    ctx: _MoeReductionContext,
    offset: int,
) -> AsyncCopyProtocol:
    """Build a local reduce-scatter copy for one routed-expert token shard."""
    source_lane = ctx.rank % mla_dcp.HOST_LANES
    target_lane = source_lane ^ offset
    comm = ctx.communication_workspace
    return tpu.make_async_remote_copy(
        _moe_expert_shard_send_view(ctx, target_lane),
        _moe_expert_shard_receive_view(ctx, source_lane),
        comm.send_semaphores.at[offset - 1],
        comm.receive_semaphores.at[offset - 1],
        device_id=(ctx.rank ^ offset,),
        device_id_type=pl.DeviceIdType.MESH,
    )


def _begin_moe_shared_reduction(
    ctx: _MoeReductionContext,
    local_shared_outputs: jax.Array,
) -> None:
    """Encode and launch local peer transfers for the shared-expert output."""
    if ctx.reduce_scatter_shared_output:
        lane = ctx.rank % mla_dcp.HOST_LANES
        for target_lane in range(_ACTIVE_HIDDEN_SHARD_LANES):
            columns = slice(
                target_lane * _HIDDEN_SHARD_COLS,
                (target_lane + 1) * _HIDDEN_SHARD_COLS,
            )
            if ctx.shared_sharded:
                if ctx.shared_down_weights is None:
                    msg = "shared shards need pooled weights"
                    raise ValueError(msg)
                shard_values = dot(
                    local_shared_outputs, ctx.shared_down_weights[:, columns]
                )
            else:
                shard_values = local_shared_outputs[:, columns]
            _encode_moe_bf16_rows(
                shard_values,
                _moe_shared_shard_send_view(ctx, target_lane),
                ctx.shared_shard_rows,
                ctx.shared_shard_payload,
            )
        _moe_shared_shard_send_view(ctx, _ACTIVE_HIDDEN_SHARD_LANES)[...] = jnp.zeros(
            (ctx.dims.batch_size * ctx.shared_shard_payload, VREG_LANE_COUNT),
            jnp.float32,
        )
        _moe_shared_shard_receive_view(ctx, lane)[...] = _moe_shared_shard_send_view(
            ctx, lane
        )[...]
        for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
            _make_moe_shared_shard_copy(ctx, offset).start()
    else:
        payload = local_shared_outputs.shape[0] * ctx.shared_payload
        local_buf = ctx.communication_workspace.shared_local_reduction_vmem
        if payload > local_buf.shape[1]:
            msg = "Shared local reduction payload exceeds workspace rows"
            raise ValueError(msg)
        _encode_moe_bf16_rows(
            local_shared_outputs,
            local_buf.at[ctx.rank % mla_dcp.HOST_LANES],
            ctx.shared_rows,
            ctx.shared_payload,
        )
        for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
            _make_moe_shared_copy(ctx, payload, offset).start()


def _finish_moe_shared_reduction(
    ctx: _MoeReductionContext,
    payload: int,
) -> None:
    """Complete intra-host shared-expert sum and start cross-host exchange."""
    comm = ctx.communication_workspace
    if ctx.reduce_scatter_shared_output:
        for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
            _make_moe_shared_shard_copy(ctx, offset).wait()
        _encode_moe_bf16_rows(
            _decode_moe_bf16_sum(
                comm.shared_local_reduction_vmem[:, :payload, :],
                ctx.shared_shard_payload,
                _HIDDEN_SHARD_COLS,
            ),
            comm.shared_host_reduction_vmem.at[ctx.rank // mla_dcp.HOST_LANES],
            ctx.shared_shard_rows,
            ctx.shared_shard_payload,
        )
    else:
        for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
            _make_moe_shared_copy(ctx, payload, offset).wait()
        _encode_moe_bf16_rows(
            _decode_moe_bf16_sum(
                comm.shared_local_reduction_vmem[:, :payload, :],
                ctx.shared_payload,
                ctx.shared_rows * VREG_LANE_COUNT,
            ),
            comm.shared_host_reduction_vmem.at[ctx.rank // mla_dcp.HOST_LANES],
            ctx.shared_rows,
            ctx.shared_payload,
        )
    for offset in range(3, 0, -1):
        _make_moe_shared_copy(ctx, payload, offset, across_hosts=True).start()


def _finish_moe_shared_host_reduction(
    ctx: _MoeReductionContext,
    payload: int,
    reduced_shared_outputs: PallasRefProtocol,
) -> None:
    """Wait for cross-host shared-expert reduction and store reduced outputs."""
    for offset in range(3, 0, -1):
        _make_moe_shared_copy(ctx, payload, offset, across_hosts=True).wait()
    host_buf = ctx.communication_workspace.shared_host_reduction_vmem
    if ctx.reduce_scatter_shared_output:
        lane = ctx.rank % mla_dcp.HOST_LANES

        @pl.when(lane < _ACTIVE_HIDDEN_SHARD_LANES)
        def store_shared_shard() -> None:
            """Write this lane's reduced 1024-column shared-expert shard."""
            (
                reduced_shared_outputs
                if ctx.shared_sharded
                else reduced_shared_outputs.at[
                    :,
                    pl.ds(lane * _HIDDEN_SHARD_COLS, _HIDDEN_SHARD_COLS),
                ]
            )[...] = _decode_moe_bf16_sum(
                host_buf[:, :payload, :],
                ctx.shared_shard_payload,
                _HIDDEN_SHARD_COLS,
            )
    else:
        reduced_shared_outputs[...] = _decode_moe_bf16_sum(
            host_buf[:, :payload, :],
            ctx.shared_payload,
            ctx.shared_rows * VREG_LANE_COUNT,
        )


def _finish_moe_routed_scatter_tail(
    ctx: _MoeReductionContext,
    shared_transport_payload: int,
) -> None:
    """Complete routed reduce-scatter, cross-host sum, and local all-gather."""
    comm = ctx.communication_workspace
    routed_consumed = comm.routed_consumed
    if routed_consumed is None:
        msg = "Routed reduce-scatter requires routed_consumed semaphore"
        raise ValueError(msg)
    t3 = phase_start("t3_routed_local")
    lane = ctx.rank % mla_dcp.HOST_LANES
    for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
        _make_moe_expert_shard_copy(ctx, offset).wait()
    owned_shard = _moe_expert_shard_send_view(ctx, lane)[...]
    _moe_expert_shard_receive_view(ctx, lane)[...] = owned_shard
    _encode_moe_expert(
        ctx,
        _decode_moe_bf16_sum(
            comm.expert_local_reduction_vmem[:, : ctx.lane_payload, :],
            ctx.expert_payload,
            ctx.dims.expert_output_size,
        ),
        comm.expert_host_reduction_vmem.at[ctx.rank // mla_dcp.HOST_LANES],
    )
    # The receive slots are read: host-mates' all-gathers may reuse them.
    for offset in range(1, mla_dcp.HOST_LANES):
        pl.semaphore_signal(
            routed_consumed,
            1,
            device_id=(ctx.rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    phase_end(t3)
    t4 = phase_start("t4_routed_host")
    for offset in range(3, 0, -1):
        _make_moe_expert_copy(ctx, ctx.lane_payload, offset, across_hosts=True).start()
    for offset in range(3, 0, -1):
        _make_moe_expert_copy(ctx, ctx.lane_payload, offset, across_hosts=True).wait()
    phase_end(t4)
    t5 = phase_start("t5_routed_gather")

    reduced_token = _decode_moe_bf16_sum(
        comm.expert_host_reduction_vmem[:, : ctx.lane_payload, :],
        ctx.expert_payload,
        ctx.dims.expert_output_size,
    )
    _encode_moe_expert(
        ctx,
        reduced_token,
        comm.expert_local_reduction_vmem.at[lane],
    )
    pl.semaphore_wait(routed_consumed, _ACTIVE_HIDDEN_SHARD_LANES)
    for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
        _make_moe_expert_copy(ctx, ctx.lane_payload, offset).start()
    if ctx.overlap_shared_reduction:
        # Shared-output conversion/reduction is independent of the
        # routed gather and uses a separate transport buffer.
        with _phase("t6_shared_during_routed_gather"):
            _finish_moe_shared_host_reduction(
                ctx, shared_transport_payload, ctx.workspace.shared_outputs_vmem
            )
    for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
        _make_moe_expert_copy(ctx, ctx.lane_payload, offset).wait()
    ctx.workspace.routed_outputs_vmem[:, : ctx.dims.expert_output_size] = (
        _decode_moe_gathered_experts(
            ctx,
            comm.expert_local_reduction_vmem[:, : ctx.lane_payload, :],
        )
    )
    phase_end(t5)
    if not ctx.overlap_shared_reduction:
        t6 = phase_start("t6_shared_host")
        _finish_moe_shared_host_reduction(
            ctx, shared_transport_payload, ctx.workspace.shared_outputs_vmem
        )
        phase_end(t6)
    comm.reduction_phase[0] += 2


def _finish_moe_routed_allreduce_tail(
    ctx: _MoeReductionContext,
    expert_transport_payload: int,
    shared_transport_payload: int,
) -> None:
    """Complete full-batch routed all-reduce and shared host reduction."""
    comm = ctx.communication_workspace
    reduced_expert_outputs = ctx.workspace.routed_outputs_vmem
    for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
        _make_moe_expert_copy(ctx, expert_transport_payload, offset).wait()
    _encode_moe_expert(
        ctx,
        _decode_moe_bf16_sum(
            comm.expert_local_reduction_vmem[:, :expert_transport_payload, :],
            ctx.expert_payload,
            ctx.dims.expert_output_size,
        ),
        comm.expert_host_reduction_vmem.at[ctx.rank // mla_dcp.HOST_LANES],
    )
    for offset in range(3, 0, -1):
        _make_moe_expert_copy(
            ctx, expert_transport_payload, offset, across_hosts=True
        ).start()
    for offset in range(3, 0, -1):
        _make_moe_expert_copy(
            ctx, expert_transport_payload, offset, across_hosts=True
        ).wait()
    reduced_expert_outputs[...] = jnp.zeros(reduced_expert_outputs.shape, jnp.float32)
    reduced_expert_outputs[:, : ctx.dims.expert_output_size] = _decode_moe_bf16_sum(
        comm.expert_host_reduction_vmem[:, :expert_transport_payload, :],
        ctx.expert_payload,
        ctx.dims.expert_output_size,
    )
    _finish_moe_shared_host_reduction(
        ctx, shared_transport_payload, ctx.workspace.shared_outputs_vmem
    )
    comm.reduction_phase[0] += 2


def _validate_moe_reduction_payloads(
    ctx: _MoeReductionContext,
    shared_transport_payload: int,
    expert_transport_payload: int,
) -> None:
    """Verify collective reduction payload sizes and scatter scratch invariants."""
    comm = ctx.communication_workspace
    if shared_transport_payload > comm.shared_local_reduction_vmem.shape[1]:
        msg = "Shared transport payload exceeds local reduction workspace"
        raise ValueError(msg)
    if (
        not ctx.reduce_scatter_routed_output
        and expert_transport_payload > comm.expert_local_reduction_vmem.shape[1]
    ):
        msg = "Expert transport payload exceeds local reduction workspace"
        raise ValueError(msg)
    if comm.routed_send_vmem is not None and not ctx.reduce_scatter_routed_output:
        msg = "routed_send_vmem requires routed reduce-scatter"
        raise ValueError(msg)
    if ctx.reduce_scatter_routed_output and comm.routed_consumed is None:
        msg = "Routed reduce-scatter requires routed_consumed semaphore"
        raise ValueError(msg)


def _finish_moe_reductions(
    ctx: _MoeReductionContext,
    local_expert_outputs: jax.Array,
    latent_output_copy: AsyncCopyProtocol,
    after_experts: Callable[[], None] | None,
) -> None:
    """Reduce routed and shared expert outputs across all 32 tensor-parallel ranks."""
    transport_batch_size = local_expert_outputs.shape[0]
    shared_transport_payload = transport_batch_size * (
        ctx.shared_shard_payload
        if ctx.reduce_scatter_shared_output
        else ctx.shared_payload
    )
    expert_transport_payload = transport_batch_size * ctx.expert_payload
    _validate_moe_reduction_payloads(
        ctx, shared_transport_payload, expert_transport_payload
    )

    latent_output_copy.start()
    if ctx.reduce_scatter_routed_output:
        t1 = phase_start("t1_routed_send")
        routed_scatter_storage = _moe_routed_send_storage(ctx)
        if ctx.dims.batch_size < mla_dcp.HOST_LANES * ctx.tokens_per_lane:
            # Zero the padding tokens of the last lanes.
            routed_scatter_storage[...] = jnp.zeros(
                routed_scatter_storage.shape, jnp.float32
            )
        _encode_moe_expert(ctx, local_expert_outputs, routed_scatter_storage)
        for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
            _make_moe_expert_shard_copy(ctx, offset).start()
        phase_end(t1)
    else:
        _encode_moe_expert(
            ctx,
            local_expert_outputs,
            ctx.communication_workspace.expert_local_reduction_vmem.at[
                ctx.rank % mla_dcp.HOST_LANES
            ],
        )
        for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
            _make_moe_expert_copy(ctx, expert_transport_payload, offset).start()

    if after_experts is not None:
        after_experts()

    # The polled path completes this during or immediately after experts.
    t2 = phase_start("t2_shared_local")
    if not ctx.overlap_shared_reduction:
        _finish_moe_shared_reduction(ctx, shared_transport_payload)
    phase_end(t2)

    if ctx.reduce_scatter_routed_output:
        _finish_moe_routed_scatter_tail(ctx, shared_transport_payload)
        return
    _finish_moe_routed_allreduce_tail(
        ctx, expert_transport_payload, shared_transport_payload
    )


def _invoke_moe_routed_experts(
    ctx: _MoeReductionContext,
    weights: MixtureOfExpertsWeights,
    selected_expert_count: int,
    *,
    group_pending_expert_routes: bool,
    progress: Callable[[jax.Array], None] | None,
) -> None:
    """Execute the routed-expert MXFP4 stream inside ``p4_experts``."""

    def copy_expert(
        expert: int | jax.Array,
        slot: int | jax.Array,
        *,
        gate_up: bool = True,
        down: bool = True,
    ) -> SimpleNamespace:
        """Create async copies for one routed expert slot."""
        return weights.make_expert_copies(
            expert,
            slot,
            gate_up=gate_up,
            down=down,
        )

    ws = ctx.workspace
    with _phase("p4_experts"):
        _run_routed_expert_stream(
            ctx.communication_workspace.latent_gather_vmem,
            ws.selected_experts_vmem[:, :selected_expert_count],
            ws.selected_probabilities_vmem[:, :selected_expert_count],
            ws.local_route_counts_vmem[:, 0],
            ws.probability_totals_vmem[:, 0],
            weights.expert_gate_up_projection_vmem,
            weights.expert_gate_up_scales_vmem,
            weights.expert_down_projection_vmem,
            weights.expert_down_scales_vmem,
            ws.routed_outputs_vmem,
            ws.gate_up_accumulator_vmem,
            ws.down_accumulator_vmem,
            ws.processed_routes_vmem,
            ws.first_compact_source_vmem,
            ws.second_compact_source_vmem,
            copy_expert,
            expert_input_size=ctx.dims.expert_input_size,
            expert_width=ctx.dims.expert_width,
            expert_output_size=ctx.dims.expert_output_size,
            group_pending_routes=group_pending_expert_routes,
            progress=progress,
        )


def _run_moe_experts_with_shared_progress(
    ctx: _MoeReductionContext,
    weights: MixtureOfExpertsWeights,
    selected_expert_count: int,
    *,
    group_pending_expert_routes: bool,
) -> None:
    """Poll intra-host shared-expert transfers during routed-expert execution."""

    def run_with_shared_progress(progress_state: PallasRefProtocol) -> None:
        """Advance shared local reduction once all seven peer transfers land."""
        progress_state[0] = 0
        payload = ctx.dims.batch_size * (
            ctx.shared_shard_payload
            if ctx.reduce_scatter_shared_output
            else ctx.shared_payload
        )
        # Seven copies, each with its own send/receive semaphore. DMA counters
        # count 32-byte units; both ends must be complete before reusing sources.
        expected = payload * VREG_LANE_COUNT * 4 // 32

        def poll_shared(iteration: jax.Array) -> None:
            """Check shared-expert DMA semaphores every four expert iterations."""

            @pl.when((progress_state[0] == 0) & (iteration % 4 == 0))
            def try_progress() -> None:
                """Advance shared reduction if all seven local peers completed."""
                ready = jnp.bool_(1)
                for peer in range(_ACTIVE_HIDDEN_SHARD_LANES):
                    ready = ready & (
                        pl.semaphore_read(ctx.workspace.shared_send_semaphores.at[peer])
                        >= expected
                    )
                    ready = ready & (
                        pl.semaphore_read(
                            ctx.workspace.shared_receive_semaphores.at[peer]
                        )
                        >= expected
                    )

                @pl.when(ready)
                def advance() -> None:
                    """Finish local shared sum and launch cross-host copies."""
                    with _phase("t2_shared_local_polled"):
                        _finish_moe_shared_reduction(ctx, payload)
                    progress_state[0] = 1

        _invoke_moe_routed_experts(
            ctx,
            weights,
            selected_expert_count,
            group_pending_expert_routes=group_pending_expert_routes,
            progress=poll_shared,
        )

        @pl.when(progress_state[0] == 0)
        def drain_shared() -> None:
            """Drain intra-host shared reduction if polling did not finish it."""
            with _phase("t2_shared_local_drain"):
                _finish_moe_shared_reduction(ctx, payload)
            progress_state[0] = 1

    pl.run_scoped(
        run_with_shared_progress,
        progress_state=tpu.SMEM((1,), jnp.int32),
    )


def _project_moe_latent_up(
    ctx: _MoeReductionContext,
    weights: MixtureOfExpertsWeights,
    latent_output_copy: AsyncCopyProtocol,
    *,
    latent_feature_count: int,
    epsilon: float,
) -> jax.Array:
    """Normalize reduced routed latent features, up-project, and add shared shard."""
    t7 = phase_start("t7_latent_up")
    routed = (
        ctx.workspace.routed_outputs_vmem[...].astype(jnp.bfloat16).astype(jnp.float32)
    )
    normalized = (
        routed
        * jax.lax.rsqrt(
            jnp.sum(routed * routed, axis=1, keepdims=True) / latent_feature_count
            + epsilon
        )
        * weights.latent_normalization[...]
    ).astype(jnp.bfloat16)
    latent_output_copy.wait()
    # The contraction is independent across batch rows. Let Mosaic lower
    # it as one native row tile so the resident weight is consumed once.
    up = dot(
        normalized[:, : ctx.dims.expert_output_size],
        weights.latent_output_projection_vmem[...],
    )
    shared_part = (
        ctx.workspace.shared_outputs_vmem[...]
        if ctx.shared_sharded
        else ctx.workspace.shared_outputs_vmem[
            :,
            pl.ds(
                (ctx.rank % mla_dcp.HOST_LANES) * _HIDDEN_SHARD_COLS,
                _HIDDEN_SHARD_COLS,
            ),
        ]
    )
    local_output = (up + shared_part).astype(jnp.bfloat16)
    phase_end(t7)
    return local_output


def _transport_moe_output(
    ctx: _MoeReductionContext,
    local_output: jax.Array,
    output_ref: PallasRefProtocol,
    output_dtype: jnp.dtype,
) -> None:
    """Scatter token-owned rows or all-gather hidden shards into ``output_ref``."""
    t8 = phase_start("t8_output_transport")
    ws = ctx.workspace
    if ctx.token_scatter_output:
        # Each lane owns 1024 hidden columns for all tokens. Send each peer
        # only the rows owned by its local row owner (rank pair or rank).
        # All seven transfers are independent messages, so the token /
        # feature transpose completes in one communication phase.
        owners = ctx.token_scatter_owners
        tokens_per_pair = ctx.dims.batch_size // owners
        payload_rows = tokens_per_pair * _HIDDEN_SHARD_COLS // VREG_LANE_COUNT
        lane = ctx.rank % mla_dcp.HOST_LANES
        pair = lane // (mla_dcp.HOST_LANES // owners)
        send = ws.output_vmem.at[pl.ds(0, owners)]
        receive = ws.output_vmem.at[pl.ds(owners, mla_dcp.HOST_LANES)]
        for target_pair in range(owners):
            row_start = target_pair * tokens_per_pair
            send[target_pair] = local_output[
                row_start : row_start + tokens_per_pair
            ].reshape(payload_rows, VREG_LANE_COUNT)
        receive[lane] = send[pair]
        copies = []
        for offset in range(_ACTIVE_HIDDEN_SHARD_LANES, 0, -1):
            target_lane = lane ^ offset
            target_pair = target_lane // (mla_dcp.HOST_LANES // owners)
            copy = tpu.make_async_remote_copy(
                send.at[target_pair],
                receive.at[lane],
                ws.output_send_semaphores.at[offset - 1],
                ws.output_receive_semaphores.at[offset - 1],
                device_id=(ctx.rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

        for source_lane in range(_ACTIVE_HIDDEN_SHARD_LANES):
            block = receive[source_lane].reshape(tokens_per_pair, _HIDDEN_SHARD_COLS)
            output_ref[
                :, pl.ds(source_lane * _HIDDEN_SHARD_COLS, _HIDDEN_SHARD_COLS)
            ] = block
    else:
        ws.output_vmem[
            :,
            pl.ds(
                (ctx.rank % mla_dcp.HOST_LANES) * _HIDDEN_SHARD_COLS,
                _HIDDEN_SHARD_COLS,
            ),
        ] = local_output
        _gather_output_shards(
            ws.output_vmem,
            ws.output_send_semaphores,
            ws.output_receive_semaphores,
            ctx.communication_workspace.reduction_phase[0] % 2,
        )
        output_ref[...] = ws.output_vmem[
            : ctx.dims.batch_size, : ctx.dims.hidden_size
        ].astype(output_dtype)
    phase_end(t8)
    ctx.communication_workspace.reduction_phase[0] += 1


def _build_moe_reduction_context(
    dims: _MoeDimensions,
    workspace: MixtureOfExpertsWorkspace,
    communication_workspace: AttentionCommunicationWorkspace,
    rank: jax.Array,
    shared_down_weights: PallasRefProtocol | None,
) -> _MoeReductionContext:
    """Build base transport geometry for MoE reductions."""
    shared_rows = (
        (dims.hidden_size + _HIDDEN_SHARD_COLS - 1) // _HIDDEN_SHARD_COLS
    ) * mla_dcp.HOST_LANES
    shared_payload = shared_rows // 2
    expert_rows = dims.expert_output_size // VREG_LANE_COUNT
    expert_payload = expert_rows // 2
    # The routed reduce-scatter gives each local lane ceil(batch / 8) tokens
    # (B6 pads to eight).
    tokens_per_lane = -(-dims.batch_size // mla_dcp.HOST_LANES)
    lane_payload = tokens_per_lane * expert_payload
    shared_shard_rows = _HIDDEN_SHARD_COLS // VREG_LANE_COUNT
    shared_shard_payload = shared_shard_rows // 2
    return _MoeReductionContext(
        dims=dims,
        workspace=workspace,
        communication_workspace=communication_workspace,
        rank=rank,
        shared_down_weights=shared_down_weights,
        shared_rows=shared_rows,
        shared_payload=shared_payload,
        expert_rows=expert_rows,
        expert_payload=expert_payload,
        tokens_per_lane=tokens_per_lane,
        lane_payload=lane_payload,
        shared_shard_rows=shared_shard_rows,
        shared_shard_payload=shared_shard_payload,
        token_scatter_owners=1,
        shared_sharded=False,
        reduce_scatter_shared_output=False,
        reduce_scatter_routed_output=False,
        overlap_shared_reduction=False,
        token_scatter_output=False,
    )


def _start_moe_shared_reduction_phase(ctx: _MoeReductionContext) -> None:
    """Launch intra-host shared-expert reduction from sharded or full storage."""
    if ctx.shared_sharded:
        if not ctx.reduce_scatter_shared_output:
            msg = "Sharded shared expert outputs require shared reduce-scatter"
            raise ValueError(msg)
        _begin_moe_shared_reduction(
            ctx,
            ctx.workspace.routed_outputs_vmem[:, : ctx.dims.shared_width].astype(
                jnp.bfloat16
            ),
        )
    else:
        _begin_moe_shared_reduction(ctx, ctx.workspace.shared_outputs_vmem[...])


def moe(
    hidden_ref: PallasRefProtocol,  # [batch_size, hidden_size]
    output_ref: PallasRefProtocol,  # [batch_size, hidden_size] or owned token rows
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
    after_experts: Callable[[], None] | None = None,
    after_routing_scores: Callable[[], None] | None = None,
    overlap_router_latent: bool = False,
    overlap_shared_reduction: bool = False,
    prepare_weights_vmem: Sequence[PallasRefProtocol] | None = None,
    shared_gate_up_stream: ChunkedWeightStream | None = None,
    token_scatter_owners: int = 4,
    owned_hidden_ref: PallasRefProtocol | None = None,
    live_rows: jax.Array | None = None,
) -> None:
    """Execute one mixture-of-experts layer inside a Pallas program.

    ``token_scatter_output`` leaves each of the ``token_scatter_owners`` local
    row owners (4: rank pairs; 8: ranks) with its batch // owners rows.
    ``owned_hidden_ref`` (this rank pair's rows of ``hidden_ref``) shards the
    router's top-k over the rank pairs (_route_owned_rows).

    ``live_rows`` masks null serving slots before route prefetch/compaction;
    the router scores and shared expert still use the compiled batch shape.

    ``prepare_weights_vmem`` optionally supplies caller-owned local memory for
    the latent-down, shared gate/up and shared down projections instead of a
    scoped allocation.
    """
    dims = _derive_moe_dimensions(hidden_ref, weights)
    _validate_moe_modes_and_dims(
        dims,
        output_ref.shape,
        token_scatter_output=token_scatter_output,
        routed_reduce_scatter=routed_reduce_scatter,
        token_scatter_owners=token_scatter_owners,
    )
    _validate_moe_weight_tensors(dims, weights)
    expected_output_vmem_shape = (
        (
            token_scatter_owners + mla_dcp.HOST_LANES,
            dims.batch_size
            // token_scatter_owners
            * _HIDDEN_SHARD_COLS
            // VREG_LANE_COUNT,
            VREG_LANE_COUNT,
        )
        if token_scatter_output
        else (dims.batch_size, _MOE_GATHER_PACKET_COLS)
    )
    _validate_moe_workspaces(
        dims,
        workspace,
        communication_workspace,
        expected_output_vmem_shape,
        group_pending_expert_routes=group_pending_expert_routes,
    )

    shared_sharded = workspace.shared_outputs_vmem.shape[1] < dims.hidden_size
    # The pooled shared-down weight stays resident in region B until the
    # MoE tail, so the shards are projected when the reduce-scatter starts.
    shared_down_weights = (
        prepare_weights_vmem[2] if prepare_weights_vmem is not None else None
    )
    if shared_sharded and shared_down_weights is None:
        msg = "shared shards need pooled weights"
        raise ValueError(msg)

    rank = jax.lax.axis_index(AXIS_TP)
    if synchronize_devices:
        collectives32.barrier()
        communication_workspace.reduction_phase[0] = 0
    hidden = hidden_ref[...]

    if not routing_projection_is_prefetched:
        weights.start_routing_projection_copy()
    latent_output_copy = weights.make_latent_output_copy()
    overlap_latent_gather = selected_expert_count >= _MIN_LATENT_OVERLAP_EXPERTS
    effective_overlap_router_latent = (
        overlap_router_latent and overlap_latent_gather and owned_hidden_ref is None
    )
    overlap_shared_routing = effective_overlap_router_latent and shared_sharded
    latent_ctx = _MoeLatentGatherContext(
        rank=rank,
        latent_slot=communication_workspace.reduction_phase[0] % 2,
        latent_vmem=communication_workspace.latent_gather_vmem,
        latent_send_semaphores=workspace.latent_send_semaphores,
        latent_receive_semaphores=workspace.latent_receive_semaphores,
    )
    prepare_ctx = _MoePrepareContext(
        dims=dims,
        hidden=hidden,
        owned_hidden_ref=owned_hidden_ref,
        live_rows=live_rows,
        weights=weights,
        workspace=workspace,
        communication_workspace=communication_workspace,
        latent_ctx=latent_ctx,
        shared_gate_up_stream=shared_gate_up_stream,
        after_routing_scores=after_routing_scores,
        selected_expert_count=selected_expert_count,
        group_pending_expert_routes=group_pending_expert_routes,
        overlap_latent_gather=overlap_latent_gather,
        overlap_router_latent=effective_overlap_router_latent,
        overlap_shared_routing=overlap_shared_routing,
        shared_sharded=shared_sharded,
    )
    _run_scoped_moe_prepare(prepare_ctx, prepare_weights_vmem)

    # Each lane consumes only its 1024-column shared-expert output shard;
    # reduce-scattering it avoids replicating all 7168 columns, with the same
    # BF16 rounding points.
    reduce_scatter_shared_output = (
        dims.batch_size in _ROUTED_SCATTER_BATCHES and group_pending_expert_routes
    )
    # The routed payload is eight contiguous token shards: each local lane
    # reduces one, then a local all-gather restores the batch for latent-up.
    reduction_ctx = dataclasses.replace(
        _build_moe_reduction_context(
            dims,
            workspace,
            communication_workspace,
            rank,
            shared_down_weights,
        ),
        token_scatter_owners=token_scatter_owners,
        shared_sharded=shared_sharded,
        reduce_scatter_shared_output=reduce_scatter_shared_output,
        reduce_scatter_routed_output=routed_reduce_scatter,
        overlap_shared_reduction=overlap_shared_reduction,
        token_scatter_output=token_scatter_output,
    )
    _start_moe_shared_reduction_phase(reduction_ctx)
    if overlap_latent_gather:
        _wait_for_moe_latent_gather(latent_ctx)

    if overlap_shared_reduction:
        _run_moe_experts_with_shared_progress(
            reduction_ctx,
            weights,
            selected_expert_count,
            group_pending_expert_routes=group_pending_expert_routes,
        )
    else:
        _invoke_moe_routed_experts(
            reduction_ctx,
            weights,
            selected_expert_count,
            group_pending_expert_routes=group_pending_expert_routes,
            progress=None,
        )

    _finish_moe_reductions(
        reduction_ctx,
        workspace.routed_outputs_vmem[...],
        latent_output_copy,
        after_experts,
    )
    local_output = _project_moe_latent_up(
        reduction_ctx,
        weights,
        latent_output_copy,
        latent_feature_count=latent_feature_count,
        epsilon=epsilon,
    )
    _transport_moe_output(
        reduction_ctx,
        local_output,
        output_ref,
        hidden_ref.dtype,
    )


_PER_WEIGHT_STREAM_SLOTS: int = 4
_MIN_STREAM_SEMAPHORES: int = 8
_STATE_LOOKAHEAD_CHUNKS: int = 2
_SMALL_KDA_ROW_CHUNK: int = 8
_LARGE_KDA_ROW_CHUNK: int = 4
_CONV_WINDOW_TAPS: int = 4
_DEFERRED_STATE_DRAIN_BATCH: int = 32

# Weight streaming. Every large head-sharded attention weight is copied
# through a ChunkedWeightStream; KernelConfig.weight_streams (tuple of
# ``(name, (chunks, depth))`` entries) splits one into chunks through a ring
# of ``depth`` slots. The default, one chunk, is exactly the whole-weight
# copy and product.
STREAMED_WEIGHTS: dict[str, tuple[Literal["columns", "rows"], int]] = {
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
WEIGHT_STREAMS: dict[str, tuple[int, int]] = {
    name: (int(value[0]), int(value[1])) for name, value in _config().weight_streams
}
if set(WEIGHT_STREAMS) - set(STREAMED_WEIGHTS):
    _unknown_stream_msg = (
        "KernelConfig.weight_streams names unknown weights "
        f"{sorted(set(WEIGHT_STREAMS) - set(STREAMED_WEIGHTS))}"
    )
    raise ValueError(_unknown_stream_msg)


# Rings deeper than four slots (attention DP's small batches) signal from
# here, past every weight's own four: a KDA and an MLA layer's streams never
# overlap in time.
DEEP_STREAM_BASE: int = 24


def _stream_base(name: str, depth: int) -> int:
    """Return the starting stream-semaphore slot for ``name`` at ``depth``."""
    return (
        STREAMED_WEIGHTS[name][1]
        if depth <= _PER_WEIGHT_STREAM_SLOTS
        else DEEP_STREAM_BASE
    )


def _stream_semaphore_count(streams: Mapping[str, tuple[int, int]]) -> int:
    """Return the stream semaphore count required for ``streams``.

    Each streamed weight owns four slots from its base, with a minimum of eight
    semaphores to keep the scalar schedule stable.
    """
    return max(
        [_MIN_STREAM_SEMAPHORES]
        + [
            _stream_base(name, depth) + depth
            for name, (chunks, depth) in streams.items()
            if chunks > 1
        ]
    )


# B32/B64 paged DCP needs these chunks to leave room for communication scratch.
# These do not alter arithmetic or the resident weight representation.
DCP_B32_WEIGHT_STREAMS: dict[str, tuple[int, int]] = {
    "m_qa": (12, 4),
    "k_projection": (13, 4),
    "shared_gu": (3, 3),
    "m_gate": (3, 2),
}
# Plain decode's input-projection rings fill the arena that layer 0's dense
# slabs size.
DCP_B32_DECODE_RINGS: dict[str, tuple[int, int]] = {
    "m_qa": (12, 7),
    "k_projection": (13, 7),
}


# Attention data parallelism: with DP groups of 32 / dp ranks, every rank holds
# the logical TP32 shards t, t + 32 / dp, ... (t = rank % (32 / dp)) of each
# head-sharded attention weight, concatenated along the axis it streams on
# ("concat", per-layer axis) or stacked on a new leading axis ("stack").
# Replicated weights (k_gate, m_qa, m_ka, norms) are unchanged.
ATTENTION_DP_WEIGHTS: dict[str, tuple[Literal["concat", "stack"], int | None]] = {
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
DP_WEIGHT_STREAMS: dict[int, dict[str, tuple[int, int]]] = {
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


def _dp_dcp_weight_streams(
    dp: int,
    group_rows: int,
) -> dict[str, tuple[int, int]]:
    """Return attention-DP weight stream settings under context parallelism.

    The KDA input projection (13 chunks per held shard) and MLA query-down (12)
    stream so context-parallel pages and packets fit, while key-value-up stays
    resident. Ring depths shrink as ``group_rows`` grows; with one group row
    (DP2 B2), MLA query-up and gate are also copied whole.
    """
    if dp != DP2_GROUPS:
        depths = {8: (13, 12), 16: (7, 7)}.get(group_rows, (6, 4))
    elif group_rows <= _SMALL_DP_GROUP_ROWS:
        depths = (13, 12)
    else:
        depths = {16: (10, 9), 32: (7, 7)}.get(group_rows, (4, 4))
    streams: dict[str, tuple[int, int]] = {
        "k_projection": (13 * dp, depths[0]),
        "m_qa": (12, depths[1]),
        "m_kb": (dp, dp),
    }
    if dp == DP2_GROUPS and group_rows == 1:
        streams.update(m_qb=(dp, dp), m_gate=(dp, dp))
    return streams


def _attention_projection_streams(
    dp: int,
    schedule: str,
) -> dict[str, tuple[int, int]]:
    """Return static stream overrides for the attention-DP projection schedule.

    ``mla_up_double`` double-buffers MLA query-up and KV-up without changing
    the projection dot shapes; the gate projection stays single-buffered.
    """
    choices = ("default", "mla_up_double")
    if schedule not in choices:
        msg = f"attention_projection_schedule must be one of {choices}"
        raise ValueError(msg)
    if schedule == "default":
        return {}
    if dp not in (DP2_GROUPS, DP4_GROUPS):
        msg = f"Projection schedule experiments require attention DP2/DP4, got dp={dp}"
        raise ValueError(msg)
    return dict.fromkeys(("m_qb", "m_kb"), (dp, 2))


def _chunk_sizes(extent: int, chunks: int, unit: int) -> tuple[int, ...]:
    """Split ``extent`` into ``chunks`` whole ``unit`` blocks, larger first."""
    blocks, remainder = divmod(extent // unit, chunks)
    if extent % unit or not 1 <= chunks <= extent // unit:
        msg = f"Cannot split {extent} into {chunks} chunks of whole {unit}-blocks"
        raise ValueError(msg)
    return tuple(unit * (blocks + (index < remainder)) for index in range(chunks))


class ChunkedWeightStream:
    """A weight copied from HBM in chunks through a ring of local scratch slots.

    ``source`` ``[K, N]`` is split along ``axis`` (``"columns"``: output
    columns; ``"rows"``: the contraction) into ``sizes``. Position ``i`` of the
    stream holds chunk ``order(i)`` (default ``i``; may be a traced value when
    all chunks have one size), in slot ``i % depth``, signalling semaphore
    ``base + i % depth``. With ``depth`` equal to the chunk count, the slots
    are the chunks' own parts of ``full`` (the weight's whole local view); a
    smaller depth reuses slots inside ``full``'s pool range (``pool`` from byte
    ``offset``).

    ``start`` issues the first ``depth`` positions. Consumers ``wait`` for a
    position, read ``view``, then ``release`` it, which issues the position
    ``depth`` ahead into the freed slot; ``dot`` does this for column streams.
    Chunked products are not bit-identical to the whole-weight product because
    the compiler's matrix accumulation split depends on chunk shape.
    """

    def __init__(
        self,
        source: PallasRefProtocol,
        full: PallasRefProtocol,
        semaphores: PallasRefProtocol,
        *,
        axis: str,
        sizes: Sequence[int],
        depth: int,
        base: int = 0,
        pool: PallasRefProtocol | None = None,
        offset: int | None = None,
        order: Callable[[int], int | jax.Array] | None = None,
    ) -> None:
        """Initialize a chunked weight stream and validate its ring geometry."""
        if axis not in ("columns", "rows"):
            msg = "Stream axis must be 'columns' or 'rows'"
            raise ValueError(msg)
        self.axis: str = axis
        self.sizes: tuple[int, ...] = tuple(sizes)
        self.depth: int = depth
        self.chunks: int = len(self.sizes)
        self.shards: int = 1
        extent = source.shape[1 if axis == "columns" else 0]
        if sum(self.sizes) != extent:
            msg = "Chunk sizes must cover the streamed axis"
            raise ValueError(msg)
        if not 1 <= depth <= self.chunks or base + depth > semaphores.shape[0]:
            msg = "Stream depth must fit the chunks and the semaphores"
            raise ValueError(msg)
        if order is not None and len(set(self.sizes)) != 1:
            msg = "A reordered stream needs equal chunk sizes"
            raise ValueError(msg)
        self.source: PallasRefProtocol = source
        self.full: PallasRefProtocol = full
        self.semaphores: PallasRefProtocol = semaphores
        self.base: int = base
        self.order: Callable[[int], int | jax.Array] | None = order
        self.slots: tuple[PallasRefProtocol, ...] | None = None
        if depth < self.chunks:
            if pool is None or offset is None:
                msg = "A ring stream needs its weight's pool range"
                raise ValueError(msg)
            size = max(self.sizes)
            shape = (
                (full.shape[0], size) if axis == "columns" else (size, full.shape[1])
            )
            slot_bytes = pool_alias.storage_size(shape, full.dtype)
            if depth * slot_bytes > pool_alias.storage_size(full.shape, full.dtype):
                msg = "Ring slots exceed the weight's pool range"
                raise ValueError(msg)
            self.slots = tuple(
                cast(
                    "PallasRefProtocol",
                    pool_alias.view(
                        pool.at[
                            pl.ds(
                                (offset + slot * slot_bytes) // VREG_LANE_COUNT,
                                slot_bytes // VREG_LANE_COUNT,
                            ),
                            :,
                        ],
                        shape,
                        full.dtype,
                    ),
                )
                for slot in range(depth)
            )

    def _chunk(self, position: int) -> int | jax.Array:
        """Resolve the logical chunk index for stream ``position``."""
        return position if self.order is None else self.order(position)

    def _extent(self, position: int) -> tuple[int | jax.Array, int]:
        """Return ``(start_offset, chunk_size)`` along the streamed axis."""
        if self.order is None:
            return sum(self.sizes[:position]), self.sizes[position]
        chunk = self._chunk(position)
        return chunk * self.sizes[0], self.sizes[0]

    def _part(
        self,
        ref: PallasRefProtocol,
        start: int | jax.Array,
        size: int,
    ) -> PallasRefProtocol:
        """Slice ``ref`` along the configured stream axis."""
        if self.axis == "columns":
            return ref.at[:, pl.ds(start, size)]
        return ref.at[pl.ds(start, size)]

    def view(self, position: int) -> PallasRefProtocol:
        """Return the local scratch view holding ``position``."""
        start, size = self._extent(position)
        if self.chunks == 1:
            return self.full
        if self.slots is None:
            return self._part(self.full, start, size)
        return self._part(self.slots[position % self.depth], 0, size)

    def _copy(self, position: int) -> AsyncCopyProtocol:
        """Build the async DMA descriptor for stream ``position``."""
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
        """Issue async copies for the initial ``depth`` stream positions."""
        for position in range(self.depth):
            self._copy(position).start()

    def wait(self, position: int) -> None:
        """Wait for stream ``position``'s async copy to finish."""
        self._copy(position).wait()

    def wait_all(self) -> None:
        """Wait for every chunk when each chunk owns a dedicated slot."""
        if self.slots is not None:
            msg = "Only a stream with a slot per chunk can wait for all chunks"
            raise ValueError(msg)
        for position in range(self.chunks):
            self.wait(position)

    def release(self, position: int, guard: jax.Array | int) -> None:
        """Refill the slot freed by ``position`` when ``guard`` holds.

        ``guard`` is a runtime-true condition that keeps the refill in its own
        region, ordered after the loads of the chunk it overwrites.
        """
        if position + self.depth < self.chunks:

            @pl.when(guard)
            def refill() -> None:
                """Start the lookahead async copy into the released ring slot."""
                self._copy(position + self.depth).start()

    def dot_position(
        self,
        x: jax.Array,
        position: int,
        guard: jax.Array | int,
    ) -> jax.Array:
        """Multiply ``x`` by one column chunk in FP32, then release its slot."""
        if self.axis != "columns":
            msg = "dot_position is for column streams"
            raise ValueError(msg)
        self.wait(position)
        result = dot(x, self.view(position)[...])
        self.release(position, guard)
        return result

    def dot(self, x: jax.Array, guard: jax.Array | int) -> jax.Array:
        """Compute ``x @ source`` in FP32 across all chunks of a column stream."""
        if self.axis != "columns":
            msg = "dot is for column streams"
            raise ValueError(msg)
        if self.slots is None:
            # A DMA wait is a scheduling boundary: with a slot per chunk,
            # resolve every transfer before the products so they overlap.
            self.wait_all()
        parts: list[jax.Array] = []
        for position in range(self.chunks):
            if self.slots is not None:
                self.wait(position)
            parts.append(dot(x, self.view(position)[...]))
            self.release(position, guard)
        return parts[0] if self.chunks == 1 else jnp.concatenate(parts, axis=1)


class KdaStateStream:
    """KDA state streamed through three rolling or four private local buffers.

    Normally buffer ``c % 3`` is reused only after its write has landed. B32's
    unrolled path uses four buffers: prime chunks 0/1 before projection, then
    start chunks 2/3 at recurrence entry. Final writes drain on exit. With
    slots, read each sequence's initial state directly from the bank;
    subsequent verifier rows are produced by the recurrence before use.
    """

    def __init__(
        self,
        history_hbm: PallasRefProtocol,
        state_hbm: PallasRefProtocol,
        history_output_hbm: PallasRefProtocol,
        state_output_hbm: PallasRefProtocol,
        history_buffers: PallasRefProtocol,
        state_buffers: PallasRefProtocol,
        read_semaphores: PallasRefProtocol,
        write_semaphores: PallasRefProtocol,
        *,
        chunk: int,
        chunks: int,
        slots: PallasRefProtocol | None = None,
        sequence_rows: int = 1,
    ) -> None:
        """Initialize KDA state-stream buffers and validate buffer count."""
        self.history_hbm: PallasRefProtocol = history_hbm
        self.state_hbm: PallasRefProtocol = state_hbm
        self.history_output_hbm: PallasRefProtocol = history_output_hbm
        self.state_output_hbm: PallasRefProtocol = state_output_hbm
        self.history_buffers: PallasRefProtocol = history_buffers
        self.state_buffers: PallasRefProtocol = state_buffers
        self.read_semaphores: PallasRefProtocol = read_semaphores
        self.write_semaphores: PallasRefProtocol = write_semaphores
        self.chunk: int = chunk
        self.chunks: int = chunks
        self.slots: PallasRefProtocol | None = slots
        self.sequence_rows: int = sequence_rows
        self.pending_writes: list[int | jax.Array] = []
        self.buffer_count: int = history_buffers.shape[0]
        self.four_buffers: bool = self.buffer_count == _UNROLLED_B32_STATE_BUFFERS
        if self.four_buffers and not (
            self.chunks == _UNROLLED_B32_STATE_BUFFERS
            and self.chunk == _SMALL_KDA_ROW_CHUNK
            and not KDA_STAGED
        ):
            msg = "Four-buffer KDA state stream requires four 8-row unstaged chunks"
            raise ValueError(msg)

    def _reads(self, index: int | jax.Array) -> tuple[AsyncCopyProtocol, ...]:
        """Build async read descriptors for state chunk ``index``."""
        rows = pl.ds(index * self.chunk, self.chunk)
        slot = index % self.buffer_count
        if self.slots is not None:
            slots_ref = self.slots
            return tuple(
                tpu.make_async_copy(
                    source.at[pl.ds(slots_ref[index * self.chunk + row], 1)],
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

    def _writes(
        self,
        index: int | jax.Array,
    ) -> tuple[AsyncCopyProtocol, AsyncCopyProtocol]:
        """Build async writeback descriptors for state chunk ``index``."""
        rows = pl.ds(index * self.chunk, self.chunk)
        slot = index % self.buffer_count
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
        """Start initial lookahead reads before projections begin."""
        for index in range(min(_STATE_LOOKAHEAD_CHUNKS, self.chunks)):
            for copy in self._reads(index):
                copy.start()

    def acquire(
        self,
        index: int | jax.Array,
    ) -> tuple[PallasRefProtocol, PallasRefProtocol]:
        """Wait for state chunk ``index`` and return its local buffer views."""
        if self.four_buffers and index == 0:
            for tail in range(_STATE_LOOKAHEAD_CHUNKS, self.chunks):
                for copy in self._reads(tail):
                    copy.start()
        for copy in self._reads(index):
            copy.wait()
        slot = index % self.buffer_count
        return self.history_buffers.at[slot], self.state_buffers.at[slot]

    def release(self, index: int | jax.Array) -> None:
        """Start writebacks for chunk ``index`` and advance rolling prefetch."""
        for copy in self._writes(index):
            copy.start()
        self.pending_writes.append(index)
        if self.four_buffers:
            return
        ahead = index + _STATE_LOOKAHEAD_CHUNKS
        if ahead < self.chunks:
            if ahead >= _ROLLING_STATE_BUFFERS:
                # Buffer ahead % 3 last held chunk ahead - 3.
                for copy in self._writes(ahead - _ROLLING_STATE_BUFFERS):
                    copy.wait()
                self.pending_writes.remove(ahead - _ROLLING_STATE_BUFFERS)
            for copy in self._reads(ahead):
                copy.start()

    def finish(self) -> None:
        """Wait for all outstanding unrolled chunk writebacks."""
        for index in list(self.pending_writes):
            for copy in self._writes(index):
                copy.wait()
        self.pending_writes.clear()

    # Traced-index variants for a rolled chunk loop: chunk c's writes are
    # drained when chunk c + 3 is read into its buffer, the last three in
    # finish_rolled.
    def acquire_rolled(
        self,
        index: int | jax.Array,
    ) -> tuple[PallasRefProtocol, PallasRefProtocol]:
        """Wait for rolled state chunk ``index`` and return its slot views."""
        for copy in self._reads(index):
            copy.wait()
        slot = index % _ROLLING_STATE_BUFFERS
        return self.history_buffers.at[slot], self.state_buffers.at[slot]

    def release_rolled(self, index: int | jax.Array) -> None:
        """Issue writebacks for rolled chunk ``index`` and refill lookahead."""
        for copy in self._writes(index):
            copy.start()
        ahead = index + _STATE_LOOKAHEAD_CHUNKS

        @pl.when(ahead < self.chunks)
        def refill() -> None:
            """Drain the reused slot's prior write and start lookahead reads."""

            @pl.when(ahead >= _ROLLING_STATE_BUFFERS)
            def drain() -> None:
                """Wait for the previous write occupying this rolling slot."""
                for copy in self._writes(ahead - _ROLLING_STATE_BUFFERS):
                    copy.wait()

            for copy in self._reads(ahead):
                copy.start()

    def finish_rolled(self) -> None:
        """Drain the final rolling-buffer writebacks after a rolled loop."""
        for index in range(max(0, self.chunks - _ROLLING_STATE_BUFFERS), self.chunks):
            for copy in self._writes(index):
                copy.wait()


class KdaStateStreamSlice:
    """One head shard's rows in a shared, flattened KDA state stream.

    The caller primes the parent once and drains it after all head shards. Local
    KDA calls use shard-local chunk indices and finish calls, while the parent
    keeps its lookahead and buffer ownership across shard boundaries.
    """

    def __init__(
        self,
        parent: KdaStateStream,
        *,
        first_chunk: int | jax.Array,
        chunks: int,
    ) -> None:
        """Bind a shard-local chunk window within ``parent``."""
        if isinstance(first_chunk, int) and (
            first_chunk < 0 or chunks < 1 or first_chunk + chunks > parent.chunks
        ):
            msg = "A state-stream slice must fit within its parent"
            raise ValueError(msg)
        self.parent: KdaStateStream = parent
        self.first_chunk: int | jax.Array = first_chunk
        self.chunks: int = chunks
        self.chunk: int = parent.chunk
        self.slots: PallasRefProtocol | None = parent.slots

    def acquire(
        self,
        index: int | jax.Array,
    ) -> tuple[PallasRefProtocol, PallasRefProtocol]:
        """Acquire shard-local chunk ``index`` from the parent stream."""
        with _phase("t1_kda_state_read_wait"):
            return self.parent.acquire(self.first_chunk + index)

    def release(self, index: int | jax.Array) -> None:
        """Release shard-local chunk ``index`` back to the parent stream."""
        self.parent.release(self.first_chunk + index)

    def acquire_rolled(
        self,
        index: int | jax.Array,
    ) -> tuple[PallasRefProtocol, PallasRefProtocol]:
        """Acquire rolled shard-local chunk ``index`` from the parent stream."""
        with _phase("t1_kda_state_read_wait"):
            return self.parent.acquire_rolled(self.first_chunk + index)

    def release_rolled(self, index: int | jax.Array) -> None:
        """Release rolled shard-local chunk ``index`` to the parent stream."""
        self.parent.release_rolled(self.first_chunk + index)

    def finish(self) -> None:
        """Leave final write draining to the owner of the parent stream."""
        # The next shard can already have reads in flight in these buffers.
        # Only the owner of the flattened stream can drain its final writes.

    def finish_rolled(self) -> None:
        """Leave final rolled write draining to the owner of the parent stream."""


def _shard_dot(
    stream: ChunkedWeightStream,
    x: jax.Array,
    shard: int,
    guard: jax.Array | int,
) -> jax.Array:
    """Multiply ``x`` by one held shard's columns of ``stream`` in FP32."""
    per_shard = stream.chunks // stream.shards
    parts = [
        stream.dot_position(x, shard * per_shard + index, guard)
        for index in range(per_shard)
    ]
    return parts[0] if per_shard == 1 else jnp.concatenate(parts, axis=1)


def _traced_dot_position(
    stream: ChunkedWeightStream,
    x_ref: PallasRefProtocol,
    position: jax.Array,
    guard: jax.Array,
    out_ref: PallasRefProtocol,
) -> None:
    """Multiply ``x_ref`` by a traced ring chunk of ``stream`` into ``out_ref``.

    Waits for the equal-sized ring chunk at ``position``, stores the BF16
    product into ``out_ref``, and refills the slot ``depth`` positions ahead.
    """
    if stream.slots is None or stream.order is not None or len(set(stream.sizes)) != 1:
        msg = "A traced stream position needs a ring of equal chunks"
        raise ValueError(msg)
    size = stream.sizes[0]
    slot = position % stream.depth

    def copy(
        at: jax.Array,
        destination: PallasRefProtocol,
        semaphore: PallasRefProtocol,
    ) -> AsyncCopyProtocol:
        """Create an async copy for ring chunk ``at`` into ``destination``."""
        source = stream.source.at[
            :, pl.ds(pl.multiple_of(at * size, VREG_LANE_COUNT), size)
        ]
        return tpu.make_async_copy(source, destination, semaphore)

    for index, destination in enumerate(stream.slots):

        @pl.when(slot == index)
        def consume(
            destination: PallasRefProtocol = destination,
            semaphore: PallasRefProtocol = stream.semaphores.at[stream.base + index],
        ) -> None:
            """Wait for the active ring slot, multiply, and refill lookahead."""
            copy(position, destination, semaphore).wait()
            out_ref[...] = dot(x_ref[...], destination[...]).astype(jnp.bfloat16)

            @pl.when(guard & (position + stream.depth < stream.chunks))
            def refill() -> None:
                """Start the lookahead DMA into the consumed ring slot."""
                copy(position + stream.depth, destination, semaphore).start()


@dataclasses.dataclass(frozen=True, slots=True)
class _KdaProjectionCopies:
    """Asynchronous DMA descriptors for KDA projection weights."""

    gate: AsyncCopyProtocol
    feature_gate: AsyncCopyProtocol
    input: AsyncCopyProtocol
    output: AsyncCopyProtocol


@dataclasses.dataclass(frozen=True, slots=True)
class KimiDeltaAttentionWeights:
    """KDA parameters and their persistent projection-copy resources."""

    gate_projection: TensorLike  # [hidden_size, 128]
    gate_projection_vmem: PallasRefProtocol  # [hidden_size, 128]
    input_projection: TensorLike  # [hidden_size, 1664]
    input_projection_vmem: PallasRefProtocol  # [hidden_size, 1664]
    feature_gate_projection: TensorLike  # [128, 384]
    feature_gate_projection_vmem: PallasRefProtocol  # [128, 384]
    # [384, hidden_size], or [768, hidden_size / 2] after paired-rank repartitioning.
    output_projection: TensorLike
    output_projection_vmem: (
        PallasRefProtocol  # [384, hidden_size] or [768, hidden_size / 2]
    )
    convolution_filter: TensorLike  # [4, 3, 3, 128]
    exp_log_decay_rate: TensorLike  # [3, 1]
    time_step_bias: TensorLike  # [3, 128]
    output_normalization: TensorLike  # [128] or [8, 128]
    projection_copy_semaphores: PallasRefProtocol  # [4]

    def __post_init__(self) -> None:
        """Register megakernel PyTree dataclasses on first instantiation."""
        ensure_dataclasses_registered()

    def make_projection_copies(self) -> _KdaProjectionCopies:
        """Create the asynchronous projection-weight copies."""
        return _KdaProjectionCopies(
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
    """Return the row count per KDA recurrence chunk above eight batch rows.

    Smaller batches run whole. TP32 B128 and above use four rows because
    eight-row chunks exceed local memory capacity although ``_vmem_budget``
    accepts them; attention DP's B128 peaks outside KDA and fits eight.
    """
    if batch_size < MAX_BATCH or (attention_dp > 1 and batch_size == MAX_BATCH):
        return _SMALL_KDA_ROW_CHUNK
    return _LARGE_KDA_ROW_CHUNK


class _RowIndexableProtocol(Protocol):
    """Protocol for batch tensors or lazy row-indexed chunk evaluators."""

    def __getitem__(self, rows: int | slice | jax.Array, /) -> jax.Array:
        """Return slice or lazily evaluated chunk for ``rows``."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class _KdaContext:
    """Validated inputs and tiling configuration for one KDA invocation."""

    hidden_ref: TensorLike
    output_ref: TensorLike
    updated_convolution_history_ref: PallasRefProtocol
    updated_recurrent_state_ref: PallasRefProtocol
    weights: KimiDeltaAttentionWeights
    convolution_history_ref: TensorLike
    recurrent_state_ref: TensorLike
    communication_workspace: AttentionCommunicationWorkspace
    epsilon: float
    decay_lower_bound: float
    weights_are_prefetched: bool
    token_scatter_output: bool
    sequence_rows: int
    wait_for_state: Callable[[], None] | None
    on_state_updated: Callable[[], None] | None
    streams: Mapping[str, ChunkedWeightStream]
    shard_index: int | jax.Array | None
    static_shard: int | None
    attended_out: PallasRefProtocol | None
    gate_features_vmem: TensorLike | None
    state_stream: KdaStateStream | KdaStateStreamSlice | None
    projected_rows: jax.Array | None
    first_shard: bool
    traced_shard: bool
    batch_size: int
    staged: bool
    projection_width: int
    rolled: bool
    row_chunk: int
    chunk_count: int


@dataclasses.dataclass(frozen=True, slots=True)
class _KdaScopedScratch:
    """Scoped local scratch buffers allocated for one KDA invocation."""

    reduction_vmem: PallasRefProtocol
    attended_vmem: PallasRefProtocol | None = None
    projected_vmem: PallasRefProtocol | None = None
    gate_vmem: PallasRefProtocol | None = None
    output_gate_vmem: PallasRefProtocol | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _KdaRecurrenceInputs:
    """Projected QKV, update-rate, decay, and optional output-gate values."""

    queries_keys_values: _RowIndexableProtocol
    update_rate: _RowIndexableProtocol
    decay: _RowIndexableProtocol
    output_gate: jax.Array | None = None


def _validate_kda_context(context: _KdaContext, hidden_size: int) -> None:
    """Validate KDA input, weight, and state shapes against the batch."""
    if (context.shard_index is None) != (context.attended_out is None):
        msg = "Per-shard KDA calls must take attended_out"
        raise ValueError(msg)
    state_batch = (
        context.convolution_history_ref.shape[0]
        if context.state_stream is not None and context.state_stream.slots is not None
        else context.batch_size
    )
    state_shape = (state_batch, KDA_HEADS, HEAD_DIM, HEAD_DIM)
    history_shape = (
        state_batch,
        CONV_TAPS,
        KDA_QKV_PARTS,
        CONV_HEAD_ROWS,
        HEAD_DIM,
    )
    weights = context.weights
    if (
        context.output_ref.shape != context.hidden_ref.shape
        or weights.gate_projection.shape != (hidden_size, HEAD_DIM)
        or weights.input_projection.shape[0] != hidden_size
        or weights.input_projection.shape[1] % _KDA_PROJECTION_WIDTH
        or (
            context.shard_index is None
            and weights.input_projection.shape[1] != _KDA_PROJECTION_WIDTH
        )
        or weights.exp_log_decay_rate.shape != (KDA_HEADS, 1)
        or context.convolution_history_ref.shape != history_shape
        or context.recurrent_state_ref.shape != state_shape
    ):
        msg = "Kimi Delta Attention inputs and state must match the input batch"
        raise ValueError(msg)
    if context.batch_size % context.sequence_rows:
        msg = "sequence_rows must divide the KDA batch"
        raise ValueError(msg)


def _start_kda_projection_copies(
    context: _KdaContext,
    weight_copies: _KdaProjectionCopies,
) -> None:
    """Start async weight copies when projections were not prefetched."""
    if context.weights_are_prefetched:
        return
    weight_copies.gate.start()
    weight_copies.feature_gate.start()
    if "input" in context.streams:
        context.streams["input"].start()
    else:
        weight_copies.input.start()
    if "output" in context.streams:
        context.streams["output"].start()
    else:
        weight_copies.output.start()


def _compute_kda_gate_features(
    context: _KdaContext,
    weight_copies: _KdaProjectionCopies,
) -> jax.Array:
    """Project or read shared BF16 gate features for the current KDA shard."""
    if context.first_shard:
        weight_copies.gate.wait()
    if context.first_shard or context.gate_features_vmem is None:
        gate_features = dot(
            context.hidden_ref[...], context.weights.gate_projection_vmem[...]
        ).astype(jnp.bfloat16)
        if context.gate_features_vmem is not None:
            cast("PallasRefProtocol", context.gate_features_vmem)[:, :HEAD_DIM] = (
                gate_features
            )
        return gate_features
    return context.gate_features_vmem[:, :HEAD_DIM]


def _project_kda_recurrent_gate(
    context: _KdaContext,
    scratch: _KdaScopedScratch,
    weight_copies: _KdaProjectionCopies,
) -> jax.Array | None:
    """Project recurrent gate values, staging into local scratch when enabled."""
    attention_width = KDA_HEADS * HEAD_DIM
    gate_features = _compute_kda_gate_features(context, weight_copies)
    if context.first_shard:
        weight_copies.feature_gate.wait()
    feature_gate_vmem = context.weights.feature_gate_projection_vmem
    if context.static_shard is not None:
        feature_gate_vmem = feature_gate_vmem.at[
            :, pl.ds(context.static_shard * attention_width, attention_width)
        ]
    raw_recurrent_gate: jax.Array | None = None
    if context.rolled or context.staged:
        gate_buf = scratch.gate_vmem
        if gate_buf is None:
            msg = "Staged KDA projections require gate scratch storage"
            raise ValueError(msg)
        if (
            context.rolled
            and context.traced_shard
            and feature_gate_vmem.shape[1] > DP2_GROUPS * attention_width
        ):
            raw_gate = dot(gate_features, feature_gate_vmem[...])
            selected = raw_gate[:, :attention_width]
            for held in range(1, feature_gate_vmem.shape[1] // attention_width):
                selected = jnp.where(
                    context.shard_index == held,
                    raw_gate[:, held * attention_width : (held + 1) * attention_width],
                    selected,
                )
            gate_buf[...] = selected.astype(jnp.bfloat16).reshape(
                context.chunk_count, context.row_chunk, -1
            )
        elif context.rolled:
            gate_buf[...] = (
                dot(gate_features, feature_gate_vmem[...])
                .astype(jnp.bfloat16)
                .reshape(context.chunk_count, context.row_chunk, -1)
            )
        else:
            gate_buf[...] = dot(gate_features, feature_gate_vmem[...]).astype(
                jnp.bfloat16
            )
        return raw_recurrent_gate
    raw_recurrent_gate = dot(gate_features, feature_gate_vmem[...])
    if context.traced_shard:
        raw_recurrent_gate = jnp.where(
            context.shard_index == 0,
            raw_recurrent_gate[:, :attention_width],
            raw_recurrent_gate[:, attention_width:],
        )
    return (
        raw_recurrent_gate.astype(jnp.bfloat16)
        .reshape(context.batch_size, KDA_HEADS, HEAD_DIM)
        .astype(jnp.float32)
    )


def _compute_kda_input_projection(
    context: _KdaContext,
    weight_copies: _KdaProjectionCopies,
) -> jax.Array:
    """Compute or return pre-projected BF16 KDA input columns."""
    if context.projected_rows is not None:
        return context.projected_rows
    if context.shard_index is not None:
        if context.static_shard is None:
            msg = "Traced KDA shard calls must supply projected_rows"
            raise ValueError(msg)
        return _shard_dot(
            context.streams["input"],
            context.hidden_ref[...],
            context.static_shard,
            context.communication_workspace.reduction_phase[0] >= 0,
        ).astype(jnp.bfloat16)
    if "input" in context.streams:
        return (
            context.streams["input"]
            .dot(
                context.hidden_ref[...],
                context.communication_workspace.reduction_phase[0] >= 0,
            )
            .astype(jnp.bfloat16)
        )
    weight_copies.input.wait()
    return dot(
        context.hidden_ref[...], context.weights.input_projection_vmem[...]
    ).astype(jnp.bfloat16)


def _stage_rolled_kda_inputs(
    context: _KdaContext,
    scratch: _KdaScopedScratch,
    projected: jax.Array,
) -> _KdaRecurrenceInputs:
    """Stage chunk-major projected inputs and build rolled row evaluators."""
    attention_width = KDA_HEADS * HEAD_DIM
    projected_buf = scratch.projected_vmem
    gate_buf = scratch.gate_vmem
    output_gate_buf = scratch.output_gate_vmem
    if projected_buf is None or gate_buf is None or output_gate_buf is None:
        msg = "Rolled KDA projections require projected and gate scratch storage"
        raise ValueError(msg)
    proj_ref: PallasRefProtocol = projected_buf
    gate_ref: PallasRefProtocol = gate_buf
    proj_ref[...] = projected.reshape(context.chunk_count, context.row_chunk, -1)
    output_gate_buf[...] = projected[:, 3 * attention_width : 4 * attention_width]

    def qkv_chunk(index: int | slice | jax.Array) -> jax.Array:
        """Load one rolled chunk's QKV slice."""
        return proj_ref[index][:, : 3 * attention_width].reshape(
            context.row_chunk, KDA_QKV_PARTS, KDA_HEADS, HEAD_DIM
        )

    def update_rate_chunk(index: int | slice | jax.Array) -> jax.Array:
        """Load one rolled chunk's FP32 update-rate slice."""
        return (
            proj_ref[index][:, 4 * attention_width : 4 * attention_width + KDA_HEADS]
            .reshape(context.row_chunk, KDA_HEADS, 1)
            .astype(jnp.float32)
        )

    def decay_chunk(index: int | slice | jax.Array) -> jax.Array:
        """Compute one rolled chunk's log-space decay factors."""
        return context.decay_lower_bound * jax.nn.sigmoid(
            jnp.exp(context.weights.exp_log_decay_rate[...])[None]
            * (
                gate_ref[index]
                .reshape(context.row_chunk, KDA_HEADS, HEAD_DIM)
                .astype(jnp.float32)
                + context.weights.time_step_bias[...][None]
            )
        )

    return _KdaRecurrenceInputs(
        queries_keys_values=_RowFunction(qkv_chunk),
        update_rate=_RowFunction(update_rate_chunk),
        decay=_RowFunction(decay_chunk),
    )


def _stage_unrolled_kda_inputs(
    context: _KdaContext,
    scratch: _KdaScopedScratch,
    projected: jax.Array,
) -> _KdaRecurrenceInputs:
    """Stage batch-major projected inputs and build slice row evaluators."""
    attention_width = KDA_HEADS * HEAD_DIM
    projected_buf = scratch.projected_vmem
    gate_buf = scratch.gate_vmem
    if projected_buf is None or gate_buf is None:
        msg = "Staged KDA projections require projected and gate scratch storage"
        raise ValueError(msg)
    proj_ref: PallasRefProtocol = projected_buf
    gate_ref: PallasRefProtocol = gate_buf
    proj_ref[...] = projected

    def qkv_rows(rows: int | slice | jax.Array) -> jax.Array:
        """Load QKV slice for ``rows``."""
        part = proj_ref[rows, : 3 * attention_width]
        return part.reshape(part.shape[0], KDA_QKV_PARTS, KDA_HEADS, HEAD_DIM)

    def update_rate_rows(rows: int | slice | jax.Array) -> jax.Array:
        """Load FP32 update-rate slice for ``rows``."""
        part = proj_ref[rows, 4 * attention_width : 4 * attention_width + KDA_HEADS]
        return part.reshape(part.shape[0], KDA_HEADS, 1).astype(jnp.float32)

    def decay_rows(rows: int | slice | jax.Array) -> jax.Array:
        """Compute log-space decay factors for ``rows``."""
        part = gate_ref[rows]
        return context.decay_lower_bound * jax.nn.sigmoid(
            jnp.exp(context.weights.exp_log_decay_rate[...])[None]
            * (
                part.reshape(part.shape[0], KDA_HEADS, HEAD_DIM).astype(jnp.float32)
                + context.weights.time_step_bias[...][None]
            )
        )

    return _KdaRecurrenceInputs(
        queries_keys_values=_RowFunction(qkv_rows),
        update_rate=_RowFunction(update_rate_rows),
        decay=_RowFunction(decay_rows),
    )


def _project_kda_inputs(
    context: _KdaContext,
    scratch: _KdaScopedScratch,
    weight_copies: _KdaProjectionCopies,
    raw_recurrent_gate: jax.Array | None,
) -> _KdaRecurrenceInputs:
    """Compute KDA input projections and wait for state reads when scheduled."""
    projected = _compute_kda_input_projection(context, weight_copies)
    if context.rolled:
        staged_inputs = _stage_rolled_kda_inputs(context, scratch, projected)
        if context.wait_for_state is not None:
            context.wait_for_state()
        return staged_inputs
    if context.staged:
        staged_inputs = _stage_unrolled_kda_inputs(context, scratch, projected)
        if context.wait_for_state is not None:
            context.wait_for_state()
        return staged_inputs
    if raw_recurrent_gate is None:
        msg = "Unstaged KDA path requires a computed recurrent gate"
        raise ValueError(msg)
    attention_width = KDA_HEADS * HEAD_DIM
    queries_keys_values = projected[:, : 3 * attention_width].reshape(
        context.batch_size, KDA_QKV_PARTS, KDA_HEADS, HEAD_DIM
    )
    projected_tail = projected[:, 3 * attention_width :]
    output_gate: jax.Array | None = projected_tail[:, :attention_width].reshape(
        context.batch_size, KDA_HEADS, HEAD_DIM
    )
    update_rate = (
        projected_tail[:, attention_width : attention_width + KDA_HEADS]
        .reshape(context.batch_size, KDA_HEADS, 1)
        .astype(jnp.float32)
    )
    if context.wait_for_state is not None:
        context.wait_for_state()
    decay = context.decay_lower_bound * jax.nn.sigmoid(
        jnp.exp(context.weights.exp_log_decay_rate[...])[None]
        * (raw_recurrent_gate + context.weights.time_step_bias[...][None])
    )
    return _KdaRecurrenceInputs(
        queries_keys_values=queries_keys_values,
        update_rate=update_rate,
        decay=decay,
        output_gate=output_gate,
    )


def _kda_recurrence_step(
    context: _KdaContext,
    inputs: _KdaRecurrenceInputs,
    rows: slice | jax.Array,
    history_ref: PallasRefProtocol | None = None,
    state_ref: PallasRefProtocol | None = None,
) -> jax.Array:
    """Advance single-token KDA convolution and recurrent state for ``rows``."""
    # ``history_ref``/``state_ref``: a streamed chunk's buffers (updated in
    # place), else the whole-batch state refs.
    convolution_history = (
        context.convolution_history_ref[rows, :, :, :KDA_HEADS]
        if history_ref is None
        else history_ref[:, :, :, :KDA_HEADS]
    )
    recurrent_state = (
        context.recurrent_state_ref[rows] if state_ref is None else state_ref[...]
    ).astype(jnp.float32)
    convolution_window = jnp.concatenate(
        (convolution_history, inputs.queries_keys_values[rows][:, None]), axis=1
    )
    filtered_queries_keys_values = jax.nn.silu(
        jnp.sum(
            convolution_window.astype(jnp.float32)
            * context.weights.convolution_filter[...].astype(jnp.float32)[None],
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
        * HEAD_DIM**-0.5
    )
    key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    next_recurrent_state = recurrent_state * jnp.exp(inputs.decay[rows])[..., None]
    predicted_value = jnp.sum(key[..., None] * next_recurrent_state, axis=2)
    state_update = jax.nn.sigmoid(inputs.update_rate[rows]) * (value - predicted_value)
    next_recurrent_state += key[..., None] * state_update[..., None, :]
    attended_rows = jnp.sum(query[..., None] * next_recurrent_state, axis=2)
    if history_ref is None or state_ref is None:
        context.updated_convolution_history_ref[rows, :, :, :KDA_HEADS] = (
            convolution_window[:, 1:]
        )
        context.updated_recurrent_state_ref[rows] = next_recurrent_state.astype(
            context.updated_recurrent_state_ref.dtype
        )
    else:
        history_ref[:, :, :, :KDA_HEADS] = convolution_window[:, 1:]
        state_ref[...] = next_recurrent_state.astype(state_ref.dtype)
    return attended_rows


def _run_kda_whole_batch_recurrence(
    context: _KdaContext,
    inputs: _KdaRecurrenceInputs,
) -> jax.Array:
    """Execute whole-batch KDA recurrence when row tiling is not required."""
    stream = context.state_stream
    if stream is None:
        return _kda_recurrence_step(context, inputs, slice(None))
    # DP4 B32 has one eight-row state chunk per head shard. Its refs are
    # HBM refs, so consume the prefetched stream even though the recurrence
    # itself needs no row tiling.
    if not (stream.chunks == 1 and stream.chunk == context.batch_size):
        msg = "Whole-batch KDA recurrence requires a single batch-sized state chunk"
        raise ValueError(msg)
    history_chunk, state_chunk = stream.acquire(0)
    attended = _kda_recurrence_step(
        context, inputs, slice(None), history_chunk, state_chunk
    )
    stream.release(0)
    stream.finish()
    return attended


def _run_kda_single_row_recurrence(
    context: _KdaContext,
    scratch: _KdaScopedScratch,
    inputs: _KdaRecurrenceInputs,
) -> jax.Array:
    """Execute single-token-per-sequence KDA recurrence across the batch."""
    if context.batch_size <= _MAX_WHOLE_KDA_BATCH:
        return _run_kda_whole_batch_recurrence(context, inputs)
    stream = context.state_stream
    # Above eight rows the whole-batch state value (3 MB at B16) spills; run
    # row chunks in separate regions so only one chunk's state is live.
    attended_buf = scratch.attended_vmem
    if attended_buf is None:
        msg = "Chunked KDA recurrence requires attended scratch storage"
        raise ValueError(msg)
    chunk = context.row_chunk
    if context.rolled:
        if stream is None:
            msg = "Rolled KDA recurrence requires a state stream"
            raise ValueError(msg)

        @pl.loop(0, context.chunk_count)
        def run_rolled_chunk(index: jax.Array) -> None:
            """Run one rolled recurrence chunk and store attended rows."""
            history_chunk, state_chunk = stream.acquire_rolled(index)
            attended_buf[pl.ds(pl.multiple_of(index * chunk, chunk), chunk)] = (
                _kda_recurrence_step(context, inputs, index, history_chunk, state_chunk)
            )
            stream.release_rolled(index)

        stream.finish_rolled()
    else:
        for index, first in enumerate(range(0, context.batch_size, chunk)):
            # A runtime condition (never false) so the chunk is its own region.
            @pl.when(context.communication_workspace.reduction_phase[0] >= 0)
            def run_chunk(index: int = index, first: int = first) -> None:
                """Run one unrolled recurrence chunk in an isolated region."""
                if stream is None:
                    attended_buf[first : first + chunk] = _kda_recurrence_step(
                        context, inputs, slice(first, first + chunk)
                    )
                else:
                    history_chunk, state_chunk = stream.acquire(index)
                    attended_buf[first : first + chunk] = _kda_recurrence_step(
                        context,
                        inputs,
                        slice(first, first + chunk),
                        history_chunk,
                        state_chunk,
                    )
                    stream.release(index)

        if stream is not None and context.batch_size != _DEFERRED_STATE_DRAIN_BATCH:
            stream.finish()
    return attended_buf[...]


def _run_kda_sequence_rows(
    context: _KdaContext,
    inputs: _KdaRecurrenceInputs,
    convolution_filter: jax.Array,
    initial_state: tuple[int, jax.Array, jax.Array],
    store: Callable[[int, jax.Array, jax.Array], None],
) -> list[jax.Array]:
    """Advance one multi-token sequence from its initial convolution and state."""
    first, history, initial_recurrent = initial_state
    window = jnp.concatenate(
        (history, inputs.queries_keys_values[first : first + context.sequence_rows]),
        axis=0,
    )
    state = initial_recurrent.astype(jnp.float32)
    rows: list[jax.Array] = []
    for offset in range(context.sequence_rows):
        row = first + offset
        filtered = jax.nn.silu(
            jnp.sum(
                window[offset : offset + _CONV_WINDOW_TAPS].astype(jnp.float32)
                * convolution_filter,
                axis=0,
            )
        ).astype(jnp.bfloat16)
        query = filtered[0].astype(jnp.float32)
        key = filtered[1].astype(jnp.float32)
        value = filtered[2].astype(jnp.float32)
        query *= (
            jax.lax.rsqrt(jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
            * HEAD_DIM**-0.5
        )
        key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
        state = state * jnp.exp(inputs.decay[row])[..., None]
        predicted_value = jnp.sum(key[..., None] * state, axis=1)
        state_update = jax.nn.sigmoid(inputs.update_rate[row]) * (
            value - predicted_value
        )
        state = state + key[..., None] * state_update[:, None, :]
        rows.append(jnp.sum(query[..., None] * state, axis=1)[None])
        # Match separate decode steps: the current output uses
        # FP32 updated state, but the next token sees stored state.
        saved_state = state.astype(context.recurrent_state_ref.dtype)
        store(offset, window[offset + 1 : offset + _CONV_WINDOW_TAPS], saved_state)
        state = saved_state.astype(jnp.float32)
    return rows


def _execute_kda_sequence_chunk(
    context: _KdaContext,
    inputs: _KdaRecurrenceInputs,
    convolution_filter: jax.Array,
    stream_and_attended: tuple[KdaStateStream | KdaStateStreamSlice, PallasRefProtocol],
    chunk_position: tuple[int, int],
) -> None:
    """Acquire, step, and release one streamed multi-token sequence chunk."""
    stream, attended_buf = stream_and_attended
    index, base = chunk_position
    chunk = stream.chunk
    history_chunk, state_chunk = stream.acquire(index)
    rows: list[jax.Array] = []
    for local in range(0, chunk, context.sequence_rows):

        def store(
            offset: int,
            history: jax.Array,
            state: jax.Array,
            local: int = local,
        ) -> None:
            """Write updated sequence states into the chunk buffers."""
            history_chunk[local + offset, :, :, :KDA_HEADS] = history
            state_chunk[local + offset] = state

        rows += _run_kda_sequence_rows(
            context,
            inputs,
            convolution_filter,
            (
                base + local,
                history_chunk[local, :, :, :KDA_HEADS],
                state_chunk[local],
            ),
            store,
        )
    attended_buf[base : base + chunk] = jnp.concatenate(rows, axis=0)
    stream.release(index)


def _run_kda_multi_row_recurrence(
    context: _KdaContext,
    scratch: _KdaScopedScratch,
    inputs: _KdaRecurrenceInputs,
) -> jax.Array:
    """Execute chained multi-token speculative verification KDA recurrence."""
    convolution_filter = context.weights.convolution_filter[...].astype(jnp.float32)
    stream = context.state_stream
    if stream is None:
        attended_rows: list[jax.Array] = []
        for first in range(0, context.batch_size, context.sequence_rows):

            def store(
                offset: int,
                history: jax.Array,
                state: jax.Array,
                first: int = first,
            ) -> None:
                """Write updated sequence states into the output references."""
                context.updated_convolution_history_ref[
                    first + offset, :, :, :KDA_HEADS
                ] = history
                context.updated_recurrent_state_ref[first + offset] = state

            attended_rows += _run_kda_sequence_rows(
                context,
                inputs,
                convolution_filter,
                (
                    first,
                    context.convolution_history_ref[first, :, :, :KDA_HEADS],
                    context.recurrent_state_ref[first],
                ),
                store,
            )
        return jnp.concatenate(attended_rows, axis=0)
    # Streamed states (above 16 rows): each row chunk holds whole sequences,
    # each starting from its first row's state; every row's post-states go into
    # the chunk buffers, written back in place on release.
    attended_buf = scratch.attended_vmem
    if attended_buf is None:
        msg = "Streamed multi-row KDA recurrence requires attended scratch storage"
        raise ValueError(msg)
    chunk = stream.chunk
    if chunk % context.sequence_rows:
        msg = "Streamed KDA chunks must hold whole sequences"
        raise ValueError(msg)
    for index, base in enumerate(range(0, context.batch_size, chunk)):
        # A runtime condition (never false) so the chunk is its own region.
        @pl.when(context.communication_workspace.reduction_phase[0] >= 0)
        def run_sequence_chunk(index: int = index, base: int = base) -> None:
            """Run one streamed sequence chunk in an isolated region."""
            _execute_kda_sequence_chunk(
                context,
                inputs,
                convolution_filter,
                (stream, attended_buf),
                (index, base),
            )

    if context.batch_size != _DEFERRED_STATE_DRAIN_BATCH:
        # B32 drains after the output work (below).
        stream.finish()
    return attended_buf[...]


def _apply_kda_output_gate(
    context: _KdaContext,
    scratch: _KdaScopedScratch,
    inputs: _KdaRecurrenceInputs,
    attended: jax.Array,
) -> jax.Array:
    """Normalize attended features, apply sigmoid output gate, and signal state."""
    attention_width = KDA_HEADS * HEAD_DIM
    attended = attended.astype(jnp.bfloat16).astype(jnp.float32)
    attended *= jax.lax.rsqrt(
        jnp.mean(attended * attended, axis=-1, keepdims=True) + context.epsilon
    )
    normalization = (
        context.weights.output_normalization[...]
        if len(context.weights.output_normalization.shape) == 1
        else context.weights.output_normalization[0]
    )
    output_gate: jax.Array | None = inputs.output_gate
    if context.rolled:
        output_gate_buf = scratch.output_gate_vmem
        if output_gate_buf is None:
            msg = "Rolled KDA output gating requires output-gate scratch storage"
            raise ValueError(msg)
        output_gate = output_gate_buf[...].reshape(
            context.batch_size, KDA_HEADS, HEAD_DIM
        )
    elif context.staged:
        projected_buf = scratch.projected_vmem
        if projected_buf is None:
            msg = "Staged KDA output gating requires projected scratch storage"
            raise ValueError(msg)
        output_gate = projected_buf[
            :, 3 * attention_width : 4 * attention_width
        ].reshape(context.batch_size, KDA_HEADS, HEAD_DIM)
    if output_gate is None:
        msg = "KDA output gating requires an output gate tensor"
        raise ValueError(msg)
    gated = (attended * normalization).astype(jnp.bfloat16) * jax.nn.sigmoid(
        output_gate.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    if context.on_state_updated is not None:
        context.on_state_updated()
    return gated


def _finish_kda_output(
    context: _KdaContext,
    scratch: _KdaScopedScratch,
    weight_copies: _KdaProjectionCopies,
    attended: jax.Array,
) -> None:
    """Store per-shard attended features or project and reduce across ranks."""
    attention_width = KDA_HEADS * HEAD_DIM
    if context.attended_out is not None:
        context.attended_out[...] = attended.reshape(
            context.batch_size, attention_width
        )
        return
    output_stream = context.streams.get("output")
    if output_stream is None:
        weight_copies.output.wait()
    elif output_stream.chunks == 1:
        output_stream.wait_all()
        output_stream = None
    with _phase("p2_attn_project_reduce"):
        _project_attention_2d(
            attended.reshape(context.batch_size, attention_width),
            context.weights.output_projection_vmem,
            cast("PallasRefProtocol", context.output_ref),
            scratch.reduction_vmem,
            context.communication_workspace.shared_local_reduction_vmem,
            context.communication_workspace.shared_host_reduction_vmem,
            context.communication_workspace.attention_features_vmem,
            context.communication_workspace.attention_output_vmem,
            context.communication_workspace.send_semaphores,
            context.communication_workspace.receive_semaphores,
            context.communication_workspace.reduction_phase,
            dot,
            token_scatter_output=context.token_scatter_output,
            weight_stream=output_stream,
            staging=context.communication_workspace.projection_staging_vmem,
            gather_wire=context.communication_workspace.attention_gather_vmem,
        )


def _run_kda_scoped(context: _KdaContext, scratch: _KdaScopedScratch) -> None:
    """Execute KDA projections, recurrence, output gating, and output projection."""
    scope = phase_start("q3_kda_projections")
    weight_copies = context.weights.make_projection_copies()
    _start_kda_projection_copies(context, weight_copies)
    raw_recurrent_gate = _project_kda_recurrent_gate(context, scratch, weight_copies)
    recurrence_inputs = _project_kda_inputs(
        context, scratch, weight_copies, raw_recurrent_gate
    )
    if context.sequence_rows == 1:
        phase_end(scope)
        scope = phase_start("q4_kda_recurrence")
        attended = _run_kda_single_row_recurrence(context, scratch, recurrence_inputs)
    else:
        attended = _run_kda_multi_row_recurrence(context, scratch, recurrence_inputs)
    phase_end(scope)
    scope = phase_start("q5_kda_output_gate")
    gated = _apply_kda_output_gate(context, scratch, recurrence_inputs, attended)
    phase_end(scope)
    _finish_kda_output(context, scratch, weight_copies, gated)


def _kda_scoped_specs(context: _KdaContext) -> tuple[pl.MemoryRef, ...]:
    """Build scoped local scratch specifications for ``kda``."""
    attention_width = KDA_HEADS * HEAD_DIM
    reduction_width = (
        _HALF_HIDDEN_REDUCTION_WIDTH
        if context.batch_size > _WIDE_REDUCTION_MAX_BATCH
        else _WIDE_REDUCTION_WIDTH
    )
    attended_specs: tuple[pl.MemoryRef, ...] = (
        (tpu.VMEM((context.batch_size, KDA_HEADS, HEAD_DIM), jnp.float32),)
        if (
            context.batch_size > _MAX_WHOLE_KDA_BATCH
            and (context.sequence_rows == 1 or context.state_stream is not None)
        )
        else ()
    )
    if context.rolled:
        staging_specs: tuple[pl.MemoryRef, ...] = (
            tpu.VMEM(
                (context.chunk_count, context.row_chunk, context.projection_width),
                jnp.bfloat16,
            ),
            tpu.VMEM(
                (context.chunk_count, context.row_chunk, attention_width),
                jnp.bfloat16,
            ),
            tpu.VMEM((context.batch_size, attention_width), jnp.bfloat16),
        )
    elif context.staged:
        staging_specs = (
            tpu.VMEM((context.batch_size, context.projection_width), jnp.bfloat16),
            tpu.VMEM((context.batch_size, attention_width), jnp.bfloat16),
        )
    else:
        staging_specs = ()
    return (
        tpu.VMEM((context.batch_size, reduction_width), jnp.float32),
        *attended_specs,
        *staging_specs,
    )


def kda(
    hidden_ref: TensorLike,  # [batch_size, hidden_size]
    output_ref: TensorLike,  # [batch_size, hidden_size]
    updated_convolution_history_ref: PallasRefProtocol,  # [batch_size, 3, 3, 4, 128]
    updated_recurrent_state_ref: PallasRefProtocol,  # [batch_size, 3, 128, 128]
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
    wait_for_state: Callable[[], None] | None = None,
    on_state_updated: Callable[[], None] | None = None,
    streams: Mapping[str, ChunkedWeightStream] | None = None,
    shard_index: int | jax.Array | None = None,
    attended_out: PallasRefProtocol | None = None,
    gate_features_vmem: TensorLike | None = None,
    state_stream: KdaStateStream | KdaStateStreamSlice | None = None,
    stage_projections: bool = False,
    projected_rows: jax.Array | None = None,
    first_call: bool | None = None,
) -> None:
    """Execute one Kimi Delta Attention layer inside a Pallas program.

    ``state_stream`` (above eight rows; see :class:`KdaStateStream`) supplies
    each recurrence chunk's state from HBM instead of whole-batch state refs.

    ``streams`` optionally maps ``"input"`` and ``"output"`` to
    :class:`ChunkedWeightStream` instances that own the copies of those
    projections.

    Attention DP runs one call per held TP32 shard (``shard_index``);
    ``attended_out`` (``[batch, 384]``) receives its gated features for the
    caller's output projection, and ``gate_features_vmem`` (at least
    ``[batch, 128]`` BF16) shares the gate features between shards.
    ``stage_projections`` stages a DP shard's projections above eight rows
    (rolled over a state stream's tiles). ``projected_rows`` (``[batch, 1664]``
    BF16) is the caller's input projection for the DP2 B2/B4 shard swap, which
    passes a traced ``shard_index`` and ``first_call``.
    """
    resolved_streams = streams or {}
    static_shard = shard_index if isinstance(shard_index, int) else None
    traced_shard = shard_index is not None and static_shard is None
    first_shard = (
        (shard_index is None or static_shard == 0) if first_call is None else first_call
    )
    batch_size, hidden_size = hidden_ref.shape
    # Large DP batches keep each shard's projections in local scratch and reload
    # only a recurrence tile, instead of keeping projected/decay vectors live
    # across every state-stream region. DP1 stages only with
    # KernelConfig.kda_staged.
    staged = (
        (stage_projections or (KDA_STAGED and shard_index is None))
        and batch_size > _MAX_WHOLE_KDA_BATCH
        and sequence_rows == 1
    )
    projection_width = (
        _KDA_PROJECTION_WIDTH
        if shard_index is not None
        else weights.input_projection.shape[1]
    )
    rolled = staged and state_stream is not None
    row_chunk = (
        state_stream.chunk if state_stream is not None else _kda_row_chunk(batch_size)
    )
    context = _KdaContext(
        hidden_ref=hidden_ref,
        output_ref=output_ref,
        updated_convolution_history_ref=updated_convolution_history_ref,
        updated_recurrent_state_ref=updated_recurrent_state_ref,
        weights=weights,
        convolution_history_ref=convolution_history_ref,
        recurrent_state_ref=recurrent_state_ref,
        communication_workspace=communication_workspace,
        epsilon=epsilon,
        decay_lower_bound=decay_lower_bound,
        weights_are_prefetched=weights_are_prefetched,
        token_scatter_output=token_scatter_output,
        sequence_rows=sequence_rows,
        wait_for_state=wait_for_state,
        on_state_updated=on_state_updated,
        streams=resolved_streams,
        shard_index=shard_index,
        static_shard=static_shard,
        attended_out=attended_out,
        gate_features_vmem=gate_features_vmem,
        state_stream=state_stream,
        projected_rows=projected_rows,
        first_shard=first_shard,
        traced_shard=traced_shard,
        batch_size=batch_size,
        staged=staged,
        projection_width=projection_width,
        rolled=rolled,
        row_chunk=row_chunk,
        chunk_count=batch_size // row_chunk,
    )
    _validate_kda_context(context, hidden_size)
    if synchronize_devices:
        collectives32.barrier()
        communication_workspace.reduction_phase[0] = 0

    def run(
        reduction_vmem: PallasRefProtocol,
        attended_vmem: PallasRefProtocol | None = None,
        projected_vmem: PallasRefProtocol | None = None,
        gate_vmem: PallasRefProtocol | None = None,
        output_gate_vmem: PallasRefProtocol | None = None,
    ) -> None:
        """Bind scoped buffers and run the KDA layer body."""
        scratch = _KdaScopedScratch(
            reduction_vmem=reduction_vmem,
            attended_vmem=attended_vmem,
            projected_vmem=projected_vmem,
            gate_vmem=gate_vmem,
            output_gate_vmem=output_gate_vmem,
        )
        _run_kda_scoped(context, scratch)

    pl.run_scoped(run, *_kda_scoped_specs(context))
    if (
        state_stream is not None
        and not rolled
        and batch_size == _DEFERRED_STATE_DRAIN_BATCH
    ):
        state_stream.finish()


_MLA_KV_LORA_RANK: int = 512
_MLA_ROPE_DIM: int = 64
_MLA_ROPE_END: int = _MLA_KV_LORA_RANK + _MLA_ROPE_DIM
_MLA_QUERY_UP_WIDTH: int = mla_dcp.HEADS_PER_RANK * _MLA_KEY_HANDOFF_LANES
_MLA_KV_UP_WIDTH: int = mla_dcp.HEADS_PER_RANK * _MLA_KEY_HANDOFF_LANES
_MLA_ATTN_SCORE_DIM: int = HEAD_DIM + _MLA_ROPE_DIM
_MLA_ATTN_SCORE_SCALE: float = _MLA_ATTN_SCORE_DIM**-0.5
_MIN_MLA_CONTEXT_TOKENS: int = 384

_B16_BATCH_SIZE: int = 16
_HOSTS_IN_SLICE: int = TP_SIZE // mla_dcp.HOST_LANES
_TP16_PAIRS_TOTAL: int = TP_SIZE // _RANK_PAIR_SIZE
_ATTENTION_PAIR_GATHER_SEM_BASE: int = 10
_ATTENTION_OUTPUT_GATHER_SEM_BASE: int = 12
_ATTENTION_ALL_REDUCE_SEM_BASE: int = 14
_ATTENTION_HOST_SEM_BASE: int = 3
_ATTENTION_PAIR_SWAP_SEM: int = 6
_DP_SINGLE_OWNER_SEM: int = 12
_DP_SINGLE_OWNER_PEERS: int = _TP16_PAIRS_TOTAL - 1
_SUPPORTED_DP_DEGREES: tuple[int, int] = (2, 4)
_SUPPORTED_DP_OWNERS: tuple[int, int] = (1, 4)
_DIRECT_GATHER_MAX_ROWS: int = 2
_DP_MOE_LOCAL_SEM: int = 7
_DP_MOE_DIRECT_GROUP_SEM: int = 8
_DP_MOE_FORWARD_SEM: int = 9
_DP_MOE_GROUP_SEMS: tuple[int, int, int] = (8, 10, 11)

_DENSE_SCATTER_ROUND_ROWS: int = 8
_DENSE_HOST_SEM_BASE: int = 6
_ALL_REDUCE_LOCAL_SEM_BASE: int = -1
_ALL_REDUCE_HOST_SEM_BASE: int = 6
_SP8_OWNER_ROWS: int = 8
_SP8_PAIR_ROWS: int = 16
_MIN_SP8_LOCAL_SLOTS: int = 32
_DP2_SP8_WIRE_SLOTS: int = 16

_RESIDUAL_MIXTURE_SLOTS: int = 9
_PREFIX_RESIDUAL_SLOT: int = 8

_DEFAULT_COMPACT_ROWS: int = 8
_COMPACT_EXPERT_MIN_BATCH: int = 32
_SHARED_SHARDS_BATCH_THRESHOLD: int = 32
_COMPACT_OVERFLOW_BATCHES: tuple[int, ...] = (32, 64, 96, 128)

_SUBVECTOR_ALIGN_BYTES: int = 512
_WORD_BYTES: int = 4
_MIN_2D_RANK: int = 2
_REPORT_ENTRY_MIN_BYTES: int = 64 * _KIB_BYTES
_MOE_ROUTING_WIDTH: int = 1024
_MOE_SHARED_SEMS: int = 10
_MOE_SCATTER_OUTPUT_SEMS: int = 7
_MOE_DEFAULT_OUTPUT_SEMS: int = 2
_MOE_LATENT_SEMS: int = 2
_MOE_SHARD_ROUTING_OWNERS: int = 4
_MOE_SCATTER_EXTRA_SLOTS: int = 8


@dataclasses.dataclass(frozen=True, slots=True)
class MultiHeadLatentAttentionWeights:
    """MLA parameters and persistent projection-copy resources."""

    query_down_projection: TensorLike  # [hidden_size, 1536]
    query_down_projection_vmem: PallasRefProtocol  # [hidden_size, 1536]
    key_value_down_projection: TensorLike  # [hidden_size, 640]
    key_value_down_projection_vmem: PallasRefProtocol  # [hidden_size, 640]
    query_up_projection: TensorLike  # [1536, 768]
    query_up_projection_vmem: PallasRefProtocol  # [1536, 768]
    key_value_up_projection: TensorLike  # [512, 768]
    key_value_up_projection_vmem: PallasRefProtocol  # [512, 768]
    output_gate_projection: TensorLike  # [hidden_size, 384]
    output_gate_projection_vmem: PallasRefProtocol  # [hidden_size, 384]
    # [384, hidden_size], or [768, hidden_size / 2] after paired-rank repartitioning.
    output_projection: TensorLike
    # Output projection local buffer: [384, hidden_size] or [768, hidden_size / 2].
    output_projection_vmem: PallasRefProtocol
    query_normalization: TensorLike  # [1, 1536]
    key_value_normalization: TensorLike  # [1, 512]
    projection_copy_semaphores: PallasRefProtocol  # [6]

    def __post_init__(self) -> None:
        """Register megakernel PyTree dataclasses on first instantiation."""
        ensure_dataclasses_registered()

    def make_projection_copies(self) -> SimpleNamespace:
        """Create asynchronous projection-weight copy descriptors."""
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


@dataclasses.dataclass(frozen=True, slots=True)
class MultiHeadLatentAttentionWorkspace:
    """Persistent scratch buffers and semaphores used by one MLA invocation."""

    # Unused placeholders ([8, 128] and one DMA semaphore): the DCP attention
    # keeps its pages in mla_dcp's workspace.
    key_cache_vmem: PallasRefProtocol
    value_cache_vmem: PallasRefProtocol
    cache_copy_semaphores: PallasRefProtocol
    query_vmem: PallasRefProtocol  # [batch_size, 3, 256]
    output_gate_values_vmem: PallasRefProtocol  # [batch_size, 384]
    new_key_vmem: PallasRefProtocol  # [batch_size, 3, 256]
    new_value_vmem: PallasRefProtocol  # [batch_size, 3, 128]
    maximum_vmem: PallasRefProtocol  # [batch_size, 3, 1]
    denominator_vmem: PallasRefProtocol  # [batch_size, 3, 1]
    accumulator_vmem: PallasRefProtocol  # [batch_size, 3, 128]

    def __post_init__(self) -> None:
        """Register megakernel PyTree dataclasses on first instantiation."""
        ensure_dataclasses_registered()


@dataclasses.dataclass(frozen=True, slots=True)
class _MlaIoRefs:
    """Input, output, and cache-handoff references for one MLA call."""

    position_ref: PallasRefProtocol
    hidden_ref: PallasRefProtocol
    output_ref: PallasRefProtocol
    new_key_ref: PallasRefProtocol
    new_value_ref: PallasRefProtocol


@dataclasses.dataclass(frozen=True, slots=True)
class _MlaProjectionContext:
    """Projection transfer resources and streams for one MLA layer."""

    weights: MultiHeadLatentAttentionWeights
    streams: Mapping[str, ChunkedWeightStream]
    copy_map: Mapping[str, AsyncCopyProtocol]
    vmem_map: Mapping[str, PallasRefProtocol]
    reduction_phase: PallasRefProtocol


def _make_mla_projection_context(
    weights: MultiHeadLatentAttentionWeights,
    streams: Mapping[str, ChunkedWeightStream],
    communication_workspace: AttentionCommunicationWorkspace,
) -> _MlaProjectionContext:
    """Build typed copy and buffer lookup maps for MLA projections."""
    weight_copies = weights.make_projection_copies()
    copy_map: dict[str, AsyncCopyProtocol] = {
        "query_down": weight_copies.query_down,
        "key_value_down": weight_copies.key_value_down,
        "query_up": weight_copies.query_up,
        "key_value_up": weight_copies.key_value_up,
        "output_gate": weight_copies.output_gate,
        "output": weight_copies.output,
    }
    vmem_map: dict[str, PallasRefProtocol] = {
        "query_down": weights.query_down_projection_vmem,
        "key_value_down": weights.key_value_down_projection_vmem,
        "query_up": weights.query_up_projection_vmem,
        "key_value_up": weights.key_value_up_projection_vmem,
        "output_gate": weights.output_gate_projection_vmem,
        "output": weights.output_projection_vmem,
    }
    return _MlaProjectionContext(
        weights=weights,
        streams=streams,
        copy_map=copy_map,
        vmem_map=vmem_map,
        reduction_phase=communication_workspace.reduction_phase,
    )


def _start_mla_projection_copy(ctx: _MlaProjectionContext, name: str) -> None:
    """Start the stream or direct copy for named MLA projection ``name``."""
    if name in ctx.streams:
        ctx.streams[name].start()
    else:
        ctx.copy_map[name].start()


def _project_mla_input(
    ctx: _MlaProjectionContext,
    name: str,
    activations: jax.Array,
) -> jax.Array:
    """Multiply ``activations`` by projection ``name`` in FP32."""
    guard = ctx.reduction_phase[0] >= 0
    if name in ctx.streams:
        return ctx.streams[name].dot(activations, guard)
    ctx.copy_map[name].wait()
    return dot(activations, ctx.vmem_map[name][...])


def _validate_mla_io_and_weights(
    refs: _MlaIoRefs,
    weights: MultiHeadLatentAttentionWeights,
    dcp: tuple[mla_dcp.DcpCache, mla_dcp.DcpWorkspace],
    sequence_rows: int,
) -> None:
    """Validate MLA input, output, handoff, weight, and context shapes."""
    batch_size, hidden_size = refs.hidden_ref.shape
    cache = dcp[0]
    context_length = (
        cache.page_tokens * cache.pages * (1 if cache.global_pages else cache.group)
    )
    if not 1 <= batch_size <= MAX_BATCH:
        msg = f"MLA supports 1 to {MAX_BATCH} batch rows"
        raise ValueError(msg)
    if refs.output_ref.shape != refs.hidden_ref.shape:
        msg = "MLA output must match the hidden-state shape"
        raise ValueError(msg)
    if not 1 <= sequence_rows <= batch_size or batch_size % sequence_rows:
        msg = "sequence_rows must divide the MLA batch"
        raise ValueError(msg)
    if refs.position_ref.shape != (batch_size,):
        msg = "Positions must match the MLA batch size"
        raise ValueError(msg)
    if (
        weights.query_down_projection.shape != (hidden_size, _MLA_QUERY_LORA_RANK)
        or weights.key_value_down_projection.shape
        != (hidden_size, MLA_LATENT_ROW_LANES)
        or weights.query_up_projection.shape
        != (_MLA_QUERY_LORA_RANK, _MLA_QUERY_UP_WIDTH)
        or weights.key_value_up_projection.shape
        != (_MLA_KV_LORA_RANK, _MLA_KV_UP_WIDTH)
        or weights.output_gate_projection.shape != (hidden_size, _MLA_HEAD_OUT_WIDTH)
        or weights.query_normalization.shape != (1, _MLA_QUERY_LORA_RANK)
        or weights.key_value_normalization.shape != (1, _MLA_KV_LORA_RANK)
        or context_length < _MIN_MLA_CONTEXT_TOKENS
        or context_length % VREG_LANE_COUNT
    ):
        msg = "MLA projection and normalization shapes are invalid"
        raise ValueError(msg)
    if (
        refs.new_key_ref.shape[0] != batch_size
        or refs.new_key_ref.shape[1] < mla_dcp.HEADS_PER_RANK
        or refs.new_key_ref.shape[2:] != (_MLA_KEY_HANDOFF_LANES,)
        or refs.new_value_ref.shape[0] != batch_size
        or refs.new_value_ref.shape[1] < mla_dcp.HEADS_PER_RANK
        or refs.new_value_ref.shape[2:] != (HEAD_DIM,)
    ):
        msg = "MLA key/value outputs have invalid shapes"
        raise ValueError(msg)


def _validate_mla_workspace_shapes(
    workspace: MultiHeadLatentAttentionWorkspace,
    batch_size: int,
) -> None:
    """Validate persistent scratch shapes for an MLA invocation."""
    expected_workspace_shapes = (
        (
            workspace.query_vmem.shape,
            (batch_size, mla_dcp.HEADS_PER_RANK, _MLA_KEY_HANDOFF_LANES),
        ),
        (workspace.output_gate_values_vmem.shape, (batch_size, _MLA_HEAD_OUT_WIDTH)),
        (
            workspace.new_key_vmem.shape,
            (batch_size, mla_dcp.HEADS_PER_RANK, _MLA_KEY_HANDOFF_LANES),
        ),
        (
            workspace.new_value_vmem.shape,
            (batch_size, mla_dcp.HEADS_PER_RANK, HEAD_DIM),
        ),
        (workspace.maximum_vmem.shape, (batch_size, mla_dcp.HEADS_PER_RANK, 1)),
        (workspace.denominator_vmem.shape, (batch_size, mla_dcp.HEADS_PER_RANK, 1)),
        (
            workspace.accumulator_vmem.shape,
            (batch_size, mla_dcp.HEADS_PER_RANK, HEAD_DIM),
        ),
    )
    if any(actual != expected for actual, expected in expected_workspace_shapes):
        msg = "MLA workspace has an invalid shape"
        raise ValueError(msg)


def _write_new_mla_latent_handoff(
    new_key_ref: PallasRefProtocol,
    new_value_ref: PallasRefProtocol,
    new_latent: jax.Array,
    rows: int,
) -> None:
    """Write ``new_latent`` lanes into the key/value handoff buffers."""
    new_key_ref[:, 0] = new_latent[:, :_MLA_KEY_HANDOFF_LANES]
    new_key_ref[:, 1] = new_latent[:, _MLA_KEY_HANDOFF_LANES:_MLA_KV_LORA_RANK]
    new_key_ref[:, 2] = jnp.concatenate(
        (
            new_latent[:, _MLA_KV_LORA_RANK:],
            jnp.zeros((rows, HEAD_DIM), jnp.bfloat16),
        ),
        axis=1,
    )
    new_value_ref[:, : mla_dcp.HEADS_PER_RANK] = jnp.zeros(
        (rows, mla_dcp.HEADS_PER_RANK, HEAD_DIM),
        jnp.bfloat16,
    )


def _project_mla_latents_and_queries(
    refs: _MlaIoRefs,
    workspace: MultiHeadLatentAttentionWorkspace,
    ctx: _MlaProjectionContext,
    epsilon: float,
    *,
    output_projection_after_query_down: bool,
) -> jax.Array:
    """Project and normalize query/KV latents, query-up, and handoff rows."""
    batch_size = refs.hidden_ref.shape[0]
    if "query_down" in ctx.streams:
        query_latent = (
            ctx.streams["query_down"]
            .dot(refs.hidden_ref[...], ctx.reduction_phase[0] >= 0)
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
    else:
        ctx.copy_map["query_down"].wait()
        query_latent = (
            dot(refs.hidden_ref[...], ctx.weights.query_down_projection_vmem[...])
            .astype(jnp.bfloat16)
            .astype(jnp.float32)
        )
    if output_projection_after_query_down:
        # The pool views are reinterpret casts, so the compiler need not see
        # that this copy aliases the q-down loads above; a region (on a runtime
        # condition that is never false) keeps it from being hoisted over them.
        @pl.when(ctx.reduction_phase[0] >= 0)
        def start_output_projection_copy() -> None:
            """Start output-projection copy after q-down consumption."""
            _start_mla_projection_copy(ctx, "output")

    query_latent = (
        query_latent
        * jax.lax.rsqrt(
            jnp.mean(query_latent * query_latent, axis=1, keepdims=True) + epsilon,
        )
        * ctx.weights.query_normalization[...]
    ).astype(jnp.bfloat16)

    ctx.copy_map["key_value_down"].wait()
    compressed = dot(
        refs.hidden_ref[...],
        ctx.weights.key_value_down_projection_vmem[...],
    ).astype(jnp.bfloat16)
    key_value_latent = compressed[:, :_MLA_KV_LORA_RANK].astype(jnp.float32)
    key_value_latent = (
        key_value_latent
        * jax.lax.rsqrt(
            jnp.mean(
                key_value_latent * key_value_latent,
                axis=1,
                keepdims=True,
            )
            + epsilon,
        )
        * ctx.weights.key_value_normalization[...]
    ).astype(jnp.bfloat16)
    workspace.query_vmem[...] = (
        _project_mla_input(ctx, "query_up", query_latent)
        .astype(jnp.bfloat16)
        .reshape(batch_size, mla_dcp.HEADS_PER_RANK, _MLA_KEY_HANDOFF_LANES)
    )

    # The new token's latent row, handed out for the owner's commit.
    new_latent = jnp.concatenate(
        (
            key_value_latent,
            compressed[:, _MLA_KV_LORA_RANK:_MLA_ROPE_END],
            jnp.zeros((batch_size, _MLA_ROPE_DIM), jnp.bfloat16),
        ),
        axis=1,
    )
    _write_new_mla_latent_handoff(
        refs.new_key_ref,
        refs.new_value_ref,
        new_latent,
        batch_size,
    )
    return new_latent


def _absorb_mla_head_queries(
    query: jax.Array,
    up_weight: PallasRefProtocol,
    rows: int,
) -> list[jax.Array]:
    """Absorb W_UK into each rank head's nope query and apply score scaling."""
    absorbed: list[jax.Array] = []
    for head in range(mla_dcp.HEADS_PER_RANK):
        key_start = head * _MLA_KEY_HANDOFF_LANES
        latent_query = jax.lax.dot_general(
            query[:, head, :HEAD_DIM],
            up_weight[:, key_start : key_start + HEAD_DIM],
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        absorbed.append(
            jnp.concatenate(
                (
                    latent_query,
                    query[:, head, HEAD_DIM:_MLA_ATTN_SCORE_DIM].astype(jnp.float32),
                    jnp.zeros((rows, _MLA_ROPE_DIM), jnp.float32),
                ),
                axis=1,
            )
            * _MLA_ATTN_SCORE_SCALE,
        )
    return absorbed


def _project_mla_value_up_heads(
    merged: jax.Array,
    up_weight: PallasRefProtocol,
    head_offset: int,
) -> list[jax.Array]:
    """Project merged latent outputs through W_UV using high/low BF16 split."""
    outputs: list[jax.Array] = []
    for head in range(mla_dcp.HEADS_PER_RANK):
        value_start = head * _MLA_KEY_HANDOFF_LANES + HEAD_DIM
        value_up = up_weight[:, value_start : value_start + HEAD_DIM]
        latent = merged[:, head_offset + head]
        high = latent.astype(jnp.bfloat16)
        low = (latent - high.astype(jnp.float32)).astype(jnp.bfloat16)
        high_product = 0 + jax.lax.dot_general(
            high,
            value_up,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        low_product = jax.lax.dot_general(
            low,
            value_up,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        outputs.append(high_product + low_product)
    return outputs


def _attend_mla_dcp_heads(
    hidden_ref: PallasRefProtocol,
    workspace: MultiHeadLatentAttentionWorkspace,
    ctx: _MlaProjectionContext,
    dcp_attention: mla_dcp.DcpAttention,
    new_latent_and_layer: tuple[jax.Array, jax.Array | int],
) -> None:
    """Run DCP attention with absorbed W_UK, overlapped gate, and W_UV."""
    new_latent, layer_index = new_latent_and_layer
    batch_size = hidden_ref.shape[0]
    if "key_value_up" in ctx.streams:
        ctx.streams["key_value_up"].wait_all()
    else:
        ctx.copy_map["key_value_up"].wait()
    # m_kb [512, 768]: head h's key columns h*256 : +128, value columns +128 : +256.
    up = ctx.weights.key_value_up_projection_vmem
    query = workspace.query_vmem[...]  # [B, 3, 256]: nope 128 | rope 64 | 64
    absorbed = _absorb_mla_head_queries(query, up, batch_size)

    def output_gate() -> None:
        """Project output gate while DCP query gather is in flight."""
        workspace.output_gate_values_vmem[...] = _project_mla_input(
            ctx,
            "output_gate",
            hidden_ref[...],
        ).astype(jnp.bfloat16)

    merged = dcp_attention.attend(
        jnp.stack(absorbed, axis=1),
        new_latent.astype(jnp.float32),
        layer_index % 2,
        between=output_gate,
    )
    outputs = _project_mla_value_up_heads(merged, up, 0)
    workspace.accumulator_vmem[...] = jnp.stack(outputs, axis=1)
    workspace.denominator_vmem[...] = jnp.ones(
        (batch_size, mla_dcp.HEADS_PER_RANK, 1),
        jnp.float32,
    )


def _resolve_mla_output_stream(
    ctx: _MlaProjectionContext,
) -> ChunkedWeightStream | None:
    """Await the MLA output projection or return its multi-chunk stream."""
    output_stream = ctx.streams.get("output")
    if output_stream is None:
        ctx.copy_map["output"].wait()
    elif output_stream.chunks == 1:
        output_stream.wait_all()
        output_stream = None
    return output_stream


def mla(
    position_ref: PallasRefProtocol,  # [batch_size]
    hidden_ref: PallasRefProtocol,  # [batch_size, hidden_size]
    output_ref: PallasRefProtocol,  # [batch_size, hidden_size]
    new_key_ref: PallasRefProtocol,  # [batch_size, padded_head_count, 256]
    new_value_ref: PallasRefProtocol,  # [batch_size, padded_head_count, 128]
    *,
    weights: MultiHeadLatentAttentionWeights,
    workspace: MultiHeadLatentAttentionWorkspace,
    communication_workspace: AttentionCommunicationWorkspace,
    dcp: tuple[mla_dcp.DcpCache, mla_dcp.DcpWorkspace],
    epsilon: float = 1e-5,
    synchronize_devices: bool = True,
    weights_are_prefetched: bool = False,
    token_scatter_output: bool = False,
    sequence_rows: int = 1,
    output_projection_after_query_down: bool = False,
    streams: Mapping[str, ChunkedWeightStream] | None = None,
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
    refs = _MlaIoRefs(
        position_ref=position_ref,
        hidden_ref=hidden_ref,
        output_ref=output_ref,
        new_key_ref=new_key_ref,
        new_value_ref=new_value_ref,
    )
    batch_size = hidden_ref.shape[0]
    _validate_mla_io_and_weights(refs, weights, dcp, sequence_rows)
    _validate_mla_workspace_shapes(workspace, batch_size)
    if synchronize_devices:
        collectives32.barrier()
        communication_workspace.reduction_phase[0] = 0

    ctx = _make_mla_projection_context(weights, streams or {}, communication_workspace)
    mla_scope = [phase_start("q6_mla_projections")]

    if not weights_are_prefetched:
        ctx.copy_map["query_down"].start()
        ctx.copy_map["key_value_down"].start()
    _start_mla_projection_copy(ctx, "query_up")
    _start_mla_projection_copy(ctx, "key_value_up")
    _start_mla_projection_copy(ctx, "output_gate")
    if not output_projection_after_query_down:
        _start_mla_projection_copy(ctx, "output")

    # Diagnostic (KernelConfig.dcp_skip): skip the DCP query gather, local
    # attention and/or partial combine, e.g. "gather,combine". Results are
    # invalid; timing only.
    dcp_skip = _config().dcp_skip
    dcp_attention = mla_dcp.DcpAttention(
        dcp[0],
        dcp[1],
        position_ref,
        batch_size,
        sequence_rows=sequence_rows,
        skip=dcp_skip,
        cache_verify_rows=(sequence_rows > 1 and dcp[0].dtype == jnp.float8_e4m3fn),
    )
    dcp_attention.prime()

    new_latent = _project_mla_latents_and_queries(
        refs,
        workspace,
        ctx,
        epsilon,
        output_projection_after_query_down=output_projection_after_query_down,
    )

    phase_end(mla_scope.pop())
    mla_scope.append(phase_start("q7_mla_attend"))
    workspace.maximum_vmem[...] = jnp.full(
        (batch_size, mla_dcp.HEADS_PER_RANK, 1),
        -jnp.inf,
        jnp.float32,
    )
    workspace.denominator_vmem[...] = jnp.zeros(
        (batch_size, mla_dcp.HEADS_PER_RANK, 1),
        jnp.float32,
    )
    workspace.accumulator_vmem[...] = jnp.zeros(
        (batch_size, mla_dcp.HEADS_PER_RANK, HEAD_DIM),
        jnp.float32,
    )

    layer_index = dcp[0].layer
    if layer_index is None:
        msg = "MLA DCP cache requires a layer index"
        raise ValueError(msg)
    # sequence_rows > 1: DcpAttention folds each sequence's rows in causally.
    _attend_mla_dcp_heads(
        hidden_ref,
        workspace,
        ctx,
        dcp_attention,
        (new_latent, layer_index),
    )

    attention = (
        workspace.accumulator_vmem[...] / workspace.denominator_vmem[...]
    ).reshape(batch_size, _MLA_HEAD_OUT_WIDTH).astype(jnp.bfloat16) * jax.nn.sigmoid(
        workspace.output_gate_values_vmem[...].astype(jnp.float32),
    ).astype(jnp.bfloat16)
    phase_end(mla_scope.pop())
    output_stream = _resolve_mla_output_stream(ctx)

    def project_output(reduction_vmem: PallasRefProtocol) -> None:
        """Project gated MLA outputs and reduce across the TP mesh."""
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
                dot,
                token_scatter_output=token_scatter_output,
                weight_stream=output_stream,
                staging=communication_workspace.projection_staging_vmem,
                gather_wire=communication_workspace.attention_gather_vmem,
            )

    reduction_width = (
        _HALF_HIDDEN_REDUCTION_WIDTH
        if batch_size > _WIDE_REDUCTION_MAX_BATCH
        else _WIDE_REDUCTION_WIDTH
    )
    pl.run_scoped(
        project_output,
        tpu.VMEM((batch_size, reduction_width), jnp.float32),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _MlaDpShardContext:
    """Shared context for multi-shard attention-DP MLA execution."""

    projection: _MlaProjectionContext
    up_stream: ChunkedWeightStream
    guard: jax.Array
    rows: int
    shards: int


def _project_mla_dp_latents_and_absorbed(
    shard_ctx: _MlaDpShardContext,
    hidden_ref: PallasRefProtocol,
    handoff_refs: tuple[PallasRefProtocol, PallasRefProtocol],
    epsilon: float,
) -> tuple[list[jax.Array], jax.Array]:
    """Compute DP-shared latents, handoff rows, and absorbed shard queries."""
    ctx = shard_ctx.projection
    guard = shard_ctx.guard
    rows = shard_ctx.rows
    if "query_down" in ctx.streams:
        query_latent = ctx.streams["query_down"].dot(hidden_ref[...], guard)
    else:
        ctx.copy_map["query_down"].wait()
        query_latent = dot(
            hidden_ref[...],
            ctx.weights.query_down_projection_vmem[...],
        )
    query_latent = query_latent.astype(jnp.bfloat16).astype(jnp.float32)

    # The pooled output projection overwrites the q-down projection.
    @pl.when(guard)
    def start_output_projection_copy() -> None:
        """Start output-projection copy after q-down consumption."""
        _start_mla_projection_copy(ctx, "output")

    query_latent = (
        query_latent
        * jax.lax.rsqrt(
            jnp.mean(query_latent * query_latent, axis=1, keepdims=True) + epsilon,
        )
        * ctx.weights.query_normalization[...]
    ).astype(jnp.bfloat16)
    ctx.copy_map["key_value_down"].wait()
    compressed = dot(
        hidden_ref[...],
        ctx.weights.key_value_down_projection_vmem[...],
    ).astype(jnp.bfloat16)
    key_value_latent = compressed[:, :_MLA_KV_LORA_RANK].astype(jnp.float32)
    key_value_latent = (
        key_value_latent
        * jax.lax.rsqrt(
            jnp.mean(key_value_latent * key_value_latent, axis=1, keepdims=True)
            + epsilon,
        )
        * ctx.weights.key_value_normalization[...]
    ).astype(jnp.bfloat16)
    new_latent = jnp.concatenate(
        (
            key_value_latent,
            compressed[:, _MLA_KV_LORA_RANK:_MLA_ROPE_END],
            jnp.zeros((rows, _MLA_ROPE_DIM), jnp.bfloat16),
        ),
        axis=1,
    )
    _write_new_mla_latent_handoff(handoff_refs[0], handoff_refs[1], new_latent, rows)
    shard_ctx.up_stream.wait_all()
    absorbed: list[jax.Array] = []
    for shard in range(shard_ctx.shards):
        query = (
            _shard_dot(ctx.streams["query_up"], query_latent, shard, guard)
            .astype(jnp.bfloat16)
            .reshape(rows, mla_dcp.HEADS_PER_RANK, _MLA_KEY_HANDOFF_LANES)
        )
        # [512, 768]: head h's key columns h*256 : +128, value +128 : +256
        up = shard_ctx.up_stream.view(shard)
        absorbed.extend(_absorb_mla_head_queries(query, up, rows))
    return absorbed, new_latent


def _apply_mla_dp_value_up_and_gates(
    attended_out: PallasRefProtocol,
    up_stream: ChunkedWeightStream,
    merged: jax.Array,
    rows: int,
    shards: int,
) -> None:
    """Apply per-shard W_UV value-up and sigmoid output gates."""
    for shard in range(shards):
        up = up_stream.view(shard)
        outputs = _project_mla_value_up_heads(
            merged,
            up,
            mla_dcp.HEADS_PER_RANK * shard,
        )
        attended_out[shard] = jnp.stack(outputs, axis=1).reshape(
            rows,
            _MLA_HEAD_OUT_WIDTH,
        ).astype(jnp.bfloat16) * jax.nn.sigmoid(
            attended_out[shard].astype(jnp.float32),
        ).astype(jnp.bfloat16)


def mla_dcp_shards(
    position_ref: PallasRefProtocol,  # [rows]
    hidden_ref: PallasRefProtocol,  # [rows, hidden_size]
    new_key_ref: PallasRefProtocol,  # [rows, padded_head_count, 256]
    new_value_ref: PallasRefProtocol,  # [rows, padded_head_count, 128]
    attended_out: PallasRefProtocol,  # [shards, rows, 384]
    *,
    weights: MultiHeadLatentAttentionWeights,
    streams: Mapping[str, ChunkedWeightStream],
    communication_workspace: AttentionCommunicationWorkspace,
    dcp: tuple[mla_dcp.DcpCache, mla_dcp.DcpWorkspace],
    epsilon: float = 1e-5,
    sequence_rows: int = 1,
) -> None:
    """Execute one attention-DP MLA layer across all held head shards.

    Each rank's held TP32 shards (3 heads each) attend in one
    ``mla_dcp.DcpAttention`` call over the DP group's latent pages, so each
    page is read once for every held head. ``sequence_rows`` behaves as in
    :func:`mla` over the group's rows.

    The q/kv latents are computed once; each shard's queries come from its
    chunk of the query-up stream and are absorbed with its W_UK columns (the
    key-value-up stream holds every shard); the output gates are projected
    while the query gather is in flight. W_UV and the gate leave each shard's
    gated attention in ``attended_out[shard]`` for the caller's output
    projection. The new latent row is handed out through ``new_key_ref``
    rows 0..2 as in :func:`mla`.
    """
    rows = hidden_ref.shape[0]
    shards = attended_out.shape[0]
    up_stream = streams["key_value_up"]
    if up_stream.slots is not None or up_stream.chunks != shards:
        msg = "DCP attention DP needs every shard's key-value-up columns resident"
        raise ValueError(msg)
    ctx = _make_mla_projection_context(weights, streams, communication_workspace)
    guard = communication_workspace.reduction_phase[0] >= 0
    for name in ("query_up", "key_value_up", "output_gate"):
        _start_mla_projection_copy(ctx, name)
    dcp_attention = mla_dcp.DcpAttention(
        dcp[0],
        dcp[1],
        position_ref,
        rows,
        skip=_config().dcp_skip,
        heads=mla_dcp.HEADS_PER_RANK * shards,
        sequence_rows=sequence_rows,
    )
    dcp_attention.prime()
    scope = phase_start("q6_mla_projections")
    shard_ctx = _MlaDpShardContext(
        projection=ctx,
        up_stream=up_stream,
        guard=guard,
        rows=rows,
        shards=shards,
    )
    absorbed, new_latent = _project_mla_dp_latents_and_absorbed(
        shard_ctx,
        hidden_ref,
        (new_key_ref, new_value_ref),
        epsilon,
    )
    phase_end(scope)
    scope = phase_start("q7_mla_attend")

    def output_gates() -> None:
        """Project per-shard output gates while DCP query gather runs."""
        for shard in range(shards):
            attended_out[shard] = _shard_dot(
                streams["output_gate"],
                hidden_ref[...],
                shard,
                guard,
            ).astype(jnp.bfloat16)

    layer_index = dcp[0].layer
    if layer_index is None:
        msg = "MLA DCP shards require a cache layer index"
        raise ValueError(msg)
    merged = dcp_attention.attend(
        jnp.stack(absorbed, axis=1),
        new_latent.astype(jnp.float32),
        layer_index % 2,
        between=output_gates,
    )
    _apply_mla_dp_value_up_and_gates(attended_out, up_stream, merged, rows, shards)
    phase_end(scope)


# At B4, scatter the projection rows and gather the completed output once.
# Keep all 16 BF16 rank-pair partials until the final FP32 sum: intermediate
# host sums can change long-context logits. The final gather has private
# persistent storage so adjacent phases cannot overwrite unread packets.
B4_ATTENTION_GATHER = _config().b4_attention_gather


@dataclasses.dataclass(frozen=True, slots=True)
class _Attention2dContext:
    """Shared buffers and topology scalars for 2D attention output projection."""

    rank: jax.Array
    slot: jax.Array
    batch_size: int
    width: int
    group_size: int
    hidden: int
    half: int
    pair_count: int
    weight: PallasRefProtocol
    out: PallasRefProtocol
    comm: PallasRefProtocol
    local: PallasRefProtocol
    host_reduction: PallasRefProtocol
    output_backing: PallasRefProtocol
    sends: PallasRefProtocol
    recvs: PallasRefProtocol
    phase: PallasRefProtocol
    dot: Callable[[jax.Array, jax.Array], jax.Array]
    weight_stream: ChunkedWeightStream | None
    staging: PallasRefProtocol | None
    gather_wire: PallasRefProtocol | None


def _project_pair_attention_features(
    ctx: _Attention2dContext,
    attended: jax.Array,
    features: PallasRefProtocol,
) -> None:
    """Exchange pair attention features and accumulate paired half-projections."""
    g = ctx.group_size
    width = ctx.width
    view = features.at[:, pl.ds((ctx.rank % g) * width, width)]
    view[...] = attended
    for offset in range(g - 1, 0, -1):
        tpu.make_async_remote_copy(
            view,
            view,
            ctx.sends.at[_ATTENTION_PAIR_GATHER_SEM_BASE + ctx.slot],
            ctx.recvs.at[_ATTENTION_PAIR_GATHER_SEM_BASE + ctx.slot],
            device_id=(ctx.rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    ctx.comm[...] = jnp.zeros(ctx.comm.shape, jnp.float32)
    if ctx.weight_stream is None:
        own_weight = ctx.weight[pl.ds((ctx.rank % _RANK_PAIR_SIZE) * width, width), :]
    else:
        ctx.weight_stream.wait(0)
        own_weight = ctx.weight_stream.view(0)[...]
    ctx.comm[:, : ctx.half] = ctx.dot(attended, own_weight)
    if ctx.weight_stream is not None:
        ctx.weight_stream.release(0, ctx.phase[0] >= 0)
    count = features.at[:, : (g - 1) * width]
    tpu.make_async_copy(
        count,
        count,
        ctx.sends.at[_ATTENTION_PAIR_GATHER_SEM_BASE + ctx.slot],
    ).wait()
    tpu.make_async_copy(
        count,
        count,
        ctx.recvs.at[_ATTENTION_PAIR_GATHER_SEM_BASE + ctx.slot],
    ).wait()
    peer_lane = (ctx.rank % _RANK_PAIR_SIZE) ^ 1
    if ctx.weight_stream is None:
        peer_weight = ctx.weight[pl.ds(peer_lane * width, width), :]
    else:
        ctx.weight_stream.wait(1)
        peer_weight = ctx.weight_stream.view(1)[...]
    ctx.comm[:, : ctx.half] += ctx.dot(
        features[:, pl.ds(peer_lane * width, width)],
        peer_weight,
    )


def _exchange_local_attention_partials(
    ctx: _Attention2dContext,
    tokens_per_pair: int,
    payload_rows: int,
    pair: jax.Array,
) -> PallasRefProtocol:
    """Reduce-scatter attention partials across the four local rank pairs."""
    # Reuse attention/MoE communication storage after the input gather.
    # Keep each asynchronous role in a distinct physical lane. This
    # prevents the compiler from reusing aliased transformed refs
    # before every remote send has retired.
    if (
        ctx.staging is None
        and _BF16_PER_FP32_WORD * ctx.local.shape[1] < _LOCAL_RANK_PAIRS * payload_rows
    ):
        msg = "wide reduction rows too small for the payloads"
        raise ValueError(msg)
    local_send = (
        (
            ctx.local.at[0].bitcast(jnp.bfloat16)
            if ctx.staging is None
            else ctx.staging.at[0]
        )
        .at[pl.ds(0, _LOCAL_RANK_PAIRS * payload_rows), :]
        .reshape(_LOCAL_RANK_PAIRS, payload_rows, VREG_LANE_COUNT)
    )
    local_receive = (
        (
            ctx.local.at[1].bitcast(jnp.bfloat16)
            if ctx.staging is None
            else ctx.staging.at[1]
        )
        .at[pl.ds(0, _LOCAL_RANK_PAIRS * payload_rows), :]
        .reshape(_LOCAL_RANK_PAIRS, payload_rows, VREG_LANE_COUNT)
    )
    for target_pair in range(_LOCAL_RANK_PAIRS):
        if tokens_per_pair == 1:
            partial = ctx.comm[target_pair, : ctx.half][None, :]
        else:
            partial = ctx.comm[
                pl.ds(target_pair * tokens_per_pair, tokens_per_pair),
                : ctx.half,
            ]
        local_send[target_pair] = partial.astype(jnp.bfloat16).reshape(
            payload_rows,
            VREG_LANE_COUNT,
        )
    local_receive[pair] = local_send[pair]
    local_copies: list[AsyncCopyProtocol] = []
    for offset in range(_LOCAL_RANK_PAIRS - 1, 0, -1):
        target_pair = pair ^ offset
        copy = tpu.make_async_remote_copy(
            local_send.at[target_pair],
            local_receive.at[pair],
            ctx.sends.at[offset - 1],
            ctx.recvs.at[offset - 1],
            device_id=(ctx.rank ^ (_RANK_PAIR_SIZE * offset),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        local_copies.append(copy)
    for copy in local_copies:
        copy.wait()
    return local_receive


def _exchange_host_attention_partials(
    ctx: _Attention2dContext,
    local_receive: PallasRefProtocol,
    payload_rows: int,
    host: jax.Array,
    *,
    ordered_full_output: bool,
) -> jax.Array:
    """Exchange attention partials across the four hosts and sum in FP32."""
    if ordered_full_output:
        # Host h owns rank-pair lanes 4*h .. 4*h+3. Transport those
        # unreduced partials without changing their reduction order.
        host_wire_rows = _LOCAL_RANK_PAIRS * payload_rows
        if host_wire_rows > _BF16_PER_FP32_WORD * ctx.host_reduction.shape[1]:
            msg = "Host reduction buffer is too small for ordered attention partials"
            raise ValueError(msg)
        host_receive = ctx.host_reduction.bitcast(jnp.bfloat16).at[
            :,
            pl.ds(0, host_wire_rows),
            :,
        ]
        host_receive[host] = local_receive[...].reshape(
            host_wire_rows,
            VREG_LANE_COUNT,
        )
    else:
        local_sum = jnp.sum(local_receive[...].astype(jnp.float32), axis=0)
        # The caller supplies the four-lane FP32 host workspace
        # used by MoE after attention.
        host_receive = ctx.host_reduction.at[:, pl.ds(0, payload_rows), :]
        host_receive[host] = local_sum
    host_copies: list[AsyncCopyProtocol] = []
    host_view = host_receive.at[host]
    for offset in range(_HOSTS_IN_SLICE - 1, 0, -1):
        copy = tpu.make_async_remote_copy(
            host_view,
            host_view,
            ctx.sends.at[_ATTENTION_HOST_SEM_BASE + offset - 1],
            ctx.recvs.at[_ATTENTION_HOST_SEM_BASE + offset - 1],
            device_id=(ctx.rank ^ (mla_dcp.HOST_LANES * offset),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        host_copies.append(copy)
    for copy in host_copies:
        copy.wait()
    if ordered_full_output:
        rank_partials = (
            host_receive[...]
            .reshape(_TP16_PAIRS_TOTAL, payload_rows, VREG_LANE_COUNT)
            .astype(jnp.float32)
        )
        return jnp.sum(rank_partials, axis=0).astype(jnp.bfloat16)
    return jnp.sum(host_receive[...], axis=0).astype(jnp.bfloat16)


def _scatter_reduce_attention_2d(
    ctx: _Attention2dContext,
    *,
    ordered_full_output: bool,
) -> None:
    """Reduce attention projections to local pair owners and assemble hidden."""
    if ctx.batch_size not in (_B4_BATCH_SIZE, *SCATTER_BATCHES):
        msg = (
            f"Scattered attention output requires batch in "
            f"{(_B4_BATCH_SIZE, *SCATTER_BATCHES)}, got {ctx.batch_size}"
        )
        raise ValueError(msg)
    tokens_per_pair = ctx.batch_size // _LOCAL_RANK_PAIRS
    payload_rows = tokens_per_pair * ctx.half // VREG_LANE_COUNT
    host = ctx.rank // mla_dcp.HOST_LANES
    pair = (ctx.rank % mla_dcp.HOST_LANES) // _RANK_PAIR_SIZE
    parity = ctx.rank % _RANK_PAIR_SIZE

    local_receive = _exchange_local_attention_partials(
        ctx,
        tokens_per_pair,
        payload_rows,
        pair,
    )
    # ``output`` is backed by a native 16-row allocation; only the
    # rows owned by this local pair are live during the reduction.
    owned_output = _bf16_rows_view(ctx.output_backing, tokens_per_pair, ctx.hidden)
    reduced_half = _exchange_host_attention_partials(
        ctx,
        local_receive,
        payload_rows,
        host,
        ordered_full_output=ordered_full_output,
    )
    pair_send = ctx.local.at[2].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    pair_receive = ctx.local.at[3].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    pair_send[...] = reduced_half
    pair_copy = tpu.make_async_remote_copy(
        pair_send,
        pair_receive,
        ctx.sends.at[_ATTENTION_PAIR_SWAP_SEM],
        ctx.recvs.at[_ATTENTION_PAIR_SWAP_SEM],
        device_id=(ctx.rank ^ 1,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    pair_copy.start()
    pair_copy.wait()
    sent_half = pair_send[...].reshape(tokens_per_pair, ctx.half)
    received_half = pair_receive[...].reshape(tokens_per_pair, ctx.half)
    owned_output[:, : ctx.half] = jnp.where(parity == 0, sent_half, received_half)
    owned_output[:, ctx.half :] = jnp.where(parity == 0, received_half, sent_half)
    if ordered_full_output:
        # Full-batch rows for residual mixing and MoE; the packets are
        # private since peers may be a phase apart.
        if ctx.gather_wire is None:
            msg = "Ordered full-output attention gather requires gather_wire scratch"
            raise ValueError(msg)
        _gather_local_pair_rows(
            cast("PallasRefProtocol", owned_output),
            ctx.out,
            ctx.sends,
            ctx.recvs,
            ctx.gather_wire,
            owners=_LOCAL_RANK_PAIRS,
        )


def _all_reduce_and_gather_attention_2d(
    ctx: _Attention2dContext,
    output: PallasRefProtocol,
) -> None:
    """All-reduce half-hidden attention outputs across pairs and gather halves."""
    g = ctx.group_size
    nr = ctx.pair_count
    rows = (
        (ctx.half + (_BF16_ELEMENTS_PER_FP32_ROW - 1)) // _BF16_ELEMENTS_PER_FP32_ROW
    ) * _BF16_PER_FP32_WORD
    payload = rows // _BF16_PER_FP32_WORD
    batched_payload = ctx.batch_size * payload
    receive = ctx.local.reshape(
        nr,
        ctx.local.shape[1] * mla_dcp.HOST_LANES // nr,
        VREG_LANE_COUNT,
    )
    if batched_payload > receive.shape[1]:
        msg = "Local reduction buffer is too small for batched attention payload"
        raise ValueError(msg)
    bits = tpu.bitcast(
        ctx.comm[:, : rows * VREG_LANE_COUNT]
        .reshape(ctx.batch_size * rows, VREG_LANE_COUNT)
        .astype(jnp.bfloat16),
        jnp.uint32,
    )
    receive[ctx.rank // g, :batched_payload, :] = jax.lax.bitcast_convert_type(
        bits,
        jnp.float32,
    )
    view = receive.at[ctx.rank // g, :batched_payload, :]
    for offset in range(nr - 1, 0, -1):
        tpu.make_async_remote_copy(
            view,
            view,
            ctx.sends.at[_ATTENTION_ALL_REDUCE_SEM_BASE + ctx.slot],
            ctx.recvs.at[_ATTENTION_ALL_REDUCE_SEM_BASE + ctx.slot],
            device_id=(ctx.rank ^ (g * offset),),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    count = receive.at[: nr - 1, :batched_payload, :]
    tpu.make_async_copy(
        count,
        count,
        ctx.sends.at[_ATTENTION_ALL_REDUCE_SEM_BASE + ctx.slot],
    ).wait()
    tpu.make_async_copy(
        count,
        count,
        ctx.recvs.at[_ATTENTION_ALL_REDUCE_SEM_BASE + ctx.slot],
    ).wait()
    values = tpu.bitcast(
        jax.lax.bitcast_convert_type(receive[:, :batched_payload, :], jnp.uint32),
        jnp.bfloat16,
    ).astype(jnp.float32)
    ctx.comm[:, : rows * VREG_LANE_COUNT] = jnp.sum(values, axis=0).reshape(
        ctx.batch_size,
        rows * VREG_LANE_COUNT,
    )
    view = output.at[:, pl.ds((ctx.rank % g) * ctx.half, ctx.half)]
    view[...] = ctx.comm[:, : ctx.half].astype(jnp.bfloat16)
    for offset in range(g - 1, 0, -1):
        tpu.make_async_remote_copy(
            view,
            view,
            ctx.sends.at[_ATTENTION_OUTPUT_GATHER_SEM_BASE + ctx.slot],
            ctx.recvs.at[_ATTENTION_OUTPUT_GATHER_SEM_BASE + ctx.slot],
            device_id=(ctx.rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    count = output.at[:, : (g - 1) * ctx.half]
    tpu.make_async_copy(
        count,
        count,
        ctx.sends.at[_ATTENTION_OUTPUT_GATHER_SEM_BASE + ctx.slot],
    ).wait()
    tpu.make_async_copy(
        count,
        count,
        ctx.recvs.at[_ATTENTION_OUTPUT_GATHER_SEM_BASE + ctx.slot],
    ).wait()
    ctx.out[...] = output[...]


def _project_attention_2d(
    attended: jax.Array,
    weight: PallasRefProtocol,
    out: PallasRefProtocol,
    comm: PallasRefProtocol,
    local: PallasRefProtocol,
    host_reduction: PallasRefProtocol,
    feature_backing: PallasRefProtocol,
    output_backing: PallasRefProtocol,
    sends: PallasRefProtocol,
    recvs: PallasRefProtocol,
    phase: PallasRefProtocol,
    dot: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    token_scatter_output: bool = False,
    weight_stream: ChunkedWeightStream | None = None,
    staging: PallasRefProtocol | None = None,
    gather_wire: PallasRefProtocol | None = None,
) -> None:
    """Project attention features in 2D TP layout and reduce across the mesh.

    ``weight_stream`` (optional) is a row ``ChunkedWeightStream`` of ``weight``
    whose positions 0 and 1 are this rank's and its pair peer's feature rows.
    """
    rank = jax.lax.axis_index(AXIS_TP)
    slot = phase[0] % _DOUBLE_BUFFER_SLOTS
    batch_size = attended.shape[0]
    ordered_full_output = (
        B4_ATTENTION_GATHER
        and batch_size == _B4_BATCH_SIZE
        and not token_scatter_output
    )
    width = attended.shape[-1]
    g = weight.shape[0] // width
    if g != _RANK_PAIR_SIZE:
        msg = f"2D attention projection requires pair group size 2, got {g}"
        raise ValueError(msg)
    if not (
        1 <= batch_size <= MAX_BATCH
        and out.shape[0] == batch_size
        and comm.shape[0] == batch_size
    ):
        msg = "Attention projection batch sizes must match across inputs and buffers"
        raise ValueError(msg)
    h = out.shape[-1]
    half = h // g
    nr = TP_SIZE // g
    # Persistent addresses cannot alias preceding MoE temporaries on a slow peer.
    # Reserve 16x1024 BF16 for features and 16x7168 for outputs, including
    # native row padding. Total physical storage is exactly 256 KiB.
    if g * width > _ATTENTION_FEATURE_SHARD_WIDTH or h != HIDDEN_SIZE:
        msg = "2D attention projection requires g*width <= 1024 and hidden == 7168"
        raise ValueError(msg)

    ctx = _Attention2dContext(
        rank=rank,
        slot=slot,
        batch_size=batch_size,
        width=width,
        group_size=g,
        hidden=h,
        half=half,
        pair_count=nr,
        weight=weight,
        out=out,
        comm=comm,
        local=local,
        host_reduction=host_reduction,
        output_backing=output_backing,
        sends=sends,
        recvs=recvs,
        phase=phase,
        dot=dot,
        weight_stream=weight_stream,
        staging=staging,
        gather_wire=gather_wire,
    )
    feature_store = _bf16_rows_view(
        feature_backing,
        batch_size,
        _ATTENTION_FEATURE_SHARD_WIDTH,
    )
    output_store = _bf16_rows_view(output_backing, batch_size, HIDDEN_SIZE)
    _project_pair_attention_features(
        ctx,
        attended,
        cast("PallasRefProtocol", feature_store.at[:, : g * width]),
    )
    if token_scatter_output or ordered_full_output:
        _scatter_reduce_attention_2d(ctx, ordered_full_output=ordered_full_output)
    else:
        _all_reduce_and_gather_attention_2d(
            ctx, cast("PallasRefProtocol", output_store)
        )
    phase[0] += 1


@dataclasses.dataclass(frozen=True, slots=True)
class _AttentionDpContext:
    """Shared buffers and topology coordinates for DP attention projection."""

    rank: jax.Array
    slot: jax.Array
    lane: jax.Array
    pair: jax.Array
    host: jax.Array
    guard: jax.Array
    shards: int
    rows: int
    width: int
    hidden: int
    half: int
    owners: int
    rows_per_pair: int
    weight: PallasRefProtocol
    comm: PallasRefProtocol
    local: PallasRefProtocol
    host_reduction: PallasRefProtocol
    sends: PallasRefProtocol
    recvs: PallasRefProtocol
    dot: Callable[[jax.Array, jax.Array], jax.Array]
    weight_stream: ChunkedWeightStream | None
    staging: PallasRefProtocol | None


def _dp_weight_rows(ctx: _AttentionDpContext, position: int) -> jax.Array:
    """Return one pair-half output-projection weight tile for ``position``."""
    shard, peer = position % ctx.shards, position // ctx.shards
    if ctx.weight_stream is not None:
        ctx.weight_stream.wait(position)
        return ctx.weight_stream.view(position)[...]
    return ctx.weight[
        pl.ds((shard * _RANK_PAIR_SIZE + (ctx.lane ^ peer)) * ctx.width, ctx.width),
        :,
    ]


def _project_dp_shard_features(
    ctx: _AttentionDpContext,
    attended: PallasRefProtocol,
    feature_backing: PallasRefProtocol,
) -> None:
    """Exchange pair features and accumulate all DP head-shard projections."""
    features = _bf16_rows_view(
        feature_backing,
        ctx.rows,
        feature_backing.shape[-1],
    )
    if features.shape[-1] < _RANK_PAIR_SIZE * ctx.width * ctx.shards:
        msg = "Feature backing width is too small for attention-DP shards"
        raise ValueError(msg)
    for shard in range(ctx.shards):
        view = features.at[
            :,
            pl.ds(
                shard * _RANK_PAIR_SIZE * ctx.width + ctx.lane * ctx.width,
                ctx.width,
            ),
        ]
        view[...] = attended[shard]
        tpu.make_async_remote_copy(
            view,
            view,
            ctx.sends.at[_ATTENTION_PAIR_GATHER_SEM_BASE + ctx.slot],
            ctx.recvs.at[_ATTENTION_PAIR_GATHER_SEM_BASE + ctx.slot],
            device_id=(ctx.rank ^ 1,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
    ctx.comm[...] = jnp.zeros(ctx.comm.shape, jnp.float32)
    for shard in range(ctx.shards):
        ctx.comm[:, : ctx.half] += ctx.dot(
            attended[shard],
            _dp_weight_rows(ctx, shard),
        )
        if ctx.weight_stream is not None:
            ctx.weight_stream.release(shard, ctx.guard)
    received = features.at[:, pl.ds(0, ctx.shards * ctx.width)]
    tpu.make_async_copy(
        received,
        received,
        ctx.sends.at[_ATTENTION_PAIR_GATHER_SEM_BASE + ctx.slot],
    ).wait()
    tpu.make_async_copy(
        received,
        received,
        ctx.recvs.at[_ATTENTION_PAIR_GATHER_SEM_BASE + ctx.slot],
    ).wait()
    peer_lane = ctx.lane ^ 1
    for shard in range(ctx.shards):
        peer_features = features[
            :,
            pl.ds(
                shard * _RANK_PAIR_SIZE * ctx.width + peer_lane * ctx.width,
                ctx.width,
            ),
        ]
        ctx.comm[:, : ctx.half] += ctx.dot(
            peer_features,
            _dp_weight_rows(ctx, ctx.shards + shard),
        )
        if ctx.weight_stream is not None:
            ctx.weight_stream.release(ctx.shards + shard, ctx.guard)


def _reduce_dp_single_owner_half(
    ctx: _AttentionDpContext,
    payload_rows: int,
) -> tuple[jax.Array, int, int]:
    """All-gather DP2 single-owner pair partials across 15 same-lane ranks."""
    # Every rank needs both groups' sums: this pair's partial goes to the
    # 15 same-lane ranks in one hop; each rank adds the pairs host by
    # host, then each group's two hosts.
    host_slots = ctx.host_reduction.bitcast(jnp.bfloat16)

    def pair_partials(source_host: jax.Array | int) -> PallasRefProtocol:
        """Return ``[4, payload_rows, 128]`` view for ``source_host``."""
        return (
            host_slots.at[source_host]
            .at[pl.ds(0, _LOCAL_RANK_PAIRS * payload_rows), :]
            .reshape(_LOCAL_RANK_PAIRS, payload_rows, VREG_LANE_COUNT)
        )

    own_partial = pair_partials(ctx.host).at[ctx.pair]
    own_partial[...] = (
        ctx.comm[:, : ctx.half]
        .astype(jnp.bfloat16)
        .reshape(payload_rows, VREG_LANE_COUNT)
    )
    copies: list[AsyncCopyProtocol] = []
    for offset in range(_DP_SINGLE_OWNER_PEERS, 0, -1):
        copy = tpu.make_async_remote_copy(
            own_partial,
            own_partial,
            ctx.sends.at[_DP_SINGLE_OWNER_SEM],
            ctx.recvs.at[_DP_SINGLE_OWNER_SEM],
            device_id=(ctx.rank ^ (_RANK_PAIR_SIZE * offset),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
    host_sums = [
        jnp.sum(pair_partials(source_host)[...].astype(jnp.float32), axis=0)
        for source_host in range(_HOSTS_IN_SLICE)
    ]
    reduced_half = jnp.concatenate(
        [
            host_sums[_RANK_PAIR_SIZE * group] + host_sums[_RANK_PAIR_SIZE * group + 1]
            for group in range(DP2_GROUPS)
        ],
        axis=0,
    ).astype(jnp.bfloat16)
    return (
        reduced_half,
        payload_rows * DP2_GROUPS,
        ctx.rows_per_pair * DP2_GROUPS,
    )


def _reduce_scatter_dp_pair_half(
    ctx: _AttentionDpContext,
    payload_rows: int,
) -> jax.Array:
    """Reduce-scatter DP attention partials over local pairs and DP2 host."""
    # Reduce-scatter the group's rows: pair q of each local host owns rows
    # q * r .. q * r + r - 1. Staged slots are whole 16-row BF16 tiles
    # (DP4 B96's six rows per pair are 168 payload rows).
    slot_rows = (
        payload_rows if ctx.staging is None else _staging_slot_rows(payload_rows)
    )
    local_send = (
        (
            ctx.local.at[0].bitcast(jnp.bfloat16)
            if ctx.staging is None
            else ctx.staging.at[0]
        )
        .at[pl.ds(0, _LOCAL_RANK_PAIRS * slot_rows), :]
        .reshape(_LOCAL_RANK_PAIRS, slot_rows, VREG_LANE_COUNT)
    )
    local_receive = (
        (
            ctx.local.at[1].bitcast(jnp.bfloat16)
            if ctx.staging is None
            else ctx.staging.at[1]
        )
        .at[pl.ds(0, _LOCAL_RANK_PAIRS * slot_rows), :]
        .reshape(_LOCAL_RANK_PAIRS, slot_rows, VREG_LANE_COUNT)
    )
    for target_pair in range(_LOCAL_RANK_PAIRS):
        row_start = (
            target_pair * ctx.rows_per_pair if ctx.owners == _LOCAL_RANK_PAIRS else 0
        )
        wire = (
            ctx.comm[pl.ds(row_start, ctx.rows_per_pair), : ctx.half]
            .astype(jnp.bfloat16)
            .reshape(payload_rows, VREG_LANE_COUNT)
        )
        if slot_rows != payload_rows:
            wire = jnp.concatenate(
                [
                    wire,
                    jnp.zeros(
                        (slot_rows - payload_rows, VREG_LANE_COUNT),
                        jnp.bfloat16,
                    ),
                ],
            )
        local_send[target_pair] = wire
    local_receive[ctx.pair] = local_send[ctx.pair]
    copies: list[AsyncCopyProtocol] = []
    for offset in range(_LOCAL_RANK_PAIRS - 1, 0, -1):
        copy = tpu.make_async_remote_copy(
            local_send.at[ctx.pair ^ offset],
            local_receive.at[ctx.pair],
            ctx.sends.at[offset - 1],
            ctx.recvs.at[offset - 1],
            device_id=(ctx.rank ^ (_RANK_PAIR_SIZE * offset),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
    local_sum = jnp.sum(local_receive[...].astype(jnp.float32), axis=0)
    if slot_rows != payload_rows:
        local_sum = local_sum[:payload_rows]

    if ctx.shards == DP2_GROUPS:
        # TP16 spans two hosts; TP8 finishes its reduction on one host.
        host_receive = ctx.host_reduction.at[:DP2_GROUPS, pl.ds(0, payload_rows), :]
        host_receive[ctx.host % DP2_GROUPS] = local_sum
        host_view = host_receive.at[ctx.host % DP2_GROUPS]
        copy = tpu.make_async_remote_copy(
            host_view,
            host_view,
            ctx.sends.at[_ATTENTION_HOST_SEM_BASE],
            ctx.recvs.at[_ATTENTION_HOST_SEM_BASE],
            device_id=(ctx.rank ^ mla_dcp.HOST_LANES,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copy.wait()
        return jnp.sum(host_receive[...], axis=0).astype(jnp.bfloat16)
    return local_sum.astype(jnp.bfloat16)


def _assemble_dp_rank_pair_rows(
    ctx: _AttentionDpContext,
    reduced_half: jax.Array,
    payload_rows: int,
    rows_per_pair: int,
    out: PallasRefProtocol,
) -> None:
    """Swap reduced halves across the rank pair and write full rows to ``out``."""
    pair_send = ctx.local.at[2].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    pair_receive = ctx.local.at[3].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    pair_send[...] = reduced_half
    copy = tpu.make_async_remote_copy(
        pair_send,
        pair_receive,
        ctx.sends.at[_ATTENTION_PAIR_SWAP_SEM],
        ctx.recvs.at[_ATTENTION_PAIR_SWAP_SEM],
        device_id=(ctx.rank ^ 1,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()
    copy.wait()
    sent = pair_send[...].reshape(rows_per_pair, ctx.half)
    received_half = pair_receive[...].reshape(rows_per_pair, ctx.half)
    row = jnp.concatenate(
        (
            jnp.where(ctx.lane == 0, sent, received_half),
            jnp.where(ctx.lane == 0, received_half, sent),
        ),
        axis=1,
    )
    if out.shape != (rows_per_pair, ctx.hidden):
        msg = (
            f"DP attention output shape {out.shape} does not match "
            f"{(rows_per_pair, ctx.hidden)}"
        )
        raise ValueError(msg)
    out[...] = row


def _project_attention_dp(
    attended: PallasRefProtocol,
    weight: PallasRefProtocol,
    out: PallasRefProtocol,
    comm: PallasRefProtocol,
    local: PallasRefProtocol,
    host_reduction: PallasRefProtocol,
    feature_backing: PallasRefProtocol,
    sends: PallasRefProtocol,
    recvs: PallasRefProtocol,
    phase: PallasRefProtocol,
    dot: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    weight_stream: ChunkedWeightStream | None = None,
    staging: PallasRefProtocol | None = None,
    owners: int = _LOCAL_RANK_PAIRS,
) -> None:
    """Project DP2/TP16 or DP4/TP8 attention back to DP row ownership.

    ``attended`` [dp, 4r, 384] holds each logical TP32 head shard for the
    DP group's rows. The projection first reduces over four local rank
    pairs; DP2 also reduces over its second host. Rank pairs assemble the
    hidden dimension, so ``out`` [r, 7168] holds this rank pair's r rows of
    its DP group.
    With one row owner (DP2 B2/B4, ``attended`` [2, r, 384]) each pair's
    partial of its group's r rows goes to the 15 same-lane ranks of both
    groups in one hop, so ``out`` [2r, 7168] is the whole batch on every rank.
    Output weight chunks are all own-lane shard halves, then peer halves.
    """
    rank = jax.lax.axis_index(AXIS_TP)
    slot = phase[0] % _DOUBLE_BUFFER_SLOTS
    shards, rows, width = attended.shape
    hidden = out.shape[-1]
    half = hidden // _RANK_PAIR_SIZE
    if (
        shards not in _SUPPORTED_DP_DEGREES
        or owners not in _SUPPORTED_DP_OWNERS
        or rows % owners != 0
    ):
        msg = "Attention DP projection requires shards in (2, 4), owners in (1, 4)"
        raise ValueError(msg)
    if owners != _LOCAL_RANK_PAIRS and shards != DP2_GROUPS:
        msg = "Single-owner DP attention projection requires DP2 (shards == 2)"
        raise ValueError(msg)
    lane = rank % _RANK_PAIR_SIZE
    pair = (rank % mla_dcp.HOST_LANES) // _RANK_PAIR_SIZE
    host = rank // mla_dcp.HOST_LANES
    rows_per_pair = rows // owners
    if weight_stream is None and weight.shape != (
        shards * _RANK_PAIR_SIZE * width,
        half,
    ):
        msg = "Unstreamed DP attention weight shape does not match shard width"
        raise ValueError(msg)
    guard = phase[0] >= 0

    ctx = _AttentionDpContext(
        rank=rank,
        slot=slot,
        lane=lane,
        pair=pair,
        host=host,
        guard=guard,
        shards=shards,
        rows=rows,
        width=width,
        hidden=hidden,
        half=half,
        owners=owners,
        rows_per_pair=rows_per_pair,
        weight=weight,
        comm=comm,
        local=local,
        host_reduction=host_reduction,
        sends=sends,
        recvs=recvs,
        dot=dot,
        weight_stream=weight_stream,
        staging=staging,
    )
    _project_dp_shard_features(ctx, attended, feature_backing)
    payload_rows = rows_per_pair * half // VREG_LANE_COUNT
    if owners == 1:
        reduced_half, payload_rows, rows_per_pair = _reduce_dp_single_owner_half(
            ctx,
            payload_rows,
        )
    else:
        reduced_half = _reduce_scatter_dp_pair_half(ctx, payload_rows)
    _assemble_dp_rank_pair_rows(ctx, reduced_half, payload_rows, rows_per_pair, out)
    phase[0] += 1


@dataclasses.dataclass(frozen=True, slots=True)
class _DpMoeGatherContext:
    """Shared wire views and semaphores for overlapped DP MoE row gathers."""

    rank: jax.Array
    group_size: int
    shards: int
    payload: int
    base: int
    half: int
    start: int | jax.Array
    wires: PallasRefProtocol
    sends: PallasRefProtocol
    recvs: PallasRefProtocol
    pair_semaphores: tuple[PallasRefProtocol, PallasRefProtocol] | None


def _dp_moe_wire(
    ctx: _DpMoeGatherContext,
    group_offset: int,
    pair_offset: int,
    *,
    whole: bool = False,
) -> PallasRefProtocol:
    """Return the full or pair-split wire slice for relative ``(o, k)``."""
    slot_index = ctx.base + _LOCAL_RANK_PAIRS * group_offset + pair_offset
    if ctx.pair_semaphores is None or whole:
        return ctx.wires.at[slot_index, pl.ds(0, ctx.payload)]
    return ctx.wires.at[slot_index, pl.ds(ctx.start, ctx.half)]


def _send_dp_moe_wire(
    ctx: _DpMoeGatherContext,
    source: PallasRefProtocol,
    group_offset: int,
    pair_offset: int,
    semaphore: int,
) -> AsyncCopyProtocol:
    """Start a remote send to relative peer ``(group_offset, pair_offset)``."""
    # Peer (o, k) sees this rank at the same relative position.
    copy = tpu.make_async_remote_copy(
        source,
        _dp_moe_wire(ctx, group_offset, pair_offset),
        ctx.sends.at[semaphore],
        ctx.recvs.at[semaphore],
        device_id=(
            ctx.rank
            ^ (ctx.group_size * group_offset)
            ^ (_RANK_PAIR_SIZE * pair_offset),
        ),
        device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()
    return copy


def _exchange_dp_moe_wires(ctx: _DpMoeGatherContext, rows: int) -> None:
    """Run direct or forwarded DP MoE wire exchange and optional pair swap."""
    origin = _dp_moe_wire(ctx, 0, 0)
    if rows <= _DIRECT_GATHER_MAX_ROWS and ctx.shards == DP2_GROUPS:
        # Short wires (DP2 B8/B16) go to every peer directly: one hop.
        direct = [
            _send_dp_moe_wire(
                ctx,
                origin,
                o,
                k,
                _DP_MOE_LOCAL_SEM if o == 0 else _DP_MOE_DIRECT_GROUP_SEM,
            )
            for o in range(ctx.shards)
            for k in range(_LOCAL_RANK_PAIRS)
            if o or k
        ]
        for copy in direct:
            copy.wait()
    else:
        local_copies = [
            _send_dp_moe_wire(ctx, origin, 0, k, _DP_MOE_LOCAL_SEM)
            for k in range(1, _LOCAL_RANK_PAIRS)
        ]
        group_copies = [
            _send_dp_moe_wire(ctx, origin, o, 0, _DP_MOE_GROUP_SEMS[o - 1])
            for o in range(1, ctx.shards)
        ]
        forwarded: list[AsyncCopyProtocol] = []
        for o, copy in enumerate(group_copies, 1):
            copy.wait_recv()
            for k in range(1, _LOCAL_RANK_PAIRS):
                fwd = tpu.make_async_remote_copy(
                    _dp_moe_wire(ctx, o, 0),
                    _dp_moe_wire(ctx, o, k),
                    ctx.sends.at[_DP_MOE_FORWARD_SEM],
                    ctx.recvs.at[_DP_MOE_FORWARD_SEM],
                    device_id=(ctx.rank ^ (_RANK_PAIR_SIZE * k),),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                fwd.start()
                forwarded.append(fwd)
        for copy in group_copies:
            copy.wait_send()
        for copy in (*local_copies, *forwarded):
            copy.wait()
    if ctx.pair_semaphores is not None:
        send_sem, recv_sem = ctx.pair_semaphores
        swaps: list[AsyncCopyProtocol] = []
        for slot in range(1, _LOCAL_RANK_PAIRS * ctx.shards):
            o, k = divmod(slot, _LOCAL_RANK_PAIRS)
            wire_slice = _dp_moe_wire(ctx, o, k)
            swap = tpu.make_async_remote_copy(
                wire_slice,
                wire_slice,
                send_sem,
                recv_sem,
                device_id=(ctx.rank ^ 1,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            swap.start()
            swaps.append(swap)
        for copy in swaps:
            copy.wait()


def _gather_dp_moe_rows(
    row: jax.Array,
    owned_out: PallasRefProtocol,
    out: PallasRefProtocol,
    local: PallasRefProtocol,
    sends: PallasRefProtocol,
    recvs: PallasRefProtocol,
    shards: int,
    pair_semaphores: tuple[PallasRefProtocol, PallasRefProtocol] | None = None,
) -> None:
    """Gather the DP groups' rows and the local pairs' rows, overlapped.

    Each rank holds [batch / (4 * DP), hidden] for its (DP group, local pair)
    and ends with the whole batch in ``out`` and its pair's rows of all groups
    in ``owned_out``. Own rows go to the local pairs and the other groups
    together, so the inter-group exchange does not delay the local one; each
    group's wire is forwarded as soon as it lands. Flat wires use lanes 4..7
    of ``local``: the wire from relative position (group ^ o, pair ^ k) lands
    in slot (o, k) on every rank. With ``pair_semaphores`` the pair's two
    ranks (one chip) each carry half of every wire and then swap halves.
    """
    rank = jax.lax.axis_index(AXIS_TP)
    group_size = TP_SIZE // shards
    group = rank // group_size
    pair = (rank % mla_dcp.HOST_LANES) // _RANK_PAIR_SIZE
    rows, hidden = row.shape
    payload = rows * hidden // VREG_LANE_COUNT
    # A one-row wire (B8 DP2: 56 BF16 rows) starts its slot on a whole tile.
    stride = -(-payload // _BF16_TILE_ROWS) * _BF16_TILE_ROWS
    slot_count = _BF16_PER_FP32_WORD * local.shape[0] * local.shape[1] // stride
    base = slot_count // _RANK_PAIR_SIZE
    if base < _LOCAL_RANK_PAIRS * shards:
        msg = "Local transport buffer lacks slots for DP MoE row gather"
        raise ValueError(msg)
    wires = cast(
        "PallasRefProtocol",
        pool_alias.view(local, (slot_count, stride, VREG_LANE_COUNT), jnp.bfloat16),
    )
    half = 0
    start: int | jax.Array = 0
    if pair_semaphores is not None:
        # Halves are whole 16-row tiles; for a payload that is an odd number
        # of tiles, they share one.
        half = -(-payload // (_RANK_PAIR_SIZE * _BF16_TILE_ROWS)) * _BF16_TILE_ROWS
        start = pl.multiple_of(
            (rank % _RANK_PAIR_SIZE) * (payload - half),
            _BF16_TILE_ROWS,
        )

    ctx = _DpMoeGatherContext(
        rank=rank,
        group_size=group_size,
        shards=shards,
        payload=payload,
        base=base,
        half=half,
        start=start,
        wires=wires,
        sends=sends,
        recvs=recvs,
        pair_semaphores=pair_semaphores,
    )
    _dp_moe_wire(ctx, 0, 0, whole=True)[...] = row.reshape(payload, VREG_LANE_COUNT)
    _exchange_dp_moe_wires(ctx, rows)
    for d in range(shards):
        for p in range(_LOCAL_RANK_PAIRS):
            slot = base + _LOCAL_RANK_PAIRS * (group ^ d) + (pair ^ p)
            first = (p * shards + d) * rows
            out[first : first + rows, :] = wires[slot, pl.ds(0, payload)].reshape(
                rows,
                hidden,
            )
        owned_out[d * rows : (d + 1) * rows, :] = wires[
            base + _LOCAL_RANK_PAIRS * (group ^ d),
            pl.ds(0, payload),
        ].reshape(rows, hidden)


def _router_copy(
    weight: PallasRefProtocol,
    buffer: PallasRefProtocol,
    sems: PallasRefProtocol,
    wait: int,
    *,
    initialize_padding: bool = True,
) -> None:
    """Start or wait on the rank-pair split copy of the MoE router weight."""
    lane = jax.lax.axis_index(AXIS_TP) % _RANK_PAIR_SIZE

    @pl.when(lane < 1)
    def full() -> None:
        """Transfer lane-0's 512 router columns."""
        copy = tpu.make_async_copy(
            weight.at[
                :, pl.ds(lane * _ROUTER_LANE_SCORE_WIDTH, _ROUTER_LANE_SCORE_WIDTH)
            ],
            buffer.at[0],
            sems.at[0],
        )
        if wait:
            copy.wait()
        else:
            copy.start()

    @pl.when(lane == 1)
    def tail() -> None:
        """Transfer lane-1's 384 router columns and optionally zero padding."""
        if not wait and initialize_padding:
            buffer[0, :, _ODD_LANE_ROUTER_COLUMNS:] = jnp.zeros(
                (weight.shape[0], VREG_LANE_COUNT),
                jnp.bfloat16,
            )
        copy = tpu.make_async_copy(
            weight.at[:, pl.ds(_ROUTER_LANE_SCORE_WIDTH, _ODD_LANE_ROUTER_COLUMNS)],
            buffer.at[0, :, :_ODD_LANE_ROUTER_COLUMNS],
            sems.at[0],
        )
        if wait:
            copy.wait()
        else:
            copy.start()


@dataclasses.dataclass(frozen=True, slots=True)
class _DenseScatterContext:
    """Topology coordinates and buffers for dense layer-0 reduce-scatter."""

    rank: jax.Array
    lane: jax.Array
    host: jax.Array
    owned_rows: int
    hidden: int
    wire_rows: int
    payload: int
    owners: int
    partial: TensorLike
    output: PallasRefProtocol
    local: PallasRefProtocol
    hosts: PallasRefProtocol
    sends: PallasRefProtocol
    recvs: PallasRefProtocol


def _encode_packed_bf16_wire(value: jax.Array, wire_rows: int) -> jax.Array:
    """Pack BF16 rows into FP32 bit-reinterpreted transport words."""
    packed = tpu.bitcast(
        value.reshape(wire_rows, VREG_LANE_COUNT).astype(jnp.bfloat16),
        jnp.uint32,
    )
    return jax.lax.bitcast_convert_type(packed, jnp.float32)


def _decode_packed_bf16_sum(
    value: jax.Array,
    rows: int,
    hidden: int,
) -> jax.Array:
    """Unpack FP32-reinterpreted BF16 packets and sum across axis 0 in FP32."""
    unpacked = tpu.bitcast(
        jax.lax.bitcast_convert_type(value, jnp.uint32),
        jnp.bfloat16,
    ).astype(jnp.float32)
    return jnp.sum(unpacked, axis=0).reshape(rows, hidden)


def _run_dense_scatter_rounds(
    ctx: _DenseScatterContext,
    wires: PallasRefProtocol,
) -> None:
    """Run eight-row dense reduce-scatter rounds to each token owner."""
    lanes_per_owner = mla_dcp.HOST_LANES // ctx.owners
    for first in range(0, ctx.owned_rows, _DENSE_SCATTER_ROUND_ROWS):
        if first:
            collectives32.barrier()
        for owner in range(ctx.owners):
            start = owner * ctx.owned_rows + first
            wires[owner] = _encode_packed_bf16_wire(
                ctx.partial[start : start + _DENSE_SCATTER_ROUND_ROWS],
                ctx.wire_rows,
            )
        ctx.local[ctx.lane, : ctx.payload] = wires[ctx.lane // lanes_per_owner]
        local_copies: list[AsyncCopyProtocol] = []
        for offset in range(mla_dcp.HOST_LANES - 1, 0, -1):
            copy = tpu.make_async_remote_copy(
                wires.at[(ctx.lane ^ offset) // lanes_per_owner],
                ctx.local.at[ctx.lane, : ctx.payload],
                ctx.sends.at[offset - 1],
                ctx.recvs.at[offset - 1],
                device_id=(ctx.rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            local_copies.append(copy)
        for copy in local_copies:
            copy.wait()
        ctx.hosts[ctx.host, : ctx.payload] = _encode_packed_bf16_wire(
            _decode_packed_bf16_sum(
                ctx.local[:, : ctx.payload],
                _DENSE_SCATTER_ROUND_ROWS,
                ctx.hidden,
            ),
            ctx.wire_rows,
        )
        host_copies: list[AsyncCopyProtocol] = []
        for offset in range(_HOSTS_IN_SLICE - 1, 0, -1):
            view = ctx.hosts.at[ctx.host, : ctx.payload]
            copy = tpu.make_async_remote_copy(
                view,
                view,
                ctx.sends.at[_DENSE_HOST_SEM_BASE + offset],
                ctx.recvs.at[_DENSE_HOST_SEM_BASE + offset],
                device_id=(ctx.rank ^ (mla_dcp.HOST_LANES * offset),),
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            host_copies.append(copy)
        for copy in host_copies:
            copy.wait()
        ctx.output[first : first + _DENSE_SCATTER_ROUND_ROWS] = _decode_packed_bf16_sum(
            ctx.hosts[:, : ctx.payload],
            _DENSE_SCATTER_ROUND_ROWS,
            ctx.hidden,
        ).astype(jnp.bfloat16)


def _dense_scatter_rows(
    partial: TensorLike,
    output: PallasRefProtocol,
    local: PallasRefProtocol,
    hosts: PallasRefProtocol,
    sends: PallasRefProtocol,
    recvs: PallasRefProtocol,
    owners: int,
) -> None:
    """Reduce the dense layer directly to its token owners.

    Preserve the original BF16 wire format, local eight-rank sum and four-host
    sum, including their rounding points. Each owner consumes eight rows per
    round instead of every rank reducing the entire batch and discarding most
    rows. Pack each owner's source once; the two ranks in each owner pair send
    and receive identical logical row ranges.
    """
    rank = jax.lax.axis_index(AXIS_TP)
    lane = rank % mla_dcp.HOST_LANES
    host = rank // mla_dcp.HOST_LANES
    owned_rows = partial.shape[0] // owners
    hidden = partial.shape[1]
    wire_rows = _DENSE_SCATTER_ROUND_ROWS * hidden // VREG_LANE_COUNT
    payload = wire_rows // _BF16_PER_FP32_WORD
    if owned_rows % _DENSE_SCATTER_ROUND_ROWS != 0 or mla_dcp.HOST_LANES % owners != 0:
        msg = "Dense scatter requires owned_rows % 8 == 0 and 8 % owners == 0"
        raise ValueError(msg)
    if payload > local.shape[1] or payload > hosts.shape[1]:
        msg = "Dense scatter payload exceeds reduction transport capacity"
        raise ValueError(msg)

    ctx = _DenseScatterContext(
        rank=rank,
        lane=lane,
        host=host,
        owned_rows=owned_rows,
        hidden=hidden,
        wire_rows=wire_rows,
        payload=payload,
        owners=owners,
        partial=partial,
        output=output,
        local=local,
        hosts=hosts,
        sends=sends,
        recvs=recvs,
    )

    def reduce_owned(wires: PallasRefProtocol) -> None:
        """Execute dense scatter rounds using scoped wire staging."""
        _run_dense_scatter_rounds(ctx, wires)

    pl.run_scoped(
        reduce_owned,
        tpu.VMEM((owners, payload, VREG_LANE_COUNT), jnp.float32),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _AllReduceStageSpec:
    """Semaphore and peer-stride metadata for one all-reduce exchange stage."""

    rank: jax.Array
    payload: int
    offsets: range
    stride: int
    semaphore_base: int
    sends: PallasRefProtocol
    recvs: PallasRefProtocol


def _exchange_all_reduce_stage(
    buffer: PallasRefProtocol,
    index: jax.Array,
    stage: _AllReduceStageSpec,
) -> None:
    """Exchange one packed BF16 all-reduce stage across peers."""
    view = buffer.at[index, : stage.payload, :]
    copies: list[AsyncCopyProtocol] = []
    for offset in stage.offsets:
        copy = tpu.make_async_remote_copy(
            view,
            view,
            stage.sends.at[stage.semaphore_base + offset],
            stage.recvs.at[stage.semaphore_base + offset],
            device_id=(stage.rank ^ (stage.stride * offset),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()


def _all_reduce_rows(
    partial: jax.Array,
    local: PallasRefProtocol,
    hosts: PallasRefProtocol,
    sends: PallasRefProtocol,
    recvs: PallasRefProtocol,
) -> jax.Array:
    """TP32 sum of an FP32 ``[batch, hidden]`` partial on every rank.

    The shared-expert transport of the mixture-of-experts operation: the
    partial is rounded to BF16 and packed in pairs into 32-bit words, each
    lane sends its packet to the seven other local lanes, the local sum is
    packed again and exchanged among the four hosts. ``local`` and ``hosts``
    are the wide reduction buffers of the communication workspace
    (``[8, rows, 128]`` and ``[4, rows, 128]`` FP32); semaphores 0..6 and
    7..9 are used and left idle.
    """
    rank = jax.lax.axis_index(AXIS_TP)
    lane = rank % mla_dcp.HOST_LANES
    host = rank // mla_dcp.HOST_LANES
    batch_size, hidden = partial.shape
    rows = batch_size * hidden // VREG_LANE_COUNT
    payload = rows // _BF16_PER_FP32_WORD
    if (
        rows % _BF16_PER_FP32_WORD != 0
        or payload > local.shape[1]
        or payload > hosts.shape[1]
    ):
        msg = "All-reduce payload must be even and fit local/host reduction buffers"
        raise ValueError(msg)

    local.at[lane][:payload, :] = _encode_packed_bf16_wire(partial, rows)
    _exchange_all_reduce_stage(
        local,
        lane,
        _AllReduceStageSpec(
            rank=rank,
            payload=payload,
            offsets=range(mla_dcp.HOST_LANES - 1, 0, -1),
            stride=1,
            semaphore_base=_ALL_REDUCE_LOCAL_SEM_BASE,
            sends=sends,
            recvs=recvs,
        ),
    )
    hosts.at[host][:payload, :] = _encode_packed_bf16_wire(
        _decode_packed_bf16_sum(local[:, :payload, :], batch_size, hidden),
        rows,
    )
    _exchange_all_reduce_stage(
        hosts,
        host,
        _AllReduceStageSpec(
            rank=rank,
            payload=payload,
            offsets=range(_HOSTS_IN_SLICE - 1, 0, -1),
            stride=mla_dcp.HOST_LANES,
            semaphore_base=_ALL_REDUCE_HOST_SEM_BASE,
            sends=sends,
            recvs=recvs,
        ),
    )
    return _decode_packed_bf16_sum(hosts[:, :payload, :], batch_size, hidden)


def _gather_sp8_local_rows(
    row: jax.Array,
    out: PallasRefProtocol,
    local: PallasRefProtocol,
    sends: PallasRefProtocol,
    recvs: PallasRefProtocol,
) -> None:
    """Gather eight local-rank row shards directly into the pre-attention buffer."""
    rank = jax.lax.axis_index(AXIS_TP)
    owner = rank % mla_dcp.HOST_LANES
    rows, hidden = row.shape
    payload = rows * hidden // VREG_LANE_COUNT
    if rows != _SP8_OWNER_ROWS or payload % _BF16_TILE_ROWS != 0:
        msg = "SP8 local row gather requires 8 rows and a 16-row-aligned payload"
        raise ValueError(msg)
    slots = _BF16_PER_FP32_WORD * local.shape[0] * local.shape[1] // payload
    if slots < _MIN_SP8_LOCAL_SLOTS:
        msg = "Local transport buffer lacks 32 wire slots for SP8 local gather"
        raise ValueError(msg)
    wires = cast(
        "PallasRefProtocol",
        pool_alias.view(local, (slots, payload, VREG_LANE_COUNT), jnp.bfloat16),
    )
    source = wires.at[owner]
    source[...] = row.reshape(payload, VREG_LANE_COUNT)
    copies: list[AsyncCopyProtocol] = []
    for offset in range(mla_dcp.HOST_LANES - 1, 0, -1):
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
    for pair in range(_LOCAL_RANK_PAIRS):
        out[pair * _SP8_PAIR_ROWS : (pair + 1) * _SP8_PAIR_ROWS] = jnp.concatenate(
            (
                wires[_RANK_PAIR_SIZE * pair].reshape(rows, hidden),
                wires[_RANK_PAIR_SIZE * pair + 1].reshape(rows, hidden),
            ),
            axis=0,
        )


def _gather_sp8_dp_moe_rows(
    row: jax.Array,
    owned_out: PallasRefProtocol,
    out: PallasRefProtocol,
    local: PallasRefProtocol,
    sends: PallasRefProtocol,
    recvs: PallasRefProtocol,
) -> None:
    """Gather DP2 MoE-input rows from eight owners in pair-major ABI order.

    Payload slots are relative (DP group XOR, local core XOR). The upper
    half of the existing transport allocation is disjoint from pre gather.
    No separate pair reconstruction collective is needed.
    """
    rank = jax.lax.axis_index(AXIS_TP)
    group = rank // mla_dcp.DP2_GROUP
    owner = rank % mla_dcp.HOST_LANES
    rows, hidden = row.shape
    payload = rows * hidden // VREG_LANE_COUNT
    if rows != _SP8_OWNER_ROWS or payload % _BF16_TILE_ROWS != 0:
        msg = "SP8 DP2 MoE gather requires 8 rows and a 16-row-aligned payload"
        raise ValueError(msg)
    slots = _BF16_PER_FP32_WORD * local.shape[0] * local.shape[1] // payload
    base = slots // _RANK_PAIR_SIZE
    if base < _DP2_SP8_WIRE_SLOTS:
        msg = "Local transport buffer lacks 16 upper wire slots for SP8 DP2 gather"
        raise ValueError(msg)
    wires = cast(
        "PallasRefProtocol",
        pool_alias.view(local, (slots, payload, VREG_LANE_COUNT), jnp.bfloat16),
    )

    def wire(o: jax.Array | int, k: jax.Array | int) -> PallasRefProtocol:
        """Return wire view for relative ``(group_offset, core_offset)``."""
        return wires.at[base + mla_dcp.HOST_LANES * o + k]

    def send(
        source: PallasRefProtocol,
        o: int,
        k: int,
        semaphore: int,
        *,
        local_only: bool = False,
    ) -> AsyncCopyProtocol:
        """Start remote wire copy for relative ``(o, k)``."""
        copy = tpu.make_async_remote_copy(
            source,
            wire(o, k),
            sends.at[semaphore],
            recvs.at[semaphore],
            device_id=(rank ^ (0 if local_only else mla_dcp.DP2_GROUP * o) ^ k,),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        return copy

    wire(0, 0)[...] = row.reshape(payload, VREG_LANE_COUNT)
    local_copies = [
        send(wire(0, 0), 0, k, _DP_MOE_LOCAL_SEM) for k in range(1, mla_dcp.HOST_LANES)
    ]
    group_copy = send(wire(0, 0), 1, 0, _DP_MOE_DIRECT_GROUP_SEM)
    group_copy.wait_recv()
    forwarded = [
        send(wire(1, 0), 1, k, _DP_MOE_FORWARD_SEM, local_only=True)
        for k in range(1, mla_dcp.HOST_LANES)
    ]
    group_copy.wait_send()
    for copy in (*local_copies, *forwarded):
        copy.wait()
    for d in range(DP2_GROUPS):
        for p in range(_LOCAL_RANK_PAIRS):
            even = wire(group ^ d, owner ^ (_RANK_PAIR_SIZE * p))[...].reshape(
                rows,
                hidden,
            )
            odd = wire(group ^ d, owner ^ (_RANK_PAIR_SIZE * p + 1))[...].reshape(
                rows,
                hidden,
            )
            first = (_RANK_PAIR_SIZE * p + d) * _SP8_PAIR_ROWS
            out[first : first + _SP8_PAIR_ROWS] = jnp.concatenate(
                (even, odd),
                axis=0,
            )
        own = wire(group ^ d, 0)[...].reshape(rows, hidden)
        peer = wire(group ^ d, 1)[...].reshape(rows, hidden)
        owned_out[d * _SP8_PAIR_ROWS : (d + 1) * _SP8_PAIR_ROWS] = jnp.where(
            owner % _RANK_PAIR_SIZE == 0,
            jnp.concatenate((own, peer), axis=0),
            jnp.concatenate((peer, own), axis=0),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _LocalPairGatherSpec:
    """Topology and semaphore state for local-owner row all-gathers."""

    rank: jax.Array
    owner: jax.Array
    stride: int
    owners: int
    sends: PallasRefProtocol
    recvs: PallasRefProtocol


def _gather_packed_local_pair_rows(
    spec: _LocalPairGatherSpec,
    source: PallasRefProtocol,
    destination: PallasRefProtocol,
    scratch: PallasRefProtocol,
) -> None:
    """All-gather local owner shards through flat transport wires."""
    rows_per_owner = source.shape[0]
    payload_rows = rows_per_owner * source.shape[1] // VREG_LANE_COUNT
    source_wire = (
        scratch.at[spec.owner].bitcast(jnp.bfloat16).at[pl.ds(0, payload_rows), :]
    )
    source_wire[...] = source[...].reshape(payload_rows, VREG_LANE_COUNT)
    copies: list[AsyncCopyProtocol] = []
    for offset in range(spec.owners - 1, 0, -1):
        copy = tpu.make_async_remote_copy(
            source_wire,
            source_wire,
            spec.sends.at[offset - 1],
            spec.recvs.at[offset - 1],
            device_id=(spec.rank ^ (spec.stride * offset),),
            device_id_type=pl.DeviceIdType.MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()
    wires = tuple(
        scratch.at[source_owner].bitcast(jnp.bfloat16)[:payload_rows]
        for source_owner in range(spec.owners)
    )
    if rows_per_owner == 1:
        destination[...] = jnp.stack(
            tuple(wire.reshape(source.shape[1]) for wire in wires),
        )
    else:
        destination[...] = jnp.concatenate(
            tuple(wire.reshape(rows_per_owner, source.shape[1]) for wire in wires),
            axis=0,
        )


def _gather_local_pair_rows(
    source: PallasRefProtocol,
    destination: PallasRefProtocol,
    sends: PallasRefProtocol,
    recvs: PallasRefProtocol,
    scratch: PallasRefProtocol | None = None,
    owners: int = _LOCAL_RANK_PAIRS,
    *,
    packed: bool = False,
) -> None:
    """All-gather the token shards of the ``owners`` local row owners.

    Owner o (4: the local rank pairs; 8: the local ranks) holds rows
    o * r .. o * r + r - 1 in ``source`` [r, hidden]; every rank ends with all
    of them in ``destination``. Each rank sends to its counterpart in every
    other owner (rank ^ k * 8 / owners). ``packed`` stages through flat
    transport wires when a source row slice crosses a native BF16 tile.
    """
    rank = jax.lax.axis_index(AXIS_TP)
    stride = mla_dcp.HOST_LANES // owners
    owner = (rank % mla_dcp.HOST_LANES) // stride
    rows_per_owner = source.shape[0]
    spec = _LocalPairGatherSpec(
        rank=rank,
        owner=owner,
        stride=stride,
        owners=owners,
        sends=sends,
        recvs=recvs,
    )
    # One row, or a row count whose slice is not tile-aligned for a DMA
    # (3, 5, 6, 7 rows at B12..B28): send a flattened wire through the
    # transport scratch and assemble the destination from values.
    if packed or rows_per_owner == 1 or rows_per_owner & (rows_per_owner - 1):
        if scratch is None:
            msg = "Packed local-pair row gather requires transport scratch"
            raise ValueError(msg)
        _gather_packed_local_pair_rows(spec, source, destination, scratch)
        return
    for owner_index in range(owners):

        @pl.when(owner == owner_index)
        def gather_owner(owner_index: int = owner_index) -> None:
            """Copy this owner's tile-aligned rows to all local counterparts."""
            row_slice = pl.ds(owner_index * rows_per_owner, rows_per_owner)
            destination[row_slice] = source[...]
            copies: list[AsyncCopyProtocol] = []
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
# Layer 0 dense MLP weights, used when the stack includes layer 0.
DENSE = ("dense_gu", "dense_down")
NAMES = MOE + KDA + MLA + NORMS


def _residual_mixture_row(
    block_values: jax.Array,
    prefix_value: jax.Array,
    folded: TensorLike,
    active_block_count: jax.Array | int,
    *,
    epsilon: float,
    relaxed_reduction_order: bool = False,
) -> jax.Array:
    """Compute one row of the learned residual mixture."""

    def normalized_scores(values: jax.Array) -> jax.Array:
        """Compute RMS-normalized residual mixture logits for ``values``."""
        scores = jnp.sum(values * folded[...], axis=1, keepdims=True)
        return scores * jax.lax.rsqrt(
            jnp.mean(values * values, axis=1, keepdims=True) + epsilon,
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
        values: jax.Array | None = None
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
            jnp.mean(values * values, axis=1, keepdims=True) + epsilon,
        )

    residual_indices = jax.lax.broadcasted_iota(
        jnp.int32,
        (_RESIDUAL_MIXTURE_SLOTS, 1),
        0,
    )
    active = (residual_indices < active_block_count) | (
        residual_indices == _PREFIX_RESIDUAL_SLOT
    )
    probabilities = jax.nn.softmax(
        jnp.where(active, scores, -jnp.inf),
        axis=0,
    )
    if relaxed_reduction_order:
        mixed = (
            jnp.sum(
                block_values * probabilities[:_PREFIX_RESIDUAL_SLOT],
                axis=0,
                keepdims=True,
            )
            + prefix_value
            * probabilities[_PREFIX_RESIDUAL_SLOT:_RESIDUAL_MIXTURE_SLOTS]
        )
    else:
        if values is None:
            msg = "Combined residual values must be initialized in reference mode"
            raise ValueError(msg)
        mixed = jnp.sum(values * probabilities, axis=0, keepdims=True)
    return mixed.astype(jnp.bfloat16)


def _residual_mixture_by_row(
    blocks: jax.Array,
    prefix: jax.Array,
    destination: PallasRefProtocol,
    folded: TensorLike,
    active_block_count: jax.Array | int,
    *,
    batch_size: int,
    epsilon: float,
    relaxed_reduction_order: bool = False,
) -> None:
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
COMPACT_ROWS = _DEFAULT_COMPACT_ROWS
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
if MXFP4_DOT_TILE is not None and (
    MXFP4_DOT_TILE < 0 or MXFP4_DOT_TILE % VREG_LANE_COUNT
):
    msg = (
        "KernelConfig.mxfp4_dot_tile must be zero or a positive multiple of "
        f"{VREG_LANE_COUNT}, got {MXFP4_DOT_TILE}"
    )
    raise ValueError(msg)
MXFP4_BLOCK = _config().mxfp4_block
MXFP4_PAIR_TILE = _config().mxfp4_pair_tile
# KernelConfig.mxfp4_static_tile=T: unroll the single-expert MXFP4 product over
# static T-row tiles instead of a dynamic loop, keeping the unpacked tile small.
MXFP4_STATIC_TILE = _config().mxfp4_static_tile


# B16 runs routed experts on compact 8-row tiles.
B16_COMPACT_EXPERTS = _config().b16_compact_experts


def _compact_rows(batch_size: int) -> int:
    """Return compact routed-expert tile row capacity for ``batch_size``."""
    if batch_size == _B16_BATCH_SIZE and B16_COMPACT_EXPERTS and not COMPACT_ONLY:
        return _DEFAULT_COMPACT_ROWS
    return COMPACT_ROWS if batch_size >= _COMPACT_EXPERT_MIN_BATCH else 0


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
# Future work: derive separate staging/rolling rules for B32/B64 with 8-row chunks.
KDA_STAGED = _config().kda_staged
# Diagnostic: skip named parts of each layer; results are invalid (timing/hang
# bisection only). Every rank skips the same parts, so collectives stay paired.
DEBUG_SKIP = set(_config().debug_skip)


class _RowFunction[KeyT]:
    """Evaluate ``function(rows)`` on demand via ``instance[rows]``."""

    def __init__(self, function: Callable[[KeyT], jax.Array]) -> None:
        """Store the row-indexed computation callback."""
        self.function = function

    def __getitem__(self, rows: KeyT) -> jax.Array:
        """Return ``function(rows)`` for the requested slice or chunk index."""
        return self.function(rows)


def _shared_shards(batch_size: int) -> bool:
    """Return whether shared-expert outputs stay lane-sharded for ``batch_size``."""
    # B128 keeps only each lane's shared-expert output shard (VMEM).
    if SHARED_SHARDS is None:
        return batch_size == MAX_BATCH
    return bool(SHARED_SHARDS and batch_size > _SHARED_SHARDS_BATCH_THRESHOLD)


def _use_compact_overflow(
    batch_size: int,
    *,
    group_pending_routes: bool,
) -> bool:
    """Return whether overflowing grouped routes finish in compact passes."""
    return bool(
        COMPACT_OVERFLOW
        and not COMPACT_ONLY
        and SHARED_EXPERT_FINISHERS
        and (
            batch_size in _COMPACT_OVERFLOW_BATCHES
            or (batch_size == _B16_BATCH_SIZE and B16_COMPACT_EXPERTS)
        )
        and COMPACT_ROWS == _DEFAULT_COMPACT_ROWS
        and group_pending_routes
    )


def _expert_accumulator_rows(
    batch_size: int,
    *,
    group_pending_routes: bool = False,
) -> tuple[int, int]:
    """Return ``(gate_up_rows, down_rows)`` accumulator row counts."""
    capacity = _compact_rows(batch_size)
    if capacity and (
        COMPACT_ONLY
        or _use_compact_overflow(
            batch_size,
            group_pending_routes=group_pending_routes,
        )
    ):
        return _COMPACT_INPUT_BUFFERS * capacity, capacity
    return batch_size, batch_size


MOE_PREPARE_WEIGHTS = ("latent_down", "shared_gu", "shared_down")
MLA_EXTRA_WEIGHTS = ("m_qb", "m_kb", "m_gate")


def _pool_layout(
    local_weights: Mapping[str, jax.ShapeDtypeStruct],
    *,
    alias_expert_scales: bool,
    pooled: bool,
    managed: bool = False,
    dense_bytes: int = 0,
    arena_extra: int = 0,
) -> dict[str, int]:
    """Compute byte offsets of every view in the projection/expert pool.

    Region A (from 0) holds, one phase at a time: the next layer's attention
    projections (KDA k_gate..k_o, or MLA q/kv-down), prefetched after the
    experts finish; the MLA output projection over the consumed q-down
    projection; the four routed-expert slots during the MoE. Region B (after
    A) holds the MLA q/kv-up and gate projections during attention, the
    latent-down and shared-expert weights during routing, and latent-up in the
    MoE tail (which overlaps the next layer's attention prefetch into A).
    """

    def storage(name: str, *, experts: bool = False) -> int:
        """Return tiled byte footprint for ``name`` (or 4 expert slots)."""
        value = local_weights[name]
        shape = (_EXPERT_PREFETCH_SLOTS, *value.shape[1:]) if experts else value.shape
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
    region_b = -(-layout["pool_bytes"] // _ARENA_ALIGN_BYTES) * _ARENA_ALIGN_BYTES
    # One layer's attention input projections, as laid out by pool_views.
    layout["attention_bytes"] = max(
        sum(storage(name) for name in names) for names in (KDA[:4], MLA[:2])
    )
    if managed:
        # Region A is the arena the manager places the router, attention
        # projections, expert slots and dense slabs in.
        region_b = (
            -(
                -max(expert_bytes, layout["attention_bytes"], dense_bytes)
                // _ARENA_ALIGN_BYTES
            )
            * _ARENA_ALIGN_BYTES
        )
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
        msg = "The MLA output projection must fit over the q-down projection"
        raise ValueError(msg)
    layout["pool_bytes"] = end
    return layout


def _vmem_bytes(spec: pl.MemoryRef | tpu.SemaphoreType) -> int:
    """Return tiled local-memory byte size of a scratch buffer spec."""
    if (
        spec is tpu.SemaphoreType.REGULAR
        or spec is tpu.SemaphoreType.DMA
        or spec is tpu.SemaphoreType.BARRIER
    ):
        return 0
    if str(spec.memory_space) != "vmem":
        return 0
    shape = tuple(spec.shape)
    itemsize = jnp.dtype(spec.dtype).itemsize
    if len(shape) < _MIN_2D_RANK:
        return (
            -(-math.prod(shape) * itemsize // _SUBVECTOR_ALIGN_BYTES)
            * _SUBVECTOR_ALIGN_BYTES
        )
    packing = max(1, _WORD_BYTES // itemsize)
    rows = shape[-2]
    tile_rows = SUBLANE_ROWS * packing
    if rows < tile_rows:
        # Short second-minor dimensions use smaller native tiles (1 row for
        # 32-bit types, 2 or 4 for BF16), as in the compiler's reports.
        padded = max(packing, 1 << (rows - 1).bit_length())
    else:
        padded = -(-rows // tile_rows) * tile_rows
    return (
        math.prod(shape[:-2])
        * padded
        * (-(-shape[-1] // VREG_LANE_COUNT) * VREG_LANE_COUNT)
        * itemsize
    )


# Register-allocator spill slots are one program-wide VMEM area the compiler
# sizes; the budget only warns against this reserve.
VMEM_SPILL_RESERVE_BYTES = int(_config().vmem_spill_reserve_mb * MIB_BYTES)


def _format_mib(byte_count: int) -> str:
    """Format a byte count as an 8-column MiB string with two decimals."""
    return f"{byte_count / MIB_BYTES:8.2f}"


def _vmem_budget(
    persistent: Sequence[tuple[str, pl.MemoryRef | tpu.SemaphoreType]],
    regions: Mapping[
        str,
        tuple[
            Sequence[tuple[str, pl.MemoryRef | tpu.SemaphoreType]],
            Sequence[Sequence[tuple[str, pl.MemoryRef | tpu.SemaphoreType]]],
        ],
    ],
    *,
    limit: int,
    spill_reserve: int | None = None,
) -> tuple[str, int]:
    """Check a static local-memory plan and render it as a summary table.

    ``persistent`` is [(name, spec)], live for the whole kernel. ``regions`` is
    {region: ([(name, spec)] held for the region, [[(name, spec)], ...] nested
    alternatives)}; regions never overlap each other, and a region's nested
    alternatives never overlap each other. Returns (report, peak bytes).
    """
    resolved_reserve = (
        VMEM_SPILL_RESERVE_BYTES if spill_reserve is None else spill_reserve
    )
    lines = ["persistent VMEM (MiB):"]
    persistent_bytes = 0
    for name, spec in persistent:
        if (
            spec is tpu.SemaphoreType.REGULAR
            or spec is tpu.SemaphoreType.DMA
            or spec is tpu.SemaphoreType.BARRIER
        ):
            continue
        size = _vmem_bytes(spec)
        persistent_bytes += size
        if size >= _REPORT_ENTRY_MIN_BYTES:
            lines.append(
                f"  {_format_mib(size)}  {name} "
                f"{tuple(spec.shape)} {jnp.dtype(spec.dtype).name}",
            )
    lines.append(
        f"  {_format_mib(persistent_bytes)}  total (entries under 64 KiB included)",
    )
    peaks: dict[str, int] = {}
    for region, (held, nested) in regions.items():
        held_bytes = sum(_vmem_bytes(spec) for _, spec in held)
        nested_bytes = [
            sum(_vmem_bytes(spec) for _, spec in alternative) for alternative in nested
        ]
        peaks[region] = held_bytes + max(nested_bytes, default=0)
        lines.append(
            f"scoped region {region}: peak {_format_mib(peaks[region]).strip()} MiB",
        )
        for name, spec in held:
            entry_bytes = _vmem_bytes(spec)
            if entry_bytes >= _REPORT_ENTRY_MIN_BYTES:
                lines.append(f"  {_format_mib(entry_bytes)}  {name}")
        for alternative, size in zip(nested, nested_bytes, strict=True):
            if size:
                joined = ", ".join(name for name, _ in alternative)
                lines.append(f"  {_format_mib(size)}  nested: {joined}")
    worst = max(peaks, key=lambda region_name: peaks[region_name]) if peaks else None
    peak = persistent_bytes + (peaks[worst] if worst is not None else 0)
    lines.append(
        f"peak {_format_mib(peak).strip()} MiB (persistent + {worst}) + spill reserve "
        f"{_format_mib(resolved_reserve).strip()} of "
        f"{_format_mib(limit).strip()} MiB",
    )
    return "\n".join(lines), peak


def _mla_scoped_scratch(
    local_weights: Mapping[str, jax.ShapeDtypeStruct],
    *,
    batch_size: int,
    sequence_rows: int,
    dcp_heads: int = mla_dcp.HEADS_PER_RANK,
    dcp_dtype: jax_typing.DTypeLike = jnp.bfloat16,
    dcp_page_tokens: int = PAGE_TOKENS,
) -> tuple[pl.MemoryRef, ...]:
    """Return transient MLA scratch specs allocated per layer invocation.

    Includes placeholders for the workspace's cache fields and places
    ``mla_dcp.scoped_scratch`` last for ``dcp_heads`` heads per rank.
    """
    cache = (
        tpu.VMEM((SUBLANE_ROWS, VREG_LANE_COUNT), jnp.bfloat16),
        tpu.VMEM((SUBLANE_ROWS, VREG_LANE_COUNT), jnp.bfloat16),
        tpu.SemaphoreType.DMA((1,)),
    )
    dcp_plan = mla_dcp.page_plan(batch_size, dcp_heads, sequence_rows, dcp_dtype)
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
        tpu.VMEM(
            (batch_size, mla_dcp.HEADS_PER_RANK, _MLA_KEY_HANDOFF_LANES),
            jnp.bfloat16,
        ),
        tpu.VMEM((batch_size, _MLA_HEAD_OUT_WIDTH), jnp.bfloat16),
        tpu.VMEM(
            (batch_size, mla_dcp.HEADS_PER_RANK, _MLA_KEY_HANDOFF_LANES),
            jnp.bfloat16,
        ),
        tpu.VMEM((batch_size, mla_dcp.HEADS_PER_RANK, HEAD_DIM), jnp.bfloat16),
        tpu.VMEM((batch_size, mla_dcp.HEADS_PER_RANK, 1), jnp.float32),
        tpu.VMEM((batch_size, mla_dcp.HEADS_PER_RANK, 1), jnp.float32),
        tpu.VMEM((batch_size, mla_dcp.HEADS_PER_RANK, HEAD_DIM), jnp.float32),
        # The DCP page buffers by batch (mla_dcp.page_plan; DcpAttention
        # reads the span and the buffers from the shape).
        *mla_dcp.scoped_scratch(
            batch_size,
            span=dcp_plan["span"],
            buffers=dcp_plan["buffers"],
            heads=dcp_heads,
            cache_dtype=dcp_dtype,
            page_tokens=dcp_page_tokens,
            sequence_rows=sequence_rows,
        ),
    )


def _moe_scoped_scratch(
    local_weights: Mapping[str, jax.ShapeDtypeStruct],
    *,
    batch_size: int,
    alias_expert_scales: bool,
    token_scatter_output: bool,
    token_scatter_owners: int = _LOCAL_RANK_PAIRS,
    shard_routing: bool = False,
    group_pending_routes: bool = False,
) -> tuple[pl.MemoryRef | tpu.SemaphoreType, ...]:
    """Return transient MoE scratch specs not held in the shared weight pool."""
    gate_up = local_weights["expert_gu"]
    down = local_weights["expert_down"]
    latent_size = local_weights["latent_norm"].shape[-1]
    expert_input_size = gate_up.shape[1] * _MXFP4_VALUES_PER_WORD
    expert_width = down.shape[1] * _MXFP4_VALUES_PER_WORD
    expert_output_size = down.shape[-1]
    accumulator_rows = _expert_accumulator_rows(
        batch_size,
        group_pending_routes=group_pending_routes,
    )
    scale_scratch: tuple[pl.MemoryRef, ...] = (
        ()
        if alias_expert_scales
        else (
            tpu.VMEM(
                (_EXPERT_PREFETCH_SLOTS, *local_weights["expert_gus"].shape[1:]),
                local_weights["expert_gus"].dtype,
            ),
            tpu.VMEM(
                (_EXPERT_PREFETCH_SLOTS, *local_weights["expert_ds"].shape[1:]),
                local_weights["expert_ds"].dtype,
            ),
        )
    )
    owners = token_scatter_owners
    output_shape = (
        (
            owners + _MOE_SCATTER_EXTRA_SLOTS,
            batch_size // owners * _MOE_ROUTING_WIDTH // VREG_LANE_COUNT,
            VREG_LANE_COUNT,
        )
        if token_scatter_output
        else (batch_size, _WIDE_REDUCTION_WIDTH)
    )
    output_sem_count = (
        _MOE_SCATTER_OUTPUT_SEMS if token_scatter_output else _MOE_DEFAULT_OUTPUT_SEMS
    )
    compact_rows = _compact_rows(batch_size)
    return (
        *(
            ()
            if POOLED_WEIGHTS
            else (
                tpu.VMEM(
                    local_weights["latent_up"].shape,
                    local_weights["latent_up"].dtype,
                ),
            )
        ),
        tpu.SemaphoreType.DMA,
        tpu.SemaphoreType.DMA((_EXPERT_PREFETCH_SLOTS, _EXPERT_PREFETCH_SLOTS)),
        *scale_scratch,
        tpu.VMEM((batch_size, VREG_LANE_COUNT), jnp.int32),
        tpu.VMEM((batch_size, VREG_LANE_COUNT), jnp.float32),
        tpu.VMEM((batch_size, VREG_LANE_COUNT), jnp.int32),
        tpu.VMEM((batch_size, VREG_LANE_COUNT), jnp.float32),
        tpu.VMEM((batch_size, _MOE_ROUTING_WIDTH), jnp.float32),
        tpu.VMEM(
            (
                batch_size,
                _MOE_ROUTING_WIDTH if _shared_shards(batch_size) else HIDDEN_SIZE,
            ),
            jnp.float32,
        ),
        tpu.VMEM((batch_size, latent_size), jnp.float32),
        tpu.VMEM(
            (accumulator_rows[0], _EXPERT_PREFETCH_SLOTS * expert_width), jnp.float32
        ),
        tpu.VMEM((accumulator_rows[1], expert_output_size), jnp.float32),
        tpu.VMEM((batch_size, VREG_LANE_COUNT), jnp.int32),
        *(
            (tpu.VMEM((compact_rows, expert_input_size), jnp.float32),)
            * _COMPACT_INPUT_BUFFERS
            if compact_rows
            else (tpu.VMEM((1, expert_input_size), jnp.bfloat16),)
            * _COMPACT_INPUT_BUFFERS
        ),
        tpu.SemaphoreType.DMA((_MOE_SHARED_SEMS,)),
        tpu.SemaphoreType.DMA((_MOE_SHARED_SEMS,)),
        tpu.VMEM(output_shape, jnp.bfloat16),
        tpu.SemaphoreType.DMA((output_sem_count,)),
        tpu.SemaphoreType.DMA((output_sem_count,)),
        tpu.SemaphoreType.DMA((_MOE_LATENT_SEMS,)),
        tpu.SemaphoreType.DMA((_MOE_LATENT_SEMS,)),
        *(
            (
                tpu.VMEM(
                    (batch_size // _MOE_SHARD_ROUTING_OWNERS, VREG_LANE_COUNT),
                    jnp.float32,
                ),
                tpu.VMEM((batch_size, VREG_LANE_COUNT), jnp.float32),
            )
            if shard_routing
            else ()
        ),
    )


# Shard the MoE router's top-k over the local rank pairs in the four-owner
# token-scatter layout (KernelConfig.shard_routing=True).
SHARD_ROUTING = _config().shard_routing


# Decoder stack tiling, parallelism, and memory-planning constants.
_FINAL_LAYER_INDEX = NUM_LAYERS - 1
_RESIDUAL_BLOCK_STRIDE = 12
_RESIDUAL_SNAPSHOT_BLOCKS = 8
_BLOCK_TABLE_NDIM = 2
_KDA_STATE_PREFIX_COUNT = 2
_DP_MIN_SCATTER_BATCH = 2
_SINGLE_SP_OWNER = 1
_PAIR_SP_OWNERS = 4
_RANK_SP_OWNERS = 8
_PAIR_SPLIT_MAX_UNSPLIT_ROWS = 2
_BLOCK_MAJOR_MAX_FP32_BATCH = 8
_OWNED_SNAPSHOT_MAX_FULL_BATCH = 16
_PERSISTENT_KDA_MAX_BATCH = 4
_UNSTREAMED_KDA_MAX_BATCH = 16
_MAX_UNROLLED_MIX_ROWS = 2
_DP_NATIVE_NORM_BATCH = 64
_UNROLLED_FOUR_BUFFER_BATCH = 32
_FOUR_DENSE_SLABS_MAX_BATCH = 64
_DEFAULT_DENSE_SLABS = 4
_COMPACT_DENSE_SLABS = 3
_UNSCOPED_DENSE_MAX_BATCH = 8
_GATE_UP_FACTORS = 2
_DENSE_K_BLOCK_ROWS = 1024
_DENSE_DOWN_BLOCK_ROWS = 384
_DENSE_REDUCE_CHUNK_ROWS = 8
_STACK_OUTPUT_COUNT = 7
_STACK_COLLECTIVE_ID = 21
_DP_SWAP_SEMAPHORE_SLOT = 14
_DP_SWAP_SEND_LANE = 4
_DP_SWAP_RECV_LANE = 5
_KDA_AUXILIARY_REF_COUNT = 4
_MLA_EXTRA_PROJECTION_COUNT = 4
_DP_EXTRA_SCRATCH_COUNT = 5
_SCALAR_PREFETCH_BASE = 3
_HBM_TAIL_OUTPUT_COUNT = 5
_DP2_B2_B4_BATCHES = (2, 4)
_DP4_VALID_BATCHES = (32, 64, 96, 128)
_DP2_SMALL_BATCHES = (2, 4, 8, 16)
_DP_B64_B128_BATCHES = (64, 128)
_DP_B32_B64_BATCHES = (32, 64)
_DP4_ROLLED_GROUP_ROWS = (24, 32)
_SP_VALID_OWNERS = (1, 4, 8)
_DP_VALID_DEGREES = (1, 2, 4)
_AUX_CAPTURE_MODES = ("layer", "attention", "mixture_input")
_RESIDUAL_REDUCTION_ORDERS = ("reference", "row", "relaxed")
_METADATA_WEIGHT_NAMES = ("m_qnorm", "m_knorm", "router_bias", "latent_norm")
_MLA_NORM_NAMES = ("m_qnorm", "m_knorm")
_MOE_METADATA_NAMES = ("router_bias", "latent_norm")
_ALIASED_EXPERT_VIEW_ORDER = ("expert_gu", "expert_down", "expert_gus", "expert_ds")
_UNALIASED_EXPERT_NAMES = ("expert_gu", "expert_down")
_KDA_PREFETCH_ORDER = (0, 2, 1, 3)

type _ScratchSpec = pl.MemoryRef | tpu.SemaphoreType
type _DecodeStates = tuple[jax.Array, jax.Array, jax.Array, jax.Array]
type _StackOutputs = tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]


@dataclasses.dataclass(frozen=True, slots=True)
class _StackRequest:
    """Raw input arguments supplied to ``stack`` before validation."""

    prefix: jax.Array
    embedding: jax.Array
    weights: Mapping[str, jax.Array]
    states: Sequence[jax.Array]
    batch_size: int
    layers: int
    eps: float
    residual_reduction_order: Literal["reference", "row", "relaxed"]
    mla_cache_batch_tile_size: int | None
    alias_expert_scales: bool
    attention_token_scatter: bool
    moe_token_scatter_output: bool
    moe_routed_reduce_scatter: bool
    sequence_parallel_pre_attention: bool
    sequence_rows: int
    aux_hidden_layers: tuple[int, ...]
    aux_capture: Literal["layer", "attention", "mixture_input"]
    include_layer_zero: bool
    state_row: int | jax.Array | None
    attention_dp: int
    attention_projection_schedule: Literal["default", "mla_up_double"]
    mla_commit_tile_size: int | None
    sequence_parallel_size: int
    shard_routing: bool | None
    block_tables: jax.Array | None
    dcp_pool_map: tuple[int, ...] | None
    dcp_owner_table: tuple[int, ...] | None
    state_slots: jax.Array | None
    live_rows: jax.Array | None


@dataclasses.dataclass(frozen=True, slots=True)
class _StackPlan:
    """Validated compile-time configuration and layout invariants for ``stack``."""

    batch_size: int
    layers: int
    eps: float
    residual_reduction_order: Literal["reference", "row", "relaxed"]
    alias_expert_scales: bool
    attention_token_scatter: bool
    moe_token_scatter_output: bool
    moe_routed_reduce_scatter: bool
    sequence_parallel_pre_attention: bool
    sequence_rows: int
    aux_hidden_layers: tuple[int, ...]
    aux_capture: Literal["layer", "attention", "mixture_input"]
    include_layer_zero: bool
    select_state_row: bool
    attention_dp: int
    attention_dp_ownership: bool
    paged_pages: int
    first_layer: int
    dense_names: tuple[str, ...]
    pool_map: tuple[int, ...]
    dcp_owner_table: tuple[int, ...] | None
    b4_attention_gather: bool
    sp_owners: int
    tokens_per_pair: int
    owned_rows: int
    shard_routing: bool
    group_rows: int
    group_rows_per_pair: int
    dp_held_rows: int
    sp8_residual: bool
    pair_split_gather: bool
    kda_shard_swap: bool
    stream_config: dict[str, tuple[int, int]]
    cache_handoff_head_count: int
    hidden_size: int
    blocks_block_major: bool
    blocks_owned: bool
    compact_outputs: bool
    block_rows_held: int
    owner_generic: bool
    pool_managed: bool
    persist_kda_state: bool
    stream_kda_state: bool
    kda_row_chunk: int
    slot_reads: bool
    has_live_rows: bool
    overlap_kda_state_reads: bool
    overlap_kda_state_writes: bool
    state_count: int
    state_conv_shape: tuple[int, ...]
    state_conv_dtype: jnp.dtype
    state_rec_shape: tuple[int, ...]
    state_rec_dtype: jnp.dtype
    local_weights: dict[str, jax.ShapeDtypeStruct]
    pool_weights: dict[str, jax.ShapeDtypeStruct]


@dataclasses.dataclass(frozen=True, slots=True)
class _StackMemoryPlan:
    """Static scratch allocation layout, budget plan, and call operands."""

    args: tuple[jax.Array, ...]
    output_shapes: tuple[jax.ShapeDtypeStruct, ...]
    expert_pool_bytes: int
    dense_slabs: int
    pool_layout: dict[str, int]
    mla_scoped_scratch: tuple[_ScratchSpec, ...]
    moe_scoped_scratch: tuple[_ScratchSpec, ...]
    scratch: tuple[_ScratchSpec, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _StackRuntimeContext:
    """Unpacked Pallas references and workspace bindings inside ``body``."""

    plan: _StackPlan
    mem_plan: _StackMemoryPlan
    position_ref: PallasRefProtocol
    table_ref: PallasRefProtocol
    slots_ref: PallasRefProtocol | None
    active_rows: jax.Array | None
    selected_state_row: jax.Array | None
    prefix_ref: PallasRefProtocol
    embedding_ref: PallasRefProtocol
    layer_weights: dict[str, PallasRefProtocol]
    output_residual_ref: PallasRefProtocol
    final_normalization_ref: PallasRefProtocol
    dense_refs: dict[str, PallasRefProtocol]
    convolution_ref: PallasRefProtocol
    recurrent_ref: PallasRefProtocol
    pool_ref: tuple[PallasRefProtocol, ...]
    output_ref: PallasRefProtocol
    blocks_output_ref: PallasRefProtocol
    convolution_output_ref: PallasRefProtocol
    recurrent_output_ref: PallasRefProtocol
    key_output_ref: PallasRefProtocol
    value_output_ref: PallasRefProtocol
    aux_output_ref: PallasRefProtocol
    prefix_vmem: PallasRefProtocol
    blocks_vmem: PallasRefProtocol
    normalized_vmem: PallasRefProtocol
    attention_vmem: PallasRefProtocol
    mixture_vmem: PallasRefProtocol
    attention_residual_vmem: PallasRefProtocol
    mlp_residual_vmem: PallasRefProtocol
    attention_norm_vmem: PallasRefProtocol
    ffn_norm_vmem: PallasRefProtocol
    router_vmem: PallasRefProtocol
    router_semaphores: PallasRefProtocol
    projection_expert_pool_vmem: PallasRefProtocol
    pool_semaphores: PallasRefProtocol
    norm_semaphores: PallasRefProtocol
    auxiliary_vmem: tuple[PallasRefProtocol, ...]
    metadata_vmem: tuple[PallasRefProtocol, ...]
    auxiliary_semaphores: PallasRefProtocol
    metadata_semaphores: PallasRefProtocol
    owned_rows_vmem: PallasRefProtocol
    aux_semaphore: PallasRefProtocol
    dense_semaphores: PallasRefProtocol
    stream_semaphores: PallasRefProtocol
    group_input_vmem: PallasRefProtocol | None
    dp_attended_vmem: PallasRefProtocol | None
    group_positions_smem: PallasRefProtocol | None
    dp_compressed_vmem: PallasRefProtocol | None
    dcp_persistent: tuple[PallasRefProtocol, ...]
    headroom_pad: PallasRefProtocol | None
    pair_semaphores: tuple[PallasRefProtocol, PallasRefProtocol] | None
    communication_workspace: AttentionCommunicationWorkspace
    dp_owned_vmem: PallasRefProtocol | None
    sp8_rows_vmem: PallasRefProtocol | None
    pool: VmemPool | None


@dataclasses.dataclass(frozen=True, slots=True)
class _LayerExecutionContext:
    """Per-layer runtime context passed across attention and MLP sublayers."""

    ctx: _StackRuntimeContext
    layer_index: int | jax.Array
    attention_base: int
    kda_projection_vmem: tuple[PallasRefProtocol, ...]
    mla_projection_vmem: tuple[PallasRefProtocol, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _DenseMlpSpec:
    """Validated streaming geometry for layer-zero dense MLP execution."""

    gu_ref: PallasRefProtocol
    down_ref: PallasRefProtocol
    gu_width: int
    gu_blocks: int
    slab_rows: int
    slabs: int
    total: int
    slab_base: int


@dataclasses.dataclass(frozen=True, slots=True)
class _MlaLayerContext:
    """Per-layer MLA metadata references and optional DP output stream."""

    layer_ctx: _LayerExecutionContext
    state_index: int | jax.Array
    query_norm_vmem: PallasRefProtocol
    key_value_norm_vmem: PallasRefProtocol
    layer_metadata_semaphores: PallasRefProtocol
    dp_output: tuple[PallasRefProtocol, ChunkedWeightStream | None] | None


@dataclasses.dataclass(frozen=True, slots=True)
class _DecoderLogitsConfig:
    """Configuration for final residual mixing and LM-head logit projection."""

    layers: int
    eps: float
    sequence_parallel_pre_attention: bool
    batched_decoder_boundaries: bool


def _is_mla_layer(index: int | jax.Array) -> bool | jax.Array:
    """Return whether ``index`` is a multi-head latent attention layer."""
    return (index % MLA_LAYER_STRIDE == MLA_LAYER_MODULO) | (
        index == _FINAL_LAYER_INDEX
    )


def _is_kda_layer(index: int | jax.Array) -> bool | jax.Array:
    """Return whether ``index`` is a gated delta attention layer."""
    return (index % MLA_LAYER_STRIDE != MLA_LAYER_MODULO) & (
        index != _FINAL_LAYER_INDEX
    )


def _validate_stack_inputs(request: _StackRequest) -> tuple[tuple[int, ...], int]:
    """Validate batch bounds, unified pools, block tables, and aux capture."""
    if not 1 <= request.batch_size <= MAX_BATCH:
        msg = (
            f"The fused decoder stack supports batch sizes from 1 to {MAX_BATCH}, "
            f"got batch_size={request.batch_size}"
        )
        raise ValueError(msg)
    if request.include_layer_zero and any(
        name not in request.weights for name in DENSE
    ):
        missing = [name for name in DENSE if name not in request.weights]
        msg = f"include_layer_zero needs the dense MLP weights, missing={missing}"
        raise ValueError(msg)
    if request.block_tables is None or request.dcp_pool_map is None:
        msg = "The stack takes block_tables and dcp_pool_map (vLLM's unified pools)"
        raise ValueError(msg)
    pool_map = tuple(request.dcp_pool_map)
    expected_states = _KDA_STATE_PREFIX_COUNT + 1 + max(pool_map)
    if len(pool_map) != MLA_LAYERS or len(request.states) != expected_states:
        msg = (
            "The stack takes (conv, recurrent, *pools) and a pool per MLA layer, "
            f"got len(pool_map)={len(pool_map)}, len(states)={len(request.states)}"
        )
        raise ValueError(msg)
    pool_dtype = request.states[_KDA_STATE_PREFIX_COUNT].dtype
    pool_layout = cache_layout(str(pool_dtype))
    if any(
        pool.shape[1:] != pool_layout.block_shape or pool.dtype != pool_dtype
        for pool in request.states[_KDA_STATE_PREFIX_COUNT:]
    ):
        actual_pools = [
            (pool.shape, pool.dtype)
            for pool in request.states[_KDA_STATE_PREFIX_COUNT:]
        ]
        msg = (
            f"unified pools must match {pool_layout.block_shape} {pool_layout.dtype}, "
            f"got {actual_pools}"
        )
        raise ValueError(msg)
    if (
        request.block_tables.shape[0] != request.batch_size
        or request.block_tables.ndim != _BLOCK_TABLE_NDIM
    ):
        msg = (
            "block_tables must be [batch_size, pages], "
            f"got shape={request.block_tables.shape} "
            f"for batch_size={request.batch_size}"
        )
        raise ValueError(msg)
    return pool_map, int(request.block_tables.shape[1])


def _validate_sequence_and_aux_options(request: _StackRequest) -> int:
    """Validate sequence rows, aux capture settings, and cache batch tile."""
    if request.attention_dp != 1 and mla_dcp.GROUP * request.attention_dp != TP_SIZE:
        msg = (
            "Attention DP needs DCP over each DP group "
            "(KernelConfig.dcp_group = 32 / dp)"
        )
        raise ValueError(msg)
    if request.state_row is not None and request.sequence_rows != request.batch_size:
        msg = "state_row selection needs one sequence per tile"
        raise ValueError(msg)
    if (
        not 1 <= request.sequence_rows <= request.batch_size
        or request.batch_size % request.sequence_rows
    ):
        msg = "sequence_rows must divide the batch"
        raise ValueError(msg)
    if any(
        not 1 <= index <= request.layers - 1 for index in request.aux_hidden_layers
    ) or len(set(request.aux_hidden_layers)) != len(request.aux_hidden_layers):
        msg = (
            "aux_hidden_layers are distinct layer ids in 1..layers-1; id L "
            "captures the residual stream entering layer L"
        )
        raise ValueError(msg)
    if request.aux_capture not in _AUX_CAPTURE_MODES:
        msg = "aux_capture must be 'layer', 'attention' or 'mixture_input'"
        raise ValueError(msg)
    if request.aux_capture != "layer" and request.attention_token_scatter:
        msg = "Attention-point aux capture needs complete prefix rows"
        raise ValueError(msg)
    if request.residual_reduction_order not in _RESIDUAL_REDUCTION_ORDERS:
        msg = "residual_reduction_order must be 'reference', 'row', or 'relaxed'"
        raise ValueError(msg)
    # Rows of one sequence share one MLA cache entry, so the cache is never
    # tiled by rows; the other batch options act on rows and stay valid.
    tile = (
        request.batch_size
        if request.sequence_rows > 1
        else (request.mla_cache_batch_tile_size or request.batch_size)
    )
    if not 1 <= tile <= request.batch_size:
        msg = "MLA cache batch tile size must be within the batch"
        raise ValueError(msg)
    if request.batch_size % tile:
        msg = "MLA cache batch tile size must divide the batch"
        raise ValueError(msg)
    return tile


def _validate_scatter_and_sp_options(request: _StackRequest) -> None:
    """Validate token scatter, reduce-scatter, and sequence-parallel owners."""
    if (
        request.attention_token_scatter
        and request.batch_size not in (_B4_BATCH_SIZE, *SCATTER_BATCHES)
        and not (
            request.attention_dp > 1 and request.batch_size == _DP_MIN_SCATTER_BATCH
        )
    ):
        msg = (
            "Attention token scatter requires batch size 4, 8 or 16, "
            "or attention DP's B2"
        )
        raise ValueError(msg)
    if request.moe_token_scatter_output and not (
        request.attention_token_scatter
        and (
            request.batch_size in SCATTER_BATCHES
            or request.sequence_parallel_size < _PAIR_SP_OWNERS
        )
    ):
        msg = "MoE token-scattered output requires B8 attention token scatter"
        raise ValueError(msg)
    if request.moe_routed_reduce_scatter and request.batch_size not in (
        6,
        *SCATTER_BATCHES,
    ):
        msg = "MoE routed reduce-scatter requires batch size 6, 8 or 16"
        raise ValueError(msg)
    if request.sequence_parallel_pre_attention and not request.moe_token_scatter_output:
        msg = "Sequence-parallel pre-attention requires B8 MoE token scatter"
        raise ValueError(msg)
    if request.sequence_parallel_size not in _SP_VALID_OWNERS:
        msg = (
            "sequence_parallel_size must be 1 (attention DP2 B2/B4), "
            "4 (rank pairs) or 8 (ranks)"
        )
        raise ValueError(msg)
    if request.sequence_parallel_size == _SINGLE_SP_OWNER and not (
        request.attention_dp == DP2_GROUPS and request.batch_size in _DP2_B2_B4_BATCHES
    ):
        msg = "One row owner is attention DP2's B2/B4 layout"
        raise ValueError(msg)
    if request.sequence_parallel_size == _RANK_SP_OWNERS and not (
        request.attention_token_scatter
        and request.batch_size % _RANK_SP_OWNERS == 0
        and request.attention_dp == 1
    ):
        msg = (
            "Eight row owners need the token-scatter layout, a multiple of 8 "
            "rows and no attention DP"
        )
        raise ValueError(msg)


def _validate_dp_and_slot_options(
    request: _StackRequest,
    *,
    blocks_block_major: bool,
    persist_kda_state: bool,
) -> None:
    """Validate attention DP degree, block-major mode, and KDA slot reads."""
    sp_owners = request.sequence_parallel_size
    if request.attention_dp not in _DP_VALID_DEGREES:
        msg = "Attention DP supports dp=1, dp=2 or dp=4"
        raise ValueError(msg)
    if request.attention_dp > 1 and not (
        request.batch_size in DP_VALID_BATCHES
        and request.batch_size % (sp_owners * request.attention_dp) == 0
        and request.sequence_parallel_pre_attention
        and (request.batch_size // request.attention_dp) % request.sequence_rows == 0
        and request.state_row is None
        and POOLED_WEIGHTS
    ):
        msg = (
            "Attention DP needs a supported batch divisible by 4*DP, the "
            "sequence-parallel token-scatter layout, whole sequences in each "
            "group's rows and pooled weights"
        )
        raise ValueError(msg)
    if request.attention_dp > 1 and (
        request.batch_size not in _DP4_VALID_BATCHES
        and not (
            request.attention_dp == DP2_GROUPS
            and request.batch_size in _DP2_SMALL_BATCHES
        )
    ):
        msg = "Attention DP supports DP2 at B2..B128 or DP4 at B32..B128"
        raise ValueError(msg)
    if blocks_block_major and not request.sequence_parallel_pre_attention:
        msg = "Above eight rows the decoder needs sequence-parallel pre-attention"
        raise ValueError(msg)
    slot_reads = request.state_slots is not None
    if slot_reads and not (
        (
            request.attention_dp == 1
            and (request.state_row is not None or not persist_kda_state)
        )
        or (
            request.attention_dp == DP2_GROUPS
            and request.batch_size in DP_VALID_BATCHES
        )
        or (
            request.attention_dp == DP4_GROUPS
            and request.batch_size in _DP4_VALID_BATCHES
        )
    ):
        msg = (
            "state_slots needs DP1 with state_row or batch > 4, "
            "DP2 at B2..B128 or DP4 at B32..B128, "
            f"got attention_dp={request.attention_dp}, batch_size={request.batch_size}"
        )
        raise ValueError(msg)


def _build_stream_config(
    request: _StackRequest,
    group_rows: int,
    *,
    kda_shard_swap: bool,
) -> dict[str, tuple[int, int]]:
    """Combine default, DP, user, and projection-schedule weight streams."""
    default_streams = (
        {
            **DCP_B32_WEIGHT_STREAMS,
            **(DCP_B32_DECODE_RINGS if request.sequence_rows == 1 else {}),
        }
        if (
            request.batch_size in _DP_B32_B64_BATCHES
            and request.attention_dp == 1
            and not _config().weight_streams
        )
        else {}
    )
    projection_streams = _attention_projection_streams(
        request.attention_dp,
        request.attention_projection_schedule,
    )
    conflicts = projection_streams.keys() & WEIGHT_STREAMS.keys()
    if conflicts:
        msg = (
            "Projection schedule conflicts with KernelConfig.weight_streams: "
            f"{sorted(conflicts)}"
        )
        raise ValueError(msg)
    dp_streams = (
        _dp_dcp_weight_streams(request.attention_dp, group_rows)
        if request.attention_dp > 1
        else {}
    )
    swap_stream = {"k_projection": (request.attention_dp, 1)} if kda_shard_swap else {}
    return {
        **DP_WEIGHT_STREAMS.get(request.attention_dp, {}),
        **dp_streams,
        **swap_stream,
        **default_streams,
        **WEIGHT_STREAMS,
        **projection_streams,
    }


def _pooled_weight_shape(
    name: str,
    local_weights: Mapping[str, jax.ShapeDtypeStruct],
    stream_config: Mapping[str, tuple[int, int]],
) -> tuple[int, ...]:
    """Return pool footprint shape of ``name``, sizing streamed rings to depth."""
    weight_shape = local_weights[name].shape
    if name not in STREAMED_WEIGHTS or name not in stream_config:
        return weight_shape
    chunks, depth = stream_config[name]
    if chunks == depth:
        return weight_shape
    axis = 1 if STREAMED_WEIGHTS[name][0] == "columns" else 0
    unit = VREG_LANE_COUNT if axis == 1 else _BF16_TILE_ROWS
    largest = _chunk_sizes(weight_shape[axis], chunks, unit)[0]
    return tuple(
        depth * largest if index == axis else size
        for index, size in enumerate(weight_shape)
    )


def _validate_and_plan_stack(request: _StackRequest) -> _StackPlan:
    """Validate ``stack`` options and derive all static tile and layout fields."""
    pool_map, paged_pages = _validate_stack_inputs(request)
    mla_tile = _validate_sequence_and_aux_options(request)
    _validate_scatter_and_sp_options(request)
    # Attention DP always runs with DP row ownership: a rank's residual stream
    # holds only its own rows of its DP group.
    attention_dp_ownership = request.attention_dp > 1
    # DP row ownership keeps its rows' snapshots block-major at every batch.
    blocks_block_major = (
        request.batch_size > _BLOCK_MAJOR_MAX_FP32_BATCH or attention_dp_ownership
    )
    persist_kda_state = (
        request.batch_size <= _PERSISTENT_KDA_MAX_BATCH and request.attention_dp == 1
    )
    _validate_dp_and_slot_options(
        request,
        blocks_block_major=blocks_block_major,
        persist_kda_state=persist_kda_state,
    )
    commit_rows = request.mla_commit_tile_size or mla_tile
    if request.batch_size % commit_rows:
        msg = (
            "mla_commit_tile_size must divide the batch, "
            f"got batch_size={request.batch_size}, commit_rows={commit_rows}"
        )
        raise ValueError(msg)
    sp_owners = request.sequence_parallel_size
    # Rows of each local row owner of attention DP's layout (a rank pair, or
    # at DP2 B2/B4 every rank: all rows).
    tokens_per_pair = (
        request.batch_size
        // (sp_owners if request.attention_dp > 1 else _PAIR_SP_OWNERS)
        if request.attention_token_scatter
        else 0
    )
    owned_rows = (
        request.batch_size // sp_owners if request.attention_token_scatter else 0
    )
    shard_routing = (
        SHARD_ROUTING if request.shard_routing is None else request.shard_routing
    )
    # The router splits its columns over each rank pair, so the top-k shards
    # over pairs: four row owners.
    shard_routing = (
        shard_routing
        and request.moe_token_scatter_output
        and sp_owners == _PAIR_SP_OWNERS
    )
    group_rows = request.batch_size // request.attention_dp
    # Each local row owner holds tokens_per_pair rows; group d attends r of them.
    group_rows_per_pair = group_rows // sp_owners
    # The residual-stream rows a rank holds under DP row ownership: its own r,
    # or with one owner every row.
    dp_held_rows = (
        request.batch_size if sp_owners == _SINGLE_SP_OWNER else group_rows_per_pair
    )
    sp8_residual = (
        attention_dp_ownership
        and request.attention_dp == DP2_GROUPS
        and request.batch_size == MAX_BATCH
        and request.sequence_rows == 1
    )
    # Plain decode above two rows per rank pair: its two ranks split the
    # MoE-input gather (DP2 B128's eight residual owners gather their own).
    pair_split_gather = (
        attention_dp_ownership
        and not sp8_residual
        and sp_owners != _SINGLE_SP_OWNER
        and dp_held_rows > _PAIR_SPLIT_MAX_UNSPLIT_ROWS
        and request.sequence_rows == 1
    )
    # DP2 B2/B4 (one row owner): each rank projects only its own group's held
    # shard of the KDA input projection, for every row (swap_kda_projections).
    kda_shard_swap = attention_dp_ownership and sp_owners == _SINGLE_SP_OWNER
    stream_config = _build_stream_config(
        request, group_rows, kda_shard_swap=kda_shard_swap
    )
    # The new-row hand-out's padded head count follows these tile sizes.
    cache_handoff_head_count = max(
        mla_dcp.HEADS_PER_RANK,
        SUBLANE_ROWS // math.gcd(mla_tile, commit_rows, SUBLANE_ROWS),
    )
    # Above B16 each row owner keeps only its own rows' snapshots.
    blocks_owned = request.batch_size > _OWNED_SNAPSHOT_MAX_FULL_BATCH
    # Above B16 with token-scattered attention and MoE outputs, the
    # whole-batch attention and mixture outputs are never written.
    compact_outputs = (
        request.batch_size > _OWNED_SNAPSHOT_MAX_FULL_BATCH
        and request.attention_token_scatter
        and request.moe_token_scatter_output
    )
    # DP row ownership: a rank's residual stream (prefix and snapshots) is
    # only its own group_rows_per_pair rows, held from row 0.
    block_rows_held = (
        dp_held_rows
        if attention_dp_ownership
        else (owned_rows if blocks_owned else request.batch_size)
    )
    # One copy of each per-owner row region with a traced owner index (program
    # size O(1) in the owner count). BF16 row slices at a traced offset must be
    # provably 16-aligned, so owners need a multiple of 16 rows (B64 at sp=4,
    # B128 at sp=8); smaller owners use unrolled copies.
    owner_generic = blocks_owned and owned_rows % _BF16_TILE_ROWS == 0
    local_weights = {
        name: jax.ShapeDtypeStruct(value.shape[1:], value.dtype)
        for name, value in request.weights.items()
    }
    pool_weights = {
        name: jax.ShapeDtypeStruct(
            _pooled_weight_shape(name, local_weights, stream_config),
            value.dtype,
        )
        for name, value in local_weights.items()
    }
    overlap_kda_io = (
        request.batch_size in (6, *SCATTER_BATCHES) or request.attention_dp > 1
    )
    return _StackPlan(
        batch_size=request.batch_size,
        layers=request.layers,
        eps=request.eps,
        residual_reduction_order=request.residual_reduction_order,
        alias_expert_scales=request.alias_expert_scales,
        attention_token_scatter=request.attention_token_scatter,
        moe_token_scatter_output=request.moe_token_scatter_output,
        moe_routed_reduce_scatter=request.moe_routed_reduce_scatter,
        sequence_parallel_pre_attention=request.sequence_parallel_pre_attention,
        sequence_rows=request.sequence_rows,
        aux_hidden_layers=request.aux_hidden_layers,
        aux_capture=request.aux_capture,
        include_layer_zero=request.include_layer_zero,
        select_state_row=request.state_row is not None,
        attention_dp=request.attention_dp,
        attention_dp_ownership=attention_dp_ownership,
        paged_pages=paged_pages,
        first_layer=0 if request.include_layer_zero else 1,
        dense_names=DENSE if request.include_layer_zero else (),
        pool_map=pool_map,
        dcp_owner_table=request.dcp_owner_table,
        b4_attention_gather=(
            B4_ATTENTION_GATHER
            and request.batch_size == _B4_BATCH_SIZE
            and not request.attention_token_scatter
        ),
        sp_owners=sp_owners,
        tokens_per_pair=tokens_per_pair,
        owned_rows=owned_rows,
        shard_routing=shard_routing,
        group_rows=group_rows,
        group_rows_per_pair=group_rows_per_pair,
        dp_held_rows=dp_held_rows,
        sp8_residual=sp8_residual,
        pair_split_gather=pair_split_gather,
        kda_shard_swap=kda_shard_swap,
        stream_config=stream_config,
        cache_handoff_head_count=cache_handoff_head_count,
        hidden_size=request.prefix.shape[-1],
        blocks_block_major=blocks_block_major,
        blocks_owned=blocks_owned,
        compact_outputs=compact_outputs,
        block_rows_held=block_rows_held,
        owner_generic=owner_generic,
        # The trace-time arena manager (``VmemPool``) places the router,
        # attention input projections, routed-expert slots and layer 0's dense
        # slabs; under DP the router reuses the drained stream slots.
        pool_managed=POOLED_WEIGHTS,
        persist_kda_state=persist_kda_state,
        stream_kda_state=request.batch_size > _UNSTREAMED_KDA_MAX_BATCH,
        kda_row_chunk=_kda_row_chunk(request.batch_size, request.attention_dp),
        slot_reads=request.state_slots is not None,
        has_live_rows=request.live_rows is not None,
        overlap_kda_state_reads=overlap_kda_io,
        overlap_kda_state_writes=overlap_kda_io,
        state_count=len(request.states),
        state_conv_shape=tuple(request.states[0].shape),
        state_conv_dtype=jnp.dtype(request.states[0].dtype),
        state_rec_shape=tuple(request.states[1].shape),
        state_rec_dtype=jnp.dtype(request.states[1].dtype),
        local_weights=local_weights,
        pool_weights=pool_weights,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class _KdaExecBuffers:
    """KDA history/state buffers, semaphores, and read copies for execution."""

    history_vmem: PallasRefProtocol
    state_vmem: PallasRefProtocol
    state_semaphores: PallasRefProtocol
    state_reads: tuple[AsyncCopyProtocol, ...] = ()


@dataclasses.dataclass(frozen=True, slots=True)
class _MlaScopedBuffers:
    """Scoped handoff and DCP buffers passed into ``mla`` or ``mla_dcp_shards``."""

    new_key_vmem: PallasRefProtocol
    new_value_vmem: PallasRefProtocol
    workspace: tuple[PallasRefProtocol, ...]
    dcp_scoped: tuple[PallasRefProtocol, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _DenseStreamBuffers:
    """Optional scratch buffers for staged dense MLP accumulation."""

    projection_vmem: PallasRefProtocol | None = None
    accumulator_vmem: PallasRefProtocol | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _DpMixOptions:
    """Per-call control flags for DP residual row mixing."""

    layer_index: int | jax.Array
    attention_output: bool = False
    start_block: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class _KdaStateCallbacks:
    """Wait and write-start callbacks passed into unstreamed ``kda`` calls."""

    wait_for_state: Callable[[], None]
    start_state_writes: Callable[[], None]


def _require_ref(ref: PallasRefProtocol | None, name: str) -> PallasRefProtocol:
    """Narrow an optional Pallas reference, raising ``RuntimeError`` if absent."""
    if ref is None:
        msg = f"{name} is required in this execution regime"
        raise RuntimeError(msg)
    return ref


def _require_array(value: jax.Array | None, name: str) -> jax.Array:
    """Narrow an optional traced array, raising ``RuntimeError`` if absent."""
    if value is None:
        msg = f"{name} is required in this execution regime"
        raise RuntimeError(msg)
    return value


def _shape_struct(value: jax.Array) -> jax.ShapeDtypeStruct:
    """Return a ``ShapeDtypeStruct`` matching ``value``."""
    return jax.ShapeDtypeStruct(value.shape, value.dtype)


def _plan_scoped_scratch(
    request: _StackRequest,
    plan: _StackPlan,
) -> tuple[tuple[_ScratchSpec, ...], tuple[_ScratchSpec, ...]]:
    """Compute MLA and MoE scoped scratch descriptors for ``stack``."""
    pool_ref = request.states[_KDA_STATE_PREFIX_COUNT]
    mla_scoped = _mla_scoped_scratch(
        plan.local_weights,
        # Attention DP runs MLA on the group's rows, one shard at a time.
        batch_size=plan.batch_size // plan.attention_dp,
        sequence_rows=plan.sequence_rows,
        dcp_dtype=pool_ref.dtype,
        dcp_page_tokens=pool_ref.shape[-3] * pool_ref.shape[-2],
        dcp_heads=mla_dcp.HEADS_PER_RANK * plan.attention_dp,
    )
    moe_scoped = _moe_scoped_scratch(
        plan.local_weights,
        batch_size=plan.batch_size,
        alias_expert_scales=plan.alias_expert_scales,
        token_scatter_output=plan.moe_token_scatter_output,
        token_scatter_owners=plan.sp_owners,
        shard_routing=plan.shard_routing,
        group_pending_routes=(
            plan.batch_size in (6, *SCATTER_BATCHES)
            and plan.residual_reduction_order == "relaxed"
        ),
    )
    return mla_scoped, moe_scoped


def _build_stack_args_and_outputs(
    request: _StackRequest,
    plan: _StackPlan,
) -> tuple[tuple[jax.Array, ...], tuple[jax.ShapeDtypeStruct, ...]]:
    """Build Pallas HBM input operand tuple and output ``ShapeDtypeStruct`` tuple."""
    args = (
        request.prefix,
        request.embedding,
        *(request.weights[name] for name in NAMES + EPILOGUE + plan.dense_names),
        *request.states,
    )
    # With state_slots the inputs are the whole slot arrays; updated states are
    # per row.
    state_output_shapes = (
        jax.ShapeDtypeStruct(
            (plan.batch_size, *plan.state_conv_shape[1:]), plan.state_conv_dtype
        ),
        jax.ShapeDtypeStruct(
            (plan.batch_size, *plan.state_rec_shape[1:]), plan.state_rec_dtype
        ),
        jax.ShapeDtypeStruct(
            (
                plan.batch_size,
                MLA_LAYERS,
                plan.cache_handoff_head_count,
                _MLA_KEY_HANDOFF_LANES,
            ),
            jnp.bfloat16,
        ),
        jax.ShapeDtypeStruct(
            (
                plan.batch_size,
                MLA_LAYERS,
                plan.cache_handoff_head_count,
                HEAD_DIM,
            ),
            jnp.bfloat16,
        ),
    )
    output_shapes = (
        _shape_struct(request.prefix),
        (
            jax.ShapeDtypeStruct((SUBLANE_ROWS, VREG_LANE_COUNT), jnp.uint8)
            if plan.sequence_parallel_pre_attention
            else jax.ShapeDtypeStruct(
                (plan.batch_size, _RESIDUAL_SNAPSHOT_BLOCKS, plan.hidden_size),
                jnp.bfloat16,
            )
        ),
        *state_output_shapes,
        jax.ShapeDtypeStruct(
            (max(1, len(plan.aux_hidden_layers)), plan.batch_size, plan.hidden_size),
            jnp.bfloat16,
        ),
    )
    return args, output_shapes


def _log_pool_region_report(
    plan: _StackPlan,
    pool_layout: Mapping[str, int],
    pooled_expert_names: tuple[str, ...],
    dense_total_bytes: int,
) -> None:
    """Emit diagnostic MiB breakdown for pooled region A and region B weights."""

    def _mib(name: str, *, experts: bool = False) -> float:
        """Return pool storage size of ``name`` in MiB."""
        value = plan.pool_weights[name]
        weight_shape = (
            (_EXPERT_PREFETCH_SLOTS, *value.shape[1:]) if experts else value.shape
        )
        return pool_alias.storage_size(weight_shape, value.dtype) / MIB_BYTES

    region_a = {
        "kda": sum(_mib(name) for name in KDA[:_KDA_PROJECTION_COUNT]),
        "mla": sum(_mib(name) for name in MLA[:_MLA_INPUT_PROJECTION_COUNT]),
        "experts": sum(_mib(name, experts=True) for name in pooled_expert_names),
    }
    region_b = {
        "mla_extra": sum(_mib(name) for name in MLA_EXTRA_WEIGHTS),
        "moe_prepare": sum(_mib(name) for name in MOE_PREPARE_WEIGHTS),
        "latent_up": _mib("latent_up"),
    }
    _logger.info(
        "pool region A (MiB): %s",
        {key: round(val, 2) for key, val in region_a.items()},
    )
    if plan.pool_managed:
        _logger.info(
            "arena (managed region A): %.2f MiB, attention %.2f, dense slabs %.2f",
            pool_layout["arena_bytes"] / MIB_BYTES,
            pool_layout["attention_bytes"] / MIB_BYTES,
            dense_total_bytes / MIB_BYTES,
        )
    _logger.info(
        "pool region B (MiB): %s %s",
        {key: round(val, 2) for key, val in region_b.items()},
        {
            name: round(_mib(name), 2)
            for name in (*MLA_EXTRA_WEIGHTS, *MOE_PREPARE_WEIGHTS, "latent_up")
        },
    )


def _build_stack_scratch_shapes(
    request: _StackRequest,
    plan: _StackPlan,
    pool_bytes: int,
) -> tuple[tuple[_ScratchSpec, ...], tuple[str, ...]]:
    """Build persistent scratch shape specs and matching diagnostic names."""
    dp_scratch: tuple[_ScratchSpec, ...] = (
        (
            # Attention DP: the group's attention input rows, each shard's
            # gated attention output, the group rows' positions, and the
            # MLA latents shared by the shards.
            tpu.VMEM((plan.group_rows, plan.hidden_size), jnp.bfloat16),
            tpu.VMEM(
                (
                    plan.attention_dp,
                    plan.group_rows,
                    max(_KDA_HEAD_OUT_WIDTH, _MLA_HEAD_OUT_WIDTH),
                ),
                jnp.bfloat16,
            ),
            tpu.SMEM((plan.group_rows,), jnp.int32),
            tpu.VMEM((plan.group_rows, _MLA_QUERY_LORA_RANK), jnp.bfloat16),
            tpu.VMEM((plan.group_rows, MLA_LATENT_ROW_LANES), jnp.bfloat16),
        )
        if plan.attention_dp > 1
        else ()
    )
    pool_dtype = request.states[_KDA_STATE_PREFIX_COUNT].dtype
    vmem_pad_kb = _config().vmem_pad_kb
    scratch: tuple[_ScratchSpec, ...] = (
        tpu.VMEM(
            (
                -(-plan.block_rows_held // _BF16_TILE_ROWS) * _BF16_TILE_ROWS,
                plan.hidden_size,
            )
            if plan.attention_dp_ownership
            else request.prefix.shape,
            jnp.bfloat16,
        ),
        # Residual snapshots are exact BF16 values. Up to B8: FP32
        # [batch, 8, hidden], no unpack per mix (a two-row allocation for the
        # owned rows miscompiles across loop iterations). Above B8 or with DP
        # row ownership: BF16, two rows per native (16, 128) tile.
        tpu.VMEM(
            (-(-plan.block_rows_held // 2), _BF16_TILE_ROWS, plan.hidden_size)
            if plan.blocks_block_major
            else (plan.batch_size, _RESIDUAL_SNAPSHOT_BLOCKS, plan.hidden_size),
            jnp.bfloat16 if plan.blocks_block_major else jnp.float32,
        ),
        tpu.VMEM(request.prefix.shape, jnp.bfloat16),
        *(
            (
                tpu.VMEM((_BF16_TILE_ROWS, VREG_LANE_COUNT), jnp.bfloat16),
                tpu.VMEM((_BF16_TILE_ROWS, VREG_LANE_COUNT), jnp.bfloat16),
            )
            if plan.compact_outputs
            else (
                tpu.VMEM(request.prefix.shape, jnp.bfloat16),
                tpu.VMEM(request.prefix.shape, jnp.bfloat16),
            )
        ),
        *(
            tpu.VMEM((1, plan.hidden_size), request.weights[name].dtype)
            for name in NORMS
        ),
        # With the router in the arena this persistent slot is a stub.
        tpu.VMEM(
            (SUBLANE_ROWS, VREG_LANE_COUNT)
            if plan.pool_managed
            else (1, plan.hidden_size, _ROUTER_LANE_SCORE_WIDTH),
            jnp.bfloat16,
        ),
        tpu.SemaphoreType.DMA((2,)),
        tpu.VMEM((pool_bytes // VREG_LANE_COUNT, VREG_LANE_COUNT), jnp.uint8),
        tpu.SemaphoreType.DMA((6,)),
        tpu.SemaphoreType.DMA((4,)),
        *(
            tpu.VMEM(plan.local_weights[name].shape, plan.local_weights[name].dtype)
            for name in KDA[_KDA_AUXILIARY_REF_COUNT:]
        ),
        *(
            (
                tpu.VMEM(
                    (plan.batch_size, *plan.state_conv_shape[2:]),
                    plan.state_conv_dtype,
                ),
                tpu.VMEM(
                    (plan.batch_size, *plan.state_rec_shape[2:]),
                    plan.state_rec_dtype,
                ),
            )
            if plan.persist_kda_state
            else (
                tpu.VMEM((1,), jnp.uint8),
                tpu.VMEM((1,), jnp.uint8),
            )
        ),
        tpu.SemaphoreType.DMA((6,)),
        *(
            tpu.VMEM(
                (2, *plan.local_weights[name].shape),
                plan.local_weights[name].dtype,
            )
            for name in _METADATA_WEIGHT_NAMES
        ),
        tpu.SemaphoreType.DMA((2, 4)),
        tpu.VMEM(
            (plan.owned_rows, plan.hidden_size)
            if plan.attention_token_scatter
            else (1,),
            jnp.bfloat16,
        ),
        tpu.SemaphoreType.DMA((1,)),
        tpu.SemaphoreType.DMA((4,)),
        tpu.SemaphoreType.DMA((_stream_semaphore_count(plan.stream_config),)),
        *dp_scratch,
        # DCP: buffers group peers' remote copies land in.
        *mla_dcp.persistent_scratch(
            plan.group_rows,
            heads=mla_dcp.HEADS_PER_RANK * plan.attention_dp,
            cache_dtype=pool_dtype,
            sequence_rows=plan.sequence_rows,
        ),
        *communication_scratch(plan.batch_size, plan.attention_dp)[:-1],
        tpu.VMEM((plan.batch_size, _LATENT_GATHER_WIDTH), jnp.bfloat16),
        *projection_staging_scratch(plan.batch_size, plan.attention_dp),
        *routed_send_scratch(plan.batch_size),
        *((tpu.SemaphoreType.REGULAR,) if plan.moe_routed_reduce_scatter else ()),
        *(
            (tpu.SemaphoreType.DMA((1,)), tpu.SemaphoreType.DMA((1,)))
            if plan.pair_split_gather
            else ()
        ),
        # Four 7168-element BF16 wires, padded to 64 BF16 transport rows.
        # Private storage (see B4_ATTENTION_GATHER).
        *(
            (tpu.VMEM((_PAIR_SP_OWNERS, 32, VREG_LANE_COUNT), jnp.float32),)
            if plan.b4_attention_gather
            else ()
        ),
        # Diagnostic: vmem_pad_kb reserves unused scratch to probe headroom.
        *(
            (tpu.VMEM((vmem_pad_kb * 2, VREG_LANE_COUNT), jnp.float32),)
            if vmem_pad_kb
            else ()
        ),
    )
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
        *KDA[_KDA_AUXILIARY_REF_COUNT:],
        "kda_history",
        "kda_state",
        "auxiliary_semaphores",
        *_METADATA_WEIGHT_NAMES,
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
            if plan.attention_dp > 1
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
            if plan.batch_size > _WIDE_REDUCTION_MAX_BATCH
            else ()
        ),
        *(("routed_consumed",) if plan.moe_routed_reduce_scatter else ()),
        *(
            ("pair_send_semaphore", "pair_receive_semaphore")
            if plan.pair_split_gather
            else ()
        ),
        *(("attention_gather",) if plan.b4_attention_gather else ()),
        *(("headroom_pad",) if vmem_pad_kb else ()),
    )
    return scratch, scratch_names


def _build_attention_budget_regions(
    plan: _StackPlan,
    mla_scoped_scratch: tuple[_ScratchSpec, ...],
    projection_reduction: tuple[str, _ScratchSpec],
) -> dict[
    str,
    tuple[
        list[tuple[str, _ScratchSpec]],
        list[list[tuple[str, _ScratchSpec]]],
    ],
]:
    """Build static memory-budget region entries for MLA, DP projection, and KDA."""
    mla_base: list[tuple[str, _ScratchSpec]] = [
        (
            "new-key handoff",
            tpu.VMEM(
                (
                    plan.batch_size,
                    plan.cache_handoff_head_count,
                    _MLA_KEY_HANDOFF_LANES,
                ),
                jnp.bfloat16,
            ),
        ),
        (
            "new-value handoff",
            tpu.VMEM(
                (
                    plan.batch_size,
                    plan.cache_handoff_head_count,
                    HEAD_DIM,
                ),
                jnp.bfloat16,
            ),
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
    ]
    mla_nested: list[list[tuple[str, _ScratchSpec]]] = [
        *([] if plan.attention_dp > 1 else [[projection_reduction]]),
        [("new-row hand-out semaphores", tpu.SemaphoreType.DMA((2,)))],
    ]
    mla_region = (mla_base, mla_nested)
    if plan.persist_kda_state:
        kda_base: list[tuple[str, _ScratchSpec]] = []
    elif plan.stream_kda_state:
        kda_base = [
            (
                "kda history chunks",
                tpu.VMEM(
                    (
                        _ROLLING_STATE_BUFFERS,
                        plan.kda_row_chunk,
                        *plan.state_conv_shape[2:],
                    ),
                    plan.state_conv_dtype,
                ),
            ),
            (
                "kda state chunks",
                tpu.VMEM(
                    (
                        _ROLLING_STATE_BUFFERS,
                        plan.kda_row_chunk,
                        *plan.state_rec_shape[2:],
                    ),
                    plan.state_rec_dtype,
                ),
            ),
        ]
    else:
        kda_base = [
            (
                "kda history",
                tpu.VMEM(
                    (plan.batch_size, *plan.state_conv_shape[2:]),
                    plan.state_conv_dtype,
                ),
            ),
            (
                "kda state",
                tpu.VMEM(
                    (plan.batch_size, *plan.state_rec_shape[2:]),
                    plan.state_rec_dtype,
                ),
            ),
        ]
    kda_nested: list[tuple[str, _ScratchSpec]] = [
        projection_reduction,
        *(
            (
                (
                    "DP staged projection",
                    tpu.VMEM(
                        (
                            plan.group_rows // plan.kda_row_chunk,
                            plan.kda_row_chunk,
                            _KDA_PROJECTION_WIDTH,
                        ),
                        jnp.bfloat16,
                    ),
                ),
                (
                    "DP staged recurrence gate",
                    tpu.VMEM(
                        (
                            plan.group_rows // plan.kda_row_chunk,
                            plan.kda_row_chunk,
                            _KDA_HEAD_OUT_WIDTH,
                        ),
                        jnp.bfloat16,
                    ),
                ),
                (
                    "DP staged output gate",
                    tpu.VMEM((plan.group_rows, _KDA_HEAD_OUT_WIDTH), jnp.bfloat16),
                ),
            )
            if plan.attention_dp > 1 and plan.group_rows > _SMALL_DP_GROUP_ROWS
            else ()
        ),
        *(
            (
                (
                    "chunked attended",
                    tpu.VMEM(
                        (plan.batch_size, KDA_HEADS, HEAD_DIM),
                        jnp.float32,
                    ),
                ),
            )
            if plan.batch_size > _MAX_WHOLE_KDA_BATCH
            else ()
        ),
    ]
    regions: dict[
        str,
        tuple[
            list[tuple[str, _ScratchSpec]],
            list[list[tuple[str, _ScratchSpec]]],
        ],
    ] = {"MLA": mla_region}
    if plan.attention_dp > 1:
        regions["DP projection"] = ([projection_reduction], [])
    regions["KDA"] = (kda_base, [kda_nested])
    return regions


def _build_mlp_budget_regions(
    request: _StackRequest,
    plan: _StackPlan,
    moe_scoped_scratch: tuple[_ScratchSpec, ...],
) -> dict[
    str,
    tuple[
        list[tuple[str, _ScratchSpec]],
        list[list[tuple[str, _ScratchSpec]]],
    ],
]:
    """Build static memory-budget region entries for MoE and dense layer 0."""
    moe_names = (
        *(() if POOLED_WEIGHTS else ("latent_up",)),
        "latent-up semaphore",
        "expert semaphores",
        *(
            ()
            if plan.alias_expert_scales
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
            if plan.shard_routing
            else ()
        ),
    )
    regions: dict[
        str,
        tuple[
            list[tuple[str, _ScratchSpec]],
            list[list[tuple[str, _ScratchSpec]]],
        ],
    ] = {
        "MoE": (
            list(zip(moe_names, moe_scoped_scratch, strict=True)),
            []
            if POOLED_WEIGHTS
            else [
                [
                    (
                        name,
                        tpu.VMEM(
                            plan.local_weights[name].shape,
                            plan.local_weights[name].dtype,
                        ),
                    )
                    for name in MOE_PREPARE_WEIGHTS
                ]
            ],
        )
    }
    if plan.include_layer_zero and plan.batch_size > _UNSCOPED_DENSE_MAX_BATCH:
        regions["dense layer 0"] = (
            [
                (
                    "gate/up accumulator",
                    tpu.VMEM(
                        (plan.batch_size, request.weights["dense_gu"].shape[1]),
                        jnp.float32,
                    ),
                ),
                (
                    "down accumulator",
                    tpu.VMEM((plan.batch_size, plan.hidden_size), jnp.float32),
                ),
            ],
            [],
        )
    return regions


def _plan_stack_memory(
    request: _StackRequest,
    plan: _StackPlan,
    *,
    vmem_limit_bytes: int,
) -> _StackMemoryPlan:
    """Compute pool layout, persistent/scoped scratch, and verify budget limits."""
    mla_scoped, moe_scoped = _plan_scoped_scratch(request, plan)
    args, output_shapes = _build_stack_args_and_outputs(request, plan)
    pooled_expert_names = (
        EXPERT_FAMILIES if plan.alias_expert_scales else _UNALIASED_EXPERT_NAMES
    )
    expert_pool_bytes = sum(
        pool_alias.storage_size(
            (_EXPERT_PREFETCH_SLOTS, *plan.local_weights[name].shape[1:]),
            plan.local_weights[name].dtype,
        )
        for name in pooled_expert_names
    )
    # Above B64 three slabs keep the arena at the expert slots' size.
    dense_slabs = (
        _DEFAULT_DENSE_SLABS
        if plan.batch_size <= _FOUR_DENSE_SLABS_MAX_BATCH
        else _COMPACT_DENSE_SLABS
    )
    dense_slab_bytes = (
        max(
            pool_alias.storage_size(
                (_DENSE_K_BLOCK_ROWS, request.weights["dense_gu"].shape[1]),
                request.weights["dense_gu"].dtype,
            ),
            pool_alias.storage_size(
                (_DENSE_DOWN_BLOCK_ROWS, plan.hidden_size),
                request.weights["dense_down"].dtype,
            ),
        )
        if plan.include_layer_zero
        else 0
    )
    pool_layout = _pool_layout(
        plan.pool_weights,
        alias_expert_scales=plan.alias_expert_scales,
        pooled=POOLED_WEIGHTS,
        managed=plan.pool_managed,
        dense_bytes=dense_slabs * dense_slab_bytes,
    )
    pool_bytes = pool_layout["pool_bytes"]
    if _config().vmem_report:
        _log_pool_region_report(
            plan, pool_layout, pooled_expert_names, dense_slabs * dense_slab_bytes
        )
    if pool_bytes < expert_pool_bytes or pool_bytes % VREG_LANE_COUNT != 0:
        msg = (
            f"pool_bytes ({pool_bytes}) must cover expert_pool_bytes "
            f"({expert_pool_bytes}) and align to {VREG_LANE_COUNT}"
        )
        raise ValueError(msg)
    scratch, scratch_names = _build_stack_scratch_shapes(request, plan, pool_bytes)
    persistent: list[tuple[str, _ScratchSpec]] = [
        ("output window 1", tpu.VMEM(output_shapes[1].shape, output_shapes[1].dtype)),
        *zip(scratch_names, scratch, strict=True),
    ]
    # Attention DP projects the group's rows (after the MLA workspace).
    projection_reduction = (
        "output-projection reduction",
        tpu.VMEM((plan.group_rows, _WIDE_REDUCTION_WIDTH), jnp.float32)
        if plan.attention_dp > 1
        else tpu.VMEM(
            (
                plan.batch_size,
                _HALF_HIDDEN_REDUCTION_WIDTH
                if plan.batch_size > _WIDE_REDUCTION_MAX_BATCH
                else _WIDE_REDUCTION_WIDTH,
            ),
            jnp.float32,
        ),
    )
    regions = {
        **_build_attention_budget_regions(plan, mla_scoped, projection_reduction),
        **_build_mlp_budget_regions(request, plan, moe_scoped),
    }
    vmem_report, vmem_peak = _vmem_budget(persistent, regions, limit=vmem_limit_bytes)
    if plan.pool_managed:
        # Grow the arena, all or nothing, into headroom above the spill reserve
        # so the next layer's attention projections and router (queued together
        # once the experts finish) fit without waiting for a release.
        # Future work: derive the demand from a dry-run trace of the requests.
        demand = pool_layout["attention_bytes"] + pool_alias.storage_size(
            (1, plan.hidden_size, _ROUTER_LANE_SCORE_WIDTH), jnp.dtype(jnp.bfloat16)
        )
        shortfall = max(
            0,
            -(-(demand - pool_layout["arena_bytes"]) // _ARENA_ALIGN_BYTES)
            * _ARENA_ALIGN_BYTES,
        )
        available = (
            (vmem_limit_bytes - VMEM_SPILL_RESERVE_BYTES - vmem_peak)
            // _ARENA_ALIGN_BYTES
            * _ARENA_ALIGN_BYTES
        )
        # B128's register spills exceed the reserve: its arena does not grow.
        spare = (
            shortfall if shortfall <= available and plan.batch_size < MAX_BATCH else 0
        )
        if spare > 0:
            pool_layout = _pool_layout(
                plan.pool_weights,
                alias_expert_scales=plan.alias_expert_scales,
                pooled=POOLED_WEIGHTS,
                managed=True,
                dense_bytes=dense_slabs * dense_slab_bytes,
                arena_extra=spare,
            )
            pool_bytes = pool_layout["pool_bytes"]
            pool_spec = tpu.VMEM(
                (pool_bytes // VREG_LANE_COUNT, VREG_LANE_COUNT), jnp.uint8
            )
            index = scratch_names.index("projection_expert_pool")
            scratch = (*scratch[:index], pool_spec, *scratch[index + 1 :])
            persistent = [
                (name, pool_spec if name == "projection_expert_pool" else spec)
                for name, spec in persistent
            ]
            vmem_report, vmem_peak = _vmem_budget(
                persistent, regions, limit=vmem_limit_bytes
            )
        if _config().vmem_report:
            _logger.info(
                "arena with spare memory: %.2f MiB",
                pool_layout["arena_bytes"] / MIB_BYTES,
            )
    if _config().vmem_report:
        _logger.info("memory plan, batch %d:\n%s", plan.batch_size, vmem_report)
    if vmem_peak > vmem_limit_bytes:
        msg = f"The decoder's static VMEM plan exceeds the limit:\n{vmem_report}"
        raise ValueError(msg)
    if vmem_peak + VMEM_SPILL_RESERVE_BYTES > vmem_limit_bytes:
        _logger.warning(
            "The decoder's VMEM plan leaves less than the spill reserve:\n%s",
            vmem_report,
        )
    return _StackMemoryPlan(
        args=args,
        output_shapes=output_shapes,
        expert_pool_bytes=expert_pool_bytes,
        dense_slabs=dense_slabs,
        pool_layout=pool_layout,
        mla_scoped_scratch=mla_scoped,
        moe_scoped_scratch=moe_scoped,
        scratch=scratch,
    )


def _unpack_common_refs(
    plan: _StackPlan,
    common: list[PallasRefProtocol],
) -> tuple[
    PallasRefProtocol | None,
    PallasRefProtocol | None,
    PallasRefProtocol | None,
    PallasRefProtocol | None,
    tuple[PallasRefProtocol, ...],
    PallasRefProtocol | None,
    tuple[PallasRefProtocol, PallasRefProtocol] | None,
    AttentionCommunicationWorkspace,
]:
    """Unpack DP, DCP, headroom pad, and communication workspace references."""
    group_input_vmem: PallasRefProtocol | None = None
    dp_attended_vmem: PallasRefProtocol | None = None
    group_positions_smem: PallasRefProtocol | None = None
    dp_compressed_vmem: PallasRefProtocol | None = None
    if plan.attention_dp > 1:
        (
            group_input_vmem,
            dp_attended_vmem,
            group_positions_smem,
            _dp_query_latent_vmem,
            dp_compressed_vmem,
        ) = common[:_DP_EXTRA_SCRATCH_COUNT]
        common = common[_DP_EXTRA_SCRATCH_COUNT:]
    dcp_count = len(mla_dcp.PERSISTENT_NAMES)
    dcp_persistent = tuple(common[:dcp_count])
    common = common[dcp_count:]
    if _config().vmem_pad_kb:
        pad = common[-1]
        pad[...] = jnp.zeros(pad.shape, jnp.float32)
        headroom_pad: PallasRefProtocol | None = pad
        common = common[:-1]
    else:
        headroom_pad = None
    gather_wire = None
    if plan.b4_attention_gather:
        *common, gather_wire = common
    pair_semaphores: tuple[PallasRefProtocol, PallasRefProtocol] | None = None
    if plan.pair_split_gather:
        *common, pair_send, pair_receive = common
        pair_semaphores = (pair_send.at[0], pair_receive.at[0])
    routed_consumed = None
    if plan.moe_routed_reduce_scatter:
        *common, routed_consumed = common
    communication_workspace = AttentionCommunicationWorkspace(
        *common,
        attention_gather_vmem=gather_wire,
        routed_consumed=routed_consumed,
        pair_semaphores=pair_semaphores,
    )
    return (
        group_input_vmem,
        dp_attended_vmem,
        group_positions_smem,
        dp_compressed_vmem,
        dcp_persistent,
        headroom_pad,
        pair_semaphores,
        communication_workspace,
    )


def _build_runtime_context(
    plan: _StackPlan,
    mem_plan: _StackMemoryPlan,
    position_ref: PallasRefProtocol,
    state_row_ref: PallasRefProtocol,
    refs: tuple[PallasRefProtocol, ...],
) -> _StackRuntimeContext:
    """Unpack ``body`` Pallas references into a typed ``_StackRuntimeContext``."""
    table_ref, *rest = refs
    slots_ref: PallasRefProtocol | None = None
    if plan.slot_reads:
        slots_ref, *rest = rest
    active_rows: jax.Array | None = None
    if plan.has_live_rows:
        live_ref, *rest = rest
        # Stack scalar integers before comparison: Mosaic cannot concatenate
        # scalar predicates into a vector through bitcast_vreg.
        active_rows = jnp.stack([live_ref[row] for row in range(plan.batch_size)]) != 0
    prefix_ref, embedding_ref, *rest = rest
    layer_weights = dict(zip(NAMES, rest[: len(NAMES)], strict=True))
    output_residual_ref, final_normalization_ref = rest[
        len(NAMES) : len(NAMES) + len(EPILOGUE)
    ]
    dense_start = len(NAMES) + len(EPILOGUE)
    dense_end = dense_start + len(plan.dense_names)
    dense_refs = dict(zip(plan.dense_names, rest[dense_start:dense_end], strict=True))
    state_start = dense_end
    convolution_ref, recurrent_ref = rest[state_start : state_start + 2]
    raw_pool_refs = rest[state_start + 2 : state_start + plan.state_count]
    pool_ref = tuple(raw_pool_refs[index] for index in plan.pool_map)
    output_end = state_start + plan.state_count + _STACK_OUTPUT_COUNT
    (
        output_ref,
        blocks_output_ref,
        convolution_output_ref,
        recurrent_output_ref,
        key_output_ref,
        value_output_ref,
        aux_output_ref,
    ) = rest[state_start + plan.state_count : output_end]
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
    ) = rest[output_end:]
    (
        group_input_vmem,
        dp_attended_vmem,
        group_positions_smem,
        dp_compressed_vmem,
        dcp_persistent,
        headroom_pad,
        pair_semaphores,
        communication_workspace,
    ) = _unpack_common_refs(plan, common)
    if plan.compact_outputs:
        # Attention's whole-batch output argument is shape-checked but unused
        # with token scatter; the normalized input has its shape.
        attention_vmem = normalized_vmem
    # In sequence-row mode the chained recurrence reads only the first row's
    # input state; with a state row the accepted row of the previous step is
    # read directly into that slot.
    selected_state_row = state_row_ref[0] if plan.select_state_row else None
    # With the arena manager, where each layer's weights land (and when their
    # copies start) follows from the order of requests and releases.
    pool = (
        VmemPool(
            cast("ArenaRefProtocol", projection_expert_pool_vmem),
            mem_plan.pool_layout["arena_bytes"],
            report=_config().pool_report,
        )
        if plan.pool_managed
        else None
    )
    dp_owned_vmem = (
        _bf16_rows_view(
            communication_workspace.attention_output_vmem,
            plan.dp_held_rows,
            plan.hidden_size,
        )
        if plan.attention_dp_ownership
        else None
    )
    # Expert/projection ABIs retain their existing pair ownership. Only mixture
    # rows and the following gathers use eight independent owners.
    sp8_rows_vmem = dp_owned_vmem if plan.sp8_residual else None
    return _StackRuntimeContext(
        plan=plan,
        mem_plan=mem_plan,
        position_ref=position_ref,
        table_ref=table_ref,
        slots_ref=slots_ref,
        active_rows=active_rows,
        selected_state_row=selected_state_row,
        prefix_ref=prefix_ref,
        embedding_ref=embedding_ref,
        layer_weights=layer_weights,
        output_residual_ref=output_residual_ref,
        final_normalization_ref=final_normalization_ref,
        dense_refs=dense_refs,
        convolution_ref=convolution_ref,
        recurrent_ref=recurrent_ref,
        pool_ref=pool_ref,
        output_ref=output_ref,
        blocks_output_ref=blocks_output_ref,
        convolution_output_ref=convolution_output_ref,
        recurrent_output_ref=recurrent_output_ref,
        key_output_ref=key_output_ref,
        value_output_ref=value_output_ref,
        aux_output_ref=aux_output_ref,
        prefix_vmem=prefix_vmem,
        blocks_vmem=blocks_vmem,
        normalized_vmem=normalized_vmem,
        attention_vmem=attention_vmem,
        mixture_vmem=mixture_vmem,
        attention_residual_vmem=attention_residual_vmem,
        mlp_residual_vmem=mlp_residual_vmem,
        attention_norm_vmem=attention_norm_vmem,
        ffn_norm_vmem=ffn_norm_vmem,
        router_vmem=router_vmem,
        router_semaphores=router_semaphores,
        projection_expert_pool_vmem=projection_expert_pool_vmem,
        pool_semaphores=pool_semaphores,
        norm_semaphores=norm_semaphores,
        auxiliary_vmem=(
            convolution_filter_vmem,
            exp_log_decay_rate_vmem,
            time_step_bias_vmem,
            output_normalization_vmem,
            convolution_history_vmem,
            recurrent_state_vmem,
        ),
        metadata_vmem=(
            query_norms_vmem,
            key_value_norms_vmem,
            routing_biases_vmem,
            latent_norms_vmem,
        ),
        auxiliary_semaphores=auxiliary_semaphores,
        metadata_semaphores=metadata_semaphores,
        owned_rows_vmem=owned_rows_vmem,
        aux_semaphore=aux_semaphore,
        dense_semaphores=dense_semaphores,
        stream_semaphores=stream_semaphores,
        group_input_vmem=group_input_vmem,
        dp_attended_vmem=dp_attended_vmem,
        group_positions_smem=group_positions_smem,
        dp_compressed_vmem=dp_compressed_vmem,
        dcp_persistent=dcp_persistent,
        headroom_pad=headroom_pad,
        pair_semaphores=pair_semaphores,
        communication_workspace=communication_workspace,
        dp_owned_vmem=dp_owned_vmem,
        sp8_rows_vmem=sp8_rows_vmem,
        pool=pool,
    )


def _kda_state_source(
    ctx: _StackRuntimeContext,
    ref: PallasRefProtocol,
    state_index: int | jax.Array,
) -> PallasRefProtocol:
    """Return the HBM slice holding input KDA state for ``state_index``."""
    if ctx.plan.select_state_row:
        selected = _require_array(ctx.selected_state_row, "selected_state_row")
        row = ctx.slots_ref[selected] if ctx.slots_ref is not None else selected
        return ref.at[pl.ds(row, 1), state_index]
    return ref.at[:, state_index]


def _kda_state_destination(
    ctx: _StackRuntimeContext,
    vmem: PallasRefProtocol,
) -> PallasRefProtocol:
    """Return the scratch destination slice for input KDA state."""
    return vmem.at[pl.ds(0, 1)] if ctx.plan.select_state_row else vmem


def _kda_state_reads(
    ctx: _StackRuntimeContext,
    ref: PallasRefProtocol,
    vmem: PallasRefProtocol,
    semaphore: PallasRefProtocol,
    state_index: int | jax.Array,
) -> tuple[AsyncCopyProtocol, ...]:
    """Build async copies for a layer's KDA states (bulk or per-slot)."""
    if ctx.slots_ref is None or ctx.plan.select_state_row:
        return (
            tpu.make_async_copy(
                _kda_state_source(ctx, ref, state_index),
                _kda_state_destination(ctx, vmem),
                semaphore,
            ),
        )
    slots_ref = ctx.slots_ref
    return tuple(
        tpu.make_async_copy(
            ref.at[pl.ds(slots_ref[row], 1), state_index],
            vmem.at[pl.ds(row, 1)],
            semaphore,
        )
        for row in range(ctx.plan.batch_size)
    )


def _pool_view(
    ctx: _StackRuntimeContext,
    name: str,
    base: int = 0,
) -> PallasRefProtocol:
    """Return the pooled view of one weight at its layout offset from ``base``."""
    value = ctx.plan.pool_weights[name]
    size = math.prod(value.shape) * value.dtype.itemsize
    return cast(
        "PallasRefProtocol",
        pool_alias.view(
            ctx.projection_expert_pool_vmem.at[
                pl.ds(
                    (base + ctx.mem_plan.pool_layout[name]) // VREG_LANE_COUNT,
                    size // VREG_LANE_COUNT,
                ),
                :,
            ],
            value.shape,
            value.dtype,
        ),
    )


def _pool_views(
    ctx: _StackRuntimeContext,
    names: Sequence[str],
    *,
    experts: bool = False,
    base: int = 0,
) -> list[PallasRefProtocol]:
    """Return contiguous pooled views for ``names`` starting at byte ``base``."""
    views: list[PallasRefProtocol] = []
    offset = base
    for name in names:
        value = ctx.plan.pool_weights[name]
        logical_shape = value.shape[1:] if experts else value.shape
        view_shape = (
            (_EXPERT_PREFETCH_SLOTS, *logical_shape) if experts else logical_shape
        )
        size = math.prod(view_shape) * value.dtype.itemsize
        storage_size = pool_alias.storage_size(view_shape, value.dtype)
        views.append(
            cast(
                "PallasRefProtocol",
                pool_alias.view(
                    ctx.projection_expert_pool_vmem.at[
                        pl.ds(offset // VREG_LANE_COUNT, size // VREG_LANE_COUNT),
                        :,
                    ],
                    view_shape,
                    value.dtype,
                ),
            )
        )
        offset += storage_size
    return views


def _attention_views(
    ctx: _StackRuntimeContext,
    base: int = 0,
) -> tuple[tuple[PallasRefProtocol, ...], tuple[PallasRefProtocol, ...]]:
    """Return KDA and MLA input projection views for one layer from ``base``."""
    return (
        tuple(_pool_views(ctx, KDA[:_KDA_PROJECTION_COUNT], base=base)),
        tuple(_pool_views(ctx, MLA[:_MLA_INPUT_PROJECTION_COUNT], base=base)),
    )


def _aliased_expert_views(
    ctx: _StackRuntimeContext,
    base: int = 0,
) -> tuple[PallasRefProtocol, PallasRefProtocol, PallasRefProtocol, PallasRefProtocol]:
    """Return pooled ``(gu, gus, down, ds)`` views when scales are aliased."""
    views = _pool_views(ctx, _ALIASED_EXPERT_VIEW_ORDER, experts=True, base=base)
    return views[0], views[2], views[1], views[3]


def _unaliased_expert_views(
    ctx: _StackRuntimeContext,
    base: int = 0,
) -> tuple[PallasRefProtocol, PallasRefProtocol]:
    """Return pooled ``(gu, down)`` views when scales stay in scoped scratch."""
    views = _pool_views(ctx, _UNALIASED_EXPERT_NAMES, experts=True, base=base)
    return views[0], views[1]


def _arena_router(
    pool: VmemPool,
    hidden_size: int,
    offset: int,
) -> PallasRefProtocol:
    """Return the arena router slice at byte ``offset``."""
    return cast(
        "PallasRefProtocol",
        pool.view(
            offset, (1, hidden_size, _ROUTER_LANE_SCORE_WIDTH), jnp.dtype(jnp.bfloat16)
        ),
    )


def _request_router(
    ctx: _StackRuntimeContext,
    pool: VmemPool,
    index: int | jax.Array,
    *,
    predicate: bool | jax.Array | None = None,
) -> None:
    """Queue layer ``index``'s router weight copy into the arena."""
    pool.request(
        "router",
        pool_alias.storage_size(
            (1, ctx.plan.hidden_size, _ROUTER_LANE_SCORE_WIDTH), jnp.dtype(jnp.bfloat16)
        ),
        lambda offset, layer_idx: _router_copy(
            ctx.layer_weights["router"].at[cast("int | jax.Array", layer_idx) - 1],
            _arena_router(pool, ctx.plan.hidden_size, offset),
            ctx.router_semaphores,
            wait=False,
            initialize_padding=False,
        ),
        index=index,
        predicate=predicate,
    )


def _request_attention(
    ctx: _StackRuntimeContext,
    pool: VmemPool,
    index: int | jax.Array,
    *,
    predicate: bool | jax.Array | None = None,
) -> None:
    """Queue layer ``index``'s attention input projections into the arena."""
    pool.request(
        "attention",
        ctx.mem_plan.pool_layout["attention_bytes"],
        lambda offset, layer_idx: _prefetch_attention(
            ctx, cast("int | jax.Array", layer_idx), base=offset
        ),
        index=index,
        predicate=predicate,
    )


def _for_dp_owned_rows(
    plan: _StackPlan,
    run: Callable[[int], None],
) -> None:
    """Invoke ``run(first)`` with this rank's static first batch row."""
    if plan.sp_owners == _SINGLE_SP_OWNER:
        run(0)
        return
    rank = jax.lax.axis_index(AXIS_TP)
    pair = (rank % mla_dcp.HOST_LANES) // (mla_dcp.HOST_LANES // plan.sp_owners)
    group = rank // (TP_SIZE // plan.attention_dp)
    for owner in range(plan.sp_owners):
        for data_group in range(plan.attention_dp):

            @pl.when((pair == owner) & (group == data_group))
            def owned_region(
                owner: int = owner,
                data_group: int = data_group,
            ) -> None:
                """Run callback for this owner pair and attention DP group."""
                run(
                    owner * plan.tokens_per_pair + data_group * plan.group_rows_per_pair
                )


def _initialize_blocks(
    ctx: _StackRuntimeContext,
    embedding_vmem: PallasRefProtocol,
) -> None:
    """Seed residual snapshot slot 0 from ``embedding_vmem``."""
    plan = ctx.plan
    if plan.attention_dp_ownership:

        def initialize_owned_blocks(first: int) -> None:
            """Copy this DP rank's owned embedding rows into snapshot slot 0."""
            for local in range(plan.dp_held_rows):
                ctx.blocks_vmem[local // 2, pl.ds((local % 2) * SUBLANE_ROWS, 1)] = (
                    embedding_vmem[pl.ds(first + local, 1)].astype(jnp.bfloat16)
                )

        _for_dp_owned_rows(plan, initialize_owned_blocks)
    elif plan.blocks_owned:
        owner_of_rank = (jax.lax.axis_index(AXIS_TP) % mla_dcp.HOST_LANES) // (
            mla_dcp.HOST_LANES // plan.sp_owners
        )
        for owner_index in range(plan.sp_owners):

            @pl.when(owner_of_rank == owner_index)
            def initialize_owned_blocks(owner_index: int = owner_index) -> None:
                """Copy this sequence-parallel owner's rows into slot 0."""
                for local in range(plan.owned_rows):
                    ctx.blocks_vmem[
                        local // 2, pl.ds((local % 2) * SUBLANE_ROWS, 1)
                    ] = embedding_vmem[
                        pl.ds(owner_index * plan.owned_rows + local, 1)
                    ].astype(jnp.bfloat16)
    elif plan.blocks_block_major:
        for row in range(plan.batch_size):
            ctx.blocks_vmem[row // 2, pl.ds((row % 2) * SUBLANE_ROWS, 1)] = (
                embedding_vmem[pl.ds(row, 1)].astype(jnp.bfloat16)
            )
    else:
        ctx.blocks_vmem[:, 0] = embedding_vmem[...].astype(jnp.float32)


def _init_dp_positions_and_prefix(ctx: _StackRuntimeContext) -> None:
    """Populate DP group positions and load prefix/embedding into scratch."""
    plan = ctx.plan
    if plan.attention_dp > 1:
        group_positions_smem = _require_ref(
            ctx.group_positions_smem, "group_positions_smem"
        )
        # DP group d attends, of the rows owner p holds, the r rows
        # tokens_per_pair * p + d * r + j (group row p * r + j).
        data_group = jax.lax.axis_index(AXIS_TP) // (TP_SIZE // plan.attention_dp)
        for row in range(plan.group_rows):
            pair_index, within = divmod(row, plan.group_rows_per_pair)
            group_positions_smem[row] = ctx.position_ref[
                plan.tokens_per_pair * pair_index
                + data_group * plan.group_rows_per_pair
                + within
            ]

    # The prefix and embedding inputs and the hidden output stay in HBM:
    # one DMA each at entry/exit instead of whole-batch windows live for the
    # entire kernel.
    def load_prefix(
        staged: PallasRefProtocol,
        semaphore: PallasRefProtocol,
    ) -> None:
        """DMA prefix rows from HBM and slice DP-owned rows when active."""
        copy = tpu.make_async_copy(ctx.prefix_ref, staged, semaphore)
        copy.start()
        copy.wait()
        if plan.attention_dp_ownership:

            def take_owned_rows(first: int) -> None:
                """Slice this DP rank's owned prefix rows out of ``staged``."""
                ctx.prefix_vmem[: plan.dp_held_rows] = staged[
                    first : first + plan.dp_held_rows
                ]

            _for_dp_owned_rows(plan, take_owned_rows)

    if plan.attention_dp_ownership:
        pl.run_scoped(
            load_prefix,
            tpu.VMEM(ctx.prefix_ref.shape, ctx.prefix_ref.dtype),
            tpu.SemaphoreType.DMA,
        )
    else:
        pl.run_scoped(
            functools.partial(load_prefix, ctx.prefix_vmem),
            tpu.SemaphoreType.DMA,
        )
    ctx.blocks_vmem[...] = jnp.zeros(ctx.blocks_vmem.shape, ctx.blocks_vmem.dtype)

    def staged_blocks(
        embedding_vmem: PallasRefProtocol,
        semaphore: PallasRefProtocol,
    ) -> None:
        """DMA embeddings into scoped scratch and seed residual snapshot 0."""
        copy = tpu.make_async_copy(ctx.embedding_ref, embedding_vmem, semaphore)
        copy.start()
        copy.wait()
        _initialize_blocks(ctx, embedding_vmem)

    # With layer zero included, its pre-mixture has no active snapshots. That
    # layer seeds snapshot zero from prefix before any later read. Keep all
    # snapshot slots zero-initialized so masked arithmetic stays finite; only
    # omit the redundant embedding DMA and row insertion.
    if not (
        plan.include_layer_zero
        and plan.attention_dp == DP2_GROUPS
        and plan.batch_size in _DP_B64_B128_BATCHES
        and plan.sequence_rows == 1
    ):
        pl.run_scoped(
            staged_blocks,
            tpu.VMEM(ctx.embedding_ref.shape, ctx.embedding_ref.dtype),
            tpu.SemaphoreType.DMA,
        )


def _issue_first_layer_prefetches(ctx: _StackRuntimeContext) -> None:
    """Zero-pad odd router lanes when needed and prefetch the first layer."""
    plan = ctx.plan
    # The lane-1 projection uses 384 of its 512 columns. This persistent tail
    # is never written elsewhere, so initialize it once rather than on every
    # layer prefetch.
    if plan.include_layer_zero:
        # Layer 0 has no router; layer 1's is prefetched after the dense MLP.
        if not plan.pool_managed:

            @pl.when(jax.lax.axis_index(AXIS_TP) % _RANK_PAIR_SIZE == 1)
            def initialize_router_padding() -> None:
                """Zero-pad unused trailing router columns on odd lanes."""
                ctx.router_vmem[0, :, _ODD_LANE_ROUTER_COLUMNS:] = jnp.zeros(
                    (ctx.router_vmem.shape[1], VREG_LANE_COUNT),
                    jnp.bfloat16,
                )
    else:
        _prefetch_router(ctx, jnp.int32(1), initialize_padding=True)
    _prefetch_norms(ctx, jnp.int32(plan.first_layer))
    if ctx.pool is not None:
        _request_attention(ctx, ctx.pool, jnp.int32(plan.first_layer))
    else:
        _prefetch_attention(ctx, jnp.int32(plan.first_layer))
    _prefetch_auxiliary(ctx, jnp.int32(plan.first_layer))
    _prefetch_metadata(
        ctx,
        jnp.int32(plan.first_layer),
        mixture_metadata=not plan.include_layer_zero,
    )
    if ctx.pool is not None and not plan.include_layer_zero:
        _request_router(ctx, ctx.pool, jnp.int32(1))


def _run_stack_prologue(ctx: _StackRuntimeContext) -> None:
    """Barrier, load prefix/embedding into scratch, and issue layer-0 prefetches."""
    collectives32.barrier()
    ctx.communication_workspace.reduction_phase[0] = 0
    _init_dp_positions_and_prefix(ctx)
    _issue_first_layer_prefetches(ctx)


def _owner_indices(
    plan: _StackPlan,
    pair: jax.Array,
) -> tuple[jax.Array, ...] | range:
    """Return the owner-pair iteration range (single traced pair or static)."""
    return (pair,) if plan.owner_generic else range(plan.sp_owners)


def _owner_when(
    plan: _StackPlan,
    pair: jax.Array,
    owner: int | jax.Array,
) -> Callable[[Callable[[], None]], None]:
    """Return a decorator executing its body when ``pair == owner``."""
    if plan.owner_generic:
        return lambda function: function()
    return pl.when(pair == owner)


def _owner_start(
    plan: _StackPlan,
    start: int | jax.Array,
) -> int | jax.Array:
    """Annotate ``start`` as 16-aligned in owner-generic mode."""
    return (
        pl.multiple_of(cast("jax.Array", start), _BF16_TILE_ROWS)
        if plan.owner_generic
        else start
    )


def _owner_block_row(
    plan: _StackPlan,
    global_row: int | jax.Array,
    local_row: int,
) -> int | jax.Array:
    """Map an owner's local row to its snapshot index."""
    # Above B16 each owner holds only its own rows' snapshots.
    return local_row if plan.owner_generic else global_row


def _owner_prefix_row(
    ctx: _StackRuntimeContext,
    global_row: int | jax.Array,
    row_slice: object,
    local_row: int,
) -> jax.Array:
    """Load one prefix row for ``local_row`` under owner-generic or direct mode."""
    if ctx.plan.owner_generic:
        return ctx.prefix_vmem[row_slice][local_row : local_row + 1]
    return ctx.prefix_vmem[pl.ds(global_row, 1)]


def _block_rows(
    ctx: _StackRuntimeContext,
    row: int | jax.Array,
    count: int = _RESIDUAL_SNAPSHOT_BLOCKS,
) -> jax.Array:
    """Return the residual snapshots of one batch row, ``[count, hidden]``."""
    if ctx.plan.blocks_block_major:
        local_row = row % ctx.plan.block_rows_held  # owned rows are held locally
        return ctx.blocks_vmem[
            local_row // 2, pl.ds((local_row % 2) * SUBLANE_ROWS, count)
        ]
    return (
        ctx.blocks_vmem[row]
        if count == _RESIDUAL_SNAPSHOT_BLOCKS
        else ctx.blocks_vmem[row, :count]
    )


def _residual_mix(
    ctx: _StackRuntimeContext,
    destination: PallasRefProtocol,
    folded: PallasRefProtocol,
    active_block_count: int | jax.Array,
) -> None:
    """Run whole-batch row-wise residual mixture into ``destination``."""
    _residual_mixture_by_row(
        ctx.blocks_vmem[...],
        ctx.prefix_vmem[...],
        destination,
        folded,
        active_block_count,
        batch_size=ctx.plan.batch_size,
        epsilon=ctx.plan.eps,
        relaxed_reduction_order=(ctx.plan.residual_reduction_order == "relaxed"),
    )


def _store_mixed_rows(
    ctx: _StackRuntimeContext,
    compute_row: Callable[[int], jax.Array],
    destination: PallasRefProtocol,
) -> None:
    """Write ``destination``'s owned rows, isolating regions above two rows."""
    # Above two rows per pair (B16) each row runs in its own region: unrolled
    # rows would keep every row's 8 x hidden FP32 blocks live and spill. DP2
    # B4's four rows (one owner) stay unrolled.
    rows = destination.shape[0]
    if rows <= _MAX_UNROLLED_MIX_ROWS or ctx.plan.sp_owners == _SINGLE_SP_OWNER:
        destination[...] = jnp.concatenate(
            [compute_row(local_row) for local_row in range(rows)],
            axis=0,
        )
        return
    for local_row in range(rows):
        # A runtime condition (never false) so the row is its own region.
        @pl.when(ctx.communication_workspace.reduction_phase[0] >= 0)
        def store_row(local_row: int = local_row) -> None:
            """Compute and store one mixed residual row."""
            destination[pl.ds(local_row, 1)] = compute_row(local_row)


def _normalize_rows(
    ctx: _StackRuntimeContext,
    destination: PallasRefProtocol,
    scale: PallasRefProtocol,
) -> None:
    """Apply RMS normalization in place to ``destination`` with ``scale``."""
    value = destination[...].astype(jnp.float32)
    destination[...] = (
        value
        * jax.lax.rsqrt(jnp.mean(value * value, axis=1, keepdims=True) + ctx.plan.eps)
        * scale[...]
    ).astype(jnp.bfloat16)


def _sp8_owned_value(sp8_rows_vmem: PallasRefProtocol) -> jax.Array:
    """Select this rank's 8 owned rows from the 16-row SP8 buffer."""
    value = sp8_rows_vmem[...]
    return jnp.where(
        jax.lax.axis_index(AXIS_TP) % _RANK_PAIR_SIZE == 0,
        value[:SUBLANE_ROWS],
        value[SUBLANE_ROWS:_BF16_TILE_ROWS],
    )


def _store_sp8_mixed_rows(
    ctx: _StackRuntimeContext,
    mix_row: Callable[[int], jax.Array],
) -> None:
    """Write SP8-owned mixed rows into a native 16-row scratch tile."""
    sp8_rows_vmem = _require_ref(ctx.sp8_rows_vmem, "sp8_rows_vmem")
    # RMS keeps a native 16-row geometry; initialize unused rows rather than
    # reading undefined scratch on the first layer.
    sp8_rows_vmem[...] = jnp.zeros(
        (_BF16_TILE_ROWS, ctx.plan.hidden_size), jnp.bfloat16
    )
    for lane in range(_RANK_PAIR_SIZE):

        @pl.when(jax.lax.axis_index(AXIS_TP) % _RANK_PAIR_SIZE == lane)
        def mix_half(lane: int = lane) -> None:
            """Mix the 8 rows owned by ``lane``."""
            for local_row in range(SUBLANE_ROWS):

                @pl.when(ctx.communication_workspace.reduction_phase[0] >= 0)
                def store_row(local_row: int = local_row) -> None:
                    """Compute and write one SP8-owned residual row."""
                    row = lane * SUBLANE_ROWS + local_row
                    # Index the full transformed view directly, avoiding an
                    # unsupported nested half-row view.
                    sp8_rows_vmem[pl.ds(row, 1)] = mix_row(row)


def _mix_dp_rows(
    ctx: _StackRuntimeContext,
    folded: PallasRefProtocol,
    active_blocks: int | jax.Array,
    opts: _DpMixOptions,
) -> None:
    """Mix DP-owned residual rows and optionally snapshot block starts."""
    plan = ctx.plan
    dp_owned_vmem = _require_ref(ctx.dp_owned_vmem, "dp_owned_vmem")
    rows = slice(0, plan.dp_held_rows)
    if opts.attention_output:
        ctx.prefix_vmem[rows] = jnp.where(
            opts.layer_index % _RESIDUAL_BLOCK_STRIDE == 0,
            dp_owned_vmem[...],
            ctx.prefix_vmem[rows] + dp_owned_vmem[...],
        ).astype(jnp.bfloat16)

    def mix_row(row: int) -> jax.Array:
        """Mix residual snapshots with ``prefix_vmem[row]``."""
        return _residual_mixture_row(
            _block_rows(ctx, row),
            ctx.prefix_vmem[row : row + 1],
            folded,
            active_blocks,
            epsilon=plan.eps,
            relaxed_reduction_order=plan.residual_reduction_order == "relaxed",
        )

    if plan.sp8_residual:
        _store_sp8_mixed_rows(ctx, mix_row)
    else:
        _store_mixed_rows(
            ctx,
            mix_row,
            ctx.normalized_vmem
            if opts.start_block and plan.kda_shard_swap
            else dp_owned_vmem,
        )
    if opts.start_block:

        @pl.when(opts.layer_index % _RESIDUAL_BLOCK_STRIDE == 0)
        def snapshot() -> None:
            """Record prefix rows into the newly started residual block."""
            for row in range(plan.dp_held_rows):
                half = ctx.blocks_vmem.at[
                    row // 2, pl.ds((row % 2) * SUBLANE_ROWS, SUBLANE_ROWS)
                ]
                mask = jax.lax.broadcasted_iota(jnp.int32, half.shape, 0)
                half[...] = jnp.where(
                    mask == opts.layer_index // _RESIDUAL_BLOCK_STRIDE,
                    ctx.prefix_vmem[row : row + 1],
                    half[...],
                )


def _normalize_dp_rows(
    ctx: _StackRuntimeContext,
    scale: PallasRefProtocol,
    *,
    split: bool = False,
) -> None:
    """Normalize DP-owned rows, staging into a 16-row tile at batch 64."""
    plan = ctx.plan
    if split:
        # Preserve the 16-row normalization geometry. Unowned rows are not
        # communicated or consumed by any later computation.
        _normalize_rows(ctx, _require_ref(ctx.sp8_rows_vmem, "sp8_rows_vmem"), scale)
        return
    dp_owned_vmem = _require_ref(ctx.dp_owned_vmem, "dp_owned_vmem")
    if plan.batch_size != _DP_NATIVE_NORM_BATCH:
        _normalize_rows(ctx, dp_owned_vmem, scale)
        return
    # Normalize in a native 16-row tile at the group's row lanes
    # (owned_rows_vmem is free until the following gather).
    group = jax.lax.axis_index(AXIS_TP) // (TP_SIZE // plan.attention_dp)
    for data_group in range(plan.attention_dp):

        @pl.when(group == data_group)
        def stage_native_rows(data_group: int = data_group) -> None:
            """Place this DP group's rows at its lanes in a 16-row tile."""
            first = data_group * plan.group_rows_per_pair
            parts: list[jax.Array] = []
            if first:
                parts.append(jnp.zeros((first, plan.hidden_size), jnp.bfloat16))
            parts.append(dp_owned_vmem[...])
            tail = _BF16_TILE_ROWS - first - plan.group_rows_per_pair
            if tail:
                parts.append(jnp.zeros((tail, plan.hidden_size), jnp.bfloat16))
            ctx.owned_rows_vmem[...] = jnp.concatenate(tuple(parts), axis=0)

    _normalize_rows(ctx, ctx.owned_rows_vmem, scale)
    native_rows = ctx.owned_rows_vmem[...]
    for data_group in range(plan.attention_dp):

        @pl.when(group == data_group)
        def select_normalized_rows(data_group: int = data_group) -> None:
            """Extract this DP group's normalized rows from the 16-row tile."""
            first = data_group * plan.group_rows_per_pair
            dp_owned_vmem[...] = native_rows[first : first + plan.group_rows_per_pair]


def _swap_kda_projections(
    ctx: _StackRuntimeContext,
    stream: ChunkedWeightStream,
) -> Callable[[int], jax.Array]:
    """Exchange DP2 B2/B4 KDA input projections with the peer DP group."""
    # DP2 B2/B4's shard swap: this rank projects its own group's held shard for
    # every row, sends the other group's rows to its counterpart there (which
    # projects the other shard) and gets this group's rows of that shard back.
    plan = ctx.plan
    rank = jax.lax.axis_index(AXIS_TP)
    group = rank // (TP_SIZE // plan.attention_dp)
    projected = stream.dot(
        ctx.normalized_vmem[...],
        ctx.communication_workspace.reduction_phase[0] >= 0,
    )
    projected = projected.astype(jnp.bfloat16)
    own = jnp.where(
        group == 0, projected[: plan.group_rows], projected[plan.group_rows :]
    )
    other = jnp.where(
        group == 0, projected[plan.group_rows :], projected[: plan.group_rows]
    )
    used = plan.group_rows * projected.shape[1] // VREG_LANE_COUNT
    # The BF16 view of the FP32 lanes slices in row pairs: pad B2's 13-row
    # payload.
    payload = used + used % _BF16_PER_FP32_WORD
    local = ctx.communication_workspace.shared_local_reduction_vmem
    send = local.at[_DP_SWAP_SEND_LANE].bitcast(jnp.bfloat16).at[pl.ds(0, payload), :]
    receive = (
        local.at[_DP_SWAP_RECV_LANE].bitcast(jnp.bfloat16).at[pl.ds(0, payload), :]
    )
    wire = other.reshape(used, VREG_LANE_COUNT)
    if payload != used:
        wire = jnp.concatenate(
            [wire, jnp.zeros((payload - used, VREG_LANE_COUNT), jnp.bfloat16)]
        )
    send[...] = wire
    copy = tpu.make_async_remote_copy(
        send,
        receive,
        ctx.communication_workspace.send_semaphores.at[_DP_SWAP_SEMAPHORE_SLOT],
        ctx.communication_workspace.receive_semaphores.at[_DP_SWAP_SEMAPHORE_SLOT],
        device_id=(rank ^ (TP_SIZE // plan.attention_dp),),
        device_id_type=pl.DeviceIdType.MESH,
    )
    copy.start()

    def rows(order: int) -> jax.Array:
        """Return own shard rows (``order == 0``) or peer shard rows."""
        if order == 0:
            return own
        copy.wait()
        return receive[...][:used].reshape(own.shape)

    return rows


def _gather_dp_rows(
    ctx: _StackRuntimeContext,
    destination: PallasRefProtocol,
    *,
    split: bool = False,
) -> None:
    """Gather DP-owned rows into ``destination`` (and ``owned_rows_vmem``)."""
    comm = ctx.communication_workspace
    if split:
        _gather_sp8_dp_moe_rows(
            _sp8_owned_value(_require_ref(ctx.sp8_rows_vmem, "sp8_rows_vmem")),
            ctx.owned_rows_vmem,
            destination,
            comm.shared_local_reduction_vmem,
            comm.send_semaphores,
            comm.receive_semaphores,
        )
        return
    dp_owned_vmem = _require_ref(ctx.dp_owned_vmem, "dp_owned_vmem")
    # Restore the router's owned input as well as the complete MoE input. No
    # arithmetic occurs after normalization. Scratch lanes 4..7 for DP
    # transport are disjoint from local-gather lanes 0..3.
    if ctx.plan.sp_owners == _SINGLE_SP_OWNER:
        destination[...] = dp_owned_vmem[...]
        return
    _gather_dp_moe_rows(
        dp_owned_vmem[...],
        ctx.owned_rows_vmem,
        destination,
        comm.shared_local_reduction_vmem,
        comm.send_semaphores,
        comm.receive_semaphores,
        ctx.plan.attention_dp,
        pair_semaphores=comm.pair_semaphores,
    )


def _prefetch_norms(
    ctx: _StackRuntimeContext,
    index: int | jax.Array,
) -> None:
    """Start async DMAs for layer ``index``'s four normalization vectors."""
    destinations = (
        ctx.attention_residual_vmem,
        ctx.mlp_residual_vmem,
        ctx.attention_norm_vmem,
        ctx.ffn_norm_vmem,
    )
    for slot, (name, destination) in enumerate(zip(NORMS, destinations, strict=True)):
        tpu.make_async_copy(
            ctx.layer_weights[name].at[index],
            destination,
            ctx.norm_semaphores.at[slot],
        ).start()


def _weight_storage_bytes(plan: _StackPlan, name: str) -> int:
    """Return aligned pool byte size of weight ``name``."""
    value = plan.pool_weights[name]
    return pool_alias.storage_size(value.shape, value.dtype)


def _weight_stream(
    ctx: _StackRuntimeContext,
    name: str,
    source: PallasRefProtocol,
    full: PallasRefProtocol,
    *,
    offset: int | None = None,
) -> ChunkedWeightStream | None:
    """Return the ``ChunkedWeightStream`` for ``name``, or ``None`` if bulk."""
    plan = ctx.plan
    axis = STREAMED_WEIGHTS[name][0]
    chunks, depth = plan.stream_config.get(name, (1, 1))
    if chunks == 1:
        return None
    base = _stream_base(name, depth)
    order: Callable[[int], int | jax.Array] | None = None
    if axis == "rows":
        # Output projections stream in rank-pair feature-row halves, own rows
        # first: per shard, all own halves before the peer's.
        expected_chunks = _RANK_PAIR_SIZE * plan.attention_dp
        if chunks != expected_chunks:
            msg = f"{name} streams as its {expected_chunks} pair halves"
            raise ValueError(msg)
        lane = jax.lax.axis_index(AXIS_TP) % _RANK_PAIR_SIZE

        def _row_chunk_order(position: int) -> jax.Array:
            """Map logical stream position to rank-pair half order."""
            return (position % plan.attention_dp) * _RANK_PAIR_SIZE + (
                lane ^ (position // plan.attention_dp)
            )

        order = _row_chunk_order
    extent = source.shape[1 if axis == "columns" else 0]
    unit = VREG_LANE_COUNT if axis == "columns" else _BF16_TILE_ROWS
    if axis == "columns" and plan.attention_dp > 1:
        # Chunks never straddle two held shards.
        if chunks % plan.attention_dp:
            msg = f"{name}: chunks must be a multiple of the attention DP degree"
            raise ValueError(msg)
        sizes = (
            _chunk_sizes(extent // plan.attention_dp, chunks // plan.attention_dp, unit)
            * plan.attention_dp
        )
    else:
        sizes = _chunk_sizes(extent, chunks, unit)
    stream = ChunkedWeightStream(
        source,
        full,
        ctx.stream_semaphores,
        axis=axis,
        sizes=sizes,
        depth=depth,
        base=base,
        order=order,
        pool=None if offset is None else ctx.projection_expert_pool_vmem,
        offset=offset,
    )
    stream.shards = plan.attention_dp
    return stream


def _launch_project_attention_dp(
    ctx: _StackRuntimeContext,
    weight_view: PallasRefProtocol,
    stream: ChunkedWeightStream | None,
) -> None:
    """Project every DP shard's gated features into this rank's owned rows."""

    @_scoped("t9_dp_project")
    def project_attention_dp(
        view: PallasRefProtocol,
        weight_stream_arg: ChunkedWeightStream | None,
    ) -> None:
        """Execute scoped DP attention output projection and reduction."""
        comm_ws = ctx.communication_workspace
        dp_attended_vmem = _require_ref(ctx.dp_attended_vmem, "dp_attended_vmem")
        owned = _bf16_rows_view(
            comm_ws.attention_output_vmem,
            ctx.plan.dp_held_rows,
            ctx.plan.hidden_size,
        )
        pl.run_scoped(
            lambda comm: _project_attention_dp(
                dp_attended_vmem,
                view,
                owned,
                comm,
                comm_ws.shared_local_reduction_vmem,
                comm_ws.shared_host_reduction_vmem,
                comm_ws.attention_features_vmem,
                comm_ws.send_semaphores,
                comm_ws.receive_semaphores,
                comm_ws.reduction_phase,
                dot,
                weight_stream=weight_stream_arg,
                staging=comm_ws.projection_staging_vmem,
                owners=ctx.plan.sp_owners,
            ),
            tpu.VMEM((ctx.plan.group_rows, _WIDE_REDUCTION_WIDTH), jnp.float32),
        )

    project_attention_dp(weight_view, stream)


def _query_down_weight_stream(
    ctx: _StackRuntimeContext,
    state_index: int | jax.Array,
    base: int = 0,
) -> ChunkedWeightStream | None:
    """Return MLA's q-down column weight stream at ``base``, if configured."""
    return _weight_stream(
        ctx,
        "m_qa",
        ctx.layer_weights["m_qa"].at[state_index],
        _attention_views(ctx, base)[1][0],
        offset=base if POOLED_WEIGHTS else None,
    )


def _kda_streams(
    ctx: _StackRuntimeContext,
    state_index: int | jax.Array,
    base: int = 0,
) -> dict[str, ChunkedWeightStream]:
    """Build active KDA input/output weight streams for ``state_index``."""
    plan = ctx.plan
    kda_views = _attention_views(ctx, base)[0]
    source = ctx.layer_weights["k_projection"].at[state_index]
    if plan.kda_shard_swap:
        # One copy of this rank's own group's held shard.
        width = source.shape[1] // plan.attention_dp
        group = jax.lax.axis_index(AXIS_TP) // (TP_SIZE // plan.attention_dp)
        input_stream: ChunkedWeightStream | None = ChunkedWeightStream(
            source.at[:, pl.ds(pl.multiple_of(group * width, VREG_LANE_COUNT), width)],
            kda_views[1],
            ctx.stream_semaphores,
            axis="columns",
            sizes=(width,),
            depth=1,
        )
    else:
        input_stream = _weight_stream(
            ctx,
            "k_projection",
            source,
            kda_views[1],
            offset=base + _weight_storage_bytes(plan, "k_gate"),
        )
    output_offset = (
        base
        + _weight_storage_bytes(plan, "k_gate")
        + _weight_storage_bytes(plan, "k_projection")
        + _weight_storage_bytes(plan, "k_fb")
    )
    streams = {
        "input": input_stream,
        "output": _weight_stream(
            ctx,
            "k_o",
            ctx.layer_weights["k_o"].at[state_index],
            kda_views[3],
            offset=output_offset,
        ),
    }
    return {key: stream for key, stream in streams.items() if stream is not None}


def _prefetch_attention(
    ctx: _StackRuntimeContext,
    index: int | jax.Array,
    base: int = 0,
) -> None:
    """Start async DMAs for layer ``index``'s KDA or MLA input projections."""
    kda_projection_vmem, mla_projection_vmem = _attention_views(ctx, base)

    def prefetch_kda() -> None:
        """Issue KDA input projection DMAs or stream primes."""
        state_index = index - index // MLA_LAYER_STRIDE
        # Semaphore slots follow execution dependency order; the scratch tuple
        # follows schema order, so permute both together.
        streams = _kda_streams(ctx, state_index, base)
        streamed = {
            name: streams[key]
            for key, name in (("input", "k_projection"), ("output", "k_o"))
            if key in streams
        }
        for slot, projection_index in enumerate(_KDA_PREFETCH_ORDER):
            if KDA[projection_index] in streamed:
                streamed[KDA[projection_index]].start()
                continue
            tpu.make_async_copy(
                ctx.layer_weights[KDA[projection_index]].at[state_index],
                kda_projection_vmem[projection_index],
                ctx.pool_semaphores.at[slot],
            ).start()

    def prefetch_mla() -> None:
        """Issue MLA input projection DMAs or query-down stream prime."""
        state_index = index // MLA_LAYER_STRIDE
        query_down_stream = _query_down_weight_stream(ctx, state_index, base)
        for slot, (name, destination) in enumerate(
            zip(MLA[:_MLA_INPUT_PROJECTION_COUNT], mla_projection_vmem, strict=True)
        ):
            if name == "m_qa" and query_down_stream is not None:
                query_down_stream.start()
                continue
            tpu.make_async_copy(
                ctx.layer_weights[name].at[state_index],
                destination,
                ctx.pool_semaphores.at[slot],
            ).start()

    jax.lax.cond(
        _is_mla_layer(index),
        prefetch_mla,
        prefetch_kda,
    )


def _prefetch_auxiliary(
    ctx: _StackRuntimeContext,
    index: int | jax.Array,
) -> None:
    """Prefetch KDA auxiliary parameters and persistent states when applicable."""

    @pl.when(_is_kda_layer(index))
    def prefetch_kda_auxiliary() -> None:
        """Start KDA filter/decay/norm and optional persistent state copies."""
        state_index = index - index // MLA_LAYER_STRIDE
        sources: tuple[PallasRefProtocol, ...] = tuple(
            ctx.layer_weights[name].at[state_index]
            for name in KDA[_KDA_AUXILIARY_REF_COUNT:]
        )
        destinations = list(ctx.auxiliary_vmem[: len(sources)])
        if ctx.plan.persist_kda_state:
            sources += (
                _kda_state_source(ctx, ctx.convolution_ref, state_index),
                _kda_state_source(ctx, ctx.recurrent_ref, state_index),
            )
            destinations += [
                _kda_state_destination(ctx, ctx.auxiliary_vmem[4]),
                _kda_state_destination(ctx, ctx.auxiliary_vmem[5]),
            ]
        for slot, (source, destination) in enumerate(
            zip(sources, destinations, strict=True)
        ):
            tpu.make_async_copy(
                source,
                destination,
                ctx.auxiliary_semaphores.at[slot],
            ).start()


def _prefetch_metadata(
    ctx: _StackRuntimeContext,
    index: int | jax.Array,
    *,
    mixture_metadata: bool = True,
) -> None:
    """Prefetch MLA norm vectors and MoE router-bias/latent-norm metadata."""
    buffer_slot = index % _DOUBLE_BUFFER_SLOTS

    @pl.when(_is_mla_layer(index))
    def prefetch_mla_norms() -> None:
        """Start MLA query and key-value norm vector DMAs."""
        for slot, name in enumerate(_MLA_NORM_NAMES):
            tpu.make_async_copy(
                ctx.layer_weights[name].at[index // MLA_LAYER_STRIDE],
                ctx.metadata_vmem[slot].at[buffer_slot],
                ctx.metadata_semaphores.at[buffer_slot, slot],
            ).start()

    if not mixture_metadata:
        return  # layer 0 has no mixture of experts (dense MLP instead)
    for slot, name in enumerate(_MOE_METADATA_NAMES, _MLA_NORM_COUNT):
        tpu.make_async_copy(
            ctx.layer_weights[name].at[index - 1],
            ctx.metadata_vmem[slot].at[buffer_slot],
            ctx.metadata_semaphores.at[buffer_slot, slot],
        ).start()


def _prefetch_router(
    ctx: _StackRuntimeContext,
    index: int | jax.Array,
    *,
    initialize_padding: bool = False,
) -> None:
    """Start router weight DMA when not managed by the weight arena."""
    if ctx.plan.pool_managed:
        return  # queued in the arena
    _router_copy(
        ctx.layer_weights["router"].at[index - 1],
        ctx.router_vmem,
        ctx.router_semaphores,
        wait=False,
        initialize_padding=initialize_padding,
    )


def _capture_aux_hidden(
    layer_ctx: _LayerExecutionContext,
    source: PallasRefProtocol | None = None,
    *,
    gather_owned_rows: bool = False,
) -> None:
    """Copy residual stream to HBM for any configured ``aux_hidden_layers``."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    capture_source = ctx.prefix_vmem if source is None else source
    if gather_owned_rows and plan.attention_dp_ownership:
        # DP ranks hold only their own prefix rows.
        capture_source = ctx.normalized_vmem
    for aux_slot, aux_layer in enumerate(plan.aux_hidden_layers):
        # Layer id L names the residual stream entering layer L, which is the
        # prefix after layer L - 1 completes (or, with aux_capture="attention",
        # after its attention). The rows go to HBM so the capture costs no
        # scratch memory.
        @pl.when(layer_ctx.layer_index == aux_layer - 1)
        def capture(aux_slot: int = aux_slot) -> None:
            """Gather owned rows if needed and copy auxiliary hidden state."""
            if gather_owned_rows:
                # Sequence-parallel layers keep only this pair's prefix rows;
                # complete them for the capture.
                if plan.attention_dp_ownership:
                    _gather_dp_rows(ctx, ctx.normalized_vmem)
                else:
                    comm = ctx.communication_workspace
                    _gather_local_pair_rows(
                        ctx.owned_rows_vmem,
                        ctx.prefix_vmem,
                        comm.send_semaphores,
                        comm.receive_semaphores,
                        comm.shared_local_reduction_vmem,
                        owners=plan.sp_owners,
                    )
            aux_copy = tpu.make_async_copy(
                capture_source,
                ctx.aux_output_ref.at[aux_slot],
                ctx.aux_semaphore.at[0],
            )
            aux_copy.start()
            aux_copy.wait()


def _prepare_pre_attention_residual(layer_ctx: _LayerExecutionContext) -> None:
    """Compute pre-attention residual mixture and start new residual blocks."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    layer_index = layer_ctx.layer_index
    if plan.attention_dp_ownership:
        _mix_dp_rows(
            ctx,
            ctx.attention_residual_vmem,
            (layer_index - 1) // _RESIDUAL_BLOCK_STRIDE + 1,
            _DpMixOptions(layer_index=layer_index, start_block=True),
        )
        return
    if plan.sequence_parallel_pre_attention:
        rank = jax.lax.axis_index(AXIS_TP)
        pair = (rank % mla_dcp.HOST_LANES) // (mla_dcp.HOST_LANES // plan.sp_owners)
        for owner_pair in _owner_indices(plan, pair):

            @_owner_when(plan, pair, owner_pair)
            def prepare_owned_attention_rows(
                owner_pair: int | jax.Array = owner_pair,
            ) -> None:
                """Mix owned attention rows and snapshot block-start rows."""
                row_offset = owner_pair * plan.owned_rows
                row_slice = pl.ds(_owner_start(plan, row_offset), plan.owned_rows)

                def mix_row(local_row: int) -> jax.Array:
                    """Mix snapshots and prefix for one owned row."""
                    global_row = row_offset + local_row
                    return _residual_mixture_row(
                        _block_rows(ctx, _owner_block_row(plan, global_row, local_row)),
                        _owner_prefix_row(ctx, global_row, row_slice, local_row),
                        ctx.attention_residual_vmem,
                        (layer_index - 1) // _RESIDUAL_BLOCK_STRIDE + 1,
                        epsilon=plan.eps,
                        relaxed_reduction_order=(
                            plan.residual_reduction_order == "relaxed"
                        ),
                    )

                _store_mixed_rows(ctx, mix_row, ctx.owned_rows_vmem)

                @pl.when(
                    (layer_index % _RESIDUAL_BLOCK_STRIDE == 0)
                    if plan.blocks_block_major
                    else False
                )
                def start_owned_residual_block_major() -> None:
                    """Snapshot owned rows into block-major BF16 storage."""
                    for local_row in range(plan.owned_rows):
                        global_row = row_offset + local_row
                        row = (
                            _owner_block_row(plan, global_row, local_row)
                            % plan.block_rows_held
                        )
                        half = ctx.blocks_vmem.at[
                            row // 2,
                            pl.ds((row % 2) * SUBLANE_ROWS, SUBLANE_ROWS),
                        ]
                        block_mask = (
                            jax.lax.broadcasted_iota(jnp.int32, half.shape, 0)
                            == layer_index // _RESIDUAL_BLOCK_STRIDE
                        )
                        half[...] = jnp.where(
                            block_mask,
                            _owner_prefix_row(ctx, global_row, row_slice, local_row),
                            half[...],
                        )

                @pl.when(
                    False
                    if plan.blocks_block_major
                    else (layer_index % _RESIDUAL_BLOCK_STRIDE == 0)
                )
                def start_owned_residual_block() -> None:
                    """Snapshot owned rows into FP32 residual block storage."""
                    block_mask = (
                        jax.lax.broadcasted_iota(
                            jnp.int32,
                            ctx.blocks_vmem[row_slice].shape,
                            1,
                        )
                        == layer_index // _RESIDUAL_BLOCK_STRIDE
                    )
                    ctx.blocks_vmem[row_slice] = jnp.where(
                        block_mask,
                        ctx.prefix_vmem[row_slice][:, None, :].astype(jnp.float32),
                        ctx.blocks_vmem[row_slice],
                    )

        return
    _residual_mix(
        ctx,
        ctx.normalized_vmem,
        ctx.attention_residual_vmem,
        (layer_index - 1) // _RESIDUAL_BLOCK_STRIDE + 1,
    )

    @pl.when(layer_index % _RESIDUAL_BLOCK_STRIDE == 0)
    def start_residual_block() -> None:
        """Snapshot whole-batch prefix rows into residual block storage."""
        block_mask = (
            jax.lax.broadcasted_iota(jnp.int32, ctx.blocks_vmem.shape, 1)
            == layer_index // _RESIDUAL_BLOCK_STRIDE
        )
        ctx.blocks_vmem[...] = jnp.where(
            block_mask,
            ctx.prefix_vmem[...][:, None, :].astype(jnp.float32),
            ctx.blocks_vmem[...],
        )


def _prepare_pre_attention_norm_and_gather(
    layer_ctx: _LayerExecutionContext,
) -> None:
    """Normalize pre-attention rows and gather them for attention."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    comm = ctx.communication_workspace
    if plan.kda_shard_swap:
        # Every row is the swapped projection's input; attention reads the
        # group's.
        _normalize_rows(ctx, ctx.normalized_vmem, ctx.attention_norm_vmem)
        data_group = jax.lax.axis_index(AXIS_TP) // (TP_SIZE // plan.attention_dp)
        rows = ctx.normalized_vmem[...]
        group_input_vmem = _require_ref(ctx.group_input_vmem, "group_input_vmem")
        group_input_vmem[...] = jnp.where(
            data_group == 0,
            rows[: plan.group_rows],
            rows[plan.group_rows :],
        )
    elif plan.attention_dp_ownership:
        _normalize_dp_rows(ctx, ctx.attention_norm_vmem, split=plan.sp8_residual)
        group_input_vmem = _require_ref(ctx.group_input_vmem, "group_input_vmem")
        with _phase("t0_dp_group_gather"):
            if plan.sp8_residual:
                _gather_sp8_local_rows(
                    _sp8_owned_value(_require_ref(ctx.sp8_rows_vmem, "sp8_rows_vmem")),
                    group_input_vmem,
                    comm.shared_local_reduction_vmem,
                    comm.send_semaphores,
                    comm.receive_semaphores,
                )
            else:
                _gather_local_pair_rows(
                    _require_ref(ctx.dp_owned_vmem, "dp_owned_vmem"),
                    group_input_vmem,
                    comm.send_semaphores,
                    comm.receive_semaphores,
                    comm.shared_local_reduction_vmem,
                    packed=True,
                )
    elif plan.sequence_parallel_pre_attention:
        _normalize_rows(ctx, ctx.owned_rows_vmem, ctx.attention_norm_vmem)
        _gather_local_pair_rows(
            ctx.owned_rows_vmem,
            ctx.normalized_vmem,
            comm.send_semaphores,
            comm.receive_semaphores,
            comm.shared_local_reduction_vmem,
            owners=plan.sp_owners,
        )
    else:
        _normalize_rows(ctx, ctx.normalized_vmem, ctx.attention_norm_vmem)


def _make_kda_dp_weights(
    layer_ctx: _LayerExecutionContext,
    state_index: int | jax.Array,
    shard: int | jax.Array,
) -> KimiDeltaAttentionWeights:
    """Build per-shard ``KimiDeltaAttentionWeights`` for attention DP."""
    ctx = layer_ctx.ctx
    aux = ctx.auxiliary_vmem
    proj = layer_ctx.kda_projection_vmem
    return KimiDeltaAttentionWeights(
        ctx.layer_weights["k_gate"].at[state_index],
        proj[0],
        ctx.layer_weights["k_projection"].at[state_index],
        proj[1],
        ctx.layer_weights["k_fb"].at[state_index],
        proj[2],
        ctx.layer_weights["k_o"].at[state_index],
        proj[3],
        # Per-shard filter, decay and time-step bias; shared norm.
        aux[0].at[shard],
        aux[1].at[shard],
        aux[2].at[shard],
        aux[3],
        ctx.pool_semaphores,
    )


def _make_kda_weights(
    layer_ctx: _LayerExecutionContext,
    state_index: int | jax.Array,
) -> KimiDeltaAttentionWeights:
    """Build standard ``KimiDeltaAttentionWeights`` for non-DP execution."""
    ctx = layer_ctx.ctx
    aux = ctx.auxiliary_vmem
    proj = layer_ctx.kda_projection_vmem
    return KimiDeltaAttentionWeights(
        ctx.layer_weights["k_gate"].at[state_index],
        proj[0],
        ctx.layer_weights["k_projection"].at[state_index],
        proj[1],
        ctx.layer_weights["k_fb"].at[state_index],
        proj[2],
        ctx.layer_weights["k_o"].at[state_index],
        proj[3],
        aux[0],
        aux[1],
        aux[2],
        aux[3],
        ctx.pool_semaphores,
    )


def _execute_kda_dp_unstreamed(
    layer_ctx: _LayerExecutionContext,
    state_index: int | jax.Array,
    buffers: _KdaExecBuffers,
    callbacks: _KdaStateCallbacks,
) -> None:
    """Run unstreamed attention-DP KDA across held shards and project output."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    group_input_vmem = _require_ref(ctx.group_input_vmem, "group_input_vmem")
    dp_attended_vmem = _require_ref(ctx.dp_attended_vmem, "dp_attended_vmem")
    streams = _kda_streams(ctx, state_index, layer_ctx.attention_base)
    swapped = (
        _swap_kda_projections(ctx, streams["input"]) if plan.kda_shard_swap else None
    )
    for order in range(plan.attention_dp):
        shard: int | jax.Array = order
        if plan.kda_shard_swap:
            # The swap runs this rank's own shard first.
            own_shard = jax.lax.axis_index(AXIS_TP) // (TP_SIZE // plan.attention_dp)
            shard = (own_shard + order) % plan.attention_dp
        rows = pl.ds(shard * plan.group_rows, plan.group_rows)
        kda(
            group_input_vmem,
            group_input_vmem,
            buffers.history_vmem.at[rows],
            buffers.state_vmem.at[rows],
            weights=_make_kda_dp_weights(layer_ctx, state_index, shard),
            convolution_history_ref=buffers.history_vmem.at[rows],
            recurrent_state_ref=buffers.state_vmem.at[rows],
            communication_workspace=ctx.communication_workspace,
            epsilon=plan.eps,
            synchronize_devices=False,
            weights_are_prefetched=True,
            wait_for_state=(
                callbacks.wait_for_state
                if plan.overlap_kda_state_reads and order == 0
                else None
            ),
            on_state_updated=(
                callbacks.start_state_writes
                if plan.overlap_kda_state_writes and order == plan.attention_dp - 1
                else None
            ),
            sequence_rows=plan.sequence_rows,
            streams=streams,
            shard_index=shard,
            attended_out=dp_attended_vmem.at[shard],
            gate_features_vmem=ctx.dp_compressed_vmem,
            stage_projections=True,
            projected_rows=None if swapped is None else swapped(order),
            first_call=order == 0,
        )
    _launch_project_attention_dp(
        ctx, layer_ctx.kda_projection_vmem[3], streams.get("output")
    )


def _execute_kda_body(
    layer_ctx: _LayerExecutionContext,
    state_index: int | jax.Array,
    buffers: _KdaExecBuffers,
) -> None:
    """Run unstreamed (persistent or scoped) KDA recurrence and state writes."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    state_writes = (
        tpu.make_async_copy(
            buffers.history_vmem,
            ctx.convolution_output_ref.at[:, state_index],
            buffers.state_semaphores.at[0],
        ),
        tpu.make_async_copy(
            buffers.state_vmem,
            ctx.recurrent_output_ref.at[:, state_index],
            buffers.state_semaphores.at[1],
        ),
    )

    def wait_for_state() -> None:
        """Wait for overlapped KDA input state DMAs."""
        for copy in buffers.state_reads:
            copy.wait()

    def start_state_writes() -> None:
        """Start KDA post-state writes back to HBM."""
        for copy in state_writes:
            copy.start()

    if plan.attention_dp > 1:
        _execute_kda_dp_unstreamed(
            layer_ctx,
            state_index,
            buffers,
            _KdaStateCallbacks(wait_for_state, start_state_writes),
        )
    else:
        kda(
            ctx.normalized_vmem,
            ctx.attention_vmem,
            buffers.history_vmem,
            buffers.state_vmem,
            weights=_make_kda_weights(layer_ctx, state_index),
            convolution_history_ref=buffers.history_vmem,
            recurrent_state_ref=buffers.state_vmem,
            communication_workspace=ctx.communication_workspace,
            epsilon=plan.eps,
            synchronize_devices=False,
            weights_are_prefetched=True,
            token_scatter_output=plan.attention_token_scatter,
            sequence_rows=plan.sequence_rows,
            wait_for_state=(wait_for_state if plan.overlap_kda_state_reads else None),
            on_state_updated=(
                start_state_writes if plan.overlap_kda_state_writes else None
            ),
            streams=_kda_streams(ctx, state_index, layer_ctx.attention_base),
        )
    if not plan.overlap_kda_state_writes:
        start_state_writes()
    if not plan.persist_kda_state:
        for copy in state_writes:
            copy.wait()


def _run_kda_rolled_dp_shards(
    layer_ctx: _LayerExecutionContext,
    state_index: int | jax.Array,
    shared_state_stream: KdaStateStream,
    streams: Mapping[str, ChunkedWeightStream],
    input_stream: ChunkedWeightStream,
) -> None:
    """Run DP4 24/32-row KDA through one traced-shard loop to limit code size."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    group_input_vmem = _require_ref(ctx.group_input_vmem, "group_input_vmem")
    dp_attended_vmem = _require_ref(ctx.dp_attended_vmem, "dp_attended_vmem")
    dp_compressed_vmem = _require_ref(ctx.dp_compressed_vmem, "dp_compressed_vmem")
    chunks_per_shard = plan.group_rows // plan.kda_row_chunk
    per_shard = input_stream.chunks // plan.attention_dp
    chunk_columns = input_stream.sizes[0]

    def run_rolled_shards(projection: PallasRefProtocol) -> None:
        """Run shared gate features then traced per-shard KDA recurrence."""
        copies = _make_kda_dp_weights(
            layer_ctx, state_index, 0
        ).make_projection_copies()
        copies.gate.wait()
        copies.feature_gate.wait()
        dp_compressed_vmem[:, :VREG_LANE_COUNT] = dot(
            group_input_vmem[...], layer_ctx.kda_projection_vmem[0][...]
        ).astype(jnp.bfloat16)
        guard = ctx.communication_workspace.reduction_phase[0] >= 0

        @pl.loop(0, plan.attention_dp)
        def shard_loop(shard: jax.Array) -> None:
            """Project and execute KDA for one traced head shard."""

            @pl.loop(0, per_shard)
            def project(position: jax.Array) -> None:
                """Run one chunked input projection for ``shard``."""
                _traced_dot_position(
                    input_stream,
                    group_input_vmem,
                    shard * per_shard + position,
                    guard,
                    projection.at[position],
                )

            rows = pl.ds(
                pl.multiple_of(shard * plan.group_rows, SUBLANE_ROWS),
                plan.group_rows,
            )
            kda(
                group_input_vmem,
                group_input_vmem,
                ctx.convolution_output_ref.at[:, state_index].at[rows],
                ctx.recurrent_output_ref.at[:, state_index].at[rows],
                weights=_make_kda_dp_weights(layer_ctx, state_index, shard),
                convolution_history_ref=ctx.convolution_ref.at[:, state_index],
                recurrent_state_ref=ctx.recurrent_ref.at[:, state_index],
                communication_workspace=ctx.communication_workspace,
                epsilon=plan.eps,
                synchronize_devices=False,
                weights_are_prefetched=True,
                sequence_rows=plan.sequence_rows,
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
                    [projection[chunk_idx] for chunk_idx in range(per_shard)],
                    axis=1,
                ),
                first_call=False,
            )

    pl.run_scoped(
        run_rolled_shards,
        tpu.VMEM((per_shard, plan.group_rows, chunk_columns), jnp.bfloat16),
    )


def _run_kda_unrolled_dp_shards(
    layer_ctx: _LayerExecutionContext,
    state_index: int | jax.Array,
    shared_state_stream: KdaStateStream,
    streams: Mapping[str, ChunkedWeightStream],
) -> None:
    """Run unrolled per-shard streamed KDA calls for attention DP."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    group_input_vmem = _require_ref(ctx.group_input_vmem, "group_input_vmem")
    dp_attended_vmem = _require_ref(ctx.dp_attended_vmem, "dp_attended_vmem")
    chunks_per_shard = plan.group_rows // plan.kda_row_chunk
    for shard in range(plan.attention_dp):
        rows = pl.ds(shard * plan.group_rows, plan.group_rows)
        history_rows = ctx.convolution_ref.at[:, state_index]
        state_rows = ctx.recurrent_ref.at[:, state_index]
        if not plan.slot_reads:
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
            ctx.convolution_output_ref.at[:, state_index].at[rows],
            ctx.recurrent_output_ref.at[:, state_index].at[rows],
            weights=_make_kda_dp_weights(layer_ctx, state_index, shard),
            convolution_history_ref=history_rows,
            recurrent_state_ref=state_rows,
            communication_workspace=ctx.communication_workspace,
            epsilon=plan.eps,
            synchronize_devices=False,
            weights_are_prefetched=True,
            sequence_rows=plan.sequence_rows,
            streams=streams,
            shard_index=shard,
            attended_out=dp_attended_vmem.at[shard],
            gate_features_vmem=ctx.dp_compressed_vmem,
            stage_projections=True,
            state_stream=state_stream,
        )


def _run_kda_persistent(
    layer_ctx: _LayerExecutionContext,
    state_index: int | jax.Array,
) -> None:
    """Wait for persistent KDA state prefetches and run KDA."""
    ctx = layer_ctx.ctx
    state_sources = (
        _kda_state_source(ctx, ctx.convolution_ref, state_index),
        _kda_state_source(ctx, ctx.recurrent_ref, state_index),
    )
    state_pending = tuple(
        tpu.make_async_copy(
            source,
            destination,
            ctx.auxiliary_semaphores.at[index + _KDA_AUXILIARY_REF_COUNT],
        )
        for index, (source, destination) in enumerate(
            zip(
                state_sources,
                (
                    _kda_state_destination(ctx, ctx.auxiliary_vmem[4]),
                    _kda_state_destination(ctx, ctx.auxiliary_vmem[5]),
                ),
                strict=True,
            )
        )
    )
    for copy in state_pending:
        copy.wait()
    _execute_kda_body(
        layer_ctx,
        state_index,
        _KdaExecBuffers(
            ctx.auxiliary_vmem[4],
            ctx.auxiliary_vmem[5],
            ctx.auxiliary_semaphores,
        ),
    )


def _run_kda_streamed(
    layer_ctx: _LayerExecutionContext,
    state_index: int | jax.Array,
) -> None:
    """Run KDA with chunk-streamed state buffers (DP or non-DP)."""
    ctx = layer_ctx.ctx
    plan = ctx.plan

    def execute_streamed_dp(
        history_buffers: PallasRefProtocol,
        state_buffers: PallasRefProtocol,
        read_semaphores: PallasRefProtocol,
        write_semaphores: PallasRefProtocol,
    ) -> None:
        """Stream KDA states across attention DP shards."""
        streams = _kda_streams(ctx, state_index, layer_ctx.attention_base)
        chunks_per_shard = plan.group_rows // plan.kda_row_chunk
        # State rows are head-shard-major, then the group's sequences
        # (shard k of group row p at k * G + p). One flattened stream keeps
        # its two-chunk lookahead across those shards without allocating
        # another state buffer.
        shared_state_stream = KdaStateStream(
            ctx.convolution_ref.at[:, state_index],
            ctx.recurrent_ref.at[:, state_index],
            ctx.convolution_output_ref.at[:, state_index],
            ctx.recurrent_output_ref.at[:, state_index],
            history_buffers,
            state_buffers,
            read_semaphores,
            write_semaphores,
            chunk=plan.kda_row_chunk,
            chunks=plan.attention_dp * chunks_per_shard,
            slots=ctx.slots_ref if plan.slot_reads else None,
            sequence_rows=plan.sequence_rows,
        )
        shared_state_stream.prime()
        input_stream = streams.get("input")
        # DP4 runs its 24- and 32-row groups through one traced-shard body:
        # four unrolled copies stall on instruction overlays.
        if (
            plan.attention_dp == DP4_GROUPS
            and plan.group_rows in _DP4_ROLLED_GROUP_ROWS
            and input_stream is not None
            and input_stream.slots is not None
            and plan.slot_reads
            and "kda" not in DEBUG_SKIP
        ):
            _run_kda_rolled_dp_shards(
                layer_ctx,
                state_index,
                shared_state_stream,
                streams,
                input_stream,
            )
        else:
            _run_kda_unrolled_dp_shards(
                layer_ctx, state_index, shared_state_stream, streams
            )
        # All KDA consumers have finished. The final three writes are the
        # only outstanding DMAs: every read was acquired and earlier writes
        # drained at reuse.
        if plan.group_rows > _SMALL_DP_GROUP_ROWS and "kda" not in DEBUG_SKIP:
            shared_state_stream.finish_rolled()
        else:
            shared_state_stream.finish()
        if "project" not in DEBUG_SKIP:
            _launch_project_attention_dp(
                ctx,
                layer_ctx.kda_projection_vmem[3],
                streams.get("output"),
            )

    def execute_streamed(
        history_buffers: PallasRefProtocol,
        state_buffers: PallasRefProtocol,
        read_semaphores: PallasRefProtocol,
        write_semaphores: PallasRefProtocol,
    ) -> None:
        """Stream KDA states for non-DP batches above 16."""
        state_stream = KdaStateStream(
            ctx.convolution_ref.at[:, state_index],
            ctx.recurrent_ref.at[:, state_index],
            ctx.convolution_output_ref.at[:, state_index],
            ctx.recurrent_output_ref.at[:, state_index],
            history_buffers,
            state_buffers,
            read_semaphores,
            write_semaphores,
            chunk=plan.kda_row_chunk,
            chunks=plan.batch_size // plan.kda_row_chunk,
            slots=ctx.slots_ref if plan.slot_reads else None,
            sequence_rows=plan.sequence_rows,
        )
        state_stream.prime()
        kda(
            ctx.normalized_vmem,
            ctx.attention_vmem,
            ctx.convolution_output_ref.at[:, state_index],
            ctx.recurrent_output_ref.at[:, state_index],
            weights=_make_kda_weights(layer_ctx, state_index),
            convolution_history_ref=ctx.convolution_ref.at[:, state_index],
            recurrent_state_ref=ctx.recurrent_ref.at[:, state_index],
            communication_workspace=ctx.communication_workspace,
            epsilon=plan.eps,
            synchronize_devices=False,
            weights_are_prefetched=True,
            token_scatter_output=plan.attention_token_scatter,
            sequence_rows=plan.sequence_rows,
            streams=_kda_streams(ctx, state_index, layer_ctx.attention_base),
            state_stream=state_stream,
        )

    state_buffer_count = (
        _UNROLLED_B32_STATE_BUFFERS
        if plan.batch_size == _UNROLLED_FOUR_BUFFER_BATCH
        and plan.attention_dp == 1
        and not KDA_STAGED
        else _ROLLING_STATE_BUFFERS
    )
    pl.run_scoped(
        execute_streamed_dp if plan.attention_dp > 1 else execute_streamed,
        tpu.VMEM(
            (state_buffer_count, plan.kda_row_chunk, *plan.state_conv_shape[2:]),
            plan.state_conv_dtype,
        ),
        tpu.VMEM(
            (state_buffer_count, plan.kda_row_chunk, *plan.state_rec_shape[2:]),
            plan.state_rec_dtype,
        ),
        tpu.SemaphoreType.DMA((state_buffer_count, 2)),
        tpu.SemaphoreType.DMA((state_buffer_count, 2)),
    )


def _run_delta_attention_body(layer_ctx: _LayerExecutionContext) -> None:
    """Execute one KDA layer across persistent, streamed, or scoped states."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    state_index = layer_ctx.layer_index - layer_ctx.layer_index // MLA_LAYER_STRIDE
    auxiliary_sources = tuple(
        ctx.layer_weights[name].at[state_index]
        for name in KDA[_KDA_AUXILIARY_REF_COUNT:]
    )
    pending = tuple(
        tpu.make_async_copy(
            source,
            destination,
            ctx.auxiliary_semaphores.at[index],
        )
        for index, (source, destination) in enumerate(
            zip(
                auxiliary_sources,
                ctx.auxiliary_vmem[:_KDA_AUXILIARY_REF_COUNT],
                strict=True,
            )
        )
    )
    for copy in pending:
        copy.wait()

    if plan.persist_kda_state:
        _run_kda_persistent(layer_ctx, state_index)
        return

    if plan.stream_kda_state:
        _run_kda_streamed(layer_ctx, state_index)
        return

    def execute_scoped(
        history_vmem: PallasRefProtocol,
        state_vmem: PallasRefProtocol,
        state_semaphores: PallasRefProtocol,
    ) -> None:
        """Load whole-batch KDA states into scoped buffers and run KDA."""
        state_copies = (
            *_kda_state_reads(
                ctx,
                ctx.convolution_ref,
                history_vmem,
                state_semaphores.at[0],
                state_index,
            ),
            *_kda_state_reads(
                ctx,
                ctx.recurrent_ref,
                state_vmem,
                state_semaphores.at[1],
                state_index,
            ),
        )
        for copy in state_copies:
            copy.start()
        if not plan.overlap_kda_state_reads:
            for copy in state_copies:
                copy.wait()
        _execute_kda_body(
            layer_ctx,
            state_index,
            _KdaExecBuffers(history_vmem, state_vmem, state_semaphores, state_copies),
        )

    pl.run_scoped(
        execute_scoped,
        tpu.VMEM((plan.batch_size, *plan.state_conv_shape[2:]), plan.state_conv_dtype),
        tpu.VMEM((plan.batch_size, *plan.state_rec_shape[2:]), plan.state_rec_dtype),
        tpu.SemaphoreType.DMA((2,)),
    )


def _build_mla_streams(
    mla_ctx: _MlaLayerContext,
    projection_vmem: tuple[PallasRefProtocol, ...],
) -> dict[str, ChunkedWeightStream]:
    """Assemble configured weight streams for an MLA layer."""
    layer_ctx = mla_ctx.layer_ctx
    ctx = layer_ctx.ctx
    state_index = mla_ctx.state_index
    dp_output = mla_ctx.dp_output
    mla_streams: dict[str, ChunkedWeightStream] = {}
    for key, name, view in zip(
        ("query_up", "key_value_up", "output_gate", "output"),
        MLA[2:6],
        projection_vmem,
        strict=True,
    ):
        if dp_output is not None and name == "m_o":
            stream = dp_output[1]
        else:
            offset = (
                ctx.mem_plan.pool_layout[name]
                + (layer_ctx.attention_base if name == "m_o" else 0)
                if POOLED_WEIGHTS
                else None
            )
            stream = _weight_stream(
                ctx,
                name,
                ctx.layer_weights[name].at[state_index],
                view,
                offset=offset,
            )
        if stream is not None:
            mla_streams[key] = stream
    query_down_stream = _query_down_weight_stream(
        ctx, state_index, layer_ctx.attention_base
    )
    if query_down_stream is not None:
        mla_streams["query_down"] = query_down_stream
    return mla_streams


def _execute_mla_compute(
    mla_ctx: _MlaLayerContext,
    buffers: _MlaScopedBuffers,
) -> None:
    """Wait for MLA norms, dispatch ``mla`` or ``mla_dcp_shards``, and commit."""
    layer_ctx = mla_ctx.layer_ctx
    ctx = layer_ctx.ctx
    plan = ctx.plan
    state_index = mla_ctx.state_index
    metadata_copies = (
        tpu.make_async_copy(
            ctx.layer_weights["m_qnorm"].at[state_index],
            mla_ctx.query_norm_vmem,
            mla_ctx.layer_metadata_semaphores.at[0],
        ),
        tpu.make_async_copy(
            ctx.layer_weights["m_knorm"].at[state_index],
            mla_ctx.key_value_norm_vmem,
            mla_ctx.layer_metadata_semaphores.at[1],
        ),
    )
    for copy in metadata_copies:
        copy.wait()
    workspace = buffers.workspace
    if POOLED_WEIGHTS:
        projection_vmem = tuple(
            _pool_view(
                ctx,
                name,
                base=layer_ctx.attention_base if name == "m_o" else 0,
            )
            for name in MLA[2:6]
        )
    else:
        projection_vmem = workspace[:_MLA_EXTRA_PROJECTION_COUNT]
        workspace = workspace[_MLA_EXTRA_PROJECTION_COUNT:]
    mla_streams = _build_mla_streams(mla_ctx, projection_vmem)
    weights = MultiHeadLatentAttentionWeights(
        ctx.layer_weights["m_qa"].at[state_index],
        layer_ctx.mla_projection_vmem[0],
        ctx.layer_weights["m_ka"].at[state_index],
        layer_ctx.mla_projection_vmem[1],
        ctx.layer_weights["m_qb"].at[state_index],
        projection_vmem[0],
        ctx.layer_weights["m_kb"].at[state_index],
        projection_vmem[1],
        ctx.layer_weights["m_gate"].at[state_index],
        projection_vmem[2],
        ctx.layer_weights["m_o"].at[state_index],
        projection_vmem[3],
        mla_ctx.query_norm_vmem,
        mla_ctx.key_value_norm_vmem,
        ctx.pool_semaphores,
    )
    if plan.attention_dp > 1:
        data_group = jax.lax.axis_index(AXIS_TP) // (TP_SIZE // plan.attention_dp)

        def group_table_row(row: int | jax.Array) -> int | jax.Array:
            """Map group row ``p`` back to its global batch row."""
            # Group row p is the batch row its positions came from.
            return (
                row // plan.group_rows_per_pair * plan.tokens_per_pair
                + data_group * plan.group_rows_per_pair
                + row % plan.group_rows_per_pair
            )

        group_handoff = pl.ds(0, plan.group_rows)
        mla_dcp_shards(
            _require_ref(ctx.group_positions_smem, "group_positions_smem"),
            _require_ref(ctx.group_input_vmem, "group_input_vmem"),
            buffers.new_key_vmem.at[group_handoff],
            buffers.new_value_vmem.at[group_handoff],
            _require_ref(ctx.dp_attended_vmem, "dp_attended_vmem"),
            weights=weights,
            streams=mla_streams,
            communication_workspace=ctx.communication_workspace,
            dcp=(
                mla_dcp.DcpCache(
                    ctx.pool_ref,
                    ctx.table_ref,
                    plan.paged_pages,
                    state_index,
                    owner=(
                        None
                        if plan.dcp_owner_table is None
                        else mla_dcp.owner_index(plan.dcp_owner_table) % mla_dcp.GROUP
                    ),
                    table_row=group_table_row,
                ),
                mla_dcp.DcpWorkspace(*ctx.dcp_persistent, *buffers.dcp_scoped),
            ),
            epsilon=plan.eps,
            sequence_rows=plan.sequence_rows,
        )
    else:
        mla(
            ctx.position_ref,
            ctx.normalized_vmem,
            ctx.attention_vmem,
            buffers.new_key_vmem,
            buffers.new_value_vmem,
            weights=weights,
            workspace=MultiHeadLatentAttentionWorkspace(*workspace),
            # The pooled output projection overwrites the q-down projection, so
            # it may only load once that is consumed.
            output_projection_after_query_down=POOLED_WEIGHTS,
            streams=mla_streams,
            communication_workspace=ctx.communication_workspace,
            epsilon=plan.eps,
            synchronize_devices=False,
            weights_are_prefetched=True,
            token_scatter_output=plan.attention_token_scatter,
            sequence_rows=plan.sequence_rows,
            dcp=(
                mla_dcp.DcpCache(
                    ctx.pool_ref,
                    ctx.table_ref,
                    plan.paged_pages,
                    state_index,
                    owner=(
                        None
                        if plan.dcp_owner_table is None
                        else mla_dcp.owner_index(plan.dcp_owner_table)
                    ),
                ),
                mla_dcp.DcpWorkspace(*ctx.dcp_persistent, *buffers.dcp_scoped),
            ),
        )
    commit_scope = phase_start("q8_mla_commit")

    # The pool stays read-only: hand this layer's new rows out.
    def hand_out_rows(semaphores: PallasRefProtocol) -> None:
        """DMA newly computed MLA key/value latents to HBM output buffers."""
        copies = (
            tpu.make_async_copy(
                buffers.new_key_vmem,
                ctx.key_output_ref.at[:, state_index],
                semaphores.at[0],
            ),
            tpu.make_async_copy(
                buffers.new_value_vmem,
                ctx.value_output_ref.at[:, state_index],
                semaphores.at[1],
            ),
        )
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    pl.run_scoped(hand_out_rows, tpu.SemaphoreType.DMA((2,)))
    phase_end(commit_scope)


def _run_latent_attention_body(layer_ctx: _LayerExecutionContext) -> None:
    """Execute one MLA layer with scoped handoff and optional DP projection."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    state_index = layer_ctx.layer_index // MLA_LAYER_STRIDE
    buffer_slot = layer_ctx.layer_index % _DOUBLE_BUFFER_SLOTS
    # Attention DP projects the output once the MLA workspace is released, so
    # its reduction buffer does not add to the workspace's peak (pooled
    # weights: the projection persists).
    dp_output: tuple[PallasRefProtocol, ChunkedWeightStream | None] | None = None
    if plan.attention_dp > 1:
        output_view = _pool_view(ctx, "m_o", base=layer_ctx.attention_base)
        dp_output = (
            output_view,
            _weight_stream(
                ctx,
                "m_o",
                ctx.layer_weights["m_o"].at[state_index],
                output_view,
                offset=ctx.mem_plan.pool_layout["m_o"] + layer_ctx.attention_base,
            ),
        )
    mla_ctx = _MlaLayerContext(
        layer_ctx=layer_ctx,
        state_index=state_index,
        query_norm_vmem=ctx.metadata_vmem[0].at[buffer_slot],
        key_value_norm_vmem=ctx.metadata_vmem[1].at[buffer_slot],
        layer_metadata_semaphores=ctx.metadata_semaphores.at[buffer_slot],
        dp_output=dp_output,
    )

    def compute(
        new_key_vmem: PallasRefProtocol,
        new_value_vmem: PallasRefProtocol,
        *workspace: PallasRefProtocol,
    ) -> None:
        """Unpack scoped DCP buffers and run MLA compute + handoff."""
        count = len(mla_dcp.SCOPED_NAMES)
        _execute_mla_compute(
            mla_ctx,
            _MlaScopedBuffers(
                new_key_vmem=new_key_vmem,
                new_value_vmem=new_value_vmem,
                workspace=workspace[:-count],
                dcp_scoped=workspace[-count:],
            ),
        )

    pl.run_scoped(
        compute,
        # Pad the transient handoff so each runtime tile advances by a multiple
        # of eight vector rows.
        tpu.VMEM(
            (
                plan.batch_size,
                plan.cache_handoff_head_count,
                _MLA_KEY_HANDOFF_LANES,
            ),
            jnp.bfloat16,
        ),
        tpu.VMEM(
            (
                plan.batch_size,
                plan.cache_handoff_head_count,
                HEAD_DIM,
            ),
            jnp.bfloat16,
        ),
        *ctx.mem_plan.mla_scoped_scratch,
    )
    if dp_output is not None:
        _launch_project_attention_dp(ctx, dp_output[0], dp_output[1])


def _prepare_post_attention_scatter(
    layer_ctx: _LayerExecutionContext,
    post_scope: PhaseScopeProtocol,
) -> None:
    """Apply token-scattered post-attention residual, norm, and MoE gather."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    layer_index = layer_ctx.layer_index
    comm = ctx.communication_workspace
    rank = jax.lax.axis_index(AXIS_TP)
    pair = (rank % mla_dcp.HOST_LANES) // (mla_dcp.HOST_LANES // plan.sp_owners)
    owned_attention_vmem = _bf16_rows_view(
        comm.attention_output_vmem,
        plan.tokens_per_pair,
        plan.hidden_size,
    )
    owned_normalized_vmem = ctx.owned_rows_vmem
    # With one owner per rank each rank keeps its half of the pair's attention
    # rows.
    pair_split = plan.sp_owners // _PAIR_SP_OWNERS
    # Each row owner holds ``owned_rows`` rows. Static branches keep source
    # slices aligned while calculations use a purpose-sized workspace.
    prefix_values = ctx.prefix_vmem[...] if plan.owned_rows == 1 else None
    for owner_pair in _owner_indices(plan, pair):

        @_owner_when(plan, pair, owner_pair)
        def prepare_owned_rows(
            owner_pair: int | jax.Array = owner_pair,
        ) -> None:
            """Add attention output to owned prefix rows and mix for MoE."""
            attention_rows = (
                owned_attention_vmem
                if pair_split == 1
                else owned_attention_vmem.at[
                    pl.ds(
                        _owner_start(plan, (owner_pair % pair_split) * plan.owned_rows),
                        plan.owned_rows,
                    )
                ]
            )
            row_offset = owner_pair * plan.owned_rows
            row_slice: object | None = None
            if plan.owned_rows == 1:
                cached_prefix = _require_array(prefix_values, "prefix_values")
                prefix_rows = cached_prefix[owner_pair : owner_pair + 1]
            else:
                row_slice = pl.ds(_owner_start(plan, row_offset), plan.owned_rows)
                prefix_rows = ctx.prefix_vmem[row_slice]
            updated_prefix = jnp.where(
                layer_index % _RESIDUAL_BLOCK_STRIDE == 0,
                attention_rows[...],
                prefix_rows + attention_rows[...],
            ).astype(jnp.bfloat16)
            if row_slice is None:
                cached_prefix = _require_array(prefix_values, "prefix_values")
                row_indices = jax.lax.broadcasted_iota(
                    jnp.int32, ctx.prefix_vmem.shape, 0
                )
                ctx.prefix_vmem[...] = jnp.where(
                    row_indices == owner_pair,
                    updated_prefix,
                    cached_prefix,
                )
            else:
                ctx.prefix_vmem[row_slice] = updated_prefix

            def mix_row(local_row: int) -> jax.Array:
                """Mix snapshots with updated prefix for one owned row."""
                global_row = row_offset + local_row
                return _residual_mixture_row(
                    _block_rows(ctx, _owner_block_row(plan, global_row, local_row)),
                    (
                        updated_prefix[local_row : local_row + 1]
                        if plan.owned_rows <= _MAX_UNROLLED_MIX_ROWS
                        # stored above; re-read inside the row region
                        else _owner_prefix_row(ctx, global_row, row_slice, local_row)
                    ),
                    ctx.mlp_residual_vmem,
                    layer_index // _RESIDUAL_BLOCK_STRIDE + 1,
                    epsilon=plan.eps,
                    relaxed_reduction_order=(
                        plan.residual_reduction_order == "relaxed"
                    ),
                )

            with _phase("q9b_post_rows"):
                _store_mixed_rows(ctx, mix_row, owned_normalized_vmem)

    _normalize_rows(ctx, owned_normalized_vmem, ctx.ffn_norm_vmem)
    phase_end(post_scope)
    t10 = phase_start("t10_moe_input_gather")
    _gather_local_pair_rows(
        owned_normalized_vmem,
        ctx.normalized_vmem,
        comm.send_semaphores,
        comm.receive_semaphores,
        comm.shared_local_reduction_vmem,
        owners=plan.sp_owners,
    )
    phase_end(t10)


def _prepare_post_attention(
    layer_ctx: _LayerExecutionContext,
    post_scope: PhaseScopeProtocol,
) -> None:
    """Update prefix after attention and prepare normalized MLP/MoE input."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    layer_index = layer_ctx.layer_index
    if plan.attention_dp_ownership:
        with _phase("q9b_post_rows"):
            _mix_dp_rows(
                ctx,
                ctx.mlp_residual_vmem,
                layer_index // _RESIDUAL_BLOCK_STRIDE + 1,
                _DpMixOptions(layer_index=layer_index, attention_output=True),
            )
        _normalize_dp_rows(ctx, ctx.ffn_norm_vmem, split=plan.sp8_residual)
        phase_end(post_scope)
        with _phase("t10_moe_input_gather"):
            _gather_dp_rows(ctx, ctx.normalized_vmem, split=plan.sp8_residual)
    elif plan.attention_token_scatter:
        _prepare_post_attention_scatter(layer_ctx, post_scope)
    else:
        ctx.prefix_vmem[...] = jnp.where(
            layer_index % _RESIDUAL_BLOCK_STRIDE == 0,
            ctx.attention_vmem[...],
            ctx.prefix_vmem[...] + ctx.attention_vmem[...],
        )
        if plan.aux_capture == "attention":
            _capture_aux_hidden(layer_ctx)
        _residual_mix(
            ctx,
            ctx.normalized_vmem,
            ctx.mlp_residual_vmem,
            layer_index // _RESIDUAL_BLOCK_STRIDE + 1,
        )
        _normalize_rows(ctx, ctx.normalized_vmem, ctx.ffn_norm_vmem)
        phase_end(post_scope)


def _after_experts_body(
    layer_ctx: _LayerExecutionContext,
    *,
    dense_layer: bool,
) -> None:
    """Drain KDA post-state writes and prefetch the next layer's weights."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    layer_index = layer_ctx.layer_index
    if plan.persist_kda_state:

        @pl.when(_is_kda_layer(layer_index))
        def drain_state_writes() -> None:
            """Wait for persistent KDA state writes started during attention."""
            state_index = layer_index - layer_index // MLA_LAYER_STRIDE
            state_writes = (
                tpu.make_async_copy(
                    ctx.auxiliary_vmem[4],
                    ctx.convolution_output_ref.at[:, state_index],
                    ctx.auxiliary_semaphores.at[0],
                ),
                tpu.make_async_copy(
                    ctx.auxiliary_vmem[5],
                    ctx.recurrent_output_ref.at[:, state_index],
                    ctx.auxiliary_semaphores.at[1],
                ),
            )
            for copy in state_writes:
                copy.wait()

    if ctx.pool is not None:
        active_pool = ctx.pool
        # The expert slots are free: queue the next layer's weights (issued now
        # if the arena has room, else by the release that makes it).
        if not dense_layer:
            active_pool.release("experts")
        has_next = layer_index + 1 < plan.layers
        pl.when(has_next)(lambda: _prefetch_norms(ctx, layer_index + 1))
        _request_attention(ctx, active_pool, layer_index + 1, predicate=has_next)

        @pl.when(has_next)
        def prefetch_next_layer_metadata() -> None:
            """Prefetch auxiliary and metadata vectors for ``layer_index + 1``."""
            _prefetch_auxiliary(ctx, layer_index + 1)
            _prefetch_metadata(ctx, layer_index + 1)

        _request_router(ctx, active_pool, layer_index + 1, predicate=has_next)
        return

    @pl.when(layer_index + 1 < plan.layers)
    def prefetch_next_layer() -> None:
        """Prefetch all static weights for ``layer_index + 1``."""
        _prefetch_norms(ctx, layer_index + 1)
        _prefetch_attention(ctx, layer_index + 1)
        _prefetch_auxiliary(ctx, layer_index + 1)
        _prefetch_metadata(ctx, layer_index + 1)
        _prefetch_router(ctx, layer_index + 1)


def _run_mixture_body(
    layer_ctx: _LayerExecutionContext,
    resources: tuple[PallasRefProtocol, ...],
    after_experts: Callable[[], None],
) -> None:
    """Wait for MoE metadata, bind expert slots, and invoke ``moe``."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    layer_index = layer_ctx.layer_index
    buffer_slot = layer_index % _DOUBLE_BUFFER_SLOTS
    bias_vmem = ctx.metadata_vmem[2].at[buffer_slot]
    latent_norm_vmem = ctx.metadata_vmem[3].at[buffer_slot]
    metadata_copies = (
        tpu.make_async_copy(
            ctx.layer_weights["router_bias"].at[layer_index - 1],
            bias_vmem,
            ctx.metadata_semaphores.at[buffer_slot, 2],
        ),
        tpu.make_async_copy(
            ctx.layer_weights["latent_norm"].at[layer_index - 1],
            latent_norm_vmem,
            ctx.metadata_semaphores.at[buffer_slot, 3],
        ),
    )
    for copy in metadata_copies:
        copy.wait()

    if POOLED_WEIGHTS:
        resources = (_pool_view(ctx, "latent_up"), *resources)
    latent_output_vmem, latent_output_semaphore, expert_semaphores = resources[:3]
    layer_router_vmem = ctx.router_vmem
    expert_base = 0
    after_routing_scores: Callable[[], None] | None = None
    if ctx.pool is not None:
        active_pool = ctx.pool
        layer_router_vmem = _arena_router(
            active_pool, plan.hidden_size, active_pool.acquire("router")
        )
        # The expert slots may overlap the router: its scores are computed
        # before the first expert copy starts.
        expert_base = active_pool.allocate(
            "experts",
            ctx.mem_plan.expert_pool_bytes,
            replaces=("router",),
        )

        def after_routing_scores() -> None:
            """Release router arena slot and acquire expert weight slots."""
            active_pool.release("router")
            active_pool.acquire("experts")

    if plan.alias_expert_scales:
        (
            gate_up_vmem,
            gate_up_scales_vmem,
            down_vmem,
            down_scales_vmem,
        ) = _aliased_expert_views(ctx, expert_base)
        workspace_resources = resources[3:]
    else:
        gate_up_vmem, down_vmem = _unaliased_expert_views(ctx, expert_base)
        gate_up_scales_vmem, down_scales_vmem = resources[3:5]
        workspace_resources = resources[5:]
    mixture_output_vmem = (
        ctx.owned_rows_vmem if plan.moe_token_scatter_output else ctx.mixture_vmem
    )
    moe(
        ctx.normalized_vmem,
        mixture_output_vmem,
        weights=MixtureOfExpertsWeights(
            ctx.layer_weights["router"].at[layer_index - 1],
            layer_router_vmem,
            ctx.router_semaphores,
            bias_vmem,
            ctx.layer_weights["latent_down"].at[layer_index - 1],
            ctx.layer_weights["latent_up"].at[layer_index - 1],
            latent_output_vmem,
            latent_output_semaphore,
            latent_norm_vmem,
            ctx.layer_weights["shared_gu"].at[layer_index - 1],
            ctx.layer_weights["shared_down"].at[layer_index - 1],
            ctx.layer_weights["expert_gu"].at[layer_index - 1],
            gate_up_vmem,
            ctx.layer_weights["expert_gus"].at[layer_index - 1],
            gate_up_scales_vmem,
            ctx.layer_weights["expert_down"].at[layer_index - 1],
            down_vmem,
            ctx.layer_weights["expert_ds"].at[layer_index - 1],
            down_scales_vmem,
            expert_semaphores,
        ),
        workspace=MixtureOfExpertsWorkspace(*workspace_resources),
        communication_workspace=ctx.communication_workspace,
        epsilon=plan.eps,
        synchronize_devices=False,
        routing_projection_is_prefetched=True,
        group_pending_expert_routes=(
            plan.batch_size in (6, *SCATTER_BATCHES)
            and plan.residual_reduction_order == "relaxed"
        ),
        after_experts=after_experts,
        after_routing_scores=after_routing_scores,
        overlap_router_latent=(
            plan.attention_dp == DP2_GROUPS
            and plan.batch_size == MAX_BATCH
            and plan.sequence_rows == 1
        ),
        overlap_shared_reduction=(
            plan.attention_dp == DP2_GROUPS
            and plan.batch_size in _DP_B64_B128_BATCHES
            and plan.sequence_rows == 1
        ),
        token_scatter_output=plan.moe_token_scatter_output,
        routed_reduce_scatter=plan.moe_routed_reduce_scatter,
        prepare_weights_vmem=(
            tuple(_pool_view(ctx, name) for name in MOE_PREPARE_WEIGHTS)
            if POOLED_WEIGHTS
            else None
        ),
        shared_gate_up_stream=(
            _weight_stream(
                ctx,
                "shared_gu",
                ctx.layer_weights["shared_gu"].at[layer_index - 1],
                _pool_view(ctx, "shared_gu"),
                offset=ctx.mem_plan.pool_layout["shared_gu"],
            )
            if POOLED_WEIGHTS
            else None
        ),
        token_scatter_owners=plan.sp_owners,
        owned_hidden_ref=ctx.owned_rows_vmem if plan.shard_routing else None,
        live_rows=ctx.active_rows,
    )


def _build_dense_mlp_spec(ctx: _StackRuntimeContext) -> _DenseMlpSpec:
    """Validate layer-0 dense MLP dimensions and allocate arena slabs."""
    plan = ctx.plan
    gu_ref = ctx.dense_refs["dense_gu"]
    down_ref = ctx.dense_refs["dense_down"]
    gu_width = gu_ref.shape[1]
    gu_blocks = plan.hidden_size // _DENSE_K_BLOCK_ROWS
    down_blocks = down_ref.shape[0] // _DENSE_DOWN_BLOCK_ROWS
    if gu_ref.shape != (gu_blocks * _DENSE_K_BLOCK_ROWS, gu_width):
        msg = "dense_gu shape must align with dense K block rows"
        raise ValueError(msg)
    if down_ref.shape != (
        down_blocks * _DENSE_DOWN_BLOCK_ROWS,
        plan.hidden_size,
    ):
        msg = "dense_down shape must align with dense down block rows"
        raise ValueError(msg)
    if gu_width != _GATE_UP_FACTORS * down_blocks * _DENSE_DOWN_BLOCK_ROWS:
        msg = "dense_gu width must equal twice dense_down row count"
        raise ValueError(msg)
    slab_rows = (
        max(
            pool_alias.storage_size((_DENSE_K_BLOCK_ROWS, gu_width), gu_ref.dtype),
            pool_alias.storage_size(
                (_DENSE_DOWN_BLOCK_ROWS, plan.hidden_size), down_ref.dtype
            ),
        )
        // VREG_LANE_COUNT
    )
    slabs = ctx.mem_plan.dense_slabs
    if slabs * slab_rows > ctx.projection_expert_pool_vmem.shape[0]:
        msg = "dense MLP slabs exceed projection_expert_pool capacity"
        raise ValueError(msg)
    slab_base = 0
    if ctx.pool is not None:
        slab_base = (
            ctx.pool.allocate("dense", slabs * slab_rows * VREG_LANE_COUNT)
            // VREG_LANE_COUNT
        )
        ctx.pool.acquire("dense")
    return _DenseMlpSpec(
        gu_ref=gu_ref,
        down_ref=down_ref,
        gu_width=gu_width,
        gu_blocks=gu_blocks,
        slab_rows=slab_rows,
        slabs=slabs,
        total=gu_blocks + down_blocks,
        slab_base=slab_base,
    )


def _dense_slab_view(
    ctx: _StackRuntimeContext,
    spec: _DenseMlpSpec,
    slot_index: int,
    shape_dtype: jax.ShapeDtypeStruct,
) -> PallasRefProtocol:
    """Return the aliased pool view for dense MLP slab ``slot_index``."""
    return cast(
        "PallasRefProtocol",
        pool_alias.view(
            ctx.projection_expert_pool_vmem.at[
                pl.ds(
                    spec.slab_base + slot_index * spec.slab_rows,
                    pool_alias.storage_size(shape_dtype.shape, shape_dtype.dtype)
                    // VREG_LANE_COUNT,
                ),
                :,
            ],
            shape_dtype.shape,
            shape_dtype.dtype,
        ),
    )


def _dense_block_copy(
    ctx: _StackRuntimeContext,
    spec: _DenseMlpSpec,
    block_idx: int,
) -> AsyncCopyProtocol:
    """Build async DMA for dense MLP gate/up or down block ``block_idx``."""
    slot = block_idx % spec.slabs
    if block_idx < spec.gu_blocks:
        return tpu.make_async_copy(
            spec.gu_ref.at[
                pl.ds(block_idx * _DENSE_K_BLOCK_ROWS, _DENSE_K_BLOCK_ROWS), :
            ],
            _dense_slab_view(
                ctx,
                spec,
                slot,
                jax.ShapeDtypeStruct(
                    (_DENSE_K_BLOCK_ROWS, spec.gu_width), spec.gu_ref.dtype
                ),
            ),
            ctx.dense_semaphores.at[slot],
        )
    down_idx = block_idx - spec.gu_blocks
    return tpu.make_async_copy(
        spec.down_ref.at[
            pl.ds(down_idx * _DENSE_DOWN_BLOCK_ROWS, _DENSE_DOWN_BLOCK_ROWS), :
        ],
        _dense_slab_view(
            ctx,
            spec,
            slot,
            jax.ShapeDtypeStruct(
                (_DENSE_DOWN_BLOCK_ROWS, ctx.plan.hidden_size),
                spec.down_ref.dtype,
            ),
        ),
        ctx.dense_semaphores.at[slot],
    )


def _accumulate_dense_partial(
    part: jax.Array,
    acc_ref: PallasRefProtocol | None,
    running: jax.Array | None,
    step_idx: int,
) -> jax.Array | None:
    """Accumulate one dense projection partial into scratch or register state."""
    if acc_ref is not None:
        if step_idx == 0:
            acc_ref[...] = part
        else:
            acc_ref[...] += part
        return running
    return part if running is None else running + part


def _finalize_dense_gate_up(
    spec: _DenseMlpSpec,
    proj_vmem: PallasRefProtocol | None,
    projection: jax.Array | None,
) -> jax.Array:
    """Apply gated activation after the final dense gate/up K-block."""
    if proj_vmem is not None:
        projection = proj_vmem[...]
    proj_final = _require_array(projection, "dense projection")
    # The mathematical reference rounds the projections to BF16.
    half = spec.gu_width // _GATE_UP_FACTORS
    activation = _gated_activation(
        proj_final[:, :half].astype(jnp.bfloat16),
        proj_final[:, half:].astype(jnp.bfloat16),
    )
    if proj_vmem is not None:
        proj_vmem[:, :half] = activation.astype(jnp.float32)
    return activation


def _stream_dense_blocks(
    ctx: _StackRuntimeContext,
    spec: _DenseMlpSpec,
    buffers: _DenseStreamBuffers,
) -> jax.Array:
    """Stream dense gate/up and down projection blocks through pool slabs."""
    # Above B8 the unrolled partial sums spill; accumulate them in scratch
    # buffers instead.
    hidden_in = ctx.normalized_vmem[...]
    projection: jax.Array | None = None
    activation: jax.Array | None = None
    accumulator: jax.Array | None = None
    proj_vmem = buffers.projection_vmem
    acc_vmem = buffers.accumulator_vmem
    gu_struct = jax.ShapeDtypeStruct(
        (_DENSE_K_BLOCK_ROWS, spec.gu_width), spec.gu_ref.dtype
    )
    down_struct = jax.ShapeDtypeStruct(
        (_DENSE_DOWN_BLOCK_ROWS, ctx.plan.hidden_size), spec.down_ref.dtype
    )
    for block_idx in range(spec.total):
        slot = block_idx % spec.slabs
        _dense_block_copy(ctx, spec, block_idx).wait()
        if block_idx < spec.gu_blocks:
            part = dot(
                hidden_in[
                    :,
                    block_idx * _DENSE_K_BLOCK_ROWS : (block_idx + 1)
                    * _DENSE_K_BLOCK_ROWS,
                ],
                _dense_slab_view(ctx, spec, slot, gu_struct)[...],
            )
            projection = _accumulate_dense_partial(
                part, proj_vmem, projection, block_idx
            )
            if block_idx == spec.gu_blocks - 1:
                activation = _finalize_dense_gate_up(spec, proj_vmem, projection)
        else:
            down_idx = block_idx - spec.gu_blocks
            act_final = _require_array(activation, "dense activation")
            col_start = down_idx * _DENSE_DOWN_BLOCK_ROWS
            col_end = (down_idx + 1) * _DENSE_DOWN_BLOCK_ROWS
            activation_block = (
                proj_vmem[:, col_start:col_end].astype(act_final.dtype)
                if proj_vmem is not None
                else act_final[:, col_start:col_end]
            )
            part = dot(
                activation_block,
                _dense_slab_view(ctx, spec, slot, down_struct)[...],
            )
            accumulator = _accumulate_dense_partial(
                part, acc_vmem, accumulator, down_idx
            )
        if block_idx + spec.slabs < spec.total:
            _dense_block_copy(ctx, spec, block_idx + spec.slabs).start()
    if acc_vmem is not None:
        return acc_vmem[...]
    return _require_array(accumulator, "dense accumulator")


def _reduce_and_store_dense_output(
    ctx: _StackRuntimeContext,
    accumulator: jax.Array,
) -> None:
    """All-reduce layer-0 dense MLP partial sums and write to mixture layout."""
    plan = ctx.plan
    comm = ctx.communication_workspace
    if (
        plan.moe_token_scatter_output
        and plan.attention_dp == DP2_GROUPS
        and plan.sp_owners == _PAIR_SP_OWNERS
        and plan.batch_size in _DP_B64_B128_BATCHES
        and plan.sequence_rows == 1
    ):
        _dense_scatter_rows(
            accumulator,
            ctx.owned_rows_vmem,
            comm.shared_local_reduction_vmem,
            comm.shared_host_reduction_vmem,
            comm.send_semaphores,
            comm.receive_semaphores,
            plan.sp_owners,
        )
        return

    def reduce_half(first: int) -> jax.Array:
        """All-reduce one 8-row chunk of dense MLP outputs across TP32."""
        if first:
            # Both halves reuse the transport buffers; a peer may still be
            # decoding the first half.
            collectives32.barrier()
        return _all_reduce_rows(
            accumulator[first : first + _DENSE_REDUCE_CHUNK_ROWS],
            comm.shared_local_reduction_vmem,
            comm.shared_host_reduction_vmem,
            comm.send_semaphores,
            comm.receive_semaphores,
        )

    reduced = (
        jnp.concatenate(
            [
                reduce_half(first)
                for first in range(0, plan.batch_size, _DENSE_REDUCE_CHUNK_ROWS)
            ],
            axis=0,
        )
        if plan.batch_size > _UNSCOPED_DENSE_MAX_BATCH
        else _all_reduce_rows(
            accumulator,
            comm.shared_local_reduction_vmem,
            comm.shared_host_reduction_vmem,
            comm.send_semaphores,
            comm.receive_semaphores,
        )
    )
    if plan.moe_token_scatter_output:
        rank = jax.lax.axis_index(AXIS_TP)
        pair = (rank % mla_dcp.HOST_LANES) // (mla_dcp.HOST_LANES // plan.sp_owners)
        for owner_pair in range(plan.sp_owners):

            @pl.when(pair == owner_pair)
            def store_owned_dense_rows(owner_pair: int = owner_pair) -> None:
                """Write this owner pair's dense MLP rows."""
                ctx.owned_rows_vmem[...] = reduced[
                    owner_pair * plan.owned_rows : (owner_pair + 1) * plan.owned_rows
                ].astype(jnp.bfloat16)
    else:
        ctx.mixture_vmem[...] = reduced.astype(jnp.bfloat16)


def _run_dense_mlp_body(
    layer_ctx: _LayerExecutionContext,
    after_experts: Callable[[], None],
) -> None:
    """Execute layer 0's dense MLP by streaming row blocks through pool slabs.

    Streams contiguous row blocks (``dense_slabs`` in flight): the gate/up
    projection accumulates over 1024-row K blocks of ``dense_gu`` (a column
    block would be 7168 short DMA segments), and the down projection over
    384-row blocks of ``dense_down``; TP32 sum uses the shared-expert
    transport and writes output in the mixture's layout.
    """
    ctx = layer_ctx.ctx
    plan = ctx.plan
    spec = _build_dense_mlp_spec(ctx)
    for block_idx in range(min(spec.slabs, spec.total)):
        _dense_block_copy(ctx, spec, block_idx).start()
    if plan.batch_size <= _UNSCOPED_DENSE_MAX_BATCH:
        accumulator = _stream_dense_blocks(ctx, spec, _DenseStreamBuffers())
    else:
        accumulator = pl.run_scoped(
            lambda projection_vmem, accumulator_vmem: _stream_dense_blocks(
                ctx,
                spec,
                _DenseStreamBuffers(projection_vmem, accumulator_vmem),
            ),
            tpu.VMEM((plan.batch_size, spec.gu_width), jnp.float32),
            tpu.VMEM((plan.batch_size, plan.hidden_size), jnp.float32),
        )
    # The pool is free again: stage layer 1 while the sum travels.
    if ctx.pool is not None:
        ctx.pool.release("dense")
    after_experts()
    _reduce_and_store_dense_output(ctx, accumulator)


def _finish_layer_rows(layer_ctx: _LayerExecutionContext) -> None:
    """Fold MLP/MoE output back into ``prefix_vmem`` (and DP/SP owner buffers)."""
    ctx = layer_ctx.ctx
    plan = ctx.plan
    if plan.attention_dp_ownership:
        dp_owned_vmem = _require_ref(ctx.dp_owned_vmem, "dp_owned_vmem")
        # The MoE delivered the pair's rows of every DP group (with one owner,
        # every row).
        group = jax.lax.axis_index(AXIS_TP) // (TP_SIZE // plan.attention_dp)
        mixture = ctx.owned_rows_vmem[: plan.dp_held_rows]
        for data_group in range(1, plan.attention_dp if plan.sp_owners > 1 else 1):
            first = data_group * plan.group_rows_per_pair
            mixture = jnp.where(
                group == data_group,
                ctx.owned_rows_vmem[first : first + plan.group_rows_per_pair],
                mixture,
            )
        finished = (ctx.prefix_vmem[: plan.dp_held_rows] + mixture).astype(jnp.bfloat16)
        ctx.prefix_vmem[: plan.dp_held_rows] = finished
        dp_owned_vmem[...] = finished
        return
    if not plan.attention_token_scatter:
        ctx.prefix_vmem[...] += ctx.mixture_vmem[...]
        return
    rank = jax.lax.axis_index(AXIS_TP)
    pair = (rank % mla_dcp.HOST_LANES) // (mla_dcp.HOST_LANES // plan.sp_owners)
    owned_prefix_vmem = ctx.owned_rows_vmem
    scattered_mixture = (
        ctx.owned_rows_vmem[...] if plan.moe_token_scatter_output else None
    )
    prefix_values = ctx.prefix_vmem[...] if plan.owned_rows == 1 else None
    mixture_values = (
        ctx.mixture_vmem[...]
        if plan.owned_rows == 1 and not plan.moe_token_scatter_output
        else None
    )
    half_hidden = plan.hidden_size // _HIDDEN_HALVES
    for owner_pair in _owner_indices(plan, pair):

        @_owner_when(plan, pair, owner_pair)
        def finish_owned_rows(
            owner_pair: int | jax.Array = owner_pair,
        ) -> None:
            """Add MLP output to owned prefix rows in two hidden halves."""
            row_offset = owner_pair * plan.owned_rows
            if plan.owned_rows == 1:
                cached_prefix = _require_array(prefix_values, "prefix_values")
                finished_prefix = (
                    cached_prefix[owner_pair : owner_pair + 1]
                    + (
                        # The MoE delivered this owner's row.
                        _require_array(scattered_mixture, "scattered_mixture")
                        if plan.moe_token_scatter_output
                        else _require_array(mixture_values, "mixture_values")[
                            owner_pair : owner_pair + 1
                        ]
                    )
                ).astype(jnp.bfloat16)
            else:
                row_slice = pl.ds(_owner_start(plan, row_offset), plan.owned_rows)
                finished_prefix = (
                    ctx.prefix_vmem[row_slice]
                    + (
                        _require_array(scattered_mixture, "scattered_mixture")
                        if plan.moe_token_scatter_output
                        else ctx.mixture_vmem[row_slice]
                    )
                ).astype(jnp.bfloat16)
            owned_prefix_vmem[:, :half_hidden] = finished_prefix[:, :half_hidden]
            owned_prefix_vmem[:, half_hidden:] = finished_prefix[:, half_hidden:]

    if plan.sequence_parallel_pre_attention:
        for owner_pair in _owner_indices(plan, pair):

            @_owner_when(plan, pair, owner_pair)
            def store_owned_prefix_rows(
                owner_pair: int | jax.Array = owner_pair,
            ) -> None:
                """Write finished owned prefix rows back to ``prefix_vmem``."""
                row_offset = owner_pair * plan.owned_rows
                row_slice = pl.ds(_owner_start(plan, row_offset), plan.owned_rows)
                ctx.prefix_vmem[row_slice] = owned_prefix_vmem[...]
    else:
        comm = ctx.communication_workspace
        _gather_local_pair_rows(
            owned_prefix_vmem,
            ctx.prefix_vmem,
            comm.send_semaphores,
            comm.receive_semaphores,
            comm.shared_local_reduction_vmem,
            owners=plan.sp_owners,
        )


def _run_sp_decoder_epilogue(ctx: _StackRuntimeContext) -> None:
    """Run final sequence-parallel residual mix, norm, gather, and HBM DMA."""
    plan = ctx.plan
    final_weight_copies = (
        tpu.make_async_copy(
            ctx.output_residual_ref,
            ctx.attention_residual_vmem,
            ctx.norm_semaphores.at[0],
        ),
        tpu.make_async_copy(
            ctx.final_normalization_ref,
            ctx.attention_norm_vmem,
            ctx.norm_semaphores.at[2],
        ),
    )
    for copy in final_weight_copies:
        copy.start()
    for copy in final_weight_copies:
        copy.wait()
    final_block_count = (plan.layers - 1) // _RESIDUAL_BLOCK_STRIDE + 1
    if plan.attention_dp_ownership:
        dp_owned_vmem = _require_ref(ctx.dp_owned_vmem, "dp_owned_vmem")

        def mix_row(row: int) -> jax.Array:
            """Mix final residual snapshots for one DP-owned row."""
            values = jnp.concatenate(
                (
                    _block_rows(ctx, row, final_block_count),
                    ctx.prefix_vmem[row : row + 1],
                ),
                axis=0,
            ).astype(jnp.float32)
            scores = jnp.sum(values * ctx.attention_residual_vmem[0], axis=1)
            scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + plan.eps)
            return jnp.sum(
                jax.nn.softmax(scores)[:, None] * values,
                axis=0,
                keepdims=True,
            ).astype(jnp.bfloat16)

        _store_mixed_rows(ctx, mix_row, dp_owned_vmem)
        _normalize_dp_rows(ctx, ctx.attention_norm_vmem)
        _gather_dp_rows(ctx, ctx.normalized_vmem)
    else:
        rank = jax.lax.axis_index(AXIS_TP)
        pair = (rank % mla_dcp.HOST_LANES) // (mla_dcp.HOST_LANES // plan.sp_owners)
        for owner_pair in _owner_indices(plan, pair):

            @_owner_when(plan, pair, owner_pair)
            def finish_owned_decoder_rows(
                owner_pair: int | jax.Array = owner_pair,
            ) -> None:
                """Mix final residual snapshots for one SP owner pair."""
                row_offset = owner_pair * plan.owned_rows
                row_slice = pl.ds(_owner_start(plan, row_offset), plan.owned_rows)

                def mix_row(local_row: int) -> jax.Array:
                    """Mix final residual snapshots for one owned row."""
                    global_row = row_offset + local_row
                    values = jnp.concatenate(
                        (
                            _block_rows(
                                ctx,
                                _owner_block_row(plan, global_row, local_row),
                                final_block_count,
                            ),
                            _owner_prefix_row(ctx, global_row, row_slice, local_row),
                        ),
                        axis=0,
                    ).astype(jnp.float32)
                    scores = jnp.sum(values * ctx.attention_residual_vmem[0], axis=1)
                    scores *= jax.lax.rsqrt(
                        jnp.mean(values * values, axis=1) + plan.eps
                    )
                    return jnp.sum(
                        jax.nn.softmax(scores)[:, None] * values,
                        axis=0,
                        keepdims=True,
                    ).astype(jnp.bfloat16)

                _store_mixed_rows(ctx, mix_row, ctx.owned_rows_vmem)

        _normalize_rows(ctx, ctx.owned_rows_vmem, ctx.attention_norm_vmem)
        comm = ctx.communication_workspace
        _gather_local_pair_rows(
            ctx.owned_rows_vmem,
            ctx.normalized_vmem,
            comm.send_semaphores,
            comm.receive_semaphores,
            comm.shared_local_reduction_vmem,
            owners=plan.sp_owners,
        )

    def store_output(semaphore: PallasRefProtocol) -> None:
        """DMA final normalized hidden states to HBM."""
        copy = tpu.make_async_copy(ctx.normalized_vmem, ctx.output_ref, semaphore)
        copy.start()
        copy.wait()

    pl.run_scoped(store_output, tpu.SemaphoreType.DMA)
    ctx.blocks_output_ref[...] = jnp.zeros(ctx.blocks_output_ref.shape, jnp.uint8)


def _run_stack_epilogue(ctx: _StackRuntimeContext) -> None:
    """Apply final residual mix, normalization, and HBM output DMAs."""
    if ctx.headroom_pad is not None:
        # Keep the pad live through the whole program (it holds zeros).
        ctx.prefix_vmem[0:SUBLANE_ROWS, 0:VREG_LANE_COUNT] += jnp.sum(
            ctx.headroom_pad[...].reshape(-1, SUBLANE_ROWS, VREG_LANE_COUNT),
            axis=0,
        ).astype(jnp.bfloat16)

    if ctx.plan.sequence_parallel_pre_attention:
        _run_sp_decoder_epilogue(ctx)
        return

    def store_output(semaphore: PallasRefProtocol) -> None:
        """DMA final unnormalized prefix hidden states to HBM."""
        copy = tpu.make_async_copy(ctx.prefix_vmem, ctx.output_ref, semaphore)
        copy.start()
        copy.wait()

    pl.run_scoped(store_output, tpu.SemaphoreType.DMA)
    ctx.blocks_output_ref[...] = ctx.blocks_vmem[...].astype(jnp.bfloat16)


def _execute_layer_attention(layer_ctx: _LayerExecutionContext) -> None:
    """Acquire attention pool slot, run MLA or KDA, and release the slot."""
    ctx = layer_ctx.ctx

    def run_delta_attention() -> None:
        """Run one gated delta attention sublayer."""
        _run_delta_attention_body(layer_ctx)

    def run_latent_attention() -> None:
        """Run one multi-head latent attention sublayer."""
        _run_latent_attention_body(layer_ctx)

    if ctx.pool is not None:
        # Attention waits for its copies.
        ctx.pool.acquire("attention")
    with _phase("p1_attention"):
        jax.lax.cond(
            _is_mla_layer(layer_ctx.layer_index),
            run_latent_attention,
            run_delta_attention,
        )
    if ctx.pool is not None:
        ctx.pool.release("attention")


def _execute_layer_mlp(
    layer_ctx: _LayerExecutionContext,
    *,
    dense_layer: bool,
) -> None:
    """Run optional mixture-input aux capture, dense MLP or MoE sublayer."""
    ctx = layer_ctx.ctx

    def after_experts() -> None:
        """Drain state writes and prefetch the next layer."""
        _after_experts_body(layer_ctx, dense_layer=dense_layer)

    def run_mixture(*resources: PallasRefProtocol) -> None:
        """Execute routed + shared MoE sublayer."""
        _run_mixture_body(layer_ctx, resources, after_experts)

    if ctx.plan.aux_capture == "mixture_input":
        _capture_aux_hidden(layer_ctx, ctx.normalized_vmem)
    if dense_layer:
        if "dense" not in DEBUG_SKIP:
            _run_dense_mlp_body(layer_ctx, after_experts)
    else:
        with _phase("p0_moe"):
            if "moe" not in DEBUG_SKIP:
                pl.run_scoped(run_mixture, *ctx.mem_plan.moe_scoped_scratch)


def _execute_layer_body(
    ctx: _StackRuntimeContext,
    layer_index: int | jax.Array,
    *,
    dense_layer: bool = False,
) -> None:
    """Execute one decoder layer (attention, residual/norm, MLP/MoE)."""
    plan = ctx.plan
    # This layer's attention projections: placed by the arena manager when its
    # copy was issued, else at the start of the pool.
    attention_base = ctx.pool.offset("attention") if ctx.pool is not None else 0
    kda_projection_vmem, mla_projection_vmem = _attention_views(ctx, attention_base)
    layer_ctx = _LayerExecutionContext(
        ctx=ctx,
        layer_index=layer_index,
        attention_base=attention_base,
        kda_projection_vmem=kda_projection_vmem,
        mla_projection_vmem=mla_projection_vmem,
    )
    norm_copies = [
        tpu.make_async_copy(
            ctx.layer_weights[name].at[layer_index],
            destination,
            ctx.norm_semaphores.at[index],
        )
        for index, (name, destination) in enumerate(
            (
                ("attn_res", ctx.attention_residual_vmem),
                ("mlp_res", ctx.mlp_residual_vmem),
                ("attn_norm", ctx.attention_norm_vmem),
                ("ffn_norm", ctx.ffn_norm_vmem),
            )
        )
    ]
    norm_copies[0].wait()
    norm_copies[2].wait()
    layer_scope = [phase_start("q1_pre_mix")]
    _prepare_pre_attention_residual(layer_ctx)
    phase_end(layer_scope.pop())
    layer_scope.append(phase_start("q2_pre_norm_gather"))
    _prepare_pre_attention_norm_and_gather(layer_ctx)
    phase_end(layer_scope.pop())
    _execute_layer_attention(layer_ctx)
    post_scope = phase_start("q9_post_attn")
    with _phase("q9a_post_norm_wait"):
        norm_copies[1].wait()
        norm_copies[3].wait()
    _prepare_post_attention(layer_ctx, post_scope)
    _execute_layer_mlp(layer_ctx, dense_layer=dense_layer)
    finish_scope = phase_start("q0_finish_rows")
    _finish_layer_rows(layer_ctx)
    phase_end(finish_scope)
    if plan.aux_capture == "layer":
        _capture_aux_hidden(
            layer_ctx,
            gather_owned_rows=plan.sequence_parallel_pre_attention,
        )


def _execute_stack_body(ctx: _StackRuntimeContext) -> None:
    """Run prologue, optional layer 0, layer loop, and final epilogue."""
    _run_stack_prologue(ctx)

    # Layer 0 (dense MLP) is peeled out of the loop as straight-line code so
    # the loop body stays the routed-expert body without a conditional around
    # the mixture (a control-flow region there costs the scheduler the overlap
    # of the mixture's start and tail with attention work).
    def layer(
        layer_index: int | jax.Array,
        *,
        dense_layer: bool = False,
    ) -> None:
        """Execute one decoder layer inside the fused stack loop."""
        _execute_layer_body(ctx, layer_index, dense_layer=dense_layer)

    if ctx.plan.include_layer_zero:
        layer(jnp.int32(0), dense_layer=True)
    if ctx.pool is not None:
        ctx.pool.loop(1, ctx.plan.layers)(layer)
        if ctx.pool.report:
            _logger.info("%s", ctx.pool.timeline())
    else:
        pl.loop(1, ctx.plan.layers)(layer)
    _run_stack_epilogue(ctx)


def stack(
    prefix: jax.Array,
    embedding: jax.Array,
    weights: Mapping[str, jax.Array],
    states: Sequence[jax.Array],
    positions: jax.Array,
    *,
    batch_size: int = 1,
    layers: int = NUM_LAYERS,
    eps: float = 1e-5,
    residual_reduction_order: Literal["reference", "row", "relaxed"] = "reference",
    mla_cache_batch_tile_size: int | None = None,
    alias_expert_scales: bool = False,
    attention_token_scatter: bool = False,
    moe_token_scatter_output: bool = False,
    moe_routed_reduce_scatter: bool = False,
    sequence_parallel_pre_attention: bool = False,
    sequence_rows: int = 1,
    aux_hidden_layers: tuple[int, ...] = (),
    aux_capture: Literal["layer", "attention", "mixture_input"] = "layer",
    include_layer_zero: bool = False,
    state_row: int | jax.Array | None = None,
    attention_dp: int = 1,
    attention_projection_schedule: Literal["default", "mla_up_double"] = "default",
    mla_commit_tile_size: int | None = None,
    sequence_parallel_size: int = 4,
    shard_routing: bool | None = None,
    block_tables: jax.Array | None = None,
    dcp_pool_map: tuple[int, ...] | None = None,
    dcp_owner_table: tuple[int, ...] | None = None,
    state_slots: jax.Array | None = None,
    live_rows: jax.Array | None = None,
) -> _StackOutputs:
    """Run layers ``1..layers-1`` (or ``0..layers-1``) in one fused Pallas kernel.

    ``prefix`` and ``embedding`` are ``[batch_size, hidden]`` BF16, ``weights``
    holds the local layer arrays from ``load.selected_local_shapes()``,
    ``states`` is ``(conv, recurrent, *pools)`` over vLLM's unified pools, and
    ``positions`` is ``[batch_size]`` int32. The pools stay read-only: in place
    of an updated cache the kernel returns each row's new latent for every MLA
    layer (``paged.commit_latent_rows`` commits them). Nothing is aliased: XLA
    copies every operand a communicating kernel updates in place, so the KDA
    states come back as fresh outputs too.

    The residual stream stays in on-chip memory for the entire stack, and the
    layer weights and states stream in from HBM ahead of each layer. By default
    the MoE routed-expert weights share one aliased weight pool with the
    attention projections; with ``alias_expert_scales=True``, the four
    routed-expert scale slots move into that same pool as well.

    ``residual_reduction_order`` selects how the residual mixture sums the
    active snapshot blocks: ``"reference"`` matches the float32 tree reduction,
    ``"row"`` evaluates row-wise reductions, and ``"relaxed"`` relaxes reduction
    order and groups pending routes on supported scatter batches.

    With the token-scatter / sequence-parallel layout, ``sequence_parallel_size``
    local row owners (4: rank pairs; 8: single ranks; 1: every rank, at
    attention DP2's B2 and B4), replicated across the hosts, each own
    ``batch_size / size`` rows for the per-token work.
    ``attention_token_scatter`` and ``moe_token_scatter_output`` keep residual
    updates and layer norms distributed across those row owners.
    ``moe_routed_reduce_scatter`` reduce-scatters routed expert outputs across
    host lanes. ``sequence_parallel_pre_attention`` keeps pre-attention residual
    mixes distributed across row owners and folds the final residual mixture
    and normalization into the kernel epilogue.

    ``aux_hidden_layers`` captures the residual stream entering each named layer
    at ``aux_capture`` (``"layer"``, ``"attention"``, or ``"mixture_input"``).
    ``include_layer_zero`` peels layer 0's KDA and dense MLP ahead of the loop.
    ``state_row`` (a traced int32 scalar, one sequence per tile only) selects
    the row of the KDA input states to read for the sequence's first row
    instead of a pre-selected state copy. ``state_slots`` (int32
    ``[batch_size]``) treats ``states[:2]`` as the whole KDA slot arrays
    ``[slots, 69, ...]`` so row ``b`` reads its input state from
    ``state_slots[b]`` (DP2 indices include the head-shard bank offset and
    follow shard-major group rows; with ``state_row``, slot
    ``state_slots[state_row]``). ``attention_dp`` partitions attention across
    DP groups over DCP shards.
    """
    request = _StackRequest(
        prefix=prefix,
        embedding=embedding,
        weights=weights,
        states=states,
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
        include_layer_zero=include_layer_zero,
        state_row=state_row,
        attention_dp=attention_dp,
        attention_projection_schedule=attention_projection_schedule,
        mla_commit_tile_size=mla_commit_tile_size,
        sequence_parallel_size=sequence_parallel_size,
        shard_routing=shard_routing,
        block_tables=block_tables,
        dcp_pool_map=dcp_pool_map,
        dcp_owner_table=dcp_owner_table,
        state_slots=state_slots,
        live_rows=live_rows,
    )
    plan = _validate_and_plan_stack(request)
    vmem_limit_bytes = _config().vmem_limit_mb * MIB_BYTES
    mem_plan = _plan_stack_memory(request, plan, vmem_limit_bytes=vmem_limit_bytes)

    def body(
        position_ref: PallasRefProtocol,
        state_row_ref: PallasRefProtocol,
        *refs: PallasRefProtocol,
    ) -> None:
        """Execute the fused Kimi-K3 decoder stack inside ``pl.pallas_call``."""
        _execute_stack_body(
            _build_runtime_context(plan, mem_plan, position_ref, state_row_ref, refs)
        )

    state_row_scalar = (
        jnp.asarray(state_row, jnp.int32).reshape(1)
        if plan.select_state_row
        else jnp.zeros((1,), jnp.int32)
    )
    internal_scratch_kb = _config().internal_scratch_kb
    return cast(
        "_StackOutputs",
        pl.pallas_call(
            body,
            name="kimi_decoder_stack",
            out_shape=mem_plan.output_shapes,
            input_output_aliases={},
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=(
                    _SCALAR_PREFETCH_BASE
                    + int(plan.slot_reads)
                    + int(plan.has_live_rows)
                ),
                in_specs=[
                    pl.BlockSpec(memory_space=tpu.HBM)
                    for _ in range(len(mem_plan.args))
                ],
                out_specs=[
                    pl.BlockSpec(memory_space=tpu.HBM),
                    pl.BlockSpec(memory_space=tpu.VMEM),
                    *(
                        pl.BlockSpec(memory_space=tpu.HBM)
                        for _ in range(_HBM_TAIL_OUTPUT_COUNT)
                    ),
                ],
                scratch_shapes=mem_plan.scratch,
            ),
            compiler_params=tpu.CompilerParams(
                collective_id=_STACK_COLLECTIVE_ID,
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=not _config().bounds_checks,
                shape_invariant_numerics=True,
                internal_scratch_in_bytes=(
                    internal_scratch_kb * _KIB_BYTES if internal_scratch_kb else None
                ),
            ),
        )(
            positions,
            state_row_scalar,
            jnp.asarray(block_tables, jnp.int32).reshape(-1),
            *(
                (jnp.asarray(state_slots, jnp.int32).reshape(plan.batch_size),)
                if plan.slot_reads
                else ()
            ),
            *(
                (jnp.asarray(live_rows, jnp.int32).reshape(plan.batch_size),)
                if plan.has_live_rows
                else ()
            ),
            *mem_plan.args,
        ),
    )


def _gather_token_embeddings(
    weights: Mapping[str, jax.Array],
    tokens: jax.Array,
    *,
    batched_decoder_boundaries: bool,
) -> jax.Array:
    """Gather TP-sharded token embeddings across all 32 ranks."""
    rank = mesh_axis_index(AXIS_TP)
    vocabulary = weights["embedding"].shape[0]
    if batched_decoder_boundaries:
        local_ids = tokens - rank * vocabulary
        local_embeddings = weights["embedding"][jnp.clip(local_ids, 0, vocabulary - 1)]
        return kimi.tp_sum(
            jnp.where(
                ((local_ids >= 0) & (local_ids < vocabulary))[:, None],
                local_embeddings.astype(jnp.float32),
                0,
            )
        ).astype(jnp.bfloat16)

    # Per-row embedding gather and TP sum (see batched_decoder_boundaries).
    def prepare_first_layer_row(
        _carry: tuple[()],
        token: jax.Array,
    ) -> tuple[tuple[()], jax.Array]:
        """Gather and TP-sum one token's embedding row."""
        local_id = token - rank * vocabulary
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
        return (), embedding_row[0]

    _, embedding = jax.lax.scan(prepare_first_layer_row, (), tokens)
    return embedding


def _compute_decoder_logits(
    weights: Mapping[str, jax.Array],
    prefix: jax.Array,
    blocks: jax.Array,
    logits_config: _DecoderLogitsConfig,
) -> jax.Array:
    """Run final residual mix/RMS (if needed) and project to local LM logits."""
    if (
        logits_config.sequence_parallel_pre_attention
        and logits_config.batched_decoder_boundaries
    ):
        return jnp.dot(prefix, weights["lm_head"], preferred_element_type=jnp.float32)
    if logits_config.sequence_parallel_pre_attention:

        def project_decoder_row(
            _carry: tuple[()],
            hidden_row: jax.Array,
        ) -> tuple[tuple[()], jax.Array]:
            """Project one normalized hidden row onto the local LM head."""
            logits_row = jnp.dot(
                hidden_row[None],
                weights["lm_head"],
                preferred_element_type=jnp.float32,
            )
            return (), logits_row[0]

        _, logits = jax.lax.scan(project_decoder_row, (), prefix)
        return logits

    active_block_count = (logits_config.layers - 1) // _RESIDUAL_BLOCK_STRIDE + 1

    def finish_decoder(
        _carry: tuple[()],
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[()], jax.Array]:
        """Finish one decoder row with residual mix, final RMS, and LM head."""
        prefix_row, block_rows = inputs
        hidden_row = kimi.attention_residual(
            prefix_row,
            block_rows[:active_block_count],
            weights["output_res"][0],
            logits_config.eps,
        )[None]
        hidden_row = kimi.rms(hidden_row, weights["final_norm"], logits_config.eps)
        logits_row = jnp.dot(
            hidden_row,
            weights["lm_head"],
            preferred_element_type=jnp.float32,
        )
        return (), logits_row[0]

    _, logits = jax.lax.scan(finish_decoder, (), (prefix, blocks))
    return logits


def decode_local(
    weights: Mapping[str, jax.Array],
    states: Sequence[jax.Array],
    tokens: jax.Array,  # [batch_size]
    positions: jax.Array,  # [batch_size]
    *,
    batch_size: int,
    layers: int = NUM_LAYERS,
    eps: float = 1e-5,
    residual_reduction_order: Literal["reference", "row", "relaxed"] = "reference",
    mla_cache_batch_tile_size: int | None = None,
    alias_expert_scales: bool = False,
    attention_token_scatter: bool = False,
    moe_routed_reduce_scatter: bool = False,
    moe_token_scatter_output: bool = False,
    sequence_parallel_pre_attention: bool = False,
    batched_decoder_boundaries: bool | None = None,
    sequence_rows: int = 1,
    aux_hidden_layers: tuple[int, ...] = (),
    aux_capture: Literal["layer", "attention", "mixture_input"] = "layer",
    state_row: int | jax.Array | None = None,
    attention_dp: int = 1,
    attention_projection_schedule: Literal["default", "mla_up_double"] = "default",
    mla_commit_tile_size: int | None = None,
    sequence_parallel_size: int = 4,
    block_tables: jax.Array | None = None,
    dcp_pool_map: tuple[int, ...] | None = None,
    dcp_owner_table: tuple[int, ...] | None = None,
    state_slots: jax.Array | None = None,
    live_rows: jax.Array | None = None,
) -> tuple[jax.Array, _DecodeStates] | tuple[jax.Array, jax.Array, _DecodeStates]:
    """Decode one native batch tile with local TP32 weights and caches.

    ``states`` is ``(conv, recurrent, *pools)`` over vLLM's unified-pool
    tensors ``[blocks, 128, 2, 640]`` (read in place, never written here),
    MLA layer ``m`` in ``pools[dcp_pool_map[m]]``, ``block_tables``
    (``[batch_size, pages]`` int32) the rows' tables, ``dcp_owner_table``
    (mesh position -> page owner index, e.g. vLLM TP rank) which pages each
    rank holds. The returned states are ``(conv, recurrent, new_keys,
    new_values)``: each row's new latent for every MLA layer (``[batch, 24, H,
    256]`` / ``[batch, 24, H, 128]``), for ``paged.commit_latent_rows`` to
    commit.

    Layer zero (its KDA and dense MLP) runs inside the fused stack after the
    XLA embedding gather. ``state_row`` (int32 scalar, requires
    ``sequence_rows == batch_size``) names the row of the KDA input states to
    start the sequence from, replacing a pre-selected state copy.

    ``sequence_rows`` groups consecutive rows into one sequence of consecutive
    tokens (speculative verification): KDA and MLA states lead with rows and
    sequences respectively, every row's KDA post-state is returned, and MLA
    pools receive all rows. ``aux_hidden_layers`` additionally returns the
    residual stream entering those layers as ``[len, batch, hidden]``.
    """
    if batched_decoder_boundaries is None:
        # The embedding gather, the dense MLP and (sequence-parallel) lm_head
        # run on the whole tile instead of row by row; the collectives then
        # carry batch-sized payloads, so the results differ from the per-row
        # path at the ULP level.
        batched_decoder_boundaries = batch_size in SCATTER_BATCHES or attention_dp > 1
    if not 1 <= batch_size <= MAX_BATCH:
        msg = (
            f"The native decoder supports batch sizes from 1 to {MAX_BATCH}, "
            f"got batch_size={batch_size}"
        )
        raise ValueError(msg)
    aux_hidden_layers = tuple(aux_hidden_layers)
    embedding = _gather_token_embeddings(
        weights,
        tokens,
        batched_decoder_boundaries=batched_decoder_boundaries,
    )
    if state_row is not None and sequence_rows != batch_size:
        msg = (
            "state_row needs one sequence per tile, "
            f"got sequence_rows={sequence_rows}, batch_size={batch_size}"
        )
        raise ValueError(msg)
    # Layer zero runs inside the fused stack and starts from the embedding.
    prefix = embedding
    convolution_states, recurrent_states = states[:_KDA_STATE_PREFIX_COUNT]
    cache_states = tuple(states[_KDA_STATE_PREFIX_COUNT:])
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
    logits = _compute_decoder_logits(
        weights,
        prefix,
        blocks,
        _DecoderLogitsConfig(
            layers=layers,
            eps=eps,
            sequence_parallel_pre_attention=sequence_parallel_pre_attention,
            batched_decoder_boundaries=batched_decoder_boundaries,
        ),
    )
    new_states: _DecodeStates = (
        convolution_states,
        recurrent_states,
        new_keys,
        new_values,
    )
    if aux_hidden_layers:
        return logits[None], aux_hidden, new_states
    return logits[None], new_states
