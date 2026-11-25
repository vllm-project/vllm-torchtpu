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
# src/vllm_torchtpu/megakernel/kimi_k3/kernels/paged.py
"""The serving step of the Kimi K3 decode megakernel over vLLM's unified pools.

vLLM keeps each MLA layer's latent pages in its unified pools (``[blocks, 128,
2, 640]`` bf16, token ``t`` of a block at ``[t // 2, t % 2]``; FP8 packs 512
tokens), each DCP rank holding its share of every sequence's pages, and the
KDA state in per-request slots. Two pieces serve it:

* :func:`decode_paged` -- one decode step for ``batch_size`` rows: gathers the
  rows' KDA slots, runs the fused decoder with the pools as read-only
  operands (``decode_local``), scatters the KDA rows back and commits the new
  latent rows with :func:`commit_latent_rows`. The fused decoder
  communicates across devices, and XLA copies every operand such a kernel
  updates in place (a full copy of the pools per step), so the only in-place
  pool update is the separate, non-communicating commit.
* :func:`commit_latent_rows` -- writes each row's new latent on the rank that
  owns its position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, TypedDict, Unpack

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from vllm_torchtpu.megakernel.core.collectives.spmd import mesh_axis_index
from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_TP,
    DP2_GROUPS,
    DP4_GROUPS,
    DP_VALID_BATCHES,
    MLA_LATENT_ROW_LANES,
    SUBLANE_ROWS,
    TP_SIZE,
)
from vllm_torchtpu.models.kimi_k3.megakernel.config import active as _config
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import mla_dcp
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.decode_megakernel import (
    MLA_LAYERS,
    decode_local,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from vllm_torchtpu.models.kimi_k3.megakernel.common.protocols import (
        PallasRefProtocol,
    )

_HANDOFF_ROPE_TAIL_LANES = 128
_POOL_BLOCK_ROWS = 128
_DP1_IN_PLACE_STATE_MAX_GATHER_BATCH = 4
_SCATTER_KDA_COMMIT_MAX_BATCH = 16
_DECODE_WITH_AUX_TUPLE_LEN: Final = 3
_FUSED_COMMIT_BATCHES = (1, 2, 4, 8, 16, 24, 32, 48, 64)
_DP4_IN_PLACE_BATCHES = (32, 64, 96, 128)


class _DecodeLocalOptions(TypedDict, total=False):
    """Keyword options forwarded from ``decode_paged`` to ``decode_local``."""

    layers: int
    eps: float
    residual_reduction_order: Literal["reference", "row", "relaxed"]
    mla_cache_batch_tile_size: int | None
    alias_expert_scales: bool
    attention_token_scatter: bool
    moe_routed_reduce_scatter: bool
    moe_token_scatter_output: bool
    sequence_parallel_pre_attention: bool
    batched_decoder_boundaries: bool | None
    sequence_rows: int
    aux_hidden_layers: tuple[int, ...]
    aux_capture: Literal["layer", "attention", "mixture_input"]
    attention_dp: int
    attention_projection_schedule: Literal["default", "mla_up_double"]
    mla_commit_tile_size: int | None
    sequence_parallel_size: int


def latent_rows_from_handoff(new_keys: jax.Array) -> jax.Array:
    """Extract ``[batch, M, 640]`` latent rows from a DCP decoder's handoff.

    Latent rows are packed in new-key rows ``0..2`` (see ``decode_megakernel.mla``).
    """
    return jnp.concatenate(
        (
            new_keys[:, :, 0],
            new_keys[:, :, 1],
            new_keys[:, :, 2, :_HANDOFF_ROPE_TAIL_LANES],
        ),
        axis=-1,
    )


def _normalized_pool_block(
    pool_size: int,
    block: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Wrap negative block index and check ``[0, pool_size)`` bounds."""
    target = jnp.where(block < 0, block + pool_size, block)
    return target, (target >= 0) & (target < pool_size)


def _fused_commit_call_spec(
    pools: Sequence[jax.Array],
    layers: int,
    packing: int,
) -> tuple[
    tuple[jax.ShapeDtypeStruct, ...],
    tpu.PrefetchScalarGridSpec,
    dict[int, int],
]:
    """Build output shapes, grid spec, and input-output aliases for pool commit."""
    count = len(pools)
    dtype = pools[0].dtype
    out_shape = tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in pools)
    grid_spec = tpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=(pl.BlockSpec(memory_space=tpu.VMEM),)
        + (pl.BlockSpec(memory_space=tpu.HBM),) * count,
        out_specs=(pl.BlockSpec(memory_space=tpu.HBM),) * count,
        scratch_shapes=(
            tpu.VMEM(
                (layers, SUBLANE_ROWS, packing, MLA_LATENT_ROW_LANES),
                dtype,
            ),
            tpu.SemaphoreType.DMA((layers,)),
        ),
    )
    aliases = {2 + idx: idx for idx in range(count)}
    return out_shape, grid_spec, aliases


def _commit_latent_rows_fused(
    pools: Sequence[jax.Array],
    rows: jax.Array,
    positions: jax.Array,
    blocks: jax.Array,
    owned: jax.Array,
    pool_map: Sequence[int],
) -> tuple[jax.Array, ...]:
    """Alias all unified pools in one non-communicating cache commit.

    Overlap the layers' native-tile reads. Retire all writes before
    advancing to the next token so rows sharing a tile preserve each other.
    """
    batch, layers, _ = rows.shape
    packing, dtype = pools[0].shape[-2], pools[0].dtype
    page_tokens = pools[0].shape[-3] * packing
    tile_tokens = SUBLANE_ROWS * packing
    metadata = jnp.stack((positions[:batch], blocks, owned.astype(jnp.int32)))
    count = len(pools)

    def kernel(
        meta: PallasRefProtocol,
        values: PallasRefProtocol,
        *refs: PallasRefProtocol,
    ) -> None:
        """Read, patch, and write owned latent tiles across unified pools."""
        outputs = refs[count : 2 * count]
        tiles, semaphores = refs[-2:]

        @pl.loop(0, batch)
        def commit_row(row: jax.Array) -> None:
            """Commit all MLA layers for batch row ``row`` when owned."""

            @pl.when(meta[2, row] != 0)
            def owned_row() -> None:
                """Stage, splice, and retire the tile for an owned token position."""
                position, block = meta[0, row], meta[1, row]
                first = pl.multiple_of(
                    (position % page_tokens // tile_tokens) * SUBLANE_ROWS,
                    SUBLANE_ROWS,
                )

                for layer, index in enumerate(pool_map):
                    target, valid = _normalized_pool_block(
                        outputs[index].shape[0], block
                    )

                    @pl.when(valid)
                    def read(
                        layer: int = layer,
                        index: int = index,
                        target: jax.Array = target,
                    ) -> None:
                        """Start async read of the destination cache tile."""
                        tpu.make_async_copy(
                            outputs[index].at[target, pl.ds(first, SUBLANE_ROWS)],
                            tiles.at[layer],
                            semaphores.at[layer],
                        ).start()

                for layer, index in enumerate(pool_map):
                    target, valid = _normalized_pool_block(
                        outputs[index].shape[0], block
                    )

                    @pl.when(valid)
                    def update(
                        layer: int = layer,
                        index: int = index,
                        target: jax.Array = target,
                    ) -> None:
                        """Wait for tile read, splice row latent, and start write."""
                        copy = tpu.make_async_copy(
                            outputs[index].at[target, pl.ds(first, SUBLANE_ROWS)],
                            tiles.at[layer],
                            semaphores.at[layer],
                        )
                        copy.wait()
                        slot_offset = packing * jax.lax.broadcasted_iota(
                            jnp.int32,
                            (SUBLANE_ROWS, packing, MLA_LATENT_ROW_LANES),
                            0,
                        ) + jax.lax.broadcasted_iota(
                            jnp.int32,
                            (SUBLANE_ROWS, packing, MLA_LATENT_ROW_LANES),
                            1,
                        )
                        tiles[layer] = jnp.where(
                            slot_offset == position % tile_tokens,
                            values[row, layer][None, None, :],
                            tiles[layer],
                        )
                        tpu.make_async_copy(
                            tiles.at[layer],
                            outputs[index].at[target, pl.ds(first, SUBLANE_ROWS)],
                            semaphores.at[layer],
                        ).start()

                for layer, index in enumerate(pool_map):
                    target, valid = _normalized_pool_block(
                        outputs[index].shape[0], block
                    )

                    @pl.when(valid)
                    def retire(
                        layer: int = layer,
                        index: int = index,
                        target: jax.Array = target,
                    ) -> None:
                        """Wait for async write of the updated cache tile."""
                        tpu.make_async_copy(
                            tiles.at[layer],
                            outputs[index].at[target, pl.ds(first, SUBLANE_ROWS)],
                            semaphores.at[layer],
                        ).wait()

    out_shape, grid_spec, aliases = _fused_commit_call_spec(pools, layers, packing)
    return tuple(
        pl.pallas_call(
            kernel,
            out_shape=out_shape,
            grid_spec=grid_spec,
            input_output_aliases=aliases,
            name="kimi_commit_unified_pools",
        )(metadata, rows.astype(dtype), *pools)
    )


def commit_latent_rows(
    pools: Sequence[jax.Array],
    rows: jax.Array,
    block_tables: jax.Array,
    positions: jax.Array,
    pool_map: Sequence[int],
    *,
    group: int = mla_dcp.GROUP,
    owner_table: Sequence[int] | None = None,
    global_pages: bool | None = None,
) -> tuple[jax.Array, ...]:
    """Write each row's new latent into vLLM's unified pools on its owner rank.

    Inside ``shard_map`` over ``tp``, MLA layer ``m``'s row at position ``p``
    lands in ``pools[pool_map[m]]`` at token ``p % page_tokens`` of block
    ``block_tables[b, e]``, ``e = p // page_tokens`` (global tables) or
    ``p // (page_tokens * group)`` (local tables). ``owner_table`` matches the
    decoder's ``dcp_owner_table`` (default the mesh position), and
    ``global_pages`` defaults to ``mla_dcp.GLOBAL_PAGES``.
    """
    if global_pages is None:
        global_pages = mla_dcp.GLOBAL_PAGES
    member = mla_dcp.owner_index(owner_table, rank=mesh_axis_index(AXIS_TP)) % group
    pool_list = list(pools)
    packing = pool_list[0].shape[-2]
    page_tokens = pool_list[0].shape[-3] * packing
    pages = positions[: rows.shape[0]] // page_tokens
    owned = pages % group == member
    # Local tables hold one entry per ``group`` global pages.
    blocks = block_tables[
        jnp.arange(rows.shape[0]),
        pages if global_pages else pages // group,
    ]
    # One fused commit for all MLA layers; the generic scatter below handles
    # other layouts and repeated pool maps (ordered by layer, not token).
    if (
        _config().fused_cache_commit
        and rows.shape[0] in _FUSED_COMMIT_BATCHES
        and rows.shape[1:] == (len(pool_map), MLA_LATENT_ROW_LANES)
        and 0 < len(pool_map) <= MLA_LAYERS
        and len(set(pool_map)) == len(pool_map)
        and all(0 <= index < len(pool_list) for index in pool_map)
        and all(
            pool.shape[0] > 0
            and pool.shape[1:] == (_POOL_BLOCK_ROWS, packing, MLA_LATENT_ROW_LANES)
            and pool.dtype == pool_list[0].dtype
            for pool in pool_list
        )
    ):
        return _commit_latent_rows_fused(
            pool_list, rows, positions, blocks, owned, pool_map
        )
    pairs = positions[: rows.shape[0]] % page_tokens // packing
    halves = positions[: rows.shape[0]] % packing
    # One scatter per pool for all rows; rows this rank does not own point
    # past the pool and are dropped.
    for layer, index in enumerate(pool_map):
        pool = pool_list[index]
        targets = jnp.where(owned, blocks, pool.shape[0])
        pool_list[index] = pool.at[targets, pairs, halves].set(
            rows[:, layer].astype(pool.dtype), mode="drop"
        )
    return tuple(pool_list)


def commit_kda_states(
    conv_slots: jax.Array,
    recurrent_slots: jax.Array,
    conv: jax.Array,
    recurrent: jax.Array,
    slots: jax.Array,
    source_rows: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Copy complete KDA rows back without XLA's per-row scatter loop.

    Row writes finish in input order, including repeated null-slot indices.
    The banks alias their outputs so untouched slots retain their bytes.
    With ``source_rows``, DMA only the selected verifier post-states directly
    from their HBM buffer; do not gather a second intermediate with XLA.
    """
    batch = slots.shape[0]
    selected = source_rows is not None
    prefetch = 2 if selected else 1

    def body(
        indices: PallasRefProtocol,
        *args: PallasRefProtocol,
    ) -> None:
        """Copy each batch row's KDA conv and recurrent states into ``slots``."""
        sources, rest = (args[0], args[1:]) if selected else (None, args)
        (
            _old_conv,
            _old_state,
            new_conv,
            new_state,
            out_conv,
            out_state,
            conv_buffer,
            state_buffer,
            semaphores,
        ) = rest

        @pl.loop(0, batch)
        def row(row_idx: jax.Array) -> None:
            """Stage and commit KDA state for row ``row_idx``."""
            source = sources[row_idx] if sources is not None else row_idx
            reads = (
                tpu.make_async_copy(
                    new_conv.at[pl.ds(source, 1)],
                    conv_buffer,
                    semaphores.at[0],
                ),
                tpu.make_async_copy(
                    new_state.at[pl.ds(source, 1)],
                    state_buffer,
                    semaphores.at[1],
                ),
            )
            for item in reads:
                item.start()
            for item in reads:
                item.wait()
            target = indices[row_idx]
            writes = (
                tpu.make_async_copy(
                    conv_buffer,
                    out_conv.at[pl.ds(target, 1)],
                    semaphores.at[0],
                ),
                tpu.make_async_copy(
                    state_buffer,
                    out_state.at[pl.ds(target, 1)],
                    semaphores.at[1],
                ),
            )
            for item in writes:
                item.start()
            for item in writes:
                item.wait()

    return pl.pallas_call(
        body,
        name="kda_selected_state_commit" if selected else "kda_state_commit",
        out_shape=(
            jax.ShapeDtypeStruct(conv_slots.shape, conv_slots.dtype),
            jax.ShapeDtypeStruct(recurrent_slots.shape, recurrent_slots.dtype),
        ),
        input_output_aliases={prefetch: 0, prefetch + 1: 1},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=prefetch,
            in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * 4,
            out_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * 2,
            scratch_shapes=(
                tpu.VMEM((1, *conv.shape[1:]), conv.dtype),
                tpu.VMEM((1, *recurrent.shape[1:]), recurrent.dtype),
                tpu.SemaphoreType.DMA((2,)),
            ),
        ),
    )(
        slots,
        *((source_rows,) if selected else ()),
        conv_slots,
        recurrent_slots,
        conv,
        recurrent,
    )


def decode_paged(
    weights: Mapping[str, jax.Array],
    conv_slots: jax.Array,
    recurrent_slots: jax.Array,
    pool: Sequence[jax.Array],
    tokens: jax.Array,
    positions: jax.Array,
    block_tables: jax.Array,
    slots: jax.Array,
    *,
    batch_size: int,
    dcp_pool_map: Sequence[int],
    dcp_owner_table: Sequence[int] | None = None,
    state_row: jax.Array | int | None = None,
    defer_kda_commit: bool = False,
    **decode_options: Unpack[_DecodeLocalOptions],
) -> (
    tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]
):
    """Run one decode step on this rank inside a ``tp`` ``shard_map``.

    ``conv_slots [S, 69, 3, 3, 3, 128]`` / ``recurrent_slots [S, 69, 3, 128,
    128]`` hold every request's KDA state; row ``b`` uses slot ``slots[b]``
    (padding rows the null slot, pages the null page). ``pool`` is the tuple
    of vLLM's unified pools (MLA layer ``m`` in ``pool[dcp_pool_map[m]]``),
    read in place; ``block_tables`` are the DCP tables. Returns ``(logits
    [batch, V/32], conv_slots, recurrent_slots, pool)`` with the slots and new
    latent rows committed (callers donate them); ``aux_hidden_layers`` adds
    ``aux [L, batch, hidden]`` after the logits.

    Speculative verify: the rows are one sequence's anchor and drafts;
    ``state_row`` (traced) picks the row whose input KDA state the sequence
    continues from. Every row's post-state and latent are committed; rejected
    drafts lie past the next step's pool limit. ``defer_kda_commit``
    (verifiers): return per-row post-states (under attention DP this group's
    rows, shard-major); the caller commits accepted rows.
    """
    if (
        tokens.shape != (batch_size,)
        or positions.shape != (batch_size,)
        or slots.shape != (batch_size,)
    ):
        msg = (
            f"tokens, positions and slots must have shape ({batch_size},), "
            f"got tokens={tokens.shape}, positions={positions.shape}, "
            f"slots={slots.shape}"
        )
        raise ValueError(msg)
    dp = decode_options.get("attention_dp", 1)
    group_rows: jax.Array | None = None
    state_index = slots
    if dp > 1:
        # This rank's DP group attends its rows of each local row owner's rows
        # (see decode_megakernel.stack); KDA state row k * G + p holds shard
        # k of group row p, at bank entry ``s + k * n`` for the row's slot s
        # of the group's n slots (G = batch / dp; pools.group_slots).
        owners = decode_options.get("sequence_parallel_size", 4)
        per_pair = batch_size // dp // owners
        data_group = mesh_axis_index(AXIS_TP) // (TP_SIZE // dp)
        row = jnp.arange(batch_size // dp)
        group_rows = (
            row // per_pair * (batch_size // owners)
            + data_group * per_pair
            + row % per_pair
        )
        group_slots = conv_slots.shape[0] // dp
        local = slots[group_rows]
        local = jnp.where(local == 0, 0, local - data_group * (group_slots - 1))
        state_index = jnp.concatenate([local + k * group_slots for k in range(dp)])
    # The stack reads the input KDA states from their slots itself where it
    # can (DP1 above four rows or a selected verify row; DP2 at B2..B128,
    # DP4 at B32..B128). DP indices already include each held head shard's
    # bank offset.
    in_place = (
        (
            (state_row is not None or batch_size > _DP1_IN_PLACE_STATE_MAX_GATHER_BATCH)
            and dp == 1
        )
        or (dp == DP2_GROUPS and batch_size in DP_VALID_BATCHES)
        or (dp == DP4_GROUPS and batch_size in _DP4_IN_PLACE_BATCHES)
    )
    if in_place:
        conv, recurrent = conv_slots, recurrent_slots
    else:
        conv, recurrent = conv_slots[state_index], recurrent_slots[state_index]
    pools = tuple(pool)
    pool_map_tuple = tuple(dcp_pool_map)
    owner_table_tuple = tuple(dcp_owner_table) if dcp_owner_table is not None else None
    decoded = decode_local(
        weights,
        (conv, recurrent, *pools),
        tokens,
        positions,
        batch_size=batch_size,
        block_tables=block_tables,
        dcp_pool_map=pool_map_tuple,
        dcp_owner_table=owner_table_tuple,
        state_row=state_row,
        state_slots=state_index if in_place else None,
        live_rows=slots != 0,
        **decode_options,
    )
    if len(decoded) == _DECODE_WITH_AUX_TUPLE_LEN:
        logits, aux, (conv, recurrent, new_keys, _new_values) = decoded
    else:
        aux = None
        logits, (conv, recurrent, new_keys, _new_values) = decoded
    # Attention DP returns shard-major rows; commit each to its shard-bank
    # index, preserving untouched slots and ordered null-slot writes.
    if defer_kda_commit:
        # Keep the input banks alive until acceptance. Return the native
        # post-states without touching persistent slots or gathering rows.
        conv_slots, recurrent_slots = conv, recurrent
    elif batch_size > _SCATTER_KDA_COMMIT_MAX_BATCH and dp in (
        1,
        DP2_GROUPS,
        DP4_GROUPS,
    ):
        conv_slots, recurrent_slots = commit_kda_states(
            conv_slots, recurrent_slots, conv, recurrent, state_index
        )
    else:
        conv_slots = conv_slots.at[state_index].set(conv)
        recurrent_slots = recurrent_slots.at[state_index].set(recurrent)
    latents = latent_rows_from_handoff(new_keys)
    if group_rows is not None:
        # The handoff holds the group's rows first; only their owners commit.
        latents, block_tables, positions = (
            latents[: group_rows.shape[0]],
            block_tables[group_rows],
            positions[group_rows],
        )
    committed_pool = commit_latent_rows(
        pools,
        latents,
        block_tables,
        positions,
        pool_map_tuple,
        owner_table=owner_table_tuple,
    )
    if aux is not None:
        return logits[0], aux, conv_slots, recurrent_slots, committed_pool
    return logits[0], conv_slots, recurrent_slots, committed_pool


__all__ = [
    "commit_kda_states",
    "commit_latent_rows",
    "decode_paged",
    "latent_rows_from_handoff",
]
