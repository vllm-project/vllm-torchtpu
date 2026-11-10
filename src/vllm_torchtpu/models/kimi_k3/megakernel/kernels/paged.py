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

import dataclasses
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from ..config import active as _config
from . import mla_dcp
from .decode_megakernel import (
    MLA_LAYERS,
    decode_local,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_IN_PLACE_DP1_MIN_BATCH = 4
_DP2_GROUPS = 2
_DP4_GROUPS = 4
_FUSED_KDA_COMMIT_MIN_BATCH = 16
_AUX_RESULT_LEN = 3


class _CommitLatentOptions(TypedDict, total=False):
    """Keyword options for `commit_latent_rows`."""

    group: int
    owner_table: Sequence[int] | jax.Array | None
    global_pages: bool | None


class _CommitKdaStatesOptions(TypedDict, total=False):
    """Keyword options for `_commit_kda_states`."""

    source_rows: jax.Array | None


@dataclasses.dataclass(frozen=True, slots=True)
class PagedKdaStepConfig:
    """Per-step batch, DP, SP, and speculative verify row configuration."""

    batch_size: int
    dp: int
    owners: int
    state_row: jax.Array | int | None = None


class _DecodeLocalForwardOptions(TypedDict, total=False):
    """Keyword options forwarded from `decode_paged` to `decode_local`."""

    layers: int
    eps: float
    residual_reduction_order: Literal["reference", "relaxed", "row"]
    mla_cache_batch_tile_size: int | None
    alias_expert_scales: bool
    attention_token_scatter: bool
    moe_routed_reduce_scatter: bool
    moe_token_scatter_output: bool
    sequence_parallel_pre_attention: bool
    batched_decoder_boundaries: bool | None
    sequence_rows: int
    aux_hidden_layers: tuple[int, ...]
    aux_capture: Literal["attention", "layer", "mixture_input"]
    attention_dp: int
    attention_projection_schedule: Literal["default", "mla_up_double"]
    mla_commit_tile_size: int | None
    sequence_parallel_size: int


class _DecodePagedOptions(_DecodeLocalForwardOptions, total=False):
    """Keyword options accepted by `decode_paged`."""

    batch_size: int
    dcp_pool_map: Sequence[int]
    dcp_owner_table: Sequence[int] | None
    state_row: jax.Array | int | None
    defer_kda_commit: bool


def latent_rows_from_handoff(new_keys: jax.Array) -> jax.Array:
    """Extract `[batch, M, 640]` latent rows from a DCP decoder's handoff.

    Packed in new-key rows `0..2`; see `decode_megakernel.mla`.
    """
    return jnp.concatenate(
        (
            new_keys[:, :, 0],
            new_keys[:, :, 1],
            new_keys[:, :, 2, :128],
        ),
        axis=-1,
    )


def _normalized_pool_block(
    pool_size: int,
    block: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Wrap negative block index and check `[0, pool_size)` bounds."""
    target = jnp.where(block < 0, block + pool_size, block)
    return target, (target >= 0) & (target < pool_size)


def _fused_commit_grid_and_shapes(
    pools: Sequence[jax.Array],
    layers: int,
    packing: int,
) -> tuple[
    tuple[jax.ShapeDtypeStruct, ...],
    tpu.PrefetchScalarGridSpec,
    dict[int, int],
]:
    """Build output shapes, grid spec, and alias map for fused pool commit."""
    count = len(pools)
    dtype = pools[0].dtype
    out_shape = tuple(jax.ShapeDtypeStruct(pool.shape, pool.dtype) for pool in pools)
    grid_spec = tpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        in_specs=(
            pl.BlockSpec(memory_space=tpu.VMEM),
            *(pl.BlockSpec(memory_space=tpu.HBM) for _ in range(count)),
        ),
        out_specs=tuple(pl.BlockSpec(memory_space=tpu.HBM) for _ in range(count)),
        scratch_shapes=(
            tpu.VMEM((layers, 8, packing, 640), dtype),
            tpu.SemaphoreType.DMA((layers,)),
        ),
    )
    aliases = {2 + idx: idx for idx in range(count)}
    return out_shape, grid_spec, aliases


def _commit_latent_rows_fused(
    pools: Sequence[jax.Array],
    rows: jax.Array,
    metadata: jax.Array,
    pool_map: Sequence[int],
) -> tuple[jax.Array, ...]:
    """Alias all unified pools in one non-communicating cache commit.

    Overlap the layers' native-tile reads. Retire all writes before
    advancing to the next token so rows sharing a tile preserve each other.
    """
    batch, layers, _ = rows.shape
    packing, dtype = pools[0].shape[-2], pools[0].dtype
    page_tokens = pools[0].shape[-3] * packing
    tile_tokens = 8 * packing
    count = len(pools)

    def kernel(
        meta: jax.Array,
        values: jax.Array,
        *refs: jax.Array,
    ) -> None:
        """DMA-read, update, and DMA-write owned latent tiles across pools."""
        outputs = refs[count : 2 * count]
        tiles, semaphores = refs[-2:]

        @pl.loop(0, batch)
        def commit_row(row: jax.Array) -> None:
            """Commit all MLA layers for batch row `row` when owned."""

            @pl.when(meta[2, row] != 0)
            def owned_row() -> None:
                """Stage, patch, and retire the tile for an owned token position."""
                position, block = meta[0, row], meta[1, row]
                first = pl.multiple_of((position % page_tokens // tile_tokens) * 8, 8)

                for layer, index in enumerate(pool_map):
                    target, valid = _normalized_pool_block(
                        outputs[index].shape[0], block
                    )

                    @pl.when(valid)
                    def read(
                        *,
                        layer: int = layer,
                        index: int = index,
                        target: jax.Array = target,
                    ) -> None:
                        """Start async HBM->VMEM read of the destination tile."""
                        tpu.make_async_copy(
                            outputs[index].at[target, pl.ds(first, 8)],
                            tiles.at[layer],
                            semaphores.at[layer],
                        ).start()

                for layer, index in enumerate(pool_map):
                    target, valid = _normalized_pool_block(
                        outputs[index].shape[0], block
                    )

                    @pl.when(valid)
                    def update(
                        *,
                        layer: int = layer,
                        index: int = index,
                        target: jax.Array = target,
                    ) -> None:
                        """Wait for tile read, splice row latent, and start write."""
                        copy = tpu.make_async_copy(
                            outputs[index].at[target, pl.ds(first, 8)],
                            tiles.at[layer],
                            semaphores.at[layer],
                        )
                        copy.wait()
                        slot_offset = packing * jax.lax.broadcasted_iota(
                            jnp.int32, (8, packing, 640), 0
                        ) + jax.lax.broadcasted_iota(jnp.int32, (8, packing, 640), 1)
                        tiles[layer] = jnp.where(
                            slot_offset == position % tile_tokens,
                            values[row, layer][None, None, :],
                            tiles[layer],
                        )
                        tpu.make_async_copy(
                            tiles.at[layer],
                            outputs[index].at[target, pl.ds(first, 8)],
                            semaphores.at[layer],
                        ).start()

                for layer, index in enumerate(pool_map):
                    target, valid = _normalized_pool_block(
                        outputs[index].shape[0], block
                    )

                    @pl.when(valid)
                    def retire(
                        *,
                        layer: int = layer,
                        index: int = index,
                        target: jax.Array = target,
                    ) -> None:
                        """Wait for async VMEM->HBM write completion."""
                        tpu.make_async_copy(
                            tiles.at[layer],
                            outputs[index].at[target, pl.ds(first, 8)],
                            semaphores.at[layer],
                        ).wait()

    out_shape, grid_spec, aliases = _fused_commit_grid_and_shapes(
        pools, layers, packing
    )
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
    **latent_commit_kwargs: Unpack[_CommitLatentOptions],
) -> tuple[jax.Array, ...]:
    """Write each row's new latent into vLLM's unified pools on its owner rank.

    Inside `shard_map` over `tp`, MLA layer `m`'s row at position `p` lands in
    `pools[pool_map[m]]` at token `p % page_tokens` of block `block_tables[b, e]`,
    `e = p // page_tokens` (global tables) or `p // (page_tokens * group)`
    (local tables). `owner_table` matches the decoder's `dcp_owner_table`
    (default the mesh position), and `global_pages` defaults to
    `mla_dcp.GLOBAL_PAGES`.
    """
    group = latent_commit_kwargs.get("group", mla_dcp.GROUP)
    owner_table = latent_commit_kwargs.get("owner_table")
    global_pages_opt = latent_commit_kwargs.get("global_pages")
    global_pages = (
        mla_dcp.GLOBAL_PAGES if global_pages_opt is None else bool(global_pages_opt)
    )
    member = mla_dcp.owner_index(owner_table) % group
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
        and rows.shape[0] in (1, 2, 4, 8, 16, 24, 32, 48, 64)
        and rows.shape[1:] == (len(pool_map), 640)
        and 0 < len(pool_map) <= MLA_LAYERS
        and len(set(pool_map)) == len(pool_map)
        and all(0 <= index < len(pool_list) for index in pool_map)
        and all(
            pool.shape[0] > 0
            and pool.shape[1:] == (128, packing, 640)
            and pool.dtype == pool_list[0].dtype
            for pool in pool_list
        )
    ):
        metadata = jnp.stack(
            (positions[: rows.shape[0]], blocks, owned.astype(jnp.int32))
        )
        return _commit_latent_rows_fused(pool_list, rows, metadata, pool_map)
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


def _commit_kda_states(
    conv_slots: jax.Array,
    recurrent_slots: jax.Array,
    conv: jax.Array,
    recurrent: jax.Array,
    slots: jax.Array,
    **kda_commit_kwargs: Unpack[_CommitKdaStatesOptions],
) -> tuple[jax.Array, jax.Array]:
    """Copy complete KDA rows back without XLA's per-row scatter loop.

    Row writes finish in input order, including repeated null-slot indices.
    The banks alias their outputs so untouched slots retain their bytes.
    With ``source_rows``, DMA only the selected verifier post-states directly
    from their HBM buffer; do not gather a second intermediate with XLA.
    """
    source_rows = kda_commit_kwargs.get("source_rows")
    batch = slots.shape[0]
    selected = source_rows is not None
    prefetch = 2 if selected else 1

    def body(indices: jax.Array, *args: jax.Array) -> None:
        """DMA-copy each batch row's KDA conv and recurrent states into `slots`."""
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
            """Stage and commit KDA state for row `row_idx`."""
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


def _unpack_paged_step_inputs(
    pool: Sequence[jax.Array],
    step_tensors: tuple[jax.Array, ...],
    batch_size: int,
) -> tuple[
    tuple[jax.Array, ...],
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Validate and unpack `(pool, tokens, positions, block_tables, slots)`."""
    tokens_arg, positions_arg, block_tables_arg, slots_arg = step_tensors
    if (
        tokens_arg.shape != (batch_size,)
        or positions_arg.shape != (batch_size,)
        or slots_arg.shape != (batch_size,)
    ):
        msg = "tokens, positions and slots must have one entry per row"
        raise ValueError(msg)
    return (
        tuple(pool),
        tokens_arg,
        positions_arg,
        block_tables_arg,
        slots_arg,
    )


def prepare_paged_kda_inputs(
    conv_slots: jax.Array,
    recurrent_slots: jax.Array,
    slots: jax.Array,
    step_cfg: PagedKdaStepConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None, bool]:
    """Resolve KDA state indices, DP group rows, and in-place bank selection."""
    batch_size = step_cfg.batch_size
    dp = step_cfg.dp
    owners = step_cfg.owners
    state_row = step_cfg.state_row
    group_rows: jax.Array | None = None
    state_index = slots
    if dp > 1:
        # This rank's DP group attends its rows of each local row owner's rows
        # (see decode_megakernel.stack); KDA state row k * G + p holds shard
        # k of group row p, at bank entry ``s + k * n`` for the row's slot s
        # of the group's n slots (G = batch / dp; pools.group_slots).
        per_pair = batch_size // dp // owners
        data_group = jax.lax.axis_index("tp") // (32 // dp)
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
        ((state_row is not None or batch_size > _IN_PLACE_DP1_MIN_BATCH) and dp == 1)
        or (dp == _DP2_GROUPS and batch_size in (2, 4, 8, 16, 32, 64, 96, 128))
        or (dp == _DP4_GROUPS and batch_size in (32, 64, 96, 128))
    )
    if in_place:
        conv, recurrent = conv_slots, recurrent_slots
    else:
        conv, recurrent = conv_slots[state_index], recurrent_slots[state_index]
    return conv, recurrent, state_index, group_rows, in_place


def decode_paged(
    weights: Mapping[str, jax.Array],
    conv_slots: jax.Array,
    recurrent_slots: jax.Array,
    pool: Sequence[jax.Array],
    *step_tensors: jax.Array,
    **decode_options: Unpack[_DecodePagedOptions],
) -> (
    tuple[jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]
    | tuple[jax.Array, jax.Array, jax.Array, jax.Array, tuple[jax.Array, ...]]
):
    """Run one decode step on this rank inside a `tp` `shard_map`.

    `conv_slots [S, 69, 3, 3, 3, 128]` / `recurrent_slots [S, 69, 3, 128,
    128]` hold every request's KDA state; row `b` uses slot `slots[b]`
    (padding rows the null slot, pages the null page). `pool` is the tuple
    of vLLM's unified pools (MLA layer `m` in `pool[dcp_pool_map[m]]`),
    read in place; `block_tables` are the DCP tables. Returns `(logits
    [batch, V/32], conv_slots, recurrent_slots, pool)` with the slots and new
    latent rows committed (callers donate them); `aux_hidden_layers` adds
    `aux [L, batch, hidden]` after the logits.

    Speculative verify: the rows are one sequence's anchor and drafts;
    `state_row` (traced) picks the row whose input KDA state the sequence
    continues from. Every row's post-state and latent are committed; rejected
    drafts lie past the next step's pool limit. `defer_kda_commit`
    (verifiers): return per-row post-states (under attention DP this group's
    rows, shard-major); the caller commits accepted rows.
    """
    batch_size = decode_options["batch_size"]
    dcp_pool_map = tuple(decode_options["dcp_pool_map"])
    dcp_owner_table = decode_options.get("dcp_owner_table")
    owner_tuple: tuple[int, ...] | None = (
        tuple(int(item) for item in dcp_owner_table)
        if dcp_owner_table is not None
        else None
    )
    state_row = decode_options.get("state_row")
    defer_kda_commit = bool(decode_options.get("defer_kda_commit", False))
    dp = decode_options.get("attention_dp", 1)
    owners = decode_options.get("sequence_parallel_size", 4)
    forward_opts = cast(
        "_DecodeLocalForwardOptions",
        {
            key: val
            for key, val in decode_options.items()
            if key
            not in {
                "batch_size",
                "dcp_pool_map",
                "dcp_owner_table",
                "state_row",
                "defer_kda_commit",
            }
        },
    )

    pools, tokens, positions, block_tables, slots = _unpack_paged_step_inputs(
        pool, step_tensors, batch_size
    )
    conv, recurrent, state_index, group_rows, in_place = prepare_paged_kda_inputs(
        conv_slots,
        recurrent_slots,
        slots,
        PagedKdaStepConfig(
            batch_size=batch_size,
            dp=dp,
            owners=owners,
            state_row=state_row,
        ),
    )
    result = decode_local(
        dict(weights),
        (conv, recurrent, *pools),
        tokens,
        positions,
        batch_size=batch_size,
        block_tables=block_tables,
        dcp_pool_map=dcp_pool_map,
        dcp_owner_table=owner_tuple,
        state_row=state_row,
        state_slots=state_index if in_place else None,
        live_rows=slots != 0,
        **forward_opts,
    )
    logits = result[0]
    conv, recurrent, new_keys, _new_values = result[-1]
    aux = result[1] if len(result) == _AUX_RESULT_LEN else None
    if defer_kda_commit:
        conv_slots, recurrent_slots = conv, recurrent
    elif batch_size > _FUSED_KDA_COMMIT_MIN_BATCH and dp in (1, 2, 4):
        conv_slots, recurrent_slots = _commit_kda_states(
            conv_slots, recurrent_slots, conv, recurrent, state_index
        )
    else:
        conv_slots = conv_slots.at[state_index].set(conv)
        recurrent_slots = recurrent_slots.at[state_index].set(recurrent)
    latents = latent_rows_from_handoff(new_keys)
    if group_rows is not None:
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
        dcp_pool_map,
        owner_table=dcp_owner_table,
    )
    if aux is not None:
        return logits[0], aux, conv_slots, recurrent_slots, committed_pool
    return logits[0], conv_slots, recurrent_slots, committed_pool
