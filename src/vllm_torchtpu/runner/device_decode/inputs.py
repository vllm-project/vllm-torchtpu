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
"""Pure-decode forward inputs built on the device from the decode slots.

The counterpart of the `prepare_pos_seq_lens`,
`combine_sampled_and_draft_tokens` and `gather_block_tables` of vLLM's Model
Runner V2 (`vllm/v1/worker/gpu/input_batch.py`, `block_table.py`): one compiled
program gathers every batch row's slot through `idx_mapping` and returns what
the stock `_prepare_inputs` uploads for a pure-decode step, with the same
dtypes and shapes and the same values on live rows, so the precompiled backbone
and the attention metadata builders run unchanged.

For an M-RoPE model the positions are the decode case of MRV2's
`RopeState.prepare_positions` (`mm/rope.py`): each slot keeps its request's
delta, and a generated token's position is its index plus the delta in every
section.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch

from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilderContext,
    _staged_walk_products,
    block_table_walk_plan,
    split_staged_walk_products,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    _first_ge,
)
from vllm_torchtpu.runner.device_decode.slots import DecodeSlots, num_reqs_padding


@dataclass(frozen=True)
class DecodeShape:
    """The static sizes of a pure-decode step of `num_reqs` requests."""

    num_reqs: int
    # Logits rows, the stock request bucket.
    num_reqs_padded: int
    # The stock token bucket.
    num_tokens_padded: int
    # Rows of every attention metadata array (`num_reqs_max_model_len`).
    target_rows: int

    @property
    def program_key(self) -> tuple:
        return (
            "decode_prepare",
            self.num_tokens_padded,
            self.num_reqs_padded,
            self.target_rows,
        )

    @staticmethod
    def for_batch(runner: Any, num_reqs: int) -> DecodeShape:
        return DecodeShape(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padding(runner, num_reqs),
            num_tokens_padded=_first_ge(runner.num_tokens_paddings, num_reqs),
            target_rows=runner.num_reqs_max_model_len,
        )

    @staticmethod
    def all_decode_shapes(runner: Any) -> list[DecodeShape]:
        """One shape per distinct program, for every single-chunk batch."""
        shapes: dict[tuple, DecodeShape] = {}
        for num_reqs in range(1, runner.num_reqs_max_model_len + 1):
            shape = DecodeShape.for_batch(runner, num_reqs)
            shapes.setdefault(shape.program_key, shape)
        return list(shapes.values())


@dataclass(frozen=True)
class WalkGeometry:
    """The serving block-table walk of a pure-decode step (see
    `block_table_walk_plan`), as the static arguments of the prepare
    program."""

    gids: tuple[int, ...]
    widths: tuple[int, ...]
    table_geom: tuple[tuple[int, int, int], ...]
    mamba_geom: tuple[tuple[int, int, int, int | None], ...]
    mamba_gids: tuple[int, ...]
    ckpt_window: int
    compact_mamba: bool
    single_table: bool

    @staticmethod
    def from_runner(runner: Any) -> WalkGeometry:
        plan = block_table_walk_plan(
            runner,
            True,
            runner._mamba_ckpt_window,
            runner.num_reqs_max_model_len,
        )
        if plan is None:
            raise RuntimeError(
                "Device decode needs the KV cache groups; build the walk "
                "geometry after initialize_kv_cache."
            )
        return WalkGeometry(
            gids=plan.gids,
            widths=tuple(width for _, _, width in plan.entries),
            table_geom=plan.table_geom,
            mamba_geom=plan.mamba_geom,
            mamba_gids=tuple(plan.entries[i][0] for i in plan.mamba_entry_indices),
            ckpt_window=runner._mamba_ckpt_window,
            compact_mamba=runner._has_mamba_state,
            single_table=plan.single_table,
        )


@dataclass
class DecodeInputs:
    input_ids: torch.Tensor
    # [num_tokens_padded], or [mrope_rows, num_tokens_padded] with M-RoPE.
    positions: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    logits_indices: torch.Tensor
    request_distribution: torch.Tensor
    # gid -> flat [target_rows * width] table, as `stage_block_table_uploads`.
    staged_block_tables: dict[int, torch.Tensor]
    # gid -> (state, ckpt | None, prefill_ckpt | None); unified pool only.
    staged_mamba_products: dict[int, tuple] | None
    # Compact-mamba slot ids, [target_rows].
    mamba_state_indices: torch.Tensor | None


def _prepare_decode_impl(
    idx_mapping: torch.Tensor,
    num_computed: torch.Tensor,
    last_token: torch.Tensor,
    mamba_slot: torch.Tensor | None,
    prefill_ckpt_offsets: torch.Tensor | None,
    block_tables: tuple[torch.Tensor, ...],
    num_tokens_padded: int,
    num_reqs_padded: int,
    target_rows: int,
    null_slot: int,
    widths: tuple[int, ...],
    table_geom: tuple[tuple[int, int, int], ...],
    mamba_geom: tuple[tuple[int, int, int, int | None], ...],
    ckpt_window: int,
    single_table: bool,
    mrope_delta: torch.Tensor | None = None,
    mrope_rows: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Returns input_ids, positions, seq_lens, query_start_loc,
    logits_indices, request_distribution, then the compact-mamba slot ids if
    `mamba_slot` is given, then the flat table (`single_table`) or the
    `_staged_walk_products` outputs. Rows past the batch are zero, except
    the query_start_loc tail and logits_indices, which repeat the last live
    entry as the stock path's do. With `mrope_delta`, the positions have
    `mrope_rows` rows."""
    device = idx_mapping.device
    rows = idx_mapping[:target_rows]
    valid = rows != null_slot
    rows = rows.long()
    num_live = valid.sum(dtype=torch.int32)
    tokens = torch.arange(num_tokens_padded, device=device)
    token_rows = rows[tokens.clamp(max=target_rows - 1)]
    token_live = tokens < num_live
    input_ids = torch.where(token_live, last_token[token_rows], 0)
    positions = num_computed[token_rows]
    if mrope_delta is not None:
        # A generated token's M-RoPE position is its index plus its request's
        # delta in every section, as the stock `_calc_mrope_positions` stages
        # it (vLLM's `MRotaryEmbedding.get_next_input_positions_tensor`).
        positions = positions + mrope_delta[token_rows]
    positions = torch.where(token_live, positions, 0)
    if mrope_delta is not None:
        positions = positions.repeat(mrope_rows, 1)
    seq_lens = torch.where(valid, num_computed[rows] + 1, 0)
    query_start_loc = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(valid, 0, dtype=torch.int32),
        )
    )
    logits_indices = torch.minimum(
        torch.arange(num_reqs_padded, dtype=torch.int32, device=device),
        (num_live - 1).clamp(min=0),
    )
    request_distribution = torch.stack((num_live, num_live, num_live))
    outs = [
        input_ids,
        positions,
        seq_lens,
        query_start_loc,
        logits_indices,
        request_distribution,
    ]
    if mamba_slot is not None:
        outs.append(torch.where(valid, mamba_slot[rows], 0))
    tables = [
        torch.where(valid[:, None], table.index_select(0, rows)[:, :width], 0).reshape(
            -1
        )
        for table, width in zip(block_tables, widths)
    ]
    flat = tables[0] if len(tables) == 1 else torch.cat(tables)
    if single_table:
        outs.append(flat)
    else:
        outs.extend(
            _staged_walk_products(
                flat,
                seq_lens,
                table_geom,
                mamba_geom,
                ckpt_window,
                prefill_ckpt_offsets,
            )
        )
    return tuple(outs)


_prepare_decode_compiled = torch.compile(
    _prepare_decode_impl, backend="tpu", fullgraph=True, dynamic=False
)


def prepare_decode_inputs(
    slots: DecodeSlots,
    idx_mapping: torch.Tensor,
    shape: DecodeShape,
    geometry: WalkGeometry,
    prefill_ckpt_offsets: torch.Tensor | None,
) -> DecodeInputs:
    """Builds a pure-decode step's inputs from the slots, one program per
    (shape, geometry). `idx_mapping` is the int32 [max_num_reqs] device copy
    of `DecodeSlots.idx_mapping_host`; `prefill_ckpt_offsets` the stock
    [target_rows] offsets when GDN prefill checkpoints are on."""
    assert shape.num_reqs <= shape.target_rows, shape
    prepare = (
        _prepare_decode_compiled
        if idx_mapping.device.type == "tpu"
        else _prepare_decode_impl
    )
    with torch.no_grad():
        outs = prepare(
            idx_mapping,
            slots.num_computed,
            slots.last_token,
            slots.mamba_slot if geometry.compact_mamba else None,
            prefill_ckpt_offsets,
            tuple(slots.block_tables[gid] for gid in geometry.gids),
            shape.num_tokens_padded,
            shape.num_reqs_padded,
            shape.target_rows,
            slots.config.null_slot,
            geometry.widths,
            geometry.table_geom,
            geometry.mamba_geom,
            geometry.ckpt_window,
            geometry.single_table,
            slots.mrope_delta,
            slots.config.mrope_rows,
        )
    rest = list(outs[6:])
    mamba_state_indices = rest.pop(0) if geometry.compact_mamba else None
    if geometry.single_table:
        staged_block_tables = {geometry.gids[0]: rest[0]}
        staged_mamba_products = None
    else:
        staged_block_tables, staged_mamba_products = split_staged_walk_products(
            geometry.gids,
            geometry.mamba_geom,
            geometry.ckpt_window,
            prefill_ckpt_offsets is not None,
            tuple(rest),
        )
    return DecodeInputs(
        input_ids=outs[0],
        positions=outs[1],
        seq_lens=outs[2],
        query_start_loc=outs[3],
        logits_indices=outs[4],
        request_distribution=outs[5],
        staged_block_tables=staged_block_tables,
        staged_mamba_products=staged_mamba_products,
        mamba_state_indices=mamba_state_indices,
    )


def build_decode_attention_metadata(
    runner: Any,
    inputs: DecodeInputs,
    shape: DecodeShape,
    prefill_ckpt_offsets: torch.Tensor | None,
) -> dict[str, AttentionMetadata]:
    """The per-layer attention metadata of a device-built decode step,
    through the stock builders with the staged tables pre-filled. Also sets
    `runner.input_ids` / `runner.position_ids`, which the builders and the
    forward read."""
    # The prepare program stages max-model-len tables and one distribution.
    assert runner.most_model_len is None
    assert runner.mamba_slot_read_offsets is None
    runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
        num_reqs=shape.num_reqs,
        start_index=0,
        use_max_model_len=True,
        seq_lens=inputs.seq_lens,
        query_start_loc=inputs.query_start_loc,
        request_distribution=inputs.request_distribution,
        mamba_state_indices=inputs.mamba_state_indices,
        mamba_ckpt_window=runner._mamba_ckpt_window,
        mamba_prefill_ckpt_offsets=prefill_ckpt_offsets,
        sequence_layout_descriptor=DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
        staged_block_tables=inputs.staged_block_tables,
        staged_mamba_products=inputs.staged_mamba_products,
    )
    runner.input_ids = inputs.input_ids
    runner.position_ids = inputs.positions
    per_layer_attn_metadata, _ = runner._build_attention_metadata(
        num_tokens=shape.num_reqs,
        num_reqs=shape.target_rows,
        max_query_len=1,
        num_tokens_padded=shape.num_tokens_padded,
        num_reqs_padded=shape.target_rows,
        slot_mappings=runner.empty_slot_mappings,
    )
    return per_layer_attn_metadata


def precompile_prepare(
    runner: Any,
    slots: DecodeSlots,
    geometry: WalkGeometry,
    shapes: Iterable[DecodeShape],
) -> list[tuple]:
    """Runs the prepare program once per shape on an all-null batch.
    Returns the program keys."""
    for gid, width in zip(geometry.gids, geometry.widths):
        assert width <= slots.block_tables[gid].shape[1], (gid, width)
    idx_mapping = torch.full(
        (slots.config.max_num_reqs,), slots.config.null_slot, dtype=torch.int32
    ).to(runner.device)
    keys = []
    for shape in shapes:
        prefill_ckpt_offsets = (
            torch.zeros(shape.target_rows, dtype=torch.int32).to(runner.device)
            if runner._prefill_ckpt_units is not None
            else None
        )
        prepare_decode_inputs(slots, idx_mapping, shape, geometry, prefill_ckpt_offsets)
        keys.append(shape.program_key)
    return keys
