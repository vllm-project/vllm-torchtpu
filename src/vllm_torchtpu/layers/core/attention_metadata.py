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
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import torch
from vllm.distributed import get_pcp_group
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder as BaseAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_torchtpu import envs
from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    DEFAULT_SEQUENCE_LAYOUT_PROTOCOL,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
)


def attention_block_table_width(
    block_table: Any,
    spec: KVCacheSpec,
    vllm_config: Any,
    model_len: int | None = None,
) -> int:
    """Match allocated, rank-local kernel page ordinals, including block splits."""
    if model_len is None:
        return block_table.max_num_blocks_per_req
    pages = spec.max_num_blocks_per_req(vllm_config, model_len)
    return min(
        pages * getattr(block_table, "blocks_per_kv_block", 1),
        block_table.max_num_blocks_per_req,
    )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "mamba_state_indices",
        "mamba_slot_read_offsets",
        "mamba_ckpt_indices",
        "mamba_request_distribution",
    ],
    meta_fields=[
        "sequence_layout_kind",
        "sequence_layout_protocol",
        "sequence_layout_version",
    ],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu"],
)
@dataclass
class AttentionMetadata:
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array | torch.Tensor | None
    # (max_num_seqs * max_num_blocks_per_req,)
    block_tables: jax.Array | torch.Tensor | None = None
    # (max_num_seqs,)
    seq_lens: jax.Array | torch.Tensor | None = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array | torch.Tensor | None = None
    # (3,)
    request_distribution: jax.Array | torch.Tensor | None = None
    # (max_num_seqs,) int32 - physical slot id in the mamba kv-cache for the
    # request currently in each persistent-batch position. Compact-mamba uses
    # an explicit slot from its independent pool; unified align mode derives
    # the slot from the current entry in the device block table. None for
    # attention groups, non-mamba models, and Mamba modes whose native fallback
    # addresses block_tables[:, 0].
    mamba_state_indices: jax.Array | torch.Tensor | None = None
    # (mamba_num_blocks,) int32 — per-*slot* read offset for speculative
    # decoding with mamba layers. `mamba_slot_read_offsets[base_slot]` is
    # `num_accepted - 1` from the request's most recent verify step: the GDN
    # kernel reads the request's initial state from `base_slot + offset`
    # (the checkpoint of the last accepted token) and writes fresh
    # checkpoints starting at `base_slot`. Indexed by physical slot (not
    # batch position) so the value survives requests being rescheduled or
    # condensed. Updated on device after each sampling step; None unless the
    # model has mamba layers *and* speculative decoding is enabled.
    mamba_slot_read_offsets: jax.Array | torch.Tensor | None = None
    # `(num_reqs, num_spec + 1)` source block per checkpoint. Unified pool
    # with speculative decoding only; None elsewhere, where the sole
    # checkpoint is the request's own state block.
    mamba_ckpt_indices: jax.Array | torch.Tensor | None = None
    # (3,) int32 — GDN-specific request distribution, same format as
    # `request_distribution` but with the first segment covering all
    # *windowed* sequences (plain decodes and speculative verify windows of
    # up to num_spec + 1 tokens) instead of only 1-token decodes. The
    # persistent batch is ordered [decode][verify][prefill/mixed] so both
    # segmentations hold at once: ragged paged attention keeps its 1-token
    # decode front segment while the GDN kernel runs its windowed mode over
    # the first two groups. None unless the model has mamba layers and spec
    # decoding is enabled.
    mamba_request_distribution: jax.Array | torch.Tensor | None = None
    sequence_layout_kind: str = SequenceLayoutKind.ALL.value
    sequence_layout_protocol: str = DEFAULT_SEQUENCE_LAYOUT_PROTOCOL
    sequence_layout_version: int = 1

    query_start_loc_cpu: Any = field(init=False)
    seq_lens_cpu: Any = field(init=False)


@dataclass
class AttentionMetadataBuilderContext:
    """Per-call inputs the TPU metadata builder reads from the runner.

    Stashed on TPUModelRunner before invoking _build_attention_metadata so
    builder.build() can pick them up. Only fields TPU populates differently from
    the parent runner (seq_lens / query_start_loc) live in TPU's own _cpu
    staging tensors, not parent's CpuGpuBuffers) or that have no
    `CommonAttentionMetadata` equivalent (num_reqs, use_max_model_len,
    start_index, request_distribution, position_ids_override) live here.
    """

    num_reqs: int
    start_index: int
    use_max_model_len: bool
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    request_distribution: torch.Tensor
    position_ids_override: torch.Tensor | None = None
    # Compact-mamba per-request recurrent-slot ids (device int32, length =
    # target_num_reqs). Only the mamba group's builder reads it; None when the
    # model has no mamba layers. See AttentionMetadata.mamba_state_indices.
    mamba_state_indices: torch.Tensor | None = None
    # Spec decode with mamba layers only; see the AttentionMetadata fields of
    # the same names. Only the mamba group's builder reads them.
    mamba_slot_read_offsets: torch.Tensor | None = None
    # State checkpoints per request on the unified pool (num_spec + 1, each
    # its own block). 1 means a single state per request, so the builder
    # emits no `mamba_ckpt_indices`.
    mamba_ckpt_window: int = 1
    mamba_request_distribution: torch.Tensor | None = None
    # Unified pool + spec decode only: the runner seeds this with an empty
    # list and each mamba group's builder appends the per-request state
    # block ids it derived from its block table, so the post-sampling
    # read-offset scatter can address every group's state block (groups
    # allocate distinct blocks, unlike the compact pool's shared slots).
    unified_mamba_state_indices: list[torch.Tensor] | None = None
    sequence_layout_descriptor: SequenceLayoutDescriptor = (
        DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR
    )
    # Per-chunk cache of the shared mamba row-offset plan, keyed by
    # (target_block_size, target_num_blocks) -> (state_index, ckpt_index,
    # ckpt_in_row). Valid only because every context is constructed fresh
    # per chunk: the plan derives from this chunk's seq_lens/ckpt_window,
    # which the key deliberately omits. Do not reuse a context across
    # chunks.
    mamba_row_plans: dict[
        tuple[int, int], tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
    ] = field(default_factory=dict)
    # Per-chunk batched block-table upload: kv_cache_group_id -> flat device
    # view into one staged H2D transfer (`stage_block_table_uploads`); None
    # means each build() uploads its own table (standalone builds, tests).
    staged_block_tables: dict | None = None
    # Per-chunk batched unified-pool mamba indices: kv_cache_group_id ->
    # (state_indices, ckpt_indices | None) from the single
    # `_staged_walk_products_compiled` program; None -> per-group fallback.
    staged_mamba_products: dict | None = None


def _staged_walk_products(
    staged_dev: torch.Tensor,
    seq_lens: torch.Tensor,
    table_geom: tuple,
    mamba_geom: tuple,
    ckpt_window: int,
    include_tables: bool = True,
    class_gather: bool = False,
) -> tuple:
    """Splits the staged flat block-table buffer into per-group tables and
    derives unified-pool mamba state and checkpoint indices. Without
    `include_tables` only the mamba indices are returned (the caller still
    holds the tables of an identical earlier upload)."""
    outs = []
    if include_tables:
        for off, n, t in table_geom:
            outs.append(staged_dev[off : off + n * t].clone())
    if mamba_geom and not class_gather:
        # One gather per mamba group (VLLM_TPU_STAGED_WALK_CLASS_GATHER off).
        states_g = []
        ckpts_g = []
        active = seq_lens > 0
        for ti, t, bs in mamba_geom:
            off, n, _ = table_geom[ti]
            tbl = staged_dev[off : off + n * t].view(n, t)
            state_offsets = torch.clamp((seq_lens - 1) // bs, min=0, max=t - 1).to(
                torch.int64
            )
            gathered = torch.gather(tbl, 1, state_offsets.unsqueeze(1)).squeeze(1)
            states_g.append(
                torch.where(active, gathered, torch.zeros_like(gathered))
            )
            if ckpt_window > 1:
                ckpt_offsets = state_offsets.unsqueeze(1) + torch.arange(
                    ckpt_window, device=seq_lens.device, dtype=torch.int64
                ).unsqueeze(0)
                in_row = (ckpt_offsets < t) & active.unsqueeze(1)
                safe = torch.where(in_row, ckpt_offsets, torch.zeros_like(ckpt_offsets))
                ckpts_g.append(torch.gather(tbl, 1, safe) * in_row)
        return tuple(outs) + tuple(states_g) + tuple(ckpts_g)
    states: list = [None] * len(mamba_geom)
    ckpts: list = [None] * len(mamba_geom)
    if mamba_geom:
        active = seq_lens > 0
        device = seq_lens.device
        # Groups sharing (num_blocks, block_size) share every row offset, so
        # each such class is one gather over the flat staged buffer instead
        # of one small gather per group (~1.5 us each, ~70 per step).
        classes: dict[tuple[int, int], list[int]] = {}
        for j, (_, t, bs) in enumerate(mamba_geom):
            classes.setdefault((t, bs), []).append(j)
        for (t, bs), members in classes.items():
            n = table_geom[mamba_geom[members[0]][0]][1]
            # Flat offset of row r of member g: table offset + r * t.
            bases = torch.tensor(
                [table_geom[mamba_geom[j][0]][0] for j in members],
                device=device,
                dtype=torch.int64,
            ).unsqueeze(1) + (
                torch.arange(n, device=device, dtype=torch.int64) * t
            ).unsqueeze(0)
            state_offsets = torch.clamp((seq_lens - 1) // bs, min=0, max=t - 1).to(
                torch.int64
            )
            gathered = staged_dev[bases + state_offsets.unsqueeze(0)]
            gathered = torch.where(
                active.unsqueeze(0), gathered, torch.zeros_like(gathered)
            )
            if ckpt_window > 1:
                ckpt_offsets = state_offsets.unsqueeze(1) + torch.arange(
                    ckpt_window, device=device, dtype=torch.int64
                ).unsqueeze(0)
                in_row = (ckpt_offsets < t) & active.unsqueeze(1)
                safe = torch.where(in_row, ckpt_offsets, torch.zeros_like(ckpt_offsets))
                ck = staged_dev[bases.unsqueeze(2) + safe.unsqueeze(0)] * in_row
            # Clone each slice: a sliced graph output is a view into the
            # class's [G, ...] buffer, and the model programs compiled for a
            # standalone [n] input reject the larger backing buffer
            # (RuntimeProgramInputMismatch).
            for k, j in enumerate(members):
                states[j] = gathered[k].clone()
                if ckpt_window > 1:
                    ckpts[j] = ck[k].clone()
    if ckpt_window <= 1:
        ckpts = []
    return tuple(outs) + tuple(states) + tuple(ckpts)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _staged_walk_products_compiled(
    staged_dev: torch.Tensor,
    seq_lens: torch.Tensor,
    table_geom: tuple,
    mamba_geom: tuple,
    ckpt_window: int,
    include_tables: bool = True,
    class_gather: bool = False,
) -> tuple:
    """One device program for everything the metadata walk consumes:
    per-group block-table splits (graph outputs are distinct buffers, so no
    per-group eager clone) plus the unified-pool mamba state/checkpoint
    gathers. `table_geom` is `((offset, num_reqs, num_blocks), ...)` in walk
    order; `mamba_geom` is `((table_geom_index, num_blocks, block_size),
    ...)`; both are static per padding bucket. Returns `tables... +
    states... + (ckpts... if ckpt_window > 1)`. Issued for every staged walk
    (dummy runs included) so warmup and lockstep ranks dispatch the same
    program sequence.
    """
    return _staged_walk_products(
        staged_dev,
        seq_lens,
        table_geom,
        mamba_geom,
        ckpt_window,
        include_tables,
        class_gather,
    )


class AttentionMetadataBuilder(BaseAttentionMetadataBuilder):
    """Single shared metadata builder for every TPU attention group."""

    def __init__(
        self, kv_cache_spec, layer_names, vllm_config, device, runner, kv_cache_group_id
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.runner = runner
        self.kv_cache_group_id = kv_cache_group_id
        self.is_mamba_group = isinstance(kv_cache_spec, MambaSpec)
        self.target_block_size = self.kv_cache_spec.block_size
        if self.is_mamba_group:
            try:
                cp_world_size = get_pcp_group().world_size
            except Exception:
                cp_world_size = 1
            self.target_block_size *= cp_world_size
        # Only mamba/GDN layers consume physical state slot ids; attention
        # groups leave AttentionMetadata.mamba_state_indices None.

        block_table_obj = runner.input_batch.block_table[self.kv_cache_group_id]
        self.block_tables_cpu = torch.zeros(
            (runner.max_num_reqs, block_table_obj.max_num_blocks_per_req),
            dtype=torch.int32,
            device="cpu",
        )

    def _mamba_row_plan(
        self, ctx: AttentionMetadataBuilderContext, target_num_blocks: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Row offsets into a mamba block table, cached and shared across groups.

        Mamba groups share identical sequence lengths, block sizes, and checkpoint
        geometry. Deriving row offset indices once per (block_size, target_num_blocks)
        allows all mamba groups to share the plan, leaving only per-group gathers.
        """
        key = (self.target_block_size, target_num_blocks)
        plan = ctx.mamba_row_plans.get(key)
        if plan is not None:
            return plan

        is_active = (ctx.seq_lens > 0).unsqueeze(1)
        state_offsets = torch.clamp(
            (ctx.seq_lens - 1) // self.target_block_size,
            min=0,
            max=target_num_blocks - 1,
        ).to(torch.int64)
        state_index = state_offsets.unsqueeze(1)

        ckpt_index: torch.Tensor | None = None
        in_row: torch.Tensor | None = None
        if ctx.mamba_ckpt_window > 1:
            # Spec decoding: the manager allocates `window - 1` checkpoint
            # blocks right after the positional state block, so checkpoint t
            # is the row entry `state_offsets + t`.
            ckpt_offsets = state_index + torch.arange(
                ctx.mamba_ckpt_window,
                device=state_offsets.device,
                dtype=state_offsets.dtype,
            ).unsqueeze(0)
            # A row too short for the group would alias checkpoints onto one
            # block; clamp to the null block and let the caller zero those
            # columns via `in_row`.
            in_row = (ckpt_offsets < target_num_blocks) & is_active
            ckpt_index = torch.where(
                in_row, ckpt_offsets, torch.zeros_like(ckpt_offsets)
            )

        plan = (state_index, ckpt_index, in_row)
        ctx.mamba_row_plans[key] = plan
        return plan

    def _block_table_geometry(
        self, ctx: AttentionMetadataBuilderContext
    ) -> tuple[Any, int]:
        """The (block_table_obj, target_num_blocks) this group's build() uses.

        Shared with `stage_block_table_uploads`, which must pack each group's
        rows with exactly the width the group's build() will slice back out.
        """
        runner = self.runner
        block_table_obj = runner.input_batch.block_table[self.kv_cache_group_id]
        if ctx.use_max_model_len:
            target_num_blocks = block_table_obj.max_num_blocks_per_req
        else:
            assert runner.most_model_len is not None
            spec = self.kv_cache_spec
            first_spec = (
                spec.first_spec if isinstance(spec, UniformTypeKVCacheSpecs) else spec
            )
            if isinstance(first_spec, AttentionSpec):
                target_num_blocks = attention_block_table_width(
                    block_table_obj,
                    self.kv_cache_spec,
                    self.vllm_config,
                    runner.most_model_len,
                )
            else:
                target_num_blocks = cdiv(runner.most_model_len, self.target_block_size)
            if self.is_mamba_group and ctx.mamba_ckpt_window > 1:
                # Speculative decoding: the manager appends
                # `num_speculative_blocks` (= window - 1) checkpoint blocks
                # after the request's positional state block, so a mamba row
                # is that much wider than the positional part alone. This
                # mirrors `MambaSpec.max_num_blocks_per_req`, which the
                # use_max_model_len branch above already gets for free.
                # Without it the row stops short of the checkpoint group and
                # `in_row` below silently redirects checkpoints to the null
                # block, costing rollback accuracy with no error.
                target_num_blocks = min(
                    target_num_blocks + ctx.mamba_ckpt_window - 1,
                    block_table_obj.max_num_blocks_per_req,
                )
        return block_table_obj, target_num_blocks

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        runner = self.runner
        ctx = runner._attn_metadata_builder_ctx
        target_num_reqs = common_attn_metadata.num_reqs

        # `position_ids` is only used for the dummy run in dummy runs, where we
        # want to use fixed position IDs instead of copying from the CPU tensor
        # that gets updated every step.
        staged = (
            ctx.staged_block_tables.get(self.kv_cache_group_id)
            if ctx.staged_block_tables is not None
            else None
        )
        # The table geometry (~59 groups' width arithmetic per step for
        # Qwen4Exp) is derived only on the paths below that slice or reshape
        # a table; the staged path hands out the walk's outputs as they are.
        if staged is not None:
            # The walk pre-uploaded every group's table in one batched
            # transfer; see `stage_block_table_uploads`.
            block_tables_dev = staged
            input_positions = (
                ctx.position_ids_override
                if ctx.position_ids_override is not None
                else runner.position_ids
            )
        elif ctx.position_ids_override is not None:
            _, target_num_blocks = self._block_table_geometry(ctx)
            block_tables_dev = torch.zeros(
                (target_num_reqs * target_num_blocks,), dtype=torch.int32
            ).to(runner.device)
            input_positions = ctx.position_ids_override
        else:
            block_table_obj, target_num_blocks = self._block_table_geometry(ctx)
            block_tables = self.block_tables_cpu[:target_num_reqs, :target_num_blocks]
            block_tables.zero_()
            source_block_tables = block_table_obj.get_cpu_tensor()
            block_tables[: ctx.num_reqs, :target_num_blocks] = source_block_tables[
                ctx.start_index : ctx.start_index + ctx.num_reqs, :target_num_blocks
            ]
            # Flatten on CPU before H2D to avoid device-side
            # as_strided/reshape materialization on every decode step.
            block_tables_dev = block_tables.reshape(-1).to(
                runner.device, non_blocking=True
            )
            input_positions = runner.position_ids

        mamba_ckpt_indices = None
        if not self.is_mamba_group:
            mamba_state_indices = None
        elif ctx.mamba_state_indices is not None:
            # Default compact-mamba path: the runner provides a per-request
            # physical slot id from the compact slot pool.
            mamba_state_indices = ctx.mamba_state_indices
        elif runner._unified_kv_layout:
            # The pool keys mamba state by vLLM block ids in every cache
            # mode (it has no compact slot pool). Derive the current physical
            # state
            # slot on device from the zero-padded `block_tables_dev` (padded
            # tail rows resolve to the null block, never a stale id — the
            # GDN op scans the full length every step), avoiding a
            # D2H -> CPU gather -> H2D dependency.
            staged_products = (
                ctx.staged_mamba_products.get(self.kv_cache_group_id)
                if ctx.staged_mamba_products is not None
                else None
            )
            if staged_products is not None:
                # Derived in the single `_staged_walk_products_compiled`
                # program alongside the staged tables.
                mamba_state_indices, mamba_ckpt_indices = staged_products
            else:
                _, target_num_blocks = self._block_table_geometry(ctx)
                block_tables_2d = block_tables_dev.reshape(
                    target_num_reqs, target_num_blocks
                )
                # Everything but the gathers is shared with the other mamba
                # groups; see `_mamba_row_plan`.
                state_index, ckpt_index, ckpt_in_row = self._mamba_row_plan(
                    ctx, target_num_blocks
                )
                gathered_state_indices = torch.gather(
                    block_tables_2d,
                    dim=1,
                    index=state_index,
                ).squeeze(1)
                mamba_state_indices = torch.where(
                    ctx.seq_lens > 0,
                    gathered_state_indices,
                    torch.zeros_like(gathered_state_indices),
                )
                if ckpt_index is not None:
                    mamba_ckpt_indices = torch.gather(
                        block_tables_2d,
                        dim=1,
                        index=ckpt_index,
                    ) * cast(torch.Tensor, ckpt_in_row)
            if ctx.unified_mamba_state_indices is not None:
                # Spec decode: expose this group's state blocks for the
                # post-sampling read-offset scatter.
                ctx.unified_mamba_state_indices.append(mamba_state_indices)
        else:
            mamba_state_indices = None

        if self.is_mamba_group:
            mamba_slot_read_offsets = ctx.mamba_slot_read_offsets
            mamba_request_distribution = ctx.mamba_request_distribution
        else:
            mamba_slot_read_offsets = None
            mamba_request_distribution = None

        return AttentionMetadata(
            input_positions=input_positions,
            block_tables=block_tables_dev,
            seq_lens=ctx.seq_lens,
            query_start_loc=ctx.query_start_loc,
            request_distribution=ctx.request_distribution,
            mamba_state_indices=mamba_state_indices,
            mamba_slot_read_offsets=mamba_slot_read_offsets,
            mamba_ckpt_indices=mamba_ckpt_indices,
            mamba_request_distribution=mamba_request_distribution,
            sequence_layout_kind=ctx.sequence_layout_descriptor.kind.value,
            sequence_layout_protocol=(ctx.sequence_layout_descriptor.protocol),
            sequence_layout_version=ctx.sequence_layout_descriptor.version,
        )


# Bumped by every BlockTable mutation that can change a row's contents, so
# `stage_block_table_uploads` can tell an unchanged table set without
# re-packing it (see `_install_block_table_version`).
_BLOCK_TABLE_VERSION = [0]


def _install_block_table_version() -> None:
    from vllm.v1.worker.block_table import BlockTable

    if getattr(BlockTable, "_tpu_version_patch", False):
        return

    # Each table also keeps its own version (`_tpu_version`), so a re-pack
    # can recopy only the tables that changed (VLLM_TPU_BLOCK_TABLE_PACK_DELTA).
    def bump_if_blocks(orig):
        @functools.wraps(orig)
        def append_row(self, block_ids, row_idx):
            if len(block_ids):
                _BLOCK_TABLE_VERSION[0] += 1
                self._tpu_version = _BLOCK_TABLE_VERSION[0]
            return orig(self, block_ids, row_idx)

        return append_row

    def bump(orig):
        @functools.wraps(orig)
        def wrapper(self, *args, **kwargs):
            _BLOCK_TABLE_VERSION[0] += 1
            self._tpu_version = _BLOCK_TABLE_VERSION[0]
            return orig(self, *args, **kwargs)

        return wrapper

    BlockTable.append_row = bump_if_blocks(BlockTable.append_row)
    for name in ("add_row", "clear_row", "move_row", "swap_row", "clear"):
        setattr(BlockTable, name, bump(getattr(BlockTable, name)))
    BlockTable._tpu_version_patch = True


_install_block_table_version()


def stage_block_table_uploads(
    runner: Any,
    ctx: AttentionMetadataBuilderContext,
    num_reqs_padded: int,
    gids: set[int] | None = None,
) -> None:
    """One batched H2D transfer for every block table a walk will build.

    Packs all participating groups' rows into one flat CPU scratch, transfers
    once, and hands each group's build() a flat view of its region via
    `ctx.staged_block_tables` (instead of one small per-group H2D per build).
    `gids` limits staging to a subset of groups; None stages every group.
    Dummy runs stage the same layout zero-filled so warmup dispatches the
    program sequence serving will. Views die with the per-chunk context.
    """
    if not runner.attn_groups:
        # Pre-initialization dummy runs build metadata before the KV cache
        # groups exist; each build() synthesizes its own table there.
        return

    # Walk in the same order the builds will, deduping shared group ids. The
    # result depends only on the groups and the walk's geometry inputs, so
    # VLLM_TPU_STAGE_ENTRIES_CACHE keeps it instead of re-walking all groups
    # (59 for Qwen4Exp) every step.
    entries_key = (
        id(runner.attn_groups),
        id(runner.input_batch),
        ctx.use_max_model_len,
        ctx.mamba_ckpt_window,
        None if gids is None else frozenset(gids),
    )
    cached_entries = (
        runner.__dict__.get("_stage_entries_cache")
        if envs.VLLM_TPU_STAGE_ENTRIES_CACHE
        else None
    )
    if cached_entries is not None and cached_entries[0] == entries_key:
        entries, entry_builders = cached_entries[1], cached_entries[2]
    else:
        entries = []
        entry_builders = []
        seen: set[int] = set()
        for group_list in runner.attn_groups:
            for group in group_list:
                builder = group.metadata_builders[0]
                gid = builder.kv_cache_group_id
                if gid in seen or (gids is not None and gid not in gids):
                    continue
                seen.add(gid)
                block_table_obj, target_num_blocks = builder._block_table_geometry(
                    ctx
                )
                entries.append((gid, block_table_obj, target_num_blocks))
                entry_builders.append(builder)
        if envs.VLLM_TPU_STAGE_ENTRIES_CACHE:
            runner._stage_entries_cache = (entries_key, entries, entry_builders)
    if not entries:
        return

    # In steady decode most steps change no table at all. When no BlockTable
    # was mutated since the last staged walk and the walk's geometry is the
    # same, the previous upload is still exact: skip re-packing (a zero fill
    # plus one sliced copy per group, 59 groups for Qwen4Exp) and the H2D.
    table_geom: list[tuple[int, int, int]] = []
    offset = 0
    for _, _, target_num_blocks in entries:
        table_geom.append((offset, num_reqs_padded, target_num_blocks))
        offset += num_reqs_padded * target_num_blocks
    stage_key = (
        _BLOCK_TABLE_VERSION[0],
        tuple(id(bt) for _, bt, _ in entries),
        ctx.start_index,
        ctx.num_reqs,
        tuple(table_geom),
    )
    prev = runner.__dict__.get("_block_table_stage_prev")
    reused_key = None
    if (
        envs.VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE
        and ctx.position_ids_override is None
        and prev is not None
        and prev[3] == stage_key
    ):
        staged_dev = prev[2]
        reused_key = stage_key
    else:
        staged_dev = _pack_and_upload(
            runner, ctx, entries, num_reqs_padded, table_geom, stage_key
        )
    _issue_walk_products(
        runner,
        ctx,
        entries,
        entry_builders,
        num_reqs_padded,
        table_geom,
        staged_dev,
        reused_key,
    )


def _pack_and_upload(runner, ctx, entries, num_reqs_padded, table_geom, stage_key):
    total = sum(num_reqs_padded * tnb for _, _, tnb in entries)
    scratch = runner._block_table_stage_cpu
    if scratch is None or scratch.numel() < total:
        scratch = torch.zeros(
            (total,), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
        )
        runner._block_table_stage_cpu = scratch
    flat = scratch[:total]
    if (
        envs.VLLM_TPU_BLOCK_TABLE_PACK_DELTA
        and ctx.position_ids_override is None
        and envs.VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE
    ):
        staged_dev = _pack_delta(runner, ctx, entries, num_reqs_padded, table_geom, flat)
        runner._block_table_stage_prev = (table_geom, None, staged_dev, stage_key)
        return staged_dev
    runner.__dict__.pop("_block_table_pack_delta", None)
    flat.zero_()

    if ctx.position_ids_override is None:
        offset = 0
        for _, block_table_obj, target_num_blocks in entries:
            rows = flat[offset : offset + num_reqs_padded * target_num_blocks].view(
                num_reqs_padded, target_num_blocks
            )
            source = block_table_obj.get_cpu_tensor()
            rows[: ctx.num_reqs] = source[
                ctx.start_index : ctx.start_index + ctx.num_reqs, :target_num_blocks
            ]
            offset += num_reqs_padded * target_num_blocks
    # Dummy runs keep the zeroed scratch: same staged layout, null blocks.

    # In steady decode the tables change only when a request crosses a block,
    # so most steps would re-upload the same few MB (59 groups x requests x
    # row width for Qwen4Exp) and the walk program would wait on it. Reuse the
    # previous upload when geometry and contents match; the device copy is
    # read-only (every consumer gets the walk program's outputs).
    prev = runner.__dict__.get("_block_table_stage_prev")
    if (
        envs.VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE
        and prev is not None
        and prev[0] == table_geom
        and prev[1] is not None
        and torch.equal(prev[1], flat)
    ):
        staged_dev = prev[2]
    else:
        staged_dev = flat.to(runner.device, non_blocking=True)
    if envs.VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE:
        # Dummy runs (zero-filled) never key a reuse.
        key = stage_key if ctx.position_ids_override is None else None
        runner._block_table_stage_prev = (table_geom, flat.clone(), staged_dev, key)
    return staged_dev


def _pack_delta(runner, ctx, entries, num_reqs_padded, table_geom, flat):
    """VLLM_TPU_BLOCK_TABLE_PACK_DELTA: re-pack only the tables whose rows
    changed since the previous pack of the same layout, and upload only when
    one did. The full pack zero-fills, copies and compares the whole staged
    buffer (a few MB at 59 groups) every time any one table changed."""
    layout = (
        tuple(table_geom),
        tuple(id(bt) for _, bt, _ in entries),
        ctx.start_index,
        ctx.num_reqs,
        flat.data_ptr(),
    )
    state = runner.__dict__.get("_block_table_pack_delta")
    full = state is None or state[0] != layout
    if full:
        flat.zero_()
        versions = [None] * len(entries)
    else:
        versions = state[1]
    flat_np = flat.numpy()
    start, n = ctx.start_index, ctx.num_reqs
    changed = full
    for i, (_, block_table_obj, target_num_blocks) in enumerate(entries):
        version = getattr(block_table_obj, "_tpu_version", 0)
        if not full and versions[i] == version:
            continue
        versions[i] = version
        changed = True
        off = table_geom[i][0]
        rows = flat_np[off : off + num_reqs_padded * target_num_blocks].reshape(
            num_reqs_padded, target_num_blocks
        )
        rows[:n] = block_table_obj.get_numpy_array()[
            start : start + n, :target_num_blocks
        ]
    if changed or state[2] is None:
        staged_dev = flat.to(runner.device, non_blocking=True)
    else:
        staged_dev = state[2]
    runner._block_table_pack_delta = (layout, versions, staged_dev)
    return staged_dev


def _issue_walk_products(
    runner,
    ctx,
    entries,
    entry_builders,
    num_reqs_padded,
    table_geom,
    staged_dev,
    reused_key=None,
):
    # One compiled program splits the staged tensor into per-group tables
    # and derives every unified-pool mamba group's state/checkpoint indices
    # (replaces one micro-program per group per step).
    # Mirror of build()'s branch order: the compact path
    # (ctx.mamba_state_indices) wins, otherwise the unified layout derives
    # per-group indices — which is what this precompute batches.
    unified_mamba = ctx.mamba_state_indices is None and runner._unified_kv_layout
    mamba_geom: list[tuple[int, int, int]] = []
    mamba_entry_indices: list[int] = []
    if unified_mamba:
        for i, builder in enumerate(entry_builders):
            if builder.is_mamba_group:
                mamba_geom.append((i, table_geom[i][2], builder.target_block_size))
                mamba_entry_indices.append(i)

    if len(entries) == 1 and not mamba_geom:
        # Exactly one group with no mamba derivation: the flat upload is
        # already the complete table view; avoid launching an unnecessary
        # device clone program.
        gid, _, target_num_blocks = entries[0]
        length = num_reqs_padded * target_num_blocks
        ctx.staged_block_tables = {gid: staged_dev[:length]}
        return

    walk = (
        _staged_walk_products_compiled
        if staged_dev.device.type == "tpu"
        else _staged_walk_products
    )
    args = (
        staged_dev,
        ctx.seq_lens,
        tuple(table_geom),
        tuple(mamba_geom),
        ctx.mamba_ckpt_window,
    )
    # The upload was reused outright, so the previous walk's table outputs
    # (read-only device tensors) are still exact: derive only the mamba
    # indices, which follow seq_lens, instead of re-emitting every table.
    cached = runner.__dict__.get("_block_table_stage_tables")
    if (
        reused_key is not None
        and mamba_geom
        and cached is not None
        and cached[0] == reused_key
    ):
        ctx.staged_block_tables = cached[1]
        products = walk(*args, False, envs.VLLM_TPU_STAGED_WALK_CLASS_GATHER)
        n_tables = 0
    else:
        products = walk(*args, True, envs.VLLM_TPU_STAGED_WALK_CLASS_GATHER)
        n_tables = len(entries)
        ctx.staged_block_tables = {
            gid: products[i] for i, (gid, _, _) in enumerate(entries)
        }
        if envs.VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE:
            prev = runner.__dict__.get("_block_table_stage_prev")
            if ctx.position_ids_override is None and prev is not None:
                runner._block_table_stage_tables = (prev[3], ctx.staged_block_tables)
            elif mamba_geom:
                # Dummy runs warm the tables-free variant serving reuses.
                walk(*args, False, envs.VLLM_TPU_STAGED_WALK_CLASS_GATHER)
    n_mamba = len(mamba_geom)
    if n_mamba:
        states = products[n_tables : n_tables + n_mamba]
        has_ckpt = ctx.mamba_ckpt_window > 1
        ckpts = products[n_tables + n_mamba :] if has_ckpt else (None,) * n_mamba
        ctx.staged_mamba_products = {
            entries[ei][0]: (states[j], ckpts[j] if has_ckpt else None)
            for j, ei in enumerate(mamba_entry_indices)
        }
