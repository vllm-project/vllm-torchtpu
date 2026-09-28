from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any

import jax
import torch
from vllm.distributed import get_pcp_group
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder as BaseAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_torchtpu.layers.core.sequence_layout import (
    DEFAULT_SEQUENCE_LAYOUT_DESCRIPTOR,
    DEFAULT_SEQUENCE_LAYOUT_PROTOCOL,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
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
    input_positions: jax.Array
    # (max_num_seqs * max_num_blocks_per_req,)
    block_tables: jax.Array = None
    # (max_num_seqs,)
    seq_lens: jax.Array = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array = None
    # (3,)
    request_distribution: jax.Array = None
    # (max_num_seqs,) int32 - physical slot id in the mamba kv-cache for the
    # request currently in each persistent-batch position. Compact-mamba uses
    # an explicit slot from its independent pool; unified align mode derives
    # the slot from the current entry in the device block table. None for
    # attention groups, non-mamba models, and Mamba modes whose native fallback
    # addresses block_tables[:, 0].
    mamba_state_indices: jax.Array | None = None
    # (mamba_num_blocks,) int32 — per-*slot* read offset for speculative
    # decoding with mamba layers. `mamba_slot_read_offsets[base_slot]` is
    # `num_accepted - 1` from the request's most recent verify step: the GDN
    # kernel reads the request's initial state from `base_slot + offset`
    # (the checkpoint of the last accepted token) and writes fresh
    # checkpoints starting at `base_slot`. Indexed by physical slot (not
    # batch position) so the value survives requests being rescheduled or
    # condensed. Updated on device after each sampling step; None unless the
    # model has mamba layers *and* speculative decoding is enabled.
    mamba_slot_read_offsets: jax.Array | None = None
    # `(num_reqs, num_spec + 1)` source block per checkpoint. Unified pool
    # with speculative decoding only; None elsewhere, where the sole
    # checkpoint is the request's own state block.
    mamba_ckpt_indices: jax.Array | None = None
    # (3,) int32 — GDN-specific request distribution, same format as
    # `request_distribution` but with the first segment covering all
    # *windowed* sequences (plain decodes and speculative verify windows of
    # up to num_spec + 1 tokens) instead of only 1-token decodes. The
    # persistent batch is ordered [decode][verify][prefill/mixed] so both
    # segmentations hold at once: ragged paged attention keeps its 1-token
    # decode front segment while the GDN kernel runs its windowed mode over
    # the first two groups. None unless the model has mamba layers and spec
    # decoding is enabled.
    mamba_request_distribution: jax.Array | None = None
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
) -> tuple:
    """Splits the staged flat block-table buffer into per-group tables and
    derives unified-pool mamba state and checkpoint indices."""
    outs = []
    for off, n, t in table_geom:
        outs.append(staged_dev[off : off + n * t].clone())
    states = []
    ckpts = []
    if mamba_geom:
        active = seq_lens > 0
        for ti, t, bs in mamba_geom:
            off, n, _ = table_geom[ti]
            tbl = staged_dev[off : off + n * t].view(n, t)
            state_offsets = torch.clamp((seq_lens - 1) // bs, min=0, max=t - 1).to(
                torch.int64
            )
            gathered = torch.gather(tbl, 1, state_offsets.unsqueeze(1)).squeeze(1)
            states.append(torch.where(active, gathered, torch.zeros_like(gathered)))
            if ckpt_window > 1:
                ckpt_offsets = state_offsets.unsqueeze(1) + torch.arange(
                    ckpt_window, device=seq_lens.device, dtype=torch.int64
                ).unsqueeze(0)
                in_row = (ckpt_offsets < t) & active.unsqueeze(1)
                safe = torch.where(in_row, ckpt_offsets, torch.zeros_like(ckpt_offsets))
                ckpts.append(torch.gather(tbl, 1, safe) * in_row)
    return tuple(outs) + tuple(states) + tuple(ckpts)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _staged_walk_products_compiled(
    staged_dev: torch.Tensor,
    seq_lens: torch.Tensor,
    table_geom: tuple,
    mamba_geom: tuple,
    ckpt_window: int,
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
        staged_dev, seq_lens, table_geom, mamba_geom, ckpt_window
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
        block_table_obj, target_num_blocks = self._block_table_geometry(ctx)

        # `position_ids` is only used for the dummy run in dummy runs, where we
        # want to use fixed position IDs instead of copying from the CPU tensor
        # that gets updated every step.
        staged = (
            ctx.staged_block_tables.get(self.kv_cache_group_id)
            if ctx.staged_block_tables is not None
            else None
        )
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
            block_tables_dev = torch.zeros(
                (target_num_reqs * target_num_blocks,), dtype=torch.int32
            ).to(runner.device)
            input_positions = ctx.position_ids_override
        else:
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
                    mamba_ckpt_indices = (
                        torch.gather(
                            block_tables_2d,
                            dim=1,
                            index=ckpt_index,
                        )
                        * ckpt_in_row
                    )
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

    # Walk in the same order the builds will, deduping shared group ids.
    entries: list[tuple[int, Any, int]] = []
    entry_builders: list[Any] = []
    seen: set[int] = set()
    for group_list in runner.attn_groups:
        for group in group_list:
            builder = group.metadata_builders[0]
            gid = builder.kv_cache_group_id
            if gid in seen or (gids is not None and gid not in gids):
                continue
            seen.add(gid)
            block_table_obj, target_num_blocks = builder._block_table_geometry(ctx)
            entries.append((gid, block_table_obj, target_num_blocks))
            entry_builders.append(builder)
    if not entries:
        return

    total = sum(num_reqs_padded * tnb for _, _, tnb in entries)
    scratch = runner._block_table_stage_cpu
    if scratch is None or scratch.numel() < total:
        scratch = torch.zeros(
            (total,), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY
        )
        runner._block_table_stage_cpu = scratch
    flat = scratch[:total]
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

    staged_dev = flat.to(runner.device, non_blocking=True)

    # One compiled program splits the staged tensor into per-group tables
    # and derives every unified-pool mamba group's state/checkpoint indices
    # (replaces one micro-program per group per step).
    table_geom: list[tuple[int, int, int]] = []
    offset = 0
    for _, _, target_num_blocks in entries:
        table_geom.append((offset, num_reqs_padded, target_num_blocks))
        offset += num_reqs_padded * target_num_blocks

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

    if staged_dev.device.type == "tpu":
        products = _staged_walk_products_compiled(
            staged_dev,
            ctx.seq_lens,
            tuple(table_geom),
            tuple(mamba_geom),
            ctx.mamba_ckpt_window,
        )
    else:
        products = _staged_walk_products(
            staged_dev,
            ctx.seq_lens,
            tuple(table_geom),
            tuple(mamba_geom),
            ctx.mamba_ckpt_window,
        )

    n_tables = len(entries)
    n_mamba = len(mamba_geom)
    ctx.staged_block_tables = {
        gid: products[i] for i, (gid, _, _) in enumerate(entries)
    }
    if n_mamba:
        states = products[n_tables : n_tables + n_mamba]
        has_ckpt = ctx.mamba_ckpt_window > 1
        ckpts = products[n_tables + n_mamba :] if has_ckpt else (None,) * n_mamba
        ctx.staged_mamba_products = {
            entries[ei][0]: (states[j], ckpts[j] if has_ckpt else None)
            for j, ei in enumerate(mamba_entry_indices)
        }
