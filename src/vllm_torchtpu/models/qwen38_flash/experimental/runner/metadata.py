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
"""Qwen3.8-Flash copy of the runner's staged attention metadata walk.

Served through `ModelAdapter.build_attention_metadata` (see
`Qwen4ExpAdapter.build_attention_metadata`): the runner's own walk is not used
for this model. Copied from `layers/core/attention_metadata.py` and
`TPUModelRunner._build_attention_metadata` with these changes applied:

- #2606 block tables: reuse unchanged uploads and skip re-packing by version
  (`_install_block_table_version`, installed from `hooks.register()`);
- #2721 staged walk: one batched gather per mamba geometry class;
- #3217 cache the staged walk's per-group entries;
- #2649 build per-group attention metadata inline
  (`_build_attention_metadata_staged`);
- #2626 block tables: re-pack only the changed tables (`_pack_delta`, and the
  deferred table geometry in `_build_group`).

Only the functions those changes touch are copied (`_build_group` is
`AttentionMetadataBuilder.build`); the metadata types, the builder class and
the unchanged helpers are the shared ones. Re-sync on rebases.
"""

import functools
from types import SimpleNamespace
from typing import Any, cast

import torch

from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
    _gather_prefill_ckpt_ids,
    stage_h2d_upload,
    zeros_h2d_source,
)


def _staged_walk_products(
    staged_dev: torch.Tensor,
    seq_lens: torch.Tensor,
    table_geom: tuple,
    mamba_geom: tuple,
    ckpt_window: int,
    prefill_ckpt_offsets: torch.Tensor | None = None,
    include_tables: bool = True,
) -> tuple:
    """Splits the staged flat block-table buffer into per-group tables and
    derives unified-pool mamba state and checkpoint indices. Without
    `include_tables` only the mamba indices are returned (the caller still
    holds the tables of an identical earlier upload)."""
    outs = []
    if include_tables:
        for off, n, t in table_geom:
            outs.append(staged_dev[off : off + n * t].clone())
    states: list = [None] * len(mamba_geom)
    ckpts: list = [None] * len(mamba_geom)
    prefill_ckpts = []
    if mamba_geom:
        active = seq_lens > 0
        device = seq_lens.device
        # Groups sharing (num_blocks, block_size) share every row offset, so
        # each such class is one gather over the flat staged buffer instead
        # of one small gather per group (~1.5 us each, ~70 per step).
        classes: dict[tuple[int, int], list[int]] = {}
        for j, (_, t, bs, _) in enumerate(mamba_geom):
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
            ck: torch.Tensor | None = None
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
                if ck is not None:
                    ckpts[j] = ck[k].clone()
        # Internal prefill checkpoints stay per group, in group order: only
        # the few groups that declare them gather.
        if prefill_ckpt_offsets is not None:
            for ti, t, bs, prefill_ckpt_spec_blocks in mamba_geom:
                if prefill_ckpt_spec_blocks is None:
                    continue
                off, n, _ = table_geom[ti]
                tbl = staged_dev[off : off + n * t].view(n, t)
                state_offsets = torch.clamp((seq_lens - 1) // bs, min=0, max=t - 1).to(
                    torch.int64
                )
                prefill_ckpts.append(
                    _gather_prefill_ckpt_ids(
                        tbl,
                        state_offsets,
                        prefill_ckpt_offsets,
                        prefill_ckpt_spec_blocks,
                    )
                )
    if ckpt_window <= 1:
        ckpts = []
    return tuple(outs) + tuple(states) + tuple(ckpts) + tuple(prefill_ckpts)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _staged_walk_products_compiled(
    staged_dev: torch.Tensor,
    seq_lens: torch.Tensor,
    table_geom: tuple,
    mamba_geom: tuple,
    ckpt_window: int,
    prefill_ckpt_offsets: torch.Tensor | None = None,
    include_tables: bool = True,
) -> tuple:
    """Fused device program for per-group block-table splits and unified-pool
    mamba state/checkpoint gathers.

    - table_geom: ((offset, num_reqs, num_blocks), ...) in walk order.
    - mamba_geom: ((table_geom_index, num_blocks, block_size,
      prefill_ckpt_spec_blocks), ...), where prefill_ckpt_spec_blocks is the
      group's speculative block count when it declares prefill checkpoints and
      None otherwise. Both geoms are static per padding bucket.
    - Returns: tables + states + (ckpts if ckpt_window > 1) + (prefill_ckpts
      for declaring groups if prefill_ckpt_offsets is given).

    Issued for every staged walk, dummy runs included, so warmup and lockstep
    ranks dispatch the same program sequence.
    """
    return _staged_walk_products(
        staged_dev,
        seq_lens,
        table_geom,
        mamba_geom,
        ckpt_window,
        prefill_ckpt_offsets,
        include_tables,
    )


def _build_group(self, common_prefix_len, common_attn_metadata, fast_build=False):
    """`AttentionMetadataBuilder.build` of the served code, for `self` a
    plain `AttentionMetadataBuilder`."""
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
    # The table geometry (~60 groups' width arithmetic per step on a large
    # hybrid model) is derived only on the paths below that slice or reshape
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
        # Per-call buffer: a later chunk or draft walk must not overwrite a
        # table whose upload is still in flight (see `stage_h2d_upload`).
        flat = zeros_h2d_source(runner, target_num_reqs * target_num_blocks)
        block_tables = flat.view(target_num_reqs, target_num_blocks)
        # PCP currently only read block table inside pcpmetadata
        # TODO(weiyulin): Evaluate if we want to derive split/replicated block
        # tables for pcpmanager from md.blocktable
        if runner.pcp_manager is None:
            source_block_tables = block_table_obj.get_cpu_tensor()
            block_tables[: ctx.num_reqs, :target_num_blocks] = source_block_tables[
                ctx.start_index : ctx.start_index + ctx.num_reqs, :target_num_blocks
            ]
        # Flatten on CPU before H2D to avoid device-side
        # as_strided/reshape materialization on every decode step.
        block_tables_dev = stage_h2d_upload(runner, flat)
        input_positions = runner.position_ids

    mamba_ckpt_indices = None
    mamba_prefill_ckpt_indices = None
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
            (
                mamba_state_indices,
                mamba_ckpt_indices,
                mamba_prefill_ckpt_indices,
            ) = staged_products
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
            if (
                ctx.mamba_prefill_ckpt_offsets is not None
                and self.prefill_ckpt_spec_blocks is not None
            ):
                mamba_prefill_ckpt_indices = _gather_prefill_ckpt_ids(
                    block_tables_2d,
                    state_index.squeeze(1),
                    ctx.mamba_prefill_ckpt_offsets,
                    self.prefill_ckpt_spec_blocks,
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
    mamba_prefill_ckpt_offsets = (
        ctx.mamba_prefill_ckpt_offsets if self.has_prefill_ckpt_blocks else None
    )
    if mamba_prefill_ckpt_offsets is not None:
        assert mamba_prefill_ckpt_indices is not None, (
            "Prefill checkpoints need the unified KV pool."
        )

    pcp_metadata = (
        None
        if runner.pcp_manager is None
        else runner.pcp_manager.build_metadata(
            runner=runner,
            # As served (#2626): only the non-staged paths bind these; the
            # model runs without PCP.
            block_table_obj=block_table_obj,  # pyrefly: ignore[unbound-name]
            target_num_reqs=target_num_reqs,
            target_num_blocks=target_num_blocks,  # pyrefly: ignore[unbound-name]
        )
    )

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
        pcp_metadata=pcp_metadata,
        mamba_prefill_ckpt_offsets=mamba_prefill_ckpt_offsets,
        mamba_prefill_ckpt_indices=mamba_prefill_ckpt_indices,
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
    # can recopy only the tables that changed (`_pack_delta`).
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


def stage_block_table_uploads(
    runner: Any,
    ctx: AttentionMetadataBuilderContext,
    num_reqs_padded: int,
    gids: set[int] | None = None,
) -> None:
    """One batched H2D transfer for every block table a walk will build.

    Packs all participating groups' rows into one CPU buffer owned by this
    walk, transfers once, and hands each group's build() a flat view of its
    region via `ctx.staged_block_tables` (instead of one small per-group H2D
    per build). `gids` limits staging to a subset of groups; None stages every
    group. Dummy runs stage the same layout zero-filled so warmup dispatches
    the program sequence serving will. Views die with the per-chunk context.
    """
    if not runner.attn_groups:
        # Pre-initialization dummy runs build metadata before the KV cache
        # groups exist; each build() synthesizes its own table there.
        return

    # Walk in the same order the builds will, deduping shared group ids. The
    # result depends only on the groups and the walk's geometry inputs, so
    # cache it on the runner instead of re-walking all groups (~60 on a large
    # hybrid model) every step.
    entries_key = (
        id(runner.attn_groups),
        id(runner.input_batch),
        ctx.use_max_model_len,
        ctx.mamba_ckpt_window,
        None if gids is None else frozenset(gids),
    )
    cached_entries = runner.__dict__.get("_stage_entries_cache")
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
                block_table_obj, target_num_blocks = builder._block_table_geometry(ctx)
                entries.append((gid, block_table_obj, target_num_blocks))
                entry_builders.append(builder)
        runner._stage_entries_cache = (entries_key, entries, entry_builders)
    if not entries:
        return

    # In steady decode most steps change no table at all. When no BlockTable
    # was mutated since the last staged walk and the walk's geometry is the
    # same, the previous upload is still exact: skip re-packing (a zero fill
    # plus one sliced copy per group, ~60 groups on a large hybrid model) and
    # the H2D.
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
    if ctx.position_ids_override is None and prev is not None and prev[3] == stage_key:
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
    # Upload reuse is always on, so re-pack only the changed tables.
    if ctx.position_ids_override is None and all(
        isinstance(getattr(bt, "_tpu_version", 0), int) for _, bt, _ in entries
    ):
        staged_dev = _pack_delta(
            runner, ctx, entries, num_reqs_padded, table_geom, total
        )
        runner._block_table_stage_prev = (table_geom, None, staged_dev, stage_key)
        return staged_dev
    runner.__dict__.pop("_block_table_pack_delta", None)
    # Per-walk buffer: TorchTPU reads the CPU source until the async H2D
    # drains, so a later chunk or draft walk must not refill a shared scratch
    # (see `stage_h2d_upload`).
    flat = zeros_h2d_source(runner, total)

    # The PCP manager's layers read block tables inside pcp_metadata.
    if ctx.position_ids_override is None and runner.pcp_manager is None:
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
    # Dummy runs keep the zeroed buffer: same staged layout, null blocks.

    # In steady decode the tables change only when a request crosses a block,
    # so most steps would re-upload the same few MB (~60 groups x requests x
    # row width on a large hybrid model) and the walk program would wait on
    # it. Reuse the previous upload when geometry and contents match; the
    # device copy is read-only (every consumer gets the walk program's
    # outputs).
    prev = runner.__dict__.get("_block_table_stage_prev")
    if (
        ctx.position_ids_override is None
        and prev is not None
        and prev[0] == table_geom
        and prev[1] is not None
        and torch.equal(prev[1], flat)
    ):
        staged_dev = prev[2]
        # Nothing was uploaded from `flat`: hand a recycled buffer straight back.
        if id(flat) in runner._registered_h2d_sources:
            key = (flat.numel(), flat.dtype)
            free = runner._free_h2d_sources.pop(key, None) or []
            free.append(flat)
            runner._free_h2d_sources[key] = free
    else:
        staged_dev = stage_h2d_upload(runner, flat)
    # Dummy runs (zero-filled) never key a reuse.
    key = stage_key if ctx.position_ids_override is None else None
    runner._block_table_stage_prev = (table_geom, flat.clone(), staged_dev, key)
    return staged_dev


def _pack_delta(runner, ctx, entries, num_reqs_padded, table_geom, total):
    """Re-pack only the tables whose rows changed since the previous pack of
    the same layout, and upload only when one did. The full pack zero-fills,
    copies and compares the whole staged buffer (a few MB at 59 groups) every
    time any one table changed.

    The packed tables live in a CPU mirror that is never an H2D source; an
    upload copies it into a fresh per-walk buffer (see `zeros_h2d_source`), so
    an in-flight transfer never sees the mirror change."""
    mirror = runner.__dict__.get("_block_table_pack_mirror")
    if mirror is None or mirror.numel() < total:
        mirror = torch.zeros((total,), dtype=torch.int32, device="cpu")
        runner._block_table_pack_mirror = mirror
    flat = mirror[:total]
    layout = (
        tuple(table_geom),
        tuple(id(bt) for _, bt, _ in entries),
        ctx.start_index,
        ctx.num_reqs,
        flat.data_ptr(),
    )
    state = runner.__dict__.get("_block_table_pack_delta")
    full = state is None or state[0] != layout
    versions: list[int | None]
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
        off = table_geom[i][0]
        rows = flat_np[off : off + num_reqs_padded * target_num_blocks].reshape(
            num_reqs_padded, target_num_blocks
        )
        rows[:n] = block_table_obj.get_numpy_array()[
            start : start + n, :target_num_blocks
        ]
        versions[i] = version
        changed = True
    if changed or state is None or state[2] is None:
        source = zeros_h2d_source(runner, total)
        source.copy_(flat)
        staged_dev = stage_h2d_upload(runner, source)
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
    mamba_geom: list[tuple[int, int, int, int | None]] = []
    mamba_entry_indices: list[int] = []
    if unified_mamba:
        for i, builder in enumerate(entry_builders):
            if builder.is_mamba_group:
                mamba_geom.append(
                    (
                        i,
                        table_geom[i][2],
                        builder.target_block_size,
                        builder.prefill_ckpt_spec_blocks,
                    )
                )
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
        ctx.mamba_prefill_ckpt_offsets,
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
        products = walk(*args, False)
        n_tables = 0
    else:
        products = walk(*args)
        n_tables = len(entries)
        ctx.staged_block_tables = {
            gid: products[i] for i, (gid, _, _) in enumerate(entries)
        }
        prev = runner.__dict__.get("_block_table_stage_prev")
        if ctx.position_ids_override is None and prev is not None:
            runner._block_table_stage_tables = (prev[3], ctx.staged_block_tables)
        elif mamba_geom:
            # Dummy runs warm the tables-free variant serving reuses.
            walk(*args, False)
    n_mamba = len(mamba_geom)
    if n_mamba:
        states = products[n_tables : n_tables + n_mamba]
        rest = products[n_tables + n_mamba :]
        if ctx.mamba_ckpt_window > 1:
            ckpts = rest[:n_mamba]
            rest = rest[n_mamba:]
        else:
            ckpts = (None,) * n_mamba
        # Prefill checkpoint ids follow, one per group that declares them.
        prefill_ckpts = iter(rest)
        ctx.staged_mamba_products = {}
        for j, ei in enumerate(mamba_entry_indices):
            prefill_ckpt = (
                next(prefill_ckpts)
                if ctx.mamba_prefill_ckpt_offsets is not None
                and mamba_geom[j][3] is not None
                else None
            )
            ctx.staged_mamba_products[entries[ei][0]] = (
                states[j],
                ckpts[j],
                prefill_ckpt,
            )


def build_attention_metadata(
    self, ctx: AttentionMetadataBuilderContext, effective_num_reqs_padded: int
) -> dict[str, AttentionMetadata]:
    """`TPUModelRunner._build_attention_metadata` of the served code, after
    its `ctx`/`effective_num_reqs_padded` lines (the runner passes both to
    `ModelAdapter.build_attention_metadata`).

    Lean TPU group walk: stage one batched block-table upload, then
    build every group's metadata straight from the builder context.
    """
    if ctx.staged_block_tables is None:
        # One batched H2D for every group's block table (zero-filled for
        # dummy runs); see `stage_block_table_uploads`.
        stage_block_table_uploads(self, ctx, effective_num_reqs_padded)

    if ctx.staged_block_tables is not None and ctx.position_ids_override is None:
        fast = _build_attention_metadata_staged(self, ctx)
        if fast is not None:
            return fast
    common = SimpleNamespace(num_reqs=effective_num_reqs_padded)
    per_layer_attn_metadata: dict[str, AttentionMetadata] = {}
    for group_list in self.attn_groups:
        for group in group_list:
            built = _build(
                group.metadata_builders[0],
                common_prefix_len=0,
                common_attn_metadata=common,
            )
            for layer_name in group.layer_names:
                per_layer_attn_metadata[layer_name] = built
    return per_layer_attn_metadata


def _build(builder: Any, **kwargs: Any) -> AttentionMetadata:
    """`builder.build(**kwargs)`, with the served `build` for a plain
    `AttentionMetadataBuilder` (subclasses keep their own)."""
    if type(builder) is AttentionMetadataBuilder:
        return _build_group(builder, **kwargs)
    return builder.build(**kwargs)


def _build_attention_metadata_staged(self, ctx: AttentionMetadataBuilderContext):
    """Inlined group walk for a staged chunk: builds the same metadata
    objects (staged table, shared step tensors, the walk program's mamba
    indices) without one builder dispatch and branch per group per step.
    Returns None when a group needs `build()`'s per-group fallback (custom
    builder, missing staged table, or missing unified mamba products), so
    the caller runs the ordinary walk."""
    plan = self.__dict__.get("_fast_metadata_plan")
    if plan is None or plan[0] is not self.attn_groups:
        groups: list[tuple[int, bool, tuple[str, ...]]] = []
        supported = True
        for group_list in self.attn_groups:
            for group in group_list:
                builder = group.metadata_builders[0]
                if type(builder) is not AttentionMetadataBuilder:
                    supported = False
                    break
                groups.append(
                    (
                        builder.kv_cache_group_id,
                        builder.is_mamba_group,
                        tuple(group.layer_names),
                    )
                )
            if not supported:
                break
        plan = self._fast_metadata_plan = (
            self.attn_groups,
            groups if supported else None,
        )
    if plan[1] is None:
        return None
    staged = ctx.staged_block_tables
    if staged is None:
        return None
    compact = ctx.mamba_state_indices
    unified = compact is None and getattr(self, "_unified_kv_layout", False)
    products = ctx.staged_mamba_products
    if ctx.mamba_prefill_ckpt_offsets is not None:
        # Internal prefill checkpoints: the builders resolve per-group
        # checkpoint ids and offsets; take their walk.
        return None
    for gid, is_mamba, _ in plan[1]:
        if gid not in staged or (
            is_mamba and unified and (products is None or gid not in products)
        ):
            return None
    descriptor = ctx.sequence_layout_descriptor
    shared = dict(
        input_positions=self.position_ids,
        seq_lens=ctx.seq_lens,
        query_start_loc=ctx.query_start_loc,
        request_distribution=ctx.request_distribution,
        sequence_layout_kind=descriptor.kind.value,
        sequence_layout_protocol=descriptor.protocol,
        sequence_layout_version=descriptor.version,
    )
    per_layer: dict[str, AttentionMetadata] = {}
    for gid, is_mamba, layer_names in plan[1]:
        if is_mamba:
            ckpt = None
            if compact is not None:
                state = compact
            elif unified:
                assert products is not None
                state, ckpt, _ = products[gid]
            else:
                state = None
            if unified and ctx.unified_mamba_state_indices is not None:
                ctx.unified_mamba_state_indices.append(state)
            built = AttentionMetadata(
                block_tables=staged[gid],
                mamba_state_indices=state,
                mamba_slot_read_offsets=ctx.mamba_slot_read_offsets,
                mamba_ckpt_indices=ckpt,
                mamba_request_distribution=ctx.mamba_request_distribution,
                **shared,
            )
        else:
            built = AttentionMetadata(block_tables=staged[gid], **shared)
        for layer_name in layer_names:
            per_layer[layer_name] = built
    return per_layer
