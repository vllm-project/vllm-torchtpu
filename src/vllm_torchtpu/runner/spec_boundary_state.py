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
"""Boundary states of decode-filled mamba blocks under speculative decoding.

In mamba align mode a verify window starting at ``start`` writes checkpoint
t, the state after ``start + t + 1`` tokens, to the block t columns past its
state block. The block a boundary T completes therefore holds checkpoint 0 of
the last window that ended inside it, a few tokens short of T, yet the
scheduler hashes it once T tokens are confirmed and prefix hits resume from
it. After sampling, checkpoint ``T - 1 - start`` is copied into it instead
(the counterpart of upstream vLLM's align-mode ``postprocess_mamba``).

Whether a row reaches T depends on its accepted count, known only on device.
``plan_copies`` therefore stages every verify row whose window can reach a
boundary, and ``copy_states`` chooses the copies on device, in one compiled
program, once the chunk is sampled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm_torchtpu.layers.core.attention_metadata import stage_h2d_upload

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class SpecBoundaryPlan(NamedTuple):
    """One chunk's verify rows whose accepted tokens may reach a mamba block
    boundary (`plan_copies`)."""

    # Device `(row, boundary, num_scheduled, blocks...)` per candidate.
    entries: torch.Tensor
    # The chunk's device sequence lengths this step, one per batch row.
    seq_lens: torch.Tensor
    # Logical tokens per mamba block-table column.
    block_size: int


def _spec_boundary_pairs_impl(
    entries: torch.Tensor,
    seq_lens: torch.Tensor,
    read_offsets: torch.Tensor,
    block_size: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[num_groups, n]` source and destination state blocks of the boundary
    copies; the null block where a candidate's accepted tokens stop short of
    its boundary.

    `entries` rows are `(row, boundary, num_scheduled, blocks...)`, with
    `width` block ids per mamba group starting at the boundary's own column.
    `seq_lens` and `read_offsets` hold, per batch row, the sequence length
    this step and the accepted drafts.
    """
    rows = entries[:, 0].long()
    seq_lens = seq_lens.index_select(0, rows)
    num_valid = read_offsets.index_select(0, rows) + 1
    boundary = entries[:, 1]
    start = seq_lens - entries[:, 2]
    reached = (start < boundary) & (boundary <= start + num_valid)
    # Checkpoint t, the state after t + 1 window tokens, lies t columns past
    # the column of the window's last token.
    col = (
        (seq_lens - 1) // block_size
        + (boundary - 1 - start)
        - (boundary // block_size - 1)
    )
    blocks = entries[:, 3:].reshape(entries.shape[0], -1, width).transpose(0, 1)
    index = col.clamp(0, width - 1).view(1, -1, 1).expand(blocks.shape[0], -1, 1)
    src = blocks.gather(2, index).squeeze(2)
    dst = blocks[:, :, 0]
    copy = reached & (col >= 0) & (col < width) & (src != 0) & (dst != 0) & (src != dst)
    null = torch.zeros_like(src)
    return torch.where(copy, src, null), torch.where(copy, dst, null)


_spec_boundary_pairs_compiled = torch.compile(
    backend="tpu", fullgraph=True, dynamic=False
)(_spec_boundary_pairs_impl)


def _spec_boundary_pairs(
    entries: torch.Tensor,
    seq_lens: torch.Tensor,
    read_offsets: torch.Tensor,
    block_size: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fn = (
        _spec_boundary_pairs_compiled
        if entries.device.type == "tpu"
        else _spec_boundary_pairs_impl
    )
    return fn(entries, seq_lens, read_offsets, block_size, width)


def enabled(runner: TPUModelRunner) -> bool:
    """Whether verify steps write boundary states into the blocks they
    complete: speculative decoding on the unified pool in align mode,
    with block hashes (prefix caching or a KV connector) to reuse them.
    """
    return (
        runner.mamba_slot_read_offsets is not None
        and runner._mamba_ckpt_window > 1
        and bool(runner._mamba_copy_plan)
        and (
            runner.cache_config.enable_prefix_caching
            or runner.vllm_config.kv_transfer_config is not None
        )
    )


def _blocks_per_group(runner: TPUModelRunner) -> int:
    """Block ids staged per candidate and mamba group: the boundary's own
    column and one more per window checkpoint."""
    return runner._mamba_ckpt_window + 1


def entries_shape(runner: TPUModelRunner, num_candidates: int) -> tuple[int, int]:
    """Shape of the entries `plan_copies` stages for `num_candidates`.

    Rows pad up the copy ladder, bounded by max_num_reqs, so serving stages
    only the shapes warmup runs; columns are `(row, boundary, num_scheduled)`
    and each mamba group's block ids.
    """
    rows = min(
        runner._bucket_len(num_candidates), max(num_candidates, runner.max_num_reqs)
    )
    return rows, 3 + len(runner._mamba_copy_plan) * _blocks_per_group(runner)


def plan_copies(
    runner: TPUModelRunner,
    scheduler_output: SchedulerOutput,
    start_index: int,
    num_reqs: int,
    seq_lens: torch.Tensor,
) -> SpecBoundaryPlan | None:
    """Stage this chunk's candidates for `copy_states`; None when no row's
    window can reach a boundary.

    A candidate carries its boundary, the block ids from the boundary's
    column on, and its device sequence length (under async scheduling the
    host count runs ahead by up to num_speculative_tokens rejected drafts).
    """
    if not enabled(runner):
        return None
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    if not spec_tokens:
        return None
    block_size = runner._mamba_column_stride()
    slack = 0
    if runner.scheduler_config.async_scheduling:
        assert runner.speculative_config is not None
        slack = runner.speculative_config.num_speculative_tokens
    num_computed = runner.input_batch.num_computed_tokens_cpu
    req_ids = runner.input_batch.req_ids
    candidates: list[tuple[int, int, int]] = []
    for i in range(num_reqs):
        req_id = req_ids[start_index + i]
        if not spec_tokens.get(req_id):
            continue
        computed = int(num_computed[start_index + i])
        scheduled = scheduler_output.num_scheduled_tokens[req_id]
        boundary = (max(computed - slack, 0) // block_size + 1) * block_size
        while boundary <= computed + scheduled:
            candidates.append((i, boundary, scheduled))
            boundary += block_size
    if not candidates:
        return None

    width = _blocks_per_group(runner)
    n = len(candidates)
    host = torch.zeros(entries_shape(runner, n), dtype=torch.int32)
    host[:n, :3] = torch.tensor(candidates, dtype=torch.int32)
    for g, (gid, _) in enumerate(runner._mamba_copy_plan):
        bt = runner.input_batch.block_table[gid].get_cpu_tensor()
        row_width = bt.shape[1]
        base = 3 + g * width
        for e, (i, boundary, _) in enumerate(candidates):
            first = boundary // block_size - 1
            last = min(first + width, row_width)
            host[e, base : base + last - first] = bt[start_index + i, first:last]
    return SpecBoundaryPlan(
        entries=stage_h2d_upload(runner, host),
        seq_lens=seq_lens,
        block_size=block_size,
    )


def copy_states(
    runner: TPUModelRunner,
    plan: SpecBoundaryPlan | None,
    read_offsets: torch.Tensor | None,
) -> None:
    """Copy each boundary state this verify chunk reached into the block
    that boundary completes, for every mamba group.

    `read_offsets` are the chunk's accepted drafts per batch row, as
    `TPUModelRunner._update_mamba_slot_read_offsets` returns them. Runs right
    after sampling, before any later step can hash, free or reuse the block;
    the request still owns it, since the manager frees a state block only
    once later tokens are processed.
    """
    if plan is None:
        return
    assert read_offsets is not None
    src, dst = _spec_boundary_pairs(
        plan.entries,
        plan.seq_lens,
        read_offsets,
        plan.block_size,
        _blocks_per_group(runner),
    )
    per_raw: dict[
        tuple[int, ...],
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]],
    ] = {}
    for g, (_, raws) in enumerate(runner._mamba_copy_plan):
        src_g, dst_g = runner._expand_pool_split(src[g], dst[g])
        entry = per_raw.setdefault(tuple(id(raw) for raw in raws), (raws, [], []))
        entry[1].append(src_g)
        entry[2].append(dst_g)
    for raws, srcs, dsts in per_raw.values():
        runner._copy_mamba_state_blocks(
            raws,
            runner._pad_dev_to_bucket(torch.cat(srcs)),
            runner._pad_dev_to_bucket(torch.cat(dsts)),
        )
