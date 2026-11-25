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
"""Pure-PyTorch runtime helpers for 2-slot ping-pong GDN MTP execution.

Kept independent of `tpu_runner.py` so the runner subclass (`Qwen38Runner`) and
CPU unit tests can import and compile them without pulling full engine setup.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import torch

INVALID_TOKEN_ID = -1


class SpecBoundaryPlan(NamedTuple):
    """Staged mamba boundary-crossing candidates for one chunk."""

    entries: torch.Tensor
    rows: torch.Tensor | None
    seq_lens: torch.Tensor | None
    block_size: int
    host_entries: torch.Tensor
    chunk_seq_lens: torch.Tensor | None = None


class PingPongMambaChunkState(NamedTuple):
    """Per-chunk 2-slot ping-pong GDN state captured during `execute_model`."""

    state_indices: list[torch.Tensor]
    attn_metadata: dict[str, Any]
    chunk_num_windowed: int
    num_reqs: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    staged_verify_inputs: (
        dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
    ) = None
    staged_ping_pong_state_stack: torch.Tensor | None = None


def two_slot_entries_shape(runner: Any, num_candidates: int) -> tuple[int, int]:
    """Shape of the entries tensor `_plan_two_slot_boundary_copies` stages.

    Shared by `_plan_two_slot_boundary_copies` and `_precompile_two_slot_runtime`
    so warmup compiles the exact entries shapes serving uploads.
    """
    n = max(1, int(num_candidates))
    width = runner._mamba_ckpt_window + 1
    if getattr(runner, "device", None) is not None and runner.device.type == "tpu":
        cand_cap = min(16, max(n, runner.max_num_reqs))
        padded = max(runner._bucket_len(n), runner._bucket_len(cand_cap))
    else:
        padded = min(runner._bucket_len(n), max(n, runner.max_num_reqs))
    return (padded, 3 + len(runner._mamba_copy_plan) * width)


def is_ping_pong_copy_gid(runner: Any, gid: int) -> bool:
    """True when KV cache group `gid` tracks a 2-slot ping-pong Mamba layer."""
    pp_gids = getattr(runner, "_mamba_copy_ping_pong_gids", None)
    if pp_gids and gid in pp_gids:
        return True
    group_windows = getattr(runner, "_mamba_copy_group_windows", None) or {}
    gw = group_windows.get(gid, 0)
    return 1 < gw < getattr(runner, "_mamba_ckpt_window", 1)


def has_ping_pong_copy_groups(runner: Any) -> bool:
    """True when `runner` manages at least one 2-slot ping-pong Mamba group."""
    if bool(getattr(runner, "_mamba_copy_ping_pong_gids", None)):
        return True
    return any(
        is_ping_pong_copy_gid(runner, gid)
        for gid, _ in getattr(runner, "_mamba_copy_plan", ())
    )


def update_ping_pong_read_offsets_impl(
    read_offsets: torch.Tensor,
    groups: list[torch.Tensor],
    query_start_loc: torch.Tensor,
    num_windowed: torch.Tensor,
    num_reqs: torch.Tensor,
) -> None:
    """Flip active 2-slot ping-pong `c_read` (`0 <-> 1`) and zero non-windowed rows."""
    if not groups or read_offsets.numel() == 0:
        return
    width = groups[0].shape[-1]
    row_idx = torch.arange(width, device=read_offsets.device)
    q_lens = query_start_loc[1 : width + 1] - query_start_loc[:width]
    active = (row_idx < num_reqs) & (q_lens > 0)
    is_win = row_idx < num_windowed
    stacked = (
        torch.cat([g if g.ndim == 2 else g.unsqueeze(0) for g in groups], dim=0)
        if len(groups) > 1 or groups[0].ndim == 2
        else groups[0].unsqueeze(0)
    )
    n_off = read_offsets.shape[0]
    in_bounds = (stacked > 0) & (stacked < n_off)
    valid_target = active.unsqueeze(0) & in_bounds
    null = torch.zeros((), dtype=stacked.dtype, device=stacked.device)
    targets = torch.where(valid_target, stacked, null).long()
    old_off = read_offsets[targets].clamp(0, 1)
    flipped = (1 - old_off).to(read_offsets.dtype)
    zeros = torch.zeros_like(flipped)
    new_vals = torch.where(valid_target & is_win.unsqueeze(0), flipped, zeros)
    read_offsets.index_put_((targets.reshape(-1),), new_vals.reshape(-1))


def spec_boundary_pairs_impl(
    entries: torch.Tensor,
    seq_lens: torch.Tensor,
    num_valid: torch.Tensor,
    block_size: int,
    width: int,
    *,
    valid_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[G, C]` (src, dst) boundary copy pairs for standard multi-slot Mamba groups."""
    boundary = entries[:, 1]
    scheduled = entries[:, 2]
    start = seq_lens - scheduled
    reached = (
        (boundary > 0)
        & (scheduled > 0)
        & (start < boundary)
        & (boundary <= start + num_valid)
    )
    if valid_rows is not None:
        reached = reached & valid_rows
    col = (
        (seq_lens - 1) // block_size
        + (boundary - 1 - start)
        - (boundary // block_size - 1)
    )
    blocks = entries[:, 3:].reshape(entries.shape[0], -1, width).transpose(0, 1)
    index = col.clamp(0, width - 1).view(1, -1, 1).expand(blocks.shape[0], -1, 1)
    src = blocks.gather(2, index).squeeze(2)
    dst = blocks[:, :, 0]
    copy = reached & (col >= 0) & (col < width) & (src > 0) & (dst > 0) & (src != dst)
    null = torch.zeros_like(src)
    return torch.where(copy, src, null), torch.where(copy, dst, null)


def expand_split_masked(
    all_src: torch.Tensor,
    all_dst: torch.Tensor,
    split: int,
    out_bucket: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand `(src, dst)` manager blocks by `split`, keeping zero pairs at `(0, 0)`."""
    if split > 1:
        j = torch.arange(split, dtype=all_src.dtype, device=all_src.device)
        active = (all_src.reshape(-1, 1) > 0) & (all_dst.reshape(-1, 1) > 0)
        flat_src = torch.where(
            active,
            all_src.reshape(-1, 1) * split + j,
            torch.zeros((), dtype=all_src.dtype, device=all_src.device),
        ).reshape(-1)
        flat_dst = torch.where(
            active,
            all_dst.reshape(-1, 1) * split + j,
            torch.zeros((), dtype=all_dst.dtype, device=all_dst.device),
        ).reshape(-1)
    else:
        flat_src = all_src.reshape(-1)
        flat_dst = all_dst.reshape(-1)
    if out_bucket is not None:
        pad_len = out_bucket - flat_src.shape[0]
        if pad_len > 0:
            flat_src = torch.nn.functional.pad(flat_src, (0, pad_len))
            flat_dst = torch.nn.functional.pad(flat_dst, (0, pad_len))
    return flat_src, flat_dst


def compute_boundary_copy_and_interior_impl(
    entries: torch.Tensor,
    chunk_seq_lens: torch.Tensor,
    next_tokens: torch.Tensor,
    offsets: torch.Tensor,
    win_scalar: torch.Tensor,
    is_pp_mask: torch.Tensor,
    layer_to_plan_idx: torch.Tensor,
    block_size: int,
    width: int,
    split: int,
    out_copy_bucket: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized `[G, C]` boundary-state copy + interior replay-stack builder.

    Uses a dummy `max_seqs + 1` scatter sink column so zero-padded / inactive
    candidate entries never clobber request row `0`.
    """
    c_rows = entries.shape[0]
    max_seqs = chunk_seq_lens.shape[0]
    rows = entries[:, 0].long()
    valid_seq_row = (rows >= 0) & (rows < max_seqs)
    if max_seqs > 0:
        safe_seq_rows = rows.clamp(0, max_seqs - 1)
        plan_seq_lens = torch.where(
            valid_seq_row,
            chunk_seq_lens.index_select(0, safe_seq_rows).to(torch.int32),
            torch.zeros(c_rows, dtype=torch.int32, device=entries.device),
        )
    else:
        plan_seq_lens = torch.zeros(c_rows, dtype=torch.int32, device=entries.device)

    nt_rows = next_tokens.shape[0]
    valid_nt = valid_seq_row & (rows < nt_rows)
    if nt_rows > 0:
        safe_nt = rows.clamp(0, nt_rows - 1)
        num_valid = torch.where(
            valid_nt,
            (next_tokens.index_select(0, safe_nt) != INVALID_TOKEN_ID)
            .sum(dim=1)
            .to(torch.int32),
            torch.zeros_like(plan_seq_lens),
        )
    else:
        num_valid = torch.zeros_like(plan_seq_lens)

    std_src, std_dst = spec_boundary_pairs_impl(
        entries, plan_seq_lens, num_valid, block_size, width, valid_rows=valid_nt
    )

    boundary = entries[:, 1]
    scheduled = entries[:, 2]
    start = plan_seq_lens - scheduled
    reached = (
        valid_nt
        & (boundary > 0)
        & (scheduled > 0)
        & (start < boundary)
        & (boundary <= start + num_valid)
        & (rows < win_scalar)
    )
    exact_hit = reached & (boundary == start + num_valid)
    interior_hit = reached & (boundary < start + num_valid)

    blocks = entries[:, 3:].reshape(c_rows, -1, width).transpose(0, 1)
    g_cnt = blocks.shape[0]
    raw_state_col_rel = (plan_seq_lens - 1) // block_size - (boundary // block_size - 1)
    col_in_range = (raw_state_col_rel >= 0) & (raw_state_col_rel + 1 < width)
    state_col_rel = raw_state_col_rel.clamp(0, max(0, width - 2))
    idx0 = state_col_rel.view(1, c_rows, 1).expand(g_cnt, c_rows, 1)
    idx1 = (
        (state_col_rel + 1)
        .clamp(max=width - 1)
        .view(1, c_rows, 1)
        .expand(g_cnt, c_rows, 1)
    )
    b0 = blocks.gather(2, idx0).squeeze(2)
    b1 = blocks.gather(2, idx1).squeeze(2)
    dst_b = blocks[:, :, 0]

    n_off = offsets.shape[0]
    valid_pair = (
        col_in_range.unsqueeze(0)
        & (b0 > 0)
        & (b0 < n_off)
        & (b1 > 0)
        & (b1 < n_off)
        & (dst_b > 0)
        & (dst_b < n_off)
    )
    null_gc = torch.zeros_like(dst_b)
    if n_off > 0:
        c_read = offsets[b0.clamp(0, n_off - 1).long()].clamp(0, 1)
    else:
        c_read = null_gc
    write_blk = torch.where(c_read == 0, b1, b0)
    read_blk = torch.where(c_read == 0, b0, b1)
    do_copy = exact_hit.unsqueeze(0) & valid_pair & (write_blk != dst_b)
    pp_src = torch.where(do_copy, write_blk, null_gc)
    pp_dst = torch.where(do_copy, dst_b, null_gc)

    all_src = torch.where(is_pp_mask.unsqueeze(1), pp_src, std_src.to(pp_src.dtype))
    all_dst = torch.where(is_pp_mask.unsqueeze(1), pp_dst, std_dst.to(pp_dst.dtype))
    flat_src, flat_dst = expand_split_masked(all_src, all_dst, split, out_copy_bucket)

    r_l = read_blk.index_select(0, layer_to_plan_idx).to(torch.int32)
    d_l = dst_b.index_select(0, layer_to_plan_idx).to(torch.int32)
    vp_l = valid_pair.index_select(0, layer_to_plan_idx)
    k_1d = (boundary - start).to(torch.int32)
    valid_cand = valid_seq_row & interior_hit & (k_1d > 0)
    act_lc = vp_l & valid_cand.unsqueeze(0)

    num_l = r_l.shape[0]
    dummy_col = torch.full(
        (num_l, c_rows), max_seqs, dtype=torch.int64, device=entries.device
    )
    tgt_cols = torch.where(
        act_lc, rows.clamp(0, max_seqs).unsqueeze(0).expand(num_l, c_rows), dummy_col
    )
    zero_lc = torch.zeros_like(r_l)
    r_pad = torch.zeros(
        (num_l, max_seqs + 1), dtype=torch.int32, device=entries.device
    ).scatter(1, tgt_cols, torch.where(act_lc, r_l, zero_lc))
    d_pad = torch.zeros(
        (num_l, max_seqs + 1), dtype=torch.int32, device=entries.device
    ).scatter(1, tgt_cols, torch.where(act_lc, d_l, zero_lc))
    c_pad = torch.zeros(
        (num_l, max_seqs + 1), dtype=torch.int32, device=entries.device
    ).scatter(
        1,
        tgt_cols,
        torch.where(act_lc, k_1d.unsqueeze(0).expand(num_l, c_rows), zero_lc),
    )
    s_int = r_pad[:, :max_seqs].contiguous()
    ckpt_int = torch.stack([s_int, d_pad[:, :max_seqs]], dim=2).contiguous()
    cl_int = c_pad[:, :max_seqs].contiguous()
    return flat_src, flat_dst, s_int, ckpt_int, cl_int


def compute_boundary_copy_no_verify_impl(
    entries: torch.Tensor,
    chunk_seq_lens: torch.Tensor,
    offsets: torch.Tensor,
    win_scalar: torch.Tensor,
    is_pp_mask: torch.Tensor,
    has_win: bool,
    block_size: int,
    width: int,
    split: int,
    out_copy_bucket: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized `[G, C]` boundary-state copy pair builder for plain 1-token decode."""
    c_rows = entries.shape[0]
    max_seqs = chunk_seq_lens.shape[0]
    rows = entries[:, 0].long()
    valid_seq_row = (rows >= 0) & (rows < max_seqs)
    if max_seqs > 0:
        safe_seq_rows = rows.clamp(0, max_seqs - 1)
        plan_seq_lens = torch.where(
            valid_seq_row,
            chunk_seq_lens.index_select(0, safe_seq_rows).to(torch.int32),
            torch.zeros(c_rows, dtype=torch.int32, device=entries.device),
        )
    else:
        plan_seq_lens = torch.zeros(c_rows, dtype=torch.int32, device=entries.device)
    num_valid = torch.ones_like(plan_seq_lens)

    boundary = entries[:, 1]
    scheduled = entries[:, 2]
    start = plan_seq_lens - scheduled
    reached = (
        valid_seq_row
        & (boundary > 0)
        & (scheduled > 0)
        & (start < boundary)
        & (boundary <= start + num_valid)
    )
    if has_win:
        reached = reached & (rows < win_scalar)
    exact_hit = reached & (boundary == start + num_valid)

    blocks = entries[:, 3:].reshape(c_rows, -1, width).transpose(0, 1)
    g_cnt = blocks.shape[0]
    raw_state_col_rel = (plan_seq_lens - 1) // block_size - (boundary // block_size - 1)
    col_in_range = (raw_state_col_rel >= 0) & (raw_state_col_rel + 1 < width)
    state_col_rel = raw_state_col_rel.clamp(0, max(0, width - 2))
    idx0 = state_col_rel.view(1, c_rows, 1).expand(g_cnt, c_rows, 1)
    idx1 = (
        (state_col_rel + 1)
        .clamp(max=width - 1)
        .view(1, c_rows, 1)
        .expand(g_cnt, c_rows, 1)
    )
    b0 = blocks.gather(2, idx0).squeeze(2)
    b1 = blocks.gather(2, idx1).squeeze(2)
    dst_b = blocks[:, :, 0]

    n_off = offsets.shape[0]
    valid_pair = (
        col_in_range.unsqueeze(0)
        & (b0 > 0)
        & (b0 < n_off)
        & (b1 > 0)
        & (b1 < n_off)
        & (dst_b > 0)
        & (dst_b < n_off)
    )
    null_gc = torch.zeros_like(dst_b)
    if n_off > 0:
        c_read = offsets[b0.clamp(0, n_off - 1).long()].clamp(0, 1)
    else:
        c_read = null_gc
    write_blk = torch.where(c_read == 0, b1, b0)
    do_copy = exact_hit.unsqueeze(0) & valid_pair & (write_blk != dst_b)
    pp_src = torch.where(do_copy, write_blk, null_gc)
    pp_dst = torch.where(do_copy, dst_b, null_gc)

    all_src = torch.where(is_pp_mask.unsqueeze(1), pp_src, null_gc)
    all_dst = torch.where(is_pp_mask.unsqueeze(1), pp_dst, null_gc)
    return expand_split_masked(all_src, all_dst, split, out_copy_bucket)


def compute_replay_commit_lens(
    query_start_loc: torch.Tensor,
    next_tokens: torch.Tensor,
    num_windowed: int,
) -> torch.Tensor:
    """Compute per-sequence replay prefix length (`0` = no replay needed).

    A windowed request needs replay iff ``0 < num_valid < q_len`` (partial draft
    rejection). All-accepted (`num_valid == q_len`) keeps the optimistic
    post-window state already written to `1 - c_read` by the verify forward.
    """
    q_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int32)
    max_seqs = q_lens.shape[0]
    raw_valid = (next_tokens != INVALID_TOKEN_ID).sum(dim=1).to(torch.int32)
    if raw_valid.shape[0] >= max_seqs:
        num_valid = raw_valid[:max_seqs]
    else:
        num_valid = torch.nn.functional.pad(
            raw_valid, (0, max_seqs - raw_valid.shape[0])
        )
    row_idx = torch.arange(max_seqs, device=q_lens.device)
    is_win = row_idx < num_windowed
    rejected = is_win & (num_valid < q_lens) & (num_valid > 0)
    return torch.where(rejected, num_valid, torch.zeros_like(q_lens))


def _tpu_compile_fn(fn):
    return torch.compile(backend="tpu", fullgraph=True, dynamic=False)(fn)


update_ping_pong_read_offsets_compiled = _tpu_compile_fn(
    update_ping_pong_read_offsets_impl
)
spec_boundary_pairs_compiled = _tpu_compile_fn(spec_boundary_pairs_impl)
compute_boundary_copy_and_interior_compiled = _tpu_compile_fn(
    compute_boundary_copy_and_interior_impl
)
compute_boundary_copy_no_verify_compiled = _tpu_compile_fn(
    compute_boundary_copy_no_verify_impl
)
