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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_WIN_ROWS = 128
_REPLICA_MIN_GAP_BY_TOKENS = {8192: 256, 16384: 384}


def default_replica_min_gap(tokens: int) -> int:
    if tokens not in _REPLICA_MIN_GAP_BY_TOKENS:
        raise ValueError(f"Unsupported tokens for replica min_gap: {tokens}")
    return _REPLICA_MIN_GAP_BY_TOKENS[tokens]


@dataclass(frozen=True)
class CompactMetadataSchedule:
    steps: tuple[Callable[[], None], ...]
    tail: Callable[[], None]


@dataclass(frozen=True)
class ReplicaRoutingScratch:
    group_loads: Any
    my_drop_experts: Any
    my_add_experts: Any
    my_add_slots: Any
    my_counts: Any


def _bitonic_partner(value: jax.Array, distance: int, rows: int) -> jax.Array:
    shape = (rows, 128)
    if distance >= 1024:
        block_xor = distance // 1024
        return jnp.concatenate(
            [
                value[8 * (i ^ block_xor) : 8 * (i ^ block_xor) + 8]
                for i in range(rows // 8)
            ],
            axis=0,
        )
    if distance >= 128:
        shape3 = (rows // 8, 8, 128)
        row_group = jax.lax.broadcasted_iota(jnp.int32, shape3, 1)
        return jnp.take_along_axis(
            value.reshape(shape3),
            row_group ^ (distance // 128),
            axis=1,
            mode="promise_in_bounds",
        ).reshape(shape)
    lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    return jnp.take_along_axis(value, lane ^ distance, axis=1, mode="promise_in_bounds")


def _apply_in_window_steps(
    keys_tile: jax.Array,
    probs_tile: jax.Array,
    ordinal: jax.Array | int,
    steps: tuple[tuple[int, int], ...],
    *,
    window: int,
    rows: int,
) -> tuple[jax.Array, jax.Array]:
    shape = (rows, 128)
    row = jax.lax.broadcasted_iota(jnp.int32, shape, 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, shape, 1)
    global_position = row * 128 + lane + ordinal * window
    for distance, group in steps:
        partner_k = _bitonic_partner(keys_tile, distance, rows)
        partner_p = _bitonic_partner(probs_tile, distance, rows)
        descending = (global_position & group) != 0
        upper = (global_position & distance) != 0
        take_min = ~(descending ^ upper)
        keep = ~((keys_tile < partner_k) ^ take_min)
        keys_tile = jnp.where(keep, keys_tile, partner_k)
        probs_tile = jnp.where(keep, probs_tile, partner_p)
    return keys_tile, probs_tile


def _make_unrolled_flush_step(
    keys,
    probabilities,
    *,
    ordinal: int,
    steps: tuple[tuple[int, int], ...],
    window: int,
    rows: int,
) -> Callable[[], None]:
    def _step() -> None:
        off = ordinal * rows
        k_ref = keys.at[pl.ds(off, rows), :]
        p_ref = probabilities.at[pl.ds(off, rows), :]
        k_out, p_out = _apply_in_window_steps(
            k_ref[...], p_ref[...], ordinal, steps, window=window, rows=rows
        )
        k_ref[...], p_ref[...] = k_out, p_out

    return _step


def _make_unrolled_pair_step(
    keys,
    probabilities,
    token_ids,
    *,
    low: int,
    high: int,
    group: int,
    tail_steps: tuple[tuple[int, int], ...],
    is_final: bool,
    window: int,
    rows: int,
    mask: int,
    write_final_keys: bool = True,
) -> Callable[[], None]:
    def _step() -> None:
        off_l, off_h = low * rows, high * rows
        k_lo = keys.at[pl.ds(off_l, rows), :]
        k_hi = keys.at[pl.ds(off_h, rows), :]
        p_lo = probabilities.at[pl.ds(off_l, rows), :]
        p_hi = probabilities.at[pl.ds(off_h, rows), :]
        x, y, p, q = k_lo[...], k_hi[...], p_lo[...], p_hi[...]
        ascending = ((low * window) & group) == 0
        keep = ~((x < y) ^ ascending)
        x_lo, x_hi = jnp.where(keep, x, y), jnp.where(keep, y, x)
        p_lo_v, p_hi_v = jnp.where(keep, p, q), jnp.where(keep, q, p)
        x_lo, p_lo_v = _apply_in_window_steps(
            x_lo, p_lo_v, low, tail_steps, window=window, rows=rows
        )
        if (not is_final) or write_final_keys:
            k_lo[...] = x_lo
        p_lo[...] = p_lo_v
        if is_final:
            token_ids[pl.ds(off_l, rows), :] = (x_lo & mask) // 16
        x_hi, p_hi_v = _apply_in_window_steps(
            x_hi, p_hi_v, high, tail_steps, window=window, rows=rows
        )
        if (not is_final) or write_final_keys:
            k_hi[...] = x_hi
        p_hi[...] = p_hi_v
        if is_final:
            token_ids[pl.ds(off_h, rows), :] = (x_hi & mask) // 16

    return _step


def _make_unrolled_stride_step(
    keys,
    probabilities,
    *,
    low: int,
    high: int,
    group: int,
    window: int,
    rows: int,
) -> Callable[[], None]:
    def _step() -> None:
        off_l, off_h = low * rows, high * rows
        k_lo = keys.at[pl.ds(off_l, rows), :]
        k_hi = keys.at[pl.ds(off_h, rows), :]
        p_lo = probabilities.at[pl.ds(off_l, rows), :]
        p_hi = probabilities.at[pl.ds(off_h, rows), :]
        x, y, p, q = k_lo[...], k_hi[...], p_lo[...], p_hi[...]
        ascending = ((low * window) & group) == 0
        keep = ~((x < y) ^ ascending)
        k_lo[...], p_lo[...] = jnp.where(keep, x, y), jnp.where(keep, p, q)
        k_hi[...], p_hi[...] = jnp.where(keep, y, x), jnp.where(keep, q, p)

    return _step


def _bitonic_sort_unrolled_t0_steps(
    keys,
    probabilities,
    token_ids,
    *,
    size: int,
    bits: int,
    window: int = 4096,
    rows: int = 32,
    write_final_keys: bool = True,
) -> list[Callable[[], None]]:
    assert size >= 2 * window and (size & (size - 1)) == 0
    mask = (1 << bits) - 1
    steps_out: list[Callable[[], None]] = []
    pending: list[tuple[int, int]] = []

    def emit_flush() -> None:
        if not pending:
            return
        steps_t = tuple(pending)
        pending.clear()
        for ordinal in range(size // window):
            steps_out.append(
                _make_unrolled_flush_step(
                    keys,
                    probabilities,
                    ordinal=ordinal,
                    steps=steps_t,
                    window=window,
                    rows=rows,
                )
            )

    for stage in range(1, size.bit_length()):
        group = 1 << stage
        for shift in reversed(range(stage)):
            distance = 1 << shift
            if distance < window:
                pending.append((distance, group))
                continue
            emit_flush()
            if distance == window:
                tail_steps = tuple((1 << s, group) for s in reversed(range(shift)))
                is_final = (group == size) and (token_ids is not None)
                for ordinal in range(size // (2 * window)):
                    low = 2 * ordinal
                    steps_out.append(
                        _make_unrolled_pair_step(
                            keys,
                            probabilities,
                            token_ids,
                            low=low,
                            high=low + 1,
                            group=group,
                            tail_steps=tail_steps,
                            is_final=is_final,
                            window=window,
                            rows=rows,
                            mask=mask,
                            write_final_keys=write_final_keys,
                        )
                    )
                break
            stride = distance // window
            for ordinal in range(size // (2 * window)):
                low = (ordinal // stride) * (2 * stride) + (ordinal % stride)
                steps_out.append(
                    _make_unrolled_stride_step(
                        keys,
                        probabilities,
                        low=low,
                        high=low + stride,
                        group=group,
                        window=window,
                        rows=rows,
                    )
                )
    emit_flush()
    return steps_out


def _make_looped_flush_stage(
    keys,
    probabilities,
    *,
    num_windows: int,
    steps: tuple[tuple[int, int], ...],
    window: int,
    rows: int,
) -> Callable[[], None]:
    def _stage() -> None:
        @pl.loop(0, num_windows)
        def _win(ordinal) -> None:
            off = pl.multiple_of(ordinal * rows, rows)
            k_ref = keys.at[pl.ds(off, rows), :]
            p_ref = probabilities.at[pl.ds(off, rows), :]
            k_out, p_out = _apply_in_window_steps(
                k_ref[...], p_ref[...], ordinal, steps, window=window, rows=rows
            )
            k_ref[...], p_ref[...] = k_out, p_out

    return _stage


def _make_looped_pair_stage(
    keys,
    probabilities,
    token_ids,
    *,
    pair_limit: int,
    group: int,
    tail_steps: tuple[tuple[int, int], ...],
    is_final: bool,
    window: int,
    rows: int,
    mask: int,
) -> Callable[[], None]:
    def _stage() -> None:
        @pl.loop(0, pair_limit)
        def _pair(ordinal) -> None:
            low = 2 * ordinal
            high = low + 1
            off_l = pl.multiple_of(low * rows, rows)
            off_h = pl.multiple_of(high * rows, rows)
            k_lo = keys.at[pl.ds(off_l, rows), :]
            k_hi = keys.at[pl.ds(off_h, rows), :]
            p_lo = probabilities.at[pl.ds(off_l, rows), :]
            p_hi = probabilities.at[pl.ds(off_h, rows), :]
            x, y, p, q = k_lo[...], k_hi[...], p_lo[...], p_hi[...]
            ascending = ((low * window) & group) == 0
            keep = ~((x < y) ^ ascending)
            x_lo, x_hi = jnp.where(keep, x, y), jnp.where(keep, y, x)
            p_lo_v, p_hi_v = jnp.where(keep, p, q), jnp.where(keep, q, p)
            x_lo, p_lo_v = _apply_in_window_steps(
                x_lo, p_lo_v, low, tail_steps, window=window, rows=rows
            )
            if is_final:
                token_ids[pl.ds(off_l, rows), :] = (x_lo & mask) // 16
            else:
                k_lo[...] = x_lo
            p_lo[...] = p_lo_v
            x_hi, p_hi_v = _apply_in_window_steps(
                x_hi, p_hi_v, high, tail_steps, window=window, rows=rows
            )
            if is_final:
                token_ids[pl.ds(off_h, rows), :] = (x_hi & mask) // 16
            else:
                k_hi[...] = x_hi
            p_hi[...] = p_hi_v

    return _stage


def _make_looped_stride_stage(
    keys,
    probabilities,
    *,
    pair_limit: int,
    stride: int,
    group: int,
    window: int,
    rows: int,
) -> Callable[[], None]:
    def _stage() -> None:
        @pl.loop(0, pair_limit)
        def _stride_pair(ordinal) -> None:
            low = (ordinal // stride) * (2 * stride) + (ordinal % stride)
            high = low + stride
            off_l = pl.multiple_of(low * rows, rows)
            off_h = pl.multiple_of(high * rows, rows)
            k_lo = keys.at[pl.ds(off_l, rows), :]
            k_hi = keys.at[pl.ds(off_h, rows), :]
            p_lo = probabilities.at[pl.ds(off_l, rows), :]
            p_hi = probabilities.at[pl.ds(off_h, rows), :]
            x, y, p, q = k_lo[...], k_hi[...], p_lo[...], p_hi[...]
            ascending = ((low * window) & group) == 0
            keep = ~((x < y) ^ ascending)
            k_lo[...], p_lo[...] = jnp.where(keep, x, y), jnp.where(keep, p, q)
            k_hi[...], p_hi[...] = jnp.where(keep, y, x), jnp.where(keep, q, p)

    return _stage


def _bitonic_sort_looped_stages(
    keys,
    probabilities,
    token_ids,
    *,
    size: int,
    bits: int,
) -> list[Callable[[], None]]:
    window, rows = (8192, 64) if size >= 16384 else (4096, 32)
    assert size >= 2 * window and (size & (size - 1)) == 0
    mask = (1 << bits) - 1
    stages: list[Callable[[], None]] = []
    pending: list[tuple[int, int]] = []

    def emit_flush() -> None:
        if not pending:
            return
        steps_t = tuple(pending)
        pending.clear()
        stages.append(
            _make_looped_flush_stage(
                keys,
                probabilities,
                num_windows=size // window,
                steps=steps_t,
                window=window,
                rows=rows,
            )
        )

    for stage in range(1, size.bit_length()):
        group = 1 << stage
        for shift in reversed(range(stage)):
            distance = 1 << shift
            if distance < window:
                pending.append((distance, group))
                continue
            emit_flush()
            if distance == window:
                tail_steps = tuple((1 << s, group) for s in reversed(range(shift)))
                is_final = (group == size) and (token_ids is not None)
                stages.append(
                    _make_looped_pair_stage(
                        keys,
                        probabilities,
                        token_ids,
                        pair_limit=size // (2 * window),
                        group=group,
                        tail_steps=tail_steps,
                        is_final=is_final,
                        window=window,
                        rows=rows,
                        mask=mask,
                    )
                )
                break
            stages.append(
                _make_looped_stride_stage(
                    keys,
                    probabilities,
                    pair_limit=size // (2 * window),
                    stride=distance // window,
                    group=group,
                    window=window,
                    rows=rows,
                )
            )
    emit_flush()
    return stages


def _left_pack_128x128_matmul(
    keys_tile: jax.Array,
    probs_tile: jax.Array,
    valid_mask: jax.Array,
    *,
    sentinel_key: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    row = jax.lax.broadcasted_iota(jnp.int32, (_WIN_ROWS, 128), 0)
    lane = jax.lax.broadcasted_iota(jnp.int32, (_WIN_ROWS, 128), 1)
    tri_upper = (row <= lane).astype(jnp.bfloat16)
    tri_strict_lower = (row > lane).astype(jnp.bfloat16)

    valid_bf16 = valid_mask.astype(jnp.bfloat16)
    row_cumsum = jnp.matmul(
        valid_bf16, tri_upper, preferred_element_type=jnp.float32
    ).astype(jnp.int32)
    row_totals = row_cumsum[:, 127:128]
    col_excl = jnp.matmul(
        tri_strict_lower,
        row_totals.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    ).astype(jnp.int32)
    n_live = col_excl[127, 0] + row_totals[127, 0]

    rank_out = col_excl + (row_cumsum - 1)
    pos = row * 128 + lane
    shift = jnp.where(valid_mask, pos - rank_out, 0)
    packed_k = jnp.where(valid_mask, keys_tile, sentinel_key)
    packed_p = probs_tile

    zero_1 = jnp.zeros((1, 128), jnp.int32)
    zero_1f = jnp.zeros((1, 128), jnp.float32)
    sent_1 = jnp.full((1, 128), sentinel_key, jnp.int32)

    for bit in range(7):
        dist = 1 << bit
        src_lane = (lane + dist) & 127
        same_row = lane >= dist
        sh_pull = jnp.take_along_axis(
            jnp.where(same_row, shift, jnp.concatenate([shift[1:], zero_1], axis=0)),
            src_lane,
            axis=1,
            mode="promise_in_bounds",
        )
        k_pull = jnp.take_along_axis(
            jnp.where(
                same_row, packed_k, jnp.concatenate([packed_k[1:], sent_1], axis=0)
            ),
            src_lane,
            axis=1,
            mode="promise_in_bounds",
        )
        p_pull = jnp.take_along_axis(
            jnp.where(
                same_row, packed_p, jnp.concatenate([packed_p[1:], zero_1f], axis=0)
            ),
            src_lane,
            axis=1,
            mode="promise_in_bounds",
        )
        pull_moves = (sh_pull & dist) != 0
        self_moves = (shift & dist) != 0
        packed_k = jnp.where(pull_moves, k_pull, packed_k)
        packed_p = jnp.where(pull_moves, p_pull, packed_p)
        shift = jnp.where(pull_moves, sh_pull, jnp.where(self_moves, 0, shift))

    for bit in range(7, 14):
        dist = 1 << bit
        delta_rows = dist // 128
        sh_pull = jnp.concatenate(
            [shift[delta_rows:], jnp.zeros((delta_rows, 128), jnp.int32)], axis=0
        )
        k_pull = jnp.concatenate(
            [
                packed_k[delta_rows:],
                jnp.full((delta_rows, 128), sentinel_key, jnp.int32),
            ],
            axis=0,
        )
        p_pull = jnp.concatenate(
            [packed_p[delta_rows:], jnp.zeros((delta_rows, 128), jnp.float32)], axis=0
        )
        pull_moves = (sh_pull & dist) != 0
        self_moves = (shift & dist) != 0
        packed_k = jnp.where(pull_moves, k_pull, packed_k)
        packed_p = jnp.where(pull_moves, p_pull, packed_p)
        shift = jnp.where(pull_moves, sh_pull, jnp.where(self_moves, 0, shift))

    packed_k = jnp.where(pos < n_live, packed_k, sentinel_key)
    return packed_k, packed_p, n_live


def _append_window_rows(
    keys,
    probabilities,
    tmp_k,
    tmp_p_i32,
    *,
    base_row: jax.Array,
    n_rows: jax.Array,
) -> None:
    full_blocks = n_rows // 8

    @pl.loop(0, full_blocks)
    def _append8(block_idx) -> None:
        r0 = block_idx * 8
        keys[pl.ds(base_row + r0, 8), :] = tmp_k[pl.ds(r0, 8), :]
        probabilities[pl.ds(base_row + r0, 8), :] = jax.lax.bitcast_convert_type(
            tmp_p_i32[pl.ds(r0, 8), :], jnp.float32
        )

    @pl.loop(full_blocks * 8, n_rows)
    def _append1(row_idx) -> None:
        keys[pl.ds(base_row + row_idx, 1), :] = tmp_k[pl.ds(row_idx, 1), :]
        probabilities[pl.ds(base_row + row_idx, 1), :] = jax.lax.bitcast_convert_type(
            tmp_p_i32[pl.ds(row_idx, 1), :], jnp.float32
        )


def _pad_compact_rows_to_tier(
    keys,
    probabilities,
    *,
    used_rows: jax.Array,
    tier_rows: jax.Array,
    sentinel_key: int,
) -> None:
    sent_1 = jnp.full((1, 128), sentinel_key, jnp.int32)
    zero_1 = jnp.zeros((1, 128), jnp.float32)
    sent_8 = jnp.full((8, 128), sentinel_key, jnp.int32)
    zero_8 = jnp.zeros((8, 128), jnp.float32)
    rem_head = jnp.minimum((8 - (used_rows & 7)) & 7, tier_rows - used_rows)

    @pl.loop(0, rem_head)
    def _pad_head(i) -> None:
        r = used_rows + i
        keys[pl.ds(r, 1), :] = sent_1
        probabilities[pl.ds(r, 1), :] = zero_1

    aligned_start = used_rows + rem_head
    n_pad8 = (tier_rows - aligned_start) // 8

    @pl.loop(0, n_pad8)
    def _pad_body8(b) -> None:
        r = aligned_start + b * 8
        keys[pl.ds(r, 8), :] = sent_8
        probabilities[pl.ds(r, 8), :] = zero_8


def _count_compact_local_experts(
    keys,
    counts,
    *,
    used_rows: jax.Array,
    bits: int,
    local_experts: int,
) -> None:
    counts[...] = jnp.zeros((local_experts,), jnp.int32)
    expert_index = jnp.arange(local_experts, dtype=jnp.int32)
    n_cnt_tiles = pl.cdiv(used_rows, 32)

    @pl.loop(0, n_cnt_tiles)
    def _cnt_tile(tile_idx) -> None:
        block = keys[pl.ds(tile_idx * 32, 32), :] >> bits
        tile_counts = counts[...]
        for expert in range(local_experts):
            cnt = jnp.sum((block == expert).astype(jnp.int32))
            tile_counts = tile_counts + jnp.where(expert_index == expert, cnt, 0)
        counts[...] = tile_counts


def _make_window_pack_step(
    wire,
    keys,
    probabilities,
    tmp_k,
    tmp_p_i32,
    write_cursor,
    total_groups_box: list[jax.Array],
    group_index: jax.Array,
    *,
    win_idx: int,
    ranks_per_win: int,
    rank_stride: int,
    rank_half: int,
    rank,
    expert_tp: int,
    local_experts: int,
    ep_groups: int,
    bits: int,
    sentinel_key: int,
    track_maximum: bool,
) -> Callable[[], None]:
    def _step() -> None:
        if win_idx == 0:
            write_cursor[0] = 0
        id_parts, prob_parts = [], []
        for rank_off in range(ranks_per_win):
            rank_base = (win_idx * ranks_per_win + rank_off) * rank_stride
            id_parts.append(wire[pl.ds(rank_base, rank_half), :])
            prob_parts.append(
                jax.lax.bitcast_convert_type(
                    wire[pl.ds(rank_base + rank_half, rank_half), :], jnp.float32
                )
            )
        ids_win = jnp.concatenate(id_parts, axis=0)
        probs_win = jnp.concatenate(prob_parts, axis=0)
        live_win = probs_win != 0

        if track_maximum:
            groups = ids_win // local_experts
            tile_groups = jnp.zeros((ep_groups,), jnp.int32)
            for grp in range(ep_groups):
                cnt = jnp.sum((live_win & (groups == grp)).astype(jnp.int32))
                tile_groups = jnp.where(group_index == grp, cnt, tile_groups)
            total_groups_box[0] = total_groups_box[0] + tile_groups

        local = ids_win - (rank // expert_tp * local_experts)
        valid = live_win & (local >= 0) & (local < local_experts)
        local_or_sentinel = jnp.where(valid, local, local_experts)
        win_pos = jax.lax.broadcasted_iota(
            jnp.int32, (_WIN_ROWS, 128), 0
        ) * 128 + jax.lax.broadcasted_iota(jnp.int32, (_WIN_ROWS, 128), 1)
        packed = (local_or_sentinel << bits) | (win_idx * (_WIN_ROWS * 128) + win_pos)
        packed_k, packed_p, n_live = _left_pack_128x128_matmul(
            packed, probs_win, valid, sentinel_key=sentinel_key
        )
        tmp_k[...] = packed_k
        tmp_p_i32[...] = jax.lax.bitcast_convert_type(packed_p, jnp.int32)

        n_rows = (n_live + 127) // 128
        base_row = write_cursor[0]
        _append_window_rows(
            keys,
            probabilities,
            tmp_k,
            tmp_p_i32,
            base_row=base_row,
            n_rows=n_rows,
        )
        write_cursor[0] = base_row + n_rows

    return _step


def _make_replica_window_pack_step(
    wire,
    keys,
    probabilities,
    tmp_k,
    tmp_p_i32,
    tmp_loc,
    write_cursor,
    replica_table,
    replica_scratch: ReplicaRoutingScratch,
    *,
    win_idx: int,
    ranks_per_win: int,
    rank_stride: int,
    rank_half: int,
    rank,
    expert_tp: int,
    base_local_experts: int,
    effective_local_experts: int,
    ep_groups: int,
    n_rep: int,
    min_gap: int,
    bits: int,
    sentinel_key: int,
) -> Callable[[], None]:
    my_group = rank // expert_tp
    group_loads = replica_scratch.group_loads
    my_drop_experts = replica_scratch.my_drop_experts
    my_add_experts = replica_scratch.my_add_experts
    my_add_slots = replica_scratch.my_add_slots
    my_counts = replica_scratch.my_counts

    def _step() -> None:
        if win_idx == 0:
            write_cursor[0] = 0
            for grp in range(ep_groups):
                group_loads[grp] = 0
        id_parts, prob_parts = [], []
        for rank_off in range(ranks_per_win):
            rank_base = (win_idx * ranks_per_win + rank_off) * rank_stride
            id_parts.append(wire[pl.ds(rank_base, rank_half), :])
            prob_parts.append(
                jax.lax.bitcast_convert_type(
                    wire[pl.ds(rank_base + rank_half, rank_half), :], jnp.float32
                )
            )
        ids_win = jnp.concatenate(id_parts, axis=0)
        probs_win = jnp.concatenate(prob_parts, axis=0)
        live_win = probs_win != 0
        ids_live = jnp.where(live_win, ids_win, -1)
        groups = jnp.where(live_win, ids_win // base_local_experts, -1)

        max_l = jnp.int32(-1)
        min_l = jnp.int32(1 << 30)
        for grp in range(ep_groups):
            cnt = jnp.sum((groups == grp).astype(jnp.int32))
            updated = group_loads[grp] + cnt
            group_loads[grp] = updated
            max_l = jnp.maximum(max_l, updated)
            min_l = jnp.minimum(min_l, updated)

        tmp_k[...] = ids_live
        tmp_p_i32[...] = jax.lax.bitcast_convert_type(probs_win, jnp.int32)
        my_counts[0] = 0
        my_counts[1] = 0

        @pl.when((max_l - min_l) >= min_gap)
        def _rebalance_window() -> None:
            @pl.loop(0, n_rep)
            def _rep_loop(k) -> None:
                hg = replica_table[n_rep + k]
                rg = replica_table[2 * n_rep + k]
                l_hg = group_loads[hg]
                l_rg = group_loads[rg]
                gap = l_hg - l_rg

                @pl.when(gap >= min_gap)
                def _try_move() -> None:
                    he = replica_table[k]

                    @pl.when(he >= 0)
                    def _count_and_move() -> None:
                        mv = jnp.sum((tmp_k[...] == he).astype(jnp.int32))

                        @pl.when((mv > 0) & (mv <= gap))
                        def _commit() -> None:
                            group_loads[hg] = l_hg - mv
                            group_loads[rg] = l_rg + mv

                            @pl.when(hg == my_group)
                            def _rec_drop() -> None:
                                idx = my_counts[0]
                                my_drop_experts[idx] = he
                                my_counts[0] = idx + 1

                            @pl.when(rg == my_group)
                            def _rec_add() -> None:
                                idx = my_counts[1]
                                my_add_experts[idx] = he
                                my_add_slots[idx] = replica_table[3 * n_rep + k]
                                my_counts[1] = idx + 1

        ids_v = tmp_k[...]
        local = ids_v - (my_group * base_local_experts)
        valid0 = (ids_v >= 0) & (local >= 0) & (local < base_local_experts)
        tmp_loc[...] = jnp.where(valid0, local, effective_local_experts)

        @pl.loop(0, my_counts[0])
        def _apply_drop(di) -> None:
            de = my_drop_experts[di]
            tmp_loc[...] = jnp.where(
                tmp_k[...] == de, effective_local_experts, tmp_loc[...]
            )

        @pl.loop(0, my_counts[1])
        def _apply_add(ai) -> None:
            ae = my_add_experts[ai]
            aslot = my_add_slots[ai]
            tmp_loc[...] = jnp.where(tmp_k[...] == ae, aslot, tmp_loc[...])

        local_or_sentinel = tmp_loc[...]
        valid = local_or_sentinel < effective_local_experts
        probs_cur = jax.lax.bitcast_convert_type(tmp_p_i32[...], jnp.float32)
        win_pos = jax.lax.broadcasted_iota(
            jnp.int32, (_WIN_ROWS, 128), 0
        ) * 128 + jax.lax.broadcasted_iota(jnp.int32, (_WIN_ROWS, 128), 1)
        packed = (local_or_sentinel << bits) | (win_idx * (_WIN_ROWS * 128) + win_pos)
        packed_k, packed_p, n_live = _left_pack_128x128_matmul(
            packed, probs_cur, valid, sentinel_key=sentinel_key
        )
        tmp_k[...] = packed_k
        tmp_p_i32[...] = jax.lax.bitcast_convert_type(packed_p, jnp.int32)

        n_rows = (n_live + 127) // 128
        base_row = write_cursor[0]
        _append_window_rows(
            keys,
            probabilities,
            tmp_k,
            tmp_p_i32,
            base_row=base_row,
            n_rows=n_rows,
        )
        write_cursor[0] = base_row + n_rows

    return _step


def build_compact_metadata_schedule(
    wire,
    keys,
    probabilities,
    token_ids,
    counts,
    maximum,
    write_cursor,
    *,
    tokens: int,
    rank,
    counts_ready: Callable[[], None],
    local_experts: int = 56,
    on_tail_stage: Callable[[], None] | None = None,
    replica_table=None,
    replica_slots: int = 0,
    replica_scratch: ReplicaRoutingScratch | None = None,
    replica_min_gap: int | None = None,
) -> CompactMetadataSchedule:
    assert tokens in (8192, 16384)
    if replica_table is not None:
        if replica_slots <= 0 or replica_scratch is None:
            raise ValueError(
                "replica_table requires replica_slots > 0 and replica_scratch"
            )
        base_local_experts = local_experts - replica_slots
    else:
        base_local_experts = local_experts
    ep_groups = 896 // base_local_experts
    expert_tp = 32 // ep_groups
    bits = (tokens * 16 - 1).bit_length()
    sentinel_key = (local_experts << bits) | ((1 << bits) - 1)
    total_rows = tokens // 8
    n_wins = total_rows // _WIN_ROWS
    rank_half = tokens // 256
    rank_stride = tokens // 128
    ranks_per_win = _WIN_ROWS // rank_half

    tmp_k = token_ids.at[pl.ds(0, _WIN_ROWS), :]
    tmp_p_i32 = token_ids.at[pl.ds(_WIN_ROWS, _WIN_ROWS), :]
    group_index = jnp.arange(ep_groups, dtype=jnp.int32)
    total_groups_box = [jnp.zeros((ep_groups,), jnp.int32)]

    if replica_table is not None:
        assert replica_scratch is not None
        tmp_loc = token_ids.at[pl.ds(2 * _WIN_ROWS, _WIN_ROWS), :]
        n_rep = ep_groups * replica_slots
        min_gap = (
            default_replica_min_gap(tokens)
            if replica_min_gap is None
            else replica_min_gap
        )
        steps: list[Callable[[], None]] = [
            _make_replica_window_pack_step(
                wire,
                keys,
                probabilities,
                tmp_k,
                tmp_p_i32,
                tmp_loc,
                write_cursor,
                replica_table,
                replica_scratch,
                win_idx=win_idx,
                ranks_per_win=ranks_per_win,
                rank_stride=rank_stride,
                rank_half=rank_half,
                rank=rank,
                expert_tp=expert_tp,
                base_local_experts=base_local_experts,
                effective_local_experts=local_experts,
                ep_groups=ep_groups,
                n_rep=n_rep,
                min_gap=min_gap,
                bits=bits,
                sentinel_key=sentinel_key,
            )
            for win_idx in range(n_wins)
        ]
    else:
        steps = [
            _make_window_pack_step(
                wire,
                keys,
                probabilities,
                tmp_k,
                tmp_p_i32,
                write_cursor,
                total_groups_box,
                group_index,
                win_idx=win_idx,
                ranks_per_win=ranks_per_win,
                rank_stride=rank_stride,
                rank_half=rank_half,
                rank=rank,
                expert_tp=expert_tp,
                local_experts=local_experts,
                ep_groups=ep_groups,
                bits=bits,
                sentinel_key=sentinel_key,
                track_maximum=maximum is not None,
            )
            for win_idx in range(n_wins)
        ]

    t0_rows = 256 if tokens == 16384 else 128
    t1_rows = t0_rows * 2
    t2_rows = t0_rows * 4
    t3_rows = t0_rows * 8

    def _pad_and_count() -> None:
        used_rows = write_cursor[0]
        tier_rows = jnp.where(
            used_rows <= t0_rows,
            t0_rows,
            jnp.where(
                used_rows <= t1_rows,
                t1_rows,
                jnp.where(used_rows <= t2_rows, t2_rows, t3_rows),
            ),
        )
        _pad_compact_rows_to_tier(
            keys,
            probabilities,
            used_rows=used_rows,
            tier_rows=tier_rows,
            sentinel_key=sentinel_key,
        )
        _count_compact_local_experts(
            keys,
            counts,
            used_rows=used_rows,
            bits=bits,
            local_experts=local_experts,
        )
        if maximum is not None:
            if replica_table is not None:
                assert replica_scratch is not None
                mx = replica_scratch.group_loads[0]
                for grp in range(1, ep_groups):
                    mx = jnp.maximum(mx, replica_scratch.group_loads[grp])
                maximum[...] = mx[None]
            else:
                maximum[...] = jnp.max(total_groups_box[0])[None]
        counts_ready()

    steps.append(_pad_and_count)
    steps.extend(
        _bitonic_sort_unrolled_t0_steps(
            keys, probabilities, token_ids, size=t0_rows * 128, bits=bits
        )
    )

    t1_stages = _bitonic_sort_looped_stages(
        keys, probabilities, token_ids, size=t1_rows * 128, bits=bits
    )
    t2_stages = _bitonic_sort_looped_stages(
        keys, probabilities, token_ids, size=t2_rows * 128, bits=bits
    )
    t3_stages = _bitonic_sort_looped_stages(
        keys, probabilities, token_ids, size=t3_rows * 128, bits=bits
    )

    def _run_stages(stage_fns: list[Callable[[], None]]) -> None:
        for fn in stage_fns:
            fn()
            if on_tail_stage is not None:
                on_tail_stage()

    def _tail_fallback() -> None:
        cur_rows = write_cursor[0]

        @pl.when((cur_rows > t0_rows) & (cur_rows <= t1_rows))
        def _run_t1() -> None:
            _run_stages(t1_stages)

        @pl.when((cur_rows > t1_rows) & (cur_rows <= t2_rows))
        def _run_t2() -> None:
            _run_stages(t2_stages)

        @pl.when(cur_rows > t2_rows)
        def _run_t3() -> None:
            _run_stages(t3_stages)

    return CompactMetadataSchedule(steps=tuple(steps), tail=_tail_fallback)


def _count_eq_row1(ids_16x8x128: jax.Array, target: jax.Array | int) -> jax.Array:
    v8 = jnp.sum((ids_16x8x128 == target).astype(jnp.int32), axis=0)
    return jnp.sum(v8, axis=0, keepdims=True)


def _count_compact_local_experts_vec8(
    keys,
    counts,
    *,
    used_rows: jax.Array,
    bits: int,
    local_experts: int,
) -> None:
    n_cnt_tiles = pl.cdiv(used_rows, 32)
    padded_e = ((local_experts + 7) // 8) * 8

    @functools.partial(
        pl.run_scoped,
        acc=pltpu.VMEM((padded_e, 128), jnp.int32),
    )
    def _run(acc) -> None:
        acc[...] = jnp.zeros((padded_e, 128), jnp.int32)

        @pl.loop(0, n_cnt_tiles)
        def _cnt_tile(tile_idx) -> None:
            block = keys[pl.ds(tile_idx * 32, 32), :] >> bits
            b4 = block.reshape(4, 8, 128)
            for e0 in range(0, padded_e, 8):
                rows_8 = jnp.concatenate(
                    [
                        jnp.sum(
                            jnp.sum((b4 == (e0 + j)).astype(jnp.int32), axis=0),
                            axis=0,
                            keepdims=True,
                        )
                        if (e0 + j) < local_experts
                        else jnp.zeros((1, 128), jnp.int32)
                        for j in range(8)
                    ],
                    axis=0,
                )
                acc[pl.ds(e0, 8), :] = acc[pl.ds(e0, 8), :] + rows_8

        counts[...] = jnp.sum(acc[...], axis=1)[:local_experts]


def _emit_replica_rebalance_batch8(
    tmp_k,
    replica_table,
    replica_scratch: ReplicaRoutingScratch,
    *,
    n_rep: int,
    min_gap: int,
    my_group,
) -> None:
    group_loads = replica_scratch.group_loads
    my_drop_experts = replica_scratch.my_drop_experts
    my_add_experts = replica_scratch.my_add_experts
    my_add_slots = replica_scratch.my_add_slots
    my_counts = replica_scratch.my_counts

    def _commit_if_valid(k, mv) -> None:
        hg = replica_table[n_rep + k]
        rg = replica_table[2 * n_rep + k]
        l_hg = group_loads[hg]
        l_rg = group_loads[rg]
        gap = l_hg - l_rg

        @pl.when((gap >= min_gap) & (mv > 0) & (mv <= gap))
        def _commit() -> None:
            he = replica_table[k]
            group_loads[hg] = l_hg - mv
            group_loads[rg] = l_rg + mv

            @pl.when(hg == my_group)
            def _rec_drop() -> None:
                idx = my_counts[0]
                my_drop_experts[idx] = he
                my_counts[0] = idx + 1

            @pl.when(rg == my_group)
            def _rec_add() -> None:
                idx = my_counts[1]
                my_add_experts[idx] = he
                my_add_slots[idx] = replica_table[3 * n_rep + k]
                my_counts[1] = idx + 1

    for k0 in range(0, n_rep, 8):
        chunk = min(8, n_rep - k0)
        b_hot = jnp.int32(0)
        for j in range(chunk):
            k = k0 + j
            he = replica_table[k]
            hg = replica_table[n_rep + k]
            rg = replica_table[2 * n_rep + k]
            b_hot = b_hot | (
                (he >= 0) & ((group_loads[hg] - group_loads[rg]) >= min_gap)
            ).astype(jnp.int32)

        @pl.when(b_hot != 0)
        def _do_batch(k0=k0, chunk=chunk) -> None:
            ids16 = tmp_k[...].reshape(16, 8, 128)
            rows_8 = [
                _count_eq_row1(ids16, replica_table[k0 + j])
                if j < chunk
                else jnp.zeros((1, 128), jnp.int32)
                for j in range(8)
            ]
            c8 = jnp.sum(jnp.concatenate(rows_8, axis=0), axis=1, keepdims=True)
            for j in range(chunk):
                k = k0 + j
                he = replica_table[k]

                @pl.when(he >= 0)
                def _try(k=k, mv=c8[j, 0]) -> None:
                    _commit_if_valid(k, mv)


def _run_sequential_metadata_window(
    win_idx,
    wire,
    keys,
    probabilities,
    tmp_k,
    tmp_p_i32,
    tmp_loc,
    write_cursor,
    group_loads,
    *,
    ranks_per_win: int,
    rank_stride: int,
    rank_half: int,
    my_group,
    base_local_experts: int,
    effective_local_experts: int,
    ep_groups: int,
    bits: int,
    sentinel_key: int,
    replica_table,
    n_rep: int,
    min_gap: int,
    replica_scratch: ReplicaRoutingScratch | None,
) -> None:
    id_parts, prob_parts = [], []
    for rank_off in range(ranks_per_win):
        rank_base = pl.multiple_of(
            (win_idx * ranks_per_win + rank_off) * rank_stride, rank_stride
        )
        id_parts.append(wire[pl.ds(rank_base, rank_half), :])
        prob_parts.append(
            jax.lax.bitcast_convert_type(
                wire[pl.ds(rank_base + rank_half, rank_half), :], jnp.float32
            )
        )
    ids_win = jnp.concatenate(id_parts, axis=0)
    probs_win = jnp.concatenate(prob_parts, axis=0)
    live_win = probs_win != 0

    if replica_table is not None:
        assert replica_scratch is not None and tmp_loc is not None
        rep_scratch: ReplicaRoutingScratch = replica_scratch
        loc_ref = tmp_loc
        ids_live = jnp.where(live_win, ids_win, -1)
        groups = jnp.where(live_win, ids_win // base_local_experts, -1)
        g16 = groups.reshape(16, 8, 128)
        grp_rows = jnp.concatenate(
            [_count_eq_row1(g16, grp) for grp in range(ep_groups)], axis=0
        )
        grp_cnts = jnp.sum(grp_rows, axis=1, keepdims=True)
        max_l = jnp.int32(-1)
        min_l = jnp.int32(1 << 30)
        for grp in range(ep_groups):
            updated = group_loads[grp] + grp_cnts[grp, 0]
            group_loads[grp] = updated
            max_l = jnp.maximum(max_l, updated)
            min_l = jnp.minimum(min_l, updated)

        tmp_k[...] = ids_live
        rep_scratch.my_counts[0] = 0
        rep_scratch.my_counts[1] = 0

        @pl.when((max_l - min_l) >= min_gap)
        def _rebalance_window() -> None:
            _emit_replica_rebalance_batch8(
                tmp_k,
                replica_table,
                rep_scratch,
                n_rep=n_rep,
                min_gap=min_gap,
                my_group=my_group,
            )

        ids_v = tmp_k[...]
        local = ids_v - (my_group * base_local_experts)
        valid0 = (ids_v >= 0) & (local >= 0) & (local < base_local_experts)
        loc_ref[...] = jnp.where(valid0, local, effective_local_experts)

        @pl.when((rep_scratch.my_counts[0] > 0) | (rep_scratch.my_counts[1] > 0))
        def _apply_mutations() -> None:
            @pl.loop(0, rep_scratch.my_counts[0])
            def _apply_drop(di) -> None:
                de = rep_scratch.my_drop_experts[di]
                loc_ref[...] = jnp.where(
                    tmp_k[...] == de, effective_local_experts, loc_ref[...]
                )

            @pl.loop(0, rep_scratch.my_counts[1])
            def _apply_add(ai) -> None:
                ae = rep_scratch.my_add_experts[ai]
                aslot = rep_scratch.my_add_slots[ai]
                loc_ref[...] = jnp.where(tmp_k[...] == ae, aslot, loc_ref[...])

        local_or_sentinel = loc_ref[...]
        valid = local_or_sentinel < effective_local_experts
        probs_cur = jnp.concatenate(
            [
                jax.lax.bitcast_convert_type(
                    wire[
                        pl.ds(
                            pl.multiple_of(
                                (win_idx * ranks_per_win + rank_off) * rank_stride,
                                rank_stride,
                            )
                            + rank_half,
                            rank_half,
                        ),
                        :,
                    ],
                    jnp.float32,
                )
                for rank_off in range(ranks_per_win)
            ],
            axis=0,
        )
    else:
        if group_loads is not None:
            groups = jnp.where(live_win, ids_win // base_local_experts, -1)
            g16 = groups.reshape(16, 8, 128)
            grp_rows = jnp.concatenate(
                [_count_eq_row1(g16, grp) for grp in range(ep_groups)], axis=0
            )
            grp_cnts = jnp.sum(grp_rows, axis=1, keepdims=True)
            for grp in range(ep_groups):
                group_loads[grp] = group_loads[grp] + grp_cnts[grp, 0]
        local = ids_win - (my_group * base_local_experts)
        valid = live_win & (local >= 0) & (local < effective_local_experts)
        local_or_sentinel = jnp.where(valid, local, effective_local_experts)
        probs_cur = probs_win

    win_pos = jax.lax.broadcasted_iota(
        jnp.int32, (_WIN_ROWS, 128), 0
    ) * 128 + jax.lax.broadcasted_iota(jnp.int32, (_WIN_ROWS, 128), 1)
    packed = (local_or_sentinel << bits) | (win_idx * (_WIN_ROWS * 128) + win_pos)
    packed_k, packed_p, n_live = _left_pack_128x128_matmul(
        packed, probs_cur, valid, sentinel_key=sentinel_key
    )
    tmp_k[...] = packed_k
    tmp_p_i32[...] = jax.lax.bitcast_convert_type(packed_p, jnp.int32)
    n_rows = (n_live + 127) // 128
    base_row = write_cursor[0]
    _append_window_rows(
        keys,
        probabilities,
        tmp_k,
        tmp_p_i32,
        base_row=base_row,
        n_rows=n_rows,
    )
    write_cursor[0] = base_row + n_rows


def run_sequential_compact_metadata(
    wire,
    keys,
    probabilities,
    token_ids,
    counts,
    maximum,
    write_cursor,
    *,
    tokens: int,
    rank,
    counts_ready: Callable[[], None],
    local_experts: int = 56,
    on_poll: Callable[[], None] | None = None,
    poll_win_stride: int = 2,
    poll_sort_stride: int = 2,
    replica_table=None,
    replica_slots: int = 0,
    replica_scratch: ReplicaRoutingScratch | None = None,
    replica_min_gap: int | None = None,
) -> None:
    assert tokens in (8192, 16384)
    if replica_table is not None:
        if replica_slots <= 0 or replica_scratch is None:
            raise ValueError(
                "replica_table requires replica_slots > 0 and replica_scratch"
            )
        base_local_experts = local_experts - replica_slots
    else:
        base_local_experts = local_experts
    ep_groups = 896 // base_local_experts
    expert_tp = 32 // ep_groups
    my_group = rank // expert_tp
    bits = (tokens * 16 - 1).bit_length()
    sentinel_key = (local_experts << bits) | ((1 << bits) - 1)
    total_rows = tokens // 8
    n_wins = total_rows // _WIN_ROWS
    rank_half = tokens // 256
    rank_stride = tokens // 128
    ranks_per_win = _WIN_ROWS // rank_half
    n_rep = ep_groups * replica_slots if replica_table is not None else 0
    min_gap = (
        (
            default_replica_min_gap(tokens)
            if replica_min_gap is None
            else replica_min_gap
        )
        if replica_table is not None
        else 0
    )

    tmp_k = token_ids.at[pl.ds(0, _WIN_ROWS), :]
    tmp_p_i32 = token_ids.at[pl.ds(_WIN_ROWS, _WIN_ROWS), :]
    tmp_loc = (
        token_ids.at[pl.ds(2 * _WIN_ROWS, _WIN_ROWS), :]
        if replica_table is not None
        else None
    )

    def _run_body(group_loads) -> None:
        write_cursor[0] = 0
        if group_loads is not None:
            for grp in range(ep_groups):
                group_loads[grp] = 0

        @pl.loop(0, n_wins)
        def _win_loop(win_idx) -> None:
            _run_sequential_metadata_window(
                win_idx,
                wire,
                keys,
                probabilities,
                tmp_k,
                tmp_p_i32,
                tmp_loc,
                write_cursor,
                group_loads,
                ranks_per_win=ranks_per_win,
                rank_stride=rank_stride,
                rank_half=rank_half,
                my_group=my_group,
                base_local_experts=base_local_experts,
                effective_local_experts=local_experts,
                ep_groups=ep_groups,
                bits=bits,
                sentinel_key=sentinel_key,
                replica_table=replica_table,
                n_rep=n_rep,
                min_gap=min_gap,
                replica_scratch=replica_scratch,
            )
            if on_poll is not None:
                if poll_win_stride <= 1:
                    on_poll()
                else:

                    @pl.when((win_idx % poll_win_stride) == (poll_win_stride - 1))
                    def _do_win_poll() -> None:
                        on_poll()

        t0_rows = 256 if tokens == 16384 else 128
        t1_rows = t0_rows * 2
        t2_rows = t0_rows * 4
        t3_rows = t0_rows * 8
        used_rows = write_cursor[0]
        tier_rows = jnp.where(
            used_rows <= t0_rows,
            t0_rows,
            jnp.where(
                used_rows <= t1_rows,
                t1_rows,
                jnp.where(used_rows <= t2_rows, t2_rows, t3_rows),
            ),
        )
        _pad_compact_rows_to_tier(
            keys,
            probabilities,
            used_rows=used_rows,
            tier_rows=tier_rows,
            sentinel_key=sentinel_key,
        )
        _count_compact_local_experts_vec8(
            keys,
            counts,
            used_rows=used_rows,
            bits=bits,
            local_experts=local_experts,
        )
        if maximum is not None:
            assert group_loads is not None
            mx = group_loads[0]
            for grp in range(1, ep_groups):
                mx = jnp.maximum(mx, group_loads[grp])
            maximum[...] = mx[None]
        counts_ready()
        if on_poll is not None:
            on_poll()

        t0_steps = _bitonic_sort_unrolled_t0_steps(
            keys,
            probabilities,
            token_ids,
            size=t0_rows * 128,
            bits=bits,
            window=8192,
            rows=64,
            write_final_keys=False,
        )

        @pl.when(used_rows <= t0_rows)
        def _run_t0() -> None:
            for idx_s, fn in enumerate(t0_steps):
                fn()
                if on_poll is not None and ((idx_s + 1) % poll_sort_stride == 0):
                    on_poll()

        t1_stages = _bitonic_sort_looped_stages(
            keys, probabilities, token_ids, size=t1_rows * 128, bits=bits
        )
        t2_stages = _bitonic_sort_looped_stages(
            keys, probabilities, token_ids, size=t2_rows * 128, bits=bits
        )
        t3_stages = _bitonic_sort_looped_stages(
            keys, probabilities, token_ids, size=t3_rows * 128, bits=bits
        )

        def _run_stages(stage_fns: list[Callable[[], None]]) -> None:
            for fn in stage_fns:
                fn()
                if on_poll is not None:
                    on_poll()

        @pl.when((used_rows > t0_rows) & (used_rows <= t1_rows))
        def _run_t1() -> None:
            _run_stages(t1_stages)

        @pl.when((used_rows > t1_rows) & (used_rows <= t2_rows))
        def _run_t2() -> None:
            _run_stages(t2_stages)

        @pl.when(used_rows > t2_rows)
        def _run_t3() -> None:
            _run_stages(t3_stages)

    if replica_table is not None:
        assert replica_scratch is not None
        _run_body(replica_scratch.group_loads)
    elif maximum is not None:
        pl.run_scoped(_run_body, pltpu.SMEM((ep_groups,), jnp.int32))
    else:
        _run_body(None)
