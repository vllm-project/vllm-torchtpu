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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.common import pool_alias
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.config import active as _config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as pm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_megakernel import (
    _CHANNELS,
    _HEADS,
    _KDA_CHUNK,
    _KDA_MAX_GROUP_CHUNKS,
    _dot,
    _half_gather_step,
    _half_ring_streams,
    _kda_chunks_batched,
    _kda_conv_norm_decay_tile,
    _kda_fp8_window,
    _kda_mix_shapes,
    _kda_rms_norm_gate_group,
    _ring_enter,
    _row_where,
)

RANKS = 32
HEAD_COLS = 384
HIDDEN = 7168
HALF_HIDDEN = HIDDEN // 2
_FP8_E4M3_MAX = 448.0
_FP8_SCALE_EPS = 1e-12
KDA_RING_CYCLES = (
    (0, 1, 2, 3, 7, 6, 14, 15, 11, 10, 9, 8, 12, 13, 5, 4),
    (0, 1, 2, 3, 11, 15, 7, 6, 14, 10, 9, 13, 5, 4, 12, 8),
    (0, 4, 12, 13, 14, 15, 7, 3, 11, 10, 2, 6, 5, 1, 9, 8),
)
KDA_RING_COLUMN_BLOCKS = (17, 18, 21)


def _quantize_rows_fp8(x):
    x_f32 = x.astype(jnp.float32)
    max_abs = jnp.maximum(
        jnp.max(jnp.abs(x_f32), axis=1, keepdims=True), _FP8_SCALE_EPS
    )
    return (x_f32 * (_FP8_E4M3_MAX / max_abs)).astype(jnp.float8_e4m3fn), max_abs * (
        1.0 / _FP8_E4M3_MAX
    )


def build_kda_a2a_tables(devices, token_ranks, cycles=pm.CYCLES) -> np.ndarray:
    ring = np.asarray(pm.ring_tables(devices, token_ranks, cycles), np.int32)
    owner_to_peer = np.zeros((1, 3, len(token_ranks)), dtype=np.int32)
    for peer, owner in enumerate(token_ranks):
        owner_to_peer[0, 0, int(owner)] = peer
    return np.concatenate((ring, owner_to_peer), axis=0)


def _owner_all_to_all_copies(
    tables_ref,
    attended_ref,
    owner_rows_ref,
    send_sems,
    recv_sems,
    *,
    block,
    rank_major: bool = False,
):
    rank = jax.lax.axis_index("k3_sp")
    ranks = owner_rows_ref.shape[0] if rank_major else owner_rows_ref.shape[1]
    stripe = owner_rows_ref.shape[2]
    window_first = block * ranks * stripe
    slot = block % send_sems.shape[0]
    dst = (
        owner_rows_ref.at[rank, block] if rank_major else owner_rows_ref.at[block, rank]
    )

    def rows_for(peer):
        owner = tables_ref[0, 2, peer]
        return attended_ref.at[pl.ds(window_first + owner * stripe, stripe)]

    def remote(peer):
        return pltpu.make_async_remote_copy(
            rows_for(peer),
            dst,
            send_sems.at[slot],
            recv_sems.at[slot],
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def local():
        return pltpu.make_async_copy(rows_for(rank), dst, recv_sems.at[slot])

    return rank, ranks, remote, local


def owner_all_to_all_send(
    tables_ref,
    attended_ref,
    owner_rows_ref,
    send_sems,
    recv_sems,
    *,
    block,
    rank_major: bool = False,
):
    rank, ranks, remote, local = _owner_all_to_all_copies(
        tables_ref,
        attended_ref,
        owner_rows_ref,
        send_sems,
        recv_sems,
        block=block,
        rank_major=rank_major,
    )
    local().start()

    @pl.loop(1, ranks)
    def _(offset):
        remote((rank + offset) % ranks).start()


def owner_all_to_all_send_range(
    tables_ref,
    attended_ref,
    owner_rows_ref,
    send_sems,
    recv_sems,
    *,
    block,
    owner_start,
    owner_end,
    rank_major: bool = False,
):
    rank = jax.lax.axis_index("k3_sp")
    ranks = owner_rows_ref.shape[0] if rank_major else owner_rows_ref.shape[1]
    stripe = owner_rows_ref.shape[2]
    window_first = block * ranks * stripe
    slot = block % send_sems.shape[0]
    dst = (
        owner_rows_ref.at[rank, block] if rank_major else owner_rows_ref.at[block, rank]
    )
    span = owner_end - owner_start

    @pl.loop(0, span)
    def _(idx):
        owner = owner_start + (idx + rank) % span
        peer = tables_ref[3, 0, owner]
        src = attended_ref.at[pl.ds(window_first + owner * stripe, stripe)]

        @pl.when(peer == rank)
        def send_local():
            pltpu.make_async_copy(src, dst, recv_sems.at[slot]).start()

        @pl.when(peer != rank)
        def send_remote():
            pltpu.make_async_remote_copy(
                src,
                dst,
                send_sems.at[slot],
                recv_sems.at[slot],
                device_id=(peer,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()


def owner_all_to_all_join(
    tables_ref,
    attended_ref,
    owner_rows_ref,
    send_sems,
    recv_sems,
    *,
    block,
    rank_major: bool = False,
):
    rank, ranks, remote, local = _owner_all_to_all_copies(
        tables_ref,
        attended_ref,
        owner_rows_ref,
        send_sems,
        recv_sems,
        block=block,
        rank_major=rank_major,
    )

    @pl.loop(1, ranks)
    def _(offset):
        remote((rank + offset) % ranks).wait_send()

    @pl.loop(0, ranks)
    def _(_offset):
        local().wait()


def _owner_all_to_all_step(
    tables_ref,
    attended_ref,
    owner_rows_ref,
    send_sems,
    recv_sems,
    *,
    step,
    block,
    rank_major: bool = False,
):
    @pl.when(step == 0)
    def send():
        owner_all_to_all_send(
            tables_ref,
            attended_ref,
            owner_rows_ref,
            send_sems,
            recv_sems,
            block=block,
            rank_major=rank_major,
        )

    @pl.when(step == pm.HALF_STEPS - 1)
    def join():
        owner_all_to_all_join(
            tables_ref,
            attended_ref,
            owner_rows_ref,
            send_sems,
            recv_sems,
            block=block,
            rank_major=rank_major,
        )


def _resolve_kda_segment(segment_ref, count, first_row):
    segment, segment_end = -1, 0
    for s in range(count):
        start = segment_ref[s]
        end = start + segment_ref[count + s]
        inside = (first_row >= start) & (first_row < end)
        segment = jnp.where(inside, s, segment)
        segment_end = jnp.where(inside, end, segment_end)
    current = jnp.maximum(segment, 0)
    offset = segment_ref[current]
    position = segment_ref[2 * count + current]
    return segment, segment_end, current, offset, position


def kda_split_prefill_scratch_shapes(
    hidden_size,
    tile,
    gate_w,
    fb_w,
    proj_w,
    conv_w,
    alog,
    dt,
    norm,
    *,
    hidden_dtype=jnp.bfloat16,
    fp8_stage_p=False,
):
    total_cols = gate_w.shape[1] + proj_w.shape[1]
    base = pm.kda_prefill_tiled_scratch_shapes(
        hidden_size,
        tile,
        gate_w,
        fb_w,
        proj_w,
        conv_w,
        alog,
        dt,
        norm,
        store_out=True,
        hidden_dtype=hidden_dtype,
    )
    gate_and_proj = pltpu.VMEM((tile, total_cols), jnp.bfloat16)
    if not fp8_stage_p:
        return (*base, gate_and_proj)
    if total_cols % 256 or gate_w.shape[1] % 128 or gate_w.shape[1] >= total_cols // 2:
        raise ValueError(
            "fp8_stage_p requires 256-aligned total cols and gate_cols < "
            "total_cols // 2"
        )
    staged = list(base)
    staged[2] = pltpu.VMEM((hidden_size, total_cols // 2), proj_w.dtype)
    staged[4] = pltpu.VMEM((hidden_size, total_cols), jnp.float8_e4m3fn)
    return (*staged, gate_and_proj, pltpu.VMEM((8, total_cols), jnp.float32))


def kda_split_prefill_body(
    hidden_ref,
    gate_w_ref,
    fb_w_ref,
    proj_w_ref,
    conv_w_ref,
    alog_ref,
    dt_ref,
    norm_ref,
    rec_state_in_ref,
    conv_hist_in_ref,
    out_ref,
    rec_state_out_ref,
    conv_hist_out_ref,
    rec_checkpoint_ref,
    conv_checkpoint_ref,
    segment_ref,
    hidden_bufs,
    out_bufs,
    gate_w_vmem,
    fb_w_vmem,
    proj_w_vmem,
    conv_w_vmem,
    alog_vmem,
    dt_vmem,
    norm_vmem,
    rec_state_vmem,
    conv_hist_vmem,
    chunk_stage_vmem,
    hidden_sems,
    out_sems,
    weight_sems,
    gate_and_proj_vmem,
    *,
    tile,
    block_tiles,
    blocks,
    count,
    phase,
    ring_step,
    step,
    decay_lower_bound,
    epsilon,
    input_norm=None,
    proj_scale_vmem=None,
):
    del chunk_stage_vmem
    head_count, channels = _HEADS, _CHANNELS
    hidden_size, gate_cols = gate_w_ref.shape
    total_cols = gate_cols + proj_w_ref.shape[1]
    num_chunks = tile // _KDA_CHUNK
    fp8_stage_p = proj_scale_vmem is not None
    if proj_w_vmem.shape != (hidden_size, total_cols):
        raise ValueError("kda_split_prefill_body requires fused gate+proj VMEM layout")
    if num_chunks > _KDA_MAX_GROUP_CHUNKS:
        raise ValueError(
            f"kda_split_prefill_body supports <= {_KDA_MAX_GROUP_CHUNKS} chunks/tile"
        )
    half_cols = total_cols // 2
    first_proj_cols = half_cols - gate_cols
    if proj_scale_vmem is not None:
        if proj_w_vmem.dtype != jnp.float8_e4m3fn or proj_scale_vmem.shape != (
            8,
            total_cols,
        ):
            raise ValueError(
                "fp8_stage_p requires float8_e4m3fn proj_w_vmem and (8, cols) "
                "proj_scale_vmem"
            )
        if gate_w_vmem.shape != (hidden_size, half_cols):
            raise ValueError(
                "fp8_stage_p requires (hidden_size, total_cols // 2) staging buffer"
            )

    small_weight_copies = (
        pltpu.make_async_copy(fb_w_ref, fb_w_vmem, weight_sems.at[1]),
        pltpu.make_async_copy(conv_w_ref, conv_w_vmem, weight_sems.at[3]),
        pltpu.make_async_copy(alog_ref, alog_vmem, weight_sems.at[4]),
        pltpu.make_async_copy(dt_ref, dt_vmem, weight_sems.at[5]),
        pltpu.make_async_copy(norm_ref, norm_vmem, weight_sems.at[6]),
    )
    if proj_scale_vmem is not None:
        w_stage_vmem = gate_w_vmem
        gate_copy = pltpu.make_async_copy(
            gate_w_ref, w_stage_vmem.at[:, :gate_cols], weight_sems.at[0]
        )
        proj_lo_copy = pltpu.make_async_copy(
            proj_w_ref.at[:, :first_proj_cols],
            w_stage_vmem.at[:, gate_cols:],
            weight_sems.at[2],
        )
        proj_hi_copy = pltpu.make_async_copy(
            proj_w_ref.at[:, first_proj_cols:], w_stage_vmem, weight_sems.at[2]
        )
        norm_copy = (
            pltpu.make_async_copy(input_norm[0], input_norm[1], weight_sems.at[12])
            if input_norm is not None
            else None
        )
        weight_copies = small_weight_copies
    else:
        del gate_w_vmem
        weight_copies = (
            pltpu.make_async_copy(
                gate_w_ref, proj_w_vmem.at[:, :gate_cols], weight_sems.at[0]
            ),
            small_weight_copies[0],
            pltpu.make_async_copy(
                proj_w_ref, proj_w_vmem.at[:, gate_cols:], weight_sems.at[2]
            ),
            *small_weight_copies[1:],
        )
        if input_norm is not None:
            weight_copies += (
                pltpu.make_async_copy(input_norm[0], input_norm[1], weight_sems.at[12]),
            )

    def copy_now(pairs, first_semaphore):
        copies = [
            pltpu.make_async_copy(
                source, destination, weight_sems.at[first_semaphore + i]
            )
            for i, (source, destination) in enumerate(pairs)
        ]
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    def write_checkpoint(segment):
        copy_now(
            (
                (rec_state_vmem, rec_checkpoint_ref.at[segment]),
                (conv_hist_vmem, conv_checkpoint_ref.at[segment]),
            ),
            11,
        )

    def unused_copies(s):
        pairs = (
            (rec_state_in_ref.at[s], rec_state_out_ref.at[s]),
            (conv_hist_in_ref.at[s], conv_hist_out_ref.at[s]),
            (rec_state_in_ref.at[s], rec_checkpoint_ref.at[s]),
            (conv_hist_in_ref.at[s], conv_checkpoint_ref.at[s]),
        )
        return [
            pltpu.make_async_copy(source, target, weight_sems.at[14 + i % 2])
            for i, (source, target) in enumerate(pairs)
        ]

    def each_unused(action):
        @pl.loop(0, count)
        def visit(s):
            @pl.when(segment_ref[count + s] == 0)
            def unused():
                for copy in unused_copies(s):
                    action(copy)

    def load_tile(t, buf):
        return pltpu.make_async_copy(
            hidden_ref.at[pl.ds(t * tile, tile), :],
            hidden_bufs.at[buf],
            hidden_sems.at[buf],
        )

    def store_tile(t, buf):
        return pltpu.make_async_copy(
            out_bufs.at[buf],
            out_ref.at[pl.ds(t * tile, tile), :],
            out_sems.at[buf],
        )

    @pl.when(step == 0)
    def start_weights_and_unused():
        for copy in weight_copies:
            copy.start()
        if fp8_stage_p:
            gate_copy.start()
            proj_lo_copy.start()
            if norm_copy is not None:
                norm_copy.start()
        each_unused(lambda copy: copy.start())

    if proj_scale_vmem is not None:
        scale_ref = proj_scale_vmem

        def fold_and_quantize_half(col_start, col_end):
            row_chunk = min(hidden_size, 256)
            if hidden_size % row_chunk:
                raise ValueError(
                    f"fp8_stage_p requires hidden_size divisible by {row_chunk}"
                )
            scale_ref[:, col_start:col_end] = jnp.full(
                (8, half_cols), _FP8_SCALE_EPS, jnp.float32
            )

            @pl.loop(0, hidden_size // row_chunk)
            def reduce_max(k):
                k0 = k * row_chunk
                w_f32 = w_stage_vmem[pl.ds(k0, row_chunk), :].astype(jnp.float32)
                if input_norm is not None:
                    norm_slice = input_norm[1][0:1, pl.ds(k0, row_chunk)].astype(
                        jnp.float32
                    )
                    w_f32 = (
                        w_f32 * jnp.broadcast_to(norm_slice, (half_cols, row_chunk)).T
                    )
                chunk_max = jnp.max(jnp.abs(w_f32), axis=0, keepdims=True)
                scale_ref[:, col_start:col_end] = jnp.maximum(
                    scale_ref[:, col_start:col_end],
                    jnp.broadcast_to(chunk_max, (8, half_cols)),
                )

            max_abs = scale_ref[:, col_start:col_end]
            inv_scale = _FP8_E4M3_MAX / max_abs[0:1, :]
            scale_ref[:, col_start:col_end] = max_abs * (1.0 / _FP8_E4M3_MAX)

            @pl.loop(0, hidden_size // row_chunk)
            def quantize_chunk(k):
                k0 = k * row_chunk
                w_f32 = w_stage_vmem[pl.ds(k0, row_chunk), :].astype(jnp.float32)
                if input_norm is not None:
                    norm_slice = input_norm[1][0:1, pl.ds(k0, row_chunk)].astype(
                        jnp.float32
                    )
                    w_f32 = (
                        w_f32 * jnp.broadcast_to(norm_slice, (half_cols, row_chunk)).T
                    )
                proj_w_vmem[pl.ds(k0, row_chunk), col_start:col_end] = (
                    w_f32 * inv_scale
                ).astype(jnp.float8_e4m3fn)

        @pl.when(step == 8)
        def fold_first_weight_half():
            gate_copy.wait()
            proj_lo_copy.wait()
            if norm_copy is not None:
                norm_copy.wait()
            fold_and_quantize_half(0, half_cols)
            proj_hi_copy.start()

        @pl.when(step == 10)
        def fold_second_weight_half():
            proj_hi_copy.wait()
            fold_and_quantize_half(half_cols, total_cols)

    @pl.when((phase < blocks) & (ring_step == pm.HALF_STEPS - 1))
    def prefetch_next_block_first_tile():
        first_next = phase * block_tiles
        load_tile(first_next, first_next % 2).start()

    @pl.when(step == pm.HALF_STEPS)
    def wait_weights():
        for copy in weight_copies:
            copy.wait()

    attending = (phase >= 1) & (phase <= blocks)
    active_half = attending & (ring_step < 2 * block_tiles)
    t_local = ring_step // 2
    t = (phase - 1) * block_tiles + t_local
    buf = t % 2
    first_row = t * tile

    @pl.when(active_half & (ring_step % 2 == 0))
    def stage_project():
        load_tile(t, buf).wait()

        @pl.when(t_local + 1 < block_tiles)
        def prefetch_next_tile():
            load_tile(t + 1, 1 - buf).start()

        _, segment_end, _, _, _ = _resolve_kda_segment(segment_ref, count, first_row)
        real_count = jnp.clip(segment_end - first_row, 0, tile)

        @pl.when(real_count > 0)
        def project_active_tile():
            hidden_tile = hidden_bufs.at[buf][...]
            if fp8_stage_p:
                if input_norm is not None:
                    projected = (
                        _dot(hidden_tile, proj_w_vmem[...]) * proj_scale_vmem[0:1, :]
                    )
                else:
                    hidden_fp8, row_scale = _quantize_rows_fp8(hidden_tile)
                    projected = (
                        _dot(hidden_fp8, proj_w_vmem[...])
                        * row_scale
                        * proj_scale_vmem[0:1, :]
                    )
                gate_and_proj_vmem[...] = projected.astype(jnp.bfloat16)
            else:
                if input_norm is not None:
                    hidden_tile = hidden_tile.astype(jnp.bfloat16) * input_norm[1][...]
                gate_and_proj_vmem[...] = _dot(hidden_tile, proj_w_vmem[...]).astype(
                    jnp.bfloat16
                )

    @pl.when(active_half & (ring_step % 2 == 1))
    def stage_conv_scan():
        segment, segment_end, current, offset, position = _resolve_kda_segment(
            segment_ref,
            count,
            first_row,
        )

        @pl.when((segment >= 0) & (first_row == offset))
        def load_segment_state():
            copy_now(
                (
                    (rec_state_in_ref.at[current], rec_state_vmem),
                    (conv_hist_in_ref.at[current], conv_hist_vmem),
                ),
                7,
            )
            pl.when(position == 0)(lambda: write_checkpoint(current))

        @pl.when(t_local >= 2)
        def wait_previous_store():
            store_tile(t - 2, buf).wait()

        real_count = jnp.clip(segment_end - first_row, 0, tile)

        @pl.when(real_count == 0)
        def zero_dead_tile():
            out_bufs.at[buf][...] = jnp.zeros(out_bufs.at[buf].shape, out_bufs.dtype)

        @pl.when(real_count > 0)
        def compute_active_tile():
            gate_and_proj = gate_and_proj_vmem[...]
            gate_features = gate_and_proj[:, :gate_cols]
            projected = gate_and_proj[:, gate_cols:]
            raw_gate_2d = _dot(gate_features, fb_w_vmem[...]).astype(jnp.bfloat16)
            real_rows = jnp.arange(tile, dtype=jnp.int32) + first_row < segment_end
            (
                q_chunks,
                k_chunks,
                v_chunks,
                decay_chunks,
                beta_pad_f32,
                out_gate_sig,
                win_pad_bf16,
            ) = _kda_conv_norm_decay_tile(
                projected,
                raw_gate_2d,
                conv_hist_vmem[...],
                conv_w_vmem[...],
                alog_vmem[...],
                dt_vmem[...],
                real_rows,
                decay_lower_bound=decay_lower_bound,
            )
            beta_chunks = (
                beta_pad_f32[:, :head_count]
                .reshape(num_chunks, _KDA_CHUNK, head_count)
                .transpose(0, 2, 1)
            )
            chunk_outs, new_state = _kda_chunks_batched(
                q_chunks,
                k_chunks,
                v_chunks,
                decay_chunks,
                beta_chunks,
                rec_state_vmem[...],
            )
            rec_state_vmem[...] = new_state
            out_bufs.at[buf][...] = _kda_rms_norm_gate_group(
                chunk_outs,
                norm_vmem[0],
                out_gate_sig,
                epsilon=epsilon,
            )

            @pl.when(real_count == tile)
            def carry_full_tile():
                conv_hist_vmem[...] = win_pad_bf16[tile + 5 : tile + 8].reshape(
                    3, 3, head_count, channels
                )

            @pl.when(real_count < tile)
            def carry_partial_tile():
                rows_w = jnp.arange(tile + 8, dtype=jnp.int32)
                picked = [
                    jnp.sum(
                        _row_where(rows_w == 5 + real_count + i, win_pad_bf16), axis=0
                    )
                    for i in range(3)
                ]
                conv_hist_vmem[...] = jnp.stack(picked).reshape(
                    3, 3, head_count, channels
                )

        pl.when(
            (segment >= 0) & (position > 0) & ((t + 1) * tile == offset + position)
        )(lambda: write_checkpoint(current))

        @pl.when((segment >= 0) & (t == (segment_end - 1) // tile))
        def store_segment_state():
            copy_now(
                (
                    (rec_state_vmem, rec_state_out_ref.at[current]),
                    (conv_hist_vmem, conv_hist_out_ref.at[current]),
                ),
                9,
            )

        store_tile(t, buf).start()

    @pl.when(attending & (ring_step == pm.HALF_STEPS - 1))
    def drain_block_stores():
        t_last = (phase - 1) * block_tiles + block_tiles - 1
        last_buf = t_last % 2
        store_tile(t_last, last_buf).wait()
        if block_tiles > 1:
            store_tile(t_last - 1, 1 - last_buf).wait()

    @pl.when(step == pm.HALF_STEPS * (blocks + 1) - 1)
    def finish_unused_segments():
        each_unused(lambda copy: copy.wait())


def _kda_gather_and_attend_step(
    slot,
    step,
    phase,
    ring_step,
    stripe: int,
    segment_ref,
    tables_ref,
    length_ref,
    local_normed_ref,
    gathered_ref,
    attended_ref,
    owner_rows_ref,
    gate_family_ref,
    fb_family_ref,
    proj_family_ref,
    conv_family_ref,
    alog_family_ref,
    dt_family_ref,
    kda_norm_family_ref,
    rec_in_ref,
    hist_in_ref,
    rec_out_ref,
    hist_out_ref,
    rec_checkpoint_ref,
    hist_checkpoint_ref,
    gather_sems,
    odd_gather_sems,
    a2a_sems,
    kda_scratch,
    *,
    eps: float,
    count: int,
    blocks: int,
    kda_tile: int,
    split_phase: bool,
    fp8_stage_p: bool,
    dual_gather: bool,
    column_blocks,
    input_norm,
    normalize_block_fn: Callable[[int, bool], None] | None,
) -> None:
    proj_scale_vmem = None
    if fp8_stage_p:
        (
            hidden_bufs,
            out_bufs,
            *kda_weights_and_state,
            hidden_sems,
            out_sems,
            weight_sems,
            gate_and_proj_vmem,
            proj_scale_vmem,
        ) = kda_scratch
    elif split_phase:
        (
            hidden_bufs,
            out_bufs,
            *kda_weights_and_state,
            hidden_sems,
            out_sems,
            weight_sems,
            gate_and_proj_vmem,
        ) = kda_scratch
    else:
        (
            hidden_bufs,
            out_bufs,
            *kda_weights_and_state,
            hidden_sems,
            out_sems,
            weight_sems,
        ) = kda_scratch

    pl.when(step == 0)(lambda: _ring_enter(tables_ref, jax.lax.axis_index("k3_sp")))

    if normalize_block_fn is not None:

        @pl.when(step == 0)
        def normalize_first():
            normalize_block_fn(0, True)

    gathered_block = 2 * phase if dual_gather else phase

    @pl.when(gathered_block < blocks)
    def gather():
        _half_gather_step(
            tables_ref,
            local_normed_ref,
            gathered_ref,
            *gather_sems,
            step=ring_step,
            block=gathered_block,
            column_blocks=column_blocks,
        )

    if dual_gather:
        if normalize_block_fn is not None and blocks > 1:

            @pl.when(step == 0)
            def normalize_second():
                normalize_block_fn(1, False)

        @pl.when(2 * phase + 1 < blocks)
        def gather_odd_block():
            _half_gather_step(
                tables_ref,
                local_normed_ref,
                gathered_ref,
                *odd_gather_sems,
                step=ring_step,
                block=2 * phase + 1,
                column_blocks=column_blocks,
            )

        if normalize_block_fn is not None:

            @pl.when((step >= 1) & (step < blocks - 1))
            def normalize_rest_dual():
                normalize_block_fn(step + 1, False)
    elif normalize_block_fn is not None:

        @pl.when(step < blocks - 1)
        def normalize_rest():
            normalize_block_fn(step + 1, False)

    block_tiles = RANKS * stripe // kda_tile
    attending = (phase >= 1) & (phase <= blocks)

    if split_phase:

        @pl.when((phase >= 2) & (ring_step == pm.HALF_STEPS - 1))
        def a2a_join_prev():
            owner_all_to_all_join(
                tables_ref,
                attended_ref,
                owner_rows_ref,
                *a2a_sems,
                block=phase - 2,
                rank_major=True,
            )

        kda_split_prefill_body(
            gathered_ref,
            gate_family_ref.at[slot],
            fb_family_ref.at[slot],
            proj_family_ref.at[slot],
            conv_family_ref.at[slot],
            alog_family_ref.at[slot],
            dt_family_ref.at[slot],
            kda_norm_family_ref.at[slot],
            rec_in_ref,
            hist_in_ref,
            attended_ref,
            rec_out_ref,
            hist_out_ref,
            rec_checkpoint_ref,
            hist_checkpoint_ref,
            segment_ref,
            hidden_bufs,
            out_bufs,
            *kda_weights_and_state,
            # pyrefly: ignore[bad-argument-count]
            hidden_sems,
            out_sems,
            weight_sems,
            gate_and_proj_vmem,  # type: ignore[bad-argument-count] -- starred scratch spans the fixed refs
            tile=kda_tile,
            block_tiles=block_tiles,
            blocks=blocks,
            count=count,
            phase=phase,
            ring_step=ring_step,
            step=step,
            decay_lower_bound=-5.0,
            epsilon=eps,
            input_norm=input_norm,
            proj_scale_vmem=proj_scale_vmem,
        )

        stream_last_a2a = (not dual_gather) and block_tiles >= 3
        stripes_per_tile = RANKS // block_tiles
        if stream_last_a2a:

            @pl.when(
                (phase == blocks)
                & (ring_step >= 5)
                & (ring_step < 2 * block_tiles)
                & (ring_step % 2 == 1)
            )
            def a2a_stream_last_block_tile():
                ready_tile = (ring_step // 2) - 2
                owner_all_to_all_send_range(
                    tables_ref,
                    attended_ref,
                    owner_rows_ref,
                    *a2a_sems,
                    block=blocks - 1,
                    owner_start=ready_tile * stripes_per_tile,
                    owner_end=(ready_tile + 1) * stripes_per_tile,
                    rank_major=True,
                )

            @pl.when(attending & (ring_step == pm.HALF_STEPS - 1))
            def a2a_send_block_streamed():
                tail_start = jnp.where(
                    phase == blocks, (block_tiles - 2) * stripes_per_tile, 0
                )
                owner_all_to_all_send_range(
                    tables_ref,
                    attended_ref,
                    owner_rows_ref,
                    *a2a_sems,
                    block=phase - 1,
                    owner_start=tail_start,
                    owner_end=RANKS,
                    rank_major=True,
                )
        else:

            @pl.when(attending & (ring_step == pm.HALF_STEPS - 1))
            def a2a_send_block():
                owner_all_to_all_send(
                    tables_ref,
                    attended_ref,
                    owner_rows_ref,
                    *a2a_sems,
                    block=phase - 1,
                    rank_major=True,
                )

        @pl.when(step == pm.HALF_STEPS * (blocks + 1) - 1)
        def a2a_join_last():
            owner_all_to_all_join(
                tables_ref,
                attended_ref,
                owner_rows_ref,
                *a2a_sems,
                block=blocks - 1,
                rank_major=True,
            )

        return

    @pl.when(phase >= 2)
    def all_to_all():
        _owner_all_to_all_step(
            tables_ref,
            attended_ref,
            owner_rows_ref,
            *a2a_sems,
            step=ring_step,
            block=phase - 2,
            rank_major=True,
        )

    first_tile = jnp.where(
        attending,
        (phase - 1) * block_tiles + ring_step * block_tiles // pm.HALF_STEPS,
        0,
    )
    end_tile = jnp.where(
        attending,
        (phase - 1) * block_tiles + (ring_step + 1) * block_tiles // pm.HALF_STEPS,
        0,
    )
    pm.kda_prefill_tiled_body(
        length_ref,
        gathered_ref,
        gate_family_ref.at[slot],
        fb_family_ref.at[slot],
        proj_family_ref.at[slot],
        conv_family_ref.at[slot],
        alog_family_ref.at[slot],
        dt_family_ref.at[slot],
        kda_norm_family_ref.at[slot],
        rec_in_ref,
        hist_in_ref,
        attended_ref,
        rec_out_ref,
        hist_out_ref,
        hidden_bufs,
        out_bufs,
        *kda_weights_and_state,
        # pyrefly: ignore[bad-argument-count]
        hidden_sems,
        out_sems,
        weight_sems,  # type: ignore[bad-argument-count] -- starred scratch spans the fixed refs (as in _kda_layer_body)
        tile=kda_tile,
        decay_lower_bound=-5.0,
        epsilon=eps,
        checkpoint=(None, rec_checkpoint_ref, hist_checkpoint_ref),
        segments=(segment_ref, count),
        tiles=(
            first_tile,
            end_tile,
            step == pm.HALF_STEPS,
            step == pm.HALF_STEPS * (blocks + 1) - 1,
            step == 0,
            (attending & (ring_step == 0)) if block_tiles < pm.HALF_STEPS else None,
        ),
        block_tiles=block_tiles,
        input_norm=input_norm,
    )


def _kda_layer_body_a2a(
    slot_ref,
    segment_ref,
    active_ref,
    tables_ref,
    length_ref,
    prefix_ref,
    blocks_ref,
    folded_ref,
    norm_ref,
    gate_family_ref,
    fb_family_ref,
    proj_family_ref,
    conv_family_ref,
    alog_family_ref,
    dt_family_ref,
    kda_norm_family_ref,
    rec_in_ref,
    hist_in_ref,
    normalized_ref,
    gathered_ref,
    rec_out_ref,
    hist_out_ref,
    rec_checkpoint_ref,
    hist_checkpoint_ref,
    attended_ref,
    owner_rows_ref,
    pool,
    *scratch,
    eps,
    count,
    blocks,
    kda_tile,
    split_phase=False,
    fp8_stage_p=False,
    dual_gather=False,
    column_blocks=None,
):
    slot = slot_ref[0]
    step = pl.program_id(1)
    phase, ring_step = step // pm.HALF_STEPS, step % pm.HALF_STEPS
    stripe = pm.window_stripe(prefix_ref.shape[0])
    odd = 3 if dual_gather else 0
    mix_small, gather_sems = scratch[:8], scratch[8:11]
    odd_gather_sems = scratch[11:14] if dual_gather else ()
    a2a_sems = scratch[11 + odd : 13 + odd]
    fp8 = _kda_fp8_window(stripe)
    kda_scratch = scratch[13 + odd : -1] if fp8 else scratch[13 + odd :]
    input_norm = (norm_ref, scratch[-1]) if fp8 else None

    def mix_views():
        offset, result = 0, []
        for shape, dtype in _kda_mix_shapes(stripe, fp8):
            size = pool_alias.storage_size(shape, jnp.dtype(dtype))
            result.append(
                pool_alias.view(
                    pool.at[pl.ds(offset // 128, size // 128), :], shape, dtype
                )
            )
            offset += size
        return result

    def do_normalize(norm_block, load_weights: bool):
        rows = pl.ds(norm_block * stripe, stripe)
        pm.mixture_norm_tiled_body(
            active_ref,
            prefix_ref.at[rows],
            blocks_ref.at[:, rows],
            folded_ref,
            norm_ref,
            normalized_ref.at[rows],
            *mix_views(),
            *mix_small,
            tile=stripe,
            eps=eps,
            unit_fp8=fp8,
            load_weights=load_weights,
        )

    _kda_gather_and_attend_step(
        slot,
        step,
        phase,
        ring_step,
        stripe,
        segment_ref,
        tables_ref,
        length_ref,
        normalized_ref,
        gathered_ref,
        attended_ref,
        owner_rows_ref,
        gate_family_ref,
        fb_family_ref,
        proj_family_ref,
        conv_family_ref,
        alog_family_ref,
        dt_family_ref,
        kda_norm_family_ref,
        rec_in_ref,
        hist_in_ref,
        rec_out_ref,
        hist_out_ref,
        rec_checkpoint_ref,
        hist_checkpoint_ref,
        gather_sems,
        odd_gather_sems,
        a2a_sems,
        kda_scratch,
        eps=eps,
        count=count,
        blocks=blocks,
        kda_tile=kda_tile,
        split_phase=split_phase,
        fp8_stage_p=fp8_stage_p,
        dual_gather=dual_gather,
        column_blocks=column_blocks,
        input_norm=input_norm,
        normalize_block_fn=do_normalize,
    )


def _kda_layer_body_a2a_pre_normed(
    slot_ref,
    segment_ref,
    tables_ref,
    length_ref,
    pre_normed_ref,
    gate_family_ref,
    fb_family_ref,
    proj_family_ref,
    conv_family_ref,
    alog_family_ref,
    dt_family_ref,
    kda_norm_family_ref,
    rec_in_ref,
    hist_in_ref,
    gathered_ref,
    rec_out_ref,
    hist_out_ref,
    rec_checkpoint_ref,
    hist_checkpoint_ref,
    attended_ref,
    owner_rows_ref,
    *scratch,
    eps,
    count,
    blocks,
    kda_tile,
    split_phase=False,
    fp8_stage_p=False,
    dual_gather=False,
    column_blocks=None,
):
    slot = slot_ref[0]
    step = pl.program_id(1)
    phase, ring_step = step // pm.HALF_STEPS, step % pm.HALF_STEPS
    stripe = pm.window_stripe(pre_normed_ref.shape[0])
    odd = 3 if dual_gather else 0
    gather_sems = scratch[:3]
    odd_gather_sems = scratch[3:6] if dual_gather else ()
    a2a_sems = scratch[3 + odd : 5 + odd]
    kda_scratch = scratch[5 + odd :]
    _kda_gather_and_attend_step(
        slot,
        step,
        phase,
        ring_step,
        stripe,
        segment_ref,
        tables_ref,
        length_ref,
        pre_normed_ref,
        gathered_ref,
        attended_ref,
        owner_rows_ref,
        gate_family_ref,
        fb_family_ref,
        proj_family_ref,
        conv_family_ref,
        alog_family_ref,
        dt_family_ref,
        kda_norm_family_ref,
        rec_in_ref,
        hist_in_ref,
        rec_out_ref,
        hist_out_ref,
        rec_checkpoint_ref,
        hist_checkpoint_ref,
        gather_sems,
        odd_gather_sems,
        a2a_sems,
        kda_scratch,
        eps=eps,
        count=count,
        blocks=blocks,
        kda_tile=kda_tile,
        split_phase=split_phase,
        fp8_stage_p=fp8_stage_p,
        dual_gather=dual_gather,
        column_blocks=column_blocks,
        input_norm=None,
        normalize_block_fn=None,
    )


def owner_output_projection(
    owner_rows,
    output_projection,
    prefix,
    *,
    token_owner,
    length,
    replace,
    output_projection_scale=None,
    rank_major: bool = False,
):
    if rank_major:
        ranks, block_count, stripe, cols = owner_rows.shape
        heads = owner_rows.transpose(1, 2, 0, 3).reshape(
            block_count * stripe, ranks * cols
        )
    else:
        block_count, ranks, stripe, cols = owner_rows.shape
        heads = owner_rows.transpose(0, 2, 1, 3).reshape(
            block_count * stripe, ranks * cols
        )
    is_fp8 = output_projection.dtype == jnp.float8_e4m3fn
    if is_fp8:
        if output_projection_scale is None:
            raise ValueError("FP8 owner W_O requires output_projection_scale")
        scale_row = (
            output_projection_scale[None, :]
            if output_projection_scale.ndim == 1
            else output_projection_scale[0:1, :]
        )
        weight_f32 = output_projection.astype(jnp.float32) * scale_row
        projection = jnp.dot(
            heads.astype(jnp.float32), weight_f32, preferred_element_type=jnp.float32
        )
    else:
        if output_projection_scale is not None:
            raise ValueError(
                "output_projection_scale is only valid for float8_e4m3fn W_O"
            )
        projection = jnp.dot(
            heads, output_projection, preferred_element_type=jnp.float32
        )
    local = jnp.arange(block_count * stripe, dtype=jnp.int32)
    valid = pm.sp_token_index(token_owner, local, stripe) < length
    residual = jnp.where(replace, 0.0, prefix.astype(jnp.float32))
    return (residual + jnp.where(valid[:, None], projection, 0.0)).astype(jnp.bfloat16)


OWNER_PROJECTION_COLS = 3584
OWNER_PROJECTION_SOURCES = 4
_OWNER_PROJECTION_FP8_SOURCES = 8


def _owner_projection_kernel(
    scalars_ref,
    rows_ref,
    weight_ref,
    prefix_ref,
    out_ref,
    acc_ref,
    *,
    stripe,
    rank_major: bool = False,
):
    source_group = pl.program_id(1)
    if rank_major:
        sources, block_count = rows_ref.shape[:2]
    else:
        block_count, sources = rows_ref.shape[:2]
    rows = block_count * stripe

    @pl.when(source_group == 0)
    def init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, acc_ref.dtype)

    if rank_major:
        heads = jnp.concatenate(
            [rows_ref[j].reshape(rows, HEAD_COLS) for j in range(sources)], axis=1
        )
    else:
        heads = jnp.concatenate(
            [rows_ref[:, j].reshape(rows, HEAD_COLS) for j in range(sources)], axis=1
        )
    acc_ref[...] += jnp.dot(heads, weight_ref[...], preferred_element_type=jnp.float32)

    @pl.when(source_group == pl.num_programs(1) - 1)
    def finish():
        layer, length, token_owner = scalars_ref[0], scalars_ref[1], scalars_ref[2]
        local = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
        valid = pm.sp_token_index(token_owner, local, stripe) < length
        residual = jnp.where(layer % 9 == 0, 0.0, prefix_ref[...].astype(jnp.float32))
        out_ref[...] = (residual + jnp.where(valid, acc_ref[...], 0.0)).astype(
            out_ref.dtype
        )


def _owner_projection_fp8_kernel(
    scalars_ref,
    rows_ref,
    weight_ref,
    w_scale_ref,
    prefix_ref,
    out_ref,
    acc_ref,
    *,
    stripe,
    rank_major: bool = False,
):
    source_group = pl.program_id(1)
    if rank_major:
        sources, block_count = rows_ref.shape[:2]
    else:
        block_count, sources = rows_ref.shape[:2]
    rows = block_count * stripe

    @pl.when(source_group == 0)
    def init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, acc_ref.dtype)

    if rank_major:
        heads = jnp.concatenate(
            [rows_ref[j].reshape(rows, HEAD_COLS) for j in range(sources)], axis=1
        )
    else:
        heads = jnp.concatenate(
            [rows_ref[:, j].reshape(rows, HEAD_COLS) for j in range(sources)], axis=1
        )
    q_heads, row_scale = _quantize_rows_fp8(heads)
    dot = jnp.dot(q_heads, weight_ref[...], preferred_element_type=jnp.float32)
    acc_ref[...] += dot * row_scale

    @pl.when(source_group == pl.num_programs(1) - 1)
    def finish():
        layer, length, token_owner = scalars_ref[0], scalars_ref[1], scalars_ref[2]
        local = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
        valid = pm.sp_token_index(token_owner, local, stripe) < length
        residual = jnp.where(layer % 9 == 0, 0.0, prefix_ref[...].astype(jnp.float32))
        proj = acc_ref[...] * w_scale_ref[0:1, :]
        out_ref[...] = (residual + jnp.where(valid, proj, 0.0)).astype(out_ref.dtype)


def owner_output_projection_pallas(
    owner_rows,
    output_projections,
    prefix,
    *,
    layer,
    length,
    token_owner,
    output_projection_scales=None,
    interpret=False,
    rank_major: bool = False,
):
    if rank_major:
        ranks, block_count, stripe, cols = owner_rows.shape
    else:
        block_count, ranks, stripe, cols = owner_rows.shape
    rows = block_count * stripe
    width = output_projections.shape[-1]
    if ranks != RANKS or cols != HEAD_COLS or width % OWNER_PROJECTION_COLS:
        raise ValueError(
            "owner rows must have 32 ranks x 384 cols and W_O [slots, 12288, 7168k]"
        )
    is_fp8 = output_projections.dtype == jnp.float8_e4m3fn
    scalars = jnp.stack(
        [jnp.asarray(v, jnp.int32).reshape(()) for v in (layer, length, token_owner)]
    )
    tn = OWNER_PROJECTION_COLS
    if is_fp8:
        if output_projection_scales is None:
            raise ValueError("FP8 owner W_O requires output_projection_scales")
        if output_projection_scales.shape != (output_projections.shape[0], 8, width):
            raise ValueError(
                "FP8 owner W_O scales must be "
                f"{(output_projections.shape[0], 8, width)}, "
                f"got {output_projection_scales.shape}"
            )
        group = _OWNER_PROJECTION_FP8_SOURCES
        rows_spec = (
            pl.BlockSpec(
                (group, block_count, stripe, HEAD_COLS), lambda n, s, sc: (s, 0, 0, 0)
            )
            if rank_major
            else pl.BlockSpec(
                (block_count, group, stripe, HEAD_COLS), lambda n, s, sc: (0, s, 0, 0)
            )
        )
        return pl.pallas_call(
            functools.partial(
                _owner_projection_fp8_kernel, stripe=stripe, rank_major=rank_major
            ),
            out_shape=jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(width // tn, RANKS // group),
                in_specs=[
                    rows_spec,
                    pl.BlockSpec(
                        (None, group * HEAD_COLS, tn), lambda n, s, sc: (sc[0], s, n)
                    ),
                    pl.BlockSpec((None, 8, tn), lambda n, s, sc: (sc[0], 0, n)),
                    pl.BlockSpec((rows, tn), lambda n, s, sc: (0, n)),
                ],
                out_specs=pl.BlockSpec((rows, tn), lambda n, s, sc: (0, n)),
                scratch_shapes=[pltpu.VMEM((rows, tn), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
                vmem_limit_bytes=64 * 1024**2,
            ),
            interpret=interpret,
            name="kimi_prefill_kda_owner_projection_fp8",
        )(scalars, owner_rows, output_projections, output_projection_scales, prefix)

    if output_projection_scales is not None:
        raise ValueError("output_projection_scales is only valid for float8_e4m3fn W_O")

    group = OWNER_PROJECTION_SOURCES
    rows_spec = (
        pl.BlockSpec(
            (group, block_count, stripe, HEAD_COLS), lambda n, s, sc: (s, 0, 0, 0)
        )
        if rank_major
        else pl.BlockSpec(
            (block_count, group, stripe, HEAD_COLS), lambda n, s, sc: (0, s, 0, 0)
        )
    )
    return pl.pallas_call(
        functools.partial(
            _owner_projection_kernel, stripe=stripe, rank_major=rank_major
        ),
        out_shape=jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(width // tn, RANKS // group),
            in_specs=[
                rows_spec,
                pl.BlockSpec(
                    (None, group * HEAD_COLS, tn), lambda n, s, sc: (sc[0], s, n)
                ),
                pl.BlockSpec((rows, tn), lambda n, s, sc: (0, n)),
            ],
            out_specs=pl.BlockSpec((rows, tn), lambda n, s, sc: (0, n)),
            scratch_shapes=[pltpu.VMEM((rows, tn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=64 * 1024**2,
        ),
        interpret=interpret,
        name="kimi_prefill_kda_owner_projection",
    )(scalars, owner_rows, output_projections, prefix)


def make_norm_gather_kda_sp_a2a(
    mesh,
    *,
    seq_len,
    segments,
    token_ranks=None,
    collective_id=38,
    eps=1e-5,
    kda_tile=256,
    fp8_stage_p: bool | None = None,
    dual_gather: bool | None = None,
    ring_set: bool | None = None,
):
    if mesh.size != RANKS or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("norm/gather/KDA (all-to-all) requires a flat k3_sp TP32 mesh")
    pm.check_window(seq_len, "fused norm/gather/KDA (all-to-all)")
    if not segments:
        raise ValueError("the KDA layer runs packed windows")
    state_lead = (segments,)
    rows, width = seq_len // RANKS, HIDDEN
    stripe = pm.window_stripe(rows)
    blocks = rows // stripe
    if (RANKS * stripe) % kda_tile or kda_tile % _KDA_CHUNK:
        raise ValueError(
            f"kda_tile {kda_tile} must divide the {RANKS * stripe}-row SP block"
        )
    block_tiles = RANKS * stripe // kda_tile
    split_phase = kda_tile == 256 and 2 * block_tiles <= pm.HALF_STEPS - 1
    use_fp8_stage_p = split_phase and (
        bool(_config().prefill_kda_a2a_fp8)
        if fp8_stage_p is None
        else bool(fp8_stage_p)
    )
    use_dual_gather = (
        bool(_config().prefill_kda_dual_gather)
        if dual_gather is None
        else bool(dual_gather)
    )
    use_ring_set = (
        bool(_config().prefill_kda_ring_set) if ring_set is None else bool(ring_set)
    )
    cycles = KDA_RING_CYCLES if use_ring_set else pm.CYCLES
    column_blocks = KDA_RING_COLUMN_BLOCKS if use_ring_set else None
    if token_ranks is None:
        token_ranks = tuple(range(RANKS))
    tables = jnp.asarray(
        build_kda_a2a_tables(list(mesh.devices.flat), token_ranks, cycles), jnp.int32
    )
    streams = len(_half_ring_streams(width))
    fp8 = _kda_fp8_window(stripe)
    window_dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    pool_bytes = sum(
        pool_alias.storage_size(shape, jnp.dtype(dtype))
        for shape, dtype in _kda_mix_shapes(stripe, fp8)
    )
    shapes = (
        jax.ShapeDtypeStruct((width, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((128, 384), jnp.bfloat16),
        jax.ShapeDtypeStruct((width, 1664), jnp.bfloat16),
        jax.ShapeDtypeStruct((4, 3, 3, 128), jnp.float32),
        jax.ShapeDtypeStruct((3, 1), jnp.float32),
        jax.ShapeDtypeStruct((3, 128), jnp.float32),
        jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    a2a_slots = (
        4 if (split_phase and not use_dual_gather) else (3 if split_phase else 2)
    )
    kda_scratch_shapes = (
        kda_split_prefill_scratch_shapes(
            width,
            kda_tile,
            *shapes,
            hidden_dtype=window_dtype,
            fp8_stage_p=use_fp8_stage_p,
        )
        if split_phase
        else pm.kda_prefill_tiled_scratch_shapes(
            width, kda_tile, *shapes, store_out=True, hidden_dtype=window_dtype
        )
    )
    scratch = (
        pltpu.VMEM((pool_bytes // 128, 128), jnp.uint8),
        *pm.mixture_norm_tiled_scratch_shapes(width, stripe)[4:],
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA,
        *(
            (
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA,
            )
            if use_dual_gather
            else ()
        ),
        pltpu.SemaphoreType.DMA((a2a_slots,)),
        pltpu.SemaphoreType.DMA((a2a_slots,)),
        *kda_scratch_shapes,
        *((pltpu.VMEM((1, width), jnp.bfloat16),) if fp8 else ()),
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    output_shapes = (
        jax.ShapeDtypeStruct((rows, width), window_dtype),
        jax.ShapeDtypeStruct((seq_len, width), window_dtype),
        jax.ShapeDtypeStruct((*state_lead, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((*state_lead, 3, 3, 3, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((*state_lead, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((*state_lead, 3, 3, 3, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, HEAD_COLS), jnp.bfloat16),
        jax.ShapeDtypeStruct((RANKS, blocks, stripe, HEAD_COLS), jnp.bfloat16),
    )
    grid_phases = blocks + 1 if split_phase else blocks + 2
    call = pl.pallas_call(
        functools.partial(
            _kda_layer_body_a2a,
            count=segments,
            eps=eps,
            blocks=blocks,
            kda_tile=kda_tile,
            split_phase=split_phase,
            fp8_stage_p=use_fp8_stage_p,
            dual_gather=use_dual_gather,
            column_blocks=column_blocks,
        ),
        out_shape=output_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            grid=(1, pm.HALF_STEPS * grid_phases),
            in_specs=(hbm,) * 13,
            out_specs=(hbm,) * len(output_shapes),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=64 * 1024**2,
        ),
        name="kimi_prefill_norm_gather_kda_a2a",
    )

    out_specs = (
        P("k3_sp"),
        P(),
        P("k3_sp"),
        P("k3_sp"),
        P("k3_sp"),
        P("k3_sp"),
        P("k3_sp"),
        P("k3_sp"),
    )

    def _validate_owner_wo(w: dict[str, Any]) -> bool:
        output_projection = w["k_o"]
        is_chip_pair = output_projection.shape[-2:] == (RANKS * HEAD_COLS, width // 2)
        if not is_chip_pair and output_projection.shape[-2:] != (
            RANKS * HEAD_COLS,
            width,
        ):
            raise ValueError(
                "the all-to-all KDA layer needs [slots, 12288, 7168] or "
                "chip-pair [slots, 12288, 3584] W_O"
            )
        if is_chip_pair and (
            output_projection.dtype != jnp.bfloat16 or w.get("k_o_scale") is not None
        ):
            raise ValueError(
                "chip-pair KDA W_O requires BF16 weights without k_o_scale"
            )
        return is_chip_pair

    def _project_owner_rows(
        owner_rows, prefix_local, w: dict[str, Any], *, is_chip_pair: bool, slot, length
    ):
        output_projection = w["k_o"]
        layer = jnp.asarray(slot, jnp.int32).reshape(-1)[0]
        len_scalar = jnp.asarray(length).reshape(-1)[0]
        rank = jax.lax.axis_index("k3_sp")
        if is_chip_pair:
            return owner_output_projection_chip_pair_pallas(
                owner_rows,
                output_projection,
                prefix_local,
                layer=layer,
                length=len_scalar,
                own_even=tables[0, 2, rank & ~1],
                own_odd=tables[0, 2, rank | 1],
                rank_major=True,
            )
        return owner_output_projection_pallas(
            owner_rows,
            output_projection,
            prefix_local,
            layer=layer,
            length=len_scalar,
            token_owner=tables[0, 2, rank],
            output_projection_scales=w.get("k_o_scale"),
            rank_major=True,
        )

    def local(
        prefix,
        blocks_in,
        folded,
        norm,
        weights,
        rec,
        hist,
        active,
        length,
        slot,
        segment_desc,
    ):
        w = jax.tree.map(lambda value: value[0], weights)
        is_chip_pair = _validate_owner_wo(w)
        outputs = call(
            slot,
            segment_desc,
            active,
            tables,
            length,
            prefix[0],
            blocks_in[0],
            folded[0],
            norm[0],
            w["k_gate"],
            w["k_fb"],
            w["k_projection"],
            w["k_conv"],
            w["k_a_log"],
            w["k_dt"],
            w["k_norm"],
            rec[0],
            hist[0],
        )
        normalized, gathered, new_rec, new_hist, rec_at, hist_at, _, owner_rows = (
            outputs
        )
        final = _project_owner_rows(
            owner_rows,
            prefix[0],
            w,
            is_chip_pair=is_chip_pair,
            slot=slot,
            length=length,
        )
        attended_stub = jnp.zeros((1, 0, HEAD_COLS), jnp.bfloat16)
        return (
            normalized[None],
            gathered,
            attended_stub,
            new_rec[None],
            new_hist[None],
            final[None],
            rec_at[None],
            hist_at[None],
        )

    jit_with_norm = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("k3_sp"),) * 7 + (P(), P(), P(), P()),
            out_specs=out_specs,
            check_vma=False,
        )
    )
    jit_pre_normed_holder: list[Any] = []

    def _get_jit_pre_normed():
        if not jit_pre_normed_holder:
            scratch_pre_normed = (
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA,
                *(
                    (
                        pltpu.SemaphoreType.DMA((streams,)),
                        pltpu.SemaphoreType.DMA((streams,)),
                        pltpu.SemaphoreType.DMA,
                    )
                    if use_dual_gather
                    else ()
                ),
                pltpu.SemaphoreType.DMA((a2a_slots,)),
                pltpu.SemaphoreType.DMA((a2a_slots,)),
                *kda_scratch_shapes,
            )
            call_pre_normed = pl.pallas_call(
                functools.partial(
                    _kda_layer_body_a2a_pre_normed,
                    count=segments,
                    eps=eps,
                    blocks=blocks,
                    kda_tile=kda_tile,
                    split_phase=split_phase,
                    fp8_stage_p=use_fp8_stage_p,
                    dual_gather=use_dual_gather,
                    column_blocks=column_blocks,
                ),
                out_shape=output_shapes[1:],
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=4,
                    grid=(1, pm.HALF_STEPS * grid_phases),
                    in_specs=(hbm,) * 10,
                    out_specs=(hbm,) * (len(output_shapes) - 1),
                    scratch_shapes=scratch_pre_normed,
                ),
                compiler_params=pltpu.CompilerParams(
                    collective_id=collective_id,
                    vmem_limit_bytes=64 * 1024**2,
                ),
                name="kimi_prefill_gather_kda_a2a_prenorm",
            )

            def local_pre_normed(
                prefix, pre_normed, weights, rec, hist, length, slot, segment_desc
            ):
                w = jax.tree.map(lambda value: value[0], weights)
                is_chip_pair = _validate_owner_wo(w)
                outputs = call_pre_normed(
                    slot,
                    segment_desc,
                    tables,
                    length,
                    pre_normed[0],
                    w["k_gate"],
                    w["k_fb"],
                    w["k_projection"],
                    w["k_conv"],
                    w["k_a_log"],
                    w["k_dt"],
                    w["k_norm"],
                    rec[0],
                    hist[0],
                )
                gathered, new_rec, new_hist, rec_at, hist_at, _, owner_rows = outputs
                final = _project_owner_rows(
                    owner_rows,
                    prefix[0],
                    w,
                    is_chip_pair=is_chip_pair,
                    slot=slot,
                    length=length,
                )
                attended_stub = jnp.zeros((1, 0, HEAD_COLS), jnp.bfloat16)
                return (
                    pre_normed,
                    gathered,
                    attended_stub,
                    new_rec[None],
                    new_hist[None],
                    final[None],
                    rec_at[None],
                    hist_at[None],
                )

            jit_pre_normed_holder.append(
                jax.jit(
                    jax.shard_map(
                        local_pre_normed,
                        mesh=mesh,
                        in_specs=(P("k3_sp"),) * 5 + (P(), P(), P()),
                        out_specs=out_specs,
                        check_vma=False,
                    )
                )
            )
        return jit_pre_normed_holder[0]

    def run(
        prefix,
        blocks_in,
        folded,
        norm,
        weights,
        rec,
        hist,
        active,
        length,
        slot,
        segment_desc,
        *,
        pre_normalized=None,
    ):
        if pre_normalized is not None and not fp8:
            return _get_jit_pre_normed()(
                prefix, pre_normalized, weights, rec, hist, length, slot, segment_desc
            )
        return jit_with_norm(
            prefix,
            blocks_in,
            folded,
            norm,
            weights,
            rec,
            hist,
            active,
            length,
            slot,
            segment_desc,
        )

    return run


def _sync_partner(sem_ref, partner):
    pl.semaphore_signal(
        sem_ref, 1, device_id=(partner,), device_id_type=pl.DeviceIdType.MESH
    )
    pl.semaphore_wait(sem_ref, 1)


def _partner_enter(partner, ready_sem):
    barrier = pltpu.get_barrier_semaphore()
    pl.semaphore_signal(
        barrier, 1, device_id=(partner,), device_id_type=pl.DeviceIdType.MESH
    )
    pl.semaphore_wait(barrier, 1)
    _sync_partner(ready_sem, partner)


def _sp_token_index_ranks(token_owner, local, stripe: int, num_ranks: int):
    return (
        (local // stripe) * (num_ranks * stripe)
        + jnp.asarray(token_owner, jnp.int32) * stripe
        + (local % stripe)
    )


def owner_output_projection_chip_pair_ref(
    owner_rows,
    output_projection_half,
    prefix,
    *,
    layer,
    length,
    own_even,
    own_odd,
    sp_axis: str = "k3_sp",
    rank_major: bool = False,
):
    if rank_major:
        num_ranks, blocks, stripe, cols = owner_rows.shape
    else:
        blocks, num_ranks, stripe, cols = owner_rows.shape
    rows = blocks * stripe
    total_cols, half_w = output_projection_half.shape
    if (
        num_ranks % 2
        or total_cols != num_ranks * cols
        or prefix.shape != (rows, 2 * half_w)
        or output_projection_half.dtype != jnp.bfloat16
    ):
        raise ValueError(
            "owner_output_projection_chip_pair_ref expects even R, "
            "W_O_half=[R*cols, W//2] bf16, and prefix=[rows, W]"
        )
    swap_pairs = [(i, i ^ 1) for i in range(num_ranks)]
    parity = jax.lax.axis_index(sp_axis) & 1
    own_tr = (
        owner_rows.transpose(1, 2, 0, 3)
        if rank_major
        else owner_rows.transpose(0, 2, 1, 3)
    )
    own_flat = own_tr.reshape(rows, total_cols)
    partner_flat = jax.lax.ppermute(own_flat, sp_axis, swap_pairs)
    even_flat = jnp.where(parity == 0, own_flat, partner_flat)
    odd_flat = jnp.where(parity == 0, partner_flat, own_flat)

    own_px = jax.lax.dynamic_slice_in_dim(prefix, parity * half_w, half_w, axis=1)
    send_px = jax.lax.dynamic_slice_in_dim(
        prefix, (1 - parity) * half_w, half_w, axis=1
    )
    recv_px = jax.lax.ppermute(send_px, sp_axis, swap_pairs)
    even_px = jnp.where(parity == 0, own_px, recv_px)
    odd_px = jnp.where(parity == 0, recv_px, own_px)

    group = min(OWNER_PROJECTION_SOURCES, num_ranks)
    acc_even = jnp.zeros((rows, half_w), jnp.float32)
    acc_odd = jnp.zeros((rows, half_w), jnp.float32)
    for s in range(num_ranks // group):
        sl = slice(s * group * cols, (s + 1) * group * cols)
        w = output_projection_half[sl]
        acc_even = acc_even + jnp.dot(
            even_flat[:, sl], w, preferred_element_type=jnp.float32
        )
        acc_odd = acc_odd + jnp.dot(
            odd_flat[:, sl], w, preferred_element_type=jnp.float32
        )

    local = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    len_scalar = jnp.asarray(length, jnp.int32).reshape(())
    layer_scalar = jnp.asarray(layer, jnp.int32).reshape(())
    valid_even = _sp_token_index_ranks(own_even, local, stripe, num_ranks) < len_scalar
    valid_odd = _sp_token_index_ranks(own_odd, local, stripe, num_ranks) < len_scalar
    res_even = jnp.where(layer_scalar % 9 == 0, 0.0, even_px.astype(jnp.float32))
    res_odd = jnp.where(layer_scalar % 9 == 0, 0.0, odd_px.astype(jnp.float32))
    out_even = (res_even + jnp.where(valid_even, acc_even, 0.0)).astype(prefix.dtype)
    out_odd = (res_odd + jnp.where(valid_odd, acc_odd, 0.0)).astype(prefix.dtype)

    own_out = jnp.where(parity == 0, out_even, out_odd)
    for_partner = jnp.where(parity == 0, out_odd, out_even)
    from_partner = jax.lax.ppermute(for_partner, sp_axis, swap_pairs)
    left = jnp.where(parity == 0, own_out, from_partner)
    right = jnp.where(parity == 0, from_partner, own_out)
    return jnp.concatenate((left, right), axis=1)


def _kda_owner_projection_chip_pair_kernel(
    scalars_ref,
    owner_rows_hbm,
    k_o_half_hbm,
    prefix_hbm,
    out_hbm,
    both_rows_bufs,
    w_bufs,
    acc_even_vmem,
    acc_odd_vmem,
    prefix_half_vmem,
    out_half_vmem,
    local_sems,
    send_sems,
    recv_sems,
    sync_sems,
    *,
    stripe: int,
):
    rank = jax.lax.axis_index("k3_sp")
    parity = rank & 1
    partner = rank ^ 1
    blocks = owner_rows_hbm.shape[0]
    rows = blocks * stripe
    group = OWNER_PROJECTION_SOURCES
    steps = RANKS // group
    layer = scalars_ref[0]

    _partner_enter(partner, sync_sems.at[0])

    own_px = prefix_hbm.at[:, pl.ds(parity * HALF_HIDDEN, HALF_HIDDEN)]
    send_px = prefix_hbm.at[:, pl.ds((1 - parity) * HALF_HIDDEN, HALF_HIDDEN)]
    px_local = pltpu.make_async_copy(
        own_px, prefix_half_vmem.at[parity], local_sems.at[4]
    )
    px_remote = pltpu.make_async_remote_copy(
        send_px,
        prefix_half_vmem.at[parity],
        send_sems.at[2],
        recv_sems.at[2],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    px_local.start()
    px_remote.start()

    def start_step(s):
        slot = s % 2
        src_slice = owner_rows_hbm.at[:, pl.ds(s * group, group), :, :]
        pltpu.make_async_copy(
            src_slice, both_rows_bufs.at[slot, parity], local_sems.at[slot]
        ).start()
        pltpu.make_async_remote_copy(
            src_slice,
            both_rows_bufs.at[slot, parity],
            send_sems.at[slot],
            recv_sems.at[slot],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
        pltpu.make_async_copy(
            k_o_half_hbm.at[layer, pl.ds(s * group * HEAD_COLS, group * HEAD_COLS), :],
            w_bufs.at[slot],
            local_sems.at[2 + slot],
        ).start()

    def wait_step(s):
        slot = s % 2
        pltpu.make_async_copy(
            both_rows_bufs.at[slot, parity],
            both_rows_bufs.at[slot, parity],
            local_sems.at[slot],
        ).wait()
        src_slice = owner_rows_hbm.at[:, pl.ds(s * group, group), :, :]
        pltpu.make_async_remote_copy(
            src_slice,
            both_rows_bufs.at[slot, parity],
            send_sems.at[slot],
            recv_sems.at[slot],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        ).wait_send()
        pltpu.make_async_copy(
            both_rows_bufs.at[slot, 1 - parity],
            both_rows_bufs.at[slot, 1 - parity],
            recv_sems.at[slot],
        ).wait()
        pltpu.make_async_copy(
            w_bufs.at[slot], w_bufs.at[slot], local_sems.at[2 + slot]
        ).wait()

    start_step(0)
    start_step(1)
    acc_even_vmem[...] = jnp.zeros((rows, HALF_HIDDEN), jnp.float32)
    acc_odd_vmem[...] = jnp.zeros((rows, HALF_HIDDEN), jnp.float32)

    @pl.loop(0, steps)
    def run_step(s):
        slot = s % 2
        wait_step(s)
        even_flat = jnp.concatenate(
            [
                both_rows_bufs[slot, 0, :, j].reshape(rows, HEAD_COLS)
                for j in range(group)
            ],
            axis=1,
        )
        odd_flat = jnp.concatenate(
            [
                both_rows_bufs[slot, 1, :, j].reshape(rows, HEAD_COLS)
                for j in range(group)
            ],
            axis=1,
        )
        w = w_bufs[slot][...]
        acc_even_vmem[...] += jnp.dot(even_flat, w, preferred_element_type=jnp.float32)
        acc_odd_vmem[...] += jnp.dot(odd_flat, w, preferred_element_type=jnp.float32)

        @pl.when(s + 2 < steps)
        def prefetch_next():
            _sync_partner(sync_sems.at[1 + slot], partner)
            start_step(s + 2)

    px_local.wait()
    px_remote.wait_send()
    pltpu.make_async_copy(
        prefix_half_vmem.at[1 - parity],
        prefix_half_vmem.at[1 - parity],
        recv_sems.at[2],
    ).wait()

    length, own_even, own_odd = scalars_ref[1], scalars_ref[2], scalars_ref[3]
    local = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    valid_even = pm.sp_token_index(own_even, local, stripe) < length
    valid_odd = pm.sp_token_index(own_odd, local, stripe) < length
    res_even = jnp.where(layer % 9 == 0, 0.0, prefix_half_vmem[0].astype(jnp.float32))
    res_odd = jnp.where(layer % 9 == 0, 0.0, prefix_half_vmem[1].astype(jnp.float32))
    out_half_vmem[0, ...] = (
        res_even + jnp.where(valid_even, acc_even_vmem[...], 0.0)
    ).astype(out_hbm.dtype)
    out_half_vmem[1, ...] = (
        res_odd + jnp.where(valid_odd, acc_odd_vmem[...], 0.0)
    ).astype(out_hbm.dtype)

    own_out = out_half_vmem.at[parity]
    partner_out = out_half_vmem.at[1 - parity]
    out_local = pltpu.make_async_copy(
        own_out,
        out_hbm.at[:, pl.ds(parity * HALF_HIDDEN, HALF_HIDDEN)],
        local_sems.at[5],
    )
    out_remote = pltpu.make_async_remote_copy(
        partner_out,
        out_hbm.at[:, pl.ds(parity * HALF_HIDDEN, HALF_HIDDEN)],
        send_sems.at[3],
        recv_sems.at[3],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    out_local.start()
    out_remote.start()
    out_local.wait()
    out_remote.wait_send()
    pltpu.make_async_copy(
        out_hbm.at[:, pl.ds((1 - parity) * HALF_HIDDEN, HALF_HIDDEN)],
        out_hbm.at[:, pl.ds((1 - parity) * HALF_HIDDEN, HALF_HIDDEN)],
        recv_sems.at[3],
    ).wait()
    _sync_partner(sync_sems.at[3], partner)


def _kda_owner_projection_chip_pair_rank_major_kernel(
    scalars_ref,
    owner_rows_rm_hbm,
    k_o_half_hbm,
    prefix_hbm,
    out_hbm,
    own_bufs,
    partner_all_vmem,
    w_bufs,
    acc_even_vmem,
    acc_odd_vmem,
    px_out_half_vmem,
    local_sems,
    send_sems,
    recv_sems,
    sync_sems,
    *,
    stripe: int,
):
    rank = jax.lax.axis_index("k3_sp")
    parity = rank & 1
    partner = rank ^ 1
    blocks = owner_rows_rm_hbm.shape[1]
    rows = blocks * stripe
    group = OWNER_PROJECTION_SOURCES
    steps = RANKS // group
    half_ranks = RANKS // 2
    layer = scalars_ref[0]

    _partner_enter(partner, sync_sems.at[0])

    partner_h0 = pltpu.make_async_remote_copy(
        owner_rows_rm_hbm.at[pl.ds(0, half_ranks)],
        partner_all_vmem.at[pl.ds(0, half_ranks)],
        send_sems.at[0],
        recv_sems.at[0],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    partner_h1 = pltpu.make_async_remote_copy(
        owner_rows_rm_hbm.at[pl.ds(half_ranks, half_ranks)],
        partner_all_vmem.at[pl.ds(half_ranks, half_ranks)],
        send_sems.at[1],
        recv_sems.at[1],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    partner_h0.start()
    partner_h1.start()

    own_px = prefix_hbm.at[:, pl.ds(parity * HALF_HIDDEN, HALF_HIDDEN)]
    send_px = prefix_hbm.at[:, pl.ds((1 - parity) * HALF_HIDDEN, HALF_HIDDEN)]
    px_local = pltpu.make_async_copy(
        own_px, px_out_half_vmem.at[parity], local_sems.at[4]
    )
    px_remote = pltpu.make_async_remote_copy(
        send_px,
        px_out_half_vmem.at[parity],
        send_sems.at[2],
        recv_sems.at[2],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    px_local.start()
    px_remote.start()

    def start_own_and_w(s):
        slot = s % 2
        pltpu.make_async_copy(
            owner_rows_rm_hbm.at[pl.ds(s * group, group)],
            own_bufs.at[slot],
            local_sems.at[slot],
        ).start()
        pltpu.make_async_copy(
            k_o_half_hbm.at[layer, pl.ds(s * group * HEAD_COLS, group * HEAD_COLS), :],
            w_bufs.at[slot],
            local_sems.at[2 + slot],
        ).start()

    start_own_and_w(0)
    start_own_and_w(1)
    acc_even_vmem[...] = jnp.zeros((rows, HALF_HIDDEN), jnp.float32)
    acc_odd_vmem[...] = jnp.zeros((rows, HALF_HIDDEN), jnp.float32)

    @pl.loop(0, steps)
    def run_step(s):
        slot = s % 2
        pltpu.make_async_copy(
            own_bufs.at[slot], own_bufs.at[slot], local_sems.at[slot]
        ).wait()
        pltpu.make_async_copy(
            w_bufs.at[slot], w_bufs.at[slot], local_sems.at[2 + slot]
        ).wait()

        @pl.when(s == 0)
        def wait_partner_first_half():
            pltpu.make_async_copy(
                partner_all_vmem.at[pl.ds(0, half_ranks)],
                partner_all_vmem.at[pl.ds(0, half_ranks)],
                recv_sems.at[0],
            ).wait()

        @pl.when(s == steps // 2)
        def wait_partner_second_half():
            pltpu.make_async_copy(
                partner_all_vmem.at[pl.ds(half_ranks, half_ranks)],
                partner_all_vmem.at[pl.ds(half_ranks, half_ranks)],
                recv_sems.at[1],
            ).wait()

        own_flat = jnp.concatenate(
            [own_bufs[slot, j].reshape(rows, HEAD_COLS) for j in range(group)],
            axis=1,
        )
        part_flat = jnp.concatenate(
            [
                partner_all_vmem[s * group + j].reshape(rows, HEAD_COLS)
                for j in range(group)
            ],
            axis=1,
        )
        even_flat = jnp.where(parity == 0, own_flat, part_flat)
        odd_flat = jnp.where(parity == 0, part_flat, own_flat)
        w = w_bufs[slot][...]
        acc_even_vmem[...] += jnp.dot(even_flat, w, preferred_element_type=jnp.float32)
        acc_odd_vmem[...] += jnp.dot(odd_flat, w, preferred_element_type=jnp.float32)

        @pl.when(s + 2 < steps)
        def prefetch_next():
            start_own_and_w(s + 2)

    partner_h0.wait_send()
    partner_h1.wait_send()
    px_local.wait()
    px_remote.wait_send()
    pltpu.make_async_copy(
        px_out_half_vmem.at[1 - parity],
        px_out_half_vmem.at[1 - parity],
        recv_sems.at[2],
    ).wait()

    length, own_even, own_odd = scalars_ref[1], scalars_ref[2], scalars_ref[3]
    local = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    valid_even = pm.sp_token_index(own_even, local, stripe) < length
    valid_odd = pm.sp_token_index(own_odd, local, stripe) < length
    res_even = jnp.where(layer % 9 == 0, 0.0, px_out_half_vmem[0].astype(jnp.float32))
    res_odd = jnp.where(layer % 9 == 0, 0.0, px_out_half_vmem[1].astype(jnp.float32))
    out_ev = (res_even + jnp.where(valid_even, acc_even_vmem[...], 0.0)).astype(
        out_hbm.dtype
    )
    out_od = (res_odd + jnp.where(valid_odd, acc_odd_vmem[...], 0.0)).astype(
        out_hbm.dtype
    )
    px_out_half_vmem[0, ...] = out_ev
    px_out_half_vmem[1, ...] = out_od

    own_out = px_out_half_vmem.at[parity]
    partner_out = px_out_half_vmem.at[1 - parity]
    out_local = pltpu.make_async_copy(
        own_out,
        out_hbm.at[:, pl.ds(parity * HALF_HIDDEN, HALF_HIDDEN)],
        local_sems.at[5],
    )
    out_remote = pltpu.make_async_remote_copy(
        partner_out,
        out_hbm.at[:, pl.ds(parity * HALF_HIDDEN, HALF_HIDDEN)],
        send_sems.at[3],
        recv_sems.at[3],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    out_local.start()
    out_remote.start()
    out_local.wait()
    out_remote.wait_send()
    pltpu.make_async_copy(
        out_hbm.at[:, pl.ds((1 - parity) * HALF_HIDDEN, HALF_HIDDEN)],
        out_hbm.at[:, pl.ds((1 - parity) * HALF_HIDDEN, HALF_HIDDEN)],
        recv_sems.at[3],
    ).wait()
    _sync_partner(sync_sems.at[1], partner)


def owner_output_projection_chip_pair_pallas(
    owner_rows,
    output_projections_half,
    prefix,
    *,
    layer,
    length,
    own_even,
    own_odd,
    collective_id: int = 55,
    interpret: bool = False,
    sp_axis: str = "k3_sp",
    rank_major: bool = False,
):
    if rank_major:
        ranks, blocks, stripe, cols = owner_rows.shape
    else:
        blocks, ranks, stripe, cols = owner_rows.shape
    rows = blocks * stripe
    if (
        ranks != RANKS
        or cols != HEAD_COLS
        or output_projections_half.ndim != 3
        or output_projections_half.shape[-2:] != (RANKS * HEAD_COLS, HALF_HIDDEN)
        or output_projections_half.dtype != jnp.bfloat16
        or prefix.shape != (rows, HIDDEN)
    ):
        raise ValueError(
            "chip-pair KDA projection expects owner_rows=[blocks, 32, stripe, 384] "
            "(or [32, blocks, stripe, 384] when rank_major=True), "
            "W_O_half=[slots, 12288, 3584] bf16, and prefix=[rows, 7168]"
        )
    if interpret:
        slot_idx = jnp.asarray(layer, jnp.int32).reshape(())
        return owner_output_projection_chip_pair_ref(
            owner_rows,
            output_projections_half[slot_idx],
            prefix,
            layer=slot_idx,
            length=length,
            own_even=own_even,
            own_odd=own_odd,
            sp_axis=sp_axis,
            rank_major=rank_major,
        )
    group = OWNER_PROJECTION_SOURCES
    scalars = jnp.stack(
        [
            jnp.asarray(v, jnp.int32).reshape(())
            for v in (layer, length, own_even, own_odd)
        ]
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    if rank_major:
        return pl.pallas_call(
            functools.partial(
                _kda_owner_projection_chip_pair_rank_major_kernel, stripe=stripe
            ),
            out_shape=jax.ShapeDtypeStruct((rows, HIDDEN), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(1,),
                in_specs=[hbm, hbm, hbm],
                out_specs=hbm,
                scratch_shapes=[
                    pltpu.VMEM((2, group, blocks, stripe, HEAD_COLS), jnp.bfloat16),
                    pltpu.VMEM((RANKS, blocks, stripe, HEAD_COLS), jnp.bfloat16),
                    pltpu.VMEM((2, group * HEAD_COLS, HALF_HIDDEN), jnp.bfloat16),
                    pltpu.VMEM((rows, HALF_HIDDEN), jnp.float32),
                    pltpu.VMEM((rows, HALF_HIDDEN), jnp.float32),
                    pltpu.VMEM((2, rows, HALF_HIDDEN), jnp.bfloat16),
                    pltpu.SemaphoreType.DMA((6,)),
                    pltpu.SemaphoreType.DMA((4,)),
                    pltpu.SemaphoreType.DMA((4,)),
                    pltpu.SemaphoreType.REGULAR((2,)),
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id, vmem_limit_bytes=64 * 1024**2
            ),
            name="kimi_prefill_kda_owner_projection_chip_pair",
        )(scalars, owner_rows, output_projections_half, prefix)

    return pl.pallas_call(
        functools.partial(_kda_owner_projection_chip_pair_kernel, stripe=stripe),
        out_shape=jax.ShapeDtypeStruct((rows, HIDDEN), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[hbm, hbm, hbm],
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((2, 2, blocks, group, stripe, HEAD_COLS), jnp.bfloat16),
                pltpu.VMEM((2, group * HEAD_COLS, HALF_HIDDEN), jnp.bfloat16),
                pltpu.VMEM((rows, HALF_HIDDEN), jnp.float32),
                pltpu.VMEM((rows, HALF_HIDDEN), jnp.float32),
                pltpu.VMEM((2, rows, HALF_HIDDEN), jnp.bfloat16),
                pltpu.VMEM((2, rows, HALF_HIDDEN), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((6,)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.REGULAR((4,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id, vmem_limit_bytes=64 * 1024**2
        ),
        name="kimi_prefill_kda_owner_projection_chip_pair",
    )(scalars, owner_rows, output_projections_half, prefix)


def gather_output_projection_shares_chip_pair(share, *, axis: str):
    if share.dtype != jnp.bfloat16 or share.shape[-1] % 2:
        raise ValueError(
            "chip-pair KDA W_O gather requires BF16 shares with even hidden width"
        )
    half_w = share.shape[-1] // 2
    parity = jax.lax.axis_index(axis) & 1
    even_cols = jax.lax.all_gather(share[..., :half_w], axis, axis=2, tiled=True)
    odd_cols = jax.lax.all_gather(share[..., half_w:], axis, axis=2, tiled=True)
    return (jnp.where(parity == 0, even_cols, odd_cols),)


def gather_output_projection_shares(share, *, axis):
    return (jax.lax.all_gather(share, axis, axis=2, tiled=True),)


def gather_output_projection_shares_fp8(share, *, axis):
    share_f32 = share.astype(jnp.float32)
    global_max = jnp.maximum(
        jax.lax.pmax(jnp.max(jnp.abs(share_f32), axis=2, keepdims=True), axis),
        _FP8_SCALE_EPS,
    )
    q_share = (share_f32 * (_FP8_E4M3_MAX / global_max)).astype(jnp.float8_e4m3fn)
    full_fp8 = jax.lax.all_gather(q_share, axis, axis=2, tiled=True)
    scale_8 = jnp.broadcast_to(
        global_max * (1.0 / _FP8_E4M3_MAX), (*share.shape[:2], 8, share.shape[-1])
    )
    return full_fp8, scale_8
