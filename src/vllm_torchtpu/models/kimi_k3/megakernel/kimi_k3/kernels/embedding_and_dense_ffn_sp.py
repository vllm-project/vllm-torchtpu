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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.helpers import (
    expert_activation,
)

REDUCE_CHUNK_CANDIDATES = (256, 224, 192, 160, 128, 64, 32, 16)
COMPUTE_TILE_CANDIDATES = (16, 8, 4, 2, 1)
RECEIVE_RING_SLOTS = 4


def sp_token_index_for_ranks(rank, local, stripe: int, *, num_ranks: int):
    return (local // stripe) * num_ranks * stripe + rank * stripe + local % stripe


def half_ring_streams(width: int) -> tuple[tuple[int, int, int, int], ...]:
    if width < 384 or width % 128 != 0:
        raise ValueError("Half-ring streams require width >= 384 and divisible by 128")
    col_blocks = width // 128
    widths = [128 * (col_blocks // 3 + (i < col_blocks % 3)) for i in range(3)]
    cols = [0, widths[0], widths[0] + widths[1]]
    return tuple(
        (family, direction, cols[family], widths[family])
        for family in range(3)
        for direction in (1, -1)
    )


def select_half_reduce_chunk_rows(rows: int, *, chunk_cap: int = 256) -> int:
    if rows < 16 or rows % 16 != 0:
        raise ValueError(
            "Half-ring BF16 reduce-scatter requires rows >= 16 and % 16 == 0"
        )
    for candidate in REDUCE_CHUNK_CANDIDATES:
        if candidate <= chunk_cap and rows % candidate == 0:
            return candidate
    raise ValueError(f"No valid 16-aligned chunk <= {chunk_cap} divides rows={rows}")


def select_dense_ffn_compute_tiles(blk_tokens: int, *, max_tiles: int) -> int:
    for candidate in COMPUTE_TILE_CANDIDATES:
        if (
            candidate <= max_tiles
            and blk_tokens % candidate == 0
            and (blk_tokens // candidate) % 16 == 0
        ):
            return candidate
    raise ValueError(
        f"Cannot partition blk_tokens={blk_tokens} into <= {max_tiles} 16-row tiles"
    )


def embed_deconflicted_local_rows(
    token_ids: jax.Array,
    table: jax.Array,
    length_arr: jax.Array,
    *,
    rank: jax.Array,
    num_ranks: int,
    local_tokens: int,
    vocab_per_rank: int,
    reduce_dtype: jnp.dtype,
) -> jax.Array:
    seq_len = num_ranks * local_tokens
    stripe = min(64, local_tokens)
    window_positions = sp_token_index_for_ranks(
        jnp.arange(num_ranks, dtype=jnp.int32)[:, None],
        jnp.arange(local_tokens, dtype=jnp.int32)[None, :],
        stripe,
        num_ranks=num_ranks,
    ).reshape(seq_len)
    local_ids = token_ids - rank * vocab_per_rank
    matched_valid = (
        (local_ids >= 0)
        & (local_ids < vocab_per_rank)
        & (window_positions < length_arr[0])
    )
    deconflicted_ids = jnp.where(
        matched_valid,
        local_ids,
        jnp.arange(seq_len, dtype=jnp.int32) % vocab_per_rank,
    )
    return jnp.where(matched_valid[:, None], table[deconflicted_ids], 0).astype(
        reduce_dtype
    )


def _enter_ring_peers(tables_ref, rank, *, num_ranks: int):
    positions = [tables_ref[f, 0, rank] for f in range(3)]
    peers = [
        tables_ref[f, 1, (positions[f] + d) % num_ranks]
        for f in range(3)
        for d in (-1, 1)
    ]
    barrier = pltpu.get_barrier_semaphore()
    for peer in peers:
        pl.semaphore_signal(
            barrier, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(barrier, len(peers))

    @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
    def _(ready):
        for peer in peers:
            pl.semaphore_signal(
                ready, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(ready, len(peers))


def _sync_stream_neighbours(
    tables_ref,
    positions,
    streams_list,
    phase_sems,
    *,
    phase_slot,
    num_ranks: int,
):
    for stream, (family, direction, _, _) in enumerate(streams_list):
        peer = (tables_ref[family, 1, (positions[family] + direction) % num_ranks],)
        pl.semaphore_signal(
            phase_sems.at[phase_slot, stream ^ 1],
            1,
            device_id=peer,
            device_id_type=pl.DeviceIdType.MESH,
        )
    for stream in range(len(streams_list)):
        pl.semaphore_wait(phase_sems.at[phase_slot, stream], 1)


def rank_half_reduce_bf16_fast_pallas(
    partial_local: jax.Array,
    tables,
    *,
    num_ranks: int = 32,
    axis_name: str = "k3_sp",
    collective_id: int = 48,
    chunk_cap: int = 256,
    out_dtype: jnp.dtype = jnp.bfloat16,
    interpret: bool = False,
) -> jax.Array:
    if num_ranks < 4 or num_ranks % 2 != 0:
        raise ValueError(
            "rank_half_reduce_bf16_fast_pallas requires even num_ranks >= 4"
        )
    if partial_local.dtype != jnp.bfloat16:
        raise ValueError(
            "rank_half_reduce_bf16_fast_pallas requires BF16 partial_local"
        )
    total_rows, width = partial_local.shape
    if total_rows % num_ranks != 0:
        raise ValueError("partial_local leading axis must be divisible by num_ranks")
    rows = total_rows // num_ranks
    chunk = select_half_reduce_chunk_rows(rows, chunk_cap=chunk_cap)
    blocks = rows // chunk
    half_hops = {1: num_ranks // 2, -1: num_ranks // 2 - 1}
    half_steps = num_ranks // 2 + 1
    streams_list = half_ring_streams(width)
    n_streams = len(streams_list)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def _kernel(
        tables_ref,
        partial_ref,
        output_ref,
        read_sems,
        send_sems,
        recv_sems,
        credits,
        phase_sems,
        store_sem,
        local_bufs,
        payload,
        received,
        out_staging,
    ):
        step_all = pl.program_id(1)
        block, step = step_all // half_steps, step_all % half_steps
        rank = jax.lax.axis_index(axis_name)
        positions = [tables_ref[f, 0, rank] for f in range(3)]
        local_first = block * chunk

        pl.when(step_all == 0)(
            lambda: _enter_ring_peers(tables_ref, rank, num_ranks=num_ranks)
        )

        for stream, (family, direction, col, col_w) in enumerate(streams_list):
            cols = pl.ds(col, col_w)
            hops = half_hops[direction]
            position = positions[family]
            side = 0 if direction == 1 else 1
            target = (tables_ref[family, 1, (position + direction) % num_ranks],)
            source = (tables_ref[family, 1, (position - direction) % num_ranks],)

            def read_copy(
                idx,
                family=family,
                direction=direction,
                cols=cols,
                position=position,
                hops=hops,
                side=side,
                stream=stream,
            ):
                peer = tables_ref[
                    family, 1, (position + direction * (hops - idx)) % num_ranks
                ]
                owner = tables_ref[family, 2, peer]
                return pltpu.make_async_copy(
                    partial_ref.at[pl.ds(owner * rows + local_first, chunk), cols],
                    local_bufs.at[idx % 2, side, :, cols],
                    read_sems.at[idx % 2, stream],
                )

            def remote_copy(
                idx,
                cols=cols,
                side=side,
                target=target,
                stream=stream,
            ):
                slot4 = idx % RECEIVE_RING_SLOTS
                return pltpu.make_async_remote_copy(
                    payload.at[idx % 2, side, :, cols],
                    received.at[slot4, side, :, cols],
                    send_sems.at[idx % 2, stream],
                    recv_sems.at[slot4, stream],
                    device_id=target,
                    device_id_type=pl.DeviceIdType.MESH,
                )

            pl.when(step == 0)(lambda read_copy=read_copy: read_copy(0).start())

            @pl.when(step <= hops)
            def accumulate_step(
                read_copy=read_copy,
                remote_copy=remote_copy,
                cols=cols,
                side=side,
                hops=hops,
                stream=stream,
                source=source,
            ):
                read_copy(step).wait()
                pl.when(step < hops)(lambda: read_copy(step + 1).start())
                local_slice = local_bufs.at[step % 2, side, :, cols][...]

                @pl.when(step == 0)
                def seed_first():
                    payload.at[0, side, :, cols][...] = local_slice
                    remote_copy(0).start()

                @pl.when((step > 0) & (step < hops))
                def add_and_forward():
                    prev_idx = step - 1
                    prev_slot = prev_idx % RECEIVE_RING_SLOTS
                    remote_copy(prev_idx).wait_recv()
                    pl.when(step >= 2)(lambda: remote_copy(step - 2).wait_send())
                    payload.at[step % 2, side, :, cols][...] = (
                        local_slice + received.at[prev_slot, side, :, cols][...]
                    )
                    pl.when(prev_idx < hops - RECEIVE_RING_SLOTS)(
                        lambda: pl.semaphore_signal(
                            credits.at[prev_slot, stream],
                            1,
                            device_id=source,
                            device_id_type=pl.DeviceIdType.MESH,
                        )
                    )
                    pl.when(step >= RECEIVE_RING_SLOTS)(
                        lambda: pl.semaphore_wait(
                            credits.at[step % RECEIVE_RING_SLOTS, stream], 1
                        )
                    )
                    remote_copy(step).start()

                @pl.when(step == hops)
                def drain_last():
                    remote_copy(hops - 1).wait_recv()
                    remote_copy(hops - 1).wait_send()
                    if hops >= 2:
                        remote_copy(hops - 2).wait_send()

        @pl.when(step == half_steps - 1)
        def finish_block():
            cw_slot = (half_hops[1] - 1) % RECEIVE_RING_SLOTS
            ccw_slot = (half_hops[-1] - 1) % RECEIVE_RING_SLOTS
            own_slot = half_hops[1] % 2
            total_f32 = (
                local_bufs.at[own_slot, 0][...].astype(jnp.float32)
                + received.at[cw_slot, 0][...].astype(jnp.float32)
                + received.at[ccw_slot, 1][...].astype(jnp.float32)
            )
            out_staging[...] = total_f32.astype(out_staging.dtype)
            out_copy = pltpu.make_async_copy(
                out_staging, output_ref.at[pl.ds(local_first, chunk), :], store_sem
            )
            out_copy.start()
            out_copy.wait()
            _sync_stream_neighbours(
                tables_ref,
                positions,
                streams_list,
                phase_sems,
                phase_slot=block % 2,
                num_ranks=num_ranks,
            )

    interpret_mode = pltpu.InterpretParams() if interpret is True else interpret
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((rows, width), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1, blocks * half_steps),
            in_specs=(hbm,),
            out_specs=hbm,
            scratch_shapes=(
                pltpu.SemaphoreType.DMA((2, n_streams)),
                pltpu.SemaphoreType.DMA((2, n_streams)),
                pltpu.SemaphoreType.DMA((RECEIVE_RING_SLOTS, n_streams)),
                pltpu.SemaphoreType.REGULAR((RECEIVE_RING_SLOTS, n_streams)),
                pltpu.SemaphoreType.REGULAR((2, n_streams)),
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((2, 2, chunk, width), jnp.bfloat16),
                pltpu.VMEM((2, 2, chunk, width), jnp.bfloat16),
                pltpu.VMEM((RECEIVE_RING_SLOTS, 2, chunk, width), jnp.bfloat16),
                pltpu.VMEM((chunk, width), out_dtype),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=64 * 1024**2,
        ),
        interpret=interpret_mode,
        name="kimi_prefill_rank_half_rs_bf16_fast",
    )(jnp.asarray(tables, jnp.int32), partial_local)


def pipelined_dense_ffn_sp_pallas(
    hidden_local: jax.Array,
    prefix_local: jax.Array,
    gate_up_w: jax.Array,
    down_w: jax.Array,
    length_arr: jax.Array,
    tables,
    *,
    num_ranks: int = 32,
    axis_name: str = "k3_sp",
    collective_id: int = 47,
    interpret: bool = False,
) -> jax.Array:
    if num_ranks < 4 or num_ranks % 2 != 0:
        raise ValueError("pipelined_dense_ffn_sp_pallas requires even num_ranks >= 4")
    rows, width = hidden_local.shape
    mid_w = down_w.shape[0]
    if (
        prefix_local.shape != (rows, width)
        or gate_up_w.shape != (width, 2 * mid_w)
        or down_w.shape != (mid_w, width)
    ):
        raise ValueError("Dense FFN operand shapes do not match (rows, width, mid_w)")
    if rows < 16 or rows % 16 != 0 or width % 512 != 0 or mid_w % 128 != 0:
        raise ValueError(
            "Dense FFN requires 16-aligned rows, 512-aligned width, 128-aligned mid_w"
        )
    stripe = min(64, rows)
    blocks = rows // stripe
    blk_tokens = num_ranks * stripe
    half_hops = {1: num_ranks // 2, -1: num_ranks // 2 - 1}
    half_steps = num_ranks // 2 + 1
    total_phases = blocks + 2
    comp_tiles = select_dense_ffn_compute_tiles(blk_tokens, max_tiles=half_steps - 1)
    tile_m = blk_tokens // comp_tiles
    streams_list = half_ring_streams(width)
    n_streams = len(streams_list)
    quarter_w = width // 4
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    def _kernel(
        tables_ref,
        len_ref,
        x_local_ref,
        px_local_ref,
        gu_w_ref,
        dn_w_ref,
        out_ref,
        ag_ring_hbm,
        rs_part_hbm,
        gu_h0_vmem,
        gu_h1_vmem,
        dn_q_bufs,
        x_tile_vmem,
        part_q_bufs,
        rs_local_bufs,
        rs_payload,
        rs_received,
        px_out_vmem,
        ag_send_sems,
        ag_recv_sems,
        ag_own_sems,
        w_sems,
        tile_sems,
        rs_read_sems,
        rs_send_sems,
        rs_recv_sems,
        rs_credits,
        store_sems,
        phase_sems,
    ):
        step_all = pl.program_id(1)
        phase, step = step_all // half_steps, step_all % half_steps
        rank = jax.lax.axis_index(axis_name)
        positions = [tables_ref[f, 0, rank] for f in range(3)]

        def gu_half_copy(half_idx: int):
            dst = gu_h0_vmem if half_idx == 0 else gu_h1_vmem
            return pltpu.make_async_copy(
                gu_w_ref.at[:, pl.ds(half_idx * mid_w, mid_w)],
                dst,
                w_sems.at[half_idx],
            )

        @pl.when(step_all == 0)
        def init_ring_and_preload_gate_up():
            _enter_ring_peers(tables_ref, rank, num_ranks=num_ranks)
            gu_half_copy(0).start()
            gu_half_copy(1).start()

        @pl.when(step_all == half_steps)
        def wait_gate_up_weights():
            gu_half_copy(0).wait()
            gu_half_copy(1).wait()

        @pl.when(phase < blocks)
        def gather_block():
            ag_slot = phase % 2
            loc_sl = pl.ds(phase * stripe, stripe)

            def ag_owner_rows(family, pos):
                owner = tables_ref[family, 2, tables_ref[family, 1, pos % num_ranks]]
                return pl.ds(owner * stripe, stripe)

            def ag_own_copy():
                return pltpu.make_async_copy(
                    x_local_ref.at[loc_sl, :],
                    ag_ring_hbm.at[ag_slot, ag_owner_rows(0, positions[0]), :],
                    ag_own_sems.at[ag_slot],
                )

            pl.when(step == 0)(lambda: ag_own_copy().start())
            for stream, (family, direction, col, col_w) in enumerate(streams_list):
                cols = pl.ds(col, col_w)
                hops = half_hops[direction]
                pos = positions[family]
                target = (tables_ref[family, 1, (pos + direction) % num_ranks],)

                def ag_send(
                    idx,
                    src,
                    family=family,
                    direction=direction,
                    cols=cols,
                    pos=pos,
                    target=target,
                    stream=stream,
                ):
                    r_sl = ag_owner_rows(family, pos - direction * idx)
                    return pltpu.make_async_remote_copy(
                        src if src is not None else ag_ring_hbm.at[ag_slot, r_sl, cols],
                        ag_ring_hbm.at[ag_slot, r_sl, cols],
                        ag_send_sems.at[ag_slot, stream],
                        ag_recv_sems.at[ag_slot, stream, idx],
                        device_id=target,
                        device_id_type=pl.DeviceIdType.MESH,
                    )

                pl.when(step == 0)(
                    lambda ag_send=ag_send, cols=cols: ag_send(
                        0, x_local_ref.at[loc_sl, cols]
                    ).start()
                )

                @pl.when((step >= 1) & (step <= hops))
                def ag_arrived(ag_send=ag_send):
                    arrived = ag_send(step - 1, None)
                    arrived.wait_recv()
                    arrived.wait_send()

                @pl.when((step >= 1) & (step < hops))
                def ag_forward(ag_send=ag_send):
                    ag_send(step, None).start()

            pl.when(step == half_steps - 1)(lambda: ag_own_copy().wait())

        @pl.when(phase >= 2)
        def reduce_scatter_block():
            rs_blk = phase - 2
            rs_slot = rs_blk % 2
            loc_first = rs_blk * stripe

            def px_load_copy():
                return pltpu.make_async_copy(
                    px_local_ref.at[pl.ds(loc_first, stripe), :],
                    px_out_vmem,
                    store_sems.at[0],
                )

            pl.when(step == 0)(lambda: px_load_copy().start())
            for stream, (family, direction, col, col_w) in enumerate(streams_list):
                cols = pl.ds(col, col_w)
                hops = half_hops[direction]
                pos = positions[family]
                side = 0 if direction == 1 else 1
                target = (tables_ref[family, 1, (pos + direction) % num_ranks],)
                source = (tables_ref[family, 1, (pos - direction) % num_ranks],)

                def rs_read(
                    idx,
                    family=family,
                    direction=direction,
                    cols=cols,
                    pos=pos,
                    hops=hops,
                    side=side,
                    stream=stream,
                ):
                    peer = tables_ref[
                        family, 1, (pos + direction * (hops - idx)) % num_ranks
                    ]
                    owner = tables_ref[family, 2, peer]
                    return pltpu.make_async_copy(
                        rs_part_hbm.at[rs_slot, pl.ds(owner * stripe, stripe), cols],
                        rs_local_bufs.at[idx % 2, side, :, cols],
                        rs_read_sems.at[idx % 2, stream],
                    )

                def rs_remote(
                    idx,
                    cols=cols,
                    side=side,
                    target=target,
                    stream=stream,
                ):
                    slot4 = idx % RECEIVE_RING_SLOTS
                    return pltpu.make_async_remote_copy(
                        rs_payload.at[idx % 2, side, :, cols],
                        rs_received.at[slot4, side, :, cols],
                        rs_send_sems.at[idx % 2, stream],
                        rs_recv_sems.at[slot4, stream],
                        device_id=target,
                        device_id_type=pl.DeviceIdType.MESH,
                    )

                pl.when(step == 0)(lambda rs_read=rs_read: rs_read(0).start())

                @pl.when(step <= hops)
                def rs_step(
                    rs_read=rs_read,
                    rs_remote=rs_remote,
                    cols=cols,
                    side=side,
                    hops=hops,
                    stream=stream,
                    source=source,
                ):
                    rs_read(step).wait()
                    pl.when(step < hops)(lambda: rs_read(step + 1).start())
                    loc_slice = rs_local_bufs.at[step % 2, side, :, cols][...]

                    @pl.when(step == 0)
                    def seed_rs():
                        rs_payload.at[0, side, :, cols][...] = loc_slice
                        rs_remote(0).start()

                    @pl.when((step > 0) & (step < hops))
                    def add_and_fwd_rs():
                        prev_idx = step - 1
                        prev_slot = prev_idx % RECEIVE_RING_SLOTS
                        rs_remote(prev_idx).wait_recv()
                        pl.when(step >= 2)(lambda: rs_remote(step - 2).wait_send())
                        rs_payload.at[step % 2, side, :, cols][...] = (
                            loc_slice + rs_received.at[prev_slot, side, :, cols][...]
                        )
                        pl.when(prev_idx < hops - RECEIVE_RING_SLOTS)(
                            lambda: pl.semaphore_signal(
                                rs_credits.at[prev_slot, stream],
                                1,
                                device_id=source,
                                device_id_type=pl.DeviceIdType.MESH,
                            )
                        )
                        pl.when(step >= RECEIVE_RING_SLOTS)(
                            lambda: pl.semaphore_wait(
                                rs_credits.at[step % RECEIVE_RING_SLOTS, stream], 1
                            )
                        )
                        rs_remote(step).start()

                    @pl.when(step == hops)
                    def drain_rs():
                        rs_remote(hops - 1).wait_recv()
                        rs_remote(hops - 1).wait_send()
                        if hops >= 2:
                            rs_remote(hops - 2).wait_send()

            @pl.when(step == half_steps - 1)
            def finish_rs():
                cw_slot = (half_hops[1] - 1) % RECEIVE_RING_SLOTS
                ccw_slot = (half_hops[-1] - 1) % RECEIVE_RING_SLOTS
                own_slot = half_hops[1] % 2
                px_load_copy().wait()
                red_f32 = (
                    rs_local_bufs.at[own_slot, 0][...].astype(jnp.float32)
                    + rs_received.at[cw_slot, 0][...].astype(jnp.float32)
                    + rs_received.at[ccw_slot, 1][...].astype(jnp.float32)
                )
                owner_id = tables_ref[0, 2, rank]
                loc_iota = jax.lax.broadcasted_iota(jnp.int32, (stripe, 1), 0)
                win_tok = rs_blk * blk_tokens + owner_id * stripe + loc_iota
                valid = win_tok < len_ref[0]
                px_out_vmem[...] = (
                    px_out_vmem[...].astype(jnp.float32)
                    + jnp.where(valid, red_f32, 0.0)
                ).astype(out_ref.dtype)
                out_copy = pltpu.make_async_copy(
                    px_out_vmem,
                    out_ref.at[pl.ds(loc_first, stripe), :],
                    store_sems.at[1],
                )
                out_copy.start()
                out_copy.wait()

        @pl.when((phase >= 1) & (phase <= blocks))
        def compute_block():
            c_blk = phase - 1
            in_slot = c_blk % 2
            out_slot = c_blk % 2

            def x_tile_copy(tile_idx):
                buf_slot = tile_idx % 2
                return pltpu.make_async_copy(
                    ag_ring_hbm.at[in_slot, pl.ds(tile_idx * tile_m, tile_m), :],
                    x_tile_vmem.at[buf_slot],
                    tile_sems.at[buf_slot],
                )

            def dn_quarter_copy(quarter_idx):
                q_slot = quarter_idx % 2
                return pltpu.make_async_copy(
                    dn_w_ref.at[:, pl.ds(quarter_idx * quarter_w, quarter_w)],
                    dn_q_bufs.at[q_slot],
                    w_sems.at[2 + q_slot],
                )

            def part_quarter_store(tile_idx, quarter_idx):
                q_slot = quarter_idx % 2
                return pltpu.make_async_copy(
                    part_q_bufs.at[q_slot],
                    rs_part_hbm.at[
                        out_slot,
                        pl.ds(tile_idx * tile_m, tile_m),
                        pl.ds(quarter_idx * quarter_w, quarter_w),
                    ],
                    tile_sems.at[2 + q_slot],
                )

            pl.when(step == 0)(lambda: x_tile_copy(0).start())

            @pl.when(step < comp_tiles)
            def run_compute_tile():
                tile_idx = step
                x_tile_copy(tile_idx).wait()
                pl.when(tile_idx + 1 < comp_tiles)(
                    lambda: x_tile_copy(tile_idx + 1).start()
                )
                dn_quarter_copy(0).start()
                dn_quarter_copy(1).start()
                x_t = x_tile_vmem[tile_idx % 2][...]
                gate_f32 = (
                    jnp.dot(
                        x_t,
                        gu_h0_vmem[...],
                        preferred_element_type=jnp.float32,
                    )
                    .astype(jnp.bfloat16)
                    .astype(jnp.float32)
                )
                up_f32 = (
                    jnp.dot(
                        x_t,
                        gu_h1_vmem[...],
                        preferred_element_type=jnp.float32,
                    )
                    .astype(jnp.bfloat16)
                    .astype(jnp.float32)
                )
                mid_bf16 = expert_activation(gate_f32, up_f32).astype(jnp.bfloat16)

                @pl.loop(0, 4)
                def run_down_quarter(q_idx):
                    q_slot = q_idx % 2
                    dn_quarter_copy(q_idx).wait()
                    pl.when(q_idx >= 2)(
                        lambda: part_quarter_store(tile_idx, q_idx - 2).wait()
                    )
                    part_q_bufs[q_slot, ...] = jnp.dot(
                        mid_bf16,
                        dn_q_bufs[q_slot][...],
                        preferred_element_type=jnp.float32,
                    ).astype(jnp.bfloat16)
                    part_quarter_store(tile_idx, q_idx).start()
                    pl.when(q_idx + 2 < 4)(lambda: dn_quarter_copy(q_idx + 2).start())

                part_quarter_store(tile_idx, 2).wait()
                part_quarter_store(tile_idx, 3).wait()

        @pl.when(step == half_steps - 1)
        def finish_phase():
            _sync_stream_neighbours(
                tables_ref,
                positions,
                streams_list,
                phase_sems,
                phase_slot=phase % 2,
                num_ranks=num_ranks,
            )

    interpret_mode = pltpu.InterpretParams() if interpret is True else interpret
    return pl.pallas_call(
        _kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, blk_tokens, width), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, blk_tokens, width), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1, total_phases * half_steps),
            in_specs=(hbm, hbm, hbm, hbm),
            out_specs=(hbm, hbm, hbm),
            scratch_shapes=(
                pltpu.VMEM((width, mid_w), jnp.bfloat16),
                pltpu.VMEM((width, mid_w), jnp.bfloat16),
                pltpu.VMEM((2, mid_w, quarter_w), jnp.bfloat16),
                pltpu.VMEM((2, tile_m, width), jnp.bfloat16),
                pltpu.VMEM((2, tile_m, quarter_w), jnp.bfloat16),
                pltpu.VMEM((2, 2, stripe, width), jnp.bfloat16),
                pltpu.VMEM((2, 2, stripe, width), jnp.bfloat16),
                pltpu.VMEM((RECEIVE_RING_SLOTS, 2, stripe, width), jnp.bfloat16),
                pltpu.VMEM((stripe, width), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2, n_streams)),
                pltpu.SemaphoreType.DMA((2, n_streams, half_hops[1])),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((2, n_streams)),
                pltpu.SemaphoreType.DMA((2, n_streams)),
                pltpu.SemaphoreType.DMA((RECEIVE_RING_SLOTS, n_streams)),
                pltpu.SemaphoreType.REGULAR((RECEIVE_RING_SLOTS, n_streams)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.REGULAR((2, n_streams)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=64 * 1024**2,
        ),
        interpret=interpret_mode,
        name="kimi_prefill_pipelined_dense_ffn_sp",
    )(
        jnp.asarray(tables, jnp.int32),
        length_arr,
        hidden_local,
        prefix_local,
        gate_up_w,
        down_w,
    )[0]
