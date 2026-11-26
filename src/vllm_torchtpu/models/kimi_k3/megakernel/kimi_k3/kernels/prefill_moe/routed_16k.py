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

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .. import prefill_megakernel as k
from . import shared_down_16k, w4a8_16k

TOKENS_16K = 16384
OUTPUT_WIDTH_16K = 3584
SUBTOTAL_16K_COLS = 1536
PASS1_16K_COLS = 1280
LAST_PASS_16K_COLS = 768
TILE_M_16K = 384
BUCKET_BASE_16K = 64
GU_PANEL_WIDTH_16K = 512
GU_ACT_BANKS_16K = 3
DOWN_OVERLAP_UNROLL_16K = 16
DOWN_TAIL_UNROLL_16K = 24
DOWN_ALL_BUCKETS_16K: tuple[int, ...] = (64, 128, 192, 256, 320, 384)
DOWN_PASS0_BUCKETS_16K: tuple[int, ...] = (128, 192, 256, 320, 384)
EMIT_CHUNK_PASS0_16K = 1024
EMIT_CHUNK_PASS1_16K = 1024
HALF_RING_STREAMS = 6
HALF_RING_BUFFERS = 2
HALF_RING_META_STRIDE = 20
CW_MAX_HOPS = 16
CCW_MAX_HOPS = 15


@dataclasses.dataclass(frozen=True)
class DownPassSpec:
    col_start: int
    sub_banks: int
    bank_cols: int
    panel_width: int
    weight_buffers: int
    overlap_unroll: int
    bucket_rows: tuple[int, ...] = DOWN_ALL_BUCKETS_16K
    skip_last_panel_overlap: bool = False

    @property
    def width(self) -> int:
        return self.sub_banks * self.bank_cols


DOWN_PASS_SPECS_16K: tuple[DownPassSpec, ...] = (
    DownPassSpec(
        col_start=0,
        sub_banks=2,
        bank_cols=768,
        panel_width=768,
        weight_buffers=3,
        overlap_unroll=DOWN_OVERLAP_UNROLL_16K,
        bucket_rows=DOWN_PASS0_BUCKETS_16K,
        skip_last_panel_overlap=True,
    ),
    DownPassSpec(
        col_start=1536,
        sub_banks=2,
        bank_cols=640,
        panel_width=640,
        weight_buffers=3,
        overlap_unroll=0,
    ),
    DownPassSpec(
        col_start=2816,
        sub_banks=2,
        bank_cols=384,
        panel_width=768,
        weight_buffers=3,
        overlap_unroll=0,
    ),
)


def select_bucket_index_for_rows(
    last: jax.Array | int,
    bucket_rows: tuple[int, ...],
) -> jax.Array:
    if not bucket_rows or any(
        bucket_rows[i] <= 0 or (i > 0 and bucket_rows[i] <= bucket_rows[i - 1])
        for i in range(len(bucket_rows))
    ):
        raise ValueError(
            "bucket_rows must be non-empty and strictly increasing positive ints, "
            f"got {bucket_rows}"
        )
    idx = jnp.int32(len(bucket_rows) - 1)
    for bi in range(len(bucket_rows) - 2, -1, -1):
        idx = jnp.where(last <= bucket_rows[bi], jnp.int32(bi), idx)
    return idx


HALF_RING_1536_COL_SPEC: tuple[tuple[int, int], ...] = (
    (0, 512),
    (512, 512),
    (1024, 512),
)

HALF_RING_1280_COL_SPEC: tuple[tuple[int, int], ...] = (
    (0, 512),
    (512, 384),
    (896, 384),
)

HALF_RING_768_COL_SPEC: tuple[tuple[int, int], ...] = (
    (0, 256),
    (256, 256),
    (512, 256),
)


def half_ring_owner_positions(
    pos: int, *, direction: int, n_ranks: int = 32
) -> tuple[int, ...]:
    if n_ranks <= 2 or n_ranks % 2 != 0:
        raise ValueError("half-ring requires an even n_ranks >= 4")
    if direction not in (0, 1):
        raise ValueError("direction must be 0 (CW) or 1 (CCW)")
    half = n_ranks // 2
    if direction == 0:
        return tuple((pos + half - step) % n_ranks for step in range(half + 1))
    return tuple((pos - (half - 1 - step)) % n_ranks for step in range(half - 1))


def _build_half_ring_meta(tables, hmeta, localrows: int) -> list[tuple[jax.Array]]:
    rank = lax.axis_index("k3_sp")
    positions = [tables[f, 0, rank] for f in range(3)]
    peers: list[tuple[jax.Array]] = []
    for family in range(3):
        pos = positions[family]
        peer_neg = tables[family, 1, (pos - 1) & 31]
        peer_pos = tables[family, 1, (pos + 1) & 31]
        peers.append((peer_neg,))
        peers.append((peer_pos,))

        b_cw = (2 * family) * HALF_RING_META_STRIDE
        hmeta[b_cw] = peer_pos
        hmeta[b_cw + 1] = peer_neg

        b_ccw = (2 * family + 1) * HALF_RING_META_STRIDE
        hmeta[b_ccw] = peer_neg
        hmeta[b_ccw + 1] = peer_pos

        @pl.loop(0, CW_MAX_HOPS + 1)
        def fill_cw(step, *, family=family, pos=pos, b_cw=b_cw):
            owner = tables[family, 1, (pos + CW_MAX_HOPS - step) & 31]
            hmeta[b_cw + 2 + step] = tables[family, 2, owner] * localrows

        @pl.loop(0, CCW_MAX_HOPS)
        def fill_ccw(step, *, family=family, pos=pos, b_ccw=b_ccw):
            owner = tables[family, 1, (pos - (CCW_MAX_HOPS - step)) & 31]
            hmeta[b_ccw + 2 + step] = tables[family, 2, owner] * localrows

    return peers


def _emit_subtotals_to_hbm(
    sums,
    subtotal_hbm,
    stage_sems,
    *,
    tokens: int,
    cols: int,
    chunk_rows: int,
):
    n_chunks = tokens // chunk_rows

    @functools.partial(
        pl.run_scoped, stage=pltpu.VMEM((2, chunk_rows, cols), jnp.bfloat16)
    )
    def _copy(stage):
        @pl.loop(0, n_chunks)
        def blk(b):
            bank = b & 1

            @pl.when(b >= 2)
            def wait_prev():
                pltpu.make_async_copy(
                    stage.at[bank],
                    subtotal_hbm.at[
                        pl.ds((b - 2) * chunk_rows, chunk_rows), pl.ds(0, cols)
                    ],
                    stage_sems.at[bank],
                ).wait()

            stage.at[bank][...] = sums[pl.ds(b * chunk_rows, chunk_rows)].reshape(
                chunk_rows, cols
            )
            pltpu.make_async_copy(
                stage.at[bank],
                subtotal_hbm.at[pl.ds(b * chunk_rows, chunk_rows), pl.ds(0, cols)],
                stage_sems.at[bank],
            ).start()

        for b_tail in (n_chunks - 2, n_chunks - 1):
            bank = b_tail & 1
            pltpu.make_async_copy(
                stage.at[bank],
                subtotal_hbm.at[pl.ds(b_tail * chunk_rows, chunk_rows), pl.ds(0, cols)],
                stage_sems.at[bank],
            ).wait()


def _half_ring_hbm_init(
    x_hbm_ref,
    send_sems,
    recv_sems,
    load_sems,
    payload,
    rx,
    sub_buf,
    *,
    col_spec: tuple[tuple[int, int], ...],
    state,
    hmeta,
    localrows: int,
) -> None:
    streams = HALF_RING_STREAMS

    def remote(stream, direction, cols, target, bank):
        return pltpu.make_async_remote_copy(
            payload.at[direction, :, cols],
            rx.at[bank, direction, :, cols],
            send_sems.at[stream],
            recv_sems.at[bank, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def load_payload(stream, direction, src_rows, cols):
        return pltpu.make_async_copy(
            x_hbm_ref.at[src_rows, cols],
            payload.at[direction, :, cols],
            load_sems.at[0, stream],
        )

    def load_sub(stream, direction, src_rows, cols, lbank):
        return pltpu.make_async_copy(
            x_hbm_ref.at[src_rows, cols],
            sub_buf.at[lbank, direction, :, cols],
            load_sems.at[lbank, stream],
        )

    for stream in range(streams):
        fam, direction = stream // 2, stream % 2
        c0, cw = col_spec[fam]
        cols = pl.ds(c0, cw)
        base = stream * HALF_RING_META_STRIDE
        sr0 = pl.ds(pl.multiple_of(hmeta[base + 2], localrows), localrows)
        sr1 = pl.ds(pl.multiple_of(hmeta[base + 3], localrows), localrows)
        load_payload(stream, direction, sr0, cols).start()
        load_sub(stream, direction, sr1, cols, jnp.int32(1)).start()
    for stream in range(streams):
        fam, direction = stream // 2, stream % 2
        c0, cw = col_spec[fam]
        cols = pl.ds(c0, cw)
        base = stream * HALF_RING_META_STRIDE
        sr0 = pl.ds(pl.multiple_of(hmeta[base + 2], localrows), localrows)
        load_payload(stream, direction, sr0, cols).wait()
        remote(stream, direction, cols, (hmeta[base],), jnp.int32(0)).start()
        state[1 + stream] = 1
        state[1 + streams + stream] = 0
    state[0] = streams + 1


def _half_ring_hbm_sweep(
    x_hbm_ref,
    send_sems,
    recv_sems,
    credits,
    load_sems,
    payload,
    rx,
    sub_buf,
    *,
    col_spec: tuple[tuple[int, int], ...],
    state,
    hmeta,
    localrows: int,
) -> None:
    streams = HALF_RING_STREAMS
    buffers = HALF_RING_BUFFERS

    def remote(stream, direction, cols, target, bank):
        return pltpu.make_async_remote_copy(
            payload.at[direction, :, cols],
            rx.at[bank, direction, :, cols],
            send_sems.at[stream],
            recv_sems.at[bank, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        )

    def load_sub(stream, direction, src_rows, cols, lbank):
        return pltpu.make_async_copy(
            x_hbm_ref.at[src_rows, cols],
            sub_buf.at[lbank, direction, :, cols],
            load_sems.at[lbank, stream],
        )

    def sweep_stream(stream, direction, max_hop, cols, cw):
        base = stream * HALF_RING_META_STRIDE
        units = localrows * cw * jnp.dtype(payload.dtype).itemsize // 32
        step = state[1 + stream]

        @pl.when(state[1 + streams + stream] == 0)
        def receive():
            bank = (step - 1) & 1
            lbank = step & 1
            ready = (pl.semaphore_read(send_sems.at[stream]) >= units) & (
                pl.semaphore_read(recv_sems.at[bank, stream]) >= units
            )

            @pl.when(ready)
            def reduce():
                remote(stream, direction, cols, (hmeta[base],), bank).wait()

                @pl.when(step < max_hop)
                def mid_step():
                    sr = pl.ds(
                        pl.multiple_of(hmeta[base + 2 + step], localrows), localrows
                    )
                    load_sub(stream, direction, sr, cols, lbank).wait()
                    payload.at[direction, :, cols][...] = (
                        sub_buf.at[lbank, direction, :, cols][...].astype(jnp.float32)
                        + rx.at[bank, direction, :, cols][...].astype(jnp.float32)
                    ).astype(payload.dtype)
                    pf_cond = (step + 1 <= max_hop) & (
                        (direction == 0) | (step + 1 < max_hop)
                    )

                    @pl.when(pf_cond)
                    def prefetch_next():
                        sr_next = pl.ds(
                            pl.multiple_of(hmeta[base + 2 + step + 1], localrows),
                            localrows,
                        )
                        load_sub(
                            stream, direction, sr_next, cols, (step + 1) & 1
                        ).start()

                @pl.when(step == max_hop)
                def last_step():
                    @pl.when(direction == 0)
                    def cw_last():
                        own_sr = pl.ds(
                            pl.multiple_of(hmeta[base + 2 + max_hop], localrows),
                            localrows,
                        )
                        load_sub(stream, direction, own_sr, cols, lbank).wait()
                        payload.at[0, :, cols][...] = (
                            sub_buf.at[lbank, 0, :, cols][...].astype(jnp.float32)
                            + rx.at[bank, 0, :, cols][...].astype(jnp.float32)
                        ).astype(payload.dtype)

                    @pl.when(direction != 0)
                    def ccw_last():
                        payload.at[1, :, cols][...] = rx.at[bank, 1, :, cols][...]

                pl.semaphore_signal(
                    credits.at[stream],
                    1,
                    device_id=(hmeta[base + 1],),
                    device_id_type=pl.DeviceIdType.MESH,
                )
                state[1 + streams + stream] = jnp.where(step == max_hop, 2, 1)

        @pl.when(state[1 + streams + stream] == 1)
        def forward():
            avail = (step < buffers) | (pl.semaphore_read(credits.at[stream]) >= 1)

            @pl.when(avail)
            def send():
                @pl.when(step >= buffers)
                def acquire():
                    pl.semaphore_wait(credits.at[stream], 1)

                remote(stream, direction, cols, (hmeta[base],), step & 1).start()
                state[1 + stream] = step + 1
                state[1 + streams + stream] = 0

        @pl.when(state[1 + streams + stream] == 2)
        def finish():
            @pl.when(pl.semaphore_read(credits.at[stream]) >= buffers)
            def release():
                pl.semaphore_wait(credits.at[stream], buffers)
                state[1 + streams + stream] = 3
                state[0] = state[0] - 1

    uniform_cw = col_spec[0][1]
    is_uniform = all(
        c == i * uniform_cw and w == uniform_cw for i, (c, w) in enumerate(col_spec)
    )
    if is_uniform:

        @pl.loop(0, streams)
        def sweep_uniform(stream):
            fam = stream // 2
            direction = stream & 1
            c0 = pl.multiple_of(fam * uniform_cw, uniform_cw)
            max_hop = jnp.where(
                direction == 0, jnp.int32(CW_MAX_HOPS), jnp.int32(CCW_MAX_HOPS)
            )
            sweep_stream(stream, direction, max_hop, pl.ds(c0, uniform_cw), uniform_cw)
    else:
        for stream in range(streams):
            fam, direction = stream // 2, stream % 2
            max_hop = CW_MAX_HOPS if direction == 0 else CCW_MAX_HOPS
            c0, cw = col_spec[fam]
            sweep_stream(stream, direction, max_hop, pl.ds(c0, cw), cw)


def _half_ring_hbm_drain(
    x_hbm_ref,
    o_ref,
    send_sems,
    recv_sems,
    credits,
    store_sem,
    load_sems,
    payload,
    rx,
    sub_buf,
    *,
    col_spec: tuple[tuple[int, int], ...],
    state,
    hmeta,
) -> None:
    localrows = o_ref.shape[0]
    lax.while_loop(
        lambda i: state[0] != 1,
        lambda i: (
            _half_ring_hbm_sweep(
                x_hbm_ref,
                send_sems,
                recv_sems,
                credits,
                load_sems,
                payload,
                rx,
                sub_buf,
                col_spec=col_spec,
                state=state,
                hmeta=hmeta,
                localrows=localrows,
            ),
            i + 1,
        )[1],
        jnp.int32(0),
    )
    payload.at[0][...] = (
        payload.at[0][...].astype(jnp.float32) + payload.at[1][...].astype(jnp.float32)
    ).astype(payload.dtype)
    output = pltpu.make_async_copy(payload.at[0], o_ref, store_sem)
    output.start()
    output.wait()


def _read_banked_sums_cols(sums_ref, src_rows, *, col_start: int, col_width: int):
    bank_cols = sums_ref.shape[-1]
    if col_start + col_width <= bank_cols:
        return sums_ref.at[src_rows, 0, pl.ds(col_start, col_width)][...]
    if col_start >= bank_cols:
        return sums_ref.at[src_rows, 1, pl.ds(col_start - bank_cols, col_width)][...]
    left_width = bank_cols - col_start
    return jnp.concatenate(
        (
            sums_ref.at[src_rows, 0, pl.ds(col_start, left_width)][...],
            sums_ref.at[src_rows, 1, pl.ds(0, col_width - left_width)][...],
        ),
        axis=1,
    )


def _ready_reduce_sums_half(
    sums_ref,
    o_ref,
    send_sems,
    recv_sems,
    credits,
    store_sem,
    payload,
    rx,
    *,
    col_spec: tuple[tuple[int, int], ...],
    state,
    hmeta,
    overlap_fn: Callable[[Callable[[], None]], None] | None = None,
):
    localrows = o_ref.shape[0]
    streams = HALF_RING_STREAMS
    buffers = HALF_RING_BUFFERS

    def remote(stream, direction, dst_cols, target, bank):
        return pltpu.make_async_remote_copy(
            payload.at[direction, :, dst_cols],
            rx.at[bank, direction, :, dst_cols],
            send_sems.at[stream],
            recv_sems.at[bank, stream],
            device_id=target,
            device_id_type=pl.DeviceIdType.MESH,
        )

    for stream in range(streams):
        fam, direction = stream // 2, stream % 2
        c0, cw = col_spec[fam]
        dst_cols = pl.ds(c0, cw)
        base = stream * HALF_RING_META_STRIDE
        sr0 = pl.ds(pl.multiple_of(hmeta[base + 2], localrows), localrows)
        payload.at[direction, :, dst_cols][...] = _read_banked_sums_cols(
            sums_ref, sr0, col_start=c0, col_width=cw
        ).astype(payload.dtype)
        remote(stream, direction, dst_cols, (hmeta[base],), jnp.int32(0)).start()
        state[1 + stream] = 1
        state[1 + streams + stream] = 0
    state[0] = streams + 1

    def sweep():
        for stream in range(streams):
            fam, direction = stream // 2, stream % 2
            max_hop = CW_MAX_HOPS if direction == 0 else CCW_MAX_HOPS
            c0, cw = col_spec[fam]
            dst_cols = pl.ds(c0, cw)
            base = stream * HALF_RING_META_STRIDE
            units = localrows * cw * jnp.dtype(payload.dtype).itemsize // 32
            step = state[1 + stream]

            @pl.when(state[1 + streams + stream] == 0)
            def receive(
                *,
                step=step,
                stream=stream,
                direction=direction,
                max_hop=max_hop,
                c0=c0,
                cw=cw,
                dst_cols=dst_cols,
                base=base,
                units=units,
            ):
                bank = (step - 1) & 1
                ready = (pl.semaphore_read(send_sems.at[stream]) >= units) & (
                    pl.semaphore_read(recv_sems.at[bank, stream]) >= units
                )

                @pl.when(ready)
                def reduce():
                    remote(stream, direction, dst_cols, (hmeta[base],), bank).wait()

                    @pl.when(step < max_hop)
                    def mid_step():
                        sr = pl.ds(
                            pl.multiple_of(hmeta[base + 2 + step], localrows), localrows
                        )
                        payload.at[direction, :, dst_cols][...] = (
                            _read_banked_sums_cols(
                                sums_ref, sr, col_start=c0, col_width=cw
                            ).astype(jnp.float32)
                            + rx.at[bank, direction, :, dst_cols][...].astype(
                                jnp.float32
                            )
                        ).astype(payload.dtype)

                    @pl.when(step == max_hop)
                    def last_step():
                        if direction == 0:
                            own_sr = pl.ds(
                                pl.multiple_of(hmeta[base + 2 + max_hop], localrows),
                                localrows,
                            )
                            payload.at[0, :, dst_cols][...] = (
                                _read_banked_sums_cols(
                                    sums_ref, own_sr, col_start=c0, col_width=cw
                                ).astype(jnp.float32)
                                + rx.at[bank, 0, :, dst_cols][...].astype(jnp.float32)
                            ).astype(payload.dtype)
                        else:
                            payload.at[1, :, dst_cols][...] = rx.at[
                                bank, 1, :, dst_cols
                            ][...]

                    pl.semaphore_signal(
                        credits.at[stream],
                        1,
                        device_id=(hmeta[base + 1],),
                        device_id_type=pl.DeviceIdType.MESH,
                    )
                    state[1 + streams + stream] = jnp.where(step == max_hop, 2, 1)

            @pl.when(state[1 + streams + stream] == 1)
            def forward(
                *,
                step=step,
                stream=stream,
                direction=direction,
                dst_cols=dst_cols,
                base=base,
            ):
                avail = (step < buffers) | (pl.semaphore_read(credits.at[stream]) >= 1)

                @pl.when(avail)
                def send():
                    @pl.when(step >= buffers)
                    def acquire():
                        pl.semaphore_wait(credits.at[stream], 1)

                    remote(
                        stream, direction, dst_cols, (hmeta[base],), step & 1
                    ).start()
                    state[1 + stream] = step + 1
                    state[1 + streams + stream] = 0

            @pl.when(state[1 + streams + stream] == 2)
            def finish(*, stream=stream):
                @pl.when(pl.semaphore_read(credits.at[stream]) >= buffers)
                def release():
                    pl.semaphore_wait(credits.at[stream], buffers)
                    state[1 + streams + stream] = 3
                    state[0] = state[0] - 1

    if overlap_fn is not None:

        def progress() -> None:
            @pl.when(state[0] != 1)
            def _active() -> None:
                sweep()

        overlap_fn(progress)

    lax.while_loop(lambda i: state[0] != 1, lambda i: (sweep(), i + 1)[1], jnp.int32(0))
    payload.at[0][...] = (
        payload.at[0][...].astype(jnp.float32) + payload.at[1][...].astype(jnp.float32)
    ).astype(payload.dtype)
    output = pltpu.make_async_copy(payload.at[0], o_ref, store_sem)
    output.start()
    output.wait()


def _slice_panel_rhs(rhs, col_start: int, width: int):
    def slice_ref(ref, first: int, size: int):
        assert ref.scale is not None
        return dataclasses.replace(
            ref,
            weight=ref.weight.at[:, pl.ds(first, size)],
            scale=ref.scale.at[..., pl.ds(first, size)],
        )

    if isinstance(rhs, k.FusedWeightsRef):
        pair = 2 * pltpu.get_tpu_info().num_lanes
        assert col_start % pair == 0 and width % pair == 0
        return k.FusedWeightsRef(
            slice_ref(rhs.gate, col_start // 2, width // 2),
            slice_ref(rhs.up, col_start // 2, width // 2),
        )
    return slice_ref(rhs, col_start, width)


def _dequant_mxfp4_panel(panel_rhs, *, cfgs, panel_width: int) -> jax.Array:
    block = cfgs.rhs_cfgs.quant_block_size
    w_bf16 = (
        panel_rhs.get_weight()
        .astype(cfgs.lhs_cfgs.dtype)
        .reshape(cfgs.num_quant_blocks_per_tile_k, block, panel_width)
    )
    s_bf16 = panel_rhs.get_scale(replicate_size=block).astype(cfgs.lhs_cfgs.dtype)
    return (w_bf16 * s_bf16).reshape(cfgs.tiles.tile_k, panel_width)


def _gu_inner_16k(
    lhs,
    rhs,
    output,
    partial,
    metadata,
    *,
    cfgs,
    panel_width: int = GU_PANEL_WIDTH_16K,
):
    group = pl.program_id(1)
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    sub = cfgs.dims.size_lhs_sublane
    aligned = start - start % sub
    first, last = start - aligned, end - aligned
    width = 2 * rhs.gate.weight.shape[-1]

    def product(*, bucket_index: int):
        rows = (bucket_index + 1) * cfgs.tiles.bucket_base
        act = lhs[pl.ds(0, rows)].reshape(rows, cfgs.tiles.tile_k)
        for col0 in range(0, width, panel_width):
            pw = min(panel_width, width - col0)
            p_rhs = _slice_panel_rhs(rhs, col0, pw)
            prod = jnp.matmul(
                act,
                _dequant_mxfp4_panel(p_rhs, cfgs=cfgs, panel_width=pw),
                preferred_element_type=jnp.float32,
            ).astype(cfgs.acc_dtype)
            val = k.apply_act_fn(prod, cfgs.fuse_act)
            row = lax.broadcasted_iota(jnp.int32, val.shape, 0)
            val = jnp.where((first <= row) & (row < last), val, 0)
            out_w = pw // 2
            output[pl.ds(0, rows // sub), :, pl.ds(col0 // 2, out_w)] = val.astype(
                output.dtype
            ).reshape(rows // sub, sub, out_w)
        zeros = jnp.zeros_like(partial)
        output[0] += jnp.where(group == 0, zeros, partial[...])
        last_row = last // sub
        partial[...] = jnp.where(last % sub == 0, zeros, output[last_row])

    branches = [
        functools.partial(product, bucket_index=i)
        for i in range(cfgs.tiles.tile_m // cfgs.tiles.bucket_base)
    ]
    lax.switch(
        jnp.maximum(last - 1, 0) // cfgs.tiles.bucket_base,
        branches,
    )


def _down_inner_16k(
    lhs,
    rhs,
    output,
    metadata,
    *,
    cfgs,
    overlap_work: Callable[[int], None] | None,
    panel_width: int,
    bucket_rows: tuple[int, ...] = DOWN_ALL_BUCKETS_16K,
    skip_last_panel_overlap: bool = False,
):
    group = pl.program_id(1)
    start = metadata.gm_id_to_m_offset[group]
    end = metadata.gm_id_to_m_offset[group + 1]
    sub = cfgs.dims.size_lhs_sublane
    aligned = start - start % sub
    last = end - aligned
    width = rhs.weight.shape[-1]
    sub_banks, bank_cols = output.shape[-2], output.shape[-1]
    assert width == sub_banks * bank_cols and panel_width % bank_cols == 0
    n_panels = pl.cdiv(width, panel_width)

    def product(*, rows: int):
        act = lhs[pl.ds(0, rows // sub), :, pl.ds(0, cfgs.tiles.tile_k)].reshape(
            rows, cfgs.tiles.tile_k
        )
        for p_idx, col0 in enumerate(range(0, width, panel_width)):
            pw = min(panel_width, width - col0)
            p_rhs = _slice_panel_rhs(rhs, col0, pw)
            prod = jnp.matmul(
                act,
                _dequant_mxfp4_panel(p_rhs, cfgs=cfgs, panel_width=pw),
                preferred_element_type=jnp.float32,
            ).astype(cfgs.acc_dtype)
            bank_start = col0 // bank_cols
            bank_count = pw // bank_cols
            if bank_count == sub_banks:
                output[pl.ds(0, rows)] = prod.astype(output.dtype).reshape(
                    rows, sub_banks, bank_cols
                )
            else:
                output[pl.ds(0, rows), pl.ds(bank_start, bank_count), :] = prod.astype(
                    output.dtype
                ).reshape(rows, bank_count, bank_cols)
            if overlap_work is not None and (
                not skip_last_panel_overlap or p_idx < n_panels - 1
            ):
                overlap_work(p_idx)

    branches = [functools.partial(product, rows=rows) for rows in bucket_rows]
    lax.switch(
        select_bucket_index_for_rows(last, bucket_rows),
        branches,
    )


def routed_experts_16k(
    latent,
    gu_weights,
    gu_scales,
    rhs,
    scales,
    counts,
    destinations,
    probabilities,
    tables,
    layer,
    offset,
    middle,
    out,
    *,
    probabilities_ready: Callable[[], None],
    subtotal_hbm,
    compact_capacity: int,
    rmw_batch: int,
    rmw_dummy_slots: int,
    latent_scales_hbm=None,
    deferred_shared_hbm=None,
    deferred_shared_down_hbm=None,
    last_ring_overlap_fn: Callable[[Callable[[], None]], None] | None = None,
):
    tokens = TOKENS_16K
    gu_width = gu_weights.shape[-1]
    assert gu_width % 2 == 0
    mid_width = gu_width // 2
    assert rhs.shape[-2] * 8 == mid_width
    assert rhs.shape[-1] == OUTPUT_WIDTH_16K
    assert subtotal_hbm is not None and subtotal_hbm.shape == (
        tokens,
        SUBTOTAL_16K_COLS,
    )
    w4a8_latent = latent_scales_hbm is not None
    if w4a8_latent:
        assert latent.dtype == w4a8_16k.LATENT_E4M3_DTYPE
        assert latent.shape == (tokens, *w4a8_16k.LATENT_E4M3_TAIL_SHAPE)
        assert latent_scales_hbm.shape == w4a8_16k.LATENT_SCALES_SHAPE
        assert latent_scales_hbm.dtype == jnp.float32
    has_deferred_shared = (
        deferred_shared_hbm is not None or deferred_shared_down_hbm is not None
    )
    if has_deferred_shared:
        assert deferred_shared_hbm is not None and deferred_shared_down_hbm is not None
        assert deferred_shared_hbm.shape == (
            tokens // 32,
            shared_down_16k.SHARED_HIDDEN_16K,
        )
        assert deferred_shared_down_hbm.shape == (
            shared_down_16k.SHARED_INTER_16K,
            shared_down_16k.SHARED_HIDDEN_16K,
        )

    tm = TILE_M_16K
    bucket_base = BUCKET_BASE_16K
    localrows = tokens // 32
    rows = compact_capacity
    ring_dtype = jnp.bfloat16
    sums_dtype = jnp.bfloat16
    x_sds = jax.ShapeDtypeStruct((tokens * 16, mid_width), middle.dtype)
    assert rows <= x_sds.shape[0]
    limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.98)

    def make_down_cfg(tn_val: int):
        cfg = k.make_gmm_configs(
            jax.ShapeDtypeStruct((rows, mid_width), x_sds.dtype),
            jax.ShapeDtypeStruct((rhs.shape[1], rhs.shape[2], tn_val), rhs.dtype),
            jax.ShapeDtypeStruct(
                (scales.shape[1], scales.shape[2], tn_val), scales.dtype
            ),
            None,
            counts,
            offset,
            tile_info=k.calculate_tiling,
            vmem_limit_bytes=limit,
            out_dtype=None,
            acc_dtype=None,
            maybe_quantize_lhs=False,
            zero_initialize=False,
            fuse_act=None,
            rhs_quant_dtype=jnp.float4_e2m1fn,
        )
        return dataclasses.replace(
            cfg,
            tiles=dataclasses.replace(
                cfg.tiles, tile_m=tm, tile_n=tn_val, bucket_base=bucket_base
            ),
        )

    pass0_spec, pass1_spec, pass2_spec = DOWN_PASS_SPECS_16K
    down_cfgs = tuple(make_down_cfg(spec.width) for spec in DOWN_PASS_SPECS_16K)
    sub = down_cfgs[0].dims.size_lhs_sublane
    max_gm = (
        down_cfgs[0].dims.size_group
        + pl.cdiv(rows + down_cfgs[0].dims.size_group * (sub - 1), tm)
        - 1
    )

    gu_cfg = k.make_gmm_configs(
        jax.ShapeDtypeStruct(
            (rows, OUTPUT_WIDTH_16K),
            middle.dtype if w4a8_latent else latent.dtype,
        ),
        jax.ShapeDtypeStruct(gu_weights.shape[1:], gu_weights.dtype),
        jax.ShapeDtypeStruct(gu_scales.shape[1:], gu_scales.dtype),
        None,
        counts,
        offset,
        tile_info=k.calculate_tiling,
        vmem_limit_bytes=limit,
        out_dtype=middle.dtype if w4a8_latent else None,
        acc_dtype=jnp.float32 if w4a8_latent else None,
        maybe_quantize_lhs=False,
        zero_initialize=False,
        fuse_act="situ:4:25",
        rhs_quant_dtype=jnp.float4_e2m1fn,
    )
    gu_cfg = dataclasses.replace(
        gu_cfg,
        tiles=dataclasses.replace(
            gu_cfg.tiles,
            tile_m=tm,
            tile_n=mid_width,
            tile_k=OUTPUT_WIDTH_16K,
            bucket_base=bucket_base,
        ),
    )

    down_tail_unroll = DOWN_TAIL_UNROLL_16K
    n_act_banks = GU_ACT_BANKS_16K

    def kernel(
        counts_ref,
        offset_ref,
        layer_ref,
        tables_ref,
        destinations_ref,
        probabilities_ref,
        latent_ref,
        gu,
        w,
        middle_ref,
        out_ref,
        metadata,
        sends,
        receives,
        credits,
        stores,
        load_sems,
        stage_sems,
        state,
        hmeta,
        phase_sems,
        phase_smem,
        down_ready,
    ):
        w = dataclasses.replace(
            w,
            weight=w.weight.at[layer_ref[0]].bitcast(jnp.float4_e2m1fn),
            scale=w.scale.at[layer_ref[0]],
        )
        gu = dataclasses.replace(
            gu,
            weight=gu.weight.at[layer_ref[0]].bitcast(jnp.float4_e2m1fn),
            scale=gu.scale.at[layer_ref[0]],
        )
        gu = k.FusedWeightsRef(
            gate=gu, up=jax.tree.map(lambda r: r.at[..., mid_width:], gu)
        )

        peers = _build_half_ring_meta(tables_ref, hmeta, localrows)
        active_ng = k.fill_metadata(counts_ref, offset_ref, metadata, cfgs=down_cfgs[0])

        @pl.when(active_ng == 0)
        def empty_metadata():
            metadata.gm_id_to_group_id[0] = 0
            metadata.gm_id_to_m_offset[0] = 0
            metadata.gm_id_to_m_offset[1] = 0

        ng = jnp.maximum(active_ng, 1)
        (_, gu_ws), gu_ys = k.generate_block_specs(metadata, gu_cfg)
        gu_ws = k.FusedWeightsRef(gate=gu_ws, up=gu_ws)
        state[0] = 0
        phase_smem[0] = 0

        @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
        def initialized(ready):
            for peer in peers:
                pl.semaphore_signal(
                    ready, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(ready, len(peers))

        def signal_ring_peers():
            sem = phase_sems.at[phase_smem[0] & 1]
            for peer in peers:
                pl.semaphore_signal(
                    sem, 1, device_id=peer, device_id_type=pl.DeviceIdType.MESH
                )

        def wait_ring_peers():
            slot = phase_smem[0]
            pl.semaphore_wait(phase_sems.at[slot & 1], len(peers))
            phase_smem[0] = slot + 1

        def sync_ring_peers():
            signal_ring_peers()
            wait_ring_peers()

        def buf_n(spec, count: int):
            return dataclasses.replace(
                spec, pipeline_mode=pl.Buffered(buffer_count=count)
            )

        if w4a8_latent:
            w4a8_16k.run_gu_w4a8_16k(
                gu,
                gu_ws,
                gu_ys,
                metadata,
                latent_ref,
                latent_scales_hbm,
                destinations_ref,
                middle_ref,
                gu_cfg=gu_cfg,
                active_ng=active_ng,
                ng=ng,
                sub=sub,
                tm=tm,
                mid_width=mid_width,
                tokens=tokens,
                n_act_banks=n_act_banks,
                panel_width=GU_PANEL_WIDTH_16K,
                buf_n=buf_n,
            )
        else:

            @functools.partial(
                pl.run_scoped,
                gu_partial=pltpu.VMEM((sub, mid_width), gu_cfg.out_dtype),
                acts=pltpu.VMEM((n_act_banks, tm, 2, 1792), latent_ref.dtype),
                act_sems=pltpu.SemaphoreType.DMA((n_act_banks,)),
            )
            def run_gu(gu_partial, acts, act_sems):
                gu_ws3 = jax.tree.map(
                    lambda s: buf_n(s, 3),
                    gu_ws,
                    is_leaf=lambda v: isinstance(v, pl.BlockSpec),
                )

                def active_bounds(g):
                    raw_start = metadata.gm_id_to_m_offset[g]
                    raw_end = metadata.gm_id_to_m_offset[g + 1]
                    begin = raw_start // sub * sub
                    return (raw_start, raw_start - begin, raw_end - raw_start)

                def start_gather(g, bank):
                    raw_start, first, n_active = active_bounds(g)
                    unroll_g = 16
                    full = (n_active // unroll_g) * unroll_g

                    @pl.loop(0, n_active // unroll_g)
                    def row_vec(blk):
                        base = blk * unroll_g
                        sources = [
                            destinations_ref[raw_start + base + u]
                            for u in range(unroll_g)
                        ]
                        for u in range(unroll_g):
                            pltpu.make_async_copy(
                                latent_ref.at[pl.ds(sources[u], 1)],
                                acts.at[bank, pl.ds(first + base + u, 1)],
                                act_sems.at[bank],
                            ).start()

                    @pl.loop(full, n_active)
                    def row_rem(i):
                        source = destinations_ref[raw_start + i]
                        pltpu.make_async_copy(
                            latent_ref.at[pl.ds(source, 1)],
                            acts.at[bank, pl.ds(first + i, 1)],
                            act_sems.at[bank],
                        ).start()

                def gu_step(gw, gy, metadata, gu_partial, acts, act_sems):
                    g = pl.program_id(1)
                    bank = g % n_act_banks

                    @pl.when(active_ng > 0)
                    def gate_up():
                        @pl.when(g == 0)
                        def initial():
                            acts[...] = jnp.zeros(acts.shape, acts.dtype)
                            start_gather(0, 0)
                            for pre in range(1, n_act_banks):

                                @pl.when(ng > pre)
                                def seed_bank(*, p=pre):
                                    start_gather(p, p)

                        _, first, n_active = active_bounds(g)

                        @pl.when(n_active > 0)
                        def wait_active():
                            active = acts.at[bank, pl.ds(first, n_active)]
                            pltpu.make_async_copy(
                                active, active, act_sems.at[bank]
                            ).wait()

                        _gu_inner_16k(
                            acts.at[bank],
                            gw,
                            gy,
                            gu_partial,
                            metadata,
                            cfgs=gu_cfg,
                        )

                        @pl.when(g + n_act_banks < ng)
                        def prefetch_ahead():
                            start_gather(g + n_act_banks, bank)

                pltpu.emit_pipeline(
                    gu_step,
                    grid=(1, ng, 1),
                    in_specs=(gu_ws3,),
                    out_specs=gu_ys,
                )(
                    gu,
                    middle_ref.reshape(-1, sub, mid_width),
                    scratches=(metadata, gu_partial, acts, act_sems),
                )

        def route(i):
            return destinations_ref[i], probabilities_ref[i]

        def batched_unclamped(sums, source, begin, base_idx, width_b):
            idxs = [base_idx + r for r in range(width_b)]
            routes = [route(i_r) for i_r in idxs]
            vals = [source.at[pl.ds(i_r - begin, 1)][...] for i_r in idxs]
            olds = [sums.at[pl.ds(d_r, 1)][...] for d_r, _ in routes]
            accs = [
                olds[r] + vals[r] * routes[r][1].astype(jnp.bfloat16)
                for r in range(width_b)
            ]
            for r in range(width_b):
                sums.at[pl.ds(routes[r][0], 1)][...] = accs[r]

        def batched_clamped(
            sums, source, begin, base_idx, end, width_b, *, guard, dummy_base=0
        ):
            idxs = [base_idx + r for r in range(width_b)]
            actives = [guard & (i_r < end) for i_r in idxs]
            clamped = [jnp.where(actives[r], idxs[r], begin) for r in range(width_b)]
            routes = [route(c_r) for c_r in clamped]
            dests = [
                jnp.where(
                    actives[r],
                    routes[r][0],
                    tokens + ((dummy_base + r) % rmw_dummy_slots),
                )
                for r in range(width_b)
            ]
            weights = [jnp.where(actives[r], routes[r][1], 0.0) for r in range(width_b)]
            vals = [source.at[pl.ds(c_r - begin, 1)][...] for c_r in clamped]
            olds = [sums.at[pl.ds(d_r, 1)][...] for d_r in dests]
            accs = [
                olds[r] + vals[r] * weights[r].astype(jnp.bfloat16)
                for r in range(width_b)
            ]
            for r in range(width_b):
                sums.at[pl.ds(dests[r], 1)][...] = accs[r]

        def consume(sums, tile_g, source, first=0, *, loop_unroll=down_tail_unroll):
            start = metadata.gm_id_to_m_offset[tile_g]
            end = metadata.gm_id_to_m_offset[tile_g + 1]
            begin = start // sub * sub
            start = jnp.minimum(start + first, end)
            step_b = min(rmw_batch, loop_unroll)
            full = (end - start) // loop_unroll

            @pl.loop(0, full)
            def chunk(b):
                base = start + b * loop_unroll
                for u in range(0, loop_unroll, step_b):
                    batched_unclamped(
                        sums, source, begin, base + u, min(step_b, loop_unroll - u)
                    )

            tail_start = start + full * loop_unroll
            tail_steps = pl.cdiv(end - tail_start, step_b)

            @pl.loop(0, tail_steps)
            def tail(tb):
                batched_clamped(
                    sums,
                    source,
                    begin,
                    tail_start + tb * step_b,
                    end,
                    step_b,
                    guard=jnp.bool_(True),
                )

        def down_weights_slice(spec: DownPassSpec):
            return k.WeightsRef(
                weight=w.weight.at[..., pl.ds(spec.col_start, spec.width)],
                scale=w.scale.at[..., pl.ds(spec.col_start, spec.width)],
                bias=None,
            )

        def run_down_pass(
            pass_idx: int,
            packed,
            sums,
            *,
            ring_out_col: int | None = None,
            ring_cols: int = 0,
            ring_col_spec: tuple[tuple[int, int], ...] | None = None,
            payload_buf=None,
            rx_buf=None,
            sub_buf=None,
        ):
            spec = DOWN_PASS_SPECS_16K[pass_idx]
            cfg_use = down_cfgs[pass_idx]
            (xs_pass, ws_pass), _ = k.generate_block_specs(metadata, cfg_use)
            ws_buf = jax.tree.map(
                lambda s: buf_n(s, spec.weight_buffers),
                ws_pass,
                is_leaf=lambda v: isinstance(v, pl.BlockSpec),
            )
            panel_count = pl.cdiv(spec.width, spec.panel_width)
            ring_ctx = None
            if ring_out_col is not None:
                assert (
                    ring_col_spec is not None
                    and payload_buf is not None
                    and rx_buf is not None
                    and sub_buf is not None
                )
                ring_ctx = (
                    ring_out_col,
                    ring_col_spec,
                    payload_buf.at[:, :, pl.ds(0, ring_cols)],
                    rx_buf.at[:, :, :, pl.ds(0, ring_cols)],
                    sub_buf.at[:, :, :, pl.ds(0, ring_cols)],
                )

            if pass_idx == 0:
                probabilities_ready()
            sums[...] = jnp.zeros(sums.shape, sums_dtype)
            if spec.overlap_unroll > 0:
                packed.at[1][...] = jnp.zeros(packed.at[1].shape, packed.dtype)
            if ring_ctx is not None:
                _, r_col_spec, r_payload, r_rx, r_sub = ring_ctx
                _half_ring_hbm_init(
                    subtotal_hbm.at[:, pl.ds(0, ring_cols)],
                    sends,
                    receives,
                    load_sems,
                    r_payload,
                    r_rx,
                    r_sub,
                    col_spec=r_col_spec,
                    state=state,
                    hmeta=hmeta,
                    localrows=localrows,
                )

            def down_step(x_tile, w_tile, metadata):
                g = pl.program_id(1)

                if ring_ctx is not None:
                    _, r_col_spec, r_payload, r_rx, r_sub = ring_ctx

                    @pl.when(state[0] != 1)
                    def poll_ring():
                        _half_ring_hbm_sweep(
                            subtotal_hbm.at[:, pl.ds(0, ring_cols)],
                            sends,
                            receives,
                            credits,
                            load_sems,
                            r_payload,
                            r_rx,
                            r_sub,
                            col_spec=r_col_spec,
                            state=state,
                            hmeta=hmeta,
                            localrows=localrows,
                        )

                if spec.overlap_unroll > 0:
                    current = packed.at[g % 2]
                    previous = packed.at[(g + 1) % 2]
                    previous_g = jnp.maximum(g - 1, 0)
                    overlap_panels = (
                        max(panel_count - 1, 0)
                        if spec.skip_last_panel_overlap
                        else panel_count
                    )

                    def overlap_work(panel):
                        start = metadata.gm_id_to_m_offset[previous_g]
                        end = metadata.gm_id_to_m_offset[previous_g + 1]
                        begin = start // sub * sub
                        base_off = panel * spec.overlap_unroll
                        step_b = min(rmw_batch, spec.overlap_unroll)
                        for row in range(0, spec.overlap_unroll, step_b):
                            batched_clamped(
                                sums,
                                previous,
                                begin,
                                start + base_off + row,
                                end,
                                min(step_b, spec.overlap_unroll - row),
                                guard=(g > 0),
                                dummy_base=base_off + row,
                            )

                    @pl.when(active_ng > 0)
                    def product():
                        _down_inner_16k(
                            x_tile,
                            w_tile,
                            current,
                            metadata,
                            cfgs=cfg_use,
                            overlap_work=overlap_work,
                            panel_width=spec.panel_width,
                            bucket_rows=spec.bucket_rows,
                            skip_last_panel_overlap=spec.skip_last_panel_overlap,
                        )

                    @pl.when(g > 0)
                    def previous_tail():
                        consume(
                            sums,
                            previous_g,
                            previous,
                            overlap_panels * spec.overlap_unroll,
                        )
                else:
                    current = packed.at[0]

                    @pl.when(active_ng > 0)
                    def product():
                        _down_inner_16k(
                            x_tile,
                            w_tile,
                            current,
                            metadata,
                            cfgs=cfg_use,
                            overlap_work=None,
                            panel_width=spec.panel_width,
                            bucket_rows=spec.bucket_rows,
                        )

                    consume(sums, g, current)

            pltpu.emit_pipeline(
                down_step,
                grid=(1, ng, 1),
                in_specs=(xs_pass, ws_buf),
                out_specs=(),
            )(
                middle_ref.reshape(-1, sub, mid_width),
                down_weights_slice(spec),
                scratches=(metadata,),
            )

            if spec.overlap_unroll > 0:
                last_g = ng - 1
                consume(sums, last_g, packed.at[last_g % 2])
            if ring_ctx is not None:
                r_out_col, r_col_spec, r_payload, r_rx, r_sub = ring_ctx
                _half_ring_hbm_drain(
                    subtotal_hbm.at[:, pl.ds(0, ring_cols)],
                    out_ref.at[:, pl.ds(r_out_col, ring_cols)],
                    sends,
                    receives,
                    credits,
                    stores,
                    load_sems,
                    r_payload,
                    r_rx,
                    r_sub,
                    col_spec=r_col_spec,
                    state=state,
                    hmeta=hmeta,
                )

        if has_deferred_shared:
            rank = lax.axis_index("k3_sp")
            shared_down_16k.run_deferred_shared_down_from_hbm(
                deferred_shared_hbm,
                deferred_shared_down_hbm,
                down_ready,
                rows=localrows,
                parity=rank % 2,
                peer=rank ^ 1,
            )

        @functools.partial(
            pl.run_scoped,
            sums0=pltpu.VMEM(
                (tokens + rmw_dummy_slots, pass0_spec.sub_banks, pass0_spec.bank_cols),
                sums_dtype,
            ),
        )
        def run_pass0(sums0):
            @functools.partial(
                pl.run_scoped,
                packed0=pltpu.VMEM(
                    (2, tm, pass0_spec.sub_banks, pass0_spec.bank_cols), x_sds.dtype
                ),
            )
            def compute_pass0(packed0):
                run_down_pass(0, packed0, sums0)

            _emit_subtotals_to_hbm(
                sums0,
                subtotal_hbm,
                stage_sems,
                tokens=tokens,
                cols=pass0_spec.width,
                chunk_rows=EMIT_CHUNK_PASS0_16K,
            )
            state[0] = 0

        @functools.partial(
            pl.run_scoped,
            payload_buf=pltpu.VMEM((2, localrows, SUBTOTAL_16K_COLS), ring_dtype),
            rx_buf=pltpu.VMEM((2, 2, localrows, SUBTOTAL_16K_COLS), ring_dtype),
        )
        def run_remaining_passes(payload_buf, rx_buf):
            sync_ring_peers()

            @functools.partial(
                pl.run_scoped,
                sums1=pltpu.VMEM(
                    (
                        tokens + rmw_dummy_slots,
                        pass1_spec.sub_banks,
                        pass1_spec.bank_cols,
                    ),
                    sums_dtype,
                ),
            )
            def run_pass1(sums1):
                @functools.partial(
                    pl.run_scoped,
                    packed1=pltpu.VMEM(
                        (1, tm, pass1_spec.sub_banks, pass1_spec.bank_cols),
                        x_sds.dtype,
                    ),
                    sub_buf=pltpu.VMEM(
                        (2, 2, localrows, SUBTOTAL_16K_COLS), ring_dtype
                    ),
                )
                def compute_pass1(packed1, sub_buf):
                    run_down_pass(
                        1,
                        packed1,
                        sums1,
                        ring_out_col=pass0_spec.col_start,
                        ring_cols=pass0_spec.width,
                        ring_col_spec=HALF_RING_1536_COL_SPEC,
                        payload_buf=payload_buf,
                        rx_buf=rx_buf,
                        sub_buf=sub_buf,
                    )

                signal_ring_peers()
                _emit_subtotals_to_hbm(
                    sums1,
                    subtotal_hbm,
                    stage_sems,
                    tokens=tokens,
                    cols=pass1_spec.width,
                    chunk_rows=EMIT_CHUNK_PASS1_16K,
                )
                wait_ring_peers()
                state[0] = 0

            @functools.partial(
                pl.run_scoped,
                sums2=pltpu.VMEM(
                    (
                        tokens + rmw_dummy_slots,
                        pass2_spec.sub_banks,
                        pass2_spec.bank_cols,
                    ),
                    sums_dtype,
                ),
            )
            def run_pass2_and_last_ring(sums2):
                @functools.partial(
                    pl.run_scoped,
                    packed2=pltpu.VMEM(
                        (1, tm, pass2_spec.sub_banks, pass2_spec.bank_cols),
                        x_sds.dtype,
                    ),
                    sub_buf=pltpu.VMEM(
                        (2, 2, localrows, SUBTOTAL_16K_COLS), ring_dtype
                    ),
                )
                def compute_pass2(packed2, sub_buf):
                    run_down_pass(
                        2,
                        packed2,
                        sums2,
                        ring_out_col=pass1_spec.col_start,
                        ring_cols=pass1_spec.width,
                        ring_col_spec=HALF_RING_1280_COL_SPEC,
                        payload_buf=payload_buf,
                        rx_buf=rx_buf,
                        sub_buf=sub_buf,
                    )

                state[0] = 0
                sync_ring_peers()
                _ready_reduce_sums_half(
                    sums2,
                    out_ref.at[:, pl.ds(pass2_spec.col_start, pass2_spec.width)],
                    sends,
                    receives,
                    credits,
                    stores,
                    payload_buf.at[:, :, pl.ds(0, pass2_spec.width)],
                    rx_buf.at[:, :, :, pl.ds(0, pass2_spec.width)],
                    col_spec=HALF_RING_768_COL_SPEC,
                    state=state,
                    hmeta=hmeta,
                    overlap_fn=last_ring_overlap_fn,
                )

    def run(**scratch):
        kernel(
            counts,
            offset,
            layer,
            tables,
            destinations,
            probabilities,
            latent,
            k.WeightsRef(weight=gu_weights, scale=gu_scales, bias=None),
            k.WeightsRef(weight=rhs, scale=scales, bias=None),
            middle,
            out,
            **scratch,
        )

    pl.run_scoped(
        run,
        metadata=k.MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_gm + 1,), jnp.int32),
        ),
        sends=pltpu.SemaphoreType.DMA((HALF_RING_STREAMS,)),
        receives=pltpu.SemaphoreType.DMA((HALF_RING_BUFFERS, HALF_RING_STREAMS)),
        credits=pltpu.SemaphoreType.REGULAR((HALF_RING_STREAMS,)),
        stores=pltpu.SemaphoreType.DMA,
        load_sems=pltpu.SemaphoreType.DMA((2, HALF_RING_STREAMS)),
        stage_sems=pltpu.SemaphoreType.DMA((2,)),
        state=pltpu.SMEM((1 + 2 * HALF_RING_STREAMS,), jnp.int32),
        hmeta=pltpu.SMEM((HALF_RING_STREAMS * HALF_RING_META_STRIDE,), jnp.int32),
        phase_sems=pltpu.SemaphoreType.REGULAR((2,)),
        phase_smem=pltpu.SMEM((1,), jnp.int32),
        down_ready=pltpu.SemaphoreType.REGULAR if has_deferred_shared else None,
    )
