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
"""Row-streaming hierarchical reduce-scatter over 8 devices (tpu7x 2x2x1).

Grid step r issues phase 1 of block r+1, finishes phase 1 and issues step 0 of
block r, finishes step 0 and issues step 1 of block r-1, and finishes step 1
and stores block r-2. Each block's two row halves take the hypercube dims in
opposite orders, so both ICI links carry traffic on every step.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_NUM_DEVICES = 8
_SUB_TILE = 32
_SLOTS = 3
_ROW_BLOCK = 128  # 256 needs ~66 MiB of VMEM
_VMEM_LIMIT_BYTES = 56 << 20


def supports_stream(rows: int) -> bool:
    chunk, rem = divmod(rows, _NUM_DEVICES)
    return rem == 0 and chunk >= _ROW_BLOCK and chunk % _ROW_BLOCK == 0


def _kernel(
    x_ref,
    o_ref,
    sums,
    p1_recv_buf,
    s0_recv_buf,
    s1_recv_buf,
    load_sem,
    p1_send_sem,
    p1_recv_sem,
    s0_send_sem,
    s0_recv_sem,
    s1_send_sem,
    s1_recv_sem,
    store_sem,
    p1_slot_free,
    s0_slot_free,
    s1_slot_free,
    *,
    axis_name,
    chunk_rows,
    block_rows,
    num_blocks,
    order,
):
    half_rows = block_rows // 2
    r = pl.program_id(0)
    me = jax.lax.axis_index(axis_name)
    bit = me % 2
    chip = me // 2
    partner_bit = 1 - bit
    partner = me + 1 - 2 * bit
    neighbours = (chip ^ 1) * 2 + bit, (chip ^ 2) * 2 + bit
    dim0_bit = chip & 1
    dim1_bit = chip // 2
    logical = pl.DeviceIdType.LOGICAL

    # Row half h reduces along dim h, then dim 1 - h. After phase 1 a device
    # holds one row chunk per chip: step 0 keeps the chunks of chips that match
    # it on that dim, step 1 keeps only its own chip's.
    def step0_send_chips(h):
        return (
            [(1 - dim0_bit) + 2 * j for j in range(2)]
            if h == 0
            else [2 * (1 - dim1_bit) + j for j in range(2)]
        )

    def step0_keep_chips(h):
        return (
            [dim0_bit + 2 * j for j in range(2)]
            if h == 0
            else [2 * dim1_bit + j for j in range(2)]
        )

    def step1_send_chip(h):
        return (
            dim0_bit + 2 * (1 - dim1_bit) if h == 0 else 2 * dim1_bit + (1 - dim0_bit)
        )

    def step0_peer(h):
        return neighbours[h]

    def step1_peer(h):
        return neighbours[1 - h]

    # Reads the input in rank order; with c static this is a two-way select.
    def rows(c, b, blk):
        src = jnp.where(b == 0, order[2 * c], order[2 * c + 1])
        return pl.ds(src * chunk_rows + blk * block_rows, block_rows)

    def half(view, h):
        return view.at[pl.ds(h * half_rows, half_rows), :]

    def p1_send(blk, sl, c):
        return pltpu.make_async_remote_copy(
            src_ref=x_ref.at[rows(c, partner_bit, blk), :],
            dst_ref=p1_recv_buf.at[sl, c],
            send_sem=p1_send_sem.at[sl, c],
            recv_sem=p1_recv_sem.at[sl, c],
            device_id=partner,
            device_id_type=logical,
        )

    def load(blk, sl, c):
        return pltpu.make_async_copy(
            x_ref.at[rows(c, bit, blk), :], sums.at[sl, c], load_sem.at[sl, c]
        )

    def s0_send(sl, h, j):
        return pltpu.make_async_remote_copy(
            src_ref=half(sums.at[sl, step0_send_chips(h)[j]], h),
            dst_ref=s0_recv_buf.at[sl, h, j],
            send_sem=s0_send_sem.at[sl, h, j],
            recv_sem=s0_recv_sem.at[sl, h, j],
            device_id=step0_peer(h),
            device_id_type=logical,
        )

    def s1_send(sl, h):
        return pltpu.make_async_remote_copy(
            src_ref=half(sums.at[sl, step1_send_chip(h)], h),
            dst_ref=s1_recv_buf.at[sl, h],
            send_sem=s1_send_sem.at[sl, h],
            recv_sem=s1_recv_sem.at[sl, h],
            device_id=step1_peer(h),
            device_id_type=logical,
        )

    def store(blk, sl):
        return pltpu.make_async_copy(
            sums.at[sl, chip],
            o_ref.at[pl.ds(blk * block_rows, block_rows), :],
            store_sem.at[sl],
        )

    def wait_recv(buf, sem):
        pltpu.make_async_copy(buf, buf, sem).wait()

    def add_into(dst, src, n):
        for i in range(n // _SUB_TILE):
            s = pl.ds(i * _SUB_TILE, _SUB_TILE)
            dst[s, :] = dst[s, :] + src[s, :]

    def signal_peers(sem, peers):
        for p in peers:
            pl.semaphore_signal(sem, 1, device_id=p, device_id_type=logical)

    def issue_p1(blk, sl):
        @pl.when(blk >= _SLOTS)
        def _():
            pl.semaphore_wait(p1_slot_free.at[sl], 1)
            store(blk - _SLOTS, sl).wait()

        for c in range(4):
            p1_send(blk, sl, c).start()
            load(blk, sl, c).start()

    def finish_p1_issue_s0(blk, sl):
        for c in range(4):
            load(blk, sl, c).wait()
            wait_recv(p1_recv_buf.at[sl, c], p1_recv_sem.at[sl, c])
            add_into(sums.at[sl, c], p1_recv_buf.at[sl, c], block_rows)

        @pl.when(blk + _SLOTS < num_blocks)
        def _():
            signal_peers(p1_slot_free.at[sl], [partner])

        @pl.when(blk >= _SLOTS)
        def _():
            pl.semaphore_wait(s0_slot_free.at[sl], 2)

        for h in range(2):
            for j in range(2):
                s0_send(sl, h, j).start()

    def finish_s0_issue_s1(blk, sl):
        for h in range(2):
            for j in range(2):
                wait_recv(s0_recv_buf.at[sl, h, j], s0_recv_sem.at[sl, h, j])
                add_into(
                    half(sums.at[sl, step0_keep_chips(h)[j]], h),
                    s0_recv_buf.at[sl, h, j],
                    half_rows,
                )

        @pl.when(blk + _SLOTS < num_blocks)
        def _():
            signal_peers(s0_slot_free.at[sl], list(neighbours))

        @pl.when(blk >= _SLOTS)
        def _():
            pl.semaphore_wait(s1_slot_free.at[sl], 2)

        for h in range(2):
            s1_send(sl, h).start()

    def finish_s1_store(blk, sl):
        for h in range(2):
            wait_recv(s1_recv_buf.at[sl, h], s1_recv_sem.at[sl, h])
            add_into(half(sums.at[sl, chip], h), s1_recv_buf.at[sl, h], half_rows)

        @pl.when(blk + _SLOTS < num_blocks)
        def _():
            signal_peers(s1_slot_free.at[sl], list(neighbours))

        store(blk, sl).start()
        for c in range(4):
            p1_send(blk, sl, c).wait_send()
        for h in range(2):
            for j in range(2):
                s0_send(sl, h, j).wait_send()
            s1_send(sl, h).wait_send()

    slot = r % _SLOTS

    # Block r - 2 shares its slot with block r + 1, so it is finished first.
    @pl.when(r == 0)
    def _():
        issue_p1(0, 0)

    @pl.when(r >= 2)
    def _():
        finish_s1_store(r - 2, (r + _SLOTS - 2) % _SLOTS)

    @pl.when(r + 1 < num_blocks)
    def _():
        issue_p1(r + 1, (r + 1) % _SLOTS)

    finish_p1_issue_s0(r, slot)

    @pl.when(r >= 1)
    def _():
        finish_s0_issue_s1(r - 1, (r + _SLOTS - 1) % _SLOTS)

    @pl.when(r == num_blocks - 1)
    def _():
        finish_s0_issue_s1(r, slot)
        if num_blocks >= 2:
            finish_s1_store(r - 1, (r + _SLOTS - 1) % _SLOTS)
        finish_s1_store(r, slot)
        store(r, slot).wait()
        if num_blocks >= 2:
            store(r - 1, (r + _SLOTS - 1) % _SLOTS).wait()
        if num_blocks >= 3:
            store(r - 2, (r + _SLOTS - 2) % _SLOTS).wait()


def reduce_scatter_stream_local(
    local_x: jax.Array,
    *,
    axis_name: str = "x",
    order: tuple[int, ...] | None = None,
) -> jax.Array:
    """Mesh index i receives the sum of input row block `order[i]`."""
    rows_per_device, hidden = local_x.shape
    assert supports_stream(rows_per_device), f"{rows_per_device} rows per device"
    chunk_rows = rows_per_device // _NUM_DEVICES
    block_rows = _ROW_BLOCK
    num_blocks = chunk_rows // block_rows
    half_rows = block_rows // 2
    dtype = local_x.dtype
    # In _kernel's parameter order.
    scratch = [
        pltpu.VMEM((_SLOTS, 4, block_rows, hidden), dtype),
        pltpu.VMEM((_SLOTS, 4, block_rows, hidden), dtype),
        pltpu.VMEM((_SLOTS, 2, 2, half_rows, hidden), dtype),
        pltpu.VMEM((_SLOTS, 2, half_rows, hidden), dtype),
        pltpu.SemaphoreType.DMA((_SLOTS, 4)),
        pltpu.SemaphoreType.DMA((_SLOTS, 4)),
        pltpu.SemaphoreType.DMA((_SLOTS, 4)),
        pltpu.SemaphoreType.DMA((_SLOTS, 2, 2)),
        pltpu.SemaphoreType.DMA((_SLOTS, 2, 2)),
        pltpu.SemaphoreType.DMA((_SLOTS, 2)),
        pltpu.SemaphoreType.DMA((_SLOTS, 2)),
        pltpu.SemaphoreType.DMA((_SLOTS,)),
        pltpu.SemaphoreType.REGULAR((_SLOTS,)),
        pltpu.SemaphoreType.REGULAR((_SLOTS,)),
        pltpu.SemaphoreType.REGULAR((_SLOTS,)),
    ]
    kernel = functools.partial(
        _kernel,
        axis_name=axis_name,
        chunk_rows=chunk_rows,
        block_rows=block_rows,
        num_blocks=num_blocks,
        order=tuple(range(_NUM_DEVICES)) if order is None else tuple(order),
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((chunk_rows, hidden), dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=scratch,
            grid=(num_blocks,),
        ),
        name=f"hier_rs_stream_r{block_rows}",
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
    )(local_x)
