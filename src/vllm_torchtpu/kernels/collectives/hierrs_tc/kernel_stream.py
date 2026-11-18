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

Two optional extensions, both disabled by default:

* ``fp8_wire``: quantizes the two inter-chip steps to FP8 with one bf16 scale
  per column of each half block (halving inter-chip ICI bytes, lossy) while
  keeping the intra-chip phase in bf16.
* ``reduce_scatter_epilogue_stream_local``: runs ``residual_epilogue`` on each
  reduced block while it is still in VMEM, returning the updated residual rows
  and, optionally, the FP8-quantized next RMSNorm rows and scales.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.hierrs_tc import residual_epilogue
from vllm_torchtpu.kernels.collectives.hierrs_tc.topology import (
    Topology,
    epilogue_vmem_limit_bytes,
    rank_ordered_rows,
    signal_peers,
    step0_keep_chips,
    step0_send_chips,
    step1_send_chip,
    tpu_info,
    wait_recv,
)

_NUM_DEVICES = 8
_SUB_TILE = 32
_SLOTS = 3
_ROW_BLOCK = 128  # 256 needs ~66 MiB of VMEM
_HALF_ROWS = _ROW_BLOCK // 2
_FP8_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
_BF16_TINY = float(jnp.finfo(jnp.bfloat16).tiny)
_VMEM_LIMIT_BYTES = 56 << 20
# VMEM the compiler needs beyond the declared scratch. On tpu7x, hidden 6656
# (53.6 MiB of scratch) compiles and 7168 (57.75 MiB) runs out of VMEM.
_VMEM_HEADROOM_BYTES = 2 << 20
_MAX_EPILOGUE_WEIGHT_VECTORS = 3


def _vmem_buffer_shapes(hidden: int) -> list[tuple[int, ...]]:
    """VMEM buffers of `reduce_scatter_stream_local`."""
    shapes = _vmem_shapes(
        hidden,
        jnp.bfloat16,
        lanes=1,
        fp8_wire=False,
        residual=False,
        rows_out=False,
    )[:4]
    return [item[0] for item in shapes if item is not None]


def scratch_bytes(hidden: int, itemsize: int) -> int:
    return sum(math.prod(shape) for shape in _vmem_buffer_shapes(hidden)) * itemsize


MIN_STREAM_ROWS = _NUM_DEVICES * _ROW_BLOCK


def _has_aligned_stream_rows(rows: Any) -> bool:
    # Avoid divmod so symbolic row counts under torch.compile trace cleanly.
    return rows >= MIN_STREAM_ROWS and rows % MIN_STREAM_ROWS == 0


def supports_stream(rows: Any, hidden: int, itemsize: int) -> bool:
    """Whether `rows` x `hidden` per device fits the row-streaming kernel.

    Its scratch grows with `hidden` (33 MiB at bf16 hidden 4096), so a wide
    shape that would not fit in VMEM is left to the micro-batched kernel.
    """
    return (
        _has_aligned_stream_rows(rows)
        and scratch_bytes(hidden, itemsize) <= _VMEM_LIMIT_BYTES - _VMEM_HEADROOM_BYTES
    )


def _epilogue_vmem_bytes(hidden: int, lanes: int, *, fp8_wire: bool) -> int:
    shapes = _vmem_shapes(
        hidden,
        jnp.bfloat16,
        lanes=lanes,
        fp8_wire=fp8_wire,
        residual=True,
        rows_out=True,
    )
    scratch = sum(
        0 if item is None else math.prod(item[0]) * jnp.dtype(item[1]).itemsize
        for item in shapes
    )
    weight_bytes = (
        _MAX_EPILOGUE_WEIGHT_VECTORS * hidden * jnp.dtype(jnp.float32).itemsize
    )
    return scratch + weight_bytes


_EPILOGUE_HIDDEN_SUPPORT: dict[int, bool] = {}


def supports_stream_epilogue_hidden(hidden: int) -> bool:
    """Whether `hidden` fits VMEM and lane alignment for the epilogue kernel."""
    supported = _EPILOGUE_HIDDEN_SUPPORT.get(hidden)
    if supported is not None:
        return supported
    lanes = int(tpu_info().num_lanes)
    if hidden % lanes != 0:
        supported = False
    else:
        vmem_bytes = max(
            _epilogue_vmem_bytes(hidden, lanes, fp8_wire=False),
            _epilogue_vmem_bytes(hidden, lanes, fp8_wire=True),
        )
        supported = vmem_bytes <= epilogue_vmem_limit_bytes() - _VMEM_HEADROOM_BYTES
    _EPILOGUE_HIDDEN_SUPPORT[hidden] = supported
    return supported


def supports_stream_epilogue(rows: Any, hidden: int) -> bool:
    """Whether `[rows, hidden]` fits the streaming reduce-scatter + epilogue."""
    return supports_stream_epilogue_hidden(hidden) and _has_aligned_stream_rows(rows)


def _packet_cols(hidden: int, lanes: int) -> tuple[int, int, int]:
    """`(lane_groups, scale_blocks, packet_cols)` for an FP8 wire packet.

    The `[1, hidden]` bf16 column scales form `hidden // lanes` lane groups.
    `_HALF_ROWS // 2` groups bitcast into one `u8[_HALF_ROWS, lanes]` block,
    appended behind the FP8 payload columns.
    """
    scale_rows = _HALF_ROWS // 2
    n_lg = hidden // lanes
    n_scale_blocks = -(-n_lg // scale_rows)
    return n_lg, n_scale_blocks, hidden + n_scale_blocks * lanes


def _pack_fp8_wire(src: Any, dst_pkt: Any, hidden: int, lanes: int) -> None:
    """Quantizes `bf16[64, hidden]` into `u8[64, packet_cols]`."""
    scale_rows = _HALF_ROWS // 2
    n_lg, n_scale_blocks, _ = _packet_cols(hidden, lanes)
    tile = src[...].astype(jnp.float32)
    scale = jnp.max(jnp.abs(tile), axis=0, keepdims=True) * (1.0 / _FP8_MAX)
    scale_bf16 = jnp.where(
        scale < _BF16_TINY, jnp.bfloat16(1.0), scale.astype(jnp.bfloat16)
    )
    q = (tile * (1.0 / scale_bf16.astype(jnp.float32))).clip(-_FP8_MAX, _FP8_MAX)
    q_u8 = lax.bitcast_convert_type(q.astype(jnp.float8_e4m3fn), jnp.uint8)

    s_2d = scale_bf16.reshape(n_lg, lanes)
    pad_rows = n_scale_blocks * scale_rows - n_lg
    if pad_rows > 0:
        s_2d = jnp.pad(s_2d, ((0, pad_rows), (0, 0)))
    s_u8 = [
        pltpu.bitcast(s_2d[i * scale_rows : (i + 1) * scale_rows, :], jnp.uint8)
        for i in range(n_scale_blocks)
    ]
    dst_pkt[...] = jnp.concatenate([q_u8, *s_u8], axis=1)


def _unpack_fp8_wire(src_pkt: Any, hidden: int, lanes: int) -> jax.Array:
    """Dequantizes `u8[64, packet_cols]` into `f32[64, hidden]`."""
    n_lg, n_scale_blocks, _ = _packet_cols(hidden, lanes)
    raw = src_pkt[...]
    q = lax.bitcast_convert_type(raw[:, :hidden], jnp.float8_e4m3fn)
    s_parts = [
        pltpu.bitcast(
            raw[:, hidden + i * lanes : hidden + (i + 1) * lanes], jnp.bfloat16
        )
        for i in range(n_scale_blocks)
    ]
    scale = jnp.concatenate(s_parts, axis=0)[:n_lg, :].reshape(1, hidden)
    return q.astype(jnp.float32) * scale.astype(jnp.float32)


def _kernel(
    x_ref,
    # Optional epilogue operands; None when unused.
    residual_ref,
    post_norm_ref,
    scale_ref,
    next_norm_ref,
    *out_and_scratch,
    axis_name: str,
    chunk_rows: int,
    block_rows: int,
    num_blocks: int,
    order: tuple[int, ...],
    hidden: int,
    lanes: int,
    fp8_wire: bool,
    eps: float,
):
    if next_norm_ref is None:
        o_ref, *scratch = out_and_scratch
        next_out_ref = xs_ref = None
    else:
        o_ref, next_out_ref, xs_ref, *scratch = out_and_scratch
    (
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
        # None unless fp8_wire.
        s0_send_pkt,
        s1_send_pkt,
        # None without an epilogue.
        residual_buf,
        residual_sem,
        # None unless the epilogue returns next-norm rows.
        next_out_buf,
        xs_buf,
        next_out_sem,
        xs_sem,
    ) = scratch

    half_rows = block_rows // 2
    r = pl.program_id(0)
    topo = Topology(axis_name)
    bit = topo.cur_chiplet_bit
    chip = topo.cur_chip_id
    partner_bit = 1 - bit
    partner = topo.partner_id
    neighbours = (topo.get_neighbor_device_id(0), topo.get_neighbor_device_id(1))
    dim0_bit = chip & 1
    dim1_bit = chip // 2
    logical = pl.DeviceIdType.LOGICAL

    def rows(c, b, blk):
        return rank_ordered_rows(order, c, b, chunk_rows, blk, block_rows)

    def half(view, h):
        return view.at[pl.ds(h * half_rows, half_rows), :]

    def out_rows(blk):
        return pl.ds(blk * block_rows, block_rows)

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

    def load_residual(blk, sl):
        return pltpu.make_async_copy(
            residual_ref.at[out_rows(blk), :], residual_buf.at[sl], residual_sem.at[sl]
        )

    def s0_send(sl, h, j):
        src = half(sums.at[sl, step0_send_chips(dim0_bit, dim1_bit, h)[j]], h)
        return pltpu.make_async_remote_copy(
            src_ref=s0_send_pkt.at[sl, h, j] if fp8_wire else src,
            dst_ref=s0_recv_buf.at[sl, h, j],
            send_sem=s0_send_sem.at[sl, h, j],
            recv_sem=s0_recv_sem.at[sl, h, j],
            device_id=neighbours[h],
            device_id_type=logical,
        )

    def s1_send(sl, h):
        src = half(sums.at[sl, step1_send_chip(dim0_bit, dim1_bit, h)], h)
        return pltpu.make_async_remote_copy(
            src_ref=s1_send_pkt.at[sl, h] if fp8_wire else src,
            dst_ref=s1_recv_buf.at[sl, h],
            send_sem=s1_send_sem.at[sl, h],
            recv_sem=s1_recv_sem.at[sl, h],
            device_id=neighbours[1 - h],
            device_id_type=logical,
        )

    def stores(blk, sl):
        copies = [
            pltpu.make_async_copy(
                sums.at[sl, chip], o_ref.at[out_rows(blk), :], store_sem.at[sl]
            )
        ]
        if next_out_ref is not None and xs_ref is not None:
            copies += [
                pltpu.make_async_copy(
                    next_out_buf.at[sl],
                    next_out_ref.at[out_rows(blk), :],
                    next_out_sem.at[sl],
                ),
                pltpu.make_async_copy(
                    xs_buf.at[sl], xs_ref.at[out_rows(blk), :], xs_sem.at[sl]
                ),
            ]
        return copies

    def wait_stores(blk, sl):
        for copy in stores(blk, sl):
            copy.wait()

    def add_into(dst, src, n):
        for i in range(n // _SUB_TILE):
            s = pl.ds(i * _SUB_TILE, _SUB_TILE)
            dst[s, :] = dst[s, :] + src[s, :]

    def add_wire_into(dst, recv):
        if fp8_wire:
            total = dst[...].astype(jnp.float32) + _unpack_fp8_wire(recv, hidden, lanes)
            dst[...] = total.astype(dst.dtype)
        else:
            add_into(dst, recv, half_rows)

    def issue_p1(blk, sl):
        @pl.when(blk >= _SLOTS)
        def _():
            pl.semaphore_wait(p1_slot_free.at[sl], 1)
            wait_stores(blk - _SLOTS, sl)

        for c in range(4):
            p1_send(blk, sl, c).start()
            load(blk, sl, c).start()
        if residual_ref is not None:
            load_residual(blk, sl).start()

    def finish_p1_issue_s0(blk, sl):
        for c in range(4):
            load(blk, sl, c).wait()
            wait_recv(p1_recv_buf.at[sl, c], p1_recv_sem.at[sl, c])
            add_into(sums.at[sl, c], p1_recv_buf.at[sl, c], block_rows)

        @pl.when(blk + _SLOTS < num_blocks)
        def _():
            signal_peers(p1_slot_free.at[sl], [partner], device_id_type=logical)

        @pl.when(blk >= _SLOTS)
        def _():
            pl.semaphore_wait(s0_slot_free.at[sl], 2)

        for h in range(2):
            for j in range(2):
                if fp8_wire:
                    _pack_fp8_wire(
                        half(
                            sums.at[sl, step0_send_chips(dim0_bit, dim1_bit, h)[j]], h
                        ),
                        s0_send_pkt.at[sl, h, j],
                        hidden,
                        lanes,
                    )
                s0_send(sl, h, j).start()

    def finish_s0_issue_s1(blk, sl):
        for h in range(2):
            for j in range(2):
                wait_recv(s0_recv_buf.at[sl, h, j], s0_recv_sem.at[sl, h, j])
                add_wire_into(
                    half(sums.at[sl, step0_keep_chips(dim0_bit, dim1_bit, h)[j]], h),
                    s0_recv_buf.at[sl, h, j],
                )

        @pl.when(blk + _SLOTS < num_blocks)
        def _():
            signal_peers(s0_slot_free.at[sl], neighbours, device_id_type=logical)

        @pl.when(blk >= _SLOTS)
        def _():
            pl.semaphore_wait(s1_slot_free.at[sl], 2)

        for h in range(2):
            if fp8_wire:
                _pack_fp8_wire(
                    half(sums.at[sl, step1_send_chip(dim0_bit, dim1_bit, h)], h),
                    s1_send_pkt.at[sl, h],
                    hidden,
                    lanes,
                )
            s1_send(sl, h).start()

    def finish_s1_store(blk, sl):
        for h in range(2):
            wait_recv(s1_recv_buf.at[sl, h], s1_recv_sem.at[sl, h])
            add_wire_into(half(sums.at[sl, chip], h), s1_recv_buf.at[sl, h])

        @pl.when(blk + _SLOTS < num_blocks)
        def _():
            signal_peers(s1_slot_free.at[sl], neighbours, device_id_type=logical)

        if residual_ref is not None:
            load_residual(blk, sl).wait()
            rows_out, x_q, x_scale = residual_epilogue.apply_residual_epilogue(
                sums.at[sl, chip][...],
                residual_buf.at[sl][...],
                post_norm_weight=None if post_norm_ref is None else post_norm_ref[...],
                scale=None if scale_ref is None else scale_ref[...],
                next_norm_weight=None if next_norm_ref is None else next_norm_ref[...],
                eps=eps,
            )
            sums.at[sl, chip][...] = rows_out
            if x_q is not None and x_scale is not None:
                next_out_buf.at[sl][...] = lax.bitcast_convert_type(x_q, jnp.uint8)
                xs_buf.at[sl][...] = jnp.broadcast_to(x_scale, (block_rows, lanes))

        for copy in stores(blk, sl):
            copy.start()
        for c in range(4):
            p1_send(blk, sl, c).wait_send()
        for h in range(2):
            for j in range(2):
                s0_send(sl, h, j).wait_send()
            s1_send(sl, h).wait_send()

    slot = r % _SLOTS

    if residual_ref is not None:
        # Consecutive sublayers invoke this kernel back-to-back with matmul and
        # attention work in between; synchronize entry so no device DMAs into a
        # peer's VMEM scratch before the peer has entered the kernel.
        @pl.when(r == 0)
        def _():
            barrier = pltpu.get_barrier_semaphore()
            signal_peers(barrier, range(_NUM_DEVICES), device_id_type=logical)
            pl.semaphore_wait(barrier, _NUM_DEVICES)

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
        wait_stores(r, slot)
        if num_blocks >= 2:
            wait_stores(r - 1, (r + _SLOTS - 1) % _SLOTS)
        if num_blocks >= 3:
            wait_stores(r - 2, (r + _SLOTS - 2) % _SLOTS)


def _vmem_shapes(
    hidden: int,
    dtype: Any,
    *,
    lanes: int,
    fp8_wire: bool,
    residual: bool,
    rows_out: bool,
) -> list[tuple[tuple[int, ...], Any] | None]:
    """VMEM scratch `(shape, dtype)` entries in `_kernel` parameter order."""
    wire_cols = _packet_cols(hidden, lanes)[2] if fp8_wire else hidden
    wire_dtype = jnp.uint8 if fp8_wire else dtype
    s0_wire = (_SLOTS, 2, 2, _HALF_ROWS, wire_cols)
    s1_wire = (_SLOTS, 2, _HALF_ROWS, wire_cols)
    block = (_SLOTS, _ROW_BLOCK, hidden)
    return [
        ((_SLOTS, 4, _ROW_BLOCK, hidden), dtype),
        ((_SLOTS, 4, _ROW_BLOCK, hidden), dtype),
        (s0_wire, wire_dtype),
        (s1_wire, wire_dtype),
        (s0_wire, jnp.uint8) if fp8_wire else None,
        (s1_wire, jnp.uint8) if fp8_wire else None,
        (block, dtype) if residual else None,
        (block, jnp.uint8) if rows_out else None,
        ((_SLOTS, _ROW_BLOCK, lanes), jnp.float32) if rows_out else None,
    ]


def _scratch(
    hidden: int,
    dtype: Any,
    *,
    lanes: int,
    fp8_wire: bool,
    residual: bool,
    rows_out: bool,
) -> list[Any]:
    """Scratch shapes in `_kernel` parameter order (`None` when unused)."""
    vmem = [
        None if item is None else pltpu.VMEM(*item)
        for item in _vmem_shapes(
            hidden,
            dtype,
            lanes=lanes,
            fp8_wire=fp8_wire,
            residual=residual,
            rows_out=rows_out,
        )
    ]
    dma = pltpu.SemaphoreType.DMA
    regular = pltpu.SemaphoreType.REGULAR((_SLOTS,))
    return [
        *vmem[:4],
        dma((_SLOTS, 4)),
        dma((_SLOTS, 4)),
        dma((_SLOTS, 4)),
        dma((_SLOTS, 2, 2)),
        dma((_SLOTS, 2, 2)),
        dma((_SLOTS, 2)),
        dma((_SLOTS, 2)),
        dma((_SLOTS,)),
        regular,
        regular,
        regular,
        *vmem[4:7],
        dma((_SLOTS,)) if residual else None,
        *vmem[7:],
        dma((_SLOTS,)) if rows_out else None,
        dma((_SLOTS,)) if rows_out else None,
    ]


def _call(
    local_x: jax.Array,
    residual: jax.Array | None,
    post_norm_weight: jax.Array | None,
    scale: jax.Array | None,
    next_norm_weight: jax.Array | None,
    *,
    eps: float,
    axis_name: str,
    order: tuple[int, ...] | None,
    fp8_wire: bool,
) -> Any:
    rows_per_device, hidden = local_x.shape
    lanes = int(tpu_info().num_lanes)
    assert _has_aligned_stream_rows(rows_per_device), (
        f"{rows_per_device} rows per device"
    )
    assert not fp8_wire or hidden % lanes == 0, f"hidden {hidden} with fp8_wire"
    chunk_rows = rows_per_device // _NUM_DEVICES
    num_blocks = chunk_rows // _ROW_BLOCK
    dtype = local_x.dtype
    if residual is not None:
        assert dtype == jnp.bfloat16 and residual.dtype == jnp.bfloat16, (
            f"expected bfloat16 local_x and residual, got {dtype} and {residual.dtype}"
        )
        assert residual.shape == (chunk_rows, hidden), (
            f"expected residual {(chunk_rows, hidden)}, got {residual.shape}"
        )
    rows_out = next_norm_weight is not None

    def row_vector(w: jax.Array | None) -> jax.Array | None:
        if w is None:
            return None
        return jnp.broadcast_to(w.astype(jnp.float32).reshape(1, -1), (1, hidden))

    out = jax.ShapeDtypeStruct((chunk_rows, hidden), dtype)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    operands = (
        local_x,
        residual,
        row_vector(post_norm_weight),
        row_vector(scale),
        row_vector(next_norm_weight),
    )
    kernel = functools.partial(
        _kernel,
        axis_name=axis_name,
        chunk_rows=chunk_rows,
        block_rows=_ROW_BLOCK,
        num_blocks=num_blocks,
        order=tuple(range(_NUM_DEVICES)) if order is None else tuple(order),
        hidden=hidden,
        lanes=lanes,
        fp8_wire=fp8_wire,
        eps=eps,
    )
    name = f"hier_rs_stream_r{_ROW_BLOCK}"
    name += "_fp8" if fp8_wire else ""
    name += "_epilogue" if residual is not None else ""
    vmem_limit = _VMEM_LIMIT_BYTES if residual is None else epilogue_vmem_limit_bytes()
    return pl.pallas_call(
        kernel,
        out_shape=(
            (
                out,
                jax.ShapeDtypeStruct((chunk_rows, hidden), jnp.uint8),
                jax.ShapeDtypeStruct((chunk_rows, lanes), jnp.float32),
            )
            if rows_out
            else out
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                hbm,
                None if residual is None else hbm,
                *(None if w is None else vmem for w in operands[2:]),
            ],
            out_specs=(hbm, hbm, hbm) if rows_out else hbm,
            scratch_shapes=_scratch(
                hidden,
                dtype,
                lanes=lanes,
                fp8_wire=fp8_wire,
                residual=residual is not None,
                rows_out=rows_out,
            ),
            grid=(num_blocks,),
        ),
        name=name,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit,
            disable_bounds_checks=True,
            collective_id=None if residual is None else 0,
        ),
    )(*operands)


def reduce_scatter_stream_local(
    local_x: jax.Array,
    *,
    axis_name: str = "x",
    order: tuple[int, ...] | None = None,
) -> jax.Array:
    """Mesh index i receives the sum of input row block `order[i]`."""
    assert supports_stream(
        local_x.shape[0], local_x.shape[1], local_x.dtype.itemsize
    ), f"{local_x.shape[0]} x {local_x.shape[1]} per device"
    return _call(
        local_x,
        None,
        None,
        None,
        None,
        eps=0.0,
        axis_name=axis_name,
        order=order,
        fp8_wire=False,
    )


def reduce_scatter_epilogue_stream_local(
    local_x: jax.Array,
    residual: jax.Array,
    *,
    post_norm_weight: jax.Array | None = None,
    scale: jax.Array | None = None,
    next_norm_weight: jax.Array | None = None,
    eps: float,
    axis_name: str = "x",
    order: tuple[int, ...] | None = None,
    fp8_wire: bool = False,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """`reduce_scatter_stream_local` fused with `apply_residual_epilogue` per block.

    `residual` holds this device's `[rows // 8, hidden]` rows. Returns updated
    residual rows and, when `next_norm_weight` is provided, FP8 next-norm rows
    (`uint8`) plus `f32[rows // 8, 1]` row scales.
    """
    assert supports_stream_epilogue(local_x.shape[0], local_x.shape[1]), (
        f"{local_x.shape[0]} x {local_x.shape[1]} per device"
    )
    out = _call(
        local_x,
        residual,
        post_norm_weight,
        scale,
        next_norm_weight,
        eps=eps,
        axis_name=axis_name,
        order=order,
        fp8_wire=fp8_wire,
    )
    if next_norm_weight is None:
        return out, None, None
    rows_out, x_q, x_scale = out
    # VMEM->HBM async DMA requires lane-width (128) minor dims; slice column 0.
    return rows_out, x_q, x_scale[:, :1]
