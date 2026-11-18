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
"""Row-streaming hierarchical reduce-scatter over 4 or 8 devices (tpu7x).

Eight devices are four dual-core chips reduced over a 2x2 hypercube in two
inter-chip steps; four devices are two chips reduced over their one link.

Grid step r issues phase 1 of block r+1, finishes phase 1 and issues step 0 of
block r, finishes step 0 and issues step 1 of block r-1, and finishes step 1
and stores block r-2. Each block's two row halves take the hypercube dims in
opposite orders, so both ICI links carry traffic on every step.

Optional extensions, disabled by default:

* ``fp8_comm`` on ``reduce_scatter_stream_local``: quantizes inter-chip steps to
  FP8 with a caller-supplied static scale (`fp8_wire.quantize`).
* ``reduce_scatter_epilogue_stream_local``: runs ``residual_epilogue`` on each
  reduced block while it is still in VMEM, returning the updated residual rows
  and, optionally, the next RMSNorm rows (FP8 + scales by default, or
  unquantized bf16 with ``next_act_dtype=jnp.bfloat16``). With
  ``fp8_wire=True``, inter-chip steps quantize to FP8 with one dynamic bf16
  scale per column of each half block packed behind the payload columns.
"""

from __future__ import annotations

import functools
import math
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import fp8_wire as fp8_wire_ops
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
_BF16_TINY = float(jnp.finfo(jnp.bfloat16).tiny)
_VMEM_LIMIT_BYTES = 56 << 20
# VMEM the compiler needs beyond the declared scratch. On tpu7x, hidden 6656
# (53.6 MiB of scratch) compiles and 7168 (57.75 MiB) runs out of VMEM.
_VMEM_HEADROOM_BYTES = 2 << 20
_MAX_EPILOGUE_WEIGHT_VECTORS = 3


class _WireMode(StrEnum):
    BF16 = "bf16"
    FP8_STATIC = "fp8_static"
    FP8_PACKET = "fp8_packet"


def scratch_bytes(hidden: int, itemsize: int, fp8_comm: bool = False) -> int:
    wire = _WireMode.FP8_STATIC if fp8_comm else _WireMode.BF16
    shapes = _vmem_shapes(
        hidden,
        jnp.bfloat16,
        lanes=1,
        wire=wire,
        num_chips=4,
        residual=False,
        rows_out=False,
    )
    return sum(
        0
        if item is None
        else math.prod(item[0]) * (1 if item[1] == fp8_wire_ops.FP8 else itemsize)
        for item in shapes
    )


MIN_STREAM_ROWS = _NUM_DEVICES * _ROW_BLOCK

SUPPORTED_NUM_DEVICES = (4, 8)


def _has_aligned_stream_rows(rows: Any, num_devices: int = _NUM_DEVICES) -> bool:
    # Avoid divmod so symbolic row counts under torch.compile trace cleanly.
    min_rows = num_devices * _ROW_BLOCK
    return rows >= min_rows and rows % min_rows == 0


def supports_stream(rows: Any, hidden: int, itemsize: int) -> bool:
    """Whether `rows` x `hidden` per device fits the row-streaming kernel.

    Its scratch grows with `hidden` (33 MiB at bf16 hidden 4096), so a wide
    shape that would not fit in VMEM is left to the micro-batched kernel.
    """
    return (
        _has_aligned_stream_rows(rows)
        and scratch_bytes(hidden, itemsize) <= _VMEM_LIMIT_BYTES - _VMEM_HEADROOM_BYTES
    )


def _epilogue_vmem_bytes(
    hidden: int, lanes: int, *, wire: _WireMode, next_act_dtype: Any
) -> int:
    shapes = _vmem_shapes(
        hidden,
        jnp.bfloat16,
        lanes=lanes,
        wire=wire,
        num_chips=4,
        residual=True,
        rows_out=True,
        next_act_dtype=next_act_dtype,
    )
    scratch = sum(
        0 if item is None else math.prod(item[0]) * jnp.dtype(item[1]).itemsize
        for item in shapes
    )
    weight_bytes = (
        _MAX_EPILOGUE_WEIGHT_VECTORS * hidden * jnp.dtype(jnp.float32).itemsize
    )
    return scratch + weight_bytes


_EPILOGUE_HIDDEN_SUPPORT: dict[tuple[int, bool], bool] = {}


def _is_fp8_next_act_dtype(next_act_dtype: Any) -> bool:
    """Returns True for `float8_e4m3fn`, False for `bfloat16`, else raises.

    Identity checks avoid calling `jnp.dtype` or comparing `ml_dtypes` objects
    during TorchDynamo fullgraph tracing.
    """
    if next_act_dtype is jnp.float8_e4m3fn:
        return True
    if next_act_dtype is jnp.bfloat16:
        return False
    next_dtype = jnp.dtype(next_act_dtype)
    if next_dtype == jnp.dtype(jnp.float8_e4m3fn):
        return True
    if next_dtype == jnp.dtype(jnp.bfloat16):
        return False
    raise ValueError(f"Unsupported next_act_dtype: {next_act_dtype}")


def supports_stream_epilogue_hidden(
    hidden: int, next_act_dtype: Any = jnp.float8_e4m3fn
) -> bool:
    """Whether `hidden` fits VMEM and lane alignment for the epilogue kernel."""
    is_fp8 = _is_fp8_next_act_dtype(next_act_dtype)
    cache_key = (hidden, is_fp8)
    supported = _EPILOGUE_HIDDEN_SUPPORT.get(cache_key)
    if supported is not None:
        return supported
    lanes = int(tpu_info().num_lanes)
    if hidden % lanes != 0:
        supported = False
    else:
        next_dtype = jnp.float8_e4m3fn if is_fp8 else jnp.bfloat16
        vmem_bytes = max(
            _epilogue_vmem_bytes(hidden, lanes, wire=wire, next_act_dtype=next_dtype)
            for wire in (_WireMode.BF16, _WireMode.FP8_PACKET)
        )
        supported = vmem_bytes <= epilogue_vmem_limit_bytes() - _VMEM_HEADROOM_BYTES
    _EPILOGUE_HIDDEN_SUPPORT[cache_key] = supported
    return supported


def supports_stream_epilogue(
    rows: Any,
    hidden: int,
    num_devices: int = _NUM_DEVICES,
    next_act_dtype: Any = jnp.float8_e4m3fn,
) -> bool:
    """Whether `[rows, hidden]` fits the streaming reduce-scatter + epilogue."""
    return supports_stream_epilogue_hidden(
        hidden, next_act_dtype=next_act_dtype
    ) and _has_aligned_stream_rows(rows, num_devices)


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
    fp8_max = fp8_wire_ops.FP8_E4M3_MAX
    scale_rows = _HALF_ROWS // 2
    n_lg, n_scale_blocks, _ = _packet_cols(hidden, lanes)
    tile = src[...].astype(jnp.float32)
    scale = jnp.max(jnp.abs(tile), axis=0, keepdims=True) * (1.0 / fp8_max)
    scale_bf16 = jnp.where(
        scale < _BF16_TINY, jnp.bfloat16(1.0), scale.astype(jnp.bfloat16)
    )
    q = (tile * (1.0 / scale_bf16.astype(jnp.float32))).clip(-fp8_max, fp8_max)
    q_u8 = lax.bitcast_convert_type(q.astype(fp8_wire_ops.FP8), jnp.uint8)

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
    q = lax.bitcast_convert_type(raw[:, :hidden], fp8_wire_ops.FP8)
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
    wire: _WireMode,
    deq_scale: float,
    inv_scale: float,
    eps: float,
    num_chips: int,
    quantize_next: bool,
):
    if next_norm_ref is None:
        o_ref, *scratch = out_and_scratch
        next_out_ref = xs_ref = None
    elif quantize_next:
        o_ref, next_out_ref, xs_ref, *scratch = out_and_scratch
    else:
        o_ref, next_out_ref, *scratch = out_and_scratch
        xs_ref = None
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
        # None on the bf16 wire; s0_send_pkt is also None on 2 chips.
        s0_send_pkt,
        s1_send_pkt,
        # None without an epilogue.
        residual_buf,
        residual_sem,
        # None unless the epilogue returns next-norm rows; xs_* None on bf16.
        next_out_buf,
        xs_buf,
        next_out_sem,
        xs_sem,
    ) = scratch

    half_rows = block_rows // 2
    uses_send_pkt = wire != _WireMode.BF16
    r = pl.program_id(0)
    topo = Topology(axis_name)
    bit = topo.cur_chiplet_bit
    chip = topo.cur_chip_id
    partner_bit = 1 - bit
    partner = topo.partner_id
    # 4 chips reduce over a 2x2 hypercube in two inter-chip steps (s0, s1).
    # 2 chips have one inter-chip link: s0 is skipped and s1 sends both halves
    # of the other chip's chunk over it.
    two_step = num_chips == 4
    neighbours = (
        topo.get_neighbor_device_id(0),
        topo.get_neighbor_device_id(1 if two_step else 0),
    )
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
            src_ref=s0_send_pkt.at[sl, h, j] if uses_send_pkt else src,
            dst_ref=s0_recv_buf.at[sl, h, j],
            send_sem=s0_send_sem.at[sl, h, j],
            recv_sem=s0_recv_sem.at[sl, h, j],
            device_id=neighbours[h],
            device_id_type=logical,
        )

    def s1_chip(h):
        return step1_send_chip(dim0_bit, dim1_bit, h) if two_step else 1 - chip

    def s1_send(sl, h):
        src = half(sums.at[sl, s1_chip(h)], h)
        return pltpu.make_async_remote_copy(
            src_ref=s1_send_pkt.at[sl, h] if uses_send_pkt else src,
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
        if next_out_ref is not None:
            copies.append(
                pltpu.make_async_copy(
                    next_out_buf.at[sl],
                    next_out_ref.at[out_rows(blk), :],
                    next_out_sem.at[sl],
                )
            )
        if xs_ref is not None:
            copies.append(
                pltpu.make_async_copy(
                    xs_buf.at[sl], xs_ref.at[out_rows(blk), :], xs_sem.at[sl]
                )
            )
        return copies

    def wait_stores(blk, sl):
        for copy in stores(blk, sl):
            copy.wait()

    def add_into(dst, src, n):
        for i in range(n // _SUB_TILE):
            s = pl.ds(i * _SUB_TILE, _SUB_TILE)
            dst[s, :] = dst[s, :] + src[s, :]

    def pack_send(src, dst):
        if wire == _WireMode.FP8_STATIC:
            for i in range(half_rows // _SUB_TILE):
                s = pl.ds(i * _SUB_TILE, _SUB_TILE)
                dst[s, :] = fp8_wire_ops.quantize(
                    src[s, :].astype(jnp.float32), inv_scale
                )
        else:
            _pack_fp8_wire(src, dst, hidden, lanes)

    def add_wire_into(dst, recv):
        if wire == _WireMode.FP8_STATIC:
            for i in range(half_rows // _SUB_TILE):
                s = pl.ds(i * _SUB_TILE, _SUB_TILE)
                dq = fp8_wire_ops.dequantize_to_bf16(recv[s, :], deq_scale)
                dst[s, :] = dst[s, :] + dq
        elif wire == _WireMode.FP8_PACKET:
            total = dst[...].astype(jnp.float32) + _unpack_fp8_wire(recv, hidden, lanes)
            dst[...] = total.astype(dst.dtype)
        else:
            add_into(dst, recv, half_rows)

    def issue_p1(blk, sl):
        @pl.when(blk >= _SLOTS)
        def _():
            pl.semaphore_wait(p1_slot_free.at[sl], 1)
            wait_stores(blk - _SLOTS, sl)

        for c in range(num_chips):
            p1_send(blk, sl, c).start()
            load(blk, sl, c).start()
        if residual_ref is not None:
            load_residual(blk, sl).start()

    def finish_p1_issue_s0(blk, sl):
        for c in range(num_chips):
            load(blk, sl, c).wait()
            wait_recv(p1_recv_buf.at[sl, c], p1_recv_sem.at[sl, c])
            add_into(sums.at[sl, c], p1_recv_buf.at[sl, c], block_rows)

        @pl.when(blk + _SLOTS < num_blocks)
        def _():
            signal_peers(p1_slot_free.at[sl], [partner], device_id_type=logical)

        if not two_step:
            return

        if uses_send_pkt:
            for h in range(2):
                for j in range(2):
                    pack_send(
                        half(
                            sums.at[sl, step0_send_chips(dim0_bit, dim1_bit, h)[j]], h
                        ),
                        s0_send_pkt.at[sl, h, j],
                    )

        @pl.when(blk >= _SLOTS)
        def _():
            pl.semaphore_wait(s0_slot_free.at[sl], 2)

        for h in range(2):
            for j in range(2):
                s0_send(sl, h, j).start()

    def finish_s0_issue_s1(blk, sl):
        if two_step:
            for h in range(2):
                for j in range(2):
                    wait_recv(s0_recv_buf.at[sl, h, j], s0_recv_sem.at[sl, h, j])
                    add_wire_into(
                        half(
                            sums.at[sl, step0_keep_chips(dim0_bit, dim1_bit, h)[j]], h
                        ),
                        s0_recv_buf.at[sl, h, j],
                    )

            @pl.when(blk + _SLOTS < num_blocks)
            def _():
                signal_peers(s0_slot_free.at[sl], neighbours, device_id_type=logical)

        if uses_send_pkt:
            for h in range(2):
                pack_send(half(sums.at[sl, s1_chip(h)], h), s1_send_pkt.at[sl, h])

        @pl.when(blk >= _SLOTS)
        def _():
            pl.semaphore_wait(s1_slot_free.at[sl], 2)

        for h in range(2):
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
            rows_out, x_next, x_scale = residual_epilogue.apply_residual_epilogue(
                sums.at[sl, chip][...],
                residual_buf.at[sl][...],
                post_norm_weight=None if post_norm_ref is None else post_norm_ref[...],
                scale=None if scale_ref is None else scale_ref[...],
                next_norm_weight=None if next_norm_ref is None else next_norm_ref[...],
                eps=eps,
                next_act_dtype=jnp.float8_e4m3fn if quantize_next else jnp.bfloat16,
            )
            sums.at[sl, chip][...] = rows_out
            if x_next is not None:
                if quantize_next:
                    assert x_scale is not None
                    next_out_buf.at[sl][...] = lax.bitcast_convert_type(
                        x_next, jnp.uint8
                    )
                    xs_buf.at[sl][...] = jnp.broadcast_to(x_scale, (block_rows, lanes))
                else:
                    next_out_buf.at[sl][...] = x_next

        for copy in stores(blk, sl):
            copy.start()
        for c in range(num_chips):
            p1_send(blk, sl, c).wait_send()
        for h in range(2):
            if two_step:
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
            signal_peers(barrier, range(2 * num_chips), device_id_type=logical)
            pl.semaphore_wait(barrier, 2 * num_chips)

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
    wire: _WireMode,
    num_chips: int = 4,
    residual: bool,
    rows_out: bool,
    next_act_dtype: Any = jnp.float8_e4m3fn,
) -> list[tuple[tuple[int, ...], Any] | None]:
    """VMEM scratch `(shape, dtype)` entries in `_kernel` parameter order."""
    two_step = num_chips == 4
    uses_send_pkt = wire != _WireMode.BF16
    wire_cols = (
        _packet_cols(hidden, lanes)[2] if wire == _WireMode.FP8_PACKET else hidden
    )
    if wire == _WireMode.FP8_STATIC:
        wire_dtype: Any = fp8_wire_ops.FP8
    elif wire == _WireMode.FP8_PACKET:
        wire_dtype = jnp.uint8
    else:
        wire_dtype = dtype
    s0_wire = (_SLOTS, 2, 2, _HALF_ROWS, wire_cols)
    s1_wire = (_SLOTS, 2, _HALF_ROWS, wire_cols)
    block = (_SLOTS, _ROW_BLOCK, hidden)
    quantize_next = jnp.dtype(next_act_dtype) == jnp.float8_e4m3fn
    next_buf_dtype = jnp.uint8 if quantize_next else dtype
    return [
        ((_SLOTS, num_chips, _ROW_BLOCK, hidden), dtype),
        ((_SLOTS, num_chips, _ROW_BLOCK, hidden), dtype),
        (s0_wire, wire_dtype) if two_step else None,
        (s1_wire, wire_dtype),
        (s0_wire, wire_dtype) if (two_step and uses_send_pkt) else None,
        (s1_wire, wire_dtype) if uses_send_pkt else None,
        (block, dtype) if residual else None,
        (block, next_buf_dtype) if rows_out else None,
        ((_SLOTS, _ROW_BLOCK, lanes), jnp.float32)
        if (rows_out and quantize_next)
        else None,
    ]


def _scratch(
    hidden: int,
    dtype: Any,
    *,
    lanes: int,
    wire: _WireMode,
    num_chips: int = 4,
    residual: bool,
    rows_out: bool,
    next_act_dtype: Any = jnp.float8_e4m3fn,
) -> list[Any]:
    """Scratch shapes in `_kernel` parameter order (`None` when unused)."""
    two_step = num_chips == 4
    quantize_next = jnp.dtype(next_act_dtype) == jnp.float8_e4m3fn
    vmem = [
        None if item is None else pltpu.VMEM(*item)
        for item in _vmem_shapes(
            hidden,
            dtype,
            lanes=lanes,
            wire=wire,
            num_chips=num_chips,
            residual=residual,
            rows_out=rows_out,
            next_act_dtype=next_act_dtype,
        )
    ]
    dma = pltpu.SemaphoreType.DMA
    regular = pltpu.SemaphoreType.REGULAR((_SLOTS,))
    return [
        *vmem[:4],
        dma((_SLOTS, num_chips)),
        dma((_SLOTS, num_chips)),
        dma((_SLOTS, num_chips)),
        dma((_SLOTS, 2, 2)) if two_step else None,
        dma((_SLOTS, 2, 2)) if two_step else None,
        dma((_SLOTS, 2)),
        dma((_SLOTS, 2)),
        dma((_SLOTS,)),
        regular,
        regular if two_step else None,
        regular,
        *vmem[4:7],
        dma((_SLOTS,)) if residual else None,
        *vmem[7:],
        dma((_SLOTS,)) if rows_out else None,
        dma((_SLOTS,)) if (rows_out and quantize_next) else None,
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
    wire: _WireMode,
    fp8_static_scale: float = 16.0,
    next_act_dtype: Any = jnp.float8_e4m3fn,
) -> Any:
    rows_per_device, hidden = local_x.shape
    lanes = int(tpu_info().num_lanes)
    num_devices = _NUM_DEVICES if order is None else len(order)
    num_chips = num_devices // 2
    assert num_devices in SUPPORTED_NUM_DEVICES, f"{num_devices} devices"
    assert _has_aligned_stream_rows(rows_per_device, num_devices), (
        f"{rows_per_device} rows per device over {num_devices} devices"
    )
    assert wire != _WireMode.FP8_PACKET or hidden % lanes == 0, (
        f"hidden {hidden} with fp8_wire"
    )
    quantize_next = _is_fp8_next_act_dtype(next_act_dtype)
    next_dtype = jnp.float8_e4m3fn if quantize_next else jnp.bfloat16
    chunk_rows = rows_per_device // num_devices
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
        wire=wire,
        deq_scale=1.0 / fp8_static_scale,
        inv_scale=fp8_static_scale,
        eps=eps,
        num_chips=num_chips,
        quantize_next=quantize_next,
    )
    name = f"hier_rs_stream_r{_ROW_BLOCK}"
    name += "" if num_devices == _NUM_DEVICES else f"_d{num_devices}"
    name += "_fp8" if wire != _WireMode.BF16 else ""
    name += "_epilogue" if residual is not None else ""
    name += "_bf16next" if (rows_out and not quantize_next) else ""
    if not rows_out:
        out_shape: Any = out
        out_specs: Any = hbm
    elif quantize_next:
        out_shape = (
            out,
            jax.ShapeDtypeStruct((chunk_rows, hidden), jnp.uint8),
            jax.ShapeDtypeStruct((chunk_rows, lanes), jnp.float32),
        )
        out_specs = (hbm, hbm, hbm)
    else:
        out_shape = (out, out)
        out_specs = (hbm, hbm)
    vmem_limit = _VMEM_LIMIT_BYTES if residual is None else epilogue_vmem_limit_bytes()
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                hbm,
                None if residual is None else hbm,
                *(None if w is None else vmem for w in operands[2:]),
            ],
            out_specs=out_specs,
            scratch_shapes=_scratch(
                hidden,
                dtype,
                lanes=lanes,
                wire=wire,
                num_chips=num_chips,
                residual=residual is not None,
                rows_out=rows_out,
                next_act_dtype=next_dtype,
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
    fp8_comm: bool = False,
    fp8_static_scale: float = 16.0,
    order: tuple[int, ...] | None = None,
) -> jax.Array:
    """Mesh index i receives the sum of input row block `order[i]`."""
    assert supports_stream(
        local_x.shape[0], local_x.shape[1], local_x.dtype.itemsize
    ), f"{local_x.shape[0]} x {local_x.shape[1]} per device"
    if fp8_comm and fp8_static_scale <= 0:
        raise ValueError(f"fp8_static_scale must be > 0, got {fp8_static_scale}")
    return _call(
        local_x,
        None,
        None,
        None,
        None,
        eps=0.0,
        axis_name=axis_name,
        order=order,
        wire=_WireMode.FP8_STATIC if fp8_comm else _WireMode.BF16,
        fp8_static_scale=fp8_static_scale,
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
    next_act_dtype: Any = jnp.float8_e4m3fn,
) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
    """`reduce_scatter_stream_local` fused with `apply_residual_epilogue` per block.

    `residual` holds this device's `[rows // num_devices, hidden]` rows.
    Returns updated residual rows and, when `next_norm_weight` is provided,
    either FP8 next-norm rows (`uint8`) plus `f32[rows // num_devices, 1]` row
    scales (`float8_e4m3fn`) or unquantized `bf16` rows with `None` scale
    (`bfloat16`).
    """
    num_devices = _NUM_DEVICES if order is None else len(order)
    assert supports_stream_epilogue(
        local_x.shape[0],
        local_x.shape[1],
        num_devices,
        next_act_dtype=next_act_dtype,
    ), f"{local_x.shape[0]} x {local_x.shape[1]} per device over {num_devices}"
    out = _call(
        local_x,
        residual,
        post_norm_weight,
        scale,
        next_norm_weight,
        eps=eps,
        axis_name=axis_name,
        order=order,
        wire=_WireMode.FP8_PACKET if fp8_wire else _WireMode.BF16,
        next_act_dtype=next_act_dtype,
    )
    if next_norm_weight is None:
        return out, None, None
    if jnp.dtype(next_act_dtype) == jnp.bfloat16:
        rows_out, x_norm = out
        return rows_out, x_norm, None
    rows_out, x_q, x_scale = out
    # VMEM->HBM async DMA requires lane-width minor dims; slice column 0.
    return rows_out, x_q, x_scale[:, :1]
