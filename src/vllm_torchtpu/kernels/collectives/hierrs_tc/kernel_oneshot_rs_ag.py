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
"""Single-shot VMEM hierarchical reduce-scatter, epilogue, and all-gather.

Supports padded token buckets in `128 <= rows <= 1024` (`rows % 128 == 0`, so
each of the 8 devices owns `c_rows >= 16` tile-aligned rows with zero runtime
padding); dense sequence parallelism dispatches `256 <= rows < 1024`
buckets here and `>= 1024` to `kernel_stream.py`:

1. Intra-chip reduce-scatter across the two TensorCores of each chip, reading
   `partial_ref` in `order`.
2. Two-step 2x2 inter-chip hypercube reduce-scatter split by column halves in
   unquantized `bf16` (~1 MiB per hop at `<= 1024` tokens).
3. `residual_epilogue.apply_residual_epilogue` in VMEM on this rank's shard.
4. Two-tier hypercube + intra-chip all-gather in VMEM, unpacked in VREGs into TP
   rank order so every rank receives `uint8[rows, hidden]` FP8 activations and
   `f32[rows, 1]` scales (or `bf16[rows, hidden]` residual rows when
   `next_norm_weight` is `None`) with no separate XLA collective or unpack op.
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
_NUM_CHIPS = _NUM_DEVICES // 2
SUPPORTED_NUM_DEVICES = (_NUM_DEVICES,)
_TILE_ROWS = 16  # Sublane tile height for u16/bf16; 8 * 16 = 128 aligns u8 (32).
MIN_ROWS = _NUM_DEVICES * _TILE_ROWS
_MAX_CHUNK_ROWS = 128
_MAX_ROWS = _NUM_DEVICES * _MAX_CHUNK_ROWS
_VMEM_HEADROOM_BYTES = 2 << 20


def _ag_extra_u16_cols(h_half: int, lanes: int) -> int:
    # Two lane-wide u16 blocks carry the f32 row scales; pad one more block when
    # `h_half // lanes` is odd so `ag_cols // 2` stays a multiple of `lanes`.
    return 2 * lanes if (h_half // lanes) % 2 == 0 else 3 * lanes


def _ag_packet_cols(hidden: int, lanes: int, *, has_next: bool) -> int:
    """Columns of one all-gather packet: `u16` FP8 pairs + row scales, or `bf16`."""
    if not has_next:
        return hidden
    h_half = hidden // 2
    return h_half + _ag_extra_u16_cols(h_half, lanes)


def _vmem_scratch_shapes(
    c_rows: int, hidden: int, lanes: int, *, has_next: bool
) -> list[tuple[tuple[int, ...], Any]]:
    """`(shape, dtype)` of each VMEM scratch buffer, in `_kernel`'s unpack order."""
    h_half = hidden // 2
    ag_cols = _ag_packet_cols(hidden, lanes, has_next=has_next)
    ag_dtype = jnp.uint16 if has_next else jnp.bfloat16
    return [
        ((_NUM_CHIPS, c_rows, hidden), jnp.bfloat16),  # sums
        ((_NUM_CHIPS, c_rows, hidden), jnp.bfloat16),  # p1_recv
        ((2, 2, c_rows, h_half), jnp.bfloat16),  # s0_recv
        ((2, c_rows, h_half), jnp.bfloat16),  # s1_recv
        ((_NUM_CHIPS, c_rows, ag_cols), ag_dtype),  # ag_own_pkt
        ((_NUM_CHIPS, c_rows, ag_cols), ag_dtype),  # ag_partner_pkt
    ]


def _vmem_bytes(c_rows: int, hidden: int, lanes: int, *, has_next: bool) -> int:
    out_dtype = jnp.uint8 if has_next else jnp.bfloat16
    total_rows = _NUM_DEVICES * c_rows
    shapes = [
        # Inputs in VMEM: partial, residual, and up to three [1, hidden] vectors.
        ((total_rows, hidden), jnp.bfloat16),
        ((c_rows, hidden), jnp.bfloat16),
        ((1, hidden), jnp.float32),
        ((1, hidden), jnp.float32),
        ((1, hidden), jnp.float32),
        # Outputs in VMEM: residual, gathered rows, optional row scales.
        ((c_rows, hidden), jnp.bfloat16),
        ((total_rows, hidden), out_dtype),
        *([((total_rows, lanes), jnp.float32)] if has_next else []),
        *_vmem_scratch_shapes(c_rows, hidden, lanes, has_next=has_next),
    ]
    return sum(math.prod(shape) * jnp.dtype(dtype).itemsize for shape, dtype in shapes)


_ONESHOT_HIDDEN_SUPPORT: dict[int, bool] = {}


def supports_oneshot_rs_ag_hidden(hidden: int) -> bool:
    """Whether `hidden` fits VMEM and lane alignment for single-shot RS + AG."""
    supported = _ONESHOT_HIDDEN_SUPPORT.get(hidden)
    if supported is not None:
        return supported
    lanes = int(tpu_info().num_lanes)
    if hidden < 2 * lanes or hidden % (2 * lanes) != 0:
        supported = False
    else:
        vmem_need = max(
            _vmem_bytes(_MAX_CHUNK_ROWS, hidden, lanes, has_next=False),
            _vmem_bytes(_MAX_CHUNK_ROWS, hidden, lanes, has_next=True),
        )
        supported = vmem_need <= epilogue_vmem_limit_bytes() - _VMEM_HEADROOM_BYTES
    _ONESHOT_HIDDEN_SUPPORT[hidden] = supported
    return supported


def supports_oneshot_rs_ag(
    rows: Any, hidden: int, num_devices: int = _NUM_DEVICES
) -> bool:
    """Whether `[rows, hidden]` fits the single-shot VMEM RS + AG kernel."""
    if num_devices not in SUPPORTED_NUM_DEVICES or not supports_oneshot_rs_ag_hidden(
        hidden
    ):
        return False
    # Symbolic-shape friendly modulo check (no divmod): requires c_rows >= 16
    # and c_rows % 16 == 0 so neither inputs nor outputs need call-time padding.
    return rows >= MIN_ROWS and rows <= _MAX_ROWS and rows % MIN_ROWS == 0


def _kernel(
    partial_ref,  # bf16[8 * c_rows, hidden]
    residual_ref,  # bf16[c_rows, hidden]
    post_norm_ref,  # None or f32[1, hidden]
    scale_ref,  # None or f32[1, hidden]
    next_norm_ref,  # None or f32[1, hidden]
    *out_and_scratch,
    axis_name: str,
    c_rows: int,
    hidden: int,
    lanes: int,
    order: tuple[int, ...],
    eps: float,
):
    has_next = next_norm_ref is not None
    if has_next:
        res_out_ref, gathered_ref, xs_out_ref, *scratch_refs = out_and_scratch
    else:
        res_out_ref, gathered_ref, *scratch_refs = out_and_scratch
        xs_out_ref = None
    (
        sums,
        p1_recv,
        s0_recv,
        s1_recv,
        ag_own_pkt,
        ag_partner_pkt,
        p1_send_sem,
        p1_recv_sem,
        s0_send_sem,
        s0_recv_sem,
        s1_send_sem,
        s1_recv_sem,
        ag0_send_sem,
        ag0_recv_sem,
        ag1_send_sem,
        ag1_recv_sem,
        p2_send_sem,
        p2_recv_sem,
    ) = scratch_refs

    h_half = hidden // 2
    extra_u16_cols = _ag_extra_u16_cols(h_half, lanes)
    ag_half = _ag_packet_cols(hidden, lanes, has_next=has_next) // 2
    topo = Topology(axis_name)
    bit = topo.cur_chiplet_bit
    partner_bit = 1 - bit
    chip = topo.cur_chip_id
    partner = topo.partner_id
    neighbours = (topo.get_neighbor_device_id(0), topo.get_neighbor_device_id(1))
    dim0_bit = chip & 1
    dim1_bit = chip // 2
    logical = pl.DeviceIdType.LOGICAL

    barrier = pltpu.get_barrier_semaphore()
    signal_peers(barrier, range(_NUM_DEVICES), device_id_type=logical)
    pl.semaphore_wait(barrier, _NUM_DEVICES)

    def col_half(view, h, width):
        return view.at[:, pl.ds(h * width, width)]

    # 1. Intra-chip reduce-scatter, reading `partial_ref` in `order`.
    p1_ops = []
    for c in range(_NUM_CHIPS):
        partner_slice = rank_ordered_rows(order, c, partner_bit, c_rows)
        op = pltpu.make_async_remote_copy(
            src_ref=partial_ref.at[partner_slice, :],
            dst_ref=p1_recv.at[c],
            send_sem=p1_send_sem.at[c],
            recv_sem=p1_recv_sem.at[c],
            device_id=partner,
            device_id_type=logical,
        )
        op.start()
        p1_ops.append(op)

    for c in range(_NUM_CHIPS):
        own_slice = rank_ordered_rows(order, c, bit, c_rows)
        wait_recv(p1_recv.at[c], p1_recv_sem.at[c])
        sums.at[c][...] = (
            partial_ref[own_slice, :].astype(jnp.float32)
            + p1_recv.at[c][...].astype(jnp.float32)
        ).astype(jnp.bfloat16)

    # 2. Two-step 2x2 hypercube reduce-scatter split by column halves (bf16 wire).
    s0_ops = []
    for h in range(2):
        send_chips = step0_send_chips(dim0_bit, dim1_bit, h)
        for j in range(2):
            op = pltpu.make_async_remote_copy(
                src_ref=col_half(sums.at[send_chips[j]], h, h_half),
                dst_ref=s0_recv.at[h, j],
                send_sem=s0_send_sem.at[h, j],
                recv_sem=s0_recv_sem.at[h, j],
                device_id=neighbours[h],
                device_id_type=logical,
            )
            op.start()
            s0_ops.append(op)

    for h in range(2):
        keep_chips = step0_keep_chips(dim0_bit, dim1_bit, h)
        for j in range(2):
            wait_recv(s0_recv.at[h, j], s0_recv_sem.at[h, j])
            dst_v = col_half(sums.at[keep_chips[j]], h, h_half)
            dst_v[...] = (
                dst_v[...].astype(jnp.float32)
                + s0_recv.at[h, j][...].astype(jnp.float32)
            ).astype(jnp.bfloat16)

    s1_ops = []
    for h in range(2):
        send_chip = step1_send_chip(dim0_bit, dim1_bit, h)
        op = pltpu.make_async_remote_copy(
            src_ref=col_half(sums.at[send_chip], h, h_half),
            dst_ref=s1_recv.at[h],
            send_sem=s1_send_sem.at[h],
            recv_sem=s1_recv_sem.at[h],
            device_id=neighbours[1 - h],
            device_id_type=logical,
        )
        op.start()
        s1_ops.append(op)

    halves_f32 = []
    for h in range(2):
        wait_recv(s1_recv.at[h], s1_recv_sem.at[h])
        halves_f32.append(
            col_half(sums.at[chip], h, h_half)[...].astype(jnp.float32)
            + s1_recv.at[h][...].astype(jnp.float32)
        )
    reduced_f32 = jnp.concatenate(halves_f32, axis=1)

    # 3. Residual epilogue in VMEM.
    res_new, x_q, x_scale = residual_epilogue.apply_residual_epilogue(
        reduced_f32,
        residual_ref[...],
        post_norm_weight=None if post_norm_ref is None else post_norm_ref[...],
        scale=None if scale_ref is None else scale_ref[...],
        next_norm_weight=None if next_norm_ref is None else next_norm_ref[...],
        eps=eps,
    )
    res_out_ref[...] = res_new

    if has_next:
        assert x_q is not None and x_scale is not None
        xq_u8 = lax.bitcast_convert_type(x_q, jnp.uint8)
        q_lo = xq_u8[:, :h_half].astype(jnp.uint16)
        q_hi = xq_u8[:, h_half:].astype(jnp.uint16)
        q_u16 = q_lo | (q_hi << 8)
        xs_wide = jnp.broadcast_to(x_scale, (c_rows, lanes))
        u32_bits = lax.bitcast_convert_type(xs_wide, jnp.uint32)
        s_lo = (u32_bits & 0xFFFF).astype(jnp.uint16)
        s_hi = (u32_bits >> 16).astype(jnp.uint16)
        parts = [q_u16, s_lo, s_hi]
        if extra_u16_cols == 3 * lanes:
            parts.append(jnp.zeros((c_rows, lanes), dtype=jnp.uint16))
        ag_own_pkt.at[chip][...] = jnp.concatenate(parts, axis=1)
    else:
        ag_own_pkt.at[chip][...] = res_new

    # 4. Two-tier hypercube + intra-chip all-gather, unpacked in VREGs into `order`.
    ag0_ops = []
    for h in range(2):
        op = pltpu.make_async_remote_copy(
            src_ref=col_half(ag_own_pkt.at[chip], h, ag_half),
            dst_ref=col_half(ag_own_pkt.at[chip], h, ag_half),
            send_sem=ag0_send_sem.at[h],
            recv_sem=ag0_recv_sem.at[h],
            device_id=neighbours[h],
            device_id_type=logical,
        )
        op.start()
        ag0_ops.append(op)

    for h in range(2):
        dst_c = chip ^ 1 if h == 0 else chip ^ 2
        wait_recv(col_half(ag_own_pkt.at[dst_c], h, ag_half), ag0_recv_sem.at[h])

    ag1_ops = []
    for h in range(2):
        srcs = [chip, chip ^ 1] if h == 0 else [chip, chip ^ 2]
        for j in range(2):
            op = pltpu.make_async_remote_copy(
                src_ref=col_half(ag_own_pkt.at[srcs[j]], h, ag_half),
                dst_ref=col_half(ag_own_pkt.at[srcs[j]], h, ag_half),
                send_sem=ag1_send_sem.at[h, j],
                recv_sem=ag1_recv_sem.at[h, j],
                device_id=neighbours[1 - h],
                device_id_type=logical,
            )
            op.start()
            ag1_ops.append(op)

    for h in range(2):
        dsts = [chip ^ 2, chip ^ 3] if h == 0 else [chip ^ 1, chip ^ 3]
        for j in range(2):
            wait_recv(
                col_half(ag_own_pkt.at[dsts[j]], h, ag_half), ag1_recv_sem.at[h, j]
            )

    p2_ops = []
    for c in range(_NUM_CHIPS):
        op = pltpu.make_async_remote_copy(
            src_ref=ag_own_pkt.at[c],
            dst_ref=ag_partner_pkt.at[c],
            send_sem=p2_send_sem.at[c],
            recv_sem=p2_recv_sem.at[c],
            device_id=partner,
            device_id_type=logical,
        )
        op.start()
        p2_ops.append(op)

    for c in range(_NUM_CHIPS):
        wait_recv(ag_partner_pkt.at[c], p2_recv_sem.at[c])

    # Assemble rank-ordered packets in VREGs (static `order.index(r)` eliminates
    # dynamic VMEM output slicing and gives `8 * c_rows >= 128` rows for u8).
    ordered_pkts = []
    for tp_rank in range(_NUM_DEVICES):
        mesh_idx = order.index(tp_rank)
        pkt_chip, pkt_bit = divmod(mesh_idx, 2)
        ordered_pkts.append(
            jnp.where(
                bit == pkt_bit,
                ag_own_pkt.at[pkt_chip][...],
                ag_partner_pkt.at[pkt_chip][...],
            )
        )
    all_pkt = jnp.concatenate(ordered_pkts, axis=0)

    if has_next:
        assert xs_out_ref is not None
        q_u16 = all_pkt[:, :h_half]
        q_lo = (q_u16 & 0xFF).astype(jnp.uint8)
        q_hi = ((q_u16 >> 8) & 0xFF).astype(jnp.uint8)
        gathered_ref[...] = jnp.concatenate([q_lo, q_hi], axis=1)
        u32_rec = all_pkt[:, h_half : h_half + lanes].astype(jnp.uint32) | (
            all_pkt[:, h_half + lanes : h_half + 2 * lanes].astype(jnp.uint32) << 16
        )
        xs_out_ref[...] = lax.bitcast_convert_type(u32_rec, jnp.float32)
    else:
        gathered_ref[...] = all_pkt

    for op in (*p1_ops, *s0_ops, *s1_ops, *ag0_ops, *ag1_ops, *p2_ops):
        op.wait_send()


def reduce_scatter_epilogue_oneshot_local(
    local_x: jax.Array,
    residual: jax.Array,
    *,
    post_norm_weight: jax.Array | None = None,
    scale: jax.Array | None = None,
    next_norm_weight: jax.Array | None = None,
    eps: float,
    axis_name: str = "x",
    order: tuple[int, ...] | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Single-shot reduce-scatter + `apply_residual_epilogue` + rank-ordered AG.

    `residual` holds this device's `[rows // 8, hidden]` rows. Returns
    `(new_residual_rows, gathered_rows, x_scale_all)`: when `next_norm_weight`
    is provided, `gathered_rows` is `uint8[rows, hidden]` FP8 rows and
    `x_scale_all` is `f32[rows, 1]` scales across all ranks in `order`;
    otherwise `gathered_rows` is `bf16[rows, hidden]` residual rows and
    `x_scale_all` is `None`.
    """
    rows, hidden = local_x.shape
    num_devices = _NUM_DEVICES if order is None else len(order)
    assert supports_oneshot_rs_ag(rows, hidden, num_devices), (
        f"unsupported shape {(rows, hidden)} over {num_devices} devices"
    )
    lanes = int(tpu_info().num_lanes)
    c_rows = rows // _NUM_DEVICES
    assert local_x.dtype == jnp.bfloat16 and residual.dtype == jnp.bfloat16, (
        f"expected bfloat16 local_x and residual, got {local_x.dtype} "
        f"and {residual.dtype}"
    )
    assert residual.shape == (c_rows, hidden), (
        f"expected residual {(c_rows, hidden)}, got {residual.shape}"
    )

    def row_vector(w: jax.Array | None) -> jax.Array | None:
        if w is None:
            return None
        return jnp.broadcast_to(w.astype(jnp.float32).reshape(1, -1), (1, hidden))

    operands = (
        local_x,
        residual,
        row_vector(post_norm_weight),
        row_vector(scale),
        row_vector(next_norm_weight),
    )
    has_next = next_norm_weight is not None

    dma = pltpu.SemaphoreType.DMA
    scratch = [
        *(
            pltpu.VMEM(shape, dtype)
            for shape, dtype in _vmem_scratch_shapes(
                c_rows, hidden, lanes, has_next=has_next
            )
        ),
        dma((_NUM_CHIPS,)),
        dma((_NUM_CHIPS,)),
        dma((2, 2)),
        dma((2, 2)),
        dma((2,)),
        dma((2,)),
        dma((2,)),
        dma((2,)),
        dma((2, 2)),
        dma((2, 2)),
        dma((_NUM_CHIPS,)),
        dma((_NUM_CHIPS,)),
    ]

    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    out_shapes: Any = (
        (
            jax.ShapeDtypeStruct((c_rows, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, hidden), jnp.uint8),
            jax.ShapeDtypeStruct((rows, lanes), jnp.float32),
        )
        if has_next
        else (
            jax.ShapeDtypeStruct((c_rows, hidden), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, hidden), jnp.bfloat16),
        )
    )
    call_out = pl.pallas_call(
        functools.partial(
            _kernel,
            axis_name=axis_name,
            c_rows=c_rows,
            hidden=hidden,
            lanes=lanes,
            order=tuple(range(_NUM_DEVICES)) if order is None else tuple(order),
            eps=eps,
        ),
        out_shape=out_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[vmem, vmem, *(None if w is None else vmem for w in operands[2:])],
            out_specs=(vmem, vmem, vmem) if has_next else (vmem, vmem),
            scratch_shapes=scratch,
        ),
        name=f"hier_rs_oneshot_r{rows}" + ("_fp8ag" if has_next else ""),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=epilogue_vmem_limit_bytes(),
            disable_bounds_checks=True,
            collective_id=0,
        ),
    )(*operands)

    if has_next:
        res_out, gathered_out, xs_wide = call_out
        return res_out, gathered_out, xs_wide[:, :1]
    res_out, gathered_out = call_out
    return res_out, gathered_out, None
