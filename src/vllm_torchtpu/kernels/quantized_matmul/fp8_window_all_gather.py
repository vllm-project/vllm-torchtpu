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

"""Column-window FP8 decode linear fused with the all-gather of the windows.

Under DCP query replication every rank needs every head's q, so each rank streams
the whole replicated q_b weight ([2048, 16384] fp8, 33.5 MB) at decode. Here each
rank streams only its own column window (its heads: 1/n of the bytes), computes it
with fp8_skinny's arithmetic, and pushes it into the same window of every other
rank's output -- one round of remote copies, whose barrier is signalled at entry and
hides under the weight stream:

    entry   start the window's weight tiles (fp8_skinny's 'n' ring); signal the
            barrier on every rank, self included; quantize x
    tile j  wait W_j; out[:, window tile j] = bf16(x_q . W_j * x_scale * w_scale)
    push    wait the barrier; copy out[:, window] into out[:, window] of every
            other rank
    wait    the n - 1 inbound windows (one byte-counted semaphore); drain the sends

Each output column is fp8_skinny's expression for that column (the same x_q, scale
and f32 dot over the whole K), so the gathered output is bitwise the replicated
kernel's.

Safety, as `small_collectives.common.barrier`: a rank writes into a peer only after
the barrier of this call, which the peer signals at entry, so the peer's output is
live; a peer leaves only after every inbound window landed and its own sends
drained, so no write of call k reaches a peer in call k + 1.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

MESH = pl.DeviceIdType.MESH
# Barrier semaphore group, distinct from the fused MoE kernels (0), the
# all-reduces (5), the reduce-scatter (7), the all-gather (9), the all-to-all (11)
# and the sparse MLA kernel's exchange (13).
COLLECTIVE_ID = 15
# Row slices a weight tile is copied in, all in flight at once: the window kernel's
# single 2 MB strided copy stalls behind DMA setup, where the replicated
# kernel's four tiles in flight stream 2.5 TB/s. The tile's dot still runs once
# over the whole K, so the arithmetic is unchanged.
_TILE_COPY_PARTS = 4
_LANES = 128
# Avoid fusing q_a_layernorm for m >= 8 where the separate fusion schedules
# better with larger collective/attention buckets.
QNORM_MAX_M = 4


def _chip_xor_steps(num_chips: int) -> tuple[int, ...]:
    half = num_chips // 2
    if num_chips >= 4:
        return (*range(half, num_chips), *range(1, half))
    return tuple(range(1, num_chips))


def _resolve_d2d_pair_first(
    m: int, num_cores: int, d2d_pair_first: bool | None
) -> bool:
    if d2d_pair_first is None:
        return m >= 8 and num_cores >= 16 and (num_cores & (num_cores - 1)) == 0
    if not d2d_pair_first:
        return False
    if num_cores < 4 or (num_cores & (num_cores - 1)) != 0:
        raise ValueError(
            f"d2d_pair_first requires a power-of-2 num_cores >= 4, got {num_cores}"
        )
    return True


def _window_tile_config(
    m: int, k: int, n_out: int, width: int, use_d2d: bool
) -> tuple[int, int, bool]:
    _, t, nbuf = fp8_skinny.TUNED[(k, n_out)]
    # Avoid full-window weight DMA stalling MXU at m >= 16: double-buffer 512 halves.
    if use_d2d and m >= 16 and width >= 1024 and t == width and (t // 2) % _LANES == 0:
        ws_in_hbm = n_out % 1024 == 0 and width % _LANES == 0
        return t // 2, 2, ws_in_hbm
    n_tiles = width // t
    return t, max(1, min(nbuf, n_tiles)), False


def supports(m: int, k: int, n: int, num_cores: int) -> bool:
    """Whether `fp8_skinny.window_supports` serves the rank windows of a `(k, n)`
    weight split over `num_cores` ranks, for `m <= fp8_skinny.MAX_M` rows."""
    return (
        0 < m <= fp8_skinny.MAX_M
        and num_cores >= 2
        and n % num_cores == 0
        and fp8_skinny.window_supports(k, n, n // num_cores)
    )


def supports_qnorm(m: int, k: int, n: int, num_cores: int) -> bool:
    return 0 < m <= QNORM_MAX_M and supports(m, k, n, num_cores)


def _kernel(
    rank_ref,
    *refs,
    n,
    axis,
    lane_axis,
    t,
    n_tiles,
    nbuf,
    q_dtype,
    use_d2d,
    ws_in_hbm,
    has_rmsnorm,
    eps,
):
    lane_ref, *refs = refs if lane_axis is not None else (None, *refs)
    if has_rmsnorm:
        x_ref, ws_ref, w_hbm, nw_ref, o_ref, xq_ref, xs_ref, wbuf, wsem, *tail = refs
    else:
        nw_ref = None
        x_ref, ws_ref, w_hbm, o_ref, xq_ref, xs_ref, wbuf, wsem, *tail = refs
    if ws_in_hbm:
        ws_vmem, ws_sem, *tail = tail
    else:
        ws_vmem = ws_sem = None
    if use_d2d:
        ici_send_sem, ici_recv_sem, d2d_send_sem, d2d_recv_sem = tail
        send_sem = recv_sem = None
    else:
        send_sem, recv_sem = tail
        ici_send_sem = ici_recv_sem = d2d_send_sem = d2d_recv_sem = None
    me = rank_ref[0]
    width = n_tiles * t
    col0 = pl.multiple_of(me * width, width)
    bsem = pltpu.get_barrier_semaphore()
    chip_steps = _chip_xor_steps(n // 2) if use_d2d else ()

    def peer(j):
        """Rank j on `axis`, in this core's lane of the mesh's other axis."""
        ids = {axis: jnp.int32(j)}
        if lane_ref is not None:
            ids[lane_axis] = lane_ref[0]
        return ids

    k = w_hbm.shape[0]
    parts = _TILE_COPY_PARTS if k % (_TILE_COPY_PARTS * 32) == 0 else 1
    rows = k // parts

    def tile(j):
        return w_hbm.at[:, pl.ds(col0 + j * t, t)], wbuf.at[j % nbuf], wsem.at[j % nbuf]

    def start(j):
        src, dst, sem = tile(j)
        for p in range(parts):
            part = pl.ds(p * rows, rows)
            pltpu.make_async_copy(src.at[part], dst.at[part], sem).start()

    def wait(j):
        # The parts signal one semaphore; wait for the whole tile's bytes.
        pltpu.make_async_copy(*tile(j)).wait()

    if ws_vmem is not None and ws_sem is not None:
        ws_copy = pltpu.make_async_copy(
            ws_ref.at[:, pl.ds(col0, width)], ws_vmem, ws_sem
        )
        ws_copy.start()
    else:
        ws_copy = None

    for j in range(min(nbuf, n_tiles)):
        start(j)
    if use_d2d:
        for step in range(n):
            pl.semaphore_signal(
                bsem, 1, device_id=peer(me ^ jnp.int32(step)), device_id_type=MESH
            )
    else:
        for j in range(n):
            pl.semaphore_signal(bsem, 1, device_id=peer(j), device_id_type=MESH)

    # Activation quant: fp8_skinny._kernel's formula (bit-identical x_q and scale).
    q_max = float(jnp.finfo(q_dtype).max)
    x = x_ref[...]
    if has_rmsnorm:
        assert nw_ref is not None
        xf = x.astype(jnp.float32)
        inv = jax.lax.rsqrt(
            jnp.mean(xf * xf, axis=-1, keepdims=True) + jnp.float32(eps)
        )
        # Avoid intermediate bf16 rounding before norm_w multiply: XLA TPU
        # keeps f32 excess precision across the RMSNorm weight multiply.
        x = ((xf * inv) * nw_ref[...].astype(jnp.float32)).astype(x.dtype)
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(
        jnp.float32
    ) * jnp.float32(1.0 / q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    x_q = jnp.clip(
        x.astype(jnp.float32) * (jnp.float32(1.0) / scale), -q_max, q_max
    ).astype(q_dtype)
    xq_ref[...] = x_q.astype(xq_ref.dtype)
    xs_ref[...] = scale

    dn = (((1,), (0,)), ((), ()))
    for j in range(n_tiles):
        wait(j)
        y = jax.lax.dot_general(
            xq_ref[...], wbuf[j % nbuf], dn, preferred_element_type=jnp.float32
        )
        if j == 0 and ws_copy is not None:
            ws_copy.wait()
        cols = pl.ds(col0 + j * t, t)
        ws_tile = (
            ws_vmem[:, pl.ds(j * t, t)] if ws_vmem is not None else ws_ref[:, cols]
        )
        o_ref[:, cols] = (y * xs_ref[...] * ws_tile).astype(o_ref.dtype)
        if j + nbuf < n_tiles:
            start(j + nbuf)

    mine = o_ref.at[:, pl.ds(col0, width)]
    pl.semaphore_wait(bsem, n)
    if use_d2d:
        # Avoid sending each core's window twice across inter-chip links at m >= 8:
        # exchange across same-side remote chips and forward to partner me ^ 1.
        my_chip = me >> 1
        my_side = me & 1
        partner = me ^ 1
        n_rem_chips = (n // 2) - 1
        pltpu.make_async_remote_copy(
            src_ref=mine,
            dst_ref=mine,
            send_sem=d2d_send_sem,
            recv_sem=d2d_recv_sem,
            device_id=peer(partner),
            device_id_type=MESH,
        ).start()
        for k_xor in chip_steps:
            rem_peer = ((my_chip ^ jnp.int32(k_xor)) << 1) | my_side
            pltpu.make_async_remote_copy(
                src_ref=mine,
                dst_ref=mine,
                send_sem=ici_send_sem,
                recv_sem=ici_recv_sem,
                device_id=peer(rem_peer),
                device_id_type=MESH,
            ).start()
        rem_inbound = o_ref.at[:, pl.ds(0, n_rem_chips * width)]
        pltpu.make_async_copy(rem_inbound, rem_inbound, ici_recv_sem).wait()
        for k_xor in chip_steps:
            rem_idx = ((my_chip ^ jnp.int32(k_xor)) << 1) | my_side
            rem_win = o_ref.at[:, pl.ds(pl.multiple_of(rem_idx * width, width), width)]
            pltpu.make_async_remote_copy(
                src_ref=rem_win,
                dst_ref=rem_win,
                send_sem=d2d_send_sem,
                recv_sem=d2d_recv_sem,
                device_id=peer(partner),
                device_id_type=MESH,
            ).start()
        d2d_inbound = o_ref.at[:, pl.ds(0, (n // 2) * width)]
        pltpu.make_async_copy(d2d_inbound, d2d_inbound, d2d_recv_sem).wait()
        pltpu.make_async_copy(rem_inbound, rem_inbound, ici_send_sem).wait()
        pltpu.make_async_copy(d2d_inbound, d2d_inbound, d2d_send_sem).wait()
        return

    for j in range(n):

        @pl.when(jnp.int32(j) != me)
        def _(j=j):
            pltpu.make_async_remote_copy(
                src_ref=mine,
                dst_ref=mine,
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=peer(j),
                device_id_type=MESH,
            ).start()

    peers = o_ref.at[:, pl.ds(0, (n - 1) * width)]  # the byte count of n - 1 windows
    pltpu.make_async_copy(peers, peers, recv_sem).wait()
    pltpu.make_async_copy(peers, peers, send_sem).wait()


@functools.cache
def _build(
    m,
    k,
    n_out,
    w_dtype,
    x_dtype,
    num_cores,
    axis,
    lane_axis,
    use_d2d,
    nw_dtype=None,
    eps=1e-6,
):
    width = n_out // num_cores
    t, nbuf, ws_in_hbm = _window_tile_config(m, k, n_out, width, use_d2d)
    n_tiles = width // t
    has_rmsnorm = nw_dtype is not None
    scratch: list[Any] = [
        pltpu.VMEM((m, k), jnp.bfloat16),  # x_q (fp8 values held as bf16)
        pltpu.VMEM((m, 1), jnp.float32),  # x scale
        pltpu.VMEM((nbuf, k, t), w_dtype),  # weight ring
        pltpu.SemaphoreType.DMA((nbuf,)),
    ]
    if ws_in_hbm:
        scratch.extend(
            [
                pltpu.VMEM((1, width), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ]
        )
    scratch.extend([pltpu.SemaphoreType.DMA] * (4 if use_d2d else 2))
    kern = functools.partial(
        _kernel,
        n=num_cores,
        axis=axis,
        lane_axis=lane_axis,
        t=t,
        n_tiles=n_tiles,
        nbuf=nbuf,
        q_dtype=w_dtype,
        use_d2d=use_d2d,
        ws_in_hbm=ws_in_hbm,
        has_rmsnorm=has_rmsnorm,
        eps=float(eps),
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [vmem, hbm if ws_in_hbm else vmem, hbm]
    if has_rmsnorm:
        in_specs.append(vmem)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1 if lane_axis is None else 2,
        in_specs=in_specs,
        out_specs=vmem,
        scratch_shapes=scratch,
    )
    suffix = ("_rn" if has_rmsnorm else "") + ("_d2d" if use_d2d else "")
    call = pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n_out), x_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=f"fp8_window_all_gather_m{m}_k{k}_n{n_out}_w{width}{suffix}",
    )
    return call, ws_in_hbm


@functools.partial(jax.jit, static_argnums=(0, 5))
def _call_constrained(call, scalars, x, w_s2, w_q, ws_in_hbm):
    # Weight (and HBM-windowed scales) stays in HBM, as fp8_skinny's hbm_pin.
    w_s_in = pltpu.with_memory_space_constraint(w_s2, pltpu.HBM) if ws_in_hbm else w_s2
    return call(*scalars, x, w_s_in, pltpu.with_memory_space_constraint(w_q, pltpu.HBM))


@functools.partial(jax.jit, static_argnums=(0, 6))
def _call_constrained_rmsnorm(call, scalars, x, w_s2, w_q, norm_w2, ws_in_hbm):
    w_s_in = pltpu.with_memory_space_constraint(w_s2, pltpu.HBM) if ws_in_hbm else w_s2
    return call(
        *scalars,
        x,
        w_s_in,
        pltpu.with_memory_space_constraint(w_q, pltpu.HBM),
        norm_w2,
    )


def fp8_window_all_gather(
    x,
    w_q,
    w_s,
    rank,
    *,
    num_cores,
    axis,
    lane_axis=None,
    lane=None,
    d2d_pair_first: bool | None = None,
    norm_w=None,
    eps: float = 1e-6,
):
    """Project column windows with `fp8_skinny` and all-gather them over `axis`.

    x bf16 [m <= 16, k] (the same on every rank); w_q fp8_e4m3fn [k, n] (K-major,
    replicated); w_s f32 [n]; rank int32 [1] (this core's index on the axis, as
    data) -> bf16 [m, n]. A mesh with a second axis names it `lane_axis` and
    passes this core's index on it as `lane` (int32 [1]): peers are this lane's
    ranks. If `norm_w` (`[k]`) is provided, applies RMSNorm(x, norm_w, eps) in
    registers under the weight-DMA shadow before FP8 activation quantization.
    """
    m, k = x.shape
    n_out = w_q.shape[1]
    if not supports(m, k, n_out, num_cores):
        raise ValueError(
            f"[{m}, {k}] x [{k}, {n_out}] windows over {num_cores} ranks: unsupported"
        )
    if (lane_axis is None) != (lane is None):
        raise ValueError("lane_axis and lane go together")
    if norm_w is not None and norm_w.shape != (k,):
        raise ValueError(f"norm_w must have shape ({k},), got {norm_w.shape}")
    use_d2d = _resolve_d2d_pair_first(m, num_cores, d2d_pair_first)
    call, ws_in_hbm = _build(
        m,
        k,
        n_out,
        w_q.dtype,
        x.dtype,
        num_cores,
        axis,
        lane_axis,
        use_d2d,
        None if norm_w is None else norm_w.dtype,
        float(eps),
    )
    w_s2 = w_s.reshape(1, n_out).astype(jnp.float32)
    scalars = (rank.astype(jnp.int32),)
    if lane is not None:
        scalars += (lane.astype(jnp.int32),)
    if norm_w is not None:
        return _call_constrained_rmsnorm(
            call, scalars, x, w_s2, w_q, norm_w.reshape(1, k), ws_in_hbm
        )
    return _call_constrained(call, scalars, x, w_s2, w_q, ws_in_hbm)
