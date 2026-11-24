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
"""FP8 matmul fused with the all-reduce of its output over a mesh axis, as one
Pallas kernel, for the few rows of a decode step.

A row-parallel linear layer splits its input features over the n cores of a mesh
axis. Each core multiplies its [m, k] slice of the activation by its [k, n_out]
slice of the weight, and an all-reduce then sums the n partial [m, n_out] outputs.
At decode, m is small, so the matmul is a stream of the weight from HBM, and the
all-reduce is mostly fixed synchronization cost. Run as two calls, the
all-reduce's first barrier cannot start before the matmul call has ended. This
kernel signals the barrier at entry, so the barrier round runs while the weight
streams in, and it does the exchange on the finished output tiles in the same
call:

    entry   start the weight-scale DMA and the first nbuf weight-tile DMAs;
            signal the barrier on every core, self included; quantize x per row
    tile j  wait for weight tile j; y = x_q . W_j in float32;
            out[:, tile j] = bf16(y * x_scale * w_scale[tile j]); start tile
            j + nbuf. With ``waves``: after tile 0, wait for the barrier; after
            each tile, send every owner chunk (below) whose last column is done
    RS      reduce-scatter. Core t owns the output columns
            [t * own, (t + 1) * own), own = n_out / n. Without ``waves``, after
            the last tile: wait for the barrier and send chunk t to slot ``me``
            of core t, for every t != me, and copy chunk ``me`` to my own slot
    sum     wait for the n - 1 inbound chunks; sum the n slots in float32 in
            rank order; round once to bf16
    AG      all-gather. Send the sum into chunk ``me`` of the output of every
            other core and copy it into my own; wait for the n - 1 inbound sums

Numerics: the partial output of each core is bit for bit
``fp8_skinny.fp8_skinny_matmul`` with the same tile (the same per-row activation
quantization, a bf16 x fp8 MXU dot with exact float32 accumulation, and one
rounding to bf16). The result is the float32 sum of those bf16 partials in rank
order, rounded once to bf16, and every core gets the same result.

Example: on 8 cores, x bf16 [4, 2048], w_q fp8 [2048, 4096] and w_s f32 [4096]
on each core give each core the same bf16 [4, 4096] block, the sum over the 8
cores of fp8_skinny_matmul(x, w_q, w_s). Each core owns 512 output columns. The
weight streams in 4 tiles of 1024 columns, and the owners of columns 0-1023
(cores 0 and 1) get their chunks after tile 0.

Safety: a core writes into a peer only after the barrier of this call, which the
peer signals at entry, that is, after the peer has finished the previous call. A
peer's out[:, chunk me] is overwritten by my all-gather only after I received
that chunk from the peer, so its reduce-scatter read is complete.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MESH = pl.DeviceIdType.MESH
# Barrier semaphore group. Every kernel compiled with the same id in one program
# shares its barrier semaphore. Id 5 is shared with the small all-reduce kernel
# (kernels/experimental/small_collectives/all_reduce.py), and both kernels can
# run in one decode program. The sharing is safe because each of the two kernels
# signals every peer once at entry, waits for exactly that many signals, and
# leaves a call only after every peer has passed the barrier of the same call. So
# the signals of one call cannot release the barrier of another call. The fused
# MoE kernels use id 0 and the DeepSeek-V4 o-projection fetch uses id 3.
COLLECTIVE_ID = 5
LANES = 128
# The largest number of activation rows the kernel takes (decode blocks).
MAX_M = 16

# Per-(k, n_out) weight-ring setting: (tile, nbuf), where the weight streams in
# column tiles W[:, j * tile:(j + 1) * tile] with nbuf tile copies in flight.
# Shapes not in the table take `pick_config`'s default. The (1024, 6144) entry is
# fp8_skinny's measured setting for that shape on TPU v7x.
TUNED = {(1024, 6144): (2048, 3)}
# The default ring: 3 tiles in flight, the widest lane-multiple tile that divides
# n_out into at least 3 tiles and keeps the ring within RING_BYTES of VMEM.
DEFAULT_NBUF = 3
# The kernel passes no vmem_limit_bytes, so all of its VMEM must fit XLA's
# default scoped VMEM limit (--xla_tpu_scoped_vmem_limit_kib, 32 MiB by default).
# The ring takes at most RING_BYTES. At m = 16 the other buffers (x, x_q, the
# output, the receive slots, and the weight scale padded to 8 sublanes) take
# about 64 * k + 96 * n_out bytes: about 0.7 MiB at (k, n_out) = (1024, 6144),
# where the ring takes 6 MiB. So the total stays below 32 MiB for every n_out up
# to about 200,000 columns.
RING_BYTES = 8 * 1024 * 1024


def pick_config(k: int, n_out: int) -> tuple[int, int] | None:
    """The weight-ring setting for a [k, n_out] FP8 weight slice.

    Args:
      k: Input features per core.
      n_out: Output features.

    Returns:
      ``(tile, nbuf)``: ``TUNED[(k, n_out)]`` when present, else the widest tile
      of 2048, 1024, 512, 256 or 128 columns that divides ``n_out`` into at
      least ``DEFAULT_NBUF`` tiles with ``DEFAULT_NBUF * k * tile`` bytes at most
      ``RING_BYTES``, with ``nbuf = DEFAULT_NBUF``. None when no tile qualifies.
      Example: (2048, 4096) -> (1024, 3), because a 2048-column tile gives
      only 2 tiles.
    """
    if (k, n_out) in TUNED:
        return TUNED[(k, n_out)]
    for tile in (2048, 1024, 512, 256, 128):
        if (
            n_out % tile == 0
            and n_out // tile >= DEFAULT_NBUF
            and DEFAULT_NBUF * k * tile <= RING_BYTES
        ):
            return tile, DEFAULT_NBUF
    return None


def supports(m: int, k: int, n_out: int, num_cores: int) -> bool:
    """Whether the kernel takes a [m, k] x [k, n_out] block on each of num_cores
    cores.

    The conditions: m <= MAX_M; at least 2 cores; k a multiple of 128; n_out
    divides into num_cores owner chunks of whole 128-lane groups; and
    `pick_config` has a setting for (k, n_out). Example: (4, 1024, 6144, 16) is
    supported (owner chunks of 384 columns); (4, 1024, 6144, 32) is not
    (192 columns).
    """
    return (
        m <= MAX_M
        and num_cores >= 2
        and k % LANES == 0
        and n_out % num_cores == 0
        and (n_out // num_cores) % LANES == 0
        and pick_config(k, n_out) is not None
    )


def owner_waves(num_cores: int, n_out: int, tile: int) -> list[list[int]]:
    """For each weight tile, the owner chunks whose last column is in that tile.

    Example: owner_waves(8, 4096, 1024) == [[0, 1], [2, 3], [4, 5], [6, 7]].
    """
    own = n_out // num_cores
    waves: list[list[int]] = [[] for _ in range(n_out // tile)]
    for j in range(num_cores):
        waves[((j + 1) * own - 1) // tile].append(j)
    return waves


def _kernel(
    idx_ref,
    rank_ref,
    x_ref,
    ws_hbm,
    w_hbm,
    o_ref,
    xq_ref,
    xs_ref,
    wbuf,
    wsem,
    ws_vmem,
    ws_sem,
    recv,
    red,
    rs_send,
    rs_recv,
    ag_send,
    ag_recv,
    loc,
    *,
    n,
    axis,
    t,
    n_tiles,
    nbuf,
    q_dtype,
    waves,
):
    s = idx_ref[0]
    me = rank_ref[0]
    own = o_ref.shape[1] // n
    bsem = pltpu.get_barrier_semaphore()

    ws_copy = pltpu.make_async_copy(ws_hbm, ws_vmem, ws_sem)
    ws_copy.start()

    def copy(j):
        return pltpu.make_async_copy(
            w_hbm.at[s, :, pl.ds(j * t, t)], wbuf.at[j % nbuf], wsem.at[j % nbuf]
        )

    for j in range(min(nbuf, n_tiles)):
        copy(j).start()
    for j in range(n):
        pl.semaphore_signal(
            bsem, 1, device_id={axis: jnp.int32(j)}, device_id_type=MESH
        )

    # Activation quant: fp8_skinny._kernel's formula (bit-identical x_q and scale).
    q_max = float(jnp.finfo(q_dtype).max)
    x = x_ref[...]
    scale = jnp.max(jnp.abs(x), axis=1, keepdims=True).astype(
        jnp.float32
    ) * jnp.float32(1.0 / q_max)
    scale = jnp.where(scale == 0, jnp.float32(1.0), scale)
    x_q = jnp.clip(
        x.astype(jnp.float32) * (jnp.float32(1.0) / scale), -q_max, q_max
    ).astype(q_dtype)
    xq_ref[...] = x_q.astype(xq_ref.dtype)
    xs_ref[...] = scale

    def chunk(j):
        return o_ref.at[:, pl.ds(j * own, own)]

    def push_owner(j):
        @pl.when(jnp.int32(j) != me)
        def _():
            pltpu.make_async_remote_copy(
                src_ref=chunk(j),
                dst_ref=recv.at[me],
                send_sem=rs_send,
                recv_sem=rs_recv,
                device_id={axis: jnp.int32(j)},
                device_id_type=MESH,
            ).start()

        @pl.when(jnp.int32(j) == me)
        def _():
            pltpu.make_async_copy(chunk(j), recv.at[j], loc.at[0]).start()

    wave_of = owner_waves(n, o_ref.shape[1], t) if waves else None
    dn = (((1,), (0,)), ((), ()))
    for j in range(n_tiles):
        copy(j).wait()
        y = jax.lax.dot_general(
            xq_ref[...], wbuf[j % nbuf], dn, preferred_element_type=jnp.float32
        )
        if j == 0:
            ws_copy.wait()
        cols = slice(j * t, (j + 1) * t)
        o_ref[:, cols] = (y * xs_ref[...] * ws_vmem[:, cols]).astype(o_ref.dtype)
        if j + nbuf < n_tiles:
            copy(j + nbuf).start()
        if wave_of is not None:
            if j == 0:
                pl.semaphore_wait(bsem, n)
            for o in wave_of[j]:
                push_owner(o)
    if wave_of is None:
        pl.semaphore_wait(bsem, n)
        for o in range(n):
            push_owner(o)

    slots = recv.at[pl.ds(0, n - 1)]  # the byte count of n - 1 chunks
    pltpu.make_async_copy(recv.at[0], recv.at[0], loc.at[0]).wait()
    pltpu.make_async_copy(slots, slots, rs_recv).wait()
    acc = recv[0].astype(jnp.float32)
    for r in range(1, n):
        acc = acc + recv[r].astype(jnp.float32)
    red[...] = acc.astype(red.dtype)

    dst_me = o_ref.at[:, pl.ds(pl.multiple_of(me * own, LANES), own)]
    for j in range(n):

        @pl.when(jnp.int32(j) != me)
        def _(j=j):
            pltpu.make_async_remote_copy(
                src_ref=red,
                dst_ref=dst_me,
                send_sem=ag_send,
                recv_sem=ag_recv,
                device_id={axis: jnp.int32(j)},
                device_id_type=MESH,
            ).start()

    own_ag = pltpu.make_async_copy(red, dst_me, loc.at[1])
    own_ag.start()
    own_ag.wait()
    pltpu.make_async_copy(slots, slots, ag_recv).wait()
    pltpu.make_async_copy(slots, slots, rs_send).wait()
    pltpu.make_async_copy(slots, slots, ag_send).wait()


@functools.cache
def _build(
    m, k, n_out, n_stack, w_dtype, x_dtype, num_cores, axis, waves, collective_id
):
    t, nbuf = pick_config(k, n_out)
    n_tiles = n_out // t
    nbuf = max(1, min(nbuf, n_tiles))
    own = n_out // num_cores
    scratch = [
        pltpu.VMEM((m, k), jnp.bfloat16),  # x_q (fp8 values held as bf16)
        pltpu.VMEM((m, 1), jnp.float32),  # x scale
        pltpu.VMEM((nbuf, k, t), w_dtype),  # weight ring
        pltpu.SemaphoreType.DMA((nbuf,)),
        pltpu.VMEM((1, n_out), jnp.float32),  # weight scale
        pltpu.SemaphoreType.DMA,
        pltpu.VMEM((num_cores, m, own), x_dtype),  # receive slots, by sender rank
        pltpu.VMEM((m, own), x_dtype),  # my summed chunk
        pltpu.SemaphoreType.DMA,  # reduce-scatter send
        pltpu.SemaphoreType.DMA,  # reduce-scatter receive
        pltpu.SemaphoreType.DMA,  # all-gather send
        pltpu.SemaphoreType.DMA,  # all-gather receive
        pltpu.SemaphoreType.DMA((2,)),  # local copies
    ]
    kern = functools.partial(
        _kernel,
        n=num_cores,
        axis=axis,
        t=t,
        n_tiles=n_tiles,
        nbuf=nbuf,
        q_dtype=w_dtype,
        waves=waves,
    )
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=2,
        in_specs=[
            vmem,
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
        out_specs=vmem,
        scratch_shapes=scratch,
    )
    return pl.pallas_call(
        kern,
        out_shape=jax.ShapeDtypeStruct((m, n_out), x_dtype),
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name=f"fp8_matmul_all_reduce{'_w' if waves else ''}_m{m}_k{k}_n{n_out}",
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _call_constrained(call, idx, rank, x, w_s2, w_stack):
    # The weight scale and the weight stay in HBM; the kernel copies them in with
    # its own DMAs.
    return call(
        idx,
        rank,
        x,
        pltpu.with_memory_space_constraint(w_s2, pltpu.HBM),
        pltpu.with_memory_space_constraint(w_stack, pltpu.HBM),
    )


def fp8_matmul_all_reduce_stacked(
    x,
    w_stack,
    idx,
    w_s,
    rank=None,
    *,
    num_cores,
    axis,
    waves=False,
    collective_id=COLLECTIVE_ID,
):
    """`fp8_matmul_all_reduce` on the weight ``w_stack[idx[0]]`` of a stack.

    The kernel reads only the selected [k, n_out] weight from HBM. Use the stack to
    time the kernel with a weight that is not in any cache from the previous call.

    Args:
      x: bf16 [m, k], this core's activation slice.
      w_stack: float8_e4m3fn [L, k, n_out], L weights in K-major layout.
      idx: int32 [1], the index of the weight in ``w_stack``.
      w_s: float32 [n_out], the per-output-channel weight scale.
      rank: As in `fp8_matmul_all_reduce`.
      num_cores: As in `fp8_matmul_all_reduce`.
      axis: As in `fp8_matmul_all_reduce`.
      waves: As in `fp8_matmul_all_reduce`.
      collective_id: As in `fp8_matmul_all_reduce`.

    Returns:
      bf16 [m, n_out], as `fp8_matmul_all_reduce`.

    Raises:
      ValueError: ``x`` is not bf16, the weight is not float8_e4m3fn, the
        weight's k differs from the k of ``x``, ``w_s`` is not of shape
        [n_out], `supports` is False for the shapes, or ``num_cores`` differs
        from the size of ``axis``.
    """
    m, k = x.shape
    n_stack, w_k, n_out = w_stack.shape
    if x.dtype != jnp.bfloat16:
        raise ValueError(f"x must be bfloat16, got {x.dtype}")
    if w_stack.dtype != jnp.float8_e4m3fn:
        raise ValueError(f"the weight must be float8_e4m3fn, got {w_stack.dtype}")
    if w_k != k:
        raise ValueError(f"the weight has k={w_k}, but x [{m}, {k}] has k={k}")
    if w_s.shape != (n_out,):
        raise ValueError(f"w_s must have shape ({n_out},), got {w_s.shape}")
    if not supports(m, k, n_out, num_cores):
        raise ValueError(
            f"[{m}, {k}] x [{k}, {n_out}] over {num_cores} cores: unsupported. "
            f"The kernel needs m <= {MAX_M}, at least 2 cores, k a multiple of "
            f"{LANES}, n_out / num_cores a multiple of {LANES}, and a weight-ring "
            "setting from pick_config(k, n_out)"
        )
    # Static inside jax.shard_map: lax.axis_size of a named axis is a Python value.
    axis_cores = int(jax.lax.axis_size(axis))
    if axis_cores != num_cores:
        # A smaller num_cores leaves the cores above it waiting for a barrier
        # signal that never comes.
        raise ValueError(
            f"num_cores={num_cores}, but mesh axis {axis!r} has {axis_cores} cores"
        )
    if rank is None:
        rank = jax.lax.axis_index(axis).reshape(1)
    call = _build(
        m,
        k,
        n_out,
        n_stack,
        w_stack.dtype,
        x.dtype,
        num_cores,
        axis,
        waves,
        collective_id,
    )
    w_s2 = w_s.reshape(1, n_out).astype(jnp.float32)
    return _call_constrained(
        call, idx.astype(jnp.int32), rank.astype(jnp.int32), x, w_s2, w_stack
    )


def fp8_matmul_all_reduce(
    x,
    w_q,
    w_s,
    rank=None,
    *,
    num_cores,
    axis,
    waves=False,
    collective_id=COLLECTIVE_ID,
):
    """The sum over the cores of mesh axis ``axis`` of each core's FP8 matmul.

    Call it inside a ``jax.shard_map`` over ``axis``. Each core computes
    ``fp8_skinny_matmul(x, w_q, w_s)``: x quantized to FP8 per row, times w_q,
    scaled by the row scale and ``w_s``. Every core gets the sum of these
    partial outputs over the ``num_cores`` cores (see the module docstring for
    the exact numerics). Example: on 8 cores, x bf16 [4, 2048], w_q
    float8_e4m3fn [2048, 4096] and w_s f32 [4096] on each core give bf16
    [4, 4096] on each core.

    Args:
      x: bf16 [m, k], this core's activation slice; m <= MAX_M.
      w_q: float8_e4m3fn [k, n_out], this core's weight slice, K-major.
      w_s: float32 [n_out], the per-output-channel weight scale.
      rank: int32 [1], this core's index on ``axis``, passed as data. Use it
        where ``lax.axis_index`` cannot lower (the torch bridge's export path).
        None reads the index from the axis.
      num_cores: The number of cores on ``axis``.
      axis: The name of the mesh axis to reduce over.
      waves: Send each owner's reduce-scatter chunk as soon as the weight tile
        that holds its last column is done, instead of after the last tile.
        The result is the same.
      collective_id: The barrier semaphore group. Kernels compiled into one
        program with the same id share their barrier semaphore.

    Returns:
      bf16 [m, n_out], the same on every core.

    Raises:
      ValueError: As in `fp8_matmul_all_reduce_stacked`.
    """
    return fp8_matmul_all_reduce_stacked(
        x,
        w_q[None],
        jnp.zeros((1,), jnp.int32),
        w_s,
        rank,
        num_cores=num_cores,
        axis=axis,
        waves=waves,
        collective_id=collective_id,
    )
