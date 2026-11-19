# Copyright 2026 The vLLM-TorchTPU Authors
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

With `w_uk` (MLA query absorption) each rank also multiplies its own heads' q_nope
by their W_UK rows and gathers that latent q beside the raw one, so no rank reads
the other heads' W_UK: per head `ql = bf16(q_nope . W_UK)` with an f32 dot, the
expression the replicated absorption evaluates on the gathered q.

Safety, as `small_collectives.common.barrier`: a rank writes into a peer only after
the barrier of this call, which the peer signals at entry, so the peer's output is
live; a peer leaves only after every inbound window landed and its own sends
drained, so no write of call k reaches a peer in call k + 1.
"""

from __future__ import annotations

import functools

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
# single 2 MB strided copy took 3.6 us in situ (j-6482bc6e), where the replicated
# kernel's four tiles in flight stream 2.5 TB/s. The tile's dot still runs once
# over the whole K, so the arithmetic is unchanged.
_TILE_COPY_PARTS = 4


# Rows up to which the absorbing variant serves: its gather carries the latent q
# (kv_lora_rank per head instead of the nope dim), which costs more at more rows than
# the replicated absorption's W_UK read it removes.
ABSORB_MAX_M = 8


def supports(m: int, k: int, n: int, num_cores: int) -> bool:
    """Whether `fp8_skinny.window_supports` serves the rank windows of a `(k, n)`
    weight split over `num_cores` ranks, for `m <= fp8_skinny.MAX_M` rows."""
    return (
        m <= fp8_skinny.MAX_M
        and num_cores >= 2
        and n % num_cores == 0
        and fp8_skinny.window_supports(k, n, n // num_cores)
    )


def supports_absorb(
    m: int, k: int, n: int, num_cores: int, heads: int, nope: int, rank: int
) -> bool:
    """Whether the absorbing variant serves `w_uk` bf16 `[heads, nope, rank]` on
    the `(k, n)` windows: each window tile holds whole heads."""
    if not (m <= ABSORB_MAX_M and supports(m, k, n, num_cores)) or n % heads:
        return False
    head_dim = n // heads
    tile = fp8_skinny.TUNED[(k, n)][1]
    return tile % head_dim == 0 and nope <= head_dim and rank % fp8_skinny.LANE == 0


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
    head_dim,
):
    lane_ref, *refs = refs if lane_axis is not None else (None, *refs)
    absorb = head_dim is not None
    if absorb:
        x_ref, ws_ref, w_hbm, uk_hbm, o_ref, ql_ref, *refs = refs
        xq_ref, xs_ref, wbuf, wsem, send_sem, recv_sem, uk_buf, uk_sem, *ql_sems = refs
    else:
        x_ref, ws_ref, w_hbm, o_ref, xq_ref, xs_ref, wbuf, wsem, send_sem, recv_sem = refs
    me = rank_ref[0]
    width = n_tiles * t
    col0 = pl.multiple_of(me * width, width)
    bsem = pltpu.get_barrier_semaphore()
    if absorb:
        local_heads, nope, rank = uk_buf.shape
        ql_width = local_heads * rank
        ql0 = pl.multiple_of(me * ql_width, ql_width)
        uk_copy = pltpu.make_async_copy(
            uk_hbm.at[pl.ds(me * local_heads, local_heads)], uk_buf, uk_sem
        )

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

    for j in range(min(nbuf, n_tiles)):
        start(j)
    if absorb:
        uk_copy.start()
    for j in range(n):
        pl.semaphore_signal(bsem, 1, device_id=peer(j), device_id_type=MESH)

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

    dn = (((1,), (0,)), ((), ()))
    for j in range(n_tiles):
        wait(j)
        y = jax.lax.dot_general(
            xq_ref[...], wbuf[j % nbuf], dn, preferred_element_type=jnp.float32
        )
        cols = pl.ds(col0 + j * t, t)
        q = (y * xs_ref[...] * ws_ref[:, cols]).astype(o_ref.dtype)
        o_ref[:, cols] = q
        if absorb:
            if j == 0:
                uk_copy.wait()
            for h in range(t // head_dim):
                head = j * (t // head_dim) + h
                q_nope = q[:, h * head_dim : h * head_dim + nope]
                ql = jnp.dot(q_nope, uk_buf[head], preferred_element_type=jnp.float32)
                ql_ref[:, pl.ds(ql0 + head * rank, rank)] = ql.astype(ql_ref.dtype)
        if j + nbuf < n_tiles:
            start(j + nbuf)

    gathers = [(o_ref, col0, width, send_sem, recv_sem)]
    if absorb:
        gathers.append((ql_ref, ql0, ql_width, *ql_sems))
    pl.semaphore_wait(bsem, n)
    for out, c0, w, s_sem, r_sem in gathers:
        mine = out.at[:, pl.ds(c0, w)]
        for j in range(n):

            @pl.when(jnp.int32(j) != me)
            def _(j=j, mine=mine, s_sem=s_sem, r_sem=r_sem):
                pltpu.make_async_remote_copy(
                    src_ref=mine,
                    dst_ref=mine,
                    send_sem=s_sem,
                    recv_sem=r_sem,
                    device_id=peer(j),
                    device_id_type=MESH,
                ).start()

    for out, _, w, s_sem, r_sem in gathers:
        peers = out.at[:, pl.ds(0, (n - 1) * w)]  # the byte count of n - 1 windows
        pltpu.make_async_copy(peers, peers, r_sem).wait()
        pltpu.make_async_copy(peers, peers, s_sem).wait()


@functools.cache
def _build(m, k, n_out, w_dtype, x_dtype, num_cores, axis, lane_axis, uk_shape=None):
    _, t, nbuf = fp8_skinny.TUNED[(k, n_out)]
    width = n_out // num_cores
    n_tiles = width // t
    nbuf = max(1, min(nbuf, n_tiles))
    scratch = [
        pltpu.VMEM((m, k), jnp.bfloat16),  # x_q (fp8 values held as bf16)
        pltpu.VMEM((m, 1), jnp.float32),  # x scale
        pltpu.VMEM((nbuf, k, t), w_dtype),  # weight ring
        pltpu.SemaphoreType.DMA((nbuf,)),
        pltpu.SemaphoreType.DMA,  # window sends
        pltpu.SemaphoreType.DMA,  # window receives
    ]
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [vmem, vmem, hbm]
    out_shape = jax.ShapeDtypeStruct((m, n_out), x_dtype)
    out_specs = vmem
    head_dim = None
    name = f"fp8_window_all_gather_m{m}_k{k}_n{n_out}_w{width}"
    if uk_shape is not None:
        heads, nope, rank = uk_shape
        head_dim = n_out // heads
        local_heads = width // head_dim
        in_specs.append(hbm)  # W_UK: only this rank's heads are read
        out_shape = (out_shape, jax.ShapeDtypeStruct((m, heads * rank), x_dtype))
        out_specs = (vmem, vmem)
        scratch += [
            pltpu.VMEM((local_heads, nope, rank), jnp.bfloat16),
            pltpu.SemaphoreType.DMA,  # W_UK rows
            pltpu.SemaphoreType.DMA,  # latent window sends
            pltpu.SemaphoreType.DMA,  # latent window receives
        ]
        name += f"_uk{rank}"
    kern = functools.partial(
        _kernel,
        n=num_cores,
        axis=axis,
        lane_axis=lane_axis,
        t=t,
        n_tiles=n_tiles,
        nbuf=nbuf,
        q_dtype=w_dtype,
        head_dim=head_dim,
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1 if lane_axis is None else 2,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=scratch,
    )
    return pl.pallas_call(
        kern,
        out_shape=out_shape,
        grid_spec=grid_spec,
        compiler_params=pltpu.CompilerParams(collective_id=COLLECTIVE_ID),
        name=name,
    )


@functools.partial(jax.jit, static_argnums=(0,))
def _call_constrained(call, scalars, x, w_s2, w_q, *w_uk):
    # The weights stay in HBM (the kernel streams only its window), as
    # fp8_skinny's hbm_pin.
    hbm = [pltpu.with_memory_space_constraint(w, pltpu.HBM) for w in (w_q, *w_uk)]
    return call(*scalars, x, w_s2, *hbm)


def fp8_window_all_gather(
    x, w_q, w_s, rank, *, num_cores, axis, lane_axis=None, lane=None, w_uk=None
):
    """`fp8_skinny(x, w_q, w_s)` on every rank of mesh axis `axis`, each rank
    computing only its column window `rank` and all-gathering the rest, inside a
    shard_map over that axis.

    x bf16 [m <= 16, k] (the same on every rank); w_q fp8_e4m3fn [k, n] (K-major,
    replicated); w_s f32 [n]; rank int32 [1] (this core's index on the axis, as
    data) -> bf16 [m, n]. A mesh with a second axis names it `lane_axis` and
    passes this core's index on it as `lane` (int32 [1]): peers are this lane's
    ranks.

    With `w_uk` bf16 [heads, nope, rank] (replicated; head h owns output columns
    [h * n / heads, (h + 1) * n / heads), q_nope its first `nope`), also returns
    the absorbed q, bf16 [m, heads * rank]: head h's columns are
    `bf16(q_nope_h . w_uk[h])`, f32 accumulation (see `supports_absorb`).
    """
    m, k = x.shape
    n_out = w_q.shape[1]
    if not supports(m, k, n_out, num_cores):
        raise ValueError(
            f"[{m}, {k}] x [{k}, {n_out}] windows over {num_cores} ranks: unsupported"
        )
    if (lane_axis is None) != (lane is None):
        raise ValueError("lane_axis and lane go together")
    uk = ()
    if w_uk is not None:
        if w_uk.dtype != jnp.bfloat16 or not supports_absorb(
            m, k, n_out, num_cores, *w_uk.shape
        ):
            raise ValueError(
                f"absorbing {w_uk.dtype}{list(w_uk.shape)} into [{m}, {k}] x "
                f"[{k}, {n_out}] windows over {num_cores} ranks: unsupported"
            )
        uk = (w_uk,)
    call = _build(
        m,
        k,
        n_out,
        w_q.dtype,
        x.dtype,
        num_cores,
        axis,
        lane_axis,
        None if w_uk is None else tuple(w_uk.shape),
    )
    w_s2 = w_s.reshape(1, n_out).astype(jnp.float32)
    scalars = (rank.astype(jnp.int32),)
    if lane is not None:
        scalars += (lane.astype(jnp.int32),)
    return _call_constrained(call, scalars, x, w_s2, w_q, *uk)
