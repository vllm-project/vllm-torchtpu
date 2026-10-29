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
"""SparseCore pre-push of the push-dispatch fused EP MoE.

Before the TensorCore MoE kernel starts, a two-core SparseCore kernel pushes
this shard's routed rows of the first destination visit positions -- the
(destination, expert) segments of host.push_dispatch_segments -- straight
into every destination's expert slab (this kernel's output, which the
TensorCore kernel then takes as its aliased dispatch slab), and waits for
every row that lands on this shard. The TensorCore kernel can therefore
compute its first experts with no dispatch wait.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

# This shard's EP rank (its index on the mesh axis) reaches every SparseCore
# kernel as DATA -- an i32[8] HBM operand copied into SMEM -- never from
# `lax.axis_index(axis)`: that lowers to `partition-id`, which the SPMD
# partitioner rejects ("partitionId instruction is not supported for SPMD
# partitioning") when torch_tpu recompiles the module through its jax.export
# bridge. Same scheme as the TensorCore kernel's rank operand.
# `lax.axis_index("core")` names the SparseCore kernel mesh, not an SPMD
# partition, and stays as it is.
RANK_WORDS = 8
# The collective the SparseCore kernels' barrier semaphore belongs to: the
# TensorCore MoE kernel uses 0 and the TensorCore routing exchange 2.
SC_COLLECTIVE_ID = 1


def rank_operand(rank):
    """i32[RANK_WORDS] operand carrying the EP rank (scalar or [1, ...])."""
    r = jnp.asarray(rank).astype(jnp.int32).reshape(-1)[:1]
    return jnp.broadcast_to(r, (RANK_WORDS,))


def _sc_barrier(me, ep):
    """All-to-all barrier on the collective's barrier semaphore, signalling
    peers in rank-staggered order (me, me + 1, ...): every peer must have
    entered (allocated its output slab) before any remote row lands."""
    bsem = pltpu.get_barrier_semaphore()

    def _sig_b(i, c):
        p = lax.rem(i + me, ep)
        pl.semaphore_signal(
            bsem, 1, device_id=(p,), device_id_type=pl.DeviceIdType.MESH
        )
        return c

    lax.fori_loop(0, ep, _sig_b, jnp.int32(0))
    pl.semaphore_wait(bsem, ep)


def sc_dispatch_chunk_2core(
    q_sorted, seg, counts4, *, ep, g_local, v0, v1, slab_rows, name, rank
):
    """Push visit positions [v0, v1) of `seg` into a fresh [slab_rows, ...]
    slab on every destination, on a 2-core ScalarSubcoreMesh: core c pushes
    the positions v with (v & 1) == c across all destinations and waits on
    the rows arriving on core c. counts4 = [rows landing on core 0, on core
    1, remote rows core 0 sends, core 1 sends], i32[4].
    """
    n_segs = ep * g_local
    lanes, width = q_sorted.shape[1], q_sorted.shape[2]
    mesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=2)

    def kernel(
        q_hbm, seg_hbm, cnt_hbm, rank_hbm, out_hbm,
        seg_sm, cnt_sm, cp_sem, send_sem, recv_sem, bar, rank_sm,
    ):  # fmt: skip
        del bar  # the persistent barrier semaphore is used instead
        pltpu.sync_copy(rank_hbm, rank_sm)
        me = rank_sm[0]
        core_id = lax.axis_index("core")
        c1 = pltpu.make_async_copy(seg_hbm, seg_sm, cp_sem)
        c2 = pltpu.make_async_copy(cnt_hbm, cnt_sm, cp_sem)
        c1.start()
        c2.start()
        _sc_barrier(me, ep)
        c1.wait()
        c2.wait()

        v_first = v0 + jnp.bitwise_and(core_id - (v0 & 1), 1)
        n_v = jnp.maximum((v1 - v_first + 1) >> 1, 0)

        def one_v(iv, c):
            v = v_first + iv * 2
            base_p = v * ep

            def one_d(d, c_inner):
                # Destinations staggered by rank, like the TensorCore pushes.
                p = base_p + lax.rem(d + me + 1, ep)
                a = seg_sm[p]
                n = jnp.bitwise_and(a, 65535)

                @pl.when(n > 0)
                def _():
                    dest = jnp.right_shift(a, 24)
                    src = q_hbm.at[pl.ds(seg_sm[n_segs + p], n)]
                    dst = out_hbm.at[pl.ds(seg_sm[2 * n_segs + p], n)]

                    @pl.when(dest == me)
                    def _():
                        pltpu.make_async_copy(src, dst, recv_sem).start()

                    @pl.when(dest != me)
                    def _():
                        pltpu.make_async_remote_copy(
                            src_ref=src,
                            dst_ref=dst,
                            send_sem=send_sem,
                            recv_sem=recv_sem,
                            device_id=(dest,),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()

                return c_inner

            return lax.fori_loop(0, ep, one_d, c)

        lax.fori_loop(0, n_v, one_v, jnp.int32(0))
        n_in = cnt_sm[core_id]
        n_out = cnt_sm[2 + core_id]

        @pl.when(n_in > 0)
        def _():
            pltpu.make_async_copy(
                out_hbm.at[pl.ds(0, n_in)], out_hbm.at[pl.ds(0, n_in)], recv_sem
            ).wait()

        @pl.when(n_out > 0)
        def _():
            pltpu.make_async_copy(
                q_hbm.at[pl.ds(0, n_out)], q_hbm.at[pl.ds(0, n_out)], send_sem
            ).wait()

    return pl.kernel(
        kernel,
        out_type=jax.ShapeDtypeStruct((slab_rows, lanes, width), q_sorted.dtype),
        mesh=mesh,
        scratch_types=(
            pltpu.SMEM((3 * n_segs,), jnp.int32),
            pltpu.SMEM((8,), jnp.int32),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.REGULAR,
            pltpu.SMEM((RANK_WORDS,), jnp.int32),
        ),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True, collective_id=SC_COLLECTIVE_ID
        ),
        name=name,
    )(
        q_sorted,
        seg,
        jnp.pad(counts4.reshape(-1).astype(jnp.int32), (0, 8 - counts4.size)),
        rank_operand(rank),
    )
