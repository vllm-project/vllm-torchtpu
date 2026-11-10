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
"""Hierarchical all-reduce of a 4-core tensor-parallel group on two chips.

The group's members are ``t = 2 * chip + core``: two cores of one chip and the
two cores of a neighbouring chip. The chip-to-chip link is the slow hop, so
each chip first sums its two partials over the die-to-die link and only half
of the data crosses between chips in each direction:

1. intra-chip reduce-scatter: core ``k`` keeps column half ``k`` and adds the
   chip's other core's copy of it,
2. cross-chip reduce-scatter: inside half ``k``, chip ``c`` keeps quarter
   ``c`` and adds the partner core's (``t ^ 2``) pair sum of it,
3. cross-chip all-gather: the finished quarter goes to the partner,
4. intra-chip all-gather: the finished quarter goes to the chip's other core,
   and the partner's finished quarter is forwarded to it as it lands.

Every quarter is split into column blocks so each step starts on a block as
soon as its inputs land. Every element is summed as
``bf16(bf16(P0 + P1) + bf16(P2 + P3))`` on every member: the same values on
all four, run after run (each pairwise add is commutative).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_MESH = pl.DeviceIdType.MESH
_LANES = 128


def _add_bf16(a, b):
    return (a.astype(jnp.float32) + b.astype(jnp.float32)).astype(jnp.bfloat16)


def _barrier(axis, peers):
    """Barrier with the two peers on the global barrier semaphore, then a second
    round on a scoped semaphore so a member that re-enters the kernel on the
    next call cannot release a peer still waiting in this one."""

    def round_(sem):
        for peer in peers:
            pl.semaphore_signal(sem, 1, device_id={axis: peer}, device_id_type=_MESH)
        pl.semaphore_wait(sem, len(peers))

    round_(pltpu.get_barrier_semaphore())

    @functools.partial(pl.run_scoped, second=pltpu.SemaphoreType.REGULAR)
    def _(second):
        round_(second)


def supports(rows: int, hidden: int, num_blocks: int, max_rows: int) -> bool:
    """Shapes the kernel takes: column blocks of whole lane tiles, and rows small
    enough for the VMEM staging (1.75 x rows x hidden bf16)."""
    return 0 < rows <= max_rows and hidden % (4 * num_blocks * _LANES) == 0


def tp4_all_reduce_local(
    x: jax.Array,
    *,
    axis: str,
    num_blocks: int,
    collective_id: int,
    interpret: bool = False,
) -> jax.Array:
    """All-reduce of ``x`` [R, H] bf16 over ``axis`` (size 4); call inside shard_map.
    ``interpret`` runs the kernel in the Pallas TPU interpreter (CPU tests)."""
    rows, hidden = x.shape
    h2, h4 = hidden // 2, hidden // 4
    bw = h4 // num_blocks
    nb = num_blocks

    def kernel(
        x_hbm, o_hbm, xv, r1, r2, sb, fin, lsem, s1, s2, s3, sf, rr1, rr2, rag_p, rag_m
    ):
        me = lax.axis_index(axis)
        k = lax.rem(me, 2)
        c = me // 2
        other_core = me + 1 - 2 * k
        partner = lax.rem(me + 2, 4)

        def cols(base, i):
            return pl.ds(pl.multiple_of(base + i * bw, _LANES), bw)

        def blk(base, i):
            return pl.ds(base + i * bw, bw)

        q_partner = (
            k * h2 + (1 - c) * h4
        )  # quarter of my half that the partner finishes
        q_mine = k * h2 + c * h4
        m_partner = (1 - k) * h2 + (1 - c) * h4  # the other core's half, in its order
        m_mine = (1 - k) * h2 + c * h4

        loads = []
        for j, q in enumerate((q_partner, q_mine)):
            for i in range(nb):
                cp = pltpu.make_async_copy(
                    x_hbm.at[:, cols(q, i)],
                    xv.at[:, blk(j * h4, i)],
                    lsem.at[j * nb + i],
                )
                cp.start()
                loads.append(cp)
        _barrier(axis, (other_core, partner))

        # 1. the other core's half goes to the other core
        sends1 = []
        for j, q in enumerate((m_partner, m_mine)):
            for i in range(nb):
                d = pltpu.make_async_remote_copy(
                    x_hbm.at[:, cols(q, i)],
                    r1.at[:, blk(j * h4, i)],
                    s1.at[0],
                    rr1.at[j * nb + i],
                    device_id={axis: other_core},
                    device_id_type=_MESH,
                )
                d.start()
                sends1.append(d)
        # 2. pair sums of the partner's quarter go to the partner, block by block
        sends2 = []
        for i in range(nb):
            loads[i].wait()
            sends1[i].wait_recv()
            sb[:, blk(0, i)] = _add_bf16(xv[:, blk(0, i)], r1[:, blk(0, i)])
            d = pltpu.make_async_remote_copy(
                sb.at[:, blk(0, i)],
                r2.at[:, blk(0, i)],
                s2.at[0],
                rr2.at[i],
                device_id={axis: partner},
                device_id_type=_MESH,
            )
            d.start()
            sends2.append(d)
        for i in range(nb):
            loads[nb + i].wait()
            sends1[nb + i].wait_recv()
            fin[:, blk(0, i)] = _add_bf16(xv[:, blk(h4, i)], r1[:, blk(h4, i)])
        # 3. chip sums of my quarter: to the partner, the other core and my output
        outs, stores = [], []
        for i in range(nb):
            sends2[i].wait_recv()
            fin[:, blk(0, i)] = _add_bf16(fin[:, blk(0, i)], r2[:, blk(0, i)])
            for peer, sem, rsem in (
                (partner, s3.at[0], rag_p.at[i]),
                (other_core, s3.at[1], rag_m.at[0]),
            ):
                d = pltpu.make_async_remote_copy(
                    fin.at[:, blk(0, i)],
                    o_hbm.at[:, cols(q_mine, i)],
                    sem,
                    rsem,
                    device_id={axis: peer},
                    device_id_type=_MESH,
                )
                d.start()
                outs.append(d)
            st = pltpu.make_async_copy(
                fin.at[:, blk(0, i)], o_hbm.at[:, cols(q_mine, i)], lsem.at[2 * nb + i]
            )
            st.start()
            stores.append(st)
        # 4. the partner's finished quarter is forwarded to the other core as it lands
        fwds = []
        for i in range(nb):
            pltpu.make_async_remote_copy(
                fin.at[:, blk(0, i)],
                o_hbm.at[:, cols(q_partner, i)],
                s3.at[0],
                rag_p.at[i],
                device_id={axis: partner},
                device_id_type=_MESH,
            ).wait_recv()
            d = pltpu.make_async_remote_copy(
                o_hbm.at[:, cols(q_partner, i)],
                o_hbm.at[:, cols(q_partner, i)],
                sf.at[0],
                rag_m.at[0],
                device_id={axis: other_core},
                device_id_type=_MESH,
            )
            d.start()
            fwds.append(d)
        for d in sends1 + sends2 + outs + fwds:
            d.wait_send()
        for st in stores:
            st.wait()
        # the other core's finished quarter and its forward of its partner's
        for _ in range(2 * nb):
            fwds[0].wait_recv()

    space = pl.ANY if interpret else pltpu.MemorySpace.HBM
    extra = {"interpret": pltpu.InterpretParams()} if interpret else {}
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=space)],
        out_specs=pl.BlockSpec(memory_space=space),
        scratch_shapes=[
            pltpu.VMEM((rows, h2), jnp.bfloat16),  # my half: partner quarter, then mine
            pltpu.VMEM((rows, h2), jnp.bfloat16),  # the other core's copy of my half
            pltpu.VMEM(
                (rows, h4), jnp.bfloat16
            ),  # the partner's pair sums of my quarter
            pltpu.VMEM(
                (rows, h4), jnp.bfloat16
            ),  # my pair sums of the partner's quarter
            pltpu.VMEM((rows, h4), jnp.bfloat16),  # my quarter
            pltpu.SemaphoreType.DMA((3 * nb,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((1,)),
            pltpu.SemaphoreType.DMA((2 * nb,)),
            pltpu.SemaphoreType.DMA((nb,)),
            pltpu.SemaphoreType.DMA((nb,)),
            pltpu.SemaphoreType.DMA((1,)),
        ],
        compiler_params=pltpu.CompilerParams(collective_id=collective_id),
        name="tp4_hier_all_reduce",
        **extra,
    )(x)


def tp4_all_reduce(
    x: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    dp_axis: str,
    tp_axis: str,
    num_blocks: int,
    max_rows: int,
    collective_id: int = 0,
    interpret: bool = False,
) -> jax.Array:
    """Global view: ``x`` [dp, tp, R, H] stacks every rank's partial; returns the
    same layout with each rank's TP sum. Shapes the kernel does not take use
    XLA's all-reduce."""
    spec = jax.sharding.PartitionSpec(dp_axis, tp_axis)
    rows, hidden = x.shape[2], x.shape[3]
    if (
        mesh.shape[tp_axis] != 4
        or x.dtype != jnp.bfloat16
        or not supports(rows, hidden, num_blocks, max_rows)
    ):

        def body(a):
            return lax.psum(a, tp_axis)
    else:

        def body(a):
            return tp4_all_reduce_local(
                a[0, 0],
                axis=tp_axis,
                num_blocks=num_blocks,
                collective_id=collective_id,
                interpret=interpret,
            )[None, None]

    return jax.shard_map(
        body, mesh=mesh, in_specs=(spec,), out_specs=spec, check_vma=False
    )(x)
