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
"""Three-phase activation all-gather for a 2 x 2 x 1 dual-die mesh."""

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.compute_on import compute_on
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def all_gather(x: jax.Array, axis_name: str, masks: tuple) -> jax.Array:
    """Gather eight shards using four colors and two batches per color."""
    (xmask, ymask, diemask), (origin, d0, d1, d2) = masks
    assert x.size % 8 == 0
    chunk = x.size // 8
    data = x.reshape(-1)

    def body(src, out, send, recv, local_sem, barrier, shared):
        rank = lax.axis_index(axis_name)
        # All ranks initialize their semaphore storage before remote writes.
        for peer in range(8):
            pl.semaphore_signal(barrier, 1, device_id={axis_name: peer, "sc": 0})
        pl.semaphore_wait(barrier, 8)

        def wire_slot(source_rank, color):
            physical = (
                origin
                ^ ((source_rank & 1) * d0)
                ^ (((source_rank >> 1) & 1) * d1)
                ^ (((source_rank >> 2) & 1) * d2)
            )
            first_bit, second_bit = (1, 2) if color < 2 else (2, 1)
            return (
                (physical & 1) * 4
                + ((physical >> second_bit) & 1) * 2
                + ((physical >> first_bit) & 1)
            )

        def transfer(color, source_rank, peer, slot, pair=False):
            return pltpu.async_remote_copy(
                shared.at[
                    pl.ds(
                        (
                            color * 8
                            + (wire_slot(source_rank, color) & (~1 if pair else -1))
                        )
                        * chunk,
                        (2 if pair else 1) * chunk,
                    )
                ],
                shared.at[
                    pl.ds(
                        (
                            color * 8
                            + (wire_slot(source_rank, color) & (~1 if pair else -1))
                        )
                        * chunk,
                        (2 if pair else 1) * chunk,
                    )
                ],
                send.at[slot],
                recv.at[slot],
                device_id={axis_name: peer, "sc": 0},
            )

        @pl.loop(0, 2)
        def batches(batch):
            with jax.named_scope("phase1_four_colors"):
                pending = []
                for color in range(4):
                    first = xmask if color < 2 else ymask
                    pltpu.async_copy(
                        src.at[pl.ds((color * 2 + batch) * chunk, chunk)],
                        shared.at[
                            pl.ds((color * 8 + wire_slot(rank, color)) * chunk, chunk)
                        ],
                        local_sem.at[color],
                    ).wait()
                    pending.append(transfer(color, rank, rank ^ first, color * 7))
                for dma in pending:
                    dma.wait()
            with jax.named_scope("phase2_four_colors_cross_and_die"):
                pending = []
                for color in range(4):
                    second = ymask if color < 2 else xmask
                    pending.append(
                        transfer(color, rank, rank ^ second, color * 7 + 1, pair=True)
                    )
                    pending.append(
                        transfer(color, rank, rank ^ diemask, color * 7 + 3, pair=True)
                    )
                for dma in pending:
                    dma.wait()
            with jax.named_scope("phase3_four_colors_to_die"):
                pending = []
                for color in range(4):
                    second = ymask if color < 2 else xmask
                    pending.append(
                        transfer(
                            color,
                            rank ^ second,
                            rank ^ diemask,
                            color * 7 + 5,
                            pair=True,
                        )
                    )
                for dma in pending:
                    dma.wait()
            with jax.named_scope("drain_spmem_to_hbm"):
                drains = []
                for source_rank in range(8):
                    for color in range(4):
                        drains.append(
                            pltpu.async_copy(
                                shared.at[
                                    pl.ds(
                                        (color * 8 + wire_slot(source_rank, color))
                                        * chunk,
                                        chunk,
                                    )
                                ],
                                out.at[
                                    pl.ds(
                                        (source_rank * 8 + color * 2 + batch) * chunk,
                                        chunk,
                                    )
                                ],
                                local_sem.at[source_rank * 4 + color],
                            )
                        )
                for dma in drains:
                    dma.wait()

    kernel = pl.kernel(
        body,
        mesh=plsc.ScalarSubcoreMesh(axis_name="sc", num_cores=1),
        out_type=jax.ShapeDtypeStruct((8 * x.size,), x.dtype),
        scratch_types=[
            pltpu.SemaphoreType.DMA((28,)),
            pltpu.SemaphoreType.DMA((28,)),
            pltpu.SemaphoreType.DMA((32,)),
            pltpu.SemaphoreType.REGULAR,
            pltpu.VMEM_SHARED((32 * chunk,), x.dtype),
        ],
        name="topology_three_phase_ag",
    )
    with jax.named_scope("moe_activation_ag"):
        result = compute_on(
            kernel,
            compute_type="tpu_sparsecore",
            out_memory_spaces=jax.memory.Space.Device,
            compiler_options={"sparse_core_config": {"core_ids": [1]}},
        )(data)
    return result.reshape((8 * x.shape[0], *x.shape[1:]))
