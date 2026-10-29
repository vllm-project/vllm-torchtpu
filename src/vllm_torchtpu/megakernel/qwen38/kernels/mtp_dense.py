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
"""Native BF16 full-K/N128 MTP projections and experts, with bounded rings of weight DMAs."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

# Upper bound on one weight DMA chunk.
CHUNK_BYTES = 2 << 20
# Chunks in VMEM at once: all but the one being contracted are in flight, which
# covers a transfer's issue-to-completion latency at HBM bandwidth.
SLOTS = 4


def native_project(source, matrix, destination, *, rounded=True):
    """Preserve independent MTP reference dots, including sub128 shared/FC shards.

    Destination holds ceil(N/128)*128 columns. Padding exists only in the VMEM
    tile/output; native checkpoint buffers retain their exact logical shape.
    BF16 expert matrices are streamed by columns, never copied whole to VMEM.
    """
    contraction, columns = matrix.shape
    rows = source.shape[0]
    if matrix.dtype != jnp.bfloat16 or rows % 8 or not 8 <= rows <= 64:
        raise ValueError("Native MTP requires BF16 weights and 8 to 64 padded activation rows")
    if columns > 128 and columns % 128:
        raise ValueError("Native MTP output shards must be sub128 or complete128-column tiles")
    padded = (columns + 127) // 128 * 128
    if destination.shape[0] != rows or destination.shape[1] < padded:
        raise ValueError("Native MTP destination must include its final padded128-column tile")
    width = min(128, columns)
    # Short contractions move several 128-column tiles per DMA; every tile
    # keeps its own dot.
    tiles = 1
    while columns % (2 * tiles * width) == 0 and 2 * tiles * width * contraction * 2 <= CHUNK_BYTES:
        tiles *= 2
    chunk = tiles * width
    count = columns // chunk
    slots = min(SLOTS, count)
    activation = source[:, :contraction].astype(jnp.bfloat16)

    def compute(tile, sem):
        def transfer(index, slot):
            return tpu.make_async_copy(
                matrix.at[:, pl.ds(index * chunk, chunk)], tile.at[slot], sem.at[slot]
            )

        for index in range(slots - 1):
            transfer(index, index).start()

        @pl.loop(0, count)
        def column(index):
            slot = index % slots
            # The previous chunk's slot is free once its dots are done.
            ahead = index + slots - 1

            @pl.when(ahead < count)
            def prefetch():
                transfer(ahead, ahead % slots).start()

            transfer(index, slot).wait()

            for part in range(tiles):
                rhs = jnp.pad(
                    tile[slot, :, pl.ds(part * width, width)], ((0, 0), (0, 128 - width))
                )
                # The independent reference lowers sub128 shards with the weight
                # operand first. Preserve that FP32 accumulation association: the
                # opposite orientation can cross a final BF16 rounding boundary.
                if columns < 128:
                    value = jnp.dot(rhs.T, activation.T, preferred_element_type=jnp.float32).T
                else:
                    value = jnp.dot(activation, rhs, preferred_element_type=jnp.float32)
                if rounded:
                    value = value.astype(jnp.bfloat16)
                destination[:, pl.ds((index * tiles + part) * 128, 128)] = value.astype(
                    destination.dtype
                )

    pl.run_scoped(
        compute,
        tile=tpu.VMEM((slots, contraction, chunk), jnp.bfloat16),
        sem=tpu.SemaphoreType.DMA((slots,)),
    )



def _chunk_columns(contraction, columns):
    # native_project's DMA chunk: whole 128-column tiles up to CHUNK_BYTES.
    tiles = 1
    while columns % (2 * tiles * 128) == 0 and 2 * tiles * 128 * contraction * 2 <= CHUNK_BYTES:
        tiles *= 2
    return tiles * 128


def native_experts(source, ids, probabilities, weights, first, experts, gate, up, middle, down,
                   moe):
    """Add routed native BF16 experts into ``moe`` in ascending ID order, as native_project would.

    ``weights`` are the [local expert, ...] gate, up and DOWN matrices; ``ids`` and
    ``probabilities`` are the routes, with unrouted entries at ``experts``. Gate and up chunks
    stream through one ring and DOWN chunks through another: an expert's DOWN chunks load while
    its gate and up contract, and the next expert's leading gate chunks while its DOWN
    contracts, so no projection waits for its first chunk. Every 128-column tile keeps
    native_project's dot and rounding.
    """
    gate_weight, up_weight, down_weight = weights
    _, contraction, width = gate_weight.shape
    hidden = down_weight.shape[-1]
    if gate_weight.dtype != jnp.bfloat16 or up_weight.shape != gate_weight.shape:
        raise ValueError("Streamed experts require matching BF16 gate and up weights")
    if contraction % 128 or width % 128 or hidden % 128:
        raise ValueError("Streamed experts require complete 128-column tiles")
    gate_chunk, down_chunk = _chunk_columns(contraction, width), _chunk_columns(width, hidden)
    gate_count, down_count = width // gate_chunk, hidden // down_chunk

    def compute(gate_ring, down_ring, gate_sem, down_sem):
        # Gate ring entries are an expert's gate chunks, then its up chunks.
        def gate_copy(matrix, local, index, entry):
            slot = entry % SLOTS
            return tpu.make_async_copy(
                matrix.at[local, :, pl.ds(index * gate_chunk, gate_chunk)],
                gate_ring.at[slot],
                gate_sem.at[slot],
            )

        def down_copy(local, index):
            slot = index % SLOTS
            return tpu.make_async_copy(
                down_weight.at[local, :, pl.ds(index * down_chunk, down_chunk)],
                down_ring.at[slot],
                down_sem.at[slot],
            )

        def start_leading(local):
            for entry in range(min(SLOTS - 1, 2 * gate_count)):
                matrix = gate_weight if entry < gate_count else up_weight
                gate_copy(matrix, local, entry % gate_count, entry).start()

        def start_gate(local, entry):
            @pl.when(entry < gate_count)
            def gate_entry():
                gate_copy(gate_weight, local, entry, entry).start()

            @pl.when((entry >= gate_count) & (entry < 2 * gate_count))
            def up_entry():
                gate_copy(up_weight, local, entry - gate_count, entry).start()

        def contract(activation, tile, chunk, index, destination, rounded):
            for part in range(chunk // 128):
                value = jnp.dot(
                    activation, tile[:, pl.ds(part * 128, 128)], preferred_element_type=jnp.float32
                )
                if rounded:
                    value = value.astype(jnp.bfloat16)
                column = (index * (chunk // 128) + part) * 128
                destination[:, pl.ds(column, 128)] = value.astype(destination.dtype)

        def visit(expert):
            selected = ids[...] == expert
            probability = jnp.sum(jnp.where(selected, probabilities[...], 0), axis=-1)
            ids[...] = jnp.where(selected, experts, ids[...])
            following = jnp.min(ids[...])
            local = expert - first
            for index in range(min(SLOTS - 1, down_count)):
                down_copy(local, index).start()
            def stream_gate(activation, matrix, offset, destination):
                @pl.loop(0, gate_count)
                def gate_column(index):
                    entry = offset + index
                    # The previous entry's slot is free once its dots are done.
                    start_gate(local, entry + SLOTS - 1)
                    gate_copy(matrix, local, index, entry).wait()
                    contract(activation, gate_ring.at[entry % SLOTS], gate_chunk, index,
                             destination, True)

            activation = source[:, :contraction].astype(jnp.bfloat16)
            stream_gate(activation, gate_weight, 0, gate)
            stream_gate(activation, up_weight, gate_count, up)

            # Every gate ring slot is free again.
            pl.when(following < experts)(lambda: start_leading(following - first))
            middle[:, :width] = (
                jax.nn.silu(gate[:, :width].astype(jnp.float32)).astype(jnp.bfloat16)
                * up[:, :width].astype(jnp.bfloat16)
            ).astype(jnp.bfloat16)
            activation = middle[:, :width].astype(jnp.bfloat16)

            @pl.loop(0, down_count)
            def down_column(index):
                @pl.when(index + SLOTS - 1 < down_count)
                def prefetch():
                    down_copy(local, index + SLOTS - 1).start()

                down_copy(local, index).wait()
                contract(activation, down_ring.at[index % SLOTS], down_chunk, index, down, False)

            moe[...] += down[:, :hidden] * probability[:, None]
            return following

        leading = jnp.min(ids[...])
        pl.when(leading < experts)(lambda: start_leading(leading - first))
        jax.lax.while_loop(lambda expert: expert < experts, visit, leading)

    pl.run_scoped(
        compute,
        gate_ring=tpu.VMEM((SLOTS, contraction, gate_chunk), jnp.bfloat16),
        down_ring=tpu.VMEM((SLOTS, width, down_chunk), jnp.bfloat16),
        gate_sem=tpu.SemaphoreType.DMA((SLOTS,)),
        down_sem=tpu.SemaphoreType.DMA((SLOTS,)),
    )
