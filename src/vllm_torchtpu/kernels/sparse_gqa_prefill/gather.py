# SPDX-License-Identifier: Apache-2.0
"""Four-stream indirect DMA for native packed selected KV."""

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _gather_vectors(src, dst, sem_ref, indices, output_row, *, streams, lanes, width):
    def body(*gathers):
        copies = []
        for s, ref in enumerate(gathers):
            row = output_row + s * lanes
            copy = pltpu.make_async_copy(ref, dst.at[pl.ds(row, lanes)], sem_ref.at[s])
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait()

    specs = tuple(
        pl.BlockSpec(
            (pl.Indirect(lanes), width),
            lambda _, s=s: (indices[s], 0),
        )
        for s in range(streams)
    )
    pltpu.emit_pipeline(body, grid=(1,), in_specs=specs)(*([src] * streams))
