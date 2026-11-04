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
