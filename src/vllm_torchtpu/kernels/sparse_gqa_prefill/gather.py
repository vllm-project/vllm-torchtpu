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
"""Native packed SparseCore gather for selected-KV prefill.

Ported from the standalone qsa_prefill v0.4 sc_decode gather helpers.
Scheduling follows openxla/tokamax csa_gather and ragged_gather; neither
kernel arithmetic nor native packed row layout changes in this port.
"""

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _gather_kernel(
    cache_ref, indices_ref, out_ref, sem_ref, *, streams, core_axis, subcore_axis
):
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes = info.num_lanes
    cores = lax.axis_size((core_axis, subcore_axis))
    core = lax.axis_index((core_axis, subcore_axis))
    # Native TC row packing: bitcast halves/quarters the row count, not D.
    src = cache_ref.bitcast(jnp.int32).reshape(-1, cache_ref.shape[-1])
    dst = out_ref.bitcast(jnp.int32).reshape(-1, out_ref.shape[-1])
    width = src.shape[-1]
    rows_per_core = lanes * streams
    blocks = pl.cdiv(indices_ref.shape[0], rows_per_core * cores)

    def outer(idx_ref):
        block = pl.program_id(0)

        def body(*gathers):
            copies = []
            for s, ref in enumerate(gathers):
                row = (block * cores + core) * rows_per_core + s * lanes
                copy = pltpu.make_async_copy(
                    ref, dst.at[pl.ds(row, lanes)], sem_ref.at[s]
                )
                copy.start()
                copies.append(copy)
            for copy in copies:
                copy.wait()

        specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(lanes), width),
                lambda _, s=s: (idx_ref[pl.ds(s * lanes, lanes)], 0),
            )
            for s in range(streams)
        )
        pltpu.emit_pipeline(body, grid=(1,), in_specs=specs)(*([src] * streams))

    pltpu.emit_pipeline(
        outer,
        grid=(blocks,),
        in_specs=pl.BlockSpec((rows_per_core,), lambda b: (b * cores + core,)),
    )(indices_ref)


def _take_lane(x, indices):
    # Keep singleton batch dimensions; jnp.take_along_axis elides them, which
    # does not match Mosaic's gather lowering contract.
    return lax.gather(
        x,
        indices[..., None],
        dimension_numbers=lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(1,),
            start_index_map=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
        ),
        slice_sizes=(1, 1),
        mode="promise_in_bounds",
    )
