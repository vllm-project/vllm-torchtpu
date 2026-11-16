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
"""Direct C512 page reads, using a reference bitcast inside SparseCore.

Ported from the hardware-validated 2026-09-21 C512 experiment.
This main-branch integration requires fresh validation.
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


def _gather_c512_kernel(cache_bytes, ids_hbm, out, sem, *, core_axis_name,
                               subcore_axis_name, rows, streams, steps):
    cache = cache_bytes.bitcast(jnp.int32)
    core = lax.axis_index((core_axis_name, subcore_axis_name))
    cores = lax.axis_size((core_axis_name, subcore_axis_name))

    def outer(ids):
        base = (pl.program_id(0) * cores + core) * steps

        def indices(r, s):
            return ids[pl.ds((r * streams + s) * rows, rows)]

        def body(*refs):
            copies = []
            for s in range(streams):
                start = ((base + pl.program_id(0)) * streams + s) * rows
                cp = pltpu.make_async_copy(refs[s], out.at[pl.ds(start, rows)],
                                           sem.at[s])
                cp.start()
                copies.append(cp)
            for cp in copies:
                cp.wait()

        pltpu.emit_pipeline(
            body,
            grid=(steps, ),
            in_specs=tuple(
                pl.BlockSpec((pl.Indirect(rows), 128),
                             lambda r, s=s: (indices(r, s), 0))
                for s in range(streams)),
        )(*([cache] * streams))

    pltpu.emit_pipeline(
        outer,
        grid=(pl.cdiv(ids_hbm.shape[0], cores * steps * streams * rows), ),
        in_specs=pl.BlockSpec((steps * streams * rows, ), lambda b:
                              (b * cores + core, )),
    )(ids_hbm)


@jax.jit
def gather_c512(cache, indices):
    """Direct paged uint8 C8 -> gathered int32 quarters, without whole-cache conversion."""
    if cache.ndim != 4 or cache.shape[-2:] != (4,128) or cache.dtype != jnp.uint8:
        raise ValueError("Expected uint8 [pages,page_size,4,128] C8")
    if indices.ndim != 1:
        raise ValueError("Expected flat physical token indices")
    if indices.size == 0:
        return jnp.empty((0,128),jnp.int32)
    cache = cache.reshape(-1,128)
    sc = pltpu.get_tpu_info().sparse_core
    rows, streams, steps = sc.num_lanes, 4, 8
    multiple = sc.num_cores * sc.num_subcores * rows * streams * steps
    size = pl.cdiv(indices.size, multiple) * multiple
    ids = jnp.pad(indices, (0, size - indices.size))
    mesh = plsc.VectorSubcoreMesh(num_cores=sc.num_cores,
                                  num_subcores=sc.num_subcores,
                                  core_axis_name="core",
                                  subcore_axis_name="subcore")
    result = pl.kernel(
        functools.partial(_gather_c512_kernel,
                          core_axis_name="core",
                          subcore_axis_name="subcore",
                          rows=rows,
                          streams=streams,
                          steps=steps),
        out_type=jax.ShapeDtypeStruct((size, 128), jnp.int32),
        mesh=mesh,
        scratch_types=[pltpu.SemaphoreType.DMA((streams, ))],
        compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=True,
                                             needs_layout_passes=True,
                                             disable_bounds_checks=True),
        name="sc_csa_gather_c512",
    )(cache, ids)
    return result[:indices.size]
