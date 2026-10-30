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
"""Materialize an array in the compact tiling Pallas TPU kernels read.

XLA tiles an intermediate for its consumers: a bf16 array that also feeds an
XLA matmul gets (16, 128) row tiles, while Pallas kernels take the compact
(8, 128) tiling. Handing such an array to a Pallas kernel costs a relayout copy
of every byte, placed wherever XLA's layout assignment puts it.
"""

import jax
from jax.experimental import pallas as pl


def _keep(x_ref, o_ref):
    # o_ref aliases x_ref, so the data is already where the output wants it.
    del x_ref, o_ref


def pin_compact_tiling(x: jax.Array) -> jax.Array:
    """Return ``x`` unchanged, laid out in the compact tiling.

    The kernel does nothing and aliases its operand to its result, so it only
    makes XLA produce ``x`` in the layout every Pallas kernel takes. Applied
    before a collective, the relayout (if any) runs on the collective's input
    rather than on its gathered output.
    """
    return pl.pallas_call(
        _keep,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        input_output_aliases={0: 0},
    )(x)
