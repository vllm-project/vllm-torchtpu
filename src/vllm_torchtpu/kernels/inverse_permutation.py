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
"""Internal fixed-size integer inverse for the 128-route MoE permutation."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_SIZE = 128


def _inverse_permutation_kernel(permutation_ref, inverse_ref):
    # Destination ids occupy the lane dimension. The 128-by-128 values stay
    # inside the kernel; only the permutation and its inverse cross HBM.
    positions = jnp.arange(_SIZE, dtype=jnp.int32)[:, None]
    destinations = jnp.arange(_SIZE, dtype=jnp.int32)[None, :]
    matches = permutation_ref[...][:, None] == destinations
    contributions = jnp.where(matches, positions, jnp.int32(0))
    inverse_ref[...] = jnp.sum(contributions,
                               axis=0,
                               dtype=jnp.int32,
                               keepdims=True)


def inverse_permutation_128(permutation: jax.Array,
                            *,
                            interpret: bool = False) -> jax.Array:
    """Invert an int32 permutation of 0..127, as produced by ``argsort``.

    The caller guarantees that each value appears exactly once. Thus each
    destination has one matching position, and the integer sum is exact and
    bounded by 127. This is not an inverse for arbitrary routing expert ids.
    ``interpret=True`` executes the same Pallas body for CPU-only tests.
    """
    if permutation.shape != (_SIZE, ):
        raise ValueError("inverse_permutation_128 requires shape (128,)")
    if permutation.dtype != jnp.int32:
        raise TypeError("inverse_permutation_128 requires int32 indices")

    in_spec = pl.BlockSpec((_SIZE, ), lambda _: (0, ), memory_space=pltpu.VMEM)
    out_spec = pl.BlockSpec((1, _SIZE),
                            lambda _: (0, 0),
                            memory_space=pltpu.VMEM)
    return pl.pallas_call(
        _inverse_permutation_kernel,
        grid=(1, ),
        in_specs=[in_spec],
        out_specs=out_spec,
        out_shape=jax.ShapeDtypeStruct((1, _SIZE), jnp.int32),
        name="inverse_permutation_128",
        interpret=interpret,
    )(permutation).reshape((_SIZE, ))
