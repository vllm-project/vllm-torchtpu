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
"""Internal joint stable order and inverse for 128 signed integer keys."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_SIZE = 128


def _stable_order_and_inverse_kernel(keys_ref, order_ref, inverse_ref):
    keys = keys_ref[...]
    positions = jnp.arange(_SIZE, dtype=jnp.int32)
    # Rows are potential predecessors j; columns are original positions i.
    # Compare signed keys directly: subtraction could overflow at int32 extrema.
    precedes = ((keys[:, None] < keys[None, :]) |
                ((keys[:, None] == keys[None, :]) &
                 (positions[:, None] < positions[None, :])))
    inverse = jnp.sum(precedes.astype(jnp.int32),
                      axis=0,
                      dtype=jnp.int32,
                      keepdims=True)
    # Unique ranks select exactly one original position per destination.
    # Position zero participates too; its exact contribution is zero.
    matches = inverse.reshape((_SIZE, 1)) == positions[None, :]
    contributions = jnp.where(matches, positions[:, None], jnp.int32(0))
    order_ref[...] = jnp.sum(contributions,
                             axis=0,
                             dtype=jnp.int32,
                             keepdims=True)
    inverse_ref[...] = inverse


def stable_order_and_inverse_128(
    sort_keys: jax.Array,
    *,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return stable argsort and its inverse for arbitrary int32[128] keys.

    Lexicographic (signed key, original position) ranks are a permutation of
    0..127 even for duplicate keys. Both integer reductions are exact and
    bounded by 127; there is no packing, subtraction or key-range assumption.
    ``interpret=True`` requests Pallas interpretation for CPU-only checks;
    support depends on the installed JAX interpreter.
    """
    if sort_keys.shape != (_SIZE, ):
        raise ValueError("stable_order_and_inverse_128 requires shape (128,)")
    if sort_keys.dtype != jnp.int32:
        raise TypeError("stable_order_and_inverse_128 requires int32 keys")

    in_spec = pl.BlockSpec((_SIZE, ), lambda _: (0, ), memory_space=pltpu.VMEM)
    order_spec = pl.BlockSpec((1, _SIZE),
                              lambda _: (0, 0),
                              memory_space=pltpu.VMEM)
    inverse_spec = pl.BlockSpec((1, _SIZE),
                                lambda _: (0, 0),
                                memory_space=pltpu.VMEM)
    order, inverse = pl.pallas_call(
        _stable_order_and_inverse_kernel,
        grid=(1, ),
        in_specs=[in_spec],
        out_specs=(order_spec, inverse_spec),
        out_shape=(jax.ShapeDtypeStruct(
            (1, _SIZE), jnp.int32), jax.ShapeDtypeStruct((1, _SIZE),
                                                         jnp.int32)),
        name="stable_order_and_inverse_128",
        interpret=interpret,
    )(sort_keys)
    # Reshape array outputs outside Pallas, never the rank-one input Ref.
    return order.reshape((_SIZE, )), inverse.reshape((_SIZE, ))
