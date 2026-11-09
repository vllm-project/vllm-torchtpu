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
"""Isolate a packed suffix from layout changes in its consumers."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def extract_packed_suffix(packed: jax.Array, num_words: int) -> jax.Array:
    assert packed.dtype == jnp.int32
    assert 0 < num_words <= packed.size
    lanes = pltpu.get_tpu_info().num_lanes
    padded_words = pl.cdiv(num_words, lanes) * lanes
    suffix = packed.reshape(-1)[-num_words:]
    suffix = jnp.pad(suffix, (0, padded_words - num_words)).reshape(1, -1)

    def copy_suffix(inp, out):
        out[...] = inp[...]

    # Keep downstream unpack/reshape fusion from relaying out the whole input.
    out = pl.pallas_call(
        copy_suffix,
        out_shape=jax.ShapeDtypeStruct((1, padded_words), packed.dtype),
        compiler_params=pltpu.CompilerParams(allow_input_fusion=(False,)),
    )(suffix)
    return out[:, :num_words]
