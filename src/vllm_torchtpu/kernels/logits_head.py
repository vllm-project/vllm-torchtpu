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
"""BF16 linear head writing BF16 logits with full-K weight tiles."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Measured on v7x for the DSpark drafter's [240, 5120] x [129280, 5120]^T:
# 435 us against 530 us for the XLA dot (the HBM floor is 448 us).
TUNED_TILES = {(240, 5120, 129280): 1280}


def is_supported_shape(rows: int, weight_shape: tuple[int, ...]) -> bool:
    return (
        len(weight_shape) == 2
        and (rows, weight_shape[1], weight_shape[0]) in TUNED_TILES
    )


def logits_head(hidden: jax.Array, weight: jax.Array) -> jax.Array:
    """``hidden @ weight.T`` in BF16 with FP32 accumulation, for [vocab, K] W."""
    rows, width = hidden.shape
    vocab, weight_width = weight.shape
    assert hidden.dtype == weight.dtype == jnp.bfloat16
    assert width == weight_width
    tile = TUNED_TILES[(rows, width, vocab)]

    def kernel(hidden_ref, weight_ref, logits_ref):
        logits_ref[...] = jnp.dot(
            hidden_ref[...], weight_ref[...].T, preferred_element_type=jnp.float32
        ).astype(logits_ref.dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, vocab), jnp.bfloat16),
        grid=(vocab // tile,),
        in_specs=(
            pl.BlockSpec((rows, width), lambda n: (0, 0)),
            pl.BlockSpec((tile, width), lambda n: (n, 0)),
        ),
        out_specs=pl.BlockSpec((rows, tile), lambda n: (0, n)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name=f"logits_head_m{rows}_n{tile}",
    )(hidden, weight)
