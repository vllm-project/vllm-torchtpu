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
"""LM head on an e4m3 weight with one f32 scale per vocabulary row.

``logits = (x @ w.T) * scale``: the bf16 rows ``x`` meet the e4m3 weight
tiles straight on the MXU (a mixed bf16 x e4m3 matmul, no widening pass), so
a decode step reads half the bf16 head's bytes. The weight keeps vLLM's
``[vocab, hidden]`` layout and is contracted transposed.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)
# Vocabulary rows per weight tile (multiple of 128): 11 MiB of e4m3 at a
# 6144 hidden size, double-buffered.
BLOCK_V = 1792
# Activation rows per step; more rows split into row tiles.
BLOCK_M = 512


def _kernel(x_ref, w_ref, s_ref, o_ref):
    acc = jax.lax.dot_general(
        x_ref[...],
        w_ref[...],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    o_ref[...] = (acc * s_ref[...]).astype(o_ref.dtype)


def fp8_lm_head(
    x: jax.Array, weight: jax.Array, scale: jax.Array, interpret: bool = False
) -> jax.Array:
    """bf16 ``[rows, vocab]`` logits of bf16 ``x [rows, hidden]`` against the
    e4m3 ``weight [vocab, hidden]`` and its f32 per-row ``scale [1, vocab]``."""
    rows, hidden = x.shape
    vocab = weight.shape[0]
    bv = min(BLOCK_V, vocab)
    bm = min(BLOCK_M, rows)
    vmem = 2 * (bm * hidden * 2 + bv * hidden + bv * 4 + bm * bv * 2) + bm * bv * 4
    return pl.pallas_call(
        _kernel,
        # Row tiles innermost: a weight tile is fetched once for all of them.
        grid=(pl.cdiv(vocab, bv), pl.cdiv(rows, bm)),
        in_specs=[
            pl.BlockSpec((bm, hidden), lambda j, i: (i, 0)),
            pl.BlockSpec((bv, hidden), lambda j, i: (j, 0)),
            pl.BlockSpec((1, bv), lambda j, i: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bv), lambda j, i: (i, j)),
        out_shape=jax.ShapeDtypeStruct((rows, vocab), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=vmem + (8 << 20),
        ),
        name="fp8_lm_head",
        interpret=interpret,
    )(x, weight, scale)
