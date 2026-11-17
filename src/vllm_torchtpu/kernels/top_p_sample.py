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
"""Top-p token recovery of up to 8 rows, each row a dense ``[vocab / 128, 128]``
slab in VMEM.

XLA runs `rejection_sampler._recover_tokens` (softmax, the 31-step top-p
bisection, the Gumbel argmax) as ~37 passes over ``[rows, vocab]``, where a
row fills one of a vreg's 8 sublanes: every pass touches 8x the row's data,
~89 us at c4 b1 whatever the row count up to 8 (twin j-2e5da13f). Here each
row is a dense slab and every pass is a whole-slab vector op with the
threshold kept in a vreg: 18.5 us for one row, 43 us for 8 (kernel bench
j-87640300).

Same steps as the XLA path; the per-row sums run in another order.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROWS = 8
TEMPERATURE_EPS = 1e-9
_TINY = float(jnp.finfo(jnp.float32).tiny)
_EPS = float(jnp.finfo(jnp.float32).eps)
_BIG = 2**31 - 1
# Probabilities are below 2.0: bits 31 and 30 of the threshold are 0.
_THRESHOLD_BITS = 30


def _full(x: jax.Array, op) -> jax.Array:
    """Reduce a ``[G, 128]`` slab to ``[1, 1]``."""
    return op(op(x, axis=0, keepdims=True), axis=1, keepdims=True)


def _kernel(params_ref, draft_ref, logits_ref, u_ref, out_ref, *, rows, groups):
    ids = lax.broadcasted_iota(jnp.int32, (groups, 128), 0) * 128 + (
        lax.broadcasted_iota(jnp.int32, (groups, 128), 1)
    )
    sublane = lax.broadcasted_iota(jnp.int32, (ROWS, 128), 0)
    out = jnp.zeros((ROWS, 128), jnp.int32)
    for r in range(rows):
        temperature, top_p, draft = params_ref[0, r], params_ref[1, r], draft_ref[r]
        x = logits_ref[r].astype(jnp.float32)
        x_max = _full(x, jnp.max)
        argmax = _full(jnp.where(x == x_max, ids, _BIG), jnp.min)
        denom = temperature + TEMPERATURE_EPS
        e = jnp.exp(x / denom - x_max / denom)
        p = e / _full(e, jnp.sum)
        # The largest float32 threshold whose mass at or above stays >= top_p.
        bits = jnp.zeros((1, 1), jnp.int32)
        for bit in reversed(range(_THRESHOLD_BITS)):
            mid = bits | (1 << bit)
            t = lax.bitcast_convert_type(mid, jnp.float32)
            mass = _full(jnp.where(p >= t, p, 0.0), jnp.sum)
            bits = jnp.where(mass < top_p, bits, mid)
        threshold = jnp.where(
            top_p < 1.0, lax.bitcast_convert_type(bits, jnp.float32), -1.0
        )
        kept = jnp.where(p >= threshold, e, 0.0)
        q = jnp.where(ids == draft, 0.0, kept / _full(kept, jnp.sum))
        u = jnp.clip(u_ref[r], _TINY, 1.0 - _EPS)
        score = q / (-jnp.log(u) + _TINY)
        token = _full(jnp.where(score == _full(score, jnp.max), ids, _BIG), jnp.min)
        sampled = (temperature != 0.0) & (_full(q, jnp.max) > 0.0)
        result = jnp.where(sampled, token, argmax)
        out = jnp.where(sublane == r, jnp.broadcast_to(result, (ROWS, 128)), out)
    out_ref[...] = out


def supports(rows: int, vocab: int) -> bool:
    return 0 < rows <= ROWS and vocab % 128 == 0


def top_p_rows(
    logits: jax.Array,
    recover_u: jax.Array,
    params: jax.Array,
    draft: jax.Array,
    interpret: bool = False,
) -> jax.Array:
    """Token per row of ``logits`` ``[rows, G, 128]`` (rows <= 8; vocab id
    ``g * 128 + lane``), sampling with uniforms ``recover_u`` f32
    ``[rows, G, 128]``. ``params`` f32 ``[2, rows]``: temperature (0:
    greedy) and top_p (>= 1: no top-p); ``draft`` int32 ``[rows]``: the
    rejected draft to exclude (-1: none). Returns int32 ``[8, 128]``, row r's
    token in sublane r."""
    rows, groups, lanes = logits.shape
    if lanes != 128 or not supports(rows, groups * 128):
        raise ValueError(f"top_p_rows does not take logits {logits.shape}")
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    return pl.pallas_call(
        functools.partial(_kernel, rows=rows, groups=groups),
        in_specs=[smem, smem, vmem, vmem],
        out_specs=vmem,
        out_shape=jax.ShapeDtypeStruct((ROWS, 128), jnp.int32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=48 << 20),
        interpret=interpret,
    )(params, draft, logits, recover_u)
