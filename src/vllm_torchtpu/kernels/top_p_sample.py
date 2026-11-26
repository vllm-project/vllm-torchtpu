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
slab in VMEM (or a vocab-parallel head's shards side by side, ``[tp,
padded shard]``).

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


# Counter-based uniforms (`uniform_from_key`): a row's stream key mixed with
# each vocab id by the lowbias32 integer hash; `rejection_sampler.stream_keys`
# makes the keys and the same uniforms in torch.
ID_MULT = -2048144789  # 0x85ebca6b as int32
HASH_MULTS = (0x7FEB352D, -2073254261)  # 0x846ca68b as int32


def hash32(x: jax.Array) -> jax.Array:
    """lowbias32 on int32 (wrapping multiplies, logical shifts)."""
    x = x ^ lax.shift_right_logical(x, 16)
    x = x * jnp.int32(HASH_MULTS[0])
    x = x ^ lax.shift_right_logical(x, 15)
    x = x * jnp.int32(HASH_MULTS[1])
    return x ^ lax.shift_right_logical(x, 16)


def uniform_from_key(ids: jax.Array, key) -> jax.Array:
    """f32 uniforms in [0, 1) on 24-bit steps, one per id of a row's stream."""
    bits = hash32(ids * jnp.int32(ID_MULT) + key)
    return lax.shift_right_logical(bits, 8).astype(jnp.float32) * (1.0 / (1 << 24))


def _full(x: jax.Array, op) -> jax.Array:
    """Reduce a ``[G, W]`` slab to ``[1, 1]``, the long axis first: it spans
    vregs, which combine elementwise; the short one then reduces one vreg's
    sublanes or lanes (reducing 8 sublanes first rotated every vreg)."""
    first = 0 if x.shape[0] >= x.shape[1] else 1
    return op(op(x, axis=first, keepdims=True), axis=1 - first, keepdims=True)


def _kernel(
    params_ref,
    draft_ref,
    logits_ref,
    u_ref,
    out_ref,
    *,
    rows,
    shape,
    width,
    vocab,
    keyed,
):
    """``u_ref``: the uniforms ``[rows, G, W]`` in VMEM, or with ``keyed``
    each row's stream key in SMEM (`uniform_from_key` of the vocab ids)."""
    sub = lax.broadcasted_iota(jnp.int32, shape, 0)
    col = lax.broadcasted_iota(jnp.int32, shape, 1)
    ids = sub * width + col
    # Columns past `width` (a shard's padding; their `ids` alias the next
    # shard's, harmless only because they are masked here) and ids past the
    # vocabulary.
    valid = None
    if width != shape[1] or vocab != shape[0] * width:
        valid = (col < width) & (ids < vocab)
    sublane = lax.broadcasted_iota(jnp.int32, (ROWS, 128), 0)
    out = jnp.zeros((ROWS, 128), jnp.int32)
    for r in range(rows):
        temperature, top_p, draft = params_ref[0, r], params_ref[1, r], draft_ref[r]
        x = logits_ref[r].astype(jnp.float32)
        if valid is not None:
            x = jnp.where(valid, x, -jnp.inf)
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
        u = uniform_from_key(ids, u_ref[r]) if keyed else u_ref[r]
        u = jnp.clip(u, _TINY, 1.0 - _EPS)
        score = q / (-jnp.log(u) + _TINY)
        if valid is not None:
            score = jnp.where(valid, score, 0.0)  # whatever the padding's u
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
    row_width: int | None = None,
    vocab: int | None = None,
) -> jax.Array:
    """Token per row of ``logits`` ``[rows, G, W]`` (rows <= 8, W a multiple
    of 128), sampling with uniforms ``recover_u`` f32 ``[rows, G, W]``, or
    int32 ``[rows]`` stream keys whose uniforms the kernel makes
    (`uniform_from_key`; no draw read from HBM). Vocab
    id ``g * row_width + w`` sits at ``[g, w]``: ``row_width`` W (default) for
    a dense ``[vocab / 128, 128]`` slab, or a vocab-parallel shard's width
    for the shards side by side, columns past it padding; ids at or past
    ``vocab`` (default ``G * row_width``) are padding too. ``params`` f32
    ``[2, rows]``: temperature (0: greedy) and top_p (>= 1: no top-p);
    ``draft`` int32 ``[rows]``: the rejected draft to exclude (-1: none).
    Returns int32 ``[8, 128]``, row r's token in sublane r."""
    rows, groups, lanes = logits.shape
    if lanes % 128 or not supports(rows, groups * lanes):
        raise ValueError(f"top_p_rows does not take logits {logits.shape}")
    width = lanes if row_width is None else row_width
    keyed = recover_u.ndim == 1
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    return pl.pallas_call(
        functools.partial(
            _kernel,
            rows=rows,
            shape=(groups, lanes),
            width=width,
            vocab=groups * width if vocab is None else vocab,
            keyed=keyed,
        ),
        in_specs=[smem, smem, vmem, smem if keyed else vmem],
        out_specs=vmem,
        out_shape=jax.ShapeDtypeStruct((ROWS, 128), jnp.int32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=48 << 20),
        interpret=interpret,
    )(params, draft, logits, recover_u)
