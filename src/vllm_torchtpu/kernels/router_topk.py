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
"""Sort-free top-k expert selection for the MoE router (Pallas).

``torch.topk`` lowers to a full ``stablehlo.sort`` of every expert plus a
slice, so at 512 experts it runs a 45-stage bitonic network over three arrays
to keep 10 values. This kernel selects the same 10 with ``k`` max-and-mask
passes over one VMEM-resident block: ``O(k*N)`` instead of ``O(N*log^2 N)``.

With softmax scoring and renormalization, ``select_softmax_renormalized``
takes the logits instead: softmax is monotone, so the top-k logits name the
same experts, and the renormalized weights are the softmax of the k selected
logits alone. That skips the full-width softmax and the renormalization ops.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Clamp floor. Below any real routing score (softmax, sigmoid and sqrtsoftplus
# are all >= 0) and strictly above the dtype minimum used to mask a selected
# column, so a row's maximum is always above the mask.
NEG = -3.0e38

# Rows per grid step, measured fastest at 512 experts. The grid rounds up, so
# this does not have to divide the row count.
MAX_BLOCK_ROWS = 512


def _select_kernel(x_ref, w_ref, i_ref, *, topk: int, n: int):
    """One block of rows: top ``topk`` by repeated max-and-mask.

    Scores at or below ``NEG`` are clamped up to it first, so every row has a
    maximum some column equals and every index written is a real expert. Ties
    go to the lowest id.

    The clamp tests ``x > NEG`` rather than ``isnan(x)`` to catch all three
    spellings of the hazard -- NaN, ``-inf``, ``-FLT_MAX`` -- any of which a
    whole row can be made of. Without it such a row would stay tied with the
    ``finfo.min`` mask and every pass would return column 0.
    """
    x = jnp.where(x_ref[...] > NEG, x_ref[...], NEG)
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        # Lowest column attaining the max; the rest fill with the last expert
        # so the min never selects a phantom.
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        w_ref[:, j] = m[:, 0]
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(iota == idx[:, None], masked, x)
    # No score above the sentinel means no real maximum: keep the row
    # poisonous rather than letting renormalization make it plausible.
    w_ref[...] = jnp.where(w_ref[:, :1] <= NEG, jnp.nan, w_ref[...])


def select(
    scores: jax.Array, topk: int, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Top-k over ``[rows, experts]`` f32 scores, values descending.

    Returns ``(weights f32, indices int32)`` of shape ``[rows, topk]``, values
    descending. ``interpret`` runs the kernel on the host, for tests without a
    TPU.

    The last grid step reads a partial block when ``rows`` is not a multiple
    of the block; Mosaic discards the stores past ``rows``, and every
    reduction is along the expert axis, so surplus rows cannot perturb a real
    one.
    """
    rows, n = scores.shape
    block = min(MAX_BLOCK_ROWS, rows)
    return pl.pallas_call(
        functools.partial(_select_kernel, topk=topk, n=n),
        grid=(pl.cdiv(rows, block),),
        in_specs=[pl.BlockSpec((block, n), lambda i: (i, 0))],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        interpret=interpret,
    )(scores)


def router_topk(scores: jax.Array, topk: int) -> tuple[jax.Array, jax.Array]:
    """``select`` with a jax_op-compatible signature (arrays, then statics)."""
    return select(scores, topk)


# XOR-ing an f32's magnitude bits with its sign makes signed int32 order agree
# with float order.
_MAGNITUDE = jnp.iinfo(jnp.int32).max
# Packed keys are positive normal f32s from the smallest normal up, ordered as
# their bit patterns; they stay below +inf's while they fit in 30 bits.
_KEY_BASE = 1 << jnp.finfo(jnp.float32).nmant
_KEY_BITS = 30


def _key_layout(dtype, n: int) -> tuple[int, int, int] | None:
    """``(value_bits, zero_bits, id_bits)`` of the packed key of an upcast
    ``dtype`` logit, or None when its value and ``n`` ids do not fit: the
    upcast leaves the low ``zero_bits`` of the f32 zero."""
    zero_bits = jnp.finfo(jnp.float32).nmant - jnp.finfo(dtype).nmant
    value_bits = 32 - zero_bits
    id_bits = max(n - 1, 1).bit_length()
    if value_bits + id_bits > _KEY_BITS:
        return None
    return value_bits, zero_bits, id_bits


def _select_packed(x, topk, slot, layout):
    """Top ``topk`` by one f32 max per pass over keys that pack the logit's
    rank with ``id_mask - id``: the max is the largest logit and, among equal
    ones, the lowest id. Returns the selected logits and ids."""
    value_bits, zero_bits, id_bits = layout
    # One spelling each: -0.0 must tie +0.0, and a negative NaN rank first.
    x = jnp.where(x == 0, 0.0, jnp.where(jnp.isnan(x), jnp.nan, x))
    bits = jax.lax.bitcast_convert_type(x, jnp.int32)
    rank = jnp.right_shift(bits ^ (jnp.right_shift(bits, 31) & _MAGNITUDE), zero_bits)
    rank = rank + (1 << (value_bits - 1))
    id_mask = (1 << id_bits) - 1
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    key = _KEY_BASE + (jnp.left_shift(rank, id_bits) | (id_mask - iota))
    key = jax.lax.bitcast_convert_type(key, jnp.float32)
    best = jnp.zeros(slot.shape, jnp.float32)
    for j in range(topk):
        m = jnp.max(key, axis=1, keepdims=True)
        best = jnp.where(slot == j, m, best)
        # Below every key, and keys are distinct: only this column matches.
        key = jnp.where(key == m, 0.0, key)
    best = jax.lax.bitcast_convert_type(best, jnp.int32) - _KEY_BASE
    ids = id_mask - (best & id_mask)
    rank = jnp.right_shift(best, id_bits) - (1 << (value_bits - 1))
    bits = jnp.left_shift(rank, zero_bits)
    bits = (bits ^ (jnp.right_shift(bits, 31) & _MAGNITUDE)) & ~((1 << zero_bits) - 1)
    return jax.lax.bitcast_convert_type(bits, jnp.float32), ids


def _select_two_pass(x, topk, slot):
    """Top ``topk`` by an f32 max then the lowest matching id per pass.
    Returns the selected logits, -inf mapped to the dtype minimum, and ids."""
    # -inf moves up so that -inf can mark a taken column.
    x = jnp.where(x == -jnp.inf, jnp.finfo(jnp.float32).min, x)
    n = x.shape[1]
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1).astype(jnp.float32)
    best = jnp.zeros(slot.shape, jnp.float32)
    ids = jnp.zeros(slot.shape, jnp.float32)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n), axis=1, keepdims=True)
        best = jnp.where(slot == j, m, best)
        ids = jnp.where(slot == j, idx, ids)
        x = jnp.where(iota == idx, -jnp.inf, x)
    return best, ids.astype(jnp.int32)


def _softmax_select_kernel(x_ref, w_ref, i_ref, *, topk: int):
    """One block of rows: top ``topk`` logits and their softmax.

    Every reduction is in f32: Mosaic's int32 lane max measured about four
    times slower per pass at [48, 768]. Ties go to the lowest id, as in
    ``_select_kernel`` on the softmax, where equal logits score equal.

    A NaN, a +inf, or a row of -inf makes every weight NaN, as the softmax
    does: NaN ranks with or above +inf, and the max logit then subtracts to
    NaN.
    """
    x = x_ref[...].astype(jnp.float32)
    slot = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], topk), 1)
    layout = _key_layout(x_ref.dtype, x.shape[1])
    if layout is None:
        x = jnp.where(jnp.isnan(x), jnp.inf, x)
        top = jnp.max(x, axis=1, keepdims=True)
        logits, ids = _select_two_pass(x, topk, slot)
    else:
        logits, ids = _select_packed(x, topk, slot, layout)
        top = logits[:, :1]
    e = jnp.exp(logits - top)
    w_ref[...] = e / jnp.sum(e, axis=1, keepdims=True)
    i_ref[...] = ids


def select_softmax_renormalized(
    logits: jax.Array, topk: int, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Top-k of ``softmax(logits)`` over ``[rows, experts]``, renormalized.

    Returns ``(weights f32, indices int32)`` of shape ``[rows, topk]``, values
    descending: the experts ``select`` picks from the f32 softmax of
    ``logits``, and their scores divided by their sum, up to f32 rounding.
    Where f32 rounding makes distinct softmax scores equal (an underflow to
    0, for instance), this picks by logit and the softmax path by lowest id;
    the weights agree. ``logits`` are read in their own dtype: bf16 logits
    take one reduction per pass, f32 logits two.
    """
    rows, n = logits.shape
    if topk > n:
        raise ValueError(f"topk k={topk} exceeds the expert count {n}")
    block = min(MAX_BLOCK_ROWS, rows)
    return pl.pallas_call(
        functools.partial(_softmax_select_kernel, topk=topk),
        grid=(pl.cdiv(rows, block),),
        in_specs=[pl.BlockSpec((block, n), lambda i: (i, 0))],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        interpret=interpret,
    )(logits)


def router_softmax_topk(logits: jax.Array, topk: int) -> tuple[jax.Array, jax.Array]:
    """``select_softmax_renormalized`` with a jax_op-compatible signature."""
    return select_softmax_renormalized(logits, topk)
