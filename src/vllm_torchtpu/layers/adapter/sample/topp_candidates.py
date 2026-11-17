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
"""Top-p threshold from a candidate subset of the vocabulary
(TPU_SAMPLER_TOPP_CANDIDATES).

`top_k_top_p.topp_mask` finds the top-p probability threshold with a 32-step
bisection, and every step reduces over the whole vocabulary: 32 passes over
``[rows, vocab]`` float32 probabilities, about 50 us per sampled step at
8 rows x 200064 on v7x. This module computes the same threshold from a few
thousand candidates and keeps the full search only as a fallback.

Why the candidates give the same threshold: split the vocabulary into blocks
of 128 and keep the ``num_blocks`` blocks with the largest maxima; let ``m``
be the smallest kept block maximum. Every probability above ``m`` lies in a
kept block (its block maximum is above ``m``, so that block ranks among the
kept ones). For a threshold ``t > m`` the set ``{probs >= t}`` is therefore
the same over the candidates and over the vocabulary. If the candidates above
``m`` already hold at least ``p`` of the mass, the full search's predicate
"mass(probs >= t) < p" is False for every ``t <= m``, so the search can treat
``t <= m`` as False and evaluate only candidates otherwise: it returns the
same float. Only the order of the float32 sums differs (a boundary within
rounding of ``p`` can flip). A row whose candidates hold less than ``p``
above ``m`` (a very flat distribution) sends the whole call to the full
search via ``lax.cond``.

The candidate search resolves four bits of the float32 pattern per step (15
thresholds at once), so it takes 8 steps instead of 32; for a monotonic
predicate it returns what the bit-by-bit search returns.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

BLOCK = 128

_SIGN = -(1 << 31)
_NON_SIGN = (1 << 31) - 1
_EXPONENT = (1 << 31) - (1 << 23)
# (shift, bits) per search step after the sign bit: 31 bits as 3 + 7 x 4.
_RADIX16_STEPS = ((28, 3),) + tuple((s, 4) for s in range(24, -1, -4))
_BINARY_STEPS = tuple((30 - i, 1) for i in range(31))


def _pattern_to_f32(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Monotonic int32 -> (float32, its bit pattern); `top_k_top_p`'s map."""
    bits = x ^ jnp.where(x < 0, jnp.int32(_NON_SIGN), jnp.int32(0))
    return lax.bitcast_convert_type(bits, jnp.float32), bits


def _search(rows: int, mass_below_p, steps) -> jax.Array:
    """Largest float32 per row whose predicate is False, as
    `top_k_top_p._float32_bsearch` defines it. ``mass_below_p(t)`` maps
    finite thresholds ``[rows, d]`` to bool ``[rows, d]``."""

    def predicate(x: jax.Array) -> jax.Array:
        t, bits = _pattern_to_f32(x)
        finite = (bits & jnp.int32(_EXPONENT)) != jnp.int32(_EXPONENT)
        return jnp.where(finite, mass_below_p(t), bits >= 0)

    cur = jnp.zeros((rows,), jnp.int32)
    cur = jnp.where(predicate(cur[:, None])[:, 0], jnp.int32(_SIGN), cur)
    for shift, nbits in steps:
        digits = jnp.arange(1, 1 << nbits, dtype=jnp.int32) << shift
        false = ~predicate(cur[:, None] | digits[None, :])
        cur = cur | (jnp.sum(false.astype(jnp.int32), axis=1) << shift)
    return _pattern_to_f32(cur)[0]


def _mass_below_p(values: jax.Array, p: jax.Array):
    def fn(t: jax.Array) -> jax.Array:
        kept = values[:, None, :] >= t[:, :, None]
        mass = jnp.sum(jnp.where(kept, values[:, None, :], 0.0), axis=-1)
        return mass < p[:, None]

    return fn


def topp_threshold_full(probs: jax.Array, p: jax.Array) -> jax.Array:
    """The full-vocabulary bit-by-bit search (`topp_mask`'s threshold)."""
    return _search(probs.shape[0], _mass_below_p(probs, p), _BINARY_STEPS)


def _radix_limit() -> int:
    """Candidates up to this many search 4 bits per step (15 thresholds at
    once); larger sets search bit by bit."""
    return 16384


def _tier(probs: jax.Array, p: jax.Array, num_blocks: int, otherwise):
    rows, vocab = probs.shape
    blocks = probs.reshape(rows, vocab // BLOCK, BLOCK)
    top_max, top_ids = lax.top_k(jnp.max(blocks, axis=-1), num_blocks)
    floor = top_max[:, -1]
    cand = jnp.take_along_axis(blocks, top_ids[:, :, None], axis=1).reshape(rows, -1)
    above = jnp.sum(jnp.where(cand > floor[:, None], cand, 0.0), axis=-1)
    covered = (above >= p) | (p >= 1.0)
    cand_mass = _mass_below_p(cand, p)
    steps = _RADIX16_STEPS if cand.shape[1] <= _radix_limit() else _BINARY_STEPS

    def fast():
        return _search(rows, lambda t: cand_mass(t) & (t > floor[:, None]), steps)

    return lax.cond(jnp.all(covered), fast, otherwise), jnp.all(covered)


def topp_threshold(probs: jax.Array, p: jax.Array, num_blocks=(64,)) -> jax.Array:
    """`topp_mask`'s threshold per row from candidate blocks: the first tier
    of ``num_blocks`` whose candidates cover every row, else the full search.

    probs: float32 ``[rows, vocab]``, vocab a multiple of ``BLOCK``.
    p: float32 ``[rows]``. Rows with ``p >= 1`` get an unspecified value
    (the caller does not apply top-p to them)."""
    if isinstance(num_blocks, int):
        num_blocks = (num_blocks,)
    nb = probs.shape[1] // BLOCK
    tiers = [b for b in num_blocks if b <= nb]

    def run(k):
        if k == len(tiers):
            return topp_threshold_full(probs, p)
        return _tier(probs, p, tiers[k], lambda: run(k + 1))[0]

    return run(0)


def topp_threshold_traced(probs: jax.Array, p: jax.Array, num_blocks=(64,)):
    """`topp_threshold` plus whether the first tier covered every row."""
    if isinstance(num_blocks, int):
        num_blocks = (num_blocks,)
    first = _tier(probs, p, num_blocks[0], lambda: topp_threshold_full(probs, p))[1]
    return topp_threshold(probs, p, num_blocks), first
