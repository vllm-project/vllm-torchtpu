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
"""Non-greedy spec-decode verify sampling from per-vocab-shard candidates.

With the LM head's vocabulary split over the TP ranks, the non-greedy verify
path all-gathers every logits row and runs temperature, top-k/top-p, softmax,
the accept test and recovery sampling over the whole vocabulary on every rank.
When every sampled row keeps at most top_k tokens, only tokens that can pass
the top-k cutoff matter: each rank takes the ``kc`` largest logits of every row
of its vocabulary shard, the ranks exchange those (``kc x tp`` per row), and
the arithmetic of ``sample_from_logits`` and ``_random_rejection_sample_with_
segment`` runs on the candidates with one random draw per candidate.

The candidates hold every token the top-k cutoff keeps when every shard's
smallest candidate, after temperature scaling, lies below the row's cutoff: a
token missing from its shard's candidates is no larger than that shard's
smallest candidate. Then the kept tokens, their logits and the cutoff are the
same as on the full vocabulary, and tokens outside the candidates have zero
probability on both paths. A step with a row that fails the check (only ties
at the cutoff that fill one shard's candidates do) runs the same arithmetic on
the gathered vocabulary inside the same program.

Every function below mirrors its torch counterpart in
``layers/adapter/sample/top_k_top_p.py``, ``rejection_sampler.py`` and
``TPUModelRunner.sample_from_logits`` op for op and dtype for dtype; argmax
ties go to the lowest token id, as on the full vocabulary.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

MASKED_LOGIT_VALUE = -1e12
TEMPERATURE_EPS = 1e-9
SAMPLING_EPS = 1e-5
_INT32_MAX = 2**31 - 1


def _sum(x: jax.Array, axis: int = -1) -> jax.Array:
    """Sum in the input dtype, the reduction torch emits for ``torch.sum``."""
    axis = axis % x.ndim
    return lax.reduce(x, jnp.zeros((), x.dtype), lax.add, (axis,))


def _int32_bsearch(batch_shape, predicate) -> jax.Array:
    """``top_k_top_p._int32_bsearch``."""
    current = jnp.zeros(batch_shape, jnp.int32)
    zero = jnp.zeros_like(current)
    sign_bit = jnp.full_like(current, -(1 << 31))
    current = current | jnp.where(predicate(current), sign_bit, zero)
    for i in range(31):
        bit = jnp.full_like(current, 1 << (30 - i))
        current = current | jnp.where(predicate(current | bit), zero, bit)
    return current


def _monotonic_bits(x: jax.Array) -> jax.Array:
    non_sign_bits = jnp.full_like(x, (1 << 31) - 1)
    return x ^ jnp.where(x < 0, non_sign_bits, jnp.zeros_like(x))


def _float32_bsearch(batch_shape, predicate) -> jax.Array:
    """``top_k_top_p._float32_bsearch``."""

    def int32_predicate(x):
        exponent_bits = jnp.full_like(x, (1 << 31) - (1 << 23))
        x = _monotonic_bits(x)
        is_finite = (x & exponent_bits) != exponent_bits
        return jnp.where(
            is_finite, predicate(lax.bitcast_convert_type(x, jnp.float32)), x >= 0
        )

    result = _int32_bsearch(batch_shape, int32_predicate)
    return lax.bitcast_convert_type(_monotonic_bits(result), jnp.float32)


def topk_cutoff(logits: jax.Array, k: jax.Array) -> jax.Array:
    """The value ``top_k_top_p.topk_mask`` keeps logits at or above."""
    k = k[..., 0]

    def predicate(threshold):
        neg_threshold = (-threshold)[..., None]
        count_gt = _sum((logits > neg_threshold).astype(jnp.int32))
        return count_gt >= k

    return -_float32_bsearch(logits.shape[:-1], predicate)


def topk_mask(logits: jax.Array, cutoff: jax.Array) -> jax.Array:
    replace = jnp.full_like(logits, MASKED_LOGIT_VALUE)
    return jnp.where(logits >= cutoff[..., None], logits, replace)


def softmax(x: jax.Array) -> jax.Array:
    """torch_tpu's softmax: computed in float32 for bf16 inputs."""
    xf = x.astype(jnp.float32)
    shifted = xf - jnp.max(xf, axis=-1, keepdims=True)
    e = jnp.exp(shifted)
    return (e / _sum(e)[..., None]).astype(x.dtype)


def topp_mask(logits: jax.Array, p: jax.Array) -> jax.Array:
    """``top_k_top_p.topp_mask``."""
    p = p[..., 0]
    probs = softmax(logits)
    zero = jnp.zeros_like(probs)

    def predicate(threshold):
        threshold = threshold[..., None]
        return _sum(jnp.where(probs >= threshold, probs, zero)) < p

    threshold = _float32_bsearch(logits.shape[:-1], predicate)[..., None]
    replace = jnp.full_like(logits, MASKED_LOGIT_VALUE)
    return jnp.where(probs >= threshold, logits, replace)


def topk_cutoff_pairwise(logits: jax.Array, k: jax.Array) -> jax.Array:
    """``topk_cutoff`` of a short row from all pairwise comparisons: the
    largest value with at least k values at or above it."""
    x = logits.astype(jnp.float32)
    at_or_above = _sum((x[..., None, :] >= x[..., :, None]).astype(jnp.int32))
    return jnp.max(jnp.where(at_or_above >= k, x, -jnp.inf), axis=-1)


def topp_mask_pairwise(logits: jax.Array, p: jax.Array):
    """``topp_mask`` of a short row in one pass, and whether some threshold
    keeps mass p (else the bisection's threshold is not a value of the row).

    The bisection's predicate at any threshold equals its predicate at the
    smallest value of the row at or above it (same masked row, same sum), so
    its threshold is the largest value of the row whose mass at or above it
    is not below p; every value's mass is one masked sum here.
    """
    probs = softmax(logits)
    zero = jnp.zeros_like(probs)
    kept = probs[..., None, :] >= probs[..., :, None]
    mass = _sum(jnp.where(kept, probs[..., None, :], zero[..., None, :]))
    ok = ~(mass < p)
    threshold = jnp.max(jnp.where(ok, probs, -jnp.inf), axis=-1, keepdims=True)
    replace = jnp.full_like(logits, MASKED_LOGIT_VALUE)
    return jnp.where(probs >= threshold, logits, replace), jnp.any(ok, axis=-1)


def apply_top_k_top_p(logits, top_k, top_p, pairwise=False):
    """``top_k_top_p.apply_top_k_top_p``; also returns the top-k cutoff and
    whether the top-p threshold is a value of the row (always with
    ``pairwise=False``). ``pairwise`` = one-pass searches for short rows."""
    if pairwise:
        cutoff = topk_cutoff_pairwise(logits, top_k)
    else:
        cutoff = topk_cutoff(logits, top_k)
    masked = jnp.where(top_k > 0, topk_mask(logits, cutoff), logits)
    if pairwise:
        topp_masked, topp_ok = topp_mask_pairwise(masked, top_p)
    else:
        topp_masked, topp_ok = topp_mask(masked, top_p), True
    masked = jnp.where(top_p < 1.0, topp_masked, masked)
    return masked, cutoff, (top_p[..., 0] >= 1.0) | topp_ok


def _clamp_uniform(u: jax.Array) -> jax.Array:
    """``torch.clamp(u, min=finfo.tiny, max=1.0 - finfo.eps)`` in u's dtype."""
    fi = jnp.finfo(u.dtype)
    lo = jnp.asarray(float(fi.tiny), u.dtype)
    hi = jnp.asarray(1.0 - float(fi.eps), u.dtype)
    return jnp.clip(u, lo, hi)


def argmax_lowest_id(x: jax.Array, ids: jax.Array) -> jax.Array:
    """Token id of the row maximum, lowest id on ties."""
    best = jnp.max(x, axis=-1, keepdims=True)
    return jnp.min(jnp.where(x == best, ids, _INT32_MAX), axis=-1)


def _bonus_scale(x, temps):
    safe = jnp.where(temps == 0.0, jnp.ones_like(temps), temps)
    return x / safe


def _target_scale(x, temps):
    return x.astype(jnp.float32) / (temps.astype(jnp.float32) + TEMPERATURE_EPS)


def sample_bonus(vals, ids, greedy, temps, top_k, top_p, u, pairwise=False):
    """``TPUModelRunner.sample_from_logits(all_greedy=False)`` over (vals, ids).

    vals: [B, W] logits, ids: [B, W] their token ids, greedy: [B] first argmax
    of the full rows, temps (logits dtype) / top_k / top_p: [B, 1], u: [B, W]
    uniforms. Returns the token ids [B], the top-k cutoffs [B] and whether
    each row's top-p threshold is one of its values [B].
    """
    is_greedy = temps <= SAMPLING_EPS
    masked, cutoff, topp_ok = apply_top_k_top_p(
        _bonus_scale(vals, temps), top_k, top_p, pairwise
    )
    noisy = masked + -jnp.log(-jnp.log(_clamp_uniform(u)))
    sampled = argmax_lowest_id(jnp.where(is_greedy, vals, noisy), ids)
    return jnp.where(is_greedy[:, 0], greedy, sampled), cutoff, topp_ok


def verify_targets(
    vals,
    ids,
    greedy,
    draft,
    temps,
    top_k,
    top_p,
    accept_u,
    recover_u,
    rates,
    synthetic,
    pairwise=False,
):
    """The per-slot decisions of ``_random_rejection_sample_with_segment``.

    vals / ids / recover_u: [T, W]; greedy, draft, accept_u, rates: [T];
    temps / top_k / top_p: [T, 1]. Returns accepted [T] bool, the recovered
    token ids [T], the top-k cutoffs [T] and whether each row's top-p
    threshold is one of its values [T].
    """
    valid = draft >= 0
    safe_draft = jnp.maximum(draft, 0)
    is_greedy = temps[:, 0] == 0
    masked, cutoff, topp_ok = apply_top_k_top_p(
        _target_scale(vals, temps), top_k, top_p, pairwise
    )
    probs = softmax(masked)
    is_draft = ids == safe_draft[:, None]
    draft_prob = jnp.sum(jnp.where(is_draft, probs, 0.0), axis=-1)
    if synthetic:
        accepted = accept_u < rates
    else:
        accepted = jnp.where(is_greedy, draft == greedy, accept_u < draft_prob)
    accepted = accepted & valid
    recovered_dist = jnp.where(is_draft & valid[:, None], 0.0, probs)
    exp_noise = -jnp.log(_clamp_uniform(recover_u))
    tiny = float(jnp.finfo(probs.dtype).tiny)
    recovered = argmax_lowest_id(recovered_dist / (exp_noise + tiny), ids)
    has_mass = jnp.any(recovered_dist > 0, axis=-1)
    recovered = jnp.where(has_mass, recovered, argmax_lowest_id(probs, ids))
    return accepted, jnp.where(is_greedy, greedy, recovered), cutoff, topp_ok


def rows_exact(scaled_boundary, is_greedy, top_k, cutoff, topp_ok=True):
    """Whether the candidates hold every token the row's top-k keeps.

    scaled_boundary: [R, tp] each shard's smallest candidate after the row's
    temperature scaling; cutoff: [R] the top-k cutoff over the candidates;
    topp_ok: [R] whether the top-p threshold is a value of the row.
    """
    below = jnp.all(scaled_boundary < cutoff[:, None], axis=-1)
    return is_greedy | ((top_k[:, 0] > 0) & below & topp_ok)


_LANES = 128
_SUB = 16


def _order_key(x: jax.Array) -> jax.Array:
    """int32 keys in [-2^15, 2^15) that order bf16 values as floats do."""
    bits = lax.bitcast_convert_type(x.astype(jnp.bfloat16), jnp.int16).astype(jnp.int32)
    return jnp.where(bits < 0, bits ^ 0x7FFF, bits)


def _kth_largest_key(keys: jax.Array, k: int) -> jax.Array:
    """Per row, the k-th largest key (counting repeats): 16 count passes."""
    t = jnp.full(keys.shape[:-1], -(1 << 15), jnp.int32)
    for bit in range(15, -1, -1):
        trial = t + (1 << bit)
        count = jnp.sum((keys >= trial[..., None]).astype(jnp.int32), axis=-1)
        t = jnp.where(count >= k, trial, t)
    return t


def _select_top(keys: jax.Array, k: int) -> jax.Array:
    """[R, k, N] one-hot of k positions per row holding the k largest keys:
    every key above the k-th largest, then its repeats in position order."""
    t = _kth_largest_key(keys, k)[:, None]
    above = keys > t
    need = k - jnp.sum(above.astype(jnp.int32), axis=-1, keepdims=True)
    tie = keys == t
    chosen = above | (tie & (jnp.cumsum(tie.astype(jnp.int32), axis=-1) <= need))
    slot = jnp.cumsum(chosen.astype(jnp.int32), axis=-1) - 1
    slots = lax.broadcasted_iota(jnp.int32, (1, k, 1), 1)
    return chosen[:, None, :] & (slot[:, None, :] == slots)


def _pick(onehot: jax.Array, values: jax.Array, fill) -> jax.Array:
    """values[r, n] at each one-hot slot [R, k, N] -> [R, k], by a max."""
    return jnp.max(jnp.where(onehot, values[:, None, :], fill), axis=-1)


def _gather_chunks(onehot: jax.Array, chunks: jax.Array) -> jax.Array:
    """[R, k, M] one-hot x [R, M, c] bf16 -> [R, k, c] (one exact term each)."""
    out = jnp.einsum(
        "rkm,rmc->rkc",
        onehot.astype(jnp.bfloat16),
        chunks,
        preferred_element_type=jnp.float32,
    )
    return out.astype(jnp.bfloat16)


def shard_candidates(logits, bounds, kc):
    """Per row of this rank's shard [R, Vl]: the ``kc`` largest logits [R, kc]
    with their token ids, the smallest of them, and the shard's (max, first
    id) pair. Every logit left out is no larger than the smallest candidate.

    The selection narrows 128-lane chunks to the ``kc`` chunks with the largest
    maxima (they hold the shard's ``kc`` largest logits), those to ``kc``
    16-lane chunks the same way, then picks ``kc`` of the remaining
    ``16 kc`` exactly; gathers are one-hot matmuls. Columns at or past the
    shard's vocabulary end are padding and never chosen.
    """
    rows, width = logits.shape
    assert width % _LANES == 0 and (kc * _LANES) % _SUB == 0, (width, kc)
    start, end = bounds[0], bounds[1]
    col = lax.broadcasted_iota(jnp.int32, logits.shape, 1)
    floor = jnp.finfo(jnp.bfloat16).min
    x = jnp.where(start + col < end, logits.astype(jnp.bfloat16), floor)
    top = jnp.max(x, axis=-1)
    first = jnp.min(jnp.where(x == top[:, None], col, _INT32_MAX), axis=-1)

    # Level 1: kc of the width / 128 lane chunks.
    m1 = width // _LANES
    x1 = x.reshape(rows, m1, _LANES)
    oh1 = _select_top(jnp.max(_order_key(x1), axis=-1), kc)
    chunk1 = _pick(oh1, lax.broadcasted_iota(jnp.int32, (rows, m1), 1), -1)
    s1 = _gather_chunks(oh1, x1)  # [R, kc, 128]
    # Level 2: kc of the 8 kc sub-chunks of 16 lanes.
    per = _LANES // _SUB
    m2 = kc * per
    x2 = s1.reshape(rows, m2, _SUB)
    oh2 = _select_top(jnp.max(_order_key(x2), axis=-1), kc)
    chunk2 = _pick(oh2, lax.broadcasted_iota(jnp.int32, (rows, m2), 1), -1)
    s2 = _gather_chunks(oh2, x2).reshape(rows, kc * _SUB)
    parent = jnp.max(
        jnp.where(
            (chunk2 // per)[:, :, None]
            == lax.broadcasted_iota(jnp.int32, (1, 1, kc), 2),
            chunk1[:, None, :],
            -1,
        ),
        axis=-1,
    )
    base = parent * _LANES + (chunk2 % per) * _SUB
    pos2 = (
        base[:, :, None] + lax.broadcasted_iota(jnp.int32, (1, 1, _SUB), 2)
    ).reshape(rows, kc * _SUB)
    # Level 3: the kc largest of the 16 kc logits left.
    oh3 = _select_top(_order_key(s2), kc)
    vals = _pick(oh3, s2, floor)
    ids = start + _pick(oh3, pos2, -1)
    return x, vals, ids, jnp.min(vals, axis=-1), top, start + first


def verify_local(
    bonus_logits,
    target_logits,
    bounds,
    draft,
    temps_b,
    top_k_b,
    top_p_b,
    temps_t,
    top_k_t,
    top_p_t,
    bonus_u,
    accept_u,
    recover_u,
    rates,
    seed,
    *,
    axis: str,
    kc: int,
    vocab_size: int,
    synthetic: bool,
    force_full: bool = False,
    full_uniforms=None,
):
    """Per-rank body. In: this rank's shard of the bonus rows [B, Vl] and of
    the verified rows [T, Vl]. Out (replicated over the TP group): bonus ids
    [B], accept flags [T] (int32), recovered ids [T], whether the candidates
    were exact (int32 scalar), and the candidate ids [B, C] / [T, C].

    bonus_u [B, C] / recover_u [T, C] are the draws for the candidate columns
    (C = kc x tp, shard-major); the full-vocabulary path draws its own from
    ``seed``, or uses ``full_uniforms`` = (bonus [B, V], recover [T, V]).
    """
    num_bonus = bonus_logits.shape[0]
    rows = jnp.concatenate([bonus_logits, target_logits], axis=0)
    x, vals, ids, boundary, top, first = shard_candidates(rows, bounds, kc)

    def gather(a):
        return lax.all_gather(a, axis, axis=1, tiled=True)

    c_vals, c_ids = gather(vals), gather(ids)
    c_bound, c_top, c_first = (
        gather(boundary[:, None]),
        gather(top[:, None]),
        gather(first[:, None]),
    )
    # First shard holding the row maximum: its first id is the global argmax.
    shard_ids = lax.broadcasted_iota(jnp.int32, c_top.shape, 1)
    best = jnp.max(c_top, axis=-1, keepdims=True)
    shard = jnp.min(jnp.where(c_top == best, shard_ids, _INT32_MAX), axis=-1)
    greedy = jnp.max(jnp.where(shard_ids == shard[:, None], c_first, -1), axis=-1)

    def split(a):
        return a[:num_bonus], a[num_bonus:]

    (vb, vt), (ib, it) = split(c_vals), split(c_ids)
    (bb, bt), (greedy_b, greedy_t) = split(c_bound), split(greedy)
    bonus, cut_b, ok_b = sample_bonus(
        vb, ib, greedy_b, temps_b, top_k_b, top_p_b, bonus_u, pairwise=True
    )
    accepted, recovered, cut_t, ok_t = verify_targets(
        vt,
        it,
        greedy_t,
        draft,
        temps_t,
        top_k_t,
        top_p_t,
        accept_u,
        recover_u,
        rates,
        synthetic,
        pairwise=True,
    )
    exact_b = rows_exact(
        _bonus_scale(bb, temps_b), temps_b[:, 0] <= SAMPLING_EPS, top_k_b, cut_b, ok_b
    )
    exact_t = rows_exact(
        _target_scale(bt, temps_t), temps_t[:, 0] == 0, top_k_t, cut_t, ok_t
    )
    exact = jnp.all(exact_b) & jnp.all(exact_t)
    if force_full:
        exact = jnp.zeros_like(exact)

    def candidates():
        return bonus, accepted.astype(jnp.int32), recovered

    def full():
        full_rows = gather(x)[:, :vocab_size]
        rows_b, rows_t = split(full_rows)
        full_ids = lax.broadcasted_iota(jnp.int32, (1, vocab_size), 1)
        if full_uniforms is not None:
            u_b, u_r = full_uniforms
        else:
            u_b, u_r = full_vocab_uniforms(
                seed, num_bonus, rows_t.shape[0], vocab_size, bonus_u.dtype
            )
        f_bonus, _, _ = sample_bonus(
            rows_b, full_ids, greedy_b, temps_b, top_k_b, top_p_b, u_b
        )
        f_acc, f_rec, _, _ = verify_targets(
            rows_t,
            full_ids,
            greedy_t,
            draft,
            temps_t,
            top_k_t,
            top_p_t,
            accept_u,
            u_r,
            rates,
            synthetic,
        )
        return f_bonus, f_acc.astype(jnp.int32), f_rec

    out_bonus, out_acc, out_rec = lax.cond(exact, candidates, full)
    return out_bonus, out_acc, out_rec, exact.astype(jnp.int32), ib, it


def full_vocab_uniforms(seed, num_bonus, num_targets, vocab_size, bonus_dtype):
    """The full-vocabulary path's draws: bonus [B, V], recovery [T, V]."""
    key = jax.random.fold_in(jax.random.fold_in(jax.random.key(0), seed[0]), seed[1])
    k_b, k_r = jax.random.split(key)
    u_b = jax.random.uniform(k_b, (num_bonus, vocab_size), dtype=bonus_dtype)
    u_r = jax.random.uniform(k_r, (num_targets, vocab_size), dtype=jnp.float32)
    return u_b, u_r


def candidate_verify(*operands, mesh, axis, kc, vocab_size, synthetic):
    """Global-view entry: every operand is the ranks' local arrays stacked on
    one leading axis [tp, ...], and so are the six results."""
    spec = P(axis)
    local = functools.partial(
        verify_local, axis=axis, kc=kc, vocab_size=vocab_size, synthetic=synthetic
    )

    def body(*args):
        return tuple(o[None] for o in local(*(a[0] for a in args)))

    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(spec,) * len(operands),
        out_specs=(spec,) * 6,
        check_vma=False,
    )(*operands)
