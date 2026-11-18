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
"""Non-greedy speculative-decoding verify on a vocab-sharded LM head.

With a tensor-parallel (TP) LM head, each rank computes the logits of one
contiguous slice of the vocabulary (its vocab shard). The usual verify
all-gathers those shards into full-vocab logits and repeats the full-vocab
sampling work on every rank. ``spec_verify`` instead keeps every vocab-sized
tensor shard-local and exchanges only per-row values between the ranks:

1. Softmax statistics: one all-gather of the per-shard (max, sum of exp) of
   each row gives the global max and normalizer on every rank.
2. Top-k cutoff (only when some row sets top_k): the bisection of
   ``top_k_top_p.topk_mask`` with each count all-reduced across ranks.
   Counts are integers, so the cutoff is exactly the one ``topk_mask`` finds.
3. Top-p threshold (only when some row sets top_p < 1): the search of
   ``top_k_top_p.topp_mask`` over the float32 bit pattern of the threshold,
   ``digit_bits`` bits per pass, each pass one all-reduce of the candidate
   masses ``[2**digit_bits - 1, rows]``.
4. Token draw: Gumbel-max on the masked shard (the recovered token excludes
   the draft token), then one all-gather of the per-shard best (score,
   token id) pair, the per-shard greedy (logit, token id) pair and the
   target probability of the draft token (non-zero only on the rank that
   owns the token). Each rank then picks the best shard.

The bonus rows (one per request) and the verify rows (one per draft slot)
run through the same steps together, so a verify chunk costs
1 + (top-k passes) + (top-p passes) + 1 small collectives.

Equivalence with the usual sampler
(``tpu_runner._sample_from_logits_impl`` for the bonus rows,
``rejection_sampler._random_rejection_sample_with_segment_impl`` for the
verify rows):

- Greedy rows give the same tokens: the shard argmax keeps the first
  maximum of its shard and the cross-shard argmax keeps the first maximal
  shard, which holds the lower token ids. This is ``torch.argmax`` over the
  full logits, ties included.
- The sampling distribution is the same, up to float32 summation order.
  The softmax normalizer and the top-p masses are sums over shards, so a
  top-p boundary whose mass lies within rounding of top_p can flip. The
  bonus rows round probabilities and masses to the logits dtype (bfloat16
  on TPU), as the usual bonus sampler does by running its softmax and sums
  on bfloat16 tensors; this assumes the usual sampler accumulates those
  sums in float32 and rounds once.
- The random draws differ. The usual sampler draws one uniform per
  (row, vocab entry) from the sampling generator. Here each rank needs
  independent noise for its own columns only, so the uniforms come from a
  counter-based hash of (two per-step seeds, row, global token id). The
  seeds are drawn from the sampling generator, so every rank uses the same
  seeds. Sampled tokens are therefore not bitwise those of the usual
  sampler. The accept uniforms are drawn exactly as before.
- The recovered token uses Gumbel-max on the masked scaled logits where the
  usual sampler uses the exponential race argmax(p / E) on the probabilities.
  Both sample token i with probability proportional to exp(logit_i) over the
  kept tokens.
"""

from collections.abc import Callable

import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    TEMPERATURE_EPS,
    _assemble_selected_tokens,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import _float32_bsearch

# Bonus rows with a temperature at or below this are greedy. Same value as
# tpu_runner.SAMPLING_EPS, which the usual bonus sampler uses.
SAMPLING_EPS = 1e-5

# Columns of the final per-row exchange (see step 4 of the module docstring).
_SCORE, _SCORE_ID, _LOGIT, _LOGIT_ID, _DRAFT_PROB = range(5)

AllReduce = Callable[[torch.Tensor], torch.Tensor]
AllGather = Callable[[torch.Tensor], torch.Tensor]


def tp_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Sum of ``x`` over the TP group (vLLM's TP all-reduce)."""
    from vllm.distributed import tensor_model_parallel_all_reduce

    return tensor_model_parallel_all_reduce(x)


def tp_all_gather(x: torch.Tensor) -> torch.Tensor:
    """``x`` of every TP rank, stacked by rank: [n, ...] -> [tp, n, ...]."""
    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )

    tp = get_tensor_model_parallel_world_size()
    return tensor_model_parallel_all_gather(x, dim=0).view(tp, *x.shape)


# Constants of the MurmurHash3 32-bit finalizer and two odd multipliers,
# written as signed int32 (the tensors below are int32; products wrap).
_FMIX_M1 = 0x85EBCA6B - (1 << 32)
_FMIX_M2 = 0xC2B2AE35 - (1 << 32)
_COL_MUL = 0x9E3779B1 - (1 << 32)
_ROW_MUL = 0x27D4EB2F


def _shift_right_logical(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Logical right shift of int32 values (``>>`` on int32 is arithmetic)."""
    return (x >> bits) & ((1 << (32 - bits)) - 1)


def _fmix32(h: torch.Tensor) -> torch.Tensor:
    """MurmurHash3's 32-bit finalizer: a bijection on 32-bit words whose
    output bits each depend on every input bit."""
    h = h ^ _shift_right_logical(h, 16)
    h = h * _FMIX_M1
    h = h ^ _shift_right_logical(h, 13)
    h = h * _FMIX_M2
    return h ^ _shift_right_logical(h, 16)


def hashed_uniform(
    seeds: torch.Tensor, rows: int, token_ids: torch.Tensor
) -> torch.Tensor:
    """Uniforms for every (row, token id), from two int32 seeds.

    The value of entry (r, j) depends only on (seeds, r, token_ids[j]), so
    the ranks of a TP group, given the same seeds, draw independent noise for
    their disjoint token ids without communicating. Each value is a multiple
    of 2**-24 in [0, 1 - 2**-24], the values ``torch.rand`` returns in
    float32, clamped to [float32 tiny, 1 - float32 eps] as the usual sampler
    clamps its uniforms. The clamp keeps the Gumbel noise
    -log(-log(u)) finite: u = 1.0 would give +inf.

    Args:
      seeds: int32 [2], the same on every rank.
      rows: number of rows.
      token_ids: int32 [width], the global token id of each column.

    Returns:
      float32 [rows, width].
    """
    col = _fmix32(token_ids * _COL_MUL + seeds[0])
    row = _fmix32(
        torch.arange(rows, dtype=torch.int32, device=token_ids.device) * _ROW_MUL
        + seeds[1]
    )
    h = _fmix32(col.unsqueeze(0) ^ row.unsqueeze(1))
    u = _shift_right_logical(h, 8).to(torch.float32) * (2.0**-24)
    finfo = torch.finfo(torch.float32)
    return torch.clamp(u, min=finfo.tiny, max=1.0 - finfo.eps)


def _softmax_stats(
    x: torch.Tensor, all_gather: AllGather
) -> tuple[torch.Tensor, torch.Tensor]:
    """Global max and sum of exp(x - max) per row of a vocab-sharded [R, W].

    Entries at -inf (masked or padding columns) contribute nothing. A rank
    whose row is all -inf sends sum 0 and its max is ignored.
    """
    local_max = x.amax(dim=-1)
    shift = torch.where(torch.isfinite(local_max), local_max, 0.0)
    local_sum = torch.exp(x - shift.unsqueeze(-1)).sum(dim=-1)
    g = all_gather(torch.stack([local_max, local_sum], dim=-1))  # [tp, R, 2]
    gmax = g[..., 0].amax(dim=0)
    gmax = torch.where(torch.isfinite(gmax), gmax, 0.0)
    has_mass = g[..., 1] > 0
    scale = torch.exp(torch.where(has_mass, g[..., 0] - gmax, 0.0))
    gsum = torch.where(has_mass, g[..., 1] * scale, 0.0).sum(dim=0)
    return gmax, gsum


def _top_k_cutoff(
    x: torch.Tensor, k: torch.Tensor, all_reduce: AllReduce, radix_bits: int
) -> torch.Tensor:
    """The cutoff of ``top_k_top_p.topk_mask`` for a vocab-sharded [R, W].

    Same search and same predicate as ``topk_mask``; the count of entries
    above each candidate is summed over the ranks (exact: integers below
    2**24 in float32). Keep entries >= the returned cutoff [R].
    """

    def predicate(threshold: torch.Tensor) -> torch.Tensor:
        neg_threshold = (-threshold).unsqueeze(-1)
        local = torch.sum((x > neg_threshold).to(torch.float32), dim=-1)
        return all_reduce(local) >= k

    return -_float32_bsearch(k.shape, predicate, x.device, radix_bits)


def _top_p_threshold(
    probs: torch.Tensor,
    p: torch.Tensor,
    lattice: torch.Tensor,
    lattice_dtype: torch.dtype,
    all_reduce: AllReduce,
    digit_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The threshold of ``top_k_top_p.topp_mask`` for vocab-sharded probs.

    ``topp_mask`` keeps probs >= t, where t is the largest float32 at which
    mass(t) = sum(probs >= t) is still >= p. Its bisection runs over the 32
    bits of t. Here:

    - The sign bit and bit 30 are settled without a pass. mass(0.0) is the
      total mass. If it is below p (only when p is within rounding of 1), t
      is negative and every entry is kept, as in ``topp_mask``. Otherwise t
      is in [0, 1], whose bit patterns have bit 30 clear (bit 30 alone is
      2.0, where the mass is 0 < p).
    - Bits 29..0 are found ``digit_bits`` at a time. A pass evaluates the
      2**digit_bits - 1 candidates of the next digit at once; the digit is
      the number of candidates whose mass is still >= p. The mass decreases
      as t grows, so this is the digit the 1-bit bisection reaches.
    - The total mass rides as one more candidate in the first pass.

    Example with digit_bits 5: passes resolve bits 29..25, 24..20, ...,
    4..0, so 6 all-reduces of [32 or 31, rows] float32.

    Args:
      probs: float32 [R, W], this rank's shard of each row's probabilities.
      p: float32 [R], top-p of each row (in (0, 1]).
      lattice: bool [R], rows whose usual sampler sums ``lattice_dtype``
        probabilities; their masses are rounded to ``lattice_dtype`` before
        the comparison with p.
      lattice_dtype: dtype of the usual sampler's probabilities on those rows.
      all_reduce: sum over the TP ranks.
      digit_bits: bits resolved per pass, 1 to 8.

    Returns:
      (threshold [R], kept mass [R]): keep probs >= threshold; the kept
      mass is the float32 sum of the kept probabilities.
    """

    def masses(candidate_bits: torch.Tensor) -> torch.Tensor:
        t = candidate_bits.view(torch.float32).unsqueeze(-1)
        local = torch.where(probs >= t, probs, 0.0).sum(dim=-1)
        return all_reduce(local)

    def comparable(mass: torch.Tensor) -> torch.Tensor:
        return torch.where(lattice, mass.to(lattice_dtype).to(torch.float32), mass)

    rows = probs.shape[0]
    device = probs.device
    current = torch.zeros((rows,), dtype=torch.int32, device=device)
    total = None
    kept_mass = None
    hi = 30
    while hi > 0:
        width = min(digit_bits, hi)
        shift = hi - width
        num_cand = (1 << width) - 1
        digits = torch.arange(1, num_cand + 1, dtype=torch.int32, device=device)
        candidates = current.unsqueeze(0) | (digits << shift).unsqueeze(-1)
        if total is None:
            candidates = torch.cat([current.unsqueeze(0), candidates], dim=0)
        mass = masses(candidates)
        if total is None:
            total, mass = mass[0], mass[1:]
            kept_mass = total
        assert kept_mass is not None
        digit = torch.sum(comparable(mass) >= p, dim=0, dtype=torch.int32)
        picked = mass.gather(0, (digit - 1).clamp(min=0).unsqueeze(0)).squeeze(0)
        kept_mass = torch.where(digit > 0, picked, kept_mass)
        current = current | (digit << shift)
        hi = shift
    assert total is not None and kept_mass is not None
    keep_all = comparable(total) < p
    threshold = torch.where(
        keep_all,
        torch.full_like(p, float("-inf")),
        current.view(torch.float32),
    )
    return threshold, torch.where(keep_all, total, kept_mass)


def spec_verify(
    local_logits: torch.Tensor,
    vocab_start: torch.Tensor,
    vocab_valid: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
    req_temperatures: torch.Tensor,
    req_top_k: torch.Tensor,
    req_top_p: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    accept_u: torch.Tensor,
    noise_seeds: torch.Tensor,
    max_draft_tokens: int,
    synthetic_conditional_rates: torch.Tensor | None,
    synthetic_mode: bool,
    all_reduce: AllReduce,
    all_gather: AllGather,
    need_top_k: bool = True,
    need_top_p: bool = True,
    radix_bits: int = 1,
    digit_bits: int = 5,
    verify_rows: int = 0,
) -> torch.Tensor:
    """Bonus-token sampling and draft rejection of one verify chunk, on this
    rank's vocab shard (see the module docstring).

    Args:
      local_logits: [L, W] this rank's logits shard of the chunk's sampled
        positions (the LM head output before the TP gather). Column j is
        token vocab_start + j; columns from vocab_valid on are padding.
      vocab_start: int32 scalar tensor, global id of column 0.
      vocab_valid: int32 scalar tensor, real (non-padding) columns.
      bonus_logits_indices: int32 [B], the row of each request's bonus slot.
      target_logits_indices: int32 [T], the row of each draft slot.
      req_temperatures: [B, 1] in the logits dtype; 0 is greedy.
      req_top_k: int32 [B, 1]; 0 disables top-k.
      req_top_p: float32 [B, 1]; 1 disables top-p.
      draft_token_ids: int32 [T], negative for placeholder drafts.
      num_draft_tokens: int32 [B] declared drafts per request.
      segment_ids: [T] request of each draft slot.
      group_indices: int32 [T] position of each slot within its request.
      accept_u: float32 [T] accept uniforms.
      noise_seeds: int32 [2], seeds of the Gumbel noise, equal on all ranks.
      max_draft_tokens: drafts per request at most.
      synthetic_conditional_rates: per-position accept probabilities of
        synthetic acceptance, or None.
      synthetic_mode: static; accept by accept_u < rate instead of the target
        probability (the recovered token is still drawn from the target).
      all_reduce: sum over the TP ranks ([...] -> [...]).
      all_gather: stack over the TP ranks ([n, ...] -> [tp, n, ...]).
      need_top_k / need_top_p: static; False promises that no row sets
        top_k > 0 / top_p < 1, and the search is skipped.
      radix_bits: static; bits per pass of the top-k bisection.
      digit_bits: static; bits per pass of the top-p search, 1 to 8.
      verify_rows: static; keep only the first that many draft slots, which
        the caller guarantees hold every real slot (0 keeps all).

    Returns:
      int32 [B, max_draft_tokens + 1], the rejection sampler's layout.
    """
    if verify_rows:
        target_logits_indices = target_logits_indices[:verify_rows]
        draft_token_ids = draft_token_ids[:verify_rows]
        segment_ids = segment_ids[:verify_rows]
        group_indices = group_indices[:verify_rows]
        accept_u = accept_u[:verify_rows]
    device = local_logits.device
    dtype = local_logits.dtype
    num_bonus = bonus_logits_indices.shape[0]
    rows = num_bonus + target_logits_indices.shape[0]
    width = local_logits.shape[-1]

    bonus_logits = torch.index_select(local_logits, 0, bonus_logits_indices)
    target_logits = torch.index_select(local_logits, 0, target_logits_indices)
    verify_temperatures = req_temperatures[segment_ids]

    # Temperature scaling exactly as each usual sampler does it: the bonus
    # sampler divides in the logits dtype by a temperature with 0 replaced by
    # 1; the verify divides in float32 by temperature + TEMPERATURE_EPS.
    safe_temperatures = torch.where(
        req_temperatures == 0.0, torch.ones_like(req_temperatures), req_temperatures
    )
    bonus_scaled = (bonus_logits / safe_temperatures).to(torch.float32)
    verify_scaled = target_logits.to(torch.float32) / (
        verify_temperatures.to(torch.float32) + TEMPERATURE_EPS
    )
    is_greedy = torch.cat(
        [
            req_temperatures.squeeze(-1) <= SAMPLING_EPS,
            verify_temperatures.squeeze(-1) == 0,
        ]
    )
    is_bonus_row = torch.arange(rows, device=device) < num_bonus
    top_k = torch.cat([req_top_k, req_top_k[segment_ids]]).squeeze(-1)
    top_p = torch.cat([req_top_p, req_top_p[segment_ids]]).squeeze(-1)

    columns = torch.arange(width, dtype=torch.int32, device=device)
    real_column = (columns < vocab_valid).unsqueeze(0)
    token_ids = columns + vocab_start
    neg_inf = torch.tensor(float("-inf"), dtype=torch.float32, device=device)

    x = torch.where(real_column, torch.cat([bonus_scaled, verify_scaled]), neg_inf)
    raw = torch.where(
        real_column,
        torch.cat([bonus_logits, target_logits]).to(torch.float32),
        neg_inf,
    )

    if need_top_k:
        cutoff = _top_k_cutoff(x, top_k, all_reduce, radix_bits)
        drop = (top_k > 0).unsqueeze(-1) & (x < cutoff.unsqueeze(-1))
        x = torch.where(drop, neg_inf, x)

    gmax, gsum = _softmax_stats(x, all_gather)
    probs = torch.exp(x - gmax.unsqueeze(-1)) / gsum.unsqueeze(-1)
    norm = torch.ones_like(gsum)
    if need_top_p:
        # The usual bonus sampler's softmax output is in the logits dtype.
        probs = torch.where(
            is_bonus_row.unsqueeze(-1), probs.to(dtype).to(torch.float32), probs
        )
        threshold, kept_mass = _top_p_threshold(
            probs, top_p, is_bonus_row, dtype, all_reduce, digit_bits
        )
        apply_top_p = top_p < 1.0
        drop = apply_top_p.unsqueeze(-1) & (probs < threshold.unsqueeze(-1))
        x = torch.where(drop, neg_inf, x)
        norm = torch.where(apply_top_p, kept_mass, norm)

    # Target probability of each draft token after masking, from the rank
    # that owns the token (0 elsewhere): exp(x_d - max) / sum over kept.
    draft = draft_token_ids.to(torch.int64)
    valid_draft = draft >= 0
    draft_local = draft - vocab_start.to(torch.int64)
    in_shard = valid_draft & (draft_local >= 0) & (draft_local < vocab_valid)
    draft_col = draft_local.clamp(min=0, max=width - 1)
    x_verify = x[num_bonus:]
    x_draft = x_verify.gather(-1, draft_col.unsqueeze(-1)).squeeze(-1)
    draft_prob = torch.where(
        in_shard,
        torch.exp(x_draft - gmax[num_bonus:]) / (gsum[num_bonus:] * norm[num_bonus:]),
        0.0,
    )

    # Gumbel-max over the kept tokens. A verify row excludes its draft token:
    # the residual max(0, p - q) of a draft that put all its mass on one token
    # is p with that token removed.
    u = hashed_uniform(noise_seeds, rows, token_ids)
    gumbel = -torch.log(-torch.log(u))
    exclude_verify = in_shard.unsqueeze(-1) & (
        columns.unsqueeze(0).to(torch.int64) == draft_col.unsqueeze(-1)
    )
    exclude = torch.cat(
        [
            torch.zeros((num_bonus, width), dtype=torch.bool, device=device),
            exclude_verify,
        ]
    )
    # Masked entries stay -inf whatever the noise (no -inf + inf = NaN).
    score = torch.where(exclude | torch.isneginf(x), neg_inf, x + gumbel)

    # Token ids travel as float32, exact for ids below 2**24.
    score_idx = torch.argmax(score, dim=-1, keepdim=True)
    logit_idx = torch.argmax(raw, dim=-1, keepdim=True)
    # Field order: _SCORE, _SCORE_ID, _LOGIT, _LOGIT_ID, _DRAFT_PROB.
    payload = torch.stack(
        [
            torch.gather(score, -1, score_idx).squeeze(-1),
            (score_idx.squeeze(-1) + vocab_start).to(torch.float32),
            torch.gather(raw, -1, logit_idx).squeeze(-1),
            (logit_idx.squeeze(-1) + vocab_start).to(torch.float32),
            torch.cat(
                [
                    torch.zeros((num_bonus,), dtype=torch.float32, device=device),
                    draft_prob,
                ]
            ),
        ],
        dim=-1,
    )
    g = all_gather(payload)  # [tp, rows, 5]

    # The first maximal shard holds the lowest token id among equal maxima.
    best = torch.argmax(g[..., _SCORE], dim=0, keepdim=True)
    sampled = torch.gather(g[..., _SCORE_ID], 0, best).squeeze(0).to(torch.int32)
    has_mass = torch.gather(g[..., _SCORE], 0, best).squeeze(0) > float("-inf")
    best_logit = torch.argmax(g[..., _LOGIT], dim=0, keepdim=True)
    argmax_ids = (
        torch.gather(g[..., _LOGIT_ID], 0, best_logit).squeeze(0).to(torch.int32)
    )
    target_token_probs = g[:, num_bonus:, _DRAFT_PROB].sum(dim=0)

    bonus_token_ids = torch.where(
        is_greedy[:num_bonus], argmax_ids[:num_bonus], sampled[:num_bonus]
    )
    target_argmax = argmax_ids[num_bonus:]
    verify_greedy = is_greedy[num_bonus:]
    if synthetic_mode:
        assert synthetic_conditional_rates is not None
        rates = synthetic_conditional_rates[group_indices.to(torch.int64)]
        accepted = accept_u < rates
    else:
        # Accept iff u < p(x), as the usual verify does.
        accepted = torch.where(
            verify_greedy,
            draft == target_argmax.to(torch.int64),
            accept_u < target_token_probs,
        )
    accepted = accepted & valid_draft
    # A row with no residual mass (its only kept token is the draft) and a
    # greedy row recover the target argmax, as the usual verify does.
    recovered = torch.where(
        verify_greedy | ~has_mass[num_bonus:], target_argmax, sampled[num_bonus:]
    )
    return _assemble_selected_tokens(
        draft,
        accepted,
        recovered,
        num_draft_tokens,
        bonus_token_ids,
        segment_ids,
        group_indices,
        max_draft_tokens,
    )
