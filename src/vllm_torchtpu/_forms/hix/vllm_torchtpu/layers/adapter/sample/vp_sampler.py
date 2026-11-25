# SPDX-License-Identifier: Apache-2.0
"""Vocab-parallel MTP verify sampler (q38mm DSTOCK, VPS; default off, not wired yet).

Today every TP rank all-gathers the target logits ([N, V/TP] -> [N, V]) and runs the fused verify
(`runner/tpu_runner.py` `_verify_fused_impl`: bonus sample + `_random_rejection_sample_impl`, with the top-k
candidate path `apply_top_k_top_p_candidates`) over the full vocabulary, replicated. At c48 load (128 rows) that is
2.22 ms per decode step (M, c48-mix j-ed3c3078; DSTOCK lane_dstock.md section 2.1).

Here each rank keeps its own vocab shard: `local_candidates` takes the shard's exact top-KC per row (value desc,
global id asc) through a block-max preselect; the caller all-gathers only (value, global id) candidates along the
last dim ([N, KC] -> [N, TP*KC]); `vp_verify` then runs the stock verify on the candidates, op for op, with the
uniforms gathered at the candidate ids (the eager full-vocab draws are unchanged, so the random stream is too).

Exactness against stock (the same caveats as the stock candidate path, top_k_top_p.apply_top_k_top_p_candidates):
- the kept set (values >= the top-k/top-p cutoff) lies in the union of per-shard top-KC whenever top_k <= KC, so
  the bonus Gumbel-max, the target argmax, the synthetic accept and the recovery see the same values;
- argmaxes use torch.argmax's first-occurrence rule in GLOBAL id order (`_argmax_min_id`);
- float32 / bf16 softmax sums run over fewer terms, so a top-p boundary or a recovery near-tie within rounding can
  flip, and more than KC exact ties at one shard's cutoff are not covered (never seen on real logits, PD DEC-FB).
Design and provenance: ~/repos/q38/mm/reports/dstock_vps_design_DRAFT.md. The algorithm is PD DEC-FB's VPS2
(tests/decfb/q38_vps_lib.py on q38-decfb-vps2 7c5dae965: `local_candidates_blockmax`, `vp_verify`), ported to
torch with the stock code's dtypes; the assembly block below is copied verbatim from rejection_sampler.py
(`_random_rejection_sample_impl`, lines 515-608 at 20ac2efc0; tests/q38mm_dstock checks it textually).
"""

from __future__ import annotations

from collections.abc import Callable

import os

import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    TEMPERATURE_EPS,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    MASKED_LOGIT_VALUE,
    apply_top_k_top_p,
    topp_mask,
)

SAMPLING_EPS = 1e-5        # runner/tpu_runner.py:659 (vllm's SAMPLING_EPS)
INVALID_TOKEN_ID = -1      # runner/tpu_runner_async_output.py:17
VP_KC = 32                 # candidates per rank and row; >= every top_k the candidate path accepts (<= 64 // 2)
VP_BLOCK = 160             # preselect block: 31,040 = 194 x 160 (the TP8 shard of the 248,320 vocab)
# q38 L4 S1 (VLLM_TPU_VPS_TOPK_KEYS, read once at import; registered and hashed in envs.py): the two "stable
# descending sort, keep the first kc" steps of `local_candidates` become an exact top-kc over tie-free int32 keys
# (bf16 logits only). Read from os.environ with envs.env_bool's true spellings, because the VPS test harness loads
# this module under stub packages without vllm_torchtpu.envs (tests/q38mm_dstock/test_vp_sampler.py).
_TOPK_KEYS = os.environ.get("VLLM_TPU_VPS_TOPK_KEYS", "").strip().lower() in ("true", "1", "yes", "on")
# q38 L4 S2 (VLLM_TPU_VPS_TOPK_CUTOFF, read once at import like S1; registered and hashed in envs.py): the candidate
# mask's top-k cutoff is the k-th value of one top-k over the candidate row instead of top_k_top_p.topk_mask's
# 32-step float32 bisection (T98f per-instance split, M: top_k_top_p.py:195 ~0.013 ms per step). Bitwise equal to
# topk_mask for every k (see `_topk_mask_cutoff`); the top-p bisection is unchanged.
_TOPK_CUTOFF = os.environ.get("VLLM_TPU_VPS_TOPK_CUTOFF", "").strip().lower() in ("true", "1", "yes", "on")
_F32_MAX = 3.4028234663852886e38    # torch.finfo(torch.float32).max


def _topk_mask_cutoff(logits: torch.Tensor, k: torch.Tensor, replace_val: float = MASKED_LOGIT_VALUE) -> torch.Tensor:
    """`top_k_top_p.topk_mask(logits, k, replace_val)`, bit for bit, with the cutoff read off one descending top-k.

    topk_mask's bisection returns, per row, cutoff = -(the largest FINITE float32 T with count(logits > -T) < k), i.e.
    the smallest finite float32 c with fewer than k values strictly above it. With v_k the k-th largest value (NaN
    never counts: `nan > c` is False, so NaN ranks as -inf here): c = v_k for finite v_k; c = -FLT_MAX when v_k is
    -inf or k > the row width (no finite c reaches k); c = +inf when v_k is +inf, or k <= 0 (every c reaches k). The
    mask is then the same `logits >= cutoff` select (signed zeros compare equal, so the zero sign of c is moot).
    k > W needs no branch: the index clamps to the row's smallest value, which is -inf (clamped to -FLT_MAX) or a
    value every finite entry reaches, so the select equals the -FLT_MAX cutoff's either way."""
    w = logits.shape[-1]
    kk = k.to(torch.int64)
    x = torch.where(torch.isnan(logits), torch.full_like(logits, float("-inf")), logits)
    vals = torch.topk(x, w, dim=-1, largest=True, sorted=True).values
    v = torch.gather(vals, -1, (kk - 1).clamp(min=0, max=w - 1)).to(torch.float32)
    v = torch.clamp(v, min=-_F32_MAX)
    cutoff = torch.where(kk <= 0, torch.full_like(v, float("inf")), v)
    replace_tensor = torch.full_like(logits, replace_val)
    return torch.where(logits >= cutoff, logits, replace_tensor)


def _apply_top_k_top_p_cutoff(logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """`top_k_top_p.apply_top_k_top_p(logits, top_k, top_p)` (topk_candidates=0) with `_topk_mask_cutoff` in place
    of topk_mask; every other op is the same, so the output is bitwise equal."""
    should_apply_topk = (top_k > 0).expand(-1, logits.shape[-1])
    topk_masked = _topk_mask_cutoff(logits, top_k, MASKED_LOGIT_VALUE)
    masked_logits = torch.where(should_apply_topk, topk_masked, logits)

    should_apply_topp = (top_p < 1.0).expand(-1, masked_logits.shape[-1])
    topp_masked = topp_mask(masked_logits, top_p, MASKED_LOGIT_VALUE)
    return torch.where(should_apply_topp, topp_masked, masked_logits)


def _desc_keys(x: torch.Tensor, index_bits: int) -> torch.Tensor:
    """int32 keys whose descending order is exactly a STABLE descending sort of bf16 `x` along the last dim
    (value desc, then position asc), with no ties: (order-preserving 16-bit value) << index_bits | (max - pos).

    The 16-bit value is bf16's bit pattern made monotone (non-negative: as is; negative: magnitude bits flipped)
    after +0.0, which maps -0.0 to +0.0 so the two zeros compare equal as in the sort. Needs 16 + index_bits <= 31
    and the last dim <= 2**index_bits."""
    n = x.shape[-1]
    assert x.dtype == torch.bfloat16 and n <= (1 << index_bits) and 16 + index_bits <= 31
    b = (x + 0.0).contiguous().view(torch.int16).to(torch.int32)
    v = torch.where(b < 0, b ^ 0x7FFF, b)
    pos = torch.arange(n, device=x.device, dtype=torch.int32)
    return v * (1 << index_bits) + ((1 << index_bits) - 1 - pos)


def _first_kc_desc(x: torch.Tensor, kc: int, index_bits: int) -> torch.Tensor:
    """Indices of `torch.sort(x, dim=-1, descending=True, stable=True).indices[:, :kc]`, exactly, as a top-kc."""
    return torch.topk(_desc_keys(x, index_bits), kc, dim=-1, largest=True, sorted=True).indices


def local_candidates(
    local_logits: torch.Tensor, shard_offset: int, kc: int = VP_KC, block: int = VP_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """The shard's exact top-kc per row, (values in the logits dtype, int32 GLOBAL ids), ranked value desc / id asc.

    Block-max preselect: the kc blocks with the largest maxima (ties -> lower block, a stable sort) are gathered in
    ascending block order and the exact top-kc is taken inside them (ties -> lower position = lower id). Exact,
    ties included: an element of the true top-kc outside the chosen blocks would sit below kc block maxima that
    each rank above it, i.e. below kc elements, a contradiction (PD DEC-FB's proof)."""
    rows, vs = local_logits.shape
    if vs % block or vs // block < kc:
        raise ValueError(f"vocab shard {vs} needs >= {kc} whole blocks of {block}")
    blocks = local_logits.reshape(rows, vs // block, block)
    keys = _TOPK_KEYS and local_logits.dtype == torch.bfloat16
    if keys:
        order = _first_kc_desc(blocks.amax(dim=-1), kc, 8)
    else:
        order = torch.sort(blocks.amax(dim=-1), dim=-1, descending=True, stable=True).indices[:, :kc]
    bid = torch.sort(order, dim=-1).values
    cand = torch.gather(blocks, 1, bid.unsqueeze(-1).expand(-1, -1, block)).reshape(rows, kc * block)
    if keys:
        pos = _first_kc_desc(cand, kc, 13)
    else:
        pos = torch.sort(cand, dim=-1, descending=True, stable=True).indices[:, :kc]
    vals = torch.gather(cand, 1, pos)
    ids = torch.gather(bid, 1, torch.div(pos, block, rounding_mode="floor")) * block + pos % block + shard_offset
    return vals, ids.to(torch.int32)


def merge_candidates(
    vals: torch.Tensor, ids: torch.Tensor, all_gather: Callable[[torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, KC] per rank -> [N, TP*KC] on every rank (rank-major, `tensor_model_parallel_all_gather(x, dim=-1)`)."""
    return all_gather(vals), all_gather(ids)


def _argmax_min_id(score: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """argmax over the candidates with torch.argmax's first-occurrence rule in global id order."""
    best = score.amax(dim=-1, keepdim=True)
    big = torch.full_like(ids, torch.iinfo(torch.int32).max)
    return torch.where(score == best, ids, big).amin(dim=-1).to(torch.int32)


def _candidate_mask(cand: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """`apply_top_k_top_p_candidates` after its block gather, on the VP candidates: the stock top-k/top-p on the
    candidate row, then everything below the smallest kept value is masked (the row is the candidates only)."""
    if _TOPK_CUTOFF:
        kept = _apply_top_k_top_p_cutoff(cand, top_k, top_p) > MASKED_LOGIT_VALUE
    else:
        kept = apply_top_k_top_p(cand, top_k, top_p) > MASKED_LOGIT_VALUE
    inf = torch.full_like(cand, float("inf"))
    cutoff = torch.where(kept, cand, inf).amin(dim=-1, keepdim=True)
    masked = torch.where(cand >= cutoff, cand, torch.full_like(cand, MASKED_LOGIT_VALUE))
    return torch.where(top_k > 0, masked, cand)


def _vp_bonus(cv: torch.Tensor, ci: torch.Tensor, temperatures: torch.Tensor, u_c: torch.Tensor,
              top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """`_sample_from_logits_impl` on the candidates (dtypes as stock: logits / temperatures in the logits dtype)."""
    is_greedy = temperatures <= SAMPLING_EPS
    safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
    masked_logits = _candidate_mask(cv / safe_temperatures, top_k, top_p)
    u_clamped = torch.clamp(u_c, min=torch.finfo(u_c.dtype).tiny, max=1.0 - torch.finfo(u_c.dtype).eps)
    gumbel_noise = -torch.log(-torch.log(u_clamped))
    noisy_logits = masked_logits + gumbel_noise
    final_logits = torch.where(is_greedy, cv, noisy_logits)
    return _argmax_min_id(final_logits, ci)


def _vp_reject(draft_token_ids, cv, ci, num_draft_tokens, bonus_token_ids, segment_ids, group_indices,
               max_draft_tokens, temperatures, top_k, top_p, accept_u, recover_u_c, synthetic_conditional_rates,
               synthetic_mode, row_override=None):
    """`_random_rejection_sample_impl` on the candidates; the assembly is the stock block, verbatim."""
    total_tokens = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]
    device = draft_token_ids.device

    draft_token_ids = draft_token_ids.to(torch.int64)
    valid_draft_token_ids = draft_token_ids >= 0
    safe_draft_token_ids = draft_token_ids.clamp(min=0)
    is_greedy = temperatures.squeeze(-1) == 0
    target_logits_argmax = _argmax_min_id(cv, ci)

    scaled_logits = cv.to(torch.float32) / (temperatures.to(torch.float32) + TEMPERATURE_EPS)
    masked_logits = _candidate_mask(scaled_logits, top_k, top_p)
    target_probs = torch.softmax(masked_logits, dim=-1)
    draft_hit = ci.to(torch.int64) == safe_draft_token_ids.unsqueeze(-1)
    target_token_probs = torch.where(draft_hit, target_probs, 0.0).sum(dim=-1)
    if synthetic_mode:
        assert synthetic_conditional_rates is not None
        conditional_rates = synthetic_conditional_rates[group_indices.to(torch.int64)]
        accepted = accept_u < conditional_rates
    else:
        accepted = torch.where(is_greedy, draft_token_ids == target_logits_argmax, accept_u < target_token_probs)
    accepted = accepted & valid_draft_token_ids

    draft_token_mask = draft_hit & valid_draft_token_ids.unsqueeze(-1)
    recovered_dist = torch.where(
        draft_token_mask, torch.tensor(0.0, dtype=target_probs.dtype, device=device), target_probs
    )
    recover_u = torch.clamp(
        recover_u_c, min=torch.finfo(target_probs.dtype).tiny, max=1.0 - torch.finfo(target_probs.dtype).eps
    )
    exp_noise = -torch.log(recover_u)
    recovered_token_ids = _argmax_min_id(recovered_dist / (exp_noise + torch.finfo(target_probs.dtype).tiny), ci)
    has_recovery_mass = torch.any(recovered_dist > 0, dim=-1)
    recovered_token_ids = torch.where(is_greedy | ~has_recovery_mass, target_logits_argmax, recovered_token_ids)
    if row_override is not None:
        # PD DEC-FB-2 per-row fallback: rows whose candidates may miss a kept token take their full-vocabulary
        # accept decision and recovered token (`_full_target`), computed outside; every other row is untouched.
        use_row, accepted_full, recovered_full = row_override
        accepted = torch.where(use_row, accepted_full, accepted)
        recovered_token_ids = torch.where(use_row, recovered_full, recovered_token_ids)

    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    match_mask = segment_ids.unsqueeze(0) == batch_indices
    # Only declared draft slots may participate in rejection sampling.
    draft_limits = num_draft_tokens.to(group_indices.dtype).unsqueeze(1)
    valid_slot_mask = torch.any(
        match_mask & (group_indices.unsqueeze(0) < draft_limits), dim=0
    )
    large_value = total_tokens

    if total_tokens > 0:
        rejection_indices = torch.where(
            valid_slot_mask & ~accepted,
            group_indices,
            torch.tensor(large_value, dtype=torch.int32, device=device),
        )
        masked_rejection_indices = torch.where(
            match_mask,
            rejection_indices.unsqueeze(0),
            torch.tensor(large_value, dtype=torch.int32, device=device),
        )
        first_rejection_idx_per_segment = torch.min(
            masked_rejection_indices, dim=1
        ).values
    else:
        first_rejection_idx_per_segment = torch.full(
            (batch_size,),
            large_value,
            dtype=torch.int32,
            device=device,
        )

    if total_tokens > 0:
        # Clamp padding segment sentinels for a safe gather.
        safe_segment_ids = segment_ids.clamp(min=0, max=batch_size - 1)
        first_rejection_idx_broadcast = first_rejection_idx_per_segment[
            safe_segment_ids
        ]
        before_rejection = valid_slot_mask & (
            group_indices < first_rejection_idx_broadcast
        )
        at_rejection = valid_slot_mask & (
            group_indices == first_rejection_idx_broadcast
        )
        main_tokens = torch.where(
            before_rejection,
            draft_token_ids.to(torch.int32),
            torch.where(
                at_rejection,
                recovered_token_ids,
                torch.tensor(PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device),
            ),
        )
    else:
        main_tokens = torch.tensor([], dtype=torch.int32, device=device)

    all_accepted = first_rejection_idx_per_segment == large_value
    no_draft_tokens = num_draft_tokens == 0
    should_get_bonus = all_accepted | no_draft_tokens

    # Route padding positions to an extra sink row for a safe scatter.
    selected_tokens = torch.full(
        (batch_size + 1, max_draft_tokens + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    segment_ids = segment_ids.to(torch.int64)
    group_indices = group_indices.to(torch.int64)
    scatter_segment_ids = torch.where(
        valid_slot_mask,
        segment_ids,
        torch.full_like(segment_ids, batch_size),
    )
    scatter_group_indices = torch.where(
        valid_slot_mask, group_indices, torch.zeros_like(group_indices)
    )
    selected_tokens[scatter_segment_ids, scatter_group_indices] = main_tokens

    accepted_count = torch.where(
        all_accepted,
        num_draft_tokens,
        first_rejection_idx_per_segment,
    ).to(torch.int64)

    batch_range = torch.arange(batch_size, dtype=torch.int64, device=device)
    bonus_token_at_accepted_count = torch.where(
        should_get_bonus,
        bonus_token_ids.to(torch.int32),
        selected_tokens[batch_range, accepted_count],
    )
    selected_tokens[batch_range, accepted_count] = bonus_token_at_accepted_count

    return selected_tokens[:batch_size]


def _counter_draws(draws, bonus_ci: torch.Tensor, target_ci: torch.Tensor, draft_token_ids: torch.Tensor):
    """q38 L4 VRNG-VPS: (bonus_u_c [B, W], accept_u, recover_u_c [T, W]) drawn in-graph at the candidate ids.

    ``draws = (counter, vocab, at)`` with ``at(counter, idx)`` = layers/adapter/sample/vps_rng.counter_uniform_at. The
    element indices are vps_rng's layout (bonus r*V + id, accept B*V + t, recover B*V + T_a + t*V + id), so these are
    exactly the gathers of the full-vocabulary draws at the candidate ids (VLLM_TPU_VPS_DRAWS=full semantics)."""
    counter, vocab, at = draws
    nb, nt, na = bonus_ci.shape[0], target_ci.shape[0], draft_token_ids.numel()
    if (nb + nt) * vocab + na >= (1 << 31):
        raise ValueError(f"VRNG-VPS: ({nb} + {nt}) x {vocab} + {na} draw elements, past int32")
    dev = bonus_ci.device
    rb = torch.arange(nb, dtype=torch.int32, device=dev).unsqueeze(-1) * vocab
    rt = torch.arange(nt, dtype=torch.int32, device=dev).unsqueeze(-1) * vocab
    bonus_u_c = at(counter, rb + bonus_ci.to(torch.int32))
    accept_u = at(counter, nb * vocab + torch.arange(na, dtype=torch.int32, device=dev)).view(draft_token_ids.shape)
    recover_u_c = at(counter, (nb * vocab + na) + rt + target_ci.to(torch.int32))
    return bonus_u_c, accept_u, recover_u_c


def _counter_rows(draws, base: int, rows: torch.Tensor) -> torch.Tensor:
    """q38 L4 VRNG-VPS: whole vocabulary rows ([F, V]) of the region at element `base`, for rows `rows` [F]."""
    counter, vocab, at = draws
    col = torch.arange(vocab, dtype=torch.int32, device=rows.device)
    return at(counter, base + rows.to(torch.int32).unsqueeze(-1) * vocab + col)


def vp_verify(
    cand_vals: torch.Tensor,
    cand_ids: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    bonus_u: torch.Tensor,
    accept_u: torch.Tensor,
    recover_u: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_lengths: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor | None,
    max_draft_tokens: int,
    synthetic_mode: bool,
    num_reqs: torch.Tensor | None = None,
    draws=None,
):
    """`_verify_fused_impl` on the merged candidates ([N, TP*KC] raw logits and global ids). The uniforms are the
    runner's eager full-vocab draws (bonus [B, V], recover [T, V]), gathered here at the candidate ids; with `draws`
    (q38 L4 VRNG-VPS) they are computed in-graph at those ids instead (bonus_u / accept_u / recover_u unused)."""
    bonus_c = torch.index_select(cand_vals, 0, bonus_logits_indices)
    bonus_ci = torch.index_select(cand_ids, 0, bonus_logits_indices)
    target_c = torch.index_select(cand_vals, 0, target_logits_indices)
    target_ci = torch.index_select(cand_ids, 0, target_logits_indices)
    if draws is None:
        bonus_u_c = torch.gather(bonus_u, 1, bonus_ci.to(torch.int64))
        recover_u_c = torch.gather(recover_u, 1, target_ci.to(torch.int64))
    else:
        bonus_u_c, accept_u, recover_u_c = _counter_draws(draws, bonus_ci, target_ci, draft_token_ids)
    bonus_token_ids = _vp_bonus(bonus_c, bonus_ci, temperatures, bonus_u_c, top_k, top_p)
    next_tokens = _vp_reject(
        draft_token_ids, target_c, target_ci, draft_lengths, bonus_token_ids, segment_ids, group_indices,
        max_draft_tokens, temperatures[segment_ids], top_k[segment_ids], top_p[segment_ids], accept_u,
        recover_u_c, synthetic_conditional_rates, synthetic_mode,
    )
    if num_reqs is None:
        return next_tokens
    num_valid = (next_tokens != INVALID_TOKEN_ID).sum(dim=1)
    valid = torch.arange(next_tokens.shape[0], device=next_tokens.device) < num_reqs
    row_offsets = torch.where(valid, (num_valid - 1).clamp(min=0), 0)
    return next_tokens, row_offsets


def vp_verify_from_local(local_logits: torch.Tensor, shard_offset: int,
                         all_gather: Callable[[torch.Tensor], torch.Tensor], *args, kc: int = VP_KC,
                         block: int = VP_BLOCK, **kwargs):
    """The rank's whole VPS step: local candidates, the candidate all-gather, `vp_verify` (args as vp_verify's
    after cand_vals / cand_ids)."""
    vals, ids = local_candidates(local_logits, shard_offset, kc, block)
    cv, ci = merge_candidates(vals, ids, all_gather)
    return vp_verify(cv, ci, *args, **kwargs)


# ---------------------------------------------------------------------------------------------------------------------
# PD DEC-FB-2 (PR 2821 adoption item 1): a per-row full-vocabulary fallback and the per-candidate-draw mode (VPB).
#
# Why. VPS's candidates (each shard's top-KC) hold every token the top-k cutoff keeps only when the row's top_k is in
# [1, KC] AND no shard has more than KC values at or above the cutoff (exact ties). The PR (049bb029b) checks this per
# row but then sends the WHOLE chunk to the full-vocabulary path. Here only the rows that need it do:
#   - "range" rows: non-greedy rows of active requests whose top_k is outside [1, KC] (unset top_k is 0 here, the
#     runner's padding of top_k >= vocab). The runner knows these on the host and sizes the slot count F from them.
#   - "tie" rows: non-greedy rows with top_k in [1, KC] that fail the exact completeness test below (never seen on
#     real logits; the stock candidate path has the same gap and does not detect it, top_k_top_p.py:276-277).
# The first F flagged rows (range rows first) gather their full vocab rows ([F, V/TP] -> [F, V], one small
# all-gather) and run the stock arithmetic on them (`_full_bonus` / `_full_target`, op for op the stock full path);
# their bonus token / accept decision / recovered token replace the candidate results before the (verbatim) assembly.
# Flagged rows beyond F are counted (stats[2], "uncovered"), never silently mis-sampled: the runner logs them. With
# F = 0 no fallback is built and only the counters run.
#
# Completeness test (exact, cheap). Each shard also sends its (KC+1)-th largest value s_r (the largest value that is
# NOT a candidate). A token x missing from shard r's candidates has x <= s_r, so after the row's (monotone) temperature
# scaling, scaled(x) <= m = max_r scaled(s_r). If at least k candidates are > m, the k-th largest candidate (<= the
# true k-th largest) is > m >= scaled(x), so no missing token reaches the cutoff and the candidates hold the whole kept
# set. Hence: complete <=> count(scaled candidates > m) >= top_k. It holds for any top_k <= TP*KC.
#
# VPB (per-candidate draws). VPBX (default) gathers the runner's full-vocabulary uniforms at the candidate ids, which
# keeps tokens identical to the stock verify. VPB instead takes one uniform per candidate column ([B, TP*KC] bonus,
# [T, TP*KC] recovery): the same distribution (tokens outside the kept set never win either path), different draws,
# and no full-vocabulary uniforms (in-engine 0.36 ms per step at O1 B64, M, lane_decfb.md:112). Fallback rows then
# draw their own full-vocabulary uniforms (fb_u, [F, V]).
# ---------------------------------------------------------------------------------------------------------------------


def local_candidates_ext(
    local_logits: torch.Tensor, shard_offset: int, kc: int = VP_KC, block: int = VP_BLOCK
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`local_candidates` plus the shard's (kc+1)-th largest value per row ([N, 1], the largest non-candidate).

    The block-max preselect over kc+1 blocks gives the exact top-(kc+1) (same proof); its first kc entries are
    exactly `local_candidates(kc)`'s (the order value desc / id asc is a total order)."""
    vals, ids = local_candidates(local_logits, shard_offset, kc + 1, block)
    return vals[:, :kc], ids[:, :kc], vals[:, kc:]


def _scaled_bonus(x: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """The bonus path's temperature scaling (`_apply_temperature_impl`, logits dtype)."""
    return x / torch.where(temperatures == 0.0, 1.0, temperatures)


def _scaled_target(x: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """The rejection path's temperature scaling (float32, + TEMPERATURE_EPS)."""
    return x.to(torch.float32) / (temperatures.to(torch.float32) + TEMPERATURE_EPS)


def _need_full(scaled_c: torch.Tensor, scaled_s: torch.Tensor, greedy: torch.Tensor, top_k: torch.Tensor,
               active: torch.Tensor, kc: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(range rows, tie rows) [R] bool. scaled_c [R, W] candidates, scaled_s [R, TP] sentinels, greedy / active [R],
    top_k [R, 1]."""
    k = top_k.squeeze(-1)
    m = scaled_s.amax(dim=-1, keepdim=True)
    complete = (scaled_c > m).sum(dim=-1) >= k
    in_range = (k >= 1) & (k <= kc)
    live = active & ~greedy
    return live & ~in_range, live & in_range & ~complete


def _full_bonus(rows: torch.Tensor, temperatures: torch.Tensor, u: torch.Tensor, top_k: torch.Tensor,
                top_p: torch.Tensor) -> torch.Tensor:
    """`_sample_from_logits_impl` with topk_candidates=0 on full vocabulary rows (op for op), int32 [F]."""
    is_greedy = temperatures <= SAMPLING_EPS
    masked_logits = apply_top_k_top_p(_scaled_bonus(rows, temperatures), top_k, top_p)
    u_clamped = torch.clamp(u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps)
    gumbel_noise = -torch.log(-torch.log(u_clamped))
    noisy_logits = masked_logits + gumbel_noise
    final_logits = torch.where(is_greedy, rows, noisy_logits)
    return torch.argmax(final_logits, dim=-1).to(torch.int32)


def _full_target(rows, draft_token_ids, temperatures, top_k, top_p, accept_u, recover_u, conditional_rates,
                 synthetic_mode):
    """The per-row decisions of `_random_rejection_sample_impl` with topk_candidates=0 on full vocabulary rows (op for
    op; local names differ from `_vp_reject` only so that source anchors stay unique): (accepted [F] bool, recovered
    token [F] int32)."""
    vocab_size = rows.shape[-1]
    drafts = draft_token_ids.to(torch.int64)
    valid = drafts >= 0
    safe = drafts.clamp(min=0)
    greedy = temperatures.squeeze(-1) == 0
    argmax_t = torch.argmax(rows, dim=-1).to(torch.int32)
    masked_logits = apply_top_k_top_p(_scaled_target(rows, temperatures), top_k, top_p)
    probs = torch.softmax(masked_logits, dim=-1)
    p_draft = probs.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    if synthetic_mode:
        acc = accept_u < conditional_rates
    else:
        acc = torch.where(greedy, drafts == argmax_t, accept_u < p_draft)
    acc = acc & valid
    hit = torch.nn.functional.one_hot(safe, num_classes=vocab_size).to(torch.bool) & valid.unsqueeze(-1)
    dist = torch.where(hit, torch.tensor(0.0, dtype=probs.dtype, device=rows.device), probs)
    ru = torch.clamp(recover_u, min=torch.finfo(probs.dtype).tiny, max=1.0 - torch.finfo(probs.dtype).eps)
    noise = -torch.log(ru)
    rec = torch.argmax(dist / (noise + torch.finfo(probs.dtype).tiny), dim=-1).to(torch.int32)
    mass = torch.any(dist > 0, dim=-1)
    return acc, torch.where(greedy | ~mass, argmax_t, rec)


def vp_verify_fb(
    local_logits: torch.Tensor,
    shard_offset: int,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
    bonus_logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    bonus_u: torch.Tensor,
    accept_u: torch.Tensor,
    recover_u: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_lengths: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor | None,
    max_draft_tokens: int,
    synthetic_mode: bool,
    num_active: torch.Tensor,
    fallback_rows: int = 0,
    cand_draws: bool = False,
    fb_u: torch.Tensor | None = None,
    kc: int = VP_KC,
    block: int = VP_BLOCK,
    draws=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The rank's whole VPS verify with the per-row fallback: (next tokens [B, K+1] int32, stats [3] int32 =
    [flagged rows, flagged rows covered by a fallback slot, flagged rows uncovered]).

    num_active: int32 scalar or [1], the chunk's real request count (padded requests are never flagged).
    fallback_rows: F, static. cand_draws: VPB (bonus_u [B, TP*KC], recover_u [T, TP*KC] are per-candidate draws, and
    fb_u [F, V] the fallback rows' draws); else VPBX (bonus_u [B, V], recover_u [T, V], as stock).
    draws: q38 L4 VRNG-VPS, (counter, full vocab, counter_uniform_at): every uniform is computed in-graph with the
    VPBX (full-vocabulary) semantics, whatever cand_draws says; bonus_u / accept_u / recover_u / fb_u are unused."""
    vals, ids, sent = local_candidates_ext(local_logits, shard_offset, kc, block)
    cv, ci = merge_candidates(vals, ids, all_gather)
    cs = all_gather(sent)                                                    # [N, TP]
    num_bonus = bonus_logits_indices.shape[0]
    num_target = target_logits_indices.shape[0]
    device = local_logits.device
    hxs = _HXS_ON  # q38 HXSMALL VERIFY_FUSE: packed gathers + int32 flag math (bitwise; see the helpers)

    if hxs:
        (bonus_c, bonus_ci, bonus_s), (target_c, target_ci, target_s) = _hxs_rows(
            (cv, ci, cs), bonus_logits_indices, target_logits_indices)
    else:
        bonus_c = torch.index_select(cv, 0, bonus_logits_indices)
        bonus_ci = torch.index_select(ci, 0, bonus_logits_indices)
        bonus_s = torch.index_select(cs, 0, bonus_logits_indices)
        target_c = torch.index_select(cv, 0, target_logits_indices)
        target_ci = torch.index_select(ci, 0, target_logits_indices)
        target_s = torch.index_select(cs, 0, target_logits_indices)
    if draws is not None:
        bonus_u_c, accept_u, recover_u_c = _counter_draws(draws, bonus_ci, target_ci, draft_token_ids)
    elif cand_draws:
        bonus_u_c, recover_u_c = bonus_u, recover_u
    else:
        bonus_u_c = torch.gather(bonus_u, 1, bonus_ci.long())
        recover_u_c = torch.gather(recover_u, 1, target_ci.long())

    # Flags. Bonus rows are requests; target rows are declared draft slots of real segments.
    if hxs:
        num_active, active_b, seg_c, active_t, temps_t, top_k_t, top_p_t = _hxs_flags(
            num_active, num_bonus, segment_ids, group_indices, draft_lengths, temperatures, top_k, top_p, device)
    else:
        num_active = num_active.reshape(()).to(torch.int64)
        active_b = torch.arange(num_bonus, device=device) < num_active
        seg64 = segment_ids.to(torch.int64)
        seg_ok = (seg64 >= 0) & (seg64 < num_bonus)
        seg_c = seg64.clamp(min=0, max=num_bonus - 1)
        active_t = seg_ok & (seg_c < num_active) & (
            group_indices.to(torch.int64) < draft_lengths.to(torch.int64)[seg_c]
        )
        temps_t, top_k_t, top_p_t = temperatures[seg_c], top_k[seg_c], top_p[seg_c]
    range_b, tie_b = _need_full(
        _scaled_bonus(bonus_c, temperatures), _scaled_bonus(bonus_s, temperatures),
        (temperatures <= SAMPLING_EPS).squeeze(-1), top_k, active_b, kc,
    )
    range_t, tie_t = _need_full(
        _scaled_target(target_c, temps_t), _scaled_target(target_s, temps_t),
        temps_t.squeeze(-1) == 0, top_k_t, active_t, kc,
    )
    flag = torch.cat([range_b | tie_b, range_t | tie_t])                   # [R], R = B + T
    first = torch.cat([range_b, range_t])
    num_flagged = flag.sum().to(torch.int32)

    bonus_token_ids = _vp_bonus(bonus_c, bonus_ci, temperatures, bonus_u_c, top_k, top_p)
    row_override = None
    if fallback_rows > 0:
        prio = torch.where(first, 0, torch.where(flag, 1, 2)).to(torch.int32)
        order = torch.sort(prio, dim=-1, stable=True).indices[:fallback_rows]   # [F] verify rows, range rows first
        used = torch.index_select(flag, 0, order)
        is_b = order < num_bonus
        b_idx = order.clamp(max=num_bonus - 1)
        t_idx = (order - num_bonus).clamp(min=0, max=max(num_target - 1, 0))
        lrow = torch.where(is_b, bonus_logits_indices[b_idx], target_logits_indices[t_idx])
        rows = all_gather(torch.index_select(local_logits, 0, lrow))       # [F, V]
        isb = is_b.unsqueeze(-1)
        t_f = torch.where(isb, temperatures[b_idx], temps_t[t_idx])
        k_f = torch.where(isb, top_k[b_idx], top_k_t[t_idx])
        p_f = torch.where(isb, top_p[b_idx], top_p_t[t_idx])
        if draws is not None:
            na = draft_token_ids.numel()
            u_b = _counter_rows(draws, 0, b_idx)
            u_t = _counter_rows(draws, num_bonus * draws[1] + na, t_idx)
        elif cand_draws:
            assert fb_u is not None and fb_u.shape[0] == fallback_rows
            u_b, u_t = fb_u, fb_u
        else:
            u_b, u_t = bonus_u[b_idx], recover_u[t_idx]
        full_b = _full_bonus(rows, t_f, u_b, k_f, p_f)
        rates_f = None
        if synthetic_mode:
            assert synthetic_conditional_rates is not None
            rates_f = synthetic_conditional_rates[group_indices.to(torch.int64)[t_idx]]
        acc_f, rec_f = _full_target(rows, draft_token_ids[t_idx], t_f, k_f, p_f, accept_u[t_idx], u_t, rates_f,
                                    synthetic_mode)
        # Scatter the used slots back (a sink row at the end takes the unused ones).
        use_b = used & is_b
        use_t = used & ~is_b
        sink_b = torch.where(use_b, b_idx, num_bonus)
        sink_t = torch.where(use_t, t_idx, num_target)
        bonus_ext = torch.cat([bonus_token_ids, bonus_token_ids[:1]])
        bonus_ext[sink_b] = torch.where(use_b, full_b, bonus_ext[sink_b])
        bonus_token_ids = bonus_ext[:num_bonus]
        flag_t = torch.zeros(num_target + 1, dtype=torch.bool, device=device)
        acc_t = torch.zeros(num_target + 1, dtype=torch.bool, device=device)
        rec_t = torch.zeros(num_target + 1, dtype=torch.int32, device=device)
        flag_t[sink_t] = use_t
        acc_t[sink_t] = acc_f
        rec_t[sink_t] = rec_f
        row_override = (flag_t[:num_target], acc_t[:num_target], rec_t[:num_target])
        num_covered = used.sum().to(torch.int32)
    else:
        num_covered = torch.zeros((), dtype=torch.int32, device=device)

    seg_params = (_hxs_take(segment_ids, temperatures, top_k, top_p) if hxs
                  else (temperatures[segment_ids], top_k[segment_ids], top_p[segment_ids]))
    next_tokens = _vp_reject(
        draft_token_ids, target_c, target_ci, draft_lengths, bonus_token_ids, segment_ids, group_indices,
        max_draft_tokens, *seg_params, accept_u,
        recover_u_c, synthetic_conditional_rates, synthetic_mode, row_override=row_override,
    )
    stats = torch.stack([num_flagged, num_covered, num_flagged - num_covered])
    return next_tokens, stats


# ---------------------------------------------------------------------------------------------------------------------
# q38 HXSMALL VERIFY_FUSE (VLLM_TPU_HXS_VERIFY_FUSE=1; registered, strict and hashed in envs.py). Read ONCE at import from
# os.environ with envs' strict rule (like S1 / S2 above: the VPS test harness loads this module under stub packages, and
# the verify program is torch.compile(fullgraph=True), which must not read the environment or log while tracing); the
# engagement line is logged at that import. The verify
# program's sampler glue with fewer gather / index ops, every value bit for bit:
#   - the six candidate-row index_selects (values, ids, sentinels at the bonus and target rows) become ONE index_select
#     of the three tensors packed as int32 columns at the concatenated row indices (a row gather copies bits; bf16 /
#     f32 values travel as their own bit patterns, int16-sign-extended or int32-viewed, and are restored exactly);
#   - the per-target-row parameter gathers (temperature, top_k, top_p, draft length at seg_c; temperature, top_k,
#     top_p at segment_ids) become one packed gather per index vector (the SAME index tensor, so clamping / wrapping
#     is per element what each separate gather did);
#   - the flag arithmetic (num_active, segment ids, group indices, draft lengths) runs in int32: the values are row
#     counts / slots / draft lengths (< 2**31), so every comparison and clamp is the int64 one.
# Nothing touches a float operation, a draw (bonus_ci / target_ci / draft_token_ids feed _counter_draws unchanged), the
# counter, or the top-p bisection.
# ---------------------------------------------------------------------------------------------------------------------
_HXS_KEY = "VLLM_TPU_HXS_VERIFY_FUSE"


def _hxs_verify_read() -> bool:
    """The strict key (envs.env_with_choices' rule: unset / "0" off, "1" on, anything else raises)."""
    raw = os.environ.get(_HXS_KEY)
    if raw is None or raw == "0":
        return False
    if raw == "1":
        return True
    raise ValueError(f"Invalid value '{raw}' for {_HXS_KEY}. Valid options: ['0', '1'].")


def _hxs_log_engaged() -> None:
    import logging

    logging.getLogger("vllm.vllm_torchtpu.hxsmall").info(
        "q38 HXSMALL %s engaged: VPS verify glue: 1 packed candidate-row gather (was 6), 2 packed per-row "
        "parameter gathers (were 7), int32 flag math and al sums", _HXS_KEY)


def _hxs_pack(t: torch.Tensor) -> tuple[torch.Tensor, tuple]:
    """(int32 [rows, cols] holding `t`'s bits exactly, how to restore it)."""
    shape = t.shape
    t2 = t.reshape(shape[0], -1)
    if t.dtype in (torch.int32, torch.float32):
        return t2.view(torch.int32), (t.dtype, shape, "view32")
    if t.dtype in (torch.bfloat16, torch.float16, torch.int16):
        return t2.view(torch.int16).to(torch.int32), (t.dtype, shape, "view16")
    if t.dtype in (torch.int64, torch.uint8, torch.int8, torch.bool):
        return t2.to(torch.int32), (t.dtype, shape, "cast")   # small integers (lengths, ks, flags): exact
    raise TypeError(f"HXSMALL VERIFY_FUSE: cannot pack {t.dtype}")


def _hxs_unpack(p: torch.Tensor, how: tuple, rows: int) -> torch.Tensor:
    dtype, shape, mode = how
    out_shape = (rows, *shape[1:])
    if mode == "view32":
        return p.contiguous().view(dtype).reshape(out_shape)
    if mode == "view16":
        return p.to(torch.int16).view(dtype).reshape(out_shape)
    return p.to(dtype).reshape(out_shape)


def _hxs_gather_packed(index: torch.Tensor, tensors, select: bool):
    """`[t[index] for t in tensors]` (select: `torch.index_select(t, 0, index)`) from ONE gather of the packed rows."""
    packs = [_hxs_pack(t) for t in tensors]
    widths = [p.shape[1] for p, _ in packs]
    packed = torch.cat([p for p, _ in packs], dim=1)
    g = torch.index_select(packed, 0, index) if select else packed[index]
    n = g.shape[0]
    out, lo = [], 0
    for (_, how), w in zip(packs, widths):
        out.append(_hxs_unpack(g[:, lo:lo + w], how, n))
        lo += w
    return out


def _hxs_rows(cands, bonus_idx: torch.Tensor, target_idx: torch.Tensor):
    """((bonus_c, bonus_ci, bonus_s), (target_c, target_ci, target_s)) from one index_select at [bonus; target]."""
    nb = bonus_idx.shape[0]
    idx = torch.cat([bonus_idx, target_idx.to(bonus_idx.dtype)])
    rows = _hxs_gather_packed(idx, cands, select=True)
    return tuple(r[:nb] for r in rows), tuple(r[nb:] for r in rows)


def _hxs_take(index: torch.Tensor, *tensors):
    """`tuple(t[index] for t in tensors)` from one packed gather (the same index tensor for every column)."""
    return tuple(_hxs_gather_packed(index, tensors, select=False))


def _hxs_flags(num_active, num_bonus, segment_ids, group_indices, draft_lengths, temperatures, top_k, top_p, device):
    """vp_verify_fb's flag block with int32 arithmetic and one packed per-row gather at seg_c."""
    na = num_active.reshape(()).to(torch.int32)
    active_b = torch.arange(num_bonus, device=device, dtype=torch.int32) < na
    seg = segment_ids.to(torch.int32)
    seg_ok = (seg >= 0) & (seg < num_bonus)
    seg_c = seg.clamp(min=0, max=num_bonus - 1)
    temps_t, top_k_t, top_p_t, dl_t = _hxs_take(seg_c, temperatures, top_k, top_p, draft_lengths.to(torch.int32))
    active_t = seg_ok & (seg_c < na) & (group_indices.to(torch.int32) < dl_t)
    return na.to(torch.int64), active_b, seg_c, active_t, temps_t, top_k_t, top_p_t


_HXS_ON = _hxs_verify_read()
if _HXS_ON:
    _hxs_log_engaged()
