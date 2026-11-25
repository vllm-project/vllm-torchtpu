# SPDX-License-Identifier: Apache-2.0
"""Speculative verify on vocabulary-shard candidates (VLLM_TPU_VERIFY_SHARD_TOPK).

Each tensor-parallel rank takes the top k' logits of its slice of the LM head
and the ranks all-gather only those (value, id) pairs (the logits program); the
verify merges them into each row's k' largest candidates, sorted by value. A
step whose requests all sample greedily or with 1 <= top_k < k', ask for no
logprobs and carry no grammar needs nothing else: top-k keeps the top_k largest
logits, top-p acts on their softmax, and the bonus sample, the recovered sample
at a rejection and the accept test read only the kept tokens. Other steps
gather the full logits.

The candidates hold every logit at or above a row's top-k cutoff unless k' or
more of them tie with it. Such rows are flagged (their last candidate equals the
cutoff) and counted, not corrected. Uniforms are drawn per candidate instead of
per vocabulary id, so a seed gives the same distribution as the full-logits
path, not the same tokens.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import torch
from vllm.distributed import tensor_model_parallel_all_gather

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    TEMPERATURE_EPS,
    _random_rejection_sample_impl,
    assemble_rejection_output,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import (
    MASKED_LOGIT_VALUE,
    apply_top_k_top_p,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

# Greedy rows (temperature at or below this) take the argmax, as
# tpu_runner._sample_from_logits_impl decides.
SAMPLING_EPS = 1e-5


def shard_candidates(
    owner: torch.nn.Module, hidden_states: torch.Tensor, k: int
) -> torch.Tensor:
    """Each rank's k largest logits of its LM-head slice, all-gathered as f32
    ``[rows, ranks * 2k]``: per rank its k values, then their token ids.

    Values are the LM head's logits converted to f32 (exact from bf16); ids
    are exact in f32 below 2**24, as in LogitsProcessor.get_top_tokens."""
    processor = cast("LogitsProcessor", owner.logits_processor)
    head = cast("VocabParallelEmbedding", owner.lm_head)
    local = processor(head, hidden_states, skip_gather=True)
    num_pad = head.shard_indices.num_org_vocab_padding
    if num_pad > 0:
        pad = torch.zeros_like(local, dtype=torch.bool)
        pad[..., -num_pad:] = True
        local = torch.where(pad, torch.full_like(local, float("-inf")), local)
    # A direct top-k on the bf16 logits: the dot writes them rows-minor, so a
    # vocab-block reshape for a two-stage selection would relayout the shard.
    values, ids = torch.topk(local, k, dim=-1, sorted=False)
    ids = ids + head.shard_indices.org_vocab_start_index
    pairs = torch.cat([values.float(), ids.float()], dim=-1)
    if head.tp_size > 1:
        pairs = tensor_model_parallel_all_gather(pairs, dim=-1)
    return pairs


def merge(pairs: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's k largest candidates of a shard_candidates result: values
    f32 [rows, k] sorted descending and their ids int64 [rows, k]."""
    rows = pairs.shape[0]
    pairs = pairs.reshape(rows, -1, 2, k)
    all_values = pairs[:, :, 0, :].reshape(rows, -1)
    all_ids = pairs[:, :, 1, :].reshape(rows, -1)
    values, pos = torch.topk(all_values, k, dim=-1, sorted=True)
    return values, torch.gather(all_ids, 1, pos).to(torch.int64)


def candidate_argmax(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The id of each row's largest value, the lowest id on ties, as
    torch.argmax over the full row gives. int32 [rows]."""
    lowest = torch.where(
        values == values[:, :1], ids, torch.full_like(ids, torch.iinfo(torch.int64).max)
    )
    return lowest.amin(dim=-1).to(torch.int32)


def keep_mask(
    values: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
) -> torch.Tensor:
    """The candidates apply_top_k_top_p keeps, for values sorted descending.

    Top-k keeps every value at or above the top_k-th largest (topk_mask's
    cutoff, ties included); top-p keeps the probabilities at or above the
    largest threshold whose mass reaches top_p (topp_mask), over the softmax of
    the top-k survivors. Rows with top_k <= 0 or top_p >= 1 skip that mask.
    Computed in the dtype of `values`, as the full-row masks are.
    """
    width = values.shape[-1]
    cutoff = torch.gather(values, 1, (top_k.clamp(min=1, max=width) - 1).long())
    kept = torch.where(
        top_k > 0, values >= cutoff, torch.ones_like(values, dtype=torch.bool)
    )
    masked = torch.where(kept, values, torch.full_like(values, MASKED_LOGIT_VALUE))
    probs = torch.softmax(masked, dim=-1)
    # mass[i]: the probability mass at or above probs[i], summed as
    # topp_mask's predicate sums it.
    mass = torch.sum(
        torch.where(
            probs.unsqueeze(-2) >= probs.unsqueeze(-1),
            probs.unsqueeze(-2),
            torch.zeros_like(probs).unsqueeze(-2),
        ),
        dim=-1,
    )
    threshold = torch.where(
        mass >= top_p, probs, torch.full_like(probs, float("-inf"))
    ).amax(dim=-1, keepdim=True)
    return torch.where(top_p < 1.0, kept & (probs >= threshold), kept)


def tie_flags(
    values: torch.Tensor, top_k: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    """Rows whose last candidate equals their top-k cutoff: there may be
    further logits equal to it that the candidates do not hold."""
    width = values.shape[-1]
    cutoff = torch.gather(values, 1, (top_k.clamp(min=1, max=width) - 1).long())
    return active & (values[:, -1:] >= cutoff).squeeze(-1)


def _bonus_tokens(
    values: torch.Tensor,
    ids: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """tpu_runner._sample_from_logits_impl on candidates: temperature in the
    logits dtype, top-k/top-p, then the Gumbel-max draw. int32 [rows]."""
    logits = values.to(temperatures.dtype)
    is_greedy = (temperatures <= SAMPLING_EPS).squeeze(-1)
    safe = torch.where(temperatures == 0.0, torch.ones_like(temperatures), temperatures)
    scaled = logits / safe
    masked = torch.where(
        keep_mask(scaled, top_k, top_p),
        scaled,
        torch.full_like(scaled, MASKED_LOGIT_VALUE),
    )
    u = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    noisy = masked + (-torch.log(-torch.log(u)))
    sampled = torch.gather(ids, 1, noisy.argmax(dim=-1, keepdim=True)).squeeze(-1)
    return torch.where(is_greedy, candidate_argmax(values, ids).long(), sampled).to(
        torch.int32
    )


def verify_candidates_impl(
    pairs: torch.Tensor,
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
    tie_counts: torch.Tensor,
    max_draft_tokens: int,
    synthetic_mode: bool,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-greedy spec verify of one chunk on candidates: tpu_runner's
    _verify_fused_impl with each row's `width` candidates in place of its
    logits, merged here from the ranks' `pairs`.

    `temperatures` is in the logits dtype; `bonus_u` and `recover_u` hold one
    uniform per candidate. Returns the [num_reqs, K+1] selected tokens and
    `tie_counts` (int32 [2]: flagged rows, rows top-k/top-p applied to) plus
    this chunk's rows."""
    all_values, all_ids = merge(pairs, width)
    bonus_values = torch.index_select(all_values, 0, bonus_logits_indices)
    bonus_ids = torch.index_select(all_ids, 0, bonus_logits_indices)
    values = torch.index_select(all_values, 0, target_logits_indices)
    ids = torch.index_select(all_ids, 0, target_logits_indices)
    bonus_token_ids = _bonus_tokens(
        bonus_values, bonus_ids, temperatures, top_k, top_p, bonus_u
    )

    row_temps = temperatures[segment_ids]
    row_top_k = top_k[segment_ids]
    row_top_p = top_p[segment_ids]
    draft_token_ids = draft_token_ids.to(torch.int64)
    valid_draft = draft_token_ids >= 0
    is_greedy = row_temps.squeeze(-1) == 0
    target_argmax = candidate_argmax(values, ids)
    scaled = values / (row_temps.to(torch.float32) + TEMPERATURE_EPS)
    masked = torch.where(
        keep_mask(scaled, row_top_k, row_top_p),
        scaled,
        torch.full_like(scaled, MASKED_LOGIT_VALUE),
    )
    probs = torch.softmax(masked, dim=-1)
    draft_mask = (ids == draft_token_ids.unsqueeze(-1)) & valid_draft.unsqueeze(-1)
    if synthetic_mode:
        assert synthetic_conditional_rates is not None
        accepted = accept_u < synthetic_conditional_rates[group_indices.to(torch.int64)]
    else:
        draft_probs = torch.where(draft_mask, probs, torch.zeros_like(probs)).sum(-1)
        accepted = torch.where(
            is_greedy, draft_token_ids == target_argmax, accept_u < draft_probs
        )
    accepted = accepted & valid_draft

    recovered_dist = torch.where(draft_mask, torch.zeros_like(probs), probs)
    recover_u = torch.clamp(
        recover_u,
        min=torch.finfo(probs.dtype).tiny,
        max=1.0 - torch.finfo(probs.dtype).eps,
    )
    race = recovered_dist / (-torch.log(recover_u) + torch.finfo(probs.dtype).tiny)
    recovered = torch.gather(ids, 1, race.argmax(dim=-1, keepdim=True)).squeeze(-1)
    has_recovery_mass = torch.any(recovered_dist > 0, dim=-1)
    recovered_token_ids = torch.where(
        is_greedy | ~has_recovery_mass, target_argmax.long(), recovered
    ).to(torch.int32)

    selected, valid_slot_mask = assemble_rejection_output(
        draft_token_ids,
        accepted,
        recovered_token_ids,
        draft_lengths,
        bonus_token_ids,
        segment_ids,
        group_indices,
        max_draft_tokens,
    )
    bonus_active = (top_k > 0).squeeze(-1) & (temperatures > SAMPLING_EPS).squeeze(-1)
    target_active = valid_slot_mask & (row_top_k > 0).squeeze(-1) & ~is_greedy
    flagged = (
        tie_flags(bonus_values, top_k, bonus_active).sum()
        + tie_flags(values, row_top_k, target_active).sum()
    )
    active = bonus_active.sum() + target_active.sum()
    counts = tie_counts + torch.stack([flagged, active]).to(tie_counts.dtype)
    return selected, counts


# Columns of check_candidates_impl's per-row output.
CHECK_COLUMNS = (
    "valid",
    "kept_mismatch",
    "argmax_mismatch",
    "tie_flag",
    "kept_full",
    "kept_candidates",
    "cls",
    "band_err_ppm",
    "logit_mismatch",
    "logit_max_steps",
    "diff_token",
    "diff_p_full_ppm",
    "diff_p_cand_ppm",
    "diff_kept_full",
    "diff_kept_cand",
)
# cls values: no kept-set difference; (a) a target row's kept set differs;
# (c) a bonus row's differs only in tokens at the top-p boundary within one bf16
# step; (d) any other bonus-row difference; a row whose full logits are not
# finite (the synthetic warmup requests), left out of (a)-(d).
CLS_NONE, CLS_TARGET, CLS_BONUS_BOUNDARY, CLS_BONUS_OTHER, CLS_DEGENERATE = range(5)


def bf16_steps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in bf16 steps (ulps) at the larger magnitude of the two, for
    values exact in bf16. f32, same shape."""
    mag = torch.maximum(a.abs(), b.abs()).clamp(min=torch.finfo(torch.bfloat16).tiny)
    return (a - b).abs() / torch.exp2(torch.floor(torch.log2(mag)) - 7)


def _column(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """x[r, index[r, 0]] per row."""
    return torch.gather(x, 1, index).squeeze(-1)


def _top_k_probs(scaled: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    """f32 softmax over the values that survive top-k (ties kept), any order."""
    width = scaled.shape[-1]
    ordered = torch.sort(scaled, dim=-1, descending=True).values
    cutoff = torch.gather(ordered, 1, (top_k.clamp(min=1, max=width) - 1).long())
    kept = torch.where(
        top_k > 0, scaled >= cutoff, torch.ones_like(scaled, dtype=torch.bool)
    )
    return torch.softmax(
        torch.where(kept, scaled, torch.full_like(scaled, MASKED_LOGIT_VALUE)), dim=-1
    )


def _dense_bonus_tokens(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    u: torch.Tensor,
    topk_candidates: int,
) -> torch.Tensor:
    """tpu_runner._sample_from_logits_impl on the full logits. int32 [rows]."""
    is_greedy = temperatures <= SAMPLING_EPS
    safe = torch.where(temperatures == 0.0, torch.ones_like(temperatures), temperatures)
    scaled = logits / safe
    masked = apply_top_k_top_p(scaled, top_k, top_p, topk_candidates)
    u = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    noisy = masked + (-torch.log(-torch.log(u)))
    return torch.where(is_greedy, logits, noisy).argmax(dim=-1).to(torch.int32)


def boundary_band(
    values: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per candidate, the distance of its top-p decision mass from top_p and
    whether that is within one bf16 step of top_p.

    A token survives top-p iff the f32 probability mass of the tokens strictly
    more probable than it is below top_p (topp_mask's threshold rule), so a
    rounding flip of its decision needs that mass within rounding of top_p.
    Probabilities are the f32 softmax of the f32 logits that survive top-k.
    Returns (|mass - top_p| f32 [rows, k], within one bf16 step bool [rows, k])."""
    width = values.shape[-1]
    safe = torch.where(temperatures == 0.0, torch.ones_like(temperatures), temperatures)
    scaled = values / safe.to(torch.float32)
    cutoff = torch.gather(scaled, 1, (top_k.clamp(min=1, max=width) - 1).long())
    kept = torch.where(
        top_k > 0, scaled >= cutoff, torch.ones_like(scaled, dtype=torch.bool)
    )
    probs = torch.softmax(
        torch.where(kept, scaled, torch.full_like(scaled, MASKED_LOGIT_VALUE)), dim=-1
    )
    above = torch.sum(
        torch.where(
            probs.unsqueeze(-2) > probs.unsqueeze(-1),
            probs.unsqueeze(-2),
            torch.zeros_like(probs).unsqueeze(-2),
        ),
        dim=-1,
    )
    err = (above - top_p).abs()
    # One bf16 step at top_p: 2 ** (floor(log2(top_p)) - 7).
    step = torch.exp2(torch.floor(torch.log2(top_p)) - 7)
    return err, err <= step


def check_candidates_impl(
    logits: torch.Tensor,
    pairs: torch.Tensor,
    selected: torch.Tensor,
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
    num_reqs: torch.Tensor,
    max_draft_tokens: int,
    synthetic_mode: bool,
    topk_candidates: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VLLM_TPU_VERIFY_SHARD_CHECK: the fused verify on the full logits against
    the candidate verify's `selected` tokens and masks, for one chunk.

    The full-logits verify draws its uniforms from the candidate ones, each
    candidate's uniform at its id (0.5 elsewhere, where only a kept token
    outside the candidates could read it), so the two emit the same tokens
    unless their kept sets or rounding differ.

    Returns int32 [bonus rows + target rows, len(CHECK_COLUMNS)] per row
    (bonus rows first) and int32 [requests, 2]: emitted tokens differ, request
    has a non-finite row."""
    all_values, all_ids = merge(pairs, width)
    num_bonus = bonus_logits_indices.shape[0]
    vocab = logits.shape[-1]
    dense_bonus = torch.index_select(logits, 0, bonus_logits_indices)
    dense_target = torch.index_select(logits, 0, target_logits_indices)
    bonus_ids = torch.index_select(all_ids, 0, bonus_logits_indices)
    target_ids = torch.index_select(all_ids, 0, target_logits_indices)
    row_temps = temperatures[segment_ids]
    row_top_k = top_k[segment_ids]
    row_top_p = top_p[segment_ids]

    bonus_dense_tokens = _dense_bonus_tokens(
        dense_bonus,
        temperatures,
        top_k,
        top_p,
        torch.full(
            (num_bonus, vocab), 0.5, dtype=bonus_u.dtype, device=logits.device
        ).scatter(1, bonus_ids, bonus_u),
        topk_candidates,
    )
    selected_dense = _random_rejection_sample_impl(
        draft_token_ids,
        dense_target,
        draft_lengths,
        bonus_dense_tokens,
        segment_ids,
        group_indices,
        max_draft_tokens,
        row_temps,
        row_top_k,
        row_top_p,
        accept_u,
        torch.full(
            (target_ids.shape[0], vocab),
            0.5,
            dtype=recover_u.dtype,
            device=logits.device,
        ).scatter(1, target_ids, recover_u),
        synthetic_conditional_rates,
        synthetic_mode,
        topk_candidates,
    )

    bonus_valid = torch.arange(num_bonus, device=logits.device) < num_reqs
    target_valid = (segment_ids < num_reqs) & (
        group_indices < draft_lengths[segment_ids.clamp(max=draft_lengths.shape[0] - 1)]
    )
    finite_bonus = torch.isfinite(dense_bonus).all(dim=-1)
    finite_target = torch.isfinite(dense_target).all(dim=-1)
    bad_target = (~finite_target & target_valid).to(torch.int32)
    request_bad = (
        torch.zeros(num_bonus + 1, dtype=torch.int32, device=logits.device).index_add(
            0, segment_ids.clamp(max=num_bonus), bad_target
        )[:num_bonus]
        > 0
    ) | ~finite_bonus
    emitted = torch.stack(
        [
            (bonus_valid & ~request_bad & (selected_dense != selected).any(dim=-1)).to(
                torch.int32
            ),
            (bonus_valid & request_bad).to(torch.int32),
        ],
        dim=-1,
    )

    bonus_values = torch.index_select(all_values, 0, bonus_logits_indices)
    target_values = torch.index_select(all_values, 0, target_logits_indices)
    rows_out = []
    for dense, values, ids, temps, k_, p_, valid, finite, is_bonus in (
        (
            dense_bonus,
            bonus_values,
            bonus_ids,
            temperatures,
            top_k,
            top_p,
            bonus_valid,
            finite_bonus,
            True,
        ),
        (
            dense_target,
            target_values,
            target_ids,
            row_temps,
            row_top_k,
            row_top_p,
            target_valid,
            finite_target,
            False,
        ),
    ):
        if is_bonus:
            safe = torch.where(temps == 0.0, torch.ones_like(temps), temps)
            dense_scaled = dense / safe
            cand_scaled = values.to(temps.dtype) / safe
        else:
            dense_scaled = dense.to(torch.float32) / (
                temps.to(torch.float32) + TEMPERATURE_EPS
            )
            cand_scaled = values / (temps.to(torch.float32) + TEMPERATURE_EPS)
        dense_kept = (
            apply_top_k_top_p(dense_scaled, k_, p_, topk_candidates)
            > MASKED_LOGIT_VALUE
        )
        cand_kept = keep_mask(cand_scaled, k_, p_)
        dense_at_ids = torch.gather(dense_kept, 1, ids)
        outside = dense_kept.sum(-1) > dense_at_ids.sum(-1)
        differ = dense_at_ids != cand_kept
        greedy = (temps <= SAMPLING_EPS).squeeze(-1)
        masked_rows = valid & finite & (k_ > 0).squeeze(-1) & ~greedy
        kept_mismatch = masked_rows & (differ.any(dim=-1) | outside)
        argmax_mismatch = (
            valid
            & finite
            & (
                torch.argmax(dense, dim=-1).to(torch.int32)
                != candidate_argmax(values, ids)
            )
        )
        flags = tie_flags(values, k_, masked_rows)
        # The full logits at the candidates' ids, and the first differing token:
        # its probability after top-k on each path and whether each keeps it.
        dense_at = torch.gather(dense, 1, ids).to(torch.float32)
        logit_differs = (values != dense_at) & (valid & finite).unsqueeze(-1)
        logit_steps = torch.where(
            logit_differs, bf16_steps(values, dense_at), torch.zeros_like(values)
        ).amax(dim=-1)
        safe32 = torch.where(temps == 0.0, torch.ones_like(temps), temps).to(
            torch.float32
        )
        p_cand = _top_k_probs(values / safe32, k_)
        p_full = _top_k_probs(dense_at / safe32, k_)
        first = torch.argmax(differ.to(torch.int32), dim=-1, keepdim=True)
        has_diff = differ.any(dim=-1)

        err, in_band = boundary_band(values, temps, k_, p_)
        band_err = torch.where(differ, err, torch.zeros_like(err)).amax(dim=-1)
        if is_bonus:
            boundary = (~differ | in_band).all(dim=-1) & ~outside
            cls = torch.where(
                kept_mismatch,
                torch.where(boundary, CLS_BONUS_BOUNDARY, CLS_BONUS_OTHER),
                CLS_NONE,
            )
        else:
            cls = torch.where(kept_mismatch, CLS_TARGET, CLS_NONE)
        cls = torch.where(valid & ~finite, CLS_DEGENERATE, cls)
        rows_out.append(
            torch.stack(
                [
                    valid.to(torch.int32),
                    kept_mismatch.to(torch.int32),
                    argmax_mismatch.to(torch.int32),
                    flags.to(torch.int32),
                    dense_kept.sum(-1).to(torch.int32),
                    cand_kept.sum(-1).to(torch.int32),
                    cls.to(torch.int32),
                    torch.where(kept_mismatch, band_err * 1e6, -1.0).to(torch.int32),
                    # Candidates whose value differs from the full logits' value at
                    # the same id (the two programs each run the LM-head dot), and
                    # the largest difference in bf16 steps.
                    logit_differs.sum(-1).to(torch.int32),
                    torch.round(logit_steps).to(torch.int32),
                    torch.where(has_diff, _column(ids, first), -1).to(torch.int32),
                    torch.where(has_diff, _column(p_full, first) * 1e6, -1.0).to(
                        torch.int32
                    ),
                    torch.where(has_diff, _column(p_cand, first) * 1e6, -1.0).to(
                        torch.int32
                    ),
                    torch.where(
                        has_diff, _column(dense_at_ids.to(torch.int32), first), -1
                    ),
                    torch.where(
                        has_diff, _column(cand_kept.to(torch.int32), first), -1
                    ),
                ],
                dim=-1,
            )
        )
    return torch.cat(rows_out, dim=0), emitted


def step_eligible(
    input_batch: InputBatch,
    requests: Mapping[str, CachedRequestState],
    width: int,
) -> bool:
    """Whether every request of the step reads only the candidates: greedy or
    1 <= top_k < width, no logprobs, no grammar, and not all greedy (an
    all-greedy step takes the greedy verify)."""
    if input_batch.all_greedy or input_batch.max_num_logprobs is not None:
        return False
    num_reqs = input_batch.num_reqs
    temps = input_batch.temperature_cpu_tensor[:num_reqs]
    top_k = input_batch.top_k_cpu_tensor[:num_reqs]
    sampled = temps > SAMPLING_EPS
    if bool((sampled & ((top_k < 1) | (top_k >= width))).any()):
        return False
    for req_id in input_batch.req_ids[:num_reqs]:
        if req_id is None:
            continue
        params = requests[req_id].sampling_params
        if params is not None and params.structured_outputs is not None:
            return False
    return True
