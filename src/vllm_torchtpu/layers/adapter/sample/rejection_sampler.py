# Copyright 2025 Google LLC
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
"""Torch-based rejection sampler for speculative decoding on TPU."""

from typing import TYPE_CHECKING

import torch
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates

from vllm_torchtpu.layers.adapter.sample import top_p_sample
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

logger = init_logger(__name__)

# Placeholder token ID for rejected tokens
PLACEHOLDER_TOKEN_ID = -1
# Keep temperature scaling defined for greedy rows in mixed batches.
TEMPERATURE_EPS = 1e-9


class RejectionSampler:
    """Torch-based rejection sampler for speculative decoding on TPU."""

    def __init__(
        self,
        speculative_config: "SpeculativeConfig | None" = None,
        device: torch.device | None = None,
    ):
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if (
            speculative_config is not None
            and speculative_config.rejection_sample_method == "synthetic"
        ):
            rates = speculative_config.synthetic_acceptance_rates
            assert rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(rates),
                dtype=torch.float32,
                device=device,
            )
        self.synthetic_mode = self.synthetic_conditional_rates is not None

    def __call__(
        self,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: torch.Tensor,
        target_logits: torch.Tensor | None,
        bonus_token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        max_draft_tokens: int,
        temperatures: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
        accept_u: torch.Tensor | None = None,
        recover_u: torch.Tensor | None = None,
        do_sampling: bool = False,
        topk_candidates: int = 0,
        target_argmax: torch.Tensor | None = None,
        apply_topk_topp: bool = True,
        draft_probs: torch.Tensor | None = None,
        apply_top_k: bool = True,
    ) -> torch.Tensor:
        return self.forward(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            target_logits=target_logits,
            bonus_token_ids=bonus_token_ids,
            segment_ids=segment_ids,
            group_indices=group_indices,
            max_draft_tokens=max_draft_tokens,
            temperatures=temperatures,
            top_k=top_k,
            top_p=top_p,
            accept_u=accept_u,
            recover_u=recover_u,
            do_sampling=do_sampling,
            topk_candidates=topk_candidates,
            target_argmax=target_argmax,
            apply_topk_topp=apply_topk_topp,
            draft_probs=draft_probs,
            apply_top_k=apply_top_k,
        )

    def forward(
        self,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: torch.Tensor,
        target_logits: torch.Tensor | None,
        bonus_token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        group_indices: torch.Tensor,
        max_draft_tokens: int,
        temperatures: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
        accept_u: torch.Tensor | None = None,
        recover_u: torch.Tensor | None = None,
        do_sampling: bool = False,
        topk_candidates: int = 0,
        target_argmax: torch.Tensor | None = None,
        apply_topk_topp: bool = True,
        draft_probs: torch.Tensor | None = None,
        apply_top_k: bool = True,
    ) -> torch.Tensor:
        """Performs rejection sampling for speculative decoding.

        This function is intended to be torch.compiled for efficiency on TPU.

        Args:
          draft_token_ids: The token IDs from the draft model.
          num_draft_tokens: The number of draft tokens per sequence in the batch.
          target_logits: The logits from the target model.
          bonus_token_ids: The bonus token IDs to append if all draft tokens are
            accepted.
          segment_ids: Segment IDs to group tokens by batch item.
          group_indices: Indices within each segment.
          max_draft_tokens: The maximum number of draft tokens.
          temperatures: Per-draft-token temperatures, padded to target_logits.
          top_k: Per-draft-token top-k values, padded to target_logits.
          top_p: Per-draft-token top-p values, padded to target_logits.
          accept_u: Uniform random values for draft accept/reject decisions.
          recover_u: Uniform random values used to sample recovered tokens.
          do_sampling: Whether to use probabilistic rejection sampling. False
            keeps the existing greedy rejection path.
          topk_candidates: Candidate vocab blocks for small top-k; 0 searches
            the full vocab. See `apply_top_k_top_p_candidates`.
          target_argmax: The target's greedy token per row, standing in for
            target_logits on the greedy path (TPU_GREEDY_TOP_TOKENS).
          apply_topk_topp: False skips `apply_top_k_top_p` on the non-greedy
            path. The caller passes False only when no request sets top-k or
            top-p, where the two 31-step bisections over the vocabulary are
            the identity (7.5 ms per 512 rows on Flash). A separate graph.
          draft_probs: float32 distribution each draft token was sampled
            from (probabilistic drafting), shaped like target_logits. None
            treats the draft as one-hot (greedy drafting). Used by the
            non-greedy path; the greedy and synthetic decisions ignore it.
          apply_top_k: False drops the top-k bisection from the non-greedy
            path while keeping top-p. The caller passes False only when no
            row sets top-k (top_k <= 0 everywhere), where top-k is the
            identity. A separate graph.

        Returns:
          A tensor of shape [batch_size, max_draft_tokens + 1] containing the
          accepted tokens and bonus tokens, padded with PLACEHOLDER_TOKEN_ID.
        """
        if do_sampling:
            if target_argmax is not None or target_logits is None:
                raise ValueError(
                    "Non-greedy rejection sampling needs target_logits, not "
                    "target_argmax."
                )
            if (
                temperatures is None
                or top_k is None
                or top_p is None
                or accept_u is None
                or recover_u is None
            ):
                raise ValueError(
                    "Non-greedy rejection sampling requires temperatures, "
                    "top_k, top_p, accept_u, and recover_u."
                )
            # Checked eagerly: a mismatched q otherwise fails inside the
            # compiled graph with an opaque tracing error, or, e.g. [rows, 1],
            # broadcasts in p - q and returns wrong tokens.
            if draft_probs is not None and draft_probs.shape != target_logits.shape:
                raise ValueError(
                    f"draft_probs shape {tuple(draft_probs.shape)} does not "
                    f"match target_logits shape {tuple(target_logits.shape)}."
                )
            if draft_probs is not None and self.synthetic_mode:
                # As upstream, the synthetic schedule decides acceptance; q
                # only shapes the recovery distribution.
                logger.warning_once(
                    "Synthetic rejection sampling ignores draft_probs for the "
                    "accept decision."
                )
            return _random_rejection_sample_with_segment(
                draft_token_ids,
                target_logits,
                num_draft_tokens,
                bonus_token_ids,
                segment_ids,
                group_indices,
                max_draft_tokens,
                temperatures,
                top_k,
                top_p,
                accept_u,
                recover_u,
                self.synthetic_conditional_rates,
                self.synthetic_mode,
                topk_candidates,
                apply_topk_topp,
                draft_probs,
                apply_top_k,
            )
        if draft_probs is not None:
            # As upstream, greedy verification is an argmax match; q plays no
            # part, and computing it for an all-greedy batch is wasted work.
            logger.warning_once(
                "draft_probs passed to the greedy rejection path is ignored."
            )
        if self.synthetic_mode and accept_u is None:
            raise ValueError("Synthetic rejection sampling requires accept_u.")
        if target_argmax is not None:
            return _greedy_rejection_sample_from_argmax(
                draft_token_ids,
                target_argmax,
                num_draft_tokens,
                bonus_token_ids,
                segment_ids,
                group_indices,
                max_draft_tokens,
                accept_u,
                self.synthetic_conditional_rates,
                self.synthetic_mode,
            )
        if target_logits is None:
            raise ValueError(
                "Greedy rejection sampling needs target_logits or target_argmax."
            )
        return _greedy_rejection_sample_with_segment(
            draft_token_ids,
            target_logits,
            num_draft_tokens,
            bonus_token_ids,
            segment_ids,
            group_indices,
            max_draft_tokens,
            accept_u,
            self.synthetic_conditional_rates,
            self.synthetic_mode,
        )


def _get_segment_info(num_draft_tokens: torch.Tensor, total_tokens: int):
    """Helper to create segment IDs and per-segment indices."""
    batch_size = num_draft_tokens.shape[0]
    device = num_draft_tokens.device

    # segment_ids assigns a unique ID to each token.
    # In Torch, we can use repeat_interleave.
    segment_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device),
        num_draft_tokens,
    )

    # group_indices creates a within-segment index for each token.
    # E.g., [0, 1, 2, 0, 1, 0, 1, 2, 3] for sequences [3, 2, 4].
    # We can use cumsum and subtraction.
    cum_draft_tokens = torch.cumsum(num_draft_tokens, dim=0)
    segment_starts = torch.cat(
        [
            torch.zeros(1, dtype=num_draft_tokens.dtype, device=device),
            cum_draft_tokens[:-1],
        ]
    )
    broadcast_starts = torch.repeat_interleave(segment_starts, num_draft_tokens)
    group_indices = (torch.arange(total_tokens, device=device) - broadcast_starts).to(
        torch.int32
    )
    return segment_ids, group_indices


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _greedy_rejection_sample_with_segment(
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    accept_u: torch.Tensor | None,
    synthetic_conditional_rates: torch.Tensor | None,
    synthetic_mode: bool,
) -> torch.Tensor:
    """Vectorized greedy speculative decoding validation in Torch."""
    return _greedy_rejection_sample(
        draft_token_ids,
        torch.argmax(target_logits, dim=-1).to(torch.int32),
        num_draft_tokens,
        bonus_token_ids,
        segment_ids,
        group_indices,
        max_draft_tokens,
        accept_u,
        synthetic_conditional_rates,
        synthetic_mode,
    )


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _greedy_rejection_sample_from_argmax(
    draft_token_ids: torch.Tensor,
    target_argmax: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    accept_u: torch.Tensor | None,
    synthetic_conditional_rates: torch.Tensor | None,
    synthetic_mode: bool,
) -> torch.Tensor:
    """Greedy validation from the target's argmax tokens, without logits."""
    return _greedy_rejection_sample(
        draft_token_ids,
        target_argmax.to(torch.int32),
        num_draft_tokens,
        bonus_token_ids,
        segment_ids,
        group_indices,
        max_draft_tokens,
        accept_u,
        synthetic_conditional_rates,
        synthetic_mode,
    )


def _greedy_rejection_sample(
    draft_token_ids: torch.Tensor,
    target_logits_argmax: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    accept_u: torch.Tensor | None,
    synthetic_conditional_rates: torch.Tensor | None,
    synthetic_mode: bool,
) -> torch.Tensor:
    total_tokens = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]
    device = draft_token_ids.device

    draft_token_ids = draft_token_ids.to(torch.int32)

    # Find the first mismatch index per segment using matrix masking.
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    match_mask = segment_ids.unsqueeze(0) == batch_indices
    # Only declared draft slots may participate in rejection sampling.
    draft_limits = num_draft_tokens.to(group_indices.dtype).unsqueeze(1)
    valid_slot_mask = torch.any(
        match_mask & (group_indices.unsqueeze(0) < draft_limits), dim=0
    )

    # Large value for positions with no mismatches
    # Create an array where mismatched positions hold their `group_index`
    # and matched positions hold a large value.
    large_value = total_tokens

    if total_tokens > 0:
        if synthetic_mode:
            if accept_u is None or synthetic_conditional_rates is None:
                raise ValueError(
                    "synthetic rejection sampling requires accept_u and "
                    "synthetic_conditional_rates"
                )
            conditional_rates = synthetic_conditional_rates[
                group_indices.to(torch.int64)
            ]
            accepted = (accept_u < conditional_rates) & (draft_token_ids >= 0)
        else:
            accepted = draft_token_ids == target_logits_argmax
        mismatch_indices = torch.where(
            valid_slot_mask & ~accepted,
            group_indices,
            torch.tensor(large_value, dtype=torch.int32, device=device),
        )

        masked_mismatch_indices = torch.where(
            match_mask,
            mismatch_indices.unsqueeze(0),
            torch.tensor(large_value, dtype=torch.int32, device=device),
        )
        first_mismatch_idx_per_segment = torch.min(
            masked_mismatch_indices, dim=1
        ).values
    else:
        first_mismatch_idx_per_segment = torch.full(
            (batch_size,), large_value, dtype=torch.int32, device=device
        )

    # Step 3: Broadcast Mismatch Info and Generate Main Token Output
    if total_tokens > 0:
        # Clamp padding segment sentinels for a safe gather.
        safe_segment_ids = segment_ids.clamp(min=0, max=batch_size - 1)
        first_mismatch_idx_broadcast = first_mismatch_idx_per_segment[safe_segment_ids]

        before_mismatch = valid_slot_mask & (
            group_indices < first_mismatch_idx_broadcast
        )
        at_mismatch = valid_slot_mask & (group_indices == first_mismatch_idx_broadcast)
        main_tokens = torch.where(
            before_mismatch,
            draft_token_ids,
            torch.where(
                at_mismatch,
                target_logits_argmax,
                torch.tensor(PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device),
            ),
        )
    else:
        main_tokens = torch.tensor([], dtype=torch.int32, device=device)

    # Step 4: Handle Bonus Tokens
    all_accepted = first_mismatch_idx_per_segment == large_value
    no_draft_tokens = num_draft_tokens == 0
    should_get_bonus = all_accepted | no_draft_tokens

    # Step 5: Arrange into a [batch_size, max_draft_tokens + 1] matrix
    # Route padding positions to an extra sink row for a safe scatter.
    selected_tokens = torch.full(
        (batch_size + 1, max_draft_tokens + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=device,
    )

    # Ensure all indices are uniformly int64 to prevent stablehlo.concatenate
    # type errors
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

    # Place main tokens at their exact per-request positions
    selected_tokens[scatter_segment_ids, scatter_group_indices] = main_tokens

    # Place bonus tokens immediately after the accepted draft sequence
    # ONLY if all tokens were accepted or there were no draft tokens.
    # If a mismatch occurred at k, the request's selected_tokens[k] already
    # contains target_logits_argmax[k].
    accepted_count = torch.where(
        first_mismatch_idx_per_segment == large_value,
        num_draft_tokens,
        first_mismatch_idx_per_segment,
    ).to(torch.int64)

    batch_range = torch.arange(batch_size, dtype=torch.int64, device=device)
    bonus_token_at_accepted_count = torch.where(
        should_get_bonus,
        bonus_token_ids.to(torch.int32),
        selected_tokens[batch_range, accepted_count],
    )
    selected_tokens[batch_range, accepted_count] = bonus_token_at_accepted_count

    return selected_tokens[:batch_size]


def _target_probs(
    target_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    topk_candidates: int,
    apply_topk_topp: bool,
    apply_top_k: bool = True,
) -> torch.Tensor:
    """Per-row target distribution after temperature, top-k and top-p."""
    scaled_logits = target_logits.to(torch.float32) / (
        temperatures.to(torch.float32) + TEMPERATURE_EPS
    )
    if apply_topk_topp:
        masked_logits = apply_top_k_top_p(
            scaled_logits, top_k, top_p, topk_candidates, apply_top_k
        )
    else:
        masked_logits = scaled_logits
    return torch.softmax(masked_logits, dim=-1)


def _recover_tokens(
    target_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    recover_u: torch.Tensor,
    topk_candidates: int,
    apply_topk_topp: bool,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor | None = None,
    target_logits_argmax: torch.Tensor | None = None,
    apply_top_k: bool = True,
) -> torch.Tensor:
    """Samples, per row, the token that replaces a rejected draft token.

    Every output row depends only on the same row of the inputs, so callers
    may pass any subset of rows. `target_probs` and `target_logits_argmax`
    are recomputed from `target_logits` when not given.
    """
    if (
        draft_probs is None
        and target_probs is None
        and target_logits_argmax is None
        and topk_candidates == 0
        and not (apply_topk_topp and apply_top_k)
        and top_p_sample.applies(target_logits)
    ):
        return top_p_sample.recover(
            target_logits,
            draft_token_ids,
            temperatures,
            top_p if apply_topk_topp else None,
            recover_u,
        )
    if target_logits_argmax is None:
        target_logits_argmax = torch.argmax(target_logits, dim=-1).to(torch.int32)
    if target_probs is None:
        target_probs = _target_probs(
            target_logits,
            temperatures,
            top_k,
            top_p,
            topk_candidates,
            apply_topk_topp,
            apply_top_k,
        )
    is_greedy = temperatures.squeeze(-1) == 0
    valid_draft_token_ids = draft_token_ids >= 0
    safe_draft_token_ids = draft_token_ids.clamp(min=0).long()
    # On rejection, resample from the residual max(0, p - q). Placeholder
    # drafts have no q mass to subtract, so their recovery distribution
    # remains the full target distribution.
    if draft_probs is not None:
        recovered_dist = torch.where(
            valid_draft_token_ids.unsqueeze(-1),
            torch.clamp(target_probs - draft_probs.to(target_probs.dtype), min=0.0),
            target_probs,
        )
    else:
        # With q a delta at x, subtracting q only touches entry x, so the
        # residual is just p with the drafted token zeroed out. Do not let
        # the safe clamp above accidentally mask vocabulary token 0.
        draft_token_mask = torch.nn.functional.one_hot(
            safe_draft_token_ids, num_classes=target_logits.shape[-1]
        ).to(torch.bool)
        draft_token_mask = draft_token_mask & valid_draft_token_ids.unsqueeze(-1)
        recovered_dist = torch.where(
            draft_token_mask,
            torch.tensor(0.0, dtype=target_probs.dtype, device=target_probs.device),
            target_probs,
        )
    recover_u = torch.clamp(
        recover_u,
        min=torch.finfo(target_probs.dtype).tiny,
        max=1.0 - torch.finfo(target_probs.dtype).eps,
    )
    exp_noise = -torch.log(recover_u)
    recovered_token_ids = torch.argmax(
        recovered_dist / (exp_noise + torch.finfo(target_probs.dtype).tiny),
        dim=-1,
    ).to(torch.int32)
    # Greedy rows recover the target argmax. Synthetic rejection can also leave
    # a random row with no residual mass: its target is one-hot at the draft,
    # so the draft is that row's unique argmax, which it recovers as well.
    has_recovery_mass = torch.any(recovered_dist > 0, dim=-1)
    return torch.where(
        is_greedy | ~has_recovery_mass,
        target_logits_argmax,
        recovered_token_ids,
    )


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _random_rejection_sample_with_segment(
    draft_token_ids: torch.Tensor,
    target_logits: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    accept_u: torch.Tensor,
    recover_u: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor | None,
    synthetic_mode: bool,
    topk_candidates: int = 0,
    apply_topk_topp: bool = True,
    draft_probs: torch.Tensor | None = None,
    apply_top_k: bool = True,
) -> torch.Tensor:
    """Vectorized non-greedy speculative decoding validation in Torch.

    With `draft_probs` (the distribution q each draft token x was sampled
    from) this is standard speculative sampling: accept iff u < p(x) / q(x),
    recover from the residual max(0, p - q). Without it the draft is treated
    as one-hot at x: accept iff u < p(x), recover from p with x removed.
    Both are exact for the target distribution p.
    """
    total_tokens = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]
    device = draft_token_ids.device

    draft_token_ids = draft_token_ids.to(torch.int64)
    valid_draft_token_ids = draft_token_ids >= 0

    target_probs = None
    target_logits_argmax = None
    if synthetic_mode:
        # The schedule alone decides acceptance, so the target distribution
        # is needed only to recover the first rejected slot of each request.
        if synthetic_conditional_rates is None:
            raise ValueError(
                "synthetic rejection sampling requires synthetic_conditional_rates"
            )
        conditional_rates = synthetic_conditional_rates[group_indices.to(torch.int64)]
        accepted = accept_u < conditional_rates
    else:
        is_greedy = temperatures.squeeze(-1) == 0
        target_logits_argmax = torch.argmax(target_logits, dim=-1).to(torch.int32)
        target_probs = _target_probs(
            target_logits,
            temperatures,
            top_k,
            top_p,
            topk_candidates,
            apply_topk_topp,
            apply_top_k,
        )
        # Accept test is the Leviathan ratio min(1, p(x) / q(x)).
        # Greedy drafting (draft_probs None) has q one-hot at the drafted
        # token x, so the ratio collapses to p(x). Accept iff u < p(x):
        # torch.rand can return exactly 0.0, and u <= p(x) would then accept a
        # token with p(x) == 0, e.g. one top-k masked out. With draft_probs
        # the same test reads u * q(x) < p(x); q(x) == 0 (a draft q could not
        # have produced) rejects rather than divide by zero, as upstream does.
        safe_draft_token_ids = draft_token_ids.clamp(min=0)
        target_token_probs = target_probs.gather(
            -1, safe_draft_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        if draft_probs is not None:
            draft_token_probs = draft_probs.gather(
                -1, safe_draft_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            random_accept = (draft_token_probs > 0) & (
                accept_u * draft_token_probs < target_token_probs
            )
        else:
            random_accept = accept_u < target_token_probs
        accepted = torch.where(
            is_greedy,
            draft_token_ids == target_logits_argmax,
            random_accept,
        )
    accepted = accepted & valid_draft_token_ids

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
        if synthetic_mode:
            # Recover on one row per request instead of all draft rows: the
            # vocabulary-wide work (top-k/top-p bisections, softmax, noise)
            # drops by the draft length. Each row's result depends only on
            # that row, so the tokens are unchanged. Requests with no rejected
            # slot point at an arbitrary row whose result is never used.
            token_range = torch.arange(total_tokens, device=device)
            rejection_rows = torch.where(
                match_mask & at_rejection.unsqueeze(0),
                token_range.unsqueeze(0),
                torch.tensor(total_tokens - 1, dtype=torch.int64, device=device),
            )
            rejection_rows = torch.min(rejection_rows, dim=1).values
            rows_draft_probs = (
                None
                if draft_probs is None
                else draft_probs.index_select(0, rejection_rows)
            )
            recovered_per_request = _recover_tokens(
                target_logits.index_select(0, rejection_rows),
                draft_token_ids.index_select(0, rejection_rows),
                temperatures.index_select(0, rejection_rows),
                top_k.index_select(0, rejection_rows),
                top_p.index_select(0, rejection_rows),
                recover_u.index_select(0, rejection_rows),
                topk_candidates,
                apply_topk_topp,
                rows_draft_probs,
                apply_top_k=apply_top_k,
            )
            recovered_token_ids = recovered_per_request[safe_segment_ids]
        else:
            assert target_probs is not None and target_logits_argmax is not None
            recovered_token_ids = _recover_tokens(
                target_logits,
                draft_token_ids,
                temperatures,
                top_k,
                top_p,
                recover_u,
                topk_candidates,
                apply_topk_topp,
                draft_probs,
                target_probs=target_probs,
                target_logits_argmax=target_logits_argmax,
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


# ---------------------------------------------------------------------------
# Synthetic verification on one logits row per request
#
# Under `rejection_sample_method=synthetic` the accept decisions depend only
# on the schedule's uniforms, not on the target logits. Each request then
# needs the target distribution at exactly one position: its first rejected
# draft slot, or its bonus slot when every draft is accepted. Deciding first
# lets the runner compute the LM head, the logits all-gather and the sampler
# on padded_num_reqs rows instead of every verify row (1 + K per request).
# ---------------------------------------------------------------------------


# Lanes past the vocabulary in the synthetic verify's one uniform draw
# (``[padded_num_reqs, vocab + lanes]``): they hold the accept uniforms, so
# one RNG program serves both the accept decisions and the recovery sampler.
SYNTHETIC_ACCEPT_LANES = 128


def synthetic_uniforms_fit(num_tokens: int, num_reqs: int) -> bool:
    """Whether ``num_tokens`` accept uniforms fit the lanes of one draw over
    ``num_reqs`` rows (`synthetic_verify_rows_from_uniforms`)."""
    return num_tokens <= num_reqs * SYNTHETIC_ACCEPT_LANES


def synthetic_shard_width(shard_vocab: int) -> int:
    """Lanes per LM head shard in the row split's layout
    (`synthetic_sample_rows_and_assemble`): the shard padded to whole lane
    groups (the sampler masks the padding)."""
    return -(-shard_vocab // 128) * 128


def synthetic_uniforms_shape(
    num_reqs: int, vocab: int, shards: tuple[int, int] | None = None
) -> tuple[int, ...]:
    """The one draw's shape. With ``shards`` ``(tp, shard vocab)`` the row
    split's layout, ``[rows + 1, tp, synthetic_shard_width]``: each request
    a dense slab of whole tiles, the layout the top-p kernel reads, and the
    accepts in the extra leading row (a major-axis slice; uniforms taken
    across the middle axis made XLA relay out the whole draw, 34-48 us at
    c48/c64, twin j-cff16624, j-eee68f3c). Otherwise ``[rows, vocab +
    SYNTHETIC_ACCEPT_LANES]``, the accepts in the last lanes."""
    if shards is not None:
        tp, shard_vocab = shards
        return (num_reqs + 1, tp, synthetic_shard_width(shard_vocab))
    return (num_reqs, vocab + SYNTHETIC_ACCEPT_LANES)


# Keyed uniforms (`TPU_SPEC_SYNTHETIC_KEYED_UNIFORMS`): instead of one draw
# in HBM, a step's uniforms are lowbias32 hashes of (row stream key, index),
# made where they are used (the top-p kernel makes its rows' own,
# kernels/top_p_sample.py `uniform_from_key`). The keys come from a counter
# state the fused verify advances on device, the same on every TP rank.
_GOLDEN = -1640531527  # 0x9e3779b9 as int32
_ROW_MULT = 0x632BE5AB


def _hash32(x: torch.Tensor) -> torch.Tensor:
    """`top_p_sample.hash32` in torch int32 ops (wrapping multiplies; the
    masks make the arithmetic shifts logical)."""
    m0, m1 = top_p_sample.HASH_MULTS
    x = x ^ ((x >> 16) & 0xFFFF)
    x = x * m0
    x = x ^ ((x >> 15) & 0x1FFFF)
    x = x * m1
    return x ^ ((x >> 16) & 0xFFFF)


def stream_keys(state: torch.Tensor, num_rows: int, seed: int) -> torch.Tensor:
    """int32 ``[num_rows]`` stream keys of the step at counter ``state``
    (int32 ``[1]``): row ``r`` of the draw's layout (the requests' rows,
    then the accepts' row, `synthetic_uniforms_shape`)."""
    step = _hash32(state.to(torch.int32) * _GOLDEN + (seed & 0x7FFFFFFF))
    rows = torch.arange(1, num_rows + 1, dtype=torch.int32, device=state.device)
    return _hash32(step + rows * _ROW_MULT)


def keyed_uniforms(keys: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """`top_p_sample.uniform_from_key`: f32 uniforms of ids ``ids`` in the
    streams ``keys`` (broadcast)."""
    bits = _hash32(ids.to(torch.int32) * top_p_sample.ID_MULT + keys.to(torch.int32))
    return ((bits >> 8) & 0xFFFFFF).to(torch.float32) * (1.0 / (1 << 24))


def _accept_uniforms(uniforms: torch.Tensor, num_tokens: int) -> torch.Tensor:
    if uniforms.dim() == 1:  # stream keys: the accepts' row is the last
        ids = torch.arange(num_tokens, dtype=torch.int32, device=uniforms.device)
        return keyed_uniforms(uniforms[-1:], ids)
    if uniforms.dim() == 3:
        # Gather the accept row first: indexing the draw itself lowers to an
        # as_strided copy of all of it (31 us at c48 b37, twin j-3d92658b).
        last = torch.full((1,), uniforms.shape[0] - 1, device=uniforms.device)
        return uniforms.index_select(0, last)[0, 0, :num_tokens]
    return uniforms[:, -SYNTHETIC_ACCEPT_LANES:].reshape(-1)[:num_tokens]


def synthetic_verify_rows_from_uniforms_body(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    final_logits_indices: torch.Tensor,
    uniforms: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`synthetic_verify_rows` with its accept uniforms taken from the last
    `SYNTHETIC_ACCEPT_LANES` lanes of ``uniforms`` (row-major,
    `synthetic_uniforms_shape`)."""
    return _synthetic_verify_rows(
        draft_token_ids,
        num_draft_tokens,
        segment_ids,
        group_indices,
        target_logits_indices,
        bonus_logits_indices,
        final_logits_indices,
        _accept_uniforms(uniforms, draft_token_ids.shape[0]),
        synthetic_conditional_rates,
    )


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def synthetic_verify_rows(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    final_logits_indices: torch.Tensor,
    accept_u: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compiled `_synthetic_verify_rows`."""
    return _synthetic_verify_rows(
        draft_token_ids,
        num_draft_tokens,
        segment_ids,
        group_indices,
        target_logits_indices,
        bonus_logits_indices,
        final_logits_indices,
        accept_u,
        synthetic_conditional_rates,
    )


def _synthetic_verify_rows(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    final_logits_indices: torch.Tensor,
    accept_u: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per request: (accepted count, the hidden-state row to sample, the
    rejected draft token or -1 when every draft is accepted).

    Same decisions as the synthetic rejection samplers above: a slot is
    accepted iff its uniform is below the schedule's conditional rate and its
    draft is a real token; a request stops at its first rejected slot.
    """
    total_tokens = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]
    device = draft_token_ids.device
    # int32 throughout: the ids and positions fit, and 64-bit integers split
    # every gather and compare into two on TPU.
    i32 = torch.int32
    draft_token_ids = draft_token_ids.to(i32)
    group_indices = group_indices.to(i32)
    num_draft_tokens = num_draft_tokens.to(i32)

    accepted = (accept_u < synthetic_conditional_rates[group_indices]) & (
        draft_token_ids >= 0
    )
    match_mask = segment_ids.to(i32).unsqueeze(0) == torch.arange(
        batch_size, dtype=i32, device=device
    ).unsqueeze(1)
    valid_slot_mask = torch.any(
        match_mask & (group_indices.unsqueeze(0) < num_draft_tokens.unsqueeze(1)),
        dim=0,
    )
    rejected_slot = match_mask & (valid_slot_mask & ~accepted).unsqueeze(0)
    # Slots of a request are consecutive with increasing group index, so its
    # first rejected slot is the smallest position.
    positions = torch.arange(total_tokens, dtype=i32, device=device)
    first_rejected_pos = torch.min(
        torch.where(rejected_slot, positions.unsqueeze(0), total_tokens), dim=1
    ).values
    any_rejected = first_rejected_pos < total_tokens
    safe_pos = torch.clamp(first_rejected_pos, max=total_tokens - 1)
    accepted_count = torch.where(
        any_rejected, group_indices[safe_pos], num_draft_tokens
    )
    logits_row = torch.where(
        any_rejected,
        target_logits_indices.to(i32)[safe_pos],
        bonus_logits_indices.to(i32),
    )
    rows = final_logits_indices[logits_row].to(i32)
    recover_draft = torch.where(any_rejected, draft_token_ids[safe_pos], -1)
    return accepted_count, rows, recover_draft


def synthetic_sample_and_assemble_body(
    logits: torch.Tensor,
    recover_draft: torch.Tensor,
    accepted_count: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    recover_u: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    topk_candidates: int = 0,
    apply_topk_topp: bool = True,
    apply_top_k: bool = True,
) -> torch.Tensor:
    """Sample each request's token at its `synthetic_verify_rows` row and
    lay out ``[padded_num_reqs, 1 + K]`` like the rejection samplers: the
    accepted drafts, then the sampled token, then placeholders.

    A rejected row samples the target distribution with the rejected draft
    removed (`_recover_tokens`, draft one-hot); a bonus row (`recover_draft`
    -1) samples the full target distribution, as the bonus sampler does.
    ``recover_u`` may be the one draw of `synthetic_uniforms_shape`.
    """
    recover_u = recover_u[:, : logits.shape[-1]]
    sampled = _recover_tokens(
        logits,
        recover_draft,
        temperatures,
        top_k,
        top_p,
        recover_u,
        topk_candidates,
        apply_topk_topp,
        None,
        apply_top_k=apply_top_k,
    )
    return _assemble_synthetic(
        sampled,
        accepted_count,
        draft_token_ids,
        num_draft_tokens,
        segment_ids,
        group_indices,
        max_draft_tokens,
    )


def synthetic_sample_rows_and_assemble_body(
    logit_rows: torch.Tensor,
    rank: torch.Tensor,
    recover_draft: torch.Tensor,
    accepted_count: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    recover_u: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
    row_width: int,
    vocab: int,
    topk_candidates: int = 0,
    apply_topk_topp: bool = True,
    apply_top_k: bool = True,
) -> torch.Tensor:
    """`synthetic_sample_and_assemble` with the requests' rows split over the
    TP ranks: ``logit_rows [rows / tp, tp, synthetic_shard_width]`` holds
    this rank's block of them (block ``rank`` of ``tp``) as the TP ranks' LM
    head shards side by side (vocab id ``s * row_width + w`` at ``[s, w]``,
    ``vocab`` ids, lanes past ``row_width`` padding;
    `TPUModelRunner.compute_selected_logit_rows`), and ``recover_u`` every
    request's uniforms in the same layout (`synthetic_uniforms_shape`). Each
    rank samples its block, the ranks all-gather the sampled tokens and every
    rank lays out the whole batch. Same distribution as one rank sampling
    every row: sampling is row-wise and every row sees the same inputs."""
    per_rank, tp = logit_rows.shape[:2]
    block = rank.view(-1) * per_rank + torch.arange(
        per_rank, dtype=rank.dtype, device=rank.device
    )

    def rows(x: torch.Tensor) -> torch.Tensor:
        # A gather along the leading axis: whole rows, or whole slabs of the
        # draw. (Viewing x as [tp, rows / tp, ...] blocks first lowered to an
        # as_strided copy, ~6 us per tensor at c48 b37, twin j-3d92658b.)
        return x.index_select(0, block)

    u_rows = rows(recover_u)
    kernel = (
        topk_candidates == 0
        and not (apply_topk_topp and apply_top_k)
        and top_p_sample.applies_shards(logit_rows, row_width, vocab)
    )
    if recover_u.dim() == 1 and not kernel:
        # Stream keys (`stream_keys`) where the kernel cannot make the
        # uniforms: the rows' uniforms in the draw's layout.
        ids = torch.arange(tp, dtype=torch.int32, device=u_rows.device)[
            :, None
        ] * row_width + torch.arange(
            logit_rows.shape[-1], dtype=torch.int32, device=u_rows.device
        )
        u_rows = keyed_uniforms(u_rows[:, None, None], ids[None])
    if kernel:
        # The shards as they came: no relayout of the rows into one vocab.
        sampled_rows = top_p_sample.recover_shards(
            logit_rows,
            rows(recover_draft),
            rows(temperatures),
            rows(top_p) if apply_topk_topp else None,
            u_rows,
            row_width,
            vocab,
        )
    else:

        def flat(x: torch.Tensor) -> torch.Tensor:
            return x[:, :, :row_width].reshape(per_rank, tp * row_width)[:, :vocab]

        sampled_rows = _recover_tokens(
            flat(logit_rows),
            rows(recover_draft),
            rows(temperatures),
            rows(top_k),
            rows(top_p),
            flat(u_rows),
            topk_candidates,
            apply_topk_topp,
            None,
            apply_top_k=apply_top_k,
        )
    return _gather_and_assemble(
        sampled_rows,
        accepted_count,
        draft_token_ids,
        num_draft_tokens,
        segment_ids,
        group_indices,
        max_draft_tokens,
    )


# Compiled, each its own program; `TPUModelRunner.synthetic_verify_and_sample`
# traces the plain bodies into one with the plan and the LM head.
synthetic_verify_rows_from_uniforms = torch.compile(
    synthetic_verify_rows_from_uniforms_body,
    backend="tpu",
    fullgraph=True,
    dynamic=False,
)
synthetic_sample_and_assemble = torch.compile(
    synthetic_sample_and_assemble_body, backend="tpu", fullgraph=True, dynamic=False
)
synthetic_sample_rows_and_assemble = torch.compile(
    synthetic_sample_rows_and_assemble_body,
    backend="tpu",
    fullgraph=True,
    dynamic=False,
)


def _gather_and_assemble(
    sampled_rows: torch.Tensor,
    accepted_count: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
) -> torch.Tensor:
    """`_assemble_synthetic` of every rank's sampled rows."""
    from vllm.distributed import tensor_model_parallel_all_gather

    # Exchanged as float32, which holds every id below 2**24 exactly, as the
    # vocab-parallel argmax exchanges its ids.
    sampled = tensor_model_parallel_all_gather(sampled_rows.float(), dim=0)
    return _assemble_synthetic(
        sampled.to(torch.int32),
        accepted_count,
        draft_token_ids,
        num_draft_tokens,
        segment_ids,
        group_indices,
        max_draft_tokens,
    )


def _assemble_synthetic(
    sampled: torch.Tensor,
    accepted_count: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    max_draft_tokens: int,
) -> torch.Tensor:
    """``[padded_num_reqs, 1 + K]``: each request's accepted drafts, then its
    sampled token, then placeholders."""
    batch_size = num_draft_tokens.shape[0]
    device = draft_token_ids.device
    # Gathers from per-request tables and the scatter into the output lower to
    # select chains on TPU (~200 us at 64 requests); masked reductions over
    # [requests, slots] do the same work vectorized. The [R, 1 + K, T] mask is
    # R * (1 + K) * T elements: 64 x 8 x 512 = 256K at max_num_seqs 64.
    i32 = torch.int32
    match = segment_ids.to(i32).unsqueeze(0) == torch.arange(
        batch_size, dtype=i32, device=device
    ).unsqueeze(1)
    limit = torch.minimum(num_draft_tokens.to(i32), accepted_count.to(i32))
    group_indices = group_indices.to(i32)
    kept = match & (group_indices.unsqueeze(0) < limit.unsqueeze(1))  # [R, T]
    column = torch.arange(max_draft_tokens + 1, dtype=i32, device=device)
    at_column = kept.unsqueeze(1) & (
        group_indices.view(1, 1, -1) == column.view(1, -1, 1)
    )  # [R, 1 + K, T]
    drafts = torch.where(
        at_column,
        draft_token_ids.to(torch.int32).view(1, 1, -1),
        torch.tensor(PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=device),
    ).amax(dim=-1)
    selected_tokens = torch.where(
        column.unsqueeze(0) == accepted_count.to(i32).unsqueeze(1),
        sampled.to(torch.int32).unsqueeze(1),
        drafts,
    )
    return selected_tokens
