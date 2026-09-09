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

from vllm_torchtpu.layers.vllm.sample.top_k_top_p import apply_top_k_top_p

if TYPE_CHECKING:
    from vllm.config.speculative import SpeculativeConfig

# Placeholder token ID for rejected tokens
PLACEHOLDER_TOKEN_ID = -1
# Added to the per-token temperature before dividing the target logits. Greedy
# rows carry temperature == 0.0, so dividing by this tiny value sharpens their
# logits into a one-hot at argmax -> deterministic accept/recover for free
# (mirrors JAX tpu-inference's `logits /= (temperatures + 1e-9)`).
TEMPERATURE_EPS = 1e-9


class RejectionSampler:
    """Torch-based rejection sampler for speculative decoding on TPU."""

    def __init__(
        self,
        speculative_config: "SpeculativeConfig | None" = None,
        device: torch.device | None = None,
    ):
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if (speculative_config is not None
                and speculative_config.rejection_sample_method == "synthetic"):
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
        target_logits: torch.Tensor,
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
        )

    def forward(
        self,
        draft_token_ids: torch.Tensor,
        num_draft_tokens: torch.Tensor,
        target_logits: torch.Tensor,
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

    Returns:
      A tensor of shape [batch_size, max_draft_tokens + 1] containing the
      accepted tokens and bonus tokens, padded with PLACEHOLDER_TOKEN_ID.
    """
        if do_sampling:
            if (temperatures is None or top_k is None or top_p is None
                    or accept_u is None or recover_u is None):
                raise ValueError(
                    "Non-greedy rejection sampling requires temperatures, "
                    "top_k, top_p, accept_u, and recover_u.")
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
            )
        if self.synthetic_mode and accept_u is None:
            raise ValueError("Synthetic rejection sampling requires accept_u.")
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
    segment_starts = torch.cat([
        torch.zeros(1, dtype=num_draft_tokens.dtype, device=device),
        cum_draft_tokens[:-1],
    ])
    broadcast_starts = torch.repeat_interleave(segment_starts,
                                               num_draft_tokens)
    group_indices = (torch.arange(total_tokens, device=device) -
                     broadcast_starts).to(torch.int32)
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
    total_tokens = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]
    device = draft_token_ids.device

    # Get target argmax
    target_logits_argmax = torch.argmax(target_logits, dim=-1).to(torch.int32)
    draft_token_ids = draft_token_ids.to(torch.int32)

    # Find the first mismatch index per segment using matrix masking.
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    match_mask = segment_ids.unsqueeze(0) == batch_indices
    # Only declared draft slots may participate in rejection sampling.
    draft_limits = num_draft_tokens.to(group_indices.dtype).unsqueeze(1)
    valid_slot_mask = torch.any(match_mask &
                                (group_indices.unsqueeze(0) < draft_limits),
                                dim=0)

    # Large value for positions with no mismatches
    # Create an array where mismatched positions hold their `group_index`
    # and matched positions hold a large value.
    large_value = total_tokens

    if total_tokens > 0:
        if synthetic_mode:
            assert accept_u is not None
            assert synthetic_conditional_rates is not None
            conditional_rates = synthetic_conditional_rates[group_indices.to(
                torch.int64)]
            accepted = ((accept_u < conditional_rates)
                        & (draft_token_ids >= 0))
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
        first_mismatch_idx_per_segment = torch.min(masked_mismatch_indices,
                                                   dim=1).values
    else:
        first_mismatch_idx_per_segment = torch.full((batch_size, ),
                                                    large_value,
                                                    dtype=torch.int32,
                                                    device=device)

    # Step 3: Broadcast Mismatch Info and Generate Main Token Output
    if total_tokens > 0:
        # Clamp padding segment sentinels for a safe gather.
        safe_segment_ids = segment_ids.clamp(min=0, max=batch_size - 1)
        first_mismatch_idx_broadcast = first_mismatch_idx_per_segment[
            safe_segment_ids]

        before_mismatch = (valid_slot_mask
                           & (group_indices < first_mismatch_idx_broadcast))
        at_mismatch = (valid_slot_mask
                       & (group_indices == first_mismatch_idx_broadcast))
        main_tokens = torch.where(
            before_mismatch,
            draft_token_ids,
            torch.where(
                at_mismatch,
                target_logits_argmax,
                torch.tensor(PLACEHOLDER_TOKEN_ID,
                             dtype=torch.int32,
                             device=device),
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
    scatter_group_indices = torch.where(valid_slot_mask, group_indices,
                                        torch.zeros_like(group_indices))

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
    selected_tokens[batch_range,
                    accepted_count] = bonus_token_at_accepted_count

    return selected_tokens[:batch_size]


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
) -> torch.Tensor:
    """Vectorized non-greedy speculative decoding validation in Torch."""
    total_tokens = draft_token_ids.shape[0]
    batch_size = num_draft_tokens.shape[0]
    device = draft_token_ids.device
    vocab_size = target_logits.shape[-1]

    draft_token_ids = draft_token_ids.to(torch.int64)
    valid_draft_token_ids = draft_token_ids >= 0
    safe_draft_token_ids = draft_token_ids.clamp(min=0)

    # Shape the target distribution with temperature + top-k/top-p, then
    # softmax. Greedy rows carry temperature == 0.0, so dividing by
    # (temperatures + TEMPERATURE_EPS) collapses them to a one-hot at argmax.
    scaled_logits = target_logits.to(
        torch.float32) / (temperatures.to(torch.float32) + TEMPERATURE_EPS)
    masked_logits = apply_top_k_top_p(scaled_logits, top_k, top_p)
    target_probs = torch.softmax(masked_logits, dim=-1)

    # Accept test is the Leviathan ratio min(1, p(x) / q(x))
    # Our draft is greedy for now, so for q: drafted token
    # x (q(x) == 1, 0 elsewhere) and the ratio collapses to p(x).
    target_token_probs = target_probs.gather(
        -1, safe_draft_token_ids.unsqueeze(-1)).squeeze(-1)
    if synthetic_mode:
        assert synthetic_conditional_rates is not None
        conditional_rates = synthetic_conditional_rates[group_indices.to(
            torch.int64)]
        accepted = accept_u < conditional_rates
    else:
        accepted = target_token_probs >= accept_u
    accepted = accepted & valid_draft_token_ids

    # On rejection, resample from the residual max(0, p - q). With q a delta at
    # x, subtracting q only touches entry x, so the residual is just p with the
    # drafted token zeroed out.
    draft_token_mask = torch.nn.functional.one_hot(safe_draft_token_ids,
                                                   num_classes=vocab_size).to(
                                                       torch.bool)
    # Placeholder drafts have no q mass to subtract, so their recovery
    # distribution remains the full target distribution. In particular, do
    # not let the safe clamp above accidentally mask vocabulary token 0.
    draft_token_mask = draft_token_mask & valid_draft_token_ids.unsqueeze(-1)
    recovered_dist = torch.where(
        draft_token_mask,
        torch.tensor(0.0, dtype=target_probs.dtype, device=device),
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
    # A synthetic rejection can reject a greedy draft that matches the target
    # argmax. In that case p and q are the same one-hot distribution, leaving
    # no residual mass. Match upstream's greedy path by recovering the target
    # argmax instead of letting argmax(all-zero) fall through to token 0.
    has_recovery_mass = torch.any(recovered_dist > 0, dim=-1)
    recovered_token_ids = torch.where(
        has_recovery_mass,
        recovered_token_ids,
        torch.argmax(target_probs, dim=-1).to(torch.int32),
    )

    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    match_mask = segment_ids.unsqueeze(0) == batch_indices
    # Only declared draft slots may participate in rejection sampling.
    draft_limits = num_draft_tokens.to(group_indices.dtype).unsqueeze(1)
    valid_slot_mask = torch.any(match_mask &
                                (group_indices.unsqueeze(0) < draft_limits),
                                dim=0)
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
        first_rejection_idx_per_segment = torch.min(masked_rejection_indices,
                                                    dim=1).values
    else:
        first_rejection_idx_per_segment = torch.full(
            (batch_size, ),
            large_value,
            dtype=torch.int32,
            device=device,
        )

    if total_tokens > 0:
        # Clamp padding segment sentinels for a safe gather.
        safe_segment_ids = segment_ids.clamp(min=0, max=batch_size - 1)
        first_rejection_idx_broadcast = first_rejection_idx_per_segment[
            safe_segment_ids]
        before_rejection = (valid_slot_mask
                            & (group_indices < first_rejection_idx_broadcast))
        at_rejection = (valid_slot_mask
                        & (group_indices == first_rejection_idx_broadcast))
        main_tokens = torch.where(
            before_rejection,
            draft_token_ids.to(torch.int32),
            torch.where(
                at_rejection,
                recovered_token_ids,
                torch.tensor(PLACEHOLDER_TOKEN_ID,
                             dtype=torch.int32,
                             device=device),
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
    scatter_group_indices = torch.where(valid_slot_mask, group_indices,
                                        torch.zeros_like(group_indices))
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
    selected_tokens[batch_range,
                    accepted_count] = bonus_token_at_accepted_count

    return selected_tokens[:batch_size]
