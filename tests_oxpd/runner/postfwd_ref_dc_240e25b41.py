# SPDX-License-Identifier: Apache-2.0
"""Verbatim copies from q38-decfb3-postfwd-pb2 240e25b41 (generated with `git show 240e25b41:...`, bodies unchanged),
for the flag-off identity checks of the draft-carries port (dongruizi's 81af97b0d / a29f195d3 / 9efe59db6 / a678097bd /
7086efb53): eagle3's `_async_seed_inputs_impl` (renamed old_async_seed_inputs_impl; it calls the unchanged
`_async_draft_seed_impl` and `_maybe_pad_dim0`) and, as text only, tpu_runner's `_verify_fused_impl`."""

import torch  # noqa: F401

from vllm_torchtpu.spec_decode.eagle3 import _async_draft_seed_impl, _maybe_pad_dim0  # noqa: F401

OLD_VERIFY_FUSED_IMPL_SRC = 'def _verify_fused_impl(\n    logits: torch.Tensor,\n    bonus_logits_indices: torch.Tensor,\n    target_logits_indices: torch.Tensor,\n    temperatures: torch.Tensor,\n    top_k: torch.Tensor,\n    top_p: torch.Tensor,\n    bonus_u: torch.Tensor,\n    accept_u: torch.Tensor,\n    recover_u: torch.Tensor,\n    draft_token_ids: torch.Tensor,\n    draft_lengths: torch.Tensor,\n    segment_ids: torch.Tensor,\n    group_indices: torch.Tensor,\n    synthetic_conditional_rates: torch.Tensor | None,\n    max_draft_tokens: int,\n    synthetic_mode: bool,\n    topk_candidates: int,\n) -> torch.Tensor:\n    """Non-greedy spec verify of one chunk as one program\n    (VLLM_TPU_VERIFY_FUSED): the bonus/target logit gathers, the bonus-token\n    sample, the per-position sampling-parameter gathers and the rejection\n    sampler, op for op the same as `_sample_spec_verify_chunk`\'s separate\n    programs. The uniforms come from the caller\'s eager draws, in the same\n    order and shapes, so the random stream is unchanged."""\n    bonus_logits = torch.index_select(logits, 0, bonus_logits_indices)\n    target_logits = torch.index_select(logits, 0, target_logits_indices)\n    bonus_token_ids = _sample_from_logits_impl(\n        bonus_logits, temperatures, bonus_u, top_k, top_p, topk_candidates\n    ).view(-1)\n    return _random_rejection_sample_impl(\n        draft_token_ids,\n        target_logits,\n        draft_lengths,\n        bonus_token_ids,\n        segment_ids,\n        group_indices,\n        max_draft_tokens,\n        temperatures[segment_ids],\n        top_k[segment_ids],\n        top_p[segment_ids],\n        accept_u,\n        recover_u,\n        synthetic_conditional_rates,\n        synthetic_mode,\n        topk_candidates,\n    )'


def old_async_seed_inputs_impl(
    next_tokens: torch.Tensor,
    num_draft: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_reqs: torch.Tensor,
    input_ids: torch.Tensor,
    p: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`_async_draft_seed_impl`, the eagle3 input shift + seed scatter
    (`Eagle3Proposer._draft_seed_input_ids`) and the loop-bucket pad of the
    gather index as one program (VLLM_TPU_DRAFT_SEED_FUSED). Returns
    `(draft_input_ids, gather_indices [p], num_rejected)`."""
    last_token_indices, next_token_ids, num_rejected = _async_draft_seed_impl(
        next_tokens, num_draft, query_start_loc, num_reqs
    )
    shifted = input_ids
    if input_ids.shape[0] > 1:
        shifted = torch.cat((input_ids[1:], input_ids[-1:]), dim=0)
    draft_input_ids = shifted.index_put(
        (last_token_indices,), next_token_ids.to(input_ids.dtype)
    )
    gather_indices = _maybe_pad_dim0(last_token_indices[:p], p)
    return draft_input_ids, gather_indices, num_rejected
