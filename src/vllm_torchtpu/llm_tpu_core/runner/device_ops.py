# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Google LLC
"""Compiled TPU device operations for model execution, sampling, and logprobs."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

SAMPLING_EPS = 1e-5


def _tpu_compile_op(fn: Callable) -> Callable:
    """Compiles function for TPU execution when running on TPU hardware, or

    returns eager callable on CPU test environments.
    """
    try:
        acc = (
            torch.accelerator.current_accelerator()
            if hasattr(torch, "accelerator")
            else None
        )
        if acc is not None and getattr(acc, "type", None) == "tpu":
            return torch.compile(fn, backend="tpu", fullgraph=True, dynamic=False)
    except Exception:
        pass
    return fn


@dataclass
class LogprobsResult:
    """Tensors representing logprobs and ranks gathered on device."""

    logprob_token_ids: torch.Tensor
    logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor


def reset_read_offsets_impl(
    read_offsets: torch.Tensor,
    keep_dev: torch.Tensor,
    stacked_groups: torch.Tensor,
    null: torch.Tensor,
) -> None:
    """Resets offsets for newly admitted requests."""
    orig = read_offsets[stacked_groups]
    reset = torch.where(keep_dev, orig, null)
    read_offsets.index_put_((stacked_groups,), reset)


def rollback_offsets_seed_impl(read_offsets: torch.Tensor, dst_t: torch.Tensor) -> None:
    """Zero-fills target slots for seed rollback."""
    read_offsets.index_put_(
        (dst_t,),
        torch.zeros(
            dst_t.shape[0], dtype=read_offsets.dtype, device=read_offsets.device
        ),
    )


def rollback_offsets_migrate_impl(
    read_offsets: torch.Tensor, src_t: torch.Tensor, dst_t: torch.Tensor
) -> None:
    """Migrates read offsets across slots."""
    read_offsets.index_put_((dst_t,), read_offsets[src_t])


def _substitute_placeholder_token_impl(
    input_ids: torch.Tensor,
    token_in_tpu_cur_input_indices: torch.Tensor,
    token_in_tpu_pre_next_tokens_indices: torch.Tensor,
    next_tokens: torch.Tensor,
) -> torch.Tensor:
    """Substitutes placeholder tokens on TPU for async scheduler."""
    mask = token_in_tpu_pre_next_tokens_indices > -1
    safe_idx = torch.clamp_min(token_in_tpu_pre_next_tokens_indices, 0)
    new_token_values = next_tokens[safe_idx].to(input_ids.dtype)
    original_values = input_ids[token_in_tpu_cur_input_indices]
    update_values = torch.where(mask, new_token_values, original_values)
    input_ids.scatter_(0, token_in_tpu_cur_input_indices, update_values)
    return input_ids


substitute_placeholder_token = _tpu_compile_op(_substitute_placeholder_token_impl)


def _compute_selected_logits_impl(
    hidden_states: torch.Tensor,
    indices_do_sample: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    org_vocab_size: int | None = None,
) -> torch.Tensor:
    """Gathers sample positions and projects logits."""
    selected = torch.index_select(hidden_states, 0, indices_do_sample)
    logits = torch.nn.functional.linear(selected, weight, bias)
    if org_vocab_size is not None:
        logits = logits[..., :org_vocab_size]
    return logits


compute_selected_logits = _tpu_compile_op(_compute_selected_logits_impl)


def _compute_logits_from_hidden_states_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    org_vocab_size: int | None = None,
) -> torch.Tensor:
    """Full-tensor linear projection for logits."""
    logits = torch.nn.functional.linear(hidden_states, weight, bias)
    if org_vocab_size is not None:
        logits = logits[..., :org_vocab_size]
    return logits


compute_logits_from_hidden_states = _tpu_compile_op(
    _compute_logits_from_hidden_states_impl
)


def _sample_from_logits_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    u: torch.Tensor,
    all_greedy: bool = False,
) -> torch.Tensor:
    """Device-compiled greedy and Gumbel-noise sampling."""
    if all_greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)
    is_greedy = temperatures <= SAMPLING_EPS
    inv_t = torch.where(is_greedy, 1.0, 1.0 / temperatures)
    scaled_logits = logits * inv_t
    u_clamped = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    gumbel_noise = -torch.log(-torch.log(u_clamped))
    noisy_logits = scaled_logits + gumbel_noise
    final_logits = torch.where(is_greedy, logits, noisy_logits)
    return torch.argmax(final_logits, dim=-1, keepdim=True)


sample_from_logits = _tpu_compile_op(_sample_from_logits_impl)


def _gather_logprobs_impl(
    logits: torch.Tensor,
    sampled_tokens: torch.Tensor,
    max_logprobs: int = 0,
) -> LogprobsResult:
    """Gathers top_logprobs and ranks directly on TPU."""
    token_ids = sampled_tokens.to(torch.int64)
    token_logits = logits.gather(-1, token_ids)
    token_ranks = (logits >= token_logits).sum(dim=-1, dtype=torch.int32)
    log_normalizers = torch.logsumexp(logits, dim=-1, keepdim=True)
    token_logprobs = (token_logits - log_normalizers).to(torch.float32)

    if max_logprobs > 0:
        topk_logits, topk_indices = torch.topk(logits, max_logprobs, dim=-1)
        topk_logprobs = (topk_logits - log_normalizers).to(torch.float32)
        logprob_token_ids = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
    else:
        logprob_token_ids = token_ids
        logprobs = token_logprobs

    return LogprobsResult(
        logprob_token_ids=logprob_token_ids.to(torch.int32),
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
    )


gather_logprobs = _tpu_compile_op(_gather_logprobs_impl)
