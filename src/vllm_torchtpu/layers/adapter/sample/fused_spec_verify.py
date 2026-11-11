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
"""Non-greedy spec-decode verify (bonus sampling + rejection) as one program.

The unfused verify runs, per step, `spec_gather_bonus_and_target_logits` (copies
the bonus rows and every draft row of the logits), the bonus sampler, three
eager `[segment_ids]` gathers of the sampling parameters and the rejection
sampler: five programs, plus a materialized `[num_tokens, vocab]` copy of the
logits that synthetic rejection reads only one row per request of.
`spec_verify_sample_fused` traces the same functions into one graph, so the
gathers fuse into their consumers. Same inputs, same math, same tokens.

Enabled by `--additional-config '{"spec_verify_fused_sampler": true}'`.
"""

from typing import Any

import torch

from vllm_torchtpu.layers.adapter.sample.rejection_sampler import (
    _random_rejection_sample_with_segment_impl,
)
from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p

FUSED_SAMPLER_KEY = "spec_verify_fused_sampler"

SAMPLING_EPS = 1e-5


def fused_sampler_enabled(additional_config: Any) -> bool:
    """`additional_config["spec_verify_fused_sampler"]`; off by default."""
    if not isinstance(additional_config, dict):
        return False
    enabled = additional_config.get(FUSED_SAMPLER_KEY, False)
    if not isinstance(enabled, bool):
        raise ValueError(f"additional_config.{FUSED_SAMPLER_KEY} must be a bool")
    return enabled


def sample_rows(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    u: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    topk_candidates: int = 0,
    apply_topk_topp: bool = True,
    apply_top_k: bool = True,
) -> torch.Tensor:
    """The non-greedy body of `TPUModelRunner.sample_from_logits`: Gumbel-max
    over the temperature-scaled, top-k/top-p masked logits; greedy rows
    (temperature <= SAMPLING_EPS) take the raw argmax. Returns [rows, 1]."""
    is_greedy = temperatures <= SAMPLING_EPS
    safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
    scaled_logits = logits / safe_temperatures
    if apply_topk_topp:
        masked_logits = apply_top_k_top_p(
            scaled_logits, top_k, top_p, topk_candidates, apply_top_k
        )
    else:
        masked_logits = scaled_logits
    u_clamped = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    gumbel_noise = -torch.log(-torch.log(u_clamped))
    noisy_logits = masked_logits + gumbel_noise
    final_logits = torch.where(is_greedy, logits, noisy_logits)
    return torch.argmax(final_logits, dim=-1, keepdim=True)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def spec_verify_sample_fused(
    logits: torch.Tensor,
    bonus_logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    segment_ids: torch.Tensor,
    group_indices: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    bonus_u: torch.Tensor,
    accept_u: torch.Tensor,
    recover_u: torch.Tensor,
    synthetic_conditional_rates: torch.Tensor | None,
    synthetic_mode: bool,
    max_draft_tokens: int,
    topk_candidates: int = 0,
    apply_topk_topp: bool = True,
    apply_top_k: bool = True,
) -> torch.Tensor:
    """Bonus sampling and non-greedy rejection of one verify chunk.

    `temperatures`, `top_k`, `top_p` are per request ([num_reqs, 1], as from
    `_build_padded_sampling_params`); `bonus_u` [num_reqs, vocab], `accept_u`
    [num_tokens] and `recover_u` [num_tokens, vocab] are the draws the unfused
    path takes, in its order. Returns the rejection sampler's
    [num_reqs, max_draft_tokens + 1] tokens.
    """
    bonus_logits = torch.index_select(logits, 0, bonus_logits_indices)
    target_logits = torch.index_select(logits, 0, target_logits_indices)
    bonus_token_ids = sample_rows(
        bonus_logits,
        temperatures,
        bonus_u,
        top_k,
        top_p,
        topk_candidates,
        apply_topk_topp,
        apply_top_k,
    ).view(-1)
    # Padding slots carry segment id num_reqs, which can equal the padded
    # request count; clamp as the TPU gather of the unfused path does. Their
    # rows are masked out of the result either way.
    param_rows = segment_ids.clamp(max=temperatures.shape[0] - 1)
    return _random_rejection_sample_with_segment_impl(
        draft_token_ids,
        target_logits,
        num_draft_tokens,
        bonus_token_ids,
        segment_ids,
        group_indices,
        max_draft_tokens,
        temperatures[param_rows],
        top_k[param_rows],
        top_p[param_rows],
        accept_u,
        recover_u,
        synthetic_conditional_rates,
        synthetic_mode,
        topk_candidates,
        apply_topk_topp,
        None,
        apply_top_k,
    )
