# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""The TPU runner's sampler and logprobs math as plain functions.

`TPUModelRunner._apply_temperature`, `sample_from_logits` and
`gather_logprobs` delegate here, so another compiled program can trace the
same math without a runner.
"""

import torch
from vllm.v1.outputs import LogprobsTensors

from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p

SAMPLING_EPS = 1e-5


def apply_temperature(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
    return logits / safe_temperatures


def sample_from_logits_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    u: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    *,
    all_greedy: bool,
    topk_candidates: int,
    apply_topk_topp: bool,
) -> torch.Tensor:
    """One token id per row of `logits` [rows, vocab], as int64 [rows, 1].

    A row with temperature <= SAMPLING_EPS takes the argmax; the others take
    the Gumbel-max sample of the (top-k / top-p masked) tempered logits, with
    `u` the uniforms, one per logit.
    """
    if all_greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)
    is_greedy = temperatures <= SAMPLING_EPS
    scaled_logits = apply_temperature(logits, temperatures)
    if apply_topk_topp:
        masked_logits = apply_top_k_top_p(scaled_logits, top_k, top_p, topk_candidates)
    else:
        # No request sets top-k/top-p: skip the two 32-step vocabulary
        # bisections (separate graph, warmed in the precompile).
        masked_logits = scaled_logits
    u_clamped = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    gumbel_noise = -torch.log(-torch.log(u_clamped))
    noisy_logits = masked_logits + gumbel_noise
    final_logits = torch.where(is_greedy, logits, noisy_logits)
    return torch.argmax(final_logits, dim=-1, keepdim=True)


def gather_logprobs_impl(
    logits: torch.Tensor, sampled_tokens: torch.Tensor, max_logprobs: int
) -> LogprobsTensors:
    """The sampled token's logprob and rank, then the top `max_logprobs`
    logprobs (none when `max_logprobs` <= 0), from the raw logits."""
    token_ids = sampled_tokens.to(torch.int64)
    token_logits = logits.gather(-1, token_ids)
    token_ranks = (logits >= token_logits).sum(dim=-1, dtype=torch.int32)
    # Normalize in FP32: a BF16 logsumexp/subtraction can round the selected
    # token's logprob to 0.0 and make the top-k probabilities sum to > 1.
    # Only the [N, 1] normalizer is materialized in FP32; the upcast feeds
    # straight into the reduction and the FP32 subtraction is done on the
    # gathered [N, k] values rather than on a full-vocab log_softmax.
    log_normalizers = torch.logsumexp(logits.to(torch.float32), dim=-1, keepdim=True)
    token_logprobs = token_logits.to(torch.float32) - log_normalizers

    if max_logprobs > 0:
        topk_logits, topk_indices = torch.topk(logits, max_logprobs, dim=-1)
        topk_logprobs = topk_logits.to(torch.float32) - log_normalizers
        logprob_token_ids = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)
    else:
        logprob_token_ids = token_ids
        logprobs = token_logprobs

    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids.to(torch.int32),
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
    )
