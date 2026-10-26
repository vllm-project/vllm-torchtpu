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
"""Token sampling from logits, as the TPU runner samples.

The exporter lowers the TPU runner's sampler (`TPUModelRunner.sample_from_logits`
and `_apply_temperature`) into every artifact, so that an artifact samples as
vLLM does. This module is a copy of it, kept inside the experimental exporter;
`tests/experimental/export/test_sampler_parity.py` checks that the two agree.

A row with temperature <= `SAMPLING_EPS` takes the argmax. Other rows are
sampled with the Gumbel-max trick: divide the logits by the temperature,
optionally mask them to the top-k / top-p set by bisection
(`apply_top_k_top_p`), add Gumbel noise from `u`, and take the argmax. No step
sorts the vocabulary.
"""

import torch

from vllm_torchtpu.layers.adapter.sample.top_k_top_p import apply_top_k_top_p

SAMPLING_EPS = 1e-5


def apply_temperature(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    """Divides logits by the temperature; a zero temperature divides by 1."""
    safe_temperatures = torch.where(temperatures == 0.0, 1.0, temperatures)
    return logits / safe_temperatures


def sample_from_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    u: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    all_greedy: bool = False,
    topk_candidates: int = 0,
    apply_topk_topp: bool = True,
) -> torch.Tensor:
    """Samples one token id per row of logits[R, V]; returns [R, 1].

    Args:
      logits: [R, V].
      temperatures: [R, 1].
      u: uniform values in (0, 1) broadcastable to [R, V]: one per logit.
      top_k: [R, 1]; a row with top_k <= 0 keeps every token.
      top_p: [R, 1]; a row with top_p >= 1 keeps every token.
      all_greedy: every row takes the argmax.
      topk_candidates: see `apply_top_k_top_p`.
      apply_topk_topp: False when no row sets top-k or top-p, which skips
        the two vocabulary bisections.
    """
    if all_greedy:
        return torch.argmax(logits, dim=-1, keepdim=True)
    is_greedy = temperatures <= SAMPLING_EPS
    scaled_logits = apply_temperature(logits, temperatures)
    if apply_topk_topp:
        masked_logits = apply_top_k_top_p(scaled_logits, top_k, top_p, topk_candidates)
    else:
        masked_logits = scaled_logits
    u_clamped = torch.clamp(
        u, min=torch.finfo(u.dtype).tiny, max=1.0 - torch.finfo(u.dtype).eps
    )
    gumbel_noise = -torch.log(-torch.log(u_clamped))
    noisy_logits = masked_logits + gumbel_noise
    final_logits = torch.where(is_greedy, logits, noisy_logits)
    return torch.argmax(final_logits, dim=-1, keepdim=True)
