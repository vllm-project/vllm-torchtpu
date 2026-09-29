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
"""Core types and data classes for structured speculative decoding."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SpecDecodeMetadata:
    """Metadata for speculative decoding on Torch/TPU, containing all necessary indices."""  # noqa: E501

    # [padded_logits_length] draft token id validated at each target_logits_indices slot.  # noqa: E501
    draft_token_ids: torch.Tensor
    # [padded_num_reqs] per-request draft-token count (device copy).
    draft_lengths: torch.Tensor
    # [num_reqs] per-request draft-token count (host copy, unpadded).
    draft_lengths_cpu: np.ndarray
    # [padded_logits_length] row into this step's logits/hidden-states for each draft token being validated.  # noqa: E501
    target_logits_indices: torch.Tensor
    # [padded_num_reqs] row into this step's logits/hidden-states for each request's bonus (last sampled) token.  # noqa: E501
    bonus_logits_indices: torch.Tensor
    # [padded_logits_length] row into the input batch for every draft + bonus
    # position the target model forward needs logits for.
    final_logits_indices: torch.Tensor
    # [padded_logits_length] request index each position belongs to; padding
    # uses the out-of-range sentinel num_reqs (not 0).
    segment_ids: torch.Tensor
    # [padded_logits_length] within-request draft-slot index of each position.
    group_indices: torch.Tensor
    # [padded_logits_length] optional per-slot draft flag (0=EMPTY, 1=MODEL, 2=GRAMMAR).
    draft_flags: torch.Tensor | None = None
    # True if any draft token requires model verification (e.g. flag == 1).
    # False if all valid draft tokens are GRAMMAR (flag == 2) and unconditionally accepted.  # noqa: E501
    has_model_drafts: bool = True
