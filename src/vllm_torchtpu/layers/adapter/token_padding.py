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

from dataclasses import dataclass

import torch

_padding_state: "TokenPaddingState | None" = None


@dataclass
class TokenPaddingState:
    """Track token padding."""

    local_padding_mask: torch.Tensor

    @classmethod
    def create(cls, max_local_tokens: int, device) -> "TokenPaddingState":
        local_padding_mask = torch.zeros(
            max_local_tokens, dtype=torch.bool, device=device
        )
        return cls(local_padding_mask=local_padding_mask)

    def update(self, local_valid_tokens: int, local_padded_tokens: int) -> None:
        """Write this rank's token padding mask."""
        host = torch.zeros(self.local_padding_mask.shape[0], dtype=torch.bool)
        host[local_valid_tokens:local_padded_tokens] = True
        # Synchronous copy since the host tensor is temporary.
        self.local_padding_mask.copy_(host)

    def get_local_padding_mask(self, num_tokens: int) -> torch.Tensor:
        return self.local_padding_mask[:num_tokens]


def set_padding_state(state: "TokenPaddingState | None") -> None:
    global _padding_state
    _padding_state = state


def zero_routing_weights_for_padding(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor, is_local_tensor: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set routing weights to 0 for padded tokens.

    A padded token keeps its selected expert ids but gets zero gate
    weight, which tells the fused MoE to skip the expert computation.

    Args:
        topk_ids: Selected expert IDs for each token.
        topk_weights: Routing weights for each token.
        is_local_tensor: Whether topk_ids and topk_weights are local tensors
            (i.e. not gathered across DP ranks). If True, skips DP all_gather.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The (topk_ids, topk_weights) tensors
            with routing weights zeroed for padded tokens.
    """
    if _padding_state is None:
        return topk_ids, topk_weights
    num_tokens = topk_ids.shape[0]
    if is_local_tensor:
        is_padding = _padding_state.get_local_padding_mask(num_tokens)
    else:
        from vllm.distributed.parallel_state import get_dp_group

        dp_group = get_dp_group()
        dp_size = dp_group.world_size
        if dp_size > 1:
            local_padding = _padding_state.get_local_padding_mask(num_tokens // dp_size)
            # TPU collectives may not support bool, so encode the mask as int8.
            gathered_padding = dp_group.all_gather(local_padding.to(torch.int8), dim=0)
            is_padding = gathered_padding.ne(0)
        else:
            is_padding = _padding_state.get_local_padding_mask(num_tokens)
    topk_weights = torch.where(
        is_padding[:, None], torch.zeros_like(topk_weights), topk_weights
    )
    return topk_ids, topk_weights
