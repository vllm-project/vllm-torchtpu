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

from dataclasses import dataclass, field

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_torchtpu import envs

_padding_state: "TokenPaddingState | None" = None
# Pairs whose device mask is kept: decode reuses a few (valid, padded) pairs,
# so its steady state builds, uploads and writes nothing.
_MAX_PAIR_MASKS = 4096


@dataclass
class TokenPaddingState:
    """Track token padding."""

    local_padding_mask: torch.Tensor
    on_device: bool = False
    # One immutable device mask per (valid, padded) pair; update() points
    # local_padding_mask at it, so a queued forward never sees a later write.
    _pair_masks: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)

    @classmethod
    def create(cls, max_local_tokens: int, device) -> "TokenPaddingState":
        local_padding_mask = torch.zeros(
            max_local_tokens, dtype=torch.bool, device=device
        )
        return cls(
            local_padding_mask=local_padding_mask,
            on_device=envs.TPU_MOE_PADDING_MASK_ON_DEVICE,
        )

    def update(self, local_valid_tokens: int, local_padded_tokens: int) -> None:
        """Write this rank's token padding mask."""
        if self.on_device:
            self._use_pair_mask(local_valid_tokens, local_padded_tokens)
            return
        host = torch.zeros(self.local_padding_mask.shape[0], dtype=torch.bool)
        host[local_valid_tokens:local_padded_tokens] = True
        # Synchronous copy since the host tensor is temporary.
        self.local_padding_mask.copy_(host)

    def _use_pair_mask(self, local_valid_tokens: int, local_padded_tokens: int) -> None:
        """The same mask, built and uploaded once per pair and then reused."""
        key = (local_valid_tokens, local_padded_tokens)
        mask = self._pair_masks.get(key)
        if mask is None:
            host = torch.zeros(self.local_padding_mask.shape[0], dtype=torch.bool)
            host[local_valid_tokens:local_padded_tokens] = True
            # Synchronous copy since the host tensor is temporary.
            mask = host.to(self.local_padding_mask.device)
            if len(self._pair_masks) < _MAX_PAIR_MASKS:
                self._pair_masks[key] = mask
        self.local_padding_mask = mask

    def get_local_padding_mask(self, num_tokens: int) -> torch.Tensor:
        return self.local_padding_mask[:num_tokens]


def set_padding_state(state: "TokenPaddingState | None") -> None:
    global _padding_state
    _padding_state = state


def mark_padding(num_valid_tokens: int, num_padded_tokens: int) -> None:
    """Describe the rows of the next forward: [num_valid_tokens,
    num_padded_tokens) are padding. A no-op when no padding state is set.

    The mask is shared by every forward on this rank, so a forward whose rows
    differ from the target step's (a drafter) must call this before it runs.
    """
    if _padding_state is not None:
        _padding_state.update(num_valid_tokens, num_padded_tokens)


def local_padding_rows(num_rows: int) -> torch.Tensor | None:
    """This rank's `[num_rows]` padding mask: the stage's own mask where the
    model supplies one, else the runner's mask; None when neither describes
    `num_rows` rows."""
    mask = None
    if is_forward_context_available():
        mask = get_forward_context().additional_kwargs.get("token_padding_mask")
    if mask is None and _padding_state is not None:
        mask = _padding_state.get_local_padding_mask(num_rows)
    if mask is None or mask.shape[0] != num_rows:
        return None
    return mask


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
    # A model may compact a sub-batch after the runner prepared its persistent
    # mask. Such a stage supplies a mask in its own routing-row layout; do not
    # reuse the enclosing batch's valid-token prefix for these rows.
    if is_forward_context_available():
        stage_mask = get_forward_context().additional_kwargs.get("token_padding_mask")
        if stage_mask is not None:
            return topk_ids, topk_weights.masked_fill(stage_mask[:, None], 0)
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
