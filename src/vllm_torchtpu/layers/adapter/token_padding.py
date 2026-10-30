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

from vllm_torchtpu import envs

_padding_state: "TokenPaddingState | None" = None


@dataclass
class TokenPaddingState:
    """Track token padding."""

    local_padding_mask: torch.Tensor
    # The padding rows [start, end) the device mask holds after the last
    # update(), or None when unknown (before the first write).
    _written: tuple[int, int] | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, max_local_tokens: int, device) -> "TokenPaddingState":
        local_padding_mask = torch.zeros(
            max_local_tokens, dtype=torch.bool, device=device
        )
        return cls(local_padding_mask=local_padding_mask)

    def update(self, local_valid_tokens: int, local_padded_tokens: int) -> None:
        """Write this rank's token padding mask.

        With TPU_TOKEN_PADDING_SKIP_UNCHANGED the upload is skipped when the
        mask would not change: the mask is a function of the padding rows
        [valid, padded) alone, and every writer goes through this method.
        """
        size = self.local_padding_mask.shape[0]
        # The rows the slice below marks, normalized exactly as Python slicing.
        start, end, _ = slice(local_valid_tokens, local_padded_tokens).indices(size)
        rows = (start, end) if start < end else (0, 0)
        if envs.TPU_TOKEN_PADDING_SKIP_UNCHANGED and rows == self._written:
            return
        host = torch.zeros(size, dtype=torch.bool)
        host[local_valid_tokens:local_padded_tokens] = True
        # Synchronous copy since the host tensor is temporary.
        self.local_padding_mask.copy_(host)
        self._written = rows

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

        from vllm_torchtpu.distributed.pcp import get_pcp_group

        dp_group = get_dp_group()
        pcp_group = get_pcp_group()
        dp_size = int(dp_group.world_size) if dp_group is not None else 1
        pcp_size = int(pcp_group.world_size) if pcp_group is not None else 1
        if dp_size > 1 or pcp_size > 1:
            local_padding = _padding_state.get_local_padding_mask(
                num_tokens // (dp_size * pcp_size)
            ).to(torch.int8)
            # TPU collectives may not support bool, so encode the mask as int8.
            if dp_group is not None and dp_size > 1:
                local_padding = dp_group.all_gather(local_padding, dim=0)
            if pcp_group is not None and pcp_size > 1:
                local_padding = pcp_group.all_gather(local_padding, dim=0)
            is_padding = local_padding.ne(0)
        else:
            is_padding = _padding_state.get_local_padding_mask(num_tokens)
    topk_weights = torch.where(
        is_padding[:, None], torch.zeros_like(topk_weights), topk_weights
    )
    return topk_ids, topk_weights


def mask_router_logits_for_padding(router_logits: torch.Tensor) -> torch.Tensor:
    """Replace padded rows of local router_logits with NaN.

    Used by the fused EP MoE path (`fused_moe_ep`), which takes this rank's
    local tokens and performs top-k routing inside the JAX program. A row of
    NaNs produces non-finite routing scores under every scoring function
    (softmax, sigmoid, sqrtsoftplus), causing `fused_ep_moe_v2` to mark the
    row as unrouted (`topk_weights = 0.0`, `topk_idx = e_total`) so it is
    skipped by the routing histogram, slab scatters, cross-chip DMAs, and
    expert matmuls.
    """
    if _padding_state is None:
        return router_logits
    is_padding = _padding_state.get_local_padding_mask(router_logits.shape[0])
    return torch.where(is_padding[:, None], float("nan"), router_logits)
