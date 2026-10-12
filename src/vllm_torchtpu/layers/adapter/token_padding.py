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

import contextlib
from dataclasses import dataclass, field

import torch

_padding_state: "TokenPaddingState | None" = None


@dataclass
class TokenPaddingState:
    """Track token padding."""

    local_padding_mask: torch.Tensor
    # The valid prefix, kept once for all MoE layers.
    local_valid_tokens: torch.Tensor | None = None
    # (valid, padded) counts the mask was last written for.
    _last_counts: tuple[int, int] | None = field(default=None, init=False)
    _mask_has_true: bool = field(default=True, init=False)
    _zero_mask_dev: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )
    _mask_cache: dict[tuple[int, int], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )
    _valid_cache: dict[int, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )

    @classmethod
    def create(cls, max_local_tokens: int, device) -> "TokenPaddingState":
        local_padding_mask = torch.zeros(
            max_local_tokens, dtype=torch.bool, device=device
        )
        local_valid_tokens = torch.zeros(1, dtype=torch.int32, device=device)
        return cls(
            local_padding_mask=local_padding_mask,
            local_valid_tokens=local_valid_tokens,
        )

    def update(self, local_valid_tokens: int, local_padded_tokens: int) -> None:
        """Write this rank's token padding mask if its counts changed.

        Caches device-resident mask and valid-count tensors per
        ``(local_valid_tokens, local_padded_tokens)`` and skips rewriting
        ``local_padding_mask`` when transitioning between unpadded sizes,
        eliminating synchronous host-to-device copies on mixed-step transitions.
        """
        valid = int(local_valid_tokens)
        padded = int(local_padded_tokens)
        counts = (valid, padded)
        if counts == self._last_counts:
            return
        self._last_counts = counts
        dev = self.local_padding_mask.device
        need_true = valid < padded
        if need_true:
            mask_dev = self._mask_cache.get(counts)
            if mask_dev is None:
                host = torch.zeros(
                    self.local_padding_mask.shape[0], dtype=torch.bool
                )
                host[valid:padded] = True
                mask_dev = host.to(device=dev)
                if len(self._mask_cache) < 128:
                    self._mask_cache[counts] = mask_dev
            self.local_padding_mask.copy_(mask_dev)
            self._mask_has_true = True
        elif self._mask_has_true:
            if self._zero_mask_dev is None:
                self._zero_mask_dev = torch.zeros_like(self.local_padding_mask)
            self.local_padding_mask.copy_(self._zero_mask_dev)
            self._mask_has_true = False
        if self.local_valid_tokens is not None:
            valid_dev = self._valid_cache.get(valid)
            if valid_dev is None:
                valid_dev = torch.tensor(
                    [valid], dtype=torch.int32, device=dev
                )
                if len(self._valid_cache) < 256:
                    self._valid_cache[valid] = valid_dev
            self.local_valid_tokens.copy_(valid_dev)

    def get_local_padding_mask(self, num_tokens: int) -> torch.Tensor:
        return self.local_padding_mask[:num_tokens]

    def get_local_valid_tokens(self) -> torch.Tensor | None:
        """This engine's valid-token count for the step."""
        return self.local_valid_tokens


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


def get_local_valid_tokens() -> torch.Tensor | None:
    """Return this rank's persistent valid-prefix length, when configured."""
    if _padding_state is None:
        return None
    return _padding_state.get_local_valid_tokens()


def retain_padding_graph_input(output: torch.Tensor) -> torch.Tensor:
    """Keep the padding-state buffer in every shape-specialized MoE graph."""
    if _padding_state is None:
        return output
    value = _padding_state.local_padding_mask.reshape(-1)[0].to(output.dtype)
    return output + (value - value)


def padding_mask_for(
    num_tokens: int, is_local_tensor: bool = False
) -> torch.Tensor | None:
    """Returns the padded-row mask for a [num_tokens, ...] tensor.

    None when no padding state is set. Unless the tensor is local, the
    rank-local masks are all-gathered over DP, then over PCP, matching the
    row order of a tensor gathered the same way.
    """
    if _padding_state is None:
        return None
    if is_local_tensor:
        return _padding_state.get_local_padding_mask(num_tokens)
    from vllm.distributed.parallel_state import get_dp_group

    from vllm_torchtpu.distributed.pcp import get_pcp_group

    dp_group = get_dp_group()
    pcp_group = get_pcp_group()
    dp_size = int(dp_group.world_size) if dp_group is not None else 1
    pcp_size = int(pcp_group.world_size) if pcp_group is not None else 1
    if dp_size == 1 and pcp_size == 1:
        return _padding_state.get_local_padding_mask(num_tokens)
    local_padding = _padding_state.get_local_padding_mask(
        num_tokens // (dp_size * pcp_size)
    ).to(torch.int8)
    # TPU collectives may not support bool, so encode the mask as int8.
    if dp_group is not None and dp_size > 1:
        local_padding = dp_group.all_gather(local_padding, dim=0)
    if pcp_group is not None and pcp_size > 1:
        local_padding = pcp_group.all_gather(local_padding, dim=0)
    return local_padding.ne(0)


def zero_scores_for_padding(
    scores: torch.Tensor, is_local_tensor: bool = False
) -> torch.Tensor:
    """Zeroes the router scores of padded tokens.

    A row of zeros selects no expert in the few-token kernel's in-kernel
    router (its weights renormalise to zero and count no hits).
    """
    is_padding = padding_mask_for(scores.shape[0], is_local_tensor)
    if is_padding is None:
        return scores
    return torch.where(is_padding[:, None], torch.zeros_like(scores), scores)


@contextlib.contextmanager
def use_padding_state(state: "TokenPaddingState | None"):
    """Temporarily bind `_padding_state` for a separately compiled loop graph."""
    global _padding_state
    prev = _padding_state
    _padding_state = state
    try:
        yield
    finally:
        _padding_state = prev


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
            (i.e. not gathered across DP or PCP ranks). If True, skips the
            mask all_gather.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The (topk_ids, topk_weights) tensors
            with routing weights zeroed for padded tokens.
    """
    is_padding = padding_mask_for(topk_ids.shape[0], is_local_tensor)
    if is_padding is None:
        return topk_ids, topk_weights
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
