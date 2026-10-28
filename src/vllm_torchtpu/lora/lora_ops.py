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
"""LoRA shrink and expand ops for a batch that mixes adapters.

Each token selects one slot of the stacked LoRA weights, or -1 for none.
Rather than gathering each token's weights, the ops compute every slot and
keep the selected one with a one-hot mask.

Cost: T is the number of tokens, L is max_loras and r is max_lora_rank,
and an adapter of lower rank is zero-padded to r. Every LoRA layer reads
all L slots of A and B and multiplies every token by each of them,
whichever adapters the batch uses. For one slice with input width d and
output width out, the two ops do 2 * T * L * r * (d + out) FLOPs and read
L * r * (d + out) weights, against 2 * T * d * out FLOPs and d * out
weights for the base layer. A merged layer such as QKV pays this once per
slice, each over the full input; the embedding layer runs only the expand.
With d = out = 4096 and max_loras = 16, that adds 12.5% to the base layer
at max_lora_rank = 16 and 50% at max_lora_rank = 64.
"""

import torch


def one_hot_slots(idx: torch.Tensor, num_slots: int) -> torch.Tensor:
    """Returns a [T, num_slots] bool mask of each token's slot; slots outside
    the range, such as -1, get an all-False row."""
    slots = torch.arange(num_slots, device=idx.device, dtype=idx.dtype)
    return idx[:, None] == slots


def bgmv_shrink(
    x: torch.Tensor,
    lora_a_stacked: torch.Tensor,
    onehot: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Projects each token into the rank space of its adapter.

    Args:
        x: [T, d] activations.
        lora_a_stacked: [L, 1, r, d] LoRA A weights, one per slot.
        onehot: [T, L] bool mask from one_hot_slots.
        scale: Factor applied to the result.

    Returns:
        [T, r] in the dtype of x.
    """
    assert x.dtype == lora_a_stacked.dtype, (
        f"LoRA weights are {lora_a_stacked.dtype} but activations are "
        f"{x.dtype}; lora_dtype must match the model dtype"
    )
    a = lora_a_stacked.squeeze(1)
    # Masks after projecting, on the rank side, so the intermediate is
    # [T, L, r] rather than [T, L, d]. Selects with where: multiplying by the
    # mask would let a NaN or Inf in another slot's A reach this token, since
    # 0 * NaN is NaN.
    xa = torch.einsum("td,lrd->tlr", x, a)
    return torch.where(onehot[:, :, None], xa, 0).sum(dim=1) * scale


def bgmv_expand(
    buf: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    onehot: torch.Tensor,
) -> torch.Tensor:
    """Projects each token from rank space back out through its adapter.

    Args:
        buf: [T, r] rank-space rows: the output of bgmv_shrink, or the
            LoRA A rows the embedding layer looked up.
        lora_b_stacked: [L, 1, out, r] LoRA B weights, one per slot.
        onehot: [T, L] bool mask from one_hot_slots.

    Returns:
        [T, out], the LoRA delta alone; the caller adds it to the base output.
    """
    assert buf.dtype == lora_b_stacked.dtype, (
        f"LoRA weights are {lora_b_stacked.dtype} but activations are "
        f"{buf.dtype}; lora_dtype must match the model dtype"
    )
    b = lora_b_stacked.squeeze(1)
    # Masks before projecting, on the rank side, so the intermediate is
    # [T, L, r] rather than [T, L, out]. Selects with where: the embedding
    # layer gives tokens without an adapter slot 0's LoRA A rows, and a
    # multiply would let a NaN or Inf there reach them.
    masked = torch.where(onehot[:, :, None], buf[:, None, :], 0)
    # The matmul sums over every slot, so every slot's B must be finite.
    return torch.einsum("tlr,lor->to", masked, b)
