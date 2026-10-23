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
"""LoRA shrink and expand ops for a batch that mixes adapters.

Each token selects one slot of the stacked LoRA weights, or -1 for none.
Rather than gathering each token's weights, the ops compute every slot and
keep the selected one with a one-hot mask.
"""

import torch


def one_hot_slots(
    idx: torch.Tensor, num_slots: int, dtype: torch.dtype
) -> torch.Tensor:
    """Returns a [T, num_slots] mask of each token's slot; slots outside the
    range, such as -1, get an all-zero row."""
    slots = torch.arange(num_slots, device=idx.device, dtype=idx.dtype)
    return (idx[:, None] == slots).to(dtype)


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
        onehot: [T, L] mask from one_hot_slots, in the dtype of x.
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
    # [T, L, r] rather than [T, L, d].
    xa = torch.einsum("td,lrd->tlr", x, a)
    return (xa * onehot[:, :, None]).sum(dim=1) * scale


def bgmv_expand(
    buf: torch.Tensor,
    lora_b_stacked: torch.Tensor,
    onehot: torch.Tensor,
) -> torch.Tensor:
    """Projects each token from rank space back out through its adapter.

    Args:
        buf: [T, r] output of bgmv_shrink.
        lora_b_stacked: [L, 1, out, r] LoRA B weights, one per slot.
        onehot: [T, L] mask from one_hot_slots, in the dtype of buf.

    Returns:
        [T, out], the LoRA delta alone; the caller adds it to the base output.
    """
    assert buf.dtype == lora_b_stacked.dtype, (
        f"LoRA weights are {lora_b_stacked.dtype} but activations are "
        f"{buf.dtype}; lora_dtype must match the model dtype"
    )
    b = lora_b_stacked.squeeze(1)
    # Masks before projecting, on the rank side, so the intermediate is
    # [T, L, r] rather than [T, L, out].
    masked = buf[:, None, :] * onehot[:, :, None]
    return torch.einsum("tlr,lor->to", masked, b)
