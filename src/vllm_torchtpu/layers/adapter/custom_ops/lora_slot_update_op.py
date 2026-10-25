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
"""Donating write of one LoRA adapter into its slot of every stacked tensor."""

import math

import jax
import torch
from torch_tpu._internal import pallas


# Without this op, vLLM's activate_adapter writes an adapter one stacked tensor
# at a time: for every LoRA module, slice and A/B tensor, reset_lora zeroes the
# slot and set_lora copies the adapter's weights into it from the host. On TPU
# each of these writes runs as its own small program and each copy is its own
# host-to-device transfer, so one load costs hundreds of program launches and
# transfers (about 800 writes for Qwen3-0.6B). Each program also has the slot
# index built in as a constant, so every slot compiles its own set.
# This op takes the adapter's slot of every stacked tensor packed into one
# tensor, so a load is one transfer and one program, and it takes the slot
# index as an input, so one compile serves every slot. It is a JAX op so the
# stacked tensors can be donated and updated in place.
def _compute_slot_update(
    stacked_weights: list[jax.Array], slot_weights: jax.Array, slot_index: jax.Array
) -> list[jax.Array]:
    updated, offset = [], 0
    for weights in stacked_weights:
        slot_size = math.prod(weights.shape[1:])
        new_slot = slot_weights[offset : offset + slot_size].reshape(
            (1, *weights.shape[1:])
        )
        updated.append(
            jax.lax.dynamic_update_slice_in_dim(
                weights, new_slot, slot_index[0], axis=0
            )
        )
        offset += slot_size
    assert offset == slot_weights.shape[0], (
        f"slot_weights holds {slot_weights.shape[0]} elements but the slots "
        f"of stacked_weights hold {offset}"
    )
    return updated


_slot_update_op = pallas.jax_op(
    "pallas::lora_slot_update", _compute_slot_update, donate_argnums=(0,)
)
_slot_update_op.register_fake(
    lambda stacked_weights, slot_weights, slot_index: [
        torch.empty_like(weights) for weights in stacked_weights
    ]
)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def write_adapter_to_slot(
    stacked_weights: list[torch.Tensor],
    slot_weights: torch.Tensor,
    slot_index: torch.Tensor,
) -> None:
    """Writes one slot of every stacked tensor from slot_weights, packed in
    stacked_weights order."""
    updated = _slot_update_op(stacked_weights, slot_weights, slot_index)
    for weights, new_weights in zip(stacked_weights, updated):
        # Writes the result back into the donated tensor. XLA aliases the
        # output to that buffer, so no second copy of the weights is made.
        weights.copy_(new_weights)
