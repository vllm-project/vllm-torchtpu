# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU Multi-LoRA Tensorized Static Weight Pool and Einsum Dispatch Engine.

Implements static stacked LoRA pools and XLA-friendly einsum computation with StableHLO select.
"""

from typing import Optional

import torch
import torch.nn as nn
from vllm.config.lora import LoRAConfig


class TPUMultiLoRAPool:
    """Manages static tensorized LoRA weight pools on TPU HBM.

    Pool slots are 0-indexed [0 .. max_loras - 1].
    lora_indices == 0 denotes Base Model (no-LoRA).
    lora_indices > 0 maps to slot = lora_indices - 1.
    """

    def __init__(
        self,
        max_loras: int,
        max_lora_rank: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.device = torch.device(device)
        self.dtype = dtype
        self.adapter_id_to_slot: dict[int, int] = {}
        self.free_slots: list[int] = list(range(max_loras))

    def get_or_allocate_slot(self, adapter_id: int) -> int:
        if adapter_id in self.adapter_id_to_slot:
            return self.adapter_id_to_slot[adapter_id]
        if not self.free_slots:
            raise RuntimeError(
                f"Exceeded max_loras ({self.max_loras}) concurrent TPU slots.")
        slot = self.free_slots.pop(0)
        self.adapter_id_to_slot[adapter_id] = slot
        return slot

    def free_slot(self, adapter_id: int) -> Optional[int]:
        slot = self.adapter_id_to_slot.pop(adapter_id, None)
        if slot is not None:
            self.free_slots.append(slot)
        return slot


class TPUMultiLoRALinear(nn.Module):
    """XLA-optimized Linear Layer with Multi-LoRA Static Einsum and StableHLO Select."""

    def __init__(
        self,
        base_layer: nn.Module,
        max_loras: int,
        max_lora_rank: int,
        lora_config: LoRAConfig,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.max_loras = max_loras
        self.max_lora_rank = max_lora_rank
        self.lora_config = lora_config

        self.input_size = getattr(base_layer, "input_size", None) or getattr(
            base_layer, "in_features", None)
        self.output_size = getattr(base_layer, "output_size", None) or getattr(
            base_layer, "out_features", None)

        if hasattr(base_layer, "output_size_per_partition"):
            self.output_size = base_layer.output_size_per_partition

        device = getattr(base_layer, "weight",
                         torch.empty(0)).device if hasattr(
                             base_layer, "weight") else torch.device("cpu")
        dtype = getattr(base_layer, "weight",
                        torch.empty(0, dtype=torch.bfloat16)).dtype if hasattr(
                            base_layer, "weight") else lora_config.lora_dtype

        # Allocate static pools (Size: max_loras)
        self.register_buffer(
            "lora_a_pool",
            torch.zeros(
                (max_loras, max_lora_rank, self.input_size),
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "lora_b_pool",
            torch.zeros(
                (max_loras, self.output_size, max_lora_rank),
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "scaling_pool",
            torch.zeros((max_loras, ), device=device, dtype=torch.float32),
        )

    def set_lora_weights(
        self,
        slot: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scaling: float = 1.0,
    ) -> None:
        """Sets LoRA weights into the pre-allocated static slot."""
        r = lora_a.shape[0]
        assert slot < self.max_loras, f"Slot {slot} exceeds max_loras {self.max_loras}"
        self.lora_a_pool[slot, :r, :].copy_(lora_a, non_blocking=True)
        self.lora_b_pool[slot, :, :r].copy_(lora_b, non_blocking=True)
        self.scaling_pool[slot] = scaling

    def clear_lora_slot(self, slot: int) -> None:
        """Zeros out a slot when adapter is evicted."""
        self.lora_a_pool[slot].zero_()
        self.lora_b_pool[slot].zero_()
        self.scaling_pool[slot] = 0.0

    def forward(
        self,
        x: torch.Tensor,
        lora_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        base_out = self.base_layer(x)
        bias = None
        if isinstance(base_out, tuple):
            base_out, bias = base_out

        if lora_indices is None:
            return (base_out, bias) if bias is not None else base_out

        has_lora = (lora_indices > 0).unsqueeze(-1)
        safe_slots = torch.clamp(lora_indices - 1, min=0)

        lora_a_selected = self.lora_a_pool[safe_slots]
        lora_b_selected = self.lora_b_pool[safe_slots]
        scales = self.scaling_pool[safe_slots].unsqueeze(-1)

        h = torch.einsum("ti, tri -> tr", x, lora_a_selected)
        delta = torch.einsum("tr, tor -> to", h, lora_b_selected) * scales.to(
            x.dtype)

        out = torch.where(has_lora, base_out + delta, base_out)
        return (out, bias) if bias is not None else out
