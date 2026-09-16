# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PunicaWrapper for TorchTPU platform."""

import torch
from vllm.lora.punica_wrapper.punica_cpu import PunicaWrapperCPU
from vllm.lora.punica_wrapper.utils import compute_meta


class PunicaWrapperTPU(PunicaWrapperCPU):
    """PunicaWrapperTPU provides the PyTorch/XLA-compatible Punica kernel interface on TPU."""

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: torch.device | str,
        **kwargs,
    ):
        super().__init__(max_num_batched_tokens, max_batches, device, **kwargs)

    def _update_prefill_metadata(self,
                                 token_lora_tensor: torch.Tensor) -> None:
        # Run compute_meta on CPU because aten::unique_consecutive is not supported on TPU device
        (
            b_seq_start_tensor,
            seq_length_tensor,
            lora_indices_tensor,
            batch_size,
            max_length,
            token_nums,
            no_lora,
        ) = compute_meta(token_lora_tensor.cpu())

        self._seq_start_locs[:b_seq_start_tensor.shape[0]].copy_(
            b_seq_start_tensor.to(self.device))
        self._seq_lengths[:seq_length_tensor.shape[0]].copy_(
            seq_length_tensor.to(self.device))
        self._lora_indices_per_batch[:lora_indices_tensor.shape[0]].copy_(
            lora_indices_tensor.to(self.device))
        self.batch_size = batch_size
        self.max_length = max_length
        self.token_nums = token_nums
        self.no_lora = no_lora

    def add_shrink(
        self,
        y: tuple[torch.Tensor, ...] | torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        super().add_shrink(y, x, lora_a_stacked, scale, **kwargs)
        return y

    def add_expand(
        self,
        y: torch.Tensor,
        x: tuple[torch.Tensor, ...] | torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs=True,
        **kwargs,
    ) -> torch.Tensor:
        super().add_expand(y, x, lora_b_stacked, output_slices, offset_start,
                           add_inputs, **kwargs)
        return y

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        super().add_lora_linear(
            y,
            x,
            lora_a_stacked,
            lora_b_stacked,
            scale,
            output_slices,
            buffer=buffer,
            **kwargs,
        )
        return y

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        super().add_lora_logits(y,
                                x,
                                lora_a_stacked,
                                lora_b_stacked,
                                scale,
                                buffer=buffer,
                                **kwargs)
        return y

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        super().add_lora_embedding(y,
                                   x,
                                   lora_b_stacked,
                                   add_inputs=add_inputs,
                                   **kwargs)
        return y
