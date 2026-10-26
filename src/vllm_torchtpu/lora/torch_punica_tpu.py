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
"""Punica wrapper that runs multi-LoRA inside the compiled TPU backbone."""

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_torchtpu.lora.lora_ops import bgmv_expand, bgmv_shrink, one_hot_slots

if TYPE_CHECKING:
    from vllm.lora.layers import LoRAMapping


class PunicaWrapperTPU(PunicaWrapperBase):
    """Punica wrapper for TPU.

    PunicaWrapperTPU maintains the per-step LoRA mapping for Multi-LoRA and
    provides the punica ops, implemented in pure torch in lora_ops.py. It is
    created by get_punica_wrapper when load_lora_model calls
    create_lora_manager.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: torch.device | str,
        **kwargs,
    ):
        super().__init__(max_num_batched_tokens, max_batches, device)

        # PunicaWrapperBase allocates these buffers with torch.empty as int64.
        # Replace the ones the TPU ops read with zero-filled int32 buffers:
        # int32 matches input_ids, and zeros keep entries never written valid
        # as embedding offsets.
        def index_buffer() -> torch.Tensor:
            return torch.zeros(max_num_batched_tokens, dtype=torch.int32, device=device)

        self._token_lora_indices = index_buffer()
        self._sampler_indices = index_buffer()
        self._embeddings_indices = index_buffer()

    def update_metadata(
        self,
        mapping: "LoRAMapping",
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
        **kwargs,
    ) -> None:
        # Id 0 means no adapter. An id without a slot raises KeyError.
        id_to_slot = {
            lora_id: slot
            for slot, lora_id in enumerate(lora_index_to_id)
            if lora_id is not None
        }
        id_to_slot[0] = -1
        token_slots = np.array(
            [id_to_slot[i] for i in mapping.index_mapping], dtype=np.int32
        )
        sampler_slots = np.array(
            [id_to_slot[i] for i in mapping.prompt_mapping], dtype=np.int32
        )
        _write_head(self._token_lora_indices, token_slots)
        _write_head(self._sampler_indices, sampler_slots)
        _write_head(
            self._embeddings_indices, np.maximum(token_slots, 0) * np.int32(vocab_size)
        )
        # This step's lengths, as PunicaWrapperBase._update_base_metadata
        # records them; index 2 belongs to a buffer this wrapper never writes.
        self.indices_len[0] = len(token_slots)
        self.indices_len[1] = len(sampler_slots)
        self.indices_len[3] = len(token_slots)

    def add_shrink(
        self,
        y: tuple[torch.Tensor, ...] | torch.Tensor | None,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ) -> torch.Tensor:
        """Performs GEMM for multiple slices of lora_a.

        Semantics:
            for i in range(len(lora_a_stacked)):
                out[i] = (x @ lora_a_stacked[i]) * scale

        Each token uses the adapter in its slot, and a token without one gets
        zeros. out is a new tensor in x.dtype, the dtype add_expand consumes,
        so the y passed in is not used.

        Args:
            y: The fp32 shrink buffer vLLM's layers pass in.
            x: Input tensor. (num_tokens, in_features)
            lora_a_stacked: lora_a's weights. lora_a_stacked[i]:
                (max_loras, 1, max_lora_rank, in_features)
            scale: Scaling factor.

        Returns:
            (n_slices, num_tokens, max_lora_rank)
        """
        idx = self._token_lora_indices[: x.size(0)]
        onehot = one_hot_slots(idx, lora_a_stacked[0].shape[0])
        return torch.stack([bgmv_shrink(x, a, onehot, scale) for a in lora_a_stacked])

    def add_expand(
        self,
        y: torch.Tensor,
        x: tuple[torch.Tensor, ...] | torch.Tensor,
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Performs GEMM for multiple slices of lora_b.

        Semantics:
            offset = offset_start
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
                offset += slice

        Each token uses the adapter in its slot, and a token without one is
        left unchanged.

        Args:
            y: Output tensor. (num_tokens, out_features)
            x: Output of add_shrink. (n_slices, num_tokens, max_lora_rank)
            lora_b_stacked: lora_b's weights. lora_b_stacked[i]:
                (max_loras, 1, output_slices[i], max_lora_rank)
            output_slices: Every slice's size.
            offset_start: The column of y where the first slice starts.
            add_inputs: If False, the result holds the LoRA output alone.
                Defaults to True. vLLM's layers pass it as add_input.

        Returns:
            The updated y as a new tensor.
        """
        torch._assert(
            tuple(b.shape[2] for b in lora_b_stacked) == tuple(output_slices),
            f"output_slices {output_slices} do not match the LoRA B widths",
        )
        idx = self._token_lora_indices[: y.size(0)]
        onehot = one_hot_slots(idx, lora_b_stacked[0].shape[0])
        delta = torch.cat(
            [bgmv_expand(x[i], b, onehot) for i, b in enumerate(lora_b_stacked)], dim=-1
        )
        add_inputs = _resolve_add_inputs(add_inputs, kwargs)
        return _add_delta(y, delta, offset_start, add_inputs)

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Applies LoRA for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        The layer has already looked the tokens up in LoRA A, so only the
        expand remains. Each token uses the adapter in its slot.

        Args:
            y: Base embeddings. (num_tokens, embedding_dim)
            x: The tokens' rows of LoRA A. (num_tokens, max_lora_rank)
            lora_b_stacked: (max_loras, 1, embedding_dim, max_lora_rank)
            add_inputs: If False, the result holds the LoRA output alone.
                Defaults to True. vLLM's layers pass it as add_input.

        Returns:
            The updated y as a new tensor.
        """
        idx = self._token_lora_indices[: x.size(0)]
        onehot = one_hot_slots(idx, lora_b_stacked.shape[0])
        delta = bgmv_expand(x, lora_b_stacked, onehot)
        add_inputs = _resolve_add_inputs(add_inputs, kwargs)
        return _add_delta(y, delta, 0, add_inputs)

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
        """Applies LoRA for linear layers: add_shrink, then add_expand.

        Semantics:
            offset = 0
            for a, b, slice in zip(lora_a_stacked, lora_b_stacked, output_slices):
                y[:, offset:offset+slice] += (x @ a @ b) * scale
                offset += slice

        Args:
            y: Output tensor. (num_tokens, out_features)
            x: Input tensor. (num_tokens, in_features)
            lora_a_stacked: lora_a's weights. lora_a_stacked[i]:
                (max_loras, 1, max_lora_rank, in_features)
            lora_b_stacked: lora_b's weights. lora_b_stacked[i]:
                (max_loras, 1, output_slices[i], max_lora_rank)
            scale: Scaling factor.
            output_slices: Every slice's size.
            buffer: Shrink buffer in vLLM's add_lora_linear signature. Unused.

        Returns:
            The updated y as a new tensor.
        """
        shrunk = self.add_shrink(buffer, x, lora_a_stacked, scale)
        return self.add_expand(y, shrunk, lora_b_stacked, output_slices, **kwargs)

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale: float,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Applies LoRA for LogitsProcessorWithLoRA.

        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        x holds one row per request, so each row uses the adapter of its
        request from the sampler mapping. Vocab padding columns of y are left
        unchanged.

        Args:
            y: Logits. (num_reqs, padded_vocab_size)
            x: Hidden states. (num_reqs, hidden_size)
            lora_a_stacked: (max_loras, 1, max_lora_rank, hidden_size)
            lora_b_stacked: (max_loras, 1, vocab_size, max_lora_rank)
            scale: Scaling factor.
            buffer: Shrink buffer in vLLM's add_lora_logits signature. Unused.

        Returns:
            The updated y as a new tensor.
        """
        idx = self._sampler_indices[: x.size(0)]
        onehot = one_hot_slots(idx, lora_a_stacked.shape[0])
        delta = bgmv_expand(
            bgmv_shrink(x, lora_a_stacked, onehot, scale), lora_b_stacked, onehot
        )
        return _add_delta(y, delta, 0, True)

    def apply_lora_full_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        weight_stacked: torch.Tensor,
        bias_stacked: torch.Tensor,
        module_enabled: torch.Tensor,
    ) -> torch.Tensor | None:
        """Applies an adapter's full classification head. Not supported on TPU.

        ClassificationHeadWithLoRA calls this for cross-encoder models when an
        adapter saves its own classification head through modules_to_save.
        Multi-LoRA on TPU targets generative models, so there is no TPU
        implementation.
        """
        raise NotImplementedError(
            "LoRA adapters that save a full classification head "
            "(modules_to_save) are not supported on TPU."
        )


def _write_head(buf: torch.Tensor, values: np.ndarray) -> None:
    """Writes values to the head of buf. The runner pads the mapping to the
    bucket the graph reads, so no stale entry is read."""
    n = len(values)
    assert n <= buf.shape[0], (
        f"LoRA mapping has {n} entries but the index buffer holds {buf.shape[0]}; "
        "the buffer must hold the largest compile size"
    )
    buf[:n].copy_(torch.from_numpy(values))


def _resolve_add_inputs(add_inputs: bool | None, kwargs: dict) -> bool:
    """vLLM's PunicaWrapperBase names the flag add_inputs, but its LoRA layers
    pass add_input. Without this, add_input=False would fall into kwargs and
    the wrapper would still add the inputs."""
    add_input = kwargs.pop("add_input", None)
    torch._assert(
        add_inputs is None or add_input is None,
        f"got both add_inputs={add_inputs} and add_input={add_input}",
    )
    if add_inputs is not None:
        return add_inputs
    return True if add_input is None else add_input


def _add_delta(
    y: torch.Tensor, delta: torch.Tensor, offset: int, add_inputs: bool
) -> torch.Tensor:
    """Lays delta over y's columns from offset on. Columns it does not reach,
    such as the vocab padding of logits, get no delta."""
    # F.pad crops delta on a negative width instead of failing.
    torch._assert(
        offset >= 0 and offset + delta.shape[-1] <= y.shape[-1],
        f"LoRA delta of width {delta.shape[-1]} at column {offset} does not fit "
        f"y of width {y.shape[-1]}",
    )
    delta = F.pad(delta, (offset, y.shape[-1] - offset - delta.shape[-1]))
    return y + delta if add_inputs else delta
