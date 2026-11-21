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
"""Static-shape placement of multimodal embeddings into token embeddings."""

import torch
from vllm.model_executor.models.utils import _flatten_embeddings
from vllm.multimodal.inputs import NestedTensors


def merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors | None,
    is_multimodal: torch.Tensor | None,
) -> torch.Tensor:
    """Write multimodal embeddings into the placeholder rows of `inputs_embeds`.

    `_patch_vllm_merge_multimodal_embeddings` installs this over vLLM's
    `_merge_multimodal_embeddings`. `masked_scatter` and boolean indexing lower
    to data-dependent shapes on TPU, so each row gathers its embedding with
    `index_select` on a prefix sum of the mask and `where` selects it.

    - `inputs_embeds` is `[num_tokens, hidden]` and `is_multimodal` has
      `num_tokens` entries.
    - The count check and the prefix sum read the mask on the host. Pass a host
      mask, as the runner does; a device mask costs a device read.
    - Returns `inputs_embeds` when there is nothing to merge and a new tensor
      otherwise; vLLM's version writes into `inputs_embeds`.
    - Raises `ValueError` if the mask is missing, if the shapes do not match, or
      if the number of embeddings differs from the number of placeholders.
    - Call it eagerly: the host reads cannot be traced into a compiled graph.
    """
    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds
    if is_multimodal is None:
        raise ValueError("is_multimodal is required when merging multimodal embeddings")

    mm_embeds = _flatten_embeddings(multimodal_embeddings).to(
        device=inputs_embeds.device, dtype=inputs_embeds.dtype
    )
    mask_cpu = is_multimodal.detach().reshape(-1).to(device="cpu", dtype=torch.bool)
    if inputs_embeds.dim() != 2 or mask_cpu.numel() != inputs_embeds.shape[0]:
        raise ValueError(
            "Expected [num_tokens, hidden] embeddings and num_tokens mask entries, "
            f"got embeddings of shape {tuple(inputs_embeds.shape)} and "
            f"{mask_cpu.numel()} mask entries"
        )
    expected = int(mask_cpu.sum().item())
    actual = mm_embeds.shape[0]
    if actual != expected:
        raise ValueError(
            f"Attempted to assign {actual} multimodal tokens to {expected} placeholders"
        )
    if actual == 0:
        return inputs_embeds

    media_indices_cpu = mask_cpu.to(torch.int64).cumsum(0).sub_(1)
    media_indices_cpu.clamp_(min=0)
    media_indices = media_indices_cpu.to(device=inputs_embeds.device)
    aligned_media = mm_embeds.index_select(0, media_indices)
    mask = is_multimodal.reshape(-1, 1).to(
        device=inputs_embeds.device, dtype=torch.bool
    )
    return torch.where(mask, aligned_media, inputs_embeds)
