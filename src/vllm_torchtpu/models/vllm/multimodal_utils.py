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

from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors


def merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors | None,
    is_multimodal: torch.Tensor | None,
) -> torch.Tensor:
    """Write multimodal embeddings into the placeholder rows of `inputs_embeds`.

    `masked_scatter` and boolean indexing lower to data-dependent shapes on
    TPU, so each row gathers its embedding with `index_select` and `where`
    selects it. The number of embeddings must equal the number of placeholders.
    """
    if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
        return inputs_embeds
    if is_multimodal is None:
        raise ValueError("is_multimodal is required when merging multimodal embeddings")

    mm_embeds = _flatten_embeddings(multimodal_embeddings).to(
        device=inputs_embeds.device, dtype=inputs_embeds.dtype
    )
    mask_cpu = is_multimodal.detach().reshape(-1).to(device="cpu", dtype=torch.bool)
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
    merged = torch.where(mask, aligned_media, inputs_embeds)
    if merged.device.type != "tpu":
        return merged

    materialized = fresh_buffer(merged)
    synchronize_tensors(materialized)
    return materialized
