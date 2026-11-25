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
"""The static multimodal embedding merge shared by TPU multimodal models."""

import pytest
import torch

from vllm_torchtpu.models.vllm.multimodal_merge import merge_multimodal_embeddings


def _inputs(device: str | torch.device = "cpu"):
    generator = torch.Generator().manual_seed(3)
    text = torch.randn(7, 4, generator=generator).to(device)
    media = [
        torch.randn(2, 4, generator=generator).to(device),
        torch.randn(1, 4, generator=generator).to(device),
    ]
    mask = torch.tensor([False, True, True, False, False, True, False])
    return text, media, mask.to(device)


def _reference(text, media, mask):
    expected = text.clone()
    expected[mask] = torch.cat(media).to(text.dtype)
    return expected


def test_merge_places_embeddings_at_placeholders_in_order():
    text, media, mask = _inputs()
    merged = merge_multimodal_embeddings(text, media, mask)
    torch.testing.assert_close(merged, _reference(text, media, mask))


def test_merge_without_embeddings_returns_inputs():
    text, _, mask = _inputs()
    assert merge_multimodal_embeddings(text, None, mask) is text
    assert merge_multimodal_embeddings(text, [], mask) is text


def test_merge_rejects_count_mismatch_and_missing_mask():
    text, media, mask = _inputs()
    with pytest.raises(ValueError, match="2 multimodal tokens to 3"):
        merge_multimodal_embeddings(text, media[:1], mask)
    with pytest.raises(ValueError, match="is_multimodal is required"):
        merge_multimodal_embeddings(text, media, None)


def test_merge_casts_embeddings_to_text_dtype():
    text, media, mask = _inputs()
    text = text.to(torch.bfloat16)
    merged = merge_multimodal_embeddings(text, media, mask)
    assert merged.dtype == torch.bfloat16
    torch.testing.assert_close(merged, _reference(text, media, mask))


@pytest.mark.tpu_test
@torch.inference_mode()
def test_merge_on_tpu_matches_cpu():
    device = torch.device("tpu")
    text, media, mask = _inputs()
    expected = merge_multimodal_embeddings(text, media, mask)
    tpu_text, tpu_media, tpu_mask = _inputs(device)
    merged = merge_multimodal_embeddings(tpu_text, tpu_media, tpu_mask)
    torch.testing.assert_close(merged.cpu(), expected)
