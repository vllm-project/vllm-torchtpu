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
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_torchtpu.models.vllm.multimodal_utils import merge_multimodal_embeddings


def _inputs():
    generator = torch.Generator().manual_seed(3)
    text = torch.randn(7, 4, generator=generator)
    media = [
        torch.randn(2, 4, generator=generator),
        torch.randn(1, 4, generator=generator),
    ]
    mask = torch.tensor([False, True, True, False, False, True, False])
    return text, media, mask


def _reference(text, media, mask):
    expected = text.clone()
    expected[mask] = torch.cat(media).to(text.dtype)
    return expected


class _RecordOps(TorchDispatchMode):
    """Records the name of every aten op dispatched while active."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func.overloadpacket))
        return func(*args, **(kwargs or {}))


def test_merge_places_embeddings_at_placeholders_in_order():
    text, media, mask = _inputs()
    merged = merge_multimodal_embeddings(text, media, mask)
    torch.testing.assert_close(merged, _reference(text, media, mask))


def test_merge_without_embeddings_returns_inputs():
    text, _, mask = _inputs()
    assert merge_multimodal_embeddings(text, None, mask) is text
    assert merge_multimodal_embeddings(text, [], mask) is text
    no_media = [torch.empty(0, 4)]
    assert merge_multimodal_embeddings(text, no_media, torch.zeros_like(mask)) is text


def test_merge_rejects_count_mismatch_and_missing_mask():
    text, media, mask = _inputs()
    with pytest.raises(ValueError, match="2 multimodal tokens to 3"):
        merge_multimodal_embeddings(text, media[:1], mask)
    with pytest.raises(ValueError, match="is_multimodal is required"):
        merge_multimodal_embeddings(text, media, None)


def test_merge_rejects_unflattened_embeddings_and_mask_length_mismatch():
    text, media, mask = _inputs()
    with pytest.raises(ValueError, match=r"\[num_tokens, hidden\]"):
        merge_multimodal_embeddings(text.unsqueeze(0), media, mask)
    with pytest.raises(ValueError, match=r"\[num_tokens, hidden\]"):
        merge_multimodal_embeddings(text, media, mask[:-1])


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("tpu", marks=pytest.mark.tpu_test)]
)
@torch.inference_mode()
def test_merge_casts_to_bfloat16_without_data_dependent_ops(device):
    """Serving dtypes (bfloat16 text, float32 media) and a host mask. Boolean
    indexing or `masked_scatter` would also match the values but recompiles on
    TPU for each placeholder count, so the dispatched ops are checked too."""
    text, media, mask = _inputs()
    text = text.to(torch.bfloat16)
    expected = _reference(text, media, mask)
    with _RecordOps() as rec:
        merged = merge_multimodal_embeddings(
            text.to(device), [m.to(device) for m in media], mask
        )
    data_dependent = {
        "aten.index_put",
        "aten.index_put_",
        "aten.masked_scatter",
        "aten.masked_scatter_",
        "aten.nonzero",
    }
    assert not data_dependent.intersection(rec.ops), rec.ops
    assert merged.dtype == torch.bfloat16
    torch.testing.assert_close(merged.cpu(), expected)
