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
"""Encoder outputs as row handles read back as the items they were stored as."""

import numpy as np
import pytest
import torch

from vllm_torchtpu.runner.mm_encoder_rows import EncoderEmbeds, split_batch_output
from vllm_torchtpu.runner.tpu_runner import _concat_mm_embeds

FIRST = torch.arange(24, dtype=torch.float32).reshape(6, 4)  # 2 items + pad
SECOND = -torch.arange(12, dtype=torch.float32).reshape(3, 4)
THIRD = 100 + torch.arange(8, dtype=torch.float32).reshape(2, 4)


def _items():
    a, b = split_batch_output(FIRST, [3, 2])
    (c,) = split_batch_output(SECOND, [3])
    (d,) = split_batch_output(THIRD, [2])
    return a, b, c, d


def test_row_handles_read_back_their_rows():
    a, b, _, _ = _items()
    assert len(a) == 3 and a.shape == (3, 4) and a.ndim == 2
    a.modality = "image"  # what `set_mm_embedding_modality` does
    assert torch.equal(b[1:].tensor(torch.clone), FIRST[4:5])
    assert np.array_equal(EncoderEmbeds([a[1:], b])[1:].parts[0].rows, [2])


@pytest.mark.parametrize(
    "order",
    [
        "ab",  # one batch output
        "acb",  # two, the second between items of the first
        "dacb",  # three
    ],
)
@pytest.mark.parametrize("window", [slice(0, None), slice(1, 5)])
def test_row_handles_gather_a_steps_tokens_at_the_bucket(order, window):
    items = dict(zip("abcd", _items(), strict=True))
    parts = [items[name] for name in order]
    want = torch.cat([part.source[part.rows] for part in parts])[window]
    embeds = EncoderEmbeds(parts)[window]
    got = embeds.gather(len(want) + 2, max_rows=16, upload=torch.clone)
    assert len(embeds) == len(want) and got.shape == (len(want) + 2, 4)
    assert torch.equal(got[: len(want)], want)


def test_mixed_row_handles_and_tensors_concatenate_in_order():
    a, _, c, _ = _items()
    loose = torch.full((2, 4), 7.0)
    got = _concat_mm_embeds([a, loose, c], torch.clone)
    assert torch.equal(got, torch.cat([FIRST[:3], loose, SECOND]))
