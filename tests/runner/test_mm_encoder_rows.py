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
import torch

from vllm_torchtpu.runner.mm_encoder_rows import (
    EncoderEmbeds,
    EncoderRows,
    split_batch_output,
)


def test_row_handles_gather_a_steps_tokens_at_the_bucket():
    first_batch = torch.arange(24, dtype=torch.float32).reshape(6, 4)  # 2 items + pad
    second_batch = -torch.arange(12, dtype=torch.float32).reshape(3, 4)
    a, b = split_batch_output(first_batch, [3, 2])
    (c,) = split_batch_output(second_batch, [3])
    assert len(a) == 3 and a.shape == (3, 4) and a.ndim == 2
    a.modality = "image"  # what `set_mm_embedding_modality` does
    assert torch.equal(b[1:].tensor(), first_batch[4:5])
    # A step's tokens from two batch outputs, sliced across items, then padded.
    embeds = EncoderEmbeds([a[1:], c, b])[1:5]
    want = torch.cat([first_batch[2:3], second_batch])
    got = embeds.gather(6)
    assert len(embeds) == 4
    assert torch.equal(got[:4], want)
    assert isinstance(embeds.parts[0], EncoderRows)
    assert np.array_equal(embeds.parts[0].rows, [2])
