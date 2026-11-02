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
"""TokenPaddingState uploads its mask only when the padded rows change."""

import torch

from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState


def test_unchanged_padding_skips_the_upload(monkeypatch):
    state = TokenPaddingState.create(16, torch.device("cpu"))
    uploads = []
    original_copy = torch.Tensor.copy_

    def counting_copy(self, src, *args, **kwargs):
        if self is state.local_padding_mask:
            uploads.append(src.clone())
        return original_copy(self, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", counting_copy)

    state.update(1, 16)
    state.update(1, 16)
    assert len(uploads) == 1
    expected = torch.zeros(16, dtype=torch.bool)
    expected[1:16] = True
    assert torch.equal(state.get_local_padding_mask(16), expected)

    state.update(4, 16)
    state.update(4, 8)
    assert len(uploads) == 3
    expected = torch.zeros(16, dtype=torch.bool)
    expected[4:8] = True
    assert torch.equal(state.get_local_padding_mask(16), expected)
