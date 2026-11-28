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
"""The Mamba index accessors return the group's own AttentionMetadata fields."""

import pytest
import torch

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.qwen4_exp.experimental.metadata_access import (
    mamba_ckpt_of,
    mamba_state_of,
)


def _metadata(state, ckpt):
    return AttentionMetadata(
        input_positions=None, mamba_state_indices=state, mamba_ckpt_indices=ckpt
    )


@pytest.mark.parametrize("with_ckpt", [False, True])
def test_accessors_return_group_indices(with_ckpt):
    state = torch.tensor([3, 0, 7], dtype=torch.int32)
    ckpt = torch.tensor([[1, 2], [4, 5], [6, 8]], dtype=torch.int32)
    md = _metadata(state, ckpt if with_ckpt else None)
    assert mamba_state_of(md) is state
    assert mamba_ckpt_of(md) is (ckpt if with_ckpt else None)


def test_accessors_for_group_without_mamba():
    md = _metadata(None, None)
    assert mamba_state_of(md) is None
    assert mamba_ckpt_of(md) is None


def test_groups_keep_their_own_indices():
    groups = [_metadata(torch.tensor([g, g + 1]), torch.tensor([[g]])) for g in (0, 5)]
    assert [mamba_state_of(md).tolist() for md in groups] == [[0, 1], [5, 6]]
    assert [mamba_ckpt_of(md).tolist() for md in groups] == [[[0]], [[5]]]
