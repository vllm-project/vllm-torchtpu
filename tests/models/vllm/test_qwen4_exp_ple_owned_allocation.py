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
"""All-owned output bytes are completely overwritten after poisoned allocation."""

import numpy as np
import pytest
import torch

from tests.models.vllm import test_qwen4_exp_ple_gather as original

make_table = original.make_table


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("tokens", [1, 8192])
def test_all_owned_poisoned_output_is_completely_overwritten(
    make_table, monkeypatch, fp8, shared, tokens
):
    table, checkpoint = make_table(fp8=fp8, shared=shared)
    ids = (torch.arange(tokens * 16, dtype=torch.int32) % 256 + 256).reshape(tokens, 16)
    before_ids = ids.clone()
    carrier = torch.uint8 if fp8 else torch.int16
    before_weight = checkpoint.view(carrier).clone()
    expected = original.expected_bits(checkpoint, ids)
    allocate = np.empty
    poisoned = []

    def empty(shape, *args, **kwargs):
        value = allocate(shape, *args, **kwargs)
        if value.shape == (ids.numel(), table.embedding_dim):
            value.fill(90)
            poisoned.append(value)
        return value

    monkeypatch.setattr(original.ple_storage.np, "empty", empty)
    actual = table.lookup_host(ids)
    assert len(poisoned) == 1
    np.testing.assert_array_equal(
        actual.view(carrier).numpy().reshape(tokens, 16, 160), expected
    )
    assert np.shares_memory(actual.numpy(), poisoned[0])
    assert torch.equal(ids, before_ids)
    assert torch.equal(checkpoint.view(carrier), before_weight)
