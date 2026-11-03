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
"""The shared and Qwen entries retain the same static input contract."""

import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import sparsecore_topk as shared
from vllm_torchtpu.kernels.qwen4_exp import resident_topk as qwen


@pytest.mark.parametrize("qwen_entry", [False, True], ids=["shared", "qwen"])
@pytest.mark.parametrize("dtype", [np.float32, np.int32], ids=["f32", "raw_i32"])
@pytest.mark.parametrize("n,k", [(16, 16), (16, 4096), (32, 4096)])
def test_valid_contract_routes_to_existing_builder(
    monkeypatch, qwen_entry, dtype, n, k
):
    module = qwen if qwen_entry else shared
    entry = qwen.qwen_sparsecore_topk if qwen_entry else shared.sparsecore_topk
    builder_name = "_sc_topk_resident" if qwen_entry else "_sc_topk_direct"
    scores = np.zeros((3, n), dtype)
    lengths = np.array([0, n - 1, n], np.int32)
    frozen_scores, frozen_lengths = scores.tobytes(), lengths.tobytes()
    result = object()
    calls = []

    def builder(values, count, row_lengths, **kwargs):
        calls.append((values, count, row_lengths, kwargs))
        return result

    # Only the device builder is replaced; the real public body and shared
    # validation execute. The existing API deliberately permits k > n.
    monkeypatch.setattr(module, builder_name, builder)
    assert (
        entry.__wrapped__(
            scores, k, lengths, write_empty_rows=False,
            scheduling_group_id=7, return_scores=True,
        ) is result
    )
    assert len(calls) == 1
    values, count, actual_lengths, kwargs = calls[0]
    assert values is scores
    assert count == k
    np.testing.assert_array_equal(actual_lengths, lengths)
    assert kwargs == {
        "p": shared._pick_partition(3, n, k),
        "write_empty": False,
        "return_scores": True,
        "scheduling_group_id": 7,
    }
    assert scores.tobytes() == frozen_scores
    assert lengths.tobytes() == frozen_lengths


@pytest.mark.parametrize("qwen_entry", [False, True], ids=["shared", "qwen"])
@pytest.mark.parametrize(
    "shape,dtype,k,error",
    [
        ((16,), np.float32, 16, "scores must be 2D, got (16,)"),
        ((2, 3, 16), np.int32, 16, "scores must be 2D, got (2, 3, 16)"),
        ((3, 16), np.float16, 16, "scores must be f32 or i32, got float16"),
        ((3, 16), np.int16, 16, "scores must be f32 or i32, got int16"),
        ((3, 17), np.float32, 16, "n=17 must be a multiple of 16"),
        ((3, 16), np.float32, 0,
         "k=0 must be a positive multiple of 16, at most 4096"),
        ((3, 16), np.int32, -16,
         "k=-16 must be a positive multiple of 16, at most 4096"),
        ((3, 16), np.float32, 17,
         "k=17 must be a positive multiple of 16, at most 4096"),
        ((3, 16), np.int32, 4112,
         "k=4112 must be a positive multiple of 16, at most 4096"),
        # Multiple invalid inputs exercise the original validation order.
        ((17,), np.int16, 0, "scores must be 2D, got (17,)"),
        ((3, 17), np.float16, 0, "scores must be f32 or i32, got float16"),
        ((3, 17), np.float32, 0, "n=17 must be a multiple of 16"),
    ],
)
def test_invalid_contract_keeps_original_errors(
    monkeypatch, qwen_entry, shape, dtype, k, error
):
    module = qwen if qwen_entry else shared
    entry = qwen.qwen_sparsecore_topk if qwen_entry else shared.sparsecore_topk
    builder_name = "_sc_topk_resident" if qwen_entry else "_sc_topk_direct"

    def forbidden_builder(*args, **kwargs):
        pytest.fail("invalid inputs reached the device builder")

    monkeypatch.setattr(module, builder_name, forbidden_builder)
    with pytest.raises(ValueError) as exc:
        entry.__wrapped__(np.zeros(shape, dtype), k)
    assert str(exc.value) == error


def test_qwen_uses_the_actual_shared_validation_helper():
    assert qwen._validate_topk_inputs is shared._validate_topk_inputs
