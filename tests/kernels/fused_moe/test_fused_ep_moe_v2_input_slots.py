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
"""The interleaved third input slot is chosen per token bucket, not per shape."""

from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2 import host, tuned_block_sizes
from vllm_torchtpu.kernels.fused_moe.v2.kernel import input_slots_for

DSV41 = (pltpu.ChipVersion.TPU_7X, 128, 5120, 2560, 512)


def _rows(tokens_per_shard, ep=8, topk=6, e_total=384, capacity=128):
    return host.ragged_stride_bound(ep * tokens_per_shard, topk, e_total, capacity)


def test_dsv41_decode_buckets_take_the_default_slots():
    for tokens in (16, 32, 64, 128):
        assert input_slots_for(DSV41, _rows(tokens)) == host.OUT_PARITIES


def test_dsv41_prefill_buckets_take_the_tuned_third_slot():
    threshold = tuned_block_sizes.INPUT_SLOTS_MIN_ROWS[DSV41]
    assert input_slots_for(DSV41, threshold) == 3
    assert input_slots_for(DSV41, _rows(2048)) == 3
    assert input_slots_for(DSV41, threshold - 1) == host.OUT_PARITIES


def test_untuned_shapes_take_the_default_slots():
    other = (pltpu.ChipVersion.TPU_7X, 128, 4096, 1024, 512)
    assert input_slots_for(other, 10**6) == host.OUT_PARITIES


def test_a_tuned_shape_without_a_minimum_applies_at_every_bucket(monkeypatch):
    monkeypatch.setattr(tuned_block_sizes, "INPUT_SLOTS_MIN_ROWS", {})
    assert input_slots_for(DSV41, _rows(16)) == 3
