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
"""The FP4 dense-expert VMEM plan: slot count and the limit it requests."""

from vllm_torchtpu.kernels.experimental import moe_dense_expert as kernel

VMEM = 64 * 1024 * 1024
H, BLOCK, EXPERTS = 6144, 128, 128


def test_full_ring_requests_its_plan_with_headroom():
    slots, limit = kernel._fp4_vmem_plan(16, H, 384, EXPERTS, BLOCK, BLOCK, VMEM)
    assert slots == kernel._MAX_FP4_EXPERT_BUFFERS
    assert limit < VMEM


def test_short_ring_requests_all_of_vmem():
    slots, limit = kernel._fp4_vmem_plan(16, H, 1536, 32, BLOCK, BLOCK, VMEM)
    assert 2 <= slots < kernel._MAX_FP4_EXPERT_BUFFERS
    assert limit == VMEM


def test_refuses_when_two_slots_do_not_fit():
    assert kernel._fp4_vmem_plan(16, H, 3072, 16, BLOCK, BLOCK, VMEM) == (0, 0)
