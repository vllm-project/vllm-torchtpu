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
"""The DSv4 collective ids are distinct and match the fused MoE kernel's."""

from vllm_torchtpu.kernels.deepseek_v4 import collective_ids
from vllm_torchtpu.kernels.fused_moe.v2 import kernel as fused_moe_kernel


def test_collective_ids_are_distinct():
    ids = {
        name: value
        for name, value in vars(collective_ids).items()
        if name.endswith("_COLLECTIVE_ID")
    }
    assert len(set(ids.values())) == len(ids), ids


def test_fused_moe_id_matches_its_kernel():
    assert collective_ids.FUSED_MOE_COLLECTIVE_ID == fused_moe_kernel.COLLECTIVE_ID
