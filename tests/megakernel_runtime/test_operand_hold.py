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
"""host_round_trip_small: small operands go through the host (values unchanged), big ones stay."""

import torch

from vllm_torchtpu.megakernel_runtime.operand_hold import host_round_trip_small


def test_small_operands_are_fresh_copies_with_equal_values():
    base = torch.zeros(4, 256)
    base[1:3] = torch.arange(512.0).reshape(2, 256)  # in-place slice write
    view = base.view(torch.int32)  # dtype reinterpretation: a view
    big = torch.randn(64, 64)
    ops = [base, view, big]
    n = host_round_trip_small(ops, limit=8 << 10)
    assert n == 2
    # (on CPU .cpu().to(cpu) is the identity; on TPU it is a host round-trip copy)
    assert torch.equal(ops[0], base) and torch.equal(ops[1], view)
    assert ops[2] is big  # over the limit: untouched
