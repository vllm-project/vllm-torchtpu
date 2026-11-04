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
"""Decode MoE layer of one TP rank, shared expert and TP all-reduce included,
in one Pallas kernel (kernel.py; the all-reduce and its self-test in
all_reduce.py)."""

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce
from vllm_torchtpu.kernels.experimental.moe_tp_decode.kernel import (
    ROW_BUCKETS,
    can_use_moe_tp_decode,
    exact_in_bf16,
    kernel_name,
    moe_tp_decode,
)

__all__ = [
    "ROW_BUCKETS",
    "all_reduce",
    "can_use_moe_tp_decode",
    "exact_in_bf16",
    "moe_tp_decode",
    "kernel_name",
]
