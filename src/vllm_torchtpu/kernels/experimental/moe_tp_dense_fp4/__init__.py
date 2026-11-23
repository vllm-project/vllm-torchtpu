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
"""TP-sliced fp4 decode MoE, dense over tokens and sparse over hit experts (prototype, not wired)."""
from vllm_torchtpu.kernels.experimental.moe_tp_dense_fp4.kernel import reference, tp_dense_moe

__all__ = ["reference", "tp_dense_moe"]
