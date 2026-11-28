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
# Provenance: copied from
# src/vllm_torchtpu/kernels/experimental/fused_moe_decode/__init__.py
# at 7d62e73201948ae35d23dbb6574ed14d5917784e (Qwen3.8 changes from
# served-stack commits 53e2d0ce8e and e57555be64, listed in kernel.py).
#   Owned by models/qwen4_exp from this copy on.
"""Fused MoE decode kernel (routed + shared experts + all-reduce in one call)."""

from .kernel import fused_moe_decode, plan_weight_ring, vmem_bytes

__all__ = ["fused_moe_decode", "plan_weight_ring", "vmem_bytes"]
