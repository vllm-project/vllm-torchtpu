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
"""Experimental push-dispatch fused EP MoE (MOE_FUSED_EP_KERNEL_IMPL=push).

The fused EP MoE v2 kernel with the activation all-gather replaced by
point-to-point pushes of each shard's routed rows: a SparseCore kernel pushes
the first visit positions before the TensorCore kernel starts, and the
TensorCore kernel pushes the rest with in-kernel remote DMAs. See layer.py.
"""

from vllm_torchtpu.kernels.experimental.push_fused_moe.layer import fused_ep_moe_push

__all__ = ["fused_ep_moe_push"]
