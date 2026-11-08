# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
#   src/vllm_torchtpu/megakernel/kimi_k3/kernels/prefill_moe/__init__.py
"""Packed EP16/TP2 prefill MoE kernels and shared Pallas building blocks.

Provides the FP32-reduction fused schedule for 512- through 8192-token windows
(``fused``) along with the indexed ragged GMM (``indexed_ragged_gmm``) and
pipelined VALU down-projection reduce-scatter (``moe_down_valu``) used on the
8192-token metadata overflow branch.
"""
