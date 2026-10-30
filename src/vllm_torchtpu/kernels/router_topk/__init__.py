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
"""Sort-free top-k expert selection for the MoE router (Pallas)."""

from vllm_torchtpu.kernels.router_topk.kernel import (
    MAX_BLOCK_ROWS,
    NEG,
    router_topk,
    router_topk_softmax,
    select,
    select_softmax,
)

__all__ = [
    "MAX_BLOCK_ROWS",
    "NEG",
    "router_topk",
    "router_topk_softmax",
    "select",
    "select_softmax",
]
