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
"""DeepSeek-V4.1 Sparse MLA + SWA and 2-bank FP8 KV gather kernels."""

from vllm_torchtpu.kernels.experimental.deepseek_v41.sparse_mla.kernel import (
    gather_sparse_fp8_kv,
    sparse_mla_swa_attention,
)

__all__ = ["gather_sparse_fp8_kv", "sparse_mla_swa_attention"]
