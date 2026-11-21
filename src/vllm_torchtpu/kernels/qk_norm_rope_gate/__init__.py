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
from vllm_torchtpu.kernels.qk_norm_rope_gate.kernel import (
    DEFAULT_MAX_TILE,
    MIN_PALLAS_QKV_BYTES,
    can_run_qk_norm_rope_gate,
    legal_row_tile,
    qk_norm_rope_gate,
    qk_norm_rope_gate_rows,
    reference,
    scoped_vmem_limit_bytes,
    unfused_qk_norm_rope_gate_rows,
)

__all__ = [
    "DEFAULT_MAX_TILE",
    "MIN_PALLAS_QKV_BYTES",
    "can_run_qk_norm_rope_gate",
    "legal_row_tile",
    "qk_norm_rope_gate",
    "qk_norm_rope_gate_rows",
    "reference",
    "scoped_vmem_limit_bytes",
    "unfused_qk_norm_rope_gate_rows",
]
