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
"""Experimental fused q/k norm, partial rope and gate split for attention."""

from vllm_torchtpu.kernels.experimental.qk_norm_rope_gate.kernel import (
    qk_norm_rope_gate,
    reference,
)

__all__ = ["qk_norm_rope_gate", "reference"]
