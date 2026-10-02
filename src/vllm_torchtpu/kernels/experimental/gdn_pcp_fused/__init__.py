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

# memory_ref and legacy_unified import each other, and legacy_unified reads
# memory_ref attributes at import time, so legacy_unified has to load first.
# V3 only gets this order by accident of what imports it.
from vllm_torchtpu.kernels.experimental.gdn_pcp_fused import (
    legacy_unified,  # noqa: F401
)
