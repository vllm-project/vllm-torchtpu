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

# Provenance: copied from src/vllm_torchtpu/models/vllm/qwen4_exp/ple_host.py at
# 75deab3a21c22d2e2f732b18e0c67252f0572ce7.
#   Owned by models/qwen4_exp from this copy on.
"""Host memory available for the CPU PLE table (``PLE_CPU_OFFLOAD``)."""

from pathlib import Path


def host_available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("PLE host capacity requires Linux MemAvailable")
