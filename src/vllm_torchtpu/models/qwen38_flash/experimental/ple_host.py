# SPDX-License-Identifier: Apache-2.0
# Provenance: copied from src/vllm_torchtpu/models/vllm/qwen4_exp/ple_host.py at
# 75deab3a21c22d2e2f732b18e0c67252f0572ce7.
#   Owned by models/qwen38_flash from this copy on.
"""Host memory available for the CPU PLE table (``PLE_CPU_OFFLOAD``)."""

from pathlib import Path


def host_available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("PLE host capacity requires Linux MemAvailable")
