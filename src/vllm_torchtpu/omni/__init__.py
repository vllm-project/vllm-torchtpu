# SPDX-License-Identifier: Apache-2.0
"""vLLM-Omni platform plugin support for Google Cloud TPU."""

from vllm_torchtpu.tpu_info import get_num_chips


def register_omni_tpu_platform() -> str | None:
    """Plugin entry point for vllm_omni.platform_plugins."""
    if get_num_chips() <= 0:
        return None
    return "vllm_torchtpu.omni.platform.OmniTpuPlatform"


__all__ = [
    "register_omni_tpu_platform",
]
