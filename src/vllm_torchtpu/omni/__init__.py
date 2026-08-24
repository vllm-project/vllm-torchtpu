# SPDX-License-Identifier: Apache-2.0
"""vLLM-Omni platform plugin support for Google Cloud TPU."""


def register_omni_tpu_platform() -> str:
    """Plugin entry point for vllm_omni.platform_plugins."""
    return "vllm_torchtpu.omni.platform.OmniTpuPlatform"


__all__ = [
    "register_omni_tpu_platform",
]
