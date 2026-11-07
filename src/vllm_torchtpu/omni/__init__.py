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
"""vLLM-Omni platform plugin support for Google Cloud TPU."""

from vllm_torchtpu.tpu_info import get_num_chips


def register_omni_tpu_platform() -> str | None:
    """Plugin entry point for vllm_omni.platform_plugins."""
    if get_num_chips() <= 0:
        return None
    return "vllm_torchtpu.omni.platform.OmniTpuPlatform"


def register_omni_general_plugins() -> None:
    """Plugin entry point for vllm_omni.general_plugins.

    vLLM-Omni runs its general plugins in every process before it builds a
    diffusion engine or worker. Returns early unless
    register_omni_tpu_platform() finds TPU chips (get_num_chips() > 0); safe
    to call more than once.
    """
    if register_omni_tpu_platform() is None:
        return
    from vllm_torchtpu.omni.patches import apply_omni_general_plugin_patches

    apply_omni_general_plugin_patches()


__all__ = [
    "register_omni_general_plugins",
    "register_omni_tpu_platform",
]
