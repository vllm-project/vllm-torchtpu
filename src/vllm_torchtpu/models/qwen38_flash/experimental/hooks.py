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
"""Shared-core interface implementations for experimental Qwen3.8-Flash."""

from . import settings

QWEN4_EXP_ARCHITECTURES = (
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
)

_PKG = "vllm_torchtpu.models.qwen38_flash.experimental"
_MODEL = f"{_PKG}.model:Qwen4ExpForCausalLM"
_MTP_MODEL = f"{_PKG}.mtp:Qwen4ExpMTP"


def enabled() -> bool:
    """Whether this directory serves the Qwen4Exp architectures."""
    return bool(settings.VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL)


def _make_adapter(runner):
    # Deferred: inputs imports the model cache, which only a runner needs.
    from .inputs import Qwen4ExpAdapter

    return Qwen4ExpAdapter(runner)


def register() -> None:
    """Register the experimental model for the Qwen4Exp architectures.

    Called through `models/qwen38_flash.register()` by the shared model
    registration, after the default registrations. A no-op unless the
    migration selector is set.
    """
    if not enabled():
        return
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from .config import (
        TPUQwen4ExpConfig,
    )

    for arch in QWEN4_EXP_ARCHITECTURES:
        ModelRegistry.register_model(arch, _MODEL)
        MODELS_CONFIG_MAP[arch] = TPUQwen4ExpConfig
    # Upstream's speculative config maps qwen4_exp to this MTP architecture.
    ModelRegistry.register_model("Qwen4ExpMTP", _MTP_MODEL)

    from vllm_torchtpu.compilation.tpu_compiler import register_compile_cache_key
    from vllm_torchtpu.platforms.tpu_platform import register_tpu_kv_connector

    from .config import PD_CONNECTOR
    from .offload import register_offload_pool

    register_offload_pool()

    register_tpu_kv_connector(PD_CONNECTOR, f"{_PKG}.pd_connector")

    # The runner hands this model its inputs and cache through the adapter.
    from vllm_torchtpu.runner.model_adapters import register_model_adapter

    register_model_adapter(QWEN4_EXP_ARCHITECTURES, _make_adapter)

    # kernels/ and ops/ are hashed by the shared rule; the pooled cache layout
    # changes graph argument extents, and the settings change graphs.
    register_compile_cache_key(
        "qwen38_flash",
        paths=("vllm_torchtpu/models/qwen38_flash/experimental/pool_layout.py",),
        env_factors=settings.compile_key_factors,
    )
