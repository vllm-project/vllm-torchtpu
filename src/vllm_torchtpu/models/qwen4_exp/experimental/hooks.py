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

import os

from . import plan

QWEN4_EXP_ARCHITECTURES = (
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
)

_PKG = "vllm_torchtpu.models.qwen4_exp.experimental"
_MODEL = f"{_PKG}.model:Qwen4ExpForCausalLM"
_MTP_MODEL = f"{_PKG}.mtp:Qwen4ExpMTP"


def enabled() -> bool:
    """Whether this directory serves the Qwen4Exp architectures."""
    # owner: dongruizi, remove-by: when models/vllm/qwen4_exp is retired
    selector = os.environ.get(  # experimental-env-read: migration selector
        "VLLM_TPU_QWEN38_FLASH_EXPERIMENTAL", "0"
    )
    return selector in ("1", "true", "True")


def _make_adapter(runner):
    # Deferred: inputs imports the model cache, which only a runner needs.
    from .inputs import Qwen4ExpAdapter

    return Qwen4ExpAdapter(runner)


def _make_draft_proposer(runner, vllm_config):
    # Deferred: the proposer copy imports the shared spec decode stack, which
    # only a runner on the last pipeline stage needs.
    from .spec_decode.eagle3 import Eagle3Proposer

    return Eagle3Proposer(runner, vllm_config)


def register() -> None:
    """Register the experimental model for the Qwen4Exp architectures.

    Called through `models/qwen4_exp.register()` by the shared model
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

    # The runner hands this model its inputs and cache through the adapter.
    from vllm_torchtpu.runner.model_adapters import register_model_adapter

    register_model_adapter(QWEN4_EXP_ARCHITECTURES, _make_adapter)
    # The adapter's metadata walk re-packs block tables by BlockTable version.
    from .runner.metadata import _install_block_table_version

    _install_block_table_version()

    # Served on every agentx config: each worker pinned to its chip's node.
    from vllm_torchtpu.worker.cpu_pinning import register_cpu_pinning

    register_cpu_pinning("numa", *QWEN4_EXP_ARCHITECTURES)

    # The MTP draft runs on this directory's copy of the EAGLE/MTP proposer.
    from vllm_torchtpu.spec_decode.proposer_registry import register_draft_proposer

    register_draft_proposer(("Qwen4ExpMTP",), _make_draft_proposer)

    # kernels/ and ops/ are hashed by the shared rule; the pooled cache layout
    # changes graph argument extents, and the model plan changes graphs.
    register_compile_cache_key(
        "qwen4_exp",
        paths=("vllm_torchtpu/models/qwen4_exp/experimental/pool_layout.py",),
        env_factors=plan.compile_key_factors,
    )
