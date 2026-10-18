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
# Imported for their registration side effects; the out-of-tree
# RowParallelLinear in `linear` only affects layers built after the import, so
# this module must be imported before any model is constructed.
from vllm_torchtpu.layers.adapter import attention as attention
from vllm_torchtpu.layers.adapter import linear as linear
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm import register_models
from vllm_torchtpu.tokenizers import register_tokenizers

logger = init_logger(__name__)


def register_layers() -> None:
    register_models()
    register_tokenizers()
    _install_frontend_patches()


def _install_frontend_patches() -> None:
    # API-server patches. vLLM loads general plugins when it parses the CLI
    # and builds engine arguments, before the renderer and the OpenAI
    # serving objects exist. A failure here leaves stock vLLM behaviour.
    try:
        from vllm_torchtpu.frontend.lazy_reasoning_count import (
            maybe_install_lazy_reasoning_count,
        )

        maybe_install_lazy_reasoning_count()
    except Exception:
        logger.warning(
            "Lazy reasoning count install failed; chat streaming keeps vLLM's "
            "per-delta reasoning-token recount.",
            exc_info=True,
        )
