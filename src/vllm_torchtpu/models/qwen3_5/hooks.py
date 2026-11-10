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
"""The Qwen3.5 config hook: TPU serving defaults for Qwen3.5 engines.

``TPUQwen35MoeConfig`` is registered in vLLM's ``MODELS_CONFIG_MAP`` for the
architectures in ``settings.ARCHITECTURES``, so vLLM calls it from
``VllmConfig.__post_init__`` (before the platform's
``check_and_update_config``) for those models only. It keeps vLLM's own
Qwen3.5 checks, validates ``additional_config["qwen3_5"]``, and applies
the TPU defaults below. Each default fills a value only when the user left it
unset, so an explicit command-line value always wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.models.config import Qwen3_5ForConditionalGenerationConfig

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.qwen3_5 import settings

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def apply_serving_defaults(vllm_config: VllmConfig) -> None:
    """Fill the Qwen3.5 TPU serving defaults the user did not set."""


class TPUQwen35MoeConfig(Qwen3_5ForConditionalGenerationConfig):
    """vLLM's Qwen3.5 checks, the Qwen3.5 settings, and the TPU defaults."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
        # Fails at engine start on an unknown setting or a wrongly typed value.
        settings.get(vllm_config)
        apply_serving_defaults(vllm_config)
