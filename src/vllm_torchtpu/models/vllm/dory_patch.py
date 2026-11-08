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
"""Installs ``TPUDoryAttention`` for Dory models on TPU."""

from typing import TYPE_CHECKING, Optional

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


def maybe_patch_dory(model_config: Optional["ModelConfig"] = None) -> bool | None:
    """Replace upstream ``DoryAttention`` for ``DoryForCausalLM`` models."""
    if model_config is None or "DoryForCausalLM" not in (
        getattr(model_config, "architectures", None) or []
    ):
        return False

    import vllm.model_executor.models.dory as dory_mod

    from vllm_torchtpu.models.vllm.dory import TPUDoryAttention

    dory_mod.DoryAttention = TPUDoryAttention
    logger.info("Applied TPU patch: Dory attention with an fp32 query.")
    return None
