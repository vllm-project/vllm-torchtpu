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
"""Installs ``TPUTPPaddedQwen2Attention`` for Qwen2 models on TPU."""

from typing import TYPE_CHECKING, Optional

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)


def maybe_patch_qwen2(model_config: Optional["ModelConfig"] = None) -> bool | None:
    """Replace upstream ``Qwen2Attention`` for ``Qwen2ForCausalLM`` models.

    The override only pads when heads % tp_size != 0 and otherwise forwards
    to upstream unchanged.
    """
    if model_config is None or "Qwen2ForCausalLM" not in (
        getattr(model_config, "architectures", None) or []
    ):
        return False

    import vllm.model_executor.models.qwen2 as qwen2_mod

    from vllm_torchtpu.models.vllm.qwen2 import TPUTPPaddedQwen2Attention

    qwen2_mod.Qwen2Attention = TPUTPPaddedQwen2Attention
    logger.info("Applied TPU patch: Qwen2 GQA query-head padding attention.")
    return None
