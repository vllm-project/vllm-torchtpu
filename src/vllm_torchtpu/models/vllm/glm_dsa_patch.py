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
"""TPU v7x patch callback for GLM-5 DSA (``GlmMoeDsaForCausalLM``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_torchtpu import tpu_info
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import (
    is_glm_dsa_model,
    set_active_model_config,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def maybe_patch_glm_dsa(model_config: ModelConfig | None = None) -> bool | None:
    """Enable TPU v7x fast paths and MoE EPxTP split strictly for GLM-5 DSA."""
    if model_config is None or not is_glm_dsa_model(model_config):
        return False
    if not tpu_info.is_tpu_v7x():
        return False

    set_active_model_config(model_config)
    tpu_info.set_v7x_model_fastpaths(True)

    from vllm.config import get_current_vllm_config_or_none

    from vllm_torchtpu.layers.adapter import (
        dcp_token_parallel_op,
        moe_hierarchical,
        moe_routing,
        moe_tp_decode,
    )

    moe_routing.set_rowmax_bias_enabled(True)
    moe_hierarchical.install_ep_tp_split_patch()
    cfg = get_current_vllm_config_or_none()
    if cfg is not None:
        dcp_token_parallel_op.configure_for(cfg)
        moe_tp_decode.set_speculative_decoding(cfg.speculative_config is not None)
    return None
