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
"""Config checks for the TPU GLM-5.3-Flash model.

Kept free of the model implementation so registering it in
`MODELS_CONFIG_MAP` costs one small import.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from vllm.model_executor.models.config import Glm5NextModelConfig

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

MLA_IMPL_RPA = "rpa"
MLA_IMPL_ABSORBED = "absorbed"

# Width of the all-zero rope slot the absorbed MLA path appends to the latent.
# The shared MLA kernel needs a non-empty rope part; zeros contribute nothing
# to the scores, so attention stays NoPE.
GLM53_MLA_ROPE_PAD = 64


def glm53_mla_impl() -> str:
    """The MLA implementation, "rpa" (default) or "absorbed".

    "rpa" expands the latent with `kv_b_proj` and runs ragged paged attention
    on per-head keys and values; "absorbed" caches the latent (plus the zero
    rope slot) and runs the absorbed MLA kernel.
    """
    return envs.VLLM_TPU_GLM53_MLA_IMPL


class TPUGlm5NextConfig(Glm5NextModelConfig):
    """Upstream GLM-5.3-Flash checks, plus what the TPU model needs."""

    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None:
        """Describes the TPU attention layout to vLLM.

        vLLM treats `glm5_next` as DeepSeek MLA: one latent KV head of
        `kv_lora_rank + qk_rope_head_dim` channels. The TPU "rpa" path
        instead caches per-head NoPE keys and values (`num_key_value_heads`
        heads of `qk_nope_head_dim` channels), and the "absorbed" path caches
        the latent plus a zero rope slot of `GLM53_MLA_ROPE_PAD` channels.
        The KV cache spec, attention backend choice and hybrid page sizing all
        read this, so it must match the attention layers the model builds.
        Only this `ModelConfig`'s own architecture summary is replaced.
        """
        Glm5NextModelConfig.verify_and_update_model_config(model_config)
        text_config = model_config.hf_text_config
        arch = model_config.model_arch_config
        if glm53_mla_impl() == MLA_IMPL_ABSORBED:
            model_config.model_arch_config = dataclasses.replace(
                arch,
                is_deepseek_mla=True,
                head_size=int(text_config.kv_lora_rank) + GLM53_MLA_ROPE_PAD,
            )
            return
        model_config.model_arch_config = dataclasses.replace(
            arch,
            is_deepseek_mla=False,
            head_size=int(text_config.qk_nope_head_dim)
            + int(text_config.qk_rope_head_dim or 0),
            total_num_kv_heads=int(text_config.num_key_value_heads),
        )

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Glm5NextModelConfig.verify_and_update_config(vllm_config)
        model_config = vllm_config.model_config
        text_config = model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        if int(text_config.qk_rope_head_dim or 0) != 0:
            raise NotImplementedError(
                "GLM-5.3-Flash on TPU supports NoPE MLA only "
                f"(qk_rope_head_dim=0), got {text_config.qk_rope_head_dim}."
            )
        if not getattr(text_config, "mhc", True):
            raise NotImplementedError(
                "GLM-5.3-Flash on TPU requires the hyper-connection (mhc=True) variant."
            )
        for name in ("num_attention_heads", "linear_num_heads"):
            heads = int(getattr(text_config, name))
            if heads % tp_size:
                raise ValueError(
                    f"GLM-5.3-Flash {name}={heads} is not divisible by "
                    f"tensor_parallel_size={tp_size}."
                )
        if getattr(vllm_config.parallel_config, "enable_eplb", False):
            raise NotImplementedError(
                "GLM-5.3-Flash on TPU does not yet support dynamic EPLB "
                "(--enable-eplb)."
            )
        kv_transfer_cfg = getattr(vllm_config, "kv_transfer_config", None)
        if kv_transfer_cfg is not None:
            from vllm_torchtpu import _OFFLOADING_CONNECTORS, gdn_pool_layout
            from vllm_torchtpu.distributed.kv_transfer.connector_names import (
                kv_connector_names,
            )

            arch = getattr(model_config, "architecture", "")
            names = kv_connector_names(kv_transfer_cfg) or (
                (kv_transfer_cfg.kv_connector,)
                if getattr(kv_transfer_cfg, "kv_connector", None)
                else ()
            )
            non_offload = [n for n in names if n not in _OFFLOADING_CONNECTORS]
            if (
                non_offload
                and gdn_pool_layout.unified_kv_layout_enabled_for_architecture(arch)
            ):
                raise NotImplementedError(
                    "GLM-5.3-Flash unified KV+KDA pool does not yet support "
                    f"disaggregated KV transfer connectors ({non_offload})."
                )
        index_topk = getattr(text_config, "index_topk", None)
        if index_topk is not None:
            # The sparse attention indexer is not run on TPU: the MLA layers
            # attend densely and causally, which selects the same keys as the
            # indexer while a sequence fits in its top-k budget (plus the
            # always-kept partial pool at the tail).
            dense_exact_len = int(index_topk)
            if getattr(text_config, "index_kpool_always_select_tail", True):
                dense_exact_len += int(getattr(text_config, "index_kpool", 1)) - 1
            if model_config.max_model_len > dense_exact_len:
                logger.warning_once(
                    "GLM-5.3-Flash runs its MLA layers as dense attention on "
                    "TPU. This matches the checkpoint's sparse attention only "
                    "up to %d tokens; longer sequences (max_model_len=%d) "
                    "attend to every earlier token instead of the indexer's "
                    "top-k selection.",
                    dense_exact_len,
                    model_config.max_model_len,
                )
