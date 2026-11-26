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
"""Config checks for the TPU Qwen4Exp text model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.models.config import Qwen4ExpForCausalLMConfig

from vllm_torchtpu.block_major_pool import unified_block_major_enabled
from vllm_torchtpu.gdn_pool_layout import unified_kv_layout_enabled_for_architecture

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class TPUQwen4ExpConfig(Qwen4ExpForCausalLMConfig):
    """Upstream Qwen4Exp checks, plus the features the TPU model lacks yet."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Qwen4ExpForCausalLMConfig.verify_and_update_config(vllm_config)
        model_config = vllm_config.model_config
        text_config = model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        # Each PLE layer keeps its state in a unified-pool block of its GDN
        # layer's spec (see Qwen4ExpPLELayer; transformers admits PLE only on
        # linear_attention layers).
        has_ple = bool(text_config.ple_layer_ids)
        has_sparse_qsa = (
            text_config.indexer_budget is not None
            and model_config.max_model_len > text_config.indexer_budget
            and "indexed_attention" in (text_config.layer_types or ())
        )
        if has_ple and not unified_kv_layout_enabled_for_architecture(
            model_config.architecture
        ):
            raise NotImplementedError(
                "Qwen4Exp n-gram PLE keeps its state in the unified KV pool; "
                "unset TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0."
            )
        if (has_ple or has_sparse_qsa) and unified_block_major_enabled(vllm_config):
            raise NotImplementedError(
                "Qwen4Exp PLE/QSA does not support block-major unified KV pools; "
                "unset VLLM_TPU_BLOCK_MAJOR_KV."
            )
        # transformers (>=5.18, see pyproject.toml) labels the checkpoint's
        # full_attention (QSA) layers indexed_attention.
        unsupported_layer_types = sorted(
            set(text_config.layer_types or ())
            - {"linear_attention", "indexed_attention"}
        )
        if unsupported_layer_types:
            raise NotImplementedError(
                f"Qwen4Exp layer types {unsupported_layer_types} are not "
                "supported on TPU yet."
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Qwen4Exp on TPU does not support speculative decoding yet."
            )
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Qwen4Exp on TPU requires pipeline_parallel_size=1."
            )
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen4Exp hyper-connections do not support sequence-parallel "
                "MoE; unset TPU_MOE_SEQUENCE_PARALLEL."
            )
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp on TPU does not support context parallelism yet."
            )
