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
        if text_config.ple_layer_ids:
            # TODO(PLE): add the n-gram PLE layer and its per-request state.
            raise NotImplementedError(
                "Qwen4Exp n-gram PLE (ple_layer_ids="
                f"{text_config.ple_layer_ids}) is not supported on TPU yet."
            )
        # The QSA fields are optional; configs without them use dense
        # attention by design.
        indexer_budget = getattr(text_config, "indexer_budget", None)
        if indexer_budget is not None and model_config.max_model_len > indexer_budget:
            # TODO(QSA): lift once the QSA indexer and sparse attention land.
            raise NotImplementedError(
                "Qwen4Exp full attention runs dense on TPU, which matches QSA "
                f"only up to indexer_budget={indexer_budget} tokens. Pass "
                f"--max-model-len {indexer_budget} or less."
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
