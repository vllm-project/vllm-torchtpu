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
"""Config checks for the TPU Beam text model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.config import VerifyAndUpdateConfig

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class TPUBeamConfig(VerifyAndUpdateConfig):
    """Rejects the configurations the TPU Beam model does not support.

    Upstream Beam has no config hook; it rejects EPLB, sequence-parallel MoE
    and tied embeddings in BeamForCausalLM.__init__. The checks run here so
    they fail before the weights load.
    """

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        parallel_config = vllm_config.parallel_config
        if model_config.dtype != torch.bfloat16:
            # BeamRouter's split weight is exact only for bf16 activations.
            raise NotImplementedError(
                f"Beam on TPU runs in bfloat16, got {model_config.dtype}; serve "
                "with --dtype bfloat16."
            )
        if vllm_config.speculative_config is not None:
            # TODO(DFlash2): add aux hidden-state capture and SupportsEagle3.
            raise NotImplementedError(
                "Beam on TPU does not support speculative decoding yet; run "
                "without --speculative-config."
            )
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Beam on TPU does not support pipeline parallelism; run with "
                "--pipeline-parallel-size 1 and scale with --data-parallel-size."
            )
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Beam on TPU does not support context parallelism yet; run with "
                "--prefill-context-parallel-size 1 and "
                "--decode-context-parallel-size 1."
            )
        # The flag, not use_sequence_parallel_moe: that property is not yet
        # TPU-patched here and is true for any DP + EP layout.
        if envs.TPU_MOE_SEQUENCE_PARALLEL:
            raise NotImplementedError(
                "Beam has not been validated with sequence-parallel MoE; unset "
                "TPU_MOE_SEQUENCE_PARALLEL."
            )
        if parallel_config.enable_eplb:
            # Expert parallelism is validated; expert rebalancing is not.
            raise NotImplementedError(
                "Beam has not been validated with EPLB; run without --enable-eplb."
            )
        if vllm_config.lora_config is not None:
            raise NotImplementedError(
                "Beam on TPU does not support LoRA; run without --enable-lora."
            )
        if hf_config.tie_word_embeddings:
            raise NotImplementedError(
                "Beam requires tie_word_embeddings=False; use a checkpoint with "
                "a separate lm_head."
            )
        # The merged QKV + output-gate projection shards whole KV heads, and
        # replicating KV heads at a higher TP saves under 1 GB per device.
        tp_size = parallel_config.tensor_parallel_size
        num_heads = hf_config.num_attention_heads
        num_kv_heads = hf_config.num_key_value_heads
        if num_kv_heads % tp_size or num_heads % tp_size:
            valid = [
                n
                for n in range(1, num_kv_heads + 1)
                if not num_kv_heads % n and not num_heads % n
            ]
            raise NotImplementedError(
                f"Beam on TPU: tensor_parallel_size must divide "
                f"num_key_value_heads ({num_kv_heads}): use one of {valid}, got "
                f"{tp_size}. Scale further with --data-parallel-size and "
                "--enable-expert-parallel."
            )
        # The plugin refuses this too, but only at KV-cache initialization
        # after the weights load, and not at all with
        # --disable-hybrid-kv-cache-manager.
        if envs.VLLM_TPU_BLOCK_MAJOR_KV:
            raise NotImplementedError(
                "VLLM_TPU_BLOCK_MAJOR_KV=1 does not support Beam's mix of "
                "sliding-window and full-attention KV caches; unset "
                "VLLM_TPU_BLOCK_MAJOR_KV."
            )
