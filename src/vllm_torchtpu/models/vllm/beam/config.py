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
        _check_features(vllm_config)
        _check_parallelism(vllm_config)
        _check_checkpoint(vllm_config)


def _check_features(vllm_config: VllmConfig) -> None:
    """Rejects engine features and TPU flags the model does not support."""
    dtype = vllm_config.model_config.dtype
    if dtype != torch.bfloat16:
        # BeamRouter's split weight is exact only for bf16 activations.
        raise NotImplementedError(
            f"Beam on TPU runs in bfloat16, got {dtype}; serve with --dtype bfloat16."
        )
    if vllm_config.speculative_config is not None:
        # TODO(parfann): add aux hidden-state capture and SupportsEagle3.
        raise NotImplementedError(
            "Beam on TPU does not support speculative decoding yet; run "
            "without --speculative-config."
        )
    if vllm_config.lora_config is not None:
        raise NotImplementedError(
            "Beam on TPU does not support LoRA; run without --enable-lora."
        )
    # The plugin refuses this too, but only at KV-cache initialization after
    # the weights load, and not at all with --disable-hybrid-kv-cache-manager.
    if envs.VLLM_TPU_BLOCK_MAJOR_KV:
        raise NotImplementedError(
            "Beam on TPU does not support VLLM_TPU_BLOCK_MAJOR_KV=1 with its "
            "mix of sliding-window and full-attention KV caches; unset "
            "VLLM_TPU_BLOCK_MAJOR_KV."
        )


def _check_parallelism(vllm_config: VllmConfig) -> None:
    """Rejects parallel layouts the model does not support."""
    parallel_config = vllm_config.parallel_config
    hf_config = vllm_config.model_config.hf_config
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
    # Checks the flag, not use_sequence_parallel_moe: on a first config build
    # this hook runs before the platform patches that property, so it may
    # still be upstream's (EP, TP > 1, DP > 1). The flag is TPU's only switch
    # for it, so it is rejected whatever the layout.
    if envs.TPU_MOE_SEQUENCE_PARALLEL:
        raise NotImplementedError(
            "Beam on TPU has not been validated with sequence-parallel MoE; "
            "unset TPU_MOE_SEQUENCE_PARALLEL."
        )
    if parallel_config.enable_eplb:
        # Expert parallelism is validated; expert rebalancing is not.
        raise NotImplementedError(
            "Beam on TPU has not been validated with EPLB; run without --enable-eplb."
        )
    # The merged QKV + output-gate projection shards whole KV heads (upstream's
    # separate projections for TP above the KV head count are not ported).
    # The query heads are a multiple of the KV heads, so they divide too.
    tp_size = parallel_config.tensor_parallel_size
    num_kv_heads = hf_config.num_key_value_heads
    if num_kv_heads % tp_size != 0:
        valid_tp_sizes = [
            n for n in range(1, num_kv_heads + 1) if num_kv_heads % n == 0
        ]
        raise NotImplementedError(
            "Beam on TPU requires tensor_parallel_size to divide "
            f"num_key_value_heads ({num_kv_heads}): use one of {valid_tp_sizes}, "
            f"got {tp_size}. Scale further with --data-parallel-size and "
            "--enable-expert-parallel."
        )


def _check_checkpoint(vllm_config: VllmConfig) -> None:
    """Rejects checkpoint configs the model does not implement."""
    hf_config = vllm_config.model_config.hf_config
    if hf_config.hidden_act != "silu":
        raise NotImplementedError(
            "Beam on TPU supports only hidden_act='silu', got "
            f"{hf_config.hidden_act!r}; use a checkpoint with SiLU MLPs."
        )
    if hf_config.tie_word_embeddings:
        raise NotImplementedError(
            "Beam on TPU requires tie_word_embeddings=False; use a checkpoint "
            "with a separate lm_head."
        )
