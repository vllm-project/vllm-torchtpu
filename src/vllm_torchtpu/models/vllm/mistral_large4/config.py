# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Supported envelope for the initial ML4 TPU definition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.models.config import VerifyAndUpdateConfig

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class TPUMistralLarge4Config(VerifyAndUpdateConfig):
    """Reject missing adapters before allocating model weights or caches."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        from vllm_torchtpu.models.vllm.mistral_large4.checkpoint_config import (
            get_layer_specs,
        )

        model_config = vllm_config.model_config
        config = model_config.hf_text_config
        parallel = vllm_config.parallel_config
        # with_hf_config preserves a separate copy of hf_text_config for
        # Pixtral's text child; object identity does not imply a wrapper.
        if model_config.hf_config.get_text_config() is not model_config.hf_config:
            # TODO: integrate the Pixtral multimodal wrapper.
            raise NotImplementedError(
                "The initial ML4 TPU definition supports text-only configs. "
                "The Pixtral multimodal wrapper still needs integration."
            )
        if vllm_config.speculative_config is not None:
            # TODO: add ML4 DSpark context-cache writes and draft attention.
            raise NotImplementedError(
                "ML4 speculative decoding is not implemented on TPU yet."
            )
        if parallel.pipeline_parallel_size != 1:
            raise NotImplementedError("ML4 on TPU requires pipeline_parallel_size=1.")
        if (
            parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            # TODO: integrate the context-parallel attention wrappers.
            raise NotImplementedError("ML4 context parallelism is not implemented yet.")
        if parallel.enable_eplb:
            # TODO: expose the expert load-balancing model interface.
            raise NotImplementedError(
                "ML4 expert load balancing is not implemented yet."
            )
        if config.q_lora_rank is not None:
            raise NotImplementedError("ML4 on TPU requires q_lora_rank=None.")
        if config.num_attention_heads % parallel.tensor_parallel_size:
            raise ValueError(
                "ML4 num_attention_heads must be divisible by tensor_parallel_size."
            )

        specs = get_layer_specs(config)
        for spec in specs:
            if spec.shares_kv and (
                not 0 <= spec.kv_source_layer < spec.layer_idx
                or specs[spec.kv_source_layer].shares_kv
            ):
                raise ValueError(
                    "ML4 KV sharing must reference an earlier layer that owns "
                    "its cache."
                )
        if model_config.quantization not in (
            None,
            "fp8",
            "compressed-tensors",
        ):
            raise NotImplementedError(
                "ML4 TPU weight quantization supports FP8 "
                "and supported compressed-tensors schemes."
            )
        # Dense auto stays in the model dtype. Sparse execution requires the
        # existing scalar-scale FP8 cache explicitly, rather than fp8_ds_mla.
        cache = vllm_config.cache_config
        if any(spec.use_sparse for spec in specs):
            from vllm_torchtpu.kernels.mla import kv_cache_utils
            from vllm_torchtpu.models.vllm.mistral_large4.indexer import MistralIndexer

            if config.index_topk <= 0 or config.index_n_heads <= 0:
                raise ValueError("ML4 index_topk and index_n_heads must be positive.")
            if not specs[0].has_indexer:
                raise ValueError("ML4's first sparse layer must run its own indexer.")
            if config.index_head_dim != MistralIndexer.quant_block_size:
                raise NotImplementedError(
                    "ML4 TPU indexing requires index_head_dim="
                    f"{MistralIndexer.quant_block_size}."
                )
            latent_dim = kv_cache_utils.WORD_BYTES * kv_cache_utils.TILE_LANE_BYTES
            rope_dim = kv_cache_utils.TILE_LANE_BYTES // 2
            if (config.kv_lora_rank, config.qk_rope_head_dim) != (latent_dim, rope_dim):
                raise NotImplementedError(
                    "The TPU sparse MLA gather requires "
                    f"kv_lora_rank={latent_dim} and qk_rope_head_dim={rope_dim}."
                )
            if model_config.dtype != torch.bfloat16:
                raise NotImplementedError("ML4 sparse MLA requires --dtype bfloat16.")
            if cache.cache_dtype not in ("fp8", "fp8_e4m3"):
                # This is the existing scalar-scale FP8 cache, not the CUDA
                # fp8_ds_mla block-scaled format, which is not integrated yet.
                raise NotImplementedError(
                    "ML4 TPU sparse MLA requires an FP8 latent KV cache. "
                    "Pass --kv-cache-dtype fp8_e4m3."
                )
        elif cache.cache_dtype == "bfloat16":
            cache.cache_dtype = "auto"
        elif cache.cache_dtype != "auto":
            # TODO: validate ML4's FP8 latent cache layout and scale handling.
            raise NotImplementedError(
                "ML4 FP8 KV caches are not integrated yet. "
                "Pass --kv-cache-dtype bfloat16 for dense bring-up."
            )
