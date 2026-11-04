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
"""MiniMax M3 config checks: what the TPU implementation supports."""

from typing import TYPE_CHECKING

from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_torchtpu import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_SUPPORTED_KV_CACHE_DTYPES = ("auto", "bfloat16")


class TPUMiniMaxM3Config(VerifyAndUpdateConfig):
    """Pin the KV page size to the sparse block and reject unported features."""

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        text_config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        sparse_cfg = text_config.sparse_attention_config
        if not sparse_cfg or not all(sparse_cfg["sparse_attention_freq"]):
            raise NotImplementedError(
                "MiniMax M3 on TPU implements sparse attention in every layer; "
                "this checkpoint has dense attention layers."
            )
        # MSA selects whole KV pages, so a page must be one sparse block.
        block_size = sparse_cfg["sparse_block_size"]
        if cache_config.user_specified_block_size and (
            cache_config.block_size != block_size
        ):
            raise ValueError(
                f"MiniMax M3 needs --block-size {block_size} (the sparse "
                f"attention block), got {cache_config.block_size}."
            )
        cache_config.block_size = block_size
        cache_config.user_specified_block_size = True

        tp_size = parallel_config.tensor_parallel_size
        num_kv_heads = text_config.num_key_value_heads
        if tp_size < num_kv_heads:
            # TODO(minimax-m3): select blocks per KV head within a rank.
            raise NotImplementedError(
                f"MiniMax M3 sparse attention on TPU needs one KV head per rank: "
                f"use --tensor-parallel-size >= {num_kv_heads}, got {tp_size}."
            )
        if cache_config.cache_dtype not in _SUPPORTED_KV_CACHE_DTYPES:
            # TODO(minimax-m3): quantize K/V on the MSA cache write.
            raise NotImplementedError(
                f"MiniMax M3 on TPU supports --kv-cache-dtype in "
                f"{_SUPPORTED_KV_CACHE_DTYPES}, got {cache_config.cache_dtype!r}."
            )
        indexer_kv_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("bf16")
        if indexer_kv_dtype != "bf16":
            # TODO(minimax-m3): fp8 index keys.
            raise NotImplementedError(
                "MiniMax M3 on TPU keeps index keys in bf16; drop "
                f"indexer_kv_dtype={indexer_kv_dtype!r} from --attention-config."
            )
        # The MSA op writes and reads the FLASH_ATTN (RPA v3) page layout.
        backend = vllm_config.attention_config.backend
        if backend not in (None, AttentionBackendEnum.FLASH_ATTN):
            raise NotImplementedError(
                "MiniMax M3 sparse attention on TPU runs on the FLASH_ATTN KV "
                f"layout; leave --attention-backend unset, got {backend.name}."
            )
        if envs.VLLM_TPU_BLOCK_MAJOR_KV:
            raise NotImplementedError(
                "MiniMax M3 on TPU does not support VLLM_TPU_BLOCK_MAJOR_KV yet."
            )
        if vllm_config.speculative_config is not None:
            # TODO(minimax-m3): port the DSpark drafter.
            raise NotImplementedError(
                "MiniMax M3 on TPU does not support speculative decoding yet."
            )
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "MiniMax M3 on TPU requires pipeline_parallel_size=1."
            )
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "MiniMax M3 on TPU does not support context parallelism yet."
            )
