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

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# --kv-cache-dtype -> the indexer dtype naming of --attention-config.
_INDEXER_DTYPE_BY_KV_CACHE_DTYPE = {
    "auto": "bf16",
    "bfloat16": "bf16",
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
    # 4-bit MSA caches (`kernels.msa_fp4`): the index keys are 4-bit too;
    # indexer_kv_dtype must stay "auto" (no --attention-config entry).
    "int4_per_token_head": "int4_per_token_head",
}


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
        kv_dtype = _INDEXER_DTYPE_BY_KV_CACHE_DTYPE.get(cache_config.cache_dtype)
        if kv_dtype is None:
            raise NotImplementedError(
                "MiniMax M3 on TPU supports --kv-cache-dtype in "
                f"{sorted(_INDEXER_DTYPE_BY_KV_CACHE_DTYPE)}, got "
                f"{cache_config.cache_dtype!r}."
            )
        indexer_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype(kv_dtype)
        if indexer_dtype not in (kv_dtype, "fp8_e4m3" if kv_dtype == "fp8" else None):
            # TODO(minimax-m3): an index-key dtype independent of the KV cache.
            raise NotImplementedError(
                f"MiniMax M3 on TPU stores index keys in the KV cache dtype "
                f"({kv_dtype}); "
                + (
                    "leave indexer_kv_dtype unset"
                    if kv_dtype == "int4_per_token_head"
                    else f"set indexer_kv_dtype to {kv_dtype!r}"
                )
                + f" in --attention-config, got {indexer_dtype!r}."
            )
        if parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError(
                "MiniMax M3 on TPU does not support prefill context parallelism."
            )
        if parallel_config.decode_context_parallel_size > 1:
            _verify_dcp(vllm_config, block_size)
        if vllm_config.speculative_config is not None:
            from vllm_torchtpu.models.minimax_m3.dspark import (
                configure_dspark_draft,
            )

            configure_dspark_draft(vllm_config.speculative_config)
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "MiniMax M3 on TPU requires pipeline_parallel_size=1."
            )


def _verify_dcp(vllm_config: "VllmConfig", block_size: int) -> None:
    """DCP splits the context over contiguous TP ranks, one whole sparse
    block per rank in turn (`kernels.msa_dcp`). A group within one KV head's
    replicas shares that head; a wider group (up to the whole TP group)
    spans several KV heads, each rank storing all of them for its blocks."""
    parallel_config = vllm_config.parallel_config
    dcp = parallel_config.decode_context_parallel_size
    tp_size = parallel_config.tensor_parallel_size
    num_kv_heads = vllm_config.model_config.hf_text_config.num_key_value_heads
    replicas = tp_size // num_kv_heads
    if tp_size % dcp or (replicas % dcp and dcp % replicas):
        raise ValueError(
            f"MiniMax M3 DCP needs --decode-context-parallel-size ({dcp}) to "
            f"divide --tensor-parallel-size ({tp_size}) and to divide, or be a "
            f"multiple of, the {replicas} ranks that replicate a KV head."
        )
    interleave = parallel_config.cp_kv_cache_interleave_size
    if interleave == 1:
        parallel_config.cp_kv_cache_interleave_size = block_size
    elif interleave != block_size:
        raise ValueError(
            "MiniMax M3 DCP keeps whole sparse blocks on a rank: use "
            f"--cp-kv-cache-interleave-size {block_size} (or leave it unset), "
            f"got {interleave}."
        )
    cache_config = vllm_config.cache_config
    if (
        getattr(cache_config, "enable_prefix_caching", False)
        and getattr(cache_config, "prefix_match_unit", None) is None
    ):
        # The DCP manager block is block_size * dcp tokens; without a finer
        # unit a prefix hit stops at that boundary (DCP 8: 1024 tokens). Hit
        # at sparse-block granularity, as without DCP (the patched single-
        # group coordinator copy-on-writes a partly shared block).
        cache_config.prefix_match_unit = block_size
    if (
        vllm_config.speculative_config is not None
        and vllm_config.scheduler_config.disable_hybrid_kv_cache_manager is False
    ):
        # The DSpark draft keeps its sliding-window KV whole on every DCP rank
        # inside the target's KV cache group (`kv_cache_manager.
        # dcp_replicated_page_tokens`), which needs the unified group.
        raise NotImplementedError(
            "MiniMax M3 DSpark with --decode-context-parallel-size needs "
            "--disable-hybrid-kv-cache-manager."
        )
