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
"""Shared builders for the sparse (DSA) indexer adapter tests."""

import contextlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import vllm.model_executor.custom_op as custom_op_mod
import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.models.deepseek_v2 as deepseek_v2
import vllm.model_executor.parameter as parameter_mod
from vllm.config import CacheConfig, DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache, Indexer

from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op


class StubIndexerCache(DeepseekV32IndexerCache):
    """Stands in for `DeepseekV32IndexerCache` (needs a real KV cache config).

    Subclasses the real cache so `VllmTPUIndexer.rebind` can retype it --
    `__class__` assignment requires a compatible instance layout -- but skips
    its `__init__`, which registers into `static_forward_context`.
    """

    def __init__(self, head_dim, dtype, prefix, cache_config):
        torch.nn.Module.__init__(self)
        self.head_dim = head_dim
        self.dtype = dtype
        self.prefix = prefix
        self.cache_config = cache_config


@contextlib.contextmanager
def default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def native_custom_ops():
    """CustomOps built under this patch resolve to their native forward.

    `CustomOp.__init__` resolves its forward at construction time and needs a
    current vLLM config for that; a bare CompilationConfig is enough and binds
    `forward_native` (the TPU dispatch target is `forward_native` too).
    """
    compilation_config = CompilationConfig(
        mode=CompilationMode.NONE, custom_ops=["none"]
    )
    return patch.object(
        custom_op_mod, "get_cached_compilation_config", lambda: compilation_config
    )


def build_indexer(
    *,
    sparse_attn_indexer: Any,
    dtype: torch.dtype,
    n_head: int,
    head_dim: int,
    rope_dim: int,
    hidden_size: int,
    q_lora_rank: int,
    topk: int,
    max_model_len: int,
    buffer_tokens: int,
    # VllmConfig resolves this to 5 for a max_model_len up to 256K.
    prefill_workspace_factor: int = 5,
    prefix: str = "model.layers.0.self_attn.indexer",
) -> mla_attention_op.VllmTPUIndexer:
    """A `VllmTPUIndexer` with the TPU-only collaborators stubbed out.

    `sparse_attn_indexer` replaces `SparseAttnIndexer`, which builds a Pallas
    op eagerly. The linear layers query the TP group at construction time, so
    TP rank/size are pinned to a single rank instead of standing up a
    distributed group.
    """
    config = SimpleNamespace(
        index_topk=topk,
        index_n_heads=n_head,
        index_head_dim=head_dim,
        qk_rope_head_dim=rope_dim,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        attention_config=SimpleNamespace(
            sparse_mla_prefill_workspace_factor=prefill_workspace_factor
        ),
    )
    topk_indices_buffer = torch.full((buffer_tokens, topk), -1, dtype=torch.int32)

    # `VllmTPUIndexer` has no `__init__`: it is a retype of an in-tree `Indexer`
    # that `DeepseekV2MLAAttention` has already built. So build the in-tree
    # layer -- stubbing the collaborators where *it* looks them up -- and then
    # rebind, exactly as `VllmTPUMultiHeadLatentAttentionWrapper` does.
    with (
        patch.object(deepseek_v2, "DeepseekV32IndexerCache", StubIndexerCache),
        patch.object(deepseek_v2, "SparseAttnIndexer", sparse_attn_indexer),
        patch.object(linear_mod, "get_tensor_model_parallel_rank", lambda: 0),
        patch.object(linear_mod, "get_tensor_model_parallel_world_size", lambda: 1),
        patch.object(parameter_mod, "get_tensor_model_parallel_rank", lambda: 0),
        patch.object(parameter_mod, "get_tensor_model_parallel_world_size", lambda: 1),
        default_dtype(dtype),
        # The in-tree layers read the current config while being built.
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))),
    ):
        indexer = Indexer(
            vllm_config=vllm_config,
            config=config,
            hidden_size=hidden_size,
            q_lora_rank=q_lora_rank,
            quant_config=None,
            cache_config=CacheConfig(),
            topk_indices_buffer=topk_indices_buffer,
            prefix=prefix,
        )
        return mla_attention_op.VllmTPUIndexer.rebind(indexer)


def make_rope(
    *,
    rope_dim: int,
    max_position_embeddings: int,
    dtype: torch.dtype,
    is_neox_style: bool = True,
    scaling_factor: float | None = None,
) -> RotaryEmbedding:
    """A real `RotaryEmbedding`, as `get_rope` would build for the indexer;
    with `scaling_factor`, the TPU DeepSeek scaling RoPE."""
    with native_custom_ops():
        if scaling_factor is not None:
            return mla_attention_op.VllmDeepseekScalingRotaryEmbedding(
                head_size=rope_dim,
                rotary_dim=rope_dim,
                max_position_embeddings=max_position_embeddings,
                base=10000.0,
                is_neox_style=is_neox_style,
                scaling_factor=scaling_factor,
                dtype=dtype,
            )
        return RotaryEmbedding(
            head_size=rope_dim,
            rotary_dim=rope_dim,
            max_position_embeddings=max_position_embeddings,
            base=10000.0,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
