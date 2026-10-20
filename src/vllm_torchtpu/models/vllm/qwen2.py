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
"""TPU Qwen2 attention override with GQA query-head zero-padding support."""

from typing import Any

from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen2 import Qwen2Attention
from vllm.v1.attention.backend import AttentionType

from vllm_torchtpu.gqa_head_padding import (
    GqaPaddingSpec,
    compute_padded_gqa_heads,
    install_o_proj_padded_loader,
    install_qkv_padded_loader,
)


class TPUTPPaddedQwen2Attention(Qwen2Attention):
    """Qwen2Attention that zero-pads Q heads per KV group when heads % tp != 0.

    Note on head-count config invariant:
        ``update_gqa_head_padding_config`` updates
        ``model_config.model_arch_config.total_num_attention_heads`` to the
        padded head count (e.g., 32) for vLLM's parallel/KV-cache validation,
        while recording ``model_config.orig_total_num_attention_heads`` (e.g.,
        28) and leaving ``hf_config.num_attention_heads`` unmutated so
        upstream ``Qwen2DecoderLayer`` passes the original unpadded
        ``num_heads`` (28) into ``TPUTPPaddedQwen2Attention``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        padded_heads, orig_q_per_kv, padded_q_per_kv = compute_padded_gqa_heads(
            num_heads, num_kv_heads, tp_size
        )
        needs_padding = padded_heads % tp_size == 0 and padded_heads != num_heads

        # Upstream derives head_dim = hidden_size // num_heads. With padding,
        # pass padded_heads * head_dim so head_dim, q_size, kv_size, scaling,
        # rotary_emb and attn are built for the padded layout; qkv_proj and
        # o_proj are rebuilt below with the true hidden_size. Without padding,
        # arguments are forwarded unchanged.
        super().__init__(
            hidden_size=(
                padded_heads * (hidden_size // num_heads)
                if needs_padding
                else hidden_size
            ),
            num_heads=padded_heads if needs_padding else num_heads,
            num_kv_heads=num_kv_heads,
            rope_parameters=rope_parameters,
            max_position=max_position,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qk_norm=qk_norm,
            rms_norm_eps=rms_norm_eps,
        )
        self.orig_total_num_heads = num_heads
        if not needs_padding:
            return

        self.hidden_size = hidden_size
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        spec = GqaPaddingSpec(
            num_kv_heads=num_kv_heads,
            orig_q_per_kv=orig_q_per_kv,
            padded_q_per_kv=padded_q_per_kv,
            head_dim=self.head_dim,
        )
        install_qkv_padded_loader(self.qkv_proj, spec)
        install_o_proj_padded_loader(self.o_proj, spec)
