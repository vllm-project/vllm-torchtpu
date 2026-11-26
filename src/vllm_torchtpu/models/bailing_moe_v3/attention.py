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
"""Bailing KDA using the shared fused conv1d/GDN TPU implementation."""

from typing import cast

import torch
from transformers import PretrainedConfig
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention

from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
    build_kimi_dispatched_kda_op,
    build_kimi_pooled_kda_op,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.sequence_layout import (
    is_pcp_streaming_attention_metadata,
)
from vllm_torchtpu.utils import fresh_buffer


@BailingMoeV3KimiDeltaAttention.register_oot
class TpuBailingKimiDeltaAttention(BailingMoeV3KimiDeltaAttention):
    """Preserve upstream projections and loading; replace the attention core."""

    tpu_conv_weight_fused: torch.Tensor

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        layer_id: int = 0,
        prefix: str = "attention",
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        num_speculative_tokens: int = 0,
    ) -> None:
        super().__init__(
            config,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            model_config=model_config,
            cache_config=cache_config,
            num_speculative_tokens=num_speculative_tokens,
        )
        self.kv_cache = ()
        self.register_buffer(
            "tpu_conv_weight_fused",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self._tpu_kda_op = build_kimi_dispatched_kda_op(
            self.prefix,
            lower_bound=self.lower_bound if self.safe_gate else None,
            eps=config.rms_norm_eps,
            state_dim_first=is_conv_state_dim_first(),
            num_spec_tokens=self.num_speculative_tokens,
        )
        self._tpu_pooled_kda_op = build_kimi_pooled_kda_op(
            self.prefix,
            lower_bound=self.lower_bound if self.safe_gate else None,
            eps=config.rms_norm_eps,
            # The op reads the final manager block size after cache setup.
            vllm_config=get_current_vllm_config(),
            num_spec_tokens=self.num_speculative_tokens,
        )

    # The TPU kernel uses a rank-three conv slot; upstream annotates its CUDA layout.
    # pyrefly: ignore[bad-override]
    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (
            (self.conv_size - 1, 1, 3 * self.projection_size_per_partition),
            (self.local_num_heads, self.head_dim, self.head_dim),
        )

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        # The fused kernel reads conv taps in fp32 even for bf16 activations.
        return torch.float32, super().get_state_dtype()[1]

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Materialize checkpoint conv columns once in [taps, qkv, heads, dim]."""
        fused = torch.stack(
            [
                cast(torch.Tensor, w)[:, 0]
                .t()
                .reshape(self.conv_size, self.local_num_heads, self.head_dim)
                for w in (
                    self.q_conv1d.weight,
                    self.k_conv1d.weight,
                    self.v_conv1d.weight,
                )
            ],
            dim=1,
        )
        self.tpu_conv_weight_fused = fresh_buffer(fused)

    def _metadata(self) -> AttentionMetadata | None:
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return None
        return cast(dict[str, AttentionMetadata], metadata)[self.prefix]

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, output: torch.Tensor
    ) -> None:
        del positions
        if self.separate_b_proj:
            assert self.qkv_proj is not None and self.b_proj is not None
            mixed_qkv = self.qkv_proj(hidden_states)[0]
            beta = self.b_proj(hidden_states)[0]
        else:
            assert self.qkvb_proj is not None
            mixed_qkv, beta = self.qkvb_proj(hidden_states)[0].split(
                [3 * self.projection_size_per_partition, self.local_num_heads],
                dim=-1,
            )
        raw_gate = self.f_proj(hidden_states)[0]
        output_gate = self.g_proj(hidden_states)[0]
        metadata = self._metadata()
        if metadata is None or not self.kv_cache or self.kv_cache[0].numel() == 0:
            core = hidden_states.new_zeros(
                (hidden_states.shape[0], self.local_num_heads, self.head_dim)
            )
        elif len(self.kv_cache) == 1:
            core = self._core_attention_pooled(
                mixed_qkv, raw_gate, beta, output_gate, self.kv_cache[0], metadata
            )
        else:
            window_distribution = metadata.mamba_request_distribution
            if window_distribution is None:
                window_distribution = metadata.request_distribution
            slot_read_offsets = metadata.mamba_slot_read_offsets
            if slot_read_offsets is None:
                # Unused by the non-speculative kernel; retain its tensor-only ABI.
                slot_read_offsets = metadata.mamba_state_indices
            sconv_cache, recurrent_cache = self.kv_cache
            core = self._tpu_kda_op(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                sconv_cache,
                recurrent_cache,
                self.tpu_conv_weight_fused,
                self.A_log.reshape(-1),
                self.dt_bias,
                self.o_norm.weight,
                cast(torch.Tensor, metadata.query_start_loc),
                cast(torch.Tensor, metadata.mamba_state_indices),
                cast(torch.Tensor, metadata.seq_lens),
                cast(torch.Tensor, metadata.request_distribution),
                cast(torch.Tensor, window_distribution),
                cast(torch.Tensor, slot_read_offsets),
            )
        output[:] = self.o_proj(core.reshape(hidden_states.shape[0], -1))[0]

    def _core_attention_pooled(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        pool: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Read and write conv/recurrent state in the shared attention pool."""
        if is_pcp_streaming_attention_metadata(metadata):
            raise NotImplementedError(
                "Bailing KDA does not support PCP streaming prefill "
                "with the unified KV pool"
            )
        state_indices = metadata.mamba_state_indices
        if state_indices is None:
            raise RuntimeError(
                "Pooled Bailing KDA requires mamba_state_indices in attention metadata"
            )
        return self._tpu_pooled_kda_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            pool,
            self.tpu_conv_weight_fused,
            self.A_log.reshape(-1),
            self.dt_bias,
            self.o_norm.weight,
            cast(torch.Tensor, metadata.query_start_loc),
            cast(torch.Tensor, state_indices),
            cast(torch.Tensor, metadata.seq_lens),
            cast(torch.Tensor, metadata.request_distribution),
            cast(torch.Tensor | None, metadata.mamba_request_distribution),
            cast(torch.Tensor | None, metadata.mamba_slot_read_offsets),
            cast(torch.Tensor | None, metadata.mamba_ckpt_indices),
        )
