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
"""Kimi Delta Attention (KDA) on the fused conv1d + GDN v3 Pallas kernel."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from vllm_torchtpu.gdn_pool_layout import unified_kv_layout_enabled_for_architecture
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
    build_kimi_dispatched_kda_op,
    build_kimi_pooled_kda_op,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.sequence_layout import (
    is_pcp_streaming_attention_metadata,
)
from vllm_torchtpu.utils import fresh_buffer


def kda_state_dtype(vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
    """Storage dtypes for one KDA layer's conv and recurrent state.

    The conv cache is fp32 regardless of ``mamba_cache_dtype``: v3's conv1d
    needs fp32 compact layout, so anything narrower makes ``fused_conv1d_gdn``
    widen the whole slot pool once per layer per step. Past about a thousand
    slots that costs more than the fused kernel saves.
    ``VllmGatedDeltaNetAttention.get_state_dtype`` declares fp32 for the same
    reason.
    """
    _, recurrent_dtype = MambaStateDtypeCalculator.kda_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
    )
    return torch.float32, recurrent_dtype


def kda_state_shape(
    *,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    conv_kernel_size: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Per-rank conv and recurrent state shapes of one KDA layer.

    The conv state is declared in the fused kernel's (taps, 1, q | k | v
    channels) slot layout so XLA does not relayout the whole pool around
    every call.
    """
    return (
        (conv_kernel_size - 1, 1, num_heads * (2 * head_k_dim + head_v_dim)),
        (num_heads, head_k_dim, head_v_dim),
    )


def load_a_log(parameter: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Load either the old ``[1, 1, H, 1]`` or current ``[H]`` layout."""

    if loaded_weight.ndim == 4:
        loaded_weight = loaded_weight.flatten()
    rank = get_tensor_model_parallel_rank()
    shard_size = parameter.shape[0]
    loaded_weight = loaded_weight.narrow(0, rank * shard_size, shard_size)
    default_weight_loader(parameter, loaded_weight)


class CausalDepthwiseConv1d(nn.Module):
    """Checkpoint-compatible depthwise-convolution weight container."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if channels % tp_size:
            raise ValueError("Convolution channels must be divisible by TP size")
        self.channels = channels // tp_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(self.channels, 1, kernel_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(
        self,
        parameter: nn.Parameter,
        loaded_weight: torch.Tensor,
    ) -> None:
        if loaded_weight.ndim == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        rank = get_tensor_model_parallel_rank()
        loaded_weight = loaded_weight.chunk(
            get_tensor_model_parallel_world_size(), dim=0
        )[rank]
        parameter.data.copy_(loaded_weight)


class KdaAttention(nn.Module, MambaBase):
    """KDA registered with vLLM and executed by TPU Pallas custom ops.

    Checkpoint q/k/v/b projections load into ``fused_qkvb_proj`` (shards 0-3)
    and the low-rank decay and output-gate down-projections into
    ``fused_fa_ga_proj`` (shards 0-1). With ``use_full_rank_gate`` the decay
    down-projection is ``f_a_proj`` and the output gate is one full-rank
    ``g_proj`` instead.

    Args:
        num_heads: KDA heads across all TP ranks.
        head_k_dim: Query and key head size; also the decay gate's width.
        head_v_dim: Value head size; also the output gate's width.
        gate_rank: Width of the low-rank decay and output-gate projections.
        norm_eps: Epsilon of the gated output RMSNorm.
        gate_bias: Whether the output-gate projection has a bias.
        use_full_rank_gate: Project the output gate straight from the hidden
            state (``g_proj``) instead of through ``gate_rank``.
        gate_lower_bound: Floor of the per-token log-decay, or ``None`` for
            the unbounded gate.
        beta_scale: Beta is ``beta_scale * sigmoid(b)``.

    The unified KV pool supports only ``beta_scale=1`` and equal head sizes;
    other configurations are rejected at construction when it is enabled.
    """

    supports_dcp = False

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        *,
        hidden_size: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        gate_rank: int,
        norm_eps: float,
        gate_bias: bool = False,
        use_full_rank_gate: bool = False,
        gate_lower_bound: float | None = None,
        beta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        supports_pooled = beta_scale == 1.0 and head_v_dim == head_k_dim
        if not supports_pooled and unified_kv_layout_enabled_for_architecture(
            vllm_config.model_config.architecture
        ):
            raise NotImplementedError(
                "The unified KV pool supports KDA only with beta_scale=1 and "
                "equal key and value head sizes; run with "
                "TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0"
            )
        self.prefix = prefix
        self.vllm_config = vllm_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.total_heads = num_heads
        if self.total_heads % self.tp_size:
            raise ValueError("KDA num_heads must be divisible by TP size")
        self.num_heads = self.total_heads // self.tp_size
        key_size = self.total_heads * head_k_dim
        value_size = self.total_heads * head_v_dim
        self.local_key_size = self.num_heads * head_k_dim
        self.local_value_size = self.num_heads * head_v_dim
        self.conv_size = conv_kernel_size
        self.num_spec_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None
            else 0
        )
        self.gate_rank = gate_rank
        self.gate_lower_bound = gate_lower_bound
        self.use_full_rank_gate = use_full_rank_gate
        quant_config = vllm_config.quant_config

        self.fused_qkvb_proj = MergedColumnParallelLinear(
            hidden_size,
            [key_size, key_size, value_size, self.total_heads],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkvb_proj",
        )
        if use_full_rank_gate:
            self.fused_fa_ga_proj = None
            self.f_a_proj = ReplicatedLinear(
                hidden_size,
                gate_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.f_a_proj",
            )
            self.g_proj = ColumnParallelLinear(
                hidden_size,
                value_size,
                bias=gate_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        else:
            self.fused_fa_ga_proj = MergedColumnParallelLinear(
                hidden_size,
                [gate_rank, gate_rank],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_fa_ga_proj",
                disable_tp=True,
            )
            self.f_a_proj = None
            self.g_proj = None
        self.q_conv1d = CausalDepthwiseConv1d(key_size, conv_kernel_size)
        self.k_conv1d = CausalDepthwiseConv1d(key_size, conv_kernel_size)
        self.v_conv1d = CausalDepthwiseConv1d(value_size, conv_kernel_size)
        # Both convolution consumers take one fused weight whose channels are
        # q | k | v; `process_weights_after_loading` folds the checkpoint's
        # three tensors into it at load time. Non-persistent: it is derived
        # from q/k/v_conv1d.weight, not a checkpoint entry of its own, and
        # must not appear in a state dict.
        self.register_buffer(
            "conv_weight_fused",
            torch.empty(self._conv_weight_shape()),
            persistent=False,
        )
        self.f_b_proj = ColumnParallelLinear(
            gate_rank,
            key_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        if use_full_rank_gate:
            self.g_b_proj = None
        else:
            self.g_b_proj = ColumnParallelLinear(
                gate_rank,
                value_size,
                bias=gate_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.g_b_proj",
            )

        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_key_size, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": load_a_log})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        self.o_norm = RMSNorm(head_v_dim, norm_eps)
        self.o_proj = RowParallelLinear(
            value_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            reduce_results=False,
        )
        self.kv_cache = None
        # The dispatched op owns the short convolution as well: the fused
        # kernel does the convolution itself, so it cannot be fed by a separate
        # convolution op.
        self.dispatched_kda_op = build_kimi_dispatched_kda_op(
            prefix,
            lower_bound=gate_lower_bound,
            eps=norm_eps,
            state_dim_first=is_conv_state_dim_first(),
            num_spec_tokens=self.num_spec_tokens,
            beta_scale=beta_scale,
        )
        # Unified-pool variant: reads and writes both state regions through
        # the single attention-shaped buffer; used when the runner binds the
        # pool instead of the per-layer conv/recurrent caches. The manager
        # block size lets the op derive the manager->pool-block split for
        # its state-index remap. It is NOT final yet: the platform's
        # block-size derivation for the pool runs after model construction
        # (hybrid layers are built before the adjusted value exists), so the
        # op reads cache_config.block_size per call, not here.
        self.pooled_kda_op = (
            build_kimi_pooled_kda_op(
                prefix,
                lower_bound=gate_lower_bound,
                eps=norm_eps,
                vllm_config=vllm_config,
                num_spec_tokens=self.num_spec_tokens,
            )
            if supports_pooled
            else None
        )

        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.LINEAR

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return kda_state_dtype(self.vllm_config)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return kda_state_shape(
            num_heads=self.num_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_kernel_size=self.conv_size,
        )

    def _conv_weight_shape(self) -> tuple[int, ...]:
        # Equal head sizes keep the [kernel_size, 3, heads, head_dim] layout
        # whose head axis the AOT exporter shards (export/weight_mapper.py).
        if self.head_k_dim == self.head_v_dim:
            return (self.conv_size, 3, self.num_heads, self.head_k_dim)
        return (self.conv_size, self.local_key_size * 2 + self.local_value_size)

    def _metadata(self) -> AttentionMetadata | None:
        if not is_forward_context_available():
            return None
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        layer_metadata = metadata[self.prefix]
        if not isinstance(layer_metadata, AttentionMetadata):
            raise TypeError("KDA received incompatible attention metadata")
        return layer_metadata

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Fold the three depthwise conv weights into the kernels' layout.

        The checkpoint keeps q/k/v as separate ``[channels, 1, kernel_size]``
        tensors, and the ops used to concatenate and transpose them on every
        step. Both convolution consumers want ``[kernel_size, q | k | v]``,
        so build that once, here, at load time.
        """
        fused = torch.cat(
            tuple(
                w[:, 0, :].t()
                for w in (
                    self.q_conv1d.weight,
                    self.k_conv1d.weight,
                    self.v_conv1d.weight,
                )
            ),
            dim=1,
        ).reshape(self._conv_weight_shape())
        # `fused` is the lazy result of a cat of views, and torch_tpu
        # re-materializes that lineage as a per-step `tt_jit_as_strided`
        # program at the dispatched op's boundary (same failure mode and fix as
        # the w_scale case in quantization/fp8.py). `fresh_buffer` breaks the
        # lazy chain so PJRT ships a plain row-major buffer.
        self.conv_weight_fused = fresh_buffer(fused)

    def _project_inputs(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the op's ``(mixed_qkv, raw_gate, beta, output_gate)``."""
        projected, _ = self.fused_qkvb_proj(hidden_states)
        query, key, value, beta = projected.split(
            [self.local_key_size, self.local_key_size, self.local_value_size]
            + [self.num_heads],
            dim=-1,
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)
        if self.use_full_rank_gate:
            assert self.f_a_proj is not None
            f_a, _ = self.f_a_proj(hidden_states)
            assert self.g_proj is not None
            output_gate, _ = self.g_proj(hidden_states)
            raw_gate, _ = self.f_b_proj(f_a)
        else:
            assert self.fused_fa_ga_proj is not None
            gate_inputs, _ = self.fused_fa_ga_proj(hidden_states)
            f_a, gate_input = gate_inputs.split(
                [self.gate_rank, self.gate_rank], dim=-1
            )
            raw_gate, _ = self.f_b_proj(f_a)
            assert self.g_b_proj is not None
            output_gate, _ = self.g_b_proj(gate_input)
        return mixed_qkv, raw_gate, beta, output_gate

    def _core_attention(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> torch.Tensor:
        """Run the KDA op on this layer's state; ``[tokens, heads * head_v]``."""
        metadata = self._metadata()
        if metadata is None or self.kv_cache is None or self.kv_cache[0].numel() == 0:
            return torch.zeros_like(output_gate)
        if len(self.kv_cache) == 1:
            # Unified pool: the single attention-shaped buffer carries both
            # KDA state regions.
            return self._core_attention_pooled(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                self.kv_cache[0],
                metadata,
            ).flatten(1)
        sconv_cache, recurrent_cache = self.kv_cache
        window_distribution = metadata.mamba_request_distribution
        if window_distribution is None:
            window_distribution = metadata.request_distribution
        slot_read_offsets = metadata.mamba_slot_read_offsets
        if slot_read_offsets is None:
            # Non-speculative KDA keeps the legacy ABI without allocating
            # a pool-sized offset tensor.  The op's static
            # num_spec_tokens=0 path never indexes this dummy.
            slot_read_offsets = metadata.mamba_state_indices
            if slot_read_offsets is None:
                slot_read_offsets = metadata.seq_lens
        return self.dispatched_kda_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            sconv_cache,
            recurrent_cache,
            self.conv_weight_fused,
            self.A_log,
            self.dt_bias,
            self.o_norm.weight,
            cast(torch.Tensor, metadata.query_start_loc),
            cast(torch.Tensor, metadata.mamba_state_indices),
            cast(torch.Tensor, metadata.seq_lens),
            cast(torch.Tensor, metadata.request_distribution),
            cast(torch.Tensor, window_distribution),
            cast(torch.Tensor, slot_read_offsets),
        ).flatten(1)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del positions
        output = self._core_attention(*self._project_inputs(hidden_states))
        output, _ = self.o_proj(output)
        return get_tp_group().all_reduce(output)

    def _core_attention_pooled(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        pool: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """KDA over the unified pool: conv and recurrent state live in the
        pool's per-block byte regions and are gathered/scattered by the op."""
        if is_pcp_streaming_attention_metadata(metadata):
            raise NotImplementedError(
                "KDA does not support PCP streaming prefill with the unified KV pool"
            )
        # `__init__` rejects the unified pool when it cannot build this op.
        assert self.pooled_kda_op is not None
        state_indices = metadata.mamba_state_indices
        if state_indices is None:
            raise RuntimeError(
                "Pooled KDA requires mamba_state_indices in the attention metadata"
            )
        # AttentionMetadata also describes the JAX-side representation. The
        # builder supplies Torch tensors here, before the Torch-to-JAX bridge.
        window_distribution = cast(
            torch.Tensor | None, metadata.mamba_request_distribution
        )
        slot_read_offsets = cast(torch.Tensor | None, metadata.mamba_slot_read_offsets)
        ckpt_indices = cast(torch.Tensor | None, metadata.mamba_ckpt_indices)
        return self.pooled_kda_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            pool,
            self.conv_weight_fused,
            self.A_log,
            self.dt_bias,
            self.o_norm.weight,
            cast(torch.Tensor, metadata.query_start_loc),
            cast(torch.Tensor, state_indices),
            cast(torch.Tensor, metadata.seq_lens),
            cast(torch.Tensor, metadata.request_distribution),
            window_distribution,
            slot_read_offsets,
            ckpt_indices,
        )
