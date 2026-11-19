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
"""TPU Kimi delta attention (KDA) layer for GLM-5.3-Flash.

GLM-5.3-Flash's linear-attention layers are vLLM's
``KimiGatedDeltaNetAttention`` (fused ``in_proj_qkvgfab`` projection, packed
q/k/v short convolution, full-rank gates). This module keeps every projection,
parameter and weight loader of that class and replaces only the compute:
``Glm53TpuKimiDeltaAttention`` runs the same Kimi KDA ops that Kimi-K3 uses
(``kda_attention_op``), around the fused conv1d + GDN v3 kernel. Those ops own
the short convolution, the q/k L2 norm, the gate, beta's sigmoid, the
recurrence, the state write-back and the gated output norm, so the layer hands
them the raw projections and gets back the normed ``[T, H, D]`` output.

The runner binds one of two state layouts:

* the unified block pool (``len(kv_cache) == 1``), the default for this
  architecture (``gdn_pool_layout.POOLED_GDN_ARCHITECTURES``). The pooled op
  gathers the conv and recurrent regions out of the pool, runs the kernel and
  scatters them back. GLM-5.3's attention layers use the ragged paged
  attention pool, which keeps one token per pool row, so the op is built with
  ``pool_tokens_per_row=1``;
* per-layer conv and recurrent caches (``TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=0``),
  on which the dispatched op runs.

Both ops take speculative-decoding verify windows the way Kimi-K3 passes them.

The class has a different name from the vLLM class it extends, so vLLM's
``PluggableLayer.__new__`` instantiates it as itself: the model constructs it
directly and nothing is registered globally. Importing this module has no side
effects.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_torchtpu.gdn_pool_layout import (
    DEFAULT_POOLED_GDN_CONV_STATE_DTYPE,
    DEFAULT_POOLED_GDN_SSM_STATE_DTYPE,
    PooledGDNStateLayout,
    derive_pooled_gdn_state_layout,
    pooled_gdn_conv_state_bytes,
    pooled_gdn_ssm_state_bytes,
)
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
    build_kimi_dispatched_kda_op,
    build_kimi_pooled_kda_op,
)
from vllm_torchtpu.layers.core.sequence_layout import (
    is_pcp_streaming_attention_metadata,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import fresh_buffer

logger = init_logger(__name__)

# GLM-5.3's attention layers use the ragged paged attention pool, which keeps
# one token per pool row. The pooled op's default infers the packing of the
# MLA pool (Kimi-Linear, Kimi-K3), which packs several.
GLM53_KDA_POOL_TOKENS_PER_ROW = 1


def _torch_dtype(name: str) -> torch.dtype:
    """``torch.float32`` for the name ``"torch.float32"``."""
    dtype = getattr(torch, name.removeprefix("torch."))
    assert isinstance(dtype, torch.dtype), name
    return dtype


def glm53_kda_state_shape(
    conv_kernel_size: int, local_num_heads: int, head_dim: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Per-slot ``(conv, recurrent)`` state shapes of one TP rank's KDA layer.

    The conv state is declared in the fused kernel's ``(taps, 1, dim)`` slot
    layout, as Kimi-K3 declares it, so the dispatched op reads the per-layer
    cache in place. The recurrent state is ``(heads, d_k, d_v)``, which is also
    what the pooled op's SSM region holds. The model's
    ``get_mamba_state_shape_from_config`` must return the same value: the
    platform sizes the unified pool from it, and vLLM allocates from the layer.
    """
    return (
        (conv_kernel_size - 1, 1, 3 * local_num_heads * head_dim),
        (local_num_heads, head_dim, head_dim),
    )


def glm53_kda_state_dtype() -> tuple[torch.dtype, torch.dtype]:
    """Per-slot ``(conv, recurrent)`` state dtypes of the KDA layer.

    Both are the unified pool's fixed state dtypes (``gdn_pool_layout``):
    fp32 conv rows, which is the only conv dtype the fused kernel accepts, and
    an fp32 recurrent state, which both Kimi ops require. The per-layer caches
    use the same dtypes so the two layouts hold identical state. The model's
    ``get_mamba_state_dtype_from_config`` must return the same value.
    """
    return (
        _torch_dtype(DEFAULT_POOLED_GDN_CONV_STATE_DTYPE),
        _torch_dtype(DEFAULT_POOLED_GDN_SSM_STATE_DTYPE),
    )


def glm53_kda_pool_state_layout(
    *,
    conv_kernel_size: int,
    local_num_heads: int,
    head_dim: int,
    token_bytes: int,
) -> PooledGDNStateLayout:
    """Where one KDA layer's state lives inside a unified-pool block.

    ``token_bytes`` is the byte size of one pool token row. The SSM region
    starts at token 0 and the conv region at ``layout.ssm_tokens``; both sizes
    come from ``gdn_pool_layout``, so anything else that reads these bytes (for
    example a decode megakernel that shares the pool) agrees with the pooled op
    on offsets and dtypes.
    """
    return derive_pooled_gdn_state_layout(
        ssm_bytes=pooled_gdn_ssm_state_bytes(
            num_v_heads=local_num_heads, head_k_dim=head_dim, head_v_dim=head_dim
        ),
        conv_bytes=pooled_gdn_conv_state_bytes(
            kernel_size=conv_kernel_size, conv_dim=3 * local_num_heads * head_dim
        ),
        token_bytes=token_bytes,
    )


class Glm53GatedRMSNorm(nn.Module):
    """Per-head RMSNorm with a sigmoid or SiLU output gate, in plain torch.

    Stands in for vLLM's Triton ``FusedRMSNormGated`` as the layer's
    ``o_norm``. It has the same parameter name, so checkpoints load unchanged.
    The Kimi ops apply the norm themselves and read only ``weight`` and
    ``eps``; ``forward`` is the reference math, for callers outside those ops.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        activation: str = "sigmoid",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if activation not in ("sigmoid", "silu", "swish"):
            raise ValueError(f"Unsupported activation: {activation!r}")
        self.hidden_size = hidden_size
        self.eps = eps
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        out = x_float * torch.rsqrt(variance + self.eps) * self.weight.float()
        g_float = g.float()
        if self.activation == "sigmoid":
            out = out * torch.sigmoid(g_float)
        else:
            out = out * torch.nn.functional.silu(g_float)
        return out.to(x.dtype)


def _sp_collectives() -> Any:
    """GLM-5.3's sequence-parallel token collectives."""
    from vllm_torchtpu.layers.adapter.custom_ops import glm53_collectives_tpu

    return glm53_collectives_tpu


class Glm53TpuKimiDeltaAttention(KimiGatedDeltaNetAttention):
    """``KimiGatedDeltaNetAttention`` on the TPU Kimi KDA ops."""

    def __init__(
        self,
        config: Any,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
        if self.conv1d.bias is not None:
            raise NotImplementedError(
                "The TPU KDA ops take a bias-free short convolution."
            )
        # The base class builds a Triton norm; swap in a TPU module with the
        # same parameter name, dtype and device so weight loading is
        # unchanged.
        base_norm = self.o_norm
        self.o_norm: Any = Glm53GatedRMSNorm(
            self.head_dim,
            eps=float(base_norm.eps),
            activation=str(getattr(base_norm, "activation", "sigmoid")),
            device=base_norm.weight.device,
            dtype=base_norm.weight.dtype,
        )
        if self.o_norm.activation != "sigmoid":
            raise NotImplementedError(
                "The TPU KDA ops apply a sigmoid output gate, got "
                f"{self.o_norm.activation!r}."
            )
        eps = self.o_norm.eps
        self.kv_cache: Any = None
        # The same pair of ops Kimi-K3 builds (models/vllm/kimi_k3/attention.py).
        self.dispatched_kda_op = build_kimi_dispatched_kda_op(
            prefix,
            lower_bound=self.gate_lower_bound,
            eps=eps,
            state_dim_first=is_conv_state_dim_first(),
            num_spec_tokens=self.num_spec,
        )
        # The manager block size is read per call inside the op: it is not
        # final at model construction.
        model_cfg = getattr(vllm_config, "model_config", None)
        arch_cfg = getattr(model_cfg, "model_arch_config", None)
        is_mla_pool = bool(
            getattr(arch_cfg, "is_deepseek_mla", False)
            if arch_cfg is not None
            else getattr(model_cfg, "is_deepseek_mla", False)
        )
        self.pooled_kda_op = build_kimi_pooled_kda_op(
            prefix,
            lower_bound=self.gate_lower_bound,
            eps=eps,
            vllm_config=vllm_config,
            num_spec_tokens=self.num_spec,
            pool_tokens_per_row=(
                None if is_mla_pool else GLM53_KDA_POOL_TOKENS_PER_ROW
            ),
        )
        # The ops take the conv weight as [kernel_size, 3, heads, head_dim];
        # `fold_conv_weight` derives it from `conv1d.weight` after loading.
        # Non-persistent: it is not a checkpoint entry.
        self.register_buffer(
            "conv_weight_fused",
            torch.empty(
                self.conv_size,
                3,
                self.local_num_heads,
                self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self._conv_weight_folded = False

    def get_state_shape(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return glm53_kda_state_shape(
            self.conv_size, self.local_num_heads, self.head_dim
        )

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return glm53_kda_state_dtype()

    def fold_conv_weight(self) -> None:
        """Fold ``conv1d.weight`` into the Kimi ops' ``[K, 3, H, D]`` layout.

        ``conv1d.weight`` is ``[3 * local_proj, 1, K]`` with q, k, v stacked in
        that order and each laid out as (head, head_dim), which is exactly the
        channel order of ``mixed_qkv``. Idempotent.
        """
        weight = cast(torch.Tensor, self.conv1d.weight)
        kernel_size = int(weight.shape[-1])
        fused = (
            weight[:, 0, :]
            .t()
            .reshape(kernel_size, 3, self.local_num_heads, self.head_dim)
        )
        # `fused` is a strided view; a fresh buffer keeps torch_tpu from
        # re-materializing it as a per-step program at the op boundary.
        self.conv_weight_fused = fresh_buffer(fused)
        self._conv_weight_folded = True

    def process_weights_after_loading(
        self, act_dtype: torch.dtype | None = None
    ) -> None:
        del act_dtype
        self.fold_conv_weight()

    def _metadata(self) -> Any:
        """This layer's attention metadata, or None before the cache exists.

        A key lookup with an explicit fallback rather than ``dict.get`` with an
        eagerly evaluated default, so Dynamo traces one branch. A prefix that
        is missing from the per-layer dict (a draft-model KDA layer outside the
        target's KV cache groups) reads the step's first entry, which carries
        the same request layout.
        """
        if not is_forward_context_available():
            return None
        metadata = get_forward_context().attn_metadata
        if metadata is None:
            return None
        if isinstance(metadata, dict):
            if not metadata:
                return None
            if self.prefix in metadata:
                metadata = metadata[self.prefix]
            else:
                metadata = next(iter(metadata.values()))
        if isinstance(metadata, GDNAttentionMetadata):
            raise NotImplementedError(
                "The TPU KDA layer runs on vllm_torchtpu AttentionMetadata, "
                "not vLLM's GDNAttentionMetadata."
            )
        return metadata

    def _has_state(self) -> bool:
        kv_cache = self.kv_cache
        return kv_cache is not None and len(kv_cache) > 0 and kv_cache[0].numel() > 0

    def _kda_core(
        self,
        metadata: Any,
        mixed_qkv: torch.Tensor,  # [T, 3 * H * D], pre-convolution
        raw_gate: torch.Tensor,  # [T, H * D]
        beta: torch.Tensor,  # [T, H], raw
        output_gate: torch.Tensor,  # [T, H * D], raw
    ) -> torch.Tensor:
        """Run the Kimi KDA op; return the gated, normed ``[T, H, D]`` output."""
        if not self._conv_weight_folded:
            raise RuntimeError(
                f"{self.prefix}: conv_weight_fused was never folded; "
                "process_weights_after_loading did not run."
            )
        state_indices = metadata.mamba_state_indices
        if state_indices is None:
            raise RuntimeError(
                "The TPU KDA ops require mamba_state_indices in the attention metadata."
            )
        if getattr(metadata, "mamba_prefill_ckpt_offsets", None) is not None:
            raise NotImplementedError(
                "The TPU KDA ops do not write internal prefill checkpoints."
            )
        # No-ops for tensors that are already contiguous; column splits of the
        # fused projection need them.
        mixed_qkv = mixed_qkv.contiguous()
        raw_gate = raw_gate.contiguous()
        beta = beta.contiguous()
        output_gate = output_gate.contiguous()
        kv_cache = self.kv_cache
        window_distribution = metadata.mamba_request_distribution
        slot_read_offsets = metadata.mamba_slot_read_offsets

        if len(kv_cache) == 1:
            if is_pcp_streaming_attention_metadata(metadata):
                raise NotImplementedError(
                    "KDA does not support PCP streaming prefill with the "
                    "unified KV pool."
                )
            ckpt_indices = getattr(metadata, "mamba_ckpt_indices", None)
            if self.num_spec and (slot_read_offsets is None or ckpt_indices is None):
                raise RuntimeError(
                    "Speculative decoding with the pooled KDA op requires "
                    "mamba_slot_read_offsets and mamba_ckpt_indices in the "
                    "attention metadata."
                )
            return self.pooled_kda_op(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                kv_cache[0],
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

        conv_cache, recurrent_cache = kv_cache[0], kv_cache[1]
        if window_distribution is None:
            window_distribution = metadata.request_distribution
        if slot_read_offsets is None:
            # Non-speculative steps keep the op's ABI without allocating a
            # pool-sized offset tensor; the num_spec_tokens=0 op never reads
            # this placeholder.
            slot_read_offsets = state_indices
        return self.dispatched_kda_op(
            mixed_qkv,
            raw_gate,
            beta,
            output_gate,
            conv_cache,
            recurrent_cache,
            self.conv_weight_fused,
            self.A_log,
            self.dt_bias,
            self.o_norm.weight,
            cast(torch.Tensor, metadata.query_start_loc),
            cast(torch.Tensor, state_indices),
            cast(torch.Tensor, metadata.seq_lens),
            cast(torch.Tensor, metadata.request_distribution),
            cast(torch.Tensor, window_distribution),
            cast(torch.Tensor, slot_read_offsets),
        )

    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        """The base class's kernel seam, on the TPU ops.

        ``forward`` does not go through here; this keeps the base contract
        (fill ``core_attn_out`` with the normed, gated output) for callers
        that use it directly.
        """
        num_tokens = mixed_qkv.shape[0]
        projection = self.local_num_heads * self.head_dim
        metadata = self._metadata()
        if metadata is None or not self._has_state():
            # Profiling runs before the runner binds the state cache.
            core_attn_out.zero_()
            return
        output = self._kda_core(
            metadata,
            mixed_qkv,
            g1.reshape(num_tokens, projection),
            beta.reshape(num_tokens, self.local_num_heads),
            g2.reshape(num_tokens, projection),
        )
        core_attn_out.copy_(output.reshape(core_attn_out.shape))

    def forward_direct(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """Projections, KDA and ``o_proj``; returns the layer output.

        With ``sequence_parallel`` the caller holds ``T / tp`` tokens: they are
        all-gathered before the projections, and ``o_proj``'s partial sums are
        reduce-scattered back instead of all-reduced.

        Before the state cache exists (profiling runs, including the compiled
        memory profile) the KDA output is zeros, so the projections and
        ``o_proj`` still run at the step's shapes, as in Kimi-K3.
        """
        del positions
        collectives = _sp_collectives() if sequence_parallel else None
        if collectives is not None:
            hidden_states = collectives.token_all_gather(hidden_states)
        projected = self.in_proj_qkvgfab(hidden_states)[0]
        if self.use_full_rank_gate:
            split_sizes = [
                3 * self.local_projection_size,
                self.local_projection_size,
                self.head_dim,
                self.local_num_heads,
            ]
            if self.in_proj_padding:
                split_sizes.append(self.in_proj_padding)
            mixed_qkv, output_gate, f_a, beta = projected.split(split_sizes, dim=-1)[:4]
        else:
            mixed_qkv, beta, f_a = projected.split(
                [
                    3 * self.local_projection_size,
                    self.local_num_heads,
                    self.head_dim,
                ],
                dim=-1,
            )
            output_gate = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        raw_gate = self.f_b_proj(f_a)[0]

        metadata = self._metadata()
        if metadata is None or not self._has_state():
            core = torch.zeros_like(output_gate)
        else:
            core = self._kda_core(
                metadata, mixed_qkv, raw_gate, beta, output_gate
            ).view(-1, self.local_projection_size)
        if collectives is not None:
            assert self.o_proj.quant_method is not None
            partial = self.o_proj.quant_method.apply(self.o_proj, core, bias=None)
            return collectives.token_reduce_scatter(partial)
        return self.o_proj(core)[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        sequence_parallel: bool = False,
    ) -> torch.Tensor:
        return self.forward_direct(
            hidden_states, positions, sequence_parallel=sequence_parallel
        )
