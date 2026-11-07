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
"""Ling 3.x (vLLM's ``BailingMoeV3ForCausalLM``) on TPU.

vLLM's model runs unchanged except for its Kimi Delta Attention layers. Its MLA
layers already build the TPU MLA wrapper (``MultiHeadLatentAttentionWrapper``
is replaced out of tree), and its fused MoE and linears use the TPU adapters.
Its KDA layer runs the delta rule through Triton (``fla``) under a CUDA-graph
splitting op, so this module swaps in, out of tree, a KDA layer that keeps the
checkpoint's parameter names and loaders but runs the Kimi KDA ops (fused
conv1d + GDN v3 in KDA mode, per-layer or unified-pool state). Ling differs
from Kimi-Linear only in the projections: full-rank ``f_proj`` / ``g_proj``
(no LoRA) and an unquantized ``b_proj`` beside the fp8 ``q|k|v``.

The model class overrides the hybrid-state hooks, so the cache is sized from
the same TPU state layout the layer declares, and selects the TPU MoE kernels'
clamped SiLU on the routed experts of the layers with a SwiGLU limit.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models import bailing_moe_v3 as upstream
from vllm.model_executor.utils import set_weight_attrs

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.layers.adapter.custom_ops.kda_attention_op import (
    build_kimi_dispatched_kda_op,
    build_kimi_pooled_kda_op,
)
from vllm_torchtpu.models.vllm.kimi_k3.attention import (
    CausalDepthwiseConv1d,
    KimiDeltaAttention,
    _load_a_log,
    kda_state_dtype,
)


def kda_head_dim(config) -> int:
    """KDA head size. transformers' BailingHybridConfig keeps the checkpoint's
    `head_dim` (128) as `linear_head_dim` and then overwrites `head_dim` with
    qk_rope_head_dim (64); the checkpoint's remote-code config has only
    `head_dim`."""
    return getattr(config, "linear_head_dim", config.head_dim)


@upstream.BailingMoeV3KimiDeltaAttention.register_oot
class TPUBailingMoeV3KimiDeltaAttention(upstream.BailingMoeV3KimiDeltaAttention):
    """Ling KDA on the Kimi TPU KDA ops.

    Subclasses the upstream layer so ``PluggableLayer.__new__`` can return it
    and Python still runs this ``__init__`` (with the upstream signature).
    Parameter names match upstream's ``load_weights``; the conv weights and
    ``A_log`` use the Kimi containers so no linear adapter repacks them.
    """

    supports_dcp = False

    def __init__(
        self,
        config,
        quant_config=None,
        layer_id: int = 0,
        prefix: str = "attention",
        model_config=None,
        cache_config=None,
        num_speculative_tokens: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        if not getattr(config, "no_kda_lora", False):
            raise ValueError("BailingMoeV3 KDA expects no_kda_lora=True")
        self.vllm_config = get_current_vllm_config()
        self.prefix = prefix
        self.layer_id = layer_id
        self.model_config = model_config
        self.cache_config = cache_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.head_dim = kda_head_dim(config)
        self.total_heads = config.num_attention_heads
        if self.total_heads % self.tp_size:
            raise ValueError("KDA num_heads must be divisible by TP size")
        self.num_heads = self.total_heads // self.tp_size
        self.local_projection_size = self.num_heads * self.head_dim
        self.conv_size = config.short_conv_kernel_size
        self.num_spec_tokens = num_speculative_tokens
        # The gate is bounded only under safe_gate (vLLM's fused_kda_gate call).
        self.gate_lower_bound = (
            getattr(config, "kda_lower_bound", -5.0)
            if getattr(config, "kda_safe_gate", True)
            else None
        )
        projection_size = self.total_heads * self.head_dim

        # Same projections and names as upstream. b_proj is unquantized in
        # the released checkpoint, so it cannot share the fp8 q|k|v matmul.
        self.separate_b_proj = upstream._is_fp8_module_excluded(
            quant_config, f"{prefix}.b_proj"
        )
        if self.separate_b_proj:
            self.qkv_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [projection_size] * 3,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.b_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.b_proj",
            )
            self.qkvb_proj = None
        else:
            self.qkvb_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [projection_size] * 3 + [self.total_heads],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkvb_proj",
            )
            self.qkv_proj = None
            self.b_proj = None
        self.f_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_proj",
        )
        self.g_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32)
        )
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})
        self.A_log = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        set_weight_attrs(self.A_log, {"weight_loader": _load_a_log})
        self.q_conv1d = CausalDepthwiseConv1d(projection_size, self.conv_size)
        self.k_conv1d = CausalDepthwiseConv1d(projection_size, self.conv_size)
        self.v_conv1d = CausalDepthwiseConv1d(projection_size, self.conv_size)
        # Folded from q/k/v_conv1d at load time into the kernels'
        # [kernel_size, 3, heads, head_dim] layout (KimiDeltaAttention).
        self.register_buffer(
            "conv_weight_fused",
            torch.empty(self.conv_size, 3, self.num_heads, self.head_dim),
            persistent=False,
        )
        # Weight holder only: the KDA op applies the sigmoid-gated RMSNorm.
        self.o_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # Kimi's sequence-parallel prefill packing is K3-specific; off here.
        self.prefill_group = None
        self.packed_prefill_weight = None
        self.kv_cache = None
        self.dispatched_kda_op = build_kimi_dispatched_kda_op(
            prefix,
            lower_bound=self.gate_lower_bound,
            eps=config.rms_norm_eps,
            state_dim_first=is_conv_state_dim_first(),
            num_spec_tokens=self.num_spec_tokens,
        )
        self.pooled_kda_op = build_kimi_pooled_kda_op(
            prefix,
            lower_bound=self.gate_lower_bound,
            eps=config.rms_norm_eps,
            vllm_config=self.vllm_config,
            num_spec_tokens=self.num_spec_tokens,
        )
        context = self.vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    # The state declaration, metadata lookup, conv-weight folding and pooled
    # dispatch read only the attributes set above, so share Kimi's code.
    mamba_type = KimiDeltaAttention.mamba_type
    get_state_dtype = KimiDeltaAttention.get_state_dtype
    get_state_shape = KimiDeltaAttention.get_state_shape
    _metadata = KimiDeltaAttention._metadata
    process_weights_after_loading = KimiDeltaAttention.process_weights_after_loading
    _pack_prefill_projection = KimiDeltaAttention._pack_prefill_projection
    _core_attention_pooled = KimiDeltaAttention._core_attention_pooled

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Upstream's call contract: writes the attention output into `output`."""
        del positions
        if self.separate_b_proj:
            assert self.qkv_proj is not None and self.b_proj is not None
            mixed_qkv, _ = self.qkv_proj(hidden_states)
            beta, _ = self.b_proj(hidden_states)
        else:
            assert self.qkvb_proj is not None
            projected, _ = self.qkvb_proj(hidden_states)
            mixed_qkv, beta = projected.split(
                [3 * self.local_projection_size, self.num_heads], dim=-1
            )
        # Raw f, beta logits and output gate: the op applies the bounded gate,
        # the beta sigmoid and the sigmoid-gated RMSNorm itself.
        raw_gate, _ = self.f_proj(hidden_states)
        output_gate, _ = self.g_proj(hidden_states)

        metadata = self._metadata()
        if metadata is None or self.kv_cache is None or self.kv_cache[0].numel() == 0:
            core = torch.zeros_like(raw_gate)
        elif len(self.kv_cache) == 1:
            core = self._core_attention_pooled(
                mixed_qkv,
                raw_gate,
                beta,
                output_gate,
                self.kv_cache[0],
                metadata,
            ).flatten(1)
        else:
            sconv_cache, recurrent_cache = self.kv_cache
            window_distribution = metadata.mamba_request_distribution
            if window_distribution is None:
                window_distribution = metadata.request_distribution
            slot_read_offsets = metadata.mamba_slot_read_offsets
            if slot_read_offsets is None:
                slot_read_offsets = metadata.mamba_state_indices
                if slot_read_offsets is None:
                    slot_read_offsets = metadata.seq_lens
            core = self.dispatched_kda_op(
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
                metadata.query_start_loc,
                metadata.mamba_state_indices,
                metadata.seq_lens,
                metadata.request_distribution,
                window_distribution,
                slot_read_offsets,
            ).flatten(1)
        out, _ = self.o_proj(core)
        output.copy_(out)


# The clamp the GMM / fused-EP kernels implement as "silu_and_mul_with_clamp":
# silu(min(g, 10)) * clip(u, +-10). Ling clamps after the SiLU
# (min(silu(g), 10)); the two differ by at most 10 - silu(10) = 4.5e-4 on the
# gate, far below bf16 resolution at 10 (0.0625).
_KERNEL_CLAMP_LIMIT = 10.0


def _clamp_routed_experts(model: nn.Module) -> None:
    """Select the kernels' clamped SiLU on MoE layers with a swiglu limit.

    vLLM passes Ling's per-layer limit (10 on layers 65-71) to the routed
    experts' moe_config, but this base's TPU activation resolver
    (get_fused_moe_activation) returns plain "silu" and drops it. Must run
    before weight processing, which prebuilds the kernels per activation.
    """
    seen = clamped = 0
    for module in model.modules():
        limit = getattr(getattr(module, "moe_config", None), "swiglu_limit", None)
        if not limit:
            continue
        seen += 1
        if float(limit) != _KERNEL_CLAMP_LIMIT:
            raise NotImplementedError(f"TPU MoE kernels clamp at 10, not {limit}")
        # vLLM's MoE runner carries the layer's moe_config but not its quant
        # method; the routed experts below it own the method and take the clamp.
        if not hasattr(module, "quant_method"):
            continue
        method = module.quant_method
        # Fail loudly rather than serve the layer unclamped: a quant method without
        # the TPU activation hook, or the adaptive fused MoE (it accepts only "silu").
        if not hasattr(method, "_resolve_tpu_activation"):
            raise NotImplementedError(
                f"{type(method).__name__} cannot apply Ling's routed-expert clamp"
            )
        if tpu_envs.MOE_FUSED_EP_KERNEL_IMPL == "adaptive":
            raise NotImplementedError("The adaptive fused MoE has no clamped SiLU")
        method._resolve_tpu_activation = lambda layer: "silu_and_mul_with_clamp"
        clamped += 1
    if seen and not clamped:
        raise NotImplementedError(
            "no MoE quant method under the clamped layers took Ling's clamp"
        )


class BailingMoeV3ForCausalLM(upstream.BailingMoeV3ForCausalLM):
    """vLLM's model, with the KDA state hooks in the TPU layer's layout and
    the routed-expert SwiGLU clamp selected for the TPU MoE kernels."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _clamp_routed_experts(self)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        # Same helper the layer uses, so cache sizing and allocation agree.
        return kda_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], tuple[int, int, int]]:
        config = vllm_config.model_config.hf_text_config
        heads = (
            config.num_attention_heads
            // vllm_config.parallel_config.tensor_parallel_size
        )
        d = kda_head_dim(config)
        # Matches TPUBailingMoeV3KimiDeltaAttention.get_state_shape.
        return ((config.short_conv_kernel_size - 1, 1, 3 * heads * d), (heads, d, d))
