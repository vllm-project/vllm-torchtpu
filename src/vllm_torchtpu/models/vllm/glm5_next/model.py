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
"""GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) text model for TorchTPU.

Consumes `transformers.Glm5NextTextConfig` (`model_config.hf_text_config`):

1. KDA linear-attention layers (`layer_types[i] == "linear_attention"`) run
   the Kimi KDA ops through `Glm53TpuKimiDeltaAttention`, on the unified
   block pool or on per-layer conv / recurrent state caches.
2. The other layers are NoPE multi-head latent attention (`q_lora_rank`,
   `kv_lora_rank`, `qk_rope_head_dim=0`). The checkpoint's DeepSeek sparse
   attention indexer (`index_topk`, `index_kpool`, `indexer_types`) is not
   run: attention is dense and causal, which selects the same keys as the
   indexer while a sequence fits in its top-k budget. The indexer tensors are
   skipped at load. See `TPUGlm5NextConfig` for the warning past that length.
3. Manifold-constrained hyper-connections (mHC, `hc_mult` residual streams)
   around both sublayers of every decoder layer, on the Pallas mHC kernels
   (each post fused into the next pre) or the compiled reference ops.
4. Dense clamped-SwiGLU MLPs on the first layers (`mlp_layer_types`), then
   MoE layers with sigmoid `noaux_tc` routing, one shared expert and the same
   clamped SwiGLU in every expert.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from torch import nn
from torch_tpu._internal import pallas
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    unified_attention_with_output,
    unified_kv_cache_update,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import _encode_layer_name

from vllm_torchtpu.layers.adapter.custom_ops import (
    glm53_kda_tpu,
    glm53_mhc_tpu,
)
from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUMLAAttention,
)
from vllm_torchtpu.logger import init_logger

from .config import GLM53_MLA_ROPE_PAD, MLA_IMPL_ABSORBED, glm53_mla_impl

logger = init_logger(__name__)


# Checkpoint (unfused) -> vLLM fused module names. Also consumed by the FP8
# config (`is_layer_skipped`) to see that the fused KDA projection is BF16.
PACKED_MODULES_MAPPING: dict[str, list[str]] = {
    "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"],
    "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

# (fused vLLM param component, checkpoint component, shard id). The shard ids
# follow the Kimi KDA layer's `in_proj_qkvgfab` (q, k, v, beta, f_a) and its
# fused `conv1d` (q, k, v).
STACKED_PARAMS_MAPPING: tuple[tuple[str, str, int], ...] = (
    (".in_proj_qkvgfab", ".q_proj", 0),
    (".in_proj_qkvgfab", ".k_proj", 1),
    (".in_proj_qkvgfab", ".v_proj", 2),
    (".in_proj_qkvgfab", ".b_proj", 3),
    (".in_proj_qkvgfab", ".f_a_proj", 4),
    (".conv1d", ".q_conv1d", 0),
    (".conv1d", ".k_conv1d", 1),
    (".conv1d", ".v_conv1d", 2),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
)

EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.mlp\.experts\.)(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<suffix>weight(?:_scale_inv)?)$"
)
# checkpoint projection -> (FusedMoE param family, FusedMoE shard id)
EXPERT_SHARDS: dict[str, tuple[str, str]] = {
    "gate_proj": ("w13", "w1"),
    "up_proj": ("w13", "w3"),
    "down_proj": ("w2", "w2"),
}
_LAYER_IDX_RE = re.compile(r"^model\.layers\.(\d+)\.")
_CKPT_RENAMES: tuple[tuple[str, str], ...] = (
    (".hc_attn_fn", ".attn_hc.fn"),
    (".hc_attn_base", ".attn_hc.base"),
    (".hc_attn_scale", ".attn_hc.scale"),
    (".hc_ffn_fn", ".ffn_hc.fn"),
    (".hc_ffn_base", ".ffn_hc.base"),
    (".hc_ffn_scale", ".ffn_hc.scale"),
    (".input_layernorm.weight", ".attn_hc.sublayer_norm_weight"),
    (".post_attention_layernorm.weight", ".ffn_hc.sublayer_norm_weight"),
)


def normalize_glm53_checkpoint_name(name: str, num_hidden_layers: int) -> str | None:
    """Maps a GLM-5.3-Flash checkpoint tensor name into this model's namespace.

    Returns None for tensors that text serving does not use: the vision tower,
    the MTP layers (`layers.{num_hidden_layers}` and up), `rotary_emb.inv_freq`
    and the sparse attention indexer, which the TPU model does not run.
    """
    if name.startswith(("model.visual.", "visual.")):
        return None
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]
    elif name.startswith("language_model."):
        name = "model." + name[len("language_model.") :]
    layer_match = _LAYER_IDX_RE.match(name)
    if layer_match is not None and int(layer_match.group(1)) >= num_hidden_layers:
        return None
    if "rotary_emb.inv_freq" in name or ".self_attn.indexer." in name:
        return None
    for src, dst in _CKPT_RENAMES:
        if name.endswith(src):
            return name[: -len(src)] + dst
    return name


def is_kda_layer(config: Any, layer_idx: int) -> bool:
    """Whether decoder layer `layer_idx` is a KDA linear-attention layer."""
    layer_types = config.layer_types
    return (
        layer_types is not None
        and layer_idx < len(layer_types)
        and layer_types[layer_idx] == "linear_attention"
    )


def kda_linear_attn_config(config: Any) -> dict[str, Any]:
    """The `linear_attn_config` dict the Kimi KDA layer reads.

    `Glm5NextTextConfig` keeps the KDA geometry as `linear_*` attributes; the
    raw dict survives only when the checkpoint ships one. Rebuilt from the
    attributes so both forms give the same layer.
    """
    raw = getattr(config, "linear_attn_config", None)
    kda_config = dict(raw) if isinstance(raw, dict) else {}
    kda_config.update(
        num_heads=int(config.linear_num_heads),
        head_dim=int(config.linear_head_dim),
        short_conv_kernel_size=int(config.linear_conv_kernel_dim),
        gate_lower_bound=getattr(config, "linear_lower_bound", -9.0),
        safe_gate=True,
    )
    return kda_config


class KdaLayerConfig:
    """The text config as the KDA layer sees it.

    Adds `linear_attn_config` (see `kda_linear_attn_config`) and forwards
    every other attribute, so the shared `hf_text_config` is never written.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        self.linear_attn_config = kda_linear_attn_config(config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)


def _clamped_swiglu_fn(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def _clamped_swiglu_jax(gate_up: jax.Array, *, limit: float) -> jax.Array:
    """JAX twin of `_clamped_swiglu_fn` (same op order and dtypes)."""
    gate, up = jnp.split(gate_up, 2, axis=-1)
    gate = jnp.minimum(gate, limit)
    up = jnp.clip(up, -limit, limit)
    # torch's bf16 silu computes in fp32 and rounds once.
    act = jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype)
    return act * up


_SWIGLU_OPS: dict[float, Any] = {}


def get_clamped_swiglu_op(limit: float) -> Any:
    """Build (or fetch) the clamped SwiGLU op; call it at construction.

    Registering a `jax_op` cannot be traced, so the compiled backbone must
    find it cached.
    """
    op = _SWIGLU_OPS.get(limit)
    if op is None:
        op = pallas.jax_op(
            f"glm53::clamped_swiglu_l{limit:g}".replace(".", "p"),
            functools.partial(_clamped_swiglu_jax, limit=limit),
        )
        # The default fake rejects the symbolic token axis of the compiled
        # backbone's trace.
        op.register_fake(
            lambda gate_up: gate_up.new_empty(
                (*gate_up.shape[:-1], gate_up.shape[-1] // 2)
            )
        )
        _SWIGLU_OPS[limit] = op
    return op


def _clamped_swiglu(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """`silu(gate.clamp(max=L)) * up.clamp(-L, L)`; one `jax_op` on TPU."""
    if gate_up.device.type != "tpu":
        return _clamped_swiglu_fn(gate_up, limit)
    op = _SWIGLU_OPS.get(limit)
    if op is None:
        if torch.compiler.is_compiling():
            raise RuntimeError(
                f"glm53 clamped SwiGLU op for limit={limit} was not built "
                "before tracing; call get_clamped_swiglu_op at model "
                "construction."
            )
        op = get_clamped_swiglu_op(limit)
    return op(gate_up.contiguous())


# The TPU GMM kernel's clamped SwiGLU (`silu_and_mul_with_clamp`) hard-codes
# this limit; it equals the released GLM-5.3-Flash `swiglu_limit`.
_TPU_CLAMPED_SWIGLU_LIMIT = 10.0


def _enable_tpu_swiglu_clamp(experts: nn.Module, swiglu_limit: float) -> None:
    """Makes the TPU fused-MoE kernel apply GLM's clamped SwiGLU.

    The reference computes `silu(gate.clamp(max=L)) * up.clamp(-L, L)` in
    every expert. The TPU FP8 / unquantized MoE methods ignore `swiglu_limit`
    and resolve `silu` to the plain kernel activation, so this layer's method
    resolves it to the clamped kernel variant instead, as the MXFP4 method
    does. The string is resolved in `process_weights_after_loading`, after
    this is installed.
    """
    routed = getattr(experts, "routed_experts", experts)
    quant_method: Any = getattr(routed, "quant_method", None)
    resolve = getattr(quant_method, "_resolve_tpu_activation", None)
    if resolve is None:
        return  # Not a TPU MoE method.
    if abs(float(swiglu_limit) - _TPU_CLAMPED_SWIGLU_LIMIT) > 1e-6:
        raise NotImplementedError(
            f"GLM-5.3-Flash swiglu_limit={swiglu_limit} is not supported by the "
            f"TPU clamped SwiGLU kernel (limit={_TPU_CLAMPED_SWIGLU_LIMIT})."
        )

    def _resolve_clamped(layer: Any) -> str:
        activation = resolve(layer)
        if activation.lower() in ("silu", "silu_and_mul", "swiglu"):
            return "silu_and_mul_with_clamp"
        return activation

    quant_method._resolve_tpu_activation = _resolve_clamped


class Glm5NextMLP(nn.Module):
    """Clamped SwiGLU MLP used by the dense layers and the shared expert."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float = 10.0,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.swiglu_limit = float(swiglu_limit)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        # Built now so the compiled backbone finds it cached.
        get_clamped_swiglu_op(self.swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        activated = _clamped_swiglu(gate_up, self.swiglu_limit)
        out, _ = self.down_proj(activated)
        return out


class Glm5NextMoE(nn.Module):
    """GLM-5.3-Flash MoE block: routed experts, top-k, one shared expert.

    Routing matches the reference (`noaux_tc`): `sigmoid(logits) + bias` picks
    the top-k experts, the un-biased sigmoid scores of those experts are
    renormalized and multiplied by `routed_scaling_factor`. The shared expert
    is added unscaled. Routed experts are vLLM's `FusedMoE`, so the block-FP8
    checkpoint experts load through it, and its runner computes the routed
    and shared experts and reduces them across TP. The TPU MoE kernels apply
    GLM's clamped SwiGLU (see `_enable_tpu_swiglu_clamp`).
    """

    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.moe_intermediate_size = int(config.moe_intermediate_size)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.swiglu_limit = float(config.swiglu_limit)

        # Registration order matters: `gate` and `shared_experts` are
        # registered before `experts` so `named_parameters()` reports them
        # under the checkpoint names (`mlp.gate.*`, `mlp.shared_experts.*`).
        self.gate = GateLinear(
            input_size=self.hidden_size,
            output_size=self.num_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32),
            requires_grad=False,
        )

        n_shared = int(config.n_shared_experts or 0)
        self.shared_experts: Glm5NextMLP | None = None
        if n_shared > 0:
            self.shared_experts = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.moe_intermediate_size * n_shared,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = FusedMoEFactory(
            gate=self.gate,
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            renormalize=bool(config.norm_topk_prob),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=str(config.scoring_func),
            e_score_correction_bias=cast(
                torch.Tensor | None, self.gate.e_score_correction_bias
            ),
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
        )
        _enable_tpu_swiglu_clamp(self.experts, self.swiglu_limit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        is_2d = hidden_states.ndim == 2
        x_flat = hidden_states if is_2d else hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(x_flat)
        # Routed + shared expert outputs, reduced across TP by the runner.
        out = self.experts(hidden_states=x_flat, router_logits=router_logits)
        return out if is_2d else out.view_as(hidden_states)


def _run_rpa_attention(
    attn: Any, q: torch.Tensor, k_flat: torch.Tensor, v_flat: torch.Tensor
) -> torch.Tensor:
    """Run `Attention` without building a `SymInt` size tuple under `torch.compile`.

    `Attention.forward` allocates `torch.empty((num_tokens, ...))`, which fails
    TorchTPU's `fx_to_mlir` when `num_tokens` is a symbolic expression rather
    than a plain graph input. `torch.zeros_like` on
    `v_flat` (`[num_tokens, num_heads * head_size_v]`) avoids a `SymIntArrayRef`
    shape tuple while zero-initializing the output buffer for memory-probe /
    padded-tail calls where the backend leaves `out_3d` untouched.
    """
    if not torch.compiler.is_compiling() or not isinstance(attn, Attention):
        return attn(q, k_flat, v_flat)
    output_dtype = q.dtype
    if (
        attn.query_quant is not None
        and attn.impl.supports_quant_query_input
        and q.dtype != current_platform.fp8_dtype()
    ):
        q, _ = attn.query_quant(q, attn._q_scale)
    output = torch.zeros_like(
        v_flat, dtype=output_dtype, memory_format=torch.contiguous_format
    )
    q_3d = q.view(-1, attn.num_heads, attn.head_size)
    k_3d = k_flat.view(-1, attn.num_kv_heads, attn.head_size)
    v_3d = v_flat.view(-1, attn.num_kv_heads, attn.head_size_v)
    out_3d = output.view(-1, attn.num_heads, attn.head_size_v)
    kv_cache_dummy_dep = None
    if attn.use_direct_call:
        if (
            not attn.attn_backend.forward_includes_kv_cache_update
            and attn.kv_sharing_target_layer_name is None
        ):
            kv_cache_dummy_dep = unified_kv_cache_update(k_3d, v_3d, attn.layer_name)
        unified_attention_with_output(
            q_3d,
            k_3d,
            v_3d,
            out_3d,
            attn.layer_name,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
    else:
        encoded = _encode_layer_name(attn.layer_name)
        if (
            not attn.attn_backend.forward_includes_kv_cache_update
            and attn.kv_sharing_target_layer_name is None
        ):
            kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                k_3d, v_3d, encoded
            )
        torch.ops.vllm.unified_attention_with_output(
            q_3d,
            k_3d,
            v_3d,
            out_3d,
            encoded,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )
    return output


class Glm5NextMLAAttention(nn.Module):
    """NoPE multi-head latent attention.

    With `qk_rope_head_dim=0` the decompressed per-head keys and values are
    plain MHA tensors, so the default "rpa" path runs them through vLLM's
    paged-KV `Attention` (the TPU ragged paged attention backend) with
    `scale = qk_head_dim ** -0.5`. The "absorbed" path caches the latent and
    runs `VllmTPUMLAAttention` instead. Dense causal attention; see the
    module docstring for the sparse attention indexer.
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = self.num_heads // self.tp_size
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim or 0)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(config.v_head_dim)
        self.scaling = self.qk_head_dim**-0.5
        if self.qk_rope_head_dim != 0:
            raise NotImplementedError(
                "Glm5NextMLAAttention only supports NoPE MLA (qk_rope_head_dim=0)."
            )

        self.q_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        # Built now so the compiled backbone finds them cached.
        glm53_mhc_tpu.get_rms_norm_weighted_op(self.q_a_layernorm.variance_epsilon)
        glm53_mhc_tpu.get_rms_norm_weighted_op(self.kv_a_layernorm.variance_epsilon)
        self.mla_impl = glm53_mla_impl()
        self.attn: Attention | None = None
        self.mla_attn: VllmTPUMLAAttention | None = None
        if self.mla_impl == MLA_IMPL_ABSORBED:
            # The cache holds the latent plus an all-zero rope slot (one KV
            # head), and the kernel folds W_UK into q and W_UV into the
            # output. kv_b_proj is split into W_UK_T / W_UV after loading. The
            # softmax scale stays the NoPE `qk_head_dim ** -0.5`.
            self.mla_attn = VllmTPUMLAAttention(
                num_heads=self.num_local_heads,
                scale=self.scaling,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=GLM53_MLA_ROPE_PAD,
                v_head_dim=self.v_head_dim,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                kv_b_proj=self.kv_b_proj,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )
        else:
            self.attn = Attention(
                num_heads=self.num_local_heads,
                head_size=self.qk_head_dim,
                scale=self.scaling,
                num_kv_heads=self.num_local_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                head_size_v=(
                    self.v_head_dim if self.v_head_dim != self.qk_head_dim else None
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Attention output `[T, hidden]`."""
        del positions  # NoPE: no rotary embedding on the MLA layers.
        # One compiled fusion per norm instead of eager `RMSNorm` (same op
        # order as vLLM's `rms_norm`, so bit-exact).
        q_lora, _ = self.q_a_proj(hidden_states)
        q_lora = glm53_mhc_tpu.rms_norm_weighted(
            q_lora,
            self.q_a_layernorm.weight,
            self.q_a_layernorm.variance_epsilon,
        )
        kv_a, _ = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent = glm53_mhc_tpu.rms_norm_weighted(
            kv_a[..., : self.kv_lora_rank],
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.variance_epsilon,
        )
        q, _ = self.q_b_proj(q_lora)  # [T, H_local * qk_head_dim]

        if self.mla_attn is not None:
            q_nope = q.view(-1, self.num_local_heads, self.qk_nope_head_dim)
            # zeros_like over an expanded view keeps the symbolic token dim
            # under torch.compile instead of passing a size tuple.
            q_pe = torch.zeros_like(q_nope[..., :1].expand(-1, -1, GLM53_MLA_ROPE_PAD))
            k_pe = torch.zeros_like(
                kv_latent[:, None, :1].expand(-1, -1, GLM53_MLA_ROPE_PAD)
            )
            attn_out = self.mla_attn((q_nope, q_pe), kv_latent, k_pe)
        else:
            assert self.attn is not None
            kv, _ = self.kv_b_proj(kv_latent)
            kv = kv.view(
                -1,
                self.num_local_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            )
            k, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            attn_out = _run_rpa_attention(
                self.attn,
                q,
                k.reshape(-1, self.num_local_heads * self.qk_head_dim),
                v.reshape(-1, self.num_local_heads * self.v_head_dim),
            )
        proj_out, _ = self.o_proj(attn_out)
        return proj_out


class MhcDeferredPost(NamedTuple):
    """An mHC post (stream recombine) a layer owes, settled by the next pre."""

    sublayer_out: torch.Tensor  # [T, hidden_size]
    residual: torch.Tensor  # [T, hc_mult * hidden_size] bf16
    post: torch.Tensor  # [T, hc_mult, 1] f32
    comb: torch.Tensor  # [T, hc_mult, hc_mult] f32


def mhc_pallas_enabled(vllm_config: VllmConfig) -> bool:
    """Whether the Pallas mHC kernels run (they take bf16 residual streams)."""
    return (
        glm53_mhc_tpu.glm53_mhc_impl() == glm53_mhc_tpu.MHC_IMPL_PALLAS
        and vllm_config.model_config.dtype == torch.bfloat16
    )


class Glm5NextDecoderLayer(nn.Module):
    """One decoder layer: mHC around (KDA | MLA) and (MLP | MoE)."""

    def __init__(
        self,
        config: Any,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)

        hc_kwargs = dict(
            hidden_size=self.hidden_size,
            hc_mult=int(config.hc_mult),
            hc_sinkhorn_iters=int(config.hc_sinkhorn_iters),
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(config.rms_norm_eps),
        )
        self.attn_hc = glm53_mhc_tpu.FusedGlm53HyperConnectionTpu(**hc_kwargs)
        self.ffn_hc = glm53_mhc_tpu.FusedGlm53HyperConnectionTpu(**hc_kwargs)
        self._mhc_pallas = mhc_pallas_enabled(vllm_config)

        if is_kda_layer(config, layer_idx):
            self.self_attn: (
                glm53_kda_tpu.Glm53TpuKimiDeltaAttention | Glm5NextMLAAttention
            ) = glm53_kda_tpu.Glm53TpuKimiDeltaAttention(
                config=KdaLayerConfig(config),
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = Glm5NextMLAAttention(
                config=config,
                layer_idx=layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp: nn.Module = Glm5NextMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=int(config.intermediate_size),
                swiglu_limit=float(config.swiglu_limit),
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_streams: torch.Tensor,
        deferred: MhcDeferredPost | None = None,
    ) -> tuple[torch.Tensor, MhcDeferredPost | None]:
        """Runs the layer; returns `(hidden_streams, deferred_post)`.

        On the Pallas mHC path the FFN post is not applied here: it is
        returned as `deferred_post` and fused into the next layer's
        attention pre (or settled by the caller after the last layer), and
        the returned `hidden_streams` is stale. On the compiled path
        `deferred_post` is always None.
        """
        if self._mhc_pallas:
            return self._forward_pallas_mhc(positions, hidden_streams, deferred)
        assert deferred is None
        post_attn, comb_attn, norm_attn_in = self.attn_hc.collapse_and_norm(
            hidden_streams
        )
        attn_out = self._run_attention(norm_attn_in, positions)
        hidden_streams = self.attn_hc.combine(
            hidden_streams, attn_out, post_attn, comb_attn
        )
        post_ffn, comb_ffn, norm_ffn_in = self.ffn_hc.collapse_and_norm(hidden_streams)
        ffn_out = self.mlp(norm_ffn_in)
        hidden_streams = self.ffn_hc.combine(
            hidden_streams, ffn_out, post_ffn, comb_ffn
        )
        return hidden_streams, None

    def _run_attention(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.self_attn, Glm5NextMLAAttention):
            return self.self_attn(hidden_states, positions)
        out = self.self_attn.forward_direct(hidden_states, positions)
        if out is not None:
            return out
        # The KDA layer's direct path declines eager steps without a state
        # cache (the startup profile run); run its generic forward instead.
        return self.self_attn(hidden_states=hidden_states, positions=positions)

    def _forward_pallas_mhc(
        self,
        positions: torch.Tensor,
        hidden_streams: torch.Tensor,
        deferred: MhcDeferredPost | None,
    ) -> tuple[torch.Tensor, MhcDeferredPost]:
        attn_hc, ffn_hc = self.attn_hc, self.ffn_hc
        if deferred is None:
            # Chain start: a standalone pre, nothing to fuse it with.
            residual = hidden_streams
            post, comb, norm_attn_in = attn_hc.mhc_ops.pre(
                residual,
                *attn_hc.fn_split,
                attn_hc.scale_f32,
                attn_hc.base_f32,
                attn_hc.sublayer_norm_weight,
            )
        else:
            residual, post, comb, norm_attn_in = attn_hc.mhc_ops.fused(
                deferred.sublayer_out,
                deferred.residual,
                deferred.post,
                deferred.comb,
                *attn_hc.fn_split,
                attn_hc.scale_f32,
                attn_hc.base_f32,
                attn_hc.sublayer_norm_weight,
            )
        attn_out = self._run_attention(norm_attn_in, positions)
        residual, post, comb, norm_ffn_in = ffn_hc.mhc_ops.fused(
            attn_out,
            residual,
            post,
            comb,
            *ffn_hc.fn_split,
            ffn_hc.scale_f32,
            ffn_hc.base_f32,
            ffn_hc.sublayer_norm_weight,
        )
        ffn_out = self.mlp(norm_ffn_in)
        return residual, MhcDeferredPost(ffn_out, residual, post, comb)


# Without `--enforce-eager` the runner compiles the whole backbone once per
# token bucket (as for Kimi-K3 and DeepSeek-V4), replacing about a thousand
# eager dispatches per decode step with one TPU program. Under --enforce-eager
# the decorator sets `do_not_compile` and `__call__` runs `forward` directly.
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
        "intermediate_tensors": 0,
    }
)
class Glm5NextTextModel(nn.Module):
    """GLM-5.3-Flash backbone carrying the `hc_mult`-stream mHC residual."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix or "model"
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.vocab_size = int(config.vocab_size)
        self.hc_mult = int(config.hc_mult)

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def _build_layer(prefix: str) -> Glm5NextDecoderLayer:
            return Glm5NextDecoderLayer(
                config=config,
                vllm_config=vllm_config,
                layer_idx=int(prefix.rsplit(".", 1)[1]),
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            int(config.num_hidden_layers),
            _build_layer,
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(self.hidden_size, eps=float(config.rms_norm_eps))
            glm53_mhc_tpu.get_rms_norm_weighted_op(self.norm.variance_epsilon)
        else:
            self.norm = PPMissingLayer()

        # Resolved here, not in `forward`: env reads and the mHC impl choice
        # would otherwise be traced into the compiled program as constants,
        # and every decoder layer agrees on them.
        self._mhc_pallas = mhc_pallas_enabled(vllm_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        layers = self.layers[self.start_layer : self.end_layer]
        # The Pallas mHC ops carry the streams flattened to [T, M * D]; the
        # compiled path keeps [T, M, D].
        flat = bool(layers) and self._mhc_pallas
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
            if flat:
                # [T, D] -> [T, M * D]: M copies, same bytes as [T, M, D].
                hidden_streams = hidden_states.repeat(1, self.hc_mult)
            else:
                lead_expand = (-1,) * (hidden_states.ndim - 1)
                hidden_streams = (
                    hidden_states.unsqueeze(-2)
                    .expand(*lead_expand, self.hc_mult, self.hidden_size)
                    .contiguous()
                )
        else:
            assert intermediate_tensors is not None
            hidden_streams = cast(torch.Tensor, intermediate_tensors["hidden_streams"])
            if flat:
                hidden_streams = hidden_streams.reshape(
                    -1, self.hc_mult * self.hidden_size
                )

        deferred: MhcDeferredPost | None = None
        for layer in layers:
            hidden_streams, deferred = layer(
                positions=positions,
                hidden_streams=hidden_streams,
                deferred=deferred,
            )
        if deferred is not None:
            # Pallas mHC path: settle the post the last layer deferred.
            last_layer = cast(Glm5NextDecoderLayer, self.layers[self.end_layer - 1])
            residual = deferred.residual.view(-1, self.hc_mult, self.hidden_size)
            hidden_streams = last_layer.ffn_hc.mhc_ops.post(
                deferred.sublayer_out, residual, deferred.post, deferred.comb
            )
        elif flat:
            hidden_streams = hidden_streams.view(-1, self.hc_mult, self.hidden_size)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_streams": hidden_streams})

        # Collapse the mHC streams by their mean, then the final RMSNorm.
        assert isinstance(self.norm, RMSNorm)
        collapsed = hidden_streams.mean(dim=-2)
        out = glm53_mhc_tpu.rms_norm_weighted(
            collapsed, self.norm.weight, self.norm.variance_epsilon
        )
        return out


def load_glm53_checkpoint_tensor(
    module: nn.Module,
    name: str,
    ckpt_name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
    loaded_params: set[str],
    unexpected: list[str],
) -> None:
    """Dispatch one normalized GLM-5.3 checkpoint tensor into `module`."""
    # 1. Per-expert routed weights -> FusedMoE `w13_*` / `w2_*`.
    expert_match = EXPERT_WEIGHT_RE.match(name)
    if expert_match is not None:
        param_kind, shard_id = EXPERT_SHARDS[expert_match.group("proj")]
        param_name = (
            f"{expert_match.group('prefix')}routed_experts."
            f"{param_kind}_{expert_match.group('suffix')}"
        )
        if is_pp_missing_parameter(param_name, module):
            return
        param = params_dict.get(param_name)
        if param is None:
            unexpected.append(ckpt_name)
            return
        cast(Any, param).weight_loader(
            param,
            loaded_weight,
            param_name,
            shard_id=shard_id,
            expert_id=int(expert_match.group("expert")),
        )
        loaded_params.add(param_name)
        return

    # 2. Unfused checkpoint shards -> fused vLLM parameters.
    for param_part, weight_part, shard_id in STACKED_PARAMS_MAPPING:
        if f"{weight_part}." not in name:
            continue
        candidate = name.replace(f"{weight_part}.", f"{param_part}.")
        if is_pp_missing_parameter(candidate, module):
            return
        param = params_dict.get(candidate)
        if param is None:
            continue
        cast(Any, param).weight_loader(param, loaded_weight, shard_id)
        loaded_params.add(candidate)
        return

    # 3. Everything else maps 1:1.
    remapped = maybe_remap_kv_scale_name(name, params_dict)
    if remapped is None:
        return
    name = remapped
    if is_pp_missing_parameter(name, module):
        return
    param = params_dict.get(name)
    if param is None:
        unexpected.append(ckpt_name)
        return
    loader = getattr(param, "weight_loader", default_weight_loader)
    loader(param, loaded_weight)
    loaded_params.add(name)


class Glm5NextForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsPP):
    """GLM-5.3-Flash text model on TPU.

    Also serves the released checkpoint's vision-language architecture
    (`Glm5NextForConditionalGeneration`, see the subclass) as text only.
    """

    packed_modules_mapping = PACKED_MODULES_MAPPING

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = self.model_config.hf_text_config
        self.quant_config = vllm_config.quant_config

        self.model = Glm5NextTextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                int(self.config.vocab_size),
                int(self.config.hidden_size),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(int(self.config.vocab_size))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        hc_mult, hidden = int(self.config.hc_mult), int(self.config.hidden_size)
        return IntermediateTensors(
            {
                "hidden_streams": torch.zeros(
                    (batch_size, hc_mult, hidden), dtype=dtype, device=device
                )
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def process_weights_after_loading(self) -> None:
        """Post-load preparation that must happen before any traced forward.

        Folds every KDA layer's conv weight into the Kimi ops' layout (vLLM's
        per-module hook normally does this already; the fold is idempotent,
        and running it here keeps it independent of which hooks a vLLM
        version calls) and splits each mHC site's `fn` for the Pallas
        kernels.
        """
        for module in self.modules():
            if isinstance(module, glm53_kda_tpu.Glm53TpuKimiDeltaAttention):
                module.fold_conv_weight()
            elif isinstance(module, glm53_mhc_tpu.FusedGlm53HyperConnectionTpu):
                module.prepare_pallas_weights()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        # Must match the KDA layer's `get_state_dtype`: the TPU platform sizes
        # the state cache from this classmethod, and vLLM allocates from the
        # layer.
        return glm53_kda_tpu.glm53_kda_state_dtype()

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_heads = int(config.linear_num_heads)
        if num_heads % tp_size:
            raise ValueError(
                f"GLM-5.3-Flash KDA linear_num_heads={num_heads} is not "
                f"divisible by tensor_parallel_size={tp_size}."
            )
        # Must match the KDA layer's `get_state_shape`.
        return cast(
            Any,
            glm53_kda_tpu.glm53_kda_state_shape(
                int(config.linear_conv_kernel_dim),
                num_heads // tp_size,
                int(config.linear_head_dim),
            ),
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Loads `zai-org/GLM-5.3-Flash` weights.

        Handles the released checkpoint layout: `model.language_model.*`
        prefixes, per-expert block-FP8 routed experts
        (`mlp.experts.{e}.*_proj.weight` + `weight_scale_inv`), unfused KDA
        projections and short convolutions (packed into `in_proj_qkvgfab` /
        `conv1d`), and the mHC / sublayer norm tensors. The MTP layers, the
        vision tower and the sparse attention indexer are skipped.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        unexpected: list[str] = []
        num_layers = int(self.config.num_hidden_layers)

        for item in weights:
            ckpt_name, loaded_weight = item[:2]
            name = normalize_glm53_checkpoint_name(ckpt_name, num_layers)
            if name is None:
                continue
            load_glm53_checkpoint_tensor(
                self,
                name,
                ckpt_name,
                loaded_weight,
                params_dict,
                loaded_params,
                unexpected,
            )

        if unexpected:
            logger.warning(
                "GLM-5.3-Flash: %d checkpoint tensors had no matching parameter, "
                "e.g. %s",
                len(unexpected),
                unexpected[:20],
            )
        # The TPU block-FP8 linear method registers `weight_scale` as an alias
        # sharing storage with `weight_scale_inv`, so that load fills it.
        missing = sorted(
            n
            for n in params_dict
            if n not in loaded_params
            and not (n.endswith(".weight_scale") and f"{n}_inv" in loaded_params)
            and not is_pp_missing_parameter(n, self)
        )
        if missing:
            logger.warning(
                "GLM-5.3-Flash: %d parameters were not loaded from the "
                "checkpoint, e.g. %s",
                len(missing),
                missing[:40],
            )
        return loaded_params


class Glm5NextForConditionalGeneration(Glm5NextForCausalLM):
    """The released checkpoint's architecture, served as text only.

    The vision tower's weights are skipped at load.
    """
