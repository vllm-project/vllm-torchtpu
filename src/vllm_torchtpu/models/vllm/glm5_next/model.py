"""vLLM + TorchTPU model definition for GLM 5.3 Flash (`zai-org/GLM-5.3-Flash`).

Registers `Glm5NextForConditionalGeneration` and `Glm5NextForCausalLM` into
vLLM's `ModelRegistry`, wiring:
1. **34 KDA linear-attention layers** (`layer_idx % 4 != 3`) via
   `TpuKimiGatedDeltaNetAttention` (`@KimiGatedDeltaNetAttention.register_oot`
   in `kda_vllm_layer_tpu.py`), which dispatches:
   - Prefill (`num_prefills > 0`) -> `causal_conv1d_fn_paged` +
     `chunk_kda_with_fused_gate` -> `torch.ops.glm53_kda.prefill_chunk`
     (2D head-tiled Pallas TPU v7x kernel, `chunk_len=64`).
   - Decode (`num_decodes > 0`) -> `causal_conv1d_update_paged` +
     `fused_recurrent_kda_packed_decode` -> `torch.ops.glm53_kda.decode`
     (fused Pallas TPU v7x kernel with transposed `[num_slots, H, d_v, d_k]`
     recurrent state).
   - Output norm -> strict FP32 `FusedRMSNormGated`.
2. **90 Manifold-Constrained HyperConnection (`mHC`) sites** (`attn_hc` and
   `ffn_hc` on each of the 45 decoder layers) via
   `FusedGlm53HyperConnectionTpu` (`glm53_mhc_tpu.py`), keeping all 20
   Sinkhorn-Knopp iterations + 4-stream collapse + post-sublayer stream mixing
   on the TPU VPU in explicit FP32.
3. **11 MLA / DeepSeek Sparse Attention layers** (`layer_idx % 4 == 3`,
   `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=256`,
   `qk_rope_head_dim=0`, `v_head_dim=256`).
4. **3 Dense MLP layers (`0..2`, `intermediate_size=12288`) + 42 Sparse MoE
   layers (`3..44`, `288` routed experts, `top_k=8`, `1` shared expert,
   `moe_intermediate_size=2048`, `routed_scaling_factor=2.5`,
   `swiglu_limit=10.0`)**.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

# pylint: disable=g-import-not-at-top
import vllm.config as vllm_config_mod
import vllm.distributed as vllm_dist
from torch import nn
from transformers import PretrainedConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.layers.adapter.custom_ops import (
    glm53_kda_vllm_layer_tpu as kda_vllm_layer_tpu,
)
from vllm_torchtpu.layers.adapter.custom_ops import glm53_mhc_tpu

CacheConfig = vllm_config_mod.CacheConfig
ModelConfig = vllm_config_mod.ModelConfig
ParallelConfig = vllm_config_mod.ParallelConfig
VllmConfig = vllm_config_mod.VllmConfig
get_pp_group = vllm_dist.get_pp_group
get_tensor_model_parallel_rank = vllm_dist.get_tensor_model_parallel_rank
get_tensor_model_parallel_world_size = vllm_dist.get_tensor_model_parallel_world_size
tensor_model_parallel_all_reduce = vllm_dist.tensor_model_parallel_all_reduce

logger = init_logger(__name__)


def _uses_kimi_kda() -> bool:
    """Whether the KDA layers run on the Kimi ops (see `GLM53_KDA_BACKEND`)."""
    return kda_vllm_layer_tpu.glm53_kda_backend() == kda_vllm_layer_tpu.KDA_BACKEND_KIMI


# Checkpoint (unfused) -> vLLM fused module names. Also consumed by the FP8
# config (`is_layer_skipped`) to see that the fused KDA projection is BF16.
_PACKED_MODULES_MAPPING: dict[str, list[str]] = {
    "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"],
    "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

# (fused vLLM param component, checkpoint component, shard id). The shard ids
# follow `KimiGatedDeltaNetAttention.in_proj_qkvgfab` (q, k, v, beta, f_a) and
# its fused `conv1d` (q, k, v).
_STACKED_PARAMS_MAPPING: tuple[tuple[str, str, int], ...] = (
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

_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.+\.mlp\.experts\.)(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<suffix>weight(?:_scale_inv)?)$"
)
# checkpoint projection -> (FusedMoE param family, FusedMoE shard id)
_EXPERT_SHARDS: dict[str, tuple[str, str]] = {
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
    """Maps a GLM 5.3 checkpoint tensor name into this model's namespace.

    Returns None for tensors that text serving does not use: the vision tower,
    the MTP / next-n layer (`layers.{num_hidden_layers}`), and the DSA indexer.
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
    if ".self_attn.indexer." in name or "rotary_emb.inv_freq" in name:
        return None
    for src, dst in _CKPT_RENAMES:
        if name.endswith(src):
            return name[: -len(src)] + dst
    return name


_GLM5_TEXT_DEFAULTS: dict[str, Any] = {
    "num_hidden_layers": 45,
    "hidden_size": 4096,
    "hidden_act": "silu",
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "vocab_size": 154880,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 131072,
    "intermediate_size": 12288,
    "moe_intermediate_size": 2048,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "norm_topk_prob": True,
}


def _fill_glm5_text_defaults(cfg: Any) -> None:
    """Fills missing GLM 5.3 text attributes and materializes `head_dim`."""
    for key, value in _GLM5_TEXT_DEFAULTS.items():
        if getattr(cfg, key, None) is None:
            setattr(cfg, key, value)
    # The released checkpoint stores `head_dim=0` for its NoPE MLA layers. vLLM
    # sizes attention KV pages from `head_dim` (falling back to
    # `hidden_size // num_attention_heads = 64`), so set the real width (256).
    if not getattr(cfg, "head_dim", None):
        cfg.head_dim = int(cfg.qk_nope_head_dim) + int(cfg.qk_rope_head_dim or 0)


class Glm5NextTextConfig(PretrainedConfig):
    """Text sub-config of `zai-org/GLM-5.3-Flash` (`model_type=glm5_next_text`)."""

    model_type = "glm5_next_text"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        _fill_glm5_text_defaults(self)


class Glm5NextConfig(PretrainedConfig):
    """Top-level GLM 5.3 Flash config.

    The released `config.json` nests the language model under `text_config`
    (plus a `vision_config` that text serving ignores). Legacy flat configs, as
    used by the dummy-weight benchmarks, are still accepted.
    """

    model_type = "glm5_next"

    def __init__(
        self,
        text_config: Any = None,
        vision_config: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if isinstance(text_config, dict):
            text_config = Glm5NextTextConfig(**text_config)
        self.text_config = text_config
        self.vision_config = vision_config
        if text_config is None:
            _fill_glm5_text_defaults(self)
        _ensure_glm5_text_config_attrs(self)


def _ensure_glm5_text_config_attrs(raw_config: Any) -> Any:
    """Populates `linear_attn_config` and helper methods expected by vLLM KDA."""
    sub_cfg = getattr(raw_config, "text_config", None)
    config: Any = sub_cfg if sub_cfg is not None else raw_config
    # Prefer the checkpoint's own KDA geometry (`text_config.linear_attn_config`)
    # over the hard-coded fallbacks below.
    ckpt_kda = getattr(config, "linear_attn_config", None)
    ckpt_kda = dict(ckpt_kda) if isinstance(ckpt_kda, dict) else {}
    for attr, key in (
        ("linear_num_heads", "num_heads"),
        ("linear_head_dim", "head_dim"),
        ("linear_conv_kernel_dim", "short_conv_kernel_size"),
        ("linear_lower_bound", "gate_lower_bound"),
    ):
        if not hasattr(config, attr) and key in ckpt_kda:
            setattr(config, attr, ckpt_kda[key])
    if not hasattr(config, "rms_norm_eps"):
        config.rms_norm_eps = 1e-5
    if not hasattr(config, "linear_num_heads"):
        config.linear_num_heads = 64
    if not hasattr(config, "linear_head_dim"):
        config.linear_head_dim = 128
    if not hasattr(config, "linear_conv_kernel_dim"):
        config.linear_conv_kernel_dim = 4
    if not hasattr(config, "linear_lower_bound"):
        config.linear_lower_bound = -5.0
    if not hasattr(config, "hc_mult"):
        config.hc_mult = 4
    if not hasattr(config, "hc_sinkhorn_iters"):
        config.hc_sinkhorn_iters = 20
    if not hasattr(config, "hc_eps"):
        config.hc_eps = 1e-6
    if not hasattr(config, "swiglu_limit"):
        config.swiglu_limit = 10.0
    if not hasattr(config, "routed_scaling_factor"):
        config.routed_scaling_factor = 2.5
    if not hasattr(config, "n_routed_experts"):
        config.n_routed_experts = getattr(config, "num_experts", 288)
    if not hasattr(config, "n_shared_experts"):
        config.n_shared_experts = getattr(config, "num_shared_experts", 1)
    if not hasattr(config, "num_experts_per_tok"):
        config.num_experts_per_tok = getattr(config, "num_experts_per_token", 8)
    if not hasattr(config, "layer_types") or config.layer_types is None:
        config.layer_types = [
            "linear_attention" if i % 4 != 3 else "deepseek_sparse_attention"
            for i in range(config.num_hidden_layers)
        ]
    if not hasattr(config, "mlp_layer_types") or config.mlp_layer_types is None:
        config.mlp_layer_types = ["dense"] * min(3, config.num_hidden_layers) + [
            "sparse"
        ] * max(0, config.num_hidden_layers - 3)
    ckpt_kda.update(
        {
            "num_heads": int(config.linear_num_heads),
            "head_dim": int(config.linear_head_dim),
            "short_conv_kernel_size": int(config.linear_conv_kernel_dim),
            "gate_lower_bound": config.linear_lower_bound,
            "safe_gate": True,
        }
    )
    config.linear_attn_config = ckpt_kda
    return config


class Glm5NextMLP(nn.Module):
    """Clamped SwiGLU MLP (`swiglu_limit=10.0`) used in dense & shared-expert blocks."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        activated = F.silu(gate) * up
        out, _ = self.down_proj(activated)
        return out


# The TPU GMM kernel's clamped SwiGLU (`silu_and_mul_with_clamp`) hard-codes
# this limit; it equals the released GLM 5.3 Flash `swiglu_limit`.
_TPU_CLAMPED_SWIGLU_LIMIT = 10.0


def _enable_tpu_swiglu_clamp(experts: nn.Module, swiglu_limit: float) -> None:
    """Makes the TPU fused-MoE kernel apply GLM's clamped SwiGLU.

    The HF reference computes `silu(gate.clamp(max=L)) * up.clamp(-L, L)` in
    every expert. vllm-torchtpu's FP8 / unquantized MoE methods ignore
    `swiglu_limit` and resolve `silu` to the plain kernel activation, so remap it
    to the clamped kernel variant exactly as the MXFP4 method does. The string is
    resolved in `process_weights_after_loading`, after this hook is installed.
    """
    routed = getattr(experts, "routed_experts", experts)
    quant_method: Any = getattr(routed, "quant_method", None)
    resolve = getattr(quant_method, "_resolve_tpu_activation", None)
    if resolve is None:
        return  # Not a TPU MoE method (e.g. CPU-only unit tests).
    if abs(float(swiglu_limit) - _TPU_CLAMPED_SWIGLU_LIMIT) > 1e-6:
        raise NotImplementedError(
            f"GLM 5.3 swiglu_limit={swiglu_limit} is not supported by the TPU "
            f"clamped SwiGLU kernel (limit={_TPU_CLAMPED_SWIGLU_LIMIT})."
        )

    def _resolve_clamped(layer: Any) -> str:
        activation = resolve(layer)
        if activation.lower() in ("silu", "silu_and_mul", "swiglu"):
            return "silu_and_mul_with_clamp"
        return activation

    quant_method._resolve_tpu_activation = _resolve_clamped


class Glm5NextMoE(nn.Module):
    """GLM 5.3 Flash MoE block (`288` routed experts, `top_k=8`, `1` shared expert).

    Routing matches the HF reference (`noaux_tc`): `sigmoid(logits) + bias` picks
    the top-k experts, the un-biased sigmoid scores of those experts are
    renormalized and multiplied by `routed_scaling_factor`. The shared expert is
    added unscaled. Routed experts run through vLLM's `FusedMoE` so the block-FP8
    checkpoint experts are loaded and served by the TPU GMM kernel.
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
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 10.0))
        self.tp_size = get_tensor_model_parallel_world_size()

        # Registration order matters: `gate` and `shared_experts` are registered
        # before `experts` so `named_parameters()` reports them under the
        # checkpoint names (`mlp.gate.*`, `mlp.shared_experts.*`).
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

        n_shared = int(getattr(config, "n_shared_experts", 1) or 0)
        if n_shared > 0:
            self.shared_experts: Glm5NextMLP | None = Glm5NextMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.moe_intermediate_size * n_shared,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=self.hidden_size,
            intermediate_size=self.moe_intermediate_size,
            renormalize=bool(getattr(config, "norm_topk_prob", True)),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=str(getattr(config, "scoring_func", "sigmoid")),
            e_score_correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
        )
        _enable_tpu_swiglu_clamp(self.experts, self.swiglu_limit)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        x_flat = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(x_flat)
        # Routed + shared expert outputs, reduced across TP ranks by the runner.
        out = self.experts(hidden_states=x_flat, router_logits=router_logits)
        return out.view(orig_shape)


class Glm5NextMLAAttention(nn.Module):
    """NoPE Multi-Head Latent Attention (`q_lora_rank=1536`, `kv_lora_rank=512`).

    With `qk_rope_head_dim=0` the decompressed per-head keys / values are plain
    `256`-wide MHA tensors, so they go through vLLM's paged-KV `Attention`
    (the TPU ragged paged attention backend) with `scale = 256 ** -0.5`.

    The DSA indexer is not implemented: attention is dense-causal, which equals
    the reference's top-k sparse attention while the context is within
    `index_topk` tokens.
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
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_local_heads = self.num_heads // self.tp_size
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(getattr(config, "qk_rope_head_dim", 0))
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
        output: torch.Tensor,
    ) -> None:
        del positions  # NoPE: no rotary embedding on the MLA layers.
        num_tokens = hidden_states.shape[0]
        q_lora, _ = self.q_a_proj(hidden_states)
        q_lora = self.q_a_layernorm(q_lora)
        q, _ = self.q_b_proj(q_lora)  # [T, H_local * qk_head_dim]

        kv_a, _ = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent = self.kv_a_layernorm(kv_a[..., : self.kv_lora_rank])
        kv, _ = self.kv_b_proj(kv_latent)
        kv = kv.view(
            num_tokens,
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        attn_out = self.attn(
            q,
            k.reshape(num_tokens, self.num_local_heads * self.qk_head_dim),
            v.reshape(num_tokens, self.num_local_heads * self.v_head_dim),
        )
        proj_out, _ = self.o_proj(attn_out)
        output.copy_(proj_out)


class Glm5NextDecoderLayer(nn.Module):
    """Single GLM 5.3 Flash decoder layer with mHC + (KDA | MLA) + (MLP | MoE)."""

    def __init__(
        self,
        config: Any,
        vllm_config: VllmConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
        self.block_type = config.layer_types[layer_idx]
        self.mlp_type = config.mlp_layer_types[layer_idx]

        # Manifold-Constrained HyperConnections (Fused TPU VPU FP32 implementation)
        self.attn_hc = glm53_mhc_tpu.FusedGlm53HyperConnectionTpu(
            hidden_size=self.hidden_size,
            hc_mult=int(config.hc_mult),
            hc_sinkhorn_iters=int(config.hc_sinkhorn_iters),
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(config.rms_norm_eps),
        )
        self.ffn_hc = glm53_mhc_tpu.FusedGlm53HyperConnectionTpu(
            hidden_size=self.hidden_size,
            hc_mult=int(config.hc_mult),
            hc_sinkhorn_iters=int(config.hc_sinkhorn_iters),
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(config.rms_norm_eps),
        )

        if self.block_type == "linear_attention":
            # Instantiates TpuKimiGatedDeltaNetAttention via PluggableLayer OOT registry
            self.self_attn: nn.Module = KimiGatedDeltaNetAttention(
                config=config,
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

        if self.mlp_type == "sparse":
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
    ) -> torch.Tensor:
        # 1. Attention mHC pre-collapse + sublayer RMSNorm
        post_attn, comb_attn, norm_attn_in = self.attn_hc.collapse_and_norm(
            hidden_streams
        )
        attn_out = torch.empty_like(norm_attn_in)
        self.self_attn(
            hidden_states=norm_attn_in,
            positions=positions,
            output=attn_out,
        )
        hidden_streams = self.attn_hc.combine(
            hidden_streams, attn_out, post_attn, comb_attn
        )

        # 2. FFN/MoE mHC pre-collapse + sublayer RMSNorm
        post_ffn, comb_ffn, norm_ffn_in = self.ffn_hc.collapse_and_norm(hidden_streams)
        ffn_out = self.mlp(norm_ffn_in)
        hidden_streams = self.ffn_hc.combine(
            hidden_streams, ffn_out, post_ffn, comb_ffn
        )
        return hidden_streams


class Glm5NextTextModelVllm(nn.Module):
    """45-layer GLM 5.3 Flash backbone maintaining 4-stream mHC residual state."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        raw_config = vllm_config.model_config.hf_config
        self.config = _ensure_glm5_text_config_attrs(raw_config)
        self.hidden_size = int(self.config.hidden_size)
        self.vocab_size = int(self.config.vocab_size)
        self.hc_mult = int(self.config.hc_mult)

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                self.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        def _build_layer(prefix: str) -> Glm5NextDecoderLayer:
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return Glm5NextDecoderLayer(
                config=self.config,
                vllm_config=vllm_config,
                layer_idx=layer_idx,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            int(self.config.num_hidden_layers),
            _build_layer,
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(self.hidden_size, eps=float(self.config.rms_norm_eps))
        else:
            self.norm = PPMissingLayer()

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
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(input_ids)
            # Expand [T, D] -> [T, hc_mult, D]
            hidden_streams = (
                hidden_states.unsqueeze(-2)
                .expand(*hidden_states.shape[:-1], self.hc_mult, self.hidden_size)
                .contiguous()
            )
        else:
            assert intermediate_tensors is not None
            hidden_streams = intermediate_tensors["hidden_streams"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_streams = layer(positions=positions, hidden_streams=hidden_streams)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_streams": hidden_streams})

        # Collapse 4 mHC streams via mean(dim=-2) and apply final RMSNorm
        collapsed = hidden_streams.mean(dim=-2)
        return self.norm(collapsed)


class Glm5NextForConditionalGeneration(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
    """vLLM entry point for `zai-org/GLM-5.3-Flash`.

    Architecture name: `Glm5NextForConditionalGeneration`.
    """

    packed_modules_mapping = _PACKED_MODULES_MAPPING

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = _ensure_glm5_text_config_attrs(self.model_config.hf_config)
        self.quant_config = vllm_config.quant_config

        self.model = Glm5NextTextModelVllm(
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
        """Folds every KDA layer's conv weight into the Kimi ops' layout.

        vLLM's per-module hook normally does this already (the KDA layer is an
        attention-like layer); the fold is idempotent, and running it here as
        well keeps it independent of which hooks a vLLM version calls.
        """
        for module in self.modules():
            if isinstance(module, kda_vllm_layer_tpu.TpuKimiGatedDeltaNetAttention):
                module.fold_conv_weight()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> Any:
        # Must match `TpuKimiGatedDeltaNetAttention.get_state_dtype`: the TPU
        # platform sizes the unified pool from this classmethod, and vLLM
        # allocates from the layer.
        if _uses_kimi_kda():
            return kda_vllm_layer_tpu.kimi_kda_state_dtype()
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> Any:
        cfg = _ensure_glm5_text_config_attrs(vllm_config.model_config.hf_config)
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        # Must match `TpuKimiGatedDeltaNetAttention.get_state_shape`.
        if _uses_kimi_kda():
            num_heads = int(cfg.linear_num_heads)
            if num_heads % tp_size:
                raise ValueError(
                    f"GLM-5.3 KDA num_heads={num_heads} is not divisible by "
                    f"tensor_parallel_size={tp_size}."
                )
            return kda_vllm_layer_tpu.kimi_kda_state_shape(
                int(cfg.linear_conv_kernel_dim),
                num_heads // tp_size,
                int(cfg.linear_head_dim),
            )
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            int(cfg.linear_num_heads),
            int(cfg.linear_head_dim),
            conv_kernel_size=int(cfg.linear_conv_kernel_dim),
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> Any:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Loads `zai-org/GLM-5.3-Flash` HF weights into the vLLM + TorchTPU model.

        Handles the released checkpoint layout: `model.language_model.*` prefixes,
        per-expert block-FP8 routed experts (`mlp.experts.{e}.*_proj.weight` +
        `weight_scale_inv`), unfused KDA projections / short convolutions (packed
        into `in_proj_qkvgfab` / `conv1d`), and the mHC / sublayer norm tensors.
        The MTP layer (`layers.45`) and the DSA indexer tensors are skipped.
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

            # 1. Per-expert routed weights -> FusedMoE `w13_*` / `w2_*`.
            expert_match = _EXPERT_WEIGHT_RE.match(name)
            if expert_match is not None:
                param_kind, shard_id = _EXPERT_SHARDS[expert_match.group("proj")]
                param_name = (
                    f"{expert_match.group('prefix')}routed_experts."
                    f"{param_kind}_{expert_match.group('suffix')}"
                )
                if is_pp_missing_parameter(param_name, self):
                    continue
                param = params_dict.get(param_name)
                if param is None:
                    unexpected.append(ckpt_name)
                    continue
                param.weight_loader(
                    param,
                    loaded_weight,
                    param_name,
                    shard_id=shard_id,
                    expert_id=int(expert_match.group("expert")),
                )
                loaded_params.add(param_name)
                continue

            # 2. Unfused checkpoint shards -> fused vLLM parameters.
            matched_stacked = False
            for param_part, weight_part, shard_id in _STACKED_PARAMS_MAPPING:
                if f"{weight_part}." not in name:
                    continue
                candidate = name.replace(f"{weight_part}.", f"{param_part}.")
                if is_pp_missing_parameter(candidate, self):
                    matched_stacked = True
                    break
                param = params_dict.get(candidate)
                if param is None:
                    continue
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(candidate)
                matched_stacked = True
                break
            if matched_stacked:
                continue

            # 3. Everything else maps 1:1.
            remapped = maybe_remap_kv_scale_name(name, params_dict)
            if remapped is None:
                continue
            name = remapped
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict.get(name)
            if param is None:
                unexpected.append(ckpt_name)
                continue
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)
            loaded_params.add(name)

        if unexpected:
            logger.warning(
                "GLM-5.3: %d checkpoint tensors had no matching parameter, e.g. %s",
                len(unexpected),
                unexpected[:20],
            )
        # The TPU block-FP8 linear method registers `weight_scale` as an alias
        # sharing storage with `weight_scale_inv`, so it is filled by that load.
        missing = sorted(
            n
            for n in params_dict
            if n not in loaded_params
            and not (n.endswith(".weight_scale") and f"{n}_inv" in loaded_params)
            and not is_pp_missing_parameter(n, self)
        )
        if missing:
            logger.warning(
                "GLM-5.3: %d parameters were NOT loaded from the checkpoint, e.g. %s",
                len(missing),
                missing[:40],
            )
        else:
            logger.info(
                "GLM-5.3: loaded all %d parameters from the checkpoint.",
                len(loaded_params),
            )
        return loaded_params


class Glm5NextForCausalLM(Glm5NextForConditionalGeneration):
    """Alias for text-only CausalLM entrypoints."""


def register_glm53_vllm_models() -> None:
    """Idempotently registers `Glm5NextForConditionalGeneration` in vLLM."""
    ModelRegistry.register_model(
        "Glm5NextForConditionalGeneration", Glm5NextForConditionalGeneration
    )
    ModelRegistry.register_model("Glm5NextForCausalLM", Glm5NextForCausalLM)


register_glm53_vllm_models()
